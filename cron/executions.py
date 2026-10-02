"""Profile-local durable audit ledger for cron execution attempts.

The ledger records what is known about each attempt; it is not a retry queue. Interrupted attempts
become ``unknown`` only after their owner process is proved gone — a start-time reading that fails
to match the claim-time fingerprint is not proof of death. Terminal states are immutable.

Each attempt carries an explicit lease (``lease_until``, derived from the shared
``FIRE_CLAIM_TTL_SECONDS``) and an append-only ``execution_events`` trail, so an attempt can never
be reported as running past a lease that has lapsed and its history can be inspected without
reconstructing it from mutable job-store fields.
"""

from __future__ import annotations

import math
import os
import sqlite3
import threading
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from hermes_constants import get_hermes_home
from hermes_time import now as _hermes_now
from cron.constants import CLAIM_TTL_INACTIVITY_HEADROOM, FIRE_CLAIM_TTL_SECONDS
from hermes_cli.observability.shared_metrics_gateway import record_cron_finish

# Optional test override. Production resolves the path at transaction time so dashboard operations
# that temporarily enter another profile cannot leak that profile's records into the import-time
# home.
EXECUTIONS_FILE: Optional[Path] = None
MAX_TERMINAL_EXECUTIONS = 1000
HANDOFF_ADOPTION_GRACE_SECONDS = 30.0
# Floor for the live-owner stale-claim bound (#115692); see _live_owner_stale_after_seconds.
LIVE_OWNER_STALE_CLAIM_FLOOR_SECONDS = 7200.0
_TERMINAL_STATES = ("completed", "failed", "unknown")
# Frozen outcome vocabulary: a run is ``completed`` or ``failed`` only when the ledger can prove
# it; anything the ledger cannot prove (interrupted, no finish record, expired claim) is
# ``unknown`` — never rounded up to a success.
EXECUTION_OUTCOMES = ("completed", "failed", "unknown")
# Append-only per-run event kinds (OpenDots run/event split).
EVENT_CLAIMED = "claimed"
EVENT_STARTED = "started"
EVENT_FINISHED = "finished"
EVENT_FAILED = "failed"
EVENT_RECLAIMED = "reclaimed"
EVENT_DELIVERED = "delivered"
EVENT_NOT_DELIVERED = "not_delivered"
EXECUTION_EVENT_KINDS = (
    EVENT_CLAIMED, EVENT_STARTED, EVENT_FINISHED, EVENT_FAILED,
    EVENT_RECLAIMED, EVENT_DELIVERED, EVENT_NOT_DELIVERED,
)
# Delivery state: the real destination, or ``none`` when no destination was resolved.
DELIVERY_NONE = "none"
_DELIVERED_OUTCOMES = ("delivered",)
_lock = threading.RLock()
_PROCESS_ID = uuid.uuid4().hex


# --- executions ledger --------------------------------------------------------------------------

def _connect() -> sqlite3.Connection:
    # Late imports: a scheduler daemon that outlives an on-disk upgrade already has the OLD
    # ``hermes_cli.sqlite_util`` / ``cron.jobs`` cached, so new names must be resolved at call time,
    # not at import time (the guarantee cron/ledger.py used to carry, see e24c8499).
    from cron.jobs import _ensure_cron_dir
    from hermes_cli.sqlite_util import open_db

    path = EXECUTIONS_FILE or (get_hermes_home().resolve() / "cron" / "executions.db")
    _ensure_cron_dir(path.parent)
    return open_db(path, db_label="cron/executions.db", synchronous_full=True, initialize=_initialize_schema)


def _initialize_schema(conn: sqlite3.Connection) -> None:
    from hermes_cli.sqlite_util import add_column_if_missing

    conn.execute(
        """CREATE TABLE IF NOT EXISTS executions (
             id TEXT PRIMARY KEY,
             job_id TEXT NOT NULL,
             source TEXT NOT NULL,
             process_id TEXT NOT NULL,
             pid INTEGER NOT NULL,
             process_started_at INTEGER,
             status TEXT NOT NULL CHECK(status IN
               ('claimed','running','completed','failed','unknown')),
             handoff_pending INTEGER NOT NULL DEFAULT 0,
             handoff_started_at REAL,
             claimed_at TEXT NOT NULL,
             started_at TEXT,
             finished_at TEXT,
             error TEXT
           )"""
    )
    add_column_if_missing(
        conn, "executions", "handoff_pending",
        "handoff_pending INTEGER NOT NULL DEFAULT 0",
    )
    add_column_if_missing(
        conn, "executions", "handoff_started_at", "handoff_started_at REAL"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_executions_job_claimed "
        "ON executions(job_id, claimed_at DESC, id DESC)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_executions_status_claimed "
        "ON executions(status, claimed_at DESC, id DESC)"
    )
    add_column_if_missing(conn, "executions", "delivery_outcome", "delivery_outcome TEXT")
    add_column_if_missing(conn, "executions", "scheduled_instant", "scheduled_instant TEXT")
    # Explicit lease deadline (ISO instant), derived from FIRE_CLAIM_TTL_SECONDS. Legacy rows
    # without one keep the age-based reading; see execution_lease_expired().
    add_column_if_missing(conn, "executions", "lease_until", "lease_until TEXT")
    # The real delivery destination resolved for the run ("origin", "local", "telegram:123"...),
    # or the literal "none" when no destination exists.
    add_column_if_missing(conn, "executions", "delivery_target", "delivery_target TEXT")
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_executions_occurrence "
        "ON executions(job_id, scheduled_instant) WHERE status='completed'"
    )
    # Append-only per-run trail. Rows are only INSERTed (and pruned with their execution); the
    # ledger never rewrites history, so a run's lifecycle is inspectable after the fact.
    conn.execute(
        """CREATE TABLE IF NOT EXISTS execution_events (
             id INTEGER PRIMARY KEY AUTOINCREMENT,
             execution_id TEXT NOT NULL,
             job_id TEXT NOT NULL,
             kind TEXT NOT NULL CHECK(kind IN
               ('claimed','started','finished','failed','reclaimed','delivered','not_delivered')),
             detail TEXT,
             target TEXT,
             created_at TEXT NOT NULL
           )"""
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_execution_events_execution "
        "ON execution_events(execution_id, id)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_execution_events_job "
        "ON execution_events(job_id, id)"
    )


def _append_event(
    conn: sqlite3.Connection, *, execution_id: str, job_id: str, kind: str,
    detail: Optional[str] = None, target: Optional[str] = None,
) -> None:
    """Append one lifecycle event for an attempt. Append-only: never updated or deleted here."""
    conn.execute(
        """INSERT INTO execution_events
           (execution_id, job_id, kind, detail, target, created_at)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (str(execution_id), str(job_id), str(kind), detail, target, _hermes_now().isoformat()),
    )


def normalize_delivery_outcome(value: Any) -> str:
    """The recorded delivery state: the caller's real outcome, or ``none`` when there is none.

    ``none`` is a statement about the ledger (no destination/outcome was recorded), not a
    fabricated success — ``delivered`` is only ever written by a caller that observed a send.
    """
    text = str(value).strip() if value not in (None, "") else ""
    return text or DELIVERY_NONE


def _lease_deadline(now: datetime) -> str:
    """The explicit lease deadline for a claim taken at ``now``, from FIRE_CLAIM_TTL_SECONDS."""
    return (now + timedelta(seconds=FIRE_CLAIM_TTL_SECONDS)).isoformat()


@contextmanager
def _transaction() -> Iterator[sqlite3.Connection]:
    from hermes_cli.sqlite_util import transaction

    with _lock, transaction(_connect()) as conn:
        yield conn


def _fetch(conn: sqlite3.Connection, execution_id: str) -> Optional[Dict[str, Any]]:
    row = conn.execute("SELECT * FROM executions WHERE id=?", (execution_id,)).fetchone()
    return dict(row) if row is not None else None


def _emit_execution_state(
    record: Optional[Dict[str, Any]], *, delivery_outcome: Optional[str] = None
) -> None:
    """Project durable state to monitoring without affecting ledger behavior."""
    try:
        from agent.monitoring.cron_health import emit_execution_state

        emit_execution_state(record, delivery_outcome=delivery_outcome)
    except Exception:
        pass


def _process_start_time(pid: int) -> Optional[int]:
    try:
        from gateway.status import get_process_start_time
        return get_process_start_time(pid)
    except Exception:
        return None


def _owner_is_live(pid: int, started_at: Optional[int]) -> bool:
    try:
        from gateway.status import _pid_exists
        if not _pid_exists(pid):
            return False
    except Exception:
        return True  # fail safe: inability to prove death must not rewrite state
    if started_at is None:
        return pid == os.getpid()
    current = _process_start_time(pid)
    if current is None:
        return True  # cannot compare -> cannot prove death; a misread must not rewrite state
    # Drifted same-host readings (#117505) are not proof of death; a live misread is still
    # bounded by the stale-claim sweep below.
    from gateway.status import start_time_fingerprints_match
    return start_time_fingerprints_match(started_at, current)


def _live_owner_stale_after_seconds() -> Optional[float]:
    """Age past which a claimed/running row with a LIVE owner is treated as wedged.

    Derived from the existing knobs, never a bare wall-clock constant:
    ``max(3 × HERMES_CRON_TIMEOUT, cron script timeout, 7200)``. Returns ``None`` (never reclaim
    live owners — today's behaviour) when the inactivity timeout is 0/unlimited or not a finite
    positive number: with no bound to derive from, fail closed.
    """
    from cron.scheduler import _cron_inactivity_seconds
    from cron.scheduler_script import _get_script_timeout

    inactivity = float(_cron_inactivity_seconds())
    if not math.isfinite(inactivity) or inactivity <= 0:
        return None
    return max(
        inactivity * CLAIM_TTL_INACTIVITY_HEADROOM,
        float(_get_script_timeout()),
        LIVE_OWNER_STALE_CLAIM_FLOOR_SECONDS,
    )


def _claim_age_seconds(claimed_at: str) -> float:
    """Seconds since ``claimed_at`` (NOT NULL, always the aware ISO string from hermes_time.now)."""
    return (_hermes_now() - datetime.fromisoformat(claimed_at)).total_seconds()


def execution_lease_expired(
    record: Dict[str, Any], *, now: Optional[datetime] = None
) -> bool:
    """True when this attempt's explicit lease has lapsed.

    The lease is the one the attempt was claimed under (``lease_until``, derived from the shared
    ``FIRE_CLAIM_TTL_SECONDS`` and refreshed while the run is genuinely alive). A legacy row
    without an explicit deadline falls back to the same TTL over ``claimed_at`` — never a second
    TTL. A malformed/absent ``claimed_at`` reads as expired: the ledger cannot prove the lease is
    still held, so it must not be reported as running.
    """
    moment = now or _hermes_now()
    raw = record.get("lease_until")
    if raw:
        try:
            return moment >= datetime.fromisoformat(str(raw))
        except ValueError:
            return True
    claimed_at = record.get("claimed_at")
    if not claimed_at:
        return True
    try:
        return _claim_age_seconds(str(claimed_at)) >= FIRE_CLAIM_TTL_SECONDS
    except ValueError:
        return True


def _prune_unlocked(conn: sqlite3.Connection) -> None:
    conn.execute(
        """DELETE FROM executions WHERE id IN (
             SELECT id FROM executions
             WHERE status IN ('completed','failed','unknown')
             ORDER BY julianday(finished_at) DESC, finished_at DESC,
                      julianday(claimed_at) DESC, claimed_at DESC, id DESC LIMIT -1 OFFSET ?
           )""",
        (max(0, int(MAX_TERMINAL_EXECUTIONS)),),
    )
    # Events live and die with their execution: pruning an attempt prunes its trail, so the
    # append-only log stays bounded without rewriting any live run's history.
    conn.execute(
        "DELETE FROM execution_events WHERE execution_id NOT IN (SELECT id FROM executions)"
    )


def create_execution(
    job_id: str, *, source: str, scheduled_instant: Optional[str] = None,
) -> Dict[str, Any]:
    """Persist a claimed attempt before executor/provider dispatch."""
    from cron.occurrences import scheduled_instant as canonical_instant

    now = _hermes_now()
    execution_id = uuid.uuid4().hex
    pid = os.getpid()
    with _transaction() as conn:
        conn.execute(
            """INSERT INTO executions
               (id, job_id, source, process_id, pid, process_started_at,
                status, claimed_at, scheduled_instant, lease_until)
               VALUES (?, ?, ?, ?, ?, ?, 'claimed', ?, ?, ?)""",
            (execution_id, str(job_id), str(source), _PROCESS_ID, pid,
             _process_start_time(pid), now.isoformat(),
             canonical_instant(scheduled_instant), _lease_deadline(now)),
        )
        _append_event(
            conn, execution_id=execution_id, job_id=job_id,
            kind=EVENT_CLAIMED, detail=str(source),
        )
        record = _fetch(conn, execution_id)
    _emit_execution_state(record)
    return record  # type: ignore[return-value]


def set_execution_occurrence(execution_id: str, instant: Optional[str]) -> None:
    """Bind the store-claimed snapshot before a provider hands it to a worker."""
    from cron.occurrences import scheduled_instant

    with _transaction() as conn:
        cur = conn.execute(
            "UPDATE executions SET scheduled_instant=? WHERE id=? AND status='claimed' "
            "AND handoff_pending=0 AND process_id=? AND pid=?",
            (scheduled_instant(instant), execution_id, _PROCESS_ID, os.getpid()),
        )
        if cur.rowcount != 1:
            raise RuntimeError("Cron occurrence could not be bound before dispatch")


def mark_execution_handoff_pending(execution_id: str) -> Optional[Dict[str, Any]]:
    """Fence restart recovery while an external worker is adopting a claim."""
    with _transaction() as conn:
        cur = conn.execute(
            """UPDATE executions
               SET handoff_pending=1, handoff_started_at=?
               WHERE id=? AND status='claimed'
                 AND process_id=? AND pid=?""",
            (time.time(), execution_id, _PROCESS_ID, os.getpid()),
        )
        if cur.rowcount != 1:
            return None
        record = _fetch(conn, execution_id)
    _emit_execution_state(record)
    return record


def adopt_claimed_execution(execution_id: str) -> Optional[Dict[str, Any]]:
    """Atomically transfer and start an attempt in its worker process.

    The dispatching gateway creates the row before spawning a restart-safe
    worker.  Adoption is the single ``claimed`` → ``running`` gate: only the
    winner may acknowledge ownership or run side effects.
    """
    pid = os.getpid()
    process_started_at = _process_start_time(pid)
    now = _hermes_now()
    with _transaction() as conn:
        cur = conn.execute(
            """UPDATE executions
               SET process_id=?, pid=?, process_started_at=?,
                   status='running', started_at=?, handoff_pending=0,
                   handoff_started_at=NULL, lease_until=?
               WHERE id=? AND status='claimed' AND handoff_pending=1""",
            (_PROCESS_ID, pid, process_started_at, now.isoformat(),
             _lease_deadline(now), execution_id),
        )
        if cur.rowcount != 1:
            return None
        row = _fetch(conn, execution_id)
        if row is not None:
            _append_event(
                conn, execution_id=execution_id, job_id=str(row["job_id"]),
                kind=EVENT_STARTED,
            )
        record = _fetch(conn, execution_id)
    _emit_execution_state(record)
    return record


def mark_execution_running(execution_id: str) -> Optional[Dict[str, Any]]:
    """Transition one claimed attempt to running exactly once, refreshing its explicit lease."""
    now = _hermes_now()
    with _transaction() as conn:
        cur = conn.execute(
            """UPDATE executions
               SET status='running', started_at=?, handoff_pending=0,
                   handoff_started_at=NULL, lease_until=?
               WHERE id=? AND status='claimed' AND handoff_pending=0
                 AND process_id=? AND pid=?""",
            (now.isoformat(), _lease_deadline(now), execution_id, _PROCESS_ID, os.getpid()),
        )
        if cur.rowcount != 1:
            return None
        row = _fetch(conn, execution_id)
        if row is not None:
            _append_event(
                conn, execution_id=execution_id, job_id=str(row["job_id"]),
                kind=EVENT_STARTED,
            )
        record = _fetch(conn, execution_id)
    _emit_execution_state(record)
    return record


def finish_execution(
    execution_id: str, *, success: bool, error: Optional[str] = None,
    delivery_outcome: Optional[str] = None, delivery_target: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Write a terminal result once; terminal attempts cannot be rewritten.

    ``delivery_outcome`` is normalised to the real state or ``none``; the delivery destination is
    the value already bound via ``set_delivery_target`` (or ``delivery_target`` here), and reads
    back as ``none`` when no destination exists. Both are recorded as an append-only
    ``delivered``/``not_delivered`` event.
    """
    now = _hermes_now().isoformat()
    status = "completed" if success else "failed"
    detail = None if success else (str(error) if error else "unknown failure")
    outcome = normalize_delivery_outcome(delivery_outcome)
    target = str(delivery_target).strip() if delivery_target not in (None, "") else None
    with _transaction() as conn:
        cur = conn.execute(
            """UPDATE executions
               SET status=?, finished_at=?, error=?, handoff_pending=0,
                   handoff_started_at=NULL, delivery_outcome=?,
                   delivery_target=COALESCE(?, delivery_target, ?)
               WHERE id=? AND status IN ('claimed','running')
                 AND process_id=? AND pid=?""",
            (status, now, detail, outcome, target, DELIVERY_NONE,
             execution_id, _PROCESS_ID, os.getpid()),
        )
        if cur.rowcount != 1:
            return None
        row = _fetch(conn, execution_id)
        if row is not None:
            _append_event(
                conn, execution_id=execution_id, job_id=str(row["job_id"]),
                kind=EVENT_FINISHED if success else EVENT_FAILED,
                detail=detail,
            )
            _append_event(
                conn, execution_id=execution_id, job_id=str(row["job_id"]),
                kind=(EVENT_DELIVERED if outcome in _DELIVERED_OUTCOMES
                      else EVENT_NOT_DELIVERED),
                detail=outcome, target=row.get("delivery_target") or DELIVERY_NONE,
            )
        _prune_unlocked(conn)
        record = _fetch(conn, execution_id)
    # Monitoring keeps its existing semantics: the caller's raw outcome (None = "no delivery was
    # attempted") decides skipped vs failed. The durable column records it as "none" instead.
    _emit_execution_state(record, delivery_outcome=delivery_outcome)
    record_cron_finish(record, delivery_outcome)
    return record


def set_delivery_target(execution_id: str, target: Optional[str]) -> bool:
    """Bind a live attempt's real delivery destination before it is finished.

    The destination is the lane the result actually went to (the originating conversation/job
    lane). False when the attempt is terminal or not owned by this process. Recorded, never
    inferred: a later ``finish_execution`` writes ``none`` only when nothing was ever bound.
    """
    text = str(target).strip() if target not in (None, "") else None
    with _transaction() as conn:
        cur = conn.execute(
            "UPDATE executions SET delivery_target=COALESCE(?, delivery_target) "
            "WHERE id=? AND status IN ('claimed','running') AND process_id=? AND pid=?",
            (text, str(execution_id), _PROCESS_ID, os.getpid()),
        )
    return cur.rowcount == 1


_OWNER_GONE_REASON = (
    "Scheduler restarted after this execution's owner exited before a durable "
    "terminal state; whether side effects ran is unknown."
)
_OWNER_WEDGED_REASON = (
    "Owner process is still alive but the claim outlived the derived stale bound; "
    "treated as wedged (#115692). The process was not terminated; whether side effects "
    "ran is unknown."
)
_LEASE_EXPIRED_REASON = (
    "Execution lease expired before a durable terminal state was recorded (no heartbeat "
    "within FIRE_CLAIM_TTL_SECONDS); the attempt may have been redelivered, so whether "
    "side effects ran is unknown."
)


def recover_interrupted_executions() -> int:
    """Mark abandoned attempts unknown without scheduling retries: rows whose owner is provably
    dead, plus rows whose live owner holds a claim older than the derived stale bound (the
    process is not killed)."""
    now = _hermes_now().isoformat()
    changed = 0
    recovered: List[Dict[str, Any]] = []
    # Derived on the first live-owned row only: the bound reads config, and the idle gateway
    # tick must stay config-free (tests/cron/test_idle_tick_config_skip.py).
    stale_after: Optional[float] = None
    stale_after_resolved = False
    with _transaction() as conn:
        rows = conn.execute(
            """SELECT id, job_id, status, process_id, pid, process_started_at,
                      handoff_pending, handoff_started_at, claimed_at, lease_until
               FROM executions
               WHERE status IN ('claimed','running')"""
        ).fetchall()
        for row in rows:
            if row["process_id"] == _PROCESS_ID:
                continue
            # Record the real reason, never a generic one: an expired lease is reported as an
            # expired lease (the attempt may have been redelivered), not silently as a restart.
            reason = (
                _LEASE_EXPIRED_REASON
                if execution_lease_expired(dict(row))
                else _OWNER_GONE_REASON
            )
            if _owner_is_live(int(row["pid"]), row["process_started_at"]):
                # A live owner is normally a legitimately running job. A worker permanently
                # deadlocked (e.g. futex_wait behind a route/proxy flip, #115692) also passes
                # this check, so a claim older than the derived bound is treated as wedged
                # and released — the external-worker wait loop polls this ledger for a
                # terminal status, so the job can fire again. The wedged worker PROCESS is
                # NOT terminated here (leaked until host restart); rows owned by this process
                # (process_id == _PROCESS_ID, in-process runs) are skipped above and remain
                # out of scope.
                if not stale_after_resolved:
                    stale_after = _live_owner_stale_after_seconds()
                    stale_after_resolved = True
                if stale_after is None or _claim_age_seconds(row["claimed_at"]) <= stale_after:
                    continue
                reason = _OWNER_WEDGED_REASON
            handoff_started_at = row["handoff_started_at"]
            if (
                row["handoff_pending"]
                and handoff_started_at is not None
                and time.time() - float(handoff_started_at)
                < HANDOFF_ADOPTION_GRACE_SECONDS
            ):
                continue
            cur = conn.execute(
                """UPDATE executions
                   SET status='unknown', finished_at=?, error=?,
                       handoff_pending=0, handoff_started_at=NULL
                   WHERE id=? AND status=? AND process_id=? AND pid=?
                     AND handoff_pending=?
                     AND handoff_started_at IS ?""",
                (now, reason, row["id"], row["status"], row["process_id"], row["pid"],
                 row["handoff_pending"], row["handoff_started_at"]),
            )
            changed += cur.rowcount
            if cur.rowcount:
                _append_event(
                    conn, execution_id=str(row["id"]), job_id=str(row["job_id"]),
                    kind=EVENT_RECLAIMED, detail=reason,
                )
                record = _fetch(conn, row["id"])
                if record is not None:
                    recovered.append(record)
        if changed:
            _prune_unlocked(conn)
    for record in recovered:
        _emit_execution_state(record)
    return changed


def terminalize_dead_owner(execution_id: str, *, reason: str) -> bool:
    """Record one attempt as ``unknown`` with a cause this process actually observed.

    ``recover_interrupted_executions`` sweeps every attempt whose owner is provably
    dead, and knows nothing but that absence — so all it can write is
    ``_OWNER_GONE_REASON``, which asserts a scheduler restart. A waiter that held the
    worker's ``Popen`` knows more: the owner was that external worker, and it exited
    with a known status. Without this, a manual run whose worker dies is filed as
    "Scheduler restarted ..." (a restart that never happened) and, because the sweep
    leaves the row terminal, the waiter reports success and never records the run —
    the job's ``fire_claim`` then blocks the next manual fire for the whole lease
    (#128509).

    The attempt stays ``unknown``, not ``failed``: whether side effects ran is still
    unknown. Only the CAUSE becomes truthful. Returns False — leaving the caller to
    fall back to the generic sweep — when the row is absent, already terminal, owned by
    this process, inside the handoff adoption grace, or owned by a live process: a
    worker that is still running must never be terminalized out from under itself.
    """
    now = _hermes_now().isoformat()
    with _transaction() as conn:
        row = conn.execute(
            """SELECT id, status, process_id, pid, process_started_at,
                      handoff_pending, handoff_started_at
               FROM executions WHERE id=?""",
            (execution_id,),
        ).fetchone()
        if row is None or row["status"] not in ("claimed", "running"):
            return False
        if row["process_id"] == _PROCESS_ID:
            return False
        if _owner_is_live(int(row["pid"]), row["process_started_at"]):
            return False
        handoff_started_at = row["handoff_started_at"]
        if (
            row["handoff_pending"]
            and handoff_started_at is not None
            and time.time() - float(handoff_started_at) < HANDOFF_ADOPTION_GRACE_SECONDS
        ):
            return False
        cur = conn.execute(
            """UPDATE executions
               SET status='unknown', finished_at=?, error=?,
                   handoff_pending=0, handoff_started_at=NULL
               WHERE id=? AND status=? AND process_id=? AND pid=?""",
            (now, reason, row["id"], row["status"], row["process_id"], row["pid"]),
        )
        if cur.rowcount != 1:
            return False
        row = _fetch(conn, execution_id)
        if row is not None:
            _append_event(
                conn, execution_id=execution_id, job_id=str(row["job_id"]),
                kind=EVENT_RECLAIMED, detail=str(reason),
            )
        record = _fetch(conn, execution_id)
        _prune_unlocked(conn)
    _emit_execution_state(record)
    return True


def list_executions(
    *, job_id: Optional[str] = None, limit: int = 50, before_claimed_at: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Return indexed, newest-first execution history with cursor pagination."""
    clauses: List[str] = []
    params: List[Any] = []
    if job_id is not None:
        clauses.append("job_id=?")
        params.append(str(job_id))
    if before_claimed_at is not None:
        # Same (instant, text) key as the ORDER BY, so a page never skips or repeats a row.
        clauses.append("(julianday(claimed_at), claimed_at) < (julianday(?), ?)")
        params.extend([str(before_claimed_at)] * 2)
    where = " WHERE " + " AND ".join(clauses) if clauses else ""
    params.append(max(1, min(int(limit), 500)))
    # Stamps carry the local offset, which changes at DST and on a timezone change, so text order
    # is not time order. julianday() compares instants (ms); the text breaks same-ms ties.
    with _transaction() as conn:
        rows = conn.execute(
            "SELECT * FROM executions" + where
            + " ORDER BY julianday(claimed_at) DESC, claimed_at DESC, id DESC LIMIT ?",
            params,
        ).fetchall()
    return [dict(row) for row in rows]


def get_execution(execution_id: str) -> Optional[Dict[str, Any]]:
    """Return one exact execution attempt, or ``None`` when it is absent."""
    with _transaction() as conn:
        row = conn.execute(
            "SELECT * FROM executions WHERE id=?",
            (str(execution_id),),
        ).fetchone()
    return dict(row) if row is not None else None


def latest_execution(job_id: str) -> Optional[Dict[str, Any]]:
    rows = list_executions(job_id=job_id, limit=1)
    return rows[0] if rows else None


def live_inflight_execution(job_id: str) -> Optional[Dict[str, Any]]:
    """The job's latest attempt while it is still claimed/running under a LIVE owner and an
    unexpired lease, else ``None``.

    This is scheduler OWNERSHIP, not recent activity: a run inside a long tool call writes no
    heartbeat yet stays owned, while a run whose process died (watchdog kill, crash) does not.
    An attempt whose explicit lease has lapsed is NEVER reported as running — even if a pid
    reading still looks alive — because the ledger can no longer prove it holds the run.
    Read-only — unlike ``recover_interrupted_executions`` it never rewrites a row.
    """
    record = latest_execution(job_id)
    if not record or record.get("status") not in ("claimed", "running"):
        return None
    if execution_lease_expired(record):
        return None
    if not _owner_is_live(int(record["pid"]), record.get("process_started_at")):
        return None
    return record


def latest_executions(job_ids: List[str]) -> Dict[str, Dict[str, Any]]:
    """Load latest execution for many jobs in one query."""
    clean = [str(job_id) for job_id in dict.fromkeys(job_ids) if job_id]
    if not clean:
        return {}
    placeholders = ",".join("?" for _ in clean)
    # One windowed sort: a per-row correlated ORDER BY julianday() cannot use the index and
    # grows quadratically with history (~90 ms at 1000 rows).
    with _transaction() as conn:
        rows = conn.execute(
            f"""SELECT e.* FROM executions e WHERE e.id IN (
                  SELECT id FROM (
                    SELECT id, ROW_NUMBER() OVER (
                             PARTITION BY job_id
                             ORDER BY julianday(claimed_at) DESC, claimed_at DESC, id DESC
                           ) AS rn
                    FROM executions WHERE job_id IN ({placeholders}))
                  WHERE rn=1)""",
            clean,
        ).fetchall()
    return {row["job_id"]: dict(row) for row in rows}


def execution_outcome(record: Optional[Dict[str, Any]]) -> str:
    """The run's outcome in the frozen vocabulary: ``completed``, ``failed`` or ``unknown``.

    Anything the ledger cannot prove — an interrupted/abandoned attempt, a missing finish record,
    an expired claim, or an absent row — is ``unknown``. A still-in-flight attempt is ``unknown``
    because it has not produced a provable outcome yet; its live ``status`` stays visible on the
    record itself.
    """
    if not record:
        return "unknown"
    status = str(record.get("status") or "")
    return status if status in ("completed", "failed") else "unknown"


def renew_execution_lease(execution_id: str) -> bool:
    """Push a live attempt's explicit lease forward by the shared ``FIRE_CLAIM_TTL_SECONDS``.

    Called on heartbeat so a legitimately long run never reads as an expired lease. False when the
    attempt is terminal or not owned by this process (never extend another owner's lease).
    """
    with _transaction() as conn:
        cur = conn.execute(
            "UPDATE executions SET lease_until=? WHERE id=? AND status IN ('claimed','running')"
            " AND process_id=? AND pid=?",
            (_lease_deadline(_hermes_now()), str(execution_id), _PROCESS_ID, os.getpid()),
        )
    return cur.rowcount == 1


def list_execution_events(
    *, execution_id: Optional[str] = None, job_id: Optional[str] = None, limit: int = 200,
) -> List[Dict[str, Any]]:
    """Append-only event trail, oldest-first, for one attempt or one job. Read-only.

    The trail is the run's own history (claimed → started → finished/failed → delivered or
    not_delivered, or reclaimed when the lease lapsed); it is never reconstructed from mutable
    job-store fields and never rewritten after the fact.
    """
    clauses: List[str] = []
    params: List[Any] = []
    if execution_id is not None:
        clauses.append("execution_id=?")
        params.append(str(execution_id))
    if job_id is not None:
        clauses.append("job_id=?")
        params.append(str(job_id))
    where = " WHERE " + " AND ".join(clauses) if clauses else ""
    params.append(max(1, min(int(limit), 1000)))
    with _transaction() as conn:
        rows = conn.execute(
            "SELECT * FROM execution_events" + where + " ORDER BY id ASC LIMIT ?", params
        ).fetchall()
    return [dict(row) for row in rows]
