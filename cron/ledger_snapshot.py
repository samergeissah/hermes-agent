"""Pure read surface over the cron job store + execution ledger.

The EVE snapshot service calls this to render the scheduled-work view: for every job it needs the
name, state, schedule, the next run time, and the last run's record. Every field is derived from
the live store/ledger at call time. A field the ledger cannot prove is ``None`` — the surface never
falls back to a synthesised default (no "assume completed", no recomputed next-run, no invented
name).

Read-only: it opens the existing stores, never recovers an interrupted attempt, never rewrites a
job record, and never fires anything. Deriving the view must be safe to call from a dashboard or a
snapshot pump as often as it likes.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from cron import executions as _executions

# Schedule kinds Hermes actually honours. An unknown/missing kind means the store cannot prove
# when (or whether) the job runs next, so the surface reports ``None`` rather than guessing.
_KNOWN_SCHEDULE_KINDS = ("once", "interval", "cron")


def _schedule_view(record: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """The job's schedule dict, or ``None`` when it is absent or of an unknown kind."""
    if not isinstance(record, dict):
        return None
    schedule = record.get("schedule")
    if not isinstance(schedule, dict):
        return None
    if schedule.get("kind") not in _KNOWN_SCHEDULE_KINDS:
        return None
    return dict(schedule)


def _next_run_at(record: Optional[Dict[str, Any]]) -> Optional[str]:
    """The stored next occurrence, or ``None`` when the store cannot prove one.

    ``None`` covers every unprovable case: no record, unknown/absent schedule kind, a missing or
    malformed ``next_run_at``. The value is the store's own stamp — never recomputed here, since a
    recomputed instant would be this module's guess rather than the scheduler's plan.
    """
    if _schedule_view(record) is None:
        return None
    raw = record.get("next_run_at") if isinstance(record, dict) else None
    if not isinstance(raw, str) or not raw.strip():
        return None
    try:
        from datetime import datetime

        datetime.fromisoformat(raw.strip())
    except ValueError:
        return None
    return raw.strip()


def _last_run(job_id: str) -> Optional[Dict[str, Any]]:
    """The job's latest ledger attempt with its event trail, or ``None`` when there is no row.

    An absent row is reported as no last run — never as a synthesised completed one.
    """
    record = _executions.latest_execution(job_id)
    if record is None:
        return None
    view = dict(record)
    view["outcome"] = _executions.execution_outcome(record)
    view["lease_expired"] = _executions.execution_lease_expired(record)
    view["events"] = _executions.list_execution_events(execution_id=record["id"])
    return view


def _job_entry(
    job_id: str, record: Optional[Dict[str, Any]], *, name: Optional[str] = None,
) -> Dict[str, Any]:
    return {
        "job_id": job_id,
        "name": name if name is not None else (
            record.get("name") if isinstance(record, dict) else None
        ),
        "state": record.get("state") if isinstance(record, dict) else None,
        "enabled": bool(record["enabled"]) if isinstance(record, dict)
        and "enabled" in record else None,
        "schedule": _schedule_view(record),
        "schedule_display": record.get("schedule_display") if isinstance(record, dict) else None,
        "next_run_at": _next_run_at(record),
        "last_run": _last_run(job_id),
    }


def _store_index() -> Dict[str, Dict[str, Any]]:
    """Job records keyed by id, read straight from the live store (no repair, no re-save)."""
    from cron.jobs import load_jobs

    index: Dict[str, Dict[str, Any]] = {}
    for record in load_jobs():
        if isinstance(record, dict) and record.get("id"):
            index[str(record["id"])] = record
    return index


def job_ledger_snapshot(job_id: Optional[str] = None) -> Any:
    """A snapshot for one job (``job_id``) or every job (``job_id is None``).

    Each entry: ``job_id``, ``name``, ``state``, ``enabled``, ``schedule``, ``schedule_display``,
    ``next_run_at`` (``None`` when unknown), and ``last_run`` (``None`` when the ledger has no row,
    else the latest attempt plus ``outcome``/``lease_expired``/``events``).

    Returns ``None`` for a specific ``job_id`` present in neither the store nor the ledger; a list
    (store jobs first, then ledger-only ids) when called with no id.
    """
    index = _store_index()
    if job_id is not None:
        key = str(job_id)
        record = index.get(key)
        if record is None and _executions.latest_execution(key) is None:
            return None
        return _job_entry(key, record)

    entries = [_job_entry(key, record) for key, record in index.items()]
    seen = set(index)
    for row in _executions.list_executions(limit=500):
        row_job = str(row.get("job_id"))
        if row_job in seen:
            continue
        seen.add(row_job)
        entries.append(_job_entry(row_job, None))
    return entries
