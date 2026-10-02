"""Append-only activity record for ``computer_use`` actions.

Ported from OpenDots' computer service activity log (``docs/COMPUTERS.md``: "inspect action names, who
requested them, and success/failure. The audit record deliberately excludes typed values, file
contents, and full commands").

One JSON line per action under the profile's own state area: ``$HERMES_HOME/computer_use_activity.jsonl``.

Kept:        action, requester (session/agent), profile, timestamp, success/failure, the real error
             string, and a redacted ``shape`` (enum-valued options, presence flags, character counts).
Never kept:  typed text, key combos, clipboard contents, ``set_value`` values, file contents, full
             command lines -- ``action_shape`` whitelists shape keys and stores only counts/flags.

The writer FAILS LOUDLY: an append failure raises ``AuditWriteError`` rather than dropping the record.
The retention cap keeps the most recent N records per profile, trimming oldest-first.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

DEFAULT_RETENTION = 1000  # mirrors OpenDots: "activity retains the latest 1,000 completed records per Dot"
_ACTIVITY_NAME = "computer_use_activity.jsonl"
_RETENTION_ENV = "HERMES_COMPUTER_USE_ACTIVITY_RETENTION"

# Enum-valued options safe to record verbatim (they describe the action's shape, not its payload).
_SAFE_ENUM_KEYS = ("mode", "direction", "button", "delivery_mode")
# Payload keys that must NEVER be recorded; only their character count may travel.
_PAYLOAD_TEXT_KEYS = ("text", "keys", "value", "clipboard", "command")


class AuditWriteError(RuntimeError):
    """The activity record could not be appended. Never swallowed by this module."""


def activity_path(profile: Optional[Any] = None) -> Path:
    """``<profile home>/computer_use_activity.jsonl``; *profile* is a home dir, None = current profile."""
    if profile is not None:
        return Path(profile) / _ACTIVITY_NAME
    from hermes_constants import get_hermes_home
    return get_hermes_home() / _ACTIVITY_NAME


def _profile_key(profile: Optional[Any] = None) -> str:
    return str(activity_path(profile).parent)


def retention_limit() -> int:
    """Records kept per profile: env -> config -> DEFAULT_RETENTION; invalid/<=0 falls back (never unbounded-safe)."""
    raw: Any = os.environ.get(_RETENTION_ENV, "")
    if not str(raw).strip():
        with contextlib.suppress(Exception):
            from hermes_cli.config import load_config
            raw = ((load_config() or {}).get("computer_use") or {}).get("activity_retention", "")
    try:
        value = int(str(raw).strip())
    except (TypeError, ValueError):
        return DEFAULT_RETENTION
    return value if value > 0 else DEFAULT_RETENTION


def action_shape(action: str, args: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Redacted description of the action's SHAPE -- never its payload. Character counts, not contents."""
    args = args if isinstance(args, dict) else {}
    shape: Dict[str, Any] = {key: str(args[key])[:32] for key in _SAFE_ENUM_KEYS if isinstance(args.get(key), str)}
    if args.get("element") is not None:
        shape["has_element"] = True
    if args.get("coordinate") is not None:
        shape["has_coordinate"] = True
    for key in _PAYLOAD_TEXT_KEYS:
        if args.get(key) is not None:
            shape[f"{key}_chars"] = len(str(args.get(key)))
    if args.get("capture_after"):
        shape["capture_after"] = True
    return shape


def requester_id(session_id: Optional[str] = None, profile: Optional[Any] = None) -> str:
    return str(session_id or "") or _profile_key(profile)


def record(action: str, requester: str, ok: bool, *, error: Optional[str] = None, code: Optional[str] = None,
           shape: Optional[Dict[str, Any]] = None, profile: Optional[Any] = None,
           retention: Optional[int] = None, timestamp: Optional[str] = None) -> Dict[str, Any]:
    """Append one activity record and trim oldest-first. Raises AuditWriteError if the append fails."""
    line: Dict[str, Any] = {
        "ts": timestamp or datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "action": str(action or ""),
        "requester": str(requester or ""),
        "profile": _profile_key(profile),
        "ok": bool(ok),
    }
    if shape:
        line["shape"] = shape
    if code:
        line["code"] = str(code)
    if error:
        line["error"] = str(error)[:500]
    path = activity_path(profile)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(line, ensure_ascii=False, sort_keys=True) + "\n")
    except Exception as exc:  # loud: losing a record is worse than a broken action
        logger.error("computer_use audit: FAILED to append activity record for action %r to %s: %s",
                     action, path, exc)
        raise AuditWriteError(f"could not append computer_use activity record to {path}: {exc}") from exc
    _trim(path, retention if retention is not None else retention_limit())
    return line


def read_records(profile: Optional[Any] = None) -> list:
    """All recorded activity for a profile, oldest-first (test/CLI read surface)."""
    path = activity_path(profile)
    if not path.exists():
        return []
    out = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        with contextlib.suppress(Exception):
            out.append(json.loads(raw))
    return out


def _trim(path: Path, limit: int) -> int:
    """Keep the most recent *limit* records, dropping the oldest. Logs loudly on failure (record is safe)."""
    try:
        lines = [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
        if limit <= 0 or len(lines) <= limit:
            return 0
        dropped = len(lines) - limit
        tmp = path.with_suffix(".jsonl.tmp")
        tmp.write_text("\n".join(lines[-limit:]) + "\n", encoding="utf-8")
        os.replace(tmp, path)
        return dropped
    except Exception as exc:
        logger.error("computer_use audit: retention trim failed for %s: %s", path, exc)
        return 0


def record_denied(action: str, session_id: Optional[str], reason: str, *, profile: Optional[Any] = None,
                  args: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
    """Audit a refused action without ever breaking the refusal path if audit fails."""
    try:
        return record(action, requester_id(session_id, profile), ok=False, error=reason,
                      code="capability_denied", shape=action_shape(action, args), profile=profile)
    except AuditWriteError as exc:  # the refusal itself must still reach the caller
        logger.error("computer_use audit: could not record denied action %r: %s", action, exc)
        return None


class _Recorder:
    """Accumulates one action's outcome, then flushes exactly one record."""

    def __init__(self, action: str, session_id: str, args: Optional[Dict[str, Any]],
                 profile: Optional[Any], lease: Any = None, retention: Optional[int] = None) -> None:
        self.action = action
        self.requester = requester_id(session_id, profile)
        self.profile = profile
        self.lease = lease
        self.retention = retention
        self.shape = action_shape(action, args)
        self.ok: Optional[bool] = None
        self.error: Optional[str] = None
        self.code: Optional[str] = None
        self.record: Optional[Dict[str, Any]] = None
        self._flushed = False

    def assess(self, result: Any) -> None:
        """Mark success/failure from the tool's returned payload (JSON string, dict, or plain result)."""
        data = result
        if isinstance(result, str):
            with contextlib.suppress(Exception):
                data = json.loads(result)
        if isinstance(data, dict):
            err = data.get("error")
            if err or data.get("ok") is False:
                self.ok, self.error = False, str(err or "action reported failure")
                self.code = str(data.get("code") or "") or None
            else:
                self.ok = True
        else:
            self.ok = True

    def _flush(self) -> None:
        if self._flushed:
            return
        self._flushed = True
        ok, error, code = (True if self.ok is None else bool(self.ok)), self.error, self.code
        if self.lease is not None and self.lease.cancelled:
            ok, error = False, error or "capability revoked while the action was in flight; result discarded"
            code = code or "capability_revoked"
        try:
            self.record = record(self.action, self.requester, ok=ok, error=error, code=code, shape=self.shape,
                                 profile=self.profile, retention=self.retention)
        except AuditWriteError:
            raise  # loud by design


@contextlib.contextmanager
def audit_action(action: str, session_id: str, args: Optional[Dict[str, Any]] = None, lease: Any = None,
                 profile: Optional[Any] = None, retention: Optional[int] = None):
    """Context manager: write exactly one activity record per action, success or failure."""
    recorder = _Recorder(action, session_id, args, profile, lease=lease, retention=retention)
    try:
        yield recorder
    except BaseException as exc:
        recorder.ok, recorder.error = False, f"{type(exc).__name__}: {exc}"
        recorder._flush()
        raise
    recorder._flush()
