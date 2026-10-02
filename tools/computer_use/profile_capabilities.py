"""Per-profile capability grants for the ``computer_use`` toolset.

Ported from OpenDots' per-Dot computer service (``docs/COMPUTERS.md``, ``src/server/computer-service.ts``):
a capability is granted PER agent/profile, starts DISABLED, is checked before any device action, and
revoking one both refuses later actions and signals cancellation for in-flight ones. The container half
of that design is deliberately out of scope here.

Semantics (fail closed):

* A profile with a capability store present has every capability DISABLED unless the store records it
  as granted. Any capability not explicitly granted is refused with an explicit reason string.
* An unreadable / malformed store grants nothing and says why -- never a silent default-on.
* A profile with NO store is only enforced when enforcement was explicitly switched on
  (``HERMES_COMPUTER_USE_CAPABILITIES_ENFORCE=1`` or ``computer_use.capabilities_enforce: true``);
  otherwise enforcement is reported as not configured so existing installs keep their behaviour.

State lives in the profile's own state area: ``$HERMES_HOME/computer_use_capabilities.json``.
"""

from __future__ import annotations

import contextlib
import itertools
import json
import logging
import os
import threading
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

BROWSER = "browser"
FILES = "files"
SHELL = "shell"
CAPABILITIES: Tuple[str, ...] = (BROWSER, FILES, SHELL)

_STORE_NAME = "computer_use_capabilities.json"
_ENFORCE_ENV = "HERMES_COMPUTER_USE_CAPABILITIES_ENFORCE"
_TRUE_VALUES = {"1", "true", "yes", "on"}

# Which capability an action requires. ``type``/``key``/``set_value`` are the tool's only command-ish
# surface (they drive a shell, locator bar or open terminal), so they need SHELL. Everything else is
# browser/desktop control. FILES has no action yet -- it exists so a future file action cannot ship
# ungated, and ``check("files")`` refuses by default today.
CAPABILITY_BY_ACTION: Dict[str, str] = {
    "capture": BROWSER, "wait": BROWSER, "list_apps": BROWSER, "list_windows": BROWSER, "focus_app": BROWSER,
    "click": BROWSER, "double_click": BROWSER, "right_click": BROWSER, "middle_click": BROWSER,
    "drag": BROWSER, "scroll": BROWSER,
    "type": SHELL, "key": SHELL, "set_value": SHELL,
}


class CapabilityStoreError(RuntimeError):
    """The capability store could not be read/written (missing is NOT an error -- that is default-deny)."""


class ActionDenied(RuntimeError):
    """The action's capability is not granted (or the store failed closed)."""

    def __init__(self, capability: Optional[str], reason: str) -> None:
        super().__init__(reason)
        self.capability = capability
        self.reason = reason


class CapabilityRevoked(ActionDenied):
    """The capability was revoked while this action was in flight; its result must be discarded."""


# ── Store ───────────────────────────────────────────────────────────────────
def store_path(profile: Optional[Any] = None) -> Path:
    """``<profile home>/computer_use_capabilities.json``; *profile* is a home dir, None = current profile."""
    if profile is not None:
        return Path(profile) / _STORE_NAME
    from hermes_constants import get_hermes_home
    return get_hermes_home() / _STORE_NAME


def _profile_key(profile: Optional[Any] = None) -> str:
    return str(store_path(profile).parent)


def _read(path: Path) -> Optional[Dict[str, Any]]:
    """Parsed store, or None when it does not exist. Raises CapabilityStoreError when malformed/unreadable."""
    if not path.exists():
        return None
    try:
        raw = path.read_text(encoding="utf-8")
    except Exception as e:  # permission / IO
        raise CapabilityStoreError(f"{path} is unreadable: {e}") from e
    try:
        data = json.loads(raw)
    except Exception as e:
        raise CapabilityStoreError(f"{path} is not valid JSON: {e}") from e
    if not isinstance(data, dict):
        raise CapabilityStoreError(f"{path} must contain a JSON object, found {type(data).__name__}")
    caps = data.get("capabilities", {})
    if not isinstance(caps, dict) or any(not isinstance(v, bool) for v in caps.values()):
        raise CapabilityStoreError(f"{path} has a malformed `capabilities` map (expected name -> boolean)")
    return data


def _write(profile: Optional[Any], policy: Dict[str, Any]) -> None:
    """Write the store owner-only; a failure raises (never silently loses a grant change)."""
    path = store_path(profile)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(policy, indent=1, sort_keys=True) + "\n", encoding="utf-8")
        with contextlib.suppress(OSError):
            os.chmod(tmp, 0o600)
        os.replace(tmp, path)
    except Exception as e:
        raise CapabilityStoreError(f"could not write capability store {path}: {e}") from e


def load_policy(profile: Optional[Any] = None) -> Optional[Dict[str, Any]]:
    """Parsed store (None when absent); raises CapabilityStoreError when unreadable/malformed."""
    return _read(store_path(profile))


def _config_enforce_switch() -> bool:
    if os.environ.get(_ENFORCE_ENV, "").strip().lower() in _TRUE_VALUES:
        return True
    with contextlib.suppress(Exception):
        from hermes_cli.config import load_config
        return bool(((load_config() or {}).get("computer_use") or {}).get("capabilities_enforce", False))
    return False


def enforcement(profile: Optional[Any] = None) -> Tuple[bool, str]:
    """``(enforced, deny_reason)``. A malformed store fails CLOSED with an explicit reason."""
    try:
        policy = load_policy(profile)
    except CapabilityStoreError as e:
        return True, f"computer_use capability store failed closed ({e}); no capability is available"
    if policy is not None:
        return bool(policy.get("enforce", True)), ""
    if _config_enforce_switch():
        return True, ("capability enforcement is on but no capability has been granted for this profile; "
                      "capability access starts disabled")
    return False, ""


def _granted(capability: str, profile: Optional[Any] = None) -> Tuple[bool, str]:
    try:
        policy = load_policy(profile)
    except CapabilityStoreError as e:
        return False, f"computer_use capability store failed closed ({e}); {capability!r} is not available"
    if policy is None:
        return False, f"no capability grant is recorded for this profile; {capability!r} is disabled by default"
    if (policy.get("capabilities") or {}).get(capability) is True:
        return True, ""
    return False, f"capability {capability!r} is not granted for this profile"


def is_granted(capability: str, profile: Optional[Any] = None) -> Tuple[bool, str]:
    """``(granted, reason)`` -- the reason is empty only when granted."""
    cap = str(capability or "").strip().lower()
    if cap not in CAPABILITIES:
        return False, f"unknown capability {capability!r}; known capabilities are {', '.join(CAPABILITIES)}"
    enforced, deny = enforcement(profile)
    if not enforced:
        return True, ""  # enforcement not configured for this profile: legacy behaviour
    if deny:
        return False, f"capability {cap!r} is unavailable: {deny}"
    return _granted(cap, profile)


def check(capability: str, profile: Optional[Any] = None) -> Optional[str]:
    """The single check the tool calls before acting: None = allowed, else an explicit refusal reason."""
    granted, reason = is_granted(capability, profile)
    return None if granted else reason


def capability_for_action(action: str) -> Optional[str]:
    return CAPABILITY_BY_ACTION.get(str(action or "").strip().lower())


def _validate(capability: str) -> str:
    cap = str(capability or "").strip().lower()
    if cap not in CAPABILITIES:
        raise ValueError(f"unknown capability {capability!r}; known capabilities are {', '.join(CAPABILITIES)}")
    return cap


def _load_for_write(profile: Optional[Any]) -> Dict[str, Any]:
    policy = load_policy(profile)  # raises on malformed: never overwrite a store we cannot parse
    return dict(policy) if policy else {"version": 1, "capabilities": {}}


def grant(capability: str, profile: Optional[Any] = None) -> Dict[str, Any]:
    """Explicitly grant one capability for this profile (persisted; the only way anything turns on)."""
    cap = _validate(capability)
    policy = _load_for_write(profile)
    policy.setdefault("version", 1)
    policy.setdefault("capabilities", {})[cap] = True
    policy["enforce"] = True
    _write(profile, policy)
    return policy


def revoke(capability: str, profile: Optional[Any] = None) -> Dict[str, Any]:
    """Revoke one capability: refuse subsequent actions AND signal cancellation for in-flight ones."""
    cap = _validate(capability)
    policy = _load_for_write(profile)
    policy.setdefault("version", 1)
    policy.setdefault("capabilities", {})[cap] = False
    policy["enforce"] = True
    _write(profile, policy)
    _signal_inflight(cap, profile)
    return policy


# ── In-flight leases / cancellation ─────────────────────────────────────────
_inflight_lock = threading.Lock()
_inflight: Dict[str, Dict[int, "ActionLease"]] = {}
_lease_ids = itertools.count(1)
_CANCEL_METHODS = ("cancel_current_action", "cancel", "abort", "interrupt")


class ActionLease:
    """One capability-checked action. ``check()`` raises once the capability was revoked."""

    def __init__(self, capability: Optional[str], profile: Optional[Any], enforced: bool) -> None:
        self.capability = capability
        self.profile = profile
        self.enforced = enforced
        self._id = next(_lease_ids)
        self._event = threading.Event()
        self._cancel_cb: Optional[Callable[..., Any]] = None

    @property
    def cancelled(self) -> bool:
        return self._event.is_set()

    def bind_backend(self, backend: Any) -> None:
        """Cooperative cancellation hook: use the backend's own cancel entry point when it has one."""
        for name in _CANCEL_METHODS:
            cb = getattr(backend, name, None)
            if callable(cb):
                self._cancel_cb = cb
                if self._event.is_set():
                    self._fire_cancel()
                return

    def _fire_cancel(self) -> None:
        if self._cancel_cb is not None:
            with contextlib.suppress(Exception):
                self._cancel_cb()

    def _revoke(self) -> None:
        self._event.set()
        self._fire_cancel()

    def check(self) -> None:
        if self._event.is_set():
            raise CapabilityRevoked(self.capability, (
                f"capability {self.capability!r} was revoked while this action was in flight; its result was "
                "discarded and nothing further will be dispatched until it is re-granted"))

    def __enter__(self) -> "ActionLease":
        return self

    def __exit__(self, *exc: Any) -> bool:
        _unregister(self)
        return False


def _register(lease: ActionLease) -> None:
    with _inflight_lock:
        _inflight.setdefault(_profile_key(lease.profile), {})[lease._id] = lease


def _unregister(lease: ActionLease) -> None:
    with _inflight_lock:
        bucket = _inflight.get(_profile_key(lease.profile))
        if bucket is not None:
            bucket.pop(lease._id, None)
            if not bucket:
                _inflight.pop(_profile_key(lease.profile), None)


def _signal_inflight(capability: str, profile: Optional[Any]) -> int:
    with _inflight_lock:
        leases = [lease for lease in _inflight.get(_profile_key(profile), {}).values()
                  if lease.capability == capability]
    for lease in leases:
        lease._revoke()
    return len(leases)


def begin_action(action: str, session_id: str = "", profile: Optional[Any] = None) -> ActionLease:
    """Check the action's capability and open a cancellable lease. Raises ActionDenied when refused.

    Unknown actions (which ``_dispatch`` rejects anyway) are not gated here so the "did you mean"
    hint still reaches the model.
    """
    cap = capability_for_action(action)
    if cap is None:
        return ActionLease(None, profile, False)
    enforced, _ = enforcement(profile)
    reason = check(cap, profile)
    if reason is not None:
        raise ActionDenied(cap, reason)
    lease = ActionLease(cap, profile, enforced)
    if enforced:
        _register(lease)
    return lease


def reset_for_tests() -> None:  # pragma: no cover - test seam
    with _inflight_lock:
        _inflight.clear()
