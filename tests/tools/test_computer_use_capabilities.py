"""Per-profile capability + activity-audit tests for ``computer_use``.

These exercise the permission check and the audit record with an INJECTED fake backend: computer_use
drives a real desktop through cua-driver, and a test that moves the user's mouse is a failed test.
"""

from __future__ import annotations

import json
import os
from unittest.mock import MagicMock, patch

import pytest

from tools.computer_use import audit as cu_audit
from tools.computer_use import profile_capabilities as caps
from tools.computer_use.backend import ActionResult, CaptureResult
from tools.computer_use import tool as cu_tool

SENTINEL = "SENTINEL_PAYLOAD_9d3f2b"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

class FakeBackend:
    """In-memory backend: records calls, never touches a desktop."""

    def __init__(self, on_call=None):
        self.calls = []
        self.on_call = on_call          # callable(action, kwargs) invoked inside the action
        self.cancel_invoked = False
        self._last_app = "Finder"
        self._last_target = {}

    def start(self): pass

    def stop(self): pass

    def is_available(self): return True

    def _record(self, action, **kw):
        self.calls.append((action, kw))
        if self.on_call is not None:
            self.on_call(action, kw)
        return ActionResult(ok=True, action=action)

    def click(self, **kw): return self._record("click", **kw)
    def type_text(self, text="", **kw): return self._record("type", text=text, **kw)
    def key(self, keys="", **kw): return self._record("key", keys=keys, **kw)
    def set_value(self, value="", element=None, **kw): return self._record("set_value", value=value, **kw)
    def scroll(self, **kw): return self._record("scroll", **kw)
    def focus_app(self, app="", **kw): return self._record("focus_app", app=app, **kw)

    def cancel_current_action(self):
        self.cancel_invoked = True

    def capture(self, mode="som", app=None, **kw):
        self.calls.append(("capture", {"mode": mode, "app": app}))
        if self.on_call is not None:
            self.on_call("capture", {"mode": mode})
        return CaptureResult(mode=mode, width=100, height=80, png_b64=None, elements=[], app=app or "", window_title="")


@pytest.fixture(autouse=True)
def _clean_state(grant_computer_use_approvals):
    """Reset cached backends + capability leases; approvals answer "once" for destructive actions."""
    caps.reset_for_tests()
    cu_tool.reset_backend_for_tests()
    yield
    caps.reset_for_tests()
    cu_tool.reset_backend_for_tests()


@pytest.fixture
def fake_backend():
    backend = FakeBackend()
    with patch.object(cu_tool, "_new_backend", lambda permission_mode: backend):
        yield backend


@pytest.fixture
def enforce(monkeypatch):
    """Turn capability enforcement on for the current (isolated) profile."""
    monkeypatch.setenv(caps._ENFORCE_ENV, "1")


def _call(action, args=None, **kw):
    out = cu_tool.handle_computer_use({"action": action, **(args or {})}, **kw)
    return json.loads(out) if isinstance(out, str) else out


def _records():
    return cu_audit.read_records()


# ---------------------------------------------------------------------------
# A. Capability gate
# ---------------------------------------------------------------------------

def test_ungranted_capability_is_refused_with_reason(enforce, fake_backend):
    out = _call("click", {"element": 3}, session_id="sess-1")
    assert out["ok"] is False
    assert out["code"] == "capability_denied"
    assert "browser" in out["error"] and out["error"].strip()
    assert fake_backend.calls == []                       # nothing reached the device
    denied = _records()
    assert len(denied) == 1 and denied[0]["ok"] is False and denied[0]["code"] == "capability_denied"


def test_check_is_the_single_fail_closed_check(enforce):
    reason = caps.check(caps.BROWSER)
    assert isinstance(reason, str) and reason            # explicit reason, not a silent bool
    assert caps.check("not-a-capability")                 # unknown capability also refuses
    assert caps.check(caps.SHELL) is not None


def test_granted_capability_proceeds_and_writes_exactly_one_audit_record(fake_backend):
    caps.grant(caps.BROWSER)
    out = _call("click", {"element": 3}, session_id="sess-1")
    assert out.get("ok") is True and out.get("code") is None
    assert [c[0] for c in fake_backend.calls] == ["click"]
    records = _records()
    assert len(records) == 1
    assert records[0]["action"] == "click"
    assert records[0]["requester"] == "sess-1"
    assert records[0]["ok"] is True
    assert records[0]["ts"]


def test_revocation_refuses_the_next_action(fake_backend):
    caps.grant(caps.BROWSER)
    assert _call("click", {"element": 1})["ok"] is True
    caps.revoke(caps.BROWSER)
    out = _call("click", {"element": 1})
    assert out["ok"] is False and out["code"] == "capability_denied"
    assert [c[0] for c in fake_backend.calls] == ["click"]  # only the pre-revoke action dispatched


def test_in_flight_action_is_cancelled_on_revoke():
    def revoke_mid_action(action, kw):
        caps.revoke(caps.BROWSER)  # lands while the action is running

    backend = FakeBackend(on_call=revoke_mid_action)
    with patch.object(cu_tool, "_new_backend", lambda permission_mode: backend):
        caps.grant(caps.BROWSER)
        out = _call("click", {"element": 7}, session_id="sess-inflight")
    assert "revoked" in out["error"]                      # result discarded, not returned as success
    assert out.get("ok") is not True
    assert backend.cancel_invoked is True                 # cooperative cancel hook used "where supported"
    last = _records()[-1]
    assert last["ok"] is False and last["code"] == "capability_revoked"


def test_malformed_capability_store_fails_closed(fake_backend):
    caps.store_path().write_text("{ this is not json", encoding="utf-8")
    out = _call("click", {"element": 1})
    assert out["ok"] is False and out["code"] == "capability_denied"
    assert "failed closed" in out["error"] or "malformed" in out["error"]
    assert fake_backend.calls == []
    assert caps.is_granted(caps.BROWSER)[0] is False


def test_enforcement_not_configured_keeps_legacy_behaviour(fake_backend):
    """No store, no switch: the tool is not gated (explicit reason, never a silent default-on)."""
    assert caps.enforcement()[0] is False
    out = _call("click", {"element": 1})
    assert out.get("ok") is True


# ---------------------------------------------------------------------------
# B. Activity audit
# ---------------------------------------------------------------------------

def test_audit_record_excludes_payload_text(fake_backend):
    caps.grant(caps.SHELL)
    out = _call("type", {"text": SENTINEL}, session_id="sess-audit")
    assert out.get("ok") is True
    assert fake_backend.calls[0][1]["text"] == SENTINEL  # the payload did reach the backend...
    blob = json.dumps(_records(), ensure_ascii=False)
    assert SENTINEL not in blob                          # ...but never the audit record
    shape = _records()[0]["shape"]
    assert shape["text_chars"] == len(SENTINEL)
    assert "text" not in shape


def test_audit_failure_is_loud_not_silent(monkeypatch):
    cu_audit.activity_path().mkdir(parents=True, exist_ok=True)  # a directory where the file must go
    with pytest.raises(cu_audit.AuditWriteError):
        cu_audit.record("click", "sess", ok=True)


def test_retention_cap_trims_oldest_first(monkeypatch):
    monkeypatch.setenv(cu_audit._RETENTION_ENV, "3")
    for name in ("a1", "a2", "a3", "a4", "a5"):
        cu_audit.record(name, "sess", ok=True)
    records = _records()
    assert [r["action"] for r in records] == ["a3", "a4", "a5"]
