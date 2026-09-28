"""Telegram turns must honor an explicit ``platform_toolsets.telegram`` entry.

Regression guard for t_13eac455: a standalone telegram gateway resolved the explicit
``telegram: [connections, hermes-telegram, kanban]`` entry as if it were absent, stripping
``kanban`` from the agent's enabled toolsets (and therefore from its tool schemas). These
tests drive the same resolution seam a live telegram turn uses — ``GatewayTurnMixin.
_resolve_turn_toolsets`` — and pin that the explicit entry keeps ``kanban`` while the
implicit-composite fallback (no entry) excludes it.
"""
from __future__ import annotations


def _resolve_turn_toolsets(user_config, platform_key="telegram"):
    from gateway.run_turn import GatewayTurnMixin

    mixin = GatewayTurnMixin()

    class _Adapter:
        def toolsets_for_source(self, source):
            return None

    # The telegram adapter does not pin a per-source override; the base adapter returns None.
    mixin._delivery_adapter_for = lambda source: _Adapter()
    return mixin._resolve_turn_toolsets(user_config, None, platform_key)


def test_telegram_explicit_entry_keeps_kanban():
    """``platform_toolsets.telegram: [connections, hermes-telegram, kanban]`` resolves kanban."""
    enabled, disabled = _resolve_turn_toolsets({
        "platform_toolsets": {"telegram": ["connections", "hermes-telegram", "kanban"]},
    })
    assert "kanban" in enabled, enabled
    assert "kanban" not in (disabled or [])


def test_telegram_no_entry_fallback_excludes_kanban():
    """With no ``platform_toolsets.telegram`` entry the implicit composite fallback drops kanban."""
    enabled, _disabled = _resolve_turn_toolsets({})
    assert "kanban" not in enabled, enabled


def test_telegram_empty_entry_excludes_kanban():
    """An explicitly saved EMPTY list is authoritative and drops the default-off kanban toolset."""
    enabled, _disabled = _resolve_turn_toolsets({"platform_toolsets": {"telegram": []}})
    assert "kanban" not in enabled, enabled
