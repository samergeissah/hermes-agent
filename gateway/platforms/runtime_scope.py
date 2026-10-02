"""Runtime route scoping for the app-facing API-server surface.

Ported pattern from OpenDots' ``src/server/runtime-scope.ts``: the app-facing
runtime routes are an explicit **allowlist** keyed on ``(http method, agent id,
thread id)`` rather than "anything the router happens to match". Two failure
modes this closes:

* an unknown route under the runtime prefix (the framework router would 404 or,
  worse, a catch-all would accept it), and
* SDK-reserved path suffixes (``clear``, ``info``, ``transcribe``, ``annotate``,
  ``inspector-metadata``, ...) that a router resolves *before* the
  ``{thread_id}`` parameter, so accepting them as a thread id would validate a
  different target than the one dispatched.

The matching rule mirrors the reference: match the whole (profile-stripped) path,
never a suffix/prefix, and validate the captured agent/thread against a registry.
A denial is always an explicit refusal — :func:`make_runtime_scope_middleware`
returns ``403`` with a machine-readable reason, never a silent passthrough.

This module deliberately imports only the standard library at module scope; the
aiohttp dependency is confined to :func:`make_runtime_scope_middleware`, which is
only ever called from ``connect()`` (itself aiohttp-only).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Set, Tuple

__all__ = [
    "AGENT_ID_HEADER",
    "ANY_THREAD",
    "DEFAULT_AGENT_ID",
    "RESERVED_SUFFIXES",
    "RUNTIME_ROUTES",
    "AgentRegistry",
    "RouteRule",
    "RuntimeScopeDenied",
    "RuntimeTarget",
    "active_registry",
    "install_registry",
    "make_runtime_scope_middleware",
    "parse_runtime_target",
    "reset_registry",
    "runtime_scope_app_key",
    "set_active_registry",
    "validate_runtime_scope",
]

# Agent id is declared by the client; the default listener serves one agent.
AGENT_ID_HEADER = "X-Hermes-Agent"
DEFAULT_AGENT_ID = "default"

#: Registry marker for "this agent is backed by the session store", i.e. thread
#: ownership is resolved by the downstream session lookup (which 404s unknown
#: ids). Without the marker, ownership is an explicit set membership test.
ANY_THREAD = "*"

#: Path suffixes the SDK router resolves ahead of the ``{thread_id}`` param.
#: Accepting one as a thread id would scope a different target than dispatched.
RESERVED_SUFFIXES = frozenset({
    "clear",
    "info",
    "transcribe",
    "annotate",
    "inspector-metadata",
    "inspector-learning",
    "cpk-debug-events",
})

#: The runtime prefix this gate governs. Requests outside it are untouched —
#: the gate scopes the app-facing runtime surface, it is not a second router.
RUNTIME_PREFIX = "/api/sessions"

#: A thread id is a single opaque path segment: no separators, no whitespace, no
#: control characters.
_ILLEGAL_THREAD_CHARS = re.compile(r"[/\\\s\x00-\x1f]")
_PROFILE_PREFIX = re.compile(r"^/p/[^/]+(?=/|$)")


@dataclass(frozen=True)
class RouteRule:
    """One allowlist entry: an exact method plus a full-path pattern.

    ``thread_group`` names the regex group carrying the thread id (``""`` when
    the route addresses the collection, not a thread).
    """

    method: str
    pattern: "re.Pattern[str]"
    thread_group: str = ""


def _rule(method: str, path_regex: str, thread_group: str = "") -> RouteRule:
    return RouteRule(method, re.compile(path_regex), thread_group)


#: The app-facing runtime allowlist. Every route the app may reach under
#: :data:`RUNTIME_PREFIX`; anything else under the prefix is denied.
RUNTIME_ROUTES: Tuple[RouteRule, ...] = (
    _rule("GET", r"^/api/sessions$"),
    _rule("POST", r"^/api/sessions$"),
    _rule("GET", r"^/api/sessions/(?P<thread>[^/]+)$", "thread"),
    _rule("PATCH", r"^/api/sessions/(?P<thread>[^/]+)$", "thread"),
    _rule("DELETE", r"^/api/sessions/(?P<thread>[^/]+)$", "thread"),
    _rule("GET", r"^/api/sessions/(?P<thread>[^/]+)/messages$", "thread"),
    _rule("POST", r"^/api/sessions/(?P<thread>[^/]+)/fork$", "thread"),
    _rule("POST", r"^/api/sessions/(?P<thread>[^/]+)/chat$", "thread"),
    _rule("POST", r"^/api/sessions/(?P<thread>[^/]+)/chat/stream$", "thread"),
    _rule("POST", r"^/api/sessions/(?P<thread>[^/]+)/model$", "thread"),
)


class RuntimeScopeDenied(Exception):
    """A runtime route was refused. ``reason`` is safe to return to the client."""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


@dataclass
class RuntimeTarget:
    """The validated ``(method, agent, thread)`` triple a request resolved to."""

    method: str
    agent_id: str
    thread_id: str = ""
    rule: Optional[RouteRule] = None


@dataclass
class AgentRegistry:
    """Known agents and the threads each owns — the analogue of OpenDots' workspace store.

    An agent mapped to ``{ANY_THREAD}`` defers thread existence to the downstream
    session lookup; an explicit set means only those thread ids are reachable.
    """

    agents: Dict[str, Set[str]] = field(default_factory=dict)

    def is_known(self, agent_id: str) -> bool:
        return agent_id in self.agents

    def owns_thread(self, agent_id: str, thread_id: str) -> bool:
        owned = self.agents.get(agent_id)
        if owned is None:
            return False
        return ANY_THREAD in owned or thread_id in owned


#: Default registry: the single default agent, backed by the session store. It
#: is intentionally explicit — an agent id the operator never registered is
#: refused, matching the reference's ``workspace.dot`` check.
_ACTIVE_REGISTRY = AgentRegistry({DEFAULT_AGENT_ID: {ANY_THREAD}})


def active_registry() -> AgentRegistry:
    return _ACTIVE_REGISTRY


def set_active_registry(registry: AgentRegistry) -> None:
    global _ACTIVE_REGISTRY
    _ACTIVE_REGISTRY = registry


def reset_registry() -> None:
    """Restore the module default (test helper and process-restart hygiene)."""
    set_active_registry(AgentRegistry({DEFAULT_AGENT_ID: {ANY_THREAD}}))


def _strip_profile_prefix(path: str) -> str:
    """Drop a leading ``/p/<profile>`` so the multiplex mirror is scoped too.

    Without this, ``/p/<other>/api/sessions/<id>`` would bypass the gate.
    """
    return _PROFILE_PREFIX.sub("", path)


def _is_guarded(path: str) -> bool:
    return path == RUNTIME_PREFIX or path.startswith(RUNTIME_PREFIX + "/")


def parse_runtime_target(method: str, path: str) -> RuntimeTarget:
    """Resolve ``(method, path)`` to a :class:`RuntimeTarget` or raise.

    Fails closed with :class:`RuntimeScopeDenied` for: paths outside the
    governed prefix are *not* this gate's business (returns ``None``-like via
    :func:`validate_runtime_scope`); unknown routes, method mismatches, illegal
    or reserved thread ids, and unknown agents are all denied.
    """
    target = _strip_profile_prefix(path)
    if not _is_guarded(target):
        raise RuntimeScopeDenied(f"not a governed runtime route: {target}")
    path_matches = [rule for rule in RUNTIME_ROUTES if rule.pattern.match(target)]
    if not path_matches:
        raise RuntimeScopeDenied(f"unknown runtime route: {method} {target}")
    rule = next((r for r in path_matches if r.method == method), None)
    if rule is None:
        allowed = ", ".join(sorted({r.method for r in path_matches}))
        raise RuntimeScopeDenied(
            f"method {method} is not allowed on {target} (allowed: {allowed})")
    thread_id = ""
    if rule.thread_group:
        match = rule.pattern.match(target)
        thread_id = (match.group(rule.thread_group) if match else "") or ""
        if not thread_id or _ILLEGAL_THREAD_CHARS.search(thread_id):
            raise RuntimeScopeDenied(f"invalid thread id in {target}")
        if thread_id in RESERVED_SUFFIXES:
            raise RuntimeScopeDenied(
                f"'{thread_id}' is an SDK-reserved path suffix, not a thread id")
    return RuntimeTarget(method=method, agent_id="", thread_id=thread_id, rule=rule)


def validate_runtime_scope(
    method: str,
    path: str,
    *,
    agent_id: Optional[str] = None,
    registry: Optional[AgentRegistry] = None,
) -> Optional[RuntimeTarget]:
    """Validate a request against the ``(method, agent, thread)`` allowlist.

    Returns ``None`` when *path* is outside :data:`RUNTIME_PREFIX` (not governed),
    else the validated :class:`RuntimeTarget`. Every refusal raises
    :class:`RuntimeScopeDenied` carrying a reason string.
    """
    stripped = _strip_profile_prefix(path)
    if not _is_guarded(stripped):
        return None
    target = parse_runtime_target(method, path)
    resolved_agent = agent_id or DEFAULT_AGENT_ID
    reg = registry if registry is not None else active_registry()
    if not reg.is_known(resolved_agent):
        raise RuntimeScopeDenied(f"unknown agent id: {resolved_agent}")
    if target.thread_id and not reg.owns_thread(resolved_agent, target.thread_id):
        raise RuntimeScopeDenied(
            f"thread '{target.thread_id}' is not owned by agent '{resolved_agent}'")
    target.agent_id = resolved_agent
    return target


_APP_KEY = None


def runtime_scope_app_key():
    """The aiohttp ``AppKey`` carrying the registry (created on first use)."""
    global _APP_KEY
    if _APP_KEY is None:
        from aiohttp import web

        _APP_KEY = web.AppKey("runtime_scope_registry", AgentRegistry)
    return _APP_KEY


def make_runtime_scope_middleware(adapter=None):
    """Build the aiohttp middleware enforcing the runtime allowlist.

    The known-agent registry is read from the app via :func:`runtime_scope_app_key`
    when present, else the module-level active registry. The agent id is the
    ``X-Hermes-Agent`` header when supplied, else :data:`DEFAULT_AGENT_ID`.
    """
    from aiohttp import web  # local: keeps this module stdlib-only at import time

    @web.middleware
    async def runtime_scope_middleware(request, handler):
        registry = request.app.get(runtime_scope_app_key())
        try:
            validate_runtime_scope(
                request.method,
                request.path,
                agent_id=request.headers.get(AGENT_ID_HEADER) or None,
                registry=registry,
            )
        except RuntimeScopeDenied as exc:
            return web.json_response(
                {"error": {"message": exc.reason, "type": "runtime_scope_error",
                           "code": "runtime_scope_denied"}},
                status=403)
        return await handler(request)

    return runtime_scope_middleware


def install_registry(app, registry: AgentRegistry) -> None:
    """Attach *registry* to an aiohttp app for :func:`make_runtime_scope_middleware`."""
    app[runtime_scope_app_key()] = registry


#: App-state key name under which an app carries its runtime-scope registry.
RUNTIME_SCOPE_APP_KEY = "runtime_scope_registry"
