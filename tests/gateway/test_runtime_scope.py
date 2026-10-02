"""Tests for the runtime route-scoping gate (``gateway.platforms.runtime_scope``).

Ported-pattern coverage: the ``(method, agent, thread)`` allowlist, unknown-route
denial, SDK-reserved suffix denial, unknown-agent denial, method mismatch, and the
aiohttp middleware returning an explicit 403 refusal (never a silent passthrough).
"""

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.platforms.runtime_scope import (
    AGENT_ID_HEADER,
    DEFAULT_AGENT_ID,
    RESERVED_SUFFIXES,
    RUNTIME_ROUTES,
    AgentRegistry,
    RuntimeScopeDenied,
    install_registry,
    make_runtime_scope_middleware,
    reset_registry,
    validate_runtime_scope,
)


# ---------------------------------------------------------------------------
# Module-level allowlist behaviour
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clean_registry():
    reset_registry()
    yield
    reset_registry()


class TestAllowlist:
    def test_allowed_route_passes(self):
        target = validate_runtime_scope("GET", "/api/sessions/session-1")
        assert target is not None
        assert target.method == "GET"
        assert target.thread_id == "session-1"
        assert target.agent_id == DEFAULT_AGENT_ID

    @pytest.mark.parametrize(
        "method,path",
        [
            ("GET", "/api/sessions"),
            ("POST", "/api/sessions"),
            ("GET", "/api/sessions/abc"),
            ("PATCH", "/api/sessions/abc"),
            ("DELETE", "/api/sessions/abc"),
            ("GET", "/api/sessions/abc/messages"),
            ("POST", "/api/sessions/abc/fork"),
            ("POST", "/api/sessions/abc/chat"),
            ("POST", "/api/sessions/abc/chat/stream"),
            ("POST", "/api/sessions/abc/model"),
        ],
    )
    def test_every_allowlisted_route_row_passes(self, method, path):
        assert validate_runtime_scope(method, path) is not None

    def test_non_runtime_path_is_not_governed(self):
        assert validate_runtime_scope("GET", "/v1/models") is None
        assert validate_runtime_scope("GET", "/health") is None

    def test_profile_prefix_mirror_is_scoped_too(self):
        # /p/<profile> mirrors are native routes; they must not bypass the gate.
        assert validate_runtime_scope("GET", "/p/work/api/sessions/abc") is not None
        with pytest.raises(RuntimeScopeDenied):
            validate_runtime_scope("GET", "/p/work/api/sessions/clear")


class TestAllowlistParity:
    def test_allowlist_matches_the_api_server_route_table(self):
        """Drift guard: every /api/sessions row registered by the API server is
        allowed here, and the allowlist invents nothing the server does not serve."""
        from gateway.config import PlatformConfig
        from gateway.platforms.api_server import APIServerAdapter

        adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={"key": "x" * 24}))
        server_rows = {
            (method, path.replace("{session_id}", "abc"))
            for method, path, _ in adapter._http_route_table()
            if path == "/api/sessions" or path.startswith("/api/sessions/")
        }
        allowed = set()
        for rule in RUNTIME_ROUTES:
            # Reconstruct a representative concrete path from each rule regex.
            sample = rule.pattern.pattern.strip("^$").replace("(?P<thread>[^/]+)", "abc")
            allowed.add((rule.method, sample))
        assert server_rows == allowed


class TestDenials:
    def test_unknown_route_denied(self):
        with pytest.raises(RuntimeScopeDenied) as exc:
            validate_runtime_scope("GET", "/api/sessions/abc/export")
        assert "unknown runtime route" in exc.value.reason

    def test_unknown_collection_suffix_denied(self):
        with pytest.raises(RuntimeScopeDenied) as exc:
            validate_runtime_scope("POST", "/api/sessions/abc/run")
        assert "unknown runtime route" in exc.value.reason

    @pytest.mark.parametrize("suffix", sorted(RESERVED_SUFFIXES))
    def test_reserved_suffix_denied(self, suffix):
        with pytest.raises(RuntimeScopeDenied) as exc:
            validate_runtime_scope("GET", f"/api/sessions/{suffix}")
        assert "reserved" in exc.value.reason

    def test_reserved_suffix_denied_before_registry_lookup(self):
        # Even a registry that "owns" the reserved word must not open the route.
        registry = AgentRegistry({DEFAULT_AGENT_ID: {"clear"}})
        with pytest.raises(RuntimeScopeDenied) as exc:
            validate_runtime_scope("GET", "/api/sessions/clear", registry=registry)
        assert "reserved" in exc.value.reason

    def test_method_mismatch_denied(self):
        with pytest.raises(RuntimeScopeDenied) as exc:
            validate_runtime_scope("GET", "/api/sessions/abc/chat")
        assert "not allowed" in exc.value.reason
        with pytest.raises(RuntimeScopeDenied):
            validate_runtime_scope("POST", "/api/sessions/abc/messages")

    def test_unknown_agent_denied(self):
        with pytest.raises(RuntimeScopeDenied) as exc:
            validate_runtime_scope("GET", "/api/sessions/abc", agent_id="ghost")
        assert "unknown agent id" in exc.value.reason

    def test_thread_not_owned_by_agent_denied(self):
        registry = AgentRegistry({"alpha": {"thread-1"}})
        assert validate_runtime_scope(
            "GET", "/api/sessions/thread-1", agent_id="alpha", registry=registry)
        with pytest.raises(RuntimeScopeDenied) as exc:
            validate_runtime_scope(
                "GET", "/api/sessions/thread-2", agent_id="alpha", registry=registry)
        assert "not owned by agent" in exc.value.reason

    def test_illegal_thread_id_denied(self):
        # A control character in the segment is not a valid thread id.
        with pytest.raises(RuntimeScopeDenied):
            validate_runtime_scope("GET", "/api/sessions/abc\tdef")
        with pytest.raises(RuntimeScopeDenied):
            validate_runtime_scope("GET", "/api/sessions/abc\x00def")
        # A bare trailing separator leaves an empty thread id -> unknown route.
        with pytest.raises(RuntimeScopeDenied):
            validate_runtime_scope("GET", "/api/sessions/")


# ---------------------------------------------------------------------------
# Middleware integration: denials must be explicit 403 refusals
# ---------------------------------------------------------------------------


def _build_app(registry=None):
    app = web.Application(middlewares=[make_runtime_scope_middleware()])
    if registry is not None:
        install_registry(app, registry)

    async def _ok(request):
        return web.json_response({"ok": True})

    for method, path in (
        ("GET", "/api/sessions"),
        ("GET", "/api/sessions/{session_id}"),
        ("POST", "/api/sessions/{session_id}/chat"),
    ):
        app.router.add_route(method, path, _ok)
    return app


class TestMiddleware:
    @pytest.mark.asyncio
    async def test_allowed_request_passes(self):
        async with TestClient(TestServer(_build_app())) as client:
            resp = await client.get("/api/sessions/abc")
            assert resp.status == 200
            assert (await resp.json())["ok"] is True

    @pytest.mark.asyncio
    async def test_unknown_route_returns_explicit_403(self):
        async with TestClient(TestServer(_build_app())) as client:
            resp = await client.get("/api/sessions/abc/export")
            assert resp.status == 403
            body = await resp.json()
            assert body["error"]["code"] == "runtime_scope_denied"
            assert "unknown runtime route" in body["error"]["message"]

    @pytest.mark.asyncio
    async def test_reserved_suffix_returns_explicit_403(self):
        async with TestClient(TestServer(_build_app())) as client:
            resp = await client.get("/api/sessions/info")
            assert resp.status == 403
            body = await resp.json()
            assert body["error"]["code"] == "runtime_scope_denied"
            assert "reserved" in body["error"]["message"]

    @pytest.mark.asyncio
    async def test_method_mismatch_returns_explicit_403(self):
        async with TestClient(TestServer(_build_app())) as client:
            resp = await client.get("/api/sessions/abc/chat")
            assert resp.status == 403
            body = await resp.json()
            assert "not allowed" in body["error"]["message"]

    @pytest.mark.asyncio
    async def test_unknown_agent_header_returns_explicit_403(self):
        async with TestClient(TestServer(_build_app())) as client:
            resp = await client.get(
                "/api/sessions/abc", headers={AGENT_ID_HEADER: "ghost"})
            assert resp.status == 403
            body = await resp.json()
            assert "unknown agent id" in body["error"]["message"]

    @pytest.mark.asyncio
    async def test_app_registry_scopes_thread_ownership(self):
        app = _build_app(registry=AgentRegistry({"alpha": {"thread-1"}}))
        async with TestClient(TestServer(app)) as client:
            ok = await client.get(
                "/api/sessions/thread-1", headers={AGENT_ID_HEADER: "alpha"})
            assert ok.status == 200
            denied = await client.get(
                "/api/sessions/thread-2", headers={AGENT_ID_HEADER: "alpha"})
            assert denied.status == 403
            assert "not owned by agent" in (await denied.json())["error"]["message"]

    @pytest.mark.asyncio
    async def test_real_route_table_refuses_unknown_runtime_route(self):
        """The middleware runs ahead of the handlers/auth on the real route table."""
        from gateway.config import PlatformConfig
        from gateway.platforms.api_server import APIServerAdapter

        adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={"key": "x" * 24}))
        app = web.Application(middlewares=[make_runtime_scope_middleware()])
        for method, path, handler in adapter._http_route_table():
            app.router.add_route(method, path, handler)
        async with TestClient(TestServer(app)) as client:
            denied = await client.get("/api/sessions/abc/export")
            assert denied.status == 403
            assert (await denied.json())["error"]["code"] == "runtime_scope_denied"
            # A real allowlisted route reaches the handler (auth then refuses: 401).
            allowed = await client.get("/api/sessions/abc")
            assert allowed.status == 401
