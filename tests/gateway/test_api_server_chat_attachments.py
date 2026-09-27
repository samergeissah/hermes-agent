"""Focused tests for ``POST /api/sessions/{id}/chat/attachments`` and the image-input routing
every api_server lane now honours.

Behaviour under test — not snapshots:

* the route the Eve Android client posts to exists, is advertised, and sits behind the same
  Bearer auth as the rest of the ``/api`` surface;
* a multipart image upload runs ONE real session turn carrying an image part and answers with
  the same completion envelope ``/chat`` answers with;
* a non-image upload / missing file is refused before any turn runs;
* the shared image-input decision is consulted for BOTH the upload lane and the pre-existing
  inline-image lane, and BOTH of its branches are honoured: ``native`` keeps the pixels on the
  main model, ``text`` keeps them off it.
"""

from __future__ import annotations

import base64
import json
from unittest.mock import patch

import pytest
from aiohttp import FormData, web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import PlatformConfig
from gateway.platforms.api_server import APIServerAdapter
from hermes_state import SessionDB

API_KEY = "sk-test"
ATTACHMENTS_PATH = "/api/sessions/{session_id}/chat/attachments"
# Real 2x2 PNG bytes (valid enough for Pillow's verify(), which the vision path runs), so the
# upload lane's magic-byte sniffing and the vision backend both get a genuine image.
PNG_BYTES = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAIAAAACCAIAAAD91JpzAAAAEklEQVR4nGPkEpFjYGBgYgAD"
    "AALmAEAUQs4PAAAAAElFTkSuQmCC")
PDF_BYTES = b"%PDF-1.4\n% not an image at all\n"

FORWARD = "A 1x1 test image."


@pytest.fixture
def session_db(tmp_path):
    db = SessionDB(tmp_path / "state.db")
    try:
        yield db
    finally:
        close = getattr(db, "close", None)
        if callable(close):
            close()


@pytest.fixture
def adapter(session_db):
    """Authenticated adapter: the client contract always sends a session-key header, which
    requires a configured API key, so the upload lane is exercised the way the app posts it."""
    adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={"key": API_KEY}))
    adapter._session_db = session_db
    return adapter


@pytest.fixture
def session_id(session_db):
    return session_db.create_session("upload-session", "api_server")


def _client(adapter: APIServerAdapter) -> TestClient:
    """App wired from the adapter's OWN route table, so a test can never pass against a route
    that is not registered on the real server."""
    app = web.Application()
    app["api_server_adapter"] = adapter
    for method, path, handler in adapter._http_route_table():
        app.router.add_route(method, path, handler)
    return TestClient(TestServer(app))


def _headers(session_id: str) -> dict:
    return {
        "Authorization": f"Bearer {API_KEY}",
        "Accept": "application/json",
        "X-Hermes-Session-Key": f"android:default:{session_id}",
    }


def _upload_form(*, message: str = "What is this?", mime: str = "image/png",
                 data: bytes = PNG_BYTES, filename: str = "shot.png") -> FormData:
    form = FormData()
    form.add_field("message", message)
    form.add_field("filename", filename)
    form.add_field("mime_type", mime)
    form.add_field("file", data, filename=filename, content_type=mime)
    return form


class _CapturingAgent:
    """Minimal AIAgent stand-in: records exactly what ``run_conversation`` was handed."""

    session_prompt_tokens = 0
    session_completion_tokens = 0
    session_total_tokens = 0

    def __init__(self, session_id: str, *, provider: str = "main-provider", model: str = "main-model"):
        self.session_id = session_id
        self.provider = provider
        self.model = model
        self.requested_provider = "main-provider"
        self.calls: list[dict] = []

    def run_conversation(self, **kwargs):
        self.calls.append(kwargs)
        return {"final_response": FORWARD, "session_id": self.session_id}


def _data_url_parts(message):
    return [p for p in message if isinstance(p, dict) and p.get("type") == "image_url"]


# --------------------------------------------------------------------------- route contract


def test_attachments_route_is_registered(adapter):
    routes = {(method, path) for method, path, _ in adapter._http_route_table()}
    assert ("POST", ATTACHMENTS_PATH) in routes


@pytest.mark.asyncio
async def test_capabilities_advertise_the_attachments_route(adapter):
    async with _client(adapter) as cli:
        resp = await cli.get("/v1/capabilities", headers={"Authorization": f"Bearer {API_KEY}"})
        assert resp.status == 200
        data = await resp.json()

    assert data["endpoints"]["session_chat_attachments"]["method"] == "POST"
    assert data["endpoints"]["session_chat_attachments"]["path"] == ATTACHMENTS_PATH
    assert data["features"]["session_chat_attachments"] is True


@pytest.mark.asyncio
async def test_upload_requires_bearer_auth(adapter, session_id):
    async with _client(adapter) as cli:
        resp = await cli.post(
            f"/api/sessions/{session_id}/chat/attachments",
            data=_upload_form(), headers={"Accept": "application/json"})

    assert resp.status == 401


# --------------------------------------------------------------------------- upload lane


@pytest.mark.asyncio
async def test_upload_runs_a_session_turn_carrying_the_image(adapter, session_id):
    agent = _CapturingAgent(session_id)
    with patch.object(adapter, "_create_agent", return_value=agent), patch(
            "agent.image_routing.decide_image_input_mode", return_value="native") as decide:
        async with _client(adapter) as cli:
            resp = await cli.post(
                f"/api/sessions/{session_id}/chat/attachments",
                data=_upload_form(), headers=_headers(session_id))
            assert resp.status == 200, await resp.text()
            payload = await resp.json()

    # Same completion envelope /chat returns; the Android client reads ``content`` out of it.
    assert payload["object"] == "hermes.session.chat.completion"
    assert payload["session_id"] == session_id
    assert payload["message"]["role"] == "assistant"
    assert payload["message"]["content"] == FORWARD

    message = agent.calls[0]["user_message"]
    parts = _data_url_parts(message)
    assert len(parts) == 1, "the uploaded image must reach the turn as one image part"
    data_url = parts[0]["image_url"]["url"]
    assert data_url.startswith("data:image/png;base64,")
    assert base64.b64decode(data_url.split(",", 1)[1]) == PNG_BYTES
    assert any("What is this?" in str(p.get("text") or "") for p in message if isinstance(p, dict))

    # The decision is consulted with the identity of the agent that will actually run the turn.
    assert decide.call_args.args[:2] == ("main-provider", "main-model")


@pytest.mark.asyncio
async def test_upload_without_a_prompt_still_runs_the_turn(adapter, session_id):
    agent = _CapturingAgent(session_id)
    with patch.object(adapter, "_create_agent", return_value=agent), patch(
            "agent.image_routing.decide_image_input_mode", return_value="native"):
        async with _client(adapter) as cli:
            resp = await cli.post(
                f"/api/sessions/{session_id}/chat/attachments",
                data=_upload_form(message=""), headers=_headers(session_id))
            assert resp.status == 200, await resp.text()

    message = agent.calls[0]["user_message"]
    assert "What do you see in this image?" in "\n".join(
        str(p.get("text") or "") for p in message if isinstance(p, dict))


@pytest.mark.asyncio
async def test_upload_text_routing_keeps_pixels_off_the_main_model(adapter, session_id):
    agent = _CapturingAgent(session_id)
    analyzed: list[str] = []

    async def _fake_vision(*, image_url, user_prompt, **kwargs):
        analyzed.append(image_url)
        return json.dumps({"success": True, "analysis": "A single black pixel on white."})

    with patch.object(adapter, "_create_agent", return_value=agent), patch(
            "agent.image_routing.decide_image_input_mode", return_value="text"), patch(
            "tools.vision_tools.vision_analyze_tool", side_effect=_fake_vision):
        async with _client(adapter) as cli:
            resp = await cli.post(
                f"/api/sessions/{session_id}/chat/attachments",
                data=_upload_form(), headers=_headers(session_id))
            assert resp.status == 200, await resp.text()

    message = agent.calls[0]["user_message"]
    assert isinstance(message, str)
    assert "A single black pixel on white." in message
    assert "data:image/" not in message, "text routing must not leak the pixels to the main model"
    assert analyzed and analyzed[0].startswith("data:image/png;base64,"), "the vision backend got the bytes"


@pytest.mark.asyncio
async def test_non_image_upload_is_refused_without_running_a_turn(adapter, session_id):
    with patch.object(adapter, "_create_agent") as create:
        async with _client(adapter) as cli:
            resp = await cli.post(
                f"/api/sessions/{session_id}/chat/attachments",
                data=_upload_form(mime="application/pdf", data=PDF_BYTES, filename="doc.pdf"),
                headers=_headers(session_id))
            body = await resp.json()

    assert resp.status == 415
    assert body["error"]["code"] == "unsupported_media_type"
    assert not create.called


@pytest.mark.asyncio
async def test_missing_and_empty_file_parts_are_clear_400s(adapter, session_id):
    form = FormData()
    form.add_field("message", "no file here")
    empty = FormData()
    empty.add_field("message", "empty file")
    empty.add_field("filename", "shot.png")
    empty.add_field("mime_type", "image/png")
    empty.add_field("file", b"", filename="shot.png", content_type="image/png")

    async with _client(adapter) as cli:
        missing = await cli.post(
            f"/api/sessions/{session_id}/chat/attachments", data=form, headers=_headers(session_id))
        assert missing.status == 400
        missing_body = await missing.json()
        empty_resp = await cli.post(
            f"/api/sessions/{session_id}/chat/attachments", data=empty, headers=_headers(session_id))
        assert empty_resp.status == 400
        empty_body = await empty_resp.json()

    assert missing_body["error"]["code"] == "missing_file"
    assert empty_body["error"]["code"] == "empty_file"


# --------------------------------------------------------------------------- inline image lane


@pytest.mark.asyncio
async def test_inline_image_parts_are_routed_too(adapter, session_id):
    """The pre-existing inline lane (JSON content parts on /chat) must honour the same decision."""
    agent = _CapturingAgent(session_id)
    content = [
        {"type": "text", "text": "describe this"},
        {"type": "image_url", "image_url": {
            "url": f"data:image/png;base64,{base64.b64encode(PNG_BYTES).decode('ascii')}"}},
    ]

    async def _fake_vision(*, image_url, user_prompt, **kwargs):
        return json.dumps({"success": True, "analysis": "One tiny pixel."})

    with patch.object(adapter, "_create_agent", return_value=agent), patch(
            "agent.image_routing.decide_image_input_mode", return_value="text"), patch(
            "tools.vision_tools.vision_analyze_tool", side_effect=_fake_vision):
        async with _client(adapter) as cli:
            resp = await cli.post(
                f"/api/sessions/{session_id}/chat", json={"message": content}, headers=_headers(session_id))
            assert resp.status == 200, await resp.text()

    message = agent.calls[0]["user_message"]
    assert isinstance(message, str)
    assert "One tiny pixel." in message
    assert "data:image/" not in message


@pytest.mark.asyncio
async def test_inline_image_native_routing_keeps_the_parts(adapter, session_id):
    agent = _CapturingAgent(session_id)
    content = [
        {"type": "text", "text": "hi"},
        {"type": "image_url", "image_url": {"url": "https://example.com/cat.png"}},
    ]
    with patch.object(adapter, "_create_agent", return_value=agent), patch(
            "agent.image_routing.decide_image_input_mode", return_value="native"):
        async with _client(adapter) as cli:
            resp = await cli.post(
                f"/api/sessions/{session_id}/chat", json={"message": content}, headers=_headers(session_id))
            assert resp.status == 200, await resp.text()

    assert agent.calls[0]["user_message"] == content


@pytest.mark.asyncio
async def test_text_only_turn_does_not_consult_image_routing(adapter, session_id):
    agent = _CapturingAgent(session_id)
    with patch.object(adapter, "_create_agent", return_value=agent), patch(
            "agent.image_routing.decide_image_input_mode") as decide:
        async with _client(adapter) as cli:
            resp = await cli.post(
                f"/api/sessions/{session_id}/chat", json={"message": "hello"}, headers=_headers(session_id))
            assert resp.status == 200, await resp.text()

    assert agent.calls[0]["user_message"] == "hello"
    assert not decide.called, "a text-only turn must not pay for the routing decision"
