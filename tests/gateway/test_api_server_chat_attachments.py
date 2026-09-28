"""Focused tests for ``POST /api/sessions/{id}/chat/attachments`` and the image-input routing
every api_server lane now honours.

Behaviour under test — not snapshots:

* the route the Eve Android client posts to exists, is advertised, and sits behind the same
  Bearer auth as the rest of the ``/api`` surface;
* a multipart image upload runs ONE real session turn carrying an image part and answers with
  the same completion envelope ``/chat`` answers with;
* the upload is cached to disk (the inbound image cache) and the turn's caption carries that
  absolute path;
* the same lane accepts DOCUMENTS — the file source both Eve clients offer — persisting them to
  the documents cache beside the images and handing the turn their absolute path as a text turn
  (no inline part, no vision backend: a document is read with tools, not looked at);
* the document types accepted are the ones the two clients really send, including the
  ``application/octet-stream`` declaration iOS and Android fall back to for a file they cannot
  name (office documents land there);
* a non-image, non-document upload / missing file is refused before any turn runs;
* the shared image-input decision is consulted for BOTH the upload lane and the pre-existing
  inline-image lane, and BOTH of its branches are honoured: ``native`` keeps the pixels on the
  main model, ``text`` keeps them off it.
"""

from __future__ import annotations

import base64
import hashlib
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
PDF_BYTES = (b"%PDF-1.4\n1 0 obj<</Type/Catalog>>endobj\ntrailer<</Root 1 0 R>>\n"
             b"%%EOF\n")
TEXT_BYTES = b"# Notes\n\n- one\n- two\n"
# Not a real .docx (nothing here parses office formats); the bytes only have to round-trip.
DOCX_BYTES = b"PK\x03\x04" + b"\x00" * 24 + b"word/document.xml"

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
def adapter(session_db, tmp_path, monkeypatch):
    """Authenticated adapter: the client contract always sends a session-key header, which
    requires a configured API key, so the upload lane is exercised the way the app posts it.

    Both inbound caches are redirected into ``tmp_path`` (the same seam the voice-note suite uses
    for the audio cache) so a test run never litters the real profile cache — images AND the
    documents cache the widened lane writes to.
    """
    monkeypatch.setattr("gateway.platforms.base.IMAGE_CACHE_DIR", str(tmp_path / "images"))
    monkeypatch.setattr("gateway.platforms.base.DOCUMENT_CACHE_DIR", str(tmp_path / "documents"))
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


def _turn_text(message) -> str:
    """The text a turn was handed, whichever shape the lane used (plain string or text parts)."""
    if isinstance(message, str):
        return message
    return "\n".join(
        str(part.get("text") or "") for part in message if isinstance(part, dict))


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
async def test_upload_leaves_the_image_on_disk(adapter, session_id, tmp_path):
    """The bytes land in the inbound image cache, and the turn is handed the path.

    The file on disk is the point: it is what a later turn (or an operator) can re-read after the
    inline ``data:`` part is gone from the wire, and it is the same cache every messaging lane's
    inbound image goes through.
    """
    agent = _CapturingAgent(session_id)
    with patch.object(adapter, "_create_agent", return_value=agent), patch(
            "agent.image_routing.decide_image_input_mode", return_value="native"):
        async with _client(adapter) as cli:
            resp = await cli.post(
                f"/api/sessions/{session_id}/chat/attachments",
                data=_upload_form(filename="throwaway-probe.png"),
                headers=_headers(session_id))
            assert resp.status == 200, await resp.text()

    stored = sorted((tmp_path / "images").glob("img_*.png"))
    assert len(stored) == 1, [p.name for p in (tmp_path / "images").iterdir()]
    assert stored[0].read_bytes() == PNG_BYTES, "the cached file must be the uploaded bytes"

    text = _turn_text(agent.calls[0]["user_message"])
    assert f"[Image attached at: {stored[0]}]" in text
    assert "throwaway-probe.png" in text, "the client's filename is still the human label"


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


# --------------------------------------------------------------------------- document lane


@pytest.mark.asyncio
async def test_pdf_upload_runs_a_turn_carrying_the_persisted_path(adapter, session_id, tmp_path):
    """A PDF is accepted, runs ONE real turn, and reaches it with its path on disk.

    Before this lane accepted documents, a PDF from the file picker got a 415 — the same picker
    path the two Eve clients already expose. The turn must be a plain text turn (a document is
    read with tools, not looked at inline) and the persisted file must be the uploaded bytes: it
    is what the agent's own file tools will open.
    """
    agent = _CapturingAgent(session_id)
    with patch.object(adapter, "_create_agent", return_value=agent), patch(
            "agent.image_routing.decide_image_input_mode") as decide:
        async with _client(adapter) as cli:
            resp = await cli.post(
                f"/api/sessions/{session_id}/chat/attachments",
                data=_upload_form(message="Summarise this.", mime="application/pdf",
                                  data=PDF_BYTES, filename="quarterly-report.pdf"),
                headers=_headers(session_id))
            assert resp.status == 200, await resp.text()
            payload = await resp.json()

    # The envelope is the image lane's, byte for byte in shape: the clients parse it identically.
    assert payload["object"] == "hermes.session.chat.completion"
    assert payload["session_id"] == session_id
    assert payload["message"]["role"] == "assistant"
    assert payload["message"]["content"] == FORWARD

    stored = sorted((tmp_path / "documents").glob("doc_*quarterly-report.pdf"))
    assert len(stored) == 1, [p.name for p in (tmp_path / "documents").iterdir()]
    assert stored[0].read_bytes() == PDF_BYTES, "the cached file must be the uploaded bytes"
    assert hashlib.sha256(stored[0].read_bytes()).hexdigest() == hashlib.sha256(PDF_BYTES).hexdigest()

    message = agent.calls[0]["user_message"]
    assert isinstance(message, str), "a document turn is text-only, never an inline part list"
    assert "Summarise this." in message
    assert "[Document attached: quarterly-report.pdf (application/pdf)]" in message
    assert f"[Document attached at: {stored[0]}]" in message
    assert not decide.called, "a document must not pay for the image-routing decision"
    assert not (tmp_path / "images").exists() or not list((tmp_path / "images").iterdir()), \
        "a document must not be written into the image cache"


@pytest.mark.asyncio
async def test_text_upload_without_a_prompt_gets_a_document_caption(adapter, session_id, tmp_path):
    agent = _CapturingAgent(session_id)
    with patch.object(adapter, "_create_agent", return_value=agent), patch(
            "agent.image_routing.decide_image_input_mode"):
        async with _client(adapter) as cli:
            resp = await cli.post(
                f"/api/sessions/{session_id}/chat/attachments",
                data=_upload_form(message="", mime="text/plain", data=TEXT_BYTES,
                                  filename="notes.md"),
                headers=_headers(session_id))
            assert resp.status == 200, await resp.text()

    message = agent.calls[0]["user_message"]
    assert "What's in this document?" in message
    stored = sorted((tmp_path / "documents").glob("doc_*notes.md"))
    assert len(stored) == 1
    assert stored[0].read_bytes() == TEXT_BYTES


# The exact (declared MIME, filename) pairs the two clients put on the wire:
#   * iOS  — ``ChatAttachment.mimeType(forPathExtension:)`` (EveIOS/ChatAttachments.swift);
#   * Android — the picker's own type, else ``application/octet-stream``
#     (``attachAndSend`` in EveApp.kt + ``sendAttachmentWithPrompt`` in EveViewModel.kt).
# Both send ``message`` / ``filename`` / ``mime_type`` / ``file``, so one table covers both.
_CLIENT_DOCUMENT_UPLOADS = [
    ("application/pdf", "report.pdf"),                      # both clients
    ("text/plain", "notes.txt"),                            # iOS + Android
    ("text/plain", "notes.md"),                             # iOS maps .md/.log -> text/plain
    ("text/markdown", "readme.md"),                         # Android picker
    ("text/csv", "rows.csv"),                               # iOS + Android
    ("application/json", "payload.json"),                   # iOS + Android
    ("application/vnd.openxmlformats-officedocument.wordprocessingml.document", "memo.docx"),
    ("application/octet-stream", "memo.docx"),              # iOS: no case for .docx -> octet-stream
    ("application/octet-stream", "sheet.xlsx"),             # Android: ContentResolver gave nothing
    ("application/octet-stream", "deck.pptx"),
    ("application/octet-stream", "contract.rtf"),
]


@pytest.mark.asyncio
@pytest.mark.parametrize("mime,filename", _CLIENT_DOCUMENT_UPLOADS)
async def test_the_types_the_two_clients_send_are_accepted(adapter, session_id, tmp_path,
                                                           mime, filename):
    """Every declaration the iOS/Android pickers can produce for a document is accepted.

    ``application/octet-stream`` is the important row: it is what BOTH clients send for a file they
    have no MIME case for, and the only remaining evidence is the bytes' signature or the name.
    """
    agent = _CapturingAgent(session_id)
    payload_bytes = PDF_BYTES if filename.endswith(".pdf") else (
        DOCX_BYTES if filename.endswith((".docx", ".xlsx", ".pptx")) else TEXT_BYTES)
    with patch.object(adapter, "_create_agent", return_value=agent), patch(
            "agent.image_routing.decide_image_input_mode"):
        async with _client(adapter) as cli:
            resp = await cli.post(
                f"/api/sessions/{session_id}/chat/attachments",
                data=_upload_form(message="read this", mime=mime, data=payload_bytes,
                                  filename=filename),
                headers=_headers(session_id))
            body = await resp.text()
            assert resp.status == 200, f"{mime} / {filename}: {body}"
            assert (await resp.json())["object"] == "hermes.session.chat.completion"

    stored = sorted((tmp_path / "documents").glob(f"doc_*{filename}"))
    assert len(stored) == 1, [p.name for p in (tmp_path / "documents").iterdir()]
    assert stored[0].read_bytes() == payload_bytes
    assert f"[Document attached at: {stored[0]}]" in agent.calls[0]["user_message"]


@pytest.mark.asyncio
async def test_unsupported_attachment_type_is_refused_without_running_a_turn(adapter, session_id):
    """A type the lane has no reading for is still refused — with the accepted set named."""
    with patch.object(adapter, "_create_agent") as create:
        async with _client(adapter) as cli:
            resp = await cli.post(
                f"/api/sessions/{session_id}/chat/attachments",
                data=_upload_form(mime="application/x-msdownload", data=b"MZ\x90\x00",
                                  filename="installer.exe"),
                headers=_headers(session_id))
            body = await resp.json()

    assert resp.status == 415
    assert body["error"]["code"] == "unsupported_media_type"
    assert "application/x-msdownload" in body["error"]["message"]
    assert "this endpoint accepts images" in body["error"]["message"]
    assert "pdf" in body["error"]["message"], "the refusal names what IS accepted"
    assert "voice-note" in body["error"]["message"], "audio is pointed at the route that takes it"
    assert not create.called


# Types another lane owns (audio) or nothing here handles: still a 415, never a silent accept —
# a declared type that is neither opaque nor accepted is honoured as a refusal, so a filename can
# never smuggle a payload into an extension it could not be opened with.
_REFUSED_UPLOADS = [
    ("application/x-msdownload", "installer.exe", b"MZ\x90\x00"),
    ("audio/mpeg", "clip.mp3", b"ID3\x04\x00\x00"),
    ("video/mp4", "clip.mp4", b"\x00\x00\x00\x18ftypmp42"),
    ("application/x-nonsense", "payload.xyz", b"\x7fELF-not-a-document"),
]


@pytest.mark.asyncio
@pytest.mark.parametrize("mime,filename,data", _REFUSED_UPLOADS)
async def test_unhandled_types_are_refused(adapter, session_id, mime, filename, data):
    with patch.object(adapter, "_create_agent") as create:
        async with _client(adapter) as cli:
            resp = await cli.post(
                f"/api/sessions/{session_id}/chat/attachments",
                data=_upload_form(mime=mime, data=data, filename=filename),
                headers=_headers(session_id))
            body = await resp.json()

    assert resp.status == 415, f"{mime} / {filename} unexpectedly accepted"
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
