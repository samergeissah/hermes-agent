"""Focused tests for the API-server voice-note upload endpoint (GEI-129).

Contract under test: POST /api/sessions/{session_id}/voice-note accepts a multipart
form (mode / filename / mime_type / audio) with an X-Hermes-Session-Key header and
returns 2xx, storing the blob in the gateway audio cache (same primitive inbound
platform voice notes use).
"""

import struct

import pytest
from aiohttp import FormData, web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import PlatformConfig
from gateway.platforms.api_server import APIServerAdapter
from hermes_state import SessionDB


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
    monkeypatch.setattr("gateway.platforms.base.AUDIO_CACHE_DIR", str(tmp_path / "audio"))
    adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={"key": "sk-test"}))
    adapter._session_db = session_db
    return adapter


def _voice_note_app(adapter: APIServerAdapter) -> web.Application:
    app = web.Application()
    app.router.add_post("/api/sessions/{session_id}/voice-note", adapter._handle_session_voice_note)
    return app


def _wav_bytes() -> bytes:
    """A minimal valid 8 kHz 16-bit mono WAV (0.1s of silence)."""
    num_samples = 800
    data = b"\x00\x00" * num_samples
    header = b"RIFF" + struct.pack("<I", 36 + len(data)) + b"WAVE"
    header += b"fmt " + struct.pack("<IHHIIHH", 16, 1, 1, 8000, 16000, 2, 16)
    header += b"data" + struct.pack("<I", len(data))
    return header + data


def _form() -> FormData:
    data = FormData()
    data.add_field("mode", "voice_note")
    data.add_field("filename", "probe.wav")
    data.add_field("mime_type", "audio/wav")
    data.add_field("audio", _wav_bytes(), filename="probe.wav", content_type="audio/wav")
    return data


@pytest.mark.asyncio
async def test_voice_note_upload_stores_blob(adapter, session_db, tmp_path):
    session_id = session_db.create_session("ios_eve_test", "api_server")
    app = _voice_note_app(adapter)
    async with TestClient(TestServer(app)) as cli:
        resp = await cli.post(
            f"/api/sessions/{session_id}/voice-note",
            data=_form(),
            headers={
                "Authorization": "Bearer sk-test",
                "X-Hermes-Session-Key": "ios:eve:ios_eve_test",
            },
        )
        assert resp.status == 201, await resp.text()
        body = await resp.json()

    assert body["object"] == "hermes.session.voice_note"
    assert body["session_id"] == session_id
    assert body["mode"] == "voice_note"
    assert body["filename"] == "probe.wav"
    assert body["mime_type"] == "audio/wav"
    assert body["bytes"] == len(_wav_bytes())
    stored = body["path"]
    assert stored.startswith(str(tmp_path / "audio")), stored
    assert (tmp_path / "audio").exists()


@pytest.mark.asyncio
async def test_voice_note_404_unknown_session(adapter):
    app = _voice_note_app(adapter)
    async with TestClient(TestServer(app)) as cli:
        resp = await cli.post(
            "/api/sessions/does-not-exist/voice-note",
            data=_form(),
            headers={"Authorization": "Bearer sk-test"},
        )
        assert resp.status == 404


@pytest.mark.asyncio
async def test_voice_note_400_missing_audio(adapter, session_db):
    session_id = session_db.create_session("ios_eve_test", "api_server")
    data = FormData()
    data.add_field("mode", "voice_note")
    data.add_field("filename", "probe.wav")
    data.add_field("mime_type", "audio/wav")
    app = _voice_note_app(adapter)
    async with TestClient(TestServer(app)) as cli:
        resp = await cli.post(
            f"/api/sessions/{session_id}/voice-note",
            data=data,
            headers={"Authorization": "Bearer sk-test"},
        )
        assert resp.status == 400


@pytest.mark.asyncio
async def test_voice_note_401_without_auth(adapter, session_db):
    session_id = session_db.create_session("ios_eve_test", "api_server")
    app = _voice_note_app(adapter)
    async with TestClient(TestServer(app)) as cli:
        resp = await cli.post(
            f"/api/sessions/{session_id}/voice-note",
            data=_form(),
        )
        assert resp.status == 401
