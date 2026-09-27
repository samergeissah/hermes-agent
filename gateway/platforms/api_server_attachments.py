"""``/api/sessions/{session_id}/chat/attachments`` — the multipart image-upload lane.

WHY it exists: the Eve Android app (``HermesGatewayClient.sendAttachment``) uploads a picked
image as ``multipart/form-data`` (``message``, ``filename``, ``mime_type``, ``file``) and reads
``content`` out of the JSON reply, but no such route was registered — the app got a 404 and the
image never reached the session. The upload is a REAL turn on the session, so the response is the
same envelope ``/chat`` returns and the pixels go through the same image lane as every other
api_server turn (``api_server_image_lane``, wired in ``api_server._run_agent``).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, NamedTuple, Optional, Tuple

try:
    from aiohttp import web
except ImportError:  # pragma: no cover - aiohttp is required to serve this platform at all
    web = None  # type: ignore[assignment,misc]

logger = logging.getLogger("gateway.platforms.api_server")

# A declared-type echo longer than this is noise (and client input).
_MAX_ECHOED_TYPE = 40


class _Upload(NamedTuple):
    """One accepted upload: the prompt text and the agent-ready content list."""

    message: str
    content: Any


def _declared_type(value: Any) -> str:
    """Lower-cased declared MIME from a form field/file part, trimmed of parameters."""
    raw = value if isinstance(value, str) else ""
    return raw.split(";", 1)[0].strip().lower()


def _accepted_image_mime(raw: bytes, declared: str) -> str:
    """The image MIME to label the upload with; ``""`` when it is not an image at all.

    Magic bytes are authoritative (``image_routing._sniff_mime_from_bytes`` — the one sniffer in
    the tree, shared with the messaging lanes: platforms lie about content-type and a mismatched
    ``media_type`` makes the provider reject the part). The declared type is honoured only when
    the sniffer does not recognise the bytes (exotic but legitimately declared formats) — but a
    payload that declares nothing image-ish and sniffs as nothing is refused.
    """
    from agent.image_routing import _sniff_mime_from_bytes

    sniffed = _sniff_mime_from_bytes(raw)
    if sniffed:
        return sniffed
    return declared if declared.startswith("image/") else ""


async def _read_upload(request: "web.Request", _api_server) -> Tuple[Optional[_Upload], Optional["web.Response"]]:
    """``(upload, None)`` or ``(None, error)`` for the declared multipart contract.

    Ladder: unusable multipart body → 400 ``invalid_multipart_body``, no ``file`` part → 400
    ``missing_file``, empty ``file`` → 400 ``empty_file``, a payload that is not an image →
    415 ``unsupported_media_type``. An oversize body is aiohttp's own 413 (``client_max_size``),
    so ``HTTPException`` propagates instead of being rewritten into a 400.
    """
    try:
        form = await request.post()
    except web.HTTPException:
        raise
    except Exception as exc:
        logger.warning("[api_server] attachment upload rejected: unreadable multipart body: %s", exc)
        return None, _api_server._error_response(
            "Request body must be multipart/form-data with 'message' and 'file' parts.",
            400, code="invalid_multipart_body")

    file_field = form.get("file")
    if file_field is None:
        return None, _api_server._error_response(
            "Missing 'file' part", 400, code="missing_file")
    raw = file_field.file.read() if hasattr(file_field, "file") else file_field
    if not isinstance(raw, (bytes, bytearray)):
        raw = str(raw).encode("utf-8")
    if not raw:
        return None, _api_server._error_response("Empty 'file' part", 400, code="empty_file")

    # ``mime_type`` is the app's own field; a file part's own Content-Type is the fallback for
    # plain multipart clients that omit it.
    declared = _declared_type(form.get("mime_type")) or _declared_type(getattr(file_field, "content_type", ""))
    mime = _accepted_image_mime(bytes(raw), declared)
    if not mime:
        shown = (declared or "unspecified")[:_MAX_ECHOED_TYPE]
        return None, _api_server._error_response(
            f"Unsupported attachment type {shown!r}: only image uploads are supported on this "
            "endpoint.", 415, code="unsupported_media_type")

    from gateway.platforms import api_server_image_lane as _image_lane

    filename = str(form.get("filename") or getattr(file_field, "filename", "") or "")
    return _Upload(
        message=str(form.get("message") or ""),
        content=_image_lane.build_upload_content(
            str(form.get("message") or ""), filename, mime, bytes(raw))), None


async def _handle_session_chat_attachment(self, request: "web.Request", *, _api_server) -> "web.Response":
    """POST /api/sessions/{session_id}/chat/attachments — one image upload, one session turn.

    Runs the turn exactly like ``/chat`` (same ``_prepare_session_chat`` validation, auth,
    session-key scoping, model-lock handling, retry policy and completion envelope), with the
    uploaded bytes as a native ``image_url`` part that ``_run_agent`` routes through the shared
    image lane.

    Deliberately does NOT hand the turn to a live Bot Chat owner (``/chat`` does): that mailbox
    carries text only, so a native-image turn handed over would be answered with no image behind
    it. The upload runs where the pixels are.
    """
    limited = self._concurrency_limited_response()
    if limited is not None:
        return limited
    upload, err = await _read_upload(request, _api_server)
    if err is not None:
        return err
    ctx, err = await self._prepare_session_chat(
        request, payload=({"message": upload.message}, upload.content))
    if err is not None:
        return err
    return await self._complete_session_chat_turn(ctx)


def _http_routes(self) -> list:
    """Route rows this module owns, for ``APIServerAdapter._http_route_table``."""
    return [("POST", "/api/sessions/{session_id}/chat/attachments", self._handle_session_chat_attachment)]
