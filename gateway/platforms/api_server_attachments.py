"""``/api/sessions/{session_id}/chat/attachments`` — the multipart attachment lane.

WHY it exists: the Eve Android app (``HermesGatewayClient.sendAttachment``) uploads a picked
image as ``multipart/form-data`` (``message``, ``filename``, ``mime_type``, ``file``) and reads
``content`` out of the JSON reply, but no such route was registered — the app got a 404 and the
image never reached the session. The upload is a REAL turn on the session, so the response is the
same envelope ``/chat`` returns and the pixels go through the same image lane as every other
api_server turn (``api_server_image_lane``, wired in ``api_server._run_agent``).

Images AND documents (widened): the two Eve clients both offer a document source — iOS
``ChatAttachment.mimeType(forPathExtension:)`` sends ``application/pdf`` / ``text/plain`` for a
picked PDF or text file (and ``application/octet-stream`` for any extension it has no case for,
office documents included), Android sends the picker's own type or the same
``application/octet-stream`` fallback (``EveViewModel``). Refusing those with a 415 made the
document source a dead end on both clients, so this lane now accepts ordinary document types too:
an image still runs the image lane (inline ``image_url`` part), a document runs a plain text turn
whose caption names the file and the persisted path. The set of accepted document types is NOT
re-invented here — it is the gateway's own
``gateway.platforms.base.SUPPORTED_DOCUMENT_TYPES`` table (plus the MIME spellings clients really
send for the odf/rtf/epub documents ``base.MEDIA_DELIVERY_EXTS`` can hand back), so the upload side
and the delivery side cannot drift. Anything else still gets a 415 naming the accepted kinds.

Documents are READ at ingest (``api_server_document_lane``): the caption+path turn a document used
to run was only usable by a profile with file tools — a strict intake persona answered "I cannot
read PDFs" about a one-line PDF it was holding — so the upload now also carries the extracted text
as ``[Document contents (extracted, N chars): …]`` beside the path line, exactly the way the image
lane puts the vision backend's description in the turn. Any profile can then answer from its own
context with no tool call. Extraction is best-effort and BOUNDED (40k chars, truncation labelled;
oversize/skip/failure/unsupported all say so in the turn), and it never fails an upload.

The bytes are ALSO cached to disk (``cache_image_from_bytes_async`` for images — the same primitive
every messaging lane's inbound image goes through — and ``cache_document_from_bytes_async`` into
the documents cache beside it for documents) and the stored absolute path rides in the turn's
content as ``[Image attached at: <path>]`` / ``[Document attached at: <path>]``, exactly like
``agent.image_routing.build_native_content_parts`` labels a local-file attachment. So the upload is
not a fire-and-forget: a real file exists on disk for any later reader — ``vision_analyze`` on a
path, media delivery, a read-the-PDF tool call, or forensics over ``/api/sessions/{id}/messages`` —
and the transcript keeps that path alongside the caption (image pixels themselves stay inline on the
wire: a ``data:`` part in native routing, the vision backend's description in text routing). The
cache write is best-effort: a format the image cache refuses as "not an image" (SVG) still runs its
turn, it just has no path handle — an upload that used to work must not start failing.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, NamedTuple, Optional, Tuple

try:
    from aiohttp import web
except ImportError:  # pragma: no cover - aiohttp is required to serve this platform at all
    web = None  # type: ignore[assignment,misc]

logger = logging.getLogger("gateway.platforms.api_server")

# A declared-type echo longer than this is noise (and client input).
_MAX_ECHOED_TYPE = 40

# Cache extension per sniffed MIME — ``cache_image_from_bytes`` writes ``img_<uuid12><ext>`` and
# the extension is what a later reader (or the OS) opens the file with.
_EXT_BY_MIME: Dict[str, str] = {
    "image/png": ".png", "image/jpeg": ".jpg", "image/gif": ".gif", "image/webp": ".webp",
    "image/bmp": ".bmp", "image/tiff": ".tiff", "image/x-icon": ".ico", "image/svg+xml": ".svg",
}

# MIME spellings clients send that the gateway's own table spells differently, plus the ODF/RTF/ePub
# documents ``base.MEDIA_DELIVERY_EXTS`` can deliver back (so an uploaded document is also a
# re-deliverable one). Extensions here feed the filename fallback below.
_EXTRA_DOCUMENT_MIMES: Dict[str, str] = {
    "text/x-markdown": ".md", "text/markdown": ".md", "application/x-yaml": ".yaml",
    "application/rtf": ".rtf", "text/rtf": ".rtf", "application/vnd.oasis.opendocument.text": ".odt",
    "application/vnd.oasis.opendocument.spreadsheet": ".ods",
    "application/vnd.oasis.opendocument.presentation": ".odp", "application/epub+zip": ".epub",
}

# Declared types that carry no information about the payload: both Eve clients send
# ``application/octet-stream`` for a picked file whose type they cannot name (iOS for every
# extension absent from its table — .docx/.xlsx/.pptx included). For those the bytes' signature and
# the filename are the evidence; for any OTHER unknown declared type the declaration is taken at
# face value and the upload is refused.
_OPAQUE_DECLARED_TYPES = frozenset({"", "application/octet-stream", "binary/octet-stream",
                                    "application/unknown"})

# Document magic bytes the sniffer knows. Deliberately short: the zip container behind
# docx/xlsx/pptx/odt cannot be told apart without opening it, so those rely on the filename.
_PDF_MAGIC = b"%PDF-"

_DOC_MIME_TO_EXT: Optional[Dict[str, str]] = None
_DOC_EXT_TO_MIME: Optional[Dict[str, str]] = None


class _Upload(NamedTuple):
    """One accepted upload: the prompt text, the agent-ready content and the cached file."""

    message: str
    content: Any
    # Absolute path of the cache copy, ``""`` when the cache refused the bytes (an image format
    # ``gateway.platforms.base`` does not treat as a cacheable image, e.g. SVG).
    stored_path: str = ""
    # ``"image"`` (inline ``image_url`` part, routed by ``api_server_image_lane``) or
    # ``"document"`` (text turn naming the persisted path).
    kind: str = "image"


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


def _document_tables() -> Tuple[Dict[str, str], Dict[str, str]]:
    """``(mime -> ext, ext -> mime)`` for accepted documents, from the gateway's own tables.

    ``gateway.platforms.base.SUPPORTED_DOCUMENT_TYPES`` is the single canonical ext/MIME table the
    messaging lanes already classify attachments with; importing it lazily keeps this module's
    import cost unchanged (``base`` is heavyweight) and guarantees the two sides agree.
    """
    global _DOC_MIME_TO_EXT, _DOC_EXT_TO_MIME
    if _DOC_MIME_TO_EXT is None:
        from gateway.platforms.base import SUPPORTED_DOCUMENT_TYPES

        by_mime = {mime: ext for ext, mime in SUPPORTED_DOCUMENT_TYPES.items()}
        by_mime.update(_EXTRA_DOCUMENT_MIMES)
        by_ext: Dict[str, str] = {}
        for mime, ext in by_mime.items():
            by_ext.setdefault(ext, mime)
        _DOC_MIME_TO_EXT, _DOC_EXT_TO_MIME = by_mime, by_ext
    assert _DOC_EXT_TO_MIME is not None  # set together above
    return _DOC_MIME_TO_EXT, _DOC_EXT_TO_MIME


def _sniff_document_mime(raw: bytes) -> str:
    """``application/pdf`` for a payload that starts with the PDF signature, else ``""``."""
    return "application/pdf" if raw[:5] == _PDF_MAGIC else ""


def _accepted_document_mime(raw: bytes, declared: str, filename: str) -> str:
    """The document MIME to label the upload with; ``""`` when it is not an accepted document.

    Order of evidence: the client's declared type when it is one we accept (both Eve clients send
    the picker's real MIME whenever they have it), else — only for a declared type that says
    nothing (``application/octet-stream``, empty) — the bytes' own signature, else the filename
    extension. A declared type we do not accept is honoured as a refusal, never overridden by a
    filename: the client told us what it thinks the payload is, and guessing past that is how a
    mislabelled upload ends up stored under an extension it cannot be opened with.
    """
    by_mime, by_ext = _document_tables()
    if declared in by_mime:
        return declared
    if declared not in _OPAQUE_DECLARED_TYPES:
        return ""
    sniffed = _sniff_document_mime(raw)
    if sniffed:
        return sniffed
    return by_ext.get(os.path.splitext(filename or "")[1].lower(), "")


def _accepted_upload(raw: bytes, declared: str, filename: str) -> Tuple[str, str]:
    """``(mime, kind)`` for one upload; ``("", "")`` when it is neither an accepted image nor an
    accepted document."""
    image_mime = _accepted_image_mime(raw, declared)
    if image_mime:
        return image_mime, "image"
    document_mime = _accepted_document_mime(raw, declared, filename)
    if document_mime:
        return document_mime, "document"
    return "", ""


def _accepted_kinds_label() -> str:
    """Human list of what this lane accepts, for the refusal message (never a stack of names)."""
    image_exts = " ".join(sorted({ext.lstrip(".") for ext in _EXT_BY_MIME.values()}))
    _, by_ext = _document_tables()
    document_exts = " ".join(sorted({ext.lstrip(".") for ext in by_ext}))
    return f"images ({image_exts}) and documents ({document_exts})"


async def _cache_upload(raw: bytes, mime: str, filename: str, kind: str = "image") -> str:
    """Cache one upload beside the other inbound media of its kind; the absolute path, or ``""``.

    Best-effort on purpose. The image cache's ``_looks_like_image`` gate is narrower than this
    route's magic-byte sniffing (SVG — and TIFF/ICO/HEIC/AVIF — pass the sniffer and are refused by
    the cache), and an upload that used to run its turn must keep running one: it simply carries no
    path handle. Same for a filesystem failure: the inline ``data:`` part is still the turn's image,
    so a turn is never lost to the disk copy.
    """
    try:
        if kind == "document":
            # ``cache_document_from_bytes`` names the file ``doc_<uuid12>_<client filename>`` in the
            # documents cache (``<HERMES_HOME>/cache/documents``), the sibling of the image cache.
            from gateway.platforms.base import cache_document_from_bytes_async

            path = await cache_document_from_bytes_async(raw, filename or "document")
        else:
            from gateway.platforms.base import cache_image_from_bytes_async

            path = await cache_image_from_bytes_async(raw, _EXT_BY_MIME.get(mime, ".img"))
    except (ValueError, OSError) as exc:
        logger.warning(
            "[api_server] attachment not cached to disk (%s upload, %s): %s",
            filename or "unnamed", mime, exc)
        return ""
    logger.info("[api_server] %s attachment stored at %s (%d bytes, %s)",
                kind, path, len(raw), mime)
    return path


_EMPTY_DOCUMENT_CAPTION = "What's in this document?"


async def _extract_document_note(raw: bytes, mime: str, filename: str) -> str:
    """The ``[Document contents …]`` line for one upload; never empty, never raises.

    Runs in the thread pool — parsing a 20 MB office file is CPU work, not a coroutine — and is
    belt-and-braces about failure: ``api_server_document_lane.extract_note`` never raises, and if it
    somehow does, the turn still runs carrying an explicit "could not be extracted" line instead of
    a 5xx or a silent omission.
    """
    import asyncio

    from gateway.platforms import api_server_document_lane as _doc_lane

    try:
        return await asyncio.get_running_loop().run_in_executor(
            None, _doc_lane.extract_note, raw, mime, filename)
    except Exception as exc:  # noqa: BLE001 — extraction must never cost the upload its turn
        logger.warning("[api_server] document text extraction blew up for %s (%s): %s",
                       filename or "unnamed", mime, exc)
        return (f"[Document contents not extracted: extraction failed ({type(exc).__name__}). "
                f"text could not be extracted from this {mime} in this turn.]")


def _build_document_content(message: str, filename: str, mime: str, stored_path: str = "",
                            contents_note: str = "") -> str:
    """User content for one uploaded document: the caption, the two handles images get, the text.

    A plain string (not parts): the extracted text rides INLINE in the turn — the document-side
    equivalent of the vision backend's description — so a profile with no file tools can answer
    from its own context. The file still exists at ``stored_path`` for a profile that has them.
    The shape mirrors ``api_server_image_lane.build_upload_content`` (``[Image attached: …]`` /
    ``[Image attached at: …]``) with ``Document``, so one transcript reader understands both.
    """
    caption = (message or "").strip() or _EMPTY_DOCUMENT_CAPTION
    label = (filename or "").strip() or "document"
    attached = f"[Document attached: {label} ({mime})]"
    if (stored_path or "").strip():
        attached += f"\n[Document attached at: {stored_path.strip()}]"
    body = f"{caption}\n\n{attached}"
    note = (contents_note or "").strip()
    return f"{body}\n\n{note}" if note else body


async def _read_upload(request: "web.Request", _api_server) -> Tuple[Optional[_Upload], Optional["web.Response"]]:
    """``(upload, None)`` or ``(None, error)`` for the declared multipart contract.

    Ladder: unusable multipart body → 400 ``invalid_multipart_body``, no ``file`` part → 400
    ``missing_file``, empty ``file`` → 400 ``empty_file``, a payload that is neither an accepted
    image nor an accepted document → 415 ``unsupported_media_type``. An oversize body is aiohttp's
    own 413 (``client_max_size``), so ``HTTPException`` propagates instead of being rewritten into a
    400.
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
    filename = str(form.get("filename") or getattr(file_field, "filename", "") or "")
    mime, kind = _accepted_upload(bytes(raw), declared, filename)
    if not mime:
        shown = (declared or "unspecified")[:_MAX_ECHOED_TYPE]
        return None, _api_server._error_response(
            f"Unsupported attachment type {shown!r}: this endpoint accepts "
            f"{_accepted_kinds_label()}; audio goes to "
            "/api/sessions/{session_id}/voice-note.", 415, code="unsupported_media_type")

    from gateway.platforms import api_server_image_lane as _image_lane

    message = str(form.get("message") or "")
    stored_path = await _cache_upload(bytes(raw), mime, filename, kind)
    if kind == "document":
        contents_note = await _extract_document_note(bytes(raw), mime, filename)
        content: Any = _build_document_content(message, filename, mime, stored_path, contents_note)
    else:
        content = _image_lane.build_upload_content(
            message, filename, mime, bytes(raw), stored_path=stored_path)
    return _Upload(message=message, content=content, stored_path=stored_path, kind=kind), None


async def _handle_session_chat_attachment(self, request: "web.Request", *, _api_server) -> "web.Response":
    """POST /api/sessions/{session_id}/chat/attachments — one attachment upload, one session turn.

    Runs the turn exactly like ``/chat`` (same ``_prepare_session_chat`` validation, auth,
    session-key scoping, model-lock handling, retry policy and completion envelope), with an image
    upload as a native ``image_url`` part that ``_run_agent`` routes through the shared image lane,
    or a document upload as the text turn that hands the agent the persisted path.

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
