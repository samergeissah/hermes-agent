"""Inbound-image routing for every api_server turn.

WHY this seam exists: api_server clients hand over *already-normalized* OpenAI
``image_url`` content parts — inline JSON content on ``/api/sessions/{id}/chat[/stream]`` and
``/v1/chat[/responses]``, or the multipart upload on ``/api/sessions/{id}/chat/attachments`` —
while every messaging lane hands over local file paths. Because the parts were passed straight
to ``run_conversation``, a profile that routes images to a local ``auxiliary.vision`` backend
had those pixels attached to the *paid main model* on every api_server lane, with no
"Processing image with vision model" call at all (live: 336 KB JPEG, 2.4 s, main provider).

The decision itself is NEVER re-implemented here: ``agent.image_routing.decide_image_input_mode``
owns it (``agent.image_input_mode`` ``auto`` | ``native`` | ``text``), so this lane and the
messaging lanes cannot drift. Only the enrichment call site is api_server-specific, because the
image arrives as a URL/data-URL part rather than a path.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
from typing import Any, Callable, Dict, List

logger = logging.getLogger("gateway.platforms.api_server")

# Part types ``api_server._normalize_multimodal_content`` emits/outputs (``image_url`` canonical)
# plus the Responses-shaped spellings a client may send before normalization.
_IMAGE_PART_TYPES = frozenset({"image_url", "input_image"})
_TEXT_PART_TYPES = frozenset({"text", "input_text", "output_text"})

# Same brief the messaging lanes ask for (gateway/run_inbound.py::_enrich_message_with_vision),
# so the lossy text the main model receives is comparable whatever lane the image arrived on.
_VISION_PROMPT = (
    "Concisely describe this image in 2-4 sentences "
    "(~200 Chinese characters or ~150 English words). "
    "Cover the main subject, key visible text/data/code, and overall context. "
    "If it is a chart, diagram, or scientific figure, include the important "
    "labels, legend, and key values. Skip decorative details."
)

_EMPTY_CAPTION = "What do you see in this image?"


def _mime_kind(image_url: str) -> str:
    """Short, non-sensitive label for logs/hints: the data-URL MIME, else the scheme."""
    if image_url.startswith("data:"):
        return image_url[5:].split(";", 1)[0] or "data:"
    return image_url.split(":", 1)[0] if ":" in image_url else "url"


def image_parts(content: Any) -> List[Dict[str, Any]]:
    """``image_url``/``input_image`` parts of a user-content value ([] for plain string content)."""
    if not isinstance(content, list):
        return []
    return [
        part for part in content
        if isinstance(part, dict) and str(part.get("type") or "").strip().lower() in _IMAGE_PART_TYPES
    ]


def _part_url(part: Dict[str, Any]) -> str:
    ref = part.get("image_url")
    if isinstance(ref, dict):
        ref = ref.get("url")
    return ref.strip() if isinstance(ref, str) else ""


def content_text(content: Any) -> str:
    """The human caption of a user-content value (string as-is; text parts joined)."""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    return "\n".join(
        str(part.get("text") or "") for part in content
        if isinstance(part, dict) and str(part.get("type") or "").strip().lower() in _TEXT_PART_TYPES
    ).strip()


def build_upload_content(message: str, filename: str, mime: str, raw: bytes,
                         stored_path: str = "") -> List[Dict[str, Any]]:
    """Canonical user content for one uploaded image: caption text part + one ``image_url`` part.

    Same shape ``agent.image_routing.build_native_content_parts`` produces for a local file
    (caption carrying an ``[Image attached: …]`` handle), except the bytes are encoded inline —
    an upload needs no filesystem round-trip to reach the model, and the data URL survives cache
    retention. ``stored_path`` is the image-cache copy the upload lane also made
    (``api_server_attachments._cache_upload``); when present it adds the same
    ``[Image attached at: <path>]`` handle the local-file lanes emit, so a tool that takes a path
    (``vision_analyze``, media delivery) can re-read the very pixels of this turn.
    """
    caption = (message or "").strip() or _EMPTY_CAPTION
    label = (filename or "").strip() or "upload"
    data_url = f"data:{mime};base64,{base64.b64encode(raw).decode('ascii')}"
    attached = f"[Image attached: {label} ({mime})]"
    if (stored_path or "").strip():
        attached += f"\n[Image attached at: {stored_path.strip()}]"
    return [
        {"type": "text", "text": f"{caption}\n\n{attached}"},
        {"type": "image_url", "image_url": {"url": data_url}},
    ]


def decide_mode(agent: Any) -> str:
    """``"native"``/``"text"`` for the turn ``agent`` is about to run — the shared decision table.

    Uses the agent's own resolved identity (``provider``/``model``/``requested_provider``), the
    same way ``tui_gateway.session_history._active_image_routing_identity`` does, so a session
    ``/model`` override or a request-scoped provider routes against the model that will really
    answer instead of the gateway default.
    """
    from agent.image_routing import decide_image_input_mode
    from hermes_cli.config import load_config

    return decide_image_input_mode(
        str(getattr(agent, "provider", "") or ""),
        str(getattr(agent, "model", "") or ""),
        load_config(),
        requested_provider=str(getattr(agent, "requested_provider", "") or ""),
    )


def _vision_note(image_url: str, analyze: Callable[[str], Any]) -> str:
    """One up-front vision-backend description as a note the main model can read; never raises.

    A failure stays explicit in the message (``vision_analyze`` is still a tool the model can
    call) — silently dropping an attachment would leave the model describing an image it cannot
    see. The URL itself is never echoed when it is a data URL: that is megabytes of base64.
    """
    try:
        result = analyze(image_url)
        payload = json.loads(result) if isinstance(result, str) else {}
    except Exception as exc:
        logger.warning("[api_server] vision enrichment failed for %s image: %s", _mime_kind(image_url), exc)
        return "[The user sent an image but I could not look at it this time.]"
    description = str(payload.get("analysis") or "").strip() if isinstance(payload, dict) else ""
    if isinstance(payload, dict) and payload.get("success") and description:
        note = f"[The user sent an image~ Here's what I can see:\n{description}]"
        if not image_url.startswith("data:"):
            note += f"\n[If you need a closer look, use vision_analyze with image_url: {image_url} ~]"
        logger.info("Image routing: analyzed image with the configured vision backend.")
        return note
    logger.warning("[api_server] vision enrichment produced no analysis for %s image", _mime_kind(image_url))
    return "[The user sent an image but I could not look at it this time.]"


def route_user_message(agent: Any, content: Any) -> Any:
    """Route one inbound turn's images the way every other lane does, before the turn runs.

    * no image parts → ``content`` unchanged, so text-only turns pay nothing;
    * ``native``     → ``content`` unchanged: the parts stay attached to the main model;
    * ``text``       → each image is analyzed up-front by the configured vision backend and only
      its description reaches the main model, so pixels never leave the machine.
    """
    parts = image_parts(content)
    if not parts:
        return content
    try:
        mode = decide_mode(agent)
    except Exception as exc:  # matches the messaging lanes: unknown routing is not "send pixels"
        logger.warning("[api_server] image routing decision failed, defaulting to text: %s", exc)
        mode = "text"
    if mode == "native":
        logger.info(
            "Image routing: native (model supports vision). %d image(s) attached inline.", len(parts))
        return content
    logger.info(
        "Image routing: text (mode=%s). Pre-analyzing %d image(s) with the vision backend.",
        mode, len(parts))
    from tools.vision_tools import vision_analyze_tool

    def analyze(image_url: str) -> Any:
        # Executor thread for every caller (api_server turns bypass TurnRunner), so a private
        # loop here cannot collide with a running one.
        return asyncio.run(vision_analyze_tool(image_url=image_url, user_prompt=_VISION_PROMPT))

    prefix = "\n\n".join(_vision_note(url, analyze) for url in map(_part_url, parts) if url)
    caption = content_text(content)
    if not prefix:
        # Every image part was sourceless: keep the turn honest instead of an empty message.
        prefix = "[The user sent an image but no image source was available.]"
    return f"{prefix}\n\n{caption}" if caption else prefix
