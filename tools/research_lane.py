"""Read-only research lane.

Ported pattern from OpenDots' ``src/server/research.ts`` + ``src/browser/index.ts``:
a research path that is deliberately narrower than the general web tools.

Four invariants, all fail-closed:

* **Canonical-URL allowlist** — only ``https`` URLs on an operator-provided host
  allowlist, no credentials in the URL, standard port. Off-list URLs are refused,
  never fetched.
* **No redirect escapes** — the browser service is asked not to follow redirects;
  a redirect is refused outright, and a redirect target outside the allowlist is
  named as such. A final URL that drifted off the allowlist is refused too.
* **Bearer-gated browser service** — the lane requires a browser URL *and* a
  bearer secret before it contacts anything; a missing secret is a refusal, not a
  best-effort anonymous fetch.
* **Live-vs-sample honesty flag** — ``mode`` is explicit. In sample mode the lane
  refuses to produce a fabricated brief and returns an explicit "not configured"
  error instead of inventing content. There is deliberately no sample-output path.

The module registers the ``research_readonly`` tool (``is_async=True``) at import
time, so ``tools/registry.py`` discovery picks it up like any other tool file.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Sequence, Tuple
from urllib.parse import urlsplit

from tools.registry import registry, tool_error

__all__ = [
    "DEFAULT_RESEARCH_ALLOWLIST",
    "MODE_LIVE",
    "MODE_SAMPLE",
    "ResearchLaneConfig",
    "ResearchLaneError",
    "ResearchNotConfigured",
    "assert_canonical_url",
    "canonical_url_allowed",
    "fetch_research",
    "load_research_config",
    "research",
]

MODE_LIVE = "live"
MODE_SAMPLE = "sample"

#: Operator-editable default allowlist of canonical hosts. Deliberately tiny and
#: documentation-grade: the lane is opt-in, so the default is not a general web.
DEFAULT_RESEARCH_ALLOWLIST: Tuple[str, ...] = (
    "docs.python.org",
    "developer.mozilla.org",
    "www.rfc-editor.org",
)

ENV_MODE = "HERMES_RESEARCH_MODE"
ENV_BROWSER_URL = "HERMES_RESEARCH_BROWSER_URL"
ENV_BROWSER_SECRET = "HERMES_RESEARCH_BROWSER_SECRET"
ENV_ALLOWLIST = "HERMES_RESEARCH_ALLOWLIST"

_NON_HTTPS_PORTS = {"", "443"}
_MAX_TEXT_CHARS = 30_000


class ResearchLaneError(Exception):
    """A read-only research request was refused; ``reason`` is client-safe."""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


class ResearchNotConfigured(ResearchLaneError):
    """The lane is not configured (sample mode, or missing browser credentials).

    Raised instead of emitting simulated/sample content — the honesty flag.
    """


@dataclass(frozen=True)
class ResearchLaneConfig:
    """Explicit configuration for one research call.

    ``mode`` is the live-versus-sample flag; it defaults to ``sample`` so an
    unconfigured lane cannot silently act as if it were live.
    """

    mode: str = MODE_SAMPLE
    browser_url: str = ""
    browser_secret: str = ""
    allowlist: Tuple[str, ...] = DEFAULT_RESEARCH_ALLOWLIST

    @property
    def is_live(self) -> bool:
        return self.mode == MODE_LIVE

    def credentials_present(self) -> bool:
        return bool(self.browser_url and self.browser_secret)


def _normalize_host(host: str) -> str:
    return (host or "").strip().lower().rstrip(".").removeprefix("www.")


def _allowed_hosts(allowlist: Sequence[str]) -> frozenset:
    return frozenset(_normalize_host(h) for h in allowlist if h and h.strip())


def canonical_url_allowed(url: str, allowlist: Sequence[str]) -> bool:
    """True only for an ``https`` URL, credential-free, standard port, allowlisted host."""
    try:
        parts = urlsplit(url)
    except ValueError:
        return False
    if parts.scheme != "https":
        return False
    if parts.username or parts.password:
        return False
    if parts.port not in (None, 443):
        return False
    host = _normalize_host(parts.hostname or "")
    if not host:
        return False
    return host in _allowed_hosts(allowlist)


def assert_canonical_url(url: str, allowlist: Sequence[str]) -> None:
    """Raise :class:`ResearchLaneError` unless *url* is canonical and allowlisted."""
    if not canonical_url_allowed(url, allowlist):
        raise ResearchLaneError(
            f"URL is not on the read-only research allowlist: {url!r}")


def load_research_config(environ: Optional[Dict[str, str]] = None) -> ResearchLaneConfig:
    """Build a config from the environment; unset ⇒ sample mode (fail-closed)."""
    env = os.environ if environ is None else environ
    raw_mode = (env.get(ENV_MODE) or MODE_SAMPLE).strip().lower()
    mode = raw_mode if raw_mode in (MODE_LIVE, MODE_SAMPLE) else MODE_SAMPLE
    raw_allow = (env.get(ENV_ALLOWLIST) or "").strip()
    allowlist = (
        tuple(h.strip() for h in raw_allow.split(",") if h.strip())
        or DEFAULT_RESEARCH_ALLOWLIST
    )
    return ResearchLaneConfig(
        mode=mode,
        browser_url=(env.get(ENV_BROWSER_URL) or "").strip(),
        browser_secret=(env.get(ENV_BROWSER_SECRET) or "").strip(),
        allowlist=allowlist,
    )


def _refusal_for_unconfigured(config: ResearchLaneConfig) -> ResearchNotConfigured:
    if not config.is_live:
        return ResearchNotConfigured(
            "Read-only research lane is not configured (mode=sample); it will not "
            "invent or simulate an answer. Set HERMES_RESEARCH_MODE=live plus a "
            "browser URL and secret to enable it.")
    return ResearchNotConfigured(
        "Read-only research lane is not configured: a browser service URL and bearer "
        "secret are required (HERMES_RESEARCH_BROWSER_URL, "
        "HERMES_RESEARCH_BROWSER_SECRET).")


def _validate_final_url(final_url: str, config: ResearchLaneConfig) -> None:
    if not final_url:
        return
    if not canonical_url_allowed(final_url, config.allowlist):
        raise ResearchLaneError(
            "Browser response landed on a URL outside the read-only research "
            f"allowlist: {final_url!r}")


async def fetch_research(
    url: str,
    *,
    config: ResearchLaneConfig,
    http_post: Optional[Callable[..., Any]] = None,
) -> Dict[str, Any]:
    """Fetch one allowlisted page through the bearer-gated browser service.

    ``http_post`` is an injectable transport ``async (url, *, headers, json) ->
    response``; when omitted an aiohttp session is created. Redirects are never
    followed: a 3xx response is refused, naming an off-allowlist target.
    """
    if not config.is_live or not config.credentials_present():
        raise _refusal_for_unconfigured(config)
    assert_canonical_url(url, config.allowlist)

    endpoint = config.browser_url.rstrip("/") + "/browse"
    payload = {"url": url, "follow_redirects": False}
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {config.browser_secret}",
    }

    if http_post is None:
        import aiohttp

        async with aiohttp.ClientSession() as session:
            async with session.post(
                endpoint, json=payload, headers=headers, allow_redirects=False
            ) as response:
                status = response.status
                location = response.headers.get("Location", "")
                body = await _read_json(response)
    else:
        response = await http_post(endpoint, headers=headers, json=payload)
        status = getattr(response, "status", None)
        location = _header(response, "Location")
        body = await _read_json(response)

    if isinstance(status, int) and 300 <= status < 400:
        if location and not canonical_url_allowed(location, config.allowlist):
            raise ResearchLaneError(
                "Refused a redirect out of the read-only research allowlist: "
                f"{location!r}")
        raise ResearchLaneError(
            "Refused a redirect: the read-only research lane does not follow "
            f"redirects (HTTP {status}{(' → ' + location) if location else ''}).")
    if status != 200:
        message = ""
        if isinstance(body, dict):
            message = str(body.get("error") or "")
        raise ResearchLaneError(
            f"Browser service returned HTTP {status}"
            + (f": {message}" if message else ""))

    if not isinstance(body, dict):
        raise ResearchLaneError("Browser service returned an invalid response.")
    for key in ("url", "title", "text"):
        value = body.get(key)
        if not isinstance(value, str) or not value.strip():
            raise ResearchLaneError(
                "Browser service returned an invalid or empty research response.")
    _validate_final_url(body["url"], config)
    return {
        "mode": MODE_LIVE,
        "url": body["url"],
        "title": body["title"],
        "text": body["text"][:_MAX_TEXT_CHARS],
        "source": {"title": body["title"], "url": body["url"]},
    }


async def _read_json(response: Any) -> Any:
    reader = getattr(response, "json", None)
    if reader is None:
        return None
    try:
        return await reader()
    except Exception:
        return None


def _header(response: Any, name: str) -> str:
    headers = getattr(response, "headers", None)
    if headers is None:
        return ""
    getter = getattr(headers, "get", None)
    return getter(name, "") if getter else ""


async def research(
    url: str,
    *,
    config: ResearchLaneConfig,
    http_post: Optional[Callable[..., Any]] = None,
) -> Dict[str, Any]:
    """Run the read-only research lane for one URL.

    In sample mode (or when browser credentials are absent) this refuses with an
    explicit :class:`ResearchNotConfigured`; it never returns invented content.
    """
    return await fetch_research(url, config=config, http_post=http_post)


# ---------------------------------------------------------------------------
# Tool registration — own toolset, following the tools/registry idiom.
# ---------------------------------------------------------------------------

RESEARCH_READONLY_SCHEMA = {
    "name": "research_readonly",
    "description": (
        "Read-only research lane: fetch a single page from an allowlisted canonical "
        "https host through a bearer-gated browser service, with redirects refused. "
        "Deliberately narrower than web_extract. Refuses (does not fabricate) when "
        "the lane is not configured with live mode plus browser URL and secret."),
    "parameters": {
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "Allowlisted canonical https URL to read (single page).",
            },
        },
        "required": ["url"],
    },
}


def check_research_lane() -> bool:
    """Always exposable: in sample mode the tool must be reachable to refuse honestly."""
    return True


async def research_readonly_tool(args: Dict[str, Any], **kwargs: Any) -> str:
    url = args.get("url")
    if not isinstance(url, str) or not url.strip():
        return tool_error("A single 'url' string is required.")
    config = load_research_config()
    try:
        result = await research(url.strip(), config=config)
    except ResearchNotConfigured as exc:
        return tool_error(exc.reason, code="research_lane_not_configured")
    except ResearchLaneError as exc:
        return tool_error(exc.reason, code="research_lane_refused")
    return json.dumps(result, ensure_ascii=False)


registry.register(
    name="research_readonly",
    toolset="research_readonly",
    schema=RESEARCH_READONLY_SCHEMA,
    handler=research_readonly_tool,
    check_fn=check_research_lane,
    is_async=True,
    emoji="🔎",
    max_result_size_chars=60_000,
)
