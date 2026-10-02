"""Tests for the read-only research lane (``tools.research_lane``).

Coverage: allowlisted URL allowed, off-allowlist refused, redirect out of the
allowlist refused, sample mode returning an explicit refusal instead of invented
content, bearer requirement, and the registered tool's error contract.
"""

import asyncio
import json

import pytest

from tools.research_lane import (
    DEFAULT_RESEARCH_ALLOWLIST,
    MODE_LIVE,
    MODE_SAMPLE,
    ResearchLaneConfig,
    ResearchLaneError,
    ResearchNotConfigured,
    assert_canonical_url,
    canonical_url_allowed,
    fetch_research,
    load_research_config,
    research,
    research_readonly_tool,
)

ALLOWLIST = ("docs.python.org", "example.org")
LIVE = ResearchLaneConfig(
    mode=MODE_LIVE,
    browser_url="https://browser.internal/browse-root",
    browser_secret="s" * 32,
    allowlist=ALLOWLIST,
)
SAMPLE = ResearchLaneConfig(mode=MODE_SAMPLE, allowlist=ALLOWLIST)


class _FakeResponse:
    def __init__(self, status=200, body=None, headers=None):
        self.status = status
        self._body = body
        self.headers = headers or {}

    async def json(self):
        return self._body


def _run(coro):
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# Canonical-URL allowlist
# ---------------------------------------------------------------------------


class TestAllowlist:
    def test_allowlisted_url_allowed(self):
        assert canonical_url_allowed("https://docs.python.org/3/library/", ALLOWLIST)
        assert canonical_url_allowed("https://example.org/a/b?c=d", ALLOWLIST)
        assert_canonical_url("https://docs.python.org/3/", ALLOWLIST)

    @pytest.mark.parametrize(
        "url",
        [
            "https://evil.example/page",            # off-allowlist host
            "http://docs.python.org/3/",            # not https
            "https://user:pw@docs.python.org/3/",   # credentials in URL
            "https://docs.python.org:8443/3/",      # non-standard port
            "ftp://docs.python.org/3/",             # wrong scheme
            "not a url",
        ],
    )
    def test_off_allowlist_url_refused(self, url):
        with pytest.raises(ResearchLaneError):
            assert_canonical_url(url, ALLOWLIST)

    def test_default_allowlist_is_small_and_canonical(self):
        assert "docs.python.org" in DEFAULT_RESEARCH_ALLOWLIST

    def test_load_config_defaults_to_sample(self):
        cfg = load_research_config({})
        assert cfg.mode == MODE_SAMPLE

    def test_load_config_reads_env(self):
        cfg = load_research_config({
            "HERMES_RESEARCH_MODE": "live",
            "HERMES_RESEARCH_BROWSER_URL": "https://b/x",
            "HERMES_RESEARCH_BROWSER_SECRET": "k" * 32,
            "HERMES_RESEARCH_ALLOWLIST": "a.example, b.example",
        })
        assert cfg.mode == MODE_LIVE
        assert cfg.allowlist == ("a.example", "b.example")


# ---------------------------------------------------------------------------
# Live mode: bearer + no redirect escapes
# ---------------------------------------------------------------------------


class TestLiveFetch:
    def test_allowlisted_live_fetch_succeeds(self):
        seen = {}

        async def _post(endpoint, *, headers, json):
            seen["endpoint"] = endpoint
            seen["headers"] = headers
            seen["json"] = json
            return _FakeResponse(200, {
                "url": "https://docs.python.org/3/",
                "title": "Python docs",
                "text": "hello world",
            })

        result = _run(research("https://docs.python.org/3/", config=LIVE, http_post=_post))
        assert result["mode"] == MODE_LIVE
        assert result["text"] == "hello world"
        assert result["source"]["url"] == "https://docs.python.org/3/"
        # Bearer auth is required and sent; redirects are explicitly disabled.
        assert seen["headers"]["Authorization"] == f"Bearer {LIVE.browser_secret}"
        assert seen["json"]["follow_redirects"] is False
        assert seen["endpoint"] == "https://browser.internal/browse-root/browse"

    def test_off_allowlist_fetch_refused_without_contacting_browser(self):
        called = []

        async def _post(endpoint, *, headers, json):
            called.append(endpoint)
            return _FakeResponse(200, {})

        with pytest.raises(ResearchLaneError):
            _run(research("https://evil.example/x", config=LIVE, http_post=_post))
        assert called == []

    def test_redirect_out_of_allowlist_refused(self):
        async def _post(endpoint, *, headers, json):
            return _FakeResponse(
                302, headers={"Location": "https://evil.example/landing"})

        with pytest.raises(ResearchLaneError) as exc:
            _run(research("https://docs.python.org/3/", config=LIVE, http_post=_post))
        assert "redirect" in exc.value.reason
        assert "allowlist" in exc.value.reason

    def test_in_allowlist_redirect_still_refused(self):
        # The read-only lane does not follow redirects at all.
        async def _post(endpoint, *, headers, json):
            return _FakeResponse(
                301, headers={"Location": "https://example.org/moved"})

        with pytest.raises(ResearchLaneError) as exc:
            _run(research("https://docs.python.org/3/", config=LIVE, http_post=_post))
        assert "redirect" in exc.value.reason

    def test_final_url_drifted_off_allowlist_refused(self):
        async def _post(endpoint, *, headers, json):
            return _FakeResponse(200, {
                "url": "https://evil.example/snuck",
                "title": "x", "text": "y",
            })

        with pytest.raises(ResearchLaneError) as exc:
            _run(research("https://docs.python.org/3/", config=LIVE, http_post=_post))
        assert "outside" in exc.value.reason

    def test_missing_bearer_secret_refused(self):
        no_secret = ResearchLaneConfig(
            mode=MODE_LIVE, browser_url="https://browser.internal", browser_secret="",
            allowlist=ALLOWLIST)

        async def _post(endpoint, *, headers, json):
            raise AssertionError("browser must not be contacted without a bearer secret")

        with pytest.raises(ResearchNotConfigured):
            _run(research("https://docs.python.org/3/", config=no_secret, http_post=_post))

    def test_browser_error_status_refused(self):
        async def _post(endpoint, *, headers, json):
            return _FakeResponse(502, {"error": "browser failed"})

        with pytest.raises(ResearchLaneError) as exc:
            _run(research("https://docs.python.org/3/", config=LIVE, http_post=_post))
        assert "HTTP 502" in exc.value.reason


# ---------------------------------------------------------------------------
# Sample mode: refuse, never invent
# ---------------------------------------------------------------------------


class TestSampleModeHonesty:
    def test_sample_mode_refuses_with_explicit_not_configured(self):
        async def _post(endpoint, *, headers, json):
            raise AssertionError("sample mode must not contact anything")

        with pytest.raises(ResearchNotConfigured) as exc:
            _run(research("https://docs.python.org/3/", config=SAMPLE, http_post=_post))
        reason = exc.value.reason
        assert "not configured" in reason
        assert "sample" in reason

    def test_sample_mode_returns_no_invented_content(self):
        async def _post(endpoint, *, headers, json):
            raise AssertionError("no network in sample mode")

        try:
            result = _run(research("https://docs.python.org/3/", config=SAMPLE, http_post=_post))
        except ResearchNotConfigured as exc:
            assert "text" not in exc.reason
            assert "source" not in exc.reason
        else:  # pragma: no cover - the refusal is the contract
            raise AssertionError(f"sample mode produced content: {result!r}")

    def test_sample_fetch_also_refuses(self):
        with pytest.raises(ResearchNotConfigured):
            _run(fetch_research("https://docs.python.org/3/", config=SAMPLE))

    def test_tool_reports_not_configured_without_inventing(self, monkeypatch):
        for key in ("HERMES_RESEARCH_MODE", "HERMES_RESEARCH_BROWSER_URL",
                    "HERMES_RESEARCH_BROWSER_SECRET", "HERMES_RESEARCH_ALLOWLIST"):
            monkeypatch.delenv(key, raising=False)
        out = _run(research_readonly_tool({"url": "https://docs.python.org/3/"}))
        payload = json.loads(out)
        assert "error" in payload
        assert payload["code"] == "research_lane_not_configured"
        assert "not configured" in payload["error"]
        assert "text" not in payload

    def test_tool_refuses_off_allowlist_url(self, monkeypatch):
        monkeypatch.setenv("HERMES_RESEARCH_MODE", "live")
        monkeypatch.setenv("HERMES_RESEARCH_BROWSER_URL", "https://b/x")
        monkeypatch.setenv("HERMES_RESEARCH_BROWSER_SECRET", "k" * 32)
        monkeypatch.setenv("HERMES_RESEARCH_ALLOWLIST", "docs.python.org")
        out = _run(research_readonly_tool({"url": "https://evil.example/x"}))
        payload = json.loads(out)
        assert payload["code"] == "research_lane_refused"
