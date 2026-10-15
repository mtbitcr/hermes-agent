"""A connection whose sign-in was refused reads as needing a new sign-in.

The gateway (token refresh, initial connect) and the dashboard
(``_mcp_connection_status``) are separate processes, so the refusal is
recorded as a marker file beside the connection's token file. Every token
here is an obvious fake inside a temporary HERMES_HOME.
"""

import httpx
import pytest

pytest.importorskip(
    "mcp.client.auth.oauth2",
    reason="MCP SDK 1.26.0+ required for OAuth support",
)

from mcp.shared.auth import OAuthToken  # noqa: E402

from hermes_cli.mcp_config import _mcp_connection_status  # noqa: E402
from tools.mcp_oauth import HermesTokenStorage  # noqa: E402

_SERVER = "srv"
_URL = "https://mcp.example.com/mcp"
_OAUTH_CONFIG = {"url": _URL, "auth": "oauth"}


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    return tmp_path


def _fake_token(prefix: str = "fake") -> OAuthToken:
    return OAuthToken(
        access_token=f"{prefix}-access-token",
        token_type="Bearer",
        expires_in=3600,
        refresh_token=f"{prefix}-refresh-token",
    )


def _marker(home):
    return home / "mcp-tokens" / f"{_SERVER}.signin-rejected"


async def _signed_in() -> HermesTokenStorage:
    """A connection that signed in earlier: its token file is on disk."""
    storage = HermesTokenStorage(_SERVER)
    await storage.set_tokens(_fake_token())
    assert _mcp_connection_status(_SERVER, _OAUTH_CONFIG) == "connected"
    return storage


def _provider():
    """The gateway's provider, built the production way (cached token, no TTY)."""
    from tools.mcp_oauth_manager import MCPOAuthManager

    provider = MCPOAuthManager().get_or_build_provider(_SERVER, _URL, {})
    assert provider is not None
    return provider


async def _assert_refused_refresh_needs_authorization(home, status: int) -> None:
    await _signed_in()
    provider = _provider()
    response = httpx.Response(status, content=b'{"error": "invalid_grant"}')

    assert await provider._handle_refresh_response(response) is False

    assert _mcp_connection_status(_SERVER, _OAUTH_CONFIG) == "needs_authorization"
    marker = _marker(home).read_text(encoding="utf-8")
    assert "fake-access-token" not in marker
    assert "fake-refresh-token" not in marker


# -- fail before, pass after -------------------------------------------------


@pytest.mark.asyncio
async def test_refresh_refused_with_400_needs_authorization(home):
    await _assert_refused_refresh_needs_authorization(home, 400)


@pytest.mark.asyncio
async def test_refresh_refused_with_401_needs_authorization(home):
    await _assert_refused_refresh_needs_authorization(home, 401)


@pytest.mark.asyncio
async def test_initial_connect_auth_refusal_needs_authorization_after_parking(
    home, monkeypatch
):
    from tools import mcp_tool_errors
    from tools.mcp_tool import MCPServerTask
    from tools.mcp_tool_server_run import _RetryBudget

    await _signed_in()
    request = httpx.Request("POST", _URL)
    refused = httpx.HTTPStatusError(
        "401 Unauthorized", request=request, response=httpx.Response(401, request=request)
    )
    failure_class = mcp_tool_errors._classify_mcp_failure(refused)
    assert failure_class == "permanent"

    status_while_parked = []

    async def _parked_until_shutdown(self, timeout=None):
        status_while_parked.append(_mcp_connection_status(_SERVER, _OAUTH_CONFIG))
        return "shutdown"

    monkeypatch.setattr(
        MCPServerTask, "_wait_for_reconnect_or_shutdown", _parked_until_shutdown
    )
    task = MCPServerTask(_SERVER)
    task._auth_type = "oauth"

    keep_running = await task._on_initial_connect_error(
        refused, refused, failure_class, _RetryBudget()
    )

    assert keep_running is False  # shut down while parked
    assert task._was_parked is True
    assert status_while_parked == ["needs_authorization"]
    assert _mcp_connection_status(_SERVER, _OAUTH_CONFIG) == "needs_authorization"


# -- pass before and after ---------------------------------------------------


@pytest.mark.asyncio
async def test_new_sign_in_after_refusal_reads_connected(home):
    storage = await _signed_in()
    # What a refused sign-in leaves behind: the stale token file plus the marker.
    _marker(home).write_text("{}", encoding="utf-8")

    await storage.set_tokens(_fake_token("fake-new"))

    assert _mcp_connection_status(_SERVER, _OAUTH_CONFIG) == "connected"


@pytest.mark.asyncio
async def test_status_without_marker_is_unchanged(home):
    assert _mcp_connection_status(_SERVER, _OAUTH_CONFIG) == "needs_authorization"
    assert _mcp_connection_status("plain", {"url": _URL}) == "configured"

    await HermesTokenStorage(_SERVER).set_tokens(_fake_token())

    assert _mcp_connection_status(_SERVER, _OAUTH_CONFIG) == "connected"
    assert not _marker(home).exists()


@pytest.mark.asyncio
async def test_refresh_server_error_writes_no_marker(home):
    await _signed_in()
    provider = _provider()

    for status in (500, 503):
        assert await provider._handle_refresh_response(httpx.Response(status)) is False

    assert not _marker(home).exists()
    assert _mcp_connection_status(_SERVER, _OAUTH_CONFIG) == "connected"
