"""A connection whose sign-in was refused reads as needing a new sign-in.

The gateway (token refresh, initial connect) and the dashboard
(``_mcp_connection_status``) are separate processes, so the refusal is
recorded as a marker file beside the connection's token file. A refusal
only counts against the tokens it refused: a newer sign-in saved meanwhile
by another process stays connected. Every token here is an obvious fake
inside a temporary HERMES_HOME.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import httpx
import pytest

pytest.importorskip(
    "mcp.client.auth.oauth2",
    reason="MCP SDK 1.26.0+ required for OAuth support",
)

from mcp.shared.auth import (  # noqa: E402
    OAuthClientInformationFull,
    OAuthMetadata,
    OAuthToken,
)

from hermes_cli.mcp_config import _mcp_connection_status  # noqa: E402
from tools.mcp_oauth import HermesTokenStorage  # noqa: E402

_SERVER = "srv"
_URL = "https://mcp.example.com/mcp"
_OAUTH_CONFIG = {"url": _URL, "auth": "oauth"}
_REPO_ROOT = Path(__file__).resolve().parents[2]
# Run in a second OS process: a new sign-in saving new fake tokens.
_SAVE_NEW_TOKENS = """
import asyncio
from mcp.shared.auth import OAuthToken
from tools.mcp_oauth import HermesTokenStorage
asyncio.run(HermesTokenStorage("srv").set_tokens(OAuthToken(
    access_token="fake-new-access-token", token_type="Bearer",
    expires_in=3600, refresh_token="fake-new-refresh-token")))
"""


@pytest.fixture
def home(tmp_path, monkeypatch):
    from tools.mcp_oauth_manager import reset_manager_for_tests

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    # The initial attempt's provider comes from the process-wide manager.
    reset_manager_for_tests()
    yield tmp_path
    reset_manager_for_tests()


def _fake_token(prefix: str = "fake") -> OAuthToken:
    return OAuthToken(
        access_token=f"{prefix}-access-token",
        token_type="Bearer",
        expires_in=3600,
        refresh_token=f"{prefix}-refresh-token",
    )


def _marker(home):
    return home / "mcp-tokens" / f"{_SERVER}.signin-rejected"


def _stored_tokens(home) -> tuple:
    data = json.loads((home / "mcp-tokens" / f"{_SERVER}.json").read_text(encoding="utf-8"))
    return data["access_token"], data["refresh_token"]


async def _signed_in(prefix: str = "fake") -> HermesTokenStorage:
    """A connection that signed in earlier: its client, server metadata and tokens are on disk."""
    storage = HermesTokenStorage(_SERVER)
    await storage.set_client_info(OAuthClientInformationFull(client_id="fake-client-id"))
    storage.save_oauth_metadata(
        OAuthMetadata(
            issuer="https://auth.example.com",
            authorization_endpoint="https://auth.example.com/authorize",
            token_endpoint="https://auth.example.com/token",
        )
    )
    await storage.set_tokens(_fake_token(prefix))
    assert _mcp_connection_status(_SERVER, _OAUTH_CONFIG) == "connected"
    return storage


def _provider():
    """The gateway's provider, built the production way (cached token, no TTY)."""
    from tools.mcp_oauth_manager import MCPOAuthManager

    provider = MCPOAuthManager().get_or_build_provider(_SERVER, _URL, {})
    assert provider is not None
    return provider


async def _refresh_started(prefix: str = "fake"):
    """Start a refresh as the gateway does: load the stored tokens, then build the request."""
    await _signed_in(prefix)
    provider = _provider()
    await provider._initialize()
    request = await provider._refresh_token()
    assert f"{prefix}-refresh-token".encode() in request.content
    return provider


def _save_new_tokens_in_another_process(home) -> None:
    """A new sign-in finishing in a second OS process, such as the dashboard."""
    subprocess.run(
        [sys.executable, "-c", _SAVE_NEW_TOKENS],
        cwd=_REPO_ROOT,
        env={**os.environ, "HERMES_HOME": str(home)},
        check=True,
        timeout=60,
    )
    assert _stored_tokens(home) == ("fake-new-access-token", "fake-new-refresh-token")
    assert _mcp_connection_status(_SERVER, _OAUTH_CONFIG) == "connected"


async def _initial_attempt_refused(prefix: str = "fake"):
    """The initial connection attempt, made as production makes it, and its 401.

    The task binds its config as ``_prepare_run`` does and gets the attempt's
    provider from ``_build_oauth_auth`` as ``_run_http`` does; that provider
    loads the stored tokens, sends the request with them and gets the 401.
    """
    from tools.mcp_tool import MCPServerTask

    await _signed_in(prefix)
    task = MCPServerTask(_SERVER)
    task._config = dict(_OAUTH_CONFIG)
    task._auth_type = "oauth"
    provider = task._build_oauth_auth(_URL, task._config)
    flow = provider.async_auth_flow(httpx.Request("POST", _URL))
    request = await flow.__anext__()
    assert request.headers["Authorization"] == f"Bearer {prefix}-access-token"
    response = httpx.Response(401, request=request)
    await flow.asend(response)  # the SDK answers the 401 with a discovery request
    await flow.aclose()  # the attempt ends
    refused = httpx.HTTPStatusError("401 Unauthorized", request=request, response=response)
    return task, refused


def _parked_until_shutdown(monkeypatch) -> list:
    """Park until shutdown, recording the status read while parked."""
    from tools.mcp_tool import MCPServerTask

    status_while_parked = []

    async def _wait(self, timeout=None):
        status_while_parked.append(_mcp_connection_status(_SERVER, _OAUTH_CONFIG))
        return "shutdown"

    monkeypatch.setattr(MCPServerTask, "_wait_for_reconnect_or_shutdown", _wait)
    return status_while_parked


async def _assert_refused_refresh_needs_authorization(home, status: int) -> None:
    provider = await _refresh_started()
    response = httpx.Response(status, content=b'{"error": "invalid_grant"}')

    assert await provider._handle_refresh_response(response) is False

    assert _mcp_connection_status(_SERVER, _OAUTH_CONFIG) == "needs_authorization"
    marker = _marker(home).read_text(encoding="utf-8")
    assert "fake-access-token" not in marker
    assert "fake-refresh-token" not in marker


async def _assert_late_refresh_refusal_keeps_new_sign_in(home, status: int) -> None:
    provider = await _refresh_started("fake-old")
    _save_new_tokens_in_another_process(home)
    # Only now does the old refresh request get its answer.
    response = httpx.Response(status, content=b'{"error": "invalid_grant"}')

    assert await provider._handle_refresh_response(response) is False

    assert _mcp_connection_status(_SERVER, _OAUTH_CONFIG) == "connected"
    assert _stored_tokens(home) == ("fake-new-access-token", "fake-new-refresh-token")


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
    from tools.mcp_tool_server_run import _RetryBudget

    task, refused = await _initial_attempt_refused()
    failure_class = mcp_tool_errors._classify_mcp_failure(refused)
    assert failure_class == "permanent"
    status_while_parked = _parked_until_shutdown(monkeypatch)

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


# -- a late refusal of replaced tokens keeps the new sign-in connected --------


@pytest.mark.asyncio
async def test_late_refresh_refusal_400_keeps_new_sign_in_connected(home):
    await _assert_late_refresh_refusal_keeps_new_sign_in(home, 400)


@pytest.mark.asyncio
async def test_late_refresh_refusal_401_keeps_new_sign_in_connected(home):
    await _assert_late_refresh_refusal_keeps_new_sign_in(home, 401)


@pytest.mark.asyncio
async def test_late_initial_connect_refusal_keeps_new_sign_in_connected(
    home, monkeypatch
):
    from tools import mcp_tool_errors
    from tools.mcp_tool_server_run import _RetryBudget

    task, refused = await _initial_attempt_refused("fake-old")
    _save_new_tokens_in_another_process(home)
    status_while_parked = _parked_until_shutdown(monkeypatch)

    keep_running = await task._on_initial_connect_error(
        refused, refused, mcp_tool_errors._classify_mcp_failure(refused), _RetryBudget()
    )

    assert keep_running is False  # shut down while parked
    assert task._was_parked is True
    assert status_while_parked == ["connected"]
    assert _mcp_connection_status(_SERVER, _OAUTH_CONFIG) == "connected"
    assert _stored_tokens(home) == ("fake-new-access-token", "fake-new-refresh-token")
