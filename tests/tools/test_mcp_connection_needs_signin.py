"""A connection whose sign-in was refused reads as needing a new sign-in.

The gateway (token refresh, initial connect) and the dashboard
(``_mcp_connection_status``) are separate processes, so the refusal is
recorded as a marker file beside the connection's token file. A refusal
only counts against the tokens it refused: a newer sign-in saved meanwhile
by another process stays connected, and obsolete work never changes the
state of newer tokens. Every token here is an obvious fake inside a
temporary HERMES_HOME.
"""

import asyncio
import hashlib
import json
import os
import subprocess
import sys
import threading
import types
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


def _markers(home) -> list:
    return sorted((home / "mcp-tokens").glob(f"{_SERVER}.signin-rejected*"))


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
    for marker in _markers(home):
        text = marker.name + marker.read_text(encoding="utf-8")
        assert "fake-access-token" not in text
        assert "fake-refresh-token" not in text


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
    # What a refused sign-in leaves behind: the stale token file plus the marker
    # named by the SHA-256 of that token file.
    tokens_sha256 = hashlib.sha256(
        (home / "mcp-tokens" / f"{_SERVER}.json").read_bytes()
    ).hexdigest()
    (home / "mcp-tokens" / f"{_SERVER}.signin-rejected.{tokens_sha256}").touch()

    await storage.set_tokens(_fake_token("fake-new"))

    assert _mcp_connection_status(_SERVER, _OAUTH_CONFIG) == "connected"


@pytest.mark.asyncio
async def test_status_without_marker_is_unchanged(home):
    assert _mcp_connection_status(_SERVER, _OAUTH_CONFIG) == "needs_authorization"
    assert _mcp_connection_status("plain", {"url": _URL}) == "configured"

    await HermesTokenStorage(_SERVER).set_tokens(_fake_token())

    assert _mcp_connection_status(_SERVER, _OAUTH_CONFIG) == "connected"
    assert not _markers(home)


@pytest.mark.asyncio
async def test_refresh_server_error_writes_no_marker(home):
    await _signed_in()
    provider = _provider()

    for status in (500, 503):
        assert await provider._handle_refresh_response(httpx.Response(status)) is False

    assert not _markers(home)
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


# -- obsolete credential work never changes the state of newer tokens --------


@pytest.mark.asyncio
async def test_paused_old_refusal_keeps_refusal_of_replacement_tokens(home, monkeypatch):
    from tools import mcp_oauth

    old_provider = await _refresh_started("fake-old")
    paused, resume = threading.Event(), threading.Event()
    old_writer_result = []

    def _old_refresh_refused():
        response = httpx.Response(400, content=b'{"error": "invalid_grant"}')
        old_writer_result.append(asyncio.run(old_provider._handle_refresh_response(response)))

    old_writer = threading.Thread(target=_old_refresh_refused)

    def _sha256(*args, **kwargs):
        # The old writer pauses once it has read and hashed the old token file.
        digest = hashlib.sha256(*args, **kwargs)
        if threading.current_thread() is old_writer and not paused.is_set():
            paused.set()
            assert resume.wait(timeout=60)
        return digest

    monkeypatch.setattr(mcp_oauth, "hashlib", types.SimpleNamespace(sha256=_sha256))
    old_writer.start()
    try:
        assert await asyncio.to_thread(paused.wait, 60)
        _save_new_tokens_in_another_process(home)
        fresh_provider = _provider()
        await fresh_provider._initialize()
        request = await fresh_provider._refresh_token()
        assert b"fake-new-refresh-token" in request.content
        response = httpx.Response(401, content=b'{"error": "invalid_grant"}')
        assert await fresh_provider._handle_refresh_response(response) is False
        assert _mcp_connection_status(_SERVER, _OAUTH_CONFIG) == "needs_authorization"
    finally:
        resume.set()
        await asyncio.to_thread(old_writer.join, 60)

    assert old_writer_result == [False]
    assert _stored_tokens(home) == ("fake-new-access-token", "fake-new-refresh-token")
    assert _mcp_connection_status(_SERVER, _OAUTH_CONFIG) == "needs_authorization"


@pytest.mark.asyncio
async def test_initial_refusal_counts_after_later_successful_request(home, monkeypatch):
    from tools import mcp_tool_errors
    from tools.mcp_tool_server_run import _RetryBudget

    task, refused = await _initial_attempt_refused()
    # The transport's teardown DELETE on the same provider succeeds first.
    provider = task._build_oauth_auth(_URL, task._config)
    flow = provider.async_auth_flow(httpx.Request("DELETE", _URL))
    request = await flow.__anext__()
    assert request.headers["Authorization"] == "Bearer fake-access-token"
    with pytest.raises(StopAsyncIteration):
        await flow.asend(httpx.Response(204, request=request))
    status_while_parked = _parked_until_shutdown(monkeypatch)

    keep_running = await task._on_initial_connect_error(
        refused, refused, mcp_tool_errors._classify_mcp_failure(refused), _RetryBudget()
    )

    assert keep_running is False  # shut down while parked
    assert task._was_parked is True
    assert status_while_parked == ["needs_authorization"]
    assert _mcp_connection_status(_SERVER, _OAUTH_CONFIG) == "needs_authorization"


@pytest.mark.asyncio
async def test_synthetic_login_challenge_keeps_replacement_tokens_connected(
    home, monkeypatch
):
    from tools import mcp_tool_errors
    from tools.mcp_oauth_manager import get_manager
    from tools.mcp_tool import sdk_httpx
    from tools.mcp_tool_server_run import _RetryBudget

    task, refused = await _initial_attempt_refused("fake-old")
    _save_new_tokens_in_another_process(home)
    sdk = sdk_httpx()
    held, release = asyncio.Event(), asyncio.Event()
    sent = []

    async def _fake_network(self, request, **kwargs):
        sent.append(request)
        held.set()
        await release.wait()
        raise sdk.ConnectError("fake network: metadata discovery failed", request=request)

    monkeypatch.setattr(sdk.AsyncClient, "send", _fake_network)
    login = asyncio.create_task(
        get_manager().authorize_interactively(_SERVER, _URL, None)
    )
    try:
        await asyncio.wait_for(held.wait(), timeout=30)
        assert "/.well-known/" in str(sent[0].url)  # the first metadata request
        status_while_parked = _parked_until_shutdown(monkeypatch)

        keep_running = await task._on_initial_connect_error(
            refused, refused, mcp_tool_errors._classify_mcp_failure(refused), _RetryBudget()
        )
    finally:
        release.set()
    with pytest.raises(sdk.ConnectError):
        await asyncio.wait_for(login, timeout=30)

    assert keep_running is False  # shut down while parked
    assert status_while_parked == ["connected"]
    assert len(sent) == 1
    assert _stored_tokens(home) == ("fake-new-access-token", "fake-new-refresh-token")
    assert _mcp_connection_status(_SERVER, _OAUTH_CONFIG) == "connected"


# -- controls -------------------------------------------------------------------


@pytest.mark.asyncio
async def test_initial_refusal_counts_after_provider_replaced(home, monkeypatch):
    from tools import mcp_tool_errors
    from tools.mcp_oauth_manager import get_manager
    from tools.mcp_tool_server_run import _RetryBudget

    task, refused = await _initial_attempt_refused()
    old_provider = task._build_oauth_auth(_URL, task._config)
    get_manager().evict(_SERVER)
    assert task._build_oauth_auth(_URL, task._config) is not old_provider
    status_while_parked = _parked_until_shutdown(monkeypatch)

    keep_running = await task._on_initial_connect_error(
        refused, refused, mcp_tool_errors._classify_mcp_failure(refused), _RetryBudget()
    )

    assert keep_running is False  # shut down while parked
    assert status_while_parked == ["needs_authorization"]
    assert _mcp_connection_status(_SERVER, _OAUTH_CONFIG) == "needs_authorization"


@pytest.mark.asyncio
async def test_refusal_recorded_while_set_tokens_saves_stays(home, monkeypatch):
    from tools import mcp_oauth

    storage = await _signed_in()
    new_tokens = _fake_token("fake-new")
    real_write_json = mcp_oauth._write_json
    recorded = []

    def _write_json(path, data):
        real_write_json(path, data)
        if path == storage._tokens_path() and not recorded:
            # The new tokens are refused right after set_tokens wrote them.
            recorded.append(storage.mark_sign_in_rejected(new_tokens))

    monkeypatch.setattr(mcp_oauth, "_write_json", _write_json)

    await storage.set_tokens(new_tokens)

    assert recorded == [True]
    assert _stored_tokens(home) == ("fake-new-access-token", "fake-new-refresh-token")
    assert _mcp_connection_status(_SERVER, _OAUTH_CONFIG) == "needs_authorization"
