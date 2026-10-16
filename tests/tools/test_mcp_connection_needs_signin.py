"""A connection whose sign-in was refused reads as needing a new sign-in.

The gateway (token refresh, initial connect) and the dashboard
(``_mcp_connection_status``) are separate processes, so the refusal is
recorded as a marker file beside the connection's token file. A refusal
only counts against the tokens it refused: a newer sign-in saved meanwhile
by another process stays connected, and obsolete work never changes the
state of newer tokens. A failed owner transaction that restores refused
tokens restores their refusal with them. Every token here is an obvious
fake inside a temporary HERMES_HOME.
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
# Run in a second OS process: the gateway is refused the saved fake sign-in
# whose tokens start with argv[3]. Its refresh ("refresh") or its MCP request
# ("mcp") gets the answer argv[2]; only the network is fake.
_REFUSE_SAVED_SIGN_IN = """
import asyncio
import sys
from tools.mcp_oauth_manager import MCPOAuthManager
from tools.mcp_tool import sdk_httpx

async def refuse(path, status, prefix):
    sdk = sdk_httpx()
    url = "https://mcp.example.com/mcp"
    provider = MCPOAuthManager().get_or_build_provider("srv", url, {})
    if path == "refresh":
        await provider._initialize()
        request = await provider._refresh_token()
        assert (prefix + "-refresh-token").encode() in request.content
        response = sdk.Response(status, request=request, content=b'{"error": "invalid_grant"}')
        assert await provider._handle_refresh_response(response) is False
        return

    def network(request):
        # The MCP request and the SDK's retry, both sent with the saved token.
        assert request.url == sdk.URL(url)
        assert request.headers["Authorization"] == "Bearer " + prefix + "-access-token"
        return sdk.Response(status)

    async with sdk.AsyncClient(auth=provider, transport=sdk.MockTransport(network)) as client:
        response = await client.post(url, json={"jsonrpc": "2.0", "id": 1, "method": "ping"})
    assert response.status_code == status

asyncio.run(refuse(sys.argv[1], int(sys.argv[2]), sys.argv[3]))
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


async def _mcp_request_redirected_then_refused(
    location: str, status: int, *, redirected_authorization: str | None
) -> None:
    """The gateway's MCP request goes out with the saved sign-in, is redirected
    (302) to ``location``, and the redirected request is refused with ``status``.

    The SDK's own HTTP client follows the redirect as production's does, and
    drops the Authorization header on a redirect to another origin; only the
    network is fake. Whatever the SDK sends after the refusal (OAuth discovery
    after a 401, a retry after a 403) fails to connect.
    """
    from tools.mcp_tool import sdk_httpx

    sdk = sdk_httpx()
    refusals = []

    def _network(request):
        if refusals:
            raise sdk.ConnectError("fake network: nothing after the refusal", request=request)
        if request.url == sdk.URL(_URL):
            assert request.headers["Authorization"] == "Bearer fake-access-token"
            return sdk.Response(302, headers={"Location": location})
        assert request.url == sdk.URL(location)
        assert request.headers.get("Authorization") == redirected_authorization
        refusals.append(request)
        return sdk.Response(status)

    async with sdk.AsyncClient(
        auth=_provider(), follow_redirects=True, transport=sdk.MockTransport(_network)
    ) as client:
        with pytest.raises(sdk.ConnectError):
            await client.post(_URL, json={"jsonrpc": "2.0", "id": 1, "method": "ping"})
    assert len(refusals) == 1


async def _assert_cross_origin_refusal_keeps_connected(home, status: int) -> None:
    await asyncio.wait_for(_signed_in(), timeout=30)
    token_file = home / "mcp-tokens" / f"{_SERVER}.json"
    signed_in = token_file.read_bytes()

    await asyncio.wait_for(
        _mcp_request_redirected_then_refused(
            "https://other.example.net/mcp", status, redirected_authorization=None
        ),
        timeout=30,
    )

    assert _mcp_connection_status(_SERVER, _OAUTH_CONFIG) == "connected"
    assert not _markers(home)
    assert token_file.read_bytes() == signed_in


def _token_file(home) -> Path:
    return home / "mcp-tokens" / f"{_SERVER}.json"


def _configured(home) -> None:
    """The connection is in config.yaml, as the owner's operations require."""
    import yaml

    (home / "config.yaml").write_text(
        yaml.safe_dump({"mcp_servers": {_SERVER: dict(_OAUTH_CONFIG)}}), encoding="utf-8"
    )


def _refuse_in_another_process(home, path: str, status: int, prefix: str = "fake") -> None:
    """The gateway, a second OS process, is refused the saved sign-in: its
    refresh (``path="refresh"``) or its MCP request (``path="mcp"``) is
    answered with ``status``."""
    subprocess.run(
        [sys.executable, "-c", _REFUSE_SAVED_SIGN_IN, path, str(status), prefix],
        cwd=_REPO_ROOT,
        env={**os.environ, "HERMES_HOME": str(home)},
        check=True,
        timeout=60,
    )
    assert _mcp_connection_status(_SERVER, _OAUTH_CONFIG) == "needs_authorization"


def _after_snapshot(monkeypatch, then=None) -> list:
    """Record the snapshots an owner transaction takes, and run ``then`` right
    after its snapshot, before its temporary removal."""
    real_snapshot = HermesTokenStorage.snapshot
    snapshots = []

    def _snapshot(self):
        snapshot = real_snapshot(self)
        snapshots.append(snapshot)
        if then is not None:
            then()
        return snapshot

    monkeypatch.setattr(HermesTokenStorage, "snapshot", _snapshot)
    return snapshots


def _dashboard_sign_in_fails(home, monkeypatch) -> None:
    """The dashboard's sign-in worker removes the saved sign-in, the owner's
    new sign-in fails, and the worker rolls back."""
    from hermes_cli import mcp_config, web_server
    from tools.mcp_dashboard_oauth import DashboardOAuthFlow

    def _authorization_fails(name, config, *, connect_timeout):
        raise RuntimeError("fake: the new sign-in failed")

    monkeypatch.setattr(mcp_config, "_authorize_oauth_server", _authorization_fails)
    flow = DashboardOAuthFlow(
        flow_id="flow-srv",
        server_name=_SERVER,
        profile=None,
        hermes_home=str(home),
        redirect_uri="https://agent.example/api/mcp/oauth/callback/srv",
    )

    web_server._run_dashboard_mcp_oauth(flow, dict(_OAUTH_CONFIG))

    assert flow.status == "error"
    assert "fake: the new sign-in failed" in flow.error


def _removal_config_save_fails(home, monkeypatch) -> None:
    """Connection removal clears the saved sign-in, the config save fails, and
    the removal rolls back."""
    from hermes_cli import mcp_config

    def _save_fails(*args, **kwargs):
        raise OSError("fake: config.yaml could not be written")

    monkeypatch.setattr(mcp_config, "save_config", _save_fails)

    with pytest.raises(OSError, match="fake: config.yaml"):
        mcp_config._remove_mcp_server_and_credentials(_SERVER)


async def _assert_failed_transaction_keeps_refusal(
    home, monkeypatch, transaction, path: str, status: int
) -> None:
    """The gateway's refusal lands after the transaction's snapshot and before
    its removal; the failed transaction restores the refused sign-in."""
    await _signed_in()
    _configured(home)
    snapshots = _after_snapshot(
        monkeypatch, lambda: _refuse_in_another_process(home, path, status)
    )

    transaction(home, monkeypatch)

    [snapshot] = snapshots
    assert not [name for name in snapshot if ".signin-rejected." in name]
    assert _token_file(home).read_bytes() == snapshot[f"{_SERVER}.json"]
    assert _mcp_connection_status(_SERVER, _OAUTH_CONFIG) == "needs_authorization"


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


# -- only a refused request that carried the sign-in refuses it ----------------


@pytest.mark.asyncio
async def test_cross_origin_redirect_refused_with_401_stays_connected(home):
    await _assert_cross_origin_refusal_keeps_connected(home, 401)


@pytest.mark.asyncio
async def test_cross_origin_redirect_refused_with_403_stays_connected(home):
    await _assert_cross_origin_refusal_keeps_connected(home, 403)


# -- a failed owner transaction restores a refused sign-in with its refusal ----


@pytest.mark.asyncio
async def test_failed_dashboard_sign_in_keeps_refresh_400_refusal(home, monkeypatch):
    await _assert_failed_transaction_keeps_refusal(
        home, monkeypatch, _dashboard_sign_in_fails, "refresh", 400
    )


@pytest.mark.asyncio
async def test_failed_dashboard_sign_in_keeps_refresh_401_refusal(home, monkeypatch):
    await _assert_failed_transaction_keeps_refusal(
        home, monkeypatch, _dashboard_sign_in_fails, "refresh", 401
    )


@pytest.mark.asyncio
async def test_failed_dashboard_sign_in_keeps_mcp_403_refusal(home, monkeypatch):
    await _assert_failed_transaction_keeps_refusal(
        home, monkeypatch, _dashboard_sign_in_fails, "mcp", 403
    )


@pytest.mark.asyncio
async def test_failed_connection_removal_keeps_refresh_400_refusal(home, monkeypatch):
    await _assert_failed_transaction_keeps_refusal(
        home, monkeypatch, _removal_config_save_fails, "refresh", 400
    )


@pytest.mark.asyncio
async def test_failed_connection_removal_keeps_refresh_401_refusal(home, monkeypatch):
    await _assert_failed_transaction_keeps_refusal(
        home, monkeypatch, _removal_config_save_fails, "refresh", 401
    )


@pytest.mark.asyncio
async def test_failed_connection_removal_keeps_mcp_403_refusal(home, monkeypatch):
    await _assert_failed_transaction_keeps_refusal(
        home, monkeypatch, _removal_config_save_fails, "mcp", 403
    )


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


@pytest.mark.asyncio
async def test_same_origin_redirect_refused_with_401_needs_authorization(home):
    await asyncio.wait_for(_signed_in(), timeout=30)

    # The redirect stays on the server's origin, so the redirected request
    # keeps the saved sign-in, and its refusal counts.
    await asyncio.wait_for(
        _mcp_request_redirected_then_refused(
            "https://mcp.example.com/moved/mcp",
            401,
            redirected_authorization="Bearer fake-access-token",
        ),
        timeout=30,
    )

    assert _markers(home)
    assert _mcp_connection_status(_SERVER, _OAUTH_CONFIG) == "needs_authorization"


@pytest.mark.asyncio
async def test_failed_dashboard_sign_in_keeps_refusal_recorded_before_snapshot(
    home, monkeypatch
):
    await _signed_in()
    _configured(home)
    _refuse_in_another_process(home, "refresh", 400)
    snapshots = _after_snapshot(monkeypatch)

    _dashboard_sign_in_fails(home, monkeypatch)

    [snapshot] = snapshots
    assert _token_file(home).read_bytes() == snapshot[f"{_SERVER}.json"]
    assert _mcp_connection_status(_SERVER, _OAUTH_CONFIG) == "needs_authorization"


@pytest.mark.asyncio
async def test_failed_connection_removal_without_refusal_restores_connected(
    home, monkeypatch
):
    await _signed_in()
    _configured(home)
    snapshots = _after_snapshot(monkeypatch)

    _removal_config_save_fails(home, monkeypatch)

    [snapshot] = snapshots
    assert _token_file(home).read_bytes() == snapshot[f"{_SERVER}.json"]
    assert not _markers(home)
    assert _mcp_connection_status(_SERVER, _OAUTH_CONFIG) == "connected"


@pytest.mark.asyncio
async def test_refusal_of_replacement_tokens_leaves_restored_sign_in_connected(
    home, monkeypatch
):
    await _signed_in()
    _configured(home)

    def _replaced_then_refused():
        # The gateway saves replacement tokens, and the provider refuses those.
        _save_new_tokens_in_another_process(home)
        _refuse_in_another_process(home, "refresh", 400, "fake-new")

    snapshots = _after_snapshot(monkeypatch, _replaced_then_refused)

    _dashboard_sign_in_fails(home, monkeypatch)

    [snapshot] = snapshots
    assert _token_file(home).read_bytes() == snapshot[f"{_SERVER}.json"]
    assert _stored_tokens(home) == ("fake-access-token", "fake-refresh-token")
    assert _mcp_connection_status(_SERVER, _OAUTH_CONFIG) == "connected"


@pytest.mark.asyncio
async def test_successful_connection_removal_leaves_no_marker(home, monkeypatch):
    from hermes_cli import mcp_config

    await _signed_in()
    _configured(home)
    _after_snapshot(monkeypatch, lambda: _refuse_in_another_process(home, "mcp", 403))

    assert mcp_config._remove_mcp_server_and_credentials(_SERVER) is True

    assert not _markers(home)
    assert not _token_file(home).exists()
    assert _SERVER not in mcp_config._get_mcp_servers()
