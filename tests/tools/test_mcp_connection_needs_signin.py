"""A connection whose sign-in was refused reads as needing a new sign-in.

The gateway (token refresh, initial connect) and the dashboard
(``_mcp_connection_status``) are separate processes, so the refusal is
recorded as a marker file beside the connection's token file. A refusal
only counts against the tokens it refused: a newer sign-in saved meanwhile
by another process stays connected, and obsolete work never changes the
state of newer tokens. The marker files are the only record of refusals:
the rollback of a failed owner operation restores only the token files it
changed and never writes, deletes or restores a marker, and only a completed
connection removal deletes them. Every successful owner sign-in saves token
file bytes of its own, so no refusal of earlier bytes matches it. Every token
here is an obvious fake inside a temporary HERMES_HOME.
"""

import asyncio
import contextlib
import hashlib
import json
import os
import stat
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
# ("mcp") gets the answer argv[2]; only the network is fake. With argv[4]
# "pause", the writer matches the refusal to the token file bytes on disk,
# writes a ready line and waits for a line on stdin before it creates their
# marker; nothing it sends after the refusal connects.
_REFUSE_SAVED_SIGN_IN = """
import asyncio
import sys
from tools import mcp_oauth
from tools.mcp_oauth_manager import MCPOAuthManager
from tools.mcp_tool import sdk_httpx

pause = sys.argv[4:] == ["pause"]
create_marker = mcp_oauth._create_marker

def create_marker_after_pause(path):
    mcp_oauth._create_marker = create_marker
    print("paused before the marker", flush=True)
    sys.stdin.readline()
    create_marker(path)

if pause:
    mcp_oauth._create_marker = create_marker_after_pause

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
    refused = []

    def network(request):
        if (status == 401 and request.url != sdk.URL(url)) or (pause and refused):
            # The SDK answers a 401 with OAuth discovery, which fails to connect;
            # a paused writer's retry would go out after the owner's sign-in.
            raise sdk.ConnectError("fake network: nothing after the refusal", request=request)
        # The MCP request and the SDK's retry, both sent with the saved token.
        assert request.url == sdk.URL(url)
        assert request.headers["Authorization"] == "Bearer " + prefix + "-access-token"
        refused.append(request)
        return sdk.Response(status)

    async with sdk.AsyncClient(auth=provider, transport=sdk.MockTransport(network)) as client:
        try:
            response = await client.post(url, json={"jsonrpc": "2.0", "id": 1, "method": "ping"})
        except sdk.ConnectError:
            assert status == 401 or pause
            return
    assert response.status_code == status

asyncio.run(refuse(sys.argv[1], int(sys.argv[2]), sys.argv[3]))
"""
# Run in a second OS process: an owner operation on the saved sign-in, the
# dashboard's sign-in worker ("dashboard") or the connection removal
# ("removal"). After its own snapshot and temporary removal, its failing step
# writes a ready line, waits for a line on stdin, then fails, and the operation
# rolls back. Only the failing step is replaced.
_OWNER_OPERATION_WAITS_THEN_FAILS = """
import os
import sys
from hermes_cli import mcp_config, web_server
from tools.mcp_dashboard_oauth import DashboardOAuthFlow

def wait_then_fail(*args, **kwargs):
    print("waiting at the failing step", flush=True)
    sys.stdin.readline()
    raise OSError("fake: the overlapping operation failed")

if sys.argv[1] == "removal":
    mcp_config.save_config = wait_then_fail
    try:
        mcp_config._remove_mcp_server_and_credentials("srv")
    except OSError as exc:
        assert "fake: the overlapping operation failed" in str(exc), exc
    else:
        raise AssertionError("the connection removal did not fail")
else:
    mcp_config._authorize_oauth_server = wait_then_fail
    flow = DashboardOAuthFlow(
        flow_id="flow-srv-overlapping",
        server_name="srv",
        profile=None,
        hermes_home=os.environ["HERMES_HOME"],
        redirect_uri="https://agent.example/api/mcp/oauth/callback/srv",
    )
    web_server._run_dashboard_mcp_oauth(flow, {"url": "https://mcp.example.com/mcp", "auth": "oauth"})
    assert flow.status == "error", flow.status
    assert "fake: the overlapping operation failed" in flow.error, flow.error
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
    prefix = HermesTokenStorage(_SERVER)._sign_in_rejected_prefix()
    return sorted((home / "mcp-tokens").glob(f"{prefix}*"))


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


def _dashboard_sign_in_fails(home, monkeypatch, at_failing_step=None) -> None:
    """The dashboard's sign-in worker removes the saved sign-in, the owner's
    new sign-in fails (right after ``at_failing_step`` runs), and the worker
    rolls back."""
    from hermes_cli import mcp_config, web_server
    from tools.mcp_dashboard_oauth import DashboardOAuthFlow

    def _authorization_fails(name, config, *, connect_timeout):
        if at_failing_step is not None:
            at_failing_step()
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


def _removal_config_save_fails(home, monkeypatch, at_failing_step=None) -> None:
    """Connection removal clears the saved sign-in, the config save fails
    (right after ``at_failing_step`` runs), and the removal rolls back."""
    from hermes_cli import mcp_config

    def _save_fails(*args, **kwargs):
        if at_failing_step is not None:
            at_failing_step()
        raise OSError("fake: config.yaml could not be written")

    monkeypatch.setattr(mcp_config, "save_config", _save_fails)

    with pytest.raises(OSError, match="fake: config.yaml"):
        mcp_config._remove_mcp_server_and_credentials(_SERVER)


_OWNER_OPERATION_FAILS = {
    "dashboard": _dashboard_sign_in_fails,
    "removal": _removal_config_save_fails,
}


def _owner_operation_waits_in_another_process(home, kind: str, children: list) -> None:
    """A second owner operation (``kind``: "dashboard" or "removal") runs in
    another OS process: it takes its own snapshot, does its temporary removal
    and waits at its failing step. Returns once it is waiting there; the caller
    reaps each entry of ``children`` with ``_overlapping_operation_fails``."""
    child = subprocess.Popen(
        [sys.executable, "-c", _OWNER_OPERATION_WAITS_THEN_FAILS, kind],
        cwd=_REPO_ROOT,
        env={**os.environ, "HERMES_HOME": str(home)},
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    output, waiting = [], threading.Event()

    def _read_lines():
        for line in child.stdout:
            output.append(line)
            if line.strip() == "waiting at the failing step":
                waiting.set()
        waiting.set()  # end of output: the operation has exited

    reader = threading.Thread(target=_read_lines, daemon=True)
    reader.start()
    children.append((child, reader, output))
    # pytest.fail, not assert: the dashboard worker reports an AssertionError
    # as its own error.
    if not waiting.wait(timeout=120):
        pytest.fail(f"the {kind} operation did not reach its failing step:\n{''.join(output)}")
    if not any(line.strip() == "waiting at the failing step" for line in output):
        pytest.fail(f"the {kind} operation exited before its failing step:\n{''.join(output)}")


def _overlapping_operation_fails(child, reader, output) -> tuple:
    """Let the waiting operation fail and roll back: send its continue line and
    wait for it to exit, killing it if it does not. Returns its exit status and
    its output."""
    try:
        child.stdin.write("continue\n")
        child.stdin.close()
    except OSError:
        pass  # it has exited already
    try:
        child.wait(timeout=120)
    except subprocess.TimeoutExpired:
        child.kill()
        child.wait(timeout=60)
    reader.join(timeout=60)
    return child.returncode, "".join(output)


def _marker_files(home) -> list:
    """Each marker file's name, bytes and inode."""
    return [(m.name, m.read_bytes(), m.stat().st_ino) for m in _markers(home)]


async def _authorized_with_stable_tokens() -> None:
    """The owner signs in and adds the connection. The token endpoint answers
    with fake tokens without a lifetime (no ``expires_in``), which the real
    provider's successful token-response handler saves; then the connection is
    saved in config.yaml. Only the token endpoint's answer is a fixture."""
    from hermes_cli import mcp_config
    from tools.mcp_oauth import force_interactive_oauth
    from tools.mcp_oauth_manager import get_manager
    from tools.mcp_tool import sdk_httpx

    storage = HermesTokenStorage(_SERVER)
    await storage.set_client_info(OAuthClientInformationFull(client_id="fake-client-id"))
    storage.save_oauth_metadata(
        OAuthMetadata(
            issuer="https://auth.example.com",
            authorization_endpoint="https://auth.example.com/authorize",
            token_endpoint="https://auth.example.com/token",
        )
    )
    with force_interactive_oauth():  # the owner's own sign-in, as the dashboard's
        provider = get_manager().get_or_build_provider(_SERVER, _URL, {})
        await provider._initialize()
        stable = {
            "access_token": "fake-stable-access-token",
            "refresh_token": "fake-stable-refresh-token",
            "token_type": "Bearer",
        }
        await provider._handle_token_response(sdk_httpx().Response(200, json=stable))
    assert mcp_config._save_mcp_server(_SERVER, dict(_OAUTH_CONFIG)) is True


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
    storage._sign_in_rejected_path(tokens_sha256).touch()

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


# -- a refusal recorded during a failed owner operation stays on disk ---------


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


# -- a rollback never touches a marker; only a completed removal deletes them --


@pytest.mark.parametrize("inner", ["dashboard", "removal"])
@pytest.mark.parametrize("outer", ["dashboard", "removal"])
@pytest.mark.asyncio
async def test_overlapping_failed_owner_operations_keep_recorded_refusal(
    home, monkeypatch, outer, inner
):
    from hermes_cli import mcp_config

    await _signed_in()
    _configured(home)
    signed_in = _token_file(home).read_bytes()
    marker = HermesTokenStorage(_SERVER)._sign_in_rejected_path(
        hashlib.sha256(signed_in).hexdigest()
    ).name
    children, while_inner_waits = [], []

    def _refused_then_inner_operation_waits():
        # The gateway records the refusal of the tokens on disk; then a second
        # owner operation removes them and waits at its failing step.
        _refuse_in_another_process(home, "refresh", 400)
        _owner_operation_waits_in_another_process(home, inner, children)
        while_inner_waits.append(
            (_token_file(home).exists(), [m.name for m in _markers(home)])
        )

    _after_snapshot(monkeypatch, _refused_then_inner_operation_waits)
    try:
        _OWNER_OPERATION_FAILS[outer](home, monkeypatch)

        assert while_inner_waits == [(False, [marker])]
        assert _mcp_connection_status(_SERVER, _OAUTH_CONFIG) == "needs_authorization"
    finally:
        finished = [_overlapping_operation_fails(*child) for child in children]

    assert [returncode for returncode, _ in finished] == [0], finished
    assert _token_file(home).read_bytes() == signed_in
    assert _SERVER in mcp_config._get_mcp_servers()
    [path] = _markers(home)
    assert path.name == marker
    assert path.read_bytes() == b""
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert _mcp_connection_status(_SERVER, _OAUTH_CONFIG) == "needs_authorization"


@pytest.mark.parametrize("operation", ["dashboard", "removal"])
@pytest.mark.asyncio
async def test_rollback_after_completed_removal_and_stable_re_add_stays_connected(
    home, monkeypatch, operation
):
    from hermes_cli import mcp_config

    await _authorized_with_stable_tokens()
    assert _mcp_connection_status(_SERVER, _OAUTH_CONFIG) == "connected"
    _refuse_in_another_process(home, "refresh", 400, "fake-stable")

    assert mcp_config._remove_mcp_server_and_credentials(_SERVER) is True
    assert not _token_file(home).exists()
    assert not _markers(home)
    assert _SERVER not in mcp_config._get_mcp_servers()

    # Tokens without a lifetime can be stable, so the new sign-in saves the
    # same credentials under the same connection name.
    await _authorized_with_stable_tokens()
    assert _stored_tokens(home) == ("fake-stable-access-token", "fake-stable-refresh-token")
    assert _mcp_connection_status(_SERVER, _OAUTH_CONFIG) == "connected"
    assert not _markers(home)

    # No refusal arrives before, during or after the failed operation.
    _OWNER_OPERATION_FAILS[operation](home, monkeypatch)

    assert _stored_tokens(home) == ("fake-stable-access-token", "fake-stable-refresh-token")
    assert _mcp_connection_status(_SERVER, _OAUTH_CONFIG) == "connected"
    assert not _markers(home)


async def _assert_failed_operation_leaves_marker_files(home, monkeypatch, operation) -> None:
    """Refusals of the signed-in tokens and of their replacement D are on disk;
    the failed operation neither deletes, rewrites nor restores a marker file."""
    await _signed_in()
    _configured(home)
    _refuse_in_another_process(home, "refresh", 400)
    _save_new_tokens_in_another_process(home)
    _refuse_in_another_process(home, "refresh", 400, "fake-new")
    replacement = _token_file(home).read_bytes()
    marker_files = _marker_files(home)
    assert len(marker_files) == 2
    at_failing_step = []

    operation(
        home,
        monkeypatch,
        lambda: at_failing_step.append((_token_file(home).exists(), _marker_files(home))),
    )

    assert at_failing_step == [(False, marker_files)]
    assert _marker_files(home) == marker_files
    assert _token_file(home).read_bytes() == replacement
    assert _mcp_connection_status(_SERVER, _OAUTH_CONFIG) == "needs_authorization"


@pytest.mark.asyncio
async def test_failed_dashboard_sign_in_leaves_marker_files_unchanged(home, monkeypatch):
    await _assert_failed_operation_leaves_marker_files(
        home, monkeypatch, _dashboard_sign_in_fails
    )


@pytest.mark.asyncio
async def test_failed_connection_removal_leaves_marker_files_unchanged(home, monkeypatch):
    await _assert_failed_operation_leaves_marker_files(
        home, monkeypatch, _removal_config_save_fails
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


# -- a successful new sign-in that receives the same tokens reads connected ----

_STABLE_TOKEN_RESPONSE = {
    "access_token": "fake-stable-access-token",
    "refresh_token": "fake-stable-refresh-token",
    "token_type": "Bearer",
}


def _stable_tokens_reissued() -> None:
    """The owner's new sign-in completes: the token endpoint answers with the
    same fake tokens, again without a lifetime, and the real provider's
    successful token-response handler saves them. The client and server
    metadata are saved again first, as the owner operation removed them."""
    from tools.mcp_oauth import force_interactive_oauth
    from tools.mcp_oauth_manager import get_manager
    from tools.mcp_tool import sdk_httpx

    async def _token_response_saved():
        storage = HermesTokenStorage(_SERVER)
        await storage.set_client_info(OAuthClientInformationFull(client_id="fake-client-id"))
        storage.save_oauth_metadata(
            OAuthMetadata(
                issuer="https://auth.example.com",
                authorization_endpoint="https://auth.example.com/authorize",
                token_endpoint="https://auth.example.com/token",
            )
        )
        provider = get_manager().get_or_build_provider(_SERVER, _URL, {})
        await provider._initialize()
        sdk = sdk_httpx()
        request = sdk.Request("POST", "https://auth.example.com/token")
        await provider._handle_token_response(
            sdk.Response(200, request=request, json=_STABLE_TOKEN_RESPONSE)
        )

    with force_interactive_oauth():  # the owner's own sign-in
        asyncio.run(_token_response_saved())


@pytest.mark.parametrize("surface", ["dashboard", "cli", "desktop"])
def test_successful_stable_re_sign_in_reads_connected(home, monkeypatch, surface):
    from hermes_cli import mcp_config
    from tools.mcp_oauth_manager import get_manager
    from tools.mcp_tool import sdk_httpx

    asyncio.run(_authorized_with_stable_tokens())
    _refuse_in_another_process(home, "refresh", 400, "fake-stable")
    marker_files = _marker_files(home)
    assert len(marker_files) == 1
    accepted = []

    def _authorize(name, config, *, connect_timeout):
        _stable_tokens_reissued()

    def _probe(name, config, connect_timeout=None, *, details=None):
        if surface == "desktop":
            # The desktop path gets its tokens during the probe.
            _stable_tokens_reissued()
        sdk = sdk_httpx()

        def _network(request):
            assert request.headers["Authorization"] == "Bearer fake-stable-access-token"
            accepted.append(request)
            return sdk.Response(200, json={"jsonrpc": "2.0", "id": 1, "result": {}})

        async def _authenticated_request():
            provider = get_manager().get_or_build_provider(_SERVER, _URL, {})
            async with sdk.AsyncClient(
                auth=provider, transport=sdk.MockTransport(_network)
            ) as client:
                response = await client.post(
                    _URL, json={"jsonrpc": "2.0", "id": 1, "method": "ping"}
                )
            assert response.status_code == 200

        asyncio.run(_authenticated_request())
        return [("fake-tool", "A fake tool of the signed-in connection")]

    monkeypatch.setattr(mcp_config, "_authorize_oauth_server", _authorize)
    monkeypatch.setattr(mcp_config, "_probe_single_server", _probe)

    if surface == "cli":
        assert mcp_config._reauth_oauth_server(_SERVER, dict(_OAUTH_CONFIG)) is True
    else:
        from tools.mcp_dashboard_oauth import DashboardOAuthFlow

        flow = DashboardOAuthFlow(
            flow_id=f"flow-srv-{surface}",
            server_name=_SERVER,
            profile=None,
            hermes_home=str(home),
            redirect_uri="https://agent.example/api/mcp/oauth/callback/srv",
        )
        if surface == "dashboard":
            from hermes_cli import web_server

            web_server._run_dashboard_mcp_oauth(flow, dict(_OAUTH_CONFIG))
        else:
            from tools.mcp_oauth import force_interactive_oauth
            from tui_gateway.mcp_oauth_sessions import _probe_with_rollback

            with force_interactive_oauth():
                _probe_with_rollback(_SERVER, dict(_OAUTH_CONFIG), str(home), flow, False)
        assert flow.status == "approved", flow.error

    assert len(accepted) == 1
    # Credentials, not file bytes: refused bytes under their kept marker never read connected.
    assert _stored_tokens(home) == ("fake-stable-access-token", "fake-stable-refresh-token")
    assert _SERVER in mcp_config._get_mcp_servers()
    assert _marker_files(home) == marker_files
    tokens = asyncio.run(HermesTokenStorage(_SERVER).get_tokens())
    assert tokens.access_token == "fake-stable-access-token"
    assert tokens.refresh_token == "fake-stable-refresh-token"
    assert tokens.expires_in is None
    assert _mcp_connection_status(_SERVER, _OAUTH_CONFIG) == "connected"


def test_refresh_that_reissues_refused_stable_tokens_reads_connected(home):
    from tools.mcp_tool import sdk_httpx

    asyncio.run(_authorized_with_stable_tokens())
    _refuse_in_another_process(home, "mcp", 401, "fake-stable")
    marker_files = _marker_files(home)

    async def _refreshed():
        # The refresh cures the 401, and the provider answers it with the same
        # tokens, again without a lifetime.
        provider = _provider()
        await provider._initialize()
        request = await provider._refresh_token()
        assert b"fake-stable-refresh-token" in request.content
        response = sdk_httpx().Response(200, request=request, json=_STABLE_TOKEN_RESPONSE)
        return await provider._handle_refresh_response(response)

    assert asyncio.run(_refreshed()) is True

    assert _marker_files(home) == marker_files
    assert _stored_tokens(home) == ("fake-stable-access-token", "fake-stable-refresh-token")
    assert _mcp_connection_status(_SERVER, _OAUTH_CONFIG) == "connected"


def test_failed_dashboard_sign_in_after_stable_refusal_restores_refused_tokens(
    home, monkeypatch
):
    asyncio.run(_authorized_with_stable_tokens())
    _refuse_in_another_process(home, "refresh", 400, "fake-stable")
    refused = _token_file(home).read_bytes()
    marker_files = _marker_files(home)

    _dashboard_sign_in_fails(home, monkeypatch)

    assert _token_file(home).read_bytes() == refused
    assert _marker_files(home) == marker_files
    assert _mcp_connection_status(_SERVER, _OAUTH_CONFIG) == "needs_authorization"


def test_two_owner_saves_of_identical_credentials_write_different_files(home):
    asyncio.run(_authorized_with_stable_tokens())
    first = _token_file(home).read_bytes()
    first_tokens = asyncio.run(HermesTokenStorage(_SERVER).get_tokens())

    _stable_tokens_reissued()

    assert asyncio.run(HermesTokenStorage(_SERVER).get_tokens()) == first_tokens
    assert not _markers(home)
    assert _mcp_connection_status(_SERVER, _OAUTH_CONFIG) == "connected"
    assert _token_file(home).read_bytes() != first


# -- a refusal writer paused before its marker never matches a new sign-in ----


@contextlib.contextmanager
def _refusal_paused_before_its_marker(home, path: str, status: int):
    """The gateway, a second OS process, is refused the saved stable sign-in:
    its refresh (``path="refresh"``) or its MCP request (``path="mcp"``) is
    answered with ``status``. It matches the refusal to the token file bytes on
    disk and pauses just before it creates their marker; leaving the block
    releases it, and it must then exit 0."""
    child = subprocess.Popen(
        [sys.executable, "-c", _REFUSE_SAVED_SIGN_IN, path, str(status), "fake-stable", "pause"],
        cwd=_REPO_ROOT,
        env={**os.environ, "HERMES_HOME": str(home)},
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    output, paused = [], threading.Event()

    def _read_lines():
        for line in child.stdout:
            output.append(line)
            if line.strip() == "paused before the marker":
                paused.set()
        paused.set()  # end of output: the writer has exited

    reader = threading.Thread(target=_read_lines, daemon=True)
    reader.start()
    try:
        if not paused.wait(timeout=120):
            pytest.fail(f"the refusal writer did not pause:\n{''.join(output)}")
        if not any(line.strip() == "paused before the marker" for line in output):
            pytest.fail(f"the refusal writer exited before its marker:\n{''.join(output)}")
        yield
    finally:
        try:
            child.stdin.write("released\n")
            child.stdin.close()
        except OSError:
            pass  # it has exited already
        try:
            child.wait(timeout=120)
        except subprocess.TimeoutExpired:
            child.kill()
            child.wait(timeout=60)
        reader.join(timeout=60)
    assert child.returncode == 0, "".join(output)


def _stored_sign_in_accepted() -> None:
    """An MCP request through the SDK's own HTTP client carries the stored
    stable sign-in, and the fake network accepts it."""
    from tools.mcp_oauth_manager import get_manager
    from tools.mcp_tool import sdk_httpx

    sdk = sdk_httpx()
    accepted = []

    def _network(request):
        assert request.headers["Authorization"] == "Bearer fake-stable-access-token"
        accepted.append(request)
        return sdk.Response(200, json={"jsonrpc": "2.0", "id": 1, "result": {}})

    async def _authenticated_request():
        provider = get_manager().get_or_build_provider(_SERVER, _URL, {})
        async with sdk.AsyncClient(auth=provider, transport=sdk.MockTransport(_network)) as client:
            return await client.post(_URL, json={"jsonrpc": "2.0", "id": 1, "method": "ping"})

    response = asyncio.run(asyncio.wait_for(_authenticated_request(), timeout=30))
    assert response.status_code == 200
    assert len(accepted) == 1


def _owner_signs_in_again(home, monkeypatch, entry: str) -> None:
    """The owner's successful new sign-in through the real entry point: the
    dashboard's sign-in worker, the CLI's reauthorization, the desktop's
    reauthorization, or a completed connection removal and the dashboard's
    add and sign-in. The token endpoint answers with the same stable tokens,
    and the probe's authenticated request is accepted."""
    from hermes_cli import mcp_config, web_server
    from tools.mcp_dashboard_oauth import DashboardOAuthFlow
    from tools.mcp_oauth import force_interactive_oauth
    from tui_gateway.mcp_oauth_sessions import _probe_with_rollback

    def _authorize(name, config, *, connect_timeout):
        _stable_tokens_reissued()

    def _probe(name, config, connect_timeout=None, *, details=None):
        if entry == "desktop":
            # The desktop path gets its tokens during the probe.
            _stable_tokens_reissued()
        _stored_sign_in_accepted()
        return [("fake-tool", "A fake tool of the signed-in connection")]

    monkeypatch.setattr(mcp_config, "_authorize_oauth_server", _authorize)
    monkeypatch.setattr(mcp_config, "_probe_single_server", _probe)

    if entry == "cli":
        assert mcp_config._reauth_oauth_server(_SERVER, dict(_OAUTH_CONFIG)) is True
        return
    if entry == "removal_and_readd":
        assert mcp_config._remove_mcp_server_and_credentials(_SERVER) is True
        assert not _token_file(home).exists()
        assert _SERVER not in mcp_config._get_mcp_servers()
        assert mcp_config._save_mcp_server(_SERVER, dict(_OAUTH_CONFIG)) is True
    flow = DashboardOAuthFlow(
        flow_id=f"flow-srv-{entry}",
        server_name=_SERVER,
        profile=None,
        hermes_home=str(home),
        redirect_uri="https://agent.example/api/mcp/oauth/callback/srv",
    )
    if entry == "desktop":
        with force_interactive_oauth():
            _probe_with_rollback(_SERVER, dict(_OAUTH_CONFIG), str(home), flow, False)
    else:
        web_server._run_dashboard_mcp_oauth(flow, dict(_OAUTH_CONFIG))
    assert flow.status == "approved", flow.error


def _listed_connection_status() -> str:
    """The connection's status in the dashboard's authenticated server list."""
    from starlette.testclient import TestClient

    from hermes_cli.web_server import _SESSION_HEADER_NAME, _SESSION_TOKEN, app

    client = TestClient(app)
    client.headers[_SESSION_HEADER_NAME] = _SESSION_TOKEN
    response = client.get("/api/mcp/servers")
    assert response.status_code == 200, response.text
    [server] = [s for s in response.json()["servers"] if s["name"] == _SERVER]
    return server["connection_status"]


@pytest.mark.parametrize(
    "path, status", [("refresh", 400), ("refresh", 401), ("mcp", 401), ("mcp", 403)]
)
@pytest.mark.parametrize("entry", ["dashboard", "cli", "desktop", "removal_and_readd"])
def test_delayed_refusal_writer_keeps_new_owner_sign_in_connected(
    home, monkeypatch, entry, path, status
):
    asyncio.run(_authorized_with_stable_tokens())
    signed_in = _token_file(home).read_bytes()
    marker = HermesTokenStorage(_SERVER)._sign_in_rejected_path(
        hashlib.sha256(signed_in).hexdigest()
    ).name

    with _refusal_paused_before_its_marker(home, path, status):
        assert _token_file(home).read_bytes() == signed_in
        assert not _markers(home)
        _owner_signs_in_again(home, monkeypatch, entry)
    # Released after the owner's save, the writer publishes the marker of the
    # bytes it matched.

    assert [m.name for m in _markers(home)] == [marker]
    assert _stored_tokens(home) == ("fake-stable-access-token", "fake-stable-refresh-token")
    _stored_sign_in_accepted()
    assert _mcp_connection_status(_SERVER, _OAUTH_CONFIG) == "connected"
    if entry == "removal_and_readd":
        assert _listed_connection_status() == "connected"


# -- markers of long multibyte connection names fit the filename limit --------

# Connection names the dashboard's create path and the config save accept. In
# UTF-8 the CJK and accented-Latin names take 192 and 180 bytes: their token
# files fit the filename limit, and their markers must fit it too. The ASCII
# name is the control.
_ACCEPTED_NAMES = [
    pytest.param("a" * 128, id="ascii-control"),
    pytest.param("\u754c" * 64, id="cjk"),
    pytest.param("\u00e9" * 90, id="accented-latin"),
]


def _accepted_connection(name: str) -> HermesTokenStorage:
    """The owner adds the OAuth connection ``name`` as the dashboard does."""
    from hermes_cli import mcp_config
    from hermes_cli.web_models import MCPServerCreate
    from hermes_cli.web_server_mcp import _normalize_mcp_server_create

    created, config, _ = _normalize_mcp_server_create(
        MCPServerCreate(name=name, url=_URL, auth="oauth")
    )
    assert (created, config) == (name, _OAUTH_CONFIG)
    assert mcp_config._save_mcp_server(name, config) is True
    assert name in mcp_config._get_mcp_servers()
    return HermesTokenStorage(name)


def _fits_name_max(path: Path) -> bool:
    return len(os.fsencode(path.name)) <= os.pathconf(path.parent, "PC_NAME_MAX")


async def _owner_signs_in_to(name: str, prefix: str = "fake"):
    """The owner signs in to ``name``: the real provider's successful
    token-response handler saves fake tokens. Returns that provider."""
    from tools.mcp_oauth import force_interactive_oauth
    from tools.mcp_oauth_manager import MCPOAuthManager
    from tools.mcp_tool import sdk_httpx

    storage = HermesTokenStorage(name)
    await storage.set_client_info(OAuthClientInformationFull(client_id="fake-client-id"))
    storage.save_oauth_metadata(
        OAuthMetadata(
            issuer="https://auth.example.com",
            authorization_endpoint="https://auth.example.com/authorize",
            token_endpoint="https://auth.example.com/token",
        )
    )
    with force_interactive_oauth():  # the owner's own sign-in
        provider = MCPOAuthManager().get_or_build_provider(name, _URL, {})
        await provider._initialize()
        tokens = _fake_token(prefix).model_dump(exclude_none=True)
        await provider._handle_token_response(sdk_httpx().Response(200, json=tokens))
    assert _fits_name_max(storage._tokens_path())
    return provider


async def _refresh_refused(provider, status: int) -> None:
    """The provider refuses the refresh of the saved sign-in with ``status``."""
    from tools.mcp_tool import sdk_httpx

    request = await provider._refresh_token()
    response = sdk_httpx().Response(
        status, request=request, content=b'{"error": "invalid_grant"}'
    )
    assert await provider._handle_refresh_response(response) is False


def _profile_markers(home) -> list:
    """The names of the markers of every connection in the profile."""
    return sorted(p.name for p in (home / "mcp-tokens").glob("*.signin-rejected.*"))


@pytest.mark.parametrize("name", _ACCEPTED_NAMES)
@pytest.mark.asyncio
async def test_owner_sign_in_of_accepted_name_reads_connected(home, name):
    _accepted_connection(name)

    await _owner_signs_in_to(name)

    assert _mcp_connection_status(name, _OAUTH_CONFIG) == "connected"


@pytest.mark.parametrize("name", _ACCEPTED_NAMES)
@pytest.mark.asyncio
async def test_successful_refresh_of_accepted_name_saves_new_grant(home, name):
    from tools.mcp_tool import sdk_httpx

    storage = _accepted_connection(name)
    provider = await _owner_signs_in_to(name)
    request = await provider._refresh_token()
    assert b"fake-refresh-token" in request.content
    refreshed = _fake_token("fake-refreshed").model_dump(exclude_none=True)

    response = sdk_httpx().Response(200, request=request, json=refreshed)
    assert await provider._handle_refresh_response(response) is True

    tokens = await storage.get_tokens()
    assert tokens.access_token == "fake-refreshed-access-token"
    assert tokens.refresh_token == "fake-refreshed-refresh-token"
    assert _mcp_connection_status(name, _OAUTH_CONFIG) == "connected"


@pytest.mark.parametrize("name", _ACCEPTED_NAMES)
@pytest.mark.asyncio
async def test_refusal_of_accepted_name_is_recorded_and_removed_with_it(home, name):
    from hermes_cli import mcp_config

    storage = _accepted_connection(name)
    provider = await _owner_signs_in_to(name)
    refused = storage._tokens_path().read_bytes()

    await _refresh_refused(provider, 400)

    assert _mcp_connection_status(name, _OAUTH_CONFIG) == "needs_authorization"
    marker = storage._sign_in_rejected_path(hashlib.sha256(refused).hexdigest())
    assert storage._sign_in_rejected_paths() == [marker]
    assert _profile_markers(home) == [marker.name]
    assert _fits_name_max(marker)

    await _owner_signs_in_to(name, "fake-new")
    assert _mcp_connection_status(name, _OAUTH_CONFIG) == "connected"

    assert mcp_config._remove_mcp_server_and_credentials(name) is True
    assert _profile_markers(home) == []
    assert not storage._tokens_path().exists()
    assert name not in mcp_config._get_mcp_servers()


@pytest.mark.asyncio
async def test_markers_of_names_differing_only_in_last_character_stay_apart(home):
    from hermes_cli import mcp_config

    first, second = "\u754c" * 64, "\u754c" * 63 + "\u4e16"
    first_storage, second_storage = _accepted_connection(first), _accepted_connection(second)
    provider = await _owner_signs_in_to(first)
    await _owner_signs_in_to(second)
    refused = first_storage._tokens_path().read_bytes()

    await _refresh_refused(provider, 401)

    assert _mcp_connection_status(first, _OAUTH_CONFIG) == "needs_authorization"
    assert _mcp_connection_status(second, _OAUTH_CONFIG) == "connected"
    marker = first_storage._sign_in_rejected_path(hashlib.sha256(refused).hexdigest())
    assert first_storage._sign_in_rejected_paths() == [marker]
    assert second_storage._sign_in_rejected_paths() == []

    assert mcp_config._remove_mcp_server_and_credentials(second) is True

    assert _profile_markers(home) == [marker.name]
    assert _mcp_connection_status(first, _OAUTH_CONFIG) == "needs_authorization"
