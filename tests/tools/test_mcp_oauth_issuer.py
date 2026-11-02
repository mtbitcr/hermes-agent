"""Google sign-in must accept the issuer with or without a final slash (RFC 3986 6.2.3)."""

import asyncio
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import parse_qs, urlsplit

import pytest

pytest.importorskip("mcp.client.auth.oauth2", reason="MCP SDK with OAuth support required")

import mcp.client.auth.oauth2 as sdk_oauth2  # noqa: E402
import mcp.client.auth.utils as sdk_utils  # noqa: E402
from mcp.client.auth import OAuthFlowError  # noqa: E402

from tools import mcp_oauth  # noqa: E402
from tools.mcp_tool import sdk_httpx  # noqa: E402

httpx = sdk_httpx()

# The SDK's own validator, captured before anything in this file can touch it.
SDK_VALIDATOR = sdk_utils.validate_metadata_issuer


def _sdk_error(oauth_metadata, expected_issuer):
    """The error the SDK's original validator raises for this pair."""
    with pytest.raises(OAuthFlowError) as raised:
        SDK_VALIDATOR(oauth_metadata, expected_issuer)
    return raised.value


MCP_URL = "https://gmail.mcp.example.com/mcp"
PRM_URL = "https://gmail.mcp.example.com/.well-known/oauth-protected-resource/mcp"
OIDC_URL = "https://accounts.google.com/.well-known/openid-configuration"
AUTH_ENDPOINT = "https://accounts.google.com/o/oauth2/v2/auth"
ASM_ISSUER = "https://accounts.google.com"
PRM_ISSUER = "https://accounts.google.com/"


@pytest.fixture(autouse=True)
def _fresh_sdk_loading(monkeypatch):
    """Start every test from a never-loaded module so the loader runs inside the test."""
    monkeypatch.setattr(sdk_oauth2, "validate_metadata_issuer", SDK_VALIDATOR)
    monkeypatch.setattr(mcp_oauth, "_SDK_CLASSES", {})
    monkeypatch.setattr(mcp_oauth, "_SDK_LOAD_FAILED", False)
    for name in (
        "OAuthClientProvider",
        "OAuthClientInformationFull",
        "OAuthClientMetadata",
        "OAuthMetadata",
        "OAuthToken",
        "HermesOAuthClientProvider",
    ):
        monkeypatch.setattr(mcp_oauth, name, None)


# ---------------------------------------------------------------------------
# Driving the SDK's browser flow by hand, with canned discovery documents
# ---------------------------------------------------------------------------


class _ReachedCallback(Exception):
    """Raised by the fake callback handler: the flow got past the issuer check."""


def _protected_resource_metadata():
    return {"resource": MCP_URL, "authorization_servers": [PRM_ISSUER]}


def _authorization_server_metadata(issuer):
    return {
        "issuer": issuer,
        "authorization_endpoint": AUTH_ENDPOINT,
        "token_endpoint": "https://oauth2.googleapis.com/token",
        "scopes_supported": ["openid", "email", "profile"],
        "response_types_supported": ["code", "token", "id_token"],
        "grant_types_supported": ["authorization_code", "refresh_token"],
        "token_endpoint_auth_methods_supported": ["client_secret_post", "client_secret_basic"],
        "code_challenge_methods_supported": ["plain", "S256"],
    }


def _respond(request, routes):
    url = str(request.url)
    if url == MCP_URL:
        return httpx.Response(401, request=request)
    status, body = routes.get(url, (404, {}))
    return httpx.Response(status, json=body, request=request)


async def _drive(provider, routes, requested):
    flow = provider.async_auth_flow(httpx.Request("POST", MCP_URL))
    try:
        request = await flow.__anext__()
        while True:
            requested.append(str(request.url))
            request = await flow.asend(_respond(request, routes))
    except (_ReachedCallback, OAuthFlowError) as outcome:
        return outcome


def _run_flow(tmp_path, monkeypatch, asm_issuer=ASM_ISSUER):
    redirects = []
    requested = []
    validator_calls = []

    async def redirect_handler(url):
        redirects.append(url)

    async def callback_handler():
        raise _ReachedCallback

    assert mcp_oauth._ensure_sdk_loaded() is True
    storage = mcp_oauth.HermesTokenStorage("google-gmail-work", hermes_home=tmp_path)
    cfg = {"_resolved_port": 8765, "client_id": "fake-client-id"}
    client_metadata = mcp_oauth._build_client_metadata(cfg)
    mcp_oauth._maybe_preregister_client(storage, cfg, client_metadata)
    provider = mcp_oauth._get_hermes_oauth_provider_class()(
        server_url=MCP_URL,
        client_metadata=client_metadata,
        storage=storage,
        redirect_handler=redirect_handler,
        callback_handler=callback_handler,
    )

    installed = sdk_oauth2.validate_metadata_issuer

    def recording(oauth_metadata, expected_issuer):
        validator_calls.append((oauth_metadata, expected_issuer))
        return installed(oauth_metadata, expected_issuer)

    monkeypatch.setattr(sdk_oauth2, "validate_metadata_issuer", recording)
    routes = {
        PRM_URL: (200, _protected_resource_metadata()),
        OIDC_URL: (200, _authorization_server_metadata(asm_issuer)),
    }
    outcome = asyncio.run(_drive(provider, routes, requested))
    return SimpleNamespace(
        outcome=outcome,
        redirects=redirects,
        requested=requested,
        validator_calls=validator_calls,
    )


def test_flow_accepts_issuer_with_and_without_final_slash(tmp_path, monkeypatch):
    run = _run_flow(tmp_path, monkeypatch)

    received = [(str(meta.issuer), expected) for meta, expected in run.validator_calls]
    print("strings received by the issuer validator:", received)
    assert received == [(ASM_ISSUER, PRM_ISSUER)]

    assert isinstance(run.outcome, _ReachedCallback), f"flow stopped early: {run.outcome!r}"
    assert len(run.redirects) == 1
    redirect_url = run.redirects[0]
    assert redirect_url.startswith(AUTH_ENDPOINT)
    query = parse_qs(urlsplit(redirect_url).query)
    assert query["client_id"] == ["fake-client-id"]
    assert query["state"]
    assert query["code_challenge"]


@pytest.mark.parametrize(
    "asm_issuer",
    [
        pytest.param("https://accounts.example.org", id="other-host"),
        pytest.param("http://accounts.google.com", id="other-scheme"),
        pytest.param("https://accounts.google.com/other", id="other-path"),
        pytest.param("https://accounts.google.com/?x=1", id="query"),
        pytest.param("https://user@accounts.google.com", id="userinfo"),
    ],
)
def test_flow_still_rejects_a_different_issuer(tmp_path, monkeypatch, asm_issuer):
    run = _run_flow(tmp_path, monkeypatch, asm_issuer)

    assert isinstance(run.outcome, OAuthFlowError), f"flow ended with {run.outcome!r}"
    assert run.redirects == []
    assert len(run.validator_calls) == 1

    sdk_error = _sdk_error(*run.validator_calls[0])
    assert type(run.outcome) is type(sdk_error)
    assert str(run.outcome) == str(sdk_error)


# ---------------------------------------------------------------------------
# The installed validator, called directly
# ---------------------------------------------------------------------------


@pytest.fixture
def installed():
    assert mcp_oauth._ensure_sdk_loaded() is True
    return sdk_oauth2.validate_metadata_issuer


ACCEPTED = [
    pytest.param("https://accounts.google.com", "https://accounts.google.com", id="identical"),
    pytest.param("https://accounts.google.com/", "https://accounts.google.com/", id="identical-slash"),
    pytest.param("https://login.example.com/tenant", "https://login.example.com/tenant", id="identical-path"),
    pytest.param("http://localhost:8080", "http://localhost:8080", id="identical-http"),
    pytest.param("not a url", "not a url", id="identical-opaque"),
    pytest.param("https://accounts.google.com", "https://accounts.google.com/", id="google"),
    pytest.param("https://accounts.google.com/", "https://accounts.google.com", id="google-reversed"),
    pytest.param("https://auth.example.com:8443", "https://auth.example.com:8443/", id="port"),
    pytest.param("https://auth.example.com:8443/", "https://auth.example.com:8443", id="port-reversed"),
    pytest.param("https://[::1]:8443", "https://[::1]:8443/", id="ipv6"),
]

REJECTED = [
    pytest.param("https://accounts.example.org", "https://accounts.google.com/", id="other-host"),
    pytest.param("https://accounts.google.com", "https://accounts.example.org", id="other-host-no-slash"),
    pytest.param("https://Accounts.Google.com", "https://accounts.google.com/", id="host-case"),
    pytest.param("https://accounts.google.com/", "https://Accounts.Google.com", id="host-case-reversed"),
    pytest.param("https://accounts.google.com:443", "https://accounts.google.com/", id="explicit-443"),
    pytest.param("https://accounts.google.com/", "https://accounts.google.com:443", id="explicit-443-reversed"),
    pytest.param("https://accounts.google.com:8443", "https://accounts.google.com/", id="other-port"),
    pytest.param("https://accounts.google.com", "https://accounts.google.com//", id="doubled-slash"),
    pytest.param("https://accounts.google.com/", "https://accounts.google.com//", id="doubled-slash-after-slash"),
    pytest.param("https://login.example.com/tenant", "https://login.example.com/tenant/", id="tenant-slash"),
    pytest.param("https://login.example.com/tenant/", "https://login.example.com/tenant", id="tenant-slash-reversed"),
    pytest.param("https://accounts.google.com/other", "https://accounts.google.com/", id="other-path"),
    pytest.param("http://accounts.google.com", "http://accounts.google.com/", id="http-slash"),
    pytest.param("http://accounts.google.com/", "http://accounts.google.com", id="http-slash-reversed"),
    pytest.param("http://accounts.google.com", "https://accounts.google.com/", id="other-scheme"),
    pytest.param("https://accounts.google.com", "http://accounts.google.com/", id="other-scheme-reversed"),
    pytest.param("https://accounts.google.com/?x=1", "https://accounts.google.com/", id="query"),
    pytest.param("https://accounts.google.com?x=1", "https://accounts.google.com/?x=1", id="query-both"),
    pytest.param("https://accounts.google.com?", "https://accounts.google.com/", id="bare-question-mark"),
    pytest.param("https://accounts.google.com/", "https://accounts.google.com?", id="bare-question-mark-reversed"),
    pytest.param("https://accounts.google.com?", "https://accounts.google.com?/", id="bare-question-mark-both"),
    pytest.param("https://accounts.google.com", "https://accounts.google.com/?", id="slash-then-question-mark"),
    pytest.param("https://accounts.google.com#frag", "https://accounts.google.com/#frag", id="fragment"),
    pytest.param("https://accounts.google.com#", "https://accounts.google.com/", id="bare-hash"),
    pytest.param("https://accounts.google.com/", "https://accounts.google.com#", id="bare-hash-reversed"),
    pytest.param("https://accounts.google.com#", "https://accounts.google.com#/", id="bare-hash-both"),
    pytest.param("https://accounts.google.com", "https://accounts.google.com/#", id="slash-then-hash"),
    pytest.param("https://user@accounts.google.com", "https://accounts.google.com/", id="userinfo-issuer"),
    pytest.param("https://accounts.google.com", "https://user@accounts.google.com/", id="userinfo-expected"),
    pytest.param("https://user@accounts.google.com", "https://user@accounts.google.com/", id="userinfo-both"),
    pytest.param("https://user:pw@accounts.google.com/", "https://user:pw@accounts.google.com", id="userinfo-password"),
    pytest.param("https://@accounts.google.com", "https://@accounts.google.com/", id="empty-userinfo"),
    pytest.param("https://", "https:///", id="no-host"),
    pytest.param("", "/", id="empty-issuer"),
    pytest.param("https://accounts.google.com", "", id="empty-expected"),
    pytest.param("https://[::1", "https://[::1/", id="malformed-ipv6"),
    pytest.param("https://[::1/", "https://[::1", id="malformed-ipv6-reversed"),
]


@pytest.mark.parametrize(("issuer", "expected"), ACCEPTED)
def test_installed_validator_accepts(installed, issuer, expected):
    assert installed(SimpleNamespace(issuer=issuer), expected) is None


@pytest.mark.parametrize(("issuer", "expected"), REJECTED)
def test_installed_validator_rejects_with_the_sdk_message(installed, issuer, expected):
    metadata = SimpleNamespace(issuer=issuer)

    with pytest.raises(OAuthFlowError) as ours:
        installed(metadata, expected)

    sdk_error = _sdk_error(metadata, expected)
    assert type(ours.value) is type(sdk_error)
    assert str(ours.value) == str(sdk_error)


def test_installed_validator_treats_a_malformed_url_as_a_mismatch(installed):
    malformed = SimpleNamespace(issuer="https://[::1")

    with pytest.raises(OAuthFlowError) as ours:
        installed(malformed, "https://[::1/")

    assert str(ours.value) == str(_sdk_error(malformed, "https://[::1/"))
    # The same pair with a well-formed host is the accepted slash difference.
    assert installed(SimpleNamespace(issuer="https://[::1]"), "https://[::1]/") is None


# ---------------------------------------------------------------------------
# How the validator gets into the SDK
# ---------------------------------------------------------------------------


def test_module_still_imports_without_the_sdk():
    repo_root = Path(__file__).resolve().parents[2]
    proc = subprocess.run(
        [sys.executable, "-c", "import sys; sys.modules['mcp'] = None; import tools.mcp_oauth"],
        capture_output=True,
        text=True,
        timeout=120,
        cwd=repo_root,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
    )
    assert proc.returncode == 0, proc.stderr


def test_loader_installs_the_validator():
    assert sdk_oauth2.validate_metadata_issuer is SDK_VALIDATOR

    assert mcp_oauth._ensure_sdk_loaded() is True

    assert sdk_oauth2.validate_metadata_issuer is mcp_oauth._validate_metadata_issuer


def test_loading_again_does_not_wrap_or_replace_the_validator(monkeypatch):
    assert mcp_oauth._ensure_sdk_loaded() is True
    first = sdk_oauth2.validate_metadata_issuer
    assert first is mcp_oauth._validate_metadata_issuer

    assert mcp_oauth._ensure_sdk_loaded() is True
    assert sdk_oauth2.validate_metadata_issuer is first

    monkeypatch.setattr(mcp_oauth, "_SDK_CLASSES", {})
    assert mcp_oauth._ensure_sdk_loaded() is True
    assert sdk_oauth2.validate_metadata_issuer is first
    assert not hasattr(sdk_oauth2.validate_metadata_issuer, "__wrapped__")


def test_loader_leaves_a_validator_someone_else_bound(monkeypatch):
    def somebody_elses(oauth_metadata, expected_issuer):
        return None

    monkeypatch.setattr(sdk_oauth2, "validate_metadata_issuer", somebody_elses)

    assert mcp_oauth._ensure_sdk_loaded() is True

    assert sdk_oauth2.validate_metadata_issuer is somebody_elses


def test_loader_installs_nothing_when_the_sdk_has_no_such_name(monkeypatch):
    monkeypatch.delattr(sdk_oauth2, "validate_metadata_issuer")

    assert mcp_oauth._ensure_sdk_loaded() is True

    assert not hasattr(sdk_oauth2, "validate_metadata_issuer")


def test_sdk_utils_keeps_its_own_validator():
    assert mcp_oauth._ensure_sdk_loaded() is True

    assert sdk_utils.validate_metadata_issuer is SDK_VALIDATOR
    assert sdk_utils.validate_metadata_issuer is not mcp_oauth._validate_metadata_issuer
    assert sdk_utils.validate_metadata_issuer is not sdk_oauth2.validate_metadata_issuer
