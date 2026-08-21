"""Contract tests for Claude Design's first-party OAuth defaults."""

from __future__ import annotations

import asyncio
import json
import stat


def test_defaults_apply_only_to_the_exact_official_connection():
    from tools.claude_design_oauth import (
        CLAUDE_DESIGN_CLIENT_ID,
        CLAUDE_DESIGN_MCP_URL,
        CLAUDE_DESIGN_REDIRECT_URI,
        CLAUDE_DESIGN_SCOPES,
        apply_claude_design_oauth_defaults,
    )

    config: dict[str, object] = {"client_name": "stale"}
    assert apply_claude_design_oauth_defaults(
        config,
        server_name="claude-design",
        server_url=CLAUDE_DESIGN_MCP_URL,
    )
    assert config == {
        "client_id": CLAUDE_DESIGN_CLIENT_ID,
        "client_name": "Claude Design",
        "redirect_uri": CLAUDE_DESIGN_REDIRECT_URI,
        "scope": CLAUDE_DESIGN_SCOPES,
        "token_endpoint_auth_method": "none",
        "application_type": "native",
    }

    other = {"client_name": "keep-me"}
    assert not apply_claude_design_oauth_defaults(
        other,
        server_name="other",
        server_url=CLAUDE_DESIGN_MCP_URL,
    )
    assert other == {"client_name": "keep-me"}


def test_metadata_seed_uses_official_endpoints_and_private_storage(
    tmp_path, monkeypatch
):
    from tools.claude_design_oauth import (
        CLAUDE_DESIGN_AUTHORIZE_URL,
        CLAUDE_DESIGN_TOKEN_URL,
        seed_claude_design_oauth_metadata,
    )
    from tools.mcp_oauth import HermesTokenStorage

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    storage = HermesTokenStorage("claude-design")
    seed_claude_design_oauth_metadata(storage)

    metadata = storage.load_oauth_metadata()
    assert metadata is not None
    assert str(metadata.authorization_endpoint) == CLAUDE_DESIGN_AUTHORIZE_URL
    assert str(metadata.token_endpoint) == CLAUDE_DESIGN_TOKEN_URL
    assert metadata.scopes_supported == [
        "user:design:read",
        "user:design:write",
    ]
    assert stat.S_IMODE(storage._meta_path().stat().st_mode) == 0o600


def test_native_manager_owns_client_registration_and_metadata(tmp_path, monkeypatch):
    from tools.claude_design_oauth import (
        CLAUDE_DESIGN_CLIENT_ID,
        CLAUDE_DESIGN_MCP_URL,
        CLAUDE_DESIGN_REDIRECT_URI,
        CLAUDE_DESIGN_TOKEN_URL,
    )
    from tools.mcp_dashboard_oauth import DashboardOAuthFlow, dashboard_oauth_flow
    from tools.mcp_oauth import HermesTokenStorage
    from tools.mcp_oauth_manager import MCPOAuthManager

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    stale = HermesTokenStorage("claude-design")
    stale._client_info_path().parent.mkdir(parents=True)
    stale._client_info_path().write_text(
        '{"client_id":"retired-fallback-client"}',
        encoding="utf-8",
    )
    stale._tokens_path().write_text(
        '{"access_token":"stale","token_type":"Bearer"}',
        encoding="utf-8",
    )
    flow = DashboardOAuthFlow(
        flow_id="design-flow",
        server_name="claude-design",
        profile="raphael-designer",
        hermes_home=str(tmp_path),
        redirect_uri=CLAUDE_DESIGN_REDIRECT_URI,
    )
    manager = MCPOAuthManager()
    with dashboard_oauth_flow(flow):
        provider = manager.get_or_build_provider(
            "claude-design",
            CLAUDE_DESIGN_MCP_URL,
            None,
        )

    assert provider is not None
    storage = HermesTokenStorage("claude-design")
    client = asyncio.run(storage.get_client_info())
    metadata = storage.load_oauth_metadata()
    assert client is not None
    assert client.client_id == CLAUDE_DESIGN_CLIENT_ID
    assert [str(uri) for uri in client.redirect_uris] == [
        CLAUDE_DESIGN_REDIRECT_URI
    ]
    assert metadata is not None
    assert str(metadata.token_endpoint) == CLAUDE_DESIGN_TOKEN_URL


def test_native_manager_matches_claude_code_token_exchange(tmp_path, monkeypatch):
    from tools.claude_design_oauth import (
        CLAUDE_DESIGN_CLIENT_ID,
        CLAUDE_DESIGN_MCP_URL,
        CLAUDE_DESIGN_REDIRECT_URI,
        CLAUDE_DESIGN_TOKEN_URL,
    )
    from tools.mcp_dashboard_oauth import DashboardOAuthFlow, dashboard_oauth_flow
    from tools.mcp_oauth import HermesTokenStorage
    from tools.mcp_oauth_manager import MCPOAuthManager

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    flow = DashboardOAuthFlow(
        flow_id="design-exchange",
        server_name="claude-design",
        profile="raphael-designer",
        hermes_home=str(tmp_path),
        redirect_uri=CLAUDE_DESIGN_REDIRECT_URI,
    )
    manager = MCPOAuthManager()
    with dashboard_oauth_flow(flow):
        provider = manager.get_or_build_provider(
            "claude-design",
            CLAUDE_DESIGN_MCP_URL,
            None,
        )
        assert provider is not None
        storage = HermesTokenStorage("claude-design")
        provider.context.oauth_metadata = storage.load_oauth_metadata()
        provider.context.client_info = asyncio.run(storage.get_client_info())
        assert provider.context.oauth_metadata is not None
        assert provider.context.client_info is not None

        async def _complete_flow():
            pending = asyncio.create_task(provider._perform_authorization())
            await flow.wait_for_authorization_url()
            assert flow.expected_state is not None
            flow.deliver_callback(
                code="one-time-code",
                state=flow.expected_state,
                error=None,
            )
            return await pending, flow.expected_state

        request, state = asyncio.run(_complete_flow())

    payload = json.loads(request.content)
    assert str(request.url) == CLAUDE_DESIGN_TOKEN_URL
    assert request.headers["content-type"] == "application/json"
    assert payload == {
        "grant_type": "authorization_code",
        "code": "one-time-code",
        "redirect_uri": CLAUDE_DESIGN_REDIRECT_URI,
        "client_id": CLAUDE_DESIGN_CLIENT_ID,
        "code_verifier": payload["code_verifier"],
        "state": state,
    }
    assert len(payload["code_verifier"]) == 128


def test_native_manager_matches_claude_code_refresh(tmp_path, monkeypatch):
    from mcp.shared.auth import OAuthToken
    from tools.claude_design_oauth import (
        CLAUDE_DESIGN_CLIENT_ID,
        CLAUDE_DESIGN_MCP_URL,
        CLAUDE_DESIGN_REDIRECT_URI,
        CLAUDE_DESIGN_SCOPES,
        CLAUDE_DESIGN_TOKEN_URL,
    )
    from tools.mcp_dashboard_oauth import DashboardOAuthFlow, dashboard_oauth_flow
    from tools.mcp_oauth import HermesTokenStorage
    from tools.mcp_oauth_manager import MCPOAuthManager

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    flow = DashboardOAuthFlow(
        flow_id="design-refresh",
        server_name="claude-design",
        profile="raphael-designer",
        hermes_home=str(tmp_path),
        redirect_uri=CLAUDE_DESIGN_REDIRECT_URI,
    )
    with dashboard_oauth_flow(flow):
        provider = MCPOAuthManager().get_or_build_provider(
            "claude-design",
            CLAUDE_DESIGN_MCP_URL,
            None,
        )
    assert provider is not None
    storage = HermesTokenStorage("claude-design")
    provider.context.oauth_metadata = storage.load_oauth_metadata()
    provider.context.client_info = asyncio.run(storage.get_client_info())
    assert provider.context.oauth_metadata is not None
    assert provider.context.client_info is not None
    provider.context.current_tokens = OAuthToken(
        access_token="expired-access",
        refresh_token="refresh-token",
        token_type="Bearer",
    )

    request = asyncio.run(provider._refresh_token())

    assert str(request.url) == CLAUDE_DESIGN_TOKEN_URL
    assert request.headers["content-type"] == "application/json"
    assert json.loads(request.content) == {
        "grant_type": "refresh_token",
        "refresh_token": "refresh-token",
        "client_id": CLAUDE_DESIGN_CLIENT_ID,
        "scope": CLAUDE_DESIGN_SCOPES,
    }
