"""Contract tests for Claude Design's first-party OAuth defaults."""

from __future__ import annotations

import asyncio
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
