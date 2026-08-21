"""Official OAuth defaults for Anthropic's hosted Claude Design MCP.

Claude Design uses a fixed public client and a hosted manual-code callback.
The generic Hermes MCP OAuth engine still owns PKCE, token exchange, refresh,
storage, rollback, and probing; this module only supplies the provider facts
that cannot be discovered reliably through the provider's protected metadata.
"""

from __future__ import annotations

from typing import Any


CLAUDE_DESIGN_SERVER_NAME = "claude-design"
CLAUDE_DESIGN_MCP_URL = "https://api.anthropic.com/v1/design/mcp"
CLAUDE_DESIGN_AUTHORIZE_URL = "https://claude.com/cai/oauth/authorize"
CLAUDE_DESIGN_TOKEN_URL = "https://platform.claude.com/v1/oauth/token"
CLAUDE_DESIGN_REDIRECT_URI = "https://platform.claude.com/oauth/code/callback"
CLAUDE_DESIGN_CLIENT_ID = "59637612-477b-4836-a601-b0589eda7704"
CLAUDE_DESIGN_SCOPES = "user:design:read user:design:write"
CLAUDE_DESIGN_FLOW_TTL_SECONDS = 15 * 60


def apply_claude_design_oauth_defaults(
    config: dict[str, Any],
    *,
    server_name: str,
    server_url: str,
) -> bool:
    """Apply the fixed official public-client contract to an exact match."""
    if (
        server_name != CLAUDE_DESIGN_SERVER_NAME
        or server_url != CLAUDE_DESIGN_MCP_URL
    ):
        return False
    config.update(
        {
            "client_id": CLAUDE_DESIGN_CLIENT_ID,
            "client_name": "Claude Design",
            "redirect_uri": CLAUDE_DESIGN_REDIRECT_URI,
            "scope": CLAUDE_DESIGN_SCOPES,
            "token_endpoint_auth_method": "none",
            "application_type": "native",
        }
    )
    return True


def seed_claude_design_oauth_metadata(storage: Any) -> None:
    """Seed exact metadata so the native SDK skips the retired fallback."""
    from mcp.shared.auth import OAuthMetadata

    metadata = OAuthMetadata.model_validate(
        {
            "issuer": "https://claude.com",
            "authorization_endpoint": CLAUDE_DESIGN_AUTHORIZE_URL,
            "token_endpoint": CLAUDE_DESIGN_TOKEN_URL,
            "scopes_supported": CLAUDE_DESIGN_SCOPES.split(),
            "response_types_supported": ["code"],
            "grant_types_supported": ["authorization_code", "refresh_token"],
            "token_endpoint_auth_methods_supported": ["none"],
            "code_challenge_methods_supported": ["S256"],
        }
    )
    existing = storage.load_oauth_metadata()
    if (
        existing is None
        or existing.model_dump(mode="json", exclude_none=True)
        != metadata.model_dump(mode="json", exclude_none=True)
    ):
        storage.save_oauth_metadata(metadata)
