"""Managed, revocable and expiring machine credentials for Raphael dashboards.

The providers below share one audited metadata registry while retaining
separate fixed token families and scopes for the Workspace work view, the
owner-only recommendations inbox, the Connections Center, and Automations.

What they are
-------------
Four service-to-service providers verify
``Authorization: Bearer <token_id>.<secret>`` against the same on-disk
registry in :mod:`token_store`. ``hrw1_`` tokens carry only
``kanban.read`` for the fixed Workspace GET surface; ``hrr1_`` tokens carry
only ``kanban:recommendations:read`` for the owner inbox; ``hrc1_`` tokens
carry only ``mcp.connections.manage`` for the exact Connections API contour.
``hra1_`` tokens carry only ``cron.automations.manage`` for list/history
and reversible pause/resume.
A token from one family is rejected by the other providers before route scope
enforcement.

Lifecycle and isolation
-----------------------
All managed families support issue/list/revoke, bounded expiry and immediate
revocation through :mod:`token_store`. Every request verifies fresh against
disk (no cache), so ``revoke`` takes effect on the next request. They share
the existing ``supports_token`` / ``verify_token`` + ``token_auth``
middleware seam and ``TokenPrincipal`` contract, while exact scopes keep the
four surfaces mutually unusable.

There is no login, cookie, session, or refresh: only the token capability is
implemented.

Configuration
-------------
Nothing goes in ``.env`` or ``config.yaml`` — there is no operator-supplied
secret to provision. Credentials are minted locally via:

    hermes kanban-workspace-token issue --surface workspace --out <path>
    hermes kanban-workspace-token issue --surface recommendations --ttl-hours 8 --out <path>
    hermes kanban-workspace-token issue --surface connections --out <path>
    hermes kanban-workspace-token issue --surface automations --out <path>

The token family, principal, scope, grant and lifetime ceiling are fixed
constants (see :mod:`token_store`), never operator-editable configuration.
All providers always register: an empty or missing registry denies every
``verify_token`` call (fail-closed), so there is no "unset env
var" skip-and-record-a-reason path like the static-secret plugins have — the
CLI is what actually brings the credential into existence.
"""
from __future__ import annotations

import logging
from typing import Optional

from hermes_cli.dashboard_auth import TokenPrincipal
from hermes_cli.dashboard_auth.base import (
    DashboardAuthProvider,
    LoginStart,
    ProviderError,
    Session,
)

from plugins.dashboard_auth.raphael_workspace import token_store
from plugins.dashboard_auth.raphael_workspace.cli import (
    register_cli,
    workspace_token_command,
)

logger = logging.getLogger(__name__)

# Re-exported so callers (the Kanban plugin API, tests) reference one source
# of truth for the fixed grant instead of re-typing literals.
PRINCIPAL = token_store.PRINCIPAL
SCOPE = token_store.SCOPE
PROJECT = token_store.PROJECT
BOARD = token_store.BOARD
TOKEN_PREFIX = token_store.TOKEN_PREFIX
GRANT = token_store.GRANT
RECOMMENDATIONS_PRINCIPAL = token_store.RECOMMENDATIONS_PRINCIPAL
RECOMMENDATIONS_SCOPE = token_store.RECOMMENDATIONS_SCOPE
RECOMMENDATIONS_PROJECT = token_store.RECOMMENDATIONS_PROJECT
RECOMMENDATIONS_BOARD = token_store.RECOMMENDATIONS_BOARD
RECOMMENDATIONS_TOKEN_PREFIX = token_store.RECOMMENDATIONS_TOKEN_PREFIX
RECOMMENDATIONS_GRANT = token_store.RECOMMENDATIONS_GRANT
CONNECTIONS_PRINCIPAL = token_store.CONNECTIONS_PRINCIPAL
CONNECTIONS_SCOPE = token_store.CONNECTIONS_SCOPE
CONNECTIONS_PROJECT = token_store.CONNECTIONS_PROJECT
CONNECTIONS_BOARD = token_store.CONNECTIONS_BOARD
CONNECTIONS_TOKEN_PREFIX = token_store.CONNECTIONS_TOKEN_PREFIX
CONNECTIONS_GRANT = token_store.CONNECTIONS_GRANT
AUTOMATIONS_PRINCIPAL = token_store.AUTOMATIONS_PRINCIPAL
AUTOMATIONS_SCOPE = token_store.AUTOMATIONS_SCOPE
AUTOMATIONS_PROJECT = token_store.AUTOMATIONS_PROJECT
AUTOMATIONS_BOARD = token_store.AUTOMATIONS_BOARD
AUTOMATIONS_TOKEN_PREFIX = token_store.AUTOMATIONS_TOKEN_PREFIX
AUTOMATIONS_GRANT = token_store.AUTOMATIONS_GRANT

__all__ = [
    "PRINCIPAL",
    "SCOPE",
    "PROJECT",
    "BOARD",
    "TOKEN_PREFIX",
    "GRANT",
    "RECOMMENDATIONS_PRINCIPAL",
    "RECOMMENDATIONS_SCOPE",
    "RECOMMENDATIONS_PROJECT",
    "RECOMMENDATIONS_BOARD",
    "RECOMMENDATIONS_TOKEN_PREFIX",
    "RECOMMENDATIONS_GRANT",
    "CONNECTIONS_PRINCIPAL",
    "CONNECTIONS_SCOPE",
    "CONNECTIONS_PROJECT",
    "CONNECTIONS_BOARD",
    "CONNECTIONS_TOKEN_PREFIX",
    "CONNECTIONS_GRANT",
    "AUTOMATIONS_PRINCIPAL",
    "AUTOMATIONS_SCOPE",
    "AUTOMATIONS_PROJECT",
    "AUTOMATIONS_BOARD",
    "AUTOMATIONS_TOKEN_PREFIX",
    "AUTOMATIONS_GRANT",
    "WorkspaceReadTokenProvider",
    "RecommendationsReadTokenProvider",
    "ConnectionsManageTokenProvider",
    "AutomationsManageTokenProvider",
    "register",
]


class WorkspaceReadTokenProvider(DashboardAuthProvider):
    """Non-interactive provider backed by the CLI-managed token registry."""

    name = "raphael-workspace-token"
    display_name = "Raphael Workspace (read-only Kanban credential)"
    token_prefix = TOKEN_PREFIX
    expected_principal = PRINCIPAL
    expected_scope = SCOPE
    supports_token = True
    supports_session = False

    def verify_token(self, *, token: str) -> Optional[TokenPrincipal]:
        """Look up ``<token_id>.<secret>`` against the on-disk registry.

        Returns ``None`` for a malformed/unknown/wrong/revoked/expired
        token — the seam falls through / fails closed. Raises
        ``ProviderError`` only when the registry itself cannot be trusted
        (missing digest, malformed JSON, unsafe file permissions); the seam
        then answers 503 rather than treating a broken store as "invalid
        credentials".
        """
        if not token.startswith(self.token_prefix) or "." not in token:
            return None
        token_id, _, secret = token.partition(".")
        if not token_id or not secret:
            return None
        try:
            record = token_store.verify(token_id, secret)
        except token_store.TokenStoreError as exc:
            raise ProviderError(str(exc)) from exc
        if (
            record is None
            or record.principal != self.expected_principal
            or record.scope != self.expected_scope
        ):
            return None
        return TokenPrincipal(
            principal=record.principal,
            provider=self.name,
            scopes=(record.scope,),
            credential_id=record.token_id,
        )

    # ---- interactive methods: unsupported (service credential only) -------

    def _no_interactive(self) -> NotImplementedError:
        return NotImplementedError(
            f"{type(self).__name__} is a non-interactive service credential; "
            "there is no login flow."
        )

    def start_login(self, *, redirect_uri: str) -> LoginStart:
        raise self._no_interactive()

    def complete_login(
        self, *, code: str, state: str, code_verifier: str, redirect_uri: str
    ) -> Session:
        raise self._no_interactive()

    def verify_session(self, *, access_token: str) -> Optional[Session]:
        # Not a cookie-session provider — stacks harmlessly instead of
        # raising, matching the drain/recommendations convention.
        return None

    def refresh_session(self, *, refresh_token: str) -> Session:
        raise self._no_interactive()

    def revoke_session(self, *, refresh_token: str) -> None:
        return None


class RecommendationsReadTokenProvider(WorkspaceReadTokenProvider):
    """Managed, revocable credential for the owner recommendations inbox."""

    name = "raphael-recommendations-token"
    display_name = "Raphael Recommendations (JIT read credential)"
    token_prefix = RECOMMENDATIONS_TOKEN_PREFIX
    expected_principal = RECOMMENDATIONS_PRINCIPAL
    expected_scope = RECOMMENDATIONS_SCOPE


class ConnectionsManageTokenProvider(WorkspaceReadTokenProvider):
    """Managed credential for the fixed owner Connections API contour."""

    name = "raphael-connections-token"
    display_name = "Raphael Connections (managed service credential)"
    token_prefix = CONNECTIONS_TOKEN_PREFIX
    expected_principal = CONNECTIONS_PRINCIPAL
    expected_scope = CONNECTIONS_SCOPE


class AutomationsManageTokenProvider(WorkspaceReadTokenProvider):
    """Managed credential for the fixed owner Automations API contour."""

    name = "raphael-automations-token"
    display_name = "Raphael Automations (managed service credential)"
    token_prefix = AUTOMATIONS_TOKEN_PREFIX
    expected_principal = AUTOMATIONS_PRINCIPAL
    expected_scope = AUTOMATIONS_SCOPE


# ---------------------------------------------------------------------------
# Plugin entry point
# ---------------------------------------------------------------------------


def register(ctx) -> None:
    """Plugin entry — registers the provider and the CLI command tree.

    Always registers (see module docstring): there is no weak-secret gate to
    fail here because there is no operator-supplied secret at all. The route
    registrations live beside their native route owners (Kanban plugin API or
    MCP router), unconditionally, so the fixed route contours stay scoped even
    if this plugin were somehow disabled.
    """
    ctx.register_dashboard_auth_provider(WorkspaceReadTokenProvider())
    ctx.register_dashboard_auth_provider(RecommendationsReadTokenProvider())
    ctx.register_dashboard_auth_provider(ConnectionsManageTokenProvider())
    ctx.register_dashboard_auth_provider(AutomationsManageTokenProvider())
    ctx.register_cli_command(
        name="kanban-workspace-token",
        help="Issue/list/revoke managed Raphael dashboard credentials",
        setup_fn=register_cli,
        handler_fn=workspace_token_command,
        description=(
            "Manage revocable, expiring machine credentials for the fixed "
            "Raphael Workspace, owner recommendations, Connections, and Automations surfaces. "
            "Recommendations tokens are independently scoped and capped at "
            "8 hours. 'issue' writes a new bearer once to an explicit path; "
            "'list' exposes only metadata; 'revoke' disables one token_id."
        ),
    )
    logger.info(
        "raphael managed tokens: registered providers %r/%r/%r/%r (scopes=%s/%s/%s/%s)",
        WorkspaceReadTokenProvider.name,
        RecommendationsReadTokenProvider.name,
        ConnectionsManageTokenProvider.name,
        AutomationsManageTokenProvider.name,
        SCOPE,
        RECOMMENDATIONS_SCOPE,
        CONNECTIONS_SCOPE,
        AUTOMATIONS_SCOPE,
    )
