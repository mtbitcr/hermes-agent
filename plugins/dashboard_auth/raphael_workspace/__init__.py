"""WorkspaceReadTokenProvider — Item 32G-A: revocable, expiring, audited
machine credential for the exact Raphael Workspace read-only Kanban surface.

What it is
----------
A service-to-service auth provider for the fixed set of 9 GET routes the
Kanban dashboard plugin API registers for this credential (see
``plugins/kanban/dashboard/plugin_api.py``, the "Item 32G-A" section). It
verifies an inbound ``Authorization: Bearer <token_id>.<secret>`` against the
on-disk registry in :mod:`token_store` and, on a match, vouches for the
fixed ``raphael-workspace`` principal carrying the single fixed scope
``kanban.read``.

How this differs from drain / recommendations
----------------------------------------------
Those two are single static secrets carried in one env var — no lifecycle,
no revocation, no expiry. This credential needs all three (issue/list/revoke,
bounded expiry, immediate revocation), so it is backed by a small on-disk
registry (:mod:`token_store`) instead: many tokens, each individually
revocable, verified fresh against disk on every request (no cache — a
``revoke`` takes effect on the very next call). What IS shared with the other
two: the seam it plugs into (``supports_token`` / ``verify_token`` +
``token_auth`` middleware), the ``TokenPrincipal`` contract, and the
constant-time (``hmac.compare_digest``) comparison discipline.

There is no login, cookie, session, or refresh: only the token capability is
implemented.

Configuration
-------------
Nothing goes in ``.env`` or ``config.yaml`` — there is no operator-supplied
secret to provision. Credentials are minted locally via
``hermes kanban-workspace-token issue --out <path>``; the principal, scope,
project, and board are fixed constants (see :mod:`token_store`), never
configuration, because the grant IS the authority boundary this plugin
exists to enforce.

The provider always registers: an empty or missing token registry already
denies every ``verify_token`` call (fail-closed), so there is no "unset env
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

__all__ = [
    "PRINCIPAL",
    "SCOPE",
    "PROJECT",
    "BOARD",
    "TOKEN_PREFIX",
    "GRANT",
    "WorkspaceReadTokenProvider",
    "register",
]


class WorkspaceReadTokenProvider(DashboardAuthProvider):
    """Non-interactive provider backed by the CLI-managed token registry."""

    name = "raphael-workspace-token"
    display_name = "Raphael Workspace (read-only Kanban credential)"
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
        if not token.startswith(TOKEN_PREFIX) or "." not in token:
            return None
        token_id, _, secret = token.partition(".")
        if not token_id or not secret:
            return None
        try:
            record = token_store.verify(token_id, secret)
        except token_store.TokenStoreError as exc:
            raise ProviderError(str(exc)) from exc
        if record is None:
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


# ---------------------------------------------------------------------------
# Plugin entry point
# ---------------------------------------------------------------------------


def register(ctx) -> None:
    """Plugin entry — registers the provider and the CLI command tree.

    Always registers (see module docstring): there is no weak-secret gate to
    fail here because there is no operator-supplied secret at all. The route
    registrations themselves live in the Kanban plugin API
    (``plugin_api.py``), unconditionally, so the 9 routes stay token-only
    even if this plugin were somehow disabled.
    """
    ctx.register_dashboard_auth_provider(WorkspaceReadTokenProvider())
    ctx.register_cli_command(
        name="kanban-workspace-token",
        help="Issue/list/revoke the read-only Raphael Workspace Kanban credential",
        setup_fn=register_cli,
        handler_fn=workspace_token_command,
        description=(
            "Manage the revocable, expiring machine credential for the fixed "
            "Raphael Workspace read-only Kanban surface (Item 32G-A). "
            "'issue' mints a new bearer token and writes it once to an "
            "explicit path; 'list' shows non-secret metadata for every "
            "issued token; 'revoke' disables one by its token_id."
        ),
    )
    logger.info(
        "raphael-workspace-token: registered provider %r (principal=%s, scope=%s)",
        WorkspaceReadTokenProvider.name, PRINCIPAL, SCOPE,
    )
