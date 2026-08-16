"""RecommendationsSecretProvider — bearer-secret auth for the recommendations read API.

Item 31BI. The second consumer of the generic non-interactive token-auth
capability (``supports_token`` / ``verify_token`` + the route-scoped
``token_auth`` middleware seam), after ``dashboard_auth/drain``.

What it is
----------
A service-to-service auth provider for exactly one endpoint:
``GET /api/plugins/kanban/recommendations``, the read-only owner-review view of
native recommendation cards. It verifies an inbound ``Authorization`` bearer
token against a per-agent static secret with a constant-time compare and, on a
match, vouches for the caller as the ``kanban-recommendations-reader``
principal carrying the single fixed scope ``kanban:recommendations:read``.

There is no login, cookie, session, or refresh: it implements ONLY the token
capability. All of its security logic — the fail-closed entropy gate and the
``hmac.compare_digest`` verify — comes from the shared
``hermes_cli.dashboard_auth.static_secret`` helper, the same code the drain
plugin uses, so the two credentials cannot drift apart.

Why a separate credential (and a separate scope)
------------------------------------------------
Before Item 31BI the token seam matched on path alone, so ANY recognised
machine token could reach EVERY token route — the drain credential could read
recommendations and vice versa. The seam now keys on (method, path,
required_scope) and this provider's principal carries only
``kanban:recommendations:read``. It is therefore recognised on the
recommendations GET and refused with a generic 403 on the drain POST, while the
drain credential gets the mirror-image treatment. One leaked credential buys
exactly one endpoint.

Configuration
-------------
The secret is a CREDENTIAL, so it is carried via an env var (the ``.env``-is-
for-secrets-only rule), provisioned at deploy time:

    HERMES_DASHBOARD_RECOMMENDATIONS_SECRET   # per-agent secret (>=43 url-safe-b64 chars)

There are no behavioural knobs: the principal and the scope are FIXED, because
the scope is the authority boundary this plugin exists to enforce and an
operator-editable scope would be a way to widen it.

When the env var is unset (or holds a weak/short/low-entropy value) the plugin
registers nothing and records a skip reason. The recommendations route stays
registered as token-only by the Kanban plugin API regardless, so it then simply
has no provider that can accept it and answers 401 — fail-closed, never open.
"""
from __future__ import annotations

import logging
import os
from typing import Optional

from hermes_cli.dashboard_auth import TokenPrincipal
from hermes_cli.dashboard_auth.static_secret import (
    StaticBearerSecretProvider,
    assess_secret_strength,
)

logger = logging.getLogger(__name__)

# The env var carrying the per-agent secret. A credential, never config.yaml.
SECRET_ENV_VAR = "HERMES_DASHBOARD_RECOMMENDATIONS_SECRET"

# The exact route this credential is for, and the ONE capability it grants.
# Both are fixed constants, not configuration: the Kanban plugin API registers
# the same method/path/scope triple with the seam, and the seam requires an
# exact match on all three.
RECOMMENDATIONS_ROUTE_PATH = "/api/plugins/kanban/recommendations"
RECOMMENDATIONS_ROUTE_METHOD = "GET"
RECOMMENDATIONS_SCOPE = "kanban:recommendations:read"
RECOMMENDATIONS_PRINCIPAL = "kanban-recommendations-reader"

LAST_SKIP_REASON: str = ""

__all__ = [
    "RECOMMENDATIONS_PRINCIPAL",
    "RECOMMENDATIONS_ROUTE_METHOD",
    "RECOMMENDATIONS_ROUTE_PATH",
    "RECOMMENDATIONS_SCOPE",
    "SECRET_ENV_VAR",
    "RecommendationsSecretProvider",
    "register",
]


class RecommendationsSecretProvider(StaticBearerSecretProvider):
    """Non-interactive static-bearer-secret provider for the recommendations read API.

    The entropy gate at construction and the constant-time verify on the
    request path are inherited from
    :class:`~hermes_cli.dashboard_auth.static_secret.StaticBearerSecretProvider`.
    This class only pins the identity: provider name, the
    ``kanban-recommendations-reader`` principal, and the fixed
    ``kanban:recommendations:read`` scope (not overridable — the scope IS the
    authority boundary).
    """

    name = "recommendations-secret"
    display_name = "Kanban Recommendations (service credential)"
    principal_id = RECOMMENDATIONS_PRINCIPAL
    credential_label = "recommendations"

    def __init__(self, *, secret: str) -> None:
        super().__init__(secret=secret, scope=RECOMMENDATIONS_SCOPE)

    def verify_token(self, *, token: str) -> Optional[TokenPrincipal]:
        """Constant-time compare against the per-agent recommendations secret.

        Returns the fixed ``kanban-recommendations-reader`` principal scoped to
        ``kanban:recommendations:read`` on an exact match, else ``None`` (the
        generic seam falls through / fails closed).
        """
        return super().verify_token(token=token)


# ---------------------------------------------------------------------------
# Plugin entry point
# ---------------------------------------------------------------------------


def register(ctx) -> None:
    """Plugin entry — registers the provider when a strong secret is set.

    No-op (records a skip reason) when ``HERMES_DASHBOARD_RECOMMENDATIONS_SECRET``
    is unset or fails the entropy gate.

    This entry point deliberately does NOT register the route: the Kanban
    plugin API owns that registration and performs it unconditionally at
    import, so the endpoint is token-only (and therefore 401) even when no
    credential is provisioned. Registering it here as well would make the
    gating depend on this plugin having loaded — the exact fail-open shape
    this repair removes.
    """
    global LAST_SKIP_REASON
    LAST_SKIP_REASON = ""

    secret = os.environ.get(SECRET_ENV_VAR, "").strip()
    if not secret:
        LAST_SKIP_REASON = (
            f"{SECRET_ENV_VAR} is not set. Set a per-agent >=256-bit secret "
            "(e.g. `python -c \"import secrets; "
            "print(secrets.token_urlsafe(32))\"`) to let a service read "
            "recommendation cards; leave it unset to keep the endpoint "
            "unreachable (it stays token-only and answers 401)."
        )
        logger.debug("dashboard-auth-recommendations: %s", LAST_SKIP_REASON)
        return

    reason = assess_secret_strength(secret)
    if reason is not None:
        LAST_SKIP_REASON = (
            f"{SECRET_ENV_VAR} rejected — {reason}. The recommendations "
            "endpoint stays unreachable (fail-closed)."
        )
        logger.warning("dashboard-auth-recommendations: %s", LAST_SKIP_REASON)
        return

    try:
        provider = RecommendationsSecretProvider(secret=secret)
    except ValueError as exc:
        LAST_SKIP_REASON = f"RecommendationsSecretProvider construction failed: {exc}"
        logger.warning("dashboard-auth-recommendations: %s", LAST_SKIP_REASON)
        return

    ctx.register_dashboard_auth_provider(provider)

    logger.info(
        "dashboard-auth-recommendations: registered recommendations "
        "service-credential provider (scope=%s, route=%s %s)",
        RECOMMENDATIONS_SCOPE,
        RECOMMENDATIONS_ROUTE_METHOD,
        RECOMMENDATIONS_ROUTE_PATH,
    )
