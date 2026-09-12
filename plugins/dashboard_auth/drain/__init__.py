"""DrainSecretProvider — shared-bearer-secret auth for the drain-control endpoint.

Task 2.0b of the safe-shutdown plan, and the FIRST consumer of the generic
non-interactive token-auth capability added in Task 2.0a
(``supports_token`` / ``verify_token`` on the ``DashboardAuthProvider`` ABC +
the route-agnostic ``token_auth`` middleware seam).

What it is
----------
A service-to-service auth provider. ``nous-account-service`` (NAS) provisions a
**per-agent unique** shared secret into each deployed agent's environment; this
provider verifies an inbound ``Authorization`` bearer token against that secret
with a constant-time compare and, on a match, vouches for the caller as the
``drain-control`` principal. It is NOT an interactive identity provider — there
is no login, cookie, session, or refresh. It implements ONLY the token
capability (``supports_token = True`` + ``verify_token``); the five interactive
ABC methods raise ``NotImplementedError``.

Why a plugin (not an ad-hoc header check on the drain route)
------------------------------------------------------------
Decisions.md Q-A: the drain credential MUST be a real auth plugin in the
dashboard auth framework, not a bolt-on. Q-C: the framework widening that
hosts it is generic (Task 2.0a) and this plugin is merely its first consumer.

Security properties (decisions.md Q-A)
--------------------------------------
* **Per-agent unique secret** — each agent gets a distinct secret; a leak's
  blast radius is one agent.
* **Entropy gate at registration** — a weak/short/low-entropy secret fails
  CLOSED at load (the plugin declines to register and records a skip reason);
  it is never silently accepted. Bar: >= 256 bits of entropy / >= 43
  url-safe-base64 chars, and the value must not be obviously structured
  (all-one-character, too few distinct characters).
* **Constant-time compare** — ``hmac.compare_digest`` on the request path, so
  the endpoint is not a timing oracle.

Configuration
-------------
The secret is a CREDENTIAL, so it is carried via an env var (the ``.env``-is-
for-secrets-only rule), provisioned by NAS at deploy time (Phase 3):

    HERMES_DASHBOARD_DRAIN_SECRET   # the per-agent shared secret (>=43 url-safe-b64 chars)

Behavioural knobs live in config.yaml (canonical surface):

    dashboard:
      drain_auth:
        scope: drain            # optional; MUST be the fixed drain scope
        min_secret_chars: 43    # entropy bar (optional; default 43 ~= 256 bits)

The scope is NOT a knob: it is the authority boundary this credential exists to
enforce, so it is pinned to the exported :data:`DRAIN_SCOPE` constant. The key
may be omitted, or spelled out as exactly ``drain`` (so an existing config that
states it keeps working); any other value — notably the recommendations reader's
``kanban:recommendations:read`` — is refused and the plugin registers NOTHING
(no provider, no route). An operator-editable scope would otherwise be a way to
mint a drain credential carrying another endpoint's capability, or to widen the
drain route to accept another endpoint's credential.

When ``HERMES_DASHBOARD_DRAIN_SECRET`` is unset, the plugin is a no-op (records
a skip reason) — agents that don't want NAS-driven drain just don't set it.
"""
from __future__ import annotations

import logging
import os
from typing import Optional

from hermes_cli.dashboard_auth import TokenPrincipal
from hermes_cli.dashboard_auth.static_secret import (
    DEFAULT_MIN_SECRET_CHARS,
    StaticBearerSecretProvider,
    assess_secret_strength,
)

logger = logging.getLogger(__name__)

# The entropy gate and the constant-time secret compare live in the shared
# ``hermes_cli.dashboard_auth.static_secret`` helper so this plugin and the
# bundled ``dashboard_auth/recommendations`` plugin cannot drift apart on
# security logic. ``assess_secret_strength`` is re-exported here because it is
# part of this module's public surface
# (``plugins.dashboard_auth.drain.assess_secret_strength``).
_DEFAULT_MIN_SECRET_CHARS = DEFAULT_MIN_SECRET_CHARS

# The exact method + path the begin/cancel-drain endpoint lives on, and the ONE
# capability it grants. Registered as a token-authable route by ``register()``
# under exactly this scope, so the generic seam guards it and no other machine
# credential can drive it. All three are fixed constants, not configuration:
# the provider vouches for ``DRAIN_SCOPE`` and the route demands ``DRAIN_SCOPE``,
# so the credential and the policy are minted from one immutable value and
# cannot drift or be widened. Kept here (not imported from web_server) to avoid
# a heavy import at plugin load.
DRAIN_ROUTE_PATH = "/api/gateway/drain"
DRAIN_ROUTE_METHOD = "POST"
DRAIN_SCOPE = "drain"
DRAIN_PRINCIPAL = "drain-control"

LAST_SKIP_REASON: str = ""

__all__ = [
    "DRAIN_PRINCIPAL",
    "DRAIN_ROUTE_METHOD",
    "DRAIN_ROUTE_PATH",
    "DRAIN_SCOPE",
    "DrainSecretProvider",
    "assess_secret_strength",
    "register",
]


def _pin_drain_scope(scope: Optional[str]) -> str:
    """Return :data:`DRAIN_SCOPE`, or raise for a request for any other scope.

    ``None`` means "unspecified" and resolves to the fixed drain scope; the
    literal fixed scope is accepted so an existing config/caller that states it
    explicitly keeps working. Everything else — another endpoint's scope, a
    renamed scope, a blank string — raises ``ValueError``, because the scope is
    the authority boundary and no caller gets to choose it.
    """
    if scope is None:
        return DRAIN_SCOPE
    candidate = str(scope).strip()
    if candidate == DRAIN_SCOPE:
        return DRAIN_SCOPE
    raise ValueError(
        f"drain scope is fixed at {DRAIN_SCOPE!r} and cannot be configured; "
        f"got {candidate!r}"
    )


class DrainSecretProvider(StaticBearerSecretProvider):
    """Non-interactive shared-bearer-secret provider for drain control.

    Everything security-relevant — the entropy gate enforced at construction
    and the ``hmac.compare_digest`` verify on the request path — is inherited
    from :class:`~hermes_cli.dashboard_auth.static_secret.StaticBearerSecretProvider`.
    This class only pins the drain identity: the provider name, the
    ``drain-control`` principal, and the fixed :data:`DRAIN_SCOPE` capability.

    ``scope`` is not caller-chosen authority: it may be omitted, or passed as
    exactly :data:`DRAIN_SCOPE`, and anything else raises ``ValueError``. A
    provider carrying some other scope would either stand in for a different
    endpoint's credential or leave the drain route demanding a capability no
    drain credential holds.
    """

    name = "drain-secret"
    display_name = "Drain Control (service credential)"
    principal_id = DRAIN_PRINCIPAL
    credential_label = "drain"

    def __init__(self, *, secret: str, scope: Optional[str] = None) -> None:
        # Defence in depth: the base constructor also enforces the entropy
        # bar, so a caller that bypasses register()'s check still can't build
        # a weak provider. register() does the friendly skip-reason path; this
        # raises. The scope pin is checked first, so a mis-scoped construction
        # fails even for a perfectly strong secret.
        super().__init__(secret=secret, scope=_pin_drain_scope(scope))

    def verify_token(self, *, token: str) -> Optional[TokenPrincipal]:
        """Constant-time compare against the per-agent shared secret.

        Returns a ``drain-control`` principal scoped to :data:`DRAIN_SCOPE` on
        an exact match, else ``None`` (the generic seam falls through / fails
        closed).
        """
        return super().verify_token(token=token)


# ---------------------------------------------------------------------------
# Plugin entry point
# ---------------------------------------------------------------------------


def _load_config_drain_auth_section() -> dict:
    """Return ``dashboard.drain_auth`` from config.yaml, or ``{}``."""
    try:
        from hermes_cli.config import cfg_get, load_config

        cfg = load_config()
    except Exception as exc:  # noqa: BLE001 — broad catch is intentional
        logger.debug(
            "dashboard-auth-drain: load_config() raised %s; "
            "falling back to env-only configuration",
            exc,
        )
        return {}
    section = cfg_get(cfg, "dashboard", "drain_auth", default=None)
    return section if isinstance(section, dict) else {}


def register(ctx) -> None:
    """Plugin entry — registers DrainSecretProvider when a strong secret is set.

    No-op (records a skip reason) when ``HERMES_DASHBOARD_DRAIN_SECRET`` is
    unset, fails the entropy gate, or when ``dashboard.drain_auth.scope`` asks
    for anything other than the fixed :data:`DRAIN_SCOPE`. On success, also
    registers the begin/cancel-drain route as token-authable via the generic
    seam, under that same fixed scope.

    Every rejection path returns before ``ctx.register_dashboard_auth_provider``
    and before ``register_token_route``, so a refused configuration never
    leaves a half-registered endpoint behind.
    """
    global LAST_SKIP_REASON
    LAST_SKIP_REASON = ""

    secret = os.environ.get("HERMES_DASHBOARD_DRAIN_SECRET", "").strip()
    if not secret:
        LAST_SKIP_REASON = (
            "HERMES_DASHBOARD_DRAIN_SECRET is not set. Set a per-agent "
            ">=256-bit secret (e.g. `python -c \"import secrets; "
            "print(secrets.token_urlsafe(32))\"`) to enable NAS-driven drain "
            "coordination; leave it unset to disable the drain endpoint."
        )
        logger.debug("dashboard-auth-drain: %s", LAST_SKIP_REASON)
        return

    section = _load_config_drain_auth_section()
    # Resolve the scope BEFORE anything is registered: a config that asks for
    # any scope other than the fixed one is a configuration error, not a
    # weaker-authority deployment, so the plugin registers neither the
    # credential provider nor the route (no partial registration) and the drain
    # endpoint stays on its ordinary cookie/session gate.
    try:
        scope = _pin_drain_scope(section.get("scope"))
    except ValueError as exc:
        LAST_SKIP_REASON = (
            f"dashboard.drain_auth.scope rejected — {exc}. The drain endpoint "
            "stays disabled (fail-closed)."
        )
        logger.warning("dashboard-auth-drain: %s", LAST_SKIP_REASON)
        return

    try:
        min_chars = int(section.get("min_secret_chars", _DEFAULT_MIN_SECRET_CHARS))
    except (TypeError, ValueError):
        min_chars = _DEFAULT_MIN_SECRET_CHARS

    reason = assess_secret_strength(secret, min_chars=min_chars)
    if reason is not None:
        LAST_SKIP_REASON = (
            f"HERMES_DASHBOARD_DRAIN_SECRET rejected — {reason}. "
            "The drain endpoint stays disabled (fail-closed)."
        )
        logger.warning("dashboard-auth-drain: %s", LAST_SKIP_REASON)
        return

    try:
        provider = DrainSecretProvider(secret=secret, scope=scope)
    except ValueError as exc:
        LAST_SKIP_REASON = f"DrainSecretProvider construction failed: {exc}"
        logger.warning("dashboard-auth-drain: %s", LAST_SKIP_REASON)
        return

    ctx.register_dashboard_auth_provider(provider)

    # Opt the begin/cancel-drain endpoint into the generic token-auth seam so
    # the dashboard's interactive cookie gate doesn't bounce NAS's bearer call.
    # Registered ONLY here, after the strong credential provider was accepted,
    # and only for POST with the fixed drain scope: a caller holding some other
    # machine credential (e.g. the recommendations reader) is recognised but
    # not authorised, and gets the seam's generic 403. When this plugin is a
    # no-op the route is not a token route at all, preserving the existing
    # browser-session fallback for dashboard-driven drain.
    try:
        from hermes_cli.dashboard_auth.token_auth import register_token_route

        register_token_route(
            DRAIN_ROUTE_PATH, method=DRAIN_ROUTE_METHOD, required_scope=scope
        )
    except Exception as exc:  # noqa: BLE001 — seam import must not crash plugin load
        logger.warning(
            "dashboard-auth-drain: could not register token route %s %s: %s",
            DRAIN_ROUTE_METHOD, DRAIN_ROUTE_PATH, exc,
        )

    logger.info(
        "dashboard-auth-drain: registered drain service-credential provider "
        "(scope=%s, route=%s %s)",
        scope, DRAIN_ROUTE_METHOD, DRAIN_ROUTE_PATH,
    )
