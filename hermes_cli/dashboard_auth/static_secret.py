"""Shared building blocks for strong static bearer-secret auth providers.

Item 31BI. Two bundled providers now authenticate a machine caller against a
per-agent static secret carried in an env var: ``dashboard_auth/drain``
(gateway drain control) and ``dashboard_auth/recommendations`` (read-only
Kanban recommendation cards). Their security logic — the fail-closed entropy
gate at registration and the constant-time compare on the request path — is
identical and MUST NOT be duplicated per plugin, so it lives here once:

  * :func:`assess_secret_strength` — the entropy gate (length, distinct-char
    count, Shannon entropy). Returns a human-readable rejection reason, or
    ``None`` when the secret passes.
  * :class:`StaticBearerSecretProvider` — a non-interactive
    :class:`~hermes_cli.dashboard_auth.base.DashboardAuthProvider` that
    verifies a bearer token against one static secret with
    ``hmac.compare_digest`` and vouches for a fixed principal carrying a
    single capability scope.

A plugin subclasses :class:`StaticBearerSecretProvider`, sets ``name`` /
``display_name`` / ``principal_id``, and is done: the token capability, the
entropy gate in ``__init__`` (defence in depth for callers that bypass the
plugin's ``register()``), and the interactive-method stubs all come from here.

This module deliberately imports nothing outside ``dashboard_auth`` so it can
be imported from plugin load, which happens early in startup.
"""
from __future__ import annotations

import hmac
import math
from collections import Counter
from typing import Optional

from hermes_cli.dashboard_auth.base import (
    DashboardAuthProvider,
    LoginStart,
    Session,
    TokenPrincipal,
)

# Default entropy bar: 43 url-safe-base64 chars ~= 256 bits. token_urlsafe(32)
# produces 43 chars, so a correctly-provisioned secret clears this exactly.
DEFAULT_MIN_SECRET_CHARS = 43
# A secret must contain at least this many DISTINCT characters — rejects
# degenerate values like "aaaa..." that are long but trivially low-entropy.
MIN_DISTINCT_CHARS = 16
# Shannon entropy floor (bits) over the secret's characters — a second,
# distribution-aware guard on top of the length + distinct-count checks.
MIN_SHANNON_BITS = 128.0


def shannon_bits(value: str) -> float:
    """Total Shannon entropy (bits) of ``value`` over its character distribution.

    H = len * sum(-p_i * log2(p_i)). A long string drawn from a wide alphabet
    scores high; a long run of one character scores ~0.
    """
    if not value:
        return 0.0
    counts = Counter(value)
    n = len(value)
    per_char = -sum((c / n) * math.log2(c / n) for c in counts.values())
    return per_char * n


def assess_secret_strength(
    secret: str, *, min_chars: int = DEFAULT_MIN_SECRET_CHARS
) -> Optional[str]:
    """Return a rejection reason if ``secret`` is too weak, else ``None``.

    Fail-closed entropy gate (decisions.md Q-A). Checks, in order:
      * length >= ``min_chars`` (default 43 url-safe-b64 chars ~= 256 bits),
      * at least ``MIN_DISTINCT_CHARS`` distinct characters,
      * Shannon entropy >= ``MIN_SHANNON_BITS`` bits.

    A ``None`` return means the secret passes. Any string return is a
    human-readable reason the caller logs + records as the skip reason.
    """
    if not secret:
        return "secret is empty"
    if len(secret) < min_chars:
        return (
            f"secret too short: {len(secret)} chars (need >= {min_chars}; "
            "use a >=256-bit value, e.g. `python -c \"import secrets; "
            "print(secrets.token_urlsafe(32))\"`)"
        )
    distinct = len(set(secret))
    if distinct < MIN_DISTINCT_CHARS:
        return (
            f"secret has only {distinct} distinct characters (need >= "
            f"{MIN_DISTINCT_CHARS}); looks structured/low-entropy"
        )
    bits = shannon_bits(secret)
    if bits < MIN_SHANNON_BITS:
        return (
            f"secret entropy too low: {bits:.0f} bits (need >= "
            f"{MIN_SHANNON_BITS:.0f}); looks structured/repeated"
        )
    return None


class StaticBearerSecretProvider(DashboardAuthProvider):
    """Non-interactive provider that verifies one strong static bearer secret.

    Subclasses set ``name``, ``display_name``, ``principal_id`` and (for the
    rejection message) ``credential_label``. Construction enforces the entropy
    gate, so a caller that bypasses a plugin's ``register()`` still cannot
    build a provider around a weak secret.

    Only the token capability is implemented (``supports_token`` /
    ``verify_token``); the interactive ABC methods raise
    ``NotImplementedError`` because there is no login, cookie, session, or
    refresh for a service credential. ``verify_session`` returns ``None``
    (rather than raising) so the provider stacks harmlessly in the
    cookie-verify loop, and ``revoke_session`` is a no-op.
    """

    supports_token = True
    supports_session = False

    # Stable identifier this provider vouches for once the secret matches.
    principal_id: str = ""
    # Human-readable credential name used in the construction error.
    credential_label: str = "static bearer"

    def __init__(
        self,
        *,
        secret: str,
        scope: str,
        min_chars: int = DEFAULT_MIN_SECRET_CHARS,
    ) -> None:
        reason = assess_secret_strength(secret, min_chars=min_chars)
        if reason is not None:
            raise ValueError(f"{self.credential_label} secret rejected: {reason}")
        scope = (scope or "").strip()
        if not scope:
            raise ValueError(
                f"{self.credential_label} secret rejected: scope must be a "
                "non-empty capability label"
            )
        if not self.principal_id:
            raise ValueError(
                f"{type(self).__name__} must set a non-empty principal_id"
            )
        self._secret = secret
        self._scope = scope

    # ---- token capability (the only thing this provider implements) --------

    def verify_token(self, *, token: str) -> Optional[TokenPrincipal]:
        """Constant-time compare against the per-agent static secret.

        Returns this provider's fixed principal, carrying its single
        capability scope, on an exact match; otherwise ``None`` (the generic
        seam falls through / fails closed). Uses ``hmac.compare_digest`` so a
        wrong token can't be recovered by timing.
        """
        if not token:
            return None
        if hmac.compare_digest(token.encode("utf-8"), self._secret.encode("utf-8")):
            return TokenPrincipal(
                principal=self.principal_id,
                provider=self.name,
                scopes=(self._scope,),
            )
        return None

    # ---- interactive methods: unsupported (service credential only) --------

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
        # Not a cookie-session provider — it never mints a Session, so it can
        # never recognise a session cookie. Return None (don't raise) so it
        # stacks harmlessly in the cookie-verify loop.
        return None

    def refresh_session(self, *, refresh_token: str) -> Session:
        raise self._no_interactive()

    def revoke_session(self, *, refresh_token: str) -> None:
        return None
