"""Tests for the RecommendationsSecretProvider plugin (non-interactive bearer secret).

Item 31BI. The second consumer of the generic token-auth capability, after
``dashboard_auth/drain``. Loads the bundled plugin module directly and
exercises:
  * the fixed identity — principal ``kanban-recommendations-reader`` and the
    single scope ``kanban:recommendations:read``, neither of them overridable,
  * that the security logic is the SHARED
    ``hermes_cli.dashboard_auth.static_secret`` helper (same entropy gate as
    drain, same ``hmac.compare_digest`` verify) rather than a per-plugin copy,
  * the non-interactive surface (token capability only, no login/session),
  * the register(ctx) entry point: no provider for a missing or weak secret,
    exactly one compliant provider for a strong one, and no route registration
    of its own (the Kanban plugin API owns the route, unconditionally).
"""
from __future__ import annotations

import hmac
import secrets
from unittest.mock import MagicMock

import pytest

import plugins.dashboard_auth.drain as drain_plugin
import plugins.dashboard_auth.recommendations as rec_plugin
from hermes_cli.dashboard_auth import TokenPrincipal, assert_protocol_compliance
from hermes_cli.dashboard_auth import static_secret, token_auth

_SCOPE = "kanban:recommendations:read"
_PRINCIPAL = "kanban-recommendations-reader"
_ROUTE_PATH = "/api/plugins/kanban/recommendations"


@pytest.fixture(scope="module")
def rec():
    return rec_plugin


@pytest.fixture(autouse=True)
def _clean_env_and_routes(monkeypatch):
    monkeypatch.delenv(rec_plugin.SECRET_ENV_VAR, raising=False)
    token_auth.clear_token_routes()
    yield
    token_auth.clear_token_routes()


def _strong_secret() -> str:
    # token_urlsafe(32) → 43 url-safe-b64 chars ≈ 256 bits.
    return secrets.token_urlsafe(32)


# ---------------------------------------------------------------------------
# Fixed identity: principal + scope are constants, not configuration
# ---------------------------------------------------------------------------


class TestFixedIdentity:
    def test_verify_token_vouches_for_the_fixed_principal_and_scope(self, rec):
        s = _strong_secret()
        principal = rec.RecommendationsSecretProvider(secret=s).verify_token(token=s)
        assert isinstance(principal, TokenPrincipal)
        assert principal.principal == _PRINCIPAL
        assert principal.provider == "recommendations-secret"
        assert principal.scopes == (_SCOPE,)

    def test_scope_cannot_be_overridden_at_construction(self, rec):
        # The scope IS the authority boundary: the constructor takes only the
        # secret, so no caller can widen (or rename) it.
        with pytest.raises(TypeError):
            rec.RecommendationsSecretProvider(
                secret=_strong_secret(), scope="something-else"
            )

    def test_exported_constants_match_the_credential_it_mints(self, rec):
        s = _strong_secret()
        principal = rec.RecommendationsSecretProvider(secret=s).verify_token(token=s)
        # The constants the Kanban plugin API's route policy is compared
        # against must be the ones the credential actually carries.
        assert rec.RECOMMENDATIONS_SCOPE == principal.scopes[0] == _SCOPE
        assert rec.RECOMMENDATIONS_PRINCIPAL == principal.principal == _PRINCIPAL
        assert rec.RECOMMENDATIONS_ROUTE_METHOD == "GET"
        assert rec.RECOMMENDATIONS_ROUTE_PATH == _ROUTE_PATH

    def test_scope_is_not_the_drain_scope(self, rec):
        s = _strong_secret()
        rec_scopes = rec.RecommendationsSecretProvider(secret=s).verify_token(
            token=s
        ).scopes
        drain_scopes = drain_plugin.DrainSecretProvider(secret=s).verify_token(
            token=s
        ).scopes
        # Disjoint capability sets: neither credential can stand in for the
        # other at the seam's scope check.
        assert set(rec_scopes).isdisjoint(drain_scopes)


# ---------------------------------------------------------------------------
# Shared static-secret helper (no per-plugin copy of the security logic)
# ---------------------------------------------------------------------------


class TestSharedStaticSecretHelper:
    def test_is_a_static_bearer_secret_provider(self, rec):
        assert issubclass(
            rec.RecommendationsSecretProvider, static_secret.StaticBearerSecretProvider
        )

    def test_verify_token_goes_through_constant_time_compare(self, rec, monkeypatch):
        s = _strong_secret()
        provider = rec.RecommendationsSecretProvider(secret=s)
        calls: list[tuple[bytes, bytes]] = []
        real = hmac.compare_digest

        def _spy(a, b):
            calls.append((a, b))
            return real(a, b)

        # Patch on the shared helper module: if this provider ever grew its own
        # comparison, the spy would not be consulted.
        monkeypatch.setattr(static_secret.hmac, "compare_digest", _spy)
        assert provider.verify_token(token=s) is not None
        assert provider.verify_token(token=s + "x") is None
        assert len(calls) == 2
        # The full presented token is compared in one shot — no early-exit
        # prefix comparison and no plaintext equality on the secret.
        assert calls[0] == (s.encode("utf-8"), s.encode("utf-8"))

    @pytest.mark.parametrize(
        "secret",
        ["", "short", "a" * 60, "ab" * 30, secrets.token_urlsafe(32)[:42]],
        ids=["empty", "short", "one-char", "two-chars", "one-char-under-bar"],
    )
    def test_weak_secrets_are_rejected_by_the_shared_gate(self, rec, secret):
        assert static_secret.assess_secret_strength(secret) is not None
        with pytest.raises(ValueError):
            rec.RecommendationsSecretProvider(secret=secret)

    def test_entropy_gate_verdicts_match_the_drain_plugin(self, rec):
        # One gate, two credentials: the same secret must be accepted or
        # rejected identically by both, so they cannot drift apart.
        for secret in ("", "short", "a" * 60, _strong_secret()):
            rec_ok = static_secret.assess_secret_strength(secret) is None
            drain_ok = drain_plugin.assess_secret_strength(secret) is None
            assert rec_ok == drain_ok


# ---------------------------------------------------------------------------
# Provider behaviour: non-interactive, token capability only
# ---------------------------------------------------------------------------


class TestProvider:
    def test_protocol_compliance(self, rec):
        assert_protocol_compliance(rec.RecommendationsSecretProvider)

    def test_supports_token_only(self, rec):
        p = rec.RecommendationsSecretProvider(secret=_strong_secret())
        assert p.supports_token is True
        # Excluded from interactive surfaces via list_session_providers().
        assert p.supports_session is False

    @pytest.mark.parametrize(
        "token", ["", "wrong", "  ", "Bearer", "kanban:recommendations:read"]
    )
    def test_verify_token_rejects_non_matching_tokens(self, rec, token):
        p = rec.RecommendationsSecretProvider(secret=_strong_secret())
        assert p.verify_token(token=token) is None

    @pytest.mark.parametrize("mutate", [lambda s: s[:-1], lambda s: s + "x",
                                        lambda s: " " + s, lambda s: s.upper()])
    def test_verify_token_requires_an_exact_match(self, rec, mutate):
        s = _strong_secret()
        p = rec.RecommendationsSecretProvider(secret=s)
        assert p.verify_token(token=mutate(s)) is None
        assert p.verify_token(token=s) is not None

    def test_interactive_methods_raise_and_session_methods_are_inert(self, rec):
        p = rec.RecommendationsSecretProvider(secret=_strong_secret())
        with pytest.raises(NotImplementedError):
            p.start_login(redirect_uri="r")
        with pytest.raises(NotImplementedError):
            p.complete_login(code="c", state="s", code_verifier="v", redirect_uri="r")
        with pytest.raises(NotImplementedError):
            p.refresh_session(refresh_token="r")
        # Stacks harmlessly in the cookie-verify loop instead of raising.
        assert p.verify_session(access_token="anything") is None
        assert p.revoke_session(refresh_token="anything") is None


# ---------------------------------------------------------------------------
# register() entry point
# ---------------------------------------------------------------------------


class TestRegister:
    def test_skips_when_no_secret(self, rec):
        ctx = MagicMock()
        rec.register(ctx)
        ctx.register_dashboard_auth_provider.assert_not_called()
        assert rec.SECRET_ENV_VAR in rec.LAST_SKIP_REASON

    @pytest.mark.parametrize(
        "weak",
        ["tooweak", "   ", "a" * 60, "ab" * 30, "0123456789" * 5],
        ids=["short", "blank", "one-char", "two-chars", "repeating-digits"],
    )
    def test_skips_and_fails_closed_on_weak_secret(self, rec, monkeypatch, weak):
        monkeypatch.setenv(rec.SECRET_ENV_VAR, weak)
        ctx = MagicMock()
        rec.register(ctx)
        ctx.register_dashboard_auth_provider.assert_not_called()
        assert rec.LAST_SKIP_REASON != ""

    def test_registers_exactly_one_compliant_provider(self, rec, monkeypatch):
        s = _strong_secret()
        monkeypatch.setenv(rec.SECRET_ENV_VAR, s)
        ctx = MagicMock()
        rec.register(ctx)
        assert ctx.register_dashboard_auth_provider.call_count == 1
        provider = ctx.register_dashboard_auth_provider.call_args.args[0]
        assert isinstance(provider, rec.RecommendationsSecretProvider)
        assert_protocol_compliance(type(provider))
        principal = provider.verify_token(token=s)
        assert principal is not None
        assert (principal.principal, principal.scopes) == (_PRINCIPAL, (_SCOPE,))
        assert rec.LAST_SKIP_REASON == ""

    def test_surrounding_whitespace_in_the_env_var_is_stripped(self, rec, monkeypatch):
        s = _strong_secret()
        monkeypatch.setenv(rec.SECRET_ENV_VAR, f"  {s}\n")
        ctx = MagicMock()
        rec.register(ctx)
        provider = ctx.register_dashboard_auth_provider.call_args.args[0]
        assert provider.verify_token(token=s) is not None

    def test_skip_reason_is_cleared_by_a_later_successful_register(
        self, rec, monkeypatch
    ):
        rec.register(MagicMock())  # no secret → skip reason recorded
        assert rec.LAST_SKIP_REASON != ""
        monkeypatch.setenv(rec.SECRET_ENV_VAR, _strong_secret())
        rec.register(MagicMock())
        assert rec.LAST_SKIP_REASON == ""

    @pytest.mark.parametrize("method", ["GET", "POST"])
    def test_register_does_not_register_any_token_route(
        self, rec, monkeypatch, method
    ):
        # Route ownership belongs to the Kanban plugin API, which registers it
        # unconditionally at import: the endpoint must be token-only (and so
        # answer 401) whether or not this credential plugin ever loaded.
        monkeypatch.setenv(rec.SECRET_ENV_VAR, _strong_secret())
        rec.register(MagicMock())
        assert token_auth.get_token_route_policy(method, _ROUTE_PATH) is None
        assert token_auth.get_token_route_policy("POST", "/api/gateway/drain") is None
