"""Tests for the DrainSecretProvider plugin (non-interactive bearer secret).

Task 2.0b / Item 31BI. Loads the bundled drain plugin module directly and
exercises:
  * the entropy gate (assess_secret_strength) — fail-closed on weak secrets,
  * constant-time verify_token returning a scoped TokenPrincipal,
  * the FIXED drain scope: the credential and the route policy are both minted
    from the exported ``DRAIN_SCOPE`` constant, no caller or config value can
    choose another one, and a config that asks for one registers nothing at all,
  * the register(ctx) entry point's env/config resolution, skip reasons, and
    the (method, path, required_scope) token-route registration: exactly
    ``POST /api/gateway/drain`` under the fixed drain scope, and nothing at all
    when the credential is missing, weak, or mis-scoped,
  * end-to-end confinement over HTTP: the drain and recommendations service
    credentials each work on their own endpoint only.
"""
from __future__ import annotations

import secrets
from pathlib import Path
from unittest.mock import MagicMock

import pytest

import plugins.dashboard_auth.drain as drain_plugin
import plugins.dashboard_auth.recommendations as rec_plugin
from hermes_cli.dashboard_auth import TokenPrincipal, assert_protocol_compliance
from hermes_cli.dashboard_auth import clear_providers, register_provider
from hermes_cli.dashboard_auth import token_auth

# A path this plugin must never claim — used to show its registration is
# confined to the drain endpoint.
_OTHER_ROUTE_PATH = "/api/plugins/kanban/recommendations"
# The other bundled machine credential's scope. The drain scope must never be
# this, or one credential would buy both endpoints.
_OTHER_SCOPE = "kanban:recommendations:read"


@pytest.fixture(scope="module")
def drain():
    return drain_plugin


def _rec_route_policy():
    """The recommendations route policy, or None. Owned by the Kanban plugin API."""
    return token_auth.get_token_route_policy(
        rec_plugin.RECOMMENDATIONS_ROUTE_METHOD, rec_plugin.RECOMMENDATIONS_ROUTE_PATH
    )


@pytest.fixture(autouse=True)
def _clean_env_and_routes(monkeypatch):
    monkeypatch.delenv("HERMES_DASHBOARD_DRAIN_SECRET", raising=False)
    monkeypatch.delenv(rec_plugin.SECRET_ENV_VAR, raising=False)
    preexisting_rec = _rec_route_policy()
    token_auth.clear_token_routes()
    yield
    # The route registry is process-global and the Kanban plugin API registers
    # the recommendations route exactly once, at import. Put it back on the way
    # out (whether it predated this test or the test's own app import installed
    # it) so wiping this module's drain-route state can never leave another
    # module's endpoint un-gated.
    restore = _rec_route_policy() or preexisting_rec
    token_auth.clear_token_routes()
    if restore is not None:
        token_auth.register_token_route(
            restore.path, method=restore.method, required_scope=restore.required_scope
        )


def _strong_secret() -> str:
    # token_urlsafe(32) → 43 url-safe-b64 chars ≈ 256 bits.
    return secrets.token_urlsafe(32)


def _drain_policy():
    """The policy registered for the drain endpoint, or None."""
    return token_auth.get_token_route_policy("POST", drain_plugin.DRAIN_ROUTE_PATH)


def _foreign_route_keys():
    """(method, path) pairs the drain plugin must never register.

    Covers other verbs on the drain path itself, near-miss paths (prefix,
    suffix, trailing slash) and another plugin's token route — enough to show
    the registration is one exact (method, path) pair and not a pattern.
    """
    path = drain_plugin.DRAIN_ROUTE_PATH
    return [
        ("GET", path),
        ("PUT", path),
        ("PATCH", path),
        ("DELETE", path),
        ("POST", path + "/"),
        ("POST", path + "-force"),
        ("POST", "/api/gateway"),
        ("POST", _OTHER_ROUTE_PATH),
        ("GET", _OTHER_ROUTE_PATH),
    ]


# ---------------------------------------------------------------------------
# Entropy gate
# ---------------------------------------------------------------------------


class TestEntropyGate:
    def test_strong_secret_passes(self, drain):
        assert drain.assess_secret_strength(_strong_secret()) is None

    def test_empty_rejected(self, drain):
        assert drain.assess_secret_strength("") is not None

    def test_too_short_rejected(self, drain):
        # 42 chars — one under the 43-char bar.
        assert drain.assess_secret_strength("a1B2c3" * 7) is not None

    def test_long_but_repeated_rejected(self, drain):
        # 60 chars, one distinct character → low distinct count + low entropy.
        assert drain.assess_secret_strength("a" * 60) is not None


    def test_custom_min_chars_enforced(self, drain):
        s = _strong_secret()  # 43 chars
        assert drain.assess_secret_strength(s, min_chars=999) is not None


# ---------------------------------------------------------------------------
# Provider behaviour
# ---------------------------------------------------------------------------


class TestProvider:
    def test_protocol_compliance(self, drain):
        assert_protocol_compliance(drain.DrainSecretProvider)

    def test_supports_token_flag(self, drain):
        p = drain.DrainSecretProvider(secret=_strong_secret())
        assert p.supports_token is True

    def test_is_non_interactive(self, drain):
        # Excluded from interactive surfaces via list_session_providers().
        p = drain.DrainSecretProvider(secret=_strong_secret())
        assert p.supports_session is False

    def test_verify_token_accepts_matching_secret(self, drain):
        s = _strong_secret()
        p = drain.DrainSecretProvider(secret=s, scope="drain")
        principal = p.verify_token(token=s)
        assert isinstance(principal, TokenPrincipal)
        assert principal.principal == "drain-control"
        assert principal.provider == "drain-secret"
        assert principal.scopes == ("drain",)


    def test_verify_token_vouches_for_the_fixed_scope_by_default(self, drain):
        s = _strong_secret()
        principal = drain.DrainSecretProvider(secret=s).verify_token(token=s)
        assert principal.scopes == (drain.DRAIN_SCOPE,)
        assert principal.principal == drain.DRAIN_PRINCIPAL

    def test_explicit_fixed_scope_is_accepted(self, drain):
        s = _strong_secret()
        p = drain.DrainSecretProvider(secret=s, scope=drain.DRAIN_SCOPE)
        assert p.verify_token(token=s).scopes == (drain.DRAIN_SCOPE,)

    @pytest.mark.parametrize(
        "scope",
        ["lifecycle", _OTHER_SCOPE, "DRAIN", "drain:write", "", "   ", "*"],
        ids=["arbitrary", "recommendations", "wrong-case", "superstring",
             "empty", "blank", "wildcard"],
    )
    def test_constructor_refuses_any_other_scope(self, drain, scope):
        # The scope IS the authority boundary: a caller may omit it or state
        # the fixed value, and nothing else builds a provider at all.
        with pytest.raises(ValueError):
            drain.DrainSecretProvider(secret=_strong_secret(), scope=scope)

    def test_fixed_scope_is_not_the_recommendations_scope(self, drain):
        s = _strong_secret()
        drain_scopes = drain.DrainSecretProvider(secret=s).verify_token(token=s).scopes
        rec_scopes = rec_plugin.RecommendationsSecretProvider(
            secret=s
        ).verify_token(token=s).scopes
        # Disjoint capability sets: neither credential satisfies the other
        # route's scope check at the seam.
        assert set(drain_scopes).isdisjoint(rec_scopes)
        assert drain.DRAIN_SCOPE != _OTHER_SCOPE

    def test_verify_token_rejects_empty(self, drain):
        p = drain.DrainSecretProvider(secret=_strong_secret())
        assert p.verify_token(token="") is None


    def test_construction_rejects_weak_secret(self, drain):
        with pytest.raises(ValueError):
            drain.DrainSecretProvider(secret="weak")


    def test_interactive_methods_raise(self, drain):
        p = drain.DrainSecretProvider(secret=_strong_secret())
        with pytest.raises(NotImplementedError):
            p.start_login(redirect_uri="r")
        with pytest.raises(NotImplementedError):
            p.complete_login(code="c", state="s", code_verifier="v", redirect_uri="r")
        with pytest.raises(NotImplementedError):
            p.refresh_session(refresh_token="r")


# ---------------------------------------------------------------------------
# register() entry point
# ---------------------------------------------------------------------------


class TestRegister:
    def test_skips_when_no_secret(self, drain, monkeypatch):
        monkeypatch.setattr(drain, "_load_config_drain_auth_section", lambda: {})
        ctx = MagicMock()
        drain.register(ctx)
        ctx.register_dashboard_auth_provider.assert_not_called()
        assert "HERMES_DASHBOARD_DRAIN_SECRET" in drain.LAST_SKIP_REASON
        # No credential → no token route at all, for any verb.
        assert _drain_policy() is None
        assert not token_auth.is_token_route(
            drain.DRAIN_ROUTE_PATH, method=drain.DRAIN_ROUTE_METHOD
        )

    @pytest.mark.parametrize(
        "weak",
        ["tooweak", "   ", "a" * 60, "ab" * 30, "0123456789" * 5],
        ids=["short", "blank", "one-char", "two-chars", "repeating-digits"],
    )
    def test_skips_and_fails_closed_on_weak_secret(self, drain, monkeypatch, weak):
        monkeypatch.setenv("HERMES_DASHBOARD_DRAIN_SECRET", weak)
        monkeypatch.setattr(drain, "_load_config_drain_auth_section", lambda: {})
        ctx = MagicMock()
        drain.register(ctx)
        ctx.register_dashboard_auth_provider.assert_not_called()
        assert drain.LAST_SKIP_REASON != ""
        # fail-closed: the route is NOT token-authable, so it stays gated by
        # the ordinary cookie/session path — a weak secret never opens it.
        assert _drain_policy() is None

    def test_registers_with_strong_env_secret(self, drain, monkeypatch):
        s = _strong_secret()
        monkeypatch.setenv("HERMES_DASHBOARD_DRAIN_SECRET", s)
        monkeypatch.setattr(drain, "_load_config_drain_auth_section", lambda: {})
        ctx = MagicMock()
        drain.register(ctx)
        ctx.register_dashboard_auth_provider.assert_called_once()
        provider = ctx.register_dashboard_auth_provider.call_args.args[0]
        assert isinstance(provider, drain.DrainSecretProvider)
        assert provider.verify_token(token=s) is not None
        assert drain.LAST_SKIP_REASON == ""
        # Exactly one route policy: POST on the exact drain path, demanding
        # the same scope the credential vouches for.
        policy = _drain_policy()
        assert policy is not None
        assert (policy.method, policy.path) == ("POST", "/api/gateway/drain")
        assert policy.required_scope == "drain"
        assert provider.verify_token(token=s).scopes == (policy.required_scope,)

    @pytest.mark.parametrize("method,path", _foreign_route_keys())
    def test_registers_no_other_method_or_path(self, drain, monkeypatch, method, path):
        monkeypatch.setenv("HERMES_DASHBOARD_DRAIN_SECRET", _strong_secret())
        monkeypatch.setattr(drain, "_load_config_drain_auth_section", lambda: {})
        drain.register(MagicMock())
        assert token_auth.get_token_route_policy(method, path) is None

    def test_repeated_registration_is_idempotent(self, drain, monkeypatch):
        s = _strong_secret()
        monkeypatch.setenv("HERMES_DASHBOARD_DRAIN_SECRET", s)
        monkeypatch.setattr(drain, "_load_config_drain_auth_section", lambda: {})
        drain.register(MagicMock())
        first = _drain_policy()
        # A second plugin-load pass must not raise or change the authority.
        drain.register(MagicMock())
        assert _drain_policy() == first

    @pytest.mark.parametrize(
        "section",
        [{}, {"min_secret_chars": 43}, {"scope": "drain"}, {"scope": "  drain  "}],
        ids=["no-scope-key", "other-knob-only", "explicit-fixed-scope",
             "explicit-fixed-scope-padded"],
    )
    def test_fixed_scope_registration_succeeds(self, drain, monkeypatch, section):
        # Omitting the knob and spelling out the fixed scope are both valid;
        # each yields the same credential + route authority.
        s = _strong_secret()
        monkeypatch.setenv("HERMES_DASHBOARD_DRAIN_SECRET", s)
        monkeypatch.setattr(drain, "_load_config_drain_auth_section", lambda: section)
        ctx = MagicMock()
        drain.register(ctx)
        provider = ctx.register_dashboard_auth_provider.call_args.args[0]
        assert drain.LAST_SKIP_REASON == ""
        assert provider.verify_token(token=s).scopes == (drain.DRAIN_SCOPE,)
        # Credential and route policy are minted from one value: they cannot
        # drift apart into "recognised but never authorised".
        assert _drain_policy().required_scope == drain.DRAIN_SCOPE

    @pytest.mark.parametrize(
        "scope",
        ["lifecycle", _OTHER_SCOPE, "DRAIN", "drain:write", "", "   ", "*", 42],
        ids=["arbitrary", "recommendations", "wrong-case", "superstring",
             "empty", "blank", "wildcard", "non-string"],
    )
    def test_configured_other_scope_registers_nothing(self, drain, monkeypatch, scope):
        # A config asking for any scope but the fixed one is refused BEFORE the
        # provider and BEFORE the route: no credential to leak, and the drain
        # endpoint is not a token route at all (it stays on the cookie gate),
        # so there is no half-registered, mis-scoped state.
        s = _strong_secret()
        monkeypatch.setenv("HERMES_DASHBOARD_DRAIN_SECRET", s)
        monkeypatch.setattr(
            drain, "_load_config_drain_auth_section", lambda: {"scope": scope}
        )
        ctx = MagicMock()
        drain.register(ctx)
        ctx.register_dashboard_auth_provider.assert_not_called()
        assert drain.LAST_SKIP_REASON != ""
        assert _drain_policy() is None
        for method, path in _foreign_route_keys():
            assert token_auth.get_token_route_policy(method, path) is None

    def test_bad_scope_config_does_not_survive_a_later_good_load(
        self, drain, monkeypatch
    ):
        s = _strong_secret()
        monkeypatch.setenv("HERMES_DASHBOARD_DRAIN_SECRET", s)
        monkeypatch.setattr(
            drain, "_load_config_drain_auth_section", lambda: {"scope": _OTHER_SCOPE}
        )
        drain.register(MagicMock())
        assert _drain_policy() is None
        monkeypatch.setattr(drain, "_load_config_drain_auth_section", lambda: {})
        ctx = MagicMock()
        drain.register(ctx)
        assert drain.LAST_SKIP_REASON == ""
        assert _drain_policy().required_scope == drain.DRAIN_SCOPE
        provider = ctx.register_dashboard_auth_provider.call_args.args[0]
        assert provider.verify_token(token=s).scopes == (drain.DRAIN_SCOPE,)

    def test_config_min_secret_chars_can_reject_otherwise_ok_secret(
        self, drain, monkeypatch
    ):
        s = _strong_secret()  # 43 chars — fine by default, too short at 999
        monkeypatch.setenv("HERMES_DASHBOARD_DRAIN_SECRET", s)
        monkeypatch.setattr(
            drain,
            "_load_config_drain_auth_section",
            lambda: {"min_secret_chars": 999},
        )
        ctx = MagicMock()
        drain.register(ctx)
        ctx.register_dashboard_auth_provider.assert_not_called()
        assert "rejected" in drain.LAST_SKIP_REASON
        # min_secret_chars still gates the credential, and a rejection there is
        # equally total: no provider, no route.
        assert _drain_policy() is None

    def test_config_min_secret_chars_below_the_bar_still_admits_a_strong_secret(
        self, drain, monkeypatch
    ):
        s = _strong_secret()
        monkeypatch.setenv("HERMES_DASHBOARD_DRAIN_SECRET", s)
        monkeypatch.setattr(
            drain,
            "_load_config_drain_auth_section",
            lambda: {"min_secret_chars": 8, "scope": drain_plugin.DRAIN_SCOPE},
        )
        ctx = MagicMock()
        drain.register(ctx)
        provider = ctx.register_dashboard_auth_provider.call_args.args[0]
        assert provider.verify_token(token=s).scopes == (drain.DRAIN_SCOPE,)


# ---------------------------------------------------------------------------
# End-to-end: one credential buys exactly one endpoint
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clean_providers():
    clear_providers()
    yield
    clear_providers()


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    from hermes_cli import kanban_db as kb

    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


@pytest.fixture
def client(kanban_home):
    from fastapi.testclient import TestClient
    from hermes_cli import web_server as ws

    return TestClient(ws.app)


@pytest.fixture
def drain_marker_calls(monkeypatch):
    """Count drain-marker writes without touching the real HERMES_HOME."""
    from gateway import drain_control

    calls = {"write": 0, "clear": 0}

    def _write(**kwargs):
        calls["write"] += 1
        return {"requested_at": "1970-01-01T00:00:00Z", "suppress_notification": False}

    def _clear(**kwargs):
        calls["clear"] += 1
        return False

    monkeypatch.setattr(drain_control, "write_drain_request", _write)
    monkeypatch.setattr(drain_control, "clear_drain_request", _clear)
    monkeypatch.setattr(drain_control, "drain_requested", lambda **k: False)
    return calls


def _bearer(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


def _rec_params() -> dict:
    return {
        "board": "default",
        "project_id": "proj-1",
        "target_profile": "worker",
        "limit": 50,
    }


class TestCredentialConfinement:
    def test_each_credential_works_only_on_its_own_endpoint(
        self, drain, client, monkeypatch, drain_marker_calls
    ):
        """Two distinct strong credentials, two endpoints, no crossover.

        Both plugins are loaded the way production loads them (env secret +
        their own ``register()``), the verified providers are stacked in the
        one shared registry, and both token routes are live at once — so a
        refusal below is the seam's scope check, not a missing registration.
        """
        drain_secret, rec_secret = _strong_secret(), _strong_secret()
        assert drain_secret != rec_secret
        monkeypatch.setenv("HERMES_DASHBOARD_DRAIN_SECRET", drain_secret)
        monkeypatch.setenv(rec_plugin.SECRET_ENV_VAR, rec_secret)
        monkeypatch.setattr(drain, "_load_config_drain_auth_section", lambda: {})

        drain_ctx, rec_ctx = MagicMock(), MagicMock()
        drain.register(drain_ctx)  # also registers POST /api/gateway/drain
        rec_plugin.register(rec_ctx)
        # The recommendations route belongs to the Kanban plugin API (it
        # registers it unconditionally at import); re-register it from that
        # plugin's exported constants because this module wipes the route
        # registry between tests.
        token_auth.register_token_route(
            rec_plugin.RECOMMENDATIONS_ROUTE_PATH,
            method=rec_plugin.RECOMMENDATIONS_ROUTE_METHOD,
            required_scope=rec_plugin.RECOMMENDATIONS_SCOPE,
        )
        for ctx in (drain_ctx, rec_ctx):
            register_provider(ctx.register_dashboard_auth_provider.call_args.args[0])

        # Each credential on its own exact endpoint: accepted.
        own_drain = client.post(
            drain.DRAIN_ROUTE_PATH, json={"action": "cancel"},
            headers=_bearer(drain_secret),
        )
        assert own_drain.status_code == 200, own_drain.text
        assert drain_marker_calls == {"write": 0, "clear": 1}
        own_rec = client.get(
            _OTHER_ROUTE_PATH, params=_rec_params(), headers=_bearer(rec_secret)
        )
        assert own_rec.status_code == 200, own_rec.text
        assert own_rec.json()["items"] == []

        # Each credential on the other's endpoint: never a 200.
        crossed_rec = client.post(
            drain.DRAIN_ROUTE_PATH, json={"action": "drain"},
            headers=_bearer(rec_secret),
        )
        assert crossed_rec.status_code != 200
        assert crossed_rec.status_code == 403
        crossed_drain = client.get(
            _OTHER_ROUTE_PATH, params=_rec_params(), headers=_bearer(drain_secret)
        )
        assert crossed_drain.status_code != 200
        assert crossed_drain.status_code == 403

        # The refusals are generic (no scope disclosure) and the drain handler
        # was never reached by the recommendations credential.
        for body in (crossed_rec.text, crossed_drain.text):
            assert drain.DRAIN_SCOPE not in body
            assert _OTHER_SCOPE not in body
        assert drain_marker_calls == {"write": 0, "clear": 1}

    def test_a_mis_scoped_config_leaves_the_drain_endpoint_untouched(
        self, drain, client, monkeypatch, drain_marker_calls
    ):
        """A config asking for the recommendations scope registers nothing.

        The drain credential must not become usable under someone else's
        capability, and the endpoint must not end up token-gated by a policy
        no credential can satisfy.
        """
        drain_secret = _strong_secret()
        monkeypatch.setenv("HERMES_DASHBOARD_DRAIN_SECRET", drain_secret)
        monkeypatch.setattr(
            drain, "_load_config_drain_auth_section", lambda: {"scope": _OTHER_SCOPE}
        )
        ctx = MagicMock()
        drain.register(ctx)
        ctx.register_dashboard_auth_provider.assert_not_called()
        assert token_auth.get_token_route_policy("POST", drain.DRAIN_ROUTE_PATH) is None
        # No token route, no provider: the bearer buys nothing and the marker
        # is never written.
        r = client.post(
            drain.DRAIN_ROUTE_PATH, json={"action": "drain"},
            headers=_bearer(drain_secret),
        )
        assert r.status_code != 200
        assert drain_marker_calls == {"write": 0, "clear": 0}
