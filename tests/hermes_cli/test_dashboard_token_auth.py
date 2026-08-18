"""Contract tests for the generic non-interactive (bearer-token) auth seam.

Covers Task 2.0a: the reusable token-auth capability in the dashboard auth
framework — NOT the drain plugin (that's 2.0b/2.1). Asserts the ABC capability
flag, the registry filter, bearer extraction, provider stacking (verify_token),
and the route-agnostic middleware seam's fail-closed / 503 / pass-through
behaviour.
"""
from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Optional

import pytest

from hermes_cli.dashboard_auth import (
    DashboardAuthProvider,
    LoginStart,
    Session,
    TokenPrincipal,
    clear_providers,
    list_providers,
    list_session_providers,
    list_token_providers,
    register_provider,
)
from hermes_cli.dashboard_auth.base import ProviderError
from hermes_cli.dashboard_auth import token_auth


# --------------------------------------------------------------------------
# Test doubles
# --------------------------------------------------------------------------


class _OAuthOnly(DashboardAuthProvider):
    """A pure interactive provider — never token-authable."""

    name = "oauth-only"
    display_name = "OAuth Only"

    def start_login(self, *, redirect_uri):
        return LoginStart(redirect_url="x", cookie_payload={})

    def complete_login(self, *, code, state, code_verifier, redirect_uri):
        return Session("u", "e", "n", "o", self.name, 0, "a", "r")

    def verify_session(self, *, access_token):
        return None

    def refresh_session(self, *, refresh_token):
        return Session("u", "e", "n", "o", self.name, 0, "a", "r")

    def revoke_session(self, *, refresh_token):
        return None


class _TokenProvider(_OAuthOnly):
    """A token provider that accepts exactly one secret."""

    name = "tok"
    display_name = "Token Provider"
    supports_token = True

    def __init__(self, *, secret: str = "good-secret", scopes=("drain",)):
        self._secret = secret
        self._scopes = tuple(scopes)

    def verify_token(self, *, token: str) -> Optional[TokenPrincipal]:
        if token == self._secret:
            return TokenPrincipal(
                principal=self.name, provider=self.name, scopes=self._scopes
            )
        return None


class _UnreachableTokenProvider(_OAuthOnly):
    name = "tok-down"
    display_name = "Unreachable Token Provider"
    supports_token = True

    def verify_token(self, *, token: str) -> Optional[TokenPrincipal]:
        raise ProviderError("backing store down")


class _BuggyTokenProvider(_OAuthOnly):
    name = "tok-buggy"
    display_name = "Buggy Token Provider"
    supports_token = True

    def verify_token(self, *, token: str) -> Optional[TokenPrincipal]:
        raise RuntimeError("kaboom")


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _isolated_state(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    clear_providers()
    token_auth.clear_token_routes()
    yield
    clear_providers()
    token_auth.clear_token_routes()


class _FakeURL:
    def __init__(self, path):
        self.path = path


class _FakeClient:
    host = "1.2.3.4"


class _FakeRequest:
    """Minimal Request stand-in for the seam (no real Starlette needed)."""

    def __init__(self, path="/api/gateway/drain", headers=None, method="POST"):
        self.url = _FakeURL(path)
        self.headers = headers or {}
        self.client = _FakeClient()
        self.method = method

        class _State:
            pass

        self.state = _State()


def _run(coro):
    return asyncio.run(coro)


# --------------------------------------------------------------------------
# ABC + registry
# --------------------------------------------------------------------------


def test_oauth_provider_defaults_supports_token_false():
    assert _OAuthOnly().supports_token is False




class _NonInteractiveProvider(_TokenProvider):
    """A token-only credential with no interactive session."""

    name = "svc-cred"
    display_name = "Service Credential"
    supports_session = False


# --------------------------------------------------------------------------
# Bearer extraction
# --------------------------------------------------------------------------




# --------------------------------------------------------------------------
# authenticate_token (provider stacking)
# --------------------------------------------------------------------------


def test_authenticate_token_accepts_valid():
    register_provider(_TokenProvider(secret="good-secret"))
    req = _FakeRequest(headers={"authorization": "Bearer good-secret"})
    principal, unreachable = token_auth.authenticate_token(req)
    assert unreachable is None
    assert principal is not None
    assert principal.provider == "tok"
    assert principal.scopes == ("drain",)


def test_authenticate_token_rejects_wrong_secret():
    register_provider(_TokenProvider(secret="good-secret"))
    req = _FakeRequest(headers={"authorization": "Bearer wrong"})
    principal, unreachable = token_auth.authenticate_token(req)
    assert principal is None
    assert unreachable is None


def test_authenticate_token_stacks_first_match_wins():
    register_provider(_TokenProvider(secret="aaa"))
    second = _TokenProvider(secret="bbb")
    second.name = "tok2"
    register_provider(second)
    req = _FakeRequest(headers={"authorization": "Bearer bbb"})
    principal, _ = token_auth.authenticate_token(req)
    assert principal is not None and principal.provider == "tok2"


def test_authenticate_token_unreachable_then_valid_provider_wins():
    register_provider(_UnreachableTokenProvider())
    register_provider(_TokenProvider(secret="good"))
    req = _FakeRequest(headers={"authorization": "Bearer good"})
    principal, unreachable = token_auth.authenticate_token(req)
    # A later provider accepting the token beats the earlier outage.
    assert principal is not None and principal.provider == "tok"
    assert unreachable is None


def test_authenticate_token_buggy_provider_does_not_crash():
    register_provider(_BuggyTokenProvider())
    register_provider(_TokenProvider(secret="good"))
    req = _FakeRequest(headers={"authorization": "Bearer good"})
    principal, unreachable = token_auth.authenticate_token(req)
    assert principal is not None and principal.provider == "tok"


# --------------------------------------------------------------------------
# Middleware seam (route-agnostic)
# --------------------------------------------------------------------------


async def _call_next_ok(request):
    from fastapi.responses import JSONResponse

    return JSONResponse({"ok": True}, status_code=200)






def _spy_call_next():
    """A call_next stand-in that records whether the downstream was reached."""
    calls = {"n": 0}

    async def _call_next(request):
        calls["n"] += 1
        from fastapi.responses import JSONResponse

        return JSONResponse({"ok": True}, status_code=200)

    return _call_next, calls


def _register_route(path="/api/gateway/drain", method="POST", scope="drain"):
    token_auth.register_token_route(path, method=method, required_scope=scope)


# --------------------------------------------------------------------------
# register_token_route / get_token_route_policy / is_token_route —
# the (method, path, required_scope) registration contract (Item 31BI)
# --------------------------------------------------------------------------


def test_register_token_route_normalizes_method_case():
    token_auth.register_token_route("/x", method="post", required_scope="s")
    policy = token_auth.get_token_route_policy("POST", "/x")
    assert policy is not None
    assert policy.method == "POST"
    assert policy.path == "/x"
    assert policy.required_scope == "s"


def test_get_token_route_policy_lookup_is_method_case_insensitive():
    token_auth.register_token_route("/x", method="POST", required_scope="s")
    assert token_auth.get_token_route_policy("post", "/x") is not None
    assert token_auth.get_token_route_policy("PoSt", "/x") is not None


def test_register_token_route_idempotent_reregistration():
    token_auth.register_token_route("/x", method="POST", required_scope="s")
    # Re-registering the identical (method, path, scope) triple is a no-op,
    # not an error.
    token_auth.register_token_route("/x", method="post", required_scope="s")
    policy = token_auth.get_token_route_policy("POST", "/x")
    assert policy.required_scope == "s"


def test_register_token_route_conflicting_scope_rejected():
    token_auth.register_token_route("/x", method="POST", required_scope="s")
    with pytest.raises(ValueError):
        token_auth.register_token_route("/x", method="POST", required_scope="other")
    # The rejected re-registration must not have mutated the existing policy.
    assert token_auth.get_token_route_policy("POST", "/x").required_scope == "s"


@pytest.mark.parametrize("method", ["", "   ", "G3T", "GET POST", "get1"])
def test_register_token_route_rejects_blank_or_invalid_method(method):
    with pytest.raises(ValueError):
        token_auth.register_token_route("/x", method=method, required_scope="s")


@pytest.mark.parametrize("path", ["", "   "])
def test_register_token_route_rejects_blank_path(path):
    with pytest.raises(ValueError):
        token_auth.register_token_route(path, method="GET", required_scope="s")


@pytest.mark.parametrize("scope", ["", "   "])
def test_register_token_route_rejects_blank_scope(scope):
    with pytest.raises(ValueError):
        token_auth.register_token_route("/x", method="GET", required_scope=scope)


def test_get_token_route_policy_none_for_unregistered_path():
    assert token_auth.get_token_route_policy("GET", "/nope") is None


def test_is_token_route_true_only_for_exact_method_and_path():
    token_auth.register_token_route("/x", method="POST", required_scope="s")
    assert token_auth.is_token_route("/x", method="POST") is True
    assert token_auth.is_token_route("/x", method="GET") is False
    assert token_auth.is_token_route("/y", method="POST") is False
    assert token_auth.is_token_route("/x%20", method="POST") is False
    assert token_auth.is_token_route("/x ", method="POST") is False


# --------------------------------------------------------------------------
# Middleware seam — scope enforcement, fail-closed statuses, pass-through
# --------------------------------------------------------------------------


def test_middleware_correctly_scoped_principal_passes_and_sets_state():
    register_provider(_TokenProvider(secret="good", scopes=("drain",)))
    _register_route(scope="drain")
    req = _FakeRequest(headers={"authorization": "Bearer good"})
    call_next, calls = _spy_call_next()
    resp = _run(token_auth.token_auth_middleware(req, call_next))
    assert resp.status_code == 200
    assert calls["n"] == 1
    assert req.state.token_authenticated is True
    assert req.state.token_principal.principal == "tok"
    assert req.state.token_principal.scopes == ("drain",)


def test_middleware_wrong_scope_gets_generic_403_no_downstream_no_disclosure():
    register_provider(_TokenProvider(secret="good", scopes=("wrong-scope",)))
    _register_route(scope="drain")
    req = _FakeRequest(headers={"authorization": "Bearer good"})
    call_next, calls = _spy_call_next()
    resp = _run(token_auth.token_auth_middleware(req, call_next))
    assert resp.status_code == 403
    # Downstream handler never reached, and the request is not marked as
    # token-authenticated — this must never fall back to cookie/session auth.
    assert calls["n"] == 0
    assert not hasattr(req.state, "token_authenticated")
    assert not hasattr(req.state, "token_principal")
    body = resp.body.decode()
    assert "wrong-scope" not in body
    assert "drain" not in body


def test_middleware_missing_bearer_401():
    register_provider(_TokenProvider(secret="good", scopes=("drain",)))
    _register_route(scope="drain")
    req = _FakeRequest(headers={})
    call_next, calls = _spy_call_next()
    resp = _run(token_auth.token_auth_middleware(req, call_next))
    assert resp.status_code == 401
    assert calls["n"] == 0


def test_middleware_malformed_bearer_401():
    register_provider(_TokenProvider(secret="good", scopes=("drain",)))
    _register_route(scope="drain")
    req = _FakeRequest(headers={"authorization": "Basic Z29vZA=="})
    resp = _run(token_auth.token_auth_middleware(req, _call_next_ok))
    assert resp.status_code == 401


def test_middleware_unrecognized_bearer_401():
    register_provider(_TokenProvider(secret="good", scopes=("drain",)))
    _register_route(scope="drain")
    req = _FakeRequest(headers={"authorization": "Bearer nope"})
    resp = _run(token_auth.token_auth_middleware(req, _call_next_ok))
    assert resp.status_code == 401


def test_middleware_all_providers_unreachable_gets_503():
    register_provider(_UnreachableTokenProvider())
    _register_route(scope="drain")
    req = _FakeRequest(headers={"authorization": "Bearer anything"})
    resp = _run(token_auth.token_auth_middleware(req, _call_next_ok))
    assert resp.status_code == 503


def test_middleware_unreachable_provider_does_not_block_a_later_accepting_provider():
    register_provider(_UnreachableTokenProvider())
    register_provider(_TokenProvider(secret="good", scopes=("drain",)))
    _register_route(scope="drain")
    req = _FakeRequest(headers={"authorization": "Bearer good"})
    call_next, calls = _spy_call_next()
    resp = _run(token_auth.token_auth_middleware(req, call_next))
    assert resp.status_code == 200
    assert calls["n"] == 1


def test_middleware_method_mismatch_is_not_token_authenticated_and_passes_through():
    _register_route(scope="drain")  # registered for POST only
    req = _FakeRequest(headers={}, method="GET")
    call_next, calls = _spy_call_next()
    resp = _run(token_auth.token_auth_middleware(req, call_next))
    # No policy for GET on this path -> ordinary downstream gate handles it.
    assert resp.status_code == 200
    assert calls["n"] == 1
    assert not hasattr(req.state, "token_authenticated")


# --------------------------------------------------------------------------
# Item 32G-A: path templates + the "optional" dual-mode route
# --------------------------------------------------------------------------


def test_register_token_route_template_requires_a_placeholder():
    with pytest.raises(ValueError):
        token_auth.register_token_route_template(
            "/api/plugins/kanban/board", method="GET", required_scope="s"
        )


def test_register_token_route_template_matches_one_segment_only():
    token_auth.register_token_route_template(
        "/api/plugins/kanban/tasks/{task_id}", method="GET", required_scope="s"
    )
    assert token_auth.get_token_route_policy(
        "GET", "/api/plugins/kanban/tasks/abc123"
    ) is not None
    # No trailing slash, no extra segment, no empty segment.
    assert token_auth.get_token_route_policy(
        "GET", "/api/plugins/kanban/tasks/abc123/"
    ) is None
    assert token_auth.get_token_route_policy(
        "GET", "/api/plugins/kanban/tasks/abc123/attachments"
    ) is None
    assert token_auth.get_token_route_policy(
        "GET", "/api/plugins/kanban/tasks/"
    ) is None


def test_register_token_route_template_does_not_shadow_a_literal_route():
    # A literal route always wins over a template that would also match it.
    token_auth.register_token_route(
        "/api/plugins/kanban/tasks/special", method="GET", required_scope="literal-scope"
    )
    token_auth.register_token_route_template(
        "/api/plugins/kanban/tasks/{task_id}", method="GET", required_scope="template-scope"
    )
    policy = token_auth.get_token_route_policy("GET", "/api/plugins/kanban/tasks/special")
    assert policy.required_scope == "literal-scope"


def test_register_token_route_template_conflicting_scope_rejected():
    token_auth.register_token_route_template(
        "/x/{id}", method="GET", required_scope="s"
    )
    with pytest.raises(ValueError):
        token_auth.register_token_route_template(
            "/x/{id}", method="GET", required_scope="other"
        )


def test_register_token_route_template_idempotent_reregistration():
    token_auth.register_token_route_template("/x/{id}", method="GET", required_scope="s")
    token_auth.register_token_route_template("/x/{id}", method="get", required_scope="s")
    assert token_auth.get_token_route_policy("GET", "/x/123").required_scope == "s"


@pytest.mark.parametrize("template", ["/x/pre{id}", "/x/{id}tail", "/x/{bad-name}"])
def test_register_token_route_template_requires_a_whole_valid_segment(template):
    with pytest.raises(ValueError):
        token_auth.register_token_route_template(
            template, method="GET", required_scope="s"
        )


def test_register_token_route_template_rejects_duplicate_shape():
    token_auth.register_token_route_template(
        "/x/{task_id}", method="GET", required_scope="s"
    )
    with pytest.raises(ValueError):
        token_auth.register_token_route_template(
            "/x/{other_name}", method="GET", required_scope="other"
        )


def test_clear_token_routes_clears_templates_too():
    token_auth.register_token_route_template("/x/{id}", method="GET", required_scope="s")
    token_auth.clear_token_routes()
    assert token_auth.get_token_route_policy("GET", "/x/123") is None


def _register_optional_machine_route():
    token_auth.register_machine_token_family("svc1_", strict_audit=True)
    token_auth.register_token_route(
        "/api/plugins/kanban/board",
        method="GET",
        required_scope="kanban.read",
        optional=True,
        strict_audit=True,
    )


def test_optional_route_with_no_bearer_token_passes_through_unauthenticated():
    """A dual-mode route with no Authorization header at all defers to the
    ordinary interactive gates instead of 401ing — this is what lets an
    existing session/cookie route also accept a machine credential."""
    _register_optional_machine_route()
    req = _FakeRequest(path="/api/plugins/kanban/board", headers={}, method="GET")
    call_next, calls = _spy_call_next()
    resp = _run(token_auth.token_auth_middleware(req, call_next))
    assert resp.status_code == 200
    assert calls["n"] == 1
    assert not hasattr(req.state, "token_authenticated")


def test_optional_route_with_session_bearer_passes_to_interactive_auth_unchanged():
    _register_optional_machine_route()
    req = _FakeRequest(
        path="/api/plugins/kanban/board",
        headers={"authorization": "Bearer dashboard-session"},
        method="GET",
    )
    call_next, calls = _spy_call_next()
    resp = _run(token_auth.token_auth_middleware(req, call_next))
    assert resp.status_code == 200
    assert calls["n"] == 1
    assert not hasattr(req.state, "token_authenticated")


def test_optional_route_with_valid_machine_token_sets_authenticated_state():
    register_provider(_TokenProvider(secret="svc1_good", scopes=("kanban.read",)))
    _register_optional_machine_route()
    req = _FakeRequest(
        path="/api/plugins/kanban/board",
        headers={"authorization": "Bearer svc1_good"},
        method="GET",
    )
    calls = {"n": 0}
    async def call_next(request):
        calls["n"] += 1
        request.state.token_route_audited = True
        return await _call_next_ok(request)
    resp = _run(token_auth.token_auth_middleware(req, call_next))
    assert resp.status_code == 200
    assert calls["n"] == 1
    assert req.state.token_authenticated is True
    assert req.state.token_audit_strict is True


def test_strict_optional_machine_route_fails_closed_without_handler_audit():
    register_provider(_TokenProvider(secret="svc1_good", scopes=("kanban.read",)))
    _register_optional_machine_route()
    req = _FakeRequest(
        path="/api/plugins/kanban/board",
        headers={"authorization": "Bearer svc1_good"},
        method="GET",
    )
    call_next, calls = _spy_call_next()
    resp = _run(token_auth.token_auth_middleware(req, call_next))
    assert resp.status_code == 503
    assert calls["n"] == 1


def test_optional_route_with_invalid_machine_token_401s_never_falls_through():
    """A PRESENTED-but-wrong token on an optional route must not silently
    fall back to session auth — it fails, exactly like a non-optional route."""
    _register_optional_machine_route()
    req = _FakeRequest(
        path="/api/plugins/kanban/board",
        headers={"authorization": "Bearer svc1_garbage"},
        method="GET",
    )
    call_next, calls = _spy_call_next()
    resp = _run(token_auth.token_auth_middleware(req, call_next))
    assert resp.status_code == 401
    assert calls["n"] == 0


def test_optional_route_wrong_scope_still_403s_never_falls_through():
    register_provider(_TokenProvider(secret="svc1_good", scopes=("some-other-scope",)))
    _register_optional_machine_route()
    req = _FakeRequest(
        path="/api/plugins/kanban/board",
        headers={"authorization": "Bearer svc1_good"},
        method="GET",
    )
    call_next, calls = _spy_call_next()
    resp = _run(token_auth.token_auth_middleware(req, call_next))
    assert resp.status_code == 403
    assert calls["n"] == 0


@pytest.mark.parametrize("prefix", ["", "abc", "bad prefix", "a" * 33])
def test_machine_token_family_rejects_invalid_prefix(prefix):
    with pytest.raises(ValueError):
        token_auth.register_machine_token_family(prefix)


def test_machine_token_family_rejects_overlapping_prefixes():
    token_auth.register_machine_token_family("svc1_")
    with pytest.raises(ValueError):
        token_auth.register_machine_token_family("svc1_child_")


def test_reserved_machine_family_cannot_use_unregistered_method_or_path():
    _register_optional_machine_route()
    for method, path in (
        ("POST", "/api/plugins/kanban/board"),
        ("GET", "/api/plugins/kanban/not-registered"),
    ):
        req = _FakeRequest(
            path=path,
            headers={"authorization": "Bearer svc1_anything"},
            method=method,
        )
        call_next, calls = _spy_call_next()
        resp = _run(token_auth.token_auth_middleware(req, call_next))
        assert resp.status_code == 403
        assert calls["n"] == 0


def test_reserved_machine_family_rejects_method_override_headers():
    _register_optional_machine_route()
    req = _FakeRequest(
        path="/api/plugins/kanban/board",
        headers={
            "authorization": "Bearer svc1_anything",
            "x-http-method-override": "POST",
        },
        method="GET",
    )
    call_next, calls = _spy_call_next()
    resp = _run(token_auth.token_auth_middleware(req, call_next))
    assert resp.status_code == 400
    assert calls["n"] == 0


def test_transport_peer_ip_ignores_forwarded_headers():
    req = _FakeRequest(headers={"x-forwarded-for": "203.0.113.9"})
    assert token_auth.transport_peer_ip(req) == "1.2.3.4"


def test_strict_machine_denial_returns_503_when_audit_fails(monkeypatch):
    from hermes_cli.dashboard_auth.audit import AuditWriteError

    _register_optional_machine_route()
    monkeypatch.setattr(
        token_auth,
        "audit_log",
        lambda *args, **kwargs: (_ for _ in ()).throw(AuditWriteError("full")),
    )
    req = _FakeRequest(
        path="/api/plugins/kanban/board",
        headers={"authorization": "Bearer svc1_invalid"},
        method="GET",
    )
    call_next, calls = _spy_call_next()
    resp = _run(token_auth.token_auth_middleware(req, call_next))
    assert resp.status_code == 503
    assert calls["n"] == 0
