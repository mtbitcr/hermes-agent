"""Card B: a still-valid Raphael service key asks for its own successor.

Owner decision of 16 November: while its current service key is still valid,
the owner app may get its own next key, and that successor may retire the
older ones. Every key, secret and digest here is generated in a temporary
``HERMES_HOME``; the tests assert their absence and commit none.
"""

import hashlib
import json
import logging
import re
import sys
import time
import uuid
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from hermes_cli.dashboard_auth import TokenPrincipal, clear_providers, register_provider
from hermes_cli.dashboard_auth import token_auth
from hermes_cli.dashboard_auth.audit import AuditEvent, AuditWriteError
from plugins.dashboard_auth import raphael_workspace
from plugins.dashboard_auth.raphael_workspace import (
    AutomationsManageTokenProvider,
    ConnectionsManageTokenProvider,
    ModelsManageTokenProvider,
    RecommendationsReadTokenProvider,
    WorkspaceReadTokenProvider,
    token_store,
)

DAY = 86400
BASE = "/api/service-access"
#: The four fixed families that may renew themselves; recommendations never may.
FAMILIES = ("workspace", "connections", "automations", "models")
#: One existing authenticated route of each family, read with that family's key.
EXISTING_ROUTES = {
    "workspace": "/api/plugins/kanban/boards",
    "connections": "/api/mcp/servers",
    "automations": "/api/cron/jobs",
    "models": "/api/model/info",
}


@pytest.fixture
def home(tmp_path, monkeypatch):
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    return hermes_home


@pytest.fixture
def clock(monkeypatch):
    """The registry clock, moved by hand: renewal turns on whole seconds."""
    now = {"t": int(time.time())}
    monkeypatch.setattr(token_store.time, "time", lambda: now["t"])
    return now


@pytest.fixture
def client(home):
    from plugins.dashboard_auth.raphael_workspace import renewal

    clear_providers()
    token_auth.clear_token_routes()
    for provider in (
        WorkspaceReadTokenProvider(),
        RecommendationsReadTokenProvider(),
        ConnectionsManageTokenProvider(),
        AutomationsManageTokenProvider(),
        ModelsManageTokenProvider(),
    ):
        register_provider(provider)
    renewal.register_renewal_routes()
    app = FastAPI()
    app.include_router(renewal.router)

    @app.middleware("http")
    async def machine_auth(request, call_next):
        return await token_auth.token_auth_middleware(request, call_next)

    with TestClient(app) as test_client:
        yield test_client
    clear_providers()
    token_auth.clear_token_routes()


@pytest.fixture
def served(home):
    """The real served application, started as ``hermes dashboard`` starts it.

    The test mounts no router: every route is the one ``hermes_cli/web_server.py``
    mounts. As plugin discovery does before the server starts, the plugin entry
    point registers the providers and token routes; the families' existing token
    routes, registered when their modules were imported, are installed again
    (idempotent) because earlier tests may have cleared them. The bind is
    non-loopback with the auth gate on, as the owner app reaches it.
    """
    from hermes_cli import kanban_db, projects_db, web_server
    from hermes_cli.web_routers import cron, mcp

    clear_providers()
    token_auth.clear_token_routes()
    context = MagicMock()
    context.register_dashboard_auth_provider.side_effect = register_provider
    raphael_workspace.register(context)
    sys.modules["hermes_dashboard_plugin_kanban"]._register_workspace_machine_routes()
    mcp._register_connections_machine_routes()
    cron._register_automations_machine_routes()
    # A workspace key reads only its own fixed project and board.
    project_dir = home / "project"
    project_dir.mkdir()
    with projects_db.connect_closing() as conn:
        project_id = projects_db.create_project(
            conn,
            name="Raphael Workspace",
            slug=token_store.PROJECT,
            primary_path=str(project_dir),
            board_slug=token_store.BOARD,
        )
    kanban_db.create_board(token_store.BOARD, name="Raphael Workspace", project_id=project_id)
    state = web_server.app.state
    previous = {name: getattr(state, name, None) for name in ("bound_host", "bound_port", "auth_required")}
    state.bound_host, state.bound_port, state.auth_required = "agent.example.test", 443, True
    try:
        with TestClient(web_server.app, base_url="https://agent.example.test") as test_client:
            yield test_client
    finally:
        for name, value in previous.items():
            setattr(state, name, value)
        clear_providers()
        token_auth.clear_token_routes()
        kanban_db._INITIALIZED_PATHS.clear()
        projects_db._INITIALIZED_PATHS.clear()


def _issue(home, surface="workspace", ttl_seconds=20 * DAY):
    """A generated key in the temporary home; 20 days means it is already due."""
    out_dir = home / "plaintext-output"
    out_dir.mkdir(mode=0o700, exist_ok=True)
    out_path = out_dir / f"{uuid.uuid4().hex}.token"
    record = token_store.issue(out_path=out_path, ttl_seconds=ttl_seconds, surface=surface)
    return record, out_path.read_text(encoding="utf-8").strip()


def _unknown_bearer():
    return f"{token_store.TOKEN_PREFIX}{uuid.uuid4().hex}.{uuid.uuid4().hex}"


def _secret_material(bearer):
    """The bearer, its secret and the secret's digest: none may be written anywhere."""
    secret = bearer.split(".", 1)[1]
    return (bearer, secret, hashlib.sha256(secret.encode("utf-8")).hexdigest())


def _leaks(bearers, *texts):
    """How many pieces of secret material show up in ``texts`` (never the values)."""
    return sum(1 for b in bearers for value in _secret_material(b) for text in texts if value in text)


def _verifies(bearer):
    token_id, secret = bearer.split(".", 1)
    return token_store.verify(token_id, secret) is not None


def _state():
    """``{token_id: revoked_at}`` for every record: what a refusal must leave as it was."""
    return {r.token_id: r.revoked_at for r in token_store.load_records()}


def _post(client, surface, action, bearer):
    return client.post(f"{BASE}/{surface}/{action}", headers={"Authorization": f"Bearer {bearer}"})


def _audit_text(home):
    path = home / "logs" / "dashboard-auth.log"
    return path.read_text(encoding="utf-8") if path.exists() else ""


def _audit_events(home, event):
    records = [json.loads(line) for line in _audit_text(home).splitlines() if line.strip()]
    return [r for r in records if r.get("event") == event.value]


def _fail_store_write(_records):
    raise token_store.TokenStoreError("could not write token store")


def test_renew_refused_when_not_due_outside_21_days(home, clock, client):
    caller, bearer = _issue(home, ttl_seconds=21 * DAY + 60)
    clock["t"] += 1
    before = _state()

    with pytest.raises(token_store.NotDue):
        token_store.renew(caller.token_id)
    response = _post(client, "workspace", "renew", bearer)

    assert response.status_code == 409
    assert response.json() == {"error": "not_due"}
    assert _state() == before
    clock["t"] += 59  # it now ends exactly 21 days from now: due
    assert _post(client, "workspace", "renew", bearer).status_code == 200


@pytest.mark.parametrize("surface", FAMILIES)
def test_renew_success_keeps_caller_valid(home, clock, client, surface):
    caller, bearer = _issue(home, surface=surface)
    clock["t"] += 60

    response = _post(client, surface, "renew", bearer)

    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"token", "expires_at"}
    new_id = body["token"].split(".", 1)[0]
    record = token_store.find(new_id)
    policy = token_store.policy_for_surface(surface)
    assert record is not None and record.status == "active"
    assert token_store.policy_for_token_id(new_id) == policy
    assert body["expires_at"] == record.expires_at == clock["t"] + policy.default_ttl_seconds
    assert _verifies(body["token"])
    assert _verifies(bearer), "renewal must never revoke the calling key"
    issued = _audit_events(home, AuditEvent.MACHINE_TOKEN_ISSUED)
    assert [(r["credential_id"], r["replaces_credential_id"]) for r in issued[1:]] == [
        (new_id, caller.token_id)
    ]


def test_renew_race_inside_15_minutes_is_refused(home, clock, client):
    caller, bearer = _issue(home)
    clock["t"] += 60
    first = _post(client, "workspace", "renew", bearer)
    assert first.status_code == 200
    clock["t"] += 15 * 60 - 1
    before = _state()

    with pytest.raises(token_store.AlreadyRenewed):
        token_store.renew(caller.token_id)
    second = _post(client, "workspace", "renew", bearer)

    assert second.status_code == 409
    assert second.json() == {"error": "already_renewed"}
    assert _state() == before
    assert _verifies(bearer) and _verifies(first.json()["token"])


def test_renew_after_15_minutes_revokes_orphan_successor(home, clock, client):
    older, older_bearer = _issue(home)  # issued before the caller: never a successor
    clock["t"] += 1
    caller, bearer = _issue(home)
    other, other_bearer = _issue(home, surface="connections")
    clock["t"] += 60
    orphan, _ = token_store.renew(caller.token_id)  # minted, never saved by the app
    clock["t"] += 15 * 60

    response = _post(client, "workspace", "renew", bearer)

    assert response.status_code == 200
    assert token_store.find(orphan.token_id).status == "revoked"
    assert _verifies(response.json()["token"])
    assert _verifies(bearer) and _verifies(older_bearer) and _verifies(other_bearer)
    revoked = _audit_events(home, AuditEvent.MACHINE_TOKEN_REVOKED)
    assert [r["credential_id"] for r in revoked] == [orphan.token_id]


def test_retire_older_revokes_only_older_keys_of_same_surface(home, clock, client):
    oldest, oldest_bearer = _issue(home)
    clock["t"] += 10
    caller, bearer = _issue(home)
    other, other_bearer = _issue(home, surface="connections")
    clock["t"] += 10
    successor = _post(client, "workspace", "renew", bearer).json()["token"]
    clock["t"] += 10
    newer, newer_bearer = _issue(home)

    response = _post(client, "workspace", "retire-older", successor)

    assert response.status_code == 200
    assert response.json() == {"retired": 2}
    assert not _verifies(oldest_bearer) and not _verifies(bearer)
    assert _verifies(successor) and _verifies(newer_bearer) and _verifies(other_bearer)
    assert token_store.retire_older(successor.split(".", 1)[0]) == 0


@pytest.mark.parametrize("surface", FAMILIES)
def test_duplicate_renewal_in_the_issuance_second_is_already_renewed(home, clock, client, surface):
    caller, bearer = _issue(home, surface=surface)
    first = _post(client, surface, "renew", bearer)  # still the caller's issuance second
    assert first.status_code == 200
    assert token_store.find(first.json()["token"].split(".", 1)[0]).issued_at == caller.issued_at
    before = _state()

    second = _post(client, surface, "renew", bearer)

    assert second.status_code == 409
    assert second.json() == {"error": "already_renewed"}
    assert _state() == before
    assert _verifies(bearer) and _verifies(first.json()["token"])


@pytest.mark.parametrize("surface", FAMILIES)
def test_retry_one_second_after_a_same_second_renewal_is_already_renewed(home, clock, client, surface):
    caller, bearer = _issue(home, surface=surface)
    first = _post(client, surface, "renew", bearer)  # in the caller's issuance second
    assert first.status_code == 200
    assert token_store.find(first.json()["token"].split(".", 1)[0]).issued_at == caller.issued_at
    clock["t"] += 1
    before = _state()

    retry = _post(client, surface, "renew", bearer)

    assert retry.status_code == 409
    assert retry.json() == {"error": "already_renewed"}
    assert _state() == before
    assert _verifies(bearer) and _verifies(first.json()["token"])


@pytest.mark.parametrize("surface", FAMILIES)
def test_same_second_orphan_successor_is_revoked_after_15_minutes(home, clock, client, surface):
    older, older_bearer = _issue(home, surface=surface)  # same second, before the caller: never a successor
    caller, bearer = _issue(home, surface=surface)
    orphan, _ = token_store.renew(caller.token_id)  # same second; minted, never saved by the app
    assert older.issued_at == caller.issued_at == orphan.issued_at
    clock["t"] += 15 * 60 + 1

    response = _post(client, surface, "renew", bearer)

    assert response.status_code == 200
    assert token_store.find(orphan.token_id).status == "revoked"
    assert _verifies(bearer) and _verifies(response.json()["token"]) and _verifies(older_bearer)
    revoked = _audit_events(home, AuditEvent.MACHINE_TOKEN_REVOKED)
    assert [r["credential_id"] for r in revoked] == [orphan.token_id]


@pytest.mark.parametrize("surface", FAMILIES)
def test_retire_older_by_a_same_second_successor_revokes_its_predecessor(home, clock, client, surface):
    predecessor, predecessor_bearer = _issue(home, surface=surface)
    other, other_bearer = _issue(home, surface=FAMILIES[FAMILIES.index(surface) - 1])
    successor = _post(client, surface, "renew", predecessor_bearer).json()["token"]
    newer, newer_bearer = _issue(home, surface=surface)  # same second, after the successor
    successor_record = token_store.find(successor.split(".", 1)[0])
    assert predecessor.issued_at == other.issued_at == successor_record.issued_at == newer.issued_at

    response = _post(client, surface, "retire-older", successor)

    assert response.status_code == 200
    assert response.json() == {"retired": 1}
    assert not _verifies(predecessor_bearer)
    assert _verifies(successor) and _verifies(newer_bearer) and _verifies(other_bearer)


def test_cross_surface_and_recommendations_are_refused(home, clock, client):
    workspace, workspace_bearer = _issue(home)
    connections, connections_bearer = _issue(home, surface="connections")
    recommendations, recommendations_bearer = _issue(
        home, surface="recommendations", ttl_seconds=8 * 3600
    )
    clock["t"] += 60
    before = _state()

    for action in ("renew", "retire-older"):
        assert _post(client, "workspace", action, connections_bearer).status_code == 403
        assert _post(client, "workspace", action, recommendations_bearer).status_code == 403
        assert _post(client, "recommendations", action, recommendations_bearer).status_code != 200
    with pytest.raises(token_store.TokenStoreError):
        token_store.renew(recommendations.token_id)
    with pytest.raises(token_store.TokenStoreError):
        token_store.retire_older(recommendations.token_id)

    assert _state() == before
    assert _verifies(workspace_bearer) and _verifies(connections_bearer)


@pytest.mark.parametrize("ended", ["expired", "revoked", "unknown"])
def test_expired_revoked_or_unknown_caller_is_refused(home, clock, client, ended):
    if ended == "unknown":
        bearer = _unknown_bearer()
    else:
        _, bearer = _issue(home, ttl_seconds=token_store.MIN_TTL_SECONDS)
        if ended == "revoked":
            token_store.revoke(bearer.split(".", 1)[0])
    clock["t"] += token_store.MIN_TTL_SECONDS if ended == "expired" else 60
    token_id = bearer.split(".", 1)[0]
    before = _state()

    with pytest.raises(token_store.UnknownTokenError):
        token_store.renew(token_id)
    with pytest.raises(token_store.UnknownTokenError):
        token_store.retire_older(token_id)
    statuses = [_post(client, "workspace", action, bearer).status_code for action in ("renew", "retire-older")]

    assert statuses == [401, 401]
    assert _state() == before


@pytest.mark.parametrize("failure", ["store_write", "audit_write"])
def test_failed_renew_keeps_caller_valid(home, clock, client, monkeypatch, failure):
    caller, bearer = _issue(home)
    clock["t"] += 60
    before = _state()
    if failure == "store_write":
        monkeypatch.setattr(token_store, "_save_records_unlocked", _fail_store_write)
    else:
        def _fail_audit(*_args, **_kwargs):
            raise AuditWriteError("audit log unavailable")

        monkeypatch.setattr(token_store, "audit_log_batch", _fail_audit)
    out_dir = home / "renewal-output"
    out_dir.mkdir(mode=0o700)
    out_path = out_dir / "next.token"

    with pytest.raises((token_store.TokenStoreError, AuditWriteError)):
        token_store.renew(caller.token_id, out=out_path)
    response = _post(client, "workspace", "renew", bearer)

    assert response.status_code == 503
    assert response.json() == {"error": "unavailable"}
    assert not out_path.exists()
    assert _verifies(bearer)
    assert _state() == before


def test_renew_writes_no_secret_or_digest_to_audit_or_log(home, clock, client, caplog):
    caplog.set_level(logging.DEBUG)
    older, older_bearer = _issue(home)
    clock["t"] += 1
    caller, bearer = _issue(home)
    clock["t"] += 60
    orphan, orphan_secret = token_store.renew(caller.token_id)
    clock["t"] += 15 * 60
    successor = _post(client, "workspace", "renew", bearer).json()["token"]
    refused = _post(client, "workspace", "renew", bearer)
    retired = _post(client, "workspace", "retire-older", successor)

    assert refused.status_code == 409 and retired.json() == {"retired": 2}
    issued = _audit_events(home, AuditEvent.MACHINE_TOKEN_ISSUED)
    revoked = _audit_events(home, AuditEvent.MACHINE_TOKEN_REVOKED)
    assert [r["replaces_credential_id"] for r in issued[2:]] == [caller.token_id] * 2
    assert sorted(r["credential_id"] for r in revoked) == sorted(
        [orphan.token_id, older.token_id, caller.token_id]
    )
    bearers = (older_bearer, bearer, f"{orphan.token_id}.{orphan_secret}", successor)
    assert _leaks(bearers, _audit_text(home), caplog.text) == 0


def test_renew_errors_never_echo_token(home, clock, client, monkeypatch):
    fresh, fresh_bearer = _issue(home, ttl_seconds=90 * DAY)
    caller, bearer = _issue(home)
    connections, connections_bearer = _issue(home, surface="connections")
    recommendations, recommendations_bearer = _issue(
        home, surface="recommendations", ttl_seconds=8 * 3600
    )
    unknown_bearer = _unknown_bearer()
    clock["t"] += 60
    assert _post(client, "workspace", "renew", bearer).status_code == 200

    refusals = [
        (_post(client, "workspace", "renew", fresh_bearer), fresh_bearer, 409),
        (_post(client, "workspace", "renew", bearer), bearer, 409),
        (_post(client, "workspace", "renew", unknown_bearer), unknown_bearer, 401),
        (_post(client, "workspace", "renew", connections_bearer), connections_bearer, 403),
    ]
    monkeypatch.setattr(token_store, "_save_records_unlocked", _fail_store_write)
    clock["t"] += 15 * 60
    refusals.append((_post(client, "workspace", "renew", bearer), bearer, 503))

    assert [response.status_code for response, _, _ in refusals] == [s for _, _, s in refusals]
    for response, presented, status in refusals:
        assert _leaks([presented], response.text) == 0
        assert presented.split(".", 1)[0] not in response.text
        if status in (409, 503):
            assert list(response.json()) == ["error"]
    store_refusals = [
        (lambda: token_store.renew(fresh.token_id), fresh_bearer),
        (lambda: token_store.renew(recommendations.token_id), recommendations_bearer),
        (lambda: token_store.renew(unknown_bearer.split(".", 1)[0]), unknown_bearer),
        (lambda: token_store.retire_older(unknown_bearer.split(".", 1)[0]), unknown_bearer),
    ]
    for call, presented in store_refusals:
        with pytest.raises(token_store.TokenStoreError) as refused:
            call()
        assert _leaks([presented], str(refused.value)) == 0
        assert presented.split(".", 1)[0] not in str(refused.value)


def test_renew_answer_is_no_store(home, clock, client):
    caller, bearer = _issue(home)
    clock["t"] += 60

    renewed = _post(client, "workspace", "renew", bearer)
    refused = _post(client, "workspace", "renew", bearer)
    retired = _post(client, "workspace", "retire-older", renewed.json()["token"])

    assert [renewed.status_code, refused.status_code, retired.status_code] == [200, 409, 200]
    for response in (renewed, refused, retired):
        assert response.headers.get("cache-control") == "no-store"


def test_register_wires_renewal_routes_for_the_four_families(home):
    token_auth.clear_token_routes()
    try:
        raphael_workspace.register(MagicMock())
        for surface in FAMILIES:
            policy = token_store.policy_for_surface(surface)
            for action in ("renew", "retire-older"):
                route = token_auth.get_token_route_policy("POST", f"{BASE}/{surface}/{action}")
                assert route is not None, (surface, action)
                assert route.required_scope == policy.scope
                assert route.strict_audit and not route.optional
            family = token_auth.get_machine_token_family(policy.token_prefix + "x")
            assert family is not None and family.strict_audit
        for action in ("renew", "retire-older"):
            assert token_auth.get_token_route_policy("POST", f"{BASE}/recommendations/{action}") is None
    finally:
        token_auth.clear_token_routes()


def test_refusal_detail_never_holds_a_token_value_id_or_digest(home, clock, served, monkeypatch):
    """Every refusal these routes produce on the served application.

    The shared token middleware's refusals keep their two keys, error and
    detail; the renewal handler's own refusals carry error only; no detail holds
    a token value, a token id or a digest.
    """
    fresh, fresh_bearer = _issue(home, ttl_seconds=90 * DAY)
    caller, bearer = _issue(home)
    revoked, revoked_bearer = _issue(home)
    token_store.revoke(revoked.token_id)
    _, connections_bearer = _issue(home, surface="connections")
    _, recommendations_bearer = _issue(home, surface="recommendations", ttl_seconds=8 * 3600)
    unknown_bearer, foreign_bearer, revoked_in_flight_bearer = (_unknown_bearer() for _ in range(3))
    # Another provider granting the workspace scope reaches the handler's own checks.
    workspace = token_store.policy_for_surface("workspace")
    principals = {
        foreign_bearer: TokenPrincipal(
            principal="another-caller",
            provider="same-scope-test-provider",
            scopes=(workspace.scope,),
            credential_id=caller.token_id,
        ),
        revoked_in_flight_bearer: TokenPrincipal(
            principal=workspace.principal,
            provider="same-scope-test-provider",
            scopes=(workspace.scope,),
            credential_id=revoked.token_id,
        ),
    }

    class SameScopeProvider(WorkspaceReadTokenProvider):
        name = "same-scope-test-provider"

        def verify_token(self, *, token):
            return principals.get(token)

    def fail_audit(*_args, **_kwargs):
        raise AuditWriteError("audit log unavailable")

    def registry_unreadable(*_args, **_kwargs):
        raise token_store.TokenStoreError("token store unreadable")

    register_provider(SameScopeProvider())
    clock["t"] += 60
    successor = _post(served, "workspace", "renew", bearer).json()["token"]
    renew = f"{BASE}/workspace/renew"

    def post(presented, path=renew, **headers):
        return served.post(path, headers={"Authorization": f"Bearer {presented}", **headers})

    middleware = [
        post(bearer, **{"X-HTTP-Method-Override": "GET"}),
        served.post(renew),
        post(unknown_bearer),
        post(revoked_bearer),
        post(connections_bearer),
        post(recommendations_bearer),
        post(bearer, path=f"{BASE}/recommendations/renew"),
        served.get(renew, headers={"Authorization": f"Bearer {bearer}"}),
    ]
    handler = [post(fresh_bearer), post(bearer), post(foreign_bearer), post(revoked_in_flight_bearer)]
    # The retire-older route refuses through the same two layers.
    retire = f"{BASE}/workspace/retire-older"
    middleware += [served.post(retire), post(unknown_bearer, path=retire), post(connections_bearer, path=retire)]
    handler += [post(foreign_bearer, path=retire), post(revoked_in_flight_bearer, path=retire)]
    clock["t"] += 15 * 60
    with monkeypatch.context() as patch:
        patch.setattr(token_store, "_save_records_unlocked", _fail_store_write)
        handler.append(post(bearer))
        handler.append(post(successor, path=retire))
    with monkeypatch.context() as patch:
        patch.setattr(token_store, "verify", registry_unreadable)
        middleware.append(post(bearer))
    with monkeypatch.context() as patch:
        patch.setattr(token_auth, "audit_log", fail_audit)
        middleware.append(post(connections_bearer))

    assert [r.status_code for r in middleware] == [400, 401, 401, 401, 403, 403, 403, 403, 401, 401, 403, 503, 503]
    assert [r.status_code for r in handler] == [409, 409, 403, 401, 403, 401, 503, 503]
    bearers = (
        fresh_bearer,
        bearer,
        revoked_bearer,
        connections_bearer,
        recommendations_bearer,
        successor,
        unknown_bearer,
        foreign_bearer,
        revoked_in_flight_bearer,
    )
    material = {value for b in bearers for value in (*_secret_material(b), b.split(".", 1)[0])}
    material |= {value for r in token_store.load_records() for value in (r.token_id, r.digest)}

    def held(text):
        return sum(1 for value in material if value in text)

    for response in middleware:
        body = response.json()
        assert sorted(body) == ["detail", "error"]
        assert held(body["detail"]) == 0
    for response in handler:
        assert list(response.json()) == ["error"]
    assert sum(held(response.text) for response in middleware + handler) == 0


def test_real_served_application_serves_renewal_for_the_four_families(home, clock, served):
    """No router is mounted here: ``hermes_cli/web_server.py`` serves both routes."""
    from plugins.dashboard_auth.raphael_workspace import renewal

    served_paths = sorted(
        route.path
        for route in served.app.routes
        if getattr(getattr(route, "endpoint", None), "__module__", None) == renewal.__name__
    )
    assert served_paths == sorted(
        f"{BASE}/{surface}/{action}" for surface in FAMILIES for action in ("renew", "retire-older")
    )
    current = {surface: _issue(home, surface=surface)[1] for surface in FAMILIES}
    clock["t"] += 60

    for index, surface in enumerate(FAMILIES):
        other = current[FAMILIES[index - 1]]  # a valid key of another surface
        for action in ("renew", "retire-older"):
            assert _post(served, surface, action, other).status_code == 403, (surface, action)
        renewed = _post(served, surface, "renew", current[surface])
        assert renewed.status_code == 200, (surface, renewed.status_code)
        current[surface] = renewed.json()["token"]
        retired = _post(served, surface, "retire-older", current[surface])
        assert (retired.status_code, retired.json()) == (200, {"retired": 1}), surface
        existing = served.get(
            EXISTING_ROUTES[surface], headers={"Authorization": f"Bearer {current[surface]}"}
        )
        assert existing.status_code == 200, (surface, existing.status_code)


def test_this_file_commits_no_literal_key_or_digest():
    from plugins.dashboard_auth.raphael_workspace import renewal

    source = Path(__file__).read_text(encoding="utf-8")
    surfaces = (*renewal.RENEWAL_SURFACES, token_store.RECOMMENDATIONS_SURFACE)
    for surface in surfaces:
        prefix = token_store.policy_for_surface(surface).token_prefix
        assert re.search(re.escape(prefix) + r"[A-Za-z0-9_-]{12,}", source) is None, surface
    assert re.search(r"[0-9a-f]{64}", source) is None
