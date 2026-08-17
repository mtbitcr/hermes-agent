"""ITEM31BG Stage C / ITEM31BI: GET /api/plugins/kanban/recommendations — read-only,
scoped, machine-token-gated view of native recommendation cards. Covers the scoped
bearer-token auth seam (a credential must carry this route's exact capability scope),
mandatory exact board/project/profile scope, opaque scope-bound cursor pagination,
read-only DB access, centralized redaction, and a closed single-GET OpenAPI surface."""
from __future__ import annotations
import base64
import secrets
import sqlite3
import sys
import time
from pathlib import Path
from typing import Optional
from unittest.mock import MagicMock
import pytest
import plugins.dashboard_auth.recommendations as rec_plugin
from hermes_cli import kanban_db as kb
from hermes_cli.dashboard_auth import DashboardAuthProvider, TokenPrincipal, clear_providers, register_provider
from hermes_cli.dashboard_auth import token_auth
from hermes_cli.dashboard_auth.base import ProviderError
from plugins.dashboard_auth.drain import DrainSecretProvider
from plugins.dashboard_auth.recommendations import RecommendationsSecretProvider
_ROUTE = "/api/plugins/kanban/recommendations"
_REC_SCOPE = "kanban:recommendations:read"
# The other bundled machine credential, used to prove one credential buys
# exactly one endpoint. Its route is registered per-test (in production the
# drain plugin registers it only when a strong drain secret is provisioned).
_DRAIN_ROUTE = "/api/gateway/drain"
_DRAIN_SCOPE = "drain"
_PROVENANCE = dict(provenance_authority="static-analyzer", provenance_ref="finding-123", provenance_observed_at=1_700_000_000)
_EVIDENCE = dict(schema_version=1, need="Need", expected_benefit="Benefit",
    requested_scope={flag: False for flag in kb.RECOMMENDATION_SCOPE_FLAGS},
    risks="Low", cost="None", rollback="Remove it")
_ITEM_KEYS = {"id", "kind", "subject_id", "label", "rationale", "project_id", "target_profile", "status",
    "review_policy", "provenance_authority", "provenance_ref", "provenance_observed_at", "created_at", "updated_at"}
_ITEM_V2_KEYS = _ITEM_KEYS | {
    "evidence", "decision", "effective_state", "lifecycle_version",
    "lifecycle_events",
}
class _StubProvider(DashboardAuthProvider):
    name = "rec-test-provider"
    display_name = "Recommendations Test Provider"
    supports_token = True
    supports_session = False
    def __init__(self, *, secret: str = "good-secret", scopes=(_REC_SCOPE,)) -> None:
        self._secret = secret
        self._scopes = tuple(scopes)
    def verify_token(self, *, token: str) -> Optional[TokenPrincipal]:
        ok = token == self._secret
        return TokenPrincipal(principal="svc-caller", provider=self.name, scopes=self._scopes) if ok else None
    def start_login(self, *, redirect_uri): raise NotImplementedError
    def complete_login(self, *, code, state, code_verifier, redirect_uri): raise NotImplementedError
    def verify_session(self, *, access_token): return None
    def refresh_session(self, *, refresh_token): raise NotImplementedError
    def revoke_session(self, *, refresh_token): return None
class _UnreachableProvider(_StubProvider):
    name = "rec-test-unreachable"
    def verify_token(self, *, token: str): raise ProviderError("backing store down")
@pytest.fixture(autouse=True)
def _clean_auth_state():
    clear_providers()
    yield
    clear_providers()
@pytest.fixture
def kanban_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
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
def _plugin_mod():
    return sys.modules["hermes_dashboard_plugin_kanban"]
def _make_recommendation(board: Optional[str] = None, **overrides) -> str:
    kwargs = dict(project_id="proj-1", target_profile="worker", recommendation_kind="skill", recommendation_subject_id="translation",
        recommendation_label="Load the translation skill", recommendation_rationale="seen repeatedly",
        recommendation_evidence=_EVIDENCE, **_PROVENANCE)
    kwargs.update(overrides)
    with kb.connect_closing(board=board) as conn:
        return kb.create_recommendation(conn, **kwargs)
def _auth_headers(token: str = "good-secret") -> dict:
    return {"Authorization": f"Bearer {token}"}
def _params(**over) -> dict:
    base = {"board": "default", "project_id": "proj-1", "target_profile": "worker", "limit": 50}
    base.update(over)
    return base
@pytest.fixture(autouse=True)
def _isolated_token_routes(client):
    """Snapshot/restore the token-route registry around each test.

    The baseline is the Kanban plugin's own GET/_ROUTE/_REC_SCOPE policy.
    ``client`` guarantees the app/plugin modules are imported, but in a
    larger test run a module can already be cached from a prior test file,
    so importing it again doesn't re-run its module-level route
    registration. Re-installing the canonical policy here (a no-op if it's
    already registered — ``register_token_route`` is idempotent for an
    identical policy) guarantees the baseline is always correct before it's
    captured. Tests that add a route — e.g. the drain endpoint — get it
    wiped afterwards, and the baseline policy is put back exactly, so route
    state never leaks between tests regardless of execution order.
    """
    token_auth.register_token_route(_ROUTE, method="GET", required_scope=_REC_SCOPE)
    baseline = token_auth.get_token_route_policy("GET", _ROUTE)
    yield
    token_auth.clear_token_routes()
    if baseline is not None:
        token_auth.register_token_route(
            baseline.path, method=baseline.method, required_scope=baseline.required_scope
        )
@pytest.fixture
def _authed(client):
    register_provider(_StubProvider(secret="good-secret"))
    return _auth_headers("good-secret")
@pytest.fixture
def _no_downstream(client, monkeypatch):
    """Make any attempt to read the recommendations DB an error.

    Used by the refusal tests: a rejected request must be turned away by the
    auth seam, never merely filtered by the handler.
    """
    def _boom(*a, **k):
        raise AssertionError("downstream handler reached")
    monkeypatch.setattr(_plugin_mod(), "_recommendations_ro_conn", _boom)
@pytest.fixture
def _drain_calls(monkeypatch):
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
def _register_drain_route() -> None:
    token_auth.register_token_route(_DRAIN_ROUTE, method="POST", required_scope=_DRAIN_SCOPE)
# --- Auth gate: the route is token-only, and the token must carry this route's scope ---
def test_route_is_registered_token_only_under_its_own_scope(client, kanban_home) -> None:
    # Registered by the Kanban plugin API at import — unconditionally, with no
    # dependency on a credential provider existing.
    policy = token_auth.get_token_route_policy("GET", _ROUTE)
    assert policy is not None
    assert (policy.method, policy.path, policy.required_scope) == ("GET", _ROUTE, _REC_SCOPE)
@pytest.mark.parametrize("provider,headers,expected_status", [
    (None, {}, 401),
    (None, _auth_headers("anything"), 401),
    (_StubProvider(secret="good-secret"), {}, 401),
    (_StubProvider(secret="good-secret"), _auth_headers("wrong"), 401),
    (_StubProvider(secret="good-secret"), {"Authorization": "Basic Z29vZC1zZWNyZXQ="}, 401),
    (_StubProvider(secret="good-secret"), {"Authorization": "good-secret"}, 401),
    (_StubProvider(secret="good-secret"), {"Authorization": "Bearer"}, 401),
    (_StubProvider(secret="good-secret"), {"Authorization": "Bearer "}, 401),
    (_UnreachableProvider(), _auth_headers("anything"), 503),
], ids=["no-header-no-provider", "bearer-no-provider", "no-header", "wrong-bearer",
        "basic-scheme", "scheme-less", "bearer-no-value", "bearer-empty-value",
        "provider-unreachable"])
def test_bearer_gate_rejections(client, kanban_home, _no_downstream, provider, headers, expected_status) -> None:
    # Missing / malformed / unrecognised credentials never reach the handler.
    if provider is not None:
        register_provider(provider)
    assert client.get(_ROUTE, params=_params(), headers=headers).status_code == expected_status
def test_correctly_scoped_bearer_succeeds(client, kanban_home) -> None:
    register_provider(_StubProvider(secret="good-secret", scopes=(_REC_SCOPE,)))
    _make_recommendation()
    r = client.get(_ROUTE, params=_params(), headers=_auth_headers("good-secret"))
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["schema_version"] == 1 and len(body["items"]) == 1
def test_extra_scopes_alongside_the_required_one_still_pass(client, kanban_home) -> None:
    register_provider(_StubProvider(secret="good-secret", scopes=("unrelated", _REC_SCOPE, _DRAIN_SCOPE)))
    _make_recommendation()
    assert client.get(_ROUTE, params=_params(), headers=_auth_headers("good-secret")).status_code == 200
@pytest.mark.parametrize("scopes", [
    (), ("unrelated-scope",), (_DRAIN_SCOPE,), ("kanban:recommendations",),
    ("kanban:recommendations:read:extra",), ("KANBAN:RECOMMENDATIONS:READ",),
], ids=["unscoped", "unrelated", "drain", "prefix", "superstring", "wrong-case"])
def test_recognized_principal_without_the_required_scope_gets_generic_403(
    client, kanban_home, _no_downstream, scopes
) -> None:
    register_provider(_StubProvider(secret="good-secret", scopes=scopes))
    _make_recommendation()
    r = client.get(_ROUTE, params=_params(), headers=_auth_headers("good-secret"))
    assert r.status_code == 403
    # Generic refusal: it discloses neither the scope held nor the scope wanted,
    # and carries no recommendation data (the handler was never reached — see
    # the _no_downstream fixture).
    body = r.text
    assert _REC_SCOPE not in body
    for held in scopes:
        assert held not in body
    assert "items" not in r.json()
def test_dashboard_session_credentials_do_not_substitute(client, kanban_home, _no_downstream) -> None:
    # The machine credential is the only key to this route: the dashboard's own
    # session token / header / cookies buy nothing, even with the recommendations
    # credential provider registered and able to accept its own secret.
    register_provider(_StubProvider(secret="good-secret", scopes=(_REC_SCOPE,)))
    from hermes_cli import web_server as ws
    assert client.get(_ROUTE, params=_params(), headers={"Authorization": f"Bearer {ws._SESSION_TOKEN}"}).status_code == 401
    assert client.get(_ROUTE, params=_params(), headers={ws._SESSION_HEADER_NAME: ws._SESSION_TOKEN}).status_code == 401
    assert client.get(_ROUTE, params=_params(token=ws._SESSION_TOKEN)).status_code == 401
    client.cookies.set("hermes_session_at", "session-access-token")
    client.cookies.set("hermes_session_provider", "rec-test-provider")
    try:
        assert client.get(_ROUTE, params=_params()).status_code == 401
    finally:
        client.cookies.clear()
def test_endpoint_stays_token_only_when_the_credential_plugin_is_a_no_op(
    client, kanban_home, _no_downstream, monkeypatch
) -> None:
    # No HERMES_DASHBOARD_RECOMMENDATIONS_SECRET → the plugin registers nothing,
    # but the route stays token-gated and answers 401 (fail-closed, never open).
    monkeypatch.delenv(rec_plugin.SECRET_ENV_VAR, raising=False)
    ctx = MagicMock()
    rec_plugin.register(ctx)
    ctx.register_dashboard_auth_provider.assert_not_called()
    assert token_auth.get_token_route_policy("GET", _ROUTE) is not None
    assert client.get(_ROUTE, params=_params()).status_code == 401
    assert client.get(_ROUTE, params=_params(), headers=_auth_headers("any-secret")).status_code == 401
# --- Method mismatch and cross-credential confinement ---
@pytest.mark.parametrize("method", ["post", "put", "patch", "delete"])
def test_method_mismatch_does_not_widen_token_authority(client, kanban_home, _no_downstream, method) -> None:
    # The policy covers GET only. A correctly scoped bearer on another verb is
    # not token-authenticated, so the ordinary session gate turns it away.
    register_provider(_StubProvider(secret="good-secret", scopes=(_REC_SCOPE,)))
    r = getattr(client, method)(_ROUTE, params=_params(), headers=_auth_headers("good-secret"))
    assert r.status_code == 401
def test_drain_credential_cannot_read_recommendations(client, kanban_home, _no_downstream) -> None:
    drain_secret = secrets.token_urlsafe(32)
    register_provider(DrainSecretProvider(secret=drain_secret, scope=_DRAIN_SCOPE))
    _make_recommendation()
    r = client.get(_ROUTE, params=_params(), headers=_auth_headers(drain_secret))
    assert r.status_code == 403
    assert _REC_SCOPE not in r.text and _DRAIN_SCOPE not in r.text
def test_recommendations_credential_cannot_drive_drain(client, kanban_home, _drain_calls) -> None:
    rec_secret = secrets.token_urlsafe(32)
    drain_secret = secrets.token_urlsafe(32)
    register_provider(RecommendationsSecretProvider(secret=rec_secret))
    register_provider(DrainSecretProvider(secret=drain_secret, scope=_DRAIN_SCOPE))
    _register_drain_route()
    r = client.post(_DRAIN_ROUTE, json={"action": "cancel"}, headers=_auth_headers(rec_secret))
    assert r.status_code == 403
    assert _DRAIN_SCOPE not in r.text and _REC_SCOPE not in r.text
    assert _drain_calls == {"write": 0, "clear": 0}  # downstream never reached
    # Control: the drain credential itself is accepted on its own route, so the
    # 403 above is about authority, not a broken registration.
    ok = client.post(_DRAIN_ROUTE, json={"action": "cancel"}, headers=_auth_headers(drain_secret))
    assert ok.status_code == 200, ok.text
    assert _drain_calls == {"write": 0, "clear": 1}
def test_recommendations_credential_is_accepted_on_its_own_route(client, kanban_home) -> None:
    rec_secret = secrets.token_urlsafe(32)
    register_provider(RecommendationsSecretProvider(secret=rec_secret))
    register_provider(DrainSecretProvider(secret=secrets.token_urlsafe(32), scope=_DRAIN_SCOPE))
    _register_drain_route()
    rec_id = _make_recommendation()
    r = client.get(_ROUTE, params=_params(), headers=_auth_headers(rec_secret))
    assert r.status_code == 200, r.text
    assert [i["id"] for i in r.json()["items"]] == [rec_id]
# --- Mandatory exact scope + isolation ---
@pytest.mark.parametrize("overrides,expected_status", [
    ({"board": ""}, 400), ({"board": "all"}, 400), ({"board": "*"}, 400),
    ({"project_id": ""}, 400), ({"project_id": "all"}, 400), ({"project_id": "*"}, 400),
    ({"target_profile": ""}, 400), ({"target_profile": "all"}, 400), ({"target_profile": "*"}, 400),
    ({"board": "does-not-exist"}, 404), ({"limit": 0}, 422), ({"limit": 101}, 422),
])
def test_invalid_query_params_rejected(client, kanban_home, _authed, overrides, expected_status) -> None:
    r = client.get(_ROUTE, params=_params(**overrides), headers=_authed)
    assert r.status_code == expected_status, r.text
@pytest.mark.parametrize("field,val_a,val_b", [
    ("board", "default", "second-board"),
    ("project_id", "proj-a", "proj-b"),
    ("target_profile", "worker", "reviewer"),
])
def test_scope_dimension_isolation(client, kanban_home, _authed, field, val_a, val_b) -> None:
    if field == "board":
        kb.create_board("second-board")
        id_a = _make_recommendation(board=val_a if val_a != "default" else None)
        id_b = _make_recommendation(board=val_b)
    else:
        id_a = _make_recommendation(**{field: val_a})
        id_b = _make_recommendation(**{field: val_b})
    ids_a = {i["id"] for i in client.get(_ROUTE, params=_params(**{field: val_a}), headers=_authed).json()["items"]}
    assert id_a in ids_a and id_b not in ids_a
    ids_b = {i["id"] for i in client.get(_ROUTE, params=_params(**{field: val_b}), headers=_authed).json()["items"]}
    assert id_b in ids_b and id_a not in ids_b
# --- Opaque, scope-bound cursor pagination ---
def test_cursor_pagination_is_deterministic_and_opaque_and_scope_bound(client, kanban_home, _authed) -> None:
    ids = [_make_recommendation(recommendation_subject_id=f"s{i}") for i in range(5)]
    now = int(time.time())
    with kb.connect_closing() as conn:
        conn.execute("UPDATE tasks SET created_at = ? WHERE task_kind = 'recommendation'", (now,))
        conn.commit()
    seen: list[str] = []
    cursor = None
    for _ in range(10):
        params = _params(limit=2, **({"cursor": cursor} if cursor else {}))
        r = client.get(_ROUTE, params=params, headers=_authed)
        assert r.status_code == 200, r.text
        body = r.json()
        seen.extend(i["id"] for i in body["items"])
        cursor = body["next_cursor"]
        if cursor is None:
            break
    # Equal timestamps break ties by id DESC — deterministic, not insertion order.
    assert seen == sorted(ids, reverse=True)
    assert len(seen) == len(set(seen)) == 5
    cursor = client.get(_ROUTE, params=_params(limit=1), headers=_authed).json()["next_cursor"]
    assert cursor is not None
    decoded = base64.urlsafe_b64decode(cursor.encode("ascii"))
    assert b"created_at" not in decoded  # not a naive JSON dict of column names
    assert client.get(_ROUTE, params=_params(project_id="other-project", cursor=cursor), headers=_authed).status_code == 400
    assert client.get(_ROUTE, params=_params(target_profile="reviewer", cursor=cursor), headers=_authed).status_code == 400
    assert client.get(_ROUTE, params=_params(cursor="not-a-real-cursor"), headers=_authed).status_code == 400
    assert client.get(
        _ROUTE, params=_params(schema_version=2, cursor=cursor), headers=_authed
    ).status_code == 400
    cursor_v2 = client.get(
        _ROUTE, params=_params(schema_version=2, limit=1), headers=_authed
    ).json()["next_cursor"]
    assert cursor_v2 is not None
    assert client.get(
        _ROUTE, params=_params(cursor=cursor_v2), headers=_authed
    ).status_code == 400
# --- Read-only DB access: mode=ro / query_only, zero side effects, no eager DB creation ---
def test_read_only_access_has_zero_side_effects(client, kanban_home, _authed, monkeypatch) -> None:
    rec_id = _make_recommendation()
    kb.get_task(kb.connect(board="default"), kb.create_task(kb.connect(board="default"), title="ordinary work"))
    conn = _plugin_mod()._recommendations_ro_conn("default")
    try:
        assert conn.execute("PRAGMA query_only").fetchone()[0] == 1
        with pytest.raises(sqlite3.OperationalError):
            conn.execute("UPDATE tasks SET priority = 99")
    finally:
        conn.close()
    calls = {"init_db": 0, "connect": 0, "recompute_ready": 0}
    for name in ("init_db", "recompute_ready"):
        monkeypatch.setattr(kb, name, lambda *a, _n=name, **k: calls.__setitem__(_n, calls[_n] + 1))
    monkeypatch.setattr(kb, "connect", lambda *a, **k: (_ for _ in ()).throw(AssertionError("connect() called")))
    db_path = kb.kanban_db_path(board="default")
    before_mtime = db_path.stat().st_mtime_ns
    def _snapshot(raw):
        return (raw.execute("SELECT COUNT(*) FROM tasks").fetchone()[0], raw.execute("SELECT COUNT(*) FROM task_events").fetchone()[0],
                raw.execute("SELECT status FROM tasks WHERE id = ?", (rec_id,)).fetchone()[0])
    with sqlite3.connect(str(db_path)) as raw:
        before = _snapshot(raw)
    assert client.get(_ROUTE, params=_params(), headers=_authed).status_code == 200
    assert calls == {"init_db": 0, "connect": 0, "recompute_ready": 0}
    assert db_path.stat().st_mtime_ns == before_mtime
    with sqlite3.connect(str(db_path)) as raw:
        assert _snapshot(raw) == before == (before[0], before[1], "review")
    kb.create_board("empty-board")  # missing db file: no eager creation, empty result
    empty_db_path = kb.kanban_db_path(board="empty-board")
    if empty_db_path.exists():
        empty_db_path.unlink()
    r = client.get(_ROUTE, params=_params(board="empty-board"), headers=_authed)
    assert r.status_code == 200
    assert r.json() == {"schema_version": 1, "items": [], "next_cursor": None}
    r_v2 = client.get(
        _ROUTE, params=_params(board="empty-board", schema_version=2), headers=_authed
    )
    assert r_v2.status_code == 200
    assert r_v2.json() == {"schema_version": 2, "items": [], "next_cursor": None}
    assert not empty_db_path.exists()
# --- Recursive redaction + disclosure boundary ---
def test_recursive_redaction_and_disclosure_boundary(client, kanban_home, _authed) -> None:
    secret_token = "sk-ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    rec_id = _make_recommendation(
        recommendation_label=f"rotate {secret_token}",
        recommendation_rationale=f"observed key {secret_token} in logs",
        provenance_ref=f"ref={secret_token}",
    )
    r = client.get(_ROUTE, params=_params(), headers=_authed)
    assert r.status_code == 200, r.text
    item = r.json()["items"][0]
    assert secret_token not in item["label"] and secret_token not in item["rationale"] and secret_token not in item["provenance_ref"]
    with kb.connect_closing() as conn:
        raw = conn.execute("SELECT recommendation_label, recommendation_rationale, provenance_ref FROM tasks WHERE id = ?", (rec_id,)).fetchone()
        raw_event = conn.execute("SELECT payload FROM task_events WHERE task_id = ?", (rec_id,)).fetchone()[0]
    assert secret_token not in "".join(str(v or "") for v in raw) and secret_token not in raw_event
    forbidden = {"title", "body", "comments", "runs", "workspace_path", "workspace_kind", "branch_name", "claim_lock",
        "result", "commands", "prompts", "stored_path", "secrets", "token", "assignee", "worker_pid"}
    assert forbidden.isdisjoint(item.keys())
    assert set(item.keys()) == _ITEM_KEYS
# --- v2 lifecycle projection and OpenAPI: still exactly one closed GET ---
def test_v2_projects_allowlisted_lifecycle_without_raw_event_payload(
    client, kanban_home, _authed
) -> None:
    rec_id = _make_recommendation(recommendation_kind="profile_setting")
    with kb.connect_closing() as conn, kb.write_txn(conn):
        conn.execute(
            "UPDATE tasks SET recommendation_decision = 'accepted', "
            "recommendation_effective_state = 'staged', "
            "recommendation_lifecycle_version = 2 WHERE id = ?",
            (rec_id,),
        )
        kb._append_event(
            conn,
            rec_id,
            "recommendation_decided",
            {
                "lifecycle_version": 1,
                "decision": "accepted",
                "effective_state": "none",
                "authority": "owner_approved",
                "gate_ref": "owner-gate:item32e",
                "reason": "private operator reasoning",
                "actor": "raphael-owner",
                "governance_task_id": "t_11111111",
                "governance_run_id": 41,
            },
        )
        kb._append_event(
            conn,
            rec_id,
            "recommendation_transitioned",
            {
                "lifecycle_version": 2,
                "decision": "accepted",
                "effective_state": "staged",
                "reason": "must-not-leak /private/operator/path manifest.json",
                "actor": "raphael-owner",
                "governance_task_id": "t_22222222",
                "governance_run_id": 42,
                "native_surface": "hermes.profile.agent.max_turns",
                "config_identity": "a" * 64,
                "rollback_identity": "b" * 64,
            },
        )

    v1 = client.get(_ROUTE, params=_params(), headers=_authed)
    assert v1.status_code == 200
    assert set(v1.json()["items"][0]) == _ITEM_KEYS

    response = client.get(
        _ROUTE, params=_params(schema_version=2), headers=_authed
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["schema_version"] == 2 and body["next_cursor"] is None
    item = body["items"][0]
    assert item["id"] == rec_id and set(item) == _ITEM_V2_KEYS
    assert item["kind"] == "profile_setting"
    assert item["evidence"] == _EVIDENCE
    assert (
        item["decision"], item["effective_state"], item["lifecycle_version"]
    ) == ("accepted", "staged", 2)
    assert [event["kind"] for event in item["lifecycle_events"]] == [
        "recommendation_created",
        "recommendation_decided",
        "recommendation_transitioned",
    ]
    assert [event["lifecycle_version"] for event in item["lifecycle_events"]] == [
        0, 1, 2,
    ]
    rendered = response.text
    for forbidden in (
        "private operator reasoning", "/private/operator/path", "manifest.json",
        '"reason"', '"payload"',
    ):
        assert forbidden not in rendered


def test_v2_fails_closed_on_malformed_lifecycle_event_but_v1_stays_compatible(
    client, kanban_home, _authed
) -> None:
    rec_id = _make_recommendation()
    with kb.connect_closing() as conn:
        conn.execute(
            "INSERT INTO task_events (task_id, kind, payload, created_at) "
            "VALUES (?, 'recommendation_decided', '{\"unexpected\":true}', ?)",
            (rec_id, int(time.time())),
        )
        conn.commit()
    assert client.get(_ROUTE, params=_params(), headers=_authed).status_code == 200
    response = client.get(
        _ROUTE, params=_params(schema_version=2), headers=_authed
    )
    assert response.status_code == 503
    assert response.json() == {"detail": "recommendation lifecycle data is invalid"}


def test_v2_fails_closed_when_snapshot_and_events_disagree(
    client, kanban_home, _authed
) -> None:
    rec_id = _make_recommendation()
    with kb.connect_closing() as conn:
        conn.execute(
            "UPDATE tasks SET recommendation_decision = ?, "
            "recommendation_lifecycle_version = ? WHERE id = ?",
            ("accepted", 1, rec_id),
        )
        conn.commit()
    assert client.get(_ROUTE, params=_params(), headers=_authed).status_code == 200
    response = client.get(
        _ROUTE, params=_params(schema_version=2), headers=_authed
    )
    assert response.status_code == 503
    assert response.json() == {
        "detail": "recommendation lifecycle data is invalid"
    }


def test_openapi_exposes_only_one_closed_get(client, kanban_home) -> None:
    spec = client.get("/openapi.json").json()
    path_item = spec["paths"][_ROUTE]
    assert set(path_item.keys()) == {"get"}
    response_schema = path_item["get"]["responses"]["200"]["content"][
        "application/json"
    ]["schema"]
    variants = response_schema.get("anyOf", [response_schema])
    refs = {variant["$ref"].rsplit("/", 1)[-1] for variant in variants}
    assert refs == {
        "KanbanRecommendationListResponse",
        "KanbanRecommendationListV2Response",
    }
    component_schemas = spec["components"]["schemas"]
    item_schemas = {}
    for list_name in refs:
        list_schema = component_schemas[list_name]
        assert list_schema.get("additionalProperties") is False
        item_ref = list_schema["properties"]["items"]["items"]["$ref"]
        item_name = item_ref.rsplit("/", 1)[-1]
        item_schema = component_schemas[item_name]
        assert item_schema.get("additionalProperties") is False
        item_schemas[item_name] = item_schema
    assert set(item_schemas["KanbanRecommendationItem"]["properties"]) == _ITEM_KEYS
    assert set(item_schemas["KanbanRecommendationItemV2"]["properties"]) == _ITEM_V2_KEYS
    assert item_schemas["KanbanRecommendationItem"]["properties"]["kind"]["enum"] == [
        "skill", "permission", "connection", "pipeline", "provider_model_policy",
        "profile_setting",
    ]
