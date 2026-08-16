"""ITEM31BG Stage C: GET /api/plugins/kanban/recommendations — read-only, scoped,
machine-token-gated view of native recommendation cards. Covers the bearer-token auth seam,
mandatory exact board/project/profile scope, opaque scope-bound cursor pagination, read-only
DB access, centralized redaction, and a closed single-GET OpenAPI surface."""
from __future__ import annotations
import base64
import sqlite3
import sys
import time
from pathlib import Path
from typing import Optional
import pytest
from hermes_cli import kanban_db as kb
from hermes_cli.dashboard_auth import DashboardAuthProvider, TokenPrincipal, clear_providers, register_provider
from hermes_cli.dashboard_auth.base import ProviderError
_ROUTE = "/api/plugins/kanban/recommendations"
_PROVENANCE = dict(provenance_authority="static-analyzer", provenance_ref="finding-123", provenance_observed_at=1_700_000_000)
_ITEM_KEYS = {"id", "kind", "subject_id", "label", "rationale", "project_id", "target_profile", "status",
    "review_policy", "provenance_authority", "provenance_ref", "provenance_observed_at", "created_at", "updated_at"}
class _StubProvider(DashboardAuthProvider):
    name = "rec-test-provider"
    display_name = "Recommendations Test Provider"
    supports_token = True
    supports_session = False
    def __init__(self, *, secret: str = "good-secret", scopes=()) -> None:
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
        recommendation_label="Load the translation skill", recommendation_rationale="seen repeatedly", **_PROVENANCE)
    kwargs.update(overrides)
    with kb.connect_closing(board=board) as conn:
        return kb.create_recommendation(conn, **kwargs)
def _auth_headers(token: str = "good-secret") -> dict:
    return {"Authorization": f"Bearer {token}"}
def _params(**over) -> dict:
    base = {"board": "default", "project_id": "proj-1", "target_profile": "worker", "limit": 50}
    base.update(over)
    return base
@pytest.fixture
def _authed(client):
    register_provider(_StubProvider(secret="good-secret"))
    return _auth_headers("good-secret")
# --- Auth gate ---
@pytest.mark.parametrize("provider,headers,expected_status", [
    (None, {}, 401),
    (_StubProvider(secret="good-secret"), _auth_headers("wrong"), 401),
    (_UnreachableProvider(), _auth_headers("anything"), 503),
], ids=["no-header-no-provider", "wrong-bearer", "provider-unreachable"])
def test_bearer_gate_rejections(client, kanban_home, provider, headers, expected_status) -> None:
    if provider is not None:
        register_provider(provider)
    assert client.get(_ROUTE, params=_params(), headers=headers).status_code == expected_status
def test_valid_bearer_succeeds_and_cookie_session_alone_is_insufficient(client, kanban_home) -> None:
    # A valid token principal suffices regardless of scopes; dashboard session creds alone (no token provider) must still 401.
    register_provider(_StubProvider(secret="good-secret", scopes=("unrelated-scope",)))
    _make_recommendation()
    r = client.get(_ROUTE, params=_params(), headers=_auth_headers("good-secret"))
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["schema_version"] == 1 and len(body["items"]) == 1
    clear_providers()
    from hermes_cli import web_server as ws
    assert client.get(_ROUTE, params=_params(), headers={"Authorization": f"Bearer {ws._SESSION_TOKEN}"}).status_code == 401
    assert client.get(_ROUTE, params=_params(), headers={ws._SESSION_HEADER_NAME: ws._SESSION_TOKEN}).status_code == 401
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
# --- OpenAPI: exactly one closed GET ---
def test_openapi_exposes_only_one_closed_get(client, kanban_home) -> None:
    spec = client.get("/openapi.json").json()
    path_item = spec["paths"][_ROUTE]
    assert set(path_item.keys()) == {"get"}
    schema_ref = path_item["get"]["responses"]["200"]["content"]["application/json"]["schema"]["$ref"]
    list_schema = spec["components"]["schemas"][schema_ref.rsplit("/", 1)[-1]]
    assert list_schema.get("additionalProperties") is False
    item_ref = list_schema["properties"]["items"]["items"]["$ref"]
    item_schema = spec["components"]["schemas"][item_ref.rsplit("/", 1)[-1]]
    assert item_schema.get("additionalProperties") is False
    assert set(item_schema["properties"].keys()) == _ITEM_KEYS
    assert item_schema["properties"]["kind"]["enum"] == [
        "skill", "permission", "connection", "pipeline", "provider_model_policy",
    ]
