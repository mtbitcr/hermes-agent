"""ITEM31BG Stage B: native recommendation cards (task_kind='recommendation') must be invisible to and
unreachable from every ordinary Kanban surface — DB layer (kanban_db.py) and dashboard plugin
(plugin_api.py) — while ordinary work-task behavior is unchanged."""
from __future__ import annotations
import importlib.util
import sys
from pathlib import Path
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from hermes_cli import kanban_db as kb
_PROVENANCE = dict(provenance_authority="static-analyzer", provenance_ref="finding-123", provenance_observed_at=1_700_000_000)
@pytest.fixture
def kanban_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home
def _make_recommendation(conn, **overrides) -> str:
    kwargs = dict(
        project_id="proj-1", target_profile="worker", recommendation_kind="skill",
        recommendation_subject_id="translation", recommendation_label="Load the translation skill",
        recommendation_rationale="seen repeatedly", **_PROVENANCE,
    )
    kwargs.update(overrides)
    return kb.create_recommendation(conn, **kwargs)
def _force_row(conn, task_id: str, **cols) -> None:
    # Stamp raw columns directly (bypassing every mutator) to exercise the task_kind predicate itself.
    sets = ", ".join(f"{c} = ?" for c in cols)
    conn.execute(f"UPDATE tasks SET {sets} WHERE id = ?", (*cols.values(), task_id))
    conn.commit()
def _row(conn, task_id: str):
    return conn.execute("SELECT * FROM tasks WHERE id = ?", (task_id,)).fetchone()
def _event_kinds(conn, task_id: str) -> list[str]:
    return [r["kind"] for r in conn.execute("SELECT kind FROM task_events WHERE task_id = ? ORDER BY id", (task_id,))]
# --- DB layer: scan / comment / attachment / run / event / link isolation ---
def test_scan_dependent_and_related_surfaces_exclude_recommendation(kanban_home: Path) -> None:
    with kb.connect_closing() as conn:
        work_id = kb.create_task(conn, title="ordinary work")
        rec_id = _make_recommendation(conn)
        ids = {t.id for t in kb.list_tasks(conn)}
        assert work_id in ids and rec_id not in ids and kb.get_task(conn, rec_id) is None
        assert kb.board_stats(conn)["by_status"].get("review", 0) == 0  # recommendation is status='review'
        assert kb.list_comments(conn, rec_id) == []
        with pytest.raises(ValueError):
            kb.add_comment(conn, rec_id, author="op", body="hi")
        assert kb.list_attachments(conn, rec_id) == []
        with pytest.raises(ValueError):
            kb.add_attachment(conn, rec_id, filename="x.txt", content_type="text/plain", size=1, stored_path="x.txt")
        assert kb.list_runs(conn, rec_id) == []
        # Ordinary list_events positively requires task_kind='work'; the one typed audit
        # event is inspected via direct test-only SQL (_event_kinds), never this accessor.
        assert kb.list_events(conn, rec_id) == []
        assert _event_kinds(conn, rec_id) == ["recommendation_created"]
        for a, b in [(rec_id, work_id), (work_id, rec_id)]:
            with pytest.raises(ValueError):
                kb.link_tasks(conn, a, b)
        assert kb.parent_ids(conn, work_id) == [] and kb.child_ids(conn, work_id) == []
# --- DB layer: claim / dispatch / promote / mutator isolation (adversarial state) ---
@pytest.mark.parametrize("force_kwargs,call_fn,expected_status", [
    (dict(status="ready", claim_lock=None, assignee="alice"), kb.claim_task, "ready"),
    (dict(), kb.claim_review_task, "review"),
], ids=["claim_task-forced-ready", "claim_review_task-fresh"])
def test_claim_functions_cannot_claim_recommendation(kanban_home: Path, force_kwargs: dict, call_fn, expected_status: str) -> None:
    with kb.connect_closing() as conn:
        rec_id = _make_recommendation(conn)
        if force_kwargs:
            _force_row(conn, rec_id, **force_kwargs)
        assert call_fn(conn, rec_id) is None
        row = _row(conn, rec_id)
        assert row["status"] == expected_status and row["claim_lock"] is None
        assert _event_kinds(conn, rec_id) == ["recommendation_created"]
def test_promote_recompute_and_id_mutators_do_not_touch_recommendation(kanban_home: Path) -> None:
    # Merges promote/recompute, reviewer/CAS mutators, and assign/archive/delete — all "not found" for a recommendation id.
    with kb.connect_closing() as conn:
        rec_id = _make_recommendation(conn)
        mutators = (kb.assign_task(conn, rec_id, "alice"), kb.archive_task(conn, rec_id), kb.delete_task(conn, rec_id), kb.delete_archived_task(conn, rec_id))
        assert mutators == (False, False, False, False)
        assert _row(conn, rec_id)["assignee"] is None
        _force_row(conn, rec_id, status="todo")
        ok, reason = kb.promote_task(conn, rec_id, actor="op")
        assert ok is False and "not found" in reason
        kb.recompute_ready(conn)
        assert _row(conn, rec_id)["status"] == "todo"
        _force_row(conn, rec_id, status="running", current_run_id=None)
        assert kb.request_review(conn, rec_id, force=True, with_reason=True) == (False, "task not found")
        assert kb.request_changes(conn, rec_id, reason="fix it") == (False, "task not found")
        assert kb.complete_task(conn, rec_id, result="done") is False and kb.block_task(conn, rec_id, reason="stuck") is False
        assert _row(conn, rec_id)["status"] == "running"
        _force_row(conn, rec_id, status="blocked")
        assert kb.unblock_task(conn, rec_id) is False
def test_dispatch_never_spawns_or_transitions_recommendation(kanban_home: Path, all_assignees_spawnable) -> None:
    with kb.connect_closing() as conn:
        ready_rec = _make_recommendation(conn, recommendation_subject_id="s1")
        _force_row(conn, ready_rec, status="ready", claim_lock=None, assignee="alice")
        review_rec = _make_recommendation(conn, recommendation_subject_id="s2")
        _force_row(conn, review_rec, assignee="alice")  # already status='review'
        # A real work task in the review lane proves ordinary review dispatch keeps functioning while skipping the recommendation.
        work_id = kb.create_task(conn, title="ordinary work", assignee="alice")
        kb.claim_task(conn, work_id)
        kb.request_review(conn, work_id, force=True)
        result = kb.dispatch_once(conn, spawn_fn=lambda *a, **k: 4242)
        spawned_ids = {t[0] for t in result.spawned}
        assert ready_rec not in spawned_ids and review_rec not in spawned_ids and work_id in spawned_ids
        assert (_row(conn, ready_rec)["status"], _row(conn, ready_rec)["claim_lock"]) == ("ready", None)
        assert (_row(conn, review_rec)["status"], _row(conn, review_rec)["claim_lock"]) == ("review", None)
def test_ordinary_work_task_full_lifecycle_still_works(kanban_home: Path, all_assignees_spawnable) -> None:
    """Control: isolation must not regress the normal work-task flow."""
    with kb.connect_closing() as conn:
        task_id = kb.create_task(conn, title="ordinary work", assignee="alice")
        assert task_id in {t.id for t in kb.list_tasks(conn)}
        claimed = kb.claim_task(conn, task_id)
        assert claimed is not None and claimed.status == "running"
        assert kb.complete_task(conn, task_id, result="ok", summary="done") is True
        task = kb.get_task(conn, task_id)
        assert task is not None and task.status == "done"
        assert kb.board_stats(conn)["by_status"]["done"] == 1
# --- Plugin layer (plugins/kanban/dashboard/plugin_api.py) ---
def _load_plugin_router():
    repo_root = Path(__file__).resolve().parents[2]
    plugin_file = repo_root / "plugins" / "kanban" / "dashboard" / "plugin_api.py"
    spec = importlib.util.spec_from_file_location("hermes_dashboard_plugin_kanban_isolation_test", plugin_file)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod
@pytest.fixture
def plugin_mod(kanban_home):
    return _load_plugin_router()
@pytest.fixture
def client(plugin_mod):
    app = FastAPI()
    app.include_router(plugin_mod.router, prefix="/api/plugins/kanban")
    return TestClient(app)
def test_plugin_scan_link_and_bulk_reject_recommendation(client, plugin_mod, kanban_home) -> None:
    with kb.connect_closing() as conn:
        work_id = kb.create_task(conn, title="ordinary work")
        rec_id = _make_recommendation(conn)
        tail_id = conn.execute("SELECT MAX(id) FROM task_events").fetchone()[0]
    plugin_mod._ws_upgrade_authorized = lambda _ws: True
    with client.websocket_connect("/api/plugins/kanban/events?since=0") as ws:
        ws.send_text("poll")
        batch = ws.receive_json()
    assert rec_id not in {e["task_id"] for e in batch["events"]} and batch["cursor"] == tail_id
    r = client.get("/api/plugins/kanban/board")
    all_ids = {t["id"] for col in r.json()["columns"] for t in col["tasks"]}
    assert work_id in all_ids and rec_id not in all_ids
    assert client.get("/api/plugins/kanban/stats").json()["by_status"].get("review", 0) == 0
    r = client.get("/api/plugins/kanban/diagnostics")
    assert rec_id not in {d["task_id"] for d in r.json()["diagnostics"]}
    r = client.get("/api/plugins/kanban/boards")
    default_board = next(b for b in r.json()["boards"] if b["slug"] == kb.DEFAULT_BOARD)
    assert default_board["counts"].get("review", 0) == 0 and default_board["total"] == 1
    r = client.post("/api/plugins/kanban/links", json={"parent_id": rec_id, "child_id": work_id})
    assert r.status_code == 400
    r = client.post("/api/plugins/kanban/tasks/bulk", json={"ids": [rec_id], "priority": 9})
    entry = r.json()["results"][0]
    assert entry["ok"] is False and "not found" in entry["error"]
    with kb.connect_closing() as conn:
        assert _row(conn, rec_id)["priority"] == 0
@pytest.mark.parametrize("make_request,post_check", [
    (lambda c, tid: c.get(f"/api/plugins/kanban/tasks/{tid}"), None),
    (lambda c, tid: c.delete(f"/api/plugins/kanban/tasks/{tid}"), lambda conn, tid: _row(conn, tid) is not None),
    (lambda c, tid: c.post(f"/api/plugins/kanban/tasks/{tid}/comments", json={"body": "hello", "author": "dashboard"}), lambda conn, tid: kb.list_comments(conn, tid) == []),
    (lambda c, tid: c.patch(f"/api/plugins/kanban/tasks/{tid}", json={"priority": 9}), lambda conn, tid: _row(conn, tid)["priority"] == 0),
], ids=["get", "delete", "comment", "patch"])
def test_plugin_404_for_recommendation(client, kanban_home, make_request, post_check) -> None:
    with kb.connect_closing() as conn:
        rec_id = _make_recommendation(conn)
    r = make_request(client, rec_id)
    assert r.status_code == 404
    if post_check is not None:
        with kb.connect_closing() as conn:
            assert post_check(conn, rec_id)
def test_plugin_ordinary_task_control_still_works(client, kanban_home) -> None:
    """Control: ordinary create/patch/comment via the plugin API is unaffected by the isolation predicates."""
    r = client.post("/api/plugins/kanban/tasks", json={"title": "ordinary work", "assignee": "alice"})
    assert r.status_code == 200, r.text
    task_id = r.json()["task"]["id"]
    r = client.patch(f"/api/plugins/kanban/tasks/{task_id}", json={"priority": 5})
    assert r.status_code == 200, r.text
    assert r.json()["task"]["priority"] == 5
    r = client.post(f"/api/plugins/kanban/tasks/{task_id}/comments", json={"body": "hi", "author": "dashboard"})
    assert r.status_code == 200, r.text
    r = client.get("/api/plugins/kanban/board")
    ready = next(c for c in r.json()["columns"] if c["name"] == "ready")
    assert any(t["id"] == task_id for t in ready["tasks"])
