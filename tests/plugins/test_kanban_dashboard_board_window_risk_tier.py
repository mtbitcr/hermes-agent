"""The owner-workspace board window's per-task risk_tier / risk_tier_raised facts.

``risk_tier`` is the tier recorded on the card (null when none), and
``risk_tier_raised`` is true exactly when the card has a ``risk_tier_raised``
event. Both come from data the window already reads.
"""
from __future__ import annotations

from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from hermes_cli import kanban_db as kb
from hermes_cli import projects_db as pdb
from hermes_cli.dashboard_auth import clear_providers, register_provider
from hermes_cli.dashboard_auth import token_auth
from plugins.dashboard_auth.raphael_workspace import (
    BOARD,
    PROJECT,
    WorkspaceReadTokenProvider,
    token_store,
)
from plugins.kanban.dashboard import plugin_api

OLD_KEYS = {
    "id", "title", "assignee_name", "responsibility", "updated_at", "event_revision",
    "review_state", "stopped_work", "parent_ids", "child_ids",
}
NEW_KEYS = {"risk_tier", "risk_tier_raised"}


@pytest.fixture
def workspace_surface(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb._INITIALIZED_PATHS.clear()

    repo = tmp_path / "workspace-repo"
    repo.mkdir()
    with pdb.connect_closing() as conn:
        project_id = pdb.create_project(
            conn, name="Raphael Workspace", primary_path=str(repo)
        )
        assert pdb.get_project(conn, project_id).slug == PROJECT

    kb.create_board(BOARD, name="Raphael Workspace", project_id=project_id)
    kb.init_db(board=BOARD)
    conn = kb.connect(board=BOARD)
    try:
        ids = {}
        for name, tier in (("tier0", 0), ("tier1", 1), ("tier2", 2), ("none", None)):
            ids[name] = kb.create_task(
                conn, title=f"Work {name}", assignee="coder", board=BOARD,
                owned_paths=[], risk_tier=tier,
            )
        ids["raised"] = kb.create_task(
            conn, title="Work raised", assignee="coder", board=BOARD,
            owned_paths=[], risk_tier=0,
        )
        conn.execute("UPDATE tasks SET risk_tier = 1 WHERE id = ?", (ids["raised"],))
        kb._append_event(
            conn, ids["raised"], "risk_tier_raised",
            {"from": 0, "to": 1, "reviewer": "coder", "run_id": 1},
        )
        conn.commit()
    finally:
        conn.close()

    token_dir = home / "workspace-token"
    token_dir.mkdir(mode=0o700)
    token_path = token_dir / "bearer"
    token_store.issue(out_path=token_path)
    bearer = token_path.read_text(encoding="utf-8").strip()

    clear_providers()
    token_auth.clear_token_routes()
    register_provider(WorkspaceReadTokenProvider())
    plugin_api._register_workspace_machine_routes()

    app = FastAPI()
    app.include_router(plugin_api.router, prefix="/api/plugins/kanban")

    @app.middleware("http")
    async def machine_auth(request, call_next):
        return await token_auth.token_auth_middleware(request, call_next)

    with TestClient(app) as client:
        yield {
            "client": client,
            "headers": {"Authorization": f"Bearer {bearer}"},
            "ids": ids,
        }

    clear_providers()
    token_auth.clear_token_routes()
    kb._INITIALIZED_PATHS.clear()


def _rows(surface) -> dict:
    response = surface["client"].get(
        f"/api/plugins/kanban/board?board={BOARD}",
        headers={**surface["headers"], "X-Forwarded-For": "203.0.113.10"},
    )
    assert response.status_code == 200
    return {
        task["id"]: task
        for column in response.json()["columns"] for task in column["tasks"]
    }


def test_a_card_with_each_recorded_tier_shows_that_tier(workspace_surface):
    rows = _rows(workspace_surface)
    ids = workspace_surface["ids"]
    assert [rows[ids[name]]["risk_tier"] for name in ("tier0", "tier1", "tier2")] == [0, 1, 2]
    assert rows[ids["raised"]]["risk_tier"] == 1


def test_a_card_with_no_recorded_tier_shows_null(workspace_surface):
    rows = _rows(workspace_surface)
    row = rows[workspace_surface["ids"]["none"]]
    assert "risk_tier" in row and row["risk_tier"] is None


def test_only_a_card_with_a_raise_event_shows_true(workspace_surface):
    rows = _rows(workspace_surface)
    ids = workspace_surface["ids"]
    assert rows[ids["raised"]]["risk_tier_raised"] is True
    others = [row for task_id, row in rows.items() if task_id != ids["raised"]]
    assert len(others) == 4
    assert all(row["risk_tier_raised"] is False for row in others)


def test_every_existing_key_and_value_stays_the_same(workspace_surface):
    rows = _rows(workspace_surface)
    conn = kb.connect(board=BOARD)
    try:
        for task_id, row in rows.items():
            assert set(row) == OLD_KEYS | NEW_KEYS
            task = kb.get_task(conn, task_id)
            latest, revision = conn.execute(
                "SELECT MAX(created_at), MAX(id) FROM task_events WHERE task_id = ?",
                (task_id,),
            ).fetchone()
            assert {key: row[key] for key in OLD_KEYS} == {
                "id": task_id,
                "title": task.title,
                "assignee_name": "coder",
                "responsibility": task.responsibility,
                "updated_at": plugin_api._workspace_iso_timestamp(latest),
                "event_revision": revision,
                "review_state": kb.REVIEW_STATE_NONE,
                "stopped_work": kb.STOPPED_WORK_NONE,
                "parent_ids": [],
                "child_ids": [],
            }
    finally:
        conn.close()
