"""The owner-workspace board window's per-task review_state/stopped_work facts.

These two fields must be the SAME batched, read-only, closed-vocabulary facts
the owner snapshot (``hermes_cli/owner_workspace.py``) already reports, never
an invented per-window rule.
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
        plain_id = kb.create_task(
            conn, title="Plain work", assignee="coder", board=BOARD,
        )
        review_id = kb.create_task(
            conn, title="Reviewable work", assignee="coder", board=BOARD,
        )
        blocked_id = kb.create_task(
            conn, title="Stuck work", assignee="coder", board=BOARD,
        )

        ok = kb.request_review(conn, review_id, summary="ready for eyes")
        assert ok is True

        ok = kb.block_task(conn, blocked_id, reason="need creds", kind="capability")
        assert ok is True

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
            "home": home,
            "client": client,
            "headers": {"Authorization": f"Bearer {bearer}"},
            "project_id": project_id,
            "plain_id": plain_id,
            "review_id": review_id,
            "blocked_id": blocked_id,
        }

    clear_providers()
    token_auth.clear_token_routes()
    kb._INITIALIZED_PATHS.clear()


def _get(surface, path: str, *, query: str = ""):
    return surface["client"].get(
        path + query,
        headers={**surface["headers"], "X-Forwarded-For": "203.0.113.10"},
    )


def _board_tasks(surface):
    response = _get(surface, "/api/plugins/kanban/board", query=f"?board={BOARD}")
    assert response.status_code == 200
    return [
        task for column in response.json()["columns"] for task in column["tasks"]
    ]


def test_every_task_carries_both_fields(workspace_surface):
    s = workspace_surface
    tasks = _board_tasks(s)
    assert {t["id"] for t in tasks} == {
        s["plain_id"], s["review_id"], s["blocked_id"],
    }
    for task in tasks:
        assert "review_state" in task
        assert "stopped_work" in task


def test_plain_task_reports_none_for_both(workspace_surface):
    s = workspace_surface
    tasks = _board_tasks(s)
    plain = next(t for t in tasks if t["id"] == s["plain_id"])
    assert plain["review_state"] == kb.REVIEW_STATE_NONE
    assert plain["stopped_work"] == kb.STOPPED_WORK_NONE


def test_review_lane_task_reports_awaiting_review(workspace_surface):
    s = workspace_surface
    tasks = _board_tasks(s)
    reviewed = next(t for t in tasks if t["id"] == s["review_id"])
    assert reviewed["review_state"] == "awaiting_review"
    assert reviewed["review_state"] == kb.REVIEW_STATE_AWAITING_REVIEW
    assert reviewed["stopped_work"] == kb.STOPPED_WORK_NONE


def test_capability_blocked_task_reports_stopped_work_capability(workspace_surface):
    s = workspace_surface
    tasks = _board_tasks(s)
    blocked = next(t for t in tasks if t["id"] == s["blocked_id"])
    assert blocked["stopped_work"] == "capability"
    assert blocked["stopped_work"] == kb.STOPPED_WORK_CAPABILITY
    assert blocked["review_state"] == kb.REVIEW_STATE_NONE


def test_values_are_drawn_from_the_owner_snapshots_closed_vocabulary(
    workspace_surface,
):
    s = workspace_surface
    tasks = _board_tasks(s)
    review_vocabulary = {
        kb.REVIEW_STATE_NONE,
        kb.REVIEW_STATE_AWAITING_REVIEW,
        kb.REVIEW_STATE_CHANGES_REQUESTED,
        kb.REVIEW_STATE_APPROVED,
    }
    stopped_work_vocabulary = {
        kb.STOPPED_WORK_NONE,
        kb.STOPPED_WORK_GAVE_UP,
        kb.STOPPED_WORK_CAPABILITY,
    }
    # The serialized spellings are the contract the owner workspace parses:
    # a renamed or added constant must fail here, not in the owner's browser.
    assert review_vocabulary == {"none", "awaiting_review", "changes_requested", "approved"}
    assert stopped_work_vocabulary == {"none", "gave_up", "capability"}
    assert tasks
    for task in tasks:
        assert task["review_state"] in review_vocabulary
        assert task["stopped_work"] in stopped_work_vocabulary


def test_window_agrees_exactly_with_the_kernel_batched_helpers(workspace_surface):
    s = workspace_surface
    tasks = _board_tasks(s)

    conn = kb.connect(board=BOARD)
    try:
        kernel_tasks = kb.list_tasks(conn, include_archived=False)
        expected_review = kb.task_review_states(conn, kernel_tasks)
        expected_stopped = kb.task_stopped_work_states(conn, kernel_tasks)
    finally:
        conn.close()

    for task in tasks:
        assert task["review_state"] == expected_review[task["id"]]
        assert task["stopped_work"] == expected_stopped[task["id"]]


def test_helpers_are_called_exactly_once_per_board_window_build(
    workspace_surface, monkeypatch,
):
    s = workspace_surface
    calls = {"review": 0, "stopped": 0}

    real_review = plugin_api.kanban_db.task_review_states
    real_stopped = plugin_api.kanban_db.task_stopped_work_states

    def counting_review(conn, tasks):
        calls["review"] += 1
        return real_review(conn, tasks)

    def counting_stopped(conn, tasks):
        calls["stopped"] += 1
        return real_stopped(conn, tasks)

    monkeypatch.setattr(plugin_api.kanban_db, "task_review_states", counting_review)
    monkeypatch.setattr(
        plugin_api.kanban_db, "task_stopped_work_states", counting_stopped
    )

    tasks = _board_tasks(s)
    assert len(tasks) > 1

    assert calls["review"] == 1
    assert calls["stopped"] == 1
