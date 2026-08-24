"""Item 32G: exact read-only Kanban surface used by Raphael Workspace."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from hermes_cli import kanban_db as kb
from hermes_cli import owner_workspace as ow
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
        ready_id = kb.create_task(
            conn,
            title="Ready work",
            assignee="coder",
            responsibility="B03",
            board=BOARD,
        )
        running_id = kb.create_task(
            conn, title="Running work", assignee="coder", board=BOARD
        )
        claimed = kb.claim_task(conn, running_id, claimer="test:worker")
        assert claimed is not None and claimed.current_run_id is not None
        run_id = claimed.current_run_id
        conn.execute(
            "UPDATE tasks SET worker_pid = 4242 WHERE id = ?", (running_id,)
        )
        conn.execute(
            "UPDATE task_runs SET worker_pid = 4242 WHERE id = ?", (run_id,)
        )

        attachment_dir = kb.task_attachments_dir(ready_id, board=BOARD)
        attachment_dir.mkdir(parents=True, exist_ok=True)
        attachment_path = attachment_dir / "proof.txt"
        attachment_body = b"workspace proof\n"
        attachment_path.write_bytes(attachment_body)
        attachment_id = kb.add_attachment(
            conn,
            ready_id,
            filename="proof.txt",
            stored_path=str(attachment_path),
            content_type="text/plain",
            size=len(attachment_body),
            uploaded_by="test",
        )

        # A recommendation is owner-review authority, not ordinary work. It
        # must never leak into this work/task/count/attachment/run surface.
        recommendation_task_id = "t_recommendation_hidden"
        recommendation_run_id = 9001
        recommendation_attachment_id = 9001
        conn.execute(
            "INSERT INTO tasks "
            "(id, title, status, created_at, workspace_kind, task_kind, "
            "project_id, review_policy) "
            "VALUES (?, 'Hidden advice', 'review', 1, 'scratch', "
            "'capability_recommendation', ?, 'owner')",
            (recommendation_task_id, project_id),
        )
        conn.execute(
            "INSERT INTO task_runs (id, task_id, status, started_at) "
            "VALUES (?, ?, 'done', 1)",
            (recommendation_run_id, recommendation_task_id),
        )
        recommendation_path = (
            kb.task_attachments_dir(recommendation_task_id, board=BOARD)
            / "hidden.txt"
        )
        recommendation_path.parent.mkdir(parents=True, exist_ok=True)
        recommendation_path.write_bytes(b"hidden")
        conn.execute(
            "INSERT INTO task_attachments "
            "(id, task_id, filename, stored_path, content_type, size, created_at) "
            "VALUES (?, ?, 'hidden.txt', ?, 'text/plain', 6, 1)",
            (
                recommendation_attachment_id,
                recommendation_task_id,
                str(recommendation_path),
            ),
        )
        conn.commit()
        baseline_event_count = conn.execute(
            "SELECT COUNT(*) FROM task_events"
        ).fetchone()[0]
    finally:
        conn.close()

    profile_dir = home / "profiles" / "coder"
    profile_dir.mkdir(parents=True)
    (profile_dir / "config.yaml").write_text("profile: coder\n", encoding="utf-8")

    kb.create_board("other-board", name="Other")
    kb.init_db(board="other-board")
    other_conn = kb.connect(board="other-board")
    try:
        other_task_id = kb.create_task(other_conn, title="Other work", board="other-board")
        other_conn.execute(
            "INSERT INTO task_runs (id, task_id, status, started_at) "
            "VALUES (9999, ?, 'done', 1)",
            (other_task_id,),
        )
        other_file = kb.task_attachments_dir(
            other_task_id, board="other-board"
        ) / "other.txt"
        other_file.parent.mkdir(parents=True, exist_ok=True)
        other_file.write_bytes(b"other")
        other_conn.execute(
            "INSERT INTO task_attachments "
            "(id, task_id, filename, stored_path, content_type, size, created_at) "
            "VALUES (9999, ?, 'other.txt', ?, 'text/plain', 5, 1)",
            (other_task_id, str(other_file)),
        )
        other_conn.commit()
    finally:
        other_conn.close()

    token_dir = home / "workspace-token"
    token_dir.mkdir(mode=0o700)
    token_path = token_dir / "bearer"
    record = token_store.issue(out_path=token_path)
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
            "bearer": bearer,
            "record": record,
            "project_id": project_id,
            "ready_id": ready_id,
            "running_id": running_id,
            "run_id": run_id,
            "attachment_id": attachment_id,
            "attachment_body": attachment_body,
            "other_task_id": other_task_id,
            "recommendation_task_id": recommendation_task_id,
            "recommendation_run_id": recommendation_run_id,
            "recommendation_attachment_id": recommendation_attachment_id,
            "baseline_event_count": baseline_event_count,
        }

    clear_providers()
    token_auth.clear_token_routes()
    kb._INITIALIZED_PATHS.clear()


def _get(surface, path: str, *, query: str = ""):
    return surface["client"].get(
        path + query,
        headers={**surface["headers"], "X-Forwarded-For": "203.0.113.10"},
    )


def test_exact_workspace_projection_is_current_scoped_and_read_only(workspace_surface):
    s = workspace_surface
    profiles = _get(s, "/api/plugins/kanban/profiles")
    assert profiles.status_code == 200
    assert set(profiles.json()) == {"profiles"}
    coder_profile = next(
        item for item in profiles.json()["profiles"] if item["name"] == "coder"
    )
    assert coder_profile == {"name": "coder", "description": ""}
    assert all(
        set(item) == {"name", "description"}
        for item in profiles.json()["profiles"]
    )

    project = _get(s, "/api/plugins/kanban/projects")
    assert project.status_code == 200
    assert project.json() == {
        "projects": [
            {"id": s["project_id"], "slug": PROJECT, "name": "Raphael Workspace"}
        ]
    }

    boards = _get(s, "/api/plugins/kanban/boards")
    assert boards.status_code == 200
    board_item = boards.json()["boards"][0]
    assert set(board_item) == {"slug", "name", "project_id", "counts", "total"}
    assert board_item["slug"] == BOARD
    assert board_item["project_id"] == s["project_id"]
    assert board_item["counts"]["ready"] == 1
    assert board_item["counts"]["running"] == 1
    assert board_item["counts"]["review"] == 0
    assert board_item["total"] == 2

    board = _get(s, "/api/plugins/kanban/board", query=f"?board={BOARD}")
    assert board.status_code == 200
    assert set(board.json()) == {"columns"}
    task_items = [
        task for column in board.json()["columns"] for task in column["tasks"]
    ]
    assert {task["id"] for task in task_items} == {s["ready_id"], s["running_id"]}
    assert all(
        set(task) == {
            "id", "title", "assignee_name", "responsibility", "updated_at", "event_revision",
            "parent_ids", "child_ids",
        }
        for task in task_items
    )
    assert next(task for task in task_items if task["id"] == s["ready_id"])["responsibility"] == "B03"
    assert next(task for task in task_items if task["id"] == s["running_id"])["responsibility"] is None
    assert all(task["updated_at"].endswith("Z") for task in task_items)
    assert all(type(task["event_revision"]) is int and task["event_revision"] > 0 for task in task_items)
    assert all(type(task["parent_ids"]) is list for task in task_items)
    assert all(type(task["child_ids"]) is list for task in task_items)

    workers = _get(
        s, "/api/plugins/kanban/workers/active", query=f"?board={BOARD}"
    )
    assert workers.status_code == 200
    assert workers.json() == {
        "workers": [
            {
                "profile": "coder",
                "task_title": "Running work",
                "started_at": workers.json()["workers"][0]["started_at"],
            }
        ]
    }
    assert type(workers.json()["workers"][0]["started_at"]) is int

    assignees = _get(s, "/api/plugins/kanban/assignees", query=f"?board={BOARD}")
    assert assignees.status_code == 200
    coder = next(item for item in assignees.json()["assignees"] if item["name"] == "coder")
    assert set(coder) == {"name", "on_disk", "counts"}
    assert coder["on_disk"] is True
    assert coder["counts"] == {"ready": 1, "running": 1}

    task = _get(
        s,
        f"/api/plugins/kanban/tasks/{s['running_id']}",
        query=f"?board={BOARD}",
    )
    assert task.status_code == 200
    assert task.json() == {"runs": [s["run_id"]]}

    attachments = _get(
        s,
        f"/api/plugins/kanban/tasks/{s['ready_id']}/attachments",
        query=f"?board={BOARD}",
    )
    assert attachments.status_code == 200
    item = attachments.json()["attachments"][0]
    assert set(item) == {"id", "filename", "media_type", "size", "created_at"}
    assert item["id"] == s["attachment_id"]
    assert item["created_at"].endswith("Z")

    run = _get(
        s, f"/api/plugins/kanban/runs/{s['run_id']}", query=f"?board={BOARD}"
    )
    assert run.status_code == 200
    assert set(run.json()["run"]) == {"started_at", "finished_at", "receipt"}
    assert run.json()["run"]["receipt"] == {
        "outcome": "running",
        "summary": "Work is still in progress.",
        "external_effect": {
            "state": "unknown",
            "summary": (
                "This record does not confirm whether an external service changed."
            ),
        },
        "runtime": {
            "state": "unknown",
            "summary": "This record does not contain an authoritative model route.",
        },
        "cost": {
            "state": "unknown",
            "summary": "This record does not contain an authoritative cost.",
        },
        "evidence": {"state": "available", "kind": "project_activity"},
    }
    assert run.json()["run"]["started_at"].endswith("Z")
    assert run.json()["run"]["finished_at"] is None

    attachment = _get(
        s,
        f"/api/plugins/kanban/attachments/{s['attachment_id']}",
        query=f"?board={BOARD}",
    )
    assert attachment.status_code == 200
    assert attachment.content == s["attachment_body"]
    assert set(attachment.headers) == {
        "content-type", "content-length", "content-disposition", "cache-control"
    }
    assert attachment.headers["content-type"] == "text/plain"
    assert attachment.headers["cache-control"] == "no-store"

    conn = kb.connect(board=BOARD)
    try:
        assert conn.execute("SELECT COUNT(*) FROM task_events").fetchone()[0] == s[
            "baseline_event_count"
        ]
    finally:
        conn.close()

    audit_text = (s["home"] / "logs" / "dashboard-auth.log").read_text(
        encoding="utf-8"
    )
    assert s["bearer"] not in audit_text
    entries = [json.loads(line) for line in audit_text.splitlines()]
    request_entries = [entry for entry in entries if entry.get("credential_id") == s["record"].token_id]
    assert request_entries
    assert all(entry.get("ip") != "203.0.113.10" for entry in request_entries)
    assert all("secret" not in json.dumps(entry).lower() for entry in request_entries)


def test_owner_title_capability_projects_board_and_worker_titles(workspace_surface):
    s = workspace_surface
    raw_ready = "B03 — db.password=hunter2verylongpassword"
    raw_running = (
        'R07: curl -H "Authorization: Bearer sk-abcdef1234567890"'
    )
    conn = kb.connect(board=BOARD)
    try:
        conn.execute(
            "UPDATE tasks SET title = ? WHERE id = ?", (raw_ready, s["ready_id"])
        )
        conn.execute(
            "UPDATE tasks SET title = ? WHERE id = ?",
            (raw_running, s["running_id"]),
        )
        conn.commit()
    finally:
        conn.close()

    query = (
        f"?board={BOARD}&capabilities="
        f"{plugin_api.WORKSPACE_OWNER_TITLES_CAPABILITY}"
    )
    board = _get(s, "/api/plugins/kanban/board", query=query)
    workers = _get(s, "/api/plugins/kanban/workers/active", query=query)

    assert board.status_code == 200
    assert workers.status_code == 200
    task_titles = {
        task["id"]: task["title"]
        for column in board.json()["columns"]
        for task in column["tasks"]
    }
    assert task_titles == {
        s["ready_id"]: ow.owner_title(raw_ready),
        s["running_id"]: ow.owner_title(raw_running),
    }
    assert workers.json()["workers"][0]["task_title"] == ow.owner_title(
        raw_running
    )
    serialized = json.dumps({"board": board.json(), "workers": workers.json()})
    assert "hunter2verylongpassword" not in serialized
    assert "sk-abcdef1234567890" not in serialized
    assert all(not title.startswith("B03") for title in task_titles.values())
    assert all(not title.startswith("R07") for title in task_titles.values())

    # No capability means the legacy response shape and semantics remain
    # unchanged for an older Workspace during a Hermes-first rollout.
    legacy = _get(s, "/api/plugins/kanban/board", query=f"?board={BOARD}")
    legacy_titles = {
        task["id"]: task["title"]
        for column in legacy.json()["columns"]
        for task in column["tasks"]
    }
    assert legacy_titles[s["ready_id"]] == raw_ready
    assert legacy_titles[s["running_id"]] == raw_running


@pytest.mark.parametrize(
    ("path", "query"),
    [
        ("/api/plugins/kanban/profiles", "?extra=1"),
        ("/api/plugins/kanban/projects", "?extra=1"),
        ("/api/plugins/kanban/boards", "?include_archived=false"),
        ("/api/plugins/kanban/board", ""),
        ("/api/plugins/kanban/board", "?board=other-board"),
        ("/api/plugins/kanban/board", f"?board={BOARD}&extra=1"),
        ("/api/plugins/kanban/board", f"?board={BOARD}&board={BOARD}"),
        ("/api/plugins/kanban/board", f"?board={BOARD}&capabilities=unknown"),
        (
            "/api/plugins/kanban/assignees",
            f"?board={BOARD}&capabilities="
            f"{plugin_api.WORKSPACE_OWNER_TITLES_CAPABILITY}",
        ),
    ],
)
def test_query_contract_fails_closed(workspace_surface, path, query):
    response = _get(workspace_surface, path, query=query)
    assert response.status_code == 400
    assert response.json() == {"detail": "Bad Request"}


@pytest.mark.parametrize(
    "path",
    [
        "/api/plugins/kanban/tasks/{other_task_id}",
        "/api/plugins/kanban/tasks/{other_task_id}/attachments",
        "/api/plugins/kanban/attachments/9999",
        "/api/plugins/kanban/runs/9999",
    ],
)
def test_cross_board_objects_are_indistinguishable_from_missing(workspace_surface, path):
    s = workspace_surface
    response = _get(
        s,
        path.format(other_task_id=s["other_task_id"]),
        query=f"?board={BOARD}",
    )
    assert response.status_code == 404
    assert response.json() == {"detail": "Not Found"}


@pytest.mark.parametrize(
    "path",
    [
        "/api/plugins/kanban/tasks/{recommendation_task_id}",
        "/api/plugins/kanban/tasks/{recommendation_task_id}/attachments",
        "/api/plugins/kanban/attachments/{recommendation_attachment_id}",
        "/api/plugins/kanban/runs/{recommendation_run_id}",
    ],
)
def test_recommendation_objects_are_indistinguishable_from_missing(
    workspace_surface, path
):
    s = workspace_surface
    response = _get(s, path.format(**s), query=f"?board={BOARD}")
    assert response.status_code == 404
    assert response.json() == {"detail": "Not Found"}


def test_attachment_metadata_is_sanitized(workspace_surface):
    s = workspace_surface
    conn = kb.connect(board=BOARD)
    try:
        conn.execute(
            "UPDATE task_attachments SET filename = ?, content_type = ? "
            "WHERE id = ?",
            ("../../unsafe\n.txt", "text/html\r\nX-Injected: yes", s["attachment_id"]),
        )
        conn.commit()
    finally:
        conn.close()

    metadata = _get(
        s,
        f"/api/plugins/kanban/tasks/{s['ready_id']}/attachments",
        query=f"?board={BOARD}",
    )
    assert metadata.status_code == 200
    item = metadata.json()["attachments"][0]
    assert item["filename"] == "unsafe_.txt"
    assert item["media_type"] == "application/octet-stream"

    download = _get(
        s,
        f"/api/plugins/kanban/attachments/{s['attachment_id']}",
        query=f"?board={BOARD}",
    )
    assert download.status_code == 200
    assert download.headers["content-type"] == "application/octet-stream"
    assert download.headers["content-disposition"] == (
        'attachment; filename="unsafe_.txt"'
    )


@pytest.mark.parametrize(
    "path",
    [
        "/api/plugins/kanban/attachments/not-an-integer",
        "/api/plugins/kanban/runs/not-an-integer",
    ],
)
def test_typed_path_rejections_are_strictly_audited(workspace_surface, path):
    s = workspace_surface
    response = _get(s, path, query=f"?board={BOARD}")
    assert response.status_code == 422

    entries = [
        json.loads(line)
        for line in (s["home"] / "logs" / "dashboard-auth.log")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert any(
        entry.get("credential_id") == s["record"].token_id
        and entry.get("reason") == "downstream_rejected"
        and entry.get("status") == 422
        for entry in entries
    )


def test_interactive_and_session_bearer_paths_remain_interactive(workspace_surface):
    s = workspace_surface
    for headers in ({}, {"Authorization": "Bearer dashboard-session"}):
        response = s["client"].get(
            f"/api/plugins/kanban/board?board={BOARD}", headers=headers
        )
        assert response.status_code == 200
        assert {"columns", "tenants", "assignees", "latest_event_id", "now"} <= set(
            response.json()
        )


def test_revocation_is_effective_on_the_next_request(workspace_surface):
    s = workspace_surface
    assert _get(s, "/api/plugins/kanban/projects").status_code == 200
    token_store.revoke(s["record"].token_id)
    response = _get(s, "/api/plugins/kanban/projects")
    assert response.status_code == 401
    assert response.json() == {"error": "unauthenticated", "detail": "Unauthorized"}


def test_broken_scope_binding_fails_closed(workspace_surface):
    s = workspace_surface
    kb.write_board_metadata(BOARD, project_id="p_wrong")
    response = _get(s, "/api/plugins/kanban/projects")
    assert response.status_code == 404
    assert response.json() == {"detail": "Not Found"}


def test_process_path_overrides_cannot_widen_the_fixed_board(
    workspace_surface, monkeypatch
):
    s = workspace_surface
    monkeypatch.setenv(
        "HERMES_KANBAN_DB", str(kb.board_dir("other-board") / "kanban.db")
    )
    monkeypatch.setenv(
        "HERMES_KANBAN_ATTACHMENTS_ROOT",
        str(kb.board_dir("other-board") / "attachments"),
    )

    board = _get(s, "/api/plugins/kanban/board", query=f"?board={BOARD}")
    task_ids = {
        task["id"] for column in board.json()["columns"] for task in column["tasks"]
    }
    assert task_ids == {s["ready_id"], s["running_id"]}
    attachment = _get(
        s,
        f"/api/plugins/kanban/attachments/{s['attachment_id']}",
        query=f"?board={BOARD}",
    )
    assert attachment.content == s["attachment_body"]
