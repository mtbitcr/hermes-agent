"""The Project list can carry each Project's board counts, built by the snapshot's own count code."""

from __future__ import annotations

import threading
import time

import pytest

from hermes_cli import kanban_db, owner_workspace as ow, projects_db
from tools import approval


@pytest.fixture
def ctx():
    return ow.OwnerContext(actor="default", profile="default", session="run_owt")


def _auto_approve(session_key: str, timeout: float = 10.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        with approval._lock:
            queue = approval._gateway_queues.get(session_key)
            aid = queue[0].approval_id if queue else None
        if aid:
            approval.resolve_gateway_approval(session_key, "once", approval_id=aid)
            return
        time.sleep(0.005)
    raise TimeoutError("no approval was queued")


def _new_project(ctx, name: str) -> dict:
    approval.register_gateway_notify(ctx.session, lambda data: None)
    approver = threading.Thread(target=_auto_approve, args=(ctx.session,))
    approver.start()
    result = ow.bootstrap(ctx, idempotency_key=f"key-{name}", name=name)
    approver.join()
    assert result["ok"] is True
    with projects_db.connect_closing() as pconn:
        slug = projects_db.get_project(pconn, result["project_id"]).slug
    return {**result, "slug": slug}


def _add_tasks(project, statuses) -> None:
    with kanban_db.connect(board=project["board"]) as conn:
        for index, status in enumerate(statuses):
            task_id = kanban_db.create_task(
                conn, title=f"{status} work {index}", project_id=project["project_id"],
            )
            with kanban_db.write_txn(conn):
                conn.execute("UPDATE tasks SET status = ? WHERE id = ?", (status, task_id))


def _listed(ctx, **kwargs):
    return {item["project_id"]: item for item in ow.list_committed_projects(ctx, **kwargs)}


def test_board_counts_equal_the_snapshots_counts_and_total(ctx):
    project = _new_project(ctx, "Mixed")
    _add_tasks(project, [
        "triage", "ready", "ready", "running", "blocked", "review",
        "done", "done", "done", "archived", "archived",
    ])

    snapshot = ow.read_project_snapshot(ctx, project["slug"])
    item = _listed(ctx, board_counts=True)[project["project_id"]]

    assert item["board_counts"] == {
        "counts": snapshot["board"]["counts"], "total": snapshot["board"]["total"],
    }
    assert item["board_counts"]["counts"]["archived"] == 2
    assert item["board_counts"]["total"] == 9


def test_without_the_flag_the_items_are_unchanged(ctx):
    project = _new_project(ctx, "Plain")
    _add_tasks(project, ["ready", "done"])

    plain = ow.list_committed_projects(ctx)
    counted = ow.list_committed_projects(ctx, board_counts=True)

    assert all("board_counts" not in item for item in plain)
    assert [{k: v for k, v in item.items() if k != "board_counts"} for item in counted] == plain
    assert all(set(item) - set(plain[0]) == {"board_counts"} for item in counted)


def test_a_missing_board_store_is_null_and_others_are_unaffected(ctx):
    projects = [_new_project(ctx, f"Project {index}") for index in range(3)]
    for index, project in enumerate(projects):
        _add_tasks(project, ["ready"] * (index + 1))
    kanban_db.kanban_db_path(projects[1]["board"]).unlink()

    listed = _listed(ctx, board_counts=True)

    assert listed[projects[1]["project_id"]]["board_counts"] is None
    for index in (0, 2):
        counts = listed[projects[index]["project_id"]]["board_counts"]
        assert counts["counts"]["ready"] == index + 1
        assert counts["total"] == index + 1


def test_listing_does_not_call_the_snapshot_reader(ctx, monkeypatch):
    project = _new_project(ctx, "Quiet")

    def fail(*args, **kwargs):
        raise AssertionError("read_project_snapshot must not run for the list")

    monkeypatch.setattr(ow, "read_project_snapshot", fail)

    assert _listed(ctx, board_counts=True)[project["project_id"]]["board_counts"]["total"] == 0
