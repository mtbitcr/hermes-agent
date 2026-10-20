"""Finding 88: a scheduled planning job reads the shared Projects list.

A worker runs under its own profile's HERMES_HOME with the dispatcher's board
pins, yet Projects live in the root registry beside the shared board. The
steward snapshot must resolve them there, and only there.
"""

from __future__ import annotations

import os
import sqlite3
import threading
import time
from pathlib import Path

import pytest

from hermes_cli import kanban_db, owner_workspace as ow, projects_db
from tools import approval

PLANNER = "raphael-planner"

# title -> (status, block_kind, seconds since completion)
_CARDS = {
    "Publish the weekly summary": ("done", None, 60),
    "Sort the old notes": ("done", None, 10 * 86_400),
    "Confirm the visible result": ("review", None, None),
    "Choose the launch date": ("blocked", "needs_input", None),
    "Repair the broken import": ("triage", None, None),
    "Draft the next milestone": ("ready", None, None),
}


@pytest.fixture
def root(tmp_path, monkeypatch):
    """The shared root, entered as the default profile (AGENTS.md profile_env)."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    return home


@pytest.fixture
def ctx():
    return ow.OwnerContext(actor="default", profile="default", session="run_owt")


def _auto_approve(session_key: str, timeout: float = 5.0) -> None:
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


def _bootstrap(ctx, name: str) -> dict:
    approval.register_gateway_notify(ctx.session, lambda data: None)
    approver = threading.Thread(target=_auto_approve, args=(ctx.session,))
    approver.start()
    result = ow.bootstrap(
        ctx, idempotency_key="root-registry-" + name.lower().replace(" ", "-"),
        name=name,
    )
    approver.join()
    return result


def _fill_board(setup: dict) -> None:
    now = int(time.time())
    with kanban_db.connect(board=setup["board"]) as conn:
        for title, (status, block_kind, done_ago) in _CARDS.items():
            task_id = kanban_db.create_task(
                conn, title=title, project_id=setup["project_id"],
            )
            with kanban_db.write_txn(conn):
                conn.execute(
                    "UPDATE tasks SET status = ?, block_kind = ?, completed_at = ? "
                    "WHERE id = ?",
                    (status, block_kind,
                     None if done_ago is None else now - done_ago, task_id),
                )
    kanban_db.write_board_dispatch_state(setup["board"], dispatch_enabled=True)


def _planning_task(setup: dict) -> str:
    with kanban_db.connect(board=setup["board"]) as conn:
        return kanban_db.create_task(
            conn, title="Plan the next milestone", assignee=PLANNER,
            project_id=setup["project_id"],
        )


def _enter_planning_worker(monkeypatch, root: Path, board: str, task_id: str) -> None:
    """Take on the store pins kanban_db_dispatch._default_spawn gives a worker.

    The board pins are resolved first, in the dispatcher's own environment.
    """
    pins = {
        "HERMES_KANBAN_DB": str(kanban_db.kanban_db_path(board=board)),
        "HERMES_KANBAN_WORKSPACES_ROOT": str(kanban_db.workspaces_root(board=board)),
        "HERMES_KANBAN_BOARD": board,
    }
    monkeypatch.setenv("HERMES_HOME", str(root / "profiles" / PLANNER))
    monkeypatch.setenv("HERMES_PROFILE", PLANNER)
    monkeypatch.setenv("HERMES_KANBAN_TASK", task_id)
    for name, value in pins.items():
        monkeypatch.setenv(name, value)


def _rows(path: Path, sql: str, params=()) -> list:
    conn = sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True)
    try:
        return conn.execute(sql, params).fetchall()
    finally:
        conn.close()


def _board_db(root: Path, board: str) -> Path:
    return root / "kanban" / "boards" / board / "kanban.db"


def _assert_snapshot_matches_stores(snapshot: dict, root: Path, project_id: str) -> None:
    """Hold the snapshot to the root registry and the Project's board, read
    directly instead of through the code under test."""
    [(name, board)] = _rows(
        root / "projects.db",
        "SELECT name, board_slug FROM projects WHERE id = ?", (project_id,),
    )
    cards = _rows(
        _board_db(root, board),
        "SELECT title, status, block_kind, completed_at FROM tasks "
        "WHERE project_id = ? AND task_kind = 'work' AND status != 'archived'",
        (project_id,),
    )
    assert cards
    cutoff = int(time.time()) - snapshot["lookback_days"] * 86_400
    done = sorted(
        title for title, status, _, completed_at in cards
        if status == "done" and completed_at is not None and completed_at >= cutoff
    )
    decisions = sorted(
        title for title, status, block_kind, _ in cards
        if status == "blocked" and block_kind == "needs_input"
    )
    assert snapshot["project"] == {"name": name}
    assert snapshot["counts"] == {
        "open": sum(status != "done" for _, status, _, _ in cards),
        "completed_in_window": len(done),
        "needs_attention": sum(
            status == "triage" or (status == "blocked" and block_kind != "needs_input")
            for _, status, block_kind, _ in cards
        ),
        "awaiting_review": sum(status == "review" for _, status, _, _ in cards),
    }
    assert sorted(item["title"] for item in snapshot["progress"]) == done
    assert sorted(item["title"] for item in snapshot["decisions_needed"]) == decisions


def test_planning_worker_reads_every_project_from_the_root_registry(root, ctx, monkeypatch):
    """A Project registered only in the root registry gives a snapshot, from
    the worker's own board and from another one."""
    project = _bootstrap(ctx, "Garden Plan")
    _fill_board(project)
    worker = _bootstrap(ctx, "Planning Desk")
    task_id = _planning_task(worker)
    profile_home = root / "profiles" / PLANNER
    profile_home.mkdir(parents=True)
    with projects_db.connect_closing(profile_home / "projects.db"):
        pass
    _enter_planning_worker(monkeypatch, root, worker["board"], task_id)

    assert Path(os.environ["HERMES_KANBAN_DB"]).resolve() == (
        _board_db(root, worker["board"]).resolve()
    )
    assert _rows(profile_home / "projects.db", "SELECT id FROM projects") == []
    registered = sorted(
        project_id for (project_id,) in _rows(
            root / "projects.db", "SELECT id FROM projects WHERE archived = 0",
        )
    )
    assert registered == sorted([project["project_id"], worker["project_id"]])
    stores = [
        root / "projects.db",
        profile_home / "projects.db",
        _board_db(root, project["board"]),
        _board_db(root, worker["board"]),
    ]
    before = [path.stat().st_mtime_ns for path in stores]

    for project_id in registered:
        snapshot = ow.project_steward_snapshot(project_id=project_id)
        _assert_snapshot_matches_stores(snapshot, root, project_id)

    assert [path.stat().st_mtime_ns for path in stores] == before


def test_project_only_in_the_profile_registry_is_not_found(root, ctx, monkeypatch):
    """The profile's own registry is not a source, even for a Project whose
    board would verify: the root registry is the one source."""
    stray = _bootstrap(ctx, "Profile Copy")
    profile_home = root / "profiles" / PLANNER
    profile_home.mkdir(parents=True)
    source = sqlite3.connect(root / "projects.db")
    target = sqlite3.connect(profile_home / "projects.db")
    try:
        source.backup(target)
        with source:
            source.execute(
                "DELETE FROM projects WHERE id = ?", (stray["project_id"],),
            )
    finally:
        source.close()
        target.close()
    worker = _bootstrap(ctx, "Planning Desk")
    task_id = _planning_task(worker)
    _enter_planning_worker(monkeypatch, root, worker["board"], task_id)

    assert _rows(profile_home / "projects.db", "SELECT id FROM projects") == [
        (stray["project_id"],)
    ]
    assert _rows(
        root / "projects.db",
        "SELECT id FROM projects WHERE id = ?", (stray["project_id"],),
    ) == []
    assert _board_db(root, stray["board"]).is_file()

    with pytest.raises(ow.OwnerWorkspaceError) as excinfo:
        ow.project_steward_snapshot(project_id=stray["project_id"])
    assert excinfo.value.code == "project_not_found"


def test_default_profile_snapshot_is_unchanged(root, ctx):
    """The default profile's home is the root, so its snapshot stays the same."""
    project = _bootstrap(ctx, "Garden Plan")
    _fill_board(project)

    snapshot = ow.project_steward_snapshot(project_id=project["project_id"])

    _assert_snapshot_matches_stores(snapshot, root, project["project_id"])
    assert snapshot["schema_version"] == 2
    assert snapshot["lookback_days"] == 7
    assert snapshot["counts"] == {
        "open": 4, "completed_in_window": 1, "needs_attention": 1,
        "awaiting_review": 1,
    }
    assert snapshot["execution"] == {
        "state": "waiting_for_you",
        "summary": "Raphael needs your answer before the plan can continue.",
        "paused": False,
    }
    assert snapshot["decisions_needed"] == [
        {"title": "Choose the launch date", "state": "Waiting for your answer"}
    ]
