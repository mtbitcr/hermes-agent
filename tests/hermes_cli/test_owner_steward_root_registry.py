"""Finding 88: a scheduled planning job reads the shared Projects list.

A worker runs under its own profile's HERMES_HOME with the dispatcher's board
pins, yet Projects live in the root registry beside the shared board. The
steward snapshot must resolve them there, and only there.
"""

from __future__ import annotations

import contextlib
import dataclasses
import hashlib
import json
import os
import sqlite3
import threading
import time
from pathlib import Path

import pytest

from hermes_cli import kanban_db, owner_workspace as ow, projects_db
from tools import approval, owner_workspace_tools

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


# What a planning profile's own home may hold where a registry would be: no
# home at all, a home without one, a zero-byte file, the schema without any
# Project, or bytes that are not SQLite.
_PROFILE_REGISTRIES = ("no_home", "missing", "empty", "initialized", "corrupt")


def _remove(ctx, project: dict) -> None:
    """Remove the Project recoverably, as its owner, and let the real driver
    carry the removal to Done on its own thread."""
    key = "root-registry-removal-" + project["board"]
    with projects_db.connect_closing() as conn:
        revision = ow._project_lifecycle_revision(conn, ctx, project["project_id"])
    request = {
        "idempotency_key": key, "project_id": project["project_id"],
        "expected_revision": revision, "action": "start",
    }
    owner = dataclasses.replace(ctx, authority=ow.OwnerProposalAuthority(
        actor=ctx.actor, profile=ctx.profile, session=ctx.session,
        conversation="conversation", response_id="response",
        operation="owner_project_removal", idempotency_key=key,
        payload_digest=ow._digest(request),
    ))
    approval.register_gateway_notify(ctx.session, lambda data: None)
    approver = threading.Thread(target=_auto_approve, args=(ctx.session,))
    approver.start()
    started = ow.project_removal(owner, **request)
    approver.join()
    assert started["ok"], started
    for thread in list(ow._removal_background_threads):
        thread.join(timeout=60)


@pytest.fixture
def removed(root, ctx):
    """A root Project whose recoverable removal reached Done, as the default
    profile: its record and its removal stay in the root registry, its board
    is gone."""
    project = _bootstrap(ctx, "Garden Plan")
    _remove(ctx, project)
    return project


def _lay_profile_registry(home: Path, state: str) -> None:
    if state == "no_home":
        return
    home.mkdir(parents=True)
    registry = home / "projects.db"
    if state == "empty":
        registry.write_bytes(b"")
    elif state == "initialized":
        with projects_db.connect_closing(registry):
            pass
    elif state == "corrupt":
        registry.write_bytes(b"This is not a SQLite database.\n" * 64)


def _entries(path: Path):
    """Name -> (size, mtime, digest of the bytes) of a file, or of a folder
    and everything under it, the folder itself as "."; None when the path
    does not exist."""
    if not path.exists():
        return None
    entries = {}
    for entry in (path, *(sorted(path.rglob("*")) if path.is_dir() else ())):
        info = entry.stat()
        entries[entry.relative_to(path).as_posix()] = (
            info.st_size if entry.is_file() else None,
            info.st_mtime_ns,
            hashlib.sha256(entry.read_bytes()).hexdigest()
            if entry.is_file() else None,
        )
    return entries


@contextlib.contextmanager
def _unchanged(*paths: Path):
    """Hold every path to its entries, whether the body answers or refuses."""
    before = {path: _entries(path) for path in paths}
    try:
        yield
    finally:
        for path, entries in before.items():
            assert _entries(path) == entries, path


def _without_generated_at(snapshot: dict) -> dict:
    return {key: value for key, value in snapshot.items() if key != "generated_at"}


def _assert_removal_matches_root_registry(
    removal_state: dict, root: Path, project_id: str,
) -> None:
    """Hold a served removal_state to the completed operation the root
    registry records, read directly instead of through the code under test."""
    columns = (
        "phase", "mode", "retained_copy_id", "receipt_id", "consequences_digest",
        "accepted_at", "applied_at", "completed_at",
    )
    [row] = _rows(
        root / "projects.db",
        "SELECT " + ", ".join(columns) + " FROM project_removal_operations "
        "WHERE project_id = ?",
        (project_id,),
    )
    recorded = dict(zip(columns, row))
    assert recorded["phase"] == "done"
    assert recorded["mode"] == "reversible" and recorded["retained_copy_id"]
    assert removal_state["mode"] == "recoverable"
    assert removal_state["restorable"] is True
    assert {key: removal_state[key] for key in columns if key != "mode"} == {
        key: value for key, value in recorded.items() if key != "mode"
    }


def _enter_worker_beside(ctx, monkeypatch, root: Path, removed: dict) -> None:
    """Enter a planning worker pinned to its own board, then prove from the
    stores that the removed Project is registered at the root and its board
    is gone."""
    worker = _bootstrap(ctx, "Planning Desk")
    task_id = _planning_task(worker)
    _enter_planning_worker(monkeypatch, root, worker["board"], task_id)

    assert Path(os.environ["HERMES_KANBAN_DB"]).resolve() == (
        _board_db(root, worker["board"]).resolve()
    )
    assert os.environ["HERMES_KANBAN_BOARD"] == worker["board"] != removed["board"]
    assert _rows(
        root / "projects.db",
        "SELECT board_slug, archived FROM projects WHERE id = ?",
        (removed["project_id"],),
    ) == [(removed["board"], 0)]
    assert not kanban_db.board_exists(removed["board"])
    assert not _board_db(root, removed["board"]).exists()


@pytest.mark.parametrize("owner_waiting", [False, True])
@pytest.mark.parametrize("profile_registry", _PROFILE_REGISTRIES)
def test_planning_worker_reads_a_removed_project_without_writing(
    root, ctx, removed, monkeypatch, profile_registry, owner_waiting,
):
    """A root Project whose removal reached Done reads as deleted from a
    planning worker exactly as it does for the default profile, whatever the
    worker's own profile holds; that home gains no file and no byte, and the
    root registry is only read."""
    project_id = removed["project_id"]
    default_snapshot = ow.project_steward_snapshot(
        project_id=project_id, owner_waiting=owner_waiting,
    )
    assert default_snapshot["execution"]["state"] == "deleted"
    profile_home = root / "profiles" / PLANNER
    _lay_profile_registry(profile_home, profile_registry)
    _enter_worker_beside(ctx, monkeypatch, root, removed)

    with _unchanged(profile_home, root / "projects.db"):
        snapshot = ow.project_steward_snapshot(
            project_id=project_id, owner_waiting=owner_waiting,
        )

    assert snapshot["execution"]["state"] == "deleted"
    assert _without_generated_at(snapshot) == _without_generated_at(default_snapshot)
    [(name,)] = _rows(
        root / "projects.db", "SELECT name FROM projects WHERE id = ?", (project_id,),
    )
    assert snapshot["project"] == {"name": name}
    _assert_removal_matches_root_registry(snapshot["removal_state"], root, project_id)


def test_steward_tool_reads_a_removed_project_from_a_planning_worker(
    root, ctx, removed, monkeypatch,
):
    """The real tool adapter answers a planning worker whose profile has no
    registry with the deleted projection, and writes nothing in that home."""
    project_id = removed["project_id"]
    default_snapshot = ow.project_steward_snapshot(project_id=project_id)
    profile_home = root / "profiles" / PLANNER
    _lay_profile_registry(profile_home, "missing")
    _enter_worker_beside(ctx, monkeypatch, root, removed)

    with _unchanged(profile_home, root / "projects.db"):
        answer = json.loads(
            owner_workspace_tools._handle_project_steward_snapshot(
                {"project_id": project_id}
            )
        )

    assert "error" not in answer, answer
    assert answer["execution"]["state"] == "deleted"
    assert _without_generated_at(answer) == _without_generated_at(default_snapshot)
    assert list(profile_home.iterdir()) == []
    _assert_removal_matches_root_registry(answer["removal_state"], root, project_id)
