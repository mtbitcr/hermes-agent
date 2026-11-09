"""The Project list route answers ``last_changed_at`` only to a reader that names ``project_last_change_v1`` once."""

from __future__ import annotations

import contextlib
import json
import os
import sqlite3
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from hermes_cli import kanban_db, owner_workspace as ow, projects_db
from tests.gateway.test_api_server_owner_board_counts import _get

TOKEN = "?capabilities=project_last_change_v1"


@pytest.fixture
def ctx():
    return ow.OwnerContext(actor="default", profile="default", session="run_owt")


def _projects(ctx, count: int) -> list[dict]:
    """Commit ``count`` Projects; their bootstrap confirmation is not under test."""
    projects = []
    for index in range(count):
        with patch.object(ow, "_confirm", lambda *_args, **_kwargs: {"approved": True}):
            result = ow.bootstrap(ctx, idempotency_key=f"key-{index}", name=f"Project {index}")
        assert result["ok"] is True
        projects.append(result)
    return projects


def _stamp_board(project: dict, cards: list[dict]) -> None:
    """Replace the board's cards with ``cards``, each stamped with exact times."""
    with contextlib.closing(kanban_db.connect(board=project["board"])) as conn:
        with kanban_db.write_txn(conn):
            conn.execute("DELETE FROM task_events")
            conn.execute("DELETE FROM tasks")
        for card in cards:
            task_id = kanban_db.create_task(
                conn, title="card", project_id=card.get("project_id", project["project_id"]),
            )
            with kanban_db.write_txn(conn):
                conn.execute("DELETE FROM task_events WHERE task_id = ?", (task_id,))
                conn.execute(
                    "UPDATE tasks SET created_at = ?, started_at = ?, completed_at = ?, "
                    "status = ?, task_kind = ? WHERE id = ?",
                    (
                        card["created"], card.get("started"), card.get("completed"),
                        card.get("status", "ready"), card.get("kind", "work"), task_id,
                    ),
                )
                for kind, at in card.get("events", ()):
                    conn.execute(
                        "INSERT INTO task_events (task_id, kind, created_at) VALUES (?, ?, ?)",
                        (task_id, kind, at),
                    )


def _by_id(body: bytes) -> dict[str, dict]:
    return {item["project_id"]: item for item in json.loads(body)["data"]}


@pytest.mark.asyncio
async def test_each_item_carries_its_boards_newest_task_change_in_whole_seconds(ctx):
    first, second, third, fourth = _projects(ctx, 4)
    _stamp_board(first, [
        {"created": 1000, "started": 2000, "completed": 6000, "status": "done",
         "events": [("created", 1000), ("completed", 6000)]},
        {"created": 1500, "events": [("created", 1500), ("edited", 4500)]},
        # Cards that are not this Project's work never move its time.
        {"created": 50000, "kind": "control", "events": [("created", 50000)]},
        {"created": 60000, "project_id": "p_someone_else", "events": [("created", 60000)]},
    ])
    _stamp_board(second, [
        {"created": 100, "events": [("created", 100), ("blocked", 8000.9)]},
        {"created": 200, "status": "archived", "events": [("created", 200), ("archived", 7000)]},
        # A worker heartbeat is liveness, not a change of the task.
        {"created": 300, "events": [("created", 300), ("heartbeat", 99999)]},
    ])
    _stamp_board(third, [{"created": 10, "started": 20, "completed": 7777, "status": "done"}])
    _stamp_board(fourth, [{"created": 5, "events": [("created", 5), ("commented", 5)]}])

    status, body = await _get(ctx, TOKEN)

    assert status == 200
    listed = _by_id(body)
    assert [listed[p["project_id"]]["last_changed_at"] for p in (first, second, third, fourth)] == [
        6000, 8000, 7777, 5,
    ]
    assert all(type(item["last_changed_at"]) is int for item in listed.values())


@pytest.mark.asyncio
async def test_a_readable_board_without_work_is_null(ctx):
    empty, busy = _projects(ctx, 2)
    _stamp_board(empty, [{"created": 60000, "kind": "control", "events": [("created", 60000)]}])
    _stamp_board(busy, [{"created": 40, "events": [("created", 40)]}])

    _, body = await _get(ctx, TOKEN)

    listed = _by_id(body)
    assert listed[empty["project_id"]]["last_changed_at"] is None
    assert listed[busy["project_id"]]["last_changed_at"] == 40


@pytest.mark.asyncio
async def test_an_unreadable_board_is_null_and_does_not_break_the_list(ctx):
    projects = _projects(ctx, 4)
    for index, project in enumerate(projects):
        _stamp_board(project, [{"created": 100 * (index + 1), "events": [("created", 100 * (index + 1))]}])
    kanban_db.kanban_db_path(projects[1]["board"]).unlink()
    kanban_db.kanban_db_path(projects[2]["board"]).write_bytes(b"this is not a database" * 100)

    status, body = await _get(ctx, TOKEN)

    assert status == 200
    listed = _by_id(body)
    assert len(listed) == 4
    assert [listed[p["project_id"]]["last_changed_at"] for p in projects] == [100, None, None, 400]


@pytest.mark.asyncio
async def test_a_board_without_the_event_table_is_null(ctx):
    [project] = _projects(ctx, 1)
    with contextlib.closing(sqlite3.connect(kanban_db.kanban_db_path(project["board"]))) as raw:
        raw.execute("DROP TABLE task_events")
        raw.commit()

    status, body = await _get(ctx, TOKEN)

    assert status == 200
    assert _by_id(body)[project["project_id"]]["last_changed_at"] is None


@pytest.mark.asyncio
async def test_the_token_adds_exactly_one_key_and_changes_nothing_else(ctx):
    projects = _projects(ctx, 3)
    _stamp_board(projects[1], [{"created": 900, "events": [("created", 900)]}])
    kanban_db.kanban_db_path(projects[2]["board"]).unlink()

    _, plain = await _get(ctx)
    _, changed = await _get(ctx, TOKEN)
    _, both = await _get(ctx, "?capabilities=removal_state,project_last_change_v1")
    _, removal = await _get(ctx, "?capabilities=removal_state")

    plain_data = json.loads(plain)["data"]
    changed_data = json.loads(changed)["data"]
    assert [item["project_id"] for item in changed_data] == [item["project_id"] for item in plain_data]
    assert {item["project_id"] for item in changed_data} == {p["project_id"] for p in projects}
    assert all(set(item) - set(old) == {"last_changed_at"} for item, old in zip(changed_data, plain_data))
    assert [{k: v for k, v in item.items() if k != "last_changed_at"} for item in changed_data] == plain_data
    both_data = json.loads(both)["data"]
    assert [{k: v for k, v in item.items() if k != "last_changed_at"} for item in both_data] == json.loads(removal)["data"]
    assert [item["last_changed_at"] for item in both_data] == [item["last_changed_at"] for item in changed_data]


@pytest.mark.asyncio
@pytest.mark.parametrize("query", [
    "",
    "?capabilities=project_last_change_v2",
    "?capabilities=project_last_change",
    "?capabilities=PROJECT_LAST_CHANGE_V1",
    "?capabilities=project_last_change_v1&capabilities=project_last_change_v1",
    "?capabilities=removal_state&capabilities=project_last_change_v1",
    "?capabilities=board_counts_v1",
])
async def test_anything_but_one_exact_token_leaves_the_field_out(ctx, query):
    _projects(ctx, 2)
    _, baseline = await _get(ctx)

    status, body = await _get(ctx, query)

    assert status == 200
    assert all("last_changed_at" not in item for item in json.loads(body)["data"])
    if query != "?capabilities=board_counts_v1":
        assert body == baseline


@pytest.mark.asyncio
async def test_board_counts_work_alone_and_together_with_the_new_token(ctx):
    projects = _projects(ctx, 3)
    for index, project in enumerate(projects):
        _stamp_board(project, [
            {"created": 10 * (index + 1), "status": "done", "events": [("created", 10 * (index + 1))]}
            for _ in range(index + 1)
        ])
    kanban_db.kanban_db_path(projects[1]["board"]).unlink()

    _, counts_only = await _get(ctx, "?capabilities=board_counts_v1")
    _, both = await _get(ctx, "?capabilities=board_counts_v1,project_last_change_v1")
    _, time_only = await _get(ctx, TOKEN)

    counts_data, both_data, time_data = (json.loads(b)["data"] for b in (counts_only, both, time_only))
    assert all("last_changed_at" not in item for item in counts_data)
    assert [item["board_counts"]["total"] if item["board_counts"] else None for item in counts_data] == [1, None, 3]
    assert [item["board_counts"] for item in both_data] == [item["board_counts"] for item in counts_data]
    assert [item["last_changed_at"] for item in both_data] == [10, None, 30]
    assert [item["last_changed_at"] for item in time_data] == [10, None, 30]
    assert all("board_counts" not in item for item in time_data)
    assert [{k: v for k, v in item.items() if k not in ("board_counts", "last_changed_at")} for item in both_data] == [
        {k: v for k, v in item.items() if k != "last_changed_at"} for item in time_data
    ]


@pytest.mark.asyncio
async def test_a_real_task_transition_moves_only_its_own_projects_time(ctx, monkeypatch):
    first, second = _projects(ctx, 2)
    for project in (first, second):
        with contextlib.closing(kanban_db.connect(board=project["board"])) as conn:
            kanban_db.create_task(conn, title="earlier work", project_id=project["project_id"])
    _, before = await _get(ctx, TOKEN)
    later = max(item["last_changed_at"] for item in _by_id(before).values()) + 3600
    with monkeypatch.context() as clock:
        clock.setattr(time, "time", lambda: later + 0.75)
        with contextlib.closing(kanban_db.connect(board=first["board"])) as conn:
            task_id = kanban_db.create_task(conn, title="later work", project_id=first["project_id"])
            assert kanban_db.complete_task(conn, task_id, result="done") is True

    _, after = await _get(ctx, TOKEN)

    listed_before, listed_after = _by_id(before), _by_id(after)
    assert listed_before[first["project_id"]]["last_changed_at"] < later
    assert listed_after[first["project_id"]]["last_changed_at"] == later
    assert listed_after[second["project_id"]] == listed_before[second["project_id"]]


def test_a_worker_pinned_to_its_board_still_reads_every_projects_own_board_time(tmp_path, monkeypatch):
    """Under the environment the dispatcher gives a worker (its own board pinned by HERMES_KANBAN_DB and
    HERMES_KANBAN_BOARD, its task named by HERMES_KANBAN_TASK, HERMES_HOME a profile home under the root), the
    list still reaches each Project's own board file and reports that board's own newest task change."""
    root = tmp_path / "hermes_root"
    profile_home = root / "profiles" / "worker"
    profile_home.mkdir(parents=True)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(root))
    kanban_db._INITIALIZED_PATHS.clear()
    kanban_db.init_db()
    owner = ow.resolve_owner_context()
    first, second = _projects(owner, 2)
    _stamp_board(first, [{"created": 1111, "events": [("created", 1111), ("edited", 2222)]}])
    _stamp_board(second, [{"created": 3333, "events": [("created", 3333), ("completed", 4444)]}])
    with contextlib.closing(sqlite3.connect(root / "projects.db")) as source:
        with contextlib.closing(sqlite3.connect(profile_home / "projects.db")) as target:
            source.backup(target)
    monkeypatch.setenv("HERMES_HOME", str(profile_home))
    with contextlib.closing(kanban_db.connect(board=first["board"])) as conn:
        [pinned] = [row[0] for row in conn.execute("SELECT id FROM tasks")]
    paths = {p["project_id"]: kanban_db.board_dir(p["board"]) / "kanban.db" for p in (first, second)}
    monkeypatch.setenv("HERMES_KANBAN_DB", str(paths[first["project_id"]]))
    monkeypatch.setenv("HERMES_KANBAN_BOARD", first["board"])
    monkeypatch.setenv("HERMES_KANBAN_TASK", pinned)
    with contextlib.closing(kanban_db.connect()) as conn:
        opened = conn.execute("PRAGMA database_list").fetchone()[2]
    assert os.path.realpath(opened) == os.path.realpath(paths[first["project_id"]])
    assert projects_db.projects_db_path().parent == profile_home

    plain = ow.list_committed_projects(owner)
    listed = ow.list_committed_projects(owner, project_last_change=True)

    assert [item["project_id"] for item in listed] == [p["project_id"] for p in (first, second)]
    assert [item["last_changed_at"] for item in listed] == [2222, 4444]
    for item in listed:
        with contextlib.closing(sqlite3.connect(paths[item["project_id"]])) as raw:  # the independent read
            [(newest,)] = raw.execute("SELECT MAX(created_at) FROM task_events").fetchall()
        assert item["last_changed_at"] == newest
    assert [{k: v for k, v in item.items() if k != "last_changed_at"} for item in listed] == plain
