"""Card C5 (finding 144): owner pages tell finished work that is still being delivered.

A done card with an unfinished delivery row (no pull request yet, or a row open, review cards created,
auto-merge pending, armed, unknown or refused, base moved, or close refused) is active work to the steward,
listed as "Being delivered", so the overview never reads complete while it lasts. Only a snapshot reader that
names ``delivery_state_v1`` gets each card's ``delivery``; every other reader keeps today's card shape.
"""

from __future__ import annotations

import contextlib
import os
import sqlite3
from pathlib import Path

import pytest

from hermes_cli import kanban_db, owner_workspace as ow

UNFINISHED = ("open", "review_cards_created", "auto_merge_pending", "auto_merge_armed", "auto_merge_unknown",
              "auto_merge_refused", "base_moved", "close_refused")
FINISHED = ("returned_for_changes", "replaced", "updated", "merged", "closed_outside")
NO_PULL_REQUEST = None  # a row the publish step has not published yet
_TODAYS_CARD_KEYS = {
    "id", "title", "assignee_name", "responsibility", "updated_at", "event_revision", "review_state",
    "stopped_work", "risk_tier", "risk_tier_raised", "parent_ids", "child_ids",
}
_INDEPENDENT_COUNT = (
    "SELECT COUNT(DISTINCT t.id) FROM tasks AS t JOIN kanban_deliveries AS d ON d.source_task_id = t.id "
    "WHERE t.project_id = ? AND t.task_kind = 'work' AND t.status = 'done' AND ((d.pull_request_number IS NULL "
    "AND d.pull_request_state IS NULL) OR d.pull_request_state IN ({}))".format(", ".join("?" * len(UNFINISHED)))
)


@pytest.fixture
def owner() -> ow.OwnerContext:
    return ow.resolve_owner_context()


def _project(owner: ow.OwnerContext, name: str) -> dict:
    """Commit one owner Project whose board may dispatch."""
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(ow, "_confirm", lambda *_args, **_kwargs: {"approved": True})
        project = ow.bootstrap(owner, idempotency_key=f"setup-{name}", name=name)
    assert project["ok"] is True
    kanban_db.write_board_dispatch_state(project["board"], dispatch_enabled=True)
    project["slug"] = next(
        item["slug"] for item in ow.list_committed_projects(owner)
        if str(item["project_id"]) == project["project_id"]
    )
    return project


def _board(project: dict):
    return contextlib.closing(kanban_db.connect(board=project["board"]))


def _card(project: dict, title: str, *, status: str = "ready") -> str:
    with _board(project) as conn:
        task_id = kanban_db.create_task(
            conn, title=title, assignee="raphael-worker", project_id=project["project_id"],
        )
        if status == "running":
            assert kanban_db.claim_task(conn, task_id) is not None
        elif status == "done":
            assert kanban_db.complete_task(conn, task_id, result="done") is True
    return task_id


def _delivery(project: dict, task_id: str, state, *, head: str = "a" * 40) -> None:
    """One ledger row of ``task_id`` in ``state``; NO_PULL_REQUEST is a row with no pull request yet."""
    published = state is not NO_PULL_REQUEST
    with _board(project) as conn, kanban_db.write_txn(conn):
        conn.execute(
            "INSERT INTO kanban_deliveries (source_task_id, source_head, created_at, pull_request_number, "
            "pull_request_head, pull_request_state) VALUES (?, ?, 0, ?, ?, ?)",
            (task_id, head, 41 if published else None, head if published else None, state),
        )


def _cards(snapshot: dict) -> dict[str, dict]:
    return {task["title"]: task for column in snapshot["columns"] for task in column["tasks"]}


@pytest.mark.parametrize("state", [*UNFINISHED, NO_PULL_REQUEST])
def test_the_overview_is_not_complete_while_a_delivery_is_unfinished(owner, state):
    project = _project(owner, "Delivery Pilot")
    _delivery(project, _card(project, "Publish the launch page", status="done"), state)

    for steward in (
        ow.project_steward_snapshot(project_id=project["project_id"]),
        ow.read_project_snapshot(owner, project["slug"])["steward"],
    ):
        assert steward["execution"]["state"] == "working"
        assert steward["active_work"] == [{"title": "Publish the launch page", "state": "Being delivered"}]


@pytest.mark.parametrize("owner_waiting", [False, True])
def test_the_steward_lists_delivering_work_as_being_delivered(owner, owner_waiting):
    project = _project(owner, "Delivery Pilot")
    _card(project, "Draft the launch copy", status="running")
    _delivery(project, _card(project, "Publish the launch page", status="done"), "auto_merge_armed")
    _delivery(project, _card(project, "Write the press note", status="done"), "merged")
    _card(project, "Plan the follow-up", status="done")

    steward = ow.project_steward_snapshot(project_id=project["project_id"], owner_waiting=owner_waiting)

    # Without owner_waiting the steward keeps creation order, and these cards may share their creation second.
    assert (steward["active_work"] if owner_waiting else sorted(steward["active_work"], key=str)) == [
        {"title": "Draft the launch copy", "state": "In progress"},
        {"title": "Publish the launch page", "state": "Being delivered"},
    ]
    assert steward["execution"]["state"] == "working"
    assert set(steward) == {  # no new key
        "schema_version", "project", "generated_at", "lookback_days", "execution", "counts", "progress",
        "needs_attention", "decisions_needed", "active_work", "stale_candidates", "truncated",
    }
    assert steward["truncated"]["active_work"] is False


def test_a_delivery_state_reader_sees_each_card_delivery(owner):
    assert ow.DELIVERY_STATE_CAPABILITY == "delivery_state_v1"
    project = _project(owner, "Delivery Pilot")
    delivering = _card(project, "Publish the launch page", status="done")
    _delivery(project, delivering, "merged", head="b" * 40)
    _delivery(project, delivering, "base_moved")
    _delivery(project, _card(project, "Write the press note", status="done"), "closed_outside")
    _delivery(project, _card(project, "Draft the launch copy", status="running"), "open")
    _card(project, "Plan the follow-up", status="done")
    _card(project, "Book the photographer")

    cards = _cards(ow.read_project_snapshot(owner, project["slug"], delivery_state=True))

    assert {title: card["delivery"] for title, card in cards.items()} == {
        "Publish the launch page": "delivering", "Write the press note": None, "Draft the launch copy": None,
        "Plan the follow-up": None, "Book the photographer": None,
    }
    assert {frozenset(card) for card in cards.values()} == {frozenset(_TODAYS_CARD_KEYS | {"delivery"})}


@pytest.mark.parametrize("state", FINISHED)
def test_the_overview_reads_complete_once_every_delivery_is_merged_or_closed(owner, state):
    project = _project(owner, "Delivery Pilot")
    task_id = _card(project, "Publish the launch page", status="done")
    _delivery(project, task_id, "closed_outside", head="b" * 40)
    _delivery(project, task_id, state)
    _card(project, "Write the press note", status="done")

    snapshot = ow.read_project_snapshot(owner, project["slug"], delivery_state=True)

    for steward in (ow.project_steward_snapshot(project_id=project["project_id"]), snapshot["steward"]):
        assert steward["execution"] == {
            "state": "complete", "summary": "The approved work is complete.", "paused": False,
        }
        assert steward["active_work"] == []
    assert {card["delivery"] for card in _cards(snapshot).values()} == {None}


def test_a_reader_without_the_capability_gets_todays_card_shape(owner):
    project = _project(owner, "Delivery Pilot")
    _delivery(project, _card(project, "Publish the launch page", status="done"), "open")
    _delivery(project, _card(project, "Write the press note", status="done"), "merged")
    _card(project, "Draft the launch copy", status="running")

    for snapshot in (
        ow.read_project_snapshot(owner, project["slug"]),
        ow.read_project_snapshot(owner, project["slug"], run_context=True, owner_waiting=True, provider_wait=True),
    ):
        cards = _cards(snapshot)
        assert set(cards) == {"Publish the launch page", "Write the press note", "Draft the launch copy"}
        assert all("delivery" not in card for card in cards.values())
        assert all(set(card) - {"owner_wait"} == _TODAYS_CARD_KEYS for card in cards.values())


@pytest.mark.parametrize("state, unfinished", [
    *[(state, True) for state in UNFINISHED], (NO_PULL_REQUEST, True),
    *[(state, False) for state in FINISHED], ("close_pending", False)])
def test_the_kernel_read_follows_exactly_the_state_lists(owner, state, unfinished):
    from hermes_cli.kanban_delivery import unfinished_delivery_task_ids

    project = _project(owner, "Delivery Pilot")
    task_id = _card(project, "Publish the launch page", status="done")
    other = _card(project, "Write the press note", status="done")
    _delivery(project, task_id, state)
    _delivery(project, other, "open")

    with _board(project) as conn:
        assert unfinished_delivery_task_ids(conn, [task_id]) == ({task_id} if unfinished else set())
        assert unfinished_delivery_task_ids(conn, []) == set()


def test_an_older_returned_row_without_a_pull_request_is_finished_once_a_newer_row_merged(owner):
    """Owner rule of 2026-10-21, item 2: a row is unfinished by its state alone. A row returned for changes before
    the publish step took it has no pull request, and is finished: the card's newer row carries its delivery."""
    from hermes_cli.kanban_delivery import unfinished_delivery_task_ids

    project = _project(owner, "Delivery Pilot")
    task_id = _card(project, "Publish the launch page", status="done")
    with _board(project) as conn, kanban_db.write_txn(conn):
        conn.execute(
            "INSERT INTO kanban_deliveries (source_task_id, source_head, created_at, pull_request_state) "
            "VALUES (?, ?, 0, 'returned_for_changes')", (task_id, "b" * 40),
        )
    _delivery(project, task_id, "merged")

    with _board(project) as conn:
        assert unfinished_delivery_task_ids(conn, [task_id]) == set()
        assert conn.execute(_INDEPENDENT_COUNT, (project["project_id"], *UNFINISHED)).fetchone()[0] == 0
    snapshot = ow.read_project_snapshot(owner, project["slug"], delivery_state=True)
    for steward in (ow.project_steward_snapshot(project_id=project["project_id"]), snapshot["steward"]):
        assert (steward["execution"]["state"], steward["active_work"]) == ("complete", [])
    assert _cards(snapshot)["Publish the launch page"]["delivery"] is None


def test_a_missing_ledger_reads_as_no_unfinished_delivery_and_creates_no_schema(owner, tmp_path):
    from hermes_cli.kanban_delivery import unfinished_delivery_task_ids

    path = tmp_path / "board.db"
    with contextlib.closing(sqlite3.connect(path)) as conn:
        conn.execute("CREATE TABLE tasks (id TEXT)")
        conn.commit()
        assert unfinished_delivery_task_ids(conn, ["t_1"]) == set()
        assert [row[0] for row in conn.execute("SELECT name FROM sqlite_master")] == ["tasks"]

    project = _project(owner, "Delivery Pilot")
    _card(project, "Publish the launch page", status="done")
    board_path = kanban_db.board_dir(project["board"]) / "kanban.db"
    with contextlib.closing(sqlite3.connect(board_path)) as conn:
        conn.execute("DROP TABLE kanban_deliveries")
        conn.commit()
    steward = ow.project_steward_snapshot(project_id=project["project_id"])
    cards = _cards(ow.read_project_snapshot(owner, project["slug"], delivery_state=True))
    assert (steward["execution"]["state"], steward["active_work"]) == ("complete", [])
    assert cards["Publish the launch page"]["delivery"] is None


def test_any_other_ledger_error_still_raises(owner, tmp_path):
    from hermes_cli.kanban_delivery import unfinished_delivery_task_ids

    path = tmp_path / "board.db"
    with contextlib.closing(sqlite3.connect(path)) as conn:
        conn.execute("CREATE TABLE kanban_deliveries (id INTEGER PRIMARY KEY)")  # a ledger without its columns
        conn.commit()
        with pytest.raises(sqlite3.OperationalError):
            unfinished_delivery_task_ids(conn, ["t_1"])

    project = _project(owner, "Delivery Pilot")
    _card(project, "Publish the launch page", status="done")
    board_path = kanban_db.board_dir(project["board"]) / "kanban.db"
    with contextlib.closing(sqlite3.connect(board_path)) as conn:
        conn.execute("ALTER TABLE kanban_deliveries RENAME COLUMN pull_request_state TO renamed_state")
        conn.commit()
    with pytest.raises(ow.OwnerWorkspaceError) as steward_error:
        ow.project_steward_snapshot(project_id=project["project_id"])
    with pytest.raises(ow.OwnerWorkspaceError) as snapshot_error:
        ow.read_project_snapshot(owner, project["slug"], delivery_state=True)
    assert (steward_error.value.code, snapshot_error.value.code) == ("snapshot_unavailable", "snapshot_unavailable")


def test_a_worker_pinned_to_its_board_counts_the_other_projects_deliveries(tmp_path, monkeypatch):
    """Under the environment the dispatcher gives a worker (its own board pinned by HERMES_KANBAN_DB and
    HERMES_KANBAN_BOARD, its task named by HERMES_KANBAN_TASK, HERMES_HOME a profile home under the root), the
    steward of a Project on another board counts exactly the delivering work an independent read of that
    board's own file finds."""
    root = tmp_path / "hermes_root"
    profile_home = root / "profiles" / "worker"
    profile_home.mkdir(parents=True)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(root))
    kanban_db._INITIALIZED_PATHS.clear()
    kanban_db.init_db()
    owner = ow.resolve_owner_context()
    first = _project(owner, "First Pilot")
    second = _project(owner, "Second Pilot")
    with contextlib.closing(sqlite3.connect(root / "projects.db")) as source:
        with contextlib.closing(sqlite3.connect(profile_home / "projects.db")) as target:
            source.backup(target)
    monkeypatch.setenv("HERMES_HOME", str(profile_home))
    pinned = _card(first, "Draft the worker's own card", status="running")
    _delivery(first, _card(first, "Publish the first page", status="done"), "open")
    for title, state in (("Publish the launch page", "auto_merge_armed"), ("Write the press note", NO_PULL_REQUEST),
                         ("Send the newsletter", "merged"), ("Book the photographer", "base_moved")):
        _delivery(second, _card(second, title, status="done"), state)

    paths = {project["slug"]: kanban_db.board_dir(project["board"]) / "kanban.db" for project in (first, second)}
    monkeypatch.setenv("HERMES_KANBAN_DB", str(paths[first["slug"]]))
    monkeypatch.setenv("HERMES_KANBAN_BOARD", first["board"])
    monkeypatch.setenv("HERMES_KANBAN_TASK", pinned)
    with contextlib.closing(kanban_db.connect()) as conn:
        opened = conn.execute("PRAGMA database_list").fetchone()[2]
    assert os.path.realpath(opened) == os.path.realpath(paths[first["slug"]])

    for project in (first, second):
        steward = ow.project_steward_snapshot(project_id=project["project_id"], owner_waiting=True)
        with contextlib.closing(sqlite3.connect(paths[project["slug"]])) as raw:  # the independent count
            expected = raw.execute(_INDEPENDENT_COUNT, (project["project_id"], *UNFINISHED)).fetchone()[0]
        assert sum(item["state"] == "Being delivered" for item in steward["active_work"]) == expected
        assert expected == {first["slug"]: 1, second["slug"]: 3}[project["slug"]]
        assert steward["execution"]["state"] == "working"
