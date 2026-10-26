"""Show the owner where a finished change was delivered.

Only a snapshot reader that names ``delivery_outcome_v1`` gets each card's ``delivery_outcome``: what happened to the
change the card's latest delivery row carries (merged, returned by a red check, still open, or closed), the
repository the publish step recorded for that row, its pull request number and its link. A card with no row, or
whose latest row has no pull request or no published record, reads None. Every other reader keeps today's shape.
"""

from __future__ import annotations

import contextlib
import os
import sqlite3
import time
from pathlib import Path

import pytest

from hermes_cli import kanban_db, kanban_delivery as kd, owner_workspace as ow
from tests.hermes_cli.test_owner_delivery_state import (  # noqa: F401  (owner is a fixture)
    _TODAYS_CARD_KEYS,
    NO_PULL_REQUEST,
    UNFINISHED,
    _board,
    _card,
    _cards,
    _project,
    owner,
)

REPOSITORY = "studio/website"
# Every other state a row with a pull request takes: unfinished_delivery_task_ids does not count it.
CLOSED = ("returned_for_changes", "replaced", "updated", "closed_outside", "close_pending")
# The latest row of each of a Project's cards that has a pull request its own published record names.
_QUALIFYING_ROWS = (
    "SELECT COUNT(*) FROM kanban_deliveries AS d JOIN tasks AS t ON t.id = d.source_task_id "
    "WHERE t.project_id = ? AND t.task_kind = 'work' AND t.status != 'archived' "
    "AND d.id = (SELECT MAX(id) FROM kanban_deliveries WHERE source_task_id = d.source_task_id) "
    "AND d.pull_request_number IS NOT NULL AND EXISTS (SELECT 1 FROM task_events AS e "
    "WHERE e.task_id = d.source_task_id AND e.kind = 'delivery_published' "
    "AND json_extract(e.payload, '$.head') = d.pull_request_head "
    "AND json_extract(e.payload, '$.pull_request_number') = d.pull_request_number "
    "AND json_extract(e.payload, '$.repository') IS NOT NULL)"
)


def _published(project: dict, task_id: str, *, number: int = 41, head: str = "a" * 40) -> None:
    """The publish step's own record, on the card, of pull request ``number`` it published at ``head``."""
    with _board(project) as conn, kanban_db.write_txn(conn):
        kanban_db._append_event(conn, task_id, "delivery_published", {
            "repository": REPOSITORY, "branch": f"hermes/{task_id}", "pull_request_number": number,
            "head": head, "state": "created",
        })


def _delivery(project: dict, task_id: str, state, *, number=41, head: str = "a" * 40, published: bool = True) -> int:
    """One ledger row of ``task_id`` at ``head`` in ``state`` with pull request ``number`` (None: not published
    yet) and, when ``published``, its published record. Returns the row's id."""
    with _board(project) as conn, kanban_db.write_txn(conn):
        row_id = conn.execute(
            "INSERT INTO kanban_deliveries (source_task_id, source_head, created_at, pull_request_number, "
            "pull_request_head, pull_request_state) VALUES (?, ?, 0, ?, ?, ?)",
            (task_id, head, number, None if number is None else head, state),
        ).lastrowid
    if published and number is not None:
        _published(project, task_id, number=number, head=head)
    return row_id


def _red_check_returned(project: dict, task_id: str, row_id: int, *, number: int = 41, head: str = "a" * 40) -> None:
    """The delivery step's record that a red required check on ``head`` returned the work to its builder."""
    row = {"id": row_id, "source_task_id": task_id, "pull_request_head": head, "pull_request_number": number}
    with _board(project) as conn, kanban_db.write_txn(conn):
        kd._waiting(conn, row, "red_check_returned", reason="failed jobs: tests", tests=[])


def _set_state(project: dict, row_id: int, state: str) -> None:
    with _board(project) as conn, kanban_db.write_txn(conn):
        conn.execute("UPDATE kanban_deliveries SET pull_request_state = ? WHERE id = ?", (state, row_id))


def _outcomes(owner: ow.OwnerContext, project: dict) -> dict:
    cards = _cards(ow.read_project_snapshot(owner, project["slug"], delivery_outcome=True))
    return {title: card["delivery_outcome"] for title, card in cards.items()}


def _outcome(state: str, number: int = 41) -> dict:
    return {"state": state, "repository": REPOSITORY, "pull_request": number,
            "url": f"https://github.com/{REPOSITORY}/pull/{number}"}


@pytest.mark.parametrize("red_check", [False, True])
def test_a_merged_row_reads_merged(owner, red_check):
    """Merged comes first: a red check that once returned the merged head does not hide the merge."""
    assert ow.DELIVERY_OUTCOME_CAPABILITY == "delivery_outcome_v1"
    project = _project(owner, "Website Pilot")
    task_id = _card(project, "Publish the site", status="done")
    row_id = _delivery(project, task_id, "merged", number=7)
    if red_check:
        _red_check_returned(project, task_id, row_id, number=7)

    assert _outcomes(owner, project) == {"Publish the site": _outcome("merged", 7)}


def test_a_returned_row_that_was_later_replaced_reads_returned(owner):
    project = _project(owner, "Website Pilot")
    source = _card(project, "Publish the site", status="done")
    row_id = _delivery(project, source, "open")
    _red_check_returned(project, source, row_id)
    assert _outcomes(owner, project) == {"Publish the site": _outcome("returned")}

    # The rework card's own change replaces the returned one, whose pull request the delivery step then closes.
    rework = _card(project, "Fix the failing check", status="done")
    _delivery(project, rework, "auto_merge_armed", number=42, head="b" * 40)
    _set_state(project, row_id, "replaced")

    assert _outcomes(owner, project) == {
        "Publish the site": _outcome("returned"), "Fix the failing check": _outcome("open", 42),
    }


def test_a_red_check_of_another_head_does_not_return_the_latest_row(owner):
    project = _project(owner, "Website Pilot")
    task_id = _card(project, "Publish the site", status="done")
    older = _delivery(project, task_id, "replaced", number=40, head="b" * 40)
    _red_check_returned(project, task_id, older, number=40, head="b" * 40)
    _delivery(project, task_id, "open")

    assert _outcomes(owner, project) == {"Publish the site": _outcome("open")}


@pytest.mark.parametrize("state", UNFINISHED)
def test_an_open_row_reads_open(owner, state):
    from hermes_cli.kanban_delivery import unfinished_delivery_task_ids

    project = _project(owner, "Website Pilot")
    task_id = _card(project, "Publish the site", status="done")
    _delivery(project, task_id, "returned_for_changes", number=40, head="b" * 40)  # the card's earlier change
    _delivery(project, task_id, state)

    with _board(project) as conn:
        assert unfinished_delivery_task_ids(conn, [task_id]) == {task_id}
    assert _outcomes(owner, project) == {"Publish the site": _outcome("open")}


@pytest.mark.parametrize("state", CLOSED)
def test_a_closed_row_reads_closed(owner, state):
    from hermes_cli.kanban_delivery import unfinished_delivery_task_ids

    project = _project(owner, "Website Pilot")
    task_id = _card(project, "Publish the site", status="done")
    _delivery(project, task_id, state)

    with _board(project) as conn:
        assert unfinished_delivery_task_ids(conn, [task_id]) == set()
    assert _outcomes(owner, project) == {"Publish the site": _outcome("closed")}


def test_only_the_latest_row_decides(owner):
    """An earlier row the kernel still counts as unfinished (an armed row keeps its hold when a new head is
    approved) does not open the card's closed latest change, and an earlier merged change does not merge it."""
    from hermes_cli.kanban_delivery import unfinished_delivery_task_ids

    project = _project(owner, "Website Pilot")
    held = _card(project, "Publish the site", status="done")
    _delivery(project, held, "auto_merge_armed", number=40, head="b" * 40)
    _delivery(project, held, "closed_outside")
    merged = _card(project, "Write the about page", status="done")
    _delivery(project, merged, "merged", number=40, head="b" * 40)
    _delivery(project, merged, "returned_for_changes")

    with _board(project) as conn:
        assert unfinished_delivery_task_ids(conn, [held]) == {held}  # by its earlier row
    assert _outcomes(owner, project) == {
        "Publish the site": _outcome("closed"), "Write the about page": _outcome("closed"),
    }


def test_a_card_with_no_row_reads_none(owner):
    project = _project(owner, "Website Pilot")
    _card(project, "Plan the follow-up", status="done")
    _card(project, "Draft the launch copy", status="running")
    _card(project, "Book the photographer")
    _delivery(project, _card(project, "Publish the site", status="done"), "merged")

    cards = _cards(ow.read_project_snapshot(owner, project["slug"], delivery_outcome=True))

    assert {title: card["delivery_outcome"] for title, card in cards.items()} == {
        "Plan the follow-up": None, "Draft the launch copy": None, "Book the photographer": None,
        "Publish the site": _outcome("merged"),
    }
    assert {frozenset(card) for card in cards.values()} == {frozenset(_TODAYS_CARD_KEYS | {"delivery_outcome"})}


def test_a_reader_without_the_capability_gets_todays_exact_shape(owner, monkeypatch):
    project = _project(owner, "Website Pilot")
    merged = _card(project, "Publish the site", status="done")
    _delivery(project, merged, "merged")
    returned = _card(project, "Write the about page", status="done")
    _red_check_returned(project, returned, _delivery(project, returned, "replaced"))
    _delivery(project, _card(project, "Add the contact form", status="done"), "auto_merge_armed")
    _delivery(project, _card(project, "Send the newsletter", status="done"), NO_PULL_REQUEST, number=None)
    _card(project, "Draft the launch copy", status="running")
    now = time.time()
    monkeypatch.setattr(time, "time", lambda: now)  # one instant, so reads compare whole

    for options in ({}, {"run_context": True, "owner_waiting": True, "provider_wait": True, "delivery_state": True}):
        today = ow.read_project_snapshot(owner, project["slug"], **options)
        assert ow.read_project_snapshot(owner, project["slug"], delivery_outcome=False, **options) == today
        outcome = ow.read_project_snapshot(owner, project["slug"], delivery_outcome=True, **options)
        outcomes = [card.pop("delivery_outcome") for card in _cards(outcome).values()]
        assert {item and item["state"] for item in outcomes} == {"merged", "returned", "open", None}
        assert outcome == today  # the capability adds that one card key and changes nothing else
    assert {frozenset(card) for card in _cards(ow.read_project_snapshot(owner, project["slug"])).values()} == {
        frozenset(_TODAYS_CARD_KEYS)}


def test_a_row_with_no_pull_request_or_no_published_record_reads_none(owner):
    project = _project(owner, "Website Pilot")
    _delivery(project, _card(project, "Publish the site", status="done"), NO_PULL_REQUEST, number=None)
    _delivery(project, _card(project, "Send the newsletter", status="done"), "returned_for_changes", number=None)
    _delivery(project, _card(project, "Write the about page", status="done"), "merged", published=False)
    # The card's records of another pull request, or of the same one at another head, are not this row's.
    elsewhere = _card(project, "Add the contact form", status="done")
    _delivery(project, elsewhere, "open", published=False)
    _published(project, elsewhere, number=40)
    _published(project, elsewhere, head="b" * 40)
    # An earlier row's record is not the latest row's either.
    earlier = _card(project, "Book the photographer", status="done")
    _delivery(project, earlier, "replaced", number=40, head="b" * 40)
    _delivery(project, earlier, "auto_merge_armed", published=False)

    assert _outcomes(owner, project) == {
        "Publish the site": None, "Send the newsletter": None, "Write the about page": None,
        "Add the contact form": None, "Book the photographer": None,
    }


def test_a_board_without_the_deliveries_table_gives_every_card_none(owner):
    from hermes_cli.kanban_delivery_outcome import delivery_outcomes

    project = _project(owner, "Website Pilot")
    merged = _card(project, "Publish the site", status="done")
    _delivery(project, merged, "merged")
    running = _card(project, "Draft the launch copy", status="running")
    with contextlib.closing(sqlite3.connect(kanban_db.board_dir(project["board"]) / "kanban.db")) as conn:
        conn.execute("DROP TABLE kanban_deliveries")
        conn.commit()
        schema = conn.execute("SELECT * FROM sqlite_master ORDER BY name").fetchall()
        assert delivery_outcomes(conn, [merged, running]) == {merged: None, running: None}
        assert conn.execute("SELECT * FROM sqlite_master ORDER BY name").fetchall() == schema  # no schema made

    assert _outcomes(owner, project) == {"Publish the site": None, "Draft the launch copy": None}


def test_the_reader_writes_nothing(owner):
    from hermes_cli.kanban_delivery_outcome import delivery_outcomes

    project = _project(owner, "Website Pilot")
    merged = _card(project, "Publish the site", status="done")
    _delivery(project, merged, "merged")
    returned = _card(project, "Write the about page", status="done")
    _red_check_returned(project, returned, _delivery(project, returned, "replaced"))
    opened = _card(project, "Add the contact form", status="done")
    _delivery(project, opened, "open")
    closed = _card(project, "Send the newsletter", status="done")
    _delivery(project, closed, "closed_outside")
    ids = [merged, returned, opened, closed, _card(project, "Plan the follow-up", status="done")]
    path = kanban_db.board_dir(project["board"]) / "kanban.db"

    with contextlib.closing(sqlite3.connect(path)) as conn:
        board = list(conn.iterdump())
        outcomes = delivery_outcomes(conn, ids)
        assert (conn.total_changes, conn.in_transaction) == (0, False)
        assert list(conn.iterdump()) == board
    # A board opened read-only refuses any write, and gives the same answer.
    with contextlib.closing(sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True)) as conn:
        conn.execute("PRAGMA query_only = ON")
        assert delivery_outcomes(conn, ids) == outcomes
    assert [outcome and outcome["state"] for outcome in outcomes.values()] == [
        "merged", "returned", "open", "closed", None]


def test_a_worker_pinned_to_its_board_reads_each_projects_own_board(tmp_path, monkeypatch):
    """Under the environment the dispatcher gives a worker (its own board pinned by HERMES_KANBAN_DB and
    HERMES_KANBAN_BOARD, its task named by HERMES_KANBAN_TASK, HERMES_HOME a profile home under the root), each
    Project's snapshot still reads that Project's own board: exactly as many cards carry an outcome as an
    independent read of that board's own file finds qualifying rows."""
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
    for title, state, number, published in (
            ("Publish the site", "merged", 41, True), ("Write the about page", "replaced", 42, True),
            ("Add the contact form", "auto_merge_armed", 43, True),
            ("Send the newsletter", NO_PULL_REQUEST, None, True), ("Book the photographer", "open", 44, False)):
        _delivery(second, _card(second, title, status="done"), state, number=number, published=published)

    paths = {project["slug"]: kanban_db.board_dir(project["board"]) / "kanban.db" for project in (first, second)}
    monkeypatch.setenv("HERMES_KANBAN_DB", str(paths[first["slug"]]))
    monkeypatch.setenv("HERMES_KANBAN_BOARD", first["board"])
    monkeypatch.setenv("HERMES_KANBAN_TASK", pinned)
    with contextlib.closing(kanban_db.connect()) as conn:
        opened = conn.execute("PRAGMA database_list").fetchone()[2]
    assert os.path.realpath(opened) == os.path.realpath(paths[first["slug"]])

    for project in (first, second):
        cards = _cards(ow.read_project_snapshot(owner, project["slug"], delivery_outcome=True))
        with contextlib.closing(sqlite3.connect(paths[project["slug"]])) as raw:  # the independent count
            expected = raw.execute(_QUALIFYING_ROWS, (project["project_id"],)).fetchone()[0]
        assert sum(card["delivery_outcome"] is not None for card in cards.values()) == expected
        assert expected == {first["slug"]: 1, second["slug"]: 3}[project["slug"]]
