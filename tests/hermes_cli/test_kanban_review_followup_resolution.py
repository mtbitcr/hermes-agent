"""Approving a reviewed card resolves the rework its own verdicts recorded.

A changes-requested verdict records ONE rework item for the implementer and
vouches for it on the reviewed card's own ``review_followup_recorded`` event
(``_request_changes_within_txn``). Nothing used to close that item: the card
got approved, the rework it tracked was accepted, and the item stayed open on
the board until a person noticed it and tidied it away by hand.

The approval is what ends the round-trip, so the approval is what resolves
them -- inside the same transaction that writes ``done``, for BOTH approval
shapes (the reviewer's own clean verdict, and a person approving a card parked
in review), with a receipt naming the approving run.

The boundary is the kernel's own vouching: only items named by the reviewed
card's ``review_followup_recorded`` events move, so an item anyone else linked
under the same card is left exactly where it is. And only items that are still
open move, so a second approval records nothing new.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    """Isolated HERMES_HOME with an empty kanban DB."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb._INITIALIZED_PATHS.clear()
    kb.init_db()
    return home


def _vouched_followups(conn, task_id):
    """The items the KERNEL vouched for on this card, read from raw events.

    Deliberately reads ``task_events`` itself instead of calling the module's
    own follow-up reader, so what a test compares the production result
    against is never the production result restated.
    """
    ids = []
    for row in conn.execute(
        "SELECT payload FROM task_events "
        "WHERE task_id = ? AND kind = 'review_followup_recorded' ORDER BY id",
        (task_id,),
    ).fetchall():
        payload = json.loads(row["payload"]) if row["payload"] else {}
        followup = payload.get("followup_task_id")
        if isinstance(followup, str) and followup and followup not in ids:
            ids.append(followup)
    return ids


def _receipts(conn, task_id):
    """This card's resolution receipts as ``(followup_task_id, run_id)``."""
    return [
        (json.loads(row["payload"])["followup_task_id"], row["run_id"])
        for row in conn.execute(
            "SELECT payload, run_id FROM task_events "
            "WHERE task_id = ? AND kind = 'review_followup_resolved' ORDER BY id",
            (task_id,),
        ).fetchall()
    ]


def _park_for_review(conn, task_id, *, reviewer="reviewer"):
    """Park the card on the review lane and claim it as the reviewer."""
    task = kb.get_task(conn, task_id)
    assert task is not None
    assert kb.request_review(
        conn, task_id, summary="candidate ready", reviewer=reviewer,
        expected_run_id=task.current_run_id,
    ) is True
    claim = kb.claim_review_task(conn, task_id)
    assert claim is not None
    return claim.current_run_id


def _returned_once(conn, *, title="ship the widget", assignee="worker"):
    """A card one reviewer verdict returned, plus the item that recorded."""
    tid = kb.create_task(conn, title=title, assignee=assignee)
    assert kb.claim_task(conn, tid) is not None
    review_run = _park_for_review(conn, tid)
    ok, who = kb.request_changes(
        conn, tid, reason="fix the boundary", expected_run_id=review_run,
    )
    assert (ok, who) == (True, assignee)
    followups = _vouched_followups(conn, tid)
    assert len(followups) == 1, followups
    item = kb.get_task(conn, followups[0])
    assert item is not None and item.status == "triage"
    return tid, followups[0]


def _reviewer_approves(conn, task_id):
    """The reviewer's own clean verdict on the reworked candidate."""
    assert kb.claim_task(conn, task_id) is not None
    review_run = _park_for_review(conn, task_id)
    verdict = kb.submit_review_findings(
        conn, task_id, findings=[], candidate_digest="cand-clean",
        expected_run_id=review_run,
    )
    assert verdict["outcome"] == "passed", verdict
    assert kb.get_task(conn, task_id).status == "done"


def _assert_resolved_with_receipt(conn, task_id, item):
    """``item`` is done, and its receipt names the run that approved the card."""
    approving_run = kb.latest_run(conn, task_id)
    assert approving_run is not None
    assert approving_run.outcome == "completed"

    resolved = kb.get_task(conn, item)
    assert resolved is not None
    assert resolved.status == "done"
    assert resolved.completed_at is not None
    assert resolved.result, "the resolved item carries no summary at all"
    assert str(approving_run.id) in resolved.result
    assert task_id in resolved.result

    assert _receipts(conn, task_id) == [(item, approving_run.id)]


def test_a_returned_card_that_is_approved_resolves_its_rework_item(kanban_home):
    """The reviewer's clean verdict closes the item its own verdict opened."""
    with kb.connect() as conn:
        tid, item = _returned_once(conn)
        _reviewer_approves(conn, tid)
        _assert_resolved_with_receipt(conn, tid, item)


def test_a_person_approving_the_parked_card_resolves_it_the_same_way(kanban_home):
    """The other approval shape: a card parked in review, approved by hand.

    No reviewer run ever claims this card, so the completion takes the parked
    approval path. The rework it recorded is resolved exactly as it is for the
    reviewer's own verdict -- one approval contract, not two.
    """
    with kb.connect() as conn:
        tid, item = _returned_once(conn)

        assert kb.request_review(
            conn, tid, summary="candidate ready", reviewer="reviewer",
        ) is True
        parked = kb.get_task(conn, tid)
        assert (parked.status, parked.current_run_id) == ("review", None)

        assert kb.complete_task(conn, tid, summary="approved by hand") is True
        assert kb.get_task(conn, tid).status == "done"

        _assert_resolved_with_receipt(conn, tid, item)


def test_an_item_made_by_hand_under_the_same_card_stays_open(kanban_home):
    """Only what the KERNEL vouched for is resolved.

    The hand-made card below is indistinguishable from the kernel's own item
    by every property the board shows -- same parent, same owner, same title
    shape, same column. The one thing it does not have is a
    ``review_followup_recorded`` event vouching for it, and that alone decides.
    """
    with kb.connect() as conn:
        tid, item = _returned_once(conn)
        by_hand = kb.create_task(
            conn, title="Rework: ship the widget", assignee="worker",
            triage=True, parents=[tid],
        )
        assert by_hand != item
        assert _vouched_followups(conn, tid) == [item]

        _reviewer_approves(conn, tid)

        untouched = kb.get_task(conn, by_hand)
        assert untouched is not None
        assert untouched.status == "triage"
        assert untouched.result is None
        assert untouched.completed_at is None
        assert [followup for followup, _run in _receipts(conn, tid)] == [item]


def test_a_second_approval_records_nothing_new(kanban_home):
    """The resolution happens once. A second approval writes nothing at all."""
    with kb.connect() as conn:
        tid, item = _returned_once(conn)
        _reviewer_approves(conn, tid)

        first = _receipts(conn, tid)
        assert len(first) == 1
        resolved = kb.get_task(conn, item)
        assert resolved.status == "done"
        card_revision = kb.task_event_revision(conn, tid)
        item_revision = kb.task_event_revision(conn, item)

        assert kb.complete_task(conn, tid, summary="approved again") is False

        assert _receipts(conn, tid) == first
        assert kb.task_event_revision(conn, tid) == card_revision
        assert kb.task_event_revision(conn, item) == item_revision
        again = kb.get_task(conn, item)
        assert (again.status, again.result, again.completed_at) == (
            resolved.status, resolved.result, resolved.completed_at,
        )


def test_a_worker_pinned_to_its_board_resolves_and_still_reaches_the_root(
    tmp_path, monkeypatch,
):
    """The production proof, read under the environment a worker really gets.

    The dispatcher pins a worker to the board its task lives on
    (``HERMES_KANBAN_DB`` + ``HERMES_KANBAN_BOARD``), names the task
    (``HERMES_KANBAN_TASK``), and points ``HERMES_HOME`` at the assignee's
    profile home UNDER the shared root. Kanban paths must still resolve to the
    root from there: the register that Gate A is read from on every write lives
    at ``<root>/kanban``, above the profile home, so a resolution that followed
    the profile home instead would refuse the approval's own transaction.
    """
    root = tmp_path / "hermes_root"
    profile_home = root / "profiles" / "worker"
    profile_home.mkdir(parents=True)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(root))
    kb._INITIALIZED_PATHS.clear()
    kb.init_db()
    kb.create_board("worker-board")
    board_db = kb.board_dir("worker-board") / "kanban.db"

    with kb.connect(board="worker-board") as conn:
        tid = kb.create_task(conn, title="ship the widget", assignee="worker")

    # Exactly what ``_default_spawn`` injects into the worker subprocess.
    monkeypatch.setenv("HERMES_HOME", str(profile_home))
    monkeypatch.setenv("HERMES_KANBAN_DB", str(board_db))
    monkeypatch.setenv("HERMES_KANBAN_BOARD", "worker-board")
    monkeypatch.setenv("HERMES_KANBAN_TASK", tid)

    with kb.connect() as conn:
        # The pin, not an argument, is what routed this connection.
        opened = conn.execute("PRAGMA database_list").fetchone()[2]
        assert os.path.realpath(opened) == os.path.realpath(board_db)

        assert kb.claim_task(conn, tid) is not None
        review_run = _park_for_review(conn, tid)
        ok, _who = kb.request_changes(
            conn, tid, reason="fix the boundary", expected_run_id=review_run,
        )
        assert ok is True
        vouched = _vouched_followups(conn, tid)
        assert len(vouched) == 1, vouched

        _reviewer_approves(conn, tid)
        _assert_resolved_with_receipt(conn, tid, vouched[0])

        # The independent count: how many of this card's children the board
        # itself shows finished, taken straight from the links table without
        # passing through anything the resolution wrote or read.
        finished_children = conn.execute(
            "SELECT COUNT(*) AS n FROM tasks t "
            "JOIN task_links l ON l.child_id = t.id "
            "WHERE l.parent_id = ? AND t.status = 'done'",
            (tid,),
        ).fetchone()["n"]
        assert finished_children == len(vouched)
        assert len(_receipts(conn, tid)) == finished_children

    # Pinned to one board, the worker still reaches the root: the other board,
    # the board list, and the register the approval's own Gate A read used.
    assert kb.kanban_home() == root
    assert kb.register_db_path() == root / "kanban" / "board_register.db"
    entry = kb.get_register_entry("worker-board")
    assert entry is not None
    assert entry.lifecycle is kb.BoardLifecycle.LIVE
    assert {board["slug"] for board in kb.list_boards()} == {
        "default", "worker-board",
    }
    with kb.connect(db_path=root / "kanban.db") as other_board:
        assert kb.list_tasks(other_board) == []
