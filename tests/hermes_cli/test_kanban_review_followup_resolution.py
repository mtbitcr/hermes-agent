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

A card an owner plan change replaced vouches the same way: approving its
replacement -- through any chain of replacements -- also resolves the items the
replaced cards recorded. A split card has no single replacement, so no one part
of it resolves its items.

Archiving a card is the other way a round-trip ends: the rework its verdicts
recorded will never happen. The archive closes the items that are still open,
in the archive's own transaction, with one ``review_followup_archived``
receipt on the card per item -- under the same boundary: only what the card
itself recorded, and only what is still open.
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


def test_an_item_a_worker_is_running_is_resolved_and_its_run_closed(kanban_home):
    """An item a worker holds when the approval lands is resolved all the same.

    The owner took this item out of triage and a worker claimed it, so it
    holds a live run. The approval still accepts the rework the item tracks,
    so the item closes in the approving transaction like any other -- and the
    run it held closes with it, the way ``archive_task`` closes a run it takes
    a card out from under, rather than being left open or stranded.
    """
    with kb.connect() as conn:
        tid, item = _returned_once(conn)
        assert kb.unlink_tasks(conn, tid, item) is True
        promoted = kb.cas_transition_task(
            conn, item, expected_status="triage",
            expected_revision=kb.task_event_revision(conn, item),
            to_status="ready",
        )
        assert promoted["moved"] is True, promoted
        held = kb.claim_task(conn, item)
        assert held is not None and held.status == "running"
        worker_run = held.current_run_id
        assert worker_run is not None
        kb._set_worker_pid(conn, item, os.getpid())
        live = kb.get_task(conn, item)
        assert None not in (live.claim_lock, live.claim_expires, live.worker_pid)

        _reviewer_approves(conn, tid)
        _assert_resolved_with_receipt(conn, tid, item)
        approving_run = kb.latest_run(conn, tid).id

        claim = conn.execute(
            "SELECT current_run_id, claim_lock, claim_expires, worker_pid "
            "FROM tasks WHERE id = ?",
            (item,),
        ).fetchone()
        assert tuple(claim) == (None, None, None, None)

        run_sql = (
            "SELECT status, outcome, summary, ended_at, "
            "claim_lock, claim_expires, worker_pid "
            "FROM task_runs WHERE id = ? AND task_id = ?"
        )
        run_key = (worker_run, item)
        closed = tuple(conn.execute(run_sql, run_key).fetchone())
        status, outcome, summary, ended_at, *run_claim = closed
        assert (status, outcome) == ("reclaimed", "reclaimed")
        assert ended_at is not None
        assert run_claim == [None, None, None]
        assert tid in summary and f"run {approving_run}" in summary

        # The worker that held the item cannot finish it through its old run.
        resolved = kb.get_task(conn, item)
        item_revision = kb.task_event_revision(conn, item)
        assert kb.complete_task(
            conn, item, summary="finished late", expected_run_id=worker_run,
        ) is False
        late = kb.get_task(conn, item)
        assert (late.status, late.result, late.completed_at) == (
            resolved.status, resolved.result, resolved.completed_at,
        )
        assert late.current_run_id is None
        assert tuple(conn.execute(run_sql, run_key).fetchone()) == closed
        assert kb.task_event_revision(conn, item) == item_revision

        # And a second approval records nothing new.
        card_revision = kb.task_event_revision(conn, tid)
        assert kb.complete_task(conn, tid, summary="approved again") is False
        assert _receipts(conn, tid) == [(item, approving_run)]
        assert kb.task_event_revision(conn, tid) == card_revision
        assert kb.task_event_revision(conn, item) == item_revision
        assert tuple(conn.execute(run_sql, run_key).fetchone()) == closed


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


def _archive_by_plan_change(conn, task_id, *, action, **replaced_by):
    """Archive a card the way an owner plan change replaces it.

    The plan applier archives the replaced card through the public
    compare-and-swap move and writes an ``owner_project_plan_change`` event on
    THAT card, naming what took its place: ``replacement_task_id`` for a
    replace or a merge, ``replacement_task_ids`` for a split. The payload here
    carries the same fields the applier writes.
    """
    task = kb.get_task(conn, task_id)
    assert task is not None
    moved = kb.cas_transition_task(
        conn, task_id, expected_status=task.status,
        expected_revision=kb.task_event_revision(conn, task_id),
        to_status="archived", event_kind="owner_project_plan_change",
        event_payload={
            "actor": "steward", "profile": "steward",
            "idempotency_key": f"plan-{task_id}", "request_digest": "digest",
            "action": action, "reason": "superseded", **replaced_by,
        },
    )
    assert moved["moved"] is True, moved
    assert kb.get_task(conn, task_id).status == "archived"


def _assert_all_resolved_with_receipts(conn, task_id, items):
    """Every item is done, each with ONE receipt on the approved card."""
    approving_run = kb.latest_run(conn, task_id)
    assert approving_run is not None
    assert approving_run.outcome == "completed"

    for item in items:
        resolved = kb.get_task(conn, item)
        assert resolved is not None
        assert resolved.status == "done"
        assert resolved.completed_at is not None
        assert str(approving_run.id) in (resolved.result or "")
        assert task_id in (resolved.result or "")

    assert sorted(_receipts(conn, task_id)) == sorted(
        (item, approving_run.id) for item in items
    )


def _assert_left_alone(conn, item, *, status="triage"):
    """``item`` is where it was: its status, no result, no completion time."""
    card = kb.get_task(conn, item)
    assert card is not None
    assert (card.status, card.result, card.completed_at) == (status, None, None)


def test_approving_a_replacement_resolves_the_item_of_the_card_it_replaced(
    kanban_home,
):
    """The replaced card's rework is accepted along with its replacement.

    The first card was returned once and then superseded by an owner plan
    change, so the item that verdict recorded would stay open for good:
    nothing ever approves an archived card. The replacement's approval is what
    accepts the work, so it resolves that item -- one receipt, on the
    replacement, naming the approving run.
    """
    with kb.connect() as conn:
        old, item = _returned_once(conn, title="ship the widget")
        new = kb.create_task(
            conn, title="ship the widget, rebuilt", assignee="worker",
        )
        _archive_by_plan_change(
            conn, old, action="replace", replacement_task_id=new,
        )
        _assert_left_alone(conn, item)

        _reviewer_approves(conn, new)

        _assert_resolved_with_receipt(conn, new, item)
        assert kb.get_task(conn, old).status == "archived"
        assert _receipts(conn, old) == []


def test_approving_the_end_of_a_replacement_chain_resolves_both_items(
    kanban_home,
):
    """A card replaced by a card that was itself replaced is still replaced.

    ``first`` was superseded by ``second``, which was superseded in turn by
    ``third``; each was returned once, so each left an item open. Approving
    ``third`` accepts the work of the whole chain: both items close, one
    receipt each, all on ``third``.
    """
    with kb.connect() as conn:
        first, first_item = _returned_once(conn, title="first attempt")
        second, second_item = _returned_once(conn, title="second attempt")
        third = kb.create_task(conn, title="third attempt", assignee="worker")
        _archive_by_plan_change(
            conn, first, action="replace", replacement_task_id=second,
        )
        _archive_by_plan_change(
            conn, second, action="replace", replacement_task_id=third,
        )
        _assert_left_alone(conn, first_item)
        _assert_left_alone(conn, second_item)

        _reviewer_approves(conn, third)

        _assert_all_resolved_with_receipts(conn, third, [first_item, second_item])
        assert _receipts(conn, first) == []
        assert _receipts(conn, second) == []


def test_approving_the_card_two_cards_were_merged_into_resolves_both_items(
    kanban_home,
):
    """A merge archives each source card naming the same replacement.

    Both sources were returned once and left an item open. The merged card is
    the replacement of each of them, so its approval resolves both items --
    each exactly once, though the card is reached from two directions.
    """
    with kb.connect() as conn:
        left, left_item = _returned_once(conn, title="left half")
        right, right_item = _returned_once(conn, title="right half")
        merged = kb.create_task(conn, title="both halves", assignee="worker")
        for source in (left, right):
            _archive_by_plan_change(
                conn, source, action="merge", replacement_task_id=merged,
            )
        _assert_left_alone(conn, left_item)
        _assert_left_alone(conn, right_item)

        _reviewer_approves(conn, merged)

        _assert_all_resolved_with_receipts(conn, merged, [left_item, right_item])


def test_approving_one_part_of_a_split_card_leaves_the_split_cards_item_open(
    kanban_home,
):
    """No single part of a split card replaces it, so its rework stays open.

    A split hands the card's work to SEVERAL parts, and approving one part
    accepts only that part of it: the item the split card recorded cannot be
    called accepted, and neither part may close it. A card the first part
    replaced outright is the control -- the very same approval does resolve
    its item, so the split card's item is left alone by the rule, not by
    accident.
    """
    with kb.connect() as conn:
        whole, whole_item = _returned_once(conn, title="the whole")
        outright, outright_item = _returned_once(conn, title="replaced outright")
        part = kb.create_task(conn, title="first part", assignee="worker")
        other_part = kb.create_task(conn, title="second part", assignee="worker")
        _archive_by_plan_change(
            conn, whole, action="split",
            replacement_task_ids=[part, other_part],
        )
        _archive_by_plan_change(
            conn, outright, action="replace", replacement_task_id=part,
        )

        _reviewer_approves(conn, part)

        _assert_left_alone(conn, whole_item)
        _assert_resolved_with_receipt(conn, part, outright_item)

        _reviewer_approves(conn, other_part)

        _assert_left_alone(conn, whole_item)
        assert _receipts(conn, other_part) == []
        assert kb.get_task(conn, whole).status == "archived"


def test_an_item_already_archived_stays_archived_when_its_card_is_replaced(
    kanban_home,
):
    """An archived item is a closed record: approval neither reopens nor re-closes it.

    Somebody archived the rework item by hand before the replacement came
    back. The approval resolves the items of the replaced cards that are still
    open -- the control, on the second source -- and leaves this one exactly
    as it is: archived, no result, no receipt, nothing appended to it.
    """
    with kb.connect() as conn:
        old, item = _returned_once(conn, title="ship the widget")
        sibling, sibling_item = _returned_once(conn, title="ship the gadget")
        new = kb.create_task(conn, title="ship both", assignee="worker")
        for source in (old, sibling):
            _archive_by_plan_change(
                conn, source, action="merge", replacement_task_id=new,
            )
        assert kb.archive_task(conn, item) is True
        item_revision = kb.task_event_revision(conn, item)

        _reviewer_approves(conn, new)

        _assert_left_alone(conn, item, status="archived")
        assert kb.task_event_revision(conn, item) == item_revision
        _assert_resolved_with_receipt(conn, new, sibling_item)


def test_approving_a_card_that_replaced_nothing_records_nothing_new(
    kanban_home,
):
    """With nothing replaced there is nothing to resolve, and nothing is written.

    A rework item is open elsewhere on the board, and its card WAS replaced --
    by a different card, not the one approved here. Approving a card that
    replaced nothing records no receipt anywhere, and leaves that item and
    every card around it exactly as it was.
    """
    with kb.connect() as conn:
        old, item = _returned_once(conn, title="ship the widget")
        elsewhere = kb.create_task(
            conn, title="ship the widget, rebuilt", assignee="worker",
        )
        _archive_by_plan_change(
            conn, old, action="replace", replacement_task_id=elsewhere,
        )
        lone = kb.create_task(conn, title="a card of its own", assignee="worker")
        around = (old, item, elsewhere)
        revisions = {card: kb.task_event_revision(conn, card) for card in around}

        _reviewer_approves(conn, lone)

        assert _receipts(conn, lone) == []
        resolved_anywhere = conn.execute(
            "SELECT COUNT(*) AS n FROM task_events "
            "WHERE kind = 'review_followup_resolved'"
        ).fetchone()["n"]
        assert resolved_anywhere == 0
        assert {
            card: kb.task_event_revision(conn, card) for card in around
        } == revisions
        _assert_left_alone(conn, item)


def test_an_item_of_an_unrelated_card_stays_open(kanban_home):
    """Only the cards the approved card replaced are resolved with it.

    Two other cards each left an item open: one was never replaced, the other
    was replaced by a card that is not the one approved. Neither is in the
    approved card's replacement history, so neither item moves -- while the
    item of the card it did replace (the control) closes in the same approval.
    """
    with kb.connect() as conn:
        old, item = _returned_once(conn, title="ship the widget")
        bystander, bystander_item = _returned_once(conn, title="a card of its own")
        rival, rival_item = _returned_once(conn, title="ship the gadget")
        new = kb.create_task(
            conn, title="ship the widget, rebuilt", assignee="worker",
        )
        rival_new = kb.create_task(
            conn, title="ship the gadget, rebuilt", assignee="worker",
        )
        _archive_by_plan_change(
            conn, old, action="replace", replacement_task_id=new,
        )
        _archive_by_plan_change(
            conn, rival, action="replace", replacement_task_id=rival_new,
        )

        _reviewer_approves(conn, new)

        _assert_left_alone(conn, bystander_item)
        _assert_left_alone(conn, rival_item)
        assert kb.get_task(conn, bystander).status == "ready"
        _assert_resolved_with_receipt(conn, new, item)


def _archive_receipts(conn):
    """Every ``review_followup_archived`` event on the board, as ``(card, item)``."""
    return [
        (row["task_id"], json.loads(row["payload"])["followup_task_id"])
        for row in conn.execute(
            "SELECT task_id, payload FROM task_events "
            "WHERE kind = 'review_followup_archived' ORDER BY id"
        ).fetchall()
    ]


def _event_kinds(conn, task_id):
    """The kinds of the events written on this card, oldest first."""
    return [
        row["kind"]
        for row in conn.execute(
            "SELECT kind FROM task_events WHERE task_id = ? ORDER BY id",
            (task_id,),
        ).fetchall()
    ]


def _standing(conn, task_id):
    """What the board shows of a card, plus how many events it has written."""
    card = kb.get_task(conn, task_id)
    assert card is not None
    return (
        card.status, card.result, card.completed_at,
        kb.task_event_revision(conn, task_id),
    )


def _assert_archived_with_its_card(conn, task_id, item):
    """``item`` is archived, and its result says its card was archived."""
    closed = kb.get_task(conn, item)
    assert closed is not None
    assert closed.status == "archived"
    assert closed.result, "the closed item carries no result at all"
    assert "archived" in closed.result
    assert task_id in closed.result


def test_archiving_a_card_archives_the_open_rework_item_it_recorded(kanban_home):
    """The rework a card recorded will never happen, so its item goes with it.

    Nothing has picked the item up when the card is archived. The archive
    closes it in the same transaction, with a result that says why, and the
    card carries ONE receipt for it: an event naming the item.
    """
    with kb.connect() as conn:
        tid, item = _returned_once(conn)
        _assert_left_alone(conn, item)

        assert kb.archive_task(conn, tid) is True

        assert kb.get_task(conn, tid).status == "archived"
        _assert_archived_with_its_card(conn, tid, item)
        assert _archive_receipts(conn) == [(tid, item)]


def test_an_item_already_done_or_archived_is_left_exactly_as_it_is(kanban_home):
    """Only an item that is still open closes with its card.

    The first card was approved, so the item it recorded is ``done``; the
    second card's item was archived by hand. Archiving either card must not
    reopen, re-close or re-stamp them: same status, same result, same
    completion time, nothing appended to the item and no receipt for it. The
    control is a third card whose item is still open -- the very same archive
    does close that one.
    """
    with kb.connect() as conn:
        approved, done_item = _returned_once(conn, title="ship the widget")
        _reviewer_approves(conn, approved)
        by_hand, hand_item = _returned_once(conn, title="ship the gadget")
        assert kb.archive_task(conn, hand_item) is True
        control, open_item = _returned_once(conn, title="ship the gizmo")
        assert kb.get_task(conn, done_item).status == "done"
        assert kb.get_task(conn, hand_item).status == "archived"
        done_before = _standing(conn, done_item)
        hand_before = _standing(conn, hand_item)

        for card in (approved, by_hand, control):
            assert kb.archive_task(conn, card) is True

        assert _standing(conn, done_item) == done_before
        assert _standing(conn, hand_item) == hand_before
        _assert_archived_with_its_card(conn, control, open_item)
        assert _archive_receipts(conn) == [(control, open_item)]


def test_an_item_a_worker_holds_is_archived_and_its_run_closed(kanban_home):
    """A held item closes with its card, and the run it held closes with it.

    The owner took this item out of triage and a worker claimed it, so it
    holds a live run. Archiving the card still closes the item, and the run
    ends as ``reclaimed`` -- the way ``archive_task`` ends a run it takes a
    card out from under -- rather than being left open or stranded.
    """
    with kb.connect() as conn:
        tid, item = _returned_once(conn)
        assert kb.unlink_tasks(conn, tid, item) is True
        promoted = kb.cas_transition_task(
            conn, item, expected_status="triage",
            expected_revision=kb.task_event_revision(conn, item),
            to_status="ready",
        )
        assert promoted["moved"] is True, promoted
        held = kb.claim_task(conn, item)
        assert held is not None and held.status == "running"
        worker_run = held.current_run_id
        assert worker_run is not None
        kb._set_worker_pid(conn, item, os.getpid())
        live = kb.get_task(conn, item)
        assert None not in (live.claim_lock, live.claim_expires, live.worker_pid)

        assert kb.archive_task(conn, tid) is True

        _assert_archived_with_its_card(conn, tid, item)
        assert _archive_receipts(conn) == [(tid, item)]
        claim = conn.execute(
            "SELECT current_run_id, claim_lock, claim_expires, worker_pid "
            "FROM tasks WHERE id = ?",
            (item,),
        ).fetchone()
        assert tuple(claim) == (None, None, None, None)

        closed = conn.execute(
            "SELECT status, outcome, summary, ended_at, "
            "claim_lock, claim_expires, worker_pid "
            "FROM task_runs WHERE id = ? AND task_id = ?",
            (worker_run, item),
        ).fetchone()
        status, outcome, summary, ended_at, *run_claim = tuple(closed)
        assert (status, outcome) == ("reclaimed", "reclaimed")
        assert ended_at is not None
        assert run_claim == [None, None, None]
        assert tid in summary


def test_archiving_a_card_that_recorded_no_item_records_nothing_new(kanban_home):
    """With no rework item recorded there is nothing to close, and nothing is written.

    The plain card recorded no item. Archiving it writes what it always has --
    the one ``archived`` event on the card -- and no receipt anywhere, and
    leaves the open item of a different card, and that card, exactly as they
    were. The control is that different card: it DID record an item, and its
    archive closes it.
    """
    with kb.connect() as conn:
        plain = kb.create_task(conn, title="a card of its own", assignee="worker")
        assert _vouched_followups(conn, plain) == []
        before = _event_kinds(conn, plain)
        control, item = _returned_once(conn, title="ship the widget")
        around = (control, item)
        standing = {card: _standing(conn, card) for card in around}

        assert kb.archive_task(conn, plain) is True

        assert _event_kinds(conn, plain) == [*before, "archived"]
        assert _archive_receipts(conn) == []
        assert {card: _standing(conn, card) for card in around} == standing

        assert kb.archive_task(conn, control) is True
        _assert_archived_with_its_card(conn, control, item)
        assert _archive_receipts(conn) == [(control, item)]
        assert _event_kinds(conn, plain) == [*before, "archived"]


def test_an_item_of_another_card_stays_open_when_a_card_is_archived(kanban_home):
    """Only what the archived card itself recorded closes, however alike the rest looks.

    The other card has the same title and owner, and the item IT recorded hangs
    under the archived card as well, so nothing the board shows tells the two
    items apart. A third card hangs under the archived card by hand, with the
    same title shape. The archived card recorded neither, so neither moves --
    while the item it did record (the control) closes in the same archive.
    """
    with kb.connect() as conn:
        tid, item = _returned_once(conn, title="ship the widget")
        other, other_item = _returned_once(conn, title="ship the widget")
        kb.link_tasks(conn, tid, other_item)
        by_hand = kb.create_task(
            conn, title="Rework: ship the widget", assignee="worker",
            triage=True, parents=[tid],
        )
        assert _vouched_followups(conn, tid) == [item]
        assert _vouched_followups(conn, other) == [other_item]
        around = (other, other_item, by_hand)
        standing = {card: _standing(conn, card) for card in around}

        assert kb.archive_task(conn, tid) is True

        assert {card: _standing(conn, card) for card in around} == standing
        _assert_left_alone(conn, other_item)
        _assert_left_alone(conn, by_hand)
        assert kb.get_task(conn, other).status == "ready"
        _assert_archived_with_its_card(conn, tid, item)
        assert _archive_receipts(conn) == [(tid, item)]
