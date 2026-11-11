"""Which cards are attempts of the same work (build P1 of the plan of 31 October): each card's work item key and
whether it is the item's current attempt, and the given cards grouped into items. Real boards: the review return
runs through the delivery step, the owner's replace through _replace, the owner's replace, split and merge through
apply_owner_project_plan, a review verdict through request_changes and the event cleanup through the real hermes
kanban gc command. The records written here by hand are the ones the plan names: a delivery_updated event naming an
update card, _continuation's review_followup_recorded event for a chain or a cycle that the delivery step would need
many returns to build, and the owner's replacement record for a cycle no plan can make; and, where a test is about
the current-attempt rule alone, created times and statuses, and where it is about the cleanup, the age of events."""

from __future__ import annotations

import os
import re
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli.kanban_work_items import work_item_keys, work_items
from tests.hermes_cli.test_kanban_delivery import (  # noqa: F401  (board is a fixture)
    IMPLEMENTER, _approve, _git, board,
)
from tests.hermes_cli.test_kanban_delivery_auto_merge import armed  # noqa: F401  (armed is a fixture)
from tests.hermes_cli.test_kanban_delivery_owner_replace import _replace
from tests.hermes_cli.test_kanban_delivery_publish_step import _tick
from tests.hermes_cli.test_kanban_delivery_review_cards import _raw, world  # noqa: F401  (world is a fixture)
from tests.hermes_cli.test_kanban_delivery_review_return import CONTINUED, _changes_requested
from tests.hermes_cli.test_kanban_review_followup_resolution import (  # noqa: F401  (kanban_home is a fixture)
    _returned_once, _reviewer_approves, kanban_home,
)
from tests.hermes_cli.test_owner_plan_add_after_replace import (  # noqa: F401  (project is a fixture)
    _apply, _spec, _task, project,
)

KEY = re.compile(r"wi_[0-9a-f]{16}")
ROOT = Path(__file__).resolve().parents[2]


def _card(conn, title="build feature", **fields):
    return kb.create_task(conn, title=title, assignee="worker", **fields)


def _continued_by(conn, card, continuation):
    """``card`` records ``continuation`` as _continuation does: in no run, beside no review verdict."""
    with kb.write_txn(conn):
        kb._append_event(conn, card, "review_followup_recorded", {
            "followup_task_id": continuation, "implementer": "worker", "identity": f"{card} continuation"})


def _created(conn, minutes):
    """Each card's created time, ``minutes[card]`` minutes after one fixed moment."""
    with kb.write_txn(conn):
        for card, minute in minutes.items():
            conn.execute("UPDATE tasks SET created_at = ? WHERE id = ?", (1_790_000_000 + 60 * minute, card))


def _items(found):
    """Each card's work item as (key, current)."""
    return {card: (item["key"], item["current"]) for card, item in found.items()}


def test_a_card_retried_twice_is_one_item_whose_current_attempt_is_that_card(kanban_home):
    """Retries are runs of one card: three runs, the key the card had before them, and the card is current."""
    with kb.connect_closing() as conn:
        card = _card(conn)
        (key, _), = _items(work_item_keys(conn, [card])).values()
        for _ in range(2):
            assert kb.claim_task(conn, card) is not None
            assert kb.reclaim_task(conn, card, reason="try again") is True
        assert kb.claim_task(conn, card) is not None

        assert conn.execute("SELECT COUNT(*) FROM task_runs WHERE task_id = ?", (card,)).fetchone()[0] == 3
        assert work_item_keys(conn, [card]) == {card: {"key": key, "current": True}}
        assert work_items(conn, [card]) == {key: [{"task_id": card, "current": True}]}


def test_a_rework_after_a_review_return_shares_its_source_key_and_is_current_until_both_are_done(armed):
    """The delivery step's review return makes the continuation: it is the source's work item, and the current
    attempt while ready, running and in review. Once both cards are done, the newer one, the continuation, is."""
    _, root, _, _ = armed
    db = kb.kanban_db_path()
    source, _, _ = _changes_requested(armed)
    _raw(db, "UPDATE tasks SET created_at = created_at - 60 WHERE id = ?", source)  # a minute before its rework
    with kb.connect_closing() as conn:
        (key, _), = _items(work_item_keys(conn, [source])).values()
    _tick(kb)
    (rework,) = (card["id"] for card in _raw(db, CONTINUED, source))
    seen = {}
    with kb.connect_closing() as conn:
        seen[kb.get_task(conn, rework).status] = work_item_keys(conn, [source, rework])
    (root / "profiles" / IMPLEMENTER).mkdir(parents=True)
    _tick(kb)
    (workspace,) = (Path(card["workspace_path"]) for card in _raw(db, CONTINUED, source))
    (workspace / "src" / "impl" / "feature.py").write_text("ok = 3\n", encoding="utf-8")
    _git(workspace, "commit", "-qam", "fix: validate the input")
    with kb.connect_closing() as conn:
        seen[kb.get_task(conn, rework).status] = work_item_keys(conn, [source, rework])
        assert kb.complete_task(conn, rework, summary="validated the input",
                                expected_run_id=kb.get_task(conn, rework).current_run_id) is True
        seen[kb.get_task(conn, rework).status] = work_item_keys(conn, [source, rework])
        assert _approve(kb, conn, rework) is True
        seen[kb.get_task(conn, rework).status] = work_item_keys(conn, [source, rework])
        assert kb.get_task(conn, source).status == "done"

    assert seen == dict.fromkeys(("ready", "running", "review", "done"), {
        source: {"key": key, "current": False}, rework: {"key": key, "current": True}})


def test_an_owner_replacement_shares_the_replaced_cards_key_and_is_current(kanban_home):
    with kb.connect_closing() as conn:
        card = _card(conn, "write the content")
        (key, _), = _items(work_item_keys(conn, [card])).values()

    replacement = _replace(kb, card)

    with kb.connect_closing() as conn:
        assert work_item_keys(conn, [card, replacement]) == {card: {"key": key, "current": False},
                                                             replacement: {"key": key, "current": True}}


def test_an_update_card_on_main_shares_its_source_key_and_a_clean_update_changes_no_key(kanban_home):
    """The update card is a task_links child of the source, as _continuation makes it; the link alone joins
    nothing, the source's delivery_updated event naming it does. A clean update's event names no card."""
    with kb.connect_closing() as conn:
        source = _card(conn)
        assert kb.complete_task(conn, source, summary="approved") is True
        update = _card(conn, parents=[source])
        linked = _items(work_item_keys(conn, [source, update]))
        with kb.write_txn(conn):
            kb._append_event(conn, source, "delivery_updated", {
                "delivery_id": 1, "head": "a" * 40, "observed_main": "b" * 40, "update_task_id": update,
                "update_head": "c" * 40, "update_branch": f"delivery-update/{source}"})
        updated = work_item_keys(conn, [source, update])
        with kb.write_txn(conn):
            kb._append_event(conn, source, "delivery_updated", {
                "delivery_id": 2, "head": "a" * 40, "observed_main": "d" * 40, "update_head": "e" * 40,
                "update_branch": f"delivery/{source}", "clean": True})

        assert linked[source][0] != linked[update][0]
        assert updated == {source: {"key": linked[source][0], "current": False},
                           update: {"key": linked[source][0], "current": True}}
        assert work_item_keys(conn, [source, update]) == updated


def test_a_chain_of_two_reworks_and_an_owner_replacement_is_one_item_whose_last_card_is_current(kanban_home):
    with kb.connect_closing() as conn:
        source, rework, second = _card(conn), _card(conn), _card(conn)
        _continued_by(conn, source, rework)
        _continued_by(conn, rework, second)
        for card in (source, rework):
            assert kb.complete_task(conn, card, summary="returned") is True
        (key, _), = _items(work_item_keys(conn, [source])).values()

    replacement = _replace(kb, second)

    with kb.connect_closing() as conn:
        _created(conn, {source: 0, rework: 1, second: 2, replacement: 3})
        assert work_item_keys(conn, [source, rework, second, replacement]) == {
            source: {"key": key, "current": False}, rework: {"key": key, "current": False},
            second: {"key": key, "current": False}, replacement: {"key": key, "current": True}}


def test_a_split_and_a_merge_start_new_items_and_the_originals_keep_their_own(project):
    """The owner changed the shape of the work: each part of the split and the merged card is a new item, and
    each archived original stays the one card, so the current attempt, of its own. Each merged card's own record
    names the merge, so the merged card keeps its key when one of them is deleted from the board."""
    content, content_ref = _task(project, "Write the content")
    home, home_ref = _task(project, "Write the home page")
    about, about_ref = _task(project, "Write the about page")
    before = _items(work_item_keys(project.conn, [content, home, about]))
    split = {
        "action": "split", "reason": "The content is too broad to verify safely.", "target": content_ref,
        "replacements": [{**_spec("Draft the documents"), "parents": []},
                         {**_spec("Translate the documents"), "parents": []}],
    }
    merge = {
        "action": "merge", "reason": "One coherent set of pages is easier to verify.",
        "targets": [home_ref, about_ref], "replacement": _spec("Write the pages"),
    }

    draft, translation, pages = _apply(project, [split, merge])

    after = _items(work_item_keys(project.conn, [content, home, about, draft, translation, pages]))
    assert [kb.get_task(project.conn, card).status for card in (content, home, about)] == ["archived"] * 3
    assert {card: after[card] for card in (content, home, about)} == before
    assert len({key for key, _ in after.values()}) == 6
    assert all(current for _, current in after.values())
    assert kb.delete_archived_task(project.conn, home) is True
    assert _items(work_item_keys(project.conn, [about, pages])) == {about: after[about], pages: after[pages]}


def test_a_chain_of_33_owner_replacements_is_one_item_whose_last_card_is_current(project):
    """Each replacement is its own owner plan, with a fresh idempotency key and its target's revision of the moment:
    33 links, more than the 32-link guard. Every card keeps the key the first card had alone."""
    cards = [_task(project, "Write the content")[0]]
    (key, _), = _items(work_item_keys(project.conn, cards)).values()
    for plan in range(33):
        target = {"task_id": cards[-1], "expected_status": kb.get_task(project.conn, cards[-1]).status,
                  "expected_revision": kb.owner_plan_task_revision(project.conn, cards[-1])}
        result = kb.apply_owner_project_plan(
            project.conn, project_id=project.id, anchor_task_id=project.anchor_id, changes=[{
                "action": "replace", "reason": "Rescope the content.", "target": target,
                "replacement": _spec("Rewrite the content")}],
            actor="owner", profile="owner", idempotency_key=f"website-plan-{plan}", request_digest="d" * 64,
            trigger="owner_request", plan_summary="Rework the website content.",
            current_milestone="Launch the website", later_milestones=[])
        assert result["applied"] is True
        cards.extend(result["created_task_ids"])

    assert [kb.get_task(project.conn, card).status for card in cards] == ["archived"] * 33 + ["scheduled"]
    assert _items(work_item_keys(project.conn, cards)) == {card: (key, card == cards[-1]) for card in cards}


def test_a_cycle_written_by_hand_ends_and_its_root_is_the_oldest_card(kanban_home):
    """Three cards that each record the next as their continuation, the last the first: one item with the
    oldest card's own key. The oldest card has the highest id, so the root is not the lowest id."""
    with kb.connect_closing() as conn:
        oldest, middle, newest = sorted((_card(conn), _card(conn), _card(conn)), reverse=True)
        _created(conn, {oldest: 0, middle: 1, newest: 2})
        own = _items(work_item_keys(conn, [oldest, middle, newest]))
        _continued_by(conn, oldest, middle)
        _continued_by(conn, middle, newest)
        _continued_by(conn, newest, oldest)

        found = _items(work_item_keys(conn, [oldest, middle, newest]))

    assert len({key for key, _ in own.values()}) == 3
    assert found == {oldest: (own[oldest][0], False), middle: (own[oldest][0], False),
                     newest: (own[oldest][0], True)}


def test_a_replacement_cycle_longer_than_the_link_guard_is_one_item_whose_root_is_its_oldest_card(project):
    """34 cards of a Project, each recording the next as the one card the owner replaced it with and the last the
    first: 34 links, more than the 32-link guard. One item with the key the oldest card had alone, whichever card a
    walk starts from. The oldest card sits mid-cycle with neither the lowest nor the highest id, and the newest,
    just before it, is current: every card is still active."""
    cycle = sorted((_task(project, f"Write page {page}")[0] for page in range(34)), reverse=True)
    _created(project.conn, {card: (index - 17) % 34 for index, card in enumerate(cycle)})
    own = _items(work_item_keys(project.conn, cycle))
    with kb.write_txn(project.conn):
        for card, replacement in zip(cycle, cycle[1:] + cycle[:1]):
            kb._append_event(project.conn, card, "owner_project_plan_change", {
                "action": "replace", "reason": "Rescope the page.", "replacement_task_id": replacement})

    found = _items(work_item_keys(project.conn, cycle))

    oldest, newest = cycle[17], cycle[16]
    assert len({key for key, _ in own.values()}) == 34
    assert found == {card: (own[oldest][0], card == newest) for card in cycle}


def test_a_card_written_by_hand_as_the_continuation_of_two_cards_starts_its_own_item(kanban_home):
    """No writer records this; like the owner's merge, the card continues more than one card, so it is neither
    one's attempt, and the two stay apart."""
    with kb.connect_closing() as conn:
        first, second, both = _card(conn), _card(conn), _card(conn)
        own = _items(work_item_keys(conn, [first, second, both]))
        _continued_by(conn, first, both)
        _continued_by(conn, second, both)

        assert _items(work_item_keys(conn, [first, second, both])) == own
    assert len({key for key, _ in own.values()}) == 3


def test_a_plain_dependency_child_is_separate_work(kanban_home):
    with kb.connect_closing() as conn:
        parent = _card(conn, "design the schema")
        child = _card(conn, "build on the schema", parents=[parent])

        found = _items(work_item_keys(conn, [parent, child]))

    assert found[parent][0] != found[child][0]
    assert (found[parent][1], found[child][1]) == (True, True)


def test_the_rework_item_a_review_verdict_records_is_separate_work(kanban_home):
    """request_changes records its triage item with _continuation's event kind, in the run of its
    changes_requested verdict: a record of the verdict, not an attempt. The reviewed card goes back to the
    queue, a retry of that one card."""
    with kb.connect_closing() as conn:
        card, item = _returned_once(conn)

        found = _items(work_item_keys(conn, [card, item]))

    assert found[card][0] != found[item][0]
    assert (found[card][1], found[item][1]) == (True, True)


def test_the_key_reveals_no_card_id_and_is_stable_across_reads_restarts_and_a_reading_connection(kanban_home):
    """The same keys from a second read, from a board opened again as by a new process, and from the caller's
    own connection that may not write."""
    with kb.connect_closing() as conn:
        source, rework, other = _card(conn), _card(conn), _card(conn, "unrelated work")
        _continued_by(conn, source, rework)
        first = work_item_keys(conn, [source, rework, other])
        assert work_item_keys(conn, [source, rework, other]) == first
    kb._INITIALIZED_PATHS.clear()
    with kb.connect_closing() as conn:
        again = work_item_keys(conn, [other, rework, source])
    reader = sqlite3.connect(str(kb.kanban_db_path()))
    try:
        reader.execute("PRAGMA query_only = ON")
        read_only = work_item_keys(reader, [source, rework, other])
    finally:
        reader.close()

    assert again == read_only == first
    keys = {item["key"] for item in first.values()}
    assert len(keys) == 2 and first[source]["key"] == first[rework]["key"]
    for key in keys:
        assert KEY.fullmatch(key)
        assert not any(card in key or card.removeprefix("t_") in key for card in (source, rework, other))


def test_the_gc_command_keeps_the_keys_and_current_cards_of_an_owner_replacement_and_review_follow_ups(
        kanban_home, tmp_path, monkeypatch):
    """The real hermes kanban gc, with its default options, once every event is 31 days old: a done source that
    recorded its rework as _continuation does, the rework the owner replaced, and a done card a review verdict
    returned once, with the item that verdict recorded, done too. The gc removes the old events of the done and
    archived cards but their lineage records, and every card keeps its key and its current flag."""
    with kb.connect_closing() as conn:
        source, rework = _card(conn), _card(conn)
        _continued_by(conn, source, rework)
        assert kb.complete_task(conn, source, summary="returned") is True
        reviewed, verdict_item = _returned_once(conn)
        _reviewer_approves(conn, reviewed)
    replacement = _replace(kb, rework)
    cards = [source, rework, replacement, reviewed, verdict_item]
    with kb.connect_closing() as conn:
        with kb.write_txn(conn):
            conn.execute("UPDATE task_events SET created_at = created_at - 31 * 24 * 3600")
        before = work_item_keys(conn, cards)
    assert [before[card]["current"] for card in cards] == [False, False, True, True, True]
    assert len({before[card]["key"] for card in (source, rework, replacement)}) == 1
    assert len({item["key"] for item in before.values()}) == 3
    monkeypatch.setenv("HOME", str(tmp_path))

    gc = subprocess.run([sys.executable, "-m", "hermes_cli.main", "kanban", "gc"], cwd=ROOT, env=dict(os.environ),
                        stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=120)

    assert gc.returncode == 0, (gc.stdout, gc.stderr)
    removed = re.search(r"GC complete: 0 workspace\(s\), (\d+) event row\(s\), 0 log file\(s\) removed", gc.stdout)
    assert removed and int(removed.group(1)) > 0, (gc.stdout, gc.stderr)
    with kb.connect_closing() as conn:
        assert work_item_keys(conn, cards) == before
        kept = {card: {kind for (kind,) in conn.execute("SELECT kind FROM task_events WHERE task_id = ?", (card,))}
                for card in (source, rework, reviewed, verdict_item)}
    assert kept == {source: {"review_followup_recorded"}, rework: {"owner_project_plan_change"},
                    reviewed: {"review_followup_recorded"}, verdict_item: set()}


@pytest.mark.parametrize("purged", [0, 1], ids=["root", "middle"])
def test_the_purge_command_leaves_each_surviving_card_in_one_item_and_each_item_with_one_current_card(
        project, tmp_path, monkeypatch, purged):
    """The real hermes kanban archive --rm on the first or the middle of three cards that real owner plans replaced
    one by one, the last still active. The owner's purge ends the item's life, so a surviving card may start a new
    item with a new key; but each surviving card is in exactly one item, each item has exactly one current card,
    and no read fails, not even one that names the purged card."""
    cards = [_task(project, "Write the content")[0]]
    (key, _), = _items(work_item_keys(project.conn, cards)).values()
    for plan in range(2):
        target = {"task_id": cards[-1], "expected_status": kb.get_task(project.conn, cards[-1]).status,
                  "expected_revision": kb.owner_plan_task_revision(project.conn, cards[-1])}
        result = kb.apply_owner_project_plan(
            project.conn, project_id=project.id, anchor_task_id=project.anchor_id, changes=[{
                "action": "replace", "reason": "Rescope the content.", "target": target,
                "replacement": _spec("Rewrite the content")}],
            actor="owner", profile="owner", idempotency_key=f"website-plan-{plan}", request_digest="d" * 64,
            trigger="owner_request", plan_summary="Rework the website content.",
            current_milestone="Launch the website", later_milestones=[])
        assert result["applied"] is True
        cards.extend(result["created_task_ids"])
    assert [kb.get_task(project.conn, card).status for card in cards] == ["archived", "archived", "scheduled"]
    assert _items(work_item_keys(project.conn, cards)) == {card: (key, card == cards[-1]) for card in cards}
    gone = cards[purged]
    surviving = [card for card in cards if card != gone]
    monkeypatch.setenv("HOME", str(tmp_path))

    purge = subprocess.run([sys.executable, "-m", "hermes_cli.main", "kanban", "archive", "--rm", gone], cwd=ROOT,
                           env=dict(os.environ), stdin=subprocess.DEVNULL, capture_output=True, text=True,
                           timeout=120)

    assert purge.returncode == 0, (purge.stdout, purge.stderr)
    assert kb.get_task(project.conn, gone) is None
    found = work_item_keys(project.conn, surviving)
    grouped = work_items(project.conn, surviving)
    assert sorted(entry["task_id"] for entries in grouped.values() for entry in entries) == sorted(surviving)
    assert {entry["task_id"]: {"key": item, "current": entry["current"]}
            for item, entries in grouped.items() for entry in entries} == found
    assert {item: [entry["current"] for entry in entries].count(True)
            for item, entries in grouped.items()} == dict.fromkeys(grouped, 1)
    assert work_item_keys(project.conn, cards) == {**found, gone: None}
    assert work_items(project.conn, cards) == grouped


def test_a_board_with_no_linking_records_makes_every_card_its_own_item(kanban_home):
    """Cards with ordinary links, runs and every outcome, but no event that names a continuation: each card is
    its own item, alone and current. An id not on the board has no item."""
    with kb.connect_closing() as conn:
        first = _card(conn, "design the schema")
        second = _card(conn, "build on the schema", parents=[first])
        third = _card(conn, "write the docs")
        assert kb.claim_task(conn, third) is not None
        assert kb.complete_task(conn, first, summary="designed") is True
        assert kb.archive_task(conn, third) is True
        cards = [first, second, third]
        assert conn.execute("SELECT COUNT(*) FROM task_events WHERE kind IN ('review_followup_recorded', "
                            "'delivery_updated', 'owner_project_plan_change')").fetchone()[0] == 0

        found = work_item_keys(conn, [*cards, "t_00000000"])
        grouped = work_items(conn, [*cards, "t_00000000"])

    assert found.pop("t_00000000") is None
    assert len({item["key"] for item in found.values()}) == 3
    assert grouped == {found[card]["key"]: [{"task_id": card, "current": True}] for card in cards}


@pytest.mark.parametrize(("statuses", "current"), [
    (("done", "review", "done"), 1),
    (("blocked", "done", "archived"), 0),
    (("done", "done", "archived"), 1),
    (("archived", "archived", "archived"), 2),
])
def test_the_current_attempt_is_the_newest_active_card_else_the_newest_done_else_the_newest_archived(
        kanban_home, statuses, current):
    """Review and blocked are active, as the owner may need to act on them; an active card is current even
    when a later attempt finished."""
    with kb.connect_closing() as conn:
        cards = [_card(conn) for _ in statuses]
        _continued_by(conn, cards[0], cards[1])
        _continued_by(conn, cards[1], cards[2])
        _created(conn, {card: minute for minute, card in enumerate(cards)})
        with kb.write_txn(conn):
            for card, status in zip(cards, statuses):
                conn.execute("UPDATE tasks SET status = ? WHERE id = ?", (status, card))

        found = _items(work_item_keys(conn, cards))

    assert len({key for key, _ in found.values()}) == 1
    assert [found[card][1] for card in cards] == [index == current for index in range(len(cards))]


def test_the_current_attempt_is_shown_as_the_item_and_its_earlier_attempts_follow_oldest_first(kanban_home):
    """A card continued twice, a branch that stays one item, and one of those continued again: the newest card,
    the one still active, is the item; the others follow as earlier attempts by created time, then id. Asked
    without its current attempt, the item lists only earlier attempts."""
    with kb.connect_closing() as conn:
        source, first, second, third = _card(conn), _card(conn), _card(conn), _card(conn)
        _continued_by(conn, source, first)
        _continued_by(conn, source, second)
        _continued_by(conn, first, third)
        for card in (source, first, second):
            assert kb.complete_task(conn, card, summary="returned") is True
        _created(conn, {source: 0, first: 1, second: 1, third: 2})
        (key, _), = _items(work_item_keys(conn, [source])).values()
        earlier = [source, *sorted((first, second))]

        assert work_items(conn, [second, third, source, first]) == {key: [
            {"task_id": third, "current": True}, *({"task_id": card, "current": False} for card in earlier)]}
        assert work_items(conn, [first, second, source]) == {key: [
            {"task_id": card, "current": False} for card in earlier]}
