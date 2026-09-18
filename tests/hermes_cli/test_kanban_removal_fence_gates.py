"""The operator's recorded GA-5 migration (the backfill), exercised end to
end: an explicit, recorded command an operator runs at release — one board
or all of them — and until a board has been through it, its ordinary
writes refuse with the structured refusal. Reads keep working throughout.
"""

from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

import pytest

_WORKTREE = Path(__file__).resolve().parents[2]
if str(_WORKTREE) not in sys.path:
    sys.path.insert(0, str(_WORKTREE))

from hermes_cli import kanban_db as kb
from tests.hermes_cli._kanban_fence_support import (
    archive_records,
    close_fence,
    cli,
    gate_row,
    has_gate_table,
    intent_rows,
    make_legacy_board,
    marker_row,
    ready_task,
    register_lineage,
    register_row,
    row_count,
    start_removal,
)


# ---------------------------------------------------------------------------
# Opening never refuses; a board being removed still opens for reading
# ---------------------------------------------------------------------------

def test_a_board_being_removed_still_opens_for_reading(fence_home):
    """A board mid-removal must still be readable — the operator has to be
    able to look at the board that is going away."""
    kb.create_board("still-open")
    # ``create_board`` no longer arms a gate on its own — this board needs
    # the real recorded migration before it can take the write below.
    kb.backfill_register_entry("still-open")
    conn = kb.connect(board="still-open")
    ready_task(conn, "visible")
    conn.close()
    close_fence("still-open")

    conn = kb.connect(board="still-open")
    try:
        assert len(kb.list_tasks(conn)) == 1
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# The recorded migration
# ---------------------------------------------------------------------------

def test_backfill_command_migrates_one_board_and_records_it(fence_home, tmp_path):
    db_path = make_legacy_board(tmp_path, "one-board")

    out = cli("boards backfill-fence one-board")

    assert "ok" in out
    assert "one-board" in out
    assert gate_row(db_path) == ("open", 1)
    assert register_row("one-board") == {
        "lifecycle": "live",
        "epoch": 1,
        "epoch_before": None,
        "gate_move": "settled",
    }
    assert marker_row("one-board") is True
    # "Recorded": a receipt in the removal archive, and no intent left over.
    assert archive_records("one-board") == 1
    assert intent_rows("one-board") == 0


def test_backfill_command_migrates_every_board(fence_home, tmp_path):
    first = make_legacy_board(tmp_path, "legacy-a")
    second = make_legacy_board(tmp_path, "legacy-b")
    kb.create_board("already-fenced")
    # ``create_board`` no longer arms a gate on its own — this board is
    # only "already fenced" once it carries a register entry.
    kb.backfill_register_entry("already-fenced")

    out = cli("boards backfill-fence --all --json")
    rows = {row["board"]: row for row in json.loads(out)}

    assert rows["legacy-a"]["ok"] is True
    assert rows["legacy-b"]["ok"] is True
    assert rows["already-fenced"]["message"] == "already fenced"
    assert gate_row(first) == ("open", 1)
    assert gate_row(second) == ("open", 1)


def test_backfill_needs_a_board_or_all(fence_home):
    out = cli("boards backfill-fence")
    assert "name a board" in out


def test_backfill_refuses_a_name_whose_marker_is_already_set(fence_home, tmp_path):
    """GA-5 condition 3: the backfill is one-time. A name the fence has
    already vouched for is not migrated again."""
    make_legacy_board(tmp_path, "once-only")
    assert kb.backfill_register_entry("once-only").success is True

    with kb.register_connect() as reg:
        reg.execute("BEGIN IMMEDIATE")
        reg.execute("DELETE FROM board_register WHERE board_name = ?", ("once-only",))
        reg.execute("COMMIT")

    second = kb.backfill_register_entry("once-only")
    assert second.success is False
    assert "marker" in second.message


def test_backfill_refuses_an_incomplete_store(fence_home, tmp_path):
    board_dir = kb.board_dir("half-a-board")
    board_dir.mkdir(parents=True, exist_ok=True)
    (board_dir / "kanban.db").write_bytes(b"")

    result = kb.backfill_register_entry("half-a-board")

    assert result.success is False
    assert register_row("half-a-board") is None
    assert marker_row("half-a-board") is False


def test_backfill_refuses_when_the_archive_already_holds_a_receipt(
    fence_home, tmp_path
):
    """GA-6a: a receipt for this name means the fence cannot rule out a
    prior life for it."""
    make_legacy_board(tmp_path, "receipted")
    assert kb.record_removal_archive_entry("receipted", "audit", "prior-life") is True

    result = kb.backfill_register_entry("receipted")

    assert result.success is False
    assert "archive" in result.message
    assert register_row("receipted") is None


def test_writes_are_admitted_before_the_migration_and_still_land_after(
    fence_home, tmp_path
):
    """An un-backfilled board has no Gate B and admits ordinary writes;
    the recorded migration arms Gate B (open, matching epoch) without
    taking that admission away."""
    db_path = make_legacy_board(tmp_path, "gate-me")

    conn = kb.connect(board="gate-me")
    try:
        assert kb.add_comment(conn, kb.list_tasks(conn)[0].id, "a", "b") > 0
    finally:
        conn.close()
    assert not has_gate_table(db_path)
    assert row_count(db_path, "task_comments") == 1

    cli("boards backfill-fence gate-me")

    kb._INITIALIZED_PATHS.clear()
    conn = kb.connect(board="gate-me")
    try:
        assert kb.add_comment(conn, kb.list_tasks(conn)[0].id, "a", "b") > 0
    finally:
        conn.close()
    assert has_gate_table(db_path)
    assert row_count(db_path, "task_comments") == 2


def test_the_cli_reports_a_refused_backfill_with_a_non_zero_exit(
    fence_home, tmp_path
):
    make_legacy_board(tmp_path, "will-refuse")
    kb.record_removal_archive_entry("will-refuse", "audit", "prior-life")

    out = cli("boards backfill-fence will-refuse")

    assert "REFUSED" in out
    assert register_row("will-refuse") is None


# ---------------------------------------------------------------------------
# The register and the marker are separate stores
# ---------------------------------------------------------------------------

def test_the_register_and_the_marker_live_in_different_files(fence_home):
    """The marker's whole job is to be readable when the entry it guards
    is gone, which it cannot be if it is a column on that entry."""
    kb.create_board("two-stores")
    kb.backfill_register_entry("two-stores")
    assert kb.register_db_path() != kb.archive_db_path()
    assert kb.register_db_path().exists()
    assert kb.archive_db_path().exists()

    kb.register_db_path().unlink()
    kb._REGISTER_INITIALIZED = False
    kb._REGISTER_INITIALIZED_PATHS.clear()

    assert marker_row("two-stores") is True
    assert kb.ever_existed_marker_set("two-stores") is True


# ---------------------------------------------------------------------------
# A board is born registered
#
# The backfill above is the operator's migration for boards that PREDATE the
# fence. A board created today is not one of those: ``create_board`` publishes
# its register authority inside the same admitted-creation window that brings
# its directory and store into existence. A board that had to be backfilled by
# hand before anything could be done to it was a board born unregistered —
# every later operation that needs authority refused until an operator
# intervened, with a removal refusing outright ("no register entry ... cannot
# determine removability").
# ---------------------------------------------------------------------------

def _assert_left_nothing_behind(slug: str, *, what: str) -> None:
    """No directory, no store, no metadata, no init lock for *slug*.

    A refusal that still publishes something at the board's path has not
    refused: the directory alone makes the name discoverable to
    ``list_boards`` again.
    """
    db_path = kb.kanban_db_path(board=slug)
    init_lock = db_path.with_name(db_path.name + ".init.lock")
    assert not db_path.exists(), f"{what} left a store: {db_path}"
    assert not kb.board_metadata_path(slug).exists(), f"{what} left board.json"
    assert not init_lock.exists(), f"{what} left an init lock: {init_lock}"
    assert not kb.board_dir(slug).exists(), f"{what} left the board directory"
    assert all(entry["slug"] != slug for entry in kb.list_boards())


def _assert_born_registered(slug: str) -> None:
    """Every durable fact the backfill writes, written at creation instead.

    Read off the three stores directly — the register, the archive and the
    board's own gate — never through the module that wrote them.
    """
    assert register_row(slug) == {
        "lifecycle": "live",
        "epoch": 1,
        "epoch_before": None,
        "gate_move": "settled",
    }
    assert register_lineage(slug) == [1]
    assert marker_row(slug) is True
    assert archive_records(slug) == 1
    # The registration is DISCHARGED, not left in flight.
    assert intent_rows(slug) == 0
    assert gate_row(kb.kanban_db_path(board=slug)) == ("open", 1)


def test_a_created_board_carries_its_register_authority_immediately(fence_home):
    """Creation publishes the entry, the marker + receipt and the mirror."""
    kb.create_board("born-fenced")

    _assert_born_registered("born-fenced")


def test_a_created_board_can_start_a_reversible_removal_with_no_backfill(
    fence_home,
):
    """The defect, stated as the operator sees it: a board created a moment
    ago could not be removed, because removability could not be determined
    for a name the register had never heard of."""
    kb.create_board("removable")

    result = start_removal("removable", mode="reversible")

    assert result.success, result.message
    assert result.outcome is kb.RemovalIntentOutcome.STARTED
    assert result.mode is kb.RemovalMode.REVERSIBLE
    assert register_row("removable")["lifecycle"] == "removing"


def test_a_created_board_can_start_a_permanent_removal_with_no_backfill(
    fence_home,
):
    """Same, through the mode that also needs the operator confirmation —
    minted here by the shipped disclosure/confirm surfaces, so the
    permanence statement really is what gets confirmed."""
    kb.create_board("destroyable")

    result = start_removal("destroyable", mode="permanent")

    assert result.success, result.message
    assert result.outcome is kb.RemovalIntentOutcome.STARTED
    assert result.mode is kb.RemovalMode.PERMANENT
    assert register_row("destroyable")["lifecycle"] == "removing"


def test_creating_a_removed_name_is_still_refused_before_any_side_effect(
    fence_home,
):
    """Registering at creation must not soften the resurrection guard: the
    refusal still arrives BEFORE the directory, the metadata or the store."""
    kb.create_board("gone-for-good")
    assert kb.remove_board_fenced("gone-for-good", mode="reversible").success
    assert not kb.board_dir("gone-for-good").exists()

    with pytest.raises(kb.BoardFenceClosedError) as excinfo:
        kb.create_board("gone-for-good")

    assert excinfo.value.refusal.rule is kb.FenceRefusalRule.GA_2
    assert excinfo.value.refusal.outcome is kb.FenceOutcome.REFUSED_CLOSED
    _assert_left_nothing_behind("gone-for-good", what="the refused creation")


def test_creating_a_name_whose_marker_is_set_is_still_refused(fence_home):
    """The marker outlives the entry on purpose (it lives in the other
    store), so a name it vouches for is refused even with no entry at all —
    which is exactly the loss the second store exists to survive."""
    kb.create_board("marked")
    # Take the storage away through the legacy unguarded path, then LOSE the
    # register row itself with a plain connection: no shipped path produces
    # this state, it is the fault the marker is there to outlive.
    kb.remove_board("marked", archive=False)
    conn = sqlite3.connect(str(kb.register_db_path()))
    try:
        conn.execute("DELETE FROM board_register WHERE board_name = ?", ("marked",))
        conn.commit()
    finally:
        conn.close()
    assert register_row("marked") is None
    assert marker_row("marked") is True

    with pytest.raises(kb.BoardFenceClosedError) as excinfo:
        kb.create_board("marked")

    assert excinfo.value.refusal.rule is kb.FenceRefusalRule.GA_2
    assert "marker" in excinfo.value.refusal.message
    _assert_left_nothing_behind("marked", what="the refused creation")


def test_an_interrupted_registration_leaves_nothing_durable(fence_home):
    """The registration is part of the creation, so a creation that cannot
    register fails as a whole: no half-made board, in any store."""
    original = kb.transition_register_entry
    kb.transition_register_entry = lambda entry: (_ for _ in ()).throw(
        sqlite3.OperationalError("register store unavailable")
    )
    try:
        with pytest.raises(kb.BoardFenceClosedError) as excinfo:
            kb.create_board("half-born")
    finally:
        kb.transition_register_entry = original

    assert excinfo.value.refusal.rule is kb.FenceRefusalRule.GA_5_FAIL
    _assert_left_nothing_behind("half-born", what="the failed registration")
    assert register_row("half-born") is None
    assert marker_row("half-born") is False
    assert archive_records("half-born") == 0
    assert intent_rows("half-born") == 0

    # And the name is still creatable: the failure left it repairable.
    kb.create_board("half-born")
    _assert_born_registered("half-born")
    conn = kb.connect(board="half-born")
    try:
        assert ready_task(conn, "post-repair")
    finally:
        conn.close()
    assert start_removal("half-born").success


def test_a_registration_that_could_not_unwind_is_repaired_on_the_next_attempt(
    fence_home,
):
    """When the unwind itself cannot complete, the intent is deliberately
    RETAINED — and the next creation's own recovery (the existing
    ``recover_backfill_intent``, run before any new attempt) settles it and
    finishes the job. No second repair mechanism, and no name stuck
    uncreatable behind an intent nobody discharges."""
    originals = (kb.transition_register_entry, kb._compensate_backfill_mirror)
    kb.transition_register_entry = lambda entry: (_ for _ in ()).throw(
        sqlite3.OperationalError("register store unavailable")
    )
    kb._compensate_backfill_mirror = lambda slug: False
    try:
        with pytest.raises(kb.BoardFenceClosedError) as excinfo:
            kb.create_board("retained-intent")
    finally:
        kb.transition_register_entry, kb._compensate_backfill_mirror = originals

    assert "retained" in excinfo.value.refusal.message
    # Nothing anyone reads as authority, and nothing on disk — but the
    # journal remembers there is work left to undo.
    _assert_left_nothing_behind("retained-intent", what="the failed registration")
    assert register_row("retained-intent") is None
    assert marker_row("retained-intent") is False
    assert intent_rows("retained-intent") == 1

    kb.create_board("retained-intent")

    _assert_born_registered("retained-intent")


def test_the_named_backfill_still_refuses_an_already_registered_board(
    fence_home,
):
    """The migration is one-time, and creation is not a licence to re-run
    it: a board that already carries an entry and a marker gets the same
    refusal it has always got. (``backfill_all_boards`` reports such a
    board as already fenced, which is a different question and unchanged.)"""
    kb.create_board("already-registered")

    result = kb.backfill_register_entry("already-registered")

    assert result.success is False
    assert "marker" in result.message
    # The refusal changed nothing: the board is still exactly as created.
    _assert_born_registered("already-registered")
    assert dict(kb.backfill_all_boards())["already-registered"].message == (
        "already fenced"
    )
