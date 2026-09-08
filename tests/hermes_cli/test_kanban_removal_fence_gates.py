"""The operator's recorded GA-5 migration (the backfill), exercised end to
end: an explicit, recorded command an operator runs at release — one board
or all of them — and until a board has been through it, its ordinary
writes refuse with the structured refusal. Reads keep working throughout.
"""

from __future__ import annotations

import json
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
    register_row,
    row_count,
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
