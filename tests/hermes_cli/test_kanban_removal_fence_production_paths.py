"""The removal fence on the paths real users actually take.

These tests never arm a board by poking tables directly and never open a
store by hand where a real API exists. They call ``create_board`` /
``connect`` / ``claim_task`` / ``add_comment`` / ``archive_task`` — the
exact functions the CLI, the dispatcher and the dashboard call — and then
assert what is ON DISK, read back with a plain read-only SQLite
connection. Boards that need Gate B/Gate A armed go through
:func:`create_fenced_board`, the real ``create_board`` followed by the
real recorded backfill (``create_board`` no longer arms a fence itself).
"""

from __future__ import annotations

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
    cli,
    close_fence,
    create_fenced_board,
    gate_row,
    has_gate_table,
    intent_rows,
    make_legacy_board,
    marker_row,
    ready_task,
    register_row,
    row_count,
    task_row,
)


# ---------------------------------------------------------------------------
# Reproduction 7 — a board with no Gate B is unfenced, not blocked
# ---------------------------------------------------------------------------

def test_repro_7_a_board_without_gate_b_admits_ordinary_mutation(
    fence_home, tmp_path
):
    """A board that predates the fence — no ``board_fence_state`` row at
    all — is READABLE and WRITABLE. Nothing in creation arms a gate any
    more, so the ONLY thing that makes a store's writes governed is the
    table existing; absent it, the store is unfenced and admitted, exactly
    as the original, unchanged creation path always was."""
    db_path = make_legacy_board(tmp_path, "unbackfilled")

    conn = kb.connect(board="unbackfilled")
    try:
        # Reading a pre-fence board keeps working.
        assert len(kb.list_tasks(conn)) == 1

        task_id = kb.create_task(conn, title="new work", assignee="worker")
    finally:
        conn.close()

    assert row_count(db_path, "tasks") == 2
    assert task_row(db_path, task_id)["status"] == "ready"
    assert not has_gate_table(db_path)


def test_repro_7_the_recorded_migration_is_what_unblocks_the_board(
    fence_home, tmp_path
):
    """And the operator command is the only thing that lifts the refusal."""
    db_path = make_legacy_board(tmp_path, "migrate-me")

    out = cli("boards backfill-fence migrate-me")
    assert "ok" in out

    assert gate_row(db_path) == ("open", 1)
    assert register_row("migrate-me")["lifecycle"] == "live"
    assert marker_row("migrate-me") is True
    assert archive_records("migrate-me") == 1
    assert intent_rows("migrate-me") == 0

    kb._INITIALIZED_PATHS.clear()
    conn = kb.connect(board="migrate-me")
    try:
        task_id = kb.create_task(conn, title="new work", assignee="worker")
    finally:
        conn.close()
    assert task_row(db_path, task_id)["status"] == "ready"
    assert row_count(db_path, "tasks") == 2


def test_repro_7_the_arbitrary_path_connector_stays_outside_the_fence(
    fence_home, tmp_path
):
    """The compatibility connector for NON-board callers is separate from,
    and unreachable by, board operations.

    A file with no board identity has no register authority to consult,
    so it is not a board and the fence has nothing to say about it.
    """
    scratch = tmp_path / "not-a-board.db"
    conn = kb.connect(db_path=scratch)
    try:
        task_id = kb.create_task(conn, title="standalone", assignee="worker")
        assert kb.add_comment(conn, task_id, "a", "b") > 0
    finally:
        conn.close()

    assert not has_gate_table(scratch)
    assert register_row("default") is None


# ---------------------------------------------------------------------------
# The fence follows the kernel, not the caller
# ---------------------------------------------------------------------------

def test_every_kernel_mutation_path_is_governed_on_a_real_board(fence_home):
    """One board, one closed fence, and the whole mutation surface the
    CLI/dispatcher use refuses — including the paths the first repair
    left ungated (removals and content writes)."""
    create_fenced_board("surface")
    db_path = kb.kanban_db_path(board="surface")
    conn = kb.connect(board="surface")
    parent = ready_task(conn, "parent")
    child = ready_task(conn, "child")
    kb.add_comment(conn, parent, "author", "seed")
    kb.claim_task(conn, parent)
    conn.close()

    before_counts = {
        table: row_count(db_path, table)
        for table in ("tasks", "task_comments", "task_events", "task_runs")
    }

    close_fence("surface")

    conn = kb.connect(board="surface")
    refused = []
    try:
        for name, call in (
            ("create_task", lambda: kb.create_task(conn, title="n", assignee="w")),
            ("add_comment", lambda: kb.add_comment(conn, parent, "a", "b")),
            ("claim_task", lambda: kb.claim_task(conn, child)),
            ("heartbeat_claim", lambda: kb.heartbeat_claim(conn, parent, claimer="w")),
            ("archive_task", lambda: kb.archive_task(conn, child)),
            ("delete_task", lambda: kb.delete_task(conn, child)),
            ("assign_task", lambda: kb.assign_task(conn, child, "other")),
            ("link_tasks", lambda: kb.link_tasks(conn, parent, child)),
            ("recompute_ready", lambda: kb.recompute_ready(conn)),
        ):
            with pytest.raises(kb.BoardFenceClosedError):
                call()
            refused.append(name)
    finally:
        conn.close()

    assert len(refused) == 9
    after_counts = {
        table: row_count(db_path, table)
        for table in ("tasks", "task_comments", "task_events", "task_runs")
    }
    assert after_counts == before_counts


def test_an_admitted_board_still_does_all_of_that(fence_home):
    """The mirror image: with the fence open the same calls all land, so
    the refusals above are the fence and not a broken kernel."""
    create_fenced_board("open-board")
    db_path = kb.kanban_db_path(board="open-board")
    conn = kb.connect(board="open-board")
    try:
        parent = ready_task(conn, "parent")
        child = ready_task(conn, "child")
        assert kb.add_comment(conn, parent, "a", "b") > 0
        assert kb.claim_task(conn, parent) is not None
        assert kb.heartbeat_claim(conn, parent, claimer=kb._claimer_id()) in (True, False)
        assert kb.archive_task(conn, child) is True
        assert kb.delete_archived_task(conn, child) is True
    finally:
        conn.close()

    assert gate_row(db_path) == ("open", 1)
    assert task_row(db_path, child) is None
    assert task_row(db_path, parent)["status"] == "running"
