"""Counterexamples: the states the fence must never be caught in.

The headline one is reviewer reproduction 6. The prepared backfill wrote
its mirror phase, committed it, and then — when the register phase failed
— deleted only its intent. Gate B survived. A board that had merely not
been migrated yet became a board that carried a gate with no matching
register entry: writable before the failed migration, blocked for missing
authority afterwards, and un-migratable because a retry now saw a store
it could not account for.

Every fault test here STARTS from a board with no Gate B, and asserts the
board's schema and mirror state as well as the register and the archive.
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


def _assert_pristine_legacy(slug: str, db_path: Path) -> None:
    """The board is byte-for-byte back to "not migrated yet"."""
    assert has_gate_table(db_path) is False
    assert gate_row(db_path) is None
    assert register_row(slug) is None
    assert marker_row(slug) is False
    assert archive_records(slug) == 0
    assert intent_rows(slug) == 0


# ---------------------------------------------------------------------------
# Reproduction 6 — a failed backfill leaves nothing half-armed
# ---------------------------------------------------------------------------

def test_repro_6_a_failed_register_phase_compensates_gate_b(fence_home, tmp_path):
    """The reviewer's exact injection: a genuine legacy board, a register
    write that fails after the mirror phase committed."""
    db_path = make_legacy_board(tmp_path, "half-armed")
    assert has_gate_table(db_path) is False

    def register_store_is_down(entry):
        raise sqlite3.OperationalError("register store unavailable")

    original = kb.transition_register_entry
    kb.transition_register_entry = register_store_is_down
    try:
        result = kb.backfill_register_entry("half-armed")
    finally:
        kb.transition_register_entry = original

    assert result.success is False
    assert "compensated" in result.message
    _assert_pristine_legacy("half-armed", db_path)


def test_repro_6_the_board_is_no_worse_off_than_before_the_attempt(
    fence_home, tmp_path
):
    """Before the failed attempt the board had no Gate B and admitted
    ordinary writes; after it, it must still have no Gate B and still
    admit them — not be left half-armed with a gate and no authority."""
    db_path = make_legacy_board(tmp_path, "no-worse")
    conn = kb.connect(board="no-worse")
    try:
        task_id = kb.list_tasks(conn)[0].id
        assert kb.add_comment(conn, task_id, "a", "b") > 0
    finally:
        conn.close()
    assert has_gate_table(db_path) is False

    original = kb.transition_register_entry
    kb.transition_register_entry = lambda entry: (_ for _ in ()).throw(
        sqlite3.OperationalError("register store unavailable")
    )
    try:
        assert kb.backfill_register_entry("no-worse").success is False
    finally:
        kb.transition_register_entry = original

    kb._INITIALIZED_PATHS.clear()
    conn = kb.connect(board="no-worse")
    try:
        assert kb.add_comment(conn, task_id, "a", "b") > 0
    finally:
        conn.close()
    assert has_gate_table(db_path) is False
    assert row_count(db_path, "task_comments") == 2


def test_repro_6_a_retry_after_the_fault_succeeds_and_the_board_works(
    fence_home, tmp_path
):
    """Compensation is only meaningful if it leaves the migration
    retryable — which the un-compensated version did not."""
    db_path = make_legacy_board(tmp_path, "retry-me")

    original = kb.transition_register_entry
    kb.transition_register_entry = lambda entry: (_ for _ in ()).throw(
        sqlite3.OperationalError("register store unavailable")
    )
    try:
        assert kb.backfill_register_entry("retry-me").success is False
    finally:
        kb.transition_register_entry = original

    assert kb.backfill_register_entry("retry-me").success is True

    assert gate_row(db_path) == ("open", 1)
    assert register_row("retry-me")["lifecycle"] == "live"
    assert marker_row("retry-me") is True
    assert archive_records("retry-me") == 1
    assert intent_rows("retry-me") == 0

    kb._INITIALIZED_PATHS.clear()
    conn = kb.connect(board="retry-me")
    try:
        task_id = kb.create_task(conn, title="post-migration", assignee="worker")
    finally:
        conn.close()
    assert task_row(db_path, task_id)["status"] == "ready"


def test_repro_6_a_failed_mirror_phase_leaves_nothing_at_all(fence_home, tmp_path):
    """The mirror phase runs BEFORE any durable register fact exists, so
    failing there must leave the board exactly as found."""
    db_path = make_legacy_board(tmp_path, "mirror-fails")

    original = kb._ensure_in_board_fence_schema
    kb._ensure_in_board_fence_schema = lambda conn: (_ for _ in ()).throw(
        sqlite3.OperationalError("board store unavailable")
    )
    try:
        result = kb.backfill_register_entry("mirror-fails")
    finally:
        kb._ensure_in_board_fence_schema = original

    assert result.success is False
    _assert_pristine_legacy("mirror-fails", db_path)


def test_repro_6_a_failed_finalize_rolls_the_whole_attempt_back(
    fence_home, tmp_path
):
    """The register entry landed but the archive side did not: the entry
    is undone AND the mirror compensated, so nothing survives."""
    db_path = make_legacy_board(tmp_path, "finalize-fails")

    original = kb._finalize_backfill_intent
    kb._finalize_backfill_intent = lambda slug, intent_id, now: False
    try:
        result = kb.backfill_register_entry("finalize-fails")
    finally:
        kb._finalize_backfill_intent = original

    assert result.success is False
    _assert_pristine_legacy("finalize-fails", db_path)


def test_repro_6_an_uncompensatable_failure_retains_the_intent(
    fence_home, tmp_path
):
    """If the mirror CANNOT be compensated, the intent is deliberately
    kept: the protocol must not forget there is work left to undo."""
    db_path = make_legacy_board(tmp_path, "retained")

    originals = (kb.transition_register_entry, kb._compensate_backfill_mirror)
    kb.transition_register_entry = lambda entry: (_ for _ in ()).throw(
        sqlite3.OperationalError("register store unavailable")
    )
    kb._compensate_backfill_mirror = lambda slug: False
    try:
        result = kb.backfill_register_entry("retained")
    finally:
        kb.transition_register_entry, kb._compensate_backfill_mirror = originals

    assert result.success is False
    assert "retained" in result.message
    assert intent_rows("retained") == 1
    assert register_row("retained") is None
    # Gate B is still there — which is exactly why the intent must be.
    assert has_gate_table(db_path) is True

    # And recovery finishes the unwind on the next attempt.
    recovery = kb.recover_backfill_intent("retained")
    assert recovery == "compensated"
    _assert_pristine_legacy("retained", db_path)


def test_repro_6_recovery_finishes_an_intent_whose_register_entry_landed(
    fence_home, tmp_path
):
    """The other direction: the register entry IS there, so the intent is
    finished rather than unwound."""
    db_path = make_legacy_board(tmp_path, "finish-me")

    original = kb._finalize_backfill_intent
    calls = []

    def refuse_once(slug, intent_id, now):
        calls.append(slug)
        return False

    kb._finalize_backfill_intent = refuse_once
    kb_delete = kb._delete_register_entry
    kb._delete_register_entry = lambda slug: False
    try:
        result = kb.backfill_register_entry("finish-me")
    finally:
        kb._finalize_backfill_intent = original
        kb._delete_register_entry = kb_delete

    assert result.success is False
    assert intent_rows("finish-me") == 1
    assert register_row("finish-me")["lifecycle"] == "live"

    assert kb.recover_backfill_intent("finish-me") == "finished"
    assert intent_rows("finish-me") == 0
    assert marker_row("finish-me") is True
    assert archive_records("finish-me") == 1
    assert gate_row(db_path) == ("open", 1)


# ---------------------------------------------------------------------------
# Counterexamples the fence must keep refusing
# ---------------------------------------------------------------------------

def test_a_worker_holding_a_claim_cannot_renew_it_across_the_close(fence_home):
    """The dispatcher's heartbeat is a grant too."""
    create_fenced_board("renewal")
    db_path = kb.kanban_db_path(board="renewal")
    conn = kb.connect(board="renewal")
    task_id = ready_task(conn)
    lock = kb._claimer_id()
    assert kb.claim_task(conn, task_id, claimer=lock) is not None
    before = task_row(db_path, task_id)["claim_expires"]

    close_fence("renewal")

    with pytest.raises(kb.BoardFenceClosedError):
        kb.heartbeat_claim(conn, task_id, claimer=lock, ttl_seconds=9999)
    conn.close()
    assert task_row(db_path, task_id)["claim_expires"] == before


def test_a_second_board_is_unaffected_by_the_first_ones_removal(fence_home):
    """The fence is per board name. Closing one must not stop another."""
    create_fenced_board("alpha")
    create_fenced_board("beta")
    beta_path = kb.kanban_db_path(board="beta")

    close_fence("alpha")

    conn = kb.connect(board="beta")
    try:
        task_id = kb.create_task(conn, title="unaffected", assignee="worker")
        assert kb.claim_task(conn, task_id) is not None
    finally:
        conn.close()
    assert task_row(beta_path, task_id)["status"] == "running"
    assert gate_row(beta_path) == ("open", 1)


def test_the_intent_journal_survives_a_process_that_never_came_back(
    fence_home, tmp_path
):
    """A crash between phases leaves an intent, and the NEXT attempt is
    what brings the protocol to rest — not a background sweeper nobody
    runs."""
    db_path = make_legacy_board(tmp_path, "crashed")

    now = 1_700_000_000
    with kb.archive_connect() as arch:
        arch.execute("BEGIN IMMEDIATE")
        arch.execute(
            "INSERT INTO board_backfill_intent (board_name, intent_id, epoch, "
            "phase, gate_b_preexisting, created_at) "
            "VALUES ('crashed', 'intent-from-a-dead-process', 1, 'mirrored', 0, ?)",
            (now,),
        )
        arch.execute("COMMIT")
    # The dead process had got as far as arming Gate B.
    conn = sqlite3.connect(str(db_path))
    conn.executescript(kb.IN_BOARD_FENCE_SCHEMA_SQL)
    conn.execute(
        "INSERT OR IGNORE INTO board_fence_state (id, gate, epoch_mirror, updated_at) "
        "VALUES (1, 'open', 1, ?)",
        (now,),
    )
    conn.commit()
    conn.close()
    assert has_gate_table(db_path) is True

    result = kb.backfill_register_entry("crashed")

    assert result.success is True
    assert gate_row(db_path) == ("open", 1)
    assert register_row("crashed")["lifecycle"] == "live"
    assert intent_rows("crashed") == 0


def test_recovery_on_a_name_with_no_intent_is_a_no_op(fence_home):
    kb.create_board("nothing-to-recover")
    assert kb.recover_backfill_intent("nothing-to-recover") is None
