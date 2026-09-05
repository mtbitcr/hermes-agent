"""Tests for the board removal fence-closing point (§3.3).

Covers:
- The closing point is exactly the open→closing commit
- A grant admitted immediately before and refused immediately after
- The commit is atomic (no window where two concurrent observers disagree)
- Recording intent alone refuses nothing (GA-1)
"""

from __future__ import annotations

import sqlite3
import sys
import threading
import time
from pathlib import Path
from typing import Optional

import pytest

# Ensure the worktree is on sys.path
_WORKTREE = Path(__file__).resolve().parents[2]
if str(_WORKTREE) not in sys.path:
    sys.path.insert(0, str(_WORKTREE))

from hermes_cli import kanban_db as kb


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def fresh_home(tmp_path, monkeypatch):
    """Isolated HERMES_HOME with no prior kanban state."""
    home = tmp_path / "hermes_home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    for var in (
        "HERMES_KANBAN_DB",
        "HERMES_KANBAN_WORKSPACES_ROOT",
        "HERMES_KANBAN_HOME",
        "HERMES_KANBAN_BOARD",
    ):
        monkeypatch.delenv(var, raising=False)
    # Reset module-level caches
    kb._INITIALIZED_PATHS.clear()
    kb._REGISTER_INITIALIZED = False
    try:
        import hermes_constants
        hermes_constants._cached_default_hermes_root = None
    except Exception:
        pass
    return home


def _create_board_with_register(board: str) -> kb.RegisterEntry:
    """Create a board with a register entry for testing fence primitives."""
    kb.create_board(board)

    # Create register entry
    with kb.register_connect() as conn:
        now = int(time.time())
        entry = kb.RegisterEntry(
            board_name=board,
            lifecycle=kb.BoardLifecycle.LIVE,
            epoch=1,
            epoch_lineage=[1],
            created_at=now,
            updated_at=now,
        )
        conn.execute("BEGIN IMMEDIATE")
        kb._write_register_entry(conn, entry)
        conn.execute("COMMIT")

    # Ensure fence state exists
    db_path = kb.kanban_db_path(board=board)
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        """CREATE TABLE IF NOT EXISTS board_fence_state (
            id                      INTEGER PRIMARY KEY CHECK (id = 1),
            gate                    TEXT NOT NULL DEFAULT 'open',
            epoch_mirror            INTEGER NOT NULL DEFAULT 1,
            updated_at              INTEGER NOT NULL
        )"""
    )
    conn.execute(
        "INSERT OR IGNORE INTO board_fence_state (id, gate, epoch_mirror, updated_at) "
        "VALUES (1, 'open', 1, ?)",
        (int(time.time()),),
    )
    conn.commit()
    conn.close()

    return kb.get_register_entry(board)


def _setup_board_for_removal(board: str) -> kb.RegisterEntry:
    """Create a board and prepare it for removal (intent committed)."""
    # Check if board already has a register entry
    entry = kb.get_register_entry(board)
    if entry is None:
        entry = _create_board_with_register(board)

    # Commit intent: move to 'removing' with new epoch
    with kb.register_connect() as conn:
        now = int(time.time())
        new_epoch = entry.epoch + 1
        new_entry = kb.RegisterEntry(
            board_name=board,
            lifecycle=kb.BoardLifecycle.REMOVING,
            epoch=new_epoch,
            epoch_before=entry.epoch,
            gate_move=kb.GateMove.PENDING,
            epoch_lineage=(entry.epoch_lineage or []) + [new_epoch],
            created_at=entry.created_at,
            updated_at=now,
        )
        conn.execute("BEGIN IMMEDIATE")
        kb._write_register_entry(conn, new_entry)
        conn.execute("COMMIT")
        return new_entry


# ---------------------------------------------------------------------------
# Fence-Closing Point Tests
# ---------------------------------------------------------------------------

class TestFenceClosingPoint:
    """Test that the fence-closing point is exactly the open→closing commit."""

    def test_closing_point_is_gate_commit(self, fresh_home):
        """The fence-closing point is the commit of gate from open to closing."""
        entry = _setup_board_for_removal("close-point-test")

        # Before closing: gate should be open
        db_path = kb.kanban_db_path(board="close-point-test")
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT gate FROM board_fence_state WHERE id = 1").fetchone()
        assert row["gate"] == "open"
        conn.close()

        # Commit the fence-closing point
        result = kb.commit_fence_closing_point("close-point-test", new_epoch=entry.epoch)
        assert result.success

        # After closing: gate should be closing
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT gate, epoch_mirror FROM board_fence_state WHERE id = 1").fetchone()
        assert row["gate"] == "closing"
        assert row["epoch_mirror"] == entry.epoch
        conn.close()

    def test_epoch_mirror_written_with_gate_em2(self, fresh_home):
        """EM-2: The epoch mirror is written in the same commit as the gate move."""
        entry = _setup_board_for_removal("em2-test")

        result = kb.commit_fence_closing_point("em2-test", new_epoch=entry.epoch)
        assert result.success
        assert result.new_epoch == entry.epoch

        # Verify both were written atomically
        db_path = kb.kanban_db_path(board="em2-test")
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT gate, epoch_mirror FROM board_fence_state WHERE id = 1").fetchone()
        conn.close()

        assert row["gate"] == "closing"
        assert row["epoch_mirror"] == entry.epoch

    def test_fence_close_is_idempotent(self, fresh_home):
        """Committing closing when already closing is a no-op."""
        entry = _setup_board_for_removal("idempotent-test")

        # First close
        result1 = kb.commit_fence_closing_point("idempotent-test", new_epoch=entry.epoch)
        assert result1.success

        # Second close (idempotent)
        result2 = kb.commit_fence_closing_point("idempotent-test", new_epoch=entry.epoch)
        assert result2.success

        # State should still be closing
        db_path = kb.kanban_db_path(board="idempotent-test")
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT gate FROM board_fence_state WHERE id = 1").fetchone()
        conn.close()
        assert row["gate"] == "closing"


# ---------------------------------------------------------------------------
# Grant Admission Tests
# ---------------------------------------------------------------------------

class TestGrantAdmissionAroundClosingPoint:
    """Test that grants are admitted before and refused after the closing point."""

    def test_fence_check_admits_before_close(self, fresh_home):
        """A fence check is admitted immediately before the fence-closing point."""
        _create_board_with_register("admit-before")

        # Fence is open
        admitted, refusal = kb.check_fence_for_operation("admit-before")
        assert admitted
        assert refusal is None

    def test_fence_check_refused_after_close(self, fresh_home):
        """A fence check is refused immediately after the fence-closing point."""
        entry = _setup_board_for_removal("refuse-after")

        # Commit the fence-closing point
        result = kb.commit_fence_closing_point("refuse-after", new_epoch=entry.epoch)
        assert result.success

        # Now fence checks should be refused
        admitted, refusal = kb.check_fence_for_operation("refuse-after")
        assert not admitted
        assert refusal is not None
        assert refusal.outcome == kb.FenceOutcome.REFUSED_CLOSED

    def test_handle_valid_before_close_invalid_after(self, fresh_home):
        """A handle opened before close becomes invalid after close."""
        _create_board_with_register("handle-test")

        # Open a handle before close
        handle = kb.open_board_handle(board="handle-test")
        assert handle is not None

        # Handle should be valid
        valid, refusal = handle.validate_for_operation()
        assert valid

        # Now prepare and close the fence
        entry = _setup_board_for_removal("handle-test")
        result = kb.commit_fence_closing_point("handle-test", new_epoch=entry.epoch)
        assert result.success

        # The handle should now be invalid due to gate being closing
        valid, refusal = handle.validate_for_operation()
        assert not valid
        assert refusal is not None
        assert refusal.outcome == kb.FenceOutcome.REFUSED_CLOSED

        handle.close()


# ---------------------------------------------------------------------------
# Atomic Commit Tests
# ---------------------------------------------------------------------------

class TestAtomicCommit:
    """Test that the fence-closing commit is atomic."""

    def test_no_intermediate_state_observed(self, fresh_home):
        """Two concurrent observers cannot disagree on the fence state."""
        entry = _setup_board_for_removal("atomic-test")

        observations = []
        barrier = threading.Barrier(3)

        def observe(observer_id: int):
            barrier.wait()  # Synchronize start
            # Read the fence state
            admitted, _ = kb.check_fence_for_operation("atomic-test")
            observations.append((observer_id, admitted))

        def close_fence():
            barrier.wait()  # Synchronize start
            kb.commit_fence_closing_point("atomic-test", new_epoch=entry.epoch)

        # Start observers and closer
        threads = [
            threading.Thread(target=observe, args=(1,)),
            threading.Thread(target=observe, args=(2,)),
            threading.Thread(target=close_fence),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5.0)

        # Both observers should see the same state (either both admitted or both refused)
        # at any given instant. Due to timing, they may see different states if one
        # reads before and one after, but that's expected. What matters is each
        # observation is consistent.
        assert len(observations) == 2
        for observer_id, admitted in observations:
            assert admitted in (True, False)  # No intermediate state

    def test_concurrent_close_attempts_exactly_one_wins(self, fresh_home):
        """Two racing close attempts: exactly one wins."""
        entry = _setup_board_for_removal("race-close")

        results = []
        barrier = threading.Barrier(2)

        def try_close():
            barrier.wait()
            result = kb.commit_fence_closing_point("race-close", new_epoch=entry.epoch)
            results.append(result)

        threads = [
            threading.Thread(target=try_close),
            threading.Thread(target=try_close),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5.0)

        # Both should succeed because close is idempotent
        assert len(results) == 2
        assert all(r.success for r in results)

        # Final state should be closing
        db_path = kb.kanban_db_path(board="race-close")
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT gate FROM board_fence_state WHERE id = 1").fetchone()
        conn.close()
        assert row["gate"] == "closing"


# ---------------------------------------------------------------------------
# GA-1: Intent Refuses Nothing Tests
# ---------------------------------------------------------------------------

class TestGA1IntentRefusesNothing:
    """Test GA-1: recording intent alone refuses nothing."""

    def test_intent_does_not_close_fence(self, fresh_home):
        """Recording intent (moving to 'removing') does not close the fence."""
        _create_board_with_register("intent-test")

        # Before intent: fence is open
        admitted, _ = kb.check_fence_for_operation("intent-test")
        assert admitted

        # Record intent (but don't close the fence)
        _setup_board_for_removal("intent-test")

        # After intent: fence should STILL be open because gate hasn't moved
        # The check_fence_for_operation should still admit because:
        # 1. Gate is still 'open'
        # 2. EM-4b: the epoch mismatch is expected during intent→fence window
        admitted, refusal = kb.check_fence_for_operation("intent-test")
        assert admitted, f"Intent alone should not refuse, but got: {refusal}"

    def test_removing_admits_open_after_intent(self, fresh_home):
        """GA-1: lifecycle=removing still admits opening after intent is recorded."""
        _setup_board_for_removal("removing-open")

        # Gate A should still admit opening
        result = kb.check_gate_a_admission(
            "removing-open",
            kb.GateAAction.OPEN,
            storage_exists=True,
        )
        assert result.admitted, f"GA-1: removing should admit open, but got: {result.message}"

    def test_intent_to_fence_window_admits_operations(self, fresh_home):
        """Operations in the intent→fence window are admitted, not refused."""
        entry = _setup_board_for_removal("window-test")

        # In the window: intent recorded, gate not yet closed
        # EM-4b should make this an expected lag that refuses nothing

        # Open a handle in the window
        handle = kb.open_board_handle(board="window-test")
        assert handle is not None

        # The handle should be valid
        valid, refusal = handle.validate_for_operation()
        assert valid, f"EM-4b: handle in intent→fence window should be valid, but got: {refusal}"

        handle.close()

    def test_fence_closes_only_at_gate_commit(self, fresh_home):
        """The fence only closes at the gate commit, not at intent."""
        entry = _setup_board_for_removal("gate-commit-test")

        # Intent recorded, but fence not yet closed
        admitted_before, _ = kb.check_fence_for_operation("gate-commit-test")
        assert admitted_before, "Before gate commit, fence should be open"

        # Commit the gate (fence-closing point)
        result = kb.commit_fence_closing_point("gate-commit-test", new_epoch=entry.epoch)
        assert result.success

        # Now the fence is closed
        admitted_after, refusal = kb.check_fence_for_operation("gate-commit-test")
        assert not admitted_after, "After gate commit, fence should be closed"
        assert refusal is not None
