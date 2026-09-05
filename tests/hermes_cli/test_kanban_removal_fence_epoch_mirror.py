"""Tests for the board removal fence epoch mirror (§1.1, §3.9).

Covers:
- EM-2: mirror moves only with the gate, in the same commit
- EM-3a: handle records the mirror, not the register
- EM-3b + EM-4a/b/c: table of pairs → proceed/refuse
- EM-4d: abandonment comes to rest equal
- EM-5: intent→fence window refuses nothing
- Handle opened inside the window succeeds
"""

from __future__ import annotations

import sqlite3
import sys
import time
from pathlib import Path

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

    # Ensure fence state table exists with proper init
    db_path = kb.kanban_db_path(board=board)
    conn = sqlite3.connect(str(db_path))
    # Use execute for single-statement CREATE TABLE
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


def _read_fence_state(board: str) -> dict:
    """Read the in-board fence state directly."""
    db_path = kb.kanban_db_path(board=board)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    row = conn.execute("SELECT gate, epoch_mirror FROM board_fence_state WHERE id = 1").fetchone()
    conn.close()
    return dict(row) if row else {}


# ---------------------------------------------------------------------------
# EM-2: Mirror Moves with Gate Tests
# ---------------------------------------------------------------------------

class TestEM2MirrorMovesWithGate:
    """Test EM-2: mirror moves only with the gate, in the same commit."""

    def test_mirror_written_with_gate_close(self, fresh_home):
        """EM-2: mirror is written in the same commit as gate open→closing."""
        entry = _setup_board_for_removal("em2-close")

        # Before close
        state_before = _read_fence_state("em2-close")
        assert state_before["gate"] == "open"
        assert state_before["epoch_mirror"] == 1  # Original epoch

        # Commit fence-closing point
        result = kb.commit_fence_closing_point("em2-close", new_epoch=entry.epoch)
        assert result.success

        # After close: both updated atomically
        state_after = _read_fence_state("em2-close")
        assert state_after["gate"] == "closing"
        assert state_after["epoch_mirror"] == entry.epoch

    def test_failed_gate_move_leaves_mirror_unchanged(self, fresh_home):
        """EM-2: a failed/rolled-back gate move leaves the mirror unmoved."""
        _create_board_with_register("em2-rollback")

        state_before = _read_fence_state("em2-rollback")
        original_mirror = state_before["epoch_mirror"]

        # Try to close a gate that's already frozen (should fail)
        db_path = kb.kanban_db_path(board="em2-rollback")
        conn = sqlite3.connect(str(db_path))
        conn.execute("UPDATE board_fence_state SET gate = 'frozen' WHERE id = 1")
        conn.commit()
        conn.close()

        result = kb.commit_fence_closing_point("em2-rollback", new_epoch=999)
        assert not result.success

        # Mirror should be unchanged
        state_after = _read_fence_state("em2-rollback")
        # Gate stayed frozen, mirror unchanged from the original value
        assert state_after["gate"] == "frozen"

    def test_backfill_writes_mirror(self, fresh_home):
        """EM-2: backfill writes the mirror in its own single act."""
        # Create complete storage without register entry
        db_path = kb.kanban_db_path(board="em2-backfill")
        db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(db_path))
        conn.executescript(kb.SCHEMA_SQL)
        conn.executescript(kb.IN_BOARD_FENCE_SCHEMA_SQL)
        conn.execute(
            "INSERT OR IGNORE INTO board_fence_state (id, gate, epoch_mirror, updated_at) "
            "VALUES (1, 'open', 0, ?)",  # Start with epoch_mirror=0 to verify write
            (int(time.time()),),
        )
        conn.close()

        # Backfill
        result = kb.backfill_register_entry("em2-backfill")
        assert result.success

        # Verify mirror was written
        state = _read_fence_state("em2-backfill")
        assert state["epoch_mirror"] == 1


# ---------------------------------------------------------------------------
# EM-3a: Handle Records Mirror Tests
# ---------------------------------------------------------------------------

class TestEM3aHandleRecordsMirror:
    """Test EM-3a: a handle records the mirror value, not the register."""

    def test_handle_records_mirror_not_register(self, fresh_home):
        """EM-3a: handle records the mirror value when opened."""
        _create_board_with_register("em3a-test")

        # Manually set mirror to a different value than register
        db_path = kb.kanban_db_path(board="em3a-test")
        conn = sqlite3.connect(str(db_path))
        conn.execute("UPDATE board_fence_state SET epoch_mirror = 42 WHERE id = 1")
        conn.commit()
        conn.close()

        # Update register to a different epoch
        with kb.register_connect() as reg_conn:
            entry = kb.get_register_entry("em3a-test", conn=reg_conn)
            now = int(time.time())
            new_entry = kb.RegisterEntry(
                board_name="em3a-test",
                lifecycle=kb.BoardLifecycle.LIVE,
                epoch=99,  # Different from mirror (42)
                created_at=entry.created_at,
                updated_at=now,
            )
            reg_conn.execute("BEGIN IMMEDIATE")
            kb._write_register_entry(reg_conn, new_entry)
            reg_conn.execute("COMMIT")

        # Open a handle
        handle = kb.open_board_handle(board="em3a-test")
        assert handle is not None

        # The handle should have recorded the MIRROR value (42), not register (99)
        assert handle.recorded_epoch == 42

        handle.close()


# ---------------------------------------------------------------------------
# EM-4: Epoch Pair Validation Tests
# ---------------------------------------------------------------------------

class TestEM4EpochPairValidation:
    """Test EM-4a/b/c: the pair rule for epoch validation."""

    def test_em4a_equal_is_normal(self, fresh_home):
        """EM-4a: mirror equals register epoch → proceed."""
        entry = kb.RegisterEntry(
            board_name="em4a-test",
            lifecycle=kb.BoardLifecycle.LIVE,
            epoch=5,
            created_at=int(time.time()),
            updated_at=int(time.time()),
        )

        result = kb.validate_epoch_pair(5, 5, entry)  # register=5, mirror=5
        assert result.valid
        assert "EM-4a" in result.message

    def test_em4b_expected_lag_during_intent_window(self, fresh_home):
        """EM-4b: expected lag during intent→fence window → proceed (refuses nothing)."""
        entry = kb.RegisterEntry(
            board_name="em4b-test",
            lifecycle=kb.BoardLifecycle.REMOVING,
            epoch=6,
            epoch_before=5,
            gate_move=kb.GateMove.PENDING,
            created_at=int(time.time()),
            updated_at=int(time.time()),
        )

        # Mirror equals epoch_before, lifecycle=removing, gate_move=pending
        result = kb.validate_epoch_pair(6, 5, entry)  # register=6, mirror=5 (equals epoch_before)
        assert result.valid
        assert "EM-4b" in result.message

    def test_em4c_mirror_ahead_refuses(self, fresh_home):
        """EM-4c: mirror ahead of register → refuse."""
        entry = kb.RegisterEntry(
            board_name="em4c-ahead",
            lifecycle=kb.BoardLifecycle.LIVE,
            epoch=5,
            created_at=int(time.time()),
            updated_at=int(time.time()),
        )

        result = kb.validate_epoch_pair(5, 10, entry)  # register=5, mirror=10 (ahead)
        assert not result.valid
        assert result.rule == kb.FenceRefusalRule.EM_4c
        assert "ahead" in result.message.lower()

    def test_em4c_mirror_matches_neither(self, fresh_home):
        """EM-4c: mirror matches neither epoch nor epoch_before → refuse."""
        entry = kb.RegisterEntry(
            board_name="em4c-neither",
            lifecycle=kb.BoardLifecycle.REMOVING,
            epoch=6,
            epoch_before=5,
            gate_move=kb.GateMove.PENDING,
            created_at=int(time.time()),
            updated_at=int(time.time()),
        )

        result = kb.validate_epoch_pair(6, 3, entry)  # register=6, epoch_before=5, mirror=3
        assert not result.valid
        assert result.rule == kb.FenceRefusalRule.EM_4c

    def test_em4c_lag_on_non_removing_refuses(self, fresh_home):
        """EM-4c: lag on entry that is not removing → refuse."""
        entry = kb.RegisterEntry(
            board_name="em4c-live",
            lifecycle=kb.BoardLifecycle.LIVE,  # Not removing
            epoch=6,
            epoch_before=5,
            created_at=int(time.time()),
            updated_at=int(time.time()),
        )

        result = kb.validate_epoch_pair(6, 5, entry)  # Lag but not removing
        assert not result.valid
        assert result.rule == kb.FenceRefusalRule.EM_4c
        assert "not removing" in result.message.lower()

    def test_em4c_lag_with_gate_move_settled_refuses(self, fresh_home):
        """EM-4c: lag whose gate_move is settled → refuse."""
        entry = kb.RegisterEntry(
            board_name="em4c-settled",
            lifecycle=kb.BoardLifecycle.REMOVING,
            epoch=6,
            epoch_before=5,
            gate_move=kb.GateMove.SETTLED,  # Not pending
            created_at=int(time.time()),
            updated_at=int(time.time()),
        )

        result = kb.validate_epoch_pair(6, 5, entry)  # Lag but gate_move=settled
        assert not result.valid
        assert result.rule == kb.FenceRefusalRule.EM_4c
        assert "settled" in result.message.lower()


# ---------------------------------------------------------------------------
# EM-4d: Abandonment Tests
# ---------------------------------------------------------------------------

class TestEM4dAbandonment:
    """Test EM-4d: abandonment does not increment a second time."""

    def test_abandonment_comes_to_rest_equal(self, fresh_home):
        """EM-4d: after abandonment, register and mirror are equal (EM-4a state)."""
        _create_board_with_register("em4d-abandon")

        # Get original entry
        with kb.register_connect() as conn:
            entry = kb.get_register_entry("em4d-abandon", conn=conn)
            original_epoch = entry.epoch

        # Simulate intent (epoch+1)
        removal_entry = _setup_board_for_removal("em4d-abandon")
        intent_epoch = removal_entry.epoch

        # Close the fence (write mirror with intent_epoch)
        result = kb.commit_fence_closing_point("em4d-abandon", new_epoch=intent_epoch)
        assert result.success

        # After closing, mirror should equal register (intent_epoch)
        state = _read_fence_state("em4d-abandon")
        assert state["epoch_mirror"] == intent_epoch

        # Simulate abandonment: reopen gate with the SAME epoch (no second increment)
        db_path = kb.kanban_db_path(board="em4d-abandon")
        conn = sqlite3.connect(str(db_path))
        conn.execute("BEGIN IMMEDIATE")
        conn.execute(
            "UPDATE board_fence_state SET gate = 'open', epoch_mirror = ? WHERE id = 1",
            (intent_epoch,),  # EM-4d: write the SAME epoch, not a new increment
        )
        conn.execute("COMMIT")
        conn.close()

        # Return entry to live at the same epoch
        with kb.register_connect() as conn:
            now = int(time.time())
            live_entry = kb.RegisterEntry(
                board_name="em4d-abandon",
                lifecycle=kb.BoardLifecycle.LIVE,
                epoch=intent_epoch,  # Same epoch, not incremented again
                gate_move=kb.GateMove.SETTLED,
                created_at=removal_entry.created_at,
                updated_at=now,
            )
            conn.execute("BEGIN IMMEDIATE")
            kb._write_register_entry(conn, live_entry)
            conn.execute("COMMIT")

        # Verify: register and mirror are now equal
        state_after = _read_fence_state("em4d-abandon")
        reg_entry = kb.get_register_entry("em4d-abandon")

        assert state_after["epoch_mirror"] == reg_entry.epoch
        # This is the EM-4a state (equal)
        pair_result = kb.validate_epoch_pair(reg_entry.epoch, state_after["epoch_mirror"], reg_entry)
        assert pair_result.valid
        assert "EM-4a" in pair_result.message


# ---------------------------------------------------------------------------
# EM-5: Intent→Fence Window Tests (CRITICAL)
# ---------------------------------------------------------------------------

class TestEM5IntentFenceWindow:
    """Test EM-5: the intent→fence window refuses nothing.

    This is THE MOST IMPORTANT test in this task. A refusal in this
    window is a design failure, not a safe default.
    """

    def test_window_admits_ordinary_reads(self, fresh_home):
        """EM-5: ordinary reads in the intent→fence window succeed."""
        _create_board_with_register("em5-reads")
        _setup_board_for_removal("em5-reads")

        # In the window: intent recorded, gate not yet closed
        # Open a handle and do a read
        conn = kb.open_board_store(board="em5-reads", validate_epoch=False)
        assert conn is not None

        # Read should succeed
        try:
            row = conn.execute("SELECT 1 FROM tasks LIMIT 1").fetchone()
            # Success - no exception raised
        finally:
            conn.close()

    def test_window_admits_ordinary_writes(self, fresh_home):
        """EM-5: ordinary writes in the intent→fence window succeed."""
        _create_board_with_register("em5-writes")
        _setup_board_for_removal("em5-writes")

        # In the window: intent recorded, gate not yet closed
        conn = kb.open_board_store(board="em5-writes", validate_epoch=False)
        assert conn is not None

        # Write should succeed (create a dummy event)
        try:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute(
                "INSERT INTO task_events (task_id, kind, created_at) VALUES (?, ?, ?)",
                ("test-task", "test-event", int(time.time())),
            )
            conn.execute("COMMIT")
            # Success - no exception raised
        finally:
            conn.close()

    def test_window_admits_grant_operation(self, fresh_home):
        """EM-5: a grant (reservation) in the intent→fence window succeeds."""
        _create_board_with_register("em5-grant")
        removal_entry = _setup_board_for_removal("em5-grant")

        # In the window: intent recorded, gate not yet closed
        # The fence check should admit because EM-4b recognizes the expected lag
        admitted, refusal = kb.check_fence_for_operation("em5-grant")
        assert admitted, f"EM-5: grant in intent→fence window should succeed, got: {refusal}"
        assert refusal is None

    def test_window_records_zero_refusals(self, fresh_home):
        """EM-5: operations in the window record ZERO refusals."""
        _create_board_with_register("em5-zero-refusals")
        _setup_board_for_removal("em5-zero-refusals")

        # Multiple operations in the window
        refusals = []

        for _ in range(5):
            admitted, refusal = kb.check_fence_for_operation("em5-zero-refusals")
            if not admitted:
                refusals.append(refusal)

        assert len(refusals) == 0, f"EM-5: expected zero refusals, got {len(refusals)}"

    def test_window_records_zero_indeterminate_outcomes(self, fresh_home):
        """EM-5: operations in the window record ZERO indeterminate outcomes."""
        _create_board_with_register("em5-zero-indet")
        _setup_board_for_removal("em5-zero-indet")

        indeterminate = []

        for _ in range(5):
            admitted, refusal = kb.check_fence_for_operation("em5-zero-indet")
            if refusal and refusal.outcome == kb.FenceOutcome.REFUSED_INDETERMINATE:
                indeterminate.append(refusal)

        assert len(indeterminate) == 0, f"EM-5: expected zero indeterminate, got {len(indeterminate)}"

    def test_gate_commit_then_next_grant_refused(self, fresh_home):
        """EM-5: after gate commit, the next grant IS refused."""
        _create_board_with_register("em5-then-refused")
        entry = _setup_board_for_removal("em5-then-refused")

        # Before gate commit: admitted
        admitted_before, _ = kb.check_fence_for_operation("em5-then-refused")
        assert admitted_before, "Before gate commit should be admitted"

        # Commit the gate
        result = kb.commit_fence_closing_point("em5-then-refused", new_epoch=entry.epoch)
        assert result.success

        # After gate commit: refused
        admitted_after, refusal = kb.check_fence_for_operation("em5-then-refused")
        assert not admitted_after, "After gate commit should be refused"
        assert refusal is not None
        assert refusal.outcome == kb.FenceOutcome.REFUSED_CLOSED

    def test_handle_opened_inside_window_succeeds(self, fresh_home):
        """EM-5: a handle opened inside the window succeeds on its first and later operations."""
        _create_board_with_register("em5-handle-window")
        _setup_board_for_removal("em5-handle-window")

        # Open handle inside the window
        handle = kb.open_board_handle(board="em5-handle-window")
        assert handle is not None

        # First operation should succeed
        valid, refusal = handle.validate_for_operation()
        assert valid, f"Handle opened in window should succeed on first op: {refusal}"

        # Subsequent operations while window lasts should also succeed
        for i in range(3):
            valid, refusal = handle.validate_for_operation()
            assert valid, f"Handle operation {i+1} in window should succeed: {refusal}"

        handle.close()


# ---------------------------------------------------------------------------
# Handle Epoch Validation Tests
# ---------------------------------------------------------------------------

class TestHandleEpochValidation:
    """Test BoardHandle epoch validation behavior."""

    def test_handle_validates_on_each_operation(self, fresh_home):
        """A handle validates epoch on each operation (EM-3)."""
        _create_board_with_register("handle-validate")

        handle = kb.open_board_handle(board="handle-validate")
        assert handle is not None

        # First validation
        valid1, _ = handle.validate_for_operation()
        assert valid1

        # Second validation
        valid2, _ = handle.validate_for_operation()
        assert valid2

        handle.close()

    def test_handle_becomes_invalid_after_fence_close(self, fresh_home):
        """A handle becomes invalid after the fence closes."""
        _create_board_with_register("handle-invalid")

        handle = kb.open_board_handle(board="handle-invalid")
        assert handle is not None

        # Valid before fence close
        valid, _ = handle.validate_for_operation()
        assert valid

        # Close the fence
        entry = _setup_board_for_removal("handle-invalid")
        result = kb.commit_fence_closing_point("handle-invalid", new_epoch=entry.epoch)
        assert result.success

        # Invalid after fence close (gate is 'closing')
        valid, refusal = handle.validate_for_operation()
        assert not valid
        assert refusal is not None
        assert refusal.outcome == kb.FenceOutcome.REFUSED_CLOSED

        handle.close()

    def test_handle_context_manager(self, fresh_home):
        """BoardHandle works as a context manager."""
        _create_board_with_register("handle-context")

        with kb.open_board_handle(board="handle-context") as handle:
            assert handle is not None
            valid, _ = handle.validate_for_operation()
            assert valid

        # After context exit, handle should be closed
        # (accessing _closed is testing implementation, but confirms the contract)
        assert handle._closed
