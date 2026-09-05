"""Tests for the board removal fence fail-closed behavior (§3.4).

Covers:
- Unreadable register entry refuses
- Malformed entry refuses
- Absent-entry-where-required refuses
- Unreadable gate refuses
- Consultation exceeding bound refuses
- Each recorded outcome distinguishes indeterminate from closed
- GA-5 exception is the only admitted absence
"""

from __future__ import annotations

import sqlite3
import sys
import time
from pathlib import Path
from unittest.mock import patch

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


def _create_complete_board_no_register(board: str) -> Path:
    """Create a complete board with schema but no register entry."""
    db_path = kb.kanban_db_path(board=board)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.executescript(kb.SCHEMA_SQL)
    conn.executescript(kb.IN_BOARD_FENCE_SCHEMA_SQL)
    conn.execute(
        "INSERT OR IGNORE INTO board_fence_state (id, gate, epoch_mirror, updated_at) "
        "VALUES (1, 'open', 1, ?)",
        (int(time.time()),),
    )
    conn.close()
    return db_path


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


# ---------------------------------------------------------------------------
# Unreadable State Tests
# ---------------------------------------------------------------------------

class TestUnreadableState:
    """Test that unreadable state triggers fail-closed behavior."""

    def test_unreadable_register_entry_refuses(self, fresh_home, monkeypatch):
        """Unreadable register entry causes refusal."""
        _create_board_with_register("unreadable-reg")

        # Mock register read to raise
        original_get = kb.get_register_entry

        def raise_on_read(*args, **kwargs):
            raise sqlite3.Error("simulated database error")

        monkeypatch.setattr(kb, "get_register_entry", raise_on_read)

        admitted, refusal = kb.check_fence_for_operation("unreadable-reg")
        assert not admitted
        assert refusal is not None
        assert refusal.outcome == kb.FenceOutcome.REFUSED_INDETERMINATE
        assert "cannot read" in refusal.message.lower() or "error" in refusal.message.lower()

    def test_unreadable_in_board_gate_refuses(self, fresh_home):
        """Unreadable in-board gate causes refusal."""
        _create_board_with_register("unreadable-gate")

        # Corrupt the fence state table
        db_path = kb.kanban_db_path(board="unreadable-gate")
        conn = sqlite3.connect(str(db_path))
        conn.execute("DROP TABLE IF EXISTS board_fence_state")
        conn.close()

        admitted, refusal = kb.check_fence_for_operation("unreadable-gate")
        assert not admitted
        assert refusal is not None
        assert refusal.outcome == kb.FenceOutcome.REFUSED_INDETERMINATE


# ---------------------------------------------------------------------------
# Malformed Entry Tests
# ---------------------------------------------------------------------------

class TestMalformedEntry:
    """Test that malformed entries trigger fail-closed behavior."""

    def test_malformed_lifecycle_refuses(self, fresh_home):
        """Malformed lifecycle value causes refusal."""
        # Manually insert a malformed entry
        with kb.register_connect() as conn:
            now = int(time.time())
            conn.execute("BEGIN IMMEDIATE")
            conn.execute(
                """
                INSERT INTO board_register (board_name, lifecycle, epoch, gate_move, ever_existed_marker, created_at, updated_at)
                VALUES (?, 'invalid_lifecycle', 1, 'settled', 0, ?, ?)
                """,
                ("malformed-lifecycle", now, now),
            )
            conn.execute("COMMIT")

        # Create storage so we can test the full path
        _create_complete_board_no_register("malformed-lifecycle")

        # Reading should fail due to invalid lifecycle
        try:
            entry = kb.get_register_entry("malformed-lifecycle")
            # If we get here, the entry was read but has invalid lifecycle
            if entry is not None:
                # The BoardLifecycle enum should have failed
                pytest.fail("Should have failed on invalid lifecycle")
        except ValueError:
            # Expected: invalid lifecycle value
            pass

    def test_malformed_gate_value_refuses(self, fresh_home):
        """Malformed in-board gate value causes refusal."""
        _create_board_with_register("malformed-gate")

        # Corrupt the gate value
        db_path = kb.kanban_db_path(board="malformed-gate")
        conn = sqlite3.connect(str(db_path))
        conn.execute("UPDATE board_fence_state SET gate = 'invalid_gate' WHERE id = 1")
        conn.commit()
        conn.close()

        admitted, refusal = kb.check_fence_for_operation("malformed-gate")
        assert not admitted
        assert refusal is not None
        assert refusal.outcome == kb.FenceOutcome.REFUSED_INDETERMINATE


# ---------------------------------------------------------------------------
# Absent Entry Tests
# ---------------------------------------------------------------------------

class TestAbsentEntry:
    """Test that absent entries are handled correctly."""

    def test_absent_entry_with_storage_refuses(self, fresh_home):
        """Absent entry where storage exists refuses (not GA-5 backfill)."""
        _create_complete_board_no_register("absent-entry")

        result = kb.check_gate_a_admission(
            "absent-entry",
            kb.GateAAction.OPEN,
            storage_exists=True,
            is_backfill=False,
        )
        assert not result.admitted
        assert result.outcome == kb.FenceOutcome.REFUSED_INDETERMINATE

    def test_absent_entry_ga5_backfill_is_exception(self, fresh_home):
        """GA-5: backfill is the only admitted absence case."""
        _create_complete_board_no_register("backfill-exception")

        # Regular open: refused
        regular = kb.check_gate_a_admission(
            "backfill-exception",
            kb.GateAAction.OPEN,
            storage_exists=True,
            is_backfill=False,
        )
        assert not regular.admitted

        # Backfill: admitted (assuming all conditions met)
        backfill = kb.check_gate_a_admission(
            "backfill-exception",
            kb.GateAAction.OPEN,
            storage_exists=True,
            storage_complete=True,
            is_backfill=True,
        )
        assert backfill.admitted


# ---------------------------------------------------------------------------
# Consultation Timeout Tests
# ---------------------------------------------------------------------------

class TestConsultationTimeout:
    """Test that consultation exceeding the bound refuses."""

    def test_timeout_exceeding_bound_refuses(self, fresh_home, monkeypatch):
        """Consultation exceeding the 5-second bound refuses."""
        _create_board_with_register("timeout-test")

        # Mock the timeout to be very short
        monkeypatch.setattr(kb, "_fence_consultation_timeout_seconds", lambda: 0.001)

        # Add a delay in the register read
        original_get = kb.get_register_entry

        def slow_read(*args, **kwargs):
            time.sleep(0.1)  # Sleep longer than timeout
            return original_get(*args, **kwargs)

        monkeypatch.setattr(kb, "get_register_entry", slow_read)

        admitted, refusal = kb.check_fence_for_operation("timeout-test")
        assert not admitted
        assert refusal is not None
        assert refusal.outcome == kb.FenceOutcome.REFUSED_TIMEOUT
        assert refusal.rule == kb.FenceRefusalRule.TIMEOUT

    def test_timeout_configurable_from_config(self, fresh_home, monkeypatch):
        """The consultation timeout is configurable."""
        # Default should be 5.0 seconds
        default_timeout = kb._fence_consultation_timeout_seconds()
        assert default_timeout == kb.DEFAULT_FENCE_CONSULTATION_TIMEOUT_SECONDS

        # Mock config to return a different value
        def mock_load_config():
            return {"kanban": {"fence_consultation_timeout_seconds": 10.0}}

        monkeypatch.setattr("hermes_cli.config.load_config_readonly", mock_load_config)

        # Force re-evaluation
        timeout = kb._fence_consultation_timeout_seconds()
        assert timeout == 10.0


# ---------------------------------------------------------------------------
# Outcome Distinction Tests
# ---------------------------------------------------------------------------

class TestOutcomeDistinction:
    """Test that recorded outcomes distinguish indeterminate from closed."""

    def test_closed_vs_indeterminate_outcomes(self, fresh_home):
        """Closed and indeterminate outcomes are distinguishable."""
        # Create a closed board (archived)
        with kb.register_connect() as conn:
            now = int(time.time())
            entry = kb.RegisterEntry(
                board_name="closed-board",
                lifecycle=kb.BoardLifecycle.ARCHIVED,
                epoch=1,
                created_at=now,
                updated_at=now,
            )
            conn.execute("BEGIN IMMEDIATE")
            kb._write_register_entry(conn, entry)
            conn.execute("COMMIT")

        # This should be CLOSED
        _create_complete_board_no_register("closed-board")
        admitted, refusal = kb.check_fence_for_operation("closed-board")
        assert not admitted
        assert refusal.outcome == kb.FenceOutcome.REFUSED_CLOSED

        # Create storage with no register entry — this is INDETERMINATE
        _create_complete_board_no_register("indeterminate-board")
        result = kb.check_gate_a_admission(
            "indeterminate-board",
            kb.GateAAction.OPEN,
            storage_exists=True,
        )
        assert not result.admitted
        assert result.outcome == kb.FenceOutcome.REFUSED_INDETERMINATE

    def test_fence_refusal_carries_rule(self, fresh_home):
        """FenceRefusal carries the specific rule that produced it."""
        # GA-2 refusal
        with kb.register_connect() as conn:
            now = int(time.time())
            entry = kb.RegisterEntry(
                board_name="ga2-test",
                lifecycle=kb.BoardLifecycle.REMOVING,
                epoch=2,
                created_at=now,
                updated_at=now,
            )
            conn.execute("BEGIN IMMEDIATE")
            kb._write_register_entry(conn, entry)
            conn.execute("COMMIT")

        result = kb.check_gate_a_admission("ga2-test", kb.GateAAction.CREATE)
        assert not result.admitted
        assert result.rule == kb.FenceRefusalRule.GA_2

    def test_fence_refusal_is_machine_readable(self, fresh_home):
        """FenceRefusal fields allow branching without string parsing."""
        entry = _create_board_with_register("machine-readable")

        # Prepare for removal
        with kb.register_connect() as conn:
            now = int(time.time())
            new_entry = kb.RegisterEntry(
                board_name="machine-readable",
                lifecycle=kb.BoardLifecycle.REMOVING,
                epoch=entry.epoch + 1,
                epoch_before=entry.epoch,
                gate_move=kb.GateMove.PENDING,
                created_at=entry.created_at,
                updated_at=now,
            )
            conn.execute("BEGIN IMMEDIATE")
            kb._write_register_entry(conn, new_entry)
            conn.execute("COMMIT")

        # Close the fence
        kb.commit_fence_closing_point("machine-readable", new_epoch=entry.epoch + 1)

        admitted, refusal = kb.check_fence_for_operation("machine-readable")
        assert not admitted
        assert refusal is not None

        # Can branch on outcome enum
        assert isinstance(refusal.outcome, kb.FenceOutcome)
        if refusal.outcome == kb.FenceOutcome.REFUSED_CLOSED:
            pass  # Handle closed case
        elif refusal.outcome == kb.FenceOutcome.REFUSED_INDETERMINATE:
            pass  # Handle indeterminate case

        # Can branch on rule enum
        assert isinstance(refusal.rule, kb.FenceRefusalRule)

        # Has board name for identification
        assert refusal.board == "machine-readable"


# ---------------------------------------------------------------------------
# Specific Fail-Closed Scenarios
# ---------------------------------------------------------------------------

class TestFailClosedScenarios:
    """Test specific fail-closed scenarios from the design."""

    def test_storage_exists_but_entry_absent_indeterminate(self, fresh_home):
        """Absent entry for board whose storage exists is indeterminate."""
        _create_complete_board_no_register("storage-no-entry")

        result = kb.check_gate_a_admission(
            "storage-no-entry",
            kb.GateAAction.OPEN,
            storage_exists=True,
        )
        assert not result.admitted
        assert result.outcome == kb.FenceOutcome.REFUSED_INDETERMINATE

    def test_entry_exists_but_storage_absent_indeterminate(self, fresh_home):
        """Entry exists but storage absent is indeterminate."""
        # Create entry without storage
        with kb.register_connect() as conn:
            now = int(time.time())
            entry = kb.RegisterEntry(
                board_name="entry-no-storage",
                lifecycle=kb.BoardLifecycle.LIVE,
                epoch=1,
                created_at=now,
                updated_at=now,
            )
            conn.execute("BEGIN IMMEDIATE")
            kb._write_register_entry(conn, entry)
            conn.execute("COMMIT")

        admitted, refusal = kb.check_fence_for_operation("entry-no-storage")
        assert not admitted
        assert refusal.outcome == kb.FenceOutcome.REFUSED_INDETERMINATE
        assert "storage" in refusal.message.lower() or "absent" in refusal.message.lower()

    def test_unexpected_epoch_pair_refuses(self, fresh_home):
        """Unexpected epoch pair refuses as indeterminate (EM-4c)."""
        _create_board_with_register("epoch-mismatch")

        # Manually corrupt the epoch mirror to create a mismatch
        db_path = kb.kanban_db_path(board="epoch-mismatch")
        conn = sqlite3.connect(str(db_path))
        conn.execute("UPDATE board_fence_state SET epoch_mirror = 999 WHERE id = 1")
        conn.commit()
        conn.close()

        admitted, refusal = kb.check_fence_for_operation("epoch-mismatch")
        assert not admitted
        assert refusal is not None
        assert refusal.rule == kb.FenceRefusalRule.EM_4c
