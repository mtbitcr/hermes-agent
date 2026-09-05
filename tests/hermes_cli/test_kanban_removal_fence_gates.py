"""Tests for the board removal fence Gate A admission rules.

Covers:
- Gate A's full admission table (every row, every action)
- GA-4/GA-4a/GA-4b (open never creates; incomplete board refused; distinct primitives)
- GA-5's five conditions each proven necessary
- GA-6a/GA-6b (archive lookup seam)
"""

from __future__ import annotations

import os
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


def _create_complete_board(board: str, home: Path) -> Path:
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
    """Create a complete board WITH a register entry for testing fence primitives."""
    # Create the board through create_board_store which handles both
    conn = kb.create_board_store(board)
    if conn:
        conn.close()
        return kb.get_register_entry(board)

    # Fallback: create manually
    kb.create_board(board)
    with kb.register_connect() as reg_conn:
        now = int(time.time())
        entry = kb.RegisterEntry(
            board_name=board,
            lifecycle=kb.BoardLifecycle.LIVE,
            epoch=1,
            epoch_lineage=[1],
            created_at=now,
            updated_at=now,
        )
        reg_conn.execute("BEGIN IMMEDIATE")
        kb._write_register_entry(reg_conn, entry)
        reg_conn.execute("COMMIT")

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


def _create_incomplete_board(board: str, home: Path) -> Path:
    """Create a board DB file without complete schema."""
    db_path = kb.kanban_db_path(board=board)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    # Create a minimal table that is NOT the full schema
    conn.execute("CREATE TABLE IF NOT EXISTS incomplete_marker (id INTEGER)")
    conn.close()
    return db_path


# ---------------------------------------------------------------------------
# Gate A Admission Table Tests
# ---------------------------------------------------------------------------

class TestGateAAdmissionTable:
    """Test every row and action in the Gate A admission table (§3.2)."""

    def test_live_entry_admits_open(self, fresh_home):
        """lifecycle=live admits open action."""
        _create_board_with_register("live-test")
        result = kb.check_gate_a_admission("live-test", kb.GateAAction.OPEN, storage_exists=True)
        assert result.admitted
        assert "live" in result.message.lower()

    def test_live_entry_admits_create(self, fresh_home):
        """lifecycle=live admits create action (deliberate creation only)."""
        with kb.register_connect() as conn:
            now = int(time.time())
            entry = kb.RegisterEntry(
                board_name="live-create",
                lifecycle=kb.BoardLifecycle.LIVE,
                epoch=1,
                created_at=now,
                updated_at=now,
            )
            conn.execute("BEGIN IMMEDIATE")
            kb._write_register_entry(conn, entry)
            conn.execute("COMMIT")

        result = kb.check_gate_a_admission("live-create", kb.GateAAction.CREATE, storage_exists=False)
        assert result.admitted

    def test_live_entry_admits_serve(self, fresh_home):
        """lifecycle=live admits serve action."""
        _create_board_with_register("serve-test")
        result = kb.check_gate_a_admission("serve-test", kb.GateAAction.SERVE, storage_exists=True)
        assert result.admitted

    def test_live_entry_admits_begin_removal(self, fresh_home):
        """lifecycle=live admits begin_removal action."""
        _create_board_with_register("removal-test")
        result = kb.check_gate_a_admission("removal-test", kb.GateAAction.BEGIN_REMOVAL, storage_exists=True)
        assert result.admitted

    def test_removing_entry_admits_open_ga1(self, fresh_home):
        """GA-1: lifecycle=removing admits open action."""
        with kb.register_connect() as conn:
            now = int(time.time())
            entry = kb.RegisterEntry(
                board_name="removing-open",
                lifecycle=kb.BoardLifecycle.REMOVING,
                epoch=2,
                epoch_before=1,
                gate_move=kb.GateMove.PENDING,
                created_at=now,
                updated_at=now,
            )
            conn.execute("BEGIN IMMEDIATE")
            kb._write_register_entry(conn, entry)
            conn.execute("COMMIT")

        _create_complete_board("removing-open", fresh_home)
        result = kb.check_gate_a_admission("removing-open", kb.GateAAction.OPEN, storage_exists=True)
        assert result.admitted
        assert "GA-1" in result.message or "removing admits" in result.message.lower()

    def test_removing_entry_refuses_create_ga2(self, fresh_home):
        """GA-2: lifecycle=removing refuses create action."""
        with kb.register_connect() as conn:
            now = int(time.time())
            entry = kb.RegisterEntry(
                board_name="removing-create",
                lifecycle=kb.BoardLifecycle.REMOVING,
                epoch=2,
                created_at=now,
                updated_at=now,
            )
            conn.execute("BEGIN IMMEDIATE")
            kb._write_register_entry(conn, entry)
            conn.execute("COMMIT")

        result = kb.check_gate_a_admission("removing-create", kb.GateAAction.CREATE, storage_exists=False)
        assert not result.admitted
        assert result.rule == kb.FenceRefusalRule.GA_2
        assert "GA-2" in result.message or "creation" in result.message.lower()

    def test_removing_entry_admits_serve(self, fresh_home):
        """lifecycle=removing admits serve action."""
        with kb.register_connect() as conn:
            now = int(time.time())
            entry = kb.RegisterEntry(
                board_name="removing-serve",
                lifecycle=kb.BoardLifecycle.REMOVING,
                epoch=2,
                created_at=now,
                updated_at=now,
            )
            conn.execute("BEGIN IMMEDIATE")
            kb._write_register_entry(conn, entry)
            conn.execute("COMMIT")

        _create_complete_board("removing-serve", fresh_home)
        result = kb.check_gate_a_admission("removing-serve", kb.GateAAction.SERVE, storage_exists=True)
        assert result.admitted

    def test_removing_entry_refuses_begin_removal(self, fresh_home):
        """lifecycle=removing refuses new begin_removal action."""
        with kb.register_connect() as conn:
            now = int(time.time())
            entry = kb.RegisterEntry(
                board_name="removing-removal",
                lifecycle=kb.BoardLifecycle.REMOVING,
                epoch=2,
                created_at=now,
                updated_at=now,
            )
            conn.execute("BEGIN IMMEDIATE")
            kb._write_register_entry(conn, entry)
            conn.execute("COMMIT")

        result = kb.check_gate_a_admission("removing-removal", kb.GateAAction.BEGIN_REMOVAL, storage_exists=True)
        assert not result.admitted
        # Join-or-refuse is out of scope; just refuse here

    def test_archived_entry_refuses_all(self, fresh_home):
        """lifecycle=archived refuses all actions."""
        with kb.register_connect() as conn:
            now = int(time.time())
            entry = kb.RegisterEntry(
                board_name="archived-test",
                lifecycle=kb.BoardLifecycle.ARCHIVED,
                epoch=1,
                created_at=now,
                updated_at=now,
            )
            conn.execute("BEGIN IMMEDIATE")
            kb._write_register_entry(conn, entry)
            conn.execute("COMMIT")

        for action in kb.GateAAction:
            result = kb.check_gate_a_admission("archived-test", action, storage_exists=True)
            assert not result.admitted
            assert result.outcome == kb.FenceOutcome.REFUSED_CLOSED

    def test_hard_removed_entry_refuses_all(self, fresh_home):
        """lifecycle=hard-removed refuses all actions."""
        with kb.register_connect() as conn:
            now = int(time.time())
            entry = kb.RegisterEntry(
                board_name="hardremoved-test",
                lifecycle=kb.BoardLifecycle.HARD_REMOVED,
                epoch=1,
                created_at=now,
                updated_at=now,
            )
            conn.execute("BEGIN IMMEDIATE")
            kb._write_register_entry(conn, entry)
            conn.execute("COMMIT")

        for action in kb.GateAAction:
            result = kb.check_gate_a_admission("hardremoved-test", action, storage_exists=True)
            assert not result.admitted
            assert result.outcome == kb.FenceOutcome.REFUSED_CLOSED

    def test_absent_entry_no_storage_admits_deliberate_create(self, fresh_home):
        """Absent entry, no storage → admit only deliberate creation."""
        result = kb.check_gate_a_admission("never-existed", kb.GateAAction.CREATE, storage_exists=False)
        assert result.admitted

        # Other actions refused
        for action in (kb.GateAAction.OPEN, kb.GateAAction.SERVE, kb.GateAAction.BEGIN_REMOVAL):
            result = kb.check_gate_a_admission("never-existed", action, storage_exists=False)
            assert not result.admitted


# ---------------------------------------------------------------------------
# GA-4: Open Never Creates Tests
# ---------------------------------------------------------------------------

class TestGA4OpenNeverCreates:
    """Test GA-4: opening never creates, in every access mode."""

    def test_open_board_store_does_not_create_directory(self, fresh_home):
        """GA-4: open_board_store never creates a containing directory."""
        result = kb.open_board_store(board="nonexistent")
        assert result is None
        # The directory should NOT have been created
        db_path = kb.kanban_db_path(board="nonexistent")
        assert not db_path.parent.exists()

    def test_open_board_store_does_not_create_db_file(self, fresh_home):
        """GA-4: open_board_store never creates a store."""
        # Create the parent directory but not the DB
        db_path = kb.kanban_db_path(board="partial")
        db_path.parent.mkdir(parents=True, exist_ok=True)

        result = kb.open_board_store(board="partial")
        assert result is None
        assert not db_path.exists()

    def test_open_board_store_refuses_empty_file(self, fresh_home):
        """GA-4: empty DB file is treated as absent."""
        db_path = kb.kanban_db_path(board="empty")
        db_path.parent.mkdir(parents=True, exist_ok=True)
        db_path.touch()  # Create empty file

        result = kb.open_board_store(board="empty")
        assert result is None

    def test_open_board_store_refuses_incomplete_schema_ga4a(self, fresh_home):
        """GA-4a: incomplete board (no complete schema) is refused."""
        _create_incomplete_board("incomplete", fresh_home)

        result = kb.open_board_store(board="incomplete")
        assert result is None

    def test_open_and_create_are_distinct_primitives_ga4b(self, fresh_home):
        """GA-4b: open and create are two distinct primitives, neither reachable through the other."""
        # open_board_store is the open primitive
        # create_board_store is the create primitive

        # Calling open on non-existent does NOT create
        result = kb.open_board_store(board="distinct-test")
        assert result is None
        db_path = kb.kanban_db_path(board="distinct-test")
        assert not db_path.exists()

        # Only create_board_store creates
        conn = kb.create_board_store(board="distinct-test")
        assert conn is not None
        conn.close()
        assert db_path.exists()

    def test_open_succeeds_for_complete_board(self, fresh_home):
        """open_board_store succeeds for a complete board with valid entry."""
        # Create a complete board with register entry through the proper channel
        _create_board_with_register("complete-test")

        # Now open should work
        conn = kb.open_board_store(board="complete-test")
        assert conn is not None
        conn.close()


# ---------------------------------------------------------------------------
# GA-5: One-Time Backfill Tests
# ---------------------------------------------------------------------------

class TestGA5Backfill:
    """Test GA-5's five conditions, each proven necessary."""

    def test_backfill_requires_named_operation(self, fresh_home):
        """GA-5 condition 1: backfill is a named operation ordinary paths cannot invoke."""
        # Create complete storage without register entry
        _create_complete_board("backfill-named", fresh_home)

        # Regular open does NOT backfill
        regular_result = kb.check_gate_a_admission("backfill-named", kb.GateAAction.OPEN, storage_exists=True, is_backfill=False)
        assert not regular_result.admitted

        # Only is_backfill=True can backfill
        backfill_result = kb.check_gate_a_admission("backfill-named", kb.GateAAction.OPEN, storage_exists=True, storage_complete=True, is_backfill=True)
        assert backfill_result.admitted

    def test_backfill_is_recorded(self, fresh_home):
        """GA-5 condition 2: backfill is recorded with time and subject."""
        _create_complete_board("backfill-record", fresh_home)

        result = kb.backfill_register_entry("backfill-record")
        assert result.success

        # Check the backfill was recorded in the archive
        with kb.register_connect() as conn:
            row = conn.execute(
                "SELECT * FROM board_removal_archive WHERE board_name = ? AND record_type = 'backfill'",
                ("backfill-record",),
            ).fetchone()
            assert row is not None
            assert row["created_at"] is not None

    def test_backfill_fails_if_marker_set(self, fresh_home):
        """GA-5 condition 3: ever-existed marker must be unset."""
        _create_complete_board("backfill-marker", fresh_home)

        # Manually set the marker
        with kb.register_connect() as conn:
            now = int(time.time())
            conn.execute("BEGIN IMMEDIATE")
            conn.execute(
                """
                INSERT INTO board_register (board_name, lifecycle, epoch, ever_existed_marker, created_at, updated_at, gate_move)
                VALUES (?, 'live', 1, 1, ?, ?, 'settled')
                """,
                ("backfill-marker", now, now),
            )
            conn.execute("COMMIT")

        result = kb.backfill_register_entry("backfill-marker")
        assert not result.success
        assert "marker" in result.message.lower()

    def test_backfill_fails_if_storage_incomplete_ga4a(self, fresh_home):
        """GA-5 condition 4: storage must be complete (GA-4a)."""
        _create_incomplete_board("backfill-incomplete", fresh_home)

        result = kb.backfill_register_entry("backfill-incomplete")
        assert not result.success
        assert "incomplete" in result.message.lower() or "GA-4a" in result.message

    def test_backfill_fails_if_archive_record_exists_ga6a(self, fresh_home):
        """GA-5 condition 5: no receipt/audit record may exist (GA-6a)."""
        _create_complete_board("backfill-archive", fresh_home)

        # Add an archive record
        with kb.register_connect() as conn:
            now = int(time.time())
            conn.execute("BEGIN IMMEDIATE")
            conn.execute(
                """
                INSERT INTO board_removal_archive (board_name, record_type, record_id, created_at)
                VALUES (?, 'audit', 'test-audit-1', ?)
                """,
                ("backfill-archive", now),
            )
            conn.execute("COMMIT")

        result = kb.backfill_register_entry("backfill-archive")
        assert not result.success
        assert "archive" in result.message.lower() or "GA-6a" in result.message

    def test_backfill_succeeds_when_all_conditions_met(self, fresh_home):
        """GA-5: backfill succeeds when all five conditions are met."""
        _create_complete_board("backfill-success", fresh_home)

        result = kb.backfill_register_entry("backfill-success")
        assert result.success
        assert result.entry is not None
        assert result.entry.lifecycle == kb.BoardLifecycle.LIVE

    def test_backfill_writes_epoch_mirror(self, fresh_home):
        """GA-5: backfill writes the in-board epoch mirror in the same act."""
        _create_complete_board("backfill-mirror", fresh_home)

        result = kb.backfill_register_entry("backfill-mirror")
        assert result.success

        # Verify the epoch mirror was written
        db_path = kb.kanban_db_path(board="backfill-mirror")
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT epoch_mirror FROM board_fence_state WHERE id = 1").fetchone()
        conn.close()
        assert row is not None
        assert row["epoch_mirror"] == 1


# ---------------------------------------------------------------------------
# GA-6: Archive Lookup Seam Tests
# ---------------------------------------------------------------------------

class TestGA6ArchiveLookup:
    """Test GA-6a/GA-6b: archive lookup seam."""

    def test_ga6a_no_entry_no_marker_archive_exists_is_indeterminate(self, fresh_home):
        """GA-6a: absent entry, no marker, but archive record exists → indeterminate."""
        # Create complete storage
        _create_complete_board("ga6a-test", fresh_home)

        # Add archive record WITHOUT register entry
        with kb.register_connect() as conn:
            now = int(time.time())
            conn.execute("BEGIN IMMEDIATE")
            conn.execute(
                """
                INSERT INTO board_removal_archive (board_name, record_type, record_id, created_at)
                VALUES (?, 'receipt', 'removal-receipt-1', ?)
                """,
                ("ga6a-test", now),
            )
            conn.execute("COMMIT")

        result = kb.check_gate_a_admission("ga6a-test", kb.GateAAction.OPEN, storage_exists=True)
        assert not result.admitted
        assert result.rule == kb.FenceRefusalRule.GA_6a

    def test_ga6b_archive_lookup_failure_refuses(self, fresh_home, monkeypatch):
        """GA-6b: archive lookup failure results in refusal."""
        _create_complete_board("ga6b-test", fresh_home)

        # Mock archive lookup to fail
        def fail_lookup(*args, **kwargs):
            return None  # None indicates lookup failure

        monkeypatch.setattr(kb, "has_removal_archive_record", fail_lookup)

        result = kb.check_gate_a_admission("ga6b-test", kb.GateAAction.OPEN, storage_exists=True)
        assert not result.admitted
        assert result.rule == kb.FenceRefusalRule.GA_6b

    def test_has_removal_archive_record_returns_true_when_exists(self, fresh_home):
        """has_removal_archive_record returns True when records exist."""
        with kb.register_connect() as conn:
            now = int(time.time())
            conn.execute("BEGIN IMMEDIATE")
            conn.execute(
                """
                INSERT INTO board_removal_archive (board_name, record_type, record_id, created_at)
                VALUES (?, 'receipt', 'test-1', ?)
                """,
                ("archive-exists", now),
            )
            conn.execute("COMMIT")

        result = kb.has_removal_archive_record("archive-exists")
        assert result is True

    def test_has_removal_archive_record_returns_false_when_empty(self, fresh_home):
        """has_removal_archive_record returns False when no records exist."""
        result = kb.has_removal_archive_record("no-archive")
        assert result is False
