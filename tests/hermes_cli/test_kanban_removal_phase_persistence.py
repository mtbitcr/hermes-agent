"""The durable phase record (design revision 5, §6): schema and reads.

``board_removal_phase`` lives in the register store — outside every
board's storage — so it survives the board's own content being destroyed.
These tests never open a board store to read it back; they either drive
the real module API or read the register file directly with a plain
read-only connection.
"""

from __future__ import annotations

import sqlite3
import sys
import time
from pathlib import Path

import pytest

_WORKTREE = Path(__file__).resolve().parents[2]
if str(_WORKTREE) not in sys.path:
    sys.path.insert(0, str(_WORKTREE))

from hermes_cli import kanban_db as kb
from tests.hermes_cli._kanban_fence_support import (
    create_fenced_board,
    read_only,
    start_removal,
)


# ---------------------------------------------------------------------------
# No record yet (P1)
# ---------------------------------------------------------------------------

def test_get_removal_phase_record_is_none_when_nothing_recorded(fence_home):
    create_fenced_board("no-removal")
    assert kb.get_removal_phase_record("no-removal") is None


def test_get_removal_phase_record_is_none_for_unknown_board(fence_home):
    assert kb.get_removal_phase_record("never-existed") is None


# ---------------------------------------------------------------------------
# An OLD register file (pre-dating this table) picks it up on next open
# ---------------------------------------------------------------------------

_OLD_REGISTER_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS board_register (
    board_name              TEXT PRIMARY KEY,
    lifecycle               TEXT NOT NULL DEFAULT 'live',
    epoch                   INTEGER NOT NULL DEFAULT 1,
    epoch_before            INTEGER,
    gate_move               TEXT NOT NULL DEFAULT 'settled',
    ever_existed_marker     INTEGER NOT NULL DEFAULT 0,
    removal_mode            TEXT,
    scope_declaration_version TEXT,
    epoch_lineage           TEXT,
    created_at              INTEGER NOT NULL,
    updated_at              INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS board_removal_archive (
    board_name              TEXT NOT NULL,
    record_type             TEXT NOT NULL,
    record_id               TEXT NOT NULL,
    created_at              INTEGER NOT NULL,
    PRIMARY KEY (board_name, record_type, record_id)
);
CREATE INDEX IF NOT EXISTS idx_register_lifecycle ON board_register(lifecycle);
"""


def test_an_old_register_file_gains_the_phase_table_on_next_open(fence_home):
    path = kb.register_db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.executescript(_OLD_REGISTER_SCHEMA_SQL)
    conn.close()

    with read_only(path) as ro:
        names_before = {
            r[0] for r in ro.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
    assert "board_removal_phase" not in names_before

    # Force the module to treat this path as un-initialized (a fresh
    # process would not have the flag set either).
    kb._REGISTER_INITIALIZED = False
    kb._REGISTER_INITIALIZED_PATHS.clear()

    result = kb.get_removal_phase_record("some-board")
    assert result is None  # no row yet, but the table now exists

    with read_only(path) as ro:
        names_after = {
            r[0] for r in ro.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
    assert "board_removal_phase" in names_after


# ---------------------------------------------------------------------------
# RemovalPhaseRecord.from_row round-trips every column
# ---------------------------------------------------------------------------

def test_removal_phase_record_round_trips_through_intent(fence_home):
    create_fenced_board("roundtrip")
    result = start_removal("roundtrip", mode="permanent")
    assert result.success, result.message

    record = kb.get_removal_phase_record("roundtrip")
    assert record is not None
    assert record.board_name == "roundtrip"
    assert record.removal_id == result.removal_id
    assert record.mode == kb.RemovalMode.PERMANENT
    assert record.phase == kb.RemovalPhase.INTENT
    assert record.epoch == 2
    assert record.quiescence_deadline is None
    assert record.deadline_basis is None
    assert record.outcome is None
    assert record.created_at is not None
    assert record.updated_at is not None


def test_removal_phase_record_read_directly_off_disk_matches_the_api(fence_home):
    create_fenced_board("disk-check")
    result = start_removal("disk-check", mode="reversible")
    assert result.success

    with read_only(kb.register_db_path()) as ro:
        row = ro.execute(
            "SELECT * FROM board_removal_phase WHERE board_name = ?",
            ("disk-check",),
        ).fetchone()
    assert row is not None
    assert row["removal_id"] == result.removal_id
    assert row["mode"] == "reversible"
    assert row["phase"] == "intent"
    assert row["epoch"] == 2
