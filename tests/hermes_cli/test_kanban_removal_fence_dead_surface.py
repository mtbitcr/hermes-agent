"""Dead surface from removed features must be deleted (Finding 3)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_WORKTREE = Path(__file__).resolve().parents[2]
if str(_WORKTREE) not in sys.path:
    sys.path.insert(0, str(_WORKTREE))

from hermes_cli import kanban_db as kb
from tests.hermes_cli._kanban_fence_support import (
    create_fenced_board,
    register_row,
)


def test_dead_functions_are_absent():
    """The leftovers from removed features must be deleted."""
    # These names must not exist in the module namespace.
    assert not hasattr(kb, 'board_register_lock_held')
    assert not hasattr(kb, '_publish_first_register_entry_locked')


def test_surviving_register_machinery_is_functional(fence_home):
    """The backfill/register machinery that does have callers must still work."""
    # backfill_register_entry is the real recorded migration.
    kb.create_board("test-board")
    result = kb.backfill_register_entry("test-board")
    assert result.success

    # get_register_entry reads the authority.
    entry = kb.get_register_entry("test-board")
    assert entry is not None
    assert entry.board_name == "test-board"
    assert entry.lifecycle == kb.BoardLifecycle.LIVE

    # transition_register_entry updates the authority.
    new_entry = kb.RegisterEntry(
        board_name="test-board",
        lifecycle=kb.BoardLifecycle.REMOVING,
        epoch=entry.epoch + 1,
        epoch_before=entry.epoch,
        gate_move=kb.GateMove.PENDING,
        epoch_lineage=(entry.epoch_lineage or []) + [entry.epoch + 1],
        created_at=entry.created_at,
        updated_at=entry.updated_at,
    )
    kb.transition_register_entry(new_entry)

    # The register row reflects the transition.
    row = register_row("test-board")
    assert row is not None
    assert row["lifecycle"] == "removing"
    assert row["epoch"] == entry.epoch + 1
