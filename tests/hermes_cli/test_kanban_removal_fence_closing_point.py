"""The fence-closing point, and the authority it is allowed to close on.

Reviewer reproduction 5: closing read the register authority BEFORE it
opened the board transaction and never revalidated it. Its compare-and-set
protected only the old Gate-B mirror value. So when Gate A moved in that
window from ``removing/epoch=2`` to ``live/epoch=3``, closing still
returned success and committed ``gate=closing, mirror=2`` — a fence
closed on an epoch that was no longer authoritative.

The closing primitive now runs under the SAME per-board register lock
every Gate A transition takes, and revalidates + compare-and-sets the
register row inside a register write transaction held open across the
Gate B commit.
"""

from __future__ import annotations

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
    begin_removal,
    gate_row,
    ready_task,
    register_row,
    task_row,
    write_register_row_behind_the_lock,
)


def _live_entry(slug: str, epoch: int) -> kb.RegisterEntry:
    now = int(time.time())
    return kb.RegisterEntry(
        board_name=slug,
        lifecycle=kb.BoardLifecycle.LIVE,
        epoch=epoch,
        epoch_before=None,
        gate_move=kb.GateMove.SETTLED,
        epoch_lineage=list(range(1, epoch + 1)),
        created_at=now,
        updated_at=now,
    )


# ---------------------------------------------------------------------------
# Reproduction 5 — closing cannot commit a stale epoch
# ---------------------------------------------------------------------------

def test_repro_5_closing_refuses_when_gate_a_moved_in_the_window(
    fence_home, monkeypatch
):
    """The exact race the reviewer ran.

    The register moves from ``removing/epoch=2`` to ``live/epoch=3``
    between the authority read and the Gate B commit. Before the repair
    this returned success and wrote ``gate=closing, mirror=2``.
    """
    create_fenced_board("stale-epoch")
    db_path = kb.kanban_db_path(board="stale-epoch")
    begin_removal("stale-epoch")
    assert register_row("stale-epoch")["epoch"] == 2
    assert gate_row(db_path) == ("open", 1)

    real = kb.expected_mirror_before_close
    fired: list[bool] = []

    def move_the_authority(entry):
        # Called between "read the authority" and "commit Gate B" — the
        # window reproduction 5 exploited. The removal is abandoned and
        # the board goes live again at a NEW epoch.
        if not fired:
            fired.append(True)
            write_register_row_behind_the_lock(_live_entry("stale-epoch", 3))
        return real(entry)

    monkeypatch.setattr(kb, "expected_mirror_before_close", move_the_authority)

    result = kb.commit_fence_closing_point("stale-epoch")

    assert fired == [True], "the race never fired — the test proves nothing"
    assert result.success is False
    assert result.transitioned is False
    assert "authority" in result.message.lower()
    # Gate B is untouched: no `closing`, and certainly no mirror=2.
    assert gate_row(db_path) == ("open", 1)
    assert register_row("stale-epoch") == {
        "lifecycle": "live",
        "epoch": 3,
        "epoch_before": None,
        "gate_move": "settled",
    }


def test_repro_5_the_board_stays_writable_after_the_refused_close(
    fence_home, monkeypatch
):
    """A refused close must leave the board exactly as it found it — the
    point of refusing is that the board is live at epoch 3, so work on it
    must carry on."""
    create_fenced_board("stale-then-live")
    db_path = kb.kanban_db_path(board="stale-then-live")
    conn = kb.connect(board="stale-then-live")
    task_id = ready_task(conn)
    begin_removal("stale-then-live")

    real = kb.expected_mirror_before_close
    fired: list[bool] = []

    def move_the_authority(entry):
        if not fired:
            fired.append(True)
            # Back to live, and the mirror is brought along, so the board
            # is a consistent live board at epoch 3.
            write_register_row_behind_the_lock(_live_entry("stale-then-live", 3))
            with kb._fence_protocol_scope():
                with kb.write_txn(conn):
                    conn.execute(
                        "UPDATE board_fence_state SET epoch_mirror = 3 WHERE id = 1"
                    )
        return real(entry)

    monkeypatch.setattr(kb, "expected_mirror_before_close", move_the_authority)
    assert kb.commit_fence_closing_point("stale-then-live").success is False

    assert gate_row(db_path) == ("open", 3)
    assert kb.claim_task(conn, task_id) is not None
    conn.close()
    assert task_row(db_path, task_id)["status"] == "running"


def test_repro_5_closing_takes_the_same_lock_gate_a_transitions_take(fence_home):
    """The revalidation is under the per-board register lock, so a
    transition attempted concurrently cannot interleave with it at all."""
    create_fenced_board("locked-close")
    begin_removal("locked-close")

    from hermes_cli.sqlite_util import InitLockUnavailable

    with kb.board_register_lock("locked-close", timeout_seconds=0.2):
        # Held: any Gate A transition, including the one closing performs,
        # fails closed rather than proceeding unserialised.
        with pytest.raises(InitLockUnavailable):
            kb.transition_register_entry(_live_entry("locked-close", 9))
        result = kb.commit_fence_closing_point("locked-close")

    assert result.success is False
    assert "lock" in result.message.lower()
    assert gate_row(kb.kanban_db_path(board="locked-close")) == ("open", 1)

    # Released: the same close now succeeds.
    assert kb.commit_fence_closing_point("locked-close").success is True


# ---------------------------------------------------------------------------
# Ordinary closing semantics
# ---------------------------------------------------------------------------

def test_the_closing_point_is_one_commit_of_gate_and_mirror(fence_home):
    create_fenced_board("closes")
    db_path = kb.kanban_db_path(board="closes")
    begin_removal("closes")

    result = kb.commit_fence_closing_point("closes")

    assert result.success is True
    assert result.transitioned is True
    assert result.new_epoch == 2
    assert gate_row(db_path) == ("closing", 2)


def test_only_one_racer_can_transition(fence_home):
    create_fenced_board("idempotent")
    begin_removal("idempotent")

    first = kb.commit_fence_closing_point("idempotent")
    second = kb.commit_fence_closing_point("idempotent")

    assert (first.success, first.transitioned) == (True, True)
    assert (second.success, second.transitioned) == (True, False)
    assert gate_row(kb.kanban_db_path(board="idempotent")) == ("closing", 2)


def test_the_epoch_is_derived_from_the_register_not_the_caller(fence_home):
    create_fenced_board("derived")
    db_path = kb.kanban_db_path(board="derived")
    begin_removal("derived")

    refused = kb.commit_fence_closing_point("derived", new_epoch=999)

    assert refused.success is False
    assert gate_row(db_path) == ("open", 1)

    accepted = kb.commit_fence_closing_point("derived", new_epoch=2)
    assert accepted.success is True
    assert gate_row(db_path) == ("closing", 2)


def test_a_drifted_mirror_refuses_instead_of_being_overwritten(fence_home):
    create_fenced_board("drifted")
    db_path = kb.kanban_db_path(board="drifted")
    conn = kb.connect(board="drifted")
    begin_removal("drifted")
    with kb._fence_protocol_scope():
        with kb.write_txn(conn):
            conn.execute("UPDATE board_fence_state SET epoch_mirror = 77 WHERE id = 1")
    conn.close()

    result = kb.commit_fence_closing_point("drifted")

    assert result.success is False
    assert "CAS failed" in result.message
    assert gate_row(db_path) == ("open", 77)


def test_closing_without_an_authority_is_refused(fence_home):
    create_fenced_board("no-entry")
    db_path = kb.kanban_db_path(board="no-entry")
    with kb.register_connect() as reg:
        reg.execute("BEGIN IMMEDIATE")
        reg.execute("DELETE FROM board_register WHERE board_name = ?", ("no-entry",))
        reg.execute("COMMIT")

    result = kb.commit_fence_closing_point("no-entry")

    assert result.success is False
    assert "authority" in result.message
    assert gate_row(db_path) == ("open", 1)


def test_closing_a_board_with_no_storage_is_refused(fence_home):
    create_fenced_board("storageless")
    db_path = kb.kanban_db_path(board="storageless")
    begin_removal("storageless")
    kb._INITIALIZED_PATHS.clear()
    for leftover in list(db_path.parent.glob("kanban.db*")):
        leftover.unlink()

    result = kb.commit_fence_closing_point("storageless")

    assert result.success is False
    assert "storage" in result.message
    assert not db_path.exists()


def test_the_intent_to_fence_window_refuses_nothing(fence_home):
    """EM-4b / EM-5: between the register intent and the fence-closing
    point the board keeps working — the mirror lags by exactly one epoch
    and that lag is expected, not indeterminate."""
    create_fenced_board("window")
    db_path = kb.kanban_db_path(board="window")
    conn = kb.connect(board="window")
    task_id = ready_task(conn)

    begin_removal("window")
    assert register_row("window")["epoch"] == 2
    assert gate_row(db_path) == ("open", 1)

    # Inside the window: work still lands.
    assert kb.claim_task(conn, task_id) is not None
    assert task_row(db_path, task_id)["status"] == "running"

    # After the closing point: it does not.
    assert kb.commit_fence_closing_point("window").success is True
    with pytest.raises(kb.BoardFenceClosedError):
        kb.add_comment(conn, task_id, "a", "b")
    conn.close()
