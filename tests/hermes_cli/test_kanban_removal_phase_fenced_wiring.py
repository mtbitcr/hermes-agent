"""Phase 2 — Fenced (design revision 5, §6.2): wiring the EXISTING
fence-closing point into the phase sequence.

``advance_removal_to_fenced`` must drive Gate B through the real
:func:`kanban_db.commit_fence_closing_point` — never write Gate B itself,
never re-implement the mirror write. These tests assert DURABLE STATE
(the gate row on disk, a real fenced operation refusing), never that a
mock was called.
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
    gate_row,
    ready_task,
    register_row,
    start_removal,
)


def _intent(slug: str, mode: str = "reversible") -> str:
    create_fenced_board(slug)
    result = start_removal(slug, mode=mode)
    assert result.success, result.message
    return result.removal_id


# ---------------------------------------------------------------------------
# The real fence really closes, and a real fenced operation really refuses
# ---------------------------------------------------------------------------

def test_advancing_to_fenced_really_closes_gate_b_at_the_authoritative_epoch(fence_home):
    removal_id = _intent("closes-for-real")
    db_path = kb.kanban_db_path(board="closes-for-real")

    result = kb.advance_removal_to_fenced("closes-for-real", removal_id=removal_id)

    assert result.success is True
    assert result.transitioned is True
    assert gate_row(db_path) == ("closing", 2)
    assert register_row("closes-for-real") == {
        "lifecycle": "removing", "epoch": 2, "epoch_before": 1, "gate_move": "settled",
    }
    record = kb.get_removal_phase_record("closes-for-real")
    assert record.phase == kb.RemovalPhase.FENCED


def test_a_real_fenced_operation_is_really_refused_after_fencing(fence_home):
    removal_id = _intent("refuses-for-real")
    conn = kb.connect(board="refuses-for-real")
    task_id = ready_task(conn)

    result = kb.advance_removal_to_fenced("refuses-for-real", removal_id=removal_id)
    assert result.success

    with pytest.raises(kb.BoardFenceClosedError) as excinfo:
        kb.claim_task(conn, task_id)
    assert excinfo.value.refusal.outcome is kb.FenceOutcome.REFUSED_CLOSED
    conn.close()


# ---------------------------------------------------------------------------
# The deadline is computed once, never recomputed
# ---------------------------------------------------------------------------

def test_the_deadline_is_computed_once_a_repeat_call_leaves_it_unchanged(fence_home, monkeypatch):
    removal_id = _intent("deadline-once")

    first = kb.advance_removal_to_fenced("deadline-once", removal_id=removal_id)
    assert first.success
    first_deadline = first.record.quiescence_deadline
    first_basis = first.record.deadline_basis
    assert first_deadline is not None

    # Move the clock forward and repeat: the recorded deadline must be
    # byte-identical, not recomputed against the new clock.
    real_time = time.time

    def later_clock():
        return real_time() + 10_000

    monkeypatch.setattr(time, "time", later_clock)
    second = kb.advance_removal_to_fenced("deadline-once", removal_id=removal_id)

    assert second.success is True
    assert second.transitioned is False  # idempotent: already Fenced
    assert second.record.quiescence_deadline == first_deadline
    assert second.record.deadline_basis == first_basis

    on_disk = kb.get_removal_phase_record("deadline-once")
    assert on_disk.quiescence_deadline == first_deadline


def test_never_earlier_than_fence_close_plus_grace_even_with_a_past_expiry(fence_home):
    """QB-2: a held reservation whose expiry had ALREADY passed when the
    fence closed still gets the whole voluntary-completion window."""
    removal_id = _intent("past-expiry")
    conn = kb.connect(board="past-expiry")
    task_id = ready_task(conn)
    claimed = kb.claim_task(conn, task_id)
    assert claimed is not None
    # Force this claim's expiry into the past, directly on disk (never
    # through a production write path that would refuse once fenced).
    with kb.write_txn(conn):
        conn.execute(
            "UPDATE tasks SET claim_expires = ? WHERE id = ?",
            (int(time.time()) - 10_000, task_id),
        )
    conn.close()

    before_close = int(time.time())
    result = kb.advance_removal_to_fenced("past-expiry", removal_id=removal_id)
    assert result.success

    record = result.record
    assert record.deadline_basis == "fence-closing-instant-expiry-already-passed"
    assert record.quiescence_deadline >= before_close + kb.QUIESCENCE_GRACE_SECONDS


def test_deadline_uses_the_latest_held_expiry_when_it_is_in_the_future(fence_home):
    removal_id = _intent("future-expiry")
    conn = kb.connect(board="future-expiry")
    task_id = ready_task(conn)
    claimed = kb.claim_task(conn, task_id)
    assert claimed is not None
    future_expiry = int(time.time()) + 5_000
    with kb.write_txn(conn):
        conn.execute(
            "UPDATE tasks SET claim_expires = ? WHERE id = ?", (future_expiry, task_id)
        )
    conn.close()

    result = kb.advance_removal_to_fenced("future-expiry", removal_id=removal_id)
    assert result.success

    record = result.record
    assert record.deadline_basis == "latest-held-reservation-expiry"
    assert record.quiescence_deadline == future_expiry + kb.QUIESCENCE_GRACE_SECONDS


def test_no_held_reservations_deadline_is_fence_close_plus_grace(fence_home):
    removal_id = _intent("no-held")
    before = int(time.time())
    result = kb.advance_removal_to_fenced("no-held", removal_id=removal_id)
    after = int(time.time())

    record = result.record
    assert record.deadline_basis == "no-reservations-held-at-fence-close"
    assert before + kb.QUIESCENCE_GRACE_SECONDS <= record.quiescence_deadline
    assert record.quiescence_deadline <= after + kb.QUIESCENCE_GRACE_SECONDS


# ---------------------------------------------------------------------------
# Refusals
# ---------------------------------------------------------------------------

def test_refuses_when_no_phase_record_exists(fence_home):
    create_fenced_board("no-intent")
    # Drive the register straight to removing without going through
    # record_removal_intent, so there is no phase record at all.
    entry = kb.get_register_entry("no-intent")
    kb.transition_register_entry(
        kb.RegisterEntry(
            board_name="no-intent", lifecycle=kb.BoardLifecycle.REMOVING,
            epoch=entry.epoch + 1, epoch_before=entry.epoch,
            gate_move=kb.GateMove.PENDING, created_at=entry.created_at,
        )
    )

    result = kb.advance_removal_to_fenced("no-intent", removal_id="whatever")

    assert result.success is False
    assert "intent must be recorded first" in result.message
    assert gate_row(kb.kanban_db_path(board="no-intent")) == ("open", 1)


def test_refuses_a_mismatched_removal_id(fence_home):
    removal_id = _intent("id-mismatch-fence")
    result = kb.advance_removal_to_fenced("id-mismatch-fence", removal_id="not-it")

    assert result.success is False
    assert "removal_id" in result.message
    assert gate_row(kb.kanban_db_path(board="id-mismatch-fence")) == ("open", 1)


def test_refuses_and_does_not_advance_the_phase_when_the_fence_close_fails(fence_home):
    """A stale epoch makes the underlying close fail; the phase record
    must stay at Intent — no advance on a failed close."""
    removal_id = _intent("close-fails")
    # Move the register authority out from under the pending removal
    # (simulating an abandon-and-restart that this call does not know
    # about), so commit_fence_closing_point's CAS is refused.
    entry = kb.get_register_entry("close-fails")
    with kb.register_connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        kb._write_register_entry(
            conn,
            kb.RegisterEntry(
                board_name="close-fails", lifecycle=kb.BoardLifecycle.LIVE,
                epoch=entry.epoch + 5, gate_move=kb.GateMove.SETTLED,
                created_at=entry.created_at,
            ),
        )
        conn.execute("COMMIT")

    result = kb.advance_removal_to_fenced("close-fails", removal_id=removal_id)

    assert result.success is False
    assert result.fence_result is not None
    assert result.fence_result.success is False
    on_disk = kb.get_removal_phase_record("close-fails")
    assert on_disk.phase == kb.RemovalPhase.INTENT


def test_a_delayed_restart_derives_the_deadline_from_the_gate_close_instant(
    fence_home, monkeypatch
):
    """QB-2: the one-time deadline comes from the DURABLE fence-closing
    instant, never from the restart's clock.

    Crash point P2a: the gate commits at instant T and the process stops
    before the register record. Recovery arrives 9000 seconds later. The
    recorded deadline must be the value derived from T — a recovery that
    recomputed from its own clock would extend a bound §6.3 says is
    computed once.
    """
    removal_id = _intent("delayed-restart")
    conn = kb.connect(board="delayed-restart")
    task_id = ready_task(conn)
    assert kb.claim_task(conn, task_id) is not None
    gate_close_instant = 1_000_000
    held_expiry = gate_close_instant + 900
    with kb.write_txn(conn):
        conn.execute(
            "UPDATE tasks SET claim_expires = ? WHERE id = ?",
            (held_expiry, task_id),
        )
    conn.close()

    # The gate commits at T, and then nothing else happens.
    monkeypatch.setattr(time, "time", lambda: float(gate_close_instant))
    close = kb.commit_fence_closing_point("delayed-restart")
    assert close.success and close.transitioned
    assert kb.get_removal_phase_record("delayed-restart").phase == kb.RemovalPhase.INTENT

    # A much later restart rolls the removal forward.
    restart = gate_close_instant + 9_000
    monkeypatch.setattr(time, "time", lambda: float(restart))
    decision = kb.resume_removal("delayed-restart")
    assert decision.point is kb.RemovalRecoveryPoint.P2A

    result = kb.advance_removal_to_fenced("delayed-restart", removal_id=removal_id)

    assert result.success is True
    record = kb.get_removal_phase_record("delayed-restart")
    assert record.gate_closed_at == gate_close_instant
    assert record.deadline_basis == "latest-held-reservation-expiry"
    assert record.quiescence_deadline == held_expiry + kb.QUIESCENCE_GRACE_SECONDS
    # Emphatically NOT derived from the restart's clock.
    assert record.quiescence_deadline < restart


def test_a_delayed_restart_with_nothing_held_still_uses_the_gate_close_instant(
    fence_home, monkeypatch
):
    removal_id = _intent("delayed-restart-empty")
    gate_close_instant = 2_000_000
    monkeypatch.setattr(time, "time", lambda: float(gate_close_instant))
    assert kb.commit_fence_closing_point("delayed-restart-empty").transitioned

    monkeypatch.setattr(time, "time", lambda: float(gate_close_instant + 9_000))
    result = kb.advance_removal_to_fenced(
        "delayed-restart-empty", removal_id=removal_id
    )

    assert result.success is True
    record = result.record
    assert record.deadline_basis == "no-reservations-held-at-fence-close"
    assert record.quiescence_deadline == (
        gate_close_instant + kb.QUIESCENCE_GRACE_SECONDS
    )


def test_fenced_refuses_when_the_reservations_cannot_be_read(fence_home):
    """A deadline may not be computed from a failed read (QB-2).

    The gate itself still reads; what cannot be read is the durable place
    the Held reservations live. A removal that treated that failure as
    "no reservations held" would compute a deadline it has no basis for.
    """
    removal_id = _intent("fenced-unreadable")
    assert kb.commit_fence_closing_point("fenced-unreadable").transitioned
    conn = kb._sqlite_connect_no_create(kb.kanban_db_path(board="fenced-unreadable"))
    try:
        conn.execute("BEGIN IMMEDIATE")
        conn.execute("DROP TABLE tasks")
        conn.execute("COMMIT")
    finally:
        conn.close()

    result = kb.advance_removal_to_fenced("fenced-unreadable", removal_id=removal_id)

    assert result.success is False
    assert result.outcome is kb.RemovalAdvanceOutcome.REFUSED_INDETERMINATE
    record = kb.get_removal_phase_record("fenced-unreadable")
    assert record.phase == kb.RemovalPhase.INTENT
    assert record.quiescence_deadline is None
