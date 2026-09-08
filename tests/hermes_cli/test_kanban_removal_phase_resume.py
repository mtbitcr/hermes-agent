"""``resume_removal`` (design revision 5, §9.3): the restart entry point.

A pure function of durable state: reads the phase record, the register
entry, and (when relevant) the in-board gate, and returns the recovery
point and prescribed action WITHOUT acting. Covers P1, P2, P2a and P3,
and proves the "two restarts choose identically" requirement directly.
"""

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
    gate_row,
    start_removal,
)


# ---------------------------------------------------------------------------
# P1 — no record at all
# ---------------------------------------------------------------------------

def test_p1_no_record_does_nothing(fence_home):
    create_fenced_board("p1")
    decision = kb.resume_removal("p1")

    assert decision.point is kb.RemovalRecoveryPoint.P1
    assert decision.action is kb.RemovalRecoveryAction.NONE
    assert decision.record is None


def test_p1_for_a_board_that_was_never_created(fence_home):
    decision = kb.resume_removal("never-existed")
    assert decision.point is kb.RemovalRecoveryPoint.P1
    assert decision.action is kb.RemovalRecoveryAction.NONE


# ---------------------------------------------------------------------------
# P2 — Intent recorded, fence not yet closed
# ---------------------------------------------------------------------------

def test_p2_intent_recorded_fence_still_open(fence_home):
    create_fenced_board("p2")
    start_removal("p2", mode="reversible")

    decision = kb.resume_removal("p2")

    assert decision.point is kb.RemovalRecoveryPoint.P2
    assert decision.action is kb.RemovalRecoveryAction.CLOSE_FENCE_AND_SETTLE
    assert decision.record.phase == kb.RemovalPhase.INTENT
    assert gate_row(kb.kanban_db_path(board="p2")) == ("open", 1)


def test_p2_rolls_forward_by_actually_calling_advance_removal_to_fenced(fence_home):
    """The prescribed action really is idempotent-safe to perform from here."""
    create_fenced_board("p2-rollforward")
    intent = start_removal("p2-rollforward", mode="reversible")
    decision = kb.resume_removal("p2-rollforward")
    assert decision.point is kb.RemovalRecoveryPoint.P2

    result = kb.advance_removal_to_fenced("p2-rollforward", removal_id=intent.removal_id)
    assert result.success is True
    assert result.transitioned is True
    assert gate_row(kb.kanban_db_path(board="p2-rollforward")) == ("closing", 2)


# ---------------------------------------------------------------------------
# P2a — gate committed, gate_move not yet settled
# ---------------------------------------------------------------------------

def test_p2a_gate_committed_but_gate_move_still_pending(fence_home):
    """Reproduces the crash point directly: close Gate B through the real
    primitive but stop BEFORE the phase wrapper records gate_move=settled."""
    create_fenced_board("p2a")
    start_removal("p2a", mode="permanent")

    # The board-store commit only — exactly what a crash between the two
    # writes §6.2 mandates would leave behind.
    close = kb.commit_fence_closing_point("p2a")
    assert close.success and close.transitioned

    entry = kb.get_register_entry("p2a")
    assert entry.gate_move == kb.GateMove.PENDING  # not yet settled

    decision = kb.resume_removal("p2a")

    assert decision.point is kb.RemovalRecoveryPoint.P2A
    assert decision.action is kb.RemovalRecoveryAction.CLOSE_FENCE_AND_SETTLE
    assert decision.record.phase == kb.RemovalPhase.INTENT


def test_p2a_rolls_forward_to_settled_and_fenced(fence_home):
    create_fenced_board("p2a-rollforward")
    intent = start_removal("p2a-rollforward", mode="permanent")
    kb.commit_fence_closing_point("p2a-rollforward")
    decision = kb.resume_removal("p2a-rollforward")
    assert decision.point is kb.RemovalRecoveryPoint.P2A

    result = kb.advance_removal_to_fenced(
        "p2a-rollforward", removal_id=intent.removal_id
    )
    assert result.success is True
    entry = kb.get_register_entry("p2a-rollforward")
    assert entry.gate_move == kb.GateMove.SETTLED
    assert kb.get_removal_phase_record("p2a-rollforward").phase == kb.RemovalPhase.FENCED


# ---------------------------------------------------------------------------
# P3 — Fenced, no reservation resolved yet
# ---------------------------------------------------------------------------

def test_p3_fenced_reads_the_recorded_deadline_never_recomputing(fence_home):
    create_fenced_board("p3")
    intent = start_removal("p3", mode="reversible")
    fenced = kb.advance_removal_to_fenced("p3", removal_id=intent.removal_id)
    assert fenced.success

    decision = kb.resume_removal("p3")

    assert decision.point is kb.RemovalRecoveryPoint.P3
    assert decision.action is kb.RemovalRecoveryAction.ROLL_FORWARD_INTO_QUIESCENCE
    assert decision.record.phase == kb.RemovalPhase.FENCED
    assert decision.record.quiescence_deadline == fenced.record.quiescence_deadline


# ---------------------------------------------------------------------------
# §9.3: phases at or beyond Quiesced each have their own recovery point
# ---------------------------------------------------------------------------

def test_quiesced_returns_p6_with_roll_forward_carry_and_release(fence_home):
    """P6 — Quiesced, before any release: roll forward by carrying
    obligations, ledger and inventory, then releasing.

    Replaces a test that asserted BEYOND_FENCED for Quiesced, which was
    the behaviour before §9.3's recovery points were implemented.
    """
    create_fenced_board("quiesced-resume")
    intent = start_removal("quiesced-resume", mode="reversible")
    kb.advance_removal_to_fenced("quiesced-resume", removal_id=intent.removal_id)
    kb.advance_removal_to_quiesced("quiesced-resume", removal_id=intent.removal_id)

    decision = kb.resume_removal("quiesced-resume")

    assert decision.point is kb.RemovalRecoveryPoint.P6
    assert decision.action is kb.RemovalRecoveryAction.ROLL_FORWARD_CARRY_AND_RELEASE
    assert decision.record.phase == kb.RemovalPhase.QUIESCED


# ---------------------------------------------------------------------------
# Purity: two restarts against the same durable state choose identically
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "setup",
    ["p1", "p2", "p2a", "p3", "beyond"],
)
def test_two_restarts_choose_identically(fence_home, setup):
    slug = f"purity-{setup}"
    create_fenced_board(slug)
    if setup != "p1":
        intent = start_removal(slug, mode="reversible")
    if setup in ("p2a", "p3", "beyond"):
        kb.commit_fence_closing_point(slug) if setup == "p2a" else None
    if setup in ("p3", "beyond"):
        kb.advance_removal_to_fenced(slug, removal_id=intent.removal_id)
    if setup == "beyond":
        kb.advance_removal_to_quiesced(slug, removal_id=intent.removal_id)

    first = kb.resume_removal(slug)
    second = kb.resume_removal(slug)

    assert first.point == second.point
    assert first.action == second.action
    assert first.record == second.record


# ---------------------------------------------------------------------------
# §9.3 recovery points for each phase at or beyond Quiesced
# ---------------------------------------------------------------------------

def _to_carried(slug: str, removal_id: str) -> None:
    """Drive a board through Quiesced to Carried."""
    assert kb.advance_removal_to_carried(slug, removal_id=removal_id).success


def _to_released(slug: str, removal_id: str) -> None:
    """Drive a board through to Released."""
    _to_carried(slug, removal_id)
    assert kb.advance_removal_to_released(slug, removal_id=removal_id).success


def _to_applied(slug: str, removal_id: str) -> None:
    """Drive a board through to Applied."""
    _to_released(slug, removal_id)
    assert kb.advance_removal_to_applied(slug, removal_id=removal_id).success


def _to_swept(slug: str, removal_id: str) -> None:
    """Drive a board through to Swept."""
    _to_applied(slug, removal_id)
    assert kb.advance_removal_to_swept(slug, removal_id=removal_id).success


def test_carried_returns_p7_with_roll_forward_retry_release(fence_home):
    """P7 — Mid-release: roll forward by retrying release by exact identity."""
    create_fenced_board("carried-resume")
    intent = start_removal("carried-resume", mode="reversible")
    kb.advance_removal_to_fenced("carried-resume", removal_id=intent.removal_id)
    kb.advance_removal_to_quiesced("carried-resume", removal_id=intent.removal_id)
    _to_carried("carried-resume", intent.removal_id)

    decision = kb.resume_removal("carried-resume")

    assert decision.point is kb.RemovalRecoveryPoint.P7
    assert decision.action is kb.RemovalRecoveryAction.ROLL_FORWARD_RETRY_RELEASE
    assert decision.record.phase == kb.RemovalPhase.CARRIED


def test_released_returns_p8_with_roll_forward_into_applied(fence_home):
    """P8 — Released, before content applied: roll forward into Applied."""
    create_fenced_board("released-resume")
    intent = start_removal("released-resume", mode="reversible")
    kb.advance_removal_to_fenced("released-resume", removal_id=intent.removal_id)
    kb.advance_removal_to_quiesced("released-resume", removal_id=intent.removal_id)
    _to_released("released-resume", intent.removal_id)

    decision = kb.resume_removal("released-resume")

    assert decision.point is kb.RemovalRecoveryPoint.P8
    assert decision.action is kb.RemovalRecoveryAction.ROLL_FORWARD_INTO_APPLIED
    assert decision.record.phase == kb.RemovalPhase.RELEASED


def test_applied_with_content_outstanding_returns_p9(fence_home):
    """P9 — Applied, but §6.6's mode-specific content has not been performed.

    Applied is TWO durable facts, and the phase alone does not say which
    of them is true. With the content outstanding and nothing journalled
    destroyed, the recovery point is INSIDE the apply: rolling forward
    into the sweep here would step over the destruction entirely.
    """
    create_fenced_board("applied-resume")
    intent = start_removal("applied-resume", mode="reversible")
    kb.advance_removal_to_fenced("applied-resume", removal_id=intent.removal_id)
    kb.advance_removal_to_quiesced("applied-resume", removal_id=intent.removal_id)
    _to_applied("applied-resume", intent.removal_id)

    decision = kb.resume_removal("applied-resume")

    assert decision.point is kb.RemovalRecoveryPoint.P9
    assert decision.action is kb.RemovalRecoveryAction.ROLL_FORWARD_APPLY_CONTENT
    assert decision.record.phase == kb.RemovalPhase.APPLIED
    assert kb.applied_mode_content_is_outstanding(decision.record)


def test_applied_returns_p10_once_the_content_is_recorded(fence_home):
    """P10 — Applied AND the content performed: roll forward into the sweep."""
    create_fenced_board("applied-content-resume")
    intent = start_removal("applied-content-resume", mode="reversible")
    kb.advance_removal_to_fenced(
        "applied-content-resume", removal_id=intent.removal_id
    )
    kb.advance_removal_to_quiesced(
        "applied-content-resume", removal_id=intent.removal_id
    )
    _to_applied("applied-content-resume", intent.removal_id)
    content = kb.apply_reversible_mode_content(
        "applied-content-resume", removal_id=intent.removal_id
    )
    assert content.success, content.message

    decision = kb.resume_removal("applied-content-resume")

    assert decision.point is kb.RemovalRecoveryPoint.P10
    assert decision.action is kb.RemovalRecoveryAction.ROLL_FORWARD_INTO_SWEEP
    assert decision.record.phase == kb.RemovalPhase.APPLIED
    assert not kb.applied_mode_content_is_outstanding(decision.record)


def test_swept_returns_p11_with_roll_forward_into_done(fence_home):
    """P11 — Swept, not yet marked complete: roll forward into Done."""
    create_fenced_board("swept-resume")
    intent = start_removal("swept-resume", mode="reversible")
    kb.advance_removal_to_fenced("swept-resume", removal_id=intent.removal_id)
    kb.advance_removal_to_quiesced("swept-resume", removal_id=intent.removal_id)
    _to_swept("swept-resume", intent.removal_id)

    decision = kb.resume_removal("swept-resume")

    assert decision.point is kb.RemovalRecoveryPoint.P11
    assert decision.action is kb.RemovalRecoveryAction.ROLL_FORWARD_INTO_DONE
    assert decision.record.phase == kb.RemovalPhase.SWEPT
