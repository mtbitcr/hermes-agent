"""The fenced transition must bind ALL THREE phase-owned facts (QB-2).

Reviewer finding: ``_precondition_fenced`` recomputed both the
quiescence deadline AND its basis from durable state, but compared only
the deadline.  A phase-intent record carrying the exact valid deadline
and the exact valid fence-closing instant alongside a FORGED
``deadline_basis`` therefore satisfied the precondition: the transition
advanced to ``fenced`` and preserved the forged basis text as durable
state.

Every case here builds its durable state through the real production
path (``create_board`` + the real backfill, ``record_removal_intent``,
``commit_fence_closing_point``), damages exactly one column behind the
primitive — the writer no code path is allowed to be — and then calls
the shipped driver :func:`kanban_db.advance_removal_to_fenced`.
Assertions read the persisted record back.
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
    damage_phase_column_behind_the_primitive,
    read_only,
    start_removal,
)


FORGED_BASIS = "FORGED-BASIS"


def _gate_closed_at(slug: str) -> int:
    """The in-board gate row's own closing instant, read off the disk."""
    with read_only(kb.kanban_db_path(board=slug)) as conn:
        row = conn.execute(
            "SELECT gate, updated_at FROM board_fence_state WHERE id = 1"
        ).fetchone()
    assert row is not None, f"{slug} has no in-board gate row"
    assert row["gate"] != "open", f"{slug}'s gate is still open"
    return int(row["updated_at"])


def _fence_closed_intent(slug: str) -> "tuple[str, int, int]":
    """A real board parked at Intent with its fence already committed.

    Returns ``(removal_id, gate_closed_at, valid_deadline)``, where the
    deadline is the one QB-2's rule yields for a board with nothing held
    at the fence-closing point: the fence-closing instant plus the grace.
    """
    create_fenced_board(slug)
    intent = start_removal(slug)
    assert intent.success, intent.message
    closed = kb.commit_fence_closing_point(slug)
    assert closed.success and closed.transitioned, closed.message
    gate_closed_at = _gate_closed_at(slug)
    return (
        intent.removal_id,
        gate_closed_at,
        gate_closed_at + kb.QUIESCENCE_GRACE_SECONDS,
    )


def _derived_basis(slug: str, gate_closed_at: int) -> str:
    """What the shipped derivation yields for this durable state.

    Read from the production computation, not restated as a literal, so
    the test never freezes the basis vocabulary — it asserts the record
    keeps whatever the derivation actually says.
    """
    _deadline, basis = kb._compute_quiescence_deadline(
        slug, gate_closed_at=int(gate_closed_at),
        record=kb.get_removal_phase_record(slug),
    )
    assert _deadline is not None, basis
    return basis


@pytest.mark.parametrize("forged", [FORGED_BASIS, None])
def test_a_basis_that_is_not_the_derived_one_refuses_the_fenced_advance(
    fence_home, forged
):
    """Deadline exact, gate instant exact, basis wrong -> refusal.

    Both halves of "wrong" are covered: a forged string, and no basis at
    all beside a perfectly valid deadline.
    """
    slug = "fenced-basis-forged" if forged else "fenced-basis-missing"
    removal_id, gate_closed_at, deadline = _fence_closed_intent(slug)
    damage_phase_column_behind_the_primitive(
        slug, "quiescence_deadline", deadline,
    )
    damage_phase_column_behind_the_primitive(
        slug, "gate_closed_at", gate_closed_at,
    )
    damage_phase_column_behind_the_primitive(slug, "deadline_basis", forged)
    expected_basis = _derived_basis(slug, gate_closed_at)
    assert expected_basis != forged

    result = kb.advance_removal_to_fenced(slug, removal_id=removal_id)

    assert result.success is False, (
        f"the fenced advance accepted a basis of {forged!r} where the "
        f"derivation says {expected_basis!r}: {result.message}"
    )
    assert result.transitioned is False
    # The reservation read SUCCEEDED here, so this is an ordinary
    # refusal — the same channel the sibling deadline mismatch uses —
    # not the indeterminate one.
    assert result.outcome is kb.RemovalAdvanceOutcome.REFUSED_PRECONDITION


@pytest.mark.parametrize("forged", [FORGED_BASIS, None])
def test_the_refused_advance_leaves_the_intent_record_unchanged(
    fence_home, forged
):
    """No half-write: the phase stays Intent and no column is upgraded."""
    slug = "fenced-basis-nohalf-forged" if forged else "fenced-basis-nohalf-missing"
    removal_id, gate_closed_at, deadline = _fence_closed_intent(slug)
    damage_phase_column_behind_the_primitive(
        slug, "quiescence_deadline", deadline,
    )
    damage_phase_column_behind_the_primitive(
        slug, "gate_closed_at", gate_closed_at,
    )
    damage_phase_column_behind_the_primitive(slug, "deadline_basis", forged)
    before = kb.get_removal_phase_record(slug)
    assert before.phase == kb.RemovalPhase.INTENT
    assert before.deadline_basis == forged

    kb.advance_removal_to_fenced(slug, removal_id=removal_id)

    after = kb.get_removal_phase_record(slug)
    assert after.phase == kb.RemovalPhase.INTENT, (
        "the record advanced past Intent on a refused transition"
    )
    assert after.deadline_basis == forged, (
        f"the refused transition rewrote the basis to "
        f"{after.deadline_basis!r} — a half-write"
    )
    assert after.quiescence_deadline == deadline
    assert after.gate_closed_at == gate_closed_at


def test_the_honest_advance_still_records_the_derived_basis(fence_home):
    """The tightened comparison does not refuse the legitimate path.

    Nothing is damaged: the driver derives all three facts itself and the
    advance must succeed with the derivation's own basis recorded.
    """
    slug = "fenced-basis-honest"
    removal_id, gate_closed_at, deadline = _fence_closed_intent(slug)
    expected_basis = _derived_basis(slug, gate_closed_at)

    result = kb.advance_removal_to_fenced(slug, removal_id=removal_id)

    assert result.success is True, result.message
    assert result.transitioned is True
    record = kb.get_removal_phase_record(slug)
    assert record.phase == kb.RemovalPhase.FENCED
    assert record.deadline_basis == expected_basis
    assert record.quiescence_deadline == deadline
    assert record.gate_closed_at == gate_closed_at


def test_a_record_carrying_only_the_derived_basis_and_no_deadline_still_advances(
    fence_home
):
    """A basis alone is not a deadline: the driver derives both, and the
    already-correct basis must not turn the honest path into a refusal."""
    slug = "fenced-basis-only"
    removal_id, gate_closed_at, deadline = _fence_closed_intent(slug)
    damage_phase_column_behind_the_primitive(
        slug, "deadline_basis", _derived_basis(slug, gate_closed_at),
    )

    result = kb.advance_removal_to_fenced(slug, removal_id=removal_id)

    assert result.success is True, result.message
    record = kb.get_removal_phase_record(slug)
    assert record.phase == kb.RemovalPhase.FENCED
    assert record.quiescence_deadline == deadline
