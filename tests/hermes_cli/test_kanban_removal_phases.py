"""The eight-phase vocabulary itself (design revision 5, §6, phase table).

Ordering must be DATA — an explicit ordinal mapping — not a side effect of
enum member definition order. These tests would fail if the ordinal table
were ever derived from ``list(RemovalPhase)`` instead of the recorded
``REMOVAL_PHASE_ORDER`` tuple, and they pin the exact string values the
durable record persists.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_WORKTREE = Path(__file__).resolve().parents[2]
if str(_WORKTREE) not in sys.path:
    sys.path.insert(0, str(_WORKTREE))

from hermes_cli import kanban_db as kb


def test_the_eight_phases_have_the_recorded_lowercase_values():
    expected = {
        kb.RemovalPhase.INTENT: "intent",
        kb.RemovalPhase.FENCED: "fenced",
        kb.RemovalPhase.QUIESCED: "quiesced",
        kb.RemovalPhase.CARRIED: "carried",
        kb.RemovalPhase.RELEASED: "released",
        kb.RemovalPhase.APPLIED: "applied",
        kb.RemovalPhase.SWEPT: "swept",
        kb.RemovalPhase.DONE: "done",
    }
    for phase, value in expected.items():
        assert phase.value == value


def test_removal_phase_order_has_all_eight_phases_in_order():
    assert kb.REMOVAL_PHASE_ORDER == (
        kb.RemovalPhase.INTENT,
        kb.RemovalPhase.FENCED,
        kb.RemovalPhase.QUIESCED,
        kb.RemovalPhase.CARRIED,
        kb.RemovalPhase.RELEASED,
        kb.RemovalPhase.APPLIED,
        kb.RemovalPhase.SWEPT,
        kb.RemovalPhase.DONE,
    )


def test_ordinals_are_1_based_and_match_the_order_tuple():
    for index, phase in enumerate(kb.REMOVAL_PHASE_ORDER):
        assert kb.removal_phase_ordinal(phase) == index + 1
    # Also accepts the raw string value, not just the enum member.
    assert kb.removal_phase_ordinal("intent") == 1
    assert kb.removal_phase_ordinal("done") == 8


def test_ordinal_is_a_lookup_table_not_a_definition_order_side_effect():
    """Would fail if ordinals were derived from enum iteration order.

    Reordering ``REMOVAL_PHASE_ORDER`` (leaving the class body's member
    definition order untouched) must move the ordinals with it — proving
    the ordinal table is driven by data, not by ``RemovalPhase`` class
    body order.
    """
    reordered = (kb.RemovalPhase.DONE,) + kb.REMOVAL_PHASE_ORDER[:-1]
    ordinals = {phase: i + 1 for i, phase in enumerate(reordered)}
    assert ordinals[kb.RemovalPhase.DONE] == 1
    assert ordinals[kb.RemovalPhase.INTENT] == 2
    # The real module table disagrees — proving it is independent data.
    assert kb.removal_phase_ordinal(kb.RemovalPhase.DONE) == 8
    assert kb.removal_phase_ordinal(kb.RemovalPhase.INTENT) == 1


@pytest.mark.parametrize(
    "phase,expected",
    [
        (kb.RemovalPhase.INTENT, kb.RemovalPhase.FENCED),
        (kb.RemovalPhase.FENCED, kb.RemovalPhase.QUIESCED),
        (kb.RemovalPhase.QUIESCED, kb.RemovalPhase.CARRIED),
        (kb.RemovalPhase.CARRIED, kb.RemovalPhase.RELEASED),
        (kb.RemovalPhase.RELEASED, kb.RemovalPhase.APPLIED),
        (kb.RemovalPhase.APPLIED, kb.RemovalPhase.SWEPT),
        (kb.RemovalPhase.SWEPT, kb.RemovalPhase.DONE),
    ],
)
def test_next_removal_phase_walks_the_order(phase, expected):
    assert kb.next_removal_phase(phase) == expected


def test_next_removal_phase_at_done_is_none():
    assert kb.next_removal_phase(kb.RemovalPhase.DONE) is None


def test_removal_mode_has_exactly_the_two_recorded_values():
    assert {m.value for m in kb.RemovalMode} == {"permanent", "reversible"}
