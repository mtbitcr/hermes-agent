"""Phase 1 — Intent (design revision 5, §6.1).

One conditional transition of the register entry from ``live`` to
``removing``, with the phase record written in the same register
transaction. Covers every named outcome of that single transition and
the mode-precedence rule.
"""

from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

import pytest

_WORKTREE = Path(__file__).resolve().parents[2]
if str(_WORKTREE) not in sys.path:
    sys.path.insert(0, str(_WORKTREE))

from hermes_cli import kanban_db as kb
from tests.hermes_cli._kanban_fence_support import (
    create_fenced_board,
    permanent_confirmation,
    ready_task,
    register_row,
    start_removal,
    write_register_row_behind_the_lock,
)


# ---------------------------------------------------------------------------
# It wins and proceeds
# ---------------------------------------------------------------------------

def test_intent_wins_moves_live_to_removing_with_the_expected_pair(fence_home):
    create_fenced_board("wins")
    result = start_removal("wins", mode="reversible")

    assert result.success is True
    assert result.outcome is kb.RemovalIntentOutcome.STARTED
    assert result.removal_id
    row = register_row("wins")
    # EM-4b's expected pair, all set in the SAME transition (§6.1).
    assert row == {
        "lifecycle": "removing",
        "epoch": 2,
        "epoch_before": 1,
        "gate_move": "pending",
    }
    record = kb.get_removal_phase_record("wins")
    assert record.phase == kb.RemovalPhase.INTENT
    assert record.mode == kb.RemovalMode.REVERSIBLE
    assert record.epoch == 2


def test_intent_refuses_nothing_on_the_board_itself(fence_home):
    """GA-1 / EM-4b: a board sitting at Intent with the fence not yet
    closed keeps serving ordinary work."""
    create_fenced_board("still-open")
    conn = kb.connect(board="still-open")
    task_id = ready_task(conn)

    result = start_removal("still-open", mode="permanent")
    assert result.success

    claimed = kb.claim_task(conn, task_id)
    assert claimed is not None
    assert claimed.status == "running"
    conn.close()


# ---------------------------------------------------------------------------
# Same mode joins
# ---------------------------------------------------------------------------

def test_same_mode_joins_and_writes_nothing(fence_home):
    create_fenced_board("join-same")
    first = start_removal("join-same", mode="reversible")
    assert first.success

    before = register_row("join-same")
    second = start_removal("join-same", mode="reversible")

    assert second.success is True
    assert second.outcome is kb.RemovalIntentOutcome.JOINED
    assert second.removal_id == first.removal_id
    # Nothing written: the register row is untouched.
    assert register_row("join-same") == before


def test_join_with_an_explicit_removal_id_still_returns_the_original(fence_home):
    create_fenced_board("join-explicit")
    first = start_removal("join-explicit", mode="permanent")
    second = start_removal(
        "join-explicit", mode="permanent", removal_id="ignored-because-joining"
    )
    assert second.outcome is kb.RemovalIntentOutcome.JOINED
    assert second.removal_id == first.removal_id


# ---------------------------------------------------------------------------
# Mode precedence: cross-mode joins are refused, in both directions
# ---------------------------------------------------------------------------

def test_reversible_cannot_join_an_in_flight_permanent_removal(fence_home):
    create_fenced_board("perm-first")
    start_removal("perm-first", mode="permanent")
    before = register_row("perm-first")

    result = start_removal("perm-first", mode="reversible")

    assert result.success is False
    assert result.outcome is kb.RemovalIntentOutcome.REFUSED_MODE_CONFLICT
    assert register_row("perm-first") == before


def test_permanent_cannot_join_an_in_flight_reversible_removal(fence_home):
    create_fenced_board("rev-first")
    start_removal("rev-first", mode="reversible")
    before = register_row("rev-first")

    result = start_removal("rev-first", mode="permanent")

    assert result.success is False
    assert result.outcome is kb.RemovalIntentOutcome.REFUSED_MODE_CONFLICT
    assert "re-issue" in result.message.lower()
    assert register_row("rev-first") == before


# ---------------------------------------------------------------------------
# Terminal / indeterminate refusals
# ---------------------------------------------------------------------------

def test_intent_refuses_an_archived_board(fence_home):
    create_fenced_board("already-archived")
    entry = kb.get_register_entry("already-archived")
    kb.transition_register_entry(
        kb.RegisterEntry(
            board_name="already-archived",
            lifecycle=kb.BoardLifecycle.ARCHIVED,
            epoch=entry.epoch,
            gate_move=kb.GateMove.SETTLED,
            created_at=entry.created_at,
        )
    )

    result = start_removal("already-archived", mode="permanent")

    assert result.success is False
    assert result.outcome is kb.RemovalIntentOutcome.REFUSED_TERMINAL


def test_intent_refuses_a_hard_removed_board(fence_home):
    create_fenced_board("already-gone")
    entry = kb.get_register_entry("already-gone")
    kb.transition_register_entry(
        kb.RegisterEntry(
            board_name="already-gone",
            lifecycle=kb.BoardLifecycle.HARD_REMOVED,
            epoch=entry.epoch,
            gate_move=kb.GateMove.SETTLED,
            created_at=entry.created_at,
        )
    )

    result = start_removal("already-gone", mode="reversible")

    assert result.success is False
    assert result.outcome is kb.RemovalIntentOutcome.REFUSED_TERMINAL


def test_intent_refuses_indeterminate_when_no_register_entry_exists(fence_home):
    # A board with no register entry at all: never backfilled.
    result = start_removal("phantom-board", mode="permanent")

    assert result.success is False
    assert result.outcome is kb.RemovalIntentOutcome.REFUSED_INDETERMINATE


# ---------------------------------------------------------------------------
# Concurrency: two racers, exactly one wins the intent transition
# ---------------------------------------------------------------------------

def test_two_racers_one_wins_the_intent_transition(fence_home):
    """Not multiprocess (the lock is a cross-process flock, exercised for
    real in tests/stress); this proves the SAME-mode race resolves to
    exactly one winner and one JOINED follower, with a single epoch bump."""
    create_fenced_board("race")

    first = start_removal("race", mode="reversible")
    second = start_removal("race", mode="reversible")

    outcomes = {first.outcome, second.outcome}
    assert outcomes == {kb.RemovalIntentOutcome.STARTED, kb.RemovalIntentOutcome.JOINED}
    assert first.removal_id == second.removal_id
    assert register_row("race")["epoch"] == 2


# ---------------------------------------------------------------------------
# Permanent mode requires the operator to be shown the permanence statement
# and to confirm it (§6.1)
# ---------------------------------------------------------------------------

def test_permanent_mode_records_the_confirmation_durably(fence_home):
    create_fenced_board("perm-confirmed")
    disclosure = kb.permanent_removal_disclosure("perm-confirmed")
    checked = kb.confirm_permanent_removal(
        "perm-confirmed",
        response=disclosure.required_response,
        confirmed_by="operator",
        disclosure=disclosure,
    )
    assert checked.confirmed, checked.message
    statement, confirmation = disclosure.statement, checked.confirmation

    result = kb.record_removal_intent(
        "perm-confirmed", mode="permanent", permanent_confirmation=confirmation
    )

    assert result.success is True
    assert result.outcome is kb.RemovalIntentOutcome.STARTED
    record = kb.get_removal_phase_record("perm-confirmed")
    assert record.permanent_confirmed_at is not None
    recorded = json.loads(record.permanent_confirmation)
    assert recorded["statement_digest"] == statement.digest()
    assert recorded["confirmed_by"] == "operator"
    # The statement names the live-work consequence AND every OUT category
    # with what is retained under each (§6.1, §12.5).
    assert "destroyed" in recorded["statement"]
    assert {c["category"] for c in recorded["out_categories"]} == {
        name for name, _ in kb.REMOVAL_OUT_CATEGORIES
    }
    assert all(c["retained"] for c in recorded["out_categories"])


def test_permanent_mode_without_confirmation_is_refused_and_recorded(fence_home):
    create_fenced_board("perm-unconfirmed")

    result = kb.record_removal_intent("perm-unconfirmed", mode="permanent")

    assert result.success is False
    assert result.outcome is kb.RemovalIntentOutcome.REFUSED_UNCONFIRMED
    assert result.refusal_recorded is True
    recorded = kb.get_recorded_removal_refusal("perm-unconfirmed")
    assert recorded["outcome"] == "refused-unconfirmed"
    # Nothing began: no phase record, and the entry is untouched.
    assert kb.get_removal_phase_record("perm-unconfirmed") is None
    assert register_row("perm-unconfirmed") == {
        "lifecycle": "live", "epoch": 1, "epoch_before": None, "gate_move": "settled",
    }


def test_a_recorded_refusal_does_not_make_a_removal_look_started(fence_home):
    create_fenced_board("refusal-not-started")
    assert kb.record_removal_intent(
        "refusal-not-started", mode="permanent"
    ).success is False

    decision = kb.resume_removal("refusal-not-started")

    assert decision.point is kb.RemovalRecoveryPoint.P1
    assert decision.action is kb.RemovalRecoveryAction.NONE
    assert decision.record is None


def test_a_confirmation_of_another_boards_statement_is_refused(fence_home):
    create_fenced_board("perm-target")
    create_fenced_board("perm-other")
    other = permanent_confirmation("perm-other")

    result = kb.record_removal_intent(
        "perm-target", mode="permanent", permanent_confirmation=other
    )

    assert result.success is False
    assert result.outcome is kb.RemovalIntentOutcome.REFUSED_UNCONFIRMED
    assert kb.get_removal_phase_record("perm-target") is None


def test_an_unconfirmed_confirmation_object_is_not_a_confirmation(fence_home):
    create_fenced_board("perm-declined")
    statement = kb.permanence_statement("perm-declined")
    declined = kb.PermanentRemovalConfirmation(
        confirmed=False, statement_digest=statement.digest()
    )

    result = kb.record_removal_intent(
        "perm-declined", mode="permanent", permanent_confirmation=declined
    )

    assert result.success is False
    assert result.outcome is kb.RemovalIntentOutcome.REFUSED_UNCONFIRMED


def test_reversible_mode_needs_no_confirmation(fence_home):
    create_fenced_board("rev-no-confirmation")

    result = kb.record_removal_intent("rev-no-confirmation", mode="reversible")

    assert result.success is True
    record = kb.get_removal_phase_record("rev-no-confirmation")
    assert record.permanent_confirmed_at is None
    assert record.permanent_confirmation is None


# ---------------------------------------------------------------------------
# Every terminal and mode-conflict refusal is recorded durably (§6.1)
# ---------------------------------------------------------------------------

def test_a_terminal_refusal_records_its_outcome(fence_home):
    create_fenced_board("terminal-recorded")
    entry = kb.get_register_entry("terminal-recorded")
    kb.transition_register_entry(
        kb.RegisterEntry(
            board_name="terminal-recorded",
            lifecycle=kb.BoardLifecycle.ARCHIVED,
            epoch=entry.epoch,
            gate_move=kb.GateMove.SETTLED,
            created_at=entry.created_at,
        )
    )

    result = start_removal("terminal-recorded", mode="reversible")

    assert result.outcome is kb.RemovalIntentOutcome.REFUSED_TERMINAL
    assert result.refusal_recorded is True
    recorded = kb.get_recorded_removal_refusal("terminal-recorded")
    assert recorded["outcome"] == "refused-terminal"
    assert "archived" in recorded["message"]
    assert kb.get_removal_phase_record("terminal-recorded") is None


def test_a_reversible_joining_a_permanent_removal_records_its_refusal(fence_home):
    create_fenced_board("conflict-rev")
    assert start_removal("conflict-rev", mode="permanent").success

    result = start_removal("conflict-rev", mode="reversible")

    assert result.outcome is kb.RemovalIntentOutcome.REFUSED_MODE_CONFLICT
    assert result.refusal_recorded is True
    recorded = kb.get_recorded_removal_refusal("conflict-rev")
    assert recorded["outcome"] == "refused-mode-conflict"
    # The in-flight removal is untouched: still permanent, still at Intent.
    record = kb.get_removal_phase_record("conflict-rev")
    assert record.mode == kb.RemovalMode.PERMANENT
    assert record.phase == kb.RemovalPhase.INTENT


def test_a_permanent_joining_a_reversible_removal_records_its_refusal(fence_home):
    create_fenced_board("conflict-perm")
    assert start_removal("conflict-perm", mode="reversible").success

    result = start_removal("conflict-perm", mode="permanent")

    assert result.outcome is kb.RemovalIntentOutcome.REFUSED_MODE_CONFLICT
    assert result.refusal_recorded is True
    recorded = kb.get_recorded_removal_refusal("conflict-perm")
    assert recorded["outcome"] == "refused-mode-conflict"
    assert "re-issue" in recorded["message"]
    assert kb.get_removal_phase_record("conflict-perm").mode == kb.RemovalMode.REVERSIBLE


# ---------------------------------------------------------------------------
# The scope-declaration version in force is RECORDED, not merely copied
# ---------------------------------------------------------------------------

def test_the_scope_declaration_version_is_recorded_on_the_phase_record(fence_home):
    create_fenced_board("scope-version")

    result = start_removal("scope-version", mode="reversible")
    assert result.success

    record = kb.get_removal_phase_record("scope-version")
    assert record.scope_declaration_version == kb.SCOPE_DECLARATION_VERSION
    assert kb.get_register_entry("scope-version").scope_declaration_version == (
        kb.SCOPE_DECLARATION_VERSION
    )


def test_the_recorded_scope_version_survives_a_restart(fence_home):
    """A restart recovers the version the removal RAN UNDER from the phase
    record, without inferring it from whatever is in force now."""
    create_fenced_board("scope-restart")
    assert start_removal("scope-restart", mode="reversible").success

    # A fresh process reads the register from scratch.
    kb._REGISTER_INITIALIZED = False
    kb._REGISTER_INITIALIZED_PATHS.clear()

    decision = kb.resume_removal("scope-restart")
    assert decision.record.scope_declaration_version == kb.SCOPE_DECLARATION_VERSION


STALE_SCOPE_VERSION = "board-removal-safety-design-r4"


def _seed_stale_scope_version(slug: str) -> None:
    """Leave an OBSOLETE declaration version on the register entry.

    The state a board carries when its entry was last written under an
    earlier revision of the design: the stale value is sitting on the
    entry, and Intent must not adopt it as the declaration THIS removal
    runs under.
    """
    entry = kb.get_register_entry(slug)
    assert entry is not None
    write_register_row_behind_the_lock(
        kb.RegisterEntry(
            board_name=slug,
            lifecycle=entry.lifecycle,
            epoch=entry.epoch,
            epoch_before=entry.epoch_before,
            gate_move=entry.gate_move,
            ever_existed_marker=entry.ever_existed_marker,
            removal_mode=entry.removal_mode,
            scope_declaration_version=STALE_SCOPE_VERSION,
            epoch_lineage=entry.epoch_lineage,
            created_at=entry.created_at,
            updated_at=entry.updated_at,
        )
    )
    assert kb.get_register_entry(slug).scope_declaration_version == (
        STALE_SCOPE_VERSION
    )
    assert kb.SCOPE_DECLARATION_VERSION != STALE_SCOPE_VERSION


def test_intent_snapshots_the_declaration_in_force_over_a_stale_entry_value(fence_home):
    """§6.1: the phase record snapshots the declaration IN FORCE at Intent.

    A restart has to recover which rules the run began under. Adopting
    whatever the entry happened to be carrying records a revision this
    run never ran under — and every later phase then reads that as the
    authority for what it may do.
    """
    create_fenced_board("scope-stale")
    _seed_stale_scope_version("scope-stale")

    result = start_removal("scope-stale", mode="reversible")
    assert result.success, result.message

    record = kb.get_removal_phase_record("scope-stale")
    assert record.scope_declaration_version == kb.SCOPE_DECLARATION_VERSION
    assert record.scope_declaration_version != STALE_SCOPE_VERSION
    # A restart recovers the in-force snapshot, not the entry's value.
    assert kb.resume_removal("scope-stale").record.scope_declaration_version == (
        kb.SCOPE_DECLARATION_VERSION
    )


def test_the_permanent_confirmation_binds_to_the_declaration_in_force(fence_home):
    """The statement an operator confirms is the one in force, and the
    confirmation is bound to THAT snapshot — never to a stale prior value
    the register entry happened to carry."""
    create_fenced_board("scope-stale-perm")
    _seed_stale_scope_version("scope-stale-perm")

    result = start_removal("scope-stale-perm", mode="permanent")
    assert result.success, result.message

    record = kb.get_removal_phase_record("scope-stale-perm")
    assert record.scope_declaration_version == kb.SCOPE_DECLARATION_VERSION
    recorded = json.loads(record.permanent_confirmation)
    assert recorded["scope_declaration_version"] == kb.SCOPE_DECLARATION_VERSION
    statement = kb.permanence_statement("scope-stale-perm")
    assert statement.scope_declaration_version == kb.SCOPE_DECLARATION_VERSION
    assert recorded["statement_digest"] == statement.digest()
    assert kb.SCOPE_DECLARATION_VERSION in recorded["statement"]
    assert STALE_SCOPE_VERSION not in recorded["statement"]


# ---------------------------------------------------------------------------
# The permanence statement names EVERY OUT category (§6.1, §12.5)
# ---------------------------------------------------------------------------

def test_the_permanence_statement_names_every_required_out_category(fence_home):
    """§12.5: the statement an operator confirms names every OUT category
    with what is retained under it — including the four retained by
    decision rather than by this project's own mechanism (C6, C7, C11,
    C12). A category the operator is never shown is a permanence
    statement that understates what survives.
    """
    create_fenced_board("statement-out")
    statement = kb.permanence_statement("statement-out")
    text = statement.text()

    for category in ("C6", "C7", "C11", "C12"):
        assert category in text, category
    # Every category the statement already named is still named.
    for category in (
        "removal-archive", "ever-existed-marker", "register-entry",
        "operator-items",
    ):
        assert category in text, category
    # None of them is a bare label: each names what is retained under it,
    # and each appears in the text the operator is shown.
    assert statement.out_categories == kb.REMOVAL_OUT_CATEGORIES
    for name, retained in statement.out_categories:
        assert name and retained, name
        assert name in text and retained in text, name
    # The live-work consequence is still there.
    assert "destroyed" in text


def test_a_confirmation_of_a_statement_missing_an_out_category_is_refused(fence_home):
    """A digest computed over a statement that leaves an OUT category out
    is not a confirmation of the statement in force, so the permanent
    removal is refused and nothing begins."""
    create_fenced_board("statement-obsolete")
    in_force = kb.permanence_statement("statement-obsolete")
    obsolete = replace(
        in_force,
        out_categories=tuple(
            pair for pair in in_force.out_categories
            if not pair[0].startswith(("C6", "C7", "C11", "C12"))
        ),
    )
    assert obsolete.digest() != in_force.digest()

    result = kb.record_removal_intent(
        "statement-obsolete", mode="permanent",
        permanent_confirmation=kb.PermanentRemovalConfirmation(
            confirmed=True, statement_digest=obsolete.digest(),
        ),
    )

    assert result.success is False
    assert result.outcome is kb.RemovalIntentOutcome.REFUSED_UNCONFIRMED
    assert kb.get_removal_phase_record("statement-obsolete") is None
    assert register_row("statement-obsolete")["lifecycle"] == "live"
