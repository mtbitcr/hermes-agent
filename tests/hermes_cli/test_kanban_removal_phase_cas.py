"""The removal-phase compare-and-set (design revision 5, §6).

Every later phase transition goes through this one primitive, and the
PUBLIC entry point is guarded: before the compare-and-set commits, the
target phase's precondition — its entry in the precondition registry — is
evaluated against durable state inside the same transaction. Each guard
(idempotent no-op, lost-race refusal, backwards, skip, stale/no-record, id
mismatch, mode mismatch, precondition) gets its own test here so a later
change that removes or inverts a branch fails a test, not just a review.

Every phase is closed to that generic public entry point, so the shape
guards at a later phase are exercised through the private primitive
every DRIVER goes through (:func:`_quiesce` below) — the guards
themselves live there, and a test that could only reach them through a
route no driver takes would not be testing them at all.
"""

from __future__ import annotations

import json
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
    read_only,
    ready_task,
    start_removal,
    write_register_row_behind_the_lock,
)


def _intent(slug: str, mode: str = "reversible") -> str:
    create_fenced_board(slug)
    result = start_removal(slug, mode=mode)
    assert result.success, result.message
    return result.removal_id


def _fenced(slug: str, mode: str = "reversible") -> str:
    """A board at Fenced, reached the only way Fenced can be reached.

    ``advance_removal_to_fenced`` drives the real fence-closing point and
    only then satisfies Fenced's precondition, so a record sitting at
    Fenced here really does have a closed gate behind it.
    """
    removal_id = _intent(slug, mode)
    result = kb.advance_removal_to_fenced(slug, removal_id=removal_id)
    assert result.success and result.transitioned, result.message
    return removal_id


def _quiesce(slug: str, removal_id: str, **kwargs):
    """FENCED -> QUIESCED through the primitive a DRIVER goes through.

    Quiesced is closed to the generic public entry point and its
    completion fact is sealed, so a test whose subject is the
    compare-and-set ITSELF at this transition drives the private
    primitive through the phase-owned channel — the same channel
    ``advance_removal_to_quiesced`` writes through, and the only way to
    reach the primitive's own guards at a phase later than Fenced.

    This asserts nothing about the work: Quiesced's precondition still
    reads the quiescence predicate from durable state inside the
    transaction, so this only succeeds on a board that really is quiet.
    """
    return kb._advance_removal_phase(
        slug, removal_id=removal_id,
        from_phase=kb.RemovalPhase.FENCED, to_phase=kb.RemovalPhase.QUIESCED,
        _phase_owned={"quiesce_completed_at": int(time.time())},
        **kwargs,
    )


# ---------------------------------------------------------------------------
# No record / wrong board
# ---------------------------------------------------------------------------

def test_refuses_when_no_phase_record_exists(fence_home):
    create_fenced_board("no-record")
    # Fenced is closed to the public entry point — assert the driver-only
    # refusal, then exercise the no-record guard through the real driver.
    public = kb.advance_removal_phase(
        "no-record", removal_id="whatever",
        from_phase=kb.RemovalPhase.INTENT, to_phase=kb.RemovalPhase.FENCED,
    )
    assert public.success is False
    assert public.outcome is kb.RemovalAdvanceOutcome.REFUSED_PRECONDITION
    assert "advance_removal_to_fenced" in public.message

    result = kb.advance_removal_to_fenced("no-record", removal_id="whatever")
    assert result.success is False
    assert "no removal phase record" in result.message


def test_refuses_an_invalid_board_name(fence_home):
    # The invalid-board guard fires before the closed-phase guard.
    result = kb.advance_removal_phase(
        "", removal_id="x",
        from_phase=kb.RemovalPhase.INTENT, to_phase=kb.RemovalPhase.FENCED,
    )
    assert result.success is False
    assert result.outcome is kb.RemovalAdvanceOutcome.REFUSED_INVALID_BOARD
    assert result.transitioned is False

    # The driver has the same guard.
    driver = kb.advance_removal_to_fenced("", removal_id="x")
    assert driver.success is False
    assert "invalid board name" in driver.message


# ---------------------------------------------------------------------------
# removal_id mismatch
# ---------------------------------------------------------------------------

def test_refuses_a_removal_id_that_does_not_match(fence_home):
    removal_id = _intent("id-mismatch")
    # Fenced is closed to the public entry point; exercise the id-mismatch
    # guard through the real driver.
    result = kb.advance_removal_to_fenced(
        "id-mismatch", removal_id="some-other-removal-id",
    )
    assert result.success is False
    assert "does not match" in result.message
    assert result.record.removal_id == removal_id


# ---------------------------------------------------------------------------
# Mode never changes
# ---------------------------------------------------------------------------

def test_refuses_a_mode_that_disagrees_with_the_recorded_mode(fence_home):
    # Fenced is closed to the public entry point, so the mode-mismatch
    # guard is exercised through the private primitive (which every driver
    # goes through) rather than the public entry point.
    removal_id = _intent("mode-mismatch", mode="reversible")
    result = kb._advance_removal_phase(
        "mode-mismatch", removal_id=removal_id,
        from_phase=kb.RemovalPhase.INTENT, to_phase=kb.RemovalPhase.FENCED,
        mode="permanent",
    )
    assert result.success is False
    assert result.outcome is kb.RemovalAdvanceOutcome.REFUSED_MODE_MISMATCH


def test_a_matching_mode_field_is_accepted(fence_home):
    removal_id = _fenced("mode-match", mode="permanent")
    result = _quiesce("mode-match", removal_id, mode="permanent")
    assert result.success is True
    assert result.outcome is kb.RemovalAdvanceOutcome.ADVANCED
    assert result.transitioned is True


# ---------------------------------------------------------------------------
# The public entry point cannot record a phase whose meaning is not true
#
# This replaces a test that positively asserted the opposite — that the raw
# compare-and-set could move intent -> fenced with the fence still open.
# That behaviour WAS the defect: a live probe read phase=fenced while Gate
# B was still 'open' at epoch 1 with no deadline recorded, an ordinary
# board write still succeeded, and resume_removal then trusted the false
# phase and chose P3.
# ---------------------------------------------------------------------------

def test_public_cas_refuses_intent_to_fenced_while_the_gate_is_still_open(fence_home):
    # Fenced is now closed to the public entry point — assert the
    # driver-only refusal, then exercise the gate-open guard through the
    # real driver (which refuses because the gate is still open).
    removal_id = _intent("no-bypass")
    db_path = kb.kanban_db_path(board="no-bypass")
    conn = kb.connect(board="no-bypass")
    task_id = ready_task(conn)

    public = kb.advance_removal_phase(
        "no-bypass", removal_id=removal_id,
        from_phase=kb.RemovalPhase.INTENT, to_phase=kb.RemovalPhase.FENCED,
    )
    assert public.success is False
    assert public.outcome is kb.RemovalAdvanceOutcome.REFUSED_PRECONDITION
    assert "advance_removal_to_fenced" in public.message

    # The recorded phase stays where it was, and the fence really is open.
    assert kb.get_removal_phase_record("no-bypass").phase == kb.RemovalPhase.INTENT
    assert gate_row(db_path) == ("open", 1)

    # An ordinary board write still succeeds: nothing was fenced, and the
    # record does not claim otherwise.
    claimed = kb.claim_task(conn, task_id)
    assert claimed is not None
    assert claimed.status == "running"
    conn.close()

    # And recovery still reports the PRE-fence point, because the durable
    # state it reads has not been falsified.
    decision = kb.resume_removal("no-bypass")
    assert decision.point is kb.RemovalRecoveryPoint.P2
    assert decision.action is kb.RemovalRecoveryAction.CLOSE_FENCE_AND_SETTLE


def test_the_raw_private_cas_derives_and_enforces_the_precondition_itself(fence_home):
    """The precondition is not a parameter a caller can omit.

    The lower two functions are reached by every driver, and one of them
    used to accept ``precondition=None`` and skip the check entirely: a
    raw ``intent -> fenced`` then recorded ``fenced`` with Gate B still
    ``open`` at epoch 1, no deadline computed, ordinary board writes still
    succeeding, and ``resume_removal`` rolling forward into quiescence
    over the open gate. The target phase's registered meaning is now
    derived from ``to_phase`` inside the transaction that writes the
    column, so there is nothing to omit.
    """
    removal_id = _intent("raw-bypass")
    db_path = kb.kanban_db_path(board="raw-bypass")
    conn = kb.connect(board="raw-bypass")
    task_id = ready_task(conn)

    result = kb._advance_removal_phase(
        "raw-bypass", removal_id=removal_id,
        from_phase=kb.RemovalPhase.INTENT, to_phase=kb.RemovalPhase.FENCED,
    )

    assert result.success is False
    assert result.outcome is kb.RemovalAdvanceOutcome.REFUSED_PRECONDITION
    assert result.transitioned is False

    record = kb.get_removal_phase_record("raw-bypass")
    assert record.phase == kb.RemovalPhase.INTENT
    assert record.quiescence_deadline is None
    assert gate_row(db_path) == ("open", 1)

    # An ordinary board write still succeeds: nothing was fenced.
    claimed = kb.claim_task(conn, task_id)
    assert claimed is not None
    assert claimed.status == "running"
    conn.close()

    decision = kb.resume_removal("raw-bypass")
    assert decision.point is kb.RemovalRecoveryPoint.P2
    assert decision.action is kb.RemovalRecoveryAction.CLOSE_FENCE_AND_SETTLE


def test_the_in_transaction_cas_takes_no_precondition_from_its_caller(fence_home):
    """The enforcement point is the transaction that writes ``phase``.

    The lowest-level function derives the target phase's precondition from
    ``to_phase`` itself; a caller cannot supply, override or omit one.
    """
    removal_id = _intent("raw-in-txn")

    with kb.board_register_lock("raw-in-txn"):
        with kb.register_connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                result = kb._advance_removal_phase_in_txn(
                    conn, "raw-in-txn", removal_id=removal_id,
                    from_phase=kb.RemovalPhase.INTENT,
                    to_phase=kb.RemovalPhase.FENCED,
                    mode_check=None, fields={},
                )
            finally:
                conn.execute("ROLLBACK")

    assert result.success is False
    assert result.outcome is kb.RemovalAdvanceOutcome.REFUSED_PRECONDITION
    assert result.transitioned is False

    # And there is no parameter left through which a caller could hand it
    # one — or hand it None.
    with pytest.raises(TypeError):
        with kb.board_register_lock("raw-in-txn"):
            with kb.register_connect() as conn:
                kb._advance_removal_phase_in_txn(
                    conn, "raw-in-txn", removal_id=removal_id,
                    from_phase=kb.RemovalPhase.INTENT,
                    to_phase=kb.RemovalPhase.FENCED,
                    mode_check=None, fields={}, precondition=None,
                )

    assert kb.get_removal_phase_record("raw-in-txn").phase == kb.RemovalPhase.INTENT
    assert gate_row(kb.kanban_db_path(board="raw-in-txn")) == ("open", 1)


def test_public_cas_refuses_fenced_when_the_epoch_mirror_lags(fence_home):
    """EM-2: the fence that is closed must be THIS removal's fence.

    Fenced is closed to the public entry point; the epoch-mirror guard is
    exercised through the private primitive (which every driver goes through).
    """
    removal_id = _intent("mirror-lag")
    close = kb.commit_fence_closing_point("mirror-lag")
    assert close.success and close.transitioned
    entry = kb.get_register_entry("mirror-lag")
    gate = kb._read_gate_instant("mirror-lag")
    # The authority moves on without the mirror: the closed gate now
    # belongs to an epoch that is no longer authoritative.
    write_register_row_behind_the_lock(
        kb.RegisterEntry(
            board_name="mirror-lag",
            lifecycle=kb.BoardLifecycle.REMOVING,
            epoch=entry.epoch + 1,
            epoch_before=entry.epoch,
            gate_move=kb.GateMove.SETTLED,
            removal_mode=entry.removal_mode,
            created_at=entry.created_at,
        )
    )

    result = kb._advance_removal_phase(
        "mirror-lag", removal_id=removal_id,
        from_phase=kb.RemovalPhase.INTENT, to_phase=kb.RemovalPhase.FENCED,
        _phase_owned={
            "quiescence_deadline": 1_000_000,
            "deadline_basis": "supplied",
            "gate_closed_at": int(gate.updated_at),
        },
    )

    assert result.success is False
    assert result.outcome is kb.RemovalAdvanceOutcome.REFUSED_PRECONDITION
    assert "mirror" in result.message
    assert kb.get_removal_phase_record("mirror-lag").phase == kb.RemovalPhase.INTENT


def test_public_cas_refuses_fenced_when_no_deadline_is_being_recorded(fence_home):
    """QB-2: Fenced without the one-time deadline is not Fenced.

    Fenced is closed to the public entry point; the no-deadline guard is
    exercised through the private primitive with no phase_owned values.
    """
    removal_id = _intent("no-deadline")
    close = kb.commit_fence_closing_point("no-deadline")
    assert close.success and close.transitioned

    result = kb._advance_removal_phase(
        "no-deadline", removal_id=removal_id,
        from_phase=kb.RemovalPhase.INTENT, to_phase=kb.RemovalPhase.FENCED,
    )

    assert result.success is False
    assert result.outcome is kb.RemovalAdvanceOutcome.REFUSED_PRECONDITION
    assert "deadline" in result.message
    record = kb.get_removal_phase_record("no-deadline")
    assert record.phase == kb.RemovalPhase.INTENT
    assert record.quiescence_deadline is None


def test_public_cas_refuses_fenced_when_the_gate_cannot_be_read(fence_home):
    """A read that FAILED is not a satisfied precondition.

    Fenced is closed to the public entry point; the gate-unreadable guard
    is exercised through the private primitive with phase_owned values.
    """
    removal_id = _intent("gate-unreadable")
    close = kb.commit_fence_closing_point("gate-unreadable")
    assert close.success
    db_path = kb.kanban_db_path(board="gate-unreadable")
    db_path.write_bytes(b"this is not a sqlite database at all")

    result = kb._advance_removal_phase(
        "gate-unreadable", removal_id=removal_id,
        from_phase=kb.RemovalPhase.INTENT, to_phase=kb.RemovalPhase.FENCED,
        _phase_owned={
            "quiescence_deadline": 1_000_000,
            "deadline_basis": "supplied",
            "gate_closed_at": 1,
        },
    )

    assert result.success is False
    assert result.outcome is kb.RemovalAdvanceOutcome.REFUSED_INDETERMINATE
    assert kb.get_removal_phase_record("gate-unreadable").phase == kb.RemovalPhase.INTENT


def test_intent_and_done_are_closed_to_the_public_entry_point(fence_home):
    """Both move the register entry in the same transaction, which the
    phase primitive cannot do — so only their own driver may record them."""
    removal_id = _intent("closed-phases")
    intent_attempt = kb.advance_removal_phase(
        "closed-phases", removal_id=removal_id,
        from_phase=kb.RemovalPhase.INTENT, to_phase=kb.RemovalPhase.INTENT,
    )
    assert intent_attempt.success is False
    assert intent_attempt.outcome is kb.RemovalAdvanceOutcome.REFUSED_PRECONDITION
    assert "record_removal_intent" in intent_attempt.message

    done_attempt = kb.advance_removal_phase(
        "closed-phases", removal_id=removal_id,
        from_phase=kb.RemovalPhase.SWEPT, to_phase=kb.RemovalPhase.DONE,
    )
    assert done_attempt.success is False
    assert done_attempt.outcome is kb.RemovalAdvanceOutcome.REFUSED_PRECONDITION
    assert "complete_removal" in done_attempt.message


# ---------------------------------------------------------------------------
# A fact supplied by the caller is never durable evidence that work happened
# ---------------------------------------------------------------------------

# Every column that records what a phase's own work established. Each one
# is written by that phase's named driver and by nothing else.
_PHASE_OWNED_FACTS = {
    "quiescence_deadline": 1,
    "deadline_basis": "forged",
    "gate_closed_at": 1,
    "quiesce_completed_at": 1,
    "carry_completed_at": 1,
    "release_completed_at": 1,
    "sweep_completed_at": 1,
    "carried_payload": "{}",
    "release_state": "{}",
    "sweep_state": "{}",
    "applied_mode_content": "{}",
    "apply_journal": "{}",
}

_FORGED_SANDBOX_ID = "sbx-forged-exact"


def _quiesced_with_a_real_unreleased_obligation(slug: str) -> str:
    """A board at Quiesced carrying ONE real, unreleased cleanup obligation.

    ``run_sandbox_cleanup_intents`` is this codebase's durable record of
    "an exact run has a machine with no durable release event". The row is
    written through the real API before the removal begins, so the work
    §6.4/§6.5 owe is genuinely outstanding.
    """
    create_fenced_board(slug)
    conn = kb.connect(board=slug)
    task_id = ready_task(conn)
    with kb.write_txn(conn):
        conn.execute(
            "INSERT INTO run_sandbox_cleanup_intents ("
            "  task_id, run_id, profile, generation, sandbox_id, "
            "  provision_event_id, attempt_count, next_attempt_at"
            ") VALUES (?, 1, 'worker', 1, ?, 1, 0, 0)",
            (task_id, _FORGED_SANDBOX_ID),
        )
    conn.close()
    removal_id = start_removal(slug, mode="reversible").removal_id
    assert kb.advance_removal_to_fenced(slug, removal_id=removal_id).success
    assert kb.advance_removal_to_quiesced(slug, removal_id=removal_id).success
    return removal_id


@pytest.mark.parametrize("field", sorted(_PHASE_OWNED_FACTS))
def test_every_phase_owned_fact_is_sealed_against_a_public_caller(fence_home, field):
    """A column that records what was VERIFIED cannot arrive as an assertion.

    Each of these is read back by a precondition as the durable proof that
    the phase's work happened. A public caller that could pass one would be
    handing the removal its own conclusion.
    """
    slug = f"sealed-{field.replace('_', '-')}"
    removal_id = _fenced(slug)

    with pytest.raises(ValueError) as excinfo:
        kb.advance_removal_phase(
            slug, removal_id=removal_id,
            from_phase=kb.RemovalPhase.FENCED, to_phase=kb.RemovalPhase.QUIESCED,
            **{field: _PHASE_OWNED_FACTS[field]},
        )

    assert field in str(excinfo.value)
    record = kb.get_removal_phase_record(slug)
    assert record.phase == kb.RemovalPhase.FENCED
    # Fenced-owned fields are legitimately recorded by the fenced driver
    # that _fenced() went through; assert they were not OVERWRITTEN by the
    # forged value rather than that they are absent.
    fenced_owned = {"quiescence_deadline", "deadline_basis", "gate_closed_at"}
    if field in fenced_owned:
        assert getattr(record, field) != _PHASE_OWNED_FACTS[field], field
    else:
        assert getattr(record, field) is None


def test_the_sealed_set_covers_every_phase_owned_fact(fence_home):
    """A new completion fact must be sealed when it is added, not later."""
    assert set(kb._REMOVAL_PHASE_SEALED_FIELDS) == set(_PHASE_OWNED_FACTS)


def test_every_closed_phase_is_recordable_only_by_its_own_named_driver(fence_home):
    """The generic public entry point is a refusal surface that names the
    driver — for ALL EIGHT phases, not just Intent and Done.

    Every phase is closed: each owns sealed, phase-owned columns whose
    value only its named driver can derive, or (for Intent/Done) a
    register-entry write the phase primitive cannot make. A generic
    transition into any of them is refused and the refusal names the
    driver that is the only legitimate path.
    """
    removal_id = _fenced("driver-only")
    assert set(kb._REMOVAL_PHASES_CLOSED_TO_PUBLIC_CAS) == set(kb.REMOVAL_PHASE_ORDER)
    for phase in kb._REMOVAL_PHASES_CLOSED_TO_PUBLIC_CAS:
        precondition = kb.removal_phase_precondition(phase)
        result = kb.advance_removal_phase(
            "driver-only", removal_id=removal_id,
            from_phase=kb.RemovalPhase.FENCED, to_phase=phase,
        )
        assert result.success is False, phase.value
        assert result.transitioned is False, phase.value
        assert precondition.driver in result.message, phase.value
    assert kb.get_removal_phase_record("driver-only").phase == kb.RemovalPhase.FENCED


def test_a_forged_work_bearing_sequence_is_refused_at_every_step(fence_home):
    """The reproduction: a structurally valid, wholly forged run.

    Every payload below has the shape each precondition looks for and
    describes work that never happened — the obligation is still owed and
    the board's store is fully intact. Before the repair this exact
    sequence walked the PUBLIC entry point from Quiesced to Swept, and the
    real sweep driver then reported ``idempotent-noop`` over it.
    """
    slug = "forged-work"
    removal_id = _quiesced_with_a_real_unreleased_obligation(slug)
    forged_payload = json.dumps({
        "version": 1,
        "release_obligations": [
            {"kind": "run-sandbox", "identity": {"sandbox_id": _FORGED_SANDBOX_ID}}
        ],
        "outside_resource_ledger": [{"member": "board-directory", "identity": "/x"}],
    })
    forged_release = json.dumps({
        "environments": [
            {
                "identity": {"sandbox_id": _FORGED_SANDBOX_ID},
                "state": kb.EnvironmentReleaseState.RELEASED.value,
            }
        ]
    })
    forged_sweep = json.dumps({
        "version": 1,
        "classes": {
            key: {"result": kb.SWEEP_RESULT_NOT_PRESENT, "reason": "forged"}
            for key, _ in kb.SWEEP_RECORD_CLASSES
        },
    })
    steps = (
        (
            kb.RemovalPhase.QUIESCED, kb.RemovalPhase.CARRIED,
            {"carried_payload": forged_payload, "carry_completed_at": 1},
        ),
        (
            kb.RemovalPhase.CARRIED, kb.RemovalPhase.RELEASED,
            {"release_state": forged_release, "release_completed_at": 2},
        ),
        (kb.RemovalPhase.RELEASED, kb.RemovalPhase.APPLIED, {}),
        (
            kb.RemovalPhase.APPLIED, kb.RemovalPhase.SWEPT,
            {"sweep_state": forged_sweep, "sweep_completed_at": 3},
        ),
    )

    for from_phase, to_phase, fields in steps:
        if fields:
            with pytest.raises(ValueError) as excinfo:
                kb.advance_removal_phase(
                    slug, removal_id=removal_id,
                    from_phase=from_phase, to_phase=to_phase, **fields,
                )
            for name in fields:
                assert name in str(excinfo.value)
        refused = kb.advance_removal_phase(
            slug, removal_id=removal_id, from_phase=from_phase, to_phase=to_phase,
        )
        assert refused.success is False, to_phase.value
        assert refused.transitioned is False, to_phase.value
        assert kb.removal_phase_precondition(to_phase).driver in refused.message

    # Nothing was recorded, and nothing about the world changed.
    record = kb.get_removal_phase_record(slug)
    assert record.phase == kb.RemovalPhase.QUIESCED
    # Fenced-owned fields are legitimately recorded by the fenced driver
    # that built the quiesced board; quiesce_completed_at by its own driver.
    fenced_owned = {"quiescence_deadline", "deadline_basis", "gate_closed_at"}
    for field in _PHASE_OWNED_FACTS:
        if field == "quiesce_completed_at":
            continue  # its own driver really did record this one
        if field in fenced_owned:
            continue  # the fenced driver really did record these
        assert getattr(record, field) is None, field
    db_path = kb.kanban_db_path(board=slug)
    assert db_path.exists()
    with read_only(db_path) as conn:
        owed = conn.execute(
            "SELECT sandbox_id FROM run_sandbox_cleanup_intents"
        ).fetchall()
    assert [row["sandbox_id"] for row in owed] == [_FORGED_SANDBOX_ID]

    # And the real driver reports no success over state it did not establish.
    swept = kb.advance_removal_to_swept(slug, removal_id=removal_id)
    assert swept.success is False
    assert kb.get_removal_phase_record(slug).phase == kb.RemovalPhase.QUIESCED


def test_every_phase_has_a_registered_precondition(fence_home):
    """A new phase cannot be added without a decision about its meaning."""
    for phase in kb.REMOVAL_PHASE_ORDER:
        precondition = kb.removal_phase_precondition(phase)
        assert precondition.rule
        assert precondition.summary
        assert precondition.driver


# ---------------------------------------------------------------------------
# Forward-only, no skipping
# ---------------------------------------------------------------------------

def test_advances_one_step_forward_once_the_phase_means_something(fence_home):
    removal_id = _fenced("one-step")
    result = _quiesce("one-step", removal_id)
    assert result.success is True
    assert result.outcome is kb.RemovalAdvanceOutcome.ADVANCED
    assert result.transitioned is True
    assert result.record.phase == kb.RemovalPhase.QUIESCED
    assert kb.get_removal_phase_record("one-step").phase == kb.RemovalPhase.QUIESCED


def test_refuses_to_skip_a_phase(fence_home):
    removal_id = _intent("skip")
    # The shape guards live in the primitive every driver goes through,
    # which is where a skip has to be refused: Intent -> Quiesced skips
    # Fenced, whoever asks for it.
    result = kb._advance_removal_phase(
        "skip", removal_id=removal_id,
        from_phase=kb.RemovalPhase.INTENT, to_phase=kb.RemovalPhase.QUIESCED,
    )
    assert result.success is False
    assert result.outcome is kb.RemovalAdvanceOutcome.REFUSED_SKIP
    # Left exactly where it was.
    assert kb.get_removal_phase_record("skip").phase == kb.RemovalPhase.INTENT

    # The generic public entry point refuses the same skip one guard
    # earlier, naming the driver Quiesced is recordable by at all.
    public = kb.advance_removal_phase(
        "skip", removal_id=removal_id,
        from_phase=kb.RemovalPhase.INTENT, to_phase=kb.RemovalPhase.QUIESCED,
    )
    assert public.success is False
    assert public.transitioned is False
    assert kb.removal_phase_precondition(
        kb.RemovalPhase.QUIESCED
    ).driver in public.message
    assert kb.get_removal_phase_record("skip").phase == kb.RemovalPhase.INTENT


def test_refuses_to_move_backwards(fence_home):
    removal_id = _fenced("backwards")

    result = kb.advance_removal_phase(
        "backwards", removal_id=removal_id,
        from_phase=kb.RemovalPhase.FENCED, to_phase=kb.RemovalPhase.INTENT,
    )
    assert result.success is False
    assert result.outcome is kb.RemovalAdvanceOutcome.REFUSED_PRECONDITION
    assert kb.get_removal_phase_record("backwards").phase == kb.RemovalPhase.FENCED

    # The backwards guard itself, exercised through the private primitive
    # (Fenced is now closed to the public entry point).
    assert _quiesce("backwards", removal_id).success
    backwards = kb._advance_removal_phase(
        "backwards", removal_id=removal_id,
        from_phase=kb.RemovalPhase.QUIESCED, to_phase=kb.RemovalPhase.FENCED,
    )
    assert backwards.success is False
    assert backwards.outcome is kb.RemovalAdvanceOutcome.REFUSED_BACKWARDS
    assert kb.get_removal_phase_record("backwards").phase == kb.RemovalPhase.QUIESCED


def test_a_stale_from_phase_that_is_neither_current_nor_target_is_refused(fence_home):
    removal_id = _fenced("stale")
    # Now at FENCED. Ask to go FENCED->QUIESCED but claim from_phase=INTENT
    # (a caller working off a stale read of the record), through the
    # primitive that holds the shape guards.
    result = kb._advance_removal_phase(
        "stale", removal_id=removal_id,
        from_phase=kb.RemovalPhase.INTENT, to_phase=kb.RemovalPhase.QUIESCED,
    )
    assert result.success is False
    assert result.outcome is kb.RemovalAdvanceOutcome.REFUSED_STALE
    assert kb.get_removal_phase_record("stale").phase == kb.RemovalPhase.FENCED


# ---------------------------------------------------------------------------
# Idempotence: re-applying an already-happened transition is a no-op success
# ---------------------------------------------------------------------------

def test_repeating_the_same_transition_is_an_idempotent_success(fence_home):
    removal_id = _fenced("repeat")
    first = _quiesce("repeat", removal_id)
    assert first.outcome is kb.RemovalAdvanceOutcome.ADVANCED
    assert first.transitioned is True

    second = _quiesce("repeat", removal_id)
    assert second.success is True
    assert second.outcome is kb.RemovalAdvanceOutcome.IDEMPOTENT_NOOP
    assert second.transitioned is False
    assert second.record.phase == kb.RemovalPhase.QUIESCED


def test_idempotent_repeat_does_not_touch_extra_fields(fence_home):
    """A repeat of Phase 2 must leave a once-recorded deadline unchanged
    (§6.2) — proven here through the real driver: a second call to the
    driver is an idempotent no-op and the once-recorded deadline is
    preserved."""
    removal_id = _fenced("repeat-fields")
    recorded = kb.get_removal_phase_record("repeat-fields")
    assert recorded.quiescence_deadline is not None

    result = kb.advance_removal_to_fenced("repeat-fields", removal_id=removal_id)
    assert result.outcome is kb.RemovalAdvanceOutcome.IDEMPOTENT_NOOP
    assert result.transitioned is False
    record = kb.get_removal_phase_record("repeat-fields")
    assert record.quiescence_deadline == recorded.quiescence_deadline
    assert record.deadline_basis == recorded.deadline_basis


# ---------------------------------------------------------------------------
# Unsupported fields are a programming error, not a silent write
# ---------------------------------------------------------------------------

def test_unsupported_extra_fields_raise(fence_home):
    # All phases are closed to the public entry point, so the unsupported-
    # field guard is exercised through the private primitive.
    removal_id = _intent("bad-field")
    with pytest.raises(ValueError):
        kb._advance_removal_phase(
            "bad-field", removal_id=removal_id,
            from_phase=kb.RemovalPhase.INTENT, to_phase=kb.RemovalPhase.FENCED,
            not_a_real_column="x",
        )


# ---------------------------------------------------------------------------
# Two racers, one winner (in-process; a real cross-process proof is in
# tests/stress)
# ---------------------------------------------------------------------------

def test_two_racers_the_same_transition_exactly_one_advances(fence_home):
    removal_id = _fenced("race-cas")
    first = _quiesce("race-cas", removal_id)
    second = _quiesce("race-cas", removal_id)
    outcomes = {first.outcome, second.outcome}
    assert outcomes == {
        kb.RemovalAdvanceOutcome.ADVANCED, kb.RemovalAdvanceOutcome.IDEMPOTENT_NOOP,
    }
    assert first.success and second.success
    assert [first.transitioned, second.transitioned].count(True) == 1


def test_a_racer_that_declares_what_it_observed_is_refused_not_no_opped(fence_home):
    """A caller that saw ``from_phase``, did the work, and arrived to find
    the phase already moved LOST a contended transition. It must be
    refused, not handed a success it did not perform."""
    removal_id = _fenced("race-loser")
    winner = _quiesce(
        "race-loser", removal_id, observed_phase=kb.RemovalPhase.FENCED
    )
    assert winner.outcome is kb.RemovalAdvanceOutcome.ADVANCED
    assert winner.transitioned is True

    loser = _quiesce(
        "race-loser", removal_id, observed_phase=kb.RemovalPhase.FENCED
    )
    assert loser.success is False
    assert loser.outcome is kb.RemovalAdvanceOutcome.REFUSED_LOST_RACE
    assert loser.transitioned is False
    # The record is still exactly what the winner left.
    assert kb.get_removal_phase_record("race-loser").phase == kb.RemovalPhase.QUIESCED


def test_a_roll_forward_repeat_that_declares_nothing_is_still_a_no_op(fence_home):
    """The legitimate idempotent repeat survives: a recovery re-driving a
    completed step declares no observation and gets a no-op success."""
    removal_id = _fenced("race-rollforward")
    assert _quiesce("race-rollforward", removal_id).transitioned is True

    repeat = _quiesce("race-rollforward", removal_id)
    assert repeat.success is True
    assert repeat.outcome is kb.RemovalAdvanceOutcome.IDEMPOTENT_NOOP
    assert repeat.transitioned is False


# ---------------------------------------------------------------------------
# Finding 1 regression: a forged PAST deadline through the public entry
# point must be refused now that Fenced is closed and its facts are sealed
# ---------------------------------------------------------------------------

def test_public_cas_with_forged_past_deadline_is_refused(fence_home):
    """Regression for the fenced-is-not-sealed-to-its-driver finding.

    Builds a real board with one HELD reservation (claim_expires well in
    the future), mints a real permanent confirmation, records intent, and
    closes the fence. Then attempts the public ``advance_removal_phase``
    with a forged PAST quiescence_deadline — exactly the reproduction the
    finding describes. The fix seals the three fenced-owned fields, so
    the public caller gets a ValueError (sealed) or a REFUSED_PRECONDITION
    (closed), and the phase stays at INTENT with no forged values on it.
    """
    slug = "forged-past-deadline"
    create_fenced_board(slug)
    conn = kb.connect(board=slug)
    task_id = ready_task(conn)
    claimed = kb.claim_task(conn, task_id, ttl_seconds=3599)
    assert claimed is not None and claimed.status == "running"
    conn.close()

    result = start_removal(slug, mode="permanent")
    assert result.success, result.message
    rid = result.removal_id

    fence = kb.commit_fence_closing_point(slug)
    assert fence.success and fence.transitioned, fence.message

    now = int(time.time())
    forged_past = now - 10_000

    # The sealed check fires first and raises ValueError.
    with pytest.raises(ValueError) as excinfo:
        kb.advance_removal_phase(
            slug, removal_id=rid,
            from_phase=kb.RemovalPhase.INTENT, to_phase=kb.RemovalPhase.FENCED,
            quiescence_deadline=forged_past,
            deadline_basis="caller-wrote-this",
            gate_closed_at=now - 99_999,
        )
    assert "quiescence_deadline" in str(excinfo.value)

    # The phase record is untouched at INTENT with no forged values.
    record = kb.get_removal_phase_record(slug)
    assert record.phase == kb.RemovalPhase.INTENT
    assert record.quiescence_deadline is None
    assert record.deadline_basis is None
    assert record.gate_closed_at is None

    # Even without sealed fields, the public entry point refuses because
    # FENCED is closed.
    refused = kb.advance_removal_phase(
        slug, removal_id=rid,
        from_phase=kb.RemovalPhase.INTENT, to_phase=kb.RemovalPhase.FENCED,
    )
    assert refused.success is False
    assert refused.outcome is kb.RemovalAdvanceOutcome.REFUSED_PRECONDITION
    assert "advance_removal_to_fenced" in refused.message

    # The honest driver path succeeds and leaves a FUTURE deadline.
    honest = kb.advance_removal_to_fenced(slug, removal_id=rid)
    assert honest.success and honest.transitioned, honest.message
    record = kb.get_removal_phase_record(slug)
    assert record.phase == kb.RemovalPhase.FENCED
    assert record.quiescence_deadline > now
    assert record.deadline_basis == "latest-held-reservation-expiry"
    assert record.gate_closed_at is not None
    # With a genuinely held reservation the honest deadline is in the
    # future, so the next phase's decision is WAIT, not force-end-held.
    action = kb.quiescence_deadline_action(record, held=1)
    assert action == kb.QuiescenceDeadlineAction.WAIT
