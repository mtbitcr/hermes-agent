"""Phases 3-8 (design revision 5, §6.3-§6.8): the shared, mode-agnostic
skeleton beyond Fenced.

Each wrapper here is thin over :func:`kanban_db.advance_removal_phase`;
what is under test is the GUARD each phase puts on its own transition —
TQ-2's predicate at Quiesced and at Applied, the QB-3 deadline decision
table, and Phase 8's single locked step to ``archived``/``hard-removed``.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import pytest

_WORKTREE = Path(__file__).resolve().parents[2]
if str(_WORKTREE) not in sys.path:
    sys.path.insert(0, str(_WORKTREE))

from hermes_cli import kanban_db as kb
from tests.hermes_cli._kanban_fence_support import (
    apply_mode_specific_content,
    create_fenced_board,
    damage_phase_column_behind_the_primitive,
    read_only,
    ready_task,
    register_row,
    start_removal,
    task_row,
)


def _fenced(slug: str, mode: str = "reversible") -> str:
    create_fenced_board(slug)
    intent = start_removal(slug, mode=mode)
    assert intent.success, intent.message
    fenced = kb.advance_removal_to_fenced(slug, removal_id=intent.removal_id)
    assert fenced.success, fenced.message
    return intent.removal_id


def _fenced_with_task(slug: str, mode: str = "reversible") -> str:
    """A fenced board carrying ONE real, unclaimed task.

    The task is created through the real API BEFORE the fence closes,
    because a closed gate refuses creation — which is the fence's whole
    point.
    """
    create_fenced_board(slug)
    conn = kb.connect(board=slug)
    ready_task(conn)
    conn.close()
    intent = start_removal(slug, mode=mode)
    assert intent.success, intent.message
    fenced = kb.advance_removal_to_fenced(slug, removal_id=intent.removal_id)
    assert fenced.success, fenced.message
    return intent.removal_id


def _held_board(slug: str, mode: str = "reversible") -> str:
    """A fenced board with ONE reservation genuinely Held across the fence.

    The claim is granted by the real ``claim_task`` before the fence
    closes, so the row this leaves behind is a reservation whose grant
    committed before the gate commit — Held, by §6.3's own definition.
    """
    create_fenced_board(slug)
    conn = kb.connect(board=slug)
    task_id = ready_task(conn)
    assert kb.claim_task(conn, task_id) is not None
    conn.close()
    intent = start_removal(slug, mode=mode)
    assert intent.success, intent.message
    fenced = kb.advance_removal_to_fenced(slug, removal_id=intent.removal_id)
    assert fenced.success, fenced.message
    return intent.removal_id


def _released_then_held_again(slug: str, removal_id: str) -> None:
    """Reach Released through every real driver, then Hold a reservation.

    Replaces a helper that drove FENCED -> QUIESCED through the raw
    private primitive with a claim still Held — a positive bypass that no
    longer exists, and should not: the primitive now derives Quiesced's
    registered precondition itself and refuses over held work, whoever
    calls it.

    So every phase up to Released is reached legitimately, with nothing
    held, and only then is a reservation Held again. Nothing
    production-shaped can grant a claim once the gate is closed (QB-1),
    and Quiesced refuses while one is Held — so a record at Released with
    a live claim is a durable-state anomaly, which is exactly what
    Applied's own TQ-2 re-read exists to catch. The row is written through
    the fence's own protocol scope (the same stand-in the quiescence tests
    use to clear a claim voluntarily), so what Applied reads is a REAL
    running reservation holding a live claim lock on this host.
    """
    for driver in (
        kb.advance_removal_to_quiesced,
        kb.advance_removal_to_carried,
        kb.advance_removal_to_released,
    ):
        result = driver(slug, removal_id=removal_id)
        assert result.success, f"{driver.__name__}: {result.message}"
    task_id = _only_task(slug)
    conn = kb.connect(board=slug)
    try:
        with kb._fence_protocol_scope():
            with kb.write_txn(conn):
                conn.execute(
                    "UPDATE tasks SET status = 'running', claim_lock = ?, "
                    "claim_expires = ?, worker_pid = ? WHERE id = ?",
                    (
                        kb._claimer_id(),
                        int(time.time()) + 3600,
                        os.getpid(),
                        task_id,
                    ),
                )
    finally:
        conn.close()
    row = task_row(kb.kanban_db_path(board=slug), task_id)
    assert row["status"] == "running" and row["claim_lock"] is not None


def _only_task(slug: str) -> str:
    with read_only(kb.kanban_db_path(board=slug)) as conn:
        rows = conn.execute("SELECT id FROM tasks").fetchall()
    assert len(rows) == 1, rows
    return rows[0]["id"]


def _open_task_removal_window(slug: str) -> None:
    """Record a multi-step task removal in the durable place that holds one.

    This codebase has no multi-step task-removal protocol yet, so nothing
    production-shaped writes this table. Writing it directly is the only
    way to prove the count is READ from durable state rather than assumed
    to be zero.
    """
    conn = kb._sqlite_connect_no_create(kb.kanban_db_path(board=slug))
    try:
        conn.execute("BEGIN IMMEDIATE")
        conn.execute(
            f"CREATE TABLE IF NOT EXISTS {kb.TASK_REMOVAL_WINDOW_TABLE} ("
            "task_id TEXT NOT NULL, started_at INTEGER NOT NULL, "
            "ended_at INTEGER)"
        )
        conn.execute(
            f"INSERT INTO {kb.TASK_REMOVAL_WINDOW_TABLE} "
            "(task_id, started_at, ended_at) VALUES ('t_x', 1, NULL)"
        )
        conn.execute("COMMIT")
    finally:
        conn.close()


def _make_store_unreadable(slug: str) -> None:
    """Replace a board's store with bytes no SQLite can read."""
    kb.kanban_db_path(board=slug).write_bytes(b"not a database, not even close")


def _to_swept(slug: str, removal_id: str) -> None:
    assert kb.advance_removal_to_quiesced(slug, removal_id=removal_id).success
    assert kb.advance_removal_to_carried(slug, removal_id=removal_id).success
    assert kb.advance_removal_to_released(slug, removal_id=removal_id).success
    assert kb.advance_removal_to_applied(slug, removal_id=removal_id).success
    assert kb.advance_removal_to_swept(slug, removal_id=removal_id).success


# ---------------------------------------------------------------------------
# Phase 3 — Quiesced: the shared predicate (zero Held, zero task removals)
# ---------------------------------------------------------------------------

def test_quiesced_advances_immediately_when_nothing_is_held(fence_home):
    removal_id = _fenced("quiesce-clear")
    result = kb.advance_removal_to_quiesced("quiesce-clear", removal_id=removal_id)

    assert result.success is True
    assert result.action is kb.QuiescenceDeadlineAction.ADVANCE
    assert result.held == 0
    assert kb.get_removal_phase_record("quiesce-clear").phase == kb.RemovalPhase.QUIESCED


def test_quiesced_refuses_while_a_reservation_is_held_before_the_deadline(fence_home):
    create_fenced_board("quiesce-held")
    conn = kb.connect(board="quiesce-held")
    task_id = ready_task(conn)
    claimed = kb.claim_task(conn, task_id)
    assert claimed is not None
    conn.close()

    intent = kb.record_removal_intent("quiesce-held", mode="reversible")
    fenced = kb.advance_removal_to_fenced("quiesce-held", removal_id=intent.removal_id)
    assert fenced.success

    result = kb.advance_removal_to_quiesced("quiesce-held", removal_id=intent.removal_id)

    assert result.success is False
    assert result.held == 1
    assert result.action is kb.QuiescenceDeadlineAction.WAIT
    assert "still quiescing" in result.message
    assert kb.get_removal_phase_record("quiesce-held").phase == kb.RemovalPhase.FENCED


def test_quiesced_advances_once_the_held_reservation_clears(fence_home):
    create_fenced_board("quiesce-clears")
    conn = kb.connect(board="quiesce-clears")
    task_id = ready_task(conn)
    claimed = kb.claim_task(conn, task_id)
    assert claimed is not None

    intent = kb.record_removal_intent("quiesce-clears", mode="reversible")
    fenced = kb.advance_removal_to_fenced("quiesce-clears", removal_id=intent.removal_id)
    assert fenced.success

    still_held = kb.advance_removal_to_quiesced("quiesce-clears", removal_id=intent.removal_id)
    assert still_held.success is False

    # The claim clears (through direct SQL — every ordinary release path is
    # refused once the fence is closed, which is the point of the fence;
    # nothing production-shaped clears a claim here, so this stands in for
    # "the work finished and released voluntarily").
    with kb._fence_protocol_scope():
        with kb.write_txn(conn):
            conn.execute(
                "UPDATE tasks SET status = 'done', claim_lock = NULL, "
                "claim_expires = NULL WHERE id = ?", (task_id,),
            )
    conn.close()

    result = kb.advance_removal_to_quiesced("quiesce-clears", removal_id=intent.removal_id)
    assert result.success is True
    assert result.held == 0
    assert kb.get_removal_phase_record("quiesce-clears").phase == kb.RemovalPhase.QUIESCED


# ---------------------------------------------------------------------------
# An idempotent repeat REVALIDATES; it never reports success over state it
# did not establish
# ---------------------------------------------------------------------------

def test_a_quiesce_repeat_revalidates_the_predicate_it_finds_recorded(fence_home):
    removal_id = _fenced("quiesce-repeat-ok")
    first = kb.advance_removal_to_quiesced("quiesce-repeat-ok", removal_id=removal_id)
    assert first.transitioned is True

    repeat = kb.advance_removal_to_quiesced("quiesce-repeat-ok", removal_id=removal_id)

    assert repeat.success is True
    assert repeat.outcome is kb.RemovalAdvanceOutcome.IDEMPOTENT_NOOP
    assert repeat.transitioned is False


def test_a_quiesce_repeat_refuses_when_the_predicate_cannot_be_reread(fence_home):
    """A no-op success is a claim that quiescence still holds. An
    unreadable store cannot support that claim."""
    removal_id = _fenced("quiesce-repeat-unreadable")
    assert kb.advance_removal_to_quiesced(
        "quiesce-repeat-unreadable", removal_id=removal_id
    ).success
    _make_store_unreadable("quiesce-repeat-unreadable")

    repeat = kb.advance_removal_to_quiesced(
        "quiesce-repeat-unreadable", removal_id=removal_id
    )

    assert repeat.success is False
    assert repeat.outcome is kb.RemovalAdvanceOutcome.REFUSED_INDETERMINATE
    assert repeat.transitioned is False
    assert kb.get_removal_phase_record(
        "quiesce-repeat-unreadable"
    ).phase == kb.RemovalPhase.QUIESCED


def test_a_quiesce_repeat_refuses_when_its_completion_fact_is_gone(fence_home):
    """The durable fact the phase's meaning consists of is re-read, not
    assumed from the phase column alone."""
    removal_id = _fenced("quiesce-repeat-factless")
    assert kb.advance_removal_to_quiesced(
        "quiesce-repeat-factless", removal_id=removal_id
    ).success
    damage_phase_column_behind_the_primitive(
        "quiesce-repeat-factless", "quiesce_completed_at", None
    )

    repeat = kb.advance_removal_to_quiesced(
        "quiesce-repeat-factless", removal_id=removal_id
    )

    assert repeat.success is False
    assert repeat.outcome is kb.RemovalAdvanceOutcome.REFUSED_WORK_INCOMPLETE
    assert repeat.transitioned is False


def test_quiesced_refuses_a_mismatched_removal_id(fence_home):
    removal_id = _fenced("quiesce-mismatch")
    result = kb.advance_removal_to_quiesced("quiesce-mismatch", removal_id="nope")
    assert result.success is False
    assert "removal_id" in result.message


def test_quiesced_refuses_when_no_record_exists(fence_home):
    create_fenced_board("quiesce-none")
    result = kb.advance_removal_to_quiesced("quiesce-none", removal_id="x")
    assert result.success is False
    assert result.record is None


# ---------------------------------------------------------------------------
# QB-3 — the shared deadline decision table
# ---------------------------------------------------------------------------

def _record(mode: kb.RemovalMode, deadline) -> kb.RemovalPhaseRecord:
    return kb.RemovalPhaseRecord(
        board_name="x", removal_id="r", mode=mode, phase=kb.RemovalPhase.FENCED,
        epoch=1, quiescence_deadline=deadline,
    )


@pytest.mark.parametrize(
    "held,deadline,now,expected",
    [
        (0, 100, 200, kb.QuiescenceDeadlineAction.ADVANCE),
        (0, None, 200, kb.QuiescenceDeadlineAction.ADVANCE),
        (2, 300, 200, kb.QuiescenceDeadlineAction.WAIT),
        (2, 200, 200, kb.QuiescenceDeadlineAction.FORCE_END_HELD),  # permanent, at deadline
        (2, None, 200, kb.QuiescenceDeadlineAction.WAIT),  # no deadline recorded yet
    ],
)
def test_deadline_action_table_permanent(held, deadline, now, expected):
    record = _record(kb.RemovalMode.PERMANENT, deadline)
    assert kb.quiescence_deadline_action(record, held=held, now=now) == expected


def test_deadline_action_reversible_abandons_instead_of_force_ending():
    record = _record(kb.RemovalMode.REVERSIBLE, deadline=200)
    action = kb.quiescence_deadline_action(record, held=2, now=200)
    assert action is kb.QuiescenceDeadlineAction.ABANDON


def test_qb_1d_i_an_expiry_alone_is_not_authority_at_or_after_the_deadline_yet_unreached():
    """QB-1d-i: before the deadline, Held stays Held regardless of any
    expiry having passed — the decision table only ever consults the
    RECORDED DEADLINE and ``held``, never a per-reservation expiry."""
    record = _record(kb.RemovalMode.PERMANENT, deadline=1000)
    # "now" is far past when some expiry could plausibly have passed, but
    # still before the recorded deadline: must WAIT, not act.
    assert kb.quiescence_deadline_action(record, held=1, now=999) == (
        kb.QuiescenceDeadlineAction.WAIT
    )


def test_deadline_action_does_not_perform_force_ending_or_abandonment():
    """The function returns a decision only; it must not mutate anything."""
    record = _record(kb.RemovalMode.PERMANENT, deadline=1)
    action = kb.quiescence_deadline_action(record, held=1, now=100)
    assert action is kb.QuiescenceDeadlineAction.FORCE_END_HELD
    # No board, no register — nothing to check on disk; the point is that
    # this call took no board argument at all and touched no store.


# ---------------------------------------------------------------------------
# Phases 4, 5, 7 — thin, ordered, forward-only wrappers
# ---------------------------------------------------------------------------

def test_carried_released_swept_advance_in_order(fence_home):
    removal_id = _fenced("linear")
    assert kb.advance_removal_to_quiesced("linear", removal_id=removal_id).success

    carried = kb.advance_removal_to_carried("linear", removal_id=removal_id)
    assert carried.success and carried.record.phase == kb.RemovalPhase.CARRIED

    released = kb.advance_removal_to_released("linear", removal_id=removal_id)
    assert released.success and released.record.phase == kb.RemovalPhase.RELEASED

    applied = kb.advance_removal_to_applied("linear", removal_id=removal_id)
    assert applied.success and applied.record.phase == kb.RemovalPhase.APPLIED

    swept = kb.advance_removal_to_swept("linear", removal_id=removal_id)
    assert swept.success and swept.record.phase == kb.RemovalPhase.SWEPT


def test_carried_refuses_to_skip_ahead_of_quiesced(fence_home):
    removal_id = _fenced("skip-carried")
    # Still at Fenced — Carried's own from_phase is Quiesced, so a record
    # still sitting at Fenced is a mismatch (refused, not silently applied).
    result = kb.advance_removal_to_carried("skip-carried", removal_id=removal_id)
    assert result.success is False
    assert result.outcome is kb.RemovalAdvanceOutcome.REFUSED_STALE
    assert kb.get_removal_phase_record("skip-carried").phase == kb.RemovalPhase.FENCED


# ---------------------------------------------------------------------------
# Phase 6 — Applied: TQ-2 as a predicate on the transition itself
# ---------------------------------------------------------------------------

def test_applied_refuses_and_reenters_quiescence_over_a_real_held_claim(fence_home):
    removal_id = _fenced_with_task("applied-held")
    _released_then_held_again("applied-held", removal_id)

    result = kb.advance_removal_to_applied("applied-held", removal_id=removal_id)

    assert result.success is False
    assert result.outcome is kb.RemovalAdvanceOutcome.REFUSED_REENTER_QUIESCENCE
    # Phase does NOT move backwards: it stays at Released.
    assert kb.get_removal_phase_record("applied-held").phase == kb.RemovalPhase.RELEASED


def test_a_caller_cannot_force_applied_over_a_live_claim(fence_home):
    """The predicate is read authoritatively inside the transition, so
    there is no caller-supplied count for a caller to lie with — not
    through the driver, not through the public entry point, and not
    through the raw private primitive either."""
    removal_id = _fenced_with_task("applied-forced", mode="permanent")
    _released_then_held_again("applied-forced", removal_id)

    with pytest.raises(TypeError):
        kb.advance_removal_to_applied(
            "applied-forced", removal_id=removal_id,
            held=0, task_removals_in_window=0,
        )

    raw = kb._advance_removal_phase(
        "applied-forced", removal_id=removal_id,
        from_phase=kb.RemovalPhase.RELEASED, to_phase=kb.RemovalPhase.APPLIED,
    )
    assert raw.success is False
    assert raw.outcome is kb.RemovalAdvanceOutcome.REFUSED_REENTER_QUIESCENCE
    assert kb.get_removal_phase_record("applied-forced").phase == kb.RemovalPhase.RELEASED

    # The task really is still running with a live claim lock.
    row = task_row(kb.kanban_db_path(board="applied-forced"), _only_task("applied-forced"))
    assert row["status"] == "running"
    assert row["claim_lock"] is not None


def test_applied_refuses_when_a_task_removal_is_in_its_window(fence_home):
    removal_id = _fenced("applied-task-removal")
    assert kb.advance_removal_to_quiesced("applied-task-removal", removal_id=removal_id).success
    assert kb.advance_removal_to_carried("applied-task-removal", removal_id=removal_id).success
    assert kb.advance_removal_to_released("applied-task-removal", removal_id=removal_id).success
    _open_task_removal_window("applied-task-removal")

    result = kb.advance_removal_to_applied(
        "applied-task-removal", removal_id=removal_id
    )

    assert result.success is False
    assert result.outcome is kb.RemovalAdvanceOutcome.REFUSED_REENTER_QUIESCENCE
    assert "task removal" in result.message
    assert kb.get_removal_phase_record(
        "applied-task-removal"
    ).phase == kb.RemovalPhase.RELEASED


def test_quiesced_refuses_when_a_task_removal_is_in_its_window(fence_home):
    """The count comes from the durable place that would record one, not
    from a literal zero in the source."""
    removal_id = _fenced("quiesce-task-removal")
    _open_task_removal_window("quiesce-task-removal")

    result = kb.advance_removal_to_quiesced(
        "quiesce-task-removal", removal_id=removal_id
    )

    assert result.success is False
    assert result.outcome is kb.RemovalAdvanceOutcome.REFUSED_REENTER_QUIESCENCE
    assert "1 task removal(s) in window" in result.message
    assert kb.get_removal_phase_record(
        "quiesce-task-removal"
    ).phase == kb.RemovalPhase.FENCED


def test_an_applied_repeat_refuses_when_the_release_fact_is_gone(fence_home):
    """Applied's meaning includes that Release completed: a repeat re-reads
    that durable fact instead of trusting the phase column."""
    removal_id = _fenced("applied-repeat-factless")
    for driver in (
        kb.advance_removal_to_quiesced,
        kb.advance_removal_to_carried,
        kb.advance_removal_to_released,
        kb.advance_removal_to_applied,
    ):
        assert driver("applied-repeat-factless", removal_id=removal_id).success
    damage_phase_column_behind_the_primitive(
        "applied-repeat-factless", "release_completed_at", None
    )

    repeat = kb.advance_removal_to_applied(
        "applied-repeat-factless", removal_id=removal_id
    )

    assert repeat.success is False
    assert repeat.outcome is kb.RemovalAdvanceOutcome.REFUSED_WORK_INCOMPLETE
    assert repeat.transitioned is False


def test_an_applied_repeat_re_records_the_marker_it_finds_missing(fence_home):
    """A record at Applied whose §6.6 marker is gone owes the marker again.

    The absence of a record of the deferred work is not evidence that it
    happened, so the repeat rolls the real work forward — it writes the
    OUTSTANDING marker back — rather than no-opping over the silence.
    """
    removal_id = _fenced("applied-repeat-marker", mode="permanent")
    for driver in (
        kb.advance_removal_to_quiesced,
        kb.advance_removal_to_carried,
        kb.advance_removal_to_released,
        kb.advance_removal_to_applied,
    ):
        assert driver("applied-repeat-marker", removal_id=removal_id).success
    damage_phase_column_behind_the_primitive(
        "applied-repeat-marker", "applied_mode_content", None
    )

    repeat = kb.advance_removal_to_applied(
        "applied-repeat-marker", removal_id=removal_id
    )

    assert repeat.success is True
    with read_only(kb.register_db_path()) as conn:
        raw = conn.execute(
            "SELECT applied_mode_content FROM board_removal_phase "
            "WHERE board_name = ?", ("applied-repeat-marker",),
        ).fetchone()["applied_mode_content"]
    assert raw, "the outstanding marker was not re-recorded"
    record = kb.get_removal_phase_record("applied-repeat-marker")
    assert record.phase == kb.RemovalPhase.APPLIED
    assert kb.applied_mode_content_is_outstanding(record) is True


def test_applied_succeeds_when_the_predicate_is_clear(fence_home):
    removal_id = _fenced("applied-clear")
    assert kb.advance_removal_to_quiesced("applied-clear", removal_id=removal_id).success
    assert kb.advance_removal_to_carried("applied-clear", removal_id=removal_id).success
    assert kb.advance_removal_to_released("applied-clear", removal_id=removal_id).success

    result = kb.advance_removal_to_applied("applied-clear", removal_id=removal_id)

    assert result.success is True
    assert result.record.phase == kb.RemovalPhase.APPLIED


# ---------------------------------------------------------------------------
# Phase 8 — Done: the terminal transition, mode-dependent target lifecycle
# ---------------------------------------------------------------------------

def test_done_moves_a_reversible_removal_to_archived(fence_home):
    removal_id = _fenced("done-reversible", mode="reversible")
    _to_swept("done-reversible", removal_id)
    # §7.3's content has to have HAPPENED before ``archived`` is a true
    # fact about the world; Done refuses while it is outstanding (that
    # refusal is asserted in
    # test_kanban_removal_phase_applied_mode_content).
    apply_mode_specific_content("done-reversible", removal_id)

    result = kb.complete_removal(
        "done-reversible", removal_id=removal_id, outcome="ended-cleanly"
    )

    assert result.success is True
    assert result.transitioned is True
    assert result.record.phase == kb.RemovalPhase.DONE
    assert result.record.outcome == "ended-cleanly"
    assert register_row("done-reversible")["lifecycle"] == "archived"


def test_done_moves_a_permanent_removal_to_hard_removed(fence_home):
    removal_id = _fenced("done-permanent", mode="permanent")
    _to_swept("done-permanent", removal_id)
    apply_mode_specific_content("done-permanent", removal_id)

    result = kb.complete_removal(
        "done-permanent", removal_id=removal_id, outcome="ended-cleanly"
    )

    assert result.success is True
    assert register_row("done-permanent")["lifecycle"] == "hard-removed"


def test_done_refuses_before_swept_is_reached(fence_home):
    removal_id = _fenced("done-too-early")
    assert kb.advance_removal_to_quiesced("done-too-early", removal_id=removal_id).success

    result = kb.complete_removal("done-too-early", removal_id=removal_id, outcome="x")

    assert result.success is False
    assert register_row("done-too-early")["lifecycle"] == "removing"


def test_done_is_idempotent_on_repeat(fence_home):
    removal_id = _fenced("done-repeat", mode="reversible")
    _to_swept("done-repeat", removal_id)
    apply_mode_specific_content("done-repeat", removal_id)

    first = kb.complete_removal("done-repeat", removal_id=removal_id, outcome="ended-cleanly")
    assert first.transitioned is True

    second = kb.complete_removal("done-repeat", removal_id=removal_id, outcome="ended-cleanly")
    assert second.success is True
    assert second.transitioned is False
    assert register_row("done-repeat")["lifecycle"] == "archived"


def test_done_refuses_a_mismatched_removal_id(fence_home):
    removal_id = _fenced("done-mismatch", mode="reversible")
    _to_swept("done-mismatch", removal_id)

    result = kb.complete_removal("done-mismatch", removal_id="nope", outcome="x")

    assert result.success is False
    assert register_row("done-mismatch")["lifecycle"] == "removing"


def test_done_moves_the_phase_through_the_named_compare_and_set(fence_home):
    """§6.8: the phase move is not an inline UPDATE of its own.

    Proven by behaviour, not by reading source: the primitive's guards
    are what refuse a Done whose meaning is not true. With the sweep's
    durable completion fact cleared, Done's precondition fails and
    NEITHER table moves — which an inline UPDATE that only matched on
    ``phase = 'swept'`` would not have noticed.
    """
    removal_id = _fenced("done-via-cas", mode="reversible")
    _to_swept("done-via-cas", removal_id)
    with kb.register_connect() as conn:
        conn.execute(
            "UPDATE board_removal_phase SET sweep_completed_at = NULL "
            "WHERE board_name = ?", ("done-via-cas",),
        )

    result = kb.complete_removal(
        "done-via-cas", removal_id=removal_id, outcome="ended-cleanly"
    )

    assert result.success is False
    assert result.outcome is kb.RemovalAdvanceOutcome.REFUSED_PRECONDITION
    assert kb.get_removal_phase_record("done-via-cas").phase == kb.RemovalPhase.SWEPT
    assert register_row("done-via-cas")["lifecycle"] == "removing"


def test_done_refuses_and_rolls_back_both_tables_when_the_phase_cas_loses(fence_home):
    """One transaction across both tables: if the phase move is refused,
    the lifecycle move it was paired with is rolled back too."""
    removal_id = _fenced("done-atomic", mode="permanent")
    _to_swept("done-atomic", removal_id)
    with kb.register_connect() as conn:
        conn.execute(
            "UPDATE board_removal_phase SET removal_id = 'someone-elses-run' "
            "WHERE board_name = ?", ("done-atomic",),
        )

    result = kb.complete_removal(
        "done-atomic", removal_id=removal_id, outcome="ended-cleanly"
    )

    assert result.success is False
    assert register_row("done-atomic")["lifecycle"] == "removing"


# ---------------------------------------------------------------------------
# Critical: an unreadable store is INDETERMINATE, never a convenient zero
# ---------------------------------------------------------------------------

def test_quiesced_refuses_when_the_board_store_cannot_be_read(fence_home):
    removal_id = _fenced("quiesce-unreadable")
    _make_store_unreadable("quiesce-unreadable")

    result = kb.advance_removal_to_quiesced(
        "quiesce-unreadable", removal_id=removal_id
    )

    assert result.success is False
    assert result.outcome is kb.RemovalAdvanceOutcome.REFUSED_INDETERMINATE
    assert result.held is None
    assert kb.get_removal_phase_record(
        "quiesce-unreadable"
    ).phase == kb.RemovalPhase.FENCED


def test_quiesced_refuses_when_the_board_store_is_missing_unaccounted_for(fence_home):
    removal_id = _fenced("quiesce-vanished")
    kb.kanban_db_path(board="quiesce-vanished").unlink()

    result = kb.advance_removal_to_quiesced(
        "quiesce-vanished", removal_id=removal_id
    )

    assert result.success is False
    assert result.outcome is kb.RemovalAdvanceOutcome.REFUSED_INDETERMINATE
    assert kb.get_removal_phase_record(
        "quiesce-vanished"
    ).phase == kb.RemovalPhase.FENCED


def test_applied_refuses_when_the_board_store_cannot_be_read(fence_home):
    removal_id = _fenced("applied-unreadable")
    assert kb.advance_removal_to_quiesced("applied-unreadable", removal_id=removal_id).success
    assert kb.advance_removal_to_carried("applied-unreadable", removal_id=removal_id).success
    assert kb.advance_removal_to_released("applied-unreadable", removal_id=removal_id).success
    _make_store_unreadable("applied-unreadable")

    result = kb.advance_removal_to_applied("applied-unreadable", removal_id=removal_id)

    assert result.success is False
    assert result.outcome is kb.RemovalAdvanceOutcome.REFUSED_INDETERMINATE
    assert kb.get_removal_phase_record(
        "applied-unreadable"
    ).phase == kb.RemovalPhase.RELEASED


def test_the_absent_store_of_a_hard_removed_board_is_a_positive_zero(fence_home):
    """A legitimately destroyed store is not the same as an unreadable one.

    Once the register records the board hard-removed, its store's absence
    is accounted for by durable state and reads as a real zero — the
    distinction the fail-closed rule must model rather than collapse.

    The store's absence here is the real §7.2 destruction (recorded
    through the seam Done requires), not a test ``unlink``: the register
    can only say ``hard-removed`` once the store is genuinely gone.
    """
    removal_id = _fenced("gone-for-good", mode="permanent")
    _to_swept("gone-for-good", removal_id)
    apply_mode_specific_content("gone-for-good", removal_id)
    assert not kb.kanban_db_path(board="gone-for-good").exists()
    assert kb.complete_removal(
        "gone-for-good", removal_id=removal_id, outcome="hard-removed"
    ).success

    reading = kb._read_quiescence_predicate("gone-for-good")
    assert reading.status is kb.DurableReadStatus.OK
    assert reading.held == 0
    assert reading.task_removals_in_window == 0


def test_a_refused_predicate_is_recorded_durably_on_the_phase_record(fence_home):
    removal_id = _held_board("refusal-recorded")

    result = kb.advance_removal_to_quiesced("refusal-recorded", removal_id=removal_id)
    assert result.success is False

    record = kb.get_removal_phase_record("refusal-recorded")
    assert record.phase == kb.RemovalPhase.FENCED
    assert record.refusal_outcome is not None
    assert "reservation(s) held" in record.refusal_outcome
