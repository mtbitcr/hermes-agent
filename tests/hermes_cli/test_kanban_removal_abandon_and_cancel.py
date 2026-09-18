"""Deadline abandonment, cancel-before-Applied, wrong-digest refusal, and
no-new-state-machine assertions (design revision 5, §9.2).
"""

from __future__ import annotations

import contextlib
import io
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
    read_only,
    ready_task,
    register_row,
    start_removal,
    permanent_confirmation,
    apply_mode_specific_content,
    gate_row,
    task_row,
)


def _fenced_board(slug, *, mode="reversible"):
    create_fenced_board(slug)
    intent = start_removal(slug, mode=mode)
    assert intent.success, intent.message
    fenced = kb.advance_removal_to_fenced(slug, removal_id=intent.removal_id)
    assert fenced.success, fenced.message
    return intent.removal_id


def _held_board(slug, *, mode="reversible"):
    create_fenced_board(slug)
    conn = kb.connect(board=slug)
    task_id = ready_task(conn)
    assert kb.claim_task(conn, task_id) is not None
    conn.close()
    intent = start_removal(slug, mode=mode)
    assert intent.success, intent.message
    fenced = kb.advance_removal_to_fenced(slug, removal_id=intent.removal_id)
    assert fenced.success, fenced.message
    return intent.removal_id, task_id


# ---------------------------------------------------------------------------
# 1. Deadline abandonment WITH a live owner present
# ---------------------------------------------------------------------------

def test_deadline_abandonment_with_live_owner(fence_home):
    """Board restored to live, retained copy gone, reservation STILL HELD,
    outcome reported — and idempotent on second run."""
    slug = "abandon-live"
    removal_id, task_id = _held_board(slug)

    kb._record_phase_fields(
        slug, removal_id=removal_id,
        expect_phase=kb.RemovalPhase.FENCED,
        quiescence_deadline=int(time.time()) - 10,
    )
    # §9.2 ships through this path now: at the deadline the reversible
    # advance abandons rather than refusing with "later work".
    q = kb.advance_removal_to_quiesced(slug, removal_id=removal_id)
    assert q.success, q.message
    assert q.action == kb.QuiescenceDeadlineAction.ABANDON

    r = kb.abandon_or_cancel_removal(slug, removal_id=removal_id, reason="deadline")
    assert r.success, r.message

    entry = kb.get_register_entry(slug)
    assert entry.lifecycle == kb.BoardLifecycle.LIVE
    assert entry.gate_move == kb.GateMove.SETTLED

    g = gate_row(kb.kanban_db_path(board=slug))
    assert g is not None and g[0] == "open" and g[1] == entry.epoch

    assert not kb.reversible_retained_path(slug, removal_id).exists()

    row = task_row(kb.kanban_db_path(board=slug), task_id)
    assert row["status"] == "running" and row["claim_lock"] is not None

    rec = kb.get_removal_phase_record(slug)
    assert rec.outcome == kb.REMOVAL_OUTCOME_ABANDONED

    assert kb.resume_removal(slug).action == kb.RemovalRecoveryAction.NO_ACTION_COMPLETE

    # Idempotence: second run, no duplicate journal entry.
    j1 = rec.journal()
    r2 = kb.abandon_or_cancel_removal(slug, removal_id=removal_id, reason="deadline")
    assert r2.success and r2.already_done
    j2 = kb.get_removal_phase_record(slug).journal()
    assert len(j1["items"]) == len(j2["items"])


# ---------------------------------------------------------------------------
# 2. Cancel after Applied — refused, forward recovery
# ---------------------------------------------------------------------------

def test_cancel_after_applied_refused_recovers_forward(fence_home):
    slug = "cancel-past"
    removal_id = _fenced_board(slug)
    kb.advance_removal_to_quiesced(slug, removal_id=removal_id)
    kb.advance_removal_to_carried(slug, removal_id=removal_id)
    kb.advance_removal_to_released(slug, removal_id=removal_id)
    kb.advance_removal_to_applied(slug, removal_id=removal_id)

    r = kb.abandon_or_cancel_removal(slug, removal_id=removal_id, reason="cancel")
    assert not r.success
    assert "at or past Applied" in r.message and "forward" in r.message

    rec = kb.get_removal_phase_record(slug)
    assert rec.refusal_outcome is not None
    assert json.loads(rec.refusal_outcome)["outcome"] == "refused-cancel-past-applied"

    d = kb.resume_removal(slug)
    assert d.action not in (
        kb.RemovalRecoveryAction.NO_ACTION_COMPLETE, kb.RemovalRecoveryAction.NONE,
    )

    apply_mode_specific_content(slug, removal_id)
    assert kb.advance_removal_to_swept(slug, removal_id=removal_id).success
    assert kb.complete_removal(slug, removal_id=removal_id, outcome="archived").success


# ---------------------------------------------------------------------------
# 3. Wrong-digest permanent confirmation — refused before content touched
# ---------------------------------------------------------------------------

def test_wrong_digest_refused_before_content(fence_home):
    slug = "wrong-dig"
    create_fenced_board(slug)
    conn = kb.connect(board=slug)
    task_id = ready_task(conn)
    conn.close()

    wrong = kb.PermanentRemovalConfirmation(
        confirmed=True, statement_digest="0" * 64,
        confirmed_by="test", confirmed_at=int(time.time()), board=slug,
    )
    r = kb.remove_board_fenced(slug, mode="permanent", permanent_confirmation=wrong)
    assert not r.success and r.refusal_reason == "unbound-confirmation"

    assert kb.board_dir(slug).exists()
    with read_only(kb.kanban_db_path(board=slug)) as rc:
        assert rc.execute("SELECT id FROM tasks WHERE id=?", (task_id,)).fetchone()
    assert register_row(slug)["lifecycle"] == kb.BoardLifecycle.LIVE.value

    ref = kb.get_recorded_removal_refusal(slug)
    assert ref is not None
    assert kb.get_removal_phase_record(slug) is None


def test_wrong_board_confirmation_refused(fence_home):
    create_fenced_board("ba")
    create_fenced_board("bb")
    ca = permanent_confirmation("ba")
    assert ca.board == "ba"
    r = kb.remove_board_fenced("bb", mode="permanent", permanent_confirmation=ca)
    assert not r.success and r.refusal_reason == "unbound-confirmation"
    assert register_row("bb")["lifecycle"] == kb.BoardLifecycle.LIVE.value


def test_board_bound_confirmation_checked_at_intent(fence_home):
    create_fenced_board("tgt")
    disc = kb.permanent_removal_disclosure("tgt")
    fake = kb.PermanentRemovalConfirmation(
        confirmed=True, statement_digest=disc.statement_digest,
        confirmed_by="t", confirmed_at=int(time.time()), board="other",
    )
    r = kb.record_removal_intent("tgt", mode="permanent", permanent_confirmation=fake)
    assert not r.success and "other" in r.message


# ---------------------------------------------------------------------------
# 4. Cancel before Applied — same live state, no new state machine
# ---------------------------------------------------------------------------

def test_cancel_before_applied_reaches_live_state(fence_home):
    slug = "cancel-bf"
    removal_id = _fenced_board(slug)
    kb.advance_removal_to_quiesced(slug, removal_id=removal_id)

    r = kb.abandon_or_cancel_removal(slug, removal_id=removal_id, reason="cancel")
    assert r.success, r.message

    entry = kb.get_register_entry(slug)
    assert entry.lifecycle == kb.BoardLifecycle.LIVE
    g = gate_row(kb.kanban_db_path(board=slug))
    assert g and g[0] == "open"

    rec = kb.get_removal_phase_record(slug)
    assert rec.outcome == kb.REMOVAL_OUTCOME_CANCELLED
    assert kb.resume_removal(slug).action == kb.RemovalRecoveryAction.NO_ACTION_COMPLETE

    # Owner-facing read surface (--json) reports the outcome.
    import argparse
    from hermes_cli import kanban as kanban_cli
    wrap = argparse.ArgumentParser(prog="t", add_help=False)
    kanban_cli.build_parser(wrap.add_subparsers(dest="_"))
    args = wrap.parse_args(["kanban", "boards", "removal-phase", slug, "--json"])
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        assert kanban_cli.kanban_command(args) == 0
    payload = json.loads(buf.getvalue())
    assert payload["outcome"] == kb.REMOVAL_OUTCOME_CANCELLED


def test_cancel_at_fenced(fence_home):
    slug = "cancel-f"
    removal_id = _fenced_board(slug)
    r = kb.abandon_or_cancel_removal(slug, removal_id=removal_id, reason="cancel")
    assert r.success
    assert kb.get_register_entry(slug).lifecycle == kb.BoardLifecycle.LIVE
    assert kb.get_removal_phase_record(slug).outcome == kb.REMOVAL_OUTCOME_CANCELLED


def test_cancel_permanent_refused(fence_home):
    slug = "cancel-p"
    create_fenced_board(slug)
    intent = start_removal(slug, mode="permanent")
    assert intent.success
    kb.advance_removal_to_fenced(slug, removal_id=intent.removal_id)
    r = kb.abandon_or_cancel_removal(slug, removal_id=intent.removal_id, reason="cancel")
    assert not r.success and "permanent" in r.message.lower()


def test_gate_reopen_failure_is_not_success(fence_home, monkeypatch):
    """Gate reopen fail → not success, refusal recorded, retry succeeds."""
    slug = "gate-fail"
    removal_id = _fenced_board(slug)

    original_commit = kb.commit_gate_state
    monkeypatch.setattr(kb, "commit_gate_state",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("unavailable")))
    r = kb.abandon_or_cancel_removal(slug, removal_id=removal_id, reason="deadline")

    assert not r.success
    assert "not yet usable" in r.message

    rec = kb.get_removal_phase_record(slug)
    assert rec.refusal_outcome is not None
    assert json.loads(rec.refusal_outcome)["outcome"] == "gate-reopen-failed"
    assert kb.get_register_entry(slug).lifecycle == kb.BoardLifecycle.LIVE
    assert rec.outcome == kb.REMOVAL_OUTCOME_ABANDONED

    monkeypatch.setattr(kb, "commit_gate_state", original_commit)
    r2 = kb.abandon_or_cancel_removal(slug, removal_id=removal_id, reason="deadline")
    assert r2.success, r2.message
    assert gate_row(kb.kanban_db_path(board=slug))[0] == "open"

    conn = kb.connect(board=slug)
    try:
        assert kb.create_task(conn, title="after abandonment", assignee="w")
    finally:
        conn.close()


def test_no_partial_state_across_restore(fence_home):
    """Register+outcome consistent after success; and when the gate step
    failed, resume prescribes the reopen rather than reporting complete."""
    slug_ok = "nps-ok"
    rid = _fenced_board(slug_ok)
    r = kb.abandon_or_cancel_removal(slug_ok, removal_id=rid, reason="deadline")
    assert r.success, r.message

    assert kb.get_register_entry(slug_ok).lifecycle == kb.BoardLifecycle.LIVE
    assert kb.get_removal_phase_record(slug_ok).outcome == kb.REMOVAL_OUTCOME_ABANDONED
    assert kb.resume_removal(slug_ok).action == kb.RemovalRecoveryAction.NO_ACTION_COMPLETE

    slug_int = "nps-int"
    rid_int = _fenced_board(slug_int)
    original = kb.commit_gate_state
    kb.commit_gate_state = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("fail"))
    try:
        assert not kb.abandon_or_cancel_removal(
            slug_int, removal_id=rid_int, reason="deadline").success
    finally:
        kb.commit_gate_state = original

    assert kb.get_register_entry(slug_int).lifecycle == kb.BoardLifecycle.LIVE
    assert kb.resume_removal(slug_int).action == (
        kb.RemovalRecoveryAction.ROLL_FORWARD_REOPEN_AFTER_ABANDON
    )


# ---------------------------------------------------------------------------
# 5. The restart path repairs a board the abandonment left unusable
# ---------------------------------------------------------------------------

def test_restart_repairs_board_after_failed_gate_reopen(fence_home, monkeypatch):
    """A gate reopen that does not take effect is repaired by the
    PRODUCTION loop, and the board is provably usable afterwards."""
    slug = "reopen-restart"
    removal_id = _fenced_board(slug)

    # The gate commit does not take effect; nothing else changes.
    original_commit = kb.commit_gate_state
    monkeypatch.setattr(kb, "commit_gate_state", lambda *a, **k: False)
    r = kb.abandon_or_cancel_removal(slug, removal_id=removal_id, reason="deadline")
    assert not r.success
    rec = kb.get_removal_phase_record(slug)
    assert json.loads(rec.refusal_outcome)["outcome"] == "gate-reopen-failed"
    assert gate_row(kb.kanban_db_path(board=slug))[0] != "open"

    # While the gate is closed the board is NOT complete, whatever the
    # outcome and the register say.
    assert kb.resume_removal(slug).action == (
        kb.RemovalRecoveryAction.ROLL_FORWARD_REOPEN_AFTER_ABANDON
    )

    monkeypatch.setattr(kb, "commit_gate_state", original_commit)
    driven = kb.drive_removal(slug)
    assert driven.success, driven.message
    assert driven.action == kb.REMOVAL_OUTCOME_ABANDONED

    entry = kb.get_register_entry(slug)
    assert gate_row(kb.kanban_db_path(board=slug)) == ("open", entry.epoch)
    assert kb.resume_removal(slug).action == kb.RemovalRecoveryAction.NO_ACTION_COMPLETE

    conn = kb.connect(board=slug)
    try:
        assert kb.create_task(conn, title="after the restart", assignee="w")
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# 6. The deadline reaches abandonment through the shipped loop
# ---------------------------------------------------------------------------

def test_drive_removal_abandons_at_deadline_with_live_owner(fence_home):
    """``drive_removal`` — not the abandon API — carries a reversible
    removal past its deadline, leaving the live owner's claim untouched."""
    slug = "drive-abandon"
    removal_id, task_id = _held_board(slug)
    kb._record_phase_fields(
        slug, removal_id=removal_id,
        expect_phase=kb.RemovalPhase.FENCED,
        quiescence_deadline=int(time.time()) - 10,
    )

    driven = kb.drive_removal(slug)
    assert driven.success, driven.message
    assert driven.action == kb.REMOVAL_OUTCOME_ABANDONED

    rec = kb.get_removal_phase_record(slug)
    assert rec.outcome == kb.REMOVAL_OUTCOME_ABANDONED
    entry = kb.get_register_entry(slug)
    assert entry.lifecycle == kb.BoardLifecycle.LIVE
    assert gate_row(kb.kanban_db_path(board=slug)) == ("open", entry.epoch)

    # The live owner's claim was never treated as absent.
    row = task_row(kb.kanban_db_path(board=slug), task_id)
    assert row["status"] == "running" and row["claim_lock"] is not None

    conn = kb.connect(board=slug)
    try:
        assert kb.create_task(conn, title="after abandonment", assignee="w")
    finally:
        conn.close()

    # Idempotent: same end state, no duplicate journal entry.
    items = len(rec.journal()["items"])
    again = kb.drive_removal(slug)
    assert again.success and again.action == kb.REMOVAL_OUTCOME_ABANDONED
    assert len(kb.get_removal_phase_record(slug).journal()["items"]) == items
    assert kb.get_register_entry(slug).lifecycle == kb.BoardLifecycle.LIVE
    assert task_row(kb.kanban_db_path(board=slug), task_id)["claim_lock"] is not None
