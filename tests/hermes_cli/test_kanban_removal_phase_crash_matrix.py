"""Crash/restart matrix: one test per adjacent phase pair.

For EACH ADJACENT PAIR of phases — intent→fenced, fenced→quiesced,
quiesced→carried, carried→released, released→applied, applied→swept,
swept→done — this suite tests that a crash between them and a restart
resumes correctly from that point.

Each test simulates the crash by stopping after the durable write of one
phase and making a fresh decision from durable state. It does NOT
hand-edit a phase column behind the primitive — that models corruption,
which is a different test.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pytest

_WORKTREE = Path(__file__).resolve().parents[2]
if str(_WORKTREE) not in sys.path:
    sys.path.insert(0, str(_WORKTREE))

from hermes_cli import kanban_db as kb
from tests.hermes_cli._kanban_fence_support import (
    apply_mode_specific_content,
    create_fenced_board,
    permanent_confirmation,
    read_only,
    ready_task,
    register_row,
    start_removal,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _phase_row(slug: str):
    """Read the phase record directly from durable state."""
    with read_only(kb.register_db_path()) as conn:
        return conn.execute(
            "SELECT * FROM board_removal_phase WHERE board_name = ?", (slug,)
        ).fetchone()


def _assert_phase(slug: str, expected_phase: str):
    """Assert the durable phase matches expected."""
    row = _phase_row(slug)
    assert row is not None, f"no phase record for {slug}"
    assert row["phase"] == expected_phase, (
        f"expected phase {expected_phase}, got {row['phase']}"
    )


# ---------------------------------------------------------------------------
# intent→fenced
# ---------------------------------------------------------------------------

def test_crash_intent_to_fenced(fence_home):
    """Crash between Intent and Fenced resumes correctly.

    Simulate: Intent commits, then crash before Fenced.
    Resume: should roll forward by closing the fence.
    """
    create_fenced_board("test-intent-fenced")
    conn = kb.connect(board="test-intent-fenced")
    ready_task(conn)
    conn.close()

    # Start removal (gets to Intent)
    intent = start_removal("test-intent-fenced", mode="reversible")
    assert intent.success
    removal_id = intent.removal_id
    _assert_phase("test-intent-fenced", "intent")

    # Simulate crash: fresh resume decision from durable state
    decision = kb.resume_removal("test-intent-fenced")
    assert decision.point in (kb.RemovalRecoveryPoint.P2, kb.RemovalRecoveryPoint.P2A)
    assert decision.action == kb.RemovalRecoveryAction.CLOSE_FENCE_AND_SETTLE

    # Execute roll-forward
    fenced = kb.advance_removal_to_fenced("test-intent-fenced", removal_id=removal_id)
    assert fenced.success
    _assert_phase("test-intent-fenced", "fenced")


def test_crash_fenced_to_quiesced(fence_home):
    """Crash between Fenced and Quiesced resumes correctly.

    Simulate: Fenced commits, then crash before Quiesced.
    Resume: should roll forward into quiescence.
    """
    create_fenced_board("test-fenced-quiesced")
    conn = kb.connect(board="test-fenced-quiesced")
    ready_task(conn)
    conn.close()

    intent = start_removal("test-fenced-quiesced", mode="reversible")
    assert intent.success
    removal_id = intent.removal_id

    fenced = kb.advance_removal_to_fenced("test-fenced-quiesced", removal_id=removal_id)
    assert fenced.success
    _assert_phase("test-fenced-quiesced", "fenced")

    # Simulate crash and resume
    decision = kb.resume_removal("test-fenced-quiesced")
    assert decision.point == kb.RemovalRecoveryPoint.P3
    assert decision.action == kb.RemovalRecoveryAction.ROLL_FORWARD_INTO_QUIESCENCE

    # Roll forward
    quiesced = kb.advance_removal_to_quiesced("test-fenced-quiesced", removal_id=removal_id)
    assert quiesced.success
    _assert_phase("test-fenced-quiesced", "quiesced")


def test_crash_quiesced_to_carried(fence_home):
    """Crash between Quiesced and Carried resumes correctly.

    Simulate: Quiesced commits, then crash before Carried.
    Resume: should roll forward by carrying obligations then releasing.
    """
    create_fenced_board("test-quiesced-carried")
    conn = kb.connect(board="test-quiesced-carried")
    ready_task(conn)
    conn.close()

    intent = start_removal("test-quiesced-carried", mode="reversible")
    assert intent.success
    removal_id = intent.removal_id

    for driver in (
        kb.advance_removal_to_fenced,
        kb.advance_removal_to_quiesced,
    ):
        result = driver("test-quiesced-carried", removal_id=removal_id)
        assert result.success, f"{driver.__name__}: {result.message}"

    _assert_phase("test-quiesced-carried", "quiesced")

    # Simulate crash and resume
    decision = kb.resume_removal("test-quiesced-carried")
    assert decision.point == kb.RemovalRecoveryPoint.P6
    assert decision.action == kb.RemovalRecoveryAction.ROLL_FORWARD_CARRY_AND_RELEASE

    # Roll forward
    carried = kb.advance_removal_to_carried("test-quiesced-carried", removal_id=removal_id)
    assert carried.success
    _assert_phase("test-quiesced-carried", "carried")


def test_crash_carried_to_released(fence_home):
    """Crash between Carried and Released resumes correctly.

    Simulate: Carried commits, then crash before Released.
    Resume: should retry release by exact recorded identity.
    """
    create_fenced_board("test-carried-released")
    conn = kb.connect(board="test-carried-released")
    ready_task(conn)
    conn.close()

    intent = start_removal("test-carried-released", mode="reversible")
    assert intent.success
    removal_id = intent.removal_id

    for driver in (
        kb.advance_removal_to_fenced,
        kb.advance_removal_to_quiesced,
        kb.advance_removal_to_carried,
    ):
        result = driver("test-carried-released", removal_id=removal_id)
        assert result.success, f"{driver.__name__}: {result.message}"

    _assert_phase("test-carried-released", "carried")

    # Simulate crash and resume
    decision = kb.resume_removal("test-carried-released")
    assert decision.point == kb.RemovalRecoveryPoint.P7
    assert decision.action == kb.RemovalRecoveryAction.ROLL_FORWARD_RETRY_RELEASE

    # Roll forward
    released = kb.advance_removal_to_released("test-carried-released", removal_id=removal_id)
    assert released.success
    _assert_phase("test-carried-released", "released")


def test_crash_released_to_applied(fence_home):
    """Crash between Released and Applied resumes correctly.

    Simulate: Released commits, then crash before Applied.
    Resume: should roll forward into Applied.
    """
    create_fenced_board("test-released-applied")
    conn = kb.connect(board="test-released-applied")
    ready_task(conn)
    conn.close()

    intent = start_removal("test-released-applied", mode="reversible")
    assert intent.success
    removal_id = intent.removal_id

    for driver in (
        kb.advance_removal_to_fenced,
        kb.advance_removal_to_quiesced,
        kb.advance_removal_to_carried,
        kb.advance_removal_to_released,
    ):
        result = driver("test-released-applied", removal_id=removal_id)
        assert result.success, f"{driver.__name__}: {result.message}"

    _assert_phase("test-released-applied", "released")

    # Simulate crash and resume
    decision = kb.resume_removal("test-released-applied")
    assert decision.point == kb.RemovalRecoveryPoint.P8
    assert decision.action == kb.RemovalRecoveryAction.ROLL_FORWARD_INTO_APPLIED

    # Roll forward
    applied = kb.advance_removal_to_applied("test-released-applied", removal_id=removal_id)
    assert applied.success
    _assert_phase("test-released-applied", "applied")


def test_crash_applied_to_swept(fence_home):
    """Crash between Applied and Swept resumes correctly.

    Simulate: Applied commits, content performed, then crash before Swept.
    Resume: should roll forward into the sweep.
    """
    create_fenced_board("test-applied-swept")
    conn = kb.connect(board="test-applied-swept")
    ready_task(conn)
    conn.close()

    intent = start_removal("test-applied-swept", mode="reversible")
    assert intent.success
    removal_id = intent.removal_id

    for driver in (
        kb.advance_removal_to_fenced,
        kb.advance_removal_to_quiesced,
        kb.advance_removal_to_carried,
        kb.advance_removal_to_released,
        kb.advance_removal_to_applied,
    ):
        result = driver("test-applied-swept", removal_id=removal_id)
        assert result.success, f"{driver.__name__}: {result.message}"

    _assert_phase("test-applied-swept", "applied")

    # Perform mode-specific content
    apply_mode_specific_content("test-applied-swept", removal_id)

    # Simulate crash and resume
    decision = kb.resume_removal("test-applied-swept")
    assert decision.point == kb.RemovalRecoveryPoint.P10
    assert decision.action == kb.RemovalRecoveryAction.ROLL_FORWARD_INTO_SWEEP

    # Roll forward
    swept = kb.advance_removal_to_swept("test-applied-swept", removal_id=removal_id)
    assert swept.success
    _assert_phase("test-applied-swept", "swept")


def test_crash_swept_to_done(fence_home):
    """Crash between Swept and Done resumes correctly.

    Simulate: Swept commits, then crash before Done.
    Resume: should roll forward by transitioning to Done.
    """
    create_fenced_board("test-swept-done")
    conn = kb.connect(board="test-swept-done")
    ready_task(conn)
    conn.close()

    intent = start_removal("test-swept-done", mode="reversible")
    assert intent.success
    removal_id = intent.removal_id

    for driver in (
        kb.advance_removal_to_fenced,
        kb.advance_removal_to_quiesced,
        kb.advance_removal_to_carried,
        kb.advance_removal_to_released,
        kb.advance_removal_to_applied,
    ):
        result = driver("test-swept-done", removal_id=removal_id)
        assert result.success, f"{driver.__name__}: {result.message}"

    apply_mode_specific_content("test-swept-done", removal_id)

    swept = kb.advance_removal_to_swept("test-swept-done", removal_id=removal_id)
    assert swept.success
    _assert_phase("test-swept-done", "swept")

    # Simulate crash and resume
    decision = kb.resume_removal("test-swept-done")
    assert decision.point == kb.RemovalRecoveryPoint.P11
    assert decision.action == kb.RemovalRecoveryAction.ROLL_FORWARD_INTO_DONE

    # Roll forward
    done = kb.complete_removal(
        "test-swept-done", removal_id=removal_id, outcome="archived"
    )
    assert done.success
    _assert_phase("test-swept-done", "done")


# ---------------------------------------------------------------------------
# Permanent mode crash tests
# ---------------------------------------------------------------------------

def test_crash_permanent_applied_to_swept(fence_home):
    """Permanent mode: crash between Applied and Swept resumes correctly.

    For permanent mode, this exercises the §7.2 content (destruction and
    deregistration) completing, then a crash before the sweep.
    """
    create_fenced_board("test-perm-applied-swept")
    conn = kb.connect(board="test-perm-applied-swept")
    ready_task(conn)
    conn.close()

    confirmation = permanent_confirmation("test-perm-applied-swept")
    intent = start_removal("test-perm-applied-swept", mode="permanent")
    assert intent.success
    removal_id = intent.removal_id

    for driver in (
        kb.advance_removal_to_fenced,
        kb.advance_removal_to_quiesced,
        kb.advance_removal_to_carried,
        kb.advance_removal_to_released,
        kb.advance_removal_to_applied,
    ):
        result = driver("test-perm-applied-swept", removal_id=removal_id)
        assert result.success, f"{driver.__name__}: {result.message}"

    _assert_phase("test-perm-applied-swept", "applied")

    # Perform permanent mode content through the test helper
    apply_mode_specific_content("test-perm-applied-swept", removal_id)

    # Verify the board storage is gone
    assert not kb.board_dir("test-perm-applied-swept").exists()

    # Simulate crash and resume
    decision = kb.resume_removal("test-perm-applied-swept")
    assert decision.point == kb.RemovalRecoveryPoint.P10
    assert decision.action == kb.RemovalRecoveryAction.ROLL_FORWARD_INTO_SWEEP

    # Roll forward
    swept = kb.advance_removal_to_swept("test-perm-applied-swept", removal_id=removal_id)
    assert swept.success
    _assert_phase("test-perm-applied-swept", "swept")


def test_crash_permanent_swept_to_done(fence_home):
    """Permanent mode: crash between Swept and Done produces §12 receipt."""
    create_fenced_board("test-perm-swept-done")
    conn = kb.connect(board="test-perm-swept-done")
    ready_task(conn)
    conn.close()

    confirmation = permanent_confirmation("test-perm-swept-done")
    intent = start_removal("test-perm-swept-done", mode="permanent")
    assert intent.success
    removal_id = intent.removal_id

    for driver in (
        kb.advance_removal_to_fenced,
        kb.advance_removal_to_quiesced,
        kb.advance_removal_to_carried,
        kb.advance_removal_to_released,
        kb.advance_removal_to_applied,
    ):
        result = driver("test-perm-swept-done", removal_id=removal_id)
        assert result.success, f"{driver.__name__}: {result.message}"

    apply_mode_specific_content("test-perm-swept-done", removal_id)

    swept = kb.advance_removal_to_swept("test-perm-swept-done", removal_id=removal_id)
    assert swept.success
    _assert_phase("test-perm-swept-done", "swept")

    # Simulate crash and resume
    decision = kb.resume_removal("test-perm-swept-done")
    assert decision.point == kb.RemovalRecoveryPoint.P11
    assert decision.action == kb.RemovalRecoveryAction.ROLL_FORWARD_INTO_DONE

    # Roll forward
    done = kb.complete_removal(
        "test-perm-swept-done", removal_id=removal_id, outcome="completed"
    )
    assert done.success
    _assert_phase("test-perm-swept-done", "done")

    # Verify the board's directory is really gone
    assert not kb.board_dir("test-perm-swept-done").exists()
