"""Crash and restart, for real: SIGKILL in a subprocess at every boundary.

Each test launches a REAL subprocess that drives ``hermes kanban boards
rm`` through the shipped CLI entry point and then SIGKILLs itself at a
named point — every adjacent phase boundary and every intra-apply point,
including after the permanent apply and after the terminal transition.
A SECOND real subprocess then re-invokes the SAME entry point, and the
removal has to finish.

Nothing here hand-invokes the next driver: if the entry point does not
consult durable state and roll forward on its own, the restart fails and
so does the test.
"""

from __future__ import annotations

import json
import signal
import subprocess
import sys
from pathlib import Path

import pytest

_WORKTREE = Path(__file__).resolve().parents[2]
if str(_WORKTREE) not in sys.path:
    sys.path.insert(0, str(_WORKTREE))

from hermes_cli import kanban_db as kb
from tests.hermes_cli._kanban_removal_crash_child import CRASH_POINTS
from tests.hermes_cli._kanban_fence_support import (
    add_linked_work_area,
    create_fenced_board,
    make_git_repo,
    read_only,
    ready_task,
    record_git_receipt,
    record_run_advance,
    register_row,
)

_CHILD = Path(__file__).parent / "_kanban_removal_crash_child.py"

# The crash points that belong to the permanent apply. A reversible
# removal never reaches them, so they are only exercised for permanent.
_PERMANENT_ONLY = {
    "apply-after-storage",
    "apply-after-work-area",
    "apply-after-deregister",
    "after-apply-content",
    "after-done",
    # The §12 receipt's prepare-then-apply protocol is permanent mode's:
    # a reversible removal never enters it, so a child told to crash there
    # would run to completion instead of dying.
    "after-prepared-receipt",
    "after-terminal-transition",
}

# The crash points INSIDE the durable prepare-then-apply receipt protocol,
# mapped to the durable state each one is expected to leave behind: the
# register lifecycle, and the status of the receipt row.
_RECEIPT_PROTOCOL_POINTS = {
    "after-prepared-receipt": (
        kb.BoardLifecycle.REMOVING, kb.RECEIPT_STATUS_PREPARED,
    ),
    "after-terminal-transition": (
        kb.BoardLifecycle.HARD_REMOVED, kb.RECEIPT_STATUS_PREPARED,
    ),
    "after-done": (
        kb.BoardLifecycle.HARD_REMOVED, kb.RECEIPT_STATUS_APPLIED,
    ),
}


def _run_child(home: Path, slug: str, crash: str, mode: str):
    """Launch the real CLI in a real subprocess."""
    return subprocess.run(
        [sys.executable, str(_CHILD), str(home), slug, crash, mode],
        capture_output=True, text=True, timeout=180,
        cwd=str(_WORKTREE),
        env={
            "PATH": "/usr/bin:/bin:/usr/local/bin",
            "PYTHONPATH": str(_WORKTREE),
            "PYTHONHASHSEED": "0",
            "TZ": "UTC",
            "LANG": "C.UTF-8",
            "HOME": str(home),
        },
    )


@pytest.fixture
def board_with_work(fence_home, tmp_path):
    """A fenced board with a real task, a real work area and a real advance."""
    slug = "crash-restart"
    repo = tmp_path / "repo"
    base = make_git_repo(repo)
    create_fenced_board(slug)
    conn = kb.connect(board=slug)
    task_id = ready_task(conn)
    branch = f"hermes/{task_id}"
    work_area, head = add_linked_work_area(repo, task_id, branch=branch)
    record_git_receipt(
        conn, task_id, workspace_path=work_area, branch_name=branch,
        base_commit=base, head_commit=head,
    )
    record_run_advance(
        conn, task_id, branch=branch, base_commit=base, head_commit=head,
    )
    conn.close()
    return {
        "home": fence_home,
        "slug": slug,
        "task": task_id,
        "repo": repo,
        "work_area": work_area,
        "base": base,
        "head": head,
    }


def _receipt_rows(slug: str) -> list:
    path = kb.archive_db_path()
    if not path.exists():
        return []
    with read_only(path) as conn:
        return [
            dict(row) for row in conn.execute(
                "SELECT removal_id, receipt_payload, status, receipt_digest "
                "FROM board_removal_receipt WHERE board_name = ?",
                (slug,),
            )
        ]


def _assert_permanent_removal_is_complete(setup: dict) -> dict:
    """Every terminal fact, read from durable state after the restart."""
    slug = setup["slug"]
    record = kb.get_removal_phase_record(slug)
    assert record is not None, "the removal record vanished"
    assert record.phase is kb.RemovalPhase.DONE, record.phase
    assert record.outcome == "completed"
    assert register_row(slug)["lifecycle"] == (
        kb.BoardLifecycle.HARD_REMOVED.value
    )
    assert not kb.board_dir(slug).exists()
    assert not kb.kanban_db_path(board=slug).exists()
    # The owned work area and its content are gone; the shared container
    # and the commits it holds are not.
    assert not setup["work_area"].exists()
    assert setup["repo"].exists()
    rows = _receipt_rows(slug)
    assert len(rows) == 1, rows
    assert rows[0]["removal_id"] == record.removal_id
    receipt = json.loads(rows[0]["receipt_payload"])
    assert receipt["subject"]["phase"] == kb.RemovalPhase.DONE.value
    assert receipt["subject"]["terminal_lifecycle"] == (
        kb.BoardLifecycle.HARD_REMOVED.value
    )
    assert receipt["verification"]["verdicts"][kb.RECEIPT_VERDICT_FAIL] == 0, (
        receipt["verification"]["checks"]
    )
    reference, = [
        ref for ref in receipt["version_control"]["references"]
        if ref["task"] == setup["task"]
    ]
    assert setup["head"] in reference["board_created_commits"]
    assert setup["base"] not in reference["board_created_commits"]
    return receipt


@pytest.mark.parametrize("crash", sorted(CRASH_POINTS))
def test_permanent_removal_survives_a_sigkill_at_every_point(
    board_with_work, crash
):
    """SIGKILL at *crash*, restart through the real entry point, finish."""
    setup = board_with_work
    slug = setup["slug"]

    crashed = _run_child(setup["home"], slug, crash, "permanent")

    assert crashed.returncode == -signal.SIGKILL, (
        f"the child did not die of SIGKILL at {crash}: "
        f"rc={crashed.returncode} out={crashed.stdout} err={crashed.stderr}"
    )
    assert "CHILD-EXIT" not in crashed.stdout, (
        "the child completed the removal instead of crashing"
    )
    if crash in _RECEIPT_PROTOCOL_POINTS:
        # Inside the prepare-then-apply protocol: the exact receipt content
        # is already durable, and which side of the terminal transition the
        # crash landed on is readable from the two durable facts together.
        lifecycle, status = _RECEIPT_PROTOCOL_POINTS[crash]
        assert register_row(slug)["lifecycle"] == lifecycle.value
        rows = _receipt_rows(slug)
        assert len(rows) == 1, rows
        assert rows[0]["status"] == status
        assert rows[0]["receipt_digest"], "the prepared receipt carries no digest"
        assert json.loads(rows[0]["receipt_payload"])["subject"], (
            "the prepared receipt is not the real, complete content"
        )
    else:
        # Anything short of the receipt protocol must not have claimed
        # a terminal lifecycle, and must not have left a receipt.
        assert register_row(slug)["lifecycle"] == (
            kb.BoardLifecycle.REMOVING.value
        )
        assert _receipt_rows(slug) == [], (
            "a receipt was written before the removal was complete"
        )

    restarted = _run_child(setup["home"], slug, "never", "permanent")

    assert restarted.returncode == 0, (
        f"the restart after {crash} failed: out={restarted.stdout} "
        f"err={restarted.stderr}"
    )
    assert "CHILD-EXIT 0" in restarted.stdout
    _assert_permanent_removal_is_complete(setup)


@pytest.mark.parametrize(
    "crash", sorted(set(CRASH_POINTS) - _PERMANENT_ONLY)
)
def test_reversible_removal_survives_a_sigkill_at_every_point(
    board_with_work, crash
):
    """The same restart story for the reversible path."""
    setup = board_with_work
    slug = setup["slug"]

    crashed = _run_child(setup["home"], slug, crash, "reversible")

    assert crashed.returncode == -signal.SIGKILL, (
        f"the child did not die of SIGKILL at {crash}: "
        f"rc={crashed.returncode} err={crashed.stderr}"
    )
    assert register_row(slug)["lifecycle"] == kb.BoardLifecycle.REMOVING.value

    restarted = _run_child(setup["home"], slug, "never", "reversible")

    assert restarted.returncode == 0, (
        f"the restart after {crash} failed: out={restarted.stdout} "
        f"err={restarted.stderr}"
    )
    record = kb.get_removal_phase_record(slug)
    assert record.phase is kb.RemovalPhase.DONE
    assert record.outcome == "archived"
    assert register_row(slug)["lifecycle"] == kb.BoardLifecycle.ARCHIVED.value
    # §7.3 kept exactly one retained copy, at the identity the removal
    # derived — a restart completes that copy rather than making another.
    retained = kb.reversible_retained_path(slug, record.removal_id)
    assert retained.exists(), "the retained copy is missing"
    siblings = sorted(p.name for p in retained.parent.iterdir())
    assert siblings == [retained.name], siblings
    assert not kb.board_dir(slug).exists()


def test_a_restart_after_the_terminal_transition_applies_the_prepared_receipt(
    board_with_work
):
    """The window between `done` and the receipt's APPLY is itself resumable.

    The terminal state is recorded and the exact receipt content is
    already durable — prepared, byte-final, digested. The roll-forward
    must apply THOSE bytes rather than rebuilding a payload, which is what
    makes recovery from this direction deterministic.
    """
    setup = board_with_work
    slug = setup["slug"]

    crashed = _run_child(
        setup["home"], slug, "after-terminal-transition", "permanent",
    )
    assert crashed.returncode == -signal.SIGKILL, crashed.stderr

    record = kb.get_removal_phase_record(slug)
    assert record.phase is kb.RemovalPhase.DONE
    assert register_row(slug)["lifecycle"] == (
        kb.BoardLifecycle.HARD_REMOVED.value
    )
    before = _receipt_rows(slug)
    assert len(before) == 1, before
    assert before[0]["status"] == kb.RECEIPT_STATUS_PREPARED
    prepared_bytes = before[0]["receipt_payload"]
    prepared_digest = before[0]["receipt_digest"]
    assert prepared_digest, "the prepared receipt carries no digest"
    decision = kb.resume_removal(slug)
    assert decision.action is (
        kb.RemovalRecoveryAction.ROLL_FORWARD_TERMINAL_RECEIPT
    )

    restarted = _run_child(setup["home"], slug, "never", "permanent")

    assert restarted.returncode == 0, restarted.stderr
    _assert_permanent_removal_is_complete(setup)
    after = _receipt_rows(slug)
    assert len(after) == 1, after
    assert after[0]["status"] == kb.RECEIPT_STATUS_APPLIED
    # The exact bytes committed before the transition are the bytes that
    # account for it: the roll-forward applied, it did not regenerate.
    assert after[0]["receipt_payload"] == prepared_bytes
    assert after[0]["receipt_digest"] == prepared_digest


def test_a_restart_after_the_prepare_completes_the_terminal_transition(
    board_with_work
):
    """The other interrupted direction: prepared, terminal state not reached.

    The receipt content is durable and the register still says the board
    is being removed. A restart must complete the transition and apply the
    SAME prepared content — never a second, differently built receipt.
    """
    setup = board_with_work
    slug = setup["slug"]

    crashed = _run_child(
        setup["home"], slug, "after-prepared-receipt", "permanent",
    )
    assert crashed.returncode == -signal.SIGKILL, crashed.stderr

    assert register_row(slug)["lifecycle"] == kb.BoardLifecycle.REMOVING.value
    assert kb.get_removal_phase_record(slug).phase is kb.RemovalPhase.SWEPT
    before = _receipt_rows(slug)
    assert len(before) == 1, before
    assert before[0]["status"] == kb.RECEIPT_STATUS_PREPARED
    prepared_bytes = before[0]["receipt_payload"]

    restarted = _run_child(setup["home"], slug, "never", "permanent")

    assert restarted.returncode == 0, restarted.stderr
    _assert_permanent_removal_is_complete(setup)
    after = _receipt_rows(slug)
    assert len(after) == 1, after
    assert after[0]["status"] == kb.RECEIPT_STATUS_APPLIED
    assert after[0]["receipt_payload"] == prepared_bytes


def test_a_crash_before_the_fence_leaves_the_board_readable(board_with_work):
    """Intent committed, nothing destroyed: the board's content is intact."""
    setup = board_with_work
    slug = setup["slug"]

    crashed = _run_child(setup["home"], slug, "at-intent", "permanent")
    assert crashed.returncode == -signal.SIGKILL, crashed.stderr

    record = kb.get_removal_phase_record(slug)
    assert record.phase is kb.RemovalPhase.INTENT
    # The intent is durable and permanent mode's confirmation went with it.
    assert record.mode is kb.RemovalMode.PERMANENT
    assert record.permanent_confirmed_at is not None
    # Nothing has been destroyed yet, in the board or in the container.
    assert kb.kanban_db_path(board=slug).exists()
    with read_only(kb.kanban_db_path(board=slug)) as conn:
        row = conn.execute(
            "SELECT id FROM tasks WHERE id = ?", (setup["task"],)
        ).fetchone()
    assert row is not None
    assert (setup["work_area"] / f"{setup['task']}.txt").exists()


def test_the_restart_reports_the_point_it_resumed_from(board_with_work):
    """The driver's own account names the recovery point it rolled from."""
    setup = board_with_work
    slug = setup["slug"]
    crashed = _run_child(setup["home"], slug, "after-carried", "permanent")
    assert crashed.returncode == -signal.SIGKILL, crashed.stderr
    assert kb.get_removal_phase_record(slug).phase is kb.RemovalPhase.CARRIED

    result = kb.drive_removal(slug, resumed=True)

    assert result.success, result.message
    assert result.resumed is True
    assert result.steps, "the driver reported no roll-forward steps"
    points = [step["point"] for step in result.steps]
    assert points[0] == kb.RemovalRecoveryPoint.P7.value, points
    assert kb.RemovalRecoveryPoint.P12.value in points, points
    for step in result.steps:
        assert step["action"], step
        assert step["message"], step
