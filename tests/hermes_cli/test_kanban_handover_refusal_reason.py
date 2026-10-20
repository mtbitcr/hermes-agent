"""A saved-result handover refused before anything is written records why.

``complete_task`` refuses two handovers before it touches the worktree: a card
whose parent is not finished, and a card that must be parked for review with no
reviewer it can be parked for. When the dead-worker scan hands a run's saved
patch over, the run now records that specific reason instead of the generic
"declined the saved patch without giving a reason" line. A plain
``complete_task`` call in the same two states still returns ``False`` and
writes nothing.

Every scenario runs on a fenced board in a temporary home, against a temporary
git repository that carries its own identity.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_WORKTREE = Path(__file__).resolve().parents[2]
if str(_WORKTREE) not in sys.path:
    sys.path.insert(0, str(_WORKTREE))

from hermes_cli import kanban_db as kb
from plugins.dashboard_auth.raphael_workspace import model_policy
from tests.hermes_cli._kanban_fence_support import (
    DEFAULT_GIT_IDENTITY, create_fenced_board, git, make_git_repo,
)

SLUG = "refusal-reasons"
OWNED = "src/owned"
WORKER_PID = 70998
GOVERNED_IMPLEMENTER = "default"
GENERIC_REFUSAL = "without giving a reason"


def _govern(conn, task_id: str) -> None:
    """Make the build owner-governed: receipt-owned under a valid route lock."""
    route = model_policy.task_assignment_for(
        GOVERNED_IMPLEMENTER, "anthropic", "routine",
    )
    lock = kb.mint_policy_lock(
        GOVERNED_IMPLEMENTER, route.provider, route.model, route.reasoning_effort,
        "routine",
    )
    with kb.write_txn(conn):
        conn.execute(
            "UPDATE tasks SET owner_receipt_bound = 1, execution_tier = 'routine', "
            "provider_override = ?, model_override = ?, reasoning_effort = ?, "
            "model_policy_lock = ? WHERE id = ?",
            (route.provider, route.model, route.reasoning_effort, lock, task_id),
        )


@pytest.fixture
def running_build(fence_home, tmp_path, monkeypatch):
    """Make a claimed, running worktree build that requires review.

    ``governed=True`` makes it owner-governed while the policy nominates no
    reviewer, so the review park has no one it can assign the card to.
    """
    opened = []

    def make(*, governed: bool = False) -> dict:
        repo = tmp_path / "repo"
        make_git_repo(repo)
        name, email = DEFAULT_GIT_IDENTITY
        git(repo, "config", "user.name", name)
        git(repo, "config", "user.email", email)
        create_fenced_board(SLUG)
        monkeypatch.setenv("HERMES_KANBAN_BOARD", SLUG)
        monkeypatch.setenv("HERMES_KANBAN_ATTACHMENTS_ROOT", str(tmp_path / "at"))
        conn = kb.connect(board=SLUG)
        opened.append(conn)
        task_id = kb.create_task(
            conn, title="build that saves its patch",
            assignee=GOVERNED_IMPLEMENTER if governed else "worker",
            workspace_kind="worktree", workspace_path=str(repo),
            branch_name="feature/refused", owned_paths=[OWNED],
            requires_review=True,
        )
        if governed:
            monkeypatch.setattr(model_policy, "reviewer_profile_ids", lambda: ())
            _govern(conn, task_id)
            assert kb.policy_resolved_reviewer() is None
        # The scan only judges claims made on this host.
        host = kb._claimer_id().split(":", 1)[0]
        claimed = kb.claim_task(conn, task_id, claimer=f"{host}:w0")
        workspace, branch = kb._resolve_worktree_workspace(claimed)
        kb.set_workspace_path(conn, task_id, workspace)
        kb.set_branch_name(conn, task_id, branch)
        return {
            "conn": conn, "task": task_id, "run": claimed.current_run_id,
            "repo": repo, "branch": branch,
            "base": kb.record_worktree_base(conn, task_id, workspace),
        }

    yield make
    for conn in opened:
        conn.close()


def _unfinished_parent(setup: dict) -> None:
    """Give the build a parent card that is not finished."""
    conn = setup["conn"]
    parent_id = kb.create_task(conn, title="parent of the build", assignee="planner")
    assert kb.complete_task(conn, parent_id)
    kb.link_tasks(conn, parent_id, setup["task"])
    # The minimal stand-in for a reopen surface: done -> todo.
    with kb.write_txn(conn):
        conn.execute(
            "UPDATE tasks SET status = 'todo', completed_at = NULL WHERE id = ?",
            (parent_id,),
        )


def _exit_cleanly_after_saving_a_patch(setup: dict, monkeypatch) -> None:
    """The run saved its patch, then its worker exited 0 without reporting."""
    kb.store_attachment_bytes(
        setup["conn"], setup["task"], "x.patch",
        (
            f"diff --git a/{OWNED}/saved.py b/{OWNED}/saved.py\n"
            "new file mode 100644\n"
            "--- /dev/null\n"
            f"+++ b/{OWNED}/saved.py\n"
            "@@ -0,0 +1 @@\n"
            "+saved = True\n"
        ).encode("utf-8"),
        content_type="text/x-diff", uploaded_by="agent",
        expected_run_id=setup["run"],
    )
    monkeypatch.setattr(kb, "_pid_alive", lambda _pid: False)
    monkeypatch.setenv("HERMES_KANBAN_CRASH_GRACE_SECONDS", "0")
    # The registry is process memory; a fresh one keeps scenarios independent.
    monkeypatch.setattr(kb, "_recent_worker_exits", {})
    kb._set_worker_pid(setup["conn"], setup["task"], WORKER_PID)
    kb._record_worker_exit(WORKER_PID, 0 << 8)


def _parked_refusal(setup: dict) -> str:
    """Read back today's park; return the refusal the run records."""
    conn = setup["conn"]
    task = kb.get_task(conn, setup["task"])
    assert (task.status, task.block_kind) == ("blocked", "needs_input")
    run = kb.get_run(conn, setup["run"])
    assert run.outcome == "completed_unreported"
    # Both refusals come before the worktree is touched.
    assert git(setup["repo"], "rev-parse", setup["branch"]) == setup["base"]
    return run.metadata["unreported_completion"]["handover_refusal"]


def test_saved_result_handover_with_an_unfinished_parent_records_the_parent_reason(
    running_build, monkeypatch,
):
    setup = running_build()
    _unfinished_parent(setup)
    _exit_cleanly_after_saving_a_patch(setup, monkeypatch)

    kb.detect_crashed_workers(setup["conn"])

    refusal = _parked_refusal(setup)
    assert "a parent card is not finished" in refusal
    assert GENERIC_REFUSAL not in refusal


def test_saved_result_handover_with_no_assignable_reviewer_records_that_reason(
    running_build, monkeypatch,
):
    setup = running_build(governed=True)
    unassignable = kb._unassignable_review_park_error(setup["conn"], setup["task"])
    assert unassignable
    _exit_cleanly_after_saving_a_patch(setup, monkeypatch)

    kb.detect_crashed_workers(setup["conn"])

    refusal = _parked_refusal(setup)
    assert refusal == unassignable
    assert GENERIC_REFUSAL not in refusal


@pytest.mark.parametrize("state", ["unfinished_parent", "no_assignable_reviewer"])
def test_plain_completion_in_either_refusal_state_returns_false_and_writes_nothing(
    running_build, state,
):
    setup = running_build(governed=state == "no_assignable_reviewer")
    if state == "unfinished_parent":
        _unfinished_parent(setup)
    conn, task_id = setup["conn"], setup["task"]

    def written() -> tuple:
        return tuple(
            [dict(row) for row in conn.execute(query, (task_id,))]
            for query in (
                "SELECT * FROM tasks WHERE id = ?",
                "SELECT * FROM task_runs WHERE task_id = ? ORDER BY id",
                "SELECT * FROM task_events WHERE task_id = ? ORDER BY id",
            )
        )

    before = written()

    assert kb.complete_task(
        conn, task_id, summary="ready for review", expected_run_id=setup["run"],
    ) is False

    assert written() == before
    assert git(setup["repo"], "rev-parse", setup["branch"]) == setup["base"]
