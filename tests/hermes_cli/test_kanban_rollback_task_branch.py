"""The materialization rollback moves only the task branch.

``_rollback_worktree_materialization`` undoes a commit the kernel itself made in
a task worktree. It moves the task branch's own ref back with an old-value
check and resets the worktree only when the worktree has that branch checked
out and is clean, so another branch checked out there is never rewound.
Without a materialized head it refuses while another branch is checked out.

Every scenario uses a temporary board and a temporary git repository that
carries its own identity.
"""

from __future__ import annotations

import contextlib
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

_WORKTREE = Path(__file__).resolve().parents[2]
if str(_WORKTREE) not in sys.path:
    sys.path.insert(0, str(_WORKTREE))

from hermes_cli import kanban_db as kb
from tests.hermes_cli._kanban_fence_support import git, make_git_repo

OTHER_BRANCH = "other-work"


def _commit(worktree: Path, relative: str, message: str) -> str:
    path = worktree / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"{path.stem} = True\n", encoding="utf-8")
    git(worktree, "add", relative)
    git(worktree, "commit", "-m", message)
    return git(worktree, "rev-parse", "HEAD")


def _branch_heads(worktree: Path) -> dict:
    listed = git(
        worktree, "for-each-ref", "--format=%(refname) %(objectname)", "refs/heads",
    )
    return dict(line.split(" ", 1) for line in listed.splitlines())


@pytest.fixture
def materialized(tmp_path):
    """A claimed worktree task whose branch holds one commit made by the call."""
    repo = tmp_path / "repo"
    make_git_repo(repo)
    git(repo, "config", "user.name", "Rollback Test")
    git(repo, "config", "user.email", "rollback@example.invalid")
    db_path = tmp_path / "kanban.db"
    kb.init_db(db_path=db_path)
    with contextlib.closing(kb.connect(db_path)) as conn:
        task_id = kb.create_task(
            conn, title="rollback", assignee="worker",
            workspace_kind="worktree", workspace_path=str(repo),
            branch_name="feature/rollback", owned_paths=["src/owned"],
        )
        claimed = kb.claim_task(conn, task_id)
        assert claimed is not None
        workspace, branch = kb._resolve_worktree_workspace(claimed)
        kb.set_workspace_path(conn, task_id, workspace)
        kb.set_branch_name(conn, task_id, branch)
        kb.record_worktree_base(conn, task_id, workspace)
        assert git(workspace, "branch", "--show-current") == branch
        original = git(workspace, "rev-parse", "HEAD")
        head = _commit(workspace, "src/owned/materialized.py", "kanban: materialize")
        yield SimpleNamespace(
            conn=conn, task=task_id, worktree=workspace, branch=branch,
            original=original, materialized=head,
        )


def _roll_back(m, *, materialized: bool = True) -> None:
    kb._rollback_worktree_materialization(
        m.conn, m.task, m.original,
        materialized_head=m.materialized if materialized else None,
    )


def test_rollback_with_another_branch_checked_out_moves_only_the_task_branch(
    materialized,
):
    m = materialized
    git(m.worktree, "switch", "-c", OTHER_BRANCH)

    _roll_back(m)

    heads = _branch_heads(m.worktree)
    assert heads[f"refs/heads/{OTHER_BRANCH}"] == m.materialized
    assert heads[f"refs/heads/{m.branch}"] == m.original
    assert git(m.worktree, "branch", "--show-current") == OTHER_BRANCH
    assert (m.worktree / "src" / "owned" / "materialized.py").exists()
    assert git(m.worktree, "status", "--porcelain") == ""


def test_rollback_without_a_materialized_head_refuses_while_another_branch_is_checked_out(
    materialized,
):
    m = materialized
    git(m.worktree, "switch", "-c", OTHER_BRANCH)
    _commit(m.worktree, "src/owned/other.py", "work on another branch")
    before = _branch_heads(m.worktree)

    with pytest.raises(kb.WorktreeScopeError, match=OTHER_BRANCH):
        _roll_back(m, materialized=False)

    assert _branch_heads(m.worktree) == before
    assert git(m.worktree, "branch", "--show-current") == OTHER_BRANCH
    assert (m.worktree / "src" / "owned" / "other.py").exists()
    assert git(m.worktree, "status", "--porcelain") == ""


def test_rollback_after_the_task_branch_moved_past_the_materialized_commit_touches_nothing(
    materialized,
):
    m = materialized
    later = _commit(m.worktree, "src/owned/later.py", "landed after the call")
    before = _branch_heads(m.worktree)

    _roll_back(m)

    assert _branch_heads(m.worktree) == before
    assert git(m.worktree, "rev-parse", "HEAD") == later
    assert (m.worktree / "src" / "owned" / "materialized.py").exists()
    assert (m.worktree / "src" / "owned" / "later.py").exists()
    assert git(m.worktree, "status", "--porcelain") == ""


def test_rollback_on_the_task_branch_at_the_materialized_commit_restores_the_original(
    materialized,
):
    m = materialized

    _roll_back(m)

    assert _branch_heads(m.worktree)[f"refs/heads/{m.branch}"] == m.original
    assert git(m.worktree, "branch", "--show-current") == m.branch
    assert git(m.worktree, "rev-parse", "HEAD") == m.original
    assert not (m.worktree / "src" / "owned" / "materialized.py").exists()
    assert git(m.worktree, "status", "--porcelain") == ""
