"""Creation-time provenance for the commits the KERNEL itself creates.

Real git repository, real kernel handoff (``_materialize_remote_worktree_
handoff`` then ``complete_task``), then what durable state says: the record
exists as soon as the commit does, carrying the task/run/board/project/
generation that produced it, and the created list comes from those records
rather than from the ``base..head`` range — so a foreign commit inside that
range is never claimed and never stops being UNVERIFIED.
"""

from __future__ import annotations

import sys
from pathlib import Path

_WORKTREE = Path(__file__).resolve().parents[2]
if str(_WORKTREE) not in sys.path:
    sys.path.insert(0, str(_WORKTREE))

from hermes_cli import kanban_db as kb
from tests.hermes_cli._kanban_fence_support import (
    FOREIGN_GIT_IDENTITY, create_fenced_board, git, make_git_repo,
)

PROJECT = "provenance-project"
PATCH = (
    "diff --git a/src/owned/remote.py b/src/owned/remote.py\n"
    "new file mode 100644\nindex 0000000..e69de29\n"
    "--- /dev/null\n+++ b/src/owned/remote.py\n@@ -0,0 +1,1 @@\n+remote = True\n"
).encode("utf-8")


def _handoff_board(slug: str, tmp_path, monkeypatch) -> dict:
    """A fenced board with a claimed worktree run, ready for a kernel commit.

    Every step is a shipped call: the board and its fence, the task, the
    claim, the resolved work area, the recorded base, and the agent-uploaded
    patch attachment the kernel may materialize.
    """
    repo = tmp_path / "repo"
    make_git_repo(repo)
    create_fenced_board(slug)
    monkeypatch.setenv("HERMES_KANBAN_BOARD", slug)
    monkeypatch.setenv("HERMES_KANBAN_ATTACHMENTS_ROOT", str(tmp_path / "at"))
    conn = kb.connect(board=slug)
    task_id = kb.create_task(
        conn, title="remote artifact the kernel materializes", assignee="worker",
        workspace_kind="worktree", workspace_path=str(repo),
        branch_name="feature/remote", owned_paths=["src/owned"],
    )
    with kb.write_txn(conn):
        conn.execute("UPDATE tasks SET project_id = ? WHERE id = ?", (PROJECT, task_id))
    claimed = kb.claim_task(conn, task_id, claimer="worker")
    workspace, branch = kb._resolve_worktree_workspace(claimed)
    kb.set_workspace_path(conn, task_id, workspace)
    kb.set_branch_name(conn, task_id, branch)
    return {
        "slug": slug, "conn": conn, "task": task_id, "branch": branch,
        "workspace": Path(workspace),
        "base": kb.record_worktree_base(conn, task_id, workspace),
        "attachment": kb.store_attachment_bytes(
            conn, task_id, "sandbox-result.patch", PATCH,
            content_type="text/x-diff", uploaded_by="agent",
        ),
    }


def _materialize(setup: dict) -> str:
    """The kernel's own handoff; returns the commit it created."""
    conn, task_id = setup["conn"], setup["task"]
    receipt, _ = kb._materialize_remote_worktree_handoff(
        conn, task_id, patch_attachment_id=setup["attachment"],
        merge_parent_heads=False,
        expected_run_id=kb.get_task(conn, task_id).current_run_id,
    )
    return receipt["materialized_head"]


def _complete(setup: dict) -> None:
    conn, task_id = setup["conn"], setup["task"]
    assert kb.complete_task(
        conn, task_id, summary="materialized",
        patch_attachment_id=setup["attachment"],
        expected_run_id=kb.get_task(conn, task_id).current_run_id,
    )
    conn.close()


def _reference(slug: str, task_id: str) -> dict:
    """The disclosed reference for this task, from durable state."""
    references = kb.permanent_removal_disclosure(slug).prospective_inventory[
        "survives"]["references"]
    match, = [ref for ref in references if ref["task"] == task_id]
    return match


def test_a_kernel_commit_is_recorded_when_it_is_created_not_derived_later(
    fence_home, tmp_path, monkeypatch
):
    """The record exists as soon as the commit does, with its full identity."""
    setup = _handoff_board("kernel-commits", tmp_path, monkeypatch)
    conn, task_id, slug = setup["conn"], setup["task"], setup["slug"]
    run_id = kb.get_task(conn, task_id).current_run_id

    head = _materialize(setup)

    # Nothing has completed: no run receipt names a head and the task's git
    # columns are still empty. The provenance is already durable.
    assert not (kb.get_task(conn, task_id).head_commit or "")
    assert not (kb.latest_run(conn, task_id).metadata or {}).get("execution_receipt")
    row, = kb.kernel_commit_records(slug, subject_id=head)
    assert (row["task_id"], row["run_id"], row["board_name"], row["project_id"],
            row["reference"]) == (task_id, run_id, slug, PROJECT, setup["branch"])
    assert row["generation"] == kb.get_register_entry(slug).epoch
    assert row["kind"], "the record does not say what created the commit"
    assert kb.verify_kernel_creation(head, board_name=slug).verdict == (
        kb.RECEIPT_VERDICT_PASS
    )

    # The disclosed list is built from that record, and completeness is
    # earned from the rows rather than asserted from a source name.
    _complete(setup)
    reference = _reference(slug, task_id)
    assert head in reference["commits"]
    completeness = reference["commit_list_completeness"]
    assert completeness["verdict"] == kb.RECEIPT_VERDICT_PASS
    assert completeness["source"] == kb.KERNEL_COMMIT_PROVENANCE_SOURCE
    assert not completeness["uncovered_commits"]


def test_a_commit_the_kernel_did_not_create_stays_unverified_forever(
    fence_home, tmp_path, monkeypatch
):
    """A foreign commit inside base..head is never claimed, and never rises."""
    setup = _handoff_board("foreign-commits", tmp_path, monkeypatch)
    task_id, slug, workspace = setup["task"], setup["slug"], setup["workspace"]
    # A real commit by a real other identity, made before the kernel's own,
    # so it is genuinely inside the range a base..head sweep would claim.
    stranger = workspace / "src" / "owned" / "stranger.py"
    stranger.parent.mkdir(parents=True, exist_ok=True)
    stranger.write_text("not ours\n", encoding="utf-8")
    git(workspace, "add", "src/owned/stranger.py")
    git(workspace, "commit", "-m", "a stranger's", author=FOREIGN_GIT_IDENTITY)
    foreign = git(workspace, "rev-parse", "HEAD")

    head = _materialize(setup)

    # The range really does contain it. Asked while the work area still
    # exists: completing the task deregisters and removes that directory,
    # so afterwards this git question has nowhere to run.
    assert foreign in git(workspace, "rev-list", head, f"^{setup['base']}").split()

    # The kernel's own list, read from durable state, really does not.
    _complete(setup)
    reference = _reference(slug, task_id)
    assert head in reference["commits"]
    assert foreign not in reference["commits"]

    verdict = kb.verify_kernel_creation(foreign, board_name=slug)
    assert verdict.verdict == kb.RECEIPT_VERDICT_UNVERIFIED
    assert verdict.authorizes_destruction is False
    # Observing it again — with the kernel's other rows on this same board
    # right there — does not upgrade it, and records nothing.
    before = kb.kernel_commit_records(slug)
    assert kb.verify_kernel_creation(foreign).verdict == kb.RECEIPT_VERDICT_UNVERIFIED
    assert kb.verify_kernel_creation(foreign, board_name=slug).verdict == (
        kb.RECEIPT_VERDICT_UNVERIFIED
    )
    assert kb.kernel_commit_records(slug) == before
    assert not kb.kernel_commit_records(slug, subject_id=foreign)
