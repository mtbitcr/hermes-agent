"""Refusals of a saved-result handover, and the rollback of the task branch."""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path
from typing import Optional

_log = logging.getLogger(__name__)


def refuse_early(reason: str, saved_result_attachment_id: Optional[int]) -> bool:
    """Refuse a completion before anything is written.

    A saved-result handover (``saved_result_attachment_id`` set) gets
    ``_SavedResultHandoverRefused(reason)``, so its run records this reason
    rather than a bare refusal; every other caller keeps ``False``.
    """
    # Late import: the facade imports this module at load time.
    from hermes_cli.kanban_db import _SavedResultHandoverRefused

    if saved_result_attachment_id is not None:
        raise _SavedResultHandoverRefused(reason)
    return False


def rollback_task_branch(
    workspace: Path,
    branch: Optional[str],
    original: str,
    materialized: Optional[str],
) -> None:
    """Move the task branch back to ``original``, and nothing else.

    With a ``materialized`` head, the branch's own ref moves with an old-value
    check (``update-ref refs/heads/<branch> <original> <materialized>``), so a
    ref that has moved since is left where it is. The worktree is reset only
    when it has that branch checked out and is clean; otherwise it is left
    alone, and the reason logged. Without one, the worktree is reset to
    ``original`` only when it has the task branch checked out; with any other
    checkout this raises ``WorktreeScopeError`` and changes nothing.
    """
    # Late import: the facade imports this module at load time.
    from hermes_cli.kanban_db import WorktreeScopeError, _git_mutation, _git_output

    try:
        branch = (branch or "").strip()
        if not branch:
            raise WorktreeScopeError(
                "cannot restore a task worktree with no task branch"
            )
        checked_out = str(_git_output(workspace, "branch", "--show-current")).strip()
        on_task_branch = checked_out == branch
        checkout = checked_out or "a detached HEAD"
        if materialized is None:
            if not on_task_branch:
                raise WorktreeScopeError(
                    f"refusing to restore {workspace}: it has {checkout} checked "
                    f"out, not the task branch {branch}"
                )
            subprocess.run(
                ["git", "-C", str(workspace), "merge", "--abort"],
                capture_output=True,
                timeout=30,
                check=False,
            )
            _git_mutation(workspace, "reset", "--hard", original)
        else:
            if on_task_branch and _git_output(
                workspace, "status", "--porcelain=v1", "-z", binary=True
            ):
                _log.warning(
                    "kanban: left %s alone: it has uncommitted changes", workspace,
                )
                return
            try:
                _git_mutation(
                    workspace, "update-ref", "-m", "kanban: roll back materialization",
                    f"refs/heads/{branch}", original, materialized,
                )
            except WorktreeScopeError:
                head = str(
                    _git_output(
                        workspace, "rev-parse", "--verify", f"refs/heads/{branch}",
                    )
                ).strip()
                if head == materialized:
                    raise
                _log.warning(
                    "kanban: left %s alone: %s moved on to %s", workspace, branch, head,
                )
                return
            if not on_task_branch:
                _log.warning(
                    "kanban: moved %s back but left %s alone: it has %s checked out",
                    branch, workspace, checkout,
                )
                return
            _git_mutation(workspace, "reset", "--hard")
        dirty = _git_output(
            workspace, "status", "--porcelain=v1", "-z", binary=True
        )
        if dirty:
            raise WorktreeScopeError(
                "isolated worktree remained dirty after kernel rollback"
            )
    except WorktreeScopeError:
        raise
    except Exception as exc:
        raise WorktreeScopeError(
            f"could not restore isolated worktree after failed handoff: {exc}"
        ) from exc
