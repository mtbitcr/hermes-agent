"""Board setting and role gate of the look-only design browser (design-loop plan v2, D3 and B4).

:func:`check_design_access` answers, at every call of ``design_browser``, whether this worker may
use it. It reads only what the kernel wrote: the dispatcher's ``HERMES_KANBAN_TASK``,
``HERMES_KANBAN_RUN_ID`` and ``HERMES_KANBAN_BOARD`` (``HERMES_KANBAN_DB`` only to compare it), and
the task and run rows in that board's own database, opened read-only. Nothing the model supplies
takes part, and the handler's ``task_id`` (its supervisor key) is not the kanban task.

* The task's board must be in ``design_browser.boards``; an empty setting refuses everyone.
* The run must be the task's current run on that board, and the run's profile and the task's
  assignee must both be design roles.
* The reviewer only in a review run (plan U3). The builder only outside a review run, on its own
  task, and only on its own build's preview (plan O10): ``builder_host``. The designer on any task
  of a listed board (owner answer F1).
"""

from __future__ import annotations

import logging
import os
import re
import sqlite3
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

DESIGNER = "raphael-designer"
REVIEWER = "raphael-verifier"
BUILDER = "raphael-claude-worker"
DESIGN_ROLES = frozenset({DESIGNER, REVIEWER, BUILDER})

NOT_IN_TASK = "it runs only inside a kanban task"
NO_BOARDS = "no board is enabled for it (design_browser.boards is empty)"
BOARD_NOT_LISTED = "this board is not in design_browser.boards"
NOT_ON_BOARD = "the task's run is not on this board"
NOT_CURRENT_RUN = "the run is not this task's current run"
NO_DESIGN_ROLE = "this task's assignee and run need a design role"
REVIEW_RUN_ONLY = "the reviewer may use it only in a review run"
NO_REVIEW_RUN = "the builder may not use it in a review run"
NOT_OWN_TASK = "the builder may use it only on its own task"
NO_PREVIEW = "no preview address of this build can be derived"
CHECK_FAILED = "the access check failed"

PAGES_ALIAS_LENGTH = 28  # Cloudflare Pages cuts a branch alias to about this many characters (O10)
_NOT_ALIAS_CHAR = re.compile(r"[^a-z0-9]")
_COMMIT = re.compile(r"[0-9a-fA-F]{8,}")


@dataclass(frozen=True)
class DesignAccess:
    allowed: bool
    reason: str
    role: Optional[str] = None
    builder_host: Optional[str] = None  # the only host the builder may open; None for the others


def pages_branch_alias(branch: str) -> str:
    """The Cloudflare Pages branch alias of ``branch``: lowercase, every character outside a to z
    and 0 to 9 a hyphen, cut to 28 characters, no leading or trailing hyphen."""
    return _NOT_ALIAS_CHAR.sub("-", branch.lower())[:PAGES_ALIAS_LENGTH].strip("-")


def check_design_access(config: Dict[str, Any]) -> DesignAccess:
    """The gate, on the ``design_browser`` config section. Never raises: a failure refuses."""
    try:
        return _check(config)
    except Exception as exc:
        logger.warning("design browser: the access check failed (%s)", type(exc).__name__)
        return DesignAccess(False, CHECK_FAILED)


def _check(config: Dict[str, Any]) -> DesignAccess:
    from hermes_cli import kanban_db as kb

    task_id = os.environ.get("HERMES_KANBAN_TASK", "")
    run = os.environ.get("HERMES_KANBAN_RUN_ID", "").strip()
    try:
        board = kb._normalize_board_slug(os.environ.get("HERMES_KANBAN_BOARD"))
    except ValueError:
        board = None
    if not task_id.strip() or not run.isascii() or not run.isdigit() or int(run) <= 0 or board is None:
        return DesignAccess(False, NOT_IN_TASK)
    run_id = int(run)

    listed = set()
    entries = config.get("boards")
    for entry in entries if isinstance(entries, list) else ():
        if isinstance(entry, str):
            try:
                slug = kb._normalize_board_slug(entry)
            except ValueError:
                continue
            if slug:
                listed.add(slug)
    if not listed:
        return DesignAccess(False, NO_BOARDS)
    if board not in listed:
        return DesignAccess(False, BOARD_NOT_LISTED)

    # The board's own file, never through kanban_db_path (HERMES_KANBAN_DB overrides its argument).
    path = (kb.kanban_home() if board == kb.DEFAULT_BOARD else kb.board_dir(board)) / "kanban.db"
    if not path.is_file():
        return DesignAccess(False, NOT_ON_BOARD)
    pinned = os.environ.get("HERMES_KANBAN_DB", "").strip()
    if pinned and Path(pinned).expanduser().resolve() != path.resolve():
        return DesignAccess(False, NOT_ON_BOARD)

    with closing(sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)) as conn:
        conn.row_factory = sqlite3.Row
        return _check_rows(kb, conn, config, task_id, run_id)


def _check_rows(kb, conn: sqlite3.Connection, config: Dict[str, Any], task_id: str, run_id: int) -> DesignAccess:
    run = kb.get_run(conn, run_id)
    task = kb.get_task(conn, task_id)
    if run is None or task is None or run.task_id != task_id or task.current_run_id != run_id:
        return DesignAccess(False, NOT_CURRENT_RUN)
    role = run.profile
    if role not in DESIGN_ROLES or task.assignee not in DESIGN_ROLES:
        return DesignAccess(False, NO_DESIGN_ROLE)

    if role == REVIEWER:
        if (kb.run_claimed_from_review(conn, task_id, run_id)
                and kb.run_requires_read_only_workspace(conn, task_id, run_id, owned_paths=task.owned_paths)):
            return DesignAccess(True, "allowed", role)
        return DesignAccess(False, REVIEW_RUN_ONLY, role)
    if role == BUILDER:
        if kb.run_claimed_from_review(conn, task_id, run_id):
            return DesignAccess(False, NO_REVIEW_RUN, role)
        if task.assignee != BUILDER:
            return DesignAccess(False, NOT_OWN_TASK, role)
        host = _builder_host(config, task_id, task.branch_name, task.head_commit)
        if host is None:
            return DesignAccess(False, NO_PREVIEW, role)
        return DesignAccess(True, "allowed", role, host)
    return DesignAccess(True, "allowed", role)


def _builder_host(config: Dict[str, Any], task_id: str, branch: Any, head: Any) -> Optional[str]:
    """The build's own preview host, from kernel-recorded data only: the branch alias when it keeps
    the task id, else the commit host; below the one ``*.<apex>`` entry of the allow-list."""
    hosts = config.get("allowed_hosts")
    apexes = [h[2:] for h in hosts if isinstance(h, str) and h.startswith("*.") and h[2:]] \
        if isinstance(hosts, list) else []
    if len(apexes) != 1 or not isinstance(branch, str) or not branch:
        return None
    alias = pages_branch_alias(branch)
    if _NOT_ALIAS_CHAR.sub("-", task_id.lower()) in alias:
        return f"{alias}.{apexes[0]}"
    if isinstance(head, str) and _COMMIT.fullmatch(head):
        return f"{head[:8].lower()}.{apexes[0]}"
    return None
