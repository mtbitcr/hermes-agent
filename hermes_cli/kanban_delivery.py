"""Delivery records, the approval proof and the integration-card hook.

When a review-required build card is approved from the review lane and
``kanban.delivery.enabled`` is true, :func:`record_approved_delivery` records
ONE delivery for the source card and its approved head and creates ONE
integration card, assigned to ``kanban.delivery.integration_profile``.
``complete_task`` calls it once, from its approval branch, inside the
approval's own transaction, so the approval, the record and the card commit
or roll back together. With the setting off, or no integration profile, it
returns before touching anything and the approval behaves exactly as before.

The approval proof is the T1 fence of :mod:`hermes_cli.kanban_delivery_fences`
applied to facts read only from the kernel's own records: the task row, its
event log, its runs and the git repository the card was built in. Worker
prose (handover text, comments, summaries, run metadata) is never read.
"""

from __future__ import annotations

import json
import logging
import re
import sqlite3
import subprocess
import time
from pathlib import Path
from typing import Optional

from hermes_cli import kanban_db as kb
from hermes_cli.kanban_delivery_fences import Decision, decide_publish

logger = logging.getLogger(__name__)

# The integration card's idempotency key is this prefix + source card id +
# ":" + approved head, so one source card and head can only ever name one card.
DELIVERY_KEY_PREFIX = "delivery:"

_SHA = re.compile(r"[0-9a-f]{40}")

# Nothing is published at approval: no pull request is recorded and no remote
# branch is read, so only the approval half of T1 can decide.
_NO_LEDGER = {
    "pull_request": None, "head": None, "state": None, "head_is_ancestor": False,
}
_NO_REMOTE = {"branch_head": None, "open_pulls": []}


def delivery_key(source_task_id: str, head: str) -> str:
    return DELIVERY_KEY_PREFIX + source_task_id + ":" + head


def delivery_settings() -> tuple[bool, Optional[str]]:
    """``(enabled, integration_profile)`` from ``kanban.delivery``; off on any doubt."""
    try:
        from hermes_cli.config import load_config

        section = (load_config() or {}).get("kanban", {}).get("delivery") or {}
        enabled = section.get("enabled") is True
        profile = section.get("integration_profile")
    except Exception:
        return False, None
    profile = profile.strip() if isinstance(profile, str) else ""
    return enabled, profile or None


def _sha(value) -> Optional[str]:
    text = str(value or "").strip().lower()
    return text if _SHA.fullmatch(text) else None


def _repository(workspace_path) -> Optional[Path]:
    """The card's worktree or, once the clean worktree is removed after the
    approval, the repository the kernel cut it from (``<repo>/.worktrees/<id>``)."""
    path = Path(workspace_path) if workspace_path else None
    if path is None or not path.is_absolute():
        return None
    for candidate in (path, path.parent.parent if path.parent.name == ".worktrees" else None):
        if candidate is not None and candidate.is_dir():
            return candidate
    return None


def _git_confirms(workdir: Optional[Path], *args: str) -> bool:
    if workdir is None:
        return False
    try:
        kb._git_output(workdir, *args)
    except (kb.WorktreeScopeError, OSError, subprocess.SubprocessError):
        return False
    return True


def approval_facts(
    conn: sqlite3.Connection, task_id: str, bound_head: Optional[str],
) -> dict:
    """The T1 approval facts for ``task_id``, from kernel records and git only."""
    row = conn.execute(
        "SELECT status, head_commit, base_commit, workspace_path FROM tasks "
        "WHERE id = ? AND task_kind = 'work'",
        (task_id,),
    ).fetchone()
    status = row["status"] if row is not None else None
    head = _sha(row["head_commit"]) if row is not None else None
    base = _sha(row["base_commit"]) if row is not None else None

    # The same cycle rule as task_review_states: the latest handover R, the
    # latest completion C after it, and no invalidating event between them.
    # The reviewer's own claim out of review is part of the cycle it reviews.
    requested = kb._latest_review_handover_id(conn, task_id)
    completed = conn.execute(
        "SELECT id, run_id FROM task_events "
        "WHERE task_id = ? AND kind = 'completed' ORDER BY id DESC LIMIT 1",
        (task_id,),
    ).fetchone()
    invalidations = []
    if requested is not None:
        kinds = kb._REVIEW_APPROVAL_INVALIDATING_EVENT_KINDS
        for event in conn.execute(
            "SELECT id, kind, payload FROM task_events WHERE task_id = ? "
            f"AND id > ? AND kind IN ({','.join('?' for _ in kinds)})",
            (task_id, requested, *kinds),
        ):
            if event["kind"] == "claimed":
                try:
                    payload = json.loads(event["payload"] or "{}")
                except (json.JSONDecodeError, TypeError):
                    payload = {}
                if isinstance(payload, dict) and payload.get("source_status") == "review":
                    continue
            invalidations.append(int(event["id"]))
    completed_id = int(completed["id"]) if completed is not None else None
    approved = (
        requested is not None
        and completed_id is not None
        and completed_id > requested
        and not any(event_id < completed_id for event_id in invalidations)
    )
    reopened = completed_id is not None and any(
        event_id > completed_id for event_id in invalidations
    )

    reviewer = None
    if completed is not None and completed["run_id"] is not None:
        run = conn.execute(
            "SELECT profile FROM task_runs WHERE id = ? AND task_id = ?",
            (completed["run_id"], task_id),
        ).fetchone()
        reviewer = run["profile"] if run is not None else None
    implementer, _ = kb._latest_review_provenance(conn, task_id)

    workdir = _repository(row["workspace_path"]) if row is not None else None
    head_exists = head is not None and _git_confirms(
        workdir, "cat-file", "-e", f"{head}^{{commit}}",
    )
    base_is_ancestor = head_exists and base is not None and _git_confirms(
        workdir, "merge-base", "--is-ancestor", base, head,
    )
    return {
        "source_done": status == "done",
        "approved_by_review_lane": approved,
        "head_commit": head,
        "review_head": _sha(kb._latest_review_head_provenance(conn, task_id)),
        "bound_head": _sha(bound_head),
        "reviewer": reviewer,
        "implementer": implementer,
        "reopened_after_approval": reopened,
        "head_exists": head_exists,
        "base": base,
        "base_is_ancestor": base_is_ancestor,
    }


def prove_approval(
    conn: sqlite3.Connection, task_id: str, bound_head: Optional[str],
) -> Decision:
    """T1 on the kernel's records: may ``bound_head`` be delivered for ``task_id``?"""
    return decide_publish(
        approval_facts(conn, task_id, bound_head), dict(_NO_LEDGER), dict(_NO_REMOTE),
    )


def record_approved_delivery(conn: sqlite3.Connection, task_id: str) -> Optional[str]:
    """Record the delivery of an approved card's head and create its one integration card.

    Returns the integration card id, or ``None`` when delivery is off, the card
    carries no review requirement, or the approval proof refuses. A second call
    for the same source card and head returns the card the first one created.
    """
    enabled, profile = delivery_settings()
    if not enabled or profile is None:
        return None
    with kb.write_txn(conn, allow_nested=True):
        row = conn.execute(
            "SELECT requires_review, head_commit FROM tasks "
            "WHERE id = ? AND task_kind = 'work'",
            (task_id,),
        ).fetchone()
        if row is None or not row["requires_review"]:
            return None
        head = _sha(row["head_commit"])
        proof = prove_approval(conn, task_id, head)
        if not proof.allowed:
            logger.warning(
                "kanban delivery: no delivery for %s: %s (%s)",
                task_id, proof.code, proof.detail,
            )
            return None
        approval = conn.execute(
            "SELECT run_id FROM task_events "
            "WHERE task_id = ? AND kind = 'completed' ORDER BY id DESC LIMIT 1",
            (task_id,),
        ).fetchone()
        conn.execute(
            "INSERT OR IGNORE INTO kanban_deliveries "
            "(source_task_id, source_head, approval_run_id, created_at) "
            "VALUES (?, ?, ?, ?)",
            (task_id, head, approval["run_id"], int(time.time())),
        )
        record = conn.execute(
            "SELECT id, integration_task_id FROM kanban_deliveries "
            "WHERE source_task_id = ? AND source_head = ?",
            (task_id, head),
        ).fetchone()
        if record["integration_task_id"]:
            return record["integration_task_id"]
        key = delivery_key(task_id, head)
        integration_id = kb.create_task(
            conn,
            title=f"Integrate {task_id} at {head[:12]}",
            body=(
                f"Approved source card: {task_id}\n"
                f"Approved head: {head}\n"
                f"Delivery key: {key}"
            ),
            assignee=profile,
            created_by="kanban-delivery",
            owned_paths=[],
            idempotency_key=key,
        )
        conn.execute(
            "UPDATE kanban_deliveries SET integration_task_id = ? WHERE id = ?",
            (integration_id, record["id"]),
        )
        return integration_id
