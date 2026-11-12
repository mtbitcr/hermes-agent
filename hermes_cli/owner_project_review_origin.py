"""Recognize an exact accepted review proposal after the owner authorizes a normal plan."""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3

from hermes_cli import kanban_db, owner_project_summary, owner_project_summary_events as events, owner_workspace, projects_db
from hermes_cli.owner_project_review_proposals import _TEXT_LIMITS

logger = logging.getLogger(__name__)


def source_digest(body, value):
    """Bind the accepted result to its frozen input without copying a proposal into another store."""
    return hashlib.sha256(json.dumps(
        {"body": body, "project_review": value}, sort_keys=True, ensure_ascii=False,
        separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")).hexdigest()


def accepted_review_source(project_id):
    """Read the pinned accepted proposal without returning another approval authority.

    The review writes the root's registry and board, so both are read there: a worker's profile home and its
    board pins name other files.
    """
    from hermes_cli import owner_project_review as review
    from hermes_cli.owner_project_review_record import TOTAL_INPUT_CAP_BYTES

    try:
        settings = review.read_settings()
        if not settings.enabled:
            return None
        with events._read_only(kanban_db.kanban_home() / "projects.db") as registry:
            source = projects_db.get_project_review_source(registry, project_id)
            project = projects_db.get_project(registry, project_id)
        if source is None or project is None or not {project_id, project.slug}.intersection(settings.allowlist):
            return None
        with events._read_only(owner_project_summary._summary_board_path()) as board:
            task = kanban_db.get_task(board, source["source_task_id"])
            run = kanban_db.get_run(board, source["source_run_id"])
        if (task is None or not review._made_for(task, project_id)
                or task.assignee != owner_project_summary.PLANNING_ROLE or task.status not in ("done", "archived")
                or review._key_date(task.idempotency_key, project_id) != source["review_date"]
                or run is None or run.task_id != task.id or run.outcome != "completed"
                or not isinstance(run.metadata, dict)):
            return None
        value = run.metadata.get("project_review")
        if (not isinstance(value, dict) or len(json.dumps(value, allow_nan=False).encode()) > TOTAL_INPUT_CAP_BYTES
                or source_digest(task.body, value) != source["source_digest"]):
            return None
        proposal = value.get("plan_proposal")
        if not isinstance(proposal, dict) or set(proposal) != set(_TEXT_LIMITS) | {"later_milestones", "changes"}:
            return None
        return project, task, proposal, source["review_date"]
    except (OSError, sqlite3.Error, TypeError, ValueError, KeyError, owner_workspace.OwnerWorkspaceError):
        logger.debug("Daily review source could not be confirmed")
    return None


def confirmed_plan_trigger(project_id, request_digest):
    """Keep owner_request unless its approved payload exactly matches an accepted daily proposal.

    The caller keeps its original approval payload and request digest. This read changes only audit provenance.
    An absent, changed or unreadable source cannot claim scheduled_review.
    """
    from agent.redact import redact_sensitive_text

    try:
        accepted = accepted_review_source(project_id)
        if accepted is None:
            return "owner_request"
        _project, _task, proposal, _day = accepted
        texts = {name: redact_sensitive_text(owner_workspace._bounded_text(proposal[name], name, limit=limit), force=True)
                 for name, limit in _TEXT_LIMITS.items()}
        changes, risk = owner_workspace._normalize_project_changes(proposal["changes"])
        expected = {"project_id": project_id, "trigger": "owner_request", **texts,
                    "later_milestones": owner_workspace._normalize_later_milestones(proposal["later_milestones"]),
                    "risk_level": risk, "changes": changes}
        if owner_workspace._digest(expected) == request_digest:
            return "scheduled_review"
    except (OSError, sqlite3.Error, TypeError, ValueError, KeyError, owner_workspace.OwnerWorkspaceError):
        logger.debug("Daily review origin could not be confirmed")
    return "owner_request"
