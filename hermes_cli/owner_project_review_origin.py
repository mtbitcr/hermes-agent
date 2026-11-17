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


def _checked_review(board, project_id, source):
    """The review task and stored result a registry source row names, while its run and digest still bind them."""
    from hermes_cli import owner_project_review as review
    from hermes_cli.owner_project_review_record import TOTAL_INPUT_CAP_BYTES

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
    return task, value


def _accepted_review(project_id):
    """The pinned accepted review: its Project, task, proposal, review date and stored result."""
    from hermes_cli import owner_project_review as review

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
            checked = _checked_review(board, project_id, source)
        if checked is None:
            return None
        task, value = checked
        return project, task, value["plan_proposal"], source["review_date"], value
    except (OSError, sqlite3.Error, TypeError, ValueError, KeyError, owner_workspace.OwnerWorkspaceError):
        logger.debug("Daily review source could not be confirmed")
    return None


def accepted_review_source(project_id):
    """Read the pinned accepted proposal without returning another approval authority.

    The review writes the root's registry and board, so both are read there: a worker's profile home and its
    board pins name other files.
    """
    accepted = _accepted_review(project_id)
    return None if accepted is None else accepted[:4]


def _plan_digest(project_id, proposal):
    """The request digest of the owner's exact approval of a reviewed proposal, built as the plan commit builds it."""
    from agent.redact import redact_sensitive_text

    texts = {name: redact_sensitive_text(owner_workspace._bounded_text(proposal[name], name, limit=limit), force=True)
             for name, limit in _TEXT_LIMITS.items()}
    changes, risk = owner_workspace._normalize_project_changes(proposal["changes"])
    return owner_workspace._digest({"project_id": project_id, "trigger": "owner_request", **texts,
                                    "later_milestones": owner_workspace._normalize_later_milestones(proposal["later_milestones"]),
                                    "risk_level": risk, "changes": changes})


def confirmed_plan_trigger(project_id, request_digest):
    """Keep owner_request unless its approved payload exactly matches an accepted daily proposal.

    The caller keeps its original approval payload and request digest. This read changes only audit provenance.
    An absent, changed or unreadable source cannot claim scheduled_review.
    """
    try:
        accepted = accepted_review_source(project_id)
        if accepted is not None and _plan_digest(project_id, accepted[2]) == request_digest:
            return "scheduled_review"
    except (OSError, sqlite3.Error, TypeError, ValueError, KeyError, owner_workspace.OwnerWorkspaceError):
        logger.debug("Daily review origin could not be confirmed")
    return "owner_request"


def confirmed_plan_links(project_id, request_digest):
    """The accepted review's pairs [(change index, suggestion id)] when the owner approved exactly its proposal.

    The same source and digest check as confirmed_plan_trigger; any other approval has no pair. A pair is only the
    review's proposal: the kernel checks it against the live store in the transaction that creates the card.
    """
    try:
        accepted = _accepted_review(project_id)
        if accepted is not None and _plan_digest(project_id, accepted[2]) == request_digest:
            return [(pair["change"], pair["suggestion"]) for pair in accepted[4].get("plan_suggestions") or []]
    except (OSError, sqlite3.Error, TypeError, ValueError, KeyError, owner_workspace.OwnerWorkspaceError):
        logger.debug("Daily review pairs could not be confirmed")
    return []


def _denied_plan(registry, project_id, proposal, suggestions=()):
    """Whether the owner denied this reviewed plan, or a reviewed plan for one of its accepted suggestions.

    A denied plan commit in the root registry is the owner's answer to its exact approval digest; a silence, an
    approval that timed out with no answer of the owner, denies nothing. The pairs of the stored accepted reviews tie
    each reviewed plan to its suggestions, so a reworded plan for a suggestion whose reviewed plan was denied is not
    proposed again while the brief history (its latest 90 briefs) holds that pair. ``suggestions`` are the plan's own
    pairs; the pairs a stored review gave this same plan count too. Read only: an absent table or an unreadable store
    denies nothing. The plan commit never reads this, so the owner can still ask for the work in the Project chat.
    """
    try:
        denied = {row["request_digest"] for row in registry.execute(
            "SELECT request_digest,status,result_json FROM owner_workspace_receipts "
            "WHERE operation='owner_project_plan_commit' AND status='denied'")
            if not owner_workspace._is_retryable_confirmation_timeout(row)}
        digest = _plan_digest(project_id, proposal)
        if not denied or digest in denied:
            return digest in denied
        reviewed = []
        with events._read_only(owner_project_summary._summary_board_path()) as board:
            for source in registry.execute(
                "SELECT review_date,source_task_id,source_run_id,source_digest FROM project_review_briefs "
                "WHERE project_id=? AND source_task_id IS NOT NULL", (project_id,),
            ).fetchall():
                run = kanban_db.get_run(board, source["source_run_id"])
                if run is None or not isinstance(run.metadata, dict) or not isinstance(
                        run.metadata.get("project_review"), dict) or not run.metadata["project_review"].get("plan_suggestions"):
                    continue
                checked = _checked_review(board, project_id, source)
                if checked is not None:
                    reviewed.append((_plan_digest(project_id, checked[1]["plan_proposal"]),
                                     {pair["suggestion"] for pair in checked[1]["plan_suggestions"]}))
        own = set(suggestions).union(*(paired for plan, paired in reviewed if plan == digest))
        return any(plan in denied and paired & own for plan, paired in reviewed)
    except (OSError, sqlite3.Error, TypeError, ValueError, KeyError, owner_workspace.OwnerWorkspaceError):
        logger.debug("Owner plan denials could not be read")
    return False
