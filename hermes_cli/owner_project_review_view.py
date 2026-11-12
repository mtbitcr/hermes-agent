"""Read the owner-safe brief and the accepted daily plan advice after the caller checks Project ownership."""

from __future__ import annotations

import contextlib
import sqlite3
import time

from hermes_cli import kanban_db, projects_db
from hermes_cli.owner_text import check_owner_text


def project_review_view(project_id: str):
    from hermes_cli.owner_workspace import OwnerWorkspaceError, _open_read_only_sqlite, owner_title

    # The review worker writes the brief into the root registry; a profile HERMES_HOME never redirects this read.
    path = kanban_db.kanban_home() / "projects.db"
    if not path.is_file():
        return None
    try:
        with contextlib.closing(_open_read_only_sqlite(path, label="projects.db")) as registry:
            stored = projects_db.get_project_review_brief(registry, project_id)
            if stored is None:
                return None
            # The plan limits the visible history to fourteen daily briefs.
            history = registry.execute(
                "SELECT review_date,headline FROM project_review_briefs WHERE project_id=? "
                "ORDER BY review_date DESC LIMIT 14", (project_id,),
            ).fetchall()
    except sqlite3.Error as exc:
        raise OwnerWorkspaceError("snapshot_unavailable", "the Project review could not be read") from exc
    latest = {name: stored[name] for name in (
        "review_date", "covered_from", "covered_to", "headline", "what_changed", "next_step", "needs_you",
        "ready_work", "cut_note", "quiet", "written_at", "revision",
    )}
    for name in ("headline", "what_changed", "next_step", "needs_you", "cut_note"):
        if latest[name] is not None:
            latest[name] = check_owner_text(latest[name], "planner_prose")
    latest["ready_work"] = [owner_title(title) for title in latest["ready_work"][:5]]
    return {"latest": latest, "history": [
        {"review_date": row["review_date"], "headline": check_owner_text(row["headline"], "planner_prose")}
        for row in history
    ]}


def _readable(proposal: dict) -> bool:
    """Whether each model-written text passes the owner-text policy as it stands."""
    texts = [(proposal[name], "planner_prose") for name in (
        "request_title", "summary", "specification", "current_milestone", "owner_visible_result",
    )]
    texts += [(item, "planner_prose") for item in proposal["later_milestones"]]
    for change in proposal["changes"]:
        specs = [change, *change.get("replacements", []), *([change["replacement"]] if "replacement" in change else [])]
        texts += [(change.get("reason"), "planner_prose")]
        texts += [(spec["title"], "title") for spec in specs if "title" in spec]
        texts += [(spec["body"], "planner_prose") for spec in specs if "body" in spec]
    return all(check_owner_text(text, kind) == text for text, kind in texts if text is not None)


def project_review_advice_view(project_id: str):
    """Read accepted advice only while its source and live task references remain valid."""
    from hermes_cli import owner_project_review as review, owner_project_review_record as records
    from hermes_cli.owner_project_review_origin import accepted_review_source
    from hermes_cli.owner_project_review_proposals import proposal_hint
    from hermes_cli.owner_workspace import OwnerWorkspaceError

    try:
        accepted = accepted_review_source(project_id)
        if accepted is None:
            return None
        _project, task, proposal, day = accepted
        sources, _cut = review._frozen_sources(task.body)
        # The registry and Project board the review reads: the root's, also under a worker's pins.
        with review._review_evidence_board(project_id) as (board, registry, project):
            hint = proposal_hint(proposal, sources, board, registry, project, now=int(time.time()),
                                 limit=review._BRIEF_LIMITS["next_step"], max_bytes=records.TOTAL_INPUT_CAP_BYTES)
        if hint is None or not _readable(proposal):
            return None
        # Return only the checked plan fields. Source bodies, Run metadata and private handles stay internal.
        return {"review_date": day, "proposal": records._redacted(proposal)}
    except (OSError, sqlite3.Error, ValueError, TypeError, KeyError, OwnerWorkspaceError):
        return None
