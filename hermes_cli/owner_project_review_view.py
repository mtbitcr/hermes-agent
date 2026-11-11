"""Read the accepted daily plan advice of an owned Project for its snapshot."""

from __future__ import annotations

import sqlite3
import time

from hermes_cli.owner_text import check_owner_text


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
