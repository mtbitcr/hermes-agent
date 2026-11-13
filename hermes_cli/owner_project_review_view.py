"""Read the owner-safe brief and the accepted daily plan advice after the caller checks Project ownership."""

from __future__ import annotations

import contextlib
import sqlite3
import time
from datetime import date, timedelta

from hermes_cli import kanban_db, projects_db
from hermes_cli.owner_text import check_owner_text

# The fixed explanation of a daily cap delay (owner rule of 2026-11-12, point 5).
_CAP_HEADLINE = "The review waits for the next day: the daily limit is reached."
_CAP_WHAT_CHANGED = (
    "The daily limit of reviews is reached, so this Project's review waits for the next day. "
    "Its news is kept for that review."
)
_CAP_NEXT_STEP = "The review runs on the next day."


def _cap_wait(registry: sqlite3.Connection, project_id: str):
    """The Project's cap mark with its last reviewed day, or None. A store from before the mark has no mark."""
    columns = {row[1] for row in registry.execute("PRAGMA table_info(project_review_state)")}
    if not {"cap_wait_date", "cap_wait_at"} <= columns:
        return None
    row = registry.execute(
        "SELECT cap_wait_date, cap_wait_at, last_reviewed_date FROM project_review_state WHERE project_id=?",
        (project_id,),
    ).fetchone()
    return None if row is None or row["cap_wait_date"] is None else row


def _cap_explanation(mark) -> dict:
    """The fixed explanation in the shape of a stored brief, for the day of the mark."""
    day, reviewed = mark["cap_wait_date"], mark["last_reviewed_date"]
    covered_from = None if reviewed is None else (date.fromisoformat(reviewed) + timedelta(days=1)).isoformat()
    return {
        "review_date": day, "covered_from": covered_from, "covered_to": day, "headline": _CAP_HEADLINE,
        "what_changed": _CAP_WHAT_CHANGED, "next_step": _CAP_NEXT_STEP, "needs_you": None, "ready_work": [],
        "cut_note": None, "quiet": False, "written_at": mark["cap_wait_at"], "revision": 1,
    }


def project_review_view(project_id: str):
    from hermes_cli.owner_workspace import OwnerWorkspaceError, _open_read_only_sqlite, owner_title

    # The review worker writes the brief into the root registry; a profile HERMES_HOME never redirects this read.
    path = kanban_db.kanban_home() / "projects.db"
    if not path.is_file():
        return None
    try:
        with contextlib.closing(_open_read_only_sqlite(path, label="projects.db")) as registry:
            stored = projects_db.get_project_review_brief(registry, project_id)
            mark = _cap_wait(registry, project_id)
            if stored is None and mark is None:
                return None
            # The plan limits the visible history to fourteen daily briefs.
            history = [] if stored is None else registry.execute(
                "SELECT review_date,headline FROM project_review_briefs WHERE project_id=? "
                "ORDER BY review_date DESC LIMIT 14", (project_id,),
            ).fetchall()
    except sqlite3.Error as exc:
        raise OwnerWorkspaceError("snapshot_unavailable", "the Project review could not be read") from exc
    # The cap mark is shown only while no stored brief has its day or a later one; the history stays the stored one.
    if mark is not None and (stored is None or stored["review_date"] < mark["cap_wait_date"]):
        stored = _cap_explanation(mark)
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
