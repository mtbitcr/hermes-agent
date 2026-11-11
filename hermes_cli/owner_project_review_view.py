"""Read the owner-safe brief after the caller checks Project ownership."""

from __future__ import annotations

import contextlib
import sqlite3

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
