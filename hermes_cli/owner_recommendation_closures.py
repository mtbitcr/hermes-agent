"""Project checked closures through the existing owner authority boundary."""

from __future__ import annotations

import contextlib

from hermes_cli import kanban_db
from hermes_cli.owner_text import check_owner_text


def _has_closures(board):
    return board.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='recommendation_closures'"
    ).fetchone() is not None


def live_closure_filter(board, enabled):
    if not enabled or not _has_closures(board):
        return ""
    # Exclude before the pending window, so closed rows cannot crowd out work that still needs the owner.
    return " AND NOT EXISTS (SELECT 1 FROM recommendation_closures c WHERE c.recommendation_id=tasks.id AND c.undone_at IS NULL)"


def recent_closed_suggestions(board, ctx, project, current):
    from hermes_cli.owner_workspace import _owner_decision_ref, owner_project_name, owner_title

    if not _has_closures(board):
        return current
    # The delivery card is joined loosely: a live closure hides its suggestion from the pending list, so the
    # closure stays listed, and undoable, after its card is deleted.
    rows = board.execute(
        "SELECT r.id,r.recommendation_label,w.title AS card_title,c.reason,c.closed_at "
        "FROM recommendation_closures c JOIN tasks r ON r.id=c.recommendation_id "
        "LEFT JOIN tasks w ON w.id=c.card_id AND w.project_id=c.project_id AND w.task_kind='work' "
        "WHERE c.project_id=? AND r.project_id=c.project_id "
        "AND r.task_kind='recommendation' AND r.status='review' "
        "AND r.review_policy='owner' AND COALESCE(r.recommendation_decision,'pending')='pending' "
        "AND c.undone_at IS NULL ORDER BY c.closed_at DESC,r.id ASC LIMIT 20", (project["project_id"],),
    ).fetchall()
    added = [{
        "decision_ref": _owner_decision_ref(ctx, authority="recommendation", native_id=str(row["id"])),
        "project_slug": project["slug"], "project_name": owner_project_name(project["name"]),
        "title": owner_title(row["recommendation_label"]), "card_title": owner_title(row["card_title"]),
        "reason": check_owner_text(row["reason"], "planner_prose"), "closed_at": int(row["closed_at"]),
    } for row in rows]
    return sorted([*current, *added], key=lambda item: (-item["closed_at"], item["decision_ref"]))[:20]


def undo_closed_suggestion(ctx, *, decision_ref, reason, expected_closed_at):
    from hermes_cli import owner_workspace as ow

    try:
        kanban_db._assert_recommendation_lifecycle_operator()
    except PermissionError as exc:
        raise ow.OwnerWorkspaceError("permission_denied", "the owner operation is unavailable") from exc
    if isinstance(reason, str):
        reason = reason.strip()
    if (isinstance(expected_closed_at, bool) or not isinstance(expected_closed_at, int) or expected_closed_at < 0
            or not isinstance(reason, str) or not reason or len(reason) > 500
            or check_owner_text(reason, "planner_prose") != reason):
        raise ow.OwnerWorkspaceError("invalid_argument", "a closure Undo needs a valid reason and token")
    found = ow._owner_pending_suggestion(ctx, decision_ref)
    if found is None:
        raise ow.OwnerWorkspaceError("decision_not_found", "this suggestion is no longer closed")
    board_slug, row = found
    # The Project's own store by its path: a pinned HERMES_KANBAN_DB never redirects the owner's Undo.
    path = ow._listed_board_path(board_slug)
    # Read before opening a write connection so a stale request cannot migrate an older board.
    # The kernel checks the same token inside its write fence.
    with contextlib.closing(ow._open_read_only_sqlite(path, label="kanban.db")) as board:
        live = _has_closures(board) and board.execute(
            "SELECT 1 FROM recommendation_closures WHERE recommendation_id=? AND closed_at=? AND undone_at IS NULL",
            (row["id"], expected_closed_at),
        ).fetchone() is not None
    if not live:
        raise ow.OwnerWorkspaceError("decision_not_found", "this suggestion is no longer closed")
    with kanban_db.connect_closing(path, board=board_slug) as board:
        result = kanban_db.undo_recommendation_closure(
            board, str(row["id"]), actor=ctx.actor, reason=reason, expected_closed_at=expected_closed_at,
        )
    if result["status"] != "undone":
        raise ow.OwnerWorkspaceError("decision_not_found", "this suggestion is no longer closed")
    return {"decision_ref": decision_ref, "undone": True}
