"""Close the rework items a card recorded when the card is archived."""

from __future__ import annotations

import sqlite3


def archive_recorded_review_followups(conn: sqlite3.Connection, task_id: str) -> None:
    """Archive the open items ``task_id`` itself recorded, in the caller's transaction."""
    # Late import: the facade imports this module at load time.
    from hermes_cli.kanban_db import _append_event, _end_run, fence_cas_conjunct

    rows = conn.execute(
        "SELECT json_extract(payload, '$.followup_task_id') AS followup_id "
        "FROM task_events "
        "WHERE task_id = ? AND kind = 'review_followup_recorded' "
        "AND CASE WHEN json_valid(payload) "
        "THEN json_type(payload, '$.followup_task_id') END = 'text' "
        "ORDER BY id",
        (task_id,),
    ).fetchall()
    followup_ids = list(
        dict.fromkeys(row["followup_id"] for row in rows if row["followup_id"])
    )
    # Each item leaves the board like a card does: same in-predicate removal gate.
    gate_sql, gate_params = fence_cas_conjunct(conn)
    summary = (
        f"Item closed because its card was archived ({task_id}): the rework "
        "it tracked will not happen."
    )
    for followup_id in followup_ids:
        cur = conn.execute(
            "UPDATE tasks SET status = 'archived', result = ?, "
            "claim_lock = NULL, claim_expires = NULL, worker_pid = NULL "
            "WHERE id = ? AND task_kind = 'work' "
            "AND status NOT IN ('done', 'archived')"
            f"{gate_sql}",
            (summary, followup_id, *gate_params),
        )
        if cur.rowcount == 1:
            _end_run(
                conn, followup_id,
                outcome="reclaimed", status="reclaimed",
                summary=(
                    f"item closed because its card {task_id} was archived "
                    "with this run still active"
                ),
            )
            _append_event(
                conn, task_id, "review_followup_archived",
                {"followup_task_id": followup_id},
            )
