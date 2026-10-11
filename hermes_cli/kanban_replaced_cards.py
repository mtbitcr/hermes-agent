"""The cards an approved card replaced, read from owner plan-change events."""

from __future__ import annotations

import json
import sqlite3


def replaced_task_ids(conn: sqlite3.Connection, task_id: str) -> list[str]:
    """Ids of the cards ``task_id`` replaced, directly or through a chain, sorted."""
    replaced_by: dict[str, set[str]] = {}
    for replaced, raw in conn.execute(
        "SELECT task_id, payload FROM task_events WHERE kind = ?",
        ("owner_project_plan_change",),
    ):
        try:
            payload = json.loads(raw) if raw else {}
        except (json.JSONDecodeError, TypeError):
            continue
        if not isinstance(payload, dict) or "replacement_task_ids" in payload:
            continue
        replacement = payload.get("replacement_task_id")
        if isinstance(replacement, str) and replacement:
            replaced_by.setdefault(replacement, set()).add(replaced)
    visited = {task_id}
    pending = [task_id]
    while pending:
        for replaced in replaced_by.get(pending.pop(), ()):
            if replaced not in visited:
                visited.add(replaced)
                pending.append(replaced)
    return sorted(visited - {task_id})
