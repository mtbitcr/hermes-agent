"""Where a finished change was delivered, for the owner's Project snapshot.

A card's latest ``kanban_deliveries`` row (hermes_cli/kanban_delivery.py) says what happened to the change it
carries: ``merged`` when the row merged; ``returned`` when a red required check returned the work for the row's own
head (the source card's ``red_check_returned`` record) and it did not merge; ``open`` while
``unfinished_delivery_task_ids`` counts the row as unfinished; ``closed`` for any other row with a pull request.
The repository is only the publish step's own record of that pull request (the rule of ``_RECEIPT``), never the git
origin: a row with no pull request or no such record has no outcome. It reads one board and never writes.
"""

from __future__ import annotations

import json
import sqlite3

from hermes_cli import kanban_delivery as kd

# The states unfinished_delivery_task_ids counts as unfinished for a row with a pull request.
_UNFINISHED = ("open", "review_cards_created", kd._PENDING, kd._ARMED, kd._ARM_UNKNOWN, kd._ARM_REFUSED,
               kd._BASE_MOVED, kd._CLOSE_REFUSED)
# The record _return_red_check makes on the source card when a red required check returns the row's head.
_RETURNED = ("EXISTS (SELECT 1 FROM task_events WHERE task_id = d.source_task_id AND kind = 'delivery_review_waiting' "
             "AND json_extract(payload, '$.code') = 'red_check_returned' "
             "AND json_extract(payload, '$.head') = d.pull_request_head)")


def delivery_outcomes(conn: sqlite3.Connection, task_ids) -> dict:
    """Each of ``task_ids`` to the outcome of its latest delivery row, ``{"state", "repository", "pull_request",
    "url"}``, or None for a card with no row or whose latest row has no pull request or no published record. It
    only reads: a board with no ledger gives every card None and gets no schema; any other database error is
    raised."""
    ids = [str(task_id) for task_id in task_ids]
    outcomes = dict.fromkeys(ids)
    if not ids or conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'kanban_deliveries'").fetchone() is None:
        return outcomes
    for task_id, number, state, repository, returned in conn.execute(
            f"SELECT d.source_task_id, d.pull_request_number, d.pull_request_state, {kd._RECEIPT.format('d')}, "
            f"{_RETURNED} FROM kanban_deliveries AS d WHERE d.id IN (SELECT MAX(id) FROM kanban_deliveries "
            "WHERE source_task_id IN (SELECT value FROM json_each(?)) GROUP BY source_task_id)", (json.dumps(ids),)):
        if number is None or repository is None:
            continue
        outcomes[str(task_id)] = {
            "state": ("merged" if state == kd._MERGED else "returned" if returned
                      else "open" if state in _UNFINISHED else "closed"),
            "repository": repository,
            "pull_request": number,
            "url": f"https://github.com/{repository}/pull/{number}",
        }
    return outcomes
