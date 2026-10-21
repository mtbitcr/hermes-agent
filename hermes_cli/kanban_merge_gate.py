"""Whether a parent card satisfies a child card: card D, owner decision of 2026-10-19.

A parent satisfies a child once it is done or archived. With delivery on, a child
whose owned paths overlap the parent's also waits until the parent's change has
merged: the parent card, or a card that carries its change on, has a delivery row
in state merged. A card's delivery_updated or review_followup_recorded event names
the card that carries its change on; that chain is followed for at most eight links,
and a card on it is never held by its own parent. A parent with no delivery row
satisfies the child at approval, as before, and a change that can never merge holds
the child: there is no timeout and no release. The four parent gates of
hermes_cli/kanban_db.py ask this one predicate, so promotion and claim never
disagree. The gate writes nothing.
"""

from __future__ import annotations

import json
import sqlite3

_LINKS = 8  # as far as kanban_delivery_update._approved_base follows its chain


def _carried_on_by(conn: sqlite3.Connection, card: str) -> list[str]:
    """The cards that ``card``'s own delivery_updated and review_followup_recorded events name."""
    named = []
    for (payload,) in conn.execute(
        "SELECT payload FROM task_events WHERE task_id = ? "
        "AND kind IN ('delivery_updated', 'review_followup_recorded') ORDER BY id",
        (card,),
    ):
        try:
            record = json.loads(payload)
        except (TypeError, ValueError):
            continue
        if isinstance(record, dict):
            carrier = record.get("update_task_id") or record.get("followup_task_id")
            if isinstance(carrier, str):
                named.append(carrier)
    return named


def _merged(conn: sqlite3.Connection, cards: list[str]) -> bool:
    """Whether one of ``cards`` has a delivery row in state merged."""
    return conn.execute(
        "SELECT 1 FROM kanban_deliveries WHERE pull_request_state = 'merged' "
        f"AND source_task_id IN ({', '.join('?' * len(cards))}) LIMIT 1",
        cards,
    ).fetchone() is not None


def parent_satisfies(conn: sqlite3.Connection, parent: sqlite3.Row, child_id: str) -> bool:
    """Whether ``parent``, a tasks row with id, status and owned_paths, satisfies ``child_id``."""
    if parent["status"] not in ("done", "archived"):
        return False
    if conn.execute(
        "SELECT 1 FROM kanban_deliveries WHERE source_task_id = ? LIMIT 1", (parent["id"],),
    ).fetchone() is None:
        return True  # nothing of it is delivered: satisfied at approval, as before
    # Late import: the facade imports this module at load time.
    from hermes_cli.kanban_db import _decode_owned_paths, _owned_path_scopes_overlap

    child = conn.execute("SELECT owned_paths FROM tasks WHERE id = ?", (child_id,)).fetchone()
    if not _owned_path_scopes_overlap(
        _decode_owned_paths(parent["owned_paths"]),
        _decode_owned_paths(child["owned_paths"]) if child is not None else None,
    ):
        return True
    cards, seen = [parent["id"]], {parent["id"]}
    for link in range(_LINKS + 1):
        if _merged(conn, cards):
            return True
        if link == _LINKS:
            break
        cards = list(dict.fromkeys(
            card for source in cards for card in _carried_on_by(conn, source) if card not in seen
        ))
        if child_id in cards:
            return True  # it carries the parent's change on, so the merge waits for it
        if not cards:
            break
        seen.update(cards)
    # Only here can the switch change the answer: shared paths, and an unmerged change.
    from hermes_cli.kanban_delivery import delivery_settings

    return not delivery_settings()
