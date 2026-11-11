"""Which cards are attempts of the same work, for the owner's Project views.

A later card continues an earlier one through a record on the earlier card: its review_followup_recorded event
names the rework or update card kanban_delivery's _continuation made, its delivery_updated event its update card
on main, its owner_project_plan_change event the card the owner replaced it with. A continuation is the same work
again, so every continuation of a card stays in its item. The owner's split (its parts in replacement_task_ids) or
merge (action merge) changed the shape of the work, so each card it made starts its own item, as does a card that
continues more than one card. request_changes records its rework item with _continuation's event kind, in the run
of the reviewed card's own changes_requested verdict, which that run's outcome names after the event cleanup has
removed the verdict's event: that item is separate work, and the reviewed card goes back to the queue. Retries are
runs of one card and need no edge.

An item's root is the card its edges lead back to; in a cycle, the cycle's oldest card. A walk the 32-link guard
stops goes on to that root, so a lineage of any length has one. The key hashes the root's project and id, so it
holds for the life of the item and names no card: the owner's runs carry no card id. The current attempt is the
item's newest active card, else its newest done card, else its newest card; newest by created time, then id. It
reads one board through the caller's connection and never writes.

The owner defines the life of an item. No automatic action ends it: the event cleanup, a retry, a review return
and an update card keep every card's key, and so does an owner replacement. A purge ends it:
hermes kanban archive --rm is the owner's deliberate act that deletes an archived card for good. A deleted card's
events go with it, its edges to later cards among them, and an edge to a card no longer on the board is left out,
so when the owner purges an archived attempt, the life of its item ends for the cards that survive, and each of
them may start a new item with a new key. After a purge the reading stays consistent: each card belongs to exactly
one item, each item has exactly one current card, and no read fails.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3

_LINKS = 32
_ACTIVE = ("running", "review", "blocked", "ready", "todo", "scheduled", "triage")
# The current attempt is the newest card of the item's highest rank.
_RANK = {**dict.fromkeys(_ACTIVE, 2), "done": 1}
# (later card, earlier card) for each edge on the board.
_EDGES = (
    "SELECT json_extract(e.payload, '$.followup_task_id'), e.task_id FROM task_events AS e "
    "WHERE e.kind = 'review_followup_recorded' "
    "AND CASE WHEN json_valid(e.payload) THEN json_type(e.payload, '$.followup_task_id') END = 'text' "
    "AND NOT EXISTS (SELECT 1 FROM task_runs AS r WHERE r.id = e.run_id AND r.task_id = e.task_id "
    "AND r.outcome = 'changes_requested') "
    "UNION SELECT json_extract(payload, '$.update_task_id'), task_id FROM task_events "
    "WHERE kind = 'delivery_updated' "
    "AND CASE WHEN json_valid(payload) THEN json_type(payload, '$.update_task_id') END = 'text' "
    "UNION SELECT json_extract(payload, '$.replacement_task_id'), task_id FROM task_events "
    "WHERE kind = 'owner_project_plan_change' "
    "AND CASE WHEN json_valid(payload) THEN json_type(payload, '$.replacement_task_id') = 'text' "
    "AND json_type(payload, '$.replacement_task_ids') IS NULL AND json_extract(payload, '$.action') IS NOT 'merge' END"
)


def work_item_keys(conn: sqlite3.Connection, task_ids) -> dict:
    """Each of ``task_ids`` to its work item, ``{"key", "current"}``: the item's key, and whether the card is the
    item's current attempt. None for an id not on the board."""
    ids = [str(task_id) for task_id in task_ids]
    found = dict.fromkeys(ids)
    for card, (key, current, _) in _attempts(conn, ids).items():
        found[card] = {"key": key, "current": current}
    return found


def work_items(conn: sqlite3.Connection, task_ids) -> dict:
    """The cards of ``task_ids`` on the board by work item key, as an item is shown: its current attempt first,
    then its earlier attempts oldest first, each ``{"task_id", "current"}``."""
    attempts = _attempts(conn, [str(task_id) for task_id in task_ids])
    items: dict[str, list[dict]] = {}
    for card in sorted(attempts, key=lambda card: (not attempts[card][1], attempts[card][2], card)):
        key, current, _ = attempts[card]
        items.setdefault(key, []).append({"task_id": card, "current": current})
    return items


def _attempts(conn: sqlite3.Connection, ids: list[str]) -> dict:
    """Each of ``ids`` on the board to its item's key, whether it is the current attempt, and its created time.
    The current attempt is chosen among every card of the item, given or not."""
    if not ids:
        return {}
    earlier: dict[str, set[str]] = {}
    later: dict[str, set[str]] = {}
    for successor, predecessor in conn.execute(_EDGES):
        earlier.setdefault(successor, set()).add(predecessor)
        later.setdefault(predecessor, set()).add(successor)
    linked, pending = set(ids), list(ids)
    while pending:
        card = pending.pop()
        for other in earlier.get(card, set()) | later.get(card, set()):
            if other not in linked:
                linked.add(other)
                pending.append(other)
    cards = {card: (status, created, project) for card, status, created, project in conn.execute(
        "SELECT id, status, created_at, project_id FROM tasks WHERE id IN (SELECT value FROM json_each(?))",
        (json.dumps(sorted(linked)),))}
    continued = {}
    for card in cards:
        before = earlier.get(card, set()) & cards.keys()
        if len(before) == 1:
            (continued[card],) = before
    roots = {card: _root(card, continued, cards) for card in cards}
    members: dict[str, list[str]] = {}
    for card, root in roots.items():
        members.setdefault(root, []).append(card)
    current = {root: max(group, key=lambda card: (_RANK.get(cards[card][0], 0), cards[card][1], card))
               for root, group in members.items()}
    keys = {root: "wi_" + hashlib.sha256(f"{cards[root][2] or ''}:{root}".encode("utf-8")).hexdigest()[:16]
            for root in members}
    return {card: (keys[roots[card]], current[roots[card]] == card, cards[card][1]) for card in ids if card in cards}


def _root(card: str, continued: dict, cards: dict) -> str:
    """The card the edges from ``card`` lead back to: the first with no predecessor, or the oldest card of the cycle
    they enter. A walk the 32-link guard stops goes on from where it stopped, with the cards it has passed, to that
    same root; no walk has more links than the board has cards."""
    path = [card]
    for links in (_LINKS, len(cards)):
        while len(path) <= links and path[-1] in continued:
            before = continued[path[-1]]
            if before in path:
                return min(path[path.index(before):], key=lambda each: (cards[each][1], each))
            path.append(before)
    return path[-1]
