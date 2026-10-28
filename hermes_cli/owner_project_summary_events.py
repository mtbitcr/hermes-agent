"""What changed in a Project since its summary was last asked for: the events that make a summary stale.

The summary tick calls ``read`` once for each live Project, after the one check for an open summary task and
before the decision whether a request is due. ``read`` looks at three sources, past the cursors of the Project's
``project_summary_state`` row:

* the ``task_events`` of the Project's own board,
* the merged rows of that board's ``kanban_deliveries``,
* the release ledger, for a released batch that holds a card of that board.

``EVENT_CLASS`` sorts each kind of event into material, urgent or ignored, and nothing else decides what an event
causes. A kind the table does not name is ignored. A material event only raises ``pending``; the tick's own rule
then batches it into one request an hour at most, counted from ``last_requested_at``. An urgent event also raises
``pending_urgent``, which lets the request out at once. While the Project has failures ``_due`` still waits for the
backoff, whatever the flags say.

``read`` writes nothing. The tick decides on the row with the flags raised (``Reading.state``). If no request is due
it stores the reading with ``record``; if one is, the request's claim moves the cursors in the same transaction that
clears the flags. A request that fails or stops before its claim therefore leaves the row as it was, and the next
tick finds the same events again.

The cursors move by compare-and-swap, in the shape of the kanban notifier's: a tick reads past the cursors it found,
then updates the row only if they are still those. Two ticks that read at the same time therefore count an event
once; the loser counts nothing and moves nothing.

Reading changes nothing and creates nothing: every file is opened read-only, a file or table that is not there is a
source without events, and no event of the summary board itself ever counts.

The delivery cursor is a row id, as the plan says. A delivery row that merges after a row with a higher id has
merged is therefore not seen as a merge; the card's own ``completed`` and ``archived`` events still are.
"""

from __future__ import annotations

import contextlib
import json
import logging
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, NamedTuple, Optional

from hermes_cli import kanban_db, projects_db, release_ledger
from hermes_cli.sqlite_util import write_txn

logger = logging.getLogger(__name__)

MATERIAL = "material"
URGENT = "urgent"
IGNORED = "ignored"

# A key is a kind of event, or "kind:detail" where the detail (the kind of a block) changes the answer.
# "merged" and "released" are the kinds the delivery and the ledger source report. Heartbeat, comment, claim,
# spawn and the like are left out on purpose: a kind that is not here is ignored.
EVENT_CLASS = {
    "created": MATERIAL,
    "linked": MATERIAL,
    "unlinked": MATERIAL,
    "decomposed": MATERIAL,
    "archived": MATERIAL,
    "completed": MATERIAL,
    "blocked": MATERIAL,
    "unblocked": MATERIAL,
    "blocked:needs_input": URGENT,
    "unblocked:needs_input": URGENT,
    "review_requested": URGENT,
    "merged": URGENT,
    "released": URGENT,
}

# kanban_delivery._MERGED. That module is large, and a tick that only reads one column does not import it.
_MERGED = "merged"
_RELEASED = "released"  # the outcome that a release_finished event of the ledger carries for a released batch


class Cursors(NamedTuple):
    event: int = 0
    delivery: int = 0
    release: int = 0


@dataclass(frozen=True)
class Batch:
    """What one read found past the cursors: how many material and urgent events, and where the cursors end up."""

    cursors: Cursors
    material: int = 0
    urgent: int = 0


def classify(kind: str, detail: Optional[str] = None) -> str:
    """The class of an event: the table's entry for ``kind:detail`` if it has one, else for ``kind``, else ignored."""
    if detail is not None and f"{kind}:{detail}" in EVENT_CLASS:
        return EVENT_CLASS[f"{kind}:{detail}"]
    return EVENT_CLASS.get(kind, IGNORED)


def _board_path(slug: str) -> Path:
    """The board's file by the root's own layout: ``kanban_db_path`` would obey a worker's ``HERMES_KANBAN_DB`` pin."""
    if slug == kanban_db.DEFAULT_BOARD:
        return kanban_db.kanban_home() / "kanban.db"
    return kanban_db.board_dir(slug) / "kanban.db"


@contextlib.contextmanager
def _read_only(path: Path) -> Iterator[sqlite3.Connection]:
    conn = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)
    try:
        conn.row_factory = sqlite3.Row
        yield conn
    finally:
        conn.close()


def _missing_schema(exc: sqlite3.OperationalError) -> bool:
    """An older board has no deliveries table or column, and a root that never released has no ledger tables."""
    return "no such table" in str(exc) or "no such column" in str(exc)


def _payload(text: Optional[str]) -> dict:
    try:
        payload = json.loads(text) if text else None
    except ValueError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _block_kind(board: sqlite3.Connection, event: sqlite3.Row) -> Optional[str]:
    """The kind of block that a ``blocked`` event starts, or that the ``unblocked`` event after it ends.

    The unblock carries no kind of its own, and the task's ``block_kind`` may already name a later block, so
    the answer comes from the latest ``blocked`` event of the task before it.
    """
    if event["kind"] == "blocked":
        return _payload(event["payload"]).get("kind")
    if event["kind"] != "unblocked":
        return None
    blocked = board.execute(
        "SELECT payload FROM task_events WHERE task_id = ? AND kind = 'blocked' AND id < ? ORDER BY id DESC LIMIT 1",
        (event["task_id"], event["id"]),
    ).fetchone()
    return None if blocked is None else _payload(blocked["payload"]).get("kind")


def _count_task_events(board: sqlite3.Connection, after: int, counts: dict) -> int:
    last = after
    for event in board.execute(
        "SELECT id, task_id, kind, payload FROM task_events WHERE id > ? ORDER BY id", (after,)
    ):
        last = event["id"]
        counts[classify(event["kind"], _block_kind(board, event))] += 1
    return last


def _count_merges(board: sqlite3.Connection, after: int, counts: dict) -> int:
    try:
        merged = [
            row["id"]
            for row in board.execute(
                "SELECT id FROM kanban_deliveries WHERE pull_request_state = ? AND id > ? ORDER BY id",
                (_MERGED, after),
            )
        ]
    except sqlite3.OperationalError as exc:
        if not _missing_schema(exc):
            raise
        return after
    counts[classify("merged")] += len(merged)
    return merged[-1] if merged else after


def _holds_card(ledger: sqlite3.Connection, board: sqlite3.Connection, slug: str, batch_id: int) -> bool:
    """Whether the batch holds a card of the board. A member that names no board can only be told by its card id."""
    for member in ledger.execute("SELECT card_id, board FROM release_members WHERE batch_id = ?", (batch_id,)):
        if member["board"] == slug:
            return True
        if member["board"] == "" and board.execute(
            "SELECT 1 FROM tasks WHERE id = ?", (member["card_id"],)
        ).fetchone():
            return True
    return False


def _count_releases(board: sqlite3.Connection, slug: str, after: int, counts: dict) -> int:
    path = release_ledger.ledger_path()
    if not path.exists():
        return after
    last = after
    try:
        with _read_only(path) as ledger:
            for event in ledger.execute(
                "SELECT id, batch_id, kind, payload FROM release_events WHERE id > ? ORDER BY id", (after,)
            ):
                last = event["id"]
                released = (
                    event["kind"] == "release_finished" and _payload(event["payload"]).get("outcome") == _RELEASED
                )
                if released and _holds_card(ledger, board, slug, event["batch_id"]):
                    counts[classify("released")] += 1
    except sqlite3.OperationalError as exc:
        if not _missing_schema(exc):
            raise
        return after
    return last


def read_events(project: projects_db.Project, cursors: Cursors) -> Batch:
    """Count what the three sources hold past ``cursors``. This reads and moves nothing.

    Every event moves its cursor on, a counted one or an ignored one, so none is read twice. A Project with no
    board, or whose board is the summary board, has no events.
    """
    # Imported here: owner_project_summary imports this module.
    from hermes_cli.owner_project_summary import SUMMARY_BOARD

    slug = project.board_slug
    if not slug or slug == SUMMARY_BOARD:
        return Batch(cursors)
    path = _board_path(slug)
    if not path.exists():
        return Batch(cursors)
    counts = {MATERIAL: 0, URGENT: 0, IGNORED: 0}
    with _read_only(path) as board:
        moved = Cursors(
            _count_task_events(board, cursors.event, counts),
            _count_merges(board, cursors.delivery, counts),
            _count_releases(board, slug, cursors.release, counts),
        )
    return Batch(moved, counts[MATERIAL], counts[URGENT])


def advance_cursors(conn: sqlite3.Connection, project_id: str, old: Cursors, batch: Batch) -> bool:
    """Move the Project's cursors from ``old`` to the batch's and raise its flags, if nobody moved them first.

    The flags only ever go up here; a request's claim is what clears them. The swap also needs the Project to have
    no open task: a claim made meanwhile has cleared the flags for events its snapshot may cover, and the events
    stay unread until the task is harvested. ``updated_at`` is left alone, since a failure's backoff runs from it.
    """
    with write_txn(conn):
        moved = conn.execute(
            "UPDATE project_summary_state SET event_cursor = ?, delivery_cursor = ?, release_cursor = ?, "
            "pending = MAX(pending, ?), pending_urgent = MAX(pending_urgent, ?) "
            "WHERE project_id = ? AND event_cursor = ? AND delivery_cursor = ? AND release_cursor = ? "
            "AND open_task_id IS NULL",
            (
                batch.cursors.event, batch.cursors.delivery, batch.cursors.release,
                int(bool(batch.material or batch.urgent)), int(bool(batch.urgent)),
                project_id, old.event, old.delivery, old.release,
            ),
        ).rowcount
    return moved == 1


@dataclass(frozen=True)
class Reading:
    """A Project's state row as the tick found it, and what the sources hold past that row's cursors."""

    row: sqlite3.Row
    batch: Batch

    @property
    def old(self) -> Cursors:
        return Cursors(self.row["event_cursor"], self.row["delivery_cursor"], self.row["release_cursor"])

    @property
    def raises_pending(self) -> bool:
        return bool(self.batch.material or self.batch.urgent)

    @property
    def state(self) -> dict:
        """The row as it will stand once the batch is counted: the flags raised, nothing stored."""
        state = dict(self.row)
        state["pending"] = max(state["pending"], int(self.raises_pending))
        state["pending_urgent"] = max(state["pending_urgent"], int(bool(self.batch.urgent)))
        return state


def read(registry: sqlite3.Connection, project: projects_db.Project) -> Reading:
    """Read the Project's new events past the cursors of its state row, which must exist. Nothing is written.

    A source that cannot be read counts as no events and moves no cursor, so the next tick reads it again.
    """
    row = registry.execute("SELECT * FROM project_summary_state WHERE project_id = ?", (project.id,)).fetchone()
    reading = Reading(row, Batch(Cursors(row["event_cursor"], row["delivery_cursor"], row["release_cursor"])))
    try:
        return Reading(row, read_events(project, reading.old))
    except (sqlite3.Error, OSError, ValueError) as exc:
        logger.warning("project summary: cannot read the events of Project %s (%s); none counted", project.id, exc)
        return reading


def record(registry: sqlite3.Connection, project_id: str, reading: Reading) -> bool:
    """Store a reading of a Project that is not asked for now: raise its flags and move its cursors.

    Returns False if there was nothing to store or another tick moved the cursors first.
    """
    if reading.batch == Batch(reading.old):
        return False
    return advance_cursors(registry, project_id, reading.old, reading.batch)
