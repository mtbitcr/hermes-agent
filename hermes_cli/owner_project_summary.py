"""The Project summary tick: Raphael's planning role writes the first summary of each Project.

The gateway's dispatcher runs ``summary_tick`` once per tick. It harvests the summary tasks that have
finished, then opens one summary task on the ``project-summaries`` board for each live Project that needs
one, at most ``MAX_NEW_PER_TICK`` per tick. A result is stored only after ``parse_summary_result`` accepts
it, and storing one resets the Project's failure count to zero. Any other end of a step stores nothing and backs
off: an hour after the first failure, doubling after each, and a whole day (``MAX_BACKOFF_SECONDS``) from the
``FAILURE_STOP``-th failure in a row on, however long the steps keep failing (owner rule 3 of 2026-10-27).

One check decides whether a Project already has a summary task (owner rule 1 of 2026-10-27). It is the first
thing in each Project's turn, before any route, snapshot or policy read: ``_find_open_task`` looks on the
summary board for a task of the Project, found by its idempotency key, that is neither done nor archived. A
task it finds is recorded as the Project's open task and the tick goes on to the next Project. Only when none
exists may the tick read the route and the snapshot and publish. The board is the proof that a task was
published, not the state row: a crash between ``create_task`` and the record, followed by a failed read, used
to change the key of the request and so publish a second task. A failure before a task is published therefore
records the failure and nothing else.

A failed summary task is archived first; only then does one transaction clear it and count the failure (owner rule 2).

Card B refreshes a summary after changes. Once per Project, after that check and the eligibility test and before the
decision whether a request is due, ``owner_project_summary_events.read`` looks at the Project's new events. A material
event waits for the hour that ``_due`` counts from ``last_requested_at``, so ten quick changes are one request; a
decision asked or answered, a merge and a release do not wait. Every request, first or later, goes through the same
rules above: the kernel's time box, the one open task, and the backoff after a failure. A Project that is archived, or
whose removal has started or completed, is skipped before any event is read, and no failure is counted. The reading is
stored by ``record`` when no request is due, and by the request's own claim when one is, so a request that fails or
stops before its claim leaves the row as it was.

A task is the tick's own only when the tick made it for that Project (``_made_for``): its creator is ``CREATED_BY``
and its idempotency key names the Project. A key alone vouches for nothing, since any worker can write one with
``kanban_create``. A task made by anyone else is never adopted, recorded, archived or stored. When ``create_task``
hands one back for the request key, the step fails before a task is published, and the failure rules apply as they are.

The task's time box is the kernel's pin for its class; this module names no limit.

Every store is reached by an explicit path, so the tick still finds the root registry and every board
under the pins the dispatcher gives a worker (``HERMES_KANBAN_DB``, ``HERMES_KANBAN_BOARD``, a profile
``HERMES_HOME``).
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, Optional

from hermes_cli import kanban_db, projects_db
from hermes_cli import owner_project_summary_events as events
from hermes_cli.owner_text import check_owner_text
from hermes_cli.sqlite_util import write_txn

logger = logging.getLogger(__name__)

HOUR = 3600
MAX_NEW_PER_TICK = 3
FAILURE_STOP = 3  # the failure limit: from this many failures in a row on, every wait is MAX_BACKOFF_SECONDS
MAX_BACKOFF_SECONDS = 86400

SUMMARY_BOARD = "project-summaries"
PLANNING_ROLE = "raphael-planner"
CREATED_BY = "project-summary"
TITLE_PREFIX = "Write the Project summary: "
EXECUTION_TIER = "routine"
RISK_TIER = 0
RESULT_KEY = "project_summary"

# The text limits of the typed result; ``owner_decision`` is the only key that may be left out.
_TEXT_LIMITS = {"outcome": 400, "next_action": 240, "owner_decision": 240}
_REQUIRED_KEYS = frozenset({"outcome", "next_action"})
_STORE_PINS = ("HERMES_KANBAN_DB", "HERMES_KANBAN_BOARD")
_SETTLED = ("done", "blocked", "archived")
_REMOVAL_ENDED = ("cancelled", "failed", "restored")  # removal phases after which the Project is live again
_refused_boards: set[tuple[str, str]] = set()  # collisions already logged by this process

INSTRUCTION = """\
Write the short summary of one Project for its owner. The steward snapshot at the end of this task is everything you \
need. Use no tool except kanban_show and kanban_complete.

1. Read this task with kanban_show.
2. Write three texts in plain words:
   - outcome: where the Project stands now (1 to 400 characters).
   - next_action: what happens next (1 to 240 characters).
   - owner_decision: a decision only the owner can make, if one really waits (at most 240 characters). Use null when \
none waits.
3. Finish with kanban_complete. Give a one-sentence summary, and pass the texts as metadata in exactly this shape:
   metadata={"project_summary": {"outcome": "...", "next_action": "...", "owner_decision": null}}

Those three keys are the only keys allowed inside project_summary. Write about the work, not about yourself. Name no \
model or provider, and put no e-mail address, phone number, chat id, link or file path in any text. If one text breaks \
a rule, the whole result is thrown away and nothing is saved.
"""


class SummaryRefused(Exception):
    """A finished summary task whose result may not be stored."""


@dataclass(frozen=True)
class SummaryResult:
    outcome: str
    next_action: str
    owner_decision: Optional[str]


def backoff_seconds(failures: int) -> int:
    """The wait after ``failures`` failures in a row: an hour, doubling, and a whole day from ``FAILURE_STOP`` on."""
    if failures >= FAILURE_STOP:
        return MAX_BACKOFF_SECONDS
    return min(HOUR * 2 ** (max(failures, 1) - 1), MAX_BACKOFF_SECONDS)


def parse_summary_result(conn: sqlite3.Connection, task_id: str) -> SummaryResult:
    """The typed result of the finished summary task ``task_id``, or :class:`SummaryRefused`.

    Nothing the worker wrote reaches the registry except through here, so a result is accepted only when
    every rule holds; the texts are stored trimmed, exactly as the policy checked them.
    """
    task = kanban_db.get_task(conn, task_id)
    if task is None:
        raise SummaryRefused("the task is gone")
    if task.created_by != CREATED_BY:
        raise SummaryRefused("the task was not made by the summary tick")
    if task.assignee != PLANNING_ROLE:
        raise SummaryRefused("the task is not assigned to the planning role")
    if task.status != "done":
        raise SummaryRefused(f"the task is {task.status}, not done")
    run = kanban_db.latest_run(conn, task_id)
    if run is None or run.outcome != "completed":
        raise SummaryRefused("the latest run did not complete")

    metadata = run.metadata
    payload = metadata.get(RESULT_KEY) if isinstance(metadata, dict) else None
    if not isinstance(payload, dict):
        raise SummaryRefused(f"the run metadata has no {RESULT_KEY} object")
    if not _REQUIRED_KEYS <= payload.keys() <= _TEXT_LIMITS.keys():
        raise SummaryRefused(f"{RESULT_KEY} does not have exactly the allowed keys")

    texts: dict[str, Optional[str]] = {}
    for key, limit in _TEXT_LIMITS.items():
        text = payload.get(key)
        if key == "owner_decision" and (text is None or (isinstance(text, str) and not text.strip())):
            texts[key] = None
            continue
        if not isinstance(text, str):
            raise SummaryRefused(f"{key} is not text")
        text = text.strip()
        if not 1 <= len(text) <= limit:
            raise SummaryRefused(f"{key} is not 1 to {limit} characters")
        if check_owner_text(text, "planner_prose") != text:
            raise SummaryRefused(f"{key} does not pass the planner-prose policy")
        texts[key] = text
    return SummaryResult(**texts)


@contextlib.contextmanager
def _unpinned_stores() -> Iterator[None]:
    """Drop the worker's store pins while a board is created, so it is made where its own name says."""
    saved = {name: os.environ.pop(name) for name in _STORE_PINS if name in os.environ}
    try:
        yield
    finally:
        os.environ.update(saved)


def _summary_board_path() -> Path:
    return kanban_db.board_dir(SUMMARY_BOARD) / "kanban.db"


def _refuse_board(path: Path, meta: dict) -> None:
    owner = meta.get("project_id") or "unreadable board.json"
    if (str(path), owner) in _refused_boards:
        return
    _refused_boards.add((str(path), owner))
    logger.warning(
        "project summary: the board %r belongs to a Project (%s) or cannot be read; "
        "no summary task is made until the owner resolves it",
        SUMMARY_BOARD, owner,
    )


def _ensure_summary_board() -> Optional[Path]:
    """The summary board's path, made and switched on when need be; None while a Project owns the slug.

    A board that was made but never switched on is switched on here. The owner lifecycle always sets
    ``dispatch_paused_by_owner`` when it stops a board, so a pause or an archive is never taken for that.
    """
    path = _summary_board_path()
    with _unpinned_stores():
        meta = None
        if kanban_db.board_exists(SUMMARY_BOARD):
            meta = kanban_db.read_board_metadata(SUMMARY_BOARD)
            if meta.get("project_id") or not kanban_db.board_ownership_verified(meta):
                _refuse_board(path, meta)
                return None
        if meta is None or not path.exists():
            meta = kanban_db.create_board(
                SUMMARY_BOARD,
                name="Project summaries",
                description="Summary tasks for the planning role. Linked to no Project.",
            )
        if (
            meta.get("dispatch_enabled") is not True
            and meta.get("dispatch_paused_by_owner") is not True
            and meta.get("archived") is not True
        ):
            kanban_db.write_board_dispatch_state(SUMMARY_BOARD, dispatch_enabled=True)
    return path


def _task_body(snapshot: dict) -> str:
    frozen = json.dumps(snapshot, indent=2, sort_keys=True, ensure_ascii=False)
    return INSTRUCTION + "\nSteward snapshot, frozen when this task was made:\n" + frozen + "\n"


def _key_prefix(project_id: str) -> str:
    """The front of every request key of the Project: the creator and the Project id."""
    return f"{CREATED_BY}:{project_id}:"


def _request_key(state: Mapping[str, Any]) -> str:
    """The identity of one request, from the state row alone, so two writers of the same row make one task.

    ``_find_open_task`` and ``_made_for`` rely only on the front of it, the creator and the Project id.
    """
    return f"{_key_prefix(state['project_id'])}{state['failures']}:{state['last_requested_at']}"


def _made_for(task: kanban_db.Task, project_id: str) -> bool:
    """Whether the tick made the task for this Project: the creator is the tick, the key names the Project."""
    return task.created_by == CREATED_BY and (task.idempotency_key or "").startswith(_key_prefix(project_id))


def _find_open_task(board: sqlite3.Connection, project_id: str) -> Optional[str]:
    """The oldest summary task of the Project that the tick made and that is neither done nor archived, or None.

    A task made by anyone else is not found, whatever its key. The prefix is compared with ``substr``, not
    LIKE: Project ids hold underscores, which LIKE reads as wildcards, so one Project would find another's
    task. A failing query is not caught here; a tick that cannot tell whether a task exists must publish nothing.
    """
    prefix = _key_prefix(project_id)
    row = board.execute(
        "SELECT id FROM tasks WHERE created_by = ? AND substr(idempotency_key, 1, ?) = ? "
        "AND status NOT IN ('done', 'archived') ORDER BY created_at, id LIMIT 1",
        (CREATED_BY, len(prefix), prefix),
    ).fetchone()
    return None if row is None else row["id"]


def _record_open_task(registry: sqlite3.Connection, project_id: str, task_id: str, now: int) -> None:
    """Record a task found on the board as the Project's open one, as the normal claim would.

    A row that already records a task, this one or another, is left alone and nothing is written.
    """
    row = registry.execute(
        "SELECT open_task_id FROM project_summary_state WHERE project_id = ?", (project_id,)
    ).fetchone()
    if row is not None and row["open_task_id"] is not None:
        return
    with write_txn(registry):
        registry.execute(
            "INSERT OR IGNORE INTO project_summary_state (project_id, updated_at) VALUES (?, ?)", (project_id, now)
        )
        recorded = registry.execute(
            "UPDATE project_summary_state SET open_task_id = ?, pending = 0, pending_urgent = 0, "
            "last_requested_at = ?, updated_at = ? WHERE project_id = ? AND open_task_id IS NULL",
            (task_id, now, now, project_id),
        ).rowcount
    if recorded == 1:
        logger.info("project summary: Project %s already has summary task %s; recorded it", project_id, task_id)


def _eligible(registry: sqlite3.Connection, project_id: str) -> bool:
    """The one rule for a first summary: the Project exists, is not archived, and no removal has started.

    A removal counts from its start to its end, a finished permanent one included (the registry keeps that
    Project's row), until it is cancelled, fails or is restored.
    """
    project = registry.execute("SELECT archived FROM projects WHERE id = ?", (project_id,)).fetchone()
    if project is None or project["archived"]:
        return False
    removal = registry.execute(
        "SELECT phase FROM project_removal_operations WHERE project_id = ? "
        "ORDER BY created_at DESC, rowid DESC LIMIT 1",
        (project_id,),
    ).fetchone()
    return removal is None or removal["phase"] in _REMOVAL_ENDED


def _archive(board: sqlite3.Connection, task_id: str) -> bool:
    """Archive the task and say whether it is archived now. An error is not caught: the caller then writes nothing."""
    kanban_db.archive_task(board, task_id)
    task = kanban_db.get_task(board, task_id)
    if task is None or task.status == "archived":
        return True
    logger.warning("project summary: summary task %s could not be archived; nothing is changed for its Project", task_id)
    return False


def _retire(
    registry: sqlite3.Connection, board: Optional[sqlite3.Connection], project_id: str, task_id: str, now: int
) -> None:
    """Close the request of a Project that may not have a summary: nothing written, no failure, still wanted later."""
    # Archive first: a stop before the clear leaves an archived task still recorded, which the next harvest clears.
    # If the Project is live again by then, that harvest counts one failure for it. Owner rule 3 accepts that: it
    # can only delay the next request, by the backoff and at most a day.
    if board is not None and not _archive(board, task_id):
        return
    with write_txn(registry):
        registry.execute(
            "UPDATE project_summary_state SET open_task_id = NULL, pending = 1, updated_at = ? "
            "WHERE project_id = ? AND open_task_id = ?",
            (now, project_id, task_id),
        )
    logger.info("project summary: Project %s may not have a summary now; its request was closed", project_id)


def _record_failure(
    registry: sqlite3.Connection, board: Optional[sqlite3.Connection], project_id: str, task_id: str, now: int
) -> None:
    """One more failure in a row for a request that ended: no open task, a new request armed, nothing else touched."""
    # Archive first: a stop before the count leaves an archived task still recorded, which the next harvest counts once.
    if board is not None and not _archive(board, task_id):
        return
    # Compare-and-set on the recorded task, so two ticks that settle the same task count one failure.
    with write_txn(registry):
        registry.execute(
            "UPDATE project_summary_state SET open_task_id = NULL, failures = failures + 1, pending = 1, "
            "updated_at = ? WHERE project_id = ? AND open_task_id = ?",
            (now, project_id, task_id),
        )


def _record_failure_before_publishing(registry: sqlite3.Connection, project_id: str, now: int) -> None:
    """One more failure in a row where no task was made: only the count and the clock move.

    ``open_task_id`` and ``last_requested_at`` stay as they are (owner rule 1 of 2026-10-27), so nothing
    that identifies a published task can move; the backoff runs from ``updated_at``.
    """
    with write_txn(registry):
        registry.execute(
            "UPDATE project_summary_state SET failures = failures + 1, updated_at = ? WHERE project_id = ?",
            (now, project_id),
        )


def _store_result(registry: sqlite3.Connection, project_id: str, result: SummaryResult, now: int) -> bool:
    """Store the result and zero the Project's failures unless it stopped being eligible, all in one transaction."""
    with write_txn(registry):
        if not _eligible(registry, project_id):
            return False
        revision = projects_db.write_project_summary(
            registry,
            project_id,
            outcome=result.outcome,
            next_action=result.next_action,
            owner_decision=result.owner_decision,
        )
        registry.execute(
            "UPDATE project_summary_state SET open_task_id = NULL, failures = 0, updated_at = ? WHERE project_id = ?",
            (now, project_id),
        )
    logger.info("project summary: stored revision %d for Project %s", revision, project_id)
    return True


def _harvest_one(
    registry: sqlite3.Connection, board: Optional[sqlite3.Connection], project_id: str, task_id: str, now: int
) -> None:
    task = None if board is None else kanban_db.get_task(board, task_id)
    # A task the tick did not make for this Project is not its request, and not the tick's to archive or change.
    foreign = task is not None and not _made_for(task, project_id)
    if not _eligible(registry, project_id):
        _retire(registry, None if foreign else board, project_id, task_id, now)
        return
    if board is None:
        logger.warning("project summary: the summary board is gone; nothing stored for Project %s", project_id)
        _record_failure(registry, None, project_id, task_id, now)
        return
    if foreign:
        logger.warning(
            "project summary: task %s was not made by the tick for Project %s; nothing stored", task_id, project_id
        )
        _record_failure(registry, None, project_id, task_id, now)
        return
    if task is not None and task.status not in _SETTLED:
        return
    try:
        result = parse_summary_result(board, task_id)
    except SummaryRefused as exc:
        logger.warning("project summary: nothing stored for Project %s: %s", project_id, exc)
        _record_failure(registry, board, project_id, task_id, now)
        return
    if not _store_result(registry, project_id, result, now):
        _retire(registry, board, project_id, task_id, now)
        return
    kanban_db.archive_task(board, task_id)


def _harvest(registry: sqlite3.Connection, now: int) -> None:
    open_tasks = registry.execute(
        "SELECT project_id, open_task_id FROM project_summary_state WHERE open_task_id IS NOT NULL"
    ).fetchall()
    if not open_tasks:
        return
    path = _summary_board_path()
    with contextlib.ExitStack() as stack:
        board = stack.enter_context(kanban_db.connect_closing(path, board=SUMMARY_BOARD)) if path.exists() else None
        for row in open_tasks:
            _harvest_one(registry, board, row["project_id"], row["open_task_id"], now)


def _due(state: Mapping[str, Any], now: int) -> bool:
    """The document's rule: nothing open, a request pending, and the hour, the backoff or an urgent flag allows it.

    The hour runs from the last request. After a failure the backoff runs from the failure, which the
    failure recorders stamp in ``updated_at``, also while no request was ever made (``last_requested_at`` is NULL).
    However often it failed, a Project is asked again when its backoff has passed; from ``FAILURE_STOP`` on, a day.
    """
    if state["open_task_id"] is not None or not state["pending"]:
        return False
    requested_at = state["last_requested_at"]
    if not state["failures"] and (requested_at is None or state["pending_urgent"]):
        return True
    since = state["updated_at"] if state["failures"] else requested_at
    return now - since >= backoff_seconds(state["failures"])


def _new_state(registry: sqlite3.Connection, project_id: str, now: int) -> sqlite3.Row:
    """The state row of a Project the tick has not seen. A Project that already has a summary needs no first one."""
    asked_for = 0 if projects_db.get_project_summary(registry, project_id) is not None else 1
    with write_txn(registry):
        registry.execute(
            "INSERT OR IGNORE INTO project_summary_state (project_id, pending, pending_urgent, updated_at) "
            "VALUES (?, ?, ?, ?)",
            (project_id, asked_for, asked_for, now),
        )
    return registry.execute("SELECT * FROM project_summary_state WHERE project_id = ?", (project_id,)).fetchone()


def _open_summary_task(
    registry: sqlite3.Connection,
    board: sqlite3.Connection,
    project: projects_db.Project,
    state: Mapping[str, Any],
    now: int,
    reading: events.Reading,
) -> bool:
    # Imported here: the owner kernel is large, and the first import must not run on the event loop.
    from hermes_cli import owner_workspace

    try:
        route = owner_workspace._resolved_route_pin(PLANNING_ROLE, EXECUTION_TIER, "execution_tier")
    except owner_workspace.OwnerWorkspaceError as exc:
        logger.warning("project summary: no usable route for the planning role (%s)", exc.code)
        _record_failure_before_publishing(registry, project.id, now)
        return False
    try:
        snapshot = owner_workspace.project_steward_snapshot(project_id=project.id)
    except owner_workspace.OwnerWorkspaceError as exc:
        logger.warning("project summary: no steward snapshot for Project %s (%s)", project.id, exc.code)
        _record_failure_before_publishing(registry, project.id, now)
        return False

    # No time box is passed: the kernel pins it for a route-locked card (_pin_new_card).
    task_id = kanban_db.create_task(
        board,
        title=TITLE_PREFIX + project.name,
        body=_task_body(snapshot),
        assignee=PLANNING_ROLE,
        created_by=CREATED_BY,
        workspace_kind="scratch",
        board=SUMMARY_BOARD,
        project_id=None,
        owned_paths=[],
        risk_tier=RISK_TIER,
        idempotency_key=_request_key(state),
        **route,
    )
    # create_task returns the task that already holds the key, whoever made it and whatever its status.
    made = kanban_db.get_task(board, task_id)
    if made is None or not _made_for(made, project.id):
        logger.warning(
            "project summary: task %s holds the request key of Project %s but the tick did not make it; "
            "nothing is recorded",
            task_id, project.id,
        )
        _record_failure_before_publishing(registry, project.id, now)
        return False
    # The claim holds only while the row is the one the key was made from. A loser archives its task only
    # if the winner recorded another one, and decides that inside the transaction that nobody else can write.
    # The cursors move here, to where the events were read before the snapshot, and not before: a request that
    # fails first must leave the row as it was. The Project may be pending only through this tick's own reading.
    with write_txn(registry):
        claimed = registry.execute(
            "UPDATE project_summary_state SET open_task_id = ?, last_requested_at = ?, pending = 0, "
            "pending_urgent = 0, updated_at = ?, event_cursor = ?, delivery_cursor = ?, release_cursor = ? "
            "WHERE project_id = ? AND open_task_id IS NULL AND (pending = 1 OR ?) "
            "AND failures = ? AND last_requested_at IS ? "
            "AND event_cursor = ? AND delivery_cursor = ? AND release_cursor = ?",
            (
                task_id, now, now, *reading.batch.cursors,
                project.id, int(reading.raises_pending), state["failures"], state["last_requested_at"],
                *reading.old,
            ),
        ).rowcount
        if claimed != 1:
            recorded = registry.execute(
                "SELECT open_task_id FROM project_summary_state WHERE project_id = ?", (project.id,)
            ).fetchone()
            if recorded is None or recorded["open_task_id"] != task_id:
                kanban_db.archive_task(board, task_id)
            return False
    logger.info("project summary: asked for the first summary of Project %s (task %s)", project.id, task_id)
    return True


def _request(registry: sqlite3.Connection, now: int) -> int:
    states = {row["project_id"]: row for row in registry.execute("SELECT * FROM project_summary_state")}
    path = _summary_board_path()
    asked = created = 0
    ensured = False
    with contextlib.ExitStack() as stack:
        board: Optional[sqlite3.Connection] = None
        for project in projects_db.list_projects(registry):
            if asked >= MAX_NEW_PER_TICK:
                break
            # Owner rule 1 of 2026-10-27: the board, not the state row, proves that a task was published, so
            # this comes before every other read of the turn. A task made before a crash is found whatever
            # fails afterwards. The check never makes the board; a failing query ends the tick.
            if board is None and path.exists():
                board = stack.enter_context(kanban_db.connect_closing(path, board=SUMMARY_BOARD))
            open_task = None if board is None else _find_open_task(board, project.id)
            if open_task is not None:
                _record_open_task(registry, project.id, open_task, now)
                continue
            if not _eligible(registry, project.id):
                continue
            if project.id not in states:
                _new_state(registry, project.id, now)
            # The one read of events in the turn (card B). Nothing is written until we know whether a request is due.
            reading = events.read(registry, project)
            state = reading.state
            if not _due(state, now):
                events.record(registry, project.id, reading)
                continue
            if not ensured:
                if _ensure_summary_board() is None:
                    break
                ensured = True
            if board is None:
                board = stack.enter_context(kanban_db.connect_closing(path, board=SUMMARY_BOARD))
            asked += 1
            created += _open_summary_task(registry, board, project, state, now, reading)
    return created


def summary_tick(now: Optional[int] = None) -> int:
    """Run one summary tick and return how many summary tasks it opened.

    ``now`` is the clock in epoch seconds; it exists so a test can fix the hour and the backoff.
    """
    now = int(time.time()) if now is None else int(now)
    registry_path = kanban_db.kanban_home() / "projects.db"
    if not registry_path.exists():
        return 0
    with projects_db.connect_closing(registry_path) as registry:
        _harvest(registry, now)
        return _request(registry, now)
