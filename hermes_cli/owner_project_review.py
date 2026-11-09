"""The daily project review tick: once per Vienna day, one review of each allowlisted Project that has news.

The gateway's dispatcher is to run ``review_tick`` right after ``summary_tick`` (card B2 wires the call). Nothing
runs until the owner switches it on: ``read_settings`` reads the ``owner_review`` section of the root config, and in
code every flag is off and the allowlist is empty (operator rule L3). Only a Project named in the allowlist, by id or
slug, that is active as the plan defines it, is reviewed: the summary's ``_eligible`` still counts it as live, the owner
has neither paused nor archived its board, and its steward state is neither "deleted" nor "complete".

The review day is the Vienna calendar date. A Project is due once the Vienna clock has reached the review hour of a
date later than its ``last_reviewed_date``. A calendar date rather than a 24-hour window keeps one review per date on
the 23-hour and the 25-hour day alike. A Project that missed days gets one catch-up review dated today, which covers
everything since its last review day (the "Review days" line of the task body).

A due Project's events past its own cursors are read with ``REVIEW_EVENT_CLASS``: the summary's ``EVENT_CLASS``,
plus owner comments and suggestion events as material. A Project with no material and no urgent event is settled by
the kernel alone: no task, the review day is today, and the cursors move past the ignored events. A Project with news
gets one task on the summary board, created by ``CREATED_BY`` under the key
``project-review:<project_id>:<review_date>:<failures>``. The date in the key makes one review per Project and day
structural; a retry changes only the failure count.

The quiet clock, ``last_news_date``, is the Vienna date of the Project's newest material or urgent event, by the
event's own time. Only ``_observe_news`` sets it: on every tick, for every allowed Project, before the review hour and
before the skips for an open review, a backoff or the cap. A review day, a deferral, a failure or a retry never sets
it. ``quiet_days`` and ``is_stale`` count from it.

The rules of the summary tick hold here as they are. The board, not the state row, proves that a task was published,
so the first thing in each Project's turn is the look for its open task, and a task found there is recorded and
counted. A task is the tick's own only when ``_made_for`` says so. A failed task is archived first, and only then does
one compare-and-set clear it and count the failure. A failure never moves ``last_reviewed_date`` or a cursor, and the
retry waits for ``backoff_seconds``: an hour, two hours, and a whole day from the third failure on.

The daily cap counts every review task started on a review day, retries included, in ``review_budget``. A task is
counted in the transaction that claims it, or that records it when a crash came between the two. When the cap is
reached, the Projects with news wait in this order: urgent first, then more material events, then more days since
the last review, then the older Project. The cross-project review has a slot of its own (``take_portfolio_slot``).
At most ``MAX_NEW_PER_TICK`` new tasks are asked for per tick.

Card B1 has no result reader yet (card B5). Its harvest refuses every finished review and counts a failure, so the
review must stay switched off until that card is in. Nor does card B1 hold a review task to kanban_show and
kanban_complete: card B-S1 does, at dispatch, for every task created by ``CREATED_BY``, and the owner switches the
review on only after that card is released.

Every store is reached by an explicit path: the root registry, the root config and every board are found under the
pins the dispatcher gives a worker (``HERMES_KANBAN_DB``, ``HERMES_KANBAN_BOARD``, a profile ``HERMES_HOME``).
"""

from __future__ import annotations

import contextlib
import logging
import re
import sqlite3
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any, Mapping, NamedTuple, Optional
from zoneinfo import ZoneInfo

from hermes_cli import kanban_db, projects_db, release_ledger
from hermes_cli import owner_project_summary as summary
from hermes_cli import owner_project_summary_events as events
from hermes_cli.sqlite_util import write_txn

logger = logging.getLogger(__name__)

VIENNA = ZoneInfo("Europe/Vienna")
SETTINGS_KEY = "owner_review"
CREATED_BY = "project-review"
TITLE_PREFIX = "Review the project: "

# The owner kernel writes an owner comment with the trusted actor, the default profile; a worker writes its own
# profile name, or "worker" (tools/kanban_tools.py).
OWNER_AUTHOR = "default"

# The summary's table, plus what matters to a daily look and not to the summary's hourly rate.
REVIEW_EVENT_CLASS = {
    **events.EVENT_CLASS,
    "commented:owner": events.MATERIAL,
    "recommendation_created": events.MATERIAL,
    "recommendation_decided": events.MATERIAL,
}

# The quiet clock reads at most this many rows of each source per tick, newest first; a source that reaches it gives no
# date in that tick.
NEWS_ROW_CAP = 100

_SETTLED = ("done", "blocked", "archived")
_NEVER_REVIEWED = "the start"
_WINDOW = re.compile(r"^Review days: from ([0-9]{4}-[0-9]{2}-[0-9]{2}|the start) to ([0-9]{4}-[0-9]{2}-[0-9]{2})$", re.MULTILINE)

INSTRUCTION = """\
Review one Project for its owner. Use no tool except kanban_show and kanban_complete.

1. Read this task with kanban_show.
2. Finish with kanban_complete and a one-sentence summary.
"""


@dataclass(frozen=True)
class ReviewSettings:
    """The owner's settings. The defaults are the code's own: off, with nothing allowed."""

    enabled: bool = False
    allowlist: tuple[str, ...] = ()
    closure: bool = False
    digest: bool = False
    review_hour: int = 6
    daily_cap: int = 20
    portfolio_cap: int = 1
    stale_days: int = 14


def _flag(value: Any) -> bool:
    return value is True


def _whole(value: Any, default: int, low: int, high: Optional[int] = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        return default
    if value < low or (high is not None and value > high):
        return default
    return value


def _names(value: Any) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    return tuple(dict.fromkeys(item.strip() for item in value if isinstance(item, str) and item.strip()))


def read_settings() -> ReviewSettings:
    """The ``owner_review`` section of the root config. A value that is not of its kind keeps the code's default.

    The root's config, not the active profile's: a worker's ``HERMES_HOME`` is its profile home.
    """
    from hermes_cli.config import cfg_get, load_config_readonly
    from hermes_constants import get_default_hermes_root, reset_hermes_home_override, set_hermes_home_override

    defaults = ReviewSettings()
    token = set_hermes_home_override(get_default_hermes_root())
    try:
        section = cfg_get(load_config_readonly(), SETTINGS_KEY, default={})
    except Exception as exc:  # a config that cannot be read keeps the review off
        logger.warning("project review: the config cannot be read (%s); the review stays off", exc)
        return defaults
    finally:
        reset_hermes_home_override(token)
    if not isinstance(section, dict):
        return defaults
    return ReviewSettings(
        enabled=_flag(section.get("enabled")),
        allowlist=_names(section.get("allowlist")),
        closure=_flag(section.get("closure")),
        digest=_flag(section.get("digest")),
        review_hour=_whole(section.get("review_hour"), defaults.review_hour, 0, 23),
        daily_cap=_whole(section.get("daily_cap"), defaults.daily_cap, 0),
        portfolio_cap=_whole(section.get("portfolio_cap"), defaults.portfolio_cap, 0),
        stale_days=_whole(section.get("stale_days"), defaults.stale_days, 1),
    )


# ---------------------------------------------------------------------------------------------------------------
# The Vienna day
# ---------------------------------------------------------------------------------------------------------------


def vienna_date(now: int) -> str:
    """The Vienna calendar date of the epoch second ``now``, as 'YYYY-MM-DD'."""
    return datetime.fromtimestamp(now, VIENNA).date().isoformat()


def review_date(now: int, review_hour: int) -> Optional[str]:
    """The Vienna date whose review may run at ``now``, or None while that date's clock is before ``review_hour``."""
    local = datetime.fromtimestamp(now, VIENNA)
    return local.date().isoformat() if local.hour >= review_hour else None


def _window_line(last_reviewed_date: Optional[str], review_day: str) -> str:
    """The days a review covers: from the day after the last review day, or from the start, to the review day."""
    start = (
        _NEVER_REVIEWED if last_reviewed_date is None
        else (date.fromisoformat(last_reviewed_date) + timedelta(days=1)).isoformat()
    )
    return f"Review days: from {start} to {review_day}"


def review_window(body: Optional[str]) -> Optional[tuple[Optional[str], str]]:
    """``(covered_from, covered_to)`` as a task's body names them; covered_from is None for a first review."""
    named = _WINDOW.findall(body or "")
    if not named:
        return None
    start, end = named[-1]
    return (None if start == _NEVER_REVIEWED else start), end


def quiet_days(last_news_date: Optional[str], today: str) -> int:
    """The Vienna days from the last day with news to ``today``; 0 while no news of the Project is known."""
    if last_news_date is None:
        return 0
    return max(0, (date.fromisoformat(today) - date.fromisoformat(last_news_date)).days)


def is_stale(last_news_date: Optional[str], today: str, stale_days: int) -> bool:
    """Whether the Project has been quiet for at least ``stale_days`` days."""
    return last_news_date is not None and quiet_days(last_news_date, today) >= stale_days


# ---------------------------------------------------------------------------------------------------------------
# What counts as news for a review
# ---------------------------------------------------------------------------------------------------------------


def review_classify(kind: str, detail: Optional[str] = None) -> str:
    """The class of an event for the review: ``kind:detail`` if the table has it, else ``kind``, else ignored."""
    if detail is not None and f"{kind}:{detail}" in REVIEW_EVENT_CLASS:
        return REVIEW_EVENT_CLASS[f"{kind}:{detail}"]
    return REVIEW_EVENT_CLASS.get(kind, events.IGNORED)


def _detail(board: sqlite3.Connection, event: sqlite3.Row) -> Optional[str]:
    if event["kind"] == "commented":
        return "owner" if events._payload(event["payload"]).get("author") == OWNER_AUTHOR else None
    return events._block_kind(board, event)


@dataclass(frozen=True)
class ReviewBatch(events.Batch):
    """``events.Batch``, and when the newest material or urgent event it counts happened (epoch seconds)."""

    newest: Optional[int] = None


def _count_review_events(board: sqlite3.Connection, after: int, upto: int, counts: dict, times: list) -> int:
    last = after
    for event in board.execute(
        "SELECT id, task_id, kind, payload, created_at FROM task_events WHERE id > ? AND id <= ? ORDER BY id",
        (after, upto),
    ):
        last = event["id"]
        event_class = review_classify(event["kind"], _detail(board, event))
        counts[event_class] += 1
        if event_class != events.IGNORED:
            times.append(event["created_at"])
    return last


def _count_review_merges(board: sqlite3.Connection, after: int, upto: int, counts: dict, times: list) -> int:
    """``events._count_merges``, dated by when each row was recorded.

    A row keeps no merge time. The ``delivery_merged`` receipt on its card, read in the same snapshot, carries it.
    """
    try:
        merged = board.execute(
            "SELECT id, created_at FROM kanban_deliveries WHERE pull_request_state = ? AND id > ? AND id <= ? "
            "ORDER BY id",
            (events._MERGED, after, upto),
        ).fetchall()
    except sqlite3.OperationalError as exc:
        if not events._missing_schema(exc):
            raise
        return after
    counts[review_classify("merged")] += len(merged)
    times.extend(row["created_at"] for row in merged)
    return merged[-1]["id"] if merged else after


def _count_review_releases(
    board: sqlite3.Connection, slug: str, after: int, upto: int, counts: dict, times: list
) -> int:
    """``events._count_releases``, dated by when each release finished."""
    path = release_ledger.ledger_path()
    if not path.exists():
        return after
    last = after
    try:
        with events._read_only(path) as ledger:
            for event in ledger.execute(
                "SELECT id, batch_id, kind, payload, created_at FROM release_events WHERE id > ? AND id <= ? "
                "ORDER BY id",
                (after, upto),
            ):
                last = event["id"]
                released = (
                    event["kind"] == "release_finished"
                    and events._payload(event["payload"]).get("outcome") == events._RELEASED
                )
                if released and events._holds_card(ledger, board, slug, event["batch_id"]):
                    counts[review_classify("released")] += 1
                    times.append(event["created_at"])
    except sqlite3.OperationalError as exc:
        if not events._missing_schema(exc):
            raise
        return after
    return last


def read_review_events(
    project: projects_db.Project, cursors: events.Cursors, until: events.Cursors = events._END
) -> ReviewBatch:
    """``events.read_events`` with the review's own table, and dated. It reads, read-only, and moves nothing."""
    slug = project.board_slug
    if not slug or slug == summary.SUMMARY_BOARD:
        return ReviewBatch(cursors)
    path = events._board_path(slug)
    if not path.exists():
        return ReviewBatch(cursors)
    counts = {events.MATERIAL: 0, events.URGENT: 0, events.IGNORED: 0}
    times: list[int] = []
    with events._read_only(path) as board:
        moved = events.Cursors(
            _count_review_events(board, cursors.event, until.event, counts, times),
            _count_review_merges(board, cursors.delivery, until.delivery, counts, times),
            _count_review_releases(board, slug, cursors.release, until.release, counts, times),
        )
    return ReviewBatch(moved, counts[events.MATERIAL], counts[events.URGENT], max(times, default=None))


# ---------------------------------------------------------------------------------------------------------------
# The daily cap
# ---------------------------------------------------------------------------------------------------------------


def budget(registry: sqlite3.Connection, review_day: str) -> tuple[int, int]:
    """``(project_reviews, portfolio_reviews)`` started on ``review_day``."""
    row = registry.execute(
        "SELECT project_reviews, portfolio_reviews FROM review_budget WHERE review_date = ?", (review_day,)
    ).fetchone()
    return (0, 0) if row is None else (row["project_reviews"], row["portfolio_reviews"])


def _spend(registry: sqlite3.Connection, review_day: str, column: str, cap: Optional[int], now: int) -> bool:
    """Count one task of ``column`` on ``review_day`` unless ``cap`` are counted. Joins the caller's transaction."""
    with write_txn(registry):
        registry.execute(
            "INSERT OR IGNORE INTO review_budget (review_date, updated_at) VALUES (?, ?)", (review_day, now)
        )
        return registry.execute(
            f"UPDATE review_budget SET {column} = {column} + 1, updated_at = ? "
            f"WHERE review_date = ? AND (? IS NULL OR {column} < ?)",
            (now, review_day, cap, cap),
        ).rowcount == 1


def take_portfolio_slot(registry: sqlite3.Connection, review_day: str, cap: int, now: int) -> bool:
    """Take the cross-project review's slot of ``review_day``, outside the project cap; False once ``cap`` are taken.

    The portfolio review (card B10) calls it in the transaction that claims its task.
    """
    return _spend(registry, review_day, "portfolio_reviews", cap, now)


# ---------------------------------------------------------------------------------------------------------------
# Tasks, keys and the one open task
# ---------------------------------------------------------------------------------------------------------------


def _key_prefix(project_id: str) -> str:
    return f"{CREATED_BY}:{project_id}:"


def _request_key(project_id: str, review_day: str, failures: int) -> str:
    return f"{_key_prefix(project_id)}{review_day}:{failures}"


def _key_date(key: Optional[str], project_id: str) -> Optional[str]:
    """The review date a key of the Project names, or None if it names none."""
    day = (key or "")[len(_key_prefix(project_id)):].split(":", 1)[0]
    try:
        return date.fromisoformat(day).isoformat()
    except ValueError:
        return None


def _made_for(task: kanban_db.Task, project_id: str) -> bool:
    """Whether the review tick made the task for this Project: the creator is the tick, the key names the Project."""
    return task.created_by == CREATED_BY and (task.idempotency_key or "").startswith(_key_prefix(project_id))


def _find_open_task(board: sqlite3.Connection, project_id: str) -> Optional[sqlite3.Row]:
    """The oldest review task the tick made for the Project that is neither done nor archived, as in the summary."""
    prefix = _key_prefix(project_id)
    return board.execute(
        "SELECT id, idempotency_key FROM tasks WHERE created_by = ? AND substr(idempotency_key, 1, ?) = ? "
        "AND status NOT IN ('done', 'archived') ORDER BY created_at, id LIMIT 1",
        (CREATED_BY, len(prefix), prefix),
    ).fetchone()


def _record_open_task(registry: sqlite3.Connection, project_id: str, task: sqlite3.Row, now: int) -> None:
    """Record a task found on the board as the Project's open review, and count it on its review day.

    A row that records a task already is left alone, so a task is counted once: by its claim, or here when the
    publisher stopped before its claim. A key without a date counts on today.
    """
    row = registry.execute(
        "SELECT open_task_id FROM project_review_state WHERE project_id = ?", (project_id,)
    ).fetchone()
    if row is not None and row["open_task_id"] is not None:
        return
    review_day = _key_date(task["idempotency_key"], project_id) or vienna_date(now)
    with write_txn(registry):
        registry.execute(
            "INSERT OR IGNORE INTO project_review_state (project_id, updated_at) VALUES (?, ?)", (project_id, now)
        )
        recorded = registry.execute(
            "UPDATE project_review_state SET open_task_id = ?, open_review_date = ?, updated_at = ? "
            "WHERE project_id = ? AND open_task_id IS NULL",
            (task["id"], review_day, now, project_id),
        ).rowcount
        if recorded == 1:
            _spend(registry, review_day, "project_reviews", None, now)
    if recorded == 1:
        logger.info("project review: Project %s already has review task %s; recorded it", project_id, task["id"])


def _allowed(registry: sqlite3.Connection, settings: ReviewSettings, project_id: str) -> bool:
    """Whether the owner allows a review of the Project now: allowlisted by id or slug, and active."""
    row = registry.execute("SELECT slug, board_slug FROM projects WHERE id = ?", (project_id,)).fetchone()
    if row is None or (project_id not in settings.allowlist and row["slug"] not in settings.allowlist):
        return False
    return summary._eligible(registry, project_id) and _active(project_id, row["board_slug"])


def _active(project_id: str, board_slug: Optional[str]) -> bool:
    """Whether the Project is active as the plan defines it, past the summary's ``_eligible``.

    The owner has neither paused nor archived its board, and its steward state is neither "deleted" nor "complete".
    A Project whose state cannot be read is not active.
    """
    if not board_slug:
        return False
    metadata = kanban_db.read_board_metadata(board_slug)
    if metadata.get("dispatch_paused_by_owner") is True or metadata.get("archived") is True:
        return False
    # Imported here: the owner kernel is large, and the first import must not run on the event loop.
    from hermes_cli import owner_workspace

    try:
        state = owner_workspace.project_steward_snapshot(project_id=project_id)["execution"]["state"]
    except owner_workspace.OwnerWorkspaceError as exc:
        logger.warning("project review: the state of Project %s cannot be read (%s)", project_id, exc.code)
        return False
    return state not in ("deleted", "complete")


def _archive(board: sqlite3.Connection, task_id: str) -> bool:
    """Archive the task and say whether it is archived now. An error is not caught: the caller then writes nothing."""
    kanban_db.archive_task(board, task_id)
    task = kanban_db.get_task(board, task_id)
    if task is None or task.status == "archived":
        return True
    logger.warning("project review: review task %s could not be archived; nothing is changed for its Project", task_id)
    return False


def _retire(
    registry: sqlite3.Connection, board: Optional[sqlite3.Connection], project_id: str, task_id: str, now: int
) -> None:
    """Close the review of a Project the owner no longer allows: no failure, and the review day stays."""
    if board is not None and not _archive(board, task_id):
        return
    with write_txn(registry):
        registry.execute(
            "UPDATE project_review_state SET open_task_id = NULL, open_review_date = NULL, updated_at = ? "
            "WHERE project_id = ? AND open_task_id = ?",
            (now, project_id, task_id),
        )
    logger.info("project review: Project %s may not be reviewed now; its review was closed", project_id)


def _record_failure(
    registry: sqlite3.Connection, board: Optional[sqlite3.Connection], project_id: str, task_id: str, now: int
) -> None:
    """One more failure in a row: archive first, then clear the task and count, by compare-and-set on the task."""
    if board is not None and not _archive(board, task_id):
        return
    with write_txn(registry):
        registry.execute(
            "UPDATE project_review_state SET open_task_id = NULL, open_review_date = NULL, "
            "failures = failures + 1, updated_at = ? WHERE project_id = ? AND open_task_id = ?",
            (now, project_id, task_id),
        )


def _record_failure_before_publishing(registry: sqlite3.Connection, project_id: str, now: int) -> None:
    """One more failure in a row where no task was made: only the count and the clock move."""
    with write_txn(registry):
        registry.execute(
            "UPDATE project_review_state SET failures = failures + 1, updated_at = ? "
            "WHERE project_id = ? AND open_task_id IS NULL",
            (now, project_id),
        )


# ---------------------------------------------------------------------------------------------------------------
# Harvest
# ---------------------------------------------------------------------------------------------------------------


def _harvest_one(
    registry: sqlite3.Connection,
    board: Optional[sqlite3.Connection],
    settings: ReviewSettings,
    project_id: str,
    task_id: str,
    now: int,
) -> None:
    task = None if board is None else kanban_db.get_task(board, task_id)
    # A task the tick did not make for this Project is not its review, and not the tick's to archive or change.
    foreign = task is not None and not _made_for(task, project_id)
    if not _allowed(registry, settings, project_id):
        _retire(registry, None if foreign else board, project_id, task_id, now)
        return
    if board is None:
        logger.warning("project review: the summary board is gone; nothing stored for Project %s", project_id)
        _record_failure(registry, None, project_id, task_id, now)
        return
    if foreign:
        logger.warning("project review: task %s was not made by the tick for Project %s", task_id, project_id)
        _record_failure(registry, None, project_id, task_id, now)
        return
    if task is not None and task.status not in _SETTLED:
        return
    # No result reader exists yet (card B5 adds parse_review_result), so no result is stored: the review counts as
    # failed, and neither the review day nor a cursor moves.
    logger.warning("project review: no result of task %s can be stored for Project %s yet", task_id, project_id)
    _record_failure(registry, board, project_id, task_id, now)


def _harvest(registry: sqlite3.Connection, settings: ReviewSettings, now: int) -> None:
    open_tasks = registry.execute(
        "SELECT project_id, open_task_id FROM project_review_state WHERE open_task_id IS NOT NULL"
    ).fetchall()
    if not open_tasks:
        return
    path = summary._summary_board_path()
    with contextlib.ExitStack() as stack:
        board = (
            stack.enter_context(kanban_db.connect_closing(path, board=summary.SUMMARY_BOARD)) if path.exists() else None
        )
        for row in open_tasks:
            _harvest_one(registry, board, settings, row["project_id"], row["open_task_id"], now)


# ---------------------------------------------------------------------------------------------------------------
# Request
# ---------------------------------------------------------------------------------------------------------------


class _News(NamedTuple):
    """A due Project whose read found news: the row as the tick found it, and what the read counted."""

    project: projects_db.Project
    state: Mapping[str, Any]
    batch: events.Batch


def _cursors(state: Mapping[str, Any]) -> events.Cursors:
    return events.Cursors(state["event_cursor"], state["delivery_cursor"], state["release_cursor"])


def _state_row(registry: sqlite3.Connection, project_id: str, now: int) -> sqlite3.Row:
    row = registry.execute("SELECT * FROM project_review_state WHERE project_id = ?", (project_id,)).fetchone()
    if row is not None:
        return row
    with write_txn(registry):
        registry.execute(
            "INSERT OR IGNORE INTO project_review_state (project_id, updated_at) VALUES (?, ?)", (project_id, now)
        )
    return registry.execute("SELECT * FROM project_review_state WHERE project_id = ?", (project_id,)).fetchone()


def _due(state: Mapping[str, Any], review_day: str, now: int) -> bool:
    """No open task, the review day not yet reviewed, and after a failure the backoff passed (from ``updated_at``)."""
    if state["open_task_id"] is not None:
        return False
    if state["last_reviewed_date"] is not None and state["last_reviewed_date"] >= review_day:
        return False
    return not state["failures"] or now - state["updated_at"] >= summary.backoff_seconds(state["failures"])


def _settle_quiet(
    registry: sqlite3.Connection, project_id: str, state: Mapping[str, Any], batch: events.Batch, review_day: str,
    now: int,
) -> None:
    """The kernel's own review of a Project without news: the review day is today and no task is made.

    The cursors move past the ignored events. The quiet clock is left to ``_observe_news``: a review day is no news.
    The swap holds only while the row is as the tick read it, so two ticks settle a day once.
    """
    with write_txn(registry):
        registry.execute(
            "UPDATE project_review_state SET last_reviewed_date = ?, last_reviewed_at = ?, failures = 0, "
            "updated_at = ?, event_cursor = ?, delivery_cursor = ?, release_cursor = ? "
            "WHERE project_id = ? AND open_task_id IS NULL AND failures = ? AND last_reviewed_date IS ? "
            "AND event_cursor = ? AND delivery_cursor = ? AND release_cursor = ?",
            (
                review_day, now, now, *batch.cursors,
                project_id, state["failures"], state["last_reviewed_date"], *_cursors(state),
            ),
        )


def _observe_news(registry: sqlite3.Connection, project: projects_db.Project, now: int) -> None:
    """The quiet clock: ``last_news_date`` becomes the Vienna date of the Project's newest material or urgent event.

    It reads the sources of ``read_review_events``, dated alike, newest first and only back to the day after the
    stored date, which it never reads at all once that date is today. Each source stops at its first news event; one
    that reaches ``NEWS_ROW_CAP`` rows first gives no date in this tick. The date is the event's own, never a review
    day's, a cursor's, a deferral's, a failure's or a retry's, and it never moves back. Nothing else in the row
    changes, ``updated_at`` included; a row is made only for a date to store. A failed read leaves the clock as it is.
    """
    row = registry.execute(
        "SELECT last_news_date FROM project_review_state WHERE project_id = ?", (project.id,)
    ).fetchone()
    after = (row["last_news_date"] if row is not None else None) or ""  # "" sorts before every date
    slug = project.board_slug
    if after >= vienna_date(now) or not slug or slug == summary.SUMMARY_BOARD:
        return
    path = events._board_path(slug)
    if not path.exists():
        return
    found: list[int] = []
    try:
        with events._read_only(path) as board:
            for event in board.execute(
                "SELECT id, task_id, kind, payload, created_at FROM task_events ORDER BY id DESC LIMIT ?",
                (NEWS_ROW_CAP,),
            ):
                if vienna_date(event["created_at"]) <= after:
                    break
                if review_classify(event["kind"], _detail(board, event)) != events.IGNORED:
                    found.append(event["created_at"])
                    break
            try:
                # Every merge is urgent, so the newest merged row decides.
                merged = board.execute(
                    "SELECT created_at FROM kanban_deliveries WHERE pull_request_state = ? ORDER BY id DESC LIMIT 1",
                    (events._MERGED,),
                ).fetchone()
            except sqlite3.OperationalError as exc:
                if not events._missing_schema(exc):
                    raise
                merged = None
            if merged is not None and vienna_date(merged["created_at"]) > after:
                found.append(merged["created_at"])
            ledger_path = release_ledger.ledger_path()
            if ledger_path.exists():
                try:
                    with events._read_only(ledger_path) as ledger:
                        for event in ledger.execute(
                            "SELECT id, batch_id, kind, payload, created_at FROM release_events "
                            "ORDER BY id DESC LIMIT ?",
                            (NEWS_ROW_CAP,),
                        ):
                            if vienna_date(event["created_at"]) <= after:
                                break
                            released = (
                                event["kind"] == "release_finished"
                                and events._payload(event["payload"]).get("outcome") == events._RELEASED
                            )
                            if released and events._holds_card(ledger, board, slug, event["batch_id"]):
                                found.append(event["created_at"])
                                break
                except sqlite3.OperationalError as exc:
                    if not events._missing_schema(exc):
                        raise
    except (sqlite3.Error, OSError, ValueError, TypeError, OverflowError) as exc:
        logger.warning(
            "project review: cannot read the news of Project %s (%s); its quiet clock stays", project.id, exc
        )
        return
    if not found:
        return
    news_day = vienna_date(max(found))
    with write_txn(registry):
        registry.execute(
            "INSERT OR IGNORE INTO project_review_state (project_id, updated_at) VALUES (?, ?)", (project.id, now)
        )
        registry.execute(
            "UPDATE project_review_state SET last_news_date = ? "
            "WHERE project_id = ? AND (last_news_date IS NULL OR last_news_date < ?)",
            (news_day, project.id, news_day),
        )


def _wait_order(news: _News, review_day: str) -> tuple:
    """Urgent first, then more material events, then more days since the last review, then the older Project."""
    last = news.state["last_reviewed_date"]
    waited = (date.fromisoformat(review_day) - date.fromisoformat(last)).days if last else 10**6
    return (-int(bool(news.batch.urgent)), -news.batch.material, -waited, news.project.created_at, news.project.id)


def _task_body(last_reviewed_date: Optional[str], review_day: str, covered: events.Cursors) -> str:
    """The instruction, the days the review covers, and last the line that names the events it covers."""
    return (
        INSTRUCTION
        + f"\nReview date: {review_day}\n"
        + _window_line(last_reviewed_date, review_day)
        + "\n"
        + events.covered_line(covered)
        + "\n"
    )


def _open_review_task(
    registry: sqlite3.Connection, board: sqlite3.Connection, news: _News, review_day: str, cap: int, now: int
) -> bool:
    # Imported here: the owner kernel is large, and the first import must not run on the event loop.
    from hermes_cli import owner_workspace

    project, state, batch = news
    try:
        route = owner_workspace._resolved_route_pin(summary.PLANNING_ROLE, summary.EXECUTION_TIER, "execution_tier")
    except owner_workspace.OwnerWorkspaceError as exc:
        logger.warning("project review: no usable route for the planning role (%s)", exc.code)
        _record_failure_before_publishing(registry, project.id, now)
        return False

    # No time box is passed: the kernel pins it for a route-locked card.
    task_id = kanban_db.create_task(
        board,
        title=TITLE_PREFIX + project.name,
        body=_task_body(state["last_reviewed_date"], review_day, batch.cursors),
        assignee=summary.PLANNING_ROLE,
        created_by=CREATED_BY,
        workspace_kind="scratch",
        board=summary.SUMMARY_BOARD,
        project_id=None,
        owned_paths=[],
        risk_tier=summary.RISK_TIER,
        idempotency_key=_request_key(project.id, review_day, state["failures"]),
        **route,
    )
    # create_task returns the task that already holds the key, whoever made it and whatever its status.
    made = kanban_db.get_task(board, task_id)
    if made is None or not _made_for(made, project.id):
        logger.warning(
            "project review: task %s holds the review key of Project %s but the tick did not make it", task_id, project.id
        )
        _record_failure_before_publishing(registry, project.id, now)
        return False
    # The claim holds only while the row is the one the key was made from, and only while the day's cap has room;
    # the task is counted in the same transaction. The cursors do not move here: a review moves them when its result
    # is stored, so a failed one leaves its events for the retry. A loser archives its task only if the row does not
    # record that very task.
    with write_txn(registry):
        claimed = 0
        if budget(registry, review_day)[0] < cap:
            claimed = registry.execute(
                "UPDATE project_review_state SET open_task_id = ?, open_review_date = ?, updated_at = ? "
                "WHERE project_id = ? AND open_task_id IS NULL AND failures = ? AND last_reviewed_date IS ?",
                (task_id, review_day, now, project.id, state["failures"], state["last_reviewed_date"]),
            ).rowcount
            if claimed == 1:
                _spend(registry, review_day, "project_reviews", None, now)
        if claimed != 1:
            recorded = registry.execute(
                "SELECT open_task_id FROM project_review_state WHERE project_id = ?", (project.id,)
            ).fetchone()
            if recorded is None or recorded["open_task_id"] != task_id:
                kanban_db.archive_task(board, task_id)
            return False
    logger.info("project review: asked for the %s review of Project %s (task %s)", review_day, project.id, task_id)
    return True


def _request(registry: sqlite3.Connection, settings: ReviewSettings, now: int) -> int:
    review_day = review_date(now, settings.review_hour)
    path = summary._summary_board_path()
    news: list[_News] = []
    created = asked = 0
    with contextlib.ExitStack() as stack:
        board: Optional[sqlite3.Connection] = None
        for project in projects_db.list_projects(registry):
            # Owner rule 1 of 2026-10-27, as in the summary tick: the board proves that a task was published, so this
            # comes before every other read of the turn. The check never makes the board; a failing query ends the tick.
            if board is None and path.exists():
                board = stack.enter_context(kanban_db.connect_closing(path, board=summary.SUMMARY_BOARD))
            open_task = None if board is None else _find_open_task(board, project.id)
            if open_task is not None:
                _record_open_task(registry, project.id, open_task, now)
            if not _allowed(registry, settings, project.id):
                continue
            # The quiet clock sees the Project's news on every tick: before the review hour, while its review is open,
            # while a failed one backs off and while the cap makes it wait.
            _observe_news(registry, project, now)
            if open_task is not None or review_day is None:
                continue
            state = _state_row(registry, project.id, now)
            if not _due(state, review_day, now):
                continue
            try:
                batch = read_review_events(project, _cursors(state))
            except (sqlite3.Error, OSError, ValueError) as exc:
                logger.warning("project review: cannot read the events of Project %s (%s)", project.id, exc)
                continue
            if batch.material or batch.urgent:
                news.append(_News(project, state, batch))
            else:
                _settle_quiet(registry, project.id, state, batch, review_day, now)

        news.sort(key=lambda item: _wait_order(item, review_day))
        ensured = False
        for index, item in enumerate(news):
            if asked >= summary.MAX_NEW_PER_TICK:
                break
            if budget(registry, review_day)[0] >= settings.daily_cap:
                logger.info(
                    "project review: the daily cap of %d is reached; %d Project(s) wait", settings.daily_cap,
                    len(news) - index,
                )
                break
            if not ensured:
                if summary._ensure_summary_board() is None:
                    break
                ensured = True
            if board is None:
                board = stack.enter_context(kanban_db.connect_closing(path, board=summary.SUMMARY_BOARD))
            asked += 1
            created += _open_review_task(registry, board, item, review_day, settings.daily_cap, now)
    return created


def review_tick(now: Optional[int] = None) -> int:
    """Run one review tick and return how many review tasks it opened. Off, it reads no store at all.

    ``now`` is the clock in epoch seconds; it exists so a test can fix the Vienna day, the hour and the backoff.
    """
    now = int(time.time()) if now is None else int(now)
    settings = read_settings()
    if not settings.enabled:
        return 0
    registry_path = kanban_db.kanban_home() / "projects.db"
    if not registry_path.exists():
        return 0
    with projects_db.connect_closing(registry_path) as registry:
        _harvest(registry, settings, now)
        return _request(registry, settings, now)
