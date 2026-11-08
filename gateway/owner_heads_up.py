"""One Telegram heads-up to the owner when a card starts waiting for them.

Without it the owner learns of a waiting card only by opening Decisions in the app. Every 30
seconds the supervised ``_owner_heads_up_watcher`` calls :func:`tick`, which reads the owner's
Project boards read-only and sends one short plain message to the owner Telegram home chat for each
wait it has not told yet: the Project name, the question and a pointer to Decisions.

Only the process that serves the owner sends (the default profile, with the owner workspace on),
only to the root profile's Telegram home chat and only through the gateway's existing Telegram
adapter; a file lock in the root Hermes home keeps a second gateway process from sending too. The
record of the waits already told (``owner_heads_up.json``, beside ``owner_channel_pause.json``)
outlives restarts, so no wait is told twice. A wait is keyed by its Project, its card and the
``blocked`` event that started it, so a new wait after an answer gets one new message; the ids stay
in that local file and never appear in a message. A wait that starts inside an owner channel pause
window, or is found while paused, is recorded "paused" and never sent, not even after the resume,
as the kanban notifier drops a card notice. A tick tries at most MAX_PER_TICK sends, oldest wait
first, and leaves the other waits due a send for the next tick. That limit counts sends only, so a
tick records as "paused" every due wait the pause holds, however many there are.

A tick never raises and writes no kanban store, so a failure here cannot change a card, the
Decisions list or the dispatcher. A Project whose board fails any read is skipped for the tick:
nothing is sent for it and its told keys stay, also when the kernel's wait helpers answer the failed
read as a board with no waits. A send that raises, or that fails with a result that is not retryable
(as the Telegram adapter answers a timeout), is recorded "unknown" and never tried again, since the
message may have arrived; only a send that fails with a retryable result is recorded "failed" and
tried again on later ticks, three attempts in all; a "sending" entry found at the start of a tick (a
crash between the write and the send) counts as told. A record file that exists but cannot be read
stops the tick rather than counting as empty, which would tell every wait again.
"""

from __future__ import annotations

import asyncio
import json
import logging
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from gateway import owner_channel_pause
from gateway.config import Platform
from gateway.status import _release_file_lock, _try_acquire_file_lock
from hermes_cli import kanban_db, owner_design_choices, owner_workspace
from hermes_cli.config import cfg_get, load_config
from hermes_constants import get_default_hermes_root
from utils import atomic_json_write

logger = logging.getLogger(__name__)

MAX_PER_TICK = 10  # sends per tick, oldest first, the rest wait for the next tick; "paused" ones never count
MAX_MESSAGE = 4000  # last guard under Telegram's 4096-character message limit
MAX_ATTEMPTS = 3  # a send refused as retryable is tried again on later ticks while attempts < 3
RECORD_FILE_NAME = "owner_heads_up.json"
LOCK_FILE_NAME = "owner_heads_up.lock"
_STATES = ("sent", "paused", "failed", "unknown", "sending")

_DESIGN_LINES = (
    "Raphael needs you to choose a design in {project_name}.",
    "The options are waiting for you on Decisions in the app.",
)
_QUESTION_LINES = (
    "Raphael needs your answer in {project_name}.",
    "Please answer on Decisions in the app.",
)
_CANDIDATES = (
    "SELECT id FROM tasks WHERE project_id = ? AND task_kind = 'work' AND status = 'blocked' "
    "AND block_kind IN ('needs_input','capability')"
)


@dataclass(frozen=True)
class _Wait:
    """One card of an owner Project waiting for the owner, as read from its board."""

    project_id: str
    project_name: str
    task_id: str
    blocked_event_id: int
    since: int
    question: str
    design: bool

    @property
    def key(self) -> str:
        return f"{self.project_id}:{self.task_id}:{self.blocked_event_id}"


def _gates(runner) -> Optional[tuple[str, Any]]:
    """The owner home chat and the Telegram adapter when this process may tell the owner, else None.

    The same rules as ``owner_channel._serves`` (the default profile) and
    ``api_server._owner_workspace_toolset_enabled`` (the literal True, never the string "true").
    """
    if runner._active_profile_name() != "default":
        return None
    if cfg_get(load_config(), "gateway", "api_server", "owner_workspace", "enabled", default=False) is not True:
        return None
    chat = owner_channel_pause._owner_home_chat_id().strip()
    if not chat:
        return None
    adapter = runner.adapters.get(Platform.TELEGRAM)
    if adapter is None:
        return None
    return chat, adapter


def _valid_entry(entry: Any) -> bool:
    return (
        isinstance(entry, dict)
        and entry.get("state") in _STATES
        and type(entry.get("attempts")) is int
        and type(entry.get("at")) in (int, float)
    )


def _read_record(path: Path) -> Optional[dict]:
    """The told entries by key: {} when there is no file yet, None when it cannot be read or is malformed."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except (OSError, ValueError):
        return None
    told = data.get("told") if isinstance(data, dict) else None
    if not isinstance(told, dict) or not all(_valid_entry(entry) for entry in told.values()):
        return None
    return told


def _write_record(path: Path, told: dict) -> None:
    atomic_json_write(path, {"told": told})


def _entry(state: str, attempts: int) -> dict:
    return {"state": state, "attempts": attempts, "at": int(time.time())}


class _Reads:
    """A thin delegating wrapper around the board connection, or a cursor it returned. Each SQLite
    error raised by execute or while rows are fetched is kept in ``failed``, one list for the
    connection and its cursors, and then raised on, so a read the kernel's helpers swallow still
    fails the board."""

    def __init__(self, target: Any, failed: Optional[list] = None):
        self._target = target
        self.failed: list = [] if failed is None else failed

    def __getattr__(self, name: str) -> Any:
        return getattr(self._target, name)

    def _kept(self, read, *args, **kwargs):
        try:
            return read(*args, **kwargs)
        except sqlite3.Error as exc:
            self.failed.append(exc)
            raise

    def execute(self, *args, **kwargs) -> "_Reads":
        return _Reads(self._kept(self._target.execute, *args, **kwargs), self.failed)

    def fetchone(self):
        return self._kept(self._target.fetchone)

    def fetchmany(self, *args, **kwargs):
        return self._kept(self._target.fetchmany, *args, **kwargs)

    def fetchall(self):
        return self._kept(self._target.fetchall)

    def __iter__(self) -> "_Reads":
        return self

    def __next__(self):
        return self._kept(next, self._target)

    def check(self) -> None:
        """Raise the first SQLite error kept, if any read failed."""
        if self.failed:
            raise self.failed[0]


def _board_waits(project: dict) -> list[_Wait]:
    """Every card of ``project`` waiting for the owner now, read from its board without writing.

    Raises when any read of the board fails, also a read the kernel's helpers answer as no waits.
    """
    board = str(project["board"])
    folder = kanban_db.kanban_home() if board == kanban_db.DEFAULT_BOARD else kanban_db.board_dir(board)
    project_id = str(project["project_id"])
    conn = owner_workspace._open_read_only_sqlite(folder / "kanban.db", label="kanban.db")
    try:
        # One read transaction, so every query sees the same snapshot of the board. task_owner_waits
        # answers {} when a read fails: here that would pass for a board with no waits and let the prune
        # forget waits already told, and inside decision_design_options for a design wait without options,
        # told as a question. So both helpers read through _Reads, and a read error it kept fails the board.
        conn.execute("BEGIN")
        candidates = [str(row[0]) for row in conn.execute(_CANDIDATES, (project_id,))]
        reads = _Reads(conn)
        waits = kanban_db.task_owner_waits(reads, candidates)
        reads.check()
        if not waits:
            return []
        options = owner_design_choices.decision_design_options(reads, list(waits))
        reads.check()
        placeholders = ",".join("?" for _ in waits)
        starts = {
            str(task_id): int(event_id)
            for task_id, event_id in conn.execute(
                "SELECT task_id, MAX(id) FROM task_events WHERE kind = 'blocked' "
                f"AND task_id IN ({placeholders}) GROUP BY task_id",
                list(waits),
            )
        }
    finally:
        conn.close()
    return [
        _Wait(
            project_id=project_id,
            project_name=str(project["name"]),
            task_id=task_id,
            blocked_event_id=starts[task_id],
            since=int(wait["since"]),
            question=(
                str(options[task_id]["question"]) if task_id in options
                else owner_workspace.owner_stop_reason(wait["reason"])
            ),
            design=task_id in options,
        )
        for task_id, wait in waits.items()
    ]


def _waiting() -> tuple[list[_Wait], set[str]]:
    """Every wait of the owner's committed Projects, and the ids of the Projects whose board was read.

    The same Projects and names as Decisions: archived ones and boards gone with a completed removal
    are skipped, as ``list_owner_decisions`` skips them. A board that fails to read is skipped for this
    tick, and its told keys stay.
    """
    ctx = owner_workspace.OwnerContext(actor="default", profile="default", session="")
    waits: list[_Wait] = []
    read: set[str] = set()
    for project in owner_workspace.list_committed_projects(ctx):
        project_id = str(project["project_id"])
        if project["archived"] or (
            not kanban_db.board_exists(str(project["board"]))
            and owner_workspace._completed_removal_state(project_id) is not None
        ):
            continue
        try:
            waits.extend(_board_waits(project))
        except Exception as exc:
            logger.debug("Owner heads-up: a Project board could not be read this tick: %s", exc)
            continue
        read.add(project_id)
    return waits, read


def _text(wait: _Wait) -> str:
    """The plain-text heads-up. A text over MAX_MESSAGE has its question cut to fit, ending with "…";
    the Project name and the Decisions line always stay."""
    first, last = _DESIGN_LINES if wait.design else _QUESTION_LINES
    first = first.format(project_name=wait.project_name)
    question = wait.question
    room = MAX_MESSAGE - len(first) - len(last) - 2
    if len(question) > room:
        question = question[:max(0, room - 1)] + "…"
    return f"{first}\n{question}\n{last}"


def _due(entry: Optional[dict]) -> bool:
    return entry is None or (entry["state"] == "failed" and entry["attempts"] < MAX_ATTEMPTS)


async def tick(runner) -> int:
    """Tell the owner of each wait not told yet, oldest first, in at most MAX_PER_TICK sends, and record
    every one the pause holds as "paused"; return the number of messages sent. Stops quietly at the
    first closed gate and never raises."""
    sent = 0
    try:
        gates = _gates(runner)
        if gates is None:
            return 0
        chat, adapter = gates
        root = get_default_hermes_root()
        record = root / RECORD_FILE_NAME
        handle = open(root / LOCK_FILE_NAME, "a+", encoding="utf-8")
        try:
            if not _try_acquire_file_lock(handle):
                return 0
            try:
                told = _read_record(record)
                if told is None:
                    logger.warning("Owner heads-up: %s cannot be read or is malformed; nothing is sent", record)
                    return 0
                waits, read = await asyncio.to_thread(_waiting)
                due = sorted(
                    (wait for wait in waits if _due(told.get(wait.key))),
                    key=lambda wait: (wait.since, wait.blocked_event_id, wait.key),
                )
                calls = 0  # adapter.send calls this tick; a wait recorded "paused" is not one
                for wait in due:
                    attempts = told[wait.key]["attempts"] if wait.key in told else 0
                    if owner_channel_pause.drops_notice(Platform.TELEGRAM, chat, wait.since):
                        told[wait.key] = _entry("paused", attempts)
                    elif calls >= MAX_PER_TICK:
                        continue
                    else:
                        calls += 1
                        told[wait.key] = _entry("sending", attempts)
                        _write_record(record, told)
                        try:
                            result = await adapter.send(chat, _text(wait))
                        except Exception as exc:
                            logger.warning("Owner heads-up send raised; it is not tried again: %s", exc)
                            told[wait.key] = _entry("unknown", attempts)
                        else:
                            if getattr(result, "success", True) is False:
                                logger.warning("Owner heads-up send failed: %s", getattr(result, "error", None))
                                if getattr(result, "retryable", False) is True:
                                    told[wait.key] = _entry("failed", attempts + 1)
                                else:
                                    # Not retryable, such as a Telegram timeout: the message may have
                                    # arrived, so it counts as told and is never tried again.
                                    told[wait.key] = _entry("unknown", attempts)
                            else:
                                told[wait.key] = _entry("sent", attempts)
                                sent += 1
                    _write_record(record, told)
                waiting_now = {wait.key for wait in waits}
                kept = {
                    key: entry for key, entry in told.items()
                    if key in waiting_now or key.split(":", 1)[0] not in read
                }
                if len(kept) != len(told):
                    _write_record(record, kept)
            finally:
                _release_file_lock(handle)
        finally:
            handle.close()
    except Exception as exc:
        logger.debug("Owner heads-up tick error: %s", exc)
    return sent
