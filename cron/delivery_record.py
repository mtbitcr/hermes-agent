"""Profile-local record of what each scheduled report delivery really did.

Before a run's report is handed to any platform, the exact final text, its attachments and every
chat target are written here with each target ``pending``; each target's outcome is written as soon
as it is known:

* ``delivered``: the platform accepted it.
* ``failed``: the code shows nothing was sent (``route_refused``, ``no_connection``,
  ``settings_not_loaded``, or ``platform_refused`` before any part was sent and with no message id).
* ``unknown``: everything else (``timeout``, ``error_after_handover``, ``partly_sent``); a target
  still pending when read was cut off by a restart and reads ``unknown``/``interrupted``.

When in doubt the outcome is ``unknown``. Raw error text is never stored. This is an audit record,
not a retry queue: nothing here re-sends anything, and a recording error is logged and never changes
delivery.

Each attempt to send a run's failed chats again is recorded here too: claimed ``in_progress`` in one
compare-and-set write, then finished with each chat's outcome, its times and its error, redacted. A
chat an attempt leaves ``failed`` may be re-sent again; one it leaves ``delivered`` or ``unknown``
never is. An attempt still in progress at start-up was cut off and is fenced ``unknown``.
"""

from __future__ import annotations

import contextvars
import functools
import inspect
import json
import logging
import math
import os
import re
import sqlite3
import threading
import time
import uuid
from contextlib import contextmanager, suppress
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional

from cron.ledger import ledger_transaction, open_ledger, prepare_ledger
from hermes_constants import get_hermes_home

logger = logging.getLogger(__name__)

# A failed chat may be re-sent while its record is younger than this; no younger record is pruned.
RESEND_WINDOW_SECONDS = 7 * 86400
# Records past the resend window are kept only while they are among the newest MAX_RECORDS.
MAX_RECORDS = 1000
# A text larger than this is not kept, so its run reads output_expired and is never re-sent.
MAX_SAVED_TEXT_BYTES = 1024 * 1024
# A target's outcome only moves up this order: a fallback the platform accepts turns an earlier
# failure into delivered, and a send that may have happened keeps a later refusal unknown.
_RANK = {"failed": 0, "unknown": 1, "delivered": 2}
# Re-send attempts are bounded: a request id's length and the attempts one run may have.
MAX_REQUEST_ID_CHARS = 128
MAX_RESEND_ATTEMPTS = 5
# The chat states a re-send attempt writes.
_CHAT_STATES = frozenset({"in_progress", "failed", "unknown", "delivered"})
# The reason codes a re-send attempt keeps for a chat; any other reason, such as error text, is dropped.
_REASONS = frozenset({
    "route_refused", "no_connection", "settings_not_loaded", "platform_refused",
    "timeout", "error_after_handover", "partly_sent", "interrupted",
})
_lock = threading.RLock()
_active = contextvars.ContextVar("cron_delivery_recorder", default=None)


def _clock() -> float:
    return time.time()


# --- delivery record store -----------------------------------------------------------------------

def _path() -> Path:
    return get_hermes_home().resolve() / "cron" / "delivery_records.db"


def _initialize_schema(conn: sqlite3.Connection) -> None:
    prepare_ledger(conn, db_label="cron/delivery_records.db")
    conn.execute(
        """CREATE TABLE IF NOT EXISTS records (
             execution_id TEXT PRIMARY KEY,
             job_id TEXT NOT NULL,
             created_at REAL NOT NULL,
             text TEXT,
             attachments TEXT NOT NULL DEFAULT '[]'
           )"""
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_records_created "
        "ON records(created_at DESC, execution_id DESC)"
    )
    conn.execute(
        """CREATE TABLE IF NOT EXISTS targets (
             execution_id TEXT NOT NULL,
             position INTEGER NOT NULL,
             platform TEXT NOT NULL,
             chat_id TEXT NOT NULL,
             thread_id TEXT,
             state TEXT NOT NULL CHECK(state IN ('pending','delivered','failed','unknown')),
             reason TEXT,
             sent_text TEXT,
             updated_at REAL NOT NULL,
             PRIMARY KEY (execution_id, position)
           )"""
    )
    conn.execute(
        """CREATE TABLE IF NOT EXISTS attempts (
             attempt_id TEXT PRIMARY KEY,
             execution_id TEXT NOT NULL,
             request_id TEXT NOT NULL,
             requested_at REAL NOT NULL,
             finished_at REAL,
             state TEXT NOT NULL CHECK(state IN ('in_progress','delivered','failed','unknown')),
             chats TEXT NOT NULL,
             error TEXT,
             UNIQUE (execution_id, request_id)
           )"""
    )
    # At most one attempt per run is in progress, whatever writes it.
    conn.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS idx_attempts_in_progress "
        "ON attempts(execution_id) WHERE state='in_progress'"
    )


def _decoded(value: bytes) -> str:
    return value.decode("utf-8", errors="replace")


def _open(path: Path) -> sqlite3.Connection:
    conn = open_ledger(path)
    # Text that is not valid UTF-8 reads with replacement characters instead of failing the whole
    # read; the readers below then refuse it as unreadable.
    conn.text_factory = _decoded
    # The store holds report text and chat ids: owner-only, like the delivery queue and cron output.
    # SQLite gives its -wal and -shm files the main file's mode.
    for target, mode in ((path.parent, 0o700), (path, 0o600)):
        with suppress(OSError):
            os.chmod(target, mode)
    return conn


@contextmanager
def _transaction(path: Path) -> Iterator[sqlite3.Connection]:
    with ledger_transaction(_lock, lambda: _open(path), _initialize_schema) as conn:
        yield conn


def _kept(text: Optional[str]) -> Optional[str]:
    """``text`` as stored: None when it is larger than MAX_SAVED_TEXT_BYTES."""
    if text is not None and len(text.encode("utf-8")) > MAX_SAVED_TEXT_BYTES:
        return None
    return text


def _prune_unlocked(conn: sqlite3.Connection, now: float) -> None:
    conn.execute(
        "DELETE FROM records WHERE created_at < ? AND execution_id NOT IN ("
        "SELECT execution_id FROM records ORDER BY created_at DESC, execution_id DESC LIMIT ?)",
        (now - RESEND_WINDOW_SECONDS, max(0, int(MAX_RECORDS))),
    )
    conn.execute("DELETE FROM targets WHERE execution_id NOT IN (SELECT execution_id FROM records)")
    conn.execute("DELETE FROM attempts WHERE execution_id NOT IN (SELECT execution_id FROM records)")


# --- recording -----------------------------------------------------------------------------------

def _single_part(text: str, limit: Any) -> bool:
    """True when ``text`` certainly went out as one message, so a refusal of it means nothing was
    sent. UTF-8 bytes bound every length unit a platform counts in; doubling covers its escaping."""
    return (
        isinstance(limit, int) and not isinstance(limit, bool) and limit > 0
        and len((text or "").encode("utf-8")) * 2 <= limit
    )


def _live_limit(adapter: Any, chat_id: Any) -> Any:
    try:
        probe = getattr(adapter, "max_message_length_for_chat", None)
        if callable(probe):
            return probe(str(chat_id))
        return getattr(adapter, "MAX_MESSAGE_LENGTH", None)
    except Exception:
        return None


def _standalone_limit(platform: Any) -> Any:
    from gateway.config import Platform
    from tools.send_message_tool import _platform_max_length

    if platform == Platform.TELEGRAM:
        return 4096  # the standalone Telegram sender splits its formatted text at this length
    if platform == Platform.WEIXIN:
        return None  # its sender splits by rules of its own
    return _platform_max_length(platform)


def _guarded(step):
    """Run a recording step only while the recording is live; an error ends the recording with a
    warning and never reaches the delivery."""

    @functools.wraps(step)
    def run(self, *args, **kwargs):
        if not self._live:
            return None
        try:
            return step(self, *args, **kwargs)
        except Exception:
            self._live = False
            logger.warning(
                "Job '%s': delivery record could not be written; delivery is unaffected",
                self.job_id, exc_info=True,
            )
            return None

    return run


class _Recorder:
    """Writes one execution's delivery record while ``_deliver_result`` walks its targets. A failure
    stays tentative until the target is left, since a fallback send may still follow it."""

    def __init__(self, execution_id: Optional[str], job_id: Optional[str]) -> None:
        self.execution_id = str(execution_id or "")
        self.job_id = str(job_id or "")
        self._live = bool(self.execution_id)
        self._claimed = False
        self._path: Optional[Path] = None
        self._text: Optional[str] = None
        self._index = -1
        self._outcome: Optional[tuple] = None
        self._written: Optional[tuple] = None
        self._sent_text: Optional[str] = None

    def _write(self, sql: str, params: tuple) -> None:
        if self._path is None:
            return
        with _transaction(self._path) as conn:
            conn.execute(sql, params)

    def _settle(self) -> None:
        if self._index < 0 or self._outcome is None or self._outcome == self._written:
            return
        state, reason = self._outcome
        self._write(
            "UPDATE targets SET state=?, reason=?, updated_at=? WHERE execution_id=? AND position=?",
            (state, reason, _clock(), self.execution_id, self._index),
        )
        self._written = self._outcome

    def _note(self, state: str, reason: Optional[str]) -> None:
        current = self._outcome
        if current is None or _RANK[state] > _RANK[current[0]] or state == current[0] == "failed":
            self._outcome = (state, reason)

    @_guarded
    def begin(self, text: str, media_files: Iterable, targets: Iterable[dict]) -> None:
        """Save the final text, attachments and every target as pending before any hand-over."""
        self._path = _path()
        self._text = text
        now = _clock()
        attachments = [{"path": str(path), "is_voice": bool(is_voice)} for path, is_voice in media_files]
        rows = [
            (
                self.execution_id, position, str(target.get("platform") or "").lower(),
                str(target.get("chat_id") or ""),
                str(target["thread_id"]) if target.get("thread_id") else None, now,
            )
            for position, target in enumerate(targets)
        ]
        with _transaction(self._path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO records (execution_id, job_id, created_at, text, attachments) "
                "VALUES (?, ?, ?, ?, ?)",
                (self.execution_id, self.job_id, now, _kept(text), json.dumps(attachments)),
            )
            conn.execute("DELETE FROM targets WHERE execution_id=?", (self.execution_id,))
            conn.executemany(
                "INSERT INTO targets (execution_id, position, platform, chat_id, thread_id, state, "
                "updated_at) VALUES (?, ?, ?, ?, ?, 'pending', ?)",
                rows,
            )
            _prune_unlocked(conn, now)

    @_guarded
    def settings_not_loaded(self) -> None:
        self._write(
            "UPDATE targets SET state='failed', reason='settings_not_loaded', updated_at=? "
            "WHERE execution_id=? AND state='pending'",
            (_clock(), self.execution_id),
        )

    @_guarded
    def next_target(self) -> None:
        self._settle()
        self._index += 1
        self._outcome = self._written = self._sent_text = None

    @_guarded
    def handing_over(self, text: str) -> None:
        """Keep the exact text about to be handed over when it is not the record's text."""
        sent_text = None if text == self._text else text
        if sent_text is not None and _kept(sent_text) is None:
            # Too large to keep: this run can no longer be re-sent as it was handed over.
            self._write("UPDATE records SET text=NULL WHERE execution_id=?", (self.execution_id,))
            sent_text = None
        if sent_text != self._sent_text:
            self._write(
                "UPDATE targets SET sent_text=? WHERE execution_id=? AND position=?",
                (sent_text, self.execution_id, self._index),
            )
            self._sent_text = sent_text

    @_guarded
    def note_failed(self, reason: str) -> None:
        self._note("failed", reason)

    @_guarded
    def note_unknown(self, reason: str) -> None:
        self._note("unknown", reason)
        self._settle()

    @_guarded
    def note_sent(self, complete: bool) -> None:
        if complete:
            self._note("delivered", None)
        else:
            self._note("unknown", "partly_sent")
        self._settle()

    @_guarded
    def live_error(self, exc: BaseException, text: str, adapter: Any, chat_id: Any) -> None:
        """A live send raised. Only a whole-chat refusal of single-part text shows nothing was sent."""
        from gateway.dead_targets import classify_dead_error

        if classify_dead_error(str(exc)) and _single_part(text, _live_limit(adapter, chat_id)):
            self._note("failed", "platform_refused")
        else:
            self._note("unknown", "error_after_handover")
            self._settle()

    @_guarded
    def standalone_started(self, coro: Any) -> None:
        """``asyncio.run`` raised and the send is retried on a fresh thread: unless the first
        coroutine provably never started, the retry may be a second copy."""
        if not (inspect.iscoroutine(coro) and inspect.getcoroutinestate(coro) == inspect.CORO_CREATED):
            self._note("unknown", "error_after_handover")
            self._settle()

    @_guarded
    def standalone_error(self, exc: BaseException) -> None:
        self._note("unknown", "timeout" if isinstance(exc, TimeoutError) else "error_after_handover")
        self._settle()

    @_guarded
    def standalone_result(self, result: Any, text: str, platform: Any, media_files: list) -> None:
        """Classify the standalone sender's reply. Its refusal shows nothing was sent only with no
        message id, no attachment and text that went out as a single part."""
        from gateway.dead_targets import classify_dead_error

        if not isinstance(result, dict) or not (result.get("error") or result.get("success")):
            outcome = ("unknown", "error_after_handover")
        elif result.get("error"):
            refused = (
                classify_dead_error(str(result["error"]))
                and not (result.get("message_id") or result.get("message_ids") or media_files)
                and _single_part(text, _standalone_limit(platform))
            )
            outcome = ("failed", "platform_refused") if refused else ("unknown", "error_after_handover")
        elif result.get("warnings"):
            outcome = ("unknown", "partly_sent")
        else:
            outcome = ("delivered", None)
        self._note(*outcome)
        self._settle()

    @_guarded
    def finish(self) -> None:
        self._settle()


_INERT = _Recorder(None, None)


@contextmanager
def recording(execution_id: Optional[str], job_id: Optional[str]) -> Iterator[None]:
    """Record the one report delivery made inside this block under ``execution_id``."""
    recorder = _Recorder(execution_id, job_id)
    token = _active.set(recorder)
    try:
        yield
        recorder.finish()
    finally:
        _active.reset(token)


def active_recorder() -> _Recorder:
    """The open recording, handed to the first delivery inside it; an inert one otherwise."""
    recorder = _active.get()
    if recorder is None or recorder._claimed:
        return _INERT
    recorder._claimed = True
    return recorder


# --- reading -------------------------------------------------------------------------------------

def load_many(execution_ids: Iterable[Any], *, bodies: bool = True) -> Dict[str, Dict[str, Any]]:
    """This profile's records for ``execution_ids``, each with its re-send attempts oldest first; a
    target still pending, or whose stored state is not one this module writes, reads ``unknown``.
    With ``bodies=False`` each ``text`` is only True when a text is kept (None otherwise), so a
    listing never loads the report bodies."""
    ids = list(dict.fromkeys(str(value) for value in execution_ids if value))
    path = _path()
    if not ids or not path.exists():
        return {}
    with _transaction(path) as conn:
        return _load_unlocked(conn, ids, bodies)


def _load_unlocked(conn: sqlite3.Connection, ids: List[str], bodies: bool) -> Dict[str, Dict[str, Any]]:
    marks = ",".join("?" * len(ids))
    text = "text" if bodies else "CASE WHEN text IS NULL THEN NULL ELSE 1 END AS text"
    sent_text = "sent_text" if bodies else "CASE WHEN sent_text IS NULL THEN NULL ELSE 1 END AS sent_text"
    records = conn.execute(
        f"SELECT execution_id, job_id, created_at, {text}, attachments FROM records "
        f"WHERE execution_id IN ({marks})", ids,
    ).fetchall()
    targets = conn.execute(
        f"SELECT execution_id, platform, chat_id, thread_id, state, reason, {sent_text} FROM targets "
        f"WHERE execution_id IN ({marks}) ORDER BY execution_id, position", ids,
    ).fetchall()
    attempts = conn.execute(
        f"SELECT * FROM attempts WHERE execution_id IN ({marks}) ORDER BY rowid", ids
    ).fetchall()
    loaded = {
        row["execution_id"]: {
            "job_id": row["job_id"],
            "created_at": row["created_at"],
            "text": _stored_text(row["text"]) if bodies or row["text"] is None else True,
            "attachments": _stored_attachments(row["attachments"]),
            "targets": [],
            "attempts": [],
        }
        for row in records
    }
    for row in targets:
        record = loaded.get(row["execution_id"])
        if record is None:
            continue
        pending = row["state"] == "pending"
        known = row["state"] in _RANK
        sent = _stored_text(row["sent_text"]) if bodies or row["sent_text"] is None else True
        record["targets"].append({
            "platform": row["platform"],
            "chat_id": row["chat_id"],
            "thread_id": row["thread_id"],
            "state": row["state"] if known else "unknown",
            "reason": "interrupted" if pending else (row["reason"] if known else None),
            "text": sent if sent is not None else record["text"],
        })
    for row in attempts:
        record = loaded.get(row["execution_id"])
        if record is not None:
            record["attempts"].append(_attempt(row))
    return loaded


def _attempt(row: Any) -> Dict[str, Any]:
    """An attempt as stored. An attempt whose ids, state, chats or times cannot be read has chats None
    and reads unknown."""
    chats = _chats(row["chats"])
    attempt_id, request_id, state = row["attempt_id"], row["request_id"], row["state"]
    requested_at = _stored_iso(row["requested_at"])
    finished_at = None if row["finished_at"] is None else _stored_iso(row["finished_at"])
    if (
        not isinstance(attempt_id, str) or not isinstance(request_id, str) or requested_at is None
        or (row["finished_at"] is not None and finished_at is None)
        or not isinstance(state, str) or state not in _CHAT_STATES
    ):
        chats = None
    return {
        "attempt_id": attempt_id if isinstance(attempt_id, str) else None,
        "request_id": request_id if isinstance(request_id, str) else None,
        "requested_at": requested_at,
        "finished_at": finished_at,
        "state": state if chats is not None else "unknown",
        "chats": chats,
        "error": row["error"] if row["error"] in _REASONS else None,
    }


def _stored_text(text: Any) -> Optional[str]:
    """Saved report text, or None when it is not text (it then reads as expired)."""
    return text if isinstance(text, str) else None


def _stored_attachments(text: Any) -> Optional[List[Any]]:
    """The saved attachment list, or None when it cannot be read (the run is then never re-sent)."""
    try:
        attachments = json.loads(text or "[]")
    except (TypeError, ValueError, RecursionError):
        return None
    return attachments if isinstance(attachments, list) else None


def _stored_iso(moment: Any) -> Optional[str]:
    """A stored time as ISO text, or None when it is not a finite number in datetime range."""
    if isinstance(moment, bool) or not isinstance(moment, (int, float)) or not math.isfinite(moment):
        return None
    try:
        return datetime.fromtimestamp(moment, timezone.utc).isoformat()
    except (OverflowError, OSError, ValueError):
        return None


def _chats(text: Any) -> Optional[List[Dict[str, Any]]]:
    """An attempt's chats, or None when any of them is not what this module writes."""
    try:
        chats = json.loads(text)
    except (TypeError, ValueError, RecursionError):
        return None
    if not isinstance(chats, list):
        return None
    for chat in chats:
        if not isinstance(chat, dict) or set(chat) != {"position", "state", "reason"}:
            return None
        position, state, reason = chat["position"], chat["state"], chat["reason"]
        if (
            not isinstance(position, int) or isinstance(position, bool)
            or not isinstance(state, str) or state not in _CHAT_STATES
            or (reason is not None and (not isinstance(reason, str) or reason not in _REASONS))
        ):
            return None
    return chats


def load(execution_id: Any) -> Optional[Dict[str, Any]]:
    return load_many([execution_id]).get(str(execution_id))


def history_deliveries(
    rows: Iterable[Optional[Dict[str, Any]]], *, now: Any = None
) -> List[Dict[str, Any]]:
    """The ``delivery`` block of each execution row of the current profile, read in one store call.
    Labels carry platform and route names only: never an id, the text or error text."""
    rows = [row or {} for row in rows]
    if now is None:
        now = _clock()
    elif isinstance(now, datetime):
        now = now.timestamp()
    else:
        now = float(now)
    try:
        records = load_many((row.get("id") for row in rows), bodies=False)
    except Exception:
        logger.warning("Cron delivery records could not be read", exc_info=True)
        records = {}
    routes = None
    deliveries = []
    for row in rows:
        record = records.get(str(row.get("id") or ""))
        if not record or not record["targets"]:
            outcome = row.get("delivery_outcome")
            state = outcome if outcome in ("not_configured", "suppressed") else "not_recorded"
            deliveries.append({"state": state, "targets": [], "resend": _resend("not_recorded")})
            continue
        if routes is None:
            from cron.scheduler_preflight import _primary_profile_routes_for_current_home

            routes = _primary_profile_routes_for_current_home()
        targets = record["targets"]
        deliveries.append({
            "state": min((target["state"] for target in targets), key=_RANK.__getitem__),
            "targets": [
                {"label": label, "state": target["state"], "reason": target["reason"]}
                for label, target in zip(_labels(targets, routes), targets)
            ],
            "resend": _resend(_resend_refusal(row, record, now), record["attempts"]),
        })
    return deliveries


def _resend(reason: Optional[str], attempts: Iterable[Dict[str, Any]] = ()) -> Dict[str, Any]:
    listed = ("attempt_id", "request_id", "requested_at", "finished_at", "state")
    return {
        "eligible": reason is None,
        "reason": reason,
        "attempts": [{key: attempt[key] for key in listed} for attempt in attempts],
    }


def _resend_refusal(row: Dict[str, Any], record: Dict[str, Any], now: float) -> Optional[str]:
    """Why the failed chats of this run may not be re-sent, or None when they may be."""
    if row.get("status") in ("claimed", "running") or any(
        attempt["state"] == "in_progress" for attempt in record["attempts"]
    ):
        return "in_progress"
    if any(attempt["chats"] is None for attempt in record["attempts"]):
        # An attempt that cannot be read may have sent: never send again.
        return "outcome_unknown"
    targets = _latest_targets(record)
    failed = [target for target in targets if target["state"] == "failed"]
    if not failed:
        unknown = any(target["state"] == "unknown" for target in targets)
        return "outcome_unknown" if unknown else "already_delivered"
    if len(record["attempts"]) >= MAX_RESEND_ATTEMPTS:
        return "too_many_attempts"
    if record["attachments"] is None:
        return "attachment_missing"
    created = record["created_at"]
    if (
        not isinstance(created, (int, float)) or isinstance(created, bool) or not math.isfinite(created)
        or now - created >= RESEND_WINDOW_SECONDS or any(t["text"] is None for t in failed)
    ):
        return "output_expired"
    return None


def _latest_targets(record: Dict[str, Any]) -> List[Dict[str, Any]]:
    """The run's targets as its re-send attempts left them, the latest one last. An attempt changes
    only a target the run itself recorded as failed and that no attempt has moved on from since: a
    delivered, unknown or in-progress outcome is final."""
    recorded = record["targets"]
    targets = [dict(target) for target in recorded]
    for attempt in record["attempts"]:
        for chat in attempt["chats"] or ():
            position = chat["position"]
            if (
                0 <= position < len(targets) and recorded[position]["state"] == "failed"
                and targets[position]["state"] == "failed"
            ):
                targets[position].update(state=chat["state"], reason=chat["reason"])
    return targets


def _names_id(name: str, value: Any) -> bool:
    """Whether a route name shows an id, also written without its sign or separators."""
    text = str(value)
    digits = re.sub(r"\D", "", text)
    return text in name or (len(digits) >= 5 and digits in re.sub(r"\D", "", name))


def _labels(targets: List[Dict[str, Any]], routes: list) -> List[str]:
    """Platform display name plus the matching enabled route's name, never an id; repeats numbered."""
    from cron.scheduler_preflight import SharedRouteAdapters
    from hermes_cli.platforms import platform_label

    labels = []
    for target in targets:
        key = target["platform"]
        label = re.sub(r"^\W+", "", platform_label(key, default="")).strip() or key.replace("_", " ").title()
        chat = {"chat_id": target["chat_id"], "thread_id": target["thread_id"]}
        for route in routes:
            if SharedRouteAdapters({key: True}, [route]).get(key, chat) is None:
                continue
            name = str(getattr(route, "name", "") or "").strip()
            ids = (target["chat_id"], target["thread_id"], route.chat_id, route.thread_id, route.guild_id)
            if name and not any(_names_id(name, value) for value in ids if value):
                label = f"{label} ({name})"
            break
        labels.append(label)
    seen: Dict[str, int] = {}
    numbered = []
    for label in labels:
        seen[label] = seen.get(label, 0) + 1
        numbered.append(label if seen[label] == 1 else f"{label} {seen[label]}")
    return numbered


# --- re-send attempts ----------------------------------------------------------------------------

def claim_resend(row: Optional[Dict[str, Any]], request_id: Any) -> Dict[str, Any]:
    """Claim the re-send of an execution row's failed chats. Only when the re-send view calls the run
    eligible, one compare-and-set write adds an attempt whose ``chats`` are those positions of
    ``load(...)["targets"]``, each ``in_progress``. A request id this run already had returns its
    first attempt and claims nothing. Returns ``{"claimed", "reason", "attempt"}``, where ``reason``
    is the view's refusal when nothing was claimed."""
    request_id = str(request_id or "")
    if not request_id or len(request_id) > MAX_REQUEST_ID_CHARS:
        raise ValueError(f"a re-send claim needs a request id of 1 to {MAX_REQUEST_ID_CHARS} characters")
    row = row or {}
    execution_id = str(row.get("id") or "")
    path = _path()
    if not execution_id or not path.exists():
        return {"claimed": False, "reason": "not_recorded", "attempt": None}
    now = _clock()
    with _transaction(path) as conn:
        # Hold the write lock from the check to the insert, against other processes too.
        conn.execute("BEGIN IMMEDIATE")
        record = _load_unlocked(conn, [execution_id], False).get(execution_id)
        if not record or not record["targets"]:
            return {"claimed": False, "reason": "not_recorded", "attempt": None}
        for attempt in record["attempts"]:
            if attempt["request_id"] == request_id:
                return {"claimed": False, "reason": None, "attempt": _without_error(attempt)}
        reason = _resend_refusal(row, record, now)
        if reason is not None:
            return {"claimed": False, "reason": reason, "attempt": None}
        chats = [
            {"position": position, "state": "in_progress", "reason": None}
            for position, target in enumerate(_latest_targets(record)) if target["state"] == "failed"
        ]
        attempt_id = uuid.uuid4().hex
        conn.execute(
            "INSERT INTO attempts (attempt_id, execution_id, request_id, requested_at, state, chats) "
            "VALUES (?, ?, ?, ?, 'in_progress', ?)",
            (attempt_id, execution_id, request_id, now, json.dumps(chats)),
        )
        attempt = _attempt(
            conn.execute("SELECT * FROM attempts WHERE attempt_id=?", (attempt_id,)).fetchone()
        )
    return {"claimed": True, "reason": None, "attempt": _without_error(attempt)}


def _without_error(attempt: Dict[str, Any]) -> Dict[str, Any]:
    """A claim's answer never carries an attempt's error."""
    return {key: value for key, value in attempt.items() if key != "error"}


def finish_resend(attempt_id: Any, results: Iterable[Dict[str, Any]], *, error: Any = None) -> bool:
    """Record how an attempt still in progress ended: each chat's ``state`` and ``reason`` from
    ``results`` (``{"position", "state", "reason"}``), the finish time and ``error`` when it is a
    reason code. Free error text is never stored: delivery errors name the chat. A chat without a
    result reads ``unknown``/``interrupted``, a state that is not an outcome reads ``unknown``, and a
    reason that is not a code is dropped. An attempt whose chats cannot be read ends ``unknown``.
    False, writing nothing, when the attempt is not in progress."""
    given = {result.get("position"): result for result in results}
    safe_error = error if isinstance(error, str) and error in _REASONS else None
    path = _path()
    if not path.exists():
        return False
    with _transaction(path) as conn:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute(
            "SELECT chats FROM attempts WHERE attempt_id=? AND state='in_progress'", (str(attempt_id),)
        ).fetchone()
        if row is None:
            return False
        stored = _chats(row["chats"])
        if stored is None:
            cur = conn.execute(
                "UPDATE attempts SET state='unknown', finished_at=? WHERE attempt_id=? AND state='in_progress'",
                (_clock(), str(attempt_id)),
            )
            return cur.rowcount == 1
        chats = [dict(chat, **_chat_result(given.get(chat["position"]))) for chat in stored]
        state = min((chat["state"] for chat in chats), key=_RANK.__getitem__, default="unknown")
        cur = conn.execute(
            "UPDATE attempts SET state=?, finished_at=?, chats=?, error=? "
            "WHERE attempt_id=? AND state='in_progress'",
            (state, _clock(), json.dumps(chats), safe_error, str(attempt_id)),
        )
    return cur.rowcount == 1


def _chat_result(result: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if result is None:
        return {"state": "unknown", "reason": "interrupted"}
    state = result.get("state")
    if state not in _RANK:
        return {"state": "unknown", "reason": None}
    reason = result.get("reason")
    return {"state": state, "reason": reason if state != "delivered" and reason in _REASONS else None}


def fence_resends() -> int:
    """Start-up fence, to run before any re-send can be claimed: an attempt still in progress was cut
    off mid-send, so it and each of its chats read ``unknown`` (``interrupted``) and its run is never
    re-sent. Returns how many attempts were fenced."""
    path = _path()
    if not path.exists():
        return 0
    now = _clock()
    fenced = 0
    with _transaction(path) as conn:
        conn.execute("BEGIN IMMEDIATE")
        rows = conn.execute("SELECT attempt_id, chats FROM attempts WHERE state='in_progress'").fetchall()
        for row in rows:
            stored = _chats(row["chats"])
            if stored is None:
                # Unreadable chats stay as they are; the attempt reads unknown and blocks the run.
                fenced += conn.execute(
                    "UPDATE attempts SET state='unknown', finished_at=? WHERE attempt_id=? AND state='in_progress'",
                    (now, row["attempt_id"]),
                ).rowcount
                continue
            chats = [dict(chat, state="unknown", reason="interrupted") for chat in stored]
            fenced += conn.execute(
                "UPDATE attempts SET state='unknown', finished_at=?, chats=? "
                "WHERE attempt_id=? AND state='in_progress'",
                (now, json.dumps(chats), row["attempt_id"]),
            ).rowcount
    return fenced
