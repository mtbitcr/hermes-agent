"""Release record: the one growing release decision (release plan N1, first build).

Every merged change joins the one waiting batch. The owner accepts or puts off
exactly the list the Decisions page showed: the decision carries the batch
version and the digest of that list, so a merge that lands after the page was
drawn makes it stale and nothing is written. Accept freezes the batch and the
next merge opens a new decision; a put-off batch keeps growing and stays put
off (owner decision 2). There is no Reject (owner decision 8).

The store is one SQLite file at the kanban root, never in a board or a profile
home: a release is host-wide, and a dispatcher worker (``HERMES_KANBAN_DB`` and
``HERMES_KANBAN_BOARD`` pinned to its board, ``HERMES_HOME`` at a profile home)
must reach the same record. The file is created on first use.

A release runs under the platform's global pause (``hermes pause``), so the
record keeps no hold of its own. An accepted batch becomes releasing when its
release begins, with PREV and NEW stored, and ends released, failed (kept for
recovery, whose attempts are counted) or folded: a restored or refused release
puts its changes back into the one waiting decision.

Nothing calls this module yet: the merge-step call, the release command, the
Decisions page and the reminder job belong to later cards.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import re
import sqlite3
import time
from datetime import date, datetime
from pathlib import Path
from typing import Any, Iterable, Iterator, Optional
from zoneinfo import ZoneInfo

from hermes_cli.kanban_db import kanban_home
from hermes_cli.kanban_db_connect import DEFAULT_BUSY_TIMEOUT_MS, write_txn

STORE_NAME = "release_ledger.db"
# K2 takes no actor: only the owner-authenticated Decisions page may decide.
DECIDED_BY = "owner"
TIER_NOT_RECORDED = "tier not recorded"
REMINDER_HOUR = 18
VIENNA = ZoneInfo("Europe/Vienna")

_COMMIT_RE = re.compile(r"[0-9a-f]{40}")
# The owner's allowlist, compared byte for byte: exactly
# https://github.com/mtbitcr/hermes-agent/pull/<number> or
# https://github.com/mtbitcr/raphael-workspace/pull/<number>, the number a positive
# ASCII decimal without leading zeros. Every other URL is refused.
_PR_URL_RE = re.compile(
    r"https://github\.com/mtbitcr/(?:hermes-agent|raphael-workspace)/pull/[1-9][0-9]*"
)
# Card ids and page references are lowercase non-secret identifiers, as in kanban.
_IDENTIFIER_RE = re.compile(r"[a-z0-9][a-z0-9_.:-]{0,199}")
_DIGEST_RE = re.compile(r"[0-9a-f]{64}")
_RISK_TIERS = frozenset({0, 1, 2})
# A missing or unreadable tier never lowers the risk (shared contract S4).
_UNRECORDED_TIER = 2
_DECISIONS = frozenset({"accepted", "deferred"})
_WAITING_STATES = ("open", "deferred")
_WAITING_SQL = "state IN ('open', 'deferred')"
# A release that began and has not ended: releasing, or failed and kept for recovery.
_UNFINISHED_STATES = ("releasing", "failed")
_UNFINISHED_SQL = "state IN ('releasing', 'failed')"
_OUTCOMES = frozenset({"released", "restored", "refused", "failed"})
# A kanban board slug, or empty when the merge step does not name one.
_BOARD_RE = re.compile(r"(?:[a-z0-9][a-z0-9_-]{0,63})?")
# The first build's batch columns, copied as they are when its store is upgraded.
_FIRST_BATCH_COLUMNS = (
    "id, state, version, tier, created_at, decided_at, decided_by, decision_ref, shown_digest"
)

_SCHEMA = (
    """CREATE TABLE IF NOT EXISTS release_batches (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        state TEXT NOT NULL CHECK (state IN (
            'open', 'deferred', 'accepted', 'releasing', 'released', 'failed', 'folded'
        )),
        version INTEGER NOT NULL,
        tier INTEGER NOT NULL CHECK (tier IN (0, 1, 2)),
        created_at INTEGER NOT NULL,
        decided_at INTEGER,
        decided_by TEXT,
        decision_ref TEXT,
        shown_digest TEXT,
        prev TEXT,
        new TEXT,
        outcome TEXT CHECK (outcome IN ('released', 'restored', 'refused', 'failed')),
        outcome_at INTEGER,
        recovery_attempts INTEGER NOT NULL DEFAULT 0
    )""",
    """CREATE TABLE IF NOT EXISTS release_members (
        batch_id INTEGER NOT NULL REFERENCES release_batches(id),
        merge_commit TEXT NOT NULL UNIQUE,
        pr_url TEXT NOT NULL,
        reviewed_base TEXT NOT NULL,
        reviewed_head TEXT NOT NULL,
        reviewed_tree TEXT NOT NULL,
        tier INTEGER NOT NULL CHECK (tier IN (0, 1, 2)),
        tier_recorded INTEGER NOT NULL CHECK (tier_recorded IN (0, 1)),
        card_id TEXT NOT NULL,
        merged_at INTEGER NOT NULL,
        position INTEGER NOT NULL,
        title TEXT NOT NULL DEFAULT '',
        board TEXT NOT NULL DEFAULT '',
        UNIQUE (batch_id, position)
    )""",
    """CREATE TABLE IF NOT EXISTS release_events (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        batch_id INTEGER NOT NULL REFERENCES release_batches(id),
        kind TEXT NOT NULL,
        payload TEXT NOT NULL,
        created_at INTEGER NOT NULL
    )""",
    """CREATE TABLE IF NOT EXISTS release_reminders (
        vienna_date TEXT NOT NULL UNIQUE,
        recorded_at INTEGER NOT NULL
    )""",
)


def ledger_path() -> Path:
    """``<kanban root>/kanban/release_ledger.db``.

    Resolved through :func:`kanban_home`, never ``kanban_db_path()``: that
    honours ``HERMES_KANBAN_DB``, which the dispatcher pins to a worker's own
    board.
    """
    return kanban_home() / "kanban" / STORE_NAME


def connect() -> sqlite3.Connection:
    """Open the release store, creating the file and its tables on first use and
    upgrading a store the first build made.

    ``isolation_level=None`` because every write runs inside the kanban
    ``write_txn`` boundary (BEGIN IMMEDIATE with busy retries). Foreign keys
    are turned on after that first transaction: SQLite ignores the switch
    inside one, and the upgrade replaces the table the others refer to.
    """
    path = ledger_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), isolation_level=None, timeout=DEFAULT_BUSY_TIMEOUT_MS / 1000)
    conn.row_factory = sqlite3.Row
    try:
        with write_txn(conn):
            _upgrade_first_schema(conn)
            for statement in _SCHEMA:
                conn.execute(statement)
        conn.execute("PRAGMA foreign_keys=ON")
    except Exception:
        conn.close()
        raise
    return conn


def record_merge(
    conn: sqlite3.Connection,
    *,
    merge_commit: str,
    pr_url: str,
    reviewed_base: str,
    reviewed_head: str,
    reviewed_tree: str,
    tier: Any,
    card_id: str,
    title: str = "",
    board: str = "",
    on_main: bool = False,
) -> dict[str, Any]:
    """K1: add one merged change to the waiting release decision.

    For the kernel merge step, after the merge is confirmed: ``on_main`` is that
    step's confirmation, made through its GitHub transport, that
    ``merge_commit`` is on main. This function never checks main itself and does
    no input or output beyond the release store; anything but ``True`` is refused
    like a malformed input, and nothing is written. A commit already recorded
    returns its existing row and writes nothing. A tier that is not exactly 0, 1
    or 2 is stored as 2, "tier not recorded": the release never parses prose and
    never assumes a lower risk than it was given. ``title`` and ``board`` are the
    card's plain title and board slug, kept for the owner page; both stay empty
    when not given.

    Returns ``recorded`` (False for a repeat), the ``member`` row and the
    ``batch`` it belongs to, read back after the write.
    """
    _own_store(conn)
    if on_main is not True:
        raise ValueError("on_main must be True: the merge step's confirmation that merge_commit is on main")
    for name, value in (
        ("merge_commit", merge_commit),
        ("reviewed_base", reviewed_base),
        ("reviewed_head", reviewed_head),
        ("reviewed_tree", reviewed_tree),
    ):
        _require(value, _COMMIT_RE, f"{name} must be a 40-character lowercase hex commit id")
    _require(
        pr_url,
        _PR_URL_RE,
        "pr_url must be https://github.com/mtbitcr/hermes-agent/pull/<number> or "
        "https://github.com/mtbitcr/raphael-workspace/pull/<number>",
    )
    _require(card_id, _IDENTIFIER_RE, "card_id must be a lowercase kanban card id")
    if type(title) is not str:
        raise ValueError("title must be the card's plain title, as text")
    _require(board, _BOARD_RE, "board must be a kanban board slug, or empty")
    tier_recorded = type(tier) is int and tier in _RISK_TIERS
    member_tier = tier if tier_recorded else _UNRECORDED_TIER

    with write_txn(conn):
        existing = _member_row(conn, merge_commit)
        if existing is not None:
            return {
                "recorded": False,
                "member": _member(existing),
                "batch": _snapshot(conn, existing["batch_id"]),
            }
        now = int(time.time())
        waiting = conn.execute(f"SELECT id FROM release_batches WHERE {_WAITING_SQL}").fetchone()
        if waiting is None:
            batch_id = conn.execute(
                "INSERT INTO release_batches (state, version, tier, created_at) VALUES ('open', 0, ?, ?)",
                (member_tier, now),
            ).lastrowid
        else:
            batch_id = waiting["id"]
        (count,) = conn.execute(
            "SELECT COUNT(*) FROM release_members WHERE batch_id = ?", (batch_id,)
        ).fetchone()
        conn.execute(
            "INSERT INTO release_members (batch_id, merge_commit, pr_url, reviewed_base, "
            "reviewed_head, reviewed_tree, tier, tier_recorded, card_id, merged_at, position, "
            "title, board) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                batch_id, merge_commit, pr_url, reviewed_base, reviewed_head, reviewed_tree,
                member_tier, int(tier_recorded), card_id, now, count + 1, title, board,
            ),
        )
        conn.execute(
            "UPDATE release_batches SET version = version + 1, tier = MAX(tier, ?) WHERE id = ?",
            (member_tier, batch_id),
        )
        return {
            "recorded": True,
            "member": _member(_member_row(conn, merge_commit)),
            "batch": _snapshot(conn, batch_id),
        }


def decide_release(
    conn: sqlite3.Connection,
    batch_id: int,
    *,
    decision: str,
    shown_digest: str,
    expected_version: int,
    decision_ref: str,
) -> dict[str, Any]:
    """K2: the owner's Accept or Put off, for the Decisions page route only.

    ``shown_digest`` and ``expected_version`` are those of the list the page
    showed. A mismatch (a merge landed after the page was drawn), a batch that
    is already accepted, or "rejected" (owner decision 8) raises ``ValueError``
    and writes nothing, so the page redraws. Accept freezes the batch; put off
    keeps it waiting and growing, and it can still be accepted later.

    Returns the batch read back after the decision: on Accept, its frozen
    member list and digest.
    """
    _own_store(conn)
    _refuse_worker_context()
    if type(decision) is not str or decision not in _DECISIONS:
        raise ValueError("a release decision is 'accepted' or 'deferred'; there is no reject")
    for name, value in (("batch_id", batch_id), ("expected_version", expected_version)):
        if type(value) is not int or value < 0:
            raise ValueError(f"{name} must be a non-negative integer")
    _require(shown_digest, _DIGEST_RE, "shown_digest must be a lowercase sha256 digest")
    _require(decision_ref, _IDENTIFIER_RE, "decision_ref must be a lowercase non-secret page reference")

    with write_txn(conn):
        current = _snapshot(conn, batch_id)
        if current["state"] not in _WAITING_STATES:
            raise ValueError(f"release batch {batch_id} is already {current['state']}")
        if current["version"] != expected_version or current["digest"] != shown_digest:
            raise ValueError(f"release batch {batch_id} changed after the page was drawn; redraw it")
        now = int(time.time())
        conn.execute(
            "UPDATE release_batches SET state = ?, version = version + 1, decided_at = ?, "
            "decided_by = ?, decision_ref = ?, shown_digest = ? WHERE id = ?",
            (decision, now, DECIDED_BY, decision_ref, shown_digest, batch_id),
        )
        decided = _snapshot(conn, batch_id)
        payload = {
            "decision": decision,
            "version": decided["version"],
            "digest": decided["digest"],
            "members": len(decided["members"]),
            "decided_by": DECIDED_BY,
            "decision_ref": decision_ref,
        }
        conn.execute(
            "INSERT INTO release_events (batch_id, kind, payload, created_at) "
            "VALUES (?, 'release_decided', ?, ?)",
            (batch_id, json.dumps(payload, sort_keys=True), now),
        )
        return decided


def begin_release(conn: sqlite3.Connection, batch_id: int, *, prev: str, new: str) -> dict[str, Any]:
    """Mark an accepted batch releasing and store its release's two versions (S-B).

    ``prev`` is what the host runs: :func:`last_released`, or the checkout's
    head while no batch was ever released. ``new`` is the merge commit of the
    batch's last member, so the release ships the whole list the owner
    accepted. One release runs at a time: while any batch is releasing, or
    failed and not yet recovered, none begins. Anything else raises
    ``ValueError`` and writes nothing.

    Returns the batch read back, now releasing.
    """
    _own_store(conn)
    _require_batch_id(batch_id)
    for name, value in (("prev", prev), ("new", new)):
        _require(value, _COMMIT_RE, f"{name} must be a 40-character lowercase hex commit id")

    with write_txn(conn):
        current = _snapshot(conn, batch_id)
        if current["state"] != "accepted":
            raise ValueError(f"release batch {batch_id} is {current['state']}, not accepted")
        unfinished = conn.execute(f"SELECT id FROM release_batches WHERE {_UNFINISHED_SQL}").fetchone()
        if unfinished is not None:
            raise ValueError(f"release batch {unfinished['id']} has not finished its release")
        if not current["members"] or new != current["members"][-1]["merge_commit"]:
            raise ValueError("new must be the merge commit of the batch's last member")
        live = last_released(conn)
        if live is not None and prev != live:
            raise ValueError("prev must be the NEW of the last released batch")
        conn.execute(
            "UPDATE release_batches SET state = 'releasing', prev = ?, new = ? WHERE id = ?",
            (prev, new, batch_id),
        )
        _record_event(conn, batch_id, "release_began", {"prev": prev, "new": new}, int(time.time()))
        return _snapshot(conn, batch_id)


def begin_recovery(conn: sqlite3.Connection, batch_id: int) -> dict[str, Any]:
    """Count one recovery attempt of a release that stopped midway: its batch
    is still releasing (the run was killed) or failed.

    The state stays until :func:`finish_release` records the outcome, and any
    cap on attempts is the caller's. Any other batch raises ``ValueError`` and
    nothing is written. Returns the batch read back with its attempt count.
    """
    _own_store(conn)
    _require_batch_id(batch_id)
    with write_txn(conn):
        current = _snapshot(conn, batch_id)
        if current["state"] not in _UNFINISHED_STATES:
            raise ValueError(f"release batch {batch_id} is {current['state']}; nothing to recover")
        conn.execute(
            "UPDATE release_batches SET recovery_attempts = recovery_attempts + 1 WHERE id = ?",
            (batch_id,),
        )
        attempt = {"attempt": current["recovery_attempts"] + 1}
        _record_event(conn, batch_id, "recovery_began", attempt, int(time.time()))
        return _snapshot(conn, batch_id)


def finish_release(conn: sqlite3.Connection, batch_id: int, *, outcome: str) -> dict[str, Any]:
    """Record how the release of a releasing or failed batch ended (S-G).

    ``released`` is final, and :func:`last_released` then names its NEW.
    ``failed`` keeps the batch for recovery, and a later outcome replaces it.
    ``restored`` and ``refused`` fold the batch: its changes go back into the
    one waiting decision, opened if none waits, which keeps its changes in
    merge order. A failed release cannot end refused, which says the host was
    never changed. Anything else raises ``ValueError`` and writes nothing.

    Returns the batch read back; a folded one keeps its versions and outcome
    but no members.
    """
    _own_store(conn)
    _require_batch_id(batch_id)
    if type(outcome) is not str or outcome not in _OUTCOMES:
        raise ValueError("a release outcome is 'released', 'restored', 'refused' or 'failed'")

    with write_txn(conn):
        current = _snapshot(conn, batch_id)
        was = current["state"]
        if was not in _UNFINISHED_STATES or (outcome, was) == ("refused", "failed"):
            raise ValueError(f"release batch {batch_id} is {was}; it cannot end {outcome}")
        now = int(time.time())
        state = outcome if outcome in ("released", "failed") else "folded"
        conn.execute(
            "UPDATE release_batches SET state = ?, outcome = ?, outcome_at = ? WHERE id = ?",
            (state, outcome, now, batch_id),
        )
        payload: dict[str, Any] = {"outcome": outcome}
        if state == "folded":
            payload["returned_to"] = _return_changes(conn, current, now)
        _record_event(conn, batch_id, "release_finished", payload, now)
        return _snapshot(conn, batch_id)


def last_released(conn: sqlite3.Connection) -> Optional[str]:
    """NEW of the batch released last: what the host runs, and the next
    release's PREV (S-B). ``None`` while no batch was ever released; PREV is
    then the checkout's head. The order is that of the events, never of the
    clock."""
    _own_store(conn)
    row = conn.execute(
        "SELECT new FROM release_batches JOIN release_events ON batch_id = release_batches.id "
        "WHERE state = 'released' AND kind = 'release_finished' ORDER BY release_events.id DESC LIMIT 1"
    ).fetchone()
    return None if row is None else row["new"]


def list_batches(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    """Every batch, oldest first, each with its members in merge order, all read
    from one snapshot."""
    _own_store(conn)
    with _read_txn(conn):
        ids = [row["id"] for row in conn.execute("SELECT id FROM release_batches ORDER BY id").fetchall()]
        return [_snapshot(conn, batch_id) for batch_id in ids]


def reminder_due_at(now: datetime, *, waiting: bool, last_reminder_day: Optional[date]) -> bool:
    """The pure K10 rule: due from 18:00 to 18:59 Europe/Vienna, once per Vienna
    day, while a batch waits.

    The zone is fixed here, never read from ``HERMES_TIMEZONE`` or a profile's
    ``timezone``: Vienna is a whole number of hours from UTC, so an hourly job at
    minute 0 meets 18:00 Vienna in summer and winter whatever zone the host uses.
    """
    vienna = _in_vienna(now)
    return (
        bool(waiting)
        and vienna.hour == REMINDER_HOUR
        and (last_reminder_day is None or last_reminder_day < vienna.date())
    )


def reminder_due(conn: sqlite3.Connection, now: datetime) -> bool:
    """Apply :func:`reminder_due_at` to the record. An open or a put-off batch
    waits (owner decision 3); an accepted one does not."""
    _own_store(conn)
    waiting = conn.execute(f"SELECT 1 FROM release_batches WHERE {_WAITING_SQL} LIMIT 1").fetchone()
    (last_day,) = conn.execute("SELECT MAX(vienna_date) FROM release_reminders").fetchone()
    return reminder_due_at(
        now,
        waiting=waiting is not None,
        last_reminder_day=date.fromisoformat(last_day) if last_day else None,
    )


def record_reminder(conn: sqlite3.Connection, now: datetime) -> str:
    """Record that the reminder went out on ``now``'s Vienna day; returns that day."""
    _own_store(conn)
    day = _in_vienna(now).date().isoformat()
    with write_txn(conn):
        conn.execute(
            "INSERT OR IGNORE INTO release_reminders (vienna_date, recorded_at) VALUES (?, ?)",
            (day, int(now.timestamp())),
        )
    return day


def _own_store(conn: sqlite3.Connection) -> None:
    """Refuse a connection to any database other than this module's own store.

    The main database must be the file of :func:`ledger_path`, and no other
    database may be attached, so no row goes to another store. Temporary
    tables are refused too, because they shadow the store's tables. The list
    is read without the caller's row factory, and only the own path is resolved.
    """
    cursor = conn.cursor()
    cursor.row_factory = None
    databases = {row[1]: row[2] for row in cursor.execute("PRAGMA database_list")}
    own = ledger_path()
    if set(databases) != {"main"} or databases["main"] not in (str(own), os.path.realpath(own)):
        raise PermissionError("the release record works only on a connection to its own store")


def _refuse_worker_context() -> None:
    """A dispatcher-spawned worker never decides a release (the same rule as
    the owner's recommendation decisions)."""
    if os.environ.get("HERMES_KANBAN_TASK"):
        raise PermissionError("release decisions are owner-only; a kanban worker cannot decide")


def _in_vienna(now: datetime) -> datetime:
    if now.utcoffset() is None:
        raise ValueError("now must be timezone-aware")
    return now.astimezone(VIENNA)


def _require(value: Any, pattern: re.Pattern[str], message: str) -> None:
    # Exactly str: SQLite can store another value for a subclass of str.
    if type(value) is not str or not pattern.fullmatch(value):
        raise ValueError(message)


def _require_batch_id(batch_id: Any) -> None:
    if type(batch_id) is not int or batch_id < 0:
        raise ValueError("batch_id must be a non-negative integer")


def _record_event(
    conn: sqlite3.Connection, batch_id: int, kind: str, payload: dict[str, Any], now: int
) -> None:
    conn.execute(
        "INSERT INTO release_events (batch_id, kind, payload, created_at) VALUES (?, ?, ?, ?)",
        (batch_id, kind, json.dumps(payload, sort_keys=True), now),
    )


def _return_changes(conn: sqlite3.Connection, folded: dict[str, Any], now: int) -> int:
    """Move a folded batch's changes into the one waiting batch, opening one if
    none waits, and keep that batch's changes in merge order; returns its id.

    The waiting batch's version rises, so a decision on the list shown before
    is stale, and a put-off batch stays put off (owner decision 2).
    """
    waiting = conn.execute(f"SELECT id FROM release_batches WHERE {_WAITING_SQL}").fetchone()
    if waiting is None:
        target = conn.execute(
            "INSERT INTO release_batches (state, version, tier, created_at) VALUES ('open', 0, ?, ?)",
            (folded["tier"], now),
        ).lastrowid
    else:
        target = waiting["id"]
    # Merge order is the order the changes were recorded in: their rowid order.
    # UNIQUE (batch_id, position) is checked row by row, so the positions pass
    # through distinct negative values first.
    conn.execute(
        "UPDATE release_members SET batch_id = ?, position = -rowid WHERE batch_id IN (?, ?)",
        (target, target, folded["batch_id"]),
    )
    conn.execute(
        "UPDATE release_members SET position = (SELECT COUNT(*) FROM release_members AS earlier "
        "WHERE earlier.batch_id = ? AND earlier.rowid <= release_members.rowid) WHERE batch_id = ?",
        (target, target),
    )
    conn.execute(
        "UPDATE release_batches SET version = version + 1, tier = MAX(tier, ?) WHERE id = ?",
        (folded["tier"], target),
    )
    return target


def _digest(commits: Iterable[str]) -> str:
    """SHA-256 over the ordered member commits, one per line."""
    return hashlib.sha256("".join(f"{commit}\n" for commit in commits).encode("ascii")).hexdigest()


def _member_row(conn: sqlite3.Connection, merge_commit: str) -> Optional[sqlite3.Row]:
    return conn.execute(
        "SELECT * FROM release_members WHERE merge_commit = ?", (merge_commit,)
    ).fetchone()


def _member(row: sqlite3.Row) -> dict[str, Any]:
    member = dict(row)
    member["tier_recorded"] = bool(member["tier_recorded"])
    member["tier_note"] = None if member["tier_recorded"] else TIER_NOT_RECORDED
    return member


@contextlib.contextmanager
def _read_txn(conn: sqlite3.Connection) -> Iterator[None]:
    """Run a readback's reads in one SQLite read transaction, so they see one
    snapshot. Inside a transaction the caller already opened (``record_merge``
    and ``decide_release`` read back inside their ``write_txn``) the reads join
    it: this never commits, rolls back or fails there."""
    if conn.in_transaction:
        yield
        return
    conn.execute("BEGIN")
    try:
        yield
    finally:
        if conn.in_transaction:
            conn.execute("ROLLBACK")  # a read snapshot keeps nothing


def _snapshot(conn: sqlite3.Connection, batch_id: int) -> dict[str, Any]:
    """The batch readback: metadata, ordered members and digest from one snapshot."""
    with _read_txn(conn):
        row = conn.execute("SELECT * FROM release_batches WHERE id = ?", (batch_id,)).fetchone()
        if row is None:
            raise ValueError(f"release batch {batch_id!r} not found")
        members = [
            _member(member)
            for member in conn.execute(
                "SELECT * FROM release_members WHERE batch_id = ? ORDER BY position", (batch_id,)
            ).fetchall()
        ]
    return {
        "batch_id": row["id"],
        "state": row["state"],
        "version": row["version"],
        "tier": row["tier"],
        "digest": _digest(member["merge_commit"] for member in members),
        "members": members,
        "created_at": row["created_at"],
        "decided_at": row["decided_at"],
        "decided_by": row["decided_by"],
        "decision_ref": row["decision_ref"],
        "shown_digest": row["shown_digest"],
        "prev": row["prev"],
        "new": row["new"],
        "outcome": row["outcome"],
        "outcome_at": row["outcome_at"],
        "recovery_attempts": row["recovery_attempts"],
    }


def _upgrade_first_schema(conn: sqlite3.Connection) -> None:
    """Bring a store the first build made up to this schema, keeping every row.

    SQLite cannot change a CHECK, so the batches are copied aside and their
    table is made anew under the name the members and events refer to. It is
    never renamed: a rename opens the temporary database, which
    :func:`_own_store` refuses. Runs inside :func:`connect`'s first
    transaction, before foreign keys are on.
    """
    columns = {row["name"] for row in conn.execute("PRAGMA table_info(release_batches)")}
    if not columns or "outcome" in columns:
        return
    conn.execute("CREATE TABLE release_batches_first AS SELECT * FROM release_batches")
    conn.execute("DROP TABLE release_batches")
    conn.execute(_SCHEMA[0])
    conn.execute(
        f"INSERT INTO release_batches ({_FIRST_BATCH_COLUMNS}) "
        f"SELECT {_FIRST_BATCH_COLUMNS} FROM release_batches_first"
    )
    conn.execute("DROP TABLE release_batches_first")
    for column in ("title", "board"):
        conn.execute(f"ALTER TABLE release_members ADD COLUMN {column} TEXT NOT NULL DEFAULT ''")
