"""Shared content evidence for daily review and checked suggestion closures."""

from __future__ import annotations

import re
import sqlite3
import time
from datetime import date, datetime, timedelta
from typing import Optional
from zoneinfo import ZoneInfo

from hermes_cli.owner_text import check_owner_text

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS recommendation_closures (
    recommendation_id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL,
    card_id TEXT NOT NULL,
    card_state TEXT NOT NULL CHECK (card_state IN ('done','merged')),
    reason TEXT NOT NULL,
    closed_by TEXT NOT NULL,
    review_date TEXT NOT NULL,
    closed_at INTEGER NOT NULL,
    undone_at INTEGER,
    undone_by TEXT,
    undo_reason TEXT
);
CREATE INDEX IF NOT EXISTS idx_recommendation_closures_project
    ON recommendation_closures(project_id,closed_at DESC);
CREATE TABLE IF NOT EXISTS recommendation_made_from (
    card_id TEXT PRIMARY KEY,
    recommendation_id TEXT NOT NULL,
    recorded_at INTEGER NOT NULL
);
"""

_VIENNA = ZoneInfo("Europe/Vienna")
# The operator addendum permits three closures per project and Vienna day.
_DAILY_CLOSURE_CAP = 3


def same_review_work(left: Optional[str], right: Optional[str]) -> bool:
    # Earlier stores can hold a recommendation without a display label.
    if not isinstance(left, str) or not isinstance(right, str):
        return False
    ignored = {"a", "an", "the", "to", "of", "for", "and"}
    left_words = set(re.findall(r"\w+", left.casefold())) - ignored
    right_words = set(re.findall(r"\w+", right.casefold())) - ignored
    # The approved content-link threshold is token-set Jaccard >= 0.8.
    return bool(left_words and right_words and len(left_words & right_words) * 5 >= len(left_words | right_words) * 4)


def record_made_from(conn: sqlite3.Connection, card_id: str, recommendation_id: str) -> None:
    """The kernel's lineage of a card it creates from a suggestion, in the transaction that inserts the card."""
    card = conn.execute("SELECT project_id,task_kind FROM tasks WHERE id=?", (card_id,)).fetchone()
    source = conn.execute(
        "SELECT project_id FROM tasks WHERE id=? AND task_kind='recommendation'", (recommendation_id,),
    ).fetchone() if isinstance(recommendation_id, str) else None
    if (card is None or card["task_kind"] != "work" or not card["project_id"] or source is None
            or source["project_id"] != card["project_id"]):
        raise ValueError("made_from_recommendation must name a suggestion of the card's Project")
    conn.execute(
        "INSERT INTO recommendation_made_from (card_id,recommendation_id,recorded_at) VALUES (?,?,?)",
        (card_id, recommendation_id, int(time.time())),
    )


def content_link(conn: sqlite3.Connection, recommendation_id: str, label: Optional[str], card_id: str,
                 title: Optional[str]) -> bool:
    """L1: a title similarity of at least 0.8, or the card was made from the suggestion by the kernel."""
    if same_review_work(label, title):
        return True
    # Older boards have no lineage table: no card there was made from a suggestion.
    return conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='recommendation_made_from'"
    ).fetchone() is not None and conn.execute(
        "SELECT 1 FROM recommendation_made_from WHERE card_id=? AND recommendation_id=?", (card_id, recommendation_id),
    ).fetchone() is not None


def has_live_recommendation_closure(conn: sqlite3.Connection, recommendation_id: str) -> bool:
    # Older boards keep their decision path before the additive schema upgrade.
    if conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='recommendation_closures'"
    ).fetchone() is None:
        return False
    return conn.execute(
        "SELECT 1 FROM recommendation_closures WHERE recommendation_id=? AND undone_at IS NULL",
        (recommendation_id,),
    ).fetchone() is not None


def delivery_after_undo(conn: sqlite3.Connection, recommendation_id: str, card_id: str) -> bool:
    undo_event = conn.execute(
        "SELECT MAX(id) FROM task_events WHERE task_id=? AND kind='recommendation_closure_undone'",
        (recommendation_id,),
    ).fetchone()[0]
    # Event order preserves an owner's undo even when delivery and undo share a timestamp.
    return undo_event is not None and conn.execute(
        "SELECT 1 FROM task_events WHERE task_id=? AND kind IN ('completed','delivery_merged') "
        "AND id>? LIMIT 1", (card_id, undo_event),
    ).fetchone() is not None


def remaining_closures_today(conn: sqlite3.Connection, project_id: str, now: int) -> int:
    today = datetime.fromtimestamp(now, _VIENNA).date()
    start = int(datetime.combine(today, datetime.min.time(), tzinfo=_VIENNA).timestamp())
    end = int(datetime.combine(today + timedelta(days=1), datetime.min.time(), tzinfo=_VIENNA).timestamp())
    # Audit events keep earlier closes countable after an undo or a later replacement of the closure row.
    used = conn.execute(
        "SELECT COUNT(*) FROM task_events e JOIN tasks t ON t.id=e.task_id "
        "WHERE t.project_id=? AND e.kind='recommendation_closed' AND e.created_at>=? AND e.created_at<?",
        (project_id, start, end),
    ).fetchone()[0]
    return max(0, _DAILY_CLOSURE_CAP - used)


def close_recommendation_delivered(
    conn: sqlite3.Connection, recommendation_id: str, card_id: str, *, reason: str,
    review_date: str, expected_lifecycle_version: int,
) -> dict:
    """Record checked delivery separately from the owner's recommendation decision."""
    from hermes_cli import kanban_db as kb
    from hermes_cli.owner_project_review_record import card_delivery_state

    kb._assert_recommendation_lifecycle_operator()
    if (isinstance(expected_lifecycle_version, bool) or not isinstance(expected_lifecycle_version, int)
            or expected_lifecycle_version < 0):
        raise ValueError("expected_lifecycle_version must be a non-negative integer")
    try:
        if not isinstance(review_date, str) or date.fromisoformat(review_date).isoformat() != review_date:
            raise ValueError
    except ValueError:
        raise ValueError("review_closure_date_invalid") from None
    if (not isinstance(reason, str) or not reason.strip() or len(reason) > 300
            or check_owner_text(reason, "planner_prose") != reason):
        raise ValueError("review_closure_reason_invalid")

    # The harvest runs these SQL-only writes in one Project board transaction, after the texts committed.
    with kb.write_txn(conn, allow_nested=True):
        row = kb._recommendation_row(conn, recommendation_id)
        snapshot = kb.recommendation_lifecycle_snapshot(row)
        if snapshot["lifecycle_version"] != expected_lifecycle_version:
            raise ValueError("recommendation lifecycle version mismatch")
        if snapshot["decision"] != "pending" or row["status"] != "review":
            raise ValueError("not_waiting")
        project_id = row["project_id"]
        card = conn.execute(
            "SELECT title,status FROM tasks WHERE id=? AND project_id=? AND task_kind='work'",
            (card_id, project_id),
        ).fetchone() if project_id else None
        if card is None:
            raise ValueError("review_closure_card_project_invalid")
        # A merge proves delivery only for a card that is done or archived, not for one that is open again.
        state = card_delivery_state(conn, card_id, card["status"])
        if state is None or not content_link(conn, recommendation_id, row["recommendation_label"], card_id, card["title"]):
            raise ValueError("review_closure_delivery_required")
        previous = conn.execute(
            "SELECT * FROM recommendation_closures WHERE recommendation_id=?", (recommendation_id,),
        ).fetchone()
        if previous is not None and previous["undone_at"] is None:
            if all(previous[key] == value for key, value in (
                ("project_id", project_id), ("card_id", card_id), ("card_state", state),
                ("reason", reason), ("review_date", review_date),
            )):
                return {"status": "already_closed", "recommendation_id": recommendation_id,
                        "closed_at": previous["closed_at"]}
            raise ValueError("not_waiting")
        if previous is not None and not delivery_after_undo(conn, recommendation_id, card_id):
            raise ValueError("review_closure_new_delivery_required")
        now = int(time.time())
        # The existing Undo contract uses integer seconds as its token. Never reuse a prior closure token.
        if previous is not None and now <= previous["closed_at"]:
            raise ValueError("review_closure_token_not_advanced")
        today = datetime.fromtimestamp(now, _VIENNA).date()
        if review_date > today.isoformat():
            raise ValueError("review_closure_date_invalid")
        if remaining_closures_today(conn, project_id, now) <= 0:
            raise ValueError("review_closure_daily_cap")
        conn.execute(
            "INSERT INTO recommendation_closures "
            "(recommendation_id,project_id,card_id,card_state,reason,closed_by,review_date,closed_at) "
            "VALUES (?,?,?,?,?,'project-review',?,?) ON CONFLICT(recommendation_id) DO UPDATE SET "
            "project_id=excluded.project_id,card_id=excluded.card_id,card_state=excluded.card_state,"
            "reason=excluded.reason,closed_by=excluded.closed_by,review_date=excluded.review_date,"
            "closed_at=excluded.closed_at,undone_at=NULL,undone_by=NULL,undo_reason=NULL",
            (recommendation_id, project_id, card_id, state, reason, review_date, now),
        )
        kb._append_event(conn, recommendation_id, "recommendation_closed", {
            "card_id": card_id, "card_state": state, "review_date": review_date, "reason": reason,
        })
    return {"status": "closed", "recommendation_id": recommendation_id, "closed_at": now}


def undo_recommendation_closure(
    conn: sqlite3.Connection, recommendation_id: str, *, actor: str, reason: str, expected_closed_at: int,
) -> dict:
    """Undo only the closure the owner saw; preserve every audit event."""
    from hermes_cli import kanban_db as kb

    kb._assert_recommendation_lifecycle_operator()
    if isinstance(expected_closed_at, bool) or not isinstance(expected_closed_at, int) or expected_closed_at < 0:
        raise ValueError("expected_closed_at must be a non-negative integer")
    actor = kb._bounded_lifecycle_text(actor, name="actor")
    # The existing owner decision route permits reasons of at most 500 characters.
    if (not isinstance(reason, str) or not reason.strip() or len(reason) > 500
            or check_owner_text(reason, "planner_prose") != reason):
        raise ValueError("review_closure_reason_invalid")
    with kb.write_txn(conn):
        row = kb._recommendation_row(conn, recommendation_id)
        if ((row["recommendation_decision"] or "pending") != "pending" or row["status"] != "review"
                or not has_live_recommendation_closure(conn, recommendation_id)):
            return {"status": "not_waiting", "recommendation_id": recommendation_id}
        now = int(time.time())
        changed = conn.execute(
            "UPDATE recommendation_closures SET undone_at=?,undone_by=?,undo_reason=? "
            "WHERE recommendation_id=? AND closed_at=? AND undone_at IS NULL",
            (now, actor, reason, recommendation_id, expected_closed_at),
        ).rowcount
        if changed != 1:
            return {"status": "not_waiting", "recommendation_id": recommendation_id}
        kb._append_event(conn, recommendation_id, "recommendation_closure_undone", {
            "closed_at": expected_closed_at, "actor": actor, "reason": reason,
        })
    return {"status": "undone", "recommendation_id": recommendation_id}
