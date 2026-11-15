"""The owner line of the daily review, the plan approvals it counts, the quiet brief's texts and the first-review rule.

The kernel names each open owner step from records only: an owner wait and a pending suggestion on the Project board,
and a plan change of the Project chat that waits for the owner's approval. Worker text never adds one. The approvals
are read from the root response store, read-only and without the store's constructor. These parts live here so the
review facade does not grow, and a later review reuses the same owner line.
"""

from __future__ import annotations

import json
import re
import sqlite3
import threading
from pathlib import Path
from typing import Any, Mapping, Optional

from hermes_cli import owner_project_summary_events as events
from hermes_constants import get_default_hermes_root

QUIET = "Nothing changed since the last review."
FIRST = "Nothing has happened in this Project yet."
FOLLOW = "Follow the current plan."
WAITS = "The work waits for you: see what needs you."
# The count frozen into a review task's body next to its two cursors, outside the record, so the total cap cannot cut it.
_APPROVALS = re.compile(r"^Plan approvals waiting: ([0-9]+)$", re.MULTILINE)
_EARLIER_REVIEW = re.compile(r"(?i)\b(?:last|previous|earlier|prior)\s+review")


def owner_projection(conn: sqlite3.Connection, path: Path) -> Any:
    """The response store's owner reads over a read-only connection. The constructor migrates the schema and changes
    file permissions, so it never runs here."""
    from gateway.platforms.api_server import ResponseStore

    store = ResponseStore.__new__(ResponseStore)
    store._conn, store._default_profile, store._db_path = conn, "default", str(path)
    store._conversation_lock = threading.RLock()
    return store


def plan_approvals_waiting(registry: sqlite3.Connection, project_id: str) -> int:
    """The Project's linked chats whose plan change proposal waits for the owner's approval: outstanding, not applied,
    not claimed and in an open conversation. A chat that cannot be read raises ValueError, so no review says that
    nothing needs the owner by mistake."""
    if registry.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='project_conversation_links'",
    ).fetchone() is None:
        return 0
    links = [row[0] for row in registry.execute(
        "SELECT conversation FROM project_conversation_links WHERE project_id=? ORDER BY linked_at,conversation",
        (project_id,),
    )]
    if not links:
        return 0
    path = get_default_hermes_root() / "response_store.db"
    if not path.exists():
        raise ValueError("linked_conversation_unavailable")
    from gateway.platforms.api_server import OwnerAuthorityBroken

    waiting = 0
    with events._read_only(path) as conn:
        store = owner_projection(conn, path)
        for name in links:
            try:
                snapshot = store.owner_history_snapshot(name, profile="default")
            except OwnerAuthorityBroken:
                raise ValueError("linked_conversation_unavailable") from None
            # A released claim waits again; one whose change was applied reads as consumed.
            waiting += snapshot["latest_response_id"] is not None and not (
                snapshot["proposal_consumed"] or snapshot["proposal_claimed"] or snapshot["conversation_closed"]
            )
    return waiting


def owner_line(waiting: int, pending: int, approvals: int) -> Optional[str]:
    """needs_you and the summary's owner_decision: the Decisions part, then the approval part; None with no owner step."""
    if waiting and pending:
        decisions = f"{waiting + pending} items are waiting on Decisions."
    elif pending:
        decisions = "A suggestion is waiting on Decisions." if pending == 1 else f"{pending} suggestions are waiting on Decisions."
    elif waiting:
        decisions = "An answer is waiting on Decisions." if waiting == 1 else f"{waiting} answers are waiting on Decisions."
    else:
        decisions = None
    approval = None
    if approvals:
        approval = ("A plan change waits for your approval in the Project chat." if approvals == 1
                    else f"{approvals} plan changes wait for your approval in the Project chat.")
    return " ".join(part for part in (decisions, approval) if part) or None


def quiet_texts(first: bool, line: Optional[str]) -> dict[str, Optional[str]]:
    """The quiet brief's fixed texts, by the first review and by an open owner step."""
    headline = FIRST if first else QUIET
    return {"headline": headline, "what_changed": headline, "next_step": FOLLOW if line is None else WAITS,
            "needs_you": line}


def frozen_plan_approvals(body: Optional[str]) -> Optional[int]:
    """The plan approvals frozen into a review task's body; None for a task made before the count was frozen."""
    found = _APPROVALS.search(body or "")
    return None if found is None else int(found[1])


def first_review(body: Optional[str]) -> bool:
    """Whether a review task's days start from the start: the Project's first review."""
    from hermes_cli.owner_project_review import review_window

    window = review_window(body)
    return window is not None and window[0] is None


def refuse_earlier_review(refreshed: Mapping[str, Any], brief: Mapping[str, Any], refused: type) -> None:
    """A first review's summary and brief never refer to an earlier review. ``refused`` is the facade's refusal."""
    fields = [
        {"field": f"{part}.{key}", "rule": "earlier_review"}
        for part, texts in (("summary", refreshed), ("brief", brief))
        for key, text in texts.items() if isinstance(text, str) and _EARLIER_REVIEW.search(text)
    ]
    if fields:
        raise refused("review_first_review_wording", {"fields": fields})


def fence_plan_approvals(registry: sqlite3.Connection, project_id: str, frozen: Optional[int]) -> None:
    """The harvest's live read inside its protected section: approvals that differ from the frozen count make the
    result stale. A task made before the count was frozen skips the check."""
    from hermes_cli.owner_project_review import ReviewRefused, ReviewStale

    if frozen is None:
        return
    try:
        live = plan_approvals_waiting(registry, project_id)
    except ValueError as exc:
        # A chat that cannot be read refuses the result, as a damaged board does; no brief leaves an approval out.
        raise ReviewRefused("review_decision_state_unavailable") from exc
    if live != frozen:
        raise ReviewStale("review_decision_state_changed")


def quiet_brief(registry: sqlite3.Connection, project: Any, first: bool) -> dict[str, Optional[str]]:
    """The quiet brief's texts, read inside the caller's protected section: the waits and the pending suggestions of
    the Project board, read-only, less each suggestion the latest real review settled, and the plan approvals."""
    from hermes_cli import owner_project_review as review

    waiting = pending = 0
    path = events._board_path(project.board_slug) if project.board_slug else None
    if path is not None and path.exists():
        with events._read_only(path) as board:
            decisions = review._owner_decisions(board, project.id)
        waiting = len(decisions["waits"])
        # A quiet day has no news, so the verdict of the last review that was not quiet still holds.
        row = registry.execute(
            "SELECT classification FROM project_review_briefs WHERE project_id=? AND quiet=0 "
            "ORDER BY review_date DESC LIMIT 1", (project.id,),
        ).fetchone()
        classification = [] if row is None else json.loads(row[0])
        if not isinstance(classification, list):
            raise ValueError("the last review classification is not a list")
        settled = {item.get("item") for item in classification if review._settled_class(item)}
        pending = sum(suggestion[0] not in settled for suggestion in decisions["suggestions"])
    return quiet_texts(first, owner_line(waiting, pending, plan_approvals_waiting(registry, project.id)))
