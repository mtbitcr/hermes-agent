"""The candidate a review verdict binds to when the card changed no file."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from typing import Optional

from hermes_cli import kanban_db as kb


@dataclass(frozen=True)
class RecordCandidate:
    digest: str
    handover_id: int


def record_candidate(conn: sqlite3.Connection, task_id: str) -> Optional[RecordCandidate]:
    """The newest handover as the reviewed candidate, or None unless the card has no code."""
    handover_id = kb._latest_review_handover_id(conn, task_id)
    if handover_id is None or kb._latest_review_head_provenance(conn, task_id) is not None:
        return None
    scope = kb._latest_review_scope_provenance(conn, task_id)
    # A handover that recorded no scope says nothing about files, so the head rule stands.
    if scope is None or scope["owned_paths"]:
        return None
    # The id is kernel-written and moves with every handover, so a stale digest is refused.
    return RecordCandidate(digest=f"record:{handover_id}", handover_id=handover_id)
