"""Reassignment cannot turn required-review work into its own reviewer.

``create_task`` refuses to attach a committed review requirement to a
read-only reviewer card, because such a card IS the independent audit and
would otherwise become its own reviewer. That refusal is worth nothing if the
same state can be reached one call later: assign implementation work that
carries ``requires_review`` to the read-only reviewer profile, and the row ends
up ``assignee=<reviewer>``, ``owned_paths=[]``, ``requires_review=true`` —
exactly the shape creation calls impossible, whose handover then parks with
implementer == reviewer.

The refusal is narrow on purpose, and both edges are asserted here:

* it applies while the card is IMPLEMENTATION work (carries the requirement
  and is not on the genuine review lane); and
* the committed requirement is never downgraded to make the assignment legal —
  refusing is the answer, silently clearing an owner's committed specification
  is not.

The two things that must keep working are asserted alongside it: explicit
assignment on the genuine ``review`` lane, and the pre-existing read-only audit
review card that carries no requirement at all.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb


READ_ONLY_REVIEWER = "raphael-verifier"


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb._INITIALIZED_PATHS.clear()
    kb.init_db()
    return home


def _events(conn, tid, kind) -> list:
    return [
        json.loads(r["payload"]) if r["payload"] else None
        for r in conn.execute(
            "SELECT payload FROM task_events WHERE task_id = ? AND kind = ? "
            "ORDER BY id",
            (tid, kind),
        ).fetchall()
    ]


def _hand_over(conn, task_id, *, claimer="worker:1", summary="ready for review"):
    """The REAL implementer handover: claim the task, then complete it."""
    run = kb.claim_task(conn, task_id, claimer=claimer)
    assert run is not None
    return kb.complete_task(
        conn, task_id, summary=summary, expected_run_id=run.current_run_id,
    )


def test_assigning_required_review_work_to_the_reviewer_is_refused(kanban_home):
    """The PUBLIC assign-then-handover bypass, end to end."""
    with kb.connect() as conn:
        tid = kb.create_task(
            conn, title="implementation work", assignee="worker",
            requires_review=True,
        )
        before = kb.get_task(conn, tid)
        assert before.requires_review is True

        with pytest.raises(RuntimeError, match="requires_review"):
            kb.assign_task(conn, tid, READ_ONLY_REVIEWER)

        after = kb.get_task(conn, tid)
        assert after.assignee == "worker"
        assert after.requires_review is True, (
            "the committed requirement must never be cleared to make the "
            "assignment legal"
        )
        assert after.owned_paths == before.owned_paths
        assert after.integrates_parent_heads == before.integrates_parent_heads
        assert _events(conn, tid, "assigned") == []

        # And the handover that follows still parks under an INDEPENDENT
        # reviewer rather than under the implementer itself.
        assert _hand_over(conn, tid) is True
        parked = kb.get_task(conn, tid)
        assert parked.status == "review"
        requested = _events(conn, tid, "review_requested")
        assert len(requested) == 1
        assert requested[0]["implementer"] == "worker"
        assert requested[0]["reviewer"] != requested[0]["implementer"], (
            "a parked card whose reviewer is its implementer is self-review"
        )
        assert parked.assignee != "worker"


def test_the_genuine_review_lane_still_accepts_an_explicit_reviewer(kanban_home):
    """REGRESSION: the refusal must not reach the real review handoff."""
    with kb.connect() as conn:
        tid = kb.create_task(
            conn, title="parked work", assignee="worker", requires_review=True,
        )
        assert _hand_over(conn, tid) is True
        assert kb.get_task(conn, tid).status == "review"

        # A human re-routing the review to the read-only reviewer while the
        # card sits on the genuine review lane is the supported action.
        assert kb.assign_task(conn, tid, READ_ONLY_REVIEWER) is True
        routed = kb.get_task(conn, tid)
        assert routed.assignee == READ_ONLY_REVIEWER
        assert routed.status == "review"
        assert routed.requires_review is True


def test_the_read_only_audit_review_card_can_still_be_assigned(kanban_home):
    """REGRESSION: work with no committed requirement is untouched.

    Reassigning ordinary work to the read-only reviewer profile still
    succeeds and still converts the scope to read-only.
    """
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="audit this", assignee="worker")
        assert kb.get_task(conn, tid).requires_review is False

        assert kb.assign_task(conn, tid, READ_ONLY_REVIEWER) is True
        audit = kb.get_task(conn, tid)
        assert audit.assignee == READ_ONLY_REVIEWER
        assert audit.owned_paths == []
        assert audit.integrates_parent_heads is False
