"""A review is always handed to someone OTHER than the work's implementer.

``_review_park_target`` already refuses to park a card under the profile that
wrote the code — but only for the handover it resolves itself. Two doors were
still open onto the same state:

* a review request that NAMES the implementer as the reviewer. The role
  transition sees the same profile going in as the one already there, treats it
  as a no-op reassignment and authorizes nothing, so the card lands on the
  review lane still assigned to its implementer — which the review dispatcher
  then re-claims for that same profile as its own reviewer; and
* a reassignment made while the card sits on the review lane, which was
  admitted for ANY replacement as long as the current holder was the reviewer
  the policy nominates — the implementer included.

Both are refused here, and the legitimate handover and reviewer swap are
asserted alongside them so the refusal cannot be a blanket one. Returning the
work TO its implementer stays exactly as supported as it was: that is what
``changes_requested`` and :func:`reopen_review_task` are for, and neither goes
through these two doors.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from plugins.dashboard_auth.raphael_workspace import model_policy


IMPLEMENTER = "worker"
REVIEWER = "raphael-verifier"
OTHER_REVIEWER = "raphael-planner"


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


def _nominate(monkeypatch, *profiles):
    """Make the model policy nominate exactly these reviewer roles."""
    monkeypatch.setattr(
        model_policy, "reviewer_profile_ids", lambda: tuple(profiles), raising=True,
    )


def _claimed_implementation(conn, *, title="implementation work"):
    """A card carrying the committed review requirement, mid-implementation."""
    tid = kb.create_task(
        conn, title=title, assignee=IMPLEMENTER, requires_review=True,
    )
    run = kb.claim_task(conn, tid, claimer=f"{IMPLEMENTER}:1")
    assert run is not None
    return tid, run


def test_review_request_naming_the_implementer_as_reviewer_is_refused(
    kanban_home, monkeypatch,
):
    """The same-assignee fast path is not a way onto the review lane."""
    _nominate(monkeypatch, REVIEWER)
    with kb.connect() as conn:
        tid, run = _claimed_implementation(conn)

        ok, reason = kb.request_review(
            conn, tid,
            summary="ready",
            reviewer=IMPLEMENTER,
            expected_run_id=run.current_run_id,
            with_reason=True,
        )

        assert ok is False
        assert "implementer" in (reason or "")
        # Nothing moved: no lane change, no handover event, no review to claim.
        after = kb.get_task(conn, tid)
        assert after.status == "running"
        assert after.assignee == IMPLEMENTER
        assert _events(conn, tid, "review_requested") == []
        assert kb.claim_review_task(conn, tid, claimer=f"{IMPLEMENTER}:2") is None

        # The genuine handover from the same state is untouched: this refuses
        # self-review, not review.
        assert kb.request_review(
            conn, tid,
            summary="ready",
            reviewer=REVIEWER,
            expected_run_id=run.current_run_id,
        ) is True
        parked = kb.get_task(conn, tid)
        assert (parked.status, parked.assignee) == ("review", REVIEWER)


def test_review_request_must_name_the_policys_current_reviewer(
    kanban_home, monkeypatch,
):
    """A committed requirement promises INDEPENDENT review, so who reviews is
    the policy's answer — not the caller's."""
    _nominate(monkeypatch, REVIEWER)
    with kb.connect() as conn:
        tid, run = _claimed_implementation(conn)

        ok, reason = kb.request_review(
            conn, tid,
            summary="ready",
            reviewer="some-other-profile",
            expected_run_id=run.current_run_id,
            with_reason=True,
        )
        assert ok is False
        assert REVIEWER in (reason or "")
        assert kb.get_task(conn, tid).status == "running"

        # The reviewer the policy DOES nominate is accepted, unchanged.
        assert kb.request_review(
            conn, tid,
            summary="ready",
            reviewer=REVIEWER,
            expected_run_id=run.current_run_id,
        ) is True
        parked = kb.get_task(conn, tid)
        assert parked.status == "review"
        assert parked.assignee == REVIEWER
        handover = _events(conn, tid, "review_requested")
        assert len(handover) == 1
        assert handover[0]["implementer"] == IMPLEMENTER
        assert handover[0]["reviewer"] == REVIEWER


def test_an_omitted_reviewer_on_required_review_work_is_the_policys_nominee(
    kanban_home, monkeypatch,
):
    """The ordinary request shape omits its optional reviewer; for work that
    carries the committed requirement that means the policy's nominee, and
    nobody nominated fails closed before anything is written."""
    _nominate(monkeypatch, REVIEWER)
    with kb.connect() as conn:
        tid, run = _claimed_implementation(conn)
        assert kb.request_review(
            conn, tid, summary="ready", expected_run_id=run.current_run_id,
        ) is True
        parked = kb.get_task(conn, tid)
        assert (parked.status, parked.assignee) == ("review", REVIEWER)
        assert parked.owned_paths == []
        handover = _events(conn, tid, "review_requested")[-1]
        assert handover["implementer"] == IMPLEMENTER
        assert handover["reviewer"] == REVIEWER
        # The review run belongs to the nominated reviewer role, not to the
        # profile that wrote the code.
        review = kb.claim_review_task(conn, tid, claimer=f"{IMPLEMENTER}:2")
        assert review is not None
        review_run = conn.execute(
            "SELECT profile FROM task_runs WHERE id = ?",
            (review.current_run_id,),
        ).fetchone()
        assert review_run["profile"] == REVIEWER

        _nominate(monkeypatch)
        other, other_run = _claimed_implementation(conn, title="second work")
        ok, reason = kb.request_review(
            conn, other,
            summary="ready",
            expected_run_id=other_run.current_run_id,
            with_reason=True,
        )
        assert ok is False
        assert "nominates no reviewer" in (reason or "")
        after = kb.get_task(conn, other)
        assert (after.status, after.assignee) == ("running", IMPLEMENTER)
        assert _events(conn, other, "review_requested") == []


def test_review_lane_reassignment_cannot_target_the_implementer(
    kanban_home, monkeypatch,
):
    """A reviewer swap may not hand the work back to whoever wrote it."""
    _nominate(monkeypatch, REVIEWER)
    with kb.connect() as conn:
        tid, run = _claimed_implementation(conn)
        assert kb.complete_task(
            conn, tid, summary="ready", expected_run_id=run.current_run_id,
        ) is True
        parked = kb.get_task(conn, tid)
        assert (parked.status, parked.assignee) == ("review", REVIEWER)
        assignments_before = _events(conn, tid, "assigned")
        handovers_before = _events(conn, tid, "review_requested")

        with pytest.raises(RuntimeError, match="reviews itself"):
            kb.assign_task(conn, tid, IMPLEMENTER)

        after = kb.get_task(conn, tid)
        assert after.assignee == REVIEWER
        assert after.status == "review"
        # No half-applied reassignment: no assignment event, and the durable
        # return authority still names the pair the handover recorded.
        assert _events(conn, tid, "assigned") == assignments_before
        assert _events(conn, tid, "review_requested") == handovers_before

        # The two things that must keep working are asserted right here, so
        # the refusal above cannot quietly become a blanket one: swapping in
        # the reviewer the policy NOW nominates, and the dedicated return leg
        # that hands the work back to the implementer on purpose.
        _nominate(monkeypatch, OTHER_REVIEWER)
        assert kb.assign_task(conn, tid, OTHER_REVIEWER) is True
        moved = kb.get_task(conn, tid)
        assert (moved.status, moved.assignee) == ("review", OTHER_REVIEWER)
        handover = _events(conn, tid, "review_requested")[-1]
        assert handover["implementer"] == IMPLEMENTER
        assert handover["reviewer"] == OTHER_REVIEWER

        review = kb.claim_review_task(conn, tid, claimer=f"{OTHER_REVIEWER}:1")
        assert review is not None
        ok, implementer = kb.request_changes(
            conn, tid,
            reason="the loop bound is off by one",
            expected_run_id=review.current_run_id,
        )
        assert (ok, implementer) == (True, IMPLEMENTER)
        back = kb.get_task(conn, tid)
        assert back.assignee == IMPLEMENTER
        assert back.status in ("ready", "todo")
