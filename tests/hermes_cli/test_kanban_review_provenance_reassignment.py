"""Durable return authority survives review-lane reassignment.

When a review card is reassigned to a different reviewer mid-flight, the
durable provenance (:func:`_latest_review_provenance`) must be updated
ATOMICALLY with the assignment write.  Without this, the reviewer provenance
stays whatever the original handover recorded, and if the policy moves while
the NEW reviewer is mid-run, the handback will be refused.

The regression scenario:
1. Create a policy-locked task with review requirement
2. Hand it over — a real `review_requested` provenance event is written
3. Reassign on the review lane to a different, policy-nominated reviewer
4. Claim as that new reviewer
5. Drift the policy mid-flight (move the selector to empty/nobody)
6. Hand back findings — must land atomically, not error out
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from plugins.dashboard_auth.raphael_workspace import model_policy

# The governed shape: a REAL owner-approved (policy-locked, receipt-bound) card
# committed through the approval kernel, the only shape whose handback
# authority consults the durable provenance at all.
from tests.hermes_cli.test_kanban_review_handback_policy_drift import (  # noqa: F401
    _finding,
    _review_required_graph_args,
)
from tests.hermes_cli.test_owner_workspace import (  # noqa: F401
    _commit_task_graph,
    _configured_provider,
    _install_profiles,
    _with_approver,
    ctx,
)


INITIAL_REVIEWER = "raphael-verifier"
REPLACEMENT_REVIEWER = "raphael-planner"
IMPLEMENTER = "raphael-builder"


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


def test_review_lane_reassignment_updates_provenance_atomically(
    kanban_home, monkeypatch, all_assignees_spawnable
):
    """The full regression: reassign + policy drift + handback must land.

    This exercises the exact scenario that was broken:
    - Task handed from implementer to initial reviewer
    - Reassigned on review lane to a different reviewer
    - Policy drifts to empty while the new reviewer is running
    - Handback must succeed (not error out) because provenance was atomically
      updated during the reassignment
    """
    with kb.connect() as conn:
        _nominate(monkeypatch, INITIAL_REVIEWER)

        tid = kb.create_task(
            conn,
            title="provenance-test",
            assignee=IMPLEMENTER,
            requires_review=True,
        )

        run = kb.claim_task(conn, tid, claimer=f"{IMPLEMENTER}:1")
        assert run is not None
        ok = kb.complete_task(
            conn, tid, summary="ready", expected_run_id=run.current_run_id,
        )
        assert ok is True

        task = kb.get_task(conn, tid)
        assert task.status == "review"
        assert task.assignee == INITIAL_REVIEWER

        requested_events = _events(conn, tid, "review_requested")
        assert len(requested_events) == 1
        assert requested_events[0]["implementer"] == IMPLEMENTER
        assert requested_events[0]["reviewer"] == INITIAL_REVIEWER

        _nominate(monkeypatch, REPLACEMENT_REVIEWER)

        assert kb.assign_task(conn, tid, REPLACEMENT_REVIEWER) is True

        task = kb.get_task(conn, tid)
        assert task.status == "review"
        assert task.assignee == REPLACEMENT_REVIEWER

        requested_events = _events(conn, tid, "review_requested")
        assert len(requested_events) == 2, (
            "reassignment on the review lane must write a new review_requested "
            "event to update the provenance atomically"
        )
        assert requested_events[1]["implementer"] == IMPLEMENTER, (
            "the implementer half of provenance must be unchanged"
        )
        assert requested_events[1]["reviewer"] == REPLACEMENT_REVIEWER, (
            "the reviewer half of provenance must be the new reviewer"
        )

        run = kb.claim_review_task(conn, tid, claimer=f"{REPLACEMENT_REVIEWER}:1")
        assert run is not None, (
            "claim_review_task must succeed for a review-lane card"
        )
        current_run_id = run.current_run_id

        _nominate(monkeypatch, "nobody-exists")

        assert kb.policy_resolved_reviewer() is None, (
            "sanity: the policy has drifted to empty"
        )

        result = kb.submit_review_findings(
            conn,
            tid,
            findings=[
                {
                    "file": "test.py",
                    "lines": "1-5",
                    "severity": "minor",
                    "problem": "test finding",
                    "impact": "test impact",
                    "smallest_fix": "fix it",
                    "candidate_digest": "deadbeef",
                }
            ],
            candidate_digest="deadbeef",
            expected_run_id=current_run_id,
        )

        assert result["outcome"] == "handed_back", (
            f"handback must succeed despite policy drift; got {result}"
        )
        assert result["implementer"] == IMPLEMENTER

        task = kb.get_task(conn, tid)
        assert task.assignee == IMPLEMENTER, (
            "the task must return to its original implementer"
        )
        assert task.status in ("ready", "todo", "blocked"), (
            "the task must be back on the implementation lane"
        )

        changes_events = _events(conn, tid, "changes_requested")
        assert len(changes_events) == 1
        assert changes_events[0]["implementer"] == IMPLEMENTER
        assert changes_events[0]["reviewer"] == REPLACEMENT_REVIEWER

        delivered_events = _events(conn, tid, "review_findings_delivered")
        assert len(delivered_events) == 1
        assert delivered_events[0]["count"] == 1
        assert "attachment_id" in delivered_events[0]


def test_non_review_lane_assignment_does_not_write_review_provenance(
    kanban_home, monkeypatch, all_assignees_spawnable
):
    """Ordinary implementation-lane assignment is completely unaffected.

    The provenance update applies ONLY to review-lane reassignment.  An
    assignment on any other lane must not write a review_requested event.
    """
    with kb.connect() as conn:
        _nominate(monkeypatch, INITIAL_REVIEWER)

        tid = kb.create_task(
            conn,
            title="impl-lane-assign",
            assignee="alice",
        )
        assert kb.assign_task(conn, tid, "bob") is True

        requested_events = _events(conn, tid, "review_requested")
        assert requested_events == [], (
            "implementation-lane assignment must not write review_requested"
        )


def test_review_lane_assignment_to_same_reviewer_does_not_duplicate_provenance(
    kanban_home, monkeypatch, all_assignees_spawnable
):
    """Assigning to the same reviewer is a no-op for provenance.

    The provenance update fires only when the assignee actually changes.
    """
    with kb.connect() as conn:
        _nominate(monkeypatch, INITIAL_REVIEWER)

        tid = kb.create_task(
            conn,
            title="same-reviewer",
            assignee=IMPLEMENTER,
            requires_review=True,
        )

        run = kb.claim_task(conn, tid, claimer=f"{IMPLEMENTER}:1")
        assert run is not None
        kb.complete_task(conn, tid, summary="ready", expected_run_id=run.current_run_id)

        task = kb.get_task(conn, tid)
        assert task.status == "review"
        assert task.assignee == INITIAL_REVIEWER

        assert kb.assign_task(conn, tid, INITIAL_REVIEWER) is True

        requested_events = _events(conn, tid, "review_requested")
        assert len(requested_events) == 1, (
            "assigning to the same reviewer must not duplicate provenance"
        )


def test_review_lane_reassignment_lands_the_handback_on_a_governed_card(
    ctx, monkeypatch,
):
    """The reroute -> claim -> policy-drift -> handback race on a governed card.

    Only a policy-governed row consults the durable provenance when the
    reviewer hands back, so this is the shape the atomic provenance update
    exists for: with the update absent the handback is refused (the
    provenance still names the first reviewer, the policy no longer names
    anyone), with it the findings land — all three records, once.
    """
    _install_profiles(INITIAL_REVIEWER, REPLACEMENT_REVIEWER)
    _nominate(monkeypatch, INITIAL_REVIEWER)
    args = _review_required_graph_args(idempotency_key="graph-reassign-drift")
    approver = _with_approver(ctx.session)
    result = _commit_task_graph(ctx, **args)
    approver.join()

    task_id = result["task_ids"][0]
    with kb.connect(board=result["board"]) as conn:
        governed = conn.execute(
            "SELECT execution_tier, model_policy_lock, owner_receipt_bound "
            "FROM tasks WHERE id = ?",
            (task_id,),
        ).fetchone()
        assert kb.task_is_policy_governed(governed), (
            "this test proves nothing on an ungoverned card"
        )
        run = kb.claim_task(conn, task_id, claimer="default:1")
        assert run is not None
        assert kb.complete_task(
            conn, task_id, summary="implemented",
            expected_run_id=run.current_run_id,
        ) is True
        assert kb.get_task(conn, task_id).assignee == INITIAL_REVIEWER

        # Reroute on the review lane to the reviewer the policy now names.
        _nominate(monkeypatch, REPLACEMENT_REVIEWER)
        assert kb.assign_task(conn, task_id, REPLACEMENT_REVIEWER) is True
        parked = kb.get_task(conn, task_id)
        assert parked.status == "review"
        assert parked.assignee == REPLACEMENT_REVIEWER

        review = kb.claim_review_task(
            conn, task_id, claimer=f"{REPLACEMENT_REVIEWER}:1",
        )
        assert review is not None

        # The policy moves while the new reviewer is mid-run.
        _nominate(monkeypatch, "nobody-exists")
        assert kb.policy_resolved_reviewer() is None, (
            "this test proves nothing unless the policy actually moved"
        )
        handback = kb.submit_review_findings(
            conn, task_id,
            findings=[_finding()],
            candidate_digest="digest-1",
            expected_run_id=review.current_run_id,
        )

        # The landing is judged first: on a governed card a stale provenance
        # (the reassignment not recorded) refuses this handback outright.
        assert handback["outcome"] == "handed_back", handback
        back = kb.get_task(conn, task_id)
        requested = _events(conn, task_id, "review_requested")
        changes = _events(conn, task_id, "changes_requested")
        delivered = _events(conn, task_id, "review_findings_delivered")
        attachments = kb.list_attachments(conn, task_id)

    assert requested[-1]["reviewer"] == REPLACEMENT_REVIEWER
    assert requested[-1]["implementer"] == "default"
    assert back.assignee == "default"
    assert back.status in ("ready", "todo")
    assert len(changes) == 1
    assert changes[0]["implementer"] == "default"
    assert changes[0]["reviewer"] == REPLACEMENT_REVIEWER
    assert len(delivered) == 1
    assert len(attachments) == 1
    assert attachments[0].id == delivered[0]["attachment_id"]
