"""A reviewer's findings survive a policy change made after the review claim.

The reviewer/implementer pair of an ACTIVE review run is a durable fact: it was
recorded on the ``review_requested`` event when the handover was accepted. The
handback authority therefore has to come from that provenance, not from
re-asking the policy who reviews RIGHT NOW — the policy is free to move while
the reviewer is mid-run, and when it does, the reviewer still has to be able to
return the work it was legitimately handed.

Re-resolving instead strands the findings: the route-authority helper refuses
the implementer transition, ``submit_review_findings`` reports an error, the
card stays ``running`` under the reviewer, and NOTHING is recorded — no
handback, no ``changes_requested`` event, no findings document. The reviewer's
work is simply lost.

The route the implementer runs on when it gets the card back is a different
question and is still derived from the live policy; only the AUTHORITY comes
from the durable provenance. Both are asserted here on a real owner-approved
(policy-locked) card, which is the only shape the authority helper governs.
"""

from __future__ import annotations

import json

import pytest

from hermes_cli import kanban_db as kb
from plugins.dashboard_auth.raphael_workspace import model_policy

# The owner-workspace harness (authorized proposal context, auto-approver,
# canonical task-graph payload) is reused verbatim so this test commits through
# the REAL approval kernel.
from tests.hermes_cli.test_owner_workspace import (  # noqa: F401
    _commit_task_graph,
    _configured_provider,
    _install_profiles,
    _task_graph_args,
    _temporarily_patch,
    _with_approver,
    ctx,
)


REVIEWER = "raphael-verifier"


def _finding(**overrides) -> dict:
    base = {
        "severity": "blocking",
        "file": "src/app.py",
        "lines": "10-20",
        "problem": "off-by-one in the loop bound",
        "impact": "drops the last item in the batch",
        "smallest_fix": "use `<=` instead of `<` in the range check",
        "candidate_digest": "digest-1",
    }
    base.update(overrides)
    return base


def _events(conn, tid, kind) -> list:
    return [
        json.loads(r["payload"]) if r["payload"] else None
        for r in conn.execute(
            "SELECT payload FROM task_events WHERE task_id = ? AND kind = ? "
            "ORDER BY id",
            (tid, kind),
        ).fetchall()
    ]


def _no_read_only_reviewer_roster():
    """A roster that admits no read-only reviewer role at all."""
    return tuple(
        p for p in model_policy._PROFILE_IDS if p not in kb._READ_ONLY_PROFILES
    )


def _review_required_graph_args(**overrides):
    args = _task_graph_args(**overrides)
    args["tasks"][0]["requires_review"] = True
    return args


def test_findings_land_after_the_reviewer_policy_moves_mid_review(ctx):
    """The whole handback is atomic on the far side of a policy change."""
    _install_profiles(REVIEWER)
    args = _review_required_graph_args(idempotency_key="graph-policy-drift")
    approver = _with_approver(ctx.session)
    result = _commit_task_graph(ctx, **args)
    approver.join()

    task_id = result["task_ids"][0]
    with kb.connect(board=result["board"]) as conn:
        implementer_route = conn.execute(
            "SELECT model_override, provider_override FROM tasks WHERE id = ?",
            (task_id,),
        ).fetchone()
        run = kb.claim_task(conn, task_id, claimer="default:1")
        assert run is not None
        assert kb.complete_task(
            conn, task_id, summary="implemented",
            expected_run_id=run.current_run_id,
        ) is True
        assert kb.get_task(conn, task_id).assignee == REVIEWER

        review = kb.claim_review_task(conn, task_id, claimer=f"{REVIEWER}:1")
        assert review is not None

        # The reviewer policy MOVES while the review run is live: the roster
        # no longer admits the role that is holding this card.
        with _temporarily_patch(
            model_policy, "admitted_profile_ids", _no_read_only_reviewer_roster,
        ):
            assert kb.policy_resolved_reviewer() is None, (
                "this test proves nothing unless the policy actually moved"
            )
            handback = kb.submit_review_findings(
                conn, task_id,
                findings=[_finding()],
                candidate_digest="digest-1",
                expected_run_id=review.current_run_id,
            )

        assert handback["outcome"] == "handed_back", handback
        assert handback["implementer"] == "default"

        back = kb.get_task(conn, task_id)
        row = conn.execute(
            "SELECT * FROM tasks WHERE id = ?", (task_id,)
        ).fetchone()
        changes = _events(conn, task_id, "changes_requested")
        attachments = kb.list_attachments(conn, task_id)
        delivered = _events(conn, task_id, "review_findings_delivered")

    # The card is back with the ORIGINAL implementer, exactly once, with the
    # findings attached — all three records or none, and here it is all three.
    assert back.assignee == "default"
    assert back.status in ("ready", "todo")
    assert len(changes) == 1
    assert changes[0]["implementer"] == "default"
    assert changes[0]["reviewer"] == REVIEWER
    assert len(delivered) == 1
    assert len(attachments) == 1
    assert attachments[0].id == delivered[0]["attachment_id"]
    # The route the implementer resumes on is still a provable authority
    # re-derived from the live policy, not a stranded lock.
    assert kb.task_policy_lock_error(row) is None
    assert row["model_override"] == implementer_route["model_override"]
    assert row["provider_override"] == implementer_route["provider_override"]


def test_a_clean_verdict_also_survives_the_policy_moving_mid_review(ctx):
    """The PASS side of the same run: approval is not re-authorized either."""
    _install_profiles(REVIEWER)
    args = _review_required_graph_args(idempotency_key="graph-drift-pass")
    approver = _with_approver(ctx.session)
    result = _commit_task_graph(ctx, **args)
    approver.join()

    task_id = result["task_ids"][0]
    with kb.connect(board=result["board"]) as conn:
        run = kb.claim_task(conn, task_id, claimer="default:1")
        assert kb.complete_task(
            conn, task_id, summary="implemented",
            expected_run_id=run.current_run_id,
        ) is True
        review = kb.claim_review_task(conn, task_id, claimer=f"{REVIEWER}:1")
        assert review is not None

        with _temporarily_patch(
            model_policy, "admitted_profile_ids", _no_read_only_reviewer_roster,
        ):
            verdict = kb.submit_review_findings(
                conn, task_id,
                findings=[],
                candidate_digest="digest-clean",
                expected_run_id=review.current_run_id,
            )

        assert verdict["outcome"] == "passed", verdict
        approved = kb.get_task(conn, task_id)

    assert approved.status == "done"
    assert approved.completed_at is not None
