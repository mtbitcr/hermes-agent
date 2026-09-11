"""A committed review requirement parks the handover; it never completes it.

A task's committed specification may state that the work is independently
reviewed before it is done (``requires_review``). The kernel honours that at
the ONE moment it matters — the implementer's handover — by routing the card
onto the EXISTING review lane instead of writing ``done``:

* ``complete_task`` parks through :func:`kanban_db.request_review`;
* the reviewer is whoever the team's model policy resolves RIGHT THEN
  (:func:`kanban_db.policy_resolved_reviewer`), never a name captured on the
  row when the specification was committed;
* findings come back through the unchanged ``request_changes`` handback; and
* a clean reviewer verdict approves the parked card through the same
  ``complete_task`` it would have used without the requirement.

The pre-existing READ-ONLY audit review task (a ``raphael-verifier`` card with
``owned_paths=[]``, which audits already-finished work) is a different thing
entirely and must keep behaving exactly as it does today — it IS the review, so
it is never parked awaiting one. That regression is asserted here at both the
kernel boundary and the committed-specification boundary.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from plugins.dashboard_auth.raphael_workspace import model_policy

# The owner-workspace harness (authorized proposal context, auto-approver,
# canonical task-graph payload) is reused verbatim rather than re-implemented,
# so these tests commit through the REAL approval kernel.
from tests.hermes_cli.test_owner_workspace import (  # noqa: F401
    _commit_task_graph,
    _configured_provider,
    _configured_raphael_role,
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


# ---------------------------------------------------------------------------
# Kernel boundary
# ---------------------------------------------------------------------------


def test_handover_parks_the_work_for_review_instead_of_completing(kanban_home):
    with kb.connect() as conn:
        tid = kb.create_task(
            conn, title="ship the thing", assignee="worker", requires_review=True,
        )
        assert kb.get_task(conn, tid).requires_review is True

        assert _hand_over(conn, tid) is True

        task = kb.get_task(conn, tid)
        assert task.status == "review", (
            f"a committed review requirement must park the handover, got "
            f"{task.status!r}"
        )
        assert task.completed_at is None
        assert task.result is None
        # Parked on the EXISTING review lane, with the implementer and the
        # live-resolved reviewer both recorded by request_review's own event.
        requested = _events(conn, tid, "review_requested")
        assert len(requested) == 1
        assert requested[0]["implementer"] == "worker"
        assert requested[0]["reviewer"] == REVIEWER
        assert task.assignee == REVIEWER
        assert _events(conn, tid, "completed") == []


def test_work_without_the_requirement_still_completes_on_handover(kanban_home):
    """The default is untouched: no requirement, no park."""
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="ordinary work", assignee="worker")
        assert kb.get_task(conn, tid).requires_review is False

        assert _hand_over(conn, tid) is True

        task = kb.get_task(conn, tid)
        assert task.status == "done"
        assert _events(conn, tid, "review_requested") == []


def test_findings_return_to_the_implementer_through_request_changes(kanban_home):
    """The handback path is the existing one, unchanged."""
    with kb.connect() as conn:
        tid = kb.create_task(
            conn, title="reviewed work", assignee="worker", requires_review=True,
        )
        assert _hand_over(conn, tid) is True

        review = kb.claim_review_task(conn, tid, claimer=f"{REVIEWER}:1")
        assert review is not None
        handback = kb.submit_review_findings(
            conn, tid,
            findings=[_finding()],
            candidate_digest="digest-1",
            expected_run_id=review.current_run_id,
        )
        assert handback["outcome"] == "handed_back"
        assert handback["implementer"] == "worker"

        task = kb.get_task(conn, tid)
        assert task.status == "ready"
        assert task.assignee == "worker"
        changes = _events(conn, tid, "changes_requested")
        assert len(changes) == 1
        assert changes[0]["implementer"] == "worker"
        assert changes[0]["reviewer"] == REVIEWER
        # The findings document reached the implementer's card.
        assert len(kb.list_attachments(conn, tid)) == 1

        # And the reworked handover parks again rather than completing.
        assert _hand_over(conn, tid, claimer="worker:2", summary="reworked") is True
        assert kb.get_task(conn, tid).status == "review"


def test_a_clean_verdict_approves_the_parked_task(kanban_home):
    """The pass goes through the existing approval route (``complete_task``)."""
    with kb.connect() as conn:
        tid = kb.create_task(
            conn, title="approved work", assignee="worker", requires_review=True,
        )
        assert _hand_over(conn, tid) is True

        review = kb.claim_review_task(conn, tid, claimer=f"{REVIEWER}:1")
        assert review is not None
        verdict = kb.submit_review_findings(
            conn, tid,
            findings=[],
            candidate_digest="digest-clean",
            expected_run_id=review.current_run_id,
        )
        assert verdict["outcome"] == "passed", verdict

        task = kb.get_task(conn, tid)
        assert task.status == "done"
        assert task.completed_at is not None
        # Exactly one park and one completion: the approval did NOT re-park.
        assert len(_events(conn, tid, "review_requested")) == 1
        assert len(_events(conn, tid, "completed")) == 1


def test_a_human_approving_from_the_review_lane_is_not_re_parked(kanban_home):
    """``complete_task`` on a card already in ``review`` is the approval."""
    with kb.connect() as conn:
        tid = kb.create_task(
            conn, title="human approval", assignee="worker", requires_review=True,
        )
        assert _hand_over(conn, tid) is True
        assert kb.get_task(conn, tid).status == "review"

        assert kb.complete_task(conn, tid, summary="approved by hand") is True
        assert kb.get_task(conn, tid).status == "done"
        assert len(_events(conn, tid, "review_requested")) == 1


def test_the_reviewer_is_resolved_live_on_the_handover_path(kanban_home, monkeypatch):
    """Nothing about WHO reviews is read off the row, and it fails CLOSED.

    The policy roster is consulted at the moment of the handover. When it
    admits no read-only reviewer role there is no independent reviewer, so the
    requirement is still honoured — the card parks instead of completing — but
    the work is parked with NO assignee. Selecting the implementer instead
    would leave the review dispatcher free to re-claim the card for the very
    profile that wrote the code.
    """
    with kb.connect() as conn:
        tid = kb.create_task(
            conn, title="no reviewer today", assignee="worker", requires_review=True,
        )
        monkeypatch.setattr(
            model_policy,
            "admitted_profile_ids",
            lambda: tuple(
                p for p in model_policy._PROFILE_IDS if p not in kb._READ_ONLY_PROFILES
            ),
        )
        assert kb.policy_resolved_reviewer() is None

        assert _hand_over(conn, tid) is True
        task = kb.get_task(conn, tid)
        assert task.status == "review"
        assert task.assignee is None, (
            "with no reviewer resolvable the kernel must park UNASSIGNED "
            "(fail closed) instead of with the implementer"
        )

        # Restore the roster: the very next handover resolves a reviewer.
        monkeypatch.undo()
        monkeypatch.setenv("HERMES_HOME", str(kanban_home))
        second = kb.create_task(
            conn, title="reviewer is back", assignee="worker", requires_review=True,
        )
        assert _hand_over(conn, second, claimer="worker:2") is True
        assert kb.get_task(conn, second).assignee == REVIEWER


def test_the_read_only_audit_review_task_is_never_parked(kanban_home):
    """REGRESSION: the pre-existing read-only audit review task is unchanged.

    A ``raphael-verifier`` card (``owned_paths=[]``, read-only) exists to audit
    already-finished work. It is not implementation work awaiting a review, so
    its handover must still COMPLETE, exactly as it does today — and the new
    requirement may never be attached to it in the first place.
    """
    with kb.connect() as conn:
        audit_id = kb.create_task(
            conn, title="Confirm the owner-visible result", assignee=REVIEWER,
        )
        audit = kb.get_task(conn, audit_id)
        assert audit.owned_paths == []
        assert audit.requires_review is False

        assert _hand_over(conn, audit_id, claimer=f"{REVIEWER}:1") is True
        after = kb.get_task(conn, audit_id)
        assert after.status == "done", (
            f"the read-only audit review task must still complete on handover, "
            f"got {after.status!r}"
        )
        assert _events(conn, audit_id, "review_requested") == []

        # And the requirement cannot be attached to one.
        with pytest.raises(ValueError, match="read-only reviewer task"):
            kb.create_task(
                conn, title="reviewer reviewing itself", assignee=REVIEWER,
                requires_review=True,
            )


def test_the_task_projection_carries_the_requirement(kanban_home):
    """``_task_dict`` serialises the whole dataclass, so the projection the
    dashboard read path returns carries the field the moment ``Task`` does."""
    from dataclasses import asdict

    with kb.connect() as conn:
        tid = kb.create_task(
            conn, title="surfaced", assignee="worker", requires_review=True,
        )
        plain = kb.create_task(conn, title="not surfaced", assignee="worker")
        assert asdict(kb.get_task(conn, tid))["requires_review"] is True
        assert asdict(kb.get_task(conn, plain))["requires_review"] is False


def test_the_dashboard_read_path_round_trips_the_requirement(kanban_home):
    # ``plugins.kanban.dashboard.plugin_api`` declares multipart upload
    # endpoints, so importing it needs ``python-multipart`` (a real project
    # dependency). Skip rather than fail where it is absent.
    pytest.importorskip(
        "multipart", reason="python-multipart is required to import plugin_api",
    )
    from plugins.kanban.dashboard import plugin_api

    with kb.connect() as conn:
        tid = kb.create_task(
            conn, title="surfaced", assignee="worker", requires_review=True,
        )
        plain = kb.create_task(conn, title="not surfaced", assignee="worker")
        assert plugin_api._task_dict(kb.get_task(conn, tid))["requires_review"] is True
        assert (
            plugin_api._task_dict(kb.get_task(conn, plain))["requires_review"] is False
        )


# ---------------------------------------------------------------------------
# Committed-specification boundary
# ---------------------------------------------------------------------------


def _review_required_graph_args(**overrides):
    args = _task_graph_args(**overrides)
    args["tasks"][0]["requires_review"] = True
    return args


def test_the_committed_specification_carries_the_requirement(ctx):
    args = _review_required_graph_args(idempotency_key="graph-review-required")
    approver = _with_approver(ctx.session)
    result = _commit_task_graph(ctx, **args)
    approver.join()

    assert result["ok"] is True
    with kb.connect(board=result["board"]) as conn:
        first = kb.get_task(conn, result["task_ids"][0])
        second = kb.get_task(conn, result["task_ids"][1])
    assert first.requires_review is True
    assert second.requires_review is False, (
        "the requirement must land on the task that states it, not its siblings"
    )


def test_a_committed_handover_parks_on_the_live_policy_route(ctx):
    """The reviewer's route is resolved at HANDOVER time, never at commit time.

    The card is committed while the policy resolves every role to Anthropic,
    and handed over after the reviewer role's configured provider has moved.
    The parked card must carry the route the policy resolves NOW — a route
    captured at commit would still name the Anthropic lane.
    """
    _install_profiles(REVIEWER)
    args = _review_required_graph_args(idempotency_key="graph-live-reviewer")
    approver = _with_approver(ctx.session)
    result = _commit_task_graph(ctx, **args)
    approver.join()

    task_id = result["task_ids"][0]
    commit_time_reviewer_route = model_policy.resolve_task_assignment(
        REVIEWER, "routine",
    )
    assert commit_time_reviewer_route.provider == "anthropic"

    with kb.connect(board=result["board"]) as conn:
        before = kb.get_task(conn, task_id)
        assert before.assignee == "default"
        assert before.model_policy_lock

        with _temporarily_patch(
            model_policy, "configured_assignment_for", _configured_raphael_role,
        ):
            handover_time = model_policy.resolve_task_assignment(REVIEWER, "routine")
            assert handover_time.provider == "openai-codex", (
                "this test proves nothing unless the policy actually moved"
            )
            run = kb.claim_task(conn, task_id, claimer="default:1")
            assert run is not None
            assert kb.complete_task(
                conn, task_id, summary="implemented",
                expected_run_id=run.current_run_id,
            ) is True

        parked = kb.get_task(conn, task_id)
        row = conn.execute(
            "SELECT * FROM tasks WHERE id = ?", (task_id,)
        ).fetchone()

    assert parked.status == "review"
    assert parked.assignee == REVIEWER
    assert parked.provider_override == handover_time.provider
    assert parked.model_override == handover_time.model
    assert parked.provider_override != commit_time_reviewer_route.provider
    assert parked.model_override != commit_time_reviewer_route.model
    # The re-pin is a real, provable authority — not a stranded lock.
    assert kb.task_policy_lock_error(row) is None


def test_a_committed_handback_returns_the_card_to_the_implementer(ctx):
    """Findings come back through ``request_changes`` and the lock follows."""
    _install_profiles(REVIEWER)
    args = _review_required_graph_args(idempotency_key="graph-review-handback")
    approver = _with_approver(ctx.session)
    result = _commit_task_graph(ctx, **args)
    approver.join()

    task_id = result["task_ids"][0]
    with kb.connect(board=result["board"]) as conn:
        implementer_route = conn.execute(
            "SELECT model_override, provider_override, model_policy_lock "
            "FROM tasks WHERE id = ?", (task_id,),
        ).fetchone()
        run = kb.claim_task(conn, task_id, claimer="default:1")
        assert kb.complete_task(
            conn, task_id, summary="implemented", expected_run_id=run.current_run_id,
        ) is True

        review = kb.claim_review_task(conn, task_id, claimer=f"{REVIEWER}:1")
        assert review is not None
        handback = kb.submit_review_findings(
            conn, task_id,
            findings=[_finding()],
            candidate_digest="digest-1",
            expected_run_id=review.current_run_id,
        )
        assert handback["outcome"] == "handed_back", handback

        back = kb.get_task(conn, task_id)
        row = conn.execute(
            "SELECT * FROM tasks WHERE id = ?", (task_id,)
        ).fetchone()

    assert back.assignee == "default"
    assert back.status in ("ready", "todo")
    assert row["model_override"] == implementer_route["model_override"]
    assert row["provider_override"] == implementer_route["provider_override"]
    assert kb.task_policy_lock_error(row) is None


def test_a_committed_task_without_the_requirement_still_refuses_review_handoff(ctx):
    """REGRESSION: the locked-route invariant is untouched where it applies.

    Without a committed review requirement, handing a locked card to a
    reviewer is still the silent re-pin the lock exists to prevent.
    """
    _install_profiles(REVIEWER)
    args = _task_graph_args(idempotency_key="graph-no-review-requirement")
    approver = _with_approver(ctx.session)
    result = _commit_task_graph(ctx, **args)
    approver.join()

    task_id = result["task_ids"][0]
    with kb.connect(board=result["board"]) as conn:
        before = conn.execute(
            "SELECT * FROM tasks WHERE id = ?", (task_id,)
        ).fetchone()
        run = kb.claim_task(conn, task_id, claimer="default:1")
        assert run is not None
        with pytest.raises(RuntimeError, match="the owner approved that exact"):
            kb.request_review(
                conn, task_id, reviewer=REVIEWER,
                expected_run_id=run.current_run_id,
            )
        after = conn.execute(
            "SELECT * FROM tasks WHERE id = ?", (task_id,)
        ).fetchone()

    for column in (
        "assignee", "model_override", "provider_override", "reasoning_effort",
        "execution_tier", "model_policy_lock",
    ):
        assert after[column] == before[column]


def test_commit_refuses_the_requirement_on_a_read_only_audit_review_task(ctx):
    """REGRESSION: the requirement can never leak onto the audit review task."""
    _install_profiles(REVIEWER)
    args = _task_graph_args(idempotency_key="graph-verifier-requires-review")
    args["tasks"][1]["assignee"] = REVIEWER
    args["tasks"][1]["requires_review"] = True

    with pytest.raises(Exception) as excinfo:
        _commit_task_graph(ctx, **args)
    assert getattr(excinfo.value, "code", None) == "invalid_argument"
    assert "requires_review" in str(excinfo.value)


def test_a_committed_read_only_audit_review_task_is_unchanged(ctx):
    """REGRESSION: the read-only audit review child keeps its exact shape."""
    _install_profiles(REVIEWER)
    args = _task_graph_args(idempotency_key="graph-verifier-untouched")
    args["tasks"][1]["assignee"] = REVIEWER
    approver = _with_approver(ctx.session)
    with _temporarily_patch(
        model_policy, "configured_assignment_for", _configured_raphael_role,
    ):
        result = _commit_task_graph(ctx, **args)
    approver.join()

    assert result["ok"] is True
    with kb.connect(board=result["board"]) as conn:
        audit = kb.get_task(conn, result["task_ids"][1])
    assert audit.assignee == REVIEWER
    assert audit.owned_paths == []
    assert audit.integrates_parent_heads is False
    assert audit.requires_review is False
