"""An explicit reopen hands the card back COMPLETE, not just re-assigned.

``reopen_review_task`` asked the one route-authority helper whether the return
to the implementer was allowed — and then threw away the answer, writing only
``assignee``. On an owner-governed card the authority helper's answer IS the
replacement route: provider, model, effort, tier and the policy lock that binds
them to a role. Dropping it left the row assigned to the implementer while
every route column still described the reviewer, which is a lock error, and the
next implementation claim was refused by the card's own lock — the work stuck
on a board that says it is ready.

The changes-requested handback already applies route, repin and assignee in one
statement; this asserts the reopen leg does the same, on a REAL owner-approved
card committed through the approval kernel (the only shape whose route the
authority helper governs at all).
"""

from __future__ import annotations

import json

import pytest

from hermes_cli import kanban_db as kb
from plugins.dashboard_auth.raphael_workspace import model_policy

# The owner-workspace harness (authorized proposal context, auto-approver,
# canonical task-graph payload) is reused verbatim so the card under test is
# committed through the REAL approval kernel and carries a real policy lock.
from tests.hermes_cli.test_kanban_review_handback_policy_drift import (  # noqa: F401
    _review_required_graph_args,
)
from tests.hermes_cli.test_owner_workspace import (  # noqa: F401
    _commit_task_graph,
    _configured_provider,
    _configured_raphael_role,
    _install_profiles,
    _temporarily_patch,
    _with_approver,
    ctx,
)


REVIEWER = "raphael-verifier"
IMPLEMENTER = "default"


def _row(conn, task_id):
    return conn.execute(
        "SELECT * FROM tasks WHERE id = ?", (task_id,)
    ).fetchone()


def _events(conn, tid, kind) -> list:
    return [
        json.loads(r["payload"]) if r["payload"] else None
        for r in conn.execute(
            "SELECT payload FROM task_events WHERE task_id = ? AND kind = ? "
            "ORDER BY id",
            (tid, kind),
        ).fetchall()
    ]


def test_governed_reopen_restores_the_implementers_whole_route(ctx):
    """Assignee, provider, model and lock all come back together — and run."""
    _install_profiles(REVIEWER, IMPLEMENTER)
    args = _review_required_graph_args(idempotency_key="graph-reopen-route")
    # The reviewer role runs on a genuinely different provider from the
    # implementer under this policy, so "the route came back" is a claim about
    # real route columns rather than two roles that happen to look alike.
    with _temporarily_patch(
        model_policy, "configured_assignment_for", _configured_raphael_role,
    ):
        _reopen_route_repair(ctx, args)


def _reopen_route_repair(ctx, args):
    approver = _with_approver(ctx.session)
    result = _commit_task_graph(ctx, **args)
    approver.join()

    task_id = result["task_ids"][0]
    with kb.connect(board=result["board"]) as conn:
        implementation = _row(conn, task_id)
        assert implementation["assignee"] == IMPLEMENTER
        assert kb.task_policy_lock_error(implementation) is None

        run = kb.claim_task(conn, task_id, claimer=f"{IMPLEMENTER}:1")
        assert run is not None
        assert kb.complete_task(
            conn, task_id, summary="implemented",
            expected_run_id=run.current_run_id,
        ) is True

        parked = _row(conn, task_id)
        assert parked["status"] == "review"
        assert parked["assignee"] == REVIEWER
        # The park really did re-pin onto the reviewer, so the reopen has a
        # genuine route repair to make rather than nothing to do.
        assert parked["provider_override"] != implementation["provider_override"]
        assert parked["model_policy_lock"] != implementation["model_policy_lock"]

        assert kb.reopen_review_task(conn, task_id) is True

        reopened = _row(conn, task_id)
        assert reopened["assignee"] == IMPLEMENTER
        assert reopened["status"] in ("ready", "todo")
        assert reopened["provider_override"] == implementation["provider_override"]
        assert reopened["model_override"] == implementation["model_override"]
        assert reopened["reasoning_effort"] == implementation["reasoning_effort"]
        assert reopened["execution_tier"] == implementation["execution_tier"]
        assert reopened["model_policy_lock"] == implementation["model_policy_lock"]
        # The lock is the row's own route authority: it must bind THIS role.
        assert kb.task_policy_lock_error(reopened) is None
        # The repin is auditable and landed with the transition.
        assert _events(conn, task_id, "model_route_repinned")[-1]["assignee"] == (
            IMPLEMENTER
        )

        # The point of the repair: the implementer can actually run again.
        reclaim = kb.claim_task(conn, task_id, claimer=f"{IMPLEMENTER}:2")
        assert reclaim is not None
        assert reclaim.assignee == IMPLEMENTER
        assert _row(conn, task_id)["status"] == "running"
