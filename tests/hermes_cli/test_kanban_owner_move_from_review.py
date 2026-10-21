"""An owner's move out of review hands the card back to its implementer.

The owner's generic move (``owner_workspace.move_task`` ->
``kanban_db.cas_transition_task``) changed only ``status``. A review card the
owner reactivated therefore kept the reviewer as its assignee, the reviewer's
route and lock, and the read-only write scope of the review run: a card that
says it is ready, held by the one role that may not build it.
``reopen_review_task`` already did the right hand-back — the newest
``review_requested`` implementer, with route and restored write scope in the
same ``UPDATE`` — and a move from ``review`` to ``ready`` or ``todo`` now
applies that same restore.

Every card here is committed through the REAL owner approval kernel, so it
carries a real policy lock and its review park genuinely re-pins it onto a
reviewer that runs on a different provider.
"""

from __future__ import annotations

import json

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import owner_workspace as ow
from plugins.dashboard_auth.raphael_workspace import model_policy

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
# The role holding the card: who it is, the route it runs on, and the
# repository write scope it may use.
LANE = (
    "assignee", "provider_override", "model_override", "reasoning_effort",
    "execution_tier", "model_policy_lock", "owned_paths",
    "integrates_parent_heads",
)


@pytest.fixture
def graph(ctx):
    """One review-required card, with the reviewer on another provider."""
    _install_profiles(REVIEWER, IMPLEMENTER)
    args = _review_required_graph_args(idempotency_key="graph-move-from-review")
    with _temporarily_patch(
        model_policy, "configured_assignment_for", _configured_raphael_role,
    ):
        approver = _with_approver(ctx.session)
        result = _commit_task_graph(ctx, **args)
        approver.join()
        yield result


def _row(conn, task_id):
    return conn.execute(
        "SELECT * FROM tasks WHERE id = ?", (task_id,)
    ).fetchone()


def _lane(row) -> dict:
    return {column: row[column] for column in LANE}


def _events_after(conn, task_id, revision) -> list:
    return [
        (r["kind"], json.loads(r["payload"]) if r["payload"] else None)
        for r in conn.execute(
            "SELECT kind, payload FROM task_events "
            "WHERE task_id = ? AND id > ? ORDER BY id",
            (task_id, revision),
        ).fetchall()
    ]


def _review_requests(conn, task_id) -> list:
    return [
        json.loads(r["payload"])
        for r in conn.execute(
            "SELECT payload FROM task_events "
            "WHERE task_id = ? AND kind = 'review_requested' ORDER BY id",
            (task_id,),
        ).fetchall()
    ]


def _owner_move(ctx, graph, task_id, *, to_status, expected_status):
    """The owner's own move, through the real owner kernel and approval."""
    with kb.connect(board=graph["board"]) as conn:
        revision = kb.task_event_revision(conn, task_id)
    approver = _with_approver(ctx.session)
    result = ow.move_task(
        ctx,
        idempotency_key=f"move-{expected_status}-to-{to_status}",
        task_id=task_id,
        to_status=to_status,
        expected_status=expected_status,
        expected_revision=revision,
        project_id=graph["project_id"],
    )
    approver.join()
    assert result["ok"] is True
    assert result["status"] == to_status
    return revision


def _parked_for_review(graph):
    """The implementer finishes and the review requirement parks the card."""
    task_id = graph["task_ids"][0]
    with kb.connect(board=graph["board"]) as conn:
        implementation = _row(conn, task_id)
        assert implementation["assignee"] == IMPLEMENTER
        run = kb.claim_task(conn, task_id, claimer=f"{IMPLEMENTER}:1")
        assert run is not None
        assert kb.complete_task(
            conn, task_id, summary="implemented",
            expected_run_id=run.current_run_id,
        ) is True
        parked = _row(conn, task_id)
        assert _review_requests(conn, task_id)[-1]["implementer"] == IMPLEMENTER
    assert parked["status"] == "review"
    assert parked["assignee"] == REVIEWER
    # The park moved route, lock and write scope onto the reviewer too, so a
    # hand-back has a genuine lane to restore rather than nothing to do.
    for column in ("provider_override", "model_policy_lock", "owned_paths"):
        assert parked[column] != implementation[column], column
    return task_id, implementation, parked


def _assert_owner_move_hands_back(ctx, graph, to_status):
    task_id, implementation, _ = _parked_for_review(graph)

    revision = _owner_move(
        ctx, graph, task_id, to_status=to_status, expected_status="review",
    )

    with kb.connect(board=graph["board"]) as conn:
        moved = _row(conn, task_id)
        events = _events_after(conn, task_id, revision)
    assert moved["status"] == to_status
    # Assignee, route and write scope come back to the implementer together.
    assert _lane(moved) == _lane(implementation)
    assert kb.task_policy_lock_error(moved) is None
    # The owner's own event stays the move's next event, which is how a
    # replay of the receipt recognizes it; the repin lands with it.
    assert [kind for kind, _ in events] == ["owner_move", "model_route_repinned"]
    assert events[1][1]["assignee"] == IMPLEMENTER


def test_owner_move_review_to_ready_hands_the_card_back_to_its_implementer(
    ctx, graph,
):
    _assert_owner_move_hands_back(ctx, graph, "ready")


def test_owner_move_review_to_todo_hands_the_card_back_the_same_way(
    ctx, graph,
):
    _assert_owner_move_hands_back(ctx, graph, "todo")


def test_owner_move_review_to_done_leaves_the_assignee_unchanged(ctx, graph):
    task_id, _, parked = _parked_for_review(graph)

    revision = _owner_move(
        ctx, graph, task_id, to_status="done", expected_status="review",
    )

    with kb.connect(board=graph["board"]) as conn:
        done = _row(conn, task_id)
        events = _events_after(conn, task_id, revision)
    assert done["status"] == "done"
    assert done["assignee"] == REVIEWER
    assert _lane(done) == _lane(parked)
    assert [kind for kind, _ in events] == ["owner_move"]


def test_review_card_without_a_review_request_keeps_its_assignee(ctx, graph):
    """No recorded implementer: nothing to hand back to, and no failure."""
    task_id = graph["task_ids"][0]
    # Onto the review lane by the owner's own move and then routed to the
    # reviewer, so no handover ever recorded an implementer for it.
    _owner_move(ctx, graph, task_id, to_status="review", expected_status="ready")
    with kb.connect(board=graph["board"]) as conn:
        assert kb.assign_task(conn, task_id, REVIEWER) is True
        in_review = _row(conn, task_id)
        assert _review_requests(conn, task_id) == []
    assert in_review["status"] == "review"
    assert in_review["assignee"] == REVIEWER

    revision = _owner_move(
        ctx, graph, task_id, to_status="ready", expected_status="review",
    )

    with kb.connect(board=graph["board"]) as conn:
        moved = _row(conn, task_id)
        events = _events_after(conn, task_id, revision)
    assert moved["status"] == "ready"
    assert _lane(moved) == _lane(in_review)
    assert [kind for kind, _ in events] == ["owner_move"]


def test_reopen_review_task_gives_the_same_row_and_events_as_today(ctx, graph):
    task_id, implementation, _ = _parked_for_review(graph)

    with kb.connect(board=graph["board"]) as conn:
        revision = kb.task_event_revision(conn, task_id)
        assert kb.reopen_review_task(conn, task_id) is True
        reopened = _row(conn, task_id)
        events = _events_after(conn, task_id, revision)

    # The whole implementation row comes back; only the first claim's start
    # time is left behind.
    def without_start(row):
        return {key: row[key] for key in row.keys() if key != "started_at"}

    assert without_start(reopened) == without_start(implementation)
    assert events == [
        ("model_route_repinned", {
            "assignee": IMPLEMENTER,
            "model": implementation["model_override"],
            "provider": implementation["provider_override"],
            "reasoning_effort": implementation["reasoning_effort"],
            "execution_tier": implementation["execution_tier"],
            "source": "committed_review_requirement",
        }),
        ("review_reopened", {"status": "ready", "implementer": IMPLEMENTER}),
    ]
