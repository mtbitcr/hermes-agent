"""The owner Plan page's board window on a project above the task limit."""

from __future__ import annotations

import inspect
import threading
import time

import pytest

from hermes_cli import kanban_db, owner_workspace as ow
from plugins.dashboard_auth.raphael_workspace import model_policy
from tools import approval

_LIMIT = 200
_TOP_KEYS = {
    "project", "board", "columns", "workers", "attachments", "runs",
    "steward", "truncated",
}
_BOARD_KEYS = {"slug", "name", "project_id", "counts", "total"}
_TRUNCATED_KEYS = {"tasks", "workers", "attachments", "runs"}
_TASK_KEYS = {
    "id", "title", "assignee_name", "responsibility", "updated_at",
    "event_revision", "review_state", "stopped_work",
    "risk_tier", "risk_tier_raised", "parent_ids", "child_ids",
}


@pytest.fixture
def ctx():
    return ow.OwnerContext(actor="default", profile="default", session="run_owt")


@pytest.fixture(autouse=True)
def _configured_provider(monkeypatch):
    monkeypatch.setattr(
        model_policy,
        "configured_assignment_for",
        lambda profile: model_policy.assignment_for(profile, "anthropic"),
    )


def _auto_approve(session_key: str, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        with approval._lock:
            queue = approval._gateway_queues.get(session_key)
            aid = queue[0].approval_id if queue else None
        if aid:
            approval.resolve_gateway_approval(session_key, "once", approval_id=aid)
            return
        time.sleep(0.01)
    raise TimeoutError("no approval was queued")


def _new_project(ctx) -> dict:
    kwargs = {
        "idempotency_key": "window-graph",
        "mode": "new",
        "project_name": "Window Project",
        "project_description": "A busy owner project.",
        "project_id": None,
        "request_title": "Keep the board window current",
        "specification": "Show the current work of a busy project.",
        "current_milestone": "Now: show current work.",
        "owner_visible_result": "The Plan page shows running work.",
        "root_assignee": "default",
        "tasks": [
            {
                "title": "Prepare the release",
                "body": "Create the smallest complete release.",
                "assignee": "default",
                "responsibility": "B03",
                "execution_tier": "routine",
                "risk_tier": 1,
                "parents": [],
            },
        ],
        "later_milestones": ["Next: learn from the first release."],
    }
    bound = inspect.signature(ow.commit_task_graph).bind(ctx, **kwargs)
    bound.apply_defaults()
    payload = dict(bound.arguments)
    payload.pop("ctx")
    authorized = ow.OwnerContext(
        actor=ctx.actor,
        profile=ctx.profile,
        session=ctx.session,
        authority=ow.OwnerProposalAuthority(
            actor=ctx.actor,
            profile=ctx.profile,
            session=ctx.session,
            conversation="raphael-owner-" + "a" * 32,
            response_id="resp_" + "b" * 32,
            operation="owner_task_graph_commit",
            idempotency_key=payload["idempotency_key"],
            payload_digest=ow._digest(payload),
        ),
    )
    approval.register_gateway_notify(ctx.session, lambda data: None)
    approver = threading.Thread(target=_auto_approve, args=(ctx.session,))
    approver.start()
    result = ow.commit_task_graph(authorized, **kwargs)
    approver.join()
    return result


def _add_tasks(
    project, count, *, status, created_start, priorities=(0,), parents=(),
):
    """Create ``count`` work tasks with explicit status and ascending ages."""
    with kanban_db.connect(board=project["board"]) as conn:
        ids = [
            kanban_db.create_task(
                conn,
                title=f"{status} work {created_start}-{index}",
                project_id=project["project_id"],
                parents=parents,
            )
            for index in range(count)
        ]
        with kanban_db.write_txn(conn):
            for index, task_id in enumerate(ids):
                conn.execute(
                    "UPDATE tasks SET status = ?, created_at = ?, priority = ? "
                    "WHERE id = ?",
                    (
                        status,
                        created_start + index,
                        priorities[index % len(priorities)],
                        task_id,
                    ),
                )
                conn.execute(
                    "UPDATE task_events SET created_at = ? WHERE task_id = ?",
                    (created_start + index, task_id),
                )
    return ids


def _work_rows(project):
    with kanban_db.connect(board=project["board"]) as conn:
        return [
            dict(row)
            for row in conn.execute(
                "SELECT id, status, priority, created_at FROM tasks "
                "WHERE project_id = ? AND task_kind = 'work' "
                "AND status != 'archived'",
                (project["project_id"],),
            )
        ]


def _carried(snapshot):
    return {
        column["name"]: [task["id"] for task in column["tasks"]]
        for column in snapshot["columns"]
    }


def _order_key(row):
    return (-int(row["priority"]), int(row["created_at"]), str(row["id"]))


def _assert_window_order(project, snapshot):
    """Every column lists its carried tasks by priority DESC, created_at, id."""
    by_id = {row["id"]: row for row in _work_rows(project)}
    for name, ids in _carried(snapshot).items():
        assert all(by_id[task_id]["status"] == name for task_id in ids)
        assert ids == sorted(ids, key=lambda task_id: _order_key(by_id[task_id]))


def _snapshot(ctx, project):
    return ow.read_project_snapshot(ctx, project["project_slug"])


def _all_carried(snapshot):
    return [task_id for ids in _carried(snapshot).values() for task_id in ids]


def test_over_limit_window_carries_every_unfinished_task(ctx):
    project = _new_project(ctx)
    finished = _add_tasks(project, 205, status="done", created_start=1_000)
    running = _add_tasks(project, 1, status="running", created_start=5_000_000)
    waiting = _add_tasks(project, 1, status="ready", created_start=5_000_100)

    snapshot = _snapshot(ctx, project)

    carried = set(_all_carried(snapshot))
    assert running[0] in carried
    assert waiting[0] in carried
    assert snapshot["truncated"]["tasks"] is True
    assert not carried & set(finished[:1])


def test_over_limit_window_keeps_the_newest_finished_tasks(ctx):
    project = _new_project(ctx)
    finished = _add_tasks(project, 205, status="done", created_start=1_000)
    _add_tasks(project, 1, status="running", created_start=5_000_000)
    _add_tasks(project, 1, status="ready", created_start=5_000_100)
    unfinished = [r for r in _work_rows(project) if r["status"] != "done"]

    snapshot = _snapshot(ctx, project)

    carried = _all_carried(snapshot)
    assert len(carried) == _LIMIT
    kept_finished = _LIMIT - len(unfinished)
    carried_done = set(_carried(snapshot)["done"])
    assert carried_done == set(finished[-kept_finished:])
    assert finished[0] not in carried_done
    assert finished[-1] in carried_done


def test_over_limit_window_keeps_linked_tasks_ahead_of_unrelated_finished(ctx):
    project = _new_project(ctx)
    priorities = (0, 2, 1)
    finished = _add_tasks(
        project, 205, status="done", created_start=1_000, priorities=priorities,
    )
    linked_parent = finished[0]
    child = _add_tasks(
        project, 1, status="running", created_start=5_000_000,
        parents=(linked_parent,),
    )[0]
    archived_child = _add_tasks(
        project, 1, status="archived", created_start=5_000_200, parents=(child,),
    )[0]
    unfinished = [r for r in _work_rows(project) if r["status"] != "done"]

    snapshot = _snapshot(ctx, project)

    carried = _all_carried(snapshot)
    assert len(carried) == _LIMIT
    assert child in carried
    assert linked_parent in carried
    assert finished[1] not in carried
    assert archived_child not in carried
    kept_unrelated = _LIMIT - len(unfinished) - 1
    assert set(carried) >= set(finished[-kept_unrelated:])
    assert snapshot["truncated"]["tasks"] is True
    _assert_window_order(project, snapshot)


def test_under_limit_project_carries_every_task_in_todays_order(ctx):
    project = _new_project(ctx)
    _add_tasks(
        project, 40, status="ready", created_start=2_000,
        priorities=(0, 5, 1, 5),
    )
    _add_tasks(project, 20, status="done", created_start=3_000, priorities=(1, 0))

    snapshot = _snapshot(ctx, project)

    expected = {row["id"] for row in _work_rows(project)}
    assert set(_all_carried(snapshot)) == expected
    assert len(_all_carried(snapshot)) == len(expected)
    assert snapshot["truncated"]["tasks"] is False
    _assert_window_order(project, snapshot)


def test_project_with_exactly_the_limit_behaves_as_today(ctx):
    project = _new_project(ctx)
    base = len(_work_rows(project))
    _add_tasks(
        project, _LIMIT - base, status="done", created_start=1_000,
        priorities=(0, 3, 1),
    )
    archived = _add_tasks(project, 3, status="archived", created_start=9_000)
    assert len(_work_rows(project)) == _LIMIT

    snapshot = _snapshot(ctx, project)

    carried = _all_carried(snapshot)
    assert set(carried) == {row["id"] for row in _work_rows(project)}
    assert len(carried) == _LIMIT
    assert not set(carried) & set(archived)
    assert snapshot["truncated"]["tasks"] is False
    _assert_window_order(project, snapshot)


def test_snapshot_keys_are_unchanged(ctx):
    project = _new_project(ctx)
    snapshots = [_snapshot(ctx, project)]
    _add_tasks(project, 205, status="done", created_start=1_000)
    snapshots.append(_snapshot(ctx, project))

    for snapshot in snapshots:
        assert set(snapshot) == _TOP_KEYS
        assert set(snapshot["board"]) == _BOARD_KEYS
        assert set(snapshot["truncated"]) == _TRUNCATED_KEYS
        tasks = [t for column in snapshot["columns"] for t in column["tasks"]]
        assert tasks
        assert all(set(task) == _TASK_KEYS for task in tasks)
