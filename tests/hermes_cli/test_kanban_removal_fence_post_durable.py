"""Post-durable follow-up must never surface a fence refusal (Finding 1).

Once a mutation's transaction has committed, the caller must receive the
durable result. If the fence closes in the window BETWEEN the commit and
the post-commit bookkeeping (ready recompute, workspace cleanup), the
follow-up is abandoned and the durable result is still returned.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from unittest import mock

import pytest

_WORKTREE = Path(__file__).resolve().parents[2]
if str(_WORKTREE) not in sys.path:
    sys.path.insert(0, str(_WORKTREE))

from hermes_cli import kanban_db as kb
from hermes_cli import projects_db
from tests.hermes_cli._kanban_fence_support import (
    close_fence,
    create_fenced_board,
    read_only,
    ready_task,
    task_row,
)


def test_complete_task_returns_true_when_fence_closes_during_post_commit(fence_home):
    """complete_task: the primary commit succeeded, so True is returned even
    when the fence closes before the follow-up completes."""
    create_fenced_board("test-board")
    conn = kb.connect(board="test-board")
    try:
        task_id = ready_task(conn, "work-task")
        kb.claim_task(conn, task_id, claimer="worker")

        # Monkeypatch recompute_ready to close the fence before it runs.
        # The mutation already committed; the fence closing now must not
        # turn the durable success into a refusal.
        original_recompute = kb.recompute_ready
        def close_then_recompute(conn):
            close_fence("test-board")
            return original_recompute(conn)

        with mock.patch.object(kb, 'recompute_ready', side_effect=close_then_recompute):
            result = kb.complete_task(conn, task_id, result="done")

        # The caller received True, not a BoardFenceClosedError.
        assert result is True
        # The task really is done on disk.
        row = task_row(kb.kanban_db_path(board="test-board"), task_id)
        assert row is not None
        assert row["status"] == "done"
    finally:
        conn.close()


def test_archive_task_returns_true_when_fence_closes_during_post_commit(fence_home):
    """archive_task: the primary commit succeeded, so True is returned even
    when the fence closes before the follow-up completes."""
    create_fenced_board("test-board")
    conn = kb.connect(board="test-board")
    try:
        task_id = ready_task(conn, "work-task")

        # Monkeypatch recompute_ready to close the fence before it runs.
        original_recompute = kb.recompute_ready
        def close_then_recompute(conn):
            close_fence("test-board")
            return original_recompute(conn)

        with mock.patch.object(kb, 'recompute_ready', side_effect=close_then_recompute):
            result = kb.archive_task(conn, task_id)

        assert result is True
        row = task_row(kb.kanban_db_path(board="test-board"), task_id)
        assert row is not None
        assert row["status"] == "archived"
    finally:
        conn.close()


def test_delete_task_returns_true_when_fence_closes_during_post_commit(fence_home):
    """delete_task: the primary commit succeeded, so True is returned even
    when the fence closes before the follow-up completes."""
    create_fenced_board("test-board")
    conn = kb.connect(board="test-board")
    try:
        task_id = ready_task(conn, "work-task")

        # Monkeypatch recompute_ready to close the fence before it runs.
        original_recompute = kb.recompute_ready
        def close_then_recompute(conn):
            close_fence("test-board")
            return original_recompute(conn)

        with mock.patch.object(kb, 'recompute_ready', side_effect=close_then_recompute):
            result = kb.delete_task(conn, task_id)

        assert result is True
        # The row really is gone.
        row = task_row(kb.kanban_db_path(board="test-board"), task_id)
        assert row is None
    finally:
        conn.close()


def test_complete_task_refuses_when_fence_already_closed_before_call(fence_home):
    """The guard must NOT swallow a refusal when the fence was already
    closed BEFORE the call — that must still refuse with REFUSED_CLOSED."""
    create_fenced_board("test-board")
    conn = kb.connect(board="test-board")
    try:
        task_id = ready_task(conn, "work-task")
        kb.claim_task(conn, task_id, claimer="worker")
        close_fence("test-board")

        with pytest.raises(kb.BoardFenceClosedError) as exc_info:
            kb.complete_task(conn, task_id, result="done")

        assert exc_info.value.refusal.outcome == kb.FenceOutcome.REFUSED_CLOSED
        # The task was NOT completed.
        row = task_row(kb.kanban_db_path(board="test-board"), task_id)
        assert row["status"] == "running"
    finally:
        conn.close()


def test_post_durable_guard_does_not_swallow_non_fence_exceptions(fence_home):
    """Non-fence exceptions raised inside the post-commit tail must still
    propagate unchanged."""
    create_fenced_board("test-board")
    conn = kb.connect(board="test-board")
    try:
        task_id = ready_task(conn, "work-task")

        # Monkeypatch recompute_ready to raise a non-fence exception.
        def raise_runtime_error(conn):
            raise RuntimeError("synthetic non-fence error")

        with mock.patch.object(kb, 'recompute_ready', side_effect=raise_runtime_error):
            with pytest.raises(RuntimeError, match="synthetic non-fence error"):
                kb.archive_task(conn, task_id)
    finally:
        conn.close()


def test_activate_owner_work_returns_list_when_fence_closes_during_post_commit(fence_home):
    """activate_owner_work: the primary commit succeeded, so the released list
    is returned even when the fence closes before the follow-up completes."""
    create_fenced_board("test-board")
    conn = kb.connect(board="test-board")
    try:
        # Create a task and park it with a generation.
        task_id = kb.create_task(conn, title="parked-task", assignee="worker")
        generation = "test-gen-001"
        with kb.write_txn(conn):
            conn.execute(
                "UPDATE tasks SET status = 'scheduled', park_generation = ? WHERE id = ?",
                (generation, task_id),
            )

        # Monkeypatch recompute_ready to close the fence before it runs.
        original_recompute = kb.recompute_ready
        def close_then_recompute(conn):
            close_fence("test-board")
            return original_recompute(conn)

        with mock.patch.object(kb, 'recompute_ready', side_effect=close_then_recompute):
            result = kb.activate_owner_work(conn, [task_id], generation=generation)

        # The caller received the list of released tasks.
        assert result == [task_id]
        # The task really is activated on disk (status changed to todo).
        db_path = kb.kanban_db_path(board="test-board")
        with read_only(db_path) as ro_conn:
            row = ro_conn.execute(
                "SELECT status, park_generation FROM tasks WHERE id = ?",
                (task_id,),
            ).fetchone()
        assert row is not None
        assert row["status"] == "todo"
        assert row["park_generation"] is None
    finally:
        conn.close()


def test_unlink_tasks_returns_true_when_fence_closes_during_post_commit(fence_home):
    """unlink_tasks: the primary commit succeeded, so True is returned even
    when the fence closes before the follow-up completes."""
    create_fenced_board("test-board")
    conn = kb.connect(board="test-board")
    try:
        parent_id = kb.create_task(conn, title="parent", assignee="worker")
        child_id = kb.create_task(conn, title="child", assignee="worker")
        kb.link_tasks(conn, parent_id, child_id)

        # Monkeypatch recompute_ready to close the fence before it runs.
        original_recompute = kb.recompute_ready
        def close_then_recompute(conn):
            close_fence("test-board")
            return original_recompute(conn)

        with mock.patch.object(kb, 'recompute_ready', side_effect=close_then_recompute):
            result = kb.unlink_tasks(conn, parent_id, child_id)

        assert result is True
        # The link really is removed on disk.
        db_path = kb.kanban_db_path(board="test-board")
        with read_only(db_path) as ro_conn:
            link_row = ro_conn.execute(
                "SELECT 1 FROM task_links WHERE parent_id = ? AND child_id = ?",
                (parent_id, child_id),
            ).fetchone()
        assert link_row is None
    finally:
        conn.close()


def test_specify_triage_task_returns_true_when_fence_closes_during_post_commit(fence_home):
    """specify_triage_task: the primary commit succeeded, so True is returned
    even when the fence closes before the follow-up completes."""
    create_fenced_board("test-board")
    conn = kb.connect(board="test-board")
    try:
        # Create a triage task.
        task_id = kb.create_task(conn, title="triage-task", assignee="worker")
        with kb.write_txn(conn):
            conn.execute(
                "UPDATE tasks SET status = 'triage' WHERE id = ?",
                (task_id,),
            )

        # Monkeypatch recompute_ready to close the fence before it runs.
        original_recompute = kb.recompute_ready
        def close_then_recompute(conn):
            close_fence("test-board")
            return original_recompute(conn)

        with mock.patch.object(kb, 'recompute_ready', side_effect=close_then_recompute):
            result = kb.specify_triage_task(
                conn, task_id, title="specified-title", assignee="worker"
            )

        assert result is True
        # The task really is specified on disk (status changed to todo, title updated).
        db_path = kb.kanban_db_path(board="test-board")
        with read_only(db_path) as ro_conn:
            row = ro_conn.execute(
                "SELECT status, title FROM tasks WHERE id = ?",
                (task_id,),
            ).fetchone()
        assert row is not None
        assert row["status"] == "todo"
        assert row["title"] == "specified-title"
    finally:
        conn.close()


def test_decompose_triage_task_returns_child_ids_when_fence_closes_during_post_commit(fence_home):
    """decompose_triage_task: the primary commit succeeded, so the child_ids
    list is returned even when the fence closes before the follow-up completes."""
    create_fenced_board("test-board")
    conn = kb.connect(board="test-board")
    try:
        # Create a triage task.
        task_id = kb.create_task(conn, title="triage-root", assignee="orchestrator")
        with kb.write_txn(conn):
            conn.execute(
                "UPDATE tasks SET status = 'triage' WHERE id = ?",
                (task_id,),
            )

        children = [
            {"title": "child-1", "assignee": "worker"},
            {"title": "child-2", "assignee": "worker"},
        ]

        # Monkeypatch recompute_ready to close the fence before it runs.
        original_recompute = kb.recompute_ready
        def close_then_recompute(conn):
            close_fence("test-board")
            return original_recompute(conn)

        with mock.patch.object(kb, 'recompute_ready', side_effect=close_then_recompute):
            result = kb.decompose_triage_task(
                conn, task_id, root_assignee="orchestrator", children=children, auto_promote=True
            )

        # The caller received the list of child IDs.
        assert result is not None
        assert len(result) == 2
        # The children really exist on disk.
        db_path = kb.kanban_db_path(board="test-board")
        with read_only(db_path) as ro_conn:
            for child_id in result:
                row = ro_conn.execute(
                    "SELECT title FROM tasks WHERE id = ?",
                    (child_id,),
                ).fetchone()
                assert row is not None
                assert row["title"] in ("child-1", "child-2")
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Finding 1 — three more post-durable follow-ups the original audit missed.
# ---------------------------------------------------------------------------


def test_reclaim_task_returns_true_when_fence_closes_during_post_commit(fence_home):
    """reclaim_task: the reclaim transaction already committed, so True is
    returned even when the fence closes before _clear_failure_counter runs."""
    create_fenced_board("test-board")
    conn = kb.connect(board="test-board")
    try:
        task_id = ready_task(conn, "work-task")
        kb.claim_task(conn, task_id)

        # Wrap _clear_failure_counter, the first (and only) function the
        # post-commit tail calls, to close the fence right before it runs.
        original_clear = kb._clear_failure_counter
        def close_then_clear(*args, **kwargs):
            close_fence("test-board")
            return original_clear(*args, **kwargs)

        with mock.patch.object(kb, '_clear_failure_counter', side_effect=close_then_clear):
            result = kb.reclaim_task(conn, task_id, reason="test", signal_fn=lambda *_a: None)

        # The caller received True, not a BoardFenceClosedError.
        assert result is True
        # The reclaim really is durable on disk.
        row = task_row(kb.kanban_db_path(board="test-board"), task_id)
        assert row is not None
        assert row["status"] == "ready"
        assert row["claim_lock"] is None
    finally:
        conn.close()


def test_enforce_max_runtime_returns_list_when_fence_closes_during_post_commit(fence_home):
    """enforce_max_runtime: the requeue transaction already committed, so the
    timed_out list is returned even when the fence closes before
    _record_task_failure runs."""
    create_fenced_board("test-board")
    conn = kb.connect(board="test-board")
    try:
        task_id = ready_task(conn, "work-task")
        with kb.write_txn(conn):
            conn.execute(
                "UPDATE tasks SET max_runtime_seconds = 1 WHERE id = ?", (task_id,),
            )
        kb.claim_task(conn, task_id)
        kb._set_worker_pid(conn, task_id, os.getpid())
        old_started = int(time.time()) - 30
        with kb.write_txn(conn):
            conn.execute(
                "UPDATE tasks SET started_at = ? WHERE id = ?", (old_started, task_id),
            )
            conn.execute(
                "UPDATE task_runs SET started_at = ? "
                "WHERE id = (SELECT current_run_id FROM tasks WHERE id = ?)",
                (old_started, task_id),
            )

        # Wrap _record_task_failure, the follow-up the requeued-per-task
        # branch calls, to close the fence right before it runs.
        original_record = kb._record_task_failure
        def close_then_record(*args, **kwargs):
            close_fence("test-board")
            return original_record(*args, **kwargs)

        with mock.patch.object(kb, '_pid_alive', lambda pid: False), \
             mock.patch.object(kb, '_record_task_failure', side_effect=close_then_record):
            timed_out = kb.enforce_max_runtime(conn, signal_fn=lambda *_a: None)

        # The caller received the durable timed_out list.
        assert timed_out == [task_id]
        # The task really is requeued on disk.
        db_path = kb.kanban_db_path(board="test-board")
        row = task_row(db_path, task_id)
        assert row is not None
        assert row["status"] == "ready"
        # ... and its timed_out event really exists on disk.
        with read_only(db_path) as ro_conn:
            ev = ro_conn.execute(
                "SELECT 1 FROM task_events WHERE task_id = ? AND kind = 'timed_out'",
                (task_id,),
            ).fetchone()
        assert ev is not None
    finally:
        conn.close()


def test_detect_crashed_workers_returns_list_when_fence_closes_during_post_commit(
    fence_home, monkeypatch,
):
    """detect_crashed_workers: the reclaim transaction already committed, so
    the crashed list is returned even when the fence closes before
    _record_task_failure runs."""
    monkeypatch.setenv("HERMES_KANBAN_CRASH_GRACE_SECONDS", "0")
    create_fenced_board("test-board")
    conn = kb.connect(board="test-board")
    try:
        task_id = ready_task(conn, "work-task")
        kb.claim_task(conn, task_id)
        kb._set_worker_pid(conn, task_id, 98765)

        # Wrap _record_task_failure, the follow-up the ordinary
        # (non-protocol-violation) crash branch calls, to close the fence
        # right before it runs.
        original_record = kb._record_task_failure
        def close_then_record(*args, **kwargs):
            close_fence("test-board")
            return original_record(*args, **kwargs)

        with mock.patch.object(kb, '_pid_alive', lambda pid: False), \
             mock.patch.object(kb, '_record_task_failure', side_effect=close_then_record):
            crashed = kb.detect_crashed_workers(conn)

        # The caller received the durable crashed list.
        assert crashed == [task_id]
        # The task really is requeued on disk.
        db_path = kb.kanban_db_path(board="test-board")
        row = task_row(db_path, task_id)
        assert row is not None
        assert row["status"] == "ready"
        # ... and its crashed event really exists on disk.
        with read_only(db_path) as ro_conn:
            ev = ro_conn.execute(
                "SELECT 1 FROM task_events WHERE task_id = ? AND kind = 'crashed'",
                (task_id,),
            ).fetchone()
        assert ev is not None
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Finding 2a — apply_owner_project_plan's post-commit tail had no coverage.
# ---------------------------------------------------------------------------


def _make_real_project(name: str) -> str:
    """A real Project row (Steward plans require a project id that resolves;
    an unresolvable id is silently dropped by ``create_task``)."""
    with projects_db.connect_closing() as pconn:
        return projects_db.create_project(pconn, name=name, board_slug="test-board")


def _make_project_anchor(conn, project_id: str) -> str:
    return kb.create_task(
        conn, title="anchor", assignee=None, control=True, project_id=project_id,
    )


def test_apply_owner_project_plan_ready_recompute_arm_survives_fence_close(fence_home):
    """apply_owner_project_plan (parked=False, no archived tasks): the plan
    transaction already committed, so the receipt is returned even when the
    fence closes before the ready-recompute half of the tail runs."""
    create_fenced_board("test-board")
    conn = kb.connect(board="test-board")
    try:
        project_id = _make_real_project("Ready Arm Project")
        anchor_id = _make_project_anchor(conn, project_id)
        task_id = kb.create_task(
            conn, title="work", assignee="worker", project_id=project_id,
        )
        with kb.write_txn(conn):
            conn.execute("UPDATE tasks SET status = 'blocked' WHERE id = ?", (task_id,))
        revision = kb.task_event_revision(conn, task_id)

        changes = [{
            "action": "move",
            "reason": "unblocking",
            "target": {
                "task_id": task_id,
                "expected_status": "blocked",
                "expected_revision": revision,
            },
            "to_status": "todo",
        }]

        original_recompute = kb.recompute_ready
        def close_then_recompute(conn):
            close_fence("test-board")
            return original_recompute(conn)

        with mock.patch.object(kb, 'recompute_ready', side_effect=close_then_recompute):
            result = kb.apply_owner_project_plan(
                conn,
                project_id=project_id,
                anchor_task_id=anchor_id,
                changes=changes,
                actor="owner",
                profile="owner-profile",
                idempotency_key="idem-ready-arm",
                request_digest="digest-ready-arm",
                trigger="test",
                plan_summary="unblock one task",
                current_milestone="m1",
                later_milestones=[],
                board="test-board",
                parked=False,
            )

        # The caller received the durable receipt, not a refusal.
        assert result["applied"] is True
        db_path = kb.kanban_db_path(board="test-board")
        row = task_row(db_path, task_id)
        assert row is not None
        assert row["status"] == "todo"
    finally:
        conn.close()


def test_apply_owner_project_plan_archived_cleanup_arm_survives_fence_close(fence_home):
    """apply_owner_project_plan (an archiving change): the plan transaction
    already committed, so the receipt is returned even when the fence closes
    before the archived-cleanup half of the tail runs."""
    create_fenced_board("test-board")
    conn = kb.connect(board="test-board")
    try:
        project_id = _make_real_project("Archive Arm Project")
        anchor_id = _make_project_anchor(conn, project_id)
        task_id = kb.create_task(
            conn, title="work", assignee="worker", project_id=project_id,
        )
        revision = kb.task_event_revision(conn, task_id)

        changes = [{
            "action": "archive",
            "reason": "superseded",
            "target": {
                "task_id": task_id,
                "expected_status": "ready",
                "expected_revision": revision,
            },
        }]

        original_cleanup = kb._cleanup_workspace
        def close_then_cleanup(conn, task_id):
            close_fence("test-board")
            return original_cleanup(conn, task_id)

        with mock.patch.object(kb, '_cleanup_workspace', side_effect=close_then_cleanup):
            result = kb.apply_owner_project_plan(
                conn,
                project_id=project_id,
                anchor_task_id=anchor_id,
                changes=changes,
                actor="owner",
                profile="owner-profile",
                idempotency_key="idem-archive-arm",
                request_digest="digest-archive-arm",
                trigger="test",
                plan_summary="archive one task",
                current_milestone="m1",
                later_milestones=[],
                board="test-board",
            )

        # The caller received the durable receipt, not a refusal.
        assert result["applied"] is True
        assert result["affected_task_ids"] == [task_id]
        db_path = kb.kanban_db_path(board="test-board")
        row = task_row(db_path, task_id)
        assert row is not None
        assert row["status"] == "archived"
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Finding 2b — post-durable coverage never exercised REFUSED_INDETERMINATE.
# ---------------------------------------------------------------------------


def _make_refusal(outcome: "kb.FenceOutcome") -> "kb.FenceRefusal":
    return kb.FenceRefusal(
        outcome=outcome,
        rule=kb.FenceRefusalRule.CLOSED,
        board="test-board",
        message="synthetic refusal for _after_durable_commit coverage",
    )


@pytest.mark.parametrize(
    "outcome",
    [
        kb.FenceOutcome.REFUSED_CLOSED,
        kb.FenceOutcome.REFUSED_INDETERMINATE,
        kb.FenceOutcome.REFUSED_TIMEOUT,
    ],
)
def test_after_durable_commit_swallows_every_refusal_outcome(outcome):
    """Every refusal outcome — closed, indeterminate, timeout — is swallowed
    by the post-durable guard; none may surface to the caller."""
    with kb._after_durable_commit("synthetic"):
        raise kb.BoardFenceClosedError(_make_refusal(outcome))
    # Reaching here means the guard swallowed the refusal, as required.


def test_after_durable_commit_still_propagates_non_fence_exceptions():
    """A non-fence, non-deadline exception inside the tail must still
    propagate unchanged, regardless of the fence outcome taxonomy."""
    with pytest.raises(RuntimeError, match="synthetic non-fence error"):
        with kb._after_durable_commit("synthetic"):
            raise RuntimeError("synthetic non-fence error")
