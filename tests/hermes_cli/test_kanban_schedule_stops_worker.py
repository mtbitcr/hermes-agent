"""Scheduling a running card from outside stops its worker (finding 97).

``schedule_task`` moves a running card to ``scheduled``, clears its claim and
closes its run. When the caller is not the run's own worker, the worker is
stopped too, after the commit, through the same start-time checked signal
path the reclaim paths use.

The worker is a REAL sleeping child process, bound to the claim by the real
``_set_worker_pid``; ``live_child`` kills and reaps every child it starts.
The recycled-pid state pairs a live pid with the start time of a process that
has really exited, as in the worker identity suites.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

_WORKTREE = Path(__file__).resolve().parents[2]
if str(_WORKTREE) not in sys.path:
    sys.path.insert(0, str(_WORKTREE))

from hermes_cli import kanban_db as kb
from tests.hermes_cli._kanban_fence_support import create_fenced_board, ready_task
from tests.hermes_cli._kanban_worker_identity_support import (
    dead_child_identity,
    events_for,
    live_child,
    pid_is_alive,
    process_start_time,
    rebind_recorded_start_time,
    run_identity,
    task_identity,
)

SLUG = "schedule-stop"


@pytest.fixture
def board(fence_home):
    create_fenced_board(SLUG)
    return SLUG


@pytest.fixture
def conn(board):
    connection = kb.connect(board=board)
    try:
        yield connection
    finally:
        connection.close()


def db_path() -> Path:
    return kb.kanban_db_path(board=SLUG)


def _running_card(conn, worker_pid: int) -> tuple[str, int]:
    """A claimed card whose recorded worker is ``worker_pid``: ``(task_id, run_id)``."""
    task_id = ready_task(conn)
    claimed = kb.claim_task(conn, task_id)
    assert claimed is not None
    kb._set_worker_pid(conn, task_id, worker_pid)
    return task_id, claimed.current_run_id


def test_scheduling_a_running_card_from_outside_stops_its_worker(board, conn):
    with live_child() as worker:
        task_id, run_id = _running_card(conn, worker.pid)

        assert kb.schedule_task(conn, task_id, reason="after the review") is True

        try:
            worker.wait(timeout=10)
        except subprocess.TimeoutExpired:
            pytest.fail("the worker process is still running after the schedule")

    card = task_identity(db_path(), task_id)
    assert card["status"] == "scheduled"
    assert card["claim_lock"] is None
    assert card["worker_pid"] is None
    run = run_identity(db_path(), run_id)
    assert run["ended_at"] is not None
    assert run["outcome"] == "scheduled"


def test_a_worker_scheduling_its_own_card_is_not_signalled(board, conn):
    with live_child() as worker:
        task_id, run_id = _running_card(conn, worker.pid)

        assert kb.schedule_task(
            conn, task_id, reason="after the review", expected_run_id=run_id,
        ) is True

        assert pid_is_alive(worker.pid), "the worker was signalled for its own schedule"
        assert task_identity(db_path(), task_id)["status"] == "scheduled"


def test_a_recorded_pid_now_worn_by_another_process_is_not_signalled(board, conn):
    with live_child() as stranger:
        task_id, _run_id = _running_card(conn, stranger.pid)
        _gone_pid, gone_start = dead_child_identity(
            distinct_from=process_start_time(stranger.pid)
        )
        rebind_recorded_start_time(conn, task_id, gone_start)

        assert kb.schedule_task(conn, task_id, reason="after the review") is True

        assert pid_is_alive(stranger.pid), "a process wearing a recycled pid was signalled"
        assert task_identity(db_path(), task_id)["status"] == "scheduled"


def test_a_ready_card_that_was_never_claimed_schedules_as_before(board, conn):
    task_id = ready_task(conn)

    assert kb.schedule_task(conn, task_id, reason="next week") is True

    card = task_identity(db_path(), task_id)
    assert card["status"] == "scheduled"
    assert card["claim_lock"] is None
    assert card["worker_pid"] is None
    assert events_for(db_path(), task_id, "scheduled") == [
        ("scheduled", {"reason": "next week"}),
    ]
