"""Archiving a running card stops its worker, as scheduling one does.

``archive_task`` moves a card to ``archived``, clears its claim and worker pid
and closes its run as ``reclaimed``. When the card was running, its worker is
stopped too, after the commit, through the same start-time checked signal
path ``schedule_task`` uses.

The worker is a REAL sleeping child process, bound to the claim by the real
``_set_worker_pid``; ``live_child`` kills and reaps every child it starts.
The recycled-pid state pairs a live pid with the start time of a process that
has really exited, as in the worker identity suites. Signals are observed at
``os.kill`` itself and delivered unchanged; each one is stored with the card
and run as a separate read-only connection sees them at that moment, which
is committed state only.
"""

from __future__ import annotations

import contextlib
import os
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

SLUG = "archive-stop"


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


@contextlib.contextmanager
def signals_sent(task_id: str, run_id: int = None):
    """Every non-zero signal sent inside the block: ``[(pid, sig, card, run)]``.

    ``card`` and ``run`` are read through a separate read-only connection at
    the moment of the signal. Liveness probes (signal 0) are not recorded.
    Delivery goes to the ``os.kill`` already in place, so the suite-wide
    live-system guard still applies.
    """
    deliver = os.kill
    sent = []

    def recording_kill(pid, sig, *args, **kwargs):
        if int(sig) != 0:
            run = None if run_id is None else run_identity(db_path(), run_id)
            sent.append((int(pid), int(sig), task_identity(db_path(), task_id), run))
        return deliver(pid, sig, *args, **kwargs)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(os, "kill", recording_kill)
        yield sent


def test_archiving_a_running_card_stops_its_worker_after_the_commit(board, conn):
    with live_child() as worker:
        task_id, run_id = _running_card(conn, worker.pid)

        with signals_sent(task_id, run_id) as sent:
            assert kb.archive_task(conn, task_id) is True

        try:
            worker.wait(timeout=10)
        except subprocess.TimeoutExpired:
            pytest.fail("the worker process is still running after the archive")

    assert sent, "no signal was sent to the worker"
    assert {pid for pid, *_ in sent} == {worker.pid}
    for _pid, _sig, card_then, run_then in sent:
        assert card_then["status"] == "archived", "the worker was signalled before the archive committed"
        assert card_then["claim_lock"] is None
        assert card_then["worker_pid"] is None
        assert run_then["ended_at"] is not None
        assert run_then["outcome"] == "reclaimed"

    card = task_identity(db_path(), task_id)
    assert card["status"] == "archived"
    assert card["claim_lock"] is None
    assert card["worker_pid"] is None
    run = run_identity(db_path(), run_id)
    assert run["ended_at"] is not None
    assert run["outcome"] == "reclaimed"


def test_a_recorded_pid_now_worn_by_another_process_is_not_signalled(board, conn):
    with live_child() as stranger:
        task_id, run_id = _running_card(conn, stranger.pid)
        _gone_pid, gone_start = dead_child_identity(
            distinct_from=process_start_time(stranger.pid)
        )
        rebind_recorded_start_time(conn, task_id, gone_start)

        with signals_sent(task_id, run_id) as sent:
            assert kb.archive_task(conn, task_id) is True

        assert [(pid, sig) for pid, sig, *_ in sent] == [], "a process wearing a recycled pid was signalled"
        assert pid_is_alive(stranger.pid), "a process wearing a recycled pid was signalled"

    card = task_identity(db_path(), task_id)
    assert card["status"] == "archived"
    assert card["worker_pid"] is None
    run = run_identity(db_path(), run_id)
    assert run["ended_at"] is not None
    assert run["outcome"] == "reclaimed"


def test_archiving_a_card_that_is_not_running_signals_nothing(board, conn):
    task_id = ready_task(conn)

    with signals_sent(task_id) as sent:
        assert kb.archive_task(conn, task_id) is True

    assert [(pid, sig) for pid, sig, *_ in sent] == [], "archiving a card that was not running sent a signal"
    card = task_identity(db_path(), task_id)
    assert card["status"] == "archived"
    assert card["claim_lock"] is None
    assert card["worker_pid"] is None
    assert events_for(db_path(), task_id, "archived") == [("archived", {})]
