"""A recycled pid is not the owner.

The OS reuses pid numbers. A claim that records only a number therefore
reads as "the owner is alive" the moment an unrelated process inherits
that number — the claim is held forever by a stranger, and the reclaim
paths, if they ever did act, would signal that stranger. The identity
witness recorded beside the pid is what tells the two apart.

How the state is built here, honestly: a REAL child process is started and
its REAL start time is observed while it lives; that child is then killed
and reaped. A SECOND real child is started and its pid is bound to the
claim by the REAL ``_set_worker_pid``. The recorded witness is then
replaced with the one observed from the first, now-dead process. Both
numbers are genuine observations of real processes; only their pairing is
arranged, because which pid the OS recycles and when is not ours to
choose. Every decision taken on the resulting row — the classification,
the release, the audit event — is the shipped code's.

What this suite holds: the stale claim IS released, it is released ONLY
through the reclaim path that already exists, the event that path already
emits carries the evidence (pid, recorded start time, observed start time,
verdict), and the unrelated process wearing the recycled number is never
signalled.
"""

from __future__ import annotations

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
    event_rows,
    events_for,
    expire_claim,
    live_child,
    pid_is_alive,
    process_start_time,
    rebind_recorded_start_time,
    rewind_started_at,
    runs_for,
    task_identity,
)


@pytest.fixture
def board(fence_home, monkeypatch):
    monkeypatch.setenv("HERMES_KANBAN_CRASH_GRACE_SECONDS", "0")
    create_fenced_board("reuse")
    return "reuse"


@pytest.fixture
def conn(board):
    connection = kb.connect(board=board)
    try:
        yield connection
    finally:
        connection.close()


def db_path(board: str) -> Path:
    return kb.kanban_db_path(board=board)


def _claim_bound_to_a_recycled_pid(conn, live_pid: int):
    """One claim naming a LIVE pid whose recorded identity is someone else's."""
    _gone_pid, gone_start = dead_child_identity(
        distinct_from=process_start_time(live_pid)
    )
    assert gone_start != process_start_time(live_pid), (
        "the two real processes must have distinguishable start times"
    )
    task_id = ready_task(conn)
    assert kb.claim_task(conn, task_id) is not None
    kb._set_worker_pid(conn, task_id, live_pid)
    rebind_recorded_start_time(conn, task_id, gone_start)
    return task_id, gone_start


# ---------------------------------------------------------------------------
# The stale-claim reclaim
# ---------------------------------------------------------------------------

def test_a_recycled_pid_is_reclaimed_and_the_stranger_is_left_alone(board, conn):
    with live_child() as stranger:
        task_id, gone_start = _claim_bound_to_a_recycled_pid(conn, stranger.pid)
        observed = process_start_time(stranger.pid)

        expire_claim(conn, task_id)
        assert kb.release_stale_claims(conn) == 1, (
            "a recycled pid is not the owner: the claim must be reclaimed"
        )

        after = task_identity(db_path(board), task_id)
        assert after["status"] == "ready"
        assert after["claim_lock"] is None
        assert after["worker_pid"] is None

        assert pid_is_alive(stranger.pid), (
            "the unrelated process wearing the recycled pid must not be signalled"
        )

    reclaimed = events_for(db_path(board), task_id, "reclaimed")
    assert len(reclaimed) == 1
    payload = reclaimed[0][1]
    # The evidence, not just the verdict.
    assert payload["worker_pid"] == stranger.pid
    assert payload["worker_start_time"] == gone_start
    assert payload["observed_worker_start_time"] == observed
    assert payload["worker_presence"] == kb.HolderPresence.PROVABLY_ABSENT.value
    assert "recycled" in payload["worker_identity_reason"]
    assert payload["pid_reused"] is True
    assert payload["termination_attempted"] is False
    # Still the ordinary automatic stale-lock release, marked as such.
    assert payload["automatic"] is True
    assert payload["stale_lock"] is not None


def test_the_release_comes_only_from_the_existing_reclaim_path(board, conn):
    """No second releasing path: exactly one claim-ending event, bound to the
    run that reclaim closed."""
    with live_child() as stranger:
        task_id, gone_start = _claim_bound_to_a_recycled_pid(conn, stranger.pid)
        before = {row["kind"] for row in event_rows(db_path(board), task_id)}
        assert "reclaimed" not in before

        expire_claim(conn, task_id)
        assert kb.release_stale_claims(conn) == 1

    rows = event_rows(db_path(board), task_id)
    ending = [r for r in rows if r["kind"] in {"reclaimed", "crashed", "released",
                                               "reclaim_deferred", "gave_up"}]
    assert [r["kind"] for r in ending] == ["reclaimed"]

    runs = runs_for(db_path(board), task_id)
    assert len(runs) == 1
    assert runs[0]["outcome"] == "reclaimed"
    assert runs[0]["ended_at"] is not None
    # The run the reclaim closed is the run the audit event is bound to.
    assert ending[0]["run_id"] == runs[0]["id"]
    # The closed run still carries the witness this attempt was bound with,
    # so the history says which process it believed it owned. (``_end_run``
    # clears the run's live claim fields, including its pid, which is why
    # the witness is what remains to read.)
    assert runs[0]["worker_start_time"] == gone_start


# ---------------------------------------------------------------------------
# The crash sweep reaches the same verdict
# ---------------------------------------------------------------------------

def test_the_crash_sweep_does_not_mistake_a_recycled_pid_for_a_live_worker(
    board, conn,
):
    with live_child() as stranger:
        task_id, gone_start = _claim_bound_to_a_recycled_pid(conn, stranger.pid)
        observed = process_start_time(stranger.pid)
        rewind_started_at(conn, task_id)

        assert kb.detect_crashed_workers(conn) == [task_id], (
            "a pid that is alive but is NOT our worker must not read as live"
        )
        assert pid_is_alive(stranger.pid)

    after = task_identity(db_path(board), task_id)
    assert after["status"] in ("ready", "todo", "blocked")
    assert after["worker_pid"] is None

    crashed = events_for(db_path(board), task_id, "crashed")
    assert len(crashed) == 1
    payload = crashed[0][1]
    assert payload["pid"] == stranger.pid
    assert payload["worker_start_time"] == gone_start
    assert payload["observed_worker_start_time"] == observed
    assert payload["worker_presence"] == kb.HolderPresence.PROVABLY_ABSENT.value


# ---------------------------------------------------------------------------
# The control: a genuine owner is still the owner
# ---------------------------------------------------------------------------

def test_the_same_pid_with_its_own_start_time_is_still_the_owner(board, conn):
    """The reuse rule must fire on a mismatch and only on a mismatch."""
    with live_child() as child:
        task_id = ready_task(conn)
        assert kb.claim_task(conn, task_id) is not None
        kb._set_worker_pid(conn, task_id, child.pid)

        expire_claim(conn, task_id)
        assert kb.release_stale_claims(conn) == 0
        rewind_started_at(conn, task_id)
        assert kb.detect_crashed_workers(conn) == []

        held = task_identity(db_path(board), task_id)
        assert held["status"] == "running"
        assert held["worker_pid"] == child.pid
        assert held["worker_start_time"] == process_start_time(child.pid)
