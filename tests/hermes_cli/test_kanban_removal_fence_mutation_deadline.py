"""The mutation deadline starts BEFORE the first read (Defect 3).

Reviewer reproduction: the single hard five-second mutation deadline was
created INSIDE ``write_txn``, but several mutators already perform
blocking board reads before they open their write transaction. Those
pre-reads ran under the ordinary (two-minute, independently configurable)
kanban busy timeout: under a real exclusive board-store lock,
``create_task(..., idempotency_key=...)`` blocked in its pre-transaction
SELECT for 6.210 s and raised a raw
``sqlite3.OperationalError: database is locked`` — past the hard ceiling
and outside the structured refusal contract.

Every case here holds a REAL ``BEGIN EXCLUSIVE`` on the board store from a
SECOND process, calls the REAL mutator, and asserts on the wall clock and
on the refusal's type.
"""

from __future__ import annotations

import sqlite3
import sys
import time
from pathlib import Path

import pytest

_WORKTREE = Path(__file__).resolve().parents[2]
if str(_WORKTREE) not in sys.path:
    sys.path.insert(0, str(_WORKTREE))

from hermes_cli import kanban_db as kb
from tests.hermes_cli._kanban_fence_support import (
    create_fenced_board,
    hold_exclusive_lock,
    ready_task,
    row_count,
    task_row,
)


CEILING = kb.MAX_FENCE_CONSULTATION_TIMEOUT_SECONDS
# Deliberately loose: the point of the bound is that the call cannot wait
# for the ordinary busy timeout (two minutes). The reviewer measured
# 6.210 s against a 5.0 s ceiling; anything under this is inside the
# fence's own budget plus generous CI slack.
UPPER = 15.0


@pytest.fixture
def running_board(fence_home):
    """A real fenced board carrying one ready and one running task."""
    create_fenced_board("deadline-preread")
    conn = kb.connect(board="deadline-preread")
    ready_id = ready_task(conn, title="ready work")
    running_id = ready_task(conn, title="running work")
    claimed = kb.claim_task(conn, running_id, claimer="host:worker")
    assert claimed is not None
    # Park the claim in the past so the stale/runtime sweeps have a
    # candidate to find in their pre-transaction query.
    with kb.write_txn(conn):
        conn.execute(
            "UPDATE tasks SET claim_expires = ?, started_at = ?, "
            "max_runtime_seconds = 1 WHERE id = ?",
            (int(time.time()) - 3600, int(time.time()) - 3600, running_id),
        )
    todo_id = kb.create_task(conn, title="todo work", assignee="worker")
    with kb.write_txn(conn):
        conn.execute("UPDATE tasks SET status = 'todo' WHERE id = ?", (todo_id,))
    return {
        "conn": conn,
        "ready": ready_id,
        "running": running_id,
        "todo": todo_id,
        "db_path": kb.kanban_db_path(board="deadline-preread"),
    }


# Every mutator the reviewer named, plus the runtime-enforcement /
# reconciliation sweeps with the same shape. Each entry is a callable
# taking the fixture dict.
PRE_READ_MUTATORS = {
    "create_task": lambda b: kb.create_task(
        b["conn"], title="blocked", assignee="worker", idempotency_key="idem-1"
    ),
    "claim_task": lambda b: kb.claim_task(b["conn"], b["ready"]),
    "release_stale_claims": lambda b: kb.release_stale_claims(
        b["conn"], signal_fn=lambda *_a: None
    ),
    "reclaim_task": lambda b: kb.reclaim_task(
        b["conn"], b["running"], signal_fn=lambda *_a: None
    ),
    "complete_task": lambda b: kb.complete_task(
        b["conn"], b["running"], result="done"
    ),
    "promote_task": lambda b: kb.promote_task(
        b["conn"], b["todo"], actor="operator"
    ),
    "enforce_max_runtime": lambda b: kb.enforce_max_runtime(
        b["conn"], signal_fn=lambda *_a: None
    ),
    "detect_stale_running": lambda b: kb.detect_stale_running(
        b["conn"], stale_timeout_seconds=1
    ),
    "reconcile_orphaned_running": lambda b: kb.reconcile_orphaned_running(b["conn"]),
    "detect_crashed_workers": lambda b: kb.detect_crashed_workers(b["conn"]),
    "heartbeat_claim": lambda b: kb.heartbeat_claim(
        b["conn"], b["running"], claimer="host:worker"
    ),
    "claim_review_task": lambda b: kb.claim_review_task(b["conn"], b["ready"]),
}


def _timed(call):
    start = time.monotonic()
    try:
        result = call()
    except kb.BoardFenceClosedError as exc:
        return time.monotonic() - start, exc, exc.refusal, None
    except BaseException as exc:  # noqa: BLE001 - the raw error IS the defect
        return time.monotonic() - start, exc, None, None
    return time.monotonic() - start, None, None, result


@pytest.mark.parametrize("name", sorted(PRE_READ_MUTATORS))
def test_defect_3_a_pre_read_under_a_real_exclusive_lock_refuses_structurally(
    running_board, name
):
    """A REAL second connection holds ``BEGIN EXCLUSIVE`` on the board store.

    The mutator's pre-transaction read is the first thing that blocks. It
    must come back inside the ceiling with the project's structured
    timeout refusal — never a raw ``sqlite3.OperationalError``.
    """
    board = running_board
    db_path = board["db_path"]
    before = {
        "tasks": row_count(db_path, "tasks"),
        "events": row_count(db_path, "task_events"),
        "ready": task_row(db_path, board["ready"])["status"],
        "running": task_row(db_path, board["running"])["status"],
    }

    with hold_exclusive_lock(db_path):
        elapsed, exc, refusal, result = _timed(
            lambda: PRE_READ_MUTATORS[name](board)
        )

    assert not isinstance(exc, sqlite3.OperationalError), (
        f"{name} surfaced a raw {exc!r} after {elapsed:.3f}s instead of the "
        "structured timeout refusal"
    )
    assert exc is not None, (
        f"{name} returned {result!r} after {elapsed:.3f}s instead of refusing "
        "while the board store was locked"
    )
    assert isinstance(exc, kb.BoardFenceClosedError), (
        f"{name} raised {type(exc).__name__}: {exc!r}"
    )
    assert refusal is not None
    assert refusal.outcome is kb.FenceOutcome.REFUSED_TIMEOUT, (
        f"{name} refused with {refusal.outcome!r} / {refusal.rule!r}: "
        f"{refusal.message}"
    )
    assert refusal.rule is kb.FenceRefusalRule.TIMEOUT
    assert elapsed < UPPER, (
        f"{name} waited {elapsed:.3f}s — past the {CEILING}s ceiling and well "
        "past any plausible allowance"
    )

    # Nothing landed.
    assert row_count(db_path, "tasks") == before["tasks"]
    assert row_count(db_path, "task_events") == before["events"]
    assert task_row(db_path, board["ready"])["status"] == before["ready"]
    assert task_row(db_path, board["running"])["status"] == before["running"]


def test_defect_3_the_deadline_is_open_before_the_first_read(running_board):
    """The budget is ticking by the time the mutator reads the board.

    Proved from inside the real call: the pre-transaction idempotency
    SELECT observes a bounded remaining budget, where before the repair it
    observed ``None`` (unbounded, the ordinary busy timeout).
    """
    board = running_board
    conn = board["conn"]
    seen: list = []

    def trace(sql: str) -> None:
        # SQLite hands every statement to the trace callback on this same
        # thread, so the budget observed here is the one the statement ran
        # under — no wrapper around the module under test.
        if sql.strip().startswith("SELECT id FROM tasks WHERE idempotency_key"):
            seen.append((sql, kb._fence_remaining_seconds()))

    conn.set_trace_callback(trace)
    try:
        kb.create_task(
            conn, title="probe", assignee="worker", idempotency_key="probe-key"
        )
    finally:
        conn.set_trace_callback(None)

    assert seen, "the idempotency pre-read never ran"
    _sql, remaining = seen[0]
    assert remaining is not None, (
        "the pre-transaction read ran with no deadline in force"
    )
    assert 0 < remaining <= CEILING


def test_defect_3_the_pre_read_and_the_transaction_share_one_budget(running_board):
    """One mutation, one ceiling — not one per store and not one per phase."""
    board = running_board
    budgets: list = []
    real_budget = kb._fence_mutation_budget

    @kb.contextlib.contextmanager
    def recording(conn):
        with real_budget(conn) as deadline:
            budgets.append(deadline)
            yield deadline

    kb._fence_mutation_budget = recording
    try:
        outer = None
        real_deadline_scope = kb._mutation_deadline

        @kb.contextlib.contextmanager
        def recording_outer(conn, what):
            nonlocal outer
            with real_deadline_scope(conn, what) as deadline:
                outer = deadline
                yield deadline

        kb._mutation_deadline = recording_outer
        try:
            kb.create_task(board["conn"], title="one budget", assignee="worker")
        finally:
            kb._mutation_deadline = real_deadline_scope
    finally:
        kb._fence_mutation_budget = real_budget

    assert outer is not None, "the mutator opened no deadline at its entry"
    assert budgets, "write_txn consulted no budget"
    assert all(b == outer for b in budgets), (
        f"write_txn started its own deadline(s) {budgets!r} beside the "
        f"mutator's {outer!r}"
    )
