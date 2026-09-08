"""Fail-closed behaviour and the ONE five-second mutation deadline.

Reviewer reproduction 4: the deadline lived only in
``check_fence_for_operation``, which had no production callers. Real
``write_txn`` / ``evaluate_fence_for_mutation`` established no deadline
at all, so Gate A was consulted with the ordinary (two-minute,
independently configurable) kanban busy timeout — and a real claim was
GRANTED after 5.666 s against a 5.0 s ceiling, with no timeout refusal.

Every test here therefore holds a REAL cross-process SQLite lock and
measures a REAL production call: elapsed wall time, the distinct
``refused-timeout`` outcome, and the persisted row afterwards.
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
    close_fence,
    gate_row,
    hold_exclusive_lock,
    ready_task,
    row_count,
    task_row,
)


# The ceiling the design fixes, and the slack a loaded CI box may add on
# top of it. The point of the assertion is that the call cannot wait for
# the ordinary busy timeout (two minutes), so the upper bound is checked
# against the ceiling plus a small allowance, never against "some number".
CEILING = kb.MAX_FENCE_CONSULTATION_TIMEOUT_SECONDS
UPPER = CEILING + 3.0


def _timed(call):
    start = time.monotonic()
    try:
        result = call()
    except kb.BoardFenceClosedError as exc:
        return time.monotonic() - start, exc.refusal, None
    return time.monotonic() - start, None, result


@pytest.fixture
def board_with_ready_task(fence_home):
    create_fenced_board("deadline")
    conn = kb.connect(board="deadline")
    task_id = ready_task(conn)
    kb.add_comment(conn, task_id, "author", "seed")
    return conn, task_id, kb.kanban_db_path(board="deadline")


# ---------------------------------------------------------------------------
# Reproduction 4 — the deadline is on the real path
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("operation", ["claim", "renew", "removal", "content-write"])
def test_repro_4_gate_a_held_exclusively_times_out_within_the_ceiling(
    board_with_ready_task, operation
):
    """A real held Gate-A lock, a real kernel mutation, a measured wait.

    The reviewer's case was a claim granted at 5.666 s. Each of the four
    operation classes the decision names — claim, renew, removal, content
    write — must instead REFUSE, with the distinct timeout outcome, inside
    the five-second budget, having changed nothing.
    """
    conn, task_id, db_path = board_with_ready_task
    calls = {
        "claim": lambda: kb.claim_task(conn, task_id),
        "renew": lambda: kb.heartbeat_claim(conn, task_id, claimer="worker"),
        "removal": lambda: kb.archive_task(conn, task_id),
        "content-write": lambda: kb.add_comment(conn, task_id, "author", "blocked"),
    }
    before = {
        "tasks": row_count(db_path, "tasks"),
        "task_comments": row_count(db_path, "task_comments"),
        "status": task_row(db_path, task_id)["status"],
    }

    with hold_exclusive_lock(kb.register_db_path(), whole_file=True):
        elapsed, refusal, result = _timed(calls[operation])

    assert refusal is not None, f"{operation} returned {result!r} instead of refusing"
    assert refusal.outcome is kb.FenceOutcome.REFUSED_TIMEOUT
    assert refusal.rule is kb.FenceRefusalRule.TIMEOUT
    assert elapsed <= UPPER, f"{operation} waited {elapsed:.3f}s past the {CEILING}s ceiling"

    assert row_count(db_path, "tasks") == before["tasks"]
    assert row_count(db_path, "task_comments") == before["task_comments"]
    assert task_row(db_path, task_id)["status"] == before["status"]


def test_repro_4_the_board_store_shares_the_same_budget(board_with_ready_task):
    """The budget is passed to BOTH stores: a board held under a real
    exclusive write lock refuses at the same ceiling, not at the ordinary
    busy timeout."""
    conn, task_id, db_path = board_with_ready_task

    with hold_exclusive_lock(db_path):
        elapsed, refusal, result = _timed(
            lambda: kb.add_comment(conn, task_id, "author", "blocked")
        )

    assert refusal is not None, f"content write returned {result!r} instead of refusing"
    assert refusal.outcome is kb.FenceOutcome.REFUSED_TIMEOUT
    assert elapsed <= UPPER


def test_repro_4_the_deadline_is_one_per_mutation_not_one_per_store(
    board_with_ready_task,
):
    """Two blocking stores must not cost two ceilings.

    Gate A held exclusively makes both the register OPEN and the register
    READ block; the mutation still comes back inside a single budget.
    """
    conn, task_id, _ = board_with_ready_task

    with hold_exclusive_lock(kb.register_db_path(), whole_file=True):
        elapsed, refusal, _ = _timed(lambda: kb.claim_task(conn, task_id))

    assert refusal is not None
    assert elapsed <= UPPER


def test_the_deadline_is_not_widenable_by_configuration(fence_home, monkeypatch):
    """A fence whose bound can be raised is a hint, not a fence."""
    monkeypatch.setattr(
        "hermes_cli.config.load_config_readonly",
        lambda: {"kanban": {"fence_consultation_timeout_seconds": 3600}},
    )
    assert kb._fence_consultation_timeout_seconds() == CEILING

    monkeypatch.setattr(
        "hermes_cli.config.load_config_readonly",
        lambda: {"kanban": {"fence_consultation_timeout_seconds": 0.05}},
    )
    assert kb._fence_consultation_timeout_seconds() == pytest.approx(0.05)


def test_a_narrowed_deadline_is_honoured_end_to_end(fence_home, monkeypatch):
    """Narrowing the documented knob really does shorten the real wait —
    proof the production path reads the budget rather than ignoring it."""
    create_fenced_board("narrow")
    conn = kb.connect(board="narrow")
    task_id = ready_task(conn)

    monkeypatch.setattr(
        "hermes_cli.config.load_config_readonly",
        lambda: {"kanban": {"fence_consultation_timeout_seconds": 0.4}},
    )

    with hold_exclusive_lock(kb.register_db_path(), whole_file=True):
        elapsed, refusal, _ = _timed(lambda: kb.claim_task(conn, task_id))

    assert refusal is not None
    assert refusal.outcome is kb.FenceOutcome.REFUSED_TIMEOUT
    assert elapsed < CEILING, (
        f"a 0.4s budget still waited {elapsed:.3f}s — the production path is "
        "not reading the budget"
    )
    conn.close()


# ---------------------------------------------------------------------------
# Fail-closed on everything the fence cannot vouch for
# ---------------------------------------------------------------------------

def test_a_fenced_board_whose_authority_vanished_refuses(fence_home):
    """Gate B present, Gate A row gone: nothing can vouch for this board."""
    create_fenced_board("no-authority")
    db_path = kb.kanban_db_path(board="no-authority")
    conn = kb.connect(board="no-authority")
    task_id = ready_task(conn)

    with kb.register_connect() as reg:
        reg.execute("BEGIN IMMEDIATE")
        reg.execute("DELETE FROM board_register WHERE board_name = ?", ("no-authority",))
        reg.execute("COMMIT")

    with pytest.raises(kb.BoardFenceClosedError) as excinfo:
        kb.claim_task(conn, task_id)
    conn.close()

    assert excinfo.value.refusal.rule is kb.FenceRefusalRule.GA_4
    assert task_row(db_path, task_id)["status"] == "ready"


def test_an_archived_board_refuses_writes(fence_home):
    """``archived`` is a CLOSED register outcome: opening still works (the
    board is not gone from disk), but every write refuses."""
    create_fenced_board("archived-board")
    conn = kb.connect(board="archived-board")
    task_id = ready_task(conn)
    entry = kb.get_register_entry("archived-board")
    kb.transition_register_entry(
        kb.RegisterEntry(
            board_name="archived-board",
            lifecycle=kb.BoardLifecycle.ARCHIVED,
            epoch=entry.epoch,
            epoch_lineage=entry.epoch_lineage,
            created_at=entry.created_at,
        )
    )

    with pytest.raises(kb.BoardFenceClosedError) as excinfo:
        kb.claim_task(conn, task_id)
    conn.close()
    assert excinfo.value.refusal.outcome is kb.FenceOutcome.REFUSED_CLOSED


def test_a_gate_row_that_lost_its_only_row_is_corruption_not_freedom(fence_home):
    """A store that opted into the fence and then lost the row refuses."""
    create_fenced_board("rowless")
    db_path = kb.kanban_db_path(board="rowless")
    conn = kb.connect(board="rowless")
    task_id = ready_task(conn)

    with kb._fence_protocol_scope():
        with kb.write_txn(conn):
            conn.execute("DELETE FROM board_fence_state")

    with pytest.raises(kb.BoardFenceClosedError) as excinfo:
        kb.add_comment(conn, task_id, "a", "b")
    conn.close()

    assert excinfo.value.refusal.outcome is kb.FenceOutcome.REFUSED_INDETERMINATE
    assert row_count(db_path, "task_comments") == 0


def test_a_malformed_gate_value_refuses(fence_home):
    create_fenced_board("malformed")
    db_path = kb.kanban_db_path(board="malformed")
    conn = kb.connect(board="malformed")
    task_id = ready_task(conn)

    with kb._fence_protocol_scope():
        with kb.write_txn(conn):
            conn.execute("UPDATE board_fence_state SET gate = 'nonsense' WHERE id = 1")

    with pytest.raises(kb.BoardFenceClosedError):
        kb.claim_task(conn, task_id)
    conn.close()
    assert task_row(db_path, task_id)["status"] == "ready"


def test_a_frozen_gate_refuses_like_a_closing_one(fence_home):
    create_fenced_board("frozen")
    db_path = kb.kanban_db_path(board="frozen")
    conn = kb.connect(board="frozen")
    task_id = ready_task(conn)

    with kb._fence_protocol_scope():
        with kb.write_txn(conn):
            conn.execute("UPDATE board_fence_state SET gate = 'frozen' WHERE id = 1")

    with pytest.raises(kb.BoardFenceClosedError) as excinfo:
        kb.claim_task(conn, task_id)
    conn.close()

    assert excinfo.value.refusal.outcome is kb.FenceOutcome.REFUSED_CLOSED
    assert gate_row(db_path) == ("frozen", 1)
    assert task_row(db_path, task_id)["status"] == "ready"


def test_reads_keep_working_on_a_closed_board(fence_home):
    """A closed fence must not make a board unreadable — the CLI's ``list``
    is what an operator uses to see what is on the board being removed."""
    create_fenced_board("readable")
    conn = kb.connect(board="readable")
    ready_task(conn, "visible")
    conn.close()

    close_fence("readable")

    conn = kb.connect(board="readable")
    try:
        assert [t.title for t in kb.list_tasks(conn)] == ["visible"]
    finally:
        conn.close()


def test_the_refusal_is_machine_readable_not_a_string(fence_home):
    create_fenced_board("structured")
    conn = kb.connect(board="structured")
    task_id = ready_task(conn)
    conn.close()
    close_fence("structured")

    conn = kb.connect(board="structured")
    try:
        with pytest.raises(kb.BoardFenceClosedError) as excinfo:
            kb.claim_task(conn, task_id)
    finally:
        conn.close()

    refusal = excinfo.value.refusal
    assert isinstance(refusal, kb.FenceRefusal)
    assert refusal.outcome is kb.FenceOutcome.REFUSED_CLOSED
    assert refusal.rule is kb.FenceRefusalRule.CLOSED
    assert refusal.board == "structured"
    assert refusal.register_epoch == 2
    assert refusal.mirror_epoch == 2


def test_an_unreadable_register_file_refuses_rather_than_admits(fence_home):
    """A register that cannot be parsed at all is indeterminate, and
    indeterminate refuses."""
    create_fenced_board("corrupt-register")
    conn = kb.connect(board="corrupt-register")
    task_id = ready_task(conn)
    db_path = kb.kanban_db_path(board="corrupt-register")

    register = kb.register_db_path()
    for leftover in list(register.parent.glob("board_register.db*")):
        leftover.unlink()
    register.write_bytes(b"this is not a sqlite database at all, not even close")
    kb._REGISTER_INITIALIZED = False
    kb._REGISTER_INITIALIZED_PATHS.clear()

    with pytest.raises((kb.BoardFenceClosedError, sqlite3.DatabaseError)):
        kb.claim_task(conn, task_id)
    conn.close()
    assert task_row(db_path, task_id)["status"] == "ready"
