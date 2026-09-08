"""The epoch mirror (EM-2, EM-4).

The mirror rules are exercised by driving real mutations and reading the
persisted rows back, never by calling a validator and trusting its
boolean. Boards need Gate B armed to exercise these rules, so fixtures go
through :func:`create_fenced_board` (``create_board`` followed by the
real recorded backfill) rather than ``create_board`` alone.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_WORKTREE = Path(__file__).resolve().parents[2]
if str(_WORKTREE) not in sys.path:
    sys.path.insert(0, str(_WORKTREE))

from hermes_cli import kanban_db as kb
from tests.hermes_cli._kanban_fence_support import (
    begin_removal,
    create_fenced_board,
    gate_row,
    ready_task,
    register_row,
    task_row,
)


def _set_mirror(board: str, value: int) -> None:
    """Move only the mirror, leaving Gate A alone — the drift EM-4 judges."""
    conn = kb.connect(board=board)
    try:
        with kb._fence_protocol_scope():
            with kb.write_txn(conn):
                conn.execute(
                    "UPDATE board_fence_state SET epoch_mirror = ? WHERE id = 1",
                    (value,),
                )
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# EM-2 — gate and mirror move in ONE commit
# ---------------------------------------------------------------------------

def test_em2_the_gate_and_the_mirror_are_never_observed_apart(fence_home):
    create_fenced_board("em2")
    db_path = kb.kanban_db_path(board="em2")
    assert gate_row(db_path) == ("open", 1)

    begin_removal("em2")
    assert gate_row(db_path) == ("open", 1), "the intent alone must not move Gate B"

    kb.commit_fence_closing_point("em2")
    assert gate_row(db_path) == ("closing", 2)


# ---------------------------------------------------------------------------
# EM-4 — the pair rule decides, not raw inequality
# ---------------------------------------------------------------------------

def test_em4a_equal_epochs_admit_work(fence_home):
    create_fenced_board("em4a")
    conn = kb.connect(board="em4a")
    try:
        task_id = ready_task(conn)
        assert kb.claim_task(conn, task_id) is not None
    finally:
        conn.close()
    assert task_row(kb.kanban_db_path(board="em4a"), task_id)["status"] == "running"


def test_em4b_the_expected_one_step_lag_refuses_nothing(fence_home):
    create_fenced_board("em4b")
    db_path = kb.kanban_db_path(board="em4b")
    conn = kb.connect(board="em4b")
    task_id = ready_task(conn)

    begin_removal("em4b")
    assert register_row("em4b")["epoch"] == 2
    assert gate_row(db_path) == ("open", 1)

    assert kb.claim_task(conn, task_id) is not None
    conn.close()
    assert task_row(db_path, task_id)["status"] == "running"


def test_em4c_a_mirror_ahead_of_the_register_refuses(fence_home):
    create_fenced_board("em4c-ahead")
    db_path = kb.kanban_db_path(board="em4c-ahead")
    conn = kb.connect(board="em4c-ahead")
    task_id = ready_task(conn)
    conn.close()

    _set_mirror("em4c-ahead", 9)

    conn = kb.connect(board="em4c-ahead")
    try:
        with pytest.raises(kb.BoardFenceClosedError) as excinfo:
            kb.claim_task(conn, task_id)
    finally:
        conn.close()

    assert excinfo.value.refusal.rule is kb.FenceRefusalRule.EM_4c
    assert task_row(db_path, task_id)["status"] == "ready"


def test_em4c_a_lag_on_a_settled_live_entry_refuses(fence_home):
    """A one-step lag is only expected DURING the intent→fence window. On
    a settled live entry the same numbers are indeterminate."""
    create_fenced_board("em4c-lag")
    db_path = kb.kanban_db_path(board="em4c-lag")
    conn = kb.connect(board="em4c-lag")
    task_id = ready_task(conn)
    conn.close()

    entry = kb.get_register_entry("em4c-lag")
    kb.transition_register_entry(
        kb.RegisterEntry(
            board_name="em4c-lag",
            lifecycle=kb.BoardLifecycle.LIVE,
            epoch=2,
            epoch_before=1,
            gate_move=kb.GateMove.SETTLED,
            epoch_lineage=[1, 2],
            created_at=entry.created_at,
        )
    )

    conn = kb.connect(board="em4c-lag")
    try:
        with pytest.raises(kb.BoardFenceClosedError):
            kb.claim_task(conn, task_id)
    finally:
        conn.close()
    assert task_row(db_path, task_id)["status"] == "ready"


def test_the_accepted_mirrors_are_what_the_statement_predicate_uses(fence_home):
    """The conjunct a mutation folds into its own WHERE clause is derived
    from EM-4's accepted set — one epoch normally, two inside the window."""
    create_fenced_board("conjunct")
    conn = kb.connect(board="conjunct")
    try:
        with kb.write_txn(conn):
            sql, params = kb.fence_cas_conjunct(conn)
        assert "board_fence_state" in sql
        assert params == [1]

        begin_removal("conjunct")
        with kb.write_txn(conn):
            sql, params = kb.fence_cas_conjunct(conn)
        assert sorted(params) == [1, 2]
    finally:
        conn.close()


def test_the_insert_form_of_the_predicate_carries_the_same_gate(fence_home):
    create_fenced_board("insert-gate")
    conn = kb.connect(board="insert-gate")
    try:
        with kb.write_txn(conn):
            sql, params = kb.fence_insert_predicate(conn)
        assert sql.startswith(" WHERE 1 = 1")
        assert "gate = 'open'" in sql
        assert params == [1]
    finally:
        conn.close()


def test_the_mirror_is_readable_atomically_with_the_gate(fence_home):
    create_fenced_board("atomic-read")
    conn = kb.connect(board="atomic-read")
    try:
        state = kb.get_in_board_fence_state(conn)
    finally:
        conn.close()
    assert (state.gate, state.epoch_mirror) == (kb.InBoardGate.OPEN, 1)


def test_a_bounded_gate_b_read_distinguishes_timeout_from_fault(fence_home):
    """``read_in_board_fence_state_bounded`` keeps "we could not get in"
    apart from "what we found was broken" — the distinction the timeout
    outcome depends on."""
    import sqlite3

    create_fenced_board("bounded")
    db_path = kb.kanban_db_path(board="bounded")

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        state, status = kb.read_in_board_fence_state_bounded(conn)
        assert status is kb.FenceReadStatus.OK
        assert state.epoch_mirror == 1

        conn.execute("DROP TABLE board_fence_state")
        _, status = kb.read_in_board_fence_state_bounded(conn)
        assert status is kb.FenceReadStatus.ERROR
    finally:
        conn.close()


def test_epoch_lineage_records_every_epoch_the_board_has_held(fence_home):
    create_fenced_board("lineage")
    assert kb.get_register_entry("lineage").epoch_lineage == [1]
    begin_removal("lineage")
    assert kb.get_register_entry("lineage").epoch_lineage == [1, 2]
