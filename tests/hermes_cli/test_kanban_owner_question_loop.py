"""A repeated owner question stays waiting for the owner (card C1, finding 116).

A ``needs_input`` block asks the owner a question. When the worker raises the
same question a second time (blocked, unblocked, run again, blocked again),
the unblock-loop breaker must not move the task to ``triage``, where the owner
never sees it: the task stays ``blocked`` and stays listed as waiting on the
owner. Every other kind keeps the loop breaker, so a repeated capability block
still goes to ``triage``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc


@pytest.fixture
def kanban_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _running_task(conn, title="t"):
    """Create a task and drive it to ``running`` so block_task can act."""
    tid = kb.create_task(conn, title=title, assignee="worker")
    with kb.write_txn(conn):
        conn.execute("UPDATE tasks SET status='ready' WHERE id=?", (tid,))
    claimed = kb.claim_task(conn, tid, claimer="worker")
    assert claimed is not None
    return tid


def _make_running_again(conn, tid):
    with kb.write_txn(conn):
        conn.execute("UPDATE tasks SET status='ready' WHERE id=?", (tid,))
    assert kb.claim_task(conn, tid, claimer="worker") is not None


def _block_twice(conn, kind):
    """Block with ``kind``, unblock, run again, then block with ``kind`` again."""
    tid = _running_task(conn)
    assert kb.block_task(conn, tid, reason="Which logo should the site use?", kind=kind)
    assert kb.unblock_task(conn, tid)
    _make_running_again(conn, tid)
    assert kb.block_task(conn, tid, reason="Which logo should the site use?", kind=kind)
    return tid


def _task_row(conn, tid):
    return conn.execute(
        "SELECT status, block_kind, block_recurrences FROM tasks WHERE id=?",
        (tid,),
    ).fetchone()


def test_a_second_owner_question_stays_blocked(kanban_home: Path) -> None:
    with kbc.connect_closing() as conn:
        tid = _block_twice(conn, "needs_input")
        row = _task_row(conn, tid)
        assert row["status"] == "blocked"
        assert row["block_kind"] == "needs_input"
        assert row["block_recurrences"] == 2


def test_a_second_owner_question_is_still_waiting_on_the_owner(kanban_home: Path) -> None:
    with kbc.connect_closing() as conn:
        tid = _block_twice(conn, "needs_input")
        waits = kb.task_owner_waits(conn, [tid])
        assert tid in waits, (
            f"not waiting on the owner; status is {_task_row(conn, tid)['status']!r}"
        )
        assert waits[tid]["kind"] == "needs_input"


def test_a_repeated_capability_block_still_goes_to_triage(kanban_home: Path) -> None:
    with kbc.connect_closing() as conn:
        tid = _block_twice(conn, "capability")
        row = _task_row(conn, tid)
        assert row["status"] == "triage"
        assert row["block_kind"] == "capability"
        assert row["block_recurrences"] == 2
