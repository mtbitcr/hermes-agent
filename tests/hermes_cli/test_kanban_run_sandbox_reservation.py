"""Per-run remote sandbox reservation held in the native task-event log.

The reservation exists so a run that hands work to a remote sandbox owns
exactly one machine: the fold + append share one ``BEGIN IMMEDIATE``
transaction, so it is a real compare-and-swap, and it lives in the
append-only event log rather than a side file that a ``task_runs.metadata``
rewrite could silently drop.
"""
from __future__ import annotations

import threading

import pytest

from hermes_cli import kanban_db as kb


RECEIPT = {
    "sandbox_id": "sbx-001",
    "image_digest": "sha256:" + "ab" * 32,
    "source_commit": "c" * 40,
    "workspace": "/workspace/task",
    "ownership_scope": ["src"],
    "policy": {"credential_vault": True, "egress_default_deny": True},
}


@pytest.fixture
def running_task():
    kb.init_db()
    with kb.connect() as conn:
        task_id = kb.create_task(conn, title="remote build", assignee="worker")
        with kb.write_txn(conn):
            conn.execute(
                "INSERT INTO task_runs (task_id, profile, status, started_at) "
                "VALUES (?, 'worker', 'running', 0)",
                (task_id,),
            )
            run_id = int(
                conn.execute("SELECT last_insert_rowid() AS id").fetchone()["id"]
            )
            conn.execute(
                "UPDATE tasks SET status = 'running', current_run_id = ? WHERE id = ?",
                (run_id, task_id),
            )
    return task_id, run_id


def test_a_fresh_run_has_no_reservation(running_task):
    task_id, run_id = running_task
    with kb.connect() as conn:
        assert kb.read_run_sandbox(conn, task_id, run_id=run_id) == {
            "generation": 0,
            "state": "absent",
            "sandbox_id": None,
            "receipt": None,
        }


def test_reserve_then_provision_records_a_durable_receipt(running_task):
    task_id, run_id = running_task
    with kb.connect() as conn:
        reserved = kb.advance_run_sandbox(
            conn, task_id, run_id=run_id,
            transition="sandbox_reserved", expected_generation=0,
        )
        assert reserved["state"] == "reserved"
        assert reserved["generation"] == 1

        active = kb.advance_run_sandbox(
            conn, task_id, run_id=run_id,
            transition="sandbox_provisioned", expected_generation=1,
            sandbox_id="sbx-001", receipt=RECEIPT,
        )
        assert active["state"] == "active"
        assert active["sandbox_id"] == "sbx-001"
        assert active["receipt"] == RECEIPT

    # Durable across connections — it is board state, not process state.
    with kb.connect() as conn:
        assert kb.read_run_sandbox(conn, task_id, run_id=run_id) == active


def test_only_one_of_two_concurrent_reservations_wins(running_task):
    task_id, run_id = running_task
    outcomes: list[str] = []
    barrier = threading.Barrier(2)

    def attempt() -> None:
        with kb.connect() as conn:
            barrier.wait(timeout=10)
            try:
                kb.advance_run_sandbox(
                    conn, task_id, run_id=run_id,
                    transition="sandbox_reserved", expected_generation=0,
                )
                outcomes.append("won")
            except kb.RunSandboxConflict:
                outcomes.append("lost")

    threads = [threading.Thread(target=attempt) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)

    assert sorted(outcomes) == ["lost", "won"]
    with kb.connect() as conn:
        assert kb.read_run_sandbox(conn, task_id, run_id=run_id)["generation"] == 1


def test_a_stale_generation_cannot_settle_a_newer_one(running_task):
    task_id, run_id = running_task
    with kb.connect() as conn:
        kb.advance_run_sandbox(
            conn, task_id, run_id=run_id,
            transition="sandbox_reserved", expected_generation=0,
        )
        with pytest.raises(kb.RunSandboxConflict):
            kb.advance_run_sandbox(
                conn, task_id, run_id=run_id,
                transition="sandbox_provisioned", expected_generation=0,
                sandbox_id="sbx-002", receipt=RECEIPT,
            )


def test_a_released_generation_opens_the_next_one(running_task):
    task_id, run_id = running_task
    with kb.connect() as conn:
        kb.advance_run_sandbox(
            conn, task_id, run_id=run_id,
            transition="sandbox_reserved", expected_generation=0,
        )
        kb.advance_run_sandbox(
            conn, task_id, run_id=run_id,
            transition="sandbox_provisioned", expected_generation=1,
            sandbox_id="sbx-001", receipt=RECEIPT,
        )
        retired = kb.advance_run_sandbox(
            conn, task_id, run_id=run_id,
            transition="sandbox_released", expected_generation=1,
            reason="sandbox_gone",
        )
        assert retired == {
            "generation": 1,
            "state": "released",
            "sandbox_id": None,
            "receipt": None,
        }
        replacement = kb.advance_run_sandbox(
            conn, task_id, run_id=run_id,
            transition="sandbox_reserved", expected_generation=1,
        )
        assert replacement["generation"] == 2
        assert replacement["state"] == "reserved"


def test_a_second_run_never_inherits_the_first_runs_machine(running_task):
    task_id, run_id = running_task
    with kb.connect() as conn:
        kb.advance_run_sandbox(
            conn, task_id, run_id=run_id,
            transition="sandbox_reserved", expected_generation=0,
        )
        kb.advance_run_sandbox(
            conn, task_id, run_id=run_id,
            transition="sandbox_provisioned", expected_generation=1,
            sandbox_id="sbx-001", receipt=RECEIPT,
        )
        with kb.write_txn(conn):
            conn.execute(
                "INSERT INTO task_runs (task_id, status, started_at) "
                "VALUES (?, 'running', 0)",
                (task_id,),
            )
            next_run = int(
                conn.execute("SELECT last_insert_rowid() AS id").fetchone()["id"]
            )
            conn.execute(
                "UPDATE tasks SET current_run_id = ? WHERE id = ?",
                (next_run, task_id),
            )
        assert kb.read_run_sandbox(conn, task_id, run_id=next_run)["state"] == "absent"
        # ...and the superseded run may no longer write to the board.
        with pytest.raises(kb.RunSandboxConflict):
            kb.advance_run_sandbox(
                conn, task_id, run_id=run_id,
                transition="sandbox_released", expected_generation=1,
                reason="late",
            )


def test_ended_run_cleanup_releases_only_its_own_active_sandbox(running_task):
    task_id, run_id = running_task
    with kb.connect() as conn:
        kb.advance_run_sandbox(
            conn, task_id, run_id=run_id,
            transition="sandbox_reserved", expected_generation=0,
        )
        kb.advance_run_sandbox(
            conn, task_id, run_id=run_id,
            transition="sandbox_provisioned", expected_generation=1,
            sandbox_id="sbx-001", receipt=RECEIPT,
        )
        with kb.write_txn(conn):
            conn.execute(
                "UPDATE task_runs SET status='done', outcome='completed', ended_at=1 "
                "WHERE id=?",
                (run_id,),
            )
            conn.execute(
                "INSERT INTO task_runs (task_id, status, started_at) "
                "VALUES (?, 'running', 2)",
                (task_id,),
            )
            next_run = int(
                conn.execute("SELECT last_insert_rowid() AS id").fetchone()["id"]
            )
            conn.execute(
                "UPDATE tasks SET status='running', current_run_id=? WHERE id=?",
                (next_run, task_id),
            )

        released = kb.release_ended_run_sandbox(
            conn,
            task_id,
            run_id=run_id,
            expected_generation=1,
            reason="worker_exited",
        )
        assert released["state"] == "released"
        assert kb.read_run_sandbox(
            conn, task_id, run_id=next_run,
        )["state"] == "absent"


def test_ended_active_sandbox_is_a_bounded_durable_cleanup_intent(running_task):
    task_id, run_id = running_task
    with kb.connect() as conn:
        kb.advance_run_sandbox(
            conn, task_id, run_id=run_id,
            transition="sandbox_reserved", expected_generation=0,
        )
        kb.advance_run_sandbox(
            conn, task_id, run_id=run_id,
            transition="sandbox_provisioned", expected_generation=1,
            sandbox_id="sbx-001", receipt=RECEIPT,
        )
        with kb.write_txn(conn):
            conn.execute(
                "UPDATE task_runs SET status='done', outcome='completed', ended_at=1 "
                "WHERE id=?",
                (run_id,),
            )
            conn.execute(
                "UPDATE tasks SET status='done', current_run_id=NULL WHERE id=?",
                (task_id,),
            )

        assert kb.list_ended_run_sandboxes(conn) == [{
            "task_id": task_id,
            "run_id": run_id,
            "profile": "worker",
        }]
        kb.release_ended_run_sandbox(
            conn,
            task_id,
            run_id=run_id,
            expected_generation=1,
            reason="dispatcher_retry",
        )
        assert kb.list_ended_run_sandboxes(conn) == []


def test_a_run_that_is_no_longer_active_cannot_reserve(running_task):
    task_id, run_id = running_task
    with kb.connect() as conn:
        with kb.write_txn(conn):
            conn.execute(
                "UPDATE tasks SET status = 'done' WHERE id = ?", (task_id,)
            )
        with pytest.raises(kb.RunSandboxConflict):
            kb.advance_run_sandbox(
                conn, task_id, run_id=run_id,
                transition="sandbox_reserved", expected_generation=0,
            )


@pytest.mark.parametrize(
    "receipt",
    [
        None,
        {},
        {"nested": {"deep": {"too": "far"}}},
        {"floaty": 1.5},
        {"blob": b"bytes"},
        {"list_of_objects": [{"a": 1}]},
    ],
)
def test_a_malformed_receipt_is_refused_at_write_time(running_task, receipt):
    task_id, run_id = running_task
    with kb.connect() as conn:
        kb.advance_run_sandbox(
            conn, task_id, run_id=run_id,
            transition="sandbox_reserved", expected_generation=0,
        )
        with pytest.raises(ValueError):
            kb.advance_run_sandbox(
                conn, task_id, run_id=run_id,
                transition="sandbox_provisioned", expected_generation=1,
                sandbox_id="sbx-001", receipt=receipt,
            )
        assert kb.read_run_sandbox(conn, task_id, run_id=run_id)["state"] == "reserved"


def test_an_unreadable_provisioned_event_is_not_reusable(running_task):
    task_id, run_id = running_task
    with kb.connect() as conn:
        kb.advance_run_sandbox(
            conn, task_id, run_id=run_id,
            transition="sandbox_reserved", expected_generation=0,
        )
        with kb.write_txn(conn):
            conn.execute(
                "INSERT INTO task_events (task_id, run_id, kind, payload, created_at) "
                "VALUES (?, ?, 'sandbox_provisioned', ?, 0)",
                (task_id, run_id, '{"generation": 1, "sandbox_id": ""}'),
            )
        record = kb.read_run_sandbox(conn, task_id, run_id=run_id)
        assert record["state"] == "reserved"
        assert record["receipt"] is None
        # Fail-closed: the open generation must be released before another
        # machine can be created for this run.
        with pytest.raises(kb.RunSandboxConflict):
            kb.advance_run_sandbox(
                conn, task_id, run_id=run_id,
                transition="sandbox_reserved", expected_generation=1,
            )
