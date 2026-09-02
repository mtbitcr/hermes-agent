"""Per-run remote sandbox reservation held in the native task-event log.

The reservation exists so a run that hands work to a remote sandbox owns
exactly one machine: the fold + append share one ``BEGIN IMMEDIATE``
transaction, so it is a real compare-and-swap, and it lives in the
append-only event log rather than a side file that a ``task_runs.metadata``
rewrite could silently drop.

The current provisioned generation is mirrored transactionally into a narrow
cleanup-intent table. It schedules crash-safe one-shot retries; the event fold
remains the authority for whether the run actually owns a live machine.
"""
from __future__ import annotations

import json
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


def test_created_machine_id_is_durable_before_provisioning(running_task):
    task_id, run_id = running_task
    with kb.connect() as conn:
        kb.advance_run_sandbox(
            conn, task_id, run_id=run_id,
            transition="sandbox_reserved", expected_generation=0,
        )
        created = kb.advance_run_sandbox(
            conn, task_id, run_id=run_id,
            transition="sandbox_created", expected_generation=1,
            sandbox_id="sbx-001",
        )
        assert created == {
            "generation": 1,
            "state": "created",
            "sandbox_id": "sbx-001",
            "receipt": None,
        }
        intent = conn.execute(
            "SELECT sandbox_id, attempt_count, exhausted_at "
            "FROM run_sandbox_cleanup_intents WHERE task_id = ? AND run_id = ?",
            (task_id, run_id),
        ).fetchone()
        assert dict(intent) == {
            "sandbox_id": "sbx-001",
            "attempt_count": 0,
            "exhausted_at": None,
        }
        with pytest.raises(kb.RunSandboxConflict):
            kb.advance_run_sandbox(
                conn, task_id, run_id=run_id,
                transition="sandbox_provisioned", expected_generation=1,
                sandbox_id="sbx-other", receipt=RECEIPT,
            )
        active = kb.advance_run_sandbox(
            conn, task_id, run_id=run_id,
            transition="sandbox_provisioned", expected_generation=1,
            sandbox_id="sbx-001", receipt=RECEIPT,
        )
        assert active["state"] == "active"


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


def test_create_returned_after_run_end_still_records_cleanup_authority(running_task):
    task_id, run_id = running_task
    with kb.connect() as conn:
        kb.advance_run_sandbox(
            conn, task_id, run_id=run_id,
            transition="sandbox_reserved", expected_generation=0,
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
        created = kb.advance_run_sandbox(
            conn, task_id, run_id=run_id,
            transition="sandbox_created", expected_generation=1,
            sandbox_id="sbx-late-create",
        )
        assert created["state"] == "created"
        intent = conn.execute(
            "SELECT sandbox_id, generation FROM run_sandbox_cleanup_intents "
            "WHERE task_id=? AND run_id=?",
            (task_id, run_id),
        ).fetchone()
        assert dict(intent) == {
            "sandbox_id": "sbx-late-create",
            "generation": 1,
        }


def test_task_deletion_waits_for_durable_sandbox_release(running_task):
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
        assert kb.delete_task(conn, task_id) is False
        assert kb.archive_task(conn, task_id)
        assert kb.delete_archived_task(conn, task_id) is False
        kb.release_ended_run_sandbox(
            conn,
            task_id,
            run_id=run_id,
            expected_generation=1,
            reason="dispatcher_retry",
        )
        assert kb.delete_archived_task(conn, task_id) is True


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

        pending = kb.claim_ended_run_sandbox_cleanups(
            conn, now=100, lease_seconds=30,
        )
        assert len(pending) == 1
        assert pending[0]["provision_event_id"] > 0
        assert {
            key: pending[0][key]
            for key in (
                "task_id", "run_id", "profile", "generation",
                "sandbox_id", "attempt_count",
            )
        } == {
            "task_id": task_id,
            "run_id": run_id,
            "profile": "worker",
            "generation": 1,
            "sandbox_id": "sbx-001",
            "attempt_count": 1,
        }
        assert kb.claim_ended_run_sandbox_cleanups(conn, now=100) == []
        assert kb.defer_run_sandbox_cleanup(
            conn,
            task_id,
            run_id,
            reason="transient",
            now=100,
        )
        assert kb.claim_ended_run_sandbox_cleanups(conn, now=114) == []
        retried = kb.claim_ended_run_sandbox_cleanups(conn, now=115)
        assert len(retried) == 1
        assert retried[0]["attempt_count"] == 2
        kb.release_ended_run_sandbox(
            conn,
            task_id,
            run_id=run_id,
            expected_generation=1,
            reason="dispatcher_retry",
        )
        assert kb.claim_ended_run_sandbox_cleanups(conn, now=200) == []


def test_cleanup_scheduler_atomically_leases_the_globally_oldest_row(running_task):
    task_id, run_id = running_task
    with kb.connect() as conn:
        kb.advance_run_sandbox(
            conn, task_id, run_id=run_id,
            transition="sandbox_reserved", expected_generation=0,
        )
        kb.advance_run_sandbox(
            conn, task_id, run_id=run_id,
            transition="sandbox_provisioned", expected_generation=1,
            sandbox_id="sbx-canonical", receipt=RECEIPT,
        )
        assert kb.record_run_sandbox_orphan_cleanup(
            conn,
            task_id,
            run_id=run_id,
            generation=1,
            sandbox_id="sbx-orphan",
            reason="fairness_fixture",
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
            conn.execute(
                "UPDATE run_sandbox_cleanup_intents SET next_attempt_at=100"
            )
            conn.execute(
                "UPDATE run_sandbox_orphan_cleanup_intents SET next_attempt_at=50"
            )
        first = kb.claim_next_run_sandbox_cleanup(conn, now=100, lease_seconds=30)
        assert len(first) == 1
        assert first[0]["orphan"] is True
        assert first[0]["sandbox_id"] == "sbx-orphan"
        second = kb.claim_next_run_sandbox_cleanup(conn, now=100, lease_seconds=30)
        assert len(second) == 1
        assert second[0]["orphan"] is False
        assert second[0]["sandbox_id"] == "sbx-canonical"


def test_cleanup_scheduler_terminalizes_expired_head_then_leases_next(running_task):
    task_id, run_id = running_task
    with kb.connect() as conn:
        kb.advance_run_sandbox(
            conn, task_id, run_id=run_id,
            transition="sandbox_reserved", expected_generation=0,
        )
        kb.advance_run_sandbox(
            conn, task_id, run_id=run_id,
            transition="sandbox_provisioned", expected_generation=1,
            sandbox_id="sbx-canonical", receipt=RECEIPT,
        )
        assert kb.record_run_sandbox_orphan_cleanup(
            conn,
            task_id,
            run_id=run_id,
            generation=1,
            sandbox_id="sbx-expired-orphan",
            reason="expired_head_fixture",
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
            conn.execute(
                "UPDATE run_sandbox_cleanup_intents SET next_attempt_at=100"
            )
            conn.execute(
                "UPDATE run_sandbox_orphan_cleanup_intents "
                "SET attempt_count=?, next_attempt_at=50",
                (kb.RUN_SANDBOX_CLEANUP_MAX_ATTEMPTS,),
            )

        claimed = kb.claim_next_run_sandbox_cleanup(conn, now=100)
        assert len(claimed) == 1
        assert claimed[0]["orphan"] is False
        assert claimed[0]["sandbox_id"] == "sbx-canonical"
        expired = conn.execute(
            "SELECT exhausted_at, last_error "
            "FROM run_sandbox_orphan_cleanup_intents "
            "WHERE task_id=? AND run_id=? AND sandbox_id=?",
            (task_id, run_id, "sbx-expired-orphan"),
        ).fetchone()
        assert dict(expired) == {
            "exhausted_at": 100,
            "last_error": "final_attempt_lease_expired",
        }
        events = conn.execute(
            "SELECT payload FROM task_events WHERE task_id=? AND run_id=? "
            "AND kind='sandbox_orphan_cleanup_exhausted'",
            (task_id, run_id),
        ).fetchall()
        assert len(events) == 1


@pytest.mark.parametrize("interrupted", ["missing_table", "empty_table"])
def test_migration_backfills_a_pre_intent_table_active_machine(
    running_task, interrupted,
):
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
                "DELETE FROM kanban_migration_markers WHERE name = ?",
                (kb._SANDBOX_CLEANUP_BACKFILL_MARKER,),
            )
            if interrupted == "missing_table":
                conn.execute("DROP TABLE run_sandbox_cleanup_intents")
            else:
                conn.execute("DELETE FROM run_sandbox_cleanup_intents")
        kb._migrate_add_optional_columns(conn)
        row = conn.execute(
            "SELECT task_id, run_id, profile, generation, sandbox_id "
            "FROM run_sandbox_cleanup_intents",
        ).fetchone()
        assert dict(row) == {
            "task_id": task_id,
            "run_id": run_id,
            "profile": "worker",
            "generation": 1,
            "sandbox_id": "sbx-001",
        }


def test_cleanup_reconciliation_recovers_a_crash_after_legacy_id_rebuild(
    running_task, monkeypatch,
):
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
        assert kb.record_run_sandbox_orphan_cleanup(
            conn,
            task_id,
            run_id=run_id,
            generation=1,
            sandbox_id="sbx-orphan",
            reason="migration_fixture",
        )
        run_columns = [
            str(row["name"])
            for row in conn.execute("PRAGMA table_info(task_runs)")
        ]
        event_columns = [
            str(row["name"])
            for row in conn.execute("PRAGMA table_info(task_events)")
        ]
        runs = [dict(row) for row in conn.execute("SELECT * FROM task_runs")]
        events = [dict(row) for row in conn.execute(
            "SELECT * FROM task_events ORDER BY id",
        )]
        for row in runs:
            row["id"] = "legacy-run-41"
        for index, row in enumerate(events, start=1):
            if row["kind"] == "sandbox_reserved":
                row["id"] = "legacy-event-reserved"
            elif row["kind"] == "sandbox_provisioned":
                row["id"] = "legacy-event-provisioned"
            elif row["kind"] == "sandbox_orphan_cleanup_pending":
                row["id"] = "legacy-event-orphan"
            else:
                row["id"] = f"legacy-event-{index}"
            if row["run_id"] is not None:
                row["run_id"] = "legacy-run-41"

        with kb.write_txn(conn):
            conn.execute(
                "UPDATE tasks SET current_run_id='legacy-run-41' WHERE id=?",
                (task_id,),
            )
            conn.execute(
                "UPDATE run_sandbox_cleanup_intents "
                "SET run_id='legacy-run-41', "
                "provision_event_id='legacy-event-provisioned' "
                "WHERE task_id=?",
                (task_id,),
            )
            conn.execute(
                "UPDATE run_sandbox_orphan_cleanup_intents "
                "SET run_id='legacy-run-41', "
                "created_event_id='legacy-event-orphan' "
                "WHERE task_id=?",
                (task_id,),
            )
            conn.execute("DROP TABLE task_events")
            conn.execute("DROP TABLE task_runs")
            conn.execute(
                kb._REBUILD_SPECS["task_runs"][0].replace(
                    "id INTEGER PRIMARY KEY AUTOINCREMENT", "id TEXT",
                )
            )
            conn.execute(
                kb._REBUILD_SPECS["task_events"][0].replace(
                    "id INTEGER PRIMARY KEY AUTOINCREMENT", "id TEXT",
                )
            )
            run_names = ", ".join(run_columns)
            run_slots = ", ".join("?" for _ in run_columns)
            conn.executemany(
                f"INSERT INTO task_runs ({run_names}) VALUES ({run_slots})",
                (tuple(row[name] for name in run_columns) for row in runs),
            )
            event_names = ", ".join(event_columns)
            event_slots = ", ".join("?" for _ in event_columns)
            conn.executemany(
                f"INSERT INTO task_events ({event_names}) VALUES ({event_slots})",
                (tuple(row[name] for name in event_columns) for row in events),
            )

        original_reconcile = kb._reconcile_run_sandbox_cleanup_intents

        def crash_before_reconcile(*_args, **_kwargs):
            raise RuntimeError("crash after rebuild commit")

        monkeypatch.setattr(
            kb, "_reconcile_run_sandbox_cleanup_intents", crash_before_reconcile,
        )
        with pytest.raises(RuntimeError, match="crash after rebuild commit"):
            kb._migrate_add_optional_columns(conn)
        assert conn.execute(
            "SELECT 1 FROM kanban_migration_markers WHERE name=?",
            (kb._SANDBOX_CLEANUP_BACKFILL_MARKER,),
        ).fetchone() is None
        monkeypatch.setattr(
            kb, "_reconcile_run_sandbox_cleanup_intents", original_reconcile,
        )
        kb._migrate_add_optional_columns(conn)

        assert conn.execute(
            "SELECT type FROM pragma_table_info('task_events') WHERE name='id'",
        ).fetchone()["type"] == "INTEGER"
        assert conn.execute(
            "SELECT type FROM pragma_table_info('task_runs') WHERE name='id'",
        ).fetchone()["type"] == "INTEGER"
        rebuilt_run = int(conn.execute(
            "SELECT id FROM task_runs WHERE task_id=?",
            (task_id,),
        ).fetchone()["id"])
        assert conn.execute(
            "SELECT current_run_id FROM tasks WHERE id=?",
            (task_id,),
        ).fetchone()["current_run_id"] == rebuilt_run
        assert {
            int(row["run_id"])
            for row in conn.execute(
                "SELECT DISTINCT run_id FROM task_events "
                "WHERE task_id=? AND run_id IS NOT NULL",
                (task_id,),
            )
        } == {rebuilt_run}
        intent = conn.execute(
            "SELECT task_id, run_id, profile, generation, sandbox_id, "
            "provision_event_id "
            "FROM run_sandbox_cleanup_intents",
        ).fetchone()
        assert dict(intent) == {
            "task_id": task_id,
            "run_id": rebuilt_run,
            "profile": "worker",
            "generation": 1,
            "sandbox_id": "sbx-001",
            "provision_event_id": conn.execute(
                "SELECT id FROM task_events WHERE task_id=? AND run_id=? "
                "AND kind='sandbox_provisioned'",
                (task_id, rebuilt_run),
            ).fetchone()["id"],
        }
        orphan = conn.execute(
            "SELECT task_id, run_id, generation, sandbox_id, created_event_id "
            "FROM run_sandbox_orphan_cleanup_intents",
        ).fetchone()
        assert dict(orphan) == {
            "task_id": task_id,
            "run_id": rebuilt_run,
            "generation": 1,
            "sandbox_id": "sbx-orphan",
            "created_event_id": conn.execute(
                "SELECT id FROM task_events WHERE task_id=? AND run_id=? "
                "AND kind='sandbox_orphan_cleanup_pending'",
                (task_id, rebuilt_run),
            ).fetchone()["id"],
        }


def test_legacy_id_rebuild_stages_overlaps_and_hidden_rowid_references():
    kb.init_db()
    with kb.connect() as conn:
        task_id = kb.create_task(conn, title="legacy remap", assignee="worker")
        with kb.write_txn(conn):
            conn.execute("DROP TABLE task_events")
            conn.execute("DROP TABLE task_runs")
            conn.execute(
                kb._REBUILD_SPECS["task_runs"][0].replace(
                    "id INTEGER PRIMARY KEY AUTOINCREMENT", "id TEXT",
                )
            )
            conn.execute(
                kb._REBUILD_SPECS["task_events"][0].replace(
                    "id INTEGER PRIMARY KEY AUTOINCREMENT", "id TEXT",
                )
            )
            # In insertion order these become hidden rowids 1, 2, 3. The
            # declared IDs swap 1/2 and leave row 3 NULL, while persisted
            # references use the hidden rowid for that NULL row.
            conn.executemany(
                "INSERT INTO task_runs "
                "(id, task_id, profile, step_key, status, started_at, ended_at) "
                "VALUES (?, ?, 'worker', ?, 'done', ?, ?)",
                (
                    ("2", task_id, "old-2", 1, 11),
                    ("1", task_id, "old-1", 2, 12),
                    (None, task_id, "rowid-3", 3, 13),
                ),
            )
            conn.executemany(
                "INSERT INTO task_events "
                "(id, task_id, run_id, kind, payload, created_at) "
                "VALUES (?, ?, ?, ?, '{}', ?)",
                (
                    ("2", task_id, "2", "legacy-old-2", 1),
                    ("1", task_id, "1", "legacy-old-1", 2),
                    (None, task_id, "3", "legacy-rowid-3", 3),
                ),
            )
            conn.execute(
                "UPDATE tasks SET status='running', current_run_id='2' WHERE id=?",
                (task_id,),
            )
            for old_id, label in (("2", "old-2"), ("1", "old-1"), ("3", "rowid-3")):
                conn.execute(
                    "INSERT INTO run_sandbox_cleanup_intents "
                    "(task_id, run_id, profile, generation, sandbox_id, "
                    "provision_event_id) VALUES (?, ?, 'worker', 1, ?, ?)",
                    (task_id, old_id, f"sbx-main-{label}", old_id),
                )
                conn.execute(
                    "INSERT INTO run_sandbox_orphan_cleanup_intents "
                    "(task_id, run_id, profile, generation, sandbox_id, "
                    "created_event_id) VALUES (?, ?, 'worker', 1, ?, ?)",
                    (task_id, old_id, f"sbx-orphan-{label}", old_id),
                )

        assert kb._rebuild_drifted_tables(conn) == {"task_runs", "task_events"}

        run_ids = {
            str(row["step_key"]): int(row["id"])
            for row in conn.execute(
                "SELECT id, step_key FROM task_runs WHERE task_id=?",
                (task_id,),
            )
        }
        event_ids = {
            str(row["kind"]): int(row["id"])
            for row in conn.execute(
                "SELECT id, kind FROM task_events WHERE task_id=?",
                (task_id,),
            )
        }
        assert run_ids == {"old-2": 1, "old-1": 2, "rowid-3": 3}
        assert event_ids == {
            "legacy-old-2": 1,
            "legacy-old-1": 2,
            "legacy-rowid-3": 3,
        }
        assert conn.execute(
            "SELECT current_run_id FROM tasks WHERE id=?", (task_id,),
        ).fetchone()["current_run_id"] == run_ids["old-2"]
        assert {
            str(row["kind"]): int(row["run_id"])
            for row in conn.execute(
                "SELECT kind, run_id FROM task_events WHERE task_id=?",
                (task_id,),
            )
        } == {
            "legacy-old-2": run_ids["old-2"],
            "legacy-old-1": run_ids["old-1"],
            "legacy-rowid-3": run_ids["rowid-3"],
        }
        for prefix, authority_column in (
            ("sbx-main-", "provision_event_id"),
            ("sbx-orphan-", "created_event_id"),
        ):
            table = (
                "run_sandbox_cleanup_intents"
                if prefix == "sbx-main-"
                else "run_sandbox_orphan_cleanup_intents"
            )
            actual = {
                str(row["sandbox_id"]): (int(row["run_id"]), int(row["authority_id"]))
                for row in conn.execute(
                    f"SELECT sandbox_id, run_id, {authority_column} AS authority_id "
                    f"FROM {table} WHERE task_id=?",
                    (task_id,),
                )
            }
            assert actual == {
                f"{prefix}old-2": (run_ids["old-2"], event_ids["legacy-old-2"]),
                f"{prefix}old-1": (run_ids["old-1"], event_ids["legacy-old-1"]),
                f"{prefix}rowid-3": (
                    run_ids["rowid-3"], event_ids["legacy-rowid-3"],
                ),
            }


def test_cleanup_intent_query_is_bounded_away_from_all_event_history(running_task):
    task_id, run_id = running_task
    with kb.connect() as conn:
        with kb.write_txn(conn):
            conn.executemany(
                "INSERT INTO task_events "
                "(task_id, run_id, kind, payload, created_at) "
                "VALUES (?, ?, 'sandbox_released', ?, ?)",
                (
                    (
                        task_id,
                        run_id + index + 1,
                        '{"generation": 1}',
                        index,
                    )
                    for index in range(2_000)
                ),
            )
        plan = conn.execute(
            "EXPLAIN QUERY PLAN SELECT i.task_id "
            "FROM run_sandbox_cleanup_intents i "
            "INDEXED BY idx_sandbox_cleanup_due "
            "CROSS JOIN task_runs r ON r.id = i.run_id AND r.task_id = i.task_id "
            "CROSS JOIN tasks t ON t.id = i.task_id AND t.task_kind = 'work' "
            "WHERE r.ended_at IS NOT NULL AND i.exhausted_at IS NULL "
            "AND i.next_attempt_at <= ? "
            "ORDER BY i.next_attempt_at ASC, i.provision_event_id ASC LIMIT ?",
            (100, 1),
        ).fetchall()
        details = [str(row["detail"]) for row in plan]
        assert any(
            "idx_sandbox_cleanup_due" in detail
            for detail in details
        ), details
        assert all("task_events" not in detail for detail in details), details


def test_cleanup_backfill_marker_skips_normal_reopen_reconciliation(running_task):
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
            conn.execute("DELETE FROM run_sandbox_cleanup_intents")
        kb._migrate_add_optional_columns(conn)
        assert conn.execute(
            "SELECT 1 FROM run_sandbox_cleanup_intents"
        ).fetchone() is None


def test_cleanup_retry_exhaustion_leaves_authority_but_exits_due_queue(running_task):
    task_id, run_id = running_task
    with kb.connect() as conn:
        kb.advance_run_sandbox(
            conn, task_id, run_id=run_id,
            transition="sandbox_reserved", expected_generation=0,
        )
        kb.advance_run_sandbox(
            conn, task_id, run_id=run_id,
            transition="sandbox_created", expected_generation=1,
            sandbox_id="sbx-001",
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

        for attempt in range(1, kb.RUN_SANDBOX_CLEANUP_MAX_ATTEMPTS + 1):
            now = attempt * 10_000
            claimed = kb.claim_ended_run_sandbox_cleanups(conn, now=now)
            assert len(claimed) == 1
            assert claimed[0]["attempt_count"] == attempt
            assert kb.defer_run_sandbox_cleanup(
                conn, task_id, run_id, reason="permanent", now=now,
            )

        assert kb.claim_ended_run_sandbox_cleanups(conn, now=10**9) == []
        intent = conn.execute(
            "SELECT attempt_count, exhausted_at, sandbox_id "
            "FROM run_sandbox_cleanup_intents WHERE task_id=? AND run_id=?",
            (task_id, run_id),
        ).fetchone()
        assert dict(intent) == {
            "attempt_count": kb.RUN_SANDBOX_CLEANUP_MAX_ATTEMPTS,
            "exhausted_at": kb.RUN_SANDBOX_CLEANUP_MAX_ATTEMPTS * 10_000,
            "sandbox_id": "sbx-001",
        }
        event = conn.execute(
            "SELECT payload FROM task_events WHERE task_id=? AND run_id=? "
            "AND kind='sandbox_cleanup_exhausted' ORDER BY id DESC LIMIT 1",
            (task_id, run_id),
        ).fetchone()
        assert json.loads(event["payload"]) == {
            "attempt_count": kb.RUN_SANDBOX_CLEANUP_MAX_ATTEMPTS,
            "reason": "permanent",
            "manual_cleanup_required": True,
        }
        assert kb.delete_task(conn, task_id) is False
        released = kb.release_ended_run_sandbox(
            conn, task_id, run_id=run_id,
            expected_generation=1, reason="manual_cleanup",
        )
        assert released["state"] == "released"


def test_final_cleanup_lease_crash_exhausts_before_attempt_nine(running_task):
    task_id, run_id = running_task
    with kb.connect() as conn:
        kb.advance_run_sandbox(
            conn, task_id, run_id=run_id,
            transition="sandbox_reserved", expected_generation=0,
        )
        kb.advance_run_sandbox(
            conn, task_id, run_id=run_id,
            transition="sandbox_created", expected_generation=1,
            sandbox_id="sbx-final-lease",
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
        for attempt in range(1, kb.RUN_SANDBOX_CLEANUP_MAX_ATTEMPTS):
            now = attempt * 10_000
            claimed = kb.claim_ended_run_sandbox_cleanups(conn, now=now)
            assert claimed[0]["attempt_count"] == attempt
            assert kb.defer_run_sandbox_cleanup(
                conn, task_id, run_id, reason="transient", now=now,
            )

        final_now = kb.RUN_SANDBOX_CLEANUP_MAX_ATTEMPTS * 10_000
        final = kb.claim_ended_run_sandbox_cleanups(
            conn, now=final_now, lease_seconds=30,
        )
        assert final[0]["attempt_count"] == kb.RUN_SANDBOX_CLEANUP_MAX_ATTEMPTS
        assert kb.claim_ended_run_sandbox_cleanups(conn, now=final_now + 29) == []
        assert kb.claim_ended_run_sandbox_cleanups(conn, now=final_now + 30) == []
        row = conn.execute(
            "SELECT attempt_count, exhausted_at, last_error "
            "FROM run_sandbox_cleanup_intents WHERE task_id=? AND run_id=?",
            (task_id, run_id),
        ).fetchone()
        assert dict(row) == {
            "attempt_count": kb.RUN_SANDBOX_CLEANUP_MAX_ATTEMPTS,
            "exhausted_at": final_now + 30,
            "last_error": "final_attempt_lease_expired",
        }
        events = conn.execute(
            "SELECT payload FROM task_events WHERE task_id=? AND run_id=? "
            "AND kind='sandbox_cleanup_exhausted' ORDER BY id",
            (task_id, run_id),
        ).fetchall()
        assert len(events) == 1
        assert json.loads(events[0]["payload"])["attempt_count"] == 8
        assert kb.delete_task(conn, task_id) is False


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
