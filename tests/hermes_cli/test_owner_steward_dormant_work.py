"""The Overview headline reads working only while live work remains (finding 103).

Parked work, and todo work waiting only on parked work, is dormant: when it is
all that remains, the steward reads the existing waiting-for-approval state.
"""

from __future__ import annotations

import threading
import time

import pytest

from hermes_cli import kanban_db, owner_workspace as ow
from tools import approval


@pytest.fixture
def ctx():
    return ow.OwnerContext(actor="default", profile="default", session="run_dormant")


def _auto_approve(session_key: str, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        with approval._lock:
            queue = approval._gateway_queues.get(session_key)
            aid = queue[0].approval_id if queue else None
        if aid:
            approval.resolve_gateway_approval(session_key, "once", approval_id=aid)
            return
        time.sleep(0.005)
    raise TimeoutError("no approval was queued")


def _project(ctx) -> dict:
    approval.register_gateway_notify(ctx.session, lambda data: None)
    approver = threading.Thread(target=_auto_approve, args=(ctx.session,))
    approver.start()
    setup = ow.bootstrap(
        ctx, idempotency_key=f"dormant-{time.monotonic()}", name="Dormant Work",
    )
    approver.join()
    kanban_db.write_board_dispatch_state(setup["board"], dispatch_enabled=True)
    return setup


def _set_status(setup: dict, task_id: str, status: str) -> None:
    with kanban_db.connect(board=setup["board"]) as conn:
        with kanban_db.write_txn(conn):
            conn.execute(
                "UPDATE tasks SET status = ? WHERE id = ?", (status, task_id)
            )


def _execution(setup: dict) -> dict:
    return ow.project_steward_snapshot(project_id=setup["project_id"])["execution"]


_WAITING = {
    "state": "waiting_for_approval",
    "summary": "Raphael is waiting for an approved milestone before starting work.",
    "paused": False,
}


def test_only_parked_work_does_not_read_working(ctx):
    setup = _project(ctx)
    with kanban_db.connect(board=setup["board"]) as conn:
        parked = kanban_db.create_task(
            conn, title="Book the workshop room", project_id=setup["project_id"],
        )
    _set_status(setup, parked, kanban_db.PARKED_STATUS)

    assert _execution(setup) == _WAITING


def test_todo_waiting_only_on_parked_work_does_not_read_working(ctx):
    setup = _project(ctx)
    with kanban_db.connect(board=setup["board"]) as conn:
        parked = kanban_db.create_task(
            conn, title="Book the workshop room", project_id=setup["project_id"],
        )
        waiting = kanban_db.create_task(
            conn,
            title="Send the room details",
            project_id=setup["project_id"],
            parents=(parked,),
        )
    _set_status(setup, parked, kanban_db.PARKED_STATUS)
    _set_status(setup, waiting, "todo")

    assert _execution(setup) == _WAITING


def test_running_work_still_reads_working(ctx):
    setup = _project(ctx)
    with kanban_db.connect(board=setup["board"]) as conn:
        parked = kanban_db.create_task(
            conn, title="Book the workshop room", project_id=setup["project_id"],
        )
        running = kanban_db.create_task(
            conn, title="Draft the agenda", project_id=setup["project_id"],
        )
    _set_status(setup, parked, kanban_db.PARKED_STATUS)
    _set_status(setup, running, "running")

    assert _execution(setup) == {
        "state": "working",
        "summary": "Raphael is coordinating the approved milestone.",
        "paused": False,
    }
