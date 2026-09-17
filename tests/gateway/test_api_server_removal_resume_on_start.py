"""Regression test: API server startup resumes interrupted removal operations."""
from __future__ import annotations
from unittest.mock import MagicMock

import asyncio
import json
import time
import pytest

pytest.importorskip("aiohttp")

from hermes_cli import kanban_db, owner_workspace as ow, projects_db  # noqa: E402
from gateway.platforms.api_server import APIServerAdapter  # noqa: E402
from gateway.config import PlatformConfig  # noqa: E402


_n = [0]


def _setup_board_and_operation(monkeypatch):
    """Create a test board with a non-terminal removal operation."""
    _n[0] += 1
    slug = f"resume_test_{_n[0]}"

    # Create project and board
    with projects_db.connect_closing() as c:
        pid = projects_db.create_project(c, name="ResumeTest", slug=slug,
                                         primary_path=f"/tmp/{slug}")
        projects_db.update_project(c, pid, board_slug=slug)

    # Ensure owner_workspace schema exists
    pc = projects_db.connect()
    try:
        ow._ensure_schema(pc)
        from hermes_cli.sqlite_util import write_txn
        with write_txn(pc):
            pc.execute(
                "INSERT INTO owner_workspace_receipts "
                "(actor,profile,idempotency_key,operation,request_digest,"
                "status,project_id,board_slug,result_json,"
                "terminal_generation,created_at,updated_at) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                ("default", "default", f"boot_{_n[0]}", "owner_workspace_bootstrap",
                 "d", "committed", pid, slug,
                 json.dumps({"ok": True, "project_id": pid}),
                 0, int(time.time()), int(time.time()))
            )
    finally:
        pc.close()

    # Seed a non-terminal removal operation (simulating an interrupted drive)
    idempotency_key = f"removal_{_n[0]}"
    with projects_db.connect_closing() as c:
        projects_db.record_removal_operation(
            c, project_id=pid, idempotency_key=idempotency_key,
            action="start", phase="carried", mode="reversible",
            board_slug=slug, removal_id="rm_test"
        )

    # Mock board ownership assertion (required by _dispatch_removal_drive)
    monkeypatch.setattr(ow, "_assert_board_ownership", lambda *a, **kw: None)

    return pid, slug, idempotency_key


@pytest.mark.asyncio
async def test_api_server_startup_resumes_removal_operations(monkeypatch):
    """Startup calls resume_removal_operations(), advancing interrupted operations to terminal."""
    pid, slug, idempotency_key = _setup_board_and_operation(monkeypatch)

    # Verify initial state: operation is non-terminal
    with projects_db.connect_closing() as c:
        op_before = projects_db.get_removal_operation(c, pid, idempotency_key)
    assert op_before["phase"] == "carried"

    # Mock kanban_db to simulate drive completing
    phase_rec = MagicMock()
    phase_rec.phase = MagicMock()
    phase_rec.phase.value = "done"
    phase_rec.mode = kanban_db.RemovalMode.REVERSIBLE
    phase_rec.removal_id = "rm_test"

    driven = []
    def _fake_drive(board, *a, **kw):
        driven.append(board)
        return kanban_db.FencedRemovalResult(True, "removal complete")

    monkeypatch.setattr(kanban_db, "drive_removal", _fake_drive)
    monkeypatch.setattr(kanban_db, "get_removal_phase_record",
                        lambda *a, **kw: phase_rec)

    # Patch _recover_orphaned_owner_jobs to avoid side effects
    monkeypatch.setattr(
        "gateway.platforms.api_server.APIServerAdapter._recover_orphaned_owner_jobs",
        lambda self: None
    )

    # Create adapter with port 0 (OS picks available port) and a valid key
    # Key must be >= 16 chars to avoid the placeholder/too-short rejection
    test_key = "test-key-1234567890abcdef"
    monkeypatch.setenv("API_SERVER_KEY", test_key)
    adapter = APIServerAdapter(
        PlatformConfig(
            enabled=True,
            extra={"host": "127.0.0.1", "port": 0, "key": test_key}
        )
    )

    try:
        # Call the real connect() - this is the production startup path
        await adapter.connect()

        # Join background threads (removal drives run in threads)
        for t in list(ow._removal_background_threads):
            t.join(timeout=20)

        # connect() runs under a running event loop, so _dispatch_removal_drive
        # takes the asyncio branch and registers the drive in
        # ow._removal_background_tasks rather than as a thread; await those
        # tasks too or the assertions below race a still-running drive.
        await asyncio.wait_for(
            asyncio.gather(*ow._removal_background_tasks, return_exceptions=True),
            timeout=30,
        )

        # Assert: the drive was dispatched
        assert driven == [slug], f"Expected drive for {slug}, got {driven}"

        # Assert: the operation advanced to terminal phase
        with projects_db.connect_closing() as c:
            op_after = projects_db.get_removal_operation(c, pid, idempotency_key)
        assert op_after["phase"] == "done", \
            f"Expected phase 'done', got '{op_after['phase']}'"

    finally:
        await adapter.disconnect()


def _setup_real_board_operation(monkeypatch):
    """The same setup, on a real fenced board with a real recorded intent."""
    from tests.hermes_cli._kanban_fence_support import create_fenced_board

    pid, slug, key = _setup_board_and_operation(monkeypatch)
    create_fenced_board(slug)
    assert kanban_db.record_removal_intent(
        slug, mode=kanban_db.RemovalMode.REVERSIBLE, removal_id="rm_test",
    ).success
    return pid, slug, key


@pytest.mark.asyncio
async def test_removal_drive_is_not_queued_behind_the_server_run_pool(monkeypatch):
    """A removal drive must not wait behind the server's blocking-run pool:
    the startup resume must drive a removal to completion while every
    thread of the loop's default executor is held by blocking run work, and
    a drive that REFUSES must say so on the operation. This guards the
    executor-coupling fix on its own merits; it is not a reproduction of
    the previously reported stall, whose actual cause (a sweep refusal) is
    covered by test_removal_resume_stalls_on_indeterminate_sweep_then_recovers
    below."""
    import concurrent.futures
    import threading

    pid, slug, key = _setup_real_board_operation(monkeypatch)
    # No recorded removal at all: the driver REFUSES BY RETURN VALUE there.
    refused_pid, _, refused_key = _setup_board_and_operation(monkeypatch)

    loop = asyncio.get_running_loop()
    release = threading.Event()
    busy = concurrent.futures.ThreadPoolExecutor(max_workers=2)
    loop.set_default_executor(busy)
    # The server's blocking-run pool, fully occupied as it is mid-run.
    held = [loop.run_in_executor(busy, release.wait) for _ in range(2)]
    try:
        ow.resume_removal_operations()          # the startup entry point
        # Owner-facing durable state only, while the pool is still saturated.
        deadline = time.monotonic() + 30
        while True:
            with projects_db.connect_closing() as c:
                completed = projects_db.get_removal_operation(c, pid, key)
                refused = projects_db.get_removal_operation(
                    c, refused_pid, refused_key)
            if completed["phase"] == "done" and refused["last_error"]:
                break
            if time.monotonic() >= deadline:
                break
            await asyncio.sleep(0.1)
    finally:
        release.set()
        await asyncio.gather(*held, return_exceptions=True)
        busy.shutdown(wait=True)
    assert completed["phase"] == "done", completed["phase"]
    assert completed["retained_copy_id"] is not None
    assert completed["receipt_id"] == key
    assert completed["last_error"] is None
    # The refusal is recorded, and the phase is left on its durable value
    # rather than re-stamped from a partial kernel record.
    assert refused["last_error"] == ow._REMOVAL_SAFE_ERRORS["driver_failed"]
    assert refused["phase"] == "carried"


def test_removal_resume_stalls_on_indeterminate_sweep_then_recovers(monkeypatch):
    """The previously reported stall was NOT the server's run-pool: a drive
    that is never dispatched leaves the kernel phase at 'intent' with no
    gate/quiesce/carry stamps and the board directory still present — the
    opposite of what was observed. The real mechanism is a sweep record
    class (§6.7) that reads INDETERMINATE (e.g. a locked store):
    advance_removal_to_swept correctly REFUSES rather than treat a failed
    read as proof of absence, so the kernel phase legitimately holds at
    'applied' with those stamps set and the board directory already gone.
    Restart resume must retry the sweep — re-refusing while the condition
    persists, and completing to 'done' once it clears."""
    pid, slug, key = _setup_real_board_operation(monkeypatch)

    original_sweep_record_class = kanban_db._sweep_record_class
    state = {"indeterminate": True, "sweep_attempts": 0}

    def _flaky_sweep_record_class(rec_key, board_slug, *, reversible, record):
        if rec_key == "subscriptions":
            state["sweep_attempts"] += 1
            if state["indeterminate"]:
                return {
                    "result": kanban_db.SWEEP_RESULT_INDETERMINATE,
                    "key": f"locked:{board_slug}",
                    "reason": "simulated: database is locked",
                }
        return original_sweep_record_class(
            rec_key, board_slug, reversible=reversible, record=record,
        )

    monkeypatch.setattr(kanban_db, "_sweep_record_class", _flaky_sweep_record_class)

    def _wait_until(predicate, timeout=20.0, interval=0.02):
        deadline = time.monotonic() + timeout
        while True:
            if predicate():
                return True
            if time.monotonic() >= deadline:
                return False
            time.sleep(interval)

    def _owner_op():
        with projects_db.connect_closing() as c:
            return projects_db.get_removal_operation(c, pid, key)

    # (a) The drive rolls all the way forward to Applied, then the sweep
    # refuses: the owner operation records the refusal.
    ow.resume_removal_operations()
    assert _wait_until(lambda: state["sweep_attempts"] >= 1), \
        "the sweep was never attempted"
    assert _wait_until(
        lambda: (_owner_op() or {})["last_error"]
        == ow._REMOVAL_SAFE_ERRORS["driver_failed"]
    ), "the owner operation never recorded the sweep refusal"

    phase_rec = kanban_db.get_removal_phase_record(slug)
    op = _owner_op()
    assert phase_rec.phase == kanban_db.RemovalPhase.APPLIED, phase_rec.phase
    assert phase_rec.gate_closed_at is not None
    assert phase_rec.quiesce_completed_at is not None
    assert phase_rec.carry_completed_at is not None
    assert phase_rec.outcome is None
    assert not kanban_db.board_dir(slug).exists()
    assert op["last_error"] == ow._REMOVAL_SAFE_ERRORS["driver_failed"]

    # (b) Resume while the indeterminate condition is STILL present must
    # not falsely advance the phase.
    attempts_before = state["sweep_attempts"]
    ow.resume_removal_operations()
    assert _wait_until(lambda: state["sweep_attempts"] > attempts_before), \
        "the resumed drive never re-attempted the sweep"
    time.sleep(0.2)  # let the durable write following this attempt land
    phase_rec = kanban_db.get_removal_phase_record(slug)
    op = _owner_op()
    assert phase_rec.phase == kanban_db.RemovalPhase.APPLIED, phase_rec.phase
    assert op["last_error"] == ow._REMOVAL_SAFE_ERRORS["driver_failed"]

    # (c) Once the condition clears, resume completes to Done.
    state["indeterminate"] = False
    ow.resume_removal_operations()
    assert _wait_until(
        lambda: kanban_db.get_removal_phase_record(slug).phase
        == kanban_db.RemovalPhase.DONE
    ), "the removal never completed once the sweep could be read"
    phase_rec = kanban_db.get_removal_phase_record(slug)
    op = _owner_op()
    assert phase_rec.outcome == "archived", phase_rec.outcome
    assert op["applied_at"] is not None
    assert op["completed_at"] is not None
    assert op["last_error"] is None
