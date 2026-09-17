"""Regression test: API server startup resumes interrupted removal operations."""
from __future__ import annotations
from unittest.mock import MagicMock

import json
import time
import pytest
from hermes_cli import kanban_db, owner_workspace as ow, projects_db
from gateway.platforms.api_server import APIServerAdapter
from gateway.config import PlatformConfig


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
        return None

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

        # Assert: the drive was dispatched
        assert driven == [slug], f"Expected drive for {slug}, got {driven}"

        # Assert: the operation advanced to terminal phase
        with projects_db.connect_closing() as c:
            op_after = projects_db.get_removal_operation(c, pid, idempotency_key)
        assert op_after["phase"] == "done", \
            f"Expected phase 'done', got '{op_after['phase']}'"

    finally:
        await adapter.disconnect()
