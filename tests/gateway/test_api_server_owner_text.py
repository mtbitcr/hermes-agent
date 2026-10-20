"""Owner-facing run and history text through the one owner text policy.

A failed owner run's reason and the planner prose of a readable history reply
reach the owner through :func:`hermes_cli.owner_text.check_owner_text`. Both
are checked where they are read, never where they are stored: the transport
status, the queued event, the durable run row and the stored transcript keep
exactly what they said.
"""

import asyncio
import hashlib
import json
from unittest.mock import patch

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import PlatformConfig
from gateway.platforms.api_server import APIServerAdapter, ResponseStore
from hermes_cli.owner_text import check_owner_text

# Each kind's fixed sentence, read from the one policy: text that is not a
# string always gets it.
_STORED_FAILURE = check_owner_text(None, "stored_failure")
_PLANNER_PROSE = check_owner_text(None, "planner_prose")

_PATH_FAILURE = "could not open /srv/placeholder/notes.txt for reading"
_PLAIN_FAILURE = "The calendar was busy, so the booking did not finish."
_RUN_ID = "run_" + "b" * 32
_NAME = "raphael-owner-" + "8" * 32
_RESPONSE_ID = "resp_owner_text"


def _adapter():
    return APIServerAdapter(PlatformConfig(enabled=True, extra={}))


def _runs_app(adapter):
    app = web.Application()
    app["api_server_adapter"] = adapter
    app.router.add_post("/v1/runs", adapter._handle_runs)
    app.router.add_get("/v1/runs/{run_id}", adapter._handle_get_run)
    app.router.add_get("/v1/runs/{run_id}/events", adapter._handle_run_events)
    return app


async def _failed_run(failure):
    """Start a run whose agent cannot be built and poll it until it fails."""
    adapter = _adapter()
    async with TestClient(TestServer(_runs_app(adapter))) as cli:
        with patch.object(
            adapter, "_create_agent", side_effect=RuntimeError(failure),
        ):
            resp = await cli.post("/v1/runs", json={"input": "hello"})
            assert resp.status == 202
            run_id = (await resp.json())["run_id"]
            for _ in range(40):
                status = await (await cli.get(f"/v1/runs/{run_id}")).json()
                if status["status"] == "failed":
                    break
                await asyncio.sleep(0.05)
    assert status["status"] == "failed"
    return adapter, run_id, status


@pytest.mark.asyncio
async def test_a_failed_owner_run_reason_is_the_fixed_sentence():
    adapter, run_id, status = await _failed_run(_PATH_FAILURE)

    assert status["error"] == _STORED_FAILURE
    assert status["last_event"] == "run.failed"
    # Only the read changed: the transport status still holds the reason.
    assert adapter._run_statuses[run_id]["error"] == _PATH_FAILURE


@pytest.mark.asyncio
async def test_a_plain_failure_sentence_passes_unchanged():
    _adapter_, _run_id, status = await _failed_run(_PLAIN_FAILURE)

    assert status["error"] == _PLAIN_FAILURE


@pytest.mark.asyncio
async def test_a_failed_owner_run_event_carries_the_fixed_sentence():
    adapter = _adapter()
    event = {
        "event": "run.failed",
        "run_id": _RUN_ID,
        "timestamp": 1.0,
        "error": _PATH_FAILURE,
    }
    queue = asyncio.Queue()
    queue.put_nowait(event)
    queue.put_nowait(None)
    adapter._run_streams[_RUN_ID] = queue

    async with TestClient(TestServer(_runs_app(adapter))) as cli:
        body = await (await cli.get(f"/v1/runs/{_RUN_ID}/events")).text()

    (frame,) = [
        json.loads(line[len("data: "):])
        for line in body.splitlines()
        if line.startswith("data: ")
    ]
    assert frame == {**event, "error": _STORED_FAILURE}
    # The queued event itself is untouched.
    assert event["error"] == _PATH_FAILURE


def test_a_refused_owner_mutation_reads_back_as_the_fixed_sentence():
    """The durable row keeps the refusal it recorded; its read is checked."""
    adapter = _adapter()
    store = adapter._response_store
    reason = "owner_task_retry: the task is not part of this receipt-owned Project"
    state, _existing = store.reserve_run_idempotency(
        "default", hashlib.sha256(b"").hexdigest(), "run-key-1", "fingerprint",
        _RUN_ID,
    )
    assert state == "new"
    store.reserve_owner_job("run", _RUN_ID, "default", {})
    adapter._run_statuses[_RUN_ID] = {
        "object": "hermes.run", "run_id": _RUN_ID, "status": "failed",
        "error": f"The approved Project change was not committed: {reason}",
        "error_reason": reason,
        "error_code": "task_not_found",
        "owner_mutation_committed": False,
    }
    adapter._finalize_run_recovery_job(_RUN_ID, "default")

    durable = store.run_idempotency_status("default", _RUN_ID)

    assert durable["status"] == "failed"
    assert durable["error"] == _STORED_FAILURE
    assert durable["error_code"] == "task_not_found"
    (stored,) = store._conn.execute(
        "SELECT status_json FROM run_idempotency WHERE run_id = ?", (_RUN_ID,),
    ).fetchone()
    assert json.loads(stored)["error"] == reason


def _owner_store(reply):
    store = ResponseStore(max_size=10)
    store.put(_RESPONSE_ID, {
        "response": {"id": _RESPONSE_ID},
        "conversation_history": [
            {"role": "user", "content": "Plan the workshop."},
            {"role": "assistant", "content": json.dumps(reply)},
        ],
    })
    assert store.set_conversation(_NAME, _RESPONSE_ID) is True
    return store


def _projected_reply(store):
    stored_before = list(store._conn.iterdump())
    snapshot = store.owner_history_snapshot(_NAME)
    # Only the projection changed: the store holds exactly what it held.
    assert list(store._conn.iterdump()) == stored_before
    (turn,) = snapshot["data"]
    assert turn["owner"] == "Plan the workshop."
    return json.loads(turn["raphael"])


def test_history_prose_with_an_id_is_replaced_field_by_field():
    question = {
        "schema_version": 1,
        "kind": "question",
        "message": "Should I start once t_0000abcd finishes?",
    }
    store = _owner_store(question)

    assert _projected_reply(store) == {**question, "message": _PLANNER_PROSE}
    (data,) = store._conn.execute(
        "SELECT data FROM responses WHERE response_id = ?", (_RESPONSE_ID,),
    ).fetchone()
    assert json.loads(json.loads(data)["conversation_history"][-1]["content"]) == (
        question
    )


def test_only_the_prose_fields_that_fail_are_replaced():
    task = {
        "title": "Draft the workshop plan",
        "body": "Edit /srv/placeholder/plan.md first.",
        "assignee": "default",
        "responsibility": "B03",
        "execution_tier": "routine",
        "risk_tier": 1,
        "owned_paths": ["docs/plan.md"],
    }
    proposal = {
        "schema_version": 4,
        "kind": "proposal",
        "mode": "new",
        "project_name": "Workshop pilot",
        "project_description": "A private workshop pilot.",
        "request_title": "Prepare the workshop",
        "summary": "Prepare the first milestone after t_0000abcd.",
        "project_size": "small",
        "specification": "Create one owner-visible workshop milestone.",
        "current_milestone": "Prepare the workshop",
        "owner_visible_result": "A reviewed workshop plan.",
        "impact": ["Adds one private milestone.", "Claude Opus drafts it."],
        "later_milestones": [],
        "tasks": [task],
    }
    store = _owner_store(proposal)

    assert _projected_reply(store) == {
        **proposal,
        "summary": _PLANNER_PROSE,
        "impact": ["Adds one private milestone.", _PLANNER_PROSE],
        "tasks": [{**task, "body": _PLANNER_PROSE}],
    }
