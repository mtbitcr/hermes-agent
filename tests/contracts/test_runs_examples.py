"""Run examples (contract map section 5, test 3).

Every example comes from the adapter's real ``/v1/runs`` routes, registered by
the existing ``_create_runs_app``. An owner retry run goes through the real
kernel and its real confirmation, answered ``once`` and, for a second Project,
``deny``; a retry of work outside the Project gets the kernel's real refusal
and code. Two runs of a stub agent ask the real approval gate for a write,
answered ``session`` and ``always``. A plain run uses a stub agent that calls
the real progress and stream callbacks, so every event kind is produced without
a model. Each ``hermes.run`` status is recorded as
``_set_run_status`` returns it, which is what ``GET /v1/runs/{run_id}`` serves
at that moment; ``queued`` and ``stopping`` last too briefly to poll. The
restart failure is what the same route serves after the real recovery of a
run whose executor is gone. What is asserted is described in
tests/contracts/conftest.py; each closed vocabulary is checked in the live
answers and in the saved examples alike. The last two tests check the shared
frozen ids and clock through real producers, as no setup module is among the
owned paths.
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
import uuid
from unittest.mock import MagicMock, patch

import pytest
from aiohttp.test_utils import TestClient, TestServer

from cron import jobs as cron_jobs
from gateway.platforms import api_server
from tests.gateway import test_api_server_runs as runs_tests
from tests.gateway.test_api_server_runs import (  # noqa: F401  (an autouse fixture)
    _create_runs_app,
    _make_adapter,
    _make_slow_agent,
    _owner_retry_run_setup,
    _resolved_owner_task_routes,
)
from tools import approval as approval_mod

FAMILY = "runs"
RUN_STATUSES = {
    "queued", "running", "waiting_for_approval", "completed", "failed", "cancelled",
    "stopping",
}
EVENT_KINDS = {
    "tool.started", "tool.completed", "reasoning.available", "subagent.start",
    "subagent.complete", "message.delta", "approval.request", "approval.responded",
    "run.steered", "run.completed", "run.failed", "run.cancelled",
}
APPROVAL_CHOICES = {"once", "session", "always", "deny"}
FINISHED = {"completed", "failed", "cancelled"}
OWNER_CONFIG = {"gateway": {"api_server": {"owner_workspace": {"enabled": True}}}}
# What the delegate tool's relay sends with each subagent event
# (tools/delegate_tool_progress.py ``_identity_kwargs``), and what a finished
# child adds (tools/delegate_tool_child_run.py ``complete_kwargs``); the
# producer keeps only its own allowlist of these.
SUBAGENT = {
    "task_index": 0, "task_count": 1, "goal": "Placeholder goal.",
    "subagent_id": "subagent-1", "parent_id": "parent-1", "depth": 1,
    "model": "placeholder-model", "toolsets": ["file"], "child_session_id": "session-1",
    "delegation_id": "delegation-1", "tool_count": 1,
}
SUBAGENT_RESULT = {
    "status": "completed", "duration_seconds": 1.0, "summary": "Placeholder summary.",
    "input_tokens": 1, "output_tokens": 1, "reasoning_tokens": 0, "api_calls": 1,
    "files_read": ["plan.md"], "files_written": ["agenda.md"], "cost_usd": 0.0,
    "output_tail": [{"tool": "read_file", "preview": "Placeholder text.", "is_error": False}],
}


def _recorded_statuses(adapter, monkeypatch) -> dict:
    """Each ``hermes.run`` status, as the producer first sets it."""
    statuses: dict = {}
    produce = adapter._set_run_status

    def _set_run_status(run_id, status, **fields):
        current = produce(run_id, status, **fields)
        statuses.setdefault(status, json.loads(json.dumps(current)))
        return current

    monkeypatch.setattr(adapter, "_set_run_status", _set_run_status)
    return statuses


def _stub_agent(steered: threading.Event):
    """Stands in for ``_create_agent``: every progress kind, a delta, then a steer."""

    def _create_agent(**kwargs):
        progress = kwargs["tool_progress_callback"]
        agent = MagicMock()
        agent.session_prompt_tokens = 0
        agent.session_completion_tokens = 0
        agent.session_total_tokens = 0
        agent.steer.side_effect = lambda text: steered.set() or True

        def _run_conversation(*_args, **_kwargs):
            progress("tool.started", "read_file", "plan.md", {"path": "plan.md"})
            progress("tool.completed", "read_file", None, None, duration=0.25, is_error=False)
            progress("reasoning.available", None, "Placeholder reasoning.")
            progress("subagent.start", None, "Placeholder goal.", None, **SUBAGENT)
            progress("subagent.complete", None, "Placeholder summary.", None,
                     **SUBAGENT, **SUBAGENT_RESULT)
            kwargs["stream_delta_callback"]("Placeholder answer.")
            steered.wait(timeout=10)
            return {"final_response": "Placeholder answer."}

        agent.run_conversation.side_effect = _run_conversation
        return agent

    return _create_agent


def _gated_agent(**_kwargs):
    """Stands in for ``_create_agent``: a write the real approval gate must allow."""
    agent = MagicMock()
    agent.session_prompt_tokens = 0
    agent.session_completion_tokens = 0
    agent.session_total_tokens = 0

    def _run_conversation(user_message, **_kwargs):
        # Each input gives its own reason, so one answer never covers another run.
        gate = approval_mod.request_tool_approval("write_file", f"Write {user_message}")
        return {"final_response": json.dumps(gate)}

    agent.run_conversation.side_effect = _run_conversation
    return agent


async def _answer(response) -> dict:
    return {"status": response.status, "body": await response.json()}


async def _poll(client, run_id: str, until) -> dict:
    for _ in range(400):
        answer = await _answer(await client.get(f"/v1/runs/{run_id}"))
        if until(answer["body"]):
            return answer
        await asyncio.sleep(0.05)
    raise AssertionError(f"run {run_id} never got there: {answer}")


async def _events(client, run_id: str) -> list[dict]:
    payload = await (await client.get(f"/v1/runs/{run_id}/events")).text()
    return [
        json.loads(line[len("data: "):])
        for line in payload.splitlines() if line.startswith("data: ")
    ]


@pytest.mark.asyncio
async def test_the_run_examples_carry_every_status_event_and_answer(
    monkeypatch, owner_payload_example,
):
    adapter = _make_adapter()
    statuses = _recorded_statuses(adapter, monkeypatch)
    # An ``always`` answer joins a fresh allowlist, saved to this test's home.
    monkeypatch.setattr(approval_mod, "_permanent_approved", set())
    key, refused_key = "owner-task-retry-contract", "owner-task-retry-contract-refused"
    denied_key = "owner-task-retry-contract-denied"
    _created, _task_id, _payload, body = _owner_retry_run_setup(key)
    denied_body = _owner_retry_run_setup(denied_key)[-1]
    # A retry of work outside the Project: the kernel refuses it with its code.
    refused_body = json.loads(json.dumps(body))
    refused_body["owner_retry_authority"]["idempotency_key"] = refused_key
    refused_body["owner_retry_authority"]["payload"].update(
        idempotency_key=refused_key, task_id="t_outside_the_project")
    steered = threading.Event()
    slow_agent, slow_ready, _interrupted = _make_slow_agent()
    live: dict = {"approval_requests": [], "approval_responses": []}

    with patch("gateway.run._load_gateway_config", return_value=OWNER_CONFIG):
        async with TestClient(TestServer(_create_runs_app(adapter))) as client:

            async def approve(run_id: str, choice: str) -> list[dict]:
                """Answer the approval the run waits on; its events once it ends."""
                waiting = (await _poll(client, run_id, lambda status: (
                    status["status"] == "waiting_for_approval")))["body"]
                live["approval_requests"].append(waiting)
                live["approval_responses"].append(await _answer(await client.post(
                    f"/v1/runs/{run_id}/approval", json={
                        "choice": choice,
                        "approval_id": waiting["pending_approval"]["approval_id"]},
                )))
                await _poll(client, run_id, lambda status: status["status"] in FINISHED)
                return await _events(client, run_id)

            with patch.object(adapter, "_create_agent", side_effect=RuntimeError("no model")):
                started = await client.post(
                    "/v1/runs", json=body, headers={"Idempotency-Key": key})
                live["run_started"] = await _answer(started)
                events = await approve(live["run_started"]["body"]["run_id"], "once")
                refused = await client.post(
                    "/v1/runs", json=refused_body, headers={"Idempotency-Key": refused_key})
                refused_run = (await refused.json())["run_id"]
                await _poll(client, refused_run, lambda status: status["status"] == "failed")
                events += await _events(client, refused_run)
                denied = await client.post(
                    "/v1/runs", json=denied_body, headers={"Idempotency-Key": denied_key})
                events += await approve((await denied.json())["run_id"], "deny")

            with patch.object(adapter, "_create_agent", side_effect=_gated_agent):
                for choice in ("session", "always"):
                    gated = await client.post("/v1/runs", json={"input": f"the {choice} plan"})
                    events += await approve((await gated.json())["run_id"], choice)

            with patch.object(adapter, "_create_agent", side_effect=_stub_agent(steered)):
                plain_run = (await (await client.post(
                    "/v1/runs", json={"input": "Draft the agenda"})).json())["run_id"]
                await _poll(client, plain_run, lambda status: (
                    status.get("last_event") == "subagent.complete"))
                live["steer"] = await _answer(await client.post(
                    f"/v1/runs/{plain_run}/steer", json={"text": "Keep it short."}))
                await _poll(client, plain_run, lambda status: status["status"] == "completed")
                events += await _events(client, plain_run)

            with patch.object(adapter, "_create_agent", return_value=slow_agent):
                stopped_run = (await (await client.post(
                    "/v1/runs", json={"input": "Draft the agenda"})).json())["run_id"]
                assert await asyncio.to_thread(slow_ready.wait, 10)
                live["stop"] = await _answer(
                    await client.post(f"/v1/runs/{stopped_run}/stop"))
                await _poll(client, stopped_run, lambda status: status["status"] == "cancelled")
                events += await _events(client, stopped_run)

            live["run_not_found"] = await _answer(
                await client.get(f"/v1/runs/run_{'0' * 32}"))

    live["events"] = events
    live["approval_response"] = live["approval_responses"][0]
    live.update({f"status_{status}": record for status, record in statuses.items()})
    saved = {kind: owner_payload_example(FAMILY, kind, body) for kind, body in live.items()}

    for answers in (live, saved):
        assert {
            record["status"] for kind, record in answers.items() if kind.startswith("status_")
        } == RUN_STATUSES
        assert "pending_approval" in answers["status_waiting_for_approval"]
        assert "pending_approval" not in answers["status_completed"]
        assert {"error", "error_code", "error_reason"} <= set(answers["status_failed"])
        assert {event["event"] for event in answers["events"]} == EVENT_KINDS
        assert answers["approval_response"]["body"]["object"] == "hermes.run.approval_response"
        assert answers["steer"]["body"]["object"] == "hermes.run.steer"
        assert answers["stop"]["body"]["status"] == "stopping"
        assert answers["run_not_found"]["body"]["error"]["code"] == "run_not_found"
        assert answers["run_started"]["status"] == 202
        # Each answer is a choice its own pending approval offered; the stream
        # announced that approval with the same choices and echoes the answer.
        offered = {
            waiting["run_id"]: waiting["pending_approval"]["choices"]
            for waiting in answers["approval_requests"]
        }
        assert offered == {
            event["run_id"]: event["choices"]
            for event in answers["events"] if event["event"] == "approval.request"
        }
        responses = [response["body"] for response in answers["approval_responses"]]
        assert all(response["choice"] in offered[response["run_id"]] for response in responses)
        assert {(response["run_id"], response["choice"]) for response in responses} == {
            (event["run_id"], event["choice"])
            for event in answers["events"] if event["event"] == "approval.responded"
        }
        assert {response["choice"] for response in responses} == APPROVAL_CHOICES


@pytest.mark.asyncio
async def test_the_restart_failure_example_carries_the_restart_sentence(
    owner_payload_example,
):
    adapter = _make_adapter()
    # A queued owner run whose executor is gone, as the recovery tests leave it;
    # the class is reached through its module so it is not collected here.
    recovery = runs_tests.TestOrphanedRunRecovery()
    recovery._queued_owner_run(adapter)
    adapter._recover_orphaned_owner_jobs()

    async with TestClient(TestServer(_create_runs_app(adapter))) as client:
        live = await _answer(await client.get(f"/v1/runs/{recovery._RUN_ID}"))

    # Saved as the bare body, like every other ``status_*`` example.
    assert live["status"] == 200
    saved = owner_payload_example(FAMILY, "status_failed_restart", live["body"])
    assert saved["status"] == "failed"
    assert saved["error"] == api_server._OWNER_ORPHAN_RUN_MESSAGE


def _cron_job() -> dict:
    # A pinned route, so ``create_job`` does not resolve the configured provider.
    return cron_jobs.create_job(
        "Placeholder prompt.", "every 1h", model="placeholder-model",
        provider="placeholder-provider",
    )


def test_the_frozen_ids_stay_distinct_where_producers_cut_them():
    """Producers keep 12, 24 or 28 hex characters of a ``uuid4``."""
    assert _cron_job()["id"] != _cron_job()["id"]
    drawn = [uuid.uuid4() for _ in range(64)]
    for width in (12, 24, 28):
        assert len({value.hex[:width] for value in drawn}) == len(drawn)
    assert {(value.version, value.variant) for value in drawn} == {(4, uuid.RFC_4122)}


def test_the_frozen_clock_holds_still_for_real_producers():
    """Cron jobs created apart share their timestamps; monotonic time moves."""
    first, read, started = _cron_job(), cron_jobs._hermes_now(), time.monotonic()
    time.sleep(0.01)
    second = _cron_job()
    assert (second["created_at"], second["next_run_at"]) == (
        first["created_at"], first["next_run_at"])
    assert cron_jobs._hermes_now() == read
    assert time.monotonic() > started
