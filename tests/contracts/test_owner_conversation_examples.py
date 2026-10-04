"""Owner conversation examples (contract map section 5, test 2).

Every example comes from the adapter's real ``/v1/responses`` routes, registered
as the existing route tests register them, over the adapter's real response
store in the per-test home. The agent is replaced by a stub that calls the real
stream and tool callbacks, so every streamed event is produced without a model.
What is asserted is described in tests/contracts/conftest.py.
"""

from __future__ import annotations

import asyncio
import json
from unittest.mock import patch

import pytest
from aiohttp.test_utils import TestClient, TestServer

from gateway.platforms import api_server
from tests.gateway.test_api_server import (
    _create_app,
    _interrupt_owner_turn,
    _make_adapter,
    _owner_existing_proposal,
    _owner_new_proposal,
)
from tests.gateway.test_owner_turn_provider_unavailable import (
    _PROVIDER_FAILURE_REPLY,
    _QUESTION_REPLY,
    _REFUSAL_NOTICE,
    _failure_reply,
)

FAMILY = "owner_conversation"
REPLY_KINDS = {"question", "no_change", "proposal", "project_change_proposal", "failure"}
FAILURE_SENTENCES = {
    api_server._OWNER_INTERRUPTED_TURN_MESSAGE,
    api_server._OWNER_REFUSED_TURN_MESSAGE,
    api_server._OWNER_PROVIDER_UNAVAILABLE_TURN_MESSAGE,
}
STREAM_EVENTS = {
    "response.created", "response.output_item.added", "response.output_text.delta",
    "response.output_item.done", "response.output_text.done", "response.completed",
    "response.failed",
}
QUESTION = {"schema_version": 1, "kind": "question", "message": "Which outcome first?"}


def _name(digit: str) -> str:
    return "raphael-owner-" + digit * 32


def _seed(store, conversation: str, response_id: str, replies, **mapping) -> None:
    """One stored owner turn per reply, mapped as the conversation's head."""
    history = []
    for index, reply in enumerate(replies):
        history.append({"role": "user", "content": f"Placeholder request {index}."})
        history.append({"role": "assistant", "content": reply})
    store.put(response_id, {
        "response": {"id": response_id, "created_at": 800},
        "conversation_history": history,
    })
    assert store.set_conversation(conversation, response_id, **mapping) is True


def _stub_agent(*, fail: bool = False):
    """Stands in for ``_run_agent``: one tool call, then the structured reply."""
    final = json.dumps(QUESTION)

    async def _run(**kwargs):
        if fail:
            raise RuntimeError("agent died mid-stream")
        if kwargs.get("tool_start_callback"):
            kwargs["tool_start_callback"]("call_contract", "read_file", {"path": "plan.md"})
            kwargs["tool_complete_callback"](
                "call_contract", "read_file", {"path": "plan.md"}, "Placeholder text.",
            )
        if kwargs.get("stream_delta_callback"):
            kwargs["stream_delta_callback"](final)
        return (
            {"final_response": final, "messages": [], "api_calls": 1},
            {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
        )

    return _run


def _turn(conversation: str, previous=None, **extra) -> dict:
    return {
        "model": "hermes-agent", "input": "Prepare the private milestone",
        "conversation": conversation, "store": True,
        "expected_previous_response_id": previous, **extra,
    }


async def _answer(response) -> dict:
    return {"status": response.status, "body": await response.json()}


def _events(payload: str) -> list[dict]:
    events = []
    for block in payload.split("\n\n"):
        lines = block.splitlines()
        names = [line[len("event: "):] for line in lines if line.startswith("event: ")]
        data = [line[len("data: "):] for line in lines if line.startswith("data: ")]
        if names and data:
            events.append({"event": names[0], "data": json.loads(data[0])})
    return events


@pytest.fixture(autouse=True)
def _fresh_idempotency_cache(monkeypatch):
    monkeypatch.setattr(api_server, "_idem_cache", api_server._IdempotencyCache())


@pytest.mark.asyncio
async def test_the_history_examples_carry_every_reply_kind_and_handle(owner_payload_example):
    adapter = _make_adapter()
    store = adapter._response_store
    _seed(store, _name("1"), "resp_every_kind", [
        _QUESTION_REPLY,
        json.dumps({"schema_version": 1, "kind": "no_change", "message": "Nothing to change."}),
        json.dumps(_owner_new_proposal()),
        json.dumps(_owner_existing_proposal()),
        _failure_reply(api_server._OWNER_INTERRUPTED_TURN_MESSAGE),
    ])
    for digit, reply in (("2", "Placeholder raw text."), ("4", _REFUSAL_NOTICE),
                         ("5", _PROVIDER_FAILURE_REPLY)):
        _seed(store, _name(digit), f"resp_failure_{digit}", [_QUESTION_REPLY, reply])
    assert store.reserve_owner_conversation(
        "default", _name("d"), "resp_pending_turn", owner_message="Prepare the workshop",
    ) is True
    assert store.reserve_owner_conversation(
        "default", _name("3"), "resp_recovery", owner_message="Prepare the workshop",
    ) is True
    _interrupt_owner_turn(store, _name("3"), "resp_recovery")
    group, session = _name("7"), "8" * 32
    _seed(store, f"{group}-{session}", "resp_proposal",
          [json.dumps(_owner_existing_proposal())], owner_proposal=True)

    async with TestClient(TestServer(_create_app(adapter))) as client:
        async def history(conversation, **params):
            return await _answer(await client.get(
                f"/v1/responses/conversations/{conversation}", params=params or None,
            ))

        async def post(conversation, route, payload):
            return await _answer(await client.post(
                f"/v1/responses/conversations/{conversation}/{route}", json=payload,
            ))

        live = {
            "history": await history(_name("1")),
            "history_failure_turns": [
                await history(_name(digit)) for digit in ("2", "4", "5")
            ],
            "history_empty": await history(_name("6")),
            "history_pending": await history(_name("d")),
            "history_recovery": await history(_name("3")),
            "sessions": await history(group, view="sessions"),
            "proposal_consumption": await post(
                f"{group}-{session}", "consume", {"response_id": "resp_proposal"}),
            "proposal_consumption_refused": await post(
                f"{group}-{session}", "consume", {"response_id": "resp_elsewhere"}),
            "recovery_acknowledgements": [
                await post(_name("3"), "recovery", {"response_id": response_id})
                for response_id in ("resp_elsewhere", "resp_recovery", "resp_recovery")
            ],
        }
        with patch.object(store, "owner_history_snapshot",
                          side_effect=api_server.OwnerAuthorityBroken("unreadable")):
            live["owner_history_unavailable"] = await history(_name("1"))

    saved = {kind: owner_payload_example(FAMILY, kind, body) for kind, body in live.items()}

    turns = saved["history"]["body"]["data"] + [
        example["body"]["data"][-1] for example in saved["history_failure_turns"]
    ]
    replies = [json.loads(turn["raphael"]) for turn in turns]
    assert {reply["kind"] for reply in replies} == REPLY_KINDS
    assert FAILURE_SENTENCES <= {
        reply["message"] for reply in replies if reply["kind"] == "failure"
    }
    assert saved["history_empty"]["body"]["data"] == []
    assert saved["history_pending"]["body"]["pending"] is not None
    assert saved["history_recovery"]["body"]["recovery"] is not None
    assert {answer["body"]["outcome"] for answer in saved["recovery_acknowledgements"]} == {
        "mismatch", "retired", "absent",
    }
    assert saved["proposal_consumption_refused"]["status"] == 409
    assert saved["owner_history_unavailable"]["body"]["error"]["code"] == (
        "owner_history_unavailable")


@pytest.mark.asyncio
async def test_the_authority_examples_carry_every_answer(owner_payload_example):
    adapter = _make_adapter()
    store = adapter._response_store
    conversation, response_id = _name("a"), "resp_authority"
    claim_id, run_id = "claim_" + "a" * 32, "run_" + "b" * 32
    _seed(store, conversation, response_id,
          [json.dumps(_owner_existing_proposal())], owner_proposal=True)

    async with TestClient(TestServer(_create_app(adapter))) as client:
        async def authority(action, **extra):
            return await _answer(await client.post(
                f"/v1/responses/conversations/{conversation}/authority",
                json={"action": action, "response_id": response_id, **extra},
            ))

        live = [
            await authority("claim", claim_id=claim_id),
            await authority("abandon", claim_id=claim_id),
            await authority("claim", claim_id=claim_id),
            await authority("complete", claim_id=claim_id, run_id=run_id),
        ]
        adapter._set_run_status(run_id, "running")
        assert store.attach_owner_run(
            "default", conversation, response_id, claim_id, run_id) is True
        live.append(await authority("attach", claim_id=claim_id, run_id=run_id))
        assert store.complete_owner_claim(
            "default", conversation, response_id, claim_id, run_id) is True
        live.append(await authority("complete", claim_id=claim_id, run_id=run_id))
        live.append(await authority("close"))

    saved = owner_payload_example(FAMILY, "authority", live)
    assert {answer["body"].get("action") for answer in saved if answer["status"] == 200} == {
        "claim", "abandon", "attach", "complete", "close",
    }
    assert any(answer["status"] == 409 for answer in saved)


@pytest.mark.asyncio
async def test_the_response_examples_carry_every_object_event_and_refusal(
    owner_payload_example,
):
    adapter = _make_adapter()
    store = adapter._response_store
    _seed(store, _name("1"), "resp_head_turn", [_QUESTION_REPLY])
    assert store.reserve_owner_conversation(
        "default", _name("d"), "resp_pending_turn", owner_message="Prepare the workshop",
    ) is True

    async with TestClient(TestServer(_create_app(adapter))) as client:
        async def respond(body, key):
            return await client.post(
                "/v1/responses", json=body, headers={"Idempotency-Key": key},
            )

        with patch.object(adapter, "_run_agent", side_effect=_stub_agent()):
            live = {"response": await _answer(await respond(_turn(_name("2")), "contract-1"))}
            live["idempotency_conflict"] = await _answer(await respond(
                _turn(_name("2"), input="A different request"), "contract-1"))
            live["owner_conversation_stale"] = await _answer(await respond(
                _turn(_name("1"), previous="resp_elsewhere"), "contract-2"))
            live["owner_conversation_locked"] = await _answer(await respond(
                _turn(_name("d")), "contract-3"))
            with patch.object(store, "lookup_owner_response",
                              return_value=("incomplete", None, None)):
                live["owner_response_incomplete"] = await _answer(await respond(
                    _turn(_name("4")), "contract-4"))
            queued = await respond(_turn(_name("5"), background=True), "contract-5")
            live["response_background_queued"] = await _answer(queued)
            path = f"/v1/responses/{live['response_background_queued']['body']['id']}"
            for _ in range(200):
                live["response_stored"] = await _answer(await client.get(path))
                if live["response_stored"]["body"].get("status") == "completed":
                    break
                await asyncio.sleep(0.02)
            streamed = await respond(_turn(_name("6"), stream=True), "contract-6")
            live["streamed_events"] = _events(await streamed.text())
        with patch.object(adapter, "_run_agent", side_effect=_stub_agent(fail=True)):
            failed = await respond(_turn(_name("8"), stream=True), "contract-7")
            live["streamed_failure_events"] = _events(await failed.text())

    saved = {kind: owner_payload_example(FAMILY, kind, body) for kind, body in live.items()}

    assert {
        example["body"]["object"] for kind, example in saved.items()
        if kind.startswith("response")
    } == {"response"}
    assert saved["response_background_queued"]["body"]["status"] == "queued"
    assert saved["response_stored"]["body"]["status"] == "completed"
    assert {
        event["event"] for kind in ("streamed_events", "streamed_failure_events")
        for event in saved[kind]
    } == STREAM_EVENTS
    assert {
        saved[kind]["body"]["error"]["code"]: saved[kind]["status"] for kind in (
            "idempotency_conflict", "owner_conversation_stale",
            "owner_conversation_locked", "owner_response_incomplete",
        )
    } == {
        "idempotency_conflict": 409, "owner_conversation_stale": 409,
        "owner_conversation_locked": 409, "owner_response_incomplete": 409,
    }
