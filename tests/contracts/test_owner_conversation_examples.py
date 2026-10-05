"""Owner conversation examples (contract map section 5, test 2).

Every example comes from the adapter's real ``/v1/responses`` routes, registered
as the existing route tests register them, over the adapter's real response
store in the per-test home. The agent is replaced by a stub that calls the real
stream and tool callbacks, so every streamed event is produced without a model,
or parks until its task is cancelled. What is asserted is described in
tests/contracts/conftest.py; each closed vocabulary is checked wherever it
occurs, in the live answers and in every saved example alike, and the values
each takes must agree.
"""

from __future__ import annotations

import asyncio
import json
from unittest.mock import patch

import pytest
from aiohttp.test_utils import TestClient, TestServer

from gateway.platforms import api_server
from tests.contracts.conftest import OWNER_PAYLOADS
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
# Only the failure sentences are adapter constants. The rest are written out
# where they are produced: the reply kinds ``owner_history_snapshot`` projects
# (failures through ``_owner_failure_reply_is_projectable``), the outcomes of
# ``acknowledge_owner_conversation_recovery``, the action table of
# ``_handle_owner_conversation_authority``, the statuses ``_handle_responses``
# and ``_write_sse_responses`` give a response, and the events they stream.
REPLY_KINDS = {"question", "no_change", "proposal", "project_change_proposal", "failure"}
FAILURE_SENTENCES = {
    api_server._OWNER_INTERRUPTED_TURN_MESSAGE,
    api_server._OWNER_REFUSED_TURN_MESSAGE,
    api_server._OWNER_PROVIDER_UNAVAILABLE_TURN_MESSAGE,
}
RECOVERY_OUTCOMES = {"mismatch", "retired", "absent"}
RESPONSE_STATUSES = {"queued", "in_progress", "completed", "failed", "incomplete"}
STREAM_EVENTS = {
    "response.created", "response.output_item.added", "response.output_text.delta",
    "response.output_item.done", "response.output_text.done", "response.completed",
    "response.failed",
}
AUTHORITY_ACTIONS = {"claim", "abandon", "attach", "complete", "release", "reconcile", "close"}
VOCABULARIES = {
    "reply_kind": REPLY_KINDS, "failure_sentence": FAILURE_SENTENCES,
    "recovery_outcome": RECOVERY_OUTCOMES, "authority_action": AUTHORITY_ACTIONS,
    "response_status": RESPONSE_STATUSES, "stream_event": STREAM_EVENTS,
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


def _fields(record: dict):
    """``(vocabulary, value)`` for each closed-vocabulary field of one record."""
    if "kind" in record:  # an owner reply, wherever its JSON is carried
        yield "reply_kind", record["kind"]
        if record["kind"] == "failure":
            yield "failure_sentence", record["message"]
    if "outcome" in record:  # a recovery acknowledgement
        yield "recovery_outcome", record["outcome"]
    if record.get("object") == "hermes.response.owner_authority":
        yield "authority_action", record["action"]
    if record.get("object") == "response":  # ordinary, stored, or inside an event
        yield "response_status", record["status"]
    if "event" in record:  # a streamed event, whose data repeats its header's type
        yield "stream_event", record["event"]
        yield "stream_event", record["data"]["type"]
        assert record["data"]["type"] == record["event"], record["event"]


def _vocabularies(*answers) -> dict:
    """The values each closed vocabulary takes anywhere in ``answers``, each checked
    to belong to it, in every record, nested record and list item."""
    seen: dict = {name: set() for name in VOCABULARIES}

    def visit(value):
        if isinstance(value, str) and value[:1] in ("{", "["):
            try:
                value = json.loads(value)
            except ValueError:
                return
        if isinstance(value, dict):
            for name, item in _fields(value):
                assert item in VOCABULARIES[name], f"{name}: {item!r}"
                seen[name].add(item)
            value = list(value.values())
        if isinstance(value, list):
            for item in value:
                visit(item)

    for answer in answers:
        visit(answer)
    return seen


async def _answer(response) -> dict:
    return {"status": response.status, "body": await response.json()}


async def _terminal(client, response_id: str) -> dict:
    """The stored response once its background turn has ended."""
    for _ in range(400):
        answer = await _answer(await client.get(f"/v1/responses/{response_id}"))
        if answer["body"].get("status") not in {"queued", "in_progress"}:
            return answer
        await asyncio.sleep(0.02)
    raise AssertionError(f"{response_id} never ended: {answer}")


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
    assert _vocabularies(live) == _vocabularies(saved)

    for answers in (live, saved):
        turns = answers["history"]["body"]["data"] + [
            answer["body"]["data"][-1] for answer in answers["history_failure_turns"]
        ]
        replies = [json.loads(turn["raphael"]) for turn in turns]
        assert {reply["kind"] for reply in replies} == REPLY_KINDS
        assert {
            reply["message"] for reply in replies if reply["kind"] == "failure"
        } == FAILURE_SENTENCES
        assert answers["history_empty"]["body"]["data"] == []
        assert answers["history_pending"]["body"]["pending"] is not None
        assert answers["history_recovery"]["body"]["recovery"] is not None
        assert {
            answer["body"]["outcome"] for answer in answers["recovery_acknowledgements"]
        } == RECOVERY_OUTCOMES
        assert answers["proposal_consumption_refused"]["status"] == 409
        assert answers["owner_history_unavailable"]["body"]["error"]["code"] == (
            "owner_history_unavailable")


@pytest.mark.asyncio
async def test_the_authority_examples_carry_every_answer(owner_payload_example):
    adapter = _make_adapter()
    store = adapter._response_store
    conversation, response_id = _name("a"), "resp_authority"
    claim_id, run_id = "claim_" + "a" * 32, "run_" + "b" * 32
    _seed(store, conversation, response_id,
          [json.dumps(_owner_existing_proposal())], owner_proposal=True)
    # Another proposal's claim, attached to a run this process still sees, as
    # the reconcile route test seeds it, is the tuple to release and reconcile.
    other = (_name("b"), "resp_orphaned")
    exact = {"claim_id": "claim_" + "c" * 32, "run_id": "run_" + "d" * 32}
    _seed(store, *other, [json.dumps(_owner_new_proposal())], owner_proposal=True)
    assert store.claim_owner_proposal("default", *other, exact["claim_id"]) is True
    assert store.attach_owner_run("default", *other, *exact.values()) is True
    adapter._set_run_status(exact["run_id"], "running")

    async with TestClient(TestServer(_create_app(adapter))) as client:
        async def authority(action, at=(conversation, response_id), **extra):
            return await _answer(await client.post(
                f"/v1/responses/conversations/{at[0]}/authority",
                json={"action": action, "response_id": at[1], **extra},
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
        # While its run is observable, the other claim is neither released nor
        # orphaned; once the run is gone, reconcile releases it and release
        # confirms that.
        refused = [await authority(action, other, **exact) for action in (
            "release", "reconcile")]
        adapter._run_statuses.pop(exact["run_id"])
        live += refused + [await authority(action, other, **exact) for action in (
            "reconcile", "release")]

    saved = owner_payload_example(FAMILY, "authority", live)
    assert _vocabularies(live) == _vocabularies(saved)
    assert {answer["status"] for answer in refused} == {409}
    for answers in (live, saved):
        assert {
            answer["body"]["action"] for answer in answers if answer["status"] == 200
        } == AUTHORITY_ACTIONS
        assert {answer["status"] for answer in answers} == {200, 409}


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
            live["response_stored"] = await _terminal(
                client, live["response_background_queued"]["body"]["id"])
            streamed = await respond(_turn(_name("6"), stream=True), "contract-6")
            live["streamed_events"] = _events(await streamed.text())
        with patch.object(adapter, "_run_agent", side_effect=_stub_agent(fail=True)):
            failed = await respond(_turn(_name("8"), stream=True), "contract-7")
            live["streamed_failure_events"] = _events(await failed.text())

        # A background turn whose task is cancelled ends ``incomplete``; the
        # agent parks as in test_api_server's cancelled background turn test.
        parked: dict = {"ready": asyncio.Event()}

        async def _park(**_kwargs):
            parked["task"] = asyncio.current_task()
            parked["ready"].set()
            await asyncio.Event().wait()

        with patch.object(adapter, "_run_agent", side_effect=_park):
            cancelled = await respond(_turn(_name("9"), background=True), "contract-8")
            await asyncio.wait_for(parked["ready"].wait(), timeout=10)
            parked["task"].cancel()
            live["response_stored_incomplete"] = await _terminal(
                client, (await cancelled.json())["id"])

    # Independent requests get distinct ids, which the frozen ids must keep.
    ids = [live[kind]["body"]["id"] for kind in (
        "response", "response_background_queued", "response_stored_incomplete")]
    ids += [event["data"]["response"]["id"] for kind in (
        "streamed_events", "streamed_failure_events") for event in live[kind]
        if event["event"] == "response.created"]
    assert len(set(ids)) == len(ids)
    saved = {kind: owner_payload_example(FAMILY, kind, body) for kind, body in live.items()}
    assert _vocabularies(live) == _vocabularies(saved)

    for answers in (live, saved):
        assert {
            answer["body"]["object"] for kind, answer in answers.items()
            if kind.startswith("response")
        } == {"response"}
        assert answers["response_background_queued"]["body"]["status"] == "queued"
        assert answers["response_stored"]["body"]["status"] == "completed"
        assert answers["response_stored_incomplete"]["body"]["status"] == "incomplete"
        assert {
            event["event"] for kind in ("streamed_events", "streamed_failure_events")
            for event in answers[kind]
        } == STREAM_EVENTS
        assert {
            answers[kind]["body"]["error"]["code"]: answers[kind]["status"] for kind in (
                "idempotency_conflict", "owner_conversation_stale",
                "owner_conversation_locked", "owner_response_incomplete",
            )
        } == {
            "idempotency_conflict": 409, "owner_conversation_stale": 409,
            "owner_conversation_locked": 409, "owner_response_incomplete": 409,
        }


def test_the_conversation_examples_keep_to_every_closed_vocabulary():
    """Each saved conversation example, read whole, uses only and all of each
    vocabulary; the tests above relate each one to the live answers it was saved from."""
    saved = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted((OWNER_PAYLOADS / FAMILY).glob("*.json"))
    ]
    assert _vocabularies(*saved) == VOCABULARIES
