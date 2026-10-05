"""Owner-conversation turns run at high effort (tier plan P4, test P4-1).

The owner's conversation runs on the default profile, whose base route runs at
max. The owner decided that conversation turns run at high instead. The switch
is the per-turn effort the API server hands to ``_create_agent``; every other
request keeps the profile's own effort, or the effort it asked for.

Only ``_create_agent`` is stubbed, so the real ``/v1/responses`` handler and
the real ``_run_agent`` carry the turn's options exactly as production does.
"""

import json
import uuid
from unittest.mock import MagicMock

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import PlatformConfig
from gateway.platforms.api_server import (
    APIServerAdapter,
    _request_reasoning_config,
    cors_middleware,
    security_headers_middleware,
)
from hermes_state import SessionDB

API_KEY = "test-api-key"
OWNER_CONVERSATION = "raphael-owner-" + "5" * 32
HIGH = {"enabled": True, "effort": "high"}


@pytest.fixture
def live(tmp_path):
    """A real adapter behind the real /v1/responses route."""
    adapter = APIServerAdapter(
        PlatformConfig(enabled=True, extra={"host": "127.0.0.1", "port": 0, "key": API_KEY})
    )
    db = SessionDB(tmp_path / "state.db")
    adapter._session_db = db
    mws = [mw for mw in (cors_middleware, security_headers_middleware) if mw is not None]
    app = web.Application(middlewares=mws)
    app["api_server_adapter"] = adapter
    app.router.add_post("/v1/responses", adapter._handle_responses)
    try:
        yield adapter, app
    finally:
        db.close()


def _capture_create_agent(adapter, created):
    """Record every agent the turn builds; answer with one owner question."""

    def _create(**kw):
        created.append(kw)
        agent = MagicMock()
        agent.session_id = kw["session_id"]
        agent.session_prompt_tokens = 0
        agent.session_completion_tokens = 0
        agent.session_total_tokens = 0
        agent.run_conversation.return_value = {
            "final_response": json.dumps(
                {"schema_version": 1, "kind": "question", "message": "Who is it for?"}
            ),
            "messages": [],
            "api_calls": 1,
        }
        return agent

    adapter._create_agent = _create


async def _post(app, body, *, owner):
    headers = {"Authorization": f"Bearer {API_KEY}"}
    if owner:
        # Both owner-conversation contracts: a stated predecessor and an
        # Idempotency-Key.
        headers["Idempotency-Key"] = f"response-{uuid.uuid4().hex}"
    async with TestClient(TestServer(app)) as cli:
        resp = await cli.post("/v1/responses", json=body, headers=headers)
        return resp.status, await resp.text()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "model_options",
    [
        None,
        {"reasoning_effort": "max"},
        {"reasoning": {"enabled": True, "effort": "xhigh"}, "reasoning_effort": "low"},
    ],
    ids=["profile-effort", "asks-max", "asks-other"],
)
async def test_owner_conversation_turns_run_at_high(live, model_options):
    adapter, app = live
    created = []
    _capture_create_agent(adapter, created)
    body = {
        "model": "hermes-agent",
        "input": "Prepare a private 60-minute workshop",
        "conversation": OWNER_CONVERSATION,
        "store": True,
        "expected_previous_response_id": None,
    }
    if model_options is not None:
        body["model_options"] = model_options

    status, text = await _post(app, body, owner=True)

    assert status == 200, text
    assert len(created) == 1
    # The effort the agent is built with: high, whatever the caller asked.
    assert _request_reasoning_config(created[0]["model_options"]) == HIGH
    assert created[0]["model_options"]["reasoning_effort"] == "high"


@pytest.mark.asyncio
@pytest.mark.parametrize("conversation", [None, "my-chat"], ids=["no-conversation", "other-conversation"])
async def test_a_non_owner_request_keeps_the_profile_effort(live, conversation):
    adapter, app = live
    created = []
    _capture_create_agent(adapter, created)
    body = {"model": "hermes-agent", "input": "hi"}
    if conversation is not None:
        body["conversation"] = conversation

    status, text = await _post(app, body, owner=False)

    assert status == 200, text
    assert len(created) == 1
    # No per-turn effort, so _create_agent falls back to the profile's own.
    assert created[0]["model_options"] is None
    assert _request_reasoning_config(created[0]["model_options"]) is None


@pytest.mark.asyncio
async def test_a_non_owner_request_keeps_the_effort_it_asked_for(live):
    adapter, app = live
    created = []
    _capture_create_agent(adapter, created)
    body = {"model": "hermes-agent", "input": "hi", "model_options": {"reasoning_effort": "low"}}

    status, text = await _post(app, body, owner=False)

    assert status == 200, text
    assert created[0]["model_options"] == {"reasoning_effort": "low"}
    assert _request_reasoning_config(created[0]["model_options"]) == {
        "enabled": True, "effort": "low",
    }
