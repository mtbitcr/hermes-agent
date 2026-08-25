"""Tests for /v1/runs endpoints: start, status, events, steer, and stop.

Covers:
- POST /v1/runs — start a run (202)
- GET /v1/runs/{run_id} — poll run status
- GET /v1/runs/{run_id}/events — SSE event stream
- POST /v1/runs/{run_id}/steer — inject guidance into a running agent
- POST /v1/runs/{run_id}/stop — interrupt a running agent
- Auth, error handling, and cleanup
"""

import asyncio
import hashlib
import json
import threading
import time
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import PlatformConfig
from gateway.platforms.api_server import (
    APIServerAdapter,
    ResponseStore,
    _approval_event_choices,
    cors_middleware,
    security_headers_middleware,
)
from tools import approval as approval_mod


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("smart_denied", "allow_permanent", "expected"),
    [
        (False, True, ["once", "session", "always", "deny"]),
        (False, False, ["once", "session", "deny"]),
        (True, True, ["once", "deny"]),
        (True, False, ["once", "deny"]),
    ],
)
def test_approval_event_choices_follow_backend_capabilities(
    smart_denied, allow_permanent, expected
):
    assert _approval_event_choices(
        smart_denied=smart_denied,
        allow_permanent=allow_permanent,
    ) == expected


def _make_adapter(api_key: str = "") -> APIServerAdapter:
    """Create an adapter with optional API key."""
    extra = {}
    if api_key:
        extra["key"] = api_key
    config = PlatformConfig(enabled=True, extra=extra)
    adapter = APIServerAdapter(config)
    return adapter


def _create_runs_app(adapter: APIServerAdapter) -> web.Application:
    """Create an aiohttp app with /v1/runs routes registered."""
    mws = [mw for mw in (cors_middleware, security_headers_middleware) if mw is not None]
    app = web.Application(middlewares=mws)
    app["api_server_adapter"] = adapter
    app.router.add_post("/v1/runs", adapter._handle_runs)
    app.router.add_get("/v1/runs/{run_id}", adapter._handle_get_run)
    app.router.add_get("/v1/runs/{run_id}/events", adapter._handle_run_events)
    app.router.add_post("/v1/runs/{run_id}/approval", adapter._handle_run_approval)
    app.router.add_post("/v1/runs/{run_id}/steer", adapter._handle_steer_run)
    app.router.add_post("/v1/runs/{run_id}/stop", adapter._handle_stop_run)
    return app


def _new_owner_proposal():
    return {
        "schema_version": 2,
        "kind": "proposal",
        "mode": "new",
        "project_name": "Workshop pilot",
        "project_description": "A private workshop pilot.",
        "request_title": "Prepare the workshop",
        "summary": "Prepare the first private milestone.",
        "project_size": "small",
        "specification": "Create one owner-visible workshop milestone.",
        "current_milestone": "Prepare the workshop",
        "owner_visible_result": "A reviewed workshop plan.",
        "impact": ["Adds one private milestone."],
        "later_milestones": [],
        "tasks": [{
            "title": "Draft the workshop plan",
            "body": "Prepare the private workshop plan.",
            "assignee": "default",
            "responsibility": "B03",
            "parents": [],
        }],
    }


def _new_owner_payload(idempotency_key: str, proposal: dict):
    return {
        "idempotency_key": idempotency_key,
        "mode": "new",
        "project_name": proposal["project_name"],
        "project_description": proposal["project_description"],
        "project_id": None,
        "request_title": proposal["request_title"],
        "specification": proposal["specification"],
        "current_milestone": proposal["current_milestone"],
        "owner_visible_result": proposal["owner_visible_result"],
        "root_assignee": "default",
        "tasks": proposal["tasks"],
        "later_milestones": proposal["later_milestones"],
    }


def _make_slow_agent(**kwargs):
    """Create a mock agent that blocks in run_conversation until interrupted.

    Returns (mock_agent, agent_ready_event, interrupt_event) where
    agent_ready_event is set once run_conversation starts, and
    interrupt_event is set when interrupt() is called.
    """
    ready = threading.Event()
    interrupted = threading.Event()

    mock_agent = MagicMock()

    def _do_interrupt(message=None):
        interrupted.set()

    mock_agent.interrupt = MagicMock(side_effect=_do_interrupt)

    def _slow_run(user_message=None, conversation_history=None, task_id=None):
        ready.set()
        # Block until interrupt() is called
        interrupted.wait(timeout=10)
        return {"final_response": "interrupted"}

    mock_agent.run_conversation.side_effect = _slow_run
    mock_agent.session_prompt_tokens = 0
    mock_agent.session_completion_tokens = 0
    mock_agent.session_total_tokens = 0

    return mock_agent, ready, interrupted


@pytest.fixture
def adapter():
    return _make_adapter()


@pytest.fixture
def auth_adapter():
    return _make_adapter(api_key="sk-secret")


# ---------------------------------------------------------------------------
# POST /v1/runs — start a run
# ---------------------------------------------------------------------------


class TestStartRun:
    @pytest.mark.asyncio
    async def test_start_returns_202(self, adapter):
        app = _create_runs_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_create_agent") as mock_create:
                mock_agent = MagicMock()
                mock_agent.run_conversation.return_value = {"final_response": "done"}
                mock_agent.session_prompt_tokens = 10
                mock_agent.session_completion_tokens = 5
                mock_agent.session_total_tokens = 15
                mock_create.return_value = mock_agent

                resp = await cli.post("/v1/runs", json={"input": "hello"})
                assert resp.status == 202
                data = await resp.json()
                assert data["status"] == "started"
                assert data["run_id"].startswith("run_")

                status_resp = await cli.get(f"/v1/runs/{data['run_id']}")
                assert status_resp.status == 200
                status = await status_resp.json()
                assert status["run_id"] == data["run_id"]
                assert status["status"] in {"queued", "running", "completed"}
                assert status["object"] == "hermes.run"

    @pytest.mark.asyncio
    async def test_start_idempotency_reuses_exact_run_and_rejects_conflict(self, adapter):
        app = _create_runs_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_create_agent") as mock_create:
                mock_agent = MagicMock()
                mock_agent.run_conversation.return_value = {"final_response": "done"}
                mock_agent.session_prompt_tokens = 0
                mock_agent.session_completion_tokens = 0
                mock_agent.session_total_tokens = 0
                mock_create.return_value = mock_agent
                headers = {"Idempotency-Key": "owner-run-123"}

                first = await cli.post(
                    "/v1/runs", json={"input": "hello"}, headers=headers,
                )
                second = await cli.post(
                    "/v1/runs", json={"input": "hello"}, headers=headers,
                )
                conflict = await cli.post(
                    "/v1/runs", json={"input": "different"}, headers=headers,
                )

                assert first.status == second.status == 202
                assert (await first.json())["run_id"] == (await second.json())["run_id"]
                assert conflict.status == 409
                for _ in range(40):
                    if mock_create.call_count:
                        break
                    await asyncio.sleep(0.05)
                assert mock_create.call_count == 1

    @pytest.mark.asyncio
    async def test_owner_proposal_run_atomically_claims_commits_and_retries_same_run(
        self, adapter,
    ):
        conversation = "raphael-owner-" + "a" * 32
        response_id = "resp_owner_proposal_run"
        claim_id = "claim_" + "b" * 32
        idempotency_key = "conversation-" + hashlib.sha256(
            response_id.encode("utf-8")
        ).hexdigest()
        proposal = _new_owner_proposal()
        adapter._response_store.put(response_id, {
            "response": {"id": response_id, "created_at": 1},
            "conversation_history": [
                {"role": "user", "content": "Prepare a workshop."},
                {"role": "assistant", "content": json.dumps(proposal)},
            ],
        }, profile="raphael-planner")
        assert adapter._response_store.set_conversation(
            conversation, response_id, owner_proposal=True,
            profile="raphael-planner",
        ) is True
        payload = _new_owner_payload(idempotency_key, proposal)
        body = {
            "input": "Create the approved native task graph.",
            "owner_workspace_context": {
                "mode": "new",
                "project_slug": None,
                "project_name": "Workshop pilot",
            },
            "owner_proposal_authority": {
                "proposal_profile": "raphael-planner",
                "conversation": conversation,
                "response_id": response_id,
                "claim_id": claim_id,
                "operation": "owner_task_graph_commit",
                "idempotency_key": idempotency_key,
                "payload": payload,
            },
        }
        config = {"gateway": {"api_server": {"owner_workspace": {"enabled": True}}}}
        app = _create_runs_app(adapter)
        with (
            patch("gateway.run._load_gateway_config", return_value=config),
            patch.object(adapter, "_create_agent") as mock_create,
            patch(
                "hermes_cli.profiles.list_profiles",
                return_value=[SimpleNamespace(name="default")],
            ),
            patch(
                "hermes_cli.owner_workspace._confirm",
                return_value={"approved": True, "reason": None},
            ),
        ):
            class NativeToolAgent:
                session_prompt_tokens = 0
                session_completion_tokens = 0
                session_total_tokens = 0
                _hermes_api_runtime = {}

                def __init__(self, tool_complete_callback):
                    self._tool_complete_callback = tool_complete_callback

                def run_conversation(self, **_kwargs):
                    from tools.owner_workspace_tools import _handle_task_graph

                    result = _handle_task_graph(payload)
                    self._tool_complete_callback(
                        "tool-call-1",
                        "owner_task_graph_commit",
                        payload,
                        result,
                    )
                    return {"final_response": "model prose is not the receipt"}

                def interrupt(self, _message=None):
                    return None

            mock_create.side_effect = lambda **kwargs: NativeToolAgent(
                kwargs["tool_complete_callback"]
            )

            async with TestClient(TestServer(app)) as cli:
                mismatched_body = json.loads(json.dumps(body))
                mismatched_body["owner_proposal_authority"]["payload"][
                    "request_title"
                ] = "Different payload"
                mismatched = await cli.post(
                    "/v1/runs",
                    json=mismatched_body,
                    headers={"Idempotency-Key": idempotency_key},
                )
                assert mismatched.status == 409
                assert adapter._response_store.owner_history_snapshot(
                    conversation, profile="raphael-planner",
                )["proposal_claimed"] is False
                first = await cli.post(
                    "/v1/runs",
                    json=body,
                    headers={"Idempotency-Key": idempotency_key},
                )
                first_data = await first.json()
                first_run_id = first_data["run_id"]
                for _ in range(40):
                    status_response = await cli.get(f"/v1/runs/{first_run_id}")
                    status = await status_response.json()
                    if status["status"] == "completed":
                        break
                    await asyncio.sleep(0.05)
                retry = await cli.post(
                    "/v1/runs",
                    json=body,
                    headers={"Idempotency-Key": idempotency_key},
                )
                retry_data = await retry.json()

        assert first.status == retry.status == 202
        assert retry_data["run_id"] == first_run_id
        snapshot = adapter._response_store.owner_history_snapshot(
            conversation, profile="raphael-planner",
        )
        assert snapshot["proposal_consumed"] is True
        assert snapshot["proposal_claimed"] is False
        assert status["owner_mutation_committed"] is True
        assert json.loads(status["output"]) == {
            "ok": True,
            "project_slug": "workshop-pilot",
            "task_count": 1,
        }
        assert mock_create.call_count == 1

        from hermes_cli.owner_workspace import OwnerContext, list_committed_projects

        projects = list_committed_projects(
            OwnerContext(actor="default", profile="default", session="")
        )
        assert [project["slug"] for project in projects] == ["workshop-pilot"]

        db_path = adapter._response_store._db_path
        assert db_path is not None
        adapter._response_store.close()
        restarted_store = ResponseStore(db_path=db_path, default_profile="default")
        try:
            restarted_snapshot = restarted_store.owner_history_snapshot(
                conversation, profile="raphael-planner",
            )
            assert restarted_snapshot["proposal_consumed"] is True
            assert restarted_snapshot["proposal_claimed"] is False
        finally:
            restarted_store.close()

    @pytest.mark.asyncio
    async def test_owner_proposal_run_rejects_mismatched_claim_without_allocating_run(
        self, adapter,
    ):
        conversation = "raphael-owner-" + "c" * 32
        response_id = "resp_owner_claim_mismatch"
        real_claim = "claim_" + "d" * 32
        proposal = _new_owner_proposal()
        adapter._response_store.put(response_id, {
            "response": {"id": response_id, "created_at": 1},
            "conversation_history": [
                {"role": "user", "content": "Add the approved milestone."},
                {"role": "assistant", "content": json.dumps(proposal)},
            ],
        }, profile="raphael-planner")
        assert adapter._response_store.set_conversation(
            conversation, response_id, owner_proposal=True,
            profile="raphael-planner",
        ) is True
        assert adapter._response_store.claim_owner_proposal(
            "raphael-planner", conversation, response_id, real_claim,
        ) is True
        idempotency_key = "owner-proposal-run-2"
        payload = _new_owner_payload(idempotency_key, proposal)
        body = {
            "input": "Apply the approved Project change.",
            "owner_workspace_context": {
                "mode": "new",
                "project_slug": None,
                "project_name": "Workshop pilot",
            },
            "owner_proposal_authority": {
                "proposal_profile": "raphael-planner",
                "conversation": conversation,
                "response_id": response_id,
                "claim_id": "claim_" + "e" * 32,
                "operation": "owner_task_graph_commit",
                "idempotency_key": idempotency_key,
                "payload": payload,
            },
        }
        config = {"gateway": {"api_server": {"owner_workspace": {"enabled": True}}}}
        app = _create_runs_app(adapter)
        with (
            patch("gateway.run._load_gateway_config", return_value=config),
            patch(
                "hermes_cli.profiles.list_profiles",
                return_value=[SimpleNamespace(name="default")],
            ),
        ):
            async with TestClient(TestServer(app)) as cli:
                response = await cli.post(
                    "/v1/runs",
                    json=body,
                    headers={"Idempotency-Key": idempotency_key},
                )

        assert response.status == 409
        assert adapter._run_statuses == {}

    @pytest.mark.asyncio
    async def test_failed_owner_run_retries_same_request_with_a_new_run(self, adapter):
        conversation = "raphael-owner-" + "f" * 32
        response_id = "resp_owner_provider_retry"
        claim_id = "claim_" + "1" * 32
        idempotency_key = "conversation-" + hashlib.sha256(
            response_id.encode("utf-8")
        ).hexdigest()
        proposal = _new_owner_proposal()
        adapter._response_store.put(response_id, {
            "response": {"id": response_id, "created_at": 1},
            "conversation_history": [
                {"role": "user", "content": "Prepare a workshop."},
                {"role": "assistant", "content": json.dumps(proposal)},
            ],
        }, profile="raphael-planner")
        assert adapter._response_store.set_conversation(
            conversation, response_id, owner_proposal=True,
            profile="raphael-planner",
        ) is True
        payload = _new_owner_payload(idempotency_key, proposal)
        body = {
            "input": "Create the approved native task graph.",
            "owner_workspace_context": {
                "mode": "new",
                "project_slug": None,
                "project_name": "Workshop pilot",
            },
            "owner_proposal_authority": {
                "proposal_profile": "raphael-planner",
                "conversation": conversation,
                "response_id": response_id,
                "claim_id": claim_id,
                "operation": "owner_task_graph_commit",
                "idempotency_key": idempotency_key,
                "payload": payload,
            },
        }
        config = {"gateway": {"api_server": {"owner_workspace": {"enabled": True}}}}
        attempts = []

        class RetryAgent:
            session_prompt_tokens = 0
            session_completion_tokens = 0
            session_total_tokens = 0
            _hermes_api_runtime = {}

            def __init__(self, tool_complete_callback, succeeds):
                self._tool_complete_callback = tool_complete_callback
                self._succeeds = succeeds

            def run_conversation(self, **_kwargs):
                if not self._succeeds:
                    return {"failed": True, "error": "provider unavailable"}
                from tools.owner_workspace_tools import _handle_task_graph

                result = _handle_task_graph(payload)
                self._tool_complete_callback(
                    "tool-call-retry", "owner_task_graph_commit", payload, result,
                )
                return {"final_response": "done"}

            def interrupt(self, _message=None):
                return None

        def create_agent(**kwargs):
            succeeds = bool(attempts)
            attempts.append(succeeds)
            return RetryAgent(kwargs["tool_complete_callback"], succeeds)

        app = _create_runs_app(adapter)
        with (
            patch("gateway.run._load_gateway_config", return_value=config),
            patch.object(adapter, "_create_agent", side_effect=create_agent),
            patch(
                "hermes_cli.profiles.list_profiles",
                return_value=[SimpleNamespace(name="default")],
            ),
            patch(
                "hermes_cli.owner_workspace._confirm",
                return_value={"approved": True, "reason": None},
            ),
        ):
            async with TestClient(TestServer(app)) as cli:
                first = await cli.post(
                    "/v1/runs", json=body,
                    headers={"Idempotency-Key": idempotency_key},
                )
                first_run_id = (await first.json())["run_id"]
                for _ in range(40):
                    if adapter._run_statuses.get(first_run_id, {}).get("status") == "failed":
                        break
                    await asyncio.sleep(0.05)
                assert adapter._response_store.owner_claim_is_released(
                    "raphael-planner", conversation, response_id, claim_id, first_run_id,
                ) is True

                retry = await cli.post(
                    "/v1/runs", json=body,
                    headers={"Idempotency-Key": idempotency_key},
                )
                retry_run_id = (await retry.json())["run_id"]
                for _ in range(40):
                    if adapter._run_statuses.get(retry_run_id, {}).get("status") == "completed":
                        break
                    await asyncio.sleep(0.05)

        assert first.status == retry.status == 202
        assert retry_run_id != first_run_id
        assert attempts == [False, True]
        assert adapter._response_store.owner_history_snapshot(
            conversation, profile="raphael-planner",
        )["proposal_consumed"] is True

    @pytest.mark.asyncio
    async def test_start_rejects_invalid_idempotency_key(self, adapter):
        app = _create_runs_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            response = await cli.post(
                "/v1/runs",
                json={"input": "hello"},
                headers={"Idempotency-Key": "bad key"},
            )
        assert response.status == 400
        assert adapter._run_statuses == {}

    @pytest.mark.asyncio
    async def test_run_status_persists_exact_resolved_runtime(self, adapter):
        app = _create_runs_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_create_agent") as mock_create:
                mock_agent = MagicMock()
                mock_agent.run_conversation.return_value = {"final_response": "done"}
                mock_agent.session_prompt_tokens = 1
                mock_agent.session_completion_tokens = 2
                mock_agent.session_total_tokens = 3
                mock_agent._hermes_api_runtime = {
                    "provider": "anthropic",
                    "model": "claude-opus-5",
                    "effort": "max",
                    "engine": "native-hermes",
                    "route_source": "global",
                }
                mock_create.return_value = mock_agent

                resp = await cli.post("/v1/runs", json={"input": "hello"})
                assert resp.status == 202
                run_id = (await resp.json())["run_id"]
                for _ in range(40):
                    status_resp = await cli.get(f"/v1/runs/{run_id}")
                    status = await status_resp.json()
                    if status["status"] == "completed":
                        break
                    await asyncio.sleep(0.05)

        assert status["runtime"] == {
            "provider": "anthropic",
            "model": "claude-opus-5",
            "effort": "max",
            "engine": "native-hermes",
        }
        assert "route_source" not in status["runtime"]

    @pytest.mark.asyncio
    async def test_start_binds_chat_id_for_delegation_wake_target(self, adapter):
        """/v1/runs must bind the raw session id as the api_server chat_id
        (like every other agent-entry route does via _run_agent): the async
        delegation dispatch reads HERMES_SESSION_CHAT_ID to pick its wake
        self-post target, and an empty binding forces background delegations
        on this route back to synchronous execution."""
        app = _create_runs_app(adapter)
        captured = {}

        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_create_agent") as mock_create:
                mock_agent = MagicMock()

                def _capture_run(user_message=None, conversation_history=None, task_id=None):
                    from tools.async_delegation import _current_origin_session_id

                    captured["origin_session_id"] = _current_origin_session_id()
                    return {"final_response": "done"}

                mock_agent.run_conversation.side_effect = _capture_run
                mock_agent.session_prompt_tokens = 0
                mock_agent.session_completion_tokens = 0
                mock_agent.session_total_tokens = 0
                mock_create.return_value = mock_agent

                resp = await cli.post(
                    "/v1/runs",
                    json={"input": "hello", "session_id": "runs-raw-sid"},
                )
                assert resp.status == 202
                data = await resp.json()
                run_id = data["run_id"]

                for _ in range(40):
                    status_resp = await cli.get(f"/v1/runs/{run_id}")
                    status = await status_resp.json()
                    if status["status"] == "completed":
                        break
                    await asyncio.sleep(0.05)

        assert captured.get("origin_session_id") == "runs-raw-sid", (
            "runs route must bind chat_id so delegation dispatch sees a wake target"
        )


    @pytest.mark.asyncio
    async def test_start_rejects_conflicting_route_and_request_provider(self):
        adapter = APIServerAdapter(
            PlatformConfig(
                enabled=True,
                extra={
                    "model_routes": {
                        "alias": {
                            "model": "route/model",
                            "provider": "openrouter",
                        }
                    }
                },
            )
        )
        app = _create_runs_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_create_agent") as mock_create:
                resp = await cli.post(
                    "/v1/runs",
                    json={
                        "input": "hello",
                        "model": "alias",
                        "provider": "minimax",
                    },
                )
                data = await resp.json()

        assert resp.status == 400
        assert "provider" in data["error"]["message"].lower()
        assert adapter._run_streams == {}
        assert adapter._run_statuses == {}
        mock_create.assert_not_called()

    @pytest.mark.asyncio
    async def test_start_passes_request_model_provider_options_to_create_agent(self, adapter):
        app = _create_runs_app(adapter)
        model_options = {"reasoning_effort": "medium", "service_tier": "priority"}
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_create_agent") as mock_create:
                mock_agent = MagicMock()
                mock_agent.run_conversation.return_value = {"final_response": "done"}
                mock_agent.session_prompt_tokens = 0
                mock_agent.session_completion_tokens = 0
                mock_agent.session_total_tokens = 0
                mock_create.return_value = mock_agent

                resp = await cli.post(
                    "/v1/runs",
                    json={
                        "input": "hello",
                        "model": "MiniMax-M3",
                        "provider": "minimax",
                        "model_options": model_options,
                    },
                )
                assert resp.status == 202
                for _ in range(20):
                    if mock_create.call_args is not None:
                        break
                    await asyncio.sleep(0.05)

        kwargs = mock_create.call_args.kwargs
        assert kwargs["requested_model"] == "MiniMax-M3"
        assert kwargs["requested_provider"] == "minimax"
        assert kwargs["model_options"] == model_options


# ---------------------------------------------------------------------------
# GET /v1/runs/{run_id} — poll run status
# ---------------------------------------------------------------------------


class TestRunStatus:

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "operation",
        ["owner_task_graph_commit", "owner_project_plan_commit"],
    )
    async def test_status_exposes_redacted_approval_then_clears_it(self, adapter, operation):
        app = _create_runs_app(adapter)
        approval_ready = threading.Event()
        release = threading.Event()

        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_create_agent") as mock_create:
                mock_agent = MagicMock()
                mock_agent.session_prompt_tokens = 0
                mock_agent.session_completion_tokens = 0
                mock_agent.session_total_tokens = 0

                def _run_conversation(*_args, task_id=None, **_kwargs):
                    with approval_mod._lock:
                        notify = approval_mod._gateway_notify_cbs[task_id]
                    notify({
                        "approval_id": "approval-1",
                        "description": "Create project with token sk-live-secret-value",
                        "exact_operation": True,
                        "operation": operation,
                    })
                    approval_ready.set()
                    release.wait(timeout=5)
                    return {"final_response": "done"}

                mock_agent.run_conversation.side_effect = _run_conversation
                mock_create.return_value = mock_agent

                start = await cli.post("/v1/runs", json={"input": "create it"})
                run_id = (await start.json())["run_id"]
                assert approval_ready.wait(timeout=3)

                waiting_resp = await cli.get(f"/v1/runs/{run_id}")
                waiting = await waiting_resp.json()
                assert waiting_resp.status == 200
                assert waiting["status"] == "waiting_for_approval"
                assert waiting["pending_approval"]["approval_id"] == "approval-1"
                assert waiting["pending_approval"]["choices"] == ["once", "deny"]
                assert waiting["pending_approval"]["operation"] == operation
                assert "sk-live-secret-value" not in waiting["pending_approval"]["description"]

                release.set()
                for _ in range(40):
                    done_resp = await cli.get(f"/v1/runs/{run_id}")
                    done = await done_resp.json()
                    if done["status"] == "completed":
                        break
                    await asyncio.sleep(0.05)

                assert done["status"] == "completed"
                assert "pending_approval" not in done


    @pytest.mark.asyncio
    async def test_status_reflects_explicit_session_id(self, adapter):
        app = _create_runs_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_create_agent") as mock_create:
                mock_agent = MagicMock()
                mock_agent.run_conversation.return_value = {"final_response": "done"}
                mock_agent.session_prompt_tokens = 0
                mock_agent.session_completion_tokens = 0
                mock_agent.session_total_tokens = 0
                mock_create.return_value = mock_agent

                resp = await cli.post(
                    "/v1/runs",
                    json={"input": "hello", "session_id": "space-session"},
                )
                data = await resp.json()
                run_id = data["run_id"]

                for _ in range(20):
                    status_resp = await cli.get(f"/v1/runs/{run_id}")
                    status = await status_resp.json()
                    if status["status"] == "completed":
                        break
                    await asyncio.sleep(0.05)

                mock_agent.run_conversation.assert_called_once()
                assert mock_agent.run_conversation.call_args.kwargs["task_id"] == "space-session"
                assert status["session_id"] == "space-session"


# ---------------------------------------------------------------------------
# GET /v1/runs/{run_id}/events — SSE event stream
# ---------------------------------------------------------------------------


class TestRunEvents:
    @pytest.mark.asyncio
    async def test_events_stream_returns_completed(self, adapter):
        """Events stream should receive run.completed when agent finishes."""
        app = _create_runs_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_create_agent") as mock_create:
                mock_agent = MagicMock()
                mock_agent.run_conversation.return_value = {"final_response": "Hello!"}
                mock_agent.session_prompt_tokens = 10
                mock_agent.session_completion_tokens = 5
                mock_agent.session_total_tokens = 15
                mock_create.return_value = mock_agent

                # Start run
                resp = await cli.post("/v1/runs", json={"input": "hello"})
                assert resp.status == 202
                data = await resp.json()
                run_id = data["run_id"]

                # Subscribe to events
                events_resp = await cli.get(f"/v1/runs/{run_id}/events")
                assert events_resp.status == 200
                body = await events_resp.text()

                # Should contain run.completed
                assert "run.completed" in body
                assert "Hello!" in body


    @pytest.mark.asyncio
    async def test_approval_resolve_all_is_scoped_to_target_run(self, auth_adapter):
        """Same client session_id must not let one run approve another run's queue."""
        app = _create_runs_app(auth_adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(auth_adapter, "_create_agent") as mock_create:
                victim_agent, victim_ready, victim_interrupted = _make_slow_agent()
                attacker_agent, attacker_ready, attacker_interrupted = _make_slow_agent()
                mock_create.side_effect = [victim_agent, attacker_agent]

                victim_resp = await cli.post(
                    "/v1/runs",
                    json={"input": "victim", "session_id": "shared-project"},
                    headers={"Authorization": "Bearer sk-secret"},
                )
                attacker_resp = await cli.post(
                    "/v1/runs",
                    json={"input": "attacker", "session_id": "shared-project"},
                    headers={"Authorization": "Bearer sk-secret"},
                )
                assert victim_resp.status == 202
                assert attacker_resp.status == 202
                victim_run = (await victim_resp.json())["run_id"]
                attacker_run = (await attacker_resp.json())["run_id"]

                victim_ready.wait(timeout=3.0)
                attacker_ready.wait(timeout=3.0)
                assert auth_adapter._run_approval_sessions[victim_run] == victim_run
                assert auth_adapter._run_approval_sessions[attacker_run] == attacker_run
                assert auth_adapter._run_approval_sessions[victim_run] != auth_adapter._run_approval_sessions[attacker_run]

                victim_entry = approval_mod._ApprovalEntry({
                    "command": "bash -c victim-danger",
                    "description": "victim approval",
                    "pattern_keys": ["shell-c"],
                })
                attacker_entry = approval_mod._ApprovalEntry({
                    "command": "bash -c attacker-danger",
                    "description": "attacker approval",
                    "pattern_keys": ["shell-c"],
                })
                with approval_mod._lock:
                    approval_mod._gateway_queues[victim_run] = [victim_entry]
                    approval_mod._gateway_queues[attacker_run] = [attacker_entry]

                approval_resp = await cli.post(
                    f"/v1/runs/{attacker_run}/approval",
                    json={"choice": "always", "resolve_all": True},
                    headers={"Authorization": "Bearer sk-secret"},
                )
                approval_data = await approval_resp.json()

                assert approval_resp.status == 200
                assert approval_data["resolved"] == 1
                assert attacker_entry.result == "always"
                assert attacker_entry.event.is_set()
                assert victim_entry.result is None
                assert not victim_entry.event.is_set()
                with approval_mod._lock:
                    assert approval_mod._gateway_queues[victim_run] == [victim_entry]
                    assert victim_run in approval_mod._gateway_queues
                    assert attacker_run not in approval_mod._gateway_queues

                # Clean up the synthetic pending victim approval and unblock the
                # slow test agents so their background run tasks can finish.
                with approval_mod._lock:
                    approval_mod._gateway_queues.pop(victim_run, None)
                victim_interrupted.set()
                attacker_interrupted.set()


# ---------------------------------------------------------------------------
# POST /v1/runs/{run_id}/steer — steer a running agent
# ---------------------------------------------------------------------------


class TestSteerRun:
    @pytest.mark.asyncio
    async def test_steer_running_agent(self, adapter):
        app = _create_runs_app(adapter)
        agent = MagicMock()
        agent.steer.return_value = True
        queue = asyncio.Queue()
        adapter._active_run_agents["run_123"] = agent
        adapter._run_streams["run_123"] = queue
        adapter._set_run_status("run_123", "running")

        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post("/v1/runs/run_123/steer", json={"input": "tighten the ending"})
            payload = await resp.json()

        assert resp.status == 200
        assert payload == {
            "object": "hermes.run.steer",
            "run_id": "run_123",
            "accepted": True,
        }
        agent.steer.assert_called_once_with("tighten the ending")
        assert adapter._run_statuses["run_123"]["last_event"] == "run.steered"
        event = queue.get_nowait()
        assert event["event"] == "run.steered"
        assert event["run_id"] == "run_123"
        assert event["accepted"] is True

    @pytest.mark.asyncio
    async def test_steer_nonexistent_run_returns_404(self, adapter):
        app = _create_runs_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post("/v1/runs/run_missing/steer", json={"input": "hello"})
            payload = await resp.json()

        assert resp.status == 404
        assert payload["error"]["code"] == "run_not_found"

    @pytest.mark.asyncio
    async def test_steer_inactive_run_returns_409(self, adapter):
        app = _create_runs_app(adapter)
        adapter._set_run_status("run_done", "completed")

        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post("/v1/runs/run_done/steer", json={"input": "hello"})
            payload = await resp.json()

        assert resp.status == 409
        assert payload["error"]["code"] == "run_not_accepting_steer"

    @pytest.mark.asyncio
    async def test_steer_missing_input_returns_400(self, adapter):
        app = _create_runs_app(adapter)
        agent = MagicMock()
        agent.steer.return_value = True
        adapter._active_run_agents["run_123"] = agent
        adapter._set_run_status("run_123", "running")

        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post("/v1/runs/run_123/steer", json={"input": ""})
            payload = await resp.json()

        assert resp.status == 400
        assert payload["error"]["code"] == "invalid_steer_input"
        agent.steer.assert_not_called()

    @pytest.mark.asyncio
    async def test_stop_then_steer_rejects_retained_agent_ref(self, adapter):
        """Steer must reject a stopping run even if the executor thread is still live."""
        app = _create_runs_app(adapter)
        run_can_finish = threading.Event()
        run_started = threading.Event()

        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_create_agent") as mock_create:
                mock_agent = MagicMock()
                mock_agent.session_prompt_tokens = 0
                mock_agent.session_completion_tokens = 0
                mock_agent.session_total_tokens = 0
                mock_agent.steer = MagicMock(return_value=True)

                def _interrupt(_message=None):
                    return None

                def _run_conversation(*_args, **_kwargs):
                    run_started.set()
                    run_can_finish.wait(timeout=5)
                    return {"final_response": "late result"}

                mock_agent.interrupt = MagicMock(side_effect=_interrupt)
                mock_agent.run_conversation.side_effect = _run_conversation
                mock_create.return_value = mock_agent

                start_resp = await cli.post("/v1/runs", json={"input": "hello"})
                run_id = (await start_resp.json())["run_id"]
                assert run_started.wait(timeout=3.0)

                stop_resp = await cli.post(f"/v1/runs/{run_id}/stop")
                assert stop_resp.status == 200
                assert run_id in adapter._active_run_agents

                steer_resp = await cli.post(
                    f"/v1/runs/{run_id}/steer",
                    json={"input": "tighten the ending"},
                )
                steer_data = await steer_resp.json()

                assert steer_resp.status == 409
                assert steer_data["error"]["code"] == "run_not_accepting_steer"
                mock_agent.steer.assert_not_called()

                run_can_finish.set()
                for _ in range(40):
                    if run_id not in adapter._active_run_tasks:
                        break
                    await asyncio.sleep(0.05)

    @pytest.mark.asyncio
    async def test_pending_steer_preserved_on_run_completed(self, adapter):
        """A steer drained by the turn finalizer (accepted after the final
        response) must surface as pending_steer on the terminal run status
        instead of being silently dropped."""
        app = _create_runs_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_create_agent") as mock_create:
                mock_agent = MagicMock()
                mock_agent.session_prompt_tokens = 0
                mock_agent.session_completion_tokens = 0
                mock_agent.session_total_tokens = 0
                mock_agent.run_conversation.return_value = {
                    "final_response": "done",
                    "pending_steer": "tighten the ending",
                }
                mock_create.return_value = mock_agent

                start_resp = await cli.post("/v1/runs", json={"input": "hello"})
                run_id = (await start_resp.json())["run_id"]

                for _ in range(40):
                    status = adapter._run_statuses.get(run_id, {})
                    if status.get("status") == "completed":
                        break
                    await asyncio.sleep(0.05)

        assert adapter._run_statuses[run_id]["status"] == "completed"
        assert adapter._run_statuses[run_id]["pending_steer"] == "tighten the ending"

    @pytest.mark.asyncio
    async def test_steer_requires_auth(self, auth_adapter):
        app = _create_runs_app(auth_adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post("/v1/runs/run_any/steer", json={"input": "hello"})

        assert resp.status == 401


# ---------------------------------------------------------------------------
# Run lifecycle TTL sweeping
# ---------------------------------------------------------------------------


class TestRunLifecycleSweep:

    @pytest.mark.asyncio
    async def test_expired_live_run_drops_transport_but_keeps_control_state(self, adapter):
        """Stream TTL bounds buffering without detaching a live run."""
        app = _create_runs_app(adapter)
        adapter._max_concurrent_runs = 1

        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_create_agent") as mock_create:
                mock_agent, agent_ready, _ = _make_slow_agent()
                mock_create.return_value = mock_agent

                start_resp = await cli.post("/v1/runs", json={"input": "hello"})
                assert start_resp.status == 202
                run_id = (await start_resp.json())["run_id"]
                assert agent_ready.wait(timeout=3.0)

                task = adapter._active_run_tasks[run_id]
                assert isinstance(task, asyncio.Task)
                assert not task.done()

                pending = approval_mod._ApprovalEntry({
                    "command": "bash -c long-running",
                    "description": "approval after stream TTL",
                    "pattern_keys": ["shell-c"],
                })
                with approval_mod._lock:
                    approval_mod._gateway_queues[run_id] = [pending]

                adapter._run_streams_created[run_id] -= adapter._RUN_STREAM_TTL + 1
                # Exercise one real sweeper iteration without waiting 60 seconds.
                with patch(
                    "gateway.platforms.api_server.asyncio.sleep",
                    side_effect=[None, asyncio.CancelledError()],
                ):
                    with pytest.raises(asyncio.CancelledError):
                        await adapter._sweep_orphaned_runs()

                assert adapter._active_run_tasks[run_id] is task
                assert adapter._active_run_agents[run_id] is mock_agent
                assert run_id not in adapter._run_streams
                assert run_id not in adapter._run_streams_created
                assert adapter._run_approval_sessions[run_id] == run_id

                limited = adapter._concurrency_limited_response()
                assert limited is not None
                assert limited.status == 429

                approval_resp = await cli.post(
                    f"/v1/runs/{run_id}/approval",
                    json={"choice": "once"},
                )
                assert approval_resp.status == 200
                assert pending.event.is_set()
                assert pending.result == "once"

                stop_resp = await cli.post(f"/v1/runs/{run_id}/stop")
                assert stop_resp.status == 200
                mock_agent.interrupt.assert_called_once_with("Stop requested via API")


# ---------------------------------------------------------------------------
# POST /v1/runs/{run_id}/stop — interrupt a running agent
# ---------------------------------------------------------------------------


class TestStopRun:

    @pytest.mark.asyncio
    async def test_stop_keeps_uncooperative_executor_tracked_until_exit(self, adapter):
        """Cancelling an asyncio wrapper must not hide its live executor thread."""
        app = _create_runs_app(adapter)
        run_can_finish = threading.Event()
        run_finished = threading.Event()

        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_create_agent") as mock_create:
                mock_agent = MagicMock()
                mock_agent.session_prompt_tokens = 0
                mock_agent.session_completion_tokens = 0
                mock_agent.session_total_tokens = 0
                started = threading.Event()

                def _run_conversation(*_args, **_kwargs):
                    started.set()
                    run_can_finish.wait(timeout=5)
                    run_finished.set()
                    return {"final_response": "late result"}

                mock_agent.run_conversation.side_effect = _run_conversation
                mock_create.return_value = mock_agent

                resp = await cli.post("/v1/runs", json={"input": "hello"})
                run_id = (await resp.json())["run_id"]
                assert started.wait(timeout=3)

                stop_resp = await cli.post(f"/v1/runs/{run_id}/stop")
                assert stop_resp.status == 200
                await asyncio.sleep(0.1)

                assert not run_finished.is_set()
                assert run_id in adapter._active_run_agents
                assert run_id in adapter._active_run_tasks
                assert adapter._run_statuses[run_id]["status"] == "stopping"

                run_can_finish.set()
                for _ in range(40):
                    if run_id not in adapter._active_run_tasks:
                        break
                    await asyncio.sleep(0.05)

                assert run_id not in adapter._active_run_agents
                assert run_id not in adapter._active_run_tasks
                assert adapter._run_statuses[run_id]["status"] == "cancelled"

    @pytest.mark.asyncio
    async def test_stop_running_agent(self, adapter):
        """Stop should interrupt the agent and cancel the task."""
        app = _create_runs_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_create_agent") as mock_create:
                mock_agent, agent_ready, _ = _make_slow_agent()
                mock_create.return_value = mock_agent

                # Start run
                resp = await cli.post("/v1/runs", json={"input": "hello"})
                assert resp.status == 202
                data = await resp.json()
                run_id = data["run_id"]

                # Wait for agent to start running in the thread
                agent_ready.wait(timeout=3.0)
                await asyncio.sleep(0.1)

                # Verify agent ref is stored
                assert run_id in adapter._active_run_agents

                # Stop the run
                stop_resp = await cli.post(f"/v1/runs/{run_id}/stop")
                assert stop_resp.status == 200
                stop_data = await stop_resp.json()
                assert stop_data["run_id"] == run_id
                assert stop_data["status"] == "stopping"

                # Agent interrupt should have been called
                mock_agent.interrupt.assert_called_once_with("Stop requested via API")

                status_resp = await cli.get(f"/v1/runs/{run_id}")
                assert status_resp.status == 200
                status_data = await status_resp.json()
                assert status_data["status"] in {"stopping", "cancelled"}

                # Refs should be cleaned up
                await asyncio.sleep(0.2)
                assert run_id not in adapter._active_run_agents
                assert run_id not in adapter._active_run_tasks


    @pytest.mark.asyncio
    async def test_stop_sends_sentinel_to_events_stream(self, adapter):
        """After stop, the events stream should close."""
        app = _create_runs_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_create_agent") as mock_create:
                mock_agent, agent_ready, _ = _make_slow_agent()
                mock_create.return_value = mock_agent

                # Start run
                resp = await cli.post("/v1/runs", json={"input": "hello"})
                assert resp.status == 202
                data = await resp.json()
                run_id = data["run_id"]

                agent_ready.wait(timeout=3.0)
                await asyncio.sleep(0.1)

                # Subscribe to events in background
                events_task = asyncio.ensure_future(
                    cli.get(f"/v1/runs/{run_id}/events")
                )

                await asyncio.sleep(0.1)

                # Stop the run
                stop_resp = await cli.post(f"/v1/runs/{run_id}/stop")
                assert stop_resp.status == 200

                # Events stream should close
                events_resp = await asyncio.wait_for(events_task, timeout=5.0)
                assert events_resp.status == 200
                body = await events_resp.text()
                # Stream should have received run.failed and closed
                assert "run.failed" in body or "stream closed" in body


class TestRunsProviderAuthFailure:
    @pytest.mark.asyncio
    async def test_status_reports_provider_auth_failure_distinctly(self, adapter):
        """/v1/runs builds its own agent via _create_agent() and does not
        route through _run_agent(), so the controlled "Provider
        authentication failed" message added there does not cover this
        endpoint. _handle_runs()'s own _ProviderAuthResolutionError branch
        must give the same distinguished message instead of the generic
        except-Exception "run failed" text."""
        from gateway.platforms.api_server import _ProviderAuthResolutionError

        app = _create_runs_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_create_agent") as mock_create:
                mock_create.side_effect = _ProviderAuthResolutionError(
                    "No credentials found for provider 'nous'"
                )

                resp = await cli.post("/v1/runs", json={"input": "hello"})
                assert resp.status == 202
                data = await resp.json()
                run_id = data["run_id"]

                for _ in range(40):
                    status_resp = await cli.get(f"/v1/runs/{run_id}")
                    status = await status_resp.json()
                    if status["status"] == "failed":
                        break
                    await asyncio.sleep(0.05)

                assert status["status"] == "failed"
                assert status["error"] == "⚠️ Provider authentication failed: No credentials found for provider 'nous'"
                assert status["last_event"] == "run.failed"
