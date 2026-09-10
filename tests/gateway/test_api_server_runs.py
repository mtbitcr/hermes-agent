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
import contextlib
import hashlib
import json
import os
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
    _make_request_fingerprint,
    _OWNER_RUN_STOPPED_MESSAGE,
    _owner_mutation_refusal,
    _owner_mutation_refusal_code,
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
        # v3 is the first new-project schema that carries execution_tier, so it
        # is the first one that grants commit authority.
        "schema_version": 3,
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
            "execution_tier": "routine",
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


# ---------------------------------------------------------------------------
# owner_task_retry — the hidden run authority that tries stopped work again
# ---------------------------------------------------------------------------

_RETRY_REASON = "I have added the provider credentials to the account."


def _committed_owner_project(session: str, key: str, proposal: dict) -> dict:
    """Create one real receipt-backed Project through the owner kernel."""
    from hermes_cli import owner_workspace as ow

    setup_payload = _new_owner_payload(key, proposal)
    setup_context = ow.OwnerContext(
        actor="default",
        profile="default",
        session=session,
        authority=ow.OwnerProposalAuthority(
            actor="default",
            profile="default",
            session=session,
            conversation="raphael-owner-" + "7" * 32,
            response_id="resp_" + "6" * 32,
            operation="owner_task_graph_commit",
            idempotency_key=key,
            payload_digest=ow._digest(setup_payload),
        ),
    )
    with (
        patch(
            "hermes_cli.profiles.list_profiles",
            return_value=[SimpleNamespace(name="default")],
        ),
        patch(
            "hermes_cli.owner_workspace._confirm",
            return_value={"approved": True, "reason": None},
        ),
    ):
        return ow.commit_task_graph(setup_context, **setup_payload)


def _capability_stopped_task(board: str, project_id: str, title: str) -> str:
    """Drive real kernel transitions until a worker hits a hard wall.

    The same technique tests/hermes_cli/test_owner_workspace.py uses to make a
    task genuinely retryable: claim it, then block it on a capability wall, so
    the eligibility ``kanban_db.stopped_work_retry_evidence`` reads is the
    kernel's own record of HOW the work stopped rather than a state this test
    asserted into place.
    """
    from hermes_cli import kanban_db

    with contextlib.closing(kanban_db.connect(board=board)) as conn:
        task_id = kanban_db.create_task(
            conn, title=title, assignee="default", project_id=project_id,
        )
        assert kanban_db.claim_task(conn, task_id) is not None
        assert kanban_db.block_task(
            conn, task_id,
            reason="the provider account has no API credentials",
            kind="capability",
            expected_run_id=kanban_db.get_task(conn, task_id).current_run_id,
        )
        task = kanban_db.get_task(conn, task_id)
        assert (task.status, task.block_kind) == ("blocked", "capability")
    return task_id


def _task_status(board: str, task_id: str) -> str:
    from hermes_cli import kanban_db

    with contextlib.closing(kanban_db.connect(board=board)) as conn:
        return kanban_db.get_task(conn, task_id).status


def _owner_retry_events(board: str, task_id: str) -> list:
    """Every durable ``owner_retry`` event on one task, oldest first."""
    from hermes_cli import kanban_db

    with contextlib.closing(kanban_db.connect(board=board)) as conn:
        return [
            json.loads(row["payload"]) if row["payload"] else None
            for row in conn.execute(
                "SELECT payload FROM task_events WHERE task_id = ? "
                "AND kind = 'owner_retry' ORDER BY id ASC",
                (task_id,),
            )
        ]


def _owner_retry_run_setup(idempotency_key: str, *, reason: str = _RETRY_REASON):
    """A real Project, genuinely stopped work, and the run body that retries it."""
    proposal = _new_owner_proposal()
    created = _committed_owner_project(
        f"setup-{idempotency_key}", f"setup-{idempotency_key}", proposal,
    )
    task_id = _capability_stopped_task(
        created["board"],
        created["project_id"],
        "B03 — Connect the payment provider",
    )
    payload = {
        "idempotency_key": idempotency_key,
        "project_id": created["project_id"],
        "task_id": task_id,
        "reason": reason,
    }
    body = {
        "input": "Try the stopped work again now.",
        "owner_workspace_context": {
            "mode": "existing",
            "project_slug": created["project_slug"],
            "project_name": proposal["project_name"],
        },
        "owner_retry_authority": {
            "operation": "owner_task_retry",
            "idempotency_key": idempotency_key,
            "payload": payload,
        },
    }
    return created, task_id, payload, body


def _native_retry_handler():
    """The native ``owner_task_retry`` handler the run layer resolves by name.

    ``tools/owner_workspace_tools.py`` is outside this change's scope and does
    not carry ``_handle_task_retry`` yet, so these tests install the same thin
    kernel delegation its sibling owner handlers already are, and the run layer
    resolves it exactly as it will resolve the shipped one. Everything beneath
    it is production code: the confirmation, the board transition and the
    receipt are all ``hermes_cli.owner_workspace.retry_task``'s.
    """
    from hermes_cli import owner_workspace as ow
    from tools.registry import tool_error

    def _handle_task_retry(args, **kw):
        try:
            return json.dumps(
                ow.retry_task(
                    ow.resolve_owner_context(),
                    idempotency_key=args.get("idempotency_key"),
                    project_id=args.get("project_id"),
                    task_id=args.get("task_id"),
                    reason=args.get("reason"),
                ),
                ensure_ascii=False,
            )
        except ow.OwnerWorkspaceError as exc:
            return tool_error(f"owner_task_retry: {exc.message}", code=exc.code)

    return MagicMock(side_effect=_handle_task_retry)


def _retry_payload_digest(payload: dict) -> str:
    return hashlib.sha256(
        json.dumps(
            payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        ).encode("utf-8")
    ).hexdigest()


@pytest.fixture(autouse=True)
def _resolved_owner_task_routes():
    """Resolve owner task routes without a real profile config on disk.

    These tests exercise the run lifecycle, not the model matrix (which is
    covered by tests/plugins/dashboard_auth/test_raphael_model_policy.py); the
    owner kernel only needs *some* admitted route to pin.
    """
    from hermes_cli import owner_workspace as ow

    from plugins.dashboard_auth.raphael_workspace import model_policy

    original = model_policy.configured_assignment_for
    # Only the on-disk provider selection is faked; the admitted matrix, the
    # tier resolution and the durable lock minting are all production code.
    model_policy.configured_assignment_for = (
        lambda profile: model_policy.assignment_for(profile, "anthropic")
    )
    try:
        yield
    finally:
        model_policy.configured_assignment_for = original


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
    async def test_reservation_race_replays_winning_run_and_durable_queue(
        self, adapter,
    ):
        body = {"input": "hello"}
        idempotency_key = "reservation-race"
        session_scope = hashlib.sha256(b"").hexdigest()
        fingerprint_body = {**body, "_session_scope": session_scope}
        fingerprint = _make_request_fingerprint(
            fingerprint_body, keys=sorted(fingerprint_body),
        )
        winning_run_id = "run_" + "a" * 32
        assert adapter._response_store.reserve_run_idempotency(
            "default",
            session_scope,
            idempotency_key,
            fingerprint,
            winning_run_id,
        ) == ("new", winning_run_id)

        app = _create_runs_app(adapter)
        with (
            patch.object(
                adapter._response_store,
                "lookup_run_idempotency",
                return_value=("missing", None),
            ),
            patch.object(adapter, "_create_agent") as mock_create,
        ):
            async with TestClient(TestServer(app)) as cli:
                response = await cli.post(
                    "/v1/runs",
                    json=body,
                    headers={"Idempotency-Key": idempotency_key},
                )
                response_body = await response.json()
                status_response = await cli.get(
                    f"/v1/runs/{winning_run_id}",
                )
                status_body = await status_response.json()

        assert response.status == 202
        assert response_body == {
            "run_id": winning_run_id,
            "status": "started",
        }
        assert status_response.status == 200
        assert status_body["status"] == "queued"
        mock_create.assert_not_called()

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
        proposal["specification"] = "Keep the literal hermes-agent repository name unchanged."
        proposal["tasks"][0]["body"] = "Inspect hermes-agent without rewriting its name."
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
            mock_create.side_effect = RuntimeError("provider unavailable")

            async with TestClient(TestServer(app)) as cli:
                mismatched_context_body = json.loads(json.dumps(body))
                mismatched_context_body["owner_workspace_context"][
                    "project_name"
                ] = "Different workshop"
                mismatched_context = await cli.post(
                    "/v1/runs",
                    json=mismatched_context_body,
                    headers={"Idempotency-Key": idempotency_key},
                )
                assert mismatched_context.status == 409
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
                for _ in range(200):
                    status_response = await cli.get(f"/v1/runs/{first_run_id}")
                    status = await status_response.json()
                    if status["status"] == "completed":
                        break
                    await asyncio.sleep(0.05)
                assert status["status"] == "completed", status
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
        assert snapshot["completed_run_id"] == first_run_id
        assert status["owner_mutation_committed"] is True
        assert json.loads(status["output"]) == {
            "ok": True,
            "project_slug": "workshop-pilot",
            "task_count": 1,
        }
        mock_create.assert_not_called()

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
            assert restarted_snapshot["completed_run_id"] == first_run_id
            adapter._response_store = restarted_store
            adapter._run_statuses.clear()
            restarted_app = _create_runs_app(adapter)
            with (
                patch("gateway.run._load_gateway_config", return_value=config),
                patch.object(adapter, "_create_agent") as restarted_create,
            ):
                async with TestClient(TestServer(restarted_app)) as cli:
                    restored = await cli.get(f"/v1/runs/{first_run_id}")
                    restored_status = await restored.json()
                    replay = await cli.post(
                        "/v1/runs",
                        json=body,
                        headers={"Idempotency-Key": idempotency_key},
                    )
                    replay_data = await replay.json()
                    next_response_id = "resp_owner_follow_up"
                    restarted_store.put(next_response_id, {
                        "response": {"id": next_response_id, "created_at": 2},
                        "conversation_history": [{
                            "role": "assistant",
                            "content": json.dumps({
                                "schema_version": 1,
                                "kind": "question",
                            }),
                        }],
                    }, profile="raphael-planner")
                    assert restarted_store.set_conversation(
                        conversation,
                        next_response_id,
                        profile="raphael-planner",
                    ) is True
                    replay_after_follow_up = await cli.post(
                        "/v1/runs",
                        json=body,
                        headers={"Idempotency-Key": idempotency_key},
                    )
                    replay_after_follow_up_data = await replay_after_follow_up.json()
            assert restored.status == 200
            assert restored_status["run_id"] == first_run_id
            assert restored_status["status"] == "completed"
            assert restored_status["owner_mutation_committed"] is True
            assert restored_status["output"] == status["output"]
            assert restored_status["created_at"] == status["created_at"]
            assert replay.status == 202, replay_data
            assert replay_data["run_id"] == first_run_id
            assert replay_after_follow_up.status == 202, replay_after_follow_up_data
            assert replay_after_follow_up_data["run_id"] == first_run_id
            restarted_create.assert_not_called()
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
    async def test_owner_lifecycle_run_is_exact_native_and_restart_safe(self, adapter):
        from hermes_cli import owner_workspace as ow

        setup_key = "setup-lifecycle-project"
        proposal = _new_owner_proposal()
        setup_payload = _new_owner_payload(setup_key, proposal)
        setup_context = ow.OwnerContext(
            actor="default",
            profile="default",
            session="setup-owner-lifecycle",
            authority=ow.OwnerProposalAuthority(
                actor="default",
                profile="default",
                session="setup-owner-lifecycle",
                conversation="raphael-owner-" + "9" * 32,
                response_id="resp_" + "8" * 32,
                operation="owner_task_graph_commit",
                idempotency_key=setup_key,
                payload_digest=ow._digest(setup_payload),
            ),
        )
        with (
            patch(
                "hermes_cli.profiles.list_profiles",
                return_value=[SimpleNamespace(name="default")],
            ),
            patch(
                "hermes_cli.owner_workspace._confirm",
                return_value={"approved": True, "reason": None},
            ),
        ):
            created = ow.commit_task_graph(setup_context, **setup_payload)

        idempotency_key = "project-lifecycle-exact-run"
        lifecycle_payload = {
            "idempotency_key": idempotency_key,
            "project_id": created["project_id"],
            "expected_revision": 0,
            "action": "archive",
        }
        body = {
            "input": "Archive the confirmed Project now.",
            "owner_workspace_context": {
                "mode": "existing",
                "project_slug": created["project_slug"],
                "project_name": proposal["project_name"],
            },
            "owner_lifecycle_authority": {
                "operation": "owner_project_lifecycle",
                "idempotency_key": idempotency_key,
                "payload": lifecycle_payload,
            },
        }
        config = {"gateway": {"api_server": {"owner_workspace": {"enabled": True}}}}

        app = _create_runs_app(adapter)
        with (
            patch("gateway.run._load_gateway_config", return_value=config),
            patch.object(adapter, "_create_agent") as mock_create,
            patch(
                "hermes_cli.owner_workspace._confirm",
                return_value={"approved": True, "reason": None},
            ),
        ):
            mock_create.side_effect = RuntimeError("provider unavailable")
            async with TestClient(TestServer(app)) as cli:
                stale_body = json.loads(json.dumps(body))
                stale_body["owner_lifecycle_authority"]["payload"][
                    "expected_revision"
                ] = 99
                stale = await cli.post(
                    "/v1/runs",
                    json=stale_body,
                    headers={"Idempotency-Key": idempotency_key},
                )
                assert stale.status == 409
                assert mock_create.call_count == 0

                first = await cli.post(
                    "/v1/runs",
                    json=body,
                    headers={"Idempotency-Key": idempotency_key},
                )
                first_run_id = (await first.json())["run_id"]
                for _ in range(40):
                    polled = await cli.get(f"/v1/runs/{first_run_id}")
                    status = await polled.json()
                    if status["status"] == "completed":
                        break
                    await asyncio.sleep(0.05)

        assert first.status == 202
        assert status["owner_mutation_committed"] is True
        assert json.loads(status["output"]) == {
            "ok": True,
            "action": "archive",
            "project_slug": created["project_slug"],
            "archived": True,
            "execution_paused": True,
        }
        assert "model prose" not in status["output"]
        mock_create.assert_not_called()

        db_path = adapter._response_store._db_path
        assert db_path is not None
        adapter._response_store._conn.execute(
            "UPDATE run_idempotency SET terminal_json = NULL "
            "WHERE idempotency_key = ?",
            (idempotency_key,),
        )
        adapter._response_store._conn.commit()
        adapter._response_store.close()
        restarted_store = ResponseStore(db_path=db_path, default_profile="default")
        try:
            adapter._response_store = restarted_store
            adapter._run_statuses[first_run_id] = {
                "run_id": first_run_id,
                "status": "failed",
                "updated_at": time.time(),
                "error": "stale transport state",
            }
            restarted_app = _create_runs_app(adapter)
            with (
                patch("gateway.run._load_gateway_config", return_value=config),
                patch.object(adapter, "_create_agent") as restarted_create,
            ):
                async with TestClient(TestServer(restarted_app)) as cli:
                    missing_before_recovery = await cli.get(
                        f"/v1/runs/{first_run_id}"
                    )
                    missing_before_recovery_status = (
                        await missing_before_recovery.json()
                    )
                    replay = await cli.post(
                        "/v1/runs",
                        json=body,
                        headers={"Idempotency-Key": idempotency_key},
                    )
                    replay_data = await replay.json()
                    restored = await cli.get(f"/v1/runs/{first_run_id}")
                    restored_status = await restored.json()
            assert missing_before_recovery.status == 200
            assert missing_before_recovery_status["status"] == "failed"
            assert restored.status == 200
            assert restored_status["status"] == "completed"
            assert restored_status["owner_mutation_committed"] is True
            assert restored_status["output"] == status["output"]
            assert restored_status["created_at"] == status["created_at"]
            assert replay.status == 202, replay_data
            assert replay_data["run_id"] == first_run_id
            restarted_create.assert_not_called()
        finally:
            restarted_store.close()

    @pytest.mark.asyncio
    async def test_owner_retry_run_is_exact_native_and_transport_retry_safe(
        self, adapter,
    ):
        """One retry authority, one native call, and no model in the middle.

        The accepted shape end to end against the real kernel: a committed
        Project, work the kernel itself recorded as stopped, and a run that
        applies the authority's exact payload once. The transport retry half
        is asserted against durable state — the board's own ``owner_retry``
        events — not merely against the second response body.
        """
        from tools import owner_workspace_tools

        idempotency_key = "owner-task-retry-exact-run"
        created, task_id, retry_payload, body = _owner_retry_run_setup(
            idempotency_key,
        )
        config = {"gateway": {"api_server": {"owner_workspace": {"enabled": True}}}}
        handler = _native_retry_handler()

        app = _create_runs_app(adapter)
        with (
            patch("gateway.run._load_gateway_config", return_value=config),
            patch.object(adapter, "_create_agent") as mock_create,
            patch.object(
                owner_workspace_tools, "_handle_task_retry", handler, create=True,
            ),
            patch(
                "hermes_cli.owner_workspace._confirm",
                return_value={"approved": True, "reason": None},
            ),
        ):
            mock_create.side_effect = RuntimeError("provider unavailable")
            async with TestClient(TestServer(app)) as cli:
                first = await cli.post(
                    "/v1/runs",
                    json=body,
                    headers={"Idempotency-Key": idempotency_key},
                )
                first_run_id = (await first.json())["run_id"]
                for _ in range(200):
                    polled = await cli.get(f"/v1/runs/{first_run_id}")
                    status = await polled.json()
                    if status["status"] == "completed":
                        break
                    await asyncio.sleep(0.05)
                replay = await cli.post(
                    "/v1/runs",
                    json=body,
                    headers={"Idempotency-Key": idempotency_key},
                )
                replay_data = await replay.json()

        assert first.status == 202
        assert status["status"] == "completed", status
        assert status["owner_mutation_committed"] is True
        # The tool's own result, read back to the caller.
        output = json.loads(status["output"])
        assert set(output) == {
            "ok", "task_id", "status", "revision", "retry_reason",
        }
        assert output["ok"] is True
        assert output["task_id"] == task_id
        assert output["status"] == "ready"
        assert output["retry_reason"] == _RETRY_REASON
        assert isinstance(output["revision"], int)
        assert not isinstance(output["revision"], bool)

        # Exactly one native call, carrying exactly the authority's payload.
        assert handler.call_count == 1
        assert handler.call_args.args[0] == retry_payload
        # No agent was ever created, so no model produced that payload.
        mock_create.assert_not_called()

        # The work really moved, once, on the owner's stated reason.
        assert _task_status(created["board"], task_id) == "ready"
        assert [
            event["reason"]
            for event in _owner_retry_events(created["board"], task_id)
        ] == [_RETRY_REASON]

        assert replay.status == 202, replay_data
        assert replay_data["run_id"] == first_run_id
        assert handler.call_count == 1

    @pytest.mark.asyncio
    async def test_owner_workspace_toolset_retry_runs_under_the_real_approval(
        self, adapter,
    ):
        """The retry the owner toolset offers IS the one the run layer dispatches.

        Everything about the retry here is production's: the tool name comes
        out of the ``owner_workspace`` toolset rather than being typed as a
        literal, the handler is the registered one (nothing is patched onto
        ``owner_workspace_tools``), and the confirmation is the real hidden
        owner approval — raised by the kernel, surfaced on the run as an
        ``owner_task_retry`` decision, and answered through the run's own
        ``/approval`` endpoint. No private adapter or approval-queue state is
        seeded. The effect is then read off the board, not off a return value.
        """
        from tools import owner_workspace_tools  # noqa: F401 — registers the tools
        from tools.registry import registry as tool_registry

        from toolsets import resolve_toolset

        # A tool an agent can never be granted is not a capability. Every tool
        # the registry carries for ``owner_workspace`` must also be OFFERED by
        # that toolset's own static definition — that definition is what
        # ``_get_platform_tools`` resolves an api_server profile against, so a
        # tool missing from it never reaches an owner-workspace agent no
        # matter what registered it.
        registered = set(
            tool_registry.get_tool_names_for_toolset("owner_workspace")
        )
        offered = set(resolve_toolset("owner_workspace", include_registry=False))
        assert "owner_task_retry" in registered
        assert registered <= offered, sorted(registered - offered)
        retry_tool = "owner_task_retry"

        idempotency_key = "owner-task-retry-toolset-approved"
        created, task_id, retry_payload, body = _owner_retry_run_setup(
            idempotency_key,
        )
        # The run authority names exactly the toolset's own tool.
        assert body["owner_retry_authority"]["operation"] == retry_tool
        config = {"gateway": {"api_server": {"owner_workspace": {"enabled": True}}}}

        app = _create_runs_app(adapter)
        with (
            patch("gateway.run._load_gateway_config", return_value=config),
            patch.object(adapter, "_create_agent") as mock_create,
        ):
            mock_create.side_effect = RuntimeError("provider unavailable")
            async with TestClient(TestServer(app)) as cli:
                started = await cli.post(
                    "/v1/runs",
                    json=body,
                    headers={"Idempotency-Key": idempotency_key},
                )
                run_id = (await started.json())["run_id"]

                # The kernel's own confirmation reaches the owner as a run
                # decision naming this exact operation.
                pending = None
                for _ in range(400):
                    polled = await cli.get(f"/v1/runs/{run_id}")
                    status = await polled.json()
                    if status["status"] == "waiting_for_approval":
                        pending = status["pending_approval"]
                        break
                    await asyncio.sleep(0.05)
                assert pending is not None, status
                assert pending["operation"] == retry_tool

                answered = await cli.post(
                    f"/v1/runs/{run_id}/approval",
                    json={
                        "choice": "once",
                        "approval_id": pending["approval_id"],
                    },
                )
                assert answered.status == 200, await answered.json()

                for _ in range(400):
                    polled = await cli.get(f"/v1/runs/{run_id}")
                    status = await polled.json()
                    if status["status"] == "completed":
                        break
                    await asyncio.sleep(0.05)

        assert started.status == 202
        assert status["status"] == "completed", status
        assert status["owner_mutation_committed"] is True
        output = json.loads(status["output"])
        assert output["ok"] is True, output
        assert output["task_id"] == task_id
        assert output["retry_reason"] == retry_payload["reason"]
        # No model was ever asked to produce that call.
        mock_create.assert_not_called()

        # The work really left the state the worker stopped it in, and the
        # owner's reason is durable on the board.
        assert _task_status(created["board"], task_id) == "ready"
        assert [
            event["reason"]
            for event in _owner_retry_events(created["board"], task_id)
        ] == [_RETRY_REASON]

    @pytest.mark.asyncio
    async def test_owner_retry_transport_retry_recovers_a_committed_run(
        self, adapter,
    ):
        """A crash between the board commit and the run terminal is recovered.

        The committed retry has already moved the work out of the state the
        authority was bound to, so the exact transport retry may neither
        re-validate as fresh authority nor retry the work again: it resolves
        the SAME run from the native receipt.
        """
        from tools import owner_workspace_tools

        idempotency_key = "owner-task-retry-recovered-run"
        created, task_id, _payload, body = _owner_retry_run_setup(idempotency_key)
        config = {"gateway": {"api_server": {"owner_workspace": {"enabled": True}}}}
        handler = _native_retry_handler()

        app = _create_runs_app(adapter)
        with (
            patch("gateway.run._load_gateway_config", return_value=config),
            patch.object(adapter, "_create_agent") as mock_create,
            patch.object(
                owner_workspace_tools, "_handle_task_retry", handler, create=True,
            ),
            patch(
                "hermes_cli.owner_workspace._confirm",
                return_value={"approved": True, "reason": None},
            ),
        ):
            mock_create.side_effect = RuntimeError("provider unavailable")
            async with TestClient(TestServer(app)) as cli:
                first = await cli.post(
                    "/v1/runs",
                    json=body,
                    headers={"Idempotency-Key": idempotency_key},
                )
                first_run_id = (await first.json())["run_id"]
                for _ in range(200):
                    polled = await cli.get(f"/v1/runs/{first_run_id}")
                    status = await polled.json()
                    if status["status"] == "completed":
                        break
                    await asyncio.sleep(0.05)

        assert first.status == 202
        assert status["status"] == "completed", status
        assert handler.call_count == 1

        db_path = adapter._response_store._db_path
        assert db_path is not None
        # The gateway died after the board committed and before its own run
        # terminal was stored.
        adapter._response_store._conn.execute(
            "UPDATE run_idempotency SET terminal_json = NULL "
            "WHERE idempotency_key = ?",
            (idempotency_key,),
        )
        adapter._response_store._conn.commit()
        adapter._response_store.close()
        restarted_store = ResponseStore(db_path=db_path, default_profile="default")
        try:
            adapter._response_store = restarted_store
            adapter._run_statuses[first_run_id] = {
                "run_id": first_run_id,
                "status": "failed",
                "updated_at": time.time(),
                "error": "stale transport state",
            }
            restarted_app = _create_runs_app(adapter)
            with (
                patch("gateway.run._load_gateway_config", return_value=config),
                patch.object(adapter, "_create_agent") as restarted_create,
                patch.object(
                    owner_workspace_tools,
                    "_handle_task_retry",
                    handler,
                    create=True,
                ),
            ):
                async with TestClient(TestServer(restarted_app)) as cli:
                    replay = await cli.post(
                        "/v1/runs",
                        json=body,
                        headers={"Idempotency-Key": idempotency_key},
                    )
                    replay_data = await replay.json()
                    restored = await cli.get(f"/v1/runs/{first_run_id}")
                    restored_status = await restored.json()
            assert replay.status == 202, replay_data
            assert replay_data["run_id"] == first_run_id
            assert restored.status == 200
            assert restored_status["status"] == "completed"
            assert restored_status["owner_mutation_committed"] is True
            assert restored_status["output"] == status["output"]
            assert restored_status["created_at"] == status["created_at"]
            restarted_create.assert_not_called()
            # Durable proof the work was not retried a second time.
            assert handler.call_count == 1
            assert [
                event["reason"]
                for event in _owner_retry_events(created["board"], task_id)
            ] == [_RETRY_REASON]
        finally:
            restarted_store.close()

    @pytest.mark.asyncio
    async def test_owner_retry_recovery_finds_a_padded_reason_receipt(
        self, adapter,
    ):
        """One canonical payload binds the receipt AND finds it again.

        A reason the owner typed with surrounding whitespace is accepted, so
        the same crash between the board commit and the run terminal has to
        resolve the SAME run from the native receipt. It only can when the
        payload the run bound its digest to and the payload recovery hashes
        are the same one: binding the raw text while recovery rebuilt the
        trimmed text left the committed retry findable under neither digest.
        """
        from tools import owner_workspace_tools

        idempotency_key = "owner-task-retry-padded-reason"
        created, task_id, _payload, body = _owner_retry_run_setup(
            idempotency_key, reason=f"  {_RETRY_REASON}\n",
        )
        config = {"gateway": {"api_server": {"owner_workspace": {"enabled": True}}}}
        handler = _native_retry_handler()

        app = _create_runs_app(adapter)
        with (
            patch("gateway.run._load_gateway_config", return_value=config),
            patch.object(adapter, "_create_agent") as mock_create,
            patch.object(
                owner_workspace_tools, "_handle_task_retry", handler, create=True,
            ),
            patch(
                "hermes_cli.owner_workspace._confirm",
                return_value={"approved": True, "reason": None},
            ),
        ):
            mock_create.side_effect = RuntimeError("provider unavailable")
            async with TestClient(TestServer(app)) as cli:
                first = await cli.post(
                    "/v1/runs",
                    json=body,
                    headers={"Idempotency-Key": idempotency_key},
                )
                first_run_id = (await first.json())["run_id"]
                for _ in range(200):
                    polled = await cli.get(f"/v1/runs/{first_run_id}")
                    status = await polled.json()
                    if status["status"] == "completed":
                        break
                    await asyncio.sleep(0.05)

        assert first.status == 202
        assert status["status"] == "completed", status
        assert handler.call_count == 1

        db_path = adapter._response_store._db_path
        assert db_path is not None
        # The gateway died after the board committed and before its own run
        # terminal was stored.
        adapter._response_store._conn.execute(
            "UPDATE run_idempotency SET terminal_json = NULL "
            "WHERE idempotency_key = ?",
            (idempotency_key,),
        )
        adapter._response_store._conn.commit()
        adapter._response_store.close()
        restarted_store = ResponseStore(db_path=db_path, default_profile="default")
        try:
            adapter._response_store = restarted_store
            adapter._run_statuses[first_run_id] = {
                "run_id": first_run_id,
                "status": "failed",
                "updated_at": time.time(),
                "error": "stale transport state",
            }
            restarted_app = _create_runs_app(adapter)
            with (
                patch("gateway.run._load_gateway_config", return_value=config),
                patch.object(adapter, "_create_agent") as restarted_create,
                patch.object(
                    owner_workspace_tools,
                    "_handle_task_retry",
                    handler,
                    create=True,
                ),
            ):
                async with TestClient(TestServer(restarted_app)) as cli:
                    replay = await cli.post(
                        "/v1/runs",
                        json=body,
                        headers={"Idempotency-Key": idempotency_key},
                    )
                    replay_data = await replay.json()
                    restored = await cli.get(f"/v1/runs/{first_run_id}")
                    restored_status = await restored.json()
            assert replay.status == 202, replay_data
            assert replay_data["run_id"] == first_run_id
            assert restored.status == 200
            assert restored_status["status"] == "completed"
            assert restored_status["owner_mutation_committed"] is True
            assert restored_status["output"] == status["output"]
            restarted_create.assert_not_called()
            # Durable proof the work was not retried a second time.
            assert handler.call_count == 1
            assert [
                event["reason"]
                for event in _owner_retry_events(created["board"], task_id)
            ] == [_RETRY_REASON]
        finally:
            restarted_store.close()

    @pytest.mark.asyncio
    async def test_owner_retry_re_delivery_replays_a_refused_run(self, adapter):
        """A durably refused retry re-delivers as itself, not as an unknown.

        The kernel records its own deterministic refusal as a COMMITTED
        receipt whose result is not ok, which the successful-receipt reader
        cannot read at all. A run that already reached the durable terminal
        state we recorded needs no receipt to decide it: answering "outcome
        unconfirmed" for the exact re-delivery left the owner with a refusal
        the transport would no longer report.
        """
        from hermes_cli import kanban_db
        from tools import owner_workspace_tools

        idempotency_key = "owner-task-retry-refused-run"
        created, _stopped_task_id, _payload, body = _owner_retry_run_setup(
            idempotency_key,
        )
        # Work nobody stopped: the kernel refuses to retry it, deterministically
        # and with a committed receipt, which is the state this replay is about.
        with contextlib.closing(kanban_db.connect(board=created["board"])) as conn:
            task_id = kanban_db.create_task(
                conn, title="B04 — Write the runbook", assignee="default",
                project_id=created["project_id"],
            )
        body["owner_retry_authority"]["payload"]["task_id"] = task_id
        config = {"gateway": {"api_server": {"owner_workspace": {"enabled": True}}}}
        handler = _native_retry_handler()

        app = _create_runs_app(adapter)
        with (
            patch("gateway.run._load_gateway_config", return_value=config),
            patch.object(adapter, "_create_agent") as mock_create,
            patch.object(
                owner_workspace_tools, "_handle_task_retry", handler, create=True,
            ),
            patch(
                "hermes_cli.owner_workspace._confirm",
                return_value={"approved": True, "reason": None},
            ),
        ):
            mock_create.side_effect = RuntimeError("provider unavailable")
            async with TestClient(TestServer(app)) as cli:
                first = await cli.post(
                    "/v1/runs",
                    json=body,
                    headers={"Idempotency-Key": idempotency_key},
                )
                first_run_id = (await first.json())["run_id"]
                for _ in range(200):
                    polled = await cli.get(f"/v1/runs/{first_run_id}")
                    status = await polled.json()
                    durable = adapter._response_store.run_idempotency_status(
                        "default", first_run_id,
                    ) or {}
                    if durable.get("status") == "failed":
                        break
                    await asyncio.sleep(0.05)
                re_delivery = await cli.post(
                    "/v1/runs",
                    json=body,
                    headers={"Idempotency-Key": idempotency_key},
                )
                re_delivery_data = await re_delivery.json()
                replayed = await cli.get(f"/v1/runs/{first_run_id}")
                replayed_status = await replayed.json()

        assert first.status == 202
        assert status["status"] == "failed", status
        assert status["owner_mutation_committed"] is False
        # The refusal really is durable: it is the record the re-delivery reads.
        assert durable["status"] == "failed"

        assert re_delivery.status == 202, re_delivery_data
        assert re_delivery_data["run_id"] == first_run_id
        assert replayed.status == 200
        assert replayed_status["status"] == "failed"
        # Durable proof the refusal was neither re-run nor re-decided.
        assert handler.call_count == 1
        # The work is exactly where nobody stopped it.
        assert _task_status(created["board"], task_id) == "ready"
        assert _owner_retry_events(created["board"], task_id) == []

    @pytest.mark.asyncio
    async def test_owner_retry_payload_digest_binds_the_single_native_call(
        self, adapter,
    ):
        """The run is bound to the sha256 of exactly this payload, nothing else.

        The digest the gateway computed is the one the kernel wrote against the
        receipt its single tool call committed, so the committed receipt is
        readable under that digest and under no other.
        """
        from hermes_cli.owner_workspace import read_committed_owner_run_receipt
        from tools import owner_workspace_tools

        idempotency_key = "owner-task-retry-digest-bound"
        _created, task_id, retry_payload, body = _owner_retry_run_setup(
            idempotency_key,
        )
        config = {"gateway": {"api_server": {"owner_workspace": {"enabled": True}}}}
        handler = _native_retry_handler()

        app = _create_runs_app(adapter)
        with (
            patch("gateway.run._load_gateway_config", return_value=config),
            patch.object(adapter, "_create_agent") as mock_create,
            patch.object(
                owner_workspace_tools, "_handle_task_retry", handler, create=True,
            ),
            patch(
                "hermes_cli.owner_workspace._confirm",
                return_value={"approved": True, "reason": None},
            ),
        ):
            mock_create.side_effect = RuntimeError("provider unavailable")
            async with TestClient(TestServer(app)) as cli:
                started = await cli.post(
                    "/v1/runs",
                    json=body,
                    headers={"Idempotency-Key": idempotency_key},
                )
                run_id = (await started.json())["run_id"]
                for _ in range(200):
                    polled = await cli.get(f"/v1/runs/{run_id}")
                    status = await polled.json()
                    if status["status"] == "completed":
                        break
                    await asyncio.sleep(0.05)

        assert status["status"] == "completed", status
        assert handler.call_count == 1
        assert handler.call_args.args[0] == retry_payload

        committed = read_committed_owner_run_receipt(
            profile="default",
            idempotency_key=idempotency_key,
            operation="owner_task_retry",
            authority_digest=_retry_payload_digest(retry_payload),
        )
        assert committed is not None
        assert committed["task_id"] == task_id
        # Any other payload digest — even one differing only in the reason —
        # names no committed run.
        assert read_committed_owner_run_receipt(
            profile="default",
            idempotency_key=idempotency_key,
            operation="owner_task_retry",
            authority_digest=_retry_payload_digest(
                {**retry_payload, "reason": _RETRY_REASON + "."},
            ),
        ) is None

    @pytest.mark.asyncio
    async def test_owner_retry_refuses_a_stale_or_foreign_project(self, adapter):
        """Slug plus Project id must both name one committed Project."""
        from tools import owner_workspace_tools

        idempotency_key = "owner-task-retry-stale-project"
        created, task_id, _payload, body = _owner_retry_run_setup(idempotency_key)
        other_proposal = _new_owner_proposal()
        other_proposal["project_name"] = "Second workshop"
        other = _committed_owner_project(
            "setup-owner-retry-other", "setup-retry-other", other_proposal,
        )
        assert other["project_id"] != created["project_id"]
        config = {"gateway": {"api_server": {"owner_workspace": {"enabled": True}}}}
        handler = _native_retry_handler()

        app = _create_runs_app(adapter)
        with (
            patch("gateway.run._load_gateway_config", return_value=config),
            patch.object(adapter, "_create_agent") as mock_create,
            patch.object(
                owner_workspace_tools, "_handle_task_retry", handler, create=True,
            ),
            patch(
                "hermes_cli.owner_workspace._confirm",
                return_value={"approved": True, "reason": None},
            ),
        ):
            mock_create.side_effect = RuntimeError("provider unavailable")
            async with TestClient(TestServer(app)) as cli:
                for label, project_id in (
                    ("foreign", other["project_id"]),
                    ("unknown", "p_no_such_project"),
                ):
                    stale_body = json.loads(json.dumps(body))
                    stale_body["owner_retry_authority"]["payload"][
                        "project_id"
                    ] = project_id
                    response = await cli.post(
                        "/v1/runs",
                        json=stale_body,
                        headers={"Idempotency-Key": idempotency_key},
                    )
                    assert response.status == 409, label
                    assert (await response.json())["error"]["code"] == (
                        "owner_retry_authority_conflict"
                    ), label

        assert mock_create.call_count == 0
        assert handler.call_count == 0
        assert adapter._run_statuses == {}
        # The work is exactly where the worker left it.
        assert _task_status(created["board"], task_id) == "blocked"
        assert _owner_retry_events(created["board"], task_id) == []

    @pytest.mark.asyncio
    async def test_owner_retry_authority_cannot_be_combined_or_unscoped(
        self, adapter,
    ):
        """Two owner authorities never share a run, and none runs unscoped."""
        idempotency_key = "owner-task-retry-exclusive"
        retry_authority = {
            "operation": "owner_task_retry",
            "idempotency_key": idempotency_key,
            "payload": {
                "idempotency_key": idempotency_key,
                "project_id": "p_workshop",
                "task_id": "t_workshop",
                "reason": _RETRY_REASON,
            },
        }
        proposal_authority = {
            "proposal_profile": "raphael-planner",
            "conversation": "raphael-owner-" + "a" * 32,
            "response_id": "resp_owner_retry_exclusive",
            "claim_id": "claim_" + "b" * 32,
            "operation": "owner_task_graph_commit",
            "idempotency_key": idempotency_key,
            "payload": {"idempotency_key": idempotency_key},
        }
        lifecycle_authority = {
            "operation": "owner_project_lifecycle",
            "idempotency_key": idempotency_key,
            "payload": {
                "idempotency_key": idempotency_key,
                "project_id": "p_workshop",
                "expected_revision": 0,
                "action": "archive",
            },
        }
        cases = (
            (
                "retry+proposal",
                {"owner_proposal_authority": proposal_authority},
                "Owner authorities cannot be combined",
            ),
            (
                "retry+lifecycle",
                {"owner_lifecycle_authority": lifecycle_authority},
                "Owner authorities cannot be combined",
            ),
            (
                "retry without context",
                {},
                "Owner authority requires owner workspace context",
            ),
        )

        app = _create_runs_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            for label, extra, message in cases:
                response = await cli.post(
                    "/v1/runs",
                    json={
                        "input": "Try the stopped work again now.",
                        "owner_retry_authority": retry_authority,
                        **extra,
                    },
                    headers={"Idempotency-Key": idempotency_key},
                )
                assert response.status == 400, label
                assert (await response.json())["error"]["message"] == message, label

        assert adapter._run_statuses == {}

    @pytest.mark.asyncio
    async def test_owner_retry_authority_of_the_wrong_shape_is_a_400(self, adapter):
        """A shape this transport does not accept never reaches validation."""
        app = _create_runs_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            response = await cli.post(
                "/v1/runs",
                json={
                    "input": "Try the stopped work again now.",
                    "owner_retry_authority": {
                        "operation": "owner_task_retry",
                        "idempotency_key": "shape-400",
                        "payload": {"idempotency_key": "shape-400"},
                    },
                },
            )
            body = await response.json()
        assert response.status == 400
        assert body["error"]["message"] == "Invalid owner-workspace authority"
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

        from hermes_cli import projects_db
        real_connect = projects_db.connect

        def connect_once_recovered(*args, **kwargs):
            succeeds = bool(attempts)
            attempts.append(succeeds)
            if not succeeds:
                raise OSError("native project store temporarily unavailable")
            return real_connect(*args, **kwargs)

        app = _create_runs_app(adapter)
        with (
            patch("gateway.run._load_gateway_config", return_value=config),
            patch.object(projects_db, "connect", side_effect=connect_once_recovered),
            patch.object(adapter, "_create_agent", side_effect=RuntimeError("provider unavailable")),
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
                for _ in range(200):
                    if adapter._run_statuses.get(first_run_id, {}).get("status") == "failed":
                        break
                    await asyncio.sleep(0.05)
                assert adapter._run_statuses.get(first_run_id, {}).get("status") == "failed"
                assert adapter._response_store.owner_claim_is_released(
                    "raphael-planner", conversation, response_id, claim_id, first_run_id,
                ) is True

                retry = await cli.post(
                    "/v1/runs", json=body,
                    headers={"Idempotency-Key": idempotency_key},
                )
                retry_run_id = (await retry.json())["run_id"]
                for _ in range(200):
                    if adapter._run_statuses.get(retry_run_id, {}).get("status") == "completed":
                        break
                    await asyncio.sleep(0.05)
                assert adapter._run_statuses.get(retry_run_id, {}).get("status") == "completed"

        assert first.status == retry.status == 202
        assert retry_run_id != first_run_id
        assert attempts[0] is False
        assert len(attempts) >= 2 and all(attempts[1:])
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
# owner_retry_authority — the closed transport shape and its payload bounds
# ---------------------------------------------------------------------------


# The kernel's own bound, at module scope because the payload-bounds
# parametrization below is built from it at import time.
from hermes_cli.owner_workspace import _OWNER_RETRY_REASON_LIMIT  # noqa: E402

_RETRY_AUTHORITY_SHAPE = {
    "operation": "owner_task_retry",
    "idempotency_key": "owner-retry-shape",
    "payload": {
        "idempotency_key": "owner-retry-shape",
        "project_id": "p_workshop",
        "task_id": "t_workshop",
        "reason": _RETRY_REASON,
    },
}


def _retry_authority_shape(case: str):
    """One authority per shape this transport must refuse."""
    authority = json.loads(json.dumps(_RETRY_AUTHORITY_SHAPE))
    if case == "not_a_dict":
        return [_RETRY_AUTHORITY_SHAPE]
    if case == "missing_key":
        del authority["payload"]
        return authority
    if case == "extra_key":
        authority["owner"] = "default"
        return authority
    if case == "wrong_operation":
        authority["operation"] = "owner_task_move"
        return authority
    if case == "malformed_idempotency_key":
        authority["idempotency_key"] = "owner retry shape"
        authority["payload"]["idempotency_key"] = "owner retry shape"
        return authority
    if case == "payload_not_a_dict":
        authority["payload"] = [authority["payload"]]
        return authority
    if case == "payload_missing_key":
        del authority["payload"]["reason"]
        return authority
    if case == "payload_extra_key":
        authority["payload"]["expected_revision"] = 0
        return authority
    if case == "payload_key_disagrees":
        authority["payload"]["idempotency_key"] = "owner-retry-other"
        return authority
    raise AssertionError(f"unknown case {case!r}")


class TestOwnerRetryRunAuthorityShape:
    # Imported per test rather than at module scope so a tree without the
    # authority fails these tests one by one instead of failing collection.
    @staticmethod
    def _resolve(value):
        from gateway.platforms.api_server import (
            _resolve_owner_retry_run_authority,
        )

        return _resolve_owner_retry_run_authority(value)

    @pytest.mark.parametrize("case", [
        "not_a_dict",
        "missing_key",
        "extra_key",
        "wrong_operation",
        "malformed_idempotency_key",
        "payload_not_a_dict",
        "payload_missing_key",
        "payload_extra_key",
        "payload_key_disagrees",
    ])
    def test_refuses_every_shape_it_is_not(self, case):
        with pytest.raises(ValueError, match="invalid owner retry authority"):
            self._resolve(_retry_authority_shape(case))

    def test_absent_authority_stays_absent(self):
        assert self._resolve(None) is None

    def test_accepts_exactly_the_closed_shape(self):
        assert self._resolve(
            json.loads(json.dumps(_RETRY_AUTHORITY_SHAPE))
        ) == _RETRY_AUTHORITY_SHAPE

    @pytest.mark.parametrize(("field", "value"), [
        ("project_id", ""),
        ("project_id", "   "),
        ("project_id", "p" * 101),
        ("project_id", None),
        ("task_id", ""),
        ("task_id", "   "),
        ("task_id", None),
        ("reason", ""),
        ("reason", "   "),
        ("reason", "r" * (_OWNER_RETRY_REASON_LIMIT + 1)),
        ("reason", None),
    ])
    def test_payload_bounds_are_the_kernels_own(self, adapter, field, value):
        """The gateway refuses exactly the payloads ``retry_task`` bounds out.

        The bound comes from ``hermes_cli.owner_workspace`` rather than a
        number copied into this transport, so the two cannot drift apart.
        """
        authority = json.loads(json.dumps(_RETRY_AUTHORITY_SHAPE))
        authority["payload"][field] = value
        with pytest.raises(ValueError, match="owner retry payload is invalid"):
            adapter._validated_owner_retry_authority(
                authority,
                {
                    "profile": "default",
                    "mode": "existing",
                    "project_slug": "workshop-pilot",
                    "project_name": "Workshop pilot",
                },
                "default",
            )

    @pytest.mark.parametrize("context", [
        {"profile": "other", "mode": "existing", "project_slug": "workshop-pilot"},
        {"profile": "default", "mode": "new", "project_slug": None},
        {"profile": "default", "mode": "existing", "project_slug": None},
    ])
    def test_context_must_bind_this_profile_and_an_existing_project(
        self, adapter, context,
    ):
        with pytest.raises(ValueError, match="owner retry context mismatch"):
            adapter._validated_owner_retry_authority(
                json.loads(json.dumps(_RETRY_AUTHORITY_SHAPE)), context, "default",
            )


_RETRY_SUCCESS_RESULT = {
    "ok": True,
    "task_id": "t_workshop",
    "status": "ready",
    "revision": 12,
    "retry_reason": _RETRY_REASON,
}


class TestOwnerRetryMutationReceipt:
    """The minimal receipt projection for a retry, which names no Project."""

    @staticmethod
    def _admits_a_retry_at_all(adapter) -> None:
        """A genuine retry success projects something.

        The anchor every "projects nothing" case below needs: without it those
        cases also hold for an operation this projection does not know, which
        is exactly the state they exist to distinguish a refusal from.
        """
        assert adapter._owner_mutation_receipt(
            "owner_task_retry", dict(_RETRY_SUCCESS_RESULT),
        ) is not None

    def test_a_genuine_success_projects_only_the_bound_fields(self, adapter):
        receipt = adapter._owner_mutation_receipt("owner_task_retry", {
            **_RETRY_SUCCESS_RESULT,
            "internal_board": "workshop-pilot",
        })
        assert json.loads(receipt) == {
            "ok": True,
            "task_id": "t_workshop",
            "status": "ready",
            "revision": 12,
            "retry_reason": _RETRY_REASON,
        }
        # The tool answers with a JSON string; both forms project identically.
        assert adapter._owner_mutation_receipt(
            "owner_task_retry", json.dumps(_RETRY_SUCCESS_RESULT),
        ) == receipt

    @pytest.mark.parametrize("result", [
        {
            "ok": False,
            "error": "not_retryable",
            "task_id": "t_workshop",
            "current_status": "running",
            "reason": "This work is running and has not stopped.",
        },
        {"ok": True, "status": "ready", "revision": 1, "retry_reason": "r"},
        {"ok": True, "task_id": "", "status": "ready", "revision": 1,
         "retry_reason": "r"},
        {"ok": True, "task_id": "t", "status": "", "revision": 1,
         "retry_reason": "r"},
        {"ok": True, "task_id": "t", "status": "Ready!", "revision": 1,
         "retry_reason": "r"},
        {"ok": True, "task_id": "t", "status": "ready", "revision": True,
         "retry_reason": "r"},
        {"ok": True, "task_id": "t", "status": "ready", "revision": "1",
         "retry_reason": "r"},
        {"ok": True, "task_id": "t", "status": "ready", "revision": 1,
         "retry_reason": ""},
        {"ok": True, "task_id": "t", "status": "ready", "revision": 1},
        "not json at all",
    ])
    def test_a_refusal_or_malformed_result_projects_nothing(self, adapter, result):
        self._admits_a_retry_at_all(adapter)
        assert adapter._owner_mutation_receipt("owner_task_retry", result) is None

    def test_the_project_operations_still_require_their_slug(self, adapter):
        """Admitting a retry must not loosen the branches that name a Project."""
        self._admits_a_retry_at_all(adapter)
        assert adapter._owner_mutation_receipt("owner_project_lifecycle", {
            "ok": True,
            "action": "archive",
            "archived": True,
            "execution_paused": True,
        }) is None
        assert adapter._owner_mutation_receipt("owner_task_graph_commit", {
            "ok": True, "task_count": 1,
        }) is None
        assert json.loads(adapter._owner_mutation_receipt(
            "owner_project_lifecycle",
            {
                "ok": True,
                "action": "archive",
                "project_slug": "workshop-pilot",
                "archived": True,
                "execution_paused": True,
            },
        )) == {
            "ok": True,
            "action": "archive",
            "project_slug": "workshop-pilot",
            "archived": True,
            "execution_paused": True,
        }


# ---------------------------------------------------------------------------
# GET /v1/runs/{run_id} — poll run status
# ---------------------------------------------------------------------------


class TestRunStatus:

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "operation",
        [
            "owner_task_graph_commit",
            "owner_project_plan_commit",
            "owner_task_retry",
        ],
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


# ---------------------------------------------------------------------------
# Item 32TK: a run's only executor is this process, so a restart must
# reconcile it instead of polling "working" forever
# ---------------------------------------------------------------------------


class TestOrphanedRunRecovery:
    _RUN_ID = "run_" + "a" * 32
    _CONVERSATION = "raphael-owner-" + "1" * 32
    _RESPONSE_ID = "resp_" + "9" * 28
    _CLAIM_ID = "claim_" + "9" * 20

    def _queued_owner_run(self, adapter, *, dead: bool = True) -> dict:
        """Exactly what a 202 leaves durable: a queued run and a claimed
        proposal, attributed to a process that is gone."""
        store = adapter._response_store
        scope = hashlib.sha256(b"").hexdigest()
        key = "run-key-1"
        owner = {
            "proposal_profile": "default",
            "conversation": self._CONVERSATION,
            "response_id": self._RESPONSE_ID,
            "claim_id": self._CLAIM_ID,
            "operation": "owner_project_plan_commit",
            "payload_digest": "digest",
        }
        store.put(self._RESPONSE_ID, {
            "response": {"id": self._RESPONSE_ID, "status": "completed"},
            "conversation_history": [
                {"role": "user", "content": "Apply it."},
                {"role": "assistant", "content": json.dumps(
                    {"schema_version": 1, "kind": "question", "message": "ok?"}
                )},
            ],
        })
        assert store.set_conversation(
            self._CONVERSATION, self._RESPONSE_ID, owner_proposal=False,
        ) is True
        store._conn.execute(
            "UPDATE conversations SET proposal_response_id = ?, "
            "claimed_response_id = ?, claim_id = ?, owner_run_id = ?, "
            "claim_state = 'claimed' WHERE profile = 'default' AND name = ?",
            (
                self._RESPONSE_ID, self._RESPONSE_ID, self._CLAIM_ID,
                self._RUN_ID, self._CONVERSATION,
            ),
        )
        store._conn.commit()
        state, _existing = store.reserve_run_idempotency(
            "default", scope, key, "fingerprint", self._RUN_ID,
        )
        assert state == "new"
        store.reserve_owner_job("run", self._RUN_ID, "default", {"owner": owner})
        if dead:
            store._conn.execute(
                "UPDATE owner_executor_jobs SET executor_id = 'dead', "
                "executor_pid = 2147483646, created_at = ? WHERE job_key = ?",
                (time.time() - 3600, self._RUN_ID),
            )
            store._conn.commit()
        return {"scope": scope, "key": key, "owner": owner}

    @pytest.mark.asyncio
    async def test_a_run_whose_executor_died_stops_reporting_working(self, adapter):
        reserved = self._queued_owner_run(adapter)
        store = adapter._response_store

        # Before recovery: the durable state still says queued, and its only
        # executor no longer exists.
        assert store.run_idempotency_status("default", self._RUN_ID)["status"] == (
            "queued"
        )

        adapter._recover_orphaned_owner_jobs()

        recovered = store.run_idempotency_status("default", self._RUN_ID)
        assert recovered["status"] == "failed"
        assert "restarted" in recovered["error"]
        # The claim is released, so the owner can approve the same change again.
        assert store.owner_claim_is_released(
            "default", self._CONVERSATION, self._RESPONSE_ID,
            self._CLAIM_ID, self._RUN_ID,
        ) is True
        assert store.claim_orphaned_owner_jobs("run") == []
        assert reserved["owner"]["conversation"] == self._CONVERSATION

    @pytest.mark.asyncio
    async def test_a_recovered_run_polls_as_failed(self, adapter):
        self._queued_owner_run(adapter)
        adapter._recover_orphaned_owner_jobs()

        app = _create_runs_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.get(f"/v1/runs/{self._RUN_ID}")
            body = await resp.json()

        assert resp.status == 200
        assert body["status"] == "failed"

    @pytest.mark.asyncio
    async def test_a_run_with_a_live_executor_is_left_alone(self, adapter):
        self._queued_owner_run(adapter, dead=False)
        adapter._recover_orphaned_owner_jobs()

        store = adapter._response_store
        assert store.run_idempotency_status("default", self._RUN_ID)["status"] == (
            "queued"
        )
        assert store.owner_claim_is_released(
            "default", self._CONVERSATION, self._RESPONSE_ID,
            self._CLAIM_ID, self._RUN_ID,
        ) is False

    @pytest.mark.asyncio
    async def test_a_run_that_already_persisted_its_receipt_is_left_alone(
        self, adapter,
    ):
        reserved = self._queued_owner_run(adapter)
        store = adapter._response_store
        # Exactly what ``persist_owner_run_completion`` leaves behind.
        store._conn.execute(
            "UPDATE run_idempotency SET terminal_json = ?, status_json = NULL "
            "WHERE run_id = ?",
            (json.dumps({"status": "completed"}), self._RUN_ID),
        )
        store._conn.commit()

        adapter._recover_orphaned_owner_jobs()

        assert store.run_idempotency_status("default", self._RUN_ID) is None
        assert store.owner_claim_is_released(
            "default", self._CONVERSATION, reserved["owner"]["response_id"],
            self._CLAIM_ID, self._RUN_ID,
        ) is False

    # -----------------------------------------------------------------------
    # Item 32TK round 2
    # -----------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_a_run_reservation_and_its_recovery_job_commit_together(
        self, adapter,
    ):
        """A crash between them left a durable queued run and a claimed proposal
        with no executor: polling reported working forever and the owner could
        not approve the same proposal again."""
        store = adapter._response_store
        scope = hashlib.sha256(b"").hexdigest()
        with patch.object(
            store, "_reserve_owner_job_locked",
            side_effect=RuntimeError("disk full"),
        ):
            with pytest.raises(RuntimeError):
                store.reserve_run_idempotency(
                    "default", scope, "atomic-key", "fingerprint", self._RUN_ID,
                    job_payload={"owner": None},
                )

        # Neither half landed.
        assert store.run_idempotency_status("default", self._RUN_ID) is None
        assert store._conn.execute(
            "SELECT COUNT(*) FROM owner_executor_jobs WHERE job_key = ?",
            (self._RUN_ID,),
        ).fetchone()[0] == 0

        state, _run = store.reserve_run_idempotency(
            "default", scope, "atomic-key", "fingerprint", self._RUN_ID,
            job_payload={"owner": None},
        )
        assert state == "new"
        assert store.run_idempotency_status("default", self._RUN_ID)["status"] == (
            "queued"
        )
        assert store._conn.execute(
            "SELECT COUNT(*) FROM owner_executor_jobs WHERE job_key = ?",
            (self._RUN_ID,),
        ).fetchone()[0] == 1

    @pytest.mark.asyncio
    async def test_a_proposal_claim_and_its_recovery_job_commit_together(
        self, adapter,
    ):
        store = adapter._response_store
        conversation = "raphael-owner-" + "7" * 32
        response_id = "resp_" + "7" * 28
        claim_id = "claim_" + "7" * 20
        store.put(response_id, {
            "response": {"id": response_id, "created_at": 100},
            "conversation_history": [
                {"role": "assistant", "content": json.dumps(_new_owner_proposal())},
            ],
        })
        assert store.set_conversation(
            conversation, response_id, owner_proposal=True,
        ) is True

        with patch.object(
            store, "_reserve_owner_job_locked",
            side_effect=RuntimeError("disk full"),
        ):
            with pytest.raises(RuntimeError):
                store.claim_and_attach_owner_run(
                    "default", conversation, response_id, claim_id, self._RUN_ID,
                    operation="owner_task_graph_commit",
                    payload_digest="d" * 64,
                    job_payload={},
                    job_profile="default",
                )

        # The proposal is NOT claimed by a run nobody is driving.
        snapshot = store.owner_history_snapshot(conversation)
        assert snapshot["proposal_claimed"] is False
        assert snapshot["active_run_id"] is None
        assert store._conn.execute(
            "SELECT COUNT(*) FROM owner_executor_jobs WHERE job_key = ?",
            (self._RUN_ID,),
        ).fetchone()[0] == 0

    @pytest.mark.asyncio
    async def test_a_recycled_pid_cannot_strand_queued_owner_work(self, adapter):
        """PID liveness alone is not a fence: a recycled pid looks alive
        forever, so an expired LEASE has to be enough to reclaim the job."""
        self._queued_owner_run(adapter, dead=False)
        store = adapter._response_store
        # A sibling executor that is gone, whose pid has since been reused by
        # this very process — the exact case PID liveness cannot decide.
        store._conn.execute(
            "UPDATE owner_executor_jobs SET executor_id = 'dead-sibling', "
            "executor_pid = ?, created_at = ?, lease_expires_at = ? "
            "WHERE job_key = ?",
            (os.getpid(), time.time() - 3600, time.time() - 1, self._RUN_ID),
        )
        store._conn.commit()

        adapter._recover_orphaned_owner_jobs()

        assert store.run_idempotency_status("default", self._RUN_ID)["status"] == (
            "failed"
        )

    @pytest.mark.asyncio
    async def test_a_heartbeat_keeps_a_live_executors_job(self, adapter):
        """Work that legitimately outlives one lease must keep saying it is
        alive, or a sibling gateway would reclaim it mid-flight."""
        self._queued_owner_run(adapter, dead=False)
        store = adapter._response_store
        store._conn.execute(
            "UPDATE owner_executor_jobs SET created_at = ?, lease_expires_at = ? "
            "WHERE job_key = ?",
            (time.time() - 3600, time.time() - 1, self._RUN_ID),
        )
        store._conn.commit()
        # This process is genuinely still driving the run.
        adapter._active_run_tasks[self._RUN_ID] = SimpleNamespace(
            done=lambda: False
        )
        adapter._heartbeat_owner_job_leases()

        # The renewed lease is what a SIBLING sees.
        assert (
            store._conn.execute(
                "SELECT lease_expires_at FROM owner_executor_jobs WHERE job_key = ?",
                (self._RUN_ID,),
            ).fetchone()[0]
            > time.time()
        )
        adapter._recover_orphaned_owner_jobs()
        assert store.run_idempotency_status("default", self._RUN_ID)["status"] == (
            "queued"
        )

    @pytest.mark.asyncio
    async def test_this_process_reclaims_its_own_job_once_its_task_is_gone(
        self, adapter,
    ):
        """A task that died without recording a terminal state leaves a job row
        only this process can tell is abandoned. Skipping every one of its own
        rows left such a response or run queued until the next restart."""
        self._queued_owner_run(adapter, dead=False)
        store = adapter._response_store
        store._conn.execute(
            "UPDATE owner_executor_jobs SET created_at = ? WHERE job_key = ?",
            (time.time() - 3600, self._RUN_ID),
        )
        store._conn.commit()
        assert adapter._active_run_tasks.get(self._RUN_ID) is None

        adapter._recover_orphaned_owner_jobs()

        assert store.run_idempotency_status("default", self._RUN_ID)["status"] == (
            "failed"
        )

    @pytest.mark.asyncio
    async def test_an_orphan_job_survives_a_failed_terminalization(self, adapter):
        """The job row is the only record of what to recover, so deleting it
        before terminalizing lost the recovery authority for good."""
        self._queued_owner_run(adapter)
        store = adapter._response_store
        with patch.object(
            store, "fail_orphaned_owner_run", side_effect=RuntimeError("db is busy"),
        ):
            adapter._recover_orphaned_owner_jobs()

        # Still queued AND still recoverable.
        assert store.run_idempotency_status("default", self._RUN_ID)["status"] == (
            "queued"
        )
        store._conn.execute(
            "UPDATE owner_executor_jobs SET executor_id = 'dead', "
            "executor_pid = 2147483646, created_at = ?, lease_expires_at = ? "
            "WHERE job_key = ?",
            (time.time() - 3600, time.time() - 1, self._RUN_ID),
        )
        store._conn.commit()
        adapter._recover_orphaned_owner_jobs()
        assert store.run_idempotency_status("default", self._RUN_ID)["status"] == (
            "failed"
        )

    @pytest.mark.asyncio
    async def test_a_committed_native_receipt_is_reconciled_before_failing(
        self, adapter,
    ):
        """The gateway can die after the native database commits. Reporting that
        completed change as failed — and releasing its approval — invited the
        owner to run the very same mutation twice."""
        reserved = self._queued_owner_run(adapter)
        store = adapter._response_store
        store._conn.execute(
            "UPDATE conversations SET bound_operation = ?, "
            "bound_payload_digest = ? WHERE name = ?",
            ("owner_task_graph_commit", "d" * 64, self._CONVERSATION),
        )
        store._conn.execute(
            "UPDATE owner_executor_jobs SET payload = ? WHERE job_key = ?",
            (
                json.dumps({
                    "owner": {
                        **reserved["owner"],
                        "operation": "owner_task_graph_commit",
                    },
                    "native": {
                        "operation": "owner_task_graph_commit",
                        "idempotency_key": reserved["key"],
                        "authority_digest": "d" * 64,
                        "session_scope": reserved["scope"],
                    },
                }),
                self._RUN_ID,
            ),
        )
        store._conn.commit()

        with patch(
            "hermes_cli.owner_workspace.read_committed_owner_run_receipt",
            return_value={
                "ok": True, "project_slug": "workshop-pilot", "task_count": 2,
            },
        ):
            adapter._recover_orphaned_owner_jobs()

        completion = store.owner_run_completion("default", self._RUN_ID)
        assert completion["status"] == "completed"
        assert completion["owner_mutation_committed"] is True
        # The approval was CONSUMED, not released back to the owner.
        assert store.owner_claim_is_released(
            "default", self._CONVERSATION, self._RESPONSE_ID,
            self._CLAIM_ID, self._RUN_ID,
        ) is False

    @pytest.mark.asyncio
    async def test_an_unreadable_native_receipt_never_declares_failure(
        self, adapter,
    ):
        reserved = self._queued_owner_run(adapter)
        store = adapter._response_store
        store._conn.execute(
            "UPDATE owner_executor_jobs SET payload = ? WHERE job_key = ?",
            (
                json.dumps({
                    "owner": reserved["owner"],
                    "native": {
                        "operation": "owner_task_graph_commit",
                        "idempotency_key": reserved["key"],
                        "authority_digest": "e" * 64,
                        "session_scope": reserved["scope"],
                    },
                }),
                self._RUN_ID,
            ),
        )
        store._conn.commit()

        from hermes_cli.owner_workspace import OwnerReceiptUnreadable

        with patch(
            "hermes_cli.owner_workspace.read_committed_owner_run_receipt",
            side_effect=OwnerReceiptUnreadable("two committed receipts"),
        ):
            adapter._recover_orphaned_owner_jobs()

        # Undecided: nothing was failed, nothing was released, and the job row
        # is still there for the next pass.
        assert store.run_idempotency_status("default", self._RUN_ID)["status"] == (
            "queued"
        )
        assert store.owner_claim_is_released(
            "default", self._CONVERSATION, self._RESPONSE_ID,
            self._CLAIM_ID, self._RUN_ID,
        ) is False
        store._conn.execute(
            "UPDATE owner_executor_jobs SET executor_id = 'dead', "
            "executor_pid = 2147483646, created_at = ?, lease_expires_at = ? "
            "WHERE job_key = ?",
            (time.time() - 3600, time.time() - 1, self._RUN_ID),
        )
        store._conn.commit()
        assert [job["job_key"] for job in store.claim_orphaned_owner_jobs("run")] == [
            self._RUN_ID
        ]

    @pytest.mark.asyncio
    async def test_a_terminal_run_persists_its_status_with_its_job(self, adapter):
        """After a restart the durable row used to still say queued, so polling
        reported working forever."""
        self._queued_owner_run(adapter, dead=False)
        store = adapter._response_store
        adapter._run_statuses[self._RUN_ID] = {
            "object": "hermes.run", "run_id": self._RUN_ID, "status": "cancelled",
        }

        adapter._finalize_run_recovery_job(self._RUN_ID, "default")

        assert store.run_idempotency_status("default", self._RUN_ID)["status"] == (
            "cancelled"
        )
        assert store.claim_orphaned_owner_jobs("run") == []


    @pytest.mark.asyncio
    async def test_a_refused_owner_mutation_persists_its_reason_and_code(self, adapter):
        """The kernel's refusal is plain, redacted owner text; the durable row
        keeps it with its code so a restarted service can still say why."""
        self._queued_owner_run(adapter, dead=False)
        store = adapter._response_store
        adapter._run_statuses[self._RUN_ID] = {
            "object": "hermes.run", "run_id": self._RUN_ID, "status": "failed",
            "error": "The approved Project change was not committed: "
                     "owner_project_plan_commit: changes declares repository ownership",
            "error_reason": "owner_project_plan_commit: changes declares repository ownership",
            "error_code": "ownership_scope_unavailable",
            "owner_mutation_committed": False,
        }

        adapter._finalize_run_recovery_job(self._RUN_ID, "default")

        durable = store.run_idempotency_status("default", self._RUN_ID)
        assert durable["status"] == "failed"
        # The bare reason, whole: the transport prefix is not part of the row.
        assert durable["error"] == (
            "owner_project_plan_commit: changes declares repository ownership"
        )
        assert durable["error_code"] == "ownership_scope_unavailable"

    @pytest.mark.asyncio
    async def test_any_other_failure_stays_non_diagnostic(self, adapter):
        """Exception text never becomes a durable owner-readable row."""
        self._queued_owner_run(adapter, dead=False)
        store = adapter._response_store
        adapter._run_statuses[self._RUN_ID] = {
            "object": "hermes.run", "run_id": self._RUN_ID, "status": "failed",
            "error": "Traceback: KeyError: 'model'",
        }

        adapter._finalize_run_recovery_job(self._RUN_ID, "default")

        durable = store.run_idempotency_status("default", self._RUN_ID)
        assert durable["error"] == _OWNER_RUN_STOPPED_MESSAGE
        assert "error_code" not in durable


class TestOwnerMutationRefusal:
    """An owner mutation that did not commit reports the kernel's own reason."""

    def test_reason_is_taken_from_the_handler_error(self):
        result = {"final_response": '{"error": "owner_project_plan_commit: changes[2].title contains private operational detail"}'}
        assert _owner_mutation_refusal(result) == (
            "owner_project_plan_commit: changes[2].title contains private operational detail"
        )

    def test_secret_bearing_reasons_are_redacted(self):
        result = '{"error": "route sk-ABCDEFGHIJKLMNOPQRSTUV refused"}'
        refusal = _owner_mutation_refusal(result)
        assert refusal
        assert "sk-ABCDEFGHIJKLMNOPQRSTUV" not in refusal

    @pytest.mark.parametrize(
        "result",
        [None, "", "not json", '{"ok": true}', {"final_response": '{"error": "   "}'}, {"other": 1}],
    )
    def test_silent_when_no_reason_is_available(self, result):
        assert _owner_mutation_refusal(result) == ""

    def test_the_stable_code_rides_along(self):
        result = {"final_response": '{"error": "owner_project_plan_commit: refused", "code": "ownership_scope_unavailable"}'}
        assert _owner_mutation_refusal_code(result) == "ownership_scope_unavailable"

    @pytest.mark.parametrize(
        "result",
        [
            None,
            '{"error": "refused"}',
            '{"error": "refused", "code": ""}',
            '{"error": "refused", "code": "Not A Slug"}',
            '{"error": "refused", "code": 7}',
            {"final_response": '{"error": "refused", "code": "x" * 65}'},
        ],
    )
    def test_a_missing_or_malformed_code_is_dropped(self, result):
        assert _owner_mutation_refusal_code(result) == ""
