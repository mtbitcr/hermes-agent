"""
Tests for the OpenAI-compatible API server gateway adapter.

Tests cover:
- Chat Completions endpoint (request parsing, response format)
- Responses API endpoint (request parsing, response format)
- previous_response_id chaining (store/retrieve)
- Auth (valid key, invalid key, no key configured)
- /v1/models endpoint
- /health endpoint
- System prompt extraction
- Error handling (invalid JSON, missing fields)
"""

import asyncio
import base64
import hashlib
import hmac
import json
import os
import sqlite3
import stat
import sys
import threading
import time
import types
import uuid
from unittest.mock import ANY, AsyncMock, MagicMock, patch

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from agent.message_metadata import stamp_message_timestamp
from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.api_server import (
    APIServerAdapter,
    ResponseStore,
    _IdempotencyCache,
    _derive_chat_session_id,
    _hermes_version,
    _make_request_fingerprint,
    _owner_current_replace_shape,
    _owner_current_task_shape,
    _OWNER_PROPOSAL_ADD_KEYS,
    _OWNER_PROPOSAL_SPLIT_TASK_KEYS,
    _OWNER_INTERRUPTED_TURN_MESSAGE,
    OwnerAuthorityBroken,
    OwnerAuthorityUnavailable,
    OwnerTurnNotRecoverable,
    _redact_api_error_text,
    _resolve_owner_workspace_run_context,
    _request_reasoning_config,
    _request_agent_overrides,
    check_api_server_requirements,
    cors_middleware,
    security_headers_middleware,
)


def _owner_new_proposal(**overrides):
    proposal = {
        # v3 is the first new-project schema carrying execution_tier, and
        # therefore the first that can grant commit authority.
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
    proposal.update(overrides)
    return proposal


def _owner_existing_proposal(**overrides):
    proposal = {
        "schema_version": 5,
        "kind": "project_change_proposal",
        "mode": "existing",
        "request_title": "Add the approved milestone",
        "summary": "Add one owner-visible milestone.",
        "project_size": "small",
        "specification": "Add and verify the approved milestone.",
        "current_milestone": "Add the approved milestone",
        "owner_visible_result": "The owner can review the finished milestone.",
        "impact": ["Keeps completed work intact."],
        "later_milestones": [],
        "changes": [{
            "action": "add",
            "reason": "The approved milestone needs one task.",
            "title": "Prepare the milestone",
            "body": "Prepare the owner-visible milestone.",
            "assignee": "default",
            "responsibility": "B03",
            "execution_tier": "routine",
            "owned_paths": [],
            "existing_parent_refs": [],
            "new_parents": [],
        }],
    }
    proposal.update(overrides)
    return proposal


def test_schema_v5_created_tasks_require_explicit_array_scope_and_body_mode():
    add = _owner_existing_proposal()["changes"][0]
    assert _owner_current_task_shape(add, _OWNER_PROPOSAL_ADD_KEYS)
    assert not _owner_current_task_shape(
        {key: value for key, value in add.items() if key != "owned_paths"},
        _OWNER_PROPOSAL_ADD_KEYS,
    )
    assert not _owner_current_task_shape(
        {**add, "owned_paths": None},
        _OWNER_PROPOSAL_ADD_KEYS,
    )

    split_task = {
        "title": "Prepare the milestone",
        "body": "Prepare the owner-visible milestone.",
        "assignee": "default",
        "responsibility": "B03",
        "execution_tier": "routine",
        "owned_paths": [],
        "parents": [],
    }
    assert _owner_current_task_shape(split_task, _OWNER_PROPOSAL_SPLIT_TASK_KEYS)

    common = {
        "title": "Prepare the milestone",
        "assignee": "default",
        "responsibility": "B03",
        "execution_tier": "routine",
        "owned_paths": [],
    }
    assert _owner_current_replace_shape({**common, "body_mode": "preserve"})
    assert _owner_current_replace_shape({
        **common,
        "body_mode": "rewrite",
        "body": "Prepare the revised milestone.",
    })
    assert not _owner_current_replace_shape({
        **common,
        "body": "Legacy replacement body.",
    })
    assert not _owner_current_replace_shape({
        **common,
        "body_mode": "preserve",
        "owned_paths": None,
    })
    assert not _owner_current_replace_shape({**common, "body_mode": []})


# ---------------------------------------------------------------------------
# check_api_server_requirements
# ---------------------------------------------------------------------------


class TestCheckRequirements:

    @patch("gateway.platforms.api_server.AIOHTTP_AVAILABLE", False)
    def test_returns_false_without_aiohttp(self):
        assert check_api_server_requirements() is False


class TestRequestReasoningConfig:
    def test_accepts_max_effort(self):
        assert _request_reasoning_config({"reasoning_effort": "max"}) == {
            "enabled": True,
            "effort": "max",
        }

    def test_ignores_unknown_effort(self):
        assert _request_reasoning_config({"reasoning_effort": "ultra"}) is None


class TestOwnerWorkspaceRunContext:
    def test_existing_context_accepts_receipt_backed_archived_project(self):
        config = {"gateway": {"api_server": {"owner_workspace": {"enabled": True}}}}
        projects = [{
            "project_id": "p_archived",
            "slug": "archived-project",
            "name": "Archived Project",
            "description": "Retained owner project",
            "board": "archived-project",
            "archived": True,
        }]

        with (
            patch("gateway.run._load_gateway_config", return_value=config),
            patch(
                "hermes_cli.owner_workspace.resolve_owner_context",
                return_value=types.SimpleNamespace(profile="default"),
            ),
            patch(
                "hermes_cli.owner_workspace.list_committed_projects",
                return_value=projects,
            ),
        ):
            result = _resolve_owner_workspace_run_context({
                "mode": "existing",
                "project_slug": "archived-project",
                "project_name": "ignored client name",
            })

        assert result == {
            "mode": "existing",
            "project_slug": "archived-project",
            "project_name": "Archived Project",
            "profile": "default",
        }

    def test_retained_name_goes_through_the_canonical_project_projection(self):
        """The name kept for the Decisions inbox is owner-facing display text.

        It is the same string ``/v1`` hands back on an approval, so it must
        leave here already sanitized and credential-masked, not merely
        length-checked. Both modes are covered: a client-supplied new name
        and a stored name read back off the receipt-backed Project list.

        Projecting is not the same as accepting: a *new* name over the
        160-code-point bound ``commit_task_graph`` enforces is still rejected
        fail-fast rather than truncated into a name the client never asked
        for.
        """
        from hermes_cli.owner_workspace import owner_project_name

        config = {"gateway": {"api_server": {"owner_workspace": {"enabled": True}}}}
        zero_width, annotation_anchor = chr(0x200B), chr(0xFFF9)
        unsafe = (
            f"Wo{zero_width}rkshop\x00 \x1b[31mpilot{annotation_anchor}"
            " https://deploy:hunter2verylongpassword@git.example.com/repo.git"
        )
        projected = (
            "Workshop pilot https://***@git.example.com/repo.git"
        )
        assert owner_project_name(unsafe) == projected

        projects = [{
            "project_id": "p_unsafe",
            "slug": "stored-project",
            "name": unsafe,
            "description": "",
            "board": "stored-project",
            "archived": False,
        }]

        with (
            patch("gateway.run._load_gateway_config", return_value=config),
            patch(
                "hermes_cli.owner_workspace.resolve_owner_context",
                return_value=types.SimpleNamespace(profile="default"),
            ),
            patch(
                "hermes_cli.owner_workspace.list_committed_projects",
                return_value=projects,
            ),
        ):
            new = _resolve_owner_workspace_run_context({
                "mode": "new",
                "project_slug": None,
                "project_name": unsafe,
            })
            existing = _resolve_owner_workspace_run_context({
                "mode": "existing",
                "project_slug": "stored-project",
                "project_name": None,
            })
            at_bound = _resolve_owner_workspace_run_context({
                "mode": "new",
                "project_slug": None,
                "project_name": "n" * 160,
            })
            with pytest.raises(ValueError):
                _resolve_owner_workspace_run_context({
                    "mode": "new",
                    "project_slug": None,
                    "project_name": "n" * 161,
                })

        assert new["project_name"] == projected
        assert existing["project_name"] == projected
        assert at_bound["project_name"] == "n" * 160

    def test_an_unanswered_owner_request_authorizes_no_run(self, monkeypatch):
        """Item 32TK, finding 2: the request-level door, before the store's.

        ``/v1/runs`` derives owner mutation authority here, long before it
        reserves anything, so a conversation carrying a request that ended
        without a turn is refused at this end too. Reading the stored proposal
        at all is already work done for an approval that cannot be granted, so
        the refusal comes first — and once the record is acknowledged this
        validation carries on exactly as it always did.
        """
        conversation = "raphael-owner-" + "4" * 32
        authority = {
            "proposal_profile": "default",
            "conversation": conversation,
            "response_id": "resp_older_run_proposal",
            "claim_id": "claim_" + "6" * 32,
            "operation": "owner_task_graph_commit",
            "idempotency_key": "conversation-" + "a" * 64,
            "payload": {},
        }
        context = {
            "profile": "default",
            "mode": "new",
            "project_slug": None,
            "project_name": "Workshop pilot",
        }
        store = ResponseStore(max_size=10)
        adapter = APIServerAdapter.__new__(APIServerAdapter)
        adapter._response_store = store
        read = []
        try:
            assert store.reserve_owner_conversation(
                "default", conversation, "resp_unanswered_run_request",
                owner_message="Prepare a private 60-minute workshop",
            ) is True
            _interrupt_owner_turn(
                store, conversation, "resp_unanswered_run_request",
            )
            monkeypatch.setattr(
                store,
                "owner_proposal_record",
                lambda *args: read.append(args),
            )

            with pytest.raises(ValueError, match="unanswered"):
                adapter._validated_owner_proposal_authority(
                    authority, context, "default",
                )
            assert read == []

            assert store.acknowledge_owner_conversation_recovery(
                "default", conversation, "resp_unanswered_run_request",
            ) == "retired"
            # Past that door, and refused for what it actually is: this
            # conversation has no such proposal on it.
            with pytest.raises(ValueError, match="not current"):
                adapter._validated_owner_proposal_authority(
                    authority, context, "default",
                )
            assert read == [
                ("default", conversation, "resp_older_run_proposal"),
            ]
        finally:
            store.close()

    @pytest.mark.parametrize(
        ("replacement", "authorized"),
        [
            ({
                "title": "Complete the bounded recovery",
                "body_mode": "preserve",
                "assignee": "raphael-claude-worker",
                "responsibility": "R07",
                "execution_tier": "deep",
                "owned_paths": [],
            }, True),
            ({
                "title": "Complete the bounded recovery",
                "body": "Legacy replacement body.",
                "assignee": "raphael-claude-worker",
                "responsibility": "R07",
                "execution_tier": "deep",
            }, False),
        ],
    )
    def test_stored_replace_proposal_validates_exact_native_run_payload(
        self, replacement, authorized,
    ):
        conversation = "raphael-owner-" + "4" * 32
        response_id = "resp_native_replace_proposal"
        idempotency_key = "conversation-" + hashlib.sha256(
            response_id.encode("utf-8")
        ).hexdigest()
        secret = "owner-executor-test-key"
        task = {
            "id": "task_old_implementation",
            "title": "Repair the stopped implementation",
            "status": "triage",
            "event_revision": 7,
            "parent_ids": [],
            "child_ids": [],
            "omitted_parent_count": 0,
            "omitted_child_count": 0,
        }
        canonical = json.dumps({
            "version": 1,
            "project_id": "project_raphael",
            "task_id": task["id"],
            "title": task["title"],
            "status": task["status"],
            "event_revision": task["event_revision"],
            "parent_ids": task["parent_ids"],
            "child_ids": task["child_ids"],
            "omitted_parent_count": task["omitted_parent_count"],
            "omitted_child_count": task["omitted_child_count"],
        }, separators=(",", ":"), ensure_ascii=False)
        target_ref = "tr_" + base64.urlsafe_b64encode(
            hmac.new(
                secret.encode("utf-8"), canonical.encode("utf-8"), hashlib.sha256,
            ).digest()
        ).decode("ascii").rstrip("=")
        proposal = _owner_existing_proposal(changes=[{
            "action": "replace",
            "reason": "The earlier attempt stopped before producing a result.",
            "target_ref": target_ref,
            "replacement": replacement,
        }])
        payload = {
            "idempotency_key": idempotency_key,
            "project_id": "project_raphael",
            "trigger": "owner_request",
            "request_title": proposal["request_title"],
            "summary": proposal["summary"],
            "specification": proposal["specification"],
            "current_milestone": proposal["current_milestone"],
            "owner_visible_result": proposal["owner_visible_result"],
            "later_milestones": proposal["later_milestones"],
            "changes": [{
                "action": "replace",
                "reason": proposal["changes"][0]["reason"],
                "target": {
                    "task_id": task["id"],
                    "expected_status": task["status"],
                    "expected_revision": task["event_revision"],
                },
                "replacement": replacement,
            }],
        }
        authority = {
            "proposal_profile": "default",
            "conversation": conversation,
            "response_id": response_id,
            "claim_id": "claim_" + "6" * 32,
            "operation": "owner_project_plan_commit",
            "idempotency_key": idempotency_key,
            "payload": payload,
        }
        context = {
            "profile": "default",
            "mode": "existing",
            "project_slug": "raphael-workspace",
            "project_name": "Raphael Workspace",
        }
        snapshot = {
            "project": {"id": "project_raphael"},
            "planning_context": {
                "schema_version": 1,
                "actionable_count": 1,
                "omitted_terminal_count": 0,
                "actionable_truncated": False,
                "relations_truncated": False,
                "tasks": [task],
            },
        }
        store = ResponseStore(max_size=10)
        adapter = APIServerAdapter.__new__(APIServerAdapter)
        adapter._response_store = store
        adapter._expected_api_key = lambda: secret
        try:
            store.put(response_id, {
                "response": {"id": response_id, "created_at": 1},
                "conversation_history": [
                    {"role": "user", "content": "Continue the stopped work."},
                    {"role": "assistant", "content": json.dumps(proposal)},
                ],
            })
            assert store.set_conversation(
                conversation, response_id, owner_proposal=True,
            ) is True
            with (
                patch(
                    "hermes_cli.owner_workspace.resolve_owner_context",
                    return_value=object(),
                ),
                patch(
                    "hermes_cli.owner_workspace.read_project_snapshot",
                    return_value=snapshot,
                ),
            ):
                if authorized:
                    binding = adapter._validated_owner_proposal_authority(
                        authority, context, "default",
                    )
                else:
                    with pytest.raises(
                        ValueError, match="stored proposal change is invalid",
                    ):
                        adapter._validated_owner_proposal_authority(
                            authority, context, "default",
                        )
                    return

            assert binding["operation"] == "owner_project_plan_commit"
            assert binding["idempotency_key"] == idempotency_key
        finally:
            store.close()

    @pytest.mark.parametrize("blank", ["", "   ", "\t\n"])
    def test_new_mode_still_requires_a_name(self, blank):
        config = {"gateway": {"api_server": {"owner_workspace": {"enabled": True}}}}
        with (
            patch("gateway.run._load_gateway_config", return_value=config),
            patch(
                "hermes_cli.owner_workspace.resolve_owner_context",
                return_value=types.SimpleNamespace(profile="default"),
            ),
            patch(
                "hermes_cli.owner_workspace.list_committed_projects",
                return_value=[],
            ),
            pytest.raises(ValueError),
        ):
            _resolve_owner_workspace_run_context({
                "mode": "new", "project_slug": None, "project_name": blank,
            })


# ---------------------------------------------------------------------------
# _redact_api_error_text — guards every outward error site (envelopes, SSE
# error events, cron-endpoint 500 bodies) that routes raw exception text to
# authenticated HTTP clients. #37733
# ---------------------------------------------------------------------------


class TestRedactApiErrorText:
    def test_masks_secret_value_but_preserves_structure(self):
        secret = "sk-api-server-leak-1234567890"
        out = _redact_api_error_text(Exception(f"auth failed OPENAI_API_KEY={secret}"))
        assert secret not in out
        assert "OPENAI_API_KEY=" in out

    def test_redacts_regardless_of_global_redaction_setting(self):
        # force=True must mask even when global redaction is disabled.
        secret = "sk-forced-redaction-0987654321"
        with patch("agent.redact._REDACT_ENABLED", False):
            out = _redact_api_error_text(Exception(f"boom AWS_SECRET_ACCESS_KEY={secret}"))
        assert secret not in out

    def test_limit_truncates_after_redaction(self):
        assert len(_redact_api_error_text("x" * 500, limit=50)) == 50


# ---------------------------------------------------------------------------
# ResponseStore
# ---------------------------------------------------------------------------


def _interrupt_owner_turn(store, conversation, response_id, profile="default"):
    """End one accepted owner turn the way a background failure really does.

    Exactly the call the background path makes: the terminal body, this turn's
    fence becoming the record that carries the owner's request, the outcome its
    idempotency key replays, and the retirement of its recovery job, as ONE
    transaction. Tests set the state up through the production entry point so
    they cannot drift from it.
    """
    store.store_terminal_owner_response(
        profile=profile,
        response_id=response_id,
        data={
            "response": {
                "id": response_id,
                "object": "response",
                "status": "failed",
                "output": [],
                "error": {"code": "server_error", "message": "stopped"},
            },
            "conversation_history": [],
        },
        release_job=True,
        conversation=conversation,
        interrupted=True,
    )


class TestResponseStore:
    def test_put_and_get(self):
        store = ResponseStore(max_size=10)
        store.put("resp_1", {"output": "hello"})
        assert store.get("resp_1") == {"output": "hello"}

    def test_get_missing_returns_none(self):
        store = ResponseStore(max_size=10)
        assert store.get("resp_missing") is None

    def test_get_waits_for_the_shared_response_store_transaction_lock(self):
        store = ResponseStore(max_size=10)
        store.put("resp_locked", {"output": "safe"})
        started = threading.Event()
        finished = threading.Event()
        result = []

        def read_response():
            started.set()
            try:
                result.append(store.get("resp_locked"))
            finally:
                finished.set()

        reader = threading.Thread(target=read_response, daemon=True)
        with store._conversation_lock:
            reader.start()
            assert started.wait(1)
            blocked_while_transaction_lock_is_held = not finished.wait(0.1)

        reader.join(timeout=1)
        try:
            assert blocked_while_transaction_lock_is_held is True
            assert finished.is_set()
            assert result == [{"output": "safe"}]
        finally:
            store.close()

    def test_lru_eviction(self):
        store = ResponseStore(max_size=3)
        store.put("resp_1", {"output": "one"})
        store.put("resp_2", {"output": "two"})
        store.put("resp_3", {"output": "three"})
        # Adding a 4th should evict resp_1
        store.put("resp_4", {"output": "four"})
        assert store.get("resp_1") is None
        assert store.get("resp_2") is not None
        assert len(store) == 3


    def test_delete_clears_conversation_mapping(self):
        """Deleting a response also removes conversation mappings that reference it."""
        store = ResponseStore(max_size=10)
        store.put("resp_1", {"output": "hello"})
        store.set_conversation("chat-a", "resp_1")
        assert store.get_conversation("chat-a") == "resp_1"
        store.delete("resp_1")
        assert store.get_conversation("chat-a") is None

    def test_owner_authority_survives_lru_pressure_and_direct_delete(self):
        """Owner approval state is durable workflow data, not an LRU entry."""
        conversation = "raphael-owner-" + "f" * 32
        response_id = "resp_owner_lru_anchor"
        claim_id = "claim_" + "a" * 32
        run_id = "run_" + "b" * 32
        store = ResponseStore(max_size=1)
        store.put(response_id, {
            "response": {"id": response_id, "created_at": 1},
            "conversation_history": [
                {"role": "user", "content": "Apply the approved milestone."},
                {"role": "assistant", "content": json.dumps(
                    _owner_existing_proposal(
                        request_title="Apply the approved milestone",
                    )
                )},
            ],
        })
        assert store.set_conversation(
            conversation, response_id, owner_proposal=True,
        ) is True
        assert store.claim_owner_proposal(
            "default", conversation, response_id, claim_id,
        ) is True
        assert store.attach_owner_run(
            "default", conversation, response_id, claim_id, run_id,
        ) is True

        # The insertion itself may become the next owner response, so put()
        # leaves it available while retaining the current owner authority.
        store.put("resp_lru_pressure", {"response": {"id": "resp_lru_pressure"}})

        snapshot = store.owner_history_snapshot(conversation)
        assert snapshot["latest_response_id"] == response_id
        assert snapshot["proposal_claimed"] is True
        assert snapshot["active_run_id"] == run_id
        assert store.get(response_id) is not None
        assert store.delete(response_id) is False
        assert store.owner_history_snapshot(conversation)["active_run_id"] == run_id

    def test_incomplete_proposal_never_grants_approval_authority(self):
        conversation = "raphael-owner-" + "0" * 32
        response_id = "resp_incomplete_proposal"
        store = ResponseStore(max_size=10)
        store.put(response_id, {
            "response": {"id": response_id, "created_at": 1},
            "conversation_history": [{
                "role": "assistant",
                "content": json.dumps({
                    "schema_version": 2,
                    "kind": "proposal",
                    "mode": "new",
                }),
            }],
        })

        assert store.set_conversation(
            conversation, response_id, owner_proposal=True,
        ) is False
        assert store.owner_proposal_record(
            "default", conversation, response_id,
        ) is None

    def test_owner_history_projects_only_final_structured_turns(self):
        secret = "sk-ant-api03-" + "a" * 80
        conversation = "raphael-owner-" + "a" * 32
        store = ResponseStore(max_size=10)
        final_proposal = json.dumps({
            "schema_version": 1,
            "kind": "proposal",
            "mode": "new",
            "project_name": f"Workshop pilot {secret}",
        })
        final_question = json.dumps({
            "schema_version": 1,
            "kind": "question",
            "message": "Which week should it run?",
        })
        final_change = json.dumps(_owner_existing_proposal(
            request_title="Add a weekly summary",
        ))
        store.put("resp_owner_history", {
            "response": {"id": "resp_owner_history"},
            "conversation_history": [
                {"role": "system", "content": "private instructions"},
                {"role": "user", "content": "Plan a workshop."},
                {"role": "assistant", "content": "private intermediate reasoning"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{"function": {"name": "private_tool"}}],
                },
                {"role": "tool", "content": "private tool output"},
                {"role": "assistant", "content": final_proposal},
                {"role": "user", "content": "Make it next month."},
                {"role": "assistant", "content": final_question},
                {"role": "user", "content": "Add a weekly summary."},
                {"role": "assistant", "content": final_change},
            ],
            "instructions": "private instructions",
        })
        store.set_conversation(
            conversation, "resp_owner_history", owner_proposal=True,
        )

        history = store.owner_history(conversation)
        assert [item["owner"] for item in history] == [
            "Plan a workshop.", "Make it next month.", "Add a weekly summary.",
        ]
        assert json.loads(history[0]["raphael"])["kind"] == "proposal"
        assert json.loads(history[1]["raphael"])["kind"] == "question"
        assert json.loads(history[2]["raphael"])["kind"] == "project_change_proposal"
        assert secret not in json.dumps(history)

        snapshot = store.owner_history_snapshot(conversation)
        assert snapshot == {
            "head_response_id": "resp_owner_history",
            "latest_response_id": "resp_owner_history",
            "proposal_consumed": False,
            "proposal_claimed": False,
            "active_run_id": None,
            "completed_run_id": None,
            "conversation_closed": False,
            "truncated": False,
            "incomplete": False,
            # Nothing is being planned on this conversation right now, and
            # nothing it planned before was left interrupted.
            "pending": None,
            "recovery": None,
            "data": history,
        }
        assert "resp_owner_history" not in json.dumps(snapshot["data"])

    def test_owner_history_keeps_reply_within_native_response_limit(self):
        conversation = "raphael-owner-" + "b" * 32
        store = ResponseStore(max_size=10)
        proposal = _owner_existing_proposal(
            request_title="Add the next private milestone",
            summary="Prepare the next owner-visible milestone for review.",
            project_size="large",
            specification="Readable milestone detail. " * 500,
            current_milestone="Prepare the next private milestone.",
            owner_visible_result="The owner can review one complete proposal.",
            impact=["Keeps the current project history intact."],
        )
        structured_reply = json.dumps(proposal)
        assert 12_000 < len(structured_reply) < 50_000
        store.put(
            "resp_large_owner_history",
            {
                "response": {"id": "resp_large_owner_history"},
                "conversation_history": [
                    {
                        "role": "user",
                        "content": "Add the next private milestone.",
                    },
                    {"role": "assistant", "content": structured_reply},
                ],
            },
        )
        store.set_conversation(
            conversation, "resp_large_owner_history", owner_proposal=True,
        )

        snapshot = store.owner_history_snapshot(conversation)

        assert snapshot["latest_response_id"] == "resp_large_owner_history"
        assert snapshot["data"] == [
            {
                "owner": "Add the next private milestone.",
                "raphael": structured_reply,
            }
        ]

    def test_owner_history_accepts_a_scoped_change_session(self):
        group = "raphael-owner-" + "c" * 32
        conversation = group + "-" + "d" * 32
        store = ResponseStore(max_size=10)
        reply = json.dumps(_owner_existing_proposal(
            request_title="Add owner summaries",
        ))
        store.put("resp_scoped_history", {
            "response": {"id": "resp_scoped_history", "created_at": 200},
            "conversation_history": [
                {"role": "user", "content": "Add a weekly summary."},
                {"role": "assistant", "content": reply},
            ],
        })
        store.set_conversation(
            conversation, "resp_scoped_history", owner_proposal=True,
        )

        assert store.owner_history_snapshot(conversation)["data"] == [{
            "owner": "Add a weekly summary.",
            "raphael": reply,
        }]

    def test_owner_session_index_lists_only_safe_group_history(self):
        group = "raphael-owner-" + "e" * 32
        older = "1" * 32
        newer = "2" * 32
        store = ResponseStore(max_size=20)

        def add_session(name, response_id, created_at, owner, reply):
            store.put(response_id, {
                "response": {"id": response_id, "created_at": created_at},
                "conversation_history": [
                    {"role": "system", "content": "private instructions"},
                    {"role": "user", "content": owner},
                    {"role": "assistant", "content": reply},
                ],
            })
            store.set_conversation(name, response_id)

        add_session(
            group,
            "resp_legacy_session",
            100,
            "Plan the first change.",
            json.dumps({
                "schema_version": 3,
                "kind": "project_change_proposal",
                "mode": "existing",
                "request_title": "First change",
            }),
        )
        add_session(
            f"{group}-{older}",
            "resp_older_session",
            200,
            "Add the owner summary.",
            json.dumps({
                "schema_version": 3,
                "kind": "project_change_proposal",
                "mode": "existing",
                "request_title": "Owner summary",
            }),
        )
        add_session(
            f"{group}-{newer}",
            "resp_newer_session",
            300,
            "Improve the mobile review flow.",
            json.dumps({
                "schema_version": 1,
                "kind": "question",
                "message": "Should the history stay collapsed?",
            }),
        )
        add_session(
            "raphael-owner-" + "f" * 32,
            "resp_other_group",
            400,
            "Do not leak this project.",
            json.dumps({
                "schema_version": 1,
                "kind": "question",
                "message": "Different project",
            }),
        )
        add_session(
            f"{group}-{'4' * 32}",
            "resp_excluded_kind_session",
            450,
            "This one is an Automations request.",
            json.dumps({"schema_version": 1, "kind": "automation_proposal"}),
        )
        add_session(
            f"{group}-{'3' * 32}",
            "resp_unstructured_session",
            500,
            "This turn has no safe final reply.",
            "private intermediate reasoning",
        )

        index = store.owner_session_index("default", group)

        # Ordered by each session's own immutable sequence, newest first. The
        # session whose trailing turn had no readable reply carries that turn
        # with its terminal failure, so the owner's own words are visible
        # instead of hidden; the session whose only reply is a kind this
        # projection deliberately excludes is still RETAINED with a zero turn
        # count rather than dropped, because dropping it made a caller select
        # an older change as if the real current one did not exist.
        assert index == {
            "data": [
                {
                    "session_id": "3" * 32,
                    "updated_at": 500,
                    "preview": "This turn has no safe final reply.",
                    "visible_turn_count": 1,
                    "available": True,
                },
                {
                    "session_id": "4" * 32,
                    "updated_at": 450,
                    "preview": "",
                    "visible_turn_count": 0,
                    "available": True,
                },
                {
                    "session_id": newer,
                    "updated_at": 300,
                    "preview": "Improve the mobile review flow.",
                    "visible_turn_count": 1,
                    "available": True,
                },
                {
                    "session_id": older,
                    "updated_at": 200,
                    "preview": "Add the owner summary.",
                    "visible_turn_count": 1,
                    "available": True,
                },
                {
                    "session_id": "legacy",
                    "updated_at": 100,
                    "preview": "Plan the first change.",
                    "visible_turn_count": 1,
                    "available": True,
                },
            ],
            "truncated": False,
            "current_session_id": "3" * 32,
        }
        serialized = json.dumps(index)
        assert "resp_" not in serialized
        assert "private instructions" not in serialized
        assert "Different project" not in serialized

    def test_owner_proposal_consumption_is_durable_and_exact(self, tmp_path):
        db_path = str(tmp_path / "response-store.db")
        group = "raphael-owner-" + "6" * 32
        response_id = "resp_consumed_proposal"
        store = ResponseStore(max_size=10, db_path=db_path)
        store.put(response_id, {
            "response": {"id": response_id, "created_at": 600},
            "conversation_history": [{
                "role": "assistant",
                "content": json.dumps(_owner_new_proposal()),
            }],
        })
        store.set_conversation(group, response_id, owner_proposal=True)

        assert store.owner_history_snapshot(group)["proposal_consumed"] is False
        assert store.mark_owner_proposal_consumed(
            "default", group, "resp_wrong_proposal",
        ) is False
        assert store.mark_owner_proposal_consumed("default", group, response_id) is True
        assert store.owner_history_snapshot(group)["proposal_consumed"] is True

        store.set_conversation(group, response_id)
        assert store.owner_history_snapshot(group)["proposal_consumed"] is True
        store.close()

        reopened = ResponseStore(max_size=10, db_path=db_path)
        assert reopened.owner_history_snapshot(group)["proposal_consumed"] is True
        reopened.put("resp_newer_proposal", {
            "response": {"id": "resp_newer_proposal", "created_at": 700},
            "conversation_history": [],
        })
        reopened.set_conversation(group, "resp_newer_proposal")
        assert reopened.owner_history_snapshot(group)["proposal_consumed"] is False
        reopened.close()

    def test_owner_proposal_claim_is_atomic_durable_and_exact(self, tmp_path):
        db_path = str(tmp_path / "response-store.db")
        conversation = "raphael-owner-" + "a" * 32 + "-" + "b" * 32
        response_id = "resp_claimed_proposal"
        claim_id = "claim_" + "c" * 32
        run_id = "run_" + "d" * 32
        store = ResponseStore(max_size=10, db_path=db_path)
        store.put(response_id, {
            "response": {"id": response_id, "created_at": 900},
            "conversation_history": [
                {"role": "user", "content": "Build the approved milestone."},
                {"role": "assistant", "content": json.dumps(
                    _owner_existing_proposal(
                        request_title="Build the approved milestone",
                    )
                )},
            ],
        })
        assert store.set_conversation(
            conversation, response_id, owner_proposal=True,
        ) is True

        assert store.claim_owner_proposal("default", conversation, response_id, claim_id) is True
        assert store.claim_owner_proposal("default", conversation, response_id, claim_id) is True
        assert store.claim_owner_proposal(
            "default", conversation, response_id, "claim_" + "e" * 32,
        ) is False
        assert store.set_conversation(conversation, "resp_newer_proposal") is False
        assert store.close_owner_conversation("default", conversation, response_id) is False
        assert store.attach_owner_run("default", conversation, response_id, claim_id, run_id) is True
        assert store.complete_owner_claim(
            "default", conversation, response_id, claim_id, "run_" + "f" * 32,
        ) is False
        assert store.complete_owner_claim(
            "default", conversation, response_id, claim_id, run_id,
        ) is True
        assert store.complete_owner_claim(
            "default", conversation, response_id, claim_id, run_id,
        ) is True
        snapshot = store.owner_history_snapshot(conversation)
        assert snapshot["proposal_consumed"] is True
        assert snapshot["proposal_claimed"] is False
        assert snapshot["active_run_id"] is None
        store.close()

        reopened = ResponseStore(max_size=10, db_path=db_path)
        assert reopened.owner_history_snapshot(conversation)["proposal_consumed"] is True
        reopened.close()

    def test_unattached_owner_claim_can_be_abandoned_without_releasing_a_run(self):
        conversation = "raphael-owner-" + "9" * 32
        response_id = "resp_unattached_claim"
        claim_id = "claim_" + "8" * 32
        run_id = "run_" + "7" * 32
        store = ResponseStore(max_size=10)
        store.put(response_id, {
            "response": {"id": response_id, "created_at": 950},
            "conversation_history": [{
                "role": "assistant",
                "content": json.dumps(_owner_new_proposal()),
            }],
        })
        assert store.set_conversation(
            conversation, response_id, owner_proposal=True,
        ) is True
        assert store.claim_owner_proposal(
            "default", conversation, response_id, claim_id,
        ) is True
        assert store.abandon_unattached_owner_claim(
            "default", conversation, response_id, claim_id,
        ) is True
        assert store.owner_history_snapshot(conversation)["proposal_claimed"] is False

        assert store.claim_owner_proposal(
            "default", conversation, response_id, claim_id,
        ) is True
        assert store.attach_owner_run(
            "default", conversation, response_id, claim_id, run_id,
        ) is True
        assert store.abandon_unattached_owner_claim(
            "default", conversation, response_id, claim_id,
        ) is False
        assert store.owner_run_is_attached(
            "default", conversation, response_id, claim_id, run_id,
        ) is True
        store.close()

    def test_owner_proposal_release_and_close_fence_new_responses(self):
        conversation = "raphael-owner-" + "1" * 32
        response_id = "resp_releasable_proposal"
        claim_id = "claim_" + "2" * 32
        run_id = "run_" + "3" * 32
        store = ResponseStore(max_size=10)
        store.put(response_id, {
            "response": {"id": response_id, "created_at": 1_000},
            "conversation_history": [{
                "role": "assistant",
                "content": json.dumps(_owner_new_proposal()),
            }],
        })
        assert store.set_conversation(
            conversation, response_id, owner_proposal=True,
        ) is True
        assert store.claim_owner_proposal("default", conversation, response_id, claim_id) is True
        assert store.attach_owner_run("default", conversation, response_id, claim_id, run_id) is True
        assert store.release_owner_claim("default", conversation, response_id, claim_id, run_id) is True
        assert store.close_owner_conversation("default", conversation, response_id) is True
        assert store.close_owner_conversation("default", conversation, response_id) is True
        assert store.set_conversation(conversation, "resp_after_close") is False
        snapshot = store.owner_history_snapshot(conversation)
        assert snapshot["proposal_consumed"] is False
        assert snapshot["conversation_closed"] is True
        assert snapshot["proposal_claimed"] is False
        assert snapshot["conversation_closed"] is True

    def test_close_missing_owner_conversation_persists_a_tombstone(self, tmp_path):
        """Clear wins even when an in-flight response has not been stored yet."""
        db_path = str(tmp_path / "response-store.db")
        conversation = "raphael-owner-" + "7" * 32 + "-" + "8" * 32
        store = ResponseStore(max_size=10, db_path=db_path)

        assert store.close_owner_conversation("default", conversation, None) is True
        assert store.owner_history_snapshot(conversation)["conversation_closed"] is True
        store.close()

        reopened = ResponseStore(max_size=10, db_path=db_path)
        reopened.put("resp_late_background", {
            "response": {"id": "resp_late_background", "created_at": 1},
            "conversation_history": [],
        })
        assert reopened.set_conversation(
            conversation, "resp_late_background",
        ) is False
        assert reopened.owner_history_snapshot(conversation)["conversation_closed"] is True
        reopened.close()

    def test_latest_question_cannot_inherit_an_older_proposal_handle(self):
        conversation = "raphael-owner-" + "9" * 32
        proposal_response = "resp_actionable_proposal"
        question_response = "resp_followup_question"
        proposal = json.dumps(_owner_existing_proposal(
            request_title="Add a weekly summary",
        ))
        question = json.dumps({
            "schema_version": 1,
            "kind": "question",
            "message": "Which day should it arrive?",
        })
        store = ResponseStore(max_size=10)
        store.put(proposal_response, {
            "response": {"id": proposal_response, "created_at": 1},
            "conversation_history": [
                {"role": "user", "content": "Add a weekly summary."},
                {"role": "assistant", "content": proposal},
            ],
        })
        assert store.set_conversation(
            conversation, proposal_response, owner_proposal=True,
        ) is True
        assert store.mark_owner_proposal_consumed(
            "default", conversation, proposal_response,
        )

        store.put(question_response, {
            "response": {"id": question_response, "created_at": 2},
            "conversation_history": [
                {"role": "user", "content": "Add a weekly summary."},
                {"role": "assistant", "content": proposal},
                {"role": "user", "content": "Change the delivery time."},
                {"role": "assistant", "content": question},
            ],
        })
        assert store.set_conversation(
            conversation, question_response, owner_proposal=False,
        ) is True

        snapshot = store.owner_history_snapshot(conversation)
        assert snapshot["latest_response_id"] is None
        assert snapshot["proposal_consumed"] is False
        assert snapshot["proposal_claimed"] is False
        assert store.claim_owner_proposal(
            "default", conversation, proposal_response, "claim_" + "a" * 32,
        ) is False
        consumed = store._conn.execute(
            "SELECT consumed_response_id FROM conversations WHERE name = ?",
            (conversation,),
        ).fetchone()
        assert consumed[0] == proposal_response

    def test_owner_state_isolated_by_profile_even_with_same_ids(self):
        conversation = "raphael-owner-" + "4" * 32
        response_id = "resp_shared_profile_id"
        store = ResponseStore(max_size=10, default_profile="default")
        for profile, owner_text in (
            ("default", "Default owner request."),
            ("secondary", "Secondary owner request."),
        ):
            store.put(response_id, {
                "response": {"id": response_id, "created_at": 1},
                "conversation_history": [
                    {"role": "user", "content": owner_text},
                    {"role": "assistant", "content": json.dumps(
                        _owner_new_proposal()
                    )},
                ],
            }, profile=profile)
            assert store.set_conversation(
                conversation,
                response_id,
                owner_proposal=True,
                profile=profile,
            ) is True

        assert store.get(response_id, profile="default")["conversation_history"][0][
            "content"
        ] == "Default owner request."
        assert store.get(response_id, profile="secondary")["conversation_history"][0][
            "content"
        ] == "Secondary owner request."
        assert store.claim_owner_proposal(
            "default",
            conversation,
            response_id,
            "claim_" + "5" * 32,
        ) is True
        assert store.owner_history_snapshot(
            conversation, profile="default",
        )["proposal_claimed"] is True
        assert store.owner_history_snapshot(
            conversation, profile="secondary",
        )["proposal_claimed"] is False

    def test_a_reserved_owner_turn_projects_the_request_it_is_still_planning(self):
        """Item 32TK: the owner's own words survive a lost accept response.

        The reservation is taken before any model runs, so the request it is
        planning is durable from that moment. Projecting it is what lets a
        refreshed page show the message the owner actually sent — and recover
        the SAME response id rather than planning their words a second time.
        """
        conversation = "raphael-owner-" + "b" * 32
        store = ResponseStore(max_size=10)
        try:
            # Nothing has been published on this conversation yet: the very
            # first turn of a new Project is exactly the case that lost the
            # owner's request.
            assert store.reserve_owner_conversation(
                "default",
                conversation,
                "resp_pending_first_turn",
                owner_message="Prepare a private 60-minute workshop",
            ) is True

            snapshot = store.owner_history_snapshot(conversation)

            assert snapshot["data"] == []
            assert snapshot["pending"] == {
                "owner": "Prepare a private 60-minute workshop",
                "response_id": "resp_pending_first_turn",
            }

            # Publishing the turn releases the reservation, and the owner's
            # words are then carried by the durable turn itself.
            store.put("resp_pending_first_turn", {
                "response": {"id": "resp_pending_first_turn", "created_at": 1},
                "conversation_history": [
                    {
                        "role": "user",
                        "content": "Prepare a private 60-minute workshop",
                    },
                    {
                        "role": "assistant",
                        "content": json.dumps({
                            "schema_version": 1,
                            "kind": "question",
                            "message": "Who is the workshop for?",
                        }),
                    },
                ],
            })
            assert store.set_conversation(
                conversation,
                "resp_pending_first_turn",
                profile="default",
                reservation_id="resp_pending_first_turn",
            ) is True

            published = store.owner_history_snapshot(conversation)
            assert published["pending"] is None
            assert published["data"] == [{
                "owner": "Prepare a private 60-minute workshop",
                "raphael": ANY,
            }]
        finally:
            store.close()

    def test_a_pending_owner_turn_is_projected_on_a_conversation_with_turns(self):
        """A follow-up message is recoverable while its plan is still running."""
        conversation = "raphael-owner-" + "c" * 32
        store = ResponseStore(max_size=10)
        try:
            store.put("resp_answered_turn", {
                "response": {"id": "resp_answered_turn", "created_at": 1},
                "conversation_history": [
                    {"role": "user", "content": "Add a weekly report"},
                    {
                        "role": "assistant",
                        "content": json.dumps({
                            "schema_version": 1,
                            "kind": "question",
                            "message": "Who should receive it?",
                        }),
                    },
                ],
            })
            assert store.set_conversation(
                conversation, "resp_answered_turn", profile="default",
            ) is True
            assert store.reserve_owner_conversation(
                "default",
                conversation,
                "resp_pending_follow_up",
                owner_message="Send it to me every Friday",
            ) is True

            snapshot = store.owner_history_snapshot(conversation)

            assert len(snapshot["data"]) == 1
            assert snapshot["pending"] == {
                "owner": "Send it to me every Friday",
                "response_id": "resp_pending_follow_up",
            }

            # An expired reservation is not a pending request: nothing is
            # planning it any more, so projecting it would tell the owner their
            # words are still being worked on when they are not.
            store._conn.execute(
                "UPDATE owner_conversation_reservations SET expires_at = ? "
                "WHERE profile = ? AND name = ?",
                (time.time() - 1, "default", conversation),
            )
            store._conn.commit()
            assert store.owner_history_snapshot(conversation)["pending"] is None
        finally:
            store.close()

    def test_a_terminal_failure_survives_the_reservation_it_releases(self):
        """Item 32TK: the fence is dropped, the owner's request is not.

        A turn that fails releases its reservation, so the pending projection
        goes with it. Without this record a browser that never received the
        accept response has nothing left to find: no words, and no failure
        either. The record replaces the fence in ONE write, so a caller can
        still reach the same response id and read the same terminal outcome.
        """
        conversation = "raphael-owner-" + "e" * 32
        store = ResponseStore(max_size=10)
        try:
            assert store.reserve_owner_conversation(
                "default",
                conversation,
                "resp_failed_first_turn",
                owner_message="Prepare a private 60-minute workshop",
            ) is True

            _interrupt_owner_turn(store, conversation, "resp_failed_first_turn")

            snapshot = store.owner_history_snapshot(conversation)
            # Nothing is being planned any more...
            assert snapshot["pending"] is None
            # ...and the request that was is still recoverable.
            assert snapshot["recovery"] == {
                "owner": "Prepare a private 60-minute workshop",
                "response_id": "resp_failed_first_turn",
            }
            # It is a way back to a decided outcome, never an approvable turn.
            assert snapshot["head_response_id"] is None
            assert snapshot["latest_response_id"] is None
            assert snapshot["proposal_claimed"] is False
            assert snapshot["data"] == []
        finally:
            store.close()

    def test_only_a_live_fence_can_become_a_recovery_record(self):
        """A published turn is never restated as an unfinished one.

        The reservation is released inside the transaction that publishes the
        turn, so a failure handler arriving afterwards finds no fence to
        convert and writes nothing. That is what keeps this record from
        becoming a second, contradicting account of a turn that really did
        complete.
        """
        conversation = "raphael-owner-" + "f" * 32
        store = ResponseStore(max_size=10)
        try:
            assert store.reserve_owner_conversation(
                "default", conversation, "resp_published_turn",
                owner_message="Prepare a private 60-minute workshop",
            ) is True
            store.put("resp_published_turn", {
                "response": {"id": "resp_published_turn", "created_at": 1},
                "conversation_history": [
                    {
                        "role": "user",
                        "content": "Prepare a private 60-minute workshop",
                    },
                    {
                        "role": "assistant",
                        "content": json.dumps({
                            "schema_version": 1,
                            "kind": "question",
                            "message": "Who is the workshop for?",
                        }),
                    },
                ],
            })
            assert store.set_conversation(
                conversation, "resp_published_turn", profile="default",
                reservation_id="resp_published_turn",
            ) is True

            _interrupt_owner_turn(store, conversation, "resp_published_turn")
            assert store.owner_history_snapshot(conversation)["recovery"] is None
        finally:
            store.close()

    def test_a_recovery_record_is_retired_only_by_an_acknowledgement(self):
        """Reading it can never be what erases it.

        The answer carrying it can be lost exactly like the one that started
        all this, so it stands until a caller says it has the words and the
        outcome. Saying so twice is the same as saying it once.
        """
        conversation = "raphael-owner-" + "1" * 32
        store = ResponseStore(max_size=10)
        try:
            assert store.reserve_owner_conversation(
                "default", conversation, "resp_unacknowledged",
                owner_message="Prepare a private 60-minute workshop",
            ) is True
            _interrupt_owner_turn(store, conversation, "resp_unacknowledged")

            # Read as often as a browser needs to; it is still there.
            for _ in range(3):
                assert store.owner_history_snapshot(conversation)["recovery"] == {
                    "owner": "Prepare a private 60-minute workshop",
                    "response_id": "resp_unacknowledged",
                }

            # A different response id is not this record and does not retire it.
            assert store.acknowledge_owner_conversation_recovery(
                "default", conversation, "resp_someone_elses",
            ) == "mismatch"
            assert store.owner_history_snapshot(conversation)["recovery"] is not None

            assert store.acknowledge_owner_conversation_recovery(
                "default", conversation, "resp_unacknowledged",
            ) == "retired"
            assert store.owner_history_snapshot(conversation)["recovery"] is None
            # Idempotent: a repeated acknowledgement is not an error.
            assert store.acknowledge_owner_conversation_recovery(
                "default", conversation, "resp_unacknowledged",
            ) == "absent"
            assert store.owner_history_snapshot(conversation)["recovery"] is None
        finally:
            store.close()

    def test_an_acknowledged_recovery_becomes_a_durable_failure_turn(
        self, tmp_path,
    ):
        """Reload keeps the request and closes every stale approval path.

        A browser acknowledges only after it holds the interrupted request and
        its terminal outcome.  That acknowledgement used to delete the sole
        recovery record, so the turn disappeared on reload and the older
        proposal underneath it could become approvable again.  The exact
        interrupted response must instead become the durable, non-actionable
        conversation head in the same transaction that retires recovery.
        """
        database = str(tmp_path / "response_store.db")
        conversation = "raphael-owner-" + "a" * 32
        proposal_response_id = "resp_older_proposal"
        failed_response_id = "resp_failed_revision"
        failure_message = (
            "Raphael could not prepare a safe plan for this request. "
            "Nothing was changed. You can send it again."
        )
        store = ResponseStore(max_size=10, db_path=database)
        try:
            store.put(proposal_response_id, {
                "response": {"id": proposal_response_id, "created_at": 1},
                "conversation_history": [
                    {"role": "user", "content": "Add a weekly report"},
                    {
                        "role": "assistant",
                        "content": json.dumps(_owner_existing_proposal()),
                    },
                ],
            })
            assert store.set_conversation(
                conversation, proposal_response_id, owner_proposal=True,
            ) is True
            assert store.reserve_owner_conversation(
                "default",
                conversation,
                failed_response_id,
                expected_previous_response_id=proposal_response_id,
                owner_message="Make the report easier to understand",
            ) is True
            _interrupt_owner_turn(store, conversation, failed_response_id)

            assert store.acknowledge_owner_conversation_recovery(
                "default", conversation, failed_response_id,
            ) == "retired"
            snapshot = store.owner_history_snapshot(conversation)
            assert snapshot["recovery"] is None
            assert snapshot["head_response_id"] == failed_response_id
            assert snapshot["latest_response_id"] is None
            assert snapshot["proposal_consumed"] is False
            assert snapshot["proposal_claimed"] is False
            assert snapshot["incomplete"] is False
            assert [turn["owner"] for turn in snapshot["data"]] == [
                "Add a weekly report",
                "Make the report easier to understand",
            ]
            assert json.loads(snapshot["data"][-1]["raphael"]) == {
                "schema_version": 1,
                "kind": "failure",
                "message": failure_message,
            }
            # The later failed request superseded the older proposal.  A stale
            # tab cannot approve it after the recovery acknowledgement.
            assert store.claim_owner_proposal(
                "default",
                conversation,
                proposal_response_id,
                "claim_" + "b" * 32,
            ) is False
        finally:
            store.close()

        store = ResponseStore(max_size=10, db_path=database)
        try:
            snapshot = store.owner_history_snapshot(conversation)
            assert snapshot["head_response_id"] == failed_response_id
            assert snapshot["recovery"] is None
            assert json.loads(snapshot["data"][-1]["raphael"])["kind"] == "failure"
            # Retrying the acknowledgement neither errors nor duplicates the
            # durable turn.
            assert store.acknowledge_owner_conversation_recovery(
                "default", conversation, failed_response_id,
            ) == "absent"
            assert len(store.owner_history_snapshot(conversation)["data"]) == 2
        finally:
            store.close()

    def test_a_recovery_record_survives_restart_and_any_age(self, tmp_path):
        """Only an acknowledgement retires it — never a restart, never age.

        The Founder's request is the thing this record carries. An owner who
        comes back next week, or after the gateway was restarted, still gets
        told what became of what they sent, because nothing else can tell them:
        the browser holds no handle, and no turn was ever published.
        """
        database = str(tmp_path / "response_store.db")
        store = ResponseStore(max_size=10, db_path=database)
        conversation = "raphael-owner-" + "2" * 32
        try:
            assert store.reserve_owner_conversation(
                "default", conversation, "resp_first_attempt",
                owner_message="Prepare a private 60-minute workshop",
            ) is True
            _interrupt_owner_turn(store, conversation, "resp_first_attempt")
        finally:
            store.close()

        # The gateway restarts. The record is durable state, so it is still
        # there — and no legacy expiry column is consulted to decide that.
        store = ResponseStore(max_size=10, db_path=database)
        try:
            assert store.owner_history_snapshot(conversation)["recovery"] == {
                "owner": "Prepare a private 60-minute workshop",
                "response_id": "resp_first_attempt",
            }

            # An arbitrarily old record, written by a build that stamped a
            # bounded lease onto it, is still the owner's unanswered request.
            store._conn.execute(
                "UPDATE owner_conversation_recovery SET created_at = ?, "
                "expires_at = ? WHERE profile = ? AND name = ?",
                (0.0, 1.0, "default", conversation),
            )
            store._conn.commit()
            assert store.owner_history_snapshot(conversation)["recovery"] == {
                "owner": "Prepare a private 60-minute workshop",
                "response_id": "resp_first_attempt",
            }

            # Acknowledging it is still the one and only way it ends.
            assert store.acknowledge_owner_conversation_recovery(
                "default", conversation, "resp_first_attempt",
            ) == "retired"
            assert store.owner_history_snapshot(conversation)["recovery"] is None
        finally:
            store.close()

    def test_a_new_turn_is_refused_while_a_recovery_is_unacknowledged(self):
        """A newer request may not bury the one that was never answered.

        Superseding it deleted the only account of an interrupted request the
        moment a second browser sent anything, so the Founder's words were lost
        by another tab simply being used. The fence refuses instead, and only an
        acknowledgement opens the conversation again.
        """
        conversation = "raphael-owner-" + "2" * 32
        store = ResponseStore(max_size=10)
        try:
            assert store.reserve_owner_conversation(
                "default", conversation, "resp_first_attempt",
                owner_message="Prepare a private 60-minute workshop",
            ) is True
            _interrupt_owner_turn(store, conversation, "resp_first_attempt")

            # A different message, from a browser that knows nothing about the
            # first: refused, and the record it would have replaced still reads.
            assert store.reserve_owner_conversation(
                "default", conversation, "resp_second_attempt",
                owner_message="Prepare a private 90-minute workshop",
            ) is False
            snapshot = store.owner_history_snapshot(conversation)
            assert snapshot["pending"] is None
            assert snapshot["recovery"] == {
                "owner": "Prepare a private 60-minute workshop",
                "response_id": "resp_first_attempt",
            }
            # Not even the interrupted turn's own id may take the fence back:
            # that turn is over, and its outcome is what there is to read.
            assert store.reserve_owner_conversation(
                "default", conversation, "resp_first_attempt",
                owner_message="Prepare a private 60-minute workshop",
            ) is False

            assert store.acknowledge_owner_conversation_recovery(
                "default", conversation, "resp_first_attempt",
            ) == "retired"
            assert store.reserve_owner_conversation(
                "default", conversation, "resp_second_attempt",
                owner_message="Prepare a private 90-minute workshop",
            ) is True
            snapshot = store.owner_history_snapshot(conversation)
            assert snapshot["recovery"] is None
            assert snapshot["pending"] == {
                "owner": "Prepare a private 90-minute workshop",
                "response_id": "resp_second_attempt",
            }
        finally:
            store.close()

    def test_a_terminal_ending_without_a_fence_leaves_the_work_recoverable(self):
        """Item 32TK round 3: no turn and no record is not an ending.

        The conversion is what turns this turn's fence into the account of what
        the owner sent. Ignoring a conversion that found nothing let the
        terminal body, the replay record and the job retirement all land while
        the request itself simply vanished: no published turn, no recovery, and
        nothing left saying anybody must finish it. The ending is refused
        instead, and everything it would have written rolls back together.
        """
        conversation = "raphael-owner-" + "7" * 32
        store = ResponseStore(max_size=10)
        try:
            store.reserve_owner_job(
                "response", "resp_no_fence_left", "default",
                {"conversation": conversation},
            )
            with pytest.raises(OwnerTurnNotRecoverable):
                _interrupt_owner_turn(store, conversation, "resp_no_fence_left")

            # Nothing moved: no terminal body, no recovery, and the job row
            # still says somebody must finish this.
            assert store.get("resp_no_fence_left") is None
            snapshot = store.owner_history_snapshot(conversation)
            assert snapshot["pending"] is None
            assert snapshot["recovery"] is None
            assert store._conn.execute(
                "SELECT COUNT(*) FROM owner_executor_jobs WHERE job_key = ?",
                ("resp_no_fence_left",),
            ).fetchone()[0] == 1
        finally:
            store.close()

    def test_an_expired_fence_is_kept_while_its_work_is_unresolved(self):
        """Item 32TK round 3: expiry alone may not destroy the way back.

        A lease that runs out says the executor stopped renewing it, not that
        the work it fenced is finished. While the job row still says somebody
        must finish this response, the fence stays: every other caller is
        refused by it, and the ending that eventually arrives still has
        something to convert into the record carrying the owner's request.
        """
        conversation = "raphael-owner-" + "8" * 32
        store = ResponseStore(max_size=10)
        try:
            assert store.reserve_owner_conversation(
                "default", conversation, "resp_expired_but_queued",
                owner_message="Prepare a private 60-minute workshop",
            ) is True
            store.reserve_owner_job(
                "response", "resp_expired_but_queued", "default",
                {"conversation": conversation},
            )
            store._conn.execute(
                "UPDATE owner_conversation_reservations SET expires_at = ? "
                "WHERE profile = ? AND name = ?",
                (time.time() - 1, "default", conversation),
            )
            store._conn.commit()

            # Every fence-reading caller still sees it, so nothing takes this
            # conversation over and nothing closes it out from under the work.
            assert store.close_owner_conversation(
                "default", conversation, None,
            ) is False
            assert store.reserve_owner_conversation(
                "default", conversation, "resp_competing_turn",
            ) is False

            # And the ending, when it comes, still has a fence to convert.
            _interrupt_owner_turn(store, conversation, "resp_expired_but_queued")
            assert store.owner_history_snapshot(conversation)["recovery"] == {
                "owner": "Prepare a private 60-minute workshop",
                "response_id": "resp_expired_but_queued",
            }
        finally:
            store.close()

    def test_a_conversation_with_an_unanswered_request_cannot_be_closed(self):
        """Item 32TK round 3: closing is not an answer either.

        The fence is released the moment a plan ends, so a conversation whose
        last request ended without a turn has no live reservation to refuse a
        close. Closing it there orphaned the durable record: the owner's
        request stayed unanswered forever on a conversation nothing would ever
        read again. Only an acknowledgement opens this.
        """
        conversation = "raphael-owner-" + "9" * 32
        store = ResponseStore(max_size=10)
        try:
            assert store.reserve_owner_conversation(
                "default", conversation, "resp_unanswered",
                owner_message="Prepare a private 60-minute workshop",
            ) is True
            _interrupt_owner_turn(store, conversation, "resp_unanswered")

            assert store.close_owner_conversation(
                "default", conversation, None,
            ) is False
            assert store.owner_history_snapshot(
                conversation,
            )["conversation_closed"] is False

            assert store.acknowledge_owner_conversation_recovery(
                "default", conversation, "resp_unanswered",
            ) == "retired"
            assert store.close_owner_conversation(
                "default", conversation, None,
            ) is True
        finally:
            store.close()

    def test_an_unanswered_request_refuses_both_owner_proposal_claims(self):
        """Item 32TK, finding 2: a claim is not an answer either.

        The fence is released the moment a plan ends, so a conversation
        carrying a request that ended without a turn has no live reservation.
        Both claim transactions refused a live reservation and nothing else, so
        an OLDER proposal on that conversation could still be claimed and run
        while the later request stood unanswered behind it — and the run that
        followed consumed the proposal, which is not something an
        acknowledgement can undo. Acknowledgement therefore seals the later
        failure as the new head instead of reopening the older proposal.
        """
        conversation = "raphael-owner-" + "5" * 32
        response_id = "resp_older_proposal"
        claim_id = "claim_" + "6" * 32
        run_id = "run_" + "7" * 32
        store = ResponseStore(max_size=10)
        try:
            store.put(response_id, {
                "response": {"id": response_id, "created_at": 1_000},
                "conversation_history": [{
                    "role": "assistant",
                    "content": json.dumps(_owner_new_proposal()),
                }],
            })
            assert store.set_conversation(
                conversation, response_id, owner_proposal=True,
            ) is True

            # A LATER request, sent after that proposal, that ended without
            # ever becoming a turn. Its fence is already gone.
            assert store.reserve_owner_conversation(
                "default", conversation, "resp_later_request",
                owner_message="Prepare a private 60-minute workshop",
            ) is True
            _interrupt_owner_turn(store, conversation, "resp_later_request")
            # Nothing is fencing this conversation any more: that is exactly
            # the state both claims used to walk straight through.
            assert store.owner_history_snapshot(conversation)["pending"] is None

            assert store.claim_owner_proposal(
                "default", conversation, response_id, claim_id,
            ) is False
            assert store.claim_and_attach_owner_run(
                "default", conversation, response_id, claim_id, run_id,
                operation="owner_task_graph_commit",
                payload_digest="a" * 64,
            ) is False
            # Nothing was claimed, nothing was bound, nothing was consumed.
            snapshot = store.owner_history_snapshot(conversation)
            assert snapshot["proposal_claimed"] is False
            assert snapshot["active_run_id"] is None
            assert snapshot["proposal_consumed"] is False
            assert snapshot["recovery"] == {
                "owner": "Prepare a private 60-minute workshop",
                "response_id": "resp_later_request",
            }

            assert store.acknowledge_owner_conversation_recovery(
                "default", conversation, "resp_later_request",
            ) == "retired"
            assert store.claim_and_attach_owner_run(
                "default", conversation, response_id, claim_id, run_id,
                operation="owner_task_graph_commit",
                payload_digest="a" * 64,
            ) is False
            sealed = store.owner_history_snapshot(conversation)
            assert sealed["head_response_id"] == "resp_later_request"
            assert sealed["latest_response_id"] is None
            assert json.loads(sealed["data"][-1]["raphael"])["kind"] == "failure"
        finally:
            store.close()

    @staticmethod
    def _owner_run_binding(store, conversation, response_id):
        """One approvable proposal, and the exact owner authority a run binds.

        The same dictionary ``/v1/runs`` passes to
        :meth:`ResponseStore.reserve_run_idempotency` once it has validated the
        request, so these tests reserve through the production entry point
        rather than restating what it writes.
        """
        store.put(response_id, {
            "response": {"id": response_id, "created_at": 1_000},
            "conversation_history": [
                {"role": "user", "content": "Prepare the workshop"},
                {
                    "role": "assistant",
                    "content": json.dumps(_owner_new_proposal()),
                },
            ],
        })
        assert store.set_conversation(
            conversation, response_id, owner_proposal=True,
        ) is True
        return {
            "proposal_profile": "default",
            "conversation": conversation,
            "response_id": response_id,
            "claim_id": "claim_" + "3" * 32,
            "operation": "owner_task_graph_commit",
            "payload_digest": "b" * 64,
        }

    @staticmethod
    def _run_rows(store, run_id):
        """The durable traces a reserved run leaves: its key's row, and its job.

        The row is read by the one idempotency key these tests reserve under,
        so it reports which run that key currently names — nothing, the run
        that was already there, or a newly bound one.
        """
        return {
            "idempotency": store._conn.execute(
                "SELECT run_id FROM run_idempotency "
                "WHERE profile = ? AND session_scope = ? AND idempotency_key = ?",
                ("default", "scope", "commit-key"),
            ).fetchone(),
            "job": store._conn.execute(
                "SELECT 1 FROM owner_executor_jobs WHERE kind = 'run' AND job_key = ?",
                (run_id,),
            ).fetchone(),
        }

    def test_an_unanswered_request_refuses_a_fresh_run_reservation(self):
        """Item 32TK, finding 2: reserving a run is not an answer either.

        The reservation a plan holds is released the moment that plan ENDS, so
        a conversation carrying a request that ended without a turn looks
        completely idle to this transaction. Checking only that reservation let
        a direct ``/v1/runs`` call claim the OLDER proposal, bind a run to it
        and queue that run's executor job while the later request stood
        unanswered behind it — and the run that follows consumes the proposal,
        which is not something an acknowledgement can undo.
        """
        conversation = "raphael-owner-" + "c" * 32
        response_id = "resp_older_run_proposal"
        run_id = "run_" + "1" * 32
        store = ResponseStore(max_size=10)
        try:
            owner = self._owner_run_binding(store, conversation, response_id)
            head = store.owner_history_snapshot(conversation)["head_response_id"]

            # A LATER request, sent after that proposal, that ended without
            # ever becoming a turn. Its fence is already gone.
            assert store.reserve_owner_conversation(
                "default", conversation, "resp_later_request",
                owner_message="Prepare a private 60-minute workshop",
            ) is True
            _interrupt_owner_turn(store, conversation, "resp_later_request")
            assert store.owner_history_snapshot(conversation)["pending"] is None

            assert store.reserve_run_idempotency(
                "default", "scope", "commit-key", "fingerprint", run_id,
                owner=owner, job_payload={"owner": dict(owner)},
            ) == ("authority_conflict", None)

            # Nothing was claimed, bound, recorded or queued, and the
            # conversation still ends exactly where it did.
            snapshot = store.owner_history_snapshot(conversation)
            assert snapshot["proposal_claimed"] is False
            assert snapshot["active_run_id"] is None
            assert snapshot["proposal_consumed"] is False
            assert snapshot["head_response_id"] == head
            assert [turn["owner"] for turn in snapshot["data"]] == [
                "Prepare the workshop",
            ]
            assert snapshot["recovery"] == {
                "owner": "Prepare a private 60-minute workshop",
                "response_id": "resp_later_request",
            }
            assert self._run_rows(store, run_id) == {"idempotency": None, "job": None}

            # Acknowledgement seals the later failure. The older proposal is
            # superseded rather than quietly reactivated behind it.
            assert store.acknowledge_owner_conversation_recovery(
                "default", conversation, "resp_later_request",
            ) == "retired"
            assert store.reserve_run_idempotency(
                "default", "scope", "commit-key", "fingerprint", run_id,
                owner=owner, job_payload={"owner": dict(owner)},
            ) == ("authority_conflict", None)
            sealed = store.owner_history_snapshot(conversation)
            assert sealed["head_response_id"] == "resp_later_request"
            assert sealed["active_run_id"] is None
            assert self._run_rows(store, run_id) == {
                "idempotency": None, "job": None,
            }
        finally:
            store.close()

    def test_an_unanswered_request_refuses_a_released_run_retry(self):
        """Item 32TK, finding 2: the released-retry branch is the same door.

        A run that failed releases its claim, and an exact retry of the same
        approval re-binds that released claim to a new run. This branch checked
        the live reservation too, so a retry arriving while a LATER request
        stood unanswered rebound the older proposal, rewrote the run its
        idempotency key names and queued a second executor job — every one of
        them a mutation made on a conversation nobody had answered for.
        """
        conversation = "raphael-owner-" + "e" * 32
        response_id = "resp_released_run_proposal"
        first_run = "run_" + "2" * 32
        retry_run = "run_" + "4" * 32
        store = ResponseStore(max_size=10)
        try:
            owner = self._owner_run_binding(store, conversation, response_id)
            assert store.reserve_run_idempotency(
                "default", "scope", "commit-key", "fingerprint", first_run,
                owner=owner, job_payload={"owner": dict(owner)},
            ) == ("new", first_run)
            assert store.release_owner_claim(
                "default", conversation, response_id,
                owner["claim_id"], first_run,
            ) is True
            head = store.owner_history_snapshot(conversation)["head_response_id"]

            # ...and only then does a later request end without a turn.
            assert store.reserve_owner_conversation(
                "default", conversation, "resp_later_request",
                owner_message="Prepare a private 60-minute workshop",
            ) is True
            _interrupt_owner_turn(store, conversation, "resp_later_request")

            assert store.reserve_run_idempotency(
                "default", "scope", "commit-key", "fingerprint", retry_run,
                owner=owner, job_payload={"owner": dict(owner)},
            ) == ("authority_conflict", None)

            # The released claim is still released, still bound to the run that
            # failed, and nothing was written for the retry.
            assert store._conn.execute(
                "SELECT claim_state, owner_run_id FROM conversations "
                "WHERE profile = ? AND name = ?",
                ("default", conversation),
            ).fetchone() == ("released", first_run)
            snapshot = store.owner_history_snapshot(conversation)
            assert snapshot["proposal_claimed"] is False
            assert snapshot["active_run_id"] is None
            assert snapshot["proposal_consumed"] is False
            assert snapshot["head_response_id"] == head
            assert [turn["owner"] for turn in snapshot["data"]] == [
                "Prepare the workshop",
            ]
            assert self._run_rows(store, retry_run) == {
                "idempotency": (first_run,), "job": None,
            }

            # Acknowledgement seals the later failure, so the released older
            # approval is never rebound behind it. Its original idempotency
            # result remains readable without creating new work.
            assert store.acknowledge_owner_conversation_recovery(
                "default", conversation, "resp_later_request",
            ) == "retired"
            assert store.reserve_run_idempotency(
                "default", "scope", "commit-key", "fingerprint", retry_run,
                owner=owner, job_payload={"owner": dict(owner)},
            ) == ("existing", first_run)
            assert store.owner_history_snapshot(conversation)[
                "head_response_id"
            ] == "resp_later_request"
            assert self._run_rows(store, retry_run) == {
                "idempotency": (first_run,), "job": None,
            }
        finally:
            store.close()

    def test_the_draft_that_created_a_project_takes_no_further_turn(self):
        """Item 32TK, finding 1: a spent New Project draft is terminal.

        Its proposal is consumed, the run that consumed it is bound to it, and
        the receipt naming the Project the owner is about to be sent to is
        replayed from exactly that pair. A turn published here takes the head
        that pair is read from, so the receipt, the redirect and the visible
        conversation all become unreachable while the Project sits there. The
        Workspace refuses this, and so does Raphael: only the acknowledgement
        closes this draft, and the group moves on to a successor in the same
        write.
        """
        group = "raphael-owner-" + "d" * 32
        conversation = group + "-" + "1" * 32
        response_id = "resp_created_project_proposal"
        run_id = "run_" + "5" * 32
        store = ResponseStore(max_size=10)
        try:
            owner = self._owner_run_binding(store, conversation, response_id)
            assert store.claim_and_attach_owner_run(
                "default", conversation, response_id,
                owner["claim_id"], run_id,
                operation="owner_task_graph_commit",
                payload_digest=owner["payload_digest"],
            ) is True
            assert store.complete_owner_claim(
                "default", conversation, response_id, owner["claim_id"], run_id,
            ) is True
            snapshot = store.owner_history_snapshot(conversation)
            assert snapshot["proposal_consumed"] is True
            head = snapshot["head_response_id"]

            assert store.reserve_owner_conversation(
                "default", conversation, "resp_stale_tab_turn",
                owner_message="Actually, make it 90 minutes",
            ) is False
            settled = store.owner_history_snapshot(conversation)
            assert settled["pending"] is None
            assert settled["recovery"] is None
            assert settled["head_response_id"] == head
            assert settled["latest_response_id"] == response_id
            assert settled["conversation_closed"] is False

            # The acknowledgement is what advances this owner, and the clean
            # successor it names plans exactly as any new draft does.
            assert store.close_owner_conversation(
                "default", conversation, response_id,
                expected_head_response_id=head, next_session_id="6" * 32,
            ) is True
            successor = group + "-" + "6" * 32
            assert store.owner_session_index(
                "default", group,
            )["current_session_id"] == "6" * 32
            assert store.reserve_owner_conversation(
                "default", successor, "resp_successor_turn",
                owner_message="Prepare a different workshop",
            ) is True
        finally:
            store.close()

    def test_a_project_change_session_is_not_spent_by_its_own_run(self):
        """The other half of the same fence: a Project keeps its conversation.

        A change session's proposal is consumed by the run that applies it
        exactly as a draft's is, and the owner goes on talking about the same
        Project afterwards. The refusal above is about the draft that CREATED a
        Project — the one whose receipt is still owed to a browser — so it must
        read the operation that consumed the proposal rather than the fact that
        one was consumed.
        """
        conversation = "raphael-owner-" + "f" * 32
        response_id = "resp_project_change_proposal"
        run_id = "run_" + "8" * 32
        claim_id = "claim_" + "9" * 32
        store = ResponseStore(max_size=10)
        try:
            store.put(response_id, {
                "response": {"id": response_id, "created_at": 1_000},
                "conversation_history": [{
                    "role": "assistant",
                    "content": json.dumps(_owner_existing_proposal()),
                }],
            })
            assert store.set_conversation(
                conversation, response_id, owner_proposal=True,
            ) is True
            assert store.claim_and_attach_owner_run(
                "default", conversation, response_id, claim_id, run_id,
                operation="owner_project_plan_commit",
                payload_digest="c" * 64,
            ) is True
            assert store.complete_owner_claim(
                "default", conversation, response_id, claim_id, run_id,
            ) is True
            assert store.owner_history_snapshot(
                conversation,
            )["proposal_consumed"] is True

            assert store.reserve_owner_conversation(
                "default", conversation, "resp_next_change_turn",
                owner_message="Now add the weekly report",
            ) is True
            assert store.owner_history_snapshot(conversation)["pending"] == {
                "owner": "Now add the weekly report",
                "response_id": "resp_next_change_turn",
            }
        finally:
            store.close()

    def test_closing_a_conversation_moves_the_group_to_its_next_session(self):
        """Item 32TK round 3: which draft is current is a server-side fact.

        A caller that states the session this group moves on to gets that
        pointer advanced inside the closing transaction. Without it the durable
        pointer still named the conversation that was just closed, so a browser
        holding no cookie at all came back to the retired one.
        """
        group = "raphael-owner-" + "a" * 32
        successor = "b" * 32
        store = ResponseStore(max_size=10)
        try:
            store.put("resp_first_draft_turn", {
                "response": {"id": "resp_first_draft_turn", "created_at": 1},
                "conversation_history": [
                    {"role": "user", "content": "Prepare a workshop"},
                    {
                        "role": "assistant",
                        "content": json.dumps({
                            "schema_version": 1,
                            "kind": "question",
                            "message": "Who is the workshop for?",
                        }),
                    },
                ],
            })
            assert store.set_conversation(
                group, "resp_first_draft_turn", profile="default",
            ) is True
            assert store.owner_session_index(
                "default", group,
            )["current_session_id"] == "legacy"

            assert store.close_owner_conversation(
                "default", group, None,
                expected_head_response_id="resp_first_draft_turn",
                next_session_id=successor,
            ) is True

            index = store.owner_session_index("default", group)
            assert index["current_session_id"] == successor
            # The retired draft is still readable as a past one, and the
            # successor carries no turns yet, so it lists none.
            assert [entry["session_id"] for entry in index["data"]] == ["legacy"]
        finally:
            store.close()

    def test_owner_turn_reservation_fences_claim_close_and_competing_turn(self):
        conversation = "raphael-owner-" + "5" * 32
        response_id = "resp_reserved_source"
        reserved_id = "resp_reserved_next_turn"
        store = ResponseStore(max_size=10)
        store.put(response_id, {
            "response": {"id": response_id, "created_at": 1},
            "conversation_history": [{
                "role": "assistant",
                "content": json.dumps(_owner_new_proposal()),
            }],
        })
        assert store.set_conversation(
            conversation, response_id, owner_proposal=True,
        ) is True
        assert store.reserve_owner_conversation(
            "default", conversation, reserved_id,
        ) is True
        assert store.reserve_owner_conversation(
            "default", conversation, "resp_competing_turn",
        ) is False
        assert store.claim_owner_proposal(
            "default", conversation, response_id, "claim_" + "6" * 32,
        ) is False
        assert store.close_owner_conversation(
            "default", conversation, response_id,
        ) is False

        store.put(reserved_id, {
            "response": {"id": reserved_id, "created_at": 2},
            "conversation_history": [{
                "role": "assistant",
                "content": json.dumps({"schema_version": 1, "kind": "question"}),
            }],
        })
        assert store.set_conversation(
            conversation,
            reserved_id,
            profile="default",
            reservation_id=reserved_id,
        ) is True
        assert store.claim_owner_proposal(
            "default", conversation, response_id, "claim_" + "6" * 32,
        ) is False

    def test_expired_owner_turn_reservation_is_reclaimed_after_restart(
        self, tmp_path,
    ):
        conversation = "raphael-owner-" + "e" * 32
        db_path = tmp_path / "response-store.db"
        first = ResponseStore(db_path=db_path, max_size=10)
        assert first.reserve_owner_conversation(
            "default", conversation, "resp_crash_left_turn",
        ) is True
        assert first.renew_owner_conversation_reservation(
            "default", conversation, "resp_crash_left_turn",
        ) is True
        first._conn.execute(
            "UPDATE owner_conversation_reservations SET expires_at = ? "
            "WHERE profile = ? AND name = ?",
            (time.time() - 1, "default", conversation),
        )
        first._conn.commit()
        first.close()

        restarted = ResponseStore(db_path=db_path, max_size=10)
        try:
            assert restarted.reserve_owner_conversation(
                "default", conversation, "resp_recovered_turn",
            ) is True
            assert restarted.reserve_owner_conversation(
                "default", conversation, "resp_competing_turn",
            ) is False
            assert restarted.set_conversation(
                conversation,
                "resp_recovered_turn",
                profile="default",
                reservation_id="resp_recovered_turn",
            ) is True
            assert restarted.set_conversation(
                conversation,
                "resp_crash_left_turn",
                profile="default",
                reservation_id="resp_crash_left_turn",
            ) is False
            assert restarted.get_conversation(
                conversation, profile="default",
            ) == "resp_recovered_turn"
        finally:
            restarted.close()

    def test_expired_unattached_claim_can_be_recovered(self):
        conversation = "raphael-owner-" + "6" * 32
        response_id = "resp_expired_owner_claim"
        store = ResponseStore(max_size=10)
        store.put(response_id, {
            "response": {"id": response_id, "created_at": 1},
            "conversation_history": [{
                "role": "assistant",
                "content": json.dumps(_owner_new_proposal()),
            }],
        })
        assert store.set_conversation(
            conversation, response_id, owner_proposal=True,
        ) is True
        assert store.claim_owner_proposal(
            "default", conversation, response_id, "claim_" + "7" * 32,
        ) is True
        store._conn.execute(
            "UPDATE conversations SET claim_expires_at = ? "
            "WHERE profile = ? AND name = ?",
            (time.time() - 1, "default", conversation),
        )
        store._conn.commit()
        assert store.claim_owner_proposal(
            "default", conversation, response_id, "claim_" + "8" * 32,
        ) is True

    def test_run_idempotency_and_owner_proposal_backfill_survive_restart(
        self, tmp_path,
    ):
        db_path = str(tmp_path / "legacy-response-store.db")
        conversation = "raphael-owner-" + "9" * 32
        response_id = "resp_legacy_owner_proposal"
        raw = {
            "response": {"id": response_id, "created_at": 1},
            "conversation_history": [{
                "role": "assistant",
                "content": json.dumps(_owner_new_proposal()),
            }],
        }
        conn = sqlite3.connect(db_path)
        conn.execute(
            "CREATE TABLE responses (response_id TEXT PRIMARY KEY, data TEXT NOT NULL, "
            "accessed_at REAL NOT NULL)"
        )
        conn.execute(
            "CREATE TABLE conversations (name TEXT PRIMARY KEY, response_id TEXT NOT NULL)"
        )
        conn.execute(
            "INSERT INTO responses (response_id, data, accessed_at) VALUES (?, ?, ?)",
            (response_id, json.dumps(raw), 1.0),
        )
        conn.execute(
            "INSERT INTO conversations (name, response_id) VALUES (?, ?)",
            (conversation, response_id),
        )
        conn.commit()
        conn.close()

        store = ResponseStore(
            max_size=10, db_path=db_path, default_profile="default",
        )
        assert store.owner_proposal_record(
            "default", conversation, response_id,
        ) is not None
        assert store.reserve_run_idempotency(
            "default", "scope", "key", "fingerprint", "run_" + "a" * 32,
        ) == ("new", "run_" + "a" * 32)
        store.close()

        reopened = ResponseStore(
            max_size=10, db_path=db_path, default_profile="default",
        )
        assert reopened.lookup_run_idempotency(
            "default", "scope", "key", "fingerprint",
        ) == ("existing", "run_" + "a" * 32)
        assert reopened.lookup_run_idempotency(
            "secondary", "scope", "key", "fingerprint",
        ) == ("missing", None)
        response_pk = [
            row[1]
            for row in sorted(
                reopened._conn.execute("PRAGMA table_info(responses)").fetchall(),
                key=lambda row: row[5],
            )
            if row[5]
        ]
        assert response_pk == ["profile", "response_id"]

    def test_schema_upgrade_is_serialized_across_connections(
        self, tmp_path, monkeypatch,
    ):
        db_path = str(tmp_path / "concurrent-legacy-response-store.db")
        conn = sqlite3.connect(db_path)
        conn.execute(
            "CREATE TABLE responses (response_id TEXT PRIMARY KEY, data TEXT NOT NULL, "
            "accessed_at REAL NOT NULL)"
        )
        conn.commit()
        conn.close()

        original = ResponseStore._initialize_schema_locked
        first_inside = threading.Event()
        second_started = threading.Event()
        call_lock = threading.Lock()
        call_count = 0

        def delayed_first_migration(store):
            nonlocal call_count
            with call_lock:
                call_count += 1
                first = call_count == 1
            if first:
                first_inside.set()
                assert second_started.wait(1)
                time.sleep(0.1)
            return original(store)

        monkeypatch.setattr(
            ResponseStore, "_initialize_schema_locked", delayed_first_migration,
        )
        errors = []

        def open_store(*, second=False):
            if second:
                second_started.set()
            try:
                store = ResponseStore(db_path=db_path, max_size=10)
                store.close()
            except Exception as exc:  # pragma: no cover - asserted below
                errors.append(exc)

        first_thread = threading.Thread(target=open_store, daemon=True)
        first_thread.start()
        assert first_inside.wait(1)
        second_thread = threading.Thread(
            target=lambda: open_store(second=True), daemon=True,
        )
        second_thread.start()
        first_thread.join(timeout=3)
        second_thread.join(timeout=3)

        assert not first_thread.is_alive()
        assert not second_thread.is_alive()
        assert errors == []
        reopened = ResponseStore(db_path=db_path, max_size=10)
        try:
            response_pk = [
                row[1]
                for row in sorted(
                    reopened._conn.execute(
                        "PRAGMA table_info(responses)"
                    ).fetchall(),
                    key=lambda row: row[5],
                )
                if row[5]
            ]
            run_columns = {
                row[1]
                for row in reopened._conn.execute(
                    "PRAGMA table_info(run_idempotency)"
                )
            }
            assert response_pk == ["profile", "response_id"]
            assert {"status_json", "terminal_json"} <= run_columns
        finally:
            reopened.close()

    def test_owner_history_missing_or_invalid_conversation_is_empty(self):
        store = ResponseStore(max_size=10)
        store.put("resp_invalid_history", {
            "response": {"id": "resp_invalid_history"},
            "conversation_history": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "unstructured private text"},
            ],
        })
        store.set_conversation("invalid-history", "resp_invalid_history")

        assert store.owner_history("missing-history") == []
        assert store.owner_history("invalid-history") == []

        assert store.owner_history_snapshot("missing-history") == {
            "head_response_id": None,
            "latest_response_id": None,
            "proposal_consumed": False,
            "proposal_claimed": False,
            "active_run_id": None,
            "completed_run_id": None,
            "conversation_closed": False,
            "truncated": False,
            "incomplete": False,
            "pending": None,
            "recovery": None,
            "data": [],
        }


    # -----------------------------------------------------------------------
    # Item 32TK: absent authority, BROKEN authority, and honest truncation
    # -----------------------------------------------------------------------

    @staticmethod
    def _owner_store(history, *, name="raphael-owner-" + "9" * 32):
        store = ResponseStore(max_size=10)
        store.put("resp_broken_authority", {
            "response": {"id": "resp_broken_authority"},
            "conversation_history": history,
        })
        assert store.set_conversation(name, "resp_broken_authority") is True
        return store, name

    @staticmethod
    def _question(text):
        return json.dumps(
            {"schema_version": 1, "kind": "question", "message": text}
        )

    @staticmethod
    def _failure_reply():
        """The exact reply a completed trailing turn is shown with."""
        return json.dumps(
            {
                "schema_version": 1,
                "kind": "failure",
                "message": _OWNER_INTERRUPTED_TURN_MESSAGE,
            },
            ensure_ascii=False,
        )

    def test_a_conversation_with_no_mapping_is_empty_not_broken(self):
        store = ResponseStore(max_size=10)
        snapshot = store.owner_history_snapshot("raphael-owner-" + "8" * 32)
        assert snapshot["head_response_id"] is None
        assert snapshot["data"] == []

    def test_a_mapped_response_that_is_gone_is_a_service_failure(self):
        """Absent authority and broken authority are not the same answer."""
        store, name = self._owner_store([
            {"role": "user", "content": "Plan it."},
            {"role": "assistant", "content": self._question("Which first?")},
        ])
        # The head still names it, but the row it names is gone.
        store._conn.execute("DELETE FROM responses WHERE response_id = ?",
                            ("resp_broken_authority",))
        store._conn.commit()

        with pytest.raises(OwnerAuthorityBroken):
            store.owner_history_snapshot(name)

    def test_a_corrupt_stored_response_is_a_service_failure(self):
        store, name = self._owner_store([
            {"role": "user", "content": "Plan it."},
            {"role": "assistant", "content": self._question("Which first?")},
        ])
        store._conn.execute(
            "UPDATE responses SET data = ? WHERE response_id = ?",
            ("{not json", "resp_broken_authority"),
        )
        store._conn.commit()

        with pytest.raises(OwnerAuthorityBroken):
            store.owner_history_snapshot(name)

    def test_a_transcript_that_is_not_a_list_is_a_service_failure(self):
        store, name = self._owner_store("not a transcript")
        with pytest.raises(OwnerAuthorityBroken):
            store.owner_history_snapshot(name)

    @pytest.mark.parametrize(
        "owner_message",
        [
            {"role": "user", "content": None},
            {"role": "user", "content": ["multimodal"]},
            {"role": "user", "content": "   "},
        ],
    )
    def test_a_malformed_owner_turn_fails_instead_of_merging_turns(
        self, owner_message,
    ):
        """An owner message is a turn boundary, so dropping one would show a
        transcript that never happened."""
        store, name = self._owner_store([
            {"role": "user", "content": "First ask."},
            {"role": "assistant", "content": self._question("First reply?")},
            owner_message,
            {"role": "assistant", "content": self._question("Second reply?")},
        ])
        with pytest.raises(OwnerAuthorityBroken):
            store.owner_history_snapshot(name)

    def test_an_oversized_owner_turn_preserves_its_boundary_and_reports_truncation(
        self,
    ):
        oversized = "x" * 12_001
        first_reply = self._question("First reply?")
        second_reply = self._question("Second reply?")
        store, name = self._owner_store([
            {"role": "user", "content": "First ask."},
            {"role": "assistant", "content": first_reply},
            {"role": "user", "content": oversized},
            {"role": "assistant", "content": second_reply},
        ])

        snapshot = store.owner_history_snapshot(name)

        assert [turn["raphael"] for turn in snapshot["data"]] == [
            first_reply, second_reply,
        ]
        assert snapshot["data"][1]["owner"] == (
            "[Earlier owner message omitted from this view because it exceeded "
            "the safe display limit. The native record remains unchanged.]"
        )
        assert oversized not in json.dumps(snapshot["data"])
        assert snapshot["truncated"] is True

    def test_an_unprojectable_final_reply_fails_instead_of_showing_an_older_one(
        self,
    ):
        store, name = self._owner_store([
            {"role": "user", "content": "Plan it."},
            {"role": "assistant", "content": self._question("An early reply?")},
            {"role": "assistant", "content": self._question("y" * 50_001)},
        ])
        with pytest.raises(OwnerAuthorityBroken):
            store.owner_history_snapshot(name)

    def test_tool_traffic_and_excluded_reply_kinds_are_still_skipped(self):
        """Not every unprojectable message is a defect: the projection
        deliberately carries neither tool traffic nor an Automations reply."""
        store, name = self._owner_store([
            {"role": "user", "content": "Plan it."},
            {"role": "assistant", "content": "", "tool_calls": [{"id": "1"}]},
            {"role": "tool", "content": "private tool output"},
            {"role": "assistant", "content": self._question("Which first?")},
            {"role": "user", "content": "Automate it."},
            {"role": "assistant", "content": json.dumps({
                "schema_version": 1, "kind": "automation_proposal",
            })},
        ])
        snapshot = store.owner_history_snapshot(name)

        # One projected turn, and the head is reported regardless.
        assert [turn["owner"] for turn in snapshot["data"]] == ["Plan it."]
        assert snapshot["head_response_id"] == "resp_broken_authority"
        assert snapshot["truncated"] is False

    def test_a_head_with_no_projectable_turn_is_still_reported(self):
        """An Automations conversation projects zero turns and still has a head.

        Nulling it made every later turn on that conversation either conflict
        as stale or replay the old proposal.
        """
        store, name = self._owner_store([
            {"role": "user", "content": "Automate it."},
            {"role": "assistant", "content": json.dumps({
                "schema_version": 1, "kind": "automation_proposal",
            })},
        ])
        snapshot = store.owner_history_snapshot(name)

        assert snapshot["data"] == []
        assert snapshot["head_response_id"] == "resp_broken_authority"

    def test_history_past_the_window_is_reported_as_truncated(self):
        history = []
        for index in range(45):
            history.append({"role": "user", "content": f"Ask {index}."})
            history.append(
                {"role": "assistant", "content": self._question(f"Reply {index}?")}
            )
        store, name = self._owner_store(history)
        snapshot = store.owner_history_snapshot(name)

        assert len(snapshot["data"]) == 40
        assert snapshot["truncated"] is True
        assert snapshot["data"][0]["owner"] == "Ask 5."

    # -----------------------------------------------------------------------
    # Item 32TK round 2: malformed alternation is reported, never hidden
    # -----------------------------------------------------------------------

    @pytest.mark.parametrize("record", [None, "a bare string", 7, ["nested"]])
    def test_a_non_object_transcript_record_is_a_service_failure(self, record):
        """Skipping one would merge the owner turns on either side of it."""
        store, name = self._owner_store([
            {"role": "user", "content": "First ask."},
            record,
            {"role": "assistant", "content": self._question("First reply?")},
        ])
        with pytest.raises(OwnerAuthorityBroken):
            store.owner_history_snapshot(name)

    def test_consecutive_owner_turns_are_reported_as_incomplete(self):
        """The first owner message really happened, so the projection says it
        is not the whole conversation instead of silently dropping it."""
        store, name = self._owner_store([
            {"role": "user", "content": "Automate it."},
            {"role": "assistant", "content": json.dumps({
                "schema_version": 1, "kind": "automation_proposal",
            })},
            {"role": "user", "content": "Now plan it."},
            {"role": "assistant", "content": self._question("Which first?")},
        ])
        snapshot = store.owner_history_snapshot(name)

        assert [turn["owner"] for turn in snapshot["data"]] == ["Now plan it."]
        assert snapshot["incomplete"] is True

    def test_a_trailing_owner_turn_is_reported_as_incomplete(self):
        store, name = self._owner_store([
            {"role": "user", "content": "Plan it."},
            {"role": "assistant", "content": self._question("Which first?")},
            {"role": "user", "content": "The second one."},
        ])
        snapshot = store.owner_history_snapshot(name)

        assert [turn["owner"] for turn in snapshot["data"]] == ["Plan it."]
        assert snapshot["incomplete"] is True

    def test_a_clean_alternation_is_not_marked_incomplete(self):
        store, name = self._owner_store([
            {"role": "user", "content": "Plan it."},
            {"role": "assistant", "content": self._question("Which first?")},
        ])
        assert store.owner_history_snapshot(name)["incomplete"] is False

    # -----------------------------------------------------------------------
    # A trailing turn nobody can read ends, it does not disappear
    # -----------------------------------------------------------------------

    def test_a_trailing_turn_answered_with_plain_text_is_shown_as_failed(self):
        """The turn is over and it failed, so it is shown that way.

        Dropping it hid the owner's own latest message and left the Workspace
        waiting on a conversation that had already stopped, with no control to
        recover from.
        """
        raw = "Sure — running `deploy_workshop --force` for you now."
        store, name = self._owner_store([
            {"role": "user", "content": "Plan it."},
            {"role": "assistant", "content": self._question("Which first?")},
            {"role": "user", "content": "The second one."},
            {"role": "assistant", "content": raw},
        ])

        snapshot = store.owner_history_snapshot(name)

        assert [turn["owner"] for turn in snapshot["data"]] == [
            "Plan it.", "The second one.",
        ]
        assert snapshot["data"][-1]["raphael"] == self._failure_reply()
        assert json.loads(snapshot["data"][-1]["raphael"])["message"] == (
            _OWNER_INTERRUPTED_TURN_MESSAGE
        )
        # Shown with a terminal outcome, so it is no longer a HIDDEN turn.
        assert snapshot["incomplete"] is False
        assert snapshot["truncated"] is False
        # Never the stored text itself, and never the command inside it.
        projected = json.dumps(snapshot)
        assert raw not in projected
        assert "deploy_workshop" not in projected

    def test_a_trailing_reply_imitating_a_tool_call_is_not_treated_as_execution(
        self,
    ):
        """Structured JSON that is not a Raphael reply carries no outcome either,
        and nothing about it may read as if it had run."""
        imitation = json.dumps({
            "tool": "run_command",
            "arguments": {"command": "rm -rf /srv/workshop"},
        })
        store, name = self._owner_store([
            {"role": "user", "content": "Plan it."},
            {"role": "assistant", "content": self._question("Which first?")},
            {"role": "user", "content": "The second one."},
            {"role": "assistant", "content": imitation},
        ])

        snapshot = store.owner_history_snapshot(name)
        projected = json.dumps(snapshot)

        assert snapshot["data"][-1] == {
            "owner": "The second one.",
            "raphael": self._failure_reply(),
        }
        assert "run_command" not in projected
        assert "rm -rf" not in projected
        assert "srv/workshop" not in projected

    def test_completing_a_trailing_turn_reactivates_no_older_proposal(self):
        """A terminal failure is not a proposal, so nothing becomes approvable."""
        proposal = json.dumps(_owner_new_proposal())
        name = "raphael-owner-" + "b" * 32
        store = ResponseStore(max_size=10)
        store.put("resp_proposal_turn", {
            "response": {"id": "resp_proposal_turn"},
            "conversation_history": [
                {"role": "user", "content": "Plan it."},
                {"role": "assistant", "content": proposal},
            ],
        })
        assert store.set_conversation(
            name, "resp_proposal_turn", owner_proposal=True,
        ) is True
        assert store.owner_history_snapshot(name)["latest_response_id"] == (
            "resp_proposal_turn"
        )

        store.put("resp_stranded_turn", {
            "response": {"id": "resp_stranded_turn"},
            "conversation_history": [
                {"role": "user", "content": "Plan it."},
                {"role": "assistant", "content": proposal},
                {"role": "user", "content": "Actually, change it."},
                {"role": "assistant", "content": "thinking out loud"},
            ],
        })
        assert store.set_conversation(name, "resp_stranded_turn") is True

        snapshot = store.owner_history_snapshot(name)

        # The stranded turn is visible with its terminal failure...
        assert snapshot["data"][-1] == {
            "owner": "Actually, change it.",
            "raphael": self._failure_reply(),
        }
        # ...and grants nothing the earlier proposal had.
        assert snapshot["latest_response_id"] is None
        assert snapshot["proposal_consumed"] is False
        assert snapshot["proposal_claimed"] is False
        assert snapshot["active_run_id"] is None
        assert snapshot["completed_run_id"] is None
        assert snapshot["head_response_id"] == "resp_stranded_turn"

        # Same guarantee where the durable proposal handle SURVIVES, because
        # the head itself was rewritten in place: a projection carrying nothing
        # but the substituted failure still hands that handle to nobody.
        rewritten = ResponseStore(max_size=10)
        rewritten_name = "raphael-owner-" + "d" * 32
        rewritten.put("resp_rewritten_head", {
            "response": {"id": "resp_rewritten_head"},
            "conversation_history": [{"role": "assistant", "content": proposal}],
        })
        assert rewritten.set_conversation(
            rewritten_name, "resp_rewritten_head", owner_proposal=True,
        ) is True
        rewritten.put("resp_rewritten_head", {
            "response": {"id": "resp_rewritten_head"},
            "conversation_history": [
                {"role": "assistant", "content": proposal},
                {"role": "user", "content": "Actually, change it."},
                {"role": "assistant", "content": "thinking out loud"},
            ],
        })
        assert rewritten.set_conversation(
            rewritten_name, "resp_rewritten_head",
        ) is True

        reprojected = rewritten.owner_history_snapshot(rewritten_name)

        assert reprojected["data"] == [{
            "owner": "Actually, change it.",
            "raphael": self._failure_reply(),
        }]
        assert reprojected["latest_response_id"] is None

    def test_only_the_trailing_turn_is_completed_this_way(self):
        """An interior turn is already bounded by the owner turn after it, so
        nothing there is stranded and today's reporting stands."""
        store, name = self._owner_store([
            {"role": "user", "content": "First ask."},
            {"role": "assistant", "content": "unstructured private text"},
            {"role": "user", "content": "Second ask."},
            {"role": "assistant", "content": self._question("Which first?")},
            {"role": "user", "content": "Third ask."},
            {"role": "assistant", "content": "more unstructured text"},
        ])

        snapshot = store.owner_history_snapshot(name)

        assert [turn["owner"] for turn in snapshot["data"]] == [
            "Second ask.", "Third ask.",
        ]
        assert snapshot["data"][-1]["raphael"] == self._failure_reply()
        # The interior one is still reported as missing, never substituted.
        assert snapshot["incomplete"] is True
        assert self._failure_reply() not in [
            turn["raphael"] for turn in snapshot["data"][:-1]
        ]

    def test_completing_a_trailing_turn_never_hides_broken_authority(self):
        """Completing one turn is a projection, not a repair of the store."""
        trailing = {"role": "assistant", "content": "unstructured private text"}
        sound, sound_name = self._owner_store(
            [{"role": "user", "content": "Plan it."}, trailing],
        )
        assert sound.owner_history_snapshot(sound_name)["data"] == [{
            "owner": "Plan it.",
            "raphael": self._failure_reply(),
        }]

        # The same trailing reply over authority that is genuinely broken:
        # an unreadable owner turn, a record that is not an object, a
        # transcript that is not a list, an over-limit authoritative reply.
        for index, history in enumerate([
            [{"role": "user", "content": "   "}, trailing],
            [{"role": "user", "content": "Plan it."}, "a bare string", trailing],
            "not a transcript",
            [
                {"role": "user", "content": "Plan it."},
                {"role": "assistant", "content": self._question("y" * 50_001)},
                trailing,
            ],
        ]):
            store, name = self._owner_store(
                history, name=f"raphael-owner-{index:032x}",
            )
            with pytest.raises(OwnerAuthorityBroken):
                store.owner_history_snapshot(name)

        # A mapped response that is gone, and one whose stored JSON is corrupt.
        missing, missing_name = self._owner_store(
            [{"role": "user", "content": "Plan it."}, trailing],
            name="raphael-owner-" + "5" * 32,
        )
        missing._conn.execute(
            "DELETE FROM responses WHERE response_id = ?",
            ("resp_broken_authority",),
        )
        missing._conn.commit()
        with pytest.raises(OwnerAuthorityBroken):
            missing.owner_history_snapshot(missing_name)

        corrupt, corrupt_name = self._owner_store(
            [{"role": "user", "content": "Plan it."}, trailing],
            name="raphael-owner-" + "4" * 32,
        )
        corrupt._conn.execute(
            "UPDATE responses SET data = ? WHERE response_id = ?",
            ("{not json", "resp_broken_authority"),
        )
        corrupt._conn.commit()
        with pytest.raises(OwnerAuthorityBroken):
            corrupt.owner_history_snapshot(corrupt_name)

    def test_a_valid_reply_after_a_completed_trailing_turn_projects_normally(
        self,
    ):
        """The substitution belongs to one projection, not to the record: the
        next real reply simply lands, and that turn goes back to interior."""
        store, name = self._owner_store([
            {"role": "user", "content": "Plan it."},
            {"role": "assistant", "content": self._question("Which first?")},
            {"role": "user", "content": "The second one."},
            {"role": "assistant", "content": "unstructured private text"},
        ])
        stranded = store.owner_history_snapshot(name)
        assert stranded["data"][-1] == {
            "owner": "The second one.",
            "raphael": self._failure_reply(),
        }

        store.put("resp_follow_up", {
            "response": {"id": "resp_follow_up"},
            "conversation_history": [
                {"role": "user", "content": "Plan it."},
                {"role": "assistant", "content": self._question("Which first?")},
                {"role": "user", "content": "The second one."},
                {"role": "assistant", "content": "unstructured private text"},
                {"role": "user", "content": "Try again."},
                {"role": "assistant", "content": self._question("Ready now?")},
            ],
        })
        assert store.set_conversation(name, "resp_follow_up") is True

        resumed = store.owner_history_snapshot(name)

        assert resumed["data"][-1] == {
            "owner": "Try again.",
            "raphael": self._question("Ready now?"),
        }
        assert [turn["owner"] for turn in resumed["data"]] == [
            "Plan it.", "Try again.",
        ]
        assert resumed["incomplete"] is True
        assert self._failure_reply() not in [
            turn["raphael"] for turn in resumed["data"]
        ]

    # -----------------------------------------------------------------------
    # Item 32TK round 2: the session index cannot lose the current session
    # -----------------------------------------------------------------------

    @staticmethod
    def _session_group_store():
        group = "raphael-owner-" + "a" * 32
        store = ResponseStore(max_size=20)

        def add(name, response_id, created_at, owner):
            store.put(response_id, {
                "response": {"id": response_id, "created_at": created_at},
                "conversation_history": [
                    {"role": "user", "content": owner},
                    {"role": "assistant", "content": json.dumps({
                        "schema_version": 1, "kind": "question",
                        "message": "Which first?",
                    })},
                ],
            })
            assert store.set_conversation(name, response_id) is True

        return store, group, add

    def test_reading_an_old_session_cannot_make_it_current(self):
        """Ordering by the mapped response's LRU access time meant opening an
        old change promoted it past the real current one."""
        store, group, add = self._session_group_store()
        older = "1" * 32
        newer = "2" * 32
        add(f"{group}-{older}", "resp_session_older", 100, "The older change.")
        add(f"{group}-{newer}", "resp_session_newer", 200, "The newer change.")

        # Exactly what opening the older change does: it touches accessed_at.
        assert store.get("resp_session_older") is not None

        index = store.owner_session_index("default", group)
        assert [item["session_id"] for item in index["data"]] == [newer, older]
        assert index["current_session_id"] == newer

    def test_the_current_session_survives_the_index_bound(self):
        store, group, add = self._session_group_store()
        for index_number in range(105):
            add(
                f"{group}-{index_number:032x}",
                f"resp_session_{index_number}",
                100 + index_number,
                f"Change {index_number}.",
            )
        newest = f"{104:032x}"

        index = store.owner_session_index("default", group)
        assert index["truncated"] is True
        assert index["data"][0]["session_id"] == newest
        assert index["current_session_id"] == newest

    def test_a_session_whose_mapped_response_is_gone_is_reported_unavailable(self):
        """An inner join dropped it silently, which could remove the real
        current session before anything validated it."""
        store, group, add = self._session_group_store()
        broken = "3" * 32
        add(f"{group}-{broken}", "resp_session_broken", 300, "The current change.")
        store._conn.execute(
            "DELETE FROM responses WHERE response_id = ?", ("resp_session_broken",),
        )
        store._conn.commit()

        index = store.owner_session_index("default", group)
        assert [item["session_id"] for item in index["data"]] == [broken]
        assert index["data"][0]["available"] is False
        assert index["current_session_id"] == broken

    def test_a_pre_upgrade_store_is_seeded_with_durable_sequences(self):
        """A store written before the sequence existed still has to answer
        "which session is current" without consulting access order again."""
        store, group, add = self._session_group_store()
        older = "5" * 32
        newer = "6" * 32
        add(f"{group}-{older}", "resp_seed_older", 100, "The older change.")
        add(f"{group}-{newer}", "resp_seed_newer", 200, "The newer change.")
        # Exactly a pre-upgrade store: mapped conversations, no sequence table.
        store._conn.execute("DROP TABLE owner_conversation_sessions")
        store._conn.commit()

        store._initialize_schema()

        # Seeded oldest-access-first, so the newest sibling is current — and
        # reading the older one afterwards cannot move it.
        assert store.get("resp_seed_older") is not None
        index = store.owner_session_index("default", group)
        assert [item["session_id"] for item in index["data"]] == [newer, older]
        assert index["current_session_id"] == newer

    def test_a_session_with_a_corrupt_stored_head_is_reported_unavailable(self):
        store, group, add = self._session_group_store()
        broken = "4" * 32
        add(f"{group}-{broken}", "resp_session_corrupt", 400, "The current change.")
        store._conn.execute(
            "UPDATE responses SET data = ? WHERE response_id = ?",
            ("{not json", "resp_session_corrupt"),
        )
        store._conn.commit()

        index = store.owner_session_index("default", group)
        assert index["data"][0]["available"] is False
        assert index["current_session_id"] == broken

    # -----------------------------------------------------------------------
    # Item 32TK round 2: closing compares the conversation's REAL head
    # -----------------------------------------------------------------------

    def test_closing_refuses_when_the_head_moved_under_a_stale_tab(self):
        """An ordinary question does not change the outstanding proposal, so
        comparing only that let a stale tab close and hide the newer turn."""
        group = "raphael-owner-" + "b" * 32
        store = ResponseStore(max_size=10)
        store.put("resp_close_first", {
            "response": {"id": "resp_close_first"},
            "conversation_history": [],
        })
        assert store.set_conversation(group, "resp_close_first") is True
        # A concurrent question lands: the head moves, the proposal does not.
        store.put("resp_close_second", {
            "response": {"id": "resp_close_second"},
            "conversation_history": [],
        })
        assert store.set_conversation(group, "resp_close_second") is True

        assert store.close_owner_conversation(
            "default", group, None,
            expected_head_response_id="resp_close_first",
        ) is False
        assert store.owner_history_snapshot(group)["conversation_closed"] is False

        assert store.close_owner_conversation(
            "default", group, None,
            expected_head_response_id="resp_close_second",
        ) is True
        assert store.owner_history_snapshot(group)["conversation_closed"] is True

    def test_closing_a_conversation_with_no_mapping_refuses_a_stated_head(self):
        store = ResponseStore(max_size=10)
        group = "raphael-owner-" + "c" * 32
        assert store.close_owner_conversation(
            "default", group, None,
            expected_head_response_id="resp_never_existed",
        ) is False
        assert store.close_owner_conversation(
            "default", group, None, expected_head_response_id=None,
        ) is True


# ---------------------------------------------------------------------------
# _IdempotencyCache
# ---------------------------------------------------------------------------


class TestIdempotencyCache:
    def test_request_fingerprint_canonicalizes_nested_maps(self):
        first = {
            "input": "Apply it.",
            "owner_proposal_authority": {"payload": {"a": 1, "b": 2}},
        }
        reordered = {
            "input": "Apply it.",
            "owner_proposal_authority": {"payload": {"b": 2, "a": 1}},
        }

        assert _make_request_fingerprint(
            first, sorted(first),
        ) == _make_request_fingerprint(reordered, sorted(reordered))

    @pytest.mark.asyncio
    async def test_concurrent_same_key_and_fingerprint_runs_once(self):
        cache = _IdempotencyCache()
        gate = asyncio.Event()
        started = asyncio.Event()
        calls = 0

        async def compute():
            nonlocal calls
            calls += 1
            started.set()
            await gate.wait()
            return ("response", {"total_tokens": 1})

        first = asyncio.create_task(cache.get_or_set("idem-key", "fp-1", compute))
        second = asyncio.create_task(cache.get_or_set("idem-key", "fp-1", compute))

        await started.wait()
        assert calls == 1

        gate.set()
        first_result, second_result = await asyncio.gather(first, second)

        assert first_result == second_result == ("response", {"total_tokens": 1})


# ---------------------------------------------------------------------------
# Adapter initialization
# ---------------------------------------------------------------------------


class TestAdapterInit:
    def test_default_config(self):
        config = PlatformConfig(enabled=True)
        adapter = APIServerAdapter(config)
        assert adapter._host == "127.0.0.1"
        assert adapter._port == 8642
        assert adapter._api_key == ""
        assert adapter.platform == Platform.API_SERVER

    def test_custom_config_from_extra(self):
        config = PlatformConfig(
            enabled=True,
            extra={
                "host": "0.0.0.0",
                "port": 9999,
                "key": "sk-test",
                "cors_origins": ["http://localhost:3000"],
            },
        )
        adapter = APIServerAdapter(config)
        assert adapter._host == "0.0.0.0"
        assert adapter._port == 9999
        assert adapter._api_key == "sk-test"
        assert adapter._cors_origins == ("http://localhost:3000",)


    def test_create_agent_forwards_runtime_config(self, monkeypatch):
        captured = {}

        class FakeAgent:
            def __init__(self, **kwargs):
                captured.update(kwargs)

        monkeypatch.setattr("run_agent.AIAgent", FakeAgent)
        monkeypatch.setattr(
            "gateway.run._resolve_runtime_agent_kwargs",
            lambda: {
                "provider": "openai-codex",
                "base_url": "https://example.test/v1",
                "api_mode": "codex_responses",
            },
        )
        monkeypatch.setattr("gateway.run._resolve_gateway_model", lambda: "gpt-5.5")
        monkeypatch.setattr(
            "gateway.run._load_gateway_config",
            lambda: {
                "agent": {"reasoning_effort": "xhigh"},
                "checkpoints": {
                    "enabled": True,
                    "max_snapshots": 7,
                    "max_total_size_mb": 321,
                    "max_file_size_mb": 4,
                },
            },
        )
        monkeypatch.setattr(
            "gateway.run.GatewayRunner._load_reasoning_config",
            staticmethod(lambda model="": {"enabled": True, "effort": "xhigh"}),
        )
        monkeypatch.setattr("gateway.run.GatewayRunner._load_fallback_model", staticmethod(lambda: None))
        monkeypatch.setattr("hermes_cli.tools_config._get_platform_tools", lambda *_: set())

        adapter = APIServerAdapter(PlatformConfig(enabled=True))
        monkeypatch.setattr(adapter, "_ensure_session_db", lambda: None)

        agent = adapter._create_agent(session_id="api-session")

        assert isinstance(agent, FakeAgent)
        assert captured["reasoning_config"] == {"enabled": True, "effort": "xhigh"}
        assert captured["checkpoints_enabled"] is True
        assert captured["checkpoint_max_snapshots"] == 7
        assert captured["checkpoint_max_total_size_mb"] == 321
        assert captured["checkpoint_max_file_size_mb"] == 4


# ---------------------------------------------------------------------------
# Auth checking
# ---------------------------------------------------------------------------


class TestAuth:
    def test_no_key_configured_allows_all(self):
        config = PlatformConfig(enabled=True)
        adapter = APIServerAdapter(config)
        mock_request = MagicMock()
        mock_request.headers = {}
        assert adapter._check_auth(mock_request) is None


    def test_non_ascii_bearer_token_returns_401_not_500(self):
        """A non-ASCII byte in the bearer token must be rejected with 401, not
        crash the handler: hmac.compare_digest raises TypeError on a str with
        non-ASCII characters, and the token is raw client input."""
        config = PlatformConfig(enabled=True, extra={"key": "sk-test123"})
        adapter = APIServerAdapter(config)
        mock_request = MagicMock()
        mock_request.headers = {"Authorization": "Bearer ské-not-the-key"}
        result = adapter._check_auth(mock_request)  # must not raise
        assert result is not None
        assert result.status == 401


# ---------------------------------------------------------------------------
# Concurrency cap (gateway.api_server.max_concurrent_runs) — #7483
# ---------------------------------------------------------------------------


class TestConcurrencyCap:

    def test_resolve_reads_config_value(self):
        cfg = {"gateway": {"api_server": {"max_concurrent_runs": 3}}}
        with patch("hermes_cli.config.load_config", return_value=cfg):
            assert APIServerAdapter._resolve_max_concurrent_runs() == 3


    def test_under_cap_returns_none(self):
        adapter = _make_adapter()
        adapter._max_concurrent_runs = 5
        adapter._inflight_agent_runs = 2
        assert adapter._concurrency_limited_response() is None

    def test_at_cap_returns_429_with_retry_after(self):
        adapter = _make_adapter()
        adapter._max_concurrent_runs = 3
        adapter._inflight_agent_runs = 3
        resp = adapter._concurrency_limited_response()
        assert resp is not None
        assert resp.status == 429
        assert resp.headers.get("Retry-After")


# ---------------------------------------------------------------------------
# Helpers for HTTP tests
# ---------------------------------------------------------------------------


def _make_adapter(api_key: str = "", cors_origins=None) -> APIServerAdapter:
    """Create an adapter with optional API key."""
    extra = {}
    if api_key:
        extra["key"] = api_key
    if cors_origins is not None:
        extra["cors_origins"] = cors_origins
    config = PlatformConfig(enabled=True, extra=extra)
    return APIServerAdapter(config)


def _create_app(adapter: APIServerAdapter) -> web.Application:
    """Create the aiohttp app from the adapter (without starting the full server)."""
    mws = [mw for mw in (cors_middleware, security_headers_middleware) if mw is not None]
    app = web.Application(middlewares=mws)
    app["api_server_adapter"] = adapter
    app.router.add_get("/health", adapter._handle_health)
    app.router.add_get("/health/detailed", adapter._handle_health_detailed)
    app.router.add_get("/v1/health", adapter._handle_health)
    app.router.add_get("/v1/models", adapter._handle_models)
    app.router.add_get("/api/model/options", adapter._handle_model_options)
    app.router.add_get("/v1/capabilities", adapter._handle_capabilities)
    app.router.add_get("/v1/skills", adapter._handle_skills)
    app.router.add_get("/v1/toolsets", adapter._handle_toolsets)
    app.router.add_get("/v1/owner-workspace/projects", adapter._handle_owner_workspace_projects)
    app.router.add_get(
        "/v1/owner-workspace/projects/{project_slug}/snapshot",
        adapter._handle_owner_workspace_project_snapshot,
    )
    app.router.add_get(
        "/v1/owner-workspace/projects/{project_slug}/attachments/{attachment_id}",
        adapter._handle_owner_workspace_project_attachment,
    )
    app.router.add_get("/v1/owner-workspace/decisions", adapter._handle_owner_workspace_decisions)
    app.router.add_post("/api/sessions/{session_id}/chat", adapter._handle_session_chat)
    app.router.add_post("/api/sessions/{session_id}/chat/stream", adapter._handle_session_chat_stream)
    app.router.add_post("/v1/chat/completions", adapter._handle_chat_completions)
    app.router.add_post("/v1/responses", adapter._handle_responses)
    app.router.add_get(
        "/v1/responses/conversations/{conversation}",
        adapter._handle_owner_conversation_history,
    )
    app.router.add_post(
        "/v1/responses/conversations/{conversation}/consume",
        adapter._handle_consume_owner_proposal,
    )
    app.router.add_post(
        "/v1/responses/conversations/{conversation}/authority",
        adapter._handle_owner_conversation_authority,
    )
    app.router.add_post(
        "/v1/responses/conversations/{conversation}/recovery",
        adapter._handle_acknowledge_owner_recovery,
    )
    app.router.add_get("/v1/responses/{response_id}", adapter._handle_get_response)
    app.router.add_delete("/v1/responses/{response_id}", adapter._handle_delete_response)
    app.router.add_post(
        "/api/platforms/{platform}/events",
        adapter._handle_platform_event_callback,
    )
    return app


class _FakeGoogleChatAdapter:
    def __init__(self, *, verify_ok: bool = True, verify_code: str = ""):
        self.verify_ok = verify_ok
        self.verify_code = verify_code
        self.dispatched = []

    def verify_http_event_request(self, auth_header: str):
        self.auth_header = auth_header
        return self.verify_ok, self.verify_code

    async def dispatch_http_event(self, payload):
        self.dispatched.append(payload)
        return {"ok": True}


@pytest.fixture
def adapter():
    return _make_adapter()


@pytest.fixture
def auth_adapter():
    return _make_adapter(api_key="sk-secret")


# ---------------------------------------------------------------------------
# Adapter internals
# ---------------------------------------------------------------------------


class TestAgentExecution:
    @pytest.mark.asyncio
    async def test_run_agent_uses_session_id_as_task_id(self, adapter):
        mock_agent = MagicMock()
        mock_agent.run_conversation.return_value = {"final_response": "ok"}
        mock_agent.session_prompt_tokens = 1
        mock_agent.session_completion_tokens = 2
        mock_agent.session_total_tokens = 3

        model_options = {"reasoning": {"enabled": False}, "fast": False}
        with patch.object(adapter, "_create_agent", return_value=mock_agent) as mock_create_agent:
            result, usage = await adapter._run_agent(
                user_message="hello",
                conversation_history=[],
                session_id="session-123",
                requested_model="MiniMax-M3",
                requested_provider="minimax",
                model_options=model_options,
            )

        # _run_agent annotates result with the effective agent.session_id
        # when it's a real string, so the response-header writer can track
        # compression-triggered session rotations (#16938). The mock agent
        # here doesn't set an explicit session_id string so the guard skips
        # the annotation — header will fall back to the provided session_id.
        assert result["final_response"] == "ok"
        assert usage == {"input_tokens": 1, "output_tokens": 2, "total_tokens": 3}
        create_kwargs = mock_create_agent.call_args.kwargs
        assert create_kwargs["requested_model"] == "MiniMax-M3"
        assert create_kwargs["requested_provider"] == "minimax"
        assert create_kwargs["model_options"] == model_options
        mock_agent.run_conversation.assert_called_once_with(
            user_message="hello",
            conversation_history=[],
            task_id="session-123",
        )

    @pytest.mark.asyncio
    async def test_run_agent_sets_and_clears_process_ownership_markers(self, adapter):
        """#76188 review: this surface runs its own agent lifecycle outside
        TurnRunner, so it needs its own baseline snapshot/clear — verify the
        markers _reap_disconnected_agent_processes() reads are actually
        populated during the turn and cleared once it finishes."""
        mock_agent = MagicMock()
        mock_agent.session_prompt_tokens = 0
        mock_agent.session_completion_tokens = 0
        mock_agent.session_total_tokens = 0
        captured = {}

        def _capture_markers(**_kwargs):
            captured["task_id"] = mock_agent._gateway_turn_process_task_id
            captured["baseline"] = mock_agent._gateway_turn_process_baseline
            return {"final_response": "ok"}

        mock_agent.run_conversation.side_effect = _capture_markers

        with patch.object(adapter, "_create_agent", return_value=mock_agent):
            await adapter._run_agent(
                user_message="hello",
                conversation_history=[],
                session_id="session-456",
                requested_model="MiniMax-M3",
                requested_provider="minimax",
                model_options={"reasoning": {"enabled": False}, "fast": False},
            )

        assert captured["task_id"] == "session-456"
        assert isinstance(captured["baseline"], frozenset)
        # Turn completed normally — markers must be cleared so a disconnect
        # arriving after this point can't reap work this turn left running.
        assert mock_agent._gateway_turn_process_task_id == ""
        assert mock_agent._gateway_turn_process_baseline == frozenset()


class TestDisconnectedAgentReap:
    """#76188 review: SSE disconnect handlers must reap only the background
    processes the disconnected turn created, and must no-op when no turn
    ownership was ever recorded on the agent."""

    def test_reaps_baseline_diff_for_owned_turn(self, monkeypatch):
        from gateway.platforms.api_server import _reap_disconnected_agent_processes
        from tools.process_registry import process_registry

        calls = []
        monkeypatch.setattr(
            process_registry,
            "kill_started_since",
            lambda task_id, baseline, *, source: calls.append(
                (task_id, baseline, source)
            )
            or 1,
        )
        agent = types.SimpleNamespace(
            _gateway_turn_process_task_id="session-abc",
            _gateway_turn_process_baseline=frozenset({"proc-1"}),
        )

        _reap_disconnected_agent_processes(agent)

        deadline = time.time() + 1.0
        while not calls and time.time() < deadline:
            time.sleep(0.01)
        assert calls == [
            ("session-abc", frozenset({"proc-1"}), "api_server_sse_disconnect")
        ]

    def test_noop_when_agent_has_no_ownership_markers(self, monkeypatch):
        from gateway.platforms.api_server import _reap_disconnected_agent_processes
        from tools.process_registry import process_registry

        calls = []
        monkeypatch.setattr(
            process_registry,
            "kill_started_since",
            lambda *a, **k: calls.append(True),
        )
        agent = types.SimpleNamespace(
            _gateway_turn_process_task_id="",
            _gateway_turn_process_baseline=None,
        )

        _reap_disconnected_agent_processes(agent)

        time.sleep(0.1)
        assert calls == []

    def test_stale_epoch_skips_reap_when_newer_run_claimed_task_id(self, monkeypatch):
        """#76188 follow-up: concurrent API runs can share a client-provided
        session_id (same task_id). A disconnecting run whose epoch has been
        superseded must NOT kill the newer run's processes."""
        from gateway.platforms.api_server import (
            _clear_turn_process_ownership,
            _publish_turn_process_ownership,
            _reap_disconnected_agent_processes,
        )
        from tools.process_registry import process_registry

        calls = []
        monkeypatch.setattr(
            process_registry,
            "kill_started_since",
            lambda *a, **k: calls.append(True) or 1,
        )
        monkeypatch.setattr(
            process_registry, "snapshot_running_ids", lambda _tid: frozenset()
        )

        run_a = types.SimpleNamespace()
        run_b = types.SimpleNamespace()
        _publish_turn_process_ownership(run_a, "shared-session")
        # Run B claims the same session_id — supersedes A's epoch.
        _publish_turn_process_ownership(run_b, "shared-session")

        _reap_disconnected_agent_processes(run_a)
        time.sleep(0.2)
        assert calls == [], "stale run A must not reap run B's processes"

        # Run B disconnecting IS current — its reap proceeds.
        _reap_disconnected_agent_processes(run_b)
        deadline = time.time() + 1.0
        while not calls and time.time() < deadline:
            time.sleep(0.01)
        assert calls == [True]
        _clear_turn_process_ownership(run_b)

    def test_reap_proceeds_when_own_clear_pruned_the_epoch_entry(self, monkeypatch):
        """A missing epoch entry (the abandoned run's own finally already
        cleared it) means no newer claimant — the reap must proceed using a
        pre-captured marker snapshot, or the leak survives."""
        from gateway.platforms.api_server import (
            _clear_turn_process_ownership,
            _publish_turn_process_ownership,
            _reap_disconnected_agent_processes,
        )
        from tools.process_registry import process_registry

        calls = []
        monkeypatch.setattr(
            process_registry,
            "kill_started_since",
            lambda *a, **k: calls.append(True) or 1,
        )
        monkeypatch.setattr(
            process_registry, "snapshot_running_ids", lambda _tid: frozenset()
        )

        run = types.SimpleNamespace()
        _publish_turn_process_ownership(run, "solo-session")
        # Simulate the disconnect handler capturing the agent while the
        # worker's finally clears ownership: snapshot markers, then clear.
        stale_view = types.SimpleNamespace(
            _gateway_turn_process_task_id=run._gateway_turn_process_task_id,
            _gateway_turn_process_baseline=run._gateway_turn_process_baseline,
            _gateway_turn_process_epoch=run._gateway_turn_process_epoch,
        )
        _clear_turn_process_ownership(run)

        _reap_disconnected_agent_processes(stale_view)
        deadline = time.time() + 1.0
        while not calls and time.time() < deadline:
            time.sleep(0.01)
        assert calls == [True]

    def test_publish_and_clear_ownership_roundtrip(self, monkeypatch):
        from gateway.platforms.api_server import (
            _TURN_PROCESS_EPOCHS,
            _clear_turn_process_ownership,
            _publish_turn_process_ownership,
        )
        from tools.process_registry import process_registry

        monkeypatch.setattr(
            process_registry,
            "snapshot_running_ids",
            lambda tid: frozenset({f"pre-{tid}"}),
        )

        agent = types.SimpleNamespace()
        _publish_turn_process_ownership(agent, "sess-rt")
        assert agent._gateway_turn_process_task_id == "sess-rt"
        assert agent._gateway_turn_process_baseline == frozenset({"pre-sess-rt"})
        assert isinstance(agent._gateway_turn_process_epoch, int)
        assert "sess-rt" in _TURN_PROCESS_EPOCHS

        _clear_turn_process_ownership(agent)
        assert agent._gateway_turn_process_task_id == ""
        assert agent._gateway_turn_process_baseline == frozenset()
        assert agent._gateway_turn_process_epoch is None
        # Entry pruned — dict stays bounded to in-flight runs.
        assert "sess-rt" not in _TURN_PROCESS_EPOCHS

    @pytest.mark.asyncio
    async def test_stop_run_reaps_owned_processes(self, adapter, monkeypatch):
        """POST /v1/runs/{id}/stop abandons the run — it must reap the
        background processes that run created (#76115 sibling surface)."""
        from gateway.platforms.api_server import _publish_turn_process_ownership
        from tools.process_registry import process_registry

        calls = []
        monkeypatch.setattr(
            process_registry,
            "kill_started_since",
            lambda task_id, baseline, *, source: calls.append(
                (task_id, baseline, source)
            )
            or 1,
        )
        monkeypatch.setattr(
            process_registry, "snapshot_running_ids", lambda _tid: frozenset()
        )

        agent = MagicMock()
        _publish_turn_process_ownership(agent, "run-stop-sess")
        adapter._active_run_agents["run_x"] = agent

        request = MagicMock()
        request.match_info = {"run_id": "run_x"}
        resp = await adapter._handle_stop_run(request)
        assert resp.status == 200

        deadline = time.time() + 1.0
        while not calls and time.time() < deadline:
            time.sleep(0.01)
        assert calls == [("run-stop-sess", frozenset(), "api_server_run_stop")]
        agent.interrupt.assert_called_once()


class TestRunEventCallback:

    @pytest.mark.asyncio
    async def test_subagent_events_redact_secrets_and_carry_child_session(self, adapter):
        """Free-text fields (goal/summary/output_tail/preview) must pass the
        forced secret redaction before hitting the public /v1/runs stream,
        and child_session_id must survive the allowlist so clients can
        correlate the child's session."""
        run_id = "run_subagent_redact"
        loop = asyncio.get_running_loop()
        queue = asyncio.Queue()
        adapter._run_streams[run_id] = queue
        adapter._run_statuses.pop(run_id, None)

        callback = adapter._make_run_event_callback(run_id, loop)
        secret = "sk-proj-abcdef1234567890abcdef1234567890abcdef12"
        callback(
            "subagent.complete",
            preview=f"leaked {secret}",
            goal=f"use key {secret} to fetch data",
            subagent_id="deleg_999",
            child_session_id="child-sess-42",
            status="completed",
            summary=f"exported OPENAI_API_KEY={secret} then ran",
            output_tail=f"env shows {secret}",
        )

        event = await asyncio.wait_for(queue.get(), timeout=1.0)
        assert event["child_session_id"] == "child-sess-42"
        for field in ("preview", "goal", "summary", "output_tail"):
            assert secret not in event[field], field


# ---------------------------------------------------------------------------
# /health endpoint
# ---------------------------------------------------------------------------


class TestHealthEndpoint:
    @pytest.mark.asyncio
    async def test_security_headers_present(self, adapter):
        """Responses should include basic security headers."""
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.get("/health")
            assert resp.status == 200
            assert resp.headers.get("Content-Security-Policy") == "default-src 'none'; frame-ancestors 'none'"
            assert resp.headers.get("Permissions-Policy") == "camera=(), microphone=(), geolocation=()"
            assert resp.headers.get("Strict-Transport-Security") == "max-age=31536000; includeSubDomains"
            assert resp.headers.get("X-Content-Type-Options") == "nosniff"
            assert resp.headers.get("X-Frame-Options") == "DENY"
            assert resp.headers.get("X-XSS-Protection") == "0"
            assert resp.headers.get("Referrer-Policy") == "no-referrer"


    @pytest.mark.asyncio
    async def test_health_reports_version(self, adapter):
        """GET /health must expose a non-empty version so orchestrators (e.g.
        AgentOS) can read the gateway version without scraping. Regression
        guard for the missing-version gap."""
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.get("/health")
            assert resp.status == 200
            data = await resp.json()
            assert "version" in data
            assert isinstance(data["version"], str)
            assert data["version"] != ""


# ---------------------------------------------------------------------------
# /health/detailed endpoint
# ---------------------------------------------------------------------------


class TestHealthDetailedEndpoint:
    @pytest.mark.asyncio
    async def test_health_detailed_returns_ok(self, adapter):
        """GET /health/detailed returns status, platform, and runtime fields."""
        app = _create_app(adapter)
        with patch("gateway.status.read_runtime_status", return_value={
            "gateway_state": "running",
            "platforms": {"telegram": {"state": "connected"}},
            "active_agents": 2,
            "exit_reason": None,
            "updated_at": "2026-04-14T00:00:00Z",
        }), patch("gateway.run._resolve_gateway_model", return_value="test/model"):
            async with TestClient(TestServer(app)) as cli:
                resp = await cli.get("/health/detailed")
                assert resp.status == 200
                data = await resp.json()
                assert data["status"] == "ok"
                assert data["platform"] == "hermes-agent"
                assert data["gateway_state"] == "running"
                assert data["platforms"] == {"telegram": {"state": "connected"}}
                assert data["active_agents"] == 2
                # Derived busy/drainable: this endpoint is served BY the live
                # gateway, so running + 2 agents ⇒ busy and drainable.
                assert data["gateway_busy"] is True
                assert data["gateway_drainable"] is True
                assert isinstance(data["pid"], int)
                assert "updated_at" in data


    @pytest.mark.asyncio
    async def test_public_health_does_not_run_readiness_probes(self, adapter):
        app = _create_app(adapter)
        with patch("gateway.platforms.api_server.collect_runtime_readiness") as probe:
            async with TestClient(TestServer(app)) as cli:
                resp = await cli.get("/health")
                assert resp.status == 200
                assert (await resp.json())["status"] == "ok"
        probe.assert_not_called()


    def test_readiness_work_counts_include_stopping_runs(self, adapter):
        """Regression: _handle_stop_run() sets status="stopping" and holds it
        there — cooperatively, with no hard timeout — until the agent notices
        the interrupt and the task actually exits. A run in that window is
        still doing real executor-thread work and must count as active,
        the same as "running"; excluding it undercounts active_api_runs for
        the whole (now-unbounded) cooperative-stop duration."""
        adapter._run_statuses = {
            "queued": {"status": "queued"},
            "running": {"status": "running"},
            "approval": {"status": "waiting_for_approval"},
            "stopping": {"status": "stopping"},
            "done": {"status": "completed"},
            "cancelled": {"status": "cancelled"},
        }

        with patch("tools.process_registry.process_registry.completion_queue.qsize", return_value=0), \
             patch("tools.async_delegation.active_count", return_value=0):
            assert adapter._readiness_work_counts() == (4, 0, 0)


# ---------------------------------------------------------------------------
# /v1/models endpoint
# ---------------------------------------------------------------------------


class TestModelsEndpoint:
    @pytest.mark.asyncio
    async def test_models_returns_hermes_agent(self, adapter):
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.get("/v1/models")
            assert resp.status == 200
            data = await resp.json()
            assert data["object"] == "list"
            assert len(data["data"]) == 1
            assert data["data"][0]["id"] == "hermes-agent"
            assert data["data"][0]["owned_by"] == "hermes"

    @pytest.mark.asyncio
    async def test_models_returns_profile_name(self):
        """When running under a named profile, /v1/models advertises the profile name."""
        with patch("gateway.platforms.api_server.APIServerAdapter._resolve_model_name", return_value="lucas"):
            adapter = _make_adapter()
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.get("/v1/models")
            assert resp.status == 200
            data = await resp.json()
            assert data["data"][0]["id"] == "lucas"
            assert data["data"][0]["root"] == "lucas"


    def test_resolve_model_name_default_profile(self):
        """Default profile falls back to 'hermes-agent'."""
        with patch("hermes_cli.profiles.get_active_profile_name", return_value="default"):
            assert APIServerAdapter._resolve_model_name("") == "hermes-agent"


    @pytest.mark.asyncio
    async def test_model_options_returns_shared_inventory(self, adapter, monkeypatch):
        """GET /api/model/options builds the shared picker payload off-loop."""
        from hermes_cli import inventory

        ctx = object()
        payload = {
            "providers": [{"slug": "nous", "name": "Nous Portal", "models": ["gpt-5.5"]}],
            "model": "gpt-5.5",
            "provider": "nous",
        }
        seen = {"thread_calls": 0}

        monkeypatch.setattr(inventory, "load_picker_context", lambda: ctx)

        def fake_build_model_options_payload(received_ctx, **kwargs):
            seen["ctx"] = received_ctx
            seen["kwargs"] = kwargs
            return payload

        async def fake_to_thread(func, *args, **kwargs):
            seen["thread_calls"] += 1
            return func(*args, **kwargs)

        monkeypatch.setattr(
            inventory,
            "build_model_options_payload",
            fake_build_model_options_payload,
        )
        monkeypatch.setattr(
            "gateway.platforms.api_server.asyncio.to_thread",
            fake_to_thread,
        )

        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.get("/api/model/options?refresh=true")
            assert resp.status == 200
            data = await resp.json()

        assert data == payload
        assert seen["thread_calls"] == 1
        assert seen["ctx"] is ctx
        assert seen["kwargs"] == {
            "include_unconfigured": True,
            "refresh": True,
        }


# ---------------------------------------------------------------------------
# /v1/capabilities endpoint
# ---------------------------------------------------------------------------


class TestCapabilitiesEndpoint:
    @pytest.mark.asyncio
    async def test_capabilities_advertises_plugin_safe_contract(self, adapter):
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.get("/v1/capabilities")
            assert resp.status == 200
            data = await resp.json()
            assert data["object"] == "hermes.api_server.capabilities"
            assert data["platform"] == "hermes-agent"
            assert data["model"] == "hermes-agent"
            assert data["auth"]["type"] == "bearer"
            assert data["auth"]["required"] is False
            assert data["runtime"]["mode"] == "server_agent"
            assert data["runtime"]["tool_execution"] == "server"
            assert data["runtime"]["split_runtime"] is False
            assert "API-server host" in data["runtime"]["description"]
            assert data["features"]["chat_completions"] is True
            assert data["features"]["run_status"] is True
            assert data["features"]["run_events_sse"] is True
            assert data["features"]["model_options"] is True
            assert data["features"]["session_continuity_header"] == "X-Hermes-Session-Id"
            assert data["endpoints"]["run_status"]["path"] == "/v1/runs/{run_id}"
            assert data["endpoints"]["model_options"] == {"method": "GET", "path": "/api/model/options"}
            assert data["endpoints"]["skills"] == {"method": "GET", "path": "/v1/skills"}
            assert data["endpoints"]["toolsets"] == {"method": "GET", "path": "/v1/toolsets"}


# ---------------------------------------------------------------------------
# /v1/owner-workspace/projects endpoint
# ---------------------------------------------------------------------------


class TestOwnerWorkspaceProjectsEndpoint:
    @pytest.mark.asyncio
    async def test_disabled_surface_is_not_discoverable(self, adapter):
        app = _create_app(adapter)
        with patch("gateway.run._load_gateway_config", return_value={}):
            async with TestClient(TestServer(app)) as cli:
                resp = await cli.get("/v1/owner-workspace/projects")
                data = await resp.json()

        assert resp.status == 404
        assert data["error"]["code"] == "owner_workspace_not_enabled"

    @pytest.mark.asyncio
    async def test_enabled_surface_returns_only_kernel_projection(self, adapter):
        expected = [{
            "project_id": "p_1",
            "slug": "shoe-shop",
            "name": "Shoe shop",
            "description": "Owner project",
            "board": "shoe-shop",
            "archived": False,
        }]
        app = _create_app(adapter)
        config = {"gateway": {"api_server": {"owner_workspace": {"enabled": True}}}}
        with (
            patch("gateway.run._load_gateway_config", return_value=config),
            patch("hermes_cli.owner_workspace.resolve_owner_context", return_value=object()),
            patch("hermes_cli.owner_workspace.list_committed_projects", return_value=expected) as listed,
        ):
            async with TestClient(TestServer(app)) as cli:
                resp = await cli.get("/v1/owner-workspace/projects")
                data = await resp.json()

        assert resp.status == 200
        assert data == {
            "object": "hermes.owner_workspace.project_list",
            "data": expected,
        }
        listed.assert_called_once()

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "path",
        [
            "/v1/owner-workspace/projects",
            "/v1/owner-workspace/projects/shoe-shop/snapshot",
            "/v1/owner-workspace/projects/shoe-shop/attachments/7",
        ],
    )
    async def test_surface_requires_bearer_auth(self, auth_adapter, path):
        app = _create_app(auth_adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.get(path)

        assert resp.status == 401


class TestOwnerWorkspaceProjectSnapshotEndpoint:
    @pytest.mark.asyncio
    async def test_enabled_surface_returns_only_exact_kernel_projection(self, adapter):
        expected = {
            "project": {
                "id": "p_1", "slug": "shoe-shop", "name": "Shoe shop",
                "description": "Owner project", "board": "shoe-shop", "archived": False,
            },
            "board": {
                "slug": "shoe-shop", "name": "Shoe shop", "project_id": "p_1",
                "counts": {"ready": 1}, "total": 1,
            },
            "columns": [{"name": "ready", "tasks": []}],
            "workers": [], "attachments": [], "runs": [],
            "truncated": {"tasks": False, "workers": False, "attachments": False, "runs": False},
        }
        app = _create_app(adapter)
        config = {"gateway": {"api_server": {"owner_workspace": {"enabled": True}}}}
        with (
            patch("gateway.run._load_gateway_config", return_value=config),
            patch("hermes_cli.owner_workspace.resolve_owner_context", return_value=object()),
            patch("hermes_cli.owner_workspace.read_project_snapshot", return_value=expected) as read,
        ):
            async with TestClient(TestServer(app)) as cli:
                resp = await cli.get("/v1/owner-workspace/projects/shoe-shop/snapshot")
                data = await resp.json()

        assert resp.status == 200
        assert data == {
            "object": "hermes.owner_workspace.project_snapshot",
            "data": expected,
        }
        read.assert_called_once_with(
            ANY,
            "shoe-shop",
            run_context=False,
            planning_context=False,
        )

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "query,expected_run_context,expected_planning_context",
        [
            ("", False, False),
            ("?capabilities=", False, False),
            ("?capabilities=something_else", False, False),
            ("?capabilities=run_task_context", True, False),
            ("?capabilities=planning_context_v1", False, True),
            ("?capabilities=run_task_context,planning_context_v1", True, True),
            ("?capabilities=something_else,run_task_context", True, False),
            ("?capabilities=run_task_context_extra", False, False),
            ("?capabilities=" + "x" * 300 + ",run_task_context", False, False),
            # A repeated key has no single negotiated answer, so BOTH orders
            # fail closed to the legacy shape rather than granting whichever
            # occurrence happens to be read first.
            ("?capabilities=run_task_context&capabilities=something_else", False, False),
            ("?capabilities=something_else&capabilities=run_task_context", False, False),
            ("?capabilities=run_task_context&capabilities=run_task_context", False, False),
            # The native board/workers capability is a different contract on a
            # different surface; naming it here grants nothing.
            ("?capabilities=owner_titles_v1", False, False),
        ],
    )
    async def test_contexts_are_served_only_to_a_reader_that_asks_for_them(
        self,
        adapter,
        query,
        expected_run_context,
        expected_planning_context,
    ):
        """The /v1 run shape stays what its oldest reader validates.

        A reader that predates the added run keys never sends
        ``capabilities``, so deploying this Hermes first cannot hand that
        reader a snapshot its closed schema rejects. An oversized, repeated,
        or foreign-surface parameter grants nothing rather than being parsed.
        """
        app = _create_app(adapter)
        config = {"gateway": {"api_server": {"owner_workspace": {"enabled": True}}}}
        with (
            patch("gateway.run._load_gateway_config", return_value=config),
            patch("hermes_cli.owner_workspace.resolve_owner_context", return_value=object()),
            patch("hermes_cli.owner_workspace.read_project_snapshot", return_value={}) as read,
        ):
            async with TestClient(TestServer(app)) as cli:
                resp = await cli.get(
                    f"/v1/owner-workspace/projects/shoe-shop/snapshot{query}"
                )

        assert resp.status == 200
        read.assert_called_once_with(
            ANY,
            "shoe-shop",
            run_context=expected_run_context,
            planning_context=expected_planning_context,
        )

    @pytest.mark.asyncio
    async def test_non_receipt_project_is_indistinguishable_from_missing(self, adapter):
        from hermes_cli.owner_workspace import OwnerWorkspaceError

        app = _create_app(adapter)
        config = {"gateway": {"api_server": {"owner_workspace": {"enabled": True}}}}
        with (
            patch("gateway.run._load_gateway_config", return_value=config),
            patch("hermes_cli.owner_workspace.resolve_owner_context", return_value=object()),
            patch(
                "hermes_cli.owner_workspace.read_project_snapshot",
                side_effect=OwnerWorkspaceError("project_not_found", "private detail"),
            ),
        ):
            async with TestClient(TestServer(app)) as cli:
                resp = await cli.get("/v1/owner-workspace/projects/not-owned/snapshot")
                data = await resp.json()

        assert resp.status == 404
        assert data["error"]["code"] == "project_not_found"
        assert "private detail" not in str(data)

    @pytest.mark.asyncio
    async def test_attachment_stream_is_exact_and_non_cacheable(self, adapter):
        expected = {
            "id": "7",
            "filename": "owner-note.txt",
            "media_type": "text/plain",
            "size": 16,
            "created_at": "2026-08-21T00:00:00Z",
            "body": b"safe owner bytes",
        }
        app = _create_app(adapter)
        config = {"gateway": {"api_server": {"owner_workspace": {"enabled": True}}}}
        with (
            patch("gateway.run._load_gateway_config", return_value=config),
            patch("hermes_cli.owner_workspace.resolve_owner_context", return_value=object()),
            patch("hermes_cli.owner_workspace.read_project_attachment", return_value=expected) as read,
        ):
            async with TestClient(TestServer(app)) as cli:
                resp = await cli.get(
                    "/v1/owner-workspace/projects/shoe-shop/attachments/7"
                )
                body = await resp.read()

        assert resp.status == 200
        assert body == b"safe owner bytes"
        assert resp.headers["Content-Type"].startswith("text/plain")
        assert resp.headers["Content-Disposition"] == 'attachment; filename="owner-note.txt"'
        assert resp.headers["Cache-Control"] == "no-store"
        read.assert_called_once_with(ANY, "shoe-shop", "7")


class TestOwnerWorkspaceDecisionsEndpoint:
    @pytest.mark.asyncio
    async def test_enabled_surface_combines_durable_and_active_native_gates(self, adapter):
        durable = [{
            "decision_ref": "decision_task_safe",
            "authority": "task",
            "kind": "owner_input",
            "project_slug": "workshop-pilot",
            "project_name": "Workshop pilot",
            "title": "Choose the workshop date",
            "reason": "Raphael needs your answer before this work can continue.",
            "created_at": "2026-08-20T12:00:00Z",
        }]
        adapter._run_statuses["run_private_native_id"] = {
            "run_id": "run_private_native_id",
            "status": "waiting_for_approval",
            "created_at": 1_787_227_200,
            "pending_approval": {
                "approval_id": "approval_private_native_id",
                "operation": "owner_project_plan_commit",
                "description": "private internal operation detail",
            },
            "owner_workspace_context": {
                "mode": "existing",
                "project_slug": "workshop-pilot",
                "project_name": "Workshop pilot",
                "profile": "default",
            },
        }
        app = _create_app(adapter)
        config = {"gateway": {"api_server": {"owner_workspace": {"enabled": True}}}}
        with (
            patch("gateway.run._load_gateway_config", return_value=config),
            patch(
                "hermes_cli.owner_workspace.resolve_owner_context",
                return_value=types.SimpleNamespace(profile="default"),
            ),
            patch(
                "hermes_cli.owner_workspace.list_owner_decisions",
                return_value={"data": durable, "truncated": False},
            ),
        ):
            async with TestClient(TestServer(app)) as cli:
                resp = await cli.get("/v1/owner-workspace/decisions")
                data = await resp.json()

        assert resp.status == 200
        assert data["object"] == "hermes.owner_workspace.decision_list"
        assert data["truncated"] is False
        assert data["data"][0] == durable[0]
        assert data["data"][1] == {
            "decision_ref": data["data"][1]["decision_ref"],
            "authority": "run",
            "kind": "run_approval",
            "project_slug": "workshop-pilot",
            "project_name": "Workshop pilot",
            "title": "Approve Project changes",
            "reason": "Raphael is waiting for your confirmation before changing this Project.",
            "created_at": "2026-08-20T12:00:00Z",
        }
        assert data["data"][1]["decision_ref"].startswith("decision_")
        assert "run_private_native_id" not in json.dumps(data)
        assert "approval_private_native_id" not in json.dumps(data)
        assert "private internal operation detail" not in json.dumps(data)

    @pytest.mark.asyncio
    async def test_surface_requires_bearer_auth(self, auth_adapter):
        app = _create_app(auth_adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.get("/v1/owner-workspace/decisions")
        assert resp.status == 401


# ---------------------------------------------------------------------------
# /v1/skills and /v1/toolsets endpoints
# ---------------------------------------------------------------------------


class TestSkillsEndpoint:
    @pytest.mark.asyncio
    async def test_skills_returns_list_envelope(self, adapter):
        fake_skills = [
            {"name": "github", "description": "GitHub workflow skill", "category": "github"},
            {"name": "ascii-art", "description": "ASCII art generation", "category": "creative"},
        ]
        with patch(
            "tools.skills_tool._find_all_skills",
            return_value=list(fake_skills),
        ):
            app = _create_app(adapter)
            async with TestClient(TestServer(app)) as cli:
                resp = await cli.get("/v1/skills")
                assert resp.status == 200
                data = await resp.json()
                assert data["object"] == "list"
                names = sorted(s["name"] for s in data["data"])
                assert names == ["ascii-art", "github"]
                for entry in data["data"]:
                    assert set(entry.keys()) >= {"name", "description", "category"}


class TestToolsetsEndpoint:
    @pytest.mark.asyncio
    async def test_toolsets_returns_resolved_tools(self, adapter):
        fake_toolsets = [
            ("default", "Default Tools", "Core tools"),
            ("web", "Web Tools", "Search and extract"),
        ]
        feature_snapshot = object()
        with patch(
            "hermes_cli.tools_config._get_effective_configurable_toolsets",
            return_value=fake_toolsets,
        ), patch(
            "hermes_cli.tools_config._get_platform_tools",
            return_value={"default"},
        ), patch(
            "hermes_cli.tools_config.get_nous_subscription_features",
            return_value=feature_snapshot,
        ) as resolve_features, patch(
            "hermes_cli.tools_config._toolset_has_keys",
            return_value=True,
        ) as has_keys, patch(
            "toolsets.resolve_toolset",
            side_effect=lambda name: {
                "default": ["terminal", "read_file"],
                "web": ["web_search"],
            }[name],
        ):
            app = _create_app(adapter)
            async with TestClient(TestServer(app)) as cli:
                resp = await cli.get("/v1/toolsets")
                assert resp.status == 200
                data = await resp.json()
                assert data["object"] == "list"
                assert data["platform"] == "api_server"
                by_name = {ts["name"]: ts for ts in data["data"]}
                assert by_name["default"]["enabled"] is True
                assert by_name["default"]["tools"] == ["read_file", "terminal"]
                assert by_name["web"]["enabled"] is False
                assert by_name["web"]["tools"] == ["web_search"]
                assert by_name["default"]["configured"] is True

        resolve_features.assert_called_once()
        assert has_keys.call_count == len(fake_toolsets)
        assert all(
            call.kwargs["features"] is feature_snapshot
            for call in has_keys.call_args_list
        )


# ---------------------------------------------------------------------------
# /v1/chat/completions endpoint
# ---------------------------------------------------------------------------


class TestChatCompletionsEndpoint:
    @pytest.mark.asyncio
    async def test_invalid_json_returns_400(self, adapter):
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(
                "/v1/chat/completions",
                data="not json",
                headers={"Content-Type": "application/json"},
            )
            assert resp.status == 400
            data = await resp.json()
            assert "Invalid JSON" in data["error"]["message"]

    @pytest.mark.asyncio
    async def test_missing_messages_returns_400(self, adapter):
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post("/v1/chat/completions", json={"model": "test"})
            assert resp.status == 400
            data = await resp.json()
            assert "messages" in data["error"]["message"]


    @pytest.mark.asyncio
    async def test_chat_completions_stream_passes_request_model_provider_options(self, adapter):
        app = _create_app(adapter)
        model_options = {"reasoning": {"enabled": False}, "reasoning_effort": "none", "fast": False}

        async def _mock_run_agent(**kwargs):
            cb = kwargs.get("stream_delta_callback")
            if cb:
                cb("ok")
            return (
                {"final_response": "ok", "messages": [], "api_calls": 1},
                {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
            )

        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_run_agent", side_effect=_mock_run_agent) as mock_run:
                resp = await cli.post(
                    "/v1/chat/completions",
                    json={
                        "model": "MiniMax-M3",
                        "provider": "minimax",
                        "model_options": model_options,
                        "messages": [{"role": "user", "content": "hi"}],
                        "stream": True,
                    },
                )
                assert resp.status == 200
                body = await resp.text()

        assert "data: " in body
        kwargs = mock_run.call_args.kwargs
        assert kwargs["requested_model"] == "MiniMax-M3"
        assert kwargs["requested_provider"] == "minimax"
        assert kwargs["model_options"] == model_options


    @pytest.mark.asyncio
    async def test_session_chat_stream_passes_request_model_provider_options(self, adapter):
        app = _create_app(adapter)
        model_options = {"reasoning_effort": "medium", "service_tier": "priority"}
        async with TestClient(TestServer(app)) as cli:
            with (
                patch.object(adapter, "_get_existing_session_or_404", return_value=({"id": "s1"}, None)),
                patch.object(adapter, "_conversation_history_for_session", return_value=[]),
                patch.object(adapter, "_run_agent", new_callable=AsyncMock) as mock_run,
            ):
                mock_run.return_value = (
                    {"final_response": "ok", "messages": [], "api_calls": 1},
                    {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
                )
                resp = await cli.post(
                    "/api/sessions/s1/chat/stream",
                    json={
                        "message": "hi",
                        "model": "MiniMax-M3",
                        "provider": "minimax",
                        "model_options": model_options,
                    },
                )
                assert resp.status == 200
                body = await resp.text()

        assert "event: run.completed" in body
        kwargs = mock_run.call_args.kwargs
        assert kwargs["requested_model"] == "MiniMax-M3"
        assert kwargs["requested_provider"] == "minimax"
        assert kwargs["model_options"] == model_options


    @pytest.mark.asyncio
    async def test_stream_task_done_callback_enqueues_eos_for_chat_completions(self, adapter):
        """Regression guard for #24451: completion callback must signal SSE EOS."""
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            class _FakeTask:
                def __init__(self):
                    self.callbacks = []

                def add_done_callback(self, cb):
                    self.callbacks.append(cb)

            fake_task = _FakeTask()

            def _fake_ensure_future(coro):
                # We short-circuit task scheduling in this unit test.
                coro.close()
                return fake_task

            with (
                patch.object(
                    adapter,
                    "_run_agent",
                    new=AsyncMock(
                        return_value=(
                            {"final_response": "ok", "messages": [], "api_calls": 1},
                            {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
                        )
                    ),
                ),
                patch("gateway.platforms.api_server.asyncio.ensure_future", side_effect=_fake_ensure_future),
                patch.object(adapter, "_write_sse_chat_completion", new_callable=AsyncMock) as mock_write_sse,
            ):
                mock_write_sse.return_value = web.Response(status=200, text="ok")
                resp = await cli.post(
                    "/v1/chat/completions",
                    json={
                        "model": "test",
                        "messages": [{"role": "user", "content": "hi"}],
                        "stream": True,
                    },
                )
                assert resp.status == 200

            assert len(fake_task.callbacks) == 1
            stream_q = mock_write_sse.call_args.args[4]
            assert stream_q.empty()
            fake_task.callbacks[0](fake_task)
            assert stream_q.get_nowait() is None


    @pytest.mark.asyncio
    async def test_stream_includes_tool_progress(self, adapter):
        """tool_start_callback fires → progress appears as custom SSE event, not in delta.content."""
        import asyncio

        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            async def _mock_run_agent(**kwargs):
                cb = kwargs.get("stream_delta_callback")
                ts_cb = kwargs.get("tool_start_callback")
                # Simulate the structured tool start the gateway now consumes.
                if ts_cb:
                    ts_cb("call_terminal_1", "terminal", {"command": "ls -la"})
                if cb:
                    await asyncio.sleep(0.05)
                    cb("Here are the files.")
                return (
                    {"final_response": "Here are the files.", "messages": [], "api_calls": 1},
                    {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
                )

            with patch.object(adapter, "_run_agent", side_effect=_mock_run_agent):
                resp = await cli.post(
                    "/v1/chat/completions",
                    json={
                        "model": "test",
                        "messages": [{"role": "user", "content": "list files"}],
                        "stream": True,
                    },
                )
                assert resp.status == 200
                body = await resp.text()
                assert "[DONE]" in body
                # Tool progress must appear as a custom SSE event, not in
                # delta.content — prevents model from learning to imitate
                # markers instead of calling tools (#6972).
                assert "event: hermes.tool.progress" in body
                assert '"tool": "terminal"' in body
                # ``label`` is now derived by ``build_tool_preview`` from the
                # tool args rather than passed by the caller, so we assert
                # only that *some* label exists rather than a literal value.
                assert '"label":' in body
                # The progress marker must NOT appear inside any
                # chat.completion.chunk delta.content field.
                import json as _json
                for line in body.splitlines():
                    if line.startswith("data: ") and line.strip() != "data: [DONE]":
                        try:
                            chunk = _json.loads(line[len("data: "):])
                        except _json.JSONDecodeError:
                            continue
                        if chunk.get("object") == "chat.completion.chunk":
                            for choice in chunk.get("choices", []):
                                content = choice.get("delta", {}).get("content", "")
                                # Tool emoji markers must never leak into content
                                assert "ls -la" not in content or content == "Here are the files."
                # Final content must also be present
                assert "Here are the files." in body


    @pytest.mark.asyncio
    async def test_stream_emits_tool_lifecycle_with_call_id(self, adapter):
        """Regression for #16588.

        ``/v1/chat/completions`` streaming previously emitted only a
        ``tool.started``-style ``hermes.tool.progress`` event; clients
        rendering tool lifecycle UI had no way to mark a tool as finished
        because no matching ``status: completed`` event was emitted, and
        no ``toolCallId`` was carried for correlation.

        The fix adds ``tool_start_callback`` / ``tool_complete_callback``
        to the chat completions agent invocation and writes both halves
        of the lifecycle pair on the same ``event: hermes.tool.progress``
        SSE line, with stable ``toolCallId`` and ``status``.
        """
        import asyncio
        import json as _json

        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            async def _mock_run_agent(**kwargs):
                cb = kwargs.get("stream_delta_callback")
                ts_cb = kwargs.get("tool_start_callback")
                tc_cb = kwargs.get("tool_complete_callback")
                # The structured callbacks own the chat-completions SSE
                # channel now; ``tool_progress_callback`` is intentionally
                # not wired so each tool start emits exactly one event.
                if ts_cb:
                    ts_cb("call_terminal_1", "terminal", {"command": "ls -la"})
                if tc_cb:
                    tc_cb("call_terminal_1", "terminal", {"command": "ls -la"}, "ok")
                if cb:
                    await asyncio.sleep(0.05)
                    cb("done.")
                return (
                    {"final_response": "done.", "messages": [], "api_calls": 1},
                    {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
                )

            with patch.object(adapter, "_run_agent", side_effect=_mock_run_agent):
                resp = await cli.post(
                    "/v1/chat/completions",
                    json={
                        "model": "test",
                        "messages": [{"role": "user", "content": "list"}],
                        "stream": True,
                    },
                )
                assert resp.status == 200
                body = await resp.text()

            # Walk the SSE body and collect *(status, toolCallId)* pairs
            # per event so the assertions verify per-event correlation —
            # an event missing ``toolCallId`` would not pass even if a
            # different event happens to carry the right id.
            pairs: list[tuple[str | None, str | None]] = []
            lines = body.splitlines()
            for i, line in enumerate(lines):
                if line.strip() != "event: hermes.tool.progress":
                    continue
                for follow in lines[i + 1: i + 4]:
                    if follow.startswith("data: "):
                        try:
                            payload = _json.loads(follow[len("data: "):])
                        except _json.JSONDecodeError:
                            break
                        pairs.append((payload.get("status"), payload.get("toolCallId")))
                        break

            # Each tool start must emit exactly one event (no duplicate
            # legacy + new emit), and each lifecycle pair must carry the
            # same toolCallId on every event — not just somewhere in the
            # aggregate.
            assert len(pairs) == 2, f"expected 2 events (running+completed), got {pairs}"
            assert pairs[0] == ("running", "call_terminal_1"), pairs
            assert pairs[1] == ("completed", "call_terminal_1"), pairs

    @pytest.mark.asyncio
    async def test_stream_tool_lifecycle_skips_internal_and_orphan_completes(self, adapter):
        """Internal tools (``_thinking``-style) and ``completed`` events
        without a prior matching ``running`` must produce no lifecycle
        events on the wire — otherwise clients would see orphaned
        ``status: completed`` updates they cannot correlate."""
        import asyncio

        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            async def _mock_run_agent(**kwargs):
                cb = kwargs.get("stream_delta_callback")
                ts_cb = kwargs.get("tool_start_callback")
                tc_cb = kwargs.get("tool_complete_callback")
                # Internal tool — must be filtered.
                if ts_cb:
                    ts_cb("call_internal_1", "_thinking", {})
                if tc_cb:
                    tc_cb("call_internal_1", "_thinking", {}, "")
                # Completion without start — orphan, must be dropped.
                if tc_cb:
                    tc_cb("call_orphan_1", "web_search", {}, "ok")
                if cb:
                    await asyncio.sleep(0.05)
                    cb("ok.")
                return (
                    {"final_response": "ok.", "messages": [], "api_calls": 1},
                    {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
                )

            with patch.object(adapter, "_run_agent", side_effect=_mock_run_agent):
                resp = await cli.post(
                    "/v1/chat/completions",
                    json={
                        "model": "test",
                        "messages": [{"role": "user", "content": "ok"}],
                        "stream": True,
                    },
                )
                assert resp.status == 200
                body = await resp.text()

            # Neither the internal call_id nor the orphan call_id should
            # surface as a lifecycle payload on the wire.
            assert "call_internal_1" not in body
            assert "call_orphan_1" not in body
            assert '"status": "running"' not in body
            assert '"status": "completed"' not in body


# ---------------------------------------------------------------------------
# _derive_chat_session_id unit tests
# ---------------------------------------------------------------------------


class TestDeriveChatSessionId:
    def test_deterministic(self):
        """Same inputs always produce the same session ID."""
        a = _derive_chat_session_id("sys", "hello")
        b = _derive_chat_session_id("sys", "hello")
        assert a == b


    def test_different_system_prompt(self):
        a = _derive_chat_session_id("You are a pirate.", "Hello")
        b = _derive_chat_session_id("You are a robot.", "Hello")
        assert a != b


# ---------------------------------------------------------------------------
# /v1/responses endpoint
# ---------------------------------------------------------------------------


class TestResponsesEndpoint:

    @pytest.mark.asyncio
    async def test_owner_history_projects_the_turn_being_planned_right_now(
        self, adapter,
    ):
        """Item 32TK: the request survives at the boundary, not just in the store.

        The Workspace reads this endpoint. A browser that never received the
        accept response has no handle to the plan at all, so this projection is
        the only thing that can give the owner back their own words — and the
        response id that is already planning them.
        """
        conversation = "raphael-owner-" + "d" * 32
        assert adapter._response_store.reserve_owner_conversation(
            "default",
            conversation,
            "resp_pending_at_the_boundary",
            owner_message="Prepare a private 60-minute workshop",
        ) is True

        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            history_response = await cli.get(
                f"/v1/responses/conversations/{conversation}",
            )
            assert history_response.status == 200
            history = await history_response.json()

        assert history["data"] == []
        assert history["pending"] == {
            "owner": "Prepare a private 60-minute workshop",
            "response_id": "resp_pending_at_the_boundary",
        }

    @pytest.mark.asyncio
    async def test_a_failed_turn_is_recoverable_and_acknowledged_at_the_boundary(
        self, adapter,
    ):
        """Item 32TK: the whole way back, over HTTP, with no caller-held handle.

        The Workspace reads and retires this record across the same public
        boundary it plans turns on. Reading is repeatable because the answer
        carrying it can be lost too; retiring it is a separate, explicit,
        repeatable act.
        """
        conversation = "raphael-owner-" + "3" * 32
        assert adapter._response_store.reserve_owner_conversation(
            "default",
            conversation,
            "resp_failed_at_the_boundary",
            owner_message="Prepare a private 60-minute workshop",
        ) is True
        _interrupt_owner_turn(
            adapter._response_store, conversation, "resp_failed_at_the_boundary",
        )

        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            first = await cli.get(f"/v1/responses/conversations/{conversation}")
            assert first.status == 200
            recovery = (await first.json())["recovery"]
            assert recovery == {
                "owner": "Prepare a private 60-minute workshop",
                "response_id": "resp_failed_at_the_boundary",
            }

            # Reading again still finds it: the answer above may never have
            # arrived, which is the whole reason this record exists.
            again = await cli.get(f"/v1/responses/conversations/{conversation}")
            assert (await again.json())["recovery"] == recovery

            acknowledged = await cli.post(
                f"/v1/responses/conversations/{conversation}/recovery",
                json={"response_id": "resp_failed_at_the_boundary"},
            )
            assert acknowledged.status == 200
            assert (await acknowledged.json())["acknowledged"] is True

            # Retired, and saying so a second time is not an error.
            after = await cli.get(f"/v1/responses/conversations/{conversation}")
            assert (await after.json())["recovery"] is None
            repeated = await cli.post(
                f"/v1/responses/conversations/{conversation}/recovery",
                json={"response_id": "resp_failed_at_the_boundary"},
            )
            assert repeated.status == 200
            assert (await repeated.json())["acknowledged"] is True

    @pytest.mark.asyncio
    async def test_the_recovery_route_says_what_it_actually_did(self, adapter):
        """Item 32TK round 3: "acknowledged" was said even when nothing was.

        A caller cannot verify an answer that is the same whatever happened. It
        now distinguishes the record it really retired, a conversation that has
        nothing outstanding, and a request that names a DIFFERENT outstanding
        record — which is refused outright, because answering it as success
        would tell a caller an unanswered request had been dealt with.
        """
        conversation = "raphael-owner-" + "b" * 32
        store = adapter._response_store
        assert store.reserve_owner_conversation(
            "default", conversation, "resp_truthful_recovery",
            owner_message="Prepare a private 60-minute workshop",
        ) is True
        _interrupt_owner_turn(store, conversation, "resp_truthful_recovery")

        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            path = f"/v1/responses/conversations/{conversation}/recovery"

            # Another request entirely: refused, and it changes nothing.
            mismatched = await cli.post(
                path, json={"response_id": "resp_some_other_request"},
            )
            assert mismatched.status == 409
            body = await mismatched.json()
            assert body["acknowledged"] is False
            assert body["outcome"] == "mismatch"
            assert (await (await cli.get(
                f"/v1/responses/conversations/{conversation}"
            )).json())["recovery"] is not None

            retired = await cli.post(
                path, json={"response_id": "resp_truthful_recovery"},
            )
            assert retired.status == 200
            assert await retired.json() == {
                "object": "hermes.response.owner_recovery_acknowledgement",
                "acknowledged": True,
                "outcome": "retired",
            }

            # Saying it twice is still safe, and now says which it was.
            again = await cli.post(
                path, json={"response_id": "resp_truthful_recovery"},
            )
            assert again.status == 200
            assert await again.json() == {
                "object": "hermes.response.owner_recovery_acknowledgement",
                "acknowledged": True,
                "outcome": "absent",
            }

    @pytest.mark.asyncio
    async def test_owner_session_index_and_consumption_routes(self, adapter):
        group = "raphael-owner-" + "7" * 32
        session_id = "8" * 32
        conversation = f"{group}-{session_id}"
        response_id = "resp_route_proposal"
        reply = json.dumps(_owner_existing_proposal(
            request_title="Finish the owner flow",
        ))
        adapter._response_store.put(response_id, {
            "response": {"id": response_id, "created_at": 800},
            "conversation_history": [
                {"role": "user", "content": "Finish the owner flow."},
                {"role": "assistant", "content": reply},
            ],
        })
        adapter._response_store.set_conversation(
            conversation, response_id, owner_proposal=True,
        )

        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            index_response = await cli.get(
                f"/v1/responses/conversations/{group}?view=sessions",
            )
            assert index_response.status == 200
            assert await index_response.json() == {
                "object": "hermes.response.owner_sessions",
                "data": [{
                    "session_id": session_id,
                    "updated_at": 800,
                    "preview": "Finish the owner flow.",
                    "visible_turn_count": 1,
                    "available": True,
                }],
                "truncated": False,
                "current_session_id": session_id,
            }

            consumed_response = await cli.post(
                f"/v1/responses/conversations/{conversation}/consume",
                json={"response_id": response_id},
            )
            assert consumed_response.status == 200
            assert await consumed_response.json() == {
                "object": "hermes.response.owner_proposal_consumption",
                "consumed": True,
            }

            history_response = await cli.get(
                f"/v1/responses/conversations/{conversation}",
            )
            assert history_response.status == 200
            history = await history_response.json()
            assert history["proposal_consumed"] is True
            assert history["latest_response_id"] == response_id
            retained = await cli.delete(f"/v1/responses/{response_id}")
            assert retained.status == 409
            assert (await retained.json())["error"]["code"] == "owner_conversation_active"

            authority_response_id = "resp_route_authority"
            adapter._response_store.put(authority_response_id, {
                "response": {"id": authority_response_id, "created_at": 801},
                "conversation_history": [
                    {"role": "user", "content": "Apply the exact plan."},
                    {"role": "assistant", "content": reply},
                ],
            })
            assert adapter._response_store.set_conversation(
                conversation, authority_response_id, owner_proposal=True,
            ) is True
            authority_path = (
                f"/v1/responses/conversations/{conversation}/authority"
            )
            claim_id = "claim_" + "a" * 32
            run_id = "run_" + "b" * 32
            claim_payload = {
                "action": "claim",
                "response_id": authority_response_id,
                "claim_id": claim_id,
            }
            authority_response = await cli.post(authority_path, json=claim_payload)
            assert authority_response.status == 200

            abandon_payload = {
                "action": "abandon",
                "response_id": authority_response_id,
                "claim_id": claim_id,
            }
            abandoned = await cli.post(authority_path, json=abandon_payload)
            assert abandoned.status == 200
            assert await abandoned.json() == {
                "object": "hermes.response.owner_authority",
                "action": "abandon",
                "applied": True,
            }
            authority_response = await cli.post(authority_path, json=claim_payload)
            assert authority_response.status == 200

            # The native /v1/runs path, not this transition endpoint, owns the
            # initial binding and live run state.
            adapter._set_run_status(run_id, "running")
            assert adapter._response_store.attach_owner_run(
                "default", conversation, authority_response_id, claim_id, run_id,
            ) is True
            attach_payload = {
                "action": "attach",
                "response_id": authority_response_id,
                "claim_id": claim_id,
                "run_id": run_id,
            }
            authority_response = await cli.post(authority_path, json=attach_payload)
            assert authority_response.status == 200

            for premature_action in ("complete", "release"):
                premature = await cli.post(authority_path, json={
                    "action": premature_action,
                    "response_id": authority_response_id,
                    "claim_id": claim_id,
                    "run_id": run_id,
                })
                assert premature.status == 409

            adapter._set_run_status(run_id, "completed")
            complete_payload = {
                "action": "complete",
                "response_id": authority_response_id,
                "claim_id": claim_id,
                "run_id": run_id,
            }
            # Generic terminal status is not mutation proof. The endpoint is
            # verification-only until the native tool callback closes the
            # exact claim server-side.
            premature_complete = await cli.post(
                authority_path, json=complete_payload,
            )
            assert premature_complete.status == 409
            assert adapter._response_store.complete_owner_claim(
                "default", conversation, authority_response_id, claim_id, run_id,
            ) is True
            authority_response = await cli.post(authority_path, json=complete_payload)
            assert authority_response.status == 200
            assert await authority_response.json() == {
                "object": "hermes.response.owner_authority",
                "action": "complete",
                "applied": True,
            }
            authority_history = await (
                await cli.get(f"/v1/responses/conversations/{conversation}")
            ).json()
            assert authority_history["proposal_consumed"] is True
            assert authority_history["proposal_claimed"] is False
            assert authority_history["active_run_id"] is None

    @pytest.mark.asyncio
    async def test_owner_session_routes_reject_invalid_shapes(self, adapter):
        group = "raphael-owner-" + "9" * 32
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            assert (
                await cli.get(
                    f"/v1/responses/conversations/{group}?view=private",
                )
            ).status == 400
            assert (
                await cli.post(
                    f"/v1/responses/conversations/{group}/consume",
                    json={"response_id": "resp_safe_value", "extra": True},
                )
            ).status == 400
            assert (
                await cli.post(
                    f"/v1/responses/conversations/{group}/authority",
                    json={"action": "claim", "response_id": "resp_safe_value"},
                )
            ).status == 400

    @pytest.mark.asyncio
    async def test_the_close_route_compares_the_conversations_real_head(
        self, adapter,
    ):
        """A stale tab used to close and hide a newer turn, because an ordinary
        question leaves the outstanding proposal untouched."""
        conversation = "raphael-owner-" + "5" * 32
        store = adapter._response_store
        for response_id, created_at in (("resp_close_one", 100), ("resp_close_two", 200)):
            store.put(response_id, {
                "response": {"id": response_id, "created_at": created_at},
                "conversation_history": [],
            })
            assert store.set_conversation(conversation, response_id) is True

        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            stale = await cli.post(
                f"/v1/responses/conversations/{conversation}/authority",
                json={
                    "action": "close",
                    "response_id": None,
                    "head_response_id": "resp_close_one",
                },
            )
            assert stale.status == 409
            assert store.owner_history_snapshot(
                conversation,
            )["conversation_closed"] is False

            current = await cli.post(
                f"/v1/responses/conversations/{conversation}/authority",
                json={
                    "action": "close",
                    "response_id": None,
                    "head_response_id": "resp_close_two",
                },
            )
            assert current.status == 200

        assert store.owner_history_snapshot(
            conversation,
        )["conversation_closed"] is True

    @pytest.mark.asyncio
    async def test_the_close_route_can_state_the_session_this_group_moves_to(
        self, adapter,
    ):
        """Item 32TK round 3: retiring a draft says which one is current now.

        The caller states the successor, and the durable pointer moves with the
        close. A caller that states nothing keeps the previous behaviour, and a
        malformed successor is refused rather than quietly ignored.
        """
        group = "raphael-owner-" + "c" * 32
        successor = "d" * 32
        store = adapter._response_store
        store.put("resp_group_head", {
            "response": {"id": "resp_group_head", "created_at": 100},
            "conversation_history": [],
        })
        assert store.set_conversation(group, "resp_group_head") is True

        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            path = f"/v1/responses/conversations/{group}/authority"
            malformed = await cli.post(path, json={
                "action": "close",
                "response_id": None,
                "head_response_id": "resp_group_head",
                "next_session_id": "not-a-session",
            })
            assert malformed.status == 409
            assert store.owner_session_index(
                "default", group,
            )["current_session_id"] == "legacy"

            moved = await cli.post(path, json={
                "action": "close",
                "response_id": None,
                "head_response_id": "resp_group_head",
                "next_session_id": successor,
            })
            assert moved.status == 200

        assert store.owner_session_index(
            "default", group,
        )["current_session_id"] == successor

    @pytest.mark.asyncio
    async def test_owner_authority_reconciles_only_an_orphaned_exact_run(self, adapter):
        conversation = "raphael-owner-" + "4" * 32
        response_id = "resp_orphaned_owner_run"
        claim_id = "claim_" + "5" * 32
        run_id = "run_" + "6" * 32
        adapter._response_store.put(response_id, {
            "response": {"id": response_id, "created_at": 900},
            "conversation_history": [
                {"role": "assistant", "content": json.dumps(_owner_new_proposal())},
            ],
        })
        assert adapter._response_store.set_conversation(
            conversation, response_id, owner_proposal=True,
        ) is True
        assert adapter._response_store.claim_owner_proposal(
            "default", conversation, response_id, claim_id,
        ) is True
        assert adapter._response_store.attach_owner_run(
            "default", conversation, response_id, claim_id, run_id,
        ) is True
        payload = {
            "action": "reconcile",
            "response_id": response_id,
            "claim_id": claim_id,
            "run_id": run_id,
        }
        adapter._set_run_status(run_id, "running")

        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            live = await cli.post(
                f"/v1/responses/conversations/{conversation}/authority",
                json=payload,
            )
            assert live.status == 409
            adapter._run_statuses.pop(run_id)
            recovered = await cli.post(
                f"/v1/responses/conversations/{conversation}/authority",
                json=payload,
            )

        assert recovered.status == 200
        assert adapter._response_store.owner_claim_is_released(
            "default", conversation, response_id, claim_id, run_id,
        ) is True

    @pytest.mark.asyncio
    async def test_successful_response_with_string_input(self, adapter):
        """String input is wrapped in a user message."""
        mock_result = {
            "final_response": "Paris is the capital of France.",
            "messages": [],
            "api_calls": 1,
        }

        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_run_agent", new_callable=AsyncMock) as mock_run:
                mock_run.return_value = (mock_result, {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0})
                resp = await cli.post(
                    "/v1/responses",
                    json={
                        "model": "hermes-agent",
                        "input": "What is the capital of France?",
                    },
                )

            assert resp.status == 200
            data = await resp.json()
            assert data["object"] == "response"
            assert data["id"].startswith("resp_")
            assert data["status"] == "completed"
            assert len(data["output"]) == 1
            assert data["output"][0]["type"] == "message"
            assert data["output"][0]["content"][0]["type"] == "output_text"
            assert data["output"][0]["content"][0]["text"] == "Paris is the capital of France."

    @pytest.mark.asyncio
    async def test_idempotent_owner_response_replays_original_authority(self, adapter):
        conversation = "raphael-owner-" + "d" * 32
        body = {
            "model": "hermes-agent",
            "input": "Prepare the private milestone",
            "conversation": conversation,
            "store": True,
            "expected_previous_response_id": None,
        }
        headers = {"Idempotency-Key": f"response-{uuid.uuid4().hex}"}
        mock_result = {
            "final_response": json.dumps(_owner_new_proposal()),
            "messages": [],
            "api_calls": 1,
        }

        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(
                adapter,
                "_run_agent",
                new_callable=AsyncMock,
                return_value=(
                    mock_result,
                    {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
                ),
            ) as mock_run:
                first = await cli.post(
                    "/v1/responses", json=body, headers=headers,
                )
                assert first.status == 200
                first_body = await first.json()
                assert adapter._response_store.mark_owner_proposal_consumed(
                    "default", conversation, first_body["id"],
                ) is True

                replay = await cli.post(
                    "/v1/responses", json=body, headers=headers,
                )
                assert replay.status == 200
                replay_body = await replay.json()

        assert replay_body == first_body
        assert replay_body["id"] == first_body["id"]
        snapshot = adapter._response_store.owner_history_snapshot(conversation)
        assert snapshot["latest_response_id"] == first_body["id"]
        assert snapshot["proposal_consumed"] is True
        mock_run.assert_awaited_once()


    async def _owner_turn(
        self, cli, adapter, conversation, *, text, headers=None, **extra,
    ):
        """One complete owner turn, with a stubbed structured reply.

        Both owner-conversation contracts are mandatory, so every turn states
        its predecessor and carries an Idempotency-Key unless the caller is
        deliberately testing one of them.
        """
        result = {
            "final_response": json.dumps(
                {"schema_version": 1, "kind": "question", "message": text}
            ),
            "messages": [],
            "api_calls": 1,
        }
        with patch.object(
            adapter,
            "_run_agent",
            new_callable=AsyncMock,
            return_value=(result, {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2}),
        ):
            return await cli.post(
                "/v1/responses",
                json={
                    "model": "hermes-agent",
                    "input": text,
                    "conversation": conversation,
                    "store": True,
                    **extra,
                },
                headers=headers or {
                    "Idempotency-Key": f"response-{uuid.uuid4().hex}",
                },
            )

    @pytest.mark.asyncio
    async def test_a_turn_planned_against_a_superseded_predecessor_is_refused(
        self, adapter,
    ):
        """A delayed request must never append to, or displace, a newer turn."""
        conversation = "raphael-owner-" + "1" * 32
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            first = await self._owner_turn(
                cli, adapter, conversation,
                text="First ask",
                expected_previous_response_id=None,
            )
            assert first.status == 200
            head = (await first.json())["id"]

            second = await self._owner_turn(
                cli, adapter, conversation,
                text="Second ask",
                expected_previous_response_id=head,
            )
            assert second.status == 200
            newer_head = (await second.json())["id"]

            # Request A, planned against the first turn, arrives late.
            delayed = await self._owner_turn(
                cli, adapter, conversation,
                text="Delayed ask",
                expected_previous_response_id=head,
            )
            assert delayed.status == 409
            assert (await delayed.json())["error"]["code"] == (
                "owner_conversation_stale"
            )

        # Nothing was appended and nothing was overwritten.
        snapshot = adapter._response_store.owner_history_snapshot(conversation)
        assert snapshot["head_response_id"] == newer_head
        assert [turn["owner"] for turn in snapshot["data"]] == [
            "First ask", "Second ask",
        ]

    @pytest.mark.asyncio
    async def test_a_first_turn_asserting_no_predecessor_is_refused_once_one_exists(
        self, adapter,
    ):
        conversation = "raphael-owner-" + "2" * 32
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            first = await self._owner_turn(
                cli, adapter, conversation,
                text="First ask",
                expected_previous_response_id=None,
            )
            assert first.status == 200
            repeat = await self._owner_turn(
                cli, adapter, conversation,
                text="Another first ask",
                expected_previous_response_id=None,
            )
            assert repeat.status == 409

    @pytest.mark.asyncio
    async def test_a_background_turn_is_bound_to_its_predecessor_too(self, adapter):
        conversation = "raphael-owner-" + "3" * 32
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            queued = await self._owner_turn(
                cli, adapter, conversation,
                text="Background ask",
                background=True,
                expected_previous_response_id=None,
            )
            assert queued.status == 200
            await asyncio.sleep(0)
            stale = await self._owner_turn(
                cli, adapter, conversation,
                text="Stale background ask",
                background=True,
                expected_previous_response_id="resp_" + "b" * 28,
            )
            assert stale.status == 409
            assert (await stale.json())["error"]["code"] == (
                "owner_conversation_stale"
            )

    @pytest.mark.asyncio
    async def test_a_malformed_predecessor_is_refused_before_anything_runs(
        self, adapter,
    ):
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_run_agent", new_callable=AsyncMock) as run:
                resp = await cli.post(
                    "/v1/responses",
                    json={
                        "model": "hermes-agent",
                        "input": "Plan it",
                        "conversation": "raphael-owner-" + "4" * 32,
                        "store": True,
                        "expected_previous_response_id": "../../etc/passwd",
                    },
                )
            assert resp.status == 400
            run.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_the_owner_request_is_durable_before_the_model_answers(
        self, adapter,
    ):
        """Item 32TK: recorded by the fence, so a lost accept response is not lost work.

        The reservation is taken before ``_run_agent`` is reached, so reading the
        projection from inside the model call is the exact moment that matters:
        a browser whose POST came back as a proxy's HTML document is sitting
        here with nothing, and this is what it can recover from.
        """
        conversation = "raphael-owner-" + "9" * 32
        seen: "list[Any]" = []

        async def _run(**_kwargs):
            seen.append(
                adapter._response_store.owner_history_snapshot(
                    conversation, profile="default",
                )["pending"]
            )
            return (
                {
                    "final_response": json.dumps({
                        "schema_version": 1,
                        "kind": "question",
                        "message": "Who is the workshop for?",
                    }),
                    "messages": [],
                    "api_calls": 1,
                },
                {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
            )

        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_run_agent", side_effect=_run):
                accepted = await cli.post(
                    "/v1/responses",
                    json={
                        "model": "hermes-agent",
                        "input": "Prepare a private 60-minute workshop",
                        "conversation": conversation,
                        "store": True,
                        "expected_previous_response_id": None,
                    },
                    headers={"Idempotency-Key": f"response-{uuid.uuid4().hex}"},
                )
            assert accepted.status == 200
            accepted_id = (await accepted.json())["id"]

        assert seen == [{
            "owner": "Prepare a private 60-minute workshop",
            "response_id": accepted_id,
        }]
        # Published: the durable turn now carries the same words, so nothing is
        # left pending.
        assert adapter._response_store.owner_history_snapshot(
            conversation, profile="default",
        )["pending"] is None

    @pytest.mark.asyncio
    async def test_a_background_turn_that_fails_leaves_the_owner_a_way_back(
        self, adapter,
    ):
        """Item 32TK: a failure the browser never saw is still findable.

        The accepted turn fails, which releases its fence. If that were all,
        a browser holding no handle to this response would find an empty
        conversation and never learn either the request or its outcome.
        """
        conversation = "raphael-owner-" + "8" * 32

        async def _fail(**_kwargs):
            raise RuntimeError("planner unavailable")

        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_run_agent", side_effect=_fail):
                accepted = await cli.post(
                    "/v1/responses",
                    json={
                        "model": "hermes-agent",
                        "input": "Prepare a private 60-minute workshop",
                        "conversation": conversation,
                        "background": True,
                        "store": True,
                        "expected_previous_response_id": None,
                    },
                    headers={"Idempotency-Key": f"response-{uuid.uuid4().hex}"},
                )
                assert accepted.status == 200
                response_id = (await accepted.json())["id"]

                for _ in range(100):
                    polled = await cli.get(f"/v1/responses/{response_id}")
                    if (await polled.json())["status"] == "failed":
                        break
                    await asyncio.sleep(0.01)
                else:
                    pytest.fail("the background response never became terminal")

            history = await cli.get(
                f"/v1/responses/conversations/{conversation}",
            )
            snapshot = await history.json()

        # The fence is gone with the turn that failed...
        assert snapshot["pending"] is None
        # ...and the request, and the response id that decided it, are not.
        assert snapshot["recovery"] == {
            "owner": "Prepare a private 60-minute workshop",
            "response_id": response_id,
        }
        assert snapshot["head_response_id"] is None
        assert snapshot["data"] == []

    @pytest.mark.asyncio
    async def test_an_exact_retry_against_the_unchanged_predecessor_replays(
        self, adapter,
    ):
        conversation = "raphael-owner-" + "5" * 32
        headers = {"Idempotency-Key": f"response-{uuid.uuid4().hex}"}
        body = {
            "model": "hermes-agent",
            "input": "Prepare the private milestone",
            "conversation": conversation,
            "store": True,
            "expected_previous_response_id": None,
        }
        result = {
            "final_response": json.dumps(_owner_new_proposal()),
            "messages": [],
            "api_calls": 1,
        }
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(
                adapter, "_run_agent", new_callable=AsyncMock,
                return_value=(result, {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2}),
            ) as run:
                first = await cli.post("/v1/responses", json=body, headers=headers)
                assert first.status == 200
                first_body = await first.json()
                # The in-process cache is not what makes this work.
                adapter_module = sys.modules[APIServerAdapter.__module__]
                adapter_module._idem_cache = _IdempotencyCache()
                replay = await cli.post("/v1/responses", json=body, headers=headers)
            assert replay.status == 200
            assert await replay.json() == first_body
            run.assert_awaited_once()

    async def _owner_stream(self, cli, adapter, body, headers, *, text):
        """One streamed owner turn, with a stubbed structured reply."""
        result = {
            "final_response": json.dumps(
                {"schema_version": 1, "kind": "question", "message": text}
            ),
            "messages": [],
            "api_calls": 1,
        }

        async def _run(**kwargs):
            callback = kwargs.get("stream_delta_callback")
            if callback:
                callback(result["final_response"])
            return (
                result,
                {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
            )

        with patch.object(adapter, "_run_agent", side_effect=_run) as run:
            resp = await cli.post("/v1/responses", json=body, headers=headers)
            payload = await resp.text()
        return resp, payload, run

    @staticmethod
    def _sse_terminal(payload: str, event: str) -> dict:
        for block in payload.split("\n\n"):
            if f"event: {event}\n" in block:
                for line in block.splitlines():
                    if line.startswith("data: "):
                        return json.loads(line[len("data: "):])["response"]
        raise AssertionError(f"no {event} event in stream")

    @pytest.mark.asyncio
    async def test_a_successful_owner_stream_replays_instead_of_replanning(
        self, adapter,
    ):
        """A streamed owner turn that really completed is a durable record.

        Without the durable completion, an exact retry is refused as superseded
        by the very reply it is retrying — never answered.
        """
        conversation = "raphael-owner-" + "e" * 32
        headers = {"Idempotency-Key": f"response-{uuid.uuid4().hex}"}
        body = {
            "model": "hermes-agent",
            "input": "Prepare the private milestone",
            "conversation": conversation,
            "store": True,
            "stream": True,
            "expected_previous_response_id": None,
        }
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            first, payload, run = await self._owner_stream(
                cli, adapter, body, headers, text="Which outcome first?",
            )
            assert first.status == 200
            completed = self._sse_terminal(payload, "response.completed")
            assert completed["status"] == "completed"
            run.assert_awaited_once()

            # The exact retry is answered from the durable record, as JSON:
            # the turn already happened, so it is replayed, not replanned.
            adapter_module = sys.modules[APIServerAdapter.__module__]
            adapter_module._idem_cache = _IdempotencyCache()
            retry, retry_payload, retry_run = await self._owner_stream(
                cli, adapter, body, headers, text="Which outcome first?",
            )
            assert retry.status == 200
            assert retry.headers["Content-Type"].startswith("application/json")
            replayed = json.loads(retry_payload)
            retry_run.assert_not_awaited()

        assert replayed == completed
        snapshot = adapter._response_store.owner_history_snapshot(conversation)
        assert snapshot["head_response_id"] == completed["id"]
        assert len(snapshot["data"]) == 1

    @pytest.mark.asyncio
    async def test_an_owner_stream_that_never_completed_stays_retryable(
        self, adapter,
    ):
        """No durable stored terminal response means no durable replay record.

        The agent dies mid-stream, so nothing was mapped into the
        conversation and no proposal could have gained authority. The key is
        released rather than stranded, and the retry plans the turn.
        """
        conversation = "raphael-owner-" + "f" * 32
        headers = {"Idempotency-Key": f"response-{uuid.uuid4().hex}"}
        body = {
            "model": "hermes-agent",
            "input": "Prepare the private milestone",
            "conversation": conversation,
            "store": True,
            "stream": True,
            "expected_previous_response_id": None,
        }
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(
                adapter, "_run_agent", new_callable=AsyncMock,
                side_effect=RuntimeError("agent died mid-stream"),
            ):
                crashed = await cli.post(
                    "/v1/responses", json=body, headers=headers,
                )
                crashed_payload = await crashed.text()
            assert crashed.status == 200
            failed = self._sse_terminal(crashed_payload, "response.failed")
            assert failed["status"] == "failed"

            # Nothing became this conversation's head, so nothing is replayed.
            assert adapter._response_store.owner_history_snapshot(
                conversation,
            )["head_response_id"] is None

            adapter_module = sys.modules[APIServerAdapter.__module__]
            adapter_module._idem_cache = _IdempotencyCache()
            retry, payload, run = await self._owner_stream(
                cli, adapter, body, headers, text="Which outcome first?",
            )
            assert retry.status == 200
            completed = self._sse_terminal(payload, "response.completed")
            run.assert_awaited_once()

        assert completed["id"] != failed["id"]
        snapshot = adapter._response_store.owner_history_snapshot(conversation)
        assert snapshot["head_response_id"] == completed["id"]

    @pytest.mark.asyncio
    async def test_an_owner_stream_records_nothing_before_the_mapping_is_durable(
        self, adapter,
    ):
        """A reply the conversation refuses is not a turn, so it is not a record."""
        conversation = "raphael-owner-" + "0" * 31 + "1"
        headers = {"Idempotency-Key": f"response-{uuid.uuid4().hex}"}
        body = {
            "model": "hermes-agent",
            "input": "Prepare the private milestone",
            "conversation": conversation,
            "store": True,
            "stream": True,
            "expected_previous_response_id": None,
        }
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(
                adapter._response_store, "publish_owner_turn", return_value=False,
            ):
                refused, refused_payload, _run = await self._owner_stream(
                    cli, adapter, body, headers, text="Which outcome first?",
                )
            assert refused.status == 200
            assert "event: response.failed\n" in refused_payload

            # No durable replay record was written for a turn that never
            # became this conversation's head.
            adapter_module = sys.modules[APIServerAdapter.__module__]
            adapter_module._idem_cache = _IdempotencyCache()
            retry, payload, run = await self._owner_stream(
                cli, adapter, body, headers, text="Which outcome first?",
            )
            assert retry.status == 200
            assert self._sse_terminal(payload, "response.completed")["status"] == (
                "completed"
            )
            run.assert_awaited_once()

    def test_an_owner_response_key_replays_across_a_restart(self, tmp_path):
        """The turn that minted a proposal is replayed, never planned twice."""
        db_path = str(tmp_path / "owner-response-store.db")
        conversation = "raphael-owner-" + "7" * 32
        response_id = "resp_owner_first_attempt"
        store = ResponseStore(db_path=db_path, max_size=10)
        try:
            assert store.reserve_owner_response(
                "default", "scope", "key", "fp", conversation, response_id,
            ) == ("new", None, None)
            store.put(response_id, {"response": {"id": response_id, "created_at": 1}})
            store.complete_owner_response(
                "default", "scope", "key", response_id,
                {"id": response_id, "status": "completed"}, "sess-1",
            )
        finally:
            store.close()

        reopened = ResponseStore(db_path=db_path, max_size=10)
        try:
            outcome, replay, session = reopened.reserve_owner_response(
                "default", "scope", "key", "fp", conversation, "resp_second_attempt",
            )
            assert outcome == "replay"
            assert replay == {"id": response_id, "status": "completed"}
            assert session == "sess-1"
            # A different request under the same key is a conflict, not a replay.
            assert reopened.reserve_owner_response(
                "default", "scope", "key", "other-fp", conversation,
                "resp_third_attempt",
            )[0] == "conflict"
            # So is the same request against a different conversation.
            assert reopened.reserve_owner_response(
                "default", "scope", "key", "fp",
                "raphael-owner-" + "8" * 32, "resp_fourth_attempt",
            )[0] == "conflict"
        finally:
            reopened.close()

    def test_a_crashed_owner_turn_never_mints_a_second_proposal(self, tmp_path):
        db_path = str(tmp_path / "crashed-owner-response-store.db")
        conversation = "raphael-owner-" + "9" * 32
        minted = "resp_owner_minted_then_died"
        store = ResponseStore(db_path=db_path, max_size=10)
        try:
            assert store.reserve_owner_response(
                "default", "scope", "key", "fp", conversation, minted,
            )[0] == "new"
            # The response was minted; the process died before recording the
            # body to replay.
            store.put(minted, {"response": {"id": minted, "created_at": 1}})
        finally:
            store.close()

        reopened = ResponseStore(db_path=db_path, max_size=10)
        try:
            assert reopened.reserve_owner_response(
                "default", "scope", "key", "fp", conversation, "resp_retry",
            )[0] == "incomplete"
        finally:
            reopened.close()

    def test_a_reservation_that_minted_nothing_is_adopted_not_stranded(self, tmp_path):
        db_path = str(tmp_path / "stranded-owner-response-store.db")
        conversation = "raphael-owner-" + "a" * 32
        store = ResponseStore(db_path=db_path, max_size=10)
        try:
            assert store.reserve_owner_response(
                "default", "scope", "key", "fp", conversation, "resp_never_stored",
            )[0] == "new"
        finally:
            store.close()

        reopened = ResponseStore(db_path=db_path, max_size=10)
        try:
            assert reopened.reserve_owner_response(
                "default", "scope", "key", "fp", conversation, "resp_retry",
            ) == ("new", None, None)
        finally:
            reopened.close()

    def test_retention_bounds_the_table_without_evicting_live_authority(self):
        conversation = "raphael-owner-" + "b" * 32
        live = "resp_live_owner_proposal"
        store = ResponseStore(max_size=10)
        store.put(live, {
            "response": {"id": live, "created_at": 1},
            "conversation_history": [{
                "role": "assistant",
                "content": json.dumps(_owner_new_proposal()),
            }],
        })
        assert store.reserve_owner_response(
            "default", "scope", "live-key", "fp", conversation, live,
        )[0] == "new"
        assert store.set_conversation(
            conversation, live, owner_proposal=True, profile="default",
        ) is True
        store.complete_owner_response(
            "default", "scope", "live-key", live, {"id": live}, None,
        )
        assert store.reserve_owner_response(
            "default", "scope", "old-key", "fp",
            conversation, "resp_long_finished_turn",
        )[0] == "new"

        store.purge_owner_response_idempotency(time.time() + 1)

        assert store.reserve_owner_response(
            "default", "scope", "live-key", "fp", conversation, "resp_retry",
        )[0] == "replay"
        assert store.reserve_owner_response(
            "default", "scope", "old-key", "fp", conversation, "resp_retry",
        )[0] == "new"

    def test_mapping_refuses_a_predecessor_that_moved_while_the_turn_ran(self):
        """The same assertion is re-compared when the reply is finally mapped."""
        conversation = "raphael-owner-" + "c" * 32
        store = ResponseStore(max_size=10)
        assert store.set_conversation(
            conversation, "resp_head_a", profile="default",
        ) is True
        assert store.set_conversation(
            conversation, "resp_head_b", profile="default",
        ) is True
        assert store.set_conversation(
            conversation,
            "resp_delayed_turn",
            profile="default",
            expected_previous_response_id="resp_head_a",
        ) is False
        assert store.get_conversation(conversation, profile="default") == "resp_head_b"
        assert store.set_conversation(
            conversation,
            "resp_next_turn",
            profile="default",
            expected_previous_response_id="resp_head_b",
        ) is True

    @pytest.mark.asyncio
    async def test_previous_response_id_stores_compressed_transcript_directly(self, adapter):
        """After compression, stored history is the compressed transcript, not prior + compressed."""
        prior_history = [
            {"role": "user", "content": "What is 1+1?"},
            {"role": "assistant", "content": "2"},
        ] * 10  # 20 messages — enough to simulate a long conversation
        adapter._response_store.put(
            "resp_prev",
            {
                "response": {"id": "resp_prev", "status": "completed"},
                "conversation_history": list(prior_history),
                "session_id": "api-test-session",
            },
        )

        compressed_history = [
            # Compressed transcript starts with summary, NOT with prior[0]
            {"role": "user", "content": "[Compressed summary of earlier conversation]"},
            {"role": "user", "content": "Now add 1 more"},
            {"role": "assistant", "content": "3"},
        ]

        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_run_agent", new_callable=AsyncMock) as mock_run:
                mock_run.return_value = (
                    {
                        "final_response": "3",
                        "messages": list(compressed_history),
                        "_compressed": True,
                        "api_calls": 1,
                    },
                    {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
                )
                resp = await cli.post(
                    "/v1/responses",
                    json={
                        "model": "hermes-agent",
                        "input": "Now add 1 more",
                        "previous_response_id": "resp_prev",
                    },
                )
                assert resp.status == 200
                data = await resp.json()

        stored = adapter._response_store.get(data["id"])
        stored_history = stored["conversation_history"]
        # Must NOT contain the original prior_history messages
        for msg in prior_history:
            assert msg not in stored_history, (
                f"Prior history message leaked into stored compressed transcript: {msg}"
            )
        # Must contain the compressed transcript
        assert stored_history == compressed_history


    @pytest.mark.asyncio
    async def test_previous_response_id_outputs_only_current_turn_items(self, adapter):
        """Response output must not replay previous tool artifacts."""
        prior_history = [
            {"role": "user", "content": "Read old file"},
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call_old",
                        "function": {
                            "name": "read_file",
                            "arguments": '{"path":"old.txt"}',
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_old",
                "content": '{"content":"old"}',
            },
            {"role": "assistant", "content": "old"},
        ]
        adapter._response_store.put(
            "resp_prev",
            {
                "response": {"id": "resp_prev", "status": "completed"},
                "conversation_history": list(prior_history),
                "session_id": "api-test-session",
            },
        )
        full_agent_transcript = prior_history + [
            {"role": "user", "content": "Read new file"},
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call_new",
                        "function": {
                            "name": "read_file",
                            "arguments": '{"path":"new.txt"}',
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_new",
                "content": '{"content":"new"}',
            },
            {"role": "assistant", "content": "new"},
        ]

        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_run_agent", new_callable=AsyncMock) as mock_run:
                mock_run.return_value = (
                    {
                        "final_response": "new",
                        "messages": list(full_agent_transcript),
                        "api_calls": 1,
                    },
                    {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
                )
                resp = await cli.post(
                    "/v1/responses",
                    json={
                        "model": "hermes-agent",
                        "input": "Read new file",
                        "previous_response_id": "resp_prev",
                    },
                )
                assert resp.status == 200
                data = await resp.json()

        output_json = json.dumps(data["output"])
        assert "call_new" in output_json
        assert "call_old" not in output_json
        assert "old.txt" not in output_json


    @pytest.mark.asyncio
    async def test_invalid_previous_response_id_returns_404(self, adapter):
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(
                "/v1/responses",
                json={
                    "model": "hermes-agent",
                    "input": "follow up",
                    "previous_response_id": "resp_nonexistent",
                },
            )
            assert resp.status == 404


    @pytest.mark.asyncio
    async def test_store_string_false_does_not_store(self, adapter):
        """Quoted false must preserve ephemeral store=false semantics."""
        mock_result = {"final_response": "OK", "messages": [], "api_calls": 1}

        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_run_agent", new_callable=AsyncMock) as mock_run:
                mock_run.return_value = (
                    mock_result,
                    {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
                )
                resp = await cli.post(
                    "/v1/responses",
                    json={
                        "model": "hermes-agent",
                        "input": "Hello",
                        "store": "false",
                    },
                )

            assert resp.status == 200
            data = await resp.json()
            assert adapter._response_store.get(data["id"]) is None

    @pytest.mark.asyncio
    async def test_instructions_inherited_from_previous(self, adapter):
        """If no instructions provided, carry forward from previous response."""
        mock_result = {"final_response": "Ahoy!", "messages": [], "api_calls": 1}

        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            # First request with instructions
            with patch.object(adapter, "_run_agent", new_callable=AsyncMock) as mock_run:
                mock_run.return_value = (mock_result, {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0})
                resp1 = await cli.post(
                    "/v1/responses",
                    json={
                        "model": "hermes-agent",
                        "input": "Hello",
                        "instructions": "Be a pirate",
                    },
                )

            data1 = await resp1.json()
            resp_id = data1["id"]

            # Second request without instructions
            with patch.object(adapter, "_run_agent", new_callable=AsyncMock) as mock_run:
                mock_run.return_value = (mock_result, {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0})
                resp2 = await cli.post(
                    "/v1/responses",
                    json={
                        "model": "hermes-agent",
                        "input": "Tell me more",
                        "previous_response_id": resp_id,
                    },
                )

            assert resp2.status == 200
            call_kwargs = mock_run.call_args.kwargs
            assert call_kwargs["ephemeral_system_prompt"] == "Be a pirate"


    @pytest.mark.asyncio
    async def test_result_error_fallback_is_redacted(self, adapter):
        raw_secret = "sk-responses-leak-1234567890"
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_run_agent", new_callable=AsyncMock) as mock_run:
                mock_run.return_value = (
                    {
                        "final_response": "",
                        "error": f"provider auth failed OPENAI_API_KEY={raw_secret}",
                        "messages": [],
                        "api_calls": 1,
                    },
                    {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
                )
                resp = await cli.post(
                    "/v1/responses",
                    json={"model": "hermes-agent", "input": "Hello"},
                )

            assert resp.status == 200
            data = await resp.json()
            body = json.dumps(data)
            assert raw_secret not in body
            assert "OPENAI_API_KEY=" in body
            assert data["output"][0]["content"][0]["text"] != f"provider auth failed OPENAI_API_KEY={raw_secret}"

    @pytest.mark.asyncio
    async def test_background_response_returns_immediately_and_can_be_polled(self, adapter):
        """A long response must outlive the initiating HTTP request."""
        started = asyncio.Event()
        release = asyncio.Event()

        async def _slow_run(**kwargs):
            started.set()
            await release.wait()
            return (
                {
                    "final_response": "Milestone ready.",
                    "messages": [],
                    "api_calls": 1,
                },
                {"input_tokens": 12, "output_tokens": 4, "total_tokens": 16},
            )

        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_run_agent", side_effect=_slow_run) as mock_run:
                post_task = asyncio.create_task(cli.post(
                    "/v1/responses",
                    json={
                        "model": "hermes-agent",
                        "input": "Plan the milestone",
                        "conversation": "raphael-owner-" + "a" * 32,
                        "background": True,
                        "store": True,
                        "expected_previous_response_id": None,
                    },
                    headers={"Idempotency-Key": f"response-{uuid.uuid4().hex}"},
                ))
                await asyncio.wait_for(started.wait(), timeout=1)
                try:
                    competing = await cli.post(
                        "/v1/responses",
                        json={
                            "model": "hermes-agent",
                            "input": "Replace the in-flight request",
                            "conversation": "raphael-owner-" + "a" * 32,
                            "background": True,
                            "store": True,
                            "expected_previous_response_id": None,
                        },
                        headers={
                            "Idempotency-Key": f"response-{uuid.uuid4().hex}",
                        },
                    )
                    assert competing.status == 409
                    assert (await competing.json())["error"]["code"] == (
                        "owner_conversation_locked"
                    )
                    resp = await asyncio.wait_for(
                        asyncio.shield(post_task), timeout=0.2,
                    )
                finally:
                    release.set()
                    if not post_task.done():
                        await post_task

                assert resp.status == 200
                queued = await resp.json()
                assert queued["object"] == "response"
                assert queued["status"] == "queued"
                assert queued["background"] is True
                assert queued["output"] == []

                response_id = queued["id"]
                completed = None
                for _ in range(100):
                    polled = await cli.get(f"/v1/responses/{response_id}")
                    assert polled.status == 200
                    completed = await polled.json()
                    if completed["status"] == "completed":
                        break
                    assert completed["status"] in {"queued", "in_progress"}
                    await asyncio.sleep(0.01)

        assert completed is not None
        assert completed["status"] == "completed"
        assert completed["background"] is True
        assert completed["usage"] == {
            "input_tokens": 12,
            "output_tokens": 4,
            "total_tokens": 16,
        }
        assert completed["output"][0]["content"][0]["text"] == "Milestone ready."
        assert adapter._response_store.get_conversation(
            "raphael-owner-" + "a" * 32,
        ) == queued["id"]
        mock_run.assert_called_once()

    @pytest.mark.asyncio
    async def test_lost_owner_reservation_interrupts_background_agent(self, adapter):
        started = asyncio.Event()
        interrupted = asyncio.Event()

        class Agent:
            def interrupt(self, _message=None):
                interrupted.set()

        async def _slow_run(**kwargs):
            kwargs["agent_ref"][0] = Agent()
            started.set()
            await interrupted.wait()
            return (
                {"final_response": "Must not commit.", "messages": [], "api_calls": 1},
                {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
            )

        conversation = "raphael-owner-" + "f" * 32
        app = _create_app(adapter)
        with (
            patch(
                "gateway.platforms.api_server."
                "_OWNER_CONVERSATION_RESERVATION_RENEW_SECONDS",
                0.01,
            ),
            patch.object(
                adapter._response_store,
                "renew_owner_conversation_reservation",
                return_value=False,
            ),
            patch.object(adapter, "_run_agent", side_effect=_slow_run),
        ):
            async with TestClient(TestServer(app)) as cli:
                response = await cli.post(
                    "/v1/responses",
                    json={
                        "model": "hermes-agent",
                        "input": "Plan the milestone",
                        "conversation": conversation,
                        "background": True,
                        "store": True,
                        "expected_previous_response_id": None,
                    },
                    headers={
                        "Idempotency-Key": f"response-{uuid.uuid4().hex}",
                    },
                )
                queued = await response.json()
                await asyncio.wait_for(started.wait(), timeout=1)
                await asyncio.wait_for(interrupted.wait(), timeout=1)

                terminal = None
                for _ in range(100):
                    polled = await cli.get(f"/v1/responses/{queued['id']}")
                    terminal = await polled.json()
                    if terminal["status"] == "failed":
                        break
                    await asyncio.sleep(0.01)

        assert response.status == 200
        assert terminal is not None
        assert terminal["status"] == "failed"
        assert adapter._response_store.get_conversation(conversation) is None

    @pytest.mark.asyncio
    async def test_lost_owner_reservation_stops_agent_created_after_heartbeat(self, adapter):
        creation_started = threading.Event()
        release_creation = threading.Event()
        interrupted = threading.Event()
        ran = threading.Event()

        class Agent:
            session_prompt_tokens = 0
            session_completion_tokens = 0
            session_total_tokens = 0

            def interrupt(self, _message=None):
                interrupted.set()

            def run_conversation(self, **_kwargs):
                ran.set()
                return {"final_response": "Must not commit."}

        def _delayed_create(**_kwargs):
            creation_started.set()
            assert release_creation.wait(timeout=1)
            return Agent()

        conversation = "raphael-owner-" + "e" * 32
        app = _create_app(adapter)
        try:
            with (
                patch(
                    "gateway.platforms.api_server."
                    "_OWNER_CONVERSATION_RESERVATION_RENEW_SECONDS",
                    0.01,
                ),
                patch.object(
                    adapter._response_store,
                    "renew_owner_conversation_reservation",
                    return_value=False,
                ),
                patch.object(adapter, "_create_agent", side_effect=_delayed_create),
            ):
                async with TestClient(TestServer(app)) as cli:
                    response = await cli.post(
                        "/v1/responses",
                        json={
                            "model": "hermes-agent",
                            "input": "Plan the milestone",
                            "conversation": conversation,
                            "background": True,
                            "store": True,
                            "expected_previous_response_id": None,
                        },
                        headers={
                            "Idempotency-Key": f"response-{uuid.uuid4().hex}",
                        },
                    )
                    queued = await response.json()
                    assert await asyncio.to_thread(
                        creation_started.wait, 1,
                    ) is True
                    await asyncio.sleep(0.05)
                    release_creation.set()

                    terminal = None
                    for _ in range(100):
                        polled = await cli.get(f"/v1/responses/{queued['id']}")
                        terminal = await polled.json()
                        if terminal["status"] == "failed":
                            break
                        await asyncio.sleep(0.01)
        finally:
            release_creation.set()

        assert response.status == 200
        assert terminal is not None
        assert terminal["status"] == "failed"
        assert interrupted.is_set()
        assert not ran.is_set()
        assert adapter._response_store.get_conversation(conversation) is None

    @pytest.mark.asyncio
    async def test_background_failure_is_pollable_and_redacted(self, adapter):
        raw_secret = "sk-background-leak-1234567890"
        conversation = "raphael-owner-" + "b" * 32

        async def _failing_run(**kwargs):
            await asyncio.sleep(0)
            raise RuntimeError(f"provider failed OPENAI_API_KEY={raw_secret}")

        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_run_agent", side_effect=_failing_run):
                resp = await cli.post(
                    "/v1/responses",
                    json={
                        "model": "hermes-agent",
                        "input": "Plan the milestone",
                        "conversation": conversation,
                        "background": True,
                        "store": True,
                        "expected_previous_response_id": None,
                    },
                    headers={
                        "Idempotency-Key": f"response-{uuid.uuid4().hex}",
                    },
                )
                assert resp.status == 200
                response_id = (await resp.json())["id"]

                failed = None
                for _ in range(100):
                    polled = await cli.get(f"/v1/responses/{response_id}")
                    assert polled.status == 200
                    failed = await polled.json()
                    if failed["status"] == "failed":
                        break
                    await asyncio.sleep(0.01)

                def _send_a_different_message(previous_response_id=None):
                    return cli.post(
                        "/v1/responses",
                        json={
                            "model": "hermes-agent",
                            "input": "Try the plan again",
                            "conversation": conversation,
                            "background": True,
                            "store": True,
                            "expected_previous_response_id": previous_response_id,
                        },
                        headers={
                            "Idempotency-Key": f"response-{uuid.uuid4().hex}",
                        },
                    )

                with patch.object(
                    adapter,
                    "_run_agent",
                    new_callable=AsyncMock,
                    return_value=(
                        {
                            "final_response": "A fresh plan can start.",
                            "messages": [],
                            "api_calls": 1,
                        },
                        {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
                    ),
                ) as fresh_plan:
                    # The failed turn left an interrupted request nobody has
                    # answered for yet. Until it is acknowledged this
                    # conversation takes nothing new — a second message must not
                    # be able to bury the first — and no model runs.
                    refused = await _send_a_different_message()
                    assert refused.status == 409
                    assert (await refused.json())["error"]["code"] == (
                        "owner_conversation_locked"
                    )
                    assert fresh_plan.await_count == 0
                    history = await cli.get(
                        f"/v1/responses/conversations/{conversation}",
                    )
                    # The request itself came through the ordinary background
                    # failure path, in the same write that made the turn
                    # terminal. It is all a browser that never received the
                    # accept response has left to find.
                    assert (await history.json())["recovery"] == {
                        "owner": "Plan the milestone",
                        "response_id": response_id,
                    }

                    acknowledged = await cli.post(
                        f"/v1/responses/conversations/{conversation}/recovery",
                        json={"response_id": response_id},
                    )
                    assert acknowledged.status == 200

                    sealed_history = await cli.get(
                        f"/v1/responses/conversations/{conversation}",
                    )
                    sealed = await sealed_history.json()
                    assert sealed["head_response_id"] == response_id
                    assert sealed["recovery"] is None
                    assert sealed["latest_response_id"] is None
                    assert sealed["data"][0]["owner"] == "Plan the milestone"
                    assert json.loads(sealed["data"][0]["raphael"])["kind"] == (
                        "failure"
                    )

                    # The acknowledged failure is now a real conversation
                    # turn. A fresh request follows that exact revision rather
                    # than pretending this is still an empty conversation.
                    retry = await _send_a_different_message(response_id)
                    assert retry.status == 200

        assert failed is not None
        assert failed["status"] == "failed"
        assert failed["background"] is True
        assert failed["output"] == []
        assert raw_secret not in json.dumps(failed)
        assert "OPENAI_API_KEY=" in failed["error"]["message"]

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "unsupported",
        [
            {"store": False},
            {"stream": True},
        ],
    )
    async def test_background_requires_storage_and_non_streaming(self, adapter, unsupported):
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(
                "/v1/responses",
                json={
                    "model": "hermes-agent",
                    "input": "Plan the milestone",
                    "background": True,
                    **unsupported,
                },
            )
            data = await resp.json()

        assert resp.status == 400
        assert data["error"]["type"] == "invalid_request_error"

    # -----------------------------------------------------------------------
    # Item 32TK: the owner-conversation request contract, and one commit
    # -----------------------------------------------------------------------

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "body_extra,headers,param",
        [
            ({}, {"Idempotency-Key": "k-1"}, "expected_previous_response_id"),
            ({"expected_previous_response_id": None}, {}, "Idempotency-Key"),
        ],
    )
    async def test_an_owner_turn_without_both_contracts_is_refused(
        self, adapter, body_extra, headers, param,
    ):
        """Optional, these let an older or direct caller append to whatever the
        head happens to be and duplicate an answered turn after a restart."""
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_run_agent", new_callable=AsyncMock) as run:
                resp = await cli.post(
                    "/v1/responses",
                    json={
                        "model": "hermes-agent",
                        "input": "Plan it",
                        "conversation": "raphael-owner-" + "5" * 32,
                        "store": True,
                        **body_extra,
                    },
                    headers=headers,
                )
                body = await resp.json()
                run.assert_not_awaited()

        assert resp.status == 400
        assert body["error"]["param"] == param

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "forbidden",
        [
            {"conversation_history": [{"role": "user", "content": "fabricated"}]},
            {"conversation_history": []},
            {"previous_response_id": "resp_" + "c" * 28},
        ],
    )
    async def test_an_owner_turn_cannot_supply_its_own_history(
        self, adapter, forbidden,
    ):
        """A direct caller must not be able to publish a fabricated or
        truncated transcript as the conversation's new authority."""
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_run_agent", new_callable=AsyncMock) as run:
                resp = await cli.post(
                    "/v1/responses",
                    json={
                        "model": "hermes-agent",
                        "input": "Plan it",
                        "conversation": "raphael-owner-" + "6" * 32,
                        "store": True,
                        "expected_previous_response_id": None,
                        **forbidden,
                    },
                    headers={"Idempotency-Key": f"response-{uuid.uuid4().hex}"},
                )
                body = await resp.json()
                run.assert_not_awaited()

        assert resp.status == 400
        assert body["error"]["param"] in {
            "conversation_history", "previous_response_id",
        }

    @pytest.mark.asyncio
    async def test_a_generic_conversation_keeps_its_optional_contracts(
        self, adapter,
    ):
        """The whole contract is owner-conversation-only."""
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(
                adapter,
                "_run_agent",
                new_callable=AsyncMock,
                return_value=(
                    {"final_response": "Fine.", "messages": [], "api_calls": 1},
                    {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
                ),
            ):
                resp = await cli.post(
                    "/v1/responses",
                    json={
                        "model": "hermes-agent",
                        "input": "Plan it",
                        "conversation": "ordinary-conversation",
                        "store": True,
                        "conversation_history": [
                            {"role": "user", "content": "earlier"},
                        ],
                    },
                )

        assert resp.status == 200

    @pytest.mark.asyncio
    @pytest.mark.parametrize("mode", ["standard", "background", "stream"])
    async def test_an_owner_turn_publishes_in_one_commit(self, adapter, mode):
        """The response, the head, the reservation and the replay record are
        one event; a partial commit left a turn that could not be replayed."""
        conversation = "raphael-owner-" + "7" * 32
        key = f"response-{uuid.uuid4().hex}"
        result = {
            "final_response": json.dumps(_owner_new_proposal()),
            "messages": [],
            "api_calls": 1,
        }
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with (
                patch.object(
                    adapter,
                    "_run_agent",
                    new_callable=AsyncMock,
                    return_value=(
                        result,
                        {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
                    ),
                ),
                patch.object(
                    adapter._response_store, "publish_owner_turn",
                    side_effect=adapter._response_store.publish_owner_turn,
                ) as publish,
            ):
                resp = await cli.post(
                    "/v1/responses",
                    json={
                        "model": "hermes-agent",
                        "input": "Prepare the private milestone",
                        "conversation": conversation,
                        "store": True,
                        "expected_previous_response_id": None,
                        **(
                            {"background": True} if mode == "background"
                            else {"stream": True} if mode == "stream"
                            else {}
                        ),
                    },
                    headers={"Idempotency-Key": key},
                )
                assert resp.status == 200
                await resp.read()
                for _ in range(200):
                    if publish.call_count:
                        break
                    await asyncio.sleep(0.01)

            # Every path publishes through the one transactional method.
            assert publish.call_count == 1
            published_id = publish.call_args.kwargs["response_id"]

        snapshot = adapter._response_store.owner_history_snapshot(conversation)
        assert snapshot["head_response_id"] == published_id
        # The head and the durable replay record landed together.
        outcome, replay, _session = (
            adapter._response_store.lookup_owner_response(
                "default",
                publish.call_args.kwargs["session_scope"],
                key,
                adapter._response_store._conn.execute(
                    "SELECT fingerprint FROM owner_response_idempotency "
                    "WHERE idempotency_key = ?", (key,),
                ).fetchone()[0],
                conversation,
            )
        )
        assert outcome == "replay"
        assert replay["id"] == published_id

    @pytest.mark.asyncio
    async def test_a_queued_owner_response_is_recovered_after_a_restart(
        self, adapter,
    ):
        """A background owner turn's executor is an in-memory task, so a
        restart must close the durable queued state instead of leaving normal
        Project chat queued for good."""
        conversation = "raphael-owner-" + "c" * 32
        store = adapter._response_store
        response_id = "resp_" + "a" * 28
        store.put(response_id, {
            "response": {
                "id": response_id, "object": "response", "status": "queued",
                "background": True, "output": [],
            },
            "conversation_history": [],
        })
        # Exactly what the 202 leaves behind, attributed to a process that is
        # gone: the fence this turn took before any model ran, the job row, a
        # pid that cannot be running, and an age past the reap floor.
        assert store.reserve_owner_conversation(
            "default", conversation, response_id,
            owner_message="Prepare a private 60-minute workshop",
        ) is True
        store.reserve_owner_job(
            "response", response_id, "default", {"conversation": conversation},
        )
        store._conn.execute(
            "UPDATE owner_executor_jobs SET executor_id = 'dead', "
            "executor_pid = 2147483646, created_at = ? WHERE job_key = ?",
            (time.time() - 3600, response_id),
        )
        store._conn.commit()

        adapter._recover_orphaned_owner_jobs()

        recovered = store.get(response_id)
        assert recovered["response"]["status"] == "failed"
        assert "restarted" in recovered["response"]["error"]["message"]
        # And nothing is left claiming an executor.
        assert store.claim_orphaned_owner_jobs("response") == []
        # The request is not lost with the executor that was driving it: the
        # fence became the way back to this exact outcome.
        assert store.owner_history_snapshot(conversation)["recovery"] == {
            "owner": "Prepare a private 60-minute workshop",
            "response_id": response_id,
        }

    @pytest.mark.asyncio
    async def test_a_completed_owner_response_is_never_recovered_as_failed(
        self, adapter,
    ):
        """The job row can outlive a turn that DID complete; recovery must
        leave the published authority alone."""
        conversation = "raphael-owner-" + "d" * 31 + "e"
        store = adapter._response_store
        response_id = "resp_" + "b" * 28
        store.put(response_id, {
            "response": {"id": response_id, "status": "completed"},
            "conversation_history": [],
        })
        assert store.set_conversation(conversation, response_id) is True
        store.reserve_owner_job(
            "response", response_id, "default", {"conversation": conversation},
        )
        store._conn.execute(
            "UPDATE owner_executor_jobs SET executor_id = 'dead', "
            "executor_pid = 2147483646, created_at = ? WHERE job_key = ?",
            (time.time() - 3600, response_id),
        )
        store._conn.commit()

        adapter._recover_orphaned_owner_jobs()

        assert store.get(response_id)["response"]["status"] == "completed"

    def test_a_live_siblings_job_is_never_reaped(self, adapter):
        store = adapter._response_store
        store.reserve_owner_job(
            "run", "run_" + "f" * 32, "default", {},
        )
        store._conn.execute(
            "UPDATE owner_executor_jobs SET executor_id = 'sibling', "
            "executor_pid = ?, created_at = ?",
            (os.getpid(), time.time() - 3600),
        )
        store._conn.commit()

        assert store.claim_orphaned_owner_jobs("run") == []

    # -----------------------------------------------------------------------
    # Item 32TK round 2: everything a 202 promises commits together
    # -----------------------------------------------------------------------

    @staticmethod
    def _accepted_background_response(store, *, conversation, response_id, key):
        scope = "scope-" + key
        queued = {
            "id": response_id, "object": "response", "status": "queued",
            "background": True, "output": [],
        }
        assert store.reserve_owner_response(
            "default", scope, key, "fingerprint", conversation, response_id,
        )[0] == "new"
        store.accept_owner_background_response(
            profile="default",
            response_id=response_id,
            data={"response": queued, "conversation_history": []},
            conversation=conversation,
            replay=queued,
            session_scope=scope,
            idempotency_key=key,
            session_id="sess-1",
        )
        return scope, queued

    def test_a_failed_acceptance_leaves_no_half_accepted_owner_turn(self, adapter):
        """The queued body, the recovery job and the replay record are three
        facts about one acceptance; a crash between them left an accepted turn
        with no executor, or a key that could never record a body."""
        store = adapter._response_store
        conversation = "raphael-owner-" + "1" * 32
        response_id = "resp_" + "1" * 28
        scope = "scope-atomic"
        key = "idem-atomic"
        assert store.reserve_owner_response(
            "default", scope, key, "fingerprint", conversation, response_id,
        )[0] == "new"
        with patch.object(
            store, "_complete_owner_response_locked",
            side_effect=RuntimeError("disk full"),
        ):
            with pytest.raises(RuntimeError):
                store.accept_owner_background_response(
                    profile="default",
                    response_id=response_id,
                    data={"response": {"id": response_id}, "conversation_history": []},
                    conversation=conversation,
                    replay={"id": response_id},
                    session_scope=scope,
                    idempotency_key=key,
                    session_id=None,
                )

        # Nothing landed: no queued body, and no job claiming an executor.
        assert store.get(response_id) is None
        assert store._conn.execute(
            "SELECT COUNT(*) FROM owner_executor_jobs WHERE job_key = ?",
            (response_id,),
        ).fetchone()[0] == 0

    def test_an_accepted_background_turn_commits_body_job_and_replay(self, adapter):
        store = adapter._response_store
        conversation = "raphael-owner-" + "2" * 32
        response_id = "resp_" + "2" * 28
        scope, queued = self._accepted_background_response(
            store, conversation=conversation, response_id=response_id,
            key="idem-accepted",
        )

        assert store.get(response_id)["response"] == queued
        assert store._conn.execute(
            "SELECT COUNT(*) FROM owner_executor_jobs WHERE job_key = ?",
            (response_id,),
        ).fetchone()[0] == 1
        outcome, replay, session = store.lookup_owner_response(
            "default", scope, "idem-accepted", "fingerprint", conversation,
        )
        assert (outcome, replay, session) == ("replay", queued, "sess-1")

    def test_a_terminal_write_that_fails_keeps_the_recovery_job(self, adapter):
        """Releasing the job separately ran even when the terminal write failed,
        which left the response queued forever with nobody to recover it."""
        store = adapter._response_store
        conversation = "raphael-owner-" + "3" * 32
        response_id = "resp_" + "3" * 28
        self._accepted_background_response(
            store, conversation=conversation, response_id=response_id,
            key="idem-terminal",
        )
        with patch.object(
            store, "_put_response_locked", side_effect=RuntimeError("disk full"),
        ):
            with pytest.raises(RuntimeError):
                store.store_terminal_owner_response(
                    profile="default",
                    response_id=response_id,
                    data={"response": {"id": response_id, "status": "failed"}},
                    release_job=True,
                )

        assert store._conn.execute(
            "SELECT COUNT(*) FROM owner_executor_jobs WHERE job_key = ?",
            (response_id,),
        ).fetchone()[0] == 1

        store.store_terminal_owner_response(
            profile="default",
            response_id=response_id,
            data={"response": {"id": response_id, "status": "failed"}},
            release_job=True,
        )
        assert store._conn.execute(
            "SELECT COUNT(*) FROM owner_executor_jobs WHERE job_key = ?",
            (response_id,),
        ).fetchone()[0] == 0

    @staticmethod
    def _failed_body(response_id):
        return {
            "id": response_id, "object": "response", "status": "failed",
            "output": [], "error": {"code": "server_error", "message": "stopped"},
        }

    def test_a_terminal_owner_failure_and_its_recovery_commit_together(
        self, adapter,
    ):
        """Item 32TK, finding 3: the request cannot fall between two commits.

        Storing the terminal body and turning this turn's fence into the record
        that carries the owner's request were separate transactions. A crash
        between them left a turn that had failed with nothing left to say what
        the owner had sent — the exact way the Founder's request was lost.
        """
        store = adapter._response_store
        conversation = "raphael-owner-" + "6" * 32
        response_id = "resp_" + "6" * 28
        scope, queued = self._accepted_background_response(
            store, conversation=conversation, response_id=response_id,
            key="idem-atomic-failure",
        )
        assert store.reserve_owner_conversation(
            "default", conversation, response_id,
            owner_message="Prepare a private 60-minute workshop",
        ) is True

        # The last write of the transaction fails, which is the crash this is
        # about: everything before it is already staged.
        with patch.object(
            store, "_release_owner_job_locked",
            side_effect=RuntimeError("power lost"),
        ):
            with pytest.raises(RuntimeError):
                store.store_terminal_owner_response(
                    profile="default",
                    response_id=response_id,
                    data={"response": self._failed_body(response_id)},
                    release_job=True,
                    conversation=conversation,
                    interrupted=True,
                )

        # Nothing moved. The turn is still queued, still fenced, and its job row
        # still says somebody must finish it — so the whole thing is recoverable
        # exactly as it was.
        assert store.get(response_id)["response"] == queued
        snapshot = store.owner_history_snapshot(conversation)
        assert snapshot["pending"] == {
            "owner": "Prepare a private 60-minute workshop",
            "response_id": response_id,
        }
        assert snapshot["recovery"] is None
        assert store._conn.execute(
            "SELECT COUNT(*) FROM owner_executor_jobs WHERE job_key = ?",
            (response_id,),
        ).fetchone()[0] == 1
        assert store.lookup_owner_response(
            "default", scope, "idem-atomic-failure", "fingerprint", conversation,
        ) == ("replay", queued, "sess-1")

        store.store_terminal_owner_response(
            profile="default",
            response_id=response_id,
            data={"response": self._failed_body(response_id)},
            release_job=True,
            conversation=conversation,
            interrupted=True,
        )

        # And now all four facts about this ending landed together.
        assert store.get(response_id)["response"]["status"] == "failed"
        snapshot = store.owner_history_snapshot(conversation)
        assert snapshot["pending"] is None
        assert snapshot["recovery"] == {
            "owner": "Prepare a private 60-minute workshop",
            "response_id": response_id,
        }
        assert store._conn.execute(
            "SELECT COUNT(*) FROM owner_executor_jobs WHERE job_key = ?",
            (response_id,),
        ).fetchone()[0] == 0
        outcome, replay, _session = store.lookup_owner_response(
            "default", scope, "idem-atomic-failure", "fingerprint", conversation,
        )
        assert outcome == "replay"
        assert replay["status"] == "failed"

    def test_an_exact_retry_replays_a_recovered_failure_instead_of_a_new_turn(
        self, adapter,
    ):
        """Deleting the idempotency record made an exact retry a SECOND turn
        instead of replaying the accepted response's terminal outcome."""
        store = adapter._response_store
        conversation = "raphael-owner-" + "4" * 32
        response_id = "resp_" + "4" * 28
        assert store.reserve_owner_conversation(
            "default", conversation, response_id,
            owner_message="Prepare a private 60-minute workshop",
        ) is True
        scope, _queued = self._accepted_background_response(
            store, conversation=conversation, response_id=response_id,
            key="idem-recovered",
        )
        store._conn.execute(
            "UPDATE owner_executor_jobs SET executor_id = 'dead', "
            "executor_pid = 2147483646, created_at = ?, lease_expires_at = ? "
            "WHERE job_key = ?",
            (time.time() - 3600, time.time() - 1, response_id),
        )
        store._conn.commit()

        adapter._recover_orphaned_owner_jobs()

        outcome, replay, _session = store.lookup_owner_response(
            "default", scope, "idem-recovered", "fingerprint", conversation,
        )
        assert outcome == "replay"
        assert replay["status"] == "failed"
        assert "restarted" in replay["error"]["message"]
        # The key stayed immutable: the same response id, never a new one.
        assert replay["id"] == response_id
        # And the replay landed together with the way back to it.
        assert store.owner_history_snapshot(conversation)["recovery"] == {
            "owner": "Prepare a private 60-minute workshop",
            "response_id": response_id,
        }


class TestResponsesStreaming:


    @pytest.mark.asyncio
    async def test_stream_task_done_callback_enqueues_eos_for_responses(self, adapter):
        """Regression guard for #24451 on /v1/responses streaming path."""
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            class _FakeTask:
                def __init__(self):
                    self.callbacks = []

                def add_done_callback(self, cb):
                    self.callbacks.append(cb)

            fake_task = _FakeTask()

            def _fake_ensure_future(coro):
                # We short-circuit task scheduling in this unit test.
                coro.close()
                return fake_task

            with (
                patch.object(
                    adapter,
                    "_run_agent",
                    new=AsyncMock(
                        return_value=(
                            {"final_response": "ok", "messages": [], "api_calls": 1},
                            {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
                        )
                    ),
                ),
                patch("gateway.platforms.api_server.asyncio.ensure_future", side_effect=_fake_ensure_future),
                patch.object(adapter, "_write_sse_responses", new_callable=AsyncMock) as mock_write_sse,
            ):
                mock_write_sse.return_value = web.Response(status=200, text="ok")
                resp = await cli.post(
                    "/v1/responses",
                    json={"model": "hermes-agent", "input": "hi", "stream": True},
                )
                assert resp.status == 200

            assert len(fake_task.callbacks) == 1
            stream_q = mock_write_sse.call_args.kwargs["stream_q"]
            assert stream_q.empty()
            fake_task.callbacks[0](fake_task)
            assert stream_q.get_nowait() is None


    @pytest.mark.asyncio
    async def test_stream_cancelled_persists_incomplete_snapshot(self, adapter):
        """Server-side asyncio.CancelledError (shutdown, request timeout) must
        still leave an ``incomplete`` snapshot in ResponseStore so
        GET /v1/responses/{id} and previous_response_id chaining keep
        working.  Regression for PR #15171 follow-up.

        Calls _write_sse_responses directly so the test can await the
        handler to completion (TestClient disconnection races the server
        handler, which makes end-to-end assertion on the final stored
        snapshot flaky).
        """
        # Build a minimal fake request + stream queue the writer understands.
        fake_request = MagicMock()
        fake_request.headers = {}

        written_payloads: list = []

        class _FakeStreamResponse:
            async def prepare(self, req):
                pass

            async def write(self, payload):
                written_payloads.append(payload)

        # Patch web.StreamResponse for the duration of the writer call.
        import gateway.platforms.api_server as api_mod

        # The SSE writers consume an asyncio queue (ThreadSafeAsyncQueue),
        # not a plain queue.Queue — a stdlib queue would block the drain
        # loop's ``await stream_q.get()`` forever.
        stream_q = api_mod.ThreadSafeAsyncQueue()

        async def _agent_coro():
            # Feed one partial delta into the stream queue...
            stream_q.put_nowait("partial output")
            # ...then give the drain loop a moment to pick it up before
            # raising CancelledError to simulate a server-side cancel.
            await asyncio.sleep(0.01)
            raise asyncio.CancelledError()

        agent_task = asyncio.ensure_future(_agent_coro())
        response_id = f"resp_{uuid.uuid4().hex[:28]}"

        with patch.object(api_mod.web, "StreamResponse", return_value=_FakeStreamResponse()):
            with pytest.raises(asyncio.CancelledError):
                await adapter._write_sse_responses(
                    request=fake_request,
                    response_id=response_id,
                    model="hermes-agent",
                    created_at=int(time.time()),
                    stream_q=stream_q,
                    agent_task=agent_task,
                    agent_ref=[None],
                    conversation_history=[],
                    user_message="will be cancelled",
                    instructions=None,
                    conversation=None,
                    store=True,
                    session_id=None,
                )

        # The in_progress snapshot was persisted on response.created,
        # and the CancelledError handler must have updated it to
        # ``incomplete`` with the partial text it saw.
        stored = adapter._response_store.get(response_id)
        assert stored is not None, "snapshot must be retrievable after cancellation"
        assert stored["response"]["status"] == "incomplete"
        # Partial text captured before cancel should be preserved.
        output_text = "".join(
            part.get("text", "")
            for item in stored["response"].get("output", [])
            if item.get("type") == "message"
            for part in item.get("content", [])
        )
        assert "partial output" in output_text

    @pytest.mark.asyncio
    async def test_stream_client_disconnect_persists_incomplete_snapshot(self, adapter):
        """Client disconnect (ConnectionResetError) during streaming must
        persist an ``incomplete`` snapshot in ResponseStore.  Regression
        for PR #15171."""
        fake_request = MagicMock()
        fake_request.headers = {}

        write_call_count = {"n": 0}

        class _DisconnectingStreamResponse:
            async def prepare(self, req):
                pass

            async def write(self, payload):
                # First two writes succeed (prepare + response.created).
                # On the third write (a text delta), the "client"
                # disconnects — simulate with ConnectionResetError.
                write_call_count["n"] += 1
                if write_call_count["n"] >= 3:
                    raise ConnectionResetError("simulated client disconnect")

        import gateway.platforms.api_server as api_mod

        # asyncio queue to match the writers' consumer (see the note in
        # test_stream_cancelled_persists_incomplete_snapshot).
        stream_q = api_mod.ThreadSafeAsyncQueue()
        stream_q.put_nowait("some streamed text")
        stream_q.put_nowait(None)  # EOS sentinel

        async def _agent_coro():
            await asyncio.sleep(0.01)
            return ({"final_response": "", "messages": [], "api_calls": 0},
                    {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0})

        agent_task = asyncio.ensure_future(_agent_coro())
        response_id = f"resp_{uuid.uuid4().hex[:28]}"

        with patch.object(api_mod.web, "StreamResponse", return_value=_DisconnectingStreamResponse()):
            await adapter._write_sse_responses(
                request=fake_request,
                response_id=response_id,
                model="hermes-agent",
                created_at=int(time.time()),
                stream_q=stream_q,
                agent_task=agent_task,
                agent_ref=[None],
                conversation_history=[],
                user_message="will disconnect",
                instructions=None,
                conversation=None,
                store=True,
                session_id=None,
            )

        stored = adapter._response_store.get(response_id)
        assert stored is not None, "snapshot must survive client disconnect"
        assert stored["response"]["status"] == "incomplete"


# ---------------------------------------------------------------------------
# Auth on endpoints
# ---------------------------------------------------------------------------


class TestEndpointAuth:
    @pytest.mark.asyncio
    async def test_chat_completions_requires_auth(self, auth_adapter):
        app = _create_app(auth_adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(
                "/v1/chat/completions",
                json={"model": "test", "messages": [{"role": "user", "content": "hi"}]},
            )
            assert resp.status == 401


# ---------------------------------------------------------------------------
# Config integration
# ---------------------------------------------------------------------------


class TestConfigIntegration:
    def test_platform_enum_has_api_server(self):
        assert Platform.API_SERVER.value == "api_server"


    def test_env_override_cors_origins(self, monkeypatch):
        monkeypatch.setenv("API_SERVER_ENABLED", "true")
        monkeypatch.setenv("API_SERVER_KEY", "opensslrandhex32strongkey")
        monkeypatch.setenv(
            "API_SERVER_CORS_ORIGINS",
            "http://localhost:3000, http://127.0.0.1:3000",
        )
        from gateway.config import load_gateway_config
        config = load_gateway_config()
        assert config.platforms[Platform.API_SERVER].extra.get("cors_origins") == [
            "http://localhost:3000",
            "http://127.0.0.1:3000",
        ]

    def test_api_server_in_connected_platforms(self):
        config = GatewayConfig()
        config.platforms[Platform.API_SERVER] = PlatformConfig(
            enabled=True, extra={"key": "opensslrandhex32strongkey"}
        )
        connected = config.get_connected_platforms()
        assert Platform.API_SERVER in connected


# ---------------------------------------------------------------------------
# Multiple system messages
# ---------------------------------------------------------------------------


class TestMultipleSystemMessages:
    @pytest.mark.asyncio
    async def test_multiple_system_messages_concatenated(self, adapter):
        mock_result = {"final_response": "OK", "messages": [], "api_calls": 1}

        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_run_agent", new_callable=AsyncMock) as mock_run:
                mock_run.return_value = (mock_result, {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0})
                resp = await cli.post(
                    "/v1/chat/completions",
                    json={
                        "model": "hermes-agent",
                        "messages": [
                            {"role": "system", "content": "You are helpful."},
                            {"role": "system", "content": "Be concise."},
                            {"role": "user", "content": "Hello"},
                        ],
                    },
                )

            assert resp.status == 200
            call_kwargs = mock_run.call_args.kwargs
            prompt = call_kwargs["ephemeral_system_prompt"]
            assert "You are helpful." in prompt
            assert "Be concise." in prompt


# ---------------------------------------------------------------------------
# send() method (not used but required by base)
# ---------------------------------------------------------------------------


class TestSendMethod:
    @pytest.mark.asyncio
    async def test_send_returns_not_supported(self):
        config = PlatformConfig(enabled=True)
        adapter = APIServerAdapter(config)
        result = await adapter.send("chat1", "hello")
        assert result.success is False
        assert "HTTP request/response" in result.error


class TestPlatformEventCallbackEndpoint:

    @pytest.mark.asyncio
    async def test_rejects_invalid_google_chat_auth(self, adapter):
        app = _create_app(adapter)
        app["platform_event_adapters"] = {
            "google_chat": _FakeGoogleChatAdapter(
                verify_ok=False,
                verify_code="invalid_google_bearer",
            )
        }

        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(
                "/api/platforms/google_chat/events",
                headers={"Authorization": "Bearer bad"},
                json={"type": "MESSAGE"},
            )
            body = await resp.json()

        assert resp.status == 401
        assert body["error"]["code"] == "invalid_google_bearer"


# ---------------------------------------------------------------------------
# GET /v1/responses/{response_id}
# ---------------------------------------------------------------------------


class TestGetResponse:
    @pytest.mark.asyncio
    async def test_get_stored_response(self, adapter):
        """GET returns a previously stored response."""
        mock_result = {"final_response": "Hello!", "messages": [], "api_calls": 1}

        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            # Create a response first
            with patch.object(adapter, "_run_agent", new_callable=AsyncMock) as mock_run:
                mock_run.return_value = (mock_result, {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15})
                resp = await cli.post(
                    "/v1/responses",
                    json={"model": "hermes-agent", "input": "Hi"},
                )

            assert resp.status == 200
            data = await resp.json()
            response_id = data["id"]

            # Now GET it
            resp2 = await cli.get(f"/v1/responses/{response_id}")
            assert resp2.status == 200
            data2 = await resp2.json()
            assert data2["id"] == response_id
            assert data2["object"] == "response"
            assert data2["status"] == "completed"


# ---------------------------------------------------------------------------
# DELETE /v1/responses/{response_id}
# ---------------------------------------------------------------------------


class TestDeleteResponse:
    @pytest.mark.asyncio
    async def test_delete_stored_response(self, adapter):
        """DELETE removes a stored response and returns confirmation."""
        mock_result = {"final_response": "Hello!", "messages": [], "api_calls": 1}

        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_run_agent", new_callable=AsyncMock) as mock_run:
                mock_run.return_value = (mock_result, {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0})
                resp = await cli.post(
                    "/v1/responses",
                    json={"model": "hermes-agent", "input": "Hi"},
                )

            data = await resp.json()
            response_id = data["id"]

            # Delete it
            resp2 = await cli.delete(f"/v1/responses/{response_id}")
            assert resp2.status == 200
            data2 = await resp2.json()
            assert data2["id"] == response_id
            assert data2["object"] == "response"
            assert data2["deleted"] is True

            # Verify it's gone
            resp3 = await cli.get(f"/v1/responses/{response_id}")
            assert resp3.status == 404


# ---------------------------------------------------------------------------
# Tool calls in output
# ---------------------------------------------------------------------------


class TestToolCallsInOutput:
    @pytest.mark.asyncio
    async def test_tool_calls_in_output(self, adapter):
        """When agent returns tool calls, they appear as function_call items."""
        mock_result = {
            "final_response": "The result is 42.",
            "messages": [
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_abc123",
                            "function": {
                                "name": "calculator",
                                "arguments": '{"expression": "6*7"}',
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_abc123",
                    "content": "42",
                },
                {
                    "role": "assistant",
                    "content": "The result is 42.",
                },
            ],
            "api_calls": 2,
        }

        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_run_agent", new_callable=AsyncMock) as mock_run:
                mock_run.return_value = (mock_result, {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0})
                resp = await cli.post(
                    "/v1/responses",
                    json={"model": "hermes-agent", "input": "What is 6*7?"},
                )

            assert resp.status == 200
            data = await resp.json()
            output = data["output"]

            # Should have: function_call, function_call_output, message
            assert len(output) == 3
            assert output[0]["type"] == "function_call"
            assert output[0]["name"] == "calculator"
            assert output[0]["arguments"] == '{"expression": "6*7"}'
            assert output[0]["call_id"] == "call_abc123"
            # Replayed server-executed calls must be marked completed so
            # OpenAI clients don't treat them as pending calls to execute.
            assert output[0]["status"] == "completed"
            assert output[0]["id"].startswith("fc_")
            assert output[1]["type"] == "function_call_output"
            assert output[1]["call_id"] == "call_abc123"
            assert output[1]["output"] == "42"
            assert output[1]["status"] == "completed"
            assert output[1]["id"].startswith("fco_")
            assert output[2]["type"] == "message"
            assert output[2]["content"][0]["text"] == "The result is 42."


# ---------------------------------------------------------------------------
# Where a stored Responses transcript's current turn starts
# ---------------------------------------------------------------------------


class TestStoredTranscriptTurnStart:
    """``result["messages"]`` IS the agent's live transcript.

    The agent stamps durable bookkeeping onto every message it appends there,
    so prefix detection that compared whole dicts never recognised the history
    it had just been handed. It reported "no shared prefix" for an
    already-complete transcript and the caller concatenated that transcript
    onto itself — storing the owner's own message twice.
    """

    @staticmethod
    def _stamped(role, content, *, at=None, **fields):
        """One message exactly as the live transcript produces it."""
        return stamp_message_timestamp(
            {"role": role, "content": content, **fields}, timestamp=at,
        )

    def test_a_first_turn_stores_the_owner_message_exactly_once(self):
        result = {"messages": [
            self._stamped("user", "Plan the workshop.", at=100.0),
            self._stamped("assistant", "Here is the plan.", at=101.0),
        ]}

        stored = APIServerAdapter._build_response_conversation_history(
            [], "Plan the workshop.", result, "Here is the plan.",
        )

        assert [message["role"] for message in stored] == ["user", "assistant"]
        assert [
            message["content"] for message in stored
            if message["role"] == "user"
        ] == ["Plan the workshop."]
        assert APIServerAdapter._response_messages_turn_start_index(
            [], "Plan the workshop.", result,
        ) == 1

    @pytest.mark.asyncio
    async def test_a_db_persisted_user_only_turn_stores_owner_and_final_once(
        self, adapter,
    ):
        owner = "Plan the workshop."
        reply = json.dumps({
            "schema_version": 1,
            "kind": "question",
            "message": "Which audience should Raphael plan for?",
        })
        result = {"messages": [
            {
                **self._stamped("user", owner, at=100.0),
                "_db_persisted": True,
            },
        ], "final_response": reply, "api_calls": 1}

        conversation = "raphael-owner-" + "7" * 32
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(
                adapter,
                "_run_agent",
                new_callable=AsyncMock,
                return_value=(
                    result,
                    {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
                ),
            ):
                response = await cli.post(
                    "/v1/responses",
                    json={
                        "model": "hermes-agent",
                        "input": owner,
                        "conversation": conversation,
                        "store": True,
                        "expected_previous_response_id": None,
                    },
                    headers={"Idempotency-Key": f"response-{uuid.uuid4().hex}"},
                )
            assert response.status == 200
            response_id = (await response.json())["id"]

        stored = adapter._response_store.get(response_id)
        assert [
            (message["role"], message["content"])
            for message in stored["conversation_history"]
        ] == [
            ("user", owner),
            ("assistant", reply),
        ]
        assert adapter._response_store.owner_history_snapshot(
            conversation,
        )["data"] == [
            {"owner": owner, "raphael": reply},
        ]

    def test_an_unrecognized_nonassistant_tail_is_not_completed(self):
        result = {"messages": [{"role": "tool", "content": "Tool output."}]}

        stored = APIServerAdapter._build_response_conversation_history(
            [], "Plan the workshop.", result, "Here is the plan.",
        )

        assert [(message["role"], message["content"]) for message in stored] == [
            ("user", "Plan the workshop."),
            ("tool", "Tool output."),
        ]

    def test_a_prior_only_transcript_is_not_completed_as_the_current_turn(self):
        prior = [
            self._stamped("user", "First request.", at=100.0),
            self._stamped("assistant", "First reply.", at=101.0),
        ]
        result = {"messages": [dict(message) for message in prior]}

        stored = APIServerAdapter._build_response_conversation_history(
            prior, "Second request.", result, "Second reply.",
        )

        assert stored == result["messages"]

    def test_a_restamped_prior_prefix_is_recognised_not_reappended(self):
        """The agent restamping its copies of prior is bookkeeping, not content."""
        prior = [
            self._stamped("user", "Goal one.", at=100.0),
            self._stamped("assistant", "Understood.", at=101.0),
        ]
        result = {"messages": [
            self._stamped("user", "Goal one.", at=200.0),
            self._stamped("assistant", "Understood.", at=201.0),
            self._stamped("user", "Correction.", at=202.0),
            self._stamped("assistant", "Adjusted.", at=203.0),
        ]}

        stored = APIServerAdapter._build_response_conversation_history(
            prior, "Correction.", result, "Adjusted.",
        )

        assert [
            message["content"] for message in stored
            if message["role"] == "user"
        ] == ["Goal one.", "Correction."]
        # The same index feeds the per-turn transcript, so this turn's output
        # is neither dropped nor replayed with the previous turn's reply.
        assert [
            message["content"] for message in
            APIServerAdapter._turn_transcript_messages(
                prior, "Correction.", result,
            )
        ] == ["Adjusted."]

    def test_two_identical_owner_requests_both_survive_as_two_turns(self):
        """Sending the same words twice is two real turns, not one stored twice.

        The prefix match is positional and greedy from index 0, so each copy
        matches at its own position.
        """
        repeated = "Send it again."
        prior = [
            self._stamped("user", repeated, at=100.0),
            self._stamped("assistant", "Sent once.", at=101.0),
        ]
        result = {"messages": [
            self._stamped("user", repeated, at=200.0),
            self._stamped("assistant", "Sent once.", at=201.0),
            self._stamped("user", repeated, at=202.0),
            self._stamped("assistant", "Sent twice.", at=203.0),
        ]}

        stored = APIServerAdapter._build_response_conversation_history(
            prior, repeated, result, "Sent twice.",
        )

        assert [message["content"] for message in stored] == [
            repeated, "Sent once.", repeated, "Sent twice.",
        ]
        owner_messages = [m for m in stored if m["role"] == "user"]
        assert len(owner_messages) == 2
        assert owner_messages[0] is not owner_messages[1]
        assert owner_messages[0]["timestamp"] != owner_messages[1]["timestamp"]

    def test_tool_call_fields_are_part_of_message_identity(self):
        """Identity is the provider-visible message, not just role and content."""
        call = {"id": "call_1", "function": {"name": "shell", "arguments": "{}"}}
        prior = [
            {"role": "user", "content": "Check it."},
            {"role": "assistant", "content": None, "tool_calls": [call]},
            {"role": "tool", "content": "ok", "tool_call_id": "call_1"},
        ]
        this_turn = [
            self._stamped("user", "And again.", at=202.0),
            self._stamped("assistant", "Checked.", at=203.0),
        ]
        restamped = [
            self._stamped("user", "Check it.", at=200.0),
            self._stamped("assistant", None, at=200.5, tool_calls=[dict(call)]),
            self._stamped("tool", "ok", at=201.0, tool_call_id="call_1"),
        ]

        # Restamped copies of the same tool traffic ARE the same messages.
        assert APIServerAdapter._build_response_conversation_history(
            prior, "And again.", {"messages": restamped + this_turn}, "Checked.",
        ) == restamped + this_turn
        # ...so the output items carry this turn only: the earlier tool call is
        # not replayed as if this turn had made it.
        replayed = {"messages": restamped + this_turn, "final_response": "Checked."}
        assert [
            item["type"] for item in APIServerAdapter._extract_output_items(
                replayed,
                start_index=APIServerAdapter._response_messages_turn_start_index(
                    prior, "And again.", replayed,
                ),
            )
        ] == ["message"]

        # A DIFFERENT tool call at the same position is a different message, so
        # the transcript is not claimed to already carry the prior history.
        divergent = [
            self._stamped("user", "Check it.", at=200.0),
            self._stamped("assistant", None, at=200.5, tool_calls=[
                {"id": "call_2", "function": {"name": "shell", "arguments": "{}"}},
            ]),
            self._stamped("tool", "ok", at=201.0, tool_call_id="call_2"),
        ]
        assert APIServerAdapter._response_messages_turn_start_index(
            prior, "And again.", {"messages": divergent + this_turn},
        ) == 0

    def test_a_compressed_transcript_is_used_directly_and_never_concatenated(
        self,
    ):
        prior = [
            self._stamped("user", "Goal one.", at=100.0),
            self._stamped("assistant", "Understood.", at=101.0),
        ]
        compressed = [
            {"role": "system", "content": "Summary of earlier turns."},
            self._stamped("user", "Correction.", at=202.0),
            self._stamped("assistant", "Adjusted.", at=203.0),
        ]

        # A compressed transcript shares no prefix with prior and is stored as
        # it stands, or the uncompressed history rides back in on front.
        assert APIServerAdapter._build_response_conversation_history(
            prior, "Correction.",
            {"messages": compressed, "_compressed": True},
            "Adjusted.",
        ) == compressed

        # And that short circuit stays the compression path alone: a stamped
        # transcript that DOES carry the prior prefix is recognised on its own,
        # without needing the flag to avoid concatenation.
        uncompressed = [
            self._stamped("user", "Goal one.", at=200.0),
            self._stamped("assistant", "Understood.", at=201.0),
            self._stamped("user", "Correction.", at=202.0),
            self._stamped("assistant", "Adjusted.", at=203.0),
        ]
        assert APIServerAdapter._build_response_conversation_history(
            prior, "Correction.", {"messages": uncompressed}, "Adjusted.",
        ) == uncompressed

    def test_the_callers_prior_context_is_never_mutated(self):
        prior = [
            self._stamped("user", "Goal one.", at=100.0),
            self._stamped("assistant", "Understood.", at=101.0),
        ]
        before = json.loads(json.dumps(prior))
        member_ids = [id(message) for message in prior]

        stored = APIServerAdapter._build_response_conversation_history(
            prior,
            "Correction.",
            {"messages": [
                self._stamped("user", "Goal one.", at=200.0),
                self._stamped("assistant", "Understood.", at=201.0),
                self._stamped("user", "Correction.", at=202.0),
                self._stamped("assistant", "Adjusted.", at=203.0),
            ]},
            "Adjusted.",
        )

        # The turn is stored once...
        assert [
            message["content"] for message in stored
            if message["role"] == "user"
        ] == ["Goal one.", "Correction."]
        # ...and the prior context the caller handed in is untouched: same
        # list, same member dicts, same contents.
        assert stored is not prior
        assert prior == before
        assert [id(message) for message in prior] == member_ids


# ---------------------------------------------------------------------------
# Usage / token counting
# ---------------------------------------------------------------------------


class TestUsageCounting:
    @pytest.mark.asyncio
    async def test_responses_usage(self, adapter):
        """Responses API returns real token counts."""
        mock_result = {"final_response": "Done", "messages": [], "api_calls": 1}
        usage = {"input_tokens": 100, "output_tokens": 50, "total_tokens": 150}

        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_run_agent", new_callable=AsyncMock) as mock_run:
                mock_run.return_value = (mock_result, usage)
                resp = await cli.post(
                    "/v1/responses",
                    json={"model": "hermes-agent", "input": "Hi"},
                )

            assert resp.status == 200
            data = await resp.json()
            assert data["usage"]["input_tokens"] == 100
            assert data["usage"]["output_tokens"] == 50
            assert data["usage"]["total_tokens"] == 150


# ---------------------------------------------------------------------------
# Truncation
# ---------------------------------------------------------------------------


class TestTruncation:


    @pytest.mark.asyncio
    async def test_truncation_auto_preserves_non_leading_compaction_summary(self, adapter):
        """A summary sitting after a retained system head must survive too.

        The gateway /compress path can force a user-leading layout that
        leaves the compaction summary after a kept system message, so the
        preservation predicate must not assume the summary is at index 0.
        """
        mock_result = {"final_response": "OK", "messages": [], "api_calls": 1}

        system_head = {"role": "system", "content": "You are a helpful agent."}
        summary = {
            "role": "user",
            "content": "[CONTEXT COMPACTION — REFERENCE ONLY]\nEarlier work.",
            "_compressed_summary": True,
        }
        long_history = [system_head, summary] + [
            {"role": "user", "content": f"msg {i}"}
            for i in range(148)
        ]
        adapter._response_store.put("resp_summary_mid", {
            "response": {"id": "resp_summary_mid", "object": "response"},
            "conversation_history": long_history,
            "instructions": None,
        })

        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_run_agent", new_callable=AsyncMock) as mock_run:
                mock_run.return_value = (mock_result, {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0})
                resp = await cli.post(
                    "/v1/responses",
                    json={
                        "model": "hermes-agent",
                        "input": "follow up",
                        "previous_response_id": "resp_summary_mid",
                        "truncation": "auto",
                    },
                )

        assert resp.status == 200
        history = mock_run.call_args.kwargs["conversation_history"]
        assert len(history) == 100
        assert history[0] == summary
        assert history[1]["content"] == "msg 49"
        assert history[-1]["content"] == "msg 147"


# ---------------------------------------------------------------------------
# Response-side truncation / failure handling (issue #22496)
# ---------------------------------------------------------------------------


class TestChatCompletionsAgentIncomplete:
    """When the agent run yields a partial / failed result, the API server
    must NOT pretend it succeeded. Either signal truncation via
    finish_reason='length' (with the partial text), or 502 with an OpenAI
    error envelope (no usable text). Issue #22496."""


    @pytest.mark.asyncio
    async def test_hard_failure_redacts_secret_like_error_text(self, adapter):
        raw_secret = "sk-api-server-leak-1234567890"
        mock_result = {
            "final_response": "",
            "completed": False,
            "partial": False,
            "failed": True,
            "error": f"provider auth failed OPENAI_API_KEY={raw_secret}",
            "messages": [],
            "api_calls": 1,
        }
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_run_agent", new_callable=AsyncMock) as mock_run:
                mock_run.return_value = (mock_result, {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0})
                resp = await cli.post(
                    "/v1/chat/completions",
                    json={"model": "hermes-agent", "messages": [{"role": "user", "content": "hello"}]},
                )

            assert resp.status == 502
            data = await resp.json()
            body = json.dumps(data)
            assert raw_secret not in body
            assert raw_secret not in resp.headers.get("X-Hermes-Error", "")
            assert "OPENAI_API_KEY=" in body
            assert data["error"]["hermes"]["failed"] is True


# ---------------------------------------------------------------------------
# CORS
# ---------------------------------------------------------------------------


class TestCORS:
    def test_origin_allowed_for_non_browser_client(self, adapter):
        assert adapter._origin_allowed("") is True


    def test_origin_allowed_for_allowlist_match(self):
        adapter = _make_adapter(cors_origins=["http://localhost:3000"])
        assert adapter._origin_allowed("http://localhost:3000") is True


    @pytest.mark.asyncio
    async def test_browser_origin_rejected_by_default(self, adapter):
        """Browser-originated requests are rejected unless explicitly allowed."""
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.get("/health", headers={"Origin": "http://evil.example"})
            assert resp.status == 403
            assert resp.headers.get("Access-Control-Allow-Origin") is None


    @pytest.mark.asyncio
    async def test_cors_allows_idempotency_key_header(self):
        adapter = _make_adapter(cors_origins=["http://localhost:3000"])
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.options(
                "/v1/chat/completions",
                headers={
                    "Origin": "http://localhost:3000",
                    "Access-Control-Request-Method": "POST",
                    "Access-Control-Request-Headers": "Idempotency-Key",
                },
            )
            assert resp.status == 200
            assert "Idempotency-Key" in resp.headers.get("Access-Control-Allow-Headers", "")


    @pytest.mark.asyncio
    async def test_cors_options_preflight_allowed_for_configured_origin(self):
        """Configured origins can complete browser preflight."""
        adapter = _make_adapter(cors_origins=["http://localhost:3000"])
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.options(
                "/v1/chat/completions",
                headers={
                    "Origin": "http://localhost:3000",
                    "Access-Control-Request-Method": "POST",
                    "Access-Control-Request-Headers": "Authorization, Content-Type",
                },
            )
            assert resp.status == 200
            assert resp.headers.get("Access-Control-Allow-Origin") == "http://localhost:3000"
            assert "Authorization" in resp.headers.get("Access-Control-Allow-Headers", "")


# ---------------------------------------------------------------------------
# Conversation parameter
# ---------------------------------------------------------------------------


class TestConversationParameter:


    @pytest.mark.asyncio
    async def test_separate_conversations_are_isolated(self, adapter):
        """Different conversation names have independent histories."""
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_run_agent", new_callable=AsyncMock) as mock_run:
                mock_run.return_value = (
                    {"final_response": "Response A", "messages": [], "api_calls": 1},
                    {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
                )
                # Conversation A
                await cli.post("/v1/responses", json={"input": "conv-a msg", "conversation": "conv-a"})
                # Conversation B
                mock_run.return_value = (
                    {"final_response": "Response B", "messages": [], "api_calls": 1},
                    {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
                )
                await cli.post("/v1/responses", json={"input": "conv-b msg", "conversation": "conv-b"})

                # They should have different response IDs in the mapping
                assert adapter._response_store.get_conversation("conv-a") != adapter._response_store.get_conversation("conv-b")


    @pytest.mark.asyncio
    async def test_conversation_reuse_after_eviction_no_404(self, adapter):
        """After eviction clears a conversation mapping, reusing that name starts fresh (no 404)."""
        adapter._response_store = ResponseStore(max_size=1)
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_run_agent", new_callable=AsyncMock) as mock_run:
                mock_run.return_value = (
                    {"final_response": "First", "messages": [], "api_calls": 1},
                    {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
                )
                # Create conversation -> resp stored
                resp1 = await cli.post("/v1/responses", json={
                    "input": "hello",
                    "conversation": "my-chat",
                })
                assert resp1.status == 200

                # Evict by adding another response
                mock_run.return_value = (
                    {"final_response": "Other", "messages": [], "api_calls": 1},
                    {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
                )
                await cli.post("/v1/responses", json={"input": "other"})

                # Conversation mapping should have been cleaned by eviction
                assert adapter._response_store.get_conversation("my-chat") is None

                # Reuse conversation name — should start fresh, not 404
                mock_run.return_value = (
                    {"final_response": "Restarted", "messages": [], "api_calls": 1},
                    {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
                )
                resp3 = await cli.post("/v1/responses", json={
                    "input": "hello again",
                    "conversation": "my-chat",
                })
                assert resp3.status == 200


# ---------------------------------------------------------------------------
# X-Hermes-Session-Id header (session continuity)
# ---------------------------------------------------------------------------


class TestSessionIdHeader:


    @pytest.mark.asyncio
    async def test_traversal_session_id_header_rejected(self, auth_adapter):
        """Security (#5958): a path-traversal X-Hermes-Session-Id must be
        rejected with 400 so it can't reach the filesystem artifact paths
        (session snapshot / request dump) and escape the sessions dir."""
        app = _create_app(auth_adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(auth_adapter, "_run_agent", new_callable=AsyncMock) as mock_run:
                for bad in ("../../../../etc/pwned", "/abs/path", "..\\win"):
                    resp = await cli.post(
                        "/v1/chat/completions",
                        headers={"X-Hermes-Session-Id": bad, "Authorization": "Bearer sk-secret"},
                        json={"model": "hermes-agent", "messages": [{"role": "user", "content": "hi"}]},
                    )
                    assert resp.status == 400, f"{bad!r} should be rejected"
                # The agent is never invoked for a rejected ID.
                assert mock_run.call_count == 0

    @pytest.mark.asyncio
    async def test_provided_session_id_loads_history_from_db(self, auth_adapter):
        """When X-Hermes-Session-Id is provided, history comes from SessionDB not request body."""
        mock_result = {"final_response": "OK", "messages": [], "api_calls": 1}
        db_history = [
            {"role": "user", "content": "stored message 1"},
            {"role": "assistant", "content": "stored reply 1"},
        ]
        mock_db = MagicMock()
        mock_db.get_messages_as_conversation.return_value = db_history
        auth_adapter._session_db = mock_db
        app = _create_app(auth_adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(auth_adapter, "_run_agent", new_callable=AsyncMock) as mock_run:
                mock_run.return_value = (mock_result, {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0})

                resp = await cli.post(
                    "/v1/chat/completions",
                    headers={"X-Hermes-Session-Id": "existing-session", "Authorization": "Bearer sk-secret"},
                    # Request body has different history — should be ignored
                    json={
                        "model": "hermes-agent",
                        "messages": [
                            {"role": "user", "content": "old msg from client"},
                            {"role": "assistant", "content": "old reply from client"},
                            {"role": "user", "content": "new question"},
                        ],
                    },
                )

            assert resp.status == 200
            call_kwargs = mock_run.call_args.kwargs
            # History must come from DB, not from the request body
            assert call_kwargs["conversation_history"] == db_history
            assert call_kwargs["user_message"] == "new question"


# ---------------------------------------------------------------------------
# X-Hermes-Session-Key header (long-term memory scoping)
# ---------------------------------------------------------------------------


class TestSessionKeyHeader:
    """The session key is a stable per-channel identifier that scopes
    long-term memory (e.g. Honcho) independently of the transcript-scoped
    session_id.  A third-party Web UI passes one stable key per assistant
    channel and rotates session_id on /new, matching the native
    gateway's session_key / session_id split.
    """


    @pytest.mark.asyncio
    async def test_session_key_threads_into_create_agent(self, auth_adapter):
        """End-to-end: verify AIAgent(gateway_session_key=...) receives the key via _create_agent."""
        captured_kwargs = {}

        def _fake_create_agent(**kwargs):
            captured_kwargs.update(kwargs)
            mock_agent = MagicMock()
            mock_agent.run_conversation.return_value = {"final_response": "ok", "messages": []}
            mock_agent.session_prompt_tokens = 0
            mock_agent.session_completion_tokens = 0
            mock_agent.session_total_tokens = 0
            return mock_agent

        app = _create_app(auth_adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(auth_adapter, "_create_agent", side_effect=_fake_create_agent):
                resp = await cli.post(
                    "/v1/chat/completions",
                    headers={
                        "X-Hermes-Session-Key": "agent:main:webui:dm:user-7",
                        "Authorization": "Bearer sk-secret",
                    },
                    json={"model": "hermes-agent", "messages": [{"role": "user", "content": "hi"}]},
                )
            assert resp.status == 200
            # _create_agent must be called with gateway_session_key threaded through
            assert captured_kwargs.get("gateway_session_key") == "agent:main:webui:dm:user-7"

    @pytest.mark.asyncio
    async def test_responses_endpoint_accepts_session_key(self, auth_adapter):
        """Responses API honors the same X-Hermes-Session-Key contract."""
        mock_result = {"final_response": "ok", "messages": [], "api_calls": 1}
        app = _create_app(auth_adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(auth_adapter, "_run_agent", new_callable=AsyncMock) as mock_run:
                mock_run.return_value = (mock_result, {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0})
                resp = await cli.post(
                    "/v1/responses",
                    headers={
                        "X-Hermes-Session-Key": "webui:chan-1",
                        "Authorization": "Bearer sk-secret",
                    },
                    json={"model": "hermes-agent", "input": "hello", "store": False},
                )
            assert resp.status == 200
            assert resp.headers.get("X-Hermes-Session-Key") == "webui:chan-1"
            call_kwargs = mock_run.call_args.kwargs
            assert call_kwargs["gateway_session_key"] == "webui:chan-1"

    @pytest.mark.asyncio
    async def test_capabilities_advertises_session_key_header(self, adapter):
        """GET /v1/capabilities should advertise the new header so clients can feature-detect."""
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.get("/v1/capabilities")
            assert resp.status == 200
            data = await resp.json()
            assert data["features"]["session_key_header"] == "X-Hermes-Session-Key"


# ---------------------------------------------------------------------------
# Per-client model routing (model_routes)
# ---------------------------------------------------------------------------


def _make_routing_adapter(routes) -> APIServerAdapter:
    """Create an adapter with model_routes configured."""
    config = PlatformConfig(enabled=True, extra={"model_routes": routes})
    return APIServerAdapter(config)


def _patch_create_agent_runtime(monkeypatch, captured: dict, fake_agent_cls):
    """Stub out every external dependency of _create_agent."""
    monkeypatch.setattr("run_agent.AIAgent", fake_agent_cls)
    monkeypatch.setattr(
        "gateway.run._resolve_runtime_agent_kwargs",
        lambda: {
            "provider": "openrouter",
            "api_key": "sk-global",
            "base_url": "https://openrouter.ai/api/v1",
            "api_mode": "chat_completions",
        },
    )
    monkeypatch.setattr("gateway.run._resolve_gateway_model", lambda: "global/model")
    monkeypatch.setattr("gateway.run._load_gateway_config", lambda: {})
    monkeypatch.setattr(
        "gateway.run.GatewayRunner._load_reasoning_config", staticmethod(lambda model="": {})
    )
    monkeypatch.setattr(
        "gateway.run.GatewayRunner._load_fallback_model", staticmethod(lambda: None)
    )
    monkeypatch.setattr("gateway.run._current_max_iterations", lambda: 90)
    monkeypatch.setattr("hermes_cli.tools_config._get_platform_tools", lambda *_: set())


class TestModelRoutesParsing:
    def test_valid_routes_are_parsed(self):
        routes = {"minimax-m2": {"model": "minimax/minimax-m1", "provider": "openrouter"}}
        adapter = _make_routing_adapter(routes)
        assert adapter._model_routes == routes


    def test_route_without_model_is_dropped(self):
        adapter = _make_routing_adapter({"bad": {"provider": "openrouter"}})
        assert adapter._model_routes == {}


class TestModelRoutesModelsEndpoint:

    @pytest.mark.asyncio
    async def test_models_endpoint_route_alias_fields_and_no_secrets(self):
        routes = {"my-alias": {"model": "openai/gpt-5", "api_key": "sk-route-secret"}}
        adapter = _make_routing_adapter(routes)
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.get("/v1/models")
            data = await resp.json()
            alias_entry = next(m for m in data["data"] if m["id"] == "my-alias")
            assert alias_entry["root"] == "openai/gpt-5"
            assert alias_entry["parent"] == adapter._model_name
            # per-route api_key must never leak through the discovery endpoint
            assert "sk-route-secret" not in json.dumps(data)


class TestModelRoutesHandlers:
    @pytest.mark.asyncio
    async def test_chat_completions_passes_route_to_run_agent(self):
        routes = {"minimax-m2": {"model": "minimax/minimax-m1", "provider": "openrouter"}}
        adapter = _make_routing_adapter(routes)
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_run_agent", new_callable=AsyncMock) as mock_run:
                mock_run.return_value = (
                    {"final_response": "hi", "messages": [], "api_calls": 1},
                    {"input_tokens": 5, "output_tokens": 5, "total_tokens": 10},
                )
                resp = await cli.post("/v1/chat/completions", json={
                    "model": "minimax-m2",
                    "messages": [{"role": "user", "content": "hello"}],
                })
                assert resp.status == 200
                kwargs = mock_run.call_args.kwargs
                assert kwargs.get("route") == {
                    "model": "minimax/minimax-m1", "provider": "openrouter",
                }


class TestModelRoutesAgentCreation:

    def test_route_provider_resolves_provider_credentials(self, monkeypatch):
        captured = {}

        class FakeAgent:
            def __init__(self, **kwargs):
                captured.update(kwargs)

        _patch_create_agent_runtime(monkeypatch, captured, FakeAgent)
        monkeypatch.setattr(
            "gateway.run._resolve_runtime_agent_kwargs_for_provider",
            lambda provider: {
                "provider": provider,
                "api_key": f"sk-{provider}",
                "base_url": f"https://{provider}.example/v1",
                "api_mode": "chat_completions",
            },
        )
        adapter = _make_routing_adapter(
            {"alias": {"model": "other/model", "provider": "otherprov"}}
        )
        monkeypatch.setattr(adapter, "_ensure_session_db", lambda: None)
        monkeypatch.setattr(adapter, "_session_model_override_for", lambda *_: None)

        adapter._create_agent(session_id="s1", route=adapter._resolve_route("alias"))

        assert captured["model"] == "other/model"
        assert captured["provider"] == "otherprov"
        assert captured["api_key"] == "sk-otherprov"


    def test_session_model_override_beats_route(self, monkeypatch):
        """A user-issued /model on the session must win over static route config."""
        captured = {}

        class FakeAgent:
            def __init__(self, **kwargs):
                captured.update(kwargs)

        _patch_create_agent_runtime(monkeypatch, captured, FakeAgent)
        adapter = _make_routing_adapter({"alias": {"model": "route/model", "api_key": "sk-route"}})
        monkeypatch.setattr(adapter, "_ensure_session_db", lambda: None)
        monkeypatch.setattr(
            adapter,
            "_session_model_override_for",
            lambda key: {
                "model": "session/override-model",
                "provider": "sessionprov",
                "api_key": "sk-session",
                "base_url": "https://session.example/v1",
                "api_mode": "responses",
                "credential_pool": "pool-session",
            },
        )

        adapter._create_agent(session_id="s1", route=adapter._resolve_route("alias"))

        assert captured["model"] == "session/override-model"
        assert captured["provider"] == "sessionprov"
        assert captured["api_key"] == "sk-session"


# ---------------------------------------------------------------------------
# Event-loop offloading for synchronous SessionDB calls (P1)
# ---------------------------------------------------------------------------


class TestSessionDbOffEventLoop:
    """Regression: synchronous SessionDB calls in the OpenAI-compatible API
    server must run OFF the aiohttp event loop. A blocking SQLite read/write on
    the loop freezes every in-flight request under load (same class of bug as
    gateway build_channel_directory, #60794 / #60810), so each call is wrapped
    in asyncio.to_thread.
    """

    @pytest.mark.asyncio
    async def test_get_existing_session_or_404_offloads(self, auth_adapter):
        import threading

        captured = {}

        class FakeDB:
            def get_session(self, session_id):
                captured["thread"] = threading.current_thread()
                return {"id": session_id, "source": "api_server"}

        auth_adapter._session_db = FakeDB()
        session, err = await auth_adapter._get_existing_session_or_404("sess-x")
        assert err is None
        assert session["id"] == "sess-x"
        # The blocking DB call must NOT execute on the event-loop thread.
        assert captured["thread"] is not None
        assert captured["thread"] != threading.current_thread()


# ---------------------------------------------------------------------------
# _api_key_passes_startup_guard — fail-closed on an unverifiable key
# ---------------------------------------------------------------------------

class TestApiKeyStartupGuardFailsClosed:
    """The guard is the only thing between a guessable key and an endpoint the
    code itself describes as ``terminal-capable agent work`` where "a guessable
    key is remote code execution".

    So "the strength check could not be run" must never resolve to "start
    anyway" — the same posture ``tools/credential_files.py`` takes when its
    deny-list cannot be consulted.
    """

    class _Stub:
        name = "api_server"
        _host = "0.0.0.0"

        def __init__(self, key):
            self._api_key = key

    @staticmethod
    def _guard(key):
        return APIServerAdapter._api_key_passes_startup_guard(
            TestApiKeyStartupGuardFailsClosed._Stub(key)
        )

    @staticmethod
    def _blocking_auth_import():
        real_import = __import__

        def _blocked(name, *args, **kwargs):
            if name == "hermes_cli.auth":
                raise ImportError("simulated: hermes_cli.auth unavailable")
            return real_import(name, *args, **kwargs)

        return patch("builtins.__import__", _blocked)

    def test_weak_key_refused_when_check_is_unavailable(self):
        """The bug: an unimportable auth module silently dropped the check and
        the server started on a 4-character key."""
        with self._blocking_auth_import():
            assert self._guard("test") is False

    def test_strong_key_also_refused_when_check_is_unavailable(self):
        """Fail-closed: we cannot verify the key, so we do not expose the
        endpoint — the log tells the operator to repair the install."""
        with self._blocking_auth_import():
            assert self._guard("a" * 40) is False


class TestKeyRejectionSetsNonRetryableFatalError:
    """Each startup-guard rejection must set a non-retryable fatal error so
    the reconnect watcher drops the platform from the retry queue instead of
    looping indefinitely.

    Previously connect() returned bare ``False``, which gateway.run treated
    as retryable — re-queueing every backoff interval forever and
    re-instantiating the adapter (with its ResponseStore sqlite connection)
    each retry (#38803: ~501 leaked connections / 1002 fds over 2.5 days,
    ending in EMFILE for the whole gateway). Mirrors the port-conflict
    precedent (test_port_conflict_sets_non_retryable_fatal_error, #65665).
    """

    @staticmethod
    def _make_adapter(key, monkeypatch):
        monkeypatch.delenv("API_SERVER_KEY", raising=False)
        return APIServerAdapter(
            PlatformConfig(
                enabled=True,
                extra={"host": "127.0.0.1", "port": 0, "key": key},
            )
        )

    @staticmethod
    async def _assert_key_rejection_is_fatal(adapter):
        try:
            assert await adapter.connect() is False
            assert adapter.has_fatal_error is True
            assert adapter.fatal_error_retryable is False
            assert adapter.fatal_error_code == "api_server_key_invalid"
            assert "API_SERVER_KEY" in (adapter.fatal_error_message or "")
        finally:
            await adapter.disconnect()

    @pytest.mark.asyncio
    async def test_missing_key_sets_non_retryable_fatal_error(self, monkeypatch):
        adapter = self._make_adapter("", monkeypatch)
        await self._assert_key_rejection_is_fatal(adapter)


# ---------------------------------------------------------------------------
# Bare-model opt-in gate (direct_model_requests) for _request_agent_overrides
# ---------------------------------------------------------------------------


class TestDirectModelRequestsGate:
    """Bare ``model`` (no ``provider``) is opt-in on OpenAI-compatible
    endpoints so generic clients hardcoding "gpt-4o" keep falling back to
    the gateway default (idea credit: PR #22825 by @mssteuer)."""

    def test_bare_model_dropped_when_disallowed(self):
        overrides = _request_agent_overrides(
            {"model": "openai/gpt-5"}, allow_bare_model=False
        )
        assert "requested_model" not in overrides


    def test_adapter_flag_opt_in(self):
        adapter = APIServerAdapter(
            PlatformConfig(enabled=True, extra={"direct_model_requests": True})
        )
        assert adapter._direct_model_requests is True


    @pytest.mark.asyncio
    async def test_chat_completions_bare_model_honored_when_enabled(self):
        adapter = APIServerAdapter(
            PlatformConfig(enabled=True, extra={"direct_model_requests": True})
        )
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_run_agent", new_callable=AsyncMock) as mock_run:
                mock_run.return_value = (
                    {"final_response": "ok", "messages": [], "api_calls": 1},
                    {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
                )
                resp = await cli.post(
                    "/v1/chat/completions",
                    json={
                        "model": "openai/gpt-5",
                        "messages": [{"role": "user", "content": "hi"}],
                    },
                )
        assert resp.status == 200
        assert mock_run.call_args.kwargs.get("requested_model") == "openai/gpt-5"


class TestRouteWithoutModelKeepsDefault:
    """A model_routes alias whose route has no ``model`` key must keep the
    global default model — the alias string itself is never a model name."""

    def test_alias_never_leaks_as_model(self, monkeypatch):
        captured = {}

        class FakeAgent:
            def __init__(self, **kwargs):
                captured.update(kwargs)

        _patch_create_agent_runtime(monkeypatch, captured, FakeAgent)
        adapter = _make_routing_adapter(
            {"alias": {"model": "", "api_key": "sk-route"}}
        )
        # _parse_model_routes drops routes without model; simulate a
        # credentials-only route surviving via direct dict (defensive path).
        adapter._model_routes = {"alias": {"api_key": "sk-route"}}
        monkeypatch.setattr(adapter, "_ensure_session_db", lambda: None)
        monkeypatch.setattr(adapter, "_session_model_override_for", lambda *_: None)

        adapter._create_agent(
            session_id="s1",
            route=adapter._resolve_route("alias"),
            requested_model="alias",
        )

        assert captured["model"] == "global/model"
        assert captured["api_key"] == "sk-route"


# ---------------------------------------------------------------------------
# Empty-model recovery + provider-auth error typing in _create_agent
# (salvaged from PR #57947 by @FvanW)
# ---------------------------------------------------------------------------


class TestCreateAgentModelRecovery:
    def test_create_agent_defaults_to_provider_catalog_model_when_empty(self, monkeypatch):
        """api_server.py had no equivalent of run.py's provider-catalog
        default when model resolves empty but a provider did resolve (e.g.
        `hermes auth add openai-codex` without `hermes model`) —
        AIAgent(model="") 400s every call."""
        captured = {}

        class FakeAgent:
            def __init__(self, **kwargs):
                captured.update(kwargs)

        _patch_create_agent_runtime(monkeypatch, captured, FakeAgent)
        monkeypatch.setattr(
            "gateway.run._resolve_runtime_agent_kwargs",
            lambda: {"provider": "openai-codex", "base_url": "https://example.test/v1",
                     "api_mode": "codex_responses"},
        )
        monkeypatch.setattr("gateway.run._resolve_gateway_model", lambda: "")
        monkeypatch.setattr(
            "hermes_cli.models.get_default_model_for_provider",
            lambda provider: "gpt-5.5-codex" if provider == "openai-codex" else None,
        )

        adapter = APIServerAdapter(PlatformConfig(enabled=True))
        monkeypatch.setattr(adapter, "_ensure_session_db", lambda: None)

        agent = adapter._create_agent(session_id="api-session")

        assert isinstance(agent, FakeAgent)
        assert captured["model"] == "gpt-5.5-codex"

    def test_create_agent_recovers_last_known_good_model_when_empty(self, monkeypatch):
        """Last-known-good recovery (#35314): a transient config-cache miss
        producing an empty model would build AIAgent(model="") and fail every
        call until manual retry, instead of reusing the model that just
        worked."""
        captured = []

        class FakeAgent:
            def __init__(self, **kwargs):
                captured.append(dict(kwargs))

        _patch_create_agent_runtime(monkeypatch, {}, FakeAgent)
        monkeypatch.setattr("run_agent.AIAgent", FakeAgent)

        adapter = APIServerAdapter(PlatformConfig(enabled=True))
        monkeypatch.setattr(adapter, "_ensure_session_db", lambda: None)

        # Turn 1: model resolves fine — populates the last-known-good cache
        # (keyed on gateway_session_key).
        monkeypatch.setattr("gateway.run._resolve_gateway_model", lambda: "minimax/minimax-m3")
        adapter._create_agent(session_id="api-session", gateway_session_key="stable-chan-1")
        assert captured[0]["model"] == "minimax/minimax-m3"
        assert adapter._last_resolved_model["stable-chan-1"] == "minimax/minimax-m3"

        # Turn 2: transient empty resolution, no provider catalog default —
        # must recover the model from turn 1, not build model="".
        monkeypatch.setattr("gateway.run._resolve_gateway_model", lambda: "")
        monkeypatch.setattr(
            "gateway.run._resolve_runtime_agent_kwargs",
            lambda: {"provider": None, "base_url": None, "api_mode": None},
        )
        adapter._create_agent(session_id="another-session", gateway_session_key="stable-chan-1")
        assert captured[1]["model"] == "minimax/minimax-m3"


# ---------------------------------------------------------------------------
# Owner authority requires durable storage
# ---------------------------------------------------------------------------


_AUTHORITY_CONVERSATION = "raphael-owner-" + "d" * 32
_AUTHORITY_RESPONSE = "resp_authority_probe"
_AUTHORITY_CLAIM = "claim_" + "e" * 32
_AUTHORITY_RUN = "run_" + "f" * 32


def _owner_authority_probes(store):
    """One call per owner-authoritative surface the store is the authority for.

    Named after the four the guard's docstring calls out: proposal, claim,
    conversation closure and run idempotency.
    """
    return {
        "history": lambda: store.owner_history_snapshot(_AUTHORITY_CONVERSATION),
        "history_compat": lambda: store.owner_history(_AUTHORITY_CONVERSATION),
        "proposal_record": lambda: store.owner_proposal_record(
            "default", _AUTHORITY_CONVERSATION, _AUTHORITY_RESPONSE,
        ),
        # A run is refused while this answers "yes", so a store that cannot
        # answer at all must refuse rather than report "nothing outstanding".
        "request_is_unanswered": lambda: store.owner_request_is_unanswered(
            "default", _AUTHORITY_CONVERSATION,
        ),
        "proposal_consumed": lambda: store.mark_owner_proposal_consumed(
            "default", _AUTHORITY_CONVERSATION, _AUTHORITY_RESPONSE,
        ),
        "claim": lambda: store.claim_owner_proposal(
            "default", _AUTHORITY_CONVERSATION, _AUTHORITY_RESPONSE, _AUTHORITY_CLAIM,
        ),
        "claim_and_attach": lambda: store.claim_and_attach_owner_run(
            "default", _AUTHORITY_CONVERSATION, _AUTHORITY_RESPONSE,
            _AUTHORITY_CLAIM, _AUTHORITY_RUN,
        ),
        "attach_run": lambda: store.attach_owner_run(
            "default", _AUTHORITY_CONVERSATION, _AUTHORITY_RESPONSE,
            _AUTHORITY_CLAIM, _AUTHORITY_RUN,
        ),
        "complete_claim": lambda: store.complete_owner_claim(
            "default", _AUTHORITY_CONVERSATION, _AUTHORITY_RESPONSE,
            _AUTHORITY_CLAIM, _AUTHORITY_RUN,
        ),
        "release_claim": lambda: store.release_owner_claim(
            "default", _AUTHORITY_CONVERSATION, _AUTHORITY_RESPONSE,
            _AUTHORITY_CLAIM, _AUTHORITY_RUN,
        ),
        "close_conversation": lambda: store.close_owner_conversation(
            "default", _AUTHORITY_CONVERSATION, _AUTHORITY_RESPONSE,
        ),
        "reserve_turn": lambda: store.reserve_owner_conversation(
            "default", _AUTHORITY_CONVERSATION, _AUTHORITY_RESPONSE,
        ),
        "reserve_run_idempotency": lambda: store.reserve_run_idempotency(
            "default", "scope", "idem-1", "fingerprint", _AUTHORITY_RUN,
        ),
        "lookup_run_idempotency": lambda: store.lookup_run_idempotency(
            "default", "scope", "idem-1", "fingerprint",
        ),
        "run_completion": lambda: store.owner_run_completion("default", _AUTHORITY_RUN),
        "session_index": lambda: store.owner_session_index("default", "group"),
        "map_owner_conversation": lambda: store.set_conversation(
            _AUTHORITY_CONVERSATION, _AUTHORITY_RESPONSE,
        ),
    }


def _assert_no_owner_authority(store):
    unguarded = []
    for name, probe in _owner_authority_probes(store).items():
        try:
            probe()
        except OwnerAuthorityUnavailable:
            continue
        except Exception as exc:
            unguarded.append(f"{name}: raised {type(exc).__name__} instead")
            continue
        unguarded.append(f"{name}: answered instead of refusing")
    assert unguarded == []


def _assert_generic_traffic_still_works(store):
    """Generic Responses traffic is explicitly NOT authority — memory is fine."""
    store.put("resp_generic", {"output": "hello"})
    assert store.get("resp_generic") == {"output": "hello"}
    assert store.set_conversation("chat-session-1", "resp_generic") is True
    assert store.get_conversation("chat-session-1") == "resp_generic"


class TestOwnerAuthorityRequiresDurableStorage:
    def test_an_unresolvable_store_path_grants_no_owner_authority(self, monkeypatch):
        import hermes_cli.config as hermes_config

        def _no_home():
            raise RuntimeError("HERMES_HOME cannot be resolved")

        monkeypatch.setattr(hermes_config, "get_hermes_home", _no_home)
        store = ResponseStore(max_size=10)
        try:
            assert store._db_path is None
            _assert_no_owner_authority(store)
            _assert_generic_traffic_still_works(store)
        finally:
            store.close()

    def test_an_unopenable_store_grants_no_owner_authority(self, tmp_path):
        unwritable = tmp_path / "read-only"
        unwritable.mkdir()
        unwritable.chmod(0o500)
        try:
            store = ResponseStore(max_size=10, db_path=str(unwritable / "s.db"))
        finally:
            unwritable.chmod(0o700)
        try:
            assert store._db_path is None
            _assert_no_owner_authority(store)
            _assert_generic_traffic_still_works(store)
        finally:
            store.close()

    def test_a_corrupt_store_file_is_refused_outright(self, tmp_path):
        """No store is built at all, so no authority can be handed out."""
        corrupt = tmp_path / "corrupt.db"
        corrupt.write_bytes(b"this is definitely not a sqlite database" * 64)

        with pytest.raises(sqlite3.DatabaseError):
            ResponseStore(max_size=10, db_path=str(corrupt))

    def test_owner_authority_stops_when_the_durable_file_disappears(self, tmp_path):
        """SQLite keeps serving an unlinked inode; that is not durability."""
        db_path = tmp_path / "response-store.db"
        store = ResponseStore(max_size=10, db_path=str(db_path))
        try:
            store.put(_AUTHORITY_RESPONSE, {
                "response": {"id": _AUTHORITY_RESPONSE, "created_at": 10},
                "conversation_history": [
                    {"role": "user", "content": "Build the approved milestone."},
                    {"role": "assistant",
                     "content": json.dumps(_owner_new_proposal())},
                ],
            })
            assert store.set_conversation(
                _AUTHORITY_CONVERSATION, _AUTHORITY_RESPONSE, owner_proposal=True,
            ) is True
            assert store.owner_history_snapshot(
                _AUTHORITY_CONVERSATION
            )["latest_response_id"] == _AUTHORITY_RESPONSE

            db_path.unlink()

            _assert_no_owner_authority(store)
            # ...while the non-authoritative cache keeps serving. SQLite refuses
            # to write a file it has seen disappear, so this only holds because
            # the store demoted itself to memory when it lost durability.
            _assert_generic_traffic_still_works(store)
            assert store._db_path is None
        finally:
            store.close()

    def test_a_vanished_store_never_regains_authority_from_a_new_file(
        self, tmp_path,
    ):
        """A path that already lost this store's data is not authority again."""
        db_path = tmp_path / "response-store.db"
        store = ResponseStore(max_size=10, db_path=str(db_path))
        try:
            db_path.unlink()
            with pytest.raises(OwnerAuthorityUnavailable):
                store.owner_history_snapshot(_AUTHORITY_CONVERSATION)

            # A new file at the same path is a different inode this connection
            # was never attached to, so nothing it holds became durable.
            db_path.write_bytes(b"")
            _assert_no_owner_authority(store)
            _assert_generic_traffic_still_works(store)
        finally:
            store.close()

    def test_a_broken_connection_never_fabricates_empty_owner_history(
        self, tmp_path
    ):
        """Unreadable is not the same answer as "this owner has no history"."""
        db_path = tmp_path / "response-store.db"
        store = ResponseStore(max_size=10, db_path=str(db_path))
        store._conn.close()

        with pytest.raises(sqlite3.Error):
            store.owner_history_snapshot(_AUTHORITY_CONVERSATION)
        with pytest.raises(sqlite3.Error):
            store.claim_owner_proposal(
                "default", _AUTHORITY_CONVERSATION,
                _AUTHORITY_RESPONSE, _AUTHORITY_CLAIM,
            )

    def test_owner_authority_persists_and_replays_across_a_restart(self, tmp_path):
        db_path = tmp_path / "response-store.db"
        store = ResponseStore(max_size=10, db_path=str(db_path))
        store.put(_AUTHORITY_RESPONSE, {
            "response": {"id": _AUTHORITY_RESPONSE, "created_at": 11},
            "conversation_history": [
                {"role": "user", "content": "Build the approved milestone."},
                {"role": "assistant", "content": json.dumps(_owner_new_proposal())},
            ],
        })
        assert store.set_conversation(
            _AUTHORITY_CONVERSATION, _AUTHORITY_RESPONSE, owner_proposal=True,
        ) is True
        assert store.claim_owner_proposal(
            "default", _AUTHORITY_CONVERSATION, _AUTHORITY_RESPONSE, _AUTHORITY_CLAIM,
        ) is True
        assert store.attach_owner_run(
            "default", _AUTHORITY_CONVERSATION, _AUTHORITY_RESPONSE,
            _AUTHORITY_CLAIM, _AUTHORITY_RUN,
        ) is True
        store.close()

        restarted = ResponseStore(max_size=10, db_path=str(db_path))
        try:
            # The spent approval survives the restart: replaying the same claim
            # is recognized, and a different claimer is still refused.
            assert restarted.owner_run_is_attached(
                "default", _AUTHORITY_CONVERSATION, _AUTHORITY_RESPONSE,
                _AUTHORITY_CLAIM, _AUTHORITY_RUN,
            ) is True
            assert restarted.claim_owner_proposal(
                "default", _AUTHORITY_CONVERSATION,
                _AUTHORITY_RESPONSE, _AUTHORITY_CLAIM,
            ) is True
            assert restarted.claim_owner_proposal(
                "default", _AUTHORITY_CONVERSATION,
                _AUTHORITY_RESPONSE, "claim_" + "9" * 32,
            ) is False
            assert restarted.owner_history_snapshot(
                _AUTHORITY_CONVERSATION
            )["proposal_claimed"] is True
        finally:
            restarted.close()

    @pytest.mark.asyncio
    async def test_unavailable_owner_storage_answers_one_stable_503(self):
        """The refusal reaches the caller as a retryable 503, and starts nothing."""
        adapter = APIServerAdapter.__new__(APIServerAdapter)
        middleware = adapter._make_owner_authority_middleware()
        reached = []

        async def handler(_request):
            reached.append("ran")
            raise OwnerAuthorityUnavailable("the owner workspace store is unavailable")

        request = types.SimpleNamespace(method="POST", path="/v1/owner/runs")
        response = await middleware(request, handler)

        assert response.status == 503
        body = json.loads(response.body.decode("utf-8"))
        assert body["error"]["code"] == "owner_workspace_unavailable"
        assert body["error"]["type"] == "server_error"
        # Nothing about the storage failure leaks to the caller.
        assert "sqlite" not in response.body.decode("utf-8").lower()
        # The handler raised before doing any work; the middleware added none.
        assert reached == ["ran"]


# ---------------------------------------------------------------------------
# F12 — a background owner response always reaches a terminal state
# ---------------------------------------------------------------------------


# Small enough that the inactivity watchdog decides within a few polls. These
# tests never wait on the real 1800s default.
_F12_TIMEOUT = "0.2"
_F12_REQUEST = "Prepare a private 60-minute workshop"
_F12_RESULT = {
    "final_response": "The workshop plan is ready.",
    "messages": [],
    "api_calls": 1,
}
_F12_USAGE = {"input_tokens": 2, "output_tokens": 3, "total_tokens": 5}


class _F12Agent:
    """An agent whose turn does not return until the test lets it.

    ``idle_seconds`` is exactly what the gateway's inactivity watchdog reads:
    a large value is a genuinely wedged turn, ``0.0`` is a turn that is slow
    but demonstrably still working and must therefore never be killed.  The
    cooperative interrupt is recorded but deliberately does NOT free the
    worker — that is the wedge that used to park the awaiting coroutine
    forever, and the timeout has to survive it.
    """

    session_prompt_tokens = 2
    session_completion_tokens = 3
    session_total_tokens = 5

    def __init__(
        self,
        release,
        *,
        idle_seconds: float = 9_999.0,
        hold_seconds: float = 30.0,
        final_response: str = "A late answer nobody is waiting for.",
    ):
        self._release = release
        self._idle_seconds = idle_seconds
        self._hold_seconds = hold_seconds
        self._final_response = final_response
        self.entered = threading.Event()
        self.returned = threading.Event()
        self.interrupts: list = []

    def get_activity_summary(self):
        return {
            "seconds_since_activity": self._idle_seconds,
            "last_activity_desc": "waiting for the test",
            "api_call_count": 1,
            "max_iterations": 10,
            "current_tool": None,
        }

    def interrupt(self, message=None):
        self.interrupts.append(message)

    def run_conversation(self, **_kwargs):
        self.entered.set()
        try:
            self._release.wait(timeout=self._hold_seconds)
            return {
                "final_response": self._final_response,
                "messages": [],
                "api_calls": 1,
            }
        finally:
            self.returned.set()


def _f12_post(cli, conversation, *, key=None, message=_F12_REQUEST):
    """Accept one background owner turn through the public contract."""
    return cli.post(
        "/v1/responses",
        json={
            "model": "hermes-agent",
            "input": message,
            "conversation": conversation,
            "background": True,
            "store": True,
            "expected_previous_response_id": None,
        },
        headers={"Idempotency-Key": key or f"response-{uuid.uuid4().hex}"},
    )


async def _f12_terminal(cli, response_id, *, tries=400, delay=0.01):
    """Poll one background response until it stops being non-terminal."""
    body = None
    for _ in range(tries):
        polled = await cli.get(f"/v1/responses/{response_id}")
        assert polled.status == 200
        body = await polled.json()
        if body["status"] not in {"queued", "in_progress"}:
            return body
        await asyncio.sleep(delay)
    pytest.fail(
        f"{response_id} never became terminal (still "
        f"{(body or {}).get('status')!r})"
    )


async def _f12_wait_until(predicate, *, tries=400, delay=0.01):
    """Settle on an out-of-band condition without sleeping a fixed budget."""
    for _ in range(tries):
        if predicate():
            return True
        await asyncio.sleep(delay)
    return False


async def _f12_history(cli, conversation):
    snapshot = await cli.get(f"/v1/responses/conversations/{conversation}")
    assert snapshot.status == 200
    return await snapshot.json()


def _f12_count_terminal_writes(adapter, recorder):
    """Wrap the one transaction every terminal owner ending must go through."""
    real = adapter._response_store.store_terminal_owner_response

    def _counting(**kwargs):
        recorder.append(kwargs)
        return real(**kwargs)

    return patch.object(
        adapter._response_store, "store_terminal_owner_response", _counting,
    )


class TestBackgroundOwnerResponseAlwaysTerminalizes:
    """F12: the browser is owed a terminal answer on every ending.

    Two ways an accepted owner turn used to end with no terminal state at all:
    an executor worker that never returns (the await was unbounded), and an
    exception from the finalizer or the publication that escaped the task
    (only the compute was guarded).  Both left the response ``in_progress``
    forever with its lease still heartbeated, so no sibling reclaimed it
    either.  Every ending below must produce exactly ONE terminal owner
    response and never a second turn.
    """

    @pytest.mark.asyncio
    async def test_a_wedged_turn_becomes_terminal_instead_of_parking_forever(
        self, adapter, monkeypatch,
    ):
        """The inactivity watchdog ends a worker that never returns."""
        monkeypatch.setenv("HERMES_AGENT_TIMEOUT", _F12_TIMEOUT)
        conversation = "raphael-owner-" + "c1" * 16
        release = threading.Event()
        agent = _F12Agent(release)
        writes: list = []
        app = _create_app(adapter)
        try:
            async with TestClient(TestServer(app)) as cli:
                with (
                    _f12_count_terminal_writes(adapter, writes),
                    patch.object(adapter, "_create_agent", return_value=agent),
                ):
                    accepted = await _f12_post(cli, conversation)
                    assert accepted.status == 200
                    queued = await accepted.json()
                    assert queued["status"] == "queued"
                    response_id = queued["id"]

                    terminal = await _f12_terminal(cli, response_id)

                    # Terminal, and reached without the worker ever returning.
                    assert terminal["status"] == "failed"
                    assert terminal["id"] == response_id
                    assert agent.entered.is_set()
                    assert not agent.returned.is_set()
                    # Cooperative: the turn was asked to stop, not killed.
                    assert agent.interrupts
                    # Exactly one terminal write, and it says this ending
                    # produced no turn.
                    assert len(writes) == 1
                    assert writes[0]["response_id"] == response_id
                    assert writes[0]["interrupted"] is True
                    assert writes[0]["release_job"] is True

                    # Readback returns the same terminal result.
                    readback = await cli.get(f"/v1/responses/{response_id}")
                    assert (await readback.json()) == terminal

                    # An abandoned turn publishes nothing, but leaves the
                    # owner the whole way back to their own request.
                    snapshot = await _f12_history(cli, conversation)
                    assert snapshot["data"] == []
                    assert snapshot["head_response_id"] is None
                    assert snapshot["pending"] is None
                    assert snapshot["recovery"] == {
                        "owner": _F12_REQUEST,
                        "response_id": response_id,
                    }

                    # Acknowledged, it is exactly ONE turn — never two.
                    acknowledged = await cli.post(
                        f"/v1/responses/conversations/{conversation}/recovery",
                        json={"response_id": response_id},
                    )
                    assert acknowledged.status == 200
                    sealed = await _f12_history(cli, conversation)
                    assert len(sealed["data"]) == 1
                    assert sealed["data"][0]["owner"] == _F12_REQUEST
                    assert sealed["head_response_id"] == response_id

                    release.set()
                    assert await _f12_wait_until(agent.returned.is_set)
        finally:
            release.set()

    @pytest.mark.asyncio
    async def test_an_active_turn_is_never_killed_and_its_real_result_wins(
        self, adapter, monkeypatch,
    ):
        """Inactivity, not elapsed time: a working turn outlives the window."""
        monkeypatch.setenv("HERMES_AGENT_TIMEOUT", _F12_TIMEOUT)
        conversation = "raphael-owner-" + "c9" * 16
        release = threading.Event()
        reply = json.dumps({
            "schema_version": 1,
            "kind": "question",
            "message": "Who is the workshop for?",
        })
        # Reports activity throughout, and runs for several times the
        # configured window before answering for real.
        agent = _F12Agent(
            release, idle_seconds=0.0, hold_seconds=0.4, final_response=reply,
        )
        app = _create_app(adapter)
        try:
            async with TestClient(TestServer(app)) as cli:
                with patch.object(adapter, "_create_agent", return_value=agent):
                    accepted = await _f12_post(cli, conversation)
                    response_id = (await accepted.json())["id"]
                    terminal = await _f12_terminal(cli, response_id)

                assert terminal["status"] == "completed"
                assert terminal["output"][0]["content"][0]["text"] == reply
                assert agent.interrupts == []
                # The real answer is the published turn.
                snapshot = await _f12_history(cli, conversation)
                assert snapshot["head_response_id"] == response_id
                assert len(snapshot["data"]) == 1
                assert snapshot["data"][0]["owner"] == _F12_REQUEST
                assert json.loads(snapshot["data"][0]["raphael"])["kind"] == (
                    "question"
                )
                assert snapshot["recovery"] is None
        finally:
            release.set()

    @pytest.mark.asyncio
    async def test_a_finalizer_that_raises_stores_one_terminal_owner_response(
        self, adapter,
    ):
        """The post-compute finalizer is inside the terminal guard."""
        conversation = "raphael-owner-" + "c2" * 16
        writes: list = []
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with (
                _f12_count_terminal_writes(adapter, writes),
                patch.object(
                    adapter, "_run_agent", new_callable=AsyncMock,
                    return_value=(_F12_RESULT, _F12_USAGE),
                ),
                patch.object(
                    adapter, "_finalize_response_result",
                    side_effect=RuntimeError("finalizer exploded"),
                ),
            ):
                accepted = await _f12_post(cli, conversation)
                response_id = (await accepted.json())["id"]
                terminal = await _f12_terminal(cli, response_id)

            assert terminal["status"] == "failed"
            assert len(writes) == 1
            assert writes[0]["response_id"] == response_id
            assert writes[0]["interrupted"] is True
            # Nothing was published, and the request is still recoverable.
            snapshot = await _f12_history(cli, conversation)
            assert snapshot["data"] == []
            assert snapshot["head_response_id"] is None
            assert snapshot["recovery"] == {
                "owner": _F12_REQUEST,
                "response_id": response_id,
            }

    @pytest.mark.asyncio
    async def test_a_publication_that_raises_stores_one_terminal_owner_response(
        self, adapter,
    ):
        """The publication is inside the terminal guard too, and reads back."""
        conversation = "raphael-owner-" + "c3" * 16
        writes: list = []
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with (
                _f12_count_terminal_writes(adapter, writes),
                patch.object(
                    adapter, "_run_agent", new_callable=AsyncMock,
                    return_value=(_F12_RESULT, _F12_USAGE),
                ),
                patch.object(
                    adapter._response_store, "publish_owner_turn",
                    side_effect=RuntimeError("publication exploded"),
                ),
            ):
                accepted = await _f12_post(cli, conversation)
                response_id = (await accepted.json())["id"]
                terminal = await _f12_terminal(cli, response_id)

            assert terminal["status"] == "failed"
            assert len(writes) == 1
            assert writes[0]["interrupted"] is True

            # Readback returns the terminal result...
            readback = await cli.get(f"/v1/responses/{response_id}")
            assert (await readback.json()) == terminal

            # ...and the owner history carries exactly ONE turn for it.
            assert (await _f12_history(cli, conversation))["data"] == []
            acknowledged = await cli.post(
                f"/v1/responses/conversations/{conversation}/recovery",
                json={"response_id": response_id},
            )
            assert acknowledged.status == 200
            sealed = await _f12_history(cli, conversation)
            assert len(sealed["data"]) == 1
            assert sealed["head_response_id"] == response_id

    @pytest.mark.asyncio
    async def test_an_exact_retry_replays_the_same_response_and_adds_no_turn(
        self, adapter,
    ):
        """The key already minted this response; a retry is told what happened."""
        conversation = "raphael-owner-" + "c4" * 16
        key = f"response-{uuid.uuid4().hex}"
        writes: list = []
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with (
                _f12_count_terminal_writes(adapter, writes),
                patch.object(
                    adapter, "_run_agent", new_callable=AsyncMock,
                    return_value=(_F12_RESULT, _F12_USAGE),
                ) as run,
                patch.object(
                    adapter, "_finalize_response_result",
                    side_effect=RuntimeError("finalizer exploded"),
                ),
            ):
                accepted = await _f12_post(cli, conversation, key=key)
                response_id = (await accepted.json())["id"]
                await _f12_terminal(cli, response_id)

                # Same key, same request: the same response comes back and no
                # second turn is planned.
                replayed = await _f12_post(cli, conversation, key=key)
                assert replayed.status == 200
                assert (await replayed.json())["id"] == response_id

                # And again with the in-memory cache gone, so the answer comes
                # from the durable record a restart would also read.
                with patch(
                    "gateway.platforms.api_server._idem_cache",
                    _IdempotencyCache(),
                ):
                    durable = await _f12_post(cli, conversation, key=key)
                assert durable.status == 200
                durable_body = await durable.json()
                assert durable_body["id"] == response_id
                assert durable_body["status"] == "failed"

                assert run.await_count == 1

            assert len(writes) == 1
            snapshot = await _f12_history(cli, conversation)
            assert snapshot["data"] == []
            assert snapshot["recovery"] == {
                "owner": _F12_REQUEST,
                "response_id": response_id,
            }

    @pytest.mark.asyncio
    async def test_a_worker_that_finishes_after_the_timeout_publishes_nothing(
        self, adapter, monkeypatch,
    ):
        """Fenced: the abandoned turn is terminal, and stays the only one."""
        monkeypatch.setenv("HERMES_AGENT_TIMEOUT", _F12_TIMEOUT)
        conversation = "raphael-owner-" + "c5" * 16
        release = threading.Event()
        agent = _F12Agent(release)
        writes: list = []
        publications: list = []
        real_publish = adapter._response_store.publish_owner_turn

        def _recording_publish(**kwargs):
            publications.append(kwargs["response_id"])
            return real_publish(**kwargs)

        app = _create_app(adapter)
        try:
            async with TestClient(TestServer(app)) as cli:
                with (
                    _f12_count_terminal_writes(adapter, writes),
                    patch.object(
                        adapter._response_store, "publish_owner_turn",
                        _recording_publish,
                    ),
                    patch.object(adapter, "_create_agent", return_value=agent),
                ):
                    accepted = await _f12_post(cli, conversation)
                    response_id = (await accepted.json())["id"]
                    terminal = await _f12_terminal(cli, response_id)
                    assert terminal["status"] == "failed"

                    # The worker only now produces its answer.
                    release.set()
                    assert await _f12_wait_until(agent.returned.is_set)
                    # Give any late publication every chance to land.
                    for _ in range(20):
                        await asyncio.sleep(0.01)

                    assert publications == []
                    assert len(writes) == 1
                    readback = await cli.get(f"/v1/responses/{response_id}")
                    assert (await readback.json()) == terminal
                    snapshot = await _f12_history(cli, conversation)
                    assert snapshot["data"] == []
                    assert snapshot["head_response_id"] is None
        finally:
            release.set()

    @pytest.mark.asyncio
    async def test_a_cancelled_background_turn_stores_its_terminal_incomplete(
        self, adapter,
    ):
        """The clean cancellation path still records one terminal ending."""
        conversation = "raphael-owner-" + "c6" * 16
        parked = asyncio.Event()
        running: dict = {}
        writes: list = []

        async def _park(**_kwargs):
            running["task"] = asyncio.current_task()
            parked.set()
            await asyncio.Event().wait()

        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with (
                _f12_count_terminal_writes(adapter, writes),
                patch.object(adapter, "_run_agent", side_effect=_park),
            ):
                accepted = await _f12_post(cli, conversation)
                response_id = (await accepted.json())["id"]
                await asyncio.wait_for(parked.wait(), timeout=2)
                running["task"].cancel()
                terminal = await _f12_terminal(cli, response_id)

            assert terminal["status"] == "incomplete"
            assert terminal["incomplete_details"] == {"reason": "cancelled"}
            assert len(writes) == 1
            assert writes[0]["interrupted"] is True
            assert (await _f12_history(cli, conversation))["recovery"] == {
                "owner": _F12_REQUEST,
                "response_id": response_id,
            }

    @pytest.mark.asyncio
    async def test_a_wedged_turn_leaves_no_residue_behind(
        self, adapter, monkeypatch,
    ):
        """Nothing outlives the abandoned turn — including its watchdog."""
        monkeypatch.setenv("HERMES_AGENT_TIMEOUT", _F12_TIMEOUT)
        from gateway.platforms.api_server import _TURN_PROCESS_EPOCHS

        conversation = "raphael-owner-" + "c7" * 16
        release = threading.Event()
        agent = _F12Agent(release)
        before_threads = {t.ident for t in threading.enumerate()}
        app = _create_app(adapter)
        try:
            async with TestClient(TestServer(app)) as cli:
                with patch.object(adapter, "_create_agent", return_value=agent):
                    accepted = await _f12_post(cli, conversation)
                    session_id = accepted.headers.get("X-Hermes-Session-Id")
                    response_id = (await accepted.json())["id"]
                    assert (await _f12_terminal(cli, response_id))["status"] == (
                        "failed"
                    )
                    release.set()
                    assert await _f12_wait_until(agent.returned.is_set)
                    assert await _f12_wait_until(
                        lambda: not adapter._background_tasks
                    )

                # The job nobody is driving any more, and both heartbeats
                # with it.
                assert adapter._owner_response_jobs == set()
                assert adapter._background_tasks == set()
                assert adapter._inflight_agent_runs == 0
                # The fence is gone: the request lives on as a recovery record
                # instead, which is what an owner can act on.
                assert (await _f12_history(cli, conversation))["pending"] is None
                # Agent and turn-process registries.
                assert adapter._shutdown_interruptible_agents == {}
                assert adapter._active_run_agents == {}
                assert session_id
                assert await _f12_wait_until(
                    lambda: session_id not in _TURN_PROCESS_EPOCHS
                )

            assert await _f12_wait_until(
                lambda: not [
                    t for t in threading.enumerate()
                    if t.ident not in before_threads
                    and t.name.startswith("api-turn-watchdog-")
                    and t.is_alive()
                ]
            )
        finally:
            release.set()
