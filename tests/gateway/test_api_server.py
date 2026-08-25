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
import json
import os
import sqlite3
import stat
import sys
import time
import types
import uuid
from unittest.mock import ANY, AsyncMock, MagicMock, patch

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.api_server import (
    APIServerAdapter,
    ResponseStore,
    _IdempotencyCache,
    _derive_chat_session_id,
    _hermes_version,
    _make_request_fingerprint,
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
    proposal.update(overrides)
    return proposal


def _owner_existing_proposal(**overrides):
    proposal = {
        "schema_version": 3,
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
            "existing_parent_refs": [],
            "new_parents": [],
        }],
    }
    proposal.update(overrides)
    return proposal


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


class TestResponseStore:
    def test_put_and_get(self):
        store = ResponseStore(max_size=10)
        store.put("resp_1", {"output": "hello"})
        assert store.get("resp_1") == {"output": "hello"}

    def test_get_missing_returns_none(self):
        store = ResponseStore(max_size=10)
        assert store.get("resp_missing") is None

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
            "latest_response_id": "resp_owner_history",
            "proposal_consumed": False,
            "proposal_claimed": False,
            "active_run_id": None,
            "conversation_closed": False,
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
            f"{group}-{'3' * 32}",
            "resp_unstructured_session",
            500,
            "This turn has no safe final reply.",
            "private intermediate reasoning",
        )

        index = store.owner_session_index("default", group)

        assert index == {
            "data": [
                {
                    "session_id": newer,
                    "updated_at": 300,
                    "preview": "Improve the mobile review flow.",
                    "visible_turn_count": 1,
                },
                {
                    "session_id": older,
                    "updated_at": 200,
                    "preview": "Add the owner summary.",
                    "visible_turn_count": 1,
                },
                {
                    "session_id": "legacy",
                    "updated_at": 100,
                    "preview": "Plan the first change.",
                    "visible_turn_count": 1,
                },
            ],
            "truncated": False,
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
            "latest_response_id": None,
            "proposal_consumed": False,
            "proposal_claimed": False,
            "active_run_id": None,
            "conversation_closed": False,
            "data": [],
        }


# ---------------------------------------------------------------------------
# _IdempotencyCache
        assert store.owner_history("raphael-owner-safe") == []
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
        read.assert_called_once_with(ANY, "shoe-shop", run_context=False)

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "query,expected_run_context",
        [
            ("", False),
            ("?capabilities=", False),
            ("?capabilities=something_else", False),
            ("?capabilities=run_task_context", True),
            ("?capabilities=something_else,run_task_context", True),
            ("?capabilities=run_task_context_extra", False),
            ("?capabilities=" + "x" * 300 + ",run_task_context", False),
            # A repeated key has no single negotiated answer, so BOTH orders
            # fail closed to the legacy shape rather than granting whichever
            # occurrence happens to be read first.
            ("?capabilities=run_task_context&capabilities=something_else", False),
            ("?capabilities=something_else&capabilities=run_task_context", False),
            ("?capabilities=run_task_context&capabilities=run_task_context", False),
            # The native board/workers capability is a different contract on a
            # different surface; naming it here grants nothing.
            ("?capabilities=owner_titles_v1", False),
        ],
    )
    async def test_run_context_is_served_only_to_a_reader_that_asks_for_it(
        self, adapter, query, expected_run_context,
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
            ANY, "shoe-shop", run_context=expected_run_context
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
            patch("hermes_cli.owner_workspace.list_owner_decisions", return_value=durable),
        ):
            async with TestClient(TestServer(app)) as cli:
                resp = await cli.get("/v1/owner-workspace/decisions")
                data = await resp.json()

        assert resp.status == 200
        assert data["object"] == "hermes.owner_workspace.decision_list"
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
                }],
                "truncated": False,
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
                    },
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
                ):
                    retry = await cli.post(
                        "/v1/responses",
                        json={
                            "model": "hermes-agent",
                            "input": "Try the plan again",
                            "conversation": conversation,
                            "background": True,
                            "store": True,
                        },
                    )
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
