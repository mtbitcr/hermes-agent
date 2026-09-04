"""Tests for the thin owner-workspace tool wrappers (tools/owner_workspace_tools.py).

Scope is deliberately narrow to the WRAPPER layer — the deep kernel
(idempotency, leases, crash recovery) is covered by
``tests/hermes_cli/test_owner_workspace.py``. What matters here:

  - Trusted identity is resolved via ``resolve_owner_context()``, never
    accepted from a tool-call argument.
  - Each handler passes through exactly the validated operation fields to
    the kernel, unchanged.
  - Schemas expose only the documented, narrow parameter surface — no
    author/profile/actor/session/path/scope field, and no broad execution
    capability.
  - The registered tool surface is exactly the six documented tools.
"""
from __future__ import annotations

import json

import pytest

import tools.owner_workspace_tools as owt
from hermes_cli.owner_workspace import OwnerContext, OwnerWorkspaceError
from tools.registry import registry
from toolsets import TOOLSETS, get_kernel_gated_toolsets, resolve_toolset

TOOL_NAMES = (
    "owner_workspace_bootstrap", "owner_task_graph_commit",
    "owner_project_plan_commit",
    "owner_task_move", "owner_task_comment", "owner_project_lifecycle",
)


# ---------------------------------------------------------------------------
# Tool surface
# ---------------------------------------------------------------------------


class TestToolSurface:
    def test_exactly_six_tools_registered_under_owner_workspace(self):
        assert registry.get_tool_names_for_toolset("owner_workspace") == sorted(TOOL_NAMES)

    def test_toolset_definition_lists_exactly_these_tools(self):
        assert set(TOOLSETS["owner_workspace"]["tools"]) == set(TOOL_NAMES)

    def test_no_other_toolset_exposes_owner_tools(self):
        for name, ts in TOOLSETS.items():
            if name in {"owner_workspace", "owner_task_graph_commit", "owner_project_plan_commit", "owner_project_lifecycle"}:
                continue
            assert not set(ts.get("tools") or []) & set(TOOL_NAMES)
        assert TOOLSETS["owner_task_graph_commit"]["tools"] == [
            "owner_task_graph_commit"
        ]
        assert TOOLSETS["owner_project_plan_commit"]["tools"] == [
            "owner_project_plan_commit"
        ]
        assert TOOLSETS["owner_project_lifecycle"]["tools"] == [
            "owner_project_lifecycle"
        ]

    def test_owner_workspace_toolsets_are_kernel_gated(self):
        assert {
            "owner_workspace", "owner_task_graph_commit", "owner_project_plan_commit",
            "owner_project_lifecycle",
        } <= get_kernel_gated_toolsets()

    def test_resolve_toolset_returns_exactly_these_tools(self):
        assert set(resolve_toolset("owner_workspace", include_registry=False)) == set(TOOL_NAMES)

    def test_project_steward_is_one_separate_read_only_tool(self):
        assert registry.get_tool_names_for_toolset("project_steward") == [
            "project_steward_snapshot"
        ]
        assert TOOLSETS["project_steward"]["tools"] == [
            "project_steward_snapshot"
        ]
        assert resolve_toolset("project_steward", include_registry=False) == [
            "project_steward_snapshot"
        ]
        assert "project_steward" not in get_kernel_gated_toolsets()

    @pytest.mark.parametrize("name", TOOL_NAMES)
    def test_each_tool_is_registered_with_a_handler(self, name):
        entry = registry.get_entry(name)
        assert entry is not None
        assert entry.toolset == "owner_workspace"
        assert callable(entry.handler)


# ---------------------------------------------------------------------------
# Schema shape — no broad execution authority, no identity smuggling
# ---------------------------------------------------------------------------

_FORBIDDEN_PARAM_NAMES = {
    "actor", "profile", "session", "session_id", "session_key",
    "path", "scope", "command", "cmd", "script", "code", "shell",
}

_ALLOWED_PARAM_NAMES = {
    "idempotency_key", "name", "description",
    "task_id", "to_status", "expected_status", "expected_revision", "board",
    "body", "mode", "project_name", "project_description", "project_id",
    "request_title", "specification", "current_milestone",
    "owner_visible_result", "root_assignee", "tasks", "later_milestones",
    "trigger", "summary", "changes", "action",
}


class TestSchemas:
    @pytest.mark.parametrize("name", TOOL_NAMES)
    def test_schema_name_matches_registration(self, name):
        entry = registry.get_entry(name)
        assert entry.schema["name"] == name

    @pytest.mark.parametrize("name", TOOL_NAMES)
    def test_schema_has_no_forbidden_identity_or_execution_params(self, name):
        entry = registry.get_entry(name)
        props = set(entry.schema["parameters"]["properties"])
        assert not props & _FORBIDDEN_PARAM_NAMES
        assert props <= _ALLOWED_PARAM_NAMES

    def test_bootstrap_requires_idempotency_key_and_name(self):
        entry = registry.get_entry("owner_workspace_bootstrap")
        required = set(entry.schema["parameters"]["required"])
        assert required == {"idempotency_key", "name"}
        assert set(entry.schema["parameters"]["properties"]) == {
            "idempotency_key", "name", "description",
        }

    def test_task_graph_schema_is_closed_and_bounded(self):
        entry = registry.get_entry("owner_task_graph_commit")
        parameters = entry.schema["parameters"]
        assert parameters["additionalProperties"] is False
        assert set(parameters["required"]) == {
            "idempotency_key", "mode", "request_title", "specification",
            "current_milestone", "owner_visible_result", "root_assignee", "tasks",
        }
        tasks = parameters["properties"]["tasks"]
        assert tasks["minItems"] == 1
        assert tasks["maxItems"] == 12
        assert tasks["items"]["additionalProperties"] is False
        assert set(tasks["items"]["required"]) == {
            "title", "body", "assignee", "responsibility", "execution_tier",
            "parents",
        }
        assert tasks["items"]["properties"]["execution_tier"]["enum"] == [
            "routine", "deep",
        ]

    def test_every_created_task_schema_requires_an_admitted_execution_tier(self):
        """No closed planner schema may leave the task class optional."""
        changes = registry.get_entry(
            "owner_project_plan_commit"
        ).schema["parameters"]["properties"]["changes"]["items"]["oneOf"]
        add, split, merge = changes[0], changes[1], changes[2]
        replace = next(
            change for change in changes
            if change["properties"]["action"] == {"const": "replace"}
        )

        specs = [
            add,
            *replace["properties"]["replacement"]["oneOf"],
            split["properties"]["replacements"]["items"],
            merge["properties"]["replacement"],
        ]
        for spec in specs:
            assert "execution_tier" in spec["required"]
            assert spec["properties"]["execution_tier"]["enum"] == ["routine", "deep"]
            # The planner classifies risk; it never names a runtime route.
            assert not {"model", "provider", "reasoning_effort"} & set(
                spec["properties"]
            )

    def test_every_created_task_schema_states_the_ownership_scope_rule(self):
        """The planner must hold the scope rule before it asks for approval.

        Tool schemas arrive with the tool list, so the ``owned_paths``
        description is what puts the rule in front of the main Raphael planner
        while it is still drafting — not after an owner has already approved a
        task whose NULL scope the trusted Server 2 dispatch would then refuse
        to provision a coding sandbox for. One shared definition means the
        rule cannot drift between the two planner tools.
        """
        graph_task = registry.get_entry(
            "owner_task_graph_commit"
        ).schema["parameters"]["properties"]["tasks"]["items"]
        changes = registry.get_entry(
            "owner_project_plan_commit"
        ).schema["parameters"]["properties"]["changes"]["items"]["oneOf"]
        add, split, merge = changes[0], changes[1], changes[2]
        replace = next(
            change for change in changes
            if change["properties"]["action"] == {"const": "replace"}
        )

        assert graph_task["properties"]["owned_paths"] is owt._OWNED_PATHS
        # New-Project task graphs remain legacy-compatible because that flow
        # has no repository folder yet.
        assert "owned_paths" not in graph_task["required"]

        replacement = replace["properties"]["replacement"]
        assert set(replacement) == {"oneOf"}
        preserve, rewrite, legacy = replacement["oneOf"]
        assert preserve["properties"]["body_mode"] == {"const": "preserve"}
        assert "body" not in preserve["properties"]
        assert rewrite["properties"]["body_mode"] == {"const": "rewrite"}
        assert "body" in rewrite["required"]

        legacy_specs = [
            graph_task,
            add,
            legacy,
            split["properties"]["replacements"]["items"],
            merge["properties"]["replacement"],
        ]
        for spec in legacy_specs:
            assert spec["properties"]["owned_paths"] is owt._OWNED_PATHS
            # Expand compatibility: an in-flight pre-v5 proposal still runs.
            assert "owned_paths" not in spec["required"]

        for spec in (preserve, rewrite):
            assert spec["properties"]["owned_paths"] is owt._OWNED_PATHS
            assert "owned_paths" in spec["required"]

        text = owt._OWNED_PATHS["description"]
        assert "repository-executing or coding task MUST state its scope" in text
        assert "[] only for genuinely read-only repository work" in text
        assert (
            "['.'] only when whole-repository mutation is actually approved" in text
        )
        assert "cannot provision a coding sandbox" in text

    def test_task_move_requires_full_cas_precondition(self):
        entry = registry.get_entry("owner_task_move")
        required = set(entry.schema["parameters"]["required"])
        assert required == {
            "idempotency_key", "project_id", "task_id", "to_status",
            "expected_status", "expected_revision",
        }
        assert set(entry.schema["parameters"]["properties"]) == {
            "idempotency_key", "task_id", "to_status",
            "expected_status", "expected_revision", "project_id",
        }

    def test_project_plan_requires_bounded_exact_changes(self):
        entry = registry.get_entry("owner_project_plan_commit")
        params = entry.schema["parameters"]
        assert params["additionalProperties"] is False
        # No caller-named anchor: the Project's hidden control row is resolved
        # inside the kernel from its committed receipt.
        assert set(params["required"]) == {
            "idempotency_key", "project_id", "trigger",
            "request_title", "summary", "specification", "current_milestone",
            "owner_visible_result", "later_milestones", "changes",
        }
        assert "anchor_task_id" not in params["properties"]
        assert params["properties"]["changes"]["maxItems"] == 12
        assert len(params["properties"]["changes"]["items"]["oneOf"]) == 6

        move_schema = params["properties"]["changes"]["items"]["oneOf"][3]
        assert move_schema["properties"]["action"] == {"const": "move"}
        assert move_schema["properties"]["to_status"]["enum"] == ["ready"]

    def test_project_plan_exposes_a_closed_one_to_one_replace_change(self):
        changes = registry.get_entry(
            "owner_project_plan_commit"
        ).schema["parameters"]["properties"]["changes"]["items"]["oneOf"]
        replace = next(
            change for change in changes
            if change["properties"]["action"] == {"const": "replace"}
        )

        assert replace["additionalProperties"] is False
        assert set(replace["required"]) == {
            "action", "reason", "target", "replacement",
        }
        replacement = replace["properties"]["replacement"]
        assert set(replacement) == {"oneOf"}
        assert all(
            "execution_tier" in variant["required"]
            for variant in replacement["oneOf"]
        )

    def test_task_comment_requires_task_id_and_body(self):
        entry = registry.get_entry("owner_task_comment")
        required = set(entry.schema["parameters"]["required"])
        assert required == {"idempotency_key", "project_id", "task_id", "body"}
        assert set(entry.schema["parameters"]["properties"]) == {
            "idempotency_key", "project_id", "task_id", "body",
        }


# ---------------------------------------------------------------------------
# Trusted context resolution — never from tool-call arguments
# ---------------------------------------------------------------------------


class _RecordingKernel:
    """Captures the exact kwargs a wrapper handler passed through."""

    def __init__(self, return_value=None):
        self.calls = []
        self.return_value = return_value if return_value is not None else {"ok": True}

    def __call__(self, ctx, **kwargs):
        self.calls.append((ctx, kwargs))
        return self.return_value


@pytest.fixture
def trusted_ctx():
    return OwnerContext(actor="trusted-actor", profile="trusted-profile", session="trusted-session")


class TestRefusalRendering:
    def test_a_refusal_carries_its_kernel_code(self, monkeypatch, trusted_ctx):
        """The run layer and the owner surface name the rule that refused
        without parsing the message text."""
        monkeypatch.setattr(owt, "resolve_owner_context", lambda: trusted_ctx)

        def refuse(ctx, **kwargs):
            raise OwnerWorkspaceError(
                "ownership_scope_unavailable",
                "changes declares repository ownership, but this Project has no "
                "primary repository folder to scope a worktree in",
            )

        monkeypatch.setattr(owt._kernel, "commit_project_plan", refuse)
        payload = json.loads(owt._handle_project_plan({
            "idempotency_key": "plan-refused",
            "project_id": "p1",
            "request_title": "Reassign the edit",
            "summary": "Reassign one task.",
            "changes": [],
        }))
        assert payload["code"] == "ownership_scope_unavailable"
        assert payload["error"].startswith(
            "owner_project_plan_commit: changes declares repository ownership"
        )


class TestTrustedContextResolution:
    def test_bootstrap_uses_resolved_context_not_args(self, monkeypatch, trusted_ctx):
        monkeypatch.setattr(owt, "resolve_owner_context", lambda: trusted_ctx)
        kernel = _RecordingKernel()
        monkeypatch.setattr(owt._kernel, "bootstrap", kernel)

        owt._handle_bootstrap({
            "idempotency_key": "k1", "name": "n1",
            # Attacker-supplied identity fields the schema never declares —
            # must be silently ignored, never reach the kernel.
            "actor": "attacker", "profile": "attacker-profile", "session": "attacker-session",
        })

        assert len(kernel.calls) == 1
        ctx, kwargs = kernel.calls[0]
        assert ctx is trusted_ctx
        assert ctx.actor == "trusted-actor"
        assert "actor" not in kwargs
        assert "profile" not in kwargs
        assert "session" not in kwargs

    def test_task_graph_uses_resolved_context_not_args(self, monkeypatch, trusted_ctx):
        monkeypatch.setattr(owt, "resolve_owner_context", lambda: trusted_ctx)
        kernel = _RecordingKernel()
        monkeypatch.setattr(owt._kernel, "commit_task_graph", kernel)

        owt._handle_task_graph({
            "idempotency_key": "g1",
            "mode": "existing",
            "project_id": "p1",
            "request_title": "Improve checkout",
            "specification": "Make checkout simpler.",
            "current_milestone": "Ship the first improvement.",
            "owner_visible_result": "The owner can verify checkout.",
            "root_assignee": "coordinator",
            "tasks": [],
            "actor": "attacker",
            "profile": "attacker-profile",
            "session": "attacker-session",
        })

        ctx, kwargs = kernel.calls[0]
        assert ctx is trusted_ctx
        assert "actor" not in kwargs
        assert "profile" not in kwargs
        assert "session" not in kwargs

    def test_task_move_uses_resolved_context_not_args(self, monkeypatch, trusted_ctx):
        monkeypatch.setattr(owt, "resolve_owner_context", lambda: trusted_ctx)
        kernel = _RecordingKernel()
        monkeypatch.setattr(owt._kernel, "move_task", kernel)

        owt._handle_task_move({
            "idempotency_key": "k2", "task_id": "t1", "to_status": "done",
            "expected_status": "review", "expected_revision": 3,
            "actor": "attacker",
        })

        ctx, kwargs = kernel.calls[0]
        assert ctx is trusted_ctx
        assert "actor" not in kwargs

    def test_project_plan_uses_resolved_context_not_args(self, monkeypatch, trusted_ctx):
        monkeypatch.setattr(owt, "resolve_owner_context", lambda: trusted_ctx)
        kernel = _RecordingKernel()
        monkeypatch.setattr(owt._kernel, "commit_project_plan", kernel)
        owt._handle_project_plan({
            "idempotency_key": "p1", "project_id": "project",
            "trigger": "owner_request", "request_title": "Adapt", "summary": "Summary",
            "specification": "Spec", "current_milestone": "Now",
            "owner_visible_result": "Visible", "later_milestones": [], "changes": [],
            "actor": "attacker",
        })
        ctx, kwargs = kernel.calls[0]
        assert ctx is trusted_ctx
        assert "actor" not in kwargs

    def test_task_comment_uses_resolved_context_not_args(self, monkeypatch, trusted_ctx):
        monkeypatch.setattr(owt, "resolve_owner_context", lambda: trusted_ctx)
        kernel = _RecordingKernel()
        monkeypatch.setattr(owt._kernel, "comment_task", kernel)

        owt._handle_task_comment({
            "idempotency_key": "k3", "task_id": "t1", "body": "hi",
            "profile": "attacker-profile",
        })

        ctx, kwargs = kernel.calls[0]
        assert ctx is trusted_ctx
        assert "profile" not in kwargs

    def test_context_is_resolved_fresh_each_call_not_cached_from_first_call(self, monkeypatch):
        ctxs = [
            OwnerContext(actor="a1", profile="p1", session="s1"),
            OwnerContext(actor="a2", profile="p2", session="s2"),
        ]
        calls = iter(ctxs)
        monkeypatch.setattr(owt, "resolve_owner_context", lambda: next(calls))
        kernel = _RecordingKernel()
        monkeypatch.setattr(owt._kernel, "bootstrap", kernel)

        owt._handle_bootstrap({"idempotency_key": "k1", "name": "n1"})
        owt._handle_bootstrap({"idempotency_key": "k2", "name": "n2"})

        assert kernel.calls[0][0].actor == "a1"
        assert kernel.calls[1][0].actor == "a2"


# ---------------------------------------------------------------------------
# Wrapper delegation preserves the validated operation fields exactly
# ---------------------------------------------------------------------------


class TestFieldDelegation:
    def test_project_steward_passes_only_bounded_read_fields(self, monkeypatch):
        calls = []

        def kernel(**kwargs):
            calls.append(kwargs)
            return {"schema_version": 1}

        monkeypatch.setattr(owt._kernel, "project_steward_snapshot", kernel)

        out = owt._handle_project_steward_snapshot({
            "project_id": "p1", "lookback_days": 7, "path": "/secret",
        })

        assert json.loads(out) == {"schema_version": 1}
        assert calls == [{"project_id": "p1", "lookback_days": 7}]

    def test_bootstrap_passes_through_exact_fields(self, monkeypatch, trusted_ctx):
        monkeypatch.setattr(owt, "resolve_owner_context", lambda: trusted_ctx)
        kernel = _RecordingKernel()
        monkeypatch.setattr(owt._kernel, "bootstrap", kernel)

        owt._handle_bootstrap({
            "idempotency_key": "k1", "name": "My Project", "description": "desc",
        })

        _, kwargs = kernel.calls[0]
        assert kwargs == {
            "idempotency_key": "k1", "name": "My Project", "description": "desc",
        }

    def test_bootstrap_missing_description_passes_none(self, monkeypatch, trusted_ctx):
        monkeypatch.setattr(owt, "resolve_owner_context", lambda: trusted_ctx)
        kernel = _RecordingKernel()
        monkeypatch.setattr(owt._kernel, "bootstrap", kernel)

        owt._handle_bootstrap({"idempotency_key": "k1", "name": "My Project"})

        _, kwargs = kernel.calls[0]
        assert kwargs["description"] is None

    def test_task_graph_passes_through_exact_fields(self, monkeypatch, trusted_ctx):
        monkeypatch.setattr(owt, "resolve_owner_context", lambda: trusted_ctx)
        kernel = _RecordingKernel()
        monkeypatch.setattr(owt._kernel, "commit_task_graph", kernel)
        args = {
            "idempotency_key": "g1",
            "mode": "new",
            "project_name": "Shoe shop",
            "project_description": "Owner project",
            "project_id": None,
            "request_title": "Launch the shop",
            "specification": "Build the first useful version.",
            "current_milestone": "Launch",
            "owner_visible_result": "A working shop",
            "root_assignee": "coordinator",
            "tasks": [{
                "title": "Build",
                "body": "Build it.",
                "assignee": "coder",
                "parents": [],
            }],
            "later_milestones": ["Learn from customers"],
        }

        owt._handle_task_graph(args)

        ctx, kwargs = kernel.calls[0]
        assert ctx is trusted_ctx
        assert kwargs == args

    def test_task_move_passes_through_exact_fields(self, monkeypatch, trusted_ctx):
        monkeypatch.setattr(owt, "resolve_owner_context", lambda: trusted_ctx)
        kernel = _RecordingKernel()
        monkeypatch.setattr(owt._kernel, "move_task", kernel)

        owt._handle_task_move({
            "idempotency_key": "k2", "project_id": "p1", "task_id": "t1", "to_status": "done",
            "expected_status": "review", "expected_revision": 3,
        })

        _, kwargs = kernel.calls[0]
        assert kwargs == {
            "idempotency_key": "k2", "project_id": "p1", "task_id": "t1", "to_status": "done",
            "expected_status": "review", "expected_revision": 3,
        }

    def test_project_plan_passes_through_exact_fields(self, monkeypatch, trusted_ctx):
        monkeypatch.setattr(owt, "resolve_owner_context", lambda: trusted_ctx)
        kernel = _RecordingKernel()
        monkeypatch.setattr(owt._kernel, "commit_project_plan", kernel)
        args = {
            "idempotency_key": "p1", "project_id": "project",
            "trigger": "owner_request", "request_title": "Adapt", "summary": "Summary",
            "specification": "Spec", "current_milestone": "Now",
            "owner_visible_result": "Visible", "later_milestones": [],
            "changes": [{"action": "postpone"}],
        }
        owt._handle_project_plan(args)
        ctx, kwargs = kernel.calls[0]
        assert ctx is trusted_ctx
        assert kwargs == args

    def test_project_lifecycle_passes_through_exact_fields(
        self, monkeypatch, trusted_ctx,
    ):
        monkeypatch.setattr(owt, "resolve_owner_context", lambda: trusted_ctx)
        kernel = _RecordingKernel()
        monkeypatch.setattr(owt._kernel, "set_project_archived", kernel)

        owt._handle_project_lifecycle({
            "idempotency_key": "lifecycle-1",
            "project_id": "p1",
            "expected_revision": 4,
            "action": "archive",
        })

        ctx, kwargs = kernel.calls[0]
        assert ctx is trusted_ctx
        assert kwargs == {
            "idempotency_key": "lifecycle-1",
            "project_id": "p1",
            "expected_revision": 4,
            "action": "archive",
        }

    def test_task_comment_passes_through_exact_fields(self, monkeypatch, trusted_ctx):
        monkeypatch.setattr(owt, "resolve_owner_context", lambda: trusted_ctx)
        kernel = _RecordingKernel()
        monkeypatch.setattr(owt._kernel, "comment_task", kernel)

        owt._handle_task_comment({
            "idempotency_key": "k3", "project_id": "p1", "task_id": "t1", "body": "hi there",
        })

        _, kwargs = kernel.calls[0]
        assert kwargs == {
            "idempotency_key": "k3", "project_id": "p1", "task_id": "t1", "body": "hi there",
        }

    def test_successful_result_is_returned_verbatim_as_json(self, monkeypatch, trusted_ctx):
        monkeypatch.setattr(owt, "resolve_owner_context", lambda: trusted_ctx)
        result = {"ok": True, "task_id": "t1", "status": "done", "revision": 7}
        monkeypatch.setattr(owt._kernel, "move_task", _RecordingKernel(return_value=result))

        out = owt._handle_task_move({
            "idempotency_key": "k2", "project_id": "p1", "task_id": "t1", "to_status": "done",
            "expected_status": "review", "expected_revision": 3,
        })

        assert json.loads(out) == result


# ---------------------------------------------------------------------------
# Error handling — surfaced without leaking internals
# ---------------------------------------------------------------------------


class TestErrorHandling:
    def test_project_steward_error_does_not_leak_internal_details(self, monkeypatch):
        def _raise(**kwargs):
            raise RuntimeError("sqlite file /secret/path/kanban.db is locked")

        monkeypatch.setattr(owt._kernel, "project_steward_snapshot", _raise)
        payload = json.loads(
            owt._handle_project_steward_snapshot({"project_id": "p1"})
        )
        assert "internal error" in payload["error"]
        assert "/secret/path" not in payload["error"]

    def test_bootstrap_owner_workspace_error_message_is_surfaced(self, monkeypatch, trusted_ctx):
        monkeypatch.setattr(owt, "resolve_owner_context", lambda: trusted_ctx)

        def _raise(ctx, **kwargs):
            raise OwnerWorkspaceError("invalid_argument", "name is required")

        monkeypatch.setattr(owt._kernel, "bootstrap", _raise)

        out = owt._handle_bootstrap({"idempotency_key": "k1", "name": "n"})
        payload = json.loads(out)
        assert "error" in payload
        assert "name is required" in payload["error"]
        assert "owner_workspace_bootstrap" in payload["error"]

    def test_task_move_value_error_message_is_surfaced(self, monkeypatch, trusted_ctx):
        monkeypatch.setattr(owt, "resolve_owner_context", lambda: trusted_ctx)

        def _raise(ctx, **kwargs):
            raise ValueError("expected_revision must be an integer")

        monkeypatch.setattr(owt._kernel, "move_task", _raise)

        out = owt._handle_task_move({
            "idempotency_key": "k2", "task_id": "t1", "to_status": "done",
            "expected_status": "review", "expected_revision": "not-an-int",
        })
        payload = json.loads(out)
        assert "expected_revision must be an integer" in payload["error"]

    def test_unexpected_exception_does_not_leak_internal_details(self, monkeypatch, trusted_ctx):
        monkeypatch.setattr(owt, "resolve_owner_context", lambda: trusted_ctx)

        def _raise(ctx, **kwargs):
            raise RuntimeError("sqlite file /secret/path/projects.db is locked")

        monkeypatch.setattr(owt._kernel, "comment_task", _raise)

        out = owt._handle_task_comment({"idempotency_key": "k3", "task_id": "t1", "body": "hi"})
        payload = json.loads(out)
        assert "internal error" in payload["error"]
        assert "/secret/path" not in payload["error"]
