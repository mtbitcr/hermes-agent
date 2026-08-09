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
  - The registered tool surface is exactly the three documented tools.
"""
from __future__ import annotations

import json

import pytest

import tools.owner_workspace_tools as owt
from hermes_cli.owner_workspace import OwnerContext, OwnerWorkspaceError
from tools.registry import registry
from toolsets import TOOLSETS, get_kernel_gated_toolsets, resolve_toolset

TOOL_NAMES = ("owner_workspace_bootstrap", "owner_task_move", "owner_task_comment")


# ---------------------------------------------------------------------------
# Tool surface
# ---------------------------------------------------------------------------


class TestToolSurface:
    def test_exactly_three_tools_registered_under_owner_workspace(self):
        assert registry.get_tool_names_for_toolset("owner_workspace") == sorted(TOOL_NAMES)

    def test_toolset_definition_lists_exactly_these_tools(self):
        assert set(TOOLSETS["owner_workspace"]["tools"]) == set(TOOL_NAMES)

    def test_no_other_toolset_exposes_owner_tools(self):
        for name, ts in TOOLSETS.items():
            if name == "owner_workspace":
                continue
            assert not set(ts.get("tools") or []) & set(TOOL_NAMES)

    def test_owner_workspace_toolset_is_kernel_gated(self):
        assert "owner_workspace" in get_kernel_gated_toolsets()

    def test_resolve_toolset_returns_exactly_these_tools(self):
        assert set(resolve_toolset("owner_workspace", include_registry=False)) == set(TOOL_NAMES)

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
    "body",
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

    def test_task_move_requires_full_cas_precondition(self):
        entry = registry.get_entry("owner_task_move")
        required = set(entry.schema["parameters"]["required"])
        assert required == {
            "idempotency_key", "task_id", "to_status",
            "expected_status", "expected_revision",
        }
        assert set(entry.schema["parameters"]["properties"]) == {
            "idempotency_key", "task_id", "to_status",
            "expected_status", "expected_revision", "board",
        }

    def test_task_comment_requires_task_id_and_body(self):
        entry = registry.get_entry("owner_task_comment")
        required = set(entry.schema["parameters"]["required"])
        assert required == {"idempotency_key", "task_id", "body"}
        assert set(entry.schema["parameters"]["properties"]) == {
            "idempotency_key", "task_id", "body", "board",
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

    def test_task_move_passes_through_exact_fields(self, monkeypatch, trusted_ctx):
        monkeypatch.setattr(owt, "resolve_owner_context", lambda: trusted_ctx)
        kernel = _RecordingKernel()
        monkeypatch.setattr(owt._kernel, "move_task", kernel)

        owt._handle_task_move({
            "idempotency_key": "k2", "task_id": "t1", "to_status": "done",
            "expected_status": "review", "expected_revision": 3, "board": "b1",
        })

        _, kwargs = kernel.calls[0]
        assert kwargs == {
            "idempotency_key": "k2", "task_id": "t1", "to_status": "done",
            "expected_status": "review", "expected_revision": 3, "board": "b1",
        }

    def test_task_comment_passes_through_exact_fields(self, monkeypatch, trusted_ctx):
        monkeypatch.setattr(owt, "resolve_owner_context", lambda: trusted_ctx)
        kernel = _RecordingKernel()
        monkeypatch.setattr(owt._kernel, "comment_task", kernel)

        owt._handle_task_comment({
            "idempotency_key": "k3", "task_id": "t1", "body": "hi there", "board": "b1",
        })

        _, kwargs = kernel.calls[0]
        assert kwargs == {
            "idempotency_key": "k3", "task_id": "t1", "body": "hi there", "board": "b1",
        }

    def test_successful_result_is_returned_verbatim_as_json(self, monkeypatch, trusted_ctx):
        monkeypatch.setattr(owt, "resolve_owner_context", lambda: trusted_ctx)
        result = {"ok": True, "task_id": "t1", "status": "done", "revision": 7}
        monkeypatch.setattr(owt._kernel, "move_task", _RecordingKernel(return_value=result))

        out = owt._handle_task_move({
            "idempotency_key": "k2", "task_id": "t1", "to_status": "done",
            "expected_status": "review", "expected_revision": 3,
        })

        assert json.loads(out) == result


# ---------------------------------------------------------------------------
# Error handling — surfaced without leaking internals
# ---------------------------------------------------------------------------


class TestErrorHandling:
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
