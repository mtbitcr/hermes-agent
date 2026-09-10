"""The owner retry is not a tool a model can call.

``owner_task_retry`` persists the OWNER's own stated reason for trying stopped
work again. It is dispatched natively by the API server from an already
authenticated owner run — the run layer resolves the handler by name on
``tools.owner_workspace_tools`` — so there is no reason for it to exist as a
callable schema, and every reason for it not to: a schema is offered to the
model, and anything offered to the model can be called by the model with
arguments the model chose.

Absence is asserted through the two surfaces that actually decide what an
agent is handed — the registry entry and the resolved toolset — and across
EVERY toolset, so re-adding it anywhere is caught wherever it is added. The
native handler is asserted present in the same breath, because removing the
capability from the run layer is not the fix.
"""

from __future__ import annotations

import json

import pytest

import tools.owner_workspace_tools as owt
from hermes_cli.owner_workspace import OwnerContext
from tools.registry import registry
from toolsets import TOOLSETS, get_toolset_names, resolve_toolset


RETRY = "owner_task_retry"


class TestRetryIsNotModelFacing:
    def test_the_retry_has_no_schema_but_keeps_its_native_run_handler(
        self, monkeypatch,
    ):
        """Unreachable through the registry, still reachable by the run layer.

        The second half is exactly how ``gateway/platforms/api_server.py``
        reaches it: by name on the module, with the frozen payload the run
        authority was minted from — never through the registry.
        """
        assert registry.get_entry(RETRY) is None
        assert registry.get_schema(RETRY) is None
        assert RETRY not in registry.get_tool_to_toolset_map()

        handler = getattr(owt, "_handle_task_retry", None)
        assert callable(handler)

        seen = {}

        def _kernel(ctx, **kwargs):
            seen["ctx"] = ctx
            seen["kwargs"] = kwargs
            return {"ok": True, "task_id": kwargs["task_id"], "status": "ready"}

        monkeypatch.setattr(owt._kernel, "retry_task", _kernel)
        monkeypatch.setattr(
            owt, "resolve_owner_context",
            lambda: OwnerContext(actor="owner", profile="owner", session="s1"),
        )

        payload = {
            "idempotency_key": "retry-native-1",
            "project_id": "p_1",
            "task_id": "t_1",
            "reason": "The credentials are fixed now.",
        }
        result = json.loads(handler(dict(payload)))

        assert result["ok"] is True
        assert seen["kwargs"] == payload
        # Identity is the kernel's to resolve, never the caller's to state.
        assert seen["ctx"].actor == "owner"
        assert seen["ctx"].profile == "owner"

    def test_no_toolset_offers_the_retry_to_a_model(self):
        for name in get_toolset_names():
            assert RETRY not in resolve_toolset(name, include_registry=True), name
            assert RETRY not in resolve_toolset(name, include_registry=False), name
        for name, toolset in TOOLSETS.items():
            assert RETRY not in set(toolset.get("tools") or []), name

        # The retry is gone; the tools an owner agent needs are not.
        offered = set(resolve_toolset("owner_workspace", include_registry=True))
        assert {
            "owner_workspace_bootstrap",
            "owner_task_graph_commit",
            "owner_project_plan_commit",
            "owner_project_lifecycle",
            "owner_task_move",
            "owner_task_comment",
        } <= offered
