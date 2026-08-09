"""Owner-workspace toolset — three thin model tools over the deep kernel in
``hermes_cli.owner_workspace``.

Default-off, API-server-only surface (see ``toolsets.py``'s ``owner_workspace``
toolset — absent from ``_HERMES_CORE_TOOLS`` and every composite — and
``gateway/platforms/api_server.py``'s ``_create_agent``, the ONLY place that
folds it into ``enabled_toolsets``, gated on
``gateway.api_server.owner_workspace.enabled`` for the resolved profile).

Every tool schema is deliberately narrow: no author/profile/actor/session/
path/scope field is accepted from the model. Identity is resolved from
trusted request context (``resolve_owner_context``) inside the kernel.
"""
from __future__ import annotations

import json
import logging

from hermes_cli.owner_workspace import OwnerWorkspaceError, resolve_owner_context
from hermes_cli import owner_workspace as _kernel
from tools.registry import registry, tool_error

logger = logging.getLogger(__name__)


def _ok(result: dict) -> str:
    return json.dumps(result, ensure_ascii=False)


def _handle_bootstrap(args: dict, **kw) -> str:
    try:
        ctx = resolve_owner_context()
        result = _kernel.bootstrap(
            ctx,
            idempotency_key=args.get("idempotency_key"),
            name=args.get("name"),
            description=args.get("description"),
        )
        return _ok(result)
    except OwnerWorkspaceError as e:
        return tool_error(f"owner_workspace_bootstrap: {e.message}")
    except Exception:
        logger.exception("owner_workspace_bootstrap failed")
        return tool_error("owner_workspace_bootstrap: internal error")


def _handle_task_move(args: dict, **kw) -> str:
    try:
        ctx = resolve_owner_context()
        result = _kernel.move_task(
            ctx,
            idempotency_key=args.get("idempotency_key"),
            task_id=args.get("task_id"),
            to_status=args.get("to_status"),
            expected_status=args.get("expected_status"),
            expected_revision=args.get("expected_revision"),
            board=args.get("board"),
        )
        return _ok(result)
    except OwnerWorkspaceError as e:
        return tool_error(f"owner_task_move: {e.message}")
    except ValueError as e:
        return tool_error(f"owner_task_move: {e}")
    except Exception:
        logger.exception("owner_task_move failed")
        return tool_error("owner_task_move: internal error")


def _handle_task_comment(args: dict, **kw) -> str:
    try:
        ctx = resolve_owner_context()
        result = _kernel.comment_task(
            ctx,
            idempotency_key=args.get("idempotency_key"),
            task_id=args.get("task_id"),
            body=args.get("body"),
            board=args.get("board"),
        )
        return _ok(result)
    except OwnerWorkspaceError as e:
        return tool_error(f"owner_task_comment: {e.message}")
    except ValueError as e:
        return tool_error(f"owner_task_comment: {e}")
    except Exception:
        logger.exception("owner_task_comment failed")
        return tool_error("owner_task_comment: internal error")


registry.register(
    name="owner_workspace_bootstrap",
    toolset="owner_workspace",
    schema={
        "name": "owner_workspace_bootstrap",
        "description": (
            "Create the owner's workspace: one Project, one Kanban board, and "
            "one initial task. Idempotent — replaying the same idempotency_key "
            "with the same arguments returns the original result; reusing it "
            "with different arguments fails. Requires a fresh human confirmation. "
            "On success, the result's status and revision are the new task's "
            "current status and event revision — pass them straight through as "
            "owner_task_move's expected_status/expected_revision to move it."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "idempotency_key": {
                    "type": "string",
                    "description": "Stable client-chosen key so a retried call is safe.",
                },
                "name": {
                    "type": "string",
                    "description": "Human name for the new Project/board/initial task.",
                },
                "description": {
                    "type": "string",
                    "description": "Optional description for the Project and initial task.",
                },
            },
            "required": ["idempotency_key", "name"],
        },
    },
    handler=lambda args, **kw: _handle_bootstrap(args, **kw),
)

registry.register(
    name="owner_task_move",
    toolset="owner_workspace",
    schema={
        "name": "owner_task_move",
        "description": (
            "Move an owner-workspace task to a new status via optimistic "
            "compare-and-swap. Requires the task's current status and event "
            "revision (from owner_workspace_bootstrap's result or a prior "
            "owner_task_move/owner_task_comment response) — a mismatch returns "
            "the current snapshot with zero changes instead of an error. Cannot "
            "claim or move a running task. Idempotent; requires a fresh human "
            "confirmation."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "idempotency_key": {
                    "type": "string",
                    "description": "Stable client-chosen key so a retried call is safe.",
                },
                "task_id": {"type": "string", "description": "The task to move."},
                "to_status": {
                    "type": "string",
                    "description": "Target status (e.g. todo, ready, blocked, review, done, archived).",
                },
                "expected_status": {
                    "type": "string",
                    "description": "The task's status you last observed — the CAS precondition.",
                },
                "expected_revision": {
                    "type": "integer",
                    "description": "The task's event revision you last observed — the CAS precondition.",
                },
                "board": {
                    "type": "string",
                    "description": "Optional board slug (defaults to the current board).",
                },
            },
            "required": ["idempotency_key", "task_id", "to_status", "expected_status", "expected_revision"],
        },
    },
    handler=lambda args, **kw: _handle_task_move(args, **kw),
)

registry.register(
    name="owner_task_comment",
    toolset="owner_workspace",
    schema={
        "name": "owner_task_comment",
        "description": (
            "Append a comment to an owner-workspace task. The comment author is "
            "always the trusted caller identity — it cannot be set by this call. "
            "Idempotent; requires a fresh human confirmation. On success, the "
            "result's status and revision are the task's current status and "
            "event revision (the comment itself advances the revision) — pass "
            "them straight through as owner_task_move's expected_status/"
            "expected_revision to move it."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "idempotency_key": {
                    "type": "string",
                    "description": "Stable client-chosen key so a retried call is safe.",
                },
                "task_id": {"type": "string", "description": "The task to comment on."},
                "body": {"type": "string", "description": "Comment text."},
                "board": {
                    "type": "string",
                    "description": "Optional board slug (defaults to the current board).",
                },
            },
            "required": ["idempotency_key", "task_id", "body"],
        },
    },
    handler=lambda args, **kw: _handle_task_comment(args, **kw),
)
