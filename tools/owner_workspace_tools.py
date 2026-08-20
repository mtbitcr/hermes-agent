"""Owner-workspace tools over the deep kernel in ``hermes_cli.owner_workspace``.

Default-off, API-server-only surface (see ``toolsets.py``'s ``owner_workspace``
toolset — absent from ``_HERMES_CORE_TOOLS`` and every composite — and
``gateway/platforms/api_server.py``'s ``_create_agent``, the ONLY place that
folds it into ``enabled_toolsets``, gated on
``gateway.api_server.owner_workspace.enabled`` for the resolved profile).

Every tool schema is deliberately narrow: no author/profile/actor/session/
path/scope field is accepted from the model. Identity is resolved from
trusted request context (``resolve_owner_context``) inside the kernel.

The separate ``project_steward`` toolset exposes one read-only, owner-safe
snapshot without granting any owner-workspace mutation authority.
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


def _handle_project_steward_snapshot(args: dict, **kw) -> str:
    try:
        result = _kernel.project_steward_snapshot(
            project_id=args.get("project_id"),
            lookback_days=args.get("lookback_days", 7),
        )
        return _ok(result)
    except OwnerWorkspaceError as e:
        return tool_error(f"project_steward_snapshot: {e.message}")
    except Exception:
        logger.exception("project_steward_snapshot failed")
        return tool_error("project_steward_snapshot: internal error")



def _handle_task_graph(args: dict, **kw) -> str:
    try:
        ctx = resolve_owner_context()
        result = _kernel.commit_task_graph(
            ctx,
            idempotency_key=args.get("idempotency_key"),
            mode=args.get("mode"),
            project_name=args.get("project_name"),
            project_description=args.get("project_description"),
            project_id=args.get("project_id"),
            request_title=args.get("request_title"),
            specification=args.get("specification"),
            current_milestone=args.get("current_milestone"),
            owner_visible_result=args.get("owner_visible_result"),
            root_assignee=args.get("root_assignee"),
            tasks=args.get("tasks"),
            later_milestones=args.get("later_milestones"),
        )
        return _ok(result)
    except OwnerWorkspaceError as e:
        return tool_error(f"owner_task_graph_commit: {e.message}")
    except ValueError as e:
        return tool_error(f"owner_task_graph_commit: {e}")
    except Exception:
        logger.exception("owner_task_graph_commit failed")
        return tool_error("owner_task_graph_commit: internal error")


def _handle_project_plan(args: dict, **kw) -> str:
    try:
        ctx = resolve_owner_context()
        result = _kernel.commit_project_plan(
            ctx,
            idempotency_key=args.get("idempotency_key"),
            project_id=args.get("project_id"),
            anchor_task_id=args.get("anchor_task_id"),
            trigger=args.get("trigger"),
            request_title=args.get("request_title"),
            summary=args.get("summary"),
            specification=args.get("specification"),
            current_milestone=args.get("current_milestone"),
            owner_visible_result=args.get("owner_visible_result"),
            later_milestones=args.get("later_milestones"),
            changes=args.get("changes"),
        )
        return _ok(result)
    except OwnerWorkspaceError as e:
        return tool_error(f"owner_project_plan_commit: {e.message}")
    except ValueError as e:
        return tool_error(f"owner_project_plan_commit: {e}")
    except Exception:
        logger.exception("owner_project_plan_commit failed")
        return tool_error("owner_project_plan_commit: internal error")


def _handle_project_lifecycle(args: dict, **kw) -> str:
    try:
        ctx = resolve_owner_context()
        result = _kernel.set_project_archived(
            ctx,
            idempotency_key=args.get("idempotency_key"),
            project_id=args.get("project_id"),
            action=args.get("action"),
        )
        return _ok(result)
    except OwnerWorkspaceError as e:
        return tool_error(f"owner_project_lifecycle: {e.message}")
    except ValueError as e:
        return tool_error(f"owner_project_lifecycle: {e}")
    except Exception:
        logger.exception("owner_project_lifecycle failed")
        return tool_error("owner_project_lifecycle: internal error")


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
    name="project_steward_snapshot",
    toolset="project_steward",
    schema={
        "name": "project_steward_snapshot",
        "description": (
            "Read one bounded, owner-safe health snapshot for an existing "
            "Project. Returns recent progress, work needing attention, items "
            "awaiting review, active work, and stale candidates. It never "
            "returns task IDs, agent names, file paths, raw bodies, results, "
            "errors, or events, and it cannot mutate Project state."
        ),
        "parameters": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "project_id": {
                    "type": "string",
                    "description": "Exact Project identifier to inspect.",
                },
                "lookback_days": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 30,
                    "default": 7,
                    "description": "Recent-progress window in days.",
                },
            },
            "required": ["project_id"],
        },
    },
    handler=lambda args, **kw: _handle_project_steward_snapshot(args, **kw),
)


registry.register(
    name="owner_task_graph_commit",
    toolset="owner_workspace",
    schema={
        "name": "owner_task_graph_commit",
        "description": (
            "Commit one owner-approved Conversation proposal to the native "
            "Project and Kanban Task graph. For large projects, create only the "
            "current executable milestone (maximum 12 tasks) and keep future "
            "milestones as roadmap context. The board, author, actor, profile, "
            "session and filesystem scope are derived by the trusted kernel. "
            "Idempotent and guarded by one exact human confirmation."
        ),
        "parameters": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "idempotency_key": {
                    "type": "string",
                    "description": "Stable proposal key so retries cannot duplicate work.",
                },
                "mode": {
                    "type": "string",
                    "enum": ["new", "existing"],
                    "description": "Create a new Project or add work to an existing Project.",
                },
                "project_name": {
                    "type": "string",
                    "description": "Required only for mode=new.",
                },
                "project_description": {
                    "type": "string",
                    "description": "Optional plain-English Project description for mode=new.",
                },
                "project_id": {
                    "type": "string",
                    "description": "Required only for mode=existing; its board is derived server-side.",
                },
                "request_title": {
                    "type": "string",
                    "description": "Plain-English umbrella outcome for the approved request.",
                },
                "specification": {
                    "type": "string",
                    "description": "Complete approved specification, without credentials or secrets.",
                },
                "current_milestone": {
                    "type": "string",
                    "description": "The bounded Now milestone being committed.",
                },
                "owner_visible_result": {
                    "type": "string",
                    "description": "What the owner can inspect when this milestone is complete.",
                },
                "root_assignee": {
                    "type": "string",
                    "description": "Existing coordinator profile that reviews the completed milestone.",
                },
                "tasks": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": 12,
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "title": {"type": "string"},
                            "body": {"type": "string"},
                            "assignee": {
                                "type": "string",
                                "description": "Existing Hermes profile; validated by the kernel.",
                            },
                            "parents": {
                                "type": "array",
                                "items": {"type": "integer", "minimum": 0},
                                "description": "Indices of prerequisite tasks in this same array.",
                            },
                        },
                        "required": ["title", "body", "assignee", "parents"],
                    },
                },
                "later_milestones": {
                    "type": "array",
                    "maxItems": 12,
                    "items": {"type": "string"},
                    "description": "Visible Next/Later roadmap only; not executable Tasks yet.",
                },
            },
            "required": [
                "idempotency_key",
                "mode",
                "request_title",
                "specification",
                "current_milestone",
                "owner_visible_result",
                "root_assignee",
                "tasks",
            ],
        },
    },
    handler=lambda args, **kw: _handle_task_graph(args, **kw),
)


_PROJECT_TASK_REF = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "task_id": {"type": "string"},
        "expected_status": {"type": "string"},
        "expected_revision": {"type": "integer", "minimum": 1},
    },
    "required": ["task_id", "expected_status", "expected_revision"],
}

_PROJECT_TASK_SPEC = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "title": {"type": "string"},
        "body": {"type": "string"},
        "assignee": {"type": "string"},
    },
    "required": ["title", "body", "assignee"],
}

_PROJECT_REPLACEMENT_SPEC = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        **_PROJECT_TASK_SPEC["properties"],
        "parents": {
            "type": "array",
            "items": {"type": "integer", "minimum": 0},
        },
    },
    "required": ["title", "body", "assignee", "parents"],
}


registry.register(
    name="owner_project_plan_commit",
    toolset="owner_workspace",
    schema={
        "name": "owner_project_plan_commit",
        "description": (
            "Atomically apply one owner-approved Project Steward plan to an "
            "existing native Project board. Supports bounded add, split, merge, "
            "move, postpone, and cancel changes. Every referenced task carries "
            "the exact status and event revision last observed; any drift leaves "
            "the whole plan unchanged. Running, completed, and archived tasks "
            "cannot be rewritten. Merge or cancel must be approved alone."
        ),
        "parameters": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "idempotency_key": {"type": "string"},
                "project_id": {"type": "string"},
                "anchor_task_id": {"type": "string"},
                "trigger": {
                    "type": "string",
                    "enum": [
                        "owner_request",
                        "milestone_boundary",
                        "persistent_blocker",
                        "scheduled_review",
                    ],
                },
                "request_title": {"type": "string"},
                "summary": {"type": "string"},
                "specification": {"type": "string"},
                "current_milestone": {"type": "string"},
                "owner_visible_result": {"type": "string"},
                "later_milestones": {
                    "type": "array",
                    "maxItems": 12,
                    "items": {"type": "string"},
                },
                "changes": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": 12,
                    "items": {
                        "oneOf": [
                            {
                                "type": "object",
                                "additionalProperties": False,
                                "properties": {
                                    **_PROJECT_TASK_SPEC["properties"],
                                    "action": {"const": "add"},
                                    "reason": {"type": "string"},
                                    "existing_parents": {
                                        "type": "array",
                                        "items": _PROJECT_TASK_REF,
                                    },
                                    "new_parents": {
                                        "type": "array",
                                        "items": {"type": "integer", "minimum": 0},
                                    },
                                },
                                "required": [
                                    "action", "reason", "title", "body", "assignee",
                                    "existing_parents", "new_parents",
                                ],
                            },
                            {
                                "type": "object",
                                "additionalProperties": False,
                                "properties": {
                                    "action": {"const": "split"},
                                    "reason": {"type": "string"},
                                    "target": _PROJECT_TASK_REF,
                                    "replacements": {
                                        "type": "array",
                                        "minItems": 2,
                                        "maxItems": 6,
                                        "items": _PROJECT_REPLACEMENT_SPEC,
                                    },
                                },
                                "required": ["action", "reason", "target", "replacements"],
                            },
                            {
                                "type": "object",
                                "additionalProperties": False,
                                "properties": {
                                    "action": {"const": "merge"},
                                    "reason": {"type": "string"},
                                    "targets": {
                                        "type": "array",
                                        "minItems": 2,
                                        "maxItems": 6,
                                        "items": _PROJECT_TASK_REF,
                                    },
                                    "replacement": _PROJECT_TASK_SPEC,
                                },
                                "required": ["action", "reason", "targets", "replacement"],
                            },
                            {
                                "type": "object",
                                "additionalProperties": False,
                                "properties": {
                                    "action": {"const": "move"},
                                    "reason": {"type": "string"},
                                    "target": _PROJECT_TASK_REF,
                                    "to_status": {"type": "string", "enum": ["ready"]},
                                },
                                "required": ["action", "reason", "target", "to_status"],
                            },
                            {
                                "type": "object",
                                "additionalProperties": False,
                                "properties": {
                                    "action": {"enum": ["postpone", "cancel"]},
                                    "reason": {"type": "string"},
                                    "target": _PROJECT_TASK_REF,
                                },
                                "required": ["action", "reason", "target"],
                            },
                        ],
                    },
                },
            },
            "required": [
                "idempotency_key", "project_id", "anchor_task_id", "trigger",
                "request_title", "summary", "specification", "current_milestone",
                "owner_visible_result", "later_milestones", "changes",
            ],
        },
    },
    handler=lambda args, **kw: _handle_project_plan(args, **kw),
)


registry.register(
    name="owner_project_lifecycle",
    toolset="owner_workspace",
    schema={
        "name": "owner_project_lifecycle",
        "description": (
            "Archive or restore one owner-workspace Project while retaining "
            "its Tasks, history, documents and board. Hard delete is not "
            "available. The Project must be backed by this trusted owner's "
            "committed receipt. Idempotent and guarded by one exact human "
            "confirmation."
        ),
        "parameters": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "idempotency_key": {
                    "type": "string",
                    "description": "Stable action key so a retry is safe.",
                },
                "project_id": {"type": "string"},
                "action": {
                    "type": "string",
                    "enum": ["archive", "restore"],
                },
            },
            "required": ["idempotency_key", "project_id", "action"],
        },
    },
    handler=lambda args, **kw: _handle_project_lifecycle(args, **kw),
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
