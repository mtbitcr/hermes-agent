"""The owner approves the exact risk tier of every task a plan creates.

P1 of the risk tier plan, at the approval authority. A current stored proposal
(schema 4 for a new Project, schema 6 for a plan change) states ``risk_tier``
-- the integer 0, 1 or 2 -- on every task it creates: each new-Project task and
each ``add``, ``replace``, ``split`` and ``merge``. A missing or unknown tier
makes the stored proposal unusable as authority, refused before any run is
reserved, and the tier is part of the exact payload the owner approves.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import types
from unittest.mock import patch

import pytest

from gateway.platforms.api_server import APIServerAdapter, ResponseStore

_CONVERSATION = "raphael-owner-" + "4" * 32
_RESPONSE_ID = "resp_native_risk_tier_proposal"
_SECRET = "owner-executor-test-key"
_PROJECT_ID = "project_raphael"
_MISSING = object()
_KINDS = ["new_project", "add", "replace", "split", "merge"]
_UNKNOWN_TIERS = [
    pytest.param(tier, id=label)
    for tier, label in (
        (3, "3"), (-1, "minus-1"), ("1", "string-1"), (1.5, "1.5"),
        (True, "true"), (None, "null"),
    )
]
_REFUSAL = r"stored proposal (task|change) is invalid"


def _idempotency_key() -> str:
    return "conversation-" + hashlib.sha256(_RESPONSE_ID.encode("utf-8")).hexdigest()


def _native(task_id: str, title: str) -> dict:
    return {
        "id": task_id,
        "title": title,
        "status": "todo",
        "event_revision": 3,
        "parent_ids": [],
        "child_ids": [],
        "omitted_parent_count": 0,
        "omitted_child_count": 0,
    }


_LEFT = _native("task_left", "Left half")
_RIGHT = _native("task_right", "Right half")


def _ref(task: dict) -> str:
    canonical = json.dumps({
        "version": 1,
        "project_id": _PROJECT_ID,
        "task_id": task["id"],
        "title": task["title"],
        "status": task["status"],
        "event_revision": task["event_revision"],
        "parent_ids": task["parent_ids"],
        "child_ids": task["child_ids"],
        "omitted_parent_count": task["omitted_parent_count"],
        "omitted_child_count": task["omitted_child_count"],
    }, separators=(",", ":"), ensure_ascii=False)
    return "tr_" + base64.urlsafe_b64encode(
        hmac.new(
            _SECRET.encode("utf-8"), canonical.encode("utf-8"), hashlib.sha256,
        ).digest()
    ).decode("ascii").rstrip("=")


def _target(task: dict) -> dict:
    return {
        "task_id": task["id"],
        "expected_status": task["status"],
        "expected_revision": task["event_revision"],
    }


def _created(title: str, tier, **shape) -> dict:
    """One created task as the stored proposal states it."""
    task = {
        "title": title,
        "body": "Build the bounded change.",
        "assignee": "raphael-claude-worker",
        "responsibility": "R07",
        "execution_tier": "deep",
        "owned_paths": ["src"],
        **shape,
    }
    if tier is not _MISSING:
        task["risk_tier"] = tier
    return task


def _case(kind: str, tier, run_tier=_MISSING) -> tuple[dict, dict, str]:
    """(stored proposal, run payload, operation) for one created task.

    The stored proposal states *tier* on the created task under test; the run
    payload forwards *run_tier* (default: the stored tier) the way the
    Workspace forwards it. Any sibling created task is valid at tier 1.
    """
    forwarded = tier if run_tier is _MISSING else run_tier
    key = _idempotency_key()
    if kind == "new_project":
        proposal = {
            "schema_version": 4,
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
            "tasks": [
                _created("Draft the workshop plan", 1, parents=[]),
                _created("Check the workshop plan", tier, parents=[0]),
            ],
        }
        payload = {
            "idempotency_key": key,
            "mode": "new",
            "project_name": proposal["project_name"],
            "project_description": proposal["project_description"],
            "project_id": None,
            "request_title": proposal["request_title"],
            "specification": proposal["specification"],
            "current_milestone": proposal["current_milestone"],
            "owner_visible_result": proposal["owner_visible_result"],
            "root_assignee": "default",
            "tasks": [
                _created("Draft the workshop plan", 1, parents=[]),
                _created("Check the workshop plan", forwarded, parents=[0]),
            ],
            "later_milestones": proposal["later_milestones"],
        }
        return proposal, payload, "owner_task_graph_commit"

    reason = "The approved milestone needs this work."
    if kind == "add":
        stored = {
            "action": "add", "reason": reason,
            **_created("Prepare the milestone", tier),
            "existing_parent_refs": [], "new_parents": [],
        }
        change = {
            "action": "add", "reason": reason,
            **_created("Prepare the milestone", forwarded),
            "existing_parents": [], "new_parents": [],
        }
    elif kind == "replace":
        stored = {
            "action": "replace", "reason": reason, "target_ref": _ref(_LEFT),
            "replacement": _created(
                "Rescoped deliverable", tier, body_mode="rewrite",
            ),
        }
        change = {
            "action": "replace", "reason": reason, "target": _target(_LEFT),
            "replacement": _created(
                "Rescoped deliverable", forwarded, body_mode="rewrite",
            ),
        }
    elif kind == "split":
        stored = {
            "action": "split", "reason": reason, "target_ref": _ref(_LEFT),
            "replacements": [
                _created("Build the bounded change", 1, parents=[]),
                _created("Check the bounded change", tier, parents=[0]),
            ],
        }
        change = {
            "action": "split", "reason": reason, "target": _target(_LEFT),
            "replacements": [
                _created("Build the bounded change", 1, parents=[]),
                _created("Check the bounded change", forwarded, parents=[0]),
            ],
        }
    else:
        stored = {
            "action": "merge", "reason": reason,
            "target_refs": [_ref(_LEFT), _ref(_RIGHT)],
            "replacement": _created("Merged deliverable", tier),
        }
        change = {
            "action": "merge", "reason": reason,
            "targets": [_target(_LEFT), _target(_RIGHT)],
            "replacement": _created("Merged deliverable", forwarded),
        }
    proposal = {
        "schema_version": 6,
        "kind": "project_change_proposal",
        "mode": "existing",
        "request_title": "Adapt the approved milestone",
        "summary": "Change the plan as approved.",
        "project_size": "small",
        "specification": "Apply the approved change.",
        "current_milestone": "Adapt the approved milestone",
        "owner_visible_result": "The owner can review the changed milestone.",
        "impact": ["Keeps completed work intact."],
        "later_milestones": [],
        "changes": [stored],
    }
    payload = {
        "idempotency_key": key,
        "project_id": _PROJECT_ID,
        "trigger": "owner_request",
        "request_title": proposal["request_title"],
        "summary": proposal["summary"],
        "specification": proposal["specification"],
        "current_milestone": proposal["current_milestone"],
        "owner_visible_result": proposal["owner_visible_result"],
        "later_milestones": proposal["later_milestones"],
        "changes": [change],
    }
    return proposal, payload, "owner_project_plan_commit"


def _validate(proposal: dict, payload: dict, operation: str) -> dict:
    """Store *proposal* as the conversation's final reply and validate a run."""
    new = operation == "owner_task_graph_commit"
    authority = {
        "proposal_profile": "default",
        "conversation": _CONVERSATION,
        "response_id": _RESPONSE_ID,
        "claim_id": "claim_" + "6" * 32,
        "operation": operation,
        "idempotency_key": _idempotency_key(),
        "payload": payload,
    }
    context = (
        {
            "profile": "default", "mode": "new", "project_slug": None,
            "project_name": "Workshop pilot",
        }
        if new
        else {
            "profile": "default", "mode": "existing",
            "project_slug": "raphael-workspace",
            "project_name": "Raphael Workspace",
        }
    )
    snapshot = {
        "project": {"id": _PROJECT_ID},
        "planning_context": {
            "schema_version": 1,
            "actionable_count": 2,
            "omitted_terminal_count": 0,
            "actionable_truncated": False,
            "relations_truncated": False,
            "tasks": [_LEFT, _RIGHT],
        },
    }
    store = ResponseStore(max_size=10)
    adapter = APIServerAdapter.__new__(APIServerAdapter)
    adapter._response_store = store
    adapter._expected_api_key = lambda: _SECRET
    try:
        store.put(_RESPONSE_ID, {
            "response": {"id": _RESPONSE_ID, "created_at": 1},
            "conversation_history": [
                {"role": "user", "content": "Change the plan."},
                {"role": "assistant", "content": json.dumps(proposal)},
            ],
        })
        # Not asserted: a reply that is no actionable owner proposal is not
        # mapped, and the authority itself reports that it is not current.
        store.set_conversation(_CONVERSATION, _RESPONSE_ID, owner_proposal=True)
        with (
            patch(
                "hermes_cli.owner_workspace.resolve_owner_context",
                return_value=object(),
            ),
            patch(
                "hermes_cli.owner_workspace.read_project_snapshot",
                return_value=snapshot,
            ),
            patch(
                "hermes_cli.profiles.list_profiles",
                return_value=[types.SimpleNamespace(name="default")],
            ),
        ):
            return adapter._validated_owner_proposal_authority(
                authority, context, "default",
            )
    finally:
        store.close()


@pytest.mark.parametrize("kind", _KINDS)
def test_a_stored_proposal_without_a_risk_tier_authorizes_no_run(kind):
    with pytest.raises(ValueError, match=_REFUSAL):
        _validate(*_case(kind, _MISSING))

    # A pre-tier proposal (schema 3 for a new Project, 5 for a plan change)
    # states no tier at all: it stays readable as history, never authority.
    proposal, payload, operation = _case(kind, _MISSING)
    proposal["schema_version"] = 3 if kind == "new_project" else 5
    with pytest.raises(ValueError, match="owner proposal is not current"):
        _validate(proposal, payload, operation)

    # The same proposal with a tier is authority: only the tier was missing.
    assert _validate(*_case(kind, 1))["operation"] == operation


@pytest.mark.parametrize("tier", _UNKNOWN_TIERS)
@pytest.mark.parametrize("kind", _KINDS)
def test_an_unknown_risk_tier_authorizes_no_run(kind, tier):
    with pytest.raises(ValueError, match=_REFUSAL):
        _validate(*_case(kind, tier))


@pytest.mark.parametrize("kind", _KINDS)
def test_the_risk_tier_is_part_of_the_authority(kind):
    with pytest.raises(ValueError, match="run payload differs"):
        _validate(*_case(kind, 1, run_tier=2))

    binding = _validate(*_case(kind, 1, run_tier=1))
    assert binding["idempotency_key"] == _idempotency_key()
    bound = binding["payload"]
    created = (
        bound["tasks"][1]
        if kind == "new_project"
        else {
            "add": lambda change: change,
            "replace": lambda change: change["replacement"],
            "split": lambda change: change["replacements"][1],
            "merge": lambda change: change["replacement"],
        }[kind](bound["changes"][0])
    )
    assert created["risk_tier"] == 1
