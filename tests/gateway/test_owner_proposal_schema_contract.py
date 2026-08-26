"""Cross-repository owner-proposal schema contract.

The Workspace planner emits ``schema_version`` 3 for a new-project proposal and
4 for an existing-project change proposal, and every created task now names its
``execution_tier``. Hermes must recognise exactly those as actionable — an
older stored proposal stays readable in history but can no longer be committed,
because committing it would leave the kernel resolving a route from a class the
planner never stated.

The proposal objects below are the exact shapes ``src/lib/conversation-plan.ts``
produces; they are duplicated here on purpose because this file IS the contract
test between the two repositories.
"""

from __future__ import annotations

import json

import pytest

from gateway.platforms.api_server import (
    _OWNER_EXISTING_PROPOSAL_SCHEMA,
    _OWNER_NEW_PROPOSAL_SCHEMA,
    _owner_final_proposal,
    _owner_history_has_actionable_final_proposal,
)


def _workspace_new_proposal(**overrides):
    proposal = {
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


def _workspace_existing_proposal(**overrides):
    proposal = {
        "schema_version": 4,
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
            "execution_tier": "deep",
            "existing_parent_refs": [],
            "new_parents": [],
        }],
    }
    proposal.update(overrides)
    return proposal


def _history(proposal):
    return [
        {"role": "user", "content": "please do the thing"},
        {"role": "assistant", "content": json.dumps(proposal)},
    ]


def test_hermes_and_workspace_agree_on_the_actionable_versions():
    assert (_OWNER_NEW_PROPOSAL_SCHEMA, _OWNER_EXISTING_PROPOSAL_SCHEMA) == (3, 4)


@pytest.mark.parametrize(
    "proposal",
    [_workspace_new_proposal(), _workspace_existing_proposal()],
)
def test_current_workspace_proposals_grant_commit_authority(proposal):
    assert _owner_final_proposal(_history(proposal)) == proposal
    assert _owner_history_has_actionable_final_proposal(_history(proposal)) is True


@pytest.mark.parametrize(
    "proposal",
    [
        # The pre-tier versions the Workspace used to emit.
        _workspace_new_proposal(schema_version=2),
        _workspace_existing_proposal(schema_version=3),
        # Version numbers this build does not mint at all.
        _workspace_new_proposal(schema_version=4),
        _workspace_existing_proposal(schema_version=5),
        # Right version, wrong kind/mode pairing.
        _workspace_new_proposal(mode="existing"),
        _workspace_existing_proposal(kind="proposal"),
    ],
)
def test_non_current_proposals_grant_no_authority(proposal):
    assert _owner_final_proposal(_history(proposal)) is None
    assert _owner_history_has_actionable_final_proposal(_history(proposal)) is False


def test_existing_project_add_requires_execution_tier_in_the_run_payload():
    """The nested add validator is part of the authority, not just the header."""
    from gateway.platforms.api_server import _OWNER_PROPOSAL_ADD_KEYS

    add = _workspace_existing_proposal()["changes"][0]
    assert set(add) == set(_OWNER_PROPOSAL_ADD_KEYS)
    # Dropping the tier makes the change object unrecognisable, so the commit
    # cannot be authorized from it.
    assert set(add) - {"execution_tier"} != set(_OWNER_PROPOSAL_ADD_KEYS)
