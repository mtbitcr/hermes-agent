"""Every card an approved plan creates carries the risk tier the plan gave it.

P1 of the risk tier plan, through the real owner kernel. Every task a first
milestone commits (``owner_task_graph_commit``) and every task a plan change
creates (``add``, ``replace``, ``split``, ``merge`` through
``owner_project_plan_commit``) must state ``risk_tier``, the integer 0, 1 or 2:

* a missing or unknown tier is refused as ``invalid_argument`` naming
  ``risk_tier``, before the owner is asked, so nothing is created or archived;
* an approved tier lands on exactly the card whose specification states it.
"""

from __future__ import annotations

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import owner_workspace as ow
from hermes_cli import projects_db

# The owner-workspace harness (authorized proposal context, auto-approver,
# canonical payloads) is reused verbatim, so every commit here goes through
# the REAL kernel entry point.
from tests.hermes_cli.test_owner_workspace import (  # noqa: F401
    _bootstrap_board,
    _commit_project_plan,
    _commit_task_graph,
    _configured_provider,
    _project_plan_args,
    _project_task_ref,
    _task_graph_args,
    _with_approver,
    ctx,
)
from tools import approval

_MISSING = object()
_REFUSED_TIERS = [
    pytest.param(tier, label, id=label)
    for tier, label in (
        (_MISSING, "missing"), (3, "3"), (-1, "minus-1"), ("1", "string-1"),
        (1.5, "1.5"), (True, "true"), (None, "null"),
    )
]


def _spec(title: str, tier) -> dict:
    spec = {
        "title": title,
        "body": "Produce the owner-visible result.",
        "assignee": "default",
        "execution_tier": "routine",
    }
    if tier is not _MISSING:
        spec["risk_tier"] = tier
    return spec


def _add_change(title: str, tier) -> dict:
    return {
        "action": "add",
        "reason": "Create one bounded owner-approved task.",
        **_spec(title, tier),
        "existing_parents": [],
        "new_parents": [],
    }


def _existing_tasks(setup: dict, *titles: str) -> list[tuple[str, dict]]:
    with kb.connect(board=setup["board"]) as conn:
        ids = [
            kb.create_task(
                conn, title=title, assignee="default",
                project_id=setup["project_id"],
            )
            for title in titles
        ]
        return [(task_id, _project_task_ref(conn, task_id)) for task_id in ids]


def _project_titles(board: str, project_id: str) -> set[str]:
    with kb.connect(board=board) as conn:
        return {
            task.title
            for task in kb.list_tasks(conn)
            if task.project_id == project_id
        }


@pytest.mark.parametrize(("tier", "label"), _REFUSED_TIERS)
@pytest.mark.parametrize(
    "path", ["add", "replace", "split", "merge", "task_graph"],
)
def test_the_apply_path_refuses_a_missing_or_unknown_tier(ctx, path, tier, label):
    if path == "task_graph":
        args = _task_graph_args(
            idempotency_key=f"graph-tier-{label}",
            project_name="Tier Refusal Project",
        )
        args["tasks"][0]["risk_tier"] = 1
        if tier is _MISSING:
            args["tasks"][1].pop("risk_tier", None)
        else:
            args["tasks"][1]["risk_tier"] = tier
        approval.unregister_gateway_notify(ctx.session)

        with pytest.raises(ow.OwnerWorkspaceError) as excinfo:
            _commit_task_graph(ctx, **args)

        assert excinfo.value.code == "invalid_argument"
        assert "risk_tier" in str(excinfo.value)
        with projects_db.connect_closing() as pconn:
            names = {
                project.name
                for project in projects_db.list_projects(pconn, include_archived=True)
            }
        assert "Tier Refusal Project" not in names
        return

    setup = _bootstrap_board(ctx)
    (left_id, left), (right_id, right) = _existing_tasks(
        setup, "Left half", "Right half",
    )
    if path == "add":
        # The first add is valid: the refusal is all-or-nothing.
        changes = [
            _add_change("Tiered deliverable", 1),
            _add_change("Untiered deliverable", tier),
        ]
    elif path == "replace":
        changes = [{
            "action": "replace",
            "reason": "Rescope the stalled task.",
            "target": left,
            "replacement": _spec("Untiered deliverable", tier),
        }]
    elif path == "split":
        changes = [{
            "action": "split",
            "reason": "The current task is too broad to verify safely.",
            "target": left,
            "replacements": [
                {**_spec("Tiered deliverable", 1), "parents": []},
                {**_spec("Untiered deliverable", tier), "parents": [0]},
            ],
        }]
    else:
        changes = [{
            "action": "merge",
            "reason": "One coherent deliverable is easier to own and verify.",
            "targets": [left, right],
            "replacement": _spec("Untiered deliverable", tier),
        }]
    # Nobody is there to approve: a refusal must come before the owner is
    # ever asked.
    approval.unregister_gateway_notify(ctx.session)

    with pytest.raises(ow.OwnerWorkspaceError) as excinfo:
        _commit_project_plan(
            ctx,
            **_project_plan_args(
                setup, changes, idempotency_key=f"plan-tier-{path}-{label}",
            ),
        )

    assert excinfo.value.code == "invalid_argument"
    assert "risk_tier" in str(excinfo.value)
    titles = _project_titles(setup["board"], setup["project_id"])
    assert "Tiered deliverable" not in titles
    assert "Untiered deliverable" not in titles
    with kb.connect(board=setup["board"]) as conn:
        assert kb.get_task(conn, left_id).status != "archived"
        assert kb.get_task(conn, right_id).status != "archived"


@pytest.mark.parametrize("path", ["task_graph", "project_plan", "merge"])
def test_each_committed_card_carries_its_own_tier(ctx, path):
    if path == "task_graph":
        args = _task_graph_args(idempotency_key="graph-tier-carried")
        args["tasks"][0]["risk_tier"] = 0
        args["tasks"][1]["risk_tier"] = 2
        approver = _with_approver(ctx.session)
        result = _commit_task_graph(ctx, **args)
        approver.join()

        assert result["ok"] is True
        with kb.connect(board=result["board"]) as conn:
            created = [kb.get_task(conn, task_id) for task_id in result["task_ids"]]
        assert [(task.title, task.risk_tier) for task in created] == [
            ("Prepare the release", 0),
            ("Verify the release", 2),
        ]
        return

    setup = _bootstrap_board(ctx)
    (_, first), (_, second) = _existing_tasks(setup, "First target", "Second target")
    if path == "project_plan":
        changes = [
            _add_change("Added deliverable", 0),
            {
                "action": "replace",
                "reason": "Rescope the stalled task.",
                "target": first,
                "replacement": _spec("Replacement deliverable", 1),
            },
            {
                "action": "split",
                "reason": "The current task is too broad to verify safely.",
                "target": second,
                "replacements": [
                    {**_spec("Split build", 2), "parents": []},
                    {**_spec("Split check", 0), "parents": [0]},
                ],
            },
        ]
        expected = [
            ("Added deliverable", 0),
            ("Replacement deliverable", 1),
            ("Split build", 2),
            ("Split check", 0),
        ]
    else:
        # A merge must be the only change in its owner approval.
        changes = [{
            "action": "merge",
            "reason": "One coherent deliverable is easier to own and verify.",
            "targets": [first, second],
            "replacement": _spec("Merged deliverable", 2),
        }]
        expected = [("Merged deliverable", 2)]
    approver = _with_approver(ctx.session)
    try:
        result = _commit_project_plan(
            ctx,
            **_project_plan_args(
                setup, changes, idempotency_key=f"plan-tier-carried-{path}",
            ),
        )
    finally:
        approver.join()

    assert result["ok"] is True
    with kb.connect(board=setup["board"]) as conn:
        created = [
            kb.get_task(conn, task_id) for task_id in result["created_task_ids"]
        ]
    assert [(task.title, task.risk_tier) for task in created] == expected
