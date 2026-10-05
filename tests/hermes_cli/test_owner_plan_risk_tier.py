"""Every card an approved plan creates carries the risk tier the plan gave it.

P1 of the risk tier plan, through the real owner kernel. Every task a first
milestone commits (``owner_task_graph_commit``) and every task a plan change
creates (``add``, ``replace``, ``split``, ``merge`` through
``owner_project_plan_commit``) must state ``risk_tier``, the integer 0, 1 or 2:

* a missing or unknown tier is refused as ``invalid_argument`` naming
  ``risk_tier``, before the owner is asked, so nothing is created or archived;
* an approved tier lands on exactly the card whose specification states it.

P5 (tier plan section h): a ``replace``, ``split`` or ``merge`` never creates
a card below the highest tier of the cards it replaces, and a new Project's
root card takes the highest tier of its tasks. A replay keeps a root its
crashed commit already wrote, as written.
"""

from __future__ import annotations

import contextlib

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import owner_workspace as ow
from hermes_cli import projects_db

# The owner-workspace harness (authorized proposal context, auto-approver,
# canonical payloads) is reused verbatim, so every commit here goes through
# the REAL kernel entry point.
from tests.hermes_cli.test_owner_workspace import (  # noqa: F401
    _CrashInjected,
    _board_for_project,
    _bootstrap_board,
    _commit_project_plan,
    _commit_task_graph,
    _configured_provider,
    _expire_lock,
    _project_plan_args,
    _project_task_ref,
    _task_graph_args,
    _temporarily_patch,
    _with_approver,
    _work_rows,
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


def _existing_tasks(setup: dict, *titles: str, tiers=None) -> list[tuple[str, dict]]:
    with kb.connect(board=setup["board"]) as conn:
        ids = [
            kb.create_task(
                conn, title=title, assignee="default",
                project_id=setup["project_id"], risk_tier=tier,
            )
            for title, tier in zip(titles, tiers or [None] * len(titles))
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
    # Tier-0 targets, so no replacement floor (P5) lifts a stated tier.
    (_, first), (_, second) = _existing_tasks(
        setup, "First target", "Second target", tiers=[0, 0],
    )
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


# ---------------------------------------------------------------------------
# P5: no replacement, split or merge drops below the work it replaces
# ---------------------------------------------------------------------------


def _row(conn, task_id: str):
    return conn.execute("SELECT * FROM tasks WHERE id = ?", (task_id,)).fetchone()


@pytest.mark.parametrize(
    ("path", "target_tiers", "stated", "expected"),
    [
        ("replace", [2], [0], [2]),
        ("replace", [None], [1], [2]),
        ("split", [1], [0, 2], [1, 2]),
        ("merge", [0, 1], [0], [1]),
        ("merge", [0, 0], [1], [1]),
    ],
    ids=["replace", "replace-untiered", "split", "merge", "merge-above-the-floor"],
)
def test_a_replacement_split_or_merge_keeps_the_highest_tier_it_replaces(
    ctx, path, target_tiers, stated, expected,
):
    """A target without a tier counts as tier 2; a higher stated tier is kept."""
    setup = _bootstrap_board(ctx)
    titles = [f"Replaced work {index}" for index in range(len(target_tiers))]
    targets = [ref for _, ref in _existing_tasks(setup, *titles, tiers=target_tiers)]
    successors = [_spec(f"Successor {index}", tier) for index, tier in enumerate(stated)]
    if path == "replace":
        change = {
            "action": "replace", "reason": "Rescope the stalled task.",
            "target": targets[0], "replacement": successors[0],
        }
    elif path == "split":
        change = {
            "action": "split",
            "reason": "The current task is too broad to verify safely.",
            "target": targets[0],
            "replacements": [
                {**spec, "parents": [] if index == 0 else [0]}
                for index, spec in enumerate(successors)
            ],
        }
    else:
        change = {
            "action": "merge",
            "reason": "One coherent deliverable is easier to own and verify.",
            "targets": targets, "replacement": successors[0],
        }
    key = "-".join(str(part) for part in ("plan-floor", path, *target_tiers, *stated))
    approver = _with_approver(ctx.session)
    try:
        result = _commit_project_plan(
            ctx, **_project_plan_args(setup, [change], idempotency_key=key),
        )
    finally:
        approver.join()

    assert result["ok"] is True
    with kb.connect(board=setup["board"]) as conn:
        created = [_row(conn, task_id) for task_id in result["created_task_ids"]]
    assert [row["risk_tier"] for row in created] == expected
    # Pinned at the tier it got, on a clean seal.
    assert [row["reasoning_effort"] for row in created] == [
        "max" if tier == 2 else "high" for tier in expected
    ]
    for row in created:
        assert kb.task_policy_lock_error(row) is None


@pytest.mark.parametrize(
    ("tiers", "root_tier", "effort"),
    [((0, 1), 1, "high"), ((0, 0), 0, "high"), ((1, 2), 2, "max")],
    ids=["highest-is-1", "all-0", "highest-is-2"],
)
def test_a_new_projects_root_takes_the_highest_tier_of_its_tasks(
    ctx, tiers, root_tier, effort,
):
    args = _task_graph_args(idempotency_key=f"graph-root-tier-{tiers[0]}-{tiers[1]}")
    for task, tier in zip(args["tasks"], tiers):
        task["risk_tier"] = tier
    approver = _with_approver(ctx.session)
    try:
        result = _commit_task_graph(ctx, **args)
    finally:
        approver.join()

    assert result["ok"] is True
    with kb.connect(board=result["board"]) as conn:
        root = _row(conn, result["root_task_id"])
    assert (root["risk_tier"], root["reasoning_effort"]) == (root_tier, effort)
    assert kb.task_policy_lock_error(root) is None


# ---------------------------------------------------------------------------
# P5 recovery: a replay keeps the root its crashed commit wrote
# ---------------------------------------------------------------------------


def _graph_root(ctx, args: dict):
    """The root a task-graph commit wrote, read on a fresh connection."""
    root_key = "owgraph_" + ow._derive_id(ctx, args["idempotency_key"], "graph-root")
    board = _board_for_project(args["project_name"])
    with contextlib.closing(kb.connect(board=board)) as conn:
        return conn.execute(
            "SELECT * FROM tasks WHERE idempotency_key = ?", (root_key,),
        ).fetchone()


def _crash_at_the_terminal_receipt(ctx, args: dict, *, before_p5: bool):
    """Commit *args*, crash at its terminal receipt, expire its lease and
    return the root it wrote. The replay runs in the same HERMES_HOME, on
    fresh connections.

    *before_p5* patches only the new root-tier seam, so the root goes to the
    kernel without a tier, exactly as the code before P5 wrote it.
    """
    def crash(*_args, **_kwargs):
        raise _CrashInjected("at the terminal receipt")

    root_tier = (lambda tiers: None) if before_p5 else ow.highest_risk_tier
    approver = _with_approver(ctx.session)
    with _temporarily_patch(ow, "highest_risk_tier", root_tier):
        with _temporarily_patch(ow, "_finalize_receipt", crash):
            with pytest.raises(_CrashInjected):
                _commit_task_graph(ctx, **args)
    approver.join()
    _expire_lock(ctx, args["idempotency_key"])
    return _graph_root(ctx, args)


def _pin(row) -> tuple:
    return (row["id"], row["risk_tier"], row["reasoning_effort"], row["model_policy_lock"])


@pytest.mark.parametrize(
    ("written_by", "tier", "effort"),
    [(None, 1, "high"), ("code-before-p5", 2, "max"), ("this-code", 1, "high")],
    ids=["fresh-root", "base-to-head", "head-to-head"],
)
def test_a_replay_keeps_the_root_its_crashed_commit_wrote(ctx, written_by, tier, effort):
    """Tasks at tiers 1 and 1. A fresh root takes tier 1. A replay keeps the
    root the crashed commit wrote, at tier 1 or, written before P5, at tier 2,
    with its effort and seal unchanged, and creates nothing twice."""
    args = _task_graph_args(idempotency_key="graph-root-recovery")
    written = None
    if written_by is not None:
        written = _crash_at_the_terminal_receipt(
            ctx, args, before_p5=written_by == "code-before-p5",
        )
    approver = _with_approver(ctx.session)
    result = _commit_task_graph(ctx, **args)
    approver.join()

    assert (result["ok"], result["task_count"]) == (True, 2)
    # The root and its two children, none of them twice.
    assert len(_work_rows(result["board"], result["project_id"])) == 3
    root = _graph_root(ctx, args)
    assert root["id"] == result["root_task_id"]
    assert (root["risk_tier"], root["reasoning_effort"]) == (tier, effort)
    assert kb.task_policy_lock_error(root) is None
    if written is not None:
        assert _pin(root) == _pin(written)


@pytest.mark.parametrize(
    ("before_p5", "column", "value"),
    [
        (False, "risk_tier", 0),
        (False, "risk_tier", 2),
        (True, "reasoning_effort", "high"),
        (True, "model_policy_lock", "high"),
    ],
    ids=["tier-0", "tier-2-on-a-tier-1-pin", "effort-high-at-tier-2", "seal-for-high-at-tier-2"],
)
def test_a_replay_refuses_a_root_matching_neither_derivation(ctx, before_p5, column, value):
    """Neither this code's root (tier 1, high, its seal) nor the one written
    before P5 (tier 2, max, its seal): the replay fails closed and rewrites
    nothing."""
    args = _task_graph_args(idempotency_key="graph-root-tampered")
    root = _crash_at_the_terminal_receipt(ctx, args, before_p5=before_p5)
    if column == "model_policy_lock":
        # The seal minted for *value* on the root's own route.
        value = kb.mint_policy_lock(
            root["assignee"], root["provider_override"], root["model_override"],
            value, root["execution_tier"],
        )
    board = _board_for_project(args["project_name"])
    with contextlib.closing(kb.connect(board=board)) as conn:
        conn.execute(f"UPDATE tasks SET {column} = ? WHERE id = ?", (value, root["id"]))
        conn.commit()

    approver = _with_approver(ctx.session)
    with pytest.raises(ow.OwnerWorkspaceError) as excinfo:
        _commit_task_graph(ctx, **args)
    approver.join()

    assert excinfo.value.code == "crash_recovery_failed"
    assert _graph_root(ctx, args)[column] == value
