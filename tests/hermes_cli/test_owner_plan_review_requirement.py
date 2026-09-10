"""A plan change may commit the same review requirement a first milestone can.

``owner_task_graph_commit`` already accepts an optional ``requires_review`` on
the tasks of a new Project's first milestone. These tests hold the PLAN-CHANGE
path (``owner_project_plan_commit``) to the same contract, entirely through the
real approval kernel:

* every change that CREATES work — ``add``, ``replace``, ``split``, ``merge`` —
  may state the requirement, and it lands on exactly the task whose approved
  specification states it;
* the requirement is really committed on the row, so the implementer's handover
  parks the card on the existing review lane instead of writing ``done``;
* the pre-existing READ-ONLY audit review task (``raphael-verifier``) can never
  carry it — it IS the review, not work awaiting one; and
* a plan that does not ask for review digests EXACTLY as it did before the
  field existed, so no committed receipt or replay is disturbed.
"""

from __future__ import annotations

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import owner_workspace as ow

# The owner-workspace harness (authorized proposal context, auto-approver,
# canonical plan payload) is reused verbatim rather than re-implemented, so
# every commit here goes through the REAL kernel entry point.
from tests.hermes_cli.test_owner_workspace import (  # noqa: F401
    _CrashInjected,
    _bootstrap_board,
    _commit_project_plan,
    _configured_provider,
    _expire_lock,
    _install_profiles,
    _project_plan_args,
    _project_task_ref,
    _temporarily_patch,
    _with_approver,
    ctx,
)
from tools import approval

REVIEWER = "raphael-verifier"


def _spec(title: str, *, requires_review: bool | None = None, **overrides) -> dict:
    """One created-task specification, stating the requirement only if asked."""
    spec = {
        "title": title,
        "body": "Produce the owner-visible result.",
        "assignee": "default",
        "execution_tier": "routine",
    }
    if requires_review is not None:
        spec["requires_review"] = requires_review
    spec.update(overrides)
    return spec


def _add_change(title: str, *, requires_review: bool | None = None, **overrides) -> dict:
    change = {
        "action": "add",
        "reason": "Create one bounded owner-approved task.",
        **_spec(title, requires_review=requires_review),
        "existing_parents": [],
        "new_parents": [],
    }
    change.update(overrides)
    return change


def _commit(ctx, setup: dict, changes: list[dict], key: str) -> dict:
    approver = _with_approver(ctx.session)
    try:
        return _commit_project_plan(
            ctx, **_project_plan_args(setup, changes, idempotency_key=key)
        )
    finally:
        approver.join()


def _project_titles(board: str, project_id: str) -> set[str]:
    with kb.connect(board=board) as conn:
        return {
            task.title
            for task in kb.list_tasks(conn)
            if task.project_id == project_id
        }


# ---------------------------------------------------------------------------
# The requirement reaches the created row
# ---------------------------------------------------------------------------


def test_a_plan_add_commits_the_requirement_on_the_task_that_states_it(ctx):
    """The approved specification decides, per task — never the whole plan."""
    setup = _bootstrap_board(ctx)
    result = _commit(
        ctx,
        setup,
        [
            _add_change("Reviewed deliverable", requires_review=True),
            _add_change("Ordinary deliverable"),
            _add_change("Explicitly unreviewed deliverable", requires_review=False),
        ],
        "plan-add-requires-review",
    )

    assert result["ok"] is True
    reviewed_id, ordinary_id, declined_id = result["created_task_ids"]
    with kb.connect(board=setup["board"]) as conn:
        reviewed = kb.get_task(conn, reviewed_id)
        ordinary = kb.get_task(conn, ordinary_id)
        declined = kb.get_task(conn, declined_id)

    assert reviewed.title == "Reviewed deliverable"
    assert reviewed.requires_review is True
    assert ordinary.title == "Ordinary deliverable"
    assert ordinary.requires_review is False, (
        "the requirement must land on the task that states it, not its siblings"
    )
    # Stating it as false is accepted and means exactly what omitting it means.
    assert declined.requires_review is False


def test_a_committed_plan_handover_parks_for_review_instead_of_completing(ctx):
    """End to end: the committed requirement changes what the handover does."""
    _install_profiles(REVIEWER)
    setup = _bootstrap_board(ctx)
    result = _commit(
        ctx,
        setup,
        [
            _add_change("Reviewed deliverable", requires_review=True),
            _add_change("Ordinary deliverable"),
        ],
        "plan-handover-parks",
    )
    assert result["ok"] is True
    reviewed_id, ordinary_id = result["created_task_ids"]

    with kb.connect(board=setup["board"]) as conn:
        # The plan's own activation already released this work; nothing here
        # moves a card by hand.
        assert kb.get_task(conn, reviewed_id).status == "ready"

        run = kb.claim_task(conn, reviewed_id, claimer="default:1")
        assert run is not None
        assert kb.complete_task(
            conn, reviewed_id, summary="implemented",
            expected_run_id=run.current_run_id,
        ) is True

        plain_run = kb.claim_task(conn, ordinary_id, claimer="default:2")
        assert plain_run is not None
        assert kb.complete_task(
            conn, ordinary_id, summary="implemented",
            expected_run_id=plain_run.current_run_id,
        ) is True

        parked = kb.get_task(conn, reviewed_id)
        plain = kb.get_task(conn, ordinary_id)

    assert parked.status == "review", (
        f"a committed review requirement must park the handover, got "
        f"{parked.status!r}"
    )
    assert parked.completed_at is None
    assert parked.assignee == REVIEWER
    assert plain.status == "done", (
        "work that never asked for review must still complete on handover"
    )


@pytest.mark.parametrize("action", ["add", "replace", "split", "merge"])
def test_every_creating_change_commits_the_requirement_it_states(ctx, action):
    setup = _bootstrap_board(ctx)
    with kb.connect(board=setup["board"]) as conn:
        left_id = kb.create_task(
            conn, title="Left half", assignee="default",
            project_id=setup["project_id"],
        )
        right_id = kb.create_task(
            conn, title="Right half", assignee="default",
            project_id=setup["project_id"],
        )
        left = _project_task_ref(conn, left_id)
        right = _project_task_ref(conn, right_id)

    reviewed = _spec("Reviewed deliverable", requires_review=True)
    if action == "add":
        change = _add_change("Reviewed deliverable", requires_review=True)
    elif action == "replace":
        change = {
            "action": "replace",
            "reason": "Rescope the stalled task.",
            "target": left,
            "replacement": reviewed,
        }
    elif action == "split":
        change = {
            "action": "split",
            "reason": "The current task is too broad to verify safely.",
            "target": left,
            "replacements": [
                {**reviewed, "parents": []},
                # Deliberately silent: per-task granularity, not a blanket
                # write across everything the change creates.
                {**_spec("Unreviewed deliverable"), "parents": [0]},
            ],
        }
    else:
        change = {
            "action": "merge",
            "reason": "One coherent deliverable is easier to own and verify.",
            "targets": [left, right],
            "replacement": reviewed,
        }

    result = _commit(ctx, setup, [change], f"plan-{action}-requires-review")

    assert result["ok"] is True
    with kb.connect(board=setup["board"]) as conn:
        created = [
            kb.get_task(conn, task_id) for task_id in result["created_task_ids"]
        ]

    assert created[0].title == "Reviewed deliverable"
    assert created[0].requires_review is True
    if action == "split":
        assert created[1].title == "Unreviewed deliverable"
        assert created[1].requires_review is False, (
            "only the replacement whose specification states the requirement "
            "may carry it"
        )
    else:
        assert len(created) == 1


def test_a_crash_before_the_requirement_is_written_leaves_the_work_parked(ctx):
    """The recovered path commits what the crashed attempt never wrote.

    The board write and the requirement write are separate statements, so a
    crash can land between them. The created work is still parked there —
    un-promotable and un-claimable — and the replay that rebuilds the committed
    result writes the requirement before it activates anything.
    """
    setup = _bootstrap_board(ctx)
    args = _project_plan_args(
        setup,
        [_add_change("Reviewed deliverable", requires_review=True)],
        idempotency_key="plan-review-crash",
    )

    def crash(*a, **k):
        raise _CrashInjected("plan_review_requirement")

    approver = _with_approver(ctx.session)
    with _temporarily_patch(ow, "_commit_plan_review_requirements", crash):
        with pytest.raises(_CrashInjected):
            _commit_project_plan(ctx, **args)
    approver.join()

    with kb.connect(board=setup["board"]) as conn:
        crashed = [
            task for task in kb.list_tasks(conn)
            if task.title == "Reviewed deliverable"
        ]
    assert len(crashed) == 1
    assert crashed[0].status == kb.PARKED_STATUS, (
        "a failure before the terminal receipt must leave the created work "
        "parked, never runnable without its committed requirement"
    )
    assert crashed[0].requires_review is False

    # The replay rebuilds the applied result from the board's own event; it
    # asks for no second approval, so no approver is registered here.
    approval.unregister_gateway_notify(ctx.session)
    _expire_lock(ctx, "plan-review-crash")
    replayed = _commit_project_plan(ctx, **args)

    assert replayed["ok"] is True
    with kb.connect(board=setup["board"]) as conn:
        recovered = [
            task for task in kb.list_tasks(conn)
            if task.title == "Reviewed deliverable"
        ]
    assert len(recovered) == 1, "the replay must not create a second task"
    assert recovered[0].requires_review is True
    assert recovered[0].status != kb.PARKED_STATUS


# ---------------------------------------------------------------------------
# The read-only audit review task is never parked awaiting a review
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("action", ["add", "replace", "split", "merge"])
def test_a_plan_refuses_the_requirement_on_a_read_only_review_task(ctx, action):
    """REGRESSION: the requirement can never leak onto the audit review task."""
    _install_profiles(REVIEWER)
    setup = _bootstrap_board(ctx)
    with kb.connect(board=setup["board"]) as conn:
        left_id = kb.create_task(
            conn, title="Left half", assignee="default",
            project_id=setup["project_id"],
        )
        right_id = kb.create_task(
            conn, title="Right half", assignee="default",
            project_id=setup["project_id"],
        )
        left = _project_task_ref(conn, left_id)
        right = _project_task_ref(conn, right_id)

    audit = _spec(
        "Confirm the owner-visible result",
        requires_review=True,
        assignee=REVIEWER,
    )
    if action == "add":
        change = _add_change(
            "Confirm the owner-visible result",
            requires_review=True,
            assignee=REVIEWER,
        )
    elif action == "replace":
        change = {
            "action": "replace",
            "reason": "Hand the outcome to an auditor.",
            "target": left,
            "replacement": audit,
        }
    elif action == "split":
        change = {
            "action": "split",
            "reason": "Separate the audit from the work.",
            "target": left,
            "replacements": [
                {**_spec("Build the bounded change"), "parents": []},
                {**audit, "parents": [0]},
            ],
        }
    else:
        change = {
            "action": "merge",
            "reason": "Audit both outcomes at once.",
            "targets": [left, right],
            "replacement": audit,
        }

    with pytest.raises(ow.OwnerWorkspaceError) as excinfo:
        _commit_project_plan(
            ctx,
            **_project_plan_args(
                setup, [change], idempotency_key=f"plan-{action}-verifier",
            ),
        )

    assert excinfo.value.code == "invalid_argument"
    assert "requires_review" in str(excinfo.value)
    # Refused before the owner is ever asked, so nothing was created and no
    # existing task was archived.
    titles = _project_titles(setup["board"], setup["project_id"])
    assert "Confirm the owner-visible result" not in titles
    assert "Build the bounded change" not in titles
    with kb.connect(board=setup["board"]) as conn:
        assert kb.get_task(conn, left_id).status != "archived"
        assert kb.get_task(conn, right_id).status != "archived"


def test_the_refusal_wins_over_any_other_verifier_specific_complaint(ctx):
    """A payload wrong in two ways still reports the review refusal.

    A read-only reviewer card must also carry an empty ownership scope. When a
    change breaks BOTH rules, the review refusal is the one the owner sees:
    narrowing ``owned_paths`` would not make the requirement acceptable.
    """
    _install_profiles(REVIEWER)
    setup = _bootstrap_board(ctx)

    with pytest.raises(ow.OwnerWorkspaceError) as excinfo:
        _commit_project_plan(
            ctx,
            **_project_plan_args(
                setup,
                [
                    _add_change(
                        "Confirm the owner-visible result",
                        requires_review=True,
                        assignee=REVIEWER,
                        owned_paths=["src"],
                    )
                ],
                idempotency_key="plan-verifier-doubly-wrong",
            ),
        )

    assert excinfo.value.code == "invalid_argument"
    assert "requires_review" in str(excinfo.value)


@pytest.mark.parametrize("value", [1, 0, "true", [True]])
def test_a_non_boolean_requirement_is_refused(ctx, value):
    setup = _bootstrap_board(ctx)

    with pytest.raises(ow.OwnerWorkspaceError) as excinfo:
        _commit_project_plan(
            ctx,
            **_project_plan_args(
                setup,
                [_add_change("Reviewed deliverable", requires_review=value)],
                idempotency_key=f"plan-non-boolean-{value!r}",
            ),
        )

    assert excinfo.value.code == "invalid_argument"
    assert "requires_review" in str(excinfo.value)


# ---------------------------------------------------------------------------
# A plan that says nothing about review is byte-for-byte the plan it was
# ---------------------------------------------------------------------------

# Captured from the PRE-CHANGE implementation: the pristine baseline tree
# (/workspace/baseline, copied to /tmp/baseline-probe) ran this very test and
# digested this very payload to this value. It is frozen here so that accepting
# ``requires_review`` can never disturb the request digest — and therefore the
# receipt identity and replay — of any plan that omits it.
_OMITTED_REQUIREMENT_DIGEST = (
    "0eb2ef8b4b6d9c464913a26ffedd5234910843fea59c2ff741deda8085bf6723"
)


def _fixed_changes(*, requires_review: bool = False) -> list[dict]:
    """One frozen plan covering every creating shape, digested as literals.

    Nothing here comes from a board: task ids, statuses and revisions are
    fixed text so the digest depends only on the normalization, exactly the
    property under test.
    """
    stated = {"requires_review": True} if requires_review else {}
    return [
        {
            "action": "add",
            "reason": "Create one bounded owner-approved task.",
            "title": "Prepare the approved deliverable",
            "body": "Produce the owner-visible result.",
            "assignee": "default",
            "responsibility": "B04",
            "execution_tier": "routine",
            "existing_parents": [],
            "new_parents": [],
            **stated,
        },
        {
            "action": "replace",
            "reason": "Rescope the stalled task.",
            "target": {
                "task_id": "frozen-target-1",
                "expected_status": "todo",
                "expected_revision": 1,
            },
            "replacement": {
                "title": "Rescoped deliverable",
                "body": "Deliver the smaller outcome.",
                "assignee": "default",
                "execution_tier": "routine",
                **stated,
            },
        },
        {
            "action": "split",
            "reason": "The current task is too broad to verify safely.",
            "target": {
                "task_id": "frozen-target-2",
                "expected_status": "ready",
                "expected_revision": 2,
            },
            "replacements": [
                {
                    "title": "Build the bounded change",
                    "body": "Produce one owner-visible outcome.",
                    "assignee": "default",
                    "execution_tier": "routine",
                    "parents": [],
                    **stated,
                },
                {
                    "title": "Check the bounded change",
                    "body": "Verify the outcome before downstream work continues.",
                    "assignee": "default",
                    "execution_tier": "routine",
                    "parents": [0],
                },
            ],
        },
    ]


def _plan_request_payload(changes: list[dict]) -> dict:
    """Rebuild exactly the payload ``commit_project_plan`` digests."""
    from agent.redact import redact_sensitive_text

    args = _project_plan_args(
        {"project_id": "frozen-project"}, changes,
        idempotency_key="plan-frozen-digest",
    )
    normalized_changes, risk_level = ow._normalize_project_changes(args["changes"])
    return {
        "project_id": ow._bounded_text(args["project_id"], "project_id", limit=100),
        "trigger": str(args["trigger"] or "").strip(),
        "request_title": redact_sensitive_text(
            ow._bounded_text(args["request_title"], "request_title", limit=240),
            force=True,
        ),
        "summary": redact_sensitive_text(
            ow._bounded_text(args["summary"], "summary", limit=2_000), force=True,
        ),
        "specification": redact_sensitive_text(
            ow._bounded_text(args["specification"], "specification", limit=20_000),
            force=True,
        ),
        "current_milestone": redact_sensitive_text(
            ow._bounded_text(
                args["current_milestone"], "current_milestone", limit=1_000,
            ),
            force=True,
        ),
        "owner_visible_result": redact_sensitive_text(
            ow._bounded_text(
                args["owner_visible_result"], "owner_visible_result", limit=1_000,
            ),
            force=True,
        ),
        "later_milestones": ow._normalize_later_milestones(args["later_milestones"]),
        "risk_level": risk_level,
        "changes": normalized_changes,
    }


def _states_requires_review(value) -> bool:
    if isinstance(value, dict):
        return "requires_review" in value or any(
            _states_requires_review(item) for item in value.values()
        )
    if isinstance(value, list):
        return any(_states_requires_review(item) for item in value)
    return False


def test_a_plan_that_omits_the_requirement_keeps_its_exact_request_digest():
    payload = _plan_request_payload(_fixed_changes())

    assert not _states_requires_review(payload["changes"]), (
        "a plan that says nothing about review must normalize to a change "
        "list carrying no requires_review key anywhere"
    )
    assert ow._digest(payload) == _OMITTED_REQUIREMENT_DIGEST


def test_stating_the_requirement_is_part_of_what_the_owner_approves():
    """The opposite guarantee: when stated, it IS bound into the digest."""
    omitted = _plan_request_payload(_fixed_changes())
    stated = _plan_request_payload(_fixed_changes(requires_review=True))

    assert _states_requires_review(stated["changes"])
    assert ow._digest(stated) != ow._digest(omitted)
    # And only where it was stated: the second split replacement stays silent.
    assert "requires_review" not in stated["changes"][2]["replacements"][1]
