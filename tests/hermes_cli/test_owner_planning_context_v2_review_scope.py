"""A card parked for review shows the planner the scope its handover recorded.

Handing a card over for review stores its ``owned_paths`` as ``[]``, so the
review run is read-only, and records the scope it replaces on the handover's
own ``review_requested`` event, from which the handback, a reopen and the
approval restore it exactly. ``planning_context_v2`` serves each task the
owned paths its row stores, so a parked card reached the planner as ``[]`` and
looked read-only. Version 2 now serves a card on the review lane whose row
stores ``[]`` the scope its latest handover recorded, and ``[]`` when that
handover recorded none. The row, the reviewer's read-only scope, version 1 and
every other task stay exactly as they were.

The parked card is a real scoped git worktree card with a real commit inside
its boundary, handed over through the kernel's one handover primitive as in
``test_kanban_review_read_only_scope``. Each test drives the adapter's real
snapshot route of ``test_owner_planning_context_v2`` into the real
owner-workspace kernel inside the per-test ``HERMES_HOME``; every Project,
task and path is made up.
"""

from __future__ import annotations

import contextlib
import json
from pathlib import Path

import pytest

from hermes_cli import kanban_db

# The scoped card and its handover, reused from the read-only scope test.
from tests.hermes_cli.test_kanban_review_read_only_scope import (  # noqa: F401
    IMPLEMENTATION_SCOPE,
    IMPLEMENTER,
    REVIEWER,
    _git,
    _nominate,
    repo,
)

# The snapshot-route harness and its placeholder Project, reused verbatim.
from tests.hermes_cli.test_owner_planning_context_v2 import (  # noqa: F401
    _CONTEXT_KEYS,
    _V1,
    _V1_TASK_KEYS,
    _V2,
    _query,
    _snapshots,
    _stored_owned_paths,
    project,
)


def _project_id(conn, project: dict) -> str:
    return kanban_db.get_task(conn, project["tasks"]["design"]).project_id


def _hand_over_scoped_card(conn, repo: Path, project_id: str) -> str:
    """A claimed scoped worktree card of the Project, parked for review.

    A per-task git worktree, a recorded base and one commit inside
    ``IMPLEMENTATION_SCOPE``, then the kernel's handover parks it read-only.
    """
    tid = kanban_db.create_task(
        conn,
        title="Placeholder scoped step",
        assignee=IMPLEMENTER,
        requires_review=True,
        owned_paths=IMPLEMENTATION_SCOPE,
        workspace_kind="worktree",
        workspace_path=str(repo),
        branch_name="feature/scoped",
        project_id=project_id,
    )
    run = kanban_db.claim_task(conn, tid, claimer=f"{IMPLEMENTER}:1")
    assert run is not None
    workspace, branch = kanban_db._resolve_worktree_workspace(run)
    kanban_db.set_workspace_path(conn, tid, workspace)
    kanban_db.set_branch_name(conn, tid, branch)
    kanban_db.record_worktree_base(conn, tid, workspace)
    (workspace / "src").mkdir(parents=True, exist_ok=True)
    (workspace / "src" / "app.py").write_text("value = 1\n", encoding="utf-8")
    _git(workspace, "add", "src/app.py")
    _git(workspace, "commit", "-m", "feat: implement inside the boundary")
    assert kanban_db.request_review(
        conn, tid,
        summary="ready for review",
        reviewer=REVIEWER,
        expected_run_id=run.current_run_id,
    ) is True
    return tid


@pytest.fixture
def parked(project, repo, monkeypatch) -> dict:
    """The placeholder Project plus one real scoped card parked for review."""
    _nominate(monkeypatch, REVIEWER)
    with contextlib.closing(kanban_db.connect(board=project["board"])) as conn:
        tid = _hand_over_scoped_card(conn, repo, _project_id(conn, project))
        parked_task = kanban_db.get_task(conn, tid)
        assert (parked_task.status, parked_task.owned_paths) == ("review", [])
    return {**project, "parked": tid}


def _served(snapshot: dict) -> dict[str, dict]:
    """The planning context's tasks by id."""
    return {task["id"]: task for task in snapshot["planning_context"]["tasks"]}


@pytest.mark.asyncio
async def test_a_parked_review_card_is_served_with_its_handover_scope(parked):
    tid = parked["parked"]
    [v2] = await _snapshots(parked["slug"], _query(_V2))

    card = _served(v2)[tid]
    assert card["status"] == "review"
    # The row is read-only for the review run, but the planner is shown the
    # implementation scope the kernel recorded on the handover.
    assert json.loads(_stored_owned_paths(parked["board"])[tid]) == []
    with contextlib.closing(kanban_db.connect(board=parked["board"])) as conn:
        recorded = kanban_db._latest_review_scope_provenance(conn, tid)
    assert recorded["owned_paths"] == IMPLEMENTATION_SCOPE
    assert card["owned_paths"] == IMPLEMENTATION_SCOPE


@pytest.mark.asyncio
async def test_the_parked_row_itself_stays_read_only(parked):
    tid = parked["parked"]
    await _snapshots(parked["slug"], _query(_V2), _query(f"{_V1},{_V2}"))

    # Serving the planner is a read: the row stays read-only.
    assert json.loads(_stored_owned_paths(parked["board"])[tid]) == []
    with contextlib.closing(kanban_db.connect(board=parked["board"])) as conn:
        assert kanban_db.get_task(conn, tid).owned_paths == []
        review = kanban_db.claim_review_task(conn, tid, claimer=f"{REVIEWER}:1")
        assert review is not None
        reviewing = kanban_db.get_task(conn, tid)
        assert reviewing.owned_paths == []
        assert reviewing.integrates_parent_heads is False
        # The reviewer still cannot write any path the implementer could.
        for path in IMPLEMENTATION_SCOPE + ["src/app.py", "tests/app/test_app.py"]:
            assert kanban_db._path_is_owned(path, reviewing.owned_paths) is False, path
        assert kanban_db.run_requires_read_only_workspace(
            conn, tid, int(reviewing.current_run_id),
            owned_paths=reviewing.owned_paths,
        ) is True


@pytest.mark.asyncio
async def test_version_1_and_other_tasks_are_unchanged(parked):
    v1, v2 = await _snapshots(parked["slug"], _query(_V1), _query(_V2))
    v1_context, context = v1["planning_context"], v2["planning_context"]

    # Version 1 keeps exactly its keys, the parked card included.
    assert set(v1_context) == _CONTEXT_KEYS
    assert v1_context["schema_version"] == 1
    assert parked["parked"] in {task["id"] for task in v1_context["tasks"]}
    assert all(set(task) == _V1_TASK_KEYS for task in v1_context["tasks"])
    # Version 2 is version 1 plus each task's owned paths.
    assert {key: context[key] for key in _CONTEXT_KEYS - {"schema_version", "tasks"}} == {
        key: v1_context[key] for key in _CONTEXT_KEYS - {"schema_version", "tasks"}
    }
    assert [
        {key: value for key, value in task.items() if key != "owned_paths"}
        for task in context["tasks"]
    ] == v1_context["tasks"]

    # Every other task is still served exactly the owned paths its row stores.
    stored = _stored_owned_paths(parked["board"])
    others = {
        task["id"]: task["owned_paths"]
        for task in context["tasks"] if task["id"] != parked["parked"]
    }
    tasks = parked["tasks"]
    assert set(tasks.values()) <= set(others)
    assert others == {
        task_id: (
            [] if stored[task_id] is None
            else kanban_db.normalize_owned_paths(json.loads(stored[task_id]))
        )
        for task_id in others
    }
    assert others[tasks["design"]] == ["placeholder/src", "placeholder/docs/guide.md"]
    assert others[tasks["build"]] == ["placeholder/tests"]
    assert others[tasks["review"]] == []
    assert others[tasks["legacy"]] == []
    assert others[tasks["finished"]] == ["placeholder/old.txt"]


@pytest.mark.asyncio
async def test_a_review_card_whose_handover_recorded_no_scope_stays_empty(project):
    with contextlib.closing(kanban_db.connect(board=project["board"])) as conn:
        project_id = _project_id(conn, project)
        # Parked by a build that predates the scope record: the row stores
        # [] and the newest review_requested event names no scope at all.
        unrecorded = kanban_db.create_task(
            conn, title="Placeholder unrecorded step", assignee=IMPLEMENTER,
            owned_paths=[], project_id=project_id,
        )
        with kanban_db.write_txn(conn):
            conn.execute(
                "UPDATE tasks SET status = 'review', assignee = ? WHERE id = ?",
                (REVIEWER, unrecorded),
            )
            kanban_db._append_event(conn, unrecorded, "review_requested", {
                "summary": "ready for review",
                "implementer": IMPLEMENTER,
                "reviewer": REVIEWER,
            })
        assert kanban_db._latest_review_scope_provenance(conn, unrecorded) is None
        # A card with no scope of its own, handed over today: the handover
        # records the legacy unscoped None, which names no paths either.
        unscoped = kanban_db.create_task(
            conn, title="Placeholder unscoped step", assignee=IMPLEMENTER,
            project_id=project_id,
        )
        assert kanban_db.request_review(
            conn, unscoped, summary="ready for review", reviewer=REVIEWER,
        ) is True
        assert kanban_db._latest_review_scope_provenance(conn, unscoped) == {
            "owned_paths": None, "integrates_parent_heads": False,
        }

    [v2] = await _snapshots(project["slug"], _query(_V2))

    served = _served(v2)
    stored = _stored_owned_paths(project["board"])
    for tid in (unrecorded, unscoped):
        assert served[tid]["status"] == "review"
        assert json.loads(stored[tid]) == []
        assert served[tid]["owned_paths"] == []


@pytest.mark.asyncio
async def test_after_handback_the_row_scope_is_served(parked):
    tid = parked["parked"]
    with contextlib.closing(kanban_db.connect(board=parked["board"])) as conn:
        review = kanban_db.claim_review_task(conn, tid, claimer=f"{REVIEWER}:1")
        assert review is not None
        ok, implementer = kanban_db.request_changes(
            conn, tid,
            reason="the loop bound is off by one",
            expected_run_id=review.current_run_id,
        )
        assert (ok, implementer) == (True, IMPLEMENTER)

    [v2] = await _snapshots(parked["slug"], _query(_V2))

    card = _served(v2)[tid]
    assert card["status"] != "review"
    # The handback restored the row, and the row is what is served.
    stored = _stored_owned_paths(parked["board"])[tid]
    assert kanban_db.normalize_owned_paths(json.loads(stored)) == IMPLEMENTATION_SCOPE
    assert card["owned_paths"] == IMPLEMENTATION_SCOPE
