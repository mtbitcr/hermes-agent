"""Owner snapshot examples (contract map section 5, test 1).

Every example comes from the adapter's real ``GET
/v1/owner-workspace/projects/{slug}/snapshot`` route over
``read_project_snapshot``, on real boards in the per-test home, except the run
receipts, which come from ``_owner_project_run_projection``, the helper the
snapshot builds each run with. What is asserted is described in
tests/contracts/conftest.py.
"""

from __future__ import annotations

import contextlib
from unittest.mock import patch

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import PlatformConfig
from gateway.platforms.api_server import APIServerAdapter
from hermes_cli import kanban_db, owner_workspace as ow
from hermes_constants import get_hermes_home
from tests.gateway.test_api_server_owner_suggestion_decisions import (  # noqa: F401  (owner is a fixture)
    _project,
    _write_owner_workspace_config,
    owner,
)
from tests.gateway.test_api_server_runs import _capability_stopped_task
from tests.hermes_cli.test_owner_project_removal import _real_board, _start_and_join
from tests.hermes_cli.test_owner_provider_wait import (  # noqa: F401  (root and clock are fixtures)
    STEWARD,
    _waiting_and_refused,
    clock,
    root,
)
from tests.hermes_cli.test_owner_workspace import _ended_run, _gave_up_task

FAMILY = "owner_snapshot"
V1 = ",".join((
    ow.OWNER_PROJECT_RUN_CONTEXT_CAPABILITY,
    ow.OWNER_PROJECT_PLANNING_CONTEXT_CAPABILITY,
    ow.OWNER_WAITING_CAPABILITY,
    ow.PROVIDER_WAIT_CAPABILITY,
))
V2 = f"{V1},{ow.OWNER_PROJECT_PLANNING_CONTEXT_V2_CAPABILITY}"
STEWARD_STATES = {
    "working", "waiting_for_approval", "waiting_for_you", "paused",
    "needs_attention", "complete", "deleted",
}
RECEIPT_OUTCOMES = {"running", "completed", "attention", "waiting", "unknown"}


def _app() -> web.Application:
    """The adapter's own Project snapshot route, registered as ``connect()`` does."""
    adapter = APIServerAdapter(PlatformConfig(enabled=True))
    app = web.Application()
    for method, path, handler in adapter._http_route_table():
        if path.endswith("/snapshot"):
            app.router.add_route(method, path, handler)
    return app


async def _snapshot(client: TestClient, slug: str, capabilities: str = "") -> dict:
    params = {"capabilities": capabilities} if capabilities else None
    resp = await client.get(
        f"/v1/owner-workspace/projects/{slug}/snapshot", params=params,
    )
    return {"status": resp.status, "body": await resp.json()}


def _slug(owner: ow.OwnerContext, project: dict) -> str:
    return next(
        item["slug"] for item in ow.list_committed_projects(owner)
        if str(item["project_id"]) == project["project_id"]
    )


def _task(board: str, project: dict, title: str) -> str:
    with contextlib.closing(kanban_db.connect(board=board)) as conn:
        return kanban_db.create_task(
            conn, title=title, assignee="default", project_id=project["project_id"],
        )


def _tasks(example: dict) -> list[dict]:
    return [
        task for column in example["body"]["data"]["columns"] for task in column["tasks"]
    ]


@pytest.mark.asyncio
async def test_the_snapshot_examples_carry_every_task_state_and_capability(
    owner, owner_payload_example,
):
    project = _project(owner, "Workshop Pilot")
    board, slug = project["board"], _slug(owner, project)
    approved = _task(board, project, "Review the approved outline")
    with contextlib.closing(kanban_db.connect(board=board)) as conn:
        assert kanban_db.request_review(conn, approved, summary="Ready.")
        # A completion promotes a given-up card back to ready, so it goes first.
        assert kanban_db.complete_task(conn, approved, result="Approved.")
    # The breaker trips on every ready or review card, so this comes next.
    _gave_up_task(board, project["project_id"], "Print the workshop handouts")
    _capability_stopped_task(board, project["project_id"], "Connect the payment provider")
    _task(board, project, "Draft the workshop agenda")
    reviews = {
        state: _task(board, project, f"Review the {state} outline")
        for state in ("awaiting", "changes")
    }
    with contextlib.closing(kanban_db.connect(board=board)) as conn:
        for task_id in reviews.values():
            assert kanban_db.request_review(conn, task_id, summary="Ready.")
        assert kanban_db.claim_review_task(conn, reviews["changes"]) is not None
        assert kanban_db.request_changes(
            conn, reviews["changes"], reason="Name the date first.",
        )[0]

    async with TestClient(TestServer(_app())) as client:
        live = {
            "project_snapshot": await _snapshot(client, slug, V1),
            "project_snapshot_planning_context_v2": await _snapshot(client, slug, V2),
            "project_snapshot_without_capabilities": await _snapshot(client, slug),
            "project_not_found": await _snapshot(client, "no-such-project"),
        }
        unavailable = ow.OwnerWorkspaceError("snapshot_unavailable", "unreadable")
        with patch.object(ow, "read_project_snapshot", side_effect=unavailable):
            live["owner_workspace_unavailable"] = await _snapshot(client, slug)
        with patch.object(ow, "read_project_snapshot", side_effect=RuntimeError("x")):
            live["owner_workspace_unavailable_unexpected"] = await _snapshot(client, slug)
        _write_owner_workspace_config(enabled=False)
        live["owner_workspace_not_enabled"] = await _snapshot(client, slug)

    saved = {kind: owner_payload_example(FAMILY, kind, body) for kind, body in live.items()}

    tasks = _tasks(saved["project_snapshot"])
    assert {task["review_state"] for task in tasks} == {
        kanban_db.REVIEW_STATE_AWAITING_REVIEW, kanban_db.REVIEW_STATE_CHANGES_REQUESTED,
        kanban_db.REVIEW_STATE_APPROVED, kanban_db.REVIEW_STATE_NONE,
    }
    assert {
        kanban_db.STOPPED_WORK_GAVE_UP, kanban_db.STOPPED_WORK_CAPABILITY,
        kanban_db.STOPPED_WORK_NONE,
    } <= {task["stopped_work"] for task in tasks}
    assert any("owner_wait" in task for task in tasks)
    assert not any("owner_wait" in task for task in _tasks(
        saved["project_snapshot_without_capabilities"]))
    for kind in ("project_snapshot", "project_snapshot_planning_context_v2"):
        assert "planning_context" in saved[kind]["body"]["data"]
    assert "planning_context" not in (
        saved["project_snapshot_without_capabilities"]["body"]["data"])
    assert set(saved["project_snapshot"]["body"]["data"]["truncated"]) == {
        "tasks", "workers", "attachments", "runs",
    }
    errors = {
        kind: example for kind, example in saved.items() if "error" in example["body"]
    }
    assert {example["body"]["error"]["code"] for example in errors.values()} == {
        "project_not_found", "owner_workspace_unavailable", "owner_workspace_not_enabled",
    }
    assert all(example["status"] >= 400 for example in errors.values())


@pytest.mark.asyncio
async def test_the_steward_examples_carry_every_steward_state(
    owner, owner_payload_example, monkeypatch,
):
    projects = {name: _project(owner, f"{name.title()} Pilot") for name in (
        "fresh", "working", "complete", "paused", "attention", "waiting",
    )}
    for name in ("working", "attention", "waiting"):
        kanban_db.write_board_dispatch_state(projects[name]["board"], dispatch_enabled=True)
    _gave_up_task(projects["attention"]["board"], projects["attention"]["project_id"],
                  "Print the handouts")
    for name in ("working", "complete", "paused"):
        _task(projects[name]["board"], projects[name], "Draft the agenda")
    with contextlib.closing(kanban_db.connect(board=projects["complete"]["board"])) as conn:
        [done] = [task.id for task in kanban_db.list_tasks(conn)
                  if task.project_id == projects["complete"]["project_id"]]
        assert kanban_db.complete_task(conn, done, result="Done.")
    kanban_db.write_board_metadata(
        projects["paused"]["board"], dispatch_enabled=False, dispatch_paused_by_owner=True,
    )
    _capability_stopped_task(projects["waiting"]["board"],
                             projects["waiting"]["project_id"], "Connect the provider")
    project_id, deleted_slug = _real_board(monkeypatch)
    _start_and_join(project_id, "contract")

    async with TestClient(TestServer(_app())) as client:
        live = [
            (await _snapshot(client, _slug(owner, project), V1))["body"]["data"]["steward"]
            for project in projects.values()
        ]
        deleted = await _snapshot(client, deleted_slug, V1)

    saved = owner_payload_example(FAMILY, "steward_states", live)
    saved_deleted = owner_payload_example(FAMILY, "deleted_project", deleted)
    states = [steward["execution"]["state"] for steward in saved]
    states.append(saved_deleted["body"]["data"]["steward"]["execution"]["state"])
    assert set(states) == STEWARD_STATES
    assert "removal_state" in saved_deleted["body"]["data"]


@pytest.mark.asyncio
async def test_the_provider_wait_example_carries_the_held_card_and_its_waiting_run(
    root, clock, monkeypatch, owner_payload_example,
):
    monkeypatch.setenv("HERMES_HOME", str(root / "profiles" / STEWARD))
    scene = _waiting_and_refused(root, ow.resolve_owner_context())
    clock["t"] = scene["held"].ended_at + 60
    monkeypatch.setattr("gateway.run._hermes_home", get_hermes_home())
    _write_owner_workspace_config(enabled=True)

    async with TestClient(TestServer(_app())) as client:
        live = await _snapshot(client, scene["project"]["slug"], V1)

    saved = owner_payload_example(FAMILY, "provider_wait", live)
    assert ow._OWNER_STOPPED_WORK_PROVIDER_WAIT in {
        task["stopped_work"] for task in _tasks(saved)
    }
    assert "waiting" in {run["receipt"]["outcome"] for run in saved["body"]["data"]["runs"]}


def test_the_run_receipt_examples_carry_every_outcome_and_summary(owner_payload_example):
    resume_at = 1_790_179_200
    cases = [
        (_ended_run("running", None, ended_at=None), {}),
        (_ended_run("completed", None), {"owner_retry_reason": "The key was added."}),
        (_ended_run("review_requested", None), {}),
        (_ended_run("scheduled", None), {}),
        (_ended_run("rate_limited", None),
         {"provider_wait": True, "provider_resume_at": resume_at}),
        (_ended_run("rate_limited", None), {}),
        (_ended_run("rate_limited", {"rate_limit_reset_at": resume_at}), {}),
        (_ended_run("provider_refused", None), {"provider_wait": True}),
        (_ended_run("crashed", None), {}),
        (_ended_run("changes_requested", None), {"owner_waiting": True}),
        (_ended_run("blocked", None), {"owner_waiting": True}),
        (_ended_run("cancelled", None), {}),
    ]
    origins = [
        kanban_db.RETRY_ORIGIN_NONE, kanban_db.RETRY_ORIGIN_AUTOMATIC,
        kanban_db.RETRY_ORIGIN_OWNER, kanban_db.RETRY_ORIGIN_UNATTRIBUTED,
    ]
    live = [
        ow._owner_project_run_projection(
            run, "Draft the workshop agenda", task_pin=None,
            has_newer_run=origins[index % 4] != kanban_db.RETRY_ORIGIN_NONE,
            retry_origin=origins[index % 4], run_context=True, **options,
        )
        for index, (run, options) in enumerate(cases)
    ]

    saved = owner_payload_example(FAMILY, "run_receipts", live)
    assert {run["receipt"]["outcome"] for run in saved} == RECEIPT_OUTCOMES
    assert {run["receipt"]["summary"] for run in saved} == {
        run["receipt"]["summary"] for run in live
    }
    assert {run["retry_origin"] for run in saved} == set(origins)
