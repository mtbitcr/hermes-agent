"""The Project list route answers ``board_counts`` only to a reader that names ``board_counts_v1`` once."""

from __future__ import annotations

import json
import threading
import time
from unittest.mock import patch

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import PlatformConfig
from gateway.platforms.api_server import APIServerAdapter
from hermes_cli import kanban_db, owner_workspace as ow, projects_db
from tools import approval

_LIST = "/v1/owner-workspace/projects"
_CONFIG = {"gateway": {"api_server": {"owner_workspace": {"enabled": True}}}}


@pytest.fixture
def ctx():
    return ow.OwnerContext(actor="default", profile="default", session="run_owt")


def _app() -> web.Application:
    adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={}))
    app = web.Application()
    for method, path, handler in adapter._http_route_table():
        if method == "GET" and path == _LIST:
            app.router.add_route(method, path, handler)
    return app


def _auto_approve(session_key: str, timeout: float = 10.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        with approval._lock:
            queue = approval._gateway_queues.get(session_key)
            aid = queue[0].approval_id if queue else None
        if aid:
            approval.resolve_gateway_approval(session_key, "once", approval_id=aid)
            return
        time.sleep(0.005)
    raise TimeoutError("no approval was queued")


def _new_projects(ctx, count: int) -> list[dict]:
    projects = []
    approval.register_gateway_notify(ctx.session, lambda data: None)
    for index in range(count):
        approver = threading.Thread(target=_auto_approve, args=(ctx.session,))
        approver.start()
        result = ow.bootstrap(ctx, idempotency_key=f"key-{index}", name=f"Project {index}")
        approver.join()
        assert result["ok"] is True
        with projects_db.connect_closing() as pconn:
            slug = projects_db.get_project(pconn, result["project_id"]).slug
        projects.append({**result, "slug": slug})
    return projects


def _add_tasks(project, statuses) -> None:
    with kanban_db.connect(board=project["board"]) as conn:
        for index, status in enumerate(statuses):
            task_id = kanban_db.create_task(
                conn, title=f"{status} work {index}", project_id=project["project_id"],
            )
            with kanban_db.write_txn(conn):
                conn.execute("UPDATE tasks SET status = ? WHERE id = ?", (status, task_id))


async def _get(ctx, query: str = "") -> tuple[int, bytes]:
    with (
        patch("gateway.run._load_gateway_config", return_value=_CONFIG),
        patch("hermes_cli.owner_workspace.resolve_owner_context", return_value=ctx),
    ):
        async with TestClient(TestServer(_app())) as client:
            resp = await client.get(f"{_LIST}{query}")
            return resp.status, await resp.read()


@pytest.mark.asyncio
async def test_board_counts_equal_the_snapshots_counts_and_total(ctx):
    project = _new_projects(ctx, 1)[0]
    _add_tasks(project, ["triage", "ready", "running", "blocked", "review", "done", "done", "archived"])

    status, body = await _get(ctx, "?capabilities=board_counts_v1")
    snapshot = ow.read_project_snapshot(ctx, project["slug"])

    assert status == 200
    [item] = json.loads(body)["data"]
    assert item["board_counts"] == {
        "counts": snapshot["board"]["counts"], "total": snapshot["board"]["total"],
    }
    assert item["board_counts"]["counts"]["archived"] == 1


@pytest.mark.asyncio
async def test_the_token_adds_exactly_one_key_and_changes_nothing_else(ctx):
    projects = _new_projects(ctx, 3)
    _add_tasks(projects[1], ["ready", "done"])

    _, plain = await _get(ctx)
    _, counted = await _get(ctx, "?capabilities=board_counts_v1")
    _, both = await _get(ctx, "?capabilities=removal_state,board_counts_v1")

    plain_data = json.loads(plain)["data"]
    counted_data = json.loads(counted)["data"]
    assert [item["project_id"] for item in counted_data] == [p["project_id"] for p in projects]
    assert all(set(item) - set(old) == {"board_counts"} for item, old in zip(counted_data, plain_data))
    assert [{k: v for k, v in item.items() if k != "board_counts"} for item in counted_data] == plain_data
    both_data = json.loads(both)["data"]
    assert all("removal_state" in item and "board_counts" in item for item in both_data)
    assert [item["board_counts"] for item in both_data] == [item["board_counts"] for item in counted_data]


@pytest.mark.asyncio
@pytest.mark.parametrize("query", [
    "",
    "?capabilities=board_counts_v2",
    "?capabilities=board_counts",
    "?capabilities=BOARD_COUNTS_V1",
    "?capabilities=board_counts_v1&capabilities=board_counts_v1",
    "?capabilities=removal_state&capabilities=board_counts_v1",
    "?capabilities=board_counts_v1&capabilities=removal_state",
])
async def test_anything_but_one_exact_token_returns_todays_body(ctx, query):
    _new_projects(ctx, 2)
    _, baseline = await _get(ctx)

    assert await _get(ctx, query) == (200, baseline)
    assert all("board_counts" not in item for item in json.loads(baseline)["data"])


@pytest.mark.asyncio
async def test_listing_eighteen_projects_never_reads_a_snapshot(ctx):
    projects = _new_projects(ctx, 18)
    for index, project in enumerate(projects):
        _add_tasks(project, ["ready"] * index)

    def fail(*args, **kwargs):
        raise AssertionError("read_project_snapshot must not run for the list")

    with patch("hermes_cli.owner_workspace.read_project_snapshot", side_effect=fail):
        status, body = await _get(ctx, "?capabilities=board_counts_v1")

    data = json.loads(body)["data"]
    assert status == 200
    assert len(data) == 18
    for index, item in enumerate(data):
        assert item["board_counts"]["counts"]["ready"] == index
        assert item["board_counts"]["total"] == index


@pytest.mark.asyncio
async def test_a_missing_board_store_is_null_and_the_other_seventeen_keep_their_counts(ctx):
    projects = _new_projects(ctx, 18)
    for index, project in enumerate(projects):
        _add_tasks(project, ["done"] * index)
    kanban_db.kanban_db_path(projects[5]["board"]).unlink()

    status, body = await _get(ctx, "?capabilities=board_counts_v1")

    data = json.loads(body)["data"]
    assert status == 200
    assert len(data) == 18
    for index, item in enumerate(data):
        if index == 5:
            assert item["board_counts"] is None
        else:
            assert item["board_counts"]["counts"]["done"] == index
            assert item["board_counts"]["total"] == index
