"""The planner reads each task's approved files through planning_context_v2.

The released owner Workspace reads ``planning_context_v1``, so a request that
names only that token keeps exactly today's bounded planning context. A
request that names ``planning_context_v2`` gets the same context at schema
version 2, each task also carrying the owned paths its row stores, exactly as
``normalize_owned_paths`` returns them (``[]`` when the row stores none);
naming both gets version 2, and naming neither gets no planning context. Each
test drives the adapter's real snapshot route into the real owner-workspace
kernel inside the per-test ``HERMES_HOME``; every Project, task and path is
made up.
"""

from __future__ import annotations

import contextlib
import json
from urllib.parse import quote

import pytest
import yaml
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import PlatformConfig
from gateway.platforms.api_server import APIServerAdapter
from hermes_cli import kanban_db, owner_workspace as ow
from hermes_cli.config import get_config_path
from hermes_constants import get_hermes_home

# The tokens on the wire: the released Workspace sends the first, the planner
# the second.
_V1 = "planning_context_v1"
_V2 = "planning_context_v2"
_RUN_CONTEXT = "run_task_context"
_SNAPSHOT_ROUTE = "/v1/owner-workspace/projects/{project_slug}/snapshot"
_CONTEXT_KEYS = {
    "schema_version", "actionable_count", "omitted_terminal_count",
    "actionable_truncated", "relations_truncated", "tasks",
}
_V1_TASK_KEYS = {
    "id", "title", "status", "assignee_name", "responsibility", "updated_at",
    "event_revision", "parent_ids", "child_ids", "omitted_parent_count",
    "omitted_child_count",
}


@pytest.fixture
def project(monkeypatch) -> dict:
    """One committed placeholder Project whose tasks store every ownership form.

    ``design`` and ``build`` own paths and are related, ``review`` is
    explicitly read-only, ``legacy`` stores no ownership at all and
    ``finished`` is terminal history that still owns a path.
    """
    # gateway.run pins its config home at import; read this test's home instead.
    monkeypatch.setattr("gateway.run._hermes_home", get_hermes_home())
    path = get_config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(
        {"gateway": {"api_server": {"owner_workspace": {"enabled": True}}}}
    ), encoding="utf-8")
    owner = ow.resolve_owner_context()
    # The bootstrap confirmation is not under test.
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(ow, "_confirm", lambda *_args, **_kwargs: {"approved": True})
        created = ow.bootstrap(
            owner, idempotency_key="setup-placeholder", name="Placeholder Project",
        )
    assert created["ok"] is True
    slug = next(
        item["slug"] for item in ow.list_committed_projects(owner)
        if str(item["project_id"]) == created["project_id"]
    )

    with contextlib.closing(kanban_db.connect(board=created["board"])) as conn:
        def task(title: str, **kwargs) -> str:
            return kanban_db.create_task(
                conn, title=title, project_id=created["project_id"], **kwargs,
            )

        design = task(
            "Placeholder design step", workspace_kind="worktree",
            owned_paths=["placeholder/src", "placeholder/docs/guide.md"],
        )
        tasks = {
            "design": design,
            # Stored canonically: stripped and without the duplicate.
            "build": task(
                "Placeholder build step", workspace_kind="worktree",
                parents=[design],
                owned_paths=[" placeholder/tests ", "placeholder/tests"],
            ),
            "review": task("Placeholder review step", owned_paths=[]),
            "legacy": task("Placeholder legacy step"),
            "finished": task(
                "Placeholder finished step", workspace_kind="worktree",
                owned_paths=["placeholder/old.txt"],
            ),
        }
        with kanban_db.write_txn(conn):
            conn.execute(
                "UPDATE tasks SET status = 'done' WHERE id = ?",
                (tasks["finished"],),
            )
    return {"slug": slug, "board": created["board"], "tasks": tasks}


def _query(*values: str) -> str:
    """One ``capabilities`` parameter per value."""
    return "&".join(f"capabilities={quote(value, safe=',')}" for value in values)


async def _snapshots(slug: str, *queries: str) -> list[dict]:
    """The snapshot ``data`` the real route serves for each query, in order."""
    adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={}))
    app = web.Application()
    for method, route, handler in adapter._http_route_table():
        if method == "GET" and route == _SNAPSHOT_ROUTE:
            app.router.add_route(method, route, handler)
    path = _SNAPSHOT_ROUTE.format(project_slug=slug)
    snapshots = []
    async with TestClient(TestServer(app)) as client:
        for query in queries:
            resp = await client.get(f"{path}?{query}" if query else path)
            assert resp.status == 200, await resp.text()
            body = await resp.json()
            assert body["object"] == "hermes.owner_workspace.project_snapshot"
            snapshots.append(body["data"])
    return snapshots


def _rest(snapshot: dict) -> dict:
    """The snapshot minus its planning context and the clock it was made at."""
    rest = {key: value for key, value in snapshot.items() if key != "planning_context"}
    rest["steward"] = {
        key: value for key, value in snapshot["steward"].items()
        if key != "generated_at"
    }
    return rest


def _stored_owned_paths(board: str) -> dict[str, str | None]:
    """Each task row's owned_paths column, read straight from the board."""
    with contextlib.closing(kanban_db.connect(board=board)) as conn:
        return {
            str(row["id"]): row["owned_paths"]
            for row in conn.execute("SELECT id, owned_paths FROM tasks")
        }


@pytest.mark.asyncio
async def test_a_version_1_request_returns_exactly_the_version_1_keys(project):
    # The released Workspace sends exactly this token; version 2 is another.
    assert ow.OWNER_PROJECT_PLANNING_CONTEXT_CAPABILITY == _V1
    assert ow.OWNER_PROJECT_PLANNING_CONTEXT_V2_CAPABILITY == _V2

    contexts = [
        snapshot["planning_context"]
        for snapshot in await _snapshots(
            project["slug"],
            _query(_V1),
            _query(f"{_RUN_CONTEXT},{_V1}"),
            # Only the exact version-2 token grants version 2.
            _query(f"{_V1},{_V2}_extra"),
        )
    ]

    first = contexts[0]
    assert set(first) == _CONTEXT_KEYS
    assert first["schema_version"] == 1
    # Tasks that store owned paths are served, without them.
    assert set(project["tasks"].values()) <= {task["id"] for task in first["tasks"]}
    assert all(set(task) == _V1_TASK_KEYS for task in first["tasks"])
    assert contexts == [first] * len(contexts)


@pytest.mark.asyncio
async def test_a_version_2_request_adds_each_task_stored_owned_paths(project):
    plain, v1, v2, both, both_reversed = await _snapshots(
        project["slug"],
        "",
        _query(_V1),
        _query(_V2),
        _query(f"{_V1},{_V2}"),
        _query(f"{_V2},{_V1}"),
    )

    assert "planning_context" in v2
    context, v1_context = v2["planning_context"], v1["planning_context"]
    assert set(context) == _CONTEXT_KEYS
    assert context["schema_version"] == 2
    assert all(set(task) == _V1_TASK_KEYS | {"owned_paths"} for task in context["tasks"])

    # Each task's owned paths are its row's stored list as the kernel reads it.
    stored = _stored_owned_paths(project["board"])
    tasks = project["tasks"]
    assert stored[tasks["legacy"]] is None
    assert json.loads(stored[tasks["review"]]) == []
    served = {task["id"]: task["owned_paths"] for task in context["tasks"]}
    assert served == {
        task_id: (
            [] if stored[task_id] is None
            else kanban_db.normalize_owned_paths(json.loads(stored[task_id]))
        )
        for task_id in served
    }
    assert served[tasks["design"]] == ["placeholder/src", "placeholder/docs/guide.md"]
    assert served[tasks["build"]] == ["placeholder/tests"]
    assert served[tasks["review"]] == []
    assert served[tasks["legacy"]] == []
    assert served[tasks["finished"]] == ["placeholder/old.txt"]

    # Otherwise version 1: the same counts, tasks, order and relations.
    assert {key: context[key] for key in _CONTEXT_KEYS - {"schema_version", "tasks"}} == {
        key: v1_context[key] for key in _CONTEXT_KEYS - {"schema_version", "tasks"}
    }
    assert [
        {key: value for key, value in task.items() if key != "owned_paths"}
        for task in context["tasks"]
    ] == v1_context["tasks"]
    by_id = {task["id"]: task for task in context["tasks"]}
    assert by_id[tasks["build"]]["parent_ids"] == [tasks["design"]]
    assert by_id[tasks["design"]]["child_ids"] == [tasks["build"]]

    # Asking for both versions, in either order, gets version 2.
    assert both["planning_context"] == context
    assert both_reversed["planning_context"] == context
    # Nothing else in the snapshot changes.
    for snapshot in (v1, v2, both, both_reversed):
        assert _rest(snapshot) == _rest(plain)


@pytest.mark.asyncio
async def test_a_request_with_neither_capability_gets_no_planning_context(project):
    # Built from the kernel's own version-2 token, so a near miss of the real
    # token is what is refused.
    v2 = ow.OWNER_PROJECT_PLANNING_CONTEXT_V2_CAPABILITY
    plain, run_context, *refused = await _snapshots(
        project["slug"],
        "",
        _query(_RUN_CONTEXT),
        _query(""),
        _query(f"{v2}_extra"),
        # A repeated parameter has no single negotiated answer.
        _query(v2, v2),
        _query(_V1, v2),
        _query("x" * 300 + f",{v2}"),
    )

    assert "planning_context" not in plain
    assert "planning_context" not in run_context
    for snapshot in refused:
        assert "planning_context" not in snapshot
        assert _rest(snapshot) == _rest(plain)
