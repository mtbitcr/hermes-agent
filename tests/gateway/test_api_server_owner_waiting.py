"""The owner_waiting_v1 capability is negotiated the existing way.

The snapshot and Decisions routes serve the truthful waiting answers only to a
reader that names ``owner_waiting_v1`` in ONE ``capabilities`` parameter; the
token combines with the capabilities that already exist, and a repeated
parameter or an unknown token returns today's shape. Each test drives the
adapter's real routes into the real owner-workspace kernel inside the per-test
``HERMES_HOME``.
"""

from __future__ import annotations

import contextlib
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

_REASON = "Please connect the placeholder calendar account."
_QUESTION = "Which placeholder venue should we book?"
_DECISION_FALLBACK = "Raphael needs your answer before this work can continue."


def _write_owner_workspace_config(*, enabled: bool) -> None:
    path = get_config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    config = {"gateway": {"api_server": {"owner_workspace": {"enabled": enabled}}}}
    path.write_text(yaml.safe_dump(config), encoding="utf-8")


@pytest.fixture
def owner(monkeypatch) -> ow.OwnerContext:
    """The owner the routes resolve for themselves, with the workspace enabled."""
    # gateway.run pins its config home at import; read this test's home instead.
    monkeypatch.setattr("gateway.run._hermes_home", get_hermes_home())
    _write_owner_workspace_config(enabled=True)
    return ow.resolve_owner_context()


def _project(owner: ow.OwnerContext, name: str) -> dict:
    """Commit one owner Project; its bootstrap confirmation is not under test."""
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(ow, "_confirm", lambda *_args, **_kwargs: {"approved": True})
        project = ow.bootstrap(owner, idempotency_key=f"setup-{name}", name=name)
    assert project["ok"] is True
    kanban_db.write_board_dispatch_state(project["board"], dispatch_enabled=True)
    project["slug"] = next(
        item["slug"] for item in ow.list_committed_projects(owner)
        if str(item["project_id"]) == project["project_id"]
    )
    return project


def _stopped(conn, project: dict, title: str, *, kind: str, reason: str) -> str:
    task_id = kanban_db.create_task(
        conn, title=title, assignee="raphael-worker",
        project_id=project["project_id"],
    )
    assert kanban_db.claim_task(conn, task_id) is not None
    assert kanban_db.block_task(conn, task_id, reason=reason, kind=kind) is True
    return task_id


def _app() -> web.Application:
    """The adapter's own snapshot and Decisions routes, as ``connect()`` adds them."""
    adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={}))
    app = web.Application()
    for method, path, handler in adapter._http_route_table():
        if method == "GET" and (
            path == "/v1/owner-workspace/decisions"
            or path == "/v1/owner-workspace/projects/{project_slug}/snapshot"
        ):
            app.router.add_route(method, path, handler)
    return app


def _query(*values: str) -> str:
    return "&".join(f"capabilities={quote(value, safe=',')}" for value in values)


async def _get(client: TestClient, path: str, *values: str) -> dict:
    """One GET's whole JSON body; ``values`` are the capabilities parameters."""
    url = f"{path}?{_query(*values)}" if values else path
    resp = await client.get(url)
    assert resp.status == 200, await resp.text()
    return await resp.json()


def _without_clock(snapshot: dict) -> dict:
    """The snapshot minus the one field that records when it was generated."""
    steward = dict(snapshot["steward"])
    steward.pop("generated_at")
    return {**snapshot, "steward": steward}


def _has_waiting_keys(snapshot: dict) -> bool:
    return (
        any(
            "owner_wait" in task
            for column in snapshot["columns"] for task in column["tasks"]
        )
        or any("stop_reason" in run["receipt"] for run in snapshot["runs"])
        or any(
            {"reason", "waiting_since"} & set(item)
            for item in snapshot["steward"]["decisions_needed"]
        )
    )


@pytest.mark.asyncio
async def test_the_token_works_only_in_one_capabilities_parameter(owner):
    token = ow.OWNER_WAITING_CAPABILITY
    assert token == "owner_waiting_v1"
    project = _project(owner, "Gateway Pilot")
    with contextlib.closing(kanban_db.connect(board=project["board"])) as conn:
        _stopped(
            conn, project, "Book the placeholder room", kind="capability",
            reason=_REASON,
        )
        _stopped(
            conn, project, "Choose the placeholder venue", kind="needs_input",
            reason=_QUESTION,
        )
    snapshot_path = f"/v1/owner-workspace/projects/{project['slug']}/snapshot"
    decisions_path = "/v1/owner-workspace/decisions"

    async with TestClient(TestServer(_app())) as client:
        today_snapshot = (await _get(client, snapshot_path))["data"]
        today_decisions = await _get(client, decisions_path)
        waiting = (await _get(client, snapshot_path, token))["data"]
        combined = (await _get(
            client, snapshot_path,
            f"{ow.OWNER_PROJECT_RUN_CONTEXT_CAPABILITY}, {token}",
        ))["data"]
        waiting_decisions = await _get(client, decisions_path, token)
        refused_snapshots = [
            (await _get(client, snapshot_path, *values))["data"]
            for values in (
                (token, token),
                (ow.OWNER_PROJECT_RUN_CONTEXT_CAPABILITY, token),
                ("owner_waiting_v2",),
                ("",),
            )
        ]
        refused_decisions = [
            await _get(client, decisions_path, *values)
            for values in (
                (token, token),
                ("owner_waiting_v2",),
                (ow.OWNER_PROJECT_RUN_CONTEXT_CAPABILITY,),
            )
        ]

    # Today's shape: no new key anywhere, today's Decisions selection.
    assert not _has_waiting_keys(today_snapshot)
    assert [
        (item["title"], item["reason"]) for item in today_decisions["data"]
    ] == [("Choose the placeholder venue", _DECISION_FALLBACK)]

    # Granted on the snapshot, alone or with an existing capability.
    for granted in (waiting, combined):
        needed = {
            item["title"]: item for item in granted["steward"]["decisions_needed"]
        }
        assert set(needed) == {
            "Book the placeholder room", "Choose the placeholder venue",
        }
        assert needed["Book the placeholder room"]["reason"] == _REASON
        assert needed["Choose the placeholder venue"]["reason"] == _QUESTION
        assert all(item["waiting_since"] for item in needed.values())
        assert granted["steward"]["execution"]["state"] == "waiting_for_you"
        owner_waits = {
            task["title"]: task["owner_wait"]
            for column in granted["columns"] for task in column["tasks"]
            if "owner_wait" in task
        }
        assert {title: wait["reason"] for title, wait in owner_waits.items()} == {
            "Book the placeholder room": _REASON,
            "Choose the placeholder venue": _QUESTION,
        }
        assert sorted(
            run["receipt"]["stop_reason"] for run in granted["runs"]
            if "stop_reason" in run["receipt"]
        ) == sorted([_REASON, _QUESTION])
    # The existing capability still grants its own keys alongside.
    assert all("task_title" not in run for run in waiting["runs"])
    assert all("task_title" in run for run in combined["runs"])

    # Granted on Decisions: the capability stop joins, with its own reason.
    assert {
        (item["title"], item["kind"], item["reason"])
        for item in waiting_decisions["data"]
    } == {
        ("Book the placeholder room", "owner_input", _REASON),
        ("Choose the placeholder venue", "owner_input", _QUESTION),
    }

    # A repeated parameter or an unknown token is today's shape exactly.
    for refused in refused_snapshots:
        assert _without_clock(refused) == _without_clock(today_snapshot)
    for refused in refused_decisions:
        assert refused == today_decisions
