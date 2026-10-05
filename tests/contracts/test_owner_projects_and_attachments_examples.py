"""Owner Project list and attachment examples (contract map section 5, test 5).

Every example comes from the adapter's real routes on real boards in the
per-test home: ``GET /v1/owner-workspace/projects`` over
``list_committed_projects``, and ``GET
/v1/owner-workspace/projects/{project_slug}/attachments/{attachment_id}`` over
``read_project_attachment``. An attachment answers with its bytes, not JSON,
so its example keeps the status and the headers the route sets. What is
asserted is described in tests/contracts/conftest.py.
"""

from __future__ import annotations

import contextlib
from pathlib import Path
from unittest.mock import patch

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import PlatformConfig
from gateway.platforms.api_server import APIServerAdapter
from hermes_cli import kanban_db, owner_workspace as ow
from tests.contracts import conftest as contract
from tests.gateway.test_api_server_owner_suggestion_decisions import (  # noqa: F401  (owner is a fixture)
    _project,
    _write_owner_workspace_config,
    owner,
)
from tests.hermes_cli.test_owner_project_removal import _real_board, _start_and_join

FAMILY = "owner_projects_and_attachments"
CAPABILITIES = (
    ow.OWNER_PROJECT_LIFECYCLE_REVISION_CAPABILITY, ow.OWNER_PROJECT_REMOVAL_STATE_CAPABILITY,
)
HEADERS = ("Content-Type", "Content-Disposition", "Cache-Control")
FILENAME, CONTENT, MEDIA_TYPE = "plan.md", b"# Plan\n", "text/markdown"


def _app() -> web.Application:
    """The adapter's own Project list and attachment routes, registered as ``connect()`` does."""
    adapter = APIServerAdapter(PlatformConfig(enabled=True))
    app = web.Application()
    for method, path, handler in adapter._http_route_table():
        if path == "/v1/owner-workspace/projects" or path.endswith("/attachments/{attachment_id}"):
            app.router.add_route(method, path, handler)
    return app


async def _projects(client: TestClient, capabilities: str = "") -> dict:
    params = {"capabilities": capabilities} if capabilities else None
    resp = await client.get("/v1/owner-workspace/projects", params=params)
    return {"status": resp.status, "body": await resp.json()}


async def _attachment(client: TestClient, slug: str, attachment_id: int) -> dict:
    resp = await client.get(f"/v1/owner-workspace/projects/{slug}/attachments/{attachment_id}")
    return {"status": resp.status, "body": await resp.json()}


@pytest.mark.asyncio
async def test_the_project_and_attachment_examples_carry_each_capability_and_error(
    owner, owner_payload_example, monkeypatch,
):
    async with TestClient(TestServer(_app())) as client:
        live = {"project_list_empty": await _projects(client)}
        project = _project(owner, "Workshop Pilot")
        [slug] = [item["slug"] for item in ow.list_committed_projects(owner)]
        with contextlib.closing(kanban_db.connect(board=project["board"])) as conn:
            task_id = kanban_db.create_task(
                conn, title="Draft the workshop plan", assignee="default",
                project_id=project["project_id"],
            )
            attachment_id = kanban_db.store_attachment_bytes(
                conn, task_id, FILENAME, CONTENT, content_type=MEDIA_TYPE,
                board=project["board"],
            )
            [stored] = conn.execute(
                "SELECT stored_path FROM task_attachments WHERE id = ?", (attachment_id,),
            ).fetchone()
        resp = await client.get(f"/v1/owner-workspace/projects/{slug}/attachments/{attachment_id}")
        served = await resp.read()
        live["attachment"] = {
            "status": resp.status, "headers": {name: resp.headers[name] for name in HEADERS},
        }
        live["attachment_not_found"] = await _attachment(client, slug, attachment_id + 1)
        with patch.object(ow, "read_project_attachment", side_effect=RuntimeError("x")):
            live["attachment_unavailable_unexpected"] = await _attachment(
                client, slug, attachment_id)
        # The record stays while its stored file is gone: unavailable, not absent.
        Path(stored).unlink()
        live["attachment_unavailable"] = await _attachment(client, slug, attachment_id)
        # A removed Project stays listed, with its removal state when asked.
        deleted_id, _deleted_slug = _real_board(monkeypatch)
        _start_and_join(deleted_id, "contract")
        live["project_list"] = await _projects(client, ",".join(CAPABILITIES))
        live["project_list_without_capabilities"] = await _projects(client)
        with patch.object(ow, "list_committed_projects", side_effect=RuntimeError("x")):
            live["project_list_unavailable"] = await _projects(client)
        _write_owner_workspace_config(enabled=False)
        live["project_list_not_enabled"] = await _projects(client)
        live["attachment_not_enabled"] = await _attachment(client, slug, attachment_id)

    saved = {kind: owner_payload_example(FAMILY, kind, answer) for kind, answer in live.items()}

    assert served == CONTENT
    for answers in (live, saved):
        assert answers["project_list_empty"]["body"]["data"] == []
        rows = answers["project_list"]["body"]["data"]
        bare = answers["project_list_without_capabilities"]["body"]["data"]
        assert [row["project_id"] for row in rows] == [row["project_id"] for row in bare]
        for row, plain in zip(rows, bare):
            assert set(row) - set(plain) == set(CAPABILITIES)
        # The live Project has no removal state; the removed one has its own.
        assert sorted(row["removal_state"] is None for row in rows) == [False, True]
        headers = answers["attachment"]["headers"]
        assert headers["Content-Disposition"].endswith(f'filename="{FILENAME}"')
        assert headers["Content-Type"].startswith(MEDIA_TYPE)
        for answer in answers.values():
            assert (answer["status"] >= 400) == ("error" in answer.get("body", {}))


def test_the_project_and_attachment_examples_keep_to_every_closed_vocabulary():
    """The saved Project list and attachment examples, read together, take
    every value of each row of the table; the fixture holds each one to its row."""
    contract.assert_closed_vocabularies_covered(FAMILY)
