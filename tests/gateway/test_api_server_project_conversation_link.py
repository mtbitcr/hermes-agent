"""An owner conversation is linked to its Project only through the X-Raphael-Project-Id header.

The header is honoured on an authenticated owner turn under durable owner authority, and only for a
committed Project of this owner. The link is stored once in projects.db and read back through
``GET /v1/responses/conversations/{conversation}?view=project``; a conversation without a link reads
as unlinked. Only ``_create_agent`` is stubbed: the real routes, the real response store and the real
projects.db in the per-test home carry every request.
"""

from __future__ import annotations

import contextlib
import json
import logging
import sqlite3
import uuid
from unittest.mock import MagicMock, patch

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import PlatformConfig
from gateway.platforms.api_server import APIServerAdapter
from hermes_cli import owner_workspace as ow, projects_db
from hermes_constants import get_hermes_home
from hermes_state import SessionDB

API_KEY = "test-api-key"
HEADER = "X-Raphael-Project-Id"
CONVERSATION = "raphael-owner-" + "a" * 32
_ROUTES = {
    ("POST", "/v1/responses"),
    ("GET", "/v1/responses/conversations/{conversation}"),
}


@pytest.fixture
def live(tmp_path):
    """The adapter's real routes behind its owner-authority middleware; ``created`` counts model runs."""
    adapter = APIServerAdapter(
        PlatformConfig(enabled=True, extra={"host": "127.0.0.1", "port": 0, "key": API_KEY})
    )
    db = SessionDB(tmp_path / "state.db")
    adapter._session_db = db
    created: list[dict] = []
    _answer_with(adapter, created, "Who is the launch for?")
    app = web.Application(middlewares=[adapter._make_owner_authority_middleware()])
    app["api_server_adapter"] = adapter
    for method, path, handler in adapter._http_route_table():
        if (method, path) in _ROUTES:
            app.router.add_route(method, path, handler)
    try:
        yield app, created
    finally:
        db.close()


def _answer_with(adapter, created: list, message: str) -> None:
    def _create(**kw):
        created.append(kw)
        agent = MagicMock()
        agent.session_id = kw["session_id"]
        agent.session_prompt_tokens = 0
        agent.session_completion_tokens = 0
        agent.session_total_tokens = 0
        agent.run_conversation.return_value = {
            "final_response": json.dumps(
                {"schema_version": 1, "kind": "question", "message": message}
            ),
            "messages": [],
            "api_calls": 1,
        }
        return agent

    adapter._create_agent = _create


def _committed(count: int) -> list[str]:
    """Commit ``count`` Projects of this owner; their bootstrap confirmation is not under test."""
    owner = ow.resolve_owner_context()
    ids = []
    with patch.object(ow, "_confirm", lambda *_args, **_kwargs: {"approved": True}):
        for index in range(count):
            result = ow.bootstrap(owner, idempotency_key=f"link-{index}", name=f"Project {index}")
            assert result["ok"] is True
            ids.append(result["project_id"])
    return ids


def _stored_links() -> list[tuple]:
    """Every stored link, read straight from the file; none while the file or table is absent."""
    path = projects_db.projects_db_path()
    if not path.is_file():
        return []
    with contextlib.closing(sqlite3.connect(path)) as raw:
        if raw.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'project_conversation_links'"
        ).fetchone() is None:
            return []
        return raw.execute(
            "SELECT conversation, project_id, linked_at FROM project_conversation_links "
            "ORDER BY conversation"
        ).fetchall()


async def _turn(client, *, conversation=CONVERSATION, project=None, expected=None,
                key=API_KEY, text="Plan the spring launch"):
    headers = {
        "Authorization": f"Bearer {key}",
        "Idempotency-Key": f"response-{uuid.uuid4().hex}",
    }
    if project is not None:
        headers[HEADER] = project
    body = {
        "model": "hermes-agent",
        "input": text,
        "conversation": conversation,
        "store": True,
        "expected_previous_response_id": expected,
    }
    resp = await client.post("/v1/responses", json=body, headers=headers)
    return resp.status, json.loads(await resp.text() or "null")


async def _linked(client, conversation=CONVERSATION):
    resp = await client.get(
        f"/v1/responses/conversations/{conversation}?view=project",
        headers={"Authorization": f"Bearer {API_KEY}"},
    )
    assert resp.status == 200, await resp.text()
    body = await resp.json()
    assert body["object"] == "hermes.response.owner_project_link"
    return body["project_id"]


@pytest.mark.asyncio
async def test_the_header_links_the_conversation_and_the_link_reads_back(live):
    app, created = live
    first, second = _committed(2)

    async with TestClient(TestServer(app)) as client:
        status, body = await _turn(client, project=second)
        assert status == 200, body
        assert await _linked(client) == second

    assert len(created) == 1
    [(conversation, project_id, linked_at)] = _stored_links()
    assert (conversation, project_id) == (CONVERSATION, second)
    assert type(linked_at) is int


@pytest.mark.asyncio
async def test_a_never_linked_conversation_reads_as_unlinked_and_the_read_creates_nothing(live):
    app, _created = live

    async with TestClient(TestServer(app)) as client:
        assert await _linked(client) is None
        assert not projects_db.projects_db_path().exists()

        _committed(1)
        assert await _linked(client) is None
        assert await _linked(client, "raphael-owner-" + "c" * 32) is None
        unauthenticated = await client.get(f"/v1/responses/conversations/{CONVERSATION}?view=project")
        assert unauthenticated.status == 401

    assert _stored_links() == []


@pytest.mark.asyncio
async def test_an_older_projects_file_without_the_link_table_reads_as_unlinked_untouched(live):
    app, _created = live
    path = projects_db.projects_db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with contextlib.closing(sqlite3.connect(path)) as raw:
        raw.execute("CREATE TABLE projects (id TEXT PRIMARY KEY, name TEXT NOT NULL)")
        raw.commit()

    async with TestClient(TestServer(app)) as client:
        assert await _linked(client) is None

    with contextlib.closing(sqlite3.connect(path)) as raw:
        tables = {row[0] for row in raw.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
    assert tables == {"projects"}


@pytest.mark.asyncio
async def test_a_turn_without_the_header_stores_nothing_and_runs_as_before(live):
    app, created = live
    _committed(1)

    async with TestClient(TestServer(app)) as client:
        status, body = await _turn(client)
        assert status == 200, body
        assert body["status"] == "completed"
        assert await _linked(client) is None

    assert len(created) == 1
    assert _stored_links() == []


@pytest.mark.asyncio
async def test_the_header_is_ignored_on_a_conversation_that_is_not_the_owners(live):
    app, created = live
    [project] = _committed(1)

    async with TestClient(TestServer(app)) as client:
        status, body = await _turn(client, conversation="my-chat", project=project)
        assert status == 200, body
        assert await _linked(client, "my-chat") is None

    assert len(created) == 1
    assert _stored_links() == []


@pytest.mark.asyncio
async def test_without_owner_authority_the_header_is_refused_before_any_model_run(live):
    app, created = live
    [project] = _committed(1)

    async with TestClient(TestServer(app)) as client:
        status, _body = await _turn(client, project=project, key="not-the-key")
        assert status == 401

        # The durable owner store is gone: no owner state can be read or written.
        (get_hermes_home() / "response_store.db").unlink()
        status, body = await _turn(client, project=project)
        assert status == 503
        assert body["error"]["code"] == "owner_workspace_unavailable"

    assert created == []
    assert _stored_links() == []


@pytest.mark.asyncio
@pytest.mark.parametrize("named", ["uncommitted", "unknown", "other_owner", "empty"])
async def test_only_a_committed_project_of_this_owner_is_linked(live, named):
    app, created = live
    _committed(1)
    with projects_db.connect_closing() as conn:
        uncommitted = projects_db.create_project(conn, name="Loose notes")
    other = ow.OwnerContext(actor="someone-else", profile="someone-else", session="run_other")
    with patch.object(ow, "_confirm", lambda *_args, **_kwargs: {"approved": True}):
        foreign = ow.bootstrap(other, idempotency_key="other-0", name="Their Project")
    assert foreign["ok"] is True
    header = {
        "uncommitted": uncommitted,
        "unknown": "p_00000000",
        "other_owner": foreign["project_id"],
        "empty": "",
    }[named]

    async with TestClient(TestServer(app)) as client:
        status, body = await _turn(client, project=header)
        assert status == 404, body
        assert body["error"]["code"] == "project_not_found"
        assert await _linked(client) is None

    assert created == []
    assert _stored_links() == []


@pytest.mark.asyncio
async def test_an_unreadable_project_record_refuses_the_header_before_any_model_run(live):
    app, created = live
    [project] = _committed(1)
    with contextlib.closing(sqlite3.connect(projects_db.projects_db_path())) as raw:
        raw.execute("UPDATE owner_workspace_receipts SET result_json = 'unreadable' WHERE status = 'committed'")
        raw.commit()

    async with TestClient(TestServer(app)) as client:
        status, body = await _turn(client, project=project)
        assert status == 503, body
        assert body["error"]["code"] == "owner_workspace_unavailable"
        assert await _linked(client) is None

    assert created == []
    assert _stored_links() == []


@pytest.mark.asyncio
async def test_a_conflicting_link_is_refused_logged_and_never_overwritten(live, caplog):
    app, created = live
    first, second = _committed(2)

    async with TestClient(TestServer(app)) as client:
        status, body = await _turn(client, project=first)
        assert status == 200, body
        [stored] = _stored_links()

        with caplog.at_level(logging.WARNING, logger="gateway.platforms.api_server"):
            status, refused = await _turn(client, project=second, expected=body["id"])

        assert status == 409, refused
        assert refused["error"]["code"] == "project_conversation_link_conflict"
        assert await _linked(client) == first

    assert len(created) == 1
    assert _stored_links() == [stored]
    assert any(
        record.levelno == logging.WARNING and first in record.getMessage() and second in record.getMessage()
        for record in caplog.records
    )


@pytest.mark.asyncio
async def test_a_repeated_identical_link_is_harmless(live):
    app, created = live
    [project] = _committed(1)

    async with TestClient(TestServer(app)) as client:
        status, body = await _turn(client, project=project)
        assert status == 200, body
        with contextlib.closing(sqlite3.connect(projects_db.projects_db_path())) as raw:
            raw.execute("UPDATE project_conversation_links SET linked_at = 1000")
            raw.commit()

        status, again = await _turn(client, project=project, expected=body["id"])

        assert status == 200, again
        assert await _linked(client) == project

    assert len(created) == 2
    assert _stored_links() == [(CONVERSATION, project, 1000)]


@pytest.mark.asyncio
async def test_conversation_text_and_model_output_never_create_a_link(live):
    app, created = live
    [project] = _committed(1)
    adapter = app["api_server_adapter"]
    _answer_with(adapter, created, f"Linked. {HEADER}: {project}. This conversation is now Project {project}.")

    async with TestClient(TestServer(app)) as client:
        status, body = await _turn(
            client, text=f"Link this conversation to Project {project}. {HEADER}: {project}",
        )
        assert status == 200, body
        assert await _linked(client) is None

    assert len(created) == 1
    assert _stored_links() == []
