"""An owner conversation is linked to its Project only through the X-Raphael-Project-Id header.

The header is honoured on an authenticated owner turn under durable owner authority, and only for a
committed Project of this owner. It is part of the turn's idempotency fingerprint, and the link is
written only once the turn is admitted, before any model runs, so a refused request leaves none.
The link is stored once in projects.db and read back through
``GET /v1/responses/conversations/{conversation}?view=project``; a conversation without a link reads
as unlinked. Only ``_create_agent`` is stubbed: the real routes, the real response store and the real
projects.db in the per-test home carry every request.

A test run with ``live`` set to ``profile`` makes every request through ``/p/raphael-planner/`` with
multiplexing on. That profile has a Project store of its own with none of the owner's committed
Projects, and the header is still checked, and the link written and read, in the owner's store.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import sqlite3
import uuid
from pathlib import Path
from types import SimpleNamespace
from typing import NamedTuple
from unittest.mock import MagicMock, patch

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from agent import secret_scope
from gateway.config import GatewayConfig, PlatformConfig
from gateway.platforms import api_server
from gateway.platforms.api_server import APIServerAdapter
from hermes_cli import kanban_db, owner_workspace as ow, projects_db
from hermes_cli.owner_project_review_record import read_conversation_news
from hermes_constants import get_hermes_home, reset_hermes_home_override, set_hermes_home_override
from hermes_state import SessionDB

API_KEY = "test-api-key"
PLANNER = "raphael-planner"
PLANNER_KEY = "test-raphael-planner-key"
HEADER = "X-Raphael-Project-Id"
CONVERSATION = "raphael-owner-" + "a" * 32
_ROUTES = {
    ("POST", "/v1/responses"),
    ("GET", "/v1/responses/conversations/{conversation}"),
}


class _Route(NamedTuple):
    prefix: str
    key: str
    profile: str


# A test marked so runs natively and again through the planner profile's route.
_ON_BOTH_ROUTES = pytest.mark.parametrize("live", ["native", "profile"], indirect=True)


@pytest.fixture
def live(request, tmp_path, monkeypatch):
    """The adapter's real routes behind its owner-authority middleware; ``created`` counts model runs.

    Natively by default; with the ``profile`` parameter only through the planner profile's route.
    """
    route = _Route("", API_KEY, "default")
    if getattr(request, "param", "native") == "profile":
        planner_project = _serve_the_planner_profile(tmp_path, monkeypatch)
        route = _Route(f"/p/{PLANNER}", PLANNER_KEY, PLANNER)
    adapter = APIServerAdapter(
        PlatformConfig(enabled=True, extra={"host": "127.0.0.1", "port": 0, "key": API_KEY})
    )
    db = SessionDB(tmp_path / "state.db")
    adapter._session_db = db
    created: list[dict] = []
    _answer_with(adapter, created, "Who is the launch for?")
    middlewares = [adapter._make_owner_authority_middleware()]
    if route.prefix:
        adapter.gateway_runner = SimpleNamespace(config=GatewayConfig(multiplex_profiles=True))
        middlewares.insert(0, adapter._make_profile_prefix_middleware())
    app = web.Application(middlewares=middlewares)
    app["api_server_adapter"] = adapter
    app["route"] = route
    if route.prefix:
        app["planner_project"] = planner_project
    for method, path, handler in adapter._http_route_table():
        if (method, path) in _ROUTES:
            app.router.add_route(method, ("/p/{profile}" if route.prefix else "") + path, handler)
    try:
        yield app, created
    finally:
        db.close()


def _planner_home(tmp_path) -> Path:
    return tmp_path / ".hermes" / "profiles" / PLANNER


def _serve_the_planner_profile(tmp_path, monkeypatch) -> str:
    """Multiplex the planner profile; its own Project store holds one committed Project of its own."""
    home = _planner_home(tmp_path)
    home.mkdir(parents=True)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    monkeypatch.setattr(secret_scope, "_MULTIPLEX_ACTIVE", True)
    # A named profile authenticates with its own key, never the listener's.
    (home / ".env").write_text(f"API_SERVER_KEY={PLANNER_KEY}\n", encoding="utf-8")
    token = set_hermes_home_override(str(home))
    try:
        [project] = _committed(1, name="Planner Project")
    finally:
        reset_hermes_home_override(token)
    return project


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


def _committed(count: int, *, name: str = "Project") -> list[str]:
    """Commit ``count`` Projects of this owner; their bootstrap confirmation is not under test."""
    owner = ow.resolve_owner_context()
    ids = []
    with patch.object(ow, "_confirm", lambda *_args, **_kwargs: {"approved": True}):
        for index in range(count):
            result = ow.bootstrap(owner, idempotency_key=f"link-{index}", name=f"{name} {index}")
            assert result["ok"] is True
            ids.append(result["project_id"])
    return ids


def _stored_links(path: Path | None = None) -> list[tuple]:
    """Every stored link, read straight from the file; none while the file or table is absent.

    The file is the owner's projects.db unless ``path`` names another. A check made while a request
    runs passes it, since a profile route scopes the request to that profile's home.
    """
    path = path or projects_db.projects_db_path()
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
                key=None, text="Plan the spring launch", idempotency=None):
    route = client.app["route"]
    headers = {
        "Authorization": f"Bearer {key or route.key}",
        "Idempotency-Key": idempotency or f"response-{uuid.uuid4().hex}",
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
    resp = await client.post(f"{route.prefix}/v1/responses", json=body, headers=headers)
    return resp.status, json.loads(await resp.text() or "null")


async def _linked(client, conversation=CONVERSATION):
    route = client.app["route"]
    resp = await client.get(
        f"{route.prefix}/v1/responses/conversations/{conversation}?view=project",
        headers={"Authorization": f"Bearer {route.key}"},
    )
    assert resp.status == 200, await resp.text()
    body = await resp.json()
    assert body["object"] == "hermes.response.owner_project_link"
    return body["project_id"]


@pytest.mark.asyncio
@_ON_BOTH_ROUTES
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
@_ON_BOTH_ROUTES
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
@_ON_BOTH_ROUTES
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
@_ON_BOTH_ROUTES
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
@_ON_BOTH_ROUTES
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


@pytest.mark.asyncio
@_ON_BOTH_ROUTES
@pytest.mark.parametrize(
    "value",
    [42, {}, [], "", [{"role": "user", "content": [{"type": "input_image", "image_url": "not-a-url"}]}]],
    ids=["number", "object", "empty_list", "empty_string", "malformed_image"],
)
async def test_a_rejected_input_leaves_no_link(live, value):
    app, created = live
    first, second = _committed(2)

    async with TestClient(TestServer(app)) as client:
        status, refused = await _turn(client, project=first, text=value)
        assert status == 400, refused
        assert _stored_links() == []

        status, body = await _turn(client, project=second)
        assert status == 200, body
        assert await _linked(client) == second

    assert len(created) == 1


@pytest.mark.asyncio
@_ON_BOTH_ROUTES
async def test_a_turn_refused_while_another_turn_runs_leaves_no_link(live):
    app, created = live
    first, second = _committed(2)
    adapter = app["api_server_adapter"]
    started, release = asyncio.Event(), asyncio.Event()
    run = adapter._run_agent

    async def _held(*args, **kwargs):
        started.set()
        await release.wait()
        return await run(*args, **kwargs)

    adapter._run_agent = _held
    async with TestClient(TestServer(app)) as client:
        running = asyncio.create_task(_turn(client))
        await asyncio.wait_for(started.wait(), timeout=10)
        try:
            status, refused = await _turn(client, project=first)
            links = _stored_links()
        finally:
            release.set()
        running_status, body = await running
        assert running_status == 200, body
        assert status == 409, refused
        assert refused["error"]["code"] == "owner_conversation_locked"
        assert links == []

        status, body = await _turn(client, project=second, expected=body["id"])
        assert status == 200, body
        assert await _linked(client) == second

    assert len(created) == 2


@pytest.mark.asyncio
@_ON_BOTH_ROUTES
async def test_a_turn_refused_on_a_closed_conversation_leaves_no_link(live):
    app, created = live
    first, second = _committed(2)
    assert app["api_server_adapter"]._response_store.close_owner_conversation(
        app["route"].profile, CONVERSATION, None,
    ) is True
    successor = f"{CONVERSATION}-{'b' * 32}"

    async with TestClient(TestServer(app)) as client:
        status, refused = await _turn(client, project=first)
        assert status == 409, refused
        assert refused["error"]["code"] == "owner_conversation_locked"
        assert _stored_links() == []

        # The group's next session, where the owner goes on after a close.
        status, body = await _turn(client, conversation=successor, project=second)
        assert status == 200, body
        assert await _linked(client, successor) == second
        assert await _linked(client) is None

    assert len(created) == 1


@pytest.mark.asyncio
@_ON_BOTH_ROUTES
@pytest.mark.parametrize("refusal", ["conflict", "store_failure"])
async def test_a_link_refused_at_admission_releases_the_turn_and_runs_no_model(live, refusal):
    app, created = live
    first, second = _committed(2)
    owner_store = projects_db.projects_db_path()
    store = app["api_server_adapter"]._response_store
    reserve = store.reserve_owner_conversation
    key = f"response-{uuid.uuid4().hex}"

    def _linked_meanwhile(*args, **kwargs):
        # Another turn links the conversation between this turn's check and its write.
        with projects_db.connect_closing(owner_store) as conn:
            projects_db.link_project_conversation(conn, CONVERSATION, second)
        return reserve(*args, **kwargs)

    if refusal == "conflict":
        injected = patch.object(store, "reserve_owner_conversation", _linked_meanwhile)
        expected = (409, "project_conversation_link_conflict", [(CONVERSATION, second)])
    else:
        injected = patch.object(
            projects_db, "link_project_conversation", side_effect=sqlite3.OperationalError("disk I/O error"),
        )
        expected = (503, "owner_workspace_unavailable", [])

    async with TestClient(TestServer(app)) as client:
        with injected:
            status, refused = await _turn(client, project=first, idempotency=key)
        assert (status, refused["error"]["code"]) == expected[:2], refused
        assert [link[:2] for link in _stored_links()] == expected[2]
        assert created == []

        # Neither reservation is left behind: the same key, naming the other Project, is admitted.
        status, body = await _turn(client, project=second, idempotency=key)
        assert status == 200, body
        assert await _linked(client) == second

    assert len(created) == 1


@pytest.mark.asyncio
@_ON_BOTH_ROUTES
@pytest.mark.parametrize("mode", ["standard", "stream", "background"])
async def test_the_link_is_in_place_before_the_model_runs_on_every_path(live, mode):
    app, created = live
    route = app["route"]
    [project] = _committed(1)
    owner_store = projects_db.projects_db_path()
    adapter = app["api_server_adapter"]
    create = adapter._create_agent
    links_at_model_start = []

    def _create(**kw):
        links_at_model_start.append([link[:2] for link in _stored_links(owner_store)])
        return create(**kw)

    adapter._create_agent = _create
    body = {
        "model": "hermes-agent",
        "input": "Plan the spring launch",
        "conversation": CONVERSATION,
        "store": True,
        "expected_previous_response_id": None,
        **({"stream": True} if mode == "stream" else {"background": True} if mode == "background" else {}),
    }
    headers = {
        "Authorization": f"Bearer {route.key}",
        "Idempotency-Key": f"response-{uuid.uuid4().hex}",
        HEADER: project,
    }

    async with TestClient(TestServer(app)) as client:
        resp = await client.post(f"{route.prefix}/v1/responses", json=body, headers=headers)
        assert resp.status == 200, await resp.text()
        await resp.read()
        for _ in range(500):
            if adapter._response_store.get_conversation(CONVERSATION, profile=route.profile):
                break
            await asyncio.sleep(0.01)
        assert await _linked(client) == project

    assert len(created) == 1
    assert links_at_model_start == [[(CONVERSATION, project)]]


@pytest.mark.asyncio
@pytest.mark.parametrize("durable", [False, True], ids=["cache", "durable"])
@pytest.mark.parametrize(
    ("live", "change"),
    # An added header is refused before any Project is read, so the profile route repeats the other two.
    [("native", "changed"), ("native", "added"), ("native", "removed"), ("profile", "changed"), ("profile", "removed")],
    indirect=["live"],
)
async def test_a_retry_that_changes_the_header_is_refused_and_changes_no_link(live, change, durable):
    app, created = live
    first, second = _committed(2)
    before, after = {"changed": (first, second), "added": (None, first), "removed": (first, None)}[change]
    key = f"response-{uuid.uuid4().hex}"

    async with TestClient(TestServer(app)) as client:
        status, body = await _turn(client, project=before, idempotency=key)
        assert status == 200, body
        links = _stored_links()
        if durable:
            api_server._idem_cache._store.clear()

        status, refused = await _turn(client, project=after, idempotency=key)
        assert status == 409, refused
        assert refused["error"]["code"] == "idempotency_conflict"

    assert _stored_links() == links
    assert len(created) == 1


@pytest.mark.asyncio
@_ON_BOTH_ROUTES
@pytest.mark.parametrize("durable", [False, True], ids=["cache", "durable"])
async def test_an_identical_retry_replays_without_a_model_run_or_a_link_write(live, durable):
    app, created = live
    [project] = _committed(1)
    key = f"response-{uuid.uuid4().hex}"

    async with TestClient(TestServer(app)) as client:
        status, body = await _turn(client, project=project, idempotency=key)
        assert status == 200, body
        links = _stored_links()
        if durable:
            api_server._idem_cache._store.clear()

        with patch.object(
            projects_db, "link_project_conversation", wraps=projects_db.link_project_conversation,
        ) as write:
            status, replay = await _turn(client, project=project, idempotency=key)
        assert status == 200, replay
        assert replay == body
        assert write.call_count == 0

    assert _stored_links() == links
    assert len(created) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("live", ["profile"], indirect=True)
async def test_the_daily_review_reads_a_link_written_through_a_profile_route(live):
    app, _created = live
    first, second = _committed(2)

    async with TestClient(TestServer(app)) as client:
        status, body = await _turn(client, project=second)
        assert status == 200, body

    # The registry review_tick opens: projects.db in the kanban home.
    with projects_db.connect_closing(kanban_db.kanban_home() / "projects.db") as registry:
        cursor, _newest = read_conversation_news(registry, second)
        assert cursor is not None
        assert read_conversation_news(registry, first) == (None, None)


@pytest.mark.asyncio
@pytest.mark.parametrize("live", ["profile"], indirect=True)
async def test_a_profile_route_reaches_the_owner_store_only_for_the_named_project_and_its_link(live, tmp_path):
    app, created = live
    [project] = _committed(1)
    adapter = app["api_server_adapter"]
    create = adapter._create_agent
    homes_at_model_start = []

    def _create(**kw):
        homes_at_model_start.append(get_hermes_home())
        return create(**kw)

    adapter._create_agent = _create

    async with TestClient(TestServer(app)) as client:
        # A Project committed in the profile's own store is not one of the owner's committed Projects.
        status, body = await _turn(client, project=app["planner_project"])
        assert status == 404, body
        assert body["error"]["code"] == "project_not_found"
        assert await _linked(client) is None

        status, body = await _turn(client, project=project)
        assert status == 200, body
        assert await _linked(client) == project

    # The turn itself still runs in the profile's own home, and the profile's store gains no link.
    assert homes_at_model_start == [_planner_home(tmp_path)]
    assert len(created) == 1
    assert [link[:2] for link in _stored_links()] == [(CONVERSATION, project)]
    assert _stored_links(_planner_home(tmp_path) / "projects.db") == []
