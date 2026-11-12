"""Read portfolio advice through the receipt-backed Project list."""

import json
from pathlib import Path
import subprocess
import sys

import pytest
import yaml
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from types import SimpleNamespace
from unittest.mock import patch

from gateway.config import GatewayConfig, PlatformConfig
from gateway.platforms.api_server import APIServerAdapter
from hermes_cli import kanban_db, owner_project_review as review, owner_workspace as ow, projects_db
from tools import approval
from plugins.dashboard_auth.raphael_workspace import model_policy
from tests.gateway.test_api_server_owner_board_counts import _CONFIG, _get
from tests.gateway.test_api_server_owner_review_brief import NOW, project_review  # noqa: F401
from tests.hermes_cli.test_owner_project_review_plans import _logical_state


@pytest.fixture
def portfolio(project_review, monkeypatch, request):
    monkeypatch.setattr(model_policy, "configured_assignment_for",
                        lambda profile: model_policy.assignment_for(profile, "anthropic"))
    root = Path.home() / ".hermes"
    config = yaml.safe_load((root / "config.yaml").read_text())
    projects = [project_review]
    if getattr(request, "param", None) == "mixed":
        ctx = ow.OwnerContext(actor="secondary", profile="secondary", session="secondary-portfolio")

        def approve_fixture(data):
            assert approval.resolve_gateway_approval(ctx.session, "once", approval_id=data["approval_id"]) == 1

        approval.register_gateway_notify(ctx.session, approve_fixture)
        try:
            projects.append(ow.bootstrap(ctx, idempotency_key="secondary-portfolio", name="Guest Visit"))
        finally:
            approval.unregister_gateway_notify(ctx.session)
        assert projects[-1]["ok"] is True
    config["owner_review"] = {"enabled": True, "allowlist": [p["project_id"] for p in projects], "portfolio": True}
    (root / "config.yaml").write_text(yaml.safe_dump(config))
    for project in projects:
        with kanban_db.connect_closing(kanban_db.kanban_db_path(board=project["board"]), board=project["board"]) as board:
            kanban_db.create_task(board, title="Book the room", project_id=project["project_id"])
    assert review.review_tick(NOW) == len(projects)
    path = kanban_db.kanban_db_path(board="project-summaries")
    with kanban_db.connect_closing(path, board="project-summaries") as board:
        for project in projects:
            task_id = board.execute("SELECT id FROM tasks WHERE idempotency_key=?", (
                f"project-review:{project['project_id']}:2026-11-10:0",
            )).fetchone()[0]
            assert kanban_db.claim_task(board, task_id) is not None
            assert kanban_db.complete_task(board, task_id, summary="Reviewed the room plan.", metadata={"project_review": {
                "summary": {"outcome": "The room needs a booking.", "next_action": "Book the room.", "owner_decision": None},
                "brief": {"headline": "The room needs a booking.", "what_changed": "Book the room is ready.",
                          "next_step": "Book the room.", "needs_you": None},
                "classification": [], "proposed_closures": [],
            }})
    assert review.review_tick(NOW + 1) == 1
    with kanban_db.connect_closing(path, board="project-summaries") as board:
        task_id = board.execute("SELECT id FROM tasks WHERE idempotency_key='project-review:portfolio:2026-11-10:0'").fetchone()[0]
        assert kanban_db.claim_task(board, task_id) is not None
        assert kanban_db.complete_task(board, task_id, summary="Reviewed the Projects together.", metadata={"portfolio_review": {
            "conflicts": [], "priorities": ["Meeting Plan: Book the room."], "shared_blockers": [],
        }})
    assert review.review_tick(NOW + 2) == 0
    return root, project_review


@pytest.mark.asyncio
async def test_the_portfolio_capability_adds_one_key_and_preserves_the_receipt_backed_list(portfolio):
    ctx = ow.OwnerContext(actor="default", profile="default", session="portfolio-http")
    old_status, old_body = await _get(ctx)
    new_status, new_body = await _get(ctx, "?capabilities=portfolio_review")
    assert old_status == new_status == 200
    old, new = json.loads(old_body), json.loads(new_body)
    assert set(new) - set(old) == {"portfolio_review"}
    assert {key: value for key, value in new.items() if key != "portfolio_review"} == old
    assert new["portfolio_review"] == {
        "review_date": "2026-11-10", "conflicts": [], "priorities": ["Meeting Plan: Book the room."],
        "shared_blockers": [], "written_at": NOW + 2, "revision": 1,
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("query", [
    "?capabilities=unknown", "?capabilities=portfolio_review&capabilities=board_counts_v1",
    "?capabilities=board_counts_v1&capabilities=portfolio_review", "?capabilities=" + "x" * 257,
])
async def test_an_unnegotiated_portfolio_keeps_the_legacy_envelope(portfolio, query):
    ctx = ow.OwnerContext(actor="default", profile="default", session="portfolio-http")
    _, plain = await _get(ctx)
    status, body = await _get(ctx, query)
    assert status == 200
    assert body == plain


@pytest.mark.asyncio
async def test_portfolio_negotiation_preserves_the_existing_count_and_change_capabilities(portfolio):
    ctx = ow.OwnerContext(actor="default", profile="default", session="portfolio-http")
    _, old = await _get(ctx, "?capabilities=board_counts_v1,project_last_change_v1")
    status, new = await _get(ctx, "?capabilities=board_counts_v1,project_last_change_v1,portfolio_review")
    assert status == 200
    before, after = json.loads(old), json.loads(new)
    assert after.pop("portfolio_review")["priorities"] == ["Meeting Plan: Book the room."]
    assert after == before


@pytest.mark.asyncio
async def test_a_reader_without_the_source_receipt_gets_no_portfolio(portfolio):
    ctx = ow.OwnerContext(actor="secondary", profile="secondary", session="portfolio-http")
    status, body = await _get(ctx, "?capabilities=portfolio_review")
    assert status == 200
    assert json.loads(body) == {"object": "hermes.owner_workspace.project_list", "data": [], "portfolio_review": None}


@pytest.mark.asyncio
@pytest.mark.parametrize("portfolio", ["mixed"], indirect=True)
async def test_each_partial_owner_cannot_read_a_portfolio_from_both_receipt_scopes(portfolio):
    for profile in ("default", "secondary"):
        ctx = ow.OwnerContext(actor=profile, profile=profile, session="portfolio-http")
        status, body = await _get(ctx, "?capabilities=portfolio_review")
        assert status == 200
        value = json.loads(body)
        assert len(value["data"]) == 1
        assert value["portfolio_review"] is None


@pytest.mark.asyncio
async def test_legacy_advice_stays_hidden_after_an_additive_scope_migration(portfolio):
    root, _project = portfolio
    with projects_db.connect_closing(root / "projects.db") as registry:
        before = projects_db.get_portfolio_review_brief(registry)
        with kanban_db.write_txn(registry):
            registry.execute("ALTER TABLE portfolio_review_briefs DROP COLUMN project_scope")
    ctx = ow.OwnerContext(actor="default", profile="default", session="portfolio-http")
    status, body = await _get(ctx, "?capabilities=portfolio_review")
    assert status == 200
    assert json.loads(body)["portfolio_review"] is None
    # A separate process exercises the first-open upgrade without the fixture's schema cache.
    command = [sys.executable, "-c", "import sys;from pathlib import Path;from hermes_cli import projects_db;"
               "projects_db.connect(Path(sys.argv[1])).close()", str(root / "projects.db")]
    subprocess.run(command, check=True)
    subprocess.run(command, check=True)
    with projects_db.connect_closing(root / "projects.db") as registry:
        after = projects_db.get_portfolio_review_brief(registry)
    assert after["project_scope"] == []
    assert {key: value for key, value in after.items() if key != "project_scope"} == {
        key: value for key, value in before.items() if key != "project_scope"
    }
    status, body = await _get(ctx, "?capabilities=portfolio_review")
    assert status == 200
    assert json.loads(body)["portfolio_review"] is None


@pytest.mark.asyncio
@pytest.mark.parametrize("auth", ["missing", "wrong", "valid"])
async def test_the_portfolio_read_keeps_the_existing_http_authentication(portfolio, auth):
    adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={"key": "synthetic-portfolio-key"}))
    app = web.Application()
    for method, path, handler in adapter._http_route_table():
        if method == "GET" and path == "/v1/owner-workspace/projects":
            app.router.add_route(method, path, handler)
    headers = {} if auth == "missing" else {"Authorization": "Bearer " + (
        "synthetic-portfolio-key" if auth == "valid" else "wrong-synthetic-key"
    )}
    async with TestClient(TestServer(app)) as client:
        response = await client.get("/v1/owner-workspace/projects?capabilities=portfolio_review", headers=headers)
        body = await response.json()
    assert response.status == (200 if auth == "valid" else 401)
    if auth == "valid":
        assert body["portfolio_review"]["priorities"] == ["Meeting Plan: Book the room."]
    else:
        assert "portfolio_review" not in body


@pytest.mark.asyncio
@pytest.mark.parametrize("value", [
    ["Meeting Plan: Book the room."] * 6,
    ["Meeting Plan: " + "x" * 200],
    ["Meeting Plan: Read /tmp/private."],
])
async def test_unreadable_advice_does_not_hide_the_owned_project_list(portfolio, value):
    root, project = portfolio
    with projects_db.connect_closing(root / "projects.db") as registry:
        with kanban_db.write_txn(registry):
            registry.execute("UPDATE portfolio_review_briefs SET priorities=?", (json.dumps(value),))
    ctx = ow.OwnerContext(actor="default", profile="default", session="portfolio-http")
    status, body = await _get(ctx, "?capabilities=portfolio_review")
    assert status == 200
    result = json.loads(body)
    assert [item["project_id"] for item in result["data"]] == [project["project_id"]]
    assert result["portfolio_review"] is None


@pytest.mark.asyncio
async def test_the_portfolio_capability_does_not_write_registry_data_or_schema(portfolio):
    root, _project = portfolio
    before = _logical_state(root / "projects.db")
    ctx = ow.OwnerContext(actor="default", profile="default", session="portfolio-http")
    status, body = await _get(ctx, "?capabilities=portfolio_review")
    assert status == 200
    assert json.loads(body)["portfolio_review"] is not None
    assert _logical_state(root / "projects.db") == before


@pytest.mark.asyncio
async def test_an_advice_registry_that_cannot_be_opened_does_not_fail_the_project_list(portfolio, monkeypatch):
    _root, project = portfolio
    real = ow._open_read_only_sqlite

    def held(path, *, label):
        # Only the advice read fails, as when the registry goes away between its check and its read-only open.
        if sys._getframe(1).f_code.co_name == "_portfolio_view":
            raise ow.OwnerWorkspaceError("snapshot_unavailable", f"{label} could not be opened read-only")
        return real(path, label=label)

    monkeypatch.setattr(ow, "_open_read_only_sqlite", held)
    ctx = ow.OwnerContext(actor="default", profile="default", session="portfolio-http")
    status, body = await _get(ctx, "?capabilities=portfolio_review")
    assert status == 200
    result = json.loads(body)
    assert [item["project_id"] for item in result["data"]] == [project["project_id"]]
    assert result["portfolio_review"] is None


@pytest.mark.asyncio
@pytest.mark.parametrize("setting", ["review_off", "portfolio_off", "removed_from_allowlist", "allowlisted_by_slug"])
async def test_advice_follows_the_current_review_settings_and_allowlist(portfolio, setting):
    root, project = portfolio
    ctx = ow.OwnerContext(actor="default", profile="default", session="portfolio-http")
    _, body = await _get(ctx, "?capabilities=portfolio_review")
    assert json.loads(body)["portfolio_review"] is not None
    config = yaml.safe_load((root / "config.yaml").read_text())
    section = config["owner_review"]
    if setting == "review_off":
        section["enabled"] = False
    elif setting == "portfolio_off":
        section["portfolio"] = False
    elif setting == "removed_from_allowlist":
        section["allowlist"] = ["another-project"]
    else:
        section["allowlist"] = [project["slug"]]
    (root / "config.yaml").write_text(yaml.safe_dump(config))
    status, body = await _get(ctx, "?capabilities=portfolio_review")
    assert status == 200
    result = json.loads(body)
    assert [item["project_id"] for item in result["data"]] == [project["project_id"]]
    if setting == "allowlisted_by_slug":
        assert result["portfolio_review"]["priorities"] == ["Meeting Plan: Book the room."]
    else:
        assert result["portfolio_review"] is None


@pytest.mark.asyncio
@pytest.mark.parametrize("profile", ["default", "secondary"])
async def test_a_profile_prefixed_read_shows_advice_only_to_the_profile_that_holds_its_receipts(
    portfolio, monkeypatch, profile,
):
    root, project = portfolio
    secondary = root / "profiles" / "secondary"
    secondary.mkdir(parents=True)
    (secondary / "config.yaml").write_text((root / "config.yaml").read_text())
    # A named profile authenticates with its own key, never the listener's.
    (secondary / ".env").write_text("API_SERVER_KEY=synthetic-secondary-portfolio-key\n")
    key = "synthetic-portfolio-key" if profile == "default" else "synthetic-secondary-portfolio-key"
    monkeypatch.setattr("hermes_cli.profiles.profiles_to_serve", lambda multiplex, profile_allowlist=None: [
        ("default", str(root)), ("secondary", str(secondary)),
    ])
    adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={"key": "synthetic-portfolio-key"}))
    adapter.gateway_runner = SimpleNamespace(config=GatewayConfig(multiplex_profiles=True))
    app = web.Application(middlewares=[adapter._make_profile_prefix_middleware()])
    for method, path, handler in adapter._http_route_table():
        if method == "GET" and path == "/v1/owner-workspace/projects":
            app.router.add_route(method, "/p/{profile}" + path, handler)
    ctx = ow.OwnerContext(actor=profile, profile=profile, session="portfolio-http")
    with (
        patch("gateway.run._load_gateway_config", return_value=_CONFIG),
        patch("hermes_cli.owner_workspace.resolve_owner_context", return_value=ctx),
    ):
        async with TestClient(TestServer(app)) as client:
            response = await client.get(
                f"/p/{profile}/v1/owner-workspace/projects?capabilities=portfolio_review",
                headers={"Authorization": "Bearer " + key},
            )
            body = await response.json()
    assert response.status == 200
    if profile == "default":
        assert [item["project_id"] for item in body["data"]] == [project["project_id"]]
        assert body["portfolio_review"]["priorities"] == ["Meeting Plan: Book the room."]
    else:
        assert body == {"object": "hermes.owner_workspace.project_list", "data": [], "portfolio_review": None}
