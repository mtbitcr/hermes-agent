"""Read accepted daily advice through the existing owned Project snapshot."""

import json
from datetime import datetime, timezone

import pytest
import yaml
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import PlatformConfig
from gateway.platforms.api_server import APIServerAdapter
from hermes_cli import kanban_db, owner_project_review as review, projects_db
from tests.hermes_cli.test_owner_project_review_plans import _approved_plan, _logical_state
from tests.hermes_cli.test_owner_project_review_record import _written_at, root  # noqa: F401
from tests.hermes_cli.test_owner_workspace import _bootstrap_board, ctx  # noqa: F401

NOW = int(datetime(2026, 11, 10, 7, tzinfo=timezone.utc).timestamp())
CAPABILITY = "project_review_advice_v1"
_SNAPSHOT = "/v1/owner-workspace/projects/{project_slug}/snapshot"


def app() -> web.Application:
    """The adapter's own snapshot route, registered as ``connect()`` does."""
    adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={}))
    application = web.Application()
    for method, path, handler in adapter._http_route_table():
        if method == "GET" and path == _SNAPSHOT:
            application.router.add_route(method, path, handler)
    return application


def stable(data: dict) -> dict:
    """The snapshot without the steward's read time, which changes with each second."""
    if "steward" not in data:
        return data
    return {**data, "steward": {**data["steward"], "generated_at": None}}


@pytest.fixture
def accepted_advice(root, ctx, monkeypatch):
    monkeypatch.setattr("gateway.run._hermes_home", root)
    with _written_at(NOW - 86400):
        setup = _bootstrap_board(ctx)
        initial = _approved_plan(ctx, setup)
    [task_id] = initial["created_task_ids"]
    with projects_db.connect_closing(root / "projects.db") as registry:
        project = projects_db.get_project(registry, setup["project_id"])
    path = kanban_db.kanban_db_path(board=project.board_slug)
    with kanban_db.connect_closing(path, board=project.board_slug) as board:
        kanban_db.add_comment(board, task_id, "default", "The venue needs confirmation.")
        task = kanban_db.get_task(board, task_id)
        target = {"task_id": task_id, "expected_status": task.status,
                  "expected_revision": kanban_db.task_event_revision(board, task_id)}
    proposal = {"request_title": "Wait for the venue confirmation",
                "summary": "Postpone the room work until the venue is confirmed.",
                "specification": "Keep the room work and postpone it until the venue is confirmed.",
                "current_milestone": "Confirm the venue.",
                "owner_visible_result": "The room work waits for the venue confirmation.",
                "later_milestones": [],
                "changes": [{"action": "postpone", "reason": "The venue is not confirmed.", "target": target}]}
    config = {"owner_review": {"enabled": True, "allowlist": [project.id]},
              "gateway": {"api_server": {"owner_workspace": {"enabled": True}}}}
    (root / "config.yaml").write_text(yaml.safe_dump(config))
    assert review.review_tick(NOW) == 1
    with projects_db.connect_closing(root / "projects.db") as registry:
        review_task = registry.execute("SELECT open_task_id FROM project_review_state WHERE project_id=?", (project.id,)).fetchone()[0]
    value = {"summary": {"outcome": "The venue is not confirmed.", "next_action": "Book the room.", "owner_decision": None},
             "brief": {"headline": "The venue needs confirmation.", "what_changed": "Book the room needs the venue confirmation.",
                       "next_step": "Book the room.", "needs_you": None},
             "classification": [], "proposed_closures": [], "plan_proposal": proposal}
    with kanban_db.connect_closing(kanban_db.kanban_db_path(board="project-summaries"), board="project-summaries") as board:
        assert kanban_db.claim_task(board, review_task) is not None
        assert kanban_db.complete_task(board, review_task, summary="Reviewed the venue.", metadata={"project_review": value})
    assert review.review_tick(NOW + 1) == 0
    return {"root": root, "project": project, "path": path, "proposal": proposal, "review_task": review_task,
            "config": config, "value": value}


@pytest.mark.asyncio
async def test_the_owned_snapshot_adds_only_accepted_advice_and_keeps_both_stores_unchanged(accepted_advice):
    data = accepted_advice
    endpoint = f"/v1/owner-workspace/projects/{data['project'].slug}/snapshot?capabilities=planning_context_v2"
    before = (_logical_state(data["root"] / "projects.db"), _logical_state(data["path"]))
    async with TestClient(TestServer(app())) as client:
        response = await client.get(endpoint)
        assert response.status == 200
        legacy = (await response.json())["data"]
        response = await client.get(endpoint + "," + CAPABILITY)
        assert response.status == 200
        current = (await response.json())["data"]
    assert "raphael_review_advice" not in legacy
    assert current["raphael_review_advice"] == {"review_date": "2026-11-10", "proposal": data["proposal"]}
    assert stable({key: value for key, value in current.items() if key != "raphael_review_advice"}) == stable(legacy)
    assert (_logical_state(data["root"] / "projects.db"), _logical_state(data["path"])) == before


@pytest.mark.asyncio
@pytest.mark.parametrize("fault", ["off", "unlisted", "digest", "body", "work", "foreign", "legacy", "prose"])
async def test_unavailable_or_changed_advice_grants_no_projection_and_changes_no_store(accepted_advice, fault):
    data = accepted_advice
    project, root = data["project"], data["root"]
    review_path = kanban_db.kanban_db_path(board="project-summaries")
    if fault in {"off", "unlisted"}:
        if fault == "off":
            data["config"]["owner_review"]["enabled"] = False
        else:
            data["config"]["owner_review"]["allowlist"] = ["other-project"]
        (root / "config.yaml").write_text(yaml.safe_dump(data["config"]))
    elif fault in {"digest", "legacy"}:
        with projects_db.connect_closing(root / "projects.db") as registry:
            if fault == "digest":
                registry.execute("UPDATE project_review_briefs SET source_digest=? WHERE project_id=?", ("0" * 64, project.id))
            else:
                for name in ("source_task_id", "source_run_id", "source_digest"):
                    registry.execute(f"ALTER TABLE project_review_briefs DROP COLUMN {name}")
            registry.commit()
    elif fault in {"body", "foreign"}:
        with kanban_db.connect_closing(review_path, board="project-summaries") as board:
            if fault == "body":
                board.execute("UPDATE tasks SET body=body || ? WHERE id=?", ("\nChanged after acceptance.\n", data["review_task"]))
            else:
                board.execute("UPDATE tasks SET idempotency_key=replace(idempotency_key,?,?) WHERE id=?",
                              (project.id, "other-project", data["review_task"]))
            board.commit()
    elif fault == "prose":
        # The kernel checks only the length of this text and the hint, so a pinned proposal can still fail the prose check.
        from hermes_cli.owner_project_review_origin import accepted_review_source, source_digest

        with kanban_db.connect_closing(review_path, board="project-summaries") as board:
            body = kanban_db.get_task(board, data["review_task"]).body
            run_id, metadata = board.execute("SELECT id,metadata FROM task_runs WHERE task_id=? AND outcome='completed'",
                                             (data["review_task"],)).fetchone()
            metadata = json.loads(metadata)
            metadata["project_review"]["plan_proposal"]["summary"] = "Ask Claude to postpone the room work."
            board.execute("UPDATE task_runs SET metadata=? WHERE id=?", (json.dumps(metadata), run_id))
            board.commit()
        with projects_db.connect_closing(root / "projects.db") as registry:
            registry.execute("UPDATE project_review_briefs SET source_digest=? WHERE project_id=?",
                             (source_digest(body, metadata["project_review"]), project.id))
            registry.commit()
        assert accepted_review_source(project.id) is not None
    else:
        with kanban_db.connect_closing(data["path"], board=project.board_slug) as board:
            task = data["proposal"]["changes"][0]["target"]["task_id"]
            kanban_db.add_comment(board, task, "default", "The venue decision changed again.")
    paths = (root / "projects.db", data["path"], review_path)
    before = tuple(_logical_state(path) for path in paths)
    async with TestClient(TestServer(app())) as client:
        response = await client.get(f"/v1/owner-workspace/projects/{project.slug}/snapshot?capabilities={CAPABILITY}")
        assert response.status == 200
        assert (await response.json())["data"]["raphael_review_advice"] is None
    assert tuple(_logical_state(path) for path in paths) == before
