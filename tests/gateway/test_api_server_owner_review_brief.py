"""Read the separate review brief through the real owner HTTP snapshot."""

from __future__ import annotations

import contextlib
import json
import logging
import sqlite3
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pytest
import yaml
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import PlatformConfig
from gateway.platforms.api_server import APIServerAdapter
from hermes_cli import kanban_db as kb, owner_workspace as ow, projects_db
from tools import approval

NOW = int(datetime(2026, 11, 10, 7, tzinfo=timezone.utc).timestamp())
SNAPSHOT = "/v1/owner-workspace/projects/{project_slug}/snapshot"
CLOSED = "/v1/owner-workspace/decisions?capabilities=closed_suggestions_v1"
NOT_UNDOABLE = "This closure can no longer be undone."
INVALID_UNDO = "This Undo needs a short reason and the closure it undoes."


@pytest.fixture
def project_review(tmp_path, monkeypatch):
    root = tmp_path / ".hermes"
    root.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(root))
    for name in ("HERMES_PROFILE", "HERMES_KANBAN_HOME", "HERMES_KANBAN_DB", "HERMES_KANBAN_BOARD", "HERMES_KANBAN_TASK"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(time, "time", lambda: NOW)
    (root / "config.yaml").write_text(yaml.safe_dump({
        "gateway": {"api_server": {"owner_workspace": {"enabled": True}}},
    }))
    # The gateway keeps its startup config home; bind that filesystem boundary to this fixture's home.
    monkeypatch.setattr("gateway.run._hermes_home", root)
    ctx = ow.OwnerContext(actor="default", profile="default", session="review-view-test")

    def approve_fixture(request):
        assert approval.resolve_gateway_approval(
            ctx.session, "once", approval_id=request["approval_id"],
        ) == 1

    # Provision through the public approval queue. The snapshot keeps its real owner and receipt checks.
    approval.register_gateway_notify(ctx.session, approve_fixture)
    try:
        project = ow.bootstrap(ctx, idempotency_key="review-view", name="Meeting Plan")
    finally:
        approval.unregister_gateway_notify(ctx.session)
    assert project["ok"]
    project = next(item for item in ow.list_committed_projects(ctx) if item["project_id"] == project["project_id"])
    with projects_db.connect_closing() as registry:
        projects_db.write_project_summary(
            registry, project["project_id"], outcome="The room is booked.", next_action="Send the invitations.",
        )
        projects_db.write_project_review_brief(
            registry, project["project_id"], review_date="2026-11-10", covered_from="2026-11-09", covered_to="2026-11-10",
            headline="The room is booked.", what_changed="The room booking is complete.",
            next_step="Send the invitations.", needs_you=None, ready_work=["Send the invitations"],
            cut_note=None, quiet=False, classification=[{"item": "internal-item", "class": "superseded"}], written_at=NOW,
        )
    return project


def app(api_key=""):
    adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={"key": api_key} if api_key else {}))
    application = web.Application()
    for method, path, handler in adapter._http_route_table():
        if (method == "GET" and path == SNAPSHOT) or path.startswith("/v1/owner-workspace/decisions"):
            application.router.add_route(method, path, handler)
    return application


def _rows(path, sql, *args):
    """An independent plain SQLite read, outside the owner code under test."""
    with contextlib.closing(sqlite3.connect(path)) as conn:
        return conn.execute(sql, args).fetchall()


def _dump(path):
    with contextlib.closing(sqlite3.connect(path)) as conn:
        return list(conn.iterdump())


@pytest.mark.asyncio
async def test_the_brief_capability_adds_one_separate_key_and_preserves_the_five_field_summary(project_review):
    assert ow.OWNER_PROJECT_REVIEW_BRIEF_CAPABILITY == "project_review_brief"
    path = f"/v1/owner-workspace/projects/{project_review['slug']}/snapshot"
    async with TestClient(TestServer(app())) as client:
        old_response = await client.get(path + "?capabilities=project_summary")
        assert old_response.status == 200
        old = (await old_response.json())["data"]
        new_response = await client.get(path + "?capabilities=project_summary,project_review_brief")
        assert new_response.status == 200
        new = (await new_response.json())["data"]

    assert "raphael_review" not in old
    assert set(new) - set(old) == {"raphael_review"}
    assert {key: value for key, value in new.items() if key != "raphael_review"} == old
    assert new["raphael_summary"] == old["raphael_summary"] == {
        "outcome": "The room is booked.", "next_action": "Send the invitations.", "owner_decision": None,
        "written_at": NOW, "revision": 1,
    }
    assert new["raphael_review"] == {
        "latest": {
            "review_date": "2026-11-10", "covered_from": "2026-11-09", "covered_to": "2026-11-10",
            "headline": "The room is booked.", "what_changed": "The room booking is complete.",
            "next_step": "Send the invitations.", "needs_you": None, "ready_work": ["Send the invitations"],
            "cut_note": None, "quiet": False, "written_at": NOW, "revision": 1,
        },
        "history": [{"review_date": "2026-11-10", "headline": "The room is booked."}],
    }


@pytest.mark.asyncio
async def test_a_project_with_no_brief_yet_gets_a_null_review_only_when_it_asks(project_review):
    with projects_db.connect_closing() as registry:
        registry.execute("DELETE FROM project_review_briefs WHERE project_id=?", (project_review["project_id"],))
        registry.commit()
        before = list(registry.iterdump())
    path = f"/v1/owner-workspace/projects/{project_review['slug']}/snapshot"
    async with TestClient(TestServer(app())) as client:
        plain_response = await client.get(path + "?capabilities=project_summary")
        asked_response = await client.get(path + "?capabilities=project_summary,project_review_brief")
        assert plain_response.status == asked_response.status == 200
        plain = (await plain_response.json())["data"]
        asked = (await asked_response.json())["data"]

    assert "raphael_review" not in plain
    assert asked["raphael_review"] is None
    assert asked["raphael_summary"] == plain["raphael_summary"]
    with projects_db.connect_closing() as registry:
        assert list(registry.iterdump()) == before


@pytest.mark.asyncio
async def test_the_brief_history_has_the_latest_fourteen_review_days(project_review):
    with projects_db.connect_closing() as registry:
        for offset in range(1, 16):
            day = (date(2026, 11, 10) - timedelta(days=offset)).isoformat()
            projects_db.write_project_review_brief(
                registry, project_review["project_id"], review_date=day, covered_from=None, covered_to=day,
                headline="The invitations are ready.", what_changed="The invitation list is ready.",
                next_step="Send the invitations.", needs_you=None,
            )
    path = f"/v1/owner-workspace/projects/{project_review['slug']}/snapshot?capabilities=project_review_brief"
    async with TestClient(TestServer(app())) as client:
        response = await client.get(path)
        assert response.status == 200
        body = (await response.json())["data"]

    assert "raphael_summary" not in body
    assert body["raphael_review"]["latest"]["review_date"] == "2026-11-10"
    assert [entry["review_date"] for entry in body["raphael_review"]["history"]] == [
        "2026-11-10", "2026-11-09", "2026-11-08", "2026-11-07", "2026-11-06", "2026-11-05", "2026-11-04",
        "2026-11-03", "2026-11-02", "2026-11-01", "2026-10-31", "2026-10-30", "2026-10-29", "2026-10-28",
    ]
    assert all(set(entry) == {"review_date", "headline"} for entry in body["raphael_review"]["history"])


@pytest.mark.asyncio
async def test_a_legacy_registry_returns_no_brief_and_the_read_does_not_create_its_table(project_review):
    with projects_db.connect_closing() as registry:
        summary = projects_db.get_project_summary(registry, project_review["project_id"])
        registry.execute("DROP TABLE project_review_briefs")
        before = list(registry.iterdump())
        path = f"/v1/owner-workspace/projects/{project_review['slug']}/snapshot?capabilities=project_summary,project_review_brief"
        async with TestClient(TestServer(app())) as client:
            response = await client.get(path)
            assert response.status == 200
            body = (await response.json())["data"]

        assert body["raphael_review"] is None
        assert body["raphael_summary"] == summary
        assert list(registry.iterdump()) == before


def test_a_root_without_a_registry_yet_reads_as_no_brief_and_the_read_creates_nothing(project_review, monkeypatch, tmp_path):
    from hermes_cli.owner_project_review_view import project_review_view

    # A profile reader whose root has never had a review written: the answer is no brief, not an outage.
    empty_root = tmp_path / "empty-root"
    empty_root.mkdir()
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(empty_root))
    assert project_review_view(project_review["project_id"]) is None
    assert list(empty_root.iterdir()) == []


# Owner rule of 2026-11-12: the daily cap's mark in the review state, explained only by this view of the brief.
STORED = {
    "latest": {
        "review_date": "2026-11-10", "covered_from": "2026-11-09", "covered_to": "2026-11-10",
        "headline": "The room is booked.", "what_changed": "The room booking is complete.",
        "next_step": "Send the invitations.", "needs_you": None, "ready_work": ["Send the invitations"],
        "cut_note": None, "quiet": False, "written_at": NOW, "revision": 1,
    },
    "history": [{"review_date": "2026-11-10", "headline": "The room is booked."}],
}
SAVED_EXAMPLE = Path(__file__).resolve().parents[1] / "contracts/owner_payloads/owner_snapshot/project_review_brief.json"


def _explanation(day, covered_from, written_at):
    return {
        "review_date": day, "covered_from": covered_from, "covered_to": day,
        "headline": "The review waits for the next day: the daily limit is reached.",
        "what_changed": (
            "The daily limit of reviews is reached, so this Project's review waits for the next day. "
            "Its news is kept for that review."
        ),
        "next_step": "The review runs on the next day.", "needs_you": None, "ready_work": [], "cut_note": None,
        "quiet": False, "written_at": written_at, "revision": 1,
    }


def _cap_mark(project_id, day, at, last_reviewed_date="2026-11-10"):
    with projects_db.connect_closing() as registry:
        registry.execute(
            "INSERT OR REPLACE INTO project_review_state (project_id, last_reviewed_date, updated_at, cap_wait_date, "
            "cap_wait_at) VALUES (?, ?, ?, ?, ?)", (project_id, last_reviewed_date, NOW, day, at),
        )
        registry.commit()


async def _snapshots(project_review):
    path = f"/v1/owner-workspace/projects/{project_review['slug']}/snapshot"
    async with TestClient(TestServer(app())) as client:
        plain_response = await client.get(path + "?capabilities=project_summary")
        asked_response = await client.get(path + "?capabilities=project_summary,project_review_brief")
        assert plain_response.status == asked_response.status == 200
        return (await plain_response.json())["data"], (await asked_response.json())["data"]


@pytest.mark.asyncio
async def test_a_cap_mark_later_than_the_newest_brief_shows_the_fixed_explanation_in_the_same_payload_shape(
    project_review,
):
    _cap_mark(project_review["project_id"], "2026-11-11", NOW + 86400)
    with projects_db.connect_closing() as registry:
        before = list(registry.iterdump())
    plain, asked = await _snapshots(project_review)

    assert set(asked) - set(plain) == {"raphael_review"}
    assert {key: value for key, value in asked.items() if key != "raphael_review"} == plain
    # The explanation is ``latest``; the history holds the stored briefs only, which stay as they are.
    assert asked["raphael_review"] == {
        "latest": _explanation("2026-11-11", "2026-11-11", NOW + 86400), "history": STORED["history"],
    }
    with projects_db.connect_closing() as registry:
        assert list(registry.iterdump()) == before
    # The saved brief payload example has the same keys and value types, record by record; it is not changed.
    saved = json.loads(SAVED_EXAMPLE.read_text(encoding="utf-8"))["body"]["data"]["raphael_review"]
    served = asked["raphael_review"]
    assert sorted(served) == sorted(saved) and sorted(served["latest"]) == sorted(saved["latest"])
    assert {key: type(value) for key, value in served["latest"].items()} == {
        key: type(value) for key, value in saved["latest"].items()
    }
    assert [sorted(entry) for entry in served["history"]] == [sorted(entry) for entry in saved["history"]]


@pytest.mark.asyncio
@pytest.mark.parametrize("day", ["2026-11-10", "2026-11-09"])
async def test_a_cap_mark_on_or_before_the_day_of_the_newest_brief_leaves_the_brief_unchanged(project_review, day):
    _cap_mark(project_review["project_id"], day, NOW - 60)
    plain, asked = await _snapshots(project_review)

    assert asked["raphael_review"] == STORED
    assert {key: value for key, value in asked.items() if key != "raphael_review"} == plain


@pytest.mark.asyncio
async def test_a_project_with_a_cap_mark_and_no_stored_brief_gets_the_explanation(project_review):
    with projects_db.connect_closing() as registry:
        registry.execute("DELETE FROM project_review_briefs WHERE project_id=?", (project_review["project_id"],))
        registry.commit()
    _cap_mark(project_review["project_id"], "2026-11-10", NOW, last_reviewed_date=None)
    plain, asked = await _snapshots(project_review)

    assert asked["raphael_review"] == {"latest": _explanation("2026-11-10", None, NOW), "history": []}
    assert asked["raphael_summary"] == plain["raphael_summary"]


@pytest.mark.asyncio
async def test_a_store_without_the_cap_mark_columns_reads_as_before_and_the_read_adds_nothing(project_review):
    with projects_db.connect_closing() as registry:
        registry.execute(
            "INSERT INTO project_review_state (project_id, last_reviewed_date, updated_at) VALUES (?, ?, ?)",
            (project_review["project_id"], "2026-11-10", NOW),
        )
        registry.execute("ALTER TABLE project_review_state DROP COLUMN cap_wait_at")
        registry.execute("ALTER TABLE project_review_state DROP COLUMN cap_wait_date")
        registry.commit()
        before = list(registry.iterdump())
    plain, asked = await _snapshots(project_review)

    assert asked["raphael_review"] == STORED
    assert asked["raphael_summary"] == plain["raphael_summary"]
    assert _dump(kb.kanban_home() / "projects.db") == before


@pytest.fixture
def closed_suggestion(project_review):
    with kb.connect_closing(board=project_review["board"]) as board:
        suggestion = kb.create_recommendation(
            board, project_id=project_review["project_id"], target_profile="worker", recommendation_kind="skill",
            recommendation_subject_id="meeting-room", recommendation_label="Book the room",
            recommendation_evidence={
                "schema_version": 1, "need": "Repeated capability gap", "expected_benefit": "Complete the meeting work",
                "requested_scope": {flag: False for flag in kb.RECOMMENDATION_SCOPE_FLAGS},
                "risks": "Low", "cost": "No added cost", "rollback": "Remove the staged configuration",
            },
            provenance_authority="static-analyzer", provenance_ref="finding-123", provenance_observed_at=NOW,
        )
        delivered = kb.create_task(board, title="Book a room", project_id=project_review["project_id"])
        assert kb.complete_task(board, delivered, summary="The room is booked.")
        version = board.execute(
            "SELECT recommendation_lifecycle_version FROM tasks WHERE id=?", (suggestion,),
        ).fetchone()[0]
        assert kb.close_recommendation_delivered(
            board, suggestion, delivered, reason="The room is booked.", review_date="2026-11-10",
            expected_lifecycle_version=version,
        )["status"] == "closed"
    return {**project_review, "suggestion_id": suggestion, "card_id": delivered, "version": version}


@pytest.mark.asyncio
async def test_closed_suggestions_move_to_a_separate_list_only_for_the_named_capability(closed_suggestion):
    assert ow.OWNER_CLOSED_SUGGESTIONS_CAPABILITY == "closed_suggestions_v1"
    project_review = closed_suggestion
    async with TestClient(TestServer(app())) as client:
        old_response = await client.get("/v1/owner-workspace/decisions")
        assert old_response.status == 200
        old = await old_response.json()
        response = await client.get(CLOSED)
        assert response.status == 200
        new = await response.json()

    assert "closed" not in old
    [old_suggestion] = old["data"]
    assert new["data"] == []
    assert new["closed"] == [{
        "decision_ref": old_suggestion["decision_ref"], "project_slug": project_review["slug"],
        "project_name": "Meeting Plan", "title": "Book the room", "card_title": "Book a room",
        "reason": "The room is booked.", "closed_at": NOW,
    }]


@pytest.mark.asyncio
async def test_two_owner_tabs_undo_once_and_a_late_undo_preserves_the_next_owner_decision(closed_suggestion):
    async with TestClient(TestServer(app())) as client:
        listing = await client.get(CLOSED)
        assert listing.status == 200
        [item] = (await listing.json())["closed"]
        ref = item["decision_ref"]
        path = f"/v1/owner-workspace/decisions/{ref}"
        before_undo = await client.post(path + "/accept", json={"reason": "Keep the room work."})
        assert before_undo.status == 404
        sent = {"reason": "The room still needs a check.", "expected_closed_at": item["closed_at"]}
        first = await client.post(path + "/undo_close", json=sent)
        assert first.status == 200
        assert await first.json() == {
            "object": "hermes.owner_workspace.closure_undo", "data": {"decision_ref": ref, "undone": True},
        }
        refreshed = await client.get(CLOSED)
        feed = await refreshed.json()
        assert feed["closed"] == [] and [row["decision_ref"] for row in feed["data"]] == [ref]
        second = await client.post(path + "/undo_close", json=sent)
        assert second.status == 404
        assert (await second.json())["error"] == {
            "message": NOT_UNDOABLE, "type": "invalid_request_error", "param": None, "code": "decision_not_found",
        }
        accepted = await client.post(path + "/accept", json={"reason": "Keep the room work."})
        assert accepted.status == 200
        late = await client.post(path + "/undo_close", json=sent)
        assert late.status == 404

    with kb.connect_closing(board=closed_suggestion["board"]) as board:
        suggestion = closed_suggestion["suggestion_id"]
        assert board.execute(
            "SELECT recommendation_decision FROM tasks WHERE id=?", (suggestion,),
        ).fetchone()[0] == "accepted"
        assert board.execute(
            "SELECT undone_at FROM recommendation_closures WHERE recommendation_id=?", (suggestion,),
        ).fetchone()[0] == NOW
        assert board.execute(
            "SELECT COUNT(*) FROM task_events WHERE task_id=? AND kind='recommendation_closure_undone'", (suggestion,),
        ).fetchone()[0] == 1


@pytest.mark.asyncio
async def test_a_retried_undo_with_its_idempotency_key_gets_the_first_answer_and_undoes_once(closed_suggestion):
    async with TestClient(TestServer(app())) as client:
        listing = await client.get(CLOSED)
        [item] = (await listing.json())["closed"]
        path = f"/v1/owner-workspace/decisions/{item['decision_ref']}/undo_close"
        sent = {"reason": "The room still needs a check.", "expected_closed_at": item["closed_at"]}
        headers = {"Idempotency-Key": "undo-room-retry"}
        first = await client.post(path, json=sent, headers=headers)
        retry = await client.post(path, json=sent, headers=headers)
        assert first.status == retry.status == 200
        assert await retry.json() == await first.json() == {
            "object": "hermes.owner_workspace.closure_undo",
            "data": {"decision_ref": item["decision_ref"], "undone": True},
        }
        changed = await client.post(path, json={**sent, "reason": "Another check."}, headers=headers)
        assert changed.status == 409
        assert (await changed.json())["error"]["code"] == "idempotency_conflict"
        unkeyed = await client.post(path, json=sent)
        assert unkeyed.status == 404
        malformed = await client.post(path, json=sent, headers={"Idempotency-Key": "not a key"})
        assert malformed.status == 400

    with kb.connect_closing(board=closed_suggestion["board"]) as board:
        assert board.execute(
            "SELECT COUNT(*) FROM task_events WHERE task_id=? AND kind='recommendation_closure_undone'",
            (closed_suggestion["suggestion_id"],),
        ).fetchone()[0] == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("body", [
    None, {}, {"reason": "The room needs a check."},
    {"reason": "The room needs a check.", "expected_closed_at": True},
    {"reason": "The room needs a check.", "expected_closed_at": -1},
    {"reason": "The room needs a check.", "expected_closed_at": str(NOW)},
    {"reason": ["The room needs a check."], "expected_closed_at": NOW},
    {"reason": " ", "expected_closed_at": NOW},
    {"reason": "x" * 501, "expected_closed_at": NOW},
    {"reason": "The room needs a check.", "expected_closed_at": NOW, "actor": "another-owner"},
])
async def test_an_invalid_undo_body_leaves_the_board_unchanged(closed_suggestion, body):
    with kb.connect_closing(board=closed_suggestion["board"]) as board:
        before = list(board.iterdump())
    async with TestClient(TestServer(app())) as client:
        listing = await client.get(CLOSED)
        [item] = (await listing.json())["closed"]
        response = await client.post(f"/v1/owner-workspace/decisions/{item['decision_ref']}/undo_close", json=body)
        assert response.status == 400
        error = (await response.json())["error"]
        assert (error["message"], error["code"]) == (INVALID_UNDO, "invalid_argument")
    with kb.connect_closing(board=closed_suggestion["board"]) as board:
        assert list(board.iterdump()) == before


@pytest.mark.asyncio
@pytest.mark.parametrize("authorization", [None, "Bearer another-fixture-key"])
async def test_an_undo_without_the_owner_key_leaves_the_board_unchanged(closed_suggestion, authorization):
    with kb.connect_closing(board=closed_suggestion["board"]) as board:
        before = list(board.iterdump())
    async with TestClient(TestServer(app(api_key="fixture-owner-key"))) as client:
        listing = await client.get(CLOSED, headers={"Authorization": "Bearer fixture-owner-key"})
        assert listing.status == 200
        [item] = (await listing.json())["closed"]
        response = await client.post(
            f"/v1/owner-workspace/decisions/{item['decision_ref']}/undo_close",
            headers={"Authorization": authorization} if authorization else {},
            json={"reason": "The room needs a check.", "expected_closed_at": item["closed_at"]},
        )
        assert response.status == 401
    with kb.connect_closing(board=closed_suggestion["board"]) as board:
        assert list(board.iterdump()) == before


@pytest.mark.asyncio
async def test_the_disabled_workspace_refuses_undo_without_a_board_change(closed_suggestion):
    with kb.connect_closing(board=closed_suggestion["board"]) as board:
        before = list(board.iterdump())
    async with TestClient(TestServer(app())) as client:
        listing = await client.get(CLOSED)
        [item] = (await listing.json())["closed"]
        (Path.home() / ".hermes/config.yaml").write_text(yaml.safe_dump({
            "gateway": {"api_server": {"owner_workspace": {"enabled": False}}},
        }))
        response = await client.post(
            f"/v1/owner-workspace/decisions/{item['decision_ref']}/undo_close",
            json={"reason": "The room needs a check.", "expected_closed_at": item["closed_at"]},
        )
        assert response.status == 404
    with kb.connect_closing(board=closed_suggestion["board"]) as board:
        assert list(board.iterdump()) == before


@pytest.mark.asyncio
async def test_a_worker_context_cannot_use_the_owner_undo_endpoint(closed_suggestion, monkeypatch):
    with kb.connect_closing(board=closed_suggestion["board"]) as board:
        before = list(board.iterdump())
    async with TestClient(TestServer(app())) as client:
        listing = await client.get(CLOSED)
        [item] = (await listing.json())["closed"]
        monkeypatch.setenv("HERMES_KANBAN_TASK", "worker-task-fixture")
        response = await client.post(
            f"/v1/owner-workspace/decisions/{item['decision_ref']}/undo_close",
            json={"reason": "The room needs a check.", "expected_closed_at": item["closed_at"]},
        )
        assert response.status == 403
    with kb.connect_closing(board=closed_suggestion["board"]) as board:
        assert list(board.iterdump()) == before


@pytest.mark.asyncio
async def test_an_unexpected_owner_refusal_answers_503_and_logs_its_code(closed_suggestion, monkeypatch, caplog):
    def unavailable(*_args, **_kwargs):
        raise ow.OwnerWorkspaceError("snapshot_unavailable", "kanban.db is unavailable")

    monkeypatch.setattr(ow, "undo_owner_suggestion_closure", unavailable)
    with caplog.at_level(logging.WARNING, logger="gateway.platforms.api_server_owner_closures"):
        async with TestClient(TestServer(app())) as client:
            response = await client.post(
                "/v1/owner-workspace/decisions/decision_0123/undo_close",
                json={"reason": "The room needs a check.", "expected_closed_at": NOW},
            )
            assert response.status == 503
            assert (await response.json())["error"]["code"] == "owner_workspace_unavailable"
    assert "snapshot_unavailable" in caplog.text
    assert "kanban.db" not in caplog.text


@pytest.mark.asyncio
async def test_an_old_tab_cannot_undo_a_later_closure(closed_suggestion, monkeypatch):
    async with TestClient(TestServer(app())) as client:
        listing = await client.get(CLOSED)
        [old] = (await listing.json())["closed"]
        path = f"/v1/owner-workspace/decisions/{old['decision_ref']}/undo_close"
        sent = {"reason": "The room needs a check.", "expected_closed_at": old["closed_at"]}
        assert (await client.post(path, json=sent)).status == 200

        monkeypatch.setattr(time, "time", lambda: NOW + 1)
        with kb.connect_closing(board=closed_suggestion["board"]) as board:
            later_card = kb.create_task(board, title="Book a room", project_id=closed_suggestion["project_id"])
            assert kb.complete_task(board, later_card, summary="The room is booked.")
            assert kb.close_recommendation_delivered(
                board, closed_suggestion["suggestion_id"], later_card, reason="The room is booked.",
                review_date="2026-11-10", expected_lifecycle_version=closed_suggestion["version"],
            )["status"] == "closed"
            before = list(board.iterdump())

        assert (await client.post(path, json=sent)).status == 404
        refreshed = await client.get(CLOSED)
        [current] = (await refreshed.json())["closed"]
        assert current["decision_ref"] == old["decision_ref"]
        assert current["closed_at"] == NOW + 1
        with kb.connect_closing(board=closed_suggestion["board"]) as board:
            assert list(board.iterdump()) == before
        assert (await client.post(path, json={**sent, "expected_closed_at": current["closed_at"]})).status == 200


@pytest.mark.asyncio
async def test_a_deleted_delivery_card_keeps_the_closure_listed_and_undoable(closed_suggestion):
    with kb.connect_closing(board=closed_suggestion["board"]) as board:
        assert kb.archive_task(board, closed_suggestion["card_id"])
        assert kb.delete_archived_task(board, closed_suggestion["card_id"])
    async with TestClient(TestServer(app())) as client:
        listing = await client.get(CLOSED)
        body = await listing.json()
        # The live closure still hides the suggestion from the pending list, so the closed list must keep it.
        assert body["data"] == []
        [item] = body["closed"]
        assert (item["title"], item["card_title"], item["reason"]) == (
            "Book the room", ow.owner_title(None), "The room is booked.",
        )
        response = await client.post(
            f"/v1/owner-workspace/decisions/{item['decision_ref']}/undo_close",
            json={"reason": "The room needs a check.", "expected_closed_at": item["closed_at"]},
        )
        assert response.status == 200
        refreshed = await (await client.get(CLOSED)).json()
        assert refreshed["closed"] == [] and [row["decision_ref"] for row in refreshed["data"]] == [item["decision_ref"]]


@pytest.mark.asyncio
async def test_another_owner_sees_no_closure_and_cannot_undo_it(closed_suggestion):
    owner = ow.OwnerContext(actor="default", profile="default", session="first-tab")
    other = ow.OwnerContext(actor="another-owner", profile="default", session="other-tab")
    ref = ow._owner_decision_ref(owner, authority="recommendation", native_id=closed_suggestion["suggestion_id"])
    with kb.connect_closing(board=closed_suggestion["board"]) as board:
        before = list(board.iterdump())

    listed = ow.list_owner_decisions(other, closed_suggestions=True)
    assert listed["closed"] == [] and listed["data"] == []
    with pytest.raises(ow.OwnerWorkspaceError) as refused:
        ow.undo_owner_suggestion_closure(
            other, decision_ref=ref, reason="The room needs a check.", expected_closed_at=NOW,
        )
    assert refused.value.code == "decision_not_found"
    with kb.connect_closing(board=closed_suggestion["board"]) as board:
        assert list(board.iterdump()) == before
    assert [item["decision_ref"] for item in ow.list_owner_decisions(owner, closed_suggestions=True)["closed"]] == [ref]


@pytest.mark.asyncio
async def test_a_worker_environment_reads_the_root_registry_and_never_writes_through_its_pinned_board(
    closed_suggestion, monkeypatch, tmp_path,
):
    from hermes_cli.owner_project_review_view import project_review_view

    root = tmp_path / ".hermes"
    project_id, suggestion = closed_suggestion["project_id"], closed_suggestion["suggestion_id"]
    with projects_db.connect_closing() as registry:
        for offset in range(1, 4):
            day = (date(2026, 11, 10) - timedelta(days=offset)).isoformat()
            projects_db.write_project_review_brief(
                registry, project_id, review_date=day, covered_from=None, covered_to=day,
                headline="The invitations are ready.", what_changed="The invitation list is ready.",
                next_step="Send the invitations.", needs_you=None,
            )
    kb.create_board("worker-board")
    with kb.connect_closing(board="worker-board", create=True) as worker:
        worker_task = kb.create_task(worker, title="Check the venue")
    # Each store by its own path under the root, derived here without the code under test.
    registry_path = root / "projects.db"
    project_board = root / "kanban" / "boards" / closed_suggestion["board"] / "kanban.db"
    worker_board = root / "kanban" / "boards" / "worker-board" / "kanban.db"
    assert registry_path.is_file() and project_board.is_file() and worker_board.is_file()
    profile_home = root / "profiles" / "worker"
    profile_home.mkdir(parents=True)

    # The environment the dispatcher injects into a worker: its own board pinned, its task set, a profile home.
    monkeypatch.setenv("HERMES_HOME", str(profile_home))
    monkeypatch.setenv("HERMES_KANBAN_DB", str(worker_board))
    monkeypatch.setenv("HERMES_KANBAN_BOARD", "worker-board")
    monkeypatch.setenv("HERMES_KANBAN_TASK", worker_task)
    before = {path: _dump(path) for path in (registry_path, project_board, worker_board)}

    view = project_review_view(project_id)
    [(count,)] = _rows(registry_path, "SELECT COUNT(*) FROM project_review_briefs WHERE project_id=?", project_id)
    headlines = _rows(
        registry_path, "SELECT review_date, headline FROM project_review_briefs WHERE project_id=? "
        "ORDER BY review_date DESC", project_id,
    )
    assert count == 4
    assert [(entry["review_date"], entry["headline"]) for entry in view["history"]] == headlines
    assert (view["latest"]["review_date"], view["latest"]["headline"]) == headlines[0]

    owner = ow.OwnerContext(actor="default", profile="default", session="worker-environment")
    ref = ow._owner_decision_ref(owner, authority="recommendation", native_id=suggestion)
    with pytest.raises(ow.OwnerWorkspaceError) as refused:
        ow.undo_owner_suggestion_closure(owner, decision_ref=ref, reason="The room needs a check.", expected_closed_at=NOW)
    assert refused.value.code == "permission_denied"
    assert {path: _dump(path) for path in before} == before

    # An owner process that inherited the pinned board still writes only the Project's own board.
    monkeypatch.delenv("HERMES_KANBAN_TASK")
    monkeypatch.setenv("HERMES_HOME", str(root))
    async with TestClient(TestServer(app())) as client:
        [item] = (await (await client.get(CLOSED)).json())["closed"]
        assert item["decision_ref"] == ref
        response = await client.post(
            f"/v1/owner-workspace/decisions/{ref}/undo_close",
            json={"reason": "The room needs a check.", "expected_closed_at": item["closed_at"]},
        )
        assert response.status == 200
    assert _rows(
        project_board, "SELECT COUNT(*) FROM recommendation_closures WHERE recommendation_id=? AND undone_at=?",
        suggestion, NOW,
    ) == [(1,)]
    assert _rows(
        project_board, "SELECT COUNT(*) FROM task_events WHERE task_id=? AND kind='recommendation_closure_undone'",
        suggestion,
    ) == [(1,)]
    assert _dump(worker_board) == before[worker_board]
    assert _dump(registry_path) == before[registry_path]


@pytest.mark.asyncio
async def test_a_legacy_board_keeps_pending_suggestions_and_gains_no_closure_table(closed_suggestion):
    with kb.connect_closing(board=closed_suggestion["board"]) as board:
        board.execute("DROP TABLE recommendation_closures")
        before = list(board.iterdump())
        async with TestClient(TestServer(app())) as client:
            old = await client.get("/v1/owner-workspace/decisions")
            old_body = await old.json()
            response = await client.get(CLOSED)
            assert response.status == 200
            body = await response.json()
            assert body["closed"] == []
            assert {key: value for key, value in body.items() if key != "closed"} == old_body
            [item] = body["data"]
            refused = await client.post(
                f"/v1/owner-workspace/decisions/{item['decision_ref']}/undo_close",
                json={"reason": "The room needs a check.", "expected_closed_at": NOW},
            )
            assert refused.status == 404
        assert list(board.iterdump()) == before


@pytest.mark.asyncio
async def test_the_closed_window_keeps_the_latest_twenty_and_cannot_hide_an_older_pending_suggestion(project_review, monkeypatch):
    evidence = {
        "schema_version": 1, "need": "Repeated capability gap", "expected_benefit": "Complete the meeting work",
        "requested_scope": {flag: False for flag in kb.RECOMMENDATION_SCOPE_FLAGS},
        "risks": "Low", "cost": "No added cost", "rollback": "Remove the staged configuration",
    }
    with kb.connect_closing(board=project_review["board"]) as board:
        monkeypatch.setattr(time, "time", lambda: NOW - 36 * 86400)
        kb.create_recommendation(
            board, project_id=project_review["project_id"], target_profile="worker", recommendation_kind="skill",
            recommendation_subject_id="keep-open", recommendation_label="Call the speaker", recommendation_evidence=evidence,
            provenance_authority="static-analyzer", provenance_ref="finding-open", provenance_observed_at=int(time.time()),
        )
        # These synthetic times test the list window. They do not prove any real daily review period.
        for number in range(101):
            when = NOW - 35 * 86400 + number * 8 * 3600
            monkeypatch.setattr(time, "time", lambda: when)
            suggestion = kb.create_recommendation(
                board, project_id=project_review["project_id"], target_profile="worker", recommendation_kind="skill",
                recommendation_subject_id=f"room-{number}", recommendation_label="Book the room", recommendation_evidence=evidence,
                provenance_authority="static-analyzer", provenance_ref=f"finding-{number}", provenance_observed_at=when,
            )
            card = kb.create_task(board, title="Book a room", project_id=project_review["project_id"])
            assert kb.complete_task(board, card, summary="The room is booked.")
            version = board.execute(
                "SELECT recommendation_lifecycle_version FROM tasks WHERE id=?", (suggestion,),
            ).fetchone()[0]
            day = datetime.fromtimestamp(when, timezone.utc).date().isoformat()
            assert kb.close_recommendation_delivered(
                board, suggestion, card, reason="The room is booked.", review_date=day,
                expected_lifecycle_version=version,
            )["status"] == "closed"

    monkeypatch.setattr(time, "time", lambda: NOW)
    async with TestClient(TestServer(app())) as client:
        old = await client.get("/v1/owner-workspace/decisions")
        old_body = await old.json()
        assert old_body["truncated"] is True
        assert all(item["title"] != "Call the speaker" for item in old_body["data"])
        response = await client.get(CLOSED)
        assert response.status == 200
        body = await response.json()

    assert body["truncated"] is False
    assert [item["title"] for item in body["data"]] == ["Call the speaker"]
    assert len(body["closed"]) == 20
    times = [item["closed_at"] for item in body["closed"]]
    assert times == sorted(times, reverse=True)
    assert times[0] == NOW - 40 * 3600
    assert times[-1] == NOW - 192 * 3600
