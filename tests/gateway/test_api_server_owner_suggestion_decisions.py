"""Owner decisions on one pending suggestion from the Decisions feed.

The accept, reject and defer routes act on exactly one pending suggestion the
owner's own Decisions feed lists, addressed by that feed's opaque key. The
owner's reason is kept as the decision's record and nothing is ever applied.
Each test drives the adapter's real routes into the real owner-workspace
kernel inside the per-test ``HERMES_HOME``.
"""

from __future__ import annotations

import contextlib
import json

import pytest
import yaml
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import PlatformConfig
from gateway.platforms.api_server import APIServerAdapter
from hermes_cli import kanban_db, owner_workspace as ow
from hermes_cli.config import get_config_path
from hermes_constants import get_hermes_home

_ROUTES = {"accept": "accepted", "reject": "rejected", "defer": "deferred"}
_EVIDENCE = {
    "schema_version": 1,
    "need": "The workshop outline needs current public evidence.",
    "expected_benefit": "Keep the owner-facing advice current.",
    "requested_scope": {flag: False for flag in kanban_db.RECOMMENDATION_SCOPE_FLAGS},
    "risks": "Low",
    "cost": "No added cost",
    "rollback": "Remove the staged skill configuration.",
}


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
    return project


def _pending_suggestion(project: dict, subject: str) -> str:
    with contextlib.closing(kanban_db.connect(board=project["board"])) as conn:
        return kanban_db.create_recommendation(
            conn,
            project_id=project["project_id"],
            target_profile="raphael-planner",
            recommendation_kind="skill",
            recommendation_subject_id=subject,
            recommendation_label="Add workshop research support",
            recommendation_rationale="The current milestone needs public-source research.",
            recommendation_evidence=_EVIDENCE,
            provenance_authority="project-steward",
        )


def _reloaded(project: dict, suggestion_id: str) -> tuple[dict, list[dict]]:
    """The suggestion's lifecycle and decision records, read afresh from disk."""
    with contextlib.closing(kanban_db.connect(board=project["board"])) as conn:
        row = conn.execute(
            "SELECT * FROM tasks WHERE id = ?", (suggestion_id,)
        ).fetchone()
        decided = [
            json.loads(event["payload"])
            for event in conn.execute(
                "SELECT payload FROM task_events WHERE task_id = ? "
                "AND kind = 'recommendation_decided' ORDER BY id",
                (suggestion_id,),
            )
        ]
    return kanban_db.recommendation_lifecycle_snapshot(row), decided


def _app(api_key: str = "") -> web.Application:
    """The adapter's own Decisions routes, registered as ``connect()`` does."""
    adapter = APIServerAdapter(
        PlatformConfig(enabled=True, extra={"key": api_key} if api_key else {})
    )
    app = web.Application()
    for method, path, handler in adapter._http_route_table():
        if path.startswith("/v1/owner-workspace/decisions"):
            app.router.add_route(method, path, handler)
    return app


async def _listed_suggestion_ref(client: TestClient) -> str:
    feed = await (await client.get("/v1/owner-workspace/decisions")).json()
    [ref] = [
        item["decision_ref"] for item in feed["data"] if item["kind"] == "capability"
    ]
    return ref


@pytest.mark.asyncio
@pytest.mark.parametrize(("action", "decision"), list(_ROUTES.items()))
async def test_each_route_decides_a_listed_suggestion_and_its_reason_survives_a_reload(
    owner, action, decision,
):
    project = _project(owner, "Workshop Pilot")
    suggestion_id = _pending_suggestion(project, "workshop-research")
    reason = f"I {action} this: the workshop needs current sources"
    async with TestClient(TestServer(_app())) as client:
        ref = await _listed_suggestion_ref(client)
        resp = await client.post(
            f"/v1/owner-workspace/decisions/{ref}/{action}", json={"reason": reason},
        )
        body = await resp.json()
        feed = await (await client.get("/v1/owner-workspace/decisions")).json()

    assert resp.status == 200
    assert body == {
        "object": "hermes.owner_workspace.decision",
        "data": {"decision_ref": ref, "decision": decision},
    }
    # Decided, so the feed no longer lists it as waiting on the owner.
    assert feed["data"] == []
    snapshot, decided = _reloaded(project, suggestion_id)
    # Recorded, never applied.
    assert (snapshot["decision"], snapshot["effective_state"]) == (decision, "none")
    [record] = decided
    assert (record["reason"], record["actor"], record["authority"]) == (
        reason, owner.actor, "owner_approved",
    )
    assert not {"governance_task_id", "governance_run_id"} & set(record)


@pytest.mark.asyncio
async def test_a_second_decision_on_the_same_key_is_refused_and_changes_nothing(owner):
    project = _project(owner, "Workshop Pilot")
    suggestion_id = _pending_suggestion(project, "workshop-research")
    async with TestClient(TestServer(_app())) as client:
        ref = await _listed_suggestion_ref(client)
        first = await client.post(
            f"/v1/owner-workspace/decisions/{ref}/defer",
            json={"reason": "Not before the workshop"},
        )
        assert first.status == 200
        decided_once = _reloaded(project, suggestion_id)
        for action in _ROUTES:
            again = await client.post(
                f"/v1/owner-workspace/decisions/{ref}/{action}",
                json={"reason": "Changed my mind"},
            )
            assert again.status == 404
            assert (await again.json())["error"]["code"] == "decision_not_found"

    assert _reloaded(project, suggestion_id) == decided_once


@pytest.mark.asyncio
async def test_an_unknown_or_foreign_key_is_refused_and_changes_nothing(owner):
    project = _project(owner, "Workshop Pilot")
    suggestion_id = _pending_suggestion(project, "workshop-research")
    stranger = ow.OwnerContext(actor="stranger", profile="stranger", session="")
    foreign_project = _project(stranger, "Stranger Pilot")
    foreign_id = _pending_suggestion(foreign_project, "stranger-research")
    # A key the stranger's own feed shows: real, but not this owner's to decide.
    [foreign] = ow.list_owner_decisions(stranger)["data"]
    before = (_reloaded(project, suggestion_id), _reloaded(foreign_project, foreign_id))
    async with TestClient(TestServer(_app())) as client:
        # Unknown, foreign, and a native id, which is never a key.
        for ref in ("decision_" + "0" * 32, foreign["decision_ref"], suggestion_id):
            for action in _ROUTES:
                resp = await client.post(
                    f"/v1/owner-workspace/decisions/{ref}/{action}",
                    json={"reason": "Yes"},
                )
                assert resp.status == 404
                assert (await resp.json())["error"]["code"] == "decision_not_found"

    assert (
        _reloaded(project, suggestion_id), _reloaded(foreign_project, foreign_id)
    ) == before


@pytest.mark.asyncio
@pytest.mark.parametrize("action", list(_ROUTES))
async def test_each_route_keeps_the_feed_checks_and_needs_a_reason(owner, action):
    project = _project(owner, "Workshop Pilot")
    suggestion_id = _pending_suggestion(project, "workshop-research")
    [listed] = ow.list_owner_decisions(owner)["data"]
    path = f"/v1/owner-workspace/decisions/{listed['decision_ref']}/{action}"
    auth = {"Authorization": "Bearer sk-owner-secret"}
    before = _reloaded(project, suggestion_id)
    async with TestClient(TestServer(_app(api_key="sk-owner-secret"))) as client:
        assert (await client.post(path, json={"reason": "Yes"})).status == 401
        for body in ({}, {"reason": "   "}, {"reason": 5}, {"reason": "Yes", "apply": True}):
            assert (await client.post(path, json=body, headers=auth)).status == 400
        _write_owner_workspace_config(enabled=False)
        disabled = await client.post(path, json={"reason": "Yes"}, headers=auth)
        assert disabled.status == 404
        assert (await disabled.json())["error"]["code"] == "owner_workspace_not_enabled"

    assert _reloaded(project, suggestion_id) == before
