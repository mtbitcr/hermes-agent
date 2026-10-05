"""Owner decision examples (contract map section 5, test 4).

Every example comes from the adapter's real Decisions routes on a real board
in the per-test home: ``GET /v1/owner-workspace/decisions`` over
``list_owner_decisions``, and ``POST
/v1/owner-workspace/decisions/{decision_ref}/{action}`` over
``decide_owner_suggestion``. The run-approval items come from the adapter's
own in-memory run statuses, set as the run route sets them while a run waits
for the owner's approval. What is asserted is described in
tests/contracts/conftest.py.
"""

from __future__ import annotations

import contextlib
from unittest.mock import patch

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import PlatformConfig
from gateway.platforms.api_server import (
    APIServerAdapter,
    _resolve_owner_workspace_run_context,
)
from hermes_cli import kanban_db, owner_workspace as ow
from tests.contracts import conftest as contract
from tests.gateway.test_api_server_owner_suggestion_decisions import (  # noqa: F401  (owner is a fixture)
    _ROUTES,
    _pending_suggestion,
    _project,
    _write_owner_workspace_config,
    owner,
)

FAMILY = "owner_decisions"
# Each operation a waiting run can name in the inbox (api_server.py:13205-13213).
OPERATIONS = (
    "owner_workspace_bootstrap", "owner_task_graph_commit", "owner_project_plan_commit",
    "owner_task_move", "owner_task_comment", "owner_task_retry", "owner_project_lifecycle",
)


def _app() -> tuple[APIServerAdapter, web.Application]:
    """The adapter and its own Decisions routes, registered as ``connect()`` does."""
    adapter = APIServerAdapter(PlatformConfig(enabled=True))
    app = web.Application()
    for method, path, handler in adapter._http_route_table():
        if path.startswith("/v1/owner-workspace/decisions"):
            app.router.add_route(method, path, handler)
    return adapter, app


def _waiting_run(adapter: APIServerAdapter, run_id: str, operation: str, context: dict) -> None:
    """A run held for the owner's approval, its status set as the run route sets it."""
    adapter._set_run_status(run_id, "queued", owner_workspace_context=context)
    adapter._set_run_status(
        run_id, "waiting_for_approval", last_event="approval.request",
        pending_approval={
            "approval_id": f"approval-{run_id}", "description": "Approve this operation",
            "choices": ["once", "deny"], "operation": operation,
        },
    )


async def _list(client: TestClient) -> dict:
    resp = await client.get("/v1/owner-workspace/decisions")
    return {"status": resp.status, "body": await resp.json()}


async def _decide(client: TestClient, ref: str, action: str, body: dict) -> dict:
    resp = await client.post(f"/v1/owner-workspace/decisions/{ref}/{action}", json=body)
    return {"status": resp.status, "body": await resp.json()}


@pytest.mark.asyncio
async def test_the_decision_examples_carry_every_item_kind_answer_and_error(
    owner, owner_payload_example,
):
    project = _project(owner, "Workshop Pilot")
    for action in _ROUTES:
        _pending_suggestion(project, f"workshop-{action}")
    with contextlib.closing(kanban_db.connect(board=project["board"])) as conn:
        question = kanban_db.create_task(
            conn, title="Choose the workshop date", assignee="default",
            project_id=project["project_id"],
        )
        assert kanban_db.block_task(
            conn, question, reason="Which date suits the venue?", kind="needs_input",
        )
    [slug] = [item["slug"] for item in ow.list_committed_projects(owner)]
    existing = _resolve_owner_workspace_run_context(
        {"mode": "existing", "project_slug": slug, "project_name": None})
    # A new Project's approval has a name and no slug yet.
    new = _resolve_owner_workspace_run_context(
        {"mode": "new", "project_slug": None, "project_name": "Workshop Sequel"})
    adapter, app = _app()
    for index, operation in enumerate(OPERATIONS):
        context = new if operation == "owner_workspace_bootstrap" else existing
        _waiting_run(adapter, f"run_{index}", operation, context)

    async with TestClient(TestServer(app)) as client:
        live = {"decision_list": await _list(client)}
        refs = [
            item["decision_ref"] for item in live["decision_list"]["body"]["data"]
            if item["kind"] == "capability"
        ]
        for action, ref in zip(_ROUTES, refs):
            live[f"decision_{_ROUTES[action]}"] = await _decide(
                client, ref, action, {"reason": f"I {action} this for the workshop."})
        decided = refs[0]
        live["decision_not_found"] = await _decide(client, decided, "accept", {"reason": "Yes"})
        too_long = "x" * (kanban_db._RECOMMENDATION_LIFECYCLE_TEXT_MAX_LEN + 1)
        live["invalid_argument"] = await _decide(client, decided, "accept", {"reason": too_long})
        live["invalid_request"] = await _decide(
            client, decided, "accept", {"reason": "Yes", "apply": True})
        with patch.object(ow, "list_owner_decisions", side_effect=RuntimeError("x")):
            live["decision_list_unavailable"] = await _list(client)
        unavailable = ow.OwnerWorkspaceError("snapshot_unavailable", "unreadable")
        with patch.object(ow, "decide_owner_suggestion", side_effect=unavailable):
            live["decision_unavailable"] = await _decide(client, decided, "accept", {"reason": "Yes"})
        with patch.object(ow, "decide_owner_suggestion", side_effect=RuntimeError("x")):
            live["decision_unavailable_unexpected"] = await _decide(
                client, decided, "accept", {"reason": "Yes"})
        # More waiting runs than the inbox window holds.
        for index in range(ow._OWNER_DECISIONS_LIMIT):
            _waiting_run(adapter, f"run_more_{index}", "owner_task_move", existing)
        live["decision_list_truncated"] = await _list(client)
        _write_owner_workspace_config(enabled=False)
        live["decision_list_not_enabled"] = await _list(client)
        live["decision_not_enabled"] = await _decide(client, decided, "accept", {"reason": "Yes"})
    pending = len(adapter._run_statuses) + 1  # the waiting runs and the open question

    saved = {kind: owner_payload_example(FAMILY, kind, answer) for kind, answer in live.items()}

    for answers in (live, saved):
        listed = answers["decision_list"]["body"]
        assert listed["truncated"] is False
        assert {item["kind"] for item in listed["data"]} == {
            "capability", "owner_input", "run_approval"}
        approvals = [item for item in listed["data"] if item["kind"] == "run_approval"]
        assert len({item["title"] for item in approvals}) == len(approvals) == len(OPERATIONS)
        assert {item["project_slug"] for item in approvals} == {None} | {
            item["project_slug"] for item in listed["data"] if item["authority"] != "run"}
        answered = [answers[f"decision_{decision}"]["body"]["data"] for decision in _ROUTES.values()]
        assert [data["decision"] for data in answered] == list(_ROUTES.values())
        assert {data["decision_ref"] for data in answered} == {
            item["decision_ref"] for item in listed["data"] if item["kind"] == "capability"}
        clipped = answers["decision_list_truncated"]["body"]
        assert clipped["truncated"] is True and len(clipped["data"]) < pending
        # A decided suggestion leaves the inbox.
        assert "capability" not in {item["kind"] for item in clipped["data"]}
        for answer in answers.values():
            assert (answer["status"] >= 400) == ("error" in answer["body"])
        assert answers["invalid_request"]["body"]["error"]["code"] is None


def test_the_decision_examples_keep_to_every_closed_vocabulary():
    """The saved decision examples, read together, take every value of each
    row of the table; the fixture holds each one to its row."""
    contract.assert_closed_vocabularies_covered(FAMILY)
