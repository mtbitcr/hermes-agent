"""Owner decision examples (contract map section 5, test 4).

Every example comes from the adapter's real Decisions routes on a real board
in the per-test home: ``GET /v1/owner-workspace/decisions`` over
``list_owner_decisions``, and ``POST
/v1/owner-workspace/decisions/{decision_ref}/{action}`` over
``decide_owner_suggestion``. The run-approval items come from the adapter's
own in-memory run statuses, set as the run route sets them while a run waits
for the owner's approval. The list read with ``owner_input_upload_v1`` holds a
card the real apply path declared and the claim held for the owner's files
(card C2 of the named inputs plan). The design choice examples come from
``POST /v1/owner-workspace/decisions/{ref}/choose/{n}`` and the list read
with ``design_options_v1`` (card 2 of the design options plan). The list read
with ``owner_answer_v1`` holds the question of each stopped card (card P2 of
the owner answer plan). What is
asserted is described in tests/contracts/conftest.py.
"""

from __future__ import annotations

import contextlib
import re
from unittest.mock import patch

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import PlatformConfig
from gateway.platforms.api_server import (
    APIServerAdapter,
    _resolve_owner_workspace_run_context,
)
from hermes_cli import kanban_db, owner_named_inputs as oni, owner_questions as oq, owner_workspace as ow, release_ledger
from plugins.dashboard_auth.raphael_workspace import model_policy
from tests.contracts import conftest as contract
from tests.gateway.test_api_server_owner_choices import _designer_card, _ref
from tests.gateway.test_api_server_release import (  # noqa: F401  (root and units are fixtures)
    VIEW as RELEASE_VIEW,
    _accepted,
    _begin,
    _client as _release_client,
    _ledger,
    _merge,
    root,
    units,
)
from tests.gateway.test_api_server_owner_suggestion_decisions import (  # noqa: F401  (owner is a fixture)
    _ROUTES,
    _pending_suggestion,
    _project,
    _write_owner_workspace_config,
    owner,
)
from tests.hermes_cli.test_owner_named_inputs import _apply, _authorize, _plan_change
from tests.hermes_cli.test_owner_workspace import _configured_anthropic, _temporarily_patch

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


@pytest.mark.asyncio
async def test_the_decision_list_read_with_owner_input_upload_names_the_missing_inputs(
    owner, owner_payload_example,
):
    ctx = ow.OwnerContext(actor=owner.actor, profile=owner.profile, session="run_owt")
    project = _project(ctx, "Board Setup")
    with _temporarily_patch(model_policy, "configured_assignment_for", _configured_anthropic):
        applied = _apply(ctx, _authorize(
            *_plan_change(project["project_id"], inputs=[
                {"name": "price list", "type": "csv"},
                {"name": "opening hours", "type": "txt"},
            ]),
            project_id=project["project_id"],
        ))
    (held,) = applied["created_task_ids"]
    with contextlib.closing(kanban_db.connect(board=project["board"])) as conn:
        # The owner gave the opening hours only.
        kanban_db.store_attachment_bytes(
            conn, held, "opening-hours.txt", b"Mon-Fri 9-17\n",
            uploaded_by=oni.OWNER_UPLOADER, board=project["board"],
        )
        assert kanban_db.claim_task(conn, held) is None
        question = kanban_db.create_task(
            conn, title="Choose the opening date", assignee="default",
            project_id=project["project_id"],
        )
        assert kanban_db.block_task(
            conn, question, reason="Which date suits you?", kind="needs_input",
        )
    _adapter, app = _app()
    async with TestClient(TestServer(app)) as client:
        resp = await client.get(
            "/v1/owner-workspace/decisions",
            params={"capabilities": "owner_input_upload_v1"},
        )
        live = {"status": resp.status, "body": await resp.json()}

    saved = owner_payload_example(FAMILY, "decision_list_owner_input_upload", live)

    for answer in (live, saved):
        assert answer["status"] == 200
        (declaring,) = [item for item in answer["body"]["data"] if "missing_inputs" in item]
        (plain,) = [item for item in answer["body"]["data"] if "missing_inputs" not in item]
        assert declaring["kind"] == plain["kind"] == "owner_input"
        assert set(declaring) == set(plain) | {"missing_inputs"}
        # Only the input still missing, with its plain label and the size
        # limit; never a file name.
        assert declaring["missing_inputs"] == [{
            "n": 1, "name": "price list",
            "type_label": oni.TYPE_LABELS["csv"], "max_bytes": oni.MAX_INPUT_BYTES,
        }]


@pytest.mark.asyncio
async def test_the_decision_list_read_with_owner_answer_serves_the_question_of_each_stopped_card(
    owner, owner_payload_example,
):
    """Card P2 of the owner answer plan: the list read with ``owner_answer_v1``
    holds a worker's own question, a review limit stop and a plain stop, each
    with its ``question``; the same list read without the token is today's."""
    project = _project(owner, "Sign Up Pages")
    with contextlib.closing(kanban_db.connect(board=project["board"])) as conn:
        asked, limit, plain = (
            kanban_db.create_task(conn, title=title, assignee="default", project_id=project["project_id"])
            for title in ("Choose the form", "Fix the form", "Pick the colour")
        )
        for task_id, stop, question, choices, words in (
            (asked, "question", "Which sign up page should Raphael build first?", [
                {"n": 1, "sentence": "Build the short form with one field."},
                {"n": 2, "sentence": "Build the long form with every field."},
            ], "optional"),
            (limit, "review_limit", "The reviewer asked for a shorter form.", [], "optional"),
        ):
            claimed = kanban_db.claim_task(conn, task_id)
            assert claimed is not None
            oq.declare_owner_question(
                conn, task_id, stop, question, choices, words,
                board=project["board"], expected_run_id=claimed.current_run_id,
            )
        assert kanban_db.claim_task(conn, plain) is not None
        assert kanban_db.block_task(
            conn, plain, reason="Which colour do you want for the button?", kind="needs_input",
        )
        # The clock is not controlled, so each card gets its own fixed creation
        # time and a later, distinct stop time (the time of its blocked event).
        created_at = {asked: 1_700_000_000, limit: 1_700_001_000, plain: 1_700_002_000}
        stopped_at = {asked: 1_700_100_000, limit: 1_700_101_000, plain: 1_700_102_000}
        for task_id in created_at:
            conn.execute("UPDATE tasks SET created_at = ? WHERE id = ?", (created_at[task_id], task_id))
            changed = conn.execute(
                "UPDATE task_events SET created_at = ? WHERE task_id = ? AND kind = 'blocked'",
                (stopped_at[task_id], task_id),
            ).rowcount
            assert changed == 1
        conn.commit()
    iso = {
        task_id: ow._owner_timestamp(moment)
        for task_id, moment in {**{f"c{k}": v for k, v in created_at.items()}, **stopped_at}.items()
    }
    expected_stop = {"Choose the form": iso[asked], "Fix the form": iso[limit], "Pick the colour": iso[plain]}
    expected_created = {
        "Choose the form": iso[f"c{asked}"], "Fix the form": iso[f"c{limit}"], "Pick the colour": iso[f"c{plain}"],
    }
    assert len(set(expected_stop.values()) | set(expected_created.values())) == 6
    _adapter, app = _app()
    async with TestClient(TestServer(app)) as client:
        resp = await client.get("/v1/owner-workspace/decisions", params={"capabilities": "owner_answer_v1"})
        live = {"status": resp.status, "body": await resp.json()}
        today = await _list(client)

    saved = owner_payload_example(FAMILY, "decision_list_owner_answer", live)

    assert "question" not in str(today)
    # The opted-in item's created_at is when the stop began; the no-token item
    # keeps the card's creation time. Only that and the question may differ.
    assert {item["title"]: item["created_at"] for item in live["body"]["data"]} == expected_stop
    assert {item["title"]: item["created_at"] for item in today["body"]["data"]} == expected_created

    def _without(body: dict, *dropped: str) -> dict:
        return {**body, "data": [
            {key: value for key, value in item.items() if key not in dropped} for item in body["data"]
        ]}

    assert _without(live["body"], "question", "created_at") == _without(today["body"], "created_at")
    for answer in (live, saved):
        assert answer["status"] == 200
        items = {item["title"]: item for item in answer["body"]["data"]}
        assert set(items) == {"Choose the form", "Fix the form", "Pick the colour"}
        assert {title: item["created_at"] for title, item in items.items()} != expected_created
        questions = {title: item["question"] for title, item in items.items()}
        assert {title: question["stop"] for title, question in questions.items()} == {
            "Choose the form": "question", "Fix the form": "review_limit", "Pick the colour": "plain",
        }
        for question in questions.values():
            assert set(question) == {"object", "stop", "stop_ref", "text", "choices", "words", "words_limit"}
            assert question["object"] == "hermes.owner_workspace.owner_question"
            assert re.fullmatch(r"stop_[0-9a-f]{32}", question["stop_ref"])
            assert question["words_limit"] == 1000
        assert len({question["stop_ref"] for question in questions.values()}) == 3
        assert [choice["n"] for choice in questions["Choose the form"]["choices"]] == [1, 2]
        assert questions["Choose the form"]["words"] == "optional"
        assert questions["Fix the form"]["choices"][1] == {"n": 2, "sentence": "Stop this work and release nothing."}
        assert questions["Pick the colour"]["words"] == "required"
        assert questions["Pick the colour"]["choices"] == []
        assert questions["Pick the colour"]["text"] == "Which colour do you want for the button?"
        assert items["Pick the colour"]["reason"] == "Raphael needs your answer before this work can continue."


@pytest.mark.asyncio
async def test_the_design_choice_examples_come_from_the_real_list_and_choose_routes(
    owner, owner_payload_example,
):
    """Platform test 7 of the design options plan: a designer card declared
    its options (card 1) and the owner chose one through the choose route;
    the list read with ``design_options_v1`` holds a second card still
    waiting, with its options, and the record of that choice."""
    project = _project(owner, "Bakery Website")
    decided = _designer_card(project, title="Design the menu page", count=2)
    waiting = _designer_card(project, title="Design the home page")
    _adapter, app = _app()
    async with TestClient(TestServer(app)) as client:
        ref = await _ref(client, decided)
        answers = {}
        for name in ("design_choice_recorded", "design_choice_not_waiting"):
            resp = await client.post(f"/v1/owner-workspace/decisions/{ref}/choose/2")
            answers[name] = {"status": resp.status, "body": await resp.json()}
        resp = await client.get(
            "/v1/owner-workspace/decisions",
            params={"capabilities": "design_options_v1"},
        )
        answers["decision_list_design_options"] = {"status": resp.status, "body": await resp.json()}

    saved = {name: owner_payload_example(FAMILY, name, live) for name, live in answers.items()}

    for live in (answers, saved):
        assert live["design_choice_recorded"] == {"status": 200, "body": {
            "object": "hermes.owner_workspace.design_choice",
            "decision_ref": ref, "n": 2, "released": True,
        }}
        refused = live["design_choice_not_waiting"]
        assert refused["status"] == 404
        assert refused["body"]["error"]["message"] == "This choice is no longer waiting."
        assert refused["body"]["error"]["code"] == "decision_not_found"
        listed = live["decision_list_design_options"]
        assert listed["status"] == 200
        (item,) = listed["body"]["data"]
        assert item["title"] == waiting.title and item["kind"] == "owner_input"
        assert [option["n"] for option in item["design_options"]["options"]] == [1, 2, 3]
        (record,) = listed["body"]["recent_design_choices"]
        assert (record["n"], [option["n"] for option in record["options"]]) == (2, [1, 2])
        assert [option["picture"]["attachment_id"] for option in record["options"]] == decided.attachment_ids


@pytest.mark.asyncio
async def test_the_released_release_example_comes_from_the_real_release_route(
    root, units, owner_payload_example,
):
    """The release decision read after a release ended: its change titles, their count and the
    end time, from the real route over a release record in the per-test home."""
    _merge("a", "Show the release decision")
    batch = _accepted(_merge("b", "Keep the owner page fast"))
    _begin(batch)
    _ledger(release_ledger.finish_release, batch["batch_id"], outcome="released")
    async with _release_client() as client:
        resp = await client.get(RELEASE_VIEW)
        live = {"status": resp.status, "body": await resp.json()}

    saved = owner_payload_example(FAMILY, "release_decision_released", live)

    for answer in (live, saved):
        assert answer["status"] == 200
        release = answer["body"]["release"]
        assert release["state"] == "released" and release["outcome"] == "released"
        assert release["titles"] == [ow.owner_title(member["title"]) for member in batch["members"]]
        assert release["count"] == len(release["titles"]) == len(batch["members"])
        assert type(release["outcome_at"]) is int
        assert release["outcome_at"] > 0
        assert release["batch_id"] == batch["batch_id"]


def test_the_decision_examples_keep_to_every_closed_vocabulary():
    """The saved decision examples, read together, take every value of each
    row of the table; the fixture holds each one to its row."""
    contract.assert_closed_vocabularies_covered(FAMILY)
