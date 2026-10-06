"""Owner snapshot examples (contract map section 5, test 1).

Every example comes from the adapter's real ``GET
/v1/owner-workspace/projects/{slug}/snapshot`` route over
``read_project_snapshot``, on real boards in the per-test home, except the run
receipts, which come from ``_owner_project_run_projection``, the helper the
snapshot builds each run with. What is asserted is described in
tests/contracts/conftest.py; each closed vocabulary is checked wherever it
occurs, in the live answers and in every saved example alike, and the values
each takes must agree.
"""

from __future__ import annotations

import contextlib
import json
import os
from unittest.mock import patch

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import PlatformConfig
from gateway.platforms.api_server import APIServerAdapter
from hermes_cli import kanban_db, owner_workspace as ow, projects_db
from hermes_constants import get_hermes_home
from tests.contracts import conftest as contract
from tests.gateway.test_api_server_owner_suggestion_decisions import (  # noqa: F401  (owner is a fixture)
    _project,
    _write_owner_workspace_config,
    owner,
)
from tests.gateway.test_api_server_runs import _capability_stopped_task
from tests.hermes_cli.test_owner_project_removal import (
    _call,
    _n,
    _real_board,
    _start_and_join,
)
from tests.hermes_cli.test_owner_provider_wait import (  # noqa: F401  (root and clock are fixtures)
    STEWARD,
    _waiting_and_refused,
    clock,
    root,
)
from tests.hermes_cli.test_owner_workspace import (
    _capability_receipt_metadata,
    _capability_run,
    _ended_run,
    _gave_up_task,
)

FAMILY = "owner_snapshot"
V1 = ",".join((
    ow.OWNER_PROJECT_RUN_CONTEXT_CAPABILITY,
    ow.OWNER_PROJECT_PLANNING_CONTEXT_CAPABILITY,
    ow.OWNER_WAITING_CAPABILITY,
    ow.PROVIDER_WAIT_CAPABILITY,
))
V2 = f"{V1},{ow.OWNER_PROJECT_PLANNING_CONTEXT_V2_CAPABILITY}"


def _app() -> web.Application:
    """The adapter's own Project snapshot route, registered as ``connect()`` does."""
    adapter = APIServerAdapter(PlatformConfig(enabled=True))
    app = web.Application()
    for method, path, handler in adapter._http_route_table():
        if path.endswith("/snapshot"):
            app.router.add_route(method, path, handler)
    return app


async def _snapshot(client: TestClient, slug: str, capabilities: str = "") -> dict:
    params = {"capabilities": capabilities} if capabilities else None
    resp = await client.get(
        f"/v1/owner-workspace/projects/{slug}/snapshot", params=params,
    )
    return {"status": resp.status, "body": await resp.json()}


def _slug(owner: ow.OwnerContext, project: dict) -> str:
    return next(
        item["slug"] for item in ow.list_committed_projects(owner)
        if str(item["project_id"]) == project["project_id"]
    )


def _task(board: str, project: dict, title: str, **fields) -> str:
    with contextlib.closing(kanban_db.connect(board=board)) as conn:
        return kanban_db.create_task(
            conn, title=title, assignee="default", project_id=project["project_id"],
            **fields,
        )


def _tasks(example: dict) -> list[dict]:
    return [
        task for column in example["body"]["data"]["columns"] for task in column["tasks"]
    ]


@pytest.mark.asyncio
async def test_the_snapshot_examples_carry_every_task_state_and_capability(
    owner, owner_payload_example,
):
    project = _project(owner, "Workshop Pilot")
    board, slug = project["board"], _slug(owner, project)
    approved = _task(board, project, "Review the approved outline")
    archived = _task(board, project, "Draft the old agenda")
    with contextlib.closing(kanban_db.connect(board=board)) as conn:
        assert kanban_db.request_review(conn, approved, summary="Ready.")
        # A completion or an archive promotes a given-up card back to ready, so
        # they go first; an archived card stays in the planning context.
        assert kanban_db.complete_task(conn, approved, result="Approved.")
        assert kanban_db.archive_task(conn, archived)
    # The breaker trips on every ready or review card, so this comes next.
    _gave_up_task(board, project["project_id"], "Print the workshop handouts")
    _capability_stopped_task(board, project["project_id"], "Connect the payment provider")
    drafted = _task(board, project, "Draft the workshop agenda", risk_tier=0)
    with contextlib.closing(kanban_db.connect(board=board)) as conn:
        with kanban_db.write_txn(conn):
            conn.execute("UPDATE tasks SET risk_tier = 1 WHERE id = ?", (drafted,))
            kanban_db._append_event(
                conn, drafted, "risk_tier_raised",
                {"from": 0, "to": 1, "reviewer": "default", "run_id": 1},
            )
    reviews = {
        state: _task(board, project, f"Review the {state} outline")
        for state in ("awaiting", "changes")
    }
    working = _task(board, project, "Rehearse the workshop")
    with contextlib.closing(kanban_db.connect(board=board)) as conn:
        for task_id in reviews.values():
            assert kanban_db.request_review(conn, task_id, summary="Ready.")
        assert kanban_db.claim_review_task(conn, reviews["changes"]) is not None
        assert kanban_db.request_changes(
            conn, reviews["changes"], reason="Name the date first.",
        )[0]
        # A live claim held by this process is a verified worker, and the file
        # its run attaches is bound to that run by the attachment receipt.
        assert kanban_db.claim_task(conn, working) is not None
        kanban_db._set_worker_pid(conn, working, os.getpid())
        run_id = int(kanban_db.get_task(conn, working).current_run_id)
        kanban_db.store_attachment_bytes(
            conn, working, "plan.md", b"# Plan\n", content_type="text/markdown",
            board=board, expected_run_id=run_id,
        )
    # A card after unfinished work is planned, and one put off is scheduled.
    _task(board, project, "Book the workshop room", parents=(working,))
    scheduled = _task(board, project, "Send the invitations")
    with contextlib.closing(kanban_db.connect(board=board)) as conn:
        assert kanban_db.schedule_task(conn, scheduled, reason="After the venue answers.")

    async with TestClient(TestServer(_app())) as client:
        live = {
            "project_snapshot": await _snapshot(client, slug, V1),
            "project_snapshot_planning_context_v2": await _snapshot(client, slug, V2),
            "project_snapshot_without_capabilities": await _snapshot(client, slug),
            "project_not_found": await _snapshot(client, "no-such-project"),
        }
        unavailable = ow.OwnerWorkspaceError("snapshot_unavailable", "unreadable")
        with patch.object(ow, "read_project_snapshot", side_effect=unavailable):
            live["owner_workspace_unavailable"] = await _snapshot(client, slug)
        with patch.object(ow, "read_project_snapshot", side_effect=RuntimeError("x")):
            live["owner_workspace_unavailable_unexpected"] = await _snapshot(client, slug)
        _write_owner_workspace_config(enabled=False)
        live["owner_workspace_not_enabled"] = await _snapshot(client, slug)

    saved = {kind: owner_payload_example(FAMILY, kind, body) for kind, body in live.items()}

    for answers in (live, saved):
        tasks = _tasks(answers["project_snapshot"])
        assert any("owner_wait" in task for task in tasks)
        assert all(type(task["risk_tier_raised"]) is bool for task in tasks)
        assert {task["risk_tier"] for task in tasks} == {1, None}
        assert [task["risk_tier_raised"] for task in tasks].count(True) == 1
        assert not any("owner_wait" in task for task in _tasks(
            answers["project_snapshot_without_capabilities"]))
        data = answers["project_snapshot"]["body"]["data"]
        assert data["workers"] and data["attachments"]
        for kind in ("project_snapshot", "project_snapshot_planning_context_v2"):
            assert "planning_context" in answers[kind]["body"]["data"]
        assert "planning_context" not in (
            answers["project_snapshot_without_capabilities"]["body"]["data"])
        assert set(data["truncated"]) == {"tasks", "workers", "attachments", "runs"}
        errors = [answer for answer in answers.values() if "error" in answer["body"]]
        assert {answer["body"]["error"]["code"] for answer in errors} == {
            "project_not_found", "owner_workspace_unavailable", "owner_workspace_not_enabled",
        }
        assert all(answer["status"] >= 400 for answer in errors)


@pytest.mark.asyncio
async def test_the_steward_examples_carry_every_steward_state(
    owner, owner_payload_example, monkeypatch,
):
    projects = {name: _project(owner, f"{name.title()} Pilot") for name in (
        "fresh", "working", "complete", "paused", "finishing", "attention", "waiting",
    )}
    for name in ("working", "attention", "waiting"):
        kanban_db.write_board_dispatch_state(projects[name]["board"], dispatch_enabled=True)
    _gave_up_task(projects["attention"]["board"], projects["attention"]["project_id"],
                  "Print the handouts")
    for name in ("working", "complete", "paused", "finishing"):
        _task(projects[name]["board"], projects[name], "Draft the agenda")
    with contextlib.closing(kanban_db.connect(board=projects["complete"]["board"])) as conn:
        [done] = [task.id for task in kanban_db.list_tasks(conn)
                  if task.project_id == projects["complete"]["project_id"]]
        assert kanban_db.complete_task(conn, done, result="Done.")
    # A pause leaves the work already underway to finish.
    with contextlib.closing(kanban_db.connect(board=projects["finishing"]["board"])) as conn:
        [underway] = [task.id for task in kanban_db.list_tasks(conn)
                      if task.project_id == projects["finishing"]["project_id"]]
        assert kanban_db.claim_task(conn, underway) is not None
        kanban_db._set_worker_pid(conn, underway, os.getpid())
    for name in ("paused", "finishing"):
        kanban_db.write_board_metadata(
            projects[name]["board"], dispatch_enabled=False, dispatch_paused_by_owner=True,
        )
    _capability_stopped_task(projects["waiting"]["board"],
                             projects["waiting"]["project_id"], "Connect the provider")
    project_id, deleted_slug = _real_board(monkeypatch)
    _start_and_join(project_id, "contract")
    # A recoverable removal confirmed as permanent leaves no copy to restore.
    project_id, permanent_slug = _real_board(monkeypatch)
    _start_and_join(project_id, "contract")
    with projects_db.connect_closing() as conn:
        served = ow._project_removal_state(conn, project_id, permanent_slug)
    assert _call(project_id, f"permanent{_n[0]}", "confirm_permanent",
                 consequences_digest=served["consequences_digest"])["ok"]
    for thread in list(ow._removal_background_threads):
        thread.join(timeout=60)

    async with TestClient(TestServer(_app())) as client:
        live = [
            (await _snapshot(client, _slug(owner, project), V1))["body"]["data"]["steward"]
            for project in projects.values()
        ]
        deleted = await _snapshot(client, deleted_slug, V1)
        permanent = await _snapshot(client, permanent_slug, V1)

    saved = owner_payload_example(FAMILY, "steward_states", live)
    saved_deleted = owner_payload_example(FAMILY, "deleted_project", deleted)
    saved_permanent = owner_payload_example(FAMILY, "permanently_deleted_project", permanent)
    for removed in (deleted, saved_deleted, permanent, saved_permanent):
        data = removed["body"]["data"]
        assert data["steward"]["removal_state"] == data["removal_state"]
    for stewards in (live, saved):
        summaries = [steward["execution"]["summary"] for steward in stewards]
        assert len(set(summaries)) == len(stewards)
    for removed in (permanent, saved_permanent):
        assert removed["body"]["data"]["removal_state"]["restorable"] is False
    for removed in (deleted, saved_deleted):
        assert removed["body"]["data"]["removal_state"]["restorable"] is True


@pytest.mark.asyncio
async def test_the_provider_wait_example_carries_the_held_card_and_its_waiting_run(
    root, clock, monkeypatch, owner_payload_example,
):
    monkeypatch.setenv("HERMES_HOME", str(root / "profiles" / STEWARD))
    scene = _waiting_and_refused(root, ow.resolve_owner_context())
    clock["t"] = scene["held"].ended_at + 60
    monkeypatch.setattr("gateway.run._hermes_home", get_hermes_home())
    _write_owner_workspace_config(enabled=True)

    async with TestClient(TestServer(_app())) as client:
        live = await _snapshot(client, scene["project"]["slug"], V1)

    saved = owner_payload_example(FAMILY, "provider_wait", live)
    for answer in (live, saved):
        assert ow._OWNER_STOPPED_WORK_PROVIDER_WAIT in contract.closed_values(
            _tasks(answer), ("stopped_work",))
        assert "waiting" in contract.closed_values(
            answer["body"]["data"]["runs"], ("receipt", "outcome"))


def test_the_run_receipt_examples_carry_every_outcome_and_summary(owner_payload_example):
    resume_at = 1_790_179_200
    cases = [
        (_ended_run("running", None, ended_at=None), {}),
        (_ended_run("completed", None), {"owner_retry_reason": "The key was added."}),
        (_ended_run("review_requested", None), {}),
        (_ended_run("scheduled", None), {}),
        (_ended_run("rate_limited", None),
         {"provider_wait": True, "provider_resume_at": resume_at}),
        (_ended_run("rate_limited", None), {}),
        (_ended_run("rate_limited", {"rate_limit_reset_at": resume_at}), {}),
        (_ended_run("provider_refused", None), {"provider_wait": True}),
        (_ended_run("crashed", None), {}),
        (_ended_run("changes_requested", None), {"owner_waiting": True}),
        (_ended_run("blocked", None), {"owner_waiting": True}),
        (_ended_run("cancelled", None), {}),
    ]
    origins = [
        kanban_db.RETRY_ORIGIN_NONE, kanban_db.RETRY_ORIGIN_AUTOMATIC,
        kanban_db.RETRY_ORIGIN_OWNER, kanban_db.RETRY_ORIGIN_UNATTRIBUTED,
    ]
    live = [
        ow._owner_project_run_projection(
            run, "Draft the workshop agenda", task_pin=None,
            has_newer_run=origins[index % 4] != kanban_db.RETRY_ORIGIN_NONE,
            retry_origin=origins[index % 4], run_context=True, **options,
        )
        for index, (run, options) in enumerate(cases)
    ]

    saved = owner_payload_example(FAMILY, "run_receipts", live)
    assert {run["receipt"]["summary"] for run in saved} == {
        run["receipt"]["summary"] for run in live
    }


def test_the_run_receipt_route_examples_carry_every_runtime_and_cost_state(
    owner_payload_example,
):
    """Native runtime receipts, projected as the snapshot projects each run.

    Version 4 is not a receipt version the owner reads, so its route is
    unknown; a receipt without a cost keeps its route; the capability object
    is read from version 3 receipts only.
    """
    receipts = [_capability_receipt_metadata(4), _capability_receipt_metadata(2)]
    del receipts[-1]["runtime_receipt"]["cost"]
    for state in ("estimated", "exact", "reported", "included"):
        receipts.append(_capability_receipt_metadata(2))
        receipts[-1]["runtime_receipt"]["cost"]["state"] = state
    for source in ("session-tool-calls", "unavailable"):
        receipts.append(_capability_receipt_metadata(3, {
            "schema_version": 1, "skills": [], "skills_truncated": False,
            "tools": ["read_file"], "tools_truncated": False, "connections": [],
            "connections_truncated": False, "source": source, "truncated": False,
        }))
    live = [
        ow._owner_project_run_projection(
            _capability_run(metadata), "Draft the workshop agenda", task_pin=None,
            has_newer_run=False, run_context=True,
        )
        for metadata in receipts
    ]

    saved = owner_payload_example(FAMILY, "run_receipt_routes", live)
    for runs in (live, saved):
        runtimes = [run["receipt"]["runtime"] for run in runs]
        assert [
            "capability" in runtime for runtime in runtimes
        ] == [metadata["runtime_receipt"]["schema_version"] == 3 for metadata in receipts]
    assert {run["receipt"]["cost"]["summary"] for run in saved} == {
        run["receipt"]["cost"]["summary"] for run in live
    }


def test_the_snapshot_examples_keep_to_every_closed_vocabulary():
    """The saved snapshot examples, read together, take every value of each
    row of the table; the fixture holds each one to its row."""
    contract.assert_closed_vocabularies_covered(FAMILY)


def _saved(record_kind: str):
    path = contract.OWNER_PAYLOADS / FAMILY / f"{record_kind}.json"
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.mark.parametrize("case", ["live", "saved", "second_record"])
def test_a_value_outside_its_row_fails_the_closed_vocabulary_check(case):
    """One allowed value, changed in memory to one outside its row, fails the
    check, which names the payload, the concrete path and the value: in a live
    run receipt, in the saved snapshot and in the second steward of a list."""
    if case == "live":
        kind, payload = "run_receipts", [ow._owner_project_run_projection(
            _ended_run("completed", None), "Draft the workshop agenda", task_pin=None,
            has_newer_run=False, run_context=True,
        )]
        record, key, where = payload[0]["receipt"], "outcome", "[0].receipt.outcome"
    elif case == "saved":
        kind = "project_snapshot"
        payload = _saved(kind)
        record, key = payload["body"]["data"]["runs"][0], "retry_origin"
        where = "body.data.runs[0].retry_origin"
    else:
        kind = "steward_states"
        payload = _saved(kind)
        record, key, where = payload[1]["execution"], "state", "[1].execution.state"
    label = "live" if case == "live" else "saved"
    value = f"not_a_{key}"
    record[key] = value
    with pytest.raises(AssertionError) as failure:
        contract.check_closed_vocabularies(label, FAMILY, kind, payload)
    assert f"{label} {FAMILY}/{kind}: {where} = {value!r}" in str(failure.value)
