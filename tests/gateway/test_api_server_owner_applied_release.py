"""An applied change is never offered for approval again.

Approving a change writes to two stores: the native receipt in the Project
store first, then the "this proposal is done" mark in the conversation store.
A restart between the two left the conversation released and its run failed
with "nothing was changed", so the applied change read as waiting for approval
and could be approved a second time.

The seeded state is made with production code throughout: the real admission
binds the claim and the job row, the real ``owner_project_plan_commit`` path
commits the receipt under that run's authority, the gateway then dies before
it records the completion, and after two restarts the recovery writer seen in
the incident (``ResponseStore.fail_orphaned_owner_run``) releases the claim.

Applied means the committed receipt records success. The native plan path
commits its refusals too, so a real one is made the same way and must keep
main's behaviour: still offered for approval, with its refusal line.
"""

import asyncio
import contextlib
import hashlib
import json
import logging
import sqlite3
import time
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from aiohttp.test_utils import TestClient, TestServer

from gateway.platforms.api_server import OwnerAuthorityBroken
from hermes_cli import kanban_db, owner_workspace as ow, projects_db
from tests.gateway.test_api_server import _create_app, _owner_existing_proposal
from tests.gateway.test_api_server_runs import (  # noqa: F401  (an autouse fixture)
    _committed_owner_project,
    _create_runs_app,
    _make_adapter,
    _new_owner_proposal,
    _resolved_owner_task_routes,
)

_API_KEY = "sk-secret"
_PLANNER = "raphael-planner"
_CONVERSATION = "raphael-owner-" + "c" * 32
_RESPONSE_ID = "resp_applied_change_proposal"
_CLAIM_ID = "claim_" + "c" * 32
_KEY = "conversation-" + hashlib.sha256(_RESPONSE_ID.encode("utf-8")).hexdigest()
_OPERATION = "owner_project_plan_commit"
_CONFIG = {"gateway": {"api_server": {"owner_workspace": {"enabled": True}}}}
_HEADERS = {"Idempotency-Key": _KEY, "Authorization": f"Bearer {_API_KEY}"}
_SUCCESS = json.dumps({"ok": True, "change_count": 1})
_REFUSAL = json.dumps({"ok": False, "error": "conflict", "change_count": 0})


def _proposal_run(adapter) -> tuple[dict, dict]:
    """An owner Project and a change session whose latest turn is proposal R."""
    created = _committed_owner_project(
        "setup-applied-change", "setup-applied-change", _new_owner_proposal(),
    )
    proposal = _owner_existing_proposal()
    adapter._response_store.put(_RESPONSE_ID, {
        "response": {"id": _RESPONSE_ID, "created_at": 1},
        "conversation_history": [
            {"role": "user", "content": "Add the approved milestone."},
            {"role": "assistant", "content": json.dumps(proposal)},
        ],
    }, profile=_PLANNER)
    assert adapter._response_store.set_conversation(
        _CONVERSATION, _RESPONSE_ID, owner_proposal=True, profile=_PLANNER,
    ) is True
    add = proposal["changes"][0]
    change = {key: value for key, value in add.items() if key != "existing_parent_refs"}
    change["existing_parents"] = []
    payload = {
        "idempotency_key": _KEY,
        "project_id": created["project_id"],
        "trigger": "owner_request",
        "request_title": proposal["request_title"],
        "summary": proposal["summary"],
        "specification": proposal["specification"],
        "current_milestone": proposal["current_milestone"],
        "owner_visible_result": proposal["owner_visible_result"],
        "later_milestones": proposal["later_milestones"],
        "changes": [change],
    }
    body = {
        "input": "Apply the approved project changes.",
        "owner_workspace_context": {
            "mode": "existing",
            "project_slug": created["project_slug"],
            "project_name": "Workshop pilot",
        },
        "owner_proposal_authority": {
            "proposal_profile": _PLANNER,
            "conversation": _CONVERSATION,
            "response_id": _RESPONSE_ID,
            "claim_id": _CLAIM_ID,
            "operation": _OPERATION,
            "idempotency_key": _KEY,
            "payload": payload,
        },
    }
    return created, body


@contextlib.contextmanager
def _owner_run_patches(adapter, handler=None):
    from tools import owner_workspace_tools

    with (
        patch("gateway.run._load_gateway_config", return_value=_CONFIG),
        patch(
            "hermes_cli.profiles.list_profiles",
            return_value=[SimpleNamespace(name="default")],
        ),
        patch(
            "hermes_cli.owner_workspace._confirm",
            return_value={"approved": True, "reason": None},
        ),
        patch.object(
            adapter, "_create_agent", side_effect=RuntimeError("provider unavailable"),
        ),
        patch.object(
            owner_workspace_tools,
            "_handle_project_plan",
            handler or owner_workspace_tools._handle_project_plan,
        ),
    ):
        yield


async def _crashed_run(adapter, body, *, commits: bool) -> str:
    """Admit run X through ``POST /v1/runs`` and kill the gateway inside it.

    With ``commits`` the real native commit lands first, so the gateway dies
    in the crash window: after the receipt, before its own completion.
    Without it the gateway dies before the native call.
    """
    from tools import owner_workspace_tools

    native = owner_workspace_tools._handle_project_plan

    def _gateway_dies(args, **kw):
        result = native(args, **kw) if commits else None
        adapter._response_store.close()
        if result is None:
            raise RuntimeError("the gateway died")
        return result

    app = _create_runs_app(adapter)
    with _owner_run_patches(adapter, _gateway_dies):
        async with TestClient(TestServer(app)) as cli:
            started = await cli.post("/v1/runs", json=body, headers=_HEADERS)
            assert started.status == 202, await started.text()
            run_id = (await started.json())["run_id"]
            for _ in range(200):
                if adapter._run_statuses.get(run_id, {}).get("status") == "failed":
                    break
                await asyncio.sleep(0.05)
    assert adapter._run_statuses.get(run_id, {}).get("status") == "failed"
    return run_id


def _restarted_twice():
    """Two fresh adapters on the same files, each with empty run state."""
    _make_adapter(api_key=_API_KEY)._response_store.close()
    return _make_adapter(api_key=_API_KEY)


def _recovery_job(store, run_id: str) -> tuple[str, dict]:
    profile, payload = store._conn.execute(
        "SELECT profile, payload FROM owner_executor_jobs "
        "WHERE kind = 'run' AND job_key = ?",
        (run_id,),
    ).fetchone()
    return profile, json.loads(payload)


def _receipts_db():
    return contextlib.closing(sqlite3.connect(
        (ow.owner_executor_home() / "projects.db").resolve().as_uri() + "?mode=ro",
        uri=True,
    ))


def _committed_receipts() -> list[tuple]:
    with _receipts_db() as conn:
        return conn.execute(
            "SELECT authority_digest, result_json FROM owner_workspace_receipts "
            "WHERE idempotency_key = ? AND operation = ? AND status = 'committed'",
            (_KEY, _OPERATION),
        ).fetchall()


def _conversation_row(store) -> tuple:
    return store._conn.execute(
        "SELECT proposal_response_id, consumed_response_id, claimed_response_id, "
        "claim_id, owner_run_id, claim_state, closed, bound_operation, "
        "bound_payload_digest FROM conversations WHERE profile = ? AND name = ?",
        (_PLANNER, _CONVERSATION),
    ).fetchone()


def _count(store, table: str) -> int:
    return store._conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]


def _task_count(created: dict) -> int:
    with contextlib.closing(kanban_db.connect(board=created["board"])) as conn:
        return conn.execute(
            "SELECT COUNT(*) FROM tasks WHERE project_id = ?",
            (created["project_id"],),
        ).fetchone()[0]


async def _applied_then_released(*, recover: bool = True):
    """Approved, applied, two restarts, card and review ended, milestone complete.

    Returns the restarted adapter, the Project, the run body and run X. With
    ``recover`` false the restarted gateway has not run its recovery yet.
    """
    adapter = _make_adapter(api_key=_API_KEY)
    created, body = _proposal_run(adapter)
    run_id = await _crashed_run(adapter, body, commits=True)

    restarted = _restarted_twice()
    store = restarted._response_store
    [(authority_digest, result_json)] = _committed_receipts()
    # The receipt helper binds the same digest the conversation row is bound to.
    assert authority_digest == _conversation_row(store)[8]
    profile, job = _recovery_job(store, run_id)
    assert job["native"]["authority_digest"] == authority_digest
    assert _conversation_row(store)[4:6] == (run_id, "claimed")
    if recover:
        assert store.fail_orphaned_owner_run(profile, run_id, job["owner"]) is True

    # The card the receipt created was built and its review ended. The milestone
    # has no stored state of its own: it completes with its only task.
    with contextlib.closing(kanban_db.connect(board=created["board"])) as conn:
        for task_id in json.loads(result_json)["created_task_ids"]:
            assert kanban_db.complete_task(
                conn, task_id, result="The milestone is prepared.",
            ) is True
            assert kanban_db.get_task(conn, task_id).status == "done"
    return restarted, created, body, run_id


async def _refused_then_released(*, crash: bool, recover: bool = True):
    """Approved, then refused for real by the native plan path with zero changes.

    The Project is archived while the plan awaits confirmation, as a
    concurrent owner action would, so the unchanged native path refuses with
    ``conflict`` and commits its normal terminal receipt recording ``ok``
    false. Without ``crash`` the run fails in place; with it the gateway dies
    after that commit and, after two restarts, the recovery writer releases
    the claim unless ``recover`` is false. The Project is then restored, so
    the owner view reads the refused exchange. Returns the live adapter, the
    Project, the run body and run X.
    """
    from tools import owner_workspace_tools

    adapter = _make_adapter(api_key=_API_KEY)
    created, body = _proposal_run(adapter)
    tasks = _task_count(created)
    native = owner_workspace_tools._handle_project_plan

    def _archived(*_args, **_kw):
        with contextlib.closing(projects_db.connect()) as conn:
            assert projects_db.archive_project(conn, created["project_id"])
        return {"approved": True, "reason": None}

    def _refused(args, **kw):
        with patch.object(ow, "_confirm", side_effect=_archived):
            return native(args, **kw)

    if crash:
        with patch.object(owner_workspace_tools, "_handle_project_plan", _refused):
            run_id = await _crashed_run(adapter, body, commits=True)
        adapter = _restarted_twice()
        profile, job = _recovery_job(adapter._response_store, run_id)
        if recover:
            assert adapter._response_store.fail_orphaned_owner_run(
                profile, run_id, job["owner"],
            ) is True
    else:
        with _owner_run_patches(adapter, _refused):
            async with TestClient(TestServer(_create_runs_app(adapter))) as cli:
                started = await cli.post("/v1/runs", json=body, headers=_HEADERS)
                assert started.status == 202, await started.text()
                run_id = (await started.json())["run_id"]
                for _ in range(200):
                    if adapter._run_statuses.get(run_id, {}).get("status") == "failed":
                        break
                    await asyncio.sleep(0.05)
        assert adapter._run_statuses[run_id]["status"] == "failed"
    [(_authority_digest, result_json)] = _committed_receipts()
    result = json.loads(result_json)
    assert result["ok"] is False and result["error"] == "conflict"
    assert result["change_count"] == 0 and _task_count(created) == tasks
    if recover:
        assert _conversation_row(adapter._response_store)[4:6] == (run_id, "released")
    with contextlib.closing(projects_db.connect()) as conn:
        assert projects_db.restore_project(conn, created["project_id"])
    return adapter, created, body, run_id


async def _approve(adapter, body) -> tuple[int, dict, list]:
    """One more approval through ``POST /v1/runs``, and the native calls it made."""
    from tools import owner_workspace_tools

    native_calls = []
    native = owner_workspace_tools._handle_project_plan

    def _native(args, **kw):
        native_calls.append(args)
        return native(args, **kw)

    with _owner_run_patches(adapter, _native):
        async with TestClient(TestServer(_create_runs_app(adapter))) as cli:
            response = await cli.post("/v1/runs", json=body, headers=_HEADERS)
            return response.status, await response.json(), native_calls


def _run_rows(store) -> list[tuple]:
    return store._conn.execute(
        "SELECT * FROM run_idempotency WHERE idempotency_key = ?", (_KEY,),
    ).fetchall()


async def _history(adapter):
    """The owner history route, as the owner app reads it."""
    with patch(
        "gateway.platforms.api_server._active_owner_profile", return_value=_PLANNER,
    ):
        async with TestClient(TestServer(_create_app(adapter))) as cli:
            response = await cli.get(
                f"/v1/responses/conversations/{_CONVERSATION}", headers=_HEADERS,
            )
            return response.status, await response.json()


def _seeded_receipts(home, *results) -> None:
    """One committed plan receipt per result, under the proposal's key and digest."""
    with contextlib.closing(sqlite3.connect(home / "projects.db")) as conn:
        conn.executescript(ow._RECEIPTS_SCHEMA)
        conn.executemany(
            "INSERT INTO owner_workspace_receipts (actor, profile, idempotency_key, "
            "operation, request_digest, status, result_json, authority_digest, "
            "created_at, updated_at) "
            "VALUES (?, 'default', ?, ?, ?, 'committed', ?, ?, 1, 1)",
            [
                (f"actor-{index}", _KEY, _OPERATION, "d" * 64, result, "a" * 64)
                for index, result in enumerate(results)
            ],
        )
        conn.commit()


@pytest.mark.asyncio
async def test_a1_an_applied_change_reads_as_consumed():
    restarted, _created, _body, run_id = await _applied_then_released()
    store = restarted._response_store
    row = _conversation_row(store)
    assert row[1] != _RESPONSE_ID and row[4:6] == (run_id, "released")
    assert store.run_idempotency_status("default", run_id)["status"] == "failed"

    snapshot = store.owner_history_snapshot(_CONVERSATION, profile=_PLANNER)

    assert snapshot["proposal_consumed"] is True
    assert snapshot["released_run_id"] is None
    assert snapshot["completed_run_id"] is None
    assert snapshot["latest_response_id"] == _RESPONSE_ID
    # Read-only: the five affected sessions read correctly without a repair.
    assert _conversation_row(store) == row
    assert len(_committed_receipts()) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("purged", [False, True], ids=["a2-same-request", "a3-after-purge"])
async def test_a2_a3_a_second_approval_of_an_applied_change_is_refused(purged):
    restarted, created, body, run_id = await _applied_then_released()
    store = restarted._response_store
    if purged:
        # The 24-hour purge, and a payload with a fresh fingerprint.
        store._conn.execute(
            "DELETE FROM run_idempotency WHERE idempotency_key = ?", (_KEY,),
        )
        store._conn.commit()
        payload = body["owner_proposal_authority"]["payload"]
        payload["summary"] = f"  {payload['summary']}\n"
    row = _conversation_row(store)
    runs = _count(store, "run_idempotency")
    jobs = _count(store, "owner_executor_jobs")
    tasks = _task_count(created)

    from tools import owner_workspace_tools

    native_calls = []
    native = owner_workspace_tools._handle_project_plan

    def _native(args, **kw):
        native_calls.append(args)
        return native(args, **kw)

    app = _create_runs_app(restarted)
    with _owner_run_patches(restarted, _native):
        async with TestClient(TestServer(app)) as cli:
            second = await cli.post("/v1/runs", json=body, headers=_HEADERS)
            refused = await second.json()

    assert second.status == 409, refused
    assert refused["error"]["code"] == "owner_proposal_already_applied"
    assert "run_id" not in refused
    assert restarted._run_statuses == {}
    assert native_calls == []
    assert _conversation_row(store) == row
    assert _count(store, "run_idempotency") == runs
    assert _count(store, "owner_executor_jobs") == jobs
    assert len(_committed_receipts()) == 1
    assert _task_count(created) == tasks


@pytest.mark.asyncio
async def test_a4_a_released_proposal_without_a_committed_receipt_stays_approvable():
    adapter = _make_adapter(api_key=_API_KEY)
    created, body = _proposal_run(adapter)
    run_id = await _crashed_run(adapter, body, commits=False)
    restarted = _restarted_twice()
    store = restarted._response_store
    profile, job = _recovery_job(store, run_id)
    assert store.fail_orphaned_owner_run(profile, run_id, job["owner"]) is True
    assert _committed_receipts() == []

    snapshot = store.owner_history_snapshot(_CONVERSATION, profile=_PLANNER)
    assert snapshot["released_run_id"] == run_id
    assert snapshot["proposal_consumed"] is False
    assert snapshot["latest_response_id"] == _RESPONSE_ID
    assert store.run_idempotency_status("default", run_id)["status"] == "failed"

    app = _create_runs_app(restarted)
    with _owner_run_patches(restarted):
        async with TestClient(TestServer(app)) as cli:
            again = await cli.post("/v1/runs", json=body, headers=_HEADERS)
            again_body = await again.json()
            assert again.status == 202, again_body
            retry_run_id = again_body["run_id"]
            for _ in range(200):
                if restarted._run_statuses.get(retry_run_id, {}).get("status") == "completed":
                    break
                await asyncio.sleep(0.05)

    assert retry_run_id != run_id
    assert restarted._run_statuses[retry_run_id]["status"] == "completed"
    assert len(_committed_receipts()) == 1
    applied = store.owner_history_snapshot(_CONVERSATION, profile=_PLANNER)
    assert applied["proposal_consumed"] is True
    assert applied["completed_run_id"] == retry_run_id


@pytest.mark.asyncio
async def test_a5_an_unreadable_receipts_store_fails_closed(tmp_path):
    restarted, created, body, _run_id = await _applied_then_released()
    store = restarted._response_store
    unreadable = tmp_path / "unreadable-executor"
    unreadable.mkdir()
    (unreadable / "projects.db").write_bytes(b"not a database " * 128)

    with patch.object(ow, "owner_executor_home", return_value=unreadable):
        with pytest.raises(ow.OwnerReceiptUnreadable):
            ow.committed_owner_receipt_for_proposal(
                operation=_OPERATION, idempotency_key=_KEY, authority_digest=None,
            )
        with pytest.raises(OwnerAuthorityBroken):
            store.owner_history_snapshot(_CONVERSATION, profile=_PLANNER)

    row = _conversation_row(store)
    runs = _count(store, "run_idempotency")
    jobs = _count(store, "owner_executor_jobs")
    tasks = _task_count(created)
    app = _create_runs_app(restarted)
    with (
        _owner_run_patches(restarted),
        patch.object(
            ow,
            "committed_owner_receipt_for_proposal",
            side_effect=ow.OwnerReceiptUnreadable("the receipts store is unreadable"),
        ),
    ):
        with pytest.raises(OwnerAuthorityBroken):
            store.owner_history_snapshot(_CONVERSATION, profile=_PLANNER)
        async with TestClient(TestServer(app)) as cli:
            second = await cli.post("/v1/runs", json=body, headers=_HEADERS)
            refused = await second.json()

    assert second.status == 503, refused
    assert refused["error"]["code"] == "owner_run_outcome_unconfirmed"
    assert restarted._run_statuses == {}
    assert _conversation_row(store) == row
    assert _count(store, "run_idempotency") == runs
    assert _count(store, "owner_executor_jobs") == jobs
    assert len(_committed_receipts()) == 1
    assert _task_count(created) == tasks


def test_the_receipt_check_only_reads(tmp_path):
    empty = tmp_path / "no-executor-store"
    empty.mkdir()
    with patch.object(ow, "owner_executor_home", return_value=empty):
        assert ow.committed_owner_receipt_for_proposal(
            operation=_OPERATION, idempotency_key=_KEY, authority_digest="a" * 64,
        ) is False
    # Asking about a store that does not exist creates none.
    assert list(empty.iterdir()) == []


@pytest.mark.asyncio
async def test_recovery_logs_a_failed_run_whose_change_is_committed(caplog):
    restarted, _created, _body, run_id = await _applied_then_released(recover=False)
    store = restarted._response_store
    _profile, job = _recovery_job(store, run_id)
    store._conn.execute(
        "UPDATE owner_executor_jobs SET executor_id = 'dead', "
        "executor_pid = 2147483646, created_at = ? WHERE job_key = ?",
        (time.time() - 3600, run_id),
    )
    store._conn.commit()

    # The recovery miss of the incident, whatever its root cause: the receipt
    # lookup answers "nothing committed" for a run whose change did commit.
    with (
        patch.object(restarted, "_reconcile_orphaned_native_run", return_value=False),
        caplog.at_level(logging.WARNING, logger="gateway.platforms.api_server"),
    ):
        restarted._recover_orphaned_owner_jobs()

    # The decision is unchanged: the run fails and its claim is released.
    status = store.run_idempotency_status("default", run_id)
    assert status["status"] == "failed" and "restarted" in status["error"]
    assert _conversation_row(store)[4:6] == (run_id, "released")
    assert _count(store, "owner_executor_jobs") == 0
    flagged = [
        record for record in caplog.records
        if run_id in record.getMessage() and _KEY in record.getMessage()
    ]
    assert len(flagged) == 1
    assert flagged[0].levelno == logging.ERROR
    # The job row is gone now; the line keeps what the recovery lookup compared.
    assert job["native"]["authority_digest"] in flagged[0].getMessage()


@pytest.mark.asyncio
async def test_a_closed_session_past_the_purge_reads_its_change_as_applied():
    restarted, _created, _body, _run_id = await _applied_then_released()
    store = restarted._response_store
    # "Start a new change" closed it, as it closed the sessions found in
    # production, and the 24-hour purge took its run row: only the bound
    # digest still ties the session to its committed receipt.
    assert store.close_owner_conversation(_PLANNER, _CONVERSATION, _RESPONSE_ID) is True
    store._conn.execute("DELETE FROM run_idempotency WHERE idempotency_key = ?", (_KEY,))
    store._conn.commit()
    row = _conversation_row(store)

    snapshot = store.owner_history_snapshot(_CONVERSATION, profile=_PLANNER)

    assert snapshot["conversation_closed"] is True
    assert snapshot["proposal_consumed"] is True
    assert snapshot["released_run_id"] is None
    assert _conversation_row(store) == row


@pytest.mark.asyncio
async def test_refused_plan_is_not_reported_as_applied_by_history_route():
    adapter, _created, _body, run_id = await _refused_then_released(crash=False)
    store = adapter._response_store
    row = _conversation_row(store)

    status, snapshot = await _history(adapter)
    async with TestClient(TestServer(_create_runs_app(adapter))) as cli:
        polled = await cli.get(f"/v1/runs/{run_id}", headers=_HEADERS)
        run = await polled.json()

    # As on main: the refusal applied nothing, so the proposal is still offered
    # for approval and its released run still gives the honest refusal line.
    assert status == 200, snapshot
    assert snapshot["proposal_consumed"] is False
    assert snapshot["proposal_claimed"] is False
    assert snapshot["released_run_id"] == run_id
    assert snapshot["completed_run_id"] is None
    assert snapshot["latest_response_id"] == _RESPONSE_ID
    assert polled.status == 200, run
    assert run["status"] == "failed"
    assert run["owner_mutation_committed"] is False
    assert run["error"] == "The approved Project change was not committed: conflict"
    assert _conversation_row(store) == row


@pytest.mark.asyncio
@pytest.mark.parametrize("crash", [False, True], ids=["released", "crash-window"])
async def test_real_committed_conflict_is_not_projected_as_applied(crash):
    adapter, _created, body, run_id = await _refused_then_released(crash=crash)
    store = adapter._response_store
    [(authority_digest, _result_json)] = _committed_receipts()

    snapshot = store.owner_history_snapshot(_CONVERSATION, profile=_PLANNER)

    assert snapshot["proposal_consumed"] is False
    assert snapshot["released_run_id"] == run_id
    assert snapshot["completed_run_id"] is None
    assert snapshot["latest_response_id"] == _RESPONSE_ID
    for key, digest in ((_KEY, None), (None, authority_digest)):
        assert ow.committed_owner_receipt_for_proposal(
            operation=_OPERATION, idempotency_key=key, authority_digest=digest,
        ) is False

    # Admission is not told it was applied either: a changed request on the
    # kept row keeps main's answer for a reused key.
    row = _conversation_row(store)
    runs = _run_rows(store)
    statuses = dict(adapter._run_statuses)
    changed = {**body, "input": body["input"] + " "}
    status, refused, native_calls = await _approve(adapter, changed)
    assert status == 409, refused
    assert refused["error"]["code"] == "idempotency_conflict"
    assert native_calls == [] and adapter._run_statuses == statuses
    assert _conversation_row(store) == row and _run_rows(store) == runs


@pytest.mark.asyncio
@pytest.mark.parametrize("crash", [False, True], ids=["released", "crash-window"])
async def test_the_same_approval_of_a_committed_refusal_keeps_main_answer(crash):
    adapter, created, body, _run_id = await _refused_then_released(crash=crash)
    store = adapter._response_store
    receipts = _committed_receipts()
    row = _conversation_row(store)
    runs = _run_rows(store)
    jobs = _count(store, "owner_executor_jobs")
    statuses = dict(adapter._run_statuses)
    tasks = _task_count(created)

    status, refused, native_calls = await _approve(adapter, body)

    # As on main: a kept run whose committed outcome is not a success is
    # neither replayed nor refused as applied, and no second run starts.
    assert status == 503, refused
    assert refused["error"]["code"] == "owner_run_outcome_unconfirmed"
    assert native_calls == [] and adapter._run_statuses == statuses
    assert _conversation_row(store) == row and _run_rows(store) == runs
    assert _count(store, "owner_executor_jobs") == jobs
    assert _committed_receipts() == receipts
    assert _task_count(created) == tasks


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "change", ["payload-whitespace", "input-whitespace", "claim-id"],
)
async def test_applied_retry_with_changed_fingerprint_still_returns_stable_code(change):
    restarted, created, body, _run_id = await _applied_then_released()
    store = restarted._response_store
    if change == "payload-whitespace":
        # Canonically the same payload, so its authority still validates.
        payload = body["owner_proposal_authority"]["payload"]
        payload["summary"] = f"  {payload['summary']}\n"
    elif change == "input-whitespace":
        body["input"] += " "
    else:
        body["owner_proposal_authority"]["claim_id"] = "claim_" + "d" * 32
    row = _conversation_row(store)
    # The original run row is kept: only the request fingerprint changed.
    runs = _run_rows(store)
    assert len(runs) == 1
    jobs = _count(store, "owner_executor_jobs")
    tasks = _task_count(created)

    status, refused, native_calls = await _approve(restarted, body)

    assert status == 409, refused
    assert refused["error"]["code"] == "owner_proposal_already_applied"
    assert "run_id" not in refused
    assert restarted._run_statuses == {}
    assert native_calls == []
    assert _conversation_row(store) == row
    assert _run_rows(store) == runs
    assert _count(store, "owner_executor_jobs") == jobs
    assert len(_committed_receipts()) == 1
    assert _task_count(created) == tasks


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "result_json", ["{not json", '{"ok": null}'], ids=["not-json", "ok-null"],
)
async def test_a_committed_receipt_with_an_unreadable_result_fails_closed(result_json):
    restarted, created, body, _run_id = await _applied_then_released()
    store = restarted._response_store
    [(authority_digest, _applied)] = _committed_receipts()
    # The disposable store's one matching committed receipt, made unreadable.
    with contextlib.closing(
        sqlite3.connect(ow.owner_executor_home() / "projects.db"),
    ) as conn:
        conn.execute(
            "UPDATE owner_workspace_receipts SET result_json = ? "
            "WHERE idempotency_key = ?",
            (result_json, _KEY),
        )
        conn.commit()
    tasks = _task_count(created)

    for key, digest in ((_KEY, None), (None, authority_digest)):
        with pytest.raises(ow.OwnerReceiptUnreadable):
            ow.committed_owner_receipt_for_proposal(
                operation=_OPERATION, idempotency_key=key, authority_digest=digest,
            )
    # Neither shown as consumed nor offered for approval on a guess.
    with pytest.raises(OwnerAuthorityBroken):
        store.owner_history_snapshot(_CONVERSATION, profile=_PLANNER)
    status, unavailable = await _history(restarted)
    assert status == 503, unavailable
    assert unavailable["error"]["code"] == "owner_history_unavailable"

    # Admission starts no run and writes nothing: the same request and a
    # changed one on the kept row, then the same request after the purge.
    changed = {**body, "input": body["input"] + " "}
    for purged, request in ((False, body), (False, changed), (True, body)):
        if purged:
            store._conn.execute(
                "DELETE FROM run_idempotency WHERE idempotency_key = ?", (_KEY,),
            )
            store._conn.commit()
        row = _conversation_row(store)
        runs = _run_rows(store)
        jobs = _count(store, "owner_executor_jobs")

        status, refused, native_calls = await _approve(restarted, request)

        assert status == 503, refused
        assert refused["error"]["code"] == "owner_run_outcome_unconfirmed"
        assert restarted._run_statuses == {}
        assert native_calls == []
        assert _conversation_row(store) == row
        assert _run_rows(store) == runs
        assert _count(store, "owner_executor_jobs") == jobs
    # No repair either: the receipt is left exactly as it was.
    assert _committed_receipts() == [(authority_digest, result_json)]
    assert _task_count(created) == tasks


@pytest.mark.parametrize(
    ("results", "applied"),
    [
        ((_SUCCESS,), True),
        ((_REFUSAL,), False),
        ((_REFUSAL, _SUCCESS), True),
        ((_REFUSAL, _REFUSAL), False),
        ((None,), None),
        (("{not json",), None),
        (('["ok", true]',), None),
        (('{"ok": null}',), None),
        (('{"ok": 1}',), None),
        ((_SUCCESS, "{not json"), None),
        ((_REFUSAL, '{"ok": "false"}'), None),
    ],
    ids=[
        "success", "refusal", "refusal-and-success", "refusals",
        "no-result", "not-json", "not-an-object", "ok-null", "ok-not-a-bool",
        "success-and-unreadable", "refusal-and-unreadable",
    ],
)
def test_the_receipt_check_reads_every_matching_result(tmp_path, results, applied):
    """``applied`` None: the check fails closed, whatever the other rows say."""
    home = tmp_path / "executor-store"
    home.mkdir()
    _seeded_receipts(home, *results)
    stored = (home / "projects.db").read_bytes()
    opened, statements = [], []
    connect = ow.sqlite3.connect

    def _traced(*args, **kw):
        opened.append(args[0])
        conn = connect(*args, **kw)
        conn.set_trace_callback(statements.append)
        return conn

    with (
        patch.object(ow, "owner_executor_home", return_value=home),
        patch.object(ow.sqlite3, "connect", side_effect=_traced),
    ):
        for key, digest in ((_KEY, None), (None, "a" * 64)):
            if applied is None:
                with pytest.raises(ow.OwnerReceiptUnreadable):
                    ow.committed_owner_receipt_for_proposal(
                        operation=_OPERATION, idempotency_key=key,
                        authority_digest=digest,
                    )
            else:
                assert ow.committed_owner_receipt_for_proposal(
                    operation=_OPERATION, idempotency_key=key,
                    authority_digest=digest,
                ) is applied
        # Only this operation's receipts under this key or digest are read.
        assert ow.committed_owner_receipt_for_proposal(
            operation="owner_task_graph_commit", idempotency_key=_KEY,
            authority_digest="a" * 64,
        ) is False
        assert ow.committed_owner_receipt_for_proposal(
            operation=_OPERATION, idempotency_key="conversation-other",
            authority_digest="0" * 64,
        ) is False

    assert opened and all(uri.endswith("?mode=ro") for uri in opened)
    assert statements
    assert all(sql.lstrip().upper().startswith("SELECT") for sql in statements)
    assert (home / "projects.db").read_bytes() == stored
    assert [path.name for path in home.iterdir()] == ["projects.db"]


@pytest.mark.asyncio
async def test_recovery_logs_nothing_for_a_committed_refusal(caplog):
    adapter, _created, _body, run_id = await _refused_then_released(
        crash=True, recover=False,
    )
    store = adapter._response_store
    store._conn.execute(
        "UPDATE owner_executor_jobs SET executor_id = 'dead', "
        "executor_pid = 2147483646, created_at = ? WHERE job_key = ?",
        (time.time() - 3600, run_id),
    )
    store._conn.commit()

    # The same recovery miss as above, for a run whose committed receipt
    # records a refusal: nothing was applied, so no line says it was.
    with (
        patch.object(adapter, "_reconcile_orphaned_native_run", return_value=False),
        caplog.at_level(logging.WARNING, logger="gateway.platforms.api_server"),
    ):
        adapter._recover_orphaned_owner_jobs()

    status = store.run_idempotency_status("default", run_id)
    assert status["status"] == "failed" and "restarted" in status["error"]
    assert _conversation_row(store)[4:6] == (run_id, "released")
    assert _count(store, "owner_executor_jobs") == 0
    assert [
        record for record in caplog.records
        if run_id in record.getMessage() and _KEY in record.getMessage()
    ] == []
