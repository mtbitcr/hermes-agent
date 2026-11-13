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
from hermes_cli import kanban_db, owner_workspace as ow
from tests.gateway.test_api_server import _owner_existing_proposal
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
