"""Automations examples (contract map section 5, test 7).

Every example comes from the real cron router, called in process with a
locally issued Automations token over the profile homes of
tests/hermes_cli/test_cron_history_delivery.py. Its runs are delivered by the
real scheduler to that file's Telegram double, and the re-send is asked for a
profile that no live gateway records as served, so it is refused before
anything is sent. What is asserted is described in tests/contracts/conftest.py.
"""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from gateway.platforms.base import SendResult
from hermes_cli import web_server  # noqa: F401  (the router reads its helpers here)
from hermes_cli.dashboard_auth import register_provider, token_auth
from hermes_cli.dashboard_auth.registry import restore_registration, snapshot_registration
from hermes_cli.web_routers import cron as cron_routes
from plugins.dashboard_auth.raphael_workspace import AutomationsManageTokenProvider, token_store
from tests.contracts import conftest as contract
from tests.hermes_cli.test_cron_history_delivery import (  # noqa: F401  (two fixtures)
    BLOCKED,
    CHAT_REFUSING,
    CHAT_RESET,
    CHAT_ROUTED,
    _deliver_new_run,
    live_loop,
    profiles_home,
)

FAMILY = "automations"
PROFILE = "worker_alpha"
# The saved examples name the profile home with this placeholder: the real value is a path of
# the machine that ran the test.
NEUTRAL_HOME = "PROFILE_HOME"


def _homes(value):
    """Every hermes_home value in a JSON value."""
    if isinstance(value, dict):
        return [v for k, v in value.items() if k == "hermes_home"] + [h for v in value.values() for h in _homes(v)]
    if isinstance(value, list):
        return [h for v in value for h in _homes(v)]
    return []


def _without_host_paths(value):
    """The answer with every hermes_home value replaced by NEUTRAL_HOME."""
    if isinstance(value, dict):
        return {k: NEUTRAL_HOME if k == "hermes_home" else _without_host_paths(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_without_host_paths(v) for v in value]
    return value
JOB = {"prompt": "Send the weekly numbers", "schedule": "every 1h", "name": "Weekly numbers"}
ROW_COLUMNS = ("job_id", "status", "claimed_at", "started_at", "finished_at", "delivery_outcome")
# Each run's answer by chat, and whether the run finished.
RUNS = (
    ({CHAT_ROUTED: SendResult(success=True, message_id="m1"),
      CHAT_REFUSING: SendResult(success=False, error=BLOCKED),
      CHAT_RESET: ConnectionResetError("Connection reset by peer")}, True),
    ({CHAT_ROUTED: SendResult(success=True, message_id="m2")}, True),
    ({CHAT_RESET: ConnectionResetError("Connection reset by peer")}, True),
    ({CHAT_REFUSING: SendResult(success=False, error=BLOCKED)}, False),
)


@pytest.fixture()
def automations(profiles_home, tmp_path):
    """Call the real cron router with an Automations token, built as the setup
    file's history_client is built; that fixture can only read the history."""
    cron_routes._register_automations_machine_routes()
    token_path = tmp_path / "machine-tokens" / "automations.token"
    token_path.parent.mkdir(mode=0o700)
    token_store.issue(out_path=token_path, surface=token_store.AUTOMATIONS_SURFACE)
    bearer = {"Authorization": "Bearer " + token_path.read_text(encoding="utf-8").strip()}
    provider = AutomationsManageTokenProvider()
    previous = snapshot_registration(provider.name)
    if previous is None:
        register_provider(provider)
    app = FastAPI()
    app.include_router(cron_routes.router)

    @app.middleware("http")
    async def machine_auth(request, call_next):
        return await token_auth.token_auth_middleware(request, call_next)

    try:
        with TestClient(app) as client:
            def call(method: str, path: str, body=None, headers=None) -> dict:
                response = client.request(
                    method, f"/api/cron{path}?profile={PROFILE}", json=body,
                    headers={**bearer, **(headers or {})},
                )
                return {"status": response.status_code, "body": response.json()}

            yield call
    finally:
        if previous is None:
            restore_registration(provider.name, provider, None)


def test_the_automations_examples_carry_every_route_answer_and_error(
    automations, profiles_home, live_loop, owner_payload_example,
):
    keyed = {"Idempotency-Key": "contract-automation-0001"}
    live = {"create": automations("POST", "/jobs", JOB, keyed)}
    job_id = live["create"]["body"]["id"]
    # The same key with another request is refused, and adds no job.
    live["create_conflict"] = automations("POST", "/jobs", {**JOB, "schedule": "every 2h"}, keyed)
    automations("POST", "/jobs", {**JOB, "name": "Monday numbers", "schedule": "0 9 * * 1"})
    live["pause"] = automations("POST", f"/jobs/{job_id}/pause")
    live["resume"] = automations("POST", f"/jobs/{job_id}/resume")
    run_ids = [
        _deliver_new_run(profiles_home[PROFILE], live_loop, replies, finish=finish, job_id=job_id)
        for replies, finish in RUNS
    ]
    live["job_list"] = automations("GET", "/jobs")
    live["executions"] = automations("GET", "/executions")
    live["gateway_unavailable"] = automations(
        "POST", f"/executions/{run_ids[0]}/resend", {"request_id": "contract-resend-0001"})

    saved = {kind: owner_payload_example(FAMILY, kind, _without_host_paths(answer)) for kind, answer in live.items()}
    # No saved example keeps a path of the machine that wrote it.
    assert {home for answer in saved.values() for home in _homes(answer)} == {NEUTRAL_HOME}

    [eligible] = [
        row["delivery"]["resend"] for row in live["executions"]["body"]["executions"]
        if row["delivery"]["resend"]["eligible"]
    ]
    assert eligible["execution_id"] == run_ids[0]
    for answers in (live, saved):
        created, paused, resumed = (answers[kind]["body"] for kind in ("create", "pause", "resume"))
        jobs = answers["job_list"]["body"]
        assert len(jobs) == 2
        assert created["id"] == paused["id"] == resumed["id"] == jobs[0]["id"] != jobs[1]["id"]
        assert created["enabled"] == resumed["enabled"] != paused["enabled"]
        assert created["state"] == resumed["state"] != paused["state"]
        assert set(jobs[0]) == {*resumed, "latest_execution"}
        assert jobs[1]["latest_execution"] is None
        rows = answers["executions"]["body"]["executions"]
        assert len(rows) == len(RUNS)
        assert {row["job_id"] for row in rows} == {created["id"]}
        # The job's latest run is the newest row of the history.
        latest = jobs[0]["latest_execution"]
        assert [latest[key] for key in ROW_COLUMNS] == [rows[0][key] for key in ROW_COLUMNS]
        for row in rows:
            resend = row["delivery"]["resend"]
            assert ("execution_id" in resend) == resend["eligible"]
            assert (row["finished_at"] is None) == (row["status"] == "running")
        for answer in answers.values():
            assert (answer["status"] >= 400) == ("detail" in answer["body"])


def test_the_automations_examples_keep_to_every_closed_vocabulary():
    """The saved Automations examples, read together, take every value of each
    row of the table; the fixture holds each one to its row."""
    contract.assert_closed_vocabularies_covered(FAMILY)
