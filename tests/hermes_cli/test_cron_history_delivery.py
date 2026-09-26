"""Delivery state in the cron run history served by GET /api/cron/executions.

Runs are delivered through the real ``cron.scheduler._deliver_result`` with a
fake Telegram adapter on a real event loop; the history is read back through
the real cron router with machine-token auth.
"""

import asyncio
import importlib
import threading
from unittest.mock import MagicMock, patch

import pytest
import yaml

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import SendResult

CHAT_ROUTED = "-1004444444444"
CHAT_REFUSING = "-1006666666666"
CHAT_RESET = "-1007777777777"
BLOCKED = "Forbidden: bot was blocked by the user"
REPORT = "Quarterly numbers are up 12 percent."
DAY = 86400.0
T = 1_790_000_000.0
NOT_RECORDED = {
    "state": "not_recorded",
    "targets": [],
    "resend": {"eligible": False, "reason": "not_recorded", "attempts": []},
}


@pytest.fixture()
def profiles_home(tmp_path, monkeypatch):
    """A default home whose primary config routes one Telegram chat to the worker_alpha profile."""
    from hermes_cli import profiles

    default_home = tmp_path / ".hermes"
    profiles_root = default_home / "profiles"
    worker_home = profiles_root / "worker_alpha"
    for home in (default_home, worker_home):
        (home / "cron").mkdir(parents=True, exist_ok=True)
        (home / "config.yaml").write_text("model: test-model\n", encoding="utf-8")
    route = {"name": "Ops room", "platform": "telegram", "chat_id": CHAT_ROUTED, "profile": "worker_alpha"}
    (default_home / "config.yaml").write_text(
        yaml.safe_dump({"model": "test-model", "gateway": {"profile_routes": [route]}}), encoding="utf-8"
    )
    monkeypatch.setattr(profiles, "_get_default_hermes_home", lambda: default_home)
    monkeypatch.setattr(profiles, "_get_profiles_root", lambda: profiles_root)
    monkeypatch.setattr("hermes_constants.get_default_hermes_root", lambda: default_home)
    monkeypatch.setenv("HERMES_HOME", str(default_home))
    return {"default": default_home, "worker_alpha": worker_home}


@pytest.fixture()
def history_client(profiles_home, tmp_path):
    """GET the run history through the real cron router with an automations machine token."""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from hermes_cli import web_server  # noqa: F401 - the router resolves its workers from here
    from hermes_cli.dashboard_auth import register_provider, token_auth
    from hermes_cli.dashboard_auth.registry import restore_registration, snapshot_registration
    from hermes_cli.web_routers import cron as cron_routes
    from plugins.dashboard_auth.raphael_workspace import AutomationsManageTokenProvider, token_store

    cron_routes._register_automations_machine_routes()
    token_dir = tmp_path / "machine-tokens"
    token_dir.mkdir(mode=0o700)
    token_path = token_dir / "automations.token"
    token_store.issue(out_path=token_path, surface=token_store.AUTOMATIONS_SURFACE)
    headers = {"Authorization": "Bearer " + token_path.read_text(encoding="utf-8").strip()}
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
            def get_history(profile):
                response = client.get(f"/api/cron/executions?profile={profile}", headers=headers)
                assert response.status_code == 200, response.text
                return response

            yield get_history
    finally:
        if previous is None:
            restore_registration(provider.name, provider, None)


@pytest.fixture()
def live_loop():
    """A running gateway-style event loop on its own thread."""
    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever, daemon=True)
    thread.start()
    ready = threading.Event()
    loop.call_soon_threadsafe(ready.set)
    assert ready.wait(10)
    yield loop

    async def cancel_pending():
        current = asyncio.current_task()
        pending = [task for task in asyncio.all_tasks() if task is not current]
        for task in pending:
            task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)

    asyncio.run_coroutine_threadsafe(cancel_pending(), loop).result(timeout=10)
    loop.call_soon_threadsafe(loop.stop)
    thread.join(timeout=10)
    loop.close()


class ChatTelegram:
    """Live Telegram adapter double answering each chat with its scripted reply."""

    platform = Platform.TELEGRAM
    MAX_MESSAGE_LENGTH = 4096

    def __init__(self, replies):
        self.replies = replies
        self.sent = {}

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        self.sent.setdefault(chat_id, []).append(content)
        reply = self.replies[chat_id]
        if isinstance(reply, BaseException):
            raise reply
        return reply


def _deliver_new_run(home, loop, replies, *, finish=True, job_id="job-weekly"):
    """Claim, run and deliver one report execution under ``home``; return its execution id."""
    from cron import delivery_record, scheduler
    from cron.executions import create_execution, finish_execution, mark_execution_running
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    config = MagicMock()
    config.platforms = {Platform.TELEGRAM: PlatformConfig(enabled=True)}
    config.get_home_channel = lambda _platform: None

    async def refusing_standalone(*_args, **_kwargs):
        return {"error": "Telegram send failed: " + BLOCKED}

    job = {
        "id": job_id,
        "name": "Weekly numbers",
        "deliver": ",".join(f"telegram:{chat}" for chat in replies),
    }
    token = set_hermes_home_override(str(home))
    try:
        execution_id = create_execution(job_id, source="test")["id"]
        mark_execution_running(execution_id)
        with (
            patch("gateway.config.load_gateway_config", return_value=config),
            patch("cron.scheduler.load_config", return_value={"cron": {"wrap_response": False}}),
            patch("tools.send_message_tool._send_to_platform", new=refusing_standalone),
            delivery_record.recording(execution_id, job_id),
        ):
            error = scheduler._deliver_result(
                job, REPORT, adapters={Platform.TELEGRAM: ChatTelegram(replies)}, loop=loop
            )
        if finish:
            finish_execution(
                execution_id, success=True, delivery_outcome="failed" if error else "delivered"
            )
    finally:
        reset_hermes_home_override(token)
    return execution_id


def _history_at(home, execution_id, now):
    from cron.delivery_record import history_deliveries
    from cron.executions import get_execution
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    token = set_hermes_home_override(str(home))
    try:
        return history_deliveries([get_execution(execution_id)], now=now)[0]
    finally:
        reset_hermes_home_override(token)


def test_rows_from_before_delivery_records_read_not_recorded(profiles_home, history_client):
    from cron.executions import create_execution, finish_execution

    outcomes = ["failed", "delivered", "suppressed", "not_configured"]
    ids = []
    for index, outcome in enumerate(outcomes):
        execution = create_execution(f"job-old-{index}", source="test")
        finish_execution(execution["id"], success=True, delivery_outcome=outcome)
        ids.append(execution["id"])

    response = history_client("default")

    body = response.json()
    assert set(body) == {"executions", "limit"}
    by_job = {row["job_id"]: row for row in body["executions"]}
    assert len(by_job) == len(outcomes)
    for index, outcome in enumerate(outcomes):
        row = by_job[f"job-old-{index}"]
        assert row["delivery_outcome"] == outcome
        expected_state = outcome if outcome in ("suppressed", "not_configured") else "not_recorded"
        assert row["delivery"] == dict(NOT_RECORDED, state=expected_state)
    for execution_id in ids:
        assert execution_id not in response.text


def test_new_rows_carry_delivery_state_targets_and_resend(profiles_home, history_client, live_loop):
    execution_id = _deliver_new_run(profiles_home["worker_alpha"], live_loop, {
        CHAT_ROUTED: SendResult(success=True, message_id="m1"),
        CHAT_REFUSING: SendResult(success=False, error=BLOCKED),
        CHAT_RESET: ConnectionResetError("Connection reset by peer"),
    })

    response = history_client("worker_alpha")

    rows = response.json()["executions"]
    assert len(rows) == 1
    row = rows[0]
    assert set(row) == {
        "job_id", "status", "claimed_at", "started_at", "finished_at", "delivery_outcome", "delivery",
    }
    assert (row["job_id"], row["status"], row["delivery_outcome"]) == ("job-weekly", "completed", "failed")
    assert row["delivery"] == {
        "state": "failed",
        "targets": [
            {"label": "Telegram (Ops room)", "state": "delivered", "reason": None},
            {"label": "Telegram", "state": "failed", "reason": "platform_refused"},
            {"label": "Telegram 2", "state": "unknown", "reason": "error_after_handover"},
        ],
        "resend": {"eligible": True, "reason": None, "attempts": []},
    }
    for secret in (CHAT_ROUTED, CHAT_REFUSING, CHAT_RESET, execution_id, REPORT, "Forbidden", "Connection reset"):
        assert secret not in response.text


def test_failed_chat_can_be_resent_only_inside_seven_days(profiles_home, live_loop, monkeypatch):
    store = importlib.import_module("cron.delivery_record")
    monkeypatch.setattr(store, "_clock", lambda: T)
    home = profiles_home["worker_alpha"]
    execution_id = _deliver_new_run(home, live_loop, {
        CHAT_REFUSING: SendResult(success=False, error=BLOCKED),
    })

    inside = _history_at(home, execution_id, T + 7 * DAY - 1)
    expired = _history_at(home, execution_id, T + 7 * DAY)

    assert inside["state"] == "failed"
    assert inside["resend"] == {"eligible": True, "reason": None, "attempts": []}
    assert expired["state"] == "failed"
    assert expired["resend"] == {"eligible": False, "reason": "output_expired", "attempts": []}


def test_unknown_outcome_is_never_resent(profiles_home, live_loop):
    home = profiles_home["worker_alpha"]
    execution_id = _deliver_new_run(home, live_loop, {
        CHAT_RESET: ConnectionResetError("Connection reset by peer"),
    })

    delivery = _history_at(home, execution_id, None)

    assert delivery == {
        "state": "unknown",
        "targets": [{"label": "Telegram", "state": "unknown", "reason": "error_after_handover"}],
        "resend": {"eligible": False, "reason": "outcome_unknown", "attempts": []},
    }


def test_running_and_delivered_runs_are_not_resent(profiles_home, live_loop):
    home = profiles_home["worker_alpha"]
    running = _deliver_new_run(home, live_loop, {
        CHAT_REFUSING: SendResult(success=False, error=BLOCKED),
    }, finish=False, job_id="job-running")
    delivered = _deliver_new_run(home, live_loop, {
        CHAT_ROUTED: SendResult(success=True, message_id="m1"),
    }, job_id="job-delivered")

    in_progress = _history_at(home, running, None)
    done = _history_at(home, delivered, None)

    assert in_progress["state"] == "failed"
    assert in_progress["resend"] == {"eligible": False, "reason": "in_progress", "attempts": []}
    assert done == {
        "state": "delivered",
        "targets": [{"label": "Telegram (Ops room)", "state": "delivered", "reason": None}],
        "resend": {"eligible": False, "reason": "already_delivered", "attempts": []},
    }
