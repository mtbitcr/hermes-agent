"""Pause of the owner's Telegram messages (gateway/owner_channel_pause.py).

While paused, the chat the gateway configuration names as the Telegram home channel gets no card
notice from the kanban notifier and no scheduled report from ``cron.scheduler._deliver_result``.
A paused notice is not kept: its cursor advances without a failure, so nothing old is sent after
the resume. A paused report is recorded failed ``route_refused`` and stays eligible for Send again,
which is itself refused while paused and allowed after the resume. A missing, unreadable or
malformed pause file counts as not paused. Every other chat is served as before. The module is
imported inside each test, so on a tree without it each test fails on its own. Chat ids and the
report text are invented placeholders.
"""

import asyncio
from concurrent.futures import Future
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import pytest

from cron import delivery_record, scheduler
from cron.executions import create_execution, finish_execution
from cron.jobs import create_job
from gateway.config import GatewayConfig, HomeChannel, Platform, PlatformConfig
from gateway.platforms.base import SendResult
from gateway.run import GatewayRunner
from hermes_cli import kanban_db as kb
from hermes_constants import get_default_hermes_root

HOME_CHAT = "5550000000001"
OTHER_CHAT = "5550000000002"
REPORT = "Placeholder report: the numbers are up."
_REAL_SLEEP = asyncio.sleep


def _gateway_config():
    """The gateway configuration: Telegram names HOME_CHAT as its home channel; Discord has none."""
    home = HomeChannel(platform=Platform.TELEGRAM, chat_id=HOME_CHAT, name="Home")
    return GatewayConfig(platforms={
        Platform.TELEGRAM: PlatformConfig(enabled=True, home_channel=home),
        Platform.DISCORD: PlatformConfig(enabled=True),
    })


def _pause_file():
    return get_default_hermes_root() / "owner_channel_pause.json"


# --- card notices ----------------------------------------------------------------------------------

class _Telegram:
    """A push-capable Telegram adapter that keeps every visible message it is handed."""

    def __init__(self):
        self.sent = []

    async def send(self, chat_id, text, metadata=None):
        self.sent.append((str(chat_id), text))

    async def handle_message(self, event):
        event._gateway_accepted = True

    def to(self, chat_id):
        return [text for chat, text in self.sent if chat == chat_id]


def _runner(adapter):
    runner = GatewayRunner.__new__(GatewayRunner)
    runner.adapters = {Platform.TELEGRAM: adapter}
    runner.config = _gateway_config()
    runner._kanban_sub_fail_counts = {}
    runner._kanban_dispatcher_lock_handle = object()
    return runner


def _tick(monkeypatch, runner):
    """One notifier tick: the start-up delay is skipped and the watcher stops after the tick."""

    async def sleep(delay):
        if delay != 5:
            runner._running = False
            await _REAL_SLEEP(0)

    monkeypatch.setattr(asyncio, "sleep", sleep)
    runner._running = True
    asyncio.run(runner._kanban_notifier_watcher(interval=1))


def _completed_task(*chat_ids):
    """A task each of ``chat_ids`` subscribed to on Telegram, then completed; its id."""
    conn = kb.connect()
    try:
        task_id = kb.create_task(conn, title="pause task", assignee="worker")
        for chat_id in chat_ids:
            kb.add_notify_sub(conn, task_id=task_id, platform="telegram", chat_id=chat_id, chat_type="dm")
        kb.complete_task(conn, task_id, summary="done")
        return task_id
    finally:
        conn.close()


def _unseen(task_id, chat_id):
    conn = kb.connect()
    try:
        _, events = kb.unseen_events_for_sub(
            conn, task_id=task_id, platform="telegram", chat_id=chat_id,
            kinds=["completed", "blocked", "gave_up", "crashed", "timed_out"],
        )
        return events
    finally:
        conn.close()


def _subs(task_id):
    conn = kb.connect()
    try:
        return kb.list_notify_subs(conn, task_id)
    finally:
        conn.close()


# --- automation reports ----------------------------------------------------------------------------

def _running_loop():
    loop = MagicMock()
    loop.is_running.return_value = True
    return loop


@contextmanager
def _transports():
    """The gateway configuration above, a live router and a standalone sender that both deliver
    every chat they are handed; yields the ``platform:chat`` each path was handed, in order."""
    handed = {"live": [], "standalone": []}
    router = MagicMock()

    async def deliver_to_platform(target, text, metadata):
        handed["live"].append(f"{target.platform.value}:{target.chat_id}")
        return SendResult(success=True, message_id="m1")

    async def standalone(platform, pconfig, chat_id, message, thread_id=None, media_files=None, **_kwargs):
        handed["standalone"].append(f"{platform.value}:{chat_id}")
        return {"success": True, "message_id": "m1", "chat_id": chat_id}

    def run_now(coro, _loop):
        future = Future()
        future.set_result(asyncio.run(coro))
        return future

    router._deliver_to_platform = deliver_to_platform
    with (
        patch("gateway.config.load_gateway_config", return_value=_gateway_config()),
        patch("cron.scheduler.load_config", return_value={"cron": {"wrap_response": False}}),
        patch("gateway.delivery.DeliveryRouter", return_value=router),
        patch("asyncio.run_coroutine_threadsafe", side_effect=run_now),
        patch("tools.send_message_tool._send_to_platform", new=standalone),
    ):
        yield handed


def _report(execution_id, job, **live):
    """One run of ``job`` delivered through the real ``_deliver_result``: what each path was
    handed and the returned error."""
    with _transports() as handed, delivery_record.recording(execution_id, job["id"]):
        error = scheduler._deliver_result(job, REPORT, **live)
    return handed, error


def _outcomes(execution_id):
    return [(target["state"], target["reason"]) for target in delivery_record.load(execution_id)["targets"]]


def _resend_view(execution_id):
    return delivery_record.history_deliveries([{"id": execution_id, "status": "completed"}])[0]["resend"]


def _send_again(execution_id, request_id):
    """Send again through the real claim and ``resend_report``: what each path was handed and how
    each claimed chat of the attempt ended."""
    with _transports() as handed:
        claim = scheduler.claim_report_resend(execution_id, request_id)
        assert claim["claimed"] is True, claim
        scheduler.resend_report(execution_id, claim["attempt"])
    [attempt] = [
        attempt for attempt in delivery_record.load(execution_id)["attempts"]
        if attempt["attempt_id"] == claim["attempt"]["attempt_id"]
    ]
    return handed, attempt["chats"]


# --- the four card tests ---------------------------------------------------------------------------

def test_paused_home_chat_gets_no_card_notice_and_nothing_old_after_the_resume(tmp_path, monkeypatch):
    from gateway import owner_channel_pause

    monkeypatch.setenv("HERMES_KANBAN_DB", str(tmp_path / "pause.db"))
    kb.init_db()
    adapter = _Telegram()
    runner = _runner(adapter)

    owner_channel_pause.pause()
    assert owner_channel_pause.is_paused() is True
    assert _pause_file().is_file()
    # Every profile reads the one file in the root Hermes home.
    root = get_default_hermes_root()
    monkeypatch.setenv("HERMES_HOME", str(root / "profiles" / "coder"))
    assert owner_channel_pause.is_paused() is True
    monkeypatch.setenv("HERMES_HOME", str(root))

    held = _completed_task(HOME_CHAT, OTHER_CHAT)
    _tick(monkeypatch, runner)

    assert adapter.to(HOME_CHAT) == []
    assert [held in text for text in adapter.to(OTHER_CHAT)] == [True], "every other chat as before"
    assert _unseen(held, HOME_CHAT) == [], "the cursor advances: the notice is not kept"
    assert runner._kanban_sub_fail_counts == {}, "a paused notice is no failure"
    assert len(_subs(held)) == 2

    owner_channel_pause.resume()
    assert owner_channel_pause.is_paused() is False
    _tick(monkeypatch, runner)
    assert adapter.to(HOME_CHAT) == [], "nothing old is sent after the resume"

    fresh = _completed_task(HOME_CHAT)
    _tick(monkeypatch, runner)
    [notice] = adapter.to(HOME_CHAT)
    assert fresh in notice and held not in notice
    assert runner._kanban_sub_fail_counts == {}


def test_paused_report_is_sent_on_neither_path_reads_route_refused_and_may_be_sent_again(monkeypatch):
    from gateway import owner_channel_pause

    owner_channel_pause.pause()
    job = {
        "id": "job-report",
        "name": "Daily report",
        "deliver": f"telegram:{HOME_CHAT},telegram:{OTHER_CHAT},discord:{HOME_CHAT}",
    }

    live, live_error = _report("exec-live", job, adapters={Platform.TELEGRAM: MagicMock()}, loop=_running_loop())
    standalone, standalone_error = _report("exec-standalone", job)

    # The home chat reaches neither path; another Telegram chat and another platform go as before.
    assert live == {"live": [f"telegram:{OTHER_CHAT}"], "standalone": [f"discord:{HOME_CHAT}"]}
    assert standalone == {"live": [], "standalone": [f"telegram:{OTHER_CHAT}", f"discord:{HOME_CHAT}"]}
    for execution_id, error in (("exec-live", live_error), ("exec-standalone", standalone_error)):
        assert error, "the run reports its refused chat"
        assert _outcomes(execution_id) == [("failed", "route_refused"), ("delivered", None), ("delivered", None)]
        assert _resend_view(execution_id) == {"eligible": True, "reason": None, "attempts": []}


@pytest.mark.parametrize("pause_file", ["missing", "unreadable", "malformed"])
def test_missing_or_unreadable_pause_file_sends_notices_and_reports_as_before(tmp_path, monkeypatch, pause_file):
    from gateway import owner_channel_pause

    if pause_file == "unreadable":
        _pause_file().mkdir()  # a directory fails to read as root too
    elif pause_file == "malformed":
        _pause_file().write_text('{"paused": tru', encoding="utf-8")
    assert owner_channel_pause.is_paused() is False

    monkeypatch.setenv("HERMES_KANBAN_DB", str(tmp_path / "pause.db"))
    kb.init_db()
    task_id = _completed_task(HOME_CHAT)
    adapter = _Telegram()
    runner = _runner(adapter)
    _tick(monkeypatch, runner)
    assert [task_id in text for text in adapter.to(HOME_CHAT)] == [True]
    assert _unseen(task_id, HOME_CHAT) == []

    job = {"id": "job-report", "name": "Daily report", "deliver": f"telegram:{HOME_CHAT}"}
    standalone, error = _report("exec-standalone", job)
    live, live_error = _report("exec-live", job, adapters={Platform.TELEGRAM: MagicMock()}, loop=_running_loop())
    assert standalone == {"live": [], "standalone": [f"telegram:{HOME_CHAT}"]}
    assert live == {"live": [f"telegram:{HOME_CHAT}"], "standalone": []}
    assert (error, live_error) == (None, None)
    assert _outcomes("exec-standalone") == _outcomes("exec-live") == [("delivered", None)]


def test_send_again_is_refused_while_paused_and_allowed_after_the_resume(monkeypatch):
    from gateway import owner_channel_pause

    owner_channel_pause.pause()
    job = create_job("Placeholder prompt", "every 1d", name="Daily report", deliver=f"telegram:{HOME_CHAT}")
    execution_id = create_execution(job["id"], source="schedule")["id"]
    handed, _error = _report(execution_id, job)
    finish_execution(execution_id, success=True)
    assert handed == {"live": [], "standalone": []}
    assert _resend_view(execution_id)["eligible"] is True

    handed, chats = _send_again(execution_id, "req-1")
    assert handed == {"live": [], "standalone": []}
    assert chats == [{"position": 0, "state": "failed", "reason": "route_refused"}]
    assert _resend_view(execution_id)["eligible"] is True, "Send again stays available"

    owner_channel_pause.resume()
    handed, chats = _send_again(execution_id, "req-2")
    assert handed == {"live": [], "standalone": [f"telegram:{HOME_CHAT}"]}
    assert [(chat["position"], chat["state"]) for chat in chats] == [(0, "delivered")]
    assert _resend_view(execution_id)["reason"] == "already_delivered"
