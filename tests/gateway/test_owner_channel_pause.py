"""Pause of the owner's Telegram messages (gateway/owner_channel_pause.py).

While paused, the root profile's Telegram home channel (named by the root Hermes home's .env, else
by its config.yaml; never by the process environment) gets no card notice from the kanban notifier
and no scheduled report from ``cron.scheduler._deliver_result``, whichever profile's job or worker
sends it. A paused notice is not kept: its cursor advances without a failure, so nothing old is
sent after the resume, even when the notifier first polls after one or more pause windows. A paused
report is recorded failed ``route_refused`` and stays eligible for Send again, which is itself
refused while paused and allowed after the resume; a pause that begins after the live try still
stops the standalone fallback. The pause file keeps the pause windows of the last 7 days; a missing,
unreadable or malformed one counts as not paused. Every other chat is served as before. The module
is imported inside each test, so on a tree without it each test fails on its own. Chat ids, tokens
and the report text are invented placeholders.
"""

import asyncio
import json
import time
from concurrent.futures import Future
from contextlib import ExitStack, contextmanager
from unittest.mock import MagicMock, patch

import pytest
import yaml

from cron import delivery_record, scheduler
from cron.executions import create_execution, finish_execution, latest_execution
from cron.jobs import create_job
from gateway.config import GatewayConfig, HomeChannel, Platform, PlatformConfig
from gateway.platforms.base import SendResult
from gateway.run import GatewayRunner
from hermes_cli import kanban_db as kb
from hermes_constants import get_default_hermes_root, reset_hermes_home_override, set_hermes_home_override

HOME_CHAT = "5550000000001"
OTHER_CHAT = "5550000000002"
REPORT = "Placeholder report: the numbers are up."
REFUSED = "Forbidden: bot was blocked by the user"  # a whole-chat refusal: nothing reached the chat
_REAL_SLEEP = asyncio.sleep


def _gateway_config():
    """The gateway configuration: Telegram names HOME_CHAT as its home channel; Discord has none."""
    home = HomeChannel(platform=Platform.TELEGRAM, chat_id=HOME_CHAT, name="Home")
    return GatewayConfig(platforms={
        Platform.TELEGRAM: PlatformConfig(enabled=True, home_channel=home),
        Platform.DISCORD: PlatformConfig(enabled=True),
    })


def _name_home(home, chat_id, **telegram):
    """Write ``home``'s config.yaml naming ``chat_id`` as its Telegram home channel, as /sethome does."""
    home.mkdir(parents=True, exist_ok=True)
    section = {"home_channel": {"platform": "telegram", "chat_id": chat_id, "name": "Home"}, **telegram}
    (home / "config.yaml").write_text(yaml.safe_dump({"platforms": {"telegram": section}}), encoding="utf-8")


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
    _name_home(get_default_hermes_root(), HOME_CHAT)  # the root profile's Telegram home on disk
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
def _transports(*, fake_router=True, fake_config=True):
    """The gateway configuration above, its Telegram home also named in the root config.yaml, a
    live router and a standalone sender that both deliver every chat they are handed; yields the
    ``platform:chat`` each path was handed, in order. With ``fake_router=False`` the real
    DeliveryRouter hands the live send to the adapter passed in; with ``fake_config=False`` every
    configuration is read from disk as the test wrote it."""
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
        try:
            future.set_result(asyncio.run(coro))
        except Exception as exc:  # the live send raised on the gateway loop
            future.set_exception(exc)
        return future

    router._deliver_to_platform = deliver_to_platform
    with ExitStack() as stack:
        if fake_config:
            _name_home(get_default_hermes_root(), HOME_CHAT)
            stack.enter_context(patch("gateway.config.load_gateway_config", return_value=_gateway_config()))
        if fake_router:
            stack.enter_context(patch("gateway.delivery.DeliveryRouter", return_value=router))
        stack.enter_context(patch("cron.scheduler.load_config", return_value={"cron": {"wrap_response": False}}))
        stack.enter_context(patch("asyncio.run_coroutine_threadsafe", side_effect=run_now))
        stack.enter_context(patch("tools.send_message_tool._send_to_platform", new=standalone))
        yield handed


def _report(execution_id, job, **live):
    """One run of ``job`` delivered through the real ``_deliver_result``: what each path was
    handed and the returned error."""
    with _transports() as handed, delivery_record.recording(execution_id, job["id"]):
        error = scheduler._deliver_result(job, REPORT, **live)
    return handed, error


class _LiveTelegram:
    """The gateway's live Telegram adapter: ``send_does()`` stands in for what its send does."""

    MAX_MESSAGE_LENGTH = 4096

    def __init__(self, send_does):
        self.send_does = send_does

    async def send(self, chat_id, content, metadata=None):
        return self.send_does()


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


def _profile_outcomes(home, job_id):
    """How each chat of ``job_id``'s latest run in the profile at ``home`` ended."""
    token = set_hermes_home_override(str(home))
    try:
        return _outcomes(latest_execution(job_id)["id"])
    finally:
        reset_hermes_home_override(token)


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


# --- the root profile's home, a pause begun mid-delivery, a notice first seen after the resume -----

@pytest.mark.parametrize("paused", [True, False], ids=["paused", "unpaused"])
@pytest.mark.parametrize("satellite_home", [None, OTHER_CHAT], ids=["satellite-names-no-home", "satellite-names-other"])
def test_satellite_report_to_the_serving_gateways_home_is_refused_while_paused(
    tmp_path, monkeypatch, satellite_home, paused,
):
    """Through the real multiplex ticker (``InProcessCronScheduler`` → ``cron.scheduler.tick`` →
    ``_deliver_result``), the planning satellite's report to the chat the root profile's .env
    names as the Telegram home is refused ``route_refused`` while paused, whether planning's own
    configuration names no home or another chat; that other chat is not the owner's and is
    delivered. The route rule refuses an unrouted chat as before; unpaused, both go."""
    from gateway import owner_channel_pause
    from tests.cron.test_cron_multiplex_shared_route_delivery import (
        OWNER_CHAT, STRAY_CHAT, _primary_adapter, _tick_planning_briefs,
    )

    root = tmp_path / "root"
    planning = root / "profiles" / "planning"
    root.mkdir()
    # The root profile's .env names OWNER_CHAT as its Telegram home, as /sethome writes it.
    (root / ".env").write_text(
        f"TELEGRAM_BOT_TOKEN=5550000000:placeholder-gateway-token\nTELEGRAM_HOME_CHANNEL={OWNER_CHAT}\n",
        encoding="utf-8",
    )
    if satellite_home is not None:
        _name_home(planning, satellite_home)

    def run_job(job, **kwargs):
        if paused:
            owner_channel_pause.pause()  # before the report is delivered
        return True, "out", "Weekly report", None

    primary = _primary_adapter()
    delivery_errors, standalone = _tick_planning_briefs(
        tmp_path, monkeypatch, True,
        adapters={Platform.TELEGRAM: primary},
        planning_adapters={},  # planning holds no bot of its own
        run_job=run_job,
        routes=[
            {"name": name, "platform": "telegram", "chat_id": chat_id, "profile": "planning", "delivery_only": True}
            for name, chat_id in (("owner-reports", OWNER_CHAT), ("other-reports", OTHER_CHAT))
        ],
        jobs=[
            {"id": f"{name}-brief", "name": f"{name} brief", "deliver": f"telegram:{chat_id}"}
            for name, chat_id in (("owner", OWNER_CHAT), ("other", OTHER_CHAT), ("stray", STRAY_CHAT))
        ],
    )

    assert owner_channel_pause.is_paused() is paused
    assert standalone == []
    assert delivery_errors["other-brief"] is None
    assert _profile_outcomes(planning, "other-brief") == [("delivered", None)]
    assert "not named by an enabled route" in delivery_errors["stray-brief"], "the route rule as before"
    assert _profile_outcomes(planning, "stray-brief") == [("failed", "route_refused")]
    if paused:
        assert primary.sent == [OTHER_CHAT]
        assert "paused by the owner" in delivery_errors["owner-brief"]
        assert _profile_outcomes(planning, "owner-brief") == [("failed", "route_refused")]
    else:
        assert sorted(primary.sent) == sorted([OWNER_CHAT, OTHER_CHAT])
        assert delivery_errors["owner-brief"] is None
        assert _profile_outcomes(planning, "owner-brief") == [("delivered", None)]


@pytest.mark.parametrize("paused", [True, False], ids=["paused", "unpaused"])
@pytest.mark.parametrize("path", ["standalone", "live"])
@pytest.mark.parametrize("root_file", [".env", "config.yaml"], ids=["root-env", "root-config-yaml"])
def test_root_profiles_home_decides_whatever_the_process_environment_names(monkeypatch, root_file, path, paused):
    """The root profile names HOME_CHAT as its Telegram home in its .env, or in its config.yaml
    while its .env has no home line; the process environment names OTHER_CHAT, as a worker's may.
    Through the real ``_deliver_result`` on either path, HOME_CHAT is refused ``route_refused``
    while paused and OTHER_CHAT is delivered; unpaused, both are delivered."""
    from gateway import owner_channel_pause

    root = get_default_hermes_root()
    env = "TELEGRAM_BOT_TOKEN=5550000000:placeholder-gateway-token\n"
    if root_file == ".env":
        env += f"TELEGRAM_HOME_CHANNEL={HOME_CHAT}\n"
    else:
        _name_home(root, HOME_CHAT)
    (root / ".env").write_text(env, encoding="utf-8")
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "5550000001:placeholder-worker-token")
    monkeypatch.setenv("TELEGRAM_HOME_CHANNEL", OTHER_CHAT)
    if paused:
        owner_channel_pause.pause()

    job = {"id": "job-report", "name": "Daily report", "deliver": f"telegram:{HOME_CHAT},telegram:{OTHER_CHAT}"}
    live = {"adapters": {Platform.TELEGRAM: MagicMock()}, "loop": _running_loop()} if path == "live" else {}
    with _transports(fake_config=False) as handed, delivery_record.recording("exec-root-home", job["id"]):
        error = scheduler._deliver_result(job, REPORT, **live)

    expected = {"live": [], "standalone": []}
    expected[path] = [f"telegram:{OTHER_CHAT}"] if paused else [f"telegram:{HOME_CHAT}", f"telegram:{OTHER_CHAT}"]
    assert handed == expected
    if paused:
        assert "paused by the owner" in error
        assert _outcomes("exec-root-home") == [("failed", "route_refused"), ("delivered", None)]
    else:
        assert error is None
        assert _outcomes("exec-root-home") == [("delivered", None), ("delivered", None)]


@pytest.mark.parametrize("config_yaml", ["missing", "unreadable", "malformed", "non-mapping"])
def test_missing_unreadable_or_malformed_root_config_yaml_names_no_home_chat(monkeypatch, config_yaml):
    """The root .env has no home line and the root config.yaml is missing, unreadable (a directory
    fails to read as root too), truncated or not a mapping, while the worker's profile and the
    process environment name HOME_CHAT: no chat is the owner home chat and nothing raises, so while
    paused a report to HOME_CHAT is not refused and its notice is not dropped."""
    from gateway import owner_channel_pause

    root = get_default_hermes_root()
    (root / ".env").write_text("TELEGRAM_BOT_TOKEN=5550000000:placeholder-gateway-token\n", encoding="utf-8")
    config = root / "config.yaml"
    if config_yaml == "unreadable":
        config.mkdir()
    elif config_yaml == "malformed":
        config.write_text("platforms: {telegram: {home_channel: {chat_id: '%s'" % HOME_CHAT, encoding="utf-8")
    elif config_yaml == "non-mapping":
        section = {"platforms": {"telegram": {"home_channel": {"chat_id": HOME_CHAT}}}}
        config.write_text(yaml.safe_dump([section]), encoding="utf-8")
    worker_home = root / "profiles" / "coder"
    _name_home(worker_home, HOME_CHAT)
    monkeypatch.setenv("HERMES_HOME", str(worker_home))
    monkeypatch.setenv("TELEGRAM_HOME_CHANNEL", HOME_CHAT)
    owner_channel_pause.pause()

    assert owner_channel_pause._owner_home_chat_id() == ""
    assert owner_channel_pause.refuses_report("telegram", HOME_CHAT) is False
    assert owner_channel_pause.drops_notice("telegram", HOME_CHAT, int(time.time())) is False


@pytest.mark.parametrize("paused", [True, False], ids=["paused", "unpaused"])
@pytest.mark.parametrize("live", ["scheduling_failed", "refused", "raised", "raised_in_flight"])
def test_pause_begun_after_the_live_try_stops_the_standalone_fallback(live, paused):
    """The pause begins after the check before the live try: the scheduler reads it again before
    the standalone fallback, so a home-chat report the live try proved unsent is refused
    ``route_refused`` and not sent; one the live try may have sent keeps that outcome and is not
    sent again. Unpaused, every case falls back to the standalone send as before."""
    from gateway import owner_channel_pause

    def live_send():
        if paused:
            owner_channel_pause.pause()
        if live == "refused":
            return SendResult(success=False, error=REFUSED)
        raise RuntimeError("Connection reset by peer" if live == "raised_in_flight" else REFUSED)

    def scheduling_fails(coro, _loop):
        if paused:
            owner_channel_pause.pause()
        raise RuntimeError("Event loop is closed")

    job = {"id": "job-report", "name": "Daily report", "deliver": f"telegram:{HOME_CHAT}"}
    with ExitStack() as stack:
        handed = stack.enter_context(_transports(fake_router=False))
        if live == "scheduling_failed":
            stack.enter_context(patch("asyncio.run_coroutine_threadsafe", side_effect=scheduling_fails))
        stack.enter_context(delivery_record.recording("exec-live", job["id"]))
        error = scheduler._deliver_result(
            job, REPORT, adapters={Platform.TELEGRAM: _LiveTelegram(live_send)}, loop=_running_loop(),
        )

    assert owner_channel_pause.is_paused() is paused
    if paused:
        assert handed["standalone"] == [], "no new send once the pause began"
        assert "paused by the owner" in error
        assert _outcomes("exec-live") == (
            [("unknown", "error_after_handover")] if live == "raised_in_flight" else [("failed", "route_refused")]
        )
    else:
        assert handed["standalone"] == [f"telegram:{HOME_CHAT}"], "the fallback as before"
        assert (error, _outcomes("exec-live")) == (None, [("delivered", None)])


@pytest.mark.parametrize("paused", [True, False], ids=["paused", "unpaused"])
def test_pause_begun_before_the_standalone_retry_stops_it(paused):
    """``asyncio.run`` refuses the standalone send before starting it and the pause begins before
    the retry on a fresh thread: the retry reads the pause again, so the home chat gets nothing and
    the report reads ``route_refused``. Unpaused, the retry sends as before."""
    from gateway import owner_channel_pause

    real_run = asyncio.run
    calls = []

    def run(coro, *args, **kwargs):
        calls.append(coro)
        if len(calls) > 1:
            return real_run(coro, *args, **kwargs)
        if paused:
            owner_channel_pause.pause()
        raise RuntimeError("asyncio.run() cannot be called from a running event loop")

    job = {"id": "job-report", "name": "Daily report", "deliver": f"telegram:{HOME_CHAT}"}
    with _transports() as handed, patch("asyncio.run", side_effect=run), \
            delivery_record.recording("exec-retry", job["id"]):
        error = scheduler._deliver_result(job, REPORT)

    if paused:
        assert handed["standalone"] == []
        assert "paused by the owner" in error
        assert _outcomes("exec-retry") == [("failed", "route_refused")]
    else:
        assert handed["standalone"] == [f"telegram:{HOME_CHAT}"]
        assert (error, _outcomes("exec-retry")) == (None, [("delivered", None)])


def test_notice_created_while_paused_is_dropped_by_the_first_poll_after_the_resume(tmp_path, monkeypatch):
    """No notifier poll runs while paused: the first poll after the resume still drops the home
    chat's notice created during the pause, advancing its cursor without a failure, while the other
    chat gets it; a notice created after the resume reaches the home chat as before."""
    from gateway import owner_channel_pause

    monkeypatch.setenv("HERMES_KANBAN_DB", str(tmp_path / "pause.db"))
    kb.init_db()
    adapter = _Telegram()
    runner = _runner(adapter)

    owner_channel_pause.pause()
    held = _completed_task(HOME_CHAT, OTHER_CHAT)
    owner_channel_pause.resume()
    _tick(monkeypatch, runner)

    assert adapter.to(HOME_CHAT) == [], "nothing old after the resume"
    assert _unseen(held, HOME_CHAT) == [], "the cursor advances: the notice is not kept"
    assert [held in text for text in adapter.to(OTHER_CHAT)] == [True], "every other chat as before"
    assert runner._kanban_sub_fail_counts == {}, "a dropped notice is no failure"

    fresh = _completed_task(HOME_CHAT)
    _tick(monkeypatch, runner)
    [notice] = adapter.to(HOME_CHAT)
    assert fresh in notice and held not in notice
    assert runner._kanban_sub_fail_counts == {}


def test_notices_created_in_two_pause_windows_are_dropped_by_the_first_poll_after_both(tmp_path, monkeypatch):
    """Two pause windows with no notifier poll between them: the first poll after the second
    resume drops the home chat's notices from both windows, advancing both cursors without a
    failure, while the other chat gets both; a notice created after the resume reaches the home
    chat as before."""
    from gateway import owner_channel_pause

    monkeypatch.setenv("HERMES_KANBAN_DB", str(tmp_path / "pause.db"))
    kb.init_db()
    adapter = _Telegram()
    runner = _runner(adapter)

    owner_channel_pause.pause()
    first = _completed_task(HOME_CHAT, OTHER_CHAT)
    owner_channel_pause.resume()
    owner_channel_pause.pause()
    second = _completed_task(HOME_CHAT, OTHER_CHAT)
    owner_channel_pause.resume()
    _tick(monkeypatch, runner)

    assert adapter.to(HOME_CHAT) == [], "nothing from either pause window after the resume"
    assert _unseen(first, HOME_CHAT) == _unseen(second, HOME_CHAT) == [], "both cursors advance"
    other = adapter.to(OTHER_CHAT)
    assert len(other) == 2, "every other chat as before"
    assert {task for task in (first, second) for text in other if task in text} == {first, second}
    assert runner._kanban_sub_fail_counts == {}, "a dropped notice is no failure"

    fresh = _completed_task(HOME_CHAT)
    _tick(monkeypatch, runner)
    [notice] = adapter.to(HOME_CHAT)
    assert fresh in notice and first not in notice and second not in notice
    assert runner._kanban_sub_fail_counts == {}


def test_worker_environment_reads_the_root_pause_file_and_the_serving_gateways_home(tmp_path, monkeypatch):
    """A kanban worker's environment: its profile home, its board pinned, its task set, and its
    own Telegram settings, in its profile's files and loaded into the process env, naming
    OTHER_CHAT as the home. The module still reads the pause file in the root Hermes home and the
    home chat the root profile's files name, not the worker's."""
    from gateway import owner_channel_pause

    root = tmp_path / "root"
    worker_home = root / "profiles" / "coder"
    (worker_home / "cron").mkdir(parents=True)
    (root / ".env").write_text(
        f"TELEGRAM_BOT_TOKEN=5550000000:placeholder-gateway-token\nTELEGRAM_HOME_CHANNEL={HOME_CHAT}\n",
        encoding="utf-8",
    )
    _name_home(root, HOME_CHAT)
    (worker_home / ".env").write_text(
        f"TELEGRAM_BOT_TOKEN=5550000001:placeholder-worker-token\nTELEGRAM_HOME_CHANNEL={OTHER_CHAT}\n",
        encoding="utf-8",
    )
    _name_home(worker_home, OTHER_CHAT)
    monkeypatch.setenv("HERMES_HOME", str(root))
    owner_channel_pause.pause()  # the owner's pause, in the root pause file
    (worker_home / "owner_channel_pause.json").write_text('{"windows": []}', encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(worker_home))
    monkeypatch.setenv("HERMES_KANBAN_DB", str(tmp_path / "boards" / "ops" / "kanban.db"))
    monkeypatch.setenv("HERMES_KANBAN_BOARD", "ops")
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_placeholder")
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "5550000001:placeholder-worker-token")
    monkeypatch.setenv("TELEGRAM_HOME_CHANNEL", OTHER_CHAT)

    job = {"id": "job-report", "name": "Daily report", "deliver": f"telegram:{HOME_CHAT},telegram:{OTHER_CHAT}"}
    with _transports(fake_config=False) as handed, delivery_record.recording("exec-worker", job["id"]):
        error = scheduler._deliver_result(job, REPORT)
    assert handed == {"live": [], "standalone": [f"telegram:{OTHER_CHAT}"]}
    assert "paused by the owner" in error
    assert _outcomes("exec-worker") == [("failed", "route_refused"), ("delivered", None)]

    assert owner_channel_pause.is_paused() is True
    assert owner_channel_pause.is_owner_home_chat("telegram", HOME_CHAT) is True
    assert owner_channel_pause.is_owner_home_chat("telegram", OTHER_CHAT) is False
    owner_channel_pause.resume()
    assert owner_channel_pause.is_paused() is False
    assert (root / "owner_channel_pause.json").is_file()
    assert (worker_home / "owner_channel_pause.json").read_text(encoding="utf-8") == '{"windows": []}'


# --- the pause windows -----------------------------------------------------------------------------

def test_pause_and_resume_keep_the_pause_windows_of_the_last_seven_days():
    """The pause file keeps every pause window of the last 7 days as ``[start, end]`` in time
    order: a pause opens one, a resume closes it, a later pause or resume never changes a closed
    one, and one that ended more than 7 days ago is dropped at the next write. A home-chat notice
    created inside any kept window is dropped, and while paused every one is."""
    from gateway import owner_channel_pause

    _name_home(get_default_hermes_root(), HOME_CHAT)
    day = 24 * 3600
    now = int(time.time())
    stale, recent = [now - 9 * day, now - 8 * day], [now - 2 * day, now - day]
    _pause_file().write_text(json.dumps({"windows": [stale, recent]}), encoding="utf-8")

    def windows():
        return json.loads(_pause_file().read_text(encoding="utf-8"))["windows"]

    def drops(created_at, chat_id=HOME_CHAT):
        return owner_channel_pause.drops_notice("telegram", chat_id, created_at)

    owner_channel_pause.resume()  # not paused: nothing to close
    assert windows() == [stale, recent]

    owner_channel_pause.pause()
    kept, opened = windows()
    assert kept == recent, "the window that ended more than 7 days ago is dropped; the recent one stays"
    assert opened[0] >= now and opened[1] is None
    owner_channel_pause.pause()  # already paused: nothing to open
    assert windows() == [recent, opened]
    assert owner_channel_pause.is_paused() is True
    assert drops(now - 3 * day) is True, "while paused, an older notice is dropped too"

    owner_channel_pause.resume()
    kept, first = windows()
    assert kept == recent and first[0] == opened[0] < first[1]
    assert owner_channel_pause.is_paused() is False
    owner_channel_pause.pause()
    owner_channel_pause.resume()
    kept, closed, second = windows()
    assert [kept, closed] == [recent, first], "a later pause and resume never change a closed window"
    assert first[1] <= second[0] < second[1], "in time order"

    # A window holds its start but not its end; the last two may meet, so only their outer bounds.
    for start, end in (recent, [first[0], second[1]]):
        assert (drops(start - 1), drops(start), drops(end - 1), drops(end)) == (False, True, True, False)
    assert drops(now - 3 * day) is False and drops(stale[0]) is False
    assert drops(first[0], OTHER_CHAT) is False, "every other chat as before"


@pytest.mark.parametrize("pause_file", ["missing", "unreadable", "malformed", "wrong-shape"])
def test_missing_unreadable_or_malformed_pause_file_has_no_windows(pause_file):
    """A missing, unreadable (a directory fails to read as root too), truncated or wrongly shaped
    pause file reads as no windows without raising: not paused, and nothing is held."""
    from gateway import owner_channel_pause

    _name_home(get_default_hermes_root(), HOME_CHAT)
    now = int(time.time())
    if pause_file == "unreadable":
        _pause_file().mkdir()
    elif pause_file == "malformed":
        _pause_file().write_text('{"windows": [[%d, nu' % now, encoding="utf-8")
    elif pause_file == "wrong-shape":
        _pause_file().write_text(json.dumps({"windows": [[str(now), None]]}), encoding="utf-8")

    assert owner_channel_pause.is_paused() is False
    assert owner_channel_pause.drops_notice("telegram", HOME_CHAT, now) is False
    assert owner_channel_pause.refuses_report("telegram", HOME_CHAT) is False
