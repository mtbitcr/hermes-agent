"""Delivery records for scheduled report runs.

Each test drives the real ``cron.scheduler._deliver_result`` (or the real run
body) with fake platform adapters and a real event loop, then reads back what
the delivery record store says happened to each chat target.
"""

import asyncio
import contextlib
import importlib
import logging
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

from cron import scheduler
from cron.scheduler_preflight import SharedRouteAdapters, _primary_profile_routes_for_current_home
from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import SendResult
from hermes_constants import get_hermes_home, reset_hermes_home_override, set_hermes_home_override

REPO_ROOT = Path(__file__).resolve().parents[2]
CHAT = "-1009876543210"
OTHER_CHAT = "-1001111111111"
ROUTED_CHAT = "-1005555555555"
BLOCKED = "Forbidden: bot was blocked by the user"
STANDALONE_BLOCKED = {"error": "Telegram send failed: " + BLOCKED}
DAY = 86400.0
NOW = 1_790_000_000.0


def _store():
    return importlib.import_module("cron.delivery_record")


class FakeTelegram:
    """Live Telegram adapter double that keeps what it was handed."""

    platform = Platform.TELEGRAM
    MAX_MESSAGE_LENGTH = 4096

    def __init__(self, reply=None, *, block=False, document_reply=None):
        self.reply = reply
        self.block = block
        self.document_reply = document_reply
        self.sent = []
        self.documents = []

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        self.sent.append(content)
        if self.block:
            await asyncio.sleep(3600)
        if isinstance(self.reply, BaseException):
            raise self.reply
        return self.reply if self.reply is not None else SendResult(success=True, message_id="m1")

    async def send_document(self, chat_id, file_path, metadata=None, **_kwargs):
        self.documents.append(file_path)
        if self.document_reply is not None:
            return self.document_reply
        return SendResult(success=True, message_id="d1")


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


@pytest.fixture()
def satellite(tmp_path, monkeypatch):
    """A profile home whose primary gateway routes exactly one Telegram chat to it."""
    root = tmp_path / "root"
    home = root / "profiles" / "reports"
    home.mkdir(parents=True)
    route = {"name": "Ops room", "platform": "telegram", "chat_id": ROUTED_CHAT, "profile": "reports"}
    (root / "config.yaml").write_text(
        yaml.safe_dump({"gateway": {"profile_routes": [route]}}), encoding="utf-8"
    )
    monkeypatch.setattr("hermes_constants.get_default_hermes_root", lambda: root)
    token = set_hermes_home_override(str(home))
    try:
        routes = _primary_profile_routes_for_current_home()
        assert [item.name for item in routes] == ["Ops room"]
        yield routes
    finally:
        reset_hermes_home_override(token)


def _config(*platforms):
    config = MagicMock()
    config.platforms = {platform: PlatformConfig(enabled=True) for platform in platforms}
    config.get_home_channel = lambda _platform: None
    return config


def _job(*chats):
    return {
        "id": "job-report",
        "name": "Daily report",
        "deliver": ",".join(f"telegram:{chat}" for chat in chats),
    }


def _sent_ok(chat_id):
    return {"success": True, "platform": "telegram", "chat_id": chat_id, "message_id": "s1"}


def _deliver(job, content, *, adapters=None, loop=None, standalone_reply=None, config=None,
             execution_id="exec-1"):
    """Run the real _deliver_result inside a recording; return (error, record, standalone sends)."""
    store = _store()
    standalone = []

    async def fake_send(platform, pconfig, chat_id, message, thread_id=None, media_files=None, **_kwargs):
        standalone.append(message)
        return standalone_reply if standalone_reply is not None else _sent_ok(chat_id)

    settings = _config(Platform.TELEGRAM) if config is None else config
    load_settings = (
        {"side_effect": settings} if isinstance(settings, Exception) else {"return_value": settings}
    )
    with (
        patch("gateway.config.load_gateway_config", **load_settings),
        patch("cron.scheduler.load_config", return_value={"cron": {"wrap_response": False}}),
        patch("tools.send_message_tool._send_to_platform", new=fake_send),
        store.recording(execution_id, job["id"]),
    ):
        error = scheduler._deliver_result(job, content, adapters=adapters, loop=loop)
    return error, store.load(execution_id), standalone


def _outcome(record, index=0):
    target = record["targets"][index]
    return target["state"], target["reason"]


class _ShortWait:
    """Future proxy whose result() gives up after 0.2 s instead of the 60 s live budget."""

    def __init__(self, future):
        self._future = future

    def result(self, timeout=None):
        return self._future.result(timeout=0.2)

    def cancel(self):
        return self._future.cancel()


def _short_live_waits(monkeypatch):
    import agent.async_utils as async_utils

    real_schedule = async_utils.safe_schedule_threadsafe

    def schedule(coro, loop, **kwargs):
        future = real_schedule(coro, loop, **kwargs)
        return None if future is None else _ShortWait(future)

    monkeypatch.setattr(async_utils, "safe_schedule_threadsafe", schedule)


def test_live_delivery_reads_delivered_with_the_attachment_handed_over(tmp_path, live_loop):
    report = tmp_path / "report.txt"
    report.write_text("numbers", encoding="utf-8")
    fake = FakeTelegram()

    error, record, standalone = _deliver(
        _job(CHAT), f"Numbers are up.\nMEDIA:{report}",
        adapters={Platform.TELEGRAM: fake}, loop=live_loop,
    )

    assert error is None
    assert fake.sent == ["Numbers are up."]
    assert [Path(path).name for path in fake.documents] == ["report.txt"]
    assert standalone == []
    assert record["job_id"] == "job-report"
    assert _outcome(record) == ("delivered", None)
    assert record["attachments"] == [{"path": fake.documents[0], "is_voice": False}]
    target = record["targets"][0]
    assert (target["platform"], target["chat_id"], target["text"]) == ("telegram", CHAT, "Numbers are up.")


def test_saved_text_is_exactly_what_the_live_adapter_received(live_loop):
    fake = FakeTelegram()

    error, record, standalone = _deliver(
        _job(CHAT), "  Numbers are up.\n\n", adapters={Platform.TELEGRAM: fake}, loop=live_loop,
    )

    assert error is None
    assert standalone == []
    assert _outcome(record) == ("delivered", None)
    assert record["targets"][0]["text"] == fake.sent[0] == "Numbers are up."


def test_saved_text_is_exactly_what_the_standalone_sender_received():
    error, record, standalone = _deliver(_job(CHAT), "  Numbers are up.\n\n")

    assert error is None
    assert _outcome(record) == ("delivered", None)
    assert record["targets"][0]["text"] == standalone[0]
    assert standalone[0].strip() == "Numbers are up."


def test_platform_refusal_on_both_paths_reads_failed(live_loop):
    fake = FakeTelegram(SendResult(success=False, error=BLOCKED))

    error, record, standalone = _deliver(
        _job(CHAT), "Numbers are up.", adapters={Platform.TELEGRAM: fake}, loop=live_loop,
        standalone_reply=STANDALONE_BLOCKED,
    )

    assert error
    assert fake.sent == ["Numbers are up."]
    assert standalone == ["Numbers are up."]
    assert _outcome(record) == ("failed", "platform_refused")


def test_standalone_platform_refusal_without_message_id_reads_failed():
    error, record, standalone = _deliver(
        _job(CHAT), "Numbers are up.", standalone_reply=STANDALONE_BLOCKED,
    )

    assert error
    assert standalone == ["Numbers are up."]
    assert _outcome(record) == ("failed", "platform_refused")


def test_refusal_that_still_carries_a_message_id_reads_unknown():
    error, record, _standalone = _deliver(
        _job(CHAT), "Numbers are up.", standalone_reply=dict(STANDALONE_BLOCKED, message_id="m7"),
    )

    assert error
    assert _outcome(record) == ("unknown", "error_after_handover")


def test_route_refusal_reads_failed(satellite, live_loop):
    fake = FakeTelegram()

    error, record, standalone = _deliver(
        _job(CHAT), "Numbers are up.",
        adapters=SharedRouteAdapters({Platform.TELEGRAM: fake}, satellite), loop=live_loop,
    )

    assert error
    assert fake.sent == []
    assert standalone == []
    assert _outcome(record) == ("failed", "route_refused")


def test_routed_target_without_a_live_main_bot_reads_no_connection(satellite, live_loop):
    error, record, standalone = _deliver(
        _job(ROUTED_CHAT), "Numbers are up.", adapters=SharedRouteAdapters({}, satellite), loop=live_loop,
    )

    assert error
    assert standalone == []
    assert _outcome(record) == ("failed", "no_connection")


def test_platform_without_a_connection_reads_no_connection():
    error, record, standalone = _deliver(_job(CHAT), "Numbers are up.", config=_config())

    assert error
    assert standalone == []
    assert _outcome(record) == ("failed", "no_connection")


def test_unloadable_settings_mark_every_target_failed():
    error, record, standalone = _deliver(
        _job(CHAT, OTHER_CHAT), "Numbers are up.", config=RuntimeError("config.yaml is unreadable"),
    )

    assert error
    assert standalone == []
    assert [target["chat_id"] for target in record["targets"]] == [CHAT, OTHER_CHAT]
    assert [_outcome(record, index) for index in (0, 1)] == [("failed", "settings_not_loaded")] * 2


def test_live_send_that_times_out_reads_unknown(live_loop, monkeypatch):
    _short_live_waits(monkeypatch)
    fake = FakeTelegram(block=True)

    error, record, _standalone = _deliver(
        _job(CHAT), "Numbers are up.", adapters={Platform.TELEGRAM: fake}, loop=live_loop,
        standalone_reply=STANDALONE_BLOCKED,
    )

    assert error
    assert _outcome(record) == ("unknown", "timeout")


def test_error_after_hand_over_reads_unknown(live_loop):
    fake = FakeTelegram(ConnectionResetError("Connection reset by peer"))

    error, record, standalone = _deliver(
        _job(CHAT), "Numbers are up.", adapters={Platform.TELEGRAM: fake}, loop=live_loop,
        standalone_reply=STANDALONE_BLOCKED,
    )

    assert error
    assert fake.sent == ["Numbers are up."]
    assert standalone == ["Numbers are up."]
    assert _outcome(record) == ("unknown", "error_after_handover")


def test_text_sent_but_attachment_lost_reads_partly_sent(tmp_path, live_loop):
    report = tmp_path / "report.txt"
    report.write_text("numbers", encoding="utf-8")
    fake = FakeTelegram(document_reply=SendResult(success=False, error="upload failed"))

    error, record, standalone = _deliver(
        _job(CHAT), f"Numbers are up.\nMEDIA:{report}",
        adapters={Platform.TELEGRAM: fake}, loop=live_loop,
    )

    assert error is None
    assert fake.sent == ["Numbers are up."]
    assert len(fake.documents) == 1
    assert standalone == []
    assert _outcome(record) == ("unknown", "partly_sent")


def test_standalone_send_with_dropped_attachment_reads_partly_sent(tmp_path):
    report = tmp_path / "report.txt"
    report.write_text("numbers", encoding="utf-8")

    error, record, standalone = _deliver(
        _job(CHAT), f"Numbers are up.\nMEDIA:{report}",
        standalone_reply=dict(_sent_ok(CHAT), warnings=["Telegram media upload failed"]),
    )

    assert error is None
    assert len(standalone) == 1
    assert _outcome(record) == ("unknown", "partly_sent")


CHILD = """
import asyncio
import os
import sys
import threading
from unittest.mock import MagicMock, patch

from cron import delivery_record, scheduler
from gateway.config import Platform, PlatformConfig

marker, chat = sys.argv[1], sys.argv[2]


class BlockingTelegram:
    platform = Platform.TELEGRAM
    MAX_MESSAGE_LENGTH = 4096

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        with open(marker + ".tmp", "w", encoding="utf-8") as handle:
            handle.write(content)
        os.replace(marker + ".tmp", marker)
        await asyncio.sleep(3600)


async def no_standalone(*_args, **_kwargs):
    return {"error": "the standalone path must not run in this test"}


loop = asyncio.new_event_loop()
threading.Thread(target=loop.run_forever, daemon=True).start()
ready = threading.Event()
loop.call_soon_threadsafe(ready.set)
ready.wait(30)
config = MagicMock()
config.platforms = {Platform.TELEGRAM: PlatformConfig(enabled=True)}
job = {"id": "job-killed", "name": "Killed report", "deliver": "telegram:" + chat}
with (
    patch("gateway.config.load_gateway_config", return_value=config),
    patch("cron.scheduler.load_config", return_value={"cron": {"wrap_response": False}}),
    patch("tools.send_message_tool._send_to_platform", new=no_standalone),
    delivery_record.recording("exec-killed", job["id"]),
):
    scheduler._deliver_result(
        job, "Numbers are up.", adapters={Platform.TELEGRAM: BlockingTelegram()}, loop=loop
    )
"""


def test_run_killed_mid_send_reads_unknown(tmp_path):
    script = tmp_path / "killed_delivery.py"
    script.write_text(CHILD, encoding="utf-8")
    marker = tmp_path / "handed-over.txt"
    child_errors = tmp_path / "child-stderr.txt"
    with open(child_errors, "wb") as stderr:
        child = subprocess.Popen(
            [sys.executable, str(script), str(marker), CHAT],
            cwd=tmp_path,
            env=dict(os.environ, PYTHONPATH=str(REPO_ROOT)),
            stdout=subprocess.DEVNULL,
            stderr=stderr,
        )
    try:
        deadline = time.monotonic() + 60
        while not marker.exists():
            if child.poll() is not None:
                pytest.fail("child ended before the send: " + child_errors.read_text(errors="replace")[-3000:])
            if time.monotonic() > deadline:
                pytest.fail("child never reached the platform send")
            time.sleep(0.05)
    finally:
        child.kill()
        child.wait(timeout=30)

    record = _store().load("exec-killed")
    assert _outcome(record) == ("unknown", "interrupted")
    assert record["targets"][0]["text"] == marker.read_text(encoding="utf-8")


def _patch_report_run(monkeypatch, results):
    outcomes = iter(results)
    monkeypatch.setattr(scheduler, "run_job", lambda job, **_kwargs: next(outcomes))
    monkeypatch.setattr(scheduler, "save_job_output", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(scheduler, "mark_job_run", lambda *_args, **_kwargs: True)


@contextlib.contextmanager
def _standalone_telegram(sent):
    async def fake_send(platform, pconfig, chat_id, message, thread_id=None, media_files=None, **_kwargs):
        sent.append(message)
        return _sent_ok(chat_id)

    with (
        patch("gateway.config.load_gateway_config", return_value=_config(Platform.TELEGRAM)),
        patch("cron.scheduler.load_config", return_value={"cron": {"wrap_response": False}}),
        patch("tools.send_message_tool._send_to_platform", new=fake_send),
    ):
        yield


def test_two_runs_in_the_same_second_keep_separate_records(monkeypatch):
    from cron.executions import list_executions

    store = _store()
    monkeypatch.setattr(store, "_clock", lambda: NOW)
    _patch_report_run(monkeypatch, [
        (True, "output", "first report", None),
        (True, "output", "second report", None),
    ])
    job = {"id": "job-twice", "name": "Twice", "deliver": f"telegram:{CHAT}"}
    sent = []

    with _standalone_telegram(sent):
        assert scheduler.run_one_job(dict(job)) is True
        assert scheduler.run_one_job(dict(job)) is True

    rows = list_executions(job_id="job-twice")
    assert len(rows) == 2
    assert sent == ["first report", "second report"]
    saved = sorted(store.load(row["id"])["targets"][0]["text"] for row in rows)
    assert saved == sorted(sent)


def test_failure_notice_is_not_recorded(monkeypatch):
    from cron.executions import list_executions

    store = _store()
    _patch_report_run(monkeypatch, [(False, "output", "", "provider exploded")])
    sent = []

    with _standalone_telegram(sent):
        job = {"id": "job-broken", "name": "Broken", "deliver": f"telegram:{CHAT}"}
        assert scheduler.run_one_job(job) is True

    rows = list_executions(job_id="job-broken")
    assert len(rows) == 1
    assert len(sent) == 1
    assert store.load(rows[0]["id"]) is None


def test_nothing_is_recorded_without_an_active_recording():
    store = _store()
    sent = []

    with _standalone_telegram(sent):
        assert scheduler._deliver_result(_job(CHAT), "Numbers are up.") is None

    assert sent == ["Numbers are up."]
    assert store.load("exec-1") is None
    assert not (get_hermes_home() / "cron" / "delivery_records.db").exists()


def test_each_profile_home_keeps_its_own_store(tmp_path):
    store = _store()
    alpha = tmp_path / "profiles" / "alpha"
    beta = tmp_path / "profiles" / "beta"
    alpha.mkdir(parents=True)
    beta.mkdir(parents=True)

    token = set_hermes_home_override(str(alpha))
    try:
        _deliver(_job(CHAT), "Alpha report.", execution_id="exec-alpha")
    finally:
        reset_hermes_home_override(token)

    token = set_hermes_home_override(str(beta))
    try:
        assert store.load("exec-alpha") is None
        assert not (beta / "cron" / "delivery_records.db").exists()
        _deliver(_job(CHAT), "Beta report.", execution_id="exec-beta")
    finally:
        reset_hermes_home_override(token)

    token = set_hermes_home_override(str(alpha))
    try:
        assert store.load("exec-alpha")["targets"][0]["text"] == "Alpha report."
        assert store.load("exec-beta") is None
    finally:
        reset_hermes_home_override(token)
    assert (alpha / "cron" / "delivery_records.db").exists()
    assert (beta / "cron" / "delivery_records.db").exists()
    assert not (get_hermes_home() / "cron" / "delivery_records.db").exists()


def test_recording_error_never_changes_delivery(tmp_path, caplog):
    broken = tmp_path / "broken-home"
    broken.mkdir()
    (broken / "cron").write_text("a file where the cron directory belongs", encoding="utf-8")
    caplog.set_level(logging.WARNING)

    token = set_hermes_home_override(str(broken))
    try:
        error, record, standalone = _deliver(_job(CHAT), "Numbers are up.")
    finally:
        reset_hermes_home_override(token)

    assert error is None
    assert standalone == ["Numbers are up."]
    assert record is None
    assert any(
        entry.name == "cron.delivery_record" and entry.levelno >= logging.WARNING
        for entry in caplog.records
    )


def test_old_records_are_pruned_but_never_inside_seven_days(monkeypatch):
    store = _store()
    monkeypatch.setattr(store, "MAX_RECORDS", 2)

    def deliver_at(moment, execution_id):
        monkeypatch.setattr(store, "_clock", lambda: moment)
        _deliver(_job(CHAT), "Numbers are up.", execution_id=execution_id)

    for index in range(3):
        deliver_at(NOW - 8 * DAY + index, f"old-{index}")
    for index in range(3):
        deliver_at(NOW - 6 * DAY + index, f"recent-{index}")
    deliver_at(NOW, "today")

    assert [store.load(f"old-{index}") for index in range(3)] == [None, None, None]
    assert all(store.load(f"recent-{index}") is not None for index in range(3))
    assert store.load("today") is not None
