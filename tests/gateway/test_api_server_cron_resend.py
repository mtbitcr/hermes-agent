"""The gateway route that sends a failed scheduled report again.

``POST /api/cron/executions/{execution_id}/resend`` (and its ``/p/<profile>/`` mirror) claims the
re-send of a run's failed chats, sends the saved text again through the live delivery function and
records how each chat ended. Runs are delivered first through the real
``cron.scheduler._deliver_result`` with a faked standalone sender that answers each chat as
scripted, so every run's delivery record holds real outcomes. The route is called over HTTP through
the adapter's own route table and profile-prefix middleware. Chat ids, keys and report text are
invented placeholders.
"""

import asyncio
import json
import logging
import os
import threading
import time
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
from zoneinfo import ZoneInfo

import pytest
import yaml
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from cron import delivery_record, scheduler
from cron.executions import create_execution, finish_execution, list_executions
from cron.jobs import create_job, update_job
from gateway.config import Platform, PlatformConfig
from gateway.platforms.api_server import APIServerAdapter
from gateway.platforms.base import SendResult
from hermes_constants import reset_hermes_home_override, set_hermes_home_override
from tools import send_message_tool

# The real standalone sender, kept before the ``chats`` fixture replaces it.
REAL_SEND_TO_PLATFORM = send_message_tool._send_to_platform

KEY = "3f9c1e7a5b2d4c6e8f0a1b3c5d7e9f21"
FITNESS_KEY = "8d2b4f6a1c3e5a7b9d0f2e4c6a8b1d3f"
ZONE = "Pacific/Auckland"
REPORT = "Placeholder report: the numbers are up."
CHAT_OK = "-1009000000001"
CHAT_BAD = "-1009000000002"
ROUTE_CHAT = "5550000000001"
OTHER_CHAT = "5550000000002"
REFUSED = "Telegram send failed: Forbidden: bot was blocked by the user"
DAY = 86400.0


class _Chats:
    """The faked standalone sender: answers each chat as scripted and keeps what it was sent."""

    def __init__(self):
        self.answers = {}
        self.sent = []
        self.gate = None
        self.started = threading.Event()

    async def send(self, platform, pconfig, chat_id, message, thread_id=None, media_files=None, **_kwargs):
        self.sent.append((str(chat_id), message))
        self.started.set()
        if self.gate is not None:
            await asyncio.to_thread(self.gate.wait, 10)
        answer = self.answers.get(str(chat_id), "delivered")
        if answer == "delivered":
            return {"success": True, "message_id": "m1", "chat_id": chat_id}
        if answer == "unknown":
            # A refusal that still carries a message id may have been sent.
            return {"error": REFUSED, "message_id": "m7", "chat_id": chat_id}
        return {"error": REFUSED, "chat_id": chat_id}

    def to(self, chat_id):
        return [message for chat, message in self.sent if chat == chat_id]


class _LiveChat:
    """A live gateway adapter: keeps what it was handed and answers as scripted. ``unconfirmed`` is
    a refusal that still carries a message id; ``hangs`` never confirms. A text longer than
    ``limit`` is refused."""

    def __init__(self, answer="delivered", *, splits_long_messages=True, limit=4096):
        self.answer = answer
        self.splits_long_messages = splits_long_messages
        self.MAX_MESSAGE_LENGTH = limit
        self.sent = []
        self.started = threading.Event()

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        self.sent.append((str(chat_id), content))
        self.started.set()
        if self.answer == "hangs":
            await asyncio.sleep(10)
        if self.answer == "raises":
            raise ConnectionResetError("placeholder connection reset")
        if self.answer == "no_answer":
            return None
        if self.answer == "unconfirmed":
            return SendResult(success=False, error="confirmation lost", message_id="m7")
        if len(content) > self.MAX_MESSAGE_LENGTH:
            return SendResult(success=False, error="Bad Request: message is too long")
        return SendResult(success=True, message_id="m1")


def _confirmation_times_out(monkeypatch, live):
    """The live text send starts, but its confirmation does not come back within the wait."""
    from agent import async_utils

    real = async_utils.safe_schedule_threadsafe

    class _Unconfirmed:
        def __init__(self, future):
            self._future = future

        def result(self, timeout=None):
            assert live.started.wait(5)
            return self._future.result(timeout=0.05)

        def cancel(self):
            return self._future.cancel()

    def schedule(coro, loop, **kwargs):
        routed = getattr(getattr(coro, "cr_code", None), "co_name", "") == "_deliver_to_platform"
        future = real(coro, loop, **kwargs)
        return _Unconfirmed(future) if routed and future is not None else future

    monkeypatch.setattr(async_utils, "safe_schedule_threadsafe", schedule)


class _Transport:
    """The Telegram transport under the real standalone sender: keeps each call with the thread it
    ran in, raises the scripted errors in turn, then succeeds."""

    def __init__(self, *errors):
        self.errors = list(errors)
        self.calls = []

    async def send(self, token, chat_id, message, **_kwargs):
        self.calls.append((str(chat_id), threading.get_ident()))
        if self.errors:
            raise self.errors.pop(0)
        return {"success": True, "message_id": "m1", "chat_id": chat_id}


def _standalone_transport(monkeypatch, send):
    """The re-send goes through the real standalone sender, whose Telegram transport is ``send``."""
    monkeypatch.setattr(send_message_tool, "_send_to_platform", REAL_SEND_TO_PLATFORM)
    monkeypatch.setattr(send_message_tool, "_send_telegram", send)


def _first_run_refused(monkeypatch):
    """The first ``asyncio.run`` handed the standalone send raises before it starts it, as it does
    in a thread where a loop already runs; every other call runs as usual. The idents of the
    threads it refused in."""
    real = asyncio.run
    refused = []

    def run(main, **kwargs):
        if not refused and getattr(getattr(main, "cr_code", None), "co_name", "") == "_send_to_platform":
            refused.append(threading.get_ident())
            raise RuntimeError("asyncio.run() cannot be called from a running event loop")
        return real(main, **kwargs)

    monkeypatch.setattr(asyncio, "run", run)
    return refused


@pytest.fixture(autouse=True)
def home(tmp_path, monkeypatch):
    """The per-test HERMES_HOME the suite's isolation made, as the launch profile's root, with its
    own home directory and the owner's time zone set."""
    import hermes_time

    root = Path(os.environ["HERMES_HOME"])
    (root / "profiles").mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_TIMEZONE", ZONE)
    monkeypatch.setattr("hermes_constants.get_default_hermes_root", lambda: root)
    hermes_time.reset_cache()
    yield root
    hermes_time.reset_cache()


@pytest.fixture
def chats(monkeypatch):
    chats = _Chats()
    config = MagicMock()
    config.platforms = {
        Platform.TELEGRAM: PlatformConfig(enabled=True),
        Platform.DISCORD: PlatformConfig(enabled=True),
    }
    config.get_home_channel = lambda _platform: None
    chats.platforms = config.platforms
    monkeypatch.setattr("gateway.config.load_gateway_config", lambda *a, **k: config)
    monkeypatch.setattr("cron.scheduler.load_config", lambda *a, **k: {"cron": {"wrap_response": True}})
    monkeypatch.setattr("tools.send_message_tool._send_to_platform", chats.send)
    return chats


@pytest.fixture
def runner(monkeypatch):
    runner = SimpleNamespace(
        adapters={},
        _profile_adapters={},
        config=SimpleNamespace(multiplex_profiles=False, multiplex_profile_allowlist=None),
        _draining=False,
        _external_drain_active=False,
    )
    monkeypatch.setattr("gateway.run._gateway_runner_ref", lambda: runner)
    return runner


@pytest.fixture
def adapter(runner):
    adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={"key": KEY}))
    adapter.gateway_runner = runner
    return adapter


def _app(adapter):
    """The adapter's routes as ``connect()`` serves them: native paths plus ``/p/<profile>/`` mirrors."""
    app = web.Application(middlewares=[adapter._make_profile_prefix_middleware()])
    for method, path, handler in adapter._http_route_table():
        app.router.add_route(method, path, handler)
        app.router.add_route(method, f"/p/{{profile}}{path}", handler)
    app["api_server_adapter"] = adapter
    return app


def _client(adapter):
    return TestClient(TestServer(_app(adapter)))


async def _resend(cli, execution_id, request_id, *, profile=None, key=KEY):
    prefix = f"/p/{profile}" if profile else ""
    headers = {"Authorization": f"Bearer {key}"} if key else {}
    return await cli.post(
        f"{prefix}/api/cron/executions/{execution_id}/resend",
        headers=headers,
        json={"request_id": request_id},
    )


def _job(*chat_ids, platform="telegram"):
    return create_job(
        "Placeholder prompt",
        "every 1d",
        name="Daily report",
        deliver=",".join(f"{platform}:{chat}" for chat in chat_ids),
    )


def _run(job, text=REPORT, *, home=None):
    """One scheduled run of ``job`` delivered through the live delivery function; its execution id."""
    token = set_hermes_home_override(str(home)) if home is not None else None
    try:
        execution = create_execution(job["id"], source="schedule")
        with delivery_record.recording(execution["id"], job["id"]):
            scheduler._deliver_result(job, text)
        finish_execution(execution["id"], success=True)
        return execution["id"]
    finally:
        if token is not None:
            reset_hermes_home_override(token)


async def _failed_run(chats, *chat_ids, answers=None, text=REPORT):
    """A run whose chats answered as ``answers`` (default: every chat refused); the sender is reset."""
    chats.answers = dict(answers or {chat: "failed" for chat in chat_ids})
    job = _job(*chat_ids)
    execution_id = await asyncio.to_thread(_run, job, text)
    first = {chat: chats.to(chat) for chat in chat_ids}
    chats.sent.clear()
    chats.started.clear()
    chats.answers = {}
    return job, execution_id, first


def _view(execution_id):
    return delivery_record.history_deliveries([{"id": execution_id, "status": "completed"}])[0]


def _run_date(execution_id):
    created = delivery_record.load(execution_id)["created_at"]
    return datetime.fromtimestamp(created, ZoneInfo(ZONE)).date().isoformat()


async def _settled(execution_id, timeout=10.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        attempts = _view(execution_id)["resend"]["attempts"]
        if attempts and all(attempt["state"] != "in_progress" for attempt in attempts):
            return attempts
        await asyncio.sleep(0.02)
    raise AssertionError("the re-send never finished")


def _carries_nothing_private(body):
    text = json.dumps(body)
    for private in (CHAT_OK, CHAT_BAD, ROUTE_CHAT, OTHER_CHAT, REPORT, KEY, FITNESS_KEY, REFUSED):
        assert private not in text


@pytest.mark.asyncio
async def test_failed_chat_is_sent_again_once_with_the_top_line(adapter, chats, caplog):
    job, execution_id, first = await _failed_run(
        chats, CHAT_OK, CHAT_BAD, answers={CHAT_OK: "delivered", CHAT_BAD: "failed"}
    )
    runs_before = len(list_executions(job_id=job["id"]))
    caplog.clear()
    caplog.set_level(logging.DEBUG)

    async with _client(adapter) as cli:
        response = await _resend(cli, execution_id, "req-1")
        body = await response.json()

    assert response.status == 200, body
    assert body["execution_id"] == execution_id
    assert body["state"] == "delivered"
    assert [(target["state"], target["reason"]) for target in body["targets"]] == [("delivered", None)]
    assert body["resend"] == {"eligible": False, "reason": "already_delivered"}
    # Only the failed chat is sent again, once: the text first produced with one top line.
    assert [chat for chat, _message in chats.sent] == [CHAT_BAD]
    top, rest = chats.sent[0][1].split("\n", 1)
    assert top == f"Sent again because it did not arrive on {_run_date(execution_id)}."
    assert rest == first[CHAT_BAD][0]
    # The job never ran again and the history shows the finished attempt.
    assert len(list_executions(job_id=job["id"])) == runs_before
    attempts = _view(execution_id)["resend"]["attempts"]
    assert [(a["attempt_id"], a["state"]) for a in attempts] == [(body["attempt_id"], "delivered")]
    _carries_nothing_private(body)
    # The re-send's own log lines carry reason codes only.
    lines = [r.getMessage() for r in caplog.records if r.name.startswith(("cron.", "gateway.platforms.api_server"))]
    assert any("Cron re-send" in line for line in lines)
    _carries_nothing_private(lines)


@pytest.mark.asyncio
async def test_repeated_request_id_returns_first_attempt_without_sending(adapter, chats):
    _job_, execution_id, _first = await _failed_run(chats, CHAT_BAD)

    async with _client(adapter) as cli:
        first = await _resend(cli, execution_id, "req-1")
        first_body = await first.json()
        sent = list(chats.sent)
        again = await _resend(cli, execution_id, "req-1")
        again_body = await again.json()

    assert first.status == 200, first_body
    assert again.status == 200, again_body
    assert again_body["attempt_id"] == first_body["attempt_id"]
    assert again_body["state"] == first_body["state"] == "delivered"
    assert again_body["targets"] == first_body["targets"]
    assert chats.sent == sent and len(sent) == 1
    assert len(_view(execution_id)["resend"]["attempts"]) == 1


@pytest.mark.asyncio
async def test_repeated_request_id_after_its_attachment_is_gone_returns_first_attempt(adapter, chats, tmp_path):
    chart = tmp_path / "placeholder-chart.txt"
    chart.write_text("placeholder chart", encoding="utf-8")
    # The chat's platform stays off, for the run and the first re-send: nothing is ever handed over.
    chats.platforms[Platform.TELEGRAM] = PlatformConfig(enabled=False)
    _job_, execution_id, _first = await _failed_run(chats, CHAT_BAD, text=f"{REPORT}\nMEDIA:{chart}")
    assert delivery_record.load(execution_id)["attachments"]

    async with _client(adapter) as cli:
        first = await _resend(cli, execution_id, "same")
        first_body = await first.json()
        chart.unlink()
        again = await _resend(cli, execution_id, "same")
        again_body = await again.json()

    assert first.status == 200, first_body
    assert [(t["state"], t["reason"]) for t in first_body["targets"]] == [("failed", "no_connection")]
    assert again.status == 200, again_body
    assert again_body["attempt_id"] == first_body["attempt_id"]
    assert again_body["state"] == first_body["state"] == "failed"
    assert again_body["targets"] == first_body["targets"]
    assert chats.sent == []
    assert len(_view(execution_id)["resend"]["attempts"]) == 1


@pytest.mark.asyncio
async def test_two_simultaneous_requests_send_once(adapter, chats):
    _job_, execution_id, _first = await _failed_run(chats, CHAT_BAD)
    chats.gate = threading.Event()

    async with _client(adapter) as cli:
        first = asyncio.create_task(_resend(cli, execution_id, "req-a"))
        assert await asyncio.to_thread(chats.started.wait, 5)
        second = await _resend(cli, execution_id, "req-b")
        second_body = await second.json()
        chats.gate.set()
        first_response = await first
        first_body = await first_response.json()

    assert second.status == 409
    assert second_body == {"detail": {"code": "not_eligible", "reason": "in_progress"}}
    assert first_response.status == 200, first_body
    assert first_body["state"] == "delivered"
    assert len(chats.to(CHAT_BAD)) == 1
    assert len(_view(execution_id)["resend"]["attempts"]) == 1


@pytest.mark.asyncio
async def test_send_outliving_the_request_answers_202_and_records_its_result(adapter, chats, monkeypatch):
    monkeypatch.setattr("gateway.platforms.api_server.CRON_RESEND_WAIT_SECONDS", 0.2)
    _job_, execution_id, _first = await _failed_run(chats, CHAT_BAD)
    chats.gate = threading.Event()

    async with _client(adapter) as cli:
        response = await _resend(cli, execution_id, "req-slow")
        body = await response.json()
        assert response.status == 202, body
        assert body == {"request_id": "req-slow", "state": "in_progress"}

        chats.gate.set()
        attempts = await _settled(execution_id)
        assert [attempt["state"] for attempt in attempts] == ["delivered"]

        # Asking again with the same request id now answers the recorded result, sending nothing.
        again = await _resend(cli, execution_id, "req-slow")
        again_body = await again.json()

    assert again.status == 200, again_body
    assert again_body["attempt_id"] == attempts[0]["attempt_id"]
    assert again_body["state"] == "delivered"
    assert len(chats.to(CHAT_BAD)) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "answer, reason",
    [
        ("unconfirmed", "error_after_handover"),
        ("no_answer", "error_after_handover"),
        ("raises", "error_after_handover"),
        ("hangs", "timeout"),
    ],
)
async def test_live_send_already_out_is_not_sent_again_another_way(
    adapter, chats, runner, monkeypatch, answer, reason
):
    _job_, execution_id, _first = await _failed_run(chats, CHAT_BAD)
    live = _LiveChat(answer)
    runner.adapters = {Platform.TELEGRAM: live}
    if answer == "hangs":
        _confirmation_times_out(monkeypatch, live)

    async with _client(adapter) as cli:
        response = await _resend(cli, execution_id, "req-1")
        body = await response.json()

    assert response.status == 200, body
    # The live send was handed the chat once and may have reached it: nothing sends it again.
    assert [chat for chat, _message in live.sent] == [CHAT_BAD]
    assert chats.sent == []
    assert body["state"] == "unknown"
    assert [(t["state"], t["reason"]) for t in body["targets"]] == [("unknown", reason)]
    attempts = _view(execution_id)["resend"]["attempts"]
    assert [(a["attempt_id"], a["state"]) for a in attempts] == [(body["attempt_id"], "unknown")]
    _carries_nothing_private(body)


@pytest.mark.asyncio
async def test_standalone_send_that_raised_once_started_is_not_sent_again(adapter, chats, monkeypatch):
    _job_, execution_id, _first = await _failed_run(chats, CHAT_BAD)
    # No live adapter: the real standalone sender runs, and its transport raises once it is called.
    transport = _Transport(RuntimeError("placeholder transport failure"))
    _standalone_transport(monkeypatch, transport.send)

    async with _client(adapter) as cli:
        response = await _resend(cli, execution_id, "req-1")
        body = await response.json()

    assert response.status == 200, body
    # The send started and may have reached the chat: no fresh thread sends it again.
    assert [chat for chat, _thread in transport.calls] == [CHAT_BAD]
    assert body["state"] == "unknown"
    assert [(t["state"], t["reason"]) for t in body["targets"]] == [("unknown", "error_after_handover")]
    attempts = _view(execution_id)["resend"]["attempts"]
    assert [(a["attempt_id"], a["state"]) for a in attempts] == [(body["attempt_id"], "unknown")]
    _carries_nothing_private(body)


@pytest.mark.asyncio
async def test_standalone_send_that_never_started_is_sent_once_from_a_fresh_thread(
    adapter, chats, monkeypatch
):
    _job_, execution_id, _first = await _failed_run(chats, CHAT_BAD)
    refused = _first_run_refused(monkeypatch)
    transport = _Transport()
    _standalone_transport(monkeypatch, transport.send)

    async with _client(adapter) as cli:
        response = await _resend(cli, execution_id, "req-1")
        body = await response.json()

    assert response.status == 200, body
    # The first run never started the send, so the fresh-thread fallback sends it, once.
    assert len(refused) == 1
    assert [chat for chat, _thread in transport.calls] == [CHAT_BAD]
    assert transport.calls[0][1] != refused[0]
    assert body["state"] == "delivered"
    assert [(t["state"], t["reason"]) for t in body["targets"]] == [("delivered", None)]
    attempts = _view(execution_id)["resend"]["attempts"]
    assert [(a["attempt_id"], a["state"]) for a in attempts] == [(body["attempt_id"], "delivered")]


@pytest.mark.asyncio
@pytest.mark.parametrize("fits", [True, False])
async def test_report_just_under_the_limit_reaches_a_non_splitting_chat_unchanged(
    adapter, chats, runner, monkeypatch, fits
):
    monkeypatch.setattr("cron.scheduler.load_config", lambda *a, **k: {"cron": {"wrap_response": False}})
    text = ("placeholder report text " * 170)[:3979] + "."
    # The run's chat was not reachable: nothing was handed over and the chat reads failed.
    chats.platforms[Platform.TELEGRAM] = PlatformConfig(enabled=False)
    _job_, execution_id, _first = await _failed_run(chats, CHAT_BAD, text=text)
    chats.platforms[Platform.TELEGRAM] = PlatformConfig(enabled=True)
    assert delivery_record.load(execution_id)["targets"][0]["text"] == text
    # A chat that takes one message of at most ``limit`` characters and never splits one.
    live = _LiveChat(splits_long_messages=False, limit=4096 if fits else 4000)
    runner.adapters = {Platform.TELEGRAM: live}

    async with _client(adapter) as cli:
        response = await _resend(cli, execution_id, "req-1")
        body = await response.json()

    assert response.status == 200, body
    top = f"Sent again because it did not arrive on {_run_date(execution_id)}."
    assert live.sent == [(CHAT_BAD, f"{top}\n{text}")]
    assert chats.sent == []
    state = "delivered" if fits else "unknown"
    assert body["state"] == state
    assert [t["state"] for t in body["targets"]] == [state]
    assert [a["state"] for a in _view(execution_id)["resend"]["attempts"]] == [state]


@pytest.mark.asyncio
async def test_failing_real_adapter_send_logs_no_chat_address_or_report_text(
    adapter, chats, runner, monkeypatch, caplog
):
    from plugins.platforms.telegram.adapter import TelegramAdapter

    _job_, execution_id, _first = await _failed_run(chats, CHAT_BAD)
    bot = TelegramAdapter(PlatformConfig(enabled=True, token="placeholder-token", extra={}))
    bot._bot = MagicMock()
    bot._bot.send_chat_action = AsyncMock()

    async def refused(chat_id, chunk, *_args, **_kwargs):
        raise RuntimeError(f"placeholder refusal of {chunk!r} for chat {chat_id}")

    monkeypatch.setattr(bot, "_should_attempt_rich", lambda *_a, **_k: False)
    monkeypatch.setattr(bot, "_send_chunk_with_retries", refused)
    runner.adapters = {Platform.TELEGRAM: bot}
    caplog.clear()
    caplog.set_level(logging.DEBUG)

    async with _client(adapter) as cli:
        response = await _resend(cli, execution_id, "req-1")
        body = await response.json()

    assert response.status == 200, body
    # Every record any logger made, from the route down to the adapter's own send.
    formatter = logging.Formatter()
    assert any("Cron re-send" in record.getMessage() for record in caplog.records)
    for record in caplog.records:
        logged = " ".join((
            record.getMessage(), repr(record.args), record.exc_text or "",
            formatter.formatException(record.exc_info) if record.exc_info else "",
        ))
        for private in (CHAT_BAD, "numbers are up"):
            assert private not in logged, (record.name, record.lineno)
    assert [(t["state"], t["reason"]) for t in body["targets"]] == [("unknown", "error_after_handover")]


@pytest.mark.asyncio
async def test_fresh_thread_send_logs_no_chat_address_or_report_text(adapter, chats, monkeypatch, caplog):
    _job_, execution_id, _first = await _failed_run(chats, CHAT_BAD)
    refused = _first_run_refused(monkeypatch)
    calls = []

    async def logs_then_fails(token, chat_id, message, **_kwargs):
        calls.append(threading.get_ident())
        try:
            raise ConnectionError(f"placeholder refusal of {message!r} for chat {chat_id}")
        except ConnectionError:
            logging.getLogger("tools.send_message_senders").exception(
                "placeholder send of %r to chat %s failed", message, chat_id
            )
            raise

    _standalone_transport(monkeypatch, logs_then_fails)
    caplog.clear()
    caplog.set_level(logging.DEBUG)

    async with _client(adapter) as cli:
        response = await _resend(cli, execution_id, "req-1")
        body = await response.json()

    assert response.status == 200, body
    # Sent once, by the fresh-thread fallback, whose own logger line was made and captured.
    assert len(refused) == len(calls) == 1 and calls[0] != refused[0]
    assert any(record.name == "tools.send_message_senders" for record in caplog.records)
    # Every record any logger made, the fresh thread's included.
    formatter = logging.Formatter()
    assert any("Cron re-send" in record.getMessage() for record in caplog.records)
    for record in caplog.records:
        logged = " ".join((
            record.getMessage(), repr(record.args), record.exc_text or "",
            formatter.formatException(record.exc_info) if record.exc_info else "",
        ))
        for private in (CHAT_BAD, "numbers are up"):
            assert private not in logged, (record.name, record.lineno)
    assert [(t["state"], t["reason"]) for t in body["targets"]] == [("unknown", "error_after_handover")]


def _older_than_resend_window(monkeypatch):
    real = delivery_record._clock
    monkeypatch.setattr(delivery_record, "_clock", lambda: real() - 8 * DAY)
    return lambda: monkeypatch.setattr(delivery_record, "_clock", real)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "case, reason",
    [
        ("delivered", "already_delivered"),
        ("unknown", "outcome_unknown"),
        ("expired", "output_expired"),
        ("older_than_7_days", "output_expired"),
    ],
)
async def test_run_that_may_not_be_sent_again_answers_409_with_its_reason(
    adapter, chats, monkeypatch, case, reason
):
    answers = {"delivered": "delivered", "unknown": "unknown"}.get(case, "failed")
    # A report too large to keep has no saved text left to send again. Its chat is turned down
    # before hand-over: a refusal of a text long enough to go in parts may have sent a part.
    text = "x" * (delivery_record.MAX_SAVED_TEXT_BYTES + 1) if case == "expired" else REPORT
    if case == "expired":
        chats.platforms[Platform.TELEGRAM] = PlatformConfig(enabled=False)
    restore = _older_than_resend_window(monkeypatch) if case == "older_than_7_days" else None
    _job_, execution_id, _first = await _failed_run(chats, CHAT_BAD, answers={CHAT_BAD: answers}, text=text)
    if restore:
        restore()
    chats.platforms[Platform.TELEGRAM] = PlatformConfig(enabled=True)

    async with _client(adapter) as cli:
        response = await _resend(cli, execution_id, "req-1")
        body = await response.json()

    assert response.status == 409
    assert body == {"detail": {"code": "not_eligible", "reason": reason}}
    assert chats.sent == []
    assert _view(execution_id)["resend"]["attempts"] == []


def _write_routes(root, **route):
    """The launch profile's routes: one route lending its bot to the ``fitness`` profile."""
    base = {"name": "reports", "platform": "discord", "chat_id": ROUTE_CHAT, "profile": "fitness"}
    base.update(route)
    (root / "config.yaml").write_text(
        yaml.safe_dump({"gateway": {"multiplex_profiles": True, "profile_routes": [base]}}),
        encoding="utf-8",
    )


def _fitness_home(root):
    fitness = root / "profiles" / "fitness"
    (fitness / "cron").mkdir(parents=True, exist_ok=True)
    (fitness / "config.yaml").write_text(yaml.safe_dump({"timezone": ZONE}), encoding="utf-8")
    (fitness / ".env").write_text(f"API_SERVER_KEY={FITNESS_KEY}\n", encoding="utf-8")
    return fitness


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["disabled", "re-pointed"])
async def test_disabled_or_repointed_route_turns_chat_down_and_records_it(
    adapter, chats, runner, home, monkeypatch, change
):
    fitness = _fitness_home(home)
    _write_routes(home)
    # The run: the route named the chat, and the chat refused the report.
    chats.answers = {ROUTE_CHAT: "failed"}
    token = set_hermes_home_override(str(fitness))
    try:
        job = _job(ROUTE_CHAT, platform="discord")
    finally:
        reset_hermes_home_override(token)
    execution_id = await asyncio.to_thread(_run, job, home=fitness)
    assert chats.to(ROUTE_CHAT)
    chats.sent.clear()
    chats.answers = {}

    # The multiplex gateway serves the profile; the launch bot is live, the route has changed since.
    primary = MagicMock()
    primary.send = MagicMock(side_effect=AssertionError("the launch bot must not be used"))
    runner.adapters = {Platform.DISCORD: primary}
    runner.config.multiplex_profiles = True
    monkeypatch.setattr("agent.secret_scope.is_multiplex_active", lambda: True)
    monkeypatch.setattr(
        "hermes_cli.profiles.profiles_to_serve",
        lambda multiplex=True, profile_allowlist=None: [("default", home), ("fitness", fitness)],
    )
    if change == "disabled":
        _write_routes(home, enabled=False)
    else:
        _write_routes(home, chat_id=OTHER_CHAT)

    async with _client(adapter) as cli:
        response = await _resend(cli, execution_id, "req-1", profile="fitness", key=FITNESS_KEY)
        body = await response.json()

    assert response.status == 200, body
    assert body["state"] == "failed"
    assert [(t["state"], t["reason"]) for t in body["targets"]] == [("failed", "route_refused")]
    assert chats.sent == []
    primary.send.assert_not_called()
    _carries_nothing_private(body)
    token = set_hermes_home_override(str(fitness))
    try:
        attempts = _view(execution_id)["resend"]["attempts"]
    finally:
        reset_hermes_home_override(token)
    assert [(a["attempt_id"], a["state"]) for a in attempts] == [(body["attempt_id"], "failed")]


@pytest.mark.asyncio
async def test_chat_the_job_no_longer_targets_is_turned_down_and_recorded(adapter, chats):
    job, execution_id, _first = await _failed_run(chats, CHAT_BAD)
    update_job(job["id"], {"deliver": f"telegram:{CHAT_OK}"})

    async with _client(adapter) as cli:
        response = await _resend(cli, execution_id, "req-1")
        body = await response.json()

    assert response.status == 200, body
    assert body["state"] == "failed"
    assert [(t["state"], t["reason"]) for t in body["targets"]] == [("failed", "route_refused")]
    assert chats.sent == []
    _carries_nothing_private(body)
    attempts = _view(execution_id)["resend"]["attempts"]
    assert [(a["attempt_id"], a["state"]) for a in attempts] == [(body["attempt_id"], "failed")]


@pytest.mark.asyncio
async def test_restart_during_a_resend_leaves_the_attempt_unknown(adapter, chats, monkeypatch):
    from gateway.run import _fence_cron_resends

    monkeypatch.setattr("gateway.platforms.api_server.CRON_RESEND_WAIT_SECONDS", 0.2)
    _job_, execution_id, _first = await _failed_run(chats, CHAT_BAD)
    chats.gate = threading.Event()

    async with _client(adapter) as cli:
        response = await _resend(cli, execution_id, "req-1")
        assert response.status == 202
        assert await asyncio.to_thread(chats.started.wait, 5)

        # The gateway restarts while that send is still out: its next start fences the attempt.
        _fence_cron_resends(SimpleNamespace(multiplex_profiles=False))
        attempts = _view(execution_id)["resend"]["attempts"]
        assert [attempt["state"] for attempt in attempts] == ["unknown"]

        again = await _resend(cli, execution_id, "req-2")
        again_body = await again.json()
        chats.gate.set()
        await asyncio.sleep(0.3)

    assert again.status == 409
    assert again_body == {"detail": {"code": "not_eligible", "reason": "outcome_unknown"}}
    # The cut-off send can never be claimed again, and its late result does not change the record.
    assert len(chats.to(CHAT_BAD)) == 1
    assert [attempt["state"] for attempt in _view(execution_id)["resend"]["attempts"]] == ["unknown"]


def test_start_fence_reaches_every_served_profile_store(chats, home, monkeypatch):
    from gateway.run import _fence_cron_resends

    fitness = _fitness_home(home)
    homes = {"default": home, "fitness": fitness}
    claimed = {}
    for name, profile_home in homes.items():
        chats.answers = {CHAT_BAD: "failed"}
        token = set_hermes_home_override(str(profile_home))
        try:
            job = _job(CHAT_BAD)
        finally:
            reset_hermes_home_override(token)
        execution_id = _run(job, home=profile_home)
        token = set_hermes_home_override(str(profile_home))
        try:
            claim = delivery_record.claim_resend({"id": execution_id, "status": "completed"}, "req-1")
        finally:
            reset_hermes_home_override(token)
        assert claim["claimed"]
        claimed[name] = execution_id
    monkeypatch.setattr(
        "hermes_cli.profiles.profiles_to_serve",
        lambda multiplex=True, profile_allowlist=None: list(homes.items()),
    )

    _fence_cron_resends(SimpleNamespace(multiplex_profiles=True))

    for name, profile_home in homes.items():
        token = set_hermes_home_override(str(profile_home))
        try:
            view = _view(claimed[name])
        finally:
            reset_hermes_home_override(token)
        assert [attempt["state"] for attempt in view["resend"]["attempts"]] == ["unknown"], name
        assert view["resend"]["reason"] == "outcome_unknown"


@pytest.mark.asyncio
async def test_run_not_found_in_that_profile_answers_404(adapter, chats, runner, home, monkeypatch):
    _job_, execution_id, _first = await _failed_run(chats, CHAT_BAD)
    worker = home / "profiles" / "worker"
    (worker / "cron").mkdir(parents=True)
    (worker / ".env").write_text(f"API_SERVER_KEY={FITNESS_KEY}\n", encoding="utf-8")
    runner.config.multiplex_profiles = True
    monkeypatch.setattr(
        "hermes_cli.profiles.profiles_to_serve",
        lambda multiplex=True, profile_allowlist=None: [("default", home), ("worker", worker)],
    )

    async with _client(adapter) as cli:
        missing = await _resend(cli, "0" * 32, "req-1")
        missing_body = await missing.json()
        elsewhere = await _resend(cli, execution_id, "req-1", profile="worker", key=FITNESS_KEY)
        elsewhere_body = await elsewhere.json()

    assert missing.status == 404
    assert elsewhere.status == 404
    assert missing_body == elsewhere_body == {"detail": {"code": "not_found"}}
    assert chats.sent == []
    assert _view(execution_id)["resend"] == {"eligible": True, "reason": None, "attempts": []}


@pytest.mark.asyncio
@pytest.mark.parametrize("gateway", ["missing", "draining"])
async def test_unavailable_gateway_answers_503_and_claims_nothing(adapter, chats, runner, monkeypatch, gateway):
    _job_, execution_id, _first = await _failed_run(chats, CHAT_BAD)
    if gateway == "missing":
        adapter.gateway_runner = None
        monkeypatch.setattr("gateway.run._gateway_runner_ref", lambda: None)
    else:
        runner._draining = True

    async with _client(adapter) as cli:
        response = await _resend(cli, execution_id, "req-1")
        body = await response.json()

    assert response.status == 503
    assert body == {"detail": {"code": "gateway_unavailable"}}
    assert chats.sent == []
    assert _view(execution_id)["resend"] == {"eligible": True, "reason": None, "attempts": []}


@pytest.mark.asyncio
async def test_unauthenticated_or_malformed_request_claims_nothing(adapter, chats):
    _job_, execution_id, _first = await _failed_run(chats, CHAT_BAD)

    async with _client(adapter) as cli:
        unauthenticated = await _resend(cli, execution_id, "req-1", key=None)
        wrong_key = await _resend(cli, execution_id, "req-1", key=FITNESS_KEY)
        no_request_id = await cli.post(
            f"/api/cron/executions/{execution_id}/resend",
            headers={"Authorization": f"Bearer {KEY}"},
            json={},
        )

    assert unauthenticated.status == 401
    assert wrong_key.status == 401
    assert no_request_id.status == 400
    assert chats.sent == []
    assert _view(execution_id)["resend"] == {"eligible": True, "reason": None, "attempts": []}
