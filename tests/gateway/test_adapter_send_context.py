"""A cron re-send's log policy reaches the email and Feishu adapters' pool threads.

While a failed scheduled report is sent again, ``cron.scheduler`` sets ``_RESENDING`` and its log
record factory withholds the text of every record made in that context. The email adapter's SMTP
sends and the Feishu adapter's blocking SDK calls run in pool threads, which must run inside a copy
of the caller's context. The re-send is driven from the real scheduler through the real email
adapter; the only fake is the SMTP connection where mail would leave the process. Addresses, hosts
and passwords are placeholders on reserved example domains.
"""

import asyncio
import contextlib
import contextvars
import logging
import os
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from cron import delivery_record, scheduler
from cron.executions import create_execution, finish_execution
from cron.jobs import create_job
from gateway.config import Platform, PlatformConfig
from plugins.platforms.email import adapter as email_module
from plugins.platforms.email.adapter import EmailAdapter
from plugins.platforms.feishu.adapter import FeishuAdapter

RECIPIENT = "recipient@example.com"
REPORT = "Placeholder report: the numbers are up."
EMAIL_SETTINGS = {
    "EMAIL_ADDRESS": "hermes@example.org",
    "EMAIL_PASSWORD": "placeholder-password",
    "EMAIL_IMAP_HOST": "imap.example.net",
    "EMAIL_SMTP_HOST": "smtp.example.net",
    "EMAIL_SMTP_PORT": "587",
}
EMAIL_ADAPTER_FILE = os.path.normcase(os.path.abspath(email_module.__file__))
CALLER = contextvars.ContextVar("placeholder_caller", default="placeholder-default")


class _Outbox:
    """Stands in for ``smtplib.SMTP`` and ``smtplib.SMTP_SSL`` where mail would leave the process:
    connects nowhere and keeps each message's ``To`` header in ``sent``."""

    sent: list = []

    def __init__(self, host, port, **_kwargs):
        self.host, self.port = host, port

    def starttls(self, **_kwargs):
        return 220, b"placeholder ready"

    def login(self, _user, _password):
        return 235, b"placeholder ok"

    def send_message(self, msg):
        self.sent.append(msg["To"])
        return {}

    def quit(self):
        return 221, b"placeholder bye"

    def close(self):
        pass


@pytest.fixture(autouse=True)
def home(tmp_path, monkeypatch):
    """The per-test HERMES_HOME the suite's isolation made, as the launch profile's root."""
    root = Path(os.environ["HERMES_HOME"])
    (root / "profiles").mkdir(exist_ok=True)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setattr("hermes_constants.get_default_hermes_root", lambda: root)
    return root


@pytest.fixture
def outbox(monkeypatch):
    """The outgoing mail call faked and the email adapter's placeholder settings set: what was sent."""
    sent = []
    fake = type("PlaceholderSMTP", (_Outbox,), {"sent": sent})
    monkeypatch.setattr(email_module.smtplib, "SMTP", fake)
    monkeypatch.setattr(email_module.smtplib, "SMTP_SSL", fake)
    for key, value in EMAIL_SETTINGS.items():
        monkeypatch.setenv(key, value)
    return sent


@pytest.fixture
def platforms(monkeypatch):
    """A placeholder gateway config whose email platform starts off: its platform map."""
    config = MagicMock()
    config.platforms = {Platform.EMAIL: PlatformConfig(enabled=False)}
    config.get_home_channel = lambda _platform: None
    monkeypatch.setattr("gateway.config.load_gateway_config", lambda *a, **k: config)
    monkeypatch.setattr("cron.scheduler.load_config", lambda *a, **k: {"cron": {"wrap_response": True}})
    return config.platforms


class _Every(logging.Handler):
    def __init__(self):
        super().__init__(level=1)
        self.records = []

    def emit(self, record):
        self.records.append(record)


@contextlib.contextmanager
def _every_record():
    """Every record any logger makes meanwhile, at the lowest level, with nothing disabled and every
    logger propagating to the root; the logging state is restored afterwards."""
    root = logging.getLogger()
    manager = logging.Logger.manager
    loggers = [item for item in list(manager.loggerDict.values()) if isinstance(item, logging.Logger)]
    saved = [(item, item.level, item.propagate, item.disabled) for item in loggers]
    root_level, disabled_below = root.level, manager.disable
    handler = _Every()
    logging.disable(logging.NOTSET)
    root.setLevel(1)
    for item in loggers:
        item.setLevel(logging.NOTSET)
        item.propagate, item.disabled = True, False
    root.addHandler(handler)
    try:
        yield handler.records
    finally:
        root.removeHandler(handler)
        for item, level, propagate, disabled in saved:
            item.setLevel(level)
            item.propagate, item.disabled = propagate, disabled
        root.setLevel(root_level)
        logging.disable(disabled_below)


def _carrying(records, text):
    """Where each record carrying ``text`` in its rendered message, raw message or args was made."""
    return [
        (record.name, record.funcName, record.lineno, record.threadName)
        for record in records
        if text in record.getMessage() or text in str(record.msg) or text in repr(record.args)
    ]


def _made_by_send_email(records):
    return [
        record for record in records
        if record.funcName == "_send_email"
        and os.path.normcase(os.path.abspath(record.pathname)) == EMAIL_ADAPTER_FILE
    ]


def _failed_run(platforms, text):
    """One scheduled run of a job reporting to RECIPIENT by email, delivered through the live
    delivery function while the email platform was off, so nothing was handed over; the platform
    is on again afterwards. Its execution id."""
    job = create_job("Placeholder prompt", "every 1d", name="Daily report", deliver=f"email:{RECIPIENT}")
    execution = create_execution(job["id"], source="schedule")
    with delivery_record.recording(execution["id"], job["id"]):
        scheduler._deliver_result(job, text)
    finish_execution(execution["id"], success=True)
    platforms[Platform.EMAIL] = PlatformConfig(enabled=True)
    return execution["id"]


def _runner(adapter):
    """The running gateway as the re-send route reads it: its live adapter map holds ``adapter``."""
    return SimpleNamespace(
        adapters={Platform.EMAIL: adapter},
        _profile_adapters={},
        config=SimpleNamespace(multiplex_profiles=False, multiplex_profile_allowlist=None),
        _draining=False,
        _external_drain_active=False,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("images", [0, 3], ids=["single_reply", "several_images"])
async def test_email_resend_logs_no_recipient(outbox, platforms, tmp_path, images):
    paths = [tmp_path / f"placeholder-{number}.png" for number in range(images)]
    for path in paths:
        path.write_bytes(b"placeholder image")
    text = "\n".join([REPORT, *(f"MEDIA:{path}" for path in paths)])
    execution_id = await asyncio.to_thread(_failed_run, platforms, text)
    saved = delivery_record.load(execution_id)
    assert [Path(item["path"]).name for item in saved.get("attachments") or ()] == [path.name for path in paths]
    assert outbox == []
    adapter = EmailAdapter(PlatformConfig(enabled=True))
    runner = _runner(adapter)

    # The re-send as the gateway route runs it: claimed, then sent from a worker thread through the
    # live adapter map while the gateway loop runs; the images go one fallback notice each.
    with _every_record() as records:
        claim = await asyncio.to_thread(scheduler.claim_report_resend, execution_id, "placeholder-request-1")
        assert claim["claimed"] and claim["attempt"] is not None, claim
        await asyncio.to_thread(
            scheduler.resend_report,
            execution_id,
            claim["attempt"],
            adapters=scheduler.resend_adapters(runner),
            loop=asyncio.get_running_loop(),
        )

    # The mail really went out: the report, then one notice per image, each to RECIPIENT.
    assert outbox == [RECIPIENT] * (1 + images)
    # The adapter's own send line was made for each of them, and captured.
    assert len(_made_by_send_email(records)) == 1 + images
    # No record, wherever it was made, carries the recipient.
    assert _carrying(records, RECIPIENT) == []


@pytest.mark.asyncio
async def test_feishu_blocking_call_sees_the_callers_context():
    adapter = object.__new__(FeishuAdapter)
    seen = []

    def sdk_call(value):
        seen.append((CALLER.get(), threading.current_thread().name))
        return value

    token = CALLER.set("placeholder-caller-value")
    try:
        result = await adapter._run_blocking(sdk_call, "placeholder-result")
    finally:
        CALLER.reset(token)
        adapter._shutdown_sdk_executor()

    assert result == "placeholder-result"
    # It ran in the adapter's own pool, and saw the caller's value there.
    assert [name.startswith("hermes-feishu-sdk") for _value, name in seen] == [True]
    assert [value for value, _name in seen] == ["placeholder-caller-value"]


@pytest.mark.asyncio
async def test_ordinary_email_send_still_logs_the_recipient(outbox):
    adapter = EmailAdapter(PlatformConfig(enabled=True))

    with _every_record() as records:
        result = await adapter.send(RECIPIENT, REPORT)

    assert result.success, result
    assert outbox == [RECIPIENT]
    lines = [record.getMessage() for record in _made_by_send_email(records)]
    assert len(lines) == 1 and RECIPIENT in lines[0], lines
