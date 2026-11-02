"""Owner channel routes: the Telegram row of the owner app.

``GET /api/owner-channel`` reads whether the owner's Telegram home channel is
set and whether owner messages are paused, ``PUT /api/owner-channel/paused``
pauses or resumes them and ``POST /api/owner-channel/test`` sends one fixed
test message. Only the owner app's connections credential is admitted, every
call writes one audit row, and the routes serve only in the default profile's
process. Every token and chat id is synthetic, and every send goes to a fake
send function, or through the real one to a fake Telegram transport, never to
Telegram.
"""

import errno
import importlib
import importlib.util
import json
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

ROUTER = "hermes_cli.web_routers.owner_channel"
CHANNEL = "/api/owner-channel"
PAUSED = "/api/owner-channel/paused"
TEST = "/api/owner-channel/test"
TEST_TEXT = "Test from Raphael: your message channel works."
#: Synthetic, with digits 5 to 9 only, so a body holding any of them shows a leak.
HOME_CHAT = "-98765987659876"
BOT_TOKEN = "8899988999:SyntheticOwnerChannelToken"
HOME_NAME = "SyntheticOwnerName"
#: A pause start with digits 1 to 4 only, none of them a digit of the home chat.
PAUSE_START = 1234123412
#: What a hosting profile and the process environment name; never the owner's home or token.
OTHER_CHAT = "-12341234123412"
OTHER_TOKEN = "1122334455:SyntheticHostingProfileToken"


class _Quiet(SimpleNamespace):
    """A namespace whose repr names no value, so failure output never prints a token."""

    def __repr__(self):
        return "<quiet>"


class _Clock:
    """The routes' clock in a test: it stands still until the test moves it."""

    def __init__(self):
        self.now = 1_000_000.0

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


def _clock(monkeypatch, routes):
    clock = _Clock()
    monkeypatch.setattr(routes, "_now", clock)
    return clock


def _router(monkeypatch):
    """The routes' module, asserted inside a test so its absence fails that test; its test state
    lives in the per-test root home."""
    assert importlib.util.find_spec(ROUTER) is not None, f"{ROUTER} does not exist"
    module = importlib.import_module(ROUTER)
    module._register_owner_channel_routes()
    return module


def _name_home(root, where):
    """Name HOME_CHAT as the root profile's Telegram home in its .env or its config.yaml."""
    if where == ".env":
        (root / ".env").write_text(
            f"TELEGRAM_BOT_TOKEN={BOT_TOKEN}\nTELEGRAM_HOME_CHANNEL={HOME_CHAT}\n", encoding="utf-8"
        )
    else:
        (root / ".env").write_text(f"TELEGRAM_BOT_TOKEN={BOT_TOKEN}\n", encoding="utf-8")
        home = {"platform": "telegram", "chat_id": HOME_CHAT, "name": HOME_NAME}
        (root / "config.yaml").write_text(
            yaml.safe_dump({"platforms": {"telegram": {"home_channel": home}}}), encoding="utf-8"
        )


def _sent_answer():
    """What the real tool answers after a send to the home channel, chat id in its note."""
    return json.dumps({
        "success": True,
        "message_id": "4242",
        "note": f"Sent to telegram home channel (chat_id: {HOME_CHAT})",
        "mirrored": True,
    })


@pytest.fixture()
def home(tmp_path, monkeypatch, request):
    """The root Hermes home; it names no Telegram home yet. The dashboard runs with it as its
    HERMES_HOME or, parametrized ``"profile"``, with a profile home under it."""
    root = tmp_path / "hermes"
    root.mkdir()
    hermes_home = root
    if getattr(request, "param", "root") == "profile":
        hermes_home = root / "profiles" / "worker"
        hermes_home.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.delenv("TELEGRAM_HOME_CHANNEL", raising=False)
    return root


@pytest.fixture()
def sends(monkeypatch):
    """Every call of the fake send function; ``answer`` sets what it returns or raises, and
    ``real`` is the real send function."""
    calls = []
    answer = {"do": _sent_answer}

    def fake(args, **kw):
        calls.append(dict(args))
        return answer["do"]()

    real = importlib.import_module("tools.send_message_tool").send_message_tool
    monkeypatch.setattr("tools.send_message_tool.send_message_tool", fake)
    return _Quiet(calls=calls, answer=answer, real=real)


@pytest.fixture()
def telegram(sends, monkeypatch):
    """The real send function with only its final Telegram transport faked: ``deliveries``
    holds the token, chat id and text of each delivery, and ``after`` runs once it is made.
    The process environment holds the root bot token, as the default profile's process loads
    it from the root .env at its start."""
    deliveries = []
    after = {"do": lambda: None}

    async def deliver(token, chat_id, message, **kw):
        deliveries.append((token, chat_id, message))
        after["do"]()
        return {"success": True, "platform": "telegram", "chat_id": chat_id, "message_id": "4242"}

    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", BOT_TOKEN)
    monkeypatch.setattr("tools.send_message_tool.send_message_tool", sends.real)
    monkeypatch.setattr("tools.send_message_tool._send_telegram", deliver)
    return _Quiet(deliveries=deliveries, after=after)


@pytest.fixture()
def dash(home, sends, tmp_path, monkeypatch):
    """The real dashboard app, an owner session and a token for two machine surfaces."""
    from fastapi.testclient import TestClient

    from hermes_cli import web_server
    from hermes_cli.dashboard_auth import register_provider
    from hermes_cli.dashboard_auth.registry import restore_registration, snapshot_registration
    from plugins.dashboard_auth.raphael_workspace import (
        ConnectionsManageTokenProvider,
        WorkspaceReadTokenProvider,
        token_store,
    )

    token_dir = tmp_path / "machine-tokens"
    token_dir.mkdir(mode=0o700)

    def issue(name, surface):
        path = token_dir / f"{name}.token"
        record = token_store.issue(out_path=path, surface=surface)
        return record, path.read_text(encoding="utf-8").strip()

    connections_record, connections = issue("connections", token_store.CONNECTIONS_SURFACE)
    bearers = {
        "connections": connections,
        "connections_wrong_secret": f"{connections_record.token_id}.wrong-secret",
        "workspace": issue("workspace", token_store.WORKSPACE_SURFACE)[1],
        "session_as_bearer": web_server._SESSION_TOKEN,
    }
    headers = {who: {"Authorization": f"Bearer {bearer}"} for who, bearer in bearers.items()}
    headers["owner"] = {web_server._SESSION_HEADER_NAME: web_server._SESSION_TOKEN}
    headers["none"] = {}

    providers = [WorkspaceReadTokenProvider(), ConnectionsManageTokenProvider()]
    added = [provider for provider in providers if snapshot_registration(provider.name) is None]
    for provider in added:
        register_provider(provider)

    state = web_server.app.state
    saved_attrs = {attr: getattr(state, attr, None) for attr in ("auth_required", "bound_host")}
    state.auth_required = False
    state.bound_host = None
    client = TestClient(web_server.app)

    def call(who, method, path, **kwargs):
        return client.request(method, path, headers=headers[who], **kwargs)

    try:
        yield _Quiet(call=call, connections=connections, root=home)
    finally:
        client.close()
        for attr, previous in saved_attrs.items():
            if previous is None:
                if hasattr(state, attr):
                    delattr(state, attr)
            else:
                setattr(state, attr, previous)
        for provider in added:
            restore_registration(provider.name, provider, None)


def _post_test(dash, request_id=None):
    return dash.call("connections", "POST", TEST, json={"request_id": request_id or str(uuid.uuid4())})


def _put_paused(dash, paused):
    response = dash.call("connections", "PUT", PAUSED, json={"paused": paused})
    assert response.status_code == 200, response.text
    return response.json()


def _audit_rows():
    from hermes_cli.dashboard_auth import audit as audit_mod

    path = audit_mod._resolve_log_path()
    return path.read_text(encoding="utf-8").splitlines() if path.exists() else []


def _call_and_audit(dash, method, path, **kwargs):
    """One call with the connections credential, and the audit rows it wrote."""
    before = len(_audit_rows())
    response = dash.call("connections", method, path, **kwargs)
    return response, [json.loads(row) for row in _audit_rows()[before:]]


#: One POST of the test route served in a fresh Python process, as a second or restarted
#: dashboard serves it: the same root home and credential, the repository on sys.path, and a
#: fake send function that appends a line to a file, then answers sent or dies as a crash does.
_ONE_POST_IN_A_FRESH_PROCESS = r'''
import json
import os
import sys

given = json.loads(sys.stdin.read())
sys.path.insert(0, given["repo"])

import tools.send_message_tool as send_tool


def fake_send(args, **kw):
    with open(given["sends"], "a", encoding="utf-8") as sends:
        sends.write("sent\n")
    if given["crash"]:
        os._exit(70)
    return json.dumps({"success": True, "message_id": "4242"})


send_tool.send_message_tool = fake_send

from fastapi.testclient import TestClient

from hermes_cli import web_server
from hermes_cli.dashboard_auth import register_provider
from hermes_cli.dashboard_auth.registry import snapshot_registration
from hermes_cli.web_routers import owner_channel
from plugins.dashboard_auth.raphael_workspace import ConnectionsManageTokenProvider

provider = ConnectionsManageTokenProvider()
if snapshot_registration(provider.name) is None:
    register_provider(provider)
if given["now"] is not None:
    owner_channel._now = lambda: given["now"]
web_server.app.state.auth_required = False
web_server.app.state.bound_host = None
response = TestClient(web_server.app).post(
    "/api/owner-channel/test",
    json={"request_id": given["request_id"]},
    headers={"Authorization": "Bearer " + given["bearer"]},
)
print("ANSWER " + json.dumps({"status": response.status_code, "body": response.json()}))
'''


def _post_test_in_a_fresh_process(dash, tmp_path, request_id, *, now=None, crash=False):
    """Exit code, answer (``{"status", "body"}``, None when it died) and stderr of one POST in a
    fresh process; ``now`` fixes its clock. The bearer goes in on stdin, never on a command line."""
    given = {
        "repo": str(Path(__file__).resolve().parents[2]),
        "bearer": dash.connections,
        "request_id": request_id,
        "sends": str(tmp_path / "fresh-process-sends.txt"),
        "now": now,
        "crash": crash,
    }
    done = subprocess.run(
        [sys.executable, "-c", _ONE_POST_IN_A_FRESH_PROCESS],
        input=json.dumps(given),
        capture_output=True,
        text=True,
        encoding="utf-8",
        cwd=tmp_path,
        env=dict(os.environ),
        timeout=180,
    )
    answers = [line[len("ANSWER "):] for line in done.stdout.splitlines() if line.startswith("ANSWER ")]
    return SimpleNamespace(
        code=done.returncode, answer=json.loads(answers[-1]) if answers else None, stderr=done.stderr[-3000:]
    )


def _sends_by_fresh_processes(tmp_path):
    sends = tmp_path / "fresh-process-sends.txt"
    return len(sends.read_text(encoding="utf-8").splitlines()) if sends.exists() else 0


# ---------------------------------------------------------------------------
# 1. GET: three keys, never a token, chat id or user name
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("where", [".env", "config.yaml"], ids=["root-env", "root-config-yaml"])
def test_get_answers_exactly_three_keys_and_no_digit_of_the_home_chat(dash, monkeypatch, where):
    _router(monkeypatch)
    from gateway import owner_channel_pause

    response = dash.call("connections", "GET", CHANNEL)
    assert response.status_code == 200, response.text
    assert response.json() == {"state": "not_set", "paused": False, "paused_since": None}

    _name_home(dash.root, where)
    response = dash.call("connections", "GET", CHANNEL)
    assert response.status_code == 200, response.text
    assert response.json() == {"state": "ready", "paused": False, "paused_since": None}
    assert not any(ch.isdigit() for ch in response.text)

    monkeypatch.setattr(owner_channel_pause, "time", SimpleNamespace(time=lambda: float(PAUSE_START)))
    owner_channel_pause.pause()
    response = dash.call("connections", "GET", CHANNEL)
    assert response.status_code == 200, response.text
    assert response.json() == {"state": "ready", "paused": True, "paused_since": PAUSE_START}
    assert set(response.json()) == {"state", "paused", "paused_since"}
    leaked = {ch for ch in HOME_CHAT if ch.isdigit()} & set(response.text)
    assert not leaked, "the body holds a digit of the home chat id"
    for value in (BOT_TOKEN, HOME_NAME, dash.connections):
        assert value not in response.text


# ---------------------------------------------------------------------------
# 2. POST test: the fixed text, once, through the send function
# ---------------------------------------------------------------------------


def test_the_test_route_sends_the_fixed_text_once_through_the_send_function(dash, monkeypatch, sends):
    _router(monkeypatch)
    _name_home(dash.root, ".env")

    response = _post_test(dash)
    assert response.status_code == 200, response.text
    assert response.json() == {"outcome": "sent", "reason": None}
    assert len(sends.calls) == 1
    (call,) = sends.calls
    assert (call.get("action", "send"), call["target"], call["message"]) == (
        "send",
        f"telegram:{HOME_CHAT}",
        TEST_TEXT,
    )
    assert HOME_CHAT.lstrip("-") not in response.text
    assert "chat_id" not in response.text


def test_an_empty_body_sends_once(dash, monkeypatch, sends):
    _router(monkeypatch)
    _name_home(dash.root, ".env")

    response = dash.call("connections", "POST", TEST)
    assert (response.status_code, response.json()) == (200, {"outcome": "sent", "reason": None})
    assert [(call["target"], call["message"]) for call in sends.calls] == [(f"telegram:{HOME_CHAT}", TEST_TEXT)]


def test_a_second_request_id_within_30_seconds_gets_429_and_no_send(dash, monkeypatch, sends):
    routes = _router(monkeypatch)
    _name_home(dash.root, ".env")
    clock = _clock(monkeypatch, routes)

    assert _post_test(dash).json() == {"outcome": "sent", "reason": None}
    response = _post_test(dash)
    assert response.status_code == 429, response.text
    assert len(sends.calls) == 1

    clock.advance(30)
    assert _post_test(dash).json() == {"outcome": "sent", "reason": None}
    assert len(sends.calls) == 2


def test_two_calls_inside_30_seconds_send_once_and_the_second_answers_429(dash, monkeypatch, sends):
    routes = _router(monkeypatch)
    _name_home(dash.root, ".env")
    clock = _clock(monkeypatch, routes)
    request_id = str(uuid.uuid4())

    assert _post_test(dash, request_id).json() == {"outcome": "sent", "reason": None}
    clock.advance(29)
    again = _post_test(dash, request_id)  # a retry after a lost answer; the request_id is ignored
    assert (again.status_code, again.json()) == (429, {"error": "too_soon"})
    assert len(sends.calls) == 1

    clock.advance(1)
    assert _post_test(dash, request_id).json() == {"outcome": "sent", "reason": None}
    assert len(sends.calls) == 2


def test_a_clock_moved_back_by_31_seconds_still_answers_429(dash, monkeypatch, sends):
    routes = _router(monkeypatch)
    _name_home(dash.root, ".env")
    clock = _clock(monkeypatch, routes)

    assert _post_test(dash).json() == {"outcome": "sent", "reason": None}
    clock.advance(-31)
    response = _post_test(dash)
    assert (response.status_code, response.json()) == (429, {"error": "too_soon"})
    assert len(sends.calls) == 1


def test_the_interval_counts_from_when_a_slow_send_returns(dash, monkeypatch, sends):
    routes = _router(monkeypatch)
    _name_home(dash.root, ".env")
    clock = _clock(monkeypatch, routes)

    def slow_send():
        clock.advance(31)
        return _sent_answer()

    sends.answer["do"] = slow_send
    assert _post_test(dash).json() == {"outcome": "sent", "reason": None}
    response = _post_test(dash)
    assert response.status_code == 429, response.text
    assert len(sends.calls) == 1


def _skipped():
    """What the real tool answers when it sends nothing because a cron job delivers there itself."""
    return json.dumps({
        "success": True,
        "skipped": True,
        "reason": "cron_auto_delivery_duplicate_target",
        "target": f"telegram:{HOME_CHAT}",
        "note": "SyntheticProviderText",
    })


def test_a_paused_channel_gets_409_paused_and_no_send(dash, monkeypatch, sends):
    _router(monkeypatch)
    _name_home(dash.root, ".env")
    _put_paused(dash, True)

    response = _post_test(dash)
    assert response.status_code == 409, response.text
    assert response.json() == {"error": "paused"}
    assert sends.calls == []


def test_no_home_channel_sends_nothing_and_answers_not_sent(dash, monkeypatch, sends):
    _router(monkeypatch)
    response = _post_test(dash)
    assert response.status_code == 200, response.text
    body = response.json()
    assert (body["outcome"], sorted(body)) == ("not_sent", ["outcome", "reason"])
    assert isinstance(body["reason"], str) and body["reason"]
    assert sends.calls == []


def _refused():
    return json.dumps({"error": f"Telegram said: chat {HOME_CHAT} not found (SyntheticProviderText)"})


def _raises():
    raise RuntimeError(f"SyntheticProviderText {HOME_CHAT}")


@pytest.mark.parametrize(
    "answer, outcome",
    [(_refused, "unknown"), (_raises, "unknown"), (lambda: "not json", "unknown")],
    ids=["refused", "raised", "unreadable"],
)
def test_a_failed_send_answers_a_fixed_reason_never_the_provider_text(dash, monkeypatch, sends, answer, outcome):
    _router(monkeypatch)
    _name_home(dash.root, ".env")
    sends.answer["do"] = answer

    response = _post_test(dash)
    assert response.status_code == 200, response.text
    body = response.json()
    assert (body["outcome"], sorted(body)) == (outcome, ["outcome", "reason"])
    assert isinstance(body["reason"], str) and body["reason"]
    assert "SyntheticProviderText" not in response.text
    assert HOME_CHAT.lstrip("-") not in response.text
    assert len(sends.calls) == 1


@pytest.mark.parametrize(
    "home_set, answer, outcome",
    [(False, _sent_answer, "not_sent"), (True, _skipped, "not_sent"), (True, _raises, "unknown")],
    ids=["no-home", "skipped", "raised"],
)
def test_every_outcome_keeps_the_cooldown_not_sent_and_unknown_included(
    dash, monkeypatch, sends, home_set, answer, outcome
):
    routes = _router(monkeypatch)
    clock = _clock(monkeypatch, routes)
    if home_set:
        _name_home(dash.root, ".env")
    sends.answer["do"] = answer
    response = _post_test(dash)
    assert (response.status_code, response.json()["outcome"]) == (200, outcome)
    assert "SyntheticProviderText" not in response.text

    _name_home(dash.root, ".env")
    sends.answer["do"] = _sent_answer
    clock.advance(29)
    response = _post_test(dash)
    assert (response.status_code, response.json()) == (429, {"error": "too_soon"})
    clock.advance(1)
    assert _post_test(dash).json() == {"outcome": "sent", "reason": None}
    assert len(sends.calls) == (2 if home_set else 1)


# ---------------------------------------------------------------------------
# 2b. POST test across processes: restarts, a second process, a crash
# ---------------------------------------------------------------------------


def test_two_calls_from_two_processes_inside_30_seconds_send_once_and_the_second_answers_429(dash, tmp_path):
    _name_home(dash.root, ".env")
    request_id = str(uuid.uuid4())  # the second call is a retry after a lost answer
    now = time.time()

    first = _post_test_in_a_fresh_process(dash, tmp_path, request_id, now=now)
    second = _post_test_in_a_fresh_process(dash, tmp_path, request_id, now=now + 29)
    assert (first.code, first.answer) == (0, {"status": 200, "body": {"outcome": "sent", "reason": None}}), first.stderr
    assert (second.code, second.answer) == (0, {"status": 429, "body": {"error": "too_soon"}}), second.stderr
    assert _sends_by_fresh_processes(tmp_path) == 1


def test_a_fresh_process_within_30_seconds_of_a_send_gets_429_and_sends_nothing(dash, monkeypatch, sends, tmp_path):
    _router(monkeypatch)
    _name_home(dash.root, ".env")
    before_the_send = time.time()
    assert _post_test(dash).json() == {"outcome": "sent", "reason": None}

    later = _post_test_in_a_fresh_process(dash, tmp_path, str(uuid.uuid4()), now=before_the_send + 29)
    assert (later.code, later.answer) == (0, {"status": 429, "body": {"error": "too_soon"}}), later.stderr
    assert _sends_by_fresh_processes(tmp_path) == 0
    assert len(sends.calls) == 1


def test_a_send_cut_off_by_a_crash_keeps_the_cooldown(dash, monkeypatch, sends, tmp_path):
    _router(monkeypatch)
    _name_home(dash.root, ".env")
    request_id = str(uuid.uuid4())

    crashed = _post_test_in_a_fresh_process(dash, tmp_path, request_id, crash=True)
    assert (crashed.code, crashed.answer, _sends_by_fresh_processes(tmp_path)) == (70, None, 1), crashed.stderr

    response = _post_test(dash, request_id)
    assert (response.status_code, response.json()) == (429, {"error": "too_soon"})
    assert sends.calls == []
    assert _sends_by_fresh_processes(tmp_path) == 1


# ---------------------------------------------------------------------------
# 2c. POST test through the real send function, only the Telegram transport faked
# ---------------------------------------------------------------------------


def test_the_test_message_goes_to_the_root_home_with_the_default_process_token(dash, monkeypatch, telegram):
    _router(monkeypatch)
    from hermes_constants import get_default_hermes_root, get_hermes_home

    assert get_hermes_home() == get_default_hermes_root() == dash.root
    other_profile = dash.root / "profiles" / "worker"
    other_profile.mkdir(parents=True)
    (other_profile / ".env").write_text(
        f"TELEGRAM_BOT_TOKEN={OTHER_TOKEN}\nTELEGRAM_HOME_CHANNEL={OTHER_CHAT}\n", encoding="utf-8"
    )
    _name_home(dash.root, ".env")

    assert dash.call("connections", "GET", CHANNEL).json()["state"] == "ready"
    response = _post_test(dash)
    assert (response.status_code, response.json()) == (200, {"outcome": "sent", "reason": None})
    assert len(telegram.deliveries) == 1
    token, chat_id, text = telegram.deliveries[0]
    assert (token == BOT_TOKEN, chat_id == HOME_CHAT, text == TEST_TEXT) == (True, True, True)


def test_a_reply_lost_after_delivery_answers_unknown_and_uses_up_the_interval(dash, monkeypatch, telegram):
    routes = _router(monkeypatch)
    _name_home(dash.root, ".env")

    def lose_the_reply():
        raise TimeoutError(f"SyntheticProviderText: no reply for chat {HOME_CHAT}")

    telegram.after["do"] = lose_the_reply
    response = _post_test(dash)
    assert (response.status_code, response.json()) == (200, {"outcome": "unknown", "reason": routes._UNKNOWN})
    assert len(telegram.deliveries) == 1
    state = (dash.root / "owner_channel_test.json").read_text(encoding="utf-8")
    for text in (response.text, state):
        assert "SyntheticProviderText" not in text
        assert HOME_CHAT.lstrip("-") not in text and BOT_TOKEN not in text

    response = _post_test(dash)
    assert response.status_code == 429, response.text
    assert len(telegram.deliveries) == 1


# ---------------------------------------------------------------------------
# 3. PUT paused: pause and resume, any other body changes nothing
# ---------------------------------------------------------------------------


def test_put_pauses_and_resumes_and_get_shows_it(dash, monkeypatch):
    _router(monkeypatch)
    from gateway import owner_channel_pause

    _name_home(dash.root, ".env")

    paused = _put_paused(dash, True)
    assert owner_channel_pause.is_paused()
    assert (set(paused), paused["state"], paused["paused"]) == ({"state", "paused", "paused_since"}, "ready", True)
    assert isinstance(paused["paused_since"], int)
    assert dash.call("connections", "GET", CHANNEL).json() == paused

    resumed = _put_paused(dash, False)
    assert not owner_channel_pause.is_paused()
    assert resumed == {"state": "ready", "paused": False, "paused_since": None}
    assert dash.call("connections", "GET", CHANNEL).json() == resumed


@pytest.mark.parametrize(
    "body",
    [{}, {"paused": "true"}, {"paused": 1}, {"paused": None}, {"paused": True, "extra": 1}, [True], True, "paused"],
    ids=["empty", "string", "number", "null", "extra-key", "list", "bare-true", "bare-string"],
)
def test_put_any_other_body_is_a_client_error_and_changes_nothing(dash, monkeypatch, body):
    _router(monkeypatch)
    pause_file = dash.root / "owner_channel_pause.json"

    response = dash.call("connections", "PUT", PAUSED, json=body)
    assert 400 <= response.status_code < 500, response.text
    assert not pause_file.exists()

    _put_paused(dash, True)
    before = pause_file.read_bytes()
    response = dash.call("connections", "PUT", PAUSED, json=body)
    assert 400 <= response.status_code < 500, response.text
    assert pause_file.read_bytes() == before


def test_put_a_body_that_is_not_json_is_a_client_error(dash, monkeypatch):
    _router(monkeypatch)
    response = dash.call("connections", "PUT", PAUSED, content=b"paused=true")
    assert 400 <= response.status_code < 500, response.text
    assert not (dash.root / "owner_channel_pause.json").exists()


# ---------------------------------------------------------------------------
# 4. Only the connections credential, only in the default profile's process
# ---------------------------------------------------------------------------


def test_a_request_without_the_connections_credential_is_refused_on_all_three_routes(dash, monkeypatch, sends):
    _router(monkeypatch)
    from gateway import owner_channel_pause
    from hermes_cli.dashboard_auth.token_auth import get_token_route_policy
    from plugins.dashboard_auth.raphael_workspace import CONNECTIONS_SCOPE

    _name_home(dash.root, ".env")
    calls = [
        ("GET", CHANNEL, {}),
        ("PUT", PAUSED, {"json": {"paused": True}}),
        ("POST", TEST, {"json": {"request_id": str(uuid.uuid4())}}),
    ]
    for method, path, kwargs in calls:
        response = dash.call("workspace", method, path, **kwargs)
        assert response.status_code == 403, (method, path, response.status_code)
        for who in ("none", "owner", "session_as_bearer", "connections_wrong_secret"):
            response = dash.call(who, method, path, **kwargs)
            assert response.status_code == 401, (who, method, path, response.status_code)
        policy = get_token_route_policy(method, path)
        assert (policy.required_scope, policy.optional, policy.strict_audit) == (CONNECTIONS_SCOPE, False, True)

    assert not owner_channel_pause.is_paused()
    assert sends.calls == []
    assert dash.call("connections", "GET", CHANNEL).status_code == 200


@pytest.mark.parametrize("home", ["profile"], indirect=True)
def test_a_process_with_a_non_default_profile_gets_503_on_all_three_routes(dash, monkeypatch, sends):
    _router(monkeypatch)
    from gateway import owner_channel_pause
    from hermes_cli.profiles import get_active_profile_name

    assert get_active_profile_name() == "worker"
    _name_home(dash.root, ".env")
    calls = [
        ("GET", CHANNEL, {}),
        ("PUT", PAUSED, {"json": {"paused": True}}),
        ("POST", TEST, {}),
        ("POST", TEST, {"json": {"request_id": str(uuid.uuid4())}}),
    ]
    for method, path, kwargs in calls:
        response, rows = _call_and_audit(dash, method, path, **kwargs)
        assert (response.status_code, response.json()) == (503, {"error": "unavailable"}), (method, path)
        assert _row_kinds(rows) == [("token_auth_failure", "deny", 503)], (method, path)
    assert sends.calls == []
    for name in ("owner_channel_pause.json", "owner_channel_test.json", "owner_channel_test.lock"):
        assert not (dash.root / name).exists(), name

    owner_channel_pause.pause()  # paused by the default profile's process
    before = (dash.root / "owner_channel_pause.json").read_bytes()
    response = dash.call("connections", "PUT", PAUSED, json={"paused": False})
    assert (response.status_code, response.json()) == (503, {"error": "unavailable"})
    assert owner_channel_pause.is_paused()
    assert (dash.root / "owner_channel_pause.json").read_bytes() == before


# ---------------------------------------------------------------------------
# 5. Audit
# ---------------------------------------------------------------------------


def test_every_call_writes_exactly_one_audit_row(dash, monkeypatch, sends):
    _router(monkeypatch)
    from plugins.dashboard_auth.raphael_workspace import CONNECTIONS_GRANT, CONNECTIONS_PRINCIPAL

    _name_home(dash.root, ".env")
    calls = [
        ("connections", "GET", CHANNEL, {}, 200),
        ("connections", "PUT", PAUSED, {"json": {"paused": True}}, 200),
        ("connections", "POST", TEST, {}, 409),
        ("connections", "PUT", PAUSED, {"json": {"paused": "no"}}, 400),
        ("connections", "PUT", PAUSED, {"json": {"paused": False}}, 200),
        ("connections", "POST", TEST, {}, 200),
        ("connections", "POST", TEST, {"json": {"request_id": str(uuid.uuid4())}}, 429),
        ("workspace", "GET", CHANNEL, {}, 403),
        ("none", "PUT", PAUSED, {"json": {"paused": True}}, 401),
        ("owner", "POST", TEST, {"json": {"request_id": str(uuid.uuid4())}}, 401),
    ]
    for who, method, path, kwargs, status in calls:
        before = len(_audit_rows())
        response = dash.call(who, method, path, **kwargs)
        assert response.status_code == status, (who, method, path, response.status_code)
        rows = _audit_rows()[before:]
        assert len(rows) == 1, (who, method, path, len(rows))
        row = json.loads(rows[0])
        assert (row["method"], row["route_template"], row["status"]) == (method, path, status)
        if status == 200:
            assert (row["event"], row["decision"]) == ("token_auth_success", "allow")
            assert (row["principal"], row["grant"], row["source"]) == (
                CONNECTIONS_PRINCIPAL,
                CONNECTIONS_GRANT,
                "raphael-connections",
            )
        else:
            assert (row["event"], row["decision"]) == ("token_auth_failure", "deny")

    assert len(sends.calls) == 1
    text = "\n".join(_audit_rows())
    assert dash.connections not in text and dash.connections.split(".", 1)[1] not in text
    assert HOME_CHAT.lstrip("-") not in text


def _no_space(*args, **kwargs):
    raise OSError(errno.ENOSPC, "No space left on device (SyntheticFailureText)")


def _fails(*args, **kwargs):
    raise RuntimeError(f"SyntheticFailureText {HOME_CHAT}")


def _row_kinds(rows):
    return [(row["event"], row["decision"], row["status"]) for row in rows]


@pytest.mark.parametrize("paused", [True, False], ids=["pause", "resume"])
def test_a_storage_failure_on_pause_or_resume_answers_500_and_writes_one_audit_row(dash, monkeypatch, paused):
    _router(monkeypatch)
    from gateway import owner_channel_pause

    pause_file = dash.root / "owner_channel_pause.json"
    if not paused:
        _put_paused(dash, True)
    before = pause_file.read_bytes() if pause_file.exists() else None
    monkeypatch.setattr(owner_channel_pause, "atomic_json_write", _no_space)

    response, rows = _call_and_audit(dash, "PUT", PAUSED, json={"paused": paused})
    assert (response.status_code, response.json()) == (500, {"error": "failed"})
    assert _row_kinds(rows) == [("token_auth_failure", "deny", 500)]
    assert owner_channel_pause.is_paused() == (not paused)
    assert (pause_file.read_bytes() if pause_file.exists() else None) == before


def test_a_failure_inside_get_answers_500_and_writes_one_audit_row(dash, monkeypatch):
    _router(monkeypatch)
    from gateway import owner_channel_pause

    monkeypatch.setattr(owner_channel_pause, "paused_since", _fails)
    response, rows = _call_and_audit(dash, "GET", CHANNEL)
    assert (response.status_code, response.json()) == (500, {"error": "failed"})
    assert _row_kinds(rows) == [("token_auth_failure", "deny", 500)]


@pytest.mark.parametrize("state", ["unwritable", "not-json", "wrong-shape"])
def test_a_test_state_that_cannot_be_kept_answers_500_sends_nothing_and_writes_one_audit_row(
    dash, monkeypatch, sends, state
):
    routes = _router(monkeypatch)
    _name_home(dash.root, ".env")
    state_file = dash.root / "owner_channel_test.json"
    if state == "unwritable":
        monkeypatch.setattr(routes, "atomic_json_write", _no_space)
    elif state == "not-json":
        state_file.write_text("{not json", encoding="utf-8")
    else:
        state_file.write_text(json.dumps({"last_test_at": "soon"}), encoding="utf-8")

    response, rows = _call_and_audit(dash, "POST", TEST)
    assert (response.status_code, response.json()) == (500, {"error": "failed"})
    assert _row_kinds(rows) == [("token_auth_failure", "deny", 500)]
    assert sends.calls == []
