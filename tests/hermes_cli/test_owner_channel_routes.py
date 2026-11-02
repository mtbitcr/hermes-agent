"""Owner channel routes: the Telegram row of the owner app.

``GET /api/owner-channel`` reads whether the owner's Telegram home channel is
set and whether owner messages are paused, ``PUT /api/owner-channel/paused``
pauses or resumes them and ``POST /api/owner-channel/test`` sends one fixed
test message. Only the owner app's connections credential is admitted, and
every call writes one audit row. Every token and chat id is synthetic, and
every send goes to a fake send function, never to Telegram.
"""

import importlib
import importlib.util
import json
import uuid
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


class _Quiet(SimpleNamespace):
    """A namespace whose repr names no value, so failure output never prints a token."""

    def __repr__(self):
        return "<quiet>"


def _router(monkeypatch):
    """The routes' module, asserted inside a test so its absence fails that test; no test
    message has been answered or sent yet in this process."""
    assert importlib.util.find_spec(ROUTER) is not None, f"{ROUTER} does not exist"
    module = importlib.import_module(ROUTER)
    module._register_owner_channel_routes()
    monkeypatch.setattr(module, "_test_answers", {})
    monkeypatch.setattr(module, "_last_test_sent", None)
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
def home(tmp_path, monkeypatch):
    """The root Hermes home the dashboard runs with; it names no Telegram home yet."""
    root = tmp_path / "hermes"
    root.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(root))
    monkeypatch.delenv("TELEGRAM_HOME_CHANNEL", raising=False)
    return root


@pytest.fixture()
def sends(monkeypatch):
    """Every call of the fake send function; ``answer`` sets what it returns or raises."""
    calls = []
    answer = {"do": _sent_answer}

    def fake(args, **kw):
        calls.append(dict(args))
        return answer["do"]()

    monkeypatch.setattr("tools.send_message_tool.send_message_tool", fake)
    return _Quiet(calls=calls, answer=answer)


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
    assert (call.get("action", "send"), call["target"], call["message"]) == ("send", "telegram", TEST_TEXT)
    assert HOME_CHAT.lstrip("-") not in response.text
    assert "chat_id" not in response.text


def test_the_same_request_id_twice_sends_once_and_answers_the_same(dash, monkeypatch, sends):
    _router(monkeypatch)
    _name_home(dash.root, ".env")
    request_id = str(uuid.uuid4())

    first = _post_test(dash, request_id)
    again = _post_test(dash, request_id)
    assert (first.status_code, again.status_code) == (200, 200)
    assert again.json() == first.json() == {"outcome": "sent", "reason": None}
    assert len(sends.calls) == 1


def test_a_second_request_id_within_30_seconds_gets_429_and_no_send(dash, monkeypatch, sends):
    routes = _router(monkeypatch)
    _name_home(dash.root, ".env")

    assert _post_test(dash).json() == {"outcome": "sent", "reason": None}
    response = _post_test(dash)
    assert response.status_code == 429, response.text
    assert len(sends.calls) == 1

    routes._last_test_sent -= 30
    assert _post_test(dash).json() == {"outcome": "sent", "reason": None}
    assert len(sends.calls) == 2


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
    [(_refused, "not_sent"), (_raises, "unknown"), (lambda: "not json", "unknown")],
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


@pytest.mark.parametrize(
    "body",
    [{}, {"request_id": "not-a-uuid"}, {"request_id": 7}, {"request_id": str(uuid.uuid4()), "extra": 1}, []],
    ids=["empty", "not-a-uuid", "number", "extra-key", "list"],
)
def test_post_test_any_other_body_is_a_client_error_and_sends_nothing(dash, monkeypatch, sends, body):
    _router(monkeypatch)
    _name_home(dash.root, ".env")

    response = dash.call("connections", "POST", TEST, json=body)
    assert 400 <= response.status_code < 500, response.text
    assert sends.calls == []


# ---------------------------------------------------------------------------
# 4. Only the connections credential
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


# ---------------------------------------------------------------------------
# 5. Audit
# ---------------------------------------------------------------------------


def test_every_call_writes_exactly_one_audit_row(dash, monkeypatch, sends):
    _router(monkeypatch)
    from plugins.dashboard_auth.raphael_workspace import CONNECTIONS_GRANT, CONNECTIONS_PRINCIPAL

    _name_home(dash.root, ".env")
    request_id = str(uuid.uuid4())
    calls = [
        ("connections", "GET", CHANNEL, {}, 200),
        ("connections", "PUT", PAUSED, {"json": {"paused": True}}, 200),
        ("connections", "POST", TEST, {"json": {"request_id": request_id}}, 409),
        ("connections", "PUT", PAUSED, {"json": {"paused": "no"}}, 400),
        ("connections", "PUT", PAUSED, {"json": {"paused": False}}, 200),
        ("connections", "POST", TEST, {"json": {"request_id": request_id}}, 200),
        ("connections", "POST", TEST, {"json": {"request_id": request_id}}, 200),
        ("connections", "POST", TEST, {"json": {"request_id": str(uuid.uuid4())}}, 429),
        ("connections", "POST", TEST, {"json": {"request_id": "not-a-uuid"}}, 400),
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
