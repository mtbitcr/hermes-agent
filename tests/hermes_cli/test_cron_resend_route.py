"""Owner re-send route: POST /api/cron/executions/{execution_id}/resend on the dashboard.

The dashboard admits the owner the way pause/resume do, forwards the request id
over loopback to the target profile's own gateway with that profile's own
API_SERVER_KEY, and hands the gateway's answer back unchanged.  Every gateway
here is a local stub; every key and id is a placeholder.
"""

import asyncio
import json
import logging
import os
import socket
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace
from urllib.parse import unquote

import pytest
import yaml

DEFAULT_KEY = "placeholder-key-default-0123456789abcdef"
WORKER_KEY = "placeholder-key-worker-0123456789abcdef"
BETA_KEY = "placeholder-key-beta-0123456789abcdef"
ALL_KEYS = (DEFAULT_KEY, WORKER_KEY, BETA_KEY)
EXECUTION_ID = "exec-placeholder-1"
REQUEST_ID = "req-placeholder-1"
ROUTE_TEMPLATE = "/api/cron/executions/{execution_id}/resend"
JSON_TYPE = "application/json; charset=utf-8"
RESEND_PATH = f"/api/cron/executions/{EXECUTION_ID}/resend"

NOT_ELIGIBLE_REASONS = (
    "not_recorded",
    "in_progress",
    "outcome_unknown",
    "already_delivered",
    "too_many_attempts",
    "attachment_missing",
    "output_expired",
)
PER_CHAT_RESULT = {
    "execution_id": EXECUTION_ID,
    "attempt_id": "attempt-placeholder-1",
    "state": "partial",
    "targets": [
        {"label": "Telegram (Ops room)", "state": "delivered", "reason": None},
        {"label": "Slack (Team room)", "state": "failed", "reason": "send_failed"},
    ],
    "resend": {"eligible": True, "reason": None},
}
IN_PROGRESS = {"state": "in_progress"}
GATEWAY_UNAVAILABLE = {"detail": {"code": "gateway_unavailable"}}
GATEWAY_AUTH_FAILED = {
    "error": {
        "message": "Invalid gateway API key (API_SERVER_KEY)",
        "type": "gateway_auth_error",
        "code": "gateway_auth_failed",
    }
}
ROUTE_NOT_ALLOWED = {
    "error": {
        "message": "Route not permitted for this profile",
        "type": "invalid_request_error",
        "param": None,
        "code": "route_not_allowed",
    }
}


def _aiohttp_bytes(payload):
    """Serialize the way the gateway's json_response does (default separators)."""
    return json.dumps(payload).encode("utf-8")


class StubGateway:
    """A loopback HTTP server standing in for one profile's gateway."""

    def __init__(self, status=200, body=None, content_type=JSON_TYPE, drop=False):
        self.status = status
        self.body = _aiohttp_bytes(PER_CHAT_RESULT) if body is None else body
        self.content_type = content_type
        self.drop = drop
        self.requests = []
        stub = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def _record(self):
                length = int(self.headers.get("Content-Length") or 0)
                payload = self.rfile.read(length) if length else b""
                stub.requests.append(
                    {
                        "method": self.command,
                        "target": self.path,
                        "headers": {k.lower(): v for k, v in self.headers.items()},
                        "body": payload,
                        "client": self.client_address[0],
                    }
                )
                if stub.drop:
                    self.close_connection = True
                    try:
                        self.connection.shutdown(socket.SHUT_RDWR)
                    except OSError:
                        pass
                    return
                self.send_response(stub.status)
                if stub.content_type:
                    self.send_header("Content-Type", stub.content_type)
                self.send_header("X-Gateway-Stub", "placeholder-header")
                self.send_header("Content-Length", str(len(stub.body)))
                self.send_header("Connection", "close")
                self.end_headers()
                self.wfile.write(stub.body)
                self.close_connection = True

            do_POST = _record  # noqa: N815
            do_GET = _record  # noqa: N815
            do_PUT = _record  # noqa: N815

            def log_message(self, *args):  # keep test output quiet
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.server.daemon_threads = True
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def answer(self, status, payload=None, *, body=None, content_type=JSON_TYPE):
        self.status = status
        self.body = body if body is not None else _aiohttp_bytes(payload)
        self.content_type = content_type
        self.drop = False

    def close(self):
        self.server.shutdown()
        self.server.server_close()


def _free_port():
    """A loopback port with nothing listening on it."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


@pytest.fixture()
def profiles_home(tmp_path, monkeypatch):
    """A default home plus worker_alpha and worker_beta, all under tmp_path."""
    from hermes_cli import profiles

    default_home = tmp_path / ".hermes"
    profiles_root = default_home / "profiles"
    homes = {
        "default": default_home,
        "worker_alpha": profiles_root / "worker_alpha",
        "worker_beta": profiles_root / "worker_beta",
    }
    for home in homes.values():
        (home / "cron").mkdir(parents=True, exist_ok=True)
        (home / "config.yaml").write_text("model: test-model\n", encoding="utf-8")
    monkeypatch.setattr(profiles, "_get_default_hermes_home", lambda: default_home)
    monkeypatch.setattr(profiles, "_get_profiles_root", lambda: profiles_root)
    monkeypatch.setattr("hermes_constants.get_default_hermes_root", lambda: default_home)
    monkeypatch.setenv("HERMES_HOME", str(default_home))
    for name in ("API_SERVER_PORT", "API_SERVER_KEY", "GATEWAY_MULTIPLEX_PROFILES"):
        monkeypatch.delenv(name, raising=False)
    return homes


def _point_default(homes, monkeypatch, port):
    """Default profile: port in config.yaml, key in the dashboard's own environment."""
    config = {"model": "test-model", "platforms": {"api_server": {"extra": {"port": port}}}}
    (homes["default"] / "config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    monkeypatch.setenv("API_SERVER_KEY", DEFAULT_KEY)


def _point_worker_alpha(homes, port):
    """worker_alpha: port and key only in the profile's own .env."""
    (homes["worker_alpha"] / ".env").write_text(
        f"API_SERVER_PORT={port}\nAPI_SERVER_KEY={WORKER_KEY}\n", encoding="utf-8"
    )


def _point_worker_beta(homes, port):
    """worker_beta: port and key in the profile's own config.yaml."""
    config = {
        "model": "test-model",
        "platforms": {"api_server": {"extra": {"port": port, "key": BETA_KEY}}},
    }
    (homes["worker_beta"] / "config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")


@pytest.fixture()
def stubs():
    made = []

    def make(**kwargs):
        stub = StubGateway(**kwargs)
        made.append(stub)
        return stub

    yield make
    for stub in made:
        stub.close()


@pytest.fixture()
def dashboard(profiles_home, tmp_path):
    """The real dashboard app with an owner session and an automations machine token."""
    from fastapi.testclient import TestClient

    from hermes_cli import web_server
    from hermes_cli.dashboard_auth import register_provider
    from hermes_cli.dashboard_auth.registry import restore_registration, snapshot_registration
    from hermes_cli.web_routers import cron as cron_routes
    from plugins.dashboard_auth.raphael_workspace import AutomationsManageTokenProvider, token_store

    assert web_server._cron_default_profile() == "default"
    cron_routes._register_automations_machine_routes()
    token_dir = tmp_path / "machine-tokens"
    token_dir.mkdir(mode=0o700)
    token_path = token_dir / "automations.token"
    token_store.issue(out_path=token_path, surface=token_store.AUTOMATIONS_SURFACE)
    automations = "Bearer " + token_path.read_text(encoding="utf-8").strip()
    provider = AutomationsManageTokenProvider()
    previous = snapshot_registration(provider.name)
    if previous is None:
        register_provider(provider)

    prev_auth = getattr(web_server.app.state, "auth_required", None)
    prev_host = getattr(web_server.app.state, "bound_host", None)
    web_server.app.state.auth_required = False
    web_server.app.state.bound_host = None
    client = TestClient(web_server.app)
    headers = {
        "owner": {web_server._SESSION_HEADER_NAME: web_server._SESSION_TOKEN},
        "automations": {"Authorization": automations},
        "none": {},
        "wrong_session": {web_server._SESSION_HEADER_NAME: "placeholder-wrong-session-token"},
    }
    try:
        yield SimpleNamespace(
            client=client, headers=headers, homes=profiles_home, token_dir=token_dir, web_server=web_server
        )
    finally:
        client.close()
        for attr, prev in (("auth_required", prev_auth), ("bound_host", prev_host)):
            if prev is None:
                if hasattr(web_server.app.state, attr):
                    delattr(web_server.app.state, attr)
            else:
                setattr(web_server.app.state, attr, prev)
        if previous is None:
            restore_registration(provider.name, provider, None)


class _Collect(logging.Handler):
    def __init__(self):
        super().__init__(level=logging.DEBUG)
        self.records = []
        self._seen = []  # held so ids stay unique while collecting

    def emit(self, record):
        # One handler on every logger sees a propagating record more than once.
        if any(seen is record for seen in self._seen):
            return
        self._seen.append(record)
        try:
            text = logging.Formatter("%(name)s %(levelname)s %(message)s").format(record)
        except Exception:  # a broken record still must not hide what it carried
            text = repr(record.__dict__)
        self.records.append((record.name, record.levelno, text))


@pytest.fixture()
def every_log_line():
    """Capture every log record from every logger at DEBUG and above."""
    collector = _Collect()
    previous_disable = logging.root.manager.disable
    logging.disable(logging.NOTSET)
    loggers = [logging.getLogger()] + [
        lg for lg in logging.root.manager.loggerDict.values() if isinstance(lg, logging.Logger)
    ]
    saved = [(lg, lg.level) for lg in loggers]
    for lg in loggers:
        lg.setLevel(logging.DEBUG)
        lg.addHandler(collector)
    try:
        yield collector
    finally:
        for lg, level in saved:
            lg.removeHandler(collector)
            lg.setLevel(level)
        logging.disable(previous_disable)


def _post(dashboard, who, *, profile="worker_alpha", path=RESEND_PATH, **kwargs):
    if "json" not in kwargs and "content" not in kwargs:
        kwargs["json"] = {"request_id": REQUEST_ID}
    params = {} if profile is None else {"profile": profile}
    return dashboard.client.post(path, params=params, headers=dashboard.headers[who], **kwargs)


def _audit_lines(action="resend"):
    from hermes_cli.dashboard_auth import audit as audit_mod

    path = audit_mod._resolve_log_path()
    if not path.exists():
        return []
    lines = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        try:
            entry = json.loads(raw)
        except ValueError:
            continue
        if entry.get("action") == action:
            lines.append(entry)
    return lines


PASS_THROUGH = [
    pytest.param(200, _aiohttp_bytes(PER_CHAT_RESULT), JSON_TYPE, id="per-chat-result"),
    pytest.param(202, _aiohttp_bytes(IN_PROGRESS), JSON_TYPE, id="in-progress"),
    pytest.param(404, _aiohttp_bytes({"detail": {"code": "not_found"}}), JSON_TYPE, id="not-found"),
    pytest.param(400, _aiohttp_bytes({"detail": {"code": "invalid_request_id"}}), JSON_TYPE, id="bad-request-id"),
    pytest.param(401, _aiohttp_bytes(GATEWAY_AUTH_FAILED), JSON_TYPE, id="gateway-auth-failed"),
    pytest.param(403, _aiohttp_bytes(ROUTE_NOT_ALLOWED), JSON_TYPE, id="route-not-allowed"),
    pytest.param(503, _aiohttp_bytes(GATEWAY_UNAVAILABLE), JSON_TYPE, id="gateway-own-503"),
    pytest.param(502, b"Bad Gateway placeholder", "text/plain; charset=utf-8", id="plain-502"),
] + [
    pytest.param(
        409,
        _aiohttp_bytes({"detail": {"code": "not_eligible", "reason": reason}}),
        JSON_TYPE,
        id=f"not-eligible-{reason}",
    )
    for reason in NOT_ELIGIBLE_REASONS
]


@pytest.mark.parametrize("status,body,content_type", PASS_THROUGH)
def test_gateway_answer_passes_through_unchanged(dashboard, stubs, status, body, content_type):
    stub = stubs(status=status, body=body, content_type=content_type)
    _point_worker_alpha(dashboard.homes, stub.port)

    response = _post(dashboard, "owner")

    assert response.status_code == status
    assert response.content == body
    assert response.headers["content-type"] == content_type
    assert "x-gateway-stub" not in response.headers
    assert len(stub.requests) == 1


@pytest.mark.parametrize("who", ["owner", "automations"])
def test_allowlist_refusal_passes_through_for_every_admitted_caller(dashboard, stubs, who):
    stub = stubs(status=403, body=_aiohttp_bytes(ROUTE_NOT_ALLOWED))
    _point_worker_alpha(dashboard.homes, stub.port)

    response = _post(dashboard, who)

    assert response.status_code == 403
    assert response.json() == ROUTE_NOT_ALLOWED
    assert len(stub.requests) == 1


def test_gateway_down_answers_503_gateway_unavailable(dashboard):
    _point_worker_alpha(dashboard.homes, _free_port())

    response = _post(dashboard, "owner")

    assert response.status_code == 503
    assert response.json() == GATEWAY_UNAVAILABLE


def test_connected_but_no_answer_is_in_progress_never_503(dashboard, stubs):
    stub = stubs(drop=True)
    _point_worker_alpha(dashboard.homes, stub.port)

    response = _post(dashboard, "owner")

    assert response.status_code == 202
    assert response.json() == IN_PROGRESS
    assert len(stub.requests) == 1


def test_owner_and_automations_token_are_admitted_like_pause_resume(dashboard, stubs):
    stub = stubs()
    _point_worker_alpha(dashboard.homes, stub.port)

    for who in ("owner", "automations"):
        response = _post(dashboard, who)
        assert response.status_code == 200, who
        assert response.json() == PER_CHAT_RESULT
    assert len(stub.requests) == 2

    for who in ("none", "wrong_session"):
        response = _post(dashboard, who)
        assert response.status_code == 401, who
    assert len(stub.requests) == 2


def test_machine_token_without_automations_scope_is_refused(dashboard, stubs):
    from hermes_cli.dashboard_auth import register_provider
    from hermes_cli.dashboard_auth.registry import restore_registration, snapshot_registration
    from plugins.dashboard_auth.raphael_workspace import ModelsManageTokenProvider, token_store
    from plugins.dashboard_auth.raphael_workspace.model_policy import register_models_machine_routes

    stub = stubs()
    _point_worker_alpha(dashboard.homes, stub.port)
    response = _post(dashboard, "automations")
    assert response.status_code == 200
    assert len(stub.requests) == 1

    register_models_machine_routes()
    token_path = dashboard.token_dir / "models.token"
    token_store.issue(out_path=token_path, surface=token_store.MODELS_SURFACE)
    provider = ModelsManageTokenProvider()
    previous = snapshot_registration(provider.name)
    if previous is None:
        register_provider(provider)
    try:
        dashboard.headers["models"] = {"Authorization": "Bearer " + token_path.read_text(encoding="utf-8").strip()}
        response = _post(dashboard, "models")
        assert response.status_code == 403
        assert len(stub.requests) == 1
    finally:
        if previous is None:
            restore_registration(provider.name, provider, None)


def test_machine_success_is_audited_once_as_resend(dashboard, stubs):
    stub = stubs()
    _point_worker_alpha(dashboard.homes, stub.port)

    response = _post(dashboard, "automations")

    assert response.status_code == 200
    lines = _audit_lines()
    assert len(lines) == 1
    assert lines[0]["event"] == "token_auth_success"
    assert lines[0]["route_template"] == ROUTE_TEMPLATE
    text = json.dumps(lines[0])
    assert REQUEST_ID not in text
    assert not any(key in text for key in ALL_KEYS)


def test_audit_is_written_before_forwarding_even_when_gateway_is_down(dashboard):
    _point_worker_alpha(dashboard.homes, _free_port())

    response = _post(dashboard, "automations")

    assert response.status_code == 503
    assert response.json() == GATEWAY_UNAVAILABLE
    assert len(_audit_lines()) == 1


def test_audit_failure_answers_503_and_sends_nothing(dashboard, stubs, tmp_path, monkeypatch):
    from hermes_cli.dashboard_auth import audit as audit_mod

    stub = stubs()
    _point_worker_alpha(dashboard.homes, stub.port)
    assert _post(dashboard, "automations").status_code == 200
    assert len(stub.requests) == 1

    blocked = tmp_path / "audit-parent-is-a-file"
    blocked.write_text("not a directory", encoding="utf-8")
    monkeypatch.setattr(audit_mod, "_resolve_log_path", lambda: blocked / "dashboard-auth.log")

    response = _post(dashboard, "automations")

    assert response.status_code == 503
    assert len(stub.requests) == 1


def test_owner_session_is_not_machine_audited(dashboard, stubs):
    stub = stubs()
    _point_worker_alpha(dashboard.homes, stub.port)

    response = _post(dashboard, "owner")

    assert response.status_code == 200
    assert len(stub.requests) == 1
    assert _audit_lines() == []


@pytest.mark.parametrize(
    "profile,expected_key",
    [
        pytest.param(None, DEFAULT_KEY, id="profile-omitted"),
        pytest.param("default", DEFAULT_KEY, id="default-env-key"),
        pytest.param("worker_alpha", WORKER_KEY, id="profile-dotenv-key"),
        pytest.param("worker_beta", BETA_KEY, id="profile-config-key"),
    ],
)
def test_forward_goes_to_the_profiles_own_gateway_with_its_own_key(
    dashboard, stubs, monkeypatch, profile, expected_key
):
    gateways = {name: stubs() for name in ("default", "worker_alpha", "worker_beta")}
    _point_default(dashboard.homes, monkeypatch, gateways["default"].port)
    _point_worker_alpha(dashboard.homes, gateways["worker_alpha"].port)
    _point_worker_beta(dashboard.homes, gateways["worker_beta"].port)
    body = {"request_id": REQUEST_ID, "note": "placeholder-note", "profile": "worker_beta"}

    response = _post(dashboard, "owner", profile=profile, json=body)

    assert response.status_code == 200
    target = gateways[profile or "default"]
    assert len(target.requests) == 1
    sent = target.requests[0]
    assert sent["method"] == "POST"
    assert sent["target"] == RESEND_PATH
    assert sent["client"] == "127.0.0.1"
    assert sent["headers"]["host"] == f"127.0.0.1:{target.port}"
    assert sent["headers"]["authorization"] == f"Bearer {expected_key}"
    assert json.loads(sent["body"]) == {"request_id": REQUEST_ID}
    assert "cookie" not in sent["headers"]
    assert dashboard.web_server._SESSION_HEADER_NAME.lower() not in sent["headers"]
    others = [stub for stub in gateways.values() if stub is not target]
    assert all(stub.requests == [] for stub in others)


@pytest.mark.parametrize(
    "kwargs,forwarded",
    [
        pytest.param({"json": {"request_id": 7}}, {"request_id": 7}, id="not-validated-here"),
        pytest.param({"json": {"note": "placeholder-note"}}, {"request_id": None}, id="missing-request-id"),
        pytest.param({"json": ["placeholder"]}, {"request_id": None}, id="not-an-object"),
        pytest.param(
            {"content": b"not json", "headers_extra": {"Content-Type": "application/json"}},
            {"request_id": None},
            id="not-json",
        ),
    ],
)
def test_forwarded_body_is_only_the_request_id(dashboard, stubs, kwargs, forwarded):
    stub = stubs()
    _point_worker_alpha(dashboard.homes, stub.port)
    extra = kwargs.pop("headers_extra", None)
    if extra:
        dashboard.headers["owner"] = {**dashboard.headers["owner"], **extra}

    response = _post(dashboard, "owner", **kwargs)

    assert response.status_code == 200
    assert len(stub.requests) == 1
    assert json.loads(stub.requests[0]["body"]) == forwarded


def test_multiplexed_gateway_gets_the_profile_prefix_and_profile_key(dashboard, stubs, monkeypatch):
    stub = stubs()
    _point_worker_alpha(dashboard.homes, stub.port)
    monkeypatch.setenv("GATEWAY_MULTIPLEX_PROFILES", "1")

    response = _post(dashboard, "owner")

    assert response.status_code == 200
    assert len(stub.requests) == 1
    assert stub.requests[0]["target"] == f"/p/worker_alpha{RESEND_PATH}"
    assert stub.requests[0]["headers"]["authorization"] == f"Bearer {WORKER_KEY}"


def test_no_key_means_no_authorization_header_and_gateway_401_passes(dashboard, stubs):
    stub = stubs(status=401, body=_aiohttp_bytes(GATEWAY_AUTH_FAILED))
    (dashboard.homes["worker_alpha"] / ".env").write_text(f"API_SERVER_PORT={stub.port}\n", encoding="utf-8")

    response = _post(dashboard, "owner")

    assert response.status_code == 401
    assert response.json() == GATEWAY_AUTH_FAILED
    assert len(stub.requests) == 1
    assert "authorization" not in stub.requests[0]["headers"]


KEY_REF_NAME = "PLACEHOLDER_GATEWAY_KEY_REF"
KEY_REFS = [
    pytest.param("${API_SERVER_KEY}", "API_SERVER_KEY", id="bare-ref"),
    pytest.param("${env:" + KEY_REF_NAME + "}", KEY_REF_NAME, id="env-ref"),
]


def _point_every_gateway(dashboard, stubs, monkeypatch, **beta_stub):
    """Every profile on its own stub, as in the own-key test; the dashboard env holds DEFAULT_KEY."""
    gateways = {"default": stubs(), "worker_alpha": stubs(), "worker_beta": stubs(**beta_stub)}
    _point_default(dashboard.homes, monkeypatch, gateways["default"].port)
    _point_worker_alpha(dashboard.homes, gateways["worker_alpha"].port)
    _point_worker_beta(dashboard.homes, gateways["worker_beta"].port)
    return gateways


def _point_worker_beta_key_ref(homes, port, ref, dotenv):
    """worker_beta: port in config.yaml, key there as a ``${...}`` reference, and its own .env."""
    config = {
        "model": "test-model",
        "platforms": {"api_server": {"extra": {"port": port, "key": ref}}},
    }
    (homes["worker_beta"] / "config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    (homes["worker_beta"] / ".env").write_text(dotenv, encoding="utf-8")


@pytest.fixture()
def scopes_after_key(dashboard, monkeypatch):
    """The secret scope left in the forwarder's own context after each key resolution."""
    from agent.secret_scope import current_secret_scope

    resolve = dashboard.web_server._gateway_api_key
    seen = []

    def resolve_and_record(profile, home):
        key = resolve(profile, home)
        seen.append(current_secret_scope())
        return key

    monkeypatch.setattr(dashboard.web_server, "_gateway_api_key", resolve_and_record)
    return seen


def _post_leaving_no_trace(dashboard, scopes_after_key, profile):
    """Post as the owner; the request must change no environment and leave no secret scope."""
    from agent.secret_scope import current_secret_scope, is_multiplex_active

    environ = dict(os.environ)
    assert not is_multiplex_active()

    response = _post(dashboard, "owner", profile=profile)

    assert dict(os.environ) == environ
    assert not is_multiplex_active()
    assert current_secret_scope() is None
    assert scopes_after_key == [None]
    return response


@pytest.mark.parametrize("ref,name", KEY_REFS)
def test_config_key_reference_resolves_from_the_target_profiles_own_secrets(
    dashboard, stubs, monkeypatch, scopes_after_key, ref, name
):
    gateways = _point_every_gateway(dashboard, stubs, monkeypatch)
    monkeypatch.setenv(name, DEFAULT_KEY)
    _point_worker_beta_key_ref(dashboard.homes, gateways["worker_beta"].port, ref, f"{name}={BETA_KEY}\n")

    response = _post_leaving_no_trace(dashboard, scopes_after_key, "worker_beta")

    assert response.status_code == 200
    assert len(gateways["worker_beta"].requests) == 1
    assert gateways["worker_beta"].requests[0]["headers"]["authorization"] == f"Bearer {BETA_KEY}"
    assert gateways["default"].requests == [] and gateways["worker_alpha"].requests == []


@pytest.mark.parametrize("ref,name", KEY_REFS)
def test_missing_target_reference_sends_no_key_and_the_gateway_401_passes(
    dashboard, stubs, monkeypatch, scopes_after_key, ref, name
):
    gateways = _point_every_gateway(
        dashboard, stubs, monkeypatch, status=401, body=_aiohttp_bytes(GATEWAY_AUTH_FAILED)
    )
    monkeypatch.setenv(name, DEFAULT_KEY)
    _point_worker_beta_key_ref(
        dashboard.homes, gateways["worker_beta"].port, ref, "PLACEHOLDER_UNRELATED=placeholder-value\n"
    )

    response = _post_leaving_no_trace(dashboard, scopes_after_key, "worker_beta")

    assert response.status_code == 401
    assert response.content == _aiohttp_bytes(GATEWAY_AUTH_FAILED)
    assert response.headers["content-type"] == JSON_TYPE
    assert len(gateways["worker_beta"].requests) == 1
    sent = gateways["worker_beta"].requests[0]
    assert "authorization" not in sent["headers"]
    assert DEFAULT_KEY not in repr(sent) and "${" not in repr(sent)
    assert gateways["default"].requests == [] and gateways["worker_alpha"].requests == []


def test_missing_target_reference_falls_back_to_the_target_profiles_own_env_key(
    dashboard, stubs, monkeypatch, scopes_after_key
):
    gateways = _point_every_gateway(dashboard, stubs, monkeypatch)
    monkeypatch.setenv(KEY_REF_NAME, DEFAULT_KEY)
    _point_worker_beta_key_ref(
        dashboard.homes,
        gateways["worker_beta"].port,
        "${env:" + KEY_REF_NAME + "}",
        f"API_SERVER_KEY={BETA_KEY}\n",
    )

    response = _post_leaving_no_trace(dashboard, scopes_after_key, "worker_beta")

    assert response.status_code == 200
    assert len(gateways["worker_beta"].requests) == 1
    assert gateways["worker_beta"].requests[0]["headers"]["authorization"] == f"Bearer {BETA_KEY}"
    assert gateways["default"].requests == [] and gateways["worker_alpha"].requests == []


PORT_REF_NAME = "PLACEHOLDER_GATEWAY_PORT_REF"
PORT_REFS = [
    pytest.param("${API_SERVER_PORT}", "API_SERVER_PORT", id="bare-ref"),
    pytest.param("${env:" + PORT_REF_NAME + "}", PORT_REF_NAME, id="env-ref"),
]


def _point_worker_beta_port_ref(homes, ref, dotenv):
    """worker_beta: port in config.yaml as a ``${...}`` reference beside its key, and its own .env."""
    config = {
        "model": "test-model",
        "platforms": {"api_server": {"extra": {"port": ref, "key": BETA_KEY}}},
    }
    (homes["worker_beta"] / "config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    (homes["worker_beta"] / ".env").write_text(dotenv, encoding="utf-8")


@pytest.mark.parametrize("ref,name", PORT_REFS)
def test_port_reference_resolves_from_the_target_profiles_own_env(
    dashboard, stubs, monkeypatch, scopes_after_key, ref, name
):
    gateways = _point_every_gateway(dashboard, stubs, monkeypatch)
    monkeypatch.setenv(name, str(gateways["default"].port))
    _point_worker_beta_port_ref(dashboard.homes, ref, f"{name}={gateways['worker_beta'].port}\n")

    response = _post_leaving_no_trace(dashboard, scopes_after_key, "worker_beta")

    assert response.status_code == 200
    assert len(gateways["worker_beta"].requests) == 1
    sent = gateways["worker_beta"].requests[0]
    assert sent["method"] == "POST" and sent["target"] == RESEND_PATH
    assert sent["headers"]["authorization"] == f"Bearer {BETA_KEY}"
    assert gateways["default"].requests == [] and gateways["worker_alpha"].requests == []


def test_missing_port_reference_falls_back_to_the_target_profiles_own_env_port(
    dashboard, stubs, monkeypatch, scopes_after_key
):
    gateways = _point_every_gateway(dashboard, stubs, monkeypatch)
    monkeypatch.setenv(PORT_REF_NAME, str(gateways["default"].port))
    _point_worker_beta_port_ref(
        dashboard.homes,
        "${env:" + PORT_REF_NAME + "}",
        f"API_SERVER_PORT={gateways['worker_beta'].port}\n",
    )

    response = _post_leaving_no_trace(dashboard, scopes_after_key, "worker_beta")

    assert response.status_code == 200
    assert len(gateways["worker_beta"].requests) == 1
    assert gateways["worker_beta"].requests[0]["headers"]["authorization"] == f"Bearer {BETA_KEY}"
    assert gateways["default"].requests == [] and gateways["worker_alpha"].requests == []


@pytest.mark.parametrize(
    "key_line",
    [
        pytest.param(f"export API_SERVER_KEY={WORKER_KEY}", id="exported"),
        pytest.param(f'API_SERVER_KEY="{WORKER_KEY}" # placeholder comment', id="double-quoted-comment"),
        pytest.param(f"API_SERVER_KEY='{WORKER_KEY}'  # placeholder comment", id="single-quoted-comment"),
        pytest.param(f"API_SERVER_KEY={WORKER_KEY} # placeholder comment", id="unquoted-comment"),
    ],
)
def test_profile_dotenv_key_in_valid_dotenv_syntax_is_the_bearer(dashboard, stubs, monkeypatch, key_line):
    gateways = _point_every_gateway(dashboard, stubs, monkeypatch)
    (dashboard.homes["worker_alpha"] / ".env").write_text(
        f"API_SERVER_PORT={gateways['worker_alpha'].port}\n{key_line}\n", encoding="utf-8"
    )

    response = _post(dashboard, "owner", profile="worker_alpha")

    assert response.status_code == 200
    assert len(gateways["worker_alpha"].requests) == 1
    assert gateways["worker_alpha"].requests[0]["headers"]["authorization"] == f"Bearer {WORKER_KEY}"
    assert gateways["default"].requests == [] and gateways["worker_beta"].requests == []


def test_key_never_appears_in_any_answer_or_log_line(dashboard, stubs, every_log_line):
    stub = stubs()
    _point_worker_alpha(dashboard.homes, stub.port)
    answers = []

    answers.append(_post(dashboard, "owner"))
    assert answers[-1].status_code == 200
    assert stub.requests[0]["headers"]["authorization"] == f"Bearer {WORKER_KEY}"

    for status, payload in ((401, GATEWAY_AUTH_FAILED), (403, ROUTE_NOT_ALLOWED)):
        stub.answer(status, payload)
        answers.append(_post(dashboard, "automations"))
        assert answers[-1].status_code == status

    stub.drop = True
    answers.append(_post(dashboard, "owner"))
    assert answers[-1].status_code == 202

    _point_worker_alpha(dashboard.homes, _free_port())
    answers.append(_post(dashboard, "owner"))
    assert answers[-1].status_code == 503

    for response in answers:
        seen = response.text + " ".join(f"{k}: {v}" for k, v in response.headers.items())
        assert not any(key in seen for key in ALL_KEYS)
        assert "Bearer" not in seen
    for _name, _level, text in every_log_line.records:
        assert not any(key in text for key in ALL_KEYS), text
        assert "Bearer" not in text, text

    warnings = [
        text
        for name, level, text in every_log_line.records
        if name == "hermes_cli.web_server" and level >= logging.WARNING
    ]
    assert len(warnings) == 2  # one for the dropped answer, one for the refused connection
    for text in warnings:
        for leak in (EXECUTION_ID, REQUEST_ID, "127.0.0.1", "http://", str(stub.port), "/resend"):
            assert leak not in text, text


@pytest.mark.parametrize(
    "execution_id",
    ["a/b", "..", ".", "x?y=1", "x#y", "%2e%2e", "../../api/cron/fire", "a/../../fire"],
)
def test_execution_id_stays_one_path_segment(dashboard, stubs, execution_id):
    stub = stubs()
    _point_worker_alpha(dashboard.homes, stub.port)
    forwarder = dashboard.web_server._forward_cron_resend_to_gateway

    loop = asyncio.new_event_loop()
    try:
        result = loop.run_until_complete(forwarder("worker_alpha", execution_id, REQUEST_ID))
    finally:
        loop.close()

    assert result is not None
    assert result[0] == 200
    assert len(stub.requests) == 1
    target = stub.requests[0]["target"]
    prefix, suffix = "/api/cron/executions/", "/resend"
    assert target.startswith(prefix) and target.endswith(suffix)
    assert "?" not in target and "#" not in target
    segment = target[len(prefix) : -len(suffix)]
    assert "/" not in segment
    assert segment not in (".", "..")
    assert unquote(segment) == execution_id
    assert json.loads(stub.requests[0]["body"]) == {"request_id": REQUEST_ID}


# The TestClient decodes the request path twice (httpx's URL.path, then
# Starlette's unquote), so "%252e%252e" reaches the route as "..".
@pytest.mark.parametrize(
    "raw_segment,execution_id",
    [
        pytest.param("x%3Fy%3D1", "x?y=1", id="encoded-query"),
        pytest.param("%252e%252e", "..", id="double-encoded-dots"),
        pytest.param("x%23y", "x#y", id="encoded-fragment"),
    ],
)
def test_encoded_ids_through_the_route_stay_one_segment(dashboard, stubs, raw_segment, execution_id):
    stub = stubs()
    _point_worker_alpha(dashboard.homes, stub.port)

    response = _post(dashboard, "owner", path=f"/api/cron/executions/{raw_segment}/resend")

    assert response.status_code == 200
    assert len(stub.requests) == 1
    target = stub.requests[0]["target"]
    assert target.startswith("/api/cron/executions/") and target.endswith("/resend")
    segment = target[len("/api/cron/executions/") : -len("/resend")]
    assert "/" not in segment and "?" not in target and "#" not in target
    assert segment not in (".", "..")
    assert unquote(segment) == execution_id


def test_unknown_profile_is_404_before_anything_is_sent_or_audited(dashboard, stubs):
    stub = stubs()
    _point_worker_alpha(dashboard.homes, stub.port)

    response = _post(dashboard, "automations", profile="ghost")

    assert response.status_code == 404
    assert response.json() == {"detail": "Profile 'ghost' does not exist."}
    assert stub.requests == []
    assert _audit_lines() == []


def test_invalid_profile_name_is_400_before_anything_is_sent_or_audited(dashboard, stubs):
    stub = stubs()
    _point_worker_alpha(dashboard.homes, stub.port)

    response = _post(dashboard, "automations", profile="Bad Name!")

    assert response.status_code == 400
    assert stub.requests == []
    assert _audit_lines() == []
