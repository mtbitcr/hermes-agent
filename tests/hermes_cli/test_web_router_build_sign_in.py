"""Build sign-in routes: the pasted Claude Code setup token and the Claude usage left.

PUT /api/providers/build-sign-in saves a pasted setup token in the one
machine-wide sign-in file, GET /api/providers/build-sign-in says whether one is
in place and how long it still works, and GET /api/providers/usage/anthropic
says how much of each Claude usage window is left. The owner and the Models
machine token reach them; every caller GET /api/providers/oauth refuses is
refused. Every token here is synthetic and the usage read is a local fake; no
answer, log record or audit entry may hold any part of a token.
"""

import importlib
import importlib.util
import json
import logging
import os
import re
import stat
import sys
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from agent.account_usage import AccountUsageSnapshot, AccountUsageWindow

ROUTER = "hermes_cli.web_routers.build_sign_in"
SIGN_IN_PATH = "/api/providers/build-sign-in"
USAGE_PATH = "/api/providers/usage/anthropic"
OAUTH_PATH = "/api/providers/oauth"
#: Where the file lives under the root Hermes home, computed here on its own.
FILE_PARTS = ("secrets", "build-sign-in.json")
LIFETIME = timedelta(days=365)

BUILD_TOKEN = "sk-ant-oat01-SYNTHETIC_bsi_route_" + "Q7x9Kp2M" * 5
SEEDED_TOKEN = "sk-ant-oat01-SYNTHETIC_bsi_seeded_" + "Zr4Lm8Tw" * 5
CONSOLE_KEY = "sk-ant-api03-SYNTHETIC_bsi_console_" + "Wn5Vb3Hc" * 5
NO_PREFIX = "SYNTHETIC_bsi_noprefix_" + "Jd6Ty1Rf" * 5
SPACED = BUILD_TOKEN[:36] + " " + BUILD_TOKEN[36:]


def _sized(length, prefix="sk-ant-oat01-SYNTHETIC_bsi_"):
    """A setup-token-shaped synthetic value exactly ``length`` characters long."""
    return (prefix + "Kp2M7x9Q-_" * 40)[:length]


SYNTHETIC_TOKENS = (BUILD_TOKEN, SEEDED_TOKEN, CONSOLE_KEY, NO_PREFIX, SPACED) + tuple(
    _sized(length) for length in (39, 40, 300, 301)
)
#: One plain sentence: a capital, no other sentence end, one closing mark.
ONE_SENTENCE = re.compile(r"[A-Z][^.!?\n]*[.!?]")

WINDOWS = (
    AccountUsageWindow(
        label="Current session", used_percent=37.0, reset_at=datetime(2026, 10, 21, 15, 0, tzinfo=timezone.utc)
    ),
    AccountUsageWindow(
        label="Current week",
        used_percent=82.4,
        reset_at=datetime(2026, 10, 24, 9, 30, tzinfo=timezone(timedelta(hours=2))),
    ),
    AccountUsageWindow(label="Nearly spent week", used_percent=99.6),
    AccountUsageWindow(label="Opus week", used_percent=104.0),
    AccountUsageWindow(label="Sonnet week"),
    AccountUsageWindow(label="Unreadable week", used_percent=float("nan")),
)
EXTRA_USAGE = "Extra usage: 4.20 / 50.00 USD"
USAGE_LEFT = {
    "state": "available",
    "windows": [
        {"label": "Current session", "percent_left": 63, "resets_at": "2026-10-21T15:00:00+00:00"},
        {"label": "Current week", "percent_left": pytest.approx(17.6), "resets_at": "2026-10-24T07:30:00+00:00"},
        {"label": "Nearly spent week", "percent_left": pytest.approx(0.4), "resets_at": None},
        {"label": "Opus week", "percent_left": 0, "resets_at": None},
        {"label": "Sonnet week", "percent_left": None, "resets_at": None},
        {"label": "Unreadable week", "percent_left": None, "resets_at": None},
    ],
    "details": [EXTRA_USAGE],
}
NOT_OAUTH = "Anthropic account limits are only available for OAuth-backed Claude accounts."


def _snapshot(windows=(), details=(), unavailable_reason=None):
    return AccountUsageSnapshot(
        provider="anthropic",
        source="oauth_usage_api",
        fetched_at=datetime.now(timezone.utc),
        windows=tuple(windows),
        details=tuple(details),
        unavailable_reason=unavailable_reason,
    )


def _holds_part_of(text, value, size=8):
    """True when ``text`` holds ``value`` or any ``size``-character run of it."""
    if not value:
        return False
    if len(value) <= size:
        return value in text
    return any(value[start : start + size] in text for start in range(len(value) - size + 1))


def _router():
    """The routes' module, asserted inside a test so its absence fails that test."""
    assert importlib.util.find_spec(ROUTER) is not None, f"{ROUTER} does not exist"
    return importlib.import_module(ROUTER)


@pytest.fixture()
def home(tmp_path, monkeypatch):
    """The root Hermes home the dashboard runs with."""
    root = tmp_path / "hermes"
    root.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(root))
    return root


class _UsageReads:
    """Stands in for ``fetch_account_usage``: records each read, answers ``answer`` or raises it."""

    def __init__(self):
        self.calls = []
        self.answer = None

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        if isinstance(self.answer, BaseException):
            raise self.answer
        return self.answer


@pytest.fixture(autouse=True)
def usage_reads(monkeypatch):
    """No test reaches Anthropic: every usage read is this local fake."""
    from agent import account_usage

    reads = _UsageReads()
    monkeypatch.setattr(account_usage, "fetch_account_usage", reads)
    return reads


@pytest.fixture(autouse=True)
def fresh_usage_cache():
    """Each test starts without a usage answer kept from an earlier test."""

    def forget():
        module = sys.modules.get(ROUTER)
        if module is not None:
            module._usage_cache = None

    forget()
    yield
    forget()


def _restore_state(state, saved):
    for attr, previous in saved.items():
        if previous is None:
            if hasattr(state, attr):
                delattr(state, attr)
        else:
            setattr(state, attr, previous)


@pytest.fixture()
def dash(home, tmp_path):
    """The real dashboard app with an owner session and a token for every machine surface.

    Besides the working Models token there is a revoked one, one never issued
    and one with the wrong secret; every token is minted here, under tmp_path.
    """
    from fastapi.testclient import TestClient

    from hermes_cli import web_server
    from hermes_cli.dashboard_auth import register_provider
    from hermes_cli.dashboard_auth.registry import restore_registration, snapshot_registration
    from plugins.dashboard_auth.raphael_workspace import (
        AutomationsManageTokenProvider,
        ConnectionsManageTokenProvider,
        ModelsManageTokenProvider,
        RecommendationsReadTokenProvider,
        WorkspaceReadTokenProvider,
        token_store,
    )
    from plugins.dashboard_auth.raphael_workspace.model_policy import register_models_machine_routes

    register_models_machine_routes()
    token_dir = tmp_path / "machine-tokens"
    token_dir.mkdir(mode=0o700)

    def issue(name, surface):
        path = token_dir / f"{name}.token"
        record = token_store.issue(out_path=path, surface=surface)
        return record, "Bearer " + path.read_text(encoding="utf-8").strip()

    models_record, models = issue("models", token_store.MODELS_SURFACE)
    revoked_record, revoked = issue("models-revoked", token_store.MODELS_SURFACE)
    token_store.revoke(revoked_record.token_id)
    bearers = {
        "models": models,
        "models_revoked": revoked,
        "models_unknown": "Bearer hrm1_SyntheticUnknownId.placeholder-secret",
        "models_wrong_secret": f"Bearer {models_record.token_id}.wrong-secret",
        "workspace": issue("workspace", token_store.WORKSPACE_SURFACE)[1],
        "recommendations": issue("recommendations", token_store.RECOMMENDATIONS_SURFACE)[1],
        "connections": issue("connections", token_store.CONNECTIONS_SURFACE)[1],
        "automations": issue("automations", token_store.AUTOMATIONS_SURFACE)[1],
        "legacy_wrong": "Bearer placeholder-wrong-session-token",
    }
    headers = {who: {"Authorization": bearer} for who, bearer in bearers.items()}
    headers["owner"] = {web_server._SESSION_HEADER_NAME: web_server._SESSION_TOKEN}
    headers["none"] = {}
    headers["wrong_session"] = {web_server._SESSION_HEADER_NAME: "placeholder-wrong-session-token"}

    providers = [
        WorkspaceReadTokenProvider(),
        RecommendationsReadTokenProvider(),
        ConnectionsManageTokenProvider(),
        AutomationsManageTokenProvider(),
        ModelsManageTokenProvider(),
    ]
    added = [provider for provider in providers if snapshot_registration(provider.name) is None]
    for provider in added:
        register_provider(provider)

    state = web_server.app.state
    saved = {attr: getattr(state, attr, None) for attr in ("auth_required", "bound_host")}
    state.auth_required = False
    state.bound_host = None
    client = TestClient(web_server.app)

    def call(method, path, who, *, params=None, extra=None, via=None, **kwargs):
        sent = dict(headers[who])
        sent.update(extra or {})
        return (via or client).request(method, path, params=params, headers=sent, **kwargs)

    @contextmanager
    def gated():
        """The same app behind the sign-in gate, as on a public host."""
        from tests.hermes_cli.conftest_dashboard_auth import StubAuthProvider

        stub = StubAuthProvider()
        stub_added = snapshot_registration(stub.name) is None
        if stub_added:
            register_provider(stub)
        before = {attr: getattr(state, attr, None) for attr in ("auth_required", "bound_host", "bound_port")}
        state.auth_required = True
        state.bound_host = "fly-app.fly.dev"
        state.bound_port = 443
        gated_client = TestClient(web_server.app, base_url="https://fly-app.fly.dev")
        try:
            yield gated_client
        finally:
            gated_client.close()
            _restore_state(state, before)
            if stub_added:
                restore_registration(stub.name, stub, None)

    try:
        yield SimpleNamespace(call=call, gated=gated, web_server=web_server)
    finally:
        client.close()
        _restore_state(state, saved)
        for provider in added:
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


def _audit_text():
    from hermes_cli.dashboard_auth import audit as audit_mod

    path = audit_mod._resolve_log_path()
    return path.read_text(encoding="utf-8") if path.exists() else ""


def _successes(method, path):
    """The Models-machine success entries in the audit log for one route."""
    entries = []
    for raw in _audit_text().splitlines():
        try:
            entry = json.loads(raw)
        except ValueError:
            continue
        if (entry.get("event"), entry.get("method"), entry.get("route_template")) == (
            "token_auth_success",
            method,
            path,
        ):
            entries.append(entry)
    return entries


def _made(age, *, token=SEEDED_TOKEN):
    """A sign-in record whose token was made ``age`` ago."""
    return {"token": token, "made_at": (datetime.now(timezone.utc) - age).isoformat()}


def _place(home, record, *, mode=0o600):
    """Put a sign-in file in place the way an operator would."""
    path = home.joinpath(*FILE_PARTS)
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(record, handle)
    os.chmod(path, mode)
    return path


# ---------------------------------------------------------------------------
# 1. Saving a valid token
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("token", [BUILD_TOKEN, _sized(40), _sized(300)], ids=["typical", "40-chars", "300-chars"])
@pytest.mark.parametrize("who", ["models", "owner"])
def test_a_valid_token_is_saved_at_mode_0600_and_never_answered(dash, home, who, token):
    from plugins.dashboard_auth.raphael_workspace.build_sign_in_file import read_build_token

    before = datetime.now(timezone.utc).replace(microsecond=0)
    response = dash.call("PUT", SIGN_IN_PATH, who, json={"token": token})
    after = datetime.now(timezone.utc)

    assert response.status_code == 200, response.text
    body = response.json()
    assert set(body) == {"state", "made_on", "works_until"}
    assert body["state"] == "saved"
    made_on = datetime.fromisoformat(body["made_on"])
    works_until = datetime.fromisoformat(body["works_until"])
    assert made_on.utcoffset() == timedelta(0)
    assert before <= made_on <= after
    assert works_until - made_on == LIFETIME
    path = home.joinpath(*FILE_PARTS)
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert read_build_token() == (token, works_until)
    assert not _holds_part_of(response.text, token)
    assert not any(_holds_part_of(f"{name}: {value}", token) for name, value in response.headers.items())
    assert not _holds_part_of(_audit_text(), token)
    assert len(_successes("PUT", SIGN_IN_PATH)) == (1 if who == "models" else 0)


# ---------------------------------------------------------------------------
# 2. Refusing anything that is not a setup token
# ---------------------------------------------------------------------------

REFUSED_BODIES = [
    pytest.param({"json": {"token": SPACED}}, SPACED, id="space-inside"),
    pytest.param({"json": {"token": BUILD_TOKEN + "\n"}}, BUILD_TOKEN, id="trailing-newline"),
    pytest.param({"json": {"token": CONSOLE_KEY}}, CONSOLE_KEY, id="console-key-prefix"),
    pytest.param({"json": {"token": NO_PREFIX}}, NO_PREFIX, id="no-prefix"),
    pytest.param({"json": {"token": _sized(39)}}, _sized(39), id="39-chars"),
    pytest.param({"json": {"token": _sized(301)}}, _sized(301), id="301-chars"),
    pytest.param({"json": {"token": ""}}, "", id="empty"),
    pytest.param({"json": {"token": 12345678901234567890}}, "12345678901234567890", id="number"),
    pytest.param({"json": {"token": None}}, "", id="null"),
    pytest.param({"json": {"token": [BUILD_TOKEN]}}, BUILD_TOKEN, id="list"),
    pytest.param({"json": {"setup_token": BUILD_TOKEN}}, BUILD_TOKEN, id="no-token-key"),
    pytest.param({"content": ("token=" + BUILD_TOKEN).encode()}, BUILD_TOKEN, id="not-json"),
    pytest.param({"json": [BUILD_TOKEN]}, BUILD_TOKEN, id="json-array"),
    pytest.param({"json": BUILD_TOKEN}, BUILD_TOKEN, id="json-string"),
]


def _assert_refused(response, sent):
    assert response.status_code == 422, response.text
    body = response.json()
    assert list(body) == ["detail"] and isinstance(body["detail"], str), body
    assert ONE_SENTENCE.fullmatch(body["detail"]), body["detail"]
    for value in (sent, SEEDED_TOKEN):
        assert not _holds_part_of(response.text, value)


@pytest.mark.parametrize("body,sent", REFUSED_BODIES)
@pytest.mark.parametrize("who", ["models", "owner"])
def test_a_refused_value_answers_422_in_one_sentence_and_writes_nothing(dash, home, who, body, sent):
    from plugins.dashboard_auth.raphael_workspace.build_sign_in_file import write_build_token

    secrets_dir = home / FILE_PARTS[0]
    _assert_refused(dash.call("PUT", SIGN_IN_PATH, who, **body), sent)
    assert not secrets_dir.exists()

    write_build_token(SEEDED_TOKEN)
    path = home.joinpath(*FILE_PARTS)
    kept = (path.read_bytes(), path.stat().st_mtime_ns, sorted(os.listdir(secrets_dir)))
    _assert_refused(dash.call("PUT", SIGN_IN_PATH, who, **body), sent)
    assert (path.read_bytes(), path.stat().st_mtime_ns, sorted(os.listdir(secrets_dir))) == kept
    assert _successes("PUT", SIGN_IN_PATH) == []


# ---------------------------------------------------------------------------
# 3. Status: none, active or too_old, never the token
# ---------------------------------------------------------------------------


def _no_file(dash, home):
    return None


def _file_made(age, *, mode=0o600):
    def seed(dash, home):
        record = _made(age)
        _place(home, record, mode=mode)
        return datetime.fromisoformat(record["made_at"])

    return seed


def _saved_by_put(dash, home):
    response = dash.call("PUT", SIGN_IN_PATH, "owner", json={"token": SEEDED_TOKEN})
    assert response.status_code == 200, response.text
    return datetime.fromisoformat(response.json()["made_on"])


STATUS_CASES = [
    pytest.param(_no_file, "none", None, id="no-file"),
    pytest.param(_file_made(timedelta(days=10), mode=0o644), "none", None, id="not-mode-0600"),
    pytest.param(_file_made(timedelta(days=10)), "active", 355, id="made-10-days-ago"),
    pytest.param(_file_made(LIFETIME - timedelta(hours=2)), "active", 1, id="2-hours-left"),
    pytest.param(_file_made(LIFETIME + timedelta(minutes=1)), "too_old", None, id="ran-out-a-minute-ago"),
    pytest.param(_file_made(timedelta(days=800)), "too_old", None, id="ran-out-long-ago"),
    pytest.param(_saved_by_put, "active", 365, id="saved-just-now"),
]


@pytest.mark.parametrize("seed,state,days_left", STATUS_CASES)
@pytest.mark.parametrize("who", ["models", "owner"])
def test_status_says_none_active_or_too_old_and_never_holds_the_token(dash, home, who, seed, state, days_left):
    made = seed(dash, home)

    response = dash.call("GET", SIGN_IN_PATH, who)

    assert response.status_code == 200, response.text
    assert not _holds_part_of(response.text, SEEDED_TOKEN)
    body = response.json()
    if state == "active":
        assert set(body) == {"state", "made_on", "works_until", "days_left"}
        assert (body["state"], body["days_left"]) == ("active", days_left)
        assert datetime.fromisoformat(body["made_on"]) == made
        assert datetime.fromisoformat(body["works_until"]) == made + LIFETIME
    else:
        assert body == {"state": state}
    assert len(_successes("GET", SIGN_IN_PATH)) == (1 if who == "models" else 0)


# ---------------------------------------------------------------------------
# 4-6. Usage left
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("who", ["models", "owner"])
def test_usage_maps_each_window_to_percent_left_and_reset_time(dash, usage_reads, who):
    usage_reads.answer = _snapshot(WINDOWS, [EXTRA_USAGE])

    response = dash.call("GET", USAGE_PATH, who)

    assert response.status_code == 200, response.text
    assert response.json() == USAGE_LEFT
    assert usage_reads.calls == [(("anthropic",), {})]


@pytest.mark.parametrize("first_read", ["available", "unavailable"])
def test_a_second_usage_call_within_60_seconds_does_not_read_again(dash, usage_reads, monkeypatch, first_read):
    clock = [1000.0]
    monkeypatch.setattr(_router(), "time", SimpleNamespace(monotonic=lambda: clock[0]))
    available, unavailable = _snapshot(WINDOWS, [EXTRA_USAGE]), None
    usage_reads.answer = available if first_read == "available" else unavailable

    first = dash.call("GET", USAGE_PATH, "models")
    assert first.status_code == 200, first.text
    assert len(usage_reads.calls) == 1

    usage_reads.answer = unavailable if first_read == "available" else available
    clock[0] += 59.0
    again = dash.call("GET", USAGE_PATH, "owner")
    assert again.status_code == 200, again.text
    assert again.json() == first.json()
    assert len(usage_reads.calls) == 1

    clock[0] += 2.0
    later = dash.call("GET", USAGE_PATH, "models")
    assert later.status_code == 200, later.text
    assert len(usage_reads.calls) == 2
    assert later.json() != first.json()


FAILED_READS = [
    pytest.param(None, None, id="nothing-read"),
    pytest.param(_snapshot(unavailable_reason=NOT_OAUTH), NOT_OAUTH, id="not-an-oauth-account"),
    pytest.param(_snapshot(), None, id="empty-snapshot"),
    pytest.param(
        _snapshot(WINDOWS, [EXTRA_USAGE], unavailable_reason="Usage is paused for this account."),
        "Usage is paused for this account.",
        id="windows-but-unavailable",
    ),
    pytest.param(RuntimeError("synthetic usage read failure"), None, id="read-raises"),
]


@pytest.mark.parametrize("answer,reason", FAILED_READS)
@pytest.mark.parametrize("who", ["models", "owner"])
def test_a_failed_usage_read_answers_unavailable_never_zero(dash, usage_reads, who, answer, reason):
    usage_reads.answer = answer

    response = dash.call("GET", USAGE_PATH, who)

    assert response.status_code == 200, response.text
    body = response.json()
    assert set(body) == {"state", "reason"}, body
    assert body["state"] == "unavailable"
    assert isinstance(body["reason"], str) and ONE_SENTENCE.fullmatch(body["reason"]), body
    if reason is not None:
        assert body["reason"] == reason
    assert len(usage_reads.calls) == 1


# ---------------------------------------------------------------------------
# 7. Access: the same refusals as GET /api/providers/oauth
# ---------------------------------------------------------------------------

ROUTES = [
    pytest.param("PUT", SIGN_IN_PATH, id="save"),
    pytest.param("GET", SIGN_IN_PATH, id="status"),
    pytest.param("GET", USAGE_PATH, id="usage"),
]
#: (label, who, extra headers) refused behind the sign-in gate.
GATED_REFUSALS = (
    ("no session cookie", "none", {}),
    ("a session cookie that does not verify", "none", {"Cookie": "hermes_session_at=placeholder-unverifiable"}),
)


def _refusals(dash):
    """(label, who, query, extra headers) for each caller refused on the loopback dashboard."""
    session = dash.web_server._SESSION_TOKEN
    return (
        ("no credential", "none", None, {}),
        ("a wrong session token", "wrong_session", None, {}),
        ("the session token only in the query", "none", {"token": session}, {}),
        ("a wrong legacy bearer", "legacy_wrong", None, {}),
        ("a Models token never issued", "models_unknown", None, {}),
        ("a Models token with the wrong secret", "models_wrong_secret", None, {}),
        ("a revoked Models token", "models_revoked", None, {}),
        ("the Models token for another profile", "models", {"profile": "raphael-planner"}, {}),
        ("the Models token with a method override", "models", None, {"X-HTTP-Method-Override": "GET"}),
        ("a Workspace token", "workspace", None, {}),
        ("a recommendations token", "recommendations", None, {}),
        ("a Connections token", "connections", None, {}),
        ("an Automations token", "automations", None, {}),
    )


@pytest.mark.parametrize("method,path", ROUTES)
def test_each_route_refuses_every_caller_get_oauth_refuses(dash, home, usage_reads, method, path):
    body = {"json": {"token": BUILD_TOKEN}} if method == "PUT" else {}

    def same_refusal(label, who, params, extra, via=None):
        oauth = dash.call("GET", OAUTH_PATH, who, params=params, extra=extra, via=via)
        assert oauth.status_code >= 400, (label, oauth.status_code)
        response = dash.call(method, path, who, params=params, extra=extra, via=via, **body)
        assert response.status_code == oauth.status_code, (label, response.status_code, response.text)

    for label, who, params, extra in _refusals(dash):
        same_refusal(label, who, params, extra)
    with dash.gated() as gated_client:
        for label, who, extra in GATED_REFUSALS:
            same_refusal(label, who, None, extra, via=gated_client)

    assert not (home / FILE_PARTS[0]).exists()
    assert usage_reads.calls == []


# ---------------------------------------------------------------------------
# 8. Logs
# ---------------------------------------------------------------------------


def test_no_log_record_holds_a_synthetic_token(dash, usage_reads, every_log_line, monkeypatch):
    from plugins.dashboard_auth.raphael_workspace import build_sign_in_file

    answers = []

    def call(status, method, path, who, **kwargs):
        response = dash.call(method, path, who, **kwargs)
        assert response.status_code == status, (method, path, who, response.text)
        answers.append(response)
        return response

    # A failing read whose error text carries a token must not carry it into a log.
    usage_reads.answer = RuntimeError("synthetic usage read failure for " + SEEDED_TOKEN)
    for who in ("models", "owner"):
        call(200, "PUT", SIGN_IN_PATH, who, json={"token": BUILD_TOKEN})
        for refused in (SPACED, CONSOLE_KEY, NO_PREFIX, _sized(39), _sized(301)):
            call(422, "PUT", SIGN_IN_PATH, who, json={"token": refused})
        assert call(200, "GET", SIGN_IN_PATH, who).json()["state"] == "active"
        assert call(200, "GET", USAGE_PATH, who).json()["state"] == "unavailable"

    def disk_full(token):
        raise OSError(28, "No space left on device", token)

    monkeypatch.setattr(build_sign_in_file, "write_build_token", disk_full)
    for who in ("models", "owner"):
        detail = call(500, "PUT", SIGN_IN_PATH, who, json={"token": SEEDED_TOKEN}).json()["detail"]
        assert ONE_SENTENCE.fullmatch(detail), detail

    assert every_log_line.records
    audit = _audit_text()
    for token in SYNTHETIC_TOKENS:
        for _, _, text in every_log_line.records:
            assert not _holds_part_of(text, token), text
        assert not _holds_part_of(audit, token)
        for response in answers:
            assert not _holds_part_of(response.text, token), response.text
