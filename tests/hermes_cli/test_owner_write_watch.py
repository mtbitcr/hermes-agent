"""Hourly check of the owner write routes: plan section (e), tests 1 to 9, the owner notes'
regression tests for the four review findings of round 1 and the two of round 2, round 4's
regression tests for own executions that share a claimed_at across a page boundary, and round 5's
tests of the owner rule for the listener address.

Tests 1 to 9 fake the listener, the page check, the clock and the delivery record. The regression
tests use a real HTTP server on 127.0.0.1 and the real execution and delivery stores in the test's
temporary HERMES_HOME: nothing here reaches another host or a production service.
"""

import asyncio
import http.server
import json
import os
import signal
import sqlite3
import subprocess
import sys
import threading
import time
import uuid
from contextlib import closing, contextmanager
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import psutil
import pytest

from hermes_cli import owner_write_watch as watch
from hermes_constants import get_hermes_home

START = datetime(2026, 10, 5, 9, 0, tzinfo=timezone.utc)
HOUR = timedelta(hours=1)
REFUSALS = (
    (401, "gateway_auth_failed"),
    (403, "route_not_allowed"),
    (404, "owner_workspace_not_enabled"),
    (503, "owner_workspace_unavailable"),
)


def envelope(code, message="Request refused"):
    return {"error": {"message": message, "type": "invalid_request_error", "param": None, "code": code}}


def healthy(probe):
    return probe.status, envelope(probe.code, probe.message or "Not found")


def every_probe():
    return [probe for table in watch.PROBES.values() for probe in table]


def decisions_probe():
    return next(probe for probe in watch.PROBES["default"] if "/decisions/" in probe.path)


def runs_probe():
    return next(probe for probe in watch.PROBES["default"] if probe.path.endswith("/v1/runs"))


def release_probes():
    return [probe for probe in watch.PROBES["default"] if "/owner-workspace/release/" in probe.path]


def repository_probes():
    return [probe for probe in watch.PROBES["default"] if probe.path.endswith(("/repository/check", "/repository/connect"))]


@pytest.fixture(autouse=True)
def no_real_wait(monkeypatch):
    monkeypatch.setattr(watch, "time", SimpleNamespace(monotonic=time.monotonic, sleep=lambda seconds: None))


class Host:
    """One profile's hourly job with a fake listener, page check, clock and delivery record."""

    def __init__(self, tmp_path, profile="default"):
        self.profile = profile
        self.page_check = tmp_path / "page-check.sh"
        self.page_check.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        self.answers = {probe.name: healthy(probe) for probe in watch.PROBES[profile]}
        self.requests = []
        self.page = watch.PASS
        self.delivery = "delivered"
        self.key = "k" * 40
        self.clock = START

    def send(self, method, url, body, key):
        self.requests.append((method, url, body, key))
        for probe in watch.PROBES[self.profile]:
            if probe.method == method and url.endswith(probe.path) and body == probe.body:
                return self.answers[probe.name]
        raise AssertionError(f"request outside the probe table: {method} {url}")

    def route_policy(self, method, path):
        for signal in watch.SIGNALS:
            if (signal.method, signal.path) == (method, path):
                return SimpleNamespace(method=method, path=path, required_scope=signal.scope)
        return None

    def refuse(self, probe, status, code):
        self.answers[probe.name] = (status, envelope(code))

    def heal(self, probe):
        self.answers[probe.name] = healthy(probe)

    def run(self, **overrides):
        options = dict(
            profile=self.profile,
            base="http://listener.test",
            send=self.send,
            run_page_check=lambda path: self.page,
            read_key=lambda: self.key,
            delivery_state=lambda announced: self.delivery,
            route_policy=self.route_policy,
            now=lambda: self.clock,
        )
        options.update(overrides)
        outcome = watch.run(str(self.page_check), **options)
        self.clock += HOUR
        return outcome


def messages(outcomes):
    return [outcome.text for outcome in outcomes if outcome.text]


def test_default_profile_probes_the_three_release_routes():
    """The default profile's table holds one probe for each owner release write route, with the
    exact path under /p/default, the body {} and the healthy answer 400 invalid_argument."""
    probes = release_probes()
    assert [(probe.method, probe.path, probe.body, probe.status, probe.code) for probe in probes] == [
        ("POST", f"/p/default/v1/owner-workspace/release/0/{action}", {}, 400, "invalid_argument")
        for action in ("accept", "defer", "start")
    ]
    assert all(probe.message is None and probe.label for probe in probes)
    assert len({probe.name for probe in watch.PROBES["default"]}) == len(watch.PROBES["default"])


def test_default_profile_probes_the_two_repository_routes():
    """The default profile's table holds one probe for each owner repository write route, check and
    connect, with the exact path under /p/default, the body {} and the healthy answer 400 invalid_argument."""
    probes = repository_probes()
    assert [(probe.method, probe.path, probe.body, probe.status, probe.code) for probe in probes] == [
        ("POST", f"/p/default/v1/owner-workspace/projects/0/repository/{action}", {}, 400, "invalid_argument")
        for action in ("check", "connect")
    ]
    assert all(probe.message is None and probe.label for probe in probes)
    assert len({probe.name for probe in watch.PROBES["default"]}) == len(watch.PROBES["default"])


def test_default_profile_probes_the_input_upload_route():
    """The default profile's table holds one probe for the owner upload route of a card that waits for a
    file, with the exact path under /p/default, no payload at all (None) and the healthy answer 404
    decision_not_found. The Decisions probe is still the first probe on a decisions path."""
    probes = [probe for probe in watch.PROBES["default"] if "/inputs/" in probe.path]
    assert [(probe.method, probe.path, probe.body, probe.status, probe.code) for probe in probes] == [
        ("POST", f"/p/default/v1/owner-workspace/decisions/decision_{'0' * 32}/inputs/1", None, 404,
         "decision_not_found"),
    ]
    assert all(probe.message is None and probe.label for probe in probes)
    assert decisions_probe().name == "decisions"
    assert len({probe.name for probe in watch.PROBES["default"]}) == len(watch.PROBES["default"])


@pytest.mark.asyncio
async def test_the_input_upload_probe_gets_its_healthy_refusal_from_the_real_route(monkeypatch):
    """Sent by the check's own sender to the real upload route under /p/default, on a loopback server
    in this test's home with the owner workspace switched on, the probe's empty body gets 404
    decision_not_found with the route's own sentence, and the probe passes."""
    from aiohttp import web
    from aiohttp.test_utils import TestServer

    from gateway.config import PlatformConfig
    from gateway.platforms.api_server import APIServerAdapter

    [probe] = [probe for probe in watch.PROBES["default"] if "/inputs/" in probe.path]
    key = "probe-inputs-" + "4b7d19e2" * 4
    # gateway.run pins its config home at import; read this test's home instead.
    monkeypatch.setattr("gateway.run._hermes_home", get_hermes_home())
    (get_hermes_home() / "config.yaml").write_text(
        json.dumps({"gateway": {"api_server": {"owner_workspace": {"enabled": True}}}}), encoding="utf-8")
    adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={"key": key}))
    app = web.Application()
    for method, path, handler in adapter._http_route_table():
        if "/inputs/" in path:
            app.router.add_route(method, f"/p/{{profile}}{path}", handler)
    async with TestServer(app, host="127.0.0.1") as server:
        status, payload = await asyncio.to_thread(
            watch._http_send, probe.method, str(server.make_url(probe.path)), probe.body, key)

    assert (status, payload["error"]["code"]) == (404, "decision_not_found")
    assert payload["error"]["message"] == "This request is no longer waiting for a file."
    assert watch.classify(probe, status, payload) == watch.PASS


@pytest.mark.asyncio
async def test_the_input_upload_probe_puts_a_truly_empty_body_on_the_wire(monkeypatch):
    """A loopback relay records the bytes the check's own sender puts on the wire on their way to the real
    upload route: the upload probe sends an empty body (b''), with no JSON content type, and still gets
    404 decision_not_found; a probe with a JSON body still sends that JSON."""
    from aiohttp import web
    from aiohttp.test_utils import TestServer

    from gateway.config import PlatformConfig
    from gateway.platforms.api_server import APIServerAdapter

    [probe] = [probe for probe in watch.PROBES["default"] if "/inputs/" in probe.path]
    json_probe = next(probe for probe in watch.PROBES["default"] if probe.name == "repository_check")
    key = "probe-inputs-" + "6c2e8f05" * 4
    monkeypatch.setattr("gateway.run._hermes_home", get_hermes_home())
    (get_hermes_home() / "config.yaml").write_text(
        json.dumps({"gateway": {"api_server": {"owner_workspace": {"enabled": True}}}}), encoding="utf-8")
    adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={"key": key}))
    app = web.Application()
    for method, path, handler in adapter._http_route_table():
        if path.endswith(("/inputs/{n}", "/repository/check")):
            app.router.add_route(method, f"/p/{{profile}}{path}", handler)
    requests = []

    async def relay(reader, writer):
        """Pass one connection through to the route, keeping every byte the sender wrote."""
        upstream_reader, upstream_writer = await asyncio.open_connection("127.0.0.1", server.port)
        sent = bytearray()
        requests.append(sent)

        async def pump(source, sink, kept=None):
            while chunk := await source.read(65536):
                if kept is not None:
                    kept.extend(chunk)
                sink.write(chunk)
                await sink.drain()
            sink.close()

        await asyncio.gather(pump(reader, upstream_writer, sent), pump(upstream_reader, writer))

    async with TestServer(app, host="127.0.0.1") as server:
        wire = await asyncio.start_server(relay, "127.0.0.1", 0)
        base = f"http://127.0.0.1:{wire.sockets[0].getsockname()[1]}"
        async with wire:
            answers = [
                await asyncio.to_thread(watch._http_send, sent.method, base + sent.path, sent.body, key)
                for sent in (probe, json_probe)
            ]

    def split(request):
        head, _, body = bytes(request).partition(b"\r\n\r\n")
        return {line.split(b":", 1)[0].strip().lower(): line.split(b":", 1)[1].strip()
                for line in head.split(b"\r\n")[1:]}, body

    (headers, body), (json_headers, json_body) = (split(request) for request in requests)
    assert body == b""
    assert headers.get(b"content-length", b"0") == b"0" and b"content-type" not in headers
    status, payload = answers[0]
    assert (status, payload["error"]["code"]) == (404, "decision_not_found")
    assert watch.classify(probe, status, payload) == watch.PASS
    assert json.loads(json_body) == json_probe.body == {}
    assert json_headers[b"content-type"] == b"application/json"
    # Every other probe keeps its JSON body.
    assert [other.name for table in watch.PROBES.values() for other in table if not isinstance(other.body, dict)] == [
        "input_upload",
    ]


@pytest.mark.parametrize("status,code", REFUSALS)
def test_1_a_refusal_on_any_probed_route_counts_as_a_failure(status, code):
    """A refusal on any probed route counts as a failure: 401 gateway_auth_failed, 403
    route_not_allowed, 404 owner_workspace_not_enabled and 503 all fail; only the exact
    healthy status and code passes."""
    for probe in every_probe():
        assert watch.classify(probe, status, envelope(code)) == watch.FAIL, probe.name
        assert watch.classify(probe, status, None) == watch.FAIL, probe.name
        assert watch.classify(probe, *healthy(probe)) == watch.PASS, probe.name
        assert watch.classify(probe, probe.status, envelope("another_code")) == watch.FAIL, probe.name
        other_status = probe.status + 1
        assert watch.classify(probe, other_status, healthy(probe)[1]) == watch.FAIL, probe.name


@pytest.mark.parametrize("status,code", REFUSALS)
def test_1_a_refused_probe_fails_the_run(tmp_path, status, code):
    """A refusal on any probed route counts as a failure: 401 gateway_auth_failed, 403
    route_not_allowed, 404 owner_workspace_not_enabled and 503 all fail; only the exact
    healthy status and code passes."""
    for profile, table in watch.PROBES.items():
        for probe in table:
            host = Host(tmp_path, profile=profile)
            host.refuse(probe, status, code)
            outcome = host.run()
            assert outcome.code == 0
            assert outcome.kind == "alert", probe.name
            assert probe.label in outcome.text
            watch.state_path().unlink(missing_ok=True)


def test_2_a_404_with_any_other_code_is_a_failure(tmp_path):
    """A 404 with any other code is a failure (the switched-off Decisions case)."""
    for probe in every_probe():
        if probe.status != 404:
            continue
        for code in ("owner_workspace_not_enabled", "not_found", "", None):
            assert watch.classify(probe, 404, envelope(code)) == watch.FAIL, (probe.name, code)
        assert watch.classify(probe, 404, None) == watch.FAIL, probe.name

    host = Host(tmp_path)
    decisions = decisions_probe()
    host.refuse(decisions, 404, "owner_workspace_not_enabled")
    outcome = host.run()
    assert outcome.code == 0
    assert outcome.kind == "alert"
    assert decisions.label in outcome.text


def test_3_a_lasting_failure_alerts_once(tmp_path):
    """A lasting failure alerts once: three failing runs give one message."""
    host = Host(tmp_path)
    host.run()
    decisions = decisions_probe()
    host.refuse(decisions, 403, "route_not_allowed")
    outcomes = [host.run() for _ in range(3)]
    assert [outcome.code for outcome in outcomes] == [0, 0, 0]
    sent = messages(outcomes)
    assert len(sent) == 1
    assert decisions.label in sent[0]


def test_4_recovery_alerts_once(tmp_path):
    """Recovery alerts once: failing, then ok, then ok gives one recovery message with the
    outage length."""
    host = Host(tmp_path)
    decisions = decisions_probe()
    host.refuse(decisions, 403, "route_not_allowed")
    failing = host.run()
    host.heal(decisions)
    after = [host.run(), host.run()]
    assert failing.kind == "alert"
    sent = messages(after)
    assert len(sent) == 1
    assert after[0].kind == "recovery"
    assert watch.format_duration(HOUR) in sent[0]


def test_5_a_change_in_the_failing_set_gives_one_update(tmp_path):
    """A change in the failing set gives one update, or none (owner decision 1)."""
    host = Host(tmp_path)
    decisions, runs = decisions_probe(), runs_probe()
    host.refuse(decisions, 403, "route_not_allowed")
    alert = host.run()
    host.refuse(runs, 403, "route_not_allowed")
    after = [host.run(), host.run()]
    assert alert.kind == "alert"
    sent = messages(after)
    assert len(sent) == 1
    assert after[0].kind == "update"
    assert runs.label in sent[0]


def test_6_a_restart_with_the_same_state_file_sends_nothing(tmp_path):
    """A restart with the same state file sends nothing for an unchanged failure."""
    before = Host(tmp_path)
    before.refuse(decisions_probe(), 403, "route_not_allowed")
    assert before.run().kind == "alert"

    after = Host(tmp_path)
    after.clock = before.clock
    after.refuse(decisions_probe(), 403, "route_not_allowed")
    outcome = after.run()
    assert outcome.code == 0
    assert outcome.text == ""


def test_7_a_missed_hour_adds_one_line_once(tmp_path):
    """A missed hour: a stored last run older than two hours adds one line, once."""
    host = Host(tmp_path)
    last_run = host.clock
    host.run()
    host.clock = last_run + timedelta(hours=3)
    resumed_at = host.clock
    resumed = host.run()
    following = host.run()
    lines = resumed.text.splitlines()
    assert resumed.code == 0
    assert len(lines) == 1
    assert watch.format_time(last_run) in lines[0]
    assert watch.format_time(resumed_at) in lines[0]
    assert following.text == ""


def test_8_a_missing_key_exits_non_zero_and_sends_no_probe(tmp_path):
    """A missing key or an unreadable state file exits non-zero, and no probe is sent without
    a key."""
    host = Host(tmp_path)
    outcome = host.run(read_key=lambda: None)
    assert outcome.code != 0
    assert host.requests == []


def test_8_the_key_is_read_from_the_jobs_own_profile_only(tmp_path, monkeypatch):
    """A missing key or an unreadable state file exits non-zero, and no probe is sent without
    a key."""
    monkeypatch.setenv("API_SERVER_KEY", "e" * 40)
    host = Host(tmp_path)
    outcome = host.run(read_key=None)
    assert outcome.code != 0
    assert host.requests == []

    (get_hermes_home() / ".env").write_text("API_SERVER_KEY=" + "p" * 40 + "\n", encoding="utf-8")
    outcome = host.run(read_key=None)
    assert outcome.code == 0
    assert host.requests
    assert {request[3] for request in host.requests} == {"p" * 40}


def test_8_an_unreadable_state_file_exits_non_zero(tmp_path):
    """A missing key or an unreadable state file exits non-zero, and no probe is sent without
    a key."""
    host = Host(tmp_path)
    host.run()
    sent_before = len(host.requests)
    watch.state_path().write_text("{not json", encoding="utf-8")
    outcome = host.run()
    assert outcome.code != 0
    assert len(host.requests) == sent_before
    assert watch.state_path().read_text(encoding="utf-8") == "{not json"


@pytest.mark.parametrize("delivery,printed_again", [("failed", True), ("unknown", False)])
def test_9_an_undelivered_alert(tmp_path, delivery, printed_again):
    """An undelivered alert: if the previous run's delivery is failed, the alert prints again;
    if unknown, it does not (owner decision 2)."""
    host = Host(tmp_path)
    host.run()
    host.refuse(decisions_probe(), 403, "route_not_allowed")
    alert = host.run()
    host.delivery = delivery
    again = host.run()
    assert alert.kind == "alert"
    assert again.code == 0
    assert again.text == (alert.text if printed_again else "")


# --- Owner note, round 1: regression tests for the four review findings ---------------------------

AUTOMATIONS_SCOPE = "cron.automations.manage"
WATCH_JOB = "a1b2c3d4e5f6"
OTHER_JOB = "f6e5d4c3b2a1"
MINUTE = timedelta(minutes=1)


def automations_signals():
    return [signal for signal in watch.SIGNALS if signal.group == "automations"]


def clear_history_probe():
    return next(probe for probe in watch.PROBES["raphael-planner"] if probe.name == "conversation_history")


def test_finding1_automations_write_calls_are_bound_from_the_automations_client():
    """Each write call of the owner app's Automations client is a wiring signal with the
    Automations key's scope: create, pause, resume and resend, plus the two reads it makes."""
    zeros = "0" * 32
    assert {(signal.method, signal.path, signal.scope) for signal in automations_signals()} == {
        ("GET", "/api/cron/jobs", AUTOMATIONS_SCOPE),
        ("GET", "/api/cron/executions", AUTOMATIONS_SCOPE),
        ("POST", "/api/cron/jobs", AUTOMATIONS_SCOPE),
        ("POST", "/api/cron/jobs/000000000000/pause", AUTOMATIONS_SCOPE),
        ("POST", "/api/cron/jobs/000000000000/resume", AUTOMATIONS_SCOPE),
        ("POST", f"/api/cron/executions/{zeros}/resend", AUTOMATIONS_SCOPE),
    }


def test_finding1_removed_automations_write_permissions_fail_and_unchanged_pass():
    """Removed Automations write permissions give a failure, and unchanged permissions pass:
    read from the route policy the deployed dashboard code registers, cron routes included."""
    asked = []

    def deployed(method, path):
        asked.append((method, path))
        return watch._deployed_route_policy(method, path)

    unchanged = {result.name: result for result in watch._wiring(deployed, lambda: START)}
    assert {name: result.outcome for name, result in unchanged.items()} == {
        "automations": watch.PASS, "connections": watch.PASS, "models": watch.PASS,
    }
    assert {(signal.method, signal.path) for signal in automations_signals()} <= set(asked)

    writes = [signal for signal in automations_signals() if signal.method == "POST"]
    assert len(writes) == 4
    for removed in writes:
        def without(method, path, removed=removed):
            if (method, path) == (removed.method, removed.path):
                return None
            return watch._deployed_route_policy(method, path)

        results = {result.name: result for result in watch._wiring(without, lambda: START)}
        assert results["automations"].outcome == watch.FAIL, removed.path
        assert f"{removed.method} {removed.path}" in results["automations"].detail
        assert results["connections"].outcome == results["models"].outcome == watch.PASS


def test_finding1_a_removed_automations_permission_alerts_without_an_automations_request(tmp_path):
    """The default profile's job alerts on a removed Automations write permission and sends no
    request to any Automations route: no machine-key write request, no key."""
    host = Host(tmp_path)
    assert host.run().kind == "active"
    removed = next(signal for signal in automations_signals() if signal.path.endswith("/resend"))

    def narrowed(method, path):
        if (method, path) == (removed.method, removed.path):
            return SimpleNamespace(method=method, path=path, required_scope="cron.automations.read")
        return host.route_policy(method, path)

    alert = host.run(route_policy=narrowed)
    assert alert.kind == "alert"
    assert "Automations permission wiring" in alert.text
    assert all("/api/" not in request[1] for request in host.requests)


def test_finding1_clear_history_is_probed_as_the_authority_close_call():
    """Clear history is not a DELETE: Start over calls POST .../authority with the action close
    (applyConversationAuthority, from closeConversation), probed with an unknown conversation."""
    probe = clear_history_probe()
    conversation = "raphael-owner-" + "0" * 32
    assert (probe.method, probe.path) == (
        "POST", f"/p/raphael-planner/v1/responses/conversations/{conversation}/authority")
    assert probe.body == {"action": "close"}
    assert (probe.status, probe.code, probe.message) == (400, None, "Invalid owner proposal authority request")
    assert "route 12" in probe.label
    assert all(probe.method != "DELETE" for probe in every_probe())


def test_finding1_clear_history_probe_is_refused_before_any_lookup():
    """The platform's own authority handler refuses the clear history probe at its shape check:
    the response store, where the lookup and the close would run, is never reached."""
    from gateway.platforms.api_server import APIServerAdapter

    class Unreachable:
        def __getattr__(self, name):
            raise AssertionError(f"lookup reached: {name}")

    probe = clear_history_probe()

    async def body():
        return dict(probe.body)

    adapter = SimpleNamespace(_check_auth=lambda request: None, _response_store=Unreachable(),
                              _run_statuses=Unreachable())
    request = SimpleNamespace(json=body, match_info={"conversation": probe.path.split("/")[6]})
    response = asyncio.run(APIServerAdapter._handle_owner_conversation_authority(adapter, request))
    assert watch.classify(probe, response.status, json.loads(response.text)) == watch.PASS


class Listener:
    """A real HTTP server on 127.0.0.1 that answers each probe of ``profile`` as a healthy route, or
    with the ``(status, payload)`` set for its name in ``refusals``."""

    def __init__(self, profile):
        self.requests = []
        self.refusals = {}
        table, seen, refusals = watch.PROBES[profile], self.requests, self.refusals

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"null")
                seen.append((self.command, self.path, self.headers.get("Authorization"), body))
                probe = next((p for p in table if (p.path, p.body) == (self.path, body)), None)
                status, payload = (refusals.get(probe.name) or healthy(probe)) if probe else (418, None)
                data = json.dumps(payload).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *args):
                pass

        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.port = self.server.server_address[1]
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self):
        self.server.shutdown()
        self.server.server_close()


@pytest.fixture
def listener():
    servers = []

    def start(profile):
        servers.append(Listener(profile))
        return servers[-1]

    yield start
    for server in servers:
        server.close()


def root_profile(tmp_path, port):
    """A root profile whose own config sets the shared listener's port, with its own key."""
    root = tmp_path / "root"
    root.mkdir()
    (root / "config.yaml").write_text(
        f"gateway:\n  api_server:\n    enabled: true\n    port: {port}\n", encoding="utf-8")
    (root / ".env").write_text("API_SERVER_KEY=" + "r" * 40 + "\n", encoding="utf-8")
    return root


def assert_reached(server, profile, key):
    table = watch.PROBES[profile]
    assert sorted((m, p) for m, p, _, _ in server.requests) == sorted((pr.method, pr.path) for pr in table)
    assert all(path.startswith(f"/p/{profile}/") for _, path, _, _ in server.requests)
    assert {auth for _, _, auth, _ in server.requests} == {f"Bearer {key}"}


def test_finding2_the_planner_job_reaches_the_listener_only_the_root_profile_configures(
        tmp_path, monkeypatch, listener):
    """One listener serves every profile under /p/PROFILE/: with only the root profile configuring
    the port, the planner job reaches it with its own key and its exact /p/ prefix."""
    monkeypatch.delenv("API_SERVER_PORT", raising=False)
    server = listener("raphael-planner")
    planner = root_profile(tmp_path, server.port) / "profiles" / "raphael-planner"
    planner.mkdir(parents=True)
    (planner / ".env").write_text("API_SERVER_KEY=" + "p" * 40 + "\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(planner))

    outcome = Host(tmp_path, profile="raphael-planner").run(base=None, send=None, read_key=None)

    assert (outcome.code, outcome.kind) == (0, "active"), outcome.text
    assert_reached(server, "raphael-planner", "p" * 40)


def test_finding2_default_profile_control_on_the_same_listener(tmp_path, monkeypatch, listener):
    """The default profile's job reaches the same listener with the default profile's own key."""
    monkeypatch.delenv("API_SERVER_PORT", raising=False)
    server = listener("default")
    monkeypatch.setenv("HERMES_HOME", str(root_profile(tmp_path, server.port)))

    outcome = Host(tmp_path).run(base=None, send=None, read_key=None)

    assert (outcome.code, outcome.kind) == (0, "active"), outcome.text
    assert_reached(server, "default", "r" * 40)


def execution(monkeypatch, job_id, moment, *, finished=True):
    """One real execution of ``job_id`` claimed at ``moment``, a time or the claimed_at text stored as
    given; the current run is left unfinished."""
    from cron import executions

    stamp = SimpleNamespace(isoformat=lambda: moment) if isinstance(moment, str) else moment
    monkeypatch.setattr(executions, "_hermes_now", lambda: stamp)
    row = executions.create_execution(job_id, source="schedule")
    if finished:
        executions.finish_execution(row["id"], success=True)
    return row["id"]


def deliver(execution_id, job_id, text, *states):
    """A real delivery record of ``text`` for one execution, one Telegram chat per state."""
    from cron import delivery_record

    with delivery_record.recording(execution_id, job_id):
        recorder = delivery_record.active_recorder()
        chats = [{"platform": "telegram", "chat_id": f"-10010000000{n}"} for n in range(len(states))]
        recorder.begin(text, [], chats)
        for state in states:
            recorder.next_target()
            if state == "failed":
                recorder.note_failed("platform_refused")
            elif state == "delivered":
                recorder.note_sent(True)
            else:
                recorder.note_unknown("timeout")


def watch_alert(tmp_path, monkeypatch):
    """The watch job's runs up to its alert; returns the host, the alert and the alert's execution."""
    host = Host(tmp_path)
    assert host.run(job_id=WATCH_JOB, delivery_state=None).kind == "active"
    host.refuse(decisions_probe(), 403, "route_not_allowed")
    alert_execution = execution(monkeypatch, WATCH_JOB, host.clock - MINUTE)
    alert = host.run(job_id=WATCH_JOB, delivery_state=None)
    assert alert.kind == "alert"
    return host, alert, alert_execution


def next_watch_run(host, monkeypatch):
    execution(monkeypatch, WATCH_JOB, host.clock - MINUTE, finished=False)
    return host.run(job_id=WATCH_JOB, delivery_state=None)


def test_finding3_a_failed_alert_repeats_though_a_delivered_unrelated_report_quotes_it(
        tmp_path, monkeypatch):
    """An announcement belongs to the execution that printed it, never to a text match in another
    job's output: a failed watch alert followed by a delivered unrelated report quoting it repeats."""
    host, alert, alert_execution = watch_alert(tmp_path, monkeypatch)
    deliver(alert_execution, WATCH_JOB, alert.text, "failed")
    report = execution(monkeypatch, OTHER_JOB, host.clock - 30 * MINUTE)
    deliver(report, OTHER_JOB, "Daily digest. Quoted from the owner chat:\n" + alert.text, "delivered")

    again = next_watch_run(host, monkeypatch)

    assert (again.code, again.kind, again.text) == (0, "repeat", alert.text)


def test_finding3_a_delivered_alert_does_not_repeat_for_a_failed_unrelated_report_quoting_it(
        tmp_path, monkeypatch):
    """A delivered watch alert followed by a failed unrelated report quoting it does not repeat."""
    host, alert, alert_execution = watch_alert(tmp_path, monkeypatch)
    deliver(alert_execution, WATCH_JOB, alert.text, "delivered")
    report = execution(monkeypatch, OTHER_JOB, host.clock - 30 * MINUTE)
    deliver(report, OTHER_JOB, "Weekly summary. Quoted:\n" + alert.text, "failed")

    outcomes = [next_watch_run(host, monkeypatch), next_watch_run(host, monkeypatch)]

    assert [(outcome.code, outcome.text) for outcome in outcomes] == [(0, ""), (0, "")]


def test_finding3_more_than_50_newer_unrelated_executions_do_not_hide_the_watch_delivery(
        tmp_path, monkeypatch):
    """The watch job's own executions are read by its id, newest first, with no fixed window."""
    host, alert, alert_execution = watch_alert(tmp_path, monkeypatch)
    deliver(alert_execution, WATCH_JOB, alert.text, "failed")
    for n in range(60):
        execution(monkeypatch, OTHER_JOB, host.clock - 40 * MINUTE + timedelta(seconds=n))

    again = next_watch_run(host, monkeypatch)

    assert (again.kind, again.text) == ("repeat", alert.text)


def test_finding3_without_the_watch_job_id_no_other_output_is_searched(tmp_path, monkeypatch):
    """Without the watch job's id the delivery reads unknown and nothing repeats: no text search
    of other jobs' output stands in for it."""
    host, alert, alert_execution = watch_alert(tmp_path, monkeypatch)
    deliver(alert_execution, WATCH_JOB, alert.text, "failed")
    report = execution(monkeypatch, OTHER_JOB, host.clock - 30 * MINUTE)
    deliver(report, OTHER_JOB, alert.text, "failed")
    execution(monkeypatch, WATCH_JOB, host.clock - MINUTE, finished=False)

    again = host.run(job_id=None, delivery_state=None)

    assert (again.code, again.text) == (0, "")


@pytest.mark.parametrize("resend,printed_again", [
    ("failed", True), ("delivered", False), ("in_progress", False), ("unknown", False),
])
def test_finding4_a_resend_counts_by_how_its_targets_ended(tmp_path, monkeypatch, resend, printed_again):
    """A resend whose every target ended failed counts as failed, so the hourly repeat goes on. A
    delivered resend settles the announcement. A resend in progress or with an unknown outcome
    stops the repeat."""
    from cron import delivery_record, executions

    host, alert, alert_execution = watch_alert(tmp_path, monkeypatch)
    deliver(alert_execution, WATCH_JOB, alert.text, "failed")
    claim = delivery_record.claim_resend(executions.get_execution(alert_execution), "send-again-1")
    assert claim["claimed"], claim
    if resend != "in_progress":
        reason = "platform_refused" if resend == "failed" else None
        result = {"position": 0, "state": resend, "reason": reason}
        assert delivery_record.finish_resend(claim["attempt"]["attempt_id"], [result])

    again = next_watch_run(host, monkeypatch)

    assert (again.code, again.text) == (0, alert.text if printed_again else "")


# --- Owner note, round 2: regression tests for the two review findings ----------------------------

REPEAT_OFF = "The hourly repeat is off: if this alert is not delivered, it is not sent again."


def planner_job(tmp_path, monkeypatch, listener):
    """The planner's watch job as the scheduler starts it, with a real listener that refuses the
    planner turn route: HERMES_HOME is the planner profile, whose own .env holds its key, and only the
    root profile configures the listener's port. Returns main()'s one argument."""
    monkeypatch.delenv("API_SERVER_PORT", raising=False)
    server = listener("raphael-planner")
    server.refusals["planner_turn"] = (403, envelope("route_not_allowed"))
    planner = root_profile(tmp_path, server.port) / "profiles" / "raphael-planner"
    planner.mkdir(parents=True)
    (planner / ".env").write_text("API_SERVER_KEY=" + "p" * 40 + "\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(planner))
    page_check = tmp_path / "page-check.sh"
    page_check.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    return page_check


def main_run(monkeypatch, capsys, page_check, at):
    """One run of the job through main(): its own execution is claimed a minute before ``at`` and the
    watch's clock reads ``at``. Returns that execution, the exit code and the printed output."""

    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            return at.astimezone(tz)

    own = execution(monkeypatch, WATCH_JOB, at - MINUTE, finished=False)
    monkeypatch.setattr(watch, "datetime", Clock)
    code = watch.main([str(page_check)])
    return own, code, capsys.readouterr().out


def test_finding1_main_with_the_job_id_repeats_a_failed_own_alert_an_hour_later(
        tmp_path, monkeypatch, capsys, listener):
    """With HERMES_CRON_JOB_ID set to the watch job's own id, an alert whose own execution ended with
    its only target failed is printed again by main(), identical, one hour later."""
    page_check = planner_job(tmp_path, monkeypatch, listener)
    monkeypatch.setenv("HERMES_CRON_JOB_ID", WATCH_JOB)

    alert_execution, alert_code, alert = main_run(monkeypatch, capsys, page_check, START)
    deliver(alert_execution, WATCH_JOB, alert.strip(), "failed")
    _, again_code, again = main_run(monkeypatch, capsys, page_check, START + HOUR)

    assert (alert_code, again_code) == (0, 0)
    assert alert.startswith(f"Owner actions check: 1 failing since {watch.format_time(START)}.\n")
    assert REPEAT_OFF not in alert
    assert again == alert


@pytest.mark.parametrize("job_id", [None, ""])
def test_finding1_main_without_the_job_id_says_the_repeat_is_off_and_never_repeats(
        tmp_path, monkeypatch, capsys, listener, job_id):
    """Without HERMES_CRON_JOB_ID, unset or empty, the first alert's output ends with one plain
    sentence that the hourly repeat is off; one hour later, with the same failed delivery, main()
    prints nothing."""
    page_check = planner_job(tmp_path, monkeypatch, listener)
    if job_id is None:
        monkeypatch.delenv("HERMES_CRON_JOB_ID", raising=False)
    else:
        monkeypatch.setenv("HERMES_CRON_JOB_ID", job_id)

    alert_execution, alert_code, alert = main_run(monkeypatch, capsys, page_check, START)
    deliver(alert_execution, WATCH_JOB, alert.strip(), "failed")
    _, again_code, again = main_run(monkeypatch, capsys, page_check, START + HOUR)

    assert (alert_code, again_code) == (0, 0)
    assert alert.startswith(f"Owner actions check: 1 failing since {watch.format_time(START)}.\n")
    assert alert.splitlines()[-1] == REPEAT_OFF
    assert again == ""


def own_run(host, monkeypatch, claimed_at):
    """One hourly run of the watch job a minute after its own execution's claimed_at, stored as the
    text the scheduler writes in its local offset; the watch's clock is UTC, as in production."""
    own = execution(monkeypatch, WATCH_JOB, claimed_at, finished=False)
    host.clock = datetime.fromisoformat(claimed_at).astimezone(timezone.utc) + MINUTE
    return own, host.run(job_id=WATCH_JOB, delivery_state=None)


FAILED_ALERT_DELIVERED_REPEAT = (("alert", "failed"), ("repeat", "delivered"))


@pytest.mark.parametrize("claims,first_two,third", [
    # The review's regression: the daylight saving rollback repeats 02:49:59, so the failed alert's
    # claim sorts above the delivered repeat's by text, though it is an hour older.
    pytest.param(("2026-10-25T02:49:59+02:00", "2026-10-25T02:49:59+01:00", "2026-10-25T03:49:59+01:00"),
                 FAILED_ALERT_DELIVERED_REPEAT, None, id="daylight-saving-rollback"),
    pytest.param(("2026-10-25T00:49:59+00:00", "2026-10-25T01:49:59+00:00", "2026-10-25T02:49:59+00:00"),
                 FAILED_ALERT_DELIVERED_REPEAT, None, id="utc-control"),
    # The server's time zone changed from +05:30 to +01:00, not for daylight saving.
    pytest.param(("2026-10-06T15:19:59+05:30", "2026-10-06T11:49:59+01:00", "2026-10-06T12:49:59+01:00"),
                 FAILED_ALERT_DELIVERED_REPEAT, None, id="time-zone-change"),
    # Reverse: the older own execution, delivered, sorts first by text; the newer holds the failed alert.
    pytest.param(("2026-10-25T02:49:59+02:00", "2026-10-25T02:49:59+01:00", "2026-10-25T03:49:59+01:00"),
                 (("active", "delivered"), ("alert", "failed")), "repeat", id="reverse"),
])
def test_finding2_the_newest_own_execution_by_instant_holds_the_announcement(
        tmp_path, monkeypatch, claims, first_two, third):
    """The watch job's own executions are ordered by the instant of claimed_at in UTC, never by its
    text. The first two hourly runs print and are delivered as ``first_two`` says; one hour after the
    second, the alert prints again only when the second run's own delivery failed."""
    host = Host(tmp_path)
    for claimed_at, (kind, delivery) in zip(claims, first_two):
        if kind == "alert":
            host.refuse(decisions_probe(), 403, "route_not_allowed")
        own, printed = own_run(host, monkeypatch, claimed_at)
        assert printed.kind == kind
        deliver(own, WATCH_JOB, printed.text, delivery)

    _, later = own_run(host, monkeypatch, claims[2])

    # The second run printed the alert, the first time or again.
    assert (later.code, later.kind, later.text) == (0, third, printed.text if third else "")


@pytest.mark.parametrize("alert_claim,other_claim", [
    pytest.param("2026-10-05T08:59:00", None, id="no-utc-offset"),
    # Sorts below every ISO claimed_at by text: only a reader of all own executions meets it.
    pytest.param("2026-10-05T08:59:00+00:00", "1 October 2026", id="cannot-be-parsed"),
    pytest.param("2026-10-05T10:59:00+02:00", "2026-10-05T09:59:00+01:00", id="same-instant-twice"),
])
def test_finding2_an_own_execution_without_one_clear_instant_stops_the_repeat(
        tmp_path, monkeypatch, alert_claim, other_claim):
    """An own claimed_at that cannot be parsed or has no UTC offset, or two own executions sharing the
    newest instant at or before the announcement, read unknown: the alert, though its own delivery
    failed, does not print again."""
    host = Host(tmp_path)
    host.refuse(decisions_probe(), 403, "route_not_allowed")
    alert_execution = execution(monkeypatch, WATCH_JOB, alert_claim)
    alert = host.run(job_id=WATCH_JOB, delivery_state=None)
    deliver(alert_execution, WATCH_JOB, alert.text, "failed")
    if other_claim:
        execution(monkeypatch, WATCH_JOB, other_claim)

    again = next_watch_run(host, monkeypatch)

    assert alert.kind == "alert"
    assert (again.code, again.text) == (0, "")


# --- Round 4: own executions that share a claimed_at across a page boundary -----------------------

# The most executions one call of the store's list_executions returns: one page.
PAGE = 500


def copies(execution_id, claimed_ats):
    """More real executions of the job of the unfinished ``execution_id``, one claimed at each of
    ``claimed_ats``: the row create_execution wrote for it, copied with a new id each into the same
    store in one transaction, as one create_execution call each takes about 30 ms. Returns the ids."""
    ids = [uuid.uuid4().hex for _ in claimed_ats]
    with closing(sqlite3.connect(get_hermes_home().resolve() / "cron" / "executions.db")) as conn, conn:
        conn.executemany(
            "INSERT INTO executions (id, job_id, source, process_id, pid, process_started_at, status,"
            " claimed_at, scheduled_instant) SELECT ?, job_id, source, process_id, pid,"
            " process_started_at, status, ?, scheduled_instant FROM executions WHERE id=?",
            [(new_id, claimed_at, execution_id) for new_id, claimed_at in zip(ids, claimed_ats)],
        )
    return ids


def alert_with_history(tmp_path, monkeypatch, twins, newer):
    """The watch job's alert, printed at START by a run whose own execution was claimed a minute
    before. ``twins`` own executions in all are claimed at that same text, the alert's own among
    them, and ``newer`` own executions are claimed after the alert, the last being the next hourly
    run's own. Returns the host, the alert and the twins' ids, lowest first."""
    host = Host(tmp_path)
    host.refuse(decisions_probe(), 403, "route_not_allowed")
    claimed_at = (START - MINUTE).isoformat()
    tied = [execution(monkeypatch, WATCH_JOB, claimed_at)]
    alert = host.run(job_id=WATCH_JOB, delivery_state=None)
    assert alert.kind == "alert"
    if twins > 1:
        tied.append(execution(monkeypatch, WATCH_JOB, claimed_at))
    later = execution(monkeypatch, WATCH_JOB, host.clock - MINUTE, finished=False)
    tied += copies(later, [claimed_at] * (twins - len(tied)))
    copies(later, [(START + timedelta(seconds=n)).isoformat() for n in range(1, newer)])
    return host, alert, sorted(tied)


def announcement():
    return json.loads(watch.state_path().read_text(encoding="utf-8"))["announced"]


@pytest.mark.parametrize("twins,newer", [
    # The review's case: 499 newer own executions and the higher-id twin fill the first page, and
    # the lower-id twin is the first row after it.
    pytest.param(2, PAGE - 1, id="tie-spanning-the-page-boundary"),
    pytest.param(PAGE + 1, PAGE - 1, id="more-than-a-full-page-tied-across-the-boundary"),
    pytest.param(2, 1, id="tie-wholly-inside-one-page"),
])
@pytest.mark.parametrize("higher,lower", [("failed", "delivered"), ("delivered", "failed")])
def test_round4_own_executions_tied_at_the_newest_instant_read_unknown_and_stay_quiet(
        tmp_path, monkeypatch, twins, newer, higher, lower):
    """Own executions tied at the newest eligible claimed_at read unknown wherever the page boundary
    falls, in both delivery directions (higher-id eligible row failed and lower-id row delivered; then
    reversed), and the next hourly decision is quiet: the tie that spans the page boundary, more than
    a full page of tied rows, and the same tie wholly inside one page."""
    host, alert, tied = alert_with_history(tmp_path, monkeypatch, twins, newer)
    deliver(tied[-1], WATCH_JOB, alert.text, higher)
    deliver(tied[0], WATCH_JOB, alert.text, lower)

    reader = watch._previous_delivery(announcement(), WATCH_JOB)
    again = host.run(job_id=WATCH_JOB, delivery_state=None)

    assert (reader, again.code, again.kind, again.text) == (watch.UNKNOWN, 0, None, "")


@pytest.mark.parametrize("delivery,kind", [("failed", "repeat"), ("delivered", None)])
def test_round4_more_than_a_full_page_with_a_unique_newest_instant_keeps_its_outcome(
        tmp_path, monkeypatch, delivery, kind):
    """More than one full page of newer own executions and a unique newest eligible instant give
    that execution's correct definite outcome: failed prints the alert again an hour later,
    delivered prints nothing."""
    host, alert, (own,) = alert_with_history(tmp_path, monkeypatch, 1, PAGE)
    deliver(own, WATCH_JOB, alert.text, delivery)

    reader = watch._previous_delivery(announcement(), WATCH_JOB)
    again = host.run(job_id=WATCH_JOB, delivery_state=None)

    assert (reader, again.code, again.kind, again.text) == (delivery, 0, kind, alert.text if kind else "")


def test_round4_the_reader_creates_and_changes_no_store(monkeypatch):
    """The reader only reads: without the stores it reads unknown and creates neither, and with them
    it reads the failed delivery and leaves both store files byte for byte as they were."""
    cron = get_hermes_home().resolve() / "cron"
    stores = [cron / "executions.db", cron / "delivery_records.db"]
    announced = {"text": "Owner actions check: 1 failing.", "at": START.isoformat()}

    assert watch._previous_delivery(announced, WATCH_JOB) == watch.UNKNOWN
    assert [store.exists() for store in stores] == [False, False]
    own = execution(monkeypatch, WATCH_JOB, START - MINUTE)
    assert watch._previous_delivery(announced, WATCH_JOB) == watch.UNKNOWN
    assert [store.exists() for store in stores] == [True, False]
    deliver(own, WATCH_JOB, announced["text"], "failed")
    before = [(store.read_bytes(), store.stat().st_mtime_ns) for store in stores]

    assert watch._previous_delivery(announced, WATCH_JOB) == "failed"
    assert [(store.read_bytes(), store.stat().st_mtime_ns) for store in stores] == before


# --- Round 5: the owner rule for the listener address ---------------------------------------------

OWN_KEYS = {"default": "r" * 40, "raphael-planner": "p" * 40}
# The root profile's own listener section in its config.yaml; ``{port}`` is the test listener's port.
API_SERVER = "gateway:\n  api_server:\n    enabled: true\n    port: {port}\n"


def listener_job(tmp_path, monkeypatch, server, profile, config, env=None):
    """One run of ``profile``'s watch job with its own key from its own .env, ``config`` as the root
    profile's config.yaml and ``env`` in the job's environment, ``{port}`` in either being the port of
    the real test listener ``server`` on 127.0.0.1. Returns the outcome and every URL the job tried: a
    URL outside ``server`` is refused here and never sent, so no run reaches anything else."""
    root = tmp_path / "root"
    home = root if profile == "default" else root / "profiles" / profile
    home.mkdir(parents=True)
    (root / "config.yaml").write_text(config.format(port=server.port), encoding="utf-8")
    (home / ".env").write_text(f"API_SERVER_KEY={OWN_KEYS[profile]}\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))
    for name, value in (env or {}).items():
        monkeypatch.setenv(name, value.format(port=server.port))
    tried = []

    def send(method, url, body, key):
        tried.append(url)
        if not url.startswith(f"http://127.0.0.1:{server.port}/"):
            raise ConnectionRefusedError("not the test listener")
        return watch._http_send(method, url, body, key)

    return Host(tmp_path, profile=profile).run(base=None, send=send, read_key=None), tried


@pytest.mark.parametrize("host", ["", "    host: 0.0.0.0\n"], ids=["no-host", "host-0.0.0.0"])
@pytest.mark.parametrize("profile", ["default", "raphael-planner"])
def test_round5_1_both_serving_profiles_in_the_supported_shape_reach_the_listener(
        tmp_path, monkeypatch, listener, profile, host):
    """Both serving profiles in the supported shape reach a real local test listener with their own
    key and exact /p/ prefix: the root profile's own listener section gives the port and no host, or
    the host 0.0.0.0; config.yaml has no top-level api_server section; and the job's environment has
    no API_SERVER_HOST and no API_SERVER_PORT."""
    server = listener(profile)

    outcome, _ = listener_job(tmp_path, monkeypatch, server, profile, API_SERVER + host)

    assert (outcome.code, outcome.kind) == (0, "active"), outcome.text
    assert_reached(server, profile, OWN_KEYS[profile])


@pytest.mark.parametrize("config,env,shape", [
    pytest.param(API_SERVER + "    host: 127.0.0.2\n", None, "the host '127.0.0.2'", id="host-127.0.0.2"),
    pytest.param(API_SERVER + "    host: '::1'\n", None, "the host '::1'", id="host-ipv6-loopback"),
    # The review's variant: the port only in the top-level api_server form.
    pytest.param("api_server:\n  enabled: true\n  port: {port}\n", None, "a top-level api_server section",
                 id="top-level-api_server-section"),
    # The review's variant: a configured port, and the listener's port in API_SERVER_PORT.
    pytest.param(API_SERVER.replace("{port}", "1"), {"API_SERVER_PORT": "{port}"},
                 "API_SERVER_PORT in this job's environment", id="API_SERVER_PORT-set"),
    pytest.param(API_SERVER, {"API_SERVER_HOST": "127.0.0.2"}, "API_SERVER_HOST in this job's environment",
                 id="API_SERVER_HOST-set"),
    pytest.param(API_SERVER.replace("{port}", "0"), None, "the port 0", id="port-0"),
])
@pytest.mark.parametrize("profile", ["default", "raphael-planner"])
def test_round5_2_every_other_shape_cannot_run_and_the_listener_gets_no_request(
        tmp_path, monkeypatch, capsys, listener, profile, config, env, shape):
    """Every other shape stops the run before the first request, with the cannot-run outcome of a
    missing key and the reason "listener address unknown" naming the shape: the test listener receives
    zero requests, and nothing is sent anywhere else."""
    server = listener(profile)

    outcome, tried = listener_job(tmp_path, monkeypatch, server, profile, config, env)

    assert outcome == watch.Outcome(1)
    assert f"checks not run: listener address unknown: {shape}" in capsys.readouterr().err
    assert (server.requests, tried) == ([], [])


def test_round5_3_8642_and_127_0_0_1_are_the_api_servers_defaults():
    """8642 and 127.0.0.1, the port of a root listener section that gives none and the address every
    request goes to, equal DEFAULT_PORT and DEFAULT_HOST in gateway/platforms/api_server.py."""
    from gateway.platforms import api_server

    assert (watch.DEFAULT_PORT, watch.DEFAULT_HOST) == (8642, "127.0.0.1") == (
        api_server.DEFAULT_PORT, api_server.DEFAULT_HOST)
    assert watch._listener_base() == "http://127.0.0.1:8642"


JOURNEYS_LABEL = "Owner journeys run"
STUCK_DETAIL = "stuck since 2026-10-05 06:00 UTC; the time limit ended it"


def write_journeys(content):
    path = get_hermes_home() / "journeys" / "result"
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(content, bytes):
        path.write_bytes(content)
    else:
        path.write_text(content, encoding="utf-8")


def started_at(moment):
    stamp = moment.strftime("%Y-%m-%dT%H:%M:%SZ")
    return f"result=started\nstep=none\nreason=the run has started and has not ended\nstarted={stamp}\n"


def finished(outcome):
    return (f"result={outcome}\nstep=none\nreason=done\nstarted=2026-10-05T07:00:00Z\n"
            "finished=2026-10-05T07:20:00Z\n")


def journeys_check():
    checks = json.loads(watch.report_path().read_text(encoding="utf-8"))["checks"]
    found = [check for check in checks if check["name"] == "journeys"]
    assert len(found) == 1
    return found[0]


def test_journeys_no_file_passes(tmp_path):
    """No result file means no run has started yet: pass."""
    host = Host(tmp_path)
    outcome = host.run()
    check = journeys_check()
    assert (check["outcome"], check["label"], check["method"]) == (watch.PASS, JOURNEYS_LABEL, "read")
    assert outcome.kind == "active"


def test_journeys_started_more_than_55_minutes_ago_fails_with_the_started_time(tmp_path):
    """A run still marked started more than 55 minutes before the run's moment fails."""
    host = Host(tmp_path)
    write_journeys(started_at(START - timedelta(hours=3)))
    outcome = host.run()
    check = journeys_check()
    assert (check["outcome"], check["detail"]) == (watch.FAIL, STUCK_DETAIL)
    assert outcome.kind == "alert"
    assert f"- {JOURNEYS_LABEL}: {STUCK_DETAIL}" in outcome.text


def test_journeys_the_stuck_limit_is_55_minutes_exactly(tmp_path):
    """130 minutes is still inside the limit plus ten minutes; one second more is stuck."""
    assert watch.JOURNEYS_STUCK_AFTER == timedelta(minutes=130)
    host = Host(tmp_path)
    write_journeys(started_at(START - timedelta(minutes=130)))
    host.run()
    assert journeys_check()["outcome"] == watch.PASS
    host = Host(tmp_path)
    write_journeys(started_at(START - timedelta(minutes=130, seconds=1)))
    host.run()
    assert journeys_check()["outcome"] == watch.FAIL


@pytest.mark.parametrize("age", [timedelta(0), timedelta(minutes=20), timedelta(minutes=-5)])
def test_journeys_started_within_55_minutes_passes(tmp_path, age):
    host = Host(tmp_path)
    write_journeys(started_at(START - age))
    outcome = host.run()
    assert journeys_check()["outcome"] == watch.PASS
    assert outcome.kind == "active"


@pytest.mark.parametrize("final", ["passed", "failed"])
def test_journeys_a_final_result_passes_however_old(tmp_path, final):
    """A passed or failed run told the owner itself: pass."""
    host = Host(tmp_path)
    write_journeys(finished(final))
    outcome = host.run()
    assert journeys_check()["outcome"] == watch.PASS
    assert outcome.kind == "active"


UNREADABLE = object()


@pytest.mark.parametrize("content", [
    UNREADABLE,
    b"result=started\nstarted=2026-10-05T07:00:00Z\n\xff\xfe\n",
    "",
    "result=started\nthis is not a pair\nstarted=2026-10-05T07:00:00Z\n",
    "result=started\n\nstarted=2026-10-05T07:00:00Z\n",
    "result=started\nstep=none\n",
    "result=started\nstarted=\n",
    "result=started\nstarted=yesterday\n",
    "result=started\nstarted=2026-10-05 07:00:00\n",
    "result=started\nstarted=2026-10-05T07:00:00+02:00\n",
    "result=started\nstarted=2026-10-05T07:00:00\n",
    "result=started\nstarted=2026-13-05T07:00:00Z\n",
    "step=none\nstarted=2026-10-05T07:00:00Z\n",
    "result=unheard\nstarted=2026-10-05T07:00:00Z\n",
])
def test_journeys_an_unusable_file_gives_no_verdict(tmp_path, content):
    host = Host(tmp_path)
    if content is UNREADABLE:
        (get_hermes_home() / "journeys" / "result").mkdir(parents=True)
    else:
        write_journeys(content)
    outcome = host.run()
    assert journeys_check()["outcome"] == watch.UNKNOWN
    assert outcome.kind is None


def test_journeys_no_verdict_keeps_the_previous_state(tmp_path):
    """After a stuck alert, an unreadable file neither recovers nor re-alerts; once it is
    readable and final, the recovery follows."""
    host = Host(tmp_path)
    write_journeys(started_at(START - timedelta(hours=3)))
    assert host.run().kind == "alert"
    write_journeys("not a pair\n")
    outcome = host.run()
    assert (journeys_check()["outcome"], outcome.kind, outcome.text) == (watch.UNKNOWN, None, "")
    write_journeys(finished("passed"))
    assert host.run().kind == "recovery"


@pytest.mark.parametrize("final", ["passed", "failed"])
@pytest.mark.parametrize("stamp", [
    "yesterday",
    "",
    "2026-10-05 07:00:00",
    "2026-10-05T07:00:00",
    "2026-10-05T07:00:00+02:00",
    "2026-13-05T07:00:00Z",
    "2026-02-30T07:00:00Z",
    "2026-10-05T25:00:00Z",
])
def test_journeys_a_final_result_with_an_invalid_started_time_keeps_the_stuck_alert(tmp_path, final, stamp):
    """A final result whose started time is not a real UTC time gives no verdict: no recovery,
    no message, and the stuck state is kept until a valid result arrives."""
    host = Host(tmp_path)
    write_journeys(started_at(START - timedelta(hours=3)))
    assert host.run().kind == "alert"
    kept = lambda: {k: v for k, v in json.loads(watch.state_path().read_text(encoding="utf-8")).items()
                    if k in ("state", "failing", "since")}
    stored = kept()
    assert stored["failing"] == ["journeys"]
    write_journeys(finished(final).replace("started=2026-10-05T07:00:00Z", f"started={stamp}"))
    outcome = host.run()
    assert (journeys_check()["outcome"], outcome.kind, outcome.text) == (watch.UNKNOWN, None, "")
    assert kept() == stored
    write_journeys(finished(final))
    assert host.run().kind == "recovery"


@pytest.mark.parametrize("final", ["passed", "failed"])
def test_journeys_a_final_result_without_a_started_time_gives_no_verdict(tmp_path, final):
    host = Host(tmp_path)
    write_journeys(f"result={final}\nstep=none\n")
    outcome = host.run()
    assert (journeys_check()["outcome"], outcome.kind) == (watch.UNKNOWN, None)


def test_journeys_a_run_stuck_across_two_hourly_runs_alerts_once(tmp_path):
    host = Host(tmp_path)
    write_journeys(started_at(START - timedelta(hours=3)))
    outcomes = [host.run(), host.run(), host.run()]
    assert [outcome.code for outcome in outcomes] == [0, 0, 0]
    sent = messages(outcomes)
    assert len(sent) == 1
    assert outcomes[0].kind == "alert"
    assert STUCK_DETAIL in sent[0]


def test_journeys_a_final_result_after_a_stuck_one_gives_the_recovery_message(tmp_path):
    host = Host(tmp_path)
    write_journeys(started_at(START - timedelta(hours=3)))
    host.run()
    write_journeys(finished("failed"))
    after = [host.run(), host.run()]
    assert after[0].kind == "recovery"
    assert watch.format_duration(HOUR) in after[0].text
    assert len(messages(after)) == 1


def test_journeys_the_planning_profile_has_no_journeys_result(tmp_path):
    host = Host(tmp_path, "raphael-planner")
    write_journeys(started_at(START - timedelta(hours=2)))
    outcome = host.run()
    checks = json.loads(watch.report_path().read_text(encoding="utf-8"))["checks"]
    assert "journeys" not in [check["name"] for check in checks]
    assert outcome.kind == "active"


def real_page_run(tmp_path, body, shebang="#!/bin/sh\n", **overrides):
    """One hourly run with the real _run_page_check on a script the test writes."""
    host = Host(tmp_path)
    host.page_check.write_text(shebang + body, encoding="utf-8")
    host.page_check.chmod(0o755)
    outcome = host.run(run_page_check=None, **overrides)
    checks = json.loads(watch.report_path().read_text(encoding="utf-8"))["checks"]
    return host, outcome, next(check for check in checks if check["name"] == "page_check")


@pytest.mark.linux_only
def test_page_lines_two_failures_and_the_count_line_reach_the_report(tmp_path):
    lines = ["FAIL automations 390 no_overflow", "FAIL models 1280 console_clean",
             "page check failed: 2 failure(s)"]
    printed = "".join(f"echo '{line}'\n" for line in lines)
    host, outcome, entry = real_page_run(tmp_path, printed + "exit 1\n")
    assert entry["outcome"] == watch.FAIL
    assert entry["detail"] == "did not pass"
    assert entry["page_lines"] == {"lines": lines, "not_kept": 0}
    watch.state_path().unlink()
    host.clock = START
    injected = host.run(run_page_check=lambda path: watch.FAIL)
    assert (outcome.kind, outcome.text) == (injected.kind, injected.text)


@pytest.mark.linux_only
def test_page_lines_a_line_off_the_fixed_form_is_counted_and_not_kept(tmp_path):
    body = ("echo 'FAIL home 390 no_overflow'\n"
            "echo 'FAIL http://127.0.0.1:8642/x'\n"
            "echo 'FAIL token=abc123'\n"
            f"echo 'FAIL {'a' * 300}'\n"
            "echo 'the page said hello'\n"
            "exit 1\n")
    _, _, entry = real_page_run(tmp_path, body)
    assert entry["page_lines"] == {"lines": ["FAIL home 390 no_overflow"], "not_kept": 4}
    assert "127.0.0.1" not in json.dumps(entry) and "abc123" not in json.dumps(entry)


@pytest.mark.linux_only
def test_page_lines_keep_at_most_twenty(tmp_path):
    body = "i=0\nwhile [ $i -lt 30 ]; do echo \"FAIL page_$i 390 check\"; i=$((i+1)); done\nexit 1\n"
    _, _, entry = real_page_run(tmp_path, body)
    assert entry["page_lines"]["lines"] == [f"FAIL page_{i} 390 check" for i in range(20)]
    assert entry["page_lines"]["not_kept"] == 10


def on_page_check_reads(monkeypatch, handler):
    """Routes the reads that hermes_cli.owner_write_watch makes through handler(real_read, fd, size)."""
    real_read = os.read

    def read(fd, size):
        if sys._getframe(1).f_globals["__name__"] == watch.__name__:
            return handler(real_read, fd, size)
        return real_read(fd, size)

    monkeypatch.setattr(watch.os, "read", read)


@pytest.mark.linux_only
def test_page_lines_output_past_the_bound_is_read_and_dropped_and_the_exit_status_stands(tmp_path, monkeypatch):
    marker = tmp_path / "reached_the_end"
    chunks = []

    def recording_read(real_read, fd, size):
        data = real_read(fd, size)
        chunks.append(data)
        return data

    on_page_check_reads(monkeypatch, recording_read)
    # A limit with less than 150 seconds leaves no room for the second try, so one stream is read.
    monkeypatch.setattr(watch, "PAGE_CHECK_TIMEOUT_SECONDS", watch.PAGE_CHECK_RETRY_WAIT_SECONDS + watch.PAGE_CHECK_RETRY_MIN_SECONDS - 1)
    body = f"i=0\nwhile [ $i -lt 1500 ]; do echo \"FAIL page_$i 390 check\"; i=$((i+1)); done\ntouch {marker}\nexit 1\n"
    _, _, entry = real_page_run(tmp_path, body)
    written = "".join(f"FAIL page_{i} 390 check\n" for i in range(1500)).encode()
    assert len(written) > 16000
    assert b"".join(chunks) == written
    first = written[:16000]
    complete = first.count(b"\n")
    assert entry["page_lines"] == {"lines": [f"FAIL page_{i} 390 check" for i in range(20)],
                                   "not_kept": complete - 20 + (0 if first.endswith(b"\n") else 1)}
    assert marker.exists()
    assert entry["outcome"] == watch.FAIL


@pytest.mark.linux_only
def test_page_lines_output_past_the_bound_under_a_file_size_limit_reaches_the_end(tmp_path):
    marker = tmp_path / "reached_the_end"
    body = ("import resource\n"
            "resource.setrlimit(resource.RLIMIT_FSIZE, (4096, 4096))\n"
            "for i in range(1500):\n"
            "    print(f'FAIL page_{i} 390 check')\n"
            f"open({str(marker)!r}, 'w').close()\n"
            "raise SystemExit(1)\n")
    _, _, entry = real_page_run(tmp_path, body, shebang=f"#!{sys.executable}\n")
    assert marker.exists()
    assert entry["outcome"] == watch.FAIL
    assert len(entry["page_lines"]["lines"]) == 20


@pytest.mark.linux_only
def test_page_lines_standard_error_is_not_kept(tmp_path):
    _, _, entry = real_page_run(tmp_path, "echo 'FAIL home 390 check' >&2\nexit 1\n")
    assert entry["outcome"] == watch.FAIL
    assert entry["page_lines"] == {"lines": [], "not_kept": 0}


@pytest.mark.linux_only
def test_page_lines_a_silent_pass_holds_no_line(tmp_path):
    _, _, entry = real_page_run(tmp_path, "exit 0\n")
    assert entry["outcome"] == watch.PASS
    assert entry["page_lines"] == {"lines": [], "not_kept": 0}


@pytest.mark.parametrize("rest", ["x" * 3000, "x" * 110 + "=secret"])
@pytest.mark.linux_only
def test_page_lines_a_line_cut_by_the_bound_is_not_kept_and_is_counted(tmp_path, rest):
    body = f"echo '{'a' * 15884}'\necho 'FAIL {rest}'\nexit 1\n"
    _, _, entry = real_page_run(tmp_path, body)
    assert entry["page_lines"] == {"lines": [], "not_kept": 2}


@pytest.mark.parametrize("before", ["", "echo 'FAIL home 390 check'\n"])
@pytest.mark.linux_only
def test_page_lines_a_fragment_with_no_end_of_line_is_not_kept_and_is_counted(tmp_path, before):
    _, _, entry = real_page_run(tmp_path, before + "printf 'FAIL home 390'\nexit 1\n")
    assert entry["outcome"] == watch.FAIL
    assert entry["page_lines"] == {"lines": ["FAIL home 390 check"] if before else [], "not_kept": 1}


@pytest.mark.parametrize("tail, not_kept", [("", 0), ("printf 'FAIL home 390'\n", 1)])
@pytest.mark.linux_only
def test_page_lines_a_descendant_holding_the_output_is_not_waited_for(tmp_path, tail, not_kept):
    release = tmp_path / "release"
    pid_file = tmp_path / "descendant.pid"
    watchdog_fired = threading.Event()
    # The descendant keeps the standard output open until the test creates the release file.
    body = ("echo 'FAIL home 390 check'\n" + tail +
            f"(i=0; while [ ! -e {release} ] && [ $i -lt 1200 ]; do i=$((i+1)); sleep 0.05; done; echo ' token=x') &\n"
            f"echo $! > {pid_file}\nexit 1\n")

    def fire():
        watchdog_fired.set()
        release.touch()

    watchdog = threading.Timer(30, fire)
    watchdog.start()
    try:
        _, _, entry = real_page_run(tmp_path, body)
        assert not watchdog_fired.is_set()
        assert not release.exists()
        assert psutil.pid_exists(int(pid_file.read_text(encoding="utf-8")))
        assert entry["outcome"] == watch.FAIL
        assert entry["page_lines"] == {"lines": ["FAIL home 390 check"], "not_kept": not_kept}
    finally:
        watchdog.cancel()
        release.touch()


@pytest.mark.linux_only
def test_page_lines_a_timeout_stays_unknown_and_keeps_the_complete_lines(tmp_path, monkeypatch):
    monkeypatch.setattr(watch, "PAGE_CHECK_TIMEOUT_SECONDS", 2)
    body = "echo 'FAIL home 390 check'\nprintf 'FAIL half'\nexec sleep 60\n"
    _, _, entry = real_page_run(tmp_path, body)
    assert entry["outcome"] == watch.UNKNOWN
    assert entry["page_lines"] == {"lines": ["FAIL home 390 check"], "not_kept": 1}


@pytest.mark.linux_only
def test_page_lines_no_pipe_runs_the_script_once_and_the_report_is_saved(tmp_path, monkeypatch):
    real_pipe = watch.os.pipe
    refused = []

    def stdout_pipe_refused():
        if sys._getframe(1).f_code.co_name == "_get_handles":  # the stdout pipe of each try; the exec error pipe stays
            refused.append(True)
            raise OSError("no pipe")
        return real_pipe()

    runs = tmp_path / "runs"
    monkeypatch.setattr(watch.os, "pipe", stdout_pipe_refused)
    host, outcome, entry = real_page_run(tmp_path, f"echo run >> {runs}\necho 'FAIL home 390 check'\nexit 1\n")
    assert len(refused) == 2
    assert runs.read_text(encoding="utf-8").count("run") == 2
    assert entry["outcome"] == watch.FAIL
    assert "page_lines" not in entry
    assert [try_["page_lines"] for try_ in entry["page_tries"]] == [None, None]
    assert entry["detail"] == "did not pass"
    watch.state_path().unlink()
    host.clock = START
    injected = host.run(run_page_check=lambda path: watch.FAIL)
    assert (outcome.kind, outcome.text) == (injected.kind, injected.text)


@pytest.mark.parametrize("status, expected", [(0, watch.PASS), (1, watch.FAIL)])
@pytest.mark.linux_only
def test_page_lines_a_read_error_keeps_the_outcome_of_the_script(tmp_path, monkeypatch, status, expected):
    def failing_read(real_read, fd, size):
        raise OSError("read failed")

    runs = tmp_path / "runs"
    on_page_check_reads(monkeypatch, failing_read)
    _, _, entry = real_page_run(tmp_path, f"echo run >> {runs}\necho 'FAIL home 390 check'\nexit {status}\n")
    assert runs.read_text(encoding="utf-8").count("run") == (2 if status else 1)
    assert entry["outcome"] == expected
    assert "page_lines" not in entry


@contextmanager
def hang_watchdog(pid_file, seconds=30):
    """Only turns a hang into a failed test: after `seconds` it kills the script and sets the event."""
    fired = threading.Event()

    def fire():
        fired.set()
        try:
            os.kill(int(pid_file.read_text(encoding="utf-8")), getattr(signal, "SIGKILL", signal.SIGTERM))
        except (OSError, ValueError):
            pass

    watchdog = threading.Timer(seconds, fire)
    watchdog.start()
    try:
        yield fired
    finally:
        watchdog.cancel()


def many_failure_lines_run(tmp_path, monkeypatch):
    """Real run of a script that fills far more than a pipe buffer with valid lines and exits 7."""
    monkeypatch.setattr(watch, "PAGE_CHECK_TIMEOUT_SECONDS", 5)
    pid_file = tmp_path / "script.pid"
    body = (f"echo $$ > {pid_file}\ni=0\nwhile [ $i -lt 6000 ]; do echo \"FAIL page_$i 390 check\"; i=$((i+1)); done\n"
            "exit 7\n")
    with hang_watchdog(pid_file) as fired:
        started = time.monotonic()
        result = real_page_run(tmp_path, body)
        elapsed = time.monotonic() - started
    assert not fired.is_set()
    assert elapsed < watch.PAGE_CHECK_TIMEOUT_SECONDS
    return result


@pytest.mark.linux_only
def test_page_lines_a_read_error_with_a_chatty_script_still_fails_before_the_timeout(tmp_path, monkeypatch):
    failed = []

    def first_read_fails(real_read, fd, size):
        if not failed:
            failed.append(True)
            raise OSError("read failed")
        return real_read(fd, size)

    on_page_check_reads(monkeypatch, first_read_fails)
    host, outcome, entry = many_failure_lines_run(tmp_path, monkeypatch)
    assert failed
    assert entry["outcome"] == watch.FAIL
    assert entry["detail"] == "did not pass"
    assert "page_lines" not in entry
    watch.state_path().unlink()
    host.clock = START
    injected = host.run(run_page_check=lambda path: watch.FAIL)
    assert (outcome.kind, outcome.text) == (injected.kind, injected.text)


@pytest.mark.linux_only
def test_page_lines_a_pipe_on_a_descriptor_above_1024_keeps_its_lines(tmp_path, monkeypatch):
    import resource

    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    if hard != resource.RLIM_INFINITY and hard < 1300:
        pytest.skip("the hard limit of open files is below 1300")
    resource.setrlimit(resource.RLIMIT_NOFILE, (max(soft, 1300), hard))
    held = []
    try:
        while True:  # fill the descriptor numbers up to 1100 so that the next pipe is created above 1,024
            held.append(os.open(os.devnull, os.O_RDONLY))
            if held[-1] >= 1100:
                break
        os.close(held.pop())
        _, _, entry = many_failure_lines_run(tmp_path, monkeypatch)
    finally:
        for fd in held:
            os.close(fd)
        resource.setrlimit(resource.RLIMIT_NOFILE, (soft, hard))
    assert entry["outcome"] == watch.FAIL
    assert entry["page_lines"]["lines"] == [f"FAIL page_{i} 390 check" for i in range(20)]


LAST_FAILURE_FILE = "owner_write_watch_last_failure.json"
FAILED_LINES = {"lines": ["FAIL home 390 check"], "not_kept": 0}
PASSED_LINES = {"lines": [], "not_kept": 0}


class FakeTime:
    """Stands in for the time module of the watch: sleeping moves the clock and takes no real time."""

    def __init__(self):
        self.now = 1000.0
        self.slept = []
        self.late = 0.0

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        self.slept.append(seconds)
        self.now += seconds + self.late


@pytest.fixture
def fake_time(monkeypatch):
    fake = FakeTime()
    monkeypatch.setattr(watch, "time", fake)
    return fake


class PageTries:
    """A page check that answers one prepared result for each call; ``spend`` seconds pass during the first call."""

    def __init__(self, fake_time, *results, spend=0):
        self.fake_time, self.results, self.spend, self.calls = fake_time, list(results), spend, []

    def __call__(self, path):
        self.calls.append(path)
        if len(self.calls) == 1:
            self.fake_time.now += self.spend
        return self.results[len(self.calls) - 1]


def last_failure_path():
    return watch.report_path().with_name(LAST_FAILURE_FILE)


def page_entry(path):
    return next(check for check in json.loads(path.read_text(encoding="utf-8"))["checks"] if check["name"] == "page_check")


def recorded_run(host, fake_time, *results, spend=0):
    page = PageTries(fake_time, *results, spend=spend)
    outcome = host.run(run_page_check=page)
    return page, outcome


def test_retry_a_failed_first_try_and_a_passed_second_try_pass_and_keep_both_tries(tmp_path, fake_time):
    host = Host(tmp_path)
    host.run()
    host.run()
    state_before = json.loads(watch.state_path().read_text(encoding="utf-8"))
    page, outcome = recorded_run(host, fake_time, (watch.FAIL, FAILED_LINES), (watch.PASS, PASSED_LINES))
    assert len(page.calls) == 2
    assert fake_time.slept == [30]
    assert outcome.code == 0 and outcome.text == ""
    entry = page_entry(watch.report_path())
    assert entry["outcome"] == watch.PASS
    assert entry["page_lines"] == PASSED_LINES
    assert entry["page_tries"] == [{"outcome": watch.FAIL, "page_lines": FAILED_LINES},
                                   {"outcome": watch.PASS, "page_lines": PASSED_LINES}]
    assert last_failure_path().read_bytes() == watch.report_path().read_bytes()
    state_after = json.loads(watch.state_path().read_text(encoding="utf-8"))
    assert state_after.pop("last_run") != state_before.pop("last_run")
    assert state_after == state_before


def test_retry_two_failed_tries_fail_with_the_message_of_today_and_keep_the_report(tmp_path, fake_time):
    host = Host(tmp_path)
    other = {"lines": ["FAIL models 1280 console_clean"], "not_kept": 0}
    page, outcome = recorded_run(host, fake_time, (watch.FAIL, FAILED_LINES), (watch.FAIL, other))
    assert len(page.calls) == 2
    assert outcome.kind == "alert"
    assert "Page check of the Automations, Connections and Models pages (routes 13 to 17): did not pass" in outcome.text
    assert "FAIL" not in outcome.text.replace("failing", "")
    entry = page_entry(last_failure_path())
    assert entry["outcome"] == watch.FAIL
    assert entry["page_lines"] == other
    assert [try_["outcome"] for try_ in entry["page_tries"]] == [watch.FAIL, watch.FAIL]
    assert last_failure_path().read_bytes() == watch.report_path().read_bytes()
    assert host.key not in last_failure_path().read_text(encoding="utf-8")
    watch.state_path().unlink()
    host.clock = START
    injected = host.run(run_page_check=lambda path: watch.FAIL)
    assert (outcome.kind, outcome.text) == (injected.kind, injected.text)


def test_retry_a_passed_first_try_runs_the_script_once_and_writes_no_second_file(tmp_path, fake_time):
    host = Host(tmp_path)
    host.run()
    page, outcome = recorded_run(host, fake_time, (watch.PASS, PASSED_LINES))
    assert len(page.calls) == 1
    assert fake_time.slept == []
    assert outcome.text == ""
    assert page_entry(watch.report_path())["page_tries"] == [{"outcome": watch.PASS, "page_lines": PASSED_LINES}]
    assert not last_failure_path().exists()


@pytest.mark.parametrize("spent, second_try", [(450, True), (451, False)])
def test_retry_the_second_try_starts_only_with_120_seconds_left(tmp_path, fake_time, monkeypatch, spent, second_try):
    limits = []

    def page_check(path, deadline):
        limits.append(deadline - fake_time.now)
        if len(limits) == 1:
            fake_time.now += spent
        return watch.FAIL, FAILED_LINES

    monkeypatch.setattr(watch, "_run_page_check", page_check)
    host = Host(tmp_path)
    outcome = host.run(run_page_check=None)
    assert limits == ([600, 120] if second_try else [600])
    assert fake_time.slept == ([30] if second_try else [])
    assert outcome.kind == "alert"
    assert len(page_entry(watch.report_path())["page_tries"]) == (2 if second_try else 1)
    assert last_failure_path().exists()


def test_retry_the_second_try_gets_the_time_that_is_left(tmp_path, fake_time, monkeypatch):
    limits = []

    def page_check(path, deadline):
        limits.append(deadline - fake_time.now)
        fake_time.now += 100
        return (watch.FAIL, None) if len(limits) == 1 else (watch.PASS, None)

    monkeypatch.setattr(watch, "_run_page_check", page_check)
    Host(tmp_path).run(run_page_check=None)
    assert limits == [600, 470]


@pytest.mark.parametrize("first", [(watch.UNKNOWN, FAILED_LINES), watch.UNKNOWN])
def test_retry_a_first_try_with_the_outcome_unknown_gets_no_second_try(tmp_path, fake_time, first):
    host = Host(tmp_path)
    page, outcome = recorded_run(host, fake_time, first)
    assert len(page.calls) == 1
    assert fake_time.slept == []
    assert page_entry(watch.report_path())["outcome"] == watch.UNKNOWN
    assert len(page_entry(watch.report_path())["page_tries"]) == 1
    assert not last_failure_path().exists()


def test_last_failure_a_run_with_a_failed_probe_writes_the_second_file(tmp_path, fake_time):
    host = Host(tmp_path)
    decisions = decisions_probe()
    host.refuse(decisions, 403, "route_not_allowed")
    page, outcome = recorded_run(host, fake_time, (watch.PASS, PASSED_LINES))
    assert len(page.calls) == 1
    assert outcome.kind == "alert"
    assert last_failure_path().read_bytes() == watch.report_path().read_bytes()
    checks = json.loads(last_failure_path().read_text(encoding="utf-8"))["checks"]
    assert next(check for check in checks if check["name"] == decisions.name)["outcome"] == watch.FAIL


def test_last_failure_a_run_in_which_each_check_passed_leaves_an_earlier_second_file(tmp_path, fake_time):
    host = Host(tmp_path)
    host.run()
    earlier = '{"earlier": "failed run"}\n'
    last_failure_path().write_text(earlier, encoding="utf-8")
    before = last_failure_path().stat().st_mtime_ns
    page, outcome = recorded_run(host, fake_time, (watch.PASS, PASSED_LINES))
    assert outcome.text == ""
    assert last_failure_path().read_text(encoding="utf-8") == earlier
    assert last_failure_path().stat().st_mtime_ns == before
    assert json.loads(watch.report_path().read_text(encoding="utf-8"))["state"] == "ok"


def test_last_failure_the_next_failed_run_replaces_the_second_file(tmp_path, fake_time):
    host = Host(tmp_path)
    first_probe, second_probe = release_probes()[:2]
    host.refuse(first_probe, 403, "route_not_allowed")
    host.run(run_page_check=lambda path: watch.PASS)
    first = last_failure_path().read_bytes()
    host.heal(first_probe)
    host.run(run_page_check=lambda path: watch.PASS)
    assert last_failure_path().read_bytes() == first
    host.refuse(second_probe, 403, "route_not_allowed")
    host.run(run_page_check=lambda path: watch.PASS)
    assert last_failure_path().read_bytes() == watch.report_path().read_bytes() != first


def test_retry_an_overslept_cooldown_leaves_the_first_failure_standing(tmp_path, fake_time):
    fake_time.late = 0.2
    host = Host(tmp_path)
    page, outcome = recorded_run(host, fake_time, (watch.FAIL, FAILED_LINES), (watch.PASS, PASSED_LINES), spend=449.9)
    assert fake_time.slept == [30]
    assert len(page.calls) == 1
    assert outcome.kind == "alert"
    entry = page_entry(watch.report_path())
    assert entry["outcome"] == watch.FAIL
    assert entry["page_tries"] == [{"outcome": watch.FAIL, "page_lines": FAILED_LINES}]
    assert last_failure_path().read_bytes() == watch.report_path().read_bytes()


class FakeProcess:
    """A page check process that needs ``runs_for`` seconds of the fake clock and takes ``launch`` seconds to start."""

    def __init__(self, fake_time, launch, runs_for, status):
        fake_time.now += launch
        self.fake_time, self.left, self.status, self.waits = fake_time, runs_for, status, []
        self.stdout = None

    def poll(self):
        return None

    def wait(self, timeout=None):
        self.waits.append(timeout)
        if timeout is None or self.left <= timeout:
            self.fake_time.now += self.left
            self.left = 0
            return self.status
        self.fake_time.now += timeout
        self.left -= timeout
        raise subprocess.TimeoutExpired("page-check", timeout)

    def kill(self):
        pass


def test_retry_the_launch_of_the_second_try_comes_out_of_the_one_limit_of_600_seconds(tmp_path, fake_time, monkeypatch):
    specs = [(0, 20, 1), (5, 550, 0)]
    processes = []

    def popen(argv, **kwargs):
        processes.append(FakeProcess(fake_time, *specs[len(processes)]))
        return processes[-1]

    monkeypatch.setattr(watch, "subprocess", SimpleNamespace(
        Popen=popen, PIPE=subprocess.PIPE, DEVNULL=subprocess.DEVNULL, TimeoutExpired=subprocess.TimeoutExpired))
    host = Host(tmp_path)
    outcome = host.run(run_page_check=None)
    assert fake_time.slept == [30]
    assert len(processes) == 2
    assert processes[1].waits[0] == 545
    entry = page_entry(watch.report_path())
    assert entry["outcome"] == watch.UNKNOWN
    assert [try_["outcome"] for try_ in entry["page_tries"]] == [watch.FAIL, watch.UNKNOWN]
    assert outcome.text == ""
    assert last_failure_path().read_bytes() == watch.report_path().read_bytes()


@pytest.mark.parametrize("results", [((watch.FAIL, FAILED_LINES), (watch.FAIL, FAILED_LINES)),
                                     ((watch.FAIL, FAILED_LINES), (watch.PASS, PASSED_LINES))])
def test_last_failure_a_write_fault_on_the_second_file_alone_changes_neither_decision_nor_exit(
        tmp_path, fake_time, monkeypatch, capsys, results):
    import errno

    real_write = watch._write
    faulty = [True]

    def write(path, data):
        if faulty[0] and path.name == LAST_FAILURE_FILE:
            raise OSError(errno.ENOSPC, "No space left on device")
        real_write(path, data)

    monkeypatch.setattr(watch, "_write", write)
    host = Host(tmp_path)
    host.run()
    capsys.readouterr()
    state_before = watch.state_path().read_bytes()
    _, faulted = recorded_run(host, fake_time, *results)
    err = capsys.readouterr().err
    state = watch.state_path().read_bytes()
    report = watch.report_path().read_bytes()
    assert not last_failure_path().exists()
    assert "state not saved" not in err
    assert "last failure report not saved: OSError" in err
    faulty[0] = False
    watch.state_path().write_bytes(state_before)
    host.clock -= HOUR
    _, reference = recorded_run(host, fake_time, *results)
    assert (faulted.code, faulted.kind, faulted.text) == (reference.code, reference.kind, reference.text)
    assert faulted.code == 0
    assert state == watch.state_path().read_bytes()
    assert report == watch.report_path().read_bytes()
    assert last_failure_path().exists()


COUNTING_SCRIPT = "n=$(cat '{count}' 2>/dev/null || echo 0)\necho $((n + 1)) > '{count}'\n[ \"$n\" -ge 1 ]\n"


def scripted_launches(tmp_path, fake_time, monkeypatch, plan):
    """The real _run_page_check on a script that fails its first run and passes later, behind a Popen that follows
    ``plan``: for each launch (spend, pipe_fails). A launch that is not a pipe failure starts the real script and lets
    it exit before the fake clock moves ``spend`` seconds on, so its exit is already reapable when it is observed."""
    host = Host(tmp_path)
    host.page_check.write_text("#!/bin/sh\n" + COUNTING_SCRIPT.format(count=tmp_path / "count"), encoding="utf-8")
    host.page_check.chmod(0o755)
    launches = []

    def popen(argv, **kwargs):
        spend, pipe_fails = plan[len(launches)] if len(launches) < len(plan) else (0, False)
        launches.append(kwargs["stdout"])
        if pipe_fails and kwargs["stdout"] == subprocess.PIPE:
            fake_time.now += spend
            raise OSError("no pipe")
        process = subprocess.Popen(argv, **kwargs)
        time.sleep(0.3)
        fake_time.now += spend
        return process

    monkeypatch.setattr(watch, "subprocess", SimpleNamespace(
        Popen=popen, PIPE=subprocess.PIPE, DEVNULL=subprocess.DEVNULL, TimeoutExpired=subprocess.TimeoutExpired))
    outcome = host.run(run_page_check=None)
    return launches, outcome, page_entry(watch.report_path())


@pytest.mark.linux_only
@pytest.mark.parametrize("pipe_fails", [False, True])
def test_deadline_an_exit_first_observed_after_it_is_unknown_for_the_first_try(tmp_path, fake_time, monkeypatch, pipe_fails):
    plan = [(601, False)] if not pipe_fails else [(0, True), (601, False)]
    launches, outcome, entry = scripted_launches(tmp_path, fake_time, monkeypatch, plan)
    assert len(launches) == (1 if not pipe_fails else 2)
    assert entry["outcome"] == watch.UNKNOWN
    assert [try_["outcome"] for try_ in entry["page_tries"]] == [watch.UNKNOWN]
    assert outcome.text == ""
    assert fake_time.slept == []


@pytest.mark.linux_only
@pytest.mark.parametrize("pipe_fails", [False, True])
def test_deadline_an_exit_first_observed_after_it_is_unknown_for_the_second_try(tmp_path, fake_time, monkeypatch, pipe_fails):
    plan = [(0, False), (700, False)] if not pipe_fails else [(0, False), (0, True), (700, False)]
    launches, outcome, entry = scripted_launches(tmp_path, fake_time, monkeypatch, plan)
    assert len(launches) == (2 if not pipe_fails else 3)
    assert entry["outcome"] == watch.UNKNOWN
    assert [try_["outcome"] for try_ in entry["page_tries"]] == [watch.FAIL, watch.UNKNOWN]
    assert outcome.text == ""
    assert last_failure_path().read_bytes() == watch.report_path().read_bytes()


@pytest.mark.linux_only
@pytest.mark.parametrize("lost, fallback", [(1, True), (2, False)])
def test_fallback_the_retry_without_a_pipe_starts_only_with_120_seconds_left(tmp_path, fake_time, monkeypatch, lost, fallback):
    launches, outcome, entry = scripted_launches(tmp_path, fake_time, monkeypatch, [(449, False), (lost, True)])
    assert launches == [subprocess.PIPE, subprocess.PIPE] + ([subprocess.DEVNULL] if fallback else [])
    assert [try_["outcome"] for try_ in entry["page_tries"]] == [watch.FAIL] + ([watch.PASS] if fallback else [])
    assert entry["outcome"] == (watch.PASS if fallback else watch.FAIL)
    assert (outcome.kind == "alert") is (not fallback)
    assert last_failure_path().read_bytes() == watch.report_path().read_bytes()


@pytest.mark.linux_only
@pytest.mark.parametrize("lost, fallback", [(599, True), (600, False)])
def test_fallback_the_first_try_without_a_pipe_starts_only_before_the_deadline(tmp_path, fake_time, monkeypatch, lost, fallback):
    launches, outcome, entry = scripted_launches(tmp_path, fake_time, monkeypatch, [(lost, True)])
    assert launches == [subprocess.PIPE] + ([subprocess.DEVNULL] if fallback else [])
    assert [try_["outcome"] for try_ in entry["page_tries"]] == [watch.FAIL if fallback else watch.UNKNOWN]
    assert fake_time.slept == []
