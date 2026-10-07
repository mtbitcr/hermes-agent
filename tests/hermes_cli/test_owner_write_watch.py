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
import sqlite3
import threading
import uuid
from contextlib import closing
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

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
