"""Hourly check of the owner write routes: plan section (e), tests 1 to 9.

The listener, the page check, the clock and the delivery record are fakes:
nothing here reaches a network or a real service.
"""

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
            if probe.method == method and url.endswith(probe.path):
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
