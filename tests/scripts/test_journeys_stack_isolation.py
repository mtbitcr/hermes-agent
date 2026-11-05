"""Tests for isolation_check.sh and isolation_check.sh server1 (card P-2), part 1 of 3.

The stand-ins and helpers are in journeys_stack_harness and journeys_stack_isolation_support.
"""

from __future__ import annotations

import re

import pytest

from journeys_stack_harness import (
    all_output,
    argv_text,
    CHECK,
    CHECKS,
    failed_checks,
    Host,
    MODEL_TOKEN,
    NO_STACK_ENV,
    parse_template,
    PROD_C,
    PROD_ISSUER,
    PROD_ORIGIN,
    run_check,
    SAME_HOST_REDIRECT,
    SIGN_IN_REDIRECT,
    snapshot,
    STACK_HOST,
    verdicts,
)
from journeys_stack_isolation_support import (
    BREAKS,
    proxy_issuer,
)
from journeys_stack_harness import host, reset_tree, shared_reset, stack  # noqa: F401

pytestmark = pytest.mark.linux_only


@pytest.mark.parametrize("case", sorted(BREAKS)[:11])
def test_check_reports_a_broken_check_and_still_runs_the_others(stack: Host, case: str):
    broken, mutate, env = BREAKS[case]
    if mutate:
        mutate(stack)
    before = snapshot(stack)
    result = run_check(stack, **env)
    assert result.returncode == 1, result.stdout + result.stderr
    assert list(verdicts(result)) == CHECKS, "one line per check, in order"
    assert failed_checks(result) == {broken}
    assert snapshot(stack) == before
    for value in (STACK_HOST, PROD_ORIGIN, SAME_HOST_REDIRECT, SIGN_IN_REDIRECT, MODEL_TOKEN):
        assert value not in all_output(result)


@pytest.mark.parametrize("code, redirect", [
    (401, ""), (403, ""), (302, SIGN_IN_REDIRECT), (307, SIGN_IN_REDIRECT),
    (303, "https://other.example.invalid/"),
])
def test_check_accepts_a_refusal_or_a_redirect_elsewhere(stack: Host, code: int, redirect: str):
    stack.update(lambda s: s["curl"].update(code=code, redirect=redirect))
    result = run_check(stack)
    assert result.returncode == 0, result.stdout
    assert verdicts(result)["production-get-refused"] == "PASS"


@pytest.mark.parametrize("origin", [
    "prod.example.invalid", "ftp://prod.example.invalid", "https://", 'https://prod.example.invalid/"x',
    'https://prod.example.invalid/\nheader = "x: y"', "https://prod.example.invalid/ -o /etc/x",
])
def test_check_refuses_an_origin_that_is_not_a_plain_address_without_calling_curl(stack: Host, origin: str):
    result = run_check(stack, JOURNEY_PRODUCTION_ORIGIN=origin)
    assert result.returncode == 1
    assert failed_checks(result) == {"production-get-refused"}
    assert stack.calls("curl") == []


@pytest.mark.parametrize("args", [[], ["bogus"], [""], ["SERVER2"], ["server2", "server1"]])
def test_check_prints_usage_and_exits_2_for_a_missing_or_other_argument(host: Host, args: list):
    result = host.run(CHECK, *args)
    assert result.returncode == 2
    assert "usage" in result.stderr.lower() and result.stdout == ""
    assert host.calls() == []


@pytest.mark.parametrize("stack_host", ["some-other-host.invalid", "", None])
def test_check_server2_refuses_the_wrong_host_before_any_other_check(host: Host, stack_host):
    result = host.run(CHECK, "server2", env={"JOURNEY_STACK_HOST": stack_host})
    assert result.returncode == 1
    lines = result.stdout.splitlines()
    assert len(lines) == 1 and lines[0].startswith("FAIL host")
    assert host.calls("docker") == [] and host.calls("curl") == []
    assert STACK_HOST not in all_output(result)


# ------------------------------------------------------------------ isolation_check.sh server1

def test_check_server1_passes_when_production_names_another_issuer(host: Host):
    before = snapshot(host)
    result = host.run(CHECK, "server1", env=NO_STACK_ENV)
    assert (result.returncode, result.stdout, result.stderr) == (0, "PASS production-issuer-differs\n", "")
    calls = host.calls()
    assert [c["tool"] for c in calls] == ["docker"]
    assert calls[0]["argv"] == ["exec", PROD_C, "printenv", "RAPHAEL_ACCESS_ISSUER"]
    for value in (PROD_ISSUER, proxy_issuer()):
        assert value not in argv_text(host) and value not in all_output(result)
    assert snapshot(host) == before


@pytest.mark.parametrize("case", ["same", "empty", "unset", "container-missing", "container-stopped"])
def test_check_server1_fails_closed(host: Host, case: str):
    def mutate(state: dict) -> None:
        box = state["containers"][PROD_C]
        if case == "same":
            box["env"] = [f"RAPHAEL_ACCESS_ISSUER={proxy_issuer()}"]
        elif case == "empty":
            box["env"] = ["RAPHAEL_ACCESS_ISSUER="]
        elif case == "unset":
            box["env"] = ["PATH=/usr/bin"]
        elif case == "container-missing":
            state["containers"].pop(PROD_C)
        else:
            box["running"] = False
    host.update(mutate)
    result = host.run(CHECK, "server1", env=NO_STACK_ENV)
    assert result.returncode == 1
    assert re.fullmatch(r"FAIL production-issuer-differs: \S.*\n", result.stdout)
    for value in (PROD_ISSUER, proxy_issuer()):
        assert value not in all_output(result) and value not in argv_text(host)


def test_check_server1_touches_no_stack_container_and_names_only_production(host: Host):
    host.run(CHECK, "server1", env=NO_STACK_ENV)
    named = {part for c in host.calls("docker") for part in c["argv"] if part.startswith("raphael-")}
    assert named == set(parse_template()["production-container"])
