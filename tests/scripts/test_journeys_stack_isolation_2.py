"""Tests for isolation_check.sh (card P-2), part 2 of 3.

The stand-ins and helpers are in journeys_stack_harness and journeys_stack_isolation_support.
"""

from __future__ import annotations

import pytest

from journeys_stack_harness import (
    all_output,
    APP_C,
    BROWSER_C,
    CHECKS,
    DASH_C,
    failed_checks,
    GATE_C,
    Host,
    MODEL_TOKEN,
    PROD_ORIGIN,
    run_check,
    SAME_HOST_REDIRECT,
    set_hermes_home,
    SIGN_IN_REDIRECT,
    snapshot,
    STACK_HOST,
    verdicts,
)
from journeys_stack_isolation_support import (
    BREAKS,
)
from journeys_stack_harness import reset_tree, shared_reset, stack  # noqa: F401

pytestmark = pytest.mark.linux_only


@pytest.mark.parametrize("case", sorted(BREAKS)[11:21])
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


@pytest.mark.parametrize("value", ["/somewhere/else", "/opt/data/"])
@pytest.mark.parametrize("who", [DASH_C, GATE_C])
def test_check_fails_names_when_hermes_home_is_not_the_home_mount(stack: Host, who: str, value: str):
    stack.update(set_hermes_home(who, value))
    result = run_check(stack)
    assert result.returncode == 1, result.stdout
    assert list(verdicts(result)) == CHECKS
    line = next(line for line in result.stdout.splitlines() if line.startswith("FAIL names-on-template"))
    assert who in line
    if who == DASH_C:
        assert failed_checks(result) == {"names-on-template"}
    assert "names-on-template" in failed_checks(result)


@pytest.mark.parametrize("missing", [GATE_C, APP_C, BROWSER_C, DASH_C])
def test_check_fails_names_when_a_template_container_is_missing(stack: Host, missing: str):
    stack.update(lambda s: s["containers"].pop(missing))
    result = run_check(stack)
    assert result.returncode == 1
    assert "names-on-template" in failed_checks(result)
    assert list(verdicts(result)) == CHECKS


@pytest.mark.parametrize("code", [0, 2, 3, 126, 127])
def test_check_telegram_send_passes_only_when_delivery_fails(stack: Host, code: int):
    stack.update(lambda s: s["hermes"].update(send_rc=code))
    result = run_check(stack)
    assert result.returncode == 1
    assert failed_checks(result) == {"telegram-send-fails"}
