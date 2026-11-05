"""Tests for scripts/journeys/stack_up.sh (card 2 of the journeys stack plan, round 3), part 2 of 3.

The stand-ins and helpers are in journeys_stack_up_support. This part also holds test_4b, test_4c, test_13 and test_15.
"""

from __future__ import annotations

import pytest

from journeys_stack_up_support import (
    ANSWERS_LINE,
    APP_C,
    assert_all_running_and_healthy,
    assert_health_wait_after_lifecycle,
    assert_no_volume_removal,
    BROWSER_C,
    check_calls,
    container_ids,
    containers_by_name,
    create_names,
    DASH_C,
    FIRST_LINE,
    FORMATS,
    GATE_C,
    Host,
    lifecycle_calls,
    namespaces_stale,
    NEW_REVISION,
    output_lines,
    PLACEHOLDER,
    PROXY_C,
    RESET_LINE,
    run_names,
    seed_tokens,
    TOKEN_NAMES,
    TOKENS_V,
)
from journeys_stack_up_support import converged_root, host, stack  # noqa: F401

pytestmark = pytest.mark.linux_only


# --- round 3, rule 1 (finding 1): the namespace rule. The gateway is current only when its network mode names the
# owner app's current container id and its State.StartedAt is strictly later than the owner app's.


def started_key(text: str) -> tuple[str, int]:
    """State.StartedAt as numbers: the whole seconds (fixed width, so exact as text), then nanoseconds."""
    whole, _, fraction = text.removesuffix("Z").partition(".")
    return whole, int((fraction + "000000000")[:9])


def gateway_is_current(state: dict) -> bool:
    """The rule, read from the fake host's records only (never from the script)."""
    by_name = containers_by_name(state)
    app, gate = by_name[APP_C], by_name[GATE_C]
    return gate["network"] == f"container:{app['id']}" and started_key(gate["started_at"]) > started_key(app["started_at"])


def set_started_at(host: Host, app: str, gateway: str) -> None:
    """Overwrite State.StartedAt of the owner app and the gateway with exact strings, as if Docker had printed them."""
    state = host.state()
    containers_by_name(state)[APP_C]["started_at"] = app
    containers_by_name(state)[GATE_C]["started_at"] = gateway
    host.save(state)


def test_16_an_owner_app_restart_with_the_same_id_restarts_the_gateway_once_before_the_health_wait(stack: Host) -> None:
    """Finding 1: the app restarts (same id) and the gateway stays in the old namespace. Converge restarts the gateway
    once (same id; no rm, no create) before the health wait; afterwards it is current; a third converge changes nothing."""
    ids = container_ids(stack)
    assert gateway_is_current(stack.state())
    assert stack.docker("restart", ids[APP_C]).returncode == 0
    assert namespaces_stale(stack.state()) == [GATE_C] and not gateway_is_current(stack.state())
    mark = stack.mark()
    result = stack.converge()
    assert result.returncode == 0, result.stdout + result.stderr
    assert output_lines(result)[-1] == ANSWERS_LINE
    docker = stack.docker_calls(mark)
    assert lifecycle_calls(docker) == [["restart", ids[GATE_C]]]
    assert not [argv for argv in docker if argv[0] in ("rm", "create", "start", "stop", "build", "volume", "network")]
    assert_health_wait_after_lifecycle(docker)
    assert container_ids(stack) == ids
    assert gateway_is_current(stack.state())
    assert_all_running_and_healthy(stack)

    mark = stack.mark()
    result = stack.converge()
    assert result.returncode == 0, result.stdout + result.stderr
    assert output_lines(result)[-1] == ANSWERS_LINE
    docker = stack.docker_calls(mark)
    assert lifecycle_calls(docker) == []
    assert {argv[0] for argv in docker} <= {"info", "inspect", "cp"}, docker
    assert_no_volume_removal(stack.converged_docker + stack.docker_calls())


# (app StartedAt, gateway StartedAt, is the gateway current): written the way Docker prints them.
START_TIMES = [
    pytest.param("2026-10-28T17:34:11Z", "2026-10-28T17:34:11.5Z", True, id="later_but_sorts_lower_as_text"),
    pytest.param("2026-10-28T17:34:11.5Z", "2026-10-28T17:34:11Z", False, id="earlier_but_sorts_higher_as_text"),
    pytest.param("2026-10-28T17:34:11Z", "2026-10-28T17:34:11Z", False, id="equal_whole_seconds"),
    pytest.param("2026-10-28T17:34:11.5Z", "2026-10-28T17:34:11.5Z", False, id="equal_with_fraction"),
    pytest.param("2026-10-28T17:34:11.12345Z", "2026-10-28T17:34:11.123451Z", True, id="short_fraction_then_long_later"),
    pytest.param("2026-10-28T17:34:11.123451Z", "2026-10-28T17:34:11.12345Z", False, id="long_fraction_then_short_earlier"),
    pytest.param("2026-10-28T17:34:11.05Z", "2026-10-28T17:34:11.08Z", True, id="leading_zero_fraction_later"),
    pytest.param("2026-10-28T17:34:11.08Z", "2026-10-28T17:34:11.05Z", False, id="leading_zero_fraction_earlier"),
    pytest.param("2026-10-28T17:34:08Z", "2026-10-28T17:34:09Z", True, id="leading_zero_seconds_later"),
    pytest.param("2026-10-28T17:34:59.9Z", "2026-10-28T17:35:00Z", True, id="minute_boundary_later"),
    pytest.param("2026-10-28T17:35:00Z", "2026-10-28T17:34:59.999999999Z", False, id="nanosecond_fraction_earlier"),
]


@pytest.mark.parametrize(("app_time", "gate_time", "current"), START_TIMES)
def test_17_start_times_are_compared_as_numbers_and_never_as_text(stack: Host, app_time: str, gate_time: str, current: bool) -> None:
    """A gateway that started strictly later is left alone; equal or earlier gets exactly one restart. The cases are
    chosen so that a raw string comparison answers wrongly in both directions."""
    ids = container_ids(stack)
    set_started_at(stack, app_time, gate_time)
    assert gateway_is_current(stack.state()) is current
    mark = stack.mark()
    result = stack.converge()
    assert result.returncode == 0, result.stdout + result.stderr
    assert output_lines(result)[-1] == ANSWERS_LINE
    docker = stack.docker_calls(mark)
    if current:
        assert lifecycle_calls(docker) == []
    else:
        assert lifecycle_calls(docker) == [["restart", ids[GATE_C]]]
        assert_health_wait_after_lifecycle(docker)
    assert container_ids(stack) == ids
    assert gateway_is_current(stack.state())
    assert_all_running_and_healthy(stack)


def test_18_a_current_gateway_gets_no_lifecycle_call(stack: Host) -> None:
    """Both conditions hold: no lifecycle call and no token check; the decision is read from the two records."""
    ids = container_ids(stack)
    assert gateway_is_current(stack.state())
    mark = stack.mark()
    result = stack.converge()
    assert result.returncode == 0, result.stdout + result.stderr
    assert output_lines(result)[-1] == ANSWERS_LINE
    docker = stack.docker_calls(mark)
    assert lifecycle_calls(docker) == []
    assert {argv[0] for argv in docker} <= {"info", "inspect", "cp"}, docker
    formats = [argv[4] for argv in docker if argv[0] == "inspect"]
    assert FORMATS["network_mode"] in formats and formats.count(FORMATS["started_at"]) >= 2
    assert container_ids(stack) == ids


def test_19_a_stopped_gateway_that_names_the_current_app_is_started_once(stack: Host) -> None:
    """Stopped gateway, network mode names the app: one start call, same id; no restart, no remove, no create."""
    ids = container_ids(stack)
    assert stack.docker("stop", ids[GATE_C]).returncode == 0
    assert containers_by_name(stack.state())[GATE_C]["network"] == f"container:{ids[APP_C]}"
    mark = stack.mark()
    result = stack.converge()
    assert result.returncode == 0, result.stdout + result.stderr
    assert output_lines(result)[-1] == ANSWERS_LINE
    docker = stack.docker_calls(mark)
    assert lifecycle_calls(docker) == [["start", ids[GATE_C]]]
    assert not [argv for argv in docker if argv[0] in ("restart", "rm", "create")]
    assert container_ids(stack) == ids
    assert gateway_is_current(stack.state())
    assert_all_running_and_healthy(stack)


def test_19b_a_stopped_gateway_that_does_not_name_the_current_app_is_made_again(stack: Host) -> None:
    """Stopped gateway whose network mode names another id: removed by id and created again, never started or restarted."""
    ids = container_ids(stack)
    assert stack.docker("stop", ids[GATE_C]).returncode == 0
    state = stack.state()
    containers_by_name(state)[GATE_C]["network"] = "container:" + "0" * 64
    stack.save(state)
    mark = stack.mark()
    result = stack.converge()
    assert result.returncode == 0, result.stdout + result.stderr
    assert output_lines(result)[-1] == ANSWERS_LINE
    docker = stack.docker_calls(mark)
    assert [argv for argv in docker if argv[0] == "rm"] == [["rm", "--force", ids[GATE_C]]]
    assert run_names(docker) == [GATE_C]
    assert not [argv for argv in docker if argv[0] in ("start", "restart", "stop")]
    after = container_ids(stack)
    assert after[APP_C] == ids[APP_C] and after[GATE_C] != ids[GATE_C]
    assert gateway_is_current(stack.state())
    assert_all_running_and_healthy(stack)


def test_20_first_creation_with_a_new_owner_app_tag_makes_the_gateway_again_without_starting_it(host: Host) -> None:
    """Token files missing: only condition (a) decides and nothing starts. A new app tag makes the app again, so the
    old gateway names an app that is gone: it is removed and created (not started) into the new app's namespace."""
    assert host.converge().returncode == 0
    before = container_ids(host)
    mark = host.mark()
    result = host.converge(app=NEW_REVISION)
    assert result.returncode == 0, result.stdout + result.stderr
    assert output_lines(result)[-1] == FIRST_LINE
    docker = host.docker_calls(mark)
    assert not [argv for argv in docker if argv[0] in ("start", "restart", "stop")]
    assert sorted(argv[2] for argv in docker if argv[0] == "rm") == sorted([before[APP_C], before[GATE_C], before[BROWSER_C]])
    assert create_names(docker) == [APP_C, GATE_C]
    assert run_names(docker) == [BROWSER_C]
    after = container_ids(host)
    containers = host.containers()
    assert containers[GATE_C]["network"] == f"container:{after[APP_C]}"
    assert not containers[APP_C]["running"] and not containers[GATE_C]["running"]


def test_4b_first_creation_run_again_while_files_are_missing_changes_nothing_and_ends_the_same(host: Host) -> None:
    """Files still missing: nothing is created, started or written; the same last line; the check only reads."""
    assert host.converge().returncode == 0
    before = host.state()
    mark = host.mark()
    result = host.converge()
    assert result.returncode == 0, result.stdout + result.stderr
    assert output_lines(result)[-1] == FIRST_LINE
    calls = host.docker_calls(mark)
    assert {argv[0] for argv in calls} <= {"info", "inspect", "cp", "run"}, calls
    assert len(check_calls(calls)) == 1 and len([argv for argv in calls if argv[0] == "run"]) == 1
    assert not [argv for argv in calls if argv[0] in ("create", "start", "restart", "stop", "rm", "build")]
    assert host.state() == before


def test_4c_when_the_files_exist_the_stopped_owner_app_is_started_first_then_the_gateway(host: Host) -> None:
    """After a first creation the files appear: start the app, then the gateway, and wait for all five."""
    assert host.converge().returncode == 0
    ids = container_ids(host)
    seed_tokens(host)
    mark = host.mark()
    result = host.converge()
    assert result.returncode == 0, result.stdout + result.stderr
    assert output_lines(result)[-1] == ANSWERS_LINE
    calls = host.docker_calls(mark)
    assert [argv for argv in calls if argv[0] in ("start", "restart", "stop")] == [["start", ids[APP_C]], ["start", ids[GATE_C]]]
    assert not [argv for argv in calls if argv[0] in ("create", "rm", "volume", "network", "build")]
    assert [argv for argv in calls if argv[0] == "run"] == check_calls(calls) and len(check_calls(calls)) == 1
    state = host.state()
    for name, container in containers_by_name(state).items():
        assert container["running"] and container["status"] == "healthy", name
    assert container_ids(host) == ids
    assert namespaces_stale(state) == []
    assert state["volumes"][TOKENS_V]["files"] == {n: PLACEHOLDER for n in TOKEN_NAMES}


@pytest.mark.parametrize("role", [PROXY_C, DASH_C, BROWSER_C])
def test_13_a_stopped_kept_proxy_dashboard_or_browser_is_started_and_nothing_else_is_called(stack: Host, role: str) -> None:
    """One stopped kept container: exactly one start call, for it; no other lifecycle call; same ids; all healthy."""
    ids = container_ids(stack)
    assert stack.docker("stop", ids[role]).returncode == 0
    assert containers_by_name(stack.state())[role]["running"] is False
    mark = stack.mark()
    result = stack.converge()
    assert result.returncode == 0, result.stdout + result.stderr
    assert output_lines(result)[-1] == ANSWERS_LINE
    docker = stack.docker_calls(mark)
    assert lifecycle_calls(docker) == [["start", ids[role]]]
    assert {argv[0] for argv in docker} <= {"info", "inspect", "cp", "start"}, docker
    assert container_ids(stack) == ids
    assert_all_running_and_healthy(stack)
    assert_no_volume_removal(stack.converged_docker + stack.docker_calls())


def test_15_a_reset_that_fails_after_its_stop_phase_is_recovered_by_the_next_converge(stack: Host) -> None:
    """The unchanged reset stops the gateway and the dashboard, then one transient failure hits its first writing
    helper: the reset exits non-zero. After the stand-in is restored, converge starts the dashboard and the gateway."""
    ids = container_ids(stack)
    state = stack.state()
    state["fail_next_writing_helper"] = True
    stack.save(state)
    mark = stack.mark()
    reset = stack.run("reset_stack.sh")
    assert reset.returncode != 0, reset.stdout + reset.stderr
    assert RESET_LINE not in output_lines(reset)
    failed = [c["argv"] for c in stack.calls()[mark:] if c["tool"] == "docker" and c["rc"] == 125]
    assert len(failed) == 1 and failed[0][:2] == ["run", "--rm"]
    assert [argv for argv in stack.docker_calls(mark) if argv[0] in ("stop", "start", "restart")] == [
        ["stop", GATE_C],
        ["stop", DASH_C],
    ]
    state = stack.state()
    assert state["fail_next_writing_helper"] is False
    assert sorted(name for name, c in containers_by_name(state).items() if not c["running"]) == sorted([DASH_C, GATE_C])

    mark = stack.mark()
    result = stack.converge()
    assert result.returncode == 0, result.stdout + result.stderr
    assert output_lines(result)[-1] == ANSWERS_LINE
    docker = stack.docker_calls(mark)
    assert lifecycle_calls(docker) == [["start", ids[DASH_C]], ["start", ids[GATE_C]]]
    assert container_ids(stack) == ids
    assert_all_running_and_healthy(stack)
    assert all(stack.state()["volumes"][TOKENS_V]["files"].get(name) for name in TOKEN_NAMES)
    assert_no_volume_removal(stack.converged_docker + stack.docker_calls())
