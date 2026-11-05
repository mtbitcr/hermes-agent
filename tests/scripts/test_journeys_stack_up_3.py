"""Tests for scripts/journeys/stack_up.sh (card 2 of the journeys stack plan, rounds 2 to 5), part 3 of 3.

The stand-ins and helpers are in journeys_stack_up_support.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path

import pytest

from journeys_stack_up_support import (
    ANSWERS_LINE,
    APP_C,
    assert_all_running_and_healthy,
    assert_health_wait_after_lifecycle,
    assert_no_volume_removal,
    BROWSER_C,
    ca_part,
    changing,
    check_calls,
    command_of,
    container_ids,
    containers_by_name,
    DASH_C,
    DASHBOARD_COMMAND,
    foreign_container,
    FORMATS,
    GATE_C,
    GATEWAY_COMMAND,
    Host,
    JOURNEYS,
    LABEL,
    lifecycle_calls,
    namespaces_stale,
    NEW_PLATFORM,
    NEW_REVISION,
    output_lines,
    PLACEHOLDER,
    PLATFORM,
    PROXY_C,
    PROXY_TAG,
    REVISION,
    run_names,
    SECTIONS,
    seed_image,
    seed_tokens,
    STACK_CONTAINERS,
    sub_host,
    TOKEN_NAMES,
    TOKENS_V,
)
from journeys_stack_up_support import converged_root, host, recorded_gateway_command, stack  # noqa: F401

pytestmark = pytest.mark.linux_only


# --- round 3, rule 2 (finding 2): a stopped kept container is started, never recreated


def test_14_all_five_stopped_kept_containers_are_started_in_template_order_before_the_health_wait(stack: Host) -> None:
    """All five stopped: five start calls, proxy, owner app, dashboard, gateway, browser, all before the health wait."""
    ids = container_ids(stack)
    for name in reversed(STACK_CONTAINERS):
        assert stack.docker("stop", ids[name]).returncode == 0
    assert not [c for c in containers_by_name(stack.state()).values() if c["running"]]
    mark = stack.mark()
    result = stack.converge()
    assert result.returncode == 0, result.stdout + result.stderr
    assert output_lines(result)[-1] == ANSWERS_LINE
    docker = stack.docker_calls(mark)

    # The gateway was stopped too, so the owner app's start leaves no gateway beside a stopped app to make again:
    # the same five ids stay and the five lifecycle calls are the five starts, in the order of the template.
    assert lifecycle_calls(docker) == [["start", ids[name]] for name in STACK_CONTAINERS]
    # Every start comes before the first health inspect: the census of what runs is read before the first
    # lifecycle call. Round 3 relaxes this one assertion: the namespace rule reads the id, the network mode and
    # State.StartedAt between the app's start and the gateway's start, so only health inspects are forbidden there.
    assert_health_wait_after_lifecycle(docker)
    assert container_ids(stack) == ids
    assert_all_running_and_healthy(stack)
    assert_no_volume_removal(stack.converged_docker + stack.docker_calls())


# --- round 4, rule 1 and 3: an identity read that fails is not "absent"; it refuses before any change

TEMPLATE_CONTAINERS = SECTIONS["containers"]
TEMPLATE_PROXY, TEMPLATE_APP = TEMPLATE_CONTAINERS[0], TEMPLATE_CONTAINERS[1]
CONNECTION_ERROR = "Cannot connect to the Docker daemon at unix:///nonexistent/stand-in-only/docker.sock. Is it running?\n"


def unlabelled_stopped(host: Host, role: str) -> None:
    """The container of that role keeps its name and image, loses the label and stops (the review's reproduction)."""
    state = host.state()
    container = containers_by_name(state)[role]
    container["labels"] = {}
    container["running"] = False
    host.save(state)


@pytest.mark.parametrize("role", TEMPLATE_CONTAINERS)
def test_21_a_failed_first_identity_read_refuses_before_any_change(stack: Host, role: str) -> None:
    """One connection error on the first identity read of an unlabelled, stopped container: the fixed line, exit 1,
    no changing call, the fake host unchanged. Without the rule the script would keep the container and start it."""
    unlabelled_stopped(stack, role)
    stack.script_read(role, stderr=CONNECTION_ERROR)
    before = stack.state()
    mark = stack.mark()
    result = stack.converge()
    assert result.returncode == 1, result.stdout + result.stderr
    assert result.stdout == ""
    assert result.stderr == f"stack_up: cannot read the container {role}; nothing was changed\n"
    assert changing(stack.docker_calls(mark)) == []
    assert stack.state() == before


@pytest.mark.parametrize("role", TEMPLATE_CONTAINERS)
def test_22_the_same_unlabelled_container_without_a_read_error_refuses_before_any_change(stack: Host, role: str) -> None:
    """The control of test_21 for all five roles (test_6 keeps its proxy case): the read works, the label is missing."""
    unlabelled_stopped(stack, role)
    before = stack.state()
    mark = stack.mark()
    result = stack.converge()
    assert result.returncode != 0
    assert result.stderr == f"stack_up: container {role} exists but this script did not make it; nothing was changed\n"
    assert changing(stack.docker_calls(mark)) == []
    assert stack.state() == before


# (status, stdout, stderr) of a first identity read of the proxy that is not absent and not a good read.
UNREADABLE_PROXY_READS = {
    "connection_error": (1, "", CONNECTION_ERROR),
    "permission_denied": (1, "", "permission denied while trying to connect to the Docker daemon socket\n"),
    "no_such_container_names_another": (1, "", f"Error: No such container: {TEMPLATE_APP}\n"),
    "no_such_container_and_a_second_line": (1, "", f"Error: No such container: {TEMPLATE_PROXY}\nsecond line\n"),
    "status_0_empty_stdout": (0, "", ""),
    "status_0_odd_stdout": (0, "not an id, a space and a label\n", ""),
    "non_zero_empty_stderr": (1, "", ""),
}


@pytest.mark.parametrize("case", sorted(UNREADABLE_PROXY_READS))
def test_23_any_other_answer_to_the_first_proxy_read_refuses_before_any_change(host: Host, case: str) -> None:
    """Only Docker's own 'no such container' for the asked name is absent. On an empty host each other answer to the
    first identity read of the proxy ends in the fixed line for the proxy: nothing on stdout, no changing call."""
    status, stdout, stderr = UNREADABLE_PROXY_READS[case]
    seed_tokens(host)
    host.script_read(TEMPLATE_PROXY, status=status, stdout=stdout, stderr=stderr)
    before = host.state()
    result = host.converge()
    assert result.returncode == 1, result.stdout + result.stderr
    assert result.stdout == ""
    assert result.stderr == f"stack_up: cannot read the container {TEMPLATE_PROXY}; nothing was changed\n"
    assert changing(host.docker_calls()) == []
    assert host.state() == before


def fired_reads(host: Host) -> int:
    """How many scripted reads the stand-in has answered (scripted.json is the stand-in's own bookkeeping)."""
    entries = json.loads((host.root / "scripted.json").read_text(encoding="utf-8"))
    return sum(entry["fired"] for entry in entries)


def test_23b_the_daemons_own_wording_is_absent_and_a_warning_beside_a_good_read_is_no_failure(host: Host) -> None:
    """Control for test_23: with STUB_NOT_FOUND=daemon (the other wording of Docker's 'no such container') a first
    creation works as in test_1; then a Docker warning on stderr beside a good read of the proxy changes nothing."""
    seed_tokens(host)
    created = host.converge(STUB_NOT_FOUND="daemon")
    assert created.returncode == 0, created.stdout + created.stderr
    assert output_lines(created)[-1] == ANSWERS_LINE
    assert_all_running_and_healthy(host)

    ids = container_ids(host)
    host.script_read(TEMPLATE_PROXY, status=0, stdout=None, stderr="WARNING: a placeholder warning from the daemon\n")
    mark = host.mark()
    result = host.converge()
    assert result.returncode == 0, result.stdout + result.stderr
    assert output_lines(result)[-1] == ANSWERS_LINE
    assert fired_reads(host) == 1
    assert {argv[0] for argv in host.docker_calls(mark)} <= {"info", "inspect", "cp"}
    assert container_ids(host) == ids


def test_24_a_read_that_fails_after_the_first_change_stops_with_the_short_line_and_the_next_run_recovers(stack: Host) -> None:
    """The first identity read of the owner app after the first changing call fails (a new owner app revision, so the
    CA copy and the new image layers come first). Converge stops with 'cannot read the container ROLE' and without
    'nothing was changed', makes no lifecycle call, and the same run again, unscripted, finishes."""
    stack.script_read(TEMPLATE_APP, stderr=CONNECTION_ERROR, after_change=True)
    mark = stack.mark()
    failed = stack.converge(app=NEW_REVISION)
    assert failed.returncode == 1, failed.stdout + failed.stderr
    assert output_lines(failed)[-1] == f"stack_up: cannot read the container {TEMPLATE_APP}"
    assert "nothing was changed" not in failed.stderr
    assert fired_reads(stack) == 1
    docker = stack.docker_calls(mark)
    assert changing(docker)
    assert lifecycle_calls(docker) == []

    result = stack.converge(app=NEW_REVISION)
    assert result.returncode == 0, result.stdout + result.stderr
    assert output_lines(result)[-1] == ANSWERS_LINE
    assert_all_running_and_healthy(stack)
    assert_no_volume_removal(stack.converged_docker + stack.docker_calls())


def probe_argv(operation: str, cid: str, copy_to: Path) -> list[str]:
    """One docker call of the kinds the verified-id rule covers, aimed at the container with that id."""
    if operation == "rm":
        return ["rm", "--force", cid]
    if operation == "cp":
        return ["cp", f"{cid}:/state/ca.pem", str(copy_to)]
    if operation == "create":
        return ["create", "--label", LABEL, "--restart", "unless-stopped", "--name", f"probe-{cid[:12]}.invalid",
                "--network", f"container:{cid}", f"raphael-journeys-platform:{PLATFORM}"]
    return [operation, cid]


@pytest.mark.parametrize("operation", ["start", "stop", "restart", "rm", "cp", "create"])
def test_25_with_stub_verify_the_stand_in_refuses_an_id_that_no_read_showed_with_the_label(
    stack: Host, tmp_path: Path, operation: str
) -> None:
    """Harness self-test: with STUB_VERIFY=1 an unlabelled container's id and an id that was never read are refused
    (status 98); after a successful labelled identity read the labelled container is accepted, the unlabelled stays
    refused even after its own successful read."""
    state = stack.state()
    unlabelled = foreign_container(state, "unrelated-container.invalid")
    stack.save(state)
    labelled = container_ids(stack)[TEMPLATE_PROXY]
    (stack.root / "verified.txt").write_text("", encoding="utf-8")

    def attempt(cid: str) -> subprocess.CompletedProcess:
        return stack.docker(*probe_argv(operation, cid, tmp_path / "ca.pem"), STUB_VERIFY="1")

    def read(name: str) -> None:
        answer = stack.docker("inspect", "--type", "container", "--format", FORMATS["ident"], name, STUB_VERIFY="1")
        assert answer.returncode == 0, answer.stderr

    for cid in (unlabelled, labelled):
        refused = attempt(cid)
        assert refused.returncode == 98 and "was not read with the label" in refused.stderr, cid
    read(TEMPLATE_PROXY)
    accepted = attempt(labelled)
    assert accepted.returncode == 0, accepted.stderr
    read("unrelated-container.invalid")
    refused = attempt(unlabelled)
    assert refused.returncode == 98 and "was not read with the label" in refused.stderr


# --- round 5, rules 1 and 2: the label is compared inside Docker's template, on the raw value. The shell drops trailing
# newlines and NUL bytes from a captured value, so a script that compared the captured text would take the first three
# of these values for stack_up.

LABEL_KEY = LABEL.partition("=")[0]
FOREIGN_LABEL_VALUES = {
    "one_newline": "stack_up\n",
    "two_newlines": "stack_up\n\n",
    "nul_byte": "stack_\x00up",
    "extra_letter": "stack_upx",
}
ALL_LABEL_VALUES = {"exact": "stack_up", **FOREIGN_LABEL_VALUES}


def stopped_with_label_value(host: Host, role: str, value: str) -> None:
    """The container of that role keeps its name and image, gets the label with exactly this value and stops."""
    state = host.state()
    container = containers_by_name(state)[role]
    container["labels"] = {LABEL_KEY: value}
    container["running"] = False
    host.save(state)


@pytest.mark.parametrize("case", sorted(FOREIGN_LABEL_VALUES))
@pytest.mark.parametrize("role", TEMPLATE_CONTAINERS)
def test_26_converge_refuses_a_container_whose_label_value_is_not_exactly_stack_up(
    stack: Host, role: str, case: str
) -> None:
    """A stopped container of that role carries one of the four values: the fixed refusal for the role, nothing on
    stdout, no changing call, the fake host unchanged. Without the rule the script keeps the container and starts it."""
    stopped_with_label_value(stack, role, FOREIGN_LABEL_VALUES[case])
    before = stack.state()
    mark = stack.mark()
    result = stack.converge()
    assert result.returncode != 0
    assert result.stdout == ""
    assert result.stderr == f"stack_up: container {role} exists but this script did not make it; nothing was changed\n"
    assert changing(stack.docker_calls(mark)) == []
    assert stack.state() == before


@pytest.mark.parametrize("case", sorted(FOREIGN_LABEL_VALUES))
@pytest.mark.parametrize("role", TEMPLATE_CONTAINERS)
def test_27_versions_refuses_a_container_whose_label_value_is_not_exactly_stack_up(
    stack: Host, role: str, case: str
) -> None:
    """The read-only versions decides ownership too: on the host of test_26 it ends in the same refusal line, prints no
    version, makes no changing call and leaves the fake host unchanged."""
    stopped_with_label_value(stack, role, FOREIGN_LABEL_VALUES[case])
    before = stack.state()
    mark = stack.mark()
    result = stack.run("stack_up.sh", "versions")
    assert result.returncode != 0
    assert result.stdout == ""
    assert result.stderr == f"stack_up: container {role} exists but this script did not make it; nothing was changed\n"
    assert changing(stack.docker_calls(mark)) == []
    assert stack.state() == before


@pytest.mark.parametrize("role", TEMPLATE_CONTAINERS)
def test_28_a_stopped_container_with_exactly_the_label_value_is_accepted_and_started(stack: Host, role: str) -> None:
    """The control of test_26 and test_27 on the same host: the exact value is accepted. Versions prints the five
    'ROLE TAG' lines; converge exits 0, calls start for that container (the call test_26 must not see), keeps the five
    ids and ends with all five running and healthy."""
    stopped_with_label_value(stack, role, "stack_up")
    ids = container_ids(stack)
    ca = ca_part(stack.state())
    mark = stack.mark()
    versions = stack.run("stack_up.sh", "versions")
    assert versions.returncode == 0, versions.stderr
    assert versions.stdout.splitlines() == [
        f"proxy {PROXY_TAG}",
        f"app {REVISION}-{ca}",
        f"dashboard {PLATFORM}",
        f"gateway {PLATFORM}",
        f"browser {REVISION}-{ca}",
    ]
    assert changing(stack.docker_calls(mark)) == []

    mark = stack.mark()
    result = stack.converge()
    assert result.returncode == 0, result.stdout + result.stderr
    assert output_lines(result)[-1] == ANSWERS_LINE
    assert ["start", ids[role]] in lifecycle_calls(stack.docker_calls(mark))
    assert container_ids(stack) == ids
    assert_all_running_and_healthy(stack)


@pytest.mark.parametrize("case", sorted(ALL_LABEL_VALUES))
def test_29_the_stand_in_answers_the_identity_format_with_one_fixed_word_and_never_a_label_value(
    host: Host, case: str
) -> None:
    """Harness self-test: one direct stand-in 'docker inspect' of a container that carries this label value answers
    '<id> owned' for exactly stack_up and '<id> foreign' for every other value, shows no label text, and the id reaches
    verified.txt only when the answer is owned."""
    value = ALL_LABEL_VALUES[case]
    state = host.state()
    cid = foreign_container(state, PROXY_C)
    state["containers"][cid]["labels"] = {LABEL_KEY: value}
    host.save(state)
    (host.root / "verified.txt").write_text("", encoding="utf-8")
    answer = host.docker("inspect", "--type", "container", "--format", FORMATS["ident"], PROXY_C)
    word = "owned" if value == "stack_up" else "foreign"
    assert answer.returncode == 0, answer.stderr
    assert answer.stdout == f"{cid} {word}\n"
    assert answer.stderr == ""
    assert "stack_" not in answer.stdout
    assert (host.root / "verified.txt").read_text(encoding="utf-8").split() == ([cid] if word == "owned" else [])


# --- extras: not among the plan's eight tests ---


def test_extra_versions_prints_one_line_per_role_and_changes_nothing(tmp_path: Path) -> None:
    """versions: five "<role> <tag>" lines in template order; every tag is none on an empty host."""
    empty = sub_host(tmp_path, "empty")
    result = empty.run("stack_up.sh", "versions")
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == [f"{role} none" for role in ("proxy", "app", "dashboard", "gateway", "browser")]
    assert changing(empty.docker_calls()) == []

    made = sub_host(tmp_path, "made")
    assert made.converge().returncode == 0
    mark = made.mark()
    result = made.run("stack_up.sh", "versions")
    ca = ca_part(made.state())
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == [
        f"proxy {PROXY_TAG}",
        f"app {REVISION}-{ca}",
        f"dashboard {PLATFORM}",
        f"gateway {PLATFORM}",
        f"browser {REVISION}-{ca}",
    ]
    assert changing(made.docker_calls(mark)) == []


def test_extra_build_proxy_makes_one_image_tagged_by_the_hash_of_its_inputs(host: Host) -> None:
    """build proxy: one docker build, tag = first 12 hex of sha256 over synthetic_proxy.py then proxy.Dockerfile."""
    result = host.run("stack_up.sh", "build", "proxy")
    assert result.returncode == 0, result.stdout + result.stderr
    digest = hashlib.sha256()
    digest.update((JOURNEYS / "synthetic_proxy.py").read_bytes())
    digest.update((JOURNEYS / "images" / "proxy.Dockerfile").read_bytes())
    tag = digest.hexdigest()[:12]
    assert result.stdout.splitlines() == [f"stack_up: built proxy {tag}"]
    builds = [argv for argv in host.docker_calls() if argv[0] == "build"]
    assert len(builds) == 1 and builds[0][builds[0].index("--tag") + 1] == f"raphael-journeys-proxy:{tag}"
    assert changing(host.docker_calls()) == builds
    assert host.state()["images"][f"raphael-journeys-proxy:{tag}"]["built"]["context_files"] == ["synthetic_proxy.py"]


def test_extra_build_proxy_with_its_tag_already_there_builds_nothing_and_prints_the_same_last_line(host: Host) -> None:
    """A rebuild would move the tag to a new image and orphan the running proxy's image: the tag is kept as it is."""
    first = host.run("stack_up.sh", "build", "proxy")
    assert first.returncode == 0, first.stdout + first.stderr
    (build,) = [argv for argv in host.docker_calls() if argv[0] == "build"]  # the tag was missing: one build
    ref = build[build.index("--tag") + 1]
    tag, image_id = ref.partition(":")[2], host.state()["images"][ref]["id"]
    mark = host.mark()
    again = host.run("stack_up.sh", "build", "proxy")
    assert again.returncode == 0, again.stdout + again.stderr
    assert again.stdout.splitlines()[-1] == first.stdout.splitlines()[-1] == f"stack_up: built proxy {tag}"
    assert not [argv for argv in host.docker_calls(mark) if argv[0] == "build"]
    assert changing(host.docker_calls(mark)) == []
    assert host.state()["images"][ref]["id"] == image_id


def wait_for_image(host: Host, ref: str) -> dict:
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        images = host.state()["images"]
        if ref in images:
            return images[ref]
        time.sleep(0.2)
    raise AssertionError(f"{ref} was never built")


def test_extra_build_platform_returns_at_once_builds_detached_and_is_not_repeated(host: Host) -> None:
    """build platform: one line and exit 0 at once; the detached build reads the archive; a second call is a no-op."""
    archive = host.root / "platform.tar"
    archive.write_bytes(b"archive" * 100)
    ref = f"raphael-journeys-platform:{NEW_PLATFORM}"
    result = host.run("stack_up.sh", "build", "platform", NEW_PLATFORM, str(archive))
    assert result.returncode == 0, result.stdout + result.stderr
    log_file = host.root / "tmp" / "stack_up-platform-build.log"
    assert result.stdout.splitlines() == [f"stack_up: building platform {NEW_PLATFORM}; log {log_file}"]
    built = wait_for_image(host, ref)["built"]
    assert built["stdin_bytes"] == 700 and built["args"] == {"HERMES_GIT_SHA": NEW_PLATFORM}
    mark = host.mark()
    again = host.run("stack_up.sh", "build", "platform", NEW_PLATFORM, str(archive))
    assert again.returncode == 0 and again.stdout.splitlines() == [f"stack_up: image {ref} already exists"]
    assert not [argv for argv in host.docker_calls(mark) if argv[0] == "build"]


def test_extra_build_platform_leaves_a_running_build_alone(host: Host) -> None:
    """While another build holds the lock, build platform starts nothing and says so."""
    import fcntl

    archive = host.root / "platform.tar"
    archive.write_bytes(b"archive")
    with (host.root / "tmp" / "stack_up-platform-build.lock").open("wb") as held:
        fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
        result = host.run("stack_up.sh", "build", "platform", NEW_PLATFORM, str(archive))
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.splitlines() == ["stack_up: a platform build is already running"]
    assert not [argv for argv in host.docker_calls() if argv[0] == "build"]


def test_extra_docker_that_is_not_rootless_is_refused_before_any_change(tmp_path: Path) -> None:
    host = Host(tmp_path, security=["name=seccomp,profile=builtin"])
    before = host.state()
    result = host.converge()
    assert result.returncode != 0 and "not rootless" in result.stderr and "nothing was changed" in result.stderr
    assert changing(host.docker_calls()) == [] and host.state() == before


def test_extra_script_is_executable_and_wrong_usage_exits_2_before_any_docker_call(host: Host) -> None:
    mode = (JOURNEYS / "stack_up.sh").stat().st_mode & 0o777
    assert mode & 0o755 == 0o755 and not mode & 0o002  # rwxr-xr-x at least; a checkout may add group write (775)
    for args in ([], ["build"], ["build", "nothing"], ["build", "app"], ["converge", PROXY_TAG], ["versions", "x"]):
        result = host.run("stack_up.sh", *args)
        assert result.returncode == 2, args
        assert result.stderr.startswith("usage: stack_up.sh"), args
    assert host.docker_calls() == []


def test_extra_new_platform_commit_recreates_the_dashboard_and_the_gateway_and_keeps_the_owner_app(stack: Host) -> None:
    """A new platform commit: dashboard and gateway are removed by id and made again; the new gateway joins the app."""
    state = stack.state()
    seed_image(state, f"raphael-journeys-platform:{NEW_PLATFORM}", env={"PATH": "/usr/bin", "HERMES_HOME": "/opt/data"})
    stack.save(state)
    before = container_ids(stack)
    mark = stack.mark()
    result = stack.converge(platform=NEW_PLATFORM)
    assert result.returncode == 0, result.stdout + result.stderr
    docker = stack.docker_calls(mark)
    removed = [argv[2] for argv in docker if argv[0] == "rm"]
    assert sorted(removed) == sorted([before[DASH_C], before[GATE_C]])
    assert run_names(docker) == [DASH_C, GATE_C]
    assert len(check_calls(docker)) == 1
    after = container_ids(stack)
    for name in (PROXY_C, APP_C, BROWSER_C):
        assert after[name] == before[name], name
    containers = containers_by_name(stack.state())
    assert containers[GATE_C]["network"] == f"container:{after[APP_C]}"
    assert containers[GATE_C]["image_ref"] == containers[DASH_C]["image_ref"] == f"raphael-journeys-platform:{NEW_PLATFORM}"
    assert command_of(docker, GATE_C) == GATEWAY_COMMAND and command_of(docker, DASH_C) == DASHBOARD_COMMAND
    assert namespaces_stale(stack.state()) == []
    assert not [argv for argv in docker if argv[0] in ("volume", "network")]


# --- round 2: what the stand-in itself refuses and models (finding 2 and finding 3)


@pytest.mark.parametrize("damage", ["missing", "zero_size"])
def test_9_a_started_owner_app_without_all_seven_token_files_exits_at_once_and_is_never_healthy(
    stack: Host, damage: str
) -> None:
    """The stand-in models the real owner app: it exits at once and stays down when any named file is missing or empty."""
    app_id = container_ids(stack)[APP_C]
    for name in TOKEN_NAMES:
        state = stack.state()
        if damage == "missing":
            del state["volumes"][TOKENS_V]["files"][name]
        else:
            state["volumes"][TOKENS_V]["files"][name] = ""
        stack.save(state)
        assert stack.docker("restart", app_id).returncode == 0
        shown = stack.docker("inspect", "--type", "container", "--format", FORMATS["state"], app_id).stdout.strip()
        assert shown == "false none", name
        assert containers_by_name(stack.state())[APP_C]["running"] is False
        state = stack.state()
        state["volumes"][TOKENS_V]["files"][name] = PLACEHOLDER
        stack.save(state)
    assert stack.docker("restart", app_id).returncode == 0
    for _ in range(3):
        shown = stack.docker("inspect", "--type", "container", "--format", FORMATS["state"], app_id).stdout.strip()
    assert shown == "true healthy"


def test_10_the_stand_in_refuses_what_the_engine_refuses_for_a_shared_namespace(stack: Host) -> None:
    """Finding 3: no start while the container to join is not running; no alias, hostname, port, dns, add-host,
    expose or link together with --network container:."""
    ids = container_ids(stack)
    state_before = stack.state()
    platform_ref = f"raphael-journeys-platform:{PLATFORM}"
    base = ["run", "--detach", "--label", LABEL, "--restart", "unless-stopped", "--name", "stand-in-probe"]
    joined = ["--network", f"container:{ids[APP_C]}"]
    for extra in (
        ["--network-alias", "workspace"],
        ["--hostname", "probe"],
        ["-p", "8080:80"],
        ["--publish", "8080:80"],
        ["--dns", "192.0.2.1"],
        ["--add-host", "probe:192.0.2.2"],
        ["--expose", "80"],
        ["--link", "somewhere"],
    ):
        result = stack.docker(*base, *joined, *extra, platform_ref, *GATEWAY_COMMAND)
        assert result.returncode != 0 and "conflicting options" in result.stderr, extra
    assert stack.state() == state_before

    assert stack.docker("stop", GATE_C).returncode == 0
    assert stack.docker("stop", APP_C).returncode == 0
    refused = stack.docker("start", GATE_C)
    assert refused.returncode != 0 and "cannot join network of a non running container" in refused.stderr
    refused = stack.docker(*base, *joined, platform_ref, *GATEWAY_COMMAND)
    assert refused.returncode != 0 and "cannot join network of a non running container" in refused.stderr
    assert "stand-in-probe" not in stack.containers()
    assert stack.docker("start", APP_C).returncode == 0
    assert stack.docker("start", GATE_C).returncode == 0


# --- round 2: the gateway container runs one gateway, shown with the real boot decisions (rule 2, finding 4)
#
# These tests call the real functions with the container command that the converge recorded for the gateway
# container: the legacy-migration check and reconcile_profile_gateways(dry_run=True) of container_boot (the
# cont-init step of the image) and gateway_command -> _cmd_run -> _maybe_redirect_run_to_s6_supervision of
# gateway (the command process). The s6 service manager is a recording fake, run_gateway, os.execvp and
# _block_until_terminated are recorders, subprocess.Popen raises, and every file lives under tmp_path. No
# Python source text is read and no process is started.

ROUND1_CMD = ["gateway", "run", "--no-supervise"]

# argv of the container's main program as ``container_boot._read_container_argv`` sees it.
# ``docker/entrypoint-dispatch.sh`` execs ``/init <wrapper> "$@"`` (no ``hermes`` token); the
# wrapper-with-``hermes`` shape is the one ``_strip_container_argv_prefix`` also documents.
_INIT = "/init"
_WRAPPER = "/opt/hermes/docker/main-wrapper.sh"
ARGV_PREFIXES = [
    pytest.param((_INIT, _WRAPPER, "hermes"), id="wrapper-hermes"),
    pytest.param((_INIT, _WRAPPER), id="entrypoint-dispatch"),
]

# On-disk ``$HERMES_HOME/gateway_state.json`` per home kind. ``_read_desired_state`` returns
# ``desired_state`` when present, else ``gateway_state`` (``draining`` / ``degraded`` read as
# ``running``). ``empty`` has no state file at all.
_TS = 1234567890
HOME_STATE: dict[str, dict | None] = {
    "empty": None,
    "running": {"gateway_state": "running", "timestamp": _TS},
    "degraded": {"gateway_state": "degraded", "timestamp": _TS},
    "draining": {"gateway_state": "draining", "timestamp": _TS},
    # Shape ``S6ServiceManager.start`` writes via ``_write_gateway_desired_state``.
    "desired_running": {"desired_state": "running", "updated_at": _TS},
    # What an operator stop leaves behind (``S6ServiceManager.stop`` + the gateway's own persist).
    "stopped": {"gateway_state": "stopped", "desired_state": "stopped", "updated_at": _TS},
}
PERSISTED_RUNNING = ("running", "degraded", "draining", "desired_running")

# (profile, prior_state, action) of the default slot the boot reconcile decides, per home kind.
_STARTED = ("default", "running", "started")
EXPECTED_DEFAULT_SLOT = {
    "empty": _STARTED,  # the legacy migration seeds ``running`` for a bare ``gateway run``
    "running": _STARTED,
    "degraded": _STARTED,
    "draining": _STARTED,
    "desired_running": _STARTED,
    "stopped": ("default", "stopped", "registered"),  # registered DOWN; the redirect starts it
}
HOME_PARAMS = [pytest.param(kind, id=kind) for kind in HOME_STATE]


class _RecordingS6:
    """Stand-in for ``S6ServiceManager``: records lifecycle calls and starts nothing."""

    kind = "s6"

    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def start(self, name: str) -> None:
        self.calls.append(("start", name))
        # Same on-disk effect as the real ``S6ServiceManager.start`` (writes desired_state
        # ``running`` into the profile's gateway_state.json); a pure file write under tmp.
        from hermes_cli.service_manager import _write_gateway_desired_state

        _write_gateway_desired_state(name, "running")

    def stop(self, name: str) -> None:
        self.calls.append(("stop", name))

    def restart(self, name: str) -> None:
        self.calls.append(("restart", name))


class _NoProcess(subprocess.Popen):
    """Any attempt to start a real process fails the test."""

    def __init__(self, *args, **kwargs):
        raise AssertionError(f"boot decisions must not start a process: {args!r}")


def _noop(_args) -> None:
    return None


def _parse_gateway_args(cmd) -> argparse.Namespace:
    """Parse the container command with the real ``gateway`` argparse tree (``--no-supervise``)."""
    from hermes_cli.subcommands.gateway import build_gateway_parser

    parser = argparse.ArgumentParser(prog="hermes")
    subparsers = parser.add_subparsers(dest="command")
    build_gateway_parser(
        subparsers, cmd_gateway=_noop, cmd_proxy=_noop, cmd_gateway_enroll=_noop,
    )
    return parser.parse_args(list(cmd))


def boot_decisions(
    cmd, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, home_kind: str,
    *, argv_prefix: tuple[str, ...] = (_INIT, _WRAPPER, "hermes"),
) -> dict:
    """Run the real boot decisions for container command ``cmd`` over a ``home_kind`` home.

    Order, as in the real boot: legacy migration + reconcile (cont-init, root), then the command
    process (``hermes <cmd>``). Returns plain data:

      supervised         sorted distinct s6 services started (reconcile 'started' + redirect start)
      foreground_runs    ``run_gateway`` calls made by the command process
      execvp             argv lists passed to ``os.execvp`` by the command process
      redirected         the redirect decision returned True (and was consulted exactly once)
      reconcile_actions  [(profile, prior_state, action), ...] from the dry-run reconcile
      owners             ``supervised`` plus one ``"foreground"`` per foreground gateway
      migration          return of the legacy migration check (``"running"`` or None)
      service_calls      every call the redirect made on the service manager
      heartbeat_fallbacks   times the in-process ``_block_until_terminated`` fallback ran
      desired_state_after   ``_read_desired_state(home)`` once the boot decisions are done
    """
    # Importing ``hermes_cli.gateway`` scaffolds ``$HERMES_HOME`` (SOUL.md, sessions/, ...) and
    # ``tests/conftest.py`` sandboxes HOME/HERMES_HOME before any import for that reason. Do the
    # same here so the helper also leaves the real home alone when it is run on its own.
    monkeypatch.setenv("HOME", str(tmp_path / "import-home"))
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "import-home" / ".hermes"))
    from hermes_cli import container_boot
    from hermes_cli import gateway as gw

    home = tmp_path / "hermes-home"
    home.mkdir()
    state = HOME_STATE[home_kind]
    if state is not None:
        (home / "gateway_state.json").write_text(json.dumps(state) + "\n", encoding="utf-8")
    scandir = tmp_path / "run-service"  # dry run: reconcile must not write s6 slots here

    monkeypatch.setenv("HERMES_HOME", str(home))
    for name in ("HERMES_S6_SUPERVISED_CHILD", "HERMES_GATEWAY_NO_SUPERVISE",
                 "GATEWAY_MULTIPLEX_PROFILES"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(subprocess, "Popen", _NoProcess)

    # Seams of test_container_boot.py / test_gateway_s6_dispatch.py.
    monkeypatch.setattr("hermes_cli.container_boot._read_container_argv", lambda: ())
    rec = _RecordingS6()
    monkeypatch.setattr("hermes_cli.service_manager.detect_service_manager", lambda: "s6")
    monkeypatch.setattr("hermes_cli.service_manager.get_service_manager", lambda: rec)
    monkeypatch.setattr("hermes_cli.gateway._profile_suffix", lambda: "")
    execvp_calls: list[list[str]] = []
    monkeypatch.setattr(
        "hermes_cli.gateway.os.execvp", lambda _file, args: execvp_calls.append(list(args)),
    )
    block_calls: list[bool] = []
    monkeypatch.setattr(
        "hermes_cli.gateway._block_until_terminated", lambda: block_calls.append(True),
    )
    foreground: list[tuple] = []
    monkeypatch.setattr(
        "hermes_cli.gateway.run_gateway", lambda *a, **kw: foreground.append((a, kw)),
    )
    redirects: list[bool] = []
    real_redirect = gw._maybe_redirect_run_to_s6_supervision

    def _spy_redirect(args) -> bool:
        redirects.append(real_redirect(args))
        return redirects[-1]

    monkeypatch.setattr("hermes_cli.gateway._maybe_redirect_run_to_s6_supervision", _spy_redirect)

    # Boot, step 1 (cont-init): legacy migration, then reconcile (dry run).
    container_argv = [*argv_prefix, *cmd]
    migration = container_boot._maybe_migrate_legacy_gateway_run_state(
        home, container_argv=container_argv, dry_run=False,
    )
    actions = container_boot.reconcile_profile_gateways(
        hermes_home=home, scandir=scandir, dry_run=True, container_argv=container_argv,
    )
    # Boot, step 2: the command process, ``hermes gateway run ...`` (-> _cmd_run -> redirect).
    gw.gateway_command(_parse_gateway_args(cmd))

    started_by_reconcile = {f"gateway-{a.profile}" for a in actions if a.action == "started"}
    started_by_redirect = {name for verb, name in rec.calls if verb == "start"}
    supervised = sorted(started_by_reconcile | started_by_redirect)
    return {
        "supervised": supervised,
        "foreground_runs": len(foreground),
        "execvp": execvp_calls,
        "redirected": redirects == [True],
        "reconcile_actions": [(a.profile, a.prior_state, a.action) for a in actions],
        "owners": [*supervised, *(["foreground"] * len(foreground))],
        "migration": migration,
        "service_calls": list(rec.calls),
        "heartbeat_fallbacks": len(block_calls),
        "desired_state_after": container_boot._read_desired_state(home),
    }


@pytest.mark.parametrize("argv_prefix", ARGV_PREFIXES)
@pytest.mark.parametrize("home_kind", HOME_PARAMS)
def test_12_the_recorded_gateway_command_launches_exactly_one_gateway_owner_for_every_home(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    recorded_gateway_command: list[str],
    home_kind: str,
    argv_prefix: tuple[str, ...],
) -> None:
    """Empty, persisted running (running, degraded, draining, desired_state running) and explicitly stopped homes:
    one supervised gateway-default, and the command process is only the heartbeat."""
    result = boot_decisions(recorded_gateway_command, tmp_path, monkeypatch, home_kind, argv_prefix=argv_prefix)

    # What the boot decided for this home (the legacy migration only seeds the empty home).
    assert result["migration"] == ("running" if home_kind == "empty" else None)
    assert result["reconcile_actions"] == [EXPECTED_DEFAULT_SLOT[home_kind]]
    # (1) exactly one supervised service, started once by the redirect (idempotent with the reconcile).
    assert result["supervised"] == ["gateway-default"]
    assert result["service_calls"] == [("start", "gateway-default")]
    # (2) the command process runs no foreground gateway.
    assert result["foreground_runs"] == 0
    # (3) it hands over and is only the "sleep infinity" heartbeat (not the in-process fallback).
    assert result["redirected"] is True
    assert result["execvp"] == [["sleep", "infinity"]]
    assert result["heartbeat_fallbacks"] == 0
    assert result["owners"] == ["gateway-default"]


@pytest.mark.parametrize("home_kind", [pytest.param(kind, id=kind) for kind in PERSISTED_RUNNING])
def test_12b_control_the_round_1_command_is_two_gateway_owners_on_a_persisted_running_home(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, home_kind: str
) -> None:
    """Shows that the assertions above can fail: with --no-supervise the redirect opts out, so the boot starts
    gateway-default from the persisted state and the command process runs a second, foreground gateway."""
    result = boot_decisions(ROUND1_CMD, tmp_path, monkeypatch, home_kind)
    assert result["redirected"] is False
    assert result["execvp"] == []
    assert result["service_calls"] == []
    assert result["supervised"] == ["gateway-default"]
    assert result["foreground_runs"] == 1
    assert result["owners"] == ["gateway-default", "foreground"]
