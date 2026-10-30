"""Tests for reset_stack.sh: the refusals and a full run, with the Project and board tests (card P-2).

The stand-ins and helpers are in journeys_stack_harness.
"""

from __future__ import annotations

import io
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import tarfile
from pathlib import Path
from typing import NamedTuple

import pytest

from journeys_stack_harness import (
    ANCHOR,
    ANCHOR_WRITES,
    APP_C,
    BOARD_PROGRAM,
    BOARD_WRITE,
    BROWSER_C,
    CHECK,
    DASH_C,
    DECOY_CONTAINERS,
    DECOY_VOLUMES,
    ENV_SETTINGS,
    F_APP_SETTINGS,
    F_HEALTH_KIND,
    F_HERMES_HOME,
    GATE_C,
    HOME_V,
    Host,
    JOURNEYS,
    KEY_ENV,
    MODEL_TOKEN,
    PROD_C,
    PROJECT_CREATE,
    PROXY_C,
    PROXY_V,
    REAL_PYC,
    REPO,
    RESET,
    STACK_CONTAINERS,
    STACK_HOME,
    STACK_HOST,
    SURFACE_ENV,
    SharedReset,
    TOKENS_V,
    TOKEN_VARS,
    all_output,
    assert_nothing_changed,
    assert_secret_free,
    docker_by_hand,
    docker_names,
    docker_target,
    generated_secrets,
    hermes_gives,
    home_files,
    host,
    is_id_probe,
    mount_source,
    own_volume,
    parse_env_file,
    parse_template,
    reset_ok,
    resource_names_in,
    run_reset,
    set_app_env,
    set_hermes_home,
    sha_tree,
    shared_reset,
    snapshot,
    writer_calls,
)

pytestmark = pytest.mark.linux_only


# ------------------------------------------------------------------ reset_stack.sh: refusals


@pytest.mark.parametrize("stack_host", ["some-other-host.invalid", "", None])
def test_reset_refuses_the_wrong_host(host: Host, stack_host):
    before = snapshot(host)
    result = run_reset(host, JOURNEY_STACK_HOST=stack_host)
    assert result.returncode != 0
    assert_nothing_changed(host, before)
    assert STACK_HOST not in all_output(result)


@pytest.mark.parametrize("missing", STACK_CONTAINERS)
def test_reset_refuses_when_a_container_is_missing(host: Host, missing: str):
    host.update(lambda state: state["containers"].pop(missing))
    before = snapshot(host)
    result = run_reset(host)
    assert result.returncode != 0
    assert missing in all_output(result)
    assert_nothing_changed(host, before)


@pytest.mark.parametrize("stopped", [APP_C, DASH_C])
def test_reset_refuses_when_the_app_or_dashboard_is_not_running(host: Host, stopped: str):
    host.update(lambda state: state["containers"][stopped].update(running=False))
    before = snapshot(host)
    result = run_reset(host)
    if stopped == APP_C:  # a stopped owner app is no longer refused: the reset restarts it
        assert result.returncode == 0, result.stdout + result.stderr
        assert host.state()["containers"][APP_C]["running"]
        return
    assert result.returncode != 0
    assert_nothing_changed(host, before)


@pytest.mark.parametrize("name", list(SURFACE_ENV.values()) + list(KEY_ENV.values()))
def test_reset_refuses_a_token_path_outside_the_app_mount(host: Host, name: str):
    host.update(set_app_env(name, "/elsewhere/token"))
    before = snapshot(host)
    result = run_reset(host)
    assert result.returncode != 0
    assert_nothing_changed(host, before)


@pytest.mark.parametrize("value", ["/run/tokens/../models", "models", "/run/tokensX/models", "", None])
def test_reset_refuses_odd_token_paths(host: Host, value):
    host.update(set_app_env(SURFACE_ENV["models"], value))
    before = snapshot(host)
    result = run_reset(host)
    assert result.returncode != 0
    assert_nothing_changed(host, before)


@pytest.mark.parametrize("who", [APP_C, DASH_C])
def test_reset_refuses_when_a_container_lacks_the_token_volume(host: Host, who: str):
    def mutate(state: dict) -> None:
        box = state["containers"][who]
        box["mounts"] = [m for m in box["mounts"] if m["name"] != TOKENS_V]
    host.update(mutate)
    before = snapshot(host)
    result = run_reset(host)
    assert result.returncode != 0
    assert_nothing_changed(host, before)


@pytest.mark.parametrize("value", ["/somewhere/else", "/opt/data/", "/opt/dat"])
@pytest.mark.parametrize("who", [DASH_C, GATE_C])
def test_reset_refuses_when_hermes_home_is_not_the_home_mount(host: Host, who: str, value: str):
    host.update(set_hermes_home(who, value))
    before = snapshot(host)
    result = run_reset(host)
    assert result.returncode != 0
    assert who in result.stderr and "nothing was changed" in result.stderr
    assert_nothing_changed(host, before)


def test_reset_refuses_a_gateway_home_that_is_another_directory_with_a_stale_state(host: Host):
    other = host.root / "gateway-own-filesystem" / "data"
    shutil.copytree(STACK_HOME, other)
    (other / ".env").write_text("API_SERVER_KEY=stale-gateway-key-0123456789abcdef\n")
    (other / "stale-state.db").write_text("sentinel rows that must stay\n")
    host.update(set_hermes_home(GATE_C, str(other)))
    before, stale = snapshot(host), sha_tree(other)
    result = run_reset(host)
    assert result.returncode != 0
    assert GATE_C in result.stderr and "nothing was changed" in result.stderr
    assert_nothing_changed(host, before)
    assert sha_tree(other) == stale and (other / "stale-state.db").read_text() == "sentinel rows that must stay\n"


def test_reset_reads_hermes_home_of_both_containers_with_one_inspect_format(shared_reset: SharedReset):
    calls = shared_reset.host.calls("docker")
    read = [c["argv"][-1] for c in calls if c["argv"][0] == "inspect" and F_HERMES_HOME in c["argv"]]
    assert sorted(read) == sorted([DASH_C, GATE_C])
    assert not [c for c in calls if c["argv"][0] == "exec" and "HERMES_HOME" in c["argv"]]


def test_reset_does_not_refuse_a_stopped_gateway_whose_hermes_home_is_its_mount(host: Host):
    host.update(lambda state: state["containers"][GATE_C].update(running=False))
    result = run_reset(host)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "HERMES_HOME" not in all_output(result)


def unreadable(who: str, command: str, text: str):
    def mutate(state: dict) -> None:
        state["containers"][who].setdefault("answers", {})[command] = text
    return mutate


@pytest.mark.parametrize("who, command, text", [
    (APP_C, "id -u", "node"), (APP_C, "id -u", "0"), (APP_C, "id -u", ""),
    (APP_C, "id -g", "node"), (APP_C, "id -g", "0"), (APP_C, "id -g", ""),
    (DASH_C, "stat -c %u:%g /opt/data", "root:root"), (DASH_C, "stat -c %u:%g /opt/data", "0:0"),
    (DASH_C, "stat -c %u:%g /opt/data", "10000:0"), (DASH_C, "stat -c %u:%g /opt/data", "0:10000"),
    (DASH_C, "stat -c %u:%g /opt/data", ""), (DASH_C, "stat -c %u:%g /opt/data", "10000"),
], ids=lambda value: str(value).replace(" ", "_") or "empty")
def test_reset_refuses_an_identity_it_cannot_read_as_two_positive_numbers(host: Host, who, command, text):
    host.update(unreadable(who, command, text))
    before = snapshot(host)
    result = run_reset(host)
    assert result.returncode != 0
    assert who in result.stderr and "nothing was changed" in result.stderr
    assert_nothing_changed(host, before)


def test_reset_refuses_when_the_gateway_lacks_the_home_volume(host: Host):
    host.update(lambda state: state["containers"][GATE_C].update(mounts=[]))
    before = snapshot(host)
    result = run_reset(host)
    assert result.returncode != 0
    assert_nothing_changed(host, before)


# ------------------------------------------------------------------ reset_stack.sh: a full run


def test_reset_makes_a_fresh_home_and_token_folder(shared_reset: SharedReset):
    host, result, untouched = shared_reset.host, shared_reset.result, shared_reset.untouched
    home, tokens = host.vol(HOME_V), host.vol(TOKENS_V)

    assert not (home / "stale").exists() and not (home / "keep.txt").exists()
    assert not list(home.rglob(ENV_SETTINGS))
    for path in STACK_HOME.rglob("*"):
        if path.is_file() and path.name != ENV_SETTINGS:
            assert (home / path.relative_to(STACK_HOME)).read_bytes() == path.read_bytes()
    assert stat.S_IMODE(home.stat().st_mode) & 0o077 == 0

    allowed = set(parse_template()["home-env"])
    keys = {}
    for profile, path in home_files(host).items():
        assert path.is_file() and not path.is_symlink()
        assert stat.S_IMODE(path.stat().st_mode) == 0o600
        names, values = parse_env_file(path)
        assert set(names) <= allowed and names.count("API_SERVER_KEY") == 1
        assert len(values["API_SERVER_KEY"]) >= 16
        assert values["CLAUDE_CODE_OAUTH_TOKEN"] == MODEL_TOKEN
        keys[profile] = values["API_SERVER_KEY"]
    assert keys["default"] != keys["planner"]

    assert stat.S_IMODE(tokens.stat().st_mode) == 0o700
    assert {p.name for p in tokens.iterdir()} == set(SURFACE_ENV) | {"executor", "planner"}
    for path in tokens.iterdir():
        info = path.lstat()
        assert stat.S_ISREG(info.st_mode) and stat.S_IMODE(info.st_mode) == 0o600
        assert 0 < info.st_size <= 16 * 1024
    for surface in SURFACE_ENV:
        assert (tokens / surface).read_text().startswith(f"tok-{surface}-")
    assert (tokens / "executor").read_text().strip() == keys["default"]
    assert (tokens / "planner").read_text().strip() == keys["planner"]

    assert {v: sha_tree(host.vol(v)) for v in untouched} == untouched
    assert not host.state().get("created_volumes")
    assert_secret_free(host, result, generated_secrets(host))


def test_reset_can_run_twice_and_makes_new_values(shared_reset: SharedReset, tmp_path: Path):
    first = generated_secrets(shared_reset.host)
    assert_secret_free(shared_reset.host, shared_reset.result, first)
    host = Host.copy_of(shared_reset.host, tmp_path / "host")
    result = reset_ok(host)
    second = generated_secrets(host)
    assert len(second) == len(first)
    assert not (set(first) - {MODEL_TOKEN}) & set(second)
    for path in home_files(host).values():
        assert parse_env_file(path)[0].count("API_SERVER_KEY") == 1
    assert_secret_free(host, result, first + second)


def test_reset_order_of_stops_writes_and_starts(shared_reset: SharedReset):
    host = shared_reset.host
    steps = []
    for call in host.calls("docker"):
        sub, target = docker_target(call)
        if sub in ("stop", "start", "restart"):
            steps.append((sub, target))
        elif sub == "run":
            steps.append(("helper", mount_source(call)))
        elif sub == "exec" and "--user" in call["argv"]:
            steps.append(("write", target))
    assert set(steps[:2]) == {("stop", GATE_C), ("stop", DASH_C)}
    assert set(steps[2:4]) == {("helper", HOME_V), ("helper", TOKENS_V)}
    assert steps[4] == ("start", DASH_C)
    # the anchor Project and its board, then 5 tokens, 2 API server keys, 2 model tokens
    assert steps[5:-6] == [("write", DASH_C)] * (ANCHOR_WRITES + 9)
    assert steps[-6] == ("helper", TOKENS_V)  # the token files go to the owner app before it restarts
    assert steps[-5:-2] == [("restart", APP_C), ("restart", DASH_C), ("start", GATE_C)]
    assert set(steps[-2:]) == {("start", PROXY_C), ("start", BROWSER_C)}


def test_reset_helpers_have_the_agreed_shape(shared_reset: SharedReset):
    host = shared_reset.host
    image = host.state()["containers"][DASH_C]["image"]
    runs = [c for c in host.calls("docker") if c["argv"][0] == "run"]
    assert sorted(mount_source(c) for c in runs) == sorted([HOME_V, TOKENS_V, TOKENS_V])
    for call in runs:
        argv, source = call["argv"], mount_source(call)
        flags = ["--rm", "-i"] if source == HOME_V else ["--rm"]
        assert argv[:-1] == ["run", *flags, "--network", "none", "--user", "0", "--entrypoint", "sh",
                             "--mount", f"type=volume,source={source},target=/volume", image, "-c"]
        script = argv[-1]
        if source == HOME_V:
            assert "find /volume -mindepth 1 -delete" in script
            assert "chown -R 10000:10000 /volume" in script and "chmod go-rwx /volume" in script
            with tarfile.open(fileobj=io.BytesIO(call["stdin"])) as archive:
                members = {m.name.removeprefix("./") for m in archive.getmembers() if m.isfile()}
            assert members == {p.relative_to(STACK_HOME).as_posix() for p in STACK_HOME.rglob("*")
                               if p.is_file()}
        elif "-delete" in script:
            assert "chown 10000:10000 /volume" in script and "chmod 700 /volume" in script
            assert not call["stdin"]
        else:
            assert script.splitlines()[-1] == "chown -R 1000:1000 /volume"
            assert not call["stdin"]


def test_reset_writers_run_as_the_platform_identity_with_values_only_on_stdin(shared_reset: SharedReset):
    host = shared_reset.host
    writers = [c for c in host.calls("docker") if docker_target(c)[0] == "exec" and "--user" in c["argv"]]
    assert len(writers) == ANCHOR_WRITES + 9
    for call in writers:
        argv = call["argv"]
        assert argv[argv.index("--user") + 1] == "10000:10000"
        assert docker_target(call)[1] == DASH_C
    issues = [c for c in writers if "kanban-workspace-token" in c["argv"]]
    assert sorted(c["argv"][c["argv"].index("--surface") + 1] for c in issues) == sorted(SURFACE_ENV)
    for call in issues:
        argv = call["argv"]
        assert not [a for a in argv if a.startswith("--ttl")]
        assert argv[argv.index("--out") + 1] == "/tokens/" + argv[argv.index("--surface") + 1]
    carrying = [c for c in writers if c["stdin"] and MODEL_TOKEN.encode() in c["stdin"]]
    assert len(carrying) == 2 and all("-i" in c["argv"] for c in carrying)


# --- the Project and board that the owner app reads

# The platform's own registry and kanban code, run on the home that the reset left: what is there, as the platform reads it.
ANCHOR_READ = (
    "import json, sys\n"
    "from hermes_cli import kanban_db, projects_db\n"
    "project_slug, board_slug = sys.argv[1:3]\n"
    "with projects_db.connect_closing() as conn:\n"
    "    project = projects_db.get_project(conn, project_slug)\n"
    "meta = kanban_db.read_board_metadata(board_slug)\n"
    "print(json.dumps({'project': project and {'id': project.id, 'slug': project.slug, 'board': project.board_slug},\n"
    "                  'board_exists': kanban_db.board_exists(board_slug),\n"
    "                  'meta': {'slug': meta.get('slug'), 'project_id': meta.get('project_id')}}))\n")


def bound_hermes_source() -> tuple[str, str]:
    """The Project slug and the board slug that the owner app's settings bind: what the app reads, and through which board."""
    lines = (JOURNEYS / "stack-app.settings").read_text().splitlines()
    settings = dict(line.partition("=")[::2] for line in lines if line.strip() and not line.startswith("#"))
    (binding,) = json.loads(settings["RAPHAEL_PROJECT_BINDINGS_JSON"])
    hermes = binding["sources"]["hermes"]
    return hermes["projectSlug"], hermes["boardSlug"]


def read_anchor(host: Host, scratch: Path) -> dict:
    """The bound Project and board as the platform reads them from a copy of the home that the reset left."""
    project_slug, board_slug = bound_hermes_source()
    home = scratch / "home-copy"
    shutil.copytree(host.vol(HOME_V), home, symlinks=True)
    REAL_PYC.mkdir(parents=True, exist_ok=True)
    env = {"PATH": os.environ["PATH"], "HOME": str(scratch), "HERMES_HOME": str(home), "PYTHONPATH": str(REPO),
           "PYTHONPYCACHEPREFIX": str(REAL_PYC), "AWS_EC2_METADATA_DISABLED": "true"}
    done = subprocess.run([sys.executable, "-c", ANCHOR_READ, project_slug, board_slug], env=env, cwd=scratch,
                          capture_output=True, text=True)
    assert done.returncode == 0, done.stdout + done.stderr
    return json.loads(done.stdout.splitlines()[-1])


def test_reset_makes_the_project_the_app_binds_and_a_board_that_names_that_projects_id(
        shared_reset: SharedReset, tmp_path: Path):
    project_slug, board_slug = bound_hermes_source()
    seen = read_anchor(shared_reset.host, tmp_path)
    assert seen["project"] is not None, f"the reset made no Project with the slug {project_slug}"
    assert seen["project"]["slug"] == project_slug and seen["project"]["board"] == board_slug
    assert seen["board_exists"], f"the reset made no board {board_slug}"
    assert seen["meta"]["slug"] == board_slug
    assert seen["meta"]["project_id"] == seen["project"]["id"]


def test_reset_makes_the_project_and_board_before_it_issues_any_token(shared_reset: SharedReset):
    writers = writer_calls(shared_reset.host)
    issues = [i for i, call in enumerate(writers) if "kanban-workspace-token" in call["argv"]]
    assert len(issues) == len(SURFACE_ENV)
    assert min(issues) >= ANCHOR_WRITES


def board_root_is_a_file(host: Host) -> None:
    """The platform's kanban folder is pinned to a file, so that no board can be written (the Project registry is not affected)."""
    blocker = host.root / "not-a-folder"
    blocker.write_text("a file where the platform expects the kanban folder\n")
    host.update(lambda state: state["containers"][DASH_C]["env"].append(f"HERMES_KANBAN_HOME={blocker}"))


def project_command_gives(mode: str):
    return lambda host: host.update(lambda state: state["hermes"].update(project_fail=mode))


# One operation broken per case: how to break it, and whether the Project was made before the reset stopped.
ANCHOR_BREAKS = {
    "project-command-refused": (project_command_gives("refused"), False),
    "project-command-makes-nothing": (project_command_gives("silent"), False),
    "board-cannot-be-written": (board_root_is_a_file, True),
}


@pytest.mark.parametrize("name", ANCHOR_BREAKS)
def test_reset_stops_before_any_token_when_the_project_or_its_board_cannot_be_made(host: Host, tmp_path: Path, name):
    break_it, project_made = ANCHOR_BREAKS[name]
    break_it(host)
    hermes_gives(real_platform=True)(host)  # this test reads what the platform's code wrote, or could not write
    result = run_reset(host)
    assert result.returncode != 0, result.stdout + result.stderr
    docker = host.calls("docker")
    assert not [c for c in docker if "kanban-workspace-token" in c["argv"]]
    done = [docker_target(c) for c in docker]
    assert not [step for step in done if step[0] == "restart"] and ("start", GATE_C) not in done
    seen = read_anchor(host, tmp_path)
    assert (seen["project"] is not None) == project_made
    assert not seen["board_exists"] and seen["meta"]["project_id"] is None


# --- the stand-ins start the platform's code for the Project and board calls only where a test reads what it wrote

AS_PLATFORM = ["--user", "10000:10000"]
OTHER_PROGRAM = "import sys\nsys.exit(42)\n"


def venv_starts(host: Host) -> list[str]:
    """Each start of the venv interpreter wrapper, in order: a hermes launcher by its path, a program on standard input as -."""
    log = host.root / "interp.log"
    return log.read_text().splitlines() if log.exists() else []


def create_project(name: str = "Raphael Workspace", slug: str = ANCHOR) -> list[str]:
    return ["hermes", "project", "create", name, "--slug", slug, "--board", ANCHOR]


def test_an_ordinary_reset_records_the_project_and_board_calls_whole_and_starts_no_platform_process_for_them(host: Host):
    reset_ok(host)
    project, board = writer_calls(host)[:ANCHOR_WRITES]
    assert project["argv"] == ["exec", *AS_PLATFORM, DASH_C, *PROJECT_CREATE] and project["stdin"] is None
    assert board["argv"] == ["exec", "-i", *AS_PLATFORM, DASH_C, *BOARD_WRITE]
    assert board["stdin"] == BOARD_PROGRAM.encode()
    assert project["rc"] == 0 and board["rc"] == 0
    # the five token issues start the hermes launcher; the Project call starts none, and no program is run from standard input
    assert venv_starts(host) == [str(host.root / "ctr/venv/hermes")] * len(SURFACE_ENV)


def test_the_shared_reset_that_the_anchor_tests_read_started_the_platforms_code_for_both_calls(shared_reset: SharedReset):
    launcher = str(shared_reset.host.root / "ctr/venv/hermes")
    assert venv_starts(shared_reset.host) == [launcher, "-", *[launcher] * len(SURFACE_ENV)]


PROJECT_NEAR_MISSES = {
    "another-name": [*AS_PLATFORM, DASH_C, *create_project(name="Other Workspace")],
    "another-slug": [*AS_PLATFORM, DASH_C, *create_project(slug="other-workspace")],
    "an-extra-word": [*AS_PLATFORM, DASH_C, *PROJECT_CREATE, "extra"],
    "another-container": [*AS_PLATFORM, GATE_C, *PROJECT_CREATE],
    "with-the-input-option": ["-i", *AS_PLATFORM, DASH_C, *PROJECT_CREATE],
}
BOARD_NEAR_MISSES = {
    "another-shell-text": ("-i", DASH_C, ["sh", "-c", "exit 42", "sh"], BOARD_PROGRAM),
    "another-program": ("-i", DASH_C, BOARD_WRITE, OTHER_PROGRAM),
    "another-container": ("-i", GATE_C, BOARD_WRITE, OTHER_PROGRAM),
}


@pytest.mark.parametrize("case", PROJECT_NEAR_MISSES)
def test_a_project_call_that_is_not_the_resets_is_still_run_by_the_stand_in(host: Host, case: str):
    project_command_gives("refused")(host)  # the platform's own command is never reached: hermes answers 2 at once
    words = PROJECT_NEAR_MISSES[case]
    done = docker_by_hand(host, "exec", *words, stdin="" if "-i" in words else None)
    assert done.returncode == 2 and "project command failed" in done.stderr, done.stdout + done.stderr


@pytest.mark.parametrize("case", BOARD_NEAR_MISSES)
def test_a_board_write_that_is_not_the_resets_is_still_run_by_the_stand_in(host: Host, case: str):
    option, container, command, program = BOARD_NEAR_MISSES[case]
    done = docker_by_hand(host, "exec", option, *AS_PLATFORM, container, *command, stdin=program)
    assert done.returncode == 42, done.stdout + done.stderr


# --- the platform's runtime identity and the owner app's identity are two different things

IDENTITY_CASES = [
    pytest.param("shared_reset", (1000, 1000), (10000, 10000), id="platform-10000-app-1000"),
    pytest.param("reset_remapped", (1000, 1000), (4242, 4242), id="platform-4242-app-1000"),
    pytest.param("reset_equal", (1000, 1000), (1000, 1000), id="platform-1000-app-1000"),
]


def reset_with_identities(tmp_path_factory, app: tuple[int, int], platform: tuple[int, int]) -> SharedReset:
    first = Host(tmp_path_factory.mktemp("identities") / "host")
    first.set_identities(app, platform)
    return SharedReset(first, run_reset(first), {}, {})


@pytest.fixture(scope="module")
def reset_remapped(tmp_path_factory) -> SharedReset:
    return reset_with_identities(tmp_path_factory, (1000, 1000), (4242, 4242))


@pytest.fixture(scope="module")
def reset_equal(tmp_path_factory) -> SharedReset:
    return reset_with_identities(tmp_path_factory, (1000, 1000), (1000, 1000))


def finished(request, fixture: str) -> SharedReset:
    shared = request.getfixturevalue(fixture)
    assert shared.result.returncode == 0, shared.result.stdout + shared.result.stderr
    return shared


@pytest.mark.parametrize("fixture, app, platform", IDENTITY_CASES)
def test_reset_succeeds_whether_the_platform_and_app_identities_differ_or_match(request, fixture, app, platform):
    assert "the stack is reset" in finished(request, fixture).result.stdout


@pytest.mark.parametrize("fixture, app, platform", IDENTITY_CASES)
def test_reset_gives_the_token_files_to_the_app_identity_in_an_owner_only_folder(request, fixture, app, platform):
    host = finished(request, fixture).host
    tokens = host.vol(TOKENS_V)
    assert host.owner_of(tokens) == app and stat.S_IMODE(tokens.stat().st_mode) == 0o700
    names = set(SURFACE_ENV) | {"executor", "planner"}
    assert {p.name for p in tokens.iterdir()} == names
    for name in names:
        info = (tokens / name).lstat()
        assert host.owner_of(tokens / name) == app
        assert stat.S_ISREG(info.st_mode) and stat.S_IMODE(info.st_mode) == 0o600 and 0 < info.st_size <= 16 * 1024


@pytest.mark.parametrize("fixture, app, platform", IDENTITY_CASES)
def test_reset_leaves_the_home_and_its_files_with_the_platform_identity(request, fixture, app, platform):
    host = finished(request, fixture).host
    home = host.vol(HOME_V)
    for path in [home, *home.rglob("*")]:
        assert host.owner_of(path) == platform, path
    assert stat.S_IMODE(home.stat().st_mode) & 0o077 == 0
    for path in home_files(host).values():
        assert stat.S_IMODE(path.stat().st_mode) == 0o600
    for path in (home / "config.yaml", home / "profiles/raphael-planner/config.yaml"):
        assert path.is_file()


@pytest.mark.parametrize("fixture, app, platform", IDENTITY_CASES)
def test_reset_runs_every_platform_command_as_the_platform_identity_and_nothing_as_the_app(
        request, fixture, app, platform):
    host = finished(request, fixture).host
    calls, spec = host.calls("docker"), "%d:%d" % platform
    writers = writer_calls(host)
    assert len(writers) == ANCHOR_WRITES + 9
    for call in writers:
        assert call["argv"][call["argv"].index("--user") + 1] == spec
    for call in calls:
        argv = call["argv"]
        if argv[0] == "run":
            assert argv[argv.index("--user") + 1] == "0"
        elif argv[0] == "exec" and "--user" not in argv:
            assert argv[2:4] in (["id", "-u"], ["id", "-g"], ["printenv", argv[3]]) or argv[2:4] == ["stat", "-c"]
    app_spec = "%d:%d" % app
    named = [c for c in calls if re.search(rf"(?<![0-9]){app_spec}(?![0-9])", "\x00".join(c["argv"]))]
    if app == platform:
        return
    assert len(named) == 1 and named[0]["argv"][0] == "run" and mount_source(named[0]) == TOKENS_V
    assert named[0]["argv"][-1].splitlines()[-1] == f"chown -R {app_spec} /volume"


def test_reset_reads_both_identities_as_numbers_before_it_changes_anything(shared_reset: SharedReset):
    argvs = [c["argv"] for c in shared_reset.host.calls("docker")]
    first_change = next(i for i, argv in enumerate(argvs) if argv[0] in ("stop", "start", "restart", "run"))
    seen = argvs[:first_change]
    assert ["exec", APP_C, "id", "-u"] in seen and ["exec", APP_C, "id", "-g"] in seen
    assert ["exec", DASH_C, "stat", "-c", "%u:%g", "/opt/data"] in seen


def test_reset_reads_a_running_owner_app_only_by_docker_exec_before_it_changes_anything(shared_reset: SharedReset):
    argvs = [c["argv"] for c in shared_reset.host.calls("docker")]
    first_change = next(i for i, argv in enumerate(argvs) if argv[0] in ("stop", "start", "restart", "run"))
    into_app = [argv for argv in argvs if argv[:2] == ["exec", APP_C]]
    assert into_app == [argv for argv in argvs[:first_change] if argv[:2] == ["exec", APP_C]]
    assert into_app == [["exec", APP_C, "id", "-u"], ["exec", APP_C, "id", "-g"],
                        *(["exec", APP_C, "printenv", name] for name in TOKEN_VARS)]
    assert not [argv for argv in argvs if argv[0] == "inspect" and F_APP_SETTINGS in argv]
    assert not [argv for argv in argvs if is_id_probe(argv)]


def issue_by_hand(host: Host, user: str) -> subprocess.CompletedProcess:
    return docker_by_hand(host, "exec", "--user", user, DASH_C, "hermes", "kanban-workspace-token", "issue",
                          "--surface", "workspace", "--out", "/tokens/workspace-new")


def test_the_fake_issues_a_token_for_the_user_that_owns_the_folder(host: Host):
    own_volume(host, TOKENS_V, (10000, 10000), 0o700)
    result = issue_by_hand(host, "10000:10000")
    assert result.returncode == 0, result.stdout + result.stderr
    made = host.vol(TOKENS_V) / "workspace-new"
    assert stat.S_IMODE(made.stat().st_mode) == 0o600 and host.owner_of(made) == (10000, 10000)


@pytest.mark.parametrize("user, owner, mode, message", [
    ("1000:1000", (1000, 1000), 0o700, "Permission denied"),
    ("10000:10000", (1000, 1000), 0o700, "is not owned by the current user"),
    ("10000:10000", (10000, 10000), 0o750, "is not owner-only"),
    ("1000", (1000, 0), 0o700, "Permission denied"),
], ids=["home-of-another-user", "folder-of-another-user", "folder-not-owner-only", "uid-only"])
def test_the_fake_refuses_a_token_for_the_wrong_user_like_the_real_command(host: Host, user, owner, mode, message):
    own_volume(host, TOKENS_V, owner, mode)
    result = issue_by_hand(host, user)
    assert result.returncode == 2 and result.stdout.startswith("error: ") and message in result.stdout
    assert not (host.vol(TOKENS_V) / "workspace-new").exists()


def test_the_fake_stops_a_writer_that_runs_as_a_user_without_access(host: Host):
    result = docker_by_hand(host, "exec", "--user", "1000:1000", DASH_C, "sh", "-c", 'printf x >> "$1"', "sh",
                            "/opt/data/.env")
    assert result.returncode == 2 and "Permission denied" in result.stderr
    assert not (host.vol(HOME_V) / ".env").exists()


def test_the_fake_boot_hook_gives_the_listed_parts_of_the_home_to_the_runtime_account(host: Host):
    home = host.vol(HOME_V)
    (home / "profiles/p").mkdir(parents=True)
    for name in (".env", "config.yaml", "profiles/p/.env"):
        (home / name).write_text("x\n")
    table = host.owners()
    for path in [home, *home.rglob("*")]:
        table[str(path.relative_to(host.root / "volumes"))] = [1000, 1000]
    (host.root / "owners.json").write_text(json.dumps(table))
    assert docker_by_hand(host, "start", DASH_C).returncode == 0
    for name in ("", ".env", "config.yaml", "profiles", "profiles/p", "profiles/p/.env"):
        assert host.owner_of(home / name) == (10000, 10000), name
    assert host.owner_of(home / "keep.txt") == (1000, 1000)


def test_reset_names_only_template_containers_and_volumes(shared_reset: SharedReset):
    host = shared_reset.host
    template = parse_template()
    allowed = set(template["containers"]) | set(template["volumes"])
    calls = host.calls("docker")
    found = set().union(*(docker_names(c) for c in calls))
    assert found <= allowed
    assert not found & (set(DECOY_CONTAINERS) | set(DECOY_VOLUMES) | {PROD_C})
    for call in calls:
        argv = call["argv"]
        assert argv[0] in ("inspect", "start", "restart", "stop", "exec", "run")
        assert not {"-v", "--volume", "--volumes-from"} & set(argv)
        assert not [part for part in argv if f"source={PROXY_V}" in part]
    proxy_calls = {docker_target(c)[0] for c in calls if docker_target(c)[1] == PROXY_C}
    assert proxy_calls == {"inspect", "start"}, "reset may only look at and start the proxy container"


def test_reset_scripts_carry_no_container_or_volume_names():
    # The one name that is not a docker resource: the reset spells the anchor slug once, as a plain assignment.
    anchor_assignment = re.compile(r"""\s*[a-z_]+=["']?%s["']?\s*""" % re.escape(ANCHOR))
    for script in (RESET, CHECK):
        for number, line in enumerate(script.read_text().splitlines(), 1):
            if not line.lstrip().startswith("#") and not (script == RESET and anchor_assignment.fullmatch(line)):
                assert not resource_names_in(line), f"{script.name}:{number} hard-codes a name"


def test_reset_polls_every_two_seconds_until_healthy(host: Host):
    host.update(lambda state: state["containers"][DASH_C].update(health_delay=3))
    reset_ok(host)
    polls = [c["argv"] for c in host.calls("sleep")]
    assert 3 <= len(polls) <= 60 and all(argv == ["2"] for argv in polls)


@pytest.mark.parametrize("who, trouble", [
    (DASH_C, {"health": "unhealthy"}),
    (DASH_C, {"health_delay": 10**6}),
    (GATE_C, {"stays_down": True}),
    (PROXY_C, {"stays_down": True, "running": False}),
    (BROWSER_C, {"stays_down": True, "running": False}),
], ids=["unhealthy", "never-healthy", "gateway-stays-down", "proxy-stays-down", "browser-stays-down"])
def test_reset_gives_up_after_120_seconds_naming_the_container(host: Host, who: str, trouble: dict):
    host.update(lambda state: state["containers"][who].update(trouble))
    result = run_reset(host)
    assert result.returncode != 0
    assert who in all_output(result)
    assert 55 <= len(host.calls("sleep")) <= 60, "120 seconds in total, polled every 2 seconds"


def set_healthcheck(who: str, test):
    def mutate(state: dict) -> None:
        if test == "absent":
            state["containers"][who].pop("healthcheck", None)
        else:
            state["containers"][who]["healthcheck"] = test
    return mutate


@pytest.mark.parametrize("who", STACK_CONTAINERS)
def test_reset_refuses_a_container_without_a_health_check_before_it_changes_anything(host: Host, who: str):
    host.update(set_healthcheck(who, "absent"))
    before = snapshot(host)
    result = run_reset(host)
    assert result.returncode != 0
    assert who in result.stderr and "health check" in result.stderr
    assert "must create it with a health check" in result.stderr and "nothing was changed" in result.stderr
    assert_nothing_changed(host, before)
    assert host.calls("sleep") == []


@pytest.mark.parametrize("test, how", [(["NONE"], "kind-none"), ([], "unreadable")])
def test_reset_refuses_a_health_check_that_is_none_or_cannot_be_read(host: Host, test: list, how: str):
    host.update(set_healthcheck(PROXY_C, test))
    before = snapshot(host)
    result = run_reset(host)
    assert result.returncode != 0
    assert PROXY_C in result.stderr and "must create it with a health check" in result.stderr
    assert_nothing_changed(host, before)


def test_reset_reads_the_health_check_kind_of_all_five_containers_before_it_changes_anything(shared_reset: SharedReset):
    calls = shared_reset.host.calls("docker")
    reads = [i for i, c in enumerate(calls) if c["argv"][0] == "inspect" and F_HEALTH_KIND in c["argv"]]
    assert sorted(calls[i]["argv"][-1] for i in reads) == sorted(STACK_CONTAINERS)
    changing = [i for i, c in enumerate(calls)
                if docker_target(c)[0] not in ("inspect", "exec") or "-i" in c["argv"] or "--user" in c["argv"]]
    assert changing and max(reads) < min(changing)


def test_reset_invents_no_probe_of_its_own(shared_reset: SharedReset):
    docker = shared_reset.host.calls("docker")
    assert shared_reset.host.calls("curl") == []
    entered = {docker_target(c)[1] for c in docker if docker_target(c)[0] == "exec"}
    assert entered <= {APP_C, DASH_C}, "no command is run in the proxy, the gateway or the browser to probe it"
    words = {word for c in docker for word in c["argv"]}
    assert not words & {"curl", "wget", "nc", "ncat", "telnet", "ping"}
    assert not [word for word in words if "://" in word]


class PendingReset(NamedTuple):
    host: Host
    result: subprocess.CompletedProcess


@pytest.fixture(scope="module")
def pending_reset(tmp_path_factory) -> PendingReset:
    """One reset in which four containers never answer: two starting, one without a status, one unhealthy."""
    first = Host(tmp_path_factory.mktemp("pending") / "host")

    def never_answer(state):
        for who in (APP_C, BROWSER_C):
            state["containers"][who].update(health_delay=10**6, starting_polls=10**6)
        state["containers"][GATE_C].pop("health")
        state["containers"][PROXY_C].update(health="unhealthy")

    first.update(never_answer)
    return PendingReset(first, run_reset(first))


def pending_names(shared: PendingReset) -> str:
    """The containers that the refusal names after its budget of 120 seconds."""
    assert shared.result.returncode != 0
    assert "120 seconds" in shared.result.stderr
    return shared.result.stderr.split("120 seconds", 1)[-1]


@pytest.mark.parametrize("who", [APP_C, BROWSER_C])
def test_reset_does_not_call_a_container_ready_while_its_health_is_starting(pending_reset: PendingReset, who: str):
    assert who in pending_names(pending_reset)


def test_reset_does_not_call_a_container_ready_when_its_health_status_is_missing(pending_reset: PendingReset):
    assert GATE_C in pending_names(pending_reset)


def test_reset_does_not_call_a_container_ready_while_its_health_is_unhealthy(pending_reset: PendingReset):
    assert PROXY_C in pending_names(pending_reset)


def test_reset_names_no_container_that_answered_when_it_gives_up(pending_reset: PendingReset):
    assert DASH_C not in pending_names(pending_reset)


def test_reset_gives_pending_containers_120_seconds_in_total_polled_every_2_seconds(pending_reset: PendingReset):
    assert 55 <= len(pending_reset.host.calls("sleep")) <= 60


@pytest.mark.parametrize("who", [GATE_C, PROXY_C, BROWSER_C])
def test_reset_fails_naming_a_container_that_cannot_start(host: Host, who: str):
    host.update(lambda state: state["containers"][who].update(running=False, fails_to_start=True))
    result = run_reset(host)
    assert result.returncode != 0
    assert who in all_output(result)


@pytest.mark.parametrize("how", ["unset", "missing", "empty", "blank"])
def test_reset_skips_the_model_token_with_one_value_free_line(host: Host, how: str):
    env: dict = {}
    if how == "missing":
        env["JOURNEY_MODEL_TOKEN_FILE"] = str(host.root / "no-such-token-file")
    elif how in ("empty", "blank"):
        env["JOURNEY_MODEL_TOKEN_FILE"] = str(host.model_token_file("" if how == "empty" else " \n\n"))
    result = run_reset(host, token=False, **env)
    assert result.returncode == 0, result.stdout + result.stderr
    for path in home_files(host).values():
        assert "CLAUDE_CODE_OAUTH_TOKEN" not in path.read_text()
    skips = [line for line in all_output(result).splitlines() if "skip" in line.lower()]
    assert len(skips) == 1 and "JOURNEY_MODEL_TOKEN_FILE" in skips[0]
    if env:
        assert env["JOURNEY_MODEL_TOKEN_FILE"] not in all_output(result)
    # the anchor Project and its board, 5 tokens and 2 API server keys; no model token writes
    assert len(writer_calls(host)) == ANCHOR_WRITES + 7


def test_reset_stops_when_a_token_cannot_be_issued(host: Host):
    host.update(lambda state: state["hermes"].update(issue_fail="connections"))
    result = run_reset(host)
    assert result.returncode != 0
    assert "connections" in all_output(result)
    done = [docker_target(c) for c in host.calls("docker")]
    assert not [step for step in done if step[0] == "restart"]
    assert ("start", GATE_C) not in done
    assert_secret_free(host, result, [MODEL_TOKEN])


def test_reset_prints_no_skip_line_when_the_model_token_is_written(shared_reset: SharedReset):
    assert "skip" not in all_output(shared_reset.result).lower()
    assert len(writer_calls(shared_reset.host)) == ANCHOR_WRITES + 9


def test_reset_leaves_the_repository_files_alone(shared_reset: SharedReset):
    assert sha_tree(STACK_HOME) == shared_reset.repo_before
