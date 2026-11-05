"""Tests for scripts/journeys/stack_up.sh (card 2 of the journeys stack plan, rounds 2 to 5), part 1 of 3.

The stand-ins and helpers are in journeys_stack_up_support (the description of the stand-in docker is there). The
three test_journeys_stack_up*.py files split the tests by run time; part 1 holds the plan's tests (test_1 to test_8
except test_4b and test_4c) and test_9b, test_11 and test_extra_container_that_never_answers_..., part 2 the
namespace rule (test_16 to test_20), test_4b, test_4c, test_13 and test_15, part 3 the rest.

Tests test_1_... to test_8_... are the eight tests of the plan (test_1 and test_2 cover the round 2 rules);
test_9_... onwards and test_extra_... are additions. Round 3 adds test_13 to test_15 (stopped kept containers) and
test_16 to test_20 (the namespace rule: the gateway is current only when its network mode names the owner app's
current container id and its State.StartedAt is strictly later than the owner app's). Round 4 adds test_21 to
test_25 (an identity read that fails for any reason but Docker's own "no such container" stops converge before any
change with one fixed line; Docker is given an id only from a read of the same run that showed the label). Round 5
adds test_26 to test_29 (the label is compared by Docker's template on the raw value, so the values stack_up plus one
newline, stack_up plus two newlines, stack_ plus a NUL byte plus up and stack_upx are foreign: converge and versions
refuse them for all five roles of the template without a changing call, the exact value is accepted, and test_29
checks the stand-in's own answer to the identity read).
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

from journeys_stack_up_support import (
    ANCHOR,
    ANSWERS_LINE,
    APP_C,
    AS_PLATFORM,
    BOARD_PROGRAM,
    BOARD_SHELL,
    BOARD_WRITE,
    BROWSER_C,
    ca_part,
    changing,
    check_calls,
    command_of,
    container_ids,
    containers_by_name,
    create_names,
    DASH_C,
    DASHBOARD_COMMAND,
    FIRST_LINE,
    foreign_container,
    FORMATS,
    GATE_C,
    GATEWAY_COMMAND,
    HOME_V,
    Host,
    IMAGES_DIR_NAME,
    LABEL,
    namespaces_stale,
    NETWORK,
    NEW_REVISION,
    output_lines,
    PLACEHOLDER,
    PLATFORM,
    PROXY_C,
    PROXY_TAG,
    PROXY_V,
    RESET_LINE,
    REVISION,
    run_names,
    SECTIONS,
    seed_tokens,
    SETTINGS,
    STACK_CONTAINERS,
    STACK_VOLUMES,
    sub_host,
    TOKEN_ENV_NAMES,
    TOKEN_NAMES,
    TOKENS_V,
)
from journeys_stack_up_support import converged_root, host, shared_stack, stack  # noqa: F401

pytestmark = pytest.mark.linux_only


def check_topology(host: Host) -> None:
    """Shape of the five containers: rules 1 to 3 and decisions 1 to 7, true on first creation and when ready."""
    state = host.state()
    containers = host.containers()
    docker = host.docker_calls()
    assert sorted(containers) == sorted(STACK_CONTAINERS)
    for name, container in containers.items():
        assert container["health"] and container["health"][0] in ("CMD", "CMD-SHELL"), name
        assert all(m["name"] in SECTIONS["volumes"] for m in container["mounts"]), name
        assert container["ports"] == [], name
        assert container["restart"] == "unless-stopped", name
        assert container["labels"] == {"raphael-journeys.made-by": "stack_up"}, name
        allowed = set(SECTIONS[f"env {name}"]) | set(state["images"][container["image_ref"]]["env"])
        assert set(container["env"]) <= allowed, name
    for argv in docker:
        assert not {"-p", "--publish", "-P", "--publish-all", "--expose"} & set(argv)
        if argv[0] in ("run", "create") and "--rm" not in argv:
            assert argv[argv.index("--label") + 1] == LABEL
    assert state["networks"][NETWORK] == {"driver": "bridge", "internal": False}
    assert sorted(state["volumes"]) == sorted(STACK_VOLUMES)

    # Decision 5: images from scripts/journeys/images, or the released owner app image plus the CA layer.
    images_dir = (host.scripts / IMAGES_DIR_NAME).resolve()
    ca = ca_part(state)
    assert containers[PROXY_C]["image_ref"] == f"raphael-journeys-proxy:{PROXY_TAG}"
    assert containers[DASH_C]["image_ref"] == containers[GATE_C]["image_ref"] == f"raphael-journeys-platform:{PLATFORM}"
    app_built = state["images"][containers[APP_C]["image_ref"]]["built"]
    browser_built = state["images"][containers[BROWSER_C]["image_ref"]]["built"]
    assert containers[APP_C]["image_ref"] == f"raphael-journeys-app:{REVISION}-{ca}"
    assert containers[BROWSER_C]["image_ref"] == f"raphael-journeys-browser:{REVISION}-{ca}"
    assert Path(app_built["file"]).resolve() == images_dir / "app-ca.Dockerfile"
    assert app_built["args"] == {"APP_IMAGE_DIGEST": f"raphael-journeys-app-base:{REVISION}"}
    assert Path(browser_built["file"]).resolve() == images_dir / "browser.Dockerfile"

    # Rule 3: the gateway joins the owner app's namespace (by id), not the reverse; the app carries the alias
    # workspace, the proxy's upstream; the gateway has no alias, hostname or port of its own.
    app_id = container_ids(host)[APP_C]
    assert containers[GATE_C]["network"] == f"container:{app_id}"
    assert containers[GATE_C]["aliases"] == []
    assert containers[APP_C]["network"] == NETWORK
    assert containers[APP_C]["aliases"] == ["workspace"]
    assert containers[DASH_C]["aliases"] == ["dashboard"]
    assert sorted(containers[PROXY_C]["aliases"]) == ["synthetic-team.cloudflareaccess.com", "workspace.synthetic.test"]
    for name in (PROXY_C, DASH_C, APP_C, BROWSER_C):
        assert containers[name]["network"] == NETWORK
    assert namespaces_stale(state) == []

    # Rule 2 and rule 1: the commands, as recorded in the create calls.
    assert command_of(docker, GATE_C) == GATEWAY_COMMAND
    assert containers[GATE_C]["cmd"] == GATEWAY_COMMAND
    assert command_of(docker, DASH_C) == DASHBOARD_COMMAND
    assert containers[DASH_C]["cmd"] == DASHBOARD_COMMAND
    assert containers[DASH_C]["health"][0] == "CMD-SHELL"
    assert "127.0.0.1:9119/api/health" in containers[DASH_C]["health"][1]
    assert "127.0.0.1:8642/health" in containers[GATE_C]["health"][1]


def assert_token_check(docker: list[list[str]]) -> None:
    """Rule 4 (c): one labelled, read-only helper that tests each named file with test -s and prints nothing."""
    helpers = check_calls(docker)
    assert len(helpers) == 1
    argv = helpers[0]
    assert argv[argv.index("--label") + 1] == LABEL
    assert argv[argv.index("--network") + 1] == "none"
    assert f"type=volume,source={TOKENS_V},target=/var/lib/raphael-tokens,readonly" in argv
    assert "-i" not in argv
    script = argv[-1]
    assert sorted(re.findall(r'test -s "\$([A-Z0-9_]+)"', script)) == TOKEN_ENV_NAMES
    assert not {"cat", "head", "od", "printf", "echo", "tee", ">", ">>"} & set(script.replace('"', " ").split())
    assert not [word for word in argv if "/var/lib/raphael-tokens/" in word]


def test_1_empty_host_with_token_files_creates_network_volumes_and_five_containers_in_order(host: Host) -> None:
    """Empty host, token volume already holds the files: the order, the shape of each container, rules 1 to 4."""
    seed_tokens(host)
    result = host.converge()
    assert result.returncode == 0, result.stdout + result.stderr
    assert output_lines(result)[-1] == ANSWERS_LINE

    docker = host.docker_calls()
    made = changing(docker)
    assert made[0] == ["network", "create", "--driver", "bridge", NETWORK]
    assert made[1:3] == [["volume", "create", name] for name in (HOME_V, PROXY_V)]
    assert [argv[0] for argv in made] == (
        ["network", "volume", "volume", "run", "cp", "build", "build", "run", "run", "run", "run", "run"]
    )
    assert run_names(docker) == [PROXY_C, DASH_C, APP_C, GATE_C, BROWSER_C]
    assert not [argv for argv in docker if argv[0] in ("create", "start", "restart", "stop")]
    check_topology(host)

    # The one token check comes after the dashboard and before the owner app and the gateway are started.
    assert_token_check(docker)
    helper = check_calls(docker)[0]
    app_call = next(a for a in docker if a[:1] == ["run"] and "--name" in a and APP_C in a)
    gate_call = next(a for a in docker if a[:1] == ["run"] and "--name" in a and GATE_C in a)
    assert docker.index(helper) < docker.index(app_call) < docker.index(gate_call)

    containers = host.containers()
    for name, container in containers.items():
        assert container["running"] and container["status"] == "healthy", name
    assert host.state()["volumes"][TOKENS_V]["files"] == {n: PLACEHOLDER for n in TOKEN_NAMES}
    for name in (DASH_C, GATE_C):
        image_env = host.state()["images"][containers[name]["image_ref"]]["env"]
        assert "HERMES_DASHBOARD" not in {**image_env, **containers[name]["env"]}, name


def test_1b_first_creation_creates_the_owner_app_and_gateway_without_starting_or_waiting_for_them(host: Host) -> None:
    """Empty host, empty token volume: the owner app and the gateway are created only; the fixed line; exit 0."""
    result = host.converge()
    assert result.returncode == 0, result.stdout + result.stderr
    assert output_lines(result)[-1] == FIRST_LINE
    assert ANSWERS_LINE not in output_lines(result)

    docker = host.docker_calls()
    made = changing(docker)
    assert made[1:4] == [["volume", "create", name] for name in STACK_VOLUMES]
    assert [argv[0] for argv in made] == (
        ["network", "volume", "volume", "volume", "run", "cp", "build", "build", "run", "run", "create", "create", "run"]
    )
    assert run_names(docker) == [PROXY_C, DASH_C, BROWSER_C]
    assert create_names(docker) == [APP_C, GATE_C]
    assert not [argv for argv in docker if argv[0] in ("start", "restart", "stop")]
    check_topology(host)
    assert_token_check(docker)

    ids = container_ids(host)
    containers = host.containers()
    for name in (APP_C, GATE_C):
        assert containers[name]["running"] is False and containers[name]["status"] == "created", name
    waited = {argv[5] for argv in docker if argv[0] == "inspect" and argv[4] == FORMATS["state"]}
    assert waited == {ids[PROXY_C], ids[DASH_C], ids[BROWSER_C]}
    for name in (PROXY_C, DASH_C, BROWSER_C):
        assert containers[name]["running"] and containers[name]["status"] == "healthy", name
    assert host.state()["volumes"][TOKENS_V]["files"] == {}


def test_2_reset_after_converge_exits_0_and_all_five_containers_are_running_and_healthy(stack: Host) -> None:
    """The unchanged reset_stack.sh on the converged stack: exit 0, every container healthy, tokens issued again."""
    mark = stack.mark()
    result = stack.run("reset_stack.sh")
    text = result.stdout + result.stderr
    assert result.returncode == 0, text
    assert output_lines(result)[-1] == RESET_LINE

    state = stack.state()
    containers = containers_by_name(state)
    assert sorted(containers) == sorted(STACK_CONTAINERS)
    for name, container in containers.items():
        assert container["running"] and container["status"] == "healthy", name
    assert namespaces_stale(state) == []
    assert all(state["volumes"][TOKENS_V]["files"].get(name) for name in TOKEN_NAMES)

    order = [(argv[0], argv[1]) for argv in stack.docker_calls(mark) if argv[0] in ("stop", "start", "restart")]
    assert order.index(("stop", GATE_C)) < order.index(("restart", APP_C)) < order.index(("start", GATE_C))
    # the platform's first three writes: the board's store, then the Project, then the board metadata
    writes = [argv[argv.index(DASH_C) + 1:] for argv in stack.docker_calls(mark) if argv[0] == "exec" and "--user" in argv]
    assert writes[:3] == [create_board(), create_project(), BOARD_WRITE]


def create_board(board: str = ANCHOR) -> list[str]:
    return ["hermes", "kanban", "boards", "create", board]


def create_project(name: str = "Raphael Workspace", slug: str = ANCHOR, board: str = ANCHOR) -> list[str]:
    return ["hermes", "project", "create", name, "--slug", slug, "--board", board]


def exec_in(stack: Host, *words: str, stdin: str = BOARD_PROGRAM) -> subprocess.CompletedProcess:
    """One call of the stand-in `docker exec` with these words after exec; standard input is the reset's program
    unless stdin says otherwise (the empty string sends none)."""
    return subprocess.run(
        [str(stack.bin / "docker"), "exec", *words],
        env=stack.env(), cwd=stack.root, input=stdin, capture_output=True, text=True, timeout=60,
    )


NEAR_MISSES = {
    "store_in_another_target": [*AS_PLATFORM, GATE_C, *create_board()],
    "store_without_the_user_option": [DASH_C, *create_board()],
    "store_as_another_user": ["--user", "0:0", DASH_C, *create_board()],
    "store_with_an_extra_exec_option": [*AS_PLATFORM, "-i", DASH_C, *create_board()],
    "store_of_another_board": [*AS_PLATFORM, DASH_C, *create_board("other-workspace")],
    "store_of_a_board_that_only_starts_like_the_anchor": [*AS_PLATFORM, DASH_C, *create_board(ANCHOR + "-2")],
    "store_with_its_words_in_another_order": [*AS_PLATFORM, DASH_C, "hermes", "kanban", "create", "boards", ANCHOR],
    "store_with_an_extra_option": [*AS_PLATFORM, DASH_C, *create_board(), "--switch"],
    "store_as_one_joined_word": [*AS_PLATFORM, DASH_C, " ".join(create_board())],
    "create_in_another_target": [*AS_PLATFORM, GATE_C, *create_project()],
    "create_without_the_user_option": [DASH_C, *create_project()],
    "create_as_another_user": ["--user", "0:0", DASH_C, *create_project()],
    "create_with_an_extra_exec_option": [*AS_PLATFORM, "-i", DASH_C, *create_project()],
    "create_with_another_slug": [*AS_PLATFORM, DASH_C, *create_project(slug="other-slug")],
    "create_with_a_slug_that_only_starts_like_the_anchor": [*AS_PLATFORM, DASH_C, *create_project(slug=ANCHOR + "-2")],
    "create_with_another_board": [*AS_PLATFORM, DASH_C, *create_project(board="other-board")],
    "create_with_another_name": [*AS_PLATFORM, DASH_C, *create_project(name="Another Workspace")],
    "create_with_its_options_swapped": [
        *AS_PLATFORM, DASH_C, "hermes", "project", "create", "Raphael Workspace", "--board", ANCHOR, "--slug", ANCHOR,
    ],
    "create_without_the_board_option": [*AS_PLATFORM, DASH_C, *create_project()[:-2]],
    "create_with_an_extra_option": [*AS_PLATFORM, DASH_C, *create_project(), "--force"],
    "another_project_subcommand": [*AS_PLATFORM, DASH_C, "hermes", "project", "delete", "Raphael Workspace",
                                   "--slug", ANCHOR, "--board", ANCHOR],
    "create_words_in_another_place": [
        *AS_PLATFORM, DASH_C, "hermes", "project", "create", "--slug", ANCHOR, "--board", ANCHOR, "Raphael Workspace",
    ],
    "create_as_one_joined_word": [*AS_PLATFORM, DASH_C, " ".join(create_project())],
    "create_inside_a_shell_script": [*AS_PLATFORM, DASH_C, "sh", "-c", " ".join(create_project()), "sh"],
    "board_write_in_another_target": ["-i", *AS_PLATFORM, GATE_C, *BOARD_WRITE],
    "board_write_without_the_input_option": [*AS_PLATFORM, DASH_C, *BOARD_WRITE],
    "board_write_with_its_options_swapped": [*AS_PLATFORM, "-i", DASH_C, *BOARD_WRITE],
    "board_write_as_another_user": ["-i", "--user", "0", DASH_C, *BOARD_WRITE],
    "board_write_with_another_shell": ["-i", *AS_PLATFORM, DASH_C, "bash", "-c", BOARD_SHELL, "sh"],
    "board_write_with_another_shell_option": ["-i", *AS_PLATFORM, DASH_C, "sh", "-s", BOARD_SHELL, "sh"],
    "board_write_with_another_argument_zero": ["-i", *AS_PLATFORM, DASH_C, "sh", "-c", BOARD_SHELL, "bash"],
    "board_write_without_its_argument_zero": ["-i", *AS_PLATFORM, DASH_C, "sh", "-c", BOARD_SHELL],
    "board_write_without_a_script": ["-i", *AS_PLATFORM, DASH_C, "sh", "-c"],
    "board_write_with_an_extra_word": ["-i", *AS_PLATFORM, DASH_C, *BOARD_WRITE, "extra"],
    "board_write_words_in_another_place": ["-i", *AS_PLATFORM, DASH_C, "sh", "sh", "-c", BOARD_SHELL],
}

STILL_REFUSED = {
    "a_terminal_option": ["-t", *AS_PLATFORM, DASH_C, *create_project()],
    "an_environment_option": ["-e", "A_NAME=a-value", *AS_PLATFORM, DASH_C, *create_project()],
    "a_working_folder_option": ["-w", "/tmp", *AS_PLATFORM, DASH_C, *create_project()],
    "a_project_listing": [*AS_PLATFORM, DASH_C, "hermes", "project", "list"],
    "a_token_issue_without_out": [*AS_PLATFORM, DASH_C, "hermes", "kanban-workspace-token", "issue", "--surface", "x"],
    "a_shell_script_without_a_known_purpose": [*AS_PLATFORM, DASH_C, "sh", "-c", "an invented script", "sh"],
    "a_plain_program": [DASH_C, "rm", "-rf", "/tmp/anything"],
}


def altered(text: str, old: str, new: str) -> str:
    """TEXT with its one OLD put as NEW; a text that holds OLD more or less than once is a mistake of the case."""
    assert text.count(old) == 1, old
    return text.replace(old, new)


# The reset's board write with its options, target and words right and its shell text or its input wrong: the pair is
# (shell text, input). The first case is the reproduction of the review.
BOARD_WRITE_REFUSED = {
    "exit_42_as_the_shell_text_and_no_input": ("exit 42", ""),
    "exit_42_as_the_shell_text": ("exit 42", BOARD_PROGRAM),
    "an_empty_shell_text": ("", BOARD_PROGRAM),
    "an_empty_shell_text_and_no_input": ("", ""),
    "a_nested_docker_command_as_the_shell_text": ("docker run --rm alpine true", BOARD_PROGRAM),
    "a_shell_text_with_one_word_altered": (altered(BOARD_SHELL, "exit 127", "exit 0"), BOARD_PROGRAM),
    "a_shell_text_without_its_last_word": (altered(BOARD_SHELL, 'exec "$@" -', 'exec "$@"'), BOARD_PROGRAM),
    "a_shell_text_with_text_added_at_its_end": (BOARD_SHELL + "\nexit 42", BOARD_PROGRAM),
    "a_shell_text_with_a_newline_added_at_its_end": (BOARD_SHELL + "\n", BOARD_PROGRAM),
    "a_shell_text_with_text_added_at_its_start": ("exit 42\n" + BOARD_SHELL, BOARD_PROGRAM),
    "a_shell_text_without_its_first_line": (BOARD_SHELL.partition("\n")[2], BOARD_PROGRAM),
    "a_shell_text_without_a_middle_line": (altered(BOARD_SHELL, "  set +f\n", ""), BOARD_PROGRAM),
    "a_shell_text_without_its_last_line": (BOARD_SHELL.rpartition("\n")[0], BOARD_PROGRAM),
    "a_shell_text_with_one_space_added": (altered(BOARD_SHELL, "  found=1\n", "   found=1\n"), BOARD_PROGRAM),
    "a_shell_text_with_crlf_line_ends": (BOARD_SHELL.replace("\n", "\r\n"), BOARD_PROGRAM),
    "no_input": (BOARD_SHELL, ""),
    "a_newline_as_the_input": (BOARD_SHELL, "\n"),
    "the_program_for_another_board_slug": (BOARD_SHELL, altered(BOARD_PROGRAM, ANCHOR, "other-workspace")),
    "the_program_for_a_slug_that_only_starts_like_the_anchor": (
        BOARD_SHELL, altered(BOARD_PROGRAM, ANCHOR, ANCHOR + "-2"),
    ),
    "an_unrelated_program": (BOARD_SHELL, "import sys\nsys.exit(42)\n"),
    "the_program_with_its_board_write_altered": (
        BOARD_SHELL, altered(BOARD_PROGRAM, "write_board_metadata(slug, project_id=project.id)",
                             "write_board_metadata(slug, project_id=None)"),
    ),
    "the_program_with_its_board_write_made_a_read": (
        BOARD_SHELL, altered(BOARD_PROGRAM, "write_board_metadata(slug, project_id=project.id)",
                             "read_board_metadata(slug)"),
    ),
    "the_program_without_its_board_write": (
        BOARD_SHELL, altered(BOARD_PROGRAM, "kanban_db.write_board_metadata(slug, project_id=project.id)\n", ""),
    ),
    "the_program_with_text_added_after_it": (BOARD_SHELL, BOARD_PROGRAM + "sys.exit(42)\n"),
    "the_program_with_a_newline_added_after_it": (BOARD_SHELL, BOARD_PROGRAM + "\n"),
    "the_program_with_text_added_before_it": (BOARD_SHELL, "sys.exit(42)\n" + BOARD_PROGRAM),
    "the_program_without_its_final_newline": (BOARD_SHELL, BOARD_PROGRAM.removesuffix("\n")),
    "the_program_with_crlf_line_ends": (BOARD_SHELL, BOARD_PROGRAM.replace("\n", "\r\n")),
}


def test_2b_the_stand_in_takes_the_resets_board_store_project_create_and_board_write_calls(shared_stack: Host) -> None:
    stored = exec_in(shared_stack, *AS_PLATFORM, DASH_C, *create_board(), stdin="")
    assert stored.returncode == 0, stored.stderr
    created = exec_in(shared_stack, *AS_PLATFORM, DASH_C, *create_project())
    assert created.returncode == 0, created.stderr
    written = exec_in(shared_stack, "-i", *AS_PLATFORM, DASH_C, *BOARD_WRITE)
    assert written.returncode == 0, written.stderr


@pytest.mark.parametrize("case", sorted(NEAR_MISSES))
def test_2c_the_stand_in_refuses_a_call_that_differs_from_either_one_in_a_word_or_the_place_of_a_word(
    shared_stack: Host, case: str
) -> None:
    refused = exec_in(shared_stack, *NEAR_MISSES[case])
    assert refused.returncode == 98 and "stand-in docker: refused" in refused.stderr, refused.stderr


@pytest.mark.parametrize("case", sorted(STILL_REFUSED))
def test_2d_the_stand_in_still_refuses_what_it_refused_before(shared_stack: Host, case: str) -> None:
    refused = exec_in(shared_stack, *STILL_REFUSED[case])
    assert refused.returncode == 98 and "stand-in docker: refused" in refused.stderr, refused.stderr


@pytest.mark.parametrize("case", sorted(BOARD_WRITE_REFUSED))
def test_2e_the_stand_in_refuses_a_board_write_whose_shell_text_or_input_is_not_exactly_the_resets(
    shared_stack: Host, case: str
) -> None:
    shell, stdin = BOARD_WRITE_REFUSED[case]
    refused = exec_in(shared_stack, "-i", *AS_PLATFORM, DASH_C, "sh", "-c", shell, "sh", stdin=stdin)
    assert refused.returncode == 98 and "stand-in docker: refused" in refused.stderr, refused.stderr


def test_2f_the_cases_of_2b_to_2e_leave_the_host_they_share_as_the_converge_left_it(
    shared_stack: Host, converged_root: Path
) -> None:
    """Whichever of those cases ran before this one, only the calls log of the shared copy grew."""
    for name in ("state.json", "scripted.json", "verified.txt"):
        original, copy = converged_root / name, shared_stack.root / name
        assert copy.exists() == original.exists(), name
        if original.exists():
            expected = original.read_text(encoding="utf-8").replace(str(converged_root), str(shared_stack.root))
            assert copy.read_text(encoding="utf-8") == expected, name


def test_3_isolation_check_names_on_template_passes_on_the_created_stack(stack: Host) -> None:
    """isolation_check.sh server2: the names-on-template check passes against the same state."""
    result = stack.run("isolation_check.sh", "server2")
    assert "PASS names-on-template" in result.stdout.splitlines(), result.stdout + result.stderr


def test_4_second_converge_with_the_same_tags_makes_no_create_remove_volume_or_token_check_call(stack: Host) -> None:
    """A second converge with all five running and the same tags changes nothing and checks nothing."""
    before = {c["name"]: cid for cid, c in stack.state()["containers"].items()}
    mark = stack.mark()
    result = stack.converge()
    assert result.returncode == 0, result.stdout + result.stderr
    calls = stack.docker_calls(mark)
    assert {argv[0] for argv in calls} <= {"info", "inspect", "cp"}, calls
    assert {c["name"]: cid for cid, c in stack.state()["containers"].items()} == before
    assert output_lines(result)[-1] == ANSWERS_LINE


@pytest.mark.parametrize("index", range(7))
def test_4d_a_missing_or_zero_size_token_file_is_a_first_creation(tmp_path: Path, index: int) -> None:
    """Any one of the seven files missing (even index) or of zero size (odd index): nothing starts, same last line."""
    host = Host(tmp_path)
    name = TOKEN_NAMES[index]
    seed_tokens(host, missing=(name,) if index % 2 == 0 else (), empty=() if index % 2 == 0 else (name,))
    files = host.state()["volumes"][TOKENS_V]["files"]
    result = host.converge()
    assert result.returncode == 0, result.stdout + result.stderr
    assert output_lines(result)[-1] == FIRST_LINE
    docker = host.docker_calls()
    assert create_names(docker) == [APP_C, GATE_C]
    assert not [argv for argv in docker if argv[0] in ("start", "restart", "stop")]
    assert not [c for c in host.containers().values() if c["name"] in (APP_C, GATE_C) and c["running"]]
    assert host.state()["volumes"][TOKENS_V]["files"] == files


def test_5_new_owner_app_tag_recreates_the_owner_app_the_gateway_and_the_browser(stack: Host) -> None:
    """A new owner app tag: three removals by id, three creations, no volume call, no proxy volume in a remove."""
    before = container_ids(stack)
    mark = stack.mark()
    result = stack.converge(app=NEW_REVISION)
    assert result.returncode == 0, result.stdout + result.stderr
    docker = stack.docker_calls(mark)

    removed = [argv for argv in docker if argv[0] == "rm"]
    assert sorted(argv[2] for argv in removed) == sorted([before[APP_C], before[GATE_C], before[BROWSER_C]])
    assert all(argv[:2] == ["rm", "--force"] and len(argv) == 3 for argv in removed)
    assert run_names(docker) == [APP_C, GATE_C, BROWSER_C]
    assert len(check_calls(docker)) == 1
    assert not [argv for argv in docker if argv[0] in ("volume", "network")]

    after = container_ids(stack)
    for name in (PROXY_C, DASH_C):
        assert after[name] == before[name], name
    for name in (APP_C, GATE_C, BROWSER_C):
        assert after[name] != before[name], name
    state = stack.state()
    ca = ca_part(state)
    assert containers_by_name(state)[APP_C]["image_ref"] == f"raphael-journeys-app:{NEW_REVISION}-{ca}"
    assert containers_by_name(state)[GATE_C]["network"] == f"container:{after[APP_C]}"
    assert command_of(docker, GATE_C) == GATEWAY_COMMAND
    assert namespaces_stale(state) == []
    assert sorted(state["volumes"]) == sorted(STACK_VOLUMES)

    for argv in stack.converged_docker + stack.docker_calls():
        assert argv[0] not in ("system", "image", "rmi", "prune")
        if argv[0] == "volume":
            assert argv[1] == "create"
        if argv[0] == "rm":
            assert not {"-v", "--volumes", PROXY_V, PROXY_C} & set(argv)
            assert re.fullmatch(r"[0-9a-f]{64}", argv[-1])


PLATFORM_REF = f"raphael-journeys-platform:{PLATFORM}"


def test_5a_same_image_name_with_another_image_id_replaces_the_container(stack: Host) -> None:
    """The dashboard keeps its image name, but its image id is no longer the reference's: it is made again."""
    state = stack.state()
    by_name = containers_by_name(state)
    by_name[DASH_C]["image_id"] = state["images"][f"raphael-journeys-proxy:{PROXY_TAG}"]["id"]
    stack.save(state)
    before = container_ids(stack)
    mark = stack.mark()
    result = stack.converge()
    assert result.returncode == 0, result.stdout + result.stderr
    assert run_names(stack.docker_calls(mark)) == [DASH_C]
    after = container_ids(stack)
    assert after[DASH_C] != before[DASH_C]
    assert {n: i for n, i in after.items() if n != DASH_C} == {n: i for n, i in before.items() if n != DASH_C}


def test_5a2_same_image_name_whose_reference_image_is_missing_is_refused(stack: Host) -> None:
    """The reference image has no record any more: the existing missing-image refusal, nothing changed."""
    state = stack.state()
    state["images"]["old"] = state["images"].pop(PLATFORM_REF)
    stack.save(state)
    before = stack.state()
    mark = stack.mark()
    result = stack.converge()
    assert result.returncode != 0
    assert f"the image {PLATFORM_REF} is missing; build it first; nothing was changed" in result.stderr
    assert changing(stack.docker_calls(mark)) == []
    assert stack.state() == before


def test_5b_gateway_is_recreated_when_its_network_mode_does_not_name_the_current_owner_app(stack: Host) -> None:
    """Recovery after a failed converge: a gateway that joined an owner app that is gone is made again."""
    state = stack.state()
    containers_by_name(state)[GATE_C]["network"] = "container:" + "0" * 64
    stack.save(state)
    before = container_ids(stack)
    mark = stack.mark()
    result = stack.converge()
    assert result.returncode == 0, result.stdout + result.stderr
    docker = stack.docker_calls(mark)
    assert [argv[2] for argv in docker if argv[0] == "rm"] == [before[GATE_C]]
    assert run_names(docker) == [GATE_C]
    after = container_ids(stack)
    assert after[APP_C] == before[APP_C] and after[GATE_C] != before[GATE_C]
    assert containers_by_name(stack.state())[GATE_C]["network"] == f"container:{after[APP_C]}"
    assert namespaces_stale(stack.state()) == []


def test_5c_a_running_gateway_beside_a_stopped_owner_app_is_made_again_after_the_app_starts(stack: Host) -> None:
    """The namespace the gateway joined is gone once the app stopped: start the app, then the gateway, whose record is
    older than the app's new start, is restarted (round 3: the namespace rule; the round 2 list was rm, start, run)."""
    state = stack.state()
    containers_by_name(state)[APP_C]["running"] = False
    stack.save(state)
    before = container_ids(stack)
    mark = stack.mark()
    result = stack.converge()
    assert result.returncode == 0, result.stdout + result.stderr
    assert output_lines(result)[-1] == ANSWERS_LINE
    docker = stack.docker_calls(mark)
    assert [argv for argv in docker if argv[0] in ("rm", "start", "restart", "stop")] == [
        ["start", before[APP_C]],
        ["restart", before[GATE_C]],
    ]
    assert run_names(docker) == []
    after = container_ids(stack)
    assert after == before
    state = stack.state()
    assert namespaces_stale(state) == []
    for name, container in containers_by_name(state).items():
        assert container["running"] and container["status"] == "healthy", name
    assert containers_by_name(state)[GATE_C]["network"] == f"container:{after[APP_C]}"


@pytest.mark.parametrize("case", ["wrong_host", "tag_not_plain", "unlabelled_container", "less_than_20_gb_free"])
def test_6_refusals_change_nothing(tmp_path: Path, case: str) -> None:
    """Each refusal exits non-zero before any call that changes anything, and the fake host is unchanged."""
    overrides: dict = {}
    if case == "wrong_host":
        overrides["host"] = "some-other-host.invalid"
    if case == "less_than_20_gb_free":
        overrides["free_kb"] = 19 * 1024 * 1024
    host = Host(tmp_path, **overrides)
    if case == "unlabelled_container":
        state = host.state()
        foreign_container(state, PROXY_C)
        host.save(state)
    before = host.state()
    if case == "tag_not_plain":
        result = host.run("stack_up.sh", "converge", PROXY_TAG, "../../etc/passwd", PLATFORM, REVISION)
    else:
        result = host.converge()
    assert result.returncode != 0
    assert "nothing was changed" in result.stderr
    assert changing(host.docker_calls()) == []
    assert host.state() == before


def test_7_ca_leaves_the_proxy_only_as_ca_pem_and_only_the_proxy_mounts_the_proxy_volume(stack: Host) -> None:
    """One docker cp of ca.pem; the build folders hold ca.pem only; no other container mounts the proxy volume."""
    state = stack.state()
    proxy_id = container_ids(stack)[PROXY_C]
    docker = stack.converged_docker + stack.docker_calls()
    copies = [argv for argv in docker if argv[0] == "cp"]
    assert len(copies) == 1
    assert copies[0][1] == f"{proxy_id}:/state/ca.pem" and Path(copies[0][2]).name == "ca.pem"
    layers = [img for ref, img in state["images"].items() if ref.startswith(("raphael-journeys-app:", "raphael-journeys-browser:"))]
    assert len(layers) == 2
    assert all(img["built"]["context_files"] == ["ca.pem"] for img in layers)
    mounting = [c["name"] for c in state["containers"].values() if any(m["name"] == PROXY_V for m in c["mounts"])]
    assert mounting == [PROXY_C]
    for argv in docker:
        if argv[0] in ("run", "create"):
            mounts = [argv[i + 1] for i, word in enumerate(argv) if word == "--mount"]
            names_proxy_volume = any(f"source={PROXY_V}," in m for m in mounts)
            named = argv[argv.index("--name") + 1] if "--name" in argv else None
            assert names_proxy_volume == (named == PROXY_C)


def test_8_no_output_line_or_command_line_carries_an_environment_value(host: Host, tmp_path: Path) -> None:
    """Values reach docker only through --env-file; neither the output nor any command line shows one."""
    first = sub_host(tmp_path, "first")
    seed_tokens(host)
    results = [first.converge(), host.converge(), host.converge(app=NEW_REVISION)]
    results.append(host.run("stack_up.sh", "converge", PROXY_TAG, "bad tag", PLATFORM, REVISION))
    results.append(host.run("stack_up.sh", "versions"))
    state = host.state()
    app_env = containers_by_name(state)[APP_C]["env"]
    assert app_env == SETTINGS
    values = {value for value in SETTINGS.values() if value} | {"/state", "/opt/data"}
    for result in results:
        for line in output_lines(result):
            assert not [v for v in values if v in line], line
            assert PLACEHOLDER.strip() not in line
    for each in (first, host):
        for argv in each.docker_calls():
            for word in argv:
                if word == "--mount" or word.startswith("type=volume,"):
                    continue
                assert not [v for v in values - {"/state", "/opt/data"} if v in word and len(v) > 4], word
                assert PLACEHOLDER.strip() not in word
            assert "-e" not in argv and "--env" not in argv
            if argv[0] in ("run", "create") and "--name" in argv and argv[argv.index("--name") + 1] != BROWSER_C:
                assert "--env-file" in argv
            if argv[0] == "run" and "--rm" in argv:
                assert "--env-file" in argv and "--name" not in argv


# --- round 2: the gateway container runs one gateway (rule 2, finding 4)


@pytest.mark.parametrize("created", ["started", "created_only"])
def test_11_the_recorded_create_call_pins_the_plain_gateway_run_and_no_dashboard_variable(tmp_path: Path, created: str) -> None:
    """Rule 2: the command after the image is exactly gateway run, without --no-supervise, in both creation paths;
    HERMES_DASHBOARD is in no env name or value, and the dashboard container keeps its arguments and order."""
    host = Host(tmp_path)
    if created == "started":
        seed_tokens(host)
    result = host.converge()
    assert result.returncode == 0, result.stdout + result.stderr
    docker = host.docker_calls()
    gateway_call = next(a for a in docker if a[0] in ("run", "create") and "--name" in a and GATE_C in a)
    assert gateway_call[0] == ("run" if created == "started" else "create")
    assert command_of(docker, GATE_C) == ["gateway", "run"]
    assert "--no-supervise" not in gateway_call
    assert not [word for word in gateway_call if "HERMES_DASHBOARD" in word]
    assert command_of(docker, DASH_C) == ["dashboard", "--host", "0.0.0.0", "--port", "9119", "--no-open"]
    for argv in docker:
        assert not [word for word in argv if "HERMES_DASHBOARD" in word]
    state = host.state()
    for container in state["containers"].values():
        image_env = state["images"][container["image_ref"]]["env"]
        merged = {**image_env, **container["env"]}
        assert not [k for k in merged if "HERMES_DASHBOARD" in k]
        assert not [v for v in merged.values() if "HERMES_DASHBOARD" in v]
        gateways = [c for c in state["containers"].values() if c["cmd"][:2] == ["gateway", "run"]]
        assert [c["name"] for c in gateways] == [GATE_C]


def test_9b_converge_does_not_wait_for_an_owner_app_that_exits_at_once(tmp_path: Path) -> None:
    """With the files gone after a first creation the stand-in app would exit; converge never starts it."""
    host = Host(tmp_path)
    seed_tokens(host, empty=(TOKEN_NAMES[0],))
    result = host.converge()
    assert result.returncode == 0 and output_lines(result)[-1] == FIRST_LINE
    assert [c["name"] for c in host.containers().values() if c["running"]] == [PROXY_C, DASH_C, BROWSER_C]


def test_extra_container_that_never_answers_fails_in_one_budget_and_converge_resumes(tmp_path: Path) -> None:
    """No answer: exit 1 after at most 90 polls of 2 seconds; fixing the cause and converging again makes no change."""
    host = Host(tmp_path, never_healthy=[BROWSER_C])
    seed_tokens(host)
    result = host.converge()
    assert result.returncode == 1 and "run converge again" in result.stderr
    sleeps = [c["argv"] for c in host.calls() if c["tool"] == "sleep"]
    assert 0 < len(sleeps) <= 90 and all(argv == ["2"] for argv in sleeps)
    assert sorted(host.containers()) == sorted(STACK_CONTAINERS)
    state = host.state()
    state["never_healthy"] = []
    host.save(state)
    mark = host.mark()
    result = host.converge()
    assert result.returncode == 0, result.stdout + result.stderr
    assert {argv[0] for argv in host.docker_calls(mark)} <= {"info", "inspect", "cp"}
