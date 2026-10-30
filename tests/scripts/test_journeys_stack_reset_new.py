"""Tests for reset_stack.sh on a stack that was just created, and the identity probe (card P-2).

The stand-ins and helpers are in journeys_stack_harness.
"""

from __future__ import annotations

import stat
import subprocess

import pytest

from journeys_stack_harness import (
    ANCHOR_WRITES,
    APP_C,
    BROWSER_C,
    DASH_C,
    F_APP_SETTINGS,
    GATE_C,
    HOME_V,
    Host,
    ID_SCRIPT,
    KEY_ENV,
    MODEL_TOKEN,
    PROXY_C,
    STACK_CONTAINERS,
    SURFACE_ENV,
    TOKENS_V,
    TOKEN_VARS,
    all_output,
    argv_text,
    assert_nothing_changed,
    assert_secret_free,
    docker_by_hand,
    docker_target,
    generated_secrets,
    home_files,
    host,
    is_id_probe,
    mount_source,
    own_volume,
    parse_env_file,
    reset_ok,
    run_reset,
    set_app_env,
    snapshot,
    writer_calls,
)

pytestmark = pytest.mark.linux_only


# ------------------------------------------------------------------ reset_stack.sh: a stack that was just created

APP_USER = "svc-sentinel"  # the owner app is created with a user name; only its image says which numbers that is
APP_IDS = (4321, 4322)
SETTING_SENTINELS = {
    "RAPHAEL_ACCESS_ISSUER": "https://setting-sentinel-issuer.example.invalid/issuer",
    "RAPHAEL_OWNER_MEMBERSHIP_JSON": '{"owner": "setting-sentinel-membership"}',
    "NODE_EXTRA_CA_CERTS": "/etc/setting-sentinel/ca.pem",
}
NAME_MARK = "odd-name-marker"
CUSTOM_NAMES = dict(zip(TOKEN_VARS, ["WS_1.tok", "rec-2", "Conn.3", "auto_4", "models-5.tok", "exec_6", "plan.7-x"]))


def sentinel_settings(state: dict) -> None:
    for name, value in SETTING_SENTINELS.items():
        set_app_env(name, value)(state)


def fresh_stack(host: Host) -> None:
    """The stack as its creation leaves it: owner app and gateway never ran, the token folder is empty and root's."""
    for path in host.vol(TOKENS_V).iterdir():
        path.unlink()
    host.set_identities(APP_IDS, (10000, 10000))
    own_volume(host, TOKENS_V, (0, 0))

    def mutate(state: dict) -> None:
        containers = state["containers"]
        state["images"][containers[APP_C]["image"]]["users"] = {APP_USER: list(APP_IDS)}
        containers[APP_C].update(config_user=APP_USER, exits_without_token_files=True)
        for who in (APP_C, GATE_C):
            containers[who].pop("health")  # a container that never ran has no health state
            containers[who].update(running=False, health_on_start="healthy", health_delay=2)
        sentinel_settings(state)
    host.update(mutate)


def token_setting(var: str, *values: str):
    """The owner app holds the setting once per value (no value leaves it unset)."""
    def mutate(state: dict) -> None:
        box = state["containers"][APP_C]
        box["env"] = [item for item in box["env"] if not item.startswith(var + "=")] + [f"{var}={v}" for v in values]
    return mutate


def settings_to_hide(host: Host) -> list[str]:
    """The values of the owner app's token file settings and sentinel settings, as the container holds them now."""
    pairs = [item.partition("=") for item in host.state()["containers"][APP_C]["env"]]
    return [value for name, _, value in pairs if name in (*TOKEN_VARS, *SETTING_SENTINELS) and value]


def assert_no_setting_values(host: Host, result: subprocess.CompletedProcess, values: list[str]) -> None:
    """No output line and no recorded command line carries a setting value; only the inspect itself returns them."""
    assert values and all(len(value) >= 8 for value in values)
    answers = "\n".join(c.get("stdout", "") + c.get("stderr", "") for c in host.calls("docker")
                        if F_APP_SETTINGS not in c["argv"])
    for value in values:
        assert value not in all_output(result), "a setting value was printed"
        assert value not in argv_text(host), "a setting value reached a command line"
        assert value not in answers, "a container command printed a setting value"


def assert_user_only_on_the_probe(host: Host, result: subprocess.CompletedProcess) -> None:
    """The configured user is the one setting value that must be on a command line: --user of the id run."""
    carrying = [c["argv"] for c in host.calls() if APP_USER in "\x00".join(c["argv"])]
    probes = [c["argv"] for c in host.calls("docker") if is_id_probe(c["argv"])]
    assert carrying == probes and len(probes) <= 1
    assert all(argv[argv.index("--user") + 1] == APP_USER for argv in probes)
    assert APP_USER not in all_output(result)
    assert APP_USER not in "\n".join(c.get("stdout", "") + c.get("stderr", "") for c in host.calls("docker")
                                     if F_APP_SETTINGS not in c["argv"])


def change_steps(host: Host) -> list[tuple[str, str | None]]:
    """The changing docker calls of a reset in order: stops, starts, restarts, helper runs by volume, writes."""
    steps = []
    for call in host.calls("docker"):
        sub, target = docker_target(call)
        if sub in ("stop", "start", "restart"):
            steps.append((sub, target))
        elif sub == "run":
            steps.append(("helper", mount_source(call)))
        elif sub == "exec" and "--user" in call["argv"]:
            steps.append(("write", target))
    return steps


def test_reset_starts_a_stopped_owner_app_on_a_stack_that_was_just_created(host: Host):
    fresh_stack(host)
    image = host.state()["containers"][APP_C]["image"]
    result = reset_ok(host)

    containers = host.state()["containers"]
    for who in STACK_CONTAINERS:
        assert (containers[who]["running"], containers[who].get("health")) == (True, "healthy"), who
    assert "the stack is reset and every container answers" in result.stdout
    assert len(host.calls("sleep")) >= 2, "the two containers that were down report starting before they answer"

    tokens = host.vol(TOKENS_V)
    names = set(SURFACE_ENV) | {"executor", "planner"}
    assert {p.name for p in tokens.iterdir()} == names
    assert host.owner_of(tokens) == APP_IDS and stat.S_IMODE(tokens.stat().st_mode) == 0o700
    for name in names:
        info = (tokens / name).lstat()
        assert stat.S_ISREG(info.st_mode) and stat.S_IMODE(info.st_mode) == 0o600 and 0 < info.st_size <= 16 * 1024
        assert host.owner_of(tokens / name) == APP_IDS
    for surface in SURFACE_ENV:
        assert (tokens / surface).read_text().startswith(f"tok-{surface}-")
    for name, path in zip(("executor", "planner"), home_files(host).values()):
        assert (tokens / name).read_text().strip() == parse_env_file(path)[1]["API_SERVER_KEY"]

    steps = change_steps(host)  # the order of a reset of a running stack
    assert set(steps[:2]) == {("stop", GATE_C), ("stop", DASH_C)}
    assert set(steps[2:4]) == {("helper", HOME_V), ("helper", TOKENS_V)}
    assert steps[4] == ("start", DASH_C)
    assert steps[5:-6] == [("write", DASH_C)] * (ANCHOR_WRITES + 9)
    assert steps[-6] == ("helper", TOKENS_V)
    assert steps[-5:-2] == [("restart", APP_C), ("restart", DASH_C), ("start", GATE_C)]
    assert set(steps[-2:]) == {("start", PROXY_C), ("start", BROWSER_C)}

    calls = host.calls("docker")  # the owner app is read without running it
    assert [c["argv"] for c in calls if c["argv"][0] == "inspect" and F_APP_SETTINGS in c["argv"]] == [
        ["inspect", "--type", "container", "--format", F_APP_SETTINGS, APP_C]]
    assert [c["argv"] for c in calls if is_id_probe(c["argv"])] == [
        ["run", "--rm", "--network", "none", "--user", APP_USER, "--entrypoint", "sh", image, "-c", ID_SCRIPT]]
    assert not [c for c in calls if docker_target(c) == ("exec", APP_C)]
    kinds = [docker_target(c)[0] for c in calls]
    assert kinds.index("id-run") < kinds.index("stop")

    assert_secret_free(host, result, generated_secrets(host))
    assert_no_setting_values(host, result, settings_to_hide(host))
    assert_user_only_on_the_probe(host, result)


@pytest.mark.parametrize("app", ["running", "stopped"])
def test_reset_still_refuses_a_stopped_dashboard_and_changes_nothing(host: Host, app: str):
    if app == "stopped":
        fresh_stack(host)
    else:
        host.update(sentinel_settings)
    host.update(lambda state: state["containers"][DASH_C].update(running=False))
    before = snapshot(host)
    result = run_reset(host)
    assert result.returncode != 0
    assert DASH_C in result.stderr and "nothing was changed" in result.stderr
    assert_nothing_changed(host, before)
    assert host.calls("sleep") == []
    assert_secret_free(host, result, [MODEL_TOKEN])
    assert_no_setting_values(host, result, settings_to_hide(host))
    assert_user_only_on_the_probe(host, result)


def test_reset_uses_the_token_file_names_that_the_stopped_owner_app_settings_give(host: Host):
    fresh_stack(host)

    def custom_names(state: dict) -> None:
        for var, name in CUSTOM_NAMES.items():
            token_setting(var, f"/run/tokens/{name}")(state)
    host.update(custom_names)
    result = reset_ok(host)

    tokens = host.vol(TOKENS_V)
    assert {p.name for p in tokens.iterdir()} == set(CUSTOM_NAMES.values())
    for surface, var in SURFACE_ENV.items():
        assert (tokens / CUSTOM_NAMES[var]).read_text().startswith(f"tok-{surface}-")
    for var, path in zip(KEY_ENV.values(), home_files(host).values()):
        assert (tokens / CUSTOM_NAMES[var]).read_text().strip() == parse_env_file(path)[1]["API_SERVER_KEY"]
    issued = [c["argv"][c["argv"].index("--out") + 1] for c in writer_calls(host) if "--out" in c["argv"]]
    assert sorted(issued) == sorted(f"/tokens/{CUSTOM_NAMES[var]}" for var in SURFACE_ENV.values())
    containers = host.state()["containers"]
    assert (containers[APP_C]["running"], containers[APP_C]["health"]) == (True, "healthy")

    assert_secret_free(host, result, generated_secrets(host))
    assert_no_setting_values(host, result, settings_to_hide(host))
    assert_user_only_on_the_probe(host, result)


BAD_TOKEN_NAMES = {
    "shared-with-a-surface": (KEY_ENV["default"], ["/run/tokens/workspace"]),
    "shared-by-two-surfaces": (SURFACE_ENV["models"], ["/run/tokens/recommendations"]),
    "shared-by-the-two-keys": (KEY_ENV["planner"], ["/run/tokens/executor"]),
    "dot-dot": (SURFACE_ENV["workspace"], ["/run/tokens/.."]),
    "parent-folder": (SURFACE_ENV["workspace"], [f"/run/tokens/../{NAME_MARK}"]),
    "nested": (SURFACE_ENV["recommendations"], [f"/run/tokens/{NAME_MARK}/file"]),
    "empty": (SURFACE_ENV["connections"], ["/run/tokens/"]),
    "hidden": (SURFACE_ENV["automations"], [f"/run/tokens/.{NAME_MARK}"]),
    "leading-dash": (SURFACE_ENV["models"], [f"/run/tokens/-{NAME_MARK}"]),
    "space": (KEY_ENV["default"], [f"/run/tokens/{NAME_MARK} name"]),
    "command-substitution": (KEY_ENV["planner"], [f"/run/tokens/$({NAME_MARK})"]),
    "double-quote": (SURFACE_ENV["workspace"], [f'/run/tokens/{NAME_MARK}"quote']),
    "backslash": (SURFACE_ENV["workspace"], [f"/run/tokens/{NAME_MARK}\\back"]),
    "tab": (SURFACE_ENV["workspace"], [f"/run/tokens/{NAME_MARK}\ttab"]),
    "newline": (SURFACE_ENV["workspace"], [f"/run/tokens/{NAME_MARK}\nHERMES_API_TOKEN_FILE=/run/tokens/forged"]),
    "unset": (SURFACE_ENV["models"], []),
    "set-twice": (SURFACE_ENV["models"], [f"/run/tokens/{NAME_MARK}-a", f"/run/tokens/{NAME_MARK}-b"]),
}


@pytest.mark.parametrize("case", sorted(BAD_TOKEN_NAMES))
def test_reset_refuses_a_shared_or_not_plain_token_file_name_of_the_stopped_owner_app(host: Host, case: str):
    var, values = BAD_TOKEN_NAMES[case]
    fresh_stack(host)
    host.update(token_setting(var, *values))
    before = snapshot(host)
    result = run_reset(host)
    assert result.returncode != 0
    assert var in result.stderr and APP_C in result.stderr and "nothing was changed" in result.stderr
    assert_nothing_changed(host, before)
    assert host.calls("sleep") == []
    assert_secret_free(host, result, [MODEL_TOKEN])
    assert_no_setting_values(host, result, settings_to_hide(host) + [NAME_MARK])
    assert_user_only_on_the_probe(host, result)


# ------------------------------------------------------------------ reset_stack.sh: the identity probe fails closed

EXTRA_ID = "4999"
UID_TEXT, GID_TEXT = (str(number) for number in APP_IDS)
# (printed text, exit status) of `id -u`, then of `id -g`; None prints nothing.
BAD_PROBES = {
    "first-read-fails": ((UID_TEXT, 23), (GID_TEXT, 0)),
    "second-read-fails": ((UID_TEXT, 0), (GID_TEXT, 23)),
    "extra-line-from-id-u": ((f"{UID_TEXT}\n{EXTRA_ID}", 0), (GID_TEXT, 0)),
    "extra-line-from-id-g": ((UID_TEXT, 0), (f"{GID_TEXT}\n{EXTRA_ID}", 0)),
    "blank-line-between": ((UID_TEXT + "\n", 0), (GID_TEXT, 0)),
    "trailing-blank-line": ((UID_TEXT, 0), (GID_TEXT + "\n", 0)),
    "one-line": ((UID_TEXT, 0), (None, 0)),
    "no-output": ((None, 0), (None, 0)),
}


def probe_says(reads):
    def mutate(state: dict) -> None:
        box = state["containers"][APP_C]
        for key, (text, status) in zip(("id -u", "id -g"), reads):
            box.setdefault("answers", {})[key] = text
            box.setdefault("exit_codes", {})[key] = status
    return mutate


@pytest.mark.parametrize("case", list(BAD_PROBES))
def test_reset_refuses_an_identity_probe_unless_both_reads_succeed_and_print_exactly_two_numbers(
        host: Host, case: str):
    fresh_stack(host)
    host.update(probe_says(BAD_PROBES[case]))
    before = snapshot(host)
    result = run_reset(host)
    assert result.returncode != 0, "the reset went on after an identity probe that has to be refused"
    runs = [c["argv"] for c in host.calls("docker") if c["argv"][0] == "run"]
    assert [argv[-2:] for argv in runs] == [["-c", "id -u && id -g"]], "the identity probe is not `id -u && id -g`"
    assert APP_C in result.stderr and "nothing was changed" in result.stderr
    assert_nothing_changed(host, before)
    assert host.calls("sleep") == []
    assert_secret_free(host, result, [MODEL_TOKEN])
    assert_no_setting_values(host, result, settings_to_hide(host))
    assert_user_only_on_the_probe(host, result)
    for number in (UID_TEXT, GID_TEXT, EXTRA_ID):  # not even a helper number is printed or put on a command line
        assert number not in all_output(result), "a helper number was printed"
        assert number not in argv_text(host), "a helper number reached a command line"


def test_the_fake_runs_the_probe_text_in_a_real_shell_with_a_stand_in_id(host: Host):
    image = host.state()["containers"][APP_C]["image"]

    def probe(text: str) -> tuple[int, str]:
        done = docker_by_hand(host, "run", "--rm", "--network", "none", "--user", "1000:1000", "--entrypoint", "sh",
                              image, "-c", text)
        return done.returncode, done.stdout
    assert probe("id -u && id -g") == (0, "1000\n1000\n")
    host.update(probe_says(((UID_TEXT, 23), (GID_TEXT, 0))))  # `&&` stops after a failed first read, `;` goes on
    assert probe("id -u && id -g") == (23, f"{UID_TEXT}\n")
    assert probe("id -u; id -g") == (0, f"{UID_TEXT}\n{GID_TEXT}\n")
    assert probe("id -u; touch somewhere")[0] == 97
    host.update(probe_says(((f"{UID_TEXT}\n{EXTRA_ID}", 0), (GID_TEXT + "\n", 0))))
    assert probe("id -u && id -g") == (0, f"{UID_TEXT}\n{EXTRA_ID}\n{GID_TEXT}\n\n")
    host.update(probe_says(((None, 0), (None, 0))))
    assert probe("id -u && id -g") == (0, "")
