"""Tests for the journeys start script and the one-run script (card P-3).

start.sh and run_once.sh run for real, in a temporary copy of the journeys folder, against
stand-ins that sit alone on PATH: systemd-run, systemctl, docker, ssh, curl, hermes and sleep. The
reset and the isolation check are stand-ins in that copy too, so no test reaches a real host,
service, container or remote machine. Every stand-in logs its call to one file, in order, and
answers from answers.json; a fake docker records any call and fails it, because neither script may
call docker itself.
"""

from __future__ import annotations

import fcntl
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.linux_only

REPO = Path(__file__).resolve().parents[2]
JOURNEYS = REPO / "scripts" / "journeys"
START = JOURNEYS / "start.sh"
RUN_ONCE = JOURNEYS / "run_once.sh"
PROXY_SOURCE = JOURNEYS / "synthetic_proxy.py"

PAGE_TEXT = "Owner workspace"
GOOD_PAGE = "<html>Owner workspace PAGE-BODY-SENTINEL</html>"
BANNER_PAGE = "<html>Something went wrong PAGE-BANNER-SENTINEL</html>"
SECRET = "sk-ant-oat01-SENTINEL-secret-value-0123456789"
BASE_URL = "https://workspace.synthetic.test"
RUNNER = "runner@server2.invalid"
SENTINELS = [SECRET, "PAGE-BODY-SENTINEL", "PAGE-BANNER-SENTINEL", PAGE_TEXT]

LOCK_REASON = "Refused: a journey run is still at work (the run lock is held)."
UNIT_REASON = "Refused: the unit hermes-journeys is still loaded."
REASONS = {
    "config": "a required setting is missing or is not plain",
    "isolation": "the isolation check failed",
    "reset": "the stack reset failed",
    "proxy": "the sign-in proxy did not answer",
    "owner-app": "the owner app did not show its page text in time",
    "journeys": "the journey runner failed",
}
ISOLATION = ["isolation_check.sh server2", "isolation_check.sh server1"]
RESET = "reset_stack.sh"

STAND_IN = r'''
import json, sys
from pathlib import Path

top = Path(__file__).parent
name = Path(sys.argv[1]).name
args = sys.argv[2:]
answers = json.loads((top / "answers.json").read_text(encoding="utf-8"))
stdin = "" if sys.stdin.isatty() else sys.stdin.read()
log = top / "calls.jsonl"


def record(label):
    with open(log, "a", encoding="utf-8") as handle:
        handle.write(json.dumps({"call": label, "argv": [name, *args], "stdin": stdin}) + "\n")


def calls_so_far(label):
    if not log.exists():
        return 0
    return sum(json.loads(line)["call"] == label for line in log.read_text().splitlines())


def leak(rc):
    # every stand-in is noisy with values that must never come out of the scripts
    sys.stdout.write("leak: %s %s\n" % (answers["secret"], answers["page"]))
    sys.stderr.write("leak: %s %s\n" % (answers["secret"], answers["page"]))
    sys.exit(rc)


if name in ("isolation_check.sh", "reset_stack.sh"):
    label = " ".join([name, *args])
    record(label)
    leak(answers["rc"].get(label, 0))
if name == "docker":
    record("docker")
    sys.exit(98)
if name == "sleep":
    record("sleep")
    sys.exit(0)
if name == "systemctl":
    record("systemctl")
    if "list-units" in args:
        for line in answers["units"]:
            print(line)
    sys.exit(answers["systemctl_rc"])
if name == "systemd-run":
    record("systemd-run")
    code, error = answers["launch"]
    sys.stderr.write(error)
    sys.exit(code)
if name == "ssh":
    record("ssh")
    leak(answers["ssh_rc"])
if name == "hermes":
    record("hermes")
    sys.exit(answers["hermes_rc"])
if name == "curl":
    url = [arg for arg in args if arg.startswith("https://")][-1]
    if url == answers["proxy_url"]:
        record("curl proxy")
        spec = answers["proxy"]
    else:
        record("curl app")
        asked = calls_so_far("curl app") - 1
        spec = answers["app"][min(asked, len(answers["app"]) - 1)]
    sys.stdout.write(spec["body"])
    sys.exit(spec["rc"])
sys.exit("stand-in: unknown name " + name)
'''

TOOLS = ("bash", "flock", "mv", "rm", "mkdir", "dirname")


def _issuer_and_kid() -> tuple[str, str]:
    source = PROXY_SOURCE.read_text(encoding="utf-8")
    issuer = re.search(r'^ISSUER = "([^"]+)"$', source, re.M).group(1)
    kid = re.search(r'^KID = "([^"]+)"$', source, re.M).group(1)
    return issuer, kid


ISSUER, KID = _issuer_and_kid()
PROXY_URL = f"{ISSUER}/cdn-cgi/access/certs"
HEALTHY_PROXY = {"rc": 0, "body": json.dumps({"keys": [{"kty": "RSA", "kid": KID}]}, separators=(",", ":"))}


class Sandbox:
    """A temporary copy of the journeys folder, the stand-ins and the answers they give."""

    def __init__(self, top: Path) -> None:
        self.top = top
        self.scripts = top / "journeys"
        self.bin = top / "bin"
        self.home = top / "hermes-home"
        self.run_dir = self.home / "journeys"
        self.home.mkdir()
        self.bin.mkdir()
        self.scripts.mkdir()
        (top / "stand_in.py").write_text(STAND_IN, encoding="utf-8")
        for name in ("systemd-run", "systemctl", "docker", "ssh", "curl", "hermes", "sleep"):
            self._wrapper(self.bin / name)
        for name in TOOLS:
            (self.bin / name).symlink_to(shutil.which(name))
        for name in ("isolation_check.sh", RESET):
            self._wrapper(self.scripts / name)
        shutil.copy2(PROXY_SOURCE, self.scripts / "synthetic_proxy.py")
        self.answer()

    def _wrapper(self, path: Path) -> None:
        path.write_text(f'#!/bin/sh\nexec "{sys.executable}" -I -S "{self.top / "stand_in.py"}" "$0" "$@"\n')
        path.chmod(0o755)

    def answer(self, **changes) -> None:
        answers = {
            "rc": {}, "units": [], "systemctl_rc": 0, "launch": [0, ""], "ssh_rc": 0, "hermes_rc": 0,
            "proxy": HEALTHY_PROXY, "proxy_url": PROXY_URL,
            "app": [{"rc": 0, "body": GOOD_PAGE}], "secret": SECRET, "page": GOOD_PAGE,
        }
        answers.update(changes)
        (self.top / "answers.json").write_text(json.dumps(answers), encoding="utf-8")

    def env(self, **changes: str | None) -> dict[str, str]:
        env = {
            "PATH": str(self.bin), "HOME": str(self.top), "HERMES_HOME": str(self.home),
            "JOURNEY_STACK_HOST": "stack-host.invalid", "JOURNEY_PRODUCTION_ORIGIN": "https://prod.invalid",
            "JOURNEY_BASE_URL": BASE_URL, "JOURNEY_PAGE_TEXT": PAGE_TEXT, "JOURNEY_RUNNER_SSH": RUNNER,
            # values in the caller's environment that no script may carry along
            "CLAUDE_CODE_OAUTH_TOKEN": SECRET, "API_SERVER_KEY": SECRET,
        }
        for name, value in changes.items():
            if value is None:
                env.pop(name, None)
            else:
                env[name] = value
        return env

    def run(self, script: Path, **changes: str | None) -> subprocess.CompletedProcess[str]:
        copy = self.scripts / script.name
        shutil.copy2(script, copy)
        return subprocess.run(
            [str(copy)], env=self.env(**changes), stdin=subprocess.DEVNULL, capture_output=True,
            text=True, timeout=60, check=False,
        )

    def calls(self) -> list[dict]:
        log = self.top / "calls.jsonl"
        return [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []

    def names(self) -> list[str]:
        return [call["call"] for call in self.calls()]

    def told(self) -> list[str]:
        return [call["stdin"] for call in self.calls() if call["call"] == "hermes"]

    def result(self) -> dict[str, str]:
        text = (self.run_dir / "result").read_text(encoding="utf-8")
        return dict(line.split("=", 1) for line in text.splitlines())

    def hold_lock(self):
        self.run_dir.mkdir(parents=True, exist_ok=True)
        handle = open(self.run_dir / "run.lock", "a")
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return handle


@pytest.fixture
def box(tmp_path: Path) -> Sandbox:
    if not all(shutil.which(tool) for tool in TOOLS):
        pytest.skip("needs bash and flock")
    return Sandbox(tmp_path)


def _everything_printed(box: Sandbox, done: subprocess.CompletedProcess[str]) -> str:
    """All that a run printed or wrote down: its output, its result file and every owner line."""
    parts = [done.stdout, done.stderr, *box.told()]
    if (box.run_dir / "result").exists():
        parts.append((box.run_dir / "result").read_text(encoding="utf-8"))
    return "\n".join(parts)


def test_the_owned_scripts_are_executable_bash():
    for script in (START, RUN_ONCE):
        assert os.access(script, os.X_OK), f"{script.name} must be executable"
        assert script.read_text().splitlines()[0] == "#!/usr/bin/env bash"


# ----------------------------------------------------------------------------------- start.sh


def test_start_takes_the_lock_and_starts_one_transient_user_service(box):
    done = box.run(START)

    assert done.returncode == 0, done.stdout + done.stderr
    listing, launch = box.calls()[0]["argv"], box.calls()[1]["argv"]
    assert box.names() == ["systemctl", "systemd-run"]
    assert listing[:3] == ["systemctl", "--user", "list-units"] and "hermes-journeys.service" in listing
    options, command = launch[1 : launch.index("--")], launch[launch.index("--") + 1 :]
    # A user service of that name, collected once it exits, with a hard time limit and no restart.
    assert options[:3] == ["--user", "--unit=hermes-journeys.service", "--collect"]
    assert any(re.fullmatch(r"--property=RuntimeMaxSec=[1-9][0-9]*", option) for option in options)
    assert not any("Restart" in option for option in options)
    assert command == [str(box.scripts / "run_once.sh")]
    # What it carries along is the stack's names and addresses, never a value of the caller's secrets.
    assert f"--setenv=JOURNEY_RUNNER_SSH={RUNNER}" in options
    assert all(SECRET not in arg for call in box.calls() for arg in call["argv"])
    assert not any("OAUTH" in option or "API_SERVER_KEY" in option for option in options)
    # The lock is not left held: a run can take it.
    handle = box.hold_lock()
    handle.close()


def test_start_refuses_a_second_run_while_the_lock_is_held(box):
    handle = box.hold_lock()
    try:
        done = box.run(START)
    finally:
        handle.close()

    assert done.returncode != 0
    assert done.stdout.splitlines() == [LOCK_REASON]
    assert box.names() == []  # it asked nothing and started nothing


@pytest.mark.parametrize("state", ["active", "activating", "deactivating"])
def test_start_refuses_a_second_run_while_the_unit_is_loaded(box, state):
    box.answer(units=[f"hermes-journeys.service loaded {state} running Owner journeys"])

    done = box.run(START)

    assert done.returncode != 0
    assert done.stdout.splitlines() == [UNIT_REASON]
    assert box.names() == ["systemctl"]  # asked, then refused: nothing started


def test_start_starts_when_the_manager_keeps_only_a_failed_unit_like_the_release_unit(box):
    box.answer(units=["hermes-journeys.service loaded failed failed Owner journeys"])

    done = box.run(START)

    assert done.returncode == 0
    assert box.names() == ["systemctl", "systemd-run"]


def test_start_ignores_other_units_in_the_listing(box):
    box.answer(units=["hermes-journeys-other.service loaded active running Something else"])

    assert box.run(START).returncode == 0
    assert box.names() == ["systemctl", "systemd-run"]


def test_start_refuses_when_systemd_run_refuses_a_loaded_name(box):
    words = "Failed to start transient service unit: Unit hermes-journeys.service was already loaded.\n"
    box.answer(launch=[1, words])

    done = box.run(START)

    assert done.returncode != 0
    assert done.stdout.splitlines()[0] == "Refused: the unit hermes-journeys could not be started."
    assert words.strip() in done.stdout
    assert box.names() == ["systemctl", "systemd-run"]


def test_start_refuses_when_the_user_manager_cannot_be_asked(box):
    box.answer(systemctl_rc=1)

    done = box.run(START)

    assert done.returncode != 0
    assert done.stdout.splitlines() == ["Refused: the user manager could not be asked about hermes-journeys."]
    assert box.names() == ["systemctl"]


def test_start_refuses_without_systemd_run_and_starts_nothing(box):
    (box.bin / "systemd-run").unlink()

    done = box.run(START)

    assert done.returncode != 0
    assert done.stdout.splitlines() == ["Refused: systemd-run is missing."]
    assert box.names() == []


# -------------------------------------------------------------------------------- run_once.sh


def test_a_passing_run_goes_through_the_steps_in_order_and_writes_one_result_and_one_owner_line(box):
    done = box.run(RUN_ONCE)

    assert done.returncode == 0, done.stdout + done.stderr
    assert box.names() == [*ISOLATION, RESET, "curl proxy", "curl app", "ssh", "hermes"]
    by_name = {call["call"]: call["argv"] for call in box.calls()}
    assert by_name["curl proxy"][-1] == PROXY_URL
    assert by_name["curl app"][-1] == BASE_URL
    assert by_name["ssh"] == ["ssh", RUNNER, "journey-run.sh"]
    assert by_name["hermes"] == ["hermes", "send", "--to", "telegram", "--file", "-"]
    # Exactly one result file, and exactly one owner line.
    assert sorted(path.name for path in box.run_dir.iterdir()) == ["result", "run.lock"]
    result = box.result()
    assert (result["result"], result["step"]) == ("passed", "journeys")
    assert re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ", result["finished"])
    assert box.told() == ["Journeys passed."]
    # Nothing was retried: every stand-in was called once.
    assert len(box.names()) == len(set(box.names()))


def test_the_wait_for_the_owner_app_polls_until_the_page_text_shows(box):
    box.answer(app=[{"rc": 0, "body": BANNER_PAGE}, {"rc": 22, "body": ""}, {"rc": 0, "body": GOOD_PAGE}])

    done = box.run(RUN_ONCE)

    assert done.returncode == 0
    names = box.names()
    # A 200 with a failure banner is not ready, nor is an error status; the third answer is.
    assert names.count("curl app") == 3 and names.count("sleep") == 2
    assert names.index("ssh") > max(i for i, name in enumerate(names) if name == "curl app")
    assert box.told() == ["Journeys passed."]


# Each step: how it fails, the calls that happen up to and including the failure, and nothing after.
PROXY_UP = [*ISOLATION, RESET]
FAILURES = {
    "config-page-text": ("config", {"env": {"JOURNEY_PAGE_TEXT": None}}, []),
    "config-base-url": ("config", {"env": {"JOURNEY_BASE_URL": "http://workspace.synthetic.test"}}, []),
    "config-runner": ("config", {"env": {"JOURNEY_RUNNER_SSH": "-oProxyCommand=evil"}}, []),
    "isolation-server2": ("isolation", {"rc": {ISOLATION[0]: 1}}, ISOLATION[:1]),
    "isolation-server1": ("isolation", {"rc": {ISOLATION[1]: 1}}, ISOLATION),
    "reset": ("reset", {"rc": {RESET: 1}}, [*ISOLATION, RESET]),
    "proxy-unreachable": ("proxy", {"proxy": {"rc": 7, "body": ""}}, [*PROXY_UP, "curl proxy"]),
    "proxy-error-status": ("proxy", {"proxy": {"rc": 22, "body": ""}}, [*PROXY_UP, "curl proxy"]),
    "proxy-wrong-answer": ("proxy", {"proxy": {"rc": 0, "body": '{"keys":[]}'}}, [*PROXY_UP, "curl proxy"]),
    "owner-app-banner": (
        "owner-app", {"app": [{"rc": 0, "body": BANNER_PAGE}]},
        [*PROXY_UP, "curl proxy", *["curl app", "sleep"] * 60, "curl app"],
    ),
    "owner-app-down": (
        "owner-app", {"app": [{"rc": 7, "body": ""}]},
        [*PROXY_UP, "curl proxy", *["curl app", "sleep"] * 60, "curl app"],
    ),
    "journeys-fail": ("journeys", {"ssh_rc": 1}, [*PROXY_UP, "curl proxy", "curl app", "ssh"]),
    "journeys-ssh-down": ("journeys", {"ssh_rc": 255}, [*PROXY_UP, "curl proxy", "curl app", "ssh"]),
}


@pytest.mark.parametrize("case", FAILURES)
def test_a_failed_step_ends_the_run_with_its_name_and_reason_and_runs_no_later_step(box, case):
    step, change, calls_until_failure = FAILURES[case]
    box.answer(**{k: v for k, v in change.items() if k != "env"})

    done = box.run(RUN_ONCE, **change.get("env", {}))

    assert done.returncode != 0
    # The calls stop at the failed step: no later step ran, and the failed step ran once (no retry).
    assert [name for name in box.names() if name != "hermes"] == calls_until_failure
    result = box.result()
    assert (result["result"], result["step"], result["reason"]) == ("failed", step, REASONS[step])
    assert sorted(path.name for path in box.run_dir.iterdir()) == ["result", "run.lock"]
    assert box.told() == [f"Journeys failed at step {step}: {REASONS[step]}."]
    assert box.names()[-1] == "hermes"


def test_a_second_run_does_not_start_while_the_lock_is_held(box):
    handle = box.hold_lock()
    try:
        done = box.run(RUN_ONCE)
    finally:
        handle.close()

    assert done.returncode != 0
    assert box.names() == []  # no step, no result, no owner line
    assert not (box.run_dir / "result").exists()


def test_a_run_that_cannot_tell_the_owner_still_writes_its_result_and_fails(box):
    box.answer(hermes_rc=1)

    done = box.run(RUN_ONCE)

    assert done.returncode != 0
    assert box.result()["result"] == "passed"
    assert box.names().count("hermes") == 1  # told once, not again


def test_the_new_result_replaces_the_old_one(box):
    assert box.run(RUN_ONCE).returncode == 0
    box.answer(ssh_rc=1)
    (box.top / "calls.jsonl").unlink()

    assert box.run(RUN_ONCE).returncode != 0

    assert box.result()["result"] == "failed"
    assert sorted(path.name for path in box.run_dir.iterdir()) == ["result", "run.lock"]


# ------------------------------------------------------------------------------------ secrets


@pytest.mark.parametrize("case", ["passing", *FAILURES])
def test_no_secret_value_and_no_page_text_is_printed_anywhere(box, case):
    if case != "passing":
        step, change, _calls = FAILURES[case]
        box.answer(**{k: v for k, v in change.items() if k != "env"})
        done = box.run(RUN_ONCE, **change.get("env", {}))
    else:
        done = box.run(RUN_ONCE)

    printed = _everything_printed(box, done)
    for sentinel in SENTINELS:
        assert sentinel not in printed, sentinel
    # Nor does any argument of any call carry one: the secrets live in the caller's environment only.
    for call in box.calls():
        assert not any(SECRET in arg or "PAGE-" in arg for arg in call["argv"]), call
    assert "docker" not in box.names()  # the scripts leave docker to the reset and the check


def test_start_prints_no_secret_value(box):
    done = box.run(START)

    assert done.returncode == 0
    assert SECRET not in done.stdout + done.stderr
