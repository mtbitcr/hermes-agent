"""Tests for the journeys start script and the one-run script (card P-3).

start.sh and run_once.sh run for real, in a temporary copy of the journeys folder, against
stand-ins that sit alone on PATH: systemd-run, systemctl, docker, ssh, curl, hermes, sleep and mv.
Every stand-in logs its call to one file, in order, and answers from answers.json. The Server 2
parts run through the fake ssh, which only records the command it was given; the integration test
at the end lets the fake ssh run that command against the real isolation_check.sh and
reset_stack.sh with a fake docker. No test reaches a real host, service, container or remote
machine.
"""

from __future__ import annotations

import fcntl
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
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
PROD = "https://prod.invalid"
RUNNER = "runner@server2.invalid"
SENTINELS = [SECRET, "PAGE-BODY-SENTINEL", "PAGE-BANNER-SENTINEL", PAGE_TEXT]

S2_DIR = "/home/raphael-runner/raphael-journeys"
CONTAINER = "raphael-journeys-browser"
BROWSER_DIR = "/opt/raphael-journeys"

LOCK_REASON = "Refused: a journey run is still at work (the run lock is held)."
UNIT_REASON = "Refused: the unit hermes-journeys is still loaded."
LAUNCH_REASON = "Refused: the unit hermes-journeys could not be started."
REASONS = {
    "config": "a required setting is missing or is not plain",
    "isolation": "the isolation check failed",
    "reset": "the stack reset failed",
    "proxy": "the sign-in proxy did not answer",
    "owner-app": "the owner app did not show its page text in time",
    "journeys": "the journey runner failed",
    "result": "the result file could not be stored",
    "notify": "the owner could not be told",
}

# The calls of a run, as the stand-ins label them.
ISO2 = "ssh isolation_check.sh server2"
ISO1 = "isolation_check.sh server1"
RESET = "ssh reset_stack.sh"
JOURNEYS_CALL = "ssh journey-run.sh"
SEED_CALL = "ssh seed_stack.sh"
LOOP_ORDER = [3, 5, 2, 1, 4]
SEEDED = (3, 5, 2)
ISOLATION = [ISO2, ISO1]

CMD_ISO2 = f"cd '{S2_DIR}' && JOURNEY_PRODUCTION_ORIGIN='{PROD}' ./isolation_check.sh server2"
CMD_RESET = f"cd '{S2_DIR}' && ./reset_stack.sh"
CMDS_JOURNEYS = [
    f"JOURNEY_BASE_URL='{BASE_URL}' JOURNEY_PRODUCTION_ORIGIN='{PROD}' "
    f"docker exec -w '{BROWSER_DIR}' -e JOURNEY_BASE_URL -e JOURNEY_PRODUCTION_ORIGIN "
    f"'{CONTAINER}' sh tools/page-check/journey-run.sh {n}"
    for n in LOOP_ORDER
]

STAND_IN = r'''
import json, os, subprocess, sys, time
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


def curl_spec(label):
    if label == "curl proxy":
        return answers["proxy"]
    asked = calls_so_far("curl app") - 1
    return answers["app"][min(asked, len(answers["app"]) - 1)]


def hang_if_asked(label):
    if answers.get("hang") == label:
        (top / "hang.marker").write_text("hanging")
        time.sleep(120)


if name == "isolation_check.sh":
    label = " ".join([name, *args])
    record(label)
    leak(answers["rc"].get(label, 0))
if name == "docker":
    record("docker")
    sys.exit(98)
if name == "sleep":
    record("sleep")
    sys.exit(0)
if name == "mv":
    count_file = top / "mv.count"
    count = int(count_file.read_text()) + 1 if count_file.exists() else 1
    count_file.write_text(str(count))
    if count in answers.get("mv_fail_on", []):
        sys.stderr.write("mv: leak: %s\n" % answers["secret"])
        sys.exit(1)
    os.execv(answers["real_mv"], ["mv", *args])
if name == "systemctl":
    record("systemctl")
    if "list-units" in args:
        for line in answers["units"]:
            print(line)
    sys.exit(answers["systemctl_rc"])
if name == "systemd-run":
    record("systemd-run")
    code, error = answers["launch"]
    sys.stdout.write(error)
    sys.stderr.write(error)
    sys.exit(code)
if name == "ssh":
    command = args[-1]
    if "isolation_check.sh server2" in command:
        label = "ssh isolation_check.sh server2"
    elif "reset_stack.sh" in command:
        label = "ssh reset_stack.sh"
    elif "seed_stack.sh" in command:
        label = "ssh seed_stack.sh"
    elif "journey-run.sh" in command:
        label = "ssh journey-run.sh"
    elif " curl " in command:
        # the proxy and owner app answers: curl inside the browser container, its output kept
        label = "curl proxy" if answers["proxy_url"] in command else "curl app"
    else:
        label = "ssh other"
    record(label)
    hang_if_asked(label)
    if answers.get("remote_path"):
        env = {"PATH": answers["remote_path"], "HOME": str(top)}
        env.update(answers.get("remote_env", {}))
        done = subprocess.run(["sh", "-c", command], env=env, stdin=subprocess.DEVNULL,
                              capture_output=True, text=True)
        if label.startswith("curl"):
            sys.stdout.write(done.stdout)
            sys.exit(done.returncode)
        leak(done.returncode)
    if label.startswith("curl"):
        spec = curl_spec(label)
        sys.stdout.write(spec["body"])
        sys.exit(spec["rc"])
    rc = answers["rc"].get(label, 0)
    if isinstance(rc, list):  # one answer per call of this label, the last one for all later calls
        rc = rc[min(calls_so_far(label), len(rc)) - 1]
    leak(rc)
if name == "hermes":
    record("hermes")
    sys.exit(answers["hermes_rc"])
if name == "curl":
    record("local curl")  # run_once.sh no longer runs curl on Server 1
    sys.exit(97)
sys.exit("stand-in: unknown name " + name)
'''

TOOLS = ("bash", "flock", "rm", "mkdir", "dirname", "date")


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
        self.runner_env = top / "runner.env"
        self.home.mkdir()
        self.bin.mkdir()
        self.scripts.mkdir()
        (top / "stand_in.py").write_text(STAND_IN, encoding="utf-8")
        for name in ("systemd-run", "systemctl", "docker", "ssh", "curl", "hermes", "sleep", "mv"):
            self._wrapper(self.bin / name)
        for name in TOOLS:
            (self.bin / name).symlink_to(shutil.which(name))
        # The Server 1 part of the isolation check is a stand-in here; the integration test uses the real one.
        self._wrapper(self.scripts / "isolation_check.sh")
        # The refresh has its own tests (test_journeys_run_loop.py); here it is a refresh that succeeds.
        (self.scripts / "refresh_stack.sh").write_text("#!/bin/sh\nexit 0\n")
        (self.scripts / "refresh_stack.sh").chmod(0o755)
        shutil.copy2(PROXY_SOURCE, self.scripts / "synthetic_proxy.py")
        self.write_runner_env()
        self.answer()

    def _wrapper(self, path: Path) -> None:
        path.write_text(f'#!/bin/sh\nexec "{sys.executable}" -I -S "{self.top / "stand_in.py"}" "$0" "$@"\n')
        path.chmod(0o755)

    def write_runner_env(self, text: str | None = None) -> None:
        if text is None:
            text = f"JOURNEY_BASE_URL={BASE_URL}\nJOURNEY_PRODUCTION_ORIGIN={PROD}\n"
        self.runner_env.write_text(text, encoding="utf-8")

    def answer(self, **changes) -> None:
        answers = {
            "rc": {}, "units": [], "systemctl_rc": 0, "launch": [0, ""], "hermes_rc": 0,
            "proxy": HEALTHY_PROXY, "proxy_url": PROXY_URL,
            "app": [{"rc": 0, "body": GOOD_PAGE}], "secret": SECRET, "page": GOOD_PAGE,
            "real_mv": shutil.which("mv"),
        }
        answers.update(changes)
        (self.top / "answers.json").write_text(json.dumps(answers), encoding="utf-8")

    def env(self, **changes: str | None) -> dict[str, str]:
        env = {
            "PATH": str(self.bin), "HOME": str(self.top), "HERMES_HOME": str(self.home),
            "JOURNEY_PAGE_TEXT": PAGE_TEXT, "JOURNEY_RUNNER_SSH": RUNNER,
            "JOURNEY_RUNNER_ENV": str(self.runner_env),
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

    def call(self, label: str) -> dict:
        (found,) = [call for call in self.calls() if call["call"] == label]
        return found

    def commands(self, label: str) -> list[str]:
        """The remote command of each call with this label, in order."""
        return [call["argv"][2] for call in self.calls() if call["call"] == label]

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

    def start_and_kill(self, hang: str, **changes: str | None) -> subprocess.CompletedProcess[str]:
        """Run run_once.sh until it hangs inside the call `hang`, then kill its whole group with SIGKILL."""
        answers = json.loads((self.top / "answers.json").read_text(encoding="utf-8"))
        answers["hang"] = hang
        (self.top / "answers.json").write_text(json.dumps(answers), encoding="utf-8")
        copy = self.scripts / RUN_ONCE.name
        shutil.copy2(RUN_ONCE, copy)
        proc = subprocess.Popen(
            [str(copy)], env=self.env(**changes), stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True, start_new_session=True,
        )
        marker = self.top / "hang.marker"
        deadline = time.monotonic() + 30
        try:
            while not marker.exists():
                assert proc.poll() is None and time.monotonic() < deadline, "the run did not reach the call"
                time.sleep(0.05)
            os.killpg(proc.pid, signal.SIGKILL)
            out, err = proc.communicate(timeout=30)
        finally:
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
        marker.unlink()
        return subprocess.CompletedProcess(proc.args, proc.returncode, out, err)


@pytest.fixture
def box(tmp_path: Path) -> Sandbox:
    if not all(shutil.which(tool) for tool in (*TOOLS, "mv", "sh")):
        pytest.skip("needs bash and flock")
    return Sandbox(tmp_path)


def _everything_printed(box: Sandbox, done: subprocess.CompletedProcess[str]) -> str:
    """All that a run printed or wrote down: its output, its result file and every owner line."""
    parts = [done.stdout, done.stderr, *box.told()]
    if (box.run_dir / "result").exists():
        parts.append((box.run_dir / "result").read_text(encoding="utf-8"))
    return "\n".join(parts)


def _header() -> str:
    """The comment block that opens run_once.sh, where every setting is documented."""
    lines = RUN_ONCE.read_text().splitlines()[1:]
    return "\n".join(lines[: next(i for i, line in enumerate(lines) if not line.startswith("#"))])


def test_the_owned_scripts_are_executable_bash():
    for script in (START, RUN_ONCE):
        assert os.access(script, os.X_OK), f"{script.name} must be executable"
        assert script.stat().st_mode & 0o777 == 0o755, f"{script.name} must have mode 755"
        assert script.read_text().splitlines()[0] == "#!/usr/bin/env bash"


def test_the_script_header_names_every_variable_its_default_and_who_installs_what():
    header = _header()
    for needle in (
        "JOURNEY_S2_SCRIPTS_DIR", "/home/raphael-runner/raphael-journeys",
        "JOURNEY_BROWSER_CONTAINER", "raphael-journeys-browser",
        "JOURNEY_BROWSER_DIR", "/opt/raphael-journeys",
        "JOURNEY_RUNNER_ENV", "~/.hermes/journeys/runner.env",
        "JOURNEY_BASE_URL", "JOURNEY_PRODUCTION_ORIGIN", "JOURNEY_RUNNER_SSH", "JOURNEY_PAGE_TEXT",
        "HERMES_HOME", "operator",
    ):
        assert needle in header, needle


def test_every_name_run_once_reads_from_its_environment_is_documented_and_passed_on_by_start():
    source = RUN_ONCE.read_text()
    header = _header()
    read = set(re.findall(r"\$\{?(JOURNEY_[A-Z0-9_]+|HERMES_HOME)\b", source))
    assert {"JOURNEY_RUNNER_SSH", "JOURNEY_PAGE_TEXT", "HERMES_HOME"} <= read
    passed_on = set(re.search(r"PASS_ON=\(([^)]*)\)", START.read_text()).group(1).split())
    assert read <= passed_on, read - passed_on
    assert all(name in header for name in read)


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
    assert f"--setenv=JOURNEY_RUNNER_ENV={box.runner_env}" in options
    assert all(SECRET not in arg for call in box.calls() for arg in call["argv"])
    assert not any("OAUTH" in option or "API_SERVER_KEY" in option for option in options)
    # The lock is not left held: a run can take it.
    handle = box.hold_lock()
    handle.close()


def test_start_passes_on_the_owner_named_settings_when_they_are_set(box):
    done = box.run(
        START, JOURNEY_S2_SCRIPTS_DIR="/srv/s2", JOURNEY_BROWSER_CONTAINER="browser-x",
        JOURNEY_BROWSER_DIR="/srv/browser",
    )

    assert done.returncode == 0
    options = box.call("systemd-run")["argv"]
    for setting in (
        "JOURNEY_S2_SCRIPTS_DIR=/srv/s2", "JOURNEY_BROWSER_CONTAINER=browser-x",
        "JOURNEY_BROWSER_DIR=/srv/browser",
    ):
        assert f"--setenv={setting}" in options


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


@pytest.mark.parametrize("code", [1, 127])
def test_start_prints_only_the_fixed_reason_when_systemd_run_fails(box, code):
    noise = (
        f"Failed to start transient service unit: Unit hermes-journeys.service was already loaded. "
        f"JOURNEY_PAGE_TEXT={PAGE_TEXT} token={SECRET} PAGE-BODY-SENTINEL\n"
    )
    box.answer(launch=[code, noise])

    done = box.run(START)

    assert done.returncode != 0
    assert done.stdout.splitlines() == [LAUNCH_REASON]
    assert done.stderr == ""
    assert box.names() == ["systemctl", "systemd-run"]  # one attempt, no retry


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


def test_start_prints_no_secret_value(box):
    done = box.run(START)

    assert done.returncode == 0
    for sentinel in SENTINELS:
        assert sentinel not in done.stdout + done.stderr


# -------------------------------------------------------------------------------- run_once.sh

TIME_RE = r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ"
LOOP_CALLS = [
    call
    for n in LOOP_ORDER
    for call in (RESET, *([SEED_CALL] if n in SEEDED else []), "curl proxy", "curl app", JOURNEYS_CALL)
]
PASSING_CALLS = [RESET, *ISOLATION, *LOOP_CALLS, RESET, "hermes"]


def nth_reset_fails(n: int, rc: int = 1) -> dict:
    """The answers for a run in which only the n-th reset (1 is the run's first) fails."""
    return {"rc": {RESET: [0] * (n - 1) + [rc]}}


def test_a_passing_run_goes_through_the_steps_in_order_and_writes_one_result_and_one_owner_line(box):
    done = box.run(RUN_ONCE)

    assert done.returncode == 0, done.stdout + done.stderr
    assert box.names() == PASSING_CALLS
    # The proxy and owner app answers come from curl inside the browser container, through the runner identity.
    for command in box.commands("curl proxy"):
        assert command.startswith(f"docker exec '{CONTAINER}' curl ") and command.endswith(f" '{PROXY_URL}'")
    for command in box.commands("curl app"):
        assert command.startswith(f"docker exec '{CONTAINER}' curl ") and command.endswith(f" '{BASE_URL}'")
    # Server 2 parts go through the runner identity, in the Server 2 scripts folder; the Server 1
    # part of the isolation check runs locally.
    assert box.call(ISO2)["argv"] == ["ssh", RUNNER, CMD_ISO2]
    assert box.call(ISO1)["argv"] == ["isolation_check.sh", "server1"]
    assert box.commands(RESET) == [CMD_RESET] * 7
    assert box.commands(JOURNEYS_CALL) == CMDS_JOURNEYS
    assert box.call("hermes")["argv"] == ["hermes", "send", "--to", "telegram", "--file", "-"]
    # Exactly one result file, and exactly one owner line.
    assert sorted(path.name for path in box.run_dir.iterdir()) == ["result", "run.lock"]
    result = box.result()
    assert (result["result"], result["step"]) == ("passed", "journeys")
    assert re.fullmatch(TIME_RE, result["started"]) and re.fullmatch(TIME_RE, result["finished"])
    assert box.told() == ["Journeys passed."]
    # Nothing was retried: each journey ran its steps once.
    assert [box.names().count(call) for call in (RESET, SEED_CALL, JOURNEYS_CALL)] == [7, 3, 5]


def test_the_settings_named_by_the_owner_change_the_server_2_calls(box):
    done = box.run(
        RUN_ONCE, JOURNEY_S2_SCRIPTS_DIR="/srv/s2-scripts", JOURNEY_BROWSER_CONTAINER="browser-x",
        JOURNEY_BROWSER_DIR="/srv/browser-dir",
    )

    assert done.returncode == 0, done.stdout + done.stderr
    assert box.call(ISO2)["argv"][2] == CMD_ISO2.replace(S2_DIR, "/srv/s2-scripts")
    assert box.commands(RESET) == [CMD_RESET.replace(S2_DIR, "/srv/s2-scripts")] * 7
    expected = [c.replace(BROWSER_DIR, "/srv/browser-dir").replace(CONTAINER, "browser-x") for c in CMDS_JOURNEYS]
    assert box.commands(JOURNEYS_CALL) == expected


def test_the_runner_settings_file_is_read_from_its_default_place(box):
    default = box.top / ".hermes" / "journeys" / "runner.env"
    default.parent.mkdir(parents=True)
    box.runner_env.rename(default)

    done = box.run(RUN_ONCE, JOURNEY_RUNNER_ENV=None)

    assert done.returncode == 0, done.stdout + done.stderr
    assert box.commands(JOURNEYS_CALL) == CMDS_JOURNEYS


def test_the_journeys_docker_exec_passes_exactly_the_two_named_variables(box):
    box.write_runner_env(
        f"# the stack's two addresses\n\nJOURNEY_BASE_URL={BASE_URL}\nJOURNEY_PRODUCTION_ORIGIN={PROD}\n"
    )
    assert box.run(RUN_ONCE).returncode == 0

    for n, command in zip(LOOP_ORDER, box.commands(JOURNEYS_CALL), strict=True):
        prefix, _, rest = command.partition(" docker exec ")
        # Only these two names are set for docker, and docker passes on only these two names, no value.
        assert re.findall(r"(\w+)='", prefix) == ["JOURNEY_BASE_URL", "JOURNEY_PRODUCTION_ORIGIN"]
        assert re.findall(r"-e (\S+)", rest) == ["JOURNEY_BASE_URL", "JOURNEY_PRODUCTION_ORIGIN"]
        assert "--env" not in rest and "--env-file" not in rest and "=" not in rest
        # ... and the one digit of its journey.
        assert rest.endswith(f"'{CONTAINER}' sh tools/page-check/journey-run.sh {n}")
        assert SECRET not in command


def test_the_wait_for_the_owner_app_polls_until_the_page_text_shows(box):
    box.answer(app=[{"rc": 0, "body": BANNER_PAGE}, {"rc": 22, "body": ""}, {"rc": 0, "body": GOOD_PAGE}])

    done = box.run(RUN_ONCE)

    assert done.returncode == 0
    names = box.names()
    # A 200 with a failure banner is not ready, nor is an error status; the third answer is (journey 3's wait).
    # The four later journeys find the page at once.
    assert names.count("curl app") == 3 + 4 and names.count("sleep") == 2
    third = [i for i, name in enumerate(names) if name == "curl app"][2]
    assert names.index(JOURNEYS_CALL) > third
    assert box.told() == ["Journeys passed."]


# Each step: how it fails, the calls that happen up to and including the failure, and nothing after.
UP_TO_ISOLATION = [RESET, *ISOLATION]
UP_TO_RESET = [*UP_TO_ISOLATION, RESET]
ALL_CALLS = [RESET, *ISOLATION, *LOOP_CALLS, RESET]
UP_TO_APP = [*UP_TO_RESET, SEED_CALL, "curl proxy"]
FULL_WAIT = [*UP_TO_APP, *["curl app", "sleep"] * 60, "curl app"]
BAD_ENV_FILES = {
    "extra-name": f"JOURNEY_BASE_URL={BASE_URL}\nJOURNEY_PRODUCTION_ORIGIN={PROD}\nANTHROPIC_TOKEN={SECRET}\n",
    "extra-name-with-spaces": f"JOURNEY_BASE_URL={BASE_URL}\nJOURNEY_PRODUCTION_ORIGIN={PROD}\n  API_SERVER_KEY = {SECRET}\n",
    "missing-origin": f"JOURNEY_BASE_URL={BASE_URL}\n",
    "missing-base-url": f"JOURNEY_PRODUCTION_ORIGIN={PROD}\n",
    "not-https": f"JOURNEY_BASE_URL=http://workspace.synthetic.test\nJOURNEY_PRODUCTION_ORIGIN={PROD}\n",
    "not-plain": f"JOURNEY_BASE_URL={BASE_URL};id\nJOURNEY_PRODUCTION_ORIGIN={PROD}\n",
    "not-a-setting": f"JOURNEY_BASE_URL={BASE_URL}\nJOURNEY_PRODUCTION_ORIGIN={PROD}\nthis is no setting\n",
    "empty": "",
}
FAILURES = {
    "config-page-text": ("config", {"env": {"JOURNEY_PAGE_TEXT": None}}, []),
    "config-runner": ("config", {"env": {"JOURNEY_RUNNER_SSH": "-oProxyCommand=evil"}}, []),
    "config-no-settings-file": ("config", {"runner_env": None}, []),
    "config-s2-dir": ("config", {"env": {"JOURNEY_S2_SCRIPTS_DIR": "/srv/x'; id; '"}}, []),
    "config-container": ("config", {"env": {"JOURNEY_BROWSER_CONTAINER": "x; id"}}, []),
    "config-browser-dir": ("config", {"env": {"JOURNEY_BROWSER_DIR": "relative/dir"}}, []),
    **{f"config-file-{key}": ("config", {"runner_env": text}, []) for key, text in BAD_ENV_FILES.items()},
    "reset-first": ("reset", nth_reset_fails(1), [RESET]),
    "reset-first-ssh-down": ("reset", nth_reset_fails(1, 255), [RESET]),
    "isolation-server2": ("isolation", {"rc": {ISO2: 1}}, [RESET, ISO2]),
    "isolation-server2-ssh-down": ("isolation", {"rc": {ISO2: 255}}, [RESET, ISO2]),
    "isolation-server1": ("isolation", {"rc": {ISO1: 1}}, UP_TO_ISOLATION),
    "reset": ("reset", nth_reset_fails(2), UP_TO_RESET),
    "reset-ssh-down": ("reset", nth_reset_fails(2, 255), UP_TO_RESET),
    "proxy-unreachable": ("proxy", {"proxy": {"rc": 7, "body": ""}}, UP_TO_APP),
    "proxy-error-status": ("proxy", {"proxy": {"rc": 22, "body": ""}}, UP_TO_APP),
    "proxy-wrong-answer": ("proxy", {"proxy": {"rc": 0, "body": '{"keys":[]}'}}, UP_TO_APP),
    "owner-app-banner": ("owner-app", {"app": [{"rc": 0, "body": BANNER_PAGE}]}, FULL_WAIT),
    "owner-app-down": ("owner-app", {"app": [{"rc": 7, "body": ""}]}, FULL_WAIT),
    "journeys-fail": ("journeys", {"rc": {JOURNEYS_CALL: 1}}, [*UP_TO_APP, "curl app", JOURNEYS_CALL]),
    "journeys-ssh-down": ("journeys", {"rc": {JOURNEYS_CALL: 255}}, [*UP_TO_APP, "curl app", JOURNEYS_CALL]),
    "reset-last": ("reset", nth_reset_fails(7), ALL_CALLS),
    "reset-last-ssh-down": ("reset", nth_reset_fails(7, 255), ALL_CALLS),
}


# A failure in the loop names its journey; the first journey is 3, and every case here fails in it.
# The resets before the first journey and after the last one belong to no journey.
LOOP_STEPS = ("reset", "seed", "proxy", "owner-app", "journeys")
NO_JOURNEY = ("reset-first", "reset-first-ssh-down", "reset-last", "reset-last-ssh-down")


def _arrange(box: Sandbox, change: dict) -> dict:
    box.answer(**{k: v for k, v in change.items() if k not in ("env", "runner_env")})
    if "runner_env" in change:
        if change["runner_env"] is None:
            box.runner_env.unlink()
        else:
            box.write_runner_env(change["runner_env"])
    return change.get("env", {})


@pytest.mark.parametrize("case", FAILURES)
def test_a_failed_step_ends_the_run_with_its_name_and_reason_and_runs_no_later_step(box, case):
    step, change, calls_until_failure = FAILURES[case]
    env = _arrange(box, change)

    done = box.run(RUN_ONCE, **env)

    assert done.returncode != 0
    # The calls stop at the failed step: no later step ran, and the failed step ran once (no retry).
    assert [name for name in box.names() if name != "hermes"] == calls_until_failure
    result = box.result()
    assert (result["result"], result["step"], result["reason"]) == ("failed", step, REASONS[step])
    assert re.fullmatch(TIME_RE, result["started"]) and re.fullmatch(TIME_RE, result["finished"])
    assert sorted(path.name for path in box.run_dir.iterdir()) == ["result", "run.lock"]
    named = " for journey 3" if step in LOOP_STEPS and case not in NO_JOURNEY else ""
    assert result.get("journey") == ("3" if named else None)
    assert box.told() == [f"Journeys failed at step {step}{named}: {REASONS[step]}."]
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


def test_a_second_run_that_is_refused_leaves_the_previous_result_alone(box):
    assert box.run(RUN_ONCE).returncode == 0
    before = (box.run_dir / "result").read_text()
    handle = box.hold_lock()
    try:
        done = box.run(RUN_ONCE)
    finally:
        handle.close()

    assert done.returncode != 0
    assert (box.run_dir / "result").read_text() == before


def test_the_new_result_replaces_the_old_one(box):
    assert box.run(RUN_ONCE).returncode == 0
    box.answer(rc={JOURNEYS_CALL: 1})
    (box.top / "calls.jsonl").unlink()

    assert box.run(RUN_ONCE).returncode != 0

    assert box.result()["result"] == "failed"
    assert sorted(path.name for path in box.run_dir.iterdir()) == ["result", "run.lock"]


# --------------------------------------------------------------------------- the started result


def test_a_hard_kill_with_no_previous_result_leaves_a_started_result_with_its_time(box):
    done = box.start_and_kill(hang=ISO2)

    assert done.returncode == -signal.SIGKILL
    result = box.result()
    assert result["result"] == "started"
    assert re.fullmatch(TIME_RE, result["started"])
    assert "finished" not in result
    assert box.names() == [RESET, ISO2]  # the first reset ran; it was in the isolation check when it died
    assert sorted(path.name for path in box.run_dir.iterdir()) == ["result", "run.lock"]


def test_a_hard_kill_after_a_passed_run_does_not_leave_the_passed_result(box):
    assert box.run(RUN_ONCE).returncode == 0
    assert box.result()["result"] == "passed"

    box.start_and_kill(hang=RESET)

    result = box.result()
    assert result["result"] == "started"
    assert re.fullmatch(TIME_RE, result["started"])
    assert "finished" not in result and result.get("step") != "journeys"


def test_the_started_result_is_written_before_the_first_step_and_replaced_atomically(box):
    box.start_and_kill(hang=ISO2)
    # Nothing but the finished file and the lock is left behind by the atomic write.
    assert not any(path.name.endswith(".new") for path in box.run_dir.iterdir())


# --------------------------------------------------------- storing the result and telling the owner


def test_a_result_that_cannot_be_stored_is_a_failed_step_and_no_passed_line_is_sent(box):
    box.answer(mv_fail_on=[2])  # the started result is stored; the terminal one is not

    done = box.run(RUN_ONCE)

    assert done.returncode != 0
    assert box.told() == [f"Journeys failed at step result: {REASONS['result']}."]
    assert "passed" not in "\n".join(box.told()).lower()
    assert box.result()["result"] != "passed"  # the started result stays; it is never called passed
    assert box.names().count("hermes") == 1  # told once, not again
    assert not any(path.name.endswith(".new") for path in box.run_dir.iterdir())


def test_a_failed_run_that_cannot_store_its_result_still_names_its_step_in_the_one_line(box):
    box.answer(mv_fail_on=[2], rc={JOURNEYS_CALL: 1})

    done = box.run(RUN_ONCE)

    assert done.returncode != 0
    assert len(box.told()) == 1
    assert box.told()[0].startswith(f"Journeys failed at step journeys for journey 3: {REASONS['journeys']}.")
    assert "could not be stored" in box.told()[0]


def test_a_started_result_that_cannot_be_stored_ends_the_run_before_any_step(box):
    box.answer(mv_fail_on=[1, 2, 3])

    done = box.run(RUN_ONCE)

    assert done.returncode != 0
    assert box.names() == ["hermes"]  # no step ran
    assert box.told() == [f"Journeys failed at step result: {REASONS['result']}."]
    assert not (box.run_dir / "result").exists()


def test_a_failed_owner_line_is_recorded_in_the_result_of_a_passing_run(box):
    box.answer(hermes_rc=1)

    done = box.run(RUN_ONCE)

    assert done.returncode != 0
    result = box.result()
    assert (result["result"], result["step"], result["reason"]) == ("failed", "notify", REASONS["notify"])
    assert "earlier_step" not in result
    assert box.names().count("hermes") == 1  # one attempt, no retry
    assert box.names().count(JOURNEYS_CALL) == 5  # the journeys did pass and are not run again


def test_a_failed_owner_line_is_recorded_in_the_result_of_a_failed_run_too(box):
    box.answer(hermes_rc=1, rc={RESET: 1})

    done = box.run(RUN_ONCE)

    assert done.returncode != 0
    result = box.result()
    assert (result["result"], result["step"], result["reason"]) == ("failed", "notify", REASONS["notify"])
    assert result["earlier_step"] == "reset"
    assert box.names().count("hermes") == 1


def test_a_failed_owner_line_and_a_failed_rewrite_never_leave_a_passed_result(box):
    box.answer(hermes_rc=1, mv_fail_on=[3])  # started and terminal stored; the amended one is not

    done = box.run(RUN_ONCE)

    assert done.returncode != 0
    assert box.names().count("hermes") == 1  # told once, not again
    assert box.names().count(JOURNEYS_CALL) == 5
    # The result could not be amended, so the passed one is taken away: a reader never sees "passed".
    assert not (box.run_dir / "result").exists()
    assert sorted(path.name for path in box.run_dir.iterdir()) == ["run.lock"]
    for sentinel in SENTINELS:
        assert sentinel not in done.stdout + done.stderr


def test_a_failed_owner_line_after_a_failed_run_and_a_failed_rewrite_leave_no_passed_result(box):
    box.run(RUN_ONCE)  # an older passed result is on file
    (box.top / "calls.jsonl").unlink()
    (box.top / "mv.count").unlink()
    box.answer(hermes_rc=1, mv_fail_on=[3], rc={RESET: 1})

    done = box.run(RUN_ONCE)

    assert done.returncode != 0
    assert not (box.run_dir / "result").exists()  # neither the older passed result nor a half record


# ------------------------------------------------------------------------------------ secrets


@pytest.mark.parametrize("case", ["passing", *FAILURES])
def test_no_secret_value_and_no_page_text_is_printed_anywhere(box, case):
    if case != "passing":
        _step, change, _calls = FAILURES[case]
        done = box.run(RUN_ONCE, **_arrange(box, change))
    else:
        done = box.run(RUN_ONCE)

    printed = _everything_printed(box, done)
    for sentinel in SENTINELS:
        assert sentinel not in printed, sentinel
    # Nor does any argument of any call carry one: the secrets live in the caller's environment only.
    for call in box.calls():
        assert not any(SECRET in arg or "PAGE-" in arg for arg in call["argv"]), call
    assert "docker" not in box.names()  # the scripts leave docker to the Server 2 side


@pytest.mark.parametrize("case", ["terminal-store", "started-store", "notify", "notify-and-rewrite"])
def test_no_secret_value_is_printed_when_the_finish_fails(box, case):
    box.answer(**{
        "terminal-store": {"mv_fail_on": [2]}, "started-store": {"mv_fail_on": [1, 2]},
        "notify": {"hermes_rc": 1}, "notify-and-rewrite": {"hermes_rc": 1, "mv_fail_on": [3]},
    }[case])

    done = box.run(RUN_ONCE)

    printed = _everything_printed(box, done)
    for sentinel in SENTINELS:
        assert sentinel not in printed, sentinel


# ----------------------------------------------------------- the real helpers behind fake ssh/docker

DOCKER_FAKE = r'''
import json, os, sys
from pathlib import Path

top = Path(__file__).parent
a = sys.argv[2:]
answers = json.loads((top / "answers.json").read_text(encoding="utf-8"))
where = os.environ.get("FAKE_DOCKER_WHERE", "?")
with open(top / "docker.jsonl", "a", encoding="utf-8") as handle:
    handle.write(json.dumps({"where": where, "argv": a}) + "\n")


def out(text="", rc=0):
    sys.stdout.write(text)
    sys.exit(rc)


if a[0] in answers.get("docker_fail", []):
    out("", 1)
if a[0] in ("stop", "start", "restart"):
    out()
if a[0] == "run":
    sys.stdin.buffer.read()
    out()
if a[0] == "inspect":
    kind = a[a.index("--type") + 1]
    fmt = a[a.index("--format") + 1]
    target = a[-1]
    home, tokens = "raphael-journeys-home", "raphael-journeys-tokens"
    if kind == "image":
        out("PATH\nHERMES_HOME\n")
    if fmt == "{{.Image}}":
        out("img:1\n")
    if "println (index (split" in fmt:
        out("PATH\n" + answers.get("extra_env_name", ""))
    if "println .Type .Name" in fmt:
        out(f"volume {home}\n" + (f"volume {tokens}\n" if "app" in target or "dashboard" in target else ""))
    if "println .Name .Destination" in fmt:
        mounts = {"app": f"{tokens} /tokens\n", "dashboard": f"{tokens} /tokens\n{home} /opt/data\n",
                  "gateway": f"{home} /opt/data\n"}
        out(next((v for k, v in mounts.items() if k in target), ""))
    if "eq (index $p 0)" in fmt:
        out("/opt/data")
    if "Healthcheck" in fmt:
        out("CMD")
    if ".State.Running" in fmt:
        out("true healthy")
    sys.exit("docker fake: unknown inspect " + fmt)
if a[0] == "exec":
    rest, i = [], 1
    while i < len(a):
        if a[i] in ("--user", "-w", "-e"):
            i += 2
        elif a[i] == "-i":
            i += 1
        else:
            rest = a[i:]
            break
    container, command = rest[0], rest[1:]
    stdin = "" if sys.stdin.isatty() else sys.stdin.read()
    if container == "raphael-journeys-browser" or container == answers.get("browser_container"):
        if command[0] == "curl":  # the proxy and owner app checks run in the browser container
            out(answers["proxy"]["body"] if command[-1] == answers["proxy_url"] else answers["app"][0]["body"])
        out("", answers["rc"].get("docker exec journeys", 0))
    if command[:2] == ["id", "-u"] or command[:2] == ["id", "-g"]:
        out("1001")
    if command[0] == "stat":
        out("999:999")
    if command[0] == "printenv":
        if container == "raphael-workspace":
            out(answers.get("production_issuer", "https://prod-issuer.invalid"))
        out("/tokens/" + command[1].lower() + ".tok")
    if command[0] == "python3":
        out("aaa.bbb.ccc\n")
    if command[0] == "hermes":
        out()
    if command[0] == "sh" and command[1] == "-c":
        script = command[2]
        if 'for f in "$HERMES_HOME/.env"' in script:
            out("default API_SERVER_KEY\nraphael-planner API_SERVER_KEY\n")
        if 'printf "entry' in script:
            out("entry raphael-planner\n")
        if "cron list" in script:
            out("No scheduled jobs.\nCreate one with 'hermes cron create ...' or the /cron command in chat.")
        if "hermes send" in script:
            out("send status 1")
        if "profiles_to_serve" in stdin:
            out("served default\nserved raphael-planner\n")
        if "dispatch_in_gateway" in stdin:
            out("dispatcher off\n")
        if "get_connected_platforms" in stdin:
            out("platforms api_server\n")
        out()
sys.exit("docker fake: unknown call " + " ".join(a))
'''


def _integration(box: Sandbox, **answers) -> dict[str, str]:
    """Real isolation_check.sh and reset_stack.sh: a Server 2 scripts folder run by the fake ssh on a
    fake Server 2 (hostname, docker, curl), and the Server 1 isolation check run locally with a fake
    docker. Returns the environment changes for run_once.sh."""
    s2 = box.top / "s2-scripts"
    s2.mkdir()
    for name in ("isolation_check.sh", "reset_stack.sh", "stack.env.template", "synthetic_proxy.py"):
        shutil.copy2(JOURNEYS / name, s2 / name)
    (s2 / "seed_stack.sh").write_text("#!/bin/sh\nexit 0\n")  # the seed has its own tests (test_journeys_seed.py)
    (s2 / "seed_stack.sh").chmod(0o755)
    shutil.copytree(JOURNEYS / "stack-home", s2 / "stack-home")
    shutil.copy2(JOURNEYS / "isolation_check.sh", box.scripts / "isolation_check.sh")  # the local one too
    shutil.copy2(JOURNEYS / "stack.env.template", box.scripts / "stack.env.template")
    (box.top / "docker_fake.py").write_text(DOCKER_FAKE, encoding="utf-8")
    remote = box.top / "remote-bin"
    remote.mkdir()
    for path, where in ((box.bin / "docker", "local"), (remote / "docker", "remote")):
        path.unlink(missing_ok=True)
        path.write_text(
            f'#!/bin/sh\nFAKE_DOCKER_WHERE={where} exec "{sys.executable}" -I -S "{box.top / "docker_fake.py"}" "$0" "$@"\n'
        )
        path.chmod(0o755)
    (remote / "hostname").write_text("#!/bin/sh\necho stack-host\n")
    (remote / "curl").write_text("#!/bin/sh\ncat > /dev/null\nprintf '401 '\n")
    for path in (remote / "hostname", remote / "curl"):
        path.chmod(0o755)
    box.answer(**{
        "remote_path": f"{remote}:/usr/bin:/bin", "remote_env": {"JOURNEY_STACK_HOST": "stack-host"}, **answers,
    })
    return {"JOURNEY_S2_SCRIPTS_DIR": str(s2)}


def _docker_calls(box: Sandbox) -> list[dict]:
    log = box.top / "docker.jsonl"
    return [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []


def test_the_real_isolation_check_and_reset_run_on_server_2_through_the_runner_identity(box):
    env = _integration(box)

    done = box.run(RUN_ONCE, **env)

    assert done.returncode == 0, done.stdout + done.stderr
    # The real Server 1 check is no stand-in, so only its docker call (below) shows that it ran.
    assert box.names() == [c for c in PASSING_CALLS if c != ISO1]
    assert (box.call(ISO2)["argv"][2]).startswith(f"cd '{env['JOURNEY_S2_SCRIPTS_DIR']}' && ")
    calls = _docker_calls(box)
    local = [c["argv"] for c in calls if c["where"] == "local"]
    remote = [c["argv"] for c in calls if c["where"] == "remote"]
    # Server 1 part: only production's container is asked, on Server 1's own docker.
    assert local == [["exec", "raphael-workspace", "printenv", "RAPHAEL_ACCESS_ISSUER"]]
    # Server 2 parts: the stack's containers, never production's; the reset stopped and started them.
    assert not any("raphael-workspace" in arg for argv in remote for arg in argv)
    assert ["stop", "raphael-journeys-gateway"] in remote and ["start", "raphael-journeys-browser"] in remote
    assert [argv for argv in remote if argv[-2] == "tools/page-check/journey-run.sh"] == [
        [
            "exec", "-w", BROWSER_DIR, "-e", "JOURNEY_BASE_URL", "-e", "JOURNEY_PRODUCTION_ORIGIN", CONTAINER,
            "sh", "tools/page-check/journey-run.sh", str(n),
        ]
        for n in LOOP_ORDER
    ]
    assert box.result()["result"] == "passed"
    assert box.told() == ["Journeys passed."]


def test_the_real_server_2_isolation_check_refuses_when_the_runner_is_not_on_the_stack_host(box):
    env = _integration(box, remote_env={"JOURNEY_STACK_HOST": "some-other-host"})

    done = box.run(RUN_ONCE, **env)

    assert done.returncode != 0
    # The real reset refuses on a host that is not the stack host, and it now runs first.
    assert box.names() == [RESET, "hermes"]
    assert (box.result()["step"], box.result()["reason"]) == ("reset", REASONS["reset"])
    assert not any(c["argv"][0] in ("stop", "start") for c in _docker_calls(box))  # nothing was reset


def test_the_real_server_1_isolation_check_refuses_when_production_names_the_proxys_issuer(box):
    env = _integration(box, production_issuer=ISSUER)

    done = box.run(RUN_ONCE, **env)

    assert done.returncode != 0
    assert box.names() == [RESET, ISO2, "hermes"]  # the first reset ran; Server 2 passed, Server 1 refused
    assert [c["argv"][:2] for c in _docker_calls(box) if c["where"] == "local"] == [["exec", "raphael-workspace"]]
    assert box.result()["step"] == "isolation"


def test_the_real_server_2_isolation_check_refuses_a_name_that_is_not_on_the_template(box):
    env = _integration(box, extra_env_name="SOME_UNLISTED_NAME\n")

    done = box.run(RUN_ONCE, **env)

    assert done.returncode != 0
    assert box.names() == [RESET, ISO2, "hermes"]
    assert box.result()["step"] == "isolation"


def test_a_failing_real_reset_stops_the_run_before_the_proxy_check(box):
    env = _integration(box, docker_fail=["restart"])

    done = box.run(RUN_ONCE, **env)

    assert done.returncode != 0
    assert box.names() == [RESET, "hermes"]
    assert (box.result()["result"], box.result()["step"]) == ("failed", "reset")
    printed = _everything_printed(box, done)
    assert all(sentinel not in printed for sentinel in SENTINELS)


def test_a_failing_journey_runner_in_the_browser_container_fails_the_journeys_step(box):
    env = _integration(box, rc={"docker exec journeys": 1})

    done = box.run(RUN_ONCE, **env)

    assert done.returncode != 0
    assert box.names() == [RESET, ISO2, RESET, SEED_CALL, "curl proxy", "curl app", JOURNEYS_CALL, "hermes"]
    assert (box.result()["result"], box.result()["step"]) == ("failed", "journeys")
