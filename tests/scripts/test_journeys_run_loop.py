"""Tests for the run of the journeys in a loop, with its refresh and its time step (cards 4).

start.sh and run_once.sh run for real, in a temporary copy of the journeys folder, against stand-ins that sit
alone on PATH: ssh, date, curl, hermes, sleep, systemd-run and systemctl; refresh_stack.sh and the Server 1
part of isolation_check.sh are stand-ins in that folder. Every stand-in logs its call to one file, in order.
The date stand-in is the clock: it answers a fake time that the test moves forward after chosen calls. The
Server 2 parts run through the fake ssh, which only records the command it was given. No test reaches a real
host, service, container or remote machine.
"""

from __future__ import annotations

import json
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
TEMPLATE = JOURNEYS / "stack.env.template"
WATCH = REPO / "hermes_cli" / "owner_write_watch.py"

PAGE_TEXT = "Owner workspace"
GOOD_PAGE = "<html>Owner workspace PAGE-BODY-SENTINEL</html>"
SECRET = "sk-ant-oat01-SENTINEL-secret-value-0123456789"
BASE_URL = "https://workspace.synthetic.test"
PROD = "https://prod.invalid"
RUNNER = "runner@server2.invalid"
S2_DIR = "/home/raphael-runner/raphael-journeys"
BROWSER_DIR = "/opt/raphael-journeys"

ORDER = [3, 5, 2, 1, 4]
SEEDED = {3, 5, 2}
# The fixed reserves of one journey, in minutes: reset + seed (3, 5 and 2 only) + owner app + budget plus 1.
RESERVES = {3: 4 + 1 + 2 + 2, 5: 4 + 1 + 2 + 6, 2: 4 + 1 + 2 + 4, 1: 4 + 2 + 21, 4: 4 + 2 + 16}
LIMIT_MINUTES = 118

REASONS = {
    "refresh": "the stack could not be brought to the released versions",
    "time": "the run's time ran out before the next journey",
    "reset": "the stack reset failed",
    "seed": "the seed failed",
    "proxy": "the sign-in proxy did not answer",
    "owner-app": "the owner app did not show its page text in time",
    "journeys": "the journey runner failed",
}

STAND_IN = r'''
import json, os, sys
from pathlib import Path

top = Path(__file__).parent
name = Path(sys.argv[1]).name
args = sys.argv[2:]
answers = json.loads((top / "answers.json").read_text(encoding="utf-8"))
stdin = "" if sys.stdin.isatty() else sys.stdin.read()
log = top / "calls.jsonl"
BASE = 1_000_000_000
ORDER = answers["order"]


def counter(key):
    path = top / (key + ".count")
    count = int(path.read_text()) + 1 if path.exists() else 1
    path.write_text(str(count))
    return count


def record(label, extra=None):
    entry = {"call": label, "argv": [name, *args], "stdin": stdin, "env": extra or {}}
    with open(log, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry) + "\n")
    after = answers.get("elapsed_after", {}).get(label)
    if after is not None:
        (top / "clock").write_text(str(BASE + after))


def current_journey():
    return ORDER[int((top / "reset.count").read_text()) - 1]


def noise(rc):
    sys.stdout.write("leak: %s %s\n" % (answers["secret"], answers["page"]))
    sys.stderr.write("leak: %s %s\n" % (answers["secret"], answers["page"]))
    sys.exit(rc)


if name == "date":
    clock = top / "clock"
    print(clock.read_text() if clock.exists() else BASE)
    sys.exit(0)
if name == "refresh_stack.sh":
    record("refresh", {k: os.environ.get(k, "") for k in ("JOURNEY_RUNNER_SSH", "JOURNEY_S2_SCRIPTS_DIR")})
    noise(answers["rc"].get("refresh", 0))
if name == "isolation_check.sh":
    label = "isolation " + args[0]
    record(label)
    noise(answers["rc"].get(label, 0))
if name == "sleep":
    record("sleep")
    sys.exit(0)
if name == "hermes":
    record("hermes")
    sys.exit(answers["hermes_rc"])
if name == "curl":
    record("local curl")
    sys.exit(97)
if name == "systemctl":
    record("systemctl")
    sys.exit(0)
if name == "systemd-run":
    record("systemd-run")
    sys.exit(0)
if name == "ssh":
    command = args[-1]
    if "isolation_check.sh server2" in command:
        label = "isolation server2"
    elif "reset_stack.sh" in command:
        counter("reset")
        label = "reset %d" % current_journey()
    elif "seed_stack.sh" in command:
        label = "seed %d" % current_journey()
    elif "journey-run.sh" in command:
        label = "journey %d" % current_journey()
    elif " curl " in command:
        label = ("proxy %d" if answers["proxy_url"] in command else "app %d") % current_journey()
    else:
        label = "ssh other"
    record(label)
    rc = answers["rc"].get(label, 0)
    if label.startswith("proxy"):
        sys.stdout.write(answers["proxy"] if rc == 0 else "")
        sys.exit(rc)
    if label.startswith("app"):
        followed = " --location " in command and " --max-redirs 3 " in command
        # the owner app root answers a 307 to /projects: the page text is only seen by a call that follows it
        redirected = answers.get("redirect") and not followed
        sys.stdout.write("Temporary Redirect: /projects" if redirected else answers["page"] if rc == 0 else "")
        sys.exit(rc)
    noise(rc)
sys.exit("stand-in: unknown name " + name)
'''

TOOLS = ("bash", "flock", "rm", "mkdir", "dirname", "mv")


def _proxy_facts() -> tuple[str, str]:
    source = PROXY_SOURCE.read_text(encoding="utf-8")
    issuer = re.search(r'^ISSUER = "([^"]+)"$', source, re.M).group(1)
    kid = re.search(r'^KID = "([^"]+)"$', source, re.M).group(1)
    return issuer, kid


ISSUER, KID = _proxy_facts()
PROXY_URL = f"{ISSUER}/cdn-cgi/access/certs"
HEALTHY_PROXY = json.dumps({"keys": [{"kty": "RSA", "kid": KID}]}, separators=(",", ":"))


def _template_containers() -> list[str]:
    names, inside = [], False
    for line in TEMPLATE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line.startswith("["):
            inside = line == "[containers]"
        elif inside and line and not line.startswith("#"):
            names.append(line)
    return names


BROWSER = _template_containers()[4]


def cmd_reset(s2: str = S2_DIR) -> str:
    return f"cd '{s2}' && ./reset_stack.sh"


def cmd_seed(n: int, s2: str = S2_DIR) -> str:
    return f"cd '{s2}' && ./seed_stack.sh {n}"


def cmd_journey(n: int, container: str = BROWSER) -> str:
    return (
        f"JOURNEY_BASE_URL='{BASE_URL}' JOURNEY_PRODUCTION_ORIGIN='{PROD}' "
        f"docker exec -w '{BROWSER_DIR}' -e JOURNEY_BASE_URL -e JOURNEY_PRODUCTION_ORIGIN "
        f"'{container}' sh tools/page-check/journey-run.sh {n}"
    )


def loop_calls(journeys=ORDER) -> list[str]:
    calls: list[str] = []
    for n in journeys:
        calls += [f"reset {n}"]
        calls += [f"seed {n}"] if n in SEEDED else []
        calls += [f"proxy {n}", f"app {n}", f"journey {n}"]
    return calls


PREFIX = ["refresh", "isolation server2", "isolation server1"]


class Sandbox:
    def __init__(self, top: Path) -> None:
        self.top = top
        self.scripts = top / "journeys"
        self.bin = top / "bin"
        self.home = top / "hermes-home"
        self.run_dir = self.home / "journeys"
        self.runner_env = top / "runner.env"
        for folder in (self.home, self.bin, self.scripts):
            folder.mkdir()
        (top / "stand_in.py").write_text(STAND_IN, encoding="utf-8")
        for name in ("ssh", "date", "curl", "hermes", "sleep", "systemd-run", "systemctl"):
            self._wrapper(self.bin / name)
        for name in TOOLS:
            (self.bin / name).symlink_to(shutil.which(name))
        for name in ("isolation_check.sh", "refresh_stack.sh"):
            self._wrapper(self.scripts / name)
        shutil.copy2(PROXY_SOURCE, self.scripts / "synthetic_proxy.py")
        self.runner_env.write_text(f"JOURNEY_BASE_URL={BASE_URL}\nJOURNEY_PRODUCTION_ORIGIN={PROD}\n", encoding="utf-8")
        self.answer()

    def _wrapper(self, path: Path) -> None:
        path.write_text(f'#!/bin/sh\nexec "{sys.executable}" -I -S "{self.top / "stand_in.py"}" "$0" "$@"\n')
        path.chmod(0o755)

    def answer(self, **changes) -> None:
        answers = {
            "rc": {}, "hermes_rc": 0, "elapsed_after": {}, "order": ORDER, "secret": SECRET, "page": GOOD_PAGE,
            "proxy": HEALTHY_PROXY, "proxy_url": PROXY_URL,
        }
        answers.update(changes)
        (self.top / "answers.json").write_text(json.dumps(answers), encoding="utf-8")

    def run(self, script: Path = RUN_ONCE, **changes: str | None) -> subprocess.CompletedProcess[str]:
        copy = self.scripts / script.name
        shutil.copy2(script, copy)
        env = {
            "PATH": str(self.bin), "HOME": str(self.top), "HERMES_HOME": str(self.home),
            "JOURNEY_PAGE_TEXT": PAGE_TEXT, "JOURNEY_RUNNER_SSH": RUNNER,
            "JOURNEY_RUNNER_ENV": str(self.runner_env),
        }
        for name, value in changes.items():
            if value is None:
                env.pop(name, None)
            else:
                env[name] = value
        return subprocess.run(
            [str(copy)], env=env, stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=60, check=False,
        )

    def calls(self) -> list[dict]:
        log = self.top / "calls.jsonl"
        return [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []

    def names(self) -> list[str]:
        return [call["call"] for call in self.calls()]

    def call(self, label: str) -> dict:
        (found,) = [call for call in self.calls() if call["call"] == label]
        return found

    def told(self) -> list[str]:
        return [call["stdin"] for call in self.calls() if call["call"] == "hermes"]

    def result(self) -> dict[str, str]:
        text = (self.run_dir / "result").read_text(encoding="utf-8")
        return dict(line.split("=", 1) for line in text.splitlines())


@pytest.fixture
def box(tmp_path: Path) -> Sandbox:
    if not all(shutil.which(tool) for tool in (*TOOLS, "sh")):
        pytest.skip("needs bash and flock")
    return Sandbox(tmp_path)


# ------------------------------------------------------------- card 4: the loop, the seed, the journey line


def test_a_passing_run_refreshes_then_runs_each_journey_on_a_fresh_stack_in_order(box):
    done = box.run()

    assert done.returncode == 0, done.stdout + done.stderr
    assert box.names() == [*PREFIX, *loop_calls(), "hermes"]
    assert [n for n in box.names() if n.startswith("reset")] == [f"reset {n}" for n in (3, 5, 2, 1, 4)]
    for n in ORDER:
        assert box.call(f"reset {n}")["argv"] == ["ssh", RUNNER, cmd_reset()]
        assert box.call(f"journey {n}")["argv"] == ["ssh", RUNNER, cmd_journey(n)]
    for n in SEEDED:
        assert box.call(f"seed {n}")["argv"] == ["ssh", RUNNER, cmd_seed(n)]
    assert not {"seed 1", "seed 4"} & set(box.names())  # journeys 1 and 4 have no seed step
    result = box.result()
    assert (result["result"], result["step"]) == ("passed", "journeys")
    assert "journey" not in result
    assert box.told() == ["Journeys passed."]
    assert (done.stdout, done.stderr) == ("", "")


def test_a_seed_failure_on_journey_5_records_the_step_and_the_journey_and_starts_nothing_later(box):
    box.answer(rc={"seed 5": 1})

    done = box.run()

    assert done.returncode != 0
    assert box.names() == [*PREFIX, *loop_calls([3]), "reset 5", "seed 5", "hermes"]
    result = box.result()
    assert (result["result"], result["step"], result["journey"], result["reason"]) == (
        "failed", "seed", "5", "the seed failed",
    )
    assert box.told() == ["Journeys failed at step seed for journey 5: the seed failed."]


@pytest.mark.parametrize(
    ("step", "journey", "label"),
    [
        ("reset", 5, "reset 5"), ("seed", 2, "seed 2"), ("proxy", 2, "proxy 2"), ("owner-app", 1, "app 1"),
        ("journeys", 4, "journey 4"), ("journeys", 3, "journey 3"),
    ],
)
def test_a_failed_step_of_a_journey_names_the_journey_and_ends_the_run(box, step, journey, label):
    box.answer(rc={label: 7})

    done = box.run()

    assert done.returncode != 0
    names = box.names()
    # The failed step is the last call before the owner line; no later journey and no retry.
    assert names[-2:] == [label, "hermes"] or (step == "owner-app" and names[-1] == "hermes")
    assert names.count(f"reset {journey}") == 1 and not any(f"reset {n}" in names for n in ORDER[ORDER.index(journey) + 1 :])
    result = box.result()
    assert (result["result"], result["step"], result["journey"], result["reason"]) == (
        "failed", step, str(journey), REASONS[step],
    )
    assert box.told() == [f"Journeys failed at step {step} for journey {journey}: {REASONS[step]}."]


def test_the_journeys_command_passes_exactly_the_one_digit_and_the_two_variable_names(box):
    assert box.run().returncode == 0

    for n in ORDER:
        command = box.call(f"journey {n}")["argv"][2]
        prefix, _, rest = command.partition(" docker exec ")
        assert re.findall(r"(\w+)='", prefix) == ["JOURNEY_BASE_URL", "JOURNEY_PRODUCTION_ORIGIN"]
        assert re.findall(r"-e (\S+)", rest) == ["JOURNEY_BASE_URL", "JOURNEY_PRODUCTION_ORIGIN"]
        assert "--env" not in rest and "--env-file" not in rest and "=" not in rest
        assert rest.endswith(f"'{BROWSER}' sh tools/page-check/journey-run.sh {n}")
        assert SECRET not in command


# ----------------------------------------------------------------------------------- the time step


def _before(n: int) -> str:
    """The last call before journey n starts."""
    return "isolation server1" if n == ORDER[0] else f"journey {ORDER[ORDER.index(n) - 1]}"


@pytest.mark.parametrize("n", ORDER)
def test_a_journey_starts_when_the_time_plus_its_reserve_is_exactly_118_minutes(box, n):
    box.answer(elapsed_after={_before(n): (LIMIT_MINUTES - RESERVES[n]) * 60})

    box.run()

    assert f"reset {n}" in box.names()


@pytest.mark.parametrize("n", ORDER)
def test_a_journey_is_refused_when_the_time_plus_its_reserve_is_one_second_over_118_minutes(box, n):
    box.answer(elapsed_after={_before(n): (LIMIT_MINUTES - RESERVES[n]) * 60 + 1})

    done = box.run()

    assert done.returncode != 0
    names = box.names()
    assert names[-2:] == [_before(n), "hermes"]
    assert not any(name.startswith(("reset", "seed", "proxy", "app")) and name.endswith(f" {n}") for name in names)
    result = box.result()
    assert (result["result"], result["step"], result["journey"], result["reason"]) == (
        "failed", "time", str(n), "the run's time ran out before the next journey",
    )
    assert box.told() == [f"Journeys failed at step time for journey {n}: the run's time ran out before the next journey."]


def test_with_the_clock_past_the_reserve_for_journey_2_the_run_ends_at_time_and_resets_no_journey_2(box):
    # Journeys 3 and 5 fit; after them 108 minutes are gone, and journey 2 needs 11 more.
    box.answer(elapsed_after={"journey 3": 100 * 60, "journey 5": 108 * 60})

    done = box.run()

    assert done.returncode != 0
    assert box.names() == [*PREFIX, *loop_calls([3, 5]), "hermes"]
    assert "reset 2" not in box.names()
    result = box.result()
    assert (result["step"], result["journey"]) == ("time", "2")
    assert box.told() == [f"Journeys failed at step time for journey 2: {REASONS['time']}."]


def test_the_time_of_the_refresh_counts_towards_the_limit(box):
    box.answer(elapsed_after={"refresh": 110 * 60})  # journey 3 needs 9 more minutes: 119 are over 118

    box.run()

    assert box.names() == [*PREFIX, "hermes"]
    assert (box.result()["step"], box.result()["journey"]) == ("time", "3")


# ------------------------------------------------------------------------------- the refresh step


def test_the_refresh_runs_after_the_config_and_before_the_isolation_with_the_two_names(box):
    assert box.run(JOURNEY_S2_SCRIPTS_DIR="/srv/s2").returncode == 0

    refresh = box.call("refresh")
    assert box.names()[0] == "refresh" and box.names()[1:3] == ["isolation server2", "isolation server1"]
    assert refresh["argv"] == ["refresh_stack.sh"]  # no argument
    assert refresh["env"] == {"JOURNEY_RUNNER_SSH": RUNNER, "JOURNEY_S2_SCRIPTS_DIR": "/srv/s2"}


def test_the_refresh_gets_the_default_scripts_folder_when_none_is_set(box):
    assert box.run().returncode == 0

    assert box.call("refresh")["env"] == {"JOURNEY_RUNNER_SSH": RUNNER, "JOURNEY_S2_SCRIPTS_DIR": S2_DIR}


def test_a_config_failure_ends_the_run_before_the_refresh(box):
    done = box.run(JOURNEY_PAGE_TEXT=None)

    assert done.returncode != 0
    assert box.names() == ["hermes"]
    assert box.told() == ["Journeys failed at step config: a required setting is missing or is not plain."]


def test_a_refresh_failure_records_the_step_starts_no_isolation_or_reset_and_sends_one_owner_line(box):
    box.answer(rc={"refresh": 1})

    done = box.run()

    assert done.returncode != 0
    assert box.names() == ["refresh", "hermes"]
    result = box.result()
    assert (result["result"], result["step"], result["reason"]) == ("failed", "refresh", REASONS["refresh"])
    assert "journey" not in result
    assert box.told() == [f"Journeys failed at step refresh: {REASONS['refresh']}."]
    assert SECRET not in done.stdout + done.stderr + "".join(box.told())


# ------------------------------------------------------------------ the proxy and owner-app checks


def test_the_proxy_and_owner_app_checks_run_inside_the_browser_container_through_the_runner(box):
    assert box.run().returncode == 0

    assert "local curl" not in box.names()  # nothing runs curl on Server 1
    for n in ORDER:
        proxy, app = box.call(f"proxy {n}")["argv"], box.call(f"app {n}")["argv"]
        assert proxy[:2] == ["ssh", RUNNER] and app[:2] == ["ssh", RUNNER]
        assert proxy[2].startswith(f"docker exec '{BROWSER}' curl ") and proxy[2].endswith(f" '{PROXY_URL}'")
        assert app[2].startswith(f"docker exec '{BROWSER}' curl ") and app[2].endswith(f" '{BASE_URL}'")
        assert "--fail" in proxy[2] and "--fail" in app[2]


def test_the_owner_app_check_follows_the_redirect_of_the_root_and_the_proxy_check_does_not(box):
    box.answer(redirect=True)  # the root answers a redirect first; the page text comes only after --location

    done = box.run()

    assert done.returncode == 0, done.stdout + done.stderr
    for n in ORDER:
        app, proxy = box.call(f"app {n}")["argv"][2], box.call(f"proxy {n}")["argv"][2]
        assert " --location --max-redirs 3 " in app
        assert "--location" not in proxy and "--max-redirs" not in proxy
    assert box.told() == ["Journeys passed."]


def test_the_answers_are_checked_on_server_1(box):
    # A proxy that answers without its key id, and an owner app page without its text, both fail here.
    box.answer(proxy='{"keys":[]}')
    assert box.run().returncode != 0
    assert box.result()["step"] == "proxy"

    (box.top / "calls.jsonl").unlink()
    (box.top / "reset.count").unlink()
    box.answer(page="<html>Something went wrong</html>")
    assert box.run().returncode != 0
    assert box.result()["step"] == "owner-app"
    assert box.names().count("app 3") == 61 and box.names().count("sleep") == 60
    assert "local curl" not in box.names()


def test_the_browser_container_defaults_to_the_fifth_container_name_of_the_template(box):
    assert BROWSER == "raphael-journeys-browser"
    assert box.run().returncode == 0
    for n in ORDER:
        assert f"'{BROWSER}'" in box.call(f"journey {n}")["argv"][2]
        assert f"docker exec '{BROWSER}' curl " in box.call(f"app {n}")["argv"][2]


def test_the_settings_named_by_the_owner_change_the_container_and_the_scripts_folder(box):
    assert box.run(JOURNEY_BROWSER_CONTAINER="browser-x", JOURNEY_S2_SCRIPTS_DIR="/srv/s2").returncode == 0
    assert box.call("journey 3")["argv"][2] == cmd_journey(3, "browser-x")
    assert box.call("reset 3")["argv"][2] == cmd_reset("/srv/s2")
    assert box.call("seed 3")["argv"][2] == cmd_seed(3, "/srv/s2")
    assert box.call("app 3")["argv"][2].startswith("docker exec 'browser-x' curl ")


# --------------------------------------------------------------------------------------- the limits


def _constant(path: Path, pattern: str) -> re.Match:
    found = re.search(pattern, path.read_text(encoding="utf-8"), re.M)
    assert found, pattern
    return found


def test_start_asks_for_the_run_limit_of_7200_seconds(box):
    assert box.run(START).returncode == 0

    options = box.call("systemd-run")["argv"]
    assert "--property=RuntimeMaxSec=7200" in options


def test_the_stuck_point_of_the_hourly_check_is_the_run_limit_plus_600_seconds():
    limit = int(_constant(START, r"^RUN_MAX_SECONDS=(\d+)$").group(1))
    unit, amount = _constant(WATCH, r"^JOURNEYS_STUCK_AFTER = timedelta\((minutes|seconds|hours)=(\d+)\)$").groups()
    stuck = int(amount) * {"seconds": 1, "minutes": 60, "hours": 3600}[unit]

    assert limit == 7200
    assert stuck == limit + 600
