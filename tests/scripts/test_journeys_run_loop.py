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
# The last journey (4) also reserves the final reset that follows it.
RESERVES = {3: 4 + 1 + 2 + 2, 5: 4 + 1 + 2 + 6, 2: 4 + 1 + 2 + 4, 1: 4 + 2 + 21, 4: 4 + 2 + 16 + 4}
LIMIT_MINUTES = 118

REASONS = {
    "refresh": "the stack could not be brought to the released versions",
    "isolation": "the isolation check failed",
    "time": "the run's time ran out before the next journey",
    "reset": "the stack reset failed",
    "seed": "the seed failed",
    "proxy": "the sign-in proxy did not answer",
    "owner-app": "the owner app did not show its page text in time",
    "journeys": "the journey runner failed",
}

STAND_IN = r'''
import json, os, sys, time
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
    global last
    last = label
    entry = {"call": label, "argv": [name, *args], "stdin": stdin, "env": extra or {}}
    with open(log, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry) + "\n")
    after = answers.get("elapsed_after", {}).get(label)
    if after is not None:
        (top / "clock").write_text(str(BASE + after))
    if answers.get("break_held") == label:
        for held_dir in Path(os.environ["TMPDIR"]).glob("run_once.*"):
            (held_dir / "held").unlink(missing_ok=True)
            (held_dir / "held").mkdir()
    folder = answers.get("make_folder", {}).get(label)
    if folder is not None:
        Path(folder).mkdir(parents=True)
    link = answers.get("make_link", {}).get(label)
    if link is not None:
        Path(link[0]).symlink_to(link[1])
    target = answers.get("link_held", {}).get(label)
    if target is not None:
        for held_dir in Path(os.environ["TMPDIR"]).glob("run_once.*"):
            (held_dir / "held").unlink(missing_ok=True)
            (held_dir / "held").symlink_to(target)


def current_journey():
    # reset 1 is the run's first reset, before any journey; journey k follows reset k + 1
    return ORDER[int((top / "reset.count").read_text()) - 2]


def noise(rc):
    if last in answers.get("exact", {}):
        sys.stdout.write(answers["exact"][last])
        sys.stdout.flush()
        sys.exit(rc)
    if last in answers.get("bulk", {}):
        # exactly this many bytes, nothing else, in many short lines of 19 bytes ("row 000000", blanks, a line feed)
        rows = answers["bulk"][last] // 19
        for first in range(0, rows, 10_000):
            chunk = "".join("%-18s\n" % ("row %06d" % i) for i in range(first, min(first + 10_000, rows)))
            os.write(1, chunk.encode())
        sys.exit(rc)
    if last in answers.get("output", {}):
        sys.stdout.write(answers["output"][last] + "\n")
        sys.stdout.flush()
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
    kept = Path(os.environ["HERMES_HOME"]) / "journeys" / "failed-step.log"
    record("hermes", {"kept": str(kept.exists())})
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
        count = counter("reset")
        label = "reset first" if count == 1 else "reset %d" % ORDER[count - 2] if count - 2 < len(ORDER) else "reset last"
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
    errors = answers.get("stderr", {}).get(label, "")
    if label.startswith("proxy"):
        answer = answers["proxy"] if rc == 0 else ""
    elif label.startswith("app"):
        followed = " --location " in command and " --max-redirs 3 " in command
        # the owner app root answers a 307 to /projects: the page text is only seen by a call that follows it
        redirected = answers.get("redirect") and not followed
        answer = "Temporary Redirect: /projects" if redirected else answers["page"] if rc == 0 else ""
    else:
        sys.stderr.write(errors)
        sys.stderr.flush()
        noise(rc)
    if label in answers.get("stdout_first", []):
        sys.stdout.write(answer)
        sys.stdout.flush()
        time.sleep(0.2)  # the errors come well after the answer
        sys.stderr.write(errors)
        sys.stderr.flush()
    else:
        sys.stderr.write(errors)
        sys.stderr.flush()
        sys.stdout.write(answer)
        sys.stdout.flush()
    sys.exit(rc)
sys.exit("stand-in: unknown name " + name)
'''

TOOLS = ("bash", "flock", "rm", "mkdir", "dirname", "mv", "tail", "cut")
# Stands in for mv: it logs, for each rename, whether the failed-step file was already there, then renames.
MV_WRAPPER = """#!/bin/sh
if [ -e "$HERMES_HOME/journeys/failed-step.log" ]; then echo present >> "{log}"; else echo absent >> "{log}"; fi
exec "{real}" "$@"
"""


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


PREFIX = ["refresh", "reset first", "isolation server2", "isolation server1"]


class Sandbox:
    def __init__(self, top: Path) -> None:
        self.top = top
        self.scripts = top / "journeys"
        self.bin = top / "bin"
        self.home = top / "hermes-home"
        self.run_dir = self.home / "journeys"
        self.runner_env = top / "runner.env"
        self.tmp = top / "tmp"
        for folder in (self.home, self.bin, self.scripts, self.tmp):
            folder.mkdir()
        (top / "stand_in.py").write_text(STAND_IN, encoding="utf-8")
        for name in ("ssh", "date", "curl", "hermes", "sleep", "systemd-run", "systemctl"):
            self._wrapper(self.bin / name)
        for name in TOOLS:
            (self.bin / name).symlink_to(shutil.which(name))
        (self.bin / "mv").unlink()
        (self.bin / "mv").write_text(MV_WRAPPER.format(log=top / "mv.log", real=shutil.which("mv")), encoding="utf-8")
        (self.bin / "mv").chmod(0o755)
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

    def run(
        self, script: Path = RUN_ONCE, memory_kib: int | None = None, **changes: str | None,
    ) -> subprocess.CompletedProcess[str]:
        copy = self.scripts / script.name
        shutil.copy2(script, copy)
        env = {
            "PATH": str(self.bin), "HOME": str(self.top), "HERMES_HOME": str(self.home),
            "JOURNEY_PAGE_TEXT": PAGE_TEXT, "JOURNEY_RUNNER_SSH": RUNNER,
            "JOURNEY_RUNNER_ENV": str(self.runner_env), "TMPDIR": str(self.tmp),
        }
        for name, value in changes.items():
            if value is None:
                env.pop(name, None)
            else:
                env[name] = value
        command = [str(copy)]
        if memory_kib is not None:  # the memory limit of the service: `ulimit -v`, in KiB, for the whole run
            command = [shutil.which("bash"), "-c", f'ulimit -v {memory_kib} && exec "$0"', str(copy)]
        return subprocess.run(
            command, env=env, stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=60, check=False,
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

    def renames(self) -> list[str]:
        log = self.top / "mv.log"
        return log.read_text().split() if log.exists() else []

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
    assert box.names() == [*PREFIX, *loop_calls(), "reset last", "hermes"]
    assert [n for n in box.names() if n.startswith("reset")] == [
        "reset first", *[f"reset {n}" for n in (3, 5, 2, 1, 4)], "reset last",
    ]
    for label in ("reset first", "reset last"):
        assert box.call(label)["argv"] == ["ssh", RUNNER, cmd_reset()]
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


def test_a_failing_first_reset_stops_the_run_before_the_isolation_check_at_step_reset_with_no_journey(box):
    box.answer(rc={"reset first": 7})

    done = box.run()

    assert done.returncode != 0
    assert box.names() == ["refresh", "reset first", "hermes"]
    result = box.result()
    assert (result["result"], result["step"], result["reason"]) == ("failed", "reset", REASONS["reset"])
    assert "journey" not in result
    assert box.told() == [f"Journeys failed at step reset: {REASONS['reset']}."]


def test_a_failing_last_reset_fails_the_run_at_step_reset_and_stores_no_passed_result(box):
    box.answer(rc={"reset last": 7})

    done = box.run()

    assert done.returncode != 0
    assert box.names() == [*PREFIX, *loop_calls(), "reset last", "hermes"]
    result = box.result()
    assert (result["result"], result["step"], result["reason"]) == ("failed", "reset", REASONS["reset"])
    assert "journey" not in result
    assert box.told() == [f"Journeys failed at step reset: {REASONS['reset']}."]


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


def test_the_time_of_the_first_reset_counts_towards_the_limit(box):
    box.answer(elapsed_after={"reset first": 110 * 60})

    box.run()

    assert box.names() == [*PREFIX, "hermes"]
    assert (box.result()["step"], box.result()["journey"]) == ("time", "3")


# ------------------------------------------------------------------------------- the refresh step


def test_the_refresh_runs_after_the_config_and_before_the_isolation_with_the_two_names(box):
    assert box.run(JOURNEY_S2_SCRIPTS_DIR="/srv/s2").returncode == 0

    refresh = box.call("refresh")
    assert box.names()[0] == "refresh" and box.names()[1:4] == ["reset first", "isolation server2", "isolation server1"]
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


# ------------------------------------------------------------------- the last lines of a failed step

KEPT = "failed-step.log"
LEAK = f"leak: {SECRET} {GOOD_PAGE}"  # what every stand-in command writes to its output and to its errors


def _kept(box: Sandbox) -> list[str]:
    return (box.run_dir / KEPT).read_text(encoding="utf-8").splitlines()


def _fail_reset_of_3(box: Sandbox, **answers) -> subprocess.CompletedProcess[str]:
    box.answer(rc={"reset 3": 7}, **answers)
    return box.run()


def _rows(first: int, last: int) -> str:
    return "\n".join(f"row {i}" for i in range(first, last + 1))


def test_a_failed_reset_of_a_journey_keeps_the_step_the_journey_and_the_last_lines_of_that_command(box):
    done = _fail_reset_of_3(box, output={"reset 3": "first line\nsecond line"})

    assert done.returncode == 1
    lines = _kept(box)
    assert lines[0] == "step=reset" and lines[1] == "journey=3"
    assert lines[2] == "started=" + box.result()["started"]
    assert re.fullmatch(r"finished=\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ", lines[3])
    assert lines[3] == "finished=" + box.result()["finished"]
    assert lines[4:] == ["first line", "second line", LEAK, LEAK]  # its errors are kept with its output


def test_a_failed_step_without_a_journey_has_no_journey_line(box):
    box.answer(rc={"reset first": 7}, output={"reset first": "boom"})

    assert box.run().returncode == 1

    lines = _kept(box)
    assert [line.split("=")[0] for line in lines[:3]] == ["step", "started", "finished"] and lines[0] == "step=reset"
    assert lines[3:] == ["boom", LEAK, LEAK]


@pytest.mark.parametrize(
    ("step", "label"),
    [("refresh", "refresh"), ("isolation", "isolation server2"), ("isolation", "isolation server1"),
     ("seed", "seed 3"), ("journeys", "journey 3")],
)
def test_each_failed_command_step_keeps_the_output_of_its_newest_command(box, step, label):
    box.answer(rc={label: 7}, output={label: "NEWEST-" + label})

    assert box.run().returncode == 1

    lines = _kept(box)
    assert lines[0] == "step=" + step
    assert ("journey=3" in lines[:2]) == (step in {"seed", "journeys"})
    assert lines[-3:] == ["NEWEST-" + label, LEAK, LEAK]


def test_a_failed_owner_app_step_keeps_the_errors_then_the_last_page_but_decides_from_the_page_only(box):
    box.answer(page="<html>Something went wrong</html>", stderr={"app 3": "curl-error-line\n"})

    assert box.run().returncode == 1

    lines = _kept(box)
    assert lines[0] == "step=owner-app" and lines[1] == "journey=3"
    assert lines[4:] == ["curl-error-line", "<html>Something went wrong</html>"]


def test_the_errors_of_the_proxy_check_are_kept_before_the_answer_but_never_decide_it(box):
    box.answer(proxy='{"keys":[]}', stderr={"proxy 3": f'"kid":"{KID}"\n'})

    assert box.run().returncode == 1

    assert box.result()["step"] == "proxy"
    assert _kept(box)[4:] == [f'"kid":"{KID}"', '{"keys":[]}']


def test_a_proxy_step_that_ran_no_command_keeps_none_of_an_earlier_step_output(box):
    (box.scripts / "synthetic_proxy.py").write_text("", encoding="utf-8")  # no issuer and no key id to look for
    box.answer(output={"reset 3": "OUTPUT-OF-THE-RESET"})

    assert box.run().returncode == 1

    assert box.result()["step"] == "proxy"
    assert _kept(box) == [
        "step=proxy", "journey=3", "started=" + box.result()["started"], "finished=" + box.result()["finished"],
    ]


def test_an_output_of_more_than_40_lines_keeps_the_last_40(box):
    _fail_reset_of_3(box, output={"reset 3": _rows(1, 100)})

    body = _kept(box)[4:]
    assert body == [*[f"row {i}" for i in range(63, 101)], LEAK, LEAK]
    assert len(body) == 40


def test_an_output_of_exactly_40_lines_is_kept_in_full_and_one_more_line_drops_the_first(box):
    _fail_reset_of_3(box, output={"reset 3": _rows(1, 38)})
    assert _kept(box)[4:] == [*[f"row {i}" for i in range(1, 39)], LEAK, LEAK]

    (box.top / "reset.count").unlink()
    _fail_reset_of_3(box, output={"reset 3": _rows(1, 39)})
    assert _kept(box)[4:] == [*[f"row {i}" for i in range(2, 40)], LEAK, LEAK]


def test_a_line_of_more_than_400_characters_is_cut_at_400(box):
    _fail_reset_of_3(box, output={"reset 3": "a" * 399 + "\n" + "b" * 400 + "\n" + "c" * 401 + "\n" + "d" * 1000})

    body = _kept(box)[4:]
    assert body[:4] == ["a" * 399, "b" * 400, "c" * 400, "d" * 400]
    assert max(len(line) for line in body) == 400


def test_the_file_is_private_mode_600(box):
    _fail_reset_of_3(box, output={"reset 3": "x"})

    assert (box.run_dir / KEPT).stat().st_mode & 0o777 == 0o600


def test_a_passed_run_leaves_no_file(box):
    done = box.run()

    assert done.returncode == 0
    assert sorted(path.name for path in box.run_dir.iterdir()) == ["result", "run.lock"]


def test_a_file_of_an_earlier_failed_run_is_gone_after_a_passed_run(box):
    _fail_reset_of_3(box, output={"reset 3": "x"})
    assert (box.run_dir / KEPT).exists()
    box.answer()
    (box.top / "reset.count").unlink()

    assert box.run().returncode == 0

    assert not (box.run_dir / KEPT).exists()


def test_the_file_of_an_earlier_run_is_replaced_by_a_header_only_file_when_the_config_step_fails(box):
    _fail_reset_of_3(box, output={"reset 3": "EARLIER-RUN-LINE"})
    box.answer()

    assert box.run(JOURNEY_PAGE_TEXT=None).returncode == 1

    assert box.result()["step"] == "config"
    assert _kept(box) == ["step=config", "started=" + box.result()["started"], "finished=" + box.result()["finished"]]


def test_the_next_failed_run_replaces_the_file(box):
    _fail_reset_of_3(box, output={"reset 3": "FIRST-RUN-LINE"})
    (box.top / "reset.count").unlink()
    box.answer(rc={"seed 5": 7}, output={"seed 5": "SECOND-RUN-LINE"})

    assert box.run().returncode == 1

    lines = _kept(box)
    assert lines[:2] == ["step=seed", "journey=5"]
    assert "SECOND-RUN-LINE" in lines and "FIRST-RUN-LINE" not in lines
    assert sorted(path.name for path in box.run_dir.iterdir()) == [KEPT, "result", "run.lock"]


def test_a_step_that_ran_no_command_keeps_a_file_of_its_header_lines_only(box):
    box.answer(elapsed_after={"reset first": 118 * 60}, output={"reset first": "OUTPUT-OF-THE-RESET"})

    assert box.run().returncode == 1

    assert box.result()["step"] == "time"
    assert _kept(box) == [
        "step=time", "journey=3", "started=" + box.result()["started"], "finished=" + box.result()["finished"],
    ]


def test_the_file_is_written_after_the_result_and_before_the_owner_line_in_place_with_no_rename(box):
    _fail_reset_of_3(box, output={"reset 3": "x"})

    assert box.renames() == ["absent", "absent"]  # the started result and the failed result: no third rename
    assert box.call("hermes")["env"] == {"kept": "True"}  # the owner line follows the file
    assert box.result()["result"] == "failed"
    assert sorted(path.name for path in box.run_dir.iterdir()) == [KEPT, "result", "run.lock"]


def test_the_renames_of_a_failed_run_are_the_same_as_those_of_a_run_without_the_file(box):
    box.answer(rc={"reset last": 7})

    assert box.run().returncode == 1

    assert box.renames() == ["absent", "absent"]
    assert box.result()["step"] == "reset" and "journey" not in box.result()


def test_the_result_file_and_the_owner_line_of_a_failed_run_are_as_without_the_file(box):
    secret_lines = "\n".join(f"SENTINEL-{i}" for i in range(5))

    done = _fail_reset_of_3(box, output={"reset 3": secret_lines})

    assert done.returncode == 1
    text = (box.run_dir / "result").read_text(encoding="utf-8")
    assert re.fullmatch(
        r"result=failed\nstep=reset\nreason=the stack reset failed\nstarted=\S+Z\njourney=3\nfinished=\S+Z\n", text,
    )
    assert box.told() == [f"Journeys failed at step reset for journey 3: {REASONS['reset']}."]
    everything = text + "".join(box.told())
    assert "SENTINEL" not in everything and SECRET not in everything and GOOD_PAGE not in everything
    assert (done.stdout, done.stderr) == ("", "")


def test_a_run_folder_in_which_the_file_cannot_be_written_ends_the_run_with_the_result_of_today(box):
    done = _fail_reset_of_3(box, output={"reset 3": "x"}, make_folder={"reset 3": str(box.run_dir / KEPT)})

    assert done.returncode == 1
    assert box.names()[-2:] == ["reset 3", "hermes"]
    assert box.told() == [f"Journeys failed at step reset for journey 3: {REASONS['reset']}."]
    result = box.result()
    assert (result["result"], result["step"], result["journey"], result["reason"]) == (
        "failed", "reset", "3", REASONS["reset"],
    )
    assert (box.run_dir / KEPT).is_dir()
    assert len(done.stderr.splitlines()) == 1
    assert SECRET not in done.stderr and GOOD_PAGE not in done.stderr and done.stdout == ""


def test_a_file_that_cannot_be_removed_at_the_start_does_not_change_the_result_of_a_passing_run(box):
    box.run_dir.mkdir()
    (box.run_dir / KEPT).mkdir()
    (box.run_dir / KEPT / "inner").write_text("x")  # rm -f cannot remove a folder that has something in it

    done = box.run()

    assert done.returncode == 0
    assert box.result()["result"] == "passed" and box.told() == ["Journeys passed."]
    assert len(done.stderr.splitlines()) == 1


# ------------------------------------------------------------------------ the review round: the held output


def test_the_newest_stdout_line_after_an_early_stderr_burst_stays_in_the_kept_tail_of_the_proxy_check(box):
    burst = "".join(f"early error {i}\n" for i in range(1, 41))
    box.answer(proxy='{"keys":[]}', stderr={"proxy 3": burst})

    assert box.run().returncode == 1

    body = _kept(box)[4:]
    assert body == [*[f"early error {i}" for i in range(2, 41)], '{"keys":[]}']
    assert box.result()["step"] == "proxy"
    assert "curl --disable --silent --fail --max-time 20" in box.call("proxy 3")["argv"][-1]


def test_the_newest_stdout_line_after_an_early_stderr_burst_stays_in_the_kept_tail_of_the_owner_app_check(box):
    burst = "".join(f"early error {i}\n" for i in range(1, 41))
    box.answer(page="<html>Something went wrong</html>", stderr={"app 3": burst})

    assert box.run().returncode == 1

    assert _kept(box)[4:] == [*[f"early error {i}" for i in range(2, 41)], "<html>Something went wrong</html>"]


def test_a_failed_exit_status_of_a_check_still_fails_the_step_when_the_answer_looks_right(box):
    box.answer(rc={"proxy 3": 22})

    assert box.run().returncode == 1

    assert box.result()["step"] == "proxy"
    assert _kept(box)[:2] == ["step=proxy", "journey=3"]


def test_trailing_empty_lines_of_a_failed_command_stay_in_the_kept_tail(box):
    box.answer(rc={"refresh": 7}, exact={"refresh": "first\nlast\n\n\n"})

    assert box.run().returncode == 1

    assert _kept(box)[3:] == ["first", "last", "", ""]


def test_a_failed_command_that_prints_a_null_byte_keeps_the_byte_and_makes_no_diagnostic_on_the_runner_errors(box):
    box.answer(rc={"refresh": 7}, exact={"refresh": "before\x00after\n"})

    done = box.run()

    assert (done.returncode, done.stdout, done.stderr) == (1, "", "")
    assert _stored_body(box) == b"before\x00after\n"


def test_trailing_empty_lines_of_a_failed_remote_command_stay_in_the_kept_tail(box):
    box.answer(rc={"reset first": 7}, exact={"reset first": "one\n\n"})

    assert box.run().returncode == 1

    assert _kept(box)[3:] == ["one", ""]


def test_the_cut_at_400_counts_characters_and_never_leaves_half_of_one(box):
    box.answer(rc={"refresh": 7}, exact={"refresh": "a" * 399 + "\u00e9X\n" + "\u00e9" * 401 + "\n"})

    assert box.run().returncode == 1

    body = _kept(box)[3:]
    assert body == ["a" * 399 + "\u00e9", "\u00e9" * 400]


def test_the_locale_of_the_checks_stays_byte_wise_when_a_line_is_cut(box):
    box.answer(page="<html>Something went wrong \u00e9</html>")

    assert box.run().returncode == 1

    assert box.result()["step"] == "owner-app"
    assert _kept(box)[4:] == ["<html>Something went wrong \u00e9</html>"]


TWO_CLOCK_MV = """#!/bin/sh
"{real}" "$@" || exit $?
for last; do :; done
read -r first < "$last"
case $first in result=failed) "{python}" -c "import time; time.sleep(1.2)" ;; esac
"""

SECOND_RENAME_REFUSED_MV = """#!/bin/sh
n=0
if [ -f "{count}" ]; then read -r n < "{count}"; fi
n=$((n + 1))
echo "$n" > "{count}"
if [ "$n" -eq 2 ]; then exit 1; fi
exec "{real}" "$@"
"""


def test_the_file_and_the_result_hold_the_same_finished_time_when_the_clock_moves_between_the_two_writes(box):
    mv = box.bin / "mv"
    mv.unlink()
    mv.write_text(TWO_CLOCK_MV.format(real=shutil.which("mv"), python=sys.executable), encoding="utf-8")
    mv.chmod(0o755)

    _fail_reset_of_3(box, output={"reset 3": "x"})

    assert _kept(box)[3] == "finished=" + box.result()["finished"]


def test_the_file_is_not_written_when_the_failed_result_could_not_be_stored(box):
    mv = box.bin / "mv"
    mv.unlink()
    mv.write_text(
        SECOND_RENAME_REFUSED_MV.format(real=shutil.which("mv"), count=box.top / "mv.count"), encoding="utf-8",
    )
    mv.chmod(0o755)

    done = _fail_reset_of_3(box, output={"reset 3": "x"})

    assert done.returncode == 1
    assert box.result()["result"] == "started"
    assert not (box.run_dir / KEPT).exists()
    assert box.told() == [f"Journeys failed at step reset for journey 3: {REASONS['reset']}. The result file could not be stored."]
    assert done.stderr.splitlines() == ["run_once: the result file could not be stored"]


def test_a_result_whose_writes_all_fail_for_lack_of_room_leaves_the_started_record_as_a_plain_file(box):
    done = _fail_reset_of_3(box, make_link={"reset 3": [str(box.run_dir / "result.new"), "/dev/full"]})

    result = box.run_dir / "result"
    assert result.is_file() and not result.is_symlink()
    lines = result.read_text(encoding="utf-8").splitlines()
    assert lines[:3] == ["result=started", "step=none", "reason=the run has started and has not ended"]
    assert len(lines) == 4 and re.fullmatch(r"started=\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ", lines[3])
    assert not (box.run_dir / "result.new").exists() and not (box.run_dir / "result.new").is_symlink()
    assert box.told() == [f"Journeys failed at step reset for journey 3: {REASONS['reset']}. The result file could not be stored."]
    assert done.returncode == 1


def test_a_write_failure_after_a_successful_open_is_reported_even_when_the_body_is_empty(box):
    box.answer(
        elapsed_after={"reset first": 118 * 60}, make_link={"reset first": [str(box.run_dir / KEPT), "/dev/full"]},
    )

    done = box.run()

    assert done.returncode == 1
    assert box.result()["step"] == "time"
    assert done.stderr.splitlines() == ["run_once: the output of the failed step could not be kept"]
    assert box.told() == [f"Journeys failed at step time for journey 3: {REASONS['time']}."]


def test_a_write_failure_after_a_successful_open_is_reported_for_a_body_too(box):
    done = _fail_reset_of_3(
        box, output={"reset 3": "x"}, make_link={"reset 3": [str(box.run_dir / KEPT), "/dev/full"]},
    )

    assert done.returncode == 1
    assert box.result()["step"] == "reset"
    assert done.stderr.splitlines() == ["run_once: the output of the failed step could not be kept"]


# ------------------------------------------------------------------ round 2: order, parity, result step, NUL


def _stored_body(box: Sandbox) -> bytes:
    return (box.run_dir / KEPT).read_bytes().split(b"\n", 3)[3]  # the three header lines of a step with no journey


@pytest.mark.parametrize("label", ["proxy 3", "app 3"])
def test_an_early_stderr_fragment_before_41_stdout_rows_is_the_oldest_of_the_combined_output(box, label):
    rows = "".join(f"row {i}\n" for i in range(1, 42))
    answers = {"proxy": rows} if label.startswith("proxy") else {"page": rows}
    box.answer(stderr={label: "early fragment"}, **answers)

    assert box.run().returncode == 1

    assert _kept(box)[4:] == [f"row {i}" for i in range(2, 42)]  # row 1 carries the fragment and drops out
    assert box.result()["step"] == label.split()[0].replace("app", "owner-app")
    assert list(box.tmp.iterdir()) == []


@pytest.mark.parametrize("label", ["proxy 3", "app 3"])
def test_a_stderr_fragment_of_a_check_comes_before_the_answer_even_when_the_answer_was_written_first(box, label):
    rows = "".join(f"row {i}\n" for i in range(1, 42))
    answers = {"proxy": rows} if label.startswith("proxy") else {"page": rows}
    box.answer(stderr={label: "late fragment"}, stdout_first=[label], **answers)

    assert box.run().returncode == 1

    assert _kept(box)[4:] == [f"row {i}" for i in range(2, 42)]  # the fragment joins row 1, which drops out


@pytest.mark.parametrize("label", ["proxy 3", "app 3"])
@pytest.mark.parametrize("stdout_first", [False, True])
def test_the_log_of_a_check_holds_the_errors_first_and_then_the_answer_whichever_stream_wrote_first(
    box, label, stdout_first,
):
    answers = {"proxy": "out one\nout part"} if label.startswith("proxy") else {"page": "out one\nout part"}
    box.answer(stderr={label: "err one\nerr two\n"}, stdout_first=[label] if stdout_first else [], **answers)

    assert box.run().returncode == 1

    assert _kept(box)[4:] == ["err one", "err two", "out one", "out part"]


@pytest.mark.parametrize("label", ["proxy 3", "app 3"])
def test_partial_lines_of_the_errors_and_the_answer_of_a_check_join_with_the_errors_first(box, label):
    answers = {"proxy": "out one\nout part"} if label.startswith("proxy") else {"page": "out one\nout part"}
    box.answer(stderr={label: "err part|err end"}, stdout_first=[label], **answers)

    assert box.run().returncode == 1

    assert _kept(box)[4:] == ["err part|err endout one", "out part"]


@pytest.mark.parametrize("label", ["proxy 3", "app 3"])
def test_a_null_byte_in_the_answer_of_a_check_is_not_kept_and_makes_no_diagnostic(box, label):
    answers = {"proxy": '{"keys":[]}\x00X\n'} if label.startswith("proxy") else {"page": "bad\x00page\n"}
    box.answer(**answers)

    done = box.run()

    assert (done.returncode, done.stdout, done.stderr) == (1, "", "")
    assert b"\x00" not in (box.run_dir / KEPT).read_bytes()
    assert _kept(box)[4:] == (['{"keys":[]}X'] if label.startswith("proxy") else ["badpage"])


def test_a_null_byte_in_the_page_is_dropped_before_the_decision_as_before(box):
    box.answer(page="<html>Owner\x00 workspace</html>")

    assert box.run().returncode == 0


def test_the_stderr_of_the_proxy_check_is_still_never_part_of_the_decision_when_it_comes_after_the_answer(box):
    box.answer(proxy='{"keys":[]}', stderr={"proxy 3": f'"kid":"{KID}"'}, stdout_first=["proxy 3"])

    assert box.run().returncode == 1

    assert box.result()["step"] == "proxy"


PARITY = [
    ("READY\n", "READY", "owner-app"),
    ("READY\n", "READY\n", "owner-app"),
    ("READY\n", "READY\n\n\n", "owner-app"),
    ("READY\n", "READY\nmore", "passed"),
    ("READY", "READY\n", "passed"),
    ("READY", "READY\n\n\n", "passed"),
]


@pytest.mark.parametrize(("phrase", "page", "end"), PARITY)
def test_the_owner_app_decision_strips_all_trailing_newlines_of_the_answer_as_before(box, phrase, page, end):
    box.answer(page=page)

    done = box.run(JOURNEY_PAGE_TEXT=phrase)

    assert box.result()["step" if end != "passed" else "result"] == end
    assert done.returncode == (0 if end == "passed" else 1)


def test_the_kept_page_is_the_answer_as_the_check_read_it_without_its_trailing_newlines(box):
    box.answer(page="page\n\n")

    assert box.run(JOURNEY_PAGE_TEXT="absent").returncode == 1

    assert _kept(box)[4:] == ["page"]  # a command substitution strips the trailing line feeds; the log holds what was read


FIRST_RENAME_REFUSED_MV = """#!/bin/sh
n=0
if [ -f "{count}" ]; then read -r n < "{count}"; fi
n=$((n + 1))
echo "$n" > "{count}"
if [ "$n" -eq 1 ]; then exit 1; fi
exec "{real}" "$@"
"""


def test_a_failed_first_result_write_with_a_stored_terminal_result_keeps_a_header_only_file(box):
    mv = box.bin / "mv"
    mv.unlink()
    mv.write_text(FIRST_RENAME_REFUSED_MV.format(real=shutil.which("mv"), count=box.top / "mv.count"), encoding="utf-8")
    mv.chmod(0o755)

    done = box.run()

    assert done.returncode == 1
    result = box.result()
    assert (result["result"], result["step"], result["reason"]) == ("failed", "result", "the result file could not be stored")
    assert box.names() == ["hermes"] and box.call("hermes")["env"] == {"kept": "True"}
    assert box.told() == [f"Journeys failed at step result: the result file could not be stored."]
    assert _kept(box) == ["step=result", "started=" + result["started"], "finished=" + result["finished"]]
    assert (box.run_dir / KEPT).stat().st_mode & 0o777 == 0o600
    assert (done.stdout, done.stderr) == ("", "")


def test_a_null_byte_in_a_remote_command_is_kept_and_counts_as_one_character_of_the_cut(box):
    line = "a" * 397 + "\x00\u00e9" * 3 + "\n"
    box.answer(rc={"reset first": 7}, exact={"reset first": "x\x00y\n" + line})

    done = box.run()

    assert (done.returncode, done.stdout, done.stderr) == (1, "", "")
    assert _stored_body(box) == b"x\x00y\n" + ("a" * 397 + "\x00\u00e9\x00").encode() + b"\n"


def test_a_null_byte_as_the_last_byte_of_the_output_is_kept(box):
    box.answer(rc={"refresh": 7}, exact={"refresh": "one\ntwo\x00"})

    assert box.run().returncode == 1

    assert _stored_body(box) == b"one\ntwo\x00\n"


def test_no_file_of_the_runner_is_left_beside_the_result_or_in_the_temporary_folder(box):
    _fail_reset_of_3(box, output={"reset 3": "x"})
    assert list(box.tmp.iterdir()) == []

    (box.top / "reset.count").unlink()
    box.answer()
    assert box.run().returncode == 0
    assert list(box.tmp.iterdir()) == [] and sorted(p.name for p in box.run_dir.iterdir()) == ["result", "run.lock"]


# ------------------------------------------------ the repair: the held output is best-effort, never part of the run

CAPTURE_WARNING = "run_once: the output of the steps cannot be held; a failed step keeps its header lines only"

MKDIR_WITH_HELD_AS_FOLDER = """#!/bin/sh
case $1 in -m) "{real}" "$@" && "{real}" "$3/held" ;; *) exec "{real}" "$@" ;; esac
"""

# name: (answers, failed step, journey line); each is failed in a run with a usable folder and in one without
FAILING = {
    "refresh": ({"rc": {"refresh": 7}}, "refresh", None),
    "first-reset": ({"rc": {"reset first": 7}}, "reset", None),
    "reset": ({"rc": {"reset 3": 7}}, "reset", "3"),
    "isolation-server1": ({"rc": {"isolation server1": 7}}, "isolation", None),
    "seed": ({"rc": {"seed 3": 7}}, "seed", "3"),
    "proxy-answer": ({"proxy": '{"keys":[]}', "stderr": {"proxy 3": f'"kid":"{KID}"\n'}}, "proxy", "3"),
    "proxy-status": ({"rc": {"proxy 3": 22}}, "proxy", "3"),
    "owner-app-answer": ({"page": "<html>Something went wrong</html>", "stderr": {"app 3": "curl-error\n"}}, "owner-app", "3"),
    "journeys": ({"rc": {"journey 3": 7}}, "journeys", "3"),
}


def _control(box: Sandbox, **answers) -> tuple[Sandbox, subprocess.CompletedProcess[str]]:
    """The same run in a sandbox of its own, with the folder of the held output usable."""
    (box.top / "control").mkdir()
    control = Sandbox(box.top / "control")
    control.answer(**answers)
    return control, control.run()


def _folder_is_unusable(box: Sandbox, answers: dict | None = None, **env: str) -> subprocess.CompletedProcess[str]:
    box.answer(**(answers or {}))
    return box.run(TMPDIR="/dev/null", **env)


def _file_cannot_be_opened(box: Sandbox, answers: dict | None = None, **env: str) -> subprocess.CompletedProcess[str]:
    """The folder is made, and the held file in it is a folder, so that no open of it can succeed."""
    mkdir = box.bin / "mkdir"
    mkdir.unlink()
    mkdir.write_text(MKDIR_WITH_HELD_AS_FOLDER.format(real=shutil.which("mkdir")), encoding="utf-8")
    mkdir.chmod(0o755)
    box.answer(**(answers or {}))
    return box.run(**env)


UNUSABLE = {"no-folder": _folder_is_unusable, "unopenable-file": _file_cannot_be_opened}


def _facts(box: Sandbox) -> dict[str, str]:
    return {key: value for key, value in box.result().items() if key not in ("started", "finished")}


def _is_the_run_of_the_main_branch(box, done, control, good) -> None:
    assert done.returncode == good.returncode and done.stdout == good.stdout == ""
    assert box.calls() == control.calls()  # each original command ran once, in the same order, with the same words
    assert _facts(box) == _facts(control)
    assert box.told() == control.told()
    assert box.renames() == control.renames()
    assert good.stderr == ""
    assert done.stderr == CAPTURE_WARNING + "\n"  # one line for the run, not one for each command
    assert list(box.tmp.iterdir()) == []


@pytest.mark.parametrize("unusable", UNUSABLE)
def test_a_passing_run_goes_on_as_on_the_main_branch_when_the_held_output_cannot_be_held(box, unusable):
    control, good = _control(box)
    assert good.returncode == 0

    done = UNUSABLE[unusable](box)

    _is_the_run_of_the_main_branch(box, done, control, good)
    assert box.names() == [*PREFIX, *loop_calls(), "reset last", "hermes"]
    assert box.result()["result"] == "passed" and box.told() == ["Journeys passed."]
    assert not (box.run_dir / KEPT).exists()


# each is run once for each way that the held output can be unavailable; the owner-app one waits out its 120 seconds
UNUSABLE_AND_FAILING = [
    (unusable, name) for unusable in UNUSABLE
    for name in ("refresh", "first-reset", "reset", "isolation-server1", "seed", "proxy-answer", "proxy-status",
                 "owner-app-answer", "journeys")
]


@pytest.mark.parametrize(("unusable", "name"), UNUSABLE_AND_FAILING)
def test_a_failing_run_ends_as_on_the_main_branch_with_a_header_only_file_when_the_held_output_cannot_be_held(
    box, unusable, name,
):
    answers, step, journey = FAILING[name]
    control, good = _control(box, **answers)
    assert good.returncode == 1

    done = UNUSABLE[unusable](box, answers)

    _is_the_run_of_the_main_branch(box, done, control, good)
    result = box.result()
    assert (result["result"], result["step"], result["reason"]) == ("failed", step, REASONS[step])
    assert result.get("journey") == journey
    assert box.told() == [f"Journeys failed at step {step}{' for journey ' + journey if journey else ''}: {REASONS[step]}."]
    assert box.call("hermes")["env"] == {"kept": "True"}  # the header-only file is there before the owner line
    assert box.renames() == ["absent", "absent"]
    assert _kept(box) == [
        f"step={step}", *([f"journey={journey}"] if journey else []),
        "started=" + result["started"], "finished=" + result["finished"],
    ]
    assert (box.run_dir / KEPT).stat().st_mode & 0o777 == 0o600


@pytest.mark.parametrize("unusable", UNUSABLE)
@pytest.mark.parametrize(("phrase", "page", "end"), [PARITY[2], PARITY[5]])
def test_the_owner_app_decides_as_before_when_the_held_output_cannot_be_held(box, unusable, phrase, page, end):
    done = UNUSABLE[unusable](box, {"page": page}, JOURNEY_PAGE_TEXT=phrase)

    assert box.result()["step" if end != "passed" else "result"] == end
    assert done.returncode == (0 if end == "passed" else 1)
    assert done.stderr == CAPTURE_WARNING + "\n"


def test_a_held_file_that_breaks_in_the_middle_of_a_passing_run_does_not_change_the_run(box):
    control, good = _control(box)

    box.answer(break_held="reset first")
    done = box.run()

    _is_the_run_of_the_main_branch(box, done, control, good)
    assert box.result()["result"] == "passed" and not (box.run_dir / KEPT).exists()


@pytest.mark.parametrize("name", ["proxy-answer", "journeys"])
def test_a_held_file_that_breaks_in_the_middle_of_a_failing_run_leaves_a_header_only_file_with_none_of_the_earlier_output(
    box, name,
):
    answers, step, journey = FAILING[name]
    answers = {**answers, "output": {"reset 3": "OUTPUT-OF-THE-RESET", "journey 3": "OUTPUT-OF-THE-JOURNEY"}}
    control, good = _control(box, **answers)

    box.answer(break_held="reset 3", **answers)
    done = box.run()

    _is_the_run_of_the_main_branch(box, done, control, good)
    result = box.result()
    assert (result["step"], result["journey"]) == (step, journey)
    assert _kept(box) == [
        f"step={step}", f"journey={journey}", "started=" + result["started"], "finished=" + result["finished"],
    ]


# ------------------------------------------------------------ rework: a long output, a formatter that fails, a full capture file

LONG_OUTPUT_BYTES = 9_500_000
SERVICE_MEMORY_KIB = 65536  # ulimit -v of the service: 64 MiB
KEPT_WARNING = "run_once: the output of the failed step could not be kept"


def test_a_failed_step_with_9500000_bytes_of_output_under_the_64_mib_limit_ends_as_the_main_branch_does(box):
    # The main branch: the failed result with the fixed reason, one owner line, exit 1. The log is what the card adds.
    box.answer(rc={"reset 3": 7}, bulk={"reset 3": LONG_OUTPUT_BYTES})

    done = box.run(memory_kib=SERVICE_MEMORY_KIB)

    result = box.result()
    assert (box.run_dir / "result").read_text(encoding="utf-8") == (
        f"result=failed\nstep=reset\nreason={REASONS['reset']}\nstarted={result['started']}\njourney=3\n"
        f"finished={result['finished']}\n"
    )
    assert box.told() == [f"Journeys failed at step reset for journey 3: {REASONS['reset']}."]
    assert done.returncode == 1
    assert done.stdout == "" and done.stderr == ""
    lines = _kept(box)
    assert lines[:4] == ["step=reset", "journey=3", "started=" + result["started"], "finished=" + result["finished"]]
    assert lines[4:] == [f"{'row %06d' % i:<18}" for i in range(499_960, 500_000)]
    assert list(box.tmp.iterdir()) == []


@pytest.mark.parametrize("tool", ["tail", "cut"])
def test_a_formatter_that_fails_still_stores_the_result_tells_the_owner_once_and_warns_once(box, tool):
    answers = {"rc": {"reset 3": 7}, "output": {"reset 3": "first line\nsecond line"}}
    control, good = _control(box, **answers)
    assert good.returncode == 1 and good.stderr == ""
    (box.bin / tool).unlink()
    (box.bin / tool).write_text("#!/bin/sh\necho formatter-boom >&2\nexit 3\n", encoding="utf-8")
    (box.bin / tool).chmod(0o755)
    box.answer(**answers)

    done = box.run()

    assert done.returncode == good.returncode == 1
    assert _facts(box) == _facts(control) and box.result()["reason"] == REASONS["reset"]
    assert box.told() == control.told() == [f"Journeys failed at step reset for journey 3: {REASONS['reset']}."]
    assert box.names() == control.names()
    assert done.stdout == "" and done.stderr == KEPT_WARNING + "\n"
    result = box.result()
    assert _kept(box) == ["step=reset", "journey=3", "started=" + result["started"], "finished=" + result["finished"]]
    assert list(box.tmp.iterdir()) == []


# label of the check, the label of the call before it (the held file is made a link to /dev/full there), a failing answer
FULL_CAPTURE_CHECKS = {
    "proxy": ("seed 3", "proxy 3", {"proxy": '{"keys":[]}'}),
    "owner-app": ("proxy 3", "app 3", {"page": "<html>Something went wrong</html>"}),
}


@pytest.mark.parametrize("check", FULL_CAPTURE_CHECKS)
def test_a_check_answer_that_cannot_be_appended_to_a_full_capture_file_leaves_the_run_as_the_main_branch_would(box, check):
    before, _label, _failing = FULL_CAPTURE_CHECKS[check]
    control, good = _control(box)
    assert good.returncode == 0 and good.stderr == ""
    box.answer(link_held={before: "/dev/full"})

    done = box.run()

    assert done.returncode == good.returncode == 0
    assert done.stdout == "" and done.stderr == CAPTURE_WARNING + "\n"  # one warning, though two checks and a journey follow
    assert box.names() == control.names() and box.told() == control.told() == ["Journeys passed."]
    assert _facts(box) == _facts(control)
    assert not (box.run_dir / KEPT).exists()
    assert list(box.tmp.iterdir()) == []


@pytest.mark.parametrize("check", FULL_CAPTURE_CHECKS)
def test_a_failing_check_whose_answer_cannot_be_appended_leaves_a_header_only_log_and_the_decision_of_the_main_branch(
    box, check,
):
    before, label, failing = FULL_CAPTURE_CHECKS[check]
    control, good = _control(box, **failing)
    assert good.returncode == 1 and good.stderr == ""
    box.answer(link_held={before: "/dev/full"}, **failing)

    done = box.run()

    assert done.returncode == good.returncode == 1
    assert done.stdout == "" and done.stderr == CAPTURE_WARNING + "\n"
    assert box.names() == control.names() and box.names()[-1] == "hermes" and label in box.names()
    assert _facts(box) == _facts(control)
    result = box.result()
    assert (result["result"], result["step"], result["journey"]) == ("failed", check, "3")
    assert box.told() == control.told() == [f"Journeys failed at step {check} for journey 3: {REASONS[check]}."]
    assert _kept(box) == [f"step={check}", "journey=3", "started=" + result["started"], "finished=" + result["finished"]]
    assert list(box.tmp.iterdir()) == []
