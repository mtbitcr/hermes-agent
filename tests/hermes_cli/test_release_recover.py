"""Card 7: a release that stopped midway is brought back to exactly PREV or exactly NEW, read back,
without a person, and never moved forward. Each test runs on FakeHost, a release host in memory
built like the merged one; the command's tests under a temporary root home."""

from __future__ import annotations

import argparse
import contextlib
import errno
import hashlib
import json
import os
import signal
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import pytest
import yaml

from agent import estop
from hermes_cli import release_ledger as ledger
from hermes_cli import release_runner, release_unit
from hermes_cli.release_guards import Pins
from hermes_cli.release_runner import PLATFORM_UNITS, SANDBOX_TUNNEL_UNIT
from hermes_cli.subcommands.release import build_release_parser
from tests.hermes_cli.test_release_cmd import (
    RECOVERY_CAUSES, SECRET, _printed_outcome, _root_at, _send_fails, _two_homes, _unprivileged,
)


def _commit(label: str) -> str:
    return hashlib.sha1(label.encode()).hexdigest()


PREV, NEW, FEATURE, TREE = (_commit(label) for label in ("prev", "new", "feature", "tree"))
PINS = Pins(new=NEW, prev=PREV)
CHECKOUT = "/srv/hermes/checkout"
CONFIG = {"config.yaml": "digest"}
SETTINGS = {
    "units": {"sandbox-tunnel": "hermes-sandbox-tunnel"},
    "health_url": "http://127.0.0.1:8642/health",
}
OWNER = "owner: hold everything"
TOKEN = "placeholder-release-token"
# The merged restore steps: the units stop, the checkout goes to PREV, its environment and its
# configuration come back, and the units start.
RESTORE = ["stop", f"checkout {PREV}", "restore environment", "restore config", "start"]
_REPO = Path(__file__).resolve().parents[2]
# ``hermes release COMMAND BATCH`` in its own process, as _command runs it, on a host where
# neither version reads back, with the runner's wait between two reads replaced so none sleeps and
# its send to the owner replaced by nothing; its last line of output is its host's calls. Given
# "held", it says ready as it is about to count its recovery attempt, and goes on once its input
# ends.
# Given "at-prev", its host is PREV, healthy, with NEW's snapshot as the live reader reads it.
_PROCESS = (
    "import json, sys\n"
    "import pytest\n"
    "sys.path.insert(0, sys.argv.pop(1))\n"
    "import test_release_recover as t\n"
    "from hermes_cli import release_cmd\n"
    "pytest.MonkeyPatch().setattr(release_cmd, '_tell_owner', lambda line: None)\n"
    "t.release_runner._wait = lambda seconds: None\n"
    "begin, host = t.ledger.begin_recovery, t._midway(t.NEW, t.PREV)\n"
    "def held(*args):\n"
    "    print('ready', flush=True)\n"
    "    sys.stdin.read()\n"
    "    return begin(*args)\n"
    "if sys.argv[1] == 'at-prev':\n"
    "    saved = t._snapshots(t.Path.home() / '.hermes')\n"
    "    host = t.FakeHost(head=t.PREV, running=t.PREV, saved=saved)\n"
    "if sys.argv.pop(1) == 'held':\n"
    "    t.ledger.begin_recovery = held\n"
    "code = t._command(pytest.MonkeyPatch(), host, *sys.argv[1:])\n"
    "print(json.dumps(host.calls))\n"
    "sys.exit(code)\n"
)


@dataclass
class FakeHost:
    """A release host in memory, built like the merged one, its checkout at ``head`` and its units
    running ``running``, None once they stopped. A version in ``unhealthy`` fails its health
    check. Each action is noted in ``calls``, and the modules at the first unit stop are kept.

    The environment holds the packages of the version ``environment`` names, at first the head's;
    a sync installs the head's, and the restore moves back the release's copy, PREV's packages,
    when ``environment_snapshot`` names one."""

    head: str = NEW
    running: str | None = NEW
    active: set[str] = field(default_factory=lambda: {*PLATFORM_UNITS, SANDBOX_TUNNEL_UNIT})
    unhealthy: set[str] = field(default_factory=set)
    calls: list[str] = field(default_factory=list)
    modules_at_stop: set[str] | None = None
    config_snapshot: str = ""  # as in the live actions: the snapshot the restore puts back
    # NEW's configuration snapshot as the live reader reads it: empty when none was published.
    saved: dict[str, str] = field(default_factory=lambda: dict(CONFIG))
    environment: str | None = None
    environment_snapshot: str = ""  # as in the live actions: the copy the restore moves back

    def __post_init__(self):
        if self.environment is None:
            self.environment = self.head

    def checkout_head(self):
        return self.head

    def checkout_root(self):
        return CHECKOUT

    venv_import_root = checkout_root

    def unit_active(self, unit):
        return unit in self.active

    def health_ok(self):
        return self.running is not None and self.running not in self.unhealthy

    def workspace_reads_ok(self):
        return self.running is not None

    def fleet_version(self):
        return self.running or ""

    def live_config(self):
        return dict(CONFIG)

    def named_config_snapshot(self):
        return dict(self.saved)

    def stop_units(self, units):
        self.calls.append("stop")
        if self.modules_at_stop is None:
            self.modules_at_stop = set(sys.modules)
        self.active -= set(units)
        self.running = None

    def start_units(self, units):
        self.calls.append("start")
        self.active |= set(units)
        if self.running is None:  # units that never stopped keep running what they ran
            self.running = self.head

    def checkout(self, commit):
        self.calls.append(f"checkout {commit}")
        self.head = commit

    def restore_config(self):
        self.calls.append("restore config")

    def sync_environment(self):
        self.environment = self.head

    def environment_differences(self):
        if self.environment == self.head:
            return ()
        return (f"package expected {self.head}, installed {self.environment}",)

    def environment_matches_lock(self):
        return not self.environment_differences()

    def take_environment_snapshot(self, name):
        self.environment_snapshot = name

    def restore_environment(self):
        self.calls.append("restore environment")
        if self.environment_snapshot:
            self.environment = PREV


def _midway(*unhealthy: str) -> FakeHost:
    """A host the release left midway: its units stopped, its checkout and environment at NEW, and
    the copy of PREV's environment it took, named after NEW."""
    return FakeHost(
        running=None, active={SANDBOX_TUNNEL_UNIT}, unhealthy=set(unhealthy),
        environment_snapshot=NEW,
    )


# What a release run asks of its host beyond FakeHost's reads, answered as the merged run's host
# answers it: every guard passes for NEW on PREV.
ANSWERS = {
    "origin_main": NEW, "first_parent_chain": [NEW], "commit_parents": (PREV, FEATURE),
    "commit_tree": TREE, "changed_paths": {"hermes_cli/main.py"}, "free_disk_bytes": 5 * 1024**3,
    "snapshots": [], "open_native_runs": [], "environment_lock_differences": (),
}


@dataclass
class RunHost(FakeHost):
    """FakeHost as a release run finds it: its checkout at PREV and its units running PREV. The
    reads of the run's guards that FakeHost lacks get ANSWERS, as in the merged run's host, and
    any other read or action succeeds."""

    head: str = PREV
    running: str | None = PREV

    def __getattr__(self, name):
        return lambda *args: ANSWERS.get(name, True)

    def unit_property(self, unit, name):
        return {"KillSignal": "SIGINT", "KillMode": "mixed"}[name]


@pytest.fixture
def root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """The root Hermes home at ``<tmp>/.hermes``, with the release settings this host needs."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    (home / "config.yaml").write_text(yaml.safe_dump({"release": SETTINGS}))
    return home


@pytest.fixture(autouse=True)
def waits(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """The seconds asked for by each wait between two reads of a readback, in order. Every test
    here replaces the runner's wait function with this record, so no test sleeps; a process
    from _process replaces it too."""
    asked: list[float] = []
    monkeypatch.setattr(release_runner, "_wait", asked.append)
    return asked


class Sent(list):
    """The lines a recovery told the owner, in order; ``real`` is the send function itself."""


@pytest.fixture(autouse=True)
def sent(monkeypatch: pytest.MonkeyPatch) -> Sent:
    """Every test here: the send function is replaced by a record of its lines, so no test starts
    a real send. The record imports nothing when called, as K6 needs."""
    from hermes_cli import release_cmd

    lines = Sent()
    lines.real = release_cmd._tell_owner
    monkeypatch.setattr(release_cmd, "_tell_owner", lines.append)
    return lines


@pytest.fixture(autouse=True)
def _default_sigint():
    """A recovery installs no SIGINT handler of its own. A runner can start the tests with SIGINT
    ignored, so each test runs under Python's default handler and the caller's is put back."""
    previous = signal.signal(signal.SIGINT, signal.default_int_handler)
    yield
    signal.signal(signal.SIGINT, previous or signal.SIG_DFL)


def _accepted() -> int:
    """The record a release begins from: one change, NEW, and its batch accepted."""
    with contextlib.closing(ledger.connect()) as conn:
        batch = ledger.record_merge(
            conn, merge_commit=NEW, pr_url="https://github.com/mtbitcr/hermes-agent/pull/1",
            reviewed_base=PREV, reviewed_head=FEATURE, reviewed_tree=TREE, tier=1,
            card_id=f"t_{NEW[:8]}", on_main=True,
        )["batch"]
        return ledger.decide_release(
            conn, batch["batch_id"], decision="accepted", shown_digest=batch["digest"],
            expected_version=batch["version"], decision_ref="decisions-page:release",
        )["batch_id"]


def _stopped_midway(pause: str = "own") -> int:
    """The record and the pause a release leaves when it stops midway: one change, NEW, accepted
    and begun from PREV, and the release's own pause, with its token, or an owner's pause."""
    from hermes_cli import release_cmd

    with contextlib.closing(ledger.connect()) as conn:
        batch = ledger.record_merge(
            conn, merge_commit=NEW, pr_url="https://github.com/mtbitcr/hermes-agent/pull/1",
            reviewed_base=PREV, reviewed_head=FEATURE, reviewed_tree=TREE, tier=1,
            card_id=f"t_{NEW[:8]}", on_main=True,
        )["batch"]
        ledger.decide_release(
            conn, batch["batch_id"], decision="accepted", shown_digest=batch["digest"],
            expected_version=batch["version"], decision_ref="decisions-page:release",
        )
        batch_id = ledger.begin_release(conn, batch["batch_id"], prev=PREV, new=NEW)["batch_id"]
    if pause == "own":
        assert release_cmd._pause(f"release {batch_id}", TOKEN)
    else:
        estop.engage(OWNER)
    return batch_id


def _command(monkeypatch: pytest.MonkeyPatch, host: FakeHost, *argv: str) -> int:
    """``hermes release ARGV`` on ``host``, parsed by its own parser before anything is patched."""
    from hermes_cli import release_cmd

    parser = argparse.ArgumentParser(prog="hermes")
    build_release_parser(parser.add_subparsers())
    args = parser.parse_args(["release", *argv])
    for adapter in ("LiveHostReader", "ReleaseHostActions"):
        monkeypatch.setattr(release_cmd, adapter, lambda **settings: host)
    return args.func(args)


def _batch(batch_id: int) -> dict:
    with contextlib.closing(ledger.connect()) as conn:
        return next(batch for batch in ledger.list_batches(conn) if batch["batch_id"] == batch_id)


def _sentinel() -> bytes | None:
    """The pause sentinel's bytes, or None when there is none."""
    with contextlib.suppress(OSError):
        return estop.sentinel_path().read_bytes()
    return None


def _at_attempt_two() -> int:
    """A batch whose release stopped midway and whose recovery then failed twice: failed, at
    attempt count two, under the release's own pause."""
    batch_id = _stopped_midway()
    with contextlib.closing(ledger.connect()) as conn:
        for _attempt in range(2):
            ledger.begin_recovery(conn, batch_id)
            ledger.finish_release(conn, batch_id, outcome="failed")
    return batch_id


def _process(
    root: Path, command: str, batch_id: int, mode: str = "free", user: tuple[int, int] | None = None
) -> subprocess.Popen:
    """``hermes release COMMAND BATCH`` in its own process (see _PROCESS), on the root home
    ``root`` and its release record, as ``user``'s uid and gid when given."""
    as_user = {} if user is None else {"user": user[0], "group": user[1], "extra_groups": []}
    return subprocess.Popen(
        [sys.executable, "-c", _PROCESS, str(Path(__file__).parent), mode,
         command, str(batch_id)],
        cwd=_REPO,
        env={**os.environ, "HOME": str(root.parent), "HERMES_HOME": str(root),
             "PYTHONPATH": str(_REPO), "PYTHONDONTWRITEBYTECODE": "1"},
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, **as_user,
    )


def _finish(process: subprocess.Popen) -> tuple[int, list[str], list[str]]:
    """The exit code, output lines and host calls of a process from _process, once its input
    ends."""
    out, err = process.communicate(timeout=60)
    *lines, calls = out.splitlines() or [""]
    assert calls.startswith("["), err
    return process.returncode, lines, json.loads(calls)


def _files(root: Path) -> dict[str, bytes]:
    """Every file under the root home with its bytes, the release record's and the pause's
    among them."""
    return {str(path): path.read_bytes() for path in root.rglob("*") if path.is_file()}


# Where a release stopped before its cutover, as the snapshot root shows it (S-C): the
# configuration snapshot the release of PREV published, and of NEW's either nothing (before the
# fetch of NEW), its staging, or its staging with an earlier one set aside, not yet replaced.
_BEFORE_CUTOVER = {
    "before-fetch": [f"config/{PREV}"],
    "staged": [f"config/{PREV}", f".config-{NEW}.partial"],
    "set-aside": [f"config/{PREV}", f".config-{NEW}.partial", f".config-{NEW}.previous"],
}


def _snapshots(root: Path, *names: str) -> dict[str, str]:
    """These entries of the snapshot root, release-snapshots under the root home (S-C), each
    holding a saved configuration file; and NEW's configuration snapshot as the live reader then
    reads it, empty when there is none."""
    from hermes_cli.release_host import LiveHostReader

    for name in names:
        saved = root / "release-snapshots" / name
        saved.mkdir(parents=True)
        (saved / "config.yaml").write_text(yaml.safe_dump({"release": SETTINGS}))
    reader = LiveHostReader(
        checkout=Path(CHECKOUT), root_home=root, units={},
        snapshot_root=root / "release-snapshots", snapshot_name=NEW,
    )
    return dict(reader.named_config_snapshot())


def test_recover_keeps_new_when_it_reads_back():
    host = FakeHost()

    result = release_runner.recover(host, PINS)

    assert (result.outcome, result.steps, host.calls) == ("released", ("readback",), [])
    assert [(readback.expected, readback.ok) for readback in result.readbacks] == [(NEW, True)]


def test_recover_keeps_prev_when_it_reads_back():
    host = FakeHost(head=PREV, running=PREV)

    result = release_runner.recover(host, PINS)

    assert (result.outcome, result.steps, host.calls) == ("restored", ("readback",), [])
    assert [(readback.expected, readback.ok) for readback in result.readbacks] == [(PREV, True)]


def test_recover_restores_prev_from_a_midway_checkout():
    host = _midway(NEW)

    result = release_runner.recover(host, PINS)

    assert result.steps == ("readback", "restore", "readback")
    assert [(readback.expected, readback.ok) for readback in result.readbacks] == [
        (NEW, False), (PREV, True),
    ]
    assert (result.outcome, host.calls, host.head, host.running) == ("restored", RESTORE, PREV, PREV)
    assert result.error == f"ReadbackFailed: R3, R4, R6, fleet version did not hold on {NEW}"


@pytest.mark.parametrize("head", [PREV, _commit("other")], ids=["at-prev", "at-neither"])
def test_recover_reports_failed_and_never_moves_forward(head):
    """NEW would read back, but neither the checkout nor PREV does: recovery reports failed, and
    NEW is never checked out."""
    host = FakeHost(head=head, running=None, active={SANDBOX_TUNNEL_UNIT}, unhealthy={PREV})

    result = release_runner.recover(host, PINS)

    assert (result.outcome, host.calls, host.head) == ("failed", RESTORE, PREV)
    assert result.steps[-2:] == ("restore", "readback") and result.readbacks[-1].expected == PREV
    assert not any(readback.ok for readback in result.readbacks)


def test_recovery_reads_back_after_its_restore_until_every_check_holds(waits):
    """The readback after the restore, which starts the units, takes a release's window: PREV's
    health endpoint answers from 10 seconds after its start, so the third read is the first whose
    checks all hold, and it alone decides and is recorded. The readback of the host as recovery
    finds it follows no start: it is read once, though NEW never holds there."""
    host, reads = _midway(), []

    def health_ok():  # PREV's health endpoint answers from 10 seconds after its start
        reads.append((host.running, sum(waits)))  # only the waits move time on
        return host.running == PREV and sum(waits) >= 10

    host.health_ok = health_ok
    result = release_runner.recover(host, PINS)

    assert waits == [5, 5]
    # One read of the stopped units at NEW, then three of PREV after the restore, 5 seconds apart.
    assert reads == [(None, 0), (PREV, 0), (PREV, 5), (PREV, 10)]
    assert (result.outcome, result.steps, host.calls) == (
        "restored", ("readback", "restore", "readback"), RESTORE,
    )
    assert [(readback.expected, readback.ok) for readback in result.readbacks] == [
        (NEW, False), (PREV, True),
    ]


def test_recovery_that_never_holds_fails_after_twelve_reads(waits):
    """PREV's checks never all hold once the restore starts it: the readback reads the host 12
    times, 5 seconds apart, and recovery fails with the last read, the only one it records, whose
    errors it keeps."""
    host, reads = _midway(), []

    def health_ok():  # PREV's health endpoint never answers; each error says when it was read
        reads.append((host.running, sum(waits)))
        if host.running != PREV:
            return False
        raise ConnectionError(f"no answer from {PREV} {sum(waits)} seconds after its start")

    host.health_ok = health_ok
    result = release_runner.recover(host, PINS)

    assert waits == [5] * 11
    assert reads == [(None, 0)] + [(PREV, 5 * read) for read in range(12)]
    assert (result.outcome, result.steps, host.calls) == (
        "failed", ("readback", "restore", "readback"), RESTORE,
    )
    assert [(readback.expected, readback.ok) for readback in result.readbacks] == [
        (NEW, False), (PREV, False),
    ]
    assert result.readbacks[-1].errors == {
        "R4": f"ConnectionError: no answer from {PREV} 55 seconds after its start",
    }


def test_recovery_reads_back_once_after_a_start_that_failed(waits):
    """The restore's start of the units fails, so it started nothing to wait for: PREV is read
    back once, and recovery fails."""
    host = _midway()

    def start_units(units):  # PREV's units fail to start
        host.calls.append("start")
        raise RuntimeError(f"units failed to start on {host.head}")

    host.start_units = start_units
    result = release_runner.recover(host, PINS)

    assert waits == []
    assert (result.outcome, result.steps, host.calls) == (
        "failed", ("readback", "restore", "readback"), RESTORE,
    )
    assert result.restore_errors == (f"RuntimeError: units failed to start on {PREV}",)


@pytest.mark.parametrize(
    ("host", "pause", "outcome", "code"),
    [(FakeHost, "own", "released", 0),
     (_midway, "own", "restored", release_unit.RESTORED_EXIT_STATUS),
     (FakeHost, "owner", "released", 0)],
    ids=["new-reads-back", "prev-restored", "owner-pause-stays"],
)
def test_run_on_a_releasing_batch_recovers_and_resumes(
    root, monkeypatch, capsys, host, pause, outcome, code
):
    """The release unit's ``release run BATCH`` on a releasing batch goes straight to recovery
    with the stored versions, records the outcome and lifts the release's own pause only."""
    batch_id, host = _stopped_midway(pause), host()
    _snapshots(root, f"config/{NEW}")  # the release published it before its cutover (S-C)
    before = _sentinel()

    assert _command(monkeypatch, host, "run", str(batch_id)) == code

    out = capsys.readouterr().out
    batch = _batch(batch_id)
    assert (batch["outcome"], batch["recovery_attempts"]) == (outcome, 1)
    assert host.calls == ([] if outcome == "released" else RESTORE)
    assert host.config_snapshot == NEW  # the one the release saved before its pause (S-C)
    assert _sentinel() == (None if pause == "own" else before)
    assert out.startswith(f"Batch {batch_id} stopped midway: recovery attempt 1 of 3, PREV {PREV},"
                          f" NEW {NEW}\n")
    if pause == "owner":
        assert out.endswith(f"The platform stays paused ({OWNER}).\n")


def test_failed_recovery_keeps_the_pause(root, monkeypatch, capsys):
    batch_id = _stopped_midway()
    host, before = _midway(NEW, PREV), _sentinel()

    assert _command(monkeypatch, host, "recover", str(batch_id)) == 1

    out = capsys.readouterr().out
    batch = _batch(batch_id)
    assert (batch["state"], batch["outcome"], batch["recovery_attempts"]) == ("failed", "failed", 1)
    assert host.calls == RESTORE and _sentinel() == before
    assert f"Batch {batch_id} failed midway and is kept for recovery." in out
    assert out.endswith(f"The platform stays paused (release {batch_id}).\n")


def test_recovery_attempts_are_capped(root, monkeypatch, capsys):
    """Each failed attempt exits 1, so the unit starts again, until the third: the outcome then
    stays failed and the unit exits cleanly. A later start asks the host nothing."""
    batch_id = _stopped_midway()
    host, before = _midway(NEW, PREV), _sentinel()

    codes = [_command(monkeypatch, host, "run", str(batch_id)) for _start in range(4)]

    batch = _batch(batch_id)
    assert codes == [1, 1, 0, 0]
    assert (batch["state"], batch["outcome"], batch["recovery_attempts"]) == ("failed", "failed", 3)
    assert host.calls == RESTORE * 3 and _sentinel() == before
    capped = (f"Batch {batch_id} stays failed after 3 recovery attempts: a person must bring the"
              " platform back.")
    assert capsys.readouterr().out.splitlines().count(capped) == 2


def test_recover_imports_nothing(root, monkeypatch):
    """K6: from the first unit stop to the pause's removal, recovery imports nothing, so no code
    loads from the checkout it moves."""
    batch_id, host = _stopped_midway(), _midway(NEW)
    _snapshots(root, f"config/{NEW}")

    code = _command(monkeypatch, host, "recover", str(batch_id))
    assert code == release_unit.RESTORED_EXIT_STATUS
    assert host.calls == RESTORE and _sentinel() is None
    assert set(sys.modules) - host.modules_at_stop == set()


@pytest.mark.parametrize(("first", "second"), [("run", "recover"), ("recover", "run")])
def test_two_processes_never_count_past_three(root, first, second):
    """Owner rule 443, test 1 (finding 1 of the card review). On a temporary record at attempt
    count two, ``first`` waits in its own process as it is about to count its attempt, and
    ``second`` starts in another: it refuses at once and asks its host nothing. The first makes
    the third attempt, and the count never exceeds three."""
    batch_id = _at_attempt_two()

    with _process(root, first, batch_id, "held") as waiting:
        assert waiting.stdout.readline() == "ready\n"
        code, _out, calls = _finish(_process(root, second, batch_id))
        first_code, _first_out, first_calls = _finish(waiting)

    batch = _batch(batch_id)
    assert (batch["state"], batch["outcome"], batch["recovery_attempts"]) == ("failed", "failed", 3)
    assert (code, calls) == (1, [])  # the second asked its host nothing
    assert (first_code, first_calls) == (0, RESTORE)  # the third attempt failed: no restart


def test_a_second_recovery_refuses_at_once_and_writes_nothing(root):
    """Owner rule 443, test 2: two concurrent recover commands. While the first waits as it is
    about to count its attempt, the second refuses at once: it writes nothing to the release
    record, leaves the pause as it was and asks its host nothing."""
    batch_id = _stopped_midway()

    with _process(root, "recover", batch_id, "held") as waiting:
        assert waiting.stdout.readline() == "ready\n"
        before = _files(root)
        code, out, calls = _finish(_process(root, "recover", batch_id))
        after = _files(root)
        first_code, _first_out, first_calls = _finish(waiting)

    assert (code, calls) == (1, [])
    assert after == before  # not a byte of the release record or the pause changed
    lock = root / ".release.lock"
    assert out == [
        f"Refused: another release run or recovery holds the release lock {lock}.",
        "Nothing in the release state was written.",
    ]
    assert (first_code, first_calls, _batch(batch_id)["recovery_attempts"]) == (1, RESTORE, 1)


def _unit_starts(monkeypatch: pytest.MonkeyPatch, host: FakeHost, batch_id: int) -> tuple:
    """Every start the merged release unit makes of ``release run BATCH``: a start follows only an
    exit other than 0 and RESTORED_EXIT_STATUS, which the unit lists in RestartPreventExitStatus,
    START_LIMIT_BURST starts in all. Each start's exit code, then the batch as the record shows
    it, the host's calls and the pause after it."""
    codes, records, calls, pauses = [], [], [], []
    final = (0, release_unit.RESTORED_EXIT_STATUS)
    while len(codes) < release_unit.START_LIMIT_BURST and (not codes or codes[-1] not in final):
        done = len(host.calls)
        codes.append(_command(monkeypatch, host, "run", str(batch_id)))
        batch = _batch(batch_id)
        records.append((batch["state"], batch["outcome"], batch["recovery_attempts"]))
        calls.append(host.calls[done:])
        pauses.append(_sentinel())
    return codes, records, calls, pauses


def test_the_unit_starts_once_for_the_run_and_once_for_each_recovery_attempt():
    from hermes_cli import release_cmd

    assert release_unit.START_LIMIT_BURST == 1 + release_cmd._RECOVERY_ATTEMPTS


def test_a_release_that_fails_on_the_first_start_gets_three_recoveries(root, monkeypatch):
    """Owner rule 456, test 1, as the 5 November release showed it. The release unit's first start
    runs the release of an accepted batch on a host where neither NEW nor PREV reads back, and the
    release ends failed under its own pause. Each later start makes one recovery attempt, all
    three that recovery allows: after the third the batch stays failed with its pause, and that
    start exits 0, so the unit, with a start left for each attempt, starts no more."""
    batch_id, host = _accepted(), RunHost(unhealthy={NEW, PREV})

    codes, records, calls, pauses = _unit_starts(monkeypatch, host, batch_id)

    assert codes == [1, 1, 1, 0]  # four starts in all
    assert records == [("failed", "failed", attempts) for attempts in (0, 1, 2, 3)]
    # The release: its first unit stop, the move to NEW and the restore; then one restore a start.
    assert calls == [["stop", "stop", f"checkout {NEW}", "start", *RESTORE], *[RESTORE] * 3]
    pause = json.loads(pauses[0])  # the release's own: its reason and its token
    assert pause["reason"] == f"release {batch_id}" and pause["release_token"]
    assert pauses == [pauses[0]] * 4  # kept byte for byte


@pytest.mark.parametrize("moment", ["release", "recovery"])
def test_an_outcome_of_restored_takes_no_new_start(root, monkeypatch, moment):
    """A release, or a recovery of one that stopped midway, that ends restored exits with
    RESTORED_EXIT_STATUS: the unit does not start it again, as the next run would only refuse the
    folded batch."""
    if moment == "release":  # NEW never reads back; PREV does after the restore
        batch_id, host = _accepted(), RunHost(unhealthy={NEW})
    else:
        batch_id, host = _stopped_midway(), _midway(NEW)
        _snapshots(root, f"config/{NEW}")  # the release published it before its cutover (S-C)

    codes, records, calls, pauses = _unit_starts(monkeypatch, host, batch_id)

    attempts = 0 if moment == "release" else 1
    assert (codes, records, pauses) == (
        [release_unit.RESTORED_EXIT_STATUS], [("folded", "restored", attempts)], [None]
    )
    assert calls[0][-len(RESTORE):] == RESTORE


@pytest.mark.parametrize("failure", ["removal", "readback"])
def test_a_restored_recovery_exits_so_the_unit_does_not_start_it_again_whatever_its_pause(
    root, monkeypatch, capsys, failure
):
    """Owner rule of 5 November, item 3 (finding 2). Once a recovery records restored, the
    removal of the release's own pause fails, or its readback, both with EIO: recovery keeps the
    outcome and the pause warning, and still exits with the status the release unit never
    restarts on."""
    batch_id, host = _stopped_midway(), _midway(NEW)
    _snapshots(root, f"config/{NEW}")  # the release published it before its cutover (S-C)
    finish, read_text, unlink = ledger.finish_release, Path.read_text, os.unlink

    def eio(real):  # the call, failed on the pause sentinel as a failing disk fails it
        def call(path, *args, **kwargs):
            if Path(path).name == estop.SENTINEL_NAME:
                raise OSError(errno.EIO, os.strerror(errno.EIO), os.fspath(path))
            return real(path, *args, **kwargs)
        return call

    def finished(*args, **kwargs):  # the outcome is recorded; the pause fails from then on
        finish(*args, **kwargs)
        if failure == "readback":
            monkeypatch.setattr(Path, "read_text", eio(read_text))
        monkeypatch.setattr(os, "unlink", eio(unlink))

    monkeypatch.setattr(ledger, "finish_release", finished)
    code = _command(monkeypatch, host, "recover", str(batch_id))

    assert code == release_unit.RESTORED_EXIT_STATUS
    assert (_batch(batch_id)["outcome"], host.calls) == ("restored", RESTORE)
    assert json.loads(_sentinel())["release_token"] == TOKEN  # the release's own pause stays
    out = capsys.readouterr().out
    assert f"Batch {batch_id} was rolled back" in out
    if failure == "readback":
        assert "The pause could not be read" in out
    else:
        assert f"The platform is still paused (release {batch_id})" in out


def test_a_start_after_the_cap_makes_no_host_step_and_exits_cleanly(root, monkeypatch, capsys):
    """Owner rule 456, test 2. A start on a batch failed after three recovery attempts, under the
    release's own pause, as a new unit finds it: ``release run BATCH`` asks the host nothing and
    exits 0, and the count of three, the failed batch and the pause stay as they were."""
    batch_id = _stopped_midway()
    with contextlib.closing(ledger.connect()) as conn:
        for _attempt in range(3):
            ledger.begin_recovery(conn, batch_id)
            ledger.finish_release(conn, batch_id, outcome="failed")
    host, before = _midway(NEW, PREV), _sentinel()

    assert _command(monkeypatch, host, "run", str(batch_id)) == 0

    batch = _batch(batch_id)
    assert (batch["state"], batch["outcome"], batch["recovery_attempts"]) == ("failed", "failed", 3)
    assert host.calls == [] and _sentinel() == before
    assert capsys.readouterr().out == (
        f"Batch {batch_id} stays failed after 3 recovery attempts: a person must bring the"
        " platform back.\n"
    )


@pytest.mark.parametrize("pause", ["own", "owner"])
@pytest.mark.parametrize("stop", list(_BEFORE_CUTOVER))
def test_a_release_stopped_before_its_cutover_is_kept_as_it_is(root, monkeypatch, stop, pause):
    """Finding of the security review. A release that stopped after its releasing record and
    before it published NEW's configuration snapshot left PREV checked out, its units up and
    healthy. Recovery reads the host back as it is and records restored with no host step: no
    stop, checkout, restore or start. It lifts only the release's own pause; an owner's stays."""
    batch_id = _stopped_midway(pause)
    saved = _snapshots(root, *_BEFORE_CUTOVER[stop])
    host, before = FakeHost(head=PREV, running=PREV, saved=saved), _sentinel()

    code = _command(monkeypatch, host, "recover", str(batch_id))
    assert code == release_unit.RESTORED_EXIT_STATUS

    batch = _batch(batch_id)
    assert (batch["outcome"], batch["recovery_attempts"]) == ("restored", 1)
    assert host.calls == []
    assert _sentinel() == (None if pause == "own" else before)


def test_a_published_snapshot_of_new_is_still_compared(root, monkeypatch, capsys):
    """Once the release published NEW's configuration snapshot, recovery compares the live
    configuration with it: PREV up and healthy, its live configuration other than the snapshot's,
    is not kept as it is. The restore runs, and as this host's configuration still differs after
    it, the outcome is failed and the pause stays."""
    batch_id = _stopped_midway()
    saved = _snapshots(root, f"config/{PREV}", f"config/{NEW}")
    host, before = FakeHost(head=PREV, running=PREV, saved=saved), _sentinel()

    assert _command(monkeypatch, host, "recover", str(batch_id)) == 1

    batch = _batch(batch_id)
    assert (batch["outcome"], batch["recovery_attempts"]) == ("failed", 1)
    assert host.calls == RESTORE and _sentinel() == before
    assert f"(ReadbackFailed: configuration did not hold on {PREV})" in capsys.readouterr().out


@pytest.mark.parametrize(
    ("files", "mode"), [([], 0o755), (["notes.txt"], 0o755), (["config.yaml"], 0)],
    ids=["empty", "unrelated-files", "unreadable"],
)
def test_a_published_snapshot_of_new_read_as_empty_is_still_compared(monkeypatch, files, mode):
    """Review finding of the earlier pass: NEW's published configuration snapshot is compared as
    above when the live reader reads it as empty: empty, with no configuration file, or mode 000,
    read as nobody when the tests run as root, which searches any folder. The outcome is failed."""
    user = None if mode else _unprivileged()
    with tempfile.TemporaryDirectory(prefix="release-recover-") as name:
        top = Path(name)
        top.chmod(0o755)
        root = _root_at(top, monkeypatch)
        batch_id, published = _stopped_midway(), root / "release-snapshots" / "config" / NEW
        published.mkdir(parents=True)
        for file in files:
            (published / file).write_text(yaml.safe_dump({"release": SETTINGS}))
        if user is not None:
            for item in [top, *top.rglob("*")]:
                os.chown(item, *user)
        before = _sentinel()
        published.chmod(mode)  # TemporaryDirectory's cleanup undoes it
        code, out, calls = _finish(_process(root, "recover", batch_id, "at-prev", user))
        batch = _batch(batch_id)
        assert (batch["outcome"], batch["recovery_attempts"], code) == ("failed", 1, 1)
        assert calls == RESTORE and _sentinel() == before
        assert f"(ReadbackFailed: configuration did not hold on {PREV})" in "\n".join(out)


def test_new_is_never_kept_without_its_configuration_snapshot(root, monkeypatch):
    """A release checks NEW out only after it published NEW's configuration snapshot, so NEW
    checked out without one is not kept: nothing shows its configuration held. The restore runs,
    nothing shows PREV's held either, and the outcome is failed with the pause in place."""
    batch_id = _stopped_midway()
    saved = _snapshots(root, f"config/{PREV}")
    host, before = FakeHost(saved=saved), _sentinel()

    assert _command(monkeypatch, host, "recover", str(batch_id)) == 1

    batch = _batch(batch_id)
    assert (batch["outcome"], batch["recovery_attempts"]) == ("failed", 1)
    assert host.calls == RESTORE and _sentinel() == before


# The host a recovery ends on, by its outcome: NEW reads back, PREV reads back, or neither does.
ENDS = {"released": FakeHost, "restored": _midway, "failed": lambda: _midway(NEW, PREV)}


@pytest.mark.parametrize("command", ["recover", "run"])
@pytest.mark.parametrize("outcome", list(ENDS))
def test_a_recovery_that_ends_tells_the_owner_once(
    root, monkeypatch, capsys, sent, command, outcome
):
    batch_id = _stopped_midway()
    _snapshots(root, f"config/{NEW}")  # the release published it before its cutover (S-C)

    _command(monkeypatch, ENDS[outcome](), command, str(batch_id))

    printed = _printed_outcome(capsys.readouterr().out, batch_id)
    paused = " The platform stays paused." if outcome == "failed" else ""
    assert _batch(batch_id)["outcome"] == outcome
    assert sent == [f"Release {batch_id} {printed}{paused}"]


@pytest.mark.parametrize("command", ["recover", "run"])
@pytest.mark.parametrize("outcome", ["released", "failed"])
def test_the_real_send_of_a_recovery_reaches_the_root_owner_whatever_profile_is_sticky(
    root, monkeypatch, capsys, sent, command, outcome
):
    """The send child is really started, offline, with a named profile made sticky: the line lands
    on the root home's chat with the root's token."""
    from hermes_cli import release_cmd

    logged, _ = _two_homes(root, monkeypatch)
    batch_id = _stopped_midway()
    _snapshots(root, f"config/{NEW}")  # the release published it before its cutover (S-C)
    monkeypatch.setattr(release_cmd, "_tell_owner", sent.real)

    code = _command(monkeypatch, ENDS[outcome](), command, str(batch_id))

    out = capsys.readouterr().out
    printed = _printed_outcome(out, batch_id)
    paused = " The platform stays paused." if outcome == "failed" else ""
    assert _batch(batch_id)["outcome"] == outcome
    assert code == (0 if outcome == "released" else 1)
    assert out.splitlines().count(f"Batch {batch_id} {release_cmd._OUTCOMES[outcome]}.") == 1
    [told] = logged()
    assert told["chat_id"] == "111111", f"expected 111111, got {told['chat_id']}"
    assert told["token"] == SECRET
    assert told["message"] == f"Release {batch_id} {printed}{paused}"


def _capped(monkeypatch: pytest.MonkeyPatch, branch: str) -> int:
    """A batch stopped midway whose recovery ends at the cap, on ``branch``: the last attempt ended
    before its outcome, or the count its own write returns is past the cap."""
    batch_id = _stopped_midway()
    if branch == "counted before":
        with contextlib.closing(ledger.connect()) as conn:
            for _attempt in range(3):
                ledger.begin_recovery(conn, batch_id)
    else:
        monkeypatch.setattr(ledger, "begin_recovery", lambda conn, batch: {"recovery_attempts": 4})
    return batch_id


@pytest.mark.parametrize("branch", ["counted before", "counted by its own write"])
def test_a_batch_capped_and_recorded_failed_tells_the_owner_once(root, monkeypatch, sent, branch):
    """Both caps that record failed: the last attempt ended before its outcome, or the count its
    own write returned is past the cap. Neither asks the host anything."""
    from hermes_cli import release_cmd

    batch_id = _capped(monkeypatch, branch)
    host = _midway(NEW, PREV)

    assert _command(monkeypatch, host, "recover", str(batch_id)) == 0

    assert (_batch(batch_id)["outcome"], host.calls) == ("failed", [])
    failed = release_cmd._OUTCOMES["failed"]
    assert sent == [f"Release {batch_id} {failed}. The platform stays paused."]


@pytest.mark.parametrize("command", ["recover", "run"])
@pytest.mark.parametrize("branch", ["counted before", "counted by its own write"])
def test_the_real_send_at_the_cap_reaches_the_root_owner_whatever_profile_is_sticky(
    root, monkeypatch, capsys, sent, command, branch
):
    """Both caps that record failed, with the send child really started, offline, and a named
    profile made sticky: the line lands on the root home's chat with the root's token."""
    from hermes_cli import release_cmd

    logged, _ = _two_homes(root, monkeypatch)
    batch_id = _capped(monkeypatch, branch)
    host = _midway(NEW, PREV)
    monkeypatch.setattr(release_cmd, "_tell_owner", sent.real)

    code = _command(monkeypatch, host, command, str(batch_id))

    out = capsys.readouterr().out
    capped = release_cmd._CAPPED.format(batch_id, release_cmd._RECOVERY_ATTEMPTS)
    assert (_batch(batch_id)["outcome"], code, host.calls) == ("failed", 0, [])
    assert out.splitlines().count(capped) == 1
    [told] = logged()
    assert told["chat_id"] == "111111", f"expected 111111, got {told['chat_id']}"
    assert told["token"] == SECRET
    failed = release_cmd._OUTCOMES["failed"]
    assert told["message"] == f"Release {batch_id} {failed}. The platform stays paused."


def test_a_batch_already_failed_at_the_cap_tells_the_owner_nothing(root, monkeypatch, sent):
    batch_id = _stopped_midway()
    with contextlib.closing(ledger.connect()) as conn:
        for _attempt in range(3):
            ledger.begin_recovery(conn, batch_id)
            ledger.finish_release(conn, batch_id, outcome="failed")

    assert _command(monkeypatch, _midway(NEW, PREV), "run", str(batch_id)) == 0

    assert sent == []


@pytest.mark.parametrize("command", ["recover", "run"])
@pytest.mark.parametrize("how", list(RECOVERY_CAUSES))
@pytest.mark.parametrize("outcome", ["released", "failed"])
def test_a_send_that_fails_on_a_recovery_changes_nothing_else_and_names_its_cause(
    root, monkeypatch, capsys, sent, command, outcome, how
):
    from hermes_cli import release_cmd

    batch_id, told = _stopped_midway(), []
    _snapshots(root, f"config/{NEW}")
    monkeypatch.setattr(release_cmd, "_tell_owner", sent.real)
    _send_fails(monkeypatch, how, told)

    try:
        code = _command(monkeypatch, ENDS[outcome](), command, str(batch_id))
    except KeyboardInterrupt:
        pytest.fail("the interrupt ended the recovery")

    shown = capsys.readouterr()
    lines = shown.out.splitlines()
    assert _batch(batch_id)["outcome"] == outcome
    assert code == (0 if outcome == "released" else 1)
    assert lines.count(f"Batch {batch_id} {release_cmd._OUTCOMES[outcome]}.") == 1
    named = [line for line in lines if all(part in line for part in RECOVERY_CAUSES[how])]
    assert len(named) == 1
    assert len(told) == 1 and SECRET not in shown.out + shown.err
    assert (_sentinel() is None) == (outcome == "released")


@pytest.mark.parametrize("command", ["recover", "run"])
@pytest.mark.parametrize("branch", ["counted before", "counted by its own write"])
@pytest.mark.parametrize("how", list(RECOVERY_CAUSES))
def test_a_send_that_fails_at_the_cap_changes_nothing_else_and_names_its_cause(
    root, monkeypatch, capsys, sent, command, branch, how
):
    from hermes_cli import release_cmd

    batch_id, told = _capped(monkeypatch, branch), []
    before, host = _sentinel(), _midway(NEW, PREV)
    monkeypatch.setattr(release_cmd, "_tell_owner", sent.real)
    _send_fails(monkeypatch, how, told)

    try:
        code = _command(monkeypatch, host, command, str(batch_id))
    except KeyboardInterrupt:
        pytest.fail("the interrupt ended the recovery")

    shown = capsys.readouterr()
    lines = shown.out.splitlines()
    capped = release_cmd._CAPPED.format(batch_id, release_cmd._RECOVERY_ATTEMPTS)
    assert (_batch(batch_id)["outcome"], code, host.calls) == ("failed", 0, [])
    assert lines.count(capped) == 1
    named = [line for line in lines if all(part in line for part in RECOVERY_CAUSES[how])]
    assert len(named) == 1
    assert _sentinel() == before
    assert len(told) == 1 and SECRET not in shown.out + shown.err


def test_recover_restores_environment(root, monkeypatch):
    """A release that stopped midway after its sync comes back to PREV's packages: recovery moves
    back the environment copy the release took, named after NEW, before the units start."""
    batch_id = _stopped_midway()
    _snapshots(root, f"config/{NEW}")
    host = _midway(NEW)  # NEW's packages are installed and NEW fails its health check
    host.environment_snapshot = ""  # only recovery names the copy it moves back

    _command(monkeypatch, host, "recover", str(batch_id))

    assert host.environment_snapshot == NEW
    assert host.calls == RESTORE
    assert (host.head, host.environment, host.running) == (PREV, PREV, PREV)
    assert _batch(batch_id)["outcome"] == "restored"


HELD = (
    "R1 held, R2 held, R3 held, R4 held, R5 held, R6 held, fleet version held,"
    " configuration held, environment held"
)


def _readbacks(out: str) -> list[str]:
    """The lines a recovery printed for its readbacks, in order."""
    return [line for line in out.splitlines() if line.startswith("Readback ")]


@pytest.mark.parametrize("command", ["recover", "run"])
def test_a_recovery_that_restores_prints_each_readback_in_order(
    root, monkeypatch, capsys, command
):
    batch_id = _stopped_midway()
    _snapshots(root, f"config/{NEW}")

    _command(monkeypatch, _midway(NEW), command, str(batch_id))

    out = capsys.readouterr().out
    first, last = _readbacks(out)
    # The host as recovery found it, at NEW, which did not hold; then PREV, after the restore.
    assert first.startswith(f"Readback 1 of 2, expecting {NEW}: failed (R1 held, R2 held, ")
    assert first.endswith(", environment held).")
    assert last == f"Readback 2 of 2, expecting {PREV}: held ({HELD})."
    assert _printed_outcome(out, batch_id).startswith("was rolled back")


@pytest.mark.parametrize("command", ["recover", "run"])
def test_a_recovery_that_keeps_new_prints_its_readback(root, monkeypatch, capsys, command):
    batch_id = _stopped_midway()
    _snapshots(root, f"config/{NEW}")

    _command(monkeypatch, FakeHost(), command, str(batch_id))

    out = capsys.readouterr().out
    assert _readbacks(out) == [f"Readback 1 of 1, expecting {NEW}: held ({HELD})."]
    assert _printed_outcome(out, batch_id) == "was released."
