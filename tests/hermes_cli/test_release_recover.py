"""Card 7: a release that stopped midway is brought back to exactly PREV or exactly NEW, read back,
without a person, and never moved forward. Each test runs on FakeHost, a release host in memory
built like the merged one; the command's tests under a temporary root home."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

import pytest
import yaml

from agent import estop
from hermes_cli import release_ledger as ledger
from hermes_cli import release_runner
from hermes_cli.release_guards import Pins
from hermes_cli.release_runner import PLATFORM_UNITS, SANDBOX_TUNNEL_UNIT
from hermes_cli.subcommands.release import build_release_parser


def _commit(label: str) -> str:
    return hashlib.sha1(label.encode()).hexdigest()


PREV, NEW, FEATURE, TREE = (_commit(label) for label in ("prev", "new", "feature", "tree"))
PINS = Pins(new=NEW, prev=PREV)
CHECKOUT = "/srv/hermes/checkout"
CONFIG = {"config.yaml": "digest"}
SETTINGS = {
    "units": {"sandbox-tunnel": "hermes-sandbox-tunnel"},
    "health_url": "http://127.0.0.1:8642/health",
    "workspace_check_url": "http://127.0.0.1:8650/api/projects",
}
OWNER = "owner: hold everything"
TOKEN = "placeholder-release-token"
# The merged restore steps: the units stop, the checkout goes to PREV, its configuration comes
# back, and the units start.
RESTORE = ["stop", f"checkout {PREV}", "restore config", "start"]
_REPO = Path(__file__).resolve().parents[2]
# ``hermes release COMMAND BATCH`` in its own process, as _command runs it, on a host where
# neither version reads back; its last line of output is its host's calls. Given "held", it says
# ready as it is about to count its recovery attempt, and goes on once its input ends.
_PROCESS = (
    "import json, sys\n"
    "import pytest\n"
    "sys.path.insert(0, sys.argv.pop(1))\n"
    "import test_release_recover as t\n"
    "begin, host = t.ledger.begin_recovery, t._midway(t.NEW, t.PREV)\n"
    "def held(*args):\n"
    "    print('ready', flush=True)\n"
    "    sys.stdin.read()\n"
    "    return begin(*args)\n"
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
    check. Each action is noted in ``calls``, and the modules at the first unit stop are kept."""

    head: str = NEW
    running: str | None = NEW
    active: set[str] = field(default_factory=lambda: {*PLATFORM_UNITS, SANDBOX_TUNNEL_UNIT})
    unhealthy: set[str] = field(default_factory=set)
    calls: list[str] = field(default_factory=list)
    modules_at_stop: set[str] | None = None
    config_snapshot: str = ""  # as in the live actions: the snapshot the restore puts back

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

    named_config_snapshot = live_config

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


def _midway(*unhealthy: str) -> FakeHost:
    """A host the release left midway: its units stopped, its checkout at NEW."""
    return FakeHost(running=None, active={SANDBOX_TUNNEL_UNIT}, unhealthy=set(unhealthy))


@pytest.fixture
def root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """The root Hermes home at ``<tmp>/.hermes``, with the release settings this host needs."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    (home / "config.yaml").write_text(yaml.safe_dump({"release": SETTINGS}))
    return home


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


def _process(root: Path, command: str, batch_id: int, held: bool = False) -> subprocess.Popen:
    """``hermes release COMMAND BATCH`` in its own process (see _PROCESS), on the root home
    ``root`` and its release record."""
    return subprocess.Popen(
        [sys.executable, "-c", _PROCESS, str(Path(__file__).parent), "held" if held else "free",
         command, str(batch_id)],
        cwd=_REPO,
        env={**os.environ, "HOME": str(root.parent), "HERMES_HOME": str(root),
             "PYTHONPATH": str(_REPO), "PYTHONDONTWRITEBYTECODE": "1"},
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
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


@pytest.mark.parametrize(
    ("host", "pause", "outcome", "code"),
    [(FakeHost, "own", "released", 0), (_midway, "own", "restored", 1),
     (FakeHost, "owner", "released", 0)],
    ids=["new-reads-back", "prev-restored", "owner-pause-stays"],
)
def test_run_on_a_releasing_batch_recovers_and_resumes(
    root, monkeypatch, capsys, host, pause, outcome, code
):
    """The release unit's ``release run BATCH`` on a releasing batch goes straight to recovery
    with the stored versions, records the outcome and lifts the release's own pause only."""
    batch_id, host = _stopped_midway(pause), host()
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

    assert _command(monkeypatch, host, "recover", str(batch_id)) == 1
    assert host.calls == RESTORE and _sentinel() is None
    assert set(sys.modules) - host.modules_at_stop == set()


@pytest.mark.parametrize(("first", "second"), [("run", "recover"), ("recover", "run")])
def test_two_processes_never_count_past_three(root, first, second):
    """Owner rule 443, test 1 (finding 1 of the card review). On a temporary record at attempt
    count two, ``first`` waits in its own process as it is about to count its attempt, and
    ``second`` starts in another: it refuses at once and asks its host nothing. The first makes
    the third attempt, and the count never exceeds three."""
    batch_id = _at_attempt_two()

    with _process(root, first, batch_id, held=True) as waiting:
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

    with _process(root, "recover", batch_id, held=True) as waiting:
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
