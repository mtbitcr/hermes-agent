"""Card 7: a release that stopped midway is brought back to exactly PREV or exactly NEW, read back,
without a person, and never moved forward. Each test runs on FakeHost, a release host in memory
built like the merged one; the command's tests under a temporary root home."""

from __future__ import annotations

import argparse
import ast
import contextlib
import hashlib
import inspect
import sys
import textwrap
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
    loads from the checkout it moves; neither recover function has an import of its own."""
    from hermes_cli import release_cmd

    batch_id, host = _stopped_midway(), _midway(NEW)

    assert _command(monkeypatch, host, "recover", str(batch_id)) == 1
    assert host.calls == RESTORE and _sentinel() is None
    assert set(sys.modules) - host.modules_at_stop == set()
    for function in (release_runner.recover, release_cmd.recover):
        tree = ast.parse(textwrap.dedent(inspect.getsource(function)))
        assert not [node for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom))]
