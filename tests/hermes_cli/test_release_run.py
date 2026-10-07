"""Card 5 of the release re-cut: ``hermes release run``, one release between a pause and a resume.

Every test but the K6 one runs against FakeHost, an in-memory release host standing in for both live
adapters under a temporary root home, so no real host command runs. Its journal, with the record's
writes, the pause, the resume and the runner noted by ``_run``, shows the order of the steps. The
K6 test uses the live adapter classes, with the process boundary faked.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from agent import estop
from hermes_cli import build_info, release_guards, release_host_actions
from hermes_cli import release_ledger as ledger
from hermes_cli.release_guards import OpenRun


def _commit(label: str) -> str:
    """A 40-hex id shaped like a git commit."""
    return hashlib.sha1(label.encode()).hexdigest()


PREV, NEW, FEATURE, TREE = (_commit(label) for label in ("prev", "new", "feature", "tree"))
CHECKOUT = "/srv/hermes/checkout"
CONFIG = {"config.yaml": "digest"}
SETTINGS = {
    "units": {"sandbox-tunnel": "hermes-sandbox-tunnel"},
    "health_url": "http://127.0.0.1:8642/health",
    "workspace_check_url": "http://127.0.0.1:8650/api/projects",
    "drain_poll_seconds": 45,
}
UNITS = ("hermes-gateway", "hermes-serve", "hermes-sandbox-tunnel")
FORWARD_ONLY = ("guards", "stop", "snapshot", "forward", "readback", "start", "readback")
TIER_2 = FORWARD_ONLY[:5] + ("back", "readback", "forward", "readback") + FORWARD_ONLY[5:]


@dataclass
class Clock:
    """Stands in for the time module: a wait takes no time and moves the clock on by a day. A wait
    that never ends fails the run instead of hanging the test."""

    sleeps: list[float] = field(default_factory=list)

    def sleep(self, seconds: float) -> None:
        assert len(self.sleeps) < 5000, "the drain never ended"
        self.sleeps.append(seconds)

    def monotonic(self) -> float:
        return 86400.0 * len(self.sleeps)

    time = monotonic


@dataclass
class FakeHost:
    """A healthy release host in memory, at PREV, on which every guard passes for NEW.

    The journal notes the fetch, the configuration snapshot, each pass of the guards (at G1's read)
    and each G10 read, but nothing while the runner runs (``quiet``). The first ``open_reads`` G10
    reads see a live run, and ``on_g10`` gets the number of each. A health check fails on a version
    in ``unhealthy``; with ``read_raises``, G5's read raises once the platform is paused. Every
    other action succeeds and every other readback holds.
    """

    head: str = PREV
    origin: str = NEW
    open_reads: int = 0
    on_g10: Callable[[int], object] | None = None
    unhealthy: set[str] = field(default_factory=set)
    read_raises: bool = False
    snapshot_dirs: list[str] = field(default_factory=list)
    journal: list[str] = field(default_factory=list)
    results: list = field(default_factory=list)
    clock: Clock = field(default_factory=Clock)
    quiet: bool = False
    g10_reads: int = 0
    stops: int = 0

    def build(self, **settings):
        return self

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        return lambda *args: True

    def checkout_head(self):
        return self.head

    def checkout_is_clean(self):
        if not self.quiet:
            self.journal.append("preflight")
        return True

    def origin_main(self):
        return self.origin

    def first_parent_chain(self, prev, new):
        return [new]

    def commit_parents(self, commit):
        return (PREV, FEATURE)

    def commit_tree(self, commit):
        return TREE

    def is_ancestor(self, ancestor, descendant):
        # As in git, a commit is its own ancestor.
        return ancestor == descendant or (ancestor, descendant) == (PREV, NEW)

    def changed_paths(self, prev, new):
        if self.read_raises and estop.is_engaged():
            raise RuntimeError("the diff could not be read")
        return {"hermes_cli/main.py"}

    def checkout_root(self):
        return CHECKOUT

    venv_import_root = checkout_root

    def unit_property(self, unit, name):
        return {"KillSignal": "SIGINT", "KillMode": "mixed"}[name]

    def free_disk_bytes(self):
        return 5 * 1024**3

    def snapshots(self):
        return list(self.snapshot_dirs)

    def open_native_runs(self):
        if self.quiet:
            return []
        self.g10_reads += 1
        self.journal.append("G10")
        if self.on_g10 is not None:
            self.on_g10(self.g10_reads)
        return [OpenRun("default:7", True)] if self.g10_reads <= self.open_reads else []

    def live_config(self):
        return dict(CONFIG)

    named_config_snapshot = live_config

    def fetch(self, commit):
        self.journal.append("fetch")

    def save_config_snapshot(self, name):
        self.journal.append("configuration snapshot")

    def stop_units(self, units):
        self.stops += 1

    def checkout(self, commit):
        self.head = commit

    def take_snapshot(self, name):
        self.snapshot_dirs.append(name)

    def health_ok(self):
        return self.head not in self.unhealthy

    def fleet_version(self):
        return self.head


@dataclass
class Boundary:
    """The live host's processes, faked: git in the checkout, the user manager and the board
    reader's child. The checkout and the units move as the live adapters ask, a gateway that
    starts stamps its record with the checkout's head, and the modules loaded at the first stop
    are kept."""

    root: Path
    head: str = PREV
    active: set[str] = field(default_factory=lambda: set(UNITS))
    modules_at_stop: set[str] | None = None

    def run(self, argv, **options):
        if argv[0] == sys.executable:  # the board reader's child: no board has an open run
            return self._done(json.dumps([[] for _path in argv[6:]]))
        if argv[0] == "systemctl":
            return self._systemctl(*argv[2:])
        skipped = ("--no-optional-locks", "--end-of-options")
        args = tuple(arg for arg in argv[3:] if arg not in skipped)
        if args[:2] == ("merge-base", "--is-ancestor"):
            return self._done("", 0 if args[2] == args[3] or args[2:] == (PREV, NEW) else 1)
        if args[0] == "checkout":
            self.head = args[-1]
        return self._done({
            ("rev-parse", "--verify", "HEAD"): self.head,
            ("status", "--porcelain", "--untracked-files=normal"): "",
            ("ls-remote", "origin", "refs/heads/main"): f"{NEW}\trefs/heads/main\n",
            ("rev-list", "--first-parent", NEW, f"^{PREV}"): NEW,
            ("rev-list", "--parents", "--no-walk", NEW): f"{NEW} {PREV} {FEATURE}",
            ("rev-parse", "--verify", f"{NEW}^{{tree}}"): TREE,
            ("diff-tree", "-r", "--name-only", "--no-renames", "-z", PREV, NEW): "cli.py\0",
            ("rev-parse", "--show-toplevel"): CHECKOUT,
            ("fetch", "--no-tags", "origin", NEW): "",
            ("checkout", "--detach", "--no-overwrite-ignore", NEW): "",
        }[args])

    def _systemctl(self, verb, *rest):
        if verb == "show":  # --property=NAME -- UNIT
            name = rest[0].partition("=")[2]
            value = {"KillSignal": "2", "KillMode": "mixed"}[name]
            return self._done(f"{name}={value}\n")
        if verb == "is-active":
            return self._done("active\n") if rest[0] in self.active else self._done("inactive\n", 3)
        if verb == "stop":
            if self.modules_at_stop is None:
                self.modules_at_stop = set(sys.modules)
            self.active.discard(rest[0])
        else:
            self.active.add(rest[0])
            if rest[0] == UNITS[0]:
                self.stamp()
        return self._done("")

    def stamp(self):
        """The gateway's own record, stamped with the code it runs: the checkout's head."""
        record = {"pid": os.getpid(), "gateway_state": "running", "code_sha": self.head}
        (self.root / "gateway_state.json").write_text(json.dumps(record))

    @staticmethod
    def _done(stdout, returncode=0):
        return subprocess.CompletedProcess([], returncode, stdout, "")


@pytest.fixture
def root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """The root Hermes home at ``<tmp>/.hermes``, with the release settings this host needs."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    (home / "config.yaml").write_text(yaml.safe_dump({"release": SETTINGS}))
    return home


def _accepted(*tiers: object) -> dict:
    """Record one change per tier, NEW the last, and accept the batch they joined."""
    with contextlib.closing(ledger.connect()) as conn:
        for number, tier in enumerate(tiers, start=1):
            commit = NEW if number == len(tiers) else _commit(f"change {number}")
            batch = ledger.record_merge(
                conn,
                merge_commit=commit,
                pr_url=f"https://github.com/mtbitcr/hermes-agent/pull/{number}",
                reviewed_base=PREV,
                reviewed_head=FEATURE,
                reviewed_tree=TREE,
                tier=tier,
                card_id=f"t_{commit[:8]}",
                on_main=True,
            )["batch"]
        return ledger.decide_release(
            conn,
            batch["batch_id"],
            decision="accepted",
            shown_digest=batch["digest"],
            expected_version=batch["version"],
            decision_ref="decisions-page:release",
        )


def _batches() -> list[dict]:
    with contextlib.closing(ledger.connect()) as conn:
        return ledger.list_batches(conn)


def _run(monkeypatch: pytest.MonkeyPatch, host: FakeHost) -> int:
    """``hermes release run`` with ``host`` built in place of both live adapters and its clock in
    place of the time module. The record's writes, the pause, the resume and the runner are noted
    in the host's journal too, and the runner's result is kept in ``host.results``."""
    from hermes_cli import release_cmd

    note = host.journal.append
    monkeypatch.setattr(release_cmd, "LiveHostReader", host.build)
    monkeypatch.setattr(release_cmd, "ReleaseHostActions", host.build)
    monkeypatch.setattr(release_cmd, "time", host.clock)
    _spy(monkeypatch, ledger, "begin_release", lambda *args, **versions: note("begin"))
    _spy(monkeypatch, ledger, "finish_release", lambda *args, outcome: note(f"outcome {outcome}"))
    _spy(
        monkeypatch, estop, "engage", lambda why: note(f"pause {why!r} at {estop.sentinel_path()}")
    )
    _spy(monkeypatch, estop, "disengage", lambda: note("resume"))
    runner = release_cmd.run_release

    def run_release(*args):
        note("runner")
        host.quiet = True
        try:
            host.results.append(runner(*args))
        finally:
            host.quiet = False
        return host.results[-1]

    monkeypatch.setattr(release_cmd, "run_release", run_release)
    return release_cmd.cmd_release(argparse.Namespace(release_command="run"))


def _spy(monkeypatch: pytest.MonkeyPatch, owner: object, name: str, note: Callable) -> None:
    """Note each call of ``owner.name`` with ``note``, then make it."""
    real = getattr(owner, name)

    def spy(*args, **kwargs):
        note(*args, **kwargs)
        return real(*args, **kwargs)

    monkeypatch.setattr(owner, name, spy)


def _stop_signal() -> None:
    """What ``systemctl --user stop hermes-release-BATCH`` sends the run (S-F). Only a handler the
    run set may get it: the default one would end the test process."""
    assert callable(signal.getsignal(signal.SIGTERM)), "the run set no handler for its stop"
    signal.raise_signal(signal.SIGTERM)


def test_run_pauses_waits_for_the_drain_then_releases_and_resumes(root, monkeypatch, capsys):
    batch_id = _accepted(1)["batch_id"]
    host = FakeHost(open_reads=3)  # one live run, open at the preflight and for two polls

    assert _run(monkeypatch, host) == 0

    assert host.journal == [
        "begin", "fetch", "configuration snapshot",
        "preflight", "G10",  # the preflight asks every guard, G10 among them
        f"pause 'release {batch_id}' at {root / 'ESTOP'}",
        "G10", "G10", "G10",
        "runner", "outcome released", "resume",
    ]
    assert host.clock.sleeps == [SETTINGS["drain_poll_seconds"]] * 2
    [batch] = _batches()
    assert (batch["state"], batch["prev"], batch["new"]) == ("released", PREV, NEW)
    assert not estop.is_engaged()
    assert capsys.readouterr().out.endswith(f"Batch {batch_id} was released.\n")


def test_drain_uses_the_merged_guard_and_no_time_limit(root, monkeypatch):
    _accepted(1)
    host = FakeHost()

    def g10(reader, pins, merges):  # the merged G10, on which a live run stays for 1000 polls
        if not host.quiet:
            host.journal.append("merged G10")
        return len(host.clock.sleeps) >= 1000

    guards = tuple(
        (guard, rule, g10 if guard == "G10" else check)
        for guard, rule, check in release_guards.GUARDS
    )
    monkeypatch.setattr(release_guards, "GUARDS", guards)

    assert _run(monkeypatch, host) == 0

    paused = next(index for index, entry in enumerate(host.journal) if entry.startswith("pause"))
    assert host.journal[paused + 1 : host.journal.index("runner")] == ["merged G10"] * 1001
    # A thousand waits of the poll interval: the clock is a thousand days on, and no deadline came.
    assert host.clock.sleeps == [SETTINGS["drain_poll_seconds"]] * 1000


@pytest.mark.parametrize(
    ("tiers", "steps"),
    [
        pytest.param((0, 2), TIER_2, id="tiers-0-and-2"),
        pytest.param((0, None), TIER_2, id="tier-not-recorded"),
        pytest.param((0, 1), FORWARD_ONLY, id="tiers-0-and-1"),
    ],
)
def test_run_reads_the_tier_from_the_batch(root, monkeypatch, tiers, steps):
    _accepted(*tiers)
    host = FakeHost()

    assert _run(monkeypatch, host) == 0

    assert [result.steps for result in host.results] == [steps]


def test_an_existing_pause_refuses_and_is_not_replaced(root, monkeypatch, capsys):
    _accepted(1)
    estop.engage("owner: hold everything")
    sentinel = (root / "ESTOP").read_bytes()
    host = FakeHost()

    assert _run(monkeypatch, host) == 1

    assert host.journal == [
        "begin", "fetch", "configuration snapshot", "preflight", "G10", "outcome refused"
    ]
    assert (root / "ESTOP").read_bytes() == sentinel
    assert "Refused: the platform is already paused." in capsys.readouterr().out


def test_failed_preflight_refuses_without_pausing(root, monkeypatch, capsys):
    _accepted(1)
    host = FakeHost(origin=PREV)  # NEW is not on the history of origin main (the live G2)

    assert _run(monkeypatch, host) == 1

    assert host.journal == [
        "begin", "fetch", "configuration snapshot", "preflight", "G10", "outcome refused"
    ]
    assert not (root / "ESTOP").exists()
    [folded, waiting] = _batches()
    assert (folded["outcome"], waiting["state"]) == ("refused", "open")
    assert [member["merge_commit"] for member in waiting["members"]] == [NEW]
    out = capsys.readouterr().out
    assert "G2 failed: NEW is on the history of origin main" in out
    assert "Refused: release guard G2 failed." in out


def test_guard_read_that_raises_is_refused_and_resumes(root, monkeypatch, capsys):
    _accepted(1)
    host = FakeHost(read_raises=True)

    assert _run(monkeypatch, host) == 1

    assert host.journal[-3:] == ["runner", "outcome refused", "resume"]
    # The read escaped the runner from its fresh guards (F5): nothing stopped, nothing moved.
    assert (host.stops, host.head, host.results) == (0, PREV, [])
    assert not estop.is_engaged()
    assert "RuntimeError: the diff could not be read" in capsys.readouterr().out
    assert [batch["outcome"] for batch in _batches()] == ["refused", None]


@pytest.mark.parametrize(
    ("unhealthy", "outcome", "paused"),
    [
        pytest.param({NEW}, "restored", False, id="restored"),
        pytest.param({NEW, PREV}, "failed", True, id="failed"),
    ],
)
def test_restored_resumes_and_failed_stays_paused(root, monkeypatch, unhealthy, outcome, paused):
    batch_id = _accepted(1)["batch_id"]
    host = FakeHost(unhealthy=unhealthy)

    assert _run(monkeypatch, host) == 1

    assert [result.outcome for result in host.results] == [outcome]
    assert _batches()[0]["outcome"] == outcome
    assert ("resume" in host.journal) is not paused
    assert (estop.get_state() or {}).get("reason") == (f"release {batch_id}" if paused else None)


def test_stop_signal_during_the_drain_resumes(root, monkeypatch, capsys):
    batch_id = _accepted(1)["batch_id"]
    handler = signal.getsignal(signal.SIGTERM)
    host = FakeHost(open_reads=3, on_g10=lambda read: read == 2 and _stop_signal())

    assert _run(monkeypatch, host) == 1

    assert host.journal == [
        "begin", "fetch", "configuration snapshot", "preflight", "G10",
        f"pause 'release {batch_id}' at {root / 'ESTOP'}", "G10", "outcome refused", "resume",
    ]
    assert "Refused: stopped before cutover." in capsys.readouterr().out
    assert not estop.is_engaged()
    assert signal.getsignal(signal.SIGTERM) == handler  # the run's own handler is gone again


def test_owner_pause_during_the_run_is_left_alone(root, monkeypatch):
    _accepted(1)
    host = FakeHost(
        open_reads=3, on_g10=lambda read: read == 2 and estop.engage("owner: hold everything")
    )

    assert _run(monkeypatch, host) == 0

    assert host.journal[-2:] == ["runner", "outcome released"]
    assert estop.get_state()["reason"] == "owner: hold everything"


def test_nothing_is_imported_after_the_units_stop(root, monkeypatch):
    from hermes_cli import release_cmd

    _accepted(1)
    boundary = Boundary(root)
    boundary.stamp()  # the live gateway runs PREV
    answer = SimpleNamespace(status=200)  # each readback address answers
    direct = SimpleNamespace(open=lambda url, timeout: contextlib.nullcontext(answer))
    monkeypatch.setattr(subprocess, "run", boundary.run)
    monkeypatch.setattr(
        build_info, "get_code_identity", lambda refresh=False: {"sha": boundary.head}
    )
    monkeypatch.setattr(release_host_actions, "_DIRECT", direct)
    monkeypatch.setattr(shutil, "disk_usage", lambda path: SimpleNamespace(free=8 * 1024**3))
    monkeypatch.setattr(release_cmd, "time", Clock())

    assert release_cmd.cmd_release(argparse.Namespace(release_command="run")) == 0

    assert boundary.head == NEW and boundary.modules_at_stop is not None
    assert set(sys.modules) - boundary.modules_at_stop == set()


def test_run_is_registered(monkeypatch):
    from hermes_cli import main, release_cmd

    monkeypatch.setattr(main, "_plugin_cli_discovery_needed", lambda: False)
    parser, _subparsers = main._build_cli_parser()
    monkeypatch.setattr(release_cmd, "run", lambda: 7)

    args = parser.parse_args(["release", "run"])

    assert args.release_command == "run"
    assert args.func(args) == 7
