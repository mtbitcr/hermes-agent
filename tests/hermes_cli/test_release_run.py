"""Card 5 of the release re-cut: ``hermes release run``, one release between a pause and a resume.

Each test parses the registered command before it patches anything, then runs it on FakeHost under
a temporary root home; the K6 test fakes only the process boundary, under the live adapters."""

from __future__ import annotations

import argparse
import contextlib
import errno
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from agent import estop
from hermes_cli import build_info, release_guards, release_host_actions
from hermes_cli import release_ledger as ledger
from hermes_cli.release_guards import GUARDS, OpenRun
from hermes_cli.subcommands.release import build_release_parser


def _commit(label: str) -> str:
    return hashlib.sha1(label.encode()).hexdigest()


PREV, NEW, FEATURE, TREE = (_commit(label) for label in ("prev", "new", "feature", "tree"))
CHECKOUT = "/srv/hermes/checkout"
CONFIG = {"config.yaml": "digest"}
SETTINGS = {
    "units": {"sandbox-tunnel": "hermes-sandbox-tunnel"}, "drain_poll_seconds": 45,
    "health_url": "http://127.0.0.1:8642/health",
    "workspace_check_url": "http://127.0.0.1:8650/api/projects",
}
UNITS = ("hermes-gateway", "hermes-serve", "hermes-sandbox-tunnel")
FORWARD_ONLY = ("guards", "stop", "snapshot", "forward", "readback", "start", "readback")
TIER_2 = FORWARD_ONLY[:5] + ("back", "readback", "forward", "readback") + FORWARD_ONLY[5:]
OWNER = "owner: hold everything"
# The steps before the pause; the preflight asks every guard, G10 among them.
BEFORE_PAUSE = ["begin", "fetch", "configuration snapshot", "preflight", "G10"]
# FakeHost's answers to the reads that never change: every guard passes for NEW on PREV.
ANSWERS = {
    "first_parent_chain": [NEW], "commit_parents": (PREV, FEATURE), "commit_tree": TREE,
    "checkout_root": CHECKOUT, "venv_import_root": CHECKOUT, "free_disk_bytes": 5 * 1024**3,
    "live_config": CONFIG, "named_config_snapshot": CONFIG, "snapshots": [],
}


class Clock(list):
    """The waits made. In the time module, a wait takes no time and moves the clock on a day."""

    def sleep(self, seconds):
        assert len(self) < 5000, "the drain never ended"
        self.append(seconds)

    def now(self):
        return 86400.0 * len(self)

    def install(self, monkeypatch):
        for name, value in (("sleep", self.sleep), ("monotonic", self.now), ("time", self.now)):
            monkeypatch.setattr(time, name, value)


@dataclass
class FakeHost:
    """A healthy release host at PREV. Its journal notes the fetch, the configuration snapshot,
    each guard pass (at the clean check), each G10 read and, before them, each new pause. ``at``
    maps the n-th note of an entry to a hook run after it, into ``injected``. The first
    ``open_reads`` G10 reads see a live run. Other reads get ANSWERS; other actions succeed."""

    head: str = PREV
    origin: str = NEW
    open_reads: int = 0
    at: dict = field(default_factory=dict)
    unhealthy: set = field(default_factory=set)
    read_raises: bool = False
    journal: list = field(default_factory=list)
    injected: list = field(default_factory=list)
    results: list = field(default_factory=list)
    clock: Clock = field(default_factory=Clock)
    pause: bytes | None = None
    stops: int = 0

    def note(self, entry):
        pause = _sentinel()
        if pause is not None and pause != self.pause:  # an empty pause has no reason
            reason = json.loads(pause or "{}").get("reason")
            self.journal.append(f"pause {reason!r} at {estop.sentinel_path()}")
        self.pause = pause
        self.journal.append(entry)
        hook = self.at.pop((entry, self.journal.count(entry)), None)
        if hook is not None:
            self.injected.append(hook())

    def __getattr__(self, name):
        return lambda *args: ANSWERS.get(name, True)

    def checkout_head(self):
        return self.head

    fleet_version = checkout_head

    def checkout_is_clean(self):
        self.note("preflight")
        return True

    def origin_main(self):
        return self.origin

    def is_ancestor(self, ancestor, descendant):  # as in git, a commit is its own ancestor
        return ancestor == descendant or (ancestor, descendant) == (PREV, NEW)

    def changed_paths(self, prev, new):
        if self.read_raises and estop.is_engaged():
            raise RuntimeError("the diff could not be read")
        return {"hermes_cli/main.py"}

    def unit_property(self, unit, name):
        return {"KillSignal": "SIGINT", "KillMode": "mixed"}[name]

    def open_native_runs(self):
        self.note("G10")
        return [OpenRun("default:7", True)] if self.journal.count("G10") <= self.open_reads else []

    def fetch(self, commit):
        self.note("fetch")

    def save_config_snapshot(self, name):
        self.note("configuration snapshot")

    def stop_units(self, units):
        self.stops += 1

    def checkout(self, commit):
        self.head = commit

    def health_ok(self):
        return self.head not in self.unhealthy


@dataclass
class Boundary:
    """The live host's processes, faked: git in the checkout, the user manager and the board
    reader's child. The checkout and the units move as the live adapters ask, a starting gateway
    stamps its record with the checkout's head, and the modules at the first stop are kept."""

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

    def stamp(self):  # the gateway's own record, stamped with the code it runs: the checkout's head
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
                conn, merge_commit=commit,
                pr_url=f"https://github.com/mtbitcr/hermes-agent/pull/{number}",
                reviewed_base=PREV, reviewed_head=FEATURE, reviewed_tree=TREE, tier=tier,
                card_id=f"t_{commit[:8]}", on_main=True,
            )["batch"]
        return ledger.decide_release(
            conn, batch["batch_id"], decision="accepted", shown_digest=batch["digest"],
            expected_version=batch["version"], decision_ref="decisions-page:release",
        )


def _batches() -> list[dict]:
    with contextlib.closing(ledger.connect()) as conn:
        return ledger.list_batches(conn)


def _sentinel() -> bytes | None:
    """The pause sentinel's bytes, or None when there is none."""
    with contextlib.suppress(OSError):
        return estop.sentinel_path().read_bytes()
    return None


def _parsed_run() -> argparse.Namespace:
    """``hermes release run``, parsed by its own parser: on a base without it, this fails first."""
    parser = argparse.ArgumentParser(prog="hermes")
    build_release_parser(parser.add_subparsers())
    return parser.parse_args(["release", "run"])


def _run(monkeypatch: pytest.MonkeyPatch, host: FakeHost) -> int:
    """``hermes release run`` on ``host``, noting the record's writes, the runner and the resume."""
    from hermes_cli import release_cmd

    args = _parsed_run()
    for adapter in ("LiveHostReader", "ReleaseHostActions"):
        monkeypatch.setattr(release_cmd, adapter, lambda **settings: host)
    host.clock.install(monkeypatch)
    _spy(monkeypatch, ledger, "begin_release", lambda *args, **versions: host.note("begin"))
    _spy(monkeypatch, ledger, "finish_release", lambda *_, outcome: host.note(f"outcome {outcome}"))
    _spy(monkeypatch, estop, "disengage", lambda: host.note("resume"))
    host.results = _spy(monkeypatch, release_cmd, "run_release", lambda *args: host.note("runner"))
    host.pause = _sentinel()
    try:
        code = args.func(args)
    except KeyboardInterrupt:  # an interrupt the run lets through would end the whole session
        pytest.fail("an interrupt escaped the run")
    assert not host.at, "a hook was never reached"
    return code


def _spy(monkeypatch, owner, name, note):
    """Note each call of ``owner.name``, then make it; return the list of what it returned."""
    real, results = getattr(owner, name), []

    def spy(*args, **kwargs):
        note(*args, **kwargs)
        results.append(real(*args, **kwargs))
        return results[-1]

    monkeypatch.setattr(owner, name, spy)
    return results


def _stop_signal(signum: int = signal.SIGTERM):
    """A stop signal to the run; the default handler would end the test process instead."""
    assert callable(signal.getsignal(signum)), "the run set no handler for its stop"
    signal.raise_signal(signum)


def test_run_pauses_waits_for_the_drain_then_releases_and_resumes(root, monkeypatch, capsys):
    batch_id = _accepted(1)["batch_id"]
    host = FakeHost(open_reads=3, at={("runner", 1): _sentinel})  # one live run, for two polls

    assert _run(monkeypatch, host) == 0
    assert host.journal == BEFORE_PAUSE + [
        f"pause 'release {batch_id}' at {root / 'ESTOP'}", "G10", "G10", "G10",
        "runner", "preflight", "G10",  # the runner asks every guard again
        "outcome released", "resume",
    ]
    assert host.clock == [SETTINGS["drain_poll_seconds"]] * 2
    pause = json.loads(host.injected[0])  # the runner's pause: estop's payload and the token
    assert sorted(pause) == ["engaged_at", "reason", "release_token"]
    assert pause["reason"] == f"release {batch_id}"
    [batch] = _batches()
    assert (batch["state"], batch["prev"], batch["new"]) == ("released", PREV, NEW)
    assert _sentinel() is None
    assert capsys.readouterr().out.endswith(f"Batch {batch_id} was released.\n")


def test_drain_uses_the_merged_guard_and_no_time_limit(root, monkeypatch):
    _accepted(1)
    host = FakeHost()

    def g10(reader, pins, merges):  # the merged G10, on which a live run stays for 1000 polls
        host.note("merged G10")
        return len(host.clock) >= 1000

    guards = [(guard, rule, g10 if guard == "G10" else check) for guard, rule, check in GUARDS]
    monkeypatch.setattr(release_guards, "GUARDS", tuple(guards))

    assert _run(monkeypatch, host) == 0
    paused = next(index for index, entry in enumerate(host.journal) if entry.startswith("pause"))
    assert host.journal[paused + 1 : host.journal.index("runner")] == ["merged G10"] * 1001
    # A thousand waits of the poll interval: the clock is a thousand days on, and no deadline came.
    assert host.clock == [SETTINGS["drain_poll_seconds"]] * 1000


@pytest.mark.parametrize(
    ("tiers", "steps"),
    [((0, 2), TIER_2), ((0, None), TIER_2), ((0, 1), FORWARD_ONLY)],
    ids=["tiers-0-and-2", "tier-not-recorded", "tiers-0-and-1"],
)
def test_run_reads_the_tier_from_the_batch(root, monkeypatch, tiers, steps):
    _accepted(*tiers)
    host = FakeHost()

    assert _run(monkeypatch, host) == 0
    assert [result.steps for result in host.results] == [steps]


@pytest.mark.parametrize(
    ("cause", "refusal"),
    [("pause", "the platform is already paused"), ("guard", "release guard G2 failed")],
)
def test_refusal_before_the_pause_pauses_nothing(root, monkeypatch, capsys, cause, refusal):
    _accepted(1)
    sentinel = estop.engage(OWNER).read_bytes() if cause == "pause" else None
    host = FakeHost(origin=PREV if cause == "guard" else NEW)  # G2: NEW is not on origin main

    assert _run(monkeypatch, host) == 1
    assert (host.journal, _sentinel()) == (BEFORE_PAUSE + ["outcome refused"], sentinel)
    [folded, waiting] = _batches()
    assert (folded["outcome"], waiting["state"]) == ("refused", "open")
    assert [member["merge_commit"] for member in waiting["members"]] == [NEW]
    assert f"Refused: {refusal}." in capsys.readouterr().out


@pytest.mark.parametrize(
    ("case", "refusal"),
    [
        ("owner pause after the read", "the platform is already paused"),
        ("create fails", "stopped before cutover"),
        ("removed in the drain", "stopped before cutover"),
        ("owner pause in the drain", "stopped before cutover"),
    ],
)
def test_run_cuts_over_only_under_its_own_new_pause(root, monkeypatch, capsys, case, refusal):
    """Owner rule 1. An owner's pause written just after the run read that none was set
    (interleaving 1 of finding 1) refuses the run as one set before it does. A create that fails
    (finding 2), here for root too as the sentinel's folder is a file, or the run's pause removed
    or replaced by an owner's during the drain, ends the run before the cutover."""
    _accepted(1)
    host = FakeHost(open_reads=3)
    if case == "owner pause after the read":
        read = estop.is_engaged

        def is_engaged():
            engaged = read()
            if not host.injected:
                host.injected.append(estop.engage(OWNER).read_bytes())
            return engaged

        monkeypatch.setattr(estop, "is_engaged", is_engaged)
    elif case == "create fails":
        monkeypatch.setattr(estop, "sentinel_path", lambda: root / "config.yaml" / "ESTOP")
        with pytest.raises(NotADirectoryError):
            estop.sentinel_path().touch()
    elif case == "removed in the drain":
        host.at[("G10", 2)] = (root / "ESTOP").unlink
    else:
        host.at[("G10", 2)] = lambda: estop.engage(OWNER).read_bytes()

    assert _run(monkeypatch, host) == 1
    assert (host.stops, host.results) == (0, [])
    assert [batch["outcome"] for batch in _batches()] == ["refused", None]
    assert f"Refused: {refusal}." in capsys.readouterr().out
    if case.startswith("owner"):  # the owner's pause stays
        assert len(host.injected) == 1 and _sentinel() == host.injected[0]
    else:
        assert not estop.is_engaged()


def test_resume_lifts_only_the_pause_with_the_release_token(root, monkeypatch, capsys):
    """Owner rule 2, for interleaving 2 of finding 1. A pause written as the outcome is recorded
    stays, even with the release's own reason: only the release's pause carries its token."""
    reason = f"release {_accepted(1)['batch_id']}"
    host = FakeHost(at={("outcome released", 1): lambda: estop.engage(reason).read_bytes()})

    assert _run(monkeypatch, host) == 0
    assert "resume" not in host.journal
    assert len(host.injected) == 1 and _sentinel() == host.injected[0]
    assert capsys.readouterr().out.endswith(f"The platform stays paused ({reason}).\n")


def test_guard_read_that_raises_is_refused_and_resumes(root, monkeypatch, capsys):
    _accepted(1)
    host = FakeHost(read_raises=True)

    assert _run(monkeypatch, host) == 1
    assert host.journal[-4:] == ["runner", "preflight", "outcome refused", "resume"]
    # The read escaped the runner from its fresh guards (F5): nothing stopped, nothing moved.
    assert (host.stops, host.head, host.results, _sentinel()) == (0, PREV, [], None)
    assert "RuntimeError: the diff could not be read" in capsys.readouterr().out
    assert [batch["outcome"] for batch in _batches()] == ["refused", None]


@pytest.mark.parametrize(("unhealthy", "outcome"), [({NEW}, "restored"), ({NEW, PREV}, "failed")])
def test_restored_resumes_and_failed_stays_paused(root, monkeypatch, unhealthy, outcome):
    batch_id = _accepted(1)["batch_id"]
    host = FakeHost(unhealthy=unhealthy)

    assert _run(monkeypatch, host) == 1
    assert [result.outcome for result in host.results] == [outcome] == [_batches()[0]["outcome"]]
    paused = outcome == "failed"
    assert ("resume" in host.journal) is not paused
    assert (estop.get_state() or {}).get("reason") == (f"release {batch_id}" if paused else None)


def test_stop_signal_during_the_drain_resumes(root, monkeypatch, capsys):
    batch_id = _accepted(1)["batch_id"]
    handler = signal.getsignal(signal.SIGTERM)
    host = FakeHost(open_reads=3, at={("G10", 2): _stop_signal})

    assert _run(monkeypatch, host) == 1
    assert host.journal == BEFORE_PAUSE + [
        f"pause 'release {batch_id}' at {root / 'ESTOP'}", "G10", "outcome refused", "resume"
    ]
    assert "Refused: stopped before cutover." in capsys.readouterr().out
    assert (_sentinel(), signal.getsignal(signal.SIGTERM)) == (None, handler)  # its handler is gone


@pytest.mark.parametrize("case", ["owner pause", "write fails", "stop signal"])
def test_the_pause_is_published_whole_or_not_at_all(root, monkeypatch, capsys, case):
    """Owner rule 326, item 1 (finding 1 of the second review). An owner's pause written once the
    release has created its pause file, before the pause is published, refuses the run and stays
    as the owner wrote it. A payload write that fails there, or a stop signal, refuses the run and
    leaves no pause. No sentinel shows before the payload is whole, and no file of the release is
    left beside it."""
    _accepted(1)
    host, names, fdopen = FakeHost(), set(os.listdir(root)), os.fdopen

    def created(fd, *args, **kwargs):  # the release's pause file exists; its payload is not written
        monkeypatch.setattr(os, "fdopen", fdopen)
        host.injected.append(_sentinel())
        if case == "owner pause":
            host.injected.append(estop.engage(OWNER).read_bytes())
        elif case == "write fails":
            os.close(fd)
            raise OSError(errno.ENOSPC, os.strerror(errno.ENOSPC))
        else:
            _stop_signal()
        return fdopen(fd, *args, **kwargs)

    monkeypatch.setattr(os, "fdopen", created)

    assert _run(monkeypatch, host) == 1
    assert (host.stops, host.results, host.injected[0]) == (0, [], None)
    assert [batch["outcome"] for batch in _batches()] == ["refused", None]
    owner = case == "owner pause"
    assert _sentinel() == (host.injected[1] if owner else None)
    assert set(os.listdir(root)) == (names | {"ESTOP"} if owner else names)
    refusal = "the platform is already paused" if owner else "stopped before cutover"
    assert f"Refused: {refusal}." in capsys.readouterr().out


@pytest.mark.parametrize("at", [("G10", 2), ("preflight", 2)], ids=["drain", "fresh guards"])
def test_interrupt_before_the_cutover_refuses_and_resumes(root, monkeypatch, capsys, at):
    """Owner rule 326, item 2 (finding 2). SIGINT while the run waits for the drain, or while the
    runner asks every guard again, refuses the release as SIGTERM does: no unit stops, the changes
    go back to the waiting decision, the release's own pause is lifted, and the caller's handlers
    of both signals are back in place."""
    _accepted(1)
    handlers = [signal.getsignal(signum) for signum in (signal.SIGINT, signal.SIGTERM)]
    host = FakeHost(open_reads=3, at={at: lambda: _stop_signal(signal.SIGINT)})

    assert _run(monkeypatch, host) == 1
    assert (host.stops, host.results, host.journal[-2:]) == (0, [], ["outcome refused", "resume"])
    assert ([batch["outcome"] for batch in _batches()], _sentinel()) == (["refused", None], None)
    assert "Refused: stopped before cutover." in capsys.readouterr().out
    assert [signal.getsignal(signum) for signum in (signal.SIGINT, signal.SIGTERM)] == handlers


@pytest.mark.parametrize("signum", [signal.SIGINT, signal.SIGTERM], ids=["SIGINT", "SIGTERM"])
@pytest.mark.parametrize("moment", ["after a unit stopped", "as the runner returns"])
def test_stop_after_the_first_unit_stop_fails_midway_and_stays_paused(
    root, monkeypatch, capsys, moment, signum
):
    """Owner rule 326, item 2 (finding 2). From the first unit stop on, a stop signal is never a
    refusal: once the runner has stopped a unit, or as the unchanged runner hands back its result,
    the release failed midway. Its changes stay in the batch, kept for recovery (card 7), and the
    platform stays paused."""
    from hermes_cli import release_cmd

    batch_id = _accepted(1)["batch_id"]
    host = FakeHost()
    if moment == "after a unit stopped":
        stop_units = host.stop_units

        def stop_units_then_signal(units):
            stop_units(units)
            if host.stops == 1:
                _stop_signal(signum)

        host.stop_units = stop_units_then_signal
    else:
        runner = release_cmd.run_release

        def return_then_signal(*args):
            host.injected.append(runner(*args))  # the unchanged runner's result: released
            _stop_signal(signum)
            return host.injected[-1]

        monkeypatch.setattr(release_cmd, "run_release", return_then_signal)

    assert _run(monkeypatch, host) == 1
    assert host.stops and host.journal[-1] == "outcome failed"
    assert [(batch["state"], batch["outcome"]) for batch in _batches()] == [("failed", "failed")]
    assert (estop.get_state() or {}).get("reason") == f"release {batch_id}"
    out = capsys.readouterr().out
    assert "stopped midway" in out and "Refused" not in out


@pytest.mark.parametrize("removal", ["fails", "succeeds"])
def test_resume_is_read_back(root, monkeypatch, capsys, removal):
    """Owner rule 326, item 3 (finding 3). When the release's own pause cannot be removed, as its
    folder turned read-only once the outcome was recorded, the run says the platform is still
    paused and exits 1, and the batch stays released. The control: a removal that succeeds."""
    batch_id = _accepted(1)["batch_id"]
    host, mode, unlink = FakeHost(), root.stat().st_mode, os.unlink

    def denied(path, *args, **kwargs):  # the removal, failed as a read-only folder fails it
        if Path(path).name == estop.SENTINEL_NAME:
            raise PermissionError(errno.EACCES, os.strerror(errno.EACCES), os.fspath(path))
        return unlink(path, *args, **kwargs)

    def read_only():
        root.chmod(0o555)
        if os.access(root, os.W_OK):  # root writes in a read-only folder anyway
            monkeypatch.setattr(os, "unlink", denied)

    if removal == "fails":  # the outcome is recorded; the removal comes next
        host.at[("resume", 1)] = read_only
    try:
        code = _run(monkeypatch, host)
    finally:
        root.chmod(mode)

    paused = removal == "fails"
    assert (code, host.journal[-2:]) == (1 if paused else 0, ["outcome released", "resume"])
    assert [batch["outcome"] for batch in _batches()] == ["released"]
    assert (estop.get_state() or {}).get("reason") == (f"release {batch_id}" if paused else None)
    out = capsys.readouterr().out
    assert f"Batch {batch_id} was released." in out
    assert ("The platform is still paused" in out) is paused


def test_nothing_is_imported_after_the_units_stop(root, monkeypatch):
    _accepted(1)
    args = _parsed_run()
    boundary = Boundary(root)
    boundary.stamp()  # the live gateway runs PREV
    answer = SimpleNamespace(status=200)  # each readback address answers
    direct = SimpleNamespace(open=lambda url, timeout: contextlib.nullcontext(answer))
    monkeypatch.setattr(subprocess, "run", boundary.run)
    monkeypatch.setattr(build_info, "get_code_identity", lambda **_: {"sha": boundary.head})
    monkeypatch.setattr(release_host_actions, "_DIRECT", direct)
    monkeypatch.setattr(shutil, "disk_usage", lambda path: SimpleNamespace(free=8 * 1024**3))
    Clock().install(monkeypatch)

    assert args.func(args) == 0
    assert boundary.head == NEW and boundary.modules_at_stop is not None
    assert set(sys.modules) - boundary.modules_at_stop == set()


def test_run_is_registered(monkeypatch):
    from hermes_cli import main, release_cmd

    monkeypatch.setattr(main, "_plugin_cli_discovery_needed", lambda: False)
    parser, _subparsers = main._build_cli_parser()
    args = parser.parse_args(["release", "run"])  # parsed before the spy below replaces the run
    monkeypatch.setattr(release_cmd, "run", lambda: 7)

    assert (args.release_command, args.func(args)) == ("run", 7)
