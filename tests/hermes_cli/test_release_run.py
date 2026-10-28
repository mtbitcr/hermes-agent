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
from hermes_cli import build_info, release_guards, release_host_actions, release_runner
from hermes_cli import release_ledger as ledger
from hermes_cli.release_guards import GUARDS, OpenRun
from hermes_cli.subcommands.release import build_release_parser
from tests.hermes_cli.test_release_cmd import (
    CAUSES, SECRET, _printed_outcome, _send_fails, _two_homes,
)


def _commit(label: str) -> str:
    return hashlib.sha1(label.encode()).hexdigest()


PREV, NEW, FEATURE, TREE = (_commit(label) for label in ("prev", "new", "feature", "tree"))
CHECKOUT = "/srv/hermes/checkout"
CONFIG = {"config.yaml": "digest"}
SETTINGS = {
    "units": {"sandbox-tunnel": "hermes-sandbox-tunnel"}, "drain_poll_seconds": 45,
    "health_url": "http://127.0.0.1:8642/health",
}
UNITS = ("hermes-gateway", "hermes-serve", "hermes-sandbox-tunnel")
FORWARD_ONLY = ("guards", "stop", "snapshot", "forward", "readback", "start", "readback")
TIER_2 = FORWARD_ONLY[:5] + ("back", "readback", "forward", "readback") + FORWARD_ONLY[5:]
OWNER = "owner: hold everything"
# The release token of every run here: a placeholder, which no output stream may show.
TOKEN = "placeholder-release-token"
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
    """The live host's processes, faked: git in the checkout, the user manager, the board reader's
    child and docker, in whose default Workspace container R6's reads succeed. The checkout and the
    units move as the live adapters ask, each move noted, a starting gateway stamps its record with
    the checkout's head, and the modules at the first stop are kept."""

    root: Path
    head: str = PREV
    active: set[str] = field(default_factory=lambda: set(UNITS))
    modules_at_stop: set[str] | None = None
    moves: list[str] = field(default_factory=list)

    def run(self, argv, **options):
        if argv[0] == sys.executable:  # the board reader's child: no board has an open run
            return self._done(json.dumps([[] for _path in argv[6:]]))
        if argv[0] == "systemctl":
            return self._systemctl(*argv[2:])
        if argv[0] == "docker":  # docker exec CONTAINER node -e SCRIPT
            return self._done("", 0 if argv[2] == "raphael-workspace" else 1)
        skipped = ("--no-optional-locks", "--end-of-options")
        args = tuple(arg for arg in argv[3:] if arg not in skipped)
        if args[:2] == ("merge-base", "--is-ancestor"):
            return self._done("", 0 if args[2] == args[3] or args[2:] == (PREV, NEW) else 1)
        if args[0] == "checkout":
            self.head = args[-1]
            self.moves.append(f"checkout {self.head}")
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
        self.moves.append(f"{verb} {rest[0]}")
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


class Sent(list):
    """The lines a release told the owner, in order; ``real`` is the send function itself."""


@pytest.fixture(autouse=True)
def sent(monkeypatch: pytest.MonkeyPatch) -> Sent:
    """Every test here: the send function is replaced by a record of its lines, so no test starts
    a real send. The record imports nothing when called, as K6 needs."""
    from hermes_cli import release_cmd

    lines = Sent()
    lines.real = release_cmd._tell_owner
    monkeypatch.setattr(release_cmd, "_tell_owner", lines.append)
    return lines


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
    monkeypatch.setattr(release_cmd, "secrets", SimpleNamespace(token_hex=lambda nbytes: TOKEN))
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
    host, waits = FakeHost(unhealthy=unhealthy), []
    monkeypatch.setattr(release_runner, "_wait", waits.append)  # no wait between reads sleeps

    assert _run(monkeypatch, host) == 1
    assert [result.outcome for result in host.results] == [outcome] == [_batches()[0]["outcome"]]
    # Each unhealthy version is read 12 times after its start, 5 seconds apart, and never holds.
    assert waits == [5] * 11 * len(unhealthy)
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


@pytest.mark.parametrize(
    ("case", "error"),
    [
        ("published", "EACCES"), ("published", "EIO"), ("owner pause", "EACCES"),
        ("owner pause", "EIO"), ("read-only folder", "EACCES"), ("removed", None),
        ("already gone", None),
    ],
)
def test_a_failed_removal_of_the_pause_file_refuses_the_run(
    root, monkeypatch, capsys, case, error
):
    """Owner rule 344, item 3 (the finding of the fourth review). A removal of the release's pause
    file that fails for any reason but a missing file is reported and refuses the run before the
    cutover: once the pause is published, once an owner's pause, written just after the run's
    read, failed the publication, or as the root home is read-only for that removal. No unit
    stops, and only the release's own pause is lifted. The controls: the file is removed, or
    already gone."""
    _accepted(1)
    host, unlink, link, mode = FakeHost(), os.unlink, os.link, root.stat().st_mode
    prefix = f".{estop.SENTINEL_NAME}."  # the start of the pause file's name
    number = getattr(errno, error) if error else None

    def remove(path, *args, **kwargs):  # every removal is real, but the pause file's may fail
        if error is None or not Path(path).name.startswith(prefix):
            return unlink(path, *args, **kwargs)
        if case != "read-only folder":  # an OSError of EACCES is a PermissionError
            raise OSError(number, os.strerror(number), os.fspath(path))
        root.chmod(0o555)
        try:
            if os.access(root, os.W_OK):  # root removes from a read-only folder anyway
                raise PermissionError(number, os.strerror(number), os.fspath(path))
            return unlink(path, *args, **kwargs)
        finally:
            root.chmod(mode)

    def link_then_remove(source, *args, **kwargs):  # the pause is published; its file is gone
        link(source, *args, **kwargs)
        unlink(source)

    monkeypatch.setattr(os, "unlink", remove)
    if case == "already gone":
        monkeypatch.setattr(os, "link", link_then_remove)
    elif case == "owner pause":  # written just after the run read that none was set
        read = estop.is_engaged

        def is_engaged():
            engaged = read()
            if not host.injected:
                host.injected.append(estop.engage(OWNER).read_bytes())
            return engaged

        monkeypatch.setattr(estop, "is_engaged", is_engaged)

    code = _run(monkeypatch, host)
    out, outcomes = capsys.readouterr().out, [batch["outcome"] for batch in _batches()]
    assert _sentinel() == (host.injected[0] if case == "owner pause" else None)
    if error is None:
        assert (code, outcomes) == (0, ["released"])
        assert not [name for name in os.listdir(root) if name.startswith(prefix)]
    else:
        assert (code, host.stops, host.results, outcomes) == (1, 0, [], ["refused", None])
        kind = "PermissionError" if error == "EACCES" else "OSError"
        failure = f"({kind}: [Errno {number}] the temporary pause file could not be removed)."
        assert f"Refused: a step before the cutover failed {failure}" in out


@pytest.mark.parametrize("at", [("G10", 2), ("G10", 4)], ids=["drain", "just before the runner"])
def test_interrupt_before_the_cutover_refuses_and_resumes(root, monkeypatch, capsys, at):
    """Owner rule 326, item 2, and rule 357, item 3. SIGINT while the run waits for the drain, or
    as the drain ends just before the runner, refuses the release as SIGTERM does: no unit stops,
    the changes go back to the waiting decision, the release's own pause is lifted, and the
    caller's handlers of both signals are back in place."""
    _accepted(1)
    handlers = [signal.getsignal(signum) for signum in (signal.SIGINT, signal.SIGTERM)]
    host = FakeHost(open_reads=3, at={at: lambda: _stop_signal(signal.SIGINT)})

    assert _run(monkeypatch, host) == 1
    assert (host.stops, host.results, host.journal[-2:]) == (0, [], ["outcome refused", "resume"])
    assert ([batch["outcome"] for batch in _batches()], _sentinel()) == (["refused", None], None)
    assert "Refused: stopped before cutover." in capsys.readouterr().out
    assert [signal.getsignal(signum) for signum in (signal.SIGINT, signal.SIGTERM)] == handlers


@pytest.mark.parametrize("signum", [signal.SIGINT, signal.SIGTERM], ids=["SIGINT", "SIGTERM"])
@pytest.mark.parametrize(
    "moment", ["in the fresh guards", "after a unit stopped", "as the runner returns"]
)
def test_a_stop_after_the_last_check_keeps_the_runners_outcome(
    root, monkeypatch, capsys, moment, signum
):
    """Owner rule 357, item 3. A stop after the last check, just before the runner, is never read:
    while the runner asks every guard again, once it has stopped a unit, or as it hands back its
    result, the run goes on, records the runner's own outcome, here released, and resumes. Only a
    hard kill ends a release from there, and its recovery belongs to card 7."""
    from hermes_cli import release_cmd

    _accepted(1)
    host = FakeHost()
    if moment == "in the fresh guards":
        host.at[("preflight", 2)] = lambda: _stop_signal(signum)
    elif moment == "after a unit stopped":
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

    assert _run(monkeypatch, host) == 0
    assert host.stops and host.journal[-2:] == ["outcome released", "resume"]
    [batch] = _batches()
    assert (batch["state"], batch["outcome"], _sentinel()) == ("released", "released", None)
    out = capsys.readouterr().out
    assert "stopped midway" not in out and "Refused" not in out


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


@pytest.mark.parametrize("moment", ["before the removal", "after a failed removal"])
def test_an_unreadable_pause_at_the_end_is_never_a_success(root, monkeypatch, capsys, moment):
    """S5 of the second security review. When the pause sentinel cannot be read at the end, the
    run cannot know that its own pause is gone: it keeps the recorded outcome, says that the pause
    could not be read, and exits 1. Before the removal, nothing is removed; after a failed
    removal, the readback is what cannot be read."""
    batch_id = _accepted(1)["batch_id"]
    host, read_text, unlink = FakeHost(), Path.read_text, os.unlink

    def unreadable(path, *args, **kwargs):  # the sentinel's folder turned unreadable
        if path.name == estop.SENTINEL_NAME:
            raise PermissionError(errno.EACCES, os.strerror(errno.EACCES), os.fspath(path))
        return read_text(path, *args, **kwargs)

    def denied(path, *args, **kwargs):  # and its removal fails, as in an unwritable folder
        if Path(path).name == estop.SENTINEL_NAME:
            raise PermissionError(errno.EACCES, os.strerror(errno.EACCES), os.fspath(path))
        return unlink(path, *args, **kwargs)

    def turn_unreadable():
        monkeypatch.setattr(Path, "read_text", unreadable)
        monkeypatch.setattr(os, "unlink", denied)

    host.at[("outcome released", 1) if moment == "before the removal" else ("resume", 1)] = turn_unreadable
    code = _run(monkeypatch, host)

    assert code == 1
    assert [batch["outcome"] for batch in _batches()] == ["released"]
    assert json.loads(_sentinel())["release_token"] == TOKEN  # the release's own pause is in place
    assert ("resume" in host.journal) is (moment == "after a failed removal")
    out = capsys.readouterr().out
    assert f"Batch {batch_id} was released." in out
    assert "The pause could not be read" in out and TOKEN not in out


@pytest.mark.parametrize("signum", [signal.SIGINT, signal.SIGTERM], ids=["SIGINT", "SIGTERM"])
@pytest.mark.parametrize("moment", ["after the begin commit", "before finish_release"])
def test_a_stop_at_either_end_of_the_record_settles_the_batch(
    root, monkeypatch, capsys, moment, signum
):
    """Owner rule 357, items 1 to 3 (S1 of the security review). The stop handlers are in place
    from before the begin commit until the pause's removal is read back, and they only set the
    stop flag. A stop right after the begin commit is read just after the pause is published, and
    refuses the release; one right before finish_release, past the last check, leaves the runner's
    outcome. Either way the batch is settled, the release's own pause is lifted and the caller's
    handlers are back."""
    batch_id, host = _accepted(1)["batch_id"], FakeHost()
    stops = (signal.SIGINT, signal.SIGTERM)
    handlers = [signal.getsignal(stop) for stop in stops]
    if moment == "after the begin commit":
        begin = ledger.begin_release

        def begin_then_signal(*args, **versions):
            begun = begin(*args, **versions)
            _stop_signal(signum)
            return begun

        monkeypatch.setattr(ledger, "begin_release", begin_then_signal)
    else:
        host.at[("outcome released", 1)] = lambda: _stop_signal(signum)

    code = _run(monkeypatch, host)
    (out, err), outcomes = capsys.readouterr(), [batch["outcome"] for batch in _batches()]
    if moment == "after the begin commit":
        assert (code, host.stops, host.results, outcomes) == (1, 0, [], ["refused", None])
        assert host.journal == BEFORE_PAUSE + [
            f"pause 'release {batch_id}' at {root / 'ESTOP'}", "outcome refused", "resume"
        ]
        assert "Refused: stopped before cutover." in out
    else:  # past the last check: the runner's outcome stands
        assert (code, outcomes) == (0, ["released"])
        assert host.journal[-2:] == ["outcome released", "resume"]
    assert (_sentinel(), [signal.getsignal(stop) for stop in stops]) == (None, handlers)
    assert TOKEN not in out + err


@pytest.mark.parametrize("signum", [signal.SIGINT, signal.SIGTERM], ids=["SIGINT", "SIGTERM"])
@pytest.mark.parametrize("moment", ["drain", "first unit stop"])
def test_a_second_stop_only_sets_the_flag_again(root, monkeypatch, capsys, moment, signum):
    """Owner rule 357, item 1 (S2). A stop during the drain or at the first unit stop, then a
    second one as the run puts the caller's handlers back: each only sets the stop flag. The stop
    in the drain refuses the release, the one at the first unit stop leaves the runner's outcome,
    and either way the batch is settled and the release's own pause is lifted."""
    _accepted(1)
    host, calls, put = FakeHost(open_reads=3), [], signal.signal
    if moment == "drain":
        host.at[("G10", 2)] = lambda: _stop_signal(signum)
    else:
        stop_units = host.stop_units

        def stop_units_then_signal(units):
            stop_units(units)
            if host.stops == 1:
                _stop_signal(signum)

        host.stop_units = stop_units_then_signal

    def put_back(number, handler):  # the run sets both of its handlers, then puts both back
        calls.append(number)
        if len(calls) == 3:  # the second stop, as the first handler goes back
            _stop_signal(signum)
        return put(number, handler)

    monkeypatch.setattr(signal, "signal", put_back)

    code = _run(monkeypatch, host)
    (out, err), outcomes = capsys.readouterr(), [batch["outcome"] for batch in _batches()]
    if moment == "drain":
        assert (code, host.stops, host.results, outcomes) == (1, 0, [], ["refused", None])
        assert "Refused: stopped before cutover." in out
    else:
        assert (code, outcomes) == (0, ["released"])
    assert (len(calls), _sentinel()) == (4, None)
    assert TOKEN not in out + err


@pytest.mark.parametrize("case", ["owner pause", "pause removed"])
def test_the_first_unit_stop_reads_the_pause_again(root, monkeypatch, capsys, case):
    """Owner rule 357, item 4 (S3). An owner's pause written over the release's own while the
    runner asks every guard again, here at its G11 read, or the release's pause removed there, is
    read at the runner's first unit stop: the run records refused, no unit stops, nothing moves
    back, and an owner's pause stays."""
    _accepted(1)
    host, disengage, moves = FakeHost(), estop.disengage, []  # the real removal, not the run's spy

    def g11_read():  # the runner's G11 read: the preflight's comes before the runner
        if "runner" in host.journal and not host.injected:
            if case == "owner pause":
                estop.engage(OWNER)
            else:
                disengage()
            host.injected.append(_sentinel())
        return CONFIG

    def noted(name, move):  # a move handed to the host is noted, then made as before
        return lambda *args: moves.append(name) or move(*args)

    host.named_config_snapshot = g11_read
    for name in ("checkout", "restore_config", "start_units"):  # a restore's other moves
        setattr(host, name, noted(name, getattr(host, name)))

    code = _run(monkeypatch, host)
    (out, err), outcomes = capsys.readouterr(), [batch["outcome"] for batch in _batches()]
    assert (code, host.stops, moves, outcomes) == (1, 0, [], ["refused", None])
    assert (host.journal[-1], _sentinel()) == ("outcome refused", host.injected[0])
    assert "Refused: stopped before cutover." in out and "after the cutover" not in out
    assert TOKEN not in out + err


@pytest.mark.parametrize("last", [3, 4, 5, 6], ids=lambda last: f"denied through read {last}")
def test_a_refusal_at_the_first_unit_stop_is_final(root, monkeypatch, capsys, last):
    """Owner rule 357, item 4 (S3), as its review reproduced it on the live adapters, with only
    their processes faked. The run's third read of the pause, at the runner's first unit stop, is
    denied, and so is every read after it up to the ``last``; the first two are the reads once the
    pause is published and just before the runner. The refusal at the first unit stop is final:
    whatever a later read finds, no unit stop, checkout, configuration restore or unit start
    reaches the host, and the run records refused. A denial that reaches the run's reads after
    the outcome may leave the release's own pause in place, a safe end."""
    from hermes_cli import release_cmd

    _accepted(1)
    args = _parsed_run()
    boundary, sentinel, reads = Boundary(root), estop.sentinel_path(), []
    boundary.stamp()  # the live gateway runs PREV
    answer = SimpleNamespace(status=200)  # the health address answers
    direct = SimpleNamespace(open=lambda url, timeout: contextlib.nullcontext(answer))
    read_text, restore = Path.read_text, release_host_actions.ReleaseHostActions.restore_config

    def read_or_deny(path, *rest, **options):  # root reads any file: the denial is injected here
        if path == sentinel:
            reads.append(path)
            if 3 <= len(reads) <= last:
                raise PermissionError(errno.EACCES, os.strerror(errno.EACCES), os.fspath(path))
        return read_text(path, *rest, **options)

    def restore_config(actions):  # the configuration restore starts no process: noted here
        boundary.moves.append("restore configuration")
        return restore(actions)

    monkeypatch.setattr(Path, "read_text", read_or_deny)
    monkeypatch.setattr(release_host_actions.ReleaseHostActions, "restore_config", restore_config)
    monkeypatch.setattr(subprocess, "run", boundary.run)
    monkeypatch.setattr(build_info, "get_code_identity", lambda **_: {"sha": boundary.head})
    monkeypatch.setattr(release_host_actions, "_DIRECT", direct)
    monkeypatch.setattr(shutil, "disk_usage", lambda path: SimpleNamespace(free=8 * 1024**3))
    monkeypatch.setattr(release_cmd, "secrets", SimpleNamespace(token_hex=lambda nbytes: TOKEN))
    results = _spy(monkeypatch, release_cmd, "run_release", lambda *_: None)
    Clock().install(monkeypatch)

    code = args.func(args)
    (out, err), outcomes = capsys.readouterr(), [batch["outcome"] for batch in _batches()]
    assert TOKEN not in out + err
    assert [result.steps[:2] for result in results] == [("guards", "stop")]  # the first unit stop
    assert (code, boundary.moves, outcomes) == (1, [], ["refused", None])
    assert "Refused: stopped before cutover." in out and "after the cutover" not in out


def test_a_failed_cleanup_shows_its_step_type_and_errno_only(root, monkeypatch, capsys):
    """Owner rule 357, item 5 (S4). A removal of the temporary pause file that fails with the
    release token in its file name and its text refuses the run, reported by its step, its type
    and its errno only: no output stream carries the token."""
    _accepted(1)
    host, unlink = FakeHost(), os.unlink

    def remove(path, *args, **kwargs):  # every removal is real but the pause file's, which fails
        if TOKEN not in os.fspath(path):
            return unlink(path, *args, **kwargs)
        raise PermissionError(errno.EACCES, f"{TOKEN} could not be removed", os.fspath(path))

    monkeypatch.setattr(os, "unlink", remove)

    code = _run(monkeypatch, host)
    (out, err), outcomes = capsys.readouterr(), [batch["outcome"] for batch in _batches()]
    assert TOKEN not in out + err
    assert (code, host.stops, host.results, outcomes) == (1, 0, [], ["refused", None])
    step = f"PermissionError: [Errno {errno.EACCES}] the temporary pause file could not be removed"
    assert f"Refused: a step before the cutover failed ({step})." in out
    assert _sentinel() is None


def test_nothing_is_imported_after_the_units_stop(root, monkeypatch):
    _accepted(1)
    args = _parsed_run()
    boundary = Boundary(root)
    boundary.stamp()  # the live gateway runs PREV
    answer = SimpleNamespace(status=200)  # the health address answers
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


@pytest.mark.parametrize(
    ("host", "outcome"),
    [
        ({}, "released"), ({"unhealthy": {NEW}}, "restored"),
        ({"origin": PREV}, "refused"), ({"unhealthy": {NEW, PREV}}, "failed"),
    ],
)
def test_the_owner_is_told_once_how_the_release_ended(
    root, monkeypatch, capsys, sent, host, outcome
):
    batch_id = _accepted(1)["batch_id"]
    monkeypatch.setattr(release_runner, "_wait", lambda seconds: None)

    _run(monkeypatch, FakeHost(**host))

    printed = _printed_outcome(capsys.readouterr().out, batch_id)
    paused = " The platform stays paused." if outcome == "failed" else ""
    assert _batches()[0]["outcome"] == outcome
    assert sent == [f"Release {batch_id} {printed}{paused}"]


@pytest.mark.parametrize(
    ("host", "outcome"),
    [
        ({}, "released"), ({"unhealthy": {NEW}}, "restored"),
        ({"origin": PREV}, "refused"), ({"unhealthy": {NEW, PREV}}, "failed"),
    ],
)
def test_the_real_send_reaches_the_root_owner_whatever_profile_is_sticky(
    root, monkeypatch, capsys, sent, host, outcome
):
    """The send child is really started, offline, with a named profile made sticky: the line lands
    on the root home's chat with the root's token."""
    from hermes_cli import release_cmd

    logged, _ = _two_homes(root, monkeypatch)
    batch_id = _accepted(1)["batch_id"]
    monkeypatch.setattr(release_runner, "_wait", lambda seconds: None)
    monkeypatch.setattr(release_cmd, "_tell_owner", sent.real)

    code = _run(monkeypatch, FakeHost(**host))

    out = capsys.readouterr().out
    printed = _printed_outcome(out, batch_id)
    paused = " The platform stays paused." if outcome == "failed" else ""
    assert _batches()[0]["outcome"] == outcome
    assert code == (0 if outcome == "released" else 1)
    assert out.splitlines().count(f"Batch {batch_id} {release_cmd._OUTCOMES[outcome]}.") == 1
    [told] = logged()
    assert told["chat_id"] == "111111", f"expected 111111, got {told['chat_id']}"
    assert told["token"] == SECRET
    assert told["message"] == f"Release {batch_id} {printed}{paused}"


def test_the_real_send_starts_one_process_that_reads_the_line(root, monkeypatch, capsys, sent):
    calls, before = [], dict(os.environ)

    def run(argv, **options):
        calls.append((argv, options))
        return subprocess.CompletedProcess(argv, 0, "", "")

    monkeypatch.setattr(subprocess, "run", run)

    sent.real("Release 4 was released.")

    [(argv, options)] = calls
    assert argv == [
        sys.executable, "-m", "hermes_cli.main", "send", "--to", "telegram", "--file", "-",
    ]
    assert (options["input"], options["timeout"]) == ("Release 4 was released.", 60)
    assert not options.get("shell") and options.get("capture_output")
    pinned = {"HERMES_HOME": str(root), "HERMES_SUPERVISED_CHILD": "1"}
    assert options.get("env") == {**before, **pinned}
    assert dict(os.environ) == before
    assert capsys.readouterr() == ("", "")


@pytest.mark.parametrize("how", list(CAUSES))
@pytest.mark.parametrize(("unhealthy", "outcome"), [(set(), "released"), ({NEW, PREV}, "failed")])
def test_a_send_that_fails_changes_nothing_else_and_names_its_cause(
    root, monkeypatch, capsys, sent, unhealthy, outcome, how
):
    from hermes_cli import release_cmd

    batch_id, told = _accepted(1)["batch_id"], []
    monkeypatch.setattr(release_runner, "_wait", lambda seconds: None)
    monkeypatch.setattr(release_cmd, "_tell_owner", sent.real)
    _send_fails(monkeypatch, how, told)

    code = _run(monkeypatch, FakeHost(unhealthy=unhealthy))

    shown = capsys.readouterr()
    lines = shown.out.splitlines()
    assert _batches()[0]["outcome"] == outcome
    assert code == (0 if outcome == "released" else 1)
    assert lines.count(f"Batch {batch_id} {release_cmd._OUTCOMES[outcome]}.") == 1
    assert len([line for line in lines if all(part in line for part in CAUSES[how])]) == 1
    assert len(told) == 1 and SECRET not in shown.out + shown.err


# The owner journeys, started once after a release: scripts/journeys/start.sh of this checkout.
START_SH = Path(__file__).resolve().parents[2] / "scripts" / "journeys" / "start.sh"
STARTED_LINE = "Started the unit hermes-journeys (time limit 2700 seconds)."
NOT_VALID = "Journeys not started: the journeys settings file is not valid."
SCRIPT_FAILED = "Journeys not started: the start script failed"
# A settings file that follows every rule, and its entries; no output may show their values.
JOURNEY_SETTINGS = {
    "JOURNEY_PAGE_TEXT": "Owner app = ready",
    "JOURNEY_RUNNER_SSH": "runner@stack-two-placeholder",
    "JOURNEY_S2_SCRIPTS_DIR": "/home/runner/placeholder-journeys",
}
VALID_FILE = "".join(f"{name}={value}\n" for name, value in JOURNEY_SETTINGS.items())
# A value that breaks a rule must not be shown either.
LEAK = "hunter2-placeholder-value"


class Starts(list):
    """The runs of start.sh a release made, each ``(argv, options)``; no process is started.
    ``answer`` is what a run gives: a finished process, or an exception to raise. ``look`` is
    called as each run begins, and what it returns is kept in ``seen``."""

    def __init__(self):
        super().__init__()
        self.answer = subprocess.CompletedProcess([], 0, STARTED_LINE + "\n", "")
        self.look, self.seen = lambda: None, []

    def run(self, argv, **options):
        self.append((argv, options))
        self.seen.append(self.look())
        if isinstance(self.answer, BaseException):
            raise self.answer
        return self.answer


@pytest.fixture
def starts(monkeypatch: pytest.MonkeyPatch) -> Starts:
    """The one seam of the start, ``subprocess.run``, replaced by a record of its calls."""
    recorded = Starts()
    monkeypatch.setattr(subprocess, "run", recorded.run)
    return recorded


def _journey_file(root: Path, content: str | bytes = VALID_FILE) -> Path:
    """``journeys/settings`` under the release root, holding exactly ``content``."""
    path = root / "journeys" / "settings"
    path.parent.mkdir(exist_ok=True)
    path.write_bytes(content.encode() if isinstance(content, str) else content)
    return path


def _sized(size: int) -> str:
    """Rule-following content of exactly ``size`` bytes, one setting."""
    head = f"JOURNEY_PAGE_TEXT={LEAK}"
    return head + "x" * (size - len(head) - 1) + "\n"


def _release_unchanged(code: int, host: FakeHost, batch_id: int, sent: Sent) -> None:
    """Whatever the start did, the release ended as it ends without one: released and exit 0, its
    pause lifted, the owner told first how it ended."""
    assert code == 0
    assert [batch["outcome"] for batch in _batches()] == ["released"]
    assert (_sentinel(), host.journal[-2:]) == (None, ["outcome released", "resume"])
    assert sent[0] == f"Release {batch_id} was released."


def _shown(out_err: tuple[str, str], sent: Sent, *values: str) -> None:
    """None of ``values`` was printed or sent."""
    for value in values:
        assert value not in "".join(out_err)
        assert not any(value in line for line in sent)


def test_a_released_run_with_a_valid_settings_file_starts_the_journeys_once(
    root, monkeypatch, capsys, sent, starts
):
    batch_id = _accepted(1)["batch_id"]
    _journey_file(root)
    monkeypatch.setenv("JOURNEY_PAGE_TEXT", "inherited-placeholder")  # the file's entry wins
    monkeypatch.setenv("JOURNEY_BROWSER_DIR", "/opt/inherited")  # a name the file lacks stays
    before, host = dict(os.environ), FakeHost()
    starts.look = lambda: (_sentinel(), [batch["outcome"] for batch in _batches()], list(sent))

    code = _run(monkeypatch, host)

    [(argv, options)] = starts
    assert argv == [str(START_SH)]
    assert options["timeout"] == 60
    assert options.get("capture_output") and not options.get("shell")
    folder = str(Path(sys.executable).parent)
    assert options["env"] == {
        **before, **JOURNEY_SETTINGS, "HERMES_HOME": str(root),
        "PATH": os.pathsep.join([folder, before["PATH"]]),
    }
    assert options["env"]["PATH"].split(os.pathsep)[0] == folder
    assert dict(os.environ) == before
    # It began at the end of the release: the outcome recorded, the pause lifted, the owner told.
    assert starts.seen == [(None, ["released"], [f"Release {batch_id} was released."])]
    out_err = capsys.readouterr()
    assert out_err.out.splitlines()[-2:] == [f"Batch {batch_id} was released.", STARTED_LINE]
    assert sent == [f"Release {batch_id} was released."]  # a start that worked tells nobody more
    _release_unchanged(code, host, batch_id, sent)
    _shown(out_err, sent, *JOURNEY_SETTINGS.values())


@pytest.mark.parametrize(
    ("content", "added"),
    [
        ("", {}),
        ("JOURNEY_A=1", {"JOURNEY_A": "1"}),
        ("JOURNEY_A=\n", {"JOURNEY_A": ""}),
        ("JOURNEY_A=b=c  d \n", {"JOURNEY_A": "b=c  d "}),
        ("JOURNEY_A1_B2=x\nJOURNEY_Z=y\n", {"JOURNEY_A1_B2": "x", "JOURNEY_Z": "y"}),
        ("JOURNEY_A=héllo\n", {"JOURNEY_A": "héllo"}),
        (_sized(4000), {"JOURNEY_PAGE_TEXT": _sized(4000)[18:-1]}),
    ],
    ids=["empty", "no final newline", "empty value", "value kept as it is", "digits",
         "UTF-8", "4000 bytes"],
)
def test_a_settings_file_that_follows_every_rule_adds_its_entries_as_they_are(
    root, monkeypatch, sent, starts, content, added
):
    _accepted(1)
    _journey_file(root, content)
    before = dict(os.environ)

    assert _run(monkeypatch, FakeHost()) == 0

    [(_argv, options)] = starts
    assert {k: v for k, v in options["env"].items() if k.startswith("JOURNEY_")} == added
    assert set(options["env"]) == set(before) | set(added) | {"HERMES_HOME", "PATH"}


@pytest.mark.parametrize("path", [None, ""], ids=["unset", "empty"])
def test_a_path_that_is_not_there_is_the_interpreters_folder_alone(
    root, monkeypatch, sent, starts, path
):
    _accepted(1)
    _journey_file(root)
    if path is None:
        monkeypatch.delenv("PATH")
    else:
        monkeypatch.setenv("PATH", path)

    assert _run(monkeypatch, FakeHost()) == 0

    [(_argv, options)] = starts
    assert options["env"]["PATH"] == str(Path(sys.executable).parent)


@pytest.mark.parametrize("spelling", [None, "link"], ids=["unset", "a link to the root"])
def test_the_start_is_given_the_release_root_as_its_home(root, monkeypatch, sent, starts, spelling):
    """Pinned as the owner's send is: the root as the release names it, whatever the process has."""
    _accepted(1)
    _journey_file(root)
    if spelling is None:
        monkeypatch.delenv("HERMES_HOME")  # the default home is then the root
    else:
        (root.parent / "link").symlink_to(root)
        monkeypatch.setenv("HERMES_HOME", str(root.parent / "link"))

    assert _run(monkeypatch, FakeHost()) == 0

    [(_argv, options)] = starts
    assert options["env"]["HERMES_HOME"] == str(root)


@pytest.mark.parametrize(
    ("host", "outcome"),
    [
        ({"unhealthy": {NEW}}, "restored"),
        ({"origin": PREV}, "refused"),
        ({"unhealthy": {NEW, PREV}}, "failed"),
    ],
)
def test_a_release_that_did_not_end_released_starts_no_journeys(
    root, monkeypatch, capsys, sent, starts, host, outcome
):
    batch_id = _accepted(1)["batch_id"]
    _journey_file(root)
    monkeypatch.setattr(release_runner, "_wait", lambda seconds: None)

    code = _run(monkeypatch, FakeHost(**host))

    out_err = capsys.readouterr()
    printed = _printed_outcome(out_err.out, batch_id)
    paused = " The platform stays paused." if outcome == "failed" else ""
    assert (code, starts, _batches()[0]["outcome"]) == (1, [], outcome)
    assert sent == [f"Release {batch_id} {printed}{paused}"]
    assert "journey" not in "".join(out_err).lower()


@pytest.mark.parametrize(
    ("hook", "said"),
    [
        (("resume", 1), "The platform is still paused"),
        (("outcome released", 1), "The pause could not be read"),
    ],
    ids=["the removal fails", "the pause cannot be read"],
)
def test_a_released_run_that_exits_1_starts_no_journeys(
    root, monkeypatch, capsys, sent, starts, hook, said
):
    """The batch is released, but the end of the run cannot say that its own pause is gone: it
    exits 1, and exit 1 starts nothing. The first test is the control: the same run exits 0."""
    batch_id = _accepted(1)["batch_id"]
    _journey_file(root)
    host, unlink, read_text = FakeHost(), os.unlink, Path.read_text

    def denied(path, *args, **kwargs):  # the removal of the pause fails, as in a read-only folder
        if Path(path).name == estop.SENTINEL_NAME:
            raise PermissionError(errno.EACCES, os.strerror(errno.EACCES), os.fspath(path))
        return unlink(path, *args, **kwargs)

    def unreadable(path, *args, **kwargs):  # and the sentinel's folder turned unreadable
        if path.name == estop.SENTINEL_NAME:
            raise PermissionError(errno.EACCES, os.strerror(errno.EACCES), os.fspath(path))
        return read_text(path, *args, **kwargs)

    def turn():
        monkeypatch.setattr(os, "unlink", denied)
        if hook[0] == "outcome released":  # before the removal: the read at the end fails
            monkeypatch.setattr(Path, "read_text", unreadable)

    host.at[hook] = turn

    code = _run(monkeypatch, host)

    out_err = capsys.readouterr()
    assert (code, starts) == (1, [])
    assert [batch["outcome"] for batch in _batches()] == ["released"]
    assert said in out_err.out
    assert sent == [f"Release {batch_id} was released."]
    assert "journey" not in "".join(out_err).lower()


@pytest.mark.parametrize("kind", ["no folder", "no file", "other files only"])
def test_a_missing_settings_file_starts_nothing_and_tells_nothing(
    root, monkeypatch, capsys, sent, starts, kind
):
    batch_id = _accepted(1)["batch_id"]
    if kind != "no folder":
        (root / "journeys").mkdir()
    if kind == "other files only":
        (root / "journeys" / "runner.env").write_text(f"JOURNEY_BASE_URL={LEAK}\n")

    code = _run(monkeypatch, FakeHost())

    out_err = capsys.readouterr()
    assert (code, starts) == (0, [])
    assert sent == [f"Release {batch_id} was released."]
    assert out_err.out.splitlines()[-1] == f"Batch {batch_id} was released."
    assert "journey" not in "".join(out_err).lower() and out_err.err == ""


# Each content breaks one rule of the settings file, and holds a value that must not be shown.
INVALID_CONTENT = {
    "a name without the prefix": f"PAGE_TEXT={LEAK}\n",
    "a name in lower case": f"JOURNEY_page_text={LEAK}\n",
    "a name with a dash": f"JOURNEY_PAGE-TEXT={LEAK}\n",
    "a name with a space": f"JOURNEY_PAGE TEXT={LEAK}\n",
    "a space before the equals sign": f"JOURNEY_PAGE_TEXT ={LEAK}\n",
    "a space before the name": f" JOURNEY_PAGE_TEXT={LEAK}\n",
    "the prefix alone": f"JOURNEY_={LEAK}\n",
    "no name": f"={LEAK}\n",
    "a line without an equals sign": f"JOURNEY_PAGE_TEXT {LEAK}\n",
    "a repeated name": f"JOURNEY_PAGE_TEXT={LEAK}\nJOURNEY_PAGE_TEXT={LEAK}\n",
    "a repeated name, another value": f"JOURNEY_PAGE_TEXT=a\nJOURNEY_PAGE_TEXT={LEAK}\n",
    "good lines, then a bad one": VALID_FILE + f"OTHER={LEAK}\n",
    "a blank line": f"JOURNEY_PAGE_TEXT={LEAK}\n\nJOURNEY_RUNNER_SSH={LEAK}\n",
    "a lone newline": "\n",
    "a comment": f"# {LEAK}\nJOURNEY_PAGE_TEXT={LEAK}\n",
    "a NUL byte": f"JOURNEY_PAGE_TEXT={LEAK}\0\n",
    "bytes that are not UTF-8": b"JOURNEY_PAGE_TEXT=" + LEAK.encode() + b"\xff\n",
    "4001 bytes": _sized(4001),
}


def _file_refused(code, host, batch_id, sent, starts, out_err) -> None:
    """The release ended as it does without a start, which was refused for its settings file: one
    fixed line was printed and sent, and no value of the file was."""
    assert starts == []
    assert sent == [f"Release {batch_id} was released.", NOT_VALID]
    assert out_err.out.splitlines()[-1] == NOT_VALID and out_err.err == ""
    _release_unchanged(code, host, batch_id, sent)
    _shown(out_err, sent, LEAK, *JOURNEY_SETTINGS.values())


@pytest.mark.parametrize("content", list(INVALID_CONTENT.values()), ids=list(INVALID_CONTENT))
def test_a_settings_file_that_breaks_a_rule_starts_nothing_and_tells_the_owner(
    root, monkeypatch, capsys, sent, starts, content
):
    batch_id, host = _accepted(1)["batch_id"], FakeHost()
    _journey_file(root, content)

    code = _run(monkeypatch, host)

    _file_refused(code, host, batch_id, sent, starts, capsys.readouterr())


@pytest.mark.parametrize(
    "kind",
    ["link", "dangling link", "folder", "unreadable file", "journeys is a file"],
)
def test_a_settings_file_that_is_not_a_plain_readable_file_starts_nothing_and_tells_the_owner(
    root, monkeypatch, capsys, sent, starts, kind
):
    batch_id, host = _accepted(1)["batch_id"], FakeHost()
    path, elsewhere = root / "journeys" / "settings", root.parent / "elsewhere"
    if kind == "journeys is a file":
        path.parent.write_text(f"JOURNEY_PAGE_TEXT={LEAK}\n")
    elif kind == "link":
        path.parent.mkdir()
        elsewhere.write_text(f"JOURNEY_PAGE_TEXT={LEAK}\n")  # it follows the rules, but is a link
        path.symlink_to(elsewhere)
    elif kind == "dangling link":
        path.parent.mkdir()
        path.symlink_to(elsewhere)
    elif kind == "folder":
        path.mkdir(parents=True)
    else:
        _journey_file(root, f"JOURNEY_PAGE_TEXT={LEAK}\n").chmod(0)
        if os.access(path, os.R_OK):  # root reads it anyway: the open is denied as it is to others
            real_open = os.open

            def denied(target, *args, **kwargs):
                if Path(os.fsdecode(target)).name == "settings":
                    raise PermissionError(errno.EACCES, os.strerror(errno.EACCES), LEAK)
                return real_open(target, *args, **kwargs)

            monkeypatch.setattr(os, "open", denied)

    code = _run(monkeypatch, host)

    _file_refused(code, host, batch_id, sent, starts, capsys.readouterr())


def _not_started(starts: Starts, answer, root: Path, monkeypatch, capsys, sent, told: str) -> None:
    """A valid file, a start that ends as ``answer`` says: the owner is told one line and nothing
    more is printed; the release is as it is without a start, and no value of the file is shown."""
    batch_id, host = _accepted(1)["batch_id"], FakeHost()
    _journey_file(root)
    starts.answer = answer

    code = _run(monkeypatch, host)

    out_err = capsys.readouterr()
    assert len(starts) == 1
    assert sent == [f"Release {batch_id} was released.", told]
    assert out_err.out.splitlines()[-1] == f"Batch {batch_id} was released." and out_err.err == ""
    assert "journey" not in "".join(out_err).lower()
    _release_unchanged(code, host, batch_id, sent)
    _shown(out_err, sent, "on stderr", *JOURNEY_SETTINGS.values())


@pytest.mark.parametrize("status", [1, 2, 127, -9])
@pytest.mark.parametrize(
    ("output", "told"),
    [
        ("Refused: a journey run is still at work (the run lock is held).\n",
         "Journeys not started: a journey run is still at work (the run lock is held)."),
        ("Refused: " + "r" * 200 + "\n", "Journeys not started: " + "r" * 200),
        ("Refused: " + "é" * 200 + "\n", "Journeys not started: " + "é" * 200),
        ("Refused: " + "r" * 201 + "\n", SCRIPT_FAILED),
        ("Refused: " + "é" * 201 + "\n", SCRIPT_FAILED),
        ("Refused: \n", SCRIPT_FAILED),
        ("Refused:\n", SCRIPT_FAILED),
        ("Refused: first.\nRefused: second.\n", "Journeys not started: first."),
        ("Refused: first.\r\n", "Journeys not started: first."),
        ("Refused: first.", "Journeys not started: first."),
        ("", SCRIPT_FAILED),
        ("start.sh: line 3: boom\n", SCRIPT_FAILED),
        ("\nRefused: late.\n", SCRIPT_FAILED),
        ("refused: lower case.\n", SCRIPT_FAILED),
        (" Refused: indented.\n", SCRIPT_FAILED),
        ("Started the unit hermes-journeys (time limit 2700 seconds).\n", SCRIPT_FAILED),
    ],
    ids=["reason", "200 characters", "200 wide characters", "201 characters",
         "201 wide characters", "empty reason", "no space", "first line only", "CRLF",
         "no newline", "no output", "other words", "first line empty", "lower case",
         "indented", "a success line"],
)
def test_a_start_that_ends_with_a_status_tells_the_owner_its_reason_or_that_it_failed(
    root, monkeypatch, capsys, sent, starts, status, output, told
):
    answer = subprocess.CompletedProcess([str(START_SH)], status, output, "Refused: on stderr\n")
    _not_started(starts, answer, root, monkeypatch, capsys, sent, told)


@pytest.mark.parametrize(
    ("output", "shown"),
    [
        (STARTED_LINE + "\n", STARTED_LINE),
        (STARTED_LINE, STARTED_LINE),
        ("one line\nanother line\n", "one line"),
        ("", None),
    ],
    ids=["its line", "no newline", "first line only", "no output"],
)
def test_a_start_that_exits_0_prints_its_line_and_tells_nobody(
    root, monkeypatch, capsys, sent, starts, output, shown
):
    batch_id, host = _accepted(1)["batch_id"], FakeHost()
    _journey_file(root)
    starts.answer = subprocess.CompletedProcess([str(START_SH)], 0, output, "")

    code = _run(monkeypatch, host)

    out_err = capsys.readouterr()
    expected = [f"Batch {batch_id} was released."] + ([shown] if shown else [])
    assert out_err.out.splitlines()[-len(expected):] == expected and out_err.err == ""
    assert sent == [f"Release {batch_id} was released."]
    _release_unchanged(code, host, batch_id, sent)


@pytest.mark.parametrize(
    "error",
    [
        FileNotFoundError(f"no such file: {LEAK}"), PermissionError(errno.EACCES, LEAK),
        OSError(LEAK), ValueError(LEAK), RuntimeError(LEAK),
        UnicodeDecodeError("utf-8", b"\xff", 0, 1, LEAK),
    ],
    ids=lambda error: type(error).__name__,
)
def test_a_start_that_raises_is_named_by_its_type_and_changes_nothing(
    root, monkeypatch, capsys, sent, starts, error
):
    batch_id, host = _accepted(1)["batch_id"], FakeHost()
    _journey_file(root)
    starts.answer = error

    code = _run(monkeypatch, host)

    out_err = capsys.readouterr()
    [named] = [line for line in out_err.out.splitlines() if type(error).__name__ in line]
    assert "journeys" in named and out_err.out.splitlines()[-1] == named and out_err.err == ""
    assert len(starts) == 1 and sent == [f"Release {batch_id} was released."]
    _release_unchanged(code, host, batch_id, sent)
    _shown(out_err, sent, LEAK, *JOURNEY_SETTINGS.values())


@pytest.mark.parametrize(
    ("output", "told"),
    [
        (None, SCRIPT_FAILED),
        (b"", SCRIPT_FAILED),
        (b"Refused: the run lock is held.\n", "Journeys not started: the run lock is held."),
        ("Refused: the run lock is held.\n", "Journeys not started: the run lock is held."),
        (b"Refused: " + b"r" * 201 + b"\n", SCRIPT_FAILED),
        (b"Started the unit hermes-journeys (time limit 2700 seconds).\n", SCRIPT_FAILED),
        (b"\xff\xfe", SCRIPT_FAILED),
    ],
    ids=["no output", "empty output", "a reason", "a reason as text", "a long reason",
         "a success line", "bytes that are not UTF-8"],
)
def test_a_start_that_reaches_the_limit_tells_the_owner_and_changes_nothing(
    root, monkeypatch, capsys, sent, starts, output, told
):
    """A pipe a background process holds open is a wait that the 60 seconds end: subprocess.run
    then raises with what was read so far, and the run waits no more."""
    answer = subprocess.TimeoutExpired([str(START_SH)], 60, output=output)
    _not_started(starts, answer, root, monkeypatch, capsys, sent, told)
    assert starts[0][1]["timeout"] == 60


@pytest.mark.parametrize("signum", [signal.SIGINT, signal.SIGTERM], ids=["SIGINT", "SIGTERM"])
def test_a_stop_signal_during_the_start_changes_nothing(root, monkeypatch, sent, starts, signum):
    """The start runs under the stop handlers, which only set a flag: a stop while it waits, up to
    60 seconds, is no interrupt, and the release ends as it does without a start."""
    batch_id, host = _accepted(1)["batch_id"], FakeHost()
    _journey_file(root)
    names = (signal.SIGINT, signal.SIGTERM)
    handlers = [signal.getsignal(name) for name in names]
    starts.look = lambda: _stop_signal(signum)

    code = _run(monkeypatch, host)

    assert len(starts) == 1
    _release_unchanged(code, host, batch_id, sent)
    assert [signal.getsignal(name) for name in names] == handlers


@pytest.mark.parametrize("case", ["started", "refused", "limit", "raised", "invalid file"])
def test_the_journeys_start_imports_nothing_after_the_units_stop(root, monkeypatch, starts, case):
    """K6 for the start: it runs after the cutover, with the checkout moved to NEW, and it reads
    its settings, starts its process and reports how that ended with the modules loaded before."""
    _accepted(1)
    _journey_file(root, "not a setting\n" if case == "invalid file" else VALID_FILE)
    starts.answer = {
        "refused": subprocess.CompletedProcess([], 1, "Refused: the run lock is held.\n", ""),
        "limit": subprocess.TimeoutExpired([str(START_SH)], 60, output=b"Refused: too late.\n"),
        "raised": PermissionError(errno.EACCES, "Permission denied"),
    }.get(case, starts.answer)
    args = _parsed_run()
    boundary = Boundary(root)
    boundary.stamp()  # the live gateway runs PREV
    answer = SimpleNamespace(status=200)  # the health address answers
    direct = SimpleNamespace(open=lambda url, timeout: contextlib.nullcontext(answer))

    def run(argv, **options):
        if argv == [str(START_SH)]:
            return starts.run(argv, **options)
        return boundary.run(argv, **options)

    monkeypatch.setattr(subprocess, "run", run)
    monkeypatch.setattr(build_info, "get_code_identity", lambda **_: {"sha": boundary.head})
    monkeypatch.setattr(release_host_actions, "_DIRECT", direct)
    monkeypatch.setattr(shutil, "disk_usage", lambda path: SimpleNamespace(free=8 * 1024**3))
    Clock().install(monkeypatch)

    assert args.func(args) == 0
    assert len(starts) == (case != "invalid file")
    assert boundary.head == NEW and boundary.modules_at_stop is not None
    assert set(sys.modules) - boundary.modules_at_stop == set()
