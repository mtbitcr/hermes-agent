"""T22 to T27: the release runner's tier sequences, its readbacks and its restore path.

Every test drives the runner through FakeHost, an in-memory release host, so no real host
command runs.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field

import pytest

from hermes_cli.release_guards import Pins, RecordedMerge
from hermes_cli.release_runner import PLATFORM_UNITS, SANDBOX_TUNNEL_UNIT, run_release

PREV = "prev-sha"
NEW = "new-sha"
CHECKOUT = "/srv/hermes/checkout"
PREV_CONFIG = {"config.yaml": "prev-digest"}
DRIFTED_CONFIG = {"config.yaml": "digest-after-a-start-rewrote-it"}

PINS = Pins(new=NEW, prev=PREV)
MERGES = (
    RecordedMerge(NEW, reviewed_base=PREV, reviewed_head="feature-sha", reviewed_tree="new-tree"),
)
FORWARD_ONLY = ("guards", "stop", "snapshot", "forward", "readback", "start", "readback")
RESTORE_AFTER_FORWARD = (
    "guards", "stop", "snapshot", "forward", "readback", "restore", "readback",
)


@dataclass
class FakeHost:
    """A healthy release host in memory, running PREV, on which every guard passes.

    Each switch breaks one thing; a set switch breaks it only for the versions it holds.
    `config_drifts_on_start` makes a start rewrite the configuration, as a migration or startup
    drift would, so a test can tell whether a readback compared it with the named snapshot.
    """

    head: str = PREV
    running: str | None = PREV
    active: set[str] = field(default_factory=lambda: {*PLATFORM_UNITS, SANDBOX_TUNNEL_UNIT})
    config: dict[str, str] = field(default_factory=lambda: dict(PREV_CONFIG))
    snapshot_dirs: list[str] = field(default_factory=lambda: ["release-1"])
    unhealthy: set[str] = field(default_factory=set)
    will_not_start: set[str] = field(default_factory=set)
    will_not_stop: set[str] = field(default_factory=set)
    checkout_fails: set[str] = field(default_factory=set)
    health_unanswered: set[str] = field(default_factory=set)
    config_drifts_on_start: set[str] = field(default_factory=set)
    snapshot_fails: bool = False
    restore_config_fails: bool = False
    # Interruptions are BaseException, not Exception: a person's Ctrl-C or an exit inside a call.
    snapshot_interrupted: bool = False
    checkout_interrupted: set[str] = field(default_factory=set)
    health_exits: set[str] = field(default_factory=set)
    workspace_ok: bool = True
    modules_at_stop: set[str] | None = None

    def checkout_head(self):
        return self.head

    def checkout_is_clean(self):
        return True

    def origin_main(self):
        return NEW

    def first_parent_chain(self, prev, new):
        return [NEW]

    def commit_parents(self, commit):
        return (PREV, "feature-sha")

    def commit_tree(self, commit):
        return "new-tree"

    def is_ancestor(self, ancestor, descendant):
        return (ancestor, descendant) == (PREV, NEW)

    def changed_paths(self, prev, new):
        return {"hermes_cli/main.py"}

    def checkout_root(self):
        return CHECKOUT

    def venv_import_root(self):
        return CHECKOUT

    def unit_property(self, unit, name):
        return {"KillSignal": "SIGINT", "KillMode": "mixed"}[name]

    def free_disk_bytes(self):
        return 5 * 1024**3

    def snapshots(self):
        return list(self.snapshot_dirs)

    def open_native_runs(self):
        return []

    def live_config(self):
        return dict(self.config)

    def named_config_snapshot(self):
        return dict(PREV_CONFIG)

    def unit_active(self, unit):
        return unit in self.active

    def health_ok(self):
        if self.running in self.health_exits:
            raise SystemExit(f"health check exited on {self.running}")
        if self.running in self.health_unanswered:
            raise ConnectionError(f"health endpoint did not answer on {self.running}")
        return self.running is not None and self.running not in self.unhealthy

    def workspace_reads_ok(self):
        return self.running is not None and self.workspace_ok

    def fleet_version(self):
        return self.running or ""

    def stop_units(self, units):
        if self.running in self.will_not_stop:
            raise RuntimeError(f"units did not stop on {self.running}")
        if self.modules_at_stop is None:
            self.modules_at_stop = set(sys.modules)
        self.active -= set(units)
        self.running = None

    def start_units(self, units):
        if self.head in self.will_not_start:
            raise RuntimeError(f"units failed to start on {self.head}")
        self.active |= set(units)
        # Starting units that never stopped changes nothing; they keep running what they ran.
        if self.running is None:
            self.running = self.head
            if self.head in self.config_drifts_on_start:
                self.config = dict(DRIFTED_CONFIG)

    def checkout(self, commit):
        if commit in self.checkout_interrupted:
            raise KeyboardInterrupt(f"checkout of {commit} interrupted")
        if commit in self.checkout_fails:
            raise OSError(f"checkout of {commit} failed")
        self.head = commit

    def restore_config(self):
        if self.restore_config_fails:
            raise OSError("the named configuration snapshot could not be copied back")
        self.config = dict(PREV_CONFIG)

    def take_snapshot(self, name):
        if self.snapshot_interrupted:
            raise KeyboardInterrupt("snapshot interrupted")
        if self.snapshot_fails:
            raise OSError("no space left for the state snapshot")
        self.snapshot_dirs.append(name)

    def delete_snapshot(self, name):
        self.snapshot_dirs.remove(name)


def test_tier2_forward_back_forward():
    host = FakeHost()

    result = run_release(host, PINS, 2, MERGES)
    imported_after_stop = set(sys.modules) - host.modules_at_stop

    assert result.steps == (
        "guards", "stop", "snapshot",
        "forward", "readback", "back", "readback", "forward", "readback",
        "start", "readback",
    )
    assert result.outcome == "released"
    assert [(readback.expected, readback.ok) for readback in result.readbacks] == [
        (NEW, True), (PREV, True), (NEW, True), (NEW, True),
    ]
    # The readbacks after forward, back, forward again and the closing start each compared the
    # live configuration with the named snapshot.
    assert [readback.checks.get("configuration") for readback in result.readbacks] == [True] * 4
    assert (host.head, host.running) == (NEW, NEW)
    # K6 design rule: once the units stop, the runner imports nothing, so no code can load
    # from the checkout it is swapping.
    assert imported_after_stop == set()


def test_tier1_rolls_back_by_itself_on_failed_health_check():
    host = FakeHost(unhealthy={NEW}, config_drifts_on_start={NEW})

    result = run_release(host, PINS, 1, MERGES)

    assert result.steps == (
        "guards", "stop", "snapshot", "forward", "readback", "restore", "readback",
    )
    assert result.readbacks[0].checks["R4"] is False
    # NEW rewrote the configuration when it started, and the readback after forward saw it.
    assert result.readbacks[0].checks.get("configuration") is False
    assert result.outcome == "restored"
    assert (result.readbacks[-1].expected, result.readbacks[-1].ok) == (PREV, True)
    assert (host.head, host.running, host.config) == (PREV, PREV, PREV_CONFIG)


def test_tier1_stays_forward_when_check_passes():
    host = FakeHost()

    result = run_release(host, PINS, 1, MERGES)

    assert result.steps == FORWARD_ONLY
    assert result.outcome == "released"
    assert [(readback.expected, readback.ok) for readback in result.readbacks] == [
        (NEW, True), (NEW, True),
    ]
    assert (host.head, host.running) == (NEW, NEW)


def test_tier0_goes_forward_only():
    host = FakeHost()

    result = run_release(host, PINS, 0, MERGES)

    assert result.steps == FORWARD_ONLY
    assert result.outcome == "released"
    assert (host.head, host.running) == (NEW, NEW)


@pytest.mark.parametrize(
    ("host_setup", "steps", "outcome", "failed_checks", "errors", "read_errors"),
    [
        pytest.param(
            {"will_not_start": {NEW}},
            ("guards", "stop", "snapshot", "forward", "restore", "readback"),
            "restored",
            [],
            (f"RuntimeError: units failed to start on {NEW}",),
            {},
            id="new-version-will-not-start",
        ),
        pytest.param(
            {"snapshot_fails": True},
            ("guards", "stop", "snapshot", "restore", "readback"),
            "restored",
            [],
            ("OSError: no space left for the state snapshot",),
            {},
            id="snapshot-fails",
        ),
        pytest.param(
            {"health_unanswered": {NEW}},
            RESTORE_AFTER_FORWARD,
            "restored",
            [],
            (f"ReadbackFailed: R4 did not hold on {NEW}",),
            {0: {"R4": f"ConnectionError: health endpoint did not answer on {NEW}"}},
            id="forward-readback-read-raises",
        ),
        pytest.param(
            {"workspace_ok": False},
            ("guards", "stop", "snapshot", "forward", "readback", "restore", "readback"),
            "failed",
            ["R6"],
            (f"ReadbackFailed: R6 did not hold on {NEW}",),
            {},
            id="readback-after-restore-fails-too",
        ),
        # In the cases below NEW is unhealthy, so the readback after forward starts the restore,
        # and one step of the restore, or a read in the readback after it, goes wrong.
        pytest.param(
            # Units that never stopped keep running NEW under PREV's checkout.
            {"unhealthy": {NEW}, "will_not_stop": {NEW}},
            RESTORE_AFTER_FORWARD,
            "failed",
            ["R4", "fleet version"],
            (
                f"ReadbackFailed: R4 did not hold on {NEW}",
                f"RuntimeError: units did not stop on {NEW}",
            ),
            {},
            id="recovery-stop-fails",
        ),
        pytest.param(
            {"unhealthy": {NEW}, "checkout_fails": {PREV}},
            RESTORE_AFTER_FORWARD,
            "failed",
            ["R1", "R4", "fleet version"],
            (
                f"ReadbackFailed: R4 did not hold on {NEW}",
                f"OSError: checkout of {PREV} failed",
            ),
            {},
            id="recovery-checkout-fails",
        ),
        pytest.param(
            {"unhealthy": {NEW}, "config_drifts_on_start": {NEW}, "restore_config_fails": True},
            RESTORE_AFTER_FORWARD,
            "failed",
            ["configuration"],
            (
                f"ReadbackFailed: R4, configuration did not hold on {NEW}",
                "OSError: the named configuration snapshot could not be copied back",
            ),
            {},
            id="recovery-config-restore-fails",
        ),
        pytest.param(
            {"unhealthy": {NEW}, "will_not_start": {PREV}},
            RESTORE_AFTER_FORWARD,
            "failed",
            ["R3", "R4", "R6", "fleet version"],
            (
                f"ReadbackFailed: R4 did not hold on {NEW}",
                f"RuntimeError: units failed to start on {PREV}",
            ),
            {},
            id="recovery-prev-start-fails",
        ),
        pytest.param(
            {"unhealthy": {NEW}, "health_unanswered": {PREV}},
            RESTORE_AFTER_FORWARD,
            "failed",
            ["R4"],
            (f"ReadbackFailed: R4 did not hold on {NEW}",),
            {-1: {"R4": f"ConnectionError: health endpoint did not answer on {PREV}"}},
            id="recovery-readback-read-raises",
        ),
        pytest.param(
            # Startup drift: PREV's start rewrites the configuration the restore just put back.
            {"unhealthy": {NEW}, "config_drifts_on_start": {PREV}},
            RESTORE_AFTER_FORWARD,
            "failed",
            ["configuration"],
            (f"ReadbackFailed: R4 did not hold on {NEW}",),
            {},
            id="config-drifts-when-prev-starts",
        ),
        # Interruptions (security review of PR 143): each of the three barriers holds for a
        # BaseException that is not an Exception, as it does for an ordinary failure.
        pytest.param(
            {"snapshot_interrupted": True},
            ("guards", "stop", "snapshot", "restore", "readback"),
            "restored",
            [],
            ("KeyboardInterrupt: snapshot interrupted",),
            {},
            id="interrupted-after-the-stop",
        ),
        pytest.param(
            {"unhealthy": {NEW}, "checkout_interrupted": {PREV}},
            RESTORE_AFTER_FORWARD,
            "failed",
            ["R1", "R4", "fleet version"],
            (
                f"ReadbackFailed: R4 did not hold on {NEW}",
                f"KeyboardInterrupt: checkout of {PREV} interrupted",
            ),
            {},
            id="recovery-step-interrupted",
        ),
        pytest.param(
            {"unhealthy": {NEW}, "health_exits": {PREV}},
            RESTORE_AFTER_FORWARD,
            "failed",
            ["R4"],
            (f"ReadbackFailed: R4 did not hold on {NEW}",),
            {-1: {"R4": f"SystemExit: health check exited on {PREV}"}},
            id="recovery-readback-read-exits",
        ),
    ],
)
def test_failure_after_cutover_restores_previous_and_reads_back(
    host_setup, steps, outcome, failed_checks, errors, read_errors
):
    host = FakeHost(**host_setup)

    result = run_release(host, PINS, 2, MERGES)

    assert result.steps == steps
    assert result.outcome == outcome
    readback = result.readbacks[-1]
    assert readback.expected == PREV
    assert "configuration" in readback.checks
    assert [name for name, ok in readback.checks.items() if not ok] == failed_checks
    # The failure that started the restore comes first, then every failure inside the restore.
    assert (result.error, *result.restore_errors) == errors
    # A read that raised fails its check, and the readback it ran in keeps its cause by check
    # name. read_errors names each such readback by its index in readbacks.
    assert {index: result.readbacks[index].errors for index in read_errors} == read_errors
    # A restored result holds for the host itself, not only for its readback.
    if outcome == "restored":
        assert (host.head, host.running, host.config) == (PREV, PREV, PREV_CONFIG)


def test_snapshots_keep_only_last_two():
    host = FakeHost(snapshot_dirs=["release-1", "release-2", "release-3"])

    result = run_release(host, PINS, 0, MERGES)

    assert result.outcome == "released"
    assert host.snapshot_dirs == ["release-3", NEW]
