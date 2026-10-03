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
NEW_CONFIG = {"config.yaml": "digest-after-new-migrated-it"}

PINS = Pins(new=NEW, prev=PREV)
MERGES = (
    RecordedMerge(NEW, reviewed_base=PREV, reviewed_head="feature-sha", reviewed_tree="new-tree"),
)
FORWARD_ONLY = ("guards", "stop", "snapshot", "forward", "readback", "start", "readback")


@dataclass
class FakeHost:
    """A healthy release host in memory, running PREV, on which every guard passes.

    NEW rewrites the configuration when it starts, as a migration would, so a test can tell
    whether going back also brought back PREV's configuration.
    """

    head: str = PREV
    running: str | None = PREV
    active: set[str] = field(default_factory=lambda: {*PLATFORM_UNITS, SANDBOX_TUNNEL_UNIT})
    config: dict[str, str] = field(default_factory=lambda: dict(PREV_CONFIG))
    snapshot_dirs: list[str] = field(default_factory=lambda: ["release-1"])
    unhealthy: set[str] = field(default_factory=set)
    will_not_start: set[str] = field(default_factory=set)
    snapshot_fails: bool = False
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
        return self.running is not None and self.running not in self.unhealthy

    def workspace_reads_ok(self):
        return self.running is not None and self.workspace_ok

    def fleet_version(self):
        return self.running or ""

    def stop_units(self, units):
        if self.modules_at_stop is None:
            self.modules_at_stop = set(sys.modules)
        self.active -= set(units)
        self.running = None

    def start_units(self, units):
        if self.head in self.will_not_start:
            raise RuntimeError(f"units failed to start on {self.head}")
        self.active |= set(units)
        self.running = self.head
        if self.head == NEW:
            self.config = dict(NEW_CONFIG)

    def checkout(self, commit):
        self.head = commit

    def restore_config(self):
        self.config = dict(PREV_CONFIG)

    def take_snapshot(self, name):
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
    assert (host.head, host.running) == (NEW, NEW)
    # K6 design rule: once the units stop, the runner imports nothing, so no code can load
    # from the checkout it is swapping.
    assert imported_after_stop == set()


def test_tier1_rolls_back_by_itself_on_failed_health_check():
    host = FakeHost(unhealthy={NEW})

    result = run_release(host, PINS, 1, MERGES)

    assert result.steps == (
        "guards", "stop", "snapshot", "forward", "readback", "restore", "readback",
    )
    assert result.readbacks[0].checks["R4"] is False
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
    ("host_setup", "steps", "outcome"),
    [
        pytest.param(
            {"will_not_start": {NEW}},
            ("guards", "stop", "snapshot", "forward", "restore", "readback"),
            "restored",
            id="new-version-will-not-start",
        ),
        pytest.param(
            {"snapshot_fails": True},
            ("guards", "stop", "snapshot", "restore", "readback"),
            "restored",
            id="snapshot-fails",
        ),
        pytest.param(
            {"workspace_ok": False},
            ("guards", "stop", "snapshot", "forward", "readback", "restore", "readback"),
            "failed",
            id="readback-after-restore-fails-too",
        ),
    ],
)
def test_failure_after_cutover_restores_previous_and_reads_back(host_setup, steps, outcome):
    host = FakeHost(**host_setup)

    result = run_release(host, PINS, 2, MERGES)

    assert result.steps == steps
    assert result.outcome == outcome
    assert result.error
    assert result.readbacks[-1].expected == PREV
    assert (host.head, host.running, host.config) == (PREV, PREV, PREV_CONFIG)


def test_snapshots_keep_only_last_two():
    host = FakeHost(snapshot_dirs=["release-1", "release-2", "release-3"])

    result = run_release(host, PINS, 0, MERGES)

    assert result.outcome == "released"
    assert host.snapshot_dirs == ["release-3", NEW]
