"""Release guards G1 to G11, asked of the release host through an adapter.

Today these guards live only in the per-release wrapper script on the release server. Here they
keep what they mean but become checks over `HostReader`, so one pass serves both prepare mode
and the release unit, which re-runs it fresh just before cutover. Every `HostReader` method
only reads, which is what lets prepare mode promise to write nothing.
"""

from __future__ import annotations

from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

LOCK_FILE = "uv.lock"
PROJECT_MANIFEST = "pyproject.toml"
MIN_FREE_DISK_BYTES = 4 * 1024**3

# Units are named by role; the host adapter maps each role to its systemd unit.
GATEWAY_UNIT = "gateway"
SERVE_UNIT = "serve"


@dataclass(frozen=True)
class Pins:
    """The version a release installs and the one it replaces."""

    new: str
    prev: str

    @property
    def snapshot(self) -> str:
        """This release's state snapshot is named after the version it installs."""
        return self.new


@dataclass(frozen=True)
class RecordedMerge:
    """One batch member as the record holds it: the merge commit and what review saw."""

    commit: str
    reviewed_base: str
    reviewed_head: str
    reviewed_tree: str


@dataclass(frozen=True)
class OpenRun:
    """A native run the host still lists as open."""

    run_id: str
    worker_alive: bool


@dataclass(frozen=True)
class GuardResult:
    """One guard's verdict, with the rule it checked so a refusal can say why."""

    guard: str
    rule: str
    ok: bool


class HostReader(Protocol):
    """What the guards may ask of the release host. Every method only reads."""

    def checkout_head(self) -> str: ...

    def checkout_is_clean(self) -> bool: ...

    def origin_main(self) -> str: ...

    def first_parent_chain(self, prev: str, new: str) -> Sequence[str]:
        """Commits on `new`'s first-parent chain, newest first, stopping before `prev`."""

    def commit_parents(self, commit: str) -> tuple[str, ...]: ...

    def commit_tree(self, commit: str) -> str: ...

    def is_ancestor(self, ancestor: str, descendant: str) -> bool: ...

    def changed_paths(self, prev: str, new: str) -> Collection[str]:
        """Repository-relative paths that differ between the two commits."""

    def checkout_root(self) -> str: ...

    def venv_import_root(self) -> str:
        """The checkout the live venv's interpreter imports Hermes from."""

    def unit_property(self, unit: str, name: str) -> str:
        """A systemd property of the unit; signals come back by name, such as "SIGINT"."""

    def free_disk_bytes(self) -> int:
        """Free space on the filesystem holding the checkout and the state snapshots."""

    def snapshots(self) -> Sequence[str]:
        """Names of the state snapshots on disk, oldest first."""

    def open_native_runs(self) -> Sequence[OpenRun]: ...

    def live_config(self) -> Mapping[str, str]:
        """Each live configuration file mapped to a digest of its content."""

    def named_config_snapshot(self) -> Mapping[str, str]:
        """The same mapping for the named configuration snapshot."""


GuardCheck = Callable[[HostReader, Pins, Sequence[RecordedMerge]], bool]


def _chain_was_reviewed(host: HostReader, pins: Pins, merges: Sequence[RecordedMerge]) -> bool:
    # Every commit on the chain must be a recorded merge, not only every merge on it: a direct
    # commit was never reviewed, just as the single-release check refused a NEW that was not
    # the reviewed merge.
    recorded = {merge.commit: merge for merge in merges}
    return all(
        commit in recorded and _matches_review(host, recorded[commit])
        for commit in host.first_parent_chain(pins.prev, pins.new)
    )


def _matches_review(host: HostReader, merge: RecordedMerge) -> bool:
    return (
        host.commit_parents(merge.commit) == (merge.reviewed_base, merge.reviewed_head)
        and host.commit_tree(merge.commit) == merge.reviewed_tree
    )


def _gateway_stops_with_sigint_mixed(
    host: HostReader, pins: Pins, merges: Sequence[RecordedMerge]
) -> bool:
    return (
        host.unit_property(GATEWAY_UNIT, "KillSignal") == "SIGINT"
        and host.unit_property(GATEWAY_UNIT, "KillMode") == "mixed"
    )


GUARDS: tuple[tuple[str, str, GuardCheck], ...] = (
    (
        "G1",
        "the checkout is at PREV and clean",
        lambda host, pins, merges: host.checkout_head() == pins.prev and host.checkout_is_clean(),
    ),
    ("G2", "origin main equals NEW", lambda host, pins, merges: host.origin_main() == pins.new),
    (
        "G3",
        "every commit from PREV to NEW is a batch merge that matches its own review",
        _chain_was_reviewed,
    ),
    (
        "G4",
        "PREV is an ancestor of NEW",
        lambda host, pins, merges: host.is_ancestor(pins.prev, pins.new),
    ),
    (
        "G5",
        "the lock file and the project manifest are unchanged: releases are code-only",
        lambda host, pins, merges: not {LOCK_FILE, PROJECT_MANIFEST}.intersection(
            host.changed_paths(pins.prev, pins.new)
        ),
    ),
    (
        "G6",
        "the live venv imports this checkout",
        lambda host, pins, merges: host.venv_import_root() == host.checkout_root(),
    ),
    (
        "G7",
        "the gateway unit stops with SIGINT and KillMode mixed",
        _gateway_stops_with_sigint_mixed,
    ),
    (
        "G8",
        "at least 4 GiB of disk are free",
        lambda host, pins, merges: host.free_disk_bytes() >= MIN_FREE_DISK_BYTES,
    ),
    (
        "G9",
        "this release's state snapshot does not exist yet",
        lambda host, pins, merges: pins.snapshot not in host.snapshots(),
    ),
    (
        "G10",
        "no native run is open; a run whose worker is dead is ignored",
        lambda host, pins, merges: not any(run.worker_alive for run in host.open_native_runs()),
    ),
    (
        "G11",
        "the live configuration equals the named configuration snapshot",
        lambda host, pins, merges: host.live_config() == host.named_config_snapshot(),
    ),
)


def prepare(
    host: HostReader, pins: Pins, merges: Sequence[RecordedMerge]
) -> tuple[GuardResult, ...]:
    """Prepare mode: ask the host every guard, in order, and write nothing.

    A failing guard does not stop the pass, so one refusal names every guard that failed. The
    release unit runs the same pass again, fresh, just before cutover.
    """
    return tuple(
        GuardResult(guard, rule, check(host, pins, merges)) for guard, rule, check in GUARDS
    )
