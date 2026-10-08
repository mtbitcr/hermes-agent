"""The release runner: guards, cutover, the tier's moves, and a readback after every move; and the
recovery of a release that stopped midway.

Like the guards, these steps come from the per-release wrapper script on the release server and
now run against a host adapter, `ReleaseHost`. Pins, the tier and the batch's recorded merges
are arguments; nothing here reads the release record yet.

Design rule (K6): every import this module needs happens when it loads, before any cutover,
and no function imports anything. Once the units stop, the checkout on disk is swapped under
the running process, so a late import would load code from the version being installed
instead of the one running the release.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Protocol

from hermes_cli.release_guards import (
    GATEWAY_UNIT,
    SERVE_UNIT,
    GuardResult,
    HostReader,
    Pins,
    RecordedMerge,
    prepare,
)

PLATFORM_UNITS = (GATEWAY_UNIT, SERVE_UNIT)
SANDBOX_TUNNEL_UNIT = "sandbox-tunnel"
KEEP_SNAPSHOTS = 2

# Tier 2 rehearses the way back before settling forward. Tiers 0 and 1 only go forward; if a
# readback after that fails, the failure rule brings them back.
TIER_MOVES: dict[int, tuple[str, ...]] = {
    0: ("forward",),
    1: ("forward",),
    2: ("forward", "back", "forward"),
}


class ReleaseHost(HostReader, Protocol):
    """Everything the runner asks of the release host, on top of the guards' reads.

    The readback reads answer False, rather than raising, when the host does not answer. A read
    that raises anyway counts as a failed check.
    """

    def unit_active(self, unit: str) -> bool: ...

    def health_ok(self) -> bool:
        """The local health endpoint answers ok."""

    def workspace_reads_ok(self) -> bool:
        """A projects list and one board read both succeed from inside the Workspace container."""

    def fleet_version(self) -> str:
        """The version the running gateways report."""

    def stop_units(self, units: Sequence[str]) -> None: ...

    def start_units(self, units: Sequence[str]) -> None: ...

    def checkout(self, commit: str) -> None: ...

    def restore_config(self) -> None:
        """Put the named configuration snapshot, the one G11 compared, back in place."""

    def take_snapshot(self, name: str) -> None:
        """Copy every application database and configuration file, WAL checkpointed first."""

    def delete_snapshot(self, name: str) -> None: ...


@dataclass(frozen=True)
class Readback:
    """What one readback saw, checked against the version the host should now be running.

    A read that raises fails its check; `errors` keeps its cause, "Type: message", under the
    check's name, as evidence for the release journal and the operator.
    """

    expected: str
    checks: Mapping[str, bool]
    errors: Mapping[str, str] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return all(self.checks.values())


READBACKS: tuple[tuple[str, Callable[[ReleaseHost, str], bool]], ...] = (
    ("R1", lambda host, expected: host.checkout_head() == expected),
    ("R2", lambda host, expected: host.venv_import_root() == host.checkout_root()),
    ("R3", lambda host, expected: all(host.unit_active(unit) for unit in PLATFORM_UNITS)),
    ("R4", lambda host, expected: host.health_ok()),
    ("R5", lambda host, expected: host.unit_active(SANDBOX_TUNNEL_UNIT)),
    ("R6", lambda host, expected: host.workspace_reads_ok()),
    ("fleet version", lambda host, expected: host.fleet_version() == expected),
    # After every move the live configuration still equals the named snapshot G11 compared it
    # with: releases are code-only, and the restore puts that snapshot back.
    ("configuration", lambda host, expected: host.live_config() == host.named_config_snapshot()),
)
# Recovery compares the configuration only when NEW's configuration snapshot may exist (see
# recover). Without it, the check is left out, or counts as not held.
_NO_CONFIGURATION = tuple(read for read in READBACKS if read[0] != "configuration")
_CONFIGURATION_NOT_HELD = (*_NO_CONFIGURATION, ("configuration", lambda host, expected: False))


@dataclass(frozen=True)
class ReleaseResult:
    """How a release ended, with every step and readback on the way.

    `outcome` is "refused" (a guard failed and nothing was written), "released", "restored" (a
    failure after cutover was undone and PREV read back), or "failed" (the readback after the
    restore did not hold either, so the host needs a person). `error` is the failure that started
    the restore, and `restore_errors` holds every failure inside the restore.
    """

    outcome: str
    steps: tuple[str, ...]
    guards: tuple[GuardResult, ...]
    readbacks: tuple[Readback, ...] = ()
    error: str = ""
    restore_errors: tuple[str, ...] = ()


class ReadbackFailed(Exception):
    """A readback after a move did not hold. The failure rule treats it like any other error."""


def run_release(
    host: ReleaseHost, pins: Pins, tier: int, merges: Sequence[RecordedMerge]
) -> ReleaseResult:
    """Run one release of the batch's tier, from PREV to NEW."""
    moves = TIER_MOVES[tier]
    steps = ["guards"]
    guards = prepare(host, pins, merges)
    if not all(result.ok for result in guards):
        return ReleaseResult("refused", tuple(steps), guards)

    readbacks: list[Readback] = []
    # From the stop on, any failure, raised or read back, takes the host back to PREV and its
    # configuration and reads back again.
    try:
        steps.append("stop")
        host.stop_units(PLATFORM_UNITS)
        steps.append("snapshot")
        _snapshot(host, pins)
        for move in moves:
            steps.append(move)
            _require(_read_back(host, MOVES[move](host, pins), steps, readbacks))
        # The last move already brought both units up on NEW, so this start changes nothing on a
        # healthy host; it closes the sequence the way the plan lays it out.
        steps.append("start")
        host.start_units(PLATFORM_UNITS)
        _require(_read_back(host, pins.new, steps, readbacks))
    except BaseException as failure:
        steps.append("restore")
        restore_errors = _restore(host, pins)
        restored = _read_back(host, pins.prev, steps, readbacks).ok
        return ReleaseResult(
            "restored" if restored else "failed",
            tuple(steps),
            guards,
            tuple(readbacks),
            f"{type(failure).__name__}: {failure}",
            restore_errors,
        )
    return ReleaseResult("released", tuple(steps), guards, tuple(readbacks))


def recover(host: ReleaseHost, pins: Pins, *, saved: bool = True) -> ReleaseResult:
    """Bring the host back after a release stopped midway, to exactly PREV or exactly NEW.

    The checked-out version is read back when it is PREV or NEW, and kept when the readback holds:
    NEW ends released, PREV restored. Otherwise the merged restore steps run and PREV is read
    back, and the outcome is failed when that fails too. Recovery never checks NEW out, and asks
    no guard: the release asked them all before its cutover.

    The live configuration is compared with NEW's configuration snapshot unless ``saved`` is False:
    the caller's readback of its path found no such name, which alone proves it was not published.
    The release publishes it before its cutover, so without it PREV is read back without the
    configuration and, when that holds, kept with no host step. Any other readback then counts the
    configuration as not held.
    """
    steps: list[str] = []
    readbacks: list[Readback] = []
    reads = READBACKS if saved else _CONFIGURATION_NOT_HELD
    try:
        head = host.checkout_head()
        if head not in (pins.new, pins.prev):
            raise ReadbackFailed(f"the checkout is at {head}, neither PREV nor NEW")
        first = _NO_CONFIGURATION if head == pins.prev and not saved else reads
        _require(_read_back(host, head, steps, readbacks, first))
    except BaseException as failure:
        steps.append("restore")
        restore_errors = _restore(host, pins)
        restored = _read_back(host, pins.prev, steps, readbacks, reads).ok
        return ReleaseResult(
            "restored" if restored else "failed",
            tuple(steps),
            (),
            tuple(readbacks),
            f"{type(failure).__name__}: {failure}",
            restore_errors,
        )
    outcome = "released" if head == pins.new else "restored"
    return ReleaseResult(outcome, tuple(steps), (), tuple(readbacks))


def _snapshot(host: ReleaseHost, pins: Pins) -> None:
    host.take_snapshot(pins.snapshot)
    for name in host.snapshots()[:-KEEP_SNAPSHOTS]:
        host.delete_snapshot(name)


def _forward(host: ReleaseHost, pins: Pins) -> str:
    """Bring both units up on NEW, so the readback after the move sees NEW running."""
    host.stop_units(PLATFORM_UNITS)
    host.checkout(pins.new)
    host.start_units(PLATFORM_UNITS)
    return pins.new


# Going back brings both units up on PREV and its configuration; a restore takes the same steps.
BACK_STEPS: tuple[Callable[[ReleaseHost, Pins], None], ...] = (
    lambda host, pins: host.stop_units(PLATFORM_UNITS),
    lambda host, pins: host.checkout(pins.prev),
    lambda host, pins: host.restore_config(),
    lambda host, pins: host.start_units(PLATFORM_UNITS),
)


def _back(host: ReleaseHost, pins: Pins) -> str:
    """The back move. Like any move, a step that fails here sets off the failure rule."""
    for step in BACK_STEPS:
        step(host, pins)
    return pins.prev


def _restore(host: ReleaseHost, pins: Pins) -> tuple[str, ...]:
    """Take every step of going back, even after one fails, and return each failure.

    Only the restore contains its own failures, so the readback of PREV after it always runs.
    """
    errors: list[str] = []
    for step in BACK_STEPS:
        try:
            step(host, pins)
        except BaseException as error:
            errors.append(f"{type(error).__name__}: {error}")
    return tuple(errors)


MOVES: dict[str, Callable[[ReleaseHost, Pins], str]] = {"forward": _forward, "back": _back}


def _read_back(
    host: ReleaseHost,
    expected: str,
    steps: list[str],
    readbacks: list[Readback],
    reads: Sequence[tuple[str, Callable[[ReleaseHost, str], bool]]] = READBACKS,
) -> Readback:
    steps.append("readback")
    errors: dict[str, str] = {}
    checks = {name: _holds(name, check, host, expected, errors) for name, check in reads}
    readback = Readback(expected, checks, errors)
    readbacks.append(readback)
    return readback


def _holds(
    name: str,
    check: Callable[[ReleaseHost, str], bool],
    host: ReleaseHost,
    expected: str,
    errors: dict[str, str],
) -> bool:
    """A read that raises counts as a failed check, so it can neither pass nor skip a readback.

    Its cause goes into `errors` under the check's name.
    """
    try:
        return check(host, expected)
    except BaseException as error:
        errors[name] = f"{type(error).__name__}: {error}"
        return False


def _require(readback: Readback) -> None:
    if not readback.ok:
        failed = ", ".join(name for name, ok in readback.checks.items() if not ok)
        raise ReadbackFailed(f"{failed} did not hold on {readback.expected}")
