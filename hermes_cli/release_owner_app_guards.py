"""Owner app guards: the owner app forms of the release guards, asked before the owner app cutover.

A batch's owner app part replaces the running owner app container with one made from APP_NEW's
image. These are the owner app forms of G1 to G5 and of G8's room for the build (plan section 2),
asked as release_guards.prepare asks the platform's guards: in prepare and again, fresh, just
before the owner app cutover, each with its rule in plain words, so a refusal shows why. A guard
whose host value cannot be read fails, and the guards after it are still asked. The Pins here
are the owner app part's: `prev` is APP_PREV and `new` is APP_NEW.

The guards only read: every OwnerAppReader method is a read, and no guard starts, stops, replaces,
writes or repairs anything. Nothing calls this module yet; the release step that asks it belongs
to a later card.
"""

from __future__ import annotations

from collections.abc import Callable, Collection, Sequence
from typing import Protocol

from hermes_cli import release_guards
from hermes_cli.release_guards import MIN_FREE_DISK_BYTES, GuardResult, Pins, RecordedMerge

# Owner app images are tagged with the commit they are built from: IMAGE:<commit>.
IMAGE = "raphael-workspace"
# The page check move reuses the old copy's installed packages, which fit only an unchanged lock,
# and the release does not install the page check's packages itself.
PAGE_CHECK_LOCK = "tools/page-check/package-lock.json"


class OwnerAppReader(Protocol):
    """What the owner app guards may ask of the release host. Every method only reads."""

    def running_revision_label(self) -> str | None:
        """The running owner app container's revision label; None when none runs."""

    def running_image(self) -> str | None:
        """The running owner app container's image, as name:tag; None when none runs."""

    def on_main(self, commit: str) -> bool:
        """True when `commit` is on the history of the owner app repository's main, read through
        the fenced transport."""

    # The owner app copy's history, asked as HostReader asks the platform checkout's.
    def first_parent_chain(self, prev: str, new: str) -> Sequence[str]:
        """Commits on `new`'s first-parent chain, newest first, stopping before `prev`."""

    def commit_parents(self, commit: str) -> tuple[str, ...]: ...

    def commit_tree(self, commit: str) -> str: ...

    def is_ancestor(self, ancestor: str, descendant: str) -> bool:
        """True when `ancestor` is on the history of `descendant`.

        As in git, a commit is its own ancestor.
        """

    def changed_paths(self, prev: str, new: str) -> Collection[str]:
        """Repository-relative paths that differ between the two commits."""

    def docker_free_bytes(self) -> int:
        """Free space on the filesystem holding Docker's storage."""

    # The platform's state snapshots and environment, read as HostReader reads them for G8.
    def snapshots(self) -> Sequence[str]:
        """Names of the state snapshots on disk, oldest first."""

    def snapshot_bytes(self, name: str) -> int: ...

    def environment_snapshot_bytes(self) -> int | None:
        """Full size of the platform environment, which a release copies; None when the
        environment is not found."""


AppGuardCheck = Callable[[OwnerAppReader, Pins, Sequence[RecordedMerge], int], bool]


def _platform_check(guard: str) -> AppGuardCheck:
    # Where the owner app form is the platform's rule asked of the owner app copy, it is the
    # platform's own check, not a second copy of its rule that could drift from it.
    [check] = [check for name, _rule, check in release_guards.GUARDS if name == guard]
    return lambda reader, pins, merges, build_reserve_bytes: check(reader, pins, merges)


def _room_for_the_build(
    reader: OwnerAppReader, pins: Pins, merges: Sequence[RecordedMerge], build_reserve_bytes: int
) -> bool:
    # The build must never take the room G8 keeps for the platform's next state snapshot: the
    # newest snapshot's size, the environment's full size and G8's own margin. Only Docker's free
    # space counts, no older snapshot's room, since whether Docker's storage shares the snapshots'
    # filesystem is open (plan question Q2). Without the environment's size the need is unknown.
    # A reserve below zero would hand the build part of G8's own need, so it fails.
    if build_reserve_bytes < 0:
        return False
    environment = reader.environment_snapshot_bytes()
    if environment is None:
        return False
    names = reader.snapshots()
    newest = reader.snapshot_bytes(names[-1]) if names else 0
    need = newest + environment + MIN_FREE_DISK_BYTES
    return reader.docker_free_bytes() >= need + build_reserve_bytes


GUARDS: tuple[tuple[str, str, AppGuardCheck], ...] = (
    (
        "G1",
        "the running owner app container's revision label is APP_PREV and its image is"
        f" {IMAGE}:APP_PREV",
        lambda reader, pins, merges, build_reserve_bytes: (
            reader.running_revision_label() == pins.prev
            and reader.running_image() == f"{IMAGE}:{pins.prev}"
        ),
    ),
    (
        "G2",
        "APP_NEW is on the history of the owner app repository's main, read through the fenced"
        " transport",
        lambda reader, pins, merges, build_reserve_bytes: reader.on_main(pins.new),
    ),
    (
        "G3",
        "every commit from APP_PREV to APP_NEW on the owner app copy's first-parent chain is a"
        " batch merge that matches its own review",
        _platform_check("G3"),
    ),
    (
        "G4",
        "APP_PREV is an ancestor of APP_NEW in the owner app copy",
        _platform_check("G4"),
    ),
    (
        "G5",
        f"the page check's lock, {PAGE_CHECK_LOCK}, does not differ between APP_PREV and APP_NEW",
        lambda reader, pins, merges, build_reserve_bytes: (
            PAGE_CHECK_LOCK not in reader.changed_paths(pins.prev, pins.new)
        ),
    ),
    (
        "G8",
        "the free space on Docker's storage is at least G8's own need plus the build reserve:"
        " G8's own need is the size of the newest state snapshot plus the full size of the"
        f" platform environment plus {MIN_FREE_DISK_BYTES // 1024**3} GiB; with no state snapshot,"
        " the newest counts as 0 bytes, and an environment that is not found fails; a build"
        " reserve below zero fails",
        _room_for_the_build,
    ),
)


def _holds(
    check: AppGuardCheck,
    reader: OwnerAppReader,
    pins: Pins,
    merges: Sequence[RecordedMerge],
    build_reserve_bytes: int,
) -> bool:
    # A guard whose host value cannot be read, say because the read timed out, fails, as the
    # platform's G12 fails an environment whose lock cannot be read; the pass goes on to the next
    # guard. Nothing of the error is kept, since it may name a host (decision D-10). A
    # process-control exception, such as KeyboardInterrupt, is not an Exception: it still stops
    # the pass.
    try:
        return check(reader, pins, merges, build_reserve_bytes)
    except Exception:
        return False


def prepare(
    reader: OwnerAppReader,
    pins: Pins,
    merges: Sequence[RecordedMerge],
    *,
    build_reserve_bytes: int,
) -> tuple[GuardResult, ...]:
    """Ask the owner app part's guards, in order, and write nothing.

    As in release_guards.prepare, a failing guard does not stop the pass, so one refusal names
    every guard that failed; the release asks the same pass again, fresh, just before the owner
    app cutover. A guard whose host value cannot be read fails, and the pass goes on to the next
    guard. `merges` are the owner app part's recorded merges, with what review saw.
    `build_reserve_bytes` is the room the owner app build may take on Docker's storage: its
    number is the owner's to set (plan question Q2), so the caller passes it rather than this
    module guessing it. A reserve below zero fails G8.
    """
    return tuple(
        GuardResult(guard, rule, _holds(check, reader, pins, merges, build_reserve_bytes))
        for guard, rule, check in GUARDS
    )
