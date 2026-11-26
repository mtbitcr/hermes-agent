"""Card 5 of the owner app release plan: the owner app guards, asked before the owner app cutover.

Every test asks the guards of FakeAppHost, an in-memory release host, so no container, repository
copy or disk is read and nothing on any server changes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from subprocess import SubprocessError

import pytest

from hermes_cli.release_guards import MIN_FREE_DISK_BYTES, Pins, RecordedMerge
from hermes_cli.release_owner_app_guards import GUARDS, prepare

GiB = 1024**3
APP_PREV = "app-prev-sha"
FIRST_MERGE = "app-first-merge-sha"
APP_NEW = "app-second-merge-sha"
# The image the running owner app container must have been made from (plan section 2, G1).
APP_PREV_IMAGE = f"raphael-workspace:{APP_PREV}"
PAGE_CHECK_LOCK = "tools/page-check/package-lock.json"
ALL_GUARDS = ["G1", "G2", "G3", "G4", "G5", "G8"]
# Each guard's fixed rule, in the order the pass asks them.
RULES = [rule for _guard, rule, _check in GUARDS]
# What a read that cannot be done might say. None of it may reach a result: it is no rule, and it
# may name a host (decision D-10).
UNREADABLE = "no answer from 192.0.2.10 within 30 s"

NEWEST_SNAPSHOT = 13 * GiB
ENVIRONMENT = 2 * GiB
# Stands in for the build reserve, which the owner has yet to set (plan question Q2).
RESERVE = 3 * GiB
# G8's own need: the newest state snapshot, the platform environment and G8's own margin.
G8_NEED = NEWEST_SNAPSHOT + ENVIRONMENT + MIN_FREE_DISK_BYTES

PINS = Pins(new=APP_NEW, prev=APP_PREV)
# An owner app part of two merges: APP_NEW's first parent is the first merge, whose first parent
# is APP_PREV.
MERGES = (
    RecordedMerge(
        FIRST_MERGE,
        reviewed_base=APP_PREV,
        reviewed_head="feature-a-sha",
        reviewed_tree="first-tree",
    ),
    RecordedMerge(
        APP_NEW,
        reviewed_base=FIRST_MERGE,
        reviewed_head="feature-b-sha",
        reviewed_tree="new-tree",
    ),
)
# A commit no review saw, straight on the first-parent chain between the two merges.
CHAIN_WITH_DIRECT_COMMIT = [APP_NEW, "direct-commit-sha", FIRST_MERGE]


def read(method):
    """A read of FakeAppHost: each call is recorded in its ``reads``."""

    def recorded(self, *args):
        self.reads.append((method.__name__, *args))
        return method(self, *args)

    return recorded


@dataclass
class FakeAppHost:
    """The release host in memory, as the owner app guards read it; every guard passes on it.

    Its reads are its own methods, and each read is recorded in ``reads``. Any other call, such
    as an action that would start, stop, replace, write or repair something, is recorded in
    ``actions`` and refused.
    """

    # None when no owner app container runs.
    revision_label: str | None = APP_PREV
    image: str | None = APP_PREV_IMAGE
    # The commits on the history of the owner app repository's main.
    main: set[str] = field(default_factory=lambda: {APP_PREV, FIRST_MERGE, APP_NEW})
    chain: list[str] = field(default_factory=lambda: [APP_NEW, FIRST_MERGE])
    parents: dict[str, tuple[str, ...]] = field(
        default_factory=lambda: {
            FIRST_MERGE: (APP_PREV, "feature-a-sha"),
            APP_NEW: (FIRST_MERGE, "feature-b-sha"),
        }
    )
    trees: dict[str, str] = field(
        default_factory=lambda: {FIRST_MERGE: "first-tree", APP_NEW: "new-tree"}
    )
    prev_is_ancestor: bool = True
    # Other files of the page check may change; its lock may not.
    changed: set[str] = field(
        default_factory=lambda: {"src/app/decisions/page.tsx", "tools/page-check/pages.mjs"}
    )
    docker_free: int = G8_NEED + RESERVE
    # The platform's state snapshots, oldest first, with their sizes.
    snapshot_sizes: dict[str, int] = field(default_factory=lambda: {"release-1": NEWEST_SNAPSHOT})
    # None when the platform environment is not found.
    environment_size: int | None = ENVIRONMENT
    reads: list[tuple] = field(default_factory=list)
    actions: list[tuple] = field(default_factory=list)

    def __getattr__(self, name):
        # Only a name this host does not define comes here.
        if name.startswith("__"):
            raise AttributeError(name)

        def refused(*args, **kwargs):
            self.actions.append((name, args, kwargs))
            raise PermissionError(f"the owner app guards may only read, but asked for {name}")

        return refused

    @read
    def running_revision_label(self):
        return self.revision_label

    @read
    def running_image(self):
        return self.image

    @read
    def on_main(self, commit):
        return commit in self.main

    @read
    def first_parent_chain(self, prev, new):
        return list(self.chain)

    @read
    def commit_parents(self, commit):
        return self.parents.get(commit, ())

    @read
    def commit_tree(self, commit):
        return self.trees.get(commit, "")

    @read
    def is_ancestor(self, ancestor, descendant):
        if (ancestor, descendant) == (APP_PREV, APP_NEW):
            return self.prev_is_ancestor
        # As in git, a commit is its own ancestor.
        return ancestor == descendant

    @read
    def changed_paths(self, prev, new):
        return set(self.changed)

    @read
    def docker_free_bytes(self):
        return self.docker_free

    @read
    def snapshots(self):
        return list(self.snapshot_sizes)

    @read
    def snapshot_bytes(self, name):
        return self.snapshot_sizes[name]

    @read
    def environment_snapshot_bytes(self):
        return self.environment_size


def failing_everywhere() -> FakeAppHost:
    """A host on which every owner app guard fails."""
    return FakeAppHost(
        revision_label="someone-elses-sha",
        main={APP_PREV},
        chain=CHAIN_WITH_DIRECT_COMMIT,
        prev_is_ancestor=False,
        changed={PAGE_CHECK_LOCK},
        docker_free=0,
    )


def unreadable(host: FakeAppHost, name: str, error: BaseException) -> FakeAppHost:
    """``host`` with one read that cannot be done: it is recorded, then raises ``error``."""

    def raising(*args):
        host.reads.append((name, *args))
        raise error

    setattr(host, name, raising)
    return host


def ask(host: FakeAppHost, build_reserve_bytes: int = RESERVE):
    return prepare(host, PINS, MERGES, build_reserve_bytes=build_reserve_bytes)


def failed(results) -> list[str]:
    return [result.guard for result in results if not result.ok]


def test_every_owner_app_guard_passes_on_a_ready_host():
    host = FakeAppHost()

    results = ask(host)

    assert [(result.guard, result.ok) for result in results] == [
        (guard, True) for guard in ALL_GUARDS
    ]
    assert host.actions == []


def refusal(guard, case_id, **host_setup):
    return pytest.param(guard, host_setup, id=case_id)


@pytest.mark.parametrize(
    "guard, host_setup",
    [
        refusal("G1", "running-label-not-app-prev", revision_label="someone-elses-sha"),
        refusal("G1", "running-image-not-app-prev", image="raphael-workspace:someone-elses-sha"),
        refusal("G1", "no-running-container", revision_label=None, image=None),
        refusal("G2", "app-new-not-on-main", main={APP_PREV, FIRST_MERGE}),
        refusal("G3", "direct-commit-on-the-chain", chain=CHAIN_WITH_DIRECT_COMMIT),
        refusal("G4", "app-prev-not-an-ancestor", prev_is_ancestor=False),
        refusal(
            "G5", "page-check-lock-changed", changed={PAGE_CHECK_LOCK, "tools/page-check/pages.mjs"}
        ),
        refusal("G8", "too-little-space", docker_free=G8_NEED + RESERVE - 1),
        refusal("G8", "environment-not-found", environment_size=None),
    ],
)
def test_each_guard_fails_alone_for_its_own_case(guard, host_setup):
    host = FakeAppHost(**host_setup)

    results = ask(host)

    assert [result.guard for result in results] == ALL_GUARDS
    assert failed(results) == [guard]
    assert host.actions == []


def test_one_pass_names_every_failed_guard():
    host = failing_everywhere()

    results = ask(host)

    # A failing guard does not cut the pass short: every guard after it is still asked.
    assert failed(results) == ALL_GUARDS
    assert host.actions == []


def test_an_unreadable_guard_fails_and_the_pass_still_names_every_failed_guard():
    # The fenced read of main times out, the page check's lock changed, and Docker's storage is a
    # byte short.
    host = unreadable(
        FakeAppHost(changed={PAGE_CHECK_LOCK}, docker_free=G8_NEED + RESERVE - 1),
        "on_main",
        TimeoutError(UNREADABLE),
    )

    results = ask(host)

    assert [(result.guard, result.ok) for result in results] == [
        ("G1", True), ("G2", False), ("G3", True), ("G4", True), ("G5", False), ("G8", False)
    ]
    # Each result states its guard's fixed rule, and nothing of the error.
    assert [result.rule for result in results] == RULES
    assert UNREADABLE not in repr(results) and "TimeoutError" not in repr(results)
    assert host.actions == []


@pytest.mark.parametrize(
    "guard, name, error",
    [
        ("G1", "running_image", OSError(UNREADABLE)),
        ("G2", "on_main", TimeoutError(UNREADABLE)),
        # G3 and G4 are the platform's own checks, reading through the same reader.
        ("G3", "commit_tree", SubprocessError(UNREADABLE)),
        ("G4", "is_ancestor", SubprocessError(UNREADABLE)),
        ("G5", "changed_paths", SubprocessError(UNREADABLE)),
        # The last read of the pass: a failure there keeps every verdict before it.
        ("G8", "docker_free_bytes", OSError(UNREADABLE)),
    ],
    ids=ALL_GUARDS,
)
def test_a_guard_whose_host_value_cannot_be_read_fails_alone(guard, name, error):
    host = unreadable(FakeAppHost(), name, error)

    results = ask(host)

    assert [result.guard for result in results] == ALL_GUARDS
    assert failed(results) == [guard]
    assert [result.rule for result in results] == RULES
    assert host.actions == []


@pytest.mark.parametrize("stop", [KeyboardInterrupt, SystemExit, GeneratorExit])
def test_a_process_control_exception_still_stops_the_pass(stop):
    host = unreadable(FakeAppHost(), "on_main", stop())

    with pytest.raises(stop):
        ask(host)

    # The pass stopped at G2's read: no guard after it was asked.
    assert host.reads[-1] == ("on_main", APP_NEW)
    assert host.actions == []


def test_the_guards_only_read():
    ready, refusing = FakeAppHost(), failing_everywhere()

    ask(ready)
    ask(refusing)

    # Both hosts were read, and neither was asked anything but a read. A call that is not a read
    # is recorded in ``actions`` before it is refused, so the pass cannot hide it as a refusal.
    assert ready.reads and refusing.reads
    assert ready.actions == refusing.actions == []


def test_each_guard_states_its_rule_whether_it_passes_or_fails():
    passed, refused = ask(FakeAppHost()), ask(failing_everywhere())

    rules = [result.rule for result in passed]
    assert rules == [result.rule for result in refused]
    assert all(rule.strip() for rule in rules)
    assert len(set(rules)) == len(rules)


@pytest.mark.parametrize(
    "snapshot_sizes, docker_free, build_reserve_bytes, holds",
    [
        ({"release-1": 20 * GiB, "release-2": NEWEST_SNAPSHOT}, G8_NEED + RESERVE, RESERVE, True),
        (
            {"release-1": 20 * GiB, "release-2": NEWEST_SNAPSHOT},
            G8_NEED + RESERVE - 1,
            RESERVE,
            False,
        ),
        ({}, ENVIRONMENT + MIN_FREE_DISK_BYTES + RESERVE, RESERVE, True),
        ({}, ENVIRONMENT + MIN_FREE_DISK_BYTES + RESERVE - 1, RESERVE, False),
        # With no reserve, G8's own need alone must be free.
        ({"release-1": NEWEST_SNAPSHOT}, G8_NEED, 0, True),
        ({"release-1": NEWEST_SNAPSHOT}, G8_NEED - 1, 0, False),
        # A reserve below zero fails: it may not take back any of G8's own need.
        ({"release-1": NEWEST_SNAPSHOT}, G8_NEED, -1, False),
        ({"release-1": NEWEST_SNAPSHOT}, G8_NEED - 1, -1, False),
    ],
    ids=["newest-snapshot-just-fits", "newest-snapshot-a-byte-short", "no-snapshot-just-fits",
         "no-snapshot-a-byte-short", "zero-reserve-just-fits", "zero-reserve-a-byte-short",
         "reserve-below-zero-with-g8s-own-need-free", "reserve-below-zero-a-byte-short"],
)
def test_g8_keeps_g8s_own_need_and_the_build_reserve_free_on_dockers_storage(
    snapshot_sizes, docker_free, build_reserve_bytes, holds
):
    # The newest state snapshot is the last one; with none, it counts as 0 bytes. An older one is
    # not counted as room: Docker's storage may be another disk (plan question Q2).
    host = FakeAppHost(snapshot_sizes=snapshot_sizes, docker_free=docker_free)

    results = ask(host, build_reserve_bytes)

    assert failed(results) == ([] if holds else ["G8"])


def test_the_pass_before_the_cutover_reads_the_host_again():
    host = FakeAppHost()
    in_prepare = ask(host)

    # Between prepare and the owner app cutover, someone replaced the running container.
    host.revision_label = "someone-elses-sha"
    before_cutover = ask(host)

    assert failed(in_prepare) == []
    assert failed(before_cutover) == ["G1"]
