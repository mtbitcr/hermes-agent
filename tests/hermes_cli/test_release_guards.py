"""T28 and T29: the release guards G1 to G11 in prepare mode, and a release refusing on each.

Every test drives the guards through FakeHost, an in-memory release host, so no real host
command runs.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from hermes_cli.release_guards import GATEWAY_UNIT, OpenRun, Pins, RecordedMerge, prepare
from hermes_cli.release_runner import run_release

GiB = 1024**3
PREV = "prev-sha"
FIRST_MERGE = "first-merge-sha"
NEW = "second-merge-sha"
# A change merged after Accept: origin main has moved past NEW.
LATER = "later-merge-sha"
CHECKOUT = "/srv/hermes/checkout"
ALL_GUARDS = [f"G{number}" for number in range(1, 12)]

PINS = Pins(new=NEW, prev=PREV)
# A batch of two merges: NEW's first parent is the first merge, whose first parent is PREV.
MERGES = (
    RecordedMerge(
        FIRST_MERGE, reviewed_base=PREV, reviewed_head="feature-a-sha", reviewed_tree="first-tree"
    ),
    RecordedMerge(
        NEW, reviewed_base=FIRST_MERGE, reviewed_head="feature-b-sha", reviewed_tree="new-tree"
    ),
)
# NEW as the continuation of a returned delivery review. It is built on the earlier delivered
# head, so its publish binds that head as the reviewed base; GitHub merges it with main as the
# first parent, so that parent, the first merge, is a newer main than the reviewed base.
CONTINUATION_MERGES = (
    MERGES[0],
    RecordedMerge(
        NEW,
        reviewed_base="delivered-head-sha",
        reviewed_head="feature-b-sha",
        reviewed_tree="new-tree",
    ),
)


@dataclass
class FakeHost:
    """A release host in memory on which every guard passes; it logs any write asked of it."""

    head: str = PREV
    clean: bool = True
    origin: str = NEW
    chain: list[str] = field(default_factory=lambda: [NEW, FIRST_MERGE])
    parents: dict[str, tuple[str, ...]] = field(
        default_factory=lambda: {
            FIRST_MERGE: (PREV, "feature-a-sha"),
            NEW: (FIRST_MERGE, "feature-b-sha"),
        }
    )
    trees: dict[str, str] = field(
        default_factory=lambda: {FIRST_MERGE: "first-tree", NEW: "new-tree"}
    )
    prev_is_ancestor: bool = True
    # (ancestor, descendant) pairs on the history besides PREV to NEW.
    history: set[tuple[str, str]] = field(default_factory=lambda: {(NEW, LATER)})
    changed: set[str] = field(default_factory=lambda: {"hermes_cli/main.py"})
    venv_root: str = CHECKOUT
    gateway: dict[str, str] = field(
        default_factory=lambda: {"KillSignal": "SIGINT", "KillMode": "mixed"}
    )
    free_bytes: int = 5 * GiB
    snapshot_dirs: list[str] = field(default_factory=lambda: ["release-1"])
    # A run whose worker died stays listed as open; G10 has to look past it.
    runs: list[OpenRun] = field(
        default_factory=lambda: [OpenRun("crashed-run", worker_alive=False)]
    )
    config: dict[str, str] = field(default_factory=lambda: {"config.yaml": "digest-1"})
    writes: list[tuple] = field(default_factory=list)

    def checkout_head(self):
        return self.head

    def checkout_is_clean(self):
        return self.clean

    def origin_main(self):
        return self.origin

    def first_parent_chain(self, prev, new):
        return list(self.chain)

    def commit_parents(self, commit):
        return self.parents.get(commit, ())

    def commit_tree(self, commit):
        return self.trees.get(commit, "")

    def is_ancestor(self, ancestor, descendant):
        if (ancestor, descendant) == (PREV, NEW):
            return self.prev_is_ancestor
        # As in git, a commit is its own ancestor.
        return ancestor == descendant or (ancestor, descendant) in self.history

    def changed_paths(self, prev, new):
        return set(self.changed)

    def checkout_root(self):
        return CHECKOUT

    def venv_import_root(self):
        return self.venv_root

    def unit_property(self, unit, name):
        return self.gateway[name] if unit == GATEWAY_UNIT else ""

    def free_disk_bytes(self):
        return self.free_bytes

    def snapshots(self):
        return list(self.snapshot_dirs)

    def open_native_runs(self):
        return list(self.runs)

    def live_config(self):
        return dict(self.config)

    def named_config_snapshot(self):
        return {"config.yaml": "digest-1"}

    # Neither prepare mode nor a refused release may reach any of these.
    def stop_units(self, units):
        self.writes.append(("stop_units", units))

    def start_units(self, units):
        self.writes.append(("start_units", units))

    def checkout(self, commit):
        self.writes.append(("checkout", commit))

    def restore_config(self):
        self.writes.append(("restore_config",))

    def take_snapshot(self, name):
        self.writes.append(("take_snapshot", name))

    def delete_snapshot(self, name):
        self.writes.append(("delete_snapshot", name))


def refusal(guard, case_id, merges=MERGES, **host_setup):
    return pytest.param(guard, merges, host_setup, id=case_id)


def test_prepare_mode_runs_every_guard_and_writes_nothing():
    healthy = FakeHost()
    failing = FakeHost(clean=False, free_bytes=0)

    healthy_results = prepare(healthy, PINS, MERGES)
    failing_results = prepare(failing, PINS, MERGES)

    assert [(result.guard, result.ok) for result in healthy_results] == [
        (guard, True) for guard in ALL_GUARDS
    ]
    # Failing guards do not cut the pass short: every guard after them is still asked.
    assert [result.guard for result in failing_results] == ALL_GUARDS
    assert [result.guard for result in failing_results if not result.ok] == ["G1", "G8"]
    assert healthy.writes == failing.writes == []


def test_g2_accepts_an_origin_main_that_moved_past_new():
    # A change that merges after Accept joins the next decision; it does not refuse this release.
    host = FakeHost(origin=LATER)

    results = prepare(host, PINS, MERGES)

    assert [(result.guard, result.ok) for result in results] == [
        (guard, True) for guard in ALL_GUARDS
    ]
    assert host.writes == []


def test_g3_accepts_a_merge_built_on_a_newer_main_than_its_reviewed_base():
    # Main moved on after review, but the merge still holds exactly what review saw: the reviewed
    # head as its second parent, and the reviewed tree.
    host = FakeHost()

    results = prepare(host, PINS, CONTINUATION_MERGES)

    assert [(result.guard, result.ok) for result in results] == [
        (guard, True) for guard in ALL_GUARDS
    ]
    assert host.writes == []


@pytest.mark.parametrize(
    ("guard", "merges", "host_setup"),
    [
        refusal("G1", "G1-checkout-not-at-prev", head="other-sha"),
        refusal("G1", "G1-checkout-not-clean", clean=False),
        refusal("G2", "G2-new-not-on-the-history-of-origin-main", origin="rewritten-sha"),
        refusal("G3", "G3-merge-on-chain-not-in-batch", merges=MERGES[1:]),
        refusal(
            "G3",
            "G3-merge-parents-not-reviewed",
            parents={FIRST_MERGE: (PREV, "feature-a-sha"), NEW: (FIRST_MERGE, "unreviewed-sha")},
        ),
        refusal(
            "G3", "G3-merge-tree-not-reviewed", trees={FIRST_MERGE: "other-tree", NEW: "new-tree"}
        ),
        # Each merge matches its own record, but a direct commit between them was never reviewed.
        refusal(
            "G3",
            "G3-unreviewed-commit-between-merges",
            merges=(
                MERGES[0],
                RecordedMerge(
                    NEW,
                    reviewed_base="direct-sha",
                    reviewed_head="feature-b-sha",
                    reviewed_tree="new-tree",
                ),
            ),
            chain=[NEW, "direct-sha", FIRST_MERGE],
            parents={
                FIRST_MERGE: (PREV, "feature-a-sha"),
                "direct-sha": (FIRST_MERGE,),
                NEW: ("direct-sha", "feature-b-sha"),
            },
        ),
        # NEW merged onto a newer main than its reviewed base must still be a two-parent merge of
        # the reviewed head with exactly the reviewed tree.
        refusal(
            "G3",
            "G3-newer-main-merge-tree-not-reviewed",
            merges=CONTINUATION_MERGES,
            trees={FIRST_MERGE: "first-tree", NEW: "other-tree"},
        ),
        refusal(
            "G3",
            "G3-newer-main-merge-second-parent-not-reviewed",
            merges=CONTINUATION_MERGES,
            parents={FIRST_MERGE: (PREV, "feature-a-sha"), NEW: (FIRST_MERGE, "unreviewed-sha")},
        ),
        # A squash or rebase merge onto main leaves one parent, even with the reviewed tree.
        refusal(
            "G3",
            "G3-newer-main-commit-with-one-parent",
            merges=CONTINUATION_MERGES,
            parents={FIRST_MERGE: (PREV, "feature-a-sha"), NEW: (FIRST_MERGE,)},
        ),
        refusal("G4", "G4-prev-not-ancestor-of-new", prev_is_ancestor=False),
        refusal("G5", "G5-lock-file-changed", changed={"hermes_cli/main.py", "uv.lock"}),
        refusal(
            "G5", "G5-project-manifest-changed", changed={"hermes_cli/main.py", "pyproject.toml"}
        ),
        refusal("G6", "G6-venv-imports-another-checkout", venv_root="/srv/hermes/old-checkout"),
        refusal(
            "G7",
            "G7-gateway-stops-with-sigterm",
            gateway={"KillSignal": "SIGTERM", "KillMode": "mixed"},
        ),
        refusal(
            "G7",
            "G7-gateway-kill-mode-not-mixed",
            gateway={"KillSignal": "SIGINT", "KillMode": "control-group"},
        ),
        refusal("G8", "G8-less-than-4-gib-free", free_bytes=4 * GiB - 1),
        refusal("G9", "G9-release-snapshot-already-exists", snapshot_dirs=["release-1", NEW]),
        refusal(
            "G10",
            "G10-native-run-open",
            runs=[
                OpenRun("crashed-run", worker_alive=False),
                OpenRun("live-run", worker_alive=True),
            ],
        ),
        refusal("G11", "G11-live-config-drifted", config={"config.yaml": "digest-2"}),
    ],
)
def test_each_guard_refuses(guard, merges, host_setup):
    host = FakeHost(**host_setup)

    results = prepare(host, PINS, merges)
    release = run_release(host, PINS, 2, merges)

    assert [result.guard for result in results if not result.ok] == [guard]
    # The release unit re-runs every guard fresh and refuses before cutover, writing nothing.
    assert (release.outcome, release.steps) == ("refused", ("guards",))
    assert host.writes == []
