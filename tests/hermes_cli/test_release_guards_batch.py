"""Card 2, pure: two merges that each land on the base their review saw pass G3 in one batch, and a merge
whose change was built on an older main is refused, by G3 and by the arm's own decision. Real git in a
temporary repository answers G3's questions; release_guards.prepare is unchanged."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from hermes_cli import kanban_delivery as kd
from hermes_cli.release_guards import Pins, RecordedMerge, prepare
from tests.hermes_cli.test_kanban_delivery import _git


class GitHost:
    """A release host whose history questions git answers from ``repo``; every other answer is benign."""

    def __init__(self, repo: Path, prev: str, new: str):
        self.repo, self.prev, self.new = repo, prev, new

    def checkout_head(self):
        return self.prev

    def checkout_is_clean(self):
        return True

    def origin_main(self):
        return self.new

    def first_parent_chain(self, prev, new):
        return _git(self.repo, "rev-list", "--first-parent", f"{prev}..{new}").split()

    def commit_parents(self, commit):
        return tuple(_git(self.repo, "rev-list", "--parents", "-n", "1", commit).split()[1:])

    def commit_tree(self, commit):
        return _git(self.repo, "rev-parse", f"{commit}^{{tree}}")

    def is_ancestor(self, ancestor, descendant):
        return subprocess.run(["git", "-C", str(self.repo), "merge-base", "--is-ancestor", ancestor, descendant],
                              capture_output=True).returncode == 0

    def changed_paths(self, prev, new):
        return set(_git(self.repo, "diff", "--name-only", prev, new).split())

    def checkout_root(self):
        return "/srv/checkout"

    def venv_import_root(self):
        return "/srv/checkout"

    def unit_property(self, unit, name):
        return {"KillSignal": "SIGINT", "KillMode": "mixed"}[name]

    def free_disk_bytes(self):
        return 1 << 40

    def snapshots(self):
        return []

    def snapshot_bytes(self, name):
        return 0

    def open_native_runs(self):
        return []

    def live_config(self):
        return {}

    def named_config_snapshot(self):
        return {}


@pytest.fixture
def history(tmp_path):
    """Base T0 on main; change A built on T0; the repository and a commit maker."""
    repo = tmp_path / "repo"
    subprocess.run(["git", "init", "-q", "-b", "main", str(repo)], check=True, capture_output=True)

    def change(start: str, name: str) -> str:
        _git(repo, "checkout", "-q", "--detach", start)
        (repo / f"{name}.txt").write_text(f"{name}\n", encoding="utf-8")
        _git(repo, "add", f"{name}.txt")
        _git(repo, "commit", "-q", "-m", name)
        return _git(repo, "rev-parse", "HEAD")

    def merge(onto: str, head: str) -> str:
        """GitHub's merge: a two-parent commit on ``onto`` with ``head`` second."""
        _git(repo, "checkout", "-q", "--detach", onto)
        _git(repo, "merge", "-q", "--no-ff", "--no-edit", head)
        return _git(repo, "rev-parse", "HEAD")

    (repo / "README.md").write_text("base\n", encoding="utf-8")
    _git(repo, "add", "README.md")
    _git(repo, "commit", "-q", "-m", "T0")
    t0 = _git(repo, "rev-parse", "HEAD")
    return repo, t0, change, merge


def _record(repo, merge, base, head):
    return RecordedMerge(merge, reviewed_base=base, reviewed_head=head,
                         reviewed_tree=_git(repo, "rev-parse", f"{head}^{{tree}}"))


def _g3(repo, prev, new, merges):
    return next(result.ok for result in prepare(GitHost(repo, prev, new), Pins(new=new, prev=prev), merges)
                if result.guard == "G3")


def test_two_merges_on_their_reviewed_bases_pass_g3_in_one_batch(history):
    repo, t0, change, merge = history
    a = change(t0, "a")
    m1 = merge(t0, a)
    b = change(m1, "b")
    assert kd.base_allows_merge(repo, m1, m1, b)  # at B's arm main is M1, the base its review saw
    m2 = merge(m1, b)

    assert _g3(repo, t0, m2, [_record(repo, m1, t0, a), _record(repo, m2, m1, b)]) is True


def test_a_continuation_whose_head_contains_main_is_allowed_and_passes_g3(history):
    """Owner question 1: the recorded base is older than main, but main is already in the approved head, so
    the merge result is byte for byte what review approved."""
    repo, t0, change, merge = history
    a = change(t0, "a")
    m1 = merge(t0, a)
    continued = change(m1, "c")

    assert kd.base_allows_merge(repo, t0, m1, continued)
    m2 = merge(m1, continued)
    assert _g3(repo, t0, m2, [_record(repo, m1, t0, a), _record(repo, m2, t0, continued)]) is True


def test_a_change_built_on_t0_and_merged_onto_m1_is_refused_by_g3_and_by_the_arm(history):
    """The wrong-tree control, finding 150: B' was reviewed on T0, but main is M1 when it would merge."""
    repo, t0, change, merge = history
    a = change(t0, "a")
    m1 = merge(t0, a)
    stale = change(t0, "b")

    assert kd.base_allows_merge(repo, t0, m1, stale) is False  # the arm records base_moved and sends nothing
    m2 = merge(m1, stale)
    assert _g3(repo, t0, m2, [_record(repo, m1, t0, a), _record(repo, m2, t0, stale)]) is False


@pytest.mark.parametrize("main, workdir", [("f" * 40, "repo"), ("main", None), ("main", "missing")],
                         ids=["unknown_commit", "no_repository", "missing_repository"])
def test_any_doubt_about_containment_is_not_contained(history, tmp_path, main, workdir):
    repo, t0, change, merge = history
    head = change(t0, "b")
    main = t0 if main == "main" else main
    where = {"repo": repo, None: None, "missing": tmp_path / "gone"}[workdir]

    assert kd.base_allows_merge(where, "a" * 40, main, head) is False
