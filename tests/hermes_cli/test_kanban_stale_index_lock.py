"""A stale git lock in a task worktree no longer stalls kernel delivery (finding 163).

git holds ``index.lock`` only while a git process runs. A git killed mid-write (the gateway
stop kills its children) leaves the lock behind in the task worktree's own git folder, and
every later kernel git write failed, the rollback included. The kernel now clears that one
lock before a write, and only when it is provably stale: a plain file at least the age floor
old, no git process running on the host (a failed probe counts as running), and still the
same file right before removal. Anything else keeps the lock and the write fails as before.
Read-only kernel git runs with ``--no-optional-locks``, so it never takes the lock itself.

Every test uses a real repository with a linked worktree and patches the strict process
probe, so other git processes on the host never decide an outcome.
"""

from __future__ import annotations

import logging
import os
import subprocess
import time
from pathlib import Path
from typing import NamedTuple

import pytest

from hermes_cli import gitlock
from hermes_cli import kanban_db as kb

_PROBE = "_git_proc_running_strict"
_STALE_AGE = 20 * 60
_IDENTITY = ("-c", "user.name=Hermes Test", "-c", "user.email=test@localhost", "-c", "commit.gpgsign=false")
_PATCH = """\
diff --git a/tracked.txt b/tracked.txt
--- a/tracked.txt
+++ b/tracked.txt
@@ -1 +1,2 @@
 one
+two
"""


class _TaskWorktree(NamedTuple):
    main: Path  # the main checkout
    path: Path  # the task's linked worktree
    git_dir: Path  # the worktree's own git folder, <main>/.git/worktrees/<name>
    patch: Path  # a patch the kernel can apply to the worktree


class _Probe:
    """Stands in for the strict git-process probe; ``answer`` decides what it reports."""

    def __init__(self, real) -> None:
        self.real = real
        self.calls = 0
        self.answer = lambda: False  # no git process runs on the host

    def __call__(self) -> bool:
        self.calls += 1
        return self.answer()


def _git(cwd: Path, *args: str) -> str:
    """Setup git with an explicit identity (the host has none); returns stripped stdout."""
    result = subprocess.run(["git", "-C", str(cwd), *_IDENTITY, *args], capture_output=True, text=True, check=True)
    return result.stdout.strip()


def _plant_lock(git_dir: Path, age_seconds: float) -> Path:
    """Leave an empty ``index.lock`` behind, as a killed git does, ``age_seconds`` old."""
    lock = git_dir / "index.lock"
    lock.write_bytes(b"")
    then = time.time() - age_seconds
    os.utime(lock, (then, then))
    return lock


def _identity(path: Path) -> tuple[int, int]:
    st = path.lstat()
    return st.st_ino, st.st_mtime_ns


def _assert_the_write_fails_as_before(wt: _TaskWorktree, lock: Path) -> None:
    """The kernel write refuses on the lock exactly as before the fix, and the lock stays."""
    with pytest.raises(kb.WorktreeScopeError, match=r"git materialization command failed .*index\.lock"):
        kb._git_mutation(wt.path, "apply", "--index", str(wt.patch))
    assert os.path.lexists(lock)


@pytest.fixture(autouse=True)
def probe(monkeypatch: pytest.MonkeyPatch) -> _Probe:
    """Patch the strict probe where ``clear_stale_index_lock`` reads it, in every test.

    CI hosts run other git processes in parallel, so the real probe would make these tests
    flaky. ``raising=False`` lets a tree without the probe fail on its behaviour, not here.
    """
    fake = _Probe(getattr(gitlock, _PROBE, None))
    monkeypatch.setattr(gitlock, _PROBE, fake, raising=False)
    return fake


@pytest.fixture
def task_worktree(tmp_path: Path) -> _TaskWorktree:
    """A real repository with a clean linked task worktree and a patch for it."""
    main = tmp_path / "repo"
    main.mkdir()
    _git(main, "init", "-q")
    (main / "tracked.txt").write_text("one\n", encoding="utf-8")
    _git(main, "add", "tracked.txt")
    _git(main, "commit", "-q", "-m", "base")
    path = tmp_path / "worktrees" / "task"
    _git(main, "worktree", "add", "-q", "-b", "task/stale-lock", str(path))
    git_dir = Path(_git(path, "rev-parse", "--path-format=absolute", "--git-dir")).resolve()
    assert git_dir != (main / ".git").resolve()
    patch = tmp_path / "change.patch"
    patch.write_text(_PATCH, encoding="utf-8")
    return _TaskWorktree(main, path, git_dir, patch)


# Fail before the fix.


def test_a_stale_index_lock_no_longer_refuses_a_kernel_git_write(task_worktree, probe, caplog):
    wt = task_worktree
    lock = _plant_lock(wt.git_dir, _STALE_AGE)

    with caplog.at_level(logging.WARNING, logger=gitlock.__name__):
        kb._git_mutation(wt.path, "apply", "--index", str(wt.patch))

    assert not os.path.lexists(lock)
    assert _git(wt.path, "diff", "--cached", "--name-only") == "tracked.txt"
    assert probe.calls, "the lock went without the process probe being asked"
    assert any(r.levelno == logging.WARNING and str(lock) in r.getMessage() for r in caplog.records)


def test_the_kernel_rollback_clears_a_stale_index_lock(task_worktree):
    wt = task_worktree
    head = _git(wt.path, "rev-parse", "HEAD")
    (wt.path / "tracked.txt").write_text("half written\n", encoding="utf-8")
    lock = _plant_lock(wt.git_dir, _STALE_AGE)

    kb._git_mutation(wt.path, "reset", "--hard", head)

    assert not os.path.lexists(lock)
    assert _git(wt.path, "status", "--porcelain") == ""
    assert (wt.path / "tracked.txt").read_text(encoding="utf-8") == "one\n"


def test_kernel_read_only_git_leaves_the_index_alone(task_worktree):
    wt = task_worktree
    index = wt.git_dir / "index"
    then = time.time() - 3600
    os.utime(wt.path / "tracked.txt", (then, then))  # same content, new stat: the index is stat-dirty
    os.utime(index, (then - 60, then - 60))
    before = (index.read_bytes(), index.stat().st_mtime_ns)

    assert kb._git_output(wt.path, "status", "--porcelain=v1", "-z", binary=True) == b""

    assert (index.read_bytes(), index.stat().st_mtime_ns) == before, "read-only kernel git rewrote the index"
    assert not os.path.lexists(wt.git_dir / "index.lock")


def test_kernel_read_only_git_runs_carry_no_optional_locks(task_worktree, monkeypatch):
    wt = task_worktree
    seen: list[list[str]] = []
    real_run = subprocess.run

    def recording_run(argv, *args, **kwargs):
        seen.append(list(argv))
        return real_run(argv, *args, **kwargs)

    monkeypatch.setattr(kb.subprocess, "run", recording_run)
    kb._git_output(wt.path, "status", "--porcelain=v1", "-z", binary=True)
    kb._git_output(wt.path, "diff", "--cached", "--name-only", "-z", binary=True)
    kb._git_output(wt.path, "rev-parse", "--verify", "HEAD")

    assert len(seen) == 3
    for argv in seen:
        assert argv[:4] == ["git", "-C", str(wt.path), "--no-optional-locks"], argv


# Guards: the lock stays and the write fails as before.


def test_a_lock_is_kept_while_any_git_process_runs(task_worktree, probe):
    wt = task_worktree
    probe.answer = lambda: True
    lock = _plant_lock(wt.git_dir, _STALE_AGE)
    planted = _identity(lock)

    _assert_the_write_fails_as_before(wt, lock)

    assert gitlock.clear_stale_index_lock(wt.git_dir) is False
    assert _identity(lock) == planted
    assert probe.calls


def test_a_young_lock_is_kept(task_worktree):
    wt = task_worktree
    lock = _plant_lock(wt.git_dir, gitlock.STALE_LOCK_MIN_AGE_SECONDS - 60)
    planted = _identity(lock)

    _assert_the_write_fails_as_before(wt, lock)

    assert gitlock.clear_stale_index_lock(wt.git_dir) is False
    assert _identity(lock) == planted


def _raise_probe_error() -> bool:
    raise OSError("process table unreadable")


def _break_the_process_listing(monkeypatch: pytest.MonkeyPatch, failure: str) -> None:
    """Make the probe's own process listing fail; every other command runs for real."""
    real_run = gitlock.subprocess.run

    def run(argv, *args, **kwargs):
        if not (isinstance(argv, (list, tuple)) and argv and argv[0] in ("pgrep", "tasklist")):
            return real_run(argv, *args, **kwargs)
        if failure == "missing-tool":
            raise FileNotFoundError(2, "No such file or directory", argv[0])
        if failure == "timeout":
            raise subprocess.TimeoutExpired(argv, kwargs.get("timeout", 10))
        return subprocess.CompletedProcess(argv, 2, "", "probe error")

    monkeypatch.setattr(gitlock.subprocess, "run", run)


@pytest.mark.parametrize("failure", ["probe-raises", "missing-tool", "timeout", "error-exit"])
def test_a_failed_process_probe_removes_nothing(task_worktree, probe, monkeypatch, failure):
    wt = task_worktree
    if failure == "probe-raises":
        probe.answer = _raise_probe_error
    else:  # the strict probe itself runs, and its process listing fails
        probe.answer = probe.real
        _break_the_process_listing(monkeypatch, failure)
    lock = _plant_lock(wt.git_dir, _STALE_AGE)
    planted = _identity(lock)

    _assert_the_write_fails_as_before(wt, lock)

    assert gitlock.clear_stale_index_lock(wt.git_dir) is False
    assert _identity(lock) == planted
    assert probe.calls


def test_a_lock_replaced_before_removal_is_kept(task_worktree, probe):
    wt = task_worktree
    lock = _plant_lock(wt.git_dir, _STALE_AGE)
    swaps: list[tuple[tuple[int, int], tuple[int, int]]] = []

    def swap_the_lock_then_report_no_git() -> bool:
        # Between the first look and the removal another file takes the lock's place. It is
        # written while the old one still exists, so the inode really differs; the mtime does not.
        before = _identity(lock)
        fresh = lock.with_name("index.lock.fresh")
        fresh.write_bytes(b"")
        os.utime(fresh, ns=(before[1], before[1]))
        os.replace(fresh, lock)
        swaps.append((before, _identity(lock)))
        return False

    probe.answer = swap_the_lock_then_report_no_git

    _assert_the_write_fails_as_before(wt, lock)

    assert gitlock.clear_stale_index_lock(wt.git_dir) is False
    assert swaps, "the probe was never asked, so nothing was swapped"
    for before, after in swaps:
        assert after[0] != before[0] and after[1] == before[1]
    assert _identity(lock) == swaps[-1][1]


def test_only_the_task_worktrees_own_lock_is_touched(task_worktree):
    wt = task_worktree
    main_lock = _plant_lock(wt.main / ".git", _STALE_AGE)
    planted = _identity(main_lock)

    kb._git_mutation(wt.path, "apply", "--index", str(wt.patch))

    assert _git(wt.path, "diff", "--cached", "--name-only") == "tracked.txt"
    assert gitlock.clear_stale_index_lock(wt.git_dir) is False
    assert _identity(main_lock) == planted


def test_a_lock_that_is_a_link_is_kept(task_worktree, tmp_path):
    wt = task_worktree
    then = time.time() - _STALE_AGE
    target = tmp_path / "elsewhere"
    target.write_bytes(b"")
    os.utime(target, (then, then))
    lock = wt.git_dir / "index.lock"
    lock.symlink_to(target)
    os.utime(lock, (then, then), follow_symlinks=False)  # the link itself is old: only "not a plain file" keeps it

    _assert_the_write_fails_as_before(wt, lock)

    assert gitlock.clear_stale_index_lock(wt.git_dir) is False
    assert lock.is_symlink() and target.is_file()
