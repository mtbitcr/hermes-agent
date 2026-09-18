"""Shared helpers for the claim → spawn → worker-PID identity regressions.

Same rule as :mod:`tests.hermes_cli._kanban_fence_support`, which this
module builds on rather than duplicates: everything here either drives a
REAL production entry point or reads persisted state with a plain
read-only SQLite connection. Nothing here arms a board, binds a pid, or
ends a claim by a route production does not take — a helper that could do
that would let a suite go green while every shipped path stayed dormant.

What IS here that the fence support module does not have:

* real child processes whose pid and start time are genuine observations
  of a real process (:func:`live_child`, :func:`dead_child_identity`), and
* a reader for the persisted identity pair and for task events, so an
  assertion about what the kernel decided reads the row the kernel wrote
  rather than asking the kernel to describe itself.

The ONE state arrangement that is not a production route is
:func:`rebind_recorded_start_time`, and it is named for what it is: the
PID-reuse state cannot be produced on demand (the OS decides when it
recycles a number), so the test records a REAL start time from a REAL
process and writes it, through the ordinary write transaction, beside a
REAL live pid. Both numbers are genuine observations; only their
pairing is arranged. Everything downstream of that pairing — the
classification, the release, the audit event — is the shipped code.
"""

from __future__ import annotations

import contextlib
import json
import os
import signal
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

_WORKTREE = Path(__file__).resolve().parents[2]
if str(_WORKTREE) not in sys.path:
    sys.path.insert(0, str(_WORKTREE))

from hermes_cli import kanban_db as kb  # noqa: E402
from tests.hermes_cli._kanban_fence_support import read_only  # noqa: E402


# A child that does nothing but stay alive until the test kills it. Started
# with a fresh interpreter so it is a genuinely separate OS process.
_SLEEPER = "import time; time.sleep(600)"


def process_start_time(pid: int):
    """The host's start-time fingerprint for *pid*, via the shipped probe."""
    from gateway.status import get_process_start_time

    return get_process_start_time(int(pid))


@contextlib.contextmanager
def live_child():
    """A REAL child process that stays alive for the body of the block.

    Yields the ``subprocess.Popen``. Killed and reaped on exit, always —
    a test that asserts "the live worker was not disturbed" has to leave
    no worker behind either.
    """
    proc = subprocess.Popen([sys.executable, "-c", _SLEEPER])
    # Give the kernel a moment to publish /proc/<pid>/stat.
    for _ in range(50):
        if process_start_time(proc.pid) is not None:
            break
        time.sleep(0.02)
    try:
        yield proc
    finally:
        with contextlib.suppress(ProcessLookupError, OSError):
            proc.kill()
        with contextlib.suppress(Exception):
            proc.wait(timeout=30)


def dead_child_identity(*, distinct_from=None):
    """``(pid, start_time)`` of a REAL process observed while it was alive.

    The process is then ended and reaped, so the pair describes a process
    that certainly no longer exists — the honest source of a start time
    that cannot belong to anything running now.

    ``distinct_from`` retries until the observed start time really differs
    from the one given. The fingerprint has clock-tick resolution (100 Hz
    on Linux), so two processes started inside the same tick legitimately
    share it; waiting for the next tick is how the test gets two genuinely
    different real observations rather than asserting a coincidence.
    """
    for _ in range(200):
        proc = subprocess.Popen([sys.executable, "-c", _SLEEPER])
        start = None
        for _ in range(50):
            start = process_start_time(proc.pid)
            if start is not None:
                break
            time.sleep(0.02)
        proc.kill()
        proc.wait(timeout=30)
        assert start is not None, "could not observe the child's start time"
        if distinct_from is None or int(start) != int(distinct_from):
            return proc.pid, int(start)
        time.sleep(0.02)
    raise AssertionError("could not observe two distinct real start times")


def dead_pid() -> int:
    """A pid that certainly is not running: a child, started and reaped."""
    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    proc.wait(timeout=30)
    return proc.pid


def pid_is_alive(pid: int) -> bool:
    """Liveness read WITHOUT the module under test, for "we didn't kill it"."""
    try:
        os.kill(int(pid), 0)
    except (ProcessLookupError, OSError):
        return False
    try:
        state = Path(f"/proc/{int(pid)}/status").read_text(encoding="utf-8")
    except OSError:
        return True
    return "State:\tZ" not in state


# ---------------------------------------------------------------------------
# Persisted-state readers (never through kanban_db)
# ---------------------------------------------------------------------------

def task_identity(db_path: Path, task_id: str) -> dict:
    """The persisted ``(status, claim, pid, witness)`` of one task row."""
    with read_only(db_path) as conn:
        row = conn.execute(
            "SELECT status, claim_lock, claim_expires, worker_pid, "
            "       worker_start_time, current_run_id "
            "FROM tasks WHERE id = ?",
            (task_id,),
        ).fetchone()
    return None if row is None else dict(row)


def run_identity(db_path: Path, run_id: int) -> dict:
    with read_only(db_path) as conn:
        row = conn.execute(
            "SELECT id, task_id, status, outcome, worker_pid, worker_start_time, "
            "       ended_at "
            "FROM task_runs WHERE id = ?",
            (int(run_id),),
        ).fetchone()
    return None if row is None else dict(row)


def runs_for(db_path: Path, task_id: str) -> list:
    with read_only(db_path) as conn:
        return [
            dict(r)
            for r in conn.execute(
                "SELECT id, status, outcome, worker_pid, worker_start_time, "
                "       ended_at "
                "FROM task_runs WHERE task_id = ? ORDER BY id",
                (task_id,),
            )
        ]


def events_for(db_path: Path, task_id: str, kind: str = None) -> list:
    """``[(kind, payload_dict), ...]`` as persisted, oldest first."""
    with read_only(db_path) as conn:
        rows = conn.execute(
            "SELECT kind, payload FROM task_events WHERE task_id = ? ORDER BY id",
            (task_id,),
        ).fetchall()
    out = []
    for row in rows:
        if kind is not None and row["kind"] != kind:
            continue
        try:
            payload = json.loads(row["payload"]) if row["payload"] else {}
        except (TypeError, ValueError):
            payload = {}
        out.append((row["kind"], payload))
    return out


def event_rows(db_path: Path, task_id: str) -> list:
    """Every persisted event for a task, with its run binding and payload."""
    with read_only(db_path) as conn:
        rows = conn.execute(
            "SELECT id, kind, run_id, payload FROM task_events "
            "WHERE task_id = ? ORDER BY id",
            (task_id,),
        ).fetchall()
    out = []
    for row in rows:
        try:
            payload = json.loads(row["payload"]) if row["payload"] else {}
        except (TypeError, ValueError):
            payload = {}
        out.append({
            "id": row["id"], "kind": row["kind"],
            "run_id": row["run_id"], "payload": payload,
        })
    return out


def task_columns(db_path: Path) -> set:
    with read_only(db_path) as conn:
        return {r[1] for r in conn.execute("PRAGMA table_info(tasks)")}


# ---------------------------------------------------------------------------
# Durable-state arrangement (through the real write transaction)
# ---------------------------------------------------------------------------

def expire_claim(conn, task_id: str, *, seconds_ago: int = 3600) -> int:
    """Age this claim's TTL so the stale sweep considers it. Returns the value.

    Same shape as ``_kanban_fence_support.ready_task``'s status write:
    arranging the durable state a real passage of time would leave, through
    the real write transaction, never a substitute for the behaviour under
    test.
    """
    expires = int(time.time()) - int(seconds_ago)
    with kb.write_txn(conn):
        conn.execute(
            "UPDATE tasks SET claim_expires = ? WHERE id = ?", (expires, task_id)
        )
        conn.execute(
            "UPDATE task_runs SET claim_expires = ? "
            "WHERE task_id = ? AND ended_at IS NULL",
            (expires, task_id),
        )
    return expires


def rewind_started_at(conn, task_id: str, *, seconds: int = 9999) -> None:
    """Move this attempt's start back so the launch-window grace has passed."""
    with kb.write_txn(conn):
        conn.execute(
            "UPDATE tasks SET started_at = started_at - ? WHERE id = ?",
            (int(seconds), task_id),
        )
        conn.execute(
            "UPDATE task_runs SET started_at = started_at - ? WHERE task_id = ?",
            (int(seconds), task_id),
        )


def rebind_recorded_start_time(conn, task_id: str, start_time) -> None:
    """Pair the row's pid with a start time observed from ANOTHER process.

    This is the PID-reuse state, and it is the one thing in these suites
    that no production route can produce on demand: which number the OS
    recycles, and when, is not ours to choose. So the test supplies a start
    time it really did read from a real process that has really exited, and
    leaves the pid exactly where the real ``_set_worker_pid`` put it. The
    row that results is indistinguishable from the one a recycled pid
    produces, and every decision taken on it afterwards is the shipped
    code's.
    """
    with kb.write_txn(conn):
        conn.execute(
            "UPDATE tasks SET worker_start_time = ? WHERE id = ?",
            (None if start_time is None else int(start_time), task_id),
        )
        conn.execute(
            "UPDATE task_runs SET worker_start_time = ? "
            "WHERE task_id = ? AND ended_at IS NULL",
            (None if start_time is None else int(start_time), task_id),
        )


# ---------------------------------------------------------------------------
# The two-dispatcher public-entrypoint harness
# ---------------------------------------------------------------------------

_DISPATCH_CHILD = Path(__file__).with_name("_kanban_worker_identity_dispatch_child.py")


def run_dispatch_child(
    home: Path,
    slug: str,
    *,
    crash: str = "never",
    spawn: str = "sleeper",
    barrier: Path = None,
    timeout: int = 120,
) -> subprocess.CompletedProcess:
    """One dispatcher process, driven through the PUBLIC ``dispatch_once``.

    A separate OS process every time, so a SIGKILL at a named point really
    is a power loss to that dispatcher and the restart really is a fresh
    process reading only durable state.
    """
    argv = [
        sys.executable, str(_DISPATCH_CHILD),
        "--home", str(home), "--board", slug,
        "--crash", crash, "--spawn", spawn,
    ]
    if barrier is not None:
        argv += ["--barrier", str(barrier)]
    return subprocess.run(
        argv, capture_output=True, text=True, timeout=timeout,
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
    )


def two_dispatchers(home: Path, slug: str, *, barrier: Path, timeout: int = 120):
    """Two dispatcher PROCESSES racing one board through ``dispatch_once``.

    Both are started before either is released, and both drive the shipped
    public entry point — the board's own single-writer dispatch lock is
    what decides the race, not any arrangement made here. Returns both
    completed processes.
    """
    argv = [
        sys.executable, str(_DISPATCH_CHILD),
        "--home", str(home), "--board", slug,
        "--crash", "never", "--spawn", "sleeper",
        "--barrier", str(barrier),
    ]
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    barrier = Path(barrier)
    procs = [
        subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                         text=True, env=env)
        for _ in range(2)
    ]
    # Release only once BOTH are up and parked, so the tick really is
    # contended rather than sequential.
    deadline = time.time() + 60
    while time.time() < deadline:
        ready = list(barrier.parent.glob(f"{barrier.name}.ready.*"))
        if len(ready) >= 2:
            break
        time.sleep(0.05)
    barrier.write_text("go\n", encoding="utf-8")
    results = []
    for proc in procs:
        out, err = proc.communicate(timeout=timeout)
        results.append(subprocess.CompletedProcess(argv, proc.returncode, out, err))
    return results


def child_report(proc: subprocess.CompletedProcess) -> dict:
    """The ``REPORT {...}`` line a dispatcher child prints, as a dict."""
    for line in (proc.stdout or "").splitlines():
        if line.startswith("REPORT "):
            return json.loads(line[len("REPORT "):])
    return {}


def spawned_worker_pids(proc: subprocess.CompletedProcess) -> list:
    """Pids of the REAL worker children a dispatcher child started."""
    pids = []
    for line in (proc.stdout or "").splitlines():
        if line.startswith("SPAWNED "):
            pids.append(int(line.split()[1]))
    return pids


def kill_pids(pids) -> None:
    for pid in pids:
        with contextlib.suppress(ProcessLookupError, OSError):
            os.kill(int(pid), signal.SIGKILL)
