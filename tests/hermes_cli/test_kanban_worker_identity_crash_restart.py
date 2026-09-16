"""A dispatcher that dies around the pid binding, and the restart after it.

The binding is the seam: before ``_set_worker_pid`` commits, a real worker
is running and the row names no pid; after it commits, the row names a pid
AND the identity witness for it. A dispatcher can be killed on either side
of that commit, and what the NEXT dispatcher does has to follow from
durable state alone.

Both crashes here are real: a separate dispatcher PROCESS drives the
shipped public entry point (``kanban_db.dispatch_once`` — the same call the
gateway dispatcher and ``hermes kanban dispatch`` make), spawns a REAL
worker child, and SIGKILLs itself at the named point, so nothing unwinds
or flushes. The restart is another real process doing the same public
tick.

What this suite holds: after either crash, ``tasks``, ``task_runs`` and
the board register rows agree with each other; the half-bound row records
NEITHER a pid NOR a witness (never one without the other); a restart never
spawns a second worker beside a live one; and the orphan left by the crash
is recovered through the ordinary paths.

The two-dispatcher race is driven through the same public entry point, so
the board's own single-writer dispatch lock — not anything arranged here —
is what decides it.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

_WORKTREE = Path(__file__).resolve().parents[2]
if str(_WORKTREE) not in sys.path:
    sys.path.insert(0, str(_WORKTREE))

from hermes_cli import kanban_db as kb
from tests.hermes_cli._kanban_fence_support import (
    create_fenced_board,
    ready_task,
    register_row,
)
from tests.hermes_cli._kanban_worker_identity_support import (
    child_report,
    events_for,
    expire_claim,
    kill_pids,
    pid_is_alive,
    process_start_time,
    run_dispatch_child,
    runs_for,
    spawned_worker_pids,
    task_identity,
    two_dispatchers,
)


# Real signal delivery is the point here, not an accident: the workers these
# dispatchers spawn are orphaned by a deliberate SIGKILL of their parent, so
# by the time the test cleans them up they have been reparented to init and
# are no longer inside the test process's subtree. That is exactly the
# situation the established bypass marker exists for.
pytestmark = pytest.mark.live_system_guard_bypass


@pytest.fixture
def board(fence_home, monkeypatch):
    monkeypatch.setenv("HERMES_KANBAN_CRASH_GRACE_SECONDS", "0")
    create_fenced_board("restart")
    return "restart"


@pytest.fixture
def home(fence_home):
    return fence_home


def db_path(board: str) -> Path:
    return kb.kanban_db_path(board=board)


@pytest.fixture
def orphans():
    """Every real worker child any dispatcher started, killed at the end."""
    pids: list = []
    yield pids
    kill_pids(pids)


def _one_ready_task(board: str) -> str:
    conn = kb.connect(board=board)
    try:
        return ready_task(conn)
    finally:
        conn.close()


def _assert_register_intact(slug: str) -> None:
    entry = register_row(slug)
    assert entry is not None, "the crash must not have unmade the register row"
    assert entry["lifecycle"] == kb.BoardLifecycle.LIVE.value
    assert entry["epoch"] == 1
    live = kb.get_register_entry(slug)
    assert live is not None and live.epoch == entry["epoch"]


def _assert_pair_is_all_or_nothing(row: dict) -> None:
    """A pid without a witness (or the reverse) is the state under test."""
    if row["worker_pid"] is None:
        assert row["worker_start_time"] is None, (
            "a row with no pid must not carry a witness for one"
        )
    else:
        assert row["worker_start_time"] == process_start_time(row["worker_pid"]) or (
            row["worker_start_time"] is not None
        ), "a bound pid must carry the witness written with it"


# ---------------------------------------------------------------------------
# Crash BEFORE the pid is bound
# ---------------------------------------------------------------------------

def test_a_crash_before_binding_leaves_a_consistent_recoverable_board(
    home, board, orphans,
):
    task_id = _one_ready_task(board)

    crashed = run_dispatch_child(home, board, crash="before-bind")
    orphans.extend(spawned_worker_pids(crashed))
    assert crashed.returncode == -9, "the dispatcher must really have been killed"
    assert len(orphans) == 1, "a real worker was started before the crash"
    assert pid_is_alive(orphans[0]), "the worker outlived its dispatcher"

    row = task_identity(db_path(board), task_id)
    assert row["status"] == "running"
    assert row["claim_lock"] is not None
    assert row["worker_pid"] is None
    assert row["worker_start_time"] is None
    _assert_pair_is_all_or_nothing(row)

    runs = runs_for(db_path(board), task_id)
    assert len(runs) == 1
    assert runs[0]["status"] == "running" and runs[0]["ended_at"] is None
    assert runs[0]["worker_pid"] is None
    assert runs[0]["worker_start_time"] is None
    assert row["current_run_id"] == runs[0]["id"]
    _assert_register_intact(board)

    # A restart does not invent an owner for the unbound claim, and does not
    # reclaim it before its expiry either.
    restart = run_dispatch_child(home, board, crash="never")
    orphans.extend(spawned_worker_pids(restart))
    assert restart.returncode == 0
    assert child_report(restart)["spawned"] == [], (
        "the claim is still held: no second worker may be spawned beside it"
    )
    assert task_identity(db_path(board), task_id)["status"] == "running"

    # Its expiry is what recovers it — there is no owner to protect.
    conn = kb.connect(board=board)
    try:
        expire_claim(conn, task_id)
    finally:
        conn.close()

    recovered = run_dispatch_child(home, board, crash="never")
    orphans.extend(spawned_worker_pids(recovered))
    assert recovered.returncode == 0
    report = child_report(recovered)
    assert report["reclaimed"] == 1
    assert [item[0] for item in report["spawned"]] == [task_id]

    final = task_identity(db_path(board), task_id)
    assert final["status"] == "running"
    assert final["worker_pid"] == orphans[-1]
    assert final["worker_start_time"] == process_start_time(orphans[-1])

    runs = runs_for(db_path(board), task_id)
    assert [r["outcome"] for r in runs] == ["reclaimed", None]
    assert runs[-1]["worker_pid"] == final["worker_pid"]
    assert runs[-1]["worker_start_time"] == final["worker_start_time"]
    assert final["current_run_id"] == runs[-1]["id"]
    _assert_register_intact(board)


# ---------------------------------------------------------------------------
# Crash AFTER the pid is bound
# ---------------------------------------------------------------------------

def test_a_crash_after_binding_leaves_the_identity_durable_and_respected(
    home, board, orphans,
):
    task_id = _one_ready_task(board)

    crashed = run_dispatch_child(home, board, crash="after-bind")
    orphans.extend(spawned_worker_pids(crashed))
    assert crashed.returncode == -9
    worker = orphans[0]
    assert pid_is_alive(worker)

    row = task_identity(db_path(board), task_id)
    assert row["status"] == "running"
    assert row["worker_pid"] == worker
    assert row["worker_start_time"] == process_start_time(worker), (
        "the witness committed in the same transaction as the pid"
    )
    runs = runs_for(db_path(board), task_id)
    assert len(runs) == 1
    assert runs[0]["worker_pid"] == worker
    assert runs[0]["worker_start_time"] == row["worker_start_time"]
    assert row["current_run_id"] == runs[0]["id"]
    _assert_register_intact(board)

    # The restart sees a worker it can PROVE is the recorded one, so it
    # neither sweeps it nor spawns beside it.
    restart = run_dispatch_child(home, board, crash="never")
    orphans.extend(spawned_worker_pids(restart))
    assert restart.returncode == 0
    report = child_report(restart)
    assert report["spawned"] == []
    assert report["crashed"] == []
    assert pid_is_alive(worker)
    assert task_identity(db_path(board), task_id)["worker_pid"] == worker

    # Once the worker really dies, the same public tick sweeps it.
    kill_pids([worker])
    for _ in range(100):
        if not pid_is_alive(worker):
            break
        time.sleep(0.05)
    swept = run_dispatch_child(home, board, crash="never")
    orphans.extend(spawned_worker_pids(swept))
    assert swept.returncode == 0
    assert task_id in child_report(swept)["crashed"]

    crashed_events = events_for(db_path(board), task_id, "crashed")
    assert len(crashed_events) == 1
    assert crashed_events[0][1]["pid"] == worker
    assert crashed_events[0][1]["worker_presence"] == (
        kb.HolderPresence.PROVABLY_ABSENT.value
    )

    runs = runs_for(db_path(board), task_id)
    assert runs[0]["outcome"] == "crashed" and runs[0]["ended_at"] is not None
    _assert_register_intact(board)


# ---------------------------------------------------------------------------
# Two dispatchers, one board, one public entry point
# ---------------------------------------------------------------------------

def test_two_dispatchers_racing_the_same_board_bind_exactly_one_identity(
    home, board, orphans, tmp_path,
):
    """Contended ticks through the shipped entry point must leave exactly one
    owner, one run, and one identity pair — never two."""
    task_id = _one_ready_task(board)

    first, second = two_dispatchers(home, board, barrier=tmp_path / "race.barrier")
    for proc in (first, second):
        orphans.extend(spawned_worker_pids(proc))
        assert proc.returncode == 0, proc.stderr[-2000:]

    spawned = [
        item[0]
        for proc in (first, second)
        for item in child_report(proc)["spawned"]
    ]
    assert spawned == [task_id], "exactly one dispatcher may spawn this task"

    row = task_identity(db_path(board), task_id)
    assert row["status"] == "running"
    assert row["worker_pid"] is not None
    assert row["worker_start_time"] == process_start_time(row["worker_pid"])

    runs = runs_for(db_path(board), task_id)
    assert len(runs) == 1
    assert runs[0]["worker_pid"] == row["worker_pid"]
    assert runs[0]["worker_start_time"] == row["worker_start_time"]

    spawned_events = events_for(db_path(board), task_id, "spawned")
    assert len(spawned_events) == 1
    assert spawned_events[0][1]["pid"] == row["worker_pid"]
    _assert_register_intact(board)
