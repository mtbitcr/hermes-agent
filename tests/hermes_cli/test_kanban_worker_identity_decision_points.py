"""Every decision point that acts on a worker pid must read the IDENTITY.

Round one bound worker identity to the pair ``(worker_pid,
worker_start_time)`` and converted two decision points — the crashed-worker
sweep and the stale-claim reclaim. Five more kept branching on the pid
NUMBER alone, and a number is not an identity: the OS recycles pid numbers,
so an unrelated process wearing a recycled number reads as "the owner is
alive" to a bare liveness probe. Two of those five reach a signal path, so
the failure is not untidiness — it is SIGTERM, then SIGKILL, delivered to
an innocent process.

This suite covers one regression per decision point, plus the shared
termination helper they all reach:

* :func:`kanban_db.enforce_max_runtime`        — kills; must not kill a stranger
* :func:`kanban_db.detect_stale_running`       — kills; must not kill a stranger
* :func:`kanban_db.reconcile_orphaned_running` — must not defer to a stranger,
  and its audit event must carry the evidence, not a bare pid
* :func:`kanban_db.reclaim_task`               — operator reclaim, same rule
* :func:`kanban_db.verified_active_worker_rows` — must not badge a stranger
  as "Working now"
* :func:`kanban_db._terminate_reclaimed_worker` — refuses to signal a pid
  whose recorded identity does not match, WITHOUT changing its contract
  for any caller that does not pass the witness

Plus the two-module consistency check: ``kanban_db_dispatch`` ships its own
copies of these sweeps, and whatever ``kanban_db`` decides the dispatch copy
must decide identically — for the sweeps actually exercised here.  The crash
sweep (``detect_crashed_workers``) is explicitly EXCLUDED: the dispatch copy
cannot run it at all because ``_json_dict``, the fifth missing kernel helper,
is deliberately left unshimmed.  It does not fail cleanly either — it commits
the release and THEN raises while accounting the failure, so the task returns
to ``ready`` with its failure uncounted and the breaker never trips.  That
exclusion, and that consequence, are pinned by
``test_the_dispatch_crash_sweep_cannot_run_even_with_the_four_shims`` rather
than left implicit.

How the state is built, honestly (same as the round-one suites, and the
same shared helpers): a REAL child process is started and its REAL start
time observed while it lives; that child is killed and reaped. A SECOND
real child is started and its pid is bound to the claim by the REAL
``_set_worker_pid``. The recorded witness is then replaced with the one
observed from the first, now-dead process, via
``rebind_recorded_start_time``. Both numbers are genuine observations of
real processes; only their pairing is arranged, because which pid the OS
recycles and when is not ours to choose. Every decision taken on the
resulting row is the shipped code's.
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

import pytest

_WORKTREE = Path(__file__).resolve().parents[2]
if str(_WORKTREE) not in sys.path:
    sys.path.insert(0, str(_WORKTREE))

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_dispatch as kbd
from tests.hermes_cli._kanban_fence_support import create_fenced_board, ready_task
from tests.hermes_cli._kanban_worker_identity_support import (
    dead_child_identity,
    events_for,
    live_child,
    pid_is_alive,
    process_start_time,
    rebind_recorded_start_time,
    rewind_started_at,
    task_identity,
)


@pytest.fixture
def board(fence_home, monkeypatch):
    monkeypatch.setenv("HERMES_KANBAN_CRASH_GRACE_SECONDS", "0")
    create_fenced_board("decisions")
    return "decisions"


@pytest.fixture
def conn(board):
    connection = kb.connect(board=board)
    try:
        yield connection
    finally:
        connection.close()


def db_path(board: str) -> Path:
    return kb.kanban_db_path(board=board)


class Signals:
    """A ``signal_fn`` that RECORDS instead of signalling.

    Nothing here delivers a signal, so "the stranger is still alive" is
    proven twice over: by this recorder being empty and by reading the
    stranger's liveness without the module under test.
    """

    def __init__(self):
        self.sent: list[tuple[int, int]] = []

    def __call__(self, pid, sig):
        self.sent.append((int(pid), int(sig)))

    def to(self, pid: int) -> list:
        return [entry for entry in self.sent if entry[0] == int(pid)]


def _timed_out_task(conn, *, limit: int = 1) -> str:
    """A claimed task whose per-attempt runtime cap has a real deadline."""
    task_id = kb.create_task(
        conn, title="runtime", assignee="worker", max_runtime_seconds=limit,
    )
    with kb.write_txn(conn):
        conn.execute("UPDATE tasks SET status = 'ready' WHERE id = ?", (task_id,))
    assert kb.claim_task(conn, task_id) is not None
    return task_id


def _bind_recycled(conn, task_id: str, live_pid: int) -> int:
    """Bind a LIVE pid to the claim, then record SOMEONE ELSE's start time.

    Returns the recorded (stale) start time. The pid is placed by the real
    ``_set_worker_pid``; only the witness beside it is rebound, which is the
    one state the OS will not produce on demand.
    """
    kb._set_worker_pid(conn, task_id, live_pid)
    _gone_pid, gone_start = dead_child_identity(
        distinct_from=process_start_time(live_pid)
    )
    assert gone_start != process_start_time(live_pid)
    rebind_recorded_start_time(conn, task_id, gone_start)
    return gone_start


def _assert_identity_evidence(payload: dict, *, pid: int, recorded: int, observed):
    """The event carries the PROOF, not just the verdict."""
    assert payload["worker_pid"] == pid
    assert payload["worker_start_time"] == recorded
    assert payload["observed_worker_start_time"] == observed
    assert payload["worker_presence"] == kb.HolderPresence.PROVABLY_ABSENT.value
    assert "recycled" in payload["worker_identity_reason"]


# ---------------------------------------------------------------------------
# 1. enforce_max_runtime — this path KILLS
# ---------------------------------------------------------------------------

def test_enforce_max_runtime_never_signals_a_recycled_pid(board, conn):
    """The timeout still lands; the stranger wearing the number is spared."""
    with live_child() as stranger:
        task_id = _timed_out_task(conn)
        recorded = _bind_recycled(conn, task_id, stranger.pid)
        observed = process_start_time(stranger.pid)
        rewind_started_at(conn, task_id)

        signals = Signals()
        timed_out = kb.enforce_max_runtime(conn, signal_fn=signals)

        assert signals.to(stranger.pid) == [], (
            "a pid the OS recycled belongs to an unrelated process — "
            "SIGTERM/SIGKILL here kills an innocent bystander"
        )
        assert signals.sent == []
        assert pid_is_alive(stranger.pid)
        # The attempt really did exceed its limit, so the release still
        # happens through the ordinary timeout path.
        assert timed_out == [task_id]

    after = task_identity(db_path(board), task_id)
    assert after["status"] in ("ready", "todo", "blocked")
    assert after["worker_pid"] is None

    events = events_for(db_path(board), task_id, "timed_out")
    assert len(events) == 1
    payload = events[0][1]
    _assert_identity_evidence(
        payload, pid=stranger.pid, recorded=recorded, observed=observed,
    )
    assert payload["pid_reused"] is True
    assert payload["termination_attempted"] is False
    assert payload["sigkill"] is False


def test_enforce_max_runtime_still_signals_the_genuine_owner(board, conn):
    """The control: a matching identity is still terminated on timeout."""
    with live_child() as worker:
        task_id = _timed_out_task(conn)
        kb._set_worker_pid(conn, task_id, worker.pid)
        rewind_started_at(conn, task_id)

        signals = Signals()
        timed_out = kb.enforce_max_runtime(conn, signal_fn=signals)

        assert timed_out == [task_id]
        assert signals.to(worker.pid), (
            "a worker whose recorded identity matches must still be stopped "
            "when its runtime cap expires"
        )


# ---------------------------------------------------------------------------
# 2. detect_stale_running — this path KILLS
# ---------------------------------------------------------------------------

def test_detect_stale_running_never_signals_a_recycled_pid(board, conn):
    with live_child() as stranger:
        task_id = ready_task(conn)
        assert kb.claim_task(conn, task_id) is not None
        recorded = _bind_recycled(conn, task_id, stranger.pid)
        observed = process_start_time(stranger.pid)
        rewind_started_at(conn, task_id)

        signals = Signals()
        reclaimed = kb.detect_stale_running(
            conn, stale_timeout_seconds=1, signal_fn=signals,
        )

        assert signals.to(stranger.pid) == []
        assert signals.sent == []
        assert pid_is_alive(stranger.pid)
        # The heartbeat backstop is still the authority for RELEASING: the
        # card showed no progress and its recorded owner is gone.
        assert reclaimed == [task_id]

    events = events_for(db_path(board), task_id, "stale")
    assert len(events) == 1
    payload = events[0][1]
    _assert_identity_evidence(
        payload, pid=stranger.pid, recorded=recorded, observed=observed,
    )
    assert payload["pid_reused"] is True
    assert payload["termination_attempted"] is False
    # Still the ordinary stale release, with its ordinary keys.
    assert payload["prev_pid"] == stranger.pid
    assert payload["host_local"] is True


def test_detect_stale_running_leaves_a_live_matching_worker_alone(board, conn):
    """A genuinely LIVE owner with a fresh heartbeat is not stale at all."""
    with live_child() as worker:
        task_id = ready_task(conn)
        assert kb.claim_task(conn, task_id) is not None
        kb._set_worker_pid(conn, task_id, worker.pid)
        rewind_started_at(conn, task_id)
        assert kb.heartbeat_worker(conn, task_id) is True

        signals = Signals()
        assert kb.detect_stale_running(
            conn, stale_timeout_seconds=1, signal_fn=signals,
        ) == []
        assert signals.sent == []
        assert pid_is_alive(worker.pid)

    held = task_identity(db_path(board), task_id)
    assert held["status"] == "running"
    assert held["worker_pid"] == worker.pid


def test_detect_stale_running_treats_an_indeterminate_owner_like_a_live_one(
    board, conn,
):
    """No witness recorded => identity can be neither confirmed nor refuted.

    INDETERMINATE is not a shade of absent. It gets exactly the treatment a
    LIVE owner gets on this path: the worker is asked to stop, and because
    it does not die the claim is HELD (``reclaim_deferred``) rather than
    released beside a process that may still be doing real work. This is a
    CONTROL — it passes before and after the repair — and it is here so a
    later change cannot quietly downgrade "unknown" to "absent".
    """
    with live_child() as worker:
        task_id = ready_task(conn)
        assert kb.claim_task(conn, task_id) is not None
        kb._set_worker_pid(conn, task_id, worker.pid)
        # Pre-migration shape: a pid with no witness beside it.
        rebind_recorded_start_time(conn, task_id, None)
        rewind_started_at(conn, task_id)

        signals = Signals()
        assert kb.detect_stale_running(
            conn, stale_timeout_seconds=1, signal_fn=signals,
        ) == []
        assert pid_is_alive(worker.pid)

    held = task_identity(db_path(board), task_id)
    assert held["status"] == "running", (
        "an owner whose identity cannot be established must not be released "
        "as if it were provably gone"
    )
    assert [kind for kind, _ in events_for(db_path(board), task_id)].count(
        "reclaim_deferred"
    ) == 1


# ---------------------------------------------------------------------------
# 3. reconcile_orphaned_running — the audit event must carry the evidence
# ---------------------------------------------------------------------------

def _break_claim_bookkeeping(conn, task_id: str) -> None:
    """Leave the durable state a crash mid-claim leaves: no claim_expires.

    Arranged through the real write transaction, the same way the shared
    support module ages a TTL. This is the state
    ``reconcile_orphaned_running`` exists to recover and no production route
    produces it deliberately.
    """
    with kb.write_txn(conn):
        conn.execute(
            "UPDATE tasks SET claim_expires = NULL WHERE id = ?", (task_id,)
        )


def test_reconcile_orphaned_running_requeues_a_recycled_pid_with_the_evidence(
    board, conn,
):
    with live_child() as stranger:
        task_id = ready_task(conn)
        assert kb.claim_task(conn, task_id) is not None
        recorded = _bind_recycled(conn, task_id, stranger.pid)
        observed = process_start_time(stranger.pid)
        _break_claim_bookkeeping(conn, task_id)

        assert kb.reconcile_orphaned_running(conn) == [task_id], (
            "deferring to a recycled pid holds the card in 'running' forever"
        )
        assert pid_is_alive(stranger.pid)

    events = events_for(db_path(board), task_id, "reconciled")
    assert len(events) == 1
    payload = events[0][1]
    _assert_identity_evidence(
        payload, pid=stranger.pid, recorded=recorded, observed=observed,
    )
    assert payload["reason"] == "orphaned_running"


def test_reconcile_orphaned_running_still_defers_to_a_live_matching_worker(
    board, conn,
):
    with live_child() as worker:
        task_id = ready_task(conn)
        assert kb.claim_task(conn, task_id) is not None
        kb._set_worker_pid(conn, task_id, worker.pid)
        _break_claim_bookkeeping(conn, task_id)

        assert kb.reconcile_orphaned_running(conn) == []

    held = task_identity(db_path(board), task_id)
    assert held["status"] == "running"


# ---------------------------------------------------------------------------
# 4. reclaim_task — the operator reclaim
# ---------------------------------------------------------------------------

def test_reclaim_task_never_signals_a_recycled_pid(board, conn):
    """The operator gets the release; the stranger does not get the signal."""
    with live_child() as stranger:
        task_id = ready_task(conn)
        assert kb.claim_task(conn, task_id) is not None
        recorded = _bind_recycled(conn, task_id, stranger.pid)
        observed = process_start_time(stranger.pid)

        signals = Signals()
        assert kb.reclaim_task(
            conn, task_id, reason="operator asked", signal_fn=signals,
        ) is True

        assert signals.to(stranger.pid) == []
        assert signals.sent == []
        assert pid_is_alive(stranger.pid)

    after = task_identity(db_path(board), task_id)
    assert after["status"] in ("ready", "todo", "blocked")
    assert after["claim_lock"] is None
    assert after["worker_pid"] is None

    events = events_for(db_path(board), task_id, "reclaimed")
    assert len(events) == 1
    payload = events[0][1]
    _assert_identity_evidence(
        payload, pid=stranger.pid, recorded=recorded, observed=observed,
    )
    assert payload["manual"] is True
    assert payload["pid_reused"] is True
    assert payload["termination_attempted"] is False
    assert payload["prev_pid"] == stranger.pid


def test_reclaim_task_still_signals_the_genuine_owner(board, conn):
    with live_child() as worker:
        task_id = ready_task(conn)
        assert kb.claim_task(conn, task_id) is not None
        kb._set_worker_pid(conn, task_id, worker.pid)

        signals = Signals()
        kb.reclaim_task(conn, task_id, reason="operator", signal_fn=signals)
        assert signals.to(worker.pid), (
            "a matching owner must still be terminated by an operator reclaim"
        )


# ---------------------------------------------------------------------------
# 5. verified_active_worker_rows — "Working now" must mean this worker
# ---------------------------------------------------------------------------

def test_a_recycled_pid_is_not_a_verified_active_worker(board, conn):
    with live_child() as stranger:
        task_id = ready_task(conn)
        assert kb.claim_task(conn, task_id) is not None
        _bind_recycled(conn, task_id, stranger.pid)

        rows = kb.verified_active_worker_rows(conn)
        assert [r["worker_pid"] for r in rows] == [], (
            "the pid is alive but the process wearing it is not this worker: "
            "reporting it as active badges a stranger as 'Working now'"
        )
        assert pid_is_alive(stranger.pid)


def test_a_matching_owner_is_still_a_verified_active_worker(board, conn):
    with live_child() as worker:
        task_id = ready_task(conn)
        assert kb.claim_task(conn, task_id) is not None
        kb._set_worker_pid(conn, task_id, worker.pid)

        rows = kb.verified_active_worker_rows(conn)
        assert [r["worker_pid"] for r in rows] == [worker.pid]
        assert rows[0]["worker_start_time"] == process_start_time(worker.pid)


# ---------------------------------------------------------------------------
# 6. _terminate_reclaimed_worker — the shared signal path itself
# ---------------------------------------------------------------------------

def test_the_termination_helper_refuses_to_signal_a_mismatched_identity(board):
    with live_child() as stranger:
        _gone_pid, gone_start = dead_child_identity(
            distinct_from=process_start_time(stranger.pid)
        )
        signals = Signals()
        info = kb._terminate_reclaimed_worker(
            stranger.pid, kb._claimer_id(), signal_fn=signals,
            recorded_start_time=gone_start,
        )

        assert signals.sent == []
        assert pid_is_alive(stranger.pid)

    # The contract every caller and ``_worker_survived_termination`` read.
    assert info["prev_pid"] == stranger.pid
    assert info["host_local"] is True
    assert info["termination_attempted"] is False
    assert info["terminated"] is True      # the recorded OWNER really is gone
    assert info["sigkill"] is False
    assert info["pid_reused"] is True
    assert kb._worker_survived_termination(info) is False, (
        "refusing to signal a stranger must not look like a worker that "
        "survived termination, or the reclaim would defer forever"
    )


def test_the_termination_helper_is_unchanged_without_the_witness(board):
    """The optional parameter defaults to None: same behaviour as before."""
    with live_child() as worker:
        signals = Signals()
        info = kb._terminate_reclaimed_worker(
            worker.pid, kb._claimer_id(), signal_fn=signals,
        )
        assert info["termination_attempted"] is True
        assert signals.to(worker.pid)
        assert info["prev_pid"] == worker.pid
        assert info["host_local"] is True
        assert "pid_reused" not in info


def test_the_termination_helper_signals_a_matching_identity(board):
    with live_child() as worker:
        signals = Signals()
        info = kb._terminate_reclaimed_worker(
            worker.pid, kb._claimer_id(), signal_fn=signals,
            recorded_start_time=process_start_time(worker.pid),
        )
        assert info["termination_attempted"] is True
        assert signals.to(worker.pid)


# ---------------------------------------------------------------------------
# The two modules must decide identically
# ---------------------------------------------------------------------------

@pytest.fixture
def dispatch_shims(monkeypatch):
    """Supply four of the FIVE kernel helpers ``kanban_db_dispatch`` calls that DO NOT EXIST.

    ``kanban_db_dispatch`` reaches ``_kb._host_prefix``, ``_kb._row_get``,
    ``_kb._opt_int``, ``_kb._insert_comment`` and ``_kb._json_dict``, and
    ``hermes_cli.kanban_db`` defines none of them — in this tree OR in the
    round-one tree these tests are diffed against.  This fixture shims only
    the first four.  The fifth, ``_kb._json_dict``, is reached from
    ``_protocol_violation_streak`` via ``_account_crashes`` from
    ``detect_crashed_workers`` — the CRASH SWEEP — and is deliberately left
    unshimmed so the gap is executable rather than papered over.
    ``test_the_dispatch_crash_sweep_cannot_run_even_with_the_four_shims``
    pins that gap.

    Consequently, the dispatch-consistency tests below establish agreement
    ONLY for the sweeps they actually exercise: the runtime-cap sweep, the
    stale-claim sweep, the orphan-reconcile sweep and the termination
    helper.  The crash sweep is NOT covered — the dispatch copy raises
    ``AttributeError`` on DEFAULT settings (no optional staleness sweep
    needed), not only when an optional feature is enabled, and it raises
    AFTER its reclaim txn has committed, so the failure is never counted.

    That is a PRE-EXISTING defect, reported as an open finding rather than
    repaired here (repairing it would turn a dead module live, which is far
    outside a bounded identity fix).

    The shims are the obvious one-line bodies, orthogonal to worker identity,
    and exist only so the dispatch copies can RUN — so "both modules reach
    the same verdict" is something these tests execute rather than assert
    from a reading of the source.
    """
    monkeypatch.setattr(
        kb, "_host_prefix",
        lambda: f"{kb._claimer_id().split(':', 1)[0]}:",
        raising=False,
    )
    monkeypatch.setattr(
        kb, "_row_get",
        lambda row, key, default=None: (
            row[key] if key in row.keys() else default
        ),
        raising=False,
    )
    monkeypatch.setattr(
        kb, "_opt_int",
        lambda value: None if value is None else int(value),
        raising=False,
    )
    monkeypatch.setattr(
        kb, "_insert_comment",
        lambda conn, task_id, author, body, created_at: conn.execute(
            "INSERT INTO task_comments (task_id, author, body, created_at) "
            "VALUES (?, ?, ?, ?)",
            (task_id, author, body, created_at),
        ),
        raising=False,
    )


def test_dispatch_enforce_max_runtime_agrees_with_the_origin(
    board, conn, dispatch_shims,
):
    with live_child() as stranger:
        task_id = _timed_out_task(conn)
        recorded = _bind_recycled(conn, task_id, stranger.pid)
        observed = process_start_time(stranger.pid)
        rewind_started_at(conn, task_id)

        signals = Signals()
        assert kbd.enforce_max_runtime(conn, signal_fn=signals) == [task_id]
        assert signals.sent == []
        assert pid_is_alive(stranger.pid)

    payload = events_for(db_path(board), task_id, "timed_out")[0][1]
    _assert_identity_evidence(
        payload, pid=stranger.pid, recorded=recorded, observed=observed,
    )
    assert payload["pid_reused"] is True


def test_dispatch_detect_stale_running_agrees_with_the_origin(
    board, conn, dispatch_shims,
):
    with live_child() as stranger:
        task_id = ready_task(conn)
        assert kb.claim_task(conn, task_id) is not None
        recorded = _bind_recycled(conn, task_id, stranger.pid)
        observed = process_start_time(stranger.pid)
        rewind_started_at(conn, task_id)

        signals = Signals()
        assert kbd.detect_stale_running(
            conn, stale_timeout_seconds=1, signal_fn=signals,
        ) == [task_id]
        assert signals.sent == []
        assert pid_is_alive(stranger.pid)

    payload = events_for(db_path(board), task_id, "stale")[0][1]
    _assert_identity_evidence(
        payload, pid=stranger.pid, recorded=recorded, observed=observed,
    )
    assert payload["pid_reused"] is True


def test_dispatch_reconcile_orphaned_running_agrees_with_the_origin(
    board, conn, dispatch_shims,
):
    with live_child() as stranger:
        task_id = ready_task(conn)
        assert kb.claim_task(conn, task_id) is not None
        recorded = _bind_recycled(conn, task_id, stranger.pid)
        observed = process_start_time(stranger.pid)
        _break_claim_bookkeeping(conn, task_id)

        assert kbd.reconcile_orphaned_running(conn) == [task_id]
        assert pid_is_alive(stranger.pid)

    payload = events_for(db_path(board), task_id, "reconciled")[0][1]
    _assert_identity_evidence(
        payload, pid=stranger.pid, recorded=recorded, observed=observed,
    )


def test_the_dispatch_termination_helper_agrees_with_the_origin(
    board, dispatch_shims,
):
    with live_child() as stranger:
        _gone_pid, gone_start = dead_child_identity(
            distinct_from=process_start_time(stranger.pid)
        )
        signals = Signals()
        info = kbd._terminate_reclaimed_worker(
            stranger.pid, kb._claimer_id(), signal_fn=signals,
            recorded_start_time=gone_start,
        )
        assert signals.sent == []
        assert pid_is_alive(stranger.pid)

    assert info["prev_pid"] == stranger.pid
    assert info["host_local"] is True
    assert info["termination_attempted"] is False
    assert info["terminated"] is True
    assert info["sigkill"] is False
    assert info["pid_reused"] is True
    assert kbd._worker_survived_termination(info) is False


def test_the_dispatch_crash_sweep_cannot_run_even_with_the_four_shims(
    board, conn, dispatch_shims,
):
    """Pin a KNOWN DEFECT: the dispatch crash sweep cannot run.

    The dispatch copy of ``detect_crashed_workers`` reaches
    ``_protocol_violation_streak`` which calls ``_kb._json_dict``, the fifth
    missing helper that ``dispatch_shims`` deliberately does not supply.
    This test builds the real protocol-violation shape — a worker that exits
    cleanly (rc=0) while its task is still ``running`` — and proves:

    * the KERNEL copy classifies the case, returns the task id and COUNTS
      the failure;
    * the DISPATCH copy raises ``AttributeError`` naming ``_json_dict`` —
      but only AFTER committing the release, so the task is handed back to
      ``ready`` with ``consecutive_failures`` still 0.  Repeated forever,
      that is an unbounded respawn loop: the breaker that is supposed to
      park a task after a bounded number of clean-exit protocol violations
      can never trip through this module.

    This test passes BECAUSE the dispatch copy is broken.  It is EXPECTED TO
    FAIL, loudly and deliberately, when the separate dead-module card
    repairs ``kanban_db_dispatch``.  At that point this test should be
    converted into a real agreement assertion (like the three consistency
    tests above it).
    """

    def _violation_task():
        """Create a task whose worker exits cleanly while it is still running."""
        task_id = ready_task(conn)
        assert kb.claim_task(conn, task_id) is not None
        child = subprocess.Popen([sys.executable, "-c", "pass"])
        pid = child.pid
        child.returncode = 0
        kb._set_worker_pid(conn, task_id, pid)
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            kbd.reap_worker_zombies()
            classification, _rc = kbd._classify_worker_exit(pid)
            if classification != "unknown":
                break
            time.sleep(0.05)
        assert kbd._classify_worker_exit(pid) == ("clean_exit", 0)
        return task_id

    # First task — consumed by the kernel copy.
    task1 = _violation_task()
    reclaimed = kb.detect_crashed_workers(conn)
    assert task1 in reclaimed

    # Second task — the dispatch copy never reaches the decision.
    task2 = _violation_task()
    with pytest.raises(AttributeError, match="_json_dict"):
        kbd.detect_crashed_workers(conn)

    # It does NOT fail cleanly. The reclaim txn has already COMMITTED by the
    # time the accounting raises, so the task is released back to ``ready``
    # while its failure is never counted: ``consecutive_failures`` stays 0.
    # The kernel, on the same input, counts the failure.
    released = conn.execute(
        "SELECT status, consecutive_failures FROM tasks WHERE id = ?", (task2,)
    ).fetchone()
    assert released["status"] == "ready"
    assert released["consecutive_failures"] == 0
    assert conn.execute(
        "SELECT consecutive_failures FROM tasks WHERE id = ?", (task1,)
    ).fetchone()["consecutive_failures"] == 1
