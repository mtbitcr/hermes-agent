"""The claim → spawn → worker-pid binding, and what each gap in it proves.

Between taking a claim and recording the child's pid there is a window in
which the row carries a claim and no pid at all, and after it the row used
to carry a bare pid number. A number is not an identity — the OS recycles
them — so the kernel now records the process's start time beside the pid,
written by the same writer inside the same transaction, and classifies the
pair into the three verdicts ``HolderPresence`` already has.

What this suite holds:

* a claim whose worker NEVER spawned records no identity at all and is
  recoverable — its expiry is what recovers it, because there is no owner
  to protect and nothing to duplicate;
* a spawn that died before its pid was bound is recoverable the same way,
  and the recovered task can be claimed and bound again;
* a LIVE owner is never treated as absent: its claim is extended, not
  reclaimed, and the extension records the identity it was extended on;
* an UNKNOWN owner (a row bound before the witness existed) is never
  treated as absent either — it is HELD, not released, and not signalled;
* the removal fence reads the same witness, so "alive" and "we cannot tell"
  stop being the same answer.

Every claim is granted by the real ``claim_task``, every pid is bound by
the real ``_set_worker_pid`` the dispatcher uses, every worker is a real
child process, and every assertion about durable state reads the row with
a plain read-only connection.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_WORKTREE = Path(__file__).resolve().parents[2]
if str(_WORKTREE) not in sys.path:
    sys.path.insert(0, str(_WORKTREE))

from hermes_cli import kanban_db as kb
from tests.hermes_cli._kanban_fence_support import (
    create_fenced_board,
    ready_task,
    start_removal,
)
from tests.hermes_cli._kanban_worker_identity_support import (
    dead_pid,
    events_for,
    expire_claim,
    live_child,
    pid_is_alive,
    process_start_time,
    rebind_recorded_start_time,
    rewind_started_at,
    runs_for,
    task_identity,
)


@pytest.fixture
def board(fence_home, monkeypatch):
    """A real board, created through the real recorded creation path."""
    monkeypatch.setenv("HERMES_KANBAN_CRASH_GRACE_SECONDS", "0")
    create_fenced_board("identity")
    return "identity"


@pytest.fixture
def conn(board):
    connection = kb.connect(board=board)
    try:
        yield connection
    finally:
        connection.close()


def db_path(board: str) -> Path:
    return kb.kanban_db_path(board=board)


def _claimed(conn) -> str:
    task_id = ready_task(conn)
    assert kb.claim_task(conn, task_id) is not None
    return task_id


# ---------------------------------------------------------------------------
# A claim with no pid: unknown owner, but recoverable
# ---------------------------------------------------------------------------

def test_a_claim_whose_worker_never_spawned_is_recoverable(board, conn):
    """No pid was ever bound, so no identity was recorded — and the claim's
    own expiry is what brings the task back, with the audit event saying
    exactly what was (not) known about the owner."""
    task_id = _claimed(conn)

    before = task_identity(db_path(board), task_id)
    assert before["status"] == "running"
    assert before["worker_pid"] is None
    assert before["worker_start_time"] is None, (
        "a claim with no spawn must record no identity witness either"
    )

    # Nothing for the crash sweep to act on: it only considers rows that
    # name a pid, and this one never did.
    assert kb.detect_crashed_workers(conn) == []

    expire_claim(conn, task_id)
    assert kb.release_stale_claims(conn) == 1

    after = task_identity(db_path(board), task_id)
    assert after["status"] == "ready"
    assert after["claim_lock"] is None
    assert after["worker_pid"] is None

    reclaimed = events_for(db_path(board), task_id, "reclaimed")
    assert len(reclaimed) == 1
    payload = reclaimed[0][1]
    assert payload["automatic"] is True
    assert payload["worker_pid"] is None
    assert payload["worker_presence"] == kb.HolderPresence.INDETERMINATE.value
    assert "no worker pid" in payload["worker_identity_reason"]

    runs = runs_for(db_path(board), task_id)
    assert [r["outcome"] for r in runs] == ["reclaimed"]


def test_a_spawn_that_died_before_binding_recovers_and_rebinds(board, conn):
    """A worker that really started and really died before ``_set_worker_pid``
    ran leaves the same no-identity row — and once recovered, the next spawn
    binds a real identity onto it."""
    task_id = _claimed(conn)
    orphan = dead_pid()
    assert not pid_is_alive(orphan)

    stranded = task_identity(db_path(board), task_id)
    assert stranded["worker_pid"] is None
    assert stranded["worker_start_time"] is None

    expire_claim(conn, task_id)
    assert kb.release_stale_claims(conn) == 1
    assert task_identity(db_path(board), task_id)["status"] == "ready"

    # Recovery is complete only if the task can be worked again.
    assert kb.claim_task(conn, task_id) is not None
    with live_child() as child:
        kb._set_worker_pid(conn, task_id, child.pid)
        rebound = task_identity(db_path(board), task_id)
        assert rebound["worker_pid"] == child.pid
        assert rebound["worker_start_time"] == process_start_time(child.pid)
        assert kb._worker_identity_presence(
            rebound["worker_pid"], rebound["worker_start_time"]
        ) is kb.HolderPresence.LIVE


# ---------------------------------------------------------------------------
# A live owner is never absent
# ---------------------------------------------------------------------------

def test_a_live_owner_is_extended_not_reclaimed_and_the_witness_is_recorded(
    board, conn,
):
    task_id = _claimed(conn)
    with live_child() as child:
        kb._set_worker_pid(conn, task_id, child.pid)
        bound = task_identity(db_path(board), task_id)
        assert bound["worker_pid"] == child.pid
        assert bound["worker_start_time"] == process_start_time(child.pid)
        assert [r["worker_start_time"] for r in runs_for(db_path(board), task_id)] == [
            process_start_time(child.pid)
        ], "the run carries the same identity pair as the task row"

        expire_claim(conn, task_id)
        assert kb.release_stale_claims(conn) == 0, (
            "a live owner's claim is extended, never reclaimed"
        )

        extended = events_for(db_path(board), task_id, "claim_extended")
        assert len(extended) == 1
        payload = extended[0][1]
        assert payload["worker_pid"] == child.pid
        assert payload["worker_presence"] == kb.HolderPresence.LIVE.value
        assert payload["worker_start_time"] == process_start_time(child.pid)
        assert payload["observed_worker_start_time"] == payload["worker_start_time"]

        rewind_started_at(conn, task_id)
        assert kb.detect_crashed_workers(conn) == []
        assert task_identity(db_path(board), task_id)["status"] == "running"
        assert pid_is_alive(child.pid), "the live worker must not be disturbed"


# ---------------------------------------------------------------------------
# An unknown owner is never absent either
# ---------------------------------------------------------------------------

def test_an_unbound_witness_leaves_the_owner_unknown_and_the_claim_held(
    board, conn,
):
    """The pre-migration row shape — a live pid with no recorded start time.

    Its identity cannot be proven, so it is neither extended as live nor
    released as absent: it is HELD, and the hold says why.
    """
    task_id = _claimed(conn)
    with live_child() as child:
        kb._set_worker_pid(conn, task_id, child.pid)
        rebind_recorded_start_time(conn, task_id, None)
        assert task_identity(db_path(board), task_id)["worker_start_time"] is None

        expire_claim(conn, task_id)
        assert kb.release_stale_claims(conn) == 0, (
            "an owner that cannot be shown absent must not be reclaimed"
        )

        held = task_identity(db_path(board), task_id)
        assert held["status"] == "running"
        assert held["claim_lock"] is not None
        assert held["worker_pid"] == child.pid

        deferred = events_for(db_path(board), task_id, "reclaim_deferred")
        assert len(deferred) == 1
        payload = deferred[0][1]
        assert payload["worker_presence"] == kb.HolderPresence.INDETERMINATE.value
        assert payload["termination_attempted"] is False, (
            "an owner we cannot identify must not be signalled"
        )

        rewind_started_at(conn, task_id)
        assert kb.detect_crashed_workers(conn) == []
        assert pid_is_alive(child.pid)


# ---------------------------------------------------------------------------
# The removal fence reads the same witness
# ---------------------------------------------------------------------------

def _fenced_with_claim(slug: str, *, pid=None, witness_override=..., claimer=None):
    """A fenced board carrying one reservation held across the fence."""
    create_fenced_board(slug)
    conn = kb.connect(board=slug)
    try:
        task_id = ready_task(conn)
        assert kb.claim_task(conn, task_id, claimer=claimer) is not None
        if pid is not None:
            kb._set_worker_pid(conn, task_id, int(pid))
        if witness_override is not ...:
            rebind_recorded_start_time(conn, task_id, witness_override)
    finally:
        conn.close()
    intent = start_removal(slug, mode="reversible")
    assert intent.success, intent.message
    fenced = kb.advance_removal_to_fenced(slug, removal_id=intent.removal_id)
    assert fenced.success, fenced.message
    return intent.removal_id, task_id


def test_the_removal_fence_separates_a_proven_owner_from_an_unknown_one(
    fence_home, monkeypatch,
):
    """``LIVE`` now means "this really is the recorded process". A pid with
    no witness is not live and not absent: it is indeterminate, and both
    verdicts keep the reservation held."""
    monkeypatch.setenv("HERMES_KANBAN_CRASH_GRACE_SECONDS", "0")
    with live_child() as child:
        proven_id, proven_task = _fenced_with_claim("fence-proven", pid=child.pid)
        unknown_id, unknown_task = _fenced_with_claim(
            "fence-unknown", pid=child.pid, witness_override=None,
        )

        proven = kb.resolve_absent_holder_reservations(
            "fence-proven", removal_id=proven_id
        )
        unknown = kb.resolve_absent_holder_reservations(
            "fence-unknown", removal_id=unknown_id
        )

    assert proven.ended == ()
    assert [p.presence for p in proven.probes] == [kb.HolderPresence.LIVE]
    assert "start time matches" in proven.probes[0].reason

    assert unknown.ended == ()
    assert [p.presence for p in unknown.probes] == [kb.HolderPresence.INDETERMINATE]
    assert "no start time was recorded" in unknown.probes[0].reason

    for slug, task_id in (("fence-proven", proven_task), ("fence-unknown", unknown_task)):
        assert task_identity(db_path(slug), task_id)["claim_lock"] is not None
