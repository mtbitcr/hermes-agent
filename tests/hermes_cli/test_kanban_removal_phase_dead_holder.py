"""QB-1c / QB-1d (design revision 5, §6.3): the removal is the single
resolver of every reservation on a board being removed.

Standing every claim-ending path down at the fence leaves nobody to end a
claim whose worker died after the close — which is why the removal itself
ends one, through a single-owner conditional transition, in both modes, at
any point during quiescence. "Provably absent" is a POSITIVE standard: the
query must have succeeded and be from a scope where a known-present holder
would have been found. Anything else is indeterminate and is treated as
live, because the direction that fails closed is the one that protects
work.

Every reservation here is granted by the real ``claim_task`` before the
fence closes, and every worker pid is recorded through the real
``_set_worker_pid`` the dispatcher uses.
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
from tests.hermes_cli._kanban_fence_support import (
    create_fenced_board,
    ready_task,
    start_removal,
    task_row,
)


def _dead_pid() -> int:
    """A pid that certainly is not running: a child, started and reaped."""
    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    proc.wait(timeout=30)
    return proc.pid


def _claimed_board(
    slug: str,
    *,
    mode: str = "reversible",
    claimer=None,
    worker_pid=None,
    claim_expires=None,
) -> "tuple[str, str]":
    """A fenced board carrying one reservation Held across the fence."""
    create_fenced_board(slug)
    conn = kb.connect(board=slug)
    task_id = ready_task(conn)
    assert kb.claim_task(conn, task_id, claimer=claimer) is not None
    if worker_pid is not None:
        kb._set_worker_pid(conn, task_id, int(worker_pid))
    if claim_expires is not None:
        with kb.write_txn(conn):
            conn.execute(
                "UPDATE tasks SET claim_expires = ? WHERE id = ?",
                (int(claim_expires), task_id),
            )
    conn.close()

    intent = start_removal(slug, mode=mode)
    assert intent.success, intent.message
    fenced = kb.advance_removal_to_fenced(slug, removal_id=intent.removal_id)
    assert fenced.success, fenced.message
    return intent.removal_id, task_id


# ---------------------------------------------------------------------------
# QB-1d: a provably absent holder is ended BY THE REMOVAL
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("mode", ["reversible", "permanent"])
def test_a_provably_absent_holder_is_ended_by_the_removal(fence_home, mode):
    slug = f"dead-holder-{mode}"
    removal_id, task_id = _claimed_board(slug, mode=mode, worker_pid=_dead_pid())

    result = kb.resolve_absent_holder_reservations(slug, removal_id=removal_id)

    assert result.success is True
    assert result.ended == (task_id,)
    row = task_row(kb.kanban_db_path(board=slug), task_id)
    assert row["claim_lock"] is None
    assert row["status"] != "running"


def test_quiescence_completes_over_a_holder_that_died_after_the_fence(fence_home):
    """X-Q5: a reversible removal COMPLETES over a worker that no longer
    exists, instead of running to its deadline and abandoning with a
    report that work is still live when it is not."""
    removal_id, _task_id = _claimed_board("dead-completes", worker_pid=_dead_pid())

    result = kb.advance_removal_to_quiesced("dead-completes", removal_id=removal_id)

    assert result.success is True
    assert result.held == 0
    assert result.resolution is not None
    assert len(result.resolution.ended) == 1
    assert kb.get_removal_phase_record(
        "dead-completes"
    ).phase == kb.RemovalPhase.QUIESCED


def test_the_resolver_is_single_owner_a_second_pass_ends_nothing_twice(fence_home):
    removal_id, task_id = _claimed_board("dead-once", worker_pid=_dead_pid())

    first = kb.resolve_absent_holder_reservations("dead-once", removal_id=removal_id)
    second = kb.resolve_absent_holder_reservations("dead-once", removal_id=removal_id)

    assert first.ended == (task_id,)
    assert second.ended == ()
    assert second.success is True


# ---------------------------------------------------------------------------
# Indeterminate is treated as LIVE, in every shape it comes in
# ---------------------------------------------------------------------------

def test_a_live_holder_is_never_ended(fence_home):
    import os

    removal_id, task_id = _claimed_board("live-holder", worker_pid=os.getpid())

    result = kb.resolve_absent_holder_reservations("live-holder", removal_id=removal_id)

    assert result.ended == ()
    assert [p.presence for p in result.probes] == [kb.HolderPresence.LIVE]
    assert task_row(kb.kanban_db_path(board="live-holder"), task_id)[
        "claim_lock"
    ] is not None


def test_a_claim_with_no_recorded_pid_is_indeterminate_and_stays_held(fence_home):
    removal_id, task_id = _claimed_board("no-pid")

    result = kb.resolve_absent_holder_reservations("no-pid", removal_id=removal_id)

    assert result.ended == ()
    assert result.probes[0].presence is kb.HolderPresence.INDETERMINATE
    assert "no worker pid" in result.probes[0].reason
    assert task_row(kb.kanban_db_path(board="no-pid"), task_id)["claim_lock"] is not None


def test_a_holder_recorded_on_another_host_is_indeterminate(fence_home):
    """A pid recorded on a host this system cannot interrogate proves
    nothing: its recorded expiry is what resolves it."""
    removal_id, task_id = _claimed_board(
        "foreign-host", claimer="some-other-host:4242", worker_pid=_dead_pid()
    )

    result = kb.resolve_absent_holder_reservations(
        "foreign-host", removal_id=removal_id
    )

    assert result.ended == ()
    assert result.probes[0].presence is kb.HolderPresence.INDETERMINATE
    assert "another host" in result.probes[0].reason
    assert task_row(kb.kanban_db_path(board="foreign-host"), task_id)[
        "claim_lock"
    ] is not None


def test_a_liveness_probe_that_fails_is_indeterminate_not_absent(fence_home):
    removal_id, task_id = _claimed_board("probe-fails", worker_pid=_dead_pid())

    def exploding_probe(pid):
        raise OSError("the probe could not be performed")

    result = kb.resolve_absent_holder_reservations(
        "probe-fails", removal_id=removal_id, pid_probe=exploding_probe
    )

    assert result.ended == ()
    assert result.probes[0].presence is kb.HolderPresence.INDETERMINATE
    assert task_row(kb.kanban_db_path(board="probe-fails"), task_id)[
        "claim_lock"
    ] is not None


# ---------------------------------------------------------------------------
# QB-1d-i: expiry alone ends nothing
# ---------------------------------------------------------------------------

def test_a_passed_expiry_with_a_live_holder_ends_nothing(fence_home):
    """A live worker reaches its recorded expiry as a matter of course
    during a removal, precisely because QB-1b stopped every path from
    extending it. The timestamp passing is not evidence about the worker."""
    import os

    removal_id, task_id = _claimed_board(
        "expiry-live",
        worker_pid=os.getpid(),
        claim_expires=int(time.time()) - 10_000,
    )

    resolution = kb.resolve_absent_holder_reservations(
        "expiry-live", removal_id=removal_id
    )
    assert resolution.ended == ()

    result = kb.advance_removal_to_quiesced("expiry-live", removal_id=removal_id)
    assert result.success is False
    assert result.held == 1
    assert result.action is kb.QuiescenceDeadlineAction.WAIT
    assert task_row(kb.kanban_db_path(board="expiry-live"), task_id)[
        "claim_lock"
    ] is not None


def test_a_passed_expiry_with_an_indeterminate_holder_ends_nothing(fence_home):
    removal_id, task_id = _claimed_board(
        "expiry-indeterminate", claim_expires=int(time.time()) - 10_000
    )

    result = kb.advance_removal_to_quiesced(
        "expiry-indeterminate", removal_id=removal_id
    )

    assert result.success is False
    assert result.held == 1
    assert task_row(kb.kanban_db_path(board="expiry-indeterminate"), task_id)[
        "claim_lock"
    ] is not None


# ---------------------------------------------------------------------------
# The resolver only acts once the removal really IS the single resolver
# ---------------------------------------------------------------------------

def test_the_resolver_refuses_while_the_gate_is_still_open(fence_home):
    """With the gate open the ordinary claim-ending paths are still in
    charge; the removal is not yet the single resolver (QB-1c)."""
    create_fenced_board("gate-open")
    conn = kb.connect(board="gate-open")
    task_id = ready_task(conn)
    assert kb.claim_task(conn, task_id) is not None
    kb._set_worker_pid(conn, task_id, _dead_pid())
    conn.close()
    intent = start_removal("gate-open", mode="reversible")

    result = kb.resolve_absent_holder_reservations(
        "gate-open", removal_id=intent.removal_id
    )

    assert result.success is False
    assert "still open" in result.message
    assert task_row(kb.kanban_db_path(board="gate-open"), task_id)[
        "claim_lock"
    ] is not None


def test_the_resolver_refuses_an_unreadable_store(fence_home):
    removal_id, _task_id = _claimed_board("resolver-unreadable", worker_pid=_dead_pid())
    kb.kanban_db_path(board="resolver-unreadable").write_bytes(b"not a database")

    result = kb.resolve_absent_holder_reservations(
        "resolver-unreadable", removal_id=removal_id
    )

    assert result.success is False


def test_the_resolver_refuses_a_mismatched_removal_id(fence_home):
    removal_id, task_id = _claimed_board("resolver-mismatch", worker_pid=_dead_pid())

    result = kb.resolve_absent_holder_reservations(
        "resolver-mismatch", removal_id="not-this-run"
    )

    assert result.success is False
    assert task_row(kb.kanban_db_path(board="resolver-mismatch"), task_id)[
        "claim_lock"
    ] is not None
