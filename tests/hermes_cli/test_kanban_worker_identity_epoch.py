"""A restored board publishes a new epoch, and the work from before it stops.

A board that has been restored from a copy is not the board the old
handles were talking to, even though the file is byte-identical: the
register publishes a NEW epoch for the restored name, and until the
in-board gate (Gate B) mirrors it, the (register epoch, mirror epoch) pair
is indeterminate under EM-4c. Every mutation — a worker's heartbeat, its
completion, a fresh claim, and the dispatcher's own reclaim sweep — is
refused while that is true. That is the existing epoch machinery
(``validate_epoch_pair``, ``accepted_epoch_mirrors``, ``commit_gate_state``,
``board_register_lock``), and this suite asserts through it rather than
around it.

The identity half is what the restore makes visible: the pid and the start
time recorded beside it are durable columns, so they cross the restore
intact. The worker that was running before the outage is NOT running after
it, and once Gate B has caught up the claim is released through the
ordinary reclaim path on that evidence — never held open because a number
in a restored row happens to name some process again.

Everything durable is read with a plain read-only connection; the restore
is a real file copy taken while work was in flight and copied back.
"""

from __future__ import annotations

import shutil
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
    gate_row,
    ready_task,
    register_row,
)
from tests.hermes_cli._kanban_worker_identity_support import (
    events_for,
    expire_claim,
    live_child,
    pid_is_alive,
    process_start_time,
    task_identity,
)

SLUG = "restored"


def db_path() -> Path:
    return kb.kanban_db_path(board=SLUG)


def _publish_new_epoch(slug: str) -> "kb.RegisterEntry":
    """What a restored board does to the register: publish the next epoch.

    Through the real register primitive, which takes the per-board
    ``board_register_lock`` itself (see
    ``kanban_db.transition_register_entry``), so Gate A moves exactly the
    way every other transition moves it — and wrapping it in that lock here
    would deadlock against its own acquisition.
    """
    entry = kb.get_register_entry(slug)
    assert entry is not None
    new_epoch = entry.epoch + 1
    updated = kb.RegisterEntry(
        board_name=slug,
        lifecycle=kb.BoardLifecycle.LIVE,
        epoch=new_epoch,
        epoch_before=entry.epoch,
        gate_move=kb.GateMove.SETTLED,
        epoch_lineage=(entry.epoch_lineage or [entry.epoch]) + [new_epoch],
        created_at=entry.created_at,
        updated_at=int(time.time()),
    )
    kb.transition_register_entry(updated)
    return updated


def _mirror_the_new_epoch(slug: str, epoch: int) -> None:
    """Gate B catching up, through the shipped atomic gate+mirror commit."""
    conn = kb.connect(board=slug)
    try:
        with kb._fence_protocol_scope():
            with kb.write_txn(conn):
                assert kb.commit_gate_state(conn, kb.InBoardGate.OPEN, int(epoch))
    finally:
        conn.close()


@pytest.fixture
def restored_board(fence_home, monkeypatch, tmp_path):
    """A board with work in flight, backed up, then restored from that copy.

    Returns everything the assertions need about the pre-restore identity.
    """
    monkeypatch.setenv("HERMES_KANBAN_CRASH_GRACE_SECONDS", "0")
    create_fenced_board(SLUG)
    conn = kb.connect(board=SLUG)
    try:
        running = ready_task(conn, title="in flight")
        spare = ready_task(conn, title="not yet claimed")
        assert kb.claim_task(conn, running) is not None
        with live_child() as child:
            kb._set_worker_pid(conn, running, child.pid)
            witness = process_start_time(child.pid)
            worker_pid = child.pid
            # The backup is taken while the worker is alive, so the copy
            # really does carry a live claim and a bound identity.
            backup = Path(tmp_path) / "board-backup.db"
            shutil.copy2(db_path(), backup)
    finally:
        conn.close()

    # The outage: the worker does not survive it.
    for _ in range(100):
        if not pid_is_alive(worker_pid):
            break
        time.sleep(0.05)
    assert not pid_is_alive(worker_pid)

    # The restore: the board file really is put back from the copy.
    shutil.copy2(backup, db_path())
    kb._INITIALIZED_PATHS.discard(str(db_path().resolve()))

    return {
        "running": running,
        "spare": spare,
        "worker_pid": worker_pid,
        "witness": witness,
    }


# ---------------------------------------------------------------------------
# The restore crosses the identity intact
# ---------------------------------------------------------------------------

def test_the_recorded_identity_survives_the_restore(restored_board):
    row = task_identity(db_path(), restored_board["running"])
    assert row["status"] == "running"
    assert row["worker_pid"] == restored_board["worker_pid"]
    assert row["worker_start_time"] == restored_board["witness"], (
        "the identity witness is durable board state, not process state"
    )


# ---------------------------------------------------------------------------
# Gate B on the register: the new epoch stands the old work down
# ---------------------------------------------------------------------------

def test_old_handles_and_claims_fail_once_the_new_epoch_is_published(
    restored_board,
):
    old_handle = kb.connect(board=SLUG)
    try:
        # The handle works before the restored board republishes itself.
        assert kb.heartbeat_claim(old_handle, restored_board["running"]) is True
        # Age the claim so the stale sweep below really has a row to act on:
        # a sweep that selects nothing never reaches a mutation, and a
        # refusal it never attempted would prove nothing.
        expire_claim(old_handle, restored_board["running"])

        entry = _publish_new_epoch(SLUG)

        # Asserted through Gate B on the register, not through a boolean a
        # validator handed back: the register says 2, the in-board mirror
        # still says 1, and EM-4 judges that pair indeterminate.
        assert register_row(SLUG)["epoch"] == 2
        assert gate_row(db_path()) == ("open", 1)
        verdict = kb.validate_epoch_pair(2, 1, kb.get_register_entry(SLUG))
        assert verdict.valid is False
        assert verdict.rule is kb.FenceRefusalRule.EM_4c
        assert kb.accepted_epoch_mirrors(entry) == frozenset({2}), (
            "the old mirror is not an accepted value for the new epoch"
        )

        # The old handle's every mutation is refused — the worker's
        # heartbeat, its completion, a new claim, and the dispatcher's own
        # stale-claim sweep alike.
        for mutate in (
            lambda: kb.heartbeat_claim(old_handle, restored_board["running"]),
            lambda: kb.complete_task(
                old_handle, restored_board["running"], result="done"
            ),
            lambda: kb.claim_task(old_handle, restored_board["spare"]),
            lambda: kb.release_stale_claims(
                old_handle, signal_fn=lambda *_a: None
            ),
        ):
            with pytest.raises(kb.BoardFenceClosedError):
                mutate()

        # Nothing moved on disk.
        row = task_identity(db_path(), restored_board["running"])
        assert row["status"] == "running"
        assert row["worker_pid"] == restored_board["worker_pid"]
        assert row["worker_start_time"] == restored_board["witness"]
        assert task_identity(db_path(), restored_board["spare"])["status"] == "ready"
    finally:
        old_handle.close()


def test_once_gate_b_catches_up_the_stale_claim_is_released_on_its_identity(
    restored_board,
):
    """The epoch is what stands the old work down; the recorded identity is
    what resolves the claim it left behind."""
    _publish_new_epoch(SLUG)
    _mirror_the_new_epoch(SLUG, 2)

    assert gate_row(db_path()) == ("open", 2)
    assert kb.validate_epoch_pair(2, 2, kb.get_register_entry(SLUG)).valid is True

    conn = kb.connect(board=SLUG)
    try:
        expire_claim(conn, restored_board["running"])
        assert kb.release_stale_claims(conn, signal_fn=lambda *_a: None) == 1
    finally:
        conn.close()

    row = task_identity(db_path(), restored_board["running"])
    assert row["status"] == "ready"
    assert row["claim_lock"] is None
    assert row["worker_pid"] is None

    reclaimed = events_for(db_path(), restored_board["running"], "reclaimed")
    assert len(reclaimed) == 1
    payload = reclaimed[0][1]
    assert payload["worker_pid"] == restored_board["worker_pid"]
    assert payload["worker_start_time"] == restored_board["witness"]
    assert payload["worker_presence"] == kb.HolderPresence.PROVABLY_ABSENT.value
    assert payload["automatic"] is True
