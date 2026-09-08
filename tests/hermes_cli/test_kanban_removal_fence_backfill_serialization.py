"""An active backfill and recovery may not run past each other (Defect 2).

Reviewer reproduction: a real child process recorded the ``mirroring``
phase, the real :func:`recover_backfill_intent` ran INSIDE that
phase→board-commit window, saw no Gate B yet and no register entry,
deleted the LIVE intent and returned ``abandoned`` — and the child then
committed Gate B and died at the board-commit boundary. What was left on
disk was Gate B ``open/1``, no register row and no intent at all. A later
recovery returned ``None`` and changed nothing: the half-armed board was
permanently unrecoverable.

The second half of the same defect: ``_set_backfill_intent_phase`` blindly
reported success for an ``UPDATE`` that changed ZERO rows, so an attempt
whose journal had been discharged under it carried on regardless and
reported a completed backfill.

Everything here is real: a real child process running the real public
``backfill_register_entry``, real cross-process SQLite writer locks that
park it at a chosen boundary, a real ``SIGKILL`` for the crash, and plain
read-only connections for every assertion about what is on disk.
"""

from __future__ import annotations

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

from hermes_cli import kanban_db as kb
from tests.hermes_cli._kanban_fence_support import (
    gate_row,
    hold_write_lock,
    intent_row,
    intent_rows,
    make_legacy_board,
    read_only,
    register_row,
)

# SIGKILL does not exist on Windows; fall back to SIGTERM there so this
# module still imports. The crash-recovery behaviour under test is POSIX.
_SIGKILL = getattr(signal, "SIGKILL", signal.SIGTERM)


_CHILD_SOURCE = """
import json, sys
sys.path.insert(0, sys.argv[1])
from hermes_cli import kanban_db as kb
result = kb.backfill_register_entry(sys.argv[2])
print("RESULT " + json.dumps({"success": bool(result.success),
                              "message": str(result.message)}), flush=True)
"""


def _spawn_backfill(home: Path, slug: str) -> subprocess.Popen:
    """Run the REAL public backfill for *slug* in a separate OS process."""
    return subprocess.Popen(
        [sys.executable, "-c", _CHILD_SOURCE, str(_WORKTREE), slug],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env={**os.environ, "HERMES_HOME": str(home), "PYTHONUNBUFFERED": "1"},
    )


def _wait_for(predicate, *, timeout: float = 60.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(0.01)
    return None


def _gate_table_present(db_path: Path) -> bool:
    try:
        conn = sqlite3.connect(f"{db_path.resolve().as_uri()}?mode=ro", uri=True)
    except sqlite3.Error:
        return False
    try:
        conn.execute("PRAGMA busy_timeout=2000")
        return (
            conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' "
                "AND name='board_fence_state' LIMIT 1"
            ).fetchone()
            is not None
        )
    except sqlite3.Error:
        return False
    finally:
        conn.close()


def _half_armed(slug: str, db_path: Path) -> bool:
    """Gate B open, nothing in the register — the forbidden shape."""
    gate = gate_row(db_path)
    return gate is not None and gate[0] == "open" and register_row(slug) is None


def test_defect_2_recovery_cannot_discharge_an_active_backfills_intent(
    fence_home, tmp_path
):
    """Recovery attempted inside a live ``mirroring``→board-commit window.

    The child is parked at the board commit by a real RESERVED lock held
    from a third process (reads still work, so the child got all the way
    through its admission checks and its intent journal). Recovery then
    runs for real, the child is released, and it is SIGKILLed the moment
    its Gate B lands — the reviewer's exact crash boundary.
    """
    slug = "backfill-recovery-race"
    db_path = make_legacy_board(tmp_path, slug)
    archive_path = kb.archive_db_path()

    child = None
    try:
        with hold_write_lock(db_path) as release_board:
            child = _spawn_backfill(fence_home, slug)
            # The intent journal reached ``mirroring``: the child is now
            # inside the window this defect is about.
            parked = _wait_for(
                lambda: (intent_row(slug) or {}).get("phase")
                == kb.BACKFILL_PHASE_MIRRORING
            )
            assert parked, (
                "the child never reached the mirroring phase "
                f"(intent={intent_row(slug)!r}, alive={child.poll() is None})"
            )

            # The real recovery, concurrent with the real backfill.
            mid = kb.recover_backfill_intent(slug)

            assert intent_rows(slug) == 1, (
                f"recovery ({mid!r}) discharged the ACTIVE backfill's intent "
                "while it was still running"
            )

            # Park the child again at its NEXT write, this time on the
            # archive, so the crash can be aimed at the board-commit
            # boundary rather than raced for.
            with hold_write_lock(archive_path) as release_archive:
                release_board()
                landed = _wait_for(lambda: _gate_table_present(db_path))
                assert landed, "the child never committed Gate B"
                child.send_signal(_SIGKILL)
                child.wait(timeout=30)
                release_archive()
    finally:
        if child is not None and child.poll() is None:
            child.kill()
            child.wait(timeout=30)

    # What the crash left behind, read straight off disk.
    gate_before = gate_row(db_path)
    entry_before = register_row(slug)
    intents_before = intent_rows(slug)
    was_half_armed = _half_armed(slug, db_path)

    final = kb.recover_backfill_intent(slug)

    if was_half_armed:
        assert final is not None, (
            "the board is Gate B "
            f"{gate_before!r} with register entry {entry_before!r} and "
            f"{intents_before} intent row(s), and recovery returned None: "
            "there is nothing left that can ever repair it"
        )
    assert not _half_armed(slug, db_path), (
        "after recovery the board is STILL half-armed: Gate B "
        f"{gate_row(db_path)!r}, register entry {register_row(slug)!r}, "
        f"{intent_rows(slug)} intent row(s)"
    )


def test_defect_2_a_phase_update_that_changes_no_row_is_a_failure(
    fence_home, tmp_path
):
    """A zero-row phase ``UPDATE`` is not a recorded phase.

    The intent is discharged out from under a parked attempt by an actor
    outside the protocol (an operator repair, a stale sibling). The next
    phase write therefore matches nothing. Reporting that as success let
    the attempt walk on and finish a backfill whose journal no longer
    described it.
    """
    slug = "backfill-phase-cas"
    db_path = make_legacy_board(tmp_path, slug)

    child = None
    try:
        with hold_write_lock(db_path) as release_board:
            child = _spawn_backfill(fence_home, slug)
            parked = _wait_for(
                lambda: (intent_row(slug) or {}).get("phase")
                == kb.BACKFILL_PHASE_MIRRORING
            )
            assert parked, "the child never reached the mirroring phase"

            # Somebody outside the protocol discharges the journal row.
            with sqlite3.connect(str(kb.archive_db_path())) as arch:
                arch.execute(
                    "DELETE FROM board_backfill_intent WHERE board_name = ?",
                    (slug,),
                )
            assert intent_rows(slug) == 0
            release_board()
        stdout, stderr = child.communicate(timeout=120)
    finally:
        if child is not None and child.poll() is None:
            child.kill()
            child.wait(timeout=30)

    line = [ln for ln in stdout.splitlines() if ln.startswith("RESULT ")]
    assert line, f"the child produced no result: {stdout!r} / {stderr!r}"
    import json

    result = json.loads(line[0][len("RESULT "):])
    assert result["success"] is False, (
        "the backfill reported SUCCESS after a phase write that changed no "
        f"rows: {result['message']!r}"
    )
    # And it unwound rather than leaving a gate nobody vouches for.
    assert not _half_armed(slug, db_path), (
        f"Gate B {gate_row(db_path)!r} with register entry "
        f"{register_row(slug)!r}"
    )
    with read_only(db_path) as conn:
        assert conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0] >= 1
