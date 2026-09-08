"""The backfill journal describes the board BEFORE it touches it (Defect 2).

Reviewer reproduction: the phased backfill committed Gate B into the board
store and only THEN advanced its intent to ``mirrored`` — and ignored the
result of that phase write. A subprocess killed inside the phase write
left Gate B ``open/1`` durable, NO register entry, and an intent still
labelled ``prepared``. ``recover_backfill_intent`` treated ``prepared`` as
"never touched the board", returned ``abandoned``, deleted the intent and
left the board carrying a gate with no authority — the expressly forbidden
half-armed board.

The crash here is a REAL one: a child process running the REAL
``backfill_register_entry`` calls ``os._exit`` at the exact
board-commit → phase-record boundary. Everything asserted afterwards is
read back off disk with plain read-only connections.
"""

from __future__ import annotations

import os
import sqlite3
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

_WORKTREE = Path(__file__).resolve().parents[2]
if str(_WORKTREE) not in sys.path:
    sys.path.insert(0, str(_WORKTREE))

from hermes_cli import kanban_db as kb
from tests.hermes_cli._kanban_fence_support import (
    gate_row,
    has_gate_table,
    intent_rows,
    make_legacy_board,
    read_only,
    register_row,
)


CRASHING_BACKFILL = textwrap.dedent(
    """
    import os, sqlite3, sys

    sys.path.insert(0, {worktree!r})
    from hermes_cli import kanban_db as kb

    slug = sys.argv[1]
    db_path = kb.kanban_db_path(board=slug)


    def gate_b_durable():
        con = sqlite3.connect(
            f"{{db_path.resolve().as_uri()}}?mode=ro", uri=True
        )
        try:
            return con.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' "
                "AND name='board_fence_state' LIMIT 1"
            ).fetchone() is not None
        finally:
            con.close()


    _real_phase = kb._set_backfill_intent_phase


    def crashing_phase(*args):
        # Die at the FIRST phase write that happens once Gate B is durable
        # on the board — i.e. exactly at the board-commit -> phase-record
        # boundary, wherever the implementation puts its labels. The phase
        # being moved TO is the last positional argument in every form of
        # the compare-and-set signature.
        phase = args[-1]
        if gate_b_durable():
            sys.stderr.write("crashing-at-phase=" + str(phase) + chr(10))
            sys.stderr.flush()
            os._exit(9)
        return _real_phase(*args)


    kb._set_backfill_intent_phase = crashing_phase
    result = kb.backfill_register_entry(slug)
    sys.stdout.write("NO-CRASH " + repr(result) + chr(10))
    """
)


def _intent_phase(slug: str):
    path = kb.archive_db_path()
    if not path.exists():
        return None
    with read_only(path) as conn:
        row = conn.execute(
            "SELECT phase FROM board_backfill_intent WHERE board_name = ?",
            (slug,),
        ).fetchone()
    return None if row is None else str(row["phase"])


def _crash_a_real_backfill(slug: str) -> subprocess.CompletedProcess:
    """Run the REAL backfill in a child and kill it at the boundary."""
    return subprocess.run(
        [
            sys.executable,
            "-c",
            CRASHING_BACKFILL.format(worktree=str(_WORKTREE)),
            slug,
        ],
        capture_output=True,
        text=True,
        timeout=180,
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
    )


def _assert_no_half_armed_board(slug: str, db_path: Path) -> None:
    """The invariant: a Gate B with no authority is never left at rest.

    Acceptable end states are exactly three — fully armed with an
    authority, Gate B compensated/closed, or the intent retained so a
    later retry finishes the job.
    """
    entry = register_row(slug)
    gate = gate_row(db_path) if db_path.exists() else None
    retained = intent_rows(slug) > 0

    if gate is None:
        return  # compensated: no Gate B at all
    if entry is not None:
        assert gate[1] == entry["epoch"], (
            f"Gate B mirror {gate[1]} does not match authority {entry['epoch']}"
        )
        return  # fully armed
    if gate[0] != "open":
        return  # closed against a missing authority — fail-closed
    assert retained, (
        f"board {slug!r} is half-armed: Gate B is {gate!r}, there is no "
        "register authority, and no intent was retained to finish the unwind"
    )


# ---------------------------------------------------------------------------
# The reproduction
# ---------------------------------------------------------------------------

def test_defect_2_a_crash_at_the_phase_boundary_leaves_no_half_armed_board(
    fence_home, tmp_path
):
    slug = "crashed"
    db_path = make_legacy_board(tmp_path, slug)

    proc = _crash_a_real_backfill(slug)
    assert proc.returncode == 9, (
        f"the child did not crash at the boundary: rc={proc.returncode} "
        f"stdout={proc.stdout!r} stderr={proc.stderr[-2000:]!r}"
    )
    assert "crashing-at-phase=" in proc.stderr

    # The reviewer's persisted state: Gate B durable, no authority, one
    # unfinished intent.
    assert has_gate_table(db_path)
    assert gate_row(db_path) == ("open", 1)
    assert register_row(slug) is None
    assert intent_rows(slug) == 1

    # (a) The surviving phase must say a Gate B MAY exist. Recorded before
    # the board commit, it describes the crash window for its whole
    # duration instead of appearing only after the window closes.
    assert _intent_phase(slug) in kb.BACKFILL_PHASES_MAY_HAVE_GATE_B, (
        f"intent survived labelled {_intent_phase(slug)!r}, which recovery "
        "reads as 'the board was never touched'"
    )

    # (b) The real recovery inspects the board and compensates it.
    outcome = kb.recover_backfill_intent(slug)
    assert outcome is not None

    _assert_no_half_armed_board(slug, db_path)
    assert not has_gate_table(db_path), (
        "the Gate B this attempt installed was left behind with no authority"
    )
    assert register_row(slug) is None
    assert intent_rows(slug) == 0


def test_defect_2_the_board_is_backfillable_again_after_the_crash_recovery(
    fence_home, tmp_path
):
    """Recovery restores the board to a state a retry can complete from."""
    slug = "retryable"
    db_path = make_legacy_board(tmp_path, slug)

    proc = _crash_a_real_backfill(slug)
    assert proc.returncode == 9, proc.stderr[-2000:]

    kb.recover_backfill_intent(slug)

    result = kb.backfill_register_entry(slug)
    assert result.success, result.message
    assert register_row(slug) == {
        "lifecycle": "live",
        "epoch": 1,
        "epoch_before": None,
        "gate_move": "settled",
    }
    assert gate_row(db_path) == ("open", 1)
    _assert_no_half_armed_board(slug, db_path)

    # And the board really is writable through the real API now.
    conn = kb.connect(board=slug)
    try:
        task_id = kb.create_task(conn, title="after recovery", assignee="worker")
    finally:
        conn.close()
    with read_only(db_path) as ro:
        assert ro.execute(
            "SELECT 1 FROM tasks WHERE id = ?", (task_id,)
        ).fetchone() is not None


# ---------------------------------------------------------------------------
# (c) No phase-record failure is ever ignored
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "failing_phase",
    [
        kb.BACKFILL_PHASE_MIRRORING,
        kb.BACKFILL_PHASE_MIRRORED,
        kb.BACKFILL_PHASE_REGISTERED,
    ],
)
def test_defect_2_a_failed_phase_record_unwinds_or_retains(
    fence_home, tmp_path, monkeypatch, failing_phase
):
    slug = f"phase-{failing_phase}"
    db_path = make_legacy_board(tmp_path, slug)

    real_phase = kb._set_backfill_intent_phase

    def flaky(*args):
        if args[-1] == failing_phase:
            return False
        return real_phase(*args)

    monkeypatch.setattr(kb, "_set_backfill_intent_phase", flaky)

    result = kb.backfill_register_entry(slug)
    assert result.success is False, (
        f"a failed {failing_phase!r} phase write was ignored: {result.message}"
    )

    _assert_no_half_armed_board(slug, db_path)
    if failing_phase == kb.BACKFILL_PHASE_REGISTERED:
        # The authority landed; the intent must survive so recovery can
        # finish the archive side rather than forget it.
        assert intent_rows(slug) == 1
        assert kb.recover_backfill_intent(slug) == "finished"
        assert intent_rows(slug) == 0
    else:
        assert register_row(slug) is None
        assert not has_gate_table(db_path)
        assert intent_rows(slug) == 0
    _assert_no_half_armed_board(slug, db_path)


def test_defect_2_recovery_inspects_the_store_not_the_label(fence_home, tmp_path):
    """A ``prepared`` label over a durable Gate B is still compensated.

    Belt-and-braces for a journal written by an older build (or a crash
    that landed between two labels): the label says the board was never
    touched, the STORE says otherwise, and the store wins.
    """
    slug = "mislabelled"
    db_path = make_legacy_board(tmp_path, slug)

    proc = _crash_a_real_backfill(slug)
    assert proc.returncode == 9, proc.stderr[-2000:]
    assert has_gate_table(db_path)

    # Rewind the journal to the pre-repair label.
    with sqlite3.connect(str(kb.archive_db_path())) as arch:
        arch.execute(
            "UPDATE board_backfill_intent SET phase = ? WHERE board_name = ?",
            (kb.BACKFILL_PHASE_PREPARED, slug),
        )
    assert _intent_phase(slug) == kb.BACKFILL_PHASE_PREPARED

    assert kb.recover_backfill_intent(slug) == "compensated"
    assert not has_gate_table(db_path)
    _assert_no_half_armed_board(slug, db_path)
