"""A failed backfill may not walk away from a gate it overwrote (Defect 3).

Reviewer reproduction: on a board that ALREADY carried an orphan Gate B,
an injected ``mirrored`` phase-record failure made ``_unwind_backfill``
read ``gate_b_preexisting=True`` as permission to discharge the intent and
leave — even though phase 2b had just OVERWRITTEN that gate's mirror with
the backfill epoch. What was left was Gate B ``open/1``, no register entry
and no intent; ``recover_backfill_intent`` then returned ``None``. The
forbidden half-armed shape, and permanent.

Every phase-record failure is exercised here (``mirroring``, ``mirrored``,
``registered``) against BOTH values of pre-existing Gate B. The failure
itself is real: for the targeted phase the archive write runs while a
SECOND OS process holds a real writer lock on the archive store, so the
``UPDATE`` genuinely cannot land. Nothing about the outcome is simulated.
"""

from __future__ import annotations

import contextlib
import sys
from pathlib import Path

import pytest

_WORKTREE = Path(__file__).resolve().parents[2]
if str(_WORKTREE) not in sys.path:
    sys.path.insert(0, str(_WORKTREE))

from hermes_cli import kanban_db as kb
from tests.hermes_cli._kanban_fence_support import (
    gate_row,
    hold_write_lock,
    intent_rows,
    make_legacy_board,
    make_orphan_gate_board,
    read_only,
    register_row,
)


# Deliberately not 1: a restored mirror has to be distinguishable from the
# epoch the backfill would have written.
PRIOR_MIRROR = 7


@contextlib.contextmanager
def failing_phase_record(monkeypatch, target_phase: str):
    """Make exactly ONE phase write fail, and fail for a REAL reason.

    The shim decides *which* write fails; a real cross-process RESERVED
    lock on the archive store is what makes it fail. The lock is taken and
    released around that single call, so the unwind that follows can still
    write.
    """
    real = kb._set_backfill_intent_phase

    def shim(*args, **kwargs):
        phase = kwargs.get("phase", args[-1] if args else None)
        if phase != target_phase:
            return real(*args, **kwargs)
        with hold_write_lock(kb.archive_db_path()):
            return real(*args, **kwargs)

    monkeypatch.setattr(kb, "_set_backfill_intent_phase", shim)
    yield


def _forbidden(slug: str, db_path: Path) -> bool:
    """Gate B open, no register authority, and no intent left to fix it."""
    gate = gate_row(db_path)
    return (
        gate is not None
        and gate[0] == kb.InBoardGate.OPEN.value
        and register_row(slug) is None
        and intent_rows(slug) == 0
    )


def _describe(slug: str, db_path: Path) -> str:
    return (
        f"Gate B {gate_row(db_path)!r}, register entry {register_row(slug)!r}, "
        f"{intent_rows(slug)} intent row(s)"
    )


PHASES = (
    kb.BACKFILL_PHASE_MIRRORING,
    kb.BACKFILL_PHASE_MIRRORED,
    kb.BACKFILL_PHASE_REGISTERED,
)


@pytest.mark.parametrize("preexisting_gate_b", (False, True), ids=("no-gate", "orphan-gate"))
@pytest.mark.parametrize("phase", PHASES)
def test_defect_3_a_failed_phase_never_leaves_a_forbidden_board(
    fence_home, tmp_path, monkeypatch, phase, preexisting_gate_b
):
    """Six cases, one invariant: never open-with-no-authority-and-no-intent."""
    monkeypatch.setenv("HERMES_KANBAN_BUSY_TIMEOUT_MS", "400")
    slug = f"unwind-{phase}-{'orphan' if preexisting_gate_b else 'clean'}"
    if preexisting_gate_b:
        db_path = make_orphan_gate_board(tmp_path, slug, mirror=PRIOR_MIRROR)
        assert gate_row(db_path) == (kb.InBoardGate.OPEN.value, PRIOR_MIRROR)
    else:
        db_path = make_legacy_board(tmp_path, slug)
        assert gate_row(db_path) is None

    with failing_phase_record(monkeypatch, phase):
        result = kb.backfill_register_entry(slug)

    assert result.success is False, (
        f"the backfill reported success despite a failed {phase} phase write: "
        f"{result.message}"
    )
    assert not _forbidden(slug, db_path), (
        f"the failed {phase} phase left a permanently half-armed board: "
        f"{_describe(slug, db_path)}"
    )

    # Whatever it left, recovery must be able to bring it to rest — and it
    # must not make things worse.
    kb.recover_backfill_intent(slug)
    assert not _forbidden(slug, db_path), (
        f"recovery left the board half-armed after a failed {phase} phase: "
        f"{_describe(slug, db_path)}"
    )

    if preexisting_gate_b and phase != kb.BACKFILL_PHASE_REGISTERED:
        gate = gate_row(db_path)
        assert gate is not None, (
            "the attempt removed a Gate B it did not install: "
            f"{_describe(slug, db_path)}"
        )
        assert gate[1] == PRIOR_MIRROR, (
            "the board was left carrying the backfill's epoch mirror instead "
            f"of the one it had before the attempt: {gate!r}"
        )

    # The board's own data is never collateral.
    with read_only(db_path) as conn:
        assert conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0] >= 1
