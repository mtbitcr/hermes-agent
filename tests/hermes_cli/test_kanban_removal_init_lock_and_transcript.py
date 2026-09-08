"""Tests for the init-lock retention (DEFECT A) and transcript store
verification (DEFECT B).

DEFECT A: the init-lock is destroyed during apply (journalled), then
re-materialised by every subsequent register-lock acquisition (swept,
done, journal writes).  Removal cannot be made final without re-creating
the file, so it is explicitly retained — the receipt states this, and
the ledger carries its disposition as retained-by-design.  A receipt
that does not account for the init-lock's disposition cannot produce an
all-PASS overall verdict.

DEFECT B: an absent transcript store with attributed sessions must NOT
produce a PASS verdict.  PASS is only defensible when the store is
present OR when it is absent and zero sessions are attributed.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import sqlite3
import sys
from pathlib import Path

import pytest

_WORKTREE = Path(__file__).resolve().parents[2]
if str(_WORKTREE) not in sys.path:
    sys.path.insert(0, str(_WORKTREE))

from hermes_cli import kanban_db as kb
from tests.hermes_cli._kanban_fence_support import (
    add_linked_work_area,
    create_fenced_board,
    make_git_repo,
    permanent_confirmation,
    read_only,
    ready_task,
    record_git_receipt,
    register_row,
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _run_cli(argv: list) -> "tuple[int, str]":
    from hermes_cli import kanban as kanban_cli

    wrap = argparse.ArgumentParser(prog="hermes-test", add_help=False)
    top = wrap.add_subparsers(dest="_top")
    parser = kanban_cli.build_parser(top)
    args = parser.parse_args(argv)
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(buffer):
        code = kanban_cli.kanban_command(args)
    return code, buffer.getvalue()


def _remove_via_cli(slug: str) -> "tuple[int, str]":
    required = kb.permanent_removal_disclosure(slug).required_response
    return _run_cli(["boards", "rm", slug, "--delete", "--confirm", required])


def _board(slug: str, repo: Path) -> dict:
    """A fenced board with a real task and a real linked work area."""
    base = make_git_repo(repo)
    create_fenced_board(slug)
    conn = kb.connect(board=slug)
    task_id = ready_task(conn)
    branch = f"hermes/{task_id}"
    work_area, head = add_linked_work_area(repo, task_id, branch=branch)
    record_git_receipt(
        conn, task_id, workspace_path=work_area, branch_name=branch,
        base_commit=base, head_commit=head,
    )
    conn.close()
    return {"slug": slug, "task": task_id, "work_area": work_area}


def _board_with_session(slug: str, repo: Path, session_id: str) -> dict:
    """A fenced board whose task carries a session_id."""
    base = make_git_repo(repo)
    create_fenced_board(slug)
    conn = kb.connect(board=slug)
    task_id = kb.create_task(
        conn, title="work-with-session", assignee="worker",
        session_id=session_id,
    )
    with kb.write_txn(conn):
        conn.execute(
            "UPDATE tasks SET status = 'ready' WHERE id = ?", (task_id,),
        )
    branch = f"hermes/{task_id}"
    work_area, head = add_linked_work_area(repo, task_id, branch=branch)
    record_git_receipt(
        conn, task_id, workspace_path=work_area, branch_name=branch,
        base_commit=base, head_commit=head,
    )
    conn.close()
    return {"slug": slug, "task": task_id, "work_area": work_area}


# ---------------------------------------------------------------------------
# DEFECT A: init-lock is retained by design
# ---------------------------------------------------------------------------

class TestInitLockRetainedByDesign:
    """The init-lock is retained, not silently destroyed or silently left."""

    def test_init_lock_retained_and_disclosed_on_receipt(
        self, fence_home, tmp_path
    ):
        """The receipt carries an init-lock-retained check with PASS."""
        _board("init-lock-retained", tmp_path / "repo")
        code, output = _remove_via_cli("init-lock-retained")
        assert code == 0, output

        record = kb.get_removal_phase_record("init-lock-retained")
        receipt = kb.get_permanent_removal_receipt(
            "init-lock-retained", record.removal_id
        )
        assert receipt is not None

        checks = receipt["verification"]["checks"]
        init_checks = [
            c for c in checks if c["check"] == "init-lock-retained"
        ]
        assert len(init_checks) == 1, "no init-lock-retained check on receipt"
        check = init_checks[0]
        assert check["verdict"] == kb.RECEIPT_VERDICT_PASS
        assert "retained" in check["evidence"]["disposition"]

    def test_init_lock_in_retained_records(self, fence_home, tmp_path):
        """The honesty section lists the init-lock as retained."""
        _board("init-lock-honesty", tmp_path / "repo")
        code, output = _remove_via_cli("init-lock-honesty")
        assert code == 0, output

        record = kb.get_removal_phase_record("init-lock-honesty")
        receipt = kb.get_permanent_removal_receipt(
            "init-lock-honesty", record.removal_id
        )
        retained = receipt["honesty"]["retained_records"]
        init_lock_identity = str(
            kb.register_lock_path("init-lock-honesty").with_name(
                kb.register_lock_path("init-lock-honesty").name + ".init.lock"
            )
        )
        init_lock_entries = [
            r for r in retained
            if r.get("identity") == init_lock_identity
        ]
        assert init_lock_entries, (
            "the init-lock is not listed in the receipt's retained records"
        )

    def test_init_lock_retention_journalled(self, fence_home, tmp_path):
        """The carry ledger records register-lock-init as OUT (retained)."""
        _board("init-lock-ledger", tmp_path / "repo")
        code, output = _remove_via_cli("init-lock-ledger")
        assert code == 0, output

        record = kb.get_removal_phase_record("init-lock-ledger")
        journal = record.journal()
        retention_entries = [
            item for item in journal["items"]
            if item.get("action") == kb.APPLY_JOURNAL_RETENTION_RECORDED
            and "init.lock" in str(item.get("identity", ""))
        ]
        assert retention_entries, (
            "the init-lock retention was not journalled"
        )
        assert retention_entries[0]["ok"] is True

    def test_missing_init_lock_retention_blocks_removal(
        self, fence_home, tmp_path, monkeypatch
    ):
        """If the init-lock retention is not journalled, the removal
        CANNOT reach Done — the apply journal completeness check blocks
        it.  This is stronger than a receipt-level check: the removal
        itself refuses rather than producing a receipt that understates
        what was retained.

        Monkeypatching the journal is necessary because root defeats
        permission bits on most Linux filesystems.
        """
        _board("init-lock-blocked", tmp_path / "repo")

        real_journal = kb.journal_apply_item
        init_lock_identity = str(
            kb.register_lock_path("init-lock-blocked").with_name(
                kb.register_lock_path("init-lock-blocked").name
                + ".init.lock"
            )
        )

        def _skip_init_lock_journal(slug, *, removal_id, phase, item):
            if (
                item.get("action") == kb.APPLY_JOURNAL_RETENTION_RECORDED
                and item.get("identity") == init_lock_identity
            ):
                return True
            return real_journal(
                slug, removal_id=removal_id, phase=phase, item=item,
            )

        with monkeypatch.context() as m:
            m.setattr(kb, "journal_apply_item", _skip_init_lock_journal)
            code, output = _remove_via_cli("init-lock-blocked")

        assert code != 0, (
            "the removal succeeded despite a missing init-lock retention "
            "journal entry — the apply journal completeness check did "
            "not block it"
        )
        record = kb.get_removal_phase_record("init-lock-blocked")
        assert record.phase != kb.RemovalPhase.DONE, (
            "the removal reached Done without journalling the init-lock "
            "retention"
        )
        receipt = kb.get_permanent_removal_receipt(
            "init-lock-blocked", record.removal_id
        )
        assert receipt is None, (
            "a receipt was issued for a removal that did not journal the "
            "init-lock retention"
        )


# ---------------------------------------------------------------------------
# DEFECT B: transcript store verification
# ---------------------------------------------------------------------------

class TestTranscriptStoreVerification:
    """The transcript retention check splits on store presence x session count."""

    def test_store_present_is_pass(self, fence_home, tmp_path):
        """A present store produces a PASS verdict."""
        _board("transcript-present", tmp_path / "repo")

        store_path = kb.conversation_transcript_store_path()
        store_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(store_path))
        conn.execute("CREATE TABLE IF NOT EXISTS _marker (x INTEGER)")
        conn.close()
        assert store_path.exists()

        code, output = _remove_via_cli("transcript-present")
        assert code == 0, output

        record = kb.get_removal_phase_record("transcript-present")
        receipt = kb.get_permanent_removal_receipt(
            "transcript-present", record.removal_id
        )
        assert receipt is not None

        check = _transcript_check(receipt)
        assert check["verdict"] == kb.RECEIPT_VERDICT_PASS
        assert "verified present" in check["observed"]

    def test_store_absent_zero_sessions_is_pass(self, fence_home, tmp_path):
        """Absent store with zero sessions is a consistent PASS."""
        _board("transcript-absent-0", tmp_path / "repo")

        store_path = kb.conversation_transcript_store_path()
        if store_path.exists():
            store_path.unlink()
        assert not store_path.exists()

        code, output = _remove_via_cli("transcript-absent-0")
        assert code == 0, output

        record = kb.get_removal_phase_record("transcript-absent-0")
        receipt = kb.get_permanent_removal_receipt(
            "transcript-absent-0", record.removal_id
        )
        assert receipt is not None

        check = _transcript_check(receipt)
        assert check["verdict"] == kb.RECEIPT_VERDICT_PASS
        assert "0 session" in check["observed"]

    def test_store_absent_with_sessions_is_not_pass(
        self, fence_home, tmp_path
    ):
        """Absent store with attributed sessions must NOT be PASS.

        Sessions are attributed by creating a task with a session_id
        through the real create_task API — the same path the gateway
        takes.  _board_work_scope reads these from the tasks table and
        populates originating_sessions on the carry ledger.
        """
        _board_with_session(
            "transcript-absent-n", tmp_path / "repo",
            session_id="test-session-abc",
        )

        store_path = kb.conversation_transcript_store_path()
        if store_path.exists():
            store_path.unlink()
        assert not store_path.exists()

        code, output = _remove_via_cli("transcript-absent-n")
        assert code == 0, output

        record = kb.get_removal_phase_record("transcript-absent-n")
        receipt = kb.get_permanent_removal_receipt(
            "transcript-absent-n", record.removal_id
        )
        assert receipt is not None

        check = _transcript_check(receipt)
        assert check["verdict"] != kb.RECEIPT_VERDICT_PASS, (
            f"an absent store with attributed sessions produced PASS: "
            f"{check['observed']}"
        )
        assert check["verdict"] == kb.RECEIPT_VERDICT_UNVERIFIED
        assert "1 session" in check["observed"]
        assert check["evidence"]["store_present"] is False

        transcripts = receipt["transcripts"]
        assert transcripts["store_present"] is False
        assert transcripts["retention_verified"] is False
        assert transcripts["count"] >= 1

    def test_transcript_section_retention_verified_present(
        self, fence_home, tmp_path
    ):
        """The §12.4 section's retention_verified is True when present."""
        _board("transcript-section-ok", tmp_path / "repo")
        store_path = kb.conversation_transcript_store_path()
        store_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(store_path))
        conn.execute("CREATE TABLE IF NOT EXISTS _marker (x INTEGER)")
        conn.close()

        code, output = _remove_via_cli("transcript-section-ok")
        assert code == 0, output

        record = kb.get_removal_phase_record("transcript-section-ok")
        receipt = kb.get_permanent_removal_receipt(
            "transcript-section-ok", record.removal_id
        )
        transcripts = receipt["transcripts"]
        assert transcripts["store_present"] is True
        assert transcripts["retention_verified"] is True


def _transcript_check(receipt: dict) -> dict:
    """Extract the transcripts-retained-and-disclosed check."""
    checks = receipt["verification"]["checks"]
    matched = [
        c for c in checks
        if c["check"] == "transcripts-retained-and-disclosed"
    ]
    assert len(matched) == 1, f"expected 1 transcript check, got {len(matched)}"
    return matched[0]
