"""§6.6's mode-specific content: outstanding at Applied, refused at Done.

§7.2 (permanent: destroy the storage, deregister the registrations) and
§7.3 (reversible: complete and verify the retained copy, remove the live
one) are deferred work. Deferring them is authorised. What is not
authorised is §6.8 reporting success — and the register entry recording
``hard-removed`` / ``archived`` — while they are outstanding: those are
durable claims about the world, and a restart reading ``hard-removed``
believes the board's content is gone.

So Applied records that its mode-specific content is OUTSTANDING (a
declared exception under IN-3: recorded AND surfaced), Done refuses while
it is, and :func:`kanban_db.record_applied_mode_content` — which verifies
the claim against durable state rather than believing the caller — is the
only thing that clears it.

Every terminal assertion here reads DURABLE state through a plain
read-only connection, not through the module under test.
"""

from __future__ import annotations

import json
import shutil
import sqlite3
import sys
from pathlib import Path

import pytest

_WORKTREE = Path(__file__).resolve().parents[2]
if str(_WORKTREE) not in sys.path:
    sys.path.insert(0, str(_WORKTREE))

from hermes_cli import kanban_db as kb
from tests.hermes_cli._kanban_fence_support import (
    apply_mode_specific_content,
    create_fenced_board,
    read_only,
    ready_task,
    register_row,
    start_removal,
)


# ---------------------------------------------------------------------------
# Durable readers — never through kanban_db
# ---------------------------------------------------------------------------

def _phase_row(slug: str):
    with read_only(kb.register_db_path()) as conn:
        return conn.execute(
            "SELECT * FROM board_removal_phase WHERE board_name = ?", (slug,)
        ).fetchone()


def _durable_marker(slug: str) -> dict:
    raw = _phase_row(slug)["applied_mode_content"]
    return json.loads(raw) if raw else {}


def _tasks_present(slug: str) -> list:
    with read_only(kb.kanban_db_path(board=slug)) as conn:
        return [dict(r) for r in conn.execute("SELECT id, status FROM tasks")]


# ---------------------------------------------------------------------------
# Sequences, driven through the real production entry points only
# ---------------------------------------------------------------------------

def _swept(slug: str, mode: str) -> str:
    """A full §6 sequence with one real task on the board, up to Swept."""
    create_fenced_board(slug)
    conn = kb.connect(board=slug)
    ready_task(conn)
    conn.close()
    intent = start_removal(slug, mode=mode)
    assert intent.success, intent.message
    removal_id = intent.removal_id
    assert kb.advance_removal_to_fenced(slug, removal_id=removal_id).success
    for driver in (
        kb.advance_removal_to_quiesced,
        kb.advance_removal_to_carried,
        kb.advance_removal_to_released,
        kb.advance_removal_to_applied,
        kb.advance_removal_to_swept,
    ):
        result = driver(slug, removal_id=removal_id)
        assert result.success, f"{driver.__name__}: {result.message}"
    return removal_id


def _retained_copy(slug: str, destination: Path) -> Path:
    """A real copy of the board's storage, made outside the board (§7.3)."""
    shutil.copytree(kb.board_dir(slug), destination)
    return destination


# ---------------------------------------------------------------------------
# The reproduction: a full sequence must NOT report a terminal fact
# ---------------------------------------------------------------------------

def test_a_full_permanent_sequence_refuses_done_while_7_2_is_outstanding(fence_home):
    """The exact counterexample: every phase succeeded, so Done must not.

    Applied and Swept mean what they say and commit. Done does not: the
    board's storage is still there, its task is still there, and
    ``hard-removed`` would be a false terminal fact.
    """
    removal_id = _swept("perm-outstanding", "permanent")

    done = kb.complete_removal(
        "perm-outstanding", removal_id=removal_id, outcome="probe"
    )

    assert done.success is False, done.message
    assert done.transitioned is False
    assert done.outcome is kb.RemovalAdvanceOutcome.REFUSED_MODE_CONTENT_OUTSTANDING
    assert "§7.2" in done.message

    # Durable state: neither table moved.
    assert register_row("perm-outstanding")["lifecycle"] == "removing"
    row = _phase_row("perm-outstanding")
    assert row["phase"] == "swept"
    assert row["outcome"] is None

    # And the reason is durably recorded, naming what is missing.
    recorded = json.loads(row["refusal_outcome"])
    assert recorded["outcome"] == "refused-mode-content-outstanding"
    assert "§7.2" in recorded["message"]
    assert "record_applied_mode_content" in recorded["message"]

    # The world the register would have lied about is still intact.
    assert kb.kanban_db_path(board="perm-outstanding").exists()
    assert kb.board_dir("perm-outstanding").exists()
    assert len(_tasks_present("perm-outstanding")) == 1


def test_a_full_reversible_sequence_refuses_done_while_7_3_is_outstanding(fence_home):
    removal_id = _swept("rev-outstanding", "reversible")

    done = kb.complete_removal(
        "rev-outstanding", removal_id=removal_id, outcome="probe"
    )

    assert done.success is False, done.message
    assert done.outcome is kb.RemovalAdvanceOutcome.REFUSED_MODE_CONTENT_OUTSTANDING
    assert "§7.3" in done.message
    assert register_row("rev-outstanding")["lifecycle"] == "removing"
    assert _phase_row("rev-outstanding")["phase"] == "swept"
    assert kb.kanban_db_path(board="rev-outstanding").exists()
    assert len(_tasks_present("rev-outstanding")) == 1


def test_repeated_done_attempts_never_accumulate_into_a_success(fence_home):
    removal_id = _swept("perm-retry", "permanent")

    for _ in range(3):
        assert kb.complete_removal(
            "perm-retry", removal_id=removal_id, outcome="probe"
        ).success is False

    assert register_row("perm-retry")["lifecycle"] == "removing"
    assert _phase_row("perm-retry")["phase"] == "swept"


# ---------------------------------------------------------------------------
# Applied records the outstanding marker, in the SAME transaction
# ---------------------------------------------------------------------------

def test_applied_records_the_outstanding_marker_with_the_advance(fence_home):
    create_fenced_board("marker-at-applied")
    removal_id = start_removal("marker-at-applied", mode="permanent").removal_id
    assert kb.advance_removal_to_fenced(
        "marker-at-applied", removal_id=removal_id
    ).success
    for driver in (
        kb.advance_removal_to_quiesced,
        kb.advance_removal_to_carried,
        kb.advance_removal_to_released,
    ):
        assert driver("marker-at-applied", removal_id=removal_id).success

    # Nothing is recorded before Applied — the phase that owes the content.
    assert _phase_row("marker-at-applied")["applied_mode_content"] is None

    applied = kb.advance_removal_to_applied(
        "marker-at-applied", removal_id=removal_id
    )
    assert applied.success and applied.transitioned

    row = _phase_row("marker-at-applied")
    assert row["phase"] == "applied"
    marker = json.loads(row["applied_mode_content"])
    assert marker["state"] == kb.APPLIED_MODE_CONTENT_OUTSTANDING
    assert marker["mode"] == "permanent"
    assert marker["rule"] == "§7.2"
    assert marker["evidence"] is None
    assert marker["seam"] == "record_applied_mode_content"

    item = marker["operator_item"]
    assert item["kind"] == kb.APPLIED_MODE_CONTENT_ITEM_KIND
    assert item["identity"] == {
        "board": "marker-at-applied",
        "removal_id": removal_id,
        "mode": "permanent",
        "step": "§7.2",
    }
    assert "§7.2" in item["detail"] and "hard-removed" in item["detail"]


def test_the_outstanding_item_is_reported_as_an_operator_item(fence_home):
    removal_id = _swept("rev-item", "reversible")

    items = kb.removal_operator_items(kb.get_removal_phase_record("rev-item"))

    assert len(items) == 1
    assert items[0]["kind"] == kb.APPLIED_MODE_CONTENT_ITEM_KIND
    assert items[0]["identity"]["step"] == "§7.3"
    assert "archived" in items[0]["detail"]


def test_a_fresh_intent_clears_the_previous_runs_marker(fence_home):
    """A new removal must not inherit the last one's outstanding item —
    nor its cleared one."""
    removal_id = _swept("marker-reset", "reversible")
    assert _phase_row("marker-reset")["applied_mode_content"]

    # Abandonment is later work, so put the entry back by hand and start
    # a second, genuinely new removal for the same board name.
    entry = kb.get_register_entry("marker-reset")
    kb.transition_register_entry(
        kb.RegisterEntry(
            board_name="marker-reset",
            lifecycle=kb.BoardLifecycle.LIVE,
            epoch=entry.epoch + 1,
            epoch_before=entry.epoch,
            gate_move=kb.GateMove.SETTLED,
            epoch_lineage=(entry.epoch_lineage or []) + [entry.epoch + 1],
            created_at=entry.created_at,
        )
    )
    second = start_removal("marker-reset", mode="reversible")
    assert second.success, second.message
    assert second.removal_id != removal_id

    assert _phase_row("marker-reset")["applied_mode_content"] is None
    record = kb.get_removal_phase_record("marker-reset")
    assert kb.applied_mode_content_marker(record)["state"] == "not-applicable"
    assert kb.removal_operator_items(record) == []


# ---------------------------------------------------------------------------
# The one legitimate way to clear it
# ---------------------------------------------------------------------------

def test_done_succeeds_once_the_permanent_mode_content_is_recorded(fence_home):
    removal_id = _swept("perm-cleared", "permanent")
    db_path = kb.kanban_db_path(board="perm-cleared")

    # §7.2's work, for real, recorded through the seam.
    shutil.rmtree(kb.board_dir("perm-cleared"))
    recorded = kb.record_applied_mode_content(
        "perm-cleared",
        removal_id=removal_id,
        evidence=kb.AppliedModeContentEvidence(performed_by="§7.2 driver"),
    )
    assert recorded.success, recorded.message
    assert recorded.outcome is kb.AppliedModeContentOutcome.RECORDED

    done = kb.complete_removal(
        "perm-cleared", removal_id=removal_id, outcome="ended-cleanly"
    )

    assert done.success is True
    assert done.transitioned is True
    assert register_row("perm-cleared")["lifecycle"] == "hard-removed"
    row = _phase_row("perm-cleared")
    assert row["phase"] == "done"
    assert row["outcome"] == "ended-cleanly"
    # The terminal fact the register now records is TRUE.
    assert not db_path.exists()

    marker = json.loads(row["applied_mode_content"])
    assert marker["state"] == kb.APPLIED_MODE_CONTENT_APPLIED
    assert marker["operator_item"] is None
    assert marker["evidence"]["claimed"]["performed_by"] == "§7.2 driver"
    checks = {check["check"] for check in marker["evidence"]["checks"]}
    assert {"board-store-absent", "board-directory-absent"} <= checks
    assert kb.removal_operator_items(
        kb.get_removal_phase_record("perm-cleared")
    ) == []


def test_done_succeeds_once_the_reversible_mode_content_is_recorded(
    fence_home, tmp_path
):
    removal_id = _swept("rev-cleared", "reversible")
    retained = _retained_copy("rev-cleared", tmp_path / "retained-rev-cleared")
    shutil.rmtree(kb.board_dir("rev-cleared"))

    recorded = kb.record_applied_mode_content(
        "rev-cleared",
        removal_id=removal_id,
        evidence=kb.AppliedModeContentEvidence(
            performed_by="§7.3 driver", retained_path=str(retained)
        ),
    )
    assert recorded.success, recorded.message

    done = kb.complete_removal(
        "rev-cleared", removal_id=removal_id, outcome="ended-cleanly"
    )

    assert done.success is True
    assert register_row("rev-cleared")["lifecycle"] == "archived"
    assert _phase_row("rev-cleared")["phase"] == "done"

    # Every recorded check keeps one shape, and the verified retained
    # inventory is part of the evidence.
    checks = _durable_marker("rev-cleared")["evidence"]["checks"]
    assert {check["check"] for check in checks} == {
        "retained-copy-present", "retained-copy-verified", "live-store-absent",
    }
    verified = next(c for c in checks if c["check"] == "retained-copy-verified")
    assert verified["retained_inventory"]["count"] == 1
    assert verified["retained_inventory"]["tasks"][0]["fingerprint"]

    # The retained copy the terminal fact rests on is still readable, and
    # still holds the board's task.
    with read_only(retained / "kanban.db") as conn:
        assert conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0] == 1


def test_the_seam_is_idempotent_on_repeat(fence_home):
    removal_id = _swept("seam-repeat", "permanent")
    shutil.rmtree(kb.board_dir("seam-repeat"))
    evidence = kb.AppliedModeContentEvidence(performed_by="§7.2 driver")
    first = kb.record_applied_mode_content(
        "seam-repeat", removal_id=removal_id, evidence=evidence
    )
    assert first.outcome is kb.AppliedModeContentOutcome.RECORDED
    recorded_at = _durable_marker("seam-repeat")["applied_at"]

    second = kb.record_applied_mode_content(
        "seam-repeat", removal_id=removal_id, evidence=evidence
    )

    assert second.success is True
    assert second.outcome is kb.AppliedModeContentOutcome.IDEMPOTENT_NOOP
    assert _durable_marker("seam-repeat")["applied_at"] == recorded_at


# ---------------------------------------------------------------------------
# The seam verifies the claim; it does not believe it
# ---------------------------------------------------------------------------

def test_the_seam_refuses_while_the_permanent_storage_is_still_there(fence_home):
    removal_id = _swept("seam-not-done", "permanent")
    before = _phase_row("seam-not-done")["applied_mode_content"]

    result = kb.record_applied_mode_content(
        "seam-not-done",
        removal_id=removal_id,
        evidence=kb.AppliedModeContentEvidence(
            performed_by="a caller that says it destroyed the storage"
        ),
    )

    assert result.success is False
    assert result.outcome is kb.AppliedModeContentOutcome.REFUSED_UNVERIFIED
    assert "§7.2 is not done" in result.message
    assert str(kb.kanban_db_path(board="seam-not-done")) in result.message
    # A refused claim changes nothing.
    assert _phase_row("seam-not-done")["applied_mode_content"] == before
    assert kb.complete_removal(
        "seam-not-done", removal_id=removal_id, outcome="probe"
    ).success is False
    assert register_row("seam-not-done")["lifecycle"] == "removing"


def test_the_seam_refuses_a_permanent_claim_with_the_directory_left_behind(
    fence_home
):
    removal_id = _swept("seam-dir-left", "permanent")
    kb.kanban_db_path(board="seam-dir-left").unlink()

    result = kb.record_applied_mode_content(
        "seam-dir-left",
        removal_id=removal_id,
        evidence=kb.AppliedModeContentEvidence(performed_by="§7.2 driver"),
    )

    assert result.success is False
    assert result.outcome is kb.AppliedModeContentOutcome.REFUSED_UNVERIFIED
    assert str(kb.board_dir("seam-dir-left")) in result.message


def test_the_seam_refuses_a_permanent_claim_with_no_carried_inventory(fence_home):
    """§5.4: what was destroyed must have a durable record of what it held."""
    removal_id = _swept("seam-no-inventory", "permanent")
    with kb.register_connect() as conn:
        payload = json.loads(
            conn.execute(
                "SELECT carried_payload FROM board_removal_phase "
                "WHERE board_name = ?", ("seam-no-inventory",),
            ).fetchone()[0]
        )
        payload.pop("pre_application_inventory")
        conn.execute(
            "UPDATE board_removal_phase SET carried_payload = ? "
            "WHERE board_name = ?",
            (json.dumps(payload), "seam-no-inventory"),
        )
        conn.commit()
    shutil.rmtree(kb.board_dir("seam-no-inventory"))

    result = kb.record_applied_mode_content(
        "seam-no-inventory",
        removal_id=removal_id,
        evidence=kb.AppliedModeContentEvidence(performed_by="§7.2 driver"),
    )

    assert result.success is False
    assert result.outcome is kb.AppliedModeContentOutcome.REFUSED_UNVERIFIED
    assert "pre-application inventory" in result.message


def test_the_seam_refuses_a_reversible_claim_that_names_no_retained_copy(fence_home):
    removal_id = _swept("seam-no-copy", "reversible")
    shutil.rmtree(kb.board_dir("seam-no-copy"))

    result = kb.record_applied_mode_content(
        "seam-no-copy",
        removal_id=removal_id,
        evidence=kb.AppliedModeContentEvidence(performed_by="§7.3 driver"),
    )

    assert result.success is False
    assert result.outcome is kb.AppliedModeContentOutcome.REFUSED_UNVERIFIED
    assert "retained copy" in result.message


def test_the_seam_refuses_a_retained_copy_inside_the_board_it_removed(fence_home):
    removal_id = _swept("seam-inside", "reversible")
    inside = kb.board_dir("seam-inside") / "retained-inside"
    inside.mkdir()

    result = kb.record_applied_mode_content(
        "seam-inside",
        removal_id=removal_id,
        evidence=kb.AppliedModeContentEvidence(
            performed_by="§7.3 driver", retained_path=str(inside)
        ),
    )

    assert result.success is False
    assert result.outcome is kb.AppliedModeContentOutcome.REFUSED_UNVERIFIED
    assert "inside the live board directory" in result.message


def test_the_seam_refuses_a_reversible_claim_while_the_live_store_remains(
    fence_home, tmp_path
):
    removal_id = _swept("seam-live-left", "reversible")
    retained = _retained_copy("seam-live-left", tmp_path / "retained-live-left")

    result = kb.record_applied_mode_content(
        "seam-live-left",
        removal_id=removal_id,
        evidence=kb.AppliedModeContentEvidence(
            performed_by="§7.3 driver", retained_path=str(retained)
        ),
    )

    assert result.success is False
    assert result.outcome is kb.AppliedModeContentOutcome.REFUSED_UNVERIFIED
    assert "the live board store is still there" in result.message
    assert str(kb.kanban_db_path(board="seam-live-left")) in result.message


def test_the_seam_refuses_a_retained_copy_it_cannot_read(fence_home, tmp_path):
    removal_id = _swept("seam-unreadable-copy", "reversible")
    retained = tmp_path / "retained-unreadable"
    retained.mkdir()
    (retained / "kanban.db").write_bytes(b"not a database at all" * 20)
    shutil.rmtree(kb.board_dir("seam-unreadable-copy"))

    result = kb.record_applied_mode_content(
        "seam-unreadable-copy",
        removal_id=removal_id,
        evidence=kb.AppliedModeContentEvidence(
            performed_by="§7.3 driver", retained_path=str(retained)
        ),
    )

    assert result.success is False
    assert result.outcome is kb.AppliedModeContentOutcome.REFUSED_UNVERIFIED
    assert "not verified" in result.message
    assert kb.complete_removal(
        "seam-unreadable-copy", removal_id=removal_id, outcome="probe"
    ).success is False


def test_the_seam_refuses_a_bare_flag_from_the_caller(fence_home):
    removal_id = _swept("seam-bare-flag", "permanent")
    shutil.rmtree(kb.board_dir("seam-bare-flag"))

    for bogus in (True, "done", 1, None):
        with pytest.raises(TypeError):
            kb.record_applied_mode_content(
                "seam-bare-flag", removal_id=removal_id, evidence=bogus
            )

    assert _durable_marker("seam-bare-flag")["state"] == (
        kb.APPLIED_MODE_CONTENT_OUTSTANDING
    )


def test_the_seam_refuses_evidence_that_names_nobody(fence_home):
    removal_id = _swept("seam-anonymous", "permanent")
    shutil.rmtree(kb.board_dir("seam-anonymous"))

    result = kb.record_applied_mode_content(
        "seam-anonymous",
        removal_id=removal_id,
        evidence=kb.AppliedModeContentEvidence(performed_by="  "),
    )

    assert result.success is False
    assert result.outcome is kb.AppliedModeContentOutcome.REFUSED_UNVERIFIED
    assert _durable_marker("seam-anonymous")["state"] == (
        kb.APPLIED_MODE_CONTENT_OUTSTANDING
    )


def test_the_seam_refuses_another_runs_removal_id(fence_home):
    removal_id = _swept("seam-other-run", "permanent")
    shutil.rmtree(kb.board_dir("seam-other-run"))

    result = kb.record_applied_mode_content(
        "seam-other-run",
        removal_id="someone-elses-run",
        evidence=kb.AppliedModeContentEvidence(performed_by="§7.2 driver"),
    )

    assert result.success is False
    assert result.outcome is kb.AppliedModeContentOutcome.REFUSED_ID_MISMATCH
    assert _durable_marker("seam-other-run")["state"] == (
        kb.APPLIED_MODE_CONTENT_OUTSTANDING
    )
    assert kb.complete_removal(
        "seam-other-run", removal_id=removal_id, outcome="probe"
    ).success is False


def test_the_seam_refuses_a_board_with_no_removal_at_all(fence_home):
    create_fenced_board("seam-no-removal")

    result = kb.record_applied_mode_content(
        "seam-no-removal",
        removal_id="none",
        evidence=kb.AppliedModeContentEvidence(performed_by="§7.2 driver"),
    )

    assert result.success is False
    assert result.outcome is kb.AppliedModeContentOutcome.REFUSED_NO_RECORD


def test_the_seam_refuses_before_applied_is_reached(fence_home):
    create_fenced_board("seam-too-early")
    removal_id = start_removal("seam-too-early", mode="permanent").removal_id
    assert kb.advance_removal_to_fenced(
        "seam-too-early", removal_id=removal_id
    ).success
    shutil.rmtree(kb.board_dir("seam-too-early"))

    result = kb.record_applied_mode_content(
        "seam-too-early",
        removal_id=removal_id,
        evidence=kb.AppliedModeContentEvidence(performed_by="§7.2 driver"),
    )

    assert result.success is False
    assert result.outcome is kb.AppliedModeContentOutcome.REFUSED_PHASE
    assert _phase_row("seam-too-early")["applied_mode_content"] is None


def test_an_invalid_board_name_is_refused_not_raised(fence_home):
    result = kb.record_applied_mode_content(
        "   ", removal_id="x",
        evidence=kb.AppliedModeContentEvidence(performed_by="§7.2 driver"),
    )
    assert result.success is False
    assert result.outcome is kb.AppliedModeContentOutcome.REFUSED_INVALID_BOARD


# ---------------------------------------------------------------------------
# Fail-closed: nothing else may clear the marker
# ---------------------------------------------------------------------------

def test_the_public_cas_cannot_reach_done_while_the_marker_is_outstanding(fence_home):
    removal_id = _swept("cas-done", "permanent")

    result = kb.advance_removal_phase(
        "cas-done", removal_id=removal_id,
        from_phase=kb.RemovalPhase.SWEPT, to_phase=kb.RemovalPhase.DONE,
    )

    assert result.success is False
    assert _phase_row("cas-done")["phase"] == "swept"
    assert register_row("cas-done")["lifecycle"] == "removing"


def test_the_public_cas_cannot_write_the_marker_itself(fence_home):
    """A sealed field: the marker is never an assertion from a caller."""
    removal_id = _swept("cas-forge", "permanent")
    before = _phase_row("cas-forge")["applied_mode_content"]

    with pytest.raises(ValueError) as excinfo:
        kb.advance_removal_phase(
            "cas-forge", removal_id=removal_id,
            from_phase=kb.RemovalPhase.SWEPT, to_phase=kb.RemovalPhase.DONE,
            applied_mode_content=json.dumps(
                {"state": kb.APPLIED_MODE_CONTENT_APPLIED, "evidence": "trust me"}
            ),
        )

    assert "applied_mode_content" in str(excinfo.value)
    assert "record_applied_mode_content" in str(excinfo.value)
    assert _phase_row("cas-forge")["applied_mode_content"] == before


def test_the_public_cas_cannot_reach_applied_at_all(fence_home):
    """Replaces a test that asserted the generic compare-and-set COULD
    take a record into Applied (and then checked that the missing marker
    read as outstanding). Applied is recordable only by its own driver
    now: the generic entry point refuses and names it, so no route into
    Applied leaves the marker unwritten.
    """
    create_fenced_board("cas-applied")
    removal_id = start_removal("cas-applied", mode="permanent").removal_id
    assert kb.advance_removal_to_fenced("cas-applied", removal_id=removal_id).success
    for driver in (
        kb.advance_removal_to_quiesced,
        kb.advance_removal_to_carried,
        kb.advance_removal_to_released,
    ):
        assert driver("cas-applied", removal_id=removal_id).success

    forced = kb.advance_removal_phase(
        "cas-applied", removal_id=removal_id,
        from_phase=kb.RemovalPhase.RELEASED, to_phase=kb.RemovalPhase.APPLIED,
    )

    assert forced.success is False
    assert forced.transitioned is False
    assert "advance_removal_to_applied" in forced.message
    assert _phase_row("cas-applied")["phase"] == "released"
    assert _phase_row("cas-applied")["applied_mode_content"] is None


def test_a_record_at_applied_whose_marker_vanished_reads_as_outstanding(fence_home):
    """The absence of a record of the work is not evidence that the work
    happened: a marker damaged past the primitive reads as OUTSTANDING and
    Done stays refused."""
    removal_id = _swept("marker-vanished", "permanent")
    with kb.register_connect() as conn:
        conn.execute(
            "UPDATE board_removal_phase SET applied_mode_content = NULL "
            "WHERE board_name = ?", ("marker-vanished",),
        )
        conn.commit()
    assert _phase_row("marker-vanished")["applied_mode_content"] is None

    record = kb.get_removal_phase_record("marker-vanished")
    assert kb.applied_mode_content_is_outstanding(record) is True
    assert kb.removal_operator_items(record)[0]["kind"] == (
        kb.APPLIED_MODE_CONTENT_ITEM_KIND
    )

    done = kb.complete_removal("marker-vanished", removal_id=removal_id, outcome="probe")

    assert done.success is False
    assert done.outcome is kb.RemovalAdvanceOutcome.REFUSED_MODE_CONTENT_OUTSTANDING
    assert register_row("marker-vanished")["lifecycle"] == "removing"


def test_an_applied_marker_with_no_evidence_reads_as_outstanding(fence_home):
    """A hand-written "applied" with nothing verified behind it is refused."""
    removal_id = _swept("forged-marker", "permanent")
    with kb.register_connect() as conn:
        conn.execute(
            "UPDATE board_removal_phase SET applied_mode_content = ? "
            "WHERE board_name = ?",
            (
                json.dumps({
                    "state": kb.APPLIED_MODE_CONTENT_APPLIED,
                    "mode": "permanent",
                    "rule": "§7.2",
                    "evidence": None,
                }),
                "forged-marker",
            ),
        )
        conn.commit()

    record = kb.get_removal_phase_record("forged-marker")
    assert kb.applied_mode_content_is_outstanding(record) is True

    done = kb.complete_removal("forged-marker", removal_id=removal_id, outcome="probe")

    assert done.success is False
    assert done.outcome is kb.RemovalAdvanceOutcome.REFUSED_MODE_CONTENT_OUTSTANDING
    assert register_row("forged-marker")["lifecycle"] == "removing"
    assert kb.kanban_db_path(board="forged-marker").exists()


def test_an_unreadable_marker_reads_as_outstanding(fence_home):
    removal_id = _swept("unreadable-marker", "permanent")
    with kb.register_connect() as conn:
        conn.execute(
            "UPDATE board_removal_phase SET applied_mode_content = ? "
            "WHERE board_name = ?", ("{not json at all", "unreadable-marker"),
        )
        conn.commit()

    assert kb.applied_mode_content_is_outstanding(
        kb.get_removal_phase_record("unreadable-marker")
    ) is True
    assert kb.complete_removal(
        "unreadable-marker", removal_id=removal_id, outcome="probe"
    ).success is False
    assert register_row("unreadable-marker")["lifecycle"] == "removing"


def test_a_register_store_without_the_marker_column_gains_it_on_next_open(
    fence_home
):
    """The additive upgrade: an existing register store has the phase
    table but not this column, and a removal on it must still refuse Done
    rather than fail to record the marker at all."""
    create_fenced_board("upgraded-store")
    path = kb.register_db_path()
    with sqlite3.connect(str(path)) as conn:
        conn.execute(
            "ALTER TABLE board_removal_phase DROP COLUMN applied_mode_content"
        )
    with read_only(path) as ro:
        columns = {
            row[1] for row in ro.execute("PRAGMA table_info(board_removal_phase)")
        }
    assert "applied_mode_content" not in columns

    # A fresh process would not have this path marked initialized either.
    kb._REGISTER_INITIALIZED = False
    kb._REGISTER_INITIALIZED_PATHS.clear()

    removal_id = start_removal("upgraded-store", mode="permanent").removal_id
    assert kb.advance_removal_to_fenced(
        "upgraded-store", removal_id=removal_id
    ).success
    for driver in (
        kb.advance_removal_to_quiesced,
        kb.advance_removal_to_carried,
        kb.advance_removal_to_released,
        kb.advance_removal_to_applied,
        kb.advance_removal_to_swept,
    ):
        assert driver("upgraded-store", removal_id=removal_id).success

    assert _durable_marker("upgraded-store")["state"] == (
        kb.APPLIED_MODE_CONTENT_OUTSTANDING
    )
    assert kb.complete_removal(
        "upgraded-store", removal_id=removal_id, outcome="probe"
    ).success is False
    assert register_row("upgraded-store")["lifecycle"] == "removing"


def test_the_shared_test_helper_drives_the_real_seam(fence_home):
    """The helper the Done tests use performs the work and verifies it."""
    removal_id = _swept("helper-check", "permanent")

    apply_mode_specific_content("helper-check", removal_id)

    marker = _durable_marker("helper-check")
    assert marker["state"] == kb.APPLIED_MODE_CONTENT_APPLIED
    assert marker["evidence"]["checks"]
    assert not kb.kanban_db_path(board="helper-check").exists()
    assert kb.complete_removal(
        "helper-check", removal_id=removal_id, outcome="ended-cleanly"
    ).success is True
    assert register_row("helper-check")["lifecycle"] == "hard-removed"
