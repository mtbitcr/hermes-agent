"""Phases 4, 5 and 7 (design revision 5, §6.4, §6.5, §6.7): the common
work, not labels for it.

§6.4 migrates every outstanding release obligation WITH ITS EXACT RECORDED
IDENTITY, the outside-resource ledger and (permanent mode) the
pre-application inventory, in one atomic write, before anything is
destroyed. §6.5 releases each carried environment BY that exact identity,
and counts an absent answer as released only when the query succeeded and
the listing is proven correctly scoped. §6.7 updates or removes every
record elsewhere naming this board, each found by exact key, with a
recorded result for every class it names.

Each phase records its work durably and only then advances, and each
phase's precondition re-reads that durable fact — so these tests assert
both the work and the refusal when the work has not happened.
"""

from __future__ import annotations

import json
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
    damage_phase_column_behind_the_primitive,
    hold_exclusive_lock,
    read_only,
    ready_task,
    start_removal,
)

SANDBOX_ID = "sbx-9f3c1a77-exact"
SESSION_ID = "sess-3f9a11c4-exact"
BASE_COMMIT = "b" * 40
HEAD_COMMIT = "h" * 40
WORKTREE_TASK_TITLE = "Add login"


def _board_with_obligation(slug: str, *, mode: str = "reversible") -> str:
    """A board carrying one durable obligation to release an environment.

    ``run_sandbox_cleanup_intents`` is the durable record this codebase
    keeps of "an exact run has a machine with no durable release event".
    The row is written before the removal begins, so the carry has to
    derive it from durable board state rather than be handed it.
    """
    create_fenced_board(slug)
    conn = kb.connect(board=slug)
    task_id = ready_task(conn)
    with kb.write_txn(conn):
        conn.execute(
            "INSERT INTO run_sandbox_cleanup_intents ("
            "  task_id, run_id, profile, generation, sandbox_id, "
            "  provision_event_id, attempt_count, next_attempt_at"
            ") VALUES (?, 1, 'worker', 1, ?, 1, 0, 0)",
            (task_id, SANDBOX_ID),
        )
    conn.close()
    return _fence(slug, mode=mode)


def _fence(slug: str, *, mode: str = "reversible") -> str:
    intent = start_removal(slug, mode=mode)
    assert intent.success, intent.message
    fenced = kb.advance_removal_to_fenced(slug, removal_id=intent.removal_id)
    assert fenced.success, fenced.message
    quiesced = kb.advance_removal_to_quiesced(slug, removal_id=intent.removal_id)
    assert quiesced.success, quiesced.message
    return intent.removal_id


def _plain_board(slug: str, *, mode: str = "reversible") -> str:
    create_fenced_board(slug)
    return _fence(slug, mode=mode)


def _project_linked_worktree_task(slug: str, repo: Path):
    """One real project-linked worktree task on *slug*, with git receipts.

    Goes through the shipped ``create_task`` project link — the path that
    anchors a task's work area under a shared repository as
    ``<repo>/.worktrees/<task-id>`` and names its branch after the
    project, the task and the task's title. The two commit receipts are
    parked with a plain UPDATE, exactly as ``ready_task`` parks a status:
    the kernel derives them from a real checkout, which this test has no
    repository for, and what is under test is the ledger's derivation
    from the durable row.
    """
    from hermes_cli import projects_db as pdb

    with pdb.connect_closing() as projects:
        project = pdb.get_project(
            projects,
            pdb.create_project(projects, name="Ledger Repo", folders=[str(repo)]),
        )
    conn = kb.connect(board=slug)
    try:
        task_id = kb.create_task(
            conn,
            title=WORKTREE_TASK_TITLE,
            assignee="worker",
            project_id=project.slug,
            session_id=SESSION_ID,
        )
        with kb.write_txn(conn):
            conn.execute(
                "UPDATE tasks SET base_commit = ?, head_commit = ? WHERE id = ?",
                (BASE_COMMIT, HEAD_COMMIT, task_id),
            )
        task = kb.get_task(conn, task_id)
    finally:
        conn.close()
    assert task.workspace_kind == "worktree", task.workspace_kind
    return project, task


def _released(**kwargs) -> kb.EnvironmentReleaseAttempt:
    return kb.EnvironmentReleaseAttempt(**kwargs)


# ---------------------------------------------------------------------------
# §6.4 — Carried: one atomic write, before anything is destroyed
# ---------------------------------------------------------------------------

def test_carry_migrates_the_obligation_with_its_exact_recorded_identity(fence_home):
    removal_id = _board_with_obligation("carry-identity")

    result = kb.advance_removal_to_carried("carry-identity", removal_id=removal_id)

    assert result.success is True
    assert result.transitioned is True
    record = kb.get_removal_phase_record("carry-identity")
    assert record.phase == kb.RemovalPhase.CARRIED
    assert record.carry_completed_at is not None
    payload = record.carried()
    identities = [o["identity"]["sandbox_id"] for o in payload["release_obligations"]]
    assert identities == [SANDBOX_ID]
    assert payload["outside_resource_ledger"]


def test_the_carried_ledger_names_what_is_retained_rather_than_destroyed(fence_home):
    removal_id = _plain_board("carry-ledger")
    assert kb.advance_removal_to_carried("carry-ledger", removal_id=removal_id).success

    ledger = kb.get_removal_phase_record("carry-ledger").carried()[
        "outside_resource_ledger"
    ]
    by_member = {entry["member"]: entry for entry in ledger}
    assert by_member["ever-existed-marker"]["disposition"] == "out"
    assert by_member["removal-archive"]["disposition"] == "out"
    # Every member carries an exact identity, never a pattern.
    assert all(entry["identity"] for entry in ledger)


def test_the_carried_ledger_names_every_ledgered_scope_category(fence_home):
    """§6.4/§12.5: the ledger is this removal's account of what it holds
    OUTSIDE the board's own storage, so every category the scope table
    ledgers has to be IN it — including the two that are ledgered and
    kept rather than destroyed (C6, C7) and the one that is ledgered and
    deregistered (C13). A category left out is a removal acting without
    knowing what it holds.
    """
    removal_id = _plain_board("carry-scope")
    assert kb.advance_removal_to_carried("carry-scope", removal_id=removal_id).success

    ledger = kb.get_removal_phase_record("carry-scope").carried()[
        "outside_resource_ledger"
    ]
    by_member = {entry["member"]: entry for entry in ledger}
    # Exactly the required set: nothing missing, nothing unaccounted for.
    assert set(by_member) == set(kb._REQUIRED_CARRIED_LEDGER_MEMBERS)
    # The members the ledger already carried are still carried, with the
    # dispositions they already had.
    for member in (
        "board-directory", "workspaces-root", "register-lock", "current-pointer",
    ):
        assert by_member[member]["disposition"] == "in", member
    for member in ("register-entry", "ever-existed-marker", "removal-archive"):
        assert by_member[member]["disposition"] == "out", member

    by_category = {
        entry["category"]: entry for entry in ledger if entry.get("category")
    }
    assert {"C6", "C7", "C13"} <= set(by_category)
    # C6 — conversation transcripts: kept and disclosed, not destroyed.
    assert by_category["C6"]["disposition"] == "out"
    # C7 — the references and commits created for this board's work: kept
    # permanently, and every one of them explicitly named.
    assert by_category["C7"]["disposition"] == "out"
    assert isinstance(by_category["C7"]["references"], list)
    # C13 — the shared container's registration entry for the work area:
    # deregistered by exact recorded identity, refs and commits kept.
    assert by_category["C13"]["disposition"] == "in"
    assert isinstance(by_category["C13"]["registrations"], list)
    # Every member — old and new — still carries an exact identity and a
    # reason, never a pattern.
    assert all(entry["identity"] and entry["reason"] for entry in ledger)


def test_the_ledger_names_each_reference_and_registration_by_exact_identity(
    fence_home, tmp_path
):
    """C7 and C13 are ledgered per THING, with its exact recorded identity.

    C7 must list every retained reference and every commit created for
    this board's work; C13 must name the shared container's own
    registration entry for the work area — by exact identity, because
    that entry is the one that gets deregistered while the reference and
    its commits are kept.
    """
    repo = tmp_path / "shared-repo"
    repo.mkdir()
    create_fenced_board("carry-worktree")
    project, task = _project_linked_worktree_task("carry-worktree", repo)
    removal_id = _fence("carry-worktree")

    assert kb.advance_removal_to_carried(
        "carry-worktree", removal_id=removal_id
    ).success

    ledger = kb.get_removal_phase_record("carry-worktree").carried()[
        "outside_resource_ledger"
    ]
    by_category = {
        entry["category"]: entry for entry in ledger if entry.get("category")
    }

    # C6 — the transcript store: declared, disclosed by location, and
    # outside this board's own storage area. Nothing acts on it.
    transcripts = by_category["C6"]
    assert transcripts["declaration_only"] is True
    assert transcripts["identity"] == str(kb.conversation_transcript_store_path())
    assert not transcripts["identity"].startswith(str(kb.board_dir("carry-worktree")))
    session, = transcripts["originating_sessions"]
    assert session["task"] == task.id
    assert session["session_id"] == SESSION_ID

    # C7 — the reference and the commits, each explicitly named.
    reference, = by_category["C7"]["references"]
    assert by_category["C7"]["count"] == 1
    assert reference["task"] == task.id
    assert reference["title"] == WORKTREE_TASK_TITLE
    assert reference["project"] == project.id
    assert reference["reference"] == task.branch_name
    assert reference["reference"] == f"{project.slug}/{task.id}-add-login"
    assert reference["container"] == str(repo)
    # The base and the head are NAMED, separately — the base is where the
    # work started and this board did not create it.
    assert reference["base_commit"] == BASE_COMMIT
    assert reference["head_commit"] == HEAD_COMMIT
    # The created list is exactly this board's own commits, so the base
    # is NOT on it: carrying [base, head] would claim the base as work
    # this board created.
    assert reference["board_created_commits"] == [HEAD_COMMIT]
    assert reference["commits"] == [HEAD_COMMIT]
    assert BASE_COMMIT not in reference["board_created_commits"]
    assert reference["absorbed_heads"] == []
    assert reference["commit_rule"] == kb.BOARD_CREATED_COMMIT_RULE

    # C13 — the shared container's registration entry for the work area.
    registration, = by_category["C13"]["registrations"]
    assert by_category["C13"]["count"] == 1
    assert registration["task"] == task.id
    assert registration["title"] == WORKTREE_TASK_TITLE
    assert registration["project"] == project.id
    assert registration["work_area"] == task.workspace_path
    assert registration["work_area"] == str(repo / ".worktrees" / task.id)
    # The registration entry, not the container: the container is never
    # deleted, moved or rewritten.
    assert registration["container"] == str(repo)
    assert registration["registration"] == str(
        repo / ".git" / "worktrees" / task.id
    )
    # The reference stays even though this registration goes (C7).
    assert registration["reference"] == task.branch_name


def test_a_work_area_the_ledger_cannot_place_is_named_not_dropped(
    fence_home, tmp_path
):
    """IN-3: a work area whose shared container cannot be derived from
    durable state is ledgered as UNRESOLVED, with its reason.

    C13 is destroyed by exact recorded identity. A work area this board
    recorded creating, whose container the record does not establish,
    therefore has no identity to act on — and must be surfaced as such,
    never dropped from the ledger so that the removal looks as if it
    held nothing there.
    """
    elsewhere = tmp_path / "loose-checkout"
    elsewhere.mkdir()
    create_fenced_board("carry-unplaceable")
    conn = kb.connect(board="carry-unplaceable")
    task_id = kb.create_task(
        conn, title="loose work area", assignee="worker",
        workspace_kind="worktree", workspace_path=str(elsewhere),
    )
    conn.close()
    removal_id = _fence("carry-unplaceable")

    assert kb.advance_removal_to_carried(
        "carry-unplaceable", removal_id=removal_id
    ).success

    ledger = kb.get_removal_phase_record("carry-unplaceable").carried()[
        "outside_resource_ledger"
    ]
    registrations = {
        entry["category"]: entry for entry in ledger if entry.get("category")
    }["C13"]
    assert registrations["registrations"] == []
    assert registrations["count"] == 0
    unresolved, = registrations["unresolved"]
    assert unresolved["task"] == task_id
    assert unresolved["work_area"] == str(elsewhere)
    assert unresolved["reason"]


def test_a_board_with_no_work_area_elsewhere_still_ledgers_the_categories(fence_home):
    """IN-3: an empty category is ledgered with its count, never omitted."""
    removal_id = _plain_board("carry-no-worktree")
    assert kb.advance_removal_to_carried(
        "carry-no-worktree", removal_id=removal_id
    ).success

    ledger = kb.get_removal_phase_record("carry-no-worktree").carried()[
        "outside_resource_ledger"
    ]
    by_category = {
        entry["category"]: entry for entry in ledger if entry.get("category")
    }
    assert by_category["C7"]["references"] == []
    assert by_category["C7"]["count"] == 0
    assert by_category["C13"]["registrations"] == []
    assert by_category["C13"]["count"] == 0
    # And each says what it read, so an empty answer is a read answer.
    for category in ("C6", "C7", "C13"):
        assert "tasks" in by_category[category]["read_from"]


def test_a_ledger_with_any_required_member_left_out_is_refused(fence_home):
    """Carried's precondition requires the EXACT required member set.

    Driven through the phase-owned channel a DRIVER uses (no public
    caller can write ``carried_payload`` at all), dropping each required
    member in turn: a payload that leaves one out is refused rather than
    recorded, and nothing about the phase moves.
    """
    removal_id = _plain_board("carry-missing-member")
    record = kb.get_removal_phase_record("carry-missing-member")
    payload, _reason = kb.build_removal_carry_payload("carry-missing-member", record)
    assert payload is not None
    # The three ledgered scope categories are part of the required set, so
    # the loop below covers them too.
    assert {
        "conversation-transcripts", "version-control-references",
        "work-area-registration",
    } <= set(kb._REQUIRED_CARRIED_LEDGER_MEMBERS)

    for dropped in sorted(kb._REQUIRED_CARRIED_LEDGER_MEMBERS):
        incomplete = dict(payload)
        incomplete["outside_resource_ledger"] = [
            member
            for member in payload["outside_resource_ledger"]
            if member["member"] != dropped
        ]
        assert len(incomplete["outside_resource_ledger"]) == len(
            payload["outside_resource_ledger"]
        ) - 1, dropped

        forced = kb._advance_removal_phase(
            "carry-missing-member", removal_id=removal_id,
            from_phase=kb.RemovalPhase.QUIESCED, to_phase=kb.RemovalPhase.CARRIED,
            _phase_owned={
                "carried_payload": json.dumps(incomplete, ensure_ascii=False),
                "carry_completed_at": int(time.time()),
            },
        )

        assert forced.success is False, dropped
        assert forced.outcome is kb.RemovalAdvanceOutcome.REFUSED_PRECONDITION, dropped
        assert dropped in forced.message, dropped
        after = kb.get_removal_phase_record("carry-missing-member")
        assert after.phase == kb.RemovalPhase.QUIESCED, dropped
        assert after.carried_payload is None, dropped
        assert after.carry_completed_at is None, dropped


def test_permanent_mode_carries_a_pre_application_inventory_with_fingerprints(fence_home):
    removal_id = _plain_board("carry-inventory", mode="permanent")
    conn = kb.connect(board="carry-inventory")
    conn.close()

    assert kb.advance_removal_to_carried(
        "carry-inventory", removal_id=removal_id
    ).success

    inventory = kb.get_removal_phase_record("carry-inventory").carried()[
        "pre_application_inventory"
    ]
    assert inventory is not None
    assert inventory["count"] == len(inventory["tasks"])
    assert all(task["fingerprint"] for task in inventory["tasks"])


def test_reversible_mode_carries_no_pre_application_inventory(fence_home):
    removal_id = _plain_board("carry-no-inventory", mode="reversible")
    assert kb.advance_removal_to_carried(
        "carry-no-inventory", removal_id=removal_id
    ).success
    assert kb.get_removal_phase_record("carry-no-inventory").carried()[
        "pre_application_inventory"
    ] is None


def test_carry_refuses_when_the_board_state_cannot_be_read(fence_home):
    removal_id = _plain_board("carry-unreadable")
    kb.kanban_db_path(board="carry-unreadable").write_bytes(b"not a database")

    result = kb.advance_removal_to_carried("carry-unreadable", removal_id=removal_id)

    assert result.success is False
    assert result.outcome is kb.RemovalAdvanceOutcome.REFUSED_INDETERMINATE
    assert kb.get_removal_phase_record(
        "carry-unreadable"
    ).phase == kb.RemovalPhase.QUIESCED


def test_carried_cannot_be_recorded_without_the_payload(fence_home):
    removal_id = _plain_board("carry-empty")

    result = kb.advance_removal_phase(
        "carry-empty", removal_id=removal_id,
        from_phase=kb.RemovalPhase.QUIESCED, to_phase=kb.RemovalPhase.CARRIED,
    )

    assert result.success is False
    assert result.outcome is kb.RemovalAdvanceOutcome.REFUSED_PRECONDITION
    assert "advance_removal_to_carried" in result.message
    assert kb.get_removal_phase_record("carry-empty").phase == kb.RemovalPhase.QUIESCED

    # And the same transition through the raw private primitive, which
    # derives Carried's precondition itself: still refused, because the
    # payload the phase's meaning consists of is not there.
    raw = kb._advance_removal_phase(
        "carry-empty", removal_id=removal_id,
        from_phase=kb.RemovalPhase.QUIESCED, to_phase=kb.RemovalPhase.CARRIED,
    )
    assert raw.success is False
    assert raw.outcome is kb.RemovalAdvanceOutcome.REFUSED_PRECONDITION
    assert kb.get_removal_phase_record("carry-empty").phase == kb.RemovalPhase.QUIESCED


def test_a_carry_repeat_refuses_when_the_payload_is_undecodable(fence_home):
    """An idempotent repeat re-verifies the payload it finds recorded.

    A no-op success at Carried is a claim that the migration happened; a
    column that cannot be decoded cannot support that claim.
    """
    removal_id = _board_with_obligation("carry-repeat-undecodable")
    assert kb.advance_removal_to_carried(
        "carry-repeat-undecodable", removal_id=removal_id
    ).success
    damage_phase_column_behind_the_primitive(
        "carry-repeat-undecodable", "carried_payload", "{not json at all",
    )

    repeat = kb.advance_removal_to_carried(
        "carry-repeat-undecodable", removal_id=removal_id
    )

    assert repeat.success is False
    assert repeat.outcome is kb.RemovalAdvanceOutcome.REFUSED_WORK_INCOMPLETE
    assert repeat.transitioned is False
    assert kb.get_removal_phase_record(
        "carry-repeat-undecodable"
    ).phase == kb.RemovalPhase.CARRIED


def test_a_carry_repeat_refuses_when_its_completion_fact_is_gone(fence_home):
    removal_id = _board_with_obligation("carry-repeat-factless")
    assert kb.advance_removal_to_carried(
        "carry-repeat-factless", removal_id=removal_id
    ).success
    damage_phase_column_behind_the_primitive(
        "carry-repeat-factless", "carry_completed_at", None
    )

    repeat = kb.advance_removal_to_carried(
        "carry-repeat-factless", removal_id=removal_id
    )

    assert repeat.success is False
    assert repeat.outcome is kb.RemovalAdvanceOutcome.REFUSED_WORK_INCOMPLETE


def test_a_carry_repeat_with_the_payload_intact_is_still_a_no_op(fence_home):
    removal_id = _board_with_obligation("carry-repeat-ok")
    assert kb.advance_removal_to_carried(
        "carry-repeat-ok", removal_id=removal_id
    ).success
    before = kb.get_removal_phase_record("carry-repeat-ok")

    repeat = kb.advance_removal_to_carried("carry-repeat-ok", removal_id=removal_id)

    assert repeat.success is True
    assert repeat.outcome is kb.RemovalAdvanceOutcome.IDEMPOTENT_NOOP
    assert repeat.transitioned is False
    after = kb.get_removal_phase_record("carry-repeat-ok")
    assert after.carried_payload == before.carried_payload
    assert after.carry_completed_at == before.carry_completed_at


# ---------------------------------------------------------------------------
# §6.5 — Released: exact identity, and the qualified already-absent rule
# ---------------------------------------------------------------------------

def test_release_attempts_the_exact_recorded_identity_never_a_pattern(fence_home):
    removal_id = _board_with_obligation("release-exact")
    assert kb.advance_removal_to_carried("release-exact", removal_id=removal_id).success
    seen = []

    def releaser(environment):
        seen.append(environment["identity"]["sandbox_id"])
        return _released(released=True, detail="deleted by id")

    result = kb.advance_removal_to_released(
        "release-exact", removal_id=removal_id, releaser=releaser
    )

    assert result.success is True
    assert seen == [SANDBOX_ID]
    record = kb.get_removal_phase_record("release-exact")
    assert record.release_completed_at is not None
    env = record.releases()["environments"][0]
    assert env["state"] == kb.EnvironmentReleaseState.RELEASED.value
    assert env["intent_recorded_at"] is not None


def test_the_releasing_intent_is_durable_before_the_external_side_is_touched(
    fence_home,
):
    """§6.5's order: durably transition this environment's recorded state
    from ``owed`` to ``releasing`` by a conditional compare-and-set, and
    ONLY THEN attempt the external release by exact recorded identity."""
    removal_id = _board_with_obligation("release-intent-first")
    assert kb.advance_removal_to_carried(
        "release-intent-first", removal_id=removal_id
    ).success
    seen = []

    def releaser(environment):
        # What DURABLE state says at the instant the external side is
        # contacted, read through a plain read-only connection.
        with read_only(kb.register_db_path()) as conn:
            raw = conn.execute(
                "SELECT release_state FROM board_removal_phase "
                "WHERE board_name = ?", ("release-intent-first",),
            ).fetchone()["release_state"]
        seen.append(json.loads(raw) if raw else None)
        return _released(released=True, detail="deleted by id")

    result = kb.advance_removal_to_released(
        "release-intent-first", removal_id=removal_id, releaser=releaser
    )

    assert result.success is True
    assert len(seen) == 1
    durable = seen[0]
    assert durable is not None, "the external side was touched with no durable intent"
    env = durable["environments"][0]
    assert env["identity"]["sandbox_id"] == SANDBOX_ID
    assert env["state"] == kb.EnvironmentReleaseState.RELEASING.value
    assert env["intent_recorded_at"] is not None
    assert env["attempts"] == 1


def test_a_crash_mid_release_leaves_the_attempt_durably_recorded(fence_home):
    """A crash between contacting the external side and recording the
    result must not leave "nothing was ever attempted" on the record."""
    removal_id = _board_with_obligation("release-crash")
    assert kb.advance_removal_to_carried(
        "release-crash", removal_id=removal_id
    ).success

    class Crash(BaseException):
        """Not an Exception: nothing in the release path may swallow it."""

    def releaser(environment):
        raise Crash()

    with pytest.raises(Crash):
        kb.advance_removal_to_released(
            "release-crash", removal_id=removal_id, releaser=releaser
        )

    record = kb.get_removal_phase_record("release-crash")
    assert record.phase == kb.RemovalPhase.CARRIED
    state = record.releases()
    assert state is not None, "the crash left no record that anything was attempted"
    env = state["environments"][0]
    assert env["identity"]["sandbox_id"] == SANDBOX_ID
    assert env["state"] == kb.EnvironmentReleaseState.RELEASING.value
    assert env["intent_recorded_at"] is not None


def test_only_one_side_wins_the_owed_to_releasing_transition(fence_home):
    """Exactly one side may win ``owed -> releasing``; nothing is released
    twice, and the loser does not act."""
    removal_id = _board_with_obligation("release-one-winner")
    assert kb.advance_removal_to_carried(
        "release-one-winner", removal_id=removal_id
    ).success
    at = 4_000_000
    calls = []
    inner_results = []

    def releaser(environment):
        calls.append(environment["identity"]["sandbox_id"])
        if len(calls) == 1:
            # A second caller arrives while this attempt is in flight.
            # Durable state already records the releasing intent, so it
            # must not contact the external side for the same identity.
            inner_results.append(
                kb.advance_removal_to_released(
                    "release-one-winner", removal_id=removal_id, now=at,
                    releaser=releaser,
                )
            )
        return _released(released=True, detail="deleted by id")

    result = kb.advance_removal_to_released(
        "release-one-winner", removal_id=removal_id, now=at, releaser=releaser
    )

    assert calls == [SANDBOX_ID], "the same identity was released twice"
    assert inner_results and inner_results[0].success is False
    assert result.success is True
    env = kb.get_removal_phase_record("release-one-winner").releases()[
        "environments"
    ][0]
    assert env["state"] == kb.EnvironmentReleaseState.RELEASED.value
    assert env["attempts"] == 1


def test_an_absent_answer_from_a_proven_scope_counts_as_released(fence_home):
    """AB-1: absent counts only when the query succeeded AND the listing
    is proven correctly scoped."""
    removal_id = _board_with_obligation("release-absent-proven")
    assert kb.advance_removal_to_carried(
        "release-absent-proven", removal_id=removal_id
    ).success

    result = kb.advance_removal_to_released(
        "release-absent-proven", removal_id=removal_id,
        releaser=lambda env: _released(absent=True, scope_proven=True),
    )

    assert result.success is True
    env = kb.get_removal_phase_record("release-absent-proven").releases()[
        "environments"
    ][0]
    assert env["state"] == kb.EnvironmentReleaseState.RELEASED.value


def test_an_absent_answer_without_a_proven_scope_is_indeterminate(fence_home):
    """AB-2: any other absent answer is indeterminate, not released."""
    removal_id = _board_with_obligation("release-absent-unproven")
    assert kb.advance_removal_to_carried(
        "release-absent-unproven", removal_id=removal_id
    ).success

    result = kb.advance_removal_to_released(
        "release-absent-unproven", removal_id=removal_id,
        releaser=lambda env: _released(absent=True, scope_proven=False),
    )

    assert result.success is False
    assert result.outcome is kb.RemovalAdvanceOutcome.REFUSED_WORK_INCOMPLETE
    record = kb.get_removal_phase_record("release-absent-unproven")
    assert record.phase == kb.RemovalPhase.CARRIED
    env = record.releases()["environments"][0]
    assert env["state"] == kb.EnvironmentReleaseState.RELEASING.value
    assert "not proven" in env["last_error"]


def test_the_default_releaser_never_reports_an_unproven_absence_as_released(fence_home):
    removal_id = _board_with_obligation("release-default")
    assert kb.advance_removal_to_carried(
        "release-default", removal_id=removal_id
    ).success

    result = kb.advance_removal_to_released("release-default", removal_id=removal_id)

    assert result.success is False
    env = kb.get_removal_phase_record("release-default").releases()["environments"][0]
    assert env["state"] != kb.EnvironmentReleaseState.RELEASED.value


def test_attempts_are_durable_across_calls_a_budget_that_resets_is_no_budget(fence_home):
    removal_id = _board_with_obligation("release-budget-durable")
    assert kb.advance_removal_to_carried(
        "release-budget-durable", removal_id=removal_id
    ).success
    start = 1_000_000

    kb.advance_removal_to_released(
        "release-budget-durable", removal_id=removal_id, now=start
    )
    first = kb.get_removal_phase_record("release-budget-durable").releases()[
        "environments"
    ][0]
    assert first["attempts"] == 1
    assert first["next_attempt_at"] == start + kb.RELEASE_FIRST_WAIT_SECONDS

    # Not yet due: the call attempts nothing and the count does not move.
    kb.advance_removal_to_released(
        "release-budget-durable", removal_id=removal_id, now=start + 1
    )
    assert kb.get_removal_phase_record("release-budget-durable").releases()[
        "environments"
    ][0]["attempts"] == 1

    kb.advance_removal_to_released(
        "release-budget-durable", removal_id=removal_id,
        now=start + kb.RELEASE_FIRST_WAIT_SECONDS,
    )
    assert kb.get_removal_phase_record("release-budget-durable").releases()[
        "environments"
    ][0]["attempts"] == 2


def test_the_attempt_budget_exhausts_into_a_recorded_operator_item(fence_home):
    """§6.5: on exhaustion the obligation becomes an explicit operator item
    naming the exact identity and the failure; the removal CONTINUES and
    the item persists (IN-3)."""
    removal_id = _board_with_obligation("release-exhausts")
    assert kb.advance_removal_to_carried(
        "release-exhausts", removal_id=removal_id
    ).success

    now = 2_000_000
    result = None
    for _ in range(kb.RELEASE_MAX_ATTEMPTS):
        result = kb.advance_removal_to_released(
            "release-exhausts", removal_id=removal_id, now=now
        )
        now += kb.RELEASE_MAX_WAIT_SECONDS

    assert result.success is True
    record = kb.get_removal_phase_record("release-exhausts")
    assert record.phase == kb.RemovalPhase.RELEASED
    items = kb.removal_operator_items(record)
    assert len(items) == 1
    assert items[0]["identity"]["sandbox_id"] == SANDBOX_ID
    assert items[0]["attempts"] == kb.RELEASE_MAX_ATTEMPTS
    assert SANDBOX_ID in items[0]["detail"]


def test_the_total_elapsed_cap_binds_before_the_attempt_cap(fence_home):
    removal_id = _board_with_obligation("release-elapsed-cap")
    assert kb.advance_removal_to_carried(
        "release-elapsed-cap", removal_id=removal_id
    ).success

    now = 3_000_000
    for _ in range(kb.RELEASE_MAX_ATTEMPTS):
        result = kb.advance_removal_to_released(
            "release-elapsed-cap", removal_id=removal_id, now=now
        )
        if result.success:
            break
        now += 20_000

    items = kb.removal_operator_items(kb.get_removal_phase_record("release-elapsed-cap"))
    assert len(items) == 1
    assert items[0]["attempts"] < kb.RELEASE_MAX_ATTEMPTS


def test_the_release_waits_double_from_30_seconds_to_a_one_hour_cap(fence_home):
    waits = [kb.release_attempt_wait_seconds(n) for n in range(1, 12)]
    assert waits[0] == kb.RELEASE_FIRST_WAIT_SECONDS == 30
    assert waits[:6] == [30, 60, 120, 240, 480, 960]
    assert max(waits) == kb.RELEASE_MAX_WAIT_SECONDS == 60 * 60
    assert kb.RELEASE_MAX_ATTEMPTS == 8
    assert kb.RELEASE_TOTAL_ELAPSED_CAP_SECONDS == 24 * 60 * 60


def test_a_board_with_no_obligations_releases_immediately(fence_home):
    removal_id = _plain_board("release-nothing")
    assert kb.advance_removal_to_carried("release-nothing", removal_id=removal_id).success

    result = kb.advance_removal_to_released("release-nothing", removal_id=removal_id)

    assert result.success is True
    record = kb.get_removal_phase_record("release-nothing")
    assert record.phase == kb.RemovalPhase.RELEASED
    assert record.releases()["environments"] == []


def test_released_cannot_be_recorded_over_an_unresolved_environment(fence_home):
    """Replaces the same test driven through the PUBLIC entry point with a
    caller-supplied ``release_completed_at``: that field is the durable
    proof Release's own precondition reads back, so no caller may pass it.

    The refusal it used to assert is kept, at the registry's guard, through
    the phase-owned channel a driver uses — the environment really is still
    unresolved, and no completion instant makes it resolved.
    """
    removal_id = _board_with_obligation("release-forced")
    assert kb.advance_removal_to_carried("release-forced", removal_id=removal_id).success
    kb.advance_removal_to_released("release-forced", removal_id=removal_id)

    with pytest.raises(ValueError) as excinfo:
        kb.advance_removal_phase(
            "release-forced", removal_id=removal_id,
            from_phase=kb.RemovalPhase.CARRIED, to_phase=kb.RemovalPhase.RELEASED,
            release_completed_at=int(time.time()),
        )
    assert "release_completed_at" in str(excinfo.value)
    assert "advance_removal_to_released" in str(excinfo.value)

    unresolved = kb.get_removal_phase_record("release-forced").release_state
    forced = kb._advance_removal_phase(
        "release-forced", removal_id=removal_id,
        from_phase=kb.RemovalPhase.CARRIED, to_phase=kb.RemovalPhase.RELEASED,
        _phase_owned={
            "release_state": unresolved, "release_completed_at": int(time.time()),
        },
    )

    assert forced.success is False
    assert forced.outcome is kb.RemovalAdvanceOutcome.REFUSED_WORK_INCOMPLETE
    assert kb.get_removal_phase_record("release-forced").phase == kb.RemovalPhase.CARRIED


def test_carried_cannot_be_reached_at_all_without_the_payload(fence_home):
    """Replaces a test that asserted the RAW primitive could record
    ``carried`` with nothing carried, and then checked that Release
    noticed. That positive bypass was the defect: the raw private
    compare-and-set now derives Carried's registered precondition itself,
    so the empty carry never becomes a recorded phase in the first place.
    """
    removal_id = _plain_board("release-uncarried")

    forced = kb._advance_removal_phase(
        "release-uncarried", removal_id=removal_id,
        from_phase=kb.RemovalPhase.QUIESCED, to_phase=kb.RemovalPhase.CARRIED,
    )

    assert forced.success is False
    assert forced.outcome is kb.RemovalAdvanceOutcome.REFUSED_PRECONDITION
    assert forced.transitioned is False
    record = kb.get_removal_phase_record("release-uncarried")
    assert record.phase == kb.RemovalPhase.QUIESCED
    assert record.carried_payload is None
    assert record.carry_completed_at is None

    # And Release, asked from a record that never reached Carried, refuses
    # rather than acting on an identity nobody recorded (IN-2).
    result = kb.advance_removal_to_released("release-uncarried", removal_id=removal_id)
    assert result.success is False
    assert result.outcome is kb.RemovalAdvanceOutcome.REFUSED_WORK_INCOMPLETE
    assert kb.get_removal_phase_record(
        "release-uncarried"
    ).phase == kb.RemovalPhase.QUIESCED


def test_release_refuses_when_the_carried_payload_is_no_longer_there(fence_home):
    """Authority is never destroyed before the resource it governs is
    released: an identity that is not recorded cannot be released.

    Carried is reached by its real driver, and the payload is then damaged
    past the primitive — the crashed half-write / operator repair the
    guard exists for.
    """
    removal_id = _board_with_obligation("release-payload-lost")
    assert kb.advance_removal_to_carried(
        "release-payload-lost", removal_id=removal_id
    ).success
    damage_phase_column_behind_the_primitive(
        "release-payload-lost", "carried_payload", None
    )

    result = kb.advance_removal_to_released(
        "release-payload-lost", removal_id=removal_id
    )

    assert result.success is False
    assert result.outcome is kb.RemovalAdvanceOutcome.REFUSED_WORK_INCOMPLETE
    assert kb.get_removal_phase_record(
        "release-payload-lost"
    ).phase == kb.RemovalPhase.CARRIED


def test_a_release_repeat_refuses_when_an_environment_is_no_longer_resolved(
    fence_home,
):
    """A no-op success at Released is a claim that every carried identity
    reached a terminal state. A record that no longer says so is refused,
    not rubber-stamped."""
    removal_id = _board_with_obligation("release-repeat-unresolved")
    assert kb.advance_removal_to_carried(
        "release-repeat-unresolved", removal_id=removal_id
    ).success
    assert kb.advance_removal_to_released(
        "release-repeat-unresolved", removal_id=removal_id,
        releaser=lambda env: _released(released=True, detail="deleted by id"),
    ).success
    record = kb.get_removal_phase_record("release-repeat-unresolved")
    reverted = record.releases()
    reverted["environments"][0]["state"] = kb.EnvironmentReleaseState.OWED.value
    damage_phase_column_behind_the_primitive(
        "release-repeat-unresolved", "release_state", json.dumps(reverted),
    )

    repeat = kb.advance_removal_to_released(
        "release-repeat-unresolved", removal_id=removal_id
    )

    assert repeat.success is False
    assert repeat.outcome is kb.RemovalAdvanceOutcome.REFUSED_WORK_INCOMPLETE
    assert repeat.transitioned is False
    assert kb.get_removal_phase_record(
        "release-repeat-unresolved"
    ).phase == kb.RemovalPhase.RELEASED


def test_a_release_repeat_refuses_when_a_carried_identity_is_unaccounted_for(
    fence_home,
):
    """Every identity §6.4 carried must appear in the release state: a
    resolved-looking record that simply omits one is not a release."""
    removal_id = _board_with_obligation("release-repeat-missing")
    assert kb.advance_removal_to_carried(
        "release-repeat-missing", removal_id=removal_id
    ).success
    assert kb.advance_removal_to_released(
        "release-repeat-missing", removal_id=removal_id,
        releaser=lambda env: _released(released=True, detail="deleted by id"),
    ).success
    emptied = kb.get_removal_phase_record("release-repeat-missing").releases()
    emptied["environments"] = []
    damage_phase_column_behind_the_primitive(
        "release-repeat-missing", "release_state", json.dumps(emptied),
    )

    repeat = kb.advance_removal_to_released(
        "release-repeat-missing", removal_id=removal_id
    )

    assert repeat.success is False
    assert repeat.outcome is kb.RemovalAdvanceOutcome.REFUSED_WORK_INCOMPLETE


def test_a_release_repeat_with_every_identity_resolved_is_still_a_no_op(fence_home):
    removal_id = _board_with_obligation("release-repeat-ok")
    assert kb.advance_removal_to_carried(
        "release-repeat-ok", removal_id=removal_id
    ).success
    assert kb.advance_removal_to_released(
        "release-repeat-ok", removal_id=removal_id,
        releaser=lambda env: _released(released=True, detail="deleted by id"),
    ).success

    repeat = kb.advance_removal_to_released(
        "release-repeat-ok", removal_id=removal_id
    )

    assert repeat.success is True
    assert repeat.outcome is kb.RemovalAdvanceOutcome.IDEMPOTENT_NOOP
    assert repeat.transitioned is False


# ---------------------------------------------------------------------------
# §6.7 — Swept: every record elsewhere, found by exact key
# ---------------------------------------------------------------------------

def _to_applied(slug: str, removal_id: str) -> None:
    assert kb.advance_removal_to_carried(slug, removal_id=removal_id).success
    assert kb.advance_removal_to_released(slug, removal_id=removal_id).success
    assert kb.advance_removal_to_applied(slug, removal_id=removal_id).success


def test_every_named_record_class_gets_a_recorded_result(fence_home):
    removal_id = _plain_board("sweep-classes")
    _to_applied("sweep-classes", removal_id)

    assert kb.advance_removal_to_swept("sweep-classes", removal_id=removal_id).success

    record = kb.get_removal_phase_record("sweep-classes")
    assert record.sweep_completed_at is not None
    classes = record.sweep()["classes"]
    assert set(classes) == {key for key, _ in kb.SWEEP_RECORD_CLASSES}
    for key, entry in classes.items():
        assert entry["result"], key
        assert entry["reason"], key
        assert entry["key"], key


def test_the_current_pointer_naming_this_board_is_cleared_by_exact_key(fence_home):
    removal_id = _plain_board("sweep-pointer")
    kb.set_current_board("sweep-pointer")
    _to_applied("sweep-pointer", removal_id)

    assert kb.advance_removal_to_swept("sweep-pointer", removal_id=removal_id).success

    entry = kb.get_removal_phase_record("sweep-pointer").sweep()["classes"][
        "current-pointer"
    ]
    assert entry["result"] == kb.SWEEP_RESULT_REMOVED
    assert not kb.current_board_path().exists()


def test_a_pointer_naming_another_board_is_never_touched(fence_home):
    """IN-5: no member the board did not create is ever touched."""
    create_fenced_board("someone-else")
    kb.set_current_board("someone-else")
    removal_id = _plain_board("sweep-other-pointer")
    _to_applied("sweep-other-pointer", removal_id)

    assert kb.advance_removal_to_swept(
        "sweep-other-pointer", removal_id=removal_id
    ).success

    entry = kb.get_removal_phase_record("sweep-other-pointer").sweep()["classes"][
        "current-pointer"
    ]
    assert entry["result"] == kb.SWEEP_RESULT_NOT_PRESENT
    assert kb.current_board_path().read_text(encoding="utf-8").strip() == "someone-else"


def test_reversible_mode_copies_into_the_retained_set_before_removing(fence_home):
    removal_id = _plain_board("sweep-retained", mode="reversible")
    kb.set_current_board("sweep-retained")
    _to_applied("sweep-retained", removal_id)

    assert kb.advance_removal_to_swept("sweep-retained", removal_id=removal_id).success

    state = kb.get_removal_phase_record("sweep-retained").sweep()
    assert state["retained"]["current-pointer"] == {"current_board": "sweep-retained"}


def test_permanent_mode_retains_no_copy_of_the_swept_record(fence_home):
    removal_id = _plain_board("sweep-permanent", mode="permanent")
    kb.set_current_board("sweep-permanent")
    _to_applied("sweep-permanent", removal_id)

    assert kb.advance_removal_to_swept("sweep-permanent", removal_id=removal_id).success

    state = kb.get_removal_phase_record("sweep-permanent").sweep()
    assert state["retained"] == {}
    assert state["classes"]["current-pointer"]["result"] == kb.SWEEP_RESULT_REMOVED


def test_a_lock_held_outside_the_board_on_its_behalf_is_recorded_not_deleted(fence_home):
    removal_id = _plain_board("sweep-lock")
    _to_applied("sweep-lock", removal_id)

    assert kb.advance_removal_to_swept("sweep-lock", removal_id=removal_id).success

    entry = kb.get_removal_phase_record("sweep-lock").sweep()["classes"][
        "external-locks-and-leases"
    ]
    assert entry["result"] == kb.SWEEP_RESULT_RETAINED
    assert entry["key"] == str(kb.register_lock_path("sweep-lock"))
    assert "HELD BY THIS REMOVAL" in entry["reason"]


def test_a_class_this_codebase_has_no_store_for_is_recorded_with_its_reason(fence_home):
    """IN-3: silence is a defect, not a courtesy."""
    removal_id = _plain_board("sweep-absent-class")
    _to_applied("sweep-absent-class", removal_id)
    assert kb.advance_removal_to_swept(
        "sweep-absent-class", removal_id=removal_id
    ).success

    classes = kb.get_removal_phase_record("sweep-absent-class").sweep()["classes"]
    subs = classes["subscriptions"]
    assert subs["result"] == kb.SWEEP_RESULT_NOT_PRESENT
    assert "OWN store" in subs["reason"]


def test_the_sweep_refuses_indeterminate_when_the_pointer_cannot_be_read(fence_home):
    """A failed READ is INDETERMINATE, never a proof of absence.

    Replaces an earlier test that asserted REFUSED_WORK_INCOMPLETE and
    SWEEP_RESULT_ERROR for an unreadable pointer. The corrected behaviour:
    REFUSED_INDETERMINATE, the class recorded as INDETERMINATE, the phase
    still ``applied``, and both the sweep state AND the refusal durably
    recorded.
    """
    removal_id = _plain_board("sweep-indeterminate-pointer")
    _to_applied("sweep-indeterminate-pointer", removal_id)
    # A pointer path that cannot be read at all (it's a directory).
    pointer = kb.current_board_path()
    pointer.parent.mkdir(parents=True, exist_ok=True)
    pointer.mkdir()

    result = kb.advance_removal_to_swept(
        "sweep-indeterminate-pointer", removal_id=removal_id
    )

    assert result.success is False
    assert result.outcome is kb.RemovalAdvanceOutcome.REFUSED_INDETERMINATE
    record = kb.get_removal_phase_record("sweep-indeterminate-pointer")
    assert record.phase == kb.RemovalPhase.APPLIED
    # The partial sweep state is durably recorded.
    sweep_state = record.sweep()
    assert sweep_state is not None
    assert sweep_state["classes"]["current-pointer"]["result"] == (
        kb.SWEEP_RESULT_INDETERMINATE
    )
    assert "cannot read" in sweep_state["classes"]["current-pointer"]["reason"]
    # The refusal is also durably recorded.
    assert record.refusal_outcome is not None


def test_the_sweep_refuses_indeterminate_when_board_store_is_corrupt(fence_home):
    """A corrupt board store (non-database bytes) is INDETERMINATE.

    The board's own database replaced by non-database bytes — the open
    succeeds but the query fails (SQLite accepts any file on open but
    fails when asked to read its structure).  The result is INDETERMINATE
    with the affected classes recorded as such and a reason, the refusal
    durable, and NO class claiming a count it could not read.
    """
    removal_id = _plain_board("sweep-corrupt-store")
    _to_applied("sweep-corrupt-store", removal_id)
    # Replace the board's store with garbage.
    kb.kanban_db_path(board="sweep-corrupt-store").write_bytes(b"not a database")

    result = kb.advance_removal_to_swept("sweep-corrupt-store", removal_id=removal_id)

    assert result.success is False
    assert result.outcome is kb.RemovalAdvanceOutcome.REFUSED_INDETERMINATE
    record = kb.get_removal_phase_record("sweep-corrupt-store")
    assert record.phase == kb.RemovalPhase.APPLIED
    sweep_state = record.sweep()
    assert sweep_state is not None
    # The in-board-table classes are affected.
    subs = sweep_state["classes"]["subscriptions"]
    assert subs["result"] == kb.SWEEP_RESULT_INDETERMINATE
    assert subs["count"] is None
    # The reason explains why (either open or query failed).
    assert "could not be" in subs["reason"] or "not a database" in subs["reason"]
    # The refusal is durably recorded.
    assert record.refusal_outcome is not None


def test_the_sweep_refuses_indeterminate_when_query_raises(fence_home):
    """A query that raises (header kept, body zeroed) is INDETERMINATE.

    The store can be opened but the query fails — the count could not be
    read.  The result is INDETERMINATE, not not-present-with-no-count.
    """
    import struct

    removal_id = _plain_board("sweep-query-raises")
    _to_applied("sweep-query-raises", removal_id)
    db_path = kb.kanban_db_path(board="sweep-query-raises")
    # Corrupt the body but keep the SQLite header so the open succeeds.
    with open(db_path, "r+b") as f:
        header = f.read(100)  # SQLite header is 100 bytes
        f.seek(0)
        f.write(header)
        f.write(b"\x00" * (db_path.stat().st_size - 100))  # zero the rest

    result = kb.advance_removal_to_swept("sweep-query-raises", removal_id=removal_id)

    assert result.success is False
    assert result.outcome is kb.RemovalAdvanceOutcome.REFUSED_INDETERMINATE
    record = kb.get_removal_phase_record("sweep-query-raises")
    assert record.phase == kb.RemovalPhase.APPLIED
    sweep_state = record.sweep()
    # The class is recorded as indeterminate, with no count claimed.
    subs = sweep_state["classes"]["subscriptions"]
    assert subs["result"] == kb.SWEEP_RESULT_INDETERMINATE
    assert subs["count"] is None
    assert "could not be read" in subs["reason"]


def test_successful_in_board_store_read_records_not_present_with_real_count(fence_home):
    """A genuinely successful in-board-store read records not-present with
    a real integer count, so the legitimate case is not broken.
    """
    removal_id = _plain_board("sweep-real-count")
    _to_applied("sweep-real-count", removal_id)

    result = kb.advance_removal_to_swept("sweep-real-count", removal_id=removal_id)

    assert result.success is True
    record = kb.get_removal_phase_record("sweep-real-count")
    sweep_state = record.sweep()
    # The subscriptions class exists only inside the board's own store.
    subs = sweep_state["classes"]["subscriptions"]
    assert subs["result"] == kb.SWEEP_RESULT_NOT_PRESENT
    # A real integer count (could be 0 but must not be None).
    assert isinstance(subs["count"], int)
    assert subs["count"] >= 0
    assert "OWN store" in subs["reason"]


def test_failed_pointer_clear_records_error_and_refuses_work_incomplete(
    fence_home, monkeypatch
):
    """A failed CLEAR (the write, not the read) records SWEEP_RESULT_ERROR.

    The two failure kinds are kept distinct: a failed READ is
    INDETERMINATE (the absence is not established), while a failed
    ACTION is ERROR (the sweep could not complete its work). This test
    uses a monkeypatch to simulate a write failure after a successful
    read, since creating that condition through filesystem operations
    alone would require a race.
    """
    removal_id = _plain_board("sweep-clear-fails")
    kb.set_current_board("sweep-clear-fails")
    _to_applied("sweep-clear-fails", removal_id)

    # The read will succeed (pointer exists and names this board), but we
    # patch clear_current_board to fail.
    def failing_clear():
        raise OSError("simulated write failure: permission denied")

    monkeypatch.setattr(kb, "clear_current_board", failing_clear)

    result = kb.advance_removal_to_swept("sweep-clear-fails", removal_id=removal_id)

    assert result.success is False
    # A failed ACTION is work-incomplete, not indeterminate.
    assert result.outcome is kb.RemovalAdvanceOutcome.REFUSED_WORK_INCOMPLETE
    record = kb.get_removal_phase_record("sweep-clear-fails")
    assert record.phase == kb.RemovalPhase.APPLIED
    sweep_state = record.sweep()
    pointer_entry = sweep_state["classes"]["current-pointer"]
    # A failed CLEAR is ERROR (action failed), not INDETERMINATE (read failed).
    assert pointer_entry["result"] == kb.SWEEP_RESULT_ERROR
    assert "cannot clear" in pointer_entry["reason"]


def test_swept_is_recordable_only_by_its_own_driver(fence_home):
    """Replaces two tests that asserted the PUBLIC entry point could carry
    a forged sweep state — and its completion instant — into Swept's
    precondition, which then read the caller's own value back as durable
    proof. Both are phase-owned facts now: passing either raises, and the
    generic entry point refuses Swept outright, naming the driver.
    """
    removal_id = _plain_board("sweep-forced")
    _to_applied("sweep-forced", removal_id)
    partial = json.dumps(
        {"version": 1, "classes": {"current-pointer": {"result": "removed"}}}
    )

    for fields in (
        {"sweep_completed_at": int(time.time())},
        {"sweep_state": partial, "sweep_completed_at": int(time.time())},
    ):
        with pytest.raises(ValueError) as excinfo:
            kb.advance_removal_phase(
                "sweep-forced", removal_id=removal_id,
                from_phase=kb.RemovalPhase.APPLIED, to_phase=kb.RemovalPhase.SWEPT,
                **fields,
            )
        for name in fields:
            assert name in str(excinfo.value)
        assert "advance_removal_to_swept" in str(excinfo.value)

    refused = kb.advance_removal_phase(
        "sweep-forced", removal_id=removal_id,
        from_phase=kb.RemovalPhase.APPLIED, to_phase=kb.RemovalPhase.SWEPT,
    )
    assert refused.success is False
    assert refused.transitioned is False
    assert "advance_removal_to_swept" in refused.message

    record = kb.get_removal_phase_record("sweep-forced")
    assert record.phase == kb.RemovalPhase.APPLIED
    assert record.sweep_state is None
    assert record.sweep_completed_at is None


def test_the_precondition_refuses_a_sweep_state_with_a_class_left_out(fence_home):
    """§6.7/IN-3, at the registry's own guard: even a DRIVER that derived
    an incomplete fact cannot record Swept with a class missing. Driven
    through the phase-owned channel a driver uses, because no public
    caller can reach this field at all."""
    removal_id = _plain_board("sweep-partial")
    _to_applied("sweep-partial", removal_id)
    partial = json.dumps(
        {"version": 1, "classes": {"current-pointer": {"result": "removed"}}}
    )

    forced = kb._advance_removal_phase(
        "sweep-partial", removal_id=removal_id,
        from_phase=kb.RemovalPhase.APPLIED, to_phase=kb.RemovalPhase.SWEPT,
        _phase_owned={
            "sweep_state": partial, "sweep_completed_at": int(time.time()),
        },
    )

    assert forced.success is False
    assert forced.outcome is kb.RemovalAdvanceOutcome.REFUSED_WORK_INCOMPLETE
    assert kb.get_removal_phase_record("sweep-partial").phase == kb.RemovalPhase.APPLIED


def test_a_sweep_repeat_refuses_when_its_recorded_state_is_gone(fence_home):
    """A no-op success at Swept is a claim that every record class was
    resolved. An absent sweep state cannot support that claim."""
    removal_id = _plain_board("sweep-repeat-factless")
    _to_applied("sweep-repeat-factless", removal_id)
    assert kb.advance_removal_to_swept(
        "sweep-repeat-factless", removal_id=removal_id
    ).success
    damage_phase_column_behind_the_primitive(
        "sweep-repeat-factless", "sweep_state", None
    )

    repeat = kb.advance_removal_to_swept(
        "sweep-repeat-factless", removal_id=removal_id
    )

    assert repeat.success is False
    assert repeat.outcome is kb.RemovalAdvanceOutcome.REFUSED_WORK_INCOMPLETE
    assert repeat.transitioned is False
    assert kb.get_removal_phase_record(
        "sweep-repeat-factless"
    ).phase == kb.RemovalPhase.SWEPT


def test_a_sweep_repeat_refuses_when_durable_state_contradicts_the_record(fence_home):
    """The work itself is re-checked where it is checkable: a recorded
    ``removed`` verdict for the current-board pointer is contradicted by a
    pointer that names this board again."""
    removal_id = _plain_board("sweep-repeat-contradicted")
    kb.set_current_board("sweep-repeat-contradicted")
    _to_applied("sweep-repeat-contradicted", removal_id)
    assert kb.advance_removal_to_swept(
        "sweep-repeat-contradicted", removal_id=removal_id
    ).success
    assert not kb.current_board_path().exists()
    kb.set_current_board("sweep-repeat-contradicted")

    repeat = kb.advance_removal_to_swept(
        "sweep-repeat-contradicted", removal_id=removal_id
    )

    assert repeat.success is False
    assert repeat.outcome is kb.RemovalAdvanceOutcome.REFUSED_WORK_INCOMPLETE
    assert kb.get_removal_phase_record(
        "sweep-repeat-contradicted"
    ).phase == kb.RemovalPhase.SWEPT


def test_a_sweep_repeat_with_every_class_resolved_is_still_a_no_op(fence_home):
    removal_id = _plain_board("sweep-repeat-ok")
    _to_applied("sweep-repeat-ok", removal_id)
    assert kb.advance_removal_to_swept(
        "sweep-repeat-ok", removal_id=removal_id
    ).success
    before = kb.get_removal_phase_record("sweep-repeat-ok")

    repeat = kb.advance_removal_to_swept("sweep-repeat-ok", removal_id=removal_id)

    assert repeat.success is True
    assert repeat.outcome is kb.RemovalAdvanceOutcome.IDEMPOTENT_NOOP
    assert repeat.transitioned is False
    after = kb.get_removal_phase_record("sweep-repeat-ok")
    assert after.sweep_state == before.sweep_state
    assert after.sweep_completed_at == before.sweep_completed_at
