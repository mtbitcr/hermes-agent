"""§7.2 destroys the work areas it owns, and nothing it does not own.

Every test here creates a REAL git repository with REAL linked work
areas, so the container's own registration metadata — the thing that
establishes which board owns which directory — is written by git rather
than by the test.

What is being pinned down:

* A successful permanent removal really destroys the work-area DIRECTORY
  and its content, not just the registration metadata.
* The content goes FIRST and the metadata LAST, journalled durably as it
  happens, so no crash can orphan content nothing claims.
* A work area whose ownership is not provable is left completely intact
  and recorded as a BLOCKING failure — the removal does not reach `done`.
* An `unresolved` registration blocks too, with a recorded operator item.
* An unknown IN member type blocks; it is never reported as a success.
* Another board's work area, registration and content are never flagged
  and never touched.
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
    add_linked_work_area,
    create_fenced_board,
    git,
    make_git_repo,
    permanent_confirmation,
    read_only,
    ready_task,
    record_git_receipt,
    register_row,
)


# ---------------------------------------------------------------------------
# Setup: a fenced board that really owns a linked work area
# ---------------------------------------------------------------------------

def _board_with_work_area(
    slug: str, repo: Path, *, task_title: str = "work in a worktree"
) -> dict:
    """A fenced board whose durable task row names a REAL linked work area."""
    base = make_git_repo(repo)
    create_fenced_board(slug)
    conn = kb.connect(board=slug)
    task_id = ready_task(conn, title=task_title)
    work_area, head = add_linked_work_area(
        repo, task_id, branch=f"hermes/{task_id}",
    )
    record_git_receipt(
        conn, task_id,
        workspace_path=work_area,
        branch_name=f"hermes/{task_id}",
        base_commit=base,
        head_commit=head,
    )
    conn.close()
    return {
        "slug": slug,
        "task": task_id,
        "repo": repo,
        "work_area": work_area,
        "registration": repo / ".git" / "worktrees" / task_id,
        "base": base,
        "head": head,
    }


def _remove_permanently(slug: str) -> kb.FencedRemovalResult:
    """Drive the real common driver, with a real statement-bound confirmation."""
    return kb.remove_board_fenced(
        slug, mode="permanent",
        permanent_confirmation=permanent_confirmation(slug),
    )


def _ledgered_registrations(slug: str) -> list:
    """The C13 registrations this board's carry ledger recorded."""
    record = kb.get_removal_phase_record(slug)
    assert record is not None, f"{slug} has no removal record"
    payload = record.carried()
    assert payload is not None, f"{slug} carried nothing"
    for entry in payload.get("outside_resource_ledger") or []:
        if entry.get("member") == "work-area-registration":
            return entry.get("registrations") or []
    pytest.fail("the carry ledger names no work-area-registration member")


# ---------------------------------------------------------------------------
# The content really goes
# ---------------------------------------------------------------------------

def test_permanent_removal_destroys_the_work_area_content(fence_home, tmp_path):
    """The worktree DIRECTORY and its content are gone, not just metadata."""
    setup = _board_with_work_area("wa-destroy", tmp_path / "repo")
    content = setup["work_area"] / f"{setup['task']}.txt"
    assert content.read_text(encoding="utf-8") == "work\n"

    result = _remove_permanently("wa-destroy")

    assert result.success, result.message
    assert not content.exists(), "the work area's content is still readable"
    assert not setup["work_area"].exists(), "the work area directory survived"
    assert not setup["registration"].exists(), "the registration survived"
    # C7: the reference and its commits are KEPT. The shared repository
    # itself is never deleted, moved or rewritten.
    assert setup["repo"].exists()
    assert git(setup["repo"], "cat-file", "-t", setup["head"]) == "commit"


def test_the_content_is_journalled_before_the_metadata_goes(fence_home, tmp_path):
    """Destroy-then-deregister, in that order, durably journalled."""
    setup = _board_with_work_area("wa-order", tmp_path / "repo")

    result = _remove_permanently("wa-order")
    assert result.success, result.message

    record = kb.get_removal_phase_record("wa-order")
    items = record.journal()["items"]
    assert items, "nothing was journalled at all"
    steps = [item["step"] for item in items]
    assert kb.APPLY_JOURNAL_WORK_AREA_DESTROYED in steps
    assert kb.APPLY_JOURNAL_DEREGISTERED in steps
    assert steps.index(kb.APPLY_JOURNAL_WORK_AREA_DESTROYED) < steps.index(
        kb.APPLY_JOURNAL_DEREGISTERED
    ), (
        "the registration metadata was journalled gone before the content: a "
        f"crash between them would orphan it — {steps}"
    )
    # Each item names its EXACT identity, and the sequence is monotonic.
    assert [item["sequence"] for item in items] == list(
        range(1, len(items) + 1)
    )
    destroyed = kb.journalled_apply_identities(
        record, action=kb.APPLY_JOURNAL_WORK_AREA_DESTROYED
    )
    assert str(setup["work_area"]) in destroyed


def test_ownership_is_verified_against_the_containers_own_metadata(
    fence_home, tmp_path
):
    """A real registration is owned; a fabricated one is not."""
    setup = _board_with_work_area("wa-own", tmp_path / "repo")
    other_repo = tmp_path / "other-repo"
    make_git_repo(other_repo)
    other_area, _head = add_linked_work_area(other_repo, "theirs", branch="theirs")

    owned = kb.verify_work_area_ownership({
        "work_area": str(setup["work_area"]),
        "container": str(setup["repo"]),
        "registration": str(setup["registration"]),
    })
    assert owned.owned is True, owned.reason
    assert owned.evidence["gitdir"], "ownership was decided without evidence"

    # A registration entry in THIS board's container that points at
    # another repository's work area is not this board's to destroy.
    foreign = kb.verify_work_area_ownership({
        "work_area": str(other_area),
        "container": str(setup["repo"]),
        "registration": str(
            setup["repo"] / ".git" / "worktrees" / other_area.name
        ),
    })
    assert foreign.owned is not True, foreign.reason
    assert other_area.exists(), "the other repository's work area was touched"


def test_a_work_area_with_no_provable_owner_blocks_and_is_left_intact(
    fence_home, tmp_path
):
    """No ownership proof → nothing touched, a blocking failure, no `done`."""
    setup = _board_with_work_area("wa-unproven", tmp_path / "repo")
    content = setup["work_area"] / f"{setup['task']}.txt"

    # Remove the container's own registration metadata: the directory is
    # still there, but nothing durable now says whose it is.
    import shutil

    shutil.rmtree(setup["registration"])

    result = _remove_permanently("wa-unproven")

    assert not result.success, result.message
    assert content.exists(), "content with no provable owner was destroyed"
    assert setup["work_area"].exists()
    record = kb.get_removal_phase_record("wa-unproven")
    assert record.phase is not kb.RemovalPhase.DONE
    assert register_row("wa-unproven")["lifecycle"] == (
        kb.BoardLifecycle.REMOVING.value
    )
    assert kb.applied_mode_content_is_outstanding(record)
    blocked = [
        item for item in record.journal()["items"] if not item.get("ok")
    ]
    assert blocked, "the refusal was not journalled"
    assert any(str(setup["work_area"]) == item["identity"] for item in blocked)


def test_an_unresolved_registration_is_a_recorded_blocking_failure(
    fence_home, tmp_path
):
    """A work area whose container cannot be derived blocks, with an item."""
    slug = "wa-unresolved"
    create_fenced_board(slug)
    conn = kb.connect(board=slug)
    task_id = ready_task(conn)
    # A worktree work area recorded at a path with no ``.worktrees`` parent:
    # ``_worktree_container`` cannot derive its shared container, so the
    # carry records it as UNRESOLVED rather than dropping it.
    stray = tmp_path / "stray-work-area"
    stray.mkdir()
    (stray / "content.txt").write_text("live work\n", encoding="utf-8")
    record_git_receipt(
        conn, task_id,
        workspace_path=stray,
        branch_name=f"hermes/{task_id}",
        base_commit="0" * 40,
        head_commit="1" * 40,
    )
    conn.close()

    result = _remove_permanently(slug)

    assert not result.success, result.message
    record = kb.get_removal_phase_record(slug)
    assert record.phase is not kb.RemovalPhase.DONE
    assert (stray / "content.txt").exists(), "an unresolved area was destroyed"
    unresolved = [
        item for item in record.journal()["items"]
        if not item.get("ok") and "unresolved" in (item.get("reason") or "")
    ]
    assert unresolved, (
        "an unresolved registration reached the apply with no recorded failure"
    )
    assert kb.removal_operator_items(record), (
        "a blocked removal surfaced no operator item"
    )


def test_an_unknown_in_member_type_is_a_failure_not_a_skip(fence_home):
    """"I do not know how to destroy this" is never a success."""
    ok, reason, detail = kb._destroy_in_resource({
        "member": "some-member-nobody-taught-this-driver",
        "disposition": "in",
        "identity": "/nonexistent/exact/identity",
    })

    assert ok is False, reason
    assert detail["destroyed"] is False
    assert "no destruction is defined" in reason


# ---------------------------------------------------------------------------
# Another board's content is never flagged and never touched
# ---------------------------------------------------------------------------

def test_another_boards_work_area_is_never_flagged_or_touched(
    fence_home, tmp_path
):
    """Two boards, one shared repository: only the removed board's area goes."""
    repo = tmp_path / "shared-repo"
    mine = _board_with_work_area("wa-mine", repo)

    create_fenced_board("wa-theirs")
    other_conn = kb.connect(board="wa-theirs")
    other_task = ready_task(other_conn, title="their work")
    their_area, their_head = add_linked_work_area(
        repo, other_task, branch=f"hermes/{other_task}", content="theirs\n",
    )
    record_git_receipt(
        other_conn, other_task,
        workspace_path=their_area,
        branch_name=f"hermes/{other_task}",
        base_commit=mine["base"],
        head_commit=their_head,
    )
    other_conn.close()
    their_registration = repo / ".git" / "worktrees" / other_task
    their_content = their_area / f"{other_task}.txt"

    result = _remove_permanently("wa-mine")
    assert result.success, result.message

    # Mine is gone…
    assert not mine["work_area"].exists()
    assert not mine["registration"].exists()
    # …and theirs is completely untouched, content and metadata alike.
    assert their_content.read_text(encoding="utf-8") == "theirs\n"
    assert their_registration.exists()
    assert kb.board_dir("wa-theirs").exists()
    assert kb.kanban_db_path(board="wa-theirs").exists()
    assert register_row("wa-theirs")["lifecycle"] == kb.BoardLifecycle.LIVE.value

    # The leftovers domain never even named the other board's roots.
    roots = {
        root["root"]
        for root in kb.scoped_leftover_roots(
            "wa-mine", kb.get_removal_phase_record("wa-mine").carried()
        )
    }
    assert roots, "the leftovers domain was empty"
    assert str(mine["work_area"]) in roots
    assert str(their_area) not in roots
    assert str(their_registration) not in roots


def test_the_ledger_records_the_exact_work_area_it_will_act_on(
    fence_home, tmp_path
):
    """Each work area is carried as an owned member with an exact identity."""
    setup = _board_with_work_area("wa-ledger", tmp_path / "repo")
    intent = kb.record_removal_intent(
        "wa-ledger", mode="permanent",
        permanent_confirmation=permanent_confirmation("wa-ledger"),
    )
    assert intent.success, intent.message
    for driver in (
        kb.advance_removal_to_fenced,
        kb.advance_removal_to_quiesced,
        kb.advance_removal_to_carried,
    ):
        step = driver("wa-ledger", removal_id=intent.removal_id)
        assert step.success, f"{driver.__name__}: {step.message}"

    registrations = _ledgered_registrations("wa-ledger")

    assert len(registrations) == 1, registrations
    entry = registrations[0]
    assert entry["work_area"] == str(setup["work_area"])
    assert entry["registration"] == str(setup["registration"])
    assert entry["container"] == str(setup["repo"])
    assert entry["task"] == setup["task"]
    for value in entry.values():
        assert "*" not in str(value), "an identity carried a glob pattern"
