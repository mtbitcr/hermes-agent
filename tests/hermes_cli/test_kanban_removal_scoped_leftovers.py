"""The untracked-leftovers check is scoped to this board's own roots (§14.2).

Every test here drives the PRODUCTION checker,
``kanban_db.check_scoped_leftovers``, and the real removal path that
calls it. Nothing asserts about a key that does not exist, and every
collection is asserted non-empty before it is iterated — a loop that can
run zero times proves nothing.

What is being pinned down:

* The scan domain is derived from the carry ledger's EXACT recorded
  identities: this board's own storage area, its own work-area root, and
  each work area and registration entry it recorded creating.
* A genuine leftover INSIDE one of those roots is flagged, and it blocks
  a permanent removal from reaching `done`.
* Another board's directory, another board's registration, another
  board's work area and an unrelated file in a shared parent are never in
  the domain, never flagged and never touched.
* A root that could not be READ is recorded as unreadable, not as clean.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_WORKTREE = Path(__file__).resolve().parents[2]
if str(_WORKTREE) not in sys.path:
    sys.path.insert(0, str(_WORKTREE))

import pytest

from hermes_cli import kanban_db as kb
from tests.hermes_cli._kanban_fence_support import (
    add_linked_work_area,
    create_fenced_board,
    make_git_repo,
    permanent_confirmation,
    ready_task,
    record_git_receipt,
    register_row,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _carry(slug: str, *, mode: str = "permanent") -> str:
    """Drive a real removal as far as Carried, so the ledger exists."""
    kwargs = {}
    if kb.RemovalMode(mode) == kb.RemovalMode.PERMANENT:
        kwargs["permanent_confirmation"] = permanent_confirmation(slug)
    intent = kb.record_removal_intent(slug, mode=mode, **kwargs)
    assert intent.success, intent.message
    for driver in (
        kb.advance_removal_to_fenced,
        kb.advance_removal_to_quiesced,
        kb.advance_removal_to_carried,
    ):
        step = driver(slug, removal_id=intent.removal_id)
        assert step.success, f"{driver.__name__}: {step.message}"
    return intent.removal_id


def _board_with_work_area(slug: str, repo: Path) -> dict:
    """A fenced board that really owns a linked work area in *repo*."""
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
    return {
        "slug": slug,
        "task": task_id,
        "repo": repo,
        "work_area": work_area,
        "registration": repo / ".git" / "worktrees" / task_id,
    }


def _roots(report: kb.ScopedLeftoverReport) -> set:
    assert report.roots, "the leftovers report named no root at all"
    return {root["root"] for root in report.roots}


# ---------------------------------------------------------------------------
# The domain is exactly this board's own roots
# ---------------------------------------------------------------------------

def test_the_domain_is_derived_from_the_carried_ledger(fence_home, tmp_path):
    """Every root is an exact recorded identity, and there are no others."""
    setup = _board_with_work_area("scope-own", tmp_path / "repo")
    _carry("scope-own")

    report = kb.check_scoped_leftovers("scope-own")

    roots = _roots(report)
    assert roots == {
        str(kb.board_dir("scope-own")),
        str(kb.workspaces_root("scope-own")),
        str(setup["work_area"]),
        str(setup["registration"]),
    }, roots
    for root in report.roots:
        assert "*" not in root["root"], "a root carried a glob pattern"
        assert "%" not in root["root"], "a root carried a SQL pattern"
        assert root["source"], "a root does not say where it came from"
        assert root["member"], "a root does not say which member it is"


def test_another_boards_roots_are_never_in_the_domain(fence_home, tmp_path):
    """Another board's directory, store and work area are all outside it."""
    repo = tmp_path / "shared-repo"
    mine = _board_with_work_area("scope-mine", repo)
    theirs = _board_with_work_area("scope-theirs", repo)
    _carry("scope-mine")

    report = kb.check_scoped_leftovers("scope-mine")

    roots = _roots(report)
    assert str(mine["work_area"]) in roots
    for outside in (
        kb.board_dir("scope-theirs"),
        kb.kanban_db_path(board="scope-theirs"),
        kb.workspaces_root("scope-theirs"),
        theirs["work_area"],
        theirs["registration"],
        kb.kanban_home(),
        repo,
    ):
        assert str(outside) not in roots, f"{outside} entered the domain"


def test_an_unrelated_file_in_a_shared_parent_is_never_flagged(
    fence_home, tmp_path
):
    """A file next to the boards directory is not this board's leftover."""
    create_fenced_board("scope-parent")
    conn = kb.connect(board="scope-parent")
    ready_task(conn)
    conn.close()
    unrelated = kb.boards_root() / "not-a-board.txt"
    unrelated.write_text("unrelated\n", encoding="utf-8")
    _carry("scope-parent")

    report = kb.check_scoped_leftovers("scope-parent")

    assert _roots(report)
    flagged = {leftover["root"] for leftover in report.leftovers}
    assert str(unrelated) not in flagged
    assert str(kb.boards_root()) not in _roots(report)
    assert unrelated.read_text(encoding="utf-8") == "unrelated\n"


# ---------------------------------------------------------------------------
# A genuine leftover inside our own roots IS flagged, and it blocks
# ---------------------------------------------------------------------------

def test_a_leftover_inside_our_own_root_is_flagged_with_its_exact_path(
    fence_home, tmp_path
):
    """Content left in this board's own storage area is flagged, not ignored."""
    create_fenced_board("scope-leftover")
    conn = kb.connect(board="scope-leftover")
    ready_task(conn)
    conn.close()
    _carry("scope-leftover")

    report = kb.check_scoped_leftovers("scope-leftover")

    assert report.leftovers, (
        "the board's own storage area is still there and was not flagged"
    )
    assert not report.clean
    flagged = {leftover["root"] for leftover in report.leftovers}
    assert str(kb.board_dir("scope-leftover")) in flagged
    for leftover in report.leftovers:
        assert leftover["reason"], "a leftover was flagged with no reason"
        assert isinstance(leftover["entries"], list)


def test_a_clean_board_reports_no_leftovers(fence_home, tmp_path):
    """After a real permanent removal, every own root is gone."""
    setup = _board_with_work_area("scope-clean", tmp_path / "repo")

    result = kb.remove_board_fenced(
        "scope-clean", mode="permanent",
        permanent_confirmation=permanent_confirmation("scope-clean"),
    )
    assert result.success, result.message

    report = kb.check_scoped_leftovers("scope-clean")

    assert _roots(report), "the domain was empty, so 'clean' means nothing"
    assert report.leftovers == []
    assert report.unreadable == []
    assert report.clean
    assert not setup["work_area"].exists()
    assert not kb.board_dir("scope-clean").exists()


@pytest.mark.skipif(
    hasattr(os, "geteuid") and os.geteuid() == 0,
    reason=(
        "this case needs a root the remover genuinely cannot unlink; a mode "
        "bit does not constrain a superuser, so the scenario cannot be "
        "constructed as root and the assertion would be vacuous"
    ),
)
def test_an_undestroyable_own_root_blocks_the_removal_from_reaching_done(
    fence_home, tmp_path
):
    """A receipt may not claim destruction while the content is readable."""
    setup = _board_with_work_area("scope-blocks", tmp_path / "repo")
    board_directory = kb.board_dir("scope-blocks")
    # A genuinely undeletable subtree inside this board's OWN storage
    # area: the workspace root is read-only, so its content cannot be
    # unlinked and the board's directory really survives destruction.
    stuck = kb.workspaces_root("scope-blocks") / "keep"
    stuck.mkdir(parents=True)
    (stuck / "live-work.txt").write_text("still here\n", encoding="utf-8")
    stuck.chmod(0o500)

    try:
        result = kb.remove_board_fenced(
            "scope-blocks", mode="permanent",
            permanent_confirmation=permanent_confirmation("scope-blocks"),
        )

        assert not result.success, result.message
        assert (stuck / "live-work.txt").exists(), "live content was destroyed"
        record = kb.get_removal_phase_record("scope-blocks")
        assert record.phase is not kb.RemovalPhase.DONE
        assert register_row("scope-blocks")["lifecycle"] == (
            kb.BoardLifecycle.REMOVING.value
        )
        assert kb.get_permanent_removal_receipt(
            "scope-blocks", record.removal_id
        ) is None, "a receipt claimed a removal that did not complete"

        report = kb.check_scoped_leftovers("scope-blocks")
        flagged = {leftover["root"] for leftover in report.leftovers}
        assert flagged, "the surviving root was not flagged"
        assert str(board_directory) in flagged
        # The container is untouched: nothing about a blocked removal
        # reaches outside this board's own roots.
        assert setup["repo"].exists()
    finally:
        if stuck.exists():
            stuck.chmod(0o700)


def test_an_unreadable_root_is_recorded_as_unreadable_not_clean(
    fence_home, tmp_path
):
    """A failed read is never a proof of absence."""
    create_fenced_board("scope-unreadable")
    conn = kb.connect(board="scope-unreadable")
    ready_task(conn)
    conn.close()
    _carry("scope-unreadable")
    board_directory = kb.board_dir("scope-unreadable")
    board_directory.chmod(0o000)
    try:
        report = kb.check_scoped_leftovers("scope-unreadable")
        unreadable_roots = {item["root"] for item in report.unreadable}
        flagged_roots = {item["root"] for item in report.leftovers}
        assert not report.clean
        # Either it could not be listed (unreadable) or it is still there
        # (a leftover) — what must never happen is "clean".
        assert str(board_directory) in (unreadable_roots | flagged_roots)
    finally:
        board_directory.chmod(0o700)


def test_the_checker_touches_nothing(fence_home, tmp_path):
    """It is read-only: every root it flags is still there afterwards."""
    setup = _board_with_work_area("scope-readonly", tmp_path / "repo")
    _carry("scope-readonly")
    content = setup["work_area"] / f"{setup['task']}.txt"

    first = kb.check_scoped_leftovers("scope-readonly")
    second = kb.check_scoped_leftovers("scope-readonly")

    assert first.leftovers, "nothing was flagged, so 'touches nothing' is empty"
    assert [item["root"] for item in first.leftovers] == [
        item["root"] for item in second.leftovers
    ]
    assert content.exists()
    assert setup["registration"].exists()
    assert kb.board_dir("scope-readonly").exists()


def test_the_sweep_records_a_result_for_every_class_it_names(
    fence_home, tmp_path
):
    """§6.7's per-class results live under 'classes', and every one is there."""
    _board_with_work_area("scope-sweep", tmp_path / "repo")

    result = kb.remove_board_fenced(
        "scope-sweep", mode="permanent",
        permanent_confirmation=permanent_confirmation("scope-sweep"),
    )
    assert result.success, result.message

    sweep = kb.get_removal_phase_record("scope-sweep").sweep()
    assert sweep is not None, "no §6.7 sweep state was recorded"
    classes = sweep["classes"]
    assert set(classes) == {key for key, _ in kb.SWEEP_RECORD_CLASSES}
    for key, entry in classes.items():
        assert entry["result"] not in (
            kb.SWEEP_RESULT_ERROR, kb.SWEEP_RESULT_INDETERMINATE,
        ), f"{key}: {entry.get('reason')}"
        assert entry["reason"], f"{key} recorded no reason"
        assert "scope-sweep" not in (entry.get("reason") or "").replace(
            "scope-sweep", "", 1
        ), f"{key} mentioned another board"
