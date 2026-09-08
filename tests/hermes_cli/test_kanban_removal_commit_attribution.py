"""One consistent rule for commits, applied regardless of identity (§12.3).

Every test here builds a REAL temporary git repository and makes REAL
commits — including one authored and committed under a genuinely
different ``user.name`` / ``user.email`` — then drives the REAL removal
path (``remove_board_fenced`` / ``apply_permanent_mode_content``) and
asserts against the receipt that actually lands in the archive.

What is being pinned down:

* ``repository``, ``reference``, ``base_commit`` and ``head_commit`` are
  each emitted EXPLICITLY.
* The board-created commit list is exact, and the BASE is not on it.
* A foreign-identity head this board's receipts name IS on it — the rule
  is applied identically whoever authored the commit.
* An absorbed head is disclosed separately and is NOT claimed as this
  board's.
* Per-advance provenance comes from the recorded run receipts, and the
  receipt NAMES that source rather than leaving a reader to assume it.
* Where one advance produced SEVERAL commits, only its head is derivable,
  so the receipt marks the commit list's completeness UNVERIFIED instead
  of presenting it as a complete enumeration.

Every removal here runs through the SHIPPED ``boards rm`` command — the
real argparse tree and dispatch — so what is asserted about the receipt is
asserted about the receipt an operator's own invocation produces.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import sys
from pathlib import Path

_WORKTREE = Path(__file__).resolve().parents[2]
if str(_WORKTREE) not in sys.path:
    sys.path.insert(0, str(_WORKTREE))

from hermes_cli import kanban_db as kb
from tests.hermes_cli._kanban_fence_support import (
    DEFAULT_GIT_IDENTITY,
    FOREIGN_GIT_IDENTITY,
    add_linked_work_area,
    board_with_multi_commit_advance,
    commit_series,
    create_fenced_board,
    git,
    make_git_repo,
    ready_task,
    record_git_receipt,
    record_run_advance,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _remove_permanently(slug: str) -> "tuple[int, str, str]":
    """Run the SHIPPED permanent-removal command and return its outcome.

    ``boards rm <slug> --delete --confirm '<exact statement-bound line>'``
    is what an operator runs; ``--delete`` selects permanent and there is
    deliberately no bare ``--yes``. Returns ``(exit_code, output,
    removal_id)`` with the removal id read back from durable state.
    """
    from hermes_cli import kanban as kanban_cli

    required = kb.permanent_removal_disclosure(slug).required_response
    wrap = argparse.ArgumentParser(prog="hermes-test", add_help=False)
    top = wrap.add_subparsers(dest="_top")
    parser = kanban_cli.build_parser(top)
    args = parser.parse_args(
        ["boards", "rm", slug, "--delete", "--confirm", required]
    )
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(buffer):
        code = kanban_cli.kanban_command(args)
    record = kb.get_removal_phase_record(slug)
    return code, buffer.getvalue(), None if record is None else record.removal_id


def _receipt_references(slug: str, removal_id: str) -> list:
    """The §12.3 reference sections of the receipt that really landed."""
    receipt = kb.get_permanent_removal_receipt(slug, removal_id)
    assert receipt is not None, (
        f"no §12 receipt was recorded for {slug}/{removal_id}"
    )
    version_control = receipt["version_control"]
    assert version_control["clause"] == "C7"
    references = version_control["references"]
    assert references, "the receipt names no retained reference at all"
    return references


def _commit_identity(repo: Path, commit: str) -> "tuple[str, str]":
    """The real author name/email git recorded for *commit*."""
    line = git(repo, "show", "-s", "--format=%an|%ae", commit)
    name, email = line.split("|", 1)
    return name, email


def _board_with_foreign_head(slug: str, repo: Path) -> dict:
    """A board whose ONE recorded head was authored by a foreign identity."""
    base = make_git_repo(repo, author=DEFAULT_GIT_IDENTITY)
    create_fenced_board(slug)
    conn = kb.connect(board=slug)
    task_id = ready_task(conn, title="work a stranger committed")
    branch = f"hermes/{task_id}"
    work_area, head = add_linked_work_area(
        repo, task_id, branch=branch, author=FOREIGN_GIT_IDENTITY,
    )
    record_git_receipt(
        conn, task_id, workspace_path=work_area, branch_name=branch,
        base_commit=base, head_commit=head,
    )
    run_id = record_run_advance(
        conn, task_id, branch=branch, base_commit=base, head_commit=head,
    )
    conn.close()
    return {
        "slug": slug, "task": task_id, "repo": repo, "branch": branch,
        "base": base, "head": head, "run": run_id,
    }


# ---------------------------------------------------------------------------
# A foreign-identity commit is named, under the one rule
# ---------------------------------------------------------------------------

def test_a_genuinely_foreign_authored_head_is_named_on_the_receipt(
    fence_home, tmp_path
):
    """The rule is identity-blind: a stranger's commit is still ours to name."""
    setup = _board_with_foreign_head("foreign-author", tmp_path / "repo")
    # The commit really was made by someone else — assert that first, or
    # the rest of this test proves nothing about foreign identities.
    author_name, author_email = _commit_identity(setup["repo"], setup["head"])
    assert (author_name, author_email) == FOREIGN_GIT_IDENTITY
    assert (author_name, author_email) != DEFAULT_GIT_IDENTITY

    code, output, removal_id = _remove_permanently("foreign-author")
    assert code == 0, output

    references = _receipt_references("foreign-author", removal_id)
    reference, = [r for r in references if r["task"] == setup["task"]]
    assert reference["repository"] == str(setup["repo"])
    assert reference["reference"] == setup["branch"]
    assert reference["reference_name"] == setup["branch"]
    assert reference["base_commit"] == setup["base"]
    assert reference["head_commit"] == setup["head"]
    # The foreign-authored head is on this board's created list.
    assert setup["head"] in reference["board_created_commits"]
    assert reference["commit_count"] == len(reference["board_created_commits"])
    # The rule is stated on the receipt and says identity is irrelevant.
    assert reference["commit_rule"] == kb.BOARD_CREATED_COMMIT_RULE
    assert "regardless of the authoring" in reference["commit_rule"]


def test_the_base_commit_is_never_claimed_as_board_created(
    fence_home, tmp_path
):
    """The base is emitted explicitly AND excluded from the created list."""
    setup = _board_with_foreign_head("no-base-claim", tmp_path / "repo")

    code, output, removal_id = _remove_permanently("no-base-claim")
    assert code == 0, output

    reference, = [
        r for r in _receipt_references("no-base-claim", removal_id)
        if r["task"] == setup["task"]
    ]
    created = reference["board_created_commits"]
    assert created, "the receipt claims this board created nothing at all"
    assert setup["base"] not in created, (
        "the base commit is on the created list: this board did not create "
        "the commit its work started from"
    )
    assert reference["base_commit"] == setup["base"]
    assert reference["base_commit_excluded"] is True


def test_every_created_commit_carries_its_own_provenance(fence_home, tmp_path):
    """Per-commit provenance names the durable receipt it came from."""
    setup = _board_with_foreign_head("provenance", tmp_path / "repo")

    code, output, removal_id = _remove_permanently("provenance")
    assert code == 0, output

    reference, = [
        r for r in _receipt_references("provenance", removal_id)
        if r["task"] == setup["task"]
    ]
    created = reference["board_created_commits"]
    provenance = reference["commit_provenance"]
    assert created, "nothing was recorded as created"
    assert provenance, "no per-commit provenance was recorded"
    assert {item["commit"] for item in provenance} == set(created)
    for item in provenance:
        assert item["source"] in (
            "advance-head-receipt", "task-head-receipt",
        ), item
        assert item["rule"], item


def test_the_per_advance_ledger_comes_from_the_creation_time_run_receipts(
    fence_home, tmp_path
):
    """Two advances, two run receipts, both heads named exactly."""
    repo = tmp_path / "repo"
    base = make_git_repo(repo)
    create_fenced_board("advances")
    conn = kb.connect(board="advances")
    task_id = ready_task(conn, title="two advances")
    branch = f"hermes/{task_id}"
    work_area, first_head = add_linked_work_area(repo, task_id, branch=branch)
    (work_area / "second.txt").write_text("more\n", encoding="utf-8")
    git(work_area, "add", "second.txt")
    git(work_area, "commit", "-m", "second advance", author=FOREIGN_GIT_IDENTITY)
    second_head = git(work_area, "rev-parse", "HEAD")
    assert first_head != second_head

    first_run = record_run_advance(
        conn, task_id, branch=branch, base_commit=base, head_commit=first_head,
    )
    second_run = record_run_advance(
        conn, task_id, branch=branch, base_commit=first_head,
        head_commit=second_head,
    )
    record_git_receipt(
        conn, task_id, workspace_path=work_area, branch_name=branch,
        base_commit=base, head_commit=second_head,
    )
    conn.close()

    code, output, removal_id = _remove_permanently("advances")
    assert code == 0, output

    reference, = [
        r for r in _receipt_references("advances", removal_id)
        if r["task"] == task_id
    ]
    assert reference["advance_count"] == 2
    assert [advance["run"] for advance in reference["advances"]] == [
        first_run, second_run,
    ]
    assert reference["board_created_commits"] == [first_head, second_head]
    assert base not in reference["board_created_commits"]
    assert reference["head_commit"] == second_head
    assert reference["derivation"], "the receipt does not say what it read"


# ---------------------------------------------------------------------------
# An absorbed head is disclosed, never claimed
# ---------------------------------------------------------------------------

def test_an_absorbed_head_is_disclosed_and_not_on_the_created_list(
    fence_home, tmp_path
):
    """A head merged in from another subject's work is another's, disclosed."""
    repo = tmp_path / "repo"
    base = make_git_repo(repo)
    create_fenced_board("absorbed")
    conn = kb.connect(board="absorbed")

    parent_id = ready_task(conn, title="the parent's own work")
    parent_branch = f"hermes/{parent_id}"
    parent_area, parent_head = add_linked_work_area(
        repo, parent_id, branch=parent_branch,
    )
    record_git_receipt(
        conn, parent_id, workspace_path=parent_area,
        branch_name=parent_branch, base_commit=base, head_commit=parent_head,
    )
    record_run_advance(
        conn, parent_id, branch=parent_branch, base_commit=base,
        head_commit=parent_head,
    )

    child_id = ready_task(conn, title="work that absorbed the parent's head")
    child_branch = f"hermes/{child_id}"
    child_area, _ = add_linked_work_area(repo, child_id, branch=child_branch)
    git(
        child_area, "merge", "--no-ff", "--no-edit", parent_head,
        author=DEFAULT_GIT_IDENTITY,
    )
    child_head = git(child_area, "rev-parse", "HEAD")
    record_git_receipt(
        conn, child_id, workspace_path=child_area, branch_name=child_branch,
        base_commit=base, head_commit=child_head,
    )
    record_run_advance(
        conn, child_id, branch=child_branch, base_commit=base,
        head_commit=child_head,
        absorbed_heads=[{"task": parent_id, "head_commit": parent_head}],
    )
    conn.close()

    code, output, removal_id = _remove_permanently("absorbed")
    assert code == 0, output

    references = _receipt_references("absorbed", removal_id)
    child, = [r for r in references if r["task"] == child_id]
    parent, = [r for r in references if r["task"] == parent_id]

    # Disclosed by fact, count and exact identity on the absorbing side…
    assert child["absorbed_head_count"] == 1
    absorbed, = child["absorbed_heads"]
    assert absorbed["head_commit"] == parent_head
    assert absorbed["absorbed_from_task"] == parent_id
    assert absorbed["disclosure"]
    # …and NOT claimed as a commit that reference created.
    assert parent_head not in child["board_created_commits"]
    assert child_head in child["board_created_commits"]
    # The parent's own reference still names its own head, under the same
    # rule — the head is claimed exactly once, by the advance that made it.
    assert parent_head in parent["board_created_commits"]
    assert parent["absorbed_heads"] == []


# ---------------------------------------------------------------------------
# The rule is one rule, and the receipt says so
# ---------------------------------------------------------------------------

def test_the_same_rule_governs_own_and_foreign_identities(fence_home, tmp_path):
    """Two heads on one board, one identity each: both treated identically."""
    repo = tmp_path / "repo"
    base = make_git_repo(repo)
    create_fenced_board("one-rule")
    conn = kb.connect(board="one-rule")
    heads = {}
    for label, identity in (
        ("own", DEFAULT_GIT_IDENTITY), ("foreign", FOREIGN_GIT_IDENTITY),
    ):
        task_id = ready_task(conn, title=f"{label} work")
        branch = f"hermes/{task_id}"
        area, head = add_linked_work_area(
            repo, task_id, branch=branch, author=identity,
        )
        record_git_receipt(
            conn, task_id, workspace_path=area, branch_name=branch,
            base_commit=base, head_commit=head,
        )
        record_run_advance(
            conn, task_id, branch=branch, base_commit=base, head_commit=head,
        )
        heads[label] = {"task": task_id, "head": head, "identity": identity}
    conn.close()

    for label, expected in heads.items():
        author = _commit_identity(repo, expected["head"])
        assert author == expected["identity"], label

    code, output, removal_id = _remove_permanently("one-rule")
    assert code == 0, output

    references = _receipt_references("one-rule", removal_id)
    for label, expected in heads.items():
        reference, = [r for r in references if r["task"] == expected["task"]]
        assert expected["head"] in reference["board_created_commits"], label
        assert base not in reference["board_created_commits"], label
        assert reference["commit_rule"] == kb.BOARD_CREATED_COMMIT_RULE, label


def test_a_reference_with_nothing_derivable_says_so_explicitly(
    fence_home, tmp_path
):
    """An unsupplied field is UNVERIFIED on the receipt, never silence."""
    repo = tmp_path / "repo"
    base = make_git_repo(repo)
    create_fenced_board("unverified")
    conn = kb.connect(board="unverified")
    task_id = ready_task(conn, title="a task with no head receipt")
    branch = f"hermes/{task_id}"
    area, _head = add_linked_work_area(repo, task_id, branch=branch)
    # A task with a base but NO head: nothing durable says what it produced.
    record_git_receipt(
        conn, task_id, workspace_path=area, branch_name=branch,
        base_commit=base, head_commit="",
    )
    conn.close()

    code, output, removal_id = _remove_permanently("unverified")
    assert code == 0, output

    reference, = [
        r for r in _receipt_references("unverified", removal_id)
        if r["task"] == task_id
    ]
    assert reference["head_commit"] == kb.RECEIPT_VERDICT_UNVERIFIED
    assert reference["board_created_commits"] == []
    assert reference["commit_count"] == 0
    # …and the §12.7 check says the same thing, with a verdict.
    receipt = kb.get_permanent_removal_receipt("unverified", removal_id)
    check, = [
        c for c in receipt["verification"]["checks"]
        if c["check"] == "version-control-retained-and-named"
    ]
    assert check["verdict"] == kb.RECEIPT_VERDICT_UNVERIFIED
    assert check["observed"]


# ---------------------------------------------------------------------------
# Several commits in ONE advance, and what the receipt may claim about them
# ---------------------------------------------------------------------------

def test_several_commits_in_one_advance_are_not_claimed_as_a_complete_list(
    fence_home, tmp_path
):
    """Three real commits, one recorded advance, one derivable head.

    A receipt that listed only the final head and said nothing more would
    read as a complete enumeration of what the board created. It is not:
    the intermediate commits of that same advance are real and are not on
    it. The receipt has to say which of the two it is.
    """
    setup = board_with_multi_commit_advance(
        "one-advance-many-commits", tmp_path / "repo", extra_commits=2,
    )
    assert len(setup["created"]) == len(setup["intermediate"]) + 1
    assert setup["intermediate"], "the fixture made no intermediate commit"
    # The intermediate commits really are in the repository's history —
    # this is a gap in what is DERIVABLE, not in what exists.
    history = git(
        setup["repo"], "rev-list", setup["head"], f"^{setup['base']}",
    ).split()
    assert set(setup["created"]) == set(history)

    code, output, removal_id = _remove_permanently("one-advance-many-commits")
    assert code == 0, output

    reference, = [
        r for r in _receipt_references("one-advance-many-commits", removal_id)
        if r["task"] == setup["task"]
    ]
    assert setup["head"] in reference["board_created_commits"]
    for commit in setup["intermediate"]:
        assert commit not in reference["board_created_commits"], commit
    completeness = reference["commit_list_completeness"]
    assert completeness["verdict"] == kb.RECEIPT_VERDICT_UNVERIFIED
    assert completeness["creation_time_provenance_ledger"] == "absent"
    assert completeness["reasons"], "the marking gives no reason"
    assert completeness["advance_count"] == 1
    assert reference["commit_list_source"]["source"] == kb.COMMIT_LIST_SOURCE


def test_the_receipt_names_the_source_of_the_commit_list(fence_home, tmp_path):
    """The source is stated, so a reader never has to assume a ledger."""
    setup = board_with_multi_commit_advance(
        "named-source", tmp_path / "repo", extra_commits=2,
    )

    code, output, removal_id = _remove_permanently("named-source")
    assert code == 0, output

    receipt = kb.get_permanent_removal_receipt("named-source", removal_id)
    version_control = receipt["version_control"]
    assert version_control["commit_list_source"] == kb.COMMIT_LIST_SOURCE
    assert version_control["commit_list_source_statement"]
    assert version_control["commit_list_completeness"]["verdict"] == (
        kb.RECEIPT_VERDICT_UNVERIFIED
    )
    reference, = [
        r for r in version_control["references"] if r["task"] == setup["task"]
    ]
    source = reference["commit_list_source"]
    assert source["source"] == kb.COMMIT_LIST_SOURCE
    assert source["creation_time_provenance_ledger"] == "absent"
    assert source["read_from"], "the receipt does not say what it read"


def test_a_reference_with_a_head_and_no_recorded_receipt_says_so(
    fence_home, tmp_path
):
    """A real head with no run receipt is UNVERIFIED with its own reason."""
    repo = tmp_path / "repo"
    base = make_git_repo(repo)
    create_fenced_board("head-no-receipt")
    conn = kb.connect(board="head-no-receipt")
    task_id = ready_task(conn, title="a head nothing recorded an advance for")
    branch = f"hermes/{task_id}"
    area, head = add_linked_work_area(repo, task_id, branch=branch)
    # A real head, recorded on the task — but NO run receipt at all.
    record_git_receipt(
        conn, task_id, workspace_path=area, branch_name=branch,
        base_commit=base, head_commit=head,
    )
    conn.close()

    code, output, removal_id = _remove_permanently("head-no-receipt")
    assert code == 0, output

    reference, = [
        r for r in _receipt_references("head-no-receipt", removal_id)
        if r["task"] == task_id
    ]
    assert reference["head_commit"] == head
    assert reference["advance_count"] == 0
    completeness = reference["commit_list_completeness"]
    assert completeness["verdict"] == kb.RECEIPT_VERDICT_UNVERIFIED
    assert completeness["head_recorded"] is True
    assert len(completeness["reasons"]) > 1, (
        "a head with no recorded receipt got no reason of its own"
    )


# ---------------------------------------------------------------------------
# A head absorbed from ANOTHER BOARD's work
# ---------------------------------------------------------------------------

def test_a_head_absorbed_from_another_board_is_disclosed_not_claimed(
    fence_home, tmp_path
):
    """Two boards, one shared repository, one head that crossed between them.

    The absorbing board's receipt discloses the other board's head by
    exact identity and does NOT put it on its own created list — the head
    belongs to the advance that produced it, whichever board that was.
    """
    repo = tmp_path / "shared"
    origin = board_with_multi_commit_advance(
        "origin-board", repo, extra_commits=1,
        title="the other board's own work",
    )
    absorbing = board_with_multi_commit_advance(
        "absorbing-board", repo, extra_commits=1,
        absorbed_heads=[
            {"task": origin["task"], "head_commit": origin["head"]}
        ],
        title="work that absorbed another board's head",
    )
    assert origin["head"] != absorbing["head"]

    code, output, removal_id = _remove_permanently("absorbing-board")
    assert code == 0, output

    references = _receipt_references("absorbing-board", removal_id)
    reference, = [r for r in references if r["task"] == absorbing["task"]]
    assert reference["absorbed_head_count"] == 1
    absorbed, = reference["absorbed_heads"]
    assert absorbed["head_commit"] == origin["head"]
    assert absorbed["absorbed_from_task"] == origin["task"]
    assert absorbed["disclosure"], "the absorbed head is disclosed silently"
    assert origin["head"] not in reference["board_created_commits"], (
        "another board's head was claimed as this board's own"
    )
    assert absorbing["head"] in reference["board_created_commits"]

    # The other board is untouched, and its own removal still claims its
    # own head — the head is claimed exactly once, by whoever made it.
    assert kb.board_dir("origin-board").exists()
    assert origin["work_area"].exists()
    origin_code, origin_output, origin_removal = _remove_permanently(
        "origin-board"
    )
    assert origin_code == 0, origin_output
    origin_reference, = [
        r for r in _receipt_references("origin-board", origin_removal)
        if r["task"] == origin["task"]
    ]
    assert origin["head"] in origin_reference["board_created_commits"]
    assert origin_reference["absorbed_heads"] == []


def test_an_absorbed_head_from_a_later_commit_series_is_still_excluded(
    fence_home, tmp_path
):
    """The exclusion is by identity, not by position in the history."""
    repo = tmp_path / "shared"
    origin = board_with_multi_commit_advance(
        "series-origin", repo, extra_commits=2,
    )
    # More real commits on the other board's branch, after its advance.
    later = commit_series(
        origin["work_area"], 2, prefix="later", author=FOREIGN_GIT_IDENTITY,
    )
    absorbing = board_with_multi_commit_advance(
        "series-absorbing", repo, extra_commits=1,
        absorbed_heads=[{"task": origin["task"], "head_commit": later[-1]}],
    )

    code, output, removal_id = _remove_permanently("series-absorbing")
    assert code == 0, output

    reference, = [
        r for r in _receipt_references("series-absorbing", removal_id)
        if r["task"] == absorbing["task"]
    ]
    assert [h["head_commit"] for h in reference["absorbed_heads"]] == [later[-1]]
    assert later[-1] not in reference["board_created_commits"]
    assert absorbing["head"] in reference["board_created_commits"]
    assert reference["commit_list_completeness"]["verdict"] == (
        kb.RECEIPT_VERDICT_UNVERIFIED
    )
