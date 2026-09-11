"""The reviewed implementation head survives the review round-trip.

A committed review requirement parks the implementer's handover instead of
completing it, and the parked card is converted to a READ-ONLY scope. Two
consequences of that park are asserted here, both against a REAL git worktree,
because both are only observable when an exact commit exists:

* the execution receipt the handover derived — the exact worktree head, proven
  clean and inside the declared ownership scope — is the only proof of WHICH
  commit was reviewed. The park has to persist it as durable provenance,
  because the reviewer's own run is read-only and therefore derives no receipt
  of its own: without it the approved card finishes with ``head_commit`` NULL,
  and a downstream task that must contain every exact parent head refuses the
  finished parent it depends on; and
* the parent-head containment the handover proved is a fact about the PARENTS
  AS THEY WERE. A parent that moves while the reviewer reads makes the parked
  head stale, so the approval re-checks containment and refuses rather than
  finishing a card whose integration promise no longer holds.

The absent-reviewer park is asserted alongside them: an owner-governed card
whose policy nominates no independent reviewer cannot be parked unassigned (its
route lock names the role that holds it), so the handover is REFUSED outright —
cleanly, with no exception and no writes — instead of raising out of the
role-transition authority with the run still open.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from plugins.dashboard_auth.raphael_workspace import model_policy


IMPLEMENTER = "worker"
GOVERNED_IMPLEMENTER = "default"
REVIEWER = "raphael-verifier"
OTHER_REVIEWER = "raphael-planner"


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb._INITIALIZED_PATHS.clear()
    kb.init_db()
    return home


def _git(cwd: Path, *args: str) -> str:
    result = subprocess.run(
        [
            "git", "-C", str(cwd),
            "-c", "user.name=Test User",
            "-c", "user.email=test@example.com",
            "-c", "commit.gpgsign=false",
            *args,
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(
        ["git", "init", "-b", "main", str(repo)],
        check=True, capture_output=True, text=True,
    )
    (repo / "README.md").write_text("base\n", encoding="utf-8")
    _git(repo, "add", "README.md")
    _git(repo, "commit", "-m", "init")
    return repo


def _events(conn, tid, kind) -> list:
    return [
        json.loads(r["payload"]) if r["payload"] else None
        for r in conn.execute(
            "SELECT payload FROM task_events WHERE task_id = ? AND kind = ? "
            "ORDER BY id",
            (tid, kind),
        ).fetchall()
    ]


def _nominate(monkeypatch, *profiles):
    """Make the model policy nominate exactly these reviewer roles."""
    monkeypatch.setattr(
        model_policy, "reviewer_profile_ids", lambda: tuple(profiles), raising=True,
    )


def _scoped_task(
    conn,
    repo: Path,
    *,
    title: str,
    branch: str,
    owned_paths,
    parents=(),
    requires_review: bool = False,
    integrates_parent_heads: bool = False,
    assignee: str = IMPLEMENTER,
) -> str:
    task_id = kb.create_task(
        conn,
        title=title,
        assignee=assignee,
        workspace_kind="worktree",
        workspace_path=str(repo),
        branch_name=branch,
        owned_paths=owned_paths,
        integrates_parent_heads=integrates_parent_heads,
        requires_review=requires_review,
        parents=list(parents),
    )
    conn.execute(
        "UPDATE tasks SET project_id = 'project-1' WHERE id = ?", (task_id,),
    )
    conn.commit()
    return task_id


def _materialize(conn, task_id: str, *, claimer: str) -> Path:
    """Claim the task and give it a real, per-task git worktree."""
    claimed = kb.claim_task(conn, task_id, claimer=claimer)
    assert claimed is not None
    workspace, branch = kb._resolve_worktree_workspace(claimed)
    kb.set_workspace_path(conn, task_id, workspace)
    kb.set_branch_name(conn, task_id, branch)
    kb.record_worktree_base(conn, task_id, workspace)
    return workspace


def _commit_file(workspace: Path, relative: str, content: str, message: str) -> str:
    path = workspace / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    _git(workspace, "add", relative)
    _git(workspace, "commit", "-m", message)
    return _git(workspace, "rev-parse", "HEAD")


# ---------------------------------------------------------------------------
# Finding 1: the reviewed head is durable provenance
# ---------------------------------------------------------------------------


def test_the_direct_review_request_derives_the_head_the_approval_keeps(
    kanban_home, tmp_path, monkeypatch,
):
    """The handover primitive proves the head itself, on every surface.

    ``kb.request_review`` is the ONE kernel handover entry point, and the three
    surfaces that are not ``complete_task`` — the ``kanban_request_review``
    tool, the CLI status-to-review change and the dashboard's review route —
    reach it exactly like this call does: no receipt, no head, nothing but the
    task id and a summary. It used to park them with EMPTY provenance, so the
    approval wrote ``head_commit`` NULL over work that had been proven at an
    exact commit and an integrating child then refused the parent it depends
    on. The head is the kernel's own now, derived from the task's worktree
    before the read-only conversion, whichever surface asks.
    """
    _nominate(monkeypatch, REVIEWER)
    repo = _repo(tmp_path)
    with kb.connect() as conn:
        impl = _scoped_task(
            conn, repo,
            title="implement, hand over directly",
            branch="feature/direct",
            owned_paths=["src/impl"],
            requires_review=True,
        )
        workspace = _materialize(conn, impl, claimer=f"{IMPLEMENTER}:1")
        reviewed_head = _commit_file(
            workspace, "src/impl/feature.py", "ok = True\n", "feat: implement",
        )

        run_id = kb.get_task(conn, impl).current_run_id
        ok, reason = kb.request_review(
            conn, impl,
            summary="ready for review",
            expected_run_id=run_id,
            with_reason=True,
        )
        assert (ok, reason) == (True, None)

        parked = kb.get_task(conn, impl)
        assert (parked.status, parked.assignee) == ("review", REVIEWER)
        assert parked.owned_paths == []
        assert kb._latest_review_head_provenance(conn, impl) == reviewed_head, (
            "a handover that supplies no head must still park the exact head "
            "the kernel proved from the worktree"
        )

        review = kb.claim_review_task(conn, impl, claimer=f"{REVIEWER}:1")
        assert review is not None
        assert kb.submit_review_findings(
            conn, impl,
            findings=[],
            candidate_digest="digest-clean",
            expected_run_id=review.current_run_id,
        )["outcome"] == "passed"

        approved = kb.get_task(conn, impl)
        assert approved.status == "done"
        assert approved.head_commit == reviewed_head
        assert approved.owned_paths == ["src/impl"]

        # …and the receipt is usable downstream, which is the whole point: a
        # child that must contain every exact parent head completes against it.
        child = _scoped_task(
            conn, repo,
            title="integrate the directly-handed-over slice",
            branch="feature/direct-integrate",
            owned_paths=["."],
            parents=[impl],
            integrates_parent_heads=True,
            assignee="builder",
        )
        child_workspace = _materialize(conn, child, claimer="builder:1")
        _git(child_workspace, "merge", "--no-edit", reviewed_head)
        assert kb.complete_task(conn, child, summary="integrated") is True
        run = kb.latest_run(conn, child)
        assert run.metadata["execution_receipt"]["parent_heads"] == [
            {"task_id": impl, "head_commit": reviewed_head},
        ]


def test_a_second_handover_parks_the_new_head_and_never_the_prior_cycles(
    kanban_home, tmp_path, monkeypatch,
):
    """Rework control: provenance is re-derived per cycle, not carried over.

    A head that is merely "whatever the last accepted handover recorded" would
    survive a rework round untouched, and the reviewer would approve commit #2
    while the board finished the card at commit #1. Each handover through the
    same public handler proves the worktree afresh, so the second cycle parks
    the second commit and the first can never reappear.
    """
    _nominate(monkeypatch, REVIEWER)
    repo = _repo(tmp_path)
    with kb.connect() as conn:
        impl = _scoped_task(
            conn, repo,
            title="implement, rework, hand over again",
            branch="feature/rework",
            owned_paths=["src/impl"],
            requires_review=True,
        )
        workspace = _materialize(conn, impl, claimer=f"{IMPLEMENTER}:1")
        first_head = _commit_file(
            workspace, "src/impl/feature.py", "ok = 1\n", "feat: first pass",
        )

        first_run = kb.get_task(conn, impl).current_run_id
        assert kb.request_review(
            conn, impl, summary="v1", expected_run_id=first_run,
        ) is True
        assert kb._latest_review_head_provenance(conn, impl) == first_head

        review = kb.claim_review_task(conn, impl, claimer=f"{REVIEWER}:1")
        assert review is not None
        ok, implementer = kb.request_changes(
            conn, impl, reason="extract the helper",
            expected_run_id=review.current_run_id,
        )
        assert (ok, implementer) == (True, IMPLEMENTER)

        # Round two: real rework, a real second commit, the same handler.
        rework = kb.claim_task(conn, impl, claimer=f"{IMPLEMENTER}:2")
        assert rework is not None
        second_head = _commit_file(
            workspace, "src/impl/feature.py", "ok = 2\n", "fix: address review",
        )
        assert second_head != first_head
        assert kb.request_review(
            conn, impl, summary="v2", expected_run_id=rework.current_run_id,
        ) is True

        assert kb._latest_review_head_provenance(conn, impl) == second_head, (
            "the second handover must park the commit it actually handed over"
        )

        second_review = kb.claim_review_task(conn, impl, claimer=f"{REVIEWER}:2")
        assert second_review is not None
        assert kb.submit_review_findings(
            conn, impl,
            findings=[],
            candidate_digest="digest-clean-2",
            expected_run_id=second_review.current_run_id,
        )["outcome"] == "passed"

        approved = kb.get_task(conn, impl)
        assert approved.status == "done"
        assert approved.head_commit == second_head
        assert approved.head_commit != first_head


def test_a_scoped_handover_with_an_unprovable_worktree_is_refused(
    kanban_home, tmp_path, monkeypatch,
):
    """Fail closed: no provable head, no park, no write.

    The card declares a write boundary, so the kernel must be able to prove
    what it produced. An uncommitted worktree cannot yield a receipt, and a
    review parked from it would carry no head into an approval that writes one
    — so the handover is refused with the scope diagnostic instead, and the
    card stays with its implementer holding its own scope.
    """
    _nominate(monkeypatch, REVIEWER)
    repo = _repo(tmp_path)
    with kb.connect() as conn:
        impl = _scoped_task(
            conn, repo,
            title="nothing committed yet",
            branch="feature/dirty",
            owned_paths=["src/impl"],
            requires_review=True,
        )
        workspace = _materialize(conn, impl, claimer=f"{IMPLEMENTER}:1")
        (workspace / "src" / "impl").mkdir(parents=True, exist_ok=True)
        (workspace / "src" / "impl" / "feature.py").write_text(
            "work_in_progress = True\n", encoding="utf-8",
        )

        run_id = kb.get_task(conn, impl).current_run_id
        ok, reason = kb.request_review(
            conn, impl, summary="ready for review",
            expected_run_id=run_id, with_reason=True,
        )
        assert ok is False
        assert reason is not None and "file scope" in reason

        unchanged = kb.get_task(conn, impl)
        assert unchanged.status == "running"
        assert unchanged.assignee == IMPLEMENTER
        assert unchanged.owned_paths == ["src/impl"]
        assert _events(conn, impl, "review_requested") == []

        # Not a blanket refusal: committing the work parks it immediately.
        committed_head = _commit_file(
            workspace, "src/impl/feature.py", "done = True\n", "feat: commit it",
        )
        assert kb.request_review(
            conn, impl, summary="ready for review", expected_run_id=run_id,
        ) is True
        assert kb._latest_review_head_provenance(conn, impl) == committed_head


def test_the_approved_card_keeps_the_exact_reviewed_implementation_head(
    kanban_home, tmp_path, monkeypatch,
):
    """The park persists the verified head; the approval consumes it."""
    _nominate(monkeypatch, REVIEWER)
    repo = _repo(tmp_path)
    with kb.connect() as conn:
        impl = _scoped_task(
            conn, repo,
            title="implement the slice",
            branch="feature/impl",
            owned_paths=["src/impl"],
            requires_review=True,
        )
        workspace = _materialize(conn, impl, claimer=f"{IMPLEMENTER}:1")
        reviewed_head = _commit_file(
            workspace, "src/impl/feature.py", "ok = True\n", "feat: implement",
        )

        handover_run = kb.get_task(conn, impl).current_run_id
        assert kb.complete_task(
            conn, impl, summary="ready for review", expected_run_id=handover_run,
        ) is True
        parked = kb.get_task(conn, impl)
        assert (parked.status, parked.assignee) == ("review", REVIEWER)
        assert parked.owned_paths == []

        review = kb.claim_review_task(conn, impl, claimer=f"{REVIEWER}:1")
        assert review is not None
        # The reviewer's own clean verdict — the production approval path.
        verdict = kb.submit_review_findings(
            conn, impl,
            findings=[],
            candidate_digest="digest-clean",
            expected_run_id=review.current_run_id,
        )
        assert verdict["outcome"] == "passed", verdict

        approved = kb.get_task(conn, impl)
        assert approved.status == "done"
        assert approved.head_commit == reviewed_head, (
            "the finished card must carry the EXACT implementation commit the "
            "reviewer approved"
        )
        assert approved.owned_paths == ["src/impl"]

        # And the receipt is usable downstream: a child that must contain every
        # exact parent head completes against it.
        child = _scoped_task(
            conn, repo,
            title="integrate the slice",
            branch="feature/integrate",
            owned_paths=["."],
            parents=[impl],
            integrates_parent_heads=True,
            assignee="builder",
        )
        child_workspace = _materialize(conn, child, claimer="builder:1")
        _git(child_workspace, "merge", "--no-edit", reviewed_head)
        assert kb.complete_task(conn, child, summary="integrated") is True
        run = kb.latest_run(conn, child)
        assert run.metadata["execution_receipt"]["parent_heads"] == [
            {"task_id": impl, "head_commit": reviewed_head},
        ]


def test_the_reviewed_head_survives_a_reviewer_swap_mid_review(
    kanban_home, tmp_path, monkeypatch,
):
    """A mid-review reviewer change carries the parked head with it.

    The swap supersedes the handover event both provenance readers consult, so
    a swap that carried only the scope would leave the approval with no
    reviewed commit to restore — the same NULL receipt, reached the long way.
    """
    _nominate(monkeypatch, OTHER_REVIEWER)
    repo = _repo(tmp_path)
    with kb.connect() as conn:
        impl = _scoped_task(
            conn, repo,
            title="implement then swap reviewers",
            branch="feature/swap",
            owned_paths=["src/impl"],
            requires_review=True,
        )
        workspace = _materialize(conn, impl, claimer=f"{IMPLEMENTER}:1")
        reviewed_head = _commit_file(
            workspace, "src/impl/feature.py", "ok = True\n", "feat: implement",
        )
        handover_run = kb.get_task(conn, impl).current_run_id
        assert kb.complete_task(
            conn, impl, summary="ready for review", expected_run_id=handover_run,
        ) is True
        assert kb.get_task(conn, impl).assignee == OTHER_REVIEWER

        # The policy moves and the card is re-routed to its new nominee.
        _nominate(monkeypatch, REVIEWER)
        assert kb.assign_task(conn, impl, REVIEWER) is True

        review = kb.claim_review_task(conn, impl, claimer=f"{REVIEWER}:1")
        assert review is not None
        assert kb.submit_review_findings(
            conn, impl,
            findings=[],
            candidate_digest="digest-clean",
            expected_run_id=review.current_run_id,
        )["outcome"] == "passed"

        approved = kb.get_task(conn, impl)
        assert approved.status == "done"
        assert approved.head_commit == reviewed_head
        assert approved.owned_paths == ["src/impl"]


def test_a_parent_head_that_drifts_during_the_review_refuses_the_approval(
    kanban_home, tmp_path, monkeypatch,
):
    """The integration promise is re-proved at the approval, not assumed."""
    _nominate(monkeypatch, REVIEWER)
    repo = _repo(tmp_path)
    with kb.connect() as conn:
        parent = _scoped_task(
            conn, repo,
            title="parent slice",
            branch="feature/parent",
            owned_paths=["src/parent"],
        )
        parent_workspace = _materialize(conn, parent, claimer=f"{IMPLEMENTER}:1")
        first_head = _commit_file(
            parent_workspace, "src/parent/first.py", "first = True\n", "feat: first",
        )
        assert kb.complete_task(conn, parent, summary="parent done") is True
        assert kb.get_task(conn, parent).head_commit == first_head

        integrator = _scoped_task(
            conn, repo,
            title="integrate under review",
            branch="feature/integrator",
            owned_paths=["."],
            parents=[parent],
            integrates_parent_heads=True,
            requires_review=True,
            assignee="builder",
        )
        integration_workspace = _materialize(conn, integrator, claimer="builder:1")
        _git(integration_workspace, "merge", "--no-edit", first_head)
        integrated_head = _git(integration_workspace, "rev-parse", "HEAD")

        handover_run = kb.get_task(conn, integrator).current_run_id
        assert kb.complete_task(
            conn, integrator, summary="ready for review",
            expected_run_id=handover_run,
        ) is True
        assert kb.get_task(conn, integrator).status == "review"

        # The dependency moves WHILE the reviewer reads: the parked head no
        # longer contains the parent's current exact receipt.
        drifted_head = _commit_file(
            parent_workspace, "src/parent/second.py", "second = True\n",
            "feat: second",
        )
        conn.execute(
            "UPDATE tasks SET head_commit = ? WHERE id = ?",
            (drifted_head, parent),
        )
        conn.commit()
        assert drifted_head != integrated_head

        review = kb.claim_review_task(conn, integrator, claimer=f"{REVIEWER}:1")
        assert review is not None
        # BOTH approval shapes are refused: the reviewer's own clean verdict…
        verdict = kb.submit_review_findings(
            conn, integrator,
            findings=[],
            candidate_digest="digest-clean",
            expected_run_id=review.current_run_id,
        )
        assert verdict["outcome"] == "error", verdict
        # …and a human approving the same card by hand.
        assert kb.complete_task(
            conn, integrator, summary="approved",
            expected_run_id=review.current_run_id,
        ) is False, (
            "an approval may not finish a card whose parent-head integration "
            "promise no longer holds"
        )

        refused = kb.get_task(conn, integrator)
        assert refused.status != "done"
        assert refused.completed_at is None
        assert refused.head_commit is None
        assert _events(conn, integrator, "completed") == []

        # The refusal is about the DRIFT, not about approvals: re-integrating
        # the parent's current head and handing over again approves cleanly.
        ok, implementer = kb.request_changes(
            conn, integrator, reason="parent head moved; re-integrate",
            expected_run_id=review.current_run_id,
        )
        assert (ok, implementer) == (True, "builder")
        rework = kb.claim_task(conn, integrator, claimer="builder:2")
        assert rework is not None
        _git(integration_workspace, "merge", "--no-edit", drifted_head)
        reintegrated_head = _git(integration_workspace, "rev-parse", "HEAD")
        assert kb.complete_task(
            conn, integrator, summary="re-integrated",
            expected_run_id=rework.current_run_id,
        ) is True
        second_review = kb.claim_review_task(conn, integrator, claimer=f"{REVIEWER}:2")
        assert second_review is not None
        assert kb.submit_review_findings(
            conn, integrator,
            findings=[],
            candidate_digest="digest-clean-2",
            expected_run_id=second_review.current_run_id,
        )["outcome"] == "passed"
        approved = kb.get_task(conn, integrator)
        assert approved.status == "done"
        assert approved.head_commit == reintegrated_head


def test_a_parent_head_that_moves_after_the_proof_still_refuses_the_approval(
    kanban_home, tmp_path, monkeypatch,
):
    """The re-proof must hold at the COMMIT, not merely at the check.

    The containment walk shells out to git, so it runs before the approval's
    write transaction opens — which leaves a window. A parent that reopens, is
    reclaimed, recommits and completes inside that window is ``done`` again by
    the time the transaction starts, so "are the parents satisfied" is happy
    and the child would finish holding a head the parent no longer contains.
    The proof therefore hands back the exact receipt set it proved and the
    transaction that writes ``done`` re-reads it and requires equality.

    The window is opened deterministically here: the real resolver runs, and
    only then — with no transaction yet open — the parent's recorded head moves
    to a different REAL commit while the parent stays ``done``.
    """
    _nominate(monkeypatch, REVIEWER)
    repo = _repo(tmp_path)
    with kb.connect() as conn:
        parent = _scoped_task(
            conn, repo,
            title="parent slice",
            branch="feature/window-parent",
            owned_paths=["src/parent"],
        )
        parent_workspace = _materialize(conn, parent, claimer=f"{IMPLEMENTER}:1")
        first_head = _commit_file(
            parent_workspace, "src/parent/first.py", "first = True\n", "feat: first",
        )
        assert kb.complete_task(conn, parent, summary="parent done") is True

        integrator = _scoped_task(
            conn, repo,
            title="integrate under review",
            branch="feature/window-integrator",
            owned_paths=["."],
            parents=[parent],
            integrates_parent_heads=True,
            requires_review=True,
            assignee="builder",
        )
        integration_workspace = _materialize(conn, integrator, claimer="builder:1")
        _git(integration_workspace, "merge", "--no-edit", first_head)
        integrated_head = _git(integration_workspace, "rev-parse", "HEAD")

        handover_run = kb.get_task(conn, integrator).current_run_id
        assert kb.complete_task(
            conn, integrator, summary="ready for review",
            expected_run_id=handover_run,
        ) is True
        assert kb.get_task(conn, integrator).status == "review"

        # The parent's second commit is real work the child never merged.
        moved_head = _commit_file(
            parent_workspace, "src/parent/second.py", "second = True\n",
            "feat: second",
        )
        assert moved_head not in (first_head, integrated_head)

        real_resolver = kb._review_approval_head
        window = {"opened": 0}

        def _move_the_parent_after_the_proof(*args, **kwargs):
            """Let the proof succeed, then move the parent behind its back."""
            resolved = real_resolver(*args, **kwargs)
            window["opened"] += 1
            conn.execute(
                "UPDATE tasks SET head_commit = ? WHERE id = ?",
                (moved_head, parent),
            )
            conn.commit()
            return resolved

        monkeypatch.setattr(
            kb, "_review_approval_head", _move_the_parent_after_the_proof,
        )

        review = kb.claim_review_task(conn, integrator, claimer=f"{REVIEWER}:1")
        assert review is not None
        assert kb.complete_task(
            conn, integrator, summary="approved",
            expected_run_id=review.current_run_id,
        ) is False, (
            "an approval whose proven parent receipts changed before the "
            "done-update must be refused, not committed"
        )
        assert window["opened"] == 1, (
            "the interposition must actually have run inside the window"
        )
        # The parent is still done, so parent gating alone could not have
        # caught this — only the byte-for-byte re-read of the proven set.
        assert kb.get_task(conn, parent).status == "done"

        refused = kb.get_task(conn, integrator)
        assert refused.status != "done"
        assert refused.completed_at is None
        assert refused.head_commit is None
        # Still exactly where the reviewer left it: the read-only review run is
        # untouched, and no scope was restored.
        assert refused.current_run_id == review.current_run_id
        assert kb.run_claimed_from_review(
            conn, integrator, int(refused.current_run_id),
        ) is True
        assert refused.owned_paths == []
        assert refused.integrates_parent_heads is False
        assert _events(conn, integrator, "completed") == []

        # And with the window shut, the same approval is fine — provided the
        # parent has not actually moved.
        monkeypatch.setattr(kb, "_review_approval_head", real_resolver)
        conn.execute(
            "UPDATE tasks SET head_commit = ? WHERE id = ?", (first_head, parent),
        )
        conn.commit()
        assert kb.complete_task(
            conn, integrator, summary="approved",
            expected_run_id=review.current_run_id,
        ) is True
        approved = kb.get_task(conn, integrator)
        assert approved.status == "done"
        assert approved.head_commit == integrated_head
        assert approved.owned_paths == ["."]


def test_an_ordinary_completion_also_refuses_a_parent_that_moves_in_the_window(
    kanban_home, tmp_path, monkeypatch,
):
    """The same window, the same refusal, on the leg that has no review at all.

    A scoped completion proves containment inside its own execution receipt,
    which is derived before its write transaction for the same reason the
    approval's re-proof is: it shells out to git. So it has the identical
    window, and the identical guard — the proven receipt set is re-read under
    the write lock and must still match.
    """
    repo = _repo(tmp_path)
    with kb.connect() as conn:
        parent = _scoped_task(
            conn, repo,
            title="parent slice",
            branch="feature/plain-parent",
            owned_paths=["src/parent"],
        )
        parent_workspace = _materialize(conn, parent, claimer=f"{IMPLEMENTER}:1")
        first_head = _commit_file(
            parent_workspace, "src/parent/first.py", "first = True\n", "feat: first",
        )
        assert kb.complete_task(conn, parent, summary="parent done") is True

        child = _scoped_task(
            conn, repo,
            title="integrate without a review requirement",
            branch="feature/plain-child",
            owned_paths=["."],
            parents=[parent],
            integrates_parent_heads=True,
            assignee="builder",
        )
        child_workspace = _materialize(conn, child, claimer="builder:1")
        _git(child_workspace, "merge", "--no-edit", first_head)

        moved_head = _commit_file(
            parent_workspace, "src/parent/second.py", "second = True\n",
            "feat: second",
        )
        real_proof = kb._verify_scoped_worktree_completion
        window = {"opened": 0}

        def _move_the_parent_after_the_proof(*args, **kwargs):
            receipt = real_proof(*args, **kwargs)
            if args[1] == child:
                window["opened"] += 1
                conn.execute(
                    "UPDATE tasks SET head_commit = ? WHERE id = ?",
                    (moved_head, parent),
                )
                conn.commit()
            return receipt

        monkeypatch.setattr(
            kb, "_verify_scoped_worktree_completion",
            _move_the_parent_after_the_proof,
        )
        assert kb.complete_task(conn, child, summary="integrated") is False
        assert window["opened"] == 1
        assert kb.get_task(conn, parent).status == "done"

        refused = kb.get_task(conn, child)
        assert refused.status != "done"
        assert refused.head_commit is None
        assert _events(conn, child, "completed") == []

        # Merging what the parent actually holds now completes cleanly.
        monkeypatch.setattr(kb, "_verify_scoped_worktree_completion", real_proof)
        _git(child_workspace, "merge", "--no-edit", moved_head)
        assert kb.complete_task(conn, child, summary="re-integrated") is True
        assert kb.latest_run(conn, child).metadata["execution_receipt"][
            "parent_heads"
        ] == [{"task_id": parent, "head_commit": moved_head}]


# ---------------------------------------------------------------------------
# Finding 1b: the human approval is authorized from the parked provenance,
# never from a scope proof against the reviewer's EMPTY scope
#
# The park leaves the row holding ``owned_paths = []`` — the reviewer's
# read-only boundary — while the worktree still carries the implementation
# commits. Proving that worktree against that row asks "which of these commits
# are inside the empty set", so every implementation path reads as outside the
# declared ownership and the proof refuses the approval. The documented human
# approval (a completion made directly from ``review`` with no reviewer run
# claimed — the dashboard's status-to-done route) must therefore be classified
# and authorized BEFORE any live receipt is derived, from the provenance the
# park persisted.
#
# The regression scenario is followed by its two controls, so the pair
# "relaxed / not relaxed" differs in exactly one variable each time and the
# scope proof is proven to be still armed everywhere else.
# ---------------------------------------------------------------------------


def test_a_human_approval_from_the_review_lane_needs_no_live_scope_proof(
    kanban_home, tmp_path, monkeypatch,
):
    """THE REGRESSION: approving a parked card with no reviewer run claimed.

    This is the shape the committed-review-requirement contract asserts and
    the dashboard's status-to-done route uses: the card sits in ``review``, no
    reviewer run has been claimed, and a human says "done". Nothing on that row
    can prove the implementation — the park converted it to the reviewer's
    empty read-only scope on purpose — so the approval is authorized from the
    kernel-proven provenance the park persisted, and the empty-scope proof is
    not consulted at all.

    All four facts the approval owes the board land in ONE transaction, which
    is why they are read back from the committed row in a single ``SELECT`` and
    asserted as one tuple: the exact implementation head, the restored
    implementation ``owned_paths``, the restored ``integrates_parent_heads``
    flag, and ``done``.
    """
    _nominate(monkeypatch, REVIEWER)
    repo = _repo(tmp_path)
    with kb.connect() as conn:
        parent = _scoped_task(
            conn, repo,
            title="parent slice",
            branch="feature/human-approval-parent",
            owned_paths=["src/parent"],
        )
        parent_workspace = _materialize(conn, parent, claimer=f"{IMPLEMENTER}:1")
        parent_head = _commit_file(
            parent_workspace, "src/parent/first.py", "first = True\n", "feat: first",
        )
        assert kb.complete_task(conn, parent, summary="parent done") is True
        assert kb.get_task(conn, parent).head_commit == parent_head

        impl = _scoped_task(
            conn, repo,
            title="implement under review, approved by a human",
            branch="feature/human-approval",
            owned_paths=["."],
            parents=[parent],
            integrates_parent_heads=True,
            requires_review=True,
            assignee="builder",
        )
        workspace = _materialize(conn, impl, claimer="builder:1")
        _git(workspace, "merge", "--no-edit", parent_head)
        implementation_head = _commit_file(
            workspace, "src/app/main.py", "ok = True\n", "feat: implement",
        )

        handover_run = kb.get_task(conn, impl).current_run_id
        assert kb.complete_task(
            conn, impl, summary="ready for review", expected_run_id=handover_run,
        ) is True

        # The park this approval has to be able to end: the reviewer's empty
        # read-only scope on the row, the implementation provenance beside it,
        # and NO run claimed by anyone.
        parked = kb.get_task(conn, impl)
        assert parked.status == "review"
        assert parked.assignee == REVIEWER
        assert parked.owned_paths == []
        assert parked.integrates_parent_heads is False
        assert parked.current_run_id is None
        assert parked.head_commit is None
        assert kb._latest_review_head_provenance(conn, impl) == implementation_head
        assert kb._latest_review_scope_provenance(conn, impl) == {
            "owned_paths": ["."],
            "integrates_parent_heads": True,
        }

        # The approval itself: straight from ``review``, no reviewer run, no
        # run expectation — exactly what a human clicking "done" sends.
        assert kb.complete_task(
            conn, impl, summary="approved from the board",
        ) is True, (
            "a human approving a parked review must not be refused by a scope "
            "proof run against the reviewer's empty read-only scope"
        )

        row = conn.execute(
            "SELECT status, head_commit, owned_paths, integrates_parent_heads, "
            "completed_at FROM tasks WHERE id = ?",
            (impl,),
        ).fetchone()
        assert (
            row["status"],
            row["head_commit"],
            json.loads(row["owned_paths"]),
            row["integrates_parent_heads"],
        ) == ("done", implementation_head, ["."], 1), (
            "the head, the restored scope, the restored integration flag and "
            "``done`` must all land together"
        )
        assert row["completed_at"] is not None

        # One park, one completion, and the empty-scope proof never ran against
        # this call: a scope refusal would have left its own durable event.
        assert len(_events(conn, impl, "review_requested")) == 1
        assert len(_events(conn, impl, "completed")) == 1
        assert _events(conn, impl, "completion_blocked_file_scope") == []

        # The restored receipt is usable downstream, which is the point of
        # restoring it: a child that must contain every exact parent head
        # completes against the approved commit.
        child = _scoped_task(
            conn, repo,
            title="integrate the human-approved slice",
            branch="feature/human-approval-child",
            owned_paths=["."],
            parents=[impl],
            integrates_parent_heads=True,
            assignee="builder",
        )
        child_workspace = _materialize(conn, child, claimer="builder:2")
        _git(child_workspace, "merge", "--no-edit", implementation_head)
        assert kb.complete_task(conn, child, summary="integrated") is True
        assert kb.latest_run(conn, child).metadata["execution_receipt"][
            "parent_heads"
        ] == [{"task_id": impl, "head_commit": implementation_head}]


def test_a_human_approval_still_refuses_a_parked_head_that_lost_its_parent(
    kanban_home, tmp_path, monkeypatch,
):
    """The relaxed path drops the empty-scope proof and nothing else.

    Skipping a proof that can only ever answer "outside the empty set" is not
    the same as skipping the proofs that still mean something. The approval-time
    parent-head containment re-proof is exactly such a proof — containment was
    a claim about the parents AS THEY WERE at the park, and a parent is free to
    move while the card waits for a human — so it must still refuse the
    no-reviewer-run approval, and refuse it with no state change at all.
    """
    _nominate(monkeypatch, REVIEWER)
    repo = _repo(tmp_path)
    with kb.connect() as conn:
        parent = _scoped_task(
            conn, repo,
            title="parent slice",
            branch="feature/human-drift-parent",
            owned_paths=["src/parent"],
        )
        parent_workspace = _materialize(conn, parent, claimer=f"{IMPLEMENTER}:1")
        first_head = _commit_file(
            parent_workspace, "src/parent/first.py", "first = True\n", "feat: first",
        )
        assert kb.complete_task(conn, parent, summary="parent done") is True

        integrator = _scoped_task(
            conn, repo,
            title="integrate, then wait for a human",
            branch="feature/human-drift",
            owned_paths=["."],
            parents=[parent],
            integrates_parent_heads=True,
            requires_review=True,
            assignee="builder",
        )
        integration_workspace = _materialize(conn, integrator, claimer="builder:1")
        _git(integration_workspace, "merge", "--no-edit", first_head)
        integrated_head = _git(integration_workspace, "rev-parse", "HEAD")

        handover_run = kb.get_task(conn, integrator).current_run_id
        assert kb.complete_task(
            conn, integrator, summary="ready for review",
            expected_run_id=handover_run,
        ) is True
        assert kb.get_task(conn, integrator).current_run_id is None

        # The dependency moves while the card waits on the review lane.
        drifted_head = _commit_file(
            parent_workspace, "src/parent/second.py", "second = True\n",
            "feat: second",
        )
        conn.execute(
            "UPDATE tasks SET head_commit = ? WHERE id = ?",
            (drifted_head, parent),
        )
        conn.commit()
        assert drifted_head != integrated_head

        assert kb.complete_task(
            conn, integrator, summary="approved from the board",
        ) is False, (
            "a human approval may not finish a card whose parent-head "
            "integration promise has lapsed"
        )
        refused = kb.get_task(conn, integrator)
        assert refused.status == "review"
        assert refused.completed_at is None
        assert refused.head_commit is None
        assert refused.owned_paths == []
        assert refused.integrates_parent_heads is False
        assert _events(conn, integrator, "completed") == []

        # And the refusal is about the drift alone: with the parent back at the
        # head the parked commit actually contains, the same call approves.
        conn.execute(
            "UPDATE tasks SET head_commit = ? WHERE id = ?", (first_head, parent),
        )
        conn.commit()
        assert kb.complete_task(
            conn, integrator, summary="approved from the board",
        ) is True
        approved = kb.get_task(conn, integrator)
        assert approved.status == "done"
        assert approved.head_commit == integrated_head
        assert approved.owned_paths == ["."]
        assert approved.integrates_parent_heads is True


def test_control_a_a_reviewer_run_approval_is_unchanged(
    kanban_home, tmp_path, monkeypatch,
):
    """CONTROL A: the same card, approved from a genuine reviewer run.

    One variable apart from the regression scenario above — the reviewer
    actually claims the card out of the review lane, so the completion arrives
    from a run the kernel recognises as read-only. That leg was never broken
    and must stay exactly as it is: the read-only run derives no receipt, the
    parked head fills the gap, and the same four facts land together.
    """
    _nominate(monkeypatch, REVIEWER)
    repo = _repo(tmp_path)
    with kb.connect() as conn:
        parent = _scoped_task(
            conn, repo,
            title="parent slice",
            branch="feature/reviewer-run-parent",
            owned_paths=["src/parent"],
        )
        parent_workspace = _materialize(conn, parent, claimer=f"{IMPLEMENTER}:1")
        parent_head = _commit_file(
            parent_workspace, "src/parent/first.py", "first = True\n", "feat: first",
        )
        assert kb.complete_task(conn, parent, summary="parent done") is True

        impl = _scoped_task(
            conn, repo,
            title="implement under review, approved by its reviewer",
            branch="feature/reviewer-run",
            owned_paths=["."],
            parents=[parent],
            integrates_parent_heads=True,
            requires_review=True,
            assignee="builder",
        )
        workspace = _materialize(conn, impl, claimer="builder:1")
        _git(workspace, "merge", "--no-edit", parent_head)
        implementation_head = _commit_file(
            workspace, "src/app/main.py", "ok = True\n", "feat: implement",
        )
        handover_run = kb.get_task(conn, impl).current_run_id
        assert kb.complete_task(
            conn, impl, summary="ready for review", expected_run_id=handover_run,
        ) is True

        review = kb.claim_review_task(conn, impl, claimer=f"{REVIEWER}:1")
        assert review is not None
        # A run genuinely claimed OUT of review, which is what makes this the
        # other approval shape rather than the relaxed one.
        assert kb.run_claimed_from_review(
            conn, impl, int(review.current_run_id),
        ) is True
        assert kb.get_task(conn, impl).status == "running"

        assert kb.complete_task(
            conn, impl, summary="reviewed and approved",
            expected_run_id=review.current_run_id,
        ) is True

        row = conn.execute(
            "SELECT status, head_commit, owned_paths, integrates_parent_heads "
            "FROM tasks WHERE id = ?",
            (impl,),
        ).fetchone()
        assert (
            row["status"],
            row["head_commit"],
            json.loads(row["owned_paths"]),
            row["integrates_parent_heads"],
        ) == ("done", implementation_head, ["."], 1)
        assert _events(conn, impl, "completion_blocked_file_scope") == []
        assert len(_events(conn, impl, "completed")) == 1


def test_control_b_an_ordinary_scoped_completion_still_proves_its_scope(
    kanban_home, tmp_path,
):
    """CONTROL B: no review requirement, so nothing about the proof changes.

    This is the test that would catch a fix which "solved" the regression by
    disabling the scope proof. An ordinary scoped worker completion still
    derives its exact receipt from the worktree, and a commit that touches a
    path outside the declared ownership is still refused — with the same
    diagnostic, the same durable ``completion_blocked_file_scope`` event, and
    the card left running.
    """
    repo = _repo(tmp_path)
    with kb.connect() as conn:
        clean = _scoped_task(
            conn, repo,
            title="ordinary scoped work",
            branch="feature/plain-clean",
            owned_paths=["src/app"],
        )
        clean_workspace = _materialize(conn, clean, claimer=f"{IMPLEMENTER}:1")
        clean_base = kb.get_task(conn, clean).base_commit
        clean_head = _commit_file(
            clean_workspace, "src/app/main.py", "ok = True\n", "feat: in scope",
        )
        clean_run = kb.get_task(conn, clean).current_run_id
        assert kb.complete_task(
            conn, clean, summary="done", expected_run_id=clean_run,
        ) is True

        landed = kb.get_task(conn, clean)
        assert landed.status == "done"
        assert landed.head_commit == clean_head
        assert landed.owned_paths == ["src/app"]
        receipt = kb.latest_run(conn, clean).metadata["execution_receipt"]
        assert receipt["base_commit"] == clean_base
        assert receipt["head_commit"] == clean_head
        assert receipt["owned_paths"] == ["src/app"]
        assert receipt["changed_paths"] == ["src/app/main.py"]

        # …and the boundary is still enforced, on a card with no review
        # requirement anywhere near it.
        strays = _scoped_task(
            conn, repo,
            title="ordinary scoped work that strays",
            branch="feature/plain-stray",
            owned_paths=["src/app"],
        )
        stray_workspace = _materialize(conn, strays, claimer=f"{IMPLEMENTER}:2")
        _commit_file(
            stray_workspace, "src/app/main.py", "ok = True\n", "feat: in scope",
        )
        _commit_file(
            stray_workspace, "src/other/theirs.py", "mine = True\n",
            "feat: out of scope",
        )
        stray_run = kb.get_task(conn, strays).current_run_id
        with pytest.raises(kb.WorktreeScopeError, match="outside declared ownership"):
            kb.complete_task(
                conn, strays, summary="done", expected_run_id=stray_run,
            )
        refused = kb.get_task(conn, strays)
        assert refused.status == "running"
        assert refused.head_commit is None
        assert refused.completed_at is None
        assert _events(conn, strays, "completed") == []
        assert kb.list_events(conn, strays)[-1].kind == (
            "completion_blocked_file_scope"
        )


# ---------------------------------------------------------------------------
# Finding 2: the absent-reviewer park on owner-governed work
# ---------------------------------------------------------------------------


def _configured_role(profile):
    provider = "openai-codex" if profile == REVIEWER else "anthropic"
    return model_policy.assignment_for(profile, provider)


def _govern(conn, task_id: str, *, assignee: str = GOVERNED_IMPLEMENTER) -> None:
    """Make the row receipt-owned and pin it under a valid route lock."""
    route = model_policy.task_assignment_for(assignee, "anthropic", "routine")
    lock = kb.mint_policy_lock(
        assignee, route.provider, route.model, route.reasoning_effort, "routine",
    )
    conn.execute(
        "UPDATE tasks SET owner_receipt_bound = 1, execution_tier = 'routine', "
        "provider_override = ?, model_override = ?, reasoning_effort = ?, "
        "model_policy_lock = ? WHERE id = ?",
        (route.provider, route.model, route.reasoning_effort, lock, task_id),
    )
    conn.commit()
    row = conn.execute("SELECT * FROM tasks WHERE id = ?", (task_id,)).fetchone()
    assert kb.task_is_policy_governed(row) is True
    assert kb.task_policy_lock_error(row) is None


_GOVERNED_COLUMNS = (
    "status", "assignee", "current_run_id", "claim_lock", "owned_paths",
    "integrates_parent_heads", "head_commit", "completed_at", "result",
    "model_policy_lock", "model_override", "provider_override",
    "reasoning_effort", "execution_tier",
)


def test_a_governed_handover_with_no_reviewer_refuses_without_raising(
    kanban_home, monkeypatch,
):
    """No exception, no self-review, no writes — and no silent completion."""
    _nominate(monkeypatch)
    with kb.connect() as conn:
        tid = kb.create_task(
            conn,
            title="governed work",
            assignee=GOVERNED_IMPLEMENTER,
            requires_review=True,
        )
        _govern(conn, tid)
        run = kb.claim_task(conn, tid, claimer=f"{GOVERNED_IMPLEMENTER}:1")
        assert run is not None
        assert kb.policy_resolved_reviewer() is None

        before = conn.execute(
            "SELECT * FROM tasks WHERE id = ?", (tid,)
        ).fetchone()
        events_before = conn.execute(
            "SELECT COUNT(*) AS n FROM task_events WHERE task_id = ?", (tid,)
        ).fetchone()["n"]

        assert kb.complete_task(
            conn, tid, summary="ready for review",
            expected_run_id=run.current_run_id,
        ) is False, (
            "an owner-governed card that cannot be parked unassigned must be "
            "refused outright, not completed and not raised out of"
        )

        after = conn.execute("SELECT * FROM tasks WHERE id = ?", (tid,)).fetchone()
        for column in _GOVERNED_COLUMNS:
            assert after[column] == before[column], column
        assert conn.execute(
            "SELECT COUNT(*) AS n FROM task_events WHERE task_id = ?", (tid,)
        ).fetchone()["n"] == events_before
        assert _events(conn, tid, "review_requested") == []
        assert _events(conn, tid, "completed") == []

        # The card never reviews itself: it is not on the review lane at all,
        # so no reviewer run — least of all the implementer's — can be claimed.
        assert kb.claim_review_task(
            conn, tid, claimer=f"{GOVERNED_IMPLEMENTER}:2",
        ) is None
        assert kb.get_task(conn, tid).assignee == GOVERNED_IMPLEMENTER
        assert kb.latest_run(conn, tid).outcome is None

        # The refusal is not a blanket one: the very same handover parks the
        # moment the policy nominates an independent reviewer again.
        _nominate(monkeypatch, REVIEWER)
        monkeypatch.setattr(
            model_policy, "configured_assignment_for", _configured_role,
        )
        assert kb.complete_task(
            conn, tid, summary="ready for review",
            expected_run_id=run.current_run_id,
        ) is True
        parked = kb.get_task(conn, tid)
        assert (parked.status, parked.assignee) == ("review", REVIEWER)
