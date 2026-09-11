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
