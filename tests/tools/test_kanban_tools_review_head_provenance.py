"""The model-facing review-request tool parks the KERNEL's head, never a claim.

``kanban_request_review`` is one of the four surfaces that hand implementation
work to a reviewer, and — unlike the automatic handover inside
``complete_task`` — it arrives with nothing proven: a worker-written summary,
free-form ``metadata``, and whatever else the model chose to put in the tool
arguments. It used to park a scoped card with NO implementation head at all,
so the approval wrote ``head_commit`` NULL over work that had been made at an
exact commit and any downstream integration promise lapsed with it.

The kernel handover primitive (``kanban_db.request_review``) now derives that
head itself, from the task's own worktree, before the card is converted to the
reviewer's read-only scope. These tests drive the real tool handler against a
real git worktree to assert both halves of that:

* the head the approval finally writes is the commit that is actually in the
  worktree; and
* a head NAMED by the model — in tool arguments or in ``metadata`` prose —
  cannot become that provenance, however well-formed it looks.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest


REVIEWER = "raphael-verifier"
IMPLEMENTER = "test-worker"


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


@pytest.fixture
def repo(tmp_path) -> Path:
    origin = tmp_path / "repo"
    origin.mkdir()
    subprocess.run(
        ["git", "init", "-b", "main", str(origin)],
        check=True, capture_output=True, text=True,
    )
    (origin / "README.md").write_text("base\n", encoding="utf-8")
    _git(origin, "add", "README.md")
    _git(origin, "commit", "-m", "init")
    return origin


@pytest.fixture
def scoped_worker(tmp_path, monkeypatch, repo):
    """A dispatcher-spawned worker holding one real, scoped, committed card.

    Mirrors the worker environment the tool actually runs in
    (``HERMES_KANBAN_TASK`` / ``HERMES_KANBAN_RUN_ID`` naming its own run) and
    gives the card a per-task git worktree with one commit inside its declared
    ownership — the work the reviewer is about to be handed.
    """
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_PROFILE", IMPLEMENTER)
    monkeypatch.delenv("HERMES_SESSION_ID", raising=False)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    from hermes_cli import kanban_db as kb
    from plugins.dashboard_auth.raphael_workspace import model_policy

    monkeypatch.setattr(
        model_policy, "reviewer_profile_ids", lambda: (REVIEWER,), raising=True,
    )

    kb._INITIALIZED_PATHS.clear()
    kb.init_db()
    conn = kb.connect()
    tid = kb.create_task(
        conn,
        title="scoped work with a review requirement",
        assignee=IMPLEMENTER,
        requires_review=True,
        owned_paths=["src/impl"],
        workspace_kind="worktree",
        workspace_path=str(repo),
        branch_name="feature/tool-handover",
    )
    run = kb.claim_task(conn, tid, claimer=f"{IMPLEMENTER}:1")
    assert run is not None
    workspace, branch = kb._resolve_worktree_workspace(run)
    kb.set_workspace_path(conn, tid, workspace)
    kb.set_branch_name(conn, tid, branch)
    kb.record_worktree_base(conn, tid, workspace)
    (workspace / "src" / "impl").mkdir(parents=True, exist_ok=True)
    (workspace / "src" / "impl" / "feature.py").write_text(
        "ok = True\n", encoding="utf-8",
    )
    _git(workspace, "add", "src/impl/feature.py")
    _git(workspace, "commit", "-m", "feat: implement")
    real_head = _git(workspace, "rev-parse", "HEAD")

    monkeypatch.setenv("HERMES_KANBAN_TASK", tid)
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(run.current_run_id))
    try:
        yield kb, conn, tid, real_head
    finally:
        conn.close()


def test_the_tool_handover_persists_the_head_the_kernel_proved(scoped_worker):
    """The tool supplies no head; the approval still finishes at the commit."""
    from tools import kanban_tools as kt

    kb, conn, tid, real_head = scoped_worker
    out = json.loads(kt._handle_request_review({
        "summary": "implemented the slice and ran the unit tests",
    }))
    assert out.get("ok") is True, out
    assert out["status"] == "review"

    parked = kb.get_task(conn, tid)
    assert (parked.status, parked.assignee) == ("review", REVIEWER)
    assert parked.owned_paths == []
    assert kb._latest_review_head_provenance(conn, tid) == real_head

    review = kb.claim_review_task(conn, tid, claimer=f"{REVIEWER}:1")
    assert review is not None
    assert kb.submit_review_findings(
        conn, tid,
        findings=[],
        candidate_digest="digest-clean",
        expected_run_id=review.current_run_id,
    )["outcome"] == "passed"

    approved = kb.get_task(conn, tid)
    assert approved.status == "done"
    assert approved.head_commit == real_head
    assert approved.owned_paths == ["src/impl"]


def test_a_model_named_head_cannot_become_the_review_provenance(
    scoped_worker, repo,
):
    """Prose is not a receipt, in tool arguments or in ``metadata``.

    The named commit here is a REAL, well-formed object id — the repository's
    base commit — so nothing about its shape gives it away. It is simply not
    what this task produced, and only the kernel's own proof decides that.
    """
    from tools import kanban_tools as kt

    kb, conn, tid, real_head = scoped_worker
    stale_but_real = _git(repo, "rev-parse", "HEAD")
    assert stale_but_real != real_head

    out = json.loads(kt._handle_request_review({
        "summary": f"implemented at {stale_but_real}",
        "implementation_head": stale_but_real,
        "head_commit": stale_but_real,
        "metadata": {
            "implementation_head": stale_but_real,
            "head_commit": stale_but_real,
            "execution_receipt": {
                "kind": "scoped_worktree_v1",
                "head_commit": stale_but_real,
            },
        },
    }))
    assert out.get("ok") is True, out

    assert kb._latest_review_head_provenance(conn, tid) == real_head

    review = kb.claim_review_task(conn, tid, claimer=f"{REVIEWER}:1")
    assert review is not None
    assert kb.submit_review_findings(
        conn, tid,
        findings=[],
        candidate_digest="digest-clean",
        expected_run_id=review.current_run_id,
    )["outcome"] == "passed"

    approved = kb.get_task(conn, tid)
    assert approved.status == "done"
    assert approved.head_commit == real_head
    assert approved.head_commit != stale_but_real


def test_the_tool_refuses_a_handover_whose_file_scope_cannot_be_proven(
    scoped_worker,
):
    """Fail closed at the tool boundary too, with the kernel's diagnostic."""
    from tools import kanban_tools as kt

    kb, conn, tid, real_head = scoped_worker
    workspace = Path(kb.get_task(conn, tid).workspace_path)
    (workspace / "src" / "impl" / "leftover.py").write_text(
        "uncommitted = True\n", encoding="utf-8",
    )

    out = json.loads(kt._handle_request_review({
        "summary": "implemented the slice",
    }))
    assert out.get("ok") is not True
    assert "file scope" in json.dumps(out)

    unchanged = kb.get_task(conn, tid)
    assert unchanged.status == "running"
    assert unchanged.assignee == IMPLEMENTER
    assert unchanged.owned_paths == ["src/impl"]
