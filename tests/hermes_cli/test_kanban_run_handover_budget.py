"""Run handover on budget expiry + the gave-up breaker's
attachment-preservation guarantee.

A run that already delivered a finished handover (a ``.patch`` + a
``report.md`` attachment, both uploaded by the agent during the active run)
before its budget expires must be COMPLETED with those exact artifacts
instead of recorded ``timed_out`` and rebuilt. That holds for BOTH budgets
that can expire a run: the wall-clock one enforced by
``enforce_max_runtime``, and the ITERATION budget, whose terminal outcome the
turn finalizer records through ``_record_task_failure(outcome='timed_out')``.
Absent either artifact, the ordinary timeout behavior is unchanged
(regression guard). Separately, tripping the circuit breaker must never
discard attachments already stored for the task.
"""

from __future__ import annotations

import contextlib
import logging
import subprocess
import time
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb


def _git(cwd: Path, *args: str) -> str:
    result = subprocess.run(
        [
            "git", "-C", str(cwd),
            "-c", "user.name=Test User",
            "-c", "user.email=test@example.com",
            "-c", "commit.gpgsign=false",
            *args,
        ],
        check=True, capture_output=True, text=True,
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


def _materialize(conn, task_id: str) -> Path:
    claimed = kb.claim_task(conn, task_id)
    assert claimed is not None
    workspace, branch = kb._resolve_worktree_workspace(claimed)
    kb.set_workspace_path(conn, task_id, workspace)
    kb.set_branch_name(conn, task_id, branch)
    kb.record_worktree_base(conn, task_id, workspace)
    return workspace


def _new_file_patch(path: str, content: str) -> bytes:
    lines = content.splitlines(keepends=True)
    body = "".join(f"+{line}" for line in lines)
    return (
        f"diff --git a/{path} b/{path}\n"
        "new file mode 100644\n"
        "index 0000000..e69de29\n"
        "--- /dev/null\n"
        f"+++ b/{path}\n"
        f"@@ -0,0 +1,{len(lines)} @@\n"
        f"{body}"
    ).encode("utf-8")


def _backdate_run(conn, task_id: str, run_id: int, *, seconds_ago: int) -> None:
    old = int(time.time()) - seconds_ago
    with kb.write_txn(conn):
        conn.execute(
            "UPDATE tasks SET started_at = ? WHERE id = ?", (old, task_id),
        )
        conn.execute(
            "UPDATE task_runs SET started_at = ? WHERE id = ?", (old, run_id),
        )


def _collapse_attachment_timestamps_onto_the_run_start(conn, task_id, run_id):
    """Put the run's start and every attachment row on the SAME whole second.

    ``created_at`` really is whole-second, so this is not a contrivance: two
    artifacts produced in one dispatcher tick genuinely share a timestamp.
    Forcing the collision makes the "which run produced this?" question
    unanswerable from timestamps, which is the point.
    """
    started = int(
        conn.execute(
            "SELECT started_at FROM task_runs WHERE id = ?", (run_id,),
        ).fetchone()["started_at"]
    )
    with kb.write_txn(conn):
        conn.execute(
            "UPDATE task_attachments SET created_at = ? WHERE task_id = ?",
            (started, task_id),
        )


def _expire_runtime_budget_now(conn, task_id):
    """Make the active run's runtime budget already spent, without moving any
    clock — so the run start and the attachment rows keep their real
    same-second relationship."""
    with kb.write_txn(conn):
        conn.execute(
            "UPDATE tasks SET max_runtime_seconds = 0 WHERE id = ?", (task_id,),
        )


def _run_scoped_attached_events(conn, task_id, run_id) -> int:
    return int(
        conn.execute(
            "SELECT COUNT(*) AS n FROM task_events WHERE task_id = ? "
            "AND kind = 'attached' AND run_id = ?",
            (task_id, run_id),
        ).fetchone()["n"]
    )


def _events(conn, tid, kind=None):
    rows = conn.execute(
        "SELECT kind FROM task_events WHERE task_id = ? ORDER BY id", (tid,),
    ).fetchall()
    kinds = [r["kind"] for r in rows]
    return [k for k in kinds if kind is None or k == kind]


# ---------------------------------------------------------------------------
# 9. Handover completes when patch + report are already attached
# ---------------------------------------------------------------------------


def test_budget_expiry_with_patch_and_report_completes_with_artifacts(
    tmp_path, monkeypatch,
):
    repo = _repo(tmp_path)
    attachment_root = tmp_path / "attachments"
    monkeypatch.setenv("HERMES_KANBAN_ATTACHMENTS_ROOT", str(attachment_root))
    conn = kb.connect(tmp_path / "kanban.db")
    try:
        task_id = kb.create_task(
            conn, title="handover on budget", assignee="worker",
            workspace_kind="worktree", workspace_path=str(repo),
            branch_name="feature/handover", owned_paths=["src/owned"],
            max_runtime_seconds=60,
        )
        _materialize(conn, task_id)
        task = kb.get_task(conn, task_id)
        assert task is not None and task.current_run_id is not None
        run_id = task.current_run_id
        kb._set_worker_pid(conn, task_id, 555_001)

        patch_id = kb.store_attachment_bytes(
            conn, task_id, "sandbox.patch",
            _new_file_patch("src/owned/new.py", "value = 1\n"),
            content_type="text/x-diff", uploaded_by="agent",
            expected_run_id=run_id,
        )
        report_id = kb.store_attachment_bytes(
            conn, task_id, "report.md",
            b"## Summary\nAll tests pass; ready for handover.\n",
            content_type="text/markdown", uploaded_by="agent",
            expected_run_id=run_id,
        )

        _backdate_run(conn, task_id, run_id, seconds_ago=1000)

        killed_pids = []
        timed_out = kb.enforce_max_runtime(
            conn, signal_fn=lambda pid, sig: killed_pids.append((pid, sig)),
        )
        assert task_id not in timed_out
        # The worker is still asked to stop even though the outcome is a
        # completion, not a timeout.
        assert killed_pids and killed_pids[0][0] == 555_001

        completed = kb.get_task(conn, task_id)
        assert completed is not None
        assert completed.status == "done"
        assert completed.head_commit

        run = kb.latest_run(conn, task_id)
        assert run is not None
        assert run.outcome == "completed"
        assert run.metadata["worktree_materialization"]["patch_attachment_id"] == patch_id

        # Materialized through the SAME single path complete_task always
        # uses — the patch is really applied and committed.
        assert _git(
            repo, "show", f"{completed.head_commit}:src/owned/new.py"
        ) == "value = 1"

        assert _events(conn, task_id, kind="run_handover_completed")
        assert _events(conn, task_id, kind="timed_out") == []
        handover_event = [
            e for e in kb.list_events(conn, task_id)
            if e.kind == "run_handover_completed"
        ][0]
        assert handover_event.payload["patch_attachment_id"] == patch_id
        assert handover_event.payload["report_attachment_id"] == report_id

        # Both artifacts remain stored and readable.
        attachments = kb.list_attachments(conn, task_id)
        assert {a.id for a in attachments} == {patch_id, report_id}
        for att in attachments:
            kb.read_attachment_bytes(att)  # does not raise
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# 10. Regression guard: without both artifacts, it times out exactly as today
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("with_report", [False, True])
def test_budget_expiry_without_full_handover_still_times_out(
    tmp_path, monkeypatch, with_report,
):
    repo = _repo(tmp_path)
    attachment_root = tmp_path / "attachments"
    monkeypatch.setenv("HERMES_KANBAN_ATTACHMENTS_ROOT", str(attachment_root))
    conn = kb.connect(tmp_path / "kanban.db")
    try:
        task_id = kb.create_task(
            conn, title="no full handover", assignee="worker",
            workspace_kind="worktree", workspace_path=str(repo),
            branch_name="feature/no-handover", owned_paths=["src/owned"],
            max_runtime_seconds=60,
        )
        _materialize(conn, task_id)
        task = kb.get_task(conn, task_id)
        assert task is not None and task.current_run_id is not None
        run_id = task.current_run_id
        kb._set_worker_pid(conn, task_id, 555_002)

        # Only a report (no patch), or nothing at all — either way the
        # handover is incomplete.
        if with_report:
            kb.store_attachment_bytes(
                conn, task_id, "report.md", b"Still working.\n",
                content_type="text/markdown", uploaded_by="agent",
                expected_run_id=run_id,
            )

        _backdate_run(conn, task_id, run_id, seconds_ago=1000)

        timed_out = kb.enforce_max_runtime(conn, signal_fn=lambda *_: None)
        assert task_id in timed_out

        task_after = kb.get_task(conn, task_id)
        assert task_after is not None
        assert task_after.status == "ready"  # source phase restored
        assert task_after.head_commit is None

        run = kb.latest_run(conn, task_id)
        assert run is not None
        assert run.outcome == "timed_out"
        assert _events(conn, task_id, kind="timed_out")
        assert _events(conn, task_id, kind="run_handover_completed") == []
    finally:
        conn.close()


def test_run_handover_failure_falls_back_to_timeout(tmp_path, monkeypatch):
    """Out-of-scope patch -> materialization fails -> ordinary timeout, and
    the task is never lost."""
    repo = _repo(tmp_path)
    attachment_root = tmp_path / "attachments"
    monkeypatch.setenv("HERMES_KANBAN_ATTACHMENTS_ROOT", str(attachment_root))
    conn = kb.connect(tmp_path / "kanban.db")
    try:
        task_id = kb.create_task(
            conn, title="bad handover patch", assignee="worker",
            workspace_kind="worktree", workspace_path=str(repo),
            branch_name="feature/bad-handover", owned_paths=["src/owned"],
            max_runtime_seconds=60,
        )
        _materialize(conn, task_id)
        task = kb.get_task(conn, task_id)
        assert task is not None and task.current_run_id is not None
        run_id = task.current_run_id
        kb._set_worker_pid(conn, task_id, 555_003)

        # Patch touches a path OUTSIDE the declared ownership scope.
        kb.store_attachment_bytes(
            conn, task_id, "sandbox.patch",
            _new_file_patch("src/outside.py", "leak = True\n"),
            content_type="text/x-diff", uploaded_by="agent",
            expected_run_id=run_id,
        )
        kb.store_attachment_bytes(
            conn, task_id, "report.md", b"Done (allegedly).\n",
            content_type="text/markdown", uploaded_by="agent",
            expected_run_id=run_id,
        )

        _backdate_run(conn, task_id, run_id, seconds_ago=1000)

        timed_out = kb.enforce_max_runtime(conn, signal_fn=lambda *_: None)
        assert task_id in timed_out

        run = kb.latest_run(conn, task_id)
        assert run is not None
        assert run.outcome == "timed_out"
        assert _events(conn, task_id, kind="run_handover_failed")
        assert _events(conn, task_id, kind="run_handover_completed") == []
        failed_event = [
            e for e in kb.list_events(conn, task_id)
            if e.kind == "run_handover_failed"
        ][0]
        assert failed_event.payload.get("reason")
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# 10b. Artifact selection comes from the run-scoped ``attached`` receipts,
#      never from the stored filename or from a timestamp comparison
# ---------------------------------------------------------------------------


def test_current_run_report_under_a_collision_suffixed_name_is_found(
    tmp_path, monkeypatch,
):
    """A report the CURRENT run really delivered must be found even when the
    native store had to rename it around an existing file.

    The card already carries an operator-attached ``report.md``, so the
    agent's own ``report.md`` lands on disk as ``report (1).md``. The
    ``attached`` receipt still records ``requested_filename='report.md'``,
    which is what identifies the artifact.
    """
    repo = _repo(tmp_path)
    monkeypatch.setenv(
        "HERMES_KANBAN_ATTACHMENTS_ROOT", str(tmp_path / "attachments"),
    )
    conn = kb.connect(tmp_path / "kanban.db")
    try:
        task_id = kb.create_task(
            conn, title="collision-suffixed report", assignee="worker",
            workspace_kind="worktree", workspace_path=str(repo),
            branch_name="feature/collision", owned_paths=["src/owned"],
            max_runtime_seconds=60,
        )
        _materialize(conn, task_id)
        task = kb.get_task(conn, task_id)
        assert task is not None and task.current_run_id is not None
        run_id = task.current_run_id
        kb._set_worker_pid(conn, task_id, 555_004)

        # The operator's briefing, attached to the card by a human.
        kb.store_attachment_bytes(
            conn, task_id, "report.md", b"Operator briefing, not a handover.\n",
            content_type="text/markdown", uploaded_by="operator",
        )
        patch_id = kb.store_attachment_bytes(
            conn, task_id, "sandbox.patch",
            _new_file_patch("src/owned/new.py", "value = 2\n"),
            content_type="text/x-diff", uploaded_by="agent",
            expected_run_id=run_id,
        )
        report_id = kb.store_attachment_bytes(
            conn, task_id, "report.md",
            b"## Summary\nHandover report from the live run.\n",
            content_type="text/markdown", uploaded_by="agent",
            expected_run_id=run_id,
        )
        stored = {a.id: a.filename for a in kb.list_attachments(conn, task_id)}
        assert stored[report_id] == "report (1).md", (
            "the premise of this test is a real collision rename; got "
            f"{stored[report_id]!r}"
        )

        _backdate_run(conn, task_id, run_id, seconds_ago=1000)

        timed_out = kb.enforce_max_runtime(conn, signal_fn=lambda *_: None)
        assert task_id not in timed_out, (
            "a complete current-run handover was missed and the run was timed "
            "out for a rebuild"
        )

        completed = kb.get_task(conn, task_id)
        assert completed is not None
        assert completed.status == "done"
        assert completed.head_commit
        assert _git(
            repo, "show", f"{completed.head_commit}:src/owned/new.py"
        ) == "value = 2"

        handover = [
            e for e in kb.list_events(conn, task_id)
            if e.kind == "run_handover_completed"
        ]
        assert len(handover) == 1
        assert handover[0].payload["patch_attachment_id"] == patch_id
        assert handover[0].payload["report_attachment_id"] == report_id, (
            "the operator's report.md was materialized instead of the agent's "
            "collision-renamed handover report"
        )
        run = kb.latest_run(conn, task_id)
        assert run is not None
        assert "Handover report from the live run." in (run.summary or "")
    finally:
        conn.close()


def test_same_second_prior_run_artifacts_are_not_selected_for_a_new_run(
    tmp_path, monkeypatch,
):
    """A run with ZERO ``attached`` receipts has delivered NO handover.

    Its predecessor's patch + report are still on the card, and whole-second
    ``created_at`` cannot distinguish them from anything this run might have
    produced. Selecting them would complete the new run with another run's
    work.
    """
    repo = _repo(tmp_path)
    monkeypatch.setenv(
        "HERMES_KANBAN_ATTACHMENTS_ROOT", str(tmp_path / "attachments"),
    )
    conn = kb.connect(tmp_path / "kanban.db")
    try:
        task_id = kb.create_task(
            conn, title="prior run artifacts", assignee="worker",
            workspace_kind="worktree", workspace_path=str(repo),
            branch_name="feature/prior-run", owned_paths=["src/owned"],
            max_runtime_seconds=60,
        )
        _materialize(conn, task_id)
        first = kb.get_task(conn, task_id)
        assert first is not None and first.current_run_id is not None
        first_run = first.current_run_id

        prior_patch = kb.store_attachment_bytes(
            conn, task_id, "sandbox.patch",
            _new_file_patch("src/owned/stale.py", "stale = True\n"),
            content_type="text/x-diff", uploaded_by="agent",
            expected_run_id=first_run,
        )
        prior_report = kb.store_attachment_bytes(
            conn, task_id, "report.md", b"## Summary\nPrior run's report.\n",
            content_type="text/markdown", uploaded_by="agent",
            expected_run_id=first_run,
        )

        # The operator aborts that worker through the real reclaim path: the
        # run ends, the artifacts stay on the card.
        assert kb.reclaim_task(
            conn, task_id, reason="operator abort", signal_fn=lambda *_: None,
        ) is True

        second = kb.claim_task(conn, task_id)
        assert second is not None and second.current_run_id is not None
        second_run = second.current_run_id
        assert second_run != first_run
        kb._set_worker_pid(conn, task_id, 555_005)

        # The new run has produced nothing: no run-scoped attach receipts.
        assert _run_scoped_attached_events(conn, task_id, second_run) == 0
        assert _run_scoped_attached_events(conn, task_id, first_run) == 2

        _collapse_attachment_timestamps_onto_the_run_start(
            conn, task_id, second_run,
        )
        _expire_runtime_budget_now(conn, task_id)

        timed_out = kb.enforce_max_runtime(conn, signal_fn=lambda *_: None)
        assert task_id in timed_out, (
            "a run that delivered no handover was completed from another "
            "run's artifacts instead of timing out"
        )

        after = kb.get_task(conn, task_id)
        assert after is not None
        assert after.status == "ready"
        assert after.head_commit is None
        run = kb.latest_run(conn, task_id)
        assert run is not None
        assert run.outcome == "timed_out"
        assert _events(conn, task_id, kind="run_handover_completed") == []
        assert _events(conn, task_id, kind="run_handover_failed") == []

        # Nothing was consumed: both prior-run artifacts are still readable.
        assert {a.id for a in kb.list_attachments(conn, task_id)} == {
            prior_patch, prior_report,
        }
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# 10c. The ITERATION budget is the motivating case, and it exits through the
#      turn finalizer, not through enforce_max_runtime
# ---------------------------------------------------------------------------


@pytest.fixture
def default_board_home(tmp_path, monkeypatch):
    """A default board under a temp ``HERMES_HOME``.

    ``agent.turn_finalizer._record_kanban_budget_exhausted`` opens its own
    connection with ``kanban_db.connect()`` — no path, no board — so the test
    has to make THAT resolve to the temp store rather than hand a connection
    in. Nothing in the production path is redirected.
    """
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb._INITIALIZED_PATHS.clear()
    kb.init_db()
    return home


def _worker_env(monkeypatch, task_id: str, run_id) -> None:
    """The dispatcher's worker environment for one spawned run.

    ``kanban_db._default_spawn`` exports both of these for every worker it
    launches: the task the process owns, and the run it IS. The
    budget finalizer reads them to bind its terminal outcome to its own run,
    so a test driving that entry point has to look like a real worker.
    """
    monkeypatch.setenv("HERMES_KANBAN_TASK", task_id)
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(run_id))


def _task_with_finished_handover(conn, repo: Path, *, title: str, branch: str):
    """A claimed worktree task whose active run already delivered a patch and
    a report through the real attachment path."""
    task_id = kb.create_task(
        conn, title=title, assignee="worker",
        workspace_kind="worktree", workspace_path=str(repo),
        branch_name=branch, owned_paths=["src/owned"],
        max_runtime_seconds=3600,
    )
    _materialize(conn, task_id)
    task = kb.get_task(conn, task_id)
    assert task is not None and task.current_run_id is not None
    run_id = task.current_run_id
    kb._set_worker_pid(conn, task_id, 555_010)
    patch_id = kb.store_attachment_bytes(
        conn, task_id, "sandbox.patch",
        _new_file_patch("src/owned/handover.py", "delivered = True\n"),
        content_type="text/x-diff", uploaded_by="agent",
        expected_run_id=run_id,
    )
    report_id = kb.store_attachment_bytes(
        conn, task_id, "report.md",
        b"## Summary\nBudget ran out, but the work is done and attached.\n",
        content_type="text/markdown", uploaded_by="agent",
        expected_run_id=run_id,
    )
    return task_id, run_id, patch_id, report_id


def test_iteration_budget_exhaustion_completes_a_finished_handover(
    default_board_home, tmp_path, monkeypatch,
):
    """The motivating case: the ITERATION budget ran out, not the clock.

    That exit does not go through ``enforce_max_runtime`` at all — the turn
    finalizer records it via ``_record_task_failure(outcome='timed_out',
    release_claim=True, end_run=True)``. A run that has already attached a
    valid patch and report must be completed with them, not closed
    ``timed_out``, returned to ``ready`` with no ``head_commit``, and rebuilt.
    """
    from agent.turn_finalizer import _record_kanban_budget_exhausted

    repo = _repo(tmp_path)
    with contextlib.closing(kb.connect()) as conn:
        task_id, run_id, patch_id, report_id = _task_with_finished_handover(
            conn, repo, title="iteration budget handover",
            branch="feature/iteration-handover",
        )
    # A real dispatcher-spawned worker knows which run it is; the finalizer
    # binds its outcome to that run and records nothing without it.
    _worker_env(monkeypatch, task_id, run_id)

    # The real turn-finalizer entry point for both budget-exhausted branches
    # of ``finalize_turn``. It opens its own connection and swallows
    # exceptions, so the durable board state is the only evidence.
    _record_kanban_budget_exhausted(
        task_id, 200, 200, logging.getLogger("test.turn_finalizer"),
    )

    with contextlib.closing(kb.connect()) as conn:
        task = kb.get_task(conn, task_id)
        assert task is not None
        assert task.status == "done", (
            "the iteration-budget exit ignored an attached handover; task is "
            f"{task.status!r}"
        )
        assert task.head_commit, "completed with no git receipt"

        run = kb.latest_run(conn, task_id)
        assert run is not None
        assert run.outcome == "completed"
        assert run.metadata["worktree_materialization"][
            "patch_attachment_id"
        ] == patch_id
        assert "the work is done and attached" in (run.summary or "")

        assert _events(conn, task_id, kind="timed_out") == []
        assert _events(conn, task_id, kind="gave_up") == []
        handover = [
            e for e in kb.list_events(conn, task_id)
            if e.kind == "run_handover_completed"
        ]
        assert len(handover) == 1
        assert handover[0].payload["patch_attachment_id"] == patch_id
        assert handover[0].payload["report_attachment_id"] == report_id

    # Materialized through the same single path every completion uses.
    assert _git(
        repo, "show", f"{task.head_commit}:src/owned/handover.py"
    ) == "delivered = True"


def test_iteration_budget_exhaustion_without_a_handover_still_times_out(
    default_board_home, tmp_path, monkeypatch,
):
    """Regression guard for the same entry point: no handover, no change."""
    from agent.turn_finalizer import _record_kanban_budget_exhausted

    repo = _repo(tmp_path)
    with contextlib.closing(kb.connect()) as conn:
        task_id = kb.create_task(
            conn, title="iteration budget, nothing delivered",
            assignee="worker",
            workspace_kind="worktree", workspace_path=str(repo),
            branch_name="feature/iteration-nothing",
            owned_paths=["src/owned"], max_runtime_seconds=3600,
        )
        _materialize(conn, task_id)
        task = kb.get_task(conn, task_id)
        assert task is not None and task.current_run_id is not None
        run_id = task.current_run_id
        kb.store_attachment_bytes(
            conn, task_id, "report.md", b"Half-finished notes.\n",
            content_type="text/markdown", uploaded_by="agent",
            expected_run_id=run_id,
        )
    # Same real worker environment: this run is the one that timed out.
    _worker_env(monkeypatch, task_id, run_id)

    _record_kanban_budget_exhausted(
        task_id, 200, 200, logging.getLogger("test.turn_finalizer"),
    )

    with contextlib.closing(kb.connect()) as conn:
        task = kb.get_task(conn, task_id)
        assert task is not None
        assert task.status == "ready"
        assert task.head_commit is None
        run = kb.latest_run(conn, task_id)
        assert run is not None
        assert run.outcome == "timed_out"
        assert _events(conn, task_id, kind="timed_out")
        assert _events(conn, task_id, kind="run_handover_completed") == []


# ---------------------------------------------------------------------------
# 10d. The iteration-budget exit belongs to ONE run: the worker's own.
#      A superseded worker whose budget ends late must not act on whichever
#      later run now owns the task.
# ---------------------------------------------------------------------------


def _reclaimed_run_then_a_live_successor(conn, repo: Path):
    """Run 1 is reclaimed while its worker is still alive; run 2 then claims
    the task and delivers ITS OWN patch and report.

    Returns ``(task_id, stale_run, live_run, live_patch, live_report)``.
    """
    task_id = kb.create_task(
        conn, title="superseded worker, late budget exit", assignee="worker",
        workspace_kind="worktree", workspace_path=str(repo),
        branch_name="feature/superseded", owned_paths=["src/owned"],
        max_runtime_seconds=3600,
    )
    _materialize(conn, task_id)
    first = kb.get_task(conn, task_id)
    assert first is not None and first.current_run_id is not None
    stale_run = int(first.current_run_id)

    # The real reclaim path: run 1 ends, but its worker process keeps going
    # and will exhaust its iteration budget some time later.
    assert kb.reclaim_task(
        conn, task_id, reason="operator abort", signal_fn=lambda *_: None,
    ) is True

    _materialize(conn, task_id)
    second = kb.get_task(conn, task_id)
    assert second is not None and second.current_run_id is not None
    live_run = int(second.current_run_id)
    assert live_run != stale_run
    kb._set_worker_pid(conn, task_id, 555_011)

    live_patch = kb.store_attachment_bytes(
        conn, task_id, "successor.patch",
        _new_file_patch("src/owned/successor.py", "successor = True\n"),
        content_type="text/x-diff", uploaded_by="agent",
        expected_run_id=live_run,
    )
    live_report = kb.store_attachment_bytes(
        conn, task_id, "report.md",
        b"## Summary\nRun two's own report, not run one's.\n",
        content_type="text/markdown", uploaded_by="agent",
        expected_run_id=live_run,
    )
    return task_id, stale_run, live_run, live_patch, live_report


def test_superseded_worker_budget_exit_does_not_touch_the_successor_run(
    default_board_home, tmp_path, monkeypatch,
):
    """The stale worker's environment names run 1; run 2 owns the task.

    Driving the real finalizer must write NOTHING: it must not complete run 2
    from run 2's own handover, must not time run 2 out, must not bump the
    failure counter, and must not close run 2 or move the task's status.
    """
    from agent.turn_finalizer import _record_kanban_budget_exhausted

    repo = _repo(tmp_path)
    with contextlib.closing(kb.connect()) as conn:
        (
            task_id, stale_run, live_run, live_patch, live_report,
        ) = _reclaimed_run_then_a_live_successor(conn, repo)

    # The superseded process is still pinned to the run it was spawned for.
    _worker_env(monkeypatch, task_id, stale_run)

    _record_kanban_budget_exhausted(
        task_id, 200, 200, logging.getLogger("test.turn_finalizer"),
    )

    with contextlib.closing(kb.connect()) as conn:
        task = kb.get_task(conn, task_id)
        assert task is not None
        assert task.status == "running", (
            "a superseded worker's budget exit moved the task out from under "
            f"the run that owns it; status is {task.status!r}"
        )
        assert task.head_commit is None, (
            "the successor run was completed by another run's finalizer"
        )
        assert task.current_run_id == live_run
        assert task.worker_pid == 555_011, "the successor's claim was released"
        assert task.consecutive_failures == 0, (
            "another run's timeout was counted against this task"
        )
        assert task.last_failure_error is None
        assert task.completed_at is None

        runs = {r.id: r for r in kb.list_runs(conn, task_id)}
        live = runs[live_run]
        assert live.status == "running"
        assert live.outcome is None
        assert live.ended_at is None
        assert live.summary is None
        assert live.error is None
        assert not (live.metadata or {}).get("worktree_materialization")

        stale = runs[stale_run]
        assert stale.status == "reclaimed", (
            "the stale run's own closure was rewritten; status is "
            f"{stale.status!r}"
        )
        assert stale.outcome == "reclaimed"

        assert _events(conn, task_id, kind="run_handover_completed") == []
        assert _events(conn, task_id, kind="run_handover_failed") == []
        assert _events(conn, task_id, kind="timed_out") == []
        assert _events(conn, task_id, kind="gave_up") == []

        # Run 2's own attachments are untouched and unconsumed — still there
        # for run 2 to hand over when ITS budget ends.
        assert {a.id for a in kb.list_attachments(conn, task_id)} == {
            live_patch, live_report,
        }
        assert kb.read_attachment_bytes(
            kb.get_attachment(conn, live_report)
        ) == b"## Summary\nRun two's own report, not run one's.\n"

    # Nothing was materialized into git under any ref.
    assert "src/owned/successor.py" not in _git(
        repo, "log", "--all", "--name-only", "--pretty=format:",
    )


def test_budget_exit_completes_when_the_env_names_the_still_current_run(
    default_board_home, tmp_path, monkeypatch,
):
    """The other half of the binding: when the worker's env names the run
    that IS the task's current open run, the handover completes exactly as
    before — the guard must not turn the good path into a no-op.

    Same board shape as the superseded case (a reclaimed run 1 is still on
    the card), so the only difference is which run the environment names.
    """
    from agent.turn_finalizer import _record_kanban_budget_exhausted

    repo = _repo(tmp_path)
    with contextlib.closing(kb.connect()) as conn:
        (
            task_id, stale_run, live_run, live_patch, live_report,
        ) = _reclaimed_run_then_a_live_successor(conn, repo)

    _worker_env(monkeypatch, task_id, live_run)

    _record_kanban_budget_exhausted(
        task_id, 200, 200, logging.getLogger("test.turn_finalizer"),
    )

    with contextlib.closing(kb.connect()) as conn:
        task = kb.get_task(conn, task_id)
        assert task is not None
        assert task.status == "done", (
            "the run's own handover was not completed; task is "
            f"{task.status!r}"
        )
        assert task.head_commit, "completed with no git receipt"
        head_commit = task.head_commit

        runs = {r.id: r for r in kb.list_runs(conn, task_id)}
        live = runs[live_run]
        assert live.outcome == "completed"
        assert live.metadata["worktree_materialization"][
            "patch_attachment_id"
        ] == live_patch
        assert "Run two's own report" in (live.summary or "")
        assert runs[stale_run].outcome == "reclaimed"

        handover = [
            e for e in kb.list_events(conn, task_id)
            if e.kind == "run_handover_completed"
        ]
        assert len(handover) == 1
        assert handover[0].run_id == live_run
        assert handover[0].payload["patch_attachment_id"] == live_patch
        assert handover[0].payload["report_attachment_id"] == live_report
        assert _events(conn, task_id, kind="run_handover_failed") == []
        assert _events(conn, task_id, kind="timed_out") == []
        assert _events(conn, task_id, kind="gave_up") == []

    assert _git(
        repo, "show", f"{head_commit}:src/owned/successor.py",
    ) == "successor = True"


@pytest.mark.parametrize(
    "identity", ["absent-run-id", "non-integer-run-id", "another-task"],
)
def test_budget_exit_without_a_valid_run_identity_records_nothing(
    default_board_home, tmp_path, monkeypatch, identity,
):
    """Fail closed on an unusable identity, rather than falling back to
    whatever run currently owns the task.

    Every one of these environments belongs to a process that cannot prove
    which run it is: the dispatcher exports ``HERMES_KANBAN_RUN_ID`` for
    every worker it launches, so an absent, unparseable or foreign-task
    value is never a legitimate worker on this task. The handover here is
    complete, so an unbound exit would happily complete the task with it.
    """
    from agent.turn_finalizer import _record_kanban_budget_exhausted

    repo = _repo(tmp_path)
    with contextlib.closing(kb.connect()) as conn:
        task_id, run_id, patch_id, report_id = _task_with_finished_handover(
            conn, repo, title=f"unusable identity: {identity}",
            branch=f"feature/{identity}",
        )
        other_task_id = kb.create_task(
            conn, title="somebody else's task", assignee="worker",
        )

    _worker_env(monkeypatch, task_id, run_id)
    if identity == "absent-run-id":
        monkeypatch.delenv("HERMES_KANBAN_RUN_ID")
    elif identity == "non-integer-run-id":
        monkeypatch.setenv("HERMES_KANBAN_RUN_ID", "run-7")
    else:
        monkeypatch.setenv("HERMES_KANBAN_TASK", other_task_id)

    _record_kanban_budget_exhausted(
        task_id, 200, 200, logging.getLogger("test.turn_finalizer"),
    )

    with contextlib.closing(kb.connect()) as conn:
        task = kb.get_task(conn, task_id)
        assert task is not None
        assert task.status == "running", (
            "a process with no valid run identity recorded an outcome; task "
            f"is {task.status!r}"
        )
        assert task.head_commit is None
        assert task.current_run_id == run_id
        assert task.worker_pid == 555_010
        assert task.consecutive_failures == 0
        assert task.last_failure_error is None
        assert task.completed_at is None

        run = kb.latest_run(conn, task_id)
        assert run is not None
        assert run.id == run_id
        assert run.status == "running"
        assert run.outcome is None
        assert run.ended_at is None
        assert run.summary is None

        assert _events(conn, task_id, kind="run_handover_completed") == []
        assert _events(conn, task_id, kind="run_handover_failed") == []
        assert _events(conn, task_id, kind="timed_out") == []
        assert _events(conn, task_id, kind="gave_up") == []

        # The task the environment mis-named never heard about it either.
        assert _events(conn, other_task_id, kind="timed_out") == []
        assert _events(conn, other_task_id, kind="gave_up") == []

        assert {a.id for a in kb.list_attachments(conn, task_id)} == {
            patch_id, report_id,
        }

    assert "src/owned/handover.py" not in _git(
        repo, "log", "--all", "--name-only", "--pretty=format:",
    )


# ---------------------------------------------------------------------------
# 10e. The takeover can also land AFTER the finalizer validated its run and
#      BEFORE its completion runs. Binding the state writes to one run is not
#      enough there: the rejection must be SILENT, because a durable event is
#      just as much a write onto a card a successor now owns.
# ---------------------------------------------------------------------------


def test_takeover_inside_the_bound_handover_window_writes_no_event(
    default_board_home, tmp_path, monkeypatch,
):
    """The window the run binding does NOT close on its own.

    ``_record_task_failure`` validates the worker's run up front, but the
    completion that follows takes real time (it reads the report, applies the
    patch, runs git). A reclaim + re-claim landing inside that window makes
    the rejection correct and unavoidable — and today it is also LOUD: the
    superseded worker leaves a ``completion_blocked_file_scope`` event and a
    ``run_handover_failed`` event attributed to its own dead run on a card the
    successor is actively working, so the successor's card reads as if its
    handover and its file scope had failed.

    The barrier is installed on the module attribute the production path
    really calls (``kanban_db.read_attachment_bytes``, read between finding
    the handover and calling ``complete_task``) and the takeover runs on a
    SECOND, independent connection — the successor is a different process.
    """
    from agent.turn_finalizer import _record_kanban_budget_exhausted

    repo = _repo(tmp_path)
    with contextlib.closing(kb.connect()) as conn:
        task_id, stale_run, patch_id, report_id = _task_with_finished_handover(
            conn, repo, title="takeover inside the handover window",
            branch="feature/handover-window",
        )

    # The worker is the run it was spawned for, and that run really is the
    # task's current open run when the finalizer starts.
    _worker_env(monkeypatch, task_id, stale_run)

    barrier: dict = {"fired": False, "live_run": None}
    real_read_attachment_bytes = kb.read_attachment_bytes

    def takeover_then_read(attachment):
        if not barrier["fired"]:
            barrier["fired"] = True
            with contextlib.closing(kb.connect()) as other:
                assert kb.reclaim_task(
                    other, task_id, reason="operator abort",
                    signal_fn=lambda *_: None,
                ) is True
                successor = kb.claim_task(other, task_id)
                assert successor is not None
                assert successor.current_run_id is not None
                barrier["live_run"] = int(successor.current_run_id)
                kb._set_worker_pid(other, task_id, 555_012)
        return real_read_attachment_bytes(attachment)

    monkeypatch.setattr(kb, "read_attachment_bytes", takeover_then_read)

    _record_kanban_budget_exhausted(
        task_id, 200, 200, logging.getLogger("test.turn_finalizer"),
    )

    assert barrier["fired"], (
        "the barrier never fired, so no takeover happened inside the handover "
        "window and this test proved nothing"
    )
    live_run = barrier["live_run"]
    assert live_run is not None and live_run != stale_run

    with contextlib.closing(kb.connect()) as conn:
        # Every kind at once, so a failure names every event that leaked
        # rather than only the first.
        leaked = {
            kind: _events(conn, task_id, kind=kind)
            for kind in (
                "completed", "run_handover_completed", "run_handover_failed",
                "completion_blocked_file_scope", "timed_out", "gave_up",
            )
        }
        assert leaked == {kind: [] for kind in leaked}, (
            "a superseded worker wrote durable events onto a task the "
            "successor run owns"
        )

        task = kb.get_task(conn, task_id)
        assert task is not None
        assert task.status == "running"
        assert task.current_run_id == live_run
        assert task.worker_pid == 555_012, "the successor's claim was released"
        assert task.consecutive_failures == 0
        assert task.last_failure_error is None
        assert task.completed_at is None
        assert task.head_commit is None

        runs = {r.id: r for r in kb.list_runs(conn, task_id)}
        live = runs[live_run]
        assert live.status == "running"
        assert live.outcome is None
        assert live.ended_at is None
        assert live.summary is None
        assert live.error is None
        assert not (live.metadata or {}).get("worktree_materialization")

        stale = runs[stale_run]
        assert stale.status == "reclaimed", (
            "the stale run's own closure was rewritten; status is "
            f"{stale.status!r}"
        )
        assert stale.outcome == "reclaimed"

        # Nothing was consumed either: both artifacts are still on the card.
        assert {a.id for a in kb.list_attachments(conn, task_id)} == {
            patch_id, report_id,
        }

    # Nothing was materialized into git under any ref.
    assert "src/owned/handover.py" not in _git(
        repo, "log", "--all", "--name-only", "--pretty=format:",
    )


def test_takeover_after_materialization_leaves_the_successor_commit_intact(
    default_board_home, tmp_path, monkeypatch,
):
    """A takeover landing AFTER the stale run has materialized and verified
    its patch but BEFORE the final run compare-and-swap.

    The stale completion is rejected by that compare-and-swap and rolls its
    materialization back. Before this fix the rollback was an unconditional
    ``git reset --hard`` to the head this call started from, which erased the
    commit the successor had made on the shared worktree in the meantime,
    silently: no event, no state change, just lost work. The rollback now
    leaves a worktree alone once it has moved past the head this call
    produced, so the successor's exact head and content survive (the stale
    call's own materialized commit stays underneath it, as the successor
    built on it).

    The barrier sits on the module attribute the completion path really
    calls between materialization and the compare-and-swap
    (``kanban_db._verify_scoped_worktree_completion``); the takeover and the
    successor's commit happen on a SECOND, independent connection.
    """
    from agent.turn_finalizer import _record_kanban_budget_exhausted

    repo = _repo(tmp_path)
    with contextlib.closing(kb.connect()) as conn:
        task_id, stale_run, patch_id, report_id = _task_with_finished_handover(
            conn, repo, title="takeover after materialization",
            branch="feature/handover-post-materialization",
        )

    _worker_env(monkeypatch, task_id, stale_run)

    barrier: dict = {"fired": False, "live_run": None, "successor_head": None}
    real_verify = kb._verify_scoped_worktree_completion

    def verify_then_takeover(conn_, tid):
        receipt = real_verify(conn_, tid)
        if not barrier["fired"]:
            barrier["fired"] = True
            with contextlib.closing(kb.connect()) as other:
                assert kb.reclaim_task(
                    other, tid, reason="operator abort",
                    signal_fn=lambda *_: None,
                ) is True
                successor = kb.claim_task(other, tid)
                assert successor is not None
                assert successor.current_run_id is not None
                assert successor.workspace_path
                barrier["live_run"] = int(successor.current_run_id)
                kb._set_worker_pid(other, tid, 555_013)
            # The successor advances the SHARED task worktree (the isolated
            # worktree the kernel materialized into, not the test repo's main
            # checkout) with its own commit.
            worktree = Path(successor.workspace_path)
            (worktree / "src" / "owned").mkdir(parents=True, exist_ok=True)
            (worktree / "src" / "owned" / "successor.py").write_text(
                "successor = True\n", encoding="utf-8",
            )
            _git(worktree, "add", "src/owned/successor.py")
            _git(worktree, "commit", "-m", "successor work")
            barrier["successor_head"] = _git(worktree, "rev-parse", "HEAD")
            barrier["worktree"] = worktree
        return receipt

    monkeypatch.setattr(
        kb, "_verify_scoped_worktree_completion", verify_then_takeover,
    )

    _record_kanban_budget_exhausted(
        task_id, 200, 200, logging.getLogger("test.turn_finalizer"),
    )

    assert barrier["fired"], (
        "the barrier never fired, so no takeover happened after "
        "materialization and this test proved nothing"
    )
    live_run = barrier["live_run"]
    successor_head = barrier["successor_head"]
    assert live_run is not None and live_run != stale_run
    assert successor_head

    # The successor's exact head and content survived the stale rollback.
    worktree = barrier["worktree"]
    assert _git(worktree, "rev-parse", "HEAD") == successor_head
    assert (worktree / "src" / "owned" / "successor.py").read_text(
        encoding="utf-8"
    ) == "successor = True\n"

    with contextlib.closing(kb.connect()) as conn:
        leaked = {
            kind: _events(conn, task_id, kind=kind)
            for kind in (
                "completed", "run_handover_completed", "run_handover_failed",
                "completion_blocked_file_scope", "timed_out", "gave_up",
            )
        }
        assert leaked == {kind: [] for kind in leaked}, (
            "a superseded worker wrote durable events onto a task the "
            "successor run owns"
        )

        task = kb.get_task(conn, task_id)
        assert task is not None
        assert task.status == "running"
        assert task.current_run_id == live_run
        assert task.worker_pid == 555_013, "the successor's claim was released"
        assert task.completed_at is None

        runs = {r.id: r for r in kb.list_runs(conn, task_id)}
        assert runs[live_run].status == "running"
        assert runs[live_run].outcome is None
        assert runs[stale_run].status == "reclaimed"


def test_bound_completion_still_records_a_genuine_scope_rejection(
    tmp_path, monkeypatch,
):
    """Non-degradation control (unchanged behaviour, passes before the fix
    too): silence is for a SUPERSEDED run only.

    Same bound completion — an ``expected_run_id`` that IS the task's current
    open run — rejecting a genuinely out-of-scope patch must still leave its
    ``completion_blocked_file_scope`` audit record, and still raise.
    """
    repo = _repo(tmp_path)
    monkeypatch.setenv(
        "HERMES_KANBAN_ATTACHMENTS_ROOT", str(tmp_path / "attachments"),
    )
    conn = kb.connect(tmp_path / "kanban.db")
    try:
        task_id = kb.create_task(
            conn, title="genuinely out of scope", assignee="worker",
            workspace_kind="worktree", workspace_path=str(repo),
            branch_name="feature/genuine-scope", owned_paths=["src/owned"],
            max_runtime_seconds=3600,
        )
        workspace = _materialize(conn, task_id)
        task = kb.get_task(conn, task_id)
        assert task is not None and task.current_run_id is not None
        run_id = int(task.current_run_id)
        original_head = _git(workspace, "rev-parse", "HEAD")

        patch_id = kb.store_attachment_bytes(
            conn, task_id, "sandbox.patch",
            _new_file_patch("src/outside.py", "leak = True\n"),
            content_type="text/x-diff", uploaded_by="agent",
            expected_run_id=run_id,
        )

        with pytest.raises(kb.WorktreeScopeError):
            kb.complete_task(
                conn, task_id, summary="Must fail closed",
                patch_attachment_id=patch_id, expected_run_id=run_id,
            )

        assert _events(conn, task_id, kind="completion_blocked_file_scope"), (
            "a real scope violation on the run that still owns the task lost "
            "its audit record"
        )
        blocked = [
            e for e in kb.list_events(conn, task_id)
            if e.kind == "completion_blocked_file_scope"
        ]
        assert len(blocked) == 1
        assert blocked[0].payload.get("reason")

        # Fail-closed as before: nothing landed, the run still owns the task.
        after = kb.get_task(conn, task_id)
        assert after is not None
        assert after.status == "running"
        assert after.head_commit is None
        assert after.current_run_id == run_id
        assert _git(workspace, "rev-parse", "HEAD") == original_head
        assert _git(workspace, "status", "--porcelain") == ""
    finally:
        conn.close()


def test_bound_budget_exit_still_records_a_genuine_handover_failure(
    default_board_home, tmp_path, monkeypatch,
):
    """Non-degradation control (unchanged behaviour, passes before the fix
    too): the iteration-budget exit of a run that is STILL current and whose
    handover genuinely fails must keep reporting it — the failure event, the
    scope-rejection event, and the fallback to the ordinary timeout.
    """
    from agent.turn_finalizer import _record_kanban_budget_exhausted

    repo = _repo(tmp_path)
    with contextlib.closing(kb.connect()) as conn:
        task_id = kb.create_task(
            conn, title="genuine handover failure, current run",
            assignee="worker",
            workspace_kind="worktree", workspace_path=str(repo),
            branch_name="feature/genuine-handover-failure",
            owned_paths=["src/owned"], max_runtime_seconds=3600,
        )
        _materialize(conn, task_id)
        task = kb.get_task(conn, task_id)
        assert task is not None and task.current_run_id is not None
        run_id = int(task.current_run_id)
        kb._set_worker_pid(conn, task_id, 555_013)
        # A complete handover whose patch escapes the declared scope.
        kb.store_attachment_bytes(
            conn, task_id, "sandbox.patch",
            _new_file_patch("src/outside.py", "leak = True\n"),
            content_type="text/x-diff", uploaded_by="agent",
            expected_run_id=run_id,
        )
        kb.store_attachment_bytes(
            conn, task_id, "report.md", b"## Summary\nDone (allegedly).\n",
            content_type="text/markdown", uploaded_by="agent",
            expected_run_id=run_id,
        )

    _worker_env(monkeypatch, task_id, run_id)

    _record_kanban_budget_exhausted(
        task_id, 200, 200, logging.getLogger("test.turn_finalizer"),
    )

    with contextlib.closing(kb.connect()) as conn:
        failed = [
            e for e in kb.list_events(conn, task_id)
            if e.kind == "run_handover_failed"
        ]
        assert len(failed) == 1, (
            "a genuine handover failure on the run that still owns the task "
            "was silently dropped"
        )
        assert failed[0].run_id == run_id
        assert failed[0].payload.get("reason")
        assert _events(conn, task_id, kind="completion_blocked_file_scope"), (
            "the scope rejection behind that failure lost its audit record"
        )
        assert _events(conn, task_id, kind="run_handover_completed") == []

        # And the caller still fell back to its ordinary timeout handling.
        assert _events(conn, task_id, kind="timed_out")
        task = kb.get_task(conn, task_id)
        assert task is not None
        assert task.status == "ready"
        assert task.head_commit is None
        run = kb.latest_run(conn, task_id)
        assert run is not None
        assert run.id == run_id
        assert run.outcome == "timed_out"

    assert "src/outside.py" not in _git(
        repo, "log", "--all", "--name-only", "--pretty=format:",
    )


# ---------------------------------------------------------------------------
# 11. The gave-up breaker never discards already-attached artifacts
# ---------------------------------------------------------------------------


def test_gave_up_breaker_preserves_and_records_attachments(tmp_path, monkeypatch):
    attachment_root = tmp_path / "attachments"
    monkeypatch.setenv("HERMES_KANBAN_ATTACHMENTS_ROOT", str(attachment_root))
    conn = kb.connect(tmp_path / "kanban.db")
    try:
        task_id = kb.create_task(conn, title="flaky worker", assignee="worker")
        claimed = kb.claim_task(conn, task_id)
        assert claimed is not None

        attachment_id = kb.store_attachment_bytes(
            conn, task_id, "partial-notes.txt", b"important partial state",
            content_type="text/plain", uploaded_by="agent",
        )

        tripped = kb._record_task_failure(
            conn, task_id, "worker crashed repeatedly",
            outcome="crashed", failure_limit=1,
            release_claim=False, end_run=False,
        )
        assert tripped is True

        task = kb.get_task(conn, task_id)
        assert task is not None
        assert task.status == "blocked"

        gave_up = [
            e for e in kb.list_events(conn, task_id) if e.kind == "gave_up"
        ][-1]
        assert gave_up.payload["attachments"]["count"] == 1
        assert gave_up.payload["attachments"]["ids"] == [attachment_id]

        # Attachment rows and bytes are still present and readable.
        attachments = kb.list_attachments(conn, task_id)
        assert len(attachments) == 1
        assert attachments[0].id == attachment_id
        assert kb.read_attachment_bytes(attachments[0]) == b"important partial state"
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# 12. Regression guard: the fence decorator must sit on the real mutator,
#     not on the read-only handover-artifacts helper
# ---------------------------------------------------------------------------


def test_enforce_max_runtime_is_the_bounded_mutation_not_the_helper(
    tmp_path, monkeypatch,
):
    """``enforce_max_runtime`` is the public mutator and must open the
    fence's mutation deadline on entry; ``_run_handover_artifacts`` is a
    pure read-only helper it calls internally and must NOT open one of its
    own. Asserts the actual runtime mechanism ``@bounded_mutation``
    installs (a call into ``_mutation_deadline``), not source text.
    """
    opened_for: list[str] = []

    @contextlib.contextmanager
    def _spy_mutation_deadline(conn, what):
        opened_for.append(what)
        yield None

    monkeypatch.setattr(kb, "_mutation_deadline", _spy_mutation_deadline)

    conn = kb.connect(tmp_path / "kanban.db")
    try:
        result = kb._run_handover_artifacts(conn, "no-such-task", None)
        assert result is None
        assert opened_for == [], (
            "_run_handover_artifacts must not be a bounded mutation, but it "
            f"opened a mutation deadline for: {opened_for!r}"
        )

        timed_out = kb.enforce_max_runtime(conn)
        assert timed_out == []
        assert opened_for == ["enforce_max_runtime"], (
            "enforce_max_runtime must be wrapped in @bounded_mutation and "
            f"open its own deadline; instead observed: {opened_for!r}"
        )
    finally:
        conn.close()
