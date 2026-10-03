"""Delivery slice 3: the delivery record, the approval proof and the hook.

When ``kanban.delivery.enabled`` is true and an integration profile is set,
approving a review-required build card from the review lane records ONE
delivery for the source card and its approved head and creates ONE integration
card in the same transaction. Everything here runs on a real SQLite board and a
real git repository; nothing about either is mocked. The only stand-in is the
reviewer the model policy nominates, which is the same seam the other review
tests pin.
"""

from __future__ import annotations

import shutil
import sqlite3
import subprocess
import threading
from pathlib import Path

import pytest

REVIEWER = "raphael-verifier"
IMPLEMENTER = "test-worker"
INTEGRATOR = "integration-worker"

_WORKER_ENV = (
    "HERMES_KANBAN_DB",
    "HERMES_KANBAN_BOARD",
    "HERMES_KANBAN_TASK",
    "HERMES_KANBAN_RUN_ID",
    "HERMES_KANBAN_HOME",
    "HERMES_KANBAN_WORKSPACES_ROOT",
    "HERMES_SESSION_ID",
)


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


def _write_config(home: Path, *, enabled: bool, profile) -> None:
    (home / "config.yaml").write_text(
        "kanban:\n"
        "  delivery:\n"
        f"    enabled: {'true' if enabled else 'false'}\n"
        f"    integration_profile: {'null' if profile is None else profile}\n",
        encoding="utf-8",
    )


@pytest.fixture
def board(tmp_path, monkeypatch):
    """A Hermes root with a real git repository and the reviewer policy pinned."""
    root = tmp_path / ".hermes"
    root.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(root))
    monkeypatch.setenv("HERMES_PROFILE", IMPLEMENTER)
    for name in _WORKER_ENV:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    from hermes_cli import kanban_db as kb
    from plugins.dashboard_auth.raphael_workspace import model_policy

    monkeypatch.setattr(
        model_policy, "reviewer_profile_ids", lambda: (REVIEWER,), raising=True,
    )

    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(
        ["git", "init", "-b", "main", str(repo)],
        check=True, capture_output=True, text=True,
    )
    (repo / "README.md").write_text("base\n", encoding="utf-8")
    _git(repo, "add", "README.md")
    _git(repo, "commit", "-m", "init")

    kb._INITIALIZED_PATHS.clear()
    kb.init_db()
    return kb, root, repo


def _park(kb, conn, repo: Path, name: str = "feature") -> tuple[str, str, Path]:
    """Build a review-required card in its own worktree and hand it over."""
    tid = kb.create_task(
        conn,
        title=f"build {name}",
        assignee=IMPLEMENTER,
        requires_review=True,
        owned_paths=["src/impl"],
        workspace_kind="worktree",
        workspace_path=str(repo),
        branch_name=f"feature/{name}",
    )
    run = kb.claim_task(conn, tid, claimer=f"{IMPLEMENTER}:1")
    assert run is not None
    workspace, branch = kb._resolve_worktree_workspace(run)
    kb.set_workspace_path(conn, tid, workspace)
    kb.set_branch_name(conn, tid, branch)
    kb.record_worktree_base(conn, tid, workspace)
    (workspace / "src" / "impl").mkdir(parents=True, exist_ok=True)
    (workspace / "src" / "impl" / f"{name}.py").write_text(
        "ok = True\n", encoding="utf-8",
    )
    _git(workspace, "add", f"src/impl/{name}.py")
    _git(workspace, "commit", "-m", f"feat: {name}")
    head = _git(workspace, "rev-parse", "HEAD")
    kb.complete_task(
        conn, tid,
        summary="implemented the slice",
        expected_run_id=run.current_run_id,
    )
    parked = kb.get_task(conn, tid)
    assert (parked.status, parked.assignee) == ("review", REVIEWER)
    return tid, head, Path(workspace)


def _approve(kb, conn, tid: str, *, summary: str = "looks right") -> bool:
    review = kb.claim_review_task(conn, tid, claimer=f"{REVIEWER}:1")
    assert review is not None
    return kb.complete_task(
        conn, tid, summary=summary, expected_run_id=review.current_run_id,
    )


def _independent_counts(db_path: Path, tid: str, head: str):
    """Read the delivery straight off the board file, past the kernel."""
    raw = sqlite3.connect(str(db_path))
    try:
        records = raw.execute(
            "SELECT integration_task_id FROM kanban_deliveries "
            "WHERE source_task_id = ? AND source_head = ?",
            (tid, head),
        ).fetchall()
        cards = raw.execute(
            "SELECT id, assignee FROM tasks WHERE idempotency_key = ?",
            ("delivery:" + tid + ":" + head,),
        ).fetchall()
        total_records = raw.execute(
            "SELECT COUNT(*) FROM kanban_deliveries"
        ).fetchone()[0]
        total_tasks = raw.execute("SELECT COUNT(*) FROM tasks").fetchone()[0]
    finally:
        raw.close()
    return records, cards, total_records, total_tasks


def test_two_concurrent_approvals_give_one_record_and_one_card(board):
    kb, root, repo = board
    from hermes_cli import kanban_delivery as kd

    _write_config(root, enabled=True, profile=INTEGRATOR)
    conn = kb.connect()
    try:
        tid, head, _ = _park(kb, conn, repo)
        review = kb.claim_review_task(conn, tid, claimer=f"{REVIEWER}:1")
        assert review is not None
    finally:
        conn.close()

    def race(action) -> list:
        barrier = threading.Barrier(2)
        results: list = []

        def run() -> None:
            own = kb.connect()
            try:
                barrier.wait()
                results.append(action(own))
            except Exception as exc:  # the loser of the approval race
                results.append(exc)
            finally:
                own.close()

        threads = [threading.Thread(target=run) for _ in range(2)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=60)
        return results

    approvals = race(lambda own: kb.complete_task(
        own, tid, summary="approved", expected_run_id=review.current_run_id,
    ))
    assert [result is True for result in approvals].count(True) == 1, approvals

    records, cards, total_records, _ = _independent_counts(
        kb.kanban_db_path(), tid, head,
    )
    assert total_records == 1
    assert len(records) == 1 and len(cards) == 1
    assert records[0][0] == cards[0][0]
    assert cards[0][1] == INTEGRATOR

    # The same source card and head delivered twice more, concurrently, still
    # names the one card the approval created.
    again = race(lambda own: kd.record_approved_delivery(own, tid))
    assert again == [cards[0][0], cards[0][0]]
    assert _independent_counts(kb.kanban_db_path(), tid, head)[:3] == (
        records, cards, 1,
    )


def test_approval_proof_end_to_end_on_a_real_board_and_repository(board):
    kb, root, repo = board
    from hermes_cli import kanban_delivery as kd

    _write_config(root, enabled=True, profile=INTEGRATOR)
    base = _git(repo, "rev-parse", "HEAD")
    conn = kb.connect()
    try:
        tid, head, _ = _park(kb, conn, repo)
        # Parked but not yet approved: the proof refuses and nothing is recorded.
        assert kd.prove_approval(conn, tid, head).allowed is False
        assert kd.record_approved_delivery(conn, tid) is None

        # The reviewer's prose names another commit; the record ignores it.
        assert _approve(kb, conn, tid, summary=f"approved head {base}") is True
        done = kb.get_task(conn, tid)
        assert (done.status, done.head_commit) == ("done", head)

        records, cards, total_records, total_tasks = _independent_counts(
            kb.kanban_db_path(), tid, head,
        )
        assert (total_records, total_tasks) == (1, 2)
        assert records == [(cards[0][0],)]
        integration = kb.get_task(conn, cards[0][0])
        assert integration.assignee == INTEGRATOR
        assert integration.status != "done"

        proof = kd.prove_approval(conn, tid, head)
        assert proof.allowed is True, proof
        assert kd.prove_approval(conn, tid, base).code == "head_moved"

        # The head must exist in the card's own repository.
        moved = repo.with_name("repo-moved")
        shutil.move(str(repo), str(moved))
        try:
            assert kd.prove_approval(conn, tid, head).code == "head_unconfirmed"
        finally:
            shutil.move(str(moved), str(repo))
        assert kd.prove_approval(conn, tid, head).allowed is True

        # A move back out of done after the approval retracts it.
        with kb.write_txn(conn):
            kb._append_event(conn, tid, "owner_move", {"to": "ready"})
        refused = kd.prove_approval(conn, tid, head)
        assert (refused.allowed, refused.code) == (
            False, "reopened_after_approval",
        )
        assert kd.record_approved_delivery(conn, tid) is None
        assert _independent_counts(kb.kanban_db_path(), tid, head)[2:] == (1, 2)
    finally:
        conn.close()


@pytest.mark.parametrize(
    "setting",
    [None, (False, INTEGRATOR), (True, None)],
    ids=["defaults", "disabled", "no-integration-profile"],
)
def test_with_delivery_off_the_approval_behaves_as_today(board, setting):
    kb, root, repo = board
    from hermes_cli.config import load_config

    if setting is None:
        assert load_config()["kanban"]["delivery"] == {
            "enabled": False, "integration_profile": None,
        }
    else:
        _write_config(root, enabled=setting[0], profile=setting[1])
    conn = kb.connect()
    try:
        tid, head, _ = _park(kb, conn, repo)
        assert _approve(kb, conn, tid) is True
        done = kb.get_task(conn, tid)
        assert (done.status, done.head_commit) == ("done", head)
        assert done.owned_paths == ["src/impl"]
        assert [task.id for task in kb.list_tasks(conn)] == [tid]
        kinds = [event.kind for event in kb.list_events(conn, tid)]
        assert kinds[-1] == "completed"
    finally:
        conn.close()
    _, cards, total_records, total_tasks = _independent_counts(
        kb.kanban_db_path(), tid, head,
    )
    assert (cards, total_records, total_tasks) == ([], 0, 1)


def test_store_path_under_the_dispatcher_worker_environment(board, monkeypatch):
    kb, root, repo = board
    kb.create_board("proj-a")
    kb.create_board("other")
    own_db = root / "kanban" / "boards" / "proj-a" / "kanban.db"
    other_db = root / "kanban" / "boards" / "other" / "kanban.db"

    conn = kb.connect(board="proj-a")
    try:
        tid, head, _ = _park(kb, conn, repo)
    finally:
        conn.close()

    # The reviewer worker exactly as the dispatcher spawns it: pinned to its
    # own board, naming its task, and living in a profile home under the root.
    profile_home = root / "profiles" / REVIEWER
    profile_home.mkdir(parents=True)
    _write_config(profile_home, enabled=True, profile=INTEGRATOR)
    monkeypatch.setenv("HERMES_HOME", str(profile_home))
    monkeypatch.setenv("HERMES_PROFILE", REVIEWER)
    monkeypatch.setenv("HERMES_KANBAN_DB", str(own_db))
    monkeypatch.setenv("HERMES_KANBAN_BOARD", "proj-a")
    monkeypatch.setenv("HERMES_KANBAN_TASK", tid)

    worker = kb.connect()
    try:
        review = kb.claim_review_task(worker, tid, claimer=f"{REVIEWER}:1")
        assert review is not None
        monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(review.current_run_id))
        verdict = kb.submit_review_findings(
            worker, tid,
            findings=[],
            candidate_digest="digest-clean",
            expected_run_id=review.current_run_id,
        )
        assert verdict["outcome"] == "passed"
        produced = [
            task.id for task in kb.list_tasks(worker) if task.id != tid
        ]
    finally:
        worker.close()

    # From the worker environment the root registry and every board resolve.
    assert kb.kanban_home() == root
    assert kb.register_db_path() == root / "kanban" / "board_register.db"
    assert kb.get_register_entry("proj-a") is not None
    assert kb.get_register_entry("other") is not None
    assert {"default", "proj-a", "other"} <= {
        entry["slug"] for entry in kb.list_boards()
    }
    assert kb.board_dir("other") / "kanban.db" == other_db

    # The kernel's own answer, checked against an independent count.
    records, cards, total_records, total_tasks = _independent_counts(
        own_db, tid, head,
    )
    assert (total_records, total_tasks) == (1, 2)
    assert produced == [cards[0][0]] == [records[0][0]]
    assert cards[0][1] == INTEGRATOR
    assert _independent_counts(other_db, tid, head)[2:] == (0, 0)
    assert _independent_counts(root / "kanban.db", tid, head)[2:] == (0, 0)
    assert not (profile_home / "kanban.db").exists()
    assert not (profile_home / "kanban").exists()
