"""Delivery slice 3: the delivery record, the approval proof and the hook.

When ``kanban.delivery.enabled`` is true,
approving a review-required build card from the review lane records ONE
delivery for the source card and its approved head and creates NO integration
card in the same transaction. Everything here runs on a real SQLite board and a
real git repository; nothing about either is mocked. The only stand-ins are the
reviewer the model policy nominates, which is the same seam the other review
tests pin, and the final process launch of the dispatcher's worker spawn.
"""

from __future__ import annotations

import os
import shutil
import sqlite3
import subprocess
import threading
from pathlib import Path

import pytest

REVIEWER = "raphael-verifier"
IMPLEMENTER = "test-worker"

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


def _write_config(home: Path, *, enabled: bool) -> None:
    (home / "config.yaml").write_text(
        "kanban:\n"
        "  delivery:\n"
        f"    enabled: {'true' if enabled else 'false'}\n",
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
            "SELECT id FROM kanban_deliveries "
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


def _sql(kb, statement: str, *params):
    """Read or change a board row straight in the file, past the kernel."""
    raw = sqlite3.connect(str(kb.kanban_db_path()))
    try:
        rows = raw.execute(statement, params).fetchall()
        raw.commit()
    finally:
        raw.close()
    return rows


def _rework(kb, repo, tid: str, commit: bool = True, restore: str = "") -> str:
    """The owner sends the approved card back, unless it is back already; the implementer builds
    it again, with a new commit, without one or back at the commit ``restore``, and the reviewer
    approves again. Returns the head."""
    conn = kb.connect()
    try:
        if kb.get_task(conn, tid).status == "done":
            assert kb.cas_transition_task(
                conn, tid, expected_status="done", expected_revision=kb.task_event_revision(conn, tid),
                to_status="ready", event_kind="owner_move", event_payload={"to": "ready"},
            )["moved"] is True
        assert kb.assign_task(conn, tid, IMPLEMENTER)
        run = kb.claim_task(conn, tid, claimer=f"{IMPLEMENTER}:1")
        workspace, _ = kb._resolve_worktree_workspace(run)
        if restore:
            _git(workspace, "reset", "-q", "--hard", restore)
        elif commit:
            (workspace / "src" / "impl" / "feature.py").write_text("ok = 2\n", encoding="utf-8")
            _git(workspace, "add", "src/impl/feature.py")
            _git(workspace, "commit", "-m", "fix: feature")
        head = _git(workspace, "rev-parse", "HEAD")
        kb.complete_task(conn, tid, summary="reworked", expected_run_id=run.current_run_id)
        assert _approve(kb, conn, tid) is True
    finally:
        conn.close()
    return head


def test_a_newer_approved_head_marks_the_older_delivery_returned_for_changes(board):
    kb, root, repo = board

    _write_config(root, enabled=True)
    conn = kb.connect()
    try:
        tid, first, _ = _park(kb, conn, repo)
        assert _approve(kb, conn, tid) is True
    finally:
        conn.close()
    _sql(kb, "INSERT INTO kanban_deliveries (source_task_id, source_head, created_at) "
             "VALUES ('t_other', ?, 1)", "e" * 40)

    second = _rework(kb, repo, tid)

    assert _sql(kb, "SELECT source_task_id, source_head, pull_request_state FROM kanban_deliveries "
                    "ORDER BY id") == [
        (tid, first, "returned_for_changes"),
        ("t_other", "e" * 40, None),
        (tid, second, None),
    ]
    assert _sql(kb, "SELECT COUNT(*) FROM tasks") == [(1,)]


def test_approval_records_the_delivery_and_creates_no_card(board):
    kb, root, repo = board
    from hermes_cli import kanban_delivery as kd

    _write_config(root, enabled=True)
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

    records, cards, total_records, total_tasks = _independent_counts(
        kb.kanban_db_path(), tid, head,
    )
    assert (len(records), cards, total_records, total_tasks) == (1, [], 1, 1)

    # The same source card and head delivered twice more, concurrently, still
    # names the one row the approval recorded, and changes nothing on it.
    rows = _sql(kb, "SELECT * FROM kanban_deliveries")
    again = race(lambda own: kd.record_approved_delivery(own, tid))
    assert again == [records[0][0], records[0][0]]
    assert _sql(kb, "SELECT * FROM kanban_deliveries") == rows
    assert _independent_counts(kb.kanban_db_path(), tid, head)[:3] == (
        records, cards, 1,
    )


def test_approval_proof_end_to_end_on_a_real_board_and_repository(board):
    kb, root, repo = board
    from hermes_cli import kanban_delivery as kd

    _write_config(root, enabled=True)
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
        assert (len(records), cards, total_records, total_tasks) == (1, [], 1, 1)

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
        assert _independent_counts(kb.kanban_db_path(), tid, head)[2:] == (1, 1)
    finally:
        conn.close()


@pytest.mark.parametrize(
    "setting",
    [None, False],
    ids=["defaults", "disabled"],
)
def test_with_delivery_off_the_approval_behaves_as_today(board, setting):
    kb, root, repo = board
    from hermes_cli.config import load_config

    if setting is None:
        assert load_config()["kanban"]["delivery"] == {
            "enabled": False,
        }
    else:
        _write_config(root, enabled=setting)
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


def test_the_proof_reads_only_the_cards_own_repository(board, monkeypatch):
    kb, root, repo = board
    from hermes_cli import kanban_delivery as kd

    conn = kb.connect()
    try:
        tid, head, _ = _park(kb, conn, repo)
        assert _approve(kb, conn, tid) is True
        assert kd.prove_approval(conn, tid, head).allowed is True

        # The card's own repository is now an empty one, while another
        # repository holds the approved objects.
        other = repo.with_name("repo-other")
        shutil.move(str(repo), str(other))
        subprocess.run(
            ["git", "init", "-b", "main", str(repo)],
            check=True, capture_output=True, text=True,
        )
        assert _git(other, "cat-file", "-t", head) == "commit"

        # The control sets no variable, and the proof refuses.
        monkeypatch.delenv("GIT_DIR", raising=False)
        control = kd.prove_approval(conn, tid, head)
        assert (control.allowed, control.code) == (False, "head_unconfirmed")

        # GIT_DIR naming the other repository refuses exactly the same way.
        monkeypatch.setenv("GIT_DIR", str(other / ".git"))
        assert kd.prove_approval(conn, tid, head) == control
    finally:
        conn.close()


def _spawned_worker_env(kb, task, workspace: Path, board: str, monkeypatch) -> dict:
    """The environment kanban_db._default_spawn gives the worker it launches.

    Only the final process launch is replaced: the stand-in records the argv
    and environment the dispatcher built and returns a pid, so no process
    starts. SQLite and git stay real.
    """
    launches = []

    class _Launched:
        pid = 4242

    def _launch(argv, **kwargs):
        log = kwargs.get("stdout")
        if hasattr(log, "close"):
            log.close()
        launches.append((list(argv), dict(kwargs.get("env") or {})))
        return _Launched()

    with monkeypatch.context() as patch:
        patch.setattr(subprocess, "Popen", _launch)
        pid = kb._default_spawn(task, str(workspace), board=board)
    assert pid == _Launched.pid
    assert len(launches) == 1, launches
    argv, env = launches[0]
    assert f"work kanban task {task.id}" in argv
    return env


def _become(env: dict, monkeypatch) -> None:
    """Carry on as the worker: this process now has exactly ``env``."""
    for name in list(os.environ):
        if name not in env:
            monkeypatch.delenv(name)
    for name, value in env.items():
        if os.environ.get(name) != value:
            monkeypatch.setenv(name, value)


def test_store_path_under_the_dispatcher_worker_environment(board, monkeypatch):
    kb, root, repo = board
    kb.create_board("proj-a")
    kb.create_board("other")
    own_db = root / "kanban" / "boards" / "proj-a" / "kanban.db"
    other_db = root / "kanban" / "boards" / "other" / "kanban.db"

    conn = kb.connect(board="proj-a")
    try:
        tid, head, _ = _park(kb, conn, repo)
        parked_head = kb._latest_review_head_provenance(conn, tid)
        assert parked_head == head
        # The reviewer's profile lives under the root, with delivery on.
        profile_home = root / "profiles" / REVIEWER
        profile_home.mkdir(parents=True)
        _write_config(profile_home, enabled=True)
        # The dispatcher's review lane: claim, resolve the work area, record
        # its branch and base, then spawn through the live _default_spawn.
        claimed = kb.claim_review_task(conn, tid)
        assert claimed is not None and claimed.current_run_id is not None
        assert claimed.claim_lock
        workspace, branch = kb._resolve_worktree_workspace(claimed, board="proj-a")
        kb.set_workspace_path(conn, tid, str(workspace))
        kb.set_branch_name(conn, tid, branch)
        kb.record_worktree_base(conn, tid, workspace)
        env = _spawned_worker_env(kb, claimed, workspace, "proj-a", monkeypatch)
    finally:
        conn.close()

    # Pinned to its own card, run, claim, work area and board, and living in
    # a profile home under the root.
    assert env["HERMES_KANBAN_TASK"] == tid
    assert env["HERMES_KANBAN_RUN_ID"] == str(claimed.current_run_id)
    assert env["HERMES_KANBAN_CLAIM_LOCK"] == claimed.claim_lock
    assert env["HERMES_KANBAN_WORKSPACE"] == str(workspace)
    assert env["HERMES_KANBAN_WORKSPACES_ROOT"] == str(
        kb.workspaces_root(board="proj-a")
    )
    assert env["HERMES_KANBAN_DB"] == str(own_db)
    assert env["HERMES_KANBAN_BOARD"] == "proj-a"
    assert env["HERMES_HOME"] == str(profile_home)

    # The reviewer worker carries on with exactly that environment and
    # passes the card with an empty verdict bound to the parked head.
    _become(env, monkeypatch)
    worker = kb.connect()
    try:
        verdict = kb.submit_review_findings(
            worker, tid,
            findings=[],
            candidate_digest=parked_head,
            expected_run_id=int(env["HERMES_KANBAN_RUN_ID"]),
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
    assert (len(records), cards, total_records, total_tasks) == (1, [], 1, 1)
    assert produced == []
    assert _independent_counts(other_db, tid, head)[2:] == (0, 0)
    assert _independent_counts(root / "kanban.db", tid, head)[2:] == (0, 0)
    assert not (profile_home / "kanban.db").exists()
    assert not (profile_home / "kanban").exists()


def test_the_publish_ledger_columns_join_an_existing_delivery_table(board):
    """Slice 4 keeps its publish ledger and lease in seven new columns of the delivery row.
    A board whose table predates them gains them on open, and its rows keep
    their values with an empty ledger."""
    kb, root, repo = board
    db_path = kb.kanban_db_path()
    raw = sqlite3.connect(str(db_path))
    try:
        raw.execute("DROP TABLE kanban_deliveries")
        raw.execute(
            "CREATE TABLE kanban_deliveries ("
            "id INTEGER PRIMARY KEY AUTOINCREMENT, source_task_id TEXT NOT NULL, "
            "source_head TEXT NOT NULL, integration_task_id TEXT, approval_run_id INTEGER, "
            "created_at INTEGER NOT NULL, UNIQUE(source_task_id, source_head))"
        )
        raw.execute(
            "INSERT INTO kanban_deliveries (source_task_id, source_head, integration_task_id, "
            "approval_run_id, created_at) VALUES ('t_old', ?, 't_card', 7, 1700000000)",
            ("a" * 40,),
        )
        raw.commit()
    finally:
        raw.close()

    kb._INITIALIZED_PATHS.clear()
    kb.init_db()

    raw = sqlite3.connect(str(db_path))
    try:
        columns = [row[1] for row in raw.execute("PRAGMA table_info(kanban_deliveries)")]
        rows = raw.execute("SELECT * FROM kanban_deliveries").fetchall()
    finally:
        raw.close()
    assert columns[-7:] == [
        "pull_request_number", "pull_request_head", "pull_request_state", "pull_request_branch",
        "publish_lease", "publish_lease_until", "publish_refusal",
    ]
    assert rows == [(1, "t_old", "a" * 40, "t_card", 7, 1700000000, None, None, None, None, None, None, None)]
