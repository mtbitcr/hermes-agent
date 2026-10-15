"""Each run's receipt names the tool call that ends the run (finding 47).

A run's model and capability receipt is built from the run's own session log
while the run is ending. When a worker ends its run through one of its tools,
that call is still running: the log holds the model's request for it but not
yet its result row, so the ``tool_name`` column the receipt reads never names
it, and the stored receipt left out the very call that ended the run.

The three tools that end a worker's run now mark the call while it runs
(``tools.kanban_run_ending.ends_run``), and the receipt builder adds the
marked name. A run the kernel ends from a process that is not its worker
carries no mark, so its receipt lists exactly the session's tools, as before.
The mark is bound to the worker's own session, so an ending another agent
calls on the worker's card from outside any worker adds nothing either.

The worker side of every scenario runs in the exact environment the
dispatcher gives the worker it launches. It is taken from
``kanban_db._default_spawn`` itself, with only the process launch stubbed:
``HERMES_KANBAN_DB`` and ``HERMES_KANBAN_BOARD`` pinned to the worker's own
board, ``HERMES_KANBAN_TASK`` set, and ``HERMES_HOME`` at the worker's profile
home below a temporary root. From there the root registry and every other
board are checked to be still in reach. The worker's session lives in that
profile's real session store. Every stored receipt is read back from the
board file with plain SQL and checked against the session's tool names,
counted independently with plain SQL as well.
"""

from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
from contextlib import closing
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb

SLUG = "run-receipts"
OTHER = "other-board"
IMPLEMENTER = "builder"
# A profile the model policy admits; the policy only ever nominates those.
REVIEWER = "raphael-verifier"
BRANCH = "feature/run-receipts"
OWNED = "src/owned"
MODULE = f"{OWNED}/app.py"
SEED_AUTHOR = ("Board Owner", "owner@example.invalid")
WORKER_AUTHOR = ("Build Worker", "worker@example.invalid")
_GIT_ENV = {
    "GIT_CONFIG_GLOBAL": os.devnull,
    "GIT_CONFIG_SYSTEM": os.devnull,
    "GIT_TERMINAL_PROMPT": "0",
}
_REPO_ROOT = Path(__file__).resolve().parents[2]

# The calls the run made before the one that ends it; each has its result row.
EARLIER_CALLS = (
    ("kanban_show", {}),
    ("read_file", {"path": MODULE}),
    ("terminal", {"command": "python -m pytest -q"}),
    ("search_files", {"pattern": "value"}),
)

# The kernel ending a run from a process that is not the run's worker.
_END_ELSEWHERE = """
import sys
from hermes_cli import kanban_db as kb

board, task_id, run_id = sys.argv[1], sys.argv[2], int(sys.argv[3])
conn = kb.connect(board=board)
try:
    ended = kb.complete_task(
        conn, task_id,
        summary="closed by the kernel from another process",
        expected_run_id=run_id,
    )
finally:
    conn.close()
print("ENDED", ended)
"""


def _git(repo: Path, *argv: str, author: tuple[str, str] | None = None) -> str:
    """Run a real git command in ``repo``, isolated from the host's config."""
    identity: list[str] = []
    if author is not None:
        identity = ["-c", f"user.name={author[0]}", "-c", f"user.email={author[1]}"]
    proc = subprocess.run(
        ["git", "-C", str(repo), *identity, *argv],
        capture_output=True, text=True, timeout=60,
        env={**os.environ, **_GIT_ENV},
    )
    assert proc.returncode == 0, (argv, proc.stderr or proc.stdout)
    return proc.stdout


def _spawned_worker_env(task, workspace: Path, monkeypatch) -> dict:
    """The environment ``_default_spawn`` gives the worker process it launches.

    Only the launch is stubbed: the stub records the argv and environment the
    dispatcher built and returns a pid, so no process is started.
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
        pid = kb._default_spawn(task, str(workspace), board=SLUG)
    assert pid == _Launched.pid
    assert len(launches) == 1, launches
    argv, env = launches[0]
    assert f"work kanban task {task.id}" in argv
    return env


def _become(env: dict, monkeypatch) -> None:
    """Carry on as that process: this process now has exactly ``env``."""
    for name in list(os.environ):
        if name not in env:
            monkeypatch.delenv(name)
    for name, value in env.items():
        if os.environ.get(name) != value:
            monkeypatch.setenv(name, value)


def _assert_root_in_reach(root: Path) -> None:
    """From the worker's environment the root registry and every board are in reach.

    The kernel's answers are read against the registry file's own rows,
    counted with plain SQL.
    """
    registry = root / "kanban" / "board_register.db"
    assert kb.register_db_path().resolve() == registry.resolve()
    with closing(sqlite3.connect(registry)) as db:
        registered = {
            row[0] for row in db.execute("SELECT board_name FROM board_register")
        }
    assert {SLUG, OTHER} <= registered
    listed = {board["slug"] for board in kb.list_boards()}
    assert listed - {kb.DEFAULT_BOARD} == registered - {kb.DEFAULT_BOARD}
    for slug in registered:
        entry = kb.get_register_entry(slug)
        assert entry is not None and entry.lifecycle is kb.BoardLifecycle.LIVE, slug
        assert kb.board_dir(slug).resolve() == (root / "kanban" / "boards" / slug).resolve()


def _launch_worker(conn, claimed, root: Path, tmp_path: Path, monkeypatch) -> dict:
    """Prepare the claimed card as the dispatcher does, then become its worker."""
    task_id = claimed.id
    if claimed.workspace_kind == "worktree":
        workspace, branch = kb._resolve_worktree_workspace(claimed, board=SLUG)
    else:
        workspace, branch = kb.resolve_workspace(claimed, board=SLUG), None
    kb.set_workspace_path(conn, task_id, str(workspace))
    if claimed.workspace_kind == "worktree":
        kb.set_branch_name(
            conn, task_id,
            branch or (claimed.branch_name or "").strip() or f"wt/{task_id}",
        )
        kb.record_worktree_base(conn, task_id, workspace)

    env = _spawned_worker_env(claimed, Path(workspace), monkeypatch)

    # Pinned to its own card, run and board, under its profile home below the
    # root; the session id is the worker's own to publish once it starts.
    assert env["HERMES_KANBAN_TASK"] == task_id
    assert env["HERMES_KANBAN_RUN_ID"] == str(claimed.current_run_id)
    assert env["HERMES_KANBAN_BOARD"] == SLUG
    assert env["HERMES_KANBAN_DB"] == str(root / "kanban" / "boards" / SLUG / "kanban.db")
    assert env["HERMES_HOME"] == str(root / "profiles" / claimed.assignee)
    assert env["HERMES_PROFILE"] == claimed.assignee
    assert "HERMES_SESSION_ID" not in env
    for name in (
        "HERMES_HOME",
        "HERMES_KANBAN_DB",
        "HERMES_KANBAN_WORKSPACES_ROOT",
        "HERMES_KANBAN_WORKSPACE",
    ):
        assert Path(env[name]).resolve().is_relative_to(tmp_path.resolve()), name

    _become(env, monkeypatch)
    _assert_root_in_reach(root)
    return env


def _tool_call(call_id: str, name: str, arguments: dict) -> dict:
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(arguments)},
    }


def _record_session(
    session_id: str,
    profile: str,
    calls=EARLIER_CALLS,
    *,
    running: str | None = None,
) -> Path:
    """Write the run's session log into this worker's real session store.

    Each finished call has its request and its result row. ``running`` is the
    call still in progress when the run ends: only its request is logged.
    """
    from hermes_state import SessionDB

    state_db = Path(os.environ["HERMES_HOME"]) / "state.db"
    db = SessionDB(db_path=state_db)
    try:
        db.create_session(
            session_id, source="kanban", model="requested-model", profile_name=profile,
        )
        for index, (name, arguments) in enumerate(calls):
            call_id = f"call-{index}"
            db.append_message(
                session_id, "assistant", content="",
                tool_calls=[_tool_call(call_id, name, arguments)],
            )
            db.append_message(
                session_id, "tool", content="done", tool_name=name, tool_call_id=call_id,
            )
        if running is not None:
            db.append_message(
                session_id, "assistant", content="",
                tool_calls=[_tool_call("call-ending", running, {})],
            )
        db.update_token_counts(
            session_id, input_tokens=10, output_tokens=4,
            model="served-model", billing_provider="served-provider",
            estimated_cost_usd=0.01, cost_status="estimated",
            cost_source="official_docs_snapshot", api_call_count=1,
        )
    finally:
        db.close()
    return state_db


def _session_tool_names(state_db: Path, session_id: str) -> set[str]:
    """The session's distinct tool names, counted with plain SQL."""
    with closing(sqlite3.connect(state_db)) as db:
        rows = db.execute(
            "SELECT DISTINCT tool_name FROM messages "
            "WHERE session_id = ? AND tool_name IS NOT NULL AND tool_name != ''",
            (session_id,),
        ).fetchall()
    return {row[0] for row in rows}


def _stored_capability(root: Path, run_id: int) -> dict:
    """The run's capability receipt as stored in the board file, read with plain SQL."""
    board_db = root / "kanban" / "boards" / SLUG / "kanban.db"
    with closing(sqlite3.connect(board_db)) as db:
        row = db.execute(
            "SELECT metadata FROM task_runs WHERE id = ?", (run_id,)
        ).fetchone()
    assert row is not None and row[0], f"run {run_id} stored no metadata"
    metadata = json.loads(row[0])
    assert "runtime_receipt" in metadata, metadata
    capability = metadata["runtime_receipt"]["capability"]
    assert capability["source"] == "session-tool-calls", capability
    assert capability["tools_truncated"] is False, capability
    return capability


@pytest.fixture
def root(tmp_path, monkeypatch):
    """A Hermes root with two live boards: the worker's own and another one.

    The test process plays the dispatcher: it names no card, run, profile or
    session of its own.
    """
    home = tmp_path / "hermes_home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    for var in (
        "HERMES_KANBAN_DB",
        "HERMES_KANBAN_WORKSPACES_ROOT",
        "HERMES_KANBAN_HOME",
        "HERMES_KANBAN_BOARD",
        "HERMES_KANBAN_ATTACHMENTS_ROOT",
        "HERMES_KANBAN_TASK",
        "HERMES_KANBAN_RUN_ID",
        "HERMES_PROFILE",
        "HERMES_SESSION_ID",
    ):
        monkeypatch.delenv(var, raising=False)

    kb._INITIALIZED_PATHS.clear()
    kb._REGISTER_INITIALIZED = False
    kb._REGISTER_INITIALIZED_PATHS.clear()
    kb._ARCHIVE_INITIALIZED_PATHS.clear()

    for slug in (SLUG, OTHER):
        kb.create_board(slug)
        entry = kb.get_register_entry(slug)
        assert entry is not None and entry.lifecycle is kb.BoardLifecycle.LIVE
    return home


@pytest.fixture
def board(root):
    conn = kb.connect(board=SLUG)
    try:
        yield conn
    finally:
        conn.close()


@pytest.fixture
def review_build(root, board, tmp_path, monkeypatch):
    """A scoped build that needs review, with this process as its worker.

    The worker has committed its change inside its owned path, so the kernel
    can park the head for review. The reviewer is the one the review policy
    nominates and is an installed profile.
    """
    from plugins.dashboard_auth.raphael_workspace import model_policy

    monkeypatch.setattr(
        model_policy, "reviewer_profile_ids", lambda: (REVIEWER,), raising=True,
    )
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "--initial-branch=main")
    (repo / OWNED).mkdir(parents=True)
    (repo / MODULE).write_text("value = 1\n", encoding="utf-8")
    _git(repo, "add", MODULE)
    _git(repo, "commit", "-m", "owned module", author=SEED_AUTHOR)

    dispatcher_env = dict(os.environ)
    conn = board
    task_id = kb.create_task(
        conn, title="build that asks for review", assignee=IMPLEMENTER,
        workspace_kind="worktree", workspace_path=str(repo),
        branch_name=BRANCH, owned_paths=[OWNED], requires_review=True,
    )
    claimed = kb.claim_task(conn, task_id)
    assert claimed is not None and claimed.current_run_id is not None
    for profile in (IMPLEMENTER, REVIEWER):
        (root / "profiles" / profile).mkdir(parents=True)
    env = _launch_worker(conn, claimed, root, tmp_path, monkeypatch)

    workspace = Path(env["HERMES_KANBAN_WORKSPACE"])
    (workspace / MODULE).write_text("value = 2\n", encoding="utf-8")
    _git(workspace, "add", MODULE)
    _git(workspace, "commit", "-m", "change the owned module", author=WORKER_AUTHOR)
    return {
        "conn": conn,
        "task": task_id,
        "run": claimed.current_run_id,
        "dispatcher_env": dispatcher_env,
    }


def _request_review(build: dict, monkeypatch) -> Path:
    """The implementer asks for review through its tool; returns its session store."""
    from tools import kanban_tools as kt

    state_db = _record_session(
        "builder-review-session", IMPLEMENTER, running="kanban_request_review",
    )
    # The worker's agent publishes its own session id once it starts.
    monkeypatch.setenv("HERMES_SESSION_ID", "builder-review-session")
    out = json.loads(kt._handle_request_review({
        "summary": "changed the owned module and ran its unit tests",
    }))
    assert out.get("ok") is True, out
    assert out["status"] == "review"
    assert out["run_id"] == build["run"]
    return state_db


def test_completion_through_the_tool_names_kanban_complete(
    root, board, tmp_path, monkeypatch,
):
    """Test 1: the receipt of a run ended by ``kanban_complete`` lists it."""
    from tools import kanban_tools as kt

    conn = board
    task_id = kb.create_task(conn, title="plain build", assignee=IMPLEMENTER)
    claimed = kb.claim_task(conn, task_id)
    assert claimed is not None and claimed.current_run_id is not None
    (root / "profiles" / IMPLEMENTER).mkdir(parents=True)
    _launch_worker(conn, claimed, root, tmp_path, monkeypatch)

    state_db = _record_session(
        "builder-complete-session", IMPLEMENTER, running="kanban_complete",
    )
    # The worker's agent publishes its own session id once it starts.
    monkeypatch.setenv("HERMES_SESSION_ID", "builder-complete-session")
    out = json.loads(kt._handle_complete({"summary": "built it and ran the tests"}))
    assert out.get("ok") is True, out
    assert out["run_id"] == claimed.current_run_id

    session_tools = _session_tool_names(state_db, "builder-complete-session")
    assert session_tools == {name for name, _ in EARLIER_CALLS}
    capability = _stored_capability(root, claimed.current_run_id)
    assert capability["tools"] == sorted(session_tools | {"kanban_complete"})
    assert len(capability["tools"]) == len(session_tools) + 1


def test_review_request_through_the_tool_names_kanban_request_review(
    root, review_build, monkeypatch,
):
    """Test 2: the receipt of a run ended by ``kanban_request_review`` lists it."""
    state_db = _request_review(review_build, monkeypatch)
    assert kb.get_task(review_build["conn"], review_build["task"]).status == "review"

    session_tools = _session_tool_names(state_db, "builder-review-session")
    assert session_tools == {name for name, _ in EARLIER_CALLS}
    capability = _stored_capability(root, review_build["run"])
    assert capability["tools"] == sorted(session_tools | {"kanban_request_review"})
    assert len(capability["tools"]) == len(session_tools) + 1


def test_review_verdict_through_the_tool_names_kanban_review_findings(
    root, review_build, tmp_path, monkeypatch,
):
    """Test 3: the receipt of a review run ended by ``kanban_review_findings`` lists it."""
    from tools import kanban_tools as kt

    build = review_build
    conn, task_id = build["conn"], build["task"]
    _request_review(build, monkeypatch)

    # Back in the dispatcher: it claims the review and launches the reviewer.
    _become(build["dispatcher_env"], monkeypatch)
    review = kb.claim_review_task(conn, task_id)
    assert review is not None and review.assignee == REVIEWER
    assert review.current_run_id not in (None, build["run"])
    _launch_worker(conn, review, root, tmp_path, monkeypatch)

    state_db = _record_session(
        "reviewer-session", REVIEWER, running="kanban_review_findings",
    )
    # The reviewer's agent publishes its own session id once it starts.
    monkeypatch.setenv("HERMES_SESSION_ID", "reviewer-session")
    out = json.loads(kt._handle_review_findings({"findings": []}))
    assert out.get("ok") is True, out
    assert out["outcome"] == "passed"
    assert kb.get_task(conn, task_id).status == "done"

    session_tools = _session_tool_names(state_db, "reviewer-session")
    assert session_tools == {name for name, _ in EARLIER_CALLS}
    capability = _stored_capability(root, review.current_run_id)
    assert capability["tools"] == sorted(session_tools | {"kanban_review_findings"})
    assert len(capability["tools"]) == len(session_tools) + 1


def test_a_run_the_kernel_ends_from_another_process_lists_exactly_the_session_tools(
    root, board, tmp_path, monkeypatch,
):
    """Check 6: no ending call is added when the run is not ended by its worker's tool."""
    from tools import kanban_tools as kt

    conn = board
    dispatcher_env = dict(os.environ)
    task_id = kb.create_task(conn, title="build ended elsewhere", assignee=IMPLEMENTER)
    claimed = kb.claim_task(conn, task_id)
    assert claimed is not None and claimed.current_run_id is not None
    run_id = claimed.current_run_id
    (root / "profiles" / IMPLEMENTER).mkdir(parents=True)
    _launch_worker(conn, claimed, root, tmp_path, monkeypatch)

    # The worker works, and its heartbeat links the run to its session.
    state_db = _record_session(
        "builder-elsewhere-session", IMPLEMENTER,
        EARLIER_CALLS + (("kanban_heartbeat", {}),),
    )
    monkeypatch.setenv("HERMES_SESSION_ID", "builder-elsewhere-session")
    assert json.loads(kt._handle_heartbeat({}))["ok"] is True

    # The kernel ends the run in a separate process with the dispatcher's
    # environment: it names no card and runs no worker tool.
    _become(dispatcher_env, monkeypatch)
    assert "HERMES_KANBAN_TASK" not in os.environ
    assert "HERMES_SESSION_ID" not in os.environ
    python_path = os.pathsep.join(
        filter(None, (str(_REPO_ROOT), dispatcher_env.get("PYTHONPATH")))
    )
    child = subprocess.run(
        [sys.executable, "-c", _END_ELSEWHERE, SLUG, task_id, str(run_id)],
        capture_output=True, text=True, timeout=300, cwd=str(_REPO_ROOT),
        env={**dispatcher_env, "PYTHONPATH": python_path},
    )
    assert child.returncode == 0, child.stderr or child.stdout
    assert "ENDED True" in child.stdout, (child.stdout, child.stderr)
    assert kb.get_task(conn, task_id).status == "done"

    session_tools = _session_tool_names(state_db, "builder-elsewhere-session")
    assert session_tools == {name for name, _ in EARLIER_CALLS} | {"kanban_heartbeat"}
    capability = _stored_capability(root, run_id)
    assert capability["tools"] == sorted(session_tools)
    assert len(capability["tools"]) == len(session_tools)


@pytest.fixture
def live_worker():
    """A real live process to record as a run's worker; stopped at teardown.

    It only sleeps, in a session of its own, so no signal meant for a worker
    reaches the test process.
    """
    child = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(600)"],
        stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    try:
        yield child
    finally:
        child.kill()
        child.wait(timeout=60)


def test_an_owner_completing_the_card_adds_nothing_to_the_worker_receipt(
    root, board, tmp_path, monkeypatch, live_worker,
):
    """Review finding: an ending called from outside the run's worker names nothing.

    The worker is live and linked to its session when the owner agent closes
    its card through the tool, from the owner's own session and outside any
    worker. The run's receipt is read from the worker's session, which never
    called ``kanban_complete``, so it lists exactly that session's tools.
    """
    from tools import kanban_tools as kt

    conn = board
    dispatcher_env = dict(os.environ)
    task_id = kb.create_task(conn, title="build the owner closes", assignee=IMPLEMENTER)
    claimed = kb.claim_task(conn, task_id)
    assert claimed is not None and claimed.current_run_id is not None
    run_id = claimed.current_run_id
    (root / "profiles" / IMPLEMENTER).mkdir(parents=True)
    _launch_worker(conn, claimed, root, tmp_path, monkeypatch)
    # The dispatcher records the live process it launched as the run's worker.
    kb._set_worker_pid(conn, task_id, live_worker.pid)

    # The worker works, and its heartbeat links the run to its session.
    state_db = _record_session(
        "builder-owner-closed-session", IMPLEMENTER,
        EARLIER_CALLS + (("kanban_heartbeat", {}),),
    )
    monkeypatch.setenv("HERMES_SESSION_ID", "builder-owner-closed-session")
    assert json.loads(kt._handle_heartbeat({}))["ok"] is True

    # The owner agent, in its own session, names no card and no run.
    _become(dispatcher_env, monkeypatch)
    monkeypatch.setenv("HERMES_SESSION_ID", "owner-session")
    assert "HERMES_KANBAN_TASK" not in os.environ
    assert "HERMES_KANBAN_RUN_ID" not in os.environ
    board_db = root / "kanban" / "boards" / SLUG / "kanban.db"
    with closing(sqlite3.connect(board_db)) as db:
        recorded = db.execute(
            "SELECT worker_pid FROM task_runs WHERE id = ?", (run_id,)
        ).fetchone()
    assert recorded == (live_worker.pid,)
    assert live_worker.poll() is None, "the run's worker is not live"

    out = json.loads(kt._handle_complete(
        {"board": SLUG, "task_id": task_id, "summary": "owner closure"}
    ))
    assert out.get("ok") is True, out
    assert out["run_id"] == run_id
    assert kb.get_task(conn, task_id).status == "done"

    session_tools = _session_tool_names(state_db, "builder-owner-closed-session")
    assert session_tools == {name for name, _ in EARLIER_CALLS} | {"kanban_heartbeat"}
    capability = _stored_capability(root, run_id)
    assert "kanban_complete" not in capability["tools"], capability["tools"]
    assert capability["tools"] == sorted(session_tools)
    assert len(capability["tools"]) == len(session_tools)
