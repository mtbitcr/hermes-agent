"""A kanban worker whose provider refuses the task exits with its own code and is parked once.

The dispatcher claims the card from a named board under the test root and launches it
through ``_default_spawn`` (only ``Popen`` is faked); the worker's one-shot entry then
runs under exactly the environment that spawn built. The kernel books the exit on the
claimed board: the run ends ``provider_refused``, no failure is counted, and the card is
parked blocked as a capability problem, so it is never started again unchanged and
today's owner retry accepts it. A goal-mode worker refused on a later turn leaves with
the same code, before its goal loop's judge or turn budget sees that turn.
"""

from __future__ import annotations

import os
import subprocess
from contextlib import closing
from pathlib import Path
from types import SimpleNamespace

import pytest

import cli
from agent.turn_author import TURN_AUTHOR_ENV
from hermes_cli import goals
from hermes_cli import kanban_db as kb
from hermes_cli.kanban_provider_stops import KANBAN_PROVIDER_REFUSED_EXIT_CODE

BOARD = "refusals"
WORKER_PID = 51515
REFUSED = {
    "final_response": "",
    "completed": False,
    "failed": True,
    "error": "content_policy_blocked: the provider declined to continue this request",
}


@pytest.fixture
def root(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    (home / "profiles" / "worker").mkdir(parents=True)
    (home / "profiles" / "worker" / "config.yaml").write_text("toolsets:\n  - kanban\n", encoding="utf-8")
    (home / "config.yaml").write_text("toolsets:\n  - kanban\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    for key in [key for key in os.environ if key.startswith("HERMES_KANBAN_")]:
        monkeypatch.delenv(key)
    for key in ("HERMES_PROFILE", "HERMES_TENANT", "HERMES_TUI", TURN_AUTHOR_ENV):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("HERMES_KANBAN_CRASH_GRACE_SECONDS", "0")
    monkeypatch.setattr(kb, "_pid_alive", lambda _pid: False)
    kb.create_board(BOARD)
    return home


@pytest.fixture
def spawns(monkeypatch):
    """Every worker the dispatcher launches; nothing is started."""
    launched = []

    def fake_popen(cmd, *_args, **kwargs):
        launched.append({"cmd": list(cmd), "env": dict(kwargs.get("env") or {})})
        return SimpleNamespace(pid=WORKER_PID)

    monkeypatch.setattr(kb, "_resolve_hermes_argv", lambda: ["hermes"])
    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    return launched


class _Worker:
    """The slice of HermesCLI the one-shot branches touch; its provider refuses the turn."""

    def __init__(self, result):
        self.result = result
        self.calls = []
        self.session_id = "worker-session"
        self.conversation_history = []
        self.console = SimpleNamespace(print=lambda *_args, **_kwargs: None)
        self._active_agent_route_signature = "route"
        self.agent = SimpleNamespace(
            session_id=self.session_id, _credential_pool=None, run_conversation=self._run_conversation,
        )

    def _claim_active_session(self, _surface, *, stderr=False):
        return True

    def _show_security_advisories(self):
        pass

    def chat(self, query, images=None):
        self.calls.append("chat")
        self._last_turn_result = self.result
        return self.result.get("final_response")

    def _print_exit_summary(self, clear_screen=True):
        self.calls.append("summary")

    def _ensure_runtime_credentials(self):
        return True

    def _resolve_turn_agent_config(self, _query):
        return {"signature": "route", "model": None, "runtime": None}

    def _init_agent(self, **_kwargs):
        return True

    def _run_conversation(self, **_kwargs):
        self.calls.append("run")
        return self.result


class _GoalWorker(_Worker):
    """A goal-mode worker whose model calls answer with ``turns`` in order, one per call."""

    def __init__(self, *turns):
        super().__init__(None)
        self.turns = list(turns)

    def _run_conversation(self, **_kwargs):
        self.calls.append("run")
        return self.turns.pop(0)


def _as_spawned_worker(monkeypatch, env):
    """Make this process's environment exactly the one the dispatcher gave its worker."""
    for key in [key for key in os.environ if key not in env]:
        monkeypatch.delenv(key)
    for key, value in env.items():
        if os.environ.get(key) != value:
            monkeypatch.setenv(key, value)
    # The one-shot entry sets this marker on os.environ itself; registering it here undoes that.
    monkeypatch.setenv("HERMES_SINGLE_QUERY_SESSION", os.environ.get("HERMES_SINGLE_QUERY_SESSION", "1"))
    monkeypatch.setattr(cli, "_should_seed_interactive", lambda *_args: False)
    monkeypatch.setattr(cli, "_collect_query_images", lambda query, _image: (query, []))
    monkeypatch.setattr(cli, "_collect_kanban_task_images", lambda _images: [])
    monkeypatch.setattr(cli, "_finalize_single_query", lambda worker: worker.calls.append("finalize"))


def _worker_exit_code(worker, task_id, *, quiet=False):
    """The code the worker process ends with; a normal return means it exits 0."""
    try:
        cli._run_single_query_mode(worker, f"work kanban task {task_id}", None, quiet, False)
    except SystemExit as exc:
        return exc.code
    return 0


def test_refused_worker_exits_with_its_own_code_is_parked_once_and_never_respawned(
    root, spawns, monkeypatch,
):
    conn = kb.connect(board=BOARD)
    try:
        task_id = kb.create_task(conn, title="refused card", assignee="worker")
        kb.dispatch_once(conn, board=BOARD)
        assert len(spawns) == 1
        task = kb.get_task(conn, task_id)
        run_id = task.current_run_id
        assert task.status == "running" and task.worker_pid == WORKER_PID

        env = spawns[0]["env"]
        assert env["HERMES_HOME"] == str(root / "profiles" / "worker")
        assert env["HERMES_KANBAN_TASK"] == task_id
        assert env["HERMES_KANBAN_RUN_ID"] == str(run_id)
        assert env["HERMES_KANBAN_DB"] == str(kb.kanban_db_path(board=BOARD))
        assert env["HERMES_KANBAN_BOARD"] == BOARD
        assert env["HERMES_KANBAN_WORKSPACES_ROOT"] == str(kb.workspaces_root(board=BOARD))
        assert env["HERMES_KANBAN_WORKSPACE"]
        assert env["HERMES_PROFILE"] == "worker"

        with monkeypatch.context() as worker_process:
            _as_spawned_worker(worker_process, env)
            worker = _Worker(REFUSED)
            code = _worker_exit_code(worker, task_id)
        assert "finalize" in worker.calls
        assert code not in (0, 1, kb.KANBAN_RATE_LIMIT_EXIT_CODE)

        kb._record_worker_exit(WORKER_PID, code << 8)
        for _tick in range(5):
            kb.dispatch_once(conn, board=BOARD)

        assert len(spawns) == 1
        task = kb.get_task(conn, task_id)
        assert task.status == "blocked"
        assert task.block_kind == "capability"
        assert task.consecutive_failures == 0
        assert task.worker_pid is None
        run = kb.get_run(conn, run_id)
        assert run.outcome == "provider_refused"
        assert run.ended_at is not None
        blocked = [event for event in kb.list_events(conn, task_id) if event.kind == "blocked"]
        assert len(blocked) == 1
        assert blocked[0].run_id == run_id
        assert blocked[0].payload["kind"] == "capability"
        # The stop today's owner retry accepts.
        assert kb.stopped_work_retry_evidence(conn, task_id) == {"kind": "capability", "run_id": run_id}
    finally:
        conn.close()

    default_db = kb.kanban_db_path(board=kb.DEFAULT_BOARD)
    assert default_db != kb.kanban_db_path(board=BOARD)
    if default_db.exists():
        with closing(kb.connect(default_db)) as default_conn:
            assert default_conn.execute(
                "SELECT COUNT(*) FROM tasks WHERE id = ?", (task_id,),
            ).fetchone()[0] == 0


@pytest.mark.parametrize("quiet", [False, True])
def test_both_one_shot_paths_give_the_refusal_code(root, spawns, monkeypatch, quiet):
    conn = kb.connect(board=BOARD)
    try:
        task_id = kb.create_task(conn, title="refused card", assignee="worker")
        kb.dispatch_once(conn, board=BOARD)
        env = spawns[0]["env"]
    finally:
        conn.close()
    with monkeypatch.context() as worker_process:
        _as_spawned_worker(worker_process, env)
        refused = _worker_exit_code(_Worker(REFUSED), task_id, quiet=quiet)
        other = {**REFUSED, "error": "the provider returned an empty response"}
        ordinary = _worker_exit_code(_Worker(other), task_id, quiet=quiet)
    assert refused not in (0, 1, kb.KANBAN_RATE_LIMIT_EXIT_CODE)
    assert ordinary != refused


def test_goal_mode_worker_refused_on_a_later_turn_exits_with_the_refusal_code(root, spawns, monkeypatch):
    """The refused continuation is never judged and the goal loop never blocks the card itself."""
    conn = kb.connect(board=BOARD)
    try:
        task_id = kb.create_task(
            conn, title="goal card", body="Acceptance: the parser tests pass.", assignee="worker",
            goal_mode=True, goal_max_turns=2,
        )
        kb.dispatch_once(conn, board=BOARD)
        assert len(spawns) == 1
        run_id = kb.get_task(conn, task_id).current_run_id
    finally:
        conn.close()
    env = spawns[0]["env"]
    assert env["HERMES_KANBAN_GOAL_MODE"] == "1"
    assert env["HERMES_KANBAN_TASK"] == task_id
    assert env["HERMES_KANBAN_RUN_ID"] == str(run_id)
    assert env["HERMES_KANBAN_DB"] == str(kb.kanban_db_path(board=BOARD))
    assert "-Q" in spawns[0]["cmd"]  # goal-mode workers take the quiet one-shot path

    judged = []

    def judge_goal(_goal, last_response, **_kwargs):
        judged.append(last_response)
        return "continue", "scripted continue", False, None, False

    monkeypatch.setattr(goals, "judge_goal", judge_goal)
    worker = _GoalWorker({"final_response": "first pass", "completed": True}, REFUSED)
    with monkeypatch.context() as worker_process:
        _as_spawned_worker(worker_process, env)
        code = _worker_exit_code(worker, task_id, quiet=True)

    assert code == KANBAN_PROVIDER_REFUSED_EXIT_CODE
    assert judged == ["first pass"]  # only the first turn was judged
    assert worker.calls == ["run", "run", "finalize"]
    with closing(kb.connect(board=BOARD)) as conn:
        task = kb.get_task(conn, task_id)
        assert (task.status, task.current_run_id) == ("running", run_id)
        assert [event for event in kb.list_events(conn, task_id) if event.kind == "blocked"] == []


@pytest.mark.parametrize("quiet", [False, True])
def test_outside_a_kanban_worker_a_refusal_exits_as_before(tmp_path, monkeypatch, quiet):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    monkeypatch.setenv("HERMES_SINGLE_QUERY_SESSION", "1")
    monkeypatch.setattr(cli, "_should_seed_interactive", lambda *_args: False)
    monkeypatch.setattr(cli, "_collect_query_images", lambda query, _image: (query, []))
    monkeypatch.setattr(cli, "_collect_kanban_task_images", lambda _images: [])
    monkeypatch.setattr(cli, "_finalize_single_query", lambda worker: worker.calls.append("finalize"))

    other = {**REFUSED, "error": "the provider returned an empty response"}
    assert _worker_exit_code(_Worker(REFUSED), "t_none", quiet=quiet) == _worker_exit_code(
        _Worker(other), "t_none", quiet=quiet,
    )
