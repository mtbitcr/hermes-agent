"""A kanban worker stopped by the AI provider's usage limit exits with the rate-limit sentinel.

The dispatcher books exit ``KANBAN_RATE_LIMIT_EXIT_CODE`` as ``rate_limited`` and starts the
card again by itself, while a clean exit 0 without a terminal kanban call is booked as a
protocol violation. Workers normally run ``chat -q`` without ``-Q``, so the ordinary one-shot
branch must reach the same exit code as the quiet one; every other exit code stays as it was.
"""

from __future__ import annotations

import contextlib
import time
from types import SimpleNamespace

import pytest

import cli
from agent.credential_pool import STATUS_DEAD, STATUS_EXHAUSTED, STATUS_OK, PooledCredential
from agent.turn_author import TURN_AUTHOR_ENV
from hermes_cli import kanban_db
from hermes_cli.cli_chat_turn_mixin import CLIChatTurnMixin
from tools import skills_tool

TASK = "t_limit"
TEMPFAIL = kanban_db.KANBAN_RATE_LIMIT_EXIT_CODE
SUCCESS = {"final_response": "done", "completed": True}


def _failed(reason):
    return {"final_response": "", "failed": True, "error": "provider refused", "failure_reason": reason}


@pytest.fixture(autouse=True)
def _worker_env(monkeypatch):
    monkeypatch.setenv("HERMES_KANBAN_TASK", TASK)
    monkeypatch.delenv("HERMES_KANBAN_RUN_ID", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_GOAL_MODE", raising=False)
    monkeypatch.delenv(TURN_AUTHOR_ENV, raising=False)
    # The one-shot entry sets this marker on os.environ itself; registering it here undoes that.
    monkeypatch.setenv("HERMES_SINGLE_QUERY_SESSION", "1")
    monkeypatch.setattr(cli, "_should_seed_interactive", lambda *_args: False)
    monkeypatch.setattr(cli, "_collect_query_images", lambda query, _image: (query, []))
    monkeypatch.setattr(cli, "_collect_kanban_task_images", lambda _images: [])
    monkeypatch.setattr(cli, "_finalize_single_query", lambda worker: worker.calls.append("finalize"))


class _Worker:
    """The slice of HermesCLI the one-shot branches touch.

    ``chat`` keeps its turn result on the instance the way the real chat mixin does; the
    quiet branch reaches the same result through ``agent.run_conversation``.
    """

    def __init__(self, result, *, pool=None):
        self.result = result
        self.calls = []
        self.session_id = "worker-session"
        self.conversation_history = []
        self.console = SimpleNamespace(print=lambda *_args, **_kwargs: None)
        self._active_agent_route_signature = "route"
        self.agent = SimpleNamespace(
            session_id=self.session_id, _credential_pool=pool, run_conversation=self._run_conversation,
        )

    def _claim_active_session(self, _surface, *, stderr=False):
        return True

    def _show_security_advisories(self):
        pass

    def chat(self, query, images=None):
        self.calls.append("chat")
        self._last_turn_result = self.result
        return self.result.get("final_response") if isinstance(self.result, dict) else None

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


def _one_shot(worker, *, quiet=False):
    """The real ``-q`` entry; a normal return means the process exits 0."""
    return cli._run_single_query_mode(worker, f"work kanban task {TASK}", None, quiet, False)


def _credential(status, reset_at=None):
    return PooledCredential(
        provider="openai-codex", id=f"cred-{status}-{reset_at}", label=status, auth_type="api_key",
        priority=0, source="manual", access_token="sk-test", last_status=status,
        last_error_reset_at=reset_at,
    )


def _pool(now, entries):
    """A credential pool whose entries carry resets ``offset`` seconds after ``now``."""
    credentials = [
        _credential(status, None if offset is None else now + offset) for status, offset in entries
    ]
    return SimpleNamespace(entries=lambda: list(credentials))


@pytest.fixture
def recorded(monkeypatch):
    """The worker's own run identity, plus stand-ins for its kernel call: no board is opened."""
    calls = []
    board = object()

    @contextlib.contextmanager
    def connect_closing(*_args, **_kwargs):
        yield board

    def record_run_rate_limit_reset(conn, task_id, *, run_id, reset_at, now=None):
        calls.append((conn is board, task_id, run_id, reset_at))
        return True

    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", "42")
    monkeypatch.setattr(kanban_db, "connect_closing", connect_closing)
    monkeypatch.setattr(kanban_db, "record_run_rate_limit_reset", record_run_rate_limit_reset, raising=False)
    return calls


@pytest.mark.parametrize("reason", ["rate_limit", "billing", "overloaded"])
def test_ordinary_worker_exits_tempfail_after_a_usage_limit(reason):
    worker = _Worker(_failed(reason))

    with pytest.raises(SystemExit) as excinfo:
        _one_shot(worker)

    assert excinfo.value.code == TEMPFAIL
    # The exit summary still prints first, and the one-shot teardown still runs exactly once.
    assert worker.calls == ["chat", "summary", "finalize"]


@pytest.mark.parametrize(
    "in_task, result",
    [
        pytest.param(True, SUCCESS, id="success"),
        pytest.param(True, _failed("server_error"), id="other-failure-reason"),
        pytest.param(True, {"final_response": "", "failed": True, "error": "boom"}, id="failure-without-reason"),
        pytest.param(False, _failed("rate_limit"), id="outside-a-kanban-task"),
        pytest.param(True, None, id="no-turn-result"),
        pytest.param(True, "plain text", id="result-not-a-dict"),
    ],
)
def test_ordinary_worker_keeps_exit_zero_unless_its_task_hit_a_usage_limit(monkeypatch, in_task, result):
    if not in_task:
        monkeypatch.delenv("HERMES_KANBAN_TASK")
    worker = _Worker(result)

    assert _one_shot(worker) is None  # a normal return: the process exits 0 as before
    assert worker.calls == ["chat", "summary", "finalize"]

    # Control: the same worker inside its task after a usage limit leaves with the sentinel.
    monkeypatch.setenv("HERMES_KANBAN_TASK", TASK)
    limited = _Worker(_failed("rate_limit"))
    with pytest.raises(SystemExit) as excinfo:
        _one_shot(limited)
    assert excinfo.value.code == TEMPFAIL


def test_quiet_branch_exit_codes_follow_the_same_rule(monkeypatch):
    """-Q keeps 0 on success and 1 on any other failure; a usage limit in a task exits with the sentinel."""
    cases = [
        (True, SUCCESS, 0),
        (True, _failed("server_error"), 1),
        (False, _failed("rate_limit"), 1),
        (True, _failed("rate_limit"), TEMPFAIL),
        (True, _failed("billing"), TEMPFAIL),
        (True, _failed("overloaded"), TEMPFAIL),
    ]
    for in_task, result, code in cases:
        if in_task:
            monkeypatch.setenv("HERMES_KANBAN_TASK", TASK)
        else:
            monkeypatch.delenv("HERMES_KANBAN_TASK")
        worker = _Worker(result)

        with pytest.raises(SystemExit) as excinfo:
            _one_shot(worker, quiet=True)

        assert (excinfo.value.code, worker.calls) == (code, ["run", "finalize"]), (in_task, result)


def test_settled_turn_result_is_kept_on_the_cli():
    """The ordinary one-shot branch reads the finished turn's result from the CLI after chat()."""
    result = _failed("rate_limit")
    shell = SimpleNamespace(_prompt_start_time=None, _flush_stream=lambda: None, conversation_history=[], agent=None)

    CLIChatTurnMixin._chat_settle_turn(shell, cli._ChatTurn(result=result))

    assert shell._last_turn_result is result


def test_turn_that_never_ran_leaves_no_result_behind(monkeypatch):
    monkeypatch.setattr(skills_tool, "_secret_capture_callback", None)
    shell = SimpleNamespace(
        _secret_capture_callback=None,
        _last_turn_result=_failed("rate_limit"),  # an earlier turn's
        _ensure_runtime_credentials=lambda: False,
    )

    assert CLIChatTurnMixin.chat(shell, "hello") is None
    assert shell._last_turn_result is None


@pytest.mark.parametrize(
    "entries, expected_offset",
    [
        pytest.param([(STATUS_EXHAUSTED, 7200)], 7200, id="one-exhausted-credential"),
        pytest.param([(STATUS_EXHAUSTED, 7200), (STATUS_EXHAUSTED, 5400)], 5400, id="first-credential-to-free-up"),
        pytest.param([(STATUS_EXHAUSTED, 7200), (STATUS_DEAD, None)], 7200, id="dead-credential-ignored"),
    ],
)
def test_limit_exit_records_the_pool_reset_on_the_worker_run(recorded, entries, expected_offset):
    now = int(time.time())
    worker = _Worker(_failed("rate_limit"), pool=_pool(now, entries))

    with pytest.raises(SystemExit) as excinfo:
        _one_shot(worker)

    assert excinfo.value.code == TEMPFAIL
    assert recorded == [(True, TASK, 42, now + expected_offset)]


def test_quiet_limit_exit_records_the_pool_reset_too(recorded):
    now = int(time.time())
    worker = _Worker(_failed("overloaded"), pool=_pool(now, [(STATUS_EXHAUSTED, 7200)]))

    with pytest.raises(SystemExit) as excinfo:
        _one_shot(worker, quiet=True)

    assert excinfo.value.code == TEMPFAIL
    assert recorded == [(True, TASK, 42, now + 7200)]


@pytest.mark.parametrize(
    "entries",
    [
        pytest.param([(STATUS_EXHAUSTED, 8 * 86400)], id="reset-eight-days-out"),
        pytest.param([(STATUS_EXHAUSTED, -60)], id="reset-already-past"),
        pytest.param([(STATUS_EXHAUSTED, 7200), (STATUS_OK, None)], id="usable-credential-left"),
        pytest.param([(STATUS_EXHAUSTED, None)], id="no-reset-known"),
        pytest.param([(STATUS_DEAD, None)], id="only-dead-credentials"),
        pytest.param([], id="empty-pool"),
        pytest.param(None, id="no-pool"),
    ],
)
def test_limit_exit_records_nothing_without_a_reset_in_the_next_week(recorded, entries):
    worker = _Worker(_failed("rate_limit"), pool=None if entries is None else _pool(int(time.time()), entries))

    with pytest.raises(SystemExit) as excinfo:
        _one_shot(worker)

    assert excinfo.value.code == TEMPFAIL
    assert recorded == []


@pytest.mark.parametrize("run_id", [None, "", "run-7"], ids=["unset", "empty", "not-a-number"])
def test_limit_exit_records_nothing_without_the_worker_run_id(recorded, monkeypatch, run_id):
    if run_id is None:
        monkeypatch.delenv("HERMES_KANBAN_RUN_ID")
    else:
        monkeypatch.setenv("HERMES_KANBAN_RUN_ID", run_id)
    worker = _Worker(_failed("rate_limit"), pool=_pool(int(time.time()), [(STATUS_EXHAUSTED, 7200)]))

    with pytest.raises(SystemExit) as excinfo:
        _one_shot(worker)

    assert excinfo.value.code == TEMPFAIL
    assert recorded == []


@pytest.mark.parametrize("broken", ["connect_closing", "record_run_rate_limit_reset"])
def test_failed_reset_record_keeps_the_limit_exit(recorded, monkeypatch, broken):
    def refuse(*_args, **_kwargs):
        raise RuntimeError("board unavailable")

    monkeypatch.setattr(kanban_db, broken, refuse)
    worker = _Worker(_failed("rate_limit"), pool=_pool(int(time.time()), [(STATUS_EXHAUSTED, 7200)]))

    with pytest.raises(SystemExit) as excinfo:
        _one_shot(worker)

    assert excinfo.value.code == TEMPFAIL
    assert worker.calls == ["chat", "summary", "finalize"]
