"""A card stopped by the AI provider is held once, with a believable resume time.

A rate-limited run the worker left without a reset still holds its card past the
five-minute cooldown, because the kernel books a resume time of its own; a held
card writes one ``respawn_guarded`` event instead of one per tick; a quota stop is
never read as a sign-in problem; and a reset recorded under one provider stops
holding once the card's route moves to another, while one recorded under the
provider the card is then pinned onto, or under no known provider, still holds.
That provider is the one the run actually used: the worker's own, else the one
its runtime receipt names. Providers are compared by the id the worker resolves
them to, so a card naming its provider by an alias or in another case holds, and
its stops streak, as one naming it canonically; a card naming Vertex by another
of its names holds on the reset its worker gives; and a reset on any custom
endpoint, whatever name the card, the worker's pool or the receipt gives it, holds
its card until the reset, even once the card moves: only a move between two known
built-in providers ends a hold early. A reset the worker records
goes through the worker's own path, under exactly the environment the dispatcher
builds for the claimed run, onto a named board under the test root. The refusal
exit is covered end to end in ``tests/cli/test_kanban_worker_refusal_exit.py``.
A worker's stop stays booked as that stop when its claim expires or its heartbeat
goes stale before the dead-worker scan books it, and when a restart loses the exit
the dispatcher reaped: the worker recorded the stop on its own run on its way out.
"""

from __future__ import annotations

import json
import os
import signal
import sqlite3
import subprocess
from contextlib import closing, contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

import cli
from agent import vertex_adapter
from agent.credential_pool import STATUS_EXHAUSTED, CredentialPool, PooledCredential
from hermes_cli import kanban_db as kb
from hermes_cli.auth import resolve_provider
from hermes_cli.kanban_provider_stops import KANBAN_PROVIDER_REFUSED_EXIT_CODE
from hermes_cli.runtime_provider import resolve_runtime_provider
from plugins.dashboard_auth.raphael_workspace import model_policy

BOARD = "stops"
COOLDOWN = 300
SIX_HOURS = 6 * 3600
T0 = 1_900_000_000


@pytest.fixture
def clock(monkeypatch):
    now = {"t": T0}
    monkeypatch.setattr(kb.time, "time", lambda: now["t"])
    return now


@pytest.fixture
def conn(tmp_path, monkeypatch, clock):
    home = tmp_path / ".hermes"
    for profile in ("worker", "reviewer", "raphael-builder"):
        (home / "profiles" / profile).mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    for key in [key for key in os.environ if key.startswith("HERMES_KANBAN_")]:
        monkeypatch.delenv(key)
    monkeypatch.setenv("HERMES_KANBAN_RATE_LIMIT_COOLDOWN_SECONDS", str(COOLDOWN))
    monkeypatch.setenv("HERMES_KANBAN_CRASH_GRACE_SECONDS", "0")
    monkeypatch.setattr(kb, "_pid_alive", lambda _pid: False)
    kb.create_board(BOARD)
    connection = kb.connect(board=BOARD)
    yield connection
    connection.close()


def _claimer(pid):
    host = kb._claimer_id().split(":", 1)[0]
    return f"{host}:w{pid}"


def _start_worker(conn, task_id, pid, *, lane="ready", ttl_seconds=None):
    if lane == "review":
        task = kb.claim_review_task(conn, task_id, claimer=_claimer(pid), ttl_seconds=ttl_seconds)
    else:
        task = kb.claim_task(conn, task_id, claimer=_claimer(pid), ttl_seconds=ttl_seconds)
    assert task is not None
    conn.execute("UPDATE tasks SET worker_pid = ? WHERE id = ?", (pid, task_id))
    conn.commit()
    return task.current_run_id


def _worker_exits(conn, pid, code):
    kb._record_worker_exit(pid, code << 8)
    kb.detect_crashed_workers(conn)


def _card_in(conn, clock, lane, **fields):
    task_id = kb.create_task(conn, title=f"{lane} card", assignee="worker", **fields)
    if lane == "review":
        run_id = _start_worker(conn, task_id, 4100)
        assert kb.request_review(
            conn, task_id, summary="ready for review", reviewer="reviewer", expected_run_id=run_id,
        )
        assert kb.get_task(conn, task_id).status == "review"
        clock["t"] += 60
    return task_id


def _rate_limited_stop(conn, task_id, *, pid=4242, lane="ready"):
    run_id = _start_worker(conn, task_id, pid, lane=lane)
    _worker_exits(conn, pid, kb.KANBAN_RATE_LIMIT_EXIT_CODE)
    run = kb.get_run(conn, run_id)
    assert run.outcome == "rate_limited"
    return run


def _dispatcher_env(monkeypatch, conn, task_id):
    """The environment the live dispatcher builds for the worker of the task's claimed run."""
    launched = []

    def fake_popen(cmd, *_args, **kwargs):
        launched.append(dict(kwargs["env"]))
        return SimpleNamespace(pid=0)

    task = kb.get_task(conn, task_id)
    with monkeypatch.context() as dispatcher:
        dispatcher.setattr(kb, "_resolve_hermes_argv", lambda: ["hermes"])
        dispatcher.setattr(subprocess, "Popen", fake_popen)
        kb._default_spawn(task, str(kb.resolve_workspace(task, board=BOARD)), board=BOARD)
    return launched[0]


@contextmanager
def _as_worker_process(monkeypatch, env):
    """Run the block with exactly ``env`` as the process environment, as the worker does."""
    with monkeypatch.context() as worker_process:
        for key in [key for key in os.environ if key not in env]:
            worker_process.delenv(key)
        for key, value in env.items():
            worker_process.setenv(key, value)
        yield


def _record_as_worker(monkeypatch, conn, task_id, reset_at, *, provider, runtime=None):
    """The claimed run's worker records when its provider's limit lifts, as it does before exiting.

    The worker's own recording path runs under exactly the environment the
    dispatcher builds for that run, its credential pool (``provider``, the
    pool's key) exhausted until ``reset_at``; ``runtime`` is its agent's
    runtime provider, when given. The reset must land on the claimed board's run.
    """
    env = _dispatcher_env(monkeypatch, conn, task_id)
    assert env["HERMES_HOME"] == str(Path(os.environ["HERMES_HOME"]) / "profiles" / kb.get_task(conn, task_id).assignee)
    assert env["HERMES_KANBAN_TASK"] == task_id
    assert env["HERMES_KANBAN_DB"] == str(kb.kanban_db_path(board=BOARD))
    assert env["HERMES_KANBAN_BOARD"] == BOARD
    run_id = int(env["HERMES_KANBAN_RUN_ID"])
    credential = PooledCredential(
        provider=provider, id="key-1", label="key-1", auth_type="api_key", priority=0,
        source="test", access_token="test-token",
        last_status=STATUS_EXHAUSTED, last_error_reset_at=float(reset_at),
    )
    agent = SimpleNamespace(_credential_pool=CredentialPool(provider, [credential]))
    if runtime is not None:
        agent.provider = runtime
    worker = SimpleNamespace(agent=agent)
    with _as_worker_process(monkeypatch, env):
        cli._record_kanban_rate_limit_reset(worker, kb.KANBAN_RATE_LIMIT_EXIT_CODE)

    assert (kb.get_run(conn, run_id).metadata or {}).get("rate_limit_reset_at") == reset_at
    default_db = kb.kanban_db_path(board=kb.DEFAULT_BOARD)
    assert default_db != kb.kanban_db_path(board=BOARD)
    if default_db.exists():
        with closing(kb.connect(default_db)) as default_conn:
            assert default_conn.execute(
                "SELECT COUNT(*) FROM tasks WHERE id = ?", (task_id,),
            ).fetchone()[0] == 0


def _worker_session_on(conn, task_id, provider):
    """The claimed run's worker session, linked by its first heartbeat, served by ``provider``.

    The session lives in the assignee profile's own store under the test root,
    where the kernel reads the run's runtime receipt when it closes the run.
    """
    from hermes_cli.profiles import get_profile_dir
    from hermes_state import SessionDB

    task = kb.get_task(conn, task_id)
    session_id = f"worker-session-{task.current_run_id}"
    db = SessionDB(db_path=get_profile_dir(task.assignee) / "state.db")
    try:
        db.create_session(session_id, source="kanban", model="served-model")
        db.update_token_counts(
            session_id, input_tokens=10, output_tokens=4, model="served-model",
            billing_provider=provider, api_call_count=1,
        )
    finally:
        db.close()
    assert kb.heartbeat_worker(conn, task_id, session_id=session_id)


def _pin_onto_the_builder_route(conn, task_id):
    """Pin an unpinned builder card onto its role's policy route, as the owner workspace does.

    The card is classified first, as rows from older builds were; returns the
    provider it was pinned onto.
    """
    conn.execute("UPDATE tasks SET execution_tier = 'deep' WHERE id = ?", (task_id,))
    conn.commit()
    route = model_policy.task_assignment_for("raphael-builder", "anthropic", "deep")
    pinned = kb.pin_effective_task_routes(
        conn, task_ids=[task_id], model=route.model, provider=route.provider,
        reasoning_effort=route.reasoning_effort,
    )
    assert pinned == [task_id]
    assert kb.get_task(conn, task_id).provider_override == route.provider
    return route.provider


def _card_naming_anthropic_as(conn, named):
    """A card whose route names the anthropic provider as ``named``: an alias, or another case.

    The card keeps the name as it was entered; its worker, handed that name as
    ``--provider``, resolves it and runs on anthropic's credential pool.
    """
    task_id = kb.create_task(
        conn, title="aliased card", assignee="worker", model_override="m-1", provider_override=named,
    )
    assert kb.get_task(conn, task_id).provider_override == named
    assert resolve_provider(named) == "anthropic"
    return task_id


# Custom endpoints the worker's profile configures, each with a key in its own
# credential pool: one named ``Local LLM``, one whose name itself starts with
# ``custom:``.
_CUSTOM_ENDPOINTS = """\
custom_providers:
  - name: Local LLM
    base_url: http://127.0.0.1:9/v1
  - name: "custom:box"
    base_url: http://127.0.0.1:10/v1
"""


# A legacy endpoint named like the built-in ``anthropic``: ``custom:anthropic``
# routes to it, bare ``anthropic`` to the built-in.
_ENDPOINT_NAMED_ANTHROPIC = """\
custom_providers:
  - name: anthropic
    base_url: http://127.0.0.1:11/v1
"""

# Keyed endpoints whose display names differ from their keys; the worker's pool
# for each is its key.
_KEYED_ENDPOINTS = """\
providers:
  my-box:
    name: Local Box
    base_url: http://127.0.0.1:12/v1
  other-box:
    name: Other Box
    base_url: http://127.0.0.1:13/v1
"""
_KEYED_POOLS = ("my-box", "other-box", "anthropic")


def _worker_route(monkeypatch, named, *, config=_CUSTOM_ENDPOINTS, pools=("custom:local-llm", "custom:custom:box")):
    """``(runtime provider, credential pool)`` a worker handed ``named`` as ``--provider`` runs on.

    Resolved by the worker's own resolver under the assignee's profile home in
    the test root, where ``config`` (by default :data:`_CUSTOM_ENDPOINTS`) is
    configured and each of ``pools`` holds a placeholder key; Vertex's token is
    stubbed, never fetched. The runtime provider is the one the run's runtime
    receipt names.
    """
    from hermes_cli.profiles import get_profile_dir

    home = get_profile_dir("worker")
    (home / "config.yaml").write_text(config)
    key = {
        "id": "key-1", "label": "key-1", "auth_type": "api_key", "priority": 0,
        "source": "manual", "access_token": "test-token", "last_status": "ok",
    }
    (home / "auth.json").write_text(json.dumps({
        "version": 1, "credential_pool": {pool: [key] for pool in pools},
    }))
    with monkeypatch.context() as worker:
        worker.setenv("HERMES_HOME", str(home))
        worker.setattr(
            vertex_adapter, "get_vertex_config",
            lambda *_args, **_kwargs: ("vertex-token", "https://vertex.invalid/v1"),
        )
        runtime = resolve_runtime_provider(requested=named)
    return runtime["provider"], getattr(runtime.get("credential_pool"), "provider", None)


def _card_routed_to(conn, named):
    """A card whose route names its provider as ``named``, kept as it was entered."""
    task_id = kb.create_task(
        conn, title="routed card", assignee="worker", model_override="m-1", provider_override=named,
    )
    assert kb.get_task(conn, task_id).provider_override == named
    return task_id


def _resume_at(run):
    return kb.recorded_rate_limit_reset(run.metadata, anchor=run.ended_at)


def _spawn_recorder(spawned):
    def spawn(task, workspace, **_kwargs):
        spawned.append(task.id)
        return 9999

    return spawn


def _tick(conn, spawn_fn):
    return kb.dispatch_once(conn, spawn_fn=spawn_fn, board=BOARD)


def _latest_run(conn, task_id):
    return max(kb.list_runs(conn, task_id), key=lambda run: run.id)


@pytest.mark.parametrize("lane", ["ready", "review"])
def test_rate_limited_run_without_a_worker_reset_stays_held_past_five_minutes(conn, clock, lane):
    task_id = _card_in(conn, clock, lane)
    run = _rate_limited_stop(conn, task_id, lane=lane)
    assert kb.get_task(conn, task_id).status == lane

    resume_at = _resume_at(run)
    assert resume_at is not None and resume_at > run.ended_at + COOLDOWN

    clock["t"] = run.ended_at + COOLDOWN + 1
    assert kb.check_respawn_guard(conn, task_id, lane=lane) == "rate_limit_cooldown"
    spawned = []
    result = _tick(conn, _spawn_recorder(spawned))
    assert (task_id, "rate_limit_cooldown") in result.respawn_guarded
    assert spawned == []

    clock["t"] = resume_at
    assert kb.check_respawn_guard(conn, task_id, lane=lane) is None


def test_consecutive_stops_on_one_provider_double_the_hold_up_to_six_hours(conn, clock):
    task_id = kb.create_task(
        conn, title="streak", assignee="worker", model_override="m-1", provider_override="prov-a",
    )
    waits = []
    for stop in range(8):
        run = _rate_limited_stop(conn, task_id, pid=5000 + stop)
        assert (run.metadata or {}).get("rate_limit_reset_source") == "kernel"
        waits.append(_resume_at(run) - run.ended_at)
        clock["t"] = run.ended_at + waits[-1]
    assert waits == [600, 1200, 2400, 4800, 9600, 19200, SIX_HOURS, SIX_HOURS]


def test_a_reset_the_worker_recorded_wins_over_the_kernel_estimate(conn, clock, monkeypatch):
    task_id = _card_in(conn, clock, "ready")
    run_id = _start_worker(conn, task_id, 4500)
    reset_at = T0 + 2 * 3600
    _record_as_worker(monkeypatch, conn, task_id, reset_at, provider="anthropic")
    _worker_exits(conn, 4500, kb.KANBAN_RATE_LIMIT_EXIT_CODE)

    run = kb.get_run(conn, run_id)
    assert _resume_at(run) == reset_at
    assert run.metadata.get("rate_limit_reset_source") != "kernel"
    clock["t"] = reset_at - 1
    assert kb.check_respawn_guard(conn, task_id) == "rate_limit_cooldown"


@pytest.mark.parametrize("lane", ["ready", "review"])
def test_ten_ticks_on_a_held_card_leave_one_event_and_an_unchanged_revision(conn, clock, lane):
    task_id = _card_in(conn, clock, lane)
    run = _rate_limited_stop(conn, task_id, lane=lane)

    spawned, revisions = [], []
    for tick in range(10):
        clock["t"] = run.ended_at + 20 * (tick + 1)
        result = _tick(conn, _spawn_recorder(spawned))
        assert (task_id, "rate_limit_cooldown") in result.respawn_guarded
        revisions.append(kb.task_event_revision(conn, task_id))

    guarded = [event for event in kb.list_events(conn, task_id) if event.kind == "respawn_guarded"]
    assert len(guarded) == 1
    assert guarded[0].payload["reason"] == "rate_limit_cooldown"
    assert guarded[0].payload["resume_at"] == _resume_at(run)
    assert set(revisions) == {revisions[0]}
    assert spawned == []


def test_a_later_stop_is_announced_again(conn, clock):
    task_id = _card_in(conn, clock, "ready")
    first = _rate_limited_stop(conn, task_id, pid=4600)
    first_resume = _resume_at(first)
    assert first_resume is not None
    clock["t"] = first.ended_at + 60
    _tick(conn, _spawn_recorder([]))
    _tick(conn, _spawn_recorder([]))

    clock["t"] = first_resume
    second = _rate_limited_stop(conn, task_id, pid=4601)
    clock["t"] = second.ended_at + 60
    _tick(conn, _spawn_recorder([]))
    _tick(conn, _spawn_recorder([]))

    guarded = [event for event in kb.list_events(conn, task_id) if event.kind == "respawn_guarded"]
    assert [event.payload["resume_at"] for event in guarded] == [first_resume, _resume_at(second)]


def test_quota_stop_then_requested_changes_is_no_sign_in_blocker_but_a_sign_in_crash_is(conn, clock):
    task_id = _card_in(conn, clock, "ready")
    stop = _rate_limited_stop(conn, task_id)
    # The stop's failure message carries no quota or sign-in wording.
    assert not kb._RESPAWN_BLOCKER_RE.search(kb.get_task(conn, task_id).last_failure_error or "")

    clock["t"] = stop.ended_at + SIX_HOURS + 1
    work_run = _start_worker(conn, task_id, 4700)
    assert kb.request_review(
        conn, task_id, summary="implemented", reviewer="reviewer", expected_run_id=work_run,
    )
    assert kb.check_respawn_guard(conn, task_id, lane="review") != "blocker_auth"
    clock["t"] += 60
    review_run = _start_worker(conn, task_id, 4701, lane="review")
    ok, _message = kb.request_changes(conn, task_id, reason="tighten the tests", expected_run_id=review_run)
    assert ok
    assert kb.check_respawn_guard(conn, task_id) != "blocker_auth"

    def sign_in_refused(task, workspace, **_kwargs):
        raise RuntimeError("401 Unauthorized: invalid API key")

    clock["t"] += 2 * 3600
    _tick(conn, sign_in_refused)
    assert _latest_run(conn, task_id).outcome in {"spawn_failed", "gave_up"}
    assert kb.check_respawn_guard(conn, task_id) == "blocker_auth"


def test_reset_under_one_provider_stops_holding_after_pin_effective_task_routes_moves_the_card(
    conn, clock, monkeypatch,
):
    # An owner card that ran on its role's former route before it was
    # classified, as rows from older builds did: its worker's credential pool
    # was openrouter's. The owner workspace pins such rows onto the policy
    # route, which names another provider.
    task_id = kb.create_task(conn, title="builder card", assignee="raphael-builder")
    _start_worker(conn, task_id, 4800)
    _record_as_worker(monkeypatch, conn, task_id, T0 + 3 * 3600, provider="openrouter")
    _worker_exits(conn, 4800, kb.KANBAN_RATE_LIMIT_EXIT_CODE)

    clock["t"] = T0 + 3600
    assert kb.check_respawn_guard(conn, task_id) == "rate_limit_cooldown"
    assert _pin_onto_the_builder_route(conn, task_id) == "anthropic"
    assert kb.check_respawn_guard(conn, task_id) is None


def test_a_worker_reset_keeps_holding_after_pin_effective_task_routes_pins_the_provider_it_ran_on(
    conn, clock, monkeypatch,
):
    # Before a role's route changes, the owner workspace pins every unpinned
    # held card onto the role's current route: the provider the card ran on.
    task_id = kb.create_task(conn, title="builder card", assignee="raphael-builder")
    run_id = _start_worker(conn, task_id, 4810)
    reset_at = T0 + 3 * 3600
    _record_as_worker(monkeypatch, conn, task_id, reset_at, provider="anthropic")
    _worker_exits(conn, 4810, kb.KANBAN_RATE_LIMIT_EXIT_CODE)

    clock["t"] = T0 + 3600
    assert _pin_onto_the_builder_route(conn, task_id) == "anthropic"
    assert kb.check_respawn_guard(conn, task_id) == "rate_limit_cooldown"
    clock["t"] = reset_at - 1
    assert kb.check_respawn_guard(conn, task_id) == "rate_limit_cooldown"
    clock["t"] = reset_at
    assert kb.check_respawn_guard(conn, task_id) is None
    assert kb.get_run(conn, run_id).metadata["rate_limit_reset_provider"] == "anthropic"


def test_a_kernel_reset_keeps_holding_after_the_card_is_pinned_onto_the_provider_its_receipt_names(
    conn, clock,
):
    # The worker recorded no reset, and the card named no provider: the run's
    # runtime receipt is what says which provider it ran on.
    task_id = kb.create_task(conn, title="builder card", assignee="raphael-builder")
    run_id = _start_worker(conn, task_id, 4820)
    _worker_session_on(conn, task_id, "anthropic")
    _worker_exits(conn, 4820, kb.KANBAN_RATE_LIMIT_EXIT_CODE)
    run = kb.get_run(conn, run_id)
    assert run.metadata["runtime_receipt"]["provider"] == "anthropic"
    assert run.metadata["rate_limit_reset_source"] == "kernel"
    resume_at = _resume_at(run)
    assert resume_at > run.ended_at + COOLDOWN

    clock["t"] = run.ended_at + COOLDOWN + 1
    assert _pin_onto_the_builder_route(conn, task_id) == "anthropic"
    assert kb.check_respawn_guard(conn, task_id) == "rate_limit_cooldown"
    clock["t"] = resume_at
    assert kb.check_respawn_guard(conn, task_id) is None
    assert run.metadata["rate_limit_reset_provider"] == "anthropic"


def test_a_reset_kept_without_a_known_provider_holds_on_a_card_that_names_one(conn, clock):
    # No worker reset, no runtime receipt and no provider on the card: the
    # kernel's reset is kept, but for no known provider.
    task_id = kb.create_task(conn, title="unrouted card", assignee="worker")
    run = _rate_limited_stop(conn, task_id, pid=5200)
    assert run.metadata["rate_limit_reset_source"] == "kernel"
    assert "rate_limit_reset_provider" in run.metadata
    assert run.metadata["rate_limit_reset_provider"] is None
    resume_at = _resume_at(run)
    assert resume_at > run.ended_at + COOLDOWN

    kb.set_model_override(conn, task_id, "m-1", provider="prov-a")
    clock["t"] = run.ended_at + COOLDOWN + 1
    assert kb.check_respawn_guard(conn, task_id) == "rate_limit_cooldown"
    clock["t"] = resume_at
    assert kb.check_respawn_guard(conn, task_id) is None


def test_reset_under_one_provider_stops_holding_after_the_card_is_moved_to_another(
    conn, clock, monkeypatch,
):
    # Only a move between two known built-in providers ends a hold early.
    task_id = kb.create_task(
        conn, title="moved card", assignee="worker", model_override="m-1", provider_override="anthropic",
    )
    _start_worker(conn, task_id, 4900)
    _record_as_worker(monkeypatch, conn, task_id, T0 + 3 * 3600, provider="anthropic")
    _worker_exits(conn, 4900, kb.KANBAN_RATE_LIMIT_EXIT_CODE)

    clock["t"] = T0 + 3600
    kb.set_model_override(conn, task_id, "m-1", provider="anthropic")
    assert kb.check_respawn_guard(conn, task_id) == "rate_limit_cooldown"
    kb.set_model_override(conn, task_id, "m-2", provider="openai-codex")
    assert kb.check_respawn_guard(conn, task_id) is None


def test_a_reset_stored_before_providers_were_kept_holds_as_today(conn, clock):
    task_id = kb.create_task(
        conn, title="older board", assignee="worker", model_override="m-1", provider_override="prov-a",
    )
    run = _rate_limited_stop(conn, task_id, pid=5100)
    reset_at = run.ended_at + 2 * 3600
    conn.execute(
        "UPDATE task_runs SET metadata = ? WHERE id = ?",
        (json.dumps({"rate_limit_reset_at": reset_at}), run.id),
    )
    conn.commit()

    kb.set_model_override(conn, task_id, "m-2", provider="prov-b")
    clock["t"] = reset_at - 1
    assert kb.check_respawn_guard(conn, task_id) == "rate_limit_cooldown"
    clock["t"] = reset_at
    assert kb.check_respawn_guard(conn, task_id) is None


@pytest.mark.parametrize("named", ["claude", "Anthropic"])
def test_a_worker_reset_holds_a_card_that_names_its_provider_by_an_alias_or_in_another_case(
    conn, clock, monkeypatch, named,
):
    task_id = _card_naming_anthropic_as(conn, named)
    run_id = _start_worker(conn, task_id, 4910)
    reset_at = T0 + 3 * 3600
    _record_as_worker(monkeypatch, conn, task_id, reset_at, provider="anthropic")
    _worker_exits(conn, 4910, kb.KANBAN_RATE_LIMIT_EXIT_CODE)
    run = kb.get_run(conn, run_id)
    assert _resume_at(run) == reset_at

    clock["t"] = run.ended_at + 3600
    assert kb.check_respawn_guard(conn, task_id) == "rate_limit_cooldown"
    clock["t"] = reset_at - 1
    assert kb.check_respawn_guard(conn, task_id) == "rate_limit_cooldown"
    clock["t"] = reset_at
    assert kb.check_respawn_guard(conn, task_id) is None


@pytest.mark.parametrize("named", ["claude", "Anthropic"])
def test_a_kernel_reset_holds_a_card_that_names_its_provider_by_an_alias_or_in_another_case(
    conn, clock, named,
):
    # The worker recorded no reset; the run's runtime receipt names the
    # provider it ran on.
    task_id = _card_naming_anthropic_as(conn, named)
    run_id = _start_worker(conn, task_id, 4920)
    _worker_session_on(conn, task_id, "anthropic")
    _worker_exits(conn, 4920, kb.KANBAN_RATE_LIMIT_EXIT_CODE)
    run = kb.get_run(conn, run_id)
    assert run.metadata["runtime_receipt"]["provider"] == "anthropic"
    assert run.metadata["rate_limit_reset_source"] == "kernel"
    assert _resume_at(run) == run.ended_at + 2 * COOLDOWN

    clock["t"] = run.ended_at + COOLDOWN + 1
    assert kb.check_respawn_guard(conn, task_id) == "rate_limit_cooldown"
    clock["t"] = run.ended_at + 2 * COOLDOWN - 1
    assert kb.check_respawn_guard(conn, task_id) == "rate_limit_cooldown"
    clock["t"] = run.ended_at + 2 * COOLDOWN
    assert kb.check_respawn_guard(conn, task_id) is None


@pytest.mark.parametrize("named", ["claude", "Anthropic"])
def test_stops_on_one_provider_written_two_ways_double_the_hold(conn, clock, named):
    # The first stop knows its provider only from the card, the second from
    # its runtime receipt: the same provider, written two ways.
    task_id = _card_naming_anthropic_as(conn, named)
    first = _rate_limited_stop(conn, task_id, pid=4930)
    assert first.metadata["rate_limit_reset_source"] == "kernel"
    assert _resume_at(first) == first.ended_at + 2 * COOLDOWN

    clock["t"] = _resume_at(first)
    run_id = _start_worker(conn, task_id, 4931)
    _worker_session_on(conn, task_id, "anthropic")
    _worker_exits(conn, 4931, kb.KANBAN_RATE_LIMIT_EXIT_CODE)
    second = kb.get_run(conn, run_id)
    assert second.outcome == "rate_limited"
    assert second.metadata["runtime_receipt"]["provider"] == "anthropic"
    assert second.metadata["rate_limit_reset_source"] == "kernel"
    assert _resume_at(second) == second.ended_at + 4 * COOLDOWN

    clock["t"] = second.ended_at + 4 * COOLDOWN - 1
    assert kb.check_respawn_guard(conn, task_id) == "rate_limit_cooldown"
    clock["t"] = second.ended_at + 4 * COOLDOWN
    assert kb.check_respawn_guard(conn, task_id) is None
    # Both stops keep the one id, whichever way the provider was written.
    assert [run.metadata["rate_limit_reset_provider"] for run in (first, second)] == [
        "anthropic", "anthropic",
    ]


@pytest.mark.parametrize(
    ("named", "pool", "receipt"),
    [
        ("custom:local-llm", "custom:local-llm", "custom"),
        ("local-llm", "custom:local-llm", "custom"),
        ("Local LLM", "custom:local-llm", "custom"),
        # A custom endpoint whose own name starts with ``custom:``.
        ("custom:box", "custom:custom:box", "custom"),
        ("box", "custom:custom:box", "custom"),
        # Another of Vertex's names: no pool, a receipt naming ``vertex``.
        ("vertexai", None, "vertex"),
    ],
)
def test_a_card_holds_until_its_reset_whatever_name_its_worker_and_receipt_give_its_provider(
    conn, clock, monkeypatch, named, pool, receipt,
):
    # Handed the card's name as ``--provider``, the worker runs on ``pool`` and
    # its runtime receipt names ``receipt``.
    assert _worker_route(monkeypatch, named) == (receipt, pool)

    if pool is not None:
        task_id = _card_routed_to(conn, named)
        run_id = _start_worker(conn, task_id, 4940)
        reset_at = T0 + 3 * 3600
        _record_as_worker(monkeypatch, conn, task_id, reset_at, provider=pool)
        _worker_exits(conn, 4940, kb.KANBAN_RATE_LIMIT_EXIT_CODE)
        run = kb.get_run(conn, run_id)
        assert _resume_at(run) == reset_at

        clock["t"] = run.ended_at + 3600
        assert kb.check_respawn_guard(conn, task_id) == "rate_limit_cooldown"
        clock["t"] = reset_at - 1
        assert kb.check_respawn_guard(conn, task_id) == "rate_limit_cooldown"
        clock["t"] = reset_at
        assert kb.check_respawn_guard(conn, task_id) is None

    # No worker reset: the kernel books its own, beside the receipt's provider.
    task_id = _card_routed_to(conn, named)
    run_id = _start_worker(conn, task_id, 4941)
    _worker_session_on(conn, task_id, receipt)
    _worker_exits(conn, 4941, kb.KANBAN_RATE_LIMIT_EXIT_CODE)
    run = kb.get_run(conn, run_id)
    assert run.metadata["runtime_receipt"]["provider"] == receipt
    assert run.metadata["rate_limit_reset_source"] == "kernel"
    assert _resume_at(run) == run.ended_at + 2 * COOLDOWN

    clock["t"] = run.ended_at + COOLDOWN + 1
    assert kb.check_respawn_guard(conn, task_id) == "rate_limit_cooldown"
    clock["t"] = run.ended_at + 2 * COOLDOWN - 1
    assert kb.check_respawn_guard(conn, task_id) == "rate_limit_cooldown"
    clock["t"] = run.ended_at + 2 * COOLDOWN
    assert kb.check_respawn_guard(conn, task_id) is None


def _stopped_after_worker_reset(monkeypatch, conn, task_id, pid, reset_at, *, pool):
    """A rate-limited run of the card whose custom-endpoint worker recorded ``reset_at`` on ``pool``."""
    run_id = _start_worker(conn, task_id, pid)
    _record_as_worker(monkeypatch, conn, task_id, reset_at, provider=pool, runtime="custom")
    _worker_exits(conn, pid, kb.KANBAN_RATE_LIMIT_EXIT_CODE)
    run = kb.get_run(conn, run_id)
    assert run.outcome == "rate_limited"
    assert _resume_at(run) == reset_at
    return run


def test_a_reset_on_a_custom_endpoint_named_anthropic_holds_until_its_reset_even_once_the_card_moves_to_the_built_in(
    conn, clock, monkeypatch,
):
    pools = ("custom:anthropic", "anthropic")
    route = {
        named: _worker_route(monkeypatch, named, config=_ENDPOINT_NAMED_ANTHROPIC, pools=pools)
        for named in ("custom:anthropic", "anthropic")
    }
    assert route == {"custom:anthropic": ("custom", "custom:anthropic"), "anthropic": ("anthropic", "anthropic")}

    reset_at = T0 + 3 * 3600
    moving, staying = _card_routed_to(conn, "custom:anthropic"), _card_routed_to(conn, "custom:anthropic")
    moved_run = _stopped_after_worker_reset(monkeypatch, conn, moving, 4960, reset_at, pool="custom:anthropic")
    _stopped_after_worker_reset(monkeypatch, conn, staying, 4961, reset_at, pool="custom:anthropic")

    kb.set_model_override(conn, moving, "m-1", provider="anthropic")
    clock["t"] = T0 + 3600
    assert kb.get_run(conn, moved_run.id).metadata["rate_limit_reset_provider"] != "anthropic"
    for task_id in (moving, staying):
        assert kb.check_respawn_guard(conn, task_id) == "rate_limit_cooldown"
    clock["t"] = reset_at - 1
    for task_id in (moving, staying):
        assert kb.check_respawn_guard(conn, task_id) == "rate_limit_cooldown"
    clock["t"] = reset_at
    for task_id in (moving, staying):
        assert kb.check_respawn_guard(conn, task_id) is None


@pytest.mark.parametrize("moved_to", ["anthropic", "Other Box"])
def test_a_reset_on_a_keyed_custom_endpoint_holds_its_card_until_it_passes_even_one_moved_off_it(
    conn, clock, monkeypatch, moved_to,
):
    assert _worker_route(monkeypatch, "Local Box", config=_KEYED_ENDPOINTS, pools=_KEYED_POOLS) == (
        "custom", "my-box",
    )
    assert _worker_route(monkeypatch, "Other Box", config=_KEYED_ENDPOINTS, pools=_KEYED_POOLS) == (
        "custom", "other-box",
    )

    reset_at = T0 + 3 * 3600
    staying, moving = _card_routed_to(conn, "Local Box"), _card_routed_to(conn, "Local Box")
    _stopped_after_worker_reset(monkeypatch, conn, staying, 4962, reset_at, pool="my-box")
    _stopped_after_worker_reset(monkeypatch, conn, moving, 4963, reset_at, pool="my-box")

    clock["t"] = T0 + 3600
    assert kb.check_respawn_guard(conn, staying) == "rate_limit_cooldown"
    assert kb.check_respawn_guard(conn, moving) == "rate_limit_cooldown"
    kb.set_model_override(conn, moving, "m-2", provider=moved_to)
    assert kb.check_respawn_guard(conn, moving) == "rate_limit_cooldown"
    clock["t"] = reset_at - 1
    for task_id in (moving, staying):
        assert kb.check_respawn_guard(conn, task_id) == "rate_limit_cooldown"
    clock["t"] = reset_at
    for task_id in (moving, staying):
        assert kb.check_respawn_guard(conn, task_id) is None


def test_a_keyed_custom_endpoint_is_one_provider_whichever_of_its_names_the_card_gives(
    conn, clock, monkeypatch,
):
    names = ("Local Box", "local-box", "my-box", "custom:my-box", "custom:local-box")
    for named in names:
        assert _worker_route(monkeypatch, named, config=_KEYED_ENDPOINTS, pools=_KEYED_POOLS) == (
            "custom", "my-box",
        ), named

    reset_at = T0 + 3 * 3600
    task_id = _card_routed_to(conn, "my-box")
    _stopped_after_worker_reset(monkeypatch, conn, task_id, 4964, reset_at, pool="my-box")

    clock["t"] = T0 + 3600
    for named in names:
        kb.set_model_override(conn, task_id, "m-1", provider=named)
        assert kb.check_respawn_guard(conn, task_id) == "rate_limit_cooldown", named
    clock["t"] = reset_at
    assert kb.check_respawn_guard(conn, task_id) is None


def test_stops_on_a_keyed_custom_endpoint_streak_only_under_one_name_for_it(
    conn, clock, monkeypatch,
):
    # The first stop knows its provider from the worker's pool (``my-box``);
    # the second, with a receipt naming only ``custom``, from the card
    # (``Local Box``). Without the profile's configuration the two names are
    # two endpoints, so the second stop starts a streak of its own.
    assert _worker_route(monkeypatch, "Local Box", config=_KEYED_ENDPOINTS, pools=_KEYED_POOLS) == (
        "custom", "my-box",
    )
    task_id = _card_routed_to(conn, "Local Box")
    first = _stopped_after_worker_reset(monkeypatch, conn, task_id, 4965, T0 + 3600, pool="my-box")

    clock["t"] = T0 + 3600
    run_id = _start_worker(conn, task_id, 4966)
    _worker_session_on(conn, task_id, "custom")
    _worker_exits(conn, 4966, kb.KANBAN_RATE_LIMIT_EXIT_CODE)
    second = kb.get_run(conn, run_id)
    assert second.outcome == "rate_limited"
    assert second.metadata["runtime_receipt"]["provider"] == "custom"
    assert second.metadata["rate_limit_reset_source"] == "kernel"
    assert _resume_at(second) == second.ended_at + 2 * COOLDOWN
    assert second.metadata["rate_limit_reset_provider"] != first.metadata["rate_limit_reset_provider"]


@pytest.mark.parametrize(
    ("config", "named", "pool"),
    [
        # A key the alias table reads as the built-in ``anthropic``.
        ("providers:\n  claude:\n    base_url: http://127.0.0.1:14/v1\n", "claude", "claude"),
        # A key that is the built-in's own id, on an endpoint named otherwise.
        (
            "providers:\n  anthropic:\n    name: Work Proxy\n    base_url: http://127.0.0.1:15/v1\n",
            "Work Proxy", "anthropic",
        ),
    ],
)
def test_a_reset_from_a_custom_endpoint_pool_keyed_like_a_built_in_holds_its_cards_until_the_reset(
    conn, clock, monkeypatch, config, named, pool,
):
    pools = (pool, "anthropic")
    assert _worker_route(monkeypatch, named, config=config, pools=pools) == ("custom", pool)
    assert _worker_route(monkeypatch, "anthropic", config=config, pools=pools) == ("anthropic", "anthropic")

    reset_at = T0 + 3 * 3600
    staying, moving = _card_routed_to(conn, named), _card_routed_to(conn, named)
    _stopped_after_worker_reset(monkeypatch, conn, staying, 4967, reset_at, pool=pool)
    _stopped_after_worker_reset(monkeypatch, conn, moving, 4968, reset_at, pool=pool)

    clock["t"] = T0 + 3600
    assert kb.check_respawn_guard(conn, moving) == "rate_limit_cooldown"
    kb.set_model_override(conn, moving, "m-1", provider="anthropic")
    for task_id in (moving, staying):
        assert kb.check_respawn_guard(conn, task_id) == "rate_limit_cooldown"
    clock["t"] = reset_at - 1
    for task_id in (moving, staying):
        assert kb.check_respawn_guard(conn, task_id) == "rate_limit_cooldown"
    clock["t"] = reset_at
    for task_id in (moving, staying):
        assert kb.check_respawn_guard(conn, task_id) is None


# What a profiles dir under the worker's own home would say: that ``Local Box``
# is another endpoint. The assignee's profile is the one under the root.
_DECOY_ENDPOINTS = """\
providers:
  decoy-box:
    name: Local Box
    base_url: http://127.0.0.1:16/v1
"""


def test_booking_and_the_guard_under_the_worker_env_read_the_assignee_profile_under_the_root(
    conn, clock, monkeypatch,
):
    assert _worker_route(monkeypatch, "Local Box", config=_KEYED_ENDPOINTS, pools=_KEYED_POOLS) == (
        "custom", "my-box",
    )
    task_id = _card_routed_to(conn, "Local Box")
    run_id = _start_worker(conn, task_id, 4969)
    env = _dispatcher_env(monkeypatch, conn, task_id)
    root = Path(os.environ["HERMES_HOME"])
    assert env["HERMES_HOME"] == str(root / "profiles" / "worker")
    assert env["HERMES_KANBAN_TASK"] == task_id
    assert env["HERMES_KANBAN_DB"] == str(kb.kanban_db_path(board=BOARD))
    assert env["HERMES_KANBAN_BOARD"] == BOARD
    decoy = Path(env["HERMES_HOME"]) / "profiles" / "worker"
    decoy.mkdir(parents=True)
    (decoy / "config.yaml").write_text(_DECOY_ENDPOINTS)

    reset_at = T0 + 3 * 3600
    _record_as_worker(monkeypatch, conn, task_id, reset_at, provider="my-box", runtime="custom")
    with _as_worker_process(monkeypatch, env):
        _worker_exits(conn, 4969, kb.KANBAN_RATE_LIMIT_EXIT_CODE)
        run = kb.get_run(conn, run_id)
        assert run.outcome == "rate_limited"
        assert _resume_at(run) == reset_at
        clock["t"] = T0 + 3600
        assert kb.check_respawn_guard(conn, task_id) == "rate_limit_cooldown"
        clock["t"] = reset_at
        assert kb.check_respawn_guard(conn, task_id) is None
    assert sorted(path.name for path in decoy.iterdir()) == ["config.yaml"]


def test_a_custom_endpoint_reset_keeps_no_endpoint_name_and_holds_whatever_the_config_says(
    conn, clock, monkeypatch,
):
    from hermes_cli.profiles import get_profile_dir

    task_id = _card_routed_to(conn, "Local Box")
    reset_at = T0 + 3 * 3600
    config = get_profile_dir("worker") / "config.yaml"
    config.write_text("providers: [not, closed\n")
    run = _stopped_after_worker_reset(monkeypatch, conn, task_id, 4970, reset_at, pool="my-box")
    kept = run.metadata["rate_limit_reset_provider"]
    assert kept and "my-box" not in kept and "box" not in kept

    clock["t"] = T0 + 3600
    kb.set_model_override(conn, task_id, "m-2", provider="prov-b")
    assert kb.check_respawn_guard(conn, task_id) == "rate_limit_cooldown"
    clock["t"] = reset_at
    assert kb.check_respawn_guard(conn, task_id) is None
    assert config.read_text() == "providers: [not, closed\n"
    assert sorted(path.name for path in config.parent.iterdir() if path.name.startswith("config")) == ["config.yaml"]


def test_a_provider_a_plugin_registers_holds_like_a_custom_endpoint(conn, clock, monkeypatch):
    from hermes_cli import auth

    # A provider a model-provider plugin adds to the registry is no built-in.
    monkeypatch.setitem(auth.PROVIDER_REGISTRY, "placeholder-plugin", auth.PROVIDER_REGISTRY["anthropic"])
    reset_at = T0 + 3 * 3600
    task_id = _card_routed_to(conn, "placeholder-plugin")
    run_id = _start_worker(conn, task_id, 4980)
    _record_as_worker(monkeypatch, conn, task_id, reset_at, provider="placeholder-plugin")
    _worker_exits(conn, 4980, kb.KANBAN_RATE_LIMIT_EXIT_CODE)
    assert kb.get_run(conn, run_id).outcome == "rate_limited"

    kb.set_model_override(conn, task_id, "m-2", provider="anthropic")
    clock["t"] = T0 + 3600
    assert kb.check_respawn_guard(conn, task_id) == "rate_limit_cooldown"
    clock["t"] = reset_at
    assert kb.check_respawn_guard(conn, task_id) is None


def test_the_later_of_two_runs_ending_in_one_second_is_the_latest(conn, clock):
    task_id = kb.create_task(conn, title="tied card", assignee="worker")
    _start_worker(conn, task_id, 4985)
    _worker_exits(conn, 4985, 1)
    stop = _rate_limited_stop(conn, task_id, pid=4986)
    crashed = min(kb.list_runs(conn, task_id), key=lambda run: run.id)
    assert crashed.outcome == "crashed" and crashed.ended_at == stop.ended_at

    clock["t"] = stop.ended_at + COOLDOWN + 1
    assert kb.check_respawn_guard(conn, task_id) == "rate_limit_cooldown"


def test_a_stop_streak_counts_one_named_endpoint_and_never_merges_two(conn, clock):
    same = _card_routed_to(conn, "custom:local-llm")
    first = _rate_limited_stop(conn, same, pid=4991)
    assert _resume_at(first) == first.ended_at + 2 * COOLDOWN
    clock["t"] = _resume_at(first)
    second = _rate_limited_stop(conn, same, pid=4992)
    assert _resume_at(second) == second.ended_at + 4 * COOLDOWN

    moved = _card_routed_to(conn, "custom:local-llm")
    third = _rate_limited_stop(conn, moved, pid=4993)
    clock["t"] = _resume_at(third)
    kb.set_model_override(conn, moved, "m-1", provider="custom:box")
    fourth = _rate_limited_stop(conn, moved, pid=4994)
    assert _resume_at(fourth) == fourth.ended_at + 2 * COOLDOWN


OTHER_BOARD = "elsewhere"
# The stale-heartbeat sweep's window, as a gateway configures one.
STALE_TIMEOUT = 3600
# The two provider stops by exit code: the kind each is booked as, and the turn
# result a worker's provider stops its turn with.
STOP_KINDS = {
    kb.KANBAN_RATE_LIMIT_EXIT_CODE: "rate_limited",
    KANBAN_PROVIDER_REFUSED_EXIT_CODE: "provider_refused",
}
STOPPED_BY = {
    kb.KANBAN_RATE_LIMIT_EXIT_CODE: {
        "final_response": "", "completed": False, "failed": True,
        "failure_reason": "rate_limit", "error": "placeholder usage limit",
    },
    KANBAN_PROVIDER_REFUSED_EXIT_CODE: {
        "final_response": "", "completed": False, "failed": True,
        "error": "content_policy_blocked: placeholder refusal",
    },
}


@pytest.fixture
def signals(monkeypatch):
    """Every signal a sweep sends a worker; none reaches a process, as each placeholder pid is gone."""
    sent = []

    def kill(pid, sig):
        sent.append((pid, sig))
        raise ProcessLookupError(pid)

    monkeypatch.setattr(kb.os, "kill", kill)
    return sent


def _stopping_worker(conn, task_id, pid, lapse):
    """Claim the card for the worker ``pid``; a claim whose heartbeat is to lapse outlasts it."""
    if lapse != "heartbeat":
        return _start_worker(conn, task_id, pid)
    run_id = _start_worker(conn, task_id, pid, ttl_seconds=3 * STALE_TIMEOUT)
    assert kb.heartbeat_worker(conn, task_id)
    return run_id


def _lapse(conn, clock, task_id, lapse):
    """Let the running card's claim expire, or its heartbeat go stale while its claim holds."""
    task = kb.get_task(conn, task_id)
    if lapse == "claim":
        clock["t"] = task.claim_expires + 1
    elif lapse == "heartbeat":
        clock["t"] = task.last_heartbeat_at + kb._STALE_HEARTBEAT_GAP_SECONDS + 60
        assert clock["t"] < task.claim_expires


def _full_tick(conn, spawned):
    """One whole dispatcher tick, every sweep in its order, the stale-heartbeat sweep on."""
    return kb.dispatch_once(
        conn, spawn_fn=_spawn_recorder(spawned), board=BOARD, stale_timeout_seconds=STALE_TIMEOUT,
    )


def _booked_as_the_stop(conn, task_id, run_id, code):
    """The run ended as the stop ``code`` with no failure counted: a limit holds the card, a refusal blocks it."""
    task, run = kb.get_task(conn, task_id), kb.get_run(conn, run_id)
    assert run.ended_at is not None and run.outcome == STOP_KINDS[code]
    assert task.consecutive_failures == 0 and task.worker_pid is None
    assert not [event for event in kb.list_events(conn, task_id) if event.kind in ("reclaimed", "stale", "crashed")]
    if code == kb.KANBAN_RATE_LIMIT_EXIT_CODE:
        assert task.status == "ready" and _resume_at(run) is not None
        assert kb.check_respawn_guard(conn, task_id) == "rate_limit_cooldown"
    else:
        assert (task.status, task.block_kind) == ("blocked", "capability")


class _StoppedWorker:
    """The slice of HermesCLI a worker's one-shot turn touches; its provider stops the turn with ``result``."""

    def __init__(self, result):
        self.result = result
        self.session_id = "worker-session"
        self.conversation_history = []
        self.console = SimpleNamespace(print=lambda *_args, **_kwargs: None)
        self._active_agent_route_signature = "route"
        self.agent = SimpleNamespace(
            session_id=self.session_id, _credential_pool=None, run_conversation=lambda **_kwargs: result,
        )

    def _claim_active_session(self, _surface, *, stderr=False):
        return True

    def _show_security_advisories(self):
        pass

    def chat(self, query, images=None):
        self._last_turn_result = self.result
        return self.result.get("final_response")

    def _print_exit_summary(self, clear_screen=True):
        pass

    def _ensure_runtime_credentials(self):
        return True

    def _resolve_turn_agent_config(self, _query):
        return {"signature": "route", "model": None, "runtime": None}

    def _init_agent(self, **_kwargs):
        return True


def _reaches_every_board_and_the_root(root, boards):
    """Pinned to its card's board, the worker still reaches every other board and the root registry."""
    assert kb.kanban_home() == root
    assert kb.register_db_path() == root / "kanban" / "board_register.db"
    for board, task_id in boards.items():
        entry = kb.get_register_entry(board)
        assert entry is not None and entry.lifecycle is kb.BoardLifecycle.LIVE
        with closing(kb.connect(db_path=kb.board_dir(board) / "kanban.db")) as board_conn:
            assert kb.get_task(board_conn, task_id) is not None


def _recorded_stops_under(root):
    """Every ``(store, run id, stop)`` recorded under ``root``, read straight from each SQLite file there."""
    found = []
    for path in sorted(root.rglob("*.db")):
        with closing(sqlite3.connect(path)) as store:
            if store.execute(
                "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'task_runs'",
            ).fetchone() is None:
                continue
            for run_id, metadata in store.execute("SELECT id, metadata FROM task_runs ORDER BY id"):
                stop = json.loads(metadata or "{}").get("provider_stop")
                if stop is not None:
                    found.append((path.resolve(), run_id, stop))
    return found


@pytest.mark.parametrize("lapse", ["claim", "heartbeat"])
@pytest.mark.parametrize("code", sorted(STOP_KINDS))
def test_a_stop_the_reaper_saw_is_booked_as_that_stop_after_its_claim_or_heartbeat_lapses(
    conn, clock, signals, code, lapse,
):
    task_id = kb.create_task(conn, title="stopped card", assignee="worker")
    run_id = _stopping_worker(conn, task_id, 6101, lapse)
    _lapse(conn, clock, task_id, lapse)
    kb._record_worker_exit(6101, code << 8)
    spawned = []
    _full_tick(conn, spawned)
    _booked_as_the_stop(conn, task_id, run_id, code)
    assert spawned == [] and signals == []


@pytest.mark.parametrize("lapse", [None, "claim", "heartbeat"])
@pytest.mark.parametrize("quiet", [False, True])
@pytest.mark.parametrize("code", sorted(STOP_KINDS))
def test_after_a_restart_a_stop_the_worker_recorded_on_its_run_is_booked_as_that_stop(
    conn, clock, signals, monkeypatch, tmp_path, code, quiet, lapse,
):
    """The worker records its stop on its way out, under exactly the environment the dispatcher
    builds for its run; a restart then loses the exit the dispatcher reaped.
    """
    root = Path(os.environ["HERMES_HOME"])
    kb.create_board(OTHER_BOARD)
    with closing(kb.connect(board=OTHER_BOARD)) as other_conn:
        elsewhere = kb.create_task(other_conn, title="card elsewhere", assignee="worker")
    task_id = kb.create_task(conn, title="stopped card", assignee="worker")
    run_id = _stopping_worker(conn, task_id, 6120, lapse)
    env = _dispatcher_env(monkeypatch, conn, task_id)
    assert env["HERMES_HOME"] == str(root / "profiles" / "worker")
    assert env["HERMES_KANBAN_TASK"] == task_id
    assert env["HERMES_KANBAN_RUN_ID"] == str(run_id)
    assert env["HERMES_KANBAN_DB"] == str(kb.kanban_db_path(board=BOARD))
    assert env["HERMES_KANBAN_BOARD"] == BOARD

    # The one-shot entry sets this marker on os.environ itself; registering it here undoes that.
    monkeypatch.setenv("HERMES_SINGLE_QUERY_SESSION", "1")
    monkeypatch.setattr(cli, "_should_seed_interactive", lambda *_args: False)
    monkeypatch.setattr(cli, "_collect_query_images", lambda query, _image: (query, []))
    monkeypatch.setattr(cli, "_collect_kanban_task_images", lambda _images: [])
    monkeypatch.setattr(cli, "_finalize_single_query", lambda _worker: None)
    with _as_worker_process(monkeypatch, env):
        assert dict(os.environ) == env
        with pytest.raises(SystemExit) as exited:
            cli._run_single_query_mode(
                _StoppedWorker(STOPPED_BY[code]), f"work kanban task {task_id}", None, quiet, False,
            )
        _reaches_every_board_and_the_root(root, {BOARD: task_id, OTHER_BOARD: elsewhere})
    assert exited.value.code == code
    # Counted straight from every store under the test root: one run, on the claimed board.
    assert _recorded_stops_under(tmp_path) == [
        (kb.kanban_db_path(board=BOARD).resolve(), run_id, STOP_KINDS[code]),
    ]

    monkeypatch.setattr(kb, "_recent_worker_exits", {})  # the restart: no reaped exit survives
    _lapse(conn, clock, task_id, lapse)
    spawned = []
    _full_tick(conn, spawned)
    _booked_as_the_stop(conn, task_id, run_id, code)
    assert spawned == [] and signals == []


@pytest.mark.parametrize(
    ("lapse", "outcome"), [(None, "crashed"), ("claim", "reclaimed"), ("heartbeat", "stale")],
)
def test_a_run_with_no_recorded_stop_and_an_unknown_exit_keeps_todays_booking(
    conn, clock, signals, monkeypatch, lapse, outcome,
):
    monkeypatch.setattr(kb, "_recent_worker_exits", {})
    task_id = kb.create_task(conn, title="crashed card", assignee="worker")
    run_id = _stopping_worker(conn, task_id, 6130, lapse)
    _lapse(conn, clock, task_id, lapse)
    _full_tick(conn, [])
    assert kb.get_run(conn, run_id).outcome == outcome
    assert kb.get_task(conn, task_id).consecutive_failures == (1 if outcome == "crashed" else 0)
    assert signals == ([] if lapse is None else [(6130, signal.SIGTERM)])


def test_the_exit_the_reaper_saw_wins_over_a_recorded_stop(conn, clock, signals, monkeypatch):
    task_id = kb.create_task(conn, title="crashed card", assignee="worker")
    run_id = _start_worker(conn, task_id, 6140)
    with monkeypatch.context() as worker:
        worker.setenv("HERMES_KANBAN_TASK", task_id)
        worker.setenv("HERMES_KANBAN_RUN_ID", str(run_id))
        assert kb.record_run_provider_stop(
            conn, task_id, run_id=run_id, exit_code=kb.KANBAN_RATE_LIMIT_EXIT_CODE,
        )
    kb._record_worker_exit(6140, 1 << 8)
    _full_tick(conn, [])
    assert kb.get_run(conn, run_id).outcome == "crashed"
    assert kb.get_task(conn, task_id).consecutive_failures == 1


def test_the_stop_recorder_writes_only_the_open_run_the_dispatcher_gave_this_worker(conn, monkeypatch):
    mine = kb.create_task(conn, title="my card", assignee="worker")
    other = kb.create_task(conn, title="other card", assignee="worker")
    ended = _start_worker(conn, mine, 6150)
    _worker_exits(conn, 6150, 1)
    given = _start_worker(conn, mine, 6151)
    others = _start_worker(conn, other, 6152)
    assert kb.get_run(conn, ended).ended_at is not None

    def record(task_id, run_id, *, given_run=given, exit_code=kb.KANBAN_RATE_LIMIT_EXIT_CODE):
        """The worker the dispatcher gave ``mine`` and ``given_run`` records a stop on ``run_id``."""
        with monkeypatch.context() as worker:
            worker.setenv("HERMES_KANBAN_TASK", mine)
            worker.setenv("HERMES_KANBAN_RUN_ID", str(given_run))
            return kb.record_run_provider_stop(conn, task_id, run_id=run_id, exit_code=exit_code)

    def runs():
        return [tuple(row) for row in conn.execute("SELECT id, metadata FROM task_runs ORDER BY id")]

    before = runs()
    assert not record(other, others)  # another task's run
    assert not record(mine, others, given_run=others)  # another task's run, even when given
    assert not record(mine, ended, given_run=ended)  # an ended run
    assert not record(mine, given, given_run=ended)  # the open run, not the one given
    for not_a_stop in (0, 1):
        assert not record(mine, given, exit_code=not_a_stop)
    assert runs() == before

    assert record(mine, given, exit_code=KANBAN_PROVIDER_REFUSED_EXIT_CODE)
    assert kb.get_run(conn, given).metadata["provider_stop"] == "provider_refused"
    assert [run for run in runs() if run[0] != given] == [run for run in before if run[0] != given]
