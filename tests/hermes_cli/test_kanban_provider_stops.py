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
its stops streak, as one naming it canonically. A reset the worker records goes
through the worker's own path, under exactly the environment the dispatcher
builds for the claimed run, onto a named board under the test root. The refusal
exit is covered end to end in ``tests/cli/test_kanban_worker_refusal_exit.py``.
"""

from __future__ import annotations

import json
import os
import subprocess
from contextlib import closing
from pathlib import Path
from types import SimpleNamespace

import pytest

import cli
from agent.credential_pool import STATUS_EXHAUSTED, CredentialPool, PooledCredential
from hermes_cli import kanban_db as kb
from hermes_cli.auth import resolve_provider
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


def _start_worker(conn, task_id, pid, *, lane="ready"):
    if lane == "review":
        task = kb.claim_review_task(conn, task_id, claimer=_claimer(pid))
    else:
        task = kb.claim_task(conn, task_id, claimer=_claimer(pid))
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


def _record_as_worker(monkeypatch, conn, task_id, reset_at, *, provider):
    """The claimed run's worker records when its provider's limit lifts, as it does before exiting.

    The worker's own recording path runs under exactly the environment the
    dispatcher builds for that run, its credential pool exhausted until
    ``reset_at``; the reset must land on the claimed board's run.
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
    worker = SimpleNamespace(agent=SimpleNamespace(_credential_pool=CredentialPool(provider, [credential])))
    with monkeypatch.context() as worker_process:
        for key in [key for key in os.environ if key not in env]:
            worker_process.delenv(key)
        for key, value in env.items():
            worker_process.setenv(key, value)
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
    task_id = kb.create_task(
        conn, title="moved card", assignee="worker", model_override="m-1", provider_override="prov-a",
    )
    _start_worker(conn, task_id, 4900)
    _record_as_worker(monkeypatch, conn, task_id, T0 + 3 * 3600, provider="prov-a")
    _worker_exits(conn, 4900, kb.KANBAN_RATE_LIMIT_EXIT_CODE)

    clock["t"] = T0 + 3600
    kb.set_model_override(conn, task_id, "m-1", provider="prov-a")
    assert kb.check_respawn_guard(conn, task_id) == "rate_limit_cooldown"
    kb.set_model_override(conn, task_id, "m-2", provider="prov-b")
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
