"""The gateway dispatcher holds new work while the owner journeys run.

``_kanban_dispatch_allowed()`` keeps the owner's pause check first, then
answers False while ``journeys_hold.state(now)`` is active. So a hold stops
what the pause stops (the auto-decompose tick, the claims and the Project
ticks) and touches no worker that runs, and an error in the rule never stops
the dispatcher.
"""

from __future__ import annotations

import asyncio
import subprocess
import sys
import time

import pytest

from agent import estop
from gateway import kanban_watchers
from hermes_cli import journeys_hold, kanban_decompose, release_ledger
from hermes_cli import kanban_db as kb
from hermes_constants import get_default_hermes_root

WORKER = "profile-a"
PROJECT = "proj"
MINUTE = 60
# What the dispatcher runs off its event loop only while neither a pause nor a hold stops it.
STOPPED_WORK = {"_auto_decompose_tick", "_tick_once", "_owner_project_ticks"}


class _Served:
    def __init__(self):
        self.ticks = 0  # the ticks that ended
        self.spawned = []  # (tick, board, task id) of each worker started
        self.ran = []  # (tick, name) of each job run off the event loop
        self.pid = None  # the PID each started worker reports


@pytest.fixture
def served(monkeypatch):
    import hermes_cli.profiles as profiles

    monkeypatch.setattr(profiles, "profile_exists", lambda name: True)
    kb.init_db()
    kb.create_board(slug=kb.DEFAULT_BOARD, name="Default")
    record = _Served()

    def _spawn(task, workspace, board=None):
        record.spawned.append((record.ticks, board, task.id))
        return record.pid

    monkeypatch.setattr(kb, "_default_spawn", _spawn)
    monkeypatch.setattr(kanban_decompose, "list_triage_ids", lambda *, tenant=None: [])
    return record


@pytest.fixture
def journeys():
    """The owner journeys set up in the root home: their settings file is there."""
    folder = get_default_hermes_root() / "journeys"
    folder.mkdir(parents=True)
    (folder / "settings").write_text("JOURNEYS_STACK=build\n", encoding="utf-8")
    return folder


def _stamp(epoch):
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(epoch))


def _result(journeys, outcome, started):
    """journeys/result as scripts/journeys/run_once.sh writes it: a run that ended has a finished= line."""
    text = f"result={outcome}\nstep=\nreason=\nstarted={_stamp(started)}\n"
    if outcome != "started":
        text += f"finished={_stamp(time.time())}\n"
    (journeys / "result").write_text(text, encoding="utf-8")


def _release():
    """A release that the release command ended just now."""
    conn = release_ledger.connect()
    try:
        batch = conn.execute(
            "INSERT INTO release_batches (state, version, tier, created_at) VALUES ('releasing', 1, 0, ?)",
            (int(time.time()),),
        ).lastrowid
        release_ledger.finish_release(conn, batch, outcome="released")
    finally:
        conn.close()


def _card(board=kb.DEFAULT_BOARD):
    with kb.connect_closing(board=board) as conn:
        return kb.create_task(conn, title=f"{board} card", assignee=WORKER)


def _task(task_id):
    with kb.connect_closing(board=kb.DEFAULT_BOARD) as conn:
        return kb.get_task(conn, task_id)


def _started(served):
    return [(tick, task_id) for tick, _, task_id in served.spawned]


def _ran(served, ticks):
    return {name for tick, name in served.ran if tick in ticks}


def _run(monkeypatch, served, *, ticks, between_ticks=None, **kanban):
    """Run the gateway's dispatcher for ``ticks`` ticks with no wait;
    ``between_ticks(n)`` runs after tick n ends."""
    from gateway.run import GatewayRunner
    import hermes_cli.config as cfg_mod

    runner = object.__new__(GatewayRunner)
    runner._running = True
    cfg = {"dispatch_in_gateway": True, "dispatch_interval_seconds": 1, "auto_decompose": False, **kanban}
    monkeypatch.setattr(cfg_mod, "load_config", lambda: {"kanban": dict(cfg)})

    async def _to_thread(fn, *args, **kwargs):
        served.ran.append((served.ticks, getattr(fn, "__name__", "")))
        return fn(*args, **kwargs)

    async def _sleep(delay):
        if delay > 1:  # the start delay, before the first tick
            return
        served.ticks += 1
        if between_ticks is not None:
            between_ticks(served.ticks)
        if served.ticks >= ticks:
            runner._running = False

    monkeypatch.setattr("gateway.run.asyncio.to_thread", _to_thread)
    monkeypatch.setattr("gateway.run.asyncio.sleep", _sleep)
    asyncio.run(asyncio.wait_for(runner._kanban_dispatcher_watcher(), timeout=60.0))


class _Clock:
    """The time module as gateway.kanban_watchers sees it, on a clock the test moves."""

    def __init__(self, now):
        self.now = now

    def time(self):
        return self.now

    def __getattr__(self, name):
        return getattr(time, name)


def test_dispatch_is_held_while_a_journeys_run_is_at_work(served, journeys, monkeypatch):
    _result(journeys, "started", time.time() - MINUTE)
    task = _card()
    assert kanban_watchers._kanban_dispatch_allowed() is False

    _run(monkeypatch, served, ticks=3, auto_decompose=True)
    assert _started(served) == []
    assert _task(task).status == "ready"
    assert "reap_worker_zombies" in _ran(served, range(3))
    assert not _ran(served, range(3)) & STOPPED_WORK


def test_dispatch_is_held_between_the_release_end_and_the_journeys_start(served, journeys, monkeypatch):
    _release()
    task = _card()
    assert kanban_watchers._kanban_dispatch_allowed() is False

    def _journeys_start(tick):
        if tick == 2:
            _result(journeys, "started", time.time())

    _run(monkeypatch, served, ticks=4, between_ticks=_journeys_start)
    assert _started(served) == []
    assert _task(task).status == "ready"


def test_dispatch_starts_again_when_the_run_ends(served, journeys, monkeypatch):
    _result(journeys, "started", time.time() - MINUTE)
    task = _card()

    def _run_ends(tick):
        if tick == 2:
            _result(journeys, "passed", time.time() - MINUTE)

    _run(monkeypatch, served, ticks=4, between_ticks=_run_ends, auto_decompose=True)
    assert _started(served) == [(2, task)]
    assert not _ran(served, range(2)) & STOPPED_WORK
    assert STOPPED_WORK <= _ran(served, range(2, 4))


def test_dispatch_starts_again_at_the_limit_while_the_run_still_runs(served, journeys, monkeypatch):
    start = int(time.time()) - 45 * MINUTE + 2
    _result(journeys, "started", start)
    clock = _Clock(start + 45 * MINUTE - 2)
    monkeypatch.setattr(kanban_watchers, "time", clock)
    task = _card()

    def _clock_moves(tick):
        clock.now += 1

    _run(monkeypatch, served, ticks=4, between_ticks=_clock_moves)
    assert _started(served) == [(2, task)]
    assert (journeys / "result").read_text(encoding="utf-8") == (
        f"result=started\nstep=\nreason=\nstarted={_stamp(start)}\n"
    )


@pytest.mark.parametrize("error", ["the rule raises", "the rule cannot be imported"])
def test_an_error_in_the_hold_never_stops_dispatch(served, journeys, monkeypatch, error):
    _result(journeys, "started", time.time() - MINUTE)
    assert kanban_watchers._kanban_dispatch_allowed() is False

    if error == "the rule raises":
        def _broken(now):
            raise RuntimeError("the hold rule broke")

        monkeypatch.setattr(journeys_hold, "state", _broken)
    else:
        import hermes_cli

        monkeypatch.delattr(hermes_cli, "journeys_hold")
        monkeypatch.setitem(sys.modules, "hermes_cli.journeys_hold", None)
    assert kanban_watchers._kanban_dispatch_allowed() is True

    task = _card()
    _run(monkeypatch, served, ticks=2)
    assert _started(served) == [(0, task)]


def test_the_owner_pause_still_stops_dispatch_after_the_hold_ends(served, journeys, monkeypatch):
    estop.engage("owner pause")
    pause = estop.sentinel_path().read_bytes()
    _result(journeys, "started", time.time() - MINUTE)
    task = _card()
    seen = {}

    def _hold_ends_then_owner_resumes(tick):
        if tick == 2:
            _result(journeys, "passed", time.time() - MINUTE)
        elif tick == 4:
            seen["hold"] = journeys_hold.state(time.time())["active"]
            seen["pause"] = estop.sentinel_path().read_bytes()
            estop.disengage()

    _run(monkeypatch, served, ticks=6, between_ticks=_hold_ends_then_owner_resumes)
    assert seen == {"hold": False, "pause": pause}
    assert _started(served) == [(4, task)]


def test_a_project_paused_by_the_owner_stays_paused_after_the_hold_ends(served, journeys, monkeypatch):
    kb.create_board(slug=PROJECT, name="Project")
    _card(PROJECT)
    kb.write_board_metadata(PROJECT, dispatch_enabled=True, dispatch_paused_by_owner=True)
    board_file = kb.board_metadata_path(PROJECT).read_bytes()
    task = _card()
    kb.write_board_metadata(kb.DEFAULT_BOARD, dispatch_enabled=True)
    _result(journeys, "started", time.time() - MINUTE)

    def _run_ends(tick):
        if tick == 2:
            _result(journeys, "failed", time.time() - MINUTE)

    _run(monkeypatch, served, ticks=4, between_ticks=_run_ends, dispatch_require_board_activation=True)
    assert _started(served) == [(2, task)]
    assert kb.board_metadata_path(PROJECT).read_bytes() == board_file
    assert kb.read_board_metadata(PROJECT)["dispatch_paused_by_owner"] is True


def test_running_workers_are_not_touched_by_the_hold(served, journeys, monkeypatch):
    worker = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])
    try:
        served.pid = worker.pid
        task = _card()
        seen = {}

        def _journeys_start(tick):
            if tick == 1:
                _result(journeys, "started", time.time())
                seen["task"] = _task(task)

        _run(monkeypatch, served, ticks=4, between_ticks=_journeys_start)
        assert _started(served) == [(0, task)]
        assert seen["task"].status == "running"
        assert _task(task) == seen["task"]
        assert worker.poll() is None
    finally:
        worker.kill()
        worker.wait()
