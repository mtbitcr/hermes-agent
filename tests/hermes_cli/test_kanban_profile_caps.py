"""Per-profile caps from ``kanban.max_in_progress_by_profile``.

A profile listed in the mapping gets its own cap on running tasks, counted
across every board the gateway dispatches; every other profile keeps the
shared ``kanban.max_in_progress_per_profile``. A tick whose ready work waited
only because such a cap was full is not a stuck dispatcher.

Each test drives the gateway's embedded dispatcher loop, the path that reads
the setting, with a recording spawn stub. Profile names are placeholders.
"""
from __future__ import annotations

import asyncio
import logging
from pathlib import Path

import pytest

CAPPED = "profile-a"
OTHER = "profile-b"
SECOND_BOARD = "board-two"
STUCK_LINE = "kanban dispatcher stuck"


@pytest.fixture
def kb(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    for name in (CAPPED, OTHER):
        (home / "profiles" / name).mkdir(parents=True)
    from hermes_cli import kanban_db

    kanban_db.init_db()
    kanban_db.create_board(slug=kanban_db.DEFAULT_BOARD, name="Default")
    kanban_db.create_board(slug=SECOND_BOARD, name="Two")
    return kanban_db


class _SpawnLog(list):
    """Spawns as ``(board, assignee, task_id)``; ``failing`` assignees raise."""

    def __init__(self):
        super().__init__()
        self.failing = set()


@pytest.fixture
def spawns(kb, monkeypatch):
    """Record every spawn. Returns no PID, so a spawned task stays
    ``running`` on later ticks; a failing assignee raises like a broken
    profile venv."""
    calls = _SpawnLog()

    def _spawn(task, workspace, board=None):
        if task.assignee in calls.failing:
            raise RuntimeError("profile venv is broken")
        calls.append((board, task.assignee, task.id))
        return None

    monkeypatch.setattr(kb, "_default_spawn", _spawn)
    return calls


def _ready(kb, assignee, *, board=None):
    with kb.connect_closing(board=board or kb.DEFAULT_BOARD) as conn:
        return kb.create_task(conn, title=f"{assignee} work", assignee=assignee)


def _running(kb, assignee, *, board=None):
    task_id = _ready(kb, assignee, board=board)
    with kb.connect_closing(board=board or kb.DEFAULT_BOARD) as conn:
        assert kb.claim_task(conn, task_id) is not None
    return task_id


def _status(kb, task_id, *, board=None):
    with kb.connect_closing(board=board or kb.DEFAULT_BOARD) as conn:
        return kb.get_task(conn, task_id).status


def _run_dispatcher(monkeypatch, kanban_cfg, *, ticks):
    """Run the gateway dispatcher loop for ``ticks`` full ticks."""
    from gateway.run import GatewayRunner
    import hermes_cli.config as cfg_mod

    runner = object.__new__(GatewayRunner)
    runner._running = True
    cfg = {
        "dispatch_in_gateway": True,
        "dispatch_interval_seconds": 1,
        "auto_decompose": False,
        **kanban_cfg,
    }
    monkeypatch.setattr(cfg_mod, "load_config", lambda: {"kanban": dict(cfg)})
    done = {"ticks": 0}

    async def _to_thread(fn, *args, **kwargs):
        result = fn(*args, **kwargs)
        # The rest of a tick (probe + health telemetry) still runs after
        # the loop is told to stop; the next iteration never starts.
        if getattr(fn, "__name__", "") == "_tick_once":
            done["ticks"] += 1
            if done["ticks"] >= ticks:
                runner._running = False
        return result

    async def _sleep(_delay):
        return None

    monkeypatch.setattr("gateway.run.asyncio.to_thread", _to_thread)
    monkeypatch.setattr("gateway.run.asyncio.sleep", _sleep)
    asyncio.run(
        asyncio.wait_for(runner._kanban_dispatcher_watcher(), timeout=60.0)
    )
    assert done["ticks"] == ticks


# a. listed profile at its own cap waits; another profile starts same tick
def test_listed_profile_at_its_cap_waits_while_other_profile_starts(
    kb, spawns, monkeypatch
):
    _running(kb, CAPPED)
    capped_next = _ready(kb, CAPPED)
    other_next = _ready(kb, OTHER)

    _run_dispatcher(
        monkeypatch, {"max_in_progress_by_profile": {CAPPED: 1}}, ticks=1
    )

    assert [(a, t) for _, a, t in spawns] == [(OTHER, other_next)]
    assert _status(kb, capped_next) == "ready"


# b. the running task that fills the cap lives on a different board
def test_listed_profile_cap_counts_running_task_on_another_board(
    kb, spawns, monkeypatch
):
    _running(kb, CAPPED, board=SECOND_BOARD)
    capped_next = _ready(kb, CAPPED)
    other_next = _ready(kb, OTHER)

    _run_dispatcher(
        monkeypatch, {"max_in_progress_by_profile": {CAPPED: 1}}, ticks=1
    )

    assert [(a, t) for _, a, t in spawns] == [(OTHER, other_next)]
    assert _status(kb, capped_next) == "ready"


# b. a spawn on one board holds the same profile's card on the next board
def test_listed_profile_spawn_on_one_board_holds_card_on_next_board(
    kb, spawns, monkeypatch
):
    _ready(kb, CAPPED)
    _ready(kb, CAPPED, board=SECOND_BOARD)

    _run_dispatcher(
        monkeypatch, {"max_in_progress_by_profile": {CAPPED: 1}}, ticks=1
    )

    assert [a for _, a, _ in spawns] == [CAPPED]


# c. an invalid entry is ignored with one warning; the integer cap applies
@pytest.mark.parametrize(
    "by_profile",
    [{CAPPED: 0}, {CAPPED: True}, {CAPPED: 2.5}, {CAPPED: "two"}, 3],
    ids=["zero", "bool", "fraction", "text", "not-a-mapping"],
)
def test_invalid_mapping_entry_warns_once_and_integer_cap_applies(
    kb, spawns, monkeypatch, caplog, by_profile
):
    _running(kb, CAPPED)
    capped_next = _ready(kb, CAPPED)

    with caplog.at_level(logging.WARNING, logger="gateway.run"):
        _run_dispatcher(
            monkeypatch,
            {
                "max_in_progress_per_profile": 1,
                "max_in_progress_by_profile": by_profile,
            },
            ticks=2,
        )

    warnings = [
        r.getMessage() for r in caplog.records
        if r.levelno == logging.WARNING
        and "max_in_progress_by_profile" in r.getMessage()
    ]
    assert len(warnings) == 1, warnings
    assert spawns == []
    assert _status(kb, capped_next) == "ready"


# d. cap-held ticks never warn "stuck"; a genuinely stuck queue still does
@pytest.mark.parametrize(
    ("caps", "named_cap"),
    [
        ({"max_in_progress_per_profile": 1},
         "kanban.max_in_progress_per_profile"),
        ({"max_in_progress_by_profile": {CAPPED: 1}},
         "kanban.max_in_progress_by_profile"),
    ],
    ids=["integer-cap", "profile-cap"],
)
def test_cap_held_ticks_are_not_stuck_but_a_stuck_queue_is(
    kb, spawns, monkeypatch, caplog, caps, named_cap
):
    _running(kb, CAPPED)
    capped_next = _ready(kb, CAPPED)
    cfg = {**caps, "failure_limit": 100}

    with caplog.at_level(logging.INFO, logger="gateway.run"):
        _run_dispatcher(monkeypatch, cfg, ticks=7)

    messages = [r.getMessage() for r in caplog.records]
    assert not any(STUCK_LINE in m for m in messages)
    cap_lines = [
        r.getMessage() for r in caplog.records
        if r.levelno == logging.INFO
        and named_cap in r.getMessage()
        and CAPPED in r.getMessage()
        and "full" in r.getMessage()
    ]
    assert len(cap_lines) == 1, messages
    assert spawns == []
    assert _status(kb, capped_next) == "ready"

    # Same held card, plus a card that waits for another reason: stuck.
    spawns.failing.add(OTHER)
    _ready(kb, OTHER)
    caplog.clear()
    with caplog.at_level(logging.INFO, logger="gateway.run"):
        _run_dispatcher(monkeypatch, cfg, ticks=7)

    assert any(STUCK_LINE in r.getMessage() for r in caplog.records)
    assert spawns == []
