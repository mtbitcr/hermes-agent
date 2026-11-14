"""Card B2 of the daily project review plan: the gateway's dispatcher runs the review tick right after the summary tick.

Each test drives the real dispatcher loop (``GatewayRunner._kanban_dispatcher_watcher``) over real stores in the
per-test root, as the dispatcher tests of the summary tick do: a stubbed sleep, a recording stand-in for the worker
spawn so that no worker process starts, recording wrappers around the real summary and review tick, and a fixed clock
for the review tick, the one job that reads the Vienna day.
A test that switches the review on stores the owner's settings once on in the root config: 06:00 Vienna, a cap of 20
project reviews plus 1 cross-project slot, stale after 14 quiet days, closure and digest on, and its own allowlist.
"""

from __future__ import annotations

import asyncio
import contextlib
import importlib
import logging
import os
import sqlite3
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest
import yaml

from agent import estop
from hermes_cli import kanban_db, owner_workspace as ow, projects_db
from plugins.dashboard_auth.raphael_workspace import model_policy

PLANNER = "raphael-planner"
BOARD = "project-summaries"
CREATOR = "project-review"
SUMMARY_CREATOR = "project-summary"
MINUTE = 60
DAY_SECONDS = 24 * 3600
VIENNA = ZoneInfo("Europe/Vienna")
TODAY = "2026-11-09"
# The jobs of one dispatcher tick after the zombie reaper, in the order they run.
JOBS = ["_tick_once", "summary_tick", "review_tick", "_ready_nonempty"]
# The jobs the loop hands asyncio.to_thread directly, by name; the summary and the review tick run inside a call.
_DIRECT = ("reap_worker_zombies", "_auto_decompose_tick", "_tick_once", "_ready_nonempty")
_CARDS = ("Book the meeting room", "Write the invitations", "Order the coffee")


def _ops():
    from hermes_cli import owner_project_review

    return owner_project_review


def _summary():
    from hermes_cli import owner_project_summary

    return owner_project_summary


def _at(year, month, day, hour=0, minute=0, second=0) -> int:
    """The epoch second of a Vienna wall-clock time."""
    return int(datetime(year, month, day, hour, minute, second, tzinfo=VIENNA).timestamp())


@contextlib.contextmanager
def _clock(instant: int):
    """A fixed clock: what runs inside reads ``instant`` from ``time.time()``."""
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(time, "time", lambda: instant)
        yield


def _enter_root(monkeypatch, base: Path) -> Path:
    """A shared root under ``base``, entered as the default profile, with its default board."""
    home = base / ".hermes"
    home.mkdir(parents=True)
    monkeypatch.setattr(Path, "home", lambda: base)
    monkeypatch.setenv("HERMES_HOME", str(home))
    kanban_db.init_db()
    kanban_db.create_board(slug=kanban_db.DEFAULT_BOARD, name="Default")
    return home


@pytest.fixture
def root(tmp_path, monkeypatch):
    import hermes_cli.profiles as profiles

    for name in ("HERMES_KANBAN_HOME", "HERMES_KANBAN_DB", "HERMES_KANBAN_BOARD", "HERMES_KANBAN_TASK"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(
        model_policy, "configured_assignment_for",
        lambda profile: model_policy.assignment_for(profile, "anthropic"),
    )
    # Every assignee counts as a profile, so a ready card reaches the spawn stand-in.
    monkeypatch.setattr(profiles, "profile_exists", lambda name: True)
    monkeypatch.setattr(estop, "_logged_components", set())
    return _enter_root(monkeypatch, tmp_path)


@pytest.fixture
def owner(root):
    return ow.OwnerContext(actor="default", profile="default", session="run_owt")


def _switch_on(home: Path, *allowed: str, **values) -> None:
    """The owner's settings once on, stored in the root config, for the named Projects."""
    stored = {
        "enabled": True, "allowlist": list(allowed), "closure": True, "digest": True,
        "review_hour": 6, "daily_cap": 20, "portfolio_cap": 1, "stale_days": 14,
    }
    (home / "config.yaml").write_text(yaml.safe_dump({"owner_review": {**stored, **values}}), encoding="utf-8")


# ---------------------------------------------------------------------------------------------------------------
# Stores, read on separate connections
# ---------------------------------------------------------------------------------------------------------------


def _rows(path: Path, sql: str, params=()) -> list:
    conn = sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True)
    try:
        return conn.execute(sql, params).fetchall()
    finally:
        conn.close()


def _board_db(home: Path, slug: str) -> Path:
    return home / "kanban" / "boards" / slug / "kanban.db"


@contextlib.contextmanager
def _board(home: Path, slug: str):
    with kanban_db.connect_closing(_board_db(home, slug), board=slug) as conn:
        yield conn


def _review_rows(home: Path) -> list:
    """``(idempotency_key, status)`` of every review task on the root summary board, archived ones included."""
    path = _board_db(home, BOARD)
    if not path.exists():
        return []
    return _rows(path, "SELECT idempotency_key, status FROM tasks WHERE created_by = ? ORDER BY idempotency_key",
                 (CREATOR,))


def _key(project: dict, failures: int = 0) -> str:
    return f"{CREATOR}:{project['project_id']}:{TODAY}:{failures}"


def _budget(home: Path, review_date: str = TODAY) -> tuple:
    rows = _rows(home / "projects.db", "SELECT project_reviews, portfolio_reviews FROM review_budget "
                 "WHERE review_date = ?", (review_date,))
    return tuple(rows[0]) if rows else (0, 0)


_STATE = ("open_task_id", "open_review_date", "failures", "last_reviewed_date", "last_reviewed_at", "last_news_date",
          "event_cursor", "delivery_cursor", "release_cursor", "updated_at")


def _state(home: Path, project: dict) -> dict | None:
    rows = _rows(home / "projects.db", f"SELECT {', '.join(_STATE)} FROM project_review_state WHERE project_id = ?",
                 (project["project_id"],))
    return dict(zip(_STATE, rows[0])) if rows else None


def _no_review_store_touched(home: Path) -> None:
    assert _rows(home / "projects.db", "SELECT count(*) FROM project_review_state") == [(0,)]
    assert _rows(home / "projects.db", "SELECT count(*) FROM review_budget") == [(0,)]
    assert _review_rows(home) == []


# ---------------------------------------------------------------------------------------------------------------
# Projects
# ---------------------------------------------------------------------------------------------------------------


def _add_cards(home: Path, project: dict, titles, **fields) -> list[str]:
    with _board(home, project["board"]) as conn:
        return [kanban_db.create_task(conn, title=title, project_id=project["project_id"], **fields) for title in titles]


def _project(home: Path, owner, name: str, cards=()) -> dict:
    """An owner Project with its own board, as the owner kernel creates it, and its first cards (news for a review)."""
    with pytest.MonkeyPatch.context() as patch, _clock(_at(2026, 11, 8, 9)):
        patch.setattr(ow, "_confirm", lambda *_args, **_kwargs: {"approved": True})
        project = ow.bootstrap(owner, idempotency_key="setup-" + name.lower().replace(" ", "-"), name=name)
        assert project["ok"] is True
        project["slug"] = next(
            item["slug"] for item in ow.list_committed_projects(owner)
            if str(item["project_id"]) == project["project_id"]
        )
        _add_cards(home, project, cards)
    return project


def _quick_project(home: Path, name: str, cards: int) -> dict:
    """A registry Project with a board bound to it and ``cards`` new cards, made without the owner kernel."""
    slug = name.lower().replace(" ", "-")
    with _clock(_at(2026, 11, 8, 9)):
        with projects_db.connect_closing(home / "projects.db") as conn:
            project_id = projects_db.create_project(conn, name=name, board_slug=slug)
        kanban_db.create_board(slug, name=name, project_id=project_id)
        project = {"project_id": project_id, "board": slug, "slug": slug}
        _add_cards(home, project, [f"Card {n} of {name}" for n in range(cards)])
    return project


def _seed_reviewed(home: Path, project: dict, last_reviewed_date: str) -> None:
    """A Project the review saw before: reviewed on ``last_reviewed_date`` with every event up to now read."""
    cursor = _rows(_board_db(home, project["board"]), "SELECT COALESCE(MAX(id), 0) FROM task_events")[0][0]
    with projects_db.connect_closing(home / "projects.db") as conn:
        with kanban_db.write_txn(conn):
            conn.execute(
                "INSERT INTO project_review_state (project_id, event_cursor, last_reviewed_date, last_news_date, "
                "updated_at) VALUES (?, ?, ?, ?, 0)",
                (project["project_id"], cursor, last_reviewed_date, last_reviewed_date),
            )


def _restart() -> None:
    """A new process, as B1's own restart helper makes it: the module and the per-process schema cache start over."""
    projects_db._INITIALIZED_PATHS.clear()
    importlib.reload(_ops())


# ---------------------------------------------------------------------------------------------------------------
# The dispatcher loop
# ---------------------------------------------------------------------------------------------------------------


class _Loop:
    """What the dispatcher loop ran: the jobs of each tick in order, what each review tick returned, the spawns."""

    def __init__(self):
        self.jobs: list[list[str]] = []
        self.calls: list[list[list[str]]] = []  # per tick, per asyncio.to_thread call, the jobs that ran inside it
        self.made: list[int] = []
        self.spawned: list[tuple] = []  # (tick, board, task id, title, assignee, created_by)

    def record_spawns(self, monkeypatch) -> None:
        def _spawn(task, workspace, board=None):
            # No PID: the card stays running, and no worker process starts.
            self.spawned.append((len(self.jobs) - 1, board, task.id, task.title, task.assignee, task.created_by))
            return None

        monkeypatch.setattr(kanban_db, "_default_spawn", _spawn)


@pytest.fixture
def loop(root, monkeypatch):
    record = _Loop()
    record.record_spawns(monkeypatch)
    return record


def _drive(monkeypatch, loop: _Loop, clocks, *, kanban=None, after=None, between=None) -> None:
    """Run the gateway's dispatcher loop for one tick per clock; tick n runs its review tick at ``clocks[n]``.

    While the loop runs, the summary and the review tick on their modules are recorders around what is there before,
    so a job is recorded in the asyncio.to_thread call it runs inside. ``after`` maps a job's name to what runs right
    after that job returns. ``between(n)`` runs in the pause after tick n, for every tick but the last.
    """
    from gateway.run import GatewayRunner
    import hermes_cli.config as cfg_mod

    runner = object.__new__(GatewayRunner)
    runner._running = True
    config = {"dispatch_in_gateway": True, "dispatch_interval_seconds": 1, "auto_decompose": False, **(kanban or {})}
    monkeypatch.setattr(cfg_mod, "load_config", lambda: {"kanban": dict(config)})
    first = len(loop.jobs)
    summary_tick, review_tick = _summary().summary_tick, _ops().review_tick

    def _run(name, fn, *args, **kwargs):
        if name in JOBS:
            loop.jobs[-1].append(name)
        loop.calls[-1][-1].append(name)
        if name == "review_tick":
            with _clock(clocks[len(loop.jobs) - first - 1]):
                result = fn(*args, **kwargs)
            loop.made.append(result)
        else:
            result = fn(*args, **kwargs)
        if after and name in after:
            after[name]()
        return result

    async def _to_thread(fn, *args, **kwargs):
        name = getattr(fn, "__name__", "")
        if name == "reap_worker_zombies":
            loop.jobs.append([])
            loop.calls.append([])
            if len(loop.jobs) - first >= len(clocks):
                runner._running = False  # the rest of this tick still runs; the next one never starts
        loop.calls[-1].append([])
        if name in _DIRECT:
            return _run(name, fn, *args, **kwargs)
        return fn(*args, **kwargs)

    async def _sleep(_delay):
        if between is not None:
            between(len(loop.jobs) - 1)

    monkeypatch.setattr("gateway.run.asyncio.to_thread", _to_thread)
    monkeypatch.setattr("gateway.run.asyncio.sleep", _sleep)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(_summary(), "summary_tick", lambda *a, **kw: _run("summary_tick", summary_tick, *a, **kw))
        patch.setattr(_ops(), "review_tick", lambda *a, **kw: _run("review_tick", review_tick, *a, **kw))
        asyncio.run(asyncio.wait_for(runner._kanban_dispatcher_watcher(), timeout=120.0))
    assert len(loop.jobs) - first == len(clocks)


# ---------------------------------------------------------------------------------------------------------------
# The wiring: once per tick, right after the summary tick
# ---------------------------------------------------------------------------------------------------------------


def test_the_review_tick_runs_once_per_tick_right_after_the_summary_tick(root, loop, monkeypatch):
    _drive(monkeypatch, loop, [_at(2026, 11, 9, 7) + n * MINUTE for n in range(3)])

    assert loop.jobs == [JOBS, JOBS, JOBS]
    # Off by default: the tick starts nothing and touches no review store.
    assert loop.made == [0, 0, 0]
    assert _review_rows(root) == []


def test_the_summary_tick_and_the_review_tick_run_inside_one_and_the_same_to_thread_call_of_a_tick(
    root, owner, loop, monkeypatch
):
    garden = _project(root, owner, "Garden Plan", _CARDS)
    _switch_on(root, garden["project_id"])

    _drive(monkeypatch, loop, [_at(2026, 11, 9, 7) + n * MINUTE for n in range(3)], kanban={"auto_decompose": True})

    # A tick hands asyncio.to_thread what it did before the review tick was wired: the zombie reaper, the
    # auto-decompose tick (on, as by default), the dispatch, one call for the summary tick, and the ready check.
    # The review tick runs inside the summary tick's call, right after the summary tick.
    tick = [["reap_worker_zombies"], ["_auto_decompose_tick"], ["_tick_once"], ["summary_tick", "review_tick"],
            ["_ready_nonempty"]]
    assert loop.calls == [tick, tick, tick]
    assert loop.made == [1, 0, 0]


# ---------------------------------------------------------------------------------------------------------------
# The Vienna day: nothing before 06:00, each allowlisted Project once from 06:00 on
# ---------------------------------------------------------------------------------------------------------------


def test_a_tick_before_six_vienna_starts_nothing(root, owner, loop, monkeypatch):
    garden = _project(root, owner, "Garden Plan", _CARDS)
    desk = _project(root, owner, "Planning Desk", _CARDS)
    _switch_on(root, garden["project_id"], desk["slug"])

    _drive(monkeypatch, loop, [_at(2026, 11, 9, 0, 0, 1), _at(2026, 11, 9, 5), _at(2026, 11, 9, 5, 59, 59)])

    assert loop.jobs == [JOBS, JOBS, JOBS]
    assert loop.made == [0, 0, 0]
    assert _review_rows(root) == []
    assert _budget(root) == (0, 0)
    for project in (garden, desk):
        state = _state(root, project)
        assert state is None or (state["open_task_id"], state["last_reviewed_date"]) == (None, None)
    assert not [spawn for spawn in loop.spawned if spawn[-1] == CREATOR]


def test_the_first_tick_from_six_vienna_starts_each_allowlisted_project_once(root, owner, loop, monkeypatch):
    garden = _project(root, owner, "Garden Plan", _CARDS)
    desk = _project(root, owner, "Planning Desk", _CARDS)
    spare = _project(root, owner, "Spare Room", _CARDS)  # news too, but not allowlisted
    _switch_on(root, garden["project_id"], desk["slug"])
    clocks = [_at(2026, 11, 9, 5, 59, 59), _at(2026, 11, 9, 6), _at(2026, 11, 9, 6, 1), _at(2026, 11, 9, 12),
              _at(2026, 11, 9, 23, 59, 59)]

    _drive(monkeypatch, loop, clocks)

    assert loop.made == [0, 2, 0, 0, 0]
    rows = _review_rows(root)
    assert sorted(rows) == sorted([(_key(garden), "running"), (_key(desk), "running")])
    assert _state(root, spare) is None
    assert _budget(root) == (2, 0)
    for project in (garden, desk):
        assert _state(root, project)["open_review_date"] == TODAY
    # The review tasks are dispatched like any other card, once each, from the tick after the one that made them.
    reviews = [spawn for spawn in loop.spawned if spawn[-1] == CREATOR]
    assert sorted((tick, board, assignee) for tick, board, _id, _title, assignee, _by in reviews) == [
        (2, BOARD, PLANNER), (2, BOARD, PLANNER),
    ]


def test_a_restart_on_the_same_vienna_day_starts_no_repeat(root, owner, loop, monkeypatch):
    garden = _project(root, owner, "Garden Plan", _CARDS)
    quiet = _project(root, owner, "Quiet Corner")
    _seed_reviewed(root, quiet, "2026-11-08")
    _switch_on(root, garden["project_id"], quiet["project_id"])
    _drive(monkeypatch, loop, [_at(2026, 11, 9, 7), _at(2026, 11, 9, 7, 5)])
    assert loop.made == [1, 0]
    before = (_review_rows(root), _state(root, garden), _state(root, quiet), _budget(root))
    assert before[1]["open_review_date"] == TODAY
    assert (before[2]["open_task_id"], before[2]["last_reviewed_date"]) == (None, TODAY)

    _restart()
    after_restart = _Loop()
    after_restart.record_spawns(monkeypatch)
    _drive(monkeypatch, after_restart, [_at(2026, 11, 9, 9), _at(2026, 11, 9, 15), _at(2026, 11, 9, 23, 59, 59)])

    assert after_restart.jobs == [JOBS, JOBS, JOBS]
    assert after_restart.made == [0, 0, 0]
    assert (_review_rows(root), _state(root, garden), _state(root, quiet), _budget(root)) == before
    assert not [spawn for spawn in after_restart.spawned if spawn[-1] == CREATOR]


# ---------------------------------------------------------------------------------------------------------------
# The daily cap: 20 project reviews, and 1 cross-project slot outside the cap
# ---------------------------------------------------------------------------------------------------------------


def test_the_cap_of_twenty_plus_one_is_honored(root, loop, monkeypatch):
    projects = [_quick_project(root, f"Project {n:02d}", 2) for n in range(22)]
    _switch_on(root, *(p["project_id"] for p in projects))
    start = _at(2026, 11, 9, 7)

    _drive(monkeypatch, loop, [start + n * MINUTE for n in range(9)])

    assert loop.made == [3, 3, 3, 3, 3, 3, 2, 0, 0]
    assert len(_review_rows(root)) == 20
    assert _budget(root) == (20, 0)
    assert sum(_state(root, p)["open_task_id"] is None for p in projects) == 2

    # The cross-project slot stayed free: it is granted once, and only once.
    with projects_db.connect_closing(root / "projects.db") as conn:
        assert _ops().take_portfolio_slot(conn, TODAY, 1, start + DAY_SECONDS) is True
        assert _ops().take_portfolio_slot(conn, TODAY, 1, start + DAY_SECONDS) is False
    assert _budget(root) == (20, 1)

    # A later tick of the same Vienna day starts nothing more.
    _drive(monkeypatch, loop, [_at(2026, 11, 9, 23)])
    assert loop.made[-1] == 0
    assert len(_review_rows(root)) == 20
    assert _budget(root) == (20, 1)



# ---------------------------------------------------------------------------------------------------------------
# Off: the flag off, or the flag on with an empty allowlist, starts nothing
# ---------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("stored", ["nothing", "flag off", "empty allowlist"])
def test_flag_off_or_an_empty_allowlist_starts_nothing(root, owner, loop, monkeypatch, stored):
    garden = _project(root, owner, "Garden Plan", _CARDS)
    if stored == "flag off":
        _switch_on(root, garden["project_id"], enabled=False)
    elif stored == "empty allowlist":
        _switch_on(root)

    _drive(monkeypatch, loop, [_at(2026, 11, 9, 7), _at(2026, 11, 9, 8), _at(2026, 11, 9, 9)])

    assert loop.jobs == [JOBS, JOBS, JOBS]
    assert loop.made == [0, 0, 0]
    _no_review_store_touched(root)
    assert not [spawn for spawn in loop.spawned if spawn[-1] == CREATOR]


# ---------------------------------------------------------------------------------------------------------------
# The watcher's existing behavior stays: pause, failure, and plan F4
# ---------------------------------------------------------------------------------------------------------------


def test_neither_tick_runs_while_dispatch_is_paused(root, owner, loop, monkeypatch):
    garden = _project(root, owner, "Garden Plan", _CARDS)
    _switch_on(root, garden["project_id"])
    estop.engage(reason="test")
    try:
        _drive(monkeypatch, loop, [_at(2026, 11, 9, 7), _at(2026, 11, 9, 8)])
    finally:
        estop.disengage()

    assert loop.jobs == [[], []]
    assert loop.made == []
    _no_review_store_touched(root)


@pytest.mark.parametrize(
    ("paused_after", "ran"),
    [
        ("_tick_once", ["_tick_once", "_ready_nonempty"]),
        ("summary_tick", ["_tick_once", "summary_tick", "_ready_nonempty"]),
    ],
)
def test_neither_tick_runs_when_the_pause_comes_during_the_tick(root, owner, loop, monkeypatch, paused_after, ran):
    garden = _project(root, owner, "Garden Plan", _CARDS)
    _switch_on(root, garden["project_id"])
    try:
        _drive(monkeypatch, loop, [_at(2026, 11, 9, 7)], after={paused_after: lambda: estop.engage(reason="test")})
        assert estop.is_engaged()
    finally:
        estop.disengage()

    assert loop.jobs == [ran]
    assert loop.made == []
    _no_review_store_touched(root)


def test_a_raising_review_tick_never_stops_the_summary_tick_or_dispatch(root, owner, loop, monkeypatch, caplog):
    garden = _project(root, owner, "Garden Plan", _CARDS)
    _add_cards(root, garden, ["Paint the fence"], assignee="profile-a")
    _switch_on(root, garden["project_id"])
    boom = RuntimeError("the review tick broke")

    def _raise(*_args, **_kwargs):
        raise boom

    monkeypatch.setattr(_ops(), "review_tick", _raise)

    with caplog.at_level(logging.WARNING, logger="gateway.run"):
        _drive(monkeypatch, loop, [_at(2026, 11, 9, 7) + n * MINUTE for n in range(3)])

    assert loop.jobs == [JOBS, JOBS, JOBS]
    failures = [record for record in caplog.records if record.exc_info and record.exc_info[1] is boom]
    assert [record.getMessage() for record in failures] == ["kanban dispatcher: project review tick failed"] * 3
    assert not [record for record in caplog.records if "unexpected watcher error" in record.getMessage()]
    # Dispatch went on: the Project's card in tick 0, the summary task the summary tick made in tick 0 in tick 1.
    assert [(tick, title) for tick, _board, _id, title, _who, by in loop.spawned if by != SUMMARY_CREATOR] == [
        (0, "Paint the fence"),
    ]
    summaries = [(tick, board) for tick, board, _id, _title, _who, by in loop.spawned if by == SUMMARY_CREATOR]
    assert summaries == [(1, BOARD)]
    assert _rows(_board_db(root, BOARD), "SELECT count(*) FROM tasks WHERE created_by = ?", (SUMMARY_CREATOR,)) == [
        (1,)
    ]
    assert _review_rows(root) == []


def _f4_world(base: Path, monkeypatch, *, review: bool, kanban=None) -> dict:
    """Plan F4: a world with ready project work, driven for four ticks from 07:00 Vienna, the review on or off.

    It runs with the concurrency caps in ``kanban`` and no others: by default no max_spawn, max_in_progress,
    max_in_progress_per_profile or max_in_progress_by_profile, and no memory-derived cap (the tests' conftest turns
    it off).

    Between ticks the workers the last tick started finish: the project cards, and the review tasks the harvest has
    not closed yet, whose results the harvest then reads. The summary tasks keep running in both worlds.
    """
    home = _enter_root(monkeypatch, base)
    owner = ow.OwnerContext(actor="default", profile="default", session="run_owt")
    garden = _project(home, owner, "Garden Plan")
    desk = _project(home, owner, "Planning Desk")
    with _clock(_at(2026, 11, 8, 9)):
        [book] = _add_cards(home, garden, ["Book the meeting room"], assignee="profile-a")
        _add_cards(home, garden, ["Write the invitations"], assignee="profile-b", parents=[book])
        _add_cards(home, garden, ["Order the coffee"])  # no assignee: it stays in the ready set
        _add_cards(home, desk, ["Plan the next milestone"], assignee=PLANNER)  # the review's own assignee
        _add_cards(home, desk, ["Pick the desk colour", "Buy the desk lamp"], assignee="profile-a")
    if review:
        _switch_on(home, garden["project_id"], desk["project_id"])
    names = {garden["board"]: "Garden Plan", desk["board"]: "Planning Desk"}
    # The result the desk's review worker returns: a valid typed project review that names a work title of its input
    # and has nothing to classify. The garden's review worker returns none, and the harvest refuses that.
    desk_review = {"project_review": {
        "summary": {"outcome": "The work is done.", "next_action": "Check the finished work.", "owner_decision": None},
        "brief": {"headline": "The work is done.",
                  "what_changed": "Plan the next milestone is done, and so is the other work.",
                  "next_step": "Check the finished work.", "needs_you": None},
        "classification": [], "proposed_closures": [],
    }}
    loop = _Loop()
    loop.record_spawns(monkeypatch)
    ready: list[dict] = []

    def _ready_set() -> dict:
        return {
            names[slug]: sorted(title for (title,) in _rows(
                _board_db(home, slug), "SELECT title FROM tasks WHERE status = 'ready'"
            ))
            for slug in names
        }

    def _between(tick: int) -> None:
        ready.append(_ready_set())
        for spawn_tick, board, task_id, title, _who, by in loop.spawned:
            if spawn_tick != tick:
                continue
            with _board(home, board) as conn:
                if board in names:
                    assert kanban_db.complete_task(conn, task_id, summary="Done.")
                elif by == CREATOR and kanban_db.get_task(conn, task_id).status == "running":
                    # A review the harvest has not closed finishes too, also when its Project completed meanwhile: such
                    # a review is finished, not retired, so the harvest reads its result.
                    metadata = desk_review if title == "Review the project: Planning Desk" else None
                    assert kanban_db.complete_task(conn, task_id, summary="Done.", metadata=metadata)

    _drive(monkeypatch, loop, [_at(2026, 11, 9, 7) + n * MINUTE for n in range(4)], kanban=kanban, between=_between)
    ready.append(_ready_set())

    tasks = {}
    for slug, name in names.items():
        with _board(home, slug) as conn:
            tasks[name] = sorted(
                (row["title"], row["status"], row["assignee"], kanban_db.task_event_revision(conn, row["id"]))
                for row in conn.execute("SELECT id, title, status, assignee FROM tasks")
            )
    summary_board = _rows(_board_db(home, BOARD), "SELECT title, status, assignee, created_by FROM tasks")
    review_state = {
        name: state and (state["open_task_id"], state["failures"])
        for name, state in (("Garden Plan", _state(home, garden)), ("Planning Desk", _state(home, desk)))
    }
    return {
        "made": loop.made,
        "review_state": review_state,
        "claims": [(tick, names[board], title, who) for tick, board, _id, title, who, _by in loop.spawned
                   if board in names],
        "summary_claims": [(tick, title, who, by) for tick, board, _id, title, who, by in loop.spawned
                           if board == BOARD],
        "ready": ready,
        "tasks": tasks,
        "summary_board": sorted(row for row in summary_board if row[3] != CREATOR),
        "review_board": sorted(row for row in summary_board if row[3] == CREATOR),
        "elsewhere": [
            slug for slug in names if _rows(_board_db(home, slug), "SELECT count(*) FROM tasks WHERE created_by = ?",
                                             (CREATOR,)) != [(0,)]
        ],
    }


@pytest.mark.parametrize(
    "caps",
    [{}, {"max_in_progress_by_profile": {PLANNER: 3}}, {"max_in_progress_per_profile": 3}],
    ids=["no caps", "planner cap 3", "per-profile cap 3"],
)
def test_f4_spare_capacity_claim_order_assignees_ready_set_and_project_tasks_are_the_same_with_and_without_review(
    root, tmp_path, monkeypatch, caps
):
    without = _f4_world(tmp_path / "without", monkeypatch, review=False, kanban=caps)
    with_review = _f4_world(tmp_path / "with", monkeypatch, review=True, kanban=caps)

    # The review ran: two tasks made in the first tick and archived by the B1 harvest after their workers finished,
    # the desk's when its typed result was accepted (its Project completed meanwhile, and an open review of a complete
    # Project is finished, not retired), the garden's when its result was refused.
    assert without["made"] == [0, 0, 0, 0]
    assert with_review["made"] == [2, 0, 0, 0]
    assert [status for _title, status, _who, _by in with_review["review_board"]] == ["archived", "archived"]
    assert with_review["review_state"] == {"Garden Plan": (None, 1), "Planning Desk": (None, 0)}
    assert without["review_state"] == {"Garden Plan": None, "Planning Desk": None}
    assert with_review["elsewhere"] == without["elsewhere"] == []

    # With spare capacity, the project work is claimed in the same order by the same assignees, with the same ready
    # set after each tick, and every project task ends with the same status and revision.
    assert without["claims"], "the world must dispatch project work"
    assert with_review["claims"] == without["claims"]
    assert with_review["ready"] == without["ready"]
    assert any(ready["Garden Plan"] for ready in without["ready"])
    assert with_review["tasks"] == without["tasks"]
    # Outside the project boards, the only difference is the review's own tasks on the summary board.
    assert with_review["summary_board"] == without["summary_board"]
    summaries = [claim for claim in with_review["summary_claims"] if claim[3] != CREATOR]
    if not caps:
        assert summaries == without["summary_claims"]
    else:
        # A cap of 3 is full on the shared summary board in tick 1: two summary tasks and two review tasks, all for
        # the planner. As F4 allows, a summary task may wait there for a slot: the same claims, at most a tick later.
        assert [claim[1:] for claim in summaries] == [claim[1:] for claim in without["summary_claims"]]
        assert [late[0] - on_time[0] for late, on_time in zip(summaries, without["summary_claims"])] in (
            [0, 0], [0, 1], [1, 0], [1, 1]
        )


def _f4_full_world(base: Path, monkeypatch, *, review: bool, target: str) -> dict:
    """Plan F4 under full capacity, the world of the reviewer's capacity probe, the review on or off.

    A summary already covered the Project's cards; the daily review has not. The review task starts in tick 0 and is
    claimed by the planner in tick 1. After tick 1 new work arrives: a planning card for the planner under
    max_in_progress_by_profile {planner: 1} ("project"), or a card that makes the summary tick start a summary task
    on the shared summary board under max_in_progress_per_profile 1 ("summary"). After tick ``release`` the review
    worker completes its task. Returns the status of the waiting work after each tick and the recorded spawns.
    """
    home = _enter_root(monkeypatch, base)
    owner = ow.OwnerContext(actor="default", profile="default", session="run_owt")
    garden = _project(home, owner, "Garden Plan", _CARDS)
    cursor = _rows(_board_db(home, garden["board"]), "SELECT MAX(id) FROM task_events")[0][0]
    with projects_db.connect_closing(home / "projects.db") as conn:
        with kanban_db.write_txn(conn):
            conn.execute(
                "INSERT INTO project_summary_state (project_id, event_cursor, pending, pending_urgent, updated_at) "
                "VALUES (?, ?, 0, 0, 0)", (garden["project_id"], cursor),
            )
    if review:
        _switch_on(home, garden["project_id"])
    loop = _Loop()
    loop.record_spawns(monkeypatch)
    if target == "project":
        caps, release = {"max_in_progress_by_profile": {PLANNER: 1}}, 2
        where, sql, params = garden["board"], "SELECT status FROM tasks WHERE title = ?", ("Plan the next milestone",)
    else:
        caps, release = {"max_in_progress_per_profile": 1}, 3
        where, sql, params = BOARD, "SELECT status FROM tasks WHERE created_by = ?", (SUMMARY_CREATOR,)
    statuses: list = []

    def _waiting():
        path = _board_db(home, where)
        rows = _rows(path, sql, params) if path.exists() else []
        assert len(rows) <= 1
        return rows[0][0] if rows else None

    def _between(tick: int) -> None:
        if tick < 0:
            return  # the pause before the first tick
        statuses.append(_waiting())
        if tick == 1:
            if target == "project":
                _add_cards(home, garden, ["Plan the next milestone"], assignee=PLANNER)
            else:
                _add_cards(home, garden, ["Prepare the planting plan"])
        if tick == release:
            # The review worker finishes, as through kanban_complete, and frees the planner's slot.
            for _tick, board, task_id, _title, _who, by in loop.spawned:
                if by == CREATOR:
                    with _board(home, board) as conn:
                        assert kanban_db.get_task(conn, task_id).status == "running"
                        assert kanban_db.complete_task(conn, task_id, summary="Done.")

    _drive(monkeypatch, loop, [_at(2026, 11, 9, 7) + n * MINUTE for n in range(release + 2)], kanban=caps,
           between=_between)
    statuses.append(_waiting())
    return {
        "made": loop.made,
        "statuses": statuses,
        "spawns": [(tick, board, title, who, by) for tick, board, _id, title, who, by in loop.spawned],
    }


def test_f4_full_planner_cap_ordinary_planning_card_waits_for_the_review_and_starts_once_it_completes(
    root, tmp_path, monkeypatch
):
    without = _f4_full_world(tmp_path / "without", monkeypatch, review=False, target="project")
    with_review = _f4_full_world(tmp_path / "with", monkeypatch, review=True, target="project")
    card = ("Plan the next milestone", PLANNER, None)

    # Review off: the planning card arrives after tick 1 and starts in tick 2.
    assert without["made"] == [0, 0, 0, 0]
    assert without["statuses"] == [None, None, "running", "running"]
    assert [spawn[0] for spawn in without["spawns"] if spawn[2:] == card] == [2]
    # Review on: the review holds the planner's one slot from tick 1, so the card stays ready in tick 2; the review
    # completes after tick 2 and the card starts in tick 3, the next tick.
    assert with_review["made"] == [1, 0, 0, 0]
    assert [spawn[:2] + spawn[3:] for spawn in with_review["spawns"] if spawn[-1] == CREATOR] == [
        (1, BOARD, PLANNER, CREATOR)
    ]
    assert with_review["statuses"] == [None, None, "ready", "running"]
    assert [spawn[0] for spawn in with_review["spawns"] if spawn[2:] == card] == [3]


def test_f4_full_per_profile_cap_new_summary_task_waits_for_the_review_and_starts_once_it_completes(
    root, tmp_path, monkeypatch
):
    without = _f4_full_world(tmp_path / "without", monkeypatch, review=False, target="summary")
    with_review = _f4_full_world(tmp_path / "with", monkeypatch, review=True, target="summary")
    summary = [(tick, board, who, by) for tick, board, _title, who, by in without["spawns"] if by == SUMMARY_CREATOR]

    # Review off: the summary tick starts the summary task in tick 2, and it starts in tick 3.
    assert without["made"] == [0, 0, 0, 0, 0]
    assert without["statuses"] == [None, None, "ready", "running", "running"]
    assert summary == [(3, BOARD, PLANNER, SUMMARY_CREATOR)]
    # Review on: the review task holds the planner's one slot on the summary board from tick 1, so the summary task
    # stays ready in tick 3; the review completes after tick 3 and the summary task starts in tick 4, the next tick.
    assert with_review["made"] == [1, 0, 0, 0, 0]
    assert [spawn[:2] + spawn[3:] for spawn in with_review["spawns"] if spawn[-1] == CREATOR] == [
        (1, BOARD, PLANNER, CREATOR)
    ]
    assert with_review["statuses"] == [None, None, "ready", "ready", "running"]
    assert [(tick, board, who, by) for tick, board, _title, who, by in with_review["spawns"]
            if by == SUMMARY_CREATOR] == [(4, BOARD, PLANNER, SUMMARY_CREATOR)]


# ---------------------------------------------------------------------------------------------------------------
# Store paths: under the environment the dispatcher gives a worker, the wired tick still reaches the root stores
# ---------------------------------------------------------------------------------------------------------------


def _enter_planning_worker(monkeypatch, home: Path, slug: str, task_id: str) -> None:
    """Take on the store pins kanban_db_dispatch._default_spawn gives a worker, resolved in the dispatcher first."""
    pins = {
        "HERMES_KANBAN_DB": str(kanban_db.kanban_db_path(board=slug)),
        "HERMES_KANBAN_WORKSPACES_ROOT": str(kanban_db.workspaces_root(board=slug)),
        "HERMES_KANBAN_BOARD": slug,
    }
    monkeypatch.setenv("HERMES_HOME", str(home / "profiles" / PLANNER))
    monkeypatch.setenv("HERMES_PROFILE", PLANNER)
    monkeypatch.setenv("HERMES_KANBAN_TASK", task_id)
    for name, value in pins.items():
        monkeypatch.setenv(name, value)


def test_the_wired_tick_under_the_worker_environment_reaches_the_root_registry_and_every_board(
    root, owner, loop, monkeypatch
):
    garden = _project(root, owner, "Garden Plan", _CARDS)
    desk = _project(root, owner, "Planning Desk", _CARDS)
    spare = _project(root, owner, "Spare Room", _CARDS)  # not allowlisted
    _switch_on(root, garden["project_id"], desk["project_id"])
    with _board(root, desk["board"]) as conn:
        worker_task = kanban_db.create_task(conn, title="Plan the next milestone", project_id=desk["project_id"])
    profile_home = root / "profiles" / PLANNER
    profile_home.mkdir(parents=True)
    with projects_db.connect_closing(profile_home / "projects.db"):
        pass
    # The profile's own config would keep the review off; the review reads the root's.
    (profile_home / "config.yaml").write_text(
        yaml.safe_dump({"owner_review": {"enabled": False, "allowlist": []}}), encoding="utf-8"
    )
    _enter_planning_worker(monkeypatch, root, desk["board"], worker_task)
    pinned = {name: os.environ[name] for name in ("HERMES_KANBAN_DB", "HERMES_KANBAN_BOARD", "HERMES_KANBAN_TASK")}
    own_board = _board_db(root, desk["board"])
    assert Path(pinned["HERMES_KANBAN_DB"]).resolve() == own_board.resolve()

    _drive(monkeypatch, loop, [_at(2026, 11, 9, 7)])

    # The production proof is what the wired review tick returned; the independent count is the rows on the root
    # summary board, read on a raw read-only connection.
    assert loop.jobs == [JOBS]
    on_board = _rows(
        _board_db(root, BOARD),
        "SELECT title, idempotency_key FROM tasks WHERE created_by = ? ORDER BY title", (CREATOR,),
    )
    assert loop.made == [len(on_board)] == [2]
    assert on_board == [
        ("Review the project: Garden Plan", _key(garden)),
        ("Review the project: Planning Desk", _key(desk)),
    ]
    assert _rows(
        root / "projects.db",
        "SELECT project_id FROM project_review_state WHERE open_task_id IS NOT NULL ORDER BY project_id",
    ) == sorted([(garden["project_id"],), (desk["project_id"],)])
    assert _state(root, spare) is None
    assert _budget(root) == (2, 0)
    assert _rows(own_board, "SELECT count(*) FROM tasks WHERE created_by = ?", (CREATOR,)) == [(0,)]
    assert _rows(profile_home / "projects.db", "SELECT count(*) FROM project_review_state") == [(0,)]
    assert _rows(profile_home / "projects.db", "SELECT count(*) FROM review_budget") == [(0,)]
    assert not _board_db(profile_home, BOARD).exists()
    assert {name: os.environ[name] for name in pinned} == pinned
