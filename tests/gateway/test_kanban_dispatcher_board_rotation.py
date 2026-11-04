"""Board rotation in the gateway's embedded kanban dispatcher.

On every tick the dispatch pass and the auto-decompose (triage) pass each give
the first turn to the next dispatchable board, in a fixed order by slug that
wraps around. Serving the boards in the same order on every tick let the
boards early in the list take every free slot while a later board's ready
cards waited.

Each test drives the gateway's dispatcher loop for several ticks over real
boards in the per-test HERMES_HOME, with recording stand-ins for the worker
spawn and the decomposer. Between ticks, the workers the last tick started
finish. Board and profile names are placeholders.
"""
from __future__ import annotations

import asyncio
import os

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_decompose

WORKER = "profile-a"
DEFAULT = kb.DEFAULT_BOARD
# Each sorts after "default", so the expected order holds whether or not the
# default board is placed first.
OPS = "ops"
RESEARCH = "research"
WEB = "web"


class _Served:
    """What the loop served, each with the tick (counted from 0) it ran in."""

    def __init__(self):
        self.ticks = 0
        self.spawned = []  # (tick, board, task id): a ready card's worker
        self.decomposed = []  # (tick, board): a triage card decomposed

    def boards_by_tick(self, records):
        """For each finished tick, the boards in the order they were served."""
        return [
            [board for tick, board, *_ in records if tick == n]
            for n in range(self.ticks)
        ]


@pytest.fixture
def served(monkeypatch):
    """Real boards in the per-test HERMES_HOME; spawns and decompositions
    are recorded instead of run."""
    import hermes_cli.profiles as profiles

    monkeypatch.setattr(profiles, "profile_exists", lambda name: True)
    kb.init_db()
    kb.create_board(slug=DEFAULT, name="Default")
    record = _Served()

    def _spawn(task, workspace, board=None):
        # No PID: the card stays running until the test finishes its worker.
        record.spawned.append((record.ticks, board, task.id))
        return None

    def _list_triage_ids(*, tenant=None):
        # Every board always has a card waiting in triage.
        return [f"triage-{os.environ['HERMES_KANBAN_BOARD']}"]

    def _decompose_task(task_id, *, author=None, timeout=None):
        record.decomposed.append((record.ticks, os.environ["HERMES_KANBAN_BOARD"]))
        return kanban_decompose.DecomposeOutcome(task_id, True)

    monkeypatch.setattr(kb, "_default_spawn", _spawn)
    monkeypatch.setattr(kanban_decompose, "list_triage_ids", _list_triage_ids)
    monkeypatch.setattr(kanban_decompose, "decompose_task", _decompose_task)
    return record


def _board(slug, *, cards, **metadata):
    """A board holding ``cards`` ready cards for WORKER."""
    if slug != DEFAULT:
        kb.create_board(slug=slug, name=slug)
    with kb.connect_closing(board=slug) as conn:
        for n in range(cards):
            kb.create_task(conn, title=f"{slug} card {n}", assignee=WORKER)
    if metadata:
        kb.write_board_metadata(slug, **metadata)


def _finish_workers(served):
    """The workers the last tick started finish: archiving their cards frees
    their slots."""
    for tick, board, task_id in served.spawned:
        if tick == served.ticks - 1:
            with kb.connect_closing(board=board) as conn:
                assert kb.archive_task(conn, task_id)


def _run_dispatcher(monkeypatch, served, kanban_cfg, *, ticks, between_ticks=None):
    """Run the gateway dispatcher loop for ``ticks`` full ticks.

    In the pause after each tick but the last, the workers that tick started
    finish, then ``between_ticks()`` runs; ``served.ticks`` is the number of
    ticks finished.
    """
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

    async def _to_thread(fn, *args, **kwargs):
        result = fn(*args, **kwargs)
        if getattr(fn, "__name__", "") == "_tick_once":
            served.ticks += 1
            if served.ticks >= ticks:
                # The rest of this tick still runs; the next one never starts.
                runner._running = False
        return result

    paused_after = 0

    async def _sleep(_delay):
        # The loop's sleep sits outside its per-tick error handling, so a
        # failure here fails the test instead of only being logged.
        nonlocal paused_after
        if paused_after < served.ticks:
            paused_after = served.ticks
            _finish_workers(served)
            if between_ticks is not None:
                between_ticks()

    monkeypatch.setattr("gateway.run.asyncio.to_thread", _to_thread)
    monkeypatch.setattr("gateway.run.asyncio.sleep", _sleep)
    asyncio.run(
        asyncio.wait_for(runner._kanban_dispatcher_watcher(), timeout=60.0)
    )
    assert served.ticks == ticks


# Each board takes the first turn in turn, and every board is still served
# on every tick. max_spawn 1 lets a board run one worker at a time, so with
# the workers finishing between ticks each board starts one card per tick.
def test_each_board_goes_first_in_turn(served, monkeypatch):
    for slug in (DEFAULT, OPS, WEB):
        _board(slug, cards=4)

    _run_dispatcher(monkeypatch, served, {"max_spawn": 1}, ticks=4)

    assert served.boards_by_tick(served.spawned) == [
        [DEFAULT, OPS, WEB],
        [OPS, WEB, DEFAULT],
        [WEB, DEFAULT, OPS],
        [DEFAULT, OPS, WEB],
    ]


# The triage pass rotates its start too, within its per-tick cap. Each pass
# keeps a start of its own: with one shared start, the dispatch pass would
# begin each tick after the board the triage pass began at.
def test_auto_decompose_rotates_its_start(served, monkeypatch):
    for slug in (DEFAULT, OPS, WEB):
        _board(slug, cards=4)

    _run_dispatcher(
        monkeypatch,
        served,
        {"auto_decompose": True, "auto_decompose_per_tick": 1, "max_spawn": 1},
        ticks=4,
    )

    assert served.boards_by_tick(served.decomposed) == [
        [DEFAULT], [OPS], [WEB], [DEFAULT],
    ]
    assert served.boards_by_tick(served.spawned) == [
        [DEFAULT, OPS, WEB],
        [OPS, WEB, DEFAULT],
        [WEB, DEFAULT, OPS],
        [DEFAULT, OPS, WEB],
    ]


# A board the activation filter rejects (here: its owner paused it) is never
# served and never takes a turn, in either pass: with boards A, B (rejected)
# and C, the first turn goes A, C, A, C.
def test_a_rejected_board_never_takes_a_turn(served, monkeypatch):
    _board(DEFAULT, cards=4, dispatch_enabled=True)
    _board(OPS, cards=4, dispatch_enabled=True, dispatch_paused_by_owner=True)
    _board(WEB, cards=4, dispatch_enabled=True)

    _run_dispatcher(
        monkeypatch,
        served,
        {
            "dispatch_require_board_activation": True,
            "auto_decompose": True,
            "auto_decompose_per_tick": 1,
            "max_spawn": 1,
        },
        ticks=4,
    )

    assert served.boards_by_tick(served.spawned) == [
        [DEFAULT, WEB], [WEB, DEFAULT], [DEFAULT, WEB], [WEB, DEFAULT],
    ]
    assert served.boards_by_tick(served.decomposed) == [
        [DEFAULT], [WEB], [DEFAULT], [WEB],
    ]


# A cap that spans every board still holds: at 1, one worker starts per tick
# in total across both boards. When it finishes, the freed slot goes to the
# other board on the next tick instead of back to the board that went first.
@pytest.mark.parametrize(
    "cap",
    [{"max_in_progress_by_profile": {WORKER: 1}}, {"max_in_progress": 1}],
    ids=["profile-cap", "host-cap"],
)
def test_the_freed_slot_goes_to_the_other_board_next_tick(served, monkeypatch, cap):
    for slug in (DEFAULT, WEB):
        _board(slug, cards=2)

    _run_dispatcher(monkeypatch, served, cap, ticks=4)

    assert served.boards_by_tick(served.spawned) == [
        [DEFAULT], [WEB], [DEFAULT], [WEB],
    ]


# The turn is kept by slug, not by list position, so it carries on when the
# board list changes: a board added after the first tick joins in its place
# in the order, and when the board that went first is archived, the board
# after it takes the next turn (a kept position would skip that board).
def test_rotation_survives_a_board_being_added_or_archived(served, monkeypatch):
    for slug in (DEFAULT, OPS, WEB):
        _board(slug, cards=5)

    def _change_boards():
        if served.ticks == 1:
            _board(RESEARCH, cards=5)  # sorts between ops and web
        elif served.ticks == 2:
            kb.remove_board(OPS)  # archived right after it went first

    _run_dispatcher(
        monkeypatch,
        served,
        {"auto_decompose": True, "auto_decompose_per_tick": 1, "max_spawn": 1},
        ticks=5,
        between_ticks=_change_boards,
    )

    assert served.boards_by_tick(served.spawned) == [
        [DEFAULT, OPS, WEB],
        [OPS, RESEARCH, WEB, DEFAULT],
        [RESEARCH, WEB, DEFAULT],
        [WEB, DEFAULT, RESEARCH],
        [DEFAULT, RESEARCH, WEB],
    ]
    assert served.boards_by_tick(served.decomposed) == [
        [DEFAULT], [OPS], [RESEARCH], [WEB], [DEFAULT],
    ]
