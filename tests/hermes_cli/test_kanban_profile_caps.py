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
import sqlite3
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


class _LockedReads:
    """Stands in for ``count_running_by_assignee``. After ``fail_next(board)``
    the next read of that board's running counts raises ``database is
    locked`` (or the given message) once, as a transient lock would; every
    other read is real."""

    def __init__(self, kb):
        self._kb = kb
        self._count = kb.count_running_by_assignee
        self._pending = []

    def fail_next(self, board, message="database is locked"):
        self._pending.append(
            (self._kb.kanban_db_path(board=board).resolve(), message)
        )

    def __call__(self, conn):
        db = Path(conn.execute("PRAGMA database_list").fetchone()[2]).resolve()
        for pending in self._pending:
            if pending[0] == db:
                self._pending.remove(pending)
                raise sqlite3.OperationalError(pending[1])
        return self._count(conn)


@pytest.fixture
def locked_reads(kb, monkeypatch):
    reads = _LockedReads(kb)
    monkeypatch.setattr(kb, "count_running_by_assignee", reads)
    return reads


def _ready(kb, assignee, *, board=None):
    with kb.connect_closing(board=board or kb.DEFAULT_BOARD) as conn:
        return kb.create_task(conn, title=f"{assignee} work", assignee=assignee)


def _running(kb, assignee, *, board=None):
    task_id = _ready(kb, assignee, board=board)
    with kb.connect_closing(board=board or kb.DEFAULT_BOARD) as conn:
        assert kb.claim_task(conn, task_id) is not None
    return task_id


def _in_review(kb, assignee, *, board=None):
    """A card parked in the review lane, waiting for a reviewer spawn."""
    task_id = _ready(kb, assignee, board=board)
    with kb.connect_closing(board=board or kb.DEFAULT_BOARD) as conn:
        conn.execute(
            "UPDATE tasks SET status = 'review' WHERE id = ?", (task_id,)
        )
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


def _cap_info_lines(caplog, named_cap):
    """Info lines saying ``CAPPED`` waited only because ``named_cap`` is full."""
    return [
        r.getMessage() for r in caplog.records
        if r.levelno == logging.INFO
        and named_cap in r.getMessage()
        and CAPPED in r.getMessage()
        and "full" in r.getMessage()
    ]


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


# b. the running task that fills the cap lives on a board the owner paused
def test_listed_profile_cap_counts_running_task_on_a_paused_board(
    kb, spawns, monkeypatch
):
    # A pause stops new claims on its board; its running worker still runs.
    for board in (kb.DEFAULT_BOARD, SECOND_BOARD):
        kb.write_board_metadata(board, dispatch_enabled=True)
    _running(kb, CAPPED, board=SECOND_BOARD)
    kb.write_board_metadata(SECOND_BOARD, dispatch_paused_by_owner=True)
    capped_next = _ready(kb, CAPPED)
    other_next = _ready(kb, OTHER)

    _run_dispatcher(
        monkeypatch,
        {
            "max_in_progress_by_profile": {CAPPED: 1},
            "dispatch_require_board_activation": True,
        },
        ticks=1,
    )

    assert [(a, t) for _, a, t in spawns] == [(OTHER, other_next)]
    assert _status(kb, capped_next) == "ready"


# b. a board whose DB cannot be opened is unknown, not empty: the listed
# profile waits for that tick
def test_listed_profile_waits_when_a_board_db_cannot_be_opened(
    kb, spawns, monkeypatch
):
    _running(kb, CAPPED, board=SECOND_BOARD)
    capped_next = _ready(kb, CAPPED)
    other_next = _ready(kb, OTHER)
    # board.json keeps the board listed; its DB becomes a symlink loop, which
    # fails every stat and open with ELOOP, not with a missing file.
    kb.write_board_metadata(SECOND_BOARD, name="Two")
    db = kb.kanban_db_path(board=SECOND_BOARD)
    db.rename(db.with_name("kanban.db.moved"))
    db.symlink_to(db.name)

    _run_dispatcher(
        monkeypatch, {"max_in_progress_by_profile": {CAPPED: 1}}, ticks=1
    )

    assert [(a, t) for _, a, t in spawns] == [(OTHER, other_next)]
    assert _status(kb, capped_next) == "ready"


# b. counting reads a paused board without changing it: no migration, no
# repair, no log line with its path, and no DB for a board that has none
def test_counting_reads_boards_without_changing_them(
    kb, spawns, monkeypatch, caplog
):
    legacy, empty = "board-legacy", "board-empty"
    kb.write_board_metadata(kb.DEFAULT_BOARD, dispatch_enabled=True)
    capped_next = _ready(kb, CAPPED)
    # An older paused board: only the columns the count reads, one running
    # task of the capped profile, written without the platform's opener.
    legacy_db = kb.kanban_db_path(board=legacy)
    legacy_db.parent.mkdir(parents=True)
    with sqlite3.connect(legacy_db) as conn:
        conn.execute(
            "CREATE TABLE tasks (id TEXT PRIMARY KEY, status TEXT, "
            "assignee TEXT, task_kind TEXT)"
        )
        conn.execute(
            "INSERT INTO tasks VALUES ('t_old', 'running', ?, 'work')", (CAPPED,)
        )
    conn.close()
    kb.write_board_metadata(
        legacy, name="Legacy", dispatch_enabled=True, dispatch_paused_by_owner=True
    )
    # A board that was never activated and has no DB yet.
    kb.write_board_metadata(empty, name="Empty")
    legacy_before = legacy_db.read_bytes()

    with caplog.at_level(logging.DEBUG):
        _run_dispatcher(
            monkeypatch,
            {
                "max_in_progress_by_profile": {CAPPED: 1},
                "dispatch_require_board_activation": True,
            },
            ticks=1,
        )

    assert spawns == []
    assert _status(kb, capped_next) == "ready"
    assert legacy_db.read_bytes() == legacy_before
    assert not kb.kanban_db_path(board=empty).exists()
    assert not any(str(legacy_db) in r.getMessage() for r in caplog.records)


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


# F1. a board whose running count cannot be read holds listed profiles on
# every board for that tick only; profiles not listed start as usual
def test_unreadable_running_count_holds_listed_profile_for_that_tick(
    kb, spawns, locked_reads, monkeypatch, caplog
):
    busy = _running(kb, CAPPED, board=SECOND_BOARD)
    capped_next = _ready(kb, CAPPED)
    other_next = _ready(kb, OTHER)
    cfg = {"max_in_progress_by_profile": {CAPPED: 1}}

    locked_reads.fail_next(SECOND_BOARD)
    with caplog.at_level(logging.WARNING, logger="gateway.run"):
        _run_dispatcher(monkeypatch, cfg, ticks=1)

    assert [(a, t) for _, a, t in spawns] == [(OTHER, other_next)]
    assert _status(kb, capped_next) == "ready"
    naming_board = [
        r.getMessage() for r in caplog.records
        if r.levelno >= logging.WARNING and SECOND_BOARD in r.getMessage()
    ]
    assert len(naming_board) == 1, naming_board

    # Profile-a runs nowhere now. A tick whose read fails still holds its
    # card; the next tick reads every board again and starts it.
    with kb.connect_closing(board=SECOND_BOARD) as conn:
        assert kb.archive_task(conn, busy)
    spawns.clear()
    locked_reads.fail_next(SECOND_BOARD)
    _run_dispatcher(monkeypatch, cfg, ticks=1)
    assert spawns == []
    assert _status(kb, capped_next) == "ready"

    locked_reads.fail_next(SECOND_BOARD)
    _run_dispatcher(monkeypatch, cfg, ticks=2)
    assert [(a, t) for _, a, t in spawns] == [(CAPPED, capped_next)]


# F2. the mapping never holds the review lane: a listed profile's review
# starts, and keeps its reservation, exactly as with the mapping unset
def test_listed_cap_leaves_the_review_lane_as_without_the_mapping(
    kb, spawns, monkeypatch
):
    _running(kb, CAPPED, board=SECOND_BOARD)
    review = _in_review(kb, CAPPED)
    _ready(kb, OTHER)

    _run_dispatcher(
        monkeypatch,
        {
            "review_dispatch": True,
            "max_spawn": 1,
            "max_in_progress": 10,
            "max_in_progress_by_profile": {CAPPED: 1},
        },
        ticks=1,
    )

    assert [(a, t) for _, a, t in spawns] == [(CAPPED, review)]


# S1. a claim whose spawn preparation fails still counts for the rest of
# the tick, so the same profile's card on the next board waits
def test_claim_that_fails_before_spawn_still_holds_the_next_board(
    kb, spawns, monkeypatch
):
    first = _ready(kb, CAPPED)
    second = _ready(kb, CAPPED, board=SECOND_BOARD)
    real_set_workspace_path = kb.set_workspace_path
    calls = []

    def _fail_first(conn, task_id, workspace):
        calls.append(task_id)
        if len(calls) == 1:
            raise RuntimeError("placeholder failure after the claim")
        return real_set_workspace_path(conn, task_id, workspace)

    monkeypatch.setattr(kb, "set_workspace_path", _fail_first)
    _run_dispatcher(
        monkeypatch, {"max_in_progress_by_profile": {CAPPED: 1}}, ticks=1
    )

    assert len(calls) == 1
    assert spawns == []
    assert sorted(
        [_status(kb, first), _status(kb, second, board=SECOND_BOARD)]
    ) == ["ready", "running"]


# S2. a card reassigned to a profile at its cap between the cap check and
# the claim is not claimed in that tick
def test_card_reassigned_to_capped_profile_before_claim_is_not_claimed(
    kb, spawns, monkeypatch
):
    _running(kb, CAPPED)
    moved = _ready(kb, OTHER)
    real_route_check = kb.route_authority_error
    reassigned = []

    def _reassign_after_last_check(conn, task_id, *args, **kwargs):
        # The dispatcher's last check before the claim passes, then the
        # card moves to the capped profile, as a concurrent edit could.
        verdict = real_route_check(conn, task_id, *args, **kwargs)
        if task_id == moved and not reassigned:
            reassigned.append(task_id)
            with kb.connect_closing(board=kb.DEFAULT_BOARD) as other_conn:
                other_conn.execute(
                    "UPDATE tasks SET assignee = ? WHERE id = ?", (CAPPED, moved)
                )
        return verdict

    monkeypatch.setattr(kb, "route_authority_error", _reassign_after_last_check)
    _run_dispatcher(
        monkeypatch, {"max_in_progress_by_profile": {CAPPED: 1}}, ticks=1
    )

    assert reassigned == [moved]
    assert spawns == []
    assert _status(kb, moved) == "ready"


# S3. warnings name profiles and boards only: never a raw setting value or
# the text of the error that made a board unreadable
def test_warnings_carry_no_raw_setting_value_or_error_text(
    kb, spawns, locked_reads, monkeypatch, caplog
):
    _running(kb, CAPPED, board=SECOND_BOARD)
    locked_reads.fail_next(SECOND_BOARD, message="PLACEHOLDER-PATH is locked")

    with caplog.at_level(logging.WARNING, logger="gateway.run"):
        _run_dispatcher(
            monkeypatch,
            {"max_in_progress_by_profile": {CAPPED: 1, OTHER: "PLACEHOLDER-VALUE"}},
            ticks=1,
        )

    text = " ".join(r.getMessage() for r in caplog.records)
    assert SECOND_BOARD in text
    assert "PLACEHOLDER-PATH" not in text
    assert "PLACEHOLDER-VALUE" not in text

    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="gateway.run"):
        _run_dispatcher(
            monkeypatch, {"max_in_progress_by_profile": "PLACEHOLDER-MAPPING"}, ticks=1
        )

    assert not any("PLACEHOLDER-MAPPING" in r.getMessage() for r in caplog.records)


# F4. mapping unset: a ready card the review reservation keeps the ready
# lane from reaching, whose profile is at the integer cap, is not stuck.
# max_spawn is 2, not 1: at 1 the running card already fills the board and
# the tick returns before the reservation or either lane.
def test_integer_capped_card_behind_review_reservation_is_not_stuck(
    kb, spawns, monkeypatch, caplog
):
    _running(kb, CAPPED)
    _in_review(kb, CAPPED)
    capped_next = _ready(kb, CAPPED)
    cfg = {
        "review_dispatch": True,
        "max_spawn": 2,
        "max_in_progress": 10,
        "max_in_progress_per_profile": 1,
        "failure_limit": 100,
    }

    with caplog.at_level(logging.INFO, logger="gateway.run"):
        _run_dispatcher(monkeypatch, cfg, ticks=7)

    messages = [r.getMessage() for r in caplog.records]
    assert not any(STUCK_LINE in m for m in messages)
    cap_lines = _cap_info_lines(caplog, "kanban.max_in_progress_per_profile")
    assert len(cap_lines) == 1, messages
    assert spawns == []
    assert _status(kb, capped_next) == "ready"

    # Same held cards, plus a card that waits for another reason: stuck.
    spawns.failing.add(OTHER)
    _ready(kb, OTHER)
    caplog.clear()
    with caplog.at_level(logging.INFO, logger="gateway.run"):
        _run_dispatcher(monkeypatch, cfg, ticks=7)

    assert any(STUCK_LINE in r.getMessage() for r in caplog.records)
    assert spawns == []
