"""The journeys hold rule: ``hermes_cli.journeys_hold.state(now)``.

While the owner journeys of a release run, or are about to start, new work
waits. The rule reads the journeys files in the root home and the last
``release_finished`` event of the release record, writes nothing and never
raises: any error gives no hold.
"""

from __future__ import annotations

import sqlite3
import time
from datetime import datetime, timezone

import pytest

from agent import estop
from hermes_cli import journeys_hold, release_ledger
from hermes_cli import kanban_db as kb
from hermes_constants import get_default_hermes_root

MINUTE = 60
NOW = int(datetime(2026, 11, 26, 15, 50, tzinfo=timezone.utc).timestamp())
NO_HOLD = {"active": False, "reason": None, "latest_end": None}
# A run at work, started 5 minutes before NOW, as scripts/journeys/run_once.sh writes it.
RUN_AT_WORK = "result=started\nstep=\nreason=\nstarted=2026-11-26T15:45:00Z\n"


def _running(latest_end):
    return {"active": True, "reason": "running", "latest_end": latest_end}


def _starting(latest_end):
    return {"active": True, "reason": "starting", "latest_end": latest_end}


def _stamp(epoch):
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(epoch))


@pytest.fixture
def home():
    """The root home, with the owner journeys set up: their settings file is there."""
    root = get_default_hermes_root()
    (root / "journeys").mkdir(parents=True)
    (root / "journeys" / "settings").write_text("JOURNEYS_STACK=build\n", encoding="utf-8")
    return root


def _result(root, outcome, started, *, finished=None):
    """journeys/result in the form run_once.sh writes: a run that ended has a finished= line."""
    text = f"result={outcome}\nstep=\nreason=\nstarted={_stamp(started)}\n"
    if finished is not None:
        text += f"finished={_stamp(finished)}\n"
    (root / "journeys" / "result").write_text(text, encoding="utf-8")


def _last_end(conn):
    return conn.execute(
        "SELECT created_at FROM release_events WHERE kind = 'release_finished' ORDER BY id DESC LIMIT 1"
    ).fetchone()[0]


def _release(outcome="released", *, recovered=False):
    """One batch released through the release record's own writers; returns the
    time of its release_finished event. ``recovered``: its run was killed and a
    recovery finished the release."""
    conn = release_ledger.connect()
    try:
        batch = conn.execute(
            "INSERT INTO release_batches (state, version, tier, created_at) VALUES ('releasing', 1, 0, ?)",
            (int(time.time()),),
        ).lastrowid
        if recovered:
            release_ledger.begin_recovery(conn, batch)
        release_ledger.finish_release(conn, batch, outcome=outcome)
        return _last_end(conn)
    finally:
        conn.close()


def _settle_outside():
    """A release made outside the release command, recorded as
    release_ledger.settle_outside_release records it."""
    conn = release_ledger.connect()
    try:
        now = int(time.time())
        batch = conn.execute(
            "INSERT INTO release_batches (state, version, tier, created_at, outcome, outcome_at) "
            "VALUES ('released', 0, 0, ?, 'released', ?)",
            (now, now),
        ).lastrowid
        outside = {"outcome": "released", "outside": True}
        release_ledger._record_event(conn, batch, "release_finished", outside, now)
        return _last_end(conn)
    finally:
        conn.close()


def _tree(root):
    """Every path under ``root``, with the bytes of each file."""
    return {
        str(path.relative_to(root)): path.read_bytes() if path.is_file() else None
        for path in sorted(root.rglob("*"))
    }


def test_no_settings_file_never_holds():
    root = get_default_hermes_root()
    (root / "journeys").mkdir(parents=True)
    end = _release()
    assert journeys_hold.state(end + MINUTE) == NO_HOLD
    _result(root, "started", end + MINUTE)
    assert journeys_hold.state(end + 2 * MINUTE) == NO_HOLD

    (root / "journeys" / "settings").write_text("JOURNEYS_STACK=build\n", encoding="utf-8")
    assert journeys_hold.state(end + 2 * MINUTE) == _running(end + 46 * MINUTE)


def test_a_run_at_work_holds_until_45_minutes_after_its_start(home):
    assert journeys_hold.HOLD_LIMIT == 45 * MINUTE
    start = NOW - 5 * MINUTE
    _result(home, "started", start)

    assert journeys_hold.state(start) == _running(start + 45 * MINUTE)
    assert journeys_hold.state(NOW) == _running(start + 45 * MINUTE)
    assert journeys_hold.state(start + 45 * MINUTE - 1) == _running(start + 45 * MINUTE)
    assert journeys_hold.state(start + 45 * MINUTE) == NO_HOLD
    assert journeys_hold.state(start + 24 * 60 * MINUTE) == NO_HOLD


@pytest.mark.parametrize("outcome", ["passed", "failed", "started"])
def test_a_finished_result_ends_the_hold_at_once(home, outcome):
    start = NOW - 5 * MINUTE
    _result(home, "started", start)
    assert journeys_hold.state(NOW) == _running(start + 45 * MINUTE)

    _result(home, outcome, start, finished=NOW - MINUTE)
    assert journeys_hold.state(NOW) == NO_HOLD


def test_a_released_release_without_a_started_run_holds_for_10_minutes_then_not(home):
    assert journeys_hold.START_GRACE == 10 * MINUTE
    end = _release()

    assert journeys_hold.state(end) == _starting(end + 10 * MINUTE)
    assert journeys_hold.state(end + 10 * MINUTE - 1) == _starting(end + 10 * MINUTE)
    assert journeys_hold.state(end + 10 * MINUTE) == NO_HOLD
    # A release end later than now does not count.
    assert journeys_hold.state(end - 1) == NO_HOLD

    # The run of an earlier release, started before this release end, does not end the wait.
    _result(home, "passed", end - 30 * MINUTE, finished=end - 20 * MINUTE)
    assert journeys_hold.state(end + MINUTE) == _starting(end + 10 * MINUTE)

    # The journeys of this release start: from then on the run at work holds.
    _result(home, "started", end + MINUTE)
    assert journeys_hold.state(end + 2 * MINUTE) == _running(end + 46 * MINUTE)


@pytest.mark.parametrize("outcome", ["passed", "failed"])
def test_a_run_that_started_at_or_after_the_release_end_and_ended_never_holds(home, outcome):
    end = _release()
    _result(home, outcome, end - 1, finished=end + 3 * MINUTE)
    assert journeys_hold.state(end + 4 * MINUTE) == _starting(end + 10 * MINUTE)

    _result(home, outcome, end, finished=end + 3 * MINUTE)
    assert journeys_hold.state(end + 4 * MINUTE) == NO_HOLD
    _result(home, outcome, end + MINUTE, finished=end + 3 * MINUTE)
    assert journeys_hold.state(end + 4 * MINUTE) == NO_HOLD


@pytest.mark.parametrize("ending", ["refused", "restored", "failed", "outside"])
def test_a_refused_restored_failed_or_settled_outside_release_never_holds(home, ending):
    end = _release()
    assert journeys_hold.state(end + MINUTE) == _starting(end + 10 * MINUTE)

    # The last release_finished event of the record decides.
    end = _settle_outside() if ending == "outside" else _release(ending)
    assert journeys_hold.state(end + MINUTE) == NO_HOLD


def test_a_release_finished_by_a_recovery_never_holds(home):
    end = _release(recovered=True)
    assert journeys_hold.state(end + MINUTE) == NO_HOLD

    # A recovery of an earlier batch does not touch the next release.
    end = _release()
    assert journeys_hold.state(end + MINUTE) == _starting(end + 10 * MINUTE)


UNKNOWN_RESULTS = {
    "not UTF-8": RUN_AT_WORK.encode() + b"journey=\xff\n",
    "a line that is no NAME=value": (RUN_AT_WORK + "journey 3\n").encode(),
    "an empty line": RUN_AT_WORK.replace("\nstep=", "\n\nstep=").encode(),
    "an unknown result": RUN_AT_WORK.replace("result=started", "result=running").encode(),
    "no start time": RUN_AT_WORK.replace("started=2026-11-26T15:45:00Z\n", "").encode(),
    "a start time in another form": RUN_AT_WORK.replace("T15:45:00Z", " 15:45:00").encode(),
    "a start time that is no time": RUN_AT_WORK.replace("-26T", "-31T").encode(),
}


@pytest.mark.parametrize("damage", ["a folder", *UNKNOWN_RESULTS])
def test_an_unreadable_or_unknown_result_never_holds(home, damage):
    path = home / "journeys" / "result"
    path.write_text(RUN_AT_WORK, encoding="utf-8")
    assert journeys_hold.state(NOW) == _running(NOW + 40 * MINUTE)

    if damage == "a folder":
        path.unlink()
        path.mkdir()
    else:
        path.write_bytes(UNKNOWN_RESULTS[damage])
    assert journeys_hold.state(NOW) == NO_HOLD


UNKNOWN_RECORDS = {
    "no events table": "DROP TABLE release_events",
    "a payload that is no JSON": "UPDATE release_events SET payload = 'released'",
    "a payload that is no object": "UPDATE release_events SET payload = '[\"released\"]'",
    "a payload with no outcome": "UPDATE release_events SET payload = '{}'",
    "a time that is no number": "UPDATE release_events SET created_at = 'yesterday'",
}


@pytest.mark.parametrize("damage", ["no record", "a folder", "not a database", *UNKNOWN_RECORDS])
def test_an_unreadable_or_unknown_record_never_holds(home, damage):
    end = _release()
    assert journeys_hold.state(end + MINUTE) == _starting(end + 10 * MINUTE)

    path = release_ledger.ledger_path()
    if damage in ("no record", "a folder", "not a database"):
        path.unlink()
        if damage == "a folder":
            path.mkdir()
        elif damage == "not a database":
            path.write_bytes(b"this is no SQLite database\n" * 64)
    else:
        conn = sqlite3.connect(path)
        try:
            conn.execute(UNKNOWN_RECORDS[damage])
            conn.commit()
        finally:
            conn.close()
    assert journeys_hold.state(end + MINUTE) == NO_HOLD


def test_a_busy_record_gives_no_hold_without_waiting(home, monkeypatch):
    # The dispatcher asks the rule on the gateway event loop: a record that a
    # writer holds gives no hold at once and is never waited for.
    end = _release()
    assert journeys_hold.state(end + MINUTE) == _starting(end + 10 * MINUTE)

    real_connect = sqlite3.connect
    busy_timeouts = []

    def _spy(*args, **kwargs):
        conn = real_connect(*args, **kwargs)
        busy_timeouts.append(conn.execute("PRAGMA busy_timeout").fetchone()[0])
        return conn

    writer = sqlite3.connect(release_ledger.ledger_path(), isolation_level=None)
    try:
        writer.execute("BEGIN EXCLUSIVE")  # the record keeps a rollback journal: no reader gets in
        with monkeypatch.context() as patch:
            patch.setattr(journeys_hold.sqlite3, "connect", _spy)
            assert journeys_hold.state(end + MINUTE) == NO_HOLD
    finally:
        writer.close()
    assert busy_timeouts == [0]


def test_a_start_time_later_than_now_never_holds_from_line_b(home):
    _result(home, "started", NOW + MINUTE)
    assert journeys_hold.state(NOW) == NO_HOLD
    assert journeys_hold.state(NOW + MINUTE) == _running(NOW + 46 * MINUTE)


def test_an_error_inside_the_rule_gives_no_hold_and_raises_nothing(home, monkeypatch):
    def _broken():
        raise RuntimeError("a path cannot be resolved")

    end = _release()
    _result(home, "started", end + MINUTE)
    assert journeys_hold.state(end + 2 * MINUTE) == _running(end + 46 * MINUTE)
    with monkeypatch.context() as patch:
        patch.setattr(journeys_hold, "get_default_hermes_root", _broken)
        assert journeys_hold.state(end + 2 * MINUTE) == NO_HOLD
    assert journeys_hold.state(None) == NO_HOLD

    _result(home, "passed", end - 30 * MINUTE, finished=end - 20 * MINUTE)
    assert journeys_hold.state(end + 2 * MINUTE) == _starting(end + 10 * MINUTE)
    with monkeypatch.context() as patch:
        patch.setattr(release_ledger, "ledger_path", _broken)
        assert journeys_hold.state(end + 2 * MINUTE) == NO_HOLD


def test_the_reader_writes_nothing(home):
    end = _release()
    # A run at work that started 44 minutes before the release end: within 30
    # seconds of the end it holds, then the release end holds, then nothing.
    _result(home, "started", end - 44 * MINUTE)
    estop.engage("owner pause")
    kb.init_db()
    kb.create_board(slug="proj", name="Proj")
    with kb.connect_closing(board="proj") as conn:
        kb.create_task(conn, title="proj card", assignee="profile-a")
    kb.write_board_metadata("proj", dispatch_enabled=True, dispatch_paused_by_owner=True)

    before = _tree(home)
    for path in (
        release_ledger.ledger_path(),
        home / "journeys" / "result",
        estop.sentinel_path(),
        kb.kanban_db_path(),
        kb.kanban_db_path("proj"),
        kb.board_metadata_path("proj"),
    ):
        assert before[str(path.relative_to(home))] is not None, path

    assert journeys_hold.state(end + 30)["reason"] == "running"
    assert journeys_hold.state(end + 2 * MINUTE)["reason"] == "starting"
    assert journeys_hold.state(end + 20 * MINUTE) == NO_HOLD
    assert _tree(home) == before


def test_a_worker_environment_still_reads_the_root_record_and_result(home, monkeypatch):
    end = _release()
    _result(home, "started", end + MINUTE)
    # A profile home of its own, without journeys settings and with a run that ended.
    profile = home / "profiles" / "worker"
    (profile / "journeys").mkdir(parents=True)
    (profile / "journeys" / "result").write_text(
        f"result=passed\nstep=\nreason=\nstarted={_stamp(end + MINUTE)}\nfinished={_stamp(end + MINUTE)}\n",
        encoding="utf-8",
    )
    kb.create_board(slug="proj", name="Proj")

    monkeypatch.setenv("HERMES_HOME", str(profile))
    monkeypatch.setenv("HERMES_KANBAN_DB", str(kb.kanban_db_path("proj")))
    monkeypatch.setenv("HERMES_KANBAN_BOARD", "proj")
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_worker")
    assert get_default_hermes_root() == home

    assert journeys_hold.state(end + 2 * MINUTE) == _running(end + 46 * MINUTE)
    _result(home, "passed", end - 30 * MINUTE, finished=end - 20 * MINUTE)
    assert journeys_hold.state(end + 2 * MINUTE) == _starting(end + 10 * MINUTE)
