"""A review that the harvest refuses tells the next review task of the same Vienna date why.

The refusal's stable code, its detail and its review date are stored on the Project's review state. The next review task
of that date carries one line that names the reason, and an accepted review, another day or an unreadable value carries
none. Reviews are driven through the real tick and a plain completion that skips the hand-over, so the harvest alone
judges the result. The stores are read back with plain read-only SQLite, without the code under test.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import time
from pathlib import Path

import yaml

from hermes_cli import kanban_db, owner_project_review as review, projects_db
from hermes_cli.owner_project_review_refusal import refusal_sentences
from tests.hermes_cli.test_owner_project_review_result import NOW, _quiet_project, root  # noqa: F401

SUMMARIES = "project-summaries"
DAY = "2026-11-09"
YESTERDAY = "2026-11-08"
TOMORROW = "2026-11-10"
MARKER = "Zebra"
LEAD_IN = "Your last result for this date was refused: "
HOUR = 3600
LENGTH = "review_text_length_invalid"
COVERAGE = "review_class_coverage_invalid"
LONG_DETAIL = {"fields": [{"field": "summary.outcome", "rule": "length", "length": 412, "limit": 400}]}
INVENTED_DETAIL = {"missing": [], "extra": [], "extra_other": 1}


def _read(path, sql, *args):
    """The independent read: plain read-only SQLite on the file, without the code under test."""
    conn = sqlite3.connect(f"{Path(path).resolve().as_uri()}?mode=ro", uri=True)
    try:
        return conn.execute(sql, args).fetchall()
    finally:
        conn.close()


def _text(length, unit="The room is booked for the meeting. "):
    """Plain text of exactly ``length`` characters that stripping does not shorten."""
    return (unit * (length // len(unit) + 1))[:length - 1] + "."


def _payload(outcome="The meeting has a room.", classification=()):
    return {"project_review": {
        "summary": {"outcome": outcome, "next_action": "Book the chosen room.", "owner_decision": None},
        "brief": {"headline": "The room is chosen.", "what_changed": "Book the room now has a chosen room.",
                  "next_step": "Book the chosen room.", "needs_you": None},
        "classification": list(classification), "proposed_closures": [],
    }}


def _clock(monkeypatch, now):
    monkeypatch.setattr(time, "time", lambda: now)


def _key(project, day=DAY, failures=0, again=False):
    return f"project-review:{project.id}:{day}:{failures}" + (":again" if again else "")


def _owner_comment(project):
    with kanban_db.connect_closing(kanban_db.kanban_db_path(board=project.board_slug), board=project.board_slug) as board:
        work = board.execute("SELECT id FROM tasks WHERE task_kind='work'").fetchone()[0]
        kanban_db.add_comment(board, work, "default", "The room is chosen.")


def _news_project(root, monkeypatch, *, at=NOW, created_at=NOW - 86400):
    """A Project whose owner commented since its last review, with the first review task the tick opens for it."""
    project = _quiet_project(root, created_at=created_at)
    _clock(monkeypatch, at)
    _owner_comment(project)
    assert review.review_tick(at) == 1
    return project


def _summary_db(root):
    return root / "kanban" / "boards" / SUMMARIES / "kanban.db"


def _body(root, key):
    ((body,),) = _read(_summary_db(root), "SELECT body FROM tasks WHERE idempotency_key=?", key)
    return body


def _finish(root, key, metadata):
    """The worker's completion without the hand-over: only the harvest judges the result."""
    ((task_id,),) = _read(_summary_db(root), "SELECT id FROM tasks WHERE idempotency_key=?", key)
    with kanban_db.connect_closing(kanban_db.kanban_db_path(board=SUMMARIES), board=SUMMARIES) as board:
        assert kanban_db.claim_task(board, task_id) is not None
        assert kanban_db.complete_task(board, task_id, summary="Reviewed.", metadata=metadata)


def _stored(root, project):
    ((value,),) = _read(root / "projects.db", "SELECT last_refusal FROM project_review_state WHERE project_id=?",
                        project.id)
    return value


def _write_state(root, project, **columns):
    assignments = ", ".join(f"{name} = ?" for name in columns)
    with projects_db.connect_closing(root / "projects.db") as registry:
        with kanban_db.write_txn(registry):
            registry.execute(f"UPDATE project_review_state SET {assignments} WHERE project_id = ?",
                             (*columns.values(), project.id))


def _refusal_lines(body):
    return [line for line in body.splitlines() if "refused" in line]


def test_a_retry_of_the_same_day_names_the_last_refusal(root, monkeypatch):
    project = _news_project(root, monkeypatch)
    first = _body(root, _key(project))
    assert _refusal_lines(first) == []

    _finish(root, _key(project), _payload(_text(412, f"{MARKER} notes on the room. ")))
    assert review.review_tick(NOW + 1) == 0
    stored = _stored(root, project)
    assert json.loads(stored) == {"date": DAY, "code": LENGTH, "detail": LONG_DETAIL}
    assert MARKER not in stored and project.name not in stored

    # The failure count, the backoff and the retry key stay as they were.
    assert review.review_tick(NOW + HOUR) == 0
    _clock(monkeypatch, NOW + 1 + HOUR)
    assert review.review_tick(NOW + 1 + HOUR) == 1
    retry = _body(root, _key(project, failures=1))
    line = LEAD_IN + refusal_sentences(LENGTH, LONG_DETAIL)
    assert "summary.outcome has 412 characters; at most 400." in line and project.name not in line
    assert _refusal_lines(retry) == [line]
    assert MARKER not in retry
    # The line is the only addition, right after the plan proposal schema and before the review date.
    assert retry.replace(f"{line}\n", "", 1) == first
    lines = retry.splitlines()
    at = lines.index(line)
    assert lines[at - 1].startswith("Plan proposal schema: ") and lines[at + 1] == f"Review date: {DAY}"

    # A later refusal of the same day replaces the earlier one: the next retry names only the last.
    _finish(root, _key(project, failures=1), _payload(classification=[
        {"item": f"{MARKER}-invented-item", "class": "current", "evidence": []},
    ]))
    assert review.review_tick(NOW + 2 + HOUR) == 0
    stored = _stored(root, project)
    assert json.loads(stored) == {"date": DAY, "code": COVERAGE, "detail": INVENTED_DETAIL}
    assert MARKER not in stored
    assert review.review_tick(NOW + 1 + 3 * HOUR) == 0
    _clock(monkeypatch, NOW + 2 + 3 * HOUR)
    assert review.review_tick(NOW + 2 + 3 * HOUR) == 1
    third = _body(root, _key(project, failures=2))
    assert _refusal_lines(third) == [LEAD_IN + refusal_sentences(COVERAGE, INVENTED_DETAIL)]
    assert MARKER not in third


def test_an_accepted_review_clears_the_refusal(root, monkeypatch):
    project = _news_project(root, monkeypatch)
    _finish(root, _key(project), _payload(_text(412)))
    assert review.review_tick(NOW + 1) == 0
    assert json.loads(_stored(root, project))["code"] == LENGTH

    _clock(monkeypatch, NOW + 1 + HOUR)
    assert review.review_tick(NOW + 1 + HOUR) == 1
    _finish(root, _key(project, failures=1), _payload())
    assert review.review_tick(NOW + 2 + HOUR) == 0
    assert _read(root / "projects.db", "SELECT open_task_id, failures, last_reviewed_date, last_refusal "
                 "FROM project_review_state WHERE project_id=?", project.id) == [(None, 0, DAY, None)]

    # The next day's task has no refusal line.
    tomorrow = NOW + 86400
    _clock(monkeypatch, tomorrow)
    _owner_comment(project)
    assert review.review_tick(tomorrow) == 1
    assert _refusal_lines(_body(root, _key(project, TOMORROW))) == []

    # The kernel's own review of a quiet day settles the day as an accepted review does.
    quiet = _quiet_project(root, name="Garden Plan", created_at=tomorrow - 86400)
    _write_state(root, quiet, last_refusal=json.dumps({"date": DAY, "code": LENGTH, "detail": LONG_DETAIL}))
    assert _stored(root, quiet) is not None
    _clock(monkeypatch, tomorrow + 3 * HOUR)
    assert review.review_tick(tomorrow + 3 * HOUR) == 0
    assert _read(root / "projects.db", "SELECT last_reviewed_date, last_refusal FROM project_review_state "
                 "WHERE project_id=?", quiet.id) == [(TOMORROW, None)]

    # A failure that has no refusal to tell, such as a gone summary board, keeps no reason either.
    _write_state(root, quiet, open_task_id="t_gone",
                 last_refusal=json.dumps({"date": TOMORROW, "code": LENGTH, "detail": LONG_DETAIL}))
    with projects_db.connect_closing(root / "projects.db") as registry:
        review._record_failure(registry, None, quiet.id, "t_gone", tomorrow + 4 * HOUR)
    assert _read(root / "projects.db", "SELECT failures, last_refusal FROM project_review_state WHERE project_id=?",
                 quiet.id) == [(1, None)]


def test_a_new_day_does_not_carry_yesterdays_refusal(root, monkeypatch):
    yesterday = NOW - 86400
    project = _news_project(root, monkeypatch, at=yesterday, created_at=NOW - 2 * 86400)
    _finish(root, _key(project, YESTERDAY), _payload(_text(412)))
    assert review.review_tick(yesterday + 1) == 0
    assert json.loads(_stored(root, project)) == {"date": YESTERDAY, "code": LENGTH, "detail": LONG_DETAIL}

    _clock(monkeypatch, NOW)
    assert review.review_tick(NOW) == 1
    assert _refusal_lines(_body(root, _key(project, DAY, failures=1))) == []


def test_a_stale_result_carries_no_refusal(root, monkeypatch):
    project = _quiet_project(root)
    _clock(monkeypatch, NOW)
    with kanban_db.connect_closing(kanban_db.kanban_db_path(board=project.board_slug), board=project.board_slug) as board:
        card = board.execute("SELECT id FROM tasks LIMIT 1").fetchone()[0]
        assert kanban_db.block_task(board, card, reason="Which room?", kind="needs_input")
    assert review.review_tick(NOW) == 1

    # The owner answers after the input was frozen: the result is stale, which counts no failure.
    _clock(monkeypatch, NOW + 10)
    with kanban_db.connect_closing(kanban_db.kanban_db_path(board=project.board_slug), board=project.board_slug) as board:
        assert kanban_db.unblock_task(board, card)
    claim = {"item": card, "class": "still actionable", "evidence": [f"decisions:wait:{card}"]}
    _finish(root, _key(project), _payload(classification=[claim]))
    assert review.review_tick(NOW + 11) == 1
    assert _read(root / "projects.db", "SELECT failures, last_reviewed_date, last_refusal FROM project_review_state "
                 "WHERE project_id=?", project.id) == [(0, YESTERDAY, None)]
    assert _refusal_lines(_body(root, _key(project, again=True))) == []


def _state_rows(conn, ids):
    return {pid: dict(conn.execute("SELECT * FROM project_review_state WHERE project_id = ?", (pid,)).fetchone())
            for pid in ids}


def test_an_older_store_gains_the_column_and_keeps_its_rows(tmp_path):
    path = tmp_path / "projects.db"
    full = {
        "event_cursor": 7, "delivery_cursor": 3, "release_cursor": 2, "conversation_cursor": "ab" * 32,
        "last_reviewed_date": YESTERDAY, "last_reviewed_at": 1794200000, "last_brief_date": YESTERDAY,
        "last_news_date": YESTERDAY, "open_task_id": "t_open", "open_review_date": DAY, "failures": 2, "updated_at": 5,
        "input_too_large_date": "2026-11-07", "stale_refused_key": f"project-review:x:{YESTERDAY}:1",
        "cap_wait_date": DAY, "cap_wait_at": 1794207600,
    }
    old = projects_db.connect(db_path=path)
    try:
        ids = [projects_db.create_project(old, name=name, folders=[folder])
               for name, folder in (("Meeting Plan", "/m"), ("Garden Plan", "/g"))]
        with kanban_db.write_txn(old):
            old.execute(f"INSERT INTO project_review_state (project_id, {', '.join(full)}) "
                        f"VALUES (?{', ?' * len(full)})", (ids[0], *full.values()))
            old.execute("INSERT INTO project_review_state (project_id, updated_at) VALUES (?, ?)", (ids[1], 9))
        # The store as it was before the column: the state table without it.
        if "last_refusal" in {row["name"] for row in old.execute("PRAGMA table_info(project_review_state)")}:
            old.execute("ALTER TABLE project_review_state DROP COLUMN last_refusal")
            old.commit()
        before = _state_rows(old, ids)
        assert all("last_refusal" not in row for row in before.values())
        assert before[ids[0]]["failures"] == 2 and before[ids[1]]["updated_at"] == 9
    finally:
        old.close()

    for _ in range(2):  # a second open of the upgraded store changes nothing
        projects_db._INITIALIZED_PATHS.clear()
        upgraded = projects_db.connect(db_path=path)
        try:
            columns = [(row["name"], row["type"], row["notnull"], row["dflt_value"])
                       for row in upgraded.execute("PRAGMA table_info(project_review_state)")
                       if row["name"] == "last_refusal"]
            assert columns == [("last_refusal", "TEXT", 0, None)]
            after = _state_rows(upgraded, ids)
            for pid in ids:
                assert {name: after[pid][name] for name in before[pid]} == before[pid]
                assert after[pid]["last_refusal"] is None
            assert {project.id for project in projects_db.list_projects(upgraded)} == set(ids)
        finally:
            upgraded.close()


def test_an_unreadable_refusal_gives_no_line_and_no_error(root, monkeypatch, caplog):
    good = {"date": DAY, "code": LENGTH, "detail": LONG_DETAIL}
    two_lines = {"fields": [{"field": f"summary.outcome\n{MARKER}", "rule": "length", "length": 412, "limit": 400}]}
    too_large = ('{"date": "%s", "code": "%s", "detail": {"fields": [{"field": "summary.outcome", "rule": "length", '
                 '"length": 1e999, "limit": 400}]}}' % (DAY, LENGTH))
    plain = {"date": DAY, "code": "review_result_keys_invalid", "detail": {}}
    readable = {"the last refusal": good, "a refusal without detail": plain}
    unreadable = {
        "not json": "{not json", "empty": "", "null": "null", "an array": json.dumps([DAY, LENGTH, LONG_DETAIL]),
        "a missing key": json.dumps({"date": DAY, "code": LENGTH}),
        "an extra key": json.dumps({**good, "note": MARKER}),
        "another key": json.dumps({"day": DAY, "code": LENGTH, "detail": LONG_DETAIL}),
        "yesterday": json.dumps({**good, "date": YESTERDAY}), "no date": json.dumps({**good, "date": None}),
        "a code that is a number": json.dumps({**good, "code": 412}),
        "a detail that is a list": json.dumps({**good, "detail": [MARKER]}),
        "a detail that is text": json.dumps({**good, "detail": MARKER}),
        "a code on two lines": json.dumps({**good, "code": f"{LENGTH}\n{MARKER}"}),
        "a code that is not ASCII": json.dumps({**good, "code": f"{LENGTH}-é"}),
        "a field on two lines": json.dumps({**good, "detail": two_lines}),
        "a number that is too large": too_large,
    }
    stored = {**{name: json.dumps(value) for name, value in readable.items()}, **unreadable}
    _clock(monkeypatch, NOW)
    projects = {name: _quiet_project(root, f"Plan {number}") for number, name in enumerate(stored)}
    (root / "config.yaml").write_text(yaml.safe_dump({
        "owner_review": {"enabled": True, "allowlist": [project.id for project in projects.values()]},
    }))
    for name, project in projects.items():
        _owner_comment(project)
        _write_state(root, project, last_refusal=stored[name])

    with caplog.at_level(logging.DEBUG):
        opened = sum(review.review_tick(NOW) for _ in range(len(stored)))
    assert opened == len(stored)
    assert not [record for record in caplog.records if record.levelno >= logging.WARNING]
    assert not [record for record in caplog.records
                if "refus" in record.getMessage().lower() or MARKER in record.getMessage()]

    for name, project in projects.items():
        body = _body(root, _key(project))
        assert MARKER not in body, name
        if name in readable:
            value = readable[name]
            assert _refusal_lines(body) == [LEAD_IN + refusal_sentences(value["code"], value["detail"])], name
        else:
            assert _refusal_lines(body) == [] and LEAD_IN not in body, name
        assert _stored(root, project) == stored[name], name
