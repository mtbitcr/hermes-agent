"""A broken review or portfolio result goes back to its reviewer in the same run.

kanban_complete of a review or portfolio task runs the harvest's own result checks. A refusal returns the stable
code, its detail and plain sentences, writes nothing and leaves the task running, so the reviewer corrects the result
and calls kanban_complete again. The harvest runs the same checks and stays the authority.

Every worker call runs under the exact environment the dispatcher injects: the real spawn
(``kanban_db._default_spawn``, with only ``Popen`` captured) pins ``HERMES_KANBAN_DB`` and ``HERMES_KANBAN_BOARD`` to
the summary board, sets ``HERMES_KANBAN_TASK`` and points ``HERMES_HOME`` at a profile home under the root. The tool
runs through ``model_tools.handle_function_call``. The stores are read back with plain sqlite3, without the code under
test.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import sqlite3
import subprocess
import time
from pathlib import Path

import yaml

from hermes_cli import kanban_db, owner_project_review as review, projects_db
from tests.hermes_cli.test_owner_project_review_result import NOW, _quiet_project, _suggestion, root  # noqa: F401

SUMMARIES = "project-summaries"
DAY = "2026-11-09"
MARKER = "Zebra"
LOGGER = "hermes_cli.owner_project_review"
STILL_RUNNING = "Your task is still running. Correct the result and call kanban_complete again."


def _text(length, unit="The room is booked for the meeting. "):
    """Plain text of exactly ``length`` characters that stripping does not shorten."""
    return (unit * (length // len(unit) + 1))[:length - 1] + "."


def _read(path, sql, *args):
    """The independent read: plain read-only SQLite on the file, without the code under test."""
    conn = sqlite3.connect(f"{Path(path).resolve().as_uri()}?mode=ro", uri=True)
    try:
        return conn.execute(sql, args).fetchall()
    finally:
        conn.close()


def _summary_db(root):
    return root / "kanban" / "boards" / SUMMARIES / "kanban.db"


def _task_with_key(root, key):
    ((task_id,),) = _read(_summary_db(root), "SELECT id FROM tasks WHERE idempotency_key=?", key)
    return task_id


def _news_project(root, monkeypatch, *, suggestion=False):
    """A Project with news since its last review, and the review task the tick opens for it."""
    project = _quiet_project(root)
    monkeypatch.setattr(time, "time", lambda: NOW)
    with kanban_db.connect_closing(kanban_db.kanban_db_path(board=project.board_slug), board=project.board_slug) as board:
        work = board.execute("SELECT id FROM tasks WHERE task_kind='work'").fetchone()[0]
        kanban_db.add_comment(board, work, "default", "The room is chosen.")
        suggested = _suggestion(board, project) if suggestion else None
    assert review.review_tick(NOW) == 1
    return project, _task_with_key(root, f"project-review:{project.id}:{DAY}:0"), suggested


def _payload(outcome="The meeting has a room.", classification=(), next_action="Book the chosen room."):
    return {"project_review": {
        "summary": {"outcome": outcome, "next_action": next_action, "owner_decision": None},
        "brief": {"headline": "The room is chosen.", "what_changed": "Book the room now has a chosen room.",
                  "next_step": "Book the chosen room.", "needs_you": None},
        "classification": list(classification), "proposed_closures": [],
    }}


def _register_kanban_tools():
    import model_tools  # noqa: F401
    import tools.kanban_tools  # noqa: F401


def _complete(arguments):
    """kanban_complete through the real entry point; the raw JSON text of its answer."""
    import model_tools

    return model_tools.handle_function_call("kanban_complete", arguments)


@contextlib.contextmanager
def _worker(root, monkeypatch, task_id):
    """Claim the task as the dispatcher does, spawn its worker with only Popen captured, and become that worker."""
    _register_kanban_tools()
    with kanban_db.connect_closing(kanban_db.kanban_db_path(board=SUMMARIES), board=SUMMARIES) as board:
        task = kanban_db.claim_task(board, task_id)
    assert task is not None and task.status == "running" and task.current_run_id is not None
    (root / "profiles" / task.assignee).mkdir(parents=True, exist_ok=True)
    with monkeypatch.context() as worker:
        captured = {}

        class _Proc:
            pid = 4242

        def _popen(cmd, *args, **kwargs):
            kwargs["stdout"].close()  # the worker log the dispatcher opened for the child
            captured["env"] = dict(kwargs["env"])
            return _Proc()

        worker.setattr(kanban_db, "_resolve_hermes_argv", lambda: ["hermes"])
        worker.setattr(subprocess, "Popen", _popen)
        workspace = root.parent / f"workspace-{task.id}"
        workspace.mkdir(exist_ok=True)
        assert kanban_db._default_spawn(task, str(workspace), board=SUMMARIES) == 4242
        env = captured["env"]
        assert env["HERMES_KANBAN_TASK"] == task.id and env["HERMES_KANBAN_BOARD"] == SUMMARIES
        assert Path(env["HERMES_KANBAN_DB"]).resolve() == _summary_db(root).resolve()
        assert Path(env["HERMES_HOME"]).resolve() == (root / "profiles" / task.assignee).resolve()
        review_task = task.created_by == kanban_db.REVIEW_TASK_CREATOR
        assert env.get("HERMES_KANBAN_ALLOWED_TOOLS") == (",".join(kanban_db.REVIEW_TASK_TOOLS) if review_task else None)
        for key in set(os.environ) - set(env):
            worker.delenv(key)
        for key, value in env.items():
            if os.environ.get(key) != value:
                worker.setenv(key, value)
        yield task


def _assert_running(root, task):
    """The refused call wrote nothing: the task still runs on its open run, which holds no result."""
    assert _read(_summary_db(root), "SELECT status, current_run_id FROM tasks WHERE id=?", task.id) == [
        ("running", task.current_run_id),
    ]
    assert _read(_summary_db(root), "SELECT id, status, ended_at, outcome, metadata FROM task_runs WHERE task_id=?",
                 task.id) == [(task.current_run_id, "running", None, None, None)]


def _review_state(root, project):
    return _read(root / "projects.db", "SELECT open_task_id, last_reviewed_date, failures FROM project_review_state "
                 "WHERE project_id=?", project.id)


def test_a_long_summary_is_handed_back_and_the_corrected_result_is_accepted(root, monkeypatch):
    project, task_id, _ = _news_project(root, monkeypatch)
    long, short = _text(412), _text(380)
    with _worker(root, monkeypatch, task_id) as task:
        refused = json.loads(_complete({"summary": "Reviewed.", "metadata": _payload(long)}))
        assert refused.get("refusal") == {"code": "review_text_length_invalid", "detail": {"fields": [
            {"field": "summary.outcome", "rule": "length", "length": 412, "limit": 400},
        ]}}, refused
        assert refused["still_running"] is True and refused.get("ok") is not True
        assert "summary.outcome has 412 characters; at most 400." in refused["error"]
        assert refused["error"].endswith(STILL_RUNNING)
        _assert_running(root, task)
        accepted = json.loads(_complete({"summary": "Reviewed.", "metadata": _payload(short)}))
        assert accepted.get("ok") is True, accepted

    # The store-path proof, read against an independent count: the corrected completion is the task's only done run.
    runs = _read(_summary_db(root), "SELECT id, metadata FROM task_runs WHERE task_id=? AND status='done'", task.id)
    assert [run_id for run_id, _ in runs] == [task.current_run_id]
    assert json.loads(runs[0][1])["project_review"]["summary"]["outcome"] == short
    assert _read(_summary_db(root), "SELECT COUNT(*) FROM task_runs WHERE task_id=?", task.id) == [(1,)]

    assert review.review_tick(NOW + 1) == 0
    assert _review_state(root, project) == [(None, DAY, 0)]
    assert _read(root / "projects.db", "SELECT review_date, headline FROM project_review_briefs WHERE project_id=?",
                 project.id) == [(DAY, "The room is chosen.")]
    assert _read(root / "projects.db", "SELECT outcome FROM project_summaries WHERE project_id=?", project.id) == [
        (short,),
    ]


def test_a_coverage_miss_is_handed_back_with_the_expected_items(root, monkeypatch):
    project, task_id, suggestion = _news_project(root, monkeypatch, suggestion=True)
    with _worker(root, monkeypatch, task_id) as task:
        refused = json.loads(_complete({"summary": "Reviewed.", "metadata": _payload()}))
        assert refused.get("refusal") == {"code": "review_class_coverage_invalid", "detail": {
            "missing": [suggestion], "extra": [], "extra_other": 0,
        }}, refused
        assert refused["still_running"] is True
        assert f"Classify each of these items exactly once: {suggestion}." in refused["error"]
        assert refused["error"].endswith(STILL_RUNNING)
        _assert_running(root, task)
        claim = {"item": suggestion, "class": "still actionable", "evidence": [f"suggestions:{suggestion}"]}
        accepted = json.loads(_complete({"summary": "Reviewed.", "metadata": _payload(classification=[claim])}))
        assert accepted.get("ok") is True, accepted

    assert review.review_tick(NOW + 1) == 0
    assert _review_state(root, project) == [(None, DAY, 0)]
    ((classification,),) = _read(root / "projects.db", "SELECT classification FROM project_review_briefs "
                                 "WHERE project_id=?", project.id)
    assert [(item["item"], item["class"]) for item in json.loads(classification)] == [(suggestion, "still actionable")]


def test_a_refused_portfolio_line_is_handed_back_and_the_morning_advice_is_stored(root, monkeypatch, caplog):
    project = _quiet_project(root)
    (root / "config.yaml").write_text(yaml.safe_dump({"owner_review": {
        "enabled": True, "allowlist": [project.id], "portfolio": True,
    }}))
    assert review.review_tick(NOW) == 1
    task_id = _task_with_key(root, f"project-review:portfolio:{DAY}:0")
    long_line = "Meeting Plan: " + _text(199, f"{MARKER} notes on the room. ")
    good = ["Meeting Plan: Book the room.", "Meeting Plan: Send the invitations."]
    assert len(long_line) == 213
    with _worker(root, monkeypatch, task_id) as task:
        with caplog.at_level(logging.WARNING, logger=LOGGER):
            raw = _complete({"summary": "Advised.", "metadata": {"portfolio_review": {
                "conflicts": [], "priorities": [good[0], long_line], "shared_blockers": [],
            }}})
        refused = json.loads(raw)
        assert refused.get("refusal") == {"code": "portfolio_result_text_invalid", "detail": {"lines": [
            {"line": "priorities[1]", "rule": "length", "length": 213, "limit": 200},
        ]}}, refused
        assert refused["still_running"] is True
        assert "priorities line 2 has 213 characters; at most 200." in refused["error"]
        assert refused["error"].endswith(STILL_RUNNING)
        assert MARKER not in raw and MARKER not in caplog.text
        [line] = [text for text in caplog.text.splitlines() if "portfolio_handover_refused" in text]
        assert (f"portfolio_handover_refused: task {task.id} run {task.current_run_id} (portfolio_result_text_invalid) "
                "lines=priorities[1]:length:213/200") in line
        _assert_running(root, task)
        accepted = json.loads(_complete({"summary": "Advised.", "metadata": {"portfolio_review": {
            "conflicts": [], "priorities": good, "shared_blockers": [],
        }}}))
        assert accepted.get("ok") is True, accepted

    assert review.review_tick(NOW + 1) == 0
    with projects_db.connect_closing(root / "projects.db") as registry:
        advice = projects_db.get_portfolio_review_brief(registry, DAY)
    assert advice is not None and advice["priorities"] == good
    assert _read(_summary_db(root), "SELECT COUNT(*) FROM task_runs WHERE task_id=? AND status='done'", task.id) == [(1,)]


def test_a_stale_decision_is_not_judged_at_the_handover(root, monkeypatch, caplog):
    project = _quiet_project(root)
    monkeypatch.setattr(time, "time", lambda: NOW)
    with kanban_db.connect_closing(kanban_db.kanban_db_path(board=project.board_slug), board=project.board_slug) as board:
        card = board.execute("SELECT id FROM tasks LIMIT 1").fetchone()[0]
        assert kanban_db.block_task(board, card, reason="Which room?", kind="needs_input")
    assert review.review_tick(NOW) == 1
    task_id = _task_with_key(root, f"project-review:{project.id}:{DAY}:0")
    # The owner answers after the input was frozen. The worker cannot see it; only the harvest reads the live board.
    monkeypatch.setattr(time, "time", lambda: NOW + 10)
    with kanban_db.connect_closing(kanban_db.kanban_db_path(board=project.board_slug), board=project.board_slug) as board:
        assert kanban_db.unblock_task(board, card)
    claim = {"item": card, "class": "still actionable", "evidence": [f"decisions:wait:{card}"]}
    with _worker(root, monkeypatch, task_id):
        done = json.loads(_complete({"summary": "Reviewed.", "metadata": _payload(classification=[claim])}))
    assert done.get("ok") is True, done

    # Owner rule of 2026-11-11, point 6: the stale result counts no failure and opens the one further review.
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        assert review.review_tick(NOW + 11) == 1
    assert "review_decision_state_changed" in caplog.text
    [(open_task, reviewed, failures)] = _review_state(root, project)
    assert (reviewed, failures) == ("2026-11-08", 0)
    assert open_task not in {None, task_id}
    with projects_db.connect_closing(root / "projects.db") as registry:
        assert projects_db.get_project_review_brief(registry, project.id) is None


def test_a_broken_frozen_input_fails_open(root, monkeypatch, caplog):
    project, task_id, _ = _news_project(root, monkeypatch)
    with kanban_db.connect_closing(kanban_db.kanban_db_path(board=SUMMARIES), board=SUMMARIES) as board:
        with kanban_db.write_txn(board):
            board.execute("UPDATE tasks SET body=? WHERE id=?", ("Review the project.", task_id))
    # Even a result that breaks a text limit completes: the worker cannot fix its own input.
    with _worker(root, monkeypatch, task_id) as task:
        done = json.loads(_complete({"summary": "Reviewed.", "metadata": _payload(_text(412))}))
    assert done.get("ok") is True, done
    assert _read(_summary_db(root), "SELECT status FROM tasks WHERE id=?", task.id) == [("done",)]

    with caplog.at_level(logging.WARNING, logger=LOGGER):
        assert review.review_tick(NOW + 1) == 0
    [line] = [text for text in caplog.text.splitlines() if "refused Project" in text]
    assert "(review_input_missing)" in line and f"day={DAY} failures=1" in line
    assert _review_state(root, project) == [(None, "2026-11-08", 1)]


def test_a_result_that_skips_the_handover_is_still_refused_at_the_harvest(root, monkeypatch, caplog):
    project, task_id, _ = _news_project(root, monkeypatch)
    with kanban_db.connect_closing(kanban_db.kanban_db_path(board=SUMMARIES), board=SUMMARIES) as board:
        assert kanban_db.claim_task(board, task_id) is not None
        assert kanban_db.complete_task(board, task_id, summary="Reviewed.", metadata=_payload(_text(412)))

    with caplog.at_level(logging.WARNING, logger=LOGGER):
        assert review.review_tick(NOW + 1) == 0
    [line] = [text for text in caplog.text.splitlines() if "refused Project" in text]
    assert "(review_text_length_invalid)" in line and f"day={DAY} failures=1" in line
    assert "fields=summary.outcome:length:412/400" in line
    assert _review_state(root, project) == [(None, "2026-11-08", 1)]
    with projects_db.connect_closing(root / "projects.db") as registry:
        assert projects_db.get_project_review_brief(registry, project.id) is None


def test_the_handover_message_and_log_hold_no_worker_text(root, monkeypatch, caplog):
    project, task_id, _ = _news_project(root, monkeypatch)
    marked = _text(412, f"{MARKER} notes on the room. ")
    invented = {"item": f"{MARKER}-invented-item", "class": "current", "evidence": []}
    with _worker(root, monkeypatch, task_id) as task:
        with caplog.at_level(logging.DEBUG, logger=LOGGER):
            texts = _complete({"summary": f"{MARKER} review.", "metadata": _payload(marked, next_action=f"{MARKER}.")})
            coverage = _complete({"summary": f"{MARKER} review.", "metadata": _payload(classification=[invented])})
        _assert_running(root, task)
    for raw in (texts, coverage):
        assert MARKER not in raw and project.name not in raw
        assert json.loads(raw)["refusal"]["code"] and json.loads(raw)["still_running"] is True
    assert json.loads(texts)["refusal"]["detail"] == {"fields": [
        {"field": "summary.outcome", "rule": "length", "length": 412, "limit": 400},
    ]}
    assert json.loads(coverage)["refusal"] == {"code": "review_class_coverage_invalid", "detail": {
        "missing": [], "extra": [], "extra_other": 1,
    }}
    assert MARKER not in caplog.text and project.name not in caplog.text
    lines = [text for text in caplog.text.splitlines() if "review_handover_refused" in text]
    head = f"review_handover_refused: Project {project.id} task {task.id} run {task.current_run_id}"
    assert len(lines) == 2
    assert f"{head} (review_text_length_invalid) fields=summary.outcome:length:412/400" in lines[0]
    assert f"{head} (review_class_coverage_invalid) missing=none extra=none extra_other=1" in lines[1]


def test_an_ordinary_task_completes_unchanged(root, monkeypatch):
    monkeypatch.setattr(time, "time", lambda: NOW)
    kanban_db.create_board(SUMMARIES)
    with kanban_db.connect_closing(kanban_db.kanban_db_path(board=SUMMARIES), board=SUMMARIES) as board:
        task_id = kanban_db.create_task(
            board, title="Write the notes", body="Write the notes.", assignee="raphael-planner", created_by="default",
            idempotency_key=f"project-review:prj_alpha:{DAY}:0", board=SUMMARIES,
        )
    bad = _payload(_text(412))
    with _worker(root, monkeypatch, task_id) as task:
        done = json.loads(_complete({"summary": "Notes written.", "metadata": bad}))
    assert done.get("ok") is True, done
    assert _read(_summary_db(root), "SELECT status FROM tasks WHERE id=?", task.id) == [("done",)]
    [(metadata,)] = _read(_summary_db(root), "SELECT metadata FROM task_runs WHERE task_id=? AND status='done'", task.id)
    assert json.loads(metadata)["project_review"] == bad["project_review"]


def test_nothing_changes_while_the_daily_review_switch_is_off(root, monkeypatch, caplog):
    project, task_id, _ = _news_project(root, monkeypatch)
    # The owner turns the daily review off while its task still runs: the completion stays as it was before the card.
    (root / "config.yaml").write_text(yaml.safe_dump({"owner_review": {"enabled": False, "allowlist": [project.id]}}))
    bad = _payload(_text(412))
    with _worker(root, monkeypatch, task_id) as task:
        with caplog.at_level(logging.DEBUG, logger=LOGGER):
            done = json.loads(_complete({"summary": "Reviewed.", "metadata": bad}))
    assert done.get("ok") is True, done
    assert "refusal" not in done and "handover" not in caplog.text
    assert _read(_summary_db(root), "SELECT status FROM tasks WHERE id=?", task.id) == [("done",)]
    [(metadata,)] = _read(_summary_db(root), "SELECT metadata FROM task_runs WHERE task_id=? AND status='done'", task.id)
    assert json.loads(metadata)["project_review"] == bad["project_review"]
    assert review.review_tick(NOW + 1) == 0
    assert _review_state(root, project) == [(task_id, "2026-11-08", 0)]


def _stored_rows(root, task_id, stored, *args):
    """The independent read of the task's row and runs on the summary board and of one stored state in projects.db."""
    return (
        _read(_summary_db(root), "SELECT * FROM tasks WHERE id=?", task_id),
        _read(_summary_db(root), "SELECT * FROM task_runs WHERE task_id=? ORDER BY id", task_id),
        _read(root / "projects.db", stored, *args),
    )


def _assert_a_second_completion_is_unchanged(root, monkeypatch, caplog, task_id, valid, broken, stored, *args):
    """Complete the task with ``valid``, then call kanban_complete again with ``broken`` in the same worker.

    The run has ended, so the second call gets the answer of the completion path as it was before the card: the answer
    the same second call gives with ``valid``, which no refusal of the hand-over check can reach. It asks for no
    correction, the hand-over check logs nothing for it, and the task's row, its runs and the stored state stay as
    they were.
    """
    with _worker(root, monkeypatch, task_id) as task:
        done = json.loads(_complete(valid))
        assert done.get("ok") is True, done
        assert _read(_summary_db(root), "SELECT status, current_run_id FROM tasks WHERE id=?", task.id) == [
            ("done", None),
        ]
        before = _stored_rows(root, task.id, stored, *args)
        caplog.clear()
        with caplog.at_level(logging.DEBUG, logger=LOGGER):
            raw = _complete(broken)
        log = caplog.text
        after = _stored_rows(root, task.id, stored, *args)
        unchanged = _complete(valid)
    assert raw == unchanged
    answer = json.loads(raw)
    assert "still_running" not in answer and "refusal" not in answer, answer
    assert "still running" not in raw.lower() and "correct" not in raw.lower(), answer
    handover = ("review_handover_refused", "portfolio_handover_refused", "review_handover_check_failed")
    assert not [line for line in log.splitlines() if any(name in line for name in handover)], log
    assert after == before
    # Read against an independent count: the first completion is still the task's one run.
    assert _read(_summary_db(root), "SELECT id, status FROM task_runs WHERE task_id=?", task.id) == [
        (task.current_run_id, "done"),
    ]


def _advice(line):
    return {"summary": "Advised.", "metadata": {"portfolio_review": {
        "conflicts": [], "priorities": [line], "shared_blockers": [],
    }}}


def test_a_second_completion_of_a_finished_review_task_gets_the_unchanged_answer(root, monkeypatch, caplog):
    project, task_id, _ = _news_project(root, monkeypatch)
    _assert_a_second_completion_is_unchanged(
        root, monkeypatch, caplog, task_id,
        {"summary": "Reviewed.", "metadata": _payload()},
        {"summary": "Reviewed.", "metadata": _payload(_text(412))},
        "SELECT * FROM project_review_state WHERE project_id=?", project.id,
    )
    # No harvest ran: the registry still holds the review task open, with no failure counted.
    assert _review_state(root, project) == [(task_id, "2026-11-08", 0)]


def test_a_second_completion_of_a_finished_portfolio_task_gets_the_unchanged_answer(root, monkeypatch, caplog):
    project = _quiet_project(root)
    (root / "config.yaml").write_text(yaml.safe_dump({"owner_review": {
        "enabled": True, "allowlist": [project.id], "portfolio": True,
    }}))
    assert review.review_tick(NOW) == 1
    task_id = _task_with_key(root, f"project-review:portfolio:{DAY}:0")
    long_line = f"{project.name}: " + _text(201)
    assert len(long_line) == 215
    _assert_a_second_completion_is_unchanged(
        root, monkeypatch, caplog, task_id, _advice(f"{project.name}: Book the room."), _advice(long_line),
        "SELECT * FROM portfolio_review_briefs",
    )
    # No harvest ran: no advice is stored.
    assert _read(root / "projects.db", "SELECT COUNT(*) FROM portfolio_review_briefs") == [(0,)]
