"""Every limit is told to the reviewer, and every harvest refusal logs its reason without worker text."""

from __future__ import annotations

import logging
import time

import pytest
import yaml

from hermes_cli import kanban_db, owner_portfolio_review as portfolio, owner_project_review as review
from hermes_cli import owner_project_summary as summary, projects_db
from tests.hermes_cli.test_owner_project_review_result import NOW, _quiet_project, _suggestion, root  # noqa: F401

SUMMARIES = "project-summaries"
DAY = "2026-11-09"
MARKER = "Zebra"


def _summaries():
    return kanban_db.connect_closing(kanban_db.kanban_db_path(board=SUMMARIES), board=SUMMARIES)


def _project_board(project):
    return kanban_db.connect_closing(kanban_db.kanban_db_path(board=project.board_slug), board=project.board_slug)


def _comment_on_work(project):
    with _project_board(project) as board:
        task = board.execute("SELECT id FROM tasks WHERE task_kind='work'").fetchone()[0]
        kanban_db.add_comment(board, task, "default", "The room is chosen.")
    return task


def _review_task(project):
    with _summaries() as board:
        return board.execute(
            "SELECT id, body FROM tasks WHERE idempotency_key=?", (f"project-review:{project.id}:{DAY}:0",),
        ).fetchone()


def _complete(task_id, metadata):
    with _summaries() as board:
        assert kanban_db.claim_task(board, task_id) is not None
        assert kanban_db.complete_task(board, task_id, summary="Reviewed.", metadata=metadata)


def _payload(**summary_changes):
    value = {
        "summary": {"outcome": "The meeting has a room.", "next_action": "Book the chosen room.", "owner_decision": None},
        "brief": {"headline": "The room is chosen.", "what_changed": "Book the room now has a chosen room.",
                  "next_step": "Book the chosen room.", "needs_you": None},
        "classification": [], "proposed_closures": [],
    }
    value["summary"].update(summary_changes)
    return value


def _marked(length):
    return (MARKER * length)[:length]


def test_the_review_task_names_the_summary_limits(root, monkeypatch):
    project = _quiet_project(root)
    monkeypatch.setattr(time, "time", lambda: NOW)
    _comment_on_work(project)
    assert review.review_tick(NOW) == 1
    body = _review_task(project)["body"]
    limits, brief = summary._TEXT_LIMITS, review._BRIEF_LIMITS
    assert f"outcome has at most {limits['outcome']} characters" in body
    assert f"next_action at most {limits['next_action']}" in body
    assert (f"headline has at most {brief['headline']}, what_changed {brief['what_changed']}, "
            f"next_step {brief['next_step']}, needs_you {brief['needs_you']}") in body
    assert f"The reason has at most {review._CLOSURE_REASON_LIMIT} characters" in body


def test_a_long_summary_is_refused_with_every_breach_and_the_log_names_them(root, monkeypatch, caplog):
    project = _quiet_project(root)
    monkeypatch.setattr(time, "time", lambda: NOW)
    _comment_on_work(project)
    with projects_db.connect_closing(root / "projects.db") as registry:
        previous = projects_db.get_project_summary(registry, project.id)
    assert review.review_tick(NOW) == 1
    _complete(_review_task(project)["id"], {"project_review": _payload(outcome=_marked(412), next_action=_marked(457))})
    with caplog.at_level(logging.WARNING, logger="hermes_cli.owner_project_review"):
        assert review.review_tick(NOW + 1) == 0
    with projects_db.connect_closing(root / "projects.db") as registry:
        assert projects_db.get_project_summary(registry, project.id) == previous
        assert projects_db.get_project_review_brief(registry, project.id) is None
        failures = registry.execute(
            "SELECT failures FROM project_review_state WHERE project_id=?", (project.id,),
        ).fetchone()[0]
    assert failures == 1
    [line] = [text for text in caplog.text.splitlines() if "refused Project" in text]
    assert f"Project {project.id} task " in line
    assert "(review_text_length_invalid)" in line
    assert f"day={DAY} failures=1" in line
    assert "fields=summary.outcome:length:412/400,summary.next_action:length:457/240" in line
    assert MARKER not in caplog.text


def test_a_coverage_refusal_logs_the_missing_and_extra_items(root, monkeypatch, caplog):
    project = _quiet_project(root)
    monkeypatch.setattr(time, "time", lambda: NOW)
    work = _comment_on_work(project)
    with _project_board(project) as board:
        suggestion = _suggestion(board, project)
    assert review.review_tick(NOW) == 1
    invented = "Invented-marker-item"
    value = _payload()
    value["classification"] = [
        {"item": work, "class": "current", "evidence": [f"tasks:{work}"]},
        {"item": invented, "class": "current", "evidence": []},
    ]
    _complete(_review_task(project)["id"], {"project_review": value})
    with caplog.at_level(logging.WARNING, logger="hermes_cli.owner_project_review"):
        assert review.review_tick(NOW + 1) == 0
    [line] = [text for text in caplog.text.splitlines() if "refused Project" in text]
    assert "(review_class_coverage_invalid)" in line
    assert f"missing={suggestion}" in line
    assert f"extra={work}" in line
    assert "extra_other=1" in line
    assert invented not in caplog.text


def _portfolio_on(root, project):
    (root / "config.yaml").write_text(yaml.safe_dump({"owner_review": {
        "enabled": True, "allowlist": [project.id], "portfolio": True,
    }}))


def _finish_portfolio(result):
    with _summaries() as board:
        task_id = board.execute(
            "SELECT id FROM tasks WHERE idempotency_key=?", (f"project-review:portfolio:{DAY}:0",),
        ).fetchone()[0]
    _complete(task_id, {"portfolio_review": result})


def test_a_refused_portfolio_line_is_logged_with_its_code_and_position(root, monkeypatch, caplog):
    project = _quiet_project(root)
    _portfolio_on(root, project)
    assert review.review_tick(NOW) == 1
    long_line = "Meeting Plan: " + _marked(199)
    assert len(long_line) == 213
    _finish_portfolio({"conflicts": [], "priorities": ["Meeting Plan: Book the room.", long_line], "shared_blockers": []})
    with caplog.at_level(logging.WARNING, logger="hermes_cli.owner_project_review"):
        assert review.review_tick(NOW + 1) == 0
    [line] = [text for text in caplog.text.splitlines() if "portfolio_result_refused" in text]
    assert "portfolio_result_refused (portfolio_result_text_invalid)" in line
    assert f"day={DAY} task=" in line
    assert "lines=priorities[1]:length:213/200" in line
    assert MARKER not in caplog.text
    with projects_db.connect_closing(root / "projects.db") as registry:
        assert projects_db.get_portfolio_review_brief(registry, DAY) is None


def test_the_gateway_readers_follow_the_portfolio_limits(root, monkeypatch):
    from gateway.platforms.api_server_owner_projects import _portfolio_view

    project = _quiet_project(root)
    _portfolio_on(root, project)
    monkeypatch.setattr(portfolio, "_ITEM_LIMIT", 250, raising=False)
    assert review.review_tick(NOW) == 1
    line = "Meeting Plan: " + _marked(216)
    assert len(line) == 230
    _finish_portfolio({"conflicts": [], "priorities": [line], "shared_blockers": []})
    assert review.review_tick(NOW + 1) == 0
    with projects_db.connect_closing(root / "projects.db") as registry:
        assert projects_db.get_portfolio_review_brief(registry, DAY)["priorities"] == [line]
    view = _portfolio_view([{"project_id": project.id, "slug": project.slug, "name": project.name}])
    assert view is not None
    assert view["priorities"] == [line]
