"""Read approved plan history through the public review-input boundary."""

import contextlib
import json
import sqlite3
import time

import pytest

from hermes_cli import owner_project_review_record as record, owner_workspace, projects_db
from hermes_cli import owner_project_summary_events as events
from tests.hermes_cli.test_owner_project_review_record import _written_at, root  # noqa: F401
from tests.hermes_cli.test_owner_workspace import (
    _bootstrap_board,
    _commit_project_plan,
    _project_plan_args,
    _with_approver,
    ctx,  # noqa: F401
)


def _approved_plan(ctx, setup, *, key="review-plan", **overrides):
    changes = [{
        "action": "add", "reason": "Prepare the room for the visit.",
        "title": "Book the room", "body": "Book a room before the guests arrive.",
        "assignee": "default", "execution_tier": "routine", "risk_tier": 0,
        "existing_parents": [], "new_parents": [],
    }]
    args = _project_plan_args(
        setup, changes, idempotency_key=key, trigger="scheduled_review",
        summary="Book the room before guests arrive.",
        current_milestone="Prepare the first visit.",
    )
    args.update(overrides)
    approver = _with_approver(ctx.session)
    try:
        return _commit_project_plan(ctx, **args)
    finally:
        approver.join()


def _logical_state(path):
    with contextlib.closing(sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)) as conn:
        return tuple(conn.iterdump())


def test_the_review_reads_an_approved_native_plan_without_changing_either_store(root, ctx):
    setup = _bootstrap_board(ctx)
    committed = _approved_plan(ctx, setup)
    with projects_db.connect_closing(root / "projects.db") as registry:
        project = projects_db.get_project(registry, setup["project_id"])
    board_path = root / "kanban" / "boards" / project.board_slug / "kanban.db"
    before = (_logical_state(root / "projects.db"), _logical_state(board_path))

    first = record.build_review_input(project, events.Cursors(0, 0, 0), None)
    second = record.build_review_input(project, events.Cursors(0, 0, 0), None)

    [plan] = first["sources"]["proposals"]
    assert plan["state"] == "applied"
    assert plan["summary"] == (
        "Adapt the current plan\n\nBook the room before guests arrive.\n\n"
        "Apply only the approved changes and preserve history.\n\n"
        "Owner-visible result\nThe owner sees the revised work on the same board."
    )
    assert plan["current_milestone"] == "Prepare the first visit."
    assert plan["trigger"] == "scheduled_review"
    assert plan["created_task_ids"] == committed["created_task_ids"]
    assert plan["affected_task_ids"] == committed["affected_task_ids"]
    assert not any(cut["source"] == "proposals" for cut in first["cuts"])
    assert "<<<BEGIN UNTRUSTED DATA proposals>>>" in first["text"]
    assert first == second
    assert before == (_logical_state(root / "projects.db"), _logical_state(board_path))


def test_the_review_keeps_the_twenty_latest_plans_at_the_review_time_boundary(root, ctx):
    setup = _bootstrap_board(ctx)
    instant = int(time.time()) - 60
    with _written_at(instant - 1):
        _approved_plan(ctx, setup, key="before-the-window")
    with _written_at(instant):
        # Reverse the keys so event order, not lexical identity order, selects the latest plans.
        committed = [_approved_plan(ctx, setup, key=f"same-second-{index:02}")
                     for index in range(21, 0, -1)]
    with projects_db.connect_closing(root / "projects.db") as registry:
        project = projects_db.get_project(registry, setup["project_id"])

    review = record.build_review_input(project, events.Cursors(0, 0, 0), instant)

    assert [plan["created_task_ids"] for plan in review["sources"]["proposals"]] == [
        plan["created_task_ids"] for plan in reversed(committed[1:])
    ]
    assert [cut for cut in review["cuts"] if cut["source"] == "proposals"] == [
        {"source": "proposals", "shown": 20, "total": 21, "reason": "limit"},
    ]
    assert "Left out: 1 of 21 plan proposals." in review["text"]


def test_a_large_valid_unicode_plan_has_a_visible_cut_instead_of_stopping_the_review(root, ctx):
    setup = _bootstrap_board(ctx)
    _approved_plan(ctx, setup, specification="\U0001f600" * 20_000)
    with projects_db.connect_closing(root / "projects.db") as registry:
        project = projects_db.get_project(registry, setup["project_id"])

    review = record.build_review_input(project, events.Cursors(0, 0, 0), None)

    assert review["sources"]["proposals"] == []
    assert [cut for cut in review["cuts"] if cut["source"] == "proposals"] == [
        {"source": "proposals", "shown": 0, "total": 1, "reason": "total_cap"},
    ]
    assert "Left out: 1 of 1 plan proposals." in review["text"]
    assert review["size"] <= 128 * 1024


@pytest.mark.parametrize("corruption", ["invalid_receipt", "foreign_project", "missing_event", "duplicate_event"])
def test_an_unproven_plan_stops_the_review_without_changing_either_store(root, ctx, corruption):
    setup = _bootstrap_board(ctx)
    _approved_plan(ctx, setup)
    registry_path = root / "projects.db"
    with projects_db.connect_closing(registry_path) as registry:
        project = projects_db.get_project(registry, setup["project_id"])
    board_path = root / "kanban" / "boards" / project.board_slug / "kanban.db"
    if corruption in ("invalid_receipt", "foreign_project"):
        with contextlib.closing(sqlite3.connect(registry_path)) as conn:
            [raw] = conn.execute(
                "SELECT result_json FROM owner_workspace_receipts "
                "WHERE operation='owner_project_plan_commit' AND project_id=?", (project.id,),
            ).fetchone()
            if corruption == "invalid_receipt":
                changed = "{broken receipt"
            else:
                receipt = json.loads(raw)
                receipt["project_id"] = "a-different-project"
                changed = json.dumps(receipt)
            conn.execute(
                "UPDATE owner_workspace_receipts SET result_json=? "
                "WHERE operation='owner_project_plan_commit' AND project_id=?", (changed, project.id),
            )
            conn.commit()
    else:
        with contextlib.closing(sqlite3.connect(board_path)) as conn:
            if corruption == "missing_event":
                conn.execute("DELETE FROM task_events WHERE kind='owner_project_plan_applied'")
            else:
                conn.execute(
                    "INSERT INTO task_events(task_id,kind,payload,created_at) "
                    "SELECT task_id,kind,payload,created_at FROM task_events "
                    "WHERE kind='owner_project_plan_applied'",
                )
            conn.commit()
    before = (_logical_state(registry_path), _logical_state(board_path))

    with pytest.raises(ValueError, match="^review_plan_receipt_unavailable$"):
        record.build_review_input(project, events.Cursors(0, 0, 0), None)

    assert before == (_logical_state(registry_path), _logical_state(board_path))


def test_the_review_keeps_its_project_plan_when_a_worker_is_pinned_to_another_board(root, ctx, monkeypatch):
    setup = _bootstrap_board(ctx)
    committed = _approved_plan(ctx, setup)
    approver = _with_approver(ctx.session)
    try:
        other = owner_workspace.bootstrap(ctx, idempotency_key="other-project", name="Other Project")
    finally:
        approver.join()
    _approved_plan(ctx, other, key="other-plan", summary="A different project owns this plan.")
    with projects_db.connect_closing(root / "projects.db") as registry:
        project = projects_db.get_project(registry, setup["project_id"])
    profile_home = root / "profiles" / "raphael-planner"
    profile_home.mkdir(parents=True)
    expected = record.build_review_input(project, events.Cursors(0, 0, 0), None)
    monkeypatch.setenv("HERMES_HOME", str(profile_home))
    monkeypatch.setenv("HERMES_PROFILE", "raphael-planner")
    monkeypatch.setenv("HERMES_KANBAN_BOARD", other["board"])
    monkeypatch.setenv("HERMES_KANBAN_DB", str(root / "kanban" / "boards" / other["board"] / "kanban.db"))

    review = record.build_review_input(project, events.Cursors(0, 0, 0), None)

    assert review == expected
    assert [plan["created_task_ids"] for plan in review["sources"]["proposals"]] == [committed["created_task_ids"]]
    assert "A different project owns this plan." not in review["text"]


def test_a_plan_commit_on_a_second_connection_during_the_review_commit_does_not_deadlock(root, ctx, monkeypatch):
    """Owner rule of 2026-11-11, point 8: both commits take the registry first, then the board; both finish."""
    import threading
    from datetime import datetime, timezone

    import yaml

    from hermes_cli import kanban_db, owner_project_review as review

    now = int(datetime(2026, 11, 10, 7, tzinfo=timezone.utc).timestamp())
    with _written_at(now - 86400):
        setup = _bootstrap_board(ctx)
        _approved_plan(ctx, setup)
    with projects_db.connect_closing(root / "projects.db") as registry:
        project = projects_db.get_project(registry, setup["project_id"])
    (root / "config.yaml").write_text(yaml.safe_dump({"owner_review": {"enabled": True, "allowlist": [project.id]}}))
    monkeypatch.setattr(time, "time", lambda: now)
    assert review.review_tick(now) == 1
    with projects_db.connect_closing(root / "projects.db") as registry:
        task = registry.execute(
            "SELECT open_task_id FROM project_review_state WHERE project_id=?", (project.id,),
        ).fetchone()[0]
    payload = {
        "summary": {"outcome": "The room work is planned.", "next_action": "Book the room.", "owner_decision": None},
        "brief": {"headline": "The room work is planned.", "what_changed": "Book the room was added.",
                  "next_step": "Book the room.", "needs_you": None},
        "classification": [], "proposed_closures": [],
    }
    with kanban_db.connect_closing(kanban_db.kanban_db_path(board="project-summaries"), board="project-summaries") as board:
        assert kanban_db.claim_task(board, task) is not None
        assert kanban_db.complete_task(board, task, summary="Reviewed the plan.", metadata={"project_review": payload})

    lock, threads, commits, errors = review._lock_board, [], [], []

    def second_plan_commit():
        try:
            commits.append(_approved_plan(ctx, setup, key="concurrent-plan"))
        except Exception as exc:  # the assertion below names it
            errors.append(exc)

    def contended(*args, **kwargs):
        # The review holds the registry: the plan commit starts now and must wait for it, not hold the board against it.
        if not threads:
            threads.append(threading.Thread(target=second_plan_commit))
            threads[0].start()
            threads[0].join(timeout=0.5)
        return lock(*args, **kwargs)

    monkeypatch.setattr(review, "_lock_board", contended)
    started = time.monotonic()
    review.review_tick(now + 1)
    threads[0].join(timeout=60)
    assert not threads[0].is_alive() and not errors, errors
    assert time.monotonic() - started < 60
    assert commits and commits[0]["created_task_ids"]
    with projects_db.connect_closing(root / "projects.db") as registry:
        state = registry.execute(
            "SELECT open_task_id,last_reviewed_date,failures FROM project_review_state WHERE project_id=?", (project.id,),
        ).fetchone()
        assert tuple(state) == (None, "2026-11-10", 0)
        assert projects_db.get_project_review_brief(registry, project.id)["headline"] == "The room work is planned."
