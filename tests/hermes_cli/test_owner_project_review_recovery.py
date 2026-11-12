"""A review task published before its registry claim is counted once, however it ended and whenever a tick finds it."""

from __future__ import annotations

import time

import pytest

from hermes_cli import kanban_db, owner_project_review as review, projects_db
from tests.hermes_cli.test_owner_project_review_result import (  # noqa: F401
    NOW,
    _complete_review,
    _round3_drive,
    _round3_mutate,
    _round3_setup_result,
    root,
    round3_root,
)

SUMMARIES = "project-summaries"
DAY = "2026-11-09"
NEXT_DAY = "2026-11-10"
HOUR = 3600


def _summaries():
    return kanban_db.connect_closing(kanban_db.kanban_db_path(board=SUMMARIES), board=SUMMARIES)


def _registry(root, project, day=DAY):
    """The Project's review state and the day's budget: ``(project_reviews, portfolio_reviews)``."""
    with projects_db.connect_closing(root / "projects.db") as registry:
        row = registry.execute("SELECT * FROM project_review_state WHERE project_id = ?", (project.id,)).fetchone()
        return dict(row), review.budget(registry, day)


def _further_reviews(project, day=DAY):
    """The day's review tasks whose key ends in ``:again``: its one further review, if the tick made it."""
    prefix = f"{review.CREATED_BY}:{project.id}:{day}:"
    with _summaries() as board:
        rows = board.execute(
            "SELECT id, status, idempotency_key FROM tasks WHERE created_by = ?", (review.CREATED_BY,),
        ).fetchall()
    return [dict(row) for row in rows if row["idempotency_key"].startswith(prefix)
            and row["idempotency_key"].endswith(":again")]


def _waiting(card):
    """The classification of a review while the card's owner wait is open."""
    return [{"item": card, "class": "still actionable", "evidence": [f"decisions:wait:{card}"]}]


def _publish_before_claim(root, monkeypatch):
    """Publish the day's further review and exit before its claim: returns the Project, the task and its key."""
    project, card, _first = _round3_setup_result(root, monkeypatch)
    # A new wait on the card makes the first result stale: the day gets one further review, whose key ends in ":again".
    _round3_mutate(project, card, "new_wait")
    create_task = kanban_db.create_task
    published = []

    def publish_then_exit(conn, *args, **kwargs):
        task_id = create_task(conn, *args, **kwargs)
        if kwargs.get("created_by") == review.CREATED_BY and kwargs.get("idempotency_key", "").endswith(":again"):
            published.append(task_id)
            raise RuntimeError("the publisher exits after the board publication and before the registry claim")
        return task_id

    with monkeypatch.context() as patch:
        patch.setattr(kanban_db, "create_task", publish_then_exit)
        _round3_drive(patch, NOW + 1)
    assert len(published) == 1
    state, spent = _registry(root, project)
    assert (state["open_task_id"], state["failures"], spent) == (None, 0, (1, 0)), "only the first review is counted"
    with _summaries() as board:
        return project, published[0], kanban_db.get_task(board, published[0]).idempotency_key


def _end(task_id, outcome):
    """How a task ends before any tick records it: left ready, blocked, done without typed output, or archived."""
    with _summaries() as board:
        if outcome in ("blocked", "malformed_done"):
            assert kanban_db.claim_task(board, task_id)
            if outcome == "blocked":
                assert kanban_db.block_task(board, task_id, reason="The review cannot finish.")
            else:
                summary = "Review finished without typed output."
                assert kanban_db.complete_task(board, task_id, summary=summary, metadata={})
        elif outcome == "archived":
            assert kanban_db.archive_task(board, task_id)


@pytest.mark.parametrize("outcome", ["ready", "blocked", "malformed_done", "archived"])
def test_a_further_review_published_before_its_claim_uses_one_slot_however_it_ended(
    round3_root, monkeypatch, outcome,
):
    root = round3_root
    project, further, further_key = _publish_before_claim(root, monkeypatch)
    _end(further, outcome)
    _round3_drive(monkeypatch, NOW + 2)
    _round3_drive(monkeypatch, NOW + 3)
    _round3_drive(monkeypatch, NOW + 3 * 3600)

    with _summaries() as board:
        tasks = [dict(row) for row in board.execute(
            "SELECT id, status, idempotency_key FROM tasks WHERE created_by = ?", (review.CREATED_BY,),
        )]
    state, spent = _registry(root, project)
    assert len(tasks) == 2, f"{len(tasks)} review tasks while the budget records {spent[0]}: a further one was published"
    assert spent == (2, 0), "the first review and the further one use one slot each"
    assert state["stale_refused_key"] == further_key
    # A ready review is still open; the others were refused once, as a claimed task with the same ending is.
    assert (state["open_task_id"], state["failures"]) == ((further, 0) if outcome == "ready" else (None, 1))


@pytest.mark.parametrize("delay", [3 * HOUR, 18 * HOUR, 24 * HOUR, 48 * HOUR], ids=[
    "later-the-same-day", "next-day-before-the-review-hour", "next-day-after-the-review-hour", "two-days-later",
])
@pytest.mark.parametrize("outcome", ["ready", "blocked", "archived"])
def test_a_further_review_found_by_a_late_tick_is_counted_on_the_day_of_its_key(
    round3_root, monkeypatch, outcome, delay,
):
    root = round3_root
    project, further, further_key = _publish_before_claim(root, monkeypatch)
    _end(further, outcome)
    _round3_drive(monkeypatch, NOW + delay)

    state, spent = _registry(root, project)
    assert spent == (2, 0), f"{spent[0]} of 2 slots of {DAY} used: the further review published that day is not counted"
    assert [task["id"] for task in _further_reviews(project)] == [further], "exactly one further review of the day"
    assert state["stale_refused_key"] == further_key


@pytest.mark.parametrize("first", [18 * HOUR, 24 * HOUR], ids=[
    "next-day-before-the-review-hour", "next-day-after-the-review-hour",
])
def test_repeated_ticks_after_a_late_recovery_count_the_day_once(round3_root, monkeypatch, first):
    root = round3_root
    project, further, further_key = _publish_before_claim(root, monkeypatch)
    _end(further, "archived")
    # The first tick finds the archived task, the next harvest refuses it as it does a claimed one, later days follow.
    for delay in sorted({first, first + 1, first + 2, 24 * HOUR, 27 * HOUR, 48 * HOUR, 72 * HOUR}):
        _round3_drive(monkeypatch, NOW + delay)
        state, spent = _registry(root, project)
        assert spent == (2, 0), f"{spent[0]} of 2 slots of {DAY} used after the tick {delay} s on"
        assert [task["id"] for task in _further_reviews(project)] == [further], "exactly one further review of the day"
        assert state["stale_refused_key"] == further_key


@pytest.mark.parametrize("ending", ["typed_result", "blocked"])
def test_a_counted_further_review_the_harvest_archived_is_not_counted_again_on_later_days(
    round3_root, monkeypatch, ending,
):
    root = round3_root
    project, card, _first = _round3_setup_result(root, monkeypatch)
    _round3_mutate(project, card, "new_wait")
    _round3_drive(monkeypatch, NOW + 1)
    state, spent = _registry(root, project)
    further, further_key = state["open_task_id"], state["stale_refused_key"]
    assert further is not None and further_key.endswith(":again"), "the stale result's further review is claimed"
    assert spent == (2, 0)
    # It ends, and the next harvest archives it: a counted review that stays on the board for good.
    if ending == "typed_result":
        _complete_review(further, _waiting(card))
    else:
        _end(further, "blocked")
    _round3_drive(monkeypatch, NOW + 2)
    with _summaries() as board:
        assert kanban_db.get_task(board, further).status == "archived"
    state, spent = _registry(root, project)
    assert (state["open_task_id"], state["stale_refused_key"], spent) == (None, further_key, (2, 0))

    # The next day has news. Its first review goes stale too, which replaces the refused key of the row.
    later = NOW + 24 * HOUR
    monkeypatch.setattr(time, "time", lambda: later)
    path = kanban_db.kanban_db_path(board=project.board_slug)
    with kanban_db.connect_closing(path, board=project.board_slug) as board:
        kanban_db.add_comment(board, card, "default", "Please book the room today.")
    _round3_drive(monkeypatch, later)
    first = _registry(root, project, NEXT_DAY)[0]["open_task_id"]
    assert first not in (None, further), f"the first review of {NEXT_DAY} is published"
    _complete_review(first, _waiting(card))
    _round3_mutate(project, card, "new_suggestion")
    for instant in (later + 1, later + 2, later + 2 * HOUR, NOW + 48 * HOUR):
        _round3_drive(monkeypatch, instant)
        state, spent = _registry(root, project)
        assert spent == (2, 0), (
            f"{spent[0]} of 2 slots of {DAY} used {instant - later} s into {NEXT_DAY}: counted again"
        )
        assert [task["id"] for task in _further_reviews(project)] == [further]

    # The next day's own further review is published and counted on its own day, once.
    state, spent = _registry(root, project, NEXT_DAY)
    next_day_further = _further_reviews(project, NEXT_DAY)
    assert len(next_day_further) == 1, f"{len(next_day_further)} further reviews of {NEXT_DAY}"
    assert spent == (2, 0), "the first and the further review of the next day use one slot each"
    assert state["stale_refused_key"] == next_day_further[0]["idempotency_key"]
