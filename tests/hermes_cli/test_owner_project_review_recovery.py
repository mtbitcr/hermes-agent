"""A review task published on the summary board before its registry claim is counted once, however it ended."""

from __future__ import annotations

import pytest

from hermes_cli import kanban_db, owner_project_review as review, projects_db
from tests.hermes_cli.test_owner_project_review_result import (  # noqa: F401
    NOW,
    _round3_drive,
    _round3_mutate,
    _round3_setup_result,
    root,
    round3_root,
)

SUMMARIES = "project-summaries"
DAY = "2026-11-09"


def _summaries():
    return kanban_db.connect_closing(kanban_db.kanban_db_path(board=SUMMARIES), board=SUMMARIES)


def _registry(root, project):
    """The Project's review state and the day's budget: ``(project_reviews, portfolio_reviews)``."""
    with projects_db.connect_closing(root / "projects.db") as registry:
        row = registry.execute("SELECT * FROM project_review_state WHERE project_id = ?", (project.id,)).fetchone()
        return dict(row), review.budget(registry, DAY)


@pytest.mark.parametrize("outcome", ["ready", "blocked", "malformed_done", "archived"])
def test_a_further_review_published_before_its_claim_uses_one_slot_however_it_ended(
    round3_root, monkeypatch, outcome,
):
    root = round3_root
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
    further = published[0]
    state, spent = _registry(root, project)
    assert (state["open_task_id"], state["failures"], spent) == (None, 0, (1, 0)), "only the first review is counted"

    with _summaries() as board:
        further_key = kanban_db.get_task(board, further).idempotency_key
        if outcome in ("blocked", "malformed_done"):
            assert kanban_db.claim_task(board, further)
            if outcome == "blocked":
                assert kanban_db.block_task(board, further, reason="The review cannot finish.")
            else:
                summary = "Review finished without typed output."
                assert kanban_db.complete_task(board, further, summary=summary, metadata={})
        elif outcome == "archived":
            assert kanban_db.archive_task(board, further)
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
