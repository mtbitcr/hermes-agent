"""The hand-over check of a daily review or portfolio result, inside kanban_complete and before anything is written.

It runs the harvest's own result rules against the task's frozen input, so the reviewer corrects a broken result in the
same run. It is an early copy of the harvest's verdict, never the authority: it never writes, a broken input or any
fault of its own lets the completion through to the harvest, and its message and log carry only codes, field paths,
numbers and kernel ids.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from hermes_cli import kanban_db

logger = logging.getLogger("hermes_cli.owner_project_review")

_PORTFOLIO_KEY = "project-review:portfolio:"


def handover_refusal(task: Any, metadata: Any) -> Optional[dict]:
    """The code, detail and sentences of a review or portfolio result its frozen input alone refuses, else None."""
    try:
        if task is None or task.created_by != kanban_db.REVIEW_TASK_CREATOR:
            return None
        from hermes_cli.owner_project_review import read_settings

        # Nothing changes while the daily review switch is off; the root config is read, not the worker's profile.
        if not read_settings().enabled:
            return None
        if (task.idempotency_key or "").startswith(_PORTFOLIO_KEY):
            return _portfolio_refusal(task, metadata)
        return _review_refusal(task, metadata)
    except Exception as exc:
        # The harvest still judges the stored result, so a fault here never holds a completion back.
        logger.warning("review_handover_check_failed: task %s (%s)", getattr(task, "id", None), type(exc).__name__)
        return None


def _answer(exc: Any) -> dict:
    from hermes_cli.owner_project_review_refusal import refusal_sentences

    return {"code": exc.code, "detail": exc.detail, "message": refusal_sentences(exc.code, exc.detail)}


def _review_refusal(task: Any, metadata: Any) -> Optional[dict]:
    from hermes_cli import owner_portfolio_review as portfolio, owner_project_review as review
    from hermes_cli.owner_project_review_refusal import class_history, refusal_tokens

    fields = (task.idempotency_key or "").split(":")
    project_id = fields[1] if len(fields) > 1 else None
    # A worker cannot fix its own input: an input fault passes, and the harvest refuses it as before.
    try:
        sources, _cut = review._frozen_sources(task.body)
        if sources["project"][0].get("id") != project_id:
            return None
        class_history(sources, review._REVIEW_CLASSES, review.ReviewRefused)
    except (review.ReviewRefused, portfolio.PortfolioRefused, ValueError):
        return None
    value = metadata.get("project_review") if isinstance(metadata, dict) else None
    try:
        review.check_result_texts(value, sources)
        review.check_claims(sources, value["classification"])
        review.check_closure_shapes(value["proposed_closures"])
    except review.ReviewRefused as exc:
        tokens = refusal_tokens(exc.code, exc.detail)
        logger.warning(
            "review_handover_refused: Project %s task %s run %s (%s)%s", project_id, task.id, task.current_run_id,
            exc.code, f" {tokens}" if tokens else "",
        )
        return _answer(exc)
    return None


def _portfolio_refusal(task: Any, metadata: Any) -> Optional[dict]:
    from hermes_cli import owner_portfolio_review as portfolio, owner_project_review as review
    from hermes_cli.owner_project_review_refusal import refusal_tokens

    try:
        portfolio._portfolio_input(task.body)
    except (review.ReviewRefused, portfolio.PortfolioRefused, ValueError):
        return None
    value = metadata.get("portfolio_review") if isinstance(metadata, dict) else None
    try:
        portfolio.check_portfolio_result(value, task.body)
    except portfolio.PortfolioRefused as exc:
        tokens = refusal_tokens(exc.code, exc.detail)
        logger.warning(
            "portfolio_handover_refused: task %s run %s (%s)%s", task.id, task.current_run_id, exc.code,
            f" {tokens}" if tokens else "",
        )
        return _answer(exc)
    return None
