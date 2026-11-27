"""The owner answers the question of a stopped card (card P3 of the owner answer plan), served by the API server.

One route, ``POST /v1/owner-workspace/decisions/{ref}/answer``, behind the owner authentication of the release routes
and built as the choose route of ``api_server_owner_choices``. It checks, in that route's order: the owner authority,
never a body; that the owner workspace is enabled; the shape of the ``Idempotency-Key`` header; the body, one JSON
object of exactly ``stop_ref``, ``choice`` and ``words``, before any card is looked up; that ``{ref}`` is a key of the
owner's own Decisions list for a card stopped for the owner right now (``_waiting_card``); and the idempotency cache.
``accept_answer`` then confirms, inside its own write transaction, that the stop still takes this answer, records it
once, comments it on the card as ``owner`` and resumes the card. A refusal is never cached, and no answer holds an
internal detail.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import unicodedata
from pathlib import Path
from typing import Any, Optional

from aiohttp import web

from gateway.platforms.api_server import (
    _IdempotencyConflict,
    _idem_cache,
    _make_request_fingerprint,
    _owner_workspace_toolset_enabled,
)
from gateway.platforms.api_server_owner_inputs import _KEY, _refusal
from hermes_cli import kanban_db
from hermes_cli import owner_workspace as ow
from hermes_cli.owner_questions import MAX_CHOICES, WORDS_LIMIT, accept_answer
from hermes_cli.owner_text import check_owner_text

logger = logging.getLogger(__name__)

# The status, sentence and code of a question that no longer waits, the sentence word for word from the plan.
_NOT_WAITING = (404, "This question is no longer waiting.", "decision_not_found")
# The reference of one stop, as the Decisions list serves it.
_STOP_REF = re.compile(r"stop_[0-9a-f]{32}")


class _Refused(Exception):
    """A refusal of accept_answer, raised so that the idempotency cache never keeps it."""


def route(adapter: Any):
    """The handler of the answer route, behind the owner authentication of the release routes."""

    async def handle(request: web.Request) -> web.Response:
        failure = check_owner_text(None, "stored_failure")
        if adapter._check_auth(request):  # the owner authority, never a body
            return _refusal(401, failure, "gateway_auth_failed")
        try:
            from gateway.run import _load_gateway_config

            if not _owner_workspace_toolset_enabled(_load_gateway_config()):
                return _refusal(404, failure, "owner_workspace_not_enabled")
            owner = ow.resolve_owner_context()
            key = request.headers.get("Idempotency-Key")
            if key is not None and not _KEY.fullmatch(key):
                return _refusal(400, failure, "invalid_argument")
            body = await _body(request)
            if body is None:  # and the card is not looked up
                return _refusal(400, failure, "invalid_argument")
            ref, (stop_ref, choice, words) = request.match_info["ref"], body

            async def answer() -> dict[str, Any]:
                outcome = await asyncio.to_thread(_answer, owner, ref, stop_ref, choice, words)
                if outcome in ("not_waiting", "not_allowed"):
                    raise _Refused(outcome)
                return {
                    "object": "hermes.owner_workspace.owner_answer",
                    "decision_ref": ref,
                    "stop_ref": stop_ref,
                    "choice": choice,
                    "resumed": outcome == "resumed",
                }

            if key is None:
                return web.json_response(await answer())
            sent = {"ref": ref, "stop_ref": stop_ref, "choice": choice, "words": words}
            return web.json_response(await _idem_cache.get_or_set(
                f"owner_answer:{owner.profile}:{key}", _make_request_fingerprint(sent, sorted(sent)), answer,
            ))
        except _Refused as refused:
            if refused.args[0] == "not_allowed":
                return _refusal(400, failure, "invalid_argument")
            return _refusal(*_NOT_WAITING)
        except _IdempotencyConflict:
            return _refusal(409, failure, "idempotency_conflict")
        except Exception:
            logger.exception("[api_server] owner answer failed")
            return _refusal(503, failure, "owner_workspace_unavailable")

    return handle


async def _body(request: web.Request) -> Optional[tuple[str, Optional[int], Optional[str]]]:
    """The ``stop_ref``, ``choice`` and ``words`` of the request's body, or ``None`` for any other body."""
    try:
        body = json.loads(await request.read())
    except (ValueError, RecursionError, web.HTTPRequestEntityTooLarge):
        return None
    if not isinstance(body, dict) or set(body) != {"stop_ref", "choice", "words"}:
        return None
    stop_ref, choice, words = body["stop_ref"], body["choice"], body["words"]
    if not isinstance(stop_ref, str) or not _STOP_REF.fullmatch(stop_ref):
        return None
    if choice is not None and (type(choice) is not int or not 1 <= choice <= MAX_CHOICES):
        return None  # a boolean is refused too
    if words is not None and not (
        isinstance(words, str) and 1 <= len(words) <= WORDS_LIMIT and not words.isspace()
        and all(char == "\n" or unicodedata.category(char) != "Cc" for char in words)
    ):
        return None
    if choice is None and words is None:
        return None
    return stop_ref, choice, words


def _answer(owner: ow.OwnerContext, ref: str, stop_ref: str, choice: Optional[int], words: Optional[str]) -> str:
    card = _waiting_card(owner, ref)
    if card is None:
        return "not_waiting"
    _board, store, task_id = card
    # The card's own store by its path: a worker's pinned HERMES_KANBAN_DB never redirects it.
    with kanban_db.connect_closing(store) as conn:
        return accept_answer(conn, task_id, stop_ref, choice, words, owner=owner)


def _waiting_card(owner: ow.OwnerContext, ref: str) -> Optional[tuple[str, Path, str]]:
    """Resolve *ref* through the owner's own Decisions list to a card stopped for the owner right now.

    Built as ``_waiting_card`` of ``api_server_owner_inputs`` is built, with one difference: the card is held with the
    block kind ``needs_input`` or ``capability``. Reads the cards ``list_owner_decisions`` lists, in the same Project
    scope, and recomputes their keys, so a key is never reversed and a native id, a suggestion or a key minted for
    another owner never matches. Returns ``(board, its kanban.db, task id)`` or ``None``, and never writes.
    """
    for project in ow.list_committed_projects(owner):
        if project["archived"]:
            continue
        project_id, board = str(project["project_id"]), str(project["board"])
        if not kanban_db.board_exists(board) and ow._completed_removal_state(project_id) is not None:
            continue
        folder = kanban_db.kanban_home() if board == kanban_db.DEFAULT_BOARD else kanban_db.board_dir(board)
        store = folder / "kanban.db"
        conn = ow._open_read_only_sqlite(store, label="kanban.db")
        try:
            rows = conn.execute(
                "SELECT id FROM tasks WHERE project_id = ? AND task_kind = 'work' "
                "AND status = 'blocked' AND block_kind IN ('needs_input', 'capability')",
                (project_id,),
            ).fetchall()
        finally:
            conn.close()
        for row in rows:
            if ow._owner_decision_ref(owner, authority="task", native_id=str(row["id"])) == ref:
                return board, store, str(row["id"])
    return None
