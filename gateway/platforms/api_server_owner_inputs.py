"""The owner gives a card that waits for a file the file it names (card C3), served by the API server.

One route, ``POST /v1/owner-workspace/decisions/{ref}/inputs/{n}``, behind the owner authentication of the release
routes; its body is the raw file. Before any byte is stored it checks, in the plan's order: the owner authority, never
the body; that ``{ref}`` is a key of the owner's own Decisions feed for a card held for missing inputs right now; and,
through ``accept_upload``, that ``{n}`` is a still-missing input of that card, that the file is not empty and at most
8 MiB, that its Content-Type fits the input's declared type, and that it is strict UTF-8 text without a NUL byte, which
parses when the type is json. ``accept_upload`` then stores the file as the owner's under its derived name, and
releases the card once no input is missing, in one write transaction. With an ``Idempotency-Key``, a retry of an
accepted upload gets the first answer and stores nothing new. No answer holds an internal detail; each refusal is one
plain sentence of the plan.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import re
from pathlib import Path
from typing import Any, Optional

from aiohttp import web

from gateway.platforms.api_server import (
    _IdempotencyConflict,
    _idem_cache,
    _make_request_fingerprint,
    _openai_error,
    _owner_workspace_toolset_enabled,
)
from hermes_cli import kanban_db
from hermes_cli import owner_workspace as ow
from hermes_cli.owner_named_inputs import MAX_INPUT_BYTES, accept_upload
from hermes_cli.owner_text import check_owner_text

logger = logging.getLogger(__name__)

# The status, sentence and code of each refusal of accept_upload, the sentences word for word from the plan.
_REFUSALS = {
    "not_waiting": (404, "This request is no longer waiting for a file.", "decision_not_found"),
    "too_big": (413, "This file is larger than 8 MB. Please send a smaller file.", "body_too_large"),
    "empty": (400, "This file is empty. Please choose the file again.", "invalid_argument"),
    "wrong_type": (415, "Raphael can only take text files here: .txt, .md, .csv or .json.", "invalid_argument"),
    "type_mismatch": (
        415, "This is not the kind of file Raphael asked for. Please send the file it named.", "invalid_argument"
    ),
    "not_text": (
        400, "This file does not look like plain text. Please save it as text and try again.", "invalid_argument"
    ),
}
# An Idempotency-Key as the owner routes take it, and an input number as declared, without a leading zero.
_KEY = re.compile(r"[A-Za-z0-9._:-]{1,200}")
_NUMBER = re.compile(r"[1-9][0-9]{0,2}")


class _Refused(Exception):
    """A refusal of accept_upload, raised so that the idempotency cache never keeps it."""


def route(adapter: Any):
    """The handler of the upload route, behind the owner authentication of the release routes."""

    async def handle(request: web.Request) -> web.Response:
        failure = check_owner_text(None, "stored_failure")
        if adapter._check_auth(request):  # the owner authority, never the body
            return _refusal(401, failure, "gateway_auth_failed")
        try:
            from gateway.run import _load_gateway_config

            if not _owner_workspace_toolset_enabled(_load_gateway_config()):
                return _refusal(404, failure, "owner_workspace_not_enabled")
            owner = ow.resolve_owner_context()
            key = request.headers.get("Idempotency-Key")
            if key is not None and not _KEY.fullmatch(key):
                return _refusal(400, failure, "invalid_argument")
            ref, number = request.match_info["ref"], request.match_info["n"]
            n = int(number) if _NUMBER.fullmatch(number) else 0  # 0 is never a declared input
            data, content_type = await _body(request), request.content_type

            async def upload() -> dict[str, Any]:
                outcome = await asyncio.to_thread(_upload, owner, ref, n, data, content_type)
                if outcome in _REFUSALS:
                    raise _Refused(outcome)
                return {
                    "object": "hermes.owner_workspace.input_upload",
                    "decision_ref": ref,
                    "n": n,
                    "released": outcome == "released",
                }

            if key is None:
                return web.json_response(await upload())
            sent = {"ref": ref, "n": n, "content_type": content_type, "sha256": hashlib.sha256(data).hexdigest()}
            return web.json_response(await _idem_cache.get_or_set(
                f"owner_input:{owner.profile}:{key}", _make_request_fingerprint(sent, sorted(sent)), upload,
            ))
        except _Refused as refused:
            return _refusal(*_REFUSALS[refused.args[0]])
        except _IdempotencyConflict:
            return _refusal(409, failure, "idempotency_conflict")
        except Exception:
            logger.exception("[api_server] owner input upload failed")
            return _refusal(503, failure, "owner_workspace_unavailable")

    return handle


async def _body(request: web.Request) -> bytes:
    """The raw file, read up to one byte past the largest one, so a larger file is refused unread."""
    try:
        return await request.content.readexactly(MAX_INPUT_BYTES + 1)
    except asyncio.IncompleteReadError as whole:
        return whole.partial


def _upload(owner: ow.OwnerContext, ref: str, n: int, data: bytes, content_type: str) -> str:
    card = _waiting_card(owner, ref)
    if card is None:
        return "not_waiting"
    board, store, task_id = card
    # The card's own store by its path: a worker's pinned HERMES_KANBAN_DB never redirects it.
    with kanban_db.connect_closing(store) as conn:
        return accept_upload(conn, task_id, n, data, content_type, board=board)


def _waiting_card(owner: ow.OwnerContext, ref: str) -> Optional[tuple[str, Path, str]]:
    """Resolve *ref* through the owner's own Decisions feed to a card held for missing inputs right now.

    Reads the owner-input cards ``list_owner_decisions`` lists, in the same Project scope, and recomputes their keys,
    as ``_owner_pending_suggestion`` does for suggestions, so a key is never reversed and a native id, a suggestion or
    a key minted for another owner never matches. Returns ``(board, its kanban.db, task id)`` or ``None``, and never
    writes.
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
                "AND status = 'blocked' AND block_kind = 'needs_input'",
                (project_id,),
            ).fetchall()
        finally:
            conn.close()
        for row in rows:
            if ow._owner_decision_ref(owner, authority="task", native_id=str(row["id"])) == ref:
                return board, store, str(row["id"])
    return None


def _refusal(status: int, message: str, code: str) -> web.Response:
    return web.json_response(_openai_error(message, code=code), status=status)
