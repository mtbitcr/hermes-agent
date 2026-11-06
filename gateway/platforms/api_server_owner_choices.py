"""The owner chooses one of a designer card's pictured options (card 2 of the design options plan), served by the API
server.

One route, ``POST /v1/owner-workspace/decisions/{ref}/choose/{n}``, behind the owner authentication of the release
routes and modelled on the upload route of ``api_server_owner_inputs``. It checks, in the plan's order: the owner
authority, never a body; that the owner workspace is enabled; that ``{ref}`` is a key of the owner's own Decisions feed
for a card held ``needs_input`` right now (``_waiting_card``); and the idempotency cache. ``accept_choice`` then
confirms that the card still waits for a choice and offers option ``{n}``, records the choice once as the event
``design_option_chosen``, comments on the card and releases it, in one write transaction. A late tap, a second
different choice, an option the card never offered and an unknown or foreign key are refused alike with the plan's
sentence "This choice is no longer waiting." and ``decision_not_found``. With an ``Idempotency-Key``, a retry of a
recorded choice gets the first answer and records nothing new. No answer holds an internal detail.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from aiohttp import web

from gateway.platforms.api_server import (
    _IdempotencyConflict,
    _idem_cache,
    _make_request_fingerprint,
    _owner_workspace_toolset_enabled,
)
from gateway.platforms.api_server_owner_inputs import _KEY, _NUMBER, _refusal, _waiting_card
from hermes_cli import kanban_db
from hermes_cli import owner_workspace as ow
from hermes_cli.owner_design_choices import accept_choice
from hermes_cli.owner_text import check_owner_text

logger = logging.getLogger(__name__)

# The status, sentence and code of a choice that no longer waits, the sentence word for word from the plan (section C).
_NOT_WAITING = (404, "This choice is no longer waiting.", "decision_not_found")


class _Refused(Exception):
    """A refusal of accept_choice, raised so that the idempotency cache never keeps it."""


def route(adapter: Any):
    """The handler of the choose route, behind the owner authentication of the release routes."""

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
            ref, number = request.match_info["ref"], request.match_info["n"]
            n = int(number) if _NUMBER.fullmatch(number) else 0  # 0 is never an offered option

            async def choose() -> dict[str, Any]:
                outcome = await asyncio.to_thread(_choose, owner, ref, n)
                if outcome == "not_waiting":
                    raise _Refused(outcome)
                return {
                    "object": "hermes.owner_workspace.design_choice",
                    "decision_ref": ref,
                    "n": n,
                    "released": outcome == "released",
                }

            if key is None:
                return web.json_response(await choose())
            sent = {"ref": ref, "n": n}
            return web.json_response(await _idem_cache.get_or_set(
                f"owner_choice:{owner.profile}:{key}", _make_request_fingerprint(sent, sorted(sent)), choose,
            ))
        except _Refused:
            return _refusal(*_NOT_WAITING)
        except _IdempotencyConflict:
            return _refusal(409, failure, "idempotency_conflict")
        except Exception:
            logger.exception("[api_server] owner design choice failed")
            return _refusal(503, failure, "owner_workspace_unavailable")

    return handle


def _choose(owner: ow.OwnerContext, ref: str, n: int) -> str:
    card = _waiting_card(owner, ref)
    if card is None:
        return "not_waiting"
    _board, store, task_id = card
    # The card's own store by its path: a worker's pinned HERMES_KANBAN_DB never redirects it.
    with kanban_db.connect_closing(store) as conn:
        return accept_choice(conn, task_id, n, actor=owner.actor)
