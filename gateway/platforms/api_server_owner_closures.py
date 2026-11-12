"""Undo one displayed closure through the existing owner authority boundary."""

from __future__ import annotations

import asyncio
import logging

from aiohttp import web

from gateway.platforms.api_server import (
    _IdempotencyConflict,
    _idem_cache,
    _make_request_fingerprint,
    _owner_workspace_toolset_enabled,
)
from gateway.platforms.api_server_owner_inputs import _KEY, _refusal
from hermes_cli import owner_workspace as ow
from hermes_cli.owner_text import check_owner_text

logger = logging.getLogger(__name__)

# Plain owner sentences: the closure is no longer live, or the request does not name it with a reason.
_NOT_UNDOABLE = "This closure can no longer be undone."
_INVALID = "This Undo needs a short reason and the closure it undoes."


class _Refused(Exception):
    """An owner refusal, raised so that the idempotency cache never keeps it."""


def route(adapter):
    async def handle(request):
        denied = adapter._check_auth(request)
        if denied is not None:
            return denied
        failure = check_owner_text(None, "stored_failure")
        try:
            from gateway.run import _load_gateway_config

            if not _owner_workspace_toolset_enabled(_load_gateway_config()):
                return _refusal(404, failure, "owner_workspace_not_enabled")
            key = request.headers.get("Idempotency-Key")
            if key is not None and not _KEY.fullmatch(key):
                return _refusal(400, _INVALID, "invalid_argument")
            try:
                body = await request.json()
            except web.HTTPRequestEntityTooLarge:
                return _refusal(413, _INVALID, "invalid_argument")
            except (ValueError, UnicodeDecodeError):
                return _refusal(400, _INVALID, "invalid_argument")
            if not isinstance(body, dict) or set(body) != {"reason", "expected_closed_at"}:
                return _refusal(400, _INVALID, "invalid_argument")
            owner = ow.resolve_owner_context()
            ref = request.match_info["decision_ref"]

            async def undo():
                try:
                    result = await asyncio.to_thread(
                        ow.undo_owner_suggestion_closure, owner, decision_ref=ref, reason=body["reason"],
                        expected_closed_at=body["expected_closed_at"],
                    )
                except ow.OwnerWorkspaceError as exc:
                    raise _Refused(exc.code) from exc
                return {"object": "hermes.owner_workspace.closure_undo", "data": result}

            if key is None:
                return web.json_response(await undo())
            sent = {"decision_ref": ref, **body}
            return web.json_response(await _idem_cache.get_or_set(
                f"owner_closure_undo:{owner.profile}:{key}", _make_request_fingerprint(sent, sorted(sent)), undo,
            ))
        except _Refused as refused:
            code = str(refused)
            if code == "decision_not_found":
                return _refusal(404, _NOT_UNDOABLE, code)
            if code == "invalid_argument":
                return _refusal(400, _INVALID, code)
            if code == "permission_denied":
                return _refusal(403, failure, code)
            logger.warning("[api_server] owner closure Undo unavailable (%s)", code)
            return _refusal(503, failure, "owner_workspace_unavailable")
        except _IdempotencyConflict:
            return _refusal(409, failure, "idempotency_conflict")
        except Exception:
            logger.exception("[api_server] owner closure Undo failed")
            return _refusal(503, failure, "owner_workspace_unavailable")

    return handle
