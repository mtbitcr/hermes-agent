"""The Update route of a model update suggestion on the Decisions feed.

``POST /v1/owner-workspace/decisions/{ref}/update`` with ``{"reason": ...}``:
the owner authentication, owner-workspace check, request shape and error
answers of the accept, reject and defer routes, then one call into
:func:`hermes_cli.owner_model_updates.apply_model_update`. Later and Keep
current stay the defer and reject routes.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from aiohttp import web

from gateway.platforms.api_server import _openai_error, _owner_workspace_toolset_enabled
from hermes_cli import owner_model_updates as updates
from hermes_cli import owner_workspace as ow

logger = logging.getLogger(__name__)

# The status of each answer that leaves nothing half done; its sentence is the
# owner text the refusal carries.
_REFUSALS = {
    "decision_not_found": 404,
    "invalid_argument": 400,
    "model_update_unavailable": 409,
    "model_update_not_applied": 409,
}


def route(adapter: Any):
    """The handler of the Update route, behind the owner authentication of the decision routes."""

    async def handle(request: web.Request) -> web.Response:
        auth_err = adapter._check_auth(request)
        if auth_err:
            return auth_err
        try:
            from gateway.run import _load_gateway_config

            if not _owner_workspace_toolset_enabled(_load_gateway_config()):
                return web.json_response(
                    _openai_error(
                        "Owner workspace is not enabled for this profile",
                        code="owner_workspace_not_enabled",
                    ),
                    status=404,
                )
            body, err = await adapter._read_json_body(request)
            if err:
                return err
            reason = body.get("reason")
            if set(body) != {"reason"} or not isinstance(reason, str) or not reason.strip():
                return web.json_response(
                    _openai_error("Invalid owner decision request"), status=400
                )
            decided = await asyncio.to_thread(
                updates.apply_model_update,
                ow.resolve_owner_context(),
                decision_ref=request.match_info.get("ref", ""),
                reason=reason,
            )
        except ow.OwnerWorkspaceError as exc:
            if exc.code in _REFUSALS:
                return web.json_response(
                    _openai_error(exc.message, code=exc.code), status=_REFUSALS[exc.code]
                )
            logger.warning("[api_server] owner model update unavailable (%s)", exc.code)
            return web.json_response(
                _openai_error(
                    "Owner workspace decision is unavailable",
                    code="owner_workspace_unavailable",
                ),
                status=503,
            )
        except Exception:
            logger.exception("[api_server] owner model update failed")
            return web.json_response(
                _openai_error(
                    "Owner workspace decision is unavailable",
                    code="owner_workspace_unavailable",
                ),
                status=500,
            )

        return web.json_response({"object": "hermes.owner_workspace.decision", "data": decided})

    return handle
