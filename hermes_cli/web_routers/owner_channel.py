"""Owner channel: the Telegram row of the owner app.

``GET /api/owner-channel`` answers whether the root profile names a Telegram
home channel and whether owner messages are paused, ``PUT
/api/owner-channel/paused`` pauses or resumes them through
``gateway.owner_channel_pause``, and ``POST /api/owner-channel/test`` sends one
fixed test message to the Telegram home channel through
``tools.send_message_tool.send_message_tool``, the path ``hermes send --to
telegram`` takes. They answer the owner app's connections credential and no
other caller, and every call writes one audit row. An answer never holds a
token, chat id, user name or provider text.
"""

import asyncio
import json
import threading
import time
import uuid
from typing import Any, Dict, Optional, Tuple

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse

from gateway import owner_channel_pause
from hermes_cli.dashboard_auth.token_auth import register_machine_token_family, register_token_route
from hermes_cli.web_routers.connection_rows import _require_connections_credential
from hermes_cli.web_routers.mcp import _audit_connections_machine_success
from plugins.dashboard_auth.raphael_workspace import CONNECTIONS_SCOPE, CONNECTIONS_TOKEN_PREFIX

router = APIRouter()

_ROUTE = "/api/owner-channel"
_PAUSED_ROUTE = "/api/owner-channel/paused"
_TEST_ROUTE = "/api/owner-channel/test"

TEST_TEXT = "Test from Raphael: your message channel works."
_TEST_INTERVAL_SECONDS = 30
_NO_HOME_CHANNEL = "No Telegram home channel is set."
_NOT_SENT = "The test message could not be sent."
_UNKNOWN = "It is not known whether the test message was sent."

#: The answer given to each request_id that reached the send step, for the life of this process.
_test_answers: Dict[str, Dict[str, Optional[str]]] = {}
#: ``time.monotonic()`` when the last test went out; None before the first.
_last_test_sent: Optional[float] = None
#: Held from the request_id lookup to the stored answer, so one request_id sends at most once.
_test_lock = threading.Lock()


def _register_owner_channel_routes() -> None:
    """Token-only (no session), and only for a token carrying the connections scope."""
    register_machine_token_family(CONNECTIONS_TOKEN_PREFIX, strict_audit=True)
    register_token_route(_ROUTE, method="GET", required_scope=CONNECTIONS_SCOPE, strict_audit=True)
    register_token_route(_PAUSED_ROUTE, method="PUT", required_scope=CONNECTIONS_SCOPE, strict_audit=True)
    register_token_route(_TEST_ROUTE, method="POST", required_scope=CONNECTIONS_SCOPE, strict_audit=True)


_register_owner_channel_routes()


async def _json_body(request: Request) -> Any:
    """The parsed body; None when it is not JSON, which every route refuses as well."""
    try:
        return await request.json()
    except ValueError:
        return None


def _channel() -> Dict[str, Any]:
    """The row: whether a home channel is set and since when owner messages are paused, nothing else."""
    since = owner_channel_pause.paused_since()
    return {
        "state": "ready" if owner_channel_pause.has_owner_home_chat() else "not_set",
        "paused": since is not None,
        "paused_since": since,
    }


def _request_id(body: Any) -> Optional[str]:
    """The canonical form of the body's request_id; None for any body but ``{"request_id": "<uuid>"}``."""
    if not (isinstance(body, dict) and set(body) == {"request_id"} and isinstance(body["request_id"], str)):
        return None
    try:
        return str(uuid.UUID(body["request_id"]))
    except ValueError:
        return None


def _outcome(raw: Any) -> Dict[str, Optional[str]]:
    """The send tool's answer as a fixed outcome: never its provider text, nor the chat id in its note."""
    result = raw
    if isinstance(raw, (str, bytes)):
        try:
            result = json.loads(raw)
        except ValueError:
            result = None
    if not isinstance(result, dict):
        return {"outcome": "unknown", "reason": _UNKNOWN}
    if "error" in result or result.get("skipped") or result.get("success") is False:
        return {"outcome": "not_sent", "reason": _NOT_SENT}
    if result.get("success"):
        return {"outcome": "sent", "reason": None}
    return {"outcome": "unknown", "reason": _UNKNOWN}


def _send_test(request_id: str) -> Tuple[int, Dict[str, Any]]:
    """``(status, answer)``: an answer given before, else 409 while paused, else 429 within 30
    seconds of the last test that went out, else the send to the Telegram home channel."""
    global _last_test_sent
    with _test_lock:
        if request_id in _test_answers:
            return 200, dict(_test_answers[request_id])
        if owner_channel_pause.is_paused():
            return 409, {"error": "paused"}
        now = time.monotonic()
        if _last_test_sent is not None and now - _last_test_sent < _TEST_INTERVAL_SECONDS:
            return 429, {"error": "too_soon"}
        if not owner_channel_pause.has_owner_home_chat():
            answer = {"outcome": "not_sent", "reason": _NO_HOME_CHANNEL}
        else:
            from tools.send_message_tool import send_message_tool

            _last_test_sent = now
            try:
                answer = _outcome(send_message_tool({"action": "send", "target": "telegram", "message": TEST_TEXT}))
            except Exception:  # raised mid-send: whether the message left is not known
                answer = {"outcome": "unknown", "reason": _UNKNOWN}
        _test_answers[request_id] = answer
        return 200, dict(answer)


@router.get(_ROUTE)
async def get_owner_channel(request: Request):
    """Whether a Telegram home channel is set, and whether and since when owner messages are paused."""
    _require_connections_credential(request)
    result = await asyncio.to_thread(_channel)
    _audit_connections_machine_success(request)
    return result


@router.put(_PAUSED_ROUTE)
async def set_owner_channel_paused(request: Request):
    """``{"paused": true}`` pauses owner messages and ``{"paused": false}`` resumes them; any
    other body is refused and changes nothing. Answers the same body as the GET."""
    _require_connections_credential(request)
    body = await _json_body(request)
    if not (isinstance(body, dict) and set(body) == {"paused"} and isinstance(body["paused"], bool)):
        raise HTTPException(status_code=400, detail='Send {"paused": true} or {"paused": false}.')

    def _set() -> Dict[str, Any]:
        (owner_channel_pause.pause if body["paused"] else owner_channel_pause.resume)()
        return _channel()

    result = await asyncio.to_thread(_set)
    _audit_connections_machine_success(request)
    return result


@router.post(_TEST_ROUTE)
async def send_owner_channel_test(request: Request):
    """Send the fixed test message to the Telegram home channel at most once per request_id."""
    _require_connections_credential(request)
    request_id = _request_id(await _json_body(request))
    if request_id is None:
        raise HTTPException(status_code=400, detail='Send {"request_id": "<uuid>"}.')
    status, answer = await asyncio.to_thread(_send_test, request_id)
    if status != 200:
        return JSONResponse(status_code=status, content=answer)
    _audit_connections_machine_success(request)
    return answer
