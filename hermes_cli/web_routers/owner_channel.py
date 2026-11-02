"""Owner channel: the Telegram row of the owner app.

``GET /api/owner-channel`` answers whether the root profile names a Telegram
home channel and whether owner messages are paused, ``PUT
/api/owner-channel/paused`` pauses or resumes them through
``gateway.owner_channel_pause``, and ``POST /api/owner-channel/test`` sends one
fixed test message to that home channel through
``tools.send_message_tool.send_message_tool``, the path ``hermes send --to
telegram`` takes. They answer the owner app's connections credential and no
other caller, and every call writes one audit row, a call that fails included.
An answer never holds a token, chat id, user name or provider text.

They serve only in the default profile's process, which sends with its own
configuration and credentials, the root profile's. In any other profile's
process each route answers 503 before it reads the cooldown, writes a pause or
sends (owner rule 1 of 2026-11-02).

The test route reads no body, so a request_id in it is ignored. Its one
duplicate guard is one cooldown for the whole owner channel: under an exclusive
lock on ``owner_channel_test.lock`` in the root Hermes home it reads the time of
the last test from ``owner_channel_test.json`` there and answers 429 when less
than 30 seconds have passed, a clock that moved back included; otherwise it
keeps the current time and releases the lock, sends, and keeps the completion
time under the lock again, whatever the outcome. So the cooldown holds across
dashboard processes and restarts, also for a process that dies during its send.
A retry after a lost answer gets 429 within the 30 seconds, and after them it
may send the fixed text again, which the owner accepts.
"""

import asyncio
import json
import math
import os
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, Optional, Tuple

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse

from gateway import owner_channel_pause
from hermes_cli.dashboard_auth.token_auth import register_machine_token_family, register_token_route
from hermes_cli.web_routers.connection_rows import _require_connections_credential
from hermes_cli.web_routers.mcp import _audit_connections_machine_success
from plugins.dashboard_auth.raphael_workspace import CONNECTIONS_SCOPE, CONNECTIONS_TOKEN_PREFIX
from utils import atomic_json_write

router = APIRouter()

_ROUTE = "/api/owner-channel"
_PAUSED_ROUTE = "/api/owner-channel/paused"
_TEST_ROUTE = "/api/owner-channel/test"

TEST_TEXT = "Test from Raphael: your message channel works."
_TEST_INTERVAL_SECONDS = 30
_NO_HOME_CHANNEL = "No Telegram home channel is set."
_NOT_SENT = "The test message could not be sent."
_UNKNOWN = "It is not known whether the test message was sent."

_STATE_FILE_NAME = "owner_channel_test.json"
_LOCK_FILE_NAME = "owner_channel_test.lock"

#: The lock file serializes the test route across processes; this lock, across this process's threads.
_test_lock = threading.Lock()


def _register_owner_channel_routes() -> None:
    """Token-only (no session), and only for a token carrying the connections scope."""
    register_machine_token_family(CONNECTIONS_TOKEN_PREFIX, strict_audit=True)
    register_token_route(_ROUTE, method="GET", required_scope=CONNECTIONS_SCOPE, strict_audit=True)
    register_token_route(_PAUSED_ROUTE, method="PUT", required_scope=CONNECTIONS_SCOPE, strict_audit=True)
    register_token_route(_TEST_ROUTE, method="POST", required_scope=CONNECTIONS_SCOPE, strict_audit=True)


_register_owner_channel_routes()


def _now() -> float:
    """The test route's clock. Wall-clock seconds, since the time of the last test outlives this process."""
    return time.time()


async def _json_body(request: Request) -> Any:
    """The parsed body; None when it is not JSON, which the pause route refuses as well."""
    try:
        return await request.json()
    except ValueError:
        return None


def _failed() -> JSONResponse:
    """The answer to a call whose work raised. Its body is fixed, so no exception text leaves, and the
    token seam audits a 500 without the success mark as this call's one failure row."""
    return JSONResponse(status_code=500, content={"error": "failed"})


def _serves() -> bool:
    """True in the default profile's process, the only one these routes serve in."""
    from hermes_cli.profiles import get_active_profile_name

    return get_active_profile_name() == "default"


def _unavailable() -> JSONResponse:
    """The answer in any other profile's process; like ``_failed``, the token seam audits it as the
    call's one failure row."""
    return JSONResponse(status_code=503, content={"error": "unavailable"})


def _channel() -> Dict[str, Any]:
    """The row: whether a home channel is set and since when owner messages are paused, nothing else."""
    since = owner_channel_pause.paused_since()
    return {
        "state": "ready" if owner_channel_pause.has_owner_home_chat() else "not_set",
        "paused": since is not None,
        "paused_since": since,
    }


def _set_paused(paused: bool) -> Dict[str, Any]:
    (owner_channel_pause.pause if paused else owner_channel_pause.resume)()
    return _channel()


@contextmanager
def _locked_test_state() -> Iterator[Path]:
    """The test state file, held against every other thread and process until the block ends."""
    from hermes_constants import get_default_hermes_root

    root = get_default_hermes_root()
    with _test_lock:
        root.mkdir(parents=True, exist_ok=True)
        flags = os.O_RDWR | os.O_CREAT
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        fd = os.open(root / _LOCK_FILE_NAME, flags, 0o600)
        try:
            if os.name == "nt":
                import msvcrt

                if os.fstat(fd).st_size == 0:
                    os.write(fd, b" ")
                os.lseek(fd, 0, os.SEEK_SET)
                msvcrt.locking(fd, msvcrt.LK_LOCK, 1)
            else:
                import fcntl

                fcntl.flock(fd, fcntl.LOCK_EX)
            try:
                yield root / _STATE_FILE_NAME
            finally:
                if os.name == "nt":
                    os.lseek(fd, 0, os.SEEK_SET)
                    msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
                else:
                    fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)


def _is_seconds(value: Any) -> bool:
    return type(value) in (int, float) and math.isfinite(value)


def _read_last_test_at(path: Path) -> Optional[float]:
    """The time of the last test that may have gone out; None when the file is missing. Raises for a
    file it cannot read or that holds no such time, so the route sends nothing rather than forget the
    cooldown."""
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    state = json.loads(text)
    if not (isinstance(state, dict) and _is_seconds(state.get("last_test_at"))):
        raise ValueError("malformed owner channel test state")
    return state["last_test_at"]


def _write_last_test_at(path: Path, last_test_at: float) -> None:
    """Keep the time of the last test, and nothing else (temporary file, then rename)."""
    atomic_json_write(path, {"last_test_at": last_test_at})


def _outcome(raw: Any) -> Dict[str, Optional[str]]:
    """The send tool's answer as a fixed outcome, read from its structured fields only, never from its
    error text or note. sent needs success that was not skipped; not_sent needs a skip, which the tool
    answers before any send; anything else is unknown, an error included, since the tool also turns a
    failure after the message may have been delivered, such as a lost reply, into an error."""
    result = raw
    if isinstance(raw, (str, bytes)):
        try:
            result = json.loads(raw)
        except ValueError:
            result = None
    if not isinstance(result, dict):
        return {"outcome": "unknown", "reason": _UNKNOWN}
    skipped = result.get("skipped")
    if skipped is True:
        return {"outcome": "not_sent", "reason": _NOT_SENT}
    if result.get("success") is True and not skipped:
        return {"outcome": "sent", "reason": None}
    return {"outcome": "unknown", "reason": _UNKNOWN}


def _send_to_root_home() -> Dict[str, Optional[str]]:
    """Send the test text to the root profile's Telegram home with this default profile process's own
    configuration and credentials, the root profile's. not_sent before any send when no home channel
    is set; a send that raised is unknown."""
    from tools.send_message_tool import send_message_tool

    # Stripped, as a blank chat id would make the send tool fall back to an ambient home.
    chat_id = owner_channel_pause._owner_home_chat_id().strip()
    if not chat_id:
        return {"outcome": "not_sent", "reason": _NO_HOME_CHANNEL}
    try:
        raw = send_message_tool({"action": "send", "target": f"telegram:{chat_id}", "message": TEST_TEXT})
    except Exception:
        return {"outcome": "unknown", "reason": _UNKNOWN}
    return _outcome(raw)


def _send_test() -> Tuple[int, Dict[str, Any]]:
    """``(status, answer)``: 409 while paused, else 429 within 30 seconds of the last test, else the
    send. The last test time is read, and the start time kept, under the lock; the completion time is
    kept under it again after the send, whatever the outcome."""
    with _locked_test_state() as path:
        if owner_channel_pause.is_paused():
            return 409, {"error": "paused"}
        last_test_at, now = _read_last_test_at(path), _now()
        if last_test_at is not None and now - last_test_at < _TEST_INTERVAL_SECONDS:
            return 429, {"error": "too_soon"}  # a negative elapsed time (the clock moved back) as well
        _write_last_test_at(path, now)
    answer = _send_to_root_home()
    with _locked_test_state() as path:
        _write_last_test_at(path, _now())
    return 200, answer


@router.get(_ROUTE)
async def get_owner_channel(request: Request):
    """Whether a Telegram home channel is set, and whether and since when owner messages are paused."""
    _require_connections_credential(request)
    try:
        if not _serves():
            return _unavailable()
        result = await asyncio.to_thread(_channel)
    except Exception:
        return _failed()
    _audit_connections_machine_success(request)
    return result


@router.put(_PAUSED_ROUTE)
async def set_owner_channel_paused(request: Request):
    """``{"paused": true}`` pauses owner messages and ``{"paused": false}`` resumes them; any
    other body is refused and changes nothing. Answers the same body as the GET."""
    _require_connections_credential(request)
    try:
        if not _serves():
            return _unavailable()
        body = await _json_body(request)
        if not (isinstance(body, dict) and set(body) == {"paused"} and isinstance(body["paused"], bool)):
            raise HTTPException(status_code=400, detail='Send {"paused": true} or {"paused": false}.')
        result = await asyncio.to_thread(_set_paused, body["paused"])
    except HTTPException:
        raise
    except Exception:
        return _failed()
    _audit_connections_machine_success(request)
    return result


@router.post(_TEST_ROUTE)
async def send_owner_channel_test(request: Request):
    """Send the fixed test message to the Telegram home channel, at most once in 30 seconds. The body
    is not read: a request_id in it is ignored, and an empty body is valid."""
    _require_connections_credential(request)
    try:
        if not _serves():
            return _unavailable()
        status, answer = await asyncio.to_thread(_send_test)
    except Exception:
        return _failed()
    if status != 200:
        return JSONResponse(status_code=status, content=answer)
    _audit_connections_machine_success(request)
    return answer
