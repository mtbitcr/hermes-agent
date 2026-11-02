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

The test route keeps each answer it gives and the time of the last test that
may have gone out in ``owner_channel_test.json`` in the root Hermes home, and
holds an exclusive lock on ``owner_channel_test.lock`` there from the
request_id lookup to the kept answer, the send included. So one request_id
sends at most once and is answered the same way again, and the 30-second
interval holds, across dashboard processes and restarts.
"""

import asyncio
import json
import math
import os
import threading
import time
import uuid
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
_KEEP_SECONDS = 7 * 24 * 3600  # an answer kept longer ago is dropped at the next write

#: Every answer the test route keeps, as ``(status, body)``. A reservation is kept as
#: ``(None, unknown)``: the answer it gets when its process dies before the send returns.
_ANSWERS = (
    (200, {"outcome": "sent", "reason": None}),
    (200, {"outcome": "not_sent", "reason": _NO_HOME_CHANNEL}),
    (200, {"outcome": "not_sent", "reason": _NOT_SENT}),
    (200, {"outcome": "unknown", "reason": _UNKNOWN}),
    (409, {"error": "paused"}),
    (429, {"error": "too_soon"}),
    (None, {"outcome": "unknown", "reason": _UNKNOWN}),
)

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
    """The parsed body; None when it is not JSON, which every route refuses as well."""
    try:
        return await request.json()
    except ValueError:
        return None


def _failed() -> JSONResponse:
    """The answer to a call whose work raised. Its body is fixed, so no exception text leaves, and the
    token seam audits a 500 without the success mark as this call's one failure row."""
    return JSONResponse(status_code=500, content={"error": "failed"})


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


def _request_id(body: Any) -> Optional[str]:
    """The canonical form of the body's request_id; None for any body but ``{"request_id": "<uuid>"}``."""
    if not (isinstance(body, dict) and set(body) == {"request_id"} and isinstance(body["request_id"], str)):
        return None
    try:
        return str(uuid.UUID(body["request_id"]))
    except ValueError:
        return None


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


def _is_kept_answer(entry: Any) -> bool:
    return (
        isinstance(entry, dict)
        and set(entry) == {"status", "body", "at"}
        and type(entry["status"]) in (int, type(None))
        and (entry["status"], entry["body"]) in _ANSWERS
        and _is_seconds(entry["at"])
    )


def _read_test_state(path: Path) -> Tuple[Dict[str, Dict[str, Any]], Optional[float]]:
    """The kept answers by request_id and the time of the last test that may have gone out; neither
    when the file is missing. Raises for a file it cannot read or that this module did not write,
    so the route sends nothing rather than forget an answer or the interval."""
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return {}, None
    state = json.loads(text)
    if not (
        isinstance(state, dict)
        and set(state) == {"answers", "last_test_at"}
        and isinstance(state["answers"], dict)
        and all(_is_kept_answer(entry) for entry in state["answers"].values())
        and (state["last_test_at"] is None or _is_seconds(state["last_test_at"]))
    ):
        raise ValueError("malformed owner channel test state")
    return state["answers"], state["last_test_at"]


def _write_test_state(
    path: Path, answers: Dict[str, Dict[str, Any]], last_test_at: Optional[float], now: float
) -> None:
    """Keep the answers and the time of the last test (temporary file, then rename), dropping the
    answers kept over 7 days ago."""
    oldest = now - _KEEP_SECONDS
    kept = {request_id: entry for request_id, entry in answers.items() if entry["at"] >= oldest}
    atomic_json_write(path, {"answers": kept, "last_test_at": last_test_at})


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


def _send_to_root_home(chat_id: str) -> Dict[str, Optional[str]]:
    """Send the test text to ``chat_id``, the root profile's home, as the root profile: the send tool
    reads the root config.yaml, and the root .env's bot token over the process environment's, which
    may be the serving profile's. A send that raised is unknown."""
    from agent.secret_scope import build_profile_secret_scope, reset_secret_scope, set_secret_scope
    from hermes_constants import get_default_hermes_root, reset_hermes_home_override, set_hermes_home_override
    from tools.send_message_tool import send_message_tool

    root = get_default_hermes_root()
    secrets = build_profile_secret_scope(root)
    home_token = set_hermes_home_override(root)
    scope_token = set_secret_scope(secrets)
    try:
        raw = send_message_tool({"action": "send", "target": f"telegram:{chat_id}", "message": TEST_TEXT})
    except Exception:
        return {"outcome": "unknown", "reason": _UNKNOWN}
    finally:
        reset_secret_scope(scope_token)
        reset_hermes_home_override(home_token)
    return _outcome(raw)


def _send_test(request_id: str) -> Tuple[int, Dict[str, Any]]:
    """``(status, answer)``, every step under the lock: the answer kept for this request_id, else 409
    while paused, else 429 within 30 seconds of the last test that may have gone out, else not_sent
    with no home channel, else the send. Each answer is kept, and a send is reserved before it starts."""
    with _locked_test_state() as path:
        answers, last_test_at = _read_test_state(path)
        now = _now()
        cut_off = [key for key, entry in answers.items() if entry["status"] is None]
        if cut_off:  # cut off before their answer was kept, by a crash say: the send may have gone out
            for key in cut_off:
                answers[key] = {"status": 200, "body": {"outcome": "unknown", "reason": _UNKNOWN}, "at": now}
            last_test_at = now
            _write_test_state(path, answers, last_test_at, now)
        if request_id in answers:
            return answers[request_id]["status"], dict(answers[request_id]["body"])

        if owner_channel_pause.is_paused():
            status, answer = 409, {"error": "paused"}
        elif last_test_at is not None and abs(now - last_test_at) < _TEST_INTERVAL_SECONDS:
            # A last test ahead of the clock (it stepped back) counts until it is 30 seconds ahead.
            status, answer = 429, {"error": "too_soon"}
        else:
            # Stripped, as a blank chat id would make the send tool fall back to an ambient home.
            chat_id = owner_channel_pause._owner_home_chat_id().strip()
            status = 200
            if not chat_id:
                answer = {"outcome": "not_sent", "reason": _NO_HOME_CHANNEL}
            else:
                answers[request_id] = {"status": None, "body": {"outcome": "unknown", "reason": _UNKNOWN}, "at": now}
                _write_test_state(path, answers, last_test_at, now)
                answer = _send_to_root_home(chat_id)
                now = _now()
                if answer["outcome"] != "not_sent":  # the interval counts from when a send returned
                    last_test_at = now
        answers[request_id] = {"status": status, "body": answer, "at": now}
        _write_test_state(path, answers, last_test_at, now)
        return status, dict(answer)


@router.get(_ROUTE)
async def get_owner_channel(request: Request):
    """Whether a Telegram home channel is set, and whether and since when owner messages are paused."""
    _require_connections_credential(request)
    try:
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
    """Send the fixed test message to the Telegram home channel at most once per request_id."""
    _require_connections_credential(request)
    try:
        request_id = _request_id(await _json_body(request))
        if request_id is None:
            raise HTTPException(status_code=400, detail='Send {"request_id": "<uuid>"}.')
        status, answer = await asyncio.to_thread(_send_test, request_id)
    except HTTPException:
        raise
    except Exception:
        return _failed()
    if status != 200:
        return JSONResponse(status_code=status, content=answer)
    _audit_connections_machine_success(request)
    return answer
