"""Build sign-in routes: the pasted Claude Code setup token and the Claude usage left.

``PUT /api/providers/build-sign-in`` saves a pasted setup token (made yearly with
``claude setup-token``) as the one machine-wide build sign-in, ``GET`` on the
same path says whether one is in place and how long it still works, and
``GET /api/providers/usage/anthropic`` says how much of each Claude usage
window is left. The sign-in file and the account are the whole machine's, so
``profile`` only binds a Models machine call to the profiles it may name. The
usage is read live at most once a minute. The token never goes into an answer,
a log record or an audit entry.
"""

import asyncio
import json
import logging
import math
import re
import threading
import time
from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import APIRouter, HTTPException, Request

from hermes_cli.web_deps import late

_log = logging.getLogger("hermes_cli.web_server")

router = APIRouter()

# Late-bound web_server helper (resolved at call time; cycle-safe,
# monkeypatch-transparent).
_require_token = late("_require_token")

#: What ``claude setup-token`` prints: 40 to 300 letters, digits, ``-`` and ``_``.
_TOKEN_SHAPE = re.compile(r"[A-Za-z0-9_-]{40,300}")
_NOT_A_SETUP_TOKEN = "That is not a Claude Code setup token, so nothing was saved."
_NOT_SAVED = "The token could not be saved."
_USAGE_UNAVAILABLE = "Claude usage could not be read right now."

_USAGE_CACHE_SECONDS = 60.0
_usage_lock = threading.Lock()
_usage_cache: Optional[tuple] = None  # (monotonic_ts, answer)


def _pasted_token(raw: bytes) -> Optional[str]:
    """The setup token in a ``{"token": ...}`` body, or ``None`` for anything else."""
    from agent.anthropic_credentials import _is_oauth_token

    try:
        body = json.loads(raw)
    except (ValueError, RecursionError):
        return None
    token = body.get("token") if isinstance(body, dict) else None
    if not isinstance(token, str) or not _TOKEN_SHAPE.fullmatch(token) or not _is_oauth_token(token):
        return None
    return token


def _dates(works_until: datetime) -> dict:
    """When the saved token was made and when it stops working, in UTC."""
    from plugins.dashboard_auth.raphael_workspace.build_sign_in_file import _SETUP_TOKEN_LIFETIME

    return {
        "made_on": (works_until - _SETUP_TOKEN_LIFETIME).isoformat(),
        "works_until": works_until.isoformat(),
    }


@router.put("/api/providers/build-sign-in")
async def save_build_sign_in(request: Request, profile: Optional[str] = None):
    """Save a pasted setup token as the build sign-in; answer when it was made and runs out."""
    _require_token(request)
    from plugins.dashboard_auth.raphael_workspace.model_policy import (
        audit_models_machine_success,
        require_models_machine_profile,
    )

    scoped_profile = require_models_machine_profile(request, profile, allowed=(None, "default"))
    # Read by hand: the default 422 answer would echo the value back.
    token = _pasted_token(await request.body())
    if token is None:
        raise HTTPException(status_code=422, detail=_NOT_A_SETUP_TOKEN)

    from plugins.dashboard_auth.raphael_workspace import build_sign_in_file

    try:
        await asyncio.to_thread(build_sign_in_file.write_build_token, token)
    except (OSError, ValueError) as exc:
        # The error text can name the token; only its kind is logged.
        _log.warning("build sign-in: saving failed (%s)", type(exc).__name__)
        raise HTTPException(status_code=500, detail=_NOT_SAVED) from None
    saved = await asyncio.to_thread(build_sign_in_file.read_build_token)
    if saved is None:
        _log.warning("build sign-in: the saved file does not read back")
        raise HTTPException(status_code=500, detail=_NOT_SAVED)

    audit_models_machine_success(request, action="save", profile=scoped_profile, provider="anthropic")
    return {"state": "saved", **_dates(saved[1])}


@router.get("/api/providers/build-sign-in")
async def build_sign_in_status(request: Request, profile: Optional[str] = None):
    """Whether a build sign-in is in place (``none``, ``active`` or ``too_old``), never the token."""
    _require_token(request)
    from plugins.dashboard_auth.raphael_workspace import build_sign_in_file
    from plugins.dashboard_auth.raphael_workspace.model_policy import (
        audit_models_machine_success,
        require_models_machine_profile,
    )

    scoped_profile = require_models_machine_profile(request, profile, allowed=(None, "default"))
    saved = await asyncio.to_thread(build_sign_in_file.read_build_token)
    if saved is None:
        result = {"state": "none"}
    else:
        works_until = saved[1]
        remaining = works_until - datetime.now(timezone.utc)
        if remaining <= timedelta(0):
            result = {"state": "too_old"}
        else:
            days_left = math.ceil(remaining / timedelta(days=1))
            result = {"state": "active", **_dates(works_until), "days_left": days_left}

    audit_models_machine_success(request, action="status", profile=scoped_profile, provider="anthropic")
    return result


def _percent_left(used) -> Optional[float]:
    if isinstance(used, bool) or not isinstance(used, (int, float)) or not math.isfinite(used):
        return None
    return max(0, min(100, 100 - used))


def _resets_at(when) -> Optional[str]:
    if not isinstance(when, datetime):
        return None
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    return when.astimezone(timezone.utc).isoformat()


def _read_usage() -> dict:
    """One live read of the Claude account usage, as the answer to give."""
    from agent.account_usage import fetch_account_usage

    try:
        snapshot = fetch_account_usage("anthropic")
    except Exception as exc:
        # The error text can name the token; only its kind is logged.
        _log.warning("usage/anthropic: read failed (%s)", type(exc).__name__)
        snapshot = None
    if snapshot is None or not snapshot.available:
        reason = getattr(snapshot, "unavailable_reason", None)
        if not isinstance(reason, str) or not reason.strip():
            reason = _USAGE_UNAVAILABLE
        return {"state": "unavailable", "reason": reason.strip()}
    return {
        "state": "available",
        "windows": [
            {
                "label": str(window.label),
                "percent_left": _percent_left(window.used_percent),
                "resets_at": _resets_at(window.reset_at),
            }
            for window in snapshot.windows
        ],
        "details": [str(line) for line in snapshot.details],
    }


def _cached_usage() -> dict:
    """The usage answer, read live at most once per ``_USAGE_CACHE_SECONDS``."""
    global _usage_cache
    with _usage_lock:
        if _usage_cache is not None and time.monotonic() - _usage_cache[0] < _USAGE_CACHE_SECONDS:
            return _usage_cache[1]
        answer = _read_usage()
        _usage_cache = (time.monotonic(), answer)
        return answer


@router.get("/api/providers/usage/anthropic")
async def anthropic_usage(request: Request, profile: Optional[str] = None):
    """How much of each Claude usage window is left, or ``unavailable`` with a reason."""
    _require_token(request)
    from plugins.dashboard_auth.raphael_workspace.model_policy import (
        audit_models_machine_success,
        require_models_machine_profile,
    )

    scoped_profile = require_models_machine_profile(request, profile, allowed=(None, "default"))
    result = await asyncio.to_thread(_cached_usage)
    audit_models_machine_success(request, action="usage", profile=scoped_profile, provider="anthropic")
    return result
