"""Pause of the owner's Telegram messages.

The owner Telegram home chat is the chat the serving gateway's configuration names as the Telegram
home channel: the root Hermes home's, read the way the gateway reads its own, whichever profile's
job, worker or Send again asks. While paused, nothing the platform starts on its own reaches that
chat: the kanban notifier skips its visible card notice there the way a wake-only subscription
skips it (the cursor advances and no failure is counted, so no notice is kept), and
``cron.scheduler._deliver_result`` refuses a scheduled report, or its Send again, to that chat
``route_refused`` before the live transport and again before any standalone one. Replies inside a
chat session are never paused, and every other chat is served as before.

The state is one small file in the root Hermes home, read on every send, so every profile sees
the same pause without a restart. A missing, unreadable or malformed file counts as not paused.
The resume keeps the pause's bounds in the file, so a notice created while paused is dropped even
when the notifier first sees it after the resume: nothing old is sent.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from utils import atomic_json_write

PAUSE_FILE_NAME = "owner_channel_pause.json"


def _pause_file() -> Path:
    from hermes_constants import get_default_hermes_root

    return get_default_hermes_root() / PAUSE_FILE_NAME


def _state() -> dict:
    try:
        state = json.loads(_pause_file().read_text(encoding="utf-8"))
    except Exception:
        return {}
    return state if isinstance(state, dict) else {}


def is_paused() -> bool:
    """True only while the pause file says paused. Never raises."""
    return _state().get("paused") is True


def pause() -> None:
    """Pause the owner's Telegram messages (temporary file, then rename); a repeat keeps the start."""
    if not is_paused():
        atomic_json_write(_pause_file(), {"paused": True, "paused_since": int(time.time())})


def resume() -> None:
    """Resume the owner's Telegram messages. Card events carry whole seconds, so the resume waits
    for the next one and keeps the pause's bounds: a notice created before it is still dropped,
    one created after it is sent. Nothing was kept while paused, so nothing old is sent."""
    state = _state()
    if state.get("paused") is not True:
        return
    resumed_at = int(time.time()) + 1
    time.sleep(max(0.0, resumed_at - time.time()))
    atomic_json_write(
        _pause_file(), {"paused": False, "paused_since": state.get("paused_since"), "resumed_at": resumed_at},
    )


def _owner_home_chat_id() -> str:
    from agent.secret_scope import build_profile_secret_scope, reset_secret_scope, set_secret_scope
    from gateway.config import Platform, load_gateway_config
    from hermes_constants import get_default_hermes_root, reset_hermes_home_override, set_hermes_home_override

    # The serving gateway's own settings and secrets, as its runner loads them (``_profile_runtime_scope``).
    root = get_default_hermes_root()
    home_token = set_hermes_home_override(str(root))
    scope_token = set_secret_scope(build_profile_secret_scope(root))
    try:
        home = load_gateway_config().get_home_channel(Platform.TELEGRAM)
    finally:
        reset_secret_scope(scope_token)
        reset_hermes_home_override(home_token)
    return str(home.chat_id) if home is not None and home.chat_id else ""


def is_owner_home_chat(platform: Any, chat_id: Any) -> bool:
    """True when ``platform`` is Telegram and ``chat_id`` is the owner Telegram home chat (any
    topic of it). Never raises."""
    try:
        if str(getattr(platform, "value", platform)).lower() != "telegram":
            return False
        owner_home = _owner_home_chat_id()
        return bool(owner_home) and owner_home == str(chat_id)
    except Exception:
        return False


def refuses_report(platform: Any, chat_id: Any) -> bool:
    """True when a scheduled report, or its Send again, to ``chat_id`` must be refused now."""
    return is_paused() and is_owner_home_chat(platform, chat_id)


def drops_notice(platform: Any, chat_id: Any, created_at: Any) -> bool:
    """True when the visible card notice of an event created at ``created_at`` must not reach
    ``chat_id``: the owner home chat, while paused or for an event created while paused."""
    state = _state()
    try:
        held = state.get("paused") is True or state["paused_since"] <= created_at < state["resumed_at"]
    except Exception:
        held = False
    return held and is_owner_home_chat(platform, chat_id)
