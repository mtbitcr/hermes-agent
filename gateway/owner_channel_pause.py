"""Pause of the owner's Telegram messages.

The owner Telegram home chat is the chat the gateway configuration names as the Telegram home
channel. While paused, nothing the platform starts on its own reaches that chat: the kanban
notifier skips its visible card notice there the way a wake-only subscription skips it (the
cursor advances and no failure is counted, so no notice is kept and nothing old is sent after the
resume), and ``cron.scheduler._deliver_result`` refuses a scheduled report, or its Send again, to
that chat ``route_refused`` before any transport. Replies inside a chat session are never paused,
and every other chat is served as before.

The state is one small file in the root Hermes home, read on every send, so every profile sees
the same pause without a restart. A missing, unreadable or malformed file counts as not paused.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from utils import atomic_json_write

PAUSE_FILE_NAME = "owner_channel_pause.json"


def _pause_file() -> Path:
    from hermes_constants import get_default_hermes_root

    return get_default_hermes_root() / PAUSE_FILE_NAME


def is_paused() -> bool:
    """True only while the pause file says paused. Never raises."""
    try:
        state = json.loads(_pause_file().read_text(encoding="utf-8"))
    except Exception:
        return False
    return isinstance(state, dict) and state.get("paused") is True


def pause() -> None:
    """Pause the owner's Telegram messages (temporary file, then rename)."""
    atomic_json_write(_pause_file(), {"paused": True, "paused_since": datetime.now(timezone.utc).isoformat()})


def resume() -> None:
    """Resume the owner's Telegram messages. Nothing was kept while paused, so nothing old is sent."""
    atomic_json_write(_pause_file(), {"paused": False})


def is_owner_home_chat(config: Any, platform: Any, chat_id: Any) -> bool:
    """True when ``platform`` is Telegram and ``chat_id`` is the chat ``config``, the gateway
    configuration the caller holds, names as the Telegram home channel (any topic of it). Never
    raises."""
    from gateway.config import Platform

    try:
        if str(getattr(platform, "value", platform)).lower() != Platform.TELEGRAM.value:
            return False
        home = config.get_home_channel(Platform.TELEGRAM)
        return home is not None and bool(home.chat_id) and str(home.chat_id) == str(chat_id)
    except Exception:
        return False
