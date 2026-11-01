"""Pause of the owner's Telegram messages.

The owner Telegram home chat is the chat the root profile names as its Telegram home channel:
``TELEGRAM_HOME_CHANNEL`` in the root Hermes home's .env, else ``platforms.telegram.home_channel``
in its config.yaml. Those two files alone decide it, whichever profile's job, worker or Send again
asks: the process environment and every other profile's files are never read, so a worker's own
Telegram settings cannot move it. A gateway launched under another profile is not covered. While
paused, nothing the platform starts on its own reaches that chat: the kanban notifier skips its
visible card notice there the way a wake-only subscription skips it (the cursor advances and no
failure is counted, so no notice is kept), and ``cron.scheduler._deliver_result`` refuses a
scheduled report, or its Send again, to that chat ``route_refused`` before the live transport and
again before any standalone one. Replies inside a chat session are never paused, and every other
chat is served as before.

The state is one small file in the root Hermes home, read on every send, so every profile sees
the same pause without a restart. It keeps every pause window of the last 7 days as ``[start,
end]`` in whole seconds, in time order, ``end`` null while the window is open. A pause opens a
window and a resume closes it; a closed window never changes, and one that ended more than 7 days
ago is dropped at the next write. Paused means the last window is open. Card events carry whole
seconds, and a window holds those created at ``start`` up to but not including ``end``; the resume
waits for the next whole second to end its window, so a notice created once the pause returns is
inside it and one created once the resume returns is outside. A notice created inside any window
is dropped even when the notifier first sees it after the resume: nothing old is sent. A missing,
unreadable or malformed file has no windows and counts as not paused.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import yaml
from dotenv import dotenv_values

from utils import atomic_json_write

PAUSE_FILE_NAME = "owner_channel_pause.json"
KEEP_SECONDS = 7 * 24 * 3600  # a window that ended longer ago is dropped at the next write


def _pause_file() -> Path:
    from hermes_constants import get_default_hermes_root

    return get_default_hermes_root() / PAUSE_FILE_NAME


def _windows() -> list:
    """The pause windows in the file; none when it is missing, unreadable or malformed."""
    try:
        windows = json.loads(_pause_file().read_text(encoding="utf-8"))["windows"]
        if isinstance(windows, list) and all(
            isinstance(start, int) and (end is None or isinstance(end, int)) for start, end in windows
        ):
            return windows
    except Exception:
        pass
    return []


def _is_open(windows: list) -> bool:
    return bool(windows) and windows[-1][1] is None


def _write(windows: list) -> None:
    """Write ``windows`` (temporary file, then rename), dropping those that ended over 7 days ago."""
    oldest = int(time.time()) - KEEP_SECONDS
    atomic_json_write(_pause_file(), {"windows": [w for w in windows if w[1] is None or w[1] >= oldest]})


def is_paused() -> bool:
    """True exactly while the last pause window is open. Never raises."""
    return _is_open(_windows())


def pause() -> None:
    """Pause the owner's Telegram messages: open a window. A repeat while paused changes nothing."""
    windows = _windows()
    if not _is_open(windows):
        _write(windows + [[int(time.time()), None]])


def resume() -> None:
    """Resume the owner's Telegram messages: close the open window at the next whole second, after
    waiting for it, so a notice created before the resume returns is still dropped and one created
    after it is sent. Not paused, it changes nothing."""
    windows = _windows()
    if not _is_open(windows):
        return
    end = int(time.time()) + 1
    time.sleep(max(0.0, end - time.time()))
    windows[-1][1] = end
    _write(windows)


def _owner_home_chat_id() -> str:
    """The chat the root profile names as its Telegram home: the root .env's
    ``TELEGRAM_HOME_CHANNEL`` (empty counts as none), else the root config.yaml's; "" if neither."""
    from hermes_constants import get_default_hermes_root

    root = get_default_hermes_root()
    try:
        env = dotenv_values(root / ".env", interpolate=False, encoding="utf-8-sig")
        home = env.get("TELEGRAM_HOME_CHANNEL")
    except Exception:
        home = None
    if not home:
        try:
            config = yaml.safe_load((root / "config.yaml").read_text(encoding="utf-8"))
            home = config["platforms"]["telegram"]["home_channel"]["chat_id"]
        except Exception:
            home = None
    return str(home) if home else ""


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
    ``chat_id``: the owner home chat, while paused or for an event created inside a pause window."""
    windows = _windows()
    try:
        held = _is_open(windows) or any(
            start <= created_at and (end is None or created_at < end) for start, end in windows
        )
    except Exception:
        held = False
    return held and is_owner_home_chat(platform, chat_id)
