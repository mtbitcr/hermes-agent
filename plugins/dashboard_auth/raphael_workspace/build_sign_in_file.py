"""The one protected file build machines read their Anthropic sign-in from first.

A Claude Code setup token lasts a year and is never refreshed, which suits
build machines (see ``sandbox_dispatch._resolve_host_credential``). It is kept
in one file for the whole machine, ``secrets/build-sign-in.json`` in the root
Hermes home, so a new token takes over for the next build just by replacing
that file. The swap never cuts off a running build: its machine keeps the copy
in its own credential vault until its next lease renewal, which hands it
whichever token is current by then.

The file holds the JSON object ``{"token": ..., "made_at": ...}``, where
``made_at`` is the ISO 8601 time the token was made, with its UTC offset. It is
not encrypted: like the settings file that holds the token today, it is
protected by being the service user's own at mode 0600. Anything else — a
symlink, another mode or owner, content that does not parse — reads as no file
at all, so builds keep today's sources. The token never goes into a log record
or an exception.
"""
from __future__ import annotations

import json
import os
import stat
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

from hermes_constants import get_default_hermes_root

#: A Claude Code setup token works for one year from when it was made.
_SETUP_TOKEN_LIFETIME = timedelta(days=365)

_FILE_NAME = "build-sign-in.json"
#: Far above any real sign-in file; anything larger is not one.
_MAX_FILE_BYTES = 64 * 1024


def _file_path() -> Path:
    # One file for the whole machine: a worker runs with a profile HERMES_HOME.
    return get_default_hermes_root() / "secrets" / _FILE_NAME


def _current_uid() -> Optional[int]:
    """The service user's id, or ``None`` where files carry no POSIX owner."""
    geteuid = getattr(os, "geteuid", None)
    return geteuid() if geteuid is not None else None


def write_build_token(token: str) -> None:
    """Make *token* the build sign-in, replacing any earlier one in one step."""
    token = token.strip() if isinstance(token, str) else ""
    if not token:
        raise ValueError("a build sign-in token must be non-empty text")
    path = _file_path()
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    record = {"token": token, "made_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    tmp_path = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    fd = os.open(tmp_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(record, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
    finally:
        tmp_path.unlink(missing_ok=True)


def read_build_token() -> Optional[tuple[str, datetime]]:
    """The build sign-in token and when it runs out, or ``None`` if none can be trusted."""
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW  # a symlink fails to open instead of being followed
    if hasattr(os, "O_NOCTTY"):
        flags |= os.O_NOCTTY
    try:
        fd = os.open(_file_path(), flags)
        with os.fdopen(fd, "rb") as handle:
            info = os.fstat(handle.fileno())
            if (
                not stat.S_ISREG(info.st_mode)
                or stat.S_IMODE(info.st_mode) != 0o600
                or info.st_uid != _current_uid()
            ):
                return None
            raw = handle.read(_MAX_FILE_BYTES + 1)
    except OSError:
        return None
    if len(raw) > _MAX_FILE_BYTES:
        return None
    try:
        record = json.loads(raw.decode("utf-8"))
    except (ValueError, RecursionError):
        return None
    if not isinstance(record, dict):
        return None
    token, made_at = record.get("token"), record.get("made_at")
    if not isinstance(token, str) or not token.strip() or not isinstance(made_at, str):
        return None
    try:
        made = datetime.fromisoformat(made_at)
        if made.tzinfo is None:
            return None
        return token.strip(), made.astimezone(timezone.utc) + _SETUP_TOKEN_LIFETIME
    except (ValueError, OverflowError):
        return None
