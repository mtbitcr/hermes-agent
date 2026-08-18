"""Audit log for dashboard-auth events.

Profile-aware location: ``$HERMES_HOME/logs/dashboard-auth.log``.
Format: one JSON object per line. Token-like fields are stripped before
serialisation to avoid leaking refresh tokens or JWTs to disk.

This module deliberately keeps a minimal dependency surface — no imports
from ``hermes_constants`` or other hermes_cli modules — so it can be
imported safely from middleware code that loads early in the startup
sequence.
"""
from __future__ import annotations

import datetime as _dt
import enum
import json
import logging
import os
import stat
import threading
from pathlib import Path
from typing import Any

_log = logging.getLogger(__name__)
_write_lock = threading.Lock()

# Field names that must never appear in the log raw. Any kwarg matching
# these is silently dropped.
_REDACTED_FIELDS: frozenset = frozenset({
    "access_token", "refresh_token", "code", "code_verifier",
    "state", "ticket", "cookie", "Authorization", "authorization",
})


class AuditEvent(enum.Enum):
    """Event types written to dashboard-auth.log.

    Values are the literal ``event`` field on the JSON line.
    """

    LOGIN_START = "login_start"
    LOGIN_SUCCESS = "login_success"
    LOGIN_FAILURE = "login_failure"
    LOGOUT = "logout"
    REFRESH_SUCCESS = "refresh_success"
    REFRESH_FAILURE = "refresh_failure"
    REVOKE = "revoke"
    SESSION_VERIFY_FAILURE = "session_verify_failure"
    WS_TICKET_MINTED = "ws_ticket_minted"
    WS_TICKET_REJECTED = "ws_ticket_rejected"
    TOKEN_AUTH_SUCCESS = "token_auth_success"
    TOKEN_AUTH_FAILURE = "token_auth_failure"
    MACHINE_TOKEN_ISSUED = "machine_token_issued"
    MACHINE_TOKEN_REVOKED = "machine_token_revoked"
    # RFC 8252 native-app (system-browser + loopback + PKCE) flow.
    NATIVE_AUTHORIZE_START = "native_authorize_start"
    NATIVE_CODE_ISSUED = "native_code_issued"
    NATIVE_TOKEN_SUCCESS = "native_token_success"
    NATIVE_TOKEN_FAILURE = "native_token_failure"


class AuditWriteError(Exception):
    """Raised by ``audit_log(..., strict=True)`` when the write failed.

    Every existing call site omits ``strict`` and keeps the original
    never-raises behaviour (auth must not break because the audit logger
    broke). ``strict=True`` is for a caller whose OWN contract is the
    opposite — Item 32G-A's machine-credential routes must not serve data
    they cannot durably audit — so it needs to observe the failure instead of
    having it silently swallowed.
    """


def _resolve_log_path() -> Path:
    """``$HERMES_HOME/logs/dashboard-auth.log``.

    Uses ``hermes_constants.get_hermes_home()`` (a leaf module — no import
    cycle) so profile overrides and the native-Windows ``%LOCALAPPDATA%``
    fallback are honored.
    """
    from hermes_constants import get_hermes_home

    return get_hermes_home() / "logs" / "dashboard-auth.log"


def audit_log(event: AuditEvent, *, strict: bool = False, **fields: Any) -> None:
    """Append one event to the audit log.

    Token-like fields are dropped. Missing log directory is created.

    Write failures are logged at WARNING. With the default ``strict=False``
    they never raise — auth must not fail because the audit logger broke, and
    every existing call site (login/session/token-seam middleware) relies on
    that. Pass ``strict=True`` when the CALLER's own contract requires the
    opposite — e.g. a machine-credential route that must not serve data it
    cannot durably audit — in which case a write failure raises
    :class:`AuditWriteError` after the same WARNING log line.
    """
    safe_fields = {
        k: v for k, v in fields.items()
        if k not in _REDACTED_FIELDS
    }
    entry = {
        "ts": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "event": event.value,
        **safe_fields,
    }
    encoded = (json.dumps(entry, separators=(",", ":")) + "\n").encode("utf-8")
    path = _resolve_log_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with _write_lock:
            if not strict:
                with open(path, "a", encoding="utf-8") as handle:
                    handle.write(encoded.decode("utf-8"))
                return

            parent_stat = path.parent.lstat()
            if stat.S_ISLNK(parent_stat.st_mode) or not stat.S_ISDIR(
                parent_stat.st_mode
            ):
                raise OSError(
                    f"audit parent is not a regular directory: {path.parent}"
                )
            if os.name != "nt":
                if parent_stat.st_uid != getattr(os, "geteuid")():
                    raise OSError(
                        f"audit parent is not owned by the current user: {path.parent}"
                    )
                os.chmod(path.parent, stat.S_IRWXU)
            flags = os.O_WRONLY | os.O_APPEND | os.O_CREAT
            if hasattr(os, "O_NOFOLLOW"):
                flags |= os.O_NOFOLLOW
            fd = os.open(path, flags, stat.S_IRUSR | stat.S_IWUSR)
            try:
                st = os.fstat(fd)
                mode = stat.S_IMODE(st.st_mode)
                if not stat.S_ISREG(st.st_mode):
                    raise OSError(f"audit target is not a regular file: {path}")
                if os.name != "nt":
                    if st.st_uid != getattr(os, "geteuid")():
                        raise OSError(
                            f"audit target is not owned by the current user: {path}"
                        )
                    if mode != 0o600:
                        os.fchmod(fd, stat.S_IRUSR | stat.S_IWUSR)
                written = os.write(fd, encoded)
                if written != len(encoded):
                    raise OSError(
                        f"short audit write: expected {len(encoded)} bytes, wrote {written}"
                    )
                if strict:
                    os.fsync(fd)
            finally:
                os.close(fd)
    except Exception as e:
        _log.warning("dashboard-auth audit log write failed: %s", e)
        if strict:
            raise AuditWriteError(str(e)) from e
