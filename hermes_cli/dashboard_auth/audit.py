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

import contextlib
import datetime as _dt
import enum
import json
import logging
import os
import stat
import threading
import uuid
from pathlib import Path
from typing import Any, Iterable, Iterator, Optional

try:  # POSIX
    import fcntl as _fcntl
except ImportError:  # pragma: no cover - platform-dependent
    _fcntl = None  # type: ignore[assignment]

try:  # Windows
    import msvcrt as _msvcrt
except ImportError:  # pragma: no cover - platform-dependent
    _msvcrt = None  # type: ignore[assignment]

_log = logging.getLogger(__name__)
_write_lock = threading.Lock()

# Windows byte-range locks are MANDATORY, so the range this module locks must
# be one no reader ever wants: a single byte far past any plausible end of the
# log. POSIX ``flock`` is advisory and whole-file, so it needs no such offset.
_WINDOWS_LOCK_OFFSET = 0x7FFF_0000

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
    # Neither of these is an auth event. They are the two halves of one
    # journalled batch: ``BATCH_PREPARED`` carries an entry that has not
    # happened yet, and ``BATCH_COMMITTED`` carries the entries that did.
    # See ``begin_audit_batch`` / ``commit_audit_batch``.
    BATCH_PREPARED = "audit_batch_prepared"
    BATCH_COMMITTED = "audit_batch_committed"


# The field that ties a journalled entry to its commit marker.
_BATCH_FIELD = "audit_batch"

# Where a prepared entry's payload sits: as an OPAQUE JSON *string*, never as
# the line's own fields and never as a nested object. A prepared entry is a
# claim nobody has made yet, so the line that carries it must not read as one
# — not to this module's reader, and not to the ordinary `json.loads` per
# line (or grep) that this log's documented format invites. Escaping the
# payload means a prepared line contains no ``"event":"token_auth_success"``
# and no ``"decision":"allow"`` text at all.
_PREPARED_FIELD = "prepared_record"

# Where a committed batch's real records sit. These DID happen, so they are
# plain nested objects a line-oriented consumer can read directly.
_RECORDS_FIELD = "records"


class AuditWriteError(Exception):
    """Raised by ``audit_log(..., strict=True)`` when the write failed.

    Every existing call site omits ``strict`` and keeps the original
    never-raises behaviour (auth must not break because the audit logger
    broke). ``strict=True`` is for a caller whose OWN contract is the
    opposite — Item 32G-A's machine-credential routes must not serve data
    they cannot durably audit — so it needs to observe the failure instead of
    having it silently swallowed.
    """


class AuditRollbackUncertain(AuditWriteError):
    """A failed commit's own bytes could not be proven gone from the log.

    A subclass, so every existing ``except AuditWriteError`` call site keeps
    failing closed exactly as before. It exists because the two failures are
    not the same fact: an ordinary :class:`AuditWriteError` from
    :func:`commit_audit_batch` means the log says nothing about this
    operation, so the caller's rollback makes the log and the world agree.
    This one means the log may still be asserting a batch the caller is about
    to revert, and a caller that reports the outcome to an owner needs to be
    able to tell those apart rather than being told "write failed" for both.
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
    audit_log_batch([(event, fields)], strict=strict)


def audit_log_batch(
    entries: "list[tuple[AuditEvent, dict[str, Any]]]", *, strict: bool = False
) -> None:
    """Append several events as ONE all-or-nothing record.

    A multi-step operation that audits each step separately leaves the earlier
    steps' "allow" lines behind when a later one fails, so the log reads as a
    partial success the operation never had. Encoding every line up front and
    writing them in a single ``O_APPEND`` write makes the record match the
    operation: either the whole batch is durable or none of it is.

    ``audit_log`` is the one-entry case and keeps exactly its previous
    semantics — same redaction, same single write, same ``fsync`` under
    ``strict``, same :class:`AuditWriteError`.
    """
    if not entries:
        return
    _write_records(_encode_records(entries), strict=strict)


def _record(event: AuditEvent, fields: "dict[str, Any]") -> "dict[str, Any]":
    """One audit record, timestamped and redacted. Not yet serialized."""
    return {
        "ts": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "event": event.value,
        **{k: v for k, v in fields.items() if k not in _REDACTED_FIELDS},
    }


def _encode_line(record: "dict[str, Any]") -> bytes:
    return (json.dumps(record, separators=(",", ":")) + "\n").encode("utf-8")


def _encode_records(
    entries: "Iterable[tuple[AuditEvent, dict[str, Any]]]",
) -> bytes:
    """Serialize every line UP FRONT, before any file is touched.

    A value that cannot be serialized therefore fails the whole record before
    a single byte is appended, rather than half-way through it.
    """
    return b"".join(_encode_line(_record(event, fields)) for event, fields in entries)


@contextlib.contextmanager
def _exclusive_append_lock(fd: int) -> "Iterator[None]":
    """Hold this log against every other PROCESS for the whole write window.

    ``_write_lock`` only orders threads inside one interpreter, and the window
    a durable append must own is wider than the write itself: it spans the
    ``os.write``, the ``fsync``, and the rollback that removes the append again
    when that ``fsync`` fails. Because a rollback may only delete its own bytes
    while they are still the file's tail (see :func:`_rollback_append`), a
    second process appending anywhere inside that window pins a
    non-durable — and, for a commit line, already-reverted — record in the log
    permanently. Every writer here therefore takes the lock, non-strict ones
    included: an ordinary ``login_success`` line landing mid-window is enough
    to strand a rolled-back batch.

    The lock is the OS-native one and is held by the open file description, so
    the kernel drops it when the fd closes or the process dies — a crash inside
    the critical section cannot wedge the log, and no daemon, lock file or
    stale-lock reaper is involved. It is taken and released around the shortest
    region that is still correct, so concurrent writers queue only for the
    duration of one append, and readers are never blocked: ``flock`` is
    advisory, and the Windows range sits past any plausible EOF.

    A platform with neither primitive degrades to the previous thread-only
    ordering rather than failing the write.
    """
    if _fcntl is not None:
        _fcntl.flock(fd, _fcntl.LOCK_EX)
        try:
            yield
        finally:
            try:
                _fcntl.flock(fd, _fcntl.LOCK_UN)
            except OSError:  # pragma: no cover - the close below unlocks too
                pass
        return
    if _msvcrt is not None:  # pragma: no cover - exercised on Windows only
        # ``msvcrt.locking`` acts at the current file position, so the seek is
        # what selects the past-EOF range. Both positions are restored: the
        # descriptor is shared with the caller's own append.
        entry = os.lseek(fd, 0, os.SEEK_CUR)
        os.lseek(fd, _WINDOWS_LOCK_OFFSET, os.SEEK_SET)
        _msvcrt.locking(fd, _msvcrt.LK_LOCK, 1)
        os.lseek(fd, entry, os.SEEK_SET)
        try:
            yield
        finally:
            try:
                current = os.lseek(fd, 0, os.SEEK_CUR)
                os.lseek(fd, _WINDOWS_LOCK_OFFSET, os.SEEK_SET)
                _msvcrt.locking(fd, _msvcrt.LK_UNLCK, 1)
                os.lseek(fd, current, os.SEEK_SET)
            except OSError:
                pass
        return
    yield  # pragma: no cover - no platform ships without both


def _rollback_append(path: Path, fd: int, pre_size: int, appended: int) -> bool:
    """Undo this call's own append, and report whether that is PROVEN.

    A write whose ``fsync`` failed is not durable, but the bytes are in the
    file and every reader — this module's and a plain line-oriented one —
    would treat them as a record. The caller is about to report failure and
    roll its operation back, so the honest state is the one before the append.

    Only the exact bytes this call added are ever removed. While they are still
    the file's tail that is a plain truncate. When they are NOT — a writer that
    ignores the advisory lock appended inside the window — truncating would
    destroy that writer's record, so this call's own byte range is instead
    overwritten IN PLACE with one blank line: exactly ``appended`` bytes ending
    in the newline that keeps the following record parseable. Both readers skip
    a blank line, so the reverted batch stops asserting anything either way,
    and the interleaved record survives untouched.

    Returns ``False`` when neither could be completed — the log may still be
    claiming a batch the caller is about to revert, which is a different fact
    from "the commit was never written" and is propagated as such (see
    :class:`AuditRollbackUncertain`).
    """
    if appended <= 0:
        return True
    try:
        if os.fstat(fd).st_size == pre_size + appended:
            os.ftruncate(fd, pre_size)
            os.fsync(fd)
            return True
    except OSError:
        return False
    return _blank_own_range(path, fd, pre_size, appended)


def _blank_own_range(path: Path, fd: int, offset: int, length: int) -> bool:
    """Overwrite ``length`` bytes at ``offset`` with one blank line.

    Needs a descriptor WITHOUT ``O_APPEND`` — every write on the append-only
    one this module holds goes to the end of the file regardless of the offset
    asked for (``pwrite`` included, on Linux). The replacement descriptor is
    checked to be the very same inode as the one the caller already validated
    and locked, so re-opening by path cannot be redirected in the window.
    """
    flags = os.O_RDWR
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        other = os.open(path, flags)
    except OSError:
        return False
    try:
        locked, reopened = os.fstat(fd), os.fstat(other)
        if (locked.st_dev, locked.st_ino) != (reopened.st_dev, reopened.st_ino):
            return False
        os.lseek(other, offset, os.SEEK_SET)
        blank = b" " * (length - 1) + b"\n"
        if os.write(other, blank) != len(blank):
            return False
        os.fsync(other)
        return True
    except OSError:
        return False
    finally:
        os.close(other)


def _write_records(
    encoded: bytes, *, strict: bool, rollback_on_failure: bool = False,
) -> None:
    """Append already-encoded lines, honouring the strict durability contract."""
    path = _resolve_log_path()
    reverted = True
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with _write_lock:
            if not strict:
                with open(path, "a", encoding="utf-8") as handle:
                    with _exclusive_append_lock(handle.fileno()):
                        handle.write(encoded.decode("utf-8"))
                        # Flush inside the lock. Buffered bytes that only reach
                        # the file when the handle closes would land after the
                        # lock was released — i.e. exactly the interleaved
                        # append the lock exists to prevent.
                        handle.flush()
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
            # O_RDWR (not O_WRONLY) only so the record boundary below can be
            # checked on the same descriptor; O_APPEND still forces every write
            # to the end of the file.
            flags = os.O_RDWR | os.O_APPEND | os.O_CREAT
            if hasattr(os, "O_NOFOLLOW"):
                flags |= os.O_NOFOLLOW
            fd = os.open(path, flags, stat.S_IRUSR | stat.S_IWUSR)
            try:
                # Taken before the size is read, and released only after any
                # rollback: the boundary check, the append, the fsync and the
                # undo all have to see one uncontended tail.
                with _exclusive_append_lock(fd):
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
                    # An earlier short write can leave the file ending mid-record.
                    # Starting the next append on a fresh line keeps that damage to
                    # the one truncated record instead of swallowing this one too.
                    payload = encoded
                    if st.st_size and os.pread(fd, 1, st.st_size - 1) != b"\n":
                        payload = b"\n" + encoded
                    written = 0
                    try:
                        written = os.write(fd, payload)
                        if written != len(payload):
                            raise OSError(
                                f"short audit write: expected {len(payload)} bytes, wrote {written}"
                            )
                        if strict:
                            os.fsync(fd)
                    except Exception:
                        if rollback_on_failure:
                            reverted = _rollback_append(
                                path, fd, st.st_size, written
                            )
                        raise
            finally:
                os.close(fd)
    except Exception as e:
        _log.warning("dashboard-auth audit log write failed: %s", e)
        if not reverted:
            _log.error(
                "dashboard-auth audit log still contains a record for a "
                "reverted operation: %s", e,
            )
        if strict:
            if not reverted:
                raise AuditRollbackUncertain(str(e)) from e
            raise AuditWriteError(str(e)) from e


class AuditBatch:
    """One journalled batch: its id, and the exact records it will assert."""

    __slots__ = ("batch_id", "records")

    def __init__(self, batch_id: str, records: "tuple[dict[str, Any], ...]"):
        self.batch_id = batch_id
        self.records = records


def begin_audit_batch(
    entries: "list[tuple[AuditEvent, dict[str, Any]]]",
) -> AuditBatch:
    """Journal one batch durably, WITHOUT yet claiming it happened.

    Phase one of a two-phase record. Every entry is serialized, redacted and
    fsynced to disk here — so a caller whose contract is "never serve what you
    cannot durably audit" has that proof before it acts — but it lands inside
    an opaque ``audit_batch_prepared`` envelope (see :data:`_PREPARED_FIELD`).
    A prepared line therefore asserts nothing: not to
    :func:`read_audit_records`, and not to the ordinary one-JSON-object-per-
    line consumer this log's format invites, which sees an event named
    ``audit_batch_prepared`` with no ``decision`` and no auth event on it.

    :func:`commit_audit_batch` is what writes the real records. An ``os.write``
    that lands short, an ``fsync`` that fails after the bytes reached the file,
    a rollback, a crash, or a concurrent batch interleaving its own lines
    therefore cannot leave behind ``decision="allow"`` records — effective or
    apparent — for work that was reverted.

    Raises :class:`AuditWriteError` exactly like ``strict`` ``audit_log_batch``.
    """
    if not entries:
        raise ValueError("an audit batch needs at least one entry")
    batch_id = uuid.uuid4().hex
    # Serializing each record first is what makes an unserializable value fail
    # the whole batch before a byte is appended; decoding it back is what makes
    # the committed line byte-identical to what was journalled.
    encoded_records = [
        json.dumps(_record(event, fields), separators=(",", ":"))
        for event, fields in entries
    ]
    prepared = b"".join(
        _encode_line({
            "ts": _dt.datetime.now(_dt.timezone.utc).isoformat(),
            "event": AuditEvent.BATCH_PREPARED.value,
            _BATCH_FIELD: batch_id,
            _PREPARED_FIELD: encoded,
        })
        for encoded in encoded_records
    )
    _write_records(prepared, strict=True)
    return AuditBatch(
        batch_id, tuple(json.loads(encoded) for encoded in encoded_records),
    )


def commit_audit_batch(batch: AuditBatch) -> None:
    """Write one journalled batch's records, durably, as ONE line.

    Phase two, and the last durable act of the operation the batch describes:
    everything the batch claims must already have succeeded when this is
    called. The whole batch is one JSON line, so a short write leaves invalid
    JSON that no reader — canonical or line-oriented — can mistake for a
    record, and a batch that is never committed has no record line at all.

    An ``fsync`` that fails after the line landed is not durable either, so
    that line is removed again before the failure is reported: the caller is
    about to roll its operation back, and the log must not outlive it saying
    otherwise. Removal works whether or not the file has grown since (see
    :func:`_rollback_append`), and the batch stays committable afterwards — a
    later attempt writes a fresh commit line for the same journalled records.

    Raises :class:`AuditRollbackUncertain` — a subclass, so an existing
    ``except AuditWriteError`` still fails closed — in the one case where even
    that removal could not be completed, so a caller that reports an outcome
    knows the log may still be asserting what it is about to revert.
    """
    _write_records(
        _encode_line({
            "ts": _dt.datetime.now(_dt.timezone.utc).isoformat(),
            "event": AuditEvent.BATCH_COMMITTED.value,
            _BATCH_FIELD: batch.batch_id,
            _RECORDS_FIELD: list(batch.records),
        }),
        strict=True,
        rollback_on_failure=True,
    )


def read_audit_records(path: Optional[Path] = None) -> "list[dict[str, Any]]":
    """Every audit record that actually happened, in file order.

    The canonical reader for this log's format, and the same answer a plain
    per-line ``json.loads`` consumer gets: ordinary one-shot entries are their
    own line, a committed batch's records are the objects under
    :data:`_RECORDS_FIELD` on its commit line, and a prepared entry is not a
    record at all. Unparseable (short-written) lines are skipped.

    A log written before batches carried their records on the commit line
    still reads correctly: an entry tagged with a batch id of its own is
    included only if that id's commit marker is present. Those pre-upgrade
    lines are the one case a plain line-oriented consumer cannot resolve on
    its own — nothing can rewrite bytes already on disk — which is why the
    format changed rather than only the reader.
    """
    target = path or _resolve_log_path()
    try:
        raw = target.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    parsed: list[dict[str, Any]] = []
    committed_batches: set[str] = set()
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except (json.JSONDecodeError, ValueError):
            continue
        if not isinstance(record, dict):
            continue
        parsed.append(record)
        if record.get("event") == AuditEvent.BATCH_COMMITTED.value and isinstance(
            record.get(_BATCH_FIELD), str
        ):
            committed_batches.add(record[_BATCH_FIELD])
    records: list[dict[str, Any]] = []
    for record in parsed:
        event = record.get("event")
        if event == AuditEvent.BATCH_PREPARED.value:
            continue
        if event == AuditEvent.BATCH_COMMITTED.value:
            nested = record.get(_RECORDS_FIELD)
            records.extend(
                entry for entry in (nested if isinstance(nested, list) else [])
                if isinstance(entry, dict)
            )
            continue
        batch = record.get(_BATCH_FIELD)
        if isinstance(batch, str) and batch not in committed_batches:
            continue
        records.append(record)
    return records
