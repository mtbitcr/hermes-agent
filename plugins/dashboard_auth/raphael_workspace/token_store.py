"""On-disk registry for Raphael's managed dashboard credentials.

The Workspace, owner-recommendations, and Connections surfaces share one
lifecycle registry but use disjoint fixed token prefixes, principals, scopes,
and grants. The CLI
can issue any number of bearer tokens, each individually revocable and
independently expiring; recommendation tokens are hard-capped at eight hours.
What is persisted here is auth METADATA ONLY — a per-token id, a SHA-256 digest
of the secret (never the secret itself), the fixed grant, and
issued_at/expires_at/revoked_at. This is not a second Kanban
authority: it never stores task/board data, only facts about which bearer
tokens exist and whether they are still good.

Storage: ``$HERMES_HOME/dashboard_auth/raphael_workspace_tokens.json``, a
single JSON object rewritten atomically (temp file + fsync + ``os.replace``)
on every mutation, owner-only (0o600 file / 0o700 parent dir) — mirrors
``hermes_cli.auth._save_auth_store``. Every read (:func:`load_records`) goes
straight to disk with no in-memory cache, so ``revoke()``/``issue()`` take
effect on the very next request and survive a process restart by
construction.

Fail-closed on malformed/unsafe state: a store that fails to parse, has the
wrong shape, or (POSIX) is readable/writable by group or other raises
:class:`TokenStoreError` rather than silently behaving as an empty registry.
An empty registry and a corrupt one both deny every token at the auth layer,
but only a corrupt one should refuse to serve `list`/`issue`/`revoke` calls
loudly instead of quietly discarding record of previously issued tokens.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
import stat
import sys
import time
import uuid
import secrets as _secrets
import threading
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Optional

from hermes_constants import get_hermes_home, secure_parent_dir
from hermes_cli.dashboard_auth.audit import AuditEvent, audit_log
# Frozen authority contract (Item 32G-A) — not configurable. Widening any of
# these would widen the credential beyond the one surface it was designed
# for, so there is deliberately no code path that accepts a different value.
PRINCIPAL = "raphael-workspace"
SCOPE = "kanban.read"
PROJECT = "raphael-workspace"
BOARD = "raphael-workspace"
TOKEN_PREFIX = "hrw1_"
GRANT = "raphael-workspace-kanban-read-v1"

RECOMMENDATIONS_PRINCIPAL = "kanban-recommendations-reader"
RECOMMENDATIONS_SCOPE = "kanban:recommendations:read"
RECOMMENDATIONS_PROJECT = "raphael-workspace"
RECOMMENDATIONS_BOARD = "raphael-workspace"
RECOMMENDATIONS_TOKEN_PREFIX = "hrr1_"
RECOMMENDATIONS_GRANT = "raphael-workspace-recommendations-read-v1"

CONNECTIONS_PRINCIPAL = "raphael-connections-manager"
CONNECTIONS_SCOPE = "mcp.connections.manage"
CONNECTIONS_PROJECT = "raphael-workspace"
CONNECTIONS_BOARD = "raphael-workspace"
CONNECTIONS_TOKEN_PREFIX = "hrc1_"
CONNECTIONS_GRANT = "raphael-workspace-connections-manage-v1"

WORKSPACE_SURFACE = "workspace"
RECOMMENDATIONS_SURFACE = "recommendations"
CONNECTIONS_SURFACE = "connections"


@dataclass(frozen=True)
class FixedTokenPolicy:
    surface: str
    token_prefix: str
    principal: str
    scope: str
    project: str
    board: str
    grant: str
    provider: str
    default_ttl_seconds: int
    max_ttl_seconds: int


STORE_VERSION = 1
_TOKEN_ID_SUFFIX_RE = re.compile(r"^[A-Za-z0-9_-]{12,64}$")
_DIGEST_RE = re.compile(r"^[0-9a-f]{64}$")
_TOKEN_RECORD_KEYS = frozenset(
    {
        "token_id",
        "digest",
        "principal",
        "scope",
        "project",
        "board",
        "grant",
        "issued_at",
        "expires_at",
        "revoked_at",
    }
)
_MAX_STORE_BYTES = 4 * 1024 * 1024
_store_thread_lock = threading.RLock()

# Bounded expiry (task requirement: "Bound expiry"). A caller must pick a TTL
# inside this range; there is no "forever" option.
MIN_TTL_SECONDS = 3600  # 1 hour
DEFAULT_TTL_SECONDS = 90 * 86400  # 90 days
MAX_TTL_SECONDS = 397 * 86400  # ~13 months
RECOMMENDATIONS_DEFAULT_TTL_SECONDS = 8 * 3600
RECOMMENDATIONS_MAX_TTL_SECONDS = 8 * 3600

_FIXED_TOKEN_POLICIES = {
    WORKSPACE_SURFACE: FixedTokenPolicy(
        surface=WORKSPACE_SURFACE,
        token_prefix=TOKEN_PREFIX,
        principal=PRINCIPAL,
        scope=SCOPE,
        project=PROJECT,
        board=BOARD,
        grant=GRANT,
        provider="raphael-workspace-token",
        default_ttl_seconds=DEFAULT_TTL_SECONDS,
        max_ttl_seconds=MAX_TTL_SECONDS,
    ),
    RECOMMENDATIONS_SURFACE: FixedTokenPolicy(
        surface=RECOMMENDATIONS_SURFACE,
        token_prefix=RECOMMENDATIONS_TOKEN_PREFIX,
        principal=RECOMMENDATIONS_PRINCIPAL,
        scope=RECOMMENDATIONS_SCOPE,
        project=RECOMMENDATIONS_PROJECT,
        board=RECOMMENDATIONS_BOARD,
        grant=RECOMMENDATIONS_GRANT,
        provider="raphael-recommendations-token",
        default_ttl_seconds=RECOMMENDATIONS_DEFAULT_TTL_SECONDS,
        max_ttl_seconds=RECOMMENDATIONS_MAX_TTL_SECONDS,
    ),
    CONNECTIONS_SURFACE: FixedTokenPolicy(
        surface=CONNECTIONS_SURFACE,
        token_prefix=CONNECTIONS_TOKEN_PREFIX,
        principal=CONNECTIONS_PRINCIPAL,
        scope=CONNECTIONS_SCOPE,
        project=CONNECTIONS_PROJECT,
        board=CONNECTIONS_BOARD,
        grant=CONNECTIONS_GRANT,
        provider="raphael-connections-token",
        default_ttl_seconds=DEFAULT_TTL_SECONDS,
        max_ttl_seconds=MAX_TTL_SECONDS,
    ),
}


def policy_for_surface(surface: str) -> FixedTokenPolicy:
    try:
        return _FIXED_TOKEN_POLICIES[surface]
    except (KeyError, TypeError) as exc:
        raise ValueError(f"unknown token surface: {surface!r}") from exc


def policy_for_token_id(token_id: str) -> Optional[FixedTokenPolicy]:
    if not isinstance(token_id, str):
        return None
    return next(
        (
            policy
            for policy in _FIXED_TOKEN_POLICIES.values()
            if token_id.startswith(policy.token_prefix)
        ),
        None,
    )


class TokenStoreError(Exception):
    """The on-disk token registry is missing, malformed, or unsafe to trust."""


class UnknownTokenError(TokenStoreError):
    """``revoke()`` was asked about a token_id that does not exist."""


class PlaintextOutputError(Exception):
    """The explicit ``--out`` path for a freshly issued token could not be
    written safely (see :func:`write_plaintext_once`)."""


@dataclass(frozen=True)
class TokenRecord:
    """Auth metadata for one issued token. Never carries the secret itself."""

    token_id: str
    digest: str
    principal: str
    scope: str
    project: str
    board: str
    grant: str
    issued_at: int
    expires_at: int
    revoked_at: Optional[int]

    @property
    def status(self) -> str:
        if self.revoked_at is not None:
            return "revoked"
        if self.expires_at <= int(time.time()):
            return "expired"
        return "active"

    def to_dict(self) -> dict:
        return {
            "token_id": self.token_id,
            "digest": self.digest,
            "principal": self.principal,
            "scope": self.scope,
            "project": self.project,
            "board": self.board,
            "grant": self.grant,
            "issued_at": self.issued_at,
            "expires_at": self.expires_at,
            "revoked_at": self.revoked_at,
        }

    @classmethod
    def from_dict(cls, raw: dict) -> "TokenRecord":
        if set(raw) != _TOKEN_RECORD_KEYS:
            raise TokenStoreError("malformed token record fields")
        for field in (
            "token_id",
            "digest",
            "principal",
            "scope",
            "project",
            "board",
            "grant",
        ):
            if not isinstance(raw[field], str):
                raise TokenStoreError(f"malformed token record field: {field}")
        if type(raw["issued_at"]) is not int or type(raw["expires_at"]) is not int:
            raise TokenStoreError("malformed token lifetime fields")
        revoked_at = raw["revoked_at"]
        if revoked_at is not None and type(revoked_at) is not int:
            raise TokenStoreError("malformed revoked_at in token record")
        record = cls(
            token_id=raw["token_id"],
            digest=raw["digest"],
            principal=raw["principal"],
            scope=raw["scope"],
            project=raw["project"],
            board=raw["board"],
            grant=raw["grant"],
            issued_at=raw["issued_at"],
            expires_at=raw["expires_at"],
            revoked_at=revoked_at,
        )
        record.validate()
        return record

    def validate(self) -> None:
        """Fail closed unless the persisted record is exactly this fixed grant."""
        if any(
            not isinstance(value, str)
            for value in (
                self.token_id,
                self.digest,
                self.principal,
                self.scope,
                self.project,
                self.board,
                self.grant,
            )
        ):
            raise TokenStoreError("malformed string field in token record")
        if type(self.issued_at) is not int or type(self.expires_at) is not int:
            raise TokenStoreError("malformed token lifetime fields")
        if self.revoked_at is not None and type(self.revoked_at) is not int:
            raise TokenStoreError("malformed revoked_at in token record")
        policy = policy_for_token_id(self.token_id)
        if policy is None or not _TOKEN_ID_SUFFIX_RE.fullmatch(
            self.token_id[len(policy.token_prefix) :]
        ):
            raise TokenStoreError("malformed token_id in token record")
        if not _DIGEST_RE.fullmatch(self.digest):
            raise TokenStoreError("malformed digest in token record")
        if (
            self.principal != policy.principal
            or self.scope != policy.scope
            or self.project != policy.project
            or self.board != policy.board
            or self.grant != policy.grant
        ):
            raise TokenStoreError("token record grant does not match the fixed policy")
        lifetime = self.expires_at - self.issued_at
        if self.issued_at < 0 or not (
            MIN_TTL_SECONDS <= lifetime <= policy.max_ttl_seconds
        ):
            raise TokenStoreError("malformed token lifetime in token record")
        if self.revoked_at is not None and self.revoked_at < self.issued_at:
            raise TokenStoreError("malformed revocation time in token record")


# ---------------------------------------------------------------------------
# Paths + safety checks
# ---------------------------------------------------------------------------


def store_path() -> Path:
    """``$HERMES_HOME/dashboard_auth/raphael_workspace_tokens.json``."""
    return get_hermes_home() / "dashboard_auth" / "raphael_workspace_tokens.json"


def _lock_path() -> Path:
    path = store_path()
    return path.with_name(f".{path.name}.lock")


def _check_safe_stat(st: os.stat_result, path: Path) -> None:
    """Refuse non-files and unsafe POSIX ownership or permissions.

    No-op on Windows — POSIX mode bits aren't meaningful there, matching
    ``hermes_constants.secure_parent_dir``'s documented posture.
    """
    if not stat.S_ISREG(st.st_mode):
        raise TokenStoreError(f"{path} is not a regular file")
    if sys.platform.startswith("win"):
        return
    mode = stat.S_IMODE(st.st_mode)
    if st.st_uid != getattr(os, "geteuid")():
        raise TokenStoreError(f"{path} is not owned by the current user")
    if mode != 0o600:
        raise TokenStoreError(
            f"{path} has unsafe permissions 0o{mode:o} (must be exactly "
            "0o600) — refusing to trust it"
        )


def _check_safe_file_mode(path: Path) -> None:
    """Refuse symlinks/non-files and unsafe POSIX ownership or permissions."""
    st = path.lstat()
    if stat.S_ISLNK(st.st_mode):
        raise TokenStoreError(f"{path} must not be a symlink")
    _check_safe_stat(st, path)


def _check_safe_parent(path: Path) -> None:
    parent = path.parent
    st = parent.lstat()
    if stat.S_ISLNK(st.st_mode) or not stat.S_ISDIR(st.st_mode):
        raise TokenStoreError(f"{parent} is not a regular non-symlink directory")
    if sys.platform.startswith("win"):
        return
    mode = stat.S_IMODE(st.st_mode)
    if st.st_uid != getattr(os, "geteuid")() or mode != 0o700:
        raise TokenStoreError(
            f"{parent} has unsafe ownership or permissions 0o{mode:o}"
        )


@contextmanager
def _locked_store():
    """Serialize registry read-modify-write across threads and processes."""
    with _store_thread_lock:
        lock_path = _lock_path()
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        secure_parent_dir(lock_path)
        _check_safe_parent(lock_path)
        flags = os.O_RDWR | os.O_CREAT
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        try:
            fd = os.open(lock_path, flags, stat.S_IRUSR | stat.S_IWUSR)
        except OSError as exc:
            raise TokenStoreError(f"could not open token-store lock: {exc}") from exc
        try:
            _check_safe_stat(os.fstat(fd), lock_path)
            if os.name == "nt":  # pragma: no cover - Windows CI exercises the branch
                import msvcrt

                if os.fstat(fd).st_size == 0:
                    os.write(fd, b" ")
                os.lseek(fd, 0, os.SEEK_SET)
                msvcrt.locking(fd, msvcrt.LK_LOCK, 1)
            else:
                import fcntl

                fcntl.flock(fd, fcntl.LOCK_EX)
            try:
                yield
            finally:
                if os.name == "nt":  # pragma: no cover
                    os.lseek(fd, 0, os.SEEK_SET)
                    msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
                else:
                    fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)


# ---------------------------------------------------------------------------
# Load / save
# ---------------------------------------------------------------------------


def _load_raw() -> dict:
    path = store_path()
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        fd = os.open(path, flags)
    except FileNotFoundError:
        return {"version": STORE_VERSION, "tokens": []}
    except OSError as exc:
        raise TokenStoreError(f"could not open token store {path}: {exc}") from exc
    try:
        st = os.fstat(fd)
        _check_safe_stat(st, path)
        if st.st_size > _MAX_STORE_BYTES:
            raise TokenStoreError(f"token store {path} exceeds size limit")
        chunks: list[bytes] = []
        remaining = int(st.st_size)
        while remaining:
            chunk = os.read(fd, min(remaining, 1024 * 1024))
            if not chunk:
                raise TokenStoreError(f"short read from token store {path}")
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(fd, 1):
            raise TokenStoreError(f"token store {path} changed while being read")
        raw = json.loads(b"".join(chunks).decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TokenStoreError(f"could not read token store {path}: {exc}") from exc
    finally:
        os.close(fd)
    if (
        not isinstance(raw, dict)
        or raw.get("version") != STORE_VERSION
        or not isinstance(raw.get("tokens"), list)
    ):
        raise TokenStoreError(f"token store {path} has an unrecognised shape")
    return raw


def _load_records_unlocked() -> list[TokenRecord]:
    raw = _load_raw()
    if any(not isinstance(item, dict) for item in raw["tokens"]):
        raise TokenStoreError("malformed token record")
    records = [TokenRecord.from_dict(r) for r in raw["tokens"]]
    ids = [r.token_id for r in records]
    if len(ids) != len(set(ids)):
        raise TokenStoreError("duplicate token_id in token store")
    return records


def load_records() -> list[TokenRecord]:
    """Read every token record fresh from disk. Raises on malformed/unsafe state."""
    path = store_path()
    if path.is_symlink():
        raise TokenStoreError(f"token store {path} must not be a symlink")
    if not path.exists():
        return []
    with _locked_store():
        return _load_records_unlocked()


def _save_records_unlocked(records: list[TokenRecord]) -> None:
    """Atomic, owner-only overwrite of the whole registry.

    Mirrors ``hermes_cli.auth._save_auth_store``: temp file created with
    ``O_CREAT | O_EXCL`` at 0o600 (closes the TOCTOU window a plain
    ``open()``+later-``chmod`` leaves), fsynced, then atomically renamed onto
    the real path.
    """
    path = store_path()
    if path.is_symlink():
        raise TokenStoreError(f"token store {path} must not be a symlink")
    for record in records:
        record.validate()
    ids = [r.token_id for r in records]
    if len(ids) != len(set(ids)):
        raise TokenStoreError("duplicate token_id in token store")
    path.parent.mkdir(parents=True, exist_ok=True)
    secure_parent_dir(path)
    _check_safe_parent(path)
    payload = (
        json.dumps(
            {"version": STORE_VERSION, "tokens": [r.to_dict() for r in records]},
            indent=2,
        )
        + "\n"
    )
    tmp_path = path.with_name(f"{path.name}.tmp.{os.getpid()}.{uuid.uuid4().hex}")
    try:
        fd = os.open(
            str(tmp_path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, stat.S_IRUSR | stat.S_IWUSR
        )
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
        if os.name != "nt":
            parent_fd = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(parent_fd)
            finally:
                os.close(parent_fd)
    except OSError as exc:
        raise TokenStoreError(f"could not write token store {path}: {exc}") from exc
    finally:
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except OSError:
            pass
    _check_safe_file_mode(path)


def _save_records(records: list[TokenRecord]) -> None:
    """Test/support wrapper that preserves the same cross-process lock."""
    with _locked_store():
        _save_records_unlocked(records)


# ---------------------------------------------------------------------------
# Issue / revoke / verify
# ---------------------------------------------------------------------------


def _new_token_id(policy: FixedTokenPolicy) -> str:
    return policy.token_prefix + _secrets.token_urlsafe(12)


def _new_secret() -> str:
    return _secrets.token_urlsafe(32)  # 256 bits


def issue(
    *,
    out_path: Path,
    ttl_seconds: Optional[int] = None,
    replaces_token_id: Optional[str] = None,
    surface: str = WORKSPACE_SURFACE,
) -> TokenRecord:
    """Mint, audit and activate one token without returning its plaintext.

    The bearer bytes go only to ``out_path`` via :func:``write_plaintext_once``.
    Audit succeeds before the digest becomes active; a failed audit or store
    write removes the new output file and leaves no active credential.
    """
    policy = policy_for_surface(surface)
    if ttl_seconds is None:
        ttl_seconds = policy.default_ttl_seconds
    if type(ttl_seconds) is not int or not (
        MIN_TTL_SECONDS <= ttl_seconds <= policy.max_ttl_seconds
    ):
        raise ValueError(
            f"ttl_seconds must be between {MIN_TTL_SECONDS} and "
            f"{policy.max_ttl_seconds} for {surface}, got {ttl_seconds}"
        )
    if replaces_token_id is not None:
        previous = find(replaces_token_id)
        if previous is None or previous.status != "active":
            raise UnknownTokenError(
                f"replacement token_id is not active: {replaces_token_id!r}"
            )
        previous_policy = policy_for_token_id(previous.token_id)
        if previous_policy != policy:
            raise TokenStoreError(
                "replacement token belongs to a different credential surface"
            )
    token_id = _new_token_id(policy)
    secret = _new_secret()
    digest = hashlib.sha256(secret.encode("utf-8")).hexdigest()
    now = int(time.time())
    record = TokenRecord(
        token_id=token_id,
        digest=digest,
        principal=policy.principal,
        scope=policy.scope,
        project=policy.project,
        board=policy.board,
        grant=policy.grant,
        issued_at=now,
        expires_at=now + ttl_seconds,
        revoked_at=None,
    )
    record.validate()
    plaintext = f"{token_id}.{secret}\n"
    write_plaintext_once(out_path, plaintext)
    try:
        audit_log(
            AuditEvent.MACHINE_TOKEN_ISSUED,
            strict=True,
            provider=policy.provider,
            principal=policy.principal,
            credential_id=token_id,
            scope=policy.scope,
            project=policy.project,
            board=policy.board,
            grant=policy.grant,
            expires_at=record.expires_at,
            replaces_credential_id=replaces_token_id,
            decision="allow",
        )
        with _locked_store():
            records = _load_records_unlocked()
            if replaces_token_id is not None:
                previous = next(
                    (r for r in records if r.token_id == replaces_token_id), None
                )
                if previous is None or previous.status != "active":
                    raise UnknownTokenError(
                        "replacement token_id stopped being active before issuance: "
                        f"{replaces_token_id!r}"
                    )
                if policy_for_token_id(previous.token_id) != policy:
                    raise TokenStoreError(
                        "replacement token belongs to a different credential surface"
                    )
            if any(existing.token_id == token_id for existing in records):
                raise TokenStoreError("generated token_id already exists")
            records.append(record)
            _save_records_unlocked(records)
    except Exception:
        try:
            out_path.unlink()
        except OSError:
            pass
        raise
    return record


def find(token_id: str) -> Optional[TokenRecord]:
    path = store_path()
    if path.is_symlink():
        raise TokenStoreError(f"token store {path} must not be a symlink")
    if not path.exists():
        return None
    with _locked_store():
        for record in _load_records_unlocked():
            if record.token_id == token_id:
                return record
    return None


def revoke(token_id: str) -> TokenRecord:
    """Mark ``token_id`` revoked. Idempotent — revoking twice is not an error.

    Raises :class:`UnknownTokenError` if no such token was ever issued.
    Takes effect immediately: the next :func:`verify` call re-reads the
    store from disk.
    """
    with _locked_store():
        records = _load_records_unlocked()
        updated: Optional[TokenRecord] = None
        already_revoked = False
        for i, record in enumerate(records):
            if record.token_id != token_id:
                continue
            if record.revoked_at is not None:
                updated = record
                already_revoked = True
            else:
                updated = replace(record, revoked_at=int(time.time()))
                records[i] = updated
                _save_records_unlocked(records)
            break
        if updated is None:
            raise UnknownTokenError(f"no such token_id: {token_id!r}")
    policy = policy_for_token_id(updated.token_id)
    assert policy is not None  # TokenRecord.validate() already proved this.
    # Revocation is safety-monotonic: if the audit sink is unavailable, the
    # credential stays revoked and the CLI still fails closed instead of
    # reporting an unaudited success.
    audit_log(
        AuditEvent.MACHINE_TOKEN_REVOKED,
        strict=True,
        provider=policy.provider,
        principal=policy.principal,
        credential_id=token_id,
        grant=policy.grant,
        decision="already_revoked" if already_revoked else "revoke",
        revoked_at=updated.revoked_at,
    )
    return updated


def verify(token_id: str, secret: str) -> Optional[TokenRecord]:
    """Return the record for ``token_id`` if ``secret`` matches, else ``None``.

    ``None`` covers: unknown token_id, revoked, expired, or a wrong secret —
    the caller (the auth provider) cannot and must not distinguish these from
    each other. Uses ``hmac.compare_digest`` so a wrong secret cannot be
    recovered by timing. Re-reads the store from disk on every call (no
    cache), so revocation/expiry are always current.
    """
    path = store_path()
    if path.is_symlink():
        raise TokenStoreError(f"token store {path} must not be a symlink")
    if not path.exists():
        return None
    with _locked_store():
        record = next(
            (r for r in _load_records_unlocked() if r.token_id == token_id), None
        )
        if record is None or record.revoked_at is not None:
            return None
        if record.expires_at <= int(time.time()):
            return None
        digest = hashlib.sha256(secret.encode("utf-8")).hexdigest()
        if not hmac.compare_digest(digest, record.digest):
            return None
        return record


# ---------------------------------------------------------------------------
# Plaintext output (issue-time only)
# ---------------------------------------------------------------------------


def write_plaintext_once(path: Path, content: str) -> None:
    """Write ``content`` to ``path`` exactly once; never overwrite, never follow a symlink.

    Contract (Item 32G-A):
      * ``path``'s parent directory must already exist and (POSIX) be
        owner-only (no group/other permission bits). This function never
        creates or loosens a directory — an already-existing trusted
        owner-only directory is a precondition, not something it arranges.
      * ``path`` itself must not already exist and must not be a symlink —
        ``O_CREAT | O_EXCL | O_NOFOLLOW`` closes the TOCTOU window between
        any pre-check and the actual write.
      * The file is created at mode 0o600 directly via ``os.open`` (never a
        default-umask create, then a later ``chmod``).

    Raises :class:`PlaintextOutputError` for every failure mode above, and
    never partially writes: the file ends up with the full content or is not
    created at all (a failure mid-write unlinks the partial file).
    """
    parent = path.parent
    if parent.is_symlink() or not parent.is_dir():
        raise PlaintextOutputError(
            f"parent directory {parent} must be an existing non-symlink directory"
        )
    if not sys.platform.startswith("win"):
        parent_stat = parent.stat()
        parent_mode = stat.S_IMODE(parent_stat.st_mode)
        if parent_stat.st_uid != getattr(os, "geteuid")():
            raise PlaintextOutputError(
                f"parent directory {parent} is not owned by the current user"
            )
        if parent_mode != 0o700:
            raise PlaintextOutputError(
                f"parent directory {parent} is not owner-only (mode "
                f"0o{parent_mode:o}) — mode must be exactly 0o700"
            )
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        fd = os.open(str(path), flags, stat.S_IRUSR | stat.S_IWUSR)
    except FileExistsError as exc:
        raise PlaintextOutputError(
            f"{path} already exists — refusing to overwrite it"
        ) from exc
    except OSError as exc:
        raise PlaintextOutputError(f"could not create {path}: {exc}") from exc
    try:
        created_stat = os.fstat(fd)
        if not stat.S_ISREG(created_stat.st_mode):
            raise OSError("created output is not a regular file")
        if not sys.platform.startswith("win"):
            created_mode = stat.S_IMODE(created_stat.st_mode)
            if created_stat.st_uid != getattr(os, "geteuid")() or created_mode != 0o600:
                raise OSError(
                    f"created output has unsafe ownership/mode 0o{created_mode:o}"
                )
        handle = os.fdopen(fd, "w", encoding="utf-8")
        fd = -1
        with handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
    except OSError as exc:
        if fd >= 0:
            try:
                os.close(fd)
            except OSError:
                pass
        try:
            path.unlink()
        except OSError:
            pass
        raise PlaintextOutputError(f"failed writing {path}: {exc}") from exc
    if not sys.platform.startswith("win"):
        final_stat = path.lstat()
        if (
            not stat.S_ISREG(final_stat.st_mode)
            or final_stat.st_uid != getattr(os, "geteuid")()
            or stat.S_IMODE(final_stat.st_mode) != 0o600
        ):
            try:
                path.unlink()
            except OSError:
                pass
            raise PlaintextOutputError("plaintext output failed final safety check")
