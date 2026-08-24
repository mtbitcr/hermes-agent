"""Owner-workspace mutation kernel.

The single deep in-process boundary behind the API-server-only
``owner_workspace`` toolset (``owner_workspace_bootstrap``, ``owner_task_graph_commit``,
``owner_task_move``, ``owner_task_comment`` — see ``tools/owner_workspace_tools.py``). All four
tools are thin wrappers that resolve trusted identity + validate shape, then
call into this module.

Security contract enforced HERE (not at the tool layer):

* **Trusted context only.** ``OwnerContext`` (actor/profile/session) is
  derived from request-bound state (``resolve_owner_context``) — never from
  a tool-call argument. A multiplexed ``/p/<profile>/`` request binds ITS OWN
  profile because ``get_active_profile_name()`` resolves through
  ``get_hermes_home()``, which the API server's profile-prefix middleware
  scopes per request via a contextvar.
* **Idempotency + canonical digest.** Every mutation requires an
  ``idempotency_key``. A durable receipt, scoped to the trusted
  actor/profile boundary (not the transient session) in ``projects.db``,
  records a canonical digest of the request payload. Exact replay (same
  key + same digest) returns the ORIGINAL serialized result. Reusing a key
  with a different payload fails closed with zero domain changes.
* **Exactly-once concurrent claim.** Concurrent callers with the same key
  race to INSERT the receipt row inside one ``write_txn`` (SQLite's
  BEGIN IMMEDIATE serializes writers). The loser sees the winner's row and
  waits (bounded poll) for it to reach a terminal state rather than
  requesting its own confirmation or creating its own objects. A receipt
  whose claim lock has expired (crashed claimer) is adopted under a freshly
  minted ``lock_token``, rolling forward via deterministic identities
  (:func:`_derive_id`) — re-verifying/deriving each object before reuse,
  never a blind re-create.
* **Enforced lease ownership — fenced, not check-then-act.** ``lock_token``
  is not just written at claim time — every progress write
  (:func:`_update_progress`), every durable domain mutation, and
  finalization (:func:`_finalize_receipt`) are predicated on the CURRENT
  claimant's token still owning the row. Because the receipt lives in
  ``projects.db`` while the domain mutation may land in ``kanban.db`` or
  ``board.json``, a bare "validate, then separately mutate" would leave a
  window between the two where the lease could be stolen. Every domain
  mutation therefore runs inside a ``with write_txn(pconn):`` fence that
  ALSO contains the immediately-preceding :func:`_assert_owns_lease` call:
  the fence's ``BEGIN IMMEDIATE`` takes ``projects.db``'s write lock before
  the lease is even read, and holds it until the mutation (and, for
  same-database Project-table writes, the write itself) is done — so a
  competing claimant's own adoption (which also needs that write lock, via
  the same :func:`write_txn`) cannot interleave. Either the fenced mutation
  runs to completion first (the competitor blocks until it commits, then
  sees a terminal/foreign row and cannot adopt), or the competitor's
  adoption commits first (in which case this fence's own
  :func:`_assert_owns_lease` — executed after acquiring the same lock —
  observes the new token and fails closed with ``lease_lost`` before
  touching domain state). Once an expired lease is adopted with a new
  token, the old claimant's stale token can no longer match — it fails
  closed (``lease_lost``) instead of mutating domain state, overwriting
  progress, or finalizing on the new claimant's behalf.
* **Cross-profile board ownership.** Receipts and Project rows are
  profile-local (``projects.db``), but kanban boards are shared across
  profiles by design. The receipt lease therefore cannot order two
  same-name bootstraps started from different profiles, so the whole
  ownership check/create/bind decision is additionally held under
  :func:`_global_board_guard` — an OS-level (``fcntl.flock`` /
  ``msvcrt.locking``) exclusive lock whose identity is the canonical global
  board namespace, revalidated inside the critical section and failing
  closed if the guard cannot be created or acquired.
* **Exact-operation confirmation.** Every mutation calls
  ``tools.approval.request_exact_operation_approval`` — payload-, actor-,
  profile-, session-, expiry-, and operation-bound, "once"/"deny" only.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import secrets
import sqlite3
import stat
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from hermes_cli import kanban_db, projects_db
from hermes_cli.sqlite_util import write_txn
from plugins.dashboard_auth.raphael_workspace.model_policy import (
    validate_assignment as validate_raphael_model_assignment,
)

# fcntl is Unix-only; on Windows msvcrt provides the equivalent kernel file
# lock. Both are stdlib and both are enforced by the OS across processes —
# unlike the receipt lease, which only serializes claimants that share one
# projects.db. Absence of BOTH is fatal for the board guard (see
# :func:`_global_board_guard`), never a degraded no-op.
msvcrt = None
try:
    import fcntl
except ImportError:  # pragma: no cover - platform-specific fallback
    fcntl = None
    try:
        import msvcrt
    except ImportError:
        pass

_LOCK_TTL_SECONDS = 30
_POLL_INTERVAL_SECONDS = 0.02
_POLL_MAX_SECONDS = 8.0

_RECEIPTS_SCHEMA = """
CREATE TABLE IF NOT EXISTS owner_workspace_receipts (
    actor            TEXT NOT NULL,
    profile          TEXT NOT NULL,
    idempotency_key  TEXT NOT NULL,
    operation        TEXT NOT NULL,
    request_digest   TEXT NOT NULL,
    status           TEXT NOT NULL,
    lock_token       TEXT,
    lock_expires     INTEGER,
    project_id       TEXT,
    board_slug       TEXT,
    task_id          TEXT,
    result_json      TEXT,
    created_at       INTEGER NOT NULL,
    updated_at       INTEGER NOT NULL,
    PRIMARY KEY (actor, profile, idempotency_key)
)
"""


class OwnerWorkspaceError(Exception):
    """A validation/conflict/recovery failure the tool layer renders as an error."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass(frozen=True)
class OwnerContext:
    actor: str
    profile: str
    session: str


def resolve_owner_context() -> OwnerContext:
    """Derive the trusted actor/profile/session identity for the current call.

    Never accepts these from tool-call arguments — see module docstring.
    """
    from hermes_cli.profiles import get_active_profile_name
    from tools.approval import get_current_session_key

    profile = get_active_profile_name()
    session = get_current_session_key(default="")
    return OwnerContext(actor=profile, profile=profile, session=session)


def _ensure_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(_RECEIPTS_SCHEMA)


def _now() -> int:
    return int(time.time())


def _digest(payload: dict) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _require_str(value: Any, field: str) -> str:
    s = str(value).strip() if value is not None else ""
    if not s:
        raise OwnerWorkspaceError("invalid_argument", f"{field} is required")
    return s


def _get_receipt(conn: sqlite3.Connection, ctx: OwnerContext, key: str) -> Optional[sqlite3.Row]:
    return conn.execute(
        "SELECT * FROM owner_workspace_receipts "
        "WHERE actor = ? AND profile = ? AND idempotency_key = ?",
        (ctx.actor, ctx.profile, key),
    ).fetchone()


def _is_retryable_confirmation_timeout(row: sqlite3.Row) -> bool:
    """Return whether a terminal receipt records silence, not an owner denial."""
    if row["status"] != "denied" or not row["result_json"]:
        return False
    try:
        result = json.loads(row["result_json"])
    except (TypeError, ValueError):
        return False
    return (
        isinstance(result, dict)
        and result.get("ok") is False
        and result.get("error") == "confirmation_denied"
        and result.get("reason") == "timeout"
    )


def _claim_or_wait(
    conn: sqlite3.Connection, ctx: OwnerContext, key: str, operation: str, digest: str,
) -> tuple:
    """One short write_txn: claim a fresh key, adopt a dead claim, report a
    live claim to wait on, or report a terminal receipt to replay.

    Returns ``(state, row, token)`` where ``state`` is one of ``"own"``
    (caller now holds the lock, identified by ``token``; ``row`` is ``None``
    for a fresh claim or the prior in-progress row's recorded progress for an
    adopted one), ``"wait"`` (a live claim exists elsewhere; ``row`` is that
    row, ``token`` is ``None``), or ``"terminal"`` (``row`` already carries a
    committed/denied ``result_json``, ``token`` is ``None``).

    ``token`` is the freshly minted lock the caller now owns — every
    subsequent progress write, domain mutation, and finalization for this
    call MUST be predicated on it (see :func:`_assert_owns_lease`,
    :func:`_update_progress`, :func:`_finalize_receipt`). Adopting a dead
    claim always mints a NEW token here, so the crashed/stalled prior
    claimant's own (now stale) token can never again match this row.
    """
    token = secrets.token_hex(16)
    now = _now()
    with write_txn(conn):
        row = _get_receipt(conn, ctx, key)
        if row is None:
            conn.execute(
                "INSERT INTO owner_workspace_receipts "
                "(actor, profile, idempotency_key, operation, request_digest, status, "
                " lock_token, lock_expires, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, 'in_progress', ?, ?, ?, ?)",
                (ctx.actor, ctx.profile, key, operation, digest, token, now + _LOCK_TTL_SECONDS, now, now),
            )
            return ("own", None, token)
        if row["request_digest"] != digest:
            raise OwnerWorkspaceError(
                "idempotency_key_conflict",
                f"idempotency_key {key!r} was already used with a different request payload",
            )
        if row["operation"] != operation:
            raise OwnerWorkspaceError(
                "idempotency_key_conflict",
                f"idempotency_key {key!r} is already in use for a different operation",
            )
        if _is_retryable_confirmation_timeout(row):
            conn.execute(
                "UPDATE owner_workspace_receipts SET status = 'in_progress', "
                "lock_token = ?, lock_expires = ?, project_id = NULL, "
                "board_slug = NULL, task_id = NULL, result_json = NULL, updated_at = ? "
                "WHERE actor = ? AND profile = ? AND idempotency_key = ? AND status = 'denied'",
                (
                    token, now + _LOCK_TTL_SECONDS, now,
                    ctx.actor, ctx.profile, key,
                ),
            )
            return ("own", None, token)
        if row["status"] in ("committed", "denied"):
            return ("terminal", row, None)
        # status == "in_progress": a live claim blocks us; a dead one (lock
        # expired — the prior claimer crashed) is ours to adopt and roll
        # forward from its recorded progress.
        if row["lock_expires"] is not None and now < row["lock_expires"]:
            return ("wait", row, None)
        conn.execute(
            "UPDATE owner_workspace_receipts SET lock_token = ?, lock_expires = ?, updated_at = ? "
            "WHERE actor = ? AND profile = ? AND idempotency_key = ?",
            (token, now + _LOCK_TTL_SECONDS, now, ctx.actor, ctx.profile, key),
        )
        return ("own", row, token)


def _acquire_or_replay(
    conn: sqlite3.Connection, ctx: OwnerContext, key: str, operation: str, digest: str,
) -> tuple:
    """Bounded poll around :func:`_claim_or_wait` for the concurrent-caller case."""
    deadline = time.monotonic() + _POLL_MAX_SECONDS
    while True:
        state, row, token = _claim_or_wait(conn, ctx, key, operation, digest)
        if state != "wait":
            return (state, row, token)
        if time.monotonic() >= deadline:
            raise OwnerWorkspaceError(
                "in_progress_timeout",
                f"idempotency_key {key!r} is still being processed by another call",
            )
        time.sleep(_POLL_INTERVAL_SECONDS)


def _assert_owns_lease(conn: sqlite3.Connection, ctx: OwnerContext, key: str, token: str) -> None:
    """Re-read the receipt row and fail closed unless *token* still owns it.

    MUST be called from inside the SAME ``with write_txn(conn):`` fence that
    performs the domain mutation it guards (Project/board/Task creation, the
    CAS status move, readiness recompute, the comment insert) — never as a
    standalone check followed by a separately-committed mutation. Called on
    its own, this is just a point-in-time SELECT: a concurrent adopter could
    commit a new token in the gap between the read and the mutation it was
    meant to guard. Called inside the fence, the ``BEGIN IMMEDIATE`` already
    holds ``projects.db``'s write lock before this SELECT runs, so no
    adoption (which needs that same lock) can land until the fence's
    ``with`` block exits — closing exactly that gap. The receipts row is the
    ONLY place lease ownership is recorded, so this is what actually stops a
    claimant whose lease already expired and was adopted by someone else (a
    new token minted in :func:`_claim_or_wait`) from mutating domain state
    after the fact, even though the domain tables themselves (projects.db
    Project rows, kanban.db tasks) carry no lock_token of their own.
    """
    row = _get_receipt(conn, ctx, key)
    if row is None or row["status"] != "in_progress" or row["lock_token"] != token:
        raise OwnerWorkspaceError(
            "lease_lost",
            f"idempotency_key {key!r}'s claim lease was lost (expired and adopted by "
            "another caller); refusing to mutate domain state",
        )


def _update_progress(
    conn: sqlite3.Connection, ctx: OwnerContext, key: str, token: str, **fields: Any,
) -> None:
    if not fields:
        return
    sets = ", ".join(f"{k} = ?" for k in fields) + ", updated_at = ?"
    params = list(fields.values()) + [_now(), ctx.actor, ctx.profile, key, token]
    with write_txn(conn):
        cur = conn.execute(
            f"UPDATE owner_workspace_receipts SET {sets} "
            "WHERE actor = ? AND profile = ? AND idempotency_key = ? AND lock_token = ?",
            params,
        )
        if cur.rowcount != 1:
            raise OwnerWorkspaceError(
                "lease_lost",
                f"idempotency_key {key!r}'s claim lease was lost while writing progress; "
                "refusing to overwrite another claimant's state",
            )


def _finalize_receipt(
    conn: sqlite3.Connection, ctx: OwnerContext, key: str, token: str, *, status: str, result: dict,
) -> None:
    with write_txn(conn):
        cur = conn.execute(
            "UPDATE owner_workspace_receipts SET status = ?, result_json = ?, "
            "lock_token = NULL, lock_expires = NULL, updated_at = ? "
            "WHERE actor = ? AND profile = ? AND idempotency_key = ? AND lock_token = ?",
            (status, json.dumps(result, ensure_ascii=False), _now(), ctx.actor, ctx.profile, key, token),
        )
        if cur.rowcount != 1:
            raise OwnerWorkspaceError(
                "lease_lost",
                f"idempotency_key {key!r}'s claim lease was lost before finalization; "
                "refusing to finalize on another claimant's behalf",
            )


def _confirm(ctx: OwnerContext, *, operation: str, digest: str, description: str) -> dict:
    from tools.approval import request_exact_operation_approval

    if not ctx.session:
        return {"approved": False, "reason": "no_session"}
    return request_exact_operation_approval(
        ctx.session,
        operation=operation,
        payload_digest=digest,
        actor=ctx.actor,
        profile=ctx.profile,
        description=description,
    )


def _derive_id(ctx: OwnerContext, idempotency_key: str, salt: str) -> str:
    """Deterministic sub-identity, a pure function of (actor, profile,
    idempotency_key, salt).

    Recomputing this identically on every call — including a crash-and-retry
    — means the identity needed to find (or create) a durable object exists
    BEFORE the object does, with no separate progress record that a crash
    could leave unwritten. That closes the exact gap described in the module
    docstring's crash-safe bootstrap contract: there is no window between
    "the domain mutation committed" and "the pointer to it was persisted",
    because there never was a separate pointer to persist.
    """
    raw = f"{ctx.actor}:{ctx.profile}:{idempotency_key}:{salt}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


@contextmanager
def _global_board_guard(board_slug: str):
    """Hold an exclusive OS file lock on one GLOBAL board namespace.

    Kanban boards are shared across profiles by design
    (:func:`kanban_db.kanban_home` resolves the profile-INDEPENDENT root),
    while receipts and Project rows live in the caller's profile-local
    ``projects.db``. So the receipt lease — which only serializes claimants
    writing the same ``projects.db`` — cannot order two bootstraps run from
    DIFFERENT profiles: both derive the same board slug from the same name,
    both see "board does not exist", and both publish ``board.json``, so the
    second silently overwrites the first's ownership metadata. That race is
    invisible to every check in this module, because each check is itself
    correct in its own profile.

    The lock identity is therefore anchored to the contended resource, not to
    the caller: ``<kanban_home>/kanban/locks/board-<slug>.lock``, resolved
    through ``kanban_db`` so every competing profile computes the same path
    for the same canonical board namespace. ``fcntl.flock`` (``msvcrt.locking``
    on Windows) is a kernel-held lock: it serializes unrelated PROCESSES, and
    the kernel drops it when the holder exits, so a crash inside the critical
    section cannot wedge the namespace — no TTL, no daemon, no polling.

    Fails CLOSED: if the lock directory/file cannot be created, the lock
    cannot be acquired, or the platform offers neither primitive, the caller
    gets :class:`OwnerWorkspaceError` and performs no board mutation at all.
    """
    slug = kanban_db._normalize_board_slug(board_slug) or kanban_db.DEFAULT_BOARD
    fd = None
    try:
        if fcntl is None and msvcrt is None:  # pragma: no cover - exotic platform
            raise RuntimeError("no OS file-locking primitive is available")
        lock_dir = kanban_db.kanban_home() / "kanban" / "locks"
        lock_dir.mkdir(parents=True, exist_ok=True)
        lock_path = lock_dir / f"board-{slug}.lock"
        if msvcrt and (not lock_path.exists() or lock_path.stat().st_size == 0):
            lock_path.write_text(" ", encoding="utf-8")
        fd = open(lock_path, "r+" if msvcrt else "a+", encoding="utf-8")
        if fcntl:
            fcntl.flock(fd, fcntl.LOCK_EX)
        else:  # pragma: no cover - Windows-only path
            fd.seek(0)
            msvcrt.locking(fd.fileno(), msvcrt.LK_LOCK, 1)
    except OwnerWorkspaceError:
        if fd is not None:
            fd.close()
        raise
    except Exception as exc:
        if fd is not None:
            fd.close()
        raise OwnerWorkspaceError(
            "ownership_guard_unavailable",
            f"could not acquire the global board ownership guard for {slug!r}: {exc}",
        ) from exc

    try:
        yield
    finally:
        try:
            if fcntl:
                fcntl.flock(fd, fcntl.LOCK_UN)
            elif msvcrt:  # pragma: no cover - Windows-only path
                fd.seek(0)
                msvcrt.locking(fd.fileno(), msvcrt.LK_UNLCK, 1)
        except OSError:
            pass
        fd.close()


def _assert_board_ownership(board_slug: str, project_id: str) -> None:
    """Fail closed unless ``board_slug`` is absent or owned by ``project_id``.

    Strict equality, not truthiness: an existing board this flow did not
    create always carries its owning ``project_id``, so missing/empty/
    malformed metadata is exactly as unsafe to adopt as a foreign one.
    """
    if not kanban_db.board_exists(board_slug):
        return
    owner = kanban_db.read_board_metadata(board_slug).get("project_id")
    if owner != project_id:
        raise OwnerWorkspaceError(
            "crash_recovery_failed",
            f"board {board_slug!r} has missing or foreign ownership metadata "
            f"(project_id={owner!r}, expected {project_id!r}); refusing to "
            "adopt ambiguous board ownership",
        )


# ---------------------------------------------------------------------------
# owner_workspace_bootstrap
# ---------------------------------------------------------------------------


def bootstrap(
    ctx: OwnerContext, *, idempotency_key: str, name: str, description: Optional[str] = None,
) -> dict:
    """Create exactly one owner-owned Project + Kanban board + initial Task.

    Uses only ``projects.db`` (the Project row + this module's receipt
    table), ``board.json`` (board metadata, published via
    ``kanban_db.create_board`` — atomic fsync + ``os.replace`` write), and
    ``kanban.db`` (the board's task store). No caller-selected filesystem
    path: the Project is created with no folders.

    Crash-safe roll-forward: the Project id and the initial Task's
    idempotency key are both deterministic (:func:`_derive_id`), so a retry
    after a crash at ANY point recomputes the exact same identities and
    finds — never blindly recreates — whatever already committed. The board
    slug is never independently derived; it is read off the (uniquely
    resolved, by id) Project row, so replay can never adopt a board by
    matching on the human-provided ``name``. A board or task that already
    exists under a foreign OR missing/empty/malformed owner (anything other
    than an exact ``project_id`` match) fails closed instead of being
    silently adopted — objects this flow creates always carry exact
    ownership metadata, so an existing object without it can never be safely
    attributed to this receipt. Because boards are GLOBAL while receipts are
    profile-local, that ownership check/create/bind decision additionally runs
    inside :func:`_global_board_guard` — a kernel-held file lock on the board
    namespace, shared by every profile — and is revalidated inside it, so two
    same-name bootstraps from different profiles resolve to exactly one owner
    and one ownership conflict. Every domain mutation (Project row, board,
    Task) is fenced: the immediately-preceding lease check and the mutation
    itself share one held ``projects.db`` write lock (see
    :func:`_assert_owns_lease`), so a lease takeover can never land in the
    gap between "we validated ownership" and "we mutated".
    """
    idempotency_key = _require_str(idempotency_key, "idempotency_key")
    name = _require_str(name, "name")
    description = str(description).strip() or None if description else None

    payload = {"name": name, "description": description}
    digest = _digest(payload)
    operation = "owner_workspace_bootstrap"

    pconn = projects_db.connect()
    try:
        _ensure_schema(pconn)
        state, row, token = _acquire_or_replay(pconn, ctx, idempotency_key, operation, digest)
        if state == "terminal":
            return json.loads(row["result_json"])

        approval = _confirm(
            ctx, operation=operation, digest=digest,
            description=f"Bootstrap owner workspace project {name!r}",
        )
        if not approval.get("approved"):
            result = {"ok": False, "error": "confirmation_denied", "reason": approval.get("reason")}
            _finalize_receipt(pconn, ctx, idempotency_key, token, status="denied", result=result)
            return result

        project_id = "p_" + _derive_id(ctx, idempotency_key, "project")

        # Fence 1: lease validation + Project-row creation are BOTH statements
        # against pconn (projects.db), so this is a single real SQLite
        # transaction — genuinely atomic, not just sequential. `write_txn` is
        # reentrant, so `create_project`'s own internal `write_txn(pconn)`
        # joins this one instead of racing it.
        with write_txn(pconn):
            _assert_owns_lease(pconn, ctx, idempotency_key, token)
            project = projects_db.get_project(pconn, project_id)
            if project is None:
                created_id = projects_db.create_project(
                    pconn, id=project_id, name=name, description=description,
                )
                if created_id != project_id:
                    raise OwnerWorkspaceError(
                        "internal_error", "create_project did not honor the requested id",
                    )
                project = projects_db.get_project(pconn, project_id)

        # Never adopt by display name: board_slug comes from THIS project's
        # own persisted slug (looked up by the deterministic id above), never
        # re-derived from `name` or searched for.
        board_slug = project.slug

        # Cheap fail-fast check outside the guard — an already-foreign board
        # never becomes adoptable by waiting for a lock. It is NOT the
        # decision: the authoritative check is re-run inside the guard below.
        _assert_board_ownership(board_slug, project_id)

        # Guard 1 (cross-profile, kernel-held): the ownership CHECK, the board
        # CREATE and the project→board BIND are one indivisible decision over
        # the global board namespace, so a same-name bootstrap from another
        # profile's projects.db cannot slip between them and overwrite this
        # board's ownership metadata (see :func:`_global_board_guard`).
        # Acquired OUTSIDE the projects.db write lock, and only ever in that
        # order, so the two locks cannot deadlock against each other.
        with _global_board_guard(board_slug):
            # Revalidate under the guard: whatever we observed above may have
            # been published by a competitor between the check and the lock.
            # This is the ownership decision that counts.
            _assert_board_ownership(board_slug, project_id)
            # Fence 2: the lease check and the create_board/update_project pair
            # share one held write lock on pconn — create_board publishes
            # board.json (a resource outside projects.db, so this cannot be one
            # cross-database transaction), but no competing claimant can adopt
            # this row (which requires the same pconn write lock) while it is
            # open, so a takeover can never land strictly between "we own the
            # lease" and "the board/project-row mutation actually happened".
            with write_txn(pconn):
                _assert_owns_lease(pconn, ctx, idempotency_key, token)
                # create_board is naturally idempotent ("mkdir -p" semantics — see
                # its docstring), so no separate "already created?" progress
                # check is needed: replaying this exact call is always safe.
                published = kanban_db.create_board(
                    board_slug, name=name, description=description, project_id=project_id,
                )
                # create_board STAMPS ownership (write_board_metadata
                # overwrites project_id) — which is exactly why the guarded
                # revalidation above must have already established that this
                # slug is ours or free. Verify what actually landed on disk
                # before binding the project row to it, so any normalization
                # or partial write fails closed instead of leaving a project
                # bound to a board it does not own.
                if published.get("project_id") != project_id:
                    raise OwnerWorkspaceError(
                        "crash_recovery_failed",
                        f"board {board_slug!r} published ownership "
                        f"{published.get('project_id')!r}, not {project_id!r}; "
                        "refusing to bind a board this project does not own",
                    )
                if project.board_slug != board_slug:
                    projects_db.update_project(pconn, project_id, board_slug=board_slug)

        task_idempotency_key = "owtask_" + _derive_id(ctx, idempotency_key, "task")
        kconn = kanban_db.connect(board=board_slug)
        try:
            # Fence 3: same pattern — the lease check and the Task creation
            # (kanban.db) are covered by one held pconn write lock.
            with write_txn(pconn):
                _assert_owns_lease(pconn, ctx, idempotency_key, token)
                # create_task's own idempotency_key dedupe (kanban_db.py)
                # returns the existing task on replay instead of creating a
                # second one.
                task_id = kanban_db.create_task(
                    kconn,
                    title=name,
                    body=description,
                    created_by=ctx.actor,
                    board=board_slug,
                    project_id=project_id,
                    idempotency_key=task_idempotency_key,
                )
                task = kanban_db.get_task(kconn, task_id)
                if task is None:
                    raise OwnerWorkspaceError(
                        "crash_recovery_failed", f"created/adopted task_id {task_id!r} does not resolve",
                    )
                # Strict equality — see the board-ownership check above: a
                # pre-existing/replayed task with missing/empty ownership is
                # just as unsafe to adopt as one owned by a foreign project.
                if task.project_id != project_id:
                    raise OwnerWorkspaceError(
                        "crash_recovery_failed",
                        f"task {task_id!r} has missing or foreign ownership "
                        f"(project_id={task.project_id!r}, expected {project_id!r}); "
                        "refusing to adopt ambiguous task ownership",
                    )
                task_revision = kanban_db.task_event_revision(kconn, task_id)
        finally:
            kconn.close()

        result = {
            "ok": True, "project_id": project_id, "board": board_slug, "task_id": task_id,
            "status": task.status, "revision": task_revision,
        }
        _finalize_receipt(pconn, ctx, idempotency_key, token, status="committed", result=result)
        return result
    finally:
        pconn.close()



# ---------------------------------------------------------------------------
# owner_task_graph_commit
# ---------------------------------------------------------------------------

_MAX_GRAPH_TASKS = 12
_MAX_LATER_MILESTONES = 12


def _bounded_text(value: Any, field: str, *, limit: int, required: bool = True) -> Optional[str]:
    text = str(value).strip() if value is not None else ""
    if required and not text:
        raise OwnerWorkspaceError("invalid_argument", f"{field} is required")
    if not text:
        return None
    if len(text) > limit:
        raise OwnerWorkspaceError(
            "invalid_argument", f"{field} must be at most {limit} characters",
        )
    return text


def _normalize_graph_tasks(tasks: Any) -> list[dict]:
    """Validate one executable milestone, never a fake whole-project backlog."""
    if not isinstance(tasks, list) or not tasks:
        raise OwnerWorkspaceError("invalid_argument", "tasks must be a non-empty list")
    if len(tasks) > _MAX_GRAPH_TASKS:
        raise OwnerWorkspaceError(
            "milestone_too_large",
            f"tasks may contain at most {_MAX_GRAPH_TASKS} items; split large projects "
            "into Now / Next / Later and commit only the executable milestone",
        )

    from agent.redact import redact_sensitive_text
    from hermes_cli.profiles import normalize_profile_name, profile_exists

    normalized: list[dict] = []
    for index, raw in enumerate(tasks):
        if not isinstance(raw, dict):
            raise OwnerWorkspaceError("invalid_argument", f"tasks[{index}] must be an object")
        title = _bounded_text(raw.get("title"), f"tasks[{index}].title", limit=240)
        body = _bounded_text(raw.get("body"), f"tasks[{index}].body", limit=12_000)
        assignee_raw = _bounded_text(
            raw.get("assignee"), f"tasks[{index}].assignee", limit=64,
        )
        try:
            responsibility = kanban_db.normalize_responsibility(
                raw.get("responsibility")
            )
        except ValueError as exc:
            raise OwnerWorkspaceError("invalid_argument", str(exc)) from exc
        try:
            assignee = normalize_profile_name(assignee_raw)
        except ValueError as exc:
            raise OwnerWorkspaceError("invalid_assignee", str(exc)) from exc
        if not profile_exists(assignee):
            raise OwnerWorkspaceError(
                "invalid_assignee",
                f"tasks[{index}].assignee names an unavailable profile: {assignee!r}",
            )

        parents = raw.get("parents", [])
        if not isinstance(parents, list):
            raise OwnerWorkspaceError(
                "invalid_argument", f"tasks[{index}].parents must be a list",
            )
        clean_parents: list[int] = []
        for parent in parents:
            if isinstance(parent, bool) or not isinstance(parent, int):
                raise OwnerWorkspaceError(
                    "invalid_argument",
                    f"tasks[{index}].parents must contain only task indices",
                )
            if parent < 0 or parent >= len(tasks) or parent == index:
                raise OwnerWorkspaceError(
                    "invalid_argument",
                    f"tasks[{index}].parents contains invalid index {parent}",
                )
            if parent not in clean_parents:
                clean_parents.append(parent)

        normalized.append({
            "title": redact_sensitive_text(title, force=True),
            "body": redact_sensitive_text(body, force=True),
            "assignee": assignee,
            "responsibility": responsibility,
            "parents": clean_parents,
        })

    # Reject the whole request before approval or persistence when sibling
    # dependencies cycle. decompose_triage_task repeats this at the DB boundary.
    indegree = [0] * len(normalized)
    edges: list[list[int]] = [[] for _ in normalized]
    for child_index, task in enumerate(normalized):
        for parent_index in task["parents"]:
            edges[parent_index].append(child_index)
            indegree[child_index] += 1
    queue = [index for index, degree in enumerate(indegree) if degree == 0]
    seen = 0
    while queue:
        node = queue.pop()
        seen += 1
        for child in edges[node]:
            indegree[child] -= 1
            if indegree[child] == 0:
                queue.append(child)
    if seen != len(normalized):
        raise OwnerWorkspaceError("invalid_graph", "tasks contain a cyclic dependency")

    return normalized


def _normalize_later_milestones(value: Any) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise OwnerWorkspaceError(
            "invalid_argument", "later_milestones must be a list",
        )
    if len(value) > _MAX_LATER_MILESTONES:
        raise OwnerWorkspaceError(
            "invalid_argument",
            f"later_milestones may contain at most {_MAX_LATER_MILESTONES} items",
        )
    from agent.redact import redact_sensitive_text

    return [
        redact_sensitive_text(
            _bounded_text(item, f"later_milestones[{index}]", limit=500),
            force=True,
        )
        for index, item in enumerate(value)
    ]


def _normalize_graph_assignee(value: Any, field: str) -> str:
    from hermes_cli.profiles import normalize_profile_name, profile_exists

    raw = _bounded_text(value, field, limit=64)
    try:
        name = normalize_profile_name(raw)
    except ValueError as exc:
        raise OwnerWorkspaceError("invalid_assignee", str(exc)) from exc
    if not profile_exists(name):
        raise OwnerWorkspaceError(
            "invalid_assignee", f"{field} names an unavailable profile: {name!r}",
        )
    return name


def _render_graph_root_body(
    *, specification: str, current_milestone: str,
    owner_visible_result: str, later_milestones: list[str],
) -> str:
    parts = [
        "Specification",
        specification,
        "",
        "Current milestone",
        current_milestone,
        "",
        "Owner-visible result",
        owner_visible_result,
    ]
    if later_milestones:
        parts.extend(["", "Later roadmap"])
        parts.extend(f"- {item}" for item in later_milestones)
    return "\n".join(parts)


def _ensure_graph_project_board(
    pconn: sqlite3.Connection,
    ctx: OwnerContext,
    idempotency_key: str,
    token: str,
    *,
    project_id: str,
    create: bool,
    name: Optional[str] = None,
    description: Optional[str] = None,
):
    """Resolve/create one Project and bind its exact owner-stamped native board."""
    with write_txn(pconn):
        _assert_owns_lease(pconn, ctx, idempotency_key, token)
        project = projects_db.get_project(pconn, project_id)
        if project is None:
            if not create:
                raise OwnerWorkspaceError(
                    "project_not_found", f"no such project {project_id!r}",
                )
            created_id = projects_db.create_project(
                pconn, id=project_id, name=name, description=description,
            )
            if created_id != project_id:
                raise OwnerWorkspaceError(
                    "internal_error", "create_project did not honor the requested id",
                )
            project = projects_db.get_project(pconn, project_id)
        elif create and (
            project.name != name or (project.description or None) != (description or None)
        ):
            raise OwnerWorkspaceError(
                "crash_recovery_failed",
                f"project {project_id!r} exists with different canonical fields",
            )
        if project is None or project.archived:
            raise OwnerWorkspaceError(
                "project_not_found", f"project {project_id!r} is unavailable",
            )

    board_slug = project.board_slug or project.slug
    _assert_board_ownership(board_slug, project_id)
    with _global_board_guard(board_slug):
        _assert_board_ownership(board_slug, project_id)
        with write_txn(pconn):
            _assert_owns_lease(pconn, ctx, idempotency_key, token)
            published = kanban_db.create_board(
                board_slug,
                name=project.name,
                description=project.description,
                project_id=project_id,
            )
            if published.get("project_id") != project_id:
                raise OwnerWorkspaceError(
                    "crash_recovery_failed",
                    f"board {board_slug!r} did not retain exact project ownership",
                )
            if project.board_slug != board_slug:
                projects_db.update_project(pconn, project_id, board_slug=board_slug)

    project = projects_db.get_project(pconn, project_id)
    return project, board_slug


def _committed_graph_children(
    kconn: sqlite3.Connection,
    *,
    root_task_id: str,
    digest: str,
    idempotency_key: str,
    ctx: OwnerContext,
) -> Optional[list[str]]:
    """Recognize only this receipt's already-committed atomic decomposition."""
    for event in kanban_db.list_events(kconn, root_task_id):
        payload = event.payload or {}
        if (
            event.kind == "decomposed"
            and payload.get("owner_task_graph_digest") == digest
            and payload.get("idempotency_key") == idempotency_key
            and payload.get("actor") == ctx.actor
            and payload.get("profile") == ctx.profile
        ):
            child_ids = payload.get("child_ids")
            if not isinstance(child_ids, list) or not all(
                isinstance(child_id, str) and child_id for child_id in child_ids
            ):
                raise OwnerWorkspaceError(
                    "crash_recovery_failed",
                    "the committed task-graph event contains invalid child identities",
                )
            return child_ids
    return None


def commit_task_graph(
    ctx: OwnerContext,
    *,
    idempotency_key: str,
    mode: str,
    request_title: str,
    specification: str,
    current_milestone: str,
    owner_visible_result: str,
    root_assignee: str,
    tasks: Any,
    later_milestones: Any = None,
    project_name: Optional[str] = None,
    project_description: Optional[str] = None,
    project_id: Optional[str] = None,
) -> dict:
    """Commit one approved Conversation proposal to native Project + Task state.

    Large goals are intentionally tranche-based: at most twelve executable
    child Tasks are persisted now; future milestones remain visible in the root
    specification and are reconsidered from facts inside the Project chat.
    """
    from agent.redact import redact_sensitive_text

    idempotency_key = _bounded_text(
        idempotency_key, "idempotency_key", limit=200,
    )
    mode = str(mode or "").strip().lower()
    if mode not in {"new", "existing"}:
        raise OwnerWorkspaceError("invalid_argument", "mode must be 'new' or 'existing'")

    request_title = redact_sensitive_text(
        _bounded_text(request_title, "request_title", limit=240), force=True,
    )
    specification = redact_sensitive_text(
        _bounded_text(specification, "specification", limit=20_000), force=True,
    )
    current_milestone = redact_sensitive_text(
        _bounded_text(current_milestone, "current_milestone", limit=1_000), force=True,
    )
    owner_visible_result = redact_sensitive_text(
        _bounded_text(owner_visible_result, "owner_visible_result", limit=1_000),
        force=True,
    )
    root_assignee = _normalize_graph_assignee(root_assignee, "root_assignee")
    normalized_tasks = _normalize_graph_tasks(tasks)
    normalized_later = _normalize_later_milestones(later_milestones)

    if mode == "new":
        if project_id is not None and str(project_id).strip():
            raise OwnerWorkspaceError(
                "invalid_argument", "project_id is not accepted when mode is 'new'",
            )
        project_name = redact_sensitive_text(
            _bounded_text(project_name, "project_name", limit=160), force=True,
        )
        project_description = _bounded_text(
            project_description, "project_description", limit=2_000, required=False,
        )
        if project_description:
            project_description = redact_sensitive_text(project_description, force=True)
        canonical_project_id = "p_" + _derive_id(ctx, idempotency_key, "graph-project")
    else:
        if project_name is not None and str(project_name).strip():
            raise OwnerWorkspaceError(
                "invalid_argument",
                "project_name is not accepted when mode is 'existing'",
            )
        canonical_project_id = _bounded_text(project_id, "project_id", limit=100)
        project_name = None
        project_description = None

    payload = {
        "mode": mode,
        "project_id": canonical_project_id if mode == "existing" else None,
        "project_name": project_name,
        "project_description": project_description,
        "request_title": request_title,
        "specification": specification,
        "current_milestone": current_milestone,
        "owner_visible_result": owner_visible_result,
        "root_assignee": root_assignee,
        "tasks": normalized_tasks,
        "later_milestones": normalized_later,
    }
    digest = _digest(payload)
    operation = "owner_task_graph_commit"

    pconn = projects_db.connect()
    try:
        _ensure_schema(pconn)
        state, row, token = _acquire_or_replay(
            pconn, ctx, idempotency_key, operation, digest,
        )
        if state == "terminal":
            return json.loads(row["result_json"])

        approval = _confirm(
            ctx,
            operation=operation,
            digest=digest,
            description=(
                f"Create project {project_name!r} with {len(normalized_tasks)} tasks"
                if mode == "new"
                else f"Add {len(normalized_tasks)} tasks to project {canonical_project_id!r}"
            ),
        )
        if not approval.get("approved"):
            result = {
                "ok": False,
                "error": "confirmation_denied",
                "reason": approval.get("reason"),
            }
            _finalize_receipt(
                pconn, ctx, idempotency_key, token, status="denied", result=result,
            )
            return result

        project, board_slug = _ensure_graph_project_board(
            pconn,
            ctx,
            idempotency_key,
            token,
            project_id=canonical_project_id,
            create=(mode == "new"),
            name=project_name,
            description=project_description,
        )
        _update_progress(
            pconn,
            ctx,
            idempotency_key,
            token,
            project_id=canonical_project_id,
            board_slug=board_slug,
        )

        root_body = _render_graph_root_body(
            specification=specification,
            current_milestone=current_milestone,
            owner_visible_result=owner_visible_result,
            later_milestones=normalized_later,
        )
        root_key = "owgraph_" + _derive_id(ctx, idempotency_key, "graph-root")
        kconn = kanban_db.connect(board=board_slug)
        try:
            with write_txn(pconn):
                _assert_owns_lease(pconn, ctx, idempotency_key, token)
                root_task_id = kanban_db.create_task(
                    kconn,
                    title=request_title,
                    body=root_body,
                    created_by=ctx.actor,
                    triage=True,
                    board=board_slug,
                    project_id=canonical_project_id,
                    idempotency_key=root_key,
                )
                root = kanban_db.get_task(kconn, root_task_id)
                if (
                    root is None
                    or root.project_id != canonical_project_id
                    or root.idempotency_key != root_key
                    or root.title != request_title
                    or root.body != root_body
                ):
                    raise OwnerWorkspaceError(
                        "crash_recovery_failed",
                        "the task-graph root does not match the approved proposal",
                    )

            with write_txn(pconn):
                _assert_owns_lease(pconn, ctx, idempotency_key, token)
                child_ids = _committed_graph_children(
                    kconn,
                    root_task_id=root_task_id,
                    digest=digest,
                    idempotency_key=idempotency_key,
                    ctx=ctx,
                )
                if child_ids is None:
                    child_ids = kanban_db.decompose_triage_task(
                        kconn,
                        root_task_id,
                        root_assignee=root_assignee,
                        children=normalized_tasks,
                        author=ctx.actor,
                        auto_promote=True,
                        event_metadata={
                            "owner_task_graph_digest": digest,
                            "idempotency_key": idempotency_key,
                            "actor": ctx.actor,
                            "profile": ctx.profile,
                        },
                    )
                if child_ids is None or len(child_ids) != len(normalized_tasks):
                    raise OwnerWorkspaceError(
                        "crash_recovery_failed",
                        "the approved task graph could not be committed or recovered",
                    )
                for child_id in child_ids:
                    child = kanban_db.get_task(kconn, child_id)
                    if child is None or child.project_id != canonical_project_id:
                        raise OwnerWorkspaceError(
                            "crash_recovery_failed",
                            f"child task {child_id!r} lost exact project ownership",
                        )

            root = kanban_db.get_task(kconn, root_task_id)
            task_rows = [kanban_db.get_task(kconn, child_id) for child_id in child_ids]
            result = {
                "ok": True,
                "mode": mode,
                "project_id": canonical_project_id,
                "project_slug": project.slug,
                "board": board_slug,
                "root_task_id": root_task_id,
                "root_status": root.status,
                "task_ids": child_ids,
                "task_statuses": [task.status for task in task_rows],
                "task_count": len(child_ids),
            }
        finally:
            kconn.close()

        # One milestone approval is the execution authority. Activation is
        # idempotent and preserves an explicit owner pause, so a crash/retry
        # cannot silently resume a Project the owner stopped.
        _set_project_dispatch_state(board_slug, enabled=True)

        _update_progress(
            pconn, ctx, idempotency_key, token, task_id=root_task_id,
        )
        _finalize_receipt(
            pconn, ctx, idempotency_key, token, status="committed", result=result,
        )
        return result
    finally:
        pconn.close()


def _receipt_owns_project(
    conn: sqlite3.Connection, ctx: OwnerContext, project_id: str,
) -> bool:
    """Return whether this trusted owner has a committed Project receipt."""
    rows = conn.execute(
        "SELECT result_json FROM owner_workspace_receipts "
        "WHERE actor = ? AND profile = ? AND status = 'committed' "
        "AND operation IN ('owner_workspace_bootstrap', 'owner_task_graph_commit')",
        (ctx.actor, ctx.profile),
    ).fetchall()
    for row in rows:
        try:
            if json.loads(row["result_json"]).get("project_id") == project_id:
                return True
        except Exception:
            continue
    return False


def _set_project_dispatch_state(
    board_slug: str,
    *,
    enabled: Optional[bool] = None,
    paused_by_owner: Optional[bool] = None,
) -> dict:
    """Serialize owner execution state with the board's dispatch claim."""
    try:
        return kanban_db.write_board_dispatch_state(
            board_slug,
            dispatch_enabled=enabled,
            dispatch_paused_by_owner=paused_by_owner,
        )
    except (OSError, TimeoutError) as exc:
        raise OwnerWorkspaceError(
            "execution_state_busy",
            "the Project execution state could not be changed just now",
        ) from exc


def list_committed_projects(ctx: OwnerContext) -> list[dict]:
    """Read-only projection of projects proven by committed owner receipts."""
    path = projects_db.projects_db_path()
    if not path.is_file():
        return []

    try:
        conn = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
    except sqlite3.Error:
        return []

    try:
        tables = {
            row["name"]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table' "
                "AND name IN ('projects', 'owner_workspace_receipts')"
            )
        }
        if tables != {"projects", "owner_workspace_receipts"}:
            return []
        receipts = conn.execute(
            "SELECT result_json FROM owner_workspace_receipts "
            "WHERE actor = ? AND profile = ? AND status = 'committed' "
            "AND operation IN ('owner_workspace_bootstrap', 'owner_task_graph_commit') "
            "ORDER BY updated_at ASC",
            (ctx.actor, ctx.profile),
        ).fetchall()

        project_ids: list[str] = []
        for receipt in receipts:
            try:
                project_id = str(json.loads(receipt["result_json"]).get("project_id") or "")
            except Exception:
                continue
            if project_id and project_id not in project_ids:
                project_ids.append(project_id)

        projects: list[dict] = []
        for project_id in project_ids:
            row = conn.execute(
                "SELECT id, slug, name, description, board_slug, archived "
                "FROM projects WHERE id = ?",
                (project_id,),
            ).fetchone()
            if row is None or not row["board_slug"]:
                continue
            try:
                _assert_board_ownership(row["board_slug"], project_id)
            except OwnerWorkspaceError:
                continue
            projects.append({
                "project_id": row["id"],
                "slug": row["slug"],
                "name": owner_project_name(row["name"]),
                "description": row["description"],
                "board": row["board_slug"],
                "archived": bool(row["archived"]),
            })
        return projects
    finally:
        conn.close()


_OWNER_PROJECT_COLUMNS = (
    "triage", "todo", "scheduled", "ready", "running", "blocked", "review", "done",
)
_OWNER_PROJECT_MAX_TASKS = 200
_OWNER_PROJECT_MAX_ATTACHMENTS = 200
_OWNER_PROJECT_MAX_RUNS = 50
_OWNER_PROJECT_MAX_WORKERS = 100

# The named response capability a snapshot reader opts into to receive a run's
# own sanitized task title and its per-exact-task-id retry fact.
#
# The default projection stays the three-key run shape the first owner
# Workspace release validates as a closed schema, so deploying this Hermes
# ahead of the Workspace that understands the added keys cannot make a
# snapshot unreadable. Only a client that asks for this capability by name
# receives the wider run shape.
OWNER_PROJECT_RUN_CONTEXT_CAPABILITY = "run_task_context"
_OWNER_PROJECT_MEDIA_TYPE_RE = re.compile(
    r"^[A-Za-z0-9!#$&^_.+-]+/[A-Za-z0-9!#$&^_.+-]+$"
)
_OWNER_RUNTIME_VALUE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,127}$")
_OWNER_UNKNOWN_RUNTIME = {
    "state": "unknown",
    "summary": "This record does not contain an authoritative model route.",
}
_OWNER_UNKNOWN_COST = {
    "state": "unknown",
    "summary": "This record does not contain an authoritative cost.",
}


def _owner_project_attachment_projection(row: sqlite3.Row) -> dict:
    filename = re.sub(
        r"[^A-Za-z0-9._ -]", "_", Path(str(row["filename"] or "attachment")).name
    ).strip() or "attachment"
    media_type = str(row["content_type"] or "").strip()
    if not _OWNER_PROJECT_MEDIA_TYPE_RE.fullmatch(media_type):
        media_type = "application/octet-stream"
    return {
        "id": str(row["id"]),
        "filename": filename,
        "media_type": media_type,
        "size": int(row["size"] or 0),
        "created_at": _owner_timestamp(row["created_at"]),
    }


def _owner_runtime_value(value: object) -> Optional[str]:
    if not isinstance(value, str):
        return None
    clean = value.strip()
    return clean if _OWNER_RUNTIME_VALUE_RE.fullmatch(clean) else None


def _owner_project_runtime_and_cost(run: kanban_db.Run) -> tuple[dict, dict]:
    metadata = run.metadata if isinstance(run.metadata, dict) else {}
    raw = metadata.get("runtime_receipt")
    if not isinstance(raw, dict) or raw.get("schema_version") != 2:
        return dict(_OWNER_UNKNOWN_RUNTIME), dict(_OWNER_UNKNOWN_COST)

    engine = _owner_runtime_value(raw.get("engine"))
    profile = _owner_runtime_value(raw.get("profile"))
    provider = _owner_runtime_value(raw.get("provider"))
    model = _owner_runtime_value(raw.get("model"))
    effort = _owner_runtime_value(raw.get("reasoning_effort"))
    evidence = _owner_runtime_value(raw.get("route_evidence"))
    if (
        engine != "hermes"
        or profile != run.profile
        or not provider
        or not model
        or not effort
        or evidence not in {"dominant-session-usage", "session-row"}
    ):
        return dict(_OWNER_UNKNOWN_RUNTIME), dict(_OWNER_UNKNOWN_COST)
    try:
        validate_raphael_model_assignment(
            profile,
            provider,
            model,
            effort,
            disable_fallbacks=True,
        )
    except ValueError:
        return dict(_OWNER_UNKNOWN_RUNTIME), dict(_OWNER_UNKNOWN_COST)

    runtime = {
        "state": "known",
        "engine": engine,
        "profile": profile,
        "provider": provider,
        "model": model,
        "reasoning_effort": effort,
    }
    cost = raw.get("cost")
    if not isinstance(cost, dict):
        return runtime, dict(_OWNER_UNKNOWN_COST)
    state = _owner_runtime_value(cost.get("state"))
    currency = _owner_runtime_value(cost.get("currency"))
    scope = _owner_runtime_value(cost.get("scope"))
    amount = cost.get("amount")
    if (
        state not in {"estimated", "exact", "reported", "included"}
        or currency != "USD"
        or scope != "dominant-main-route"
        or isinstance(amount, bool)
        or not isinstance(amount, (int, float))
        or not math.isfinite(float(amount))
        or float(amount) < 0
        or float(amount) > 1_000_000
    ):
        return runtime, dict(_OWNER_UNKNOWN_COST)
    summary = {
        "estimated": "Estimated model usage for this recorded route.",
        "exact": "Recorded model usage cost for this route.",
        "reported": "Provider-reported model usage cost for this route.",
        "included": "Included in the connected provider plan.",
    }[state]
    return runtime, {
        "state": state,
        "currency": "USD",
        "amount": round(float(amount), 8),
        "summary": summary,
    }


def _owner_project_run_receipt(run: kanban_db.Run) -> dict:
    outcome = (run.outcome or run.status or "").strip().lower()
    if run.status == "running":
        owner_outcome, summary = "running", "Work is still in progress."
    elif outcome in {"completed", "done"}:
        owner_outcome, summary = "completed", "Work finished."
    elif outcome == "review_requested":
        owner_outcome, summary = "completed", "Work finished and is awaiting review."
    elif outcome == "scheduled":
        owner_outcome, summary = "unknown", "Work is scheduled for later."
    elif outcome in {
        "blocked", "changes_requested", "crashed", "gave_up", "rate_limited",
        "reclaimed", "spawn_failed", "stale", "timed_out",
    }:
        owner_outcome, summary = "attention", "Work stopped and needs attention."
    else:
        owner_outcome, summary = "unknown", "The final outcome could not be confirmed."
    runtime, cost = _owner_project_runtime_and_cost(run)
    return {
        "outcome": owner_outcome,
        "summary": summary,
        "external_effect": {
            "state": "unknown",
            "summary": "This record does not confirm whether an external service changed.",
        },
        "runtime": runtime,
        "cost": cost,
        "evidence": {"state": "available", "kind": "project_activity"},
    }


def _owner_project_run_projection(
    run: kanban_db.Run, task_title: Any, *, has_newer_run: bool, run_context: bool,
) -> dict:
    """Project one run for the owner, carrying its retry fact but never its id.

    ``has_newer_run`` is decided by the caller from the exact native task id
    while it walks the newest-first run list — never from ``task_title``,
    which two genuinely distinct tasks may legitimately sanitize to. The id
    itself never crosses this boundary, so this boolean is the only way a
    consumer can know a later attempt at the same work exists.

    Both of those keys are withheld unless ``run_context`` is set: they are
    served only to a reader that named
    ``OWNER_PROJECT_RUN_CONTEXT_CAPABILITY``, so an older reader keeps the
    exact run shape it validates as a closed schema.
    """
    projection = {
        "started_at": _owner_timestamp(run.started_at),
        "finished_at": _owner_timestamp(run.ended_at),
        "receipt": _owner_project_run_receipt(run),
    }
    if not run_context:
        return projection
    return {
        "task_title": owner_title(task_title),
        "has_newer_run": bool(has_newer_run),
        **projection,
    }


def read_project_snapshot(
    ctx: OwnerContext, project_slug: str, *, run_context: bool = False,
) -> dict:
    """Return one bounded read-only surface for an exact receipt-owned Project.

    The caller cannot select a board independently.  The Project slug must be
    present in this owner/profile's committed receipts, and its persisted board
    binding must still pass the shared-board ownership check.  Only fields
    already used by the owner Workspace cross this boundary: task bodies,
    results, paths, branches, logs, comments, raw events and model metadata do
    not.

    ``run_context`` adds each run's own sanitized task title and retry fact.
    It defaults off so the shape stays exactly what the oldest owner
    Workspace release accepts; see
    ``OWNER_PROJECT_RUN_CONTEXT_CAPABILITY``.
    """
    try:
        project_slug = projects_db.normalize_slug(project_slug) or ""
    except ValueError as exc:
        raise OwnerWorkspaceError("project_not_found", "the Project is unavailable") from exc

    project = next(
        (item for item in list_committed_projects(ctx) if item["slug"] == project_slug),
        None,
    )
    if project is None or project["archived"]:
        raise OwnerWorkspaceError("project_not_found", "the Project is unavailable")

    project_id = str(project["project_id"])
    board_slug = str(project["board"])
    try:
        _assert_board_ownership(board_slug, project_id)
        metadata = kanban_db.read_board_metadata(board_slug)
    except (OSError, TypeError, ValueError, OwnerWorkspaceError) as exc:
        raise OwnerWorkspaceError(
            "snapshot_unavailable", "the Project board ownership could not be verified"
        ) from exc
    if bool(metadata.get("archived")):
        raise OwnerWorkspaceError("project_not_found", "the Project is unavailable")

    board_path = (
        kanban_db.kanban_home() / "kanban.db"
        if board_slug == kanban_db.DEFAULT_BOARD
        else kanban_db.board_dir(board_slug) / "kanban.db"
    )
    conn = _open_read_only_sqlite(board_path, label="kanban.db")
    try:
        task_rows = conn.execute(
            "SELECT * FROM tasks WHERE project_id = ? AND task_kind = 'work' "
            "AND status != 'archived' ORDER BY priority DESC, created_at ASC, id ASC LIMIT ?",
            (project_id, _OWNER_PROJECT_MAX_TASKS + 1),
        ).fetchall()
        tasks = [kanban_db.Task.from_row(row) for row in task_rows[:_OWNER_PROJECT_MAX_TASKS]]
        task_ids = [task.id for task in tasks]
        visible = set(task_ids)

        counts = {status: 0 for status in (*_OWNER_PROJECT_COLUMNS, "archived")}
        for row in conn.execute(
            "SELECT status, COUNT(*) AS n FROM tasks "
            "WHERE project_id = ? AND task_kind = 'work' GROUP BY status",
            (project_id,),
        ):
            status = str(row["status"])
            if status not in counts:
                raise OwnerWorkspaceError("snapshot_unavailable", "kanban.db contains an unknown status")
            counts[status] = int(row["n"])

        event_state: dict[str, dict[str, int]] = {}
        if task_ids:
            placeholders = ",".join("?" for _ in task_ids)
            for row in conn.execute(
                f"SELECT task_id, MAX(created_at) AS latest, MAX(id) AS revision "
                f"FROM task_events WHERE task_id IN ({placeholders}) GROUP BY task_id",
                task_ids,
            ):
                event_state[str(row["task_id"])] = {
                    "latest": int(row["latest"]), "revision": int(row["revision"]),
                }

        parent_map = {task_id: [] for task_id in task_ids}
        child_map = {task_id: [] for task_id in task_ids}
        if task_ids:
            placeholders = ",".join("?" for _ in task_ids)
            for row in conn.execute(
                f"SELECT parent_id, child_id FROM task_links "
                f"WHERE parent_id IN ({placeholders}) OR child_id IN ({placeholders}) "
                "ORDER BY parent_id, child_id",
                (*task_ids, *task_ids),
            ):
                parent_id, child_id = str(row["parent_id"]), str(row["child_id"])
                if parent_id in visible and child_id in visible:
                    child_map[parent_id].append(child_id)
                    parent_map[child_id].append(parent_id)

        columns = {status: [] for status in _OWNER_PROJECT_COLUMNS}
        for task in tasks:
            if task.status not in columns:
                raise OwnerWorkspaceError("snapshot_unavailable", "kanban.db contains an unknown status")
            state = event_state.get(task.id)
            columns[task.status].append({
                "id": task.id,
                "title": owner_title(task.title),
                "assignee_name": task.assignee,
                "responsibility": task.responsibility,
                "updated_at": _owner_timestamp(state["latest"] if state else task.created_at),
                "event_revision": state["revision"] if state else 0,
                "parent_ids": parent_map[task.id],
                "child_ids": child_map[task.id],
            })

        worker_rows = conn.execute(
            "SELECT r.profile, t.title AS task_title, r.started_at "
            "FROM task_runs r JOIN tasks t ON t.id = r.task_id "
            "WHERE t.project_id = ? AND t.task_kind = 'work' "
            "AND r.ended_at IS NULL AND r.worker_pid IS NOT NULL "
            "AND r.profile IS NOT NULL AND t.status = 'running' "
            "ORDER BY r.started_at ASC LIMIT ?",
            (project_id, _OWNER_PROJECT_MAX_WORKERS + 1),
        ).fetchall()

        attachment_rows = conn.execute(
            "SELECT a.id, a.filename, a.content_type, a.size, a.created_at "
            "FROM task_attachments a JOIN tasks t ON t.id = a.task_id "
            "WHERE t.project_id = ? AND t.task_kind = 'work' "
            "ORDER BY a.created_at ASC, a.id ASC LIMIT ?",
            (project_id, _OWNER_PROJECT_MAX_ATTACHMENTS + 1),
        ).fetchall()
        attachments = [
            _owner_project_attachment_projection(row)
            for row in attachment_rows[:_OWNER_PROJECT_MAX_ATTACHMENTS]
        ]

        run_rows = conn.execute(
            "SELECT r.*, t.title AS task_title FROM task_runs r JOIN tasks t ON t.id = r.task_id "
            "WHERE t.project_id = ? AND t.task_kind = 'work' "
            "ORDER BY r.started_at DESC, r.id DESC LIMIT ?",
            (project_id, _OWNER_PROJECT_MAX_RUNS + 1),
        ).fetchall()
        # The list is globally newest-first, so the first row carrying a given
        # exact task id is that task's newest run and every later row for the
        # SAME id is an older attempt at the same work. Decided here, from the
        # native id, because a title cannot tell two distinct tasks apart —
        # and because a task absent from the visible columns (archived, or
        # past the task bound) leaves a reader nothing to disambiguate with.
        runs: list[dict] = []
        task_ids_with_newer_run: set[str] = set()
        for row in run_rows[:_OWNER_PROJECT_MAX_RUNS]:
            run = kanban_db.Run.from_row(row)
            run_task_id = str(run.task_id)
            runs.append(
                _owner_project_run_projection(
                    run,
                    row["task_title"],
                    has_newer_run=run_task_id in task_ids_with_newer_run,
                    run_context=run_context,
                )
            )
            task_ids_with_newer_run.add(run_task_id)
    except OwnerWorkspaceError:
        raise
    except (KeyError, TypeError, ValueError, sqlite3.Error) as exc:
        raise OwnerWorkspaceError(
            "snapshot_unavailable", "the Project snapshot could not be read"
        ) from exc
    finally:
        conn.close()

    steward = project_steward_snapshot(project_id=project_id, lookback_days=7)

    return {
        "project": {
            "id": project_id,
            "slug": project_slug,
            "name": owner_project_name(project["name"]),
            "description": project["description"],
            "board": board_slug,
            "archived": False,
        },
        "board": {
            "slug": board_slug,
            # Falls back to the Project name, so it is the same owner-facing
            # display slot and gets the same projection.
            "name": owner_project_name(metadata.get("name") or project["name"]),
            "project_id": project_id,
            "counts": counts,
            "total": sum(counts[status] for status in _OWNER_PROJECT_COLUMNS),
        },
        "columns": [{"name": status, "tasks": columns[status]} for status in _OWNER_PROJECT_COLUMNS],
        "workers": [{
            "profile": str(row["profile"]),
            "task_title": owner_title(row["task_title"]),
            "started_at": int(row["started_at"]),
        } for row in worker_rows[:_OWNER_PROJECT_MAX_WORKERS]],
        "attachments": attachments,
        "runs": runs,
        "steward": steward,
        "truncated": {
            "tasks": len(task_rows) > _OWNER_PROJECT_MAX_TASKS,
            "workers": len(worker_rows) > _OWNER_PROJECT_MAX_WORKERS,
            "attachments": len(attachment_rows) > _OWNER_PROJECT_MAX_ATTACHMENTS,
            "runs": len(run_rows) > _OWNER_PROJECT_MAX_RUNS,
        },
    }


def read_project_attachment(
    ctx: OwnerContext, project_slug: str, attachment_id: str,
) -> dict:
    """Read one regular attachment from one exact receipt-owned Project."""
    try:
        project_slug = projects_db.normalize_slug(project_slug) or ""
        if not re.fullmatch(r"[0-9]{1,20}", str(attachment_id)):
            raise ValueError("invalid attachment id")
        native_id = int(attachment_id)
    except (TypeError, ValueError) as exc:
        raise OwnerWorkspaceError("attachment_not_found", "the attachment is unavailable") from exc

    project = next(
        (item for item in list_committed_projects(ctx) if item["slug"] == project_slug),
        None,
    )
    if project is None or project["archived"]:
        raise OwnerWorkspaceError("attachment_not_found", "the attachment is unavailable")
    project_id = str(project["project_id"])
    board_slug = str(project["board"])
    try:
        _assert_board_ownership(board_slug, project_id)
    except OwnerWorkspaceError as exc:
        raise OwnerWorkspaceError(
            "snapshot_unavailable", "the Project board ownership could not be verified"
        ) from exc

    board_path = (
        kanban_db.kanban_home() / "kanban.db"
        if board_slug == kanban_db.DEFAULT_BOARD
        else kanban_db.board_dir(board_slug) / "kanban.db"
    )
    conn = _open_read_only_sqlite(board_path, label="kanban.db")
    try:
        row = conn.execute(
            "SELECT a.* FROM task_attachments a JOIN tasks t ON t.id = a.task_id "
            "WHERE a.id = ? AND t.project_id = ? AND t.task_kind = 'work'",
            (native_id, project_id),
        ).fetchone()
    except sqlite3.Error as exc:
        raise OwnerWorkspaceError(
            "snapshot_unavailable", "the Project attachment could not be read"
        ) from exc
    finally:
        conn.close()
    if row is None:
        raise OwnerWorkspaceError("attachment_not_found", "the attachment is unavailable")

    root = (
        kanban_db.kanban_home() / "kanban" / "attachments"
        if board_slug == kanban_db.DEFAULT_BOARD
        else kanban_db.board_dir(board_slug) / "attachments"
    ).resolve()
    try:
        stored = Path(str(row["stored_path"])).resolve()
        stored.relative_to(root)
    except (OSError, ValueError) as exc:
        raise OwnerWorkspaceError("attachment_not_found", "the attachment is unavailable") from exc

    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        fd = os.open(stored, flags)
    except OSError as exc:
        raise OwnerWorkspaceError("attachment_not_found", "the attachment is unavailable") from exc
    try:
        info = os.fstat(fd)
        size = int(info.st_size)
        expected_size = int(row["size"] or 0)
        if (
            not stat.S_ISREG(info.st_mode)
            or size < 0
            or size > kanban_db.KANBAN_ATTACHMENT_MAX_BYTES
            or size != expected_size
        ):
            raise OwnerWorkspaceError("attachment_not_found", "the attachment is unavailable")
        chunks: list[bytes] = []
        remaining = size
        while remaining:
            chunk = os.read(fd, min(remaining, 1024 * 1024))
            if not chunk:
                raise OwnerWorkspaceError("attachment_not_found", "the attachment is unavailable")
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(fd, 1):
            raise OwnerWorkspaceError("attachment_not_found", "the attachment is unavailable")
    except OSError as exc:
        raise OwnerWorkspaceError("attachment_not_found", "the attachment is unavailable") from exc
    finally:
        os.close(fd)

    return {
        **_owner_project_attachment_projection(row),
        "body": b"".join(chunks),
    }


_OWNER_DECISIONS_LIMIT = 100


def _owner_decision_ref(
    ctx: OwnerContext, *, authority: str, native_id: str
) -> str:
    """Return an opaque presentation key, never a native authority id."""
    canonical = "\x00".join((ctx.actor, ctx.profile, authority, native_id))
    return "decision_" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:32]


def _owner_decision_reason(value: Any, *, fallback: str) -> str:
    """Bound and redact text already intended for owner review."""
    from agent.redact import redact_sensitive_text

    text = redact_sensitive_text(str(value or ""), force=True)
    text = " ".join(text.split()).strip()
    return text[:500] or fallback


def list_owner_decisions(ctx: OwnerContext) -> list[dict]:
    """Project pending native gates into one owner-safe read-only inbox.

    This is deliberately a projection, not a decision store or mutation
    router. Work reviews and owner-input blocks stay native Tasks;
    capability suggestions stay native recommendation rows. The caller gets
    Internal work in the native review column is Raphael's responsibility,
    not an owner decision. The caller gets only an opaque presentation key plus enough Project scope to route the
    owner back to the authority-specific surface. Native ids, assignees,
    bodies, results, runs, provenance and private evidence never leave this
    boundary.
    """
    decisions: list[dict] = []
    projects = list_committed_projects(ctx)
    for project in projects:
        if project["archived"]:
            continue

        project_id = str(project["project_id"])
        board_slug = str(project["board"])
        if board_slug == kanban_db.DEFAULT_BOARD:
            board_path = kanban_db.kanban_home() / "kanban.db"
        else:
            board_path = kanban_db.board_dir(board_slug) / "kanban.db"

        conn = _open_read_only_sqlite(board_path, label="kanban.db")
        try:
            columns = {
                row["name"] for row in conn.execute("PRAGMA table_info(tasks)")
            }
            required = {
                "id", "title", "status", "created_at", "project_id",
                "task_kind", "block_kind", "review_policy",
                "recommendation_label", "recommendation_rationale",
                "recommendation_decision",
            }
            if not required <= columns:
                raise OwnerWorkspaceError(
                    "snapshot_unavailable", "kanban.db schema is unavailable"
                )
            rows = conn.execute(
                "SELECT id, title, status, created_at, task_kind, block_kind, "
                "review_policy, recommendation_label, recommendation_rationale, "
                "recommendation_decision FROM tasks WHERE project_id = ? AND ("
                "(task_kind = 'work' AND status = 'blocked' "
                "AND block_kind = 'needs_input') OR "
                "(task_kind = 'recommendation' AND status = 'review' "
                "AND review_policy = 'owner' "
                "AND COALESCE(recommendation_decision, 'pending') = 'pending')) "
                "ORDER BY created_at DESC, id ASC LIMIT ?",
                (project_id, _OWNER_DECISIONS_LIMIT + 1),
            ).fetchall()
        except sqlite3.Error as exc:
            raise OwnerWorkspaceError(
                "snapshot_unavailable", "the Project decisions could not be read"
            ) from exc
        finally:
            conn.close()

        for row in rows:
            task_kind = str(row["task_kind"])
            if task_kind == "recommendation":
                authority = "recommendation"
                kind = "capability"
                title = owner_title(row["recommendation_label"])
                reason = _owner_decision_reason(
                    row["recommendation_rationale"],
                    fallback="Raphael has suggested a capability change for your review.",
                )
            elif row["block_kind"] == "needs_input":
                authority = "task"
                kind = "owner_input"
                title = owner_title(row["title"])
                reason = "Raphael needs your answer before this work can continue."
            decisions.append({
                "decision_ref": _owner_decision_ref(
                    ctx, authority=authority, native_id=str(row["id"])
                ),
                "authority": authority,
                "kind": kind,
                "project_slug": str(project["slug"]),
                "project_name": owner_project_name(project["name"]),
                "title": title,
                "reason": reason,
                "created_at": _owner_timestamp(row["created_at"]),
            })

    authority_order = {"owner_input": 0, "capability": 1}
    decisions.sort(
        key=lambda item: (
            authority_order.get(item["kind"], 9),
            str(item["project_name"]).casefold(),
            str(item["title"]).casefold(),
            str(item["decision_ref"]),
        )
    )
    return decisions[:_OWNER_DECISIONS_LIMIT]


def set_project_archived(
    ctx: OwnerContext,
    *,
    idempotency_key: str,
    project_id: str,
    action: str,
) -> dict:
    """Apply one receipt-backed Project lifecycle or execution change.

    Hard delete and worker termination are intentionally absent. Pause blocks
    future claims after the current dispatch critical section; work already
    running may finish. Restore is deliberately paused until the owner resumes.
    """
    idempotency_key = _bounded_text(
        idempotency_key, "idempotency_key", limit=200,
    )
    project_id = _bounded_text(project_id, "project_id", limit=100)
    action = str(action or "").strip().lower()
    if action not in {"archive", "restore", "pause", "resume"}:
        raise OwnerWorkspaceError(
            "invalid_argument",
            "action must be 'archive', 'restore', 'pause', or 'resume'",
        )
    payload = {"project_id": project_id, "action": action}
    digest = _digest(payload)
    operation = "owner_project_lifecycle"
    target_archived = action == "archive"

    pconn = projects_db.connect()
    try:
        _ensure_schema(pconn)
        if not _receipt_owns_project(pconn, ctx, project_id):
            raise OwnerWorkspaceError(
                "project_not_owned",
                "the Project is not owned by this owner-workspace profile",
            )
        project = projects_db.get_project(pconn, project_id)
        if project is None or not project.board_slug:
            raise OwnerWorkspaceError(
                "project_not_found", "the Project is unavailable",
            )
        _assert_board_ownership(project.board_slug, project_id)

        state, row, token = _acquire_or_replay(
            pconn, ctx, idempotency_key, operation, digest,
        )
        if state == "terminal":
            return json.loads(row["result_json"])

        approval = _confirm(
            ctx,
            operation=operation,
            digest=digest,
            description=f"{action.title()} Project {project.name!r}",
        )
        if not approval.get("approved"):
            result = {
                "ok": False,
                "error": "confirmation_denied",
                "reason": approval.get("reason"),
            }
            _finalize_receipt(
                pconn, ctx, idempotency_key, token,
                status="denied", result=result,
            )
            return result

        with write_txn(pconn):
            _assert_owns_lease(pconn, ctx, idempotency_key, token)
            current = projects_db.get_project(pconn, project_id)
            if current is None or not current.board_slug:
                raise OwnerWorkspaceError(
                    "project_not_found", "the Project is unavailable",
                )
            metadata = kanban_db.read_board_metadata(current.board_slug)
            if action in {"pause", "resume"} and current.archived:
                result = {
                    "ok": False,
                    "error": "conflict",
                    "archived": True,
                    "execution_paused": True,
                }
            elif action in {"archive", "restore"} and (
                bool(current.archived) == target_archived and row is None
            ):
                result = {
                    "ok": False,
                    "error": "conflict",
                    "archived": bool(current.archived),
                    "execution_paused": bool(
                        metadata.get("dispatch_paused_by_owner")
                    ),
                }
            elif action == "pause" and (
                metadata.get("dispatch_paused_by_owner") is True and row is None
            ):
                result = {
                    "ok": False,
                    "error": "conflict",
                    "archived": False,
                    "execution_paused": True,
                }
            elif action == "resume" and (
                kanban_db.board_dispatch_allowed(metadata) and row is None
            ):
                result = {
                    "ok": False,
                    "error": "conflict",
                    "archived": False,
                    "execution_paused": False,
                }
            else:
                if action in {"archive", "restore"}:
                    _set_project_dispatch_state(
                        current.board_slug,
                        enabled=False,
                        paused_by_owner=True,
                    )
                    pconn.execute(
                        "UPDATE projects SET archived = ? WHERE id = ?",
                        (int(target_archived), project_id),
                    )
                    resulting_archived = target_archived
                    execution_paused = True
                elif action == "pause":
                    _set_project_dispatch_state(
                        current.board_slug,
                        paused_by_owner=True,
                    )
                    resulting_archived = False
                    execution_paused = True
                else:
                    _set_project_dispatch_state(
                        current.board_slug,
                        enabled=True,
                        paused_by_owner=False,
                    )
                    resulting_archived = False
                    execution_paused = False
                result = {
                    "ok": True,
                    "action": action,
                    "project_slug": current.slug,
                    "archived": resulting_archived,
                    "execution_paused": execution_paused,
                }

        _finalize_receipt(
            pconn, ctx, idempotency_key, token,
            status="committed", result=result,
        )
        return result
    finally:
        pconn.close()


# ---------------------------------------------------------------------------
# project_steward_snapshot (read-only)
# ---------------------------------------------------------------------------

_PROJECT_STEWARD_LIMIT = 12
_INTERNAL_TITLE_PREFIX = re.compile(
    r"^[A-Za-z]{1,4}\d{1,4}[A-Za-z]?\s*(?:[—–:-])\s*"
)
_OWNER_TITLE_LIMIT = 240
_OWNER_PROJECT_NAME_LIMIT = 160
_OWNER_STATE_LABELS = {
    "triage": "Needs attention",
    "todo": "Planned",
    "scheduled": "Scheduled",
    "ready": "Ready",
    "running": "In progress",
    "blocked": "Blocked",
    "review": "Being checked",
    "done": "Completed",
}
_OWNER_BLOCK_REASONS = {
    "needs_input": "Waiting for owner input",
    "dependency": "Waiting for another piece of work",
    "capability": "A required capability is not available",
    "transient": "A temporary problem needs another attempt",
}


def _owner_timestamp(value: Any) -> Optional[str]:
    if value is None:
        return None
    try:
        return datetime.fromtimestamp(int(value), timezone.utc).isoformat().replace(
            "+00:00", "Z"
        )
    except (OSError, OverflowError, TypeError, ValueError):
        return None


def _owner_display_text(
    value: Any, *, limit: int, strip_internal_prefix: bool = False
) -> str:
    """Sanitize, redact, normalize and bound one owner-visible string.

    This is an owner-facing NON-navigation egress boundary: nothing
    projected here is ever followed as a link, so redaction runs in the
    strict URL-credential mode. Credential-named query parameters,
    pre-signed signatures and ``user:password@`` userinfo are masked instead
    of being left actionable the way an ordinary tool flow needs them.

    Unsafe control/display characters are removed FIRST, reusing the
    repository's own sanitizers rather than a private character list:
    ``tools.ansi_strip`` for NUL/C0/C1/DEL, ESC sequences and the invisible
    plane-14 tag characters, and ``tools.threat_patterns.INVISIBLE_CHARS``
    for zero-width characters, bidi overrides/isolates and the interlinear
    annotation frame. Doing it before redaction means a credential cannot
    survive by splitting its own token across an invisible character, and
    doing it at all means the text cannot reorder or hide what the owner is
    shown. Safe human Unicode is left exactly as written. Whitespace collapse
    (which also folds U+2028/U+2029), the optional internal-prefix strip, and
    the ``limit`` Unicode code point bound then apply to already-sanitized,
    already-redacted text.

    ``strip_unicode_tags`` runs once more AFTER that bound, and the order
    matters: the first pass preserves the three pinned RGI subdivision flags
    whole, so the code point slice can land inside one and leave dangling
    invisible tag characters behind a visible U+1F3F4 base — exactly the
    smuggling frame the boundary exists to remove, and what the Workspace
    correctly rejects. Re-running the same pinned sanitizer on the bounded
    text keeps an intact flag intact and reduces a cut one to its visible
    black-flag base, with no grapheme parsing of our own.
    """
    from agent.redact import redact_sensitive_text
    from tools.ansi_strip import sanitize_display_text, strip_unicode_tags
    from tools.threat_patterns import INVISIBLE_CHARS

    text = strip_unicode_tags(sanitize_display_text(str(value or "")))
    text = "".join(char for char in text if char not in INVISIBLE_CHARS)
    text = redact_sensitive_text(text, force=True, redact_url_credentials=True)
    text = " ".join(text.split())
    if strip_internal_prefix:
        text = _INTERNAL_TITLE_PREFIX.sub("", text).strip()
    return strip_unicode_tags(text[:limit])


def owner_title(value: Any) -> str:
    """Return the canonical owner-safe, single-line work-item title.

    See :func:`_owner_display_text` for the egress contract. A work-item
    title additionally loses the internal ``B03 — ``-style dispatcher prefix,
    which is Raphael's own bookkeeping and means nothing to the owner.
    """
    return _owner_display_text(
        value, limit=_OWNER_TITLE_LIMIT, strip_internal_prefix=True
    ) or "Untitled work item"


def owner_project_name(value: Any) -> str:
    """Return the canonical owner-safe, single-line Project display name.

    Same egress contract as :func:`owner_title` — see
    :func:`_owner_display_text` — with two deliberate differences. The bound
    is 160 code points, matching the bound ``commit_task_graph`` accepts when
    a Project name is first written, so a name that was storable is never cut
    on the way back out. And no internal-prefix strip runs: a Project name is
    owner-authored text, not an internally-labelled work item, so a name that
    happens to start like a dispatcher prefix must survive intact.

    Every owner-facing surface that shows a Project name projects it through
    here, so the name an owner reads in a Project list, a snapshot, a steward
    report or a decision cannot disagree between surfaces.
    """
    return _owner_display_text(
        value, limit=_OWNER_PROJECT_NAME_LIMIT
    ) or "Untitled Project"


def _open_read_only_sqlite(path, *, label: str) -> sqlite3.Connection:
    if not path.is_file():
        raise OwnerWorkspaceError(
            "snapshot_unavailable", f"{label} is unavailable"
        )
    try:
        conn = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA query_only = ON")
        return conn
    except (OSError, sqlite3.Error) as exc:
        raise OwnerWorkspaceError(
            "snapshot_unavailable", f"{label} could not be opened read-only"
        ) from exc


def project_steward_snapshot(
    *, project_id: str, lookback_days: int = 7
) -> dict:
    """Return one bounded, owner-safe Project health snapshot without writes.

    The Project binding is resolved from the active profile's projects.db and
    checked against the shared board's ownership metadata. Only task titles,
    owner-friendly states, and timestamps are projected: bodies, results,
    errors, assignees, paths, branches, task/run IDs, and raw events never
    cross this boundary.
    """
    project_id = _bounded_text(project_id, "project_id", limit=100)
    if (
        isinstance(lookback_days, bool)
        or not isinstance(lookback_days, int)
        or not 1 <= lookback_days <= 30
    ):
        raise OwnerWorkspaceError(
            "invalid_argument", "lookback_days must be an integer from 1 to 30"
        )

    pconn = _open_read_only_sqlite(
        projects_db.projects_db_path(), label="projects.db"
    )
    try:
        project_columns = {
            row["name"] for row in pconn.execute("PRAGMA table_info(projects)")
        }
        required_project_columns = {
            "id", "name", "board_slug", "archived",
        }
        if not required_project_columns <= project_columns:
            raise OwnerWorkspaceError(
                "snapshot_unavailable", "projects.db schema is unavailable"
            )
        project = pconn.execute(
            "SELECT id, name, board_slug, archived FROM projects WHERE id = ?",
            (project_id,),
        ).fetchone()
    except sqlite3.Error as exc:
        raise OwnerWorkspaceError(
            "snapshot_unavailable", "the Project could not be read"
        ) from exc
    finally:
        pconn.close()

    if (
        project is None
        or bool(project["archived"])
        or not project["board_slug"]
    ):
        raise OwnerWorkspaceError(
            "project_not_found", "the Project is unavailable"
        )

    board_slug = str(project["board_slug"])
    if not kanban_db.board_exists(board_slug):
        raise OwnerWorkspaceError(
            "snapshot_unavailable", "the Project board is unavailable"
        )
    try:
        _assert_board_ownership(board_slug, project_id)
    except OwnerWorkspaceError as exc:
        raise OwnerWorkspaceError(
            "snapshot_unavailable",
            "the Project board ownership could not be verified",
        ) from exc
    metadata = kanban_db.read_board_metadata(board_slug)
    if board_slug == kanban_db.DEFAULT_BOARD:
        board_path = kanban_db.kanban_home() / "kanban.db"
    else:
        board_path = kanban_db.board_dir(board_slug) / "kanban.db"

    kconn = _open_read_only_sqlite(board_path, label="kanban.db")
    try:
        task_columns = {
            row["name"] for row in kconn.execute("PRAGMA table_info(tasks)")
        }
        required_task_columns = {
            "title", "status", "created_at", "started_at", "completed_at",
            "project_id", "task_kind", "block_kind",
        }
        if not required_task_columns <= task_columns:
            raise OwnerWorkspaceError(
                "snapshot_unavailable", "kanban.db schema is unavailable"
            )
        rows = kconn.execute(
            "SELECT title, status, created_at, started_at, completed_at, block_kind "
            "FROM tasks WHERE project_id = ? AND task_kind = 'work' "
            "AND status != 'archived' ORDER BY created_at ASC",
            (project_id,),
        ).fetchall()
    except sqlite3.Error as exc:
        raise OwnerWorkspaceError(
            "snapshot_unavailable", "the Project board could not be read"
        ) from exc
    finally:
        kconn.close()

    now = _now()
    cutoff = now - lookback_days * 86_400
    items = [
        {
            "title": owner_title(row["title"]),
            "status": str(row["status"]),
            "created_at": int(row["created_at"]),
            "started_at": row["started_at"],
            "completed_at": row["completed_at"],
            "block_kind": row["block_kind"],
        }
        for row in rows
    ]

    progress_rows = sorted(
        (
            item for item in items
            if item["status"] == "done"
            and item["completed_at"] is not None
            and int(item["completed_at"]) >= cutoff
        ),
        key=lambda item: int(item["completed_at"]),
        reverse=True,
    )
    attention_rows = [
        item for item in items
        if item["status"] == "triage"
        or (
            item["status"] == "blocked"
            and item["block_kind"] != "needs_input"
        )
    ]
    decision_rows = [
        item for item in items
        if item["status"] == "blocked" and item["block_kind"] == "needs_input"
    ]
    active_rows = [
        item for item in items
        if item["status"] in {"todo", "scheduled", "ready", "running", "review"}
    ]
    stale_rows = sorted(
        (
            item for item in items
            if item["status"] not in {"done"}
            and int(item["started_at"] or item["created_at"]) < cutoff
        ),
        key=lambda item: int(item["started_at"] or item["created_at"]),
    )

    def _bounded(values):
        return values[:_PROJECT_STEWARD_LIMIT]

    progress = [
        {
            "title": item["title"],
            "completed_at": _owner_timestamp(item["completed_at"]),
        }
        for item in _bounded(progress_rows)
    ]
    needs_attention = [
        {
            "title": item["title"],
            "state": _OWNER_STATE_LABELS[item["status"]],
            "reason": _OWNER_BLOCK_REASONS.get(
                str(item["block_kind"] or ""), "Reason not recorded"
            ),
        }
        for item in _bounded(attention_rows)
    ]
    decisions_needed = [
        {
            "title": item["title"],
            "state": "Waiting for your answer",
        }
        for item in _bounded(decision_rows)
    ]
    active_work = [
        {"title": item["title"], "state": _OWNER_STATE_LABELS[item["status"]]}
        for item in _bounded(active_rows)
    ]
    stale_candidates = [
        {
            "title": item["title"],
            "state": _OWNER_STATE_LABELS.get(item["status"], "Needs attention"),
            "age_days": max(
                0, (now - int(item["started_at"] or item["created_at"])) // 86_400
            ),
        }
        for item in _bounded(stale_rows)
    ]

    open_count = sum(item["status"] != "done" for item in items)
    dispatch_allowed = kanban_db.board_dispatch_allowed(metadata)
    paused_by_owner = metadata.get("dispatch_paused_by_owner") is True
    running_now = any(item["status"] == "running" for item in items)
    if not active_rows and not decision_rows and not attention_rows:
        execution_state = "complete"
        execution_summary = "The approved work is complete."
    elif not dispatch_allowed:
        if paused_by_owner and running_now:
            execution_state = "paused"
            execution_summary = (
                "Raphael is finishing work already underway, then will stay paused."
            )
        elif paused_by_owner:
            execution_state = "paused"
            execution_summary = "Raphael is paused and will not start new work."
        else:
            execution_state = "waiting_for_approval"
            execution_summary = (
                "Raphael is waiting for an approved milestone before starting work."
            )
    elif decision_rows:
        execution_state = "waiting_for_you"
        execution_summary = "Raphael needs your answer before the plan can continue."
    elif attention_rows:
        execution_state = "needs_attention"
        execution_summary = "Raphael found a problem and is preparing the safest next step."
    elif active_rows:
        execution_state = "working"
        execution_summary = "Raphael is coordinating the approved milestone."
    else:  # pragma: no cover - exhaustive state partition
        execution_state = "complete"
        execution_summary = "The approved work is complete."
    return {
        "schema_version": 2,
        "project": {"name": owner_project_name(project["name"])},
        "generated_at": _owner_timestamp(now),
        "lookback_days": lookback_days,
        "execution": {
            "state": execution_state,
            "summary": execution_summary,
            "paused": execution_state == "paused",
        },
        "counts": {
            "open": open_count,
            "completed_in_window": len(progress_rows),
            "needs_attention": len(attention_rows),
            "awaiting_review": sum(
                item["status"] == "review" for item in items
            ),
        },
        "progress": progress,
        "needs_attention": needs_attention,
        "decisions_needed": decisions_needed,
        "active_work": active_work,
        "stale_candidates": stale_candidates,
        "truncated": {
            "progress": len(progress_rows) > _PROJECT_STEWARD_LIMIT,
            "needs_attention": len(attention_rows) > _PROJECT_STEWARD_LIMIT,
            "decisions_needed": len(decision_rows) > _PROJECT_STEWARD_LIMIT,
            "active_work": len(active_rows) > _PROJECT_STEWARD_LIMIT,
            "stale_candidates": len(stale_rows) > _PROJECT_STEWARD_LIMIT,
        },
    }


# ---------------------------------------------------------------------------
# owner_project_plan_commit
# ---------------------------------------------------------------------------

_MAX_PROJECT_PLAN_CHANGES = 12
_MAX_PROJECT_PLAN_REPLACEMENTS = 6
_PROJECT_PLAN_TRIGGERS = {
    "owner_request", "milestone_boundary", "persistent_blocker", "scheduled_review",
}
_PROJECT_PLAN_MUTABLE_STATUSES = {
    "triage", "todo", "scheduled", "ready", "blocked", "review",
}


def _require_exact_keys(value: Any, field: str, keys: set[str]) -> dict:
    if not isinstance(value, dict) or set(value) != keys:
        raise OwnerWorkspaceError(
            "invalid_argument", f"{field} must contain exactly {sorted(keys)}",
        )
    return value


def _normalize_project_task_ref(value: Any, field: str, *, mutating: bool) -> dict:
    raw = _require_exact_keys(
        value, field, {"task_id", "expected_status", "expected_revision"},
    )
    task_id = _bounded_text(raw["task_id"], f"{field}.task_id", limit=100)
    status = _bounded_text(
        raw["expected_status"], f"{field}.expected_status", limit=20,
    )
    if status not in kanban_db.VALID_STATUSES:
        raise OwnerWorkspaceError("invalid_status", f"{field} has an invalid status")
    if mutating and status not in _PROJECT_PLAN_MUTABLE_STATUSES:
        raise OwnerWorkspaceError(
            "unsafe_transition",
            f"{field} cannot change a running, completed, or archived task",
        )
    revision = raw["expected_revision"]
    if isinstance(revision, bool) or not isinstance(revision, int) or revision < 1:
        raise OwnerWorkspaceError(
            "invalid_argument", f"{field}.expected_revision must be a positive integer",
        )
    return {
        "task_id": task_id,
        "expected_status": status,
        "expected_revision": revision,
    }


def _normalize_project_task_spec(
    value: Any, field: str, *, parent_limit: Optional[int] = None,
) -> dict:
    required = {"title", "body", "assignee", "parents"} if parent_limit is not None else {
        "title", "body", "assignee",
    }
    allowed = required | {"responsibility"}
    if not isinstance(value, dict) or not required.issubset(value) or not set(value).issubset(allowed):
        raise OwnerWorkspaceError(
            "invalid_argument",
            f"{field} must contain {sorted(required)} and only optional responsibility",
        )
    raw = value
    from agent.redact import redact_sensitive_text

    result = {
        "title": redact_sensitive_text(
            _bounded_text(raw["title"], f"{field}.title", limit=240), force=True,
        ),
        "body": redact_sensitive_text(
            _bounded_text(raw["body"], f"{field}.body", limit=12_000), force=True,
        ),
        "assignee": _normalize_graph_assignee(raw["assignee"], f"{field}.assignee"),
    }
    try:
        result["responsibility"] = kanban_db.normalize_responsibility(
            raw.get("responsibility")
        )
    except ValueError as exc:
        raise OwnerWorkspaceError("invalid_argument", str(exc)) from exc
    if parent_limit is not None:
        parents = raw["parents"]
        if not isinstance(parents, list):
            raise OwnerWorkspaceError("invalid_argument", f"{field}.parents must be a list")
        normalized_parents: list[int] = []
        for parent in parents:
            if isinstance(parent, bool) or not isinstance(parent, int):
                raise OwnerWorkspaceError(
                    "invalid_argument", f"{field}.parents must contain only indices",
                )
            if parent < 0 or parent >= parent_limit or parent in normalized_parents:
                raise OwnerWorkspaceError(
                    "invalid_argument", f"{field}.parents may reference only earlier replacements",
                )
            normalized_parents.append(parent)
        result["parents"] = normalized_parents
    return result


def _normalize_project_changes(value: Any) -> tuple[list[dict], str]:
    if not isinstance(value, list) or not value or len(value) > _MAX_PROJECT_PLAN_CHANGES:
        raise OwnerWorkspaceError(
            "invalid_argument",
            f"changes must contain 1-{_MAX_PROJECT_PLAN_CHANGES} items",
        )
    from agent.redact import redact_sensitive_text

    normalized: list[dict] = []
    created_count = 0
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            raise OwnerWorkspaceError("invalid_argument", f"changes[{index}] must be an object")
        action = str(item.get("action") or "").strip().lower()
        field = f"changes[{index}]"
        reason = redact_sensitive_text(
            _bounded_text(item.get("reason"), f"{field}.reason", limit=1_000),
            force=True,
        )

        if action == "add":
            required = {
                "action", "reason", "title", "body", "assignee",
                "existing_parents", "new_parents",
            }
            allowed = required | {"responsibility"}
            if not required.issubset(item) or not set(item).issubset(allowed):
                raise OwnerWorkspaceError(
                    "invalid_argument",
                    f"{field} must contain {sorted(required)} and only optional responsibility",
                )
            raw = item
            existing = raw["existing_parents"]
            new_parents = raw["new_parents"]
            if not isinstance(existing, list) or not isinstance(new_parents, list):
                raise OwnerWorkspaceError(
                    "invalid_argument", f"{field} parent fields must be lists",
                )
            parent_indexes: list[int] = []
            for parent in new_parents:
                if (
                    isinstance(parent, bool)
                    or not isinstance(parent, int)
                    or parent < 0
                    or parent >= index
                    or parent in parent_indexes
                ):
                    raise OwnerWorkspaceError(
                        "invalid_argument", f"{field}.new_parents must name unique earlier add changes",
                    )
                parent_indexes.append(parent)
            spec = _normalize_project_task_spec(
                {**{key: raw[key] for key in ("title", "body", "assignee")}, "responsibility": raw.get("responsibility")},
                field,
                parent_limit=None,
            )
            normalized.append({
                "action": action,
                "reason": reason,
                **spec,
                "existing_parents": [
                    _normalize_project_task_ref(ref, f"{field}.existing_parents[{ref_index}]", mutating=False)
                    for ref_index, ref in enumerate(existing)
                ],
                "new_parents": parent_indexes,
            })
            created_count += 1
            continue

        if action == "split":
            raw = _require_exact_keys(item, field, {"action", "reason", "target", "replacements"})
            replacements = raw["replacements"]
            if (
                not isinstance(replacements, list)
                or len(replacements) < 2
                or len(replacements) > _MAX_PROJECT_PLAN_REPLACEMENTS
            ):
                raise OwnerWorkspaceError(
                    "invalid_argument",
                    f"{field}.replacements must contain 2-{_MAX_PROJECT_PLAN_REPLACEMENTS} tasks",
                )
            normalized.append({
                "action": action,
                "reason": reason,
                "target": _normalize_project_task_ref(raw["target"], f"{field}.target", mutating=True),
                "replacements": [
                    _normalize_project_task_spec(
                        replacement,
                        f"{field}.replacements[{replacement_index}]",
                        parent_limit=replacement_index,
                    )
                    for replacement_index, replacement in enumerate(replacements)
                ],
            })
            created_count += len(replacements)
            continue

        if action == "merge":
            raw = _require_exact_keys(item, field, {"action", "reason", "targets", "replacement"})
            targets = raw["targets"]
            if (
                not isinstance(targets, list)
                or len(targets) < 2
                or len(targets) > _MAX_PROJECT_PLAN_REPLACEMENTS
            ):
                raise OwnerWorkspaceError(
                    "invalid_argument", f"{field}.targets must contain 2-{_MAX_PROJECT_PLAN_REPLACEMENTS} tasks",
                )
            refs = [
                _normalize_project_task_ref(ref, f"{field}.targets[{ref_index}]", mutating=True)
                for ref_index, ref in enumerate(targets)
            ]
            if len({ref["task_id"] for ref in refs}) != len(refs):
                raise OwnerWorkspaceError("invalid_argument", f"{field}.targets must be unique")
            normalized.append({
                "action": action,
                "reason": reason,
                "targets": refs,
                "replacement": _normalize_project_task_spec(
                    raw["replacement"], f"{field}.replacement", parent_limit=None,
                ),
            })
            created_count += 1
            continue

        if action == "move":
            raw = _require_exact_keys(item, field, {"action", "reason", "target", "to_status"})
            target = _normalize_project_task_ref(raw["target"], f"{field}.target", mutating=True)
            to_status = str(raw["to_status"] or "").strip()
            if to_status != "ready" or to_status == target["expected_status"]:
                raise OwnerWorkspaceError(
                    "unsafe_transition", f"{field}.to_status must reactivate work into ready",
                )
            normalized.append({"action": action, "reason": reason, "target": target, "to_status": to_status})
            continue

        if action in {"postpone", "cancel"}:
            raw = _require_exact_keys(item, field, {"action", "reason", "target"})
            normalized.append({
                "action": action,
                "reason": reason,
                "target": _normalize_project_task_ref(raw["target"], f"{field}.target", mutating=True),
            })
            continue

        raise OwnerWorkspaceError(
            "invalid_argument", f"{field}.action must be add, split, merge, move, postpone, or cancel",
        )

    for index, change in enumerate(normalized):
        if change["action"] == "add":
            for parent_index in change["new_parents"]:
                if normalized[parent_index]["action"] != "add":
                    raise OwnerWorkspaceError(
                        "invalid_argument", f"changes[{index}].new_parents may reference only add changes",
                    )
    if created_count > _MAX_GRAPH_TASKS:
        raise OwnerWorkspaceError(
            "milestone_too_large", f"one plan may create at most {_MAX_GRAPH_TASKS} tasks",
        )
    if any(change["action"] in {"merge", "cancel"} for change in normalized) and len(normalized) != 1:
        raise OwnerWorkspaceError(
            "separate_owner_decision", "merge and cancel must be the only change in their owner approval",
        )
    return normalized, (
        "significant_removal"
        if normalized[0]["action"] in {"merge", "cancel"}
        else "standard"
    )


def _resolve_existing_project_board(pconn: sqlite3.Connection, project_id: str):
    project = projects_db.get_project(pconn, project_id)
    if project is None or project.archived or not project.board_slug:
        raise OwnerWorkspaceError("project_not_found", f"project {project_id!r} is unavailable")
    if not kanban_db.board_exists(project.board_slug):
        raise OwnerWorkspaceError("project_not_found", "the Project board is unavailable")
    _assert_board_ownership(project.board_slug, project_id)
    return project, project.board_slug


def _committed_project_plan_result(
    kconn: sqlite3.Connection, *, anchor_task_id: str, digest: str,
    idempotency_key: str, ctx: OwnerContext,
) -> Optional[dict]:
    for event in reversed(kanban_db.list_events(kconn, anchor_task_id)):
        payload = event.payload or {}
        if (
            event.kind == "owner_project_plan_applied"
            and payload.get("request_digest") == digest
            and payload.get("idempotency_key") == idempotency_key
            and payload.get("actor") == ctx.actor
            and payload.get("profile") == ctx.profile
            and isinstance(payload.get("result"), dict)
        ):
            return payload["result"]
    return None


def commit_project_plan(
    ctx: OwnerContext,
    *,
    idempotency_key: str,
    project_id: str,
    anchor_task_id: str,
    trigger: str,
    request_title: str,
    summary: str,
    specification: str,
    current_milestone: str,
    owner_visible_result: str,
    later_milestones: Any,
    changes: Any,
) -> dict:
    """Commit one approved Project Steward plan to the existing native board."""
    from agent.redact import redact_sensitive_text

    idempotency_key = _bounded_text(idempotency_key, "idempotency_key", limit=200)
    project_id = _bounded_text(project_id, "project_id", limit=100)
    anchor_task_id = _bounded_text(anchor_task_id, "anchor_task_id", limit=100)
    trigger = str(trigger or "").strip()
    if trigger not in _PROJECT_PLAN_TRIGGERS:
        raise OwnerWorkspaceError("invalid_argument", "trigger is invalid")
    request_title = redact_sensitive_text(
        _bounded_text(request_title, "request_title", limit=240), force=True,
    )
    summary = redact_sensitive_text(_bounded_text(summary, "summary", limit=2_000), force=True)
    specification = redact_sensitive_text(
        _bounded_text(specification, "specification", limit=20_000), force=True,
    )
    current_milestone = redact_sensitive_text(
        _bounded_text(current_milestone, "current_milestone", limit=1_000), force=True,
    )
    owner_visible_result = redact_sensitive_text(
        _bounded_text(owner_visible_result, "owner_visible_result", limit=1_000), force=True,
    )
    normalized_later = _normalize_later_milestones(later_milestones)
    normalized_changes, risk_level = _normalize_project_changes(changes)
    plan_record = "\n\n".join(
        [request_title, summary, specification, f"Owner-visible result\n{owner_visible_result}"]
    )
    payload = {
        "project_id": project_id,
        "anchor_task_id": anchor_task_id,
        "trigger": trigger,
        "request_title": request_title,
        "summary": summary,
        "specification": specification,
        "current_milestone": current_milestone,
        "owner_visible_result": owner_visible_result,
        "later_milestones": normalized_later,
        "risk_level": risk_level,
        "changes": normalized_changes,
    }
    digest = _digest(payload)
    operation = "owner_project_plan_commit"

    pconn = projects_db.connect()
    try:
        _ensure_schema(pconn)
        state, row, token = _acquire_or_replay(
            pconn, ctx, idempotency_key, operation, digest,
        )
        if state == "terminal":
            return json.loads(row["result_json"])
        project, board_slug = _resolve_existing_project_board(pconn, project_id)
        kconn = kanban_db.connect(board=board_slug)
        try:
            recovered = _committed_project_plan_result(
                kconn,
                anchor_task_id=anchor_task_id,
                digest=digest,
                idempotency_key=idempotency_key,
                ctx=ctx,
            )
            if recovered is not None:
                _set_project_dispatch_state(board_slug, enabled=True)
                result = {
                    "ok": True,
                    "project_id": project_id,
                    "project_slug": project.slug,
                    "board": board_slug,
                    "risk_level": risk_level,
                    **recovered,
                }
                _finalize_receipt(
                    pconn, ctx, idempotency_key, token, status="committed", result=result,
                )
                return result

            approval = _confirm(
                ctx,
                operation=operation,
                digest=digest,
                description=(
                    f"Apply {len(normalized_changes)} approved Project change(s) "
                    f"to {project.name!r}"
                ),
            )
            if not approval.get("approved"):
                result = {"ok": False, "error": "confirmation_denied", "reason": approval.get("reason")}
                _finalize_receipt(
                    pconn, ctx, idempotency_key, token, status="denied", result=result,
                )
                return result

            with write_txn(pconn):
                _assert_owns_lease(pconn, ctx, idempotency_key, token)
                applied = kanban_db.apply_owner_project_plan(
                    kconn,
                    project_id=project_id,
                    anchor_task_id=anchor_task_id,
                    changes=normalized_changes,
                    actor=ctx.actor,
                    profile=ctx.profile,
                    idempotency_key=idempotency_key,
                    request_digest=digest,
                    trigger=trigger,
                    plan_summary=plan_record,
                    current_milestone=current_milestone,
                    later_milestones=normalized_later,
                    board=board_slug,
                )
        finally:
            kconn.close()

        if not applied["applied"]:
            result = {
                "ok": False,
                "error": "conflict",
                "project_id": project_id,
                "project_slug": project.slug,
                "change_count": 0,
            }
        else:
            _set_project_dispatch_state(board_slug, enabled=True)
            result = {
                "ok": True,
                "project_id": project_id,
                "project_slug": project.slug,
                "board": board_slug,
                "risk_level": risk_level,
                **applied,
            }
        _update_progress(
            pconn, ctx, idempotency_key, token,
            project_id=project_id, board_slug=board_slug, task_id=anchor_task_id,
        )
        _finalize_receipt(
            pconn, ctx, idempotency_key, token, status="committed", result=result,
        )
        return result
    finally:
        pconn.close()



# ---------------------------------------------------------------------------
# owner_task_move
# ---------------------------------------------------------------------------

_UNSAFE_STATUSES = {"running"}


def move_task(
    ctx: OwnerContext,
    *,
    idempotency_key: str,
    task_id: str,
    to_status: str,
    expected_status: str,
    expected_revision: Any,
    board: Optional[str] = None,
) -> dict:
    """Optimistic compare-and-swap task move inside the existing Kanban
    write transaction. See ``kanban_db.cas_transition_task``.

    Crash-safe readiness repair: if the CAS status move + event commit
    succeeds but a crash follows before ``recompute_ready`` (or before
    finalization), a replay adopting the dead claim recognizes — by the
    committed event's payload carrying THIS receipt's full identity (actor,
    profile, idempotency_key, AND the requested to_status/expected_status
    transition) at this task's very next revision after ``expected_revision``
    (see ``kanban_db.get_next_event_after`` — ``task_events.id`` is a
    board-wide sequence, not contiguous per task, so it is not necessarily
    ``expected_revision + 1``) — that it already performed the transition.
    It does not re-run the CAS (which would misread the already-advanced
    status/revision as an unrelated conflict); it rebuilds the same success
    snapshot, reruns readiness recompute (idempotent — safe to repeat), and
    finalizes the original success. Genuinely unrelated status/revision
    drift still reports a conflict — including a DIFFERENT actor/profile
    validly reusing the same idempotency_key text on the same task (receipts
    are scoped per actor/profile/key, but the task's event log is
    board-wide, so a bare idempotency_key match on the next event would
    otherwise let one receipt fabricate a success snapshot for a mutation an
    entirely different receipt performed).

    Lease-fenced: the lease check, replay recognition, CAS, and readiness
    recompute all run inside one held ``projects.db`` write lock (see
    :func:`_assert_owns_lease`) so a takeover cannot land between validating
    the lease and committing the move.
    """
    idempotency_key = _require_str(idempotency_key, "idempotency_key")
    task_id = _require_str(task_id, "task_id")
    to_status = _require_str(to_status, "to_status")
    expected_status = _require_str(expected_status, "expected_status")
    if to_status not in kanban_db.VALID_STATUSES:
        raise OwnerWorkspaceError(
            "invalid_status", f"to_status must be one of {sorted(kanban_db.VALID_STATUSES)}",
        )
    if to_status in _UNSAFE_STATUSES or expected_status in _UNSAFE_STATUSES:
        raise OwnerWorkspaceError(
            "unsafe_transition",
            "owner_task_move cannot claim or move a running/claimed task — use the "
            "worker claim/complete/block lifecycle for run-owning transitions",
        )
    try:
        expected_revision = int(expected_revision)
    except (TypeError, ValueError):
        raise OwnerWorkspaceError("invalid_argument", "expected_revision must be an integer")

    board_norm = str(board).strip() if board else None
    payload = {
        "task_id": task_id, "board": board_norm, "to_status": to_status,
        "expected_status": expected_status, "expected_revision": expected_revision,
    }
    digest = _digest(payload)
    operation = "owner_task_move"

    pconn = projects_db.connect()
    try:
        _ensure_schema(pconn)
        state, row, token = _acquire_or_replay(pconn, ctx, idempotency_key, operation, digest)
        if state == "terminal":
            return json.loads(row["result_json"])

        approval = _confirm(
            ctx, operation=operation, digest=digest,
            description=f"Move task {task_id} to {to_status!r} (from {expected_status!r})",
        )
        if not approval.get("approved"):
            result = {"ok": False, "error": "confirmation_denied", "reason": approval.get("reason")}
            _finalize_receipt(pconn, ctx, idempotency_key, token, status="denied", result=result)
            return result

        kconn = kanban_db.connect(board=board_norm)
        try:
            if kanban_db.get_task(kconn, task_id) is None:
                raise OwnerWorkspaceError("task_not_found", f"no such task {task_id}")

            # Fence: the lease check, the replay-recognition read, the CAS
            # move, and readiness recompute all run under ONE held pconn
            # write lock. A competing claimant's adoption needs that same
            # lock (see :func:`_claim_or_wait`'s own ``write_txn``), so it
            # cannot land between "we validated the lease" and "we moved the
            # task" — it either waits for this block to finish (and then
            # finds a terminal/foreign row instead of a live one to adopt),
            # or it already committed before this block started, in which
            # case `_assert_owns_lease` observes the new token right here and
            # fails closed before the CAS ever runs.
            with write_txn(pconn):
                _assert_owns_lease(pconn, ctx, idempotency_key, token)

                snapshot = None
                if row is not None:
                    # Adopting a dead claim — check whether THIS receipt
                    # already committed the CAS before crashing.
                    # ``task_events.id`` is a board-wide AUTOINCREMENT
                    # sequence (not contiguous per task), so "the next event
                    # after expected_revision" — NOT necessarily
                    # ``expected_revision + 1`` — is the one that would have
                    # been our CAS's commit, if it happened. Receipts are
                    # scoped per (actor, profile, idempotency_key), but
                    # ``task_events`` is shared across the whole board — a
                    # DIFFERENT actor/profile may legitimately reuse the same
                    # idempotency_key text on the same task, so matching on
                    # idempotency_key alone could recognize a foreign event as
                    # this receipt's own. Require the full identity this
                    # receipt actually emitted: actor, profile, idempotency
                    # key, AND the requested transition (to_status /
                    # expected_status) — anything less risks fabricating a
                    # success snapshot (and triggering readiness repair) for
                    # a mutation this receipt never performed.
                    already = kanban_db.get_next_event_after(kconn, task_id, expected_revision)
                    if (
                        already is not None
                        and already.kind == "owner_move"
                        and (already.payload or {}).get("idempotency_key") == idempotency_key
                        and (already.payload or {}).get("actor") == ctx.actor
                        and (already.payload or {}).get("profile") == ctx.profile
                        and (already.payload or {}).get("to_status") == to_status
                        and (already.payload or {}).get("expected_status") == expected_status
                    ):
                        snapshot = {"moved": True, "status": to_status, "revision": already.id}

                if snapshot is None:
                    snapshot = kanban_db.cas_transition_task(
                        kconn, task_id,
                        expected_status=expected_status,
                        expected_revision=expected_revision,
                        to_status=to_status,
                        event_kind="owner_move",
                        event_payload={
                            "actor": ctx.actor, "profile": ctx.profile,
                            "idempotency_key": idempotency_key,
                            "to_status": to_status, "expected_status": expected_status,
                        },
                    )
                if snapshot["moved"] and to_status in ("done", "archived"):
                    # Both done and archived parents satisfy readiness;
                    # promote any children this move just unblocked.
                    # Idempotent — safe to re-run on a replay that repairs a
                    # prior crash gap between the status commit above and
                    # this call, without re-emitting the move event (already
                    # committed above).
                    kanban_db.recompute_ready(kconn)
        finally:
            kconn.close()

        if snapshot["moved"]:
            result = {"ok": True, "task_id": task_id, "status": snapshot["status"], "revision": snapshot["revision"]}
        else:
            result = {
                "ok": False, "error": "conflict", "task_id": task_id,
                "current_status": snapshot["status"], "current_revision": snapshot["revision"],
            }
        _finalize_receipt(pconn, ctx, idempotency_key, token, status="committed", result=result)
        return result
    finally:
        pconn.close()


# ---------------------------------------------------------------------------
# owner_task_comment
# ---------------------------------------------------------------------------


def comment_task(
    ctx: OwnerContext,
    *,
    idempotency_key: str,
    task_id: str,
    body: str,
    board: Optional[str] = None,
) -> dict:
    """Append a comment as the trusted actor. Comment + audit event commit
    exactly once inside ``kanban_db.add_comment``'s existing transaction.

    Crash-safe replay: ``add_comment`` is called with an ``operation_key``
    derived from the trusted actor/profile/idempotency_key, durably recorded
    on the comment row itself (see ``kanban_db.add_comment``). A retry after
    a crash between the comment commit and receipt finalization re-calls
    ``add_comment`` with the SAME operation_key and gets back the ORIGINAL
    comment id — no second comment, no second event.

    Lease-fenced: the lease check and the comment insert share one held
    ``projects.db`` write lock (see :func:`_assert_owns_lease`) so a
    takeover cannot land between validating the lease and committing the
    comment.
    """
    idempotency_key = _require_str(idempotency_key, "idempotency_key")
    task_id = _require_str(task_id, "task_id")
    body = _require_str(body, "body")

    board_norm = str(board).strip() if board else None
    payload = {"task_id": task_id, "board": board_norm, "body": body}
    digest = _digest(payload)
    operation = "owner_task_comment"

    pconn = projects_db.connect()
    try:
        _ensure_schema(pconn)
        state, row, token = _acquire_or_replay(pconn, ctx, idempotency_key, operation, digest)
        if state == "terminal":
            return json.loads(row["result_json"])

        approval = _confirm(
            ctx, operation=operation, digest=digest,
            description=f"Comment on task {task_id}",
        )
        if not approval.get("approved"):
            result = {"ok": False, "error": "confirmation_denied", "reason": approval.get("reason")}
            _finalize_receipt(pconn, ctx, idempotency_key, token, status="denied", result=result)
            return result

        operation_key = f"{ctx.actor}:{ctx.profile}:{idempotency_key}"
        kconn = kanban_db.connect(board=board_norm)
        try:
            # Fence: the lease check and the comment insert share one held
            # pconn write lock, so a takeover cannot land between validating
            # the lease and actually appending the comment — see the
            # module docstring's "Real lease fencing" section.
            with write_txn(pconn):
                _assert_owns_lease(pconn, ctx, idempotency_key, token)
                # Author comes ONLY from the trusted context — never from a
                # tool-call argument (kanban_db.add_comment takes it as a
                # plain parameter; the kernel is the only caller that may
                # supply it).
                comment_id = kanban_db.add_comment(
                    kconn, task_id, author=ctx.actor, body=body, operation_key=operation_key,
                )
                task = kanban_db.get_task(kconn, task_id)
                revision = kanban_db.task_event_revision(kconn, task_id)
        finally:
            kconn.close()

        result = {
            "ok": True, "task_id": task_id, "comment_id": comment_id,
            "status": task.status, "revision": revision,
        }
        _finalize_receipt(pconn, ctx, idempotency_key, token, status="committed", result=result)
        return result
    finally:
        pconn.close()
