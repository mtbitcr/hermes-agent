"""Owner-workspace mutation kernel.

The single deep in-process boundary behind the API-server-only
``owner_workspace`` toolset (``owner_workspace_bootstrap``, ``owner_task_move``,
``owner_task_comment`` — see ``tools/owner_workspace_tools.py``). All three
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
import secrets
import sqlite3
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Optional

from hermes_cli import kanban_db, projects_db
from hermes_cli.sqlite_util import write_txn

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
