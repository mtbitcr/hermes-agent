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

import contextlib
import contextvars
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
    configured_assignment_for,
    normalize_execution_tier,
    resolve_task_assignment,
    validate_runtime_assignment as validate_raphael_model_assignment,
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
    authority_digest TEXT,
    terminal_generation INTEGER NOT NULL DEFAULT 0,
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
class OwnerProposalAuthority:
    actor: str
    profile: str
    session: str
    conversation: str
    response_id: str
    operation: str
    idempotency_key: str
    payload_digest: str


_owner_proposal_authority: contextvars.ContextVar[Optional[OwnerProposalAuthority]] = (
    contextvars.ContextVar("owner_proposal_authority", default=None)
)


def set_owner_proposal_authority(
    authority: Optional[OwnerProposalAuthority],
) -> contextvars.Token:
    return _owner_proposal_authority.set(authority)


def reset_owner_proposal_authority(token: contextvars.Token) -> None:
    _owner_proposal_authority.reset(token)


@dataclass(frozen=True)
class OwnerContext:
    actor: str
    profile: str
    session: str
    authority: Optional[OwnerProposalAuthority] = None


def resolve_owner_context() -> OwnerContext:
    """Derive the trusted actor/profile/session identity for the current call.

    Never accepts these from tool-call arguments — see module docstring.
    """
    from hermes_cli.profiles import get_active_profile_name
    from tools.approval import get_current_session_key

    profile = get_active_profile_name()
    session = get_current_session_key(default="")
    authority = _owner_proposal_authority.get()
    if authority is not None and (
        authority.actor != profile
        or authority.profile != profile
        or authority.session != session
    ):
        authority = None
    return OwnerContext(
        actor=profile,
        profile=profile,
        session=session,
        authority=authority,
    )


def _ensure_schema(conn: sqlite3.Connection) -> None:
    # Schema detection and ALTER must share one write fence. Otherwise two
    # first-use callers can both observe a missing column and one loses the
    # race with "duplicate column name".
    with write_txn(conn):
        conn.execute(_RECEIPTS_SCHEMA)
        columns = {
            str(row[1]) for row in conn.execute(
                "PRAGMA table_info(owner_workspace_receipts)"
            )
        }
        if "terminal_generation" not in columns:
            conn.execute(
                "ALTER TABLE owner_workspace_receipts "
                "ADD COLUMN terminal_generation INTEGER NOT NULL DEFAULT 0"
            )
            # Existing terminal lifecycle receipts already represent completed
            # authority generations. Preserve that fact across the migration;
            # the exact ordinal is irrelevant because readers expose only the
            # monotonic count of distinct terminal receipts.
            conn.execute(
                "UPDATE owner_workspace_receipts SET terminal_generation = 1 "
                "WHERE operation = 'owner_project_lifecycle' "
                "AND status IN ('committed', 'denied')"
            )
        if "authority_digest" not in columns:
            conn.execute(
                "ALTER TABLE owner_workspace_receipts ADD COLUMN authority_digest TEXT"
            )


def _now() -> int:
    return int(time.time())


def _receipt_lease_seconds() -> int:
    """Keep a mutation claim alive for the complete configured approval wait."""
    try:
        from tools.approval import _get_approval_timeout

        approval_timeout = int(_get_approval_timeout())
    except (ImportError, TypeError, ValueError):
        approval_timeout = 300
    return max(_LOCK_TTL_SECONDS, approval_timeout + 60)


def _digest(payload: dict) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _require_owner_run_authority(
    ctx: OwnerContext,
    *,
    operation: str,
    idempotency_key: str,
    payload: dict,
) -> None:
    authority = ctx.authority
    if (
        authority is None
        or authority.actor != ctx.actor
        or authority.profile != ctx.profile
        or authority.session != ctx.session
        or authority.operation != operation
        or authority.idempotency_key != idempotency_key
        or authority.payload_digest != _digest(payload)
    ):
        raise OwnerWorkspaceError(
            "owner_run_authority_required",
            "this mutation is not bound to the authenticated owner run",
        )


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


# The identifier each committed receipt's own operation MUST record. A
# committed receipt is durable authority: the operations that create a Project
# always write its id, so a committed row of one of those operations without a
# usable ``project_id`` is unreadable authority, never "this receipt owns no
# Project". Operations absent from this map are still required to record
# ``ok: True`` — that is the fact "committed" means.
_OWNER_RECEIPT_REQUIRED_IDENTIFIERS: dict[str, tuple[str, ...]] = {
    "owner_workspace_bootstrap": ("project_id",),
    "owner_task_graph_commit": ("project_id",),
}


class OwnerReceiptUnreadable(Exception):
    """One committed receipt could not be read as the authority it claims to be.

    Deliberately its own type rather than a ``None``/``False`` return: every
    caller that reads committed receipts is answering a question about
    authority, and "the authority is unreadable" is a third answer alongside
    "absent" and "present". Collapsing it into either of the other two is what
    turned an unreadable Project record into an authoritative empty Project
    list, an unreadable ownership row into a 404, and an unreadable native
    receipt into "the mutation did not commit" — which puts an external effect
    back on the retry path.
    """


def decode_committed_owner_receipt(result_json: Any, operation: str) -> dict:
    """Strictly decode ONE committed receipt's result, or fail closed.

    The single validator every committed-receipt reader uses — the Project
    projection, the Project ownership check, and native crash recovery — so
    "which receipts are believable" is one rule rather than three that drift.
    Raises :class:`OwnerReceiptUnreadable` for anything that is not a JSON
    object recording ``ok: True`` plus the non-empty identifiers this exact
    operation must carry.
    """
    if not isinstance(result_json, str) or not result_json:
        raise OwnerReceiptUnreadable("a committed receipt has no readable result")
    try:
        result = json.loads(result_json)
    except (TypeError, ValueError) as exc:
        raise OwnerReceiptUnreadable(
            "a committed receipt could not be read"
        ) from exc
    if not isinstance(result, dict) or result.get("ok") is not True:
        raise OwnerReceiptUnreadable("a committed receipt does not record success")
    for field in _OWNER_RECEIPT_REQUIRED_IDENTIFIERS.get(operation, ()):
        value = result.get(field)
        if not isinstance(value, str) or not value.strip():
            raise OwnerReceiptUnreadable(
                f"a committed {operation} receipt records no usable {field}"
            )
    return result


def _receipt_authority_digest(
    ctx: OwnerContext, key: str, operation: str,
) -> Optional[str]:
    """Return the exact run authority bound to this native mutation."""
    authority = ctx.authority
    if (
        authority is not None
        and authority.actor == ctx.actor
        and authority.profile == ctx.profile
        and authority.session == ctx.session
        and authority.idempotency_key == key
        and authority.operation == operation
        and re.fullmatch(r"[a-f0-9]{64}", authority.payload_digest) is not None
    ):
        return authority.payload_digest
    return None


def _terminal_replay(
    conn: sqlite3.Connection,
    ctx: OwnerContext,
    key: str,
    operation: str,
    digest: str,
) -> Optional[dict]:
    """Return one exact durable terminal result before touching live state."""
    row = _get_receipt(conn, ctx, key)
    if row is None:
        return None
    if row["request_digest"] != digest or row["operation"] != operation:
        raise OwnerWorkspaceError(
            "idempotency_key_conflict",
            f"idempotency_key {key!r} was already used for a different request",
        )
    if row["status"] not in {"committed", "denied"}:
        return None
    if _is_retryable_confirmation_timeout(row):
        return None
    try:
        result = json.loads(row["result_json"])
    except (TypeError, ValueError) as exc:
        raise OwnerWorkspaceError(
            "receipt_invalid", "the durable terminal receipt is unreadable",
        ) from exc
    if not isinstance(result, dict):
        raise OwnerWorkspaceError(
            "receipt_invalid", "the durable terminal receipt is unreadable",
        )
    return result


def read_committed_owner_run_receipt(
    *, profile: str, idempotency_key: str, operation: str, authority_digest: str,
) -> Optional[dict]:
    """Recover one successful native receipt after a gateway crash.

    The gateway may die after the native database commits but before its own
    run terminal is stored. Recovery is allowed only when the trusted profile,
    idempotency key, operation, and authority digest all match the receipt
    written by that exact run.

    Three distinct answers, never two. ``None`` means this exact run committed
    NOTHING — the only state in which its work may be reported failed and its
    proposal released. A dict is the exact committed receipt. Anything else —
    more than one matching committed row, or a row whose result cannot be read
    as a successful receipt — raises :class:`OwnerReceiptUnreadable`, because
    it is uncertainty about an EXTERNAL effect: answering "absent" there put a
    committed change back on the retry path and let a completed change be
    reported to the owner as failed.
    """
    if (
        not isinstance(profile, str)
        or not profile.strip()
        or not isinstance(idempotency_key, str)
        or not idempotency_key
        or not isinstance(operation, str)
        or not operation
        or re.fullmatch(r"[a-f0-9]{64}", authority_digest) is None
    ):
        return None
    conn = projects_db.connect()
    try:
        _ensure_schema(conn)
        rows = conn.execute(
            "SELECT result_json FROM owner_workspace_receipts "
            "WHERE actor = ? AND profile = ? AND idempotency_key = ? "
            "AND operation = ? AND authority_digest = ? AND status = 'committed' "
            "LIMIT 2",
            (
                profile.strip(), profile.strip(), idempotency_key,
                operation, authority_digest,
            ),
        ).fetchall()
        if not rows:
            return None
        if len(rows) != 1:
            raise OwnerReceiptUnreadable(
                "more than one committed receipt claims this exact native run"
            )
        return decode_committed_owner_receipt(rows[0]["result_json"], operation)
    finally:
        conn.close()


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
    authority_digest = _receipt_authority_digest(ctx, key, operation)
    with write_txn(conn):
        row = _get_receipt(conn, ctx, key)
        if row is None:
            conn.execute(
                "INSERT INTO owner_workspace_receipts "
                "(actor, profile, idempotency_key, operation, request_digest, status, "
                " lock_token, lock_expires, authority_digest, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, 'in_progress', ?, ?, ?, ?, ?)",
                (
                    ctx.actor, ctx.profile, key, operation, digest, token,
                    now + _receipt_lease_seconds(), authority_digest, now, now,
                ),
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
                "board_slug = NULL, task_id = NULL, result_json = NULL, "
                "terminal_generation = 0, authority_digest = ?, updated_at = ? "
                "WHERE actor = ? AND profile = ? AND idempotency_key = ? AND status = 'denied'",
                (
                    token, now + _receipt_lease_seconds(), authority_digest, now,
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
            "UPDATE owner_workspace_receipts SET lock_token = ?, lock_expires = ?, "
            "authority_digest = COALESCE(authority_digest, ?), updated_at = ? "
            "WHERE actor = ? AND profile = ? AND idempotency_key = ?",
            (
                token, now + _receipt_lease_seconds(), authority_digest, now,
                ctx.actor, ctx.profile, key,
            ),
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
        terminal_generation = 1 if status in {"committed", "denied"} else 0
        cur = conn.execute(
            "UPDATE owner_workspace_receipts SET status = ?, result_json = ?, "
            "terminal_generation = CASE WHEN terminal_generation > 0 "
            "THEN terminal_generation ELSE ? END, "
            "lock_token = NULL, lock_expires = NULL, updated_at = ? "
            "WHERE actor = ? AND profile = ? AND idempotency_key = ? AND lock_token = ?",
            (
                status,
                json.dumps(result, ensure_ascii=False),
                terminal_generation,
                _now(),
                ctx.actor,
                ctx.profile,
                key,
                token,
            ),
        )
        if cur.rowcount != 1:
            raise OwnerWorkspaceError(
                "lease_lost",
                f"idempotency_key {key!r}'s claim lease was lost before finalization; "
                "refusing to finalize on another claimant's behalf",
            )


def _confirm(ctx: OwnerContext, *, operation: str, digest: str, description: str) -> dict:
    """Ask the owner for one fresh decision on one exact operation.

    ``description`` is an owner egress: it reaches the gateway notify callback
    and the ``pre_approval_request``/``post_approval_response`` plugin hooks
    verbatim, and is what the owner actually reads before deciding. So any
    stored name interpolated into it is projected through
    :func:`owner_project_name` (or :func:`owner_title`) FIRST — a raw name
    would otherwise carry escape sequences, invisible reordering characters or
    a URL-borne credential onto the approval surface itself.
    """
    from tools.approval import request_exact_operation_approval

    if not ctx.session:
        return {"approved": False, "reason": "no_session"}
    safe_description = _owner_display_text(
        description,
        limit=500,
        strip_internal_prefix=False,
    ) or "Apply the confirmed Project change"
    return request_exact_operation_approval(
        ctx.session,
        operation=operation,
        payload_digest=digest,
        actor=ctx.actor,
        profile=ctx.profile,
        description=safe_description,
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
    with _global_guard(f"board-{slug}", label="global board ownership"):
        yield


@contextmanager
def _global_guard(lock_name: str, *, label: str):
    """Hold one exclusive, kernel-held file lock under the shared lock root.

    The shared primitive behind :func:`_global_board_guard` and
    :func:`profile_route_lock`: both need to order unrelated PROCESSES
    (dashboard, CLI, workers, other profiles) around a resource that is not
    itself a single database row, and both must fail closed when they cannot.
    """
    fd = None
    try:
        if fcntl is None and msvcrt is None:  # pragma: no cover - exotic platform
            raise RuntimeError("no OS file-locking primitive is available")
        lock_dir = kanban_db.kanban_home() / "kanban" / "locks"
        lock_dir.mkdir(parents=True, exist_ok=True)
        lock_path = lock_dir / f"{lock_name}.lock"
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
            f"could not acquire the {label} guard for {lock_name!r}: {exc}",
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
    from agent.redact import redact_sensitive_text

    idempotency_key = _require_str(idempotency_key, "idempotency_key")
    name = _native_owner_project_name(name, "name")
    description = _bounded_text(
        description, "description", limit=2_000, required=False,
    )
    if description:
        description = redact_sensitive_text(description, force=True)

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
            description=f"Bootstrap owner workspace project {owner_project_name(name)!r}",
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
            elif (
                project.name != name
                or (project.description or None) != (description or None)
            ):
                raise OwnerWorkspaceError(
                    "crash_recovery_failed",
                    "the recovered Project does not match the approved canonical fields",
                )

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
                # Non-executable by construction, and non-promotable too: this
                # anchor carries no assignee, no execution tier and therefore no
                # approved route, so it must never be dispatchable AND must
                # never become dispatchable. ``task_kind='control'`` is what
                # makes that structural rather than a matter of which column it
                # sits in: generic specify/decompose, the dispatcher's ready
                # queue, the claim paths and the default-assignee adoption all
                # positively require ``task_kind = 'work'``, so none of them can
                # see this row. Only owner-approved graph creation makes
                # executable tasks, and those carry exact route authority.
                task_id = kanban_db.create_task(
                    kconn,
                    title=name,
                    body=description,
                    created_by=ctx.actor,
                    triage=True,
                    control=True,
                    board=board_slug,
                    project_id=project_id,
                    idempotency_key=task_idempotency_key,
                )
                task = kanban_db.get_control_task(kconn, task_id)
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
                if (
                    task.title != name
                    or (task.body or None) != (description or None)
                    or task.idempotency_key != task_idempotency_key
                    or task.assignee is not None
                    or task.status != "triage"
                ):
                    raise OwnerWorkspaceError(
                        "crash_recovery_failed",
                        "the recovered initial task does not match the approved canonical fields",
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


def _pinned_route_fields(assignee: str, execution_tier: str, route: Any) -> dict:
    """Return the durable, owner-approved route pin for one new task.

    The planner only ever supplies the semantic tier; every model, provider
    and effort here comes from the Raphael model policy resolved against the
    role's current native setting. ``model_policy_lock`` is a versioned,
    digest-bound authority over the whole (assignee, provider, model, effort,
    tier) tuple, so none of them can move under the task afterwards.
    """
    try:
        lock = kanban_db.mint_policy_lock(
            assignee,
            route.provider,
            route.model,
            route.reasoning_effort,
            execution_tier,
        )
    except ValueError as exc:
        raise OwnerWorkspaceError("invalid_model_route", str(exc)) from exc
    return {
        "execution_tier": execution_tier,
        "model_override": route.model,
        "provider_override": route.provider,
        "reasoning_effort": route.reasoning_effort,
        "model_policy_lock": lock,
    }


def _resolved_route_pin(assignee: str, execution_tier: Any, field: str) -> dict:
    """Resolve and pin one role's approved route for a semantic task class."""
    try:
        tier = normalize_execution_tier(execution_tier)
        route = resolve_task_assignment(assignee, tier)
    except (ValueError, OSError) as exc:
        raise OwnerWorkspaceError("invalid_model_route", f"{field}: {exc}") from exc
    return _pinned_route_fields(assignee, tier, route)


def _graph_root_route(tasks: list[dict], root_assignee: str) -> dict:
    """Resolve the pin for the executable root that reviews this milestone.

    The root is not a planner-classified task — it wakes up to judge the whole
    milestone once its children are done — so its tier is derived, not
    supplied: reviewing a milestone that contains deep work is itself deep.
    """
    tier = "deep" if any(task["execution_tier"] == "deep" for task in tasks) else "routine"
    return _resolved_route_pin(root_assignee, tier, "root_assignee")


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
        title = _native_owner_title(raw.get("title"), f"tasks[{index}].title")
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

        route_pin = _resolved_route_pin(
            assignee, raw.get("execution_tier"), f"tasks[{index}].execution_tier",
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
            "title": title,
            "body": redact_sensitive_text(body, force=True),
            "assignee": assignee,
            "responsibility": responsibility,
            "parents": clean_parents,
            **route_pin,
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
        if not create and not _receipt_owns_project(pconn, ctx, project_id):
            raise OwnerWorkspaceError(
                "project_not_owned",
                "the Project is not owned by this trusted owner receipt",
            )
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


def _ensure_project_control_anchor(
    pconn: sqlite3.Connection,
    kconn: sqlite3.Connection,
    ctx: OwnerContext,
    idempotency_key: str,
    token: str,
    *,
    project_id: str,
    board_slug: str,
    name: Optional[str],
    description: Optional[str],
) -> str:
    """Create/adopt the ONE hidden control anchor this Project is steered by.

    Every later approved change to a Project is hung off its canonical control
    anchor (:func:`_receipt_bound_control_anchor`), so a Project created
    without one is permanently unable to accept another approved plan. The
    anchor is therefore created in the SAME recoverable operation that creates
    the Project, under the same lease fence, with a deterministic identity — a
    crash-and-retry recomputes the same key and adopts the same row instead of
    minting a second anchor.

    It is a ``task_kind='control'`` row, never an executable bootstrap task:
    it carries no assignee, no execution tier and therefore no approved route,
    and every dispatch/claim/promote/assign query in ``kanban_db`` positively
    requires ``task_kind = 'work'``. Owner-visible projections list work rows
    only, so it never reaches the owner either.
    """
    anchor_key = "owanchor_" + _derive_id(ctx, idempotency_key, "project-anchor")
    with write_txn(pconn):
        _assert_owns_lease(pconn, ctx, idempotency_key, token)
        existing = kconn.execute(
            "SELECT id FROM tasks WHERE project_id = ? AND task_kind = 'control'",
            (project_id,),
        ).fetchall()
        if len(existing) > 1:
            raise OwnerWorkspaceError(
                "crash_recovery_failed",
                "this Project already holds more than one control anchor",
            )
        anchor_id = kanban_db.create_task(
            kconn,
            title=name,
            body=description,
            created_by=ctx.actor,
            triage=True,
            control=True,
            board=board_slug,
            project_id=project_id,
            idempotency_key=anchor_key,
        )
        if existing and str(existing[0]["id"]) != anchor_id:
            raise OwnerWorkspaceError(
                "crash_recovery_failed",
                "this Project's control anchor is not the one this receipt owns",
            )
        anchor = kanban_db.get_control_task(kconn, anchor_id)
        if (
            anchor is None
            or anchor.project_id != project_id
            or anchor.idempotency_key != anchor_key
            or anchor.assignee is not None
        ):
            raise OwnerWorkspaceError(
                "crash_recovery_failed",
                "the Project control anchor does not match this receipt",
            )
    return anchor_id


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

    operation = "owner_task_graph_commit"
    _require_owner_run_authority(
        ctx,
        operation=operation,
        idempotency_key=idempotency_key,
        payload={
            "idempotency_key": idempotency_key,
            "mode": mode,
            "project_name": project_name,
            "project_description": project_description,
            "project_id": project_id,
            "request_title": request_title,
            "specification": specification,
            "current_milestone": current_milestone,
            "owner_visible_result": owner_visible_result,
            "root_assignee": root_assignee,
            "tasks": tasks,
            "later_milestones": later_milestones,
        },
    )

    # The root task's native title, not prose: canonical before the digest.
    request_title = _native_owner_title(request_title, "request_title")
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
    root_route = _graph_root_route(normalized_tasks, root_assignee)
    normalized_later = _normalize_later_milestones(later_milestones)

    if mode == "new":
        if project_id is not None and str(project_id).strip():
            raise OwnerWorkspaceError(
                "invalid_argument", "project_id is not accepted when mode is 'new'",
            )
        project_name = _native_owner_project_name(project_name, "project_name")
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
        "root_route": root_route,
        "tasks": normalized_tasks,
        "later_milestones": normalized_later,
    }
    digest = _digest(payload)

    pconn = projects_db.connect()
    try:
        _ensure_schema(pconn)
        if mode == "existing" and not _receipt_owns_project(
            pconn, ctx, canonical_project_id,
        ):
            raise OwnerWorkspaceError(
                "project_not_owned",
                "the Project is not owned by this trusted owner receipt",
            )
        state, row, token = _acquire_or_replay(
            pconn, ctx, idempotency_key, operation, digest,
        )
        if state == "terminal":
            # A crash between the terminal receipt and activation leaves work
            # approved but parked. Replay finishes the same activation from the
            # committed receipt — no owner confirmation, no duplicate tasks.
            committed = json.loads(row["result_json"])
            _activate_committed_owner_work(committed)
            return committed

        approval = _confirm(
            ctx,
            operation=operation,
            digest=digest,
            description=(
                f"Create project {owner_project_name(project_name)!r} "
                f"with {len(normalized_tasks)} tasks"
                if mode == "new"
                else f"Add {len(normalized_tasks)} tasks to the current Project"
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
            # A graph-created Project gets its canonical control anchor in the
            # same recoverable operation that creates it; without one, no later
            # approved plan could ever be applied to it.
            anchor_task_id = (
                _ensure_project_control_anchor(
                    pconn,
                    kconn,
                    ctx,
                    idempotency_key,
                    token,
                    project_id=canonical_project_id,
                    board_slug=board_slug,
                    name=project_name,
                    description=project_description,
                )
                if mode == "new"
                else _receipt_bound_control_anchor(
                    pconn, kconn, ctx, canonical_project_id,
                )
            )
            with write_txn(pconn):
                _assert_owns_lease(pconn, ctx, idempotency_key, token)
                # The root's assignee is written HERE, not left to the later
                # decompose flip: its lock binds the role, so the row must
                # never exist carrying a route without the role it belongs to.
                root_task_id = kanban_db.create_task(
                    kconn,
                    title=request_title,
                    body=root_body,
                    assignee=root_assignee,
                    created_by=ctx.actor,
                    triage=True,
                    board=board_slug,
                    project_id=canonical_project_id,
                    idempotency_key=root_key,
                    model_override=root_route["model_override"],
                    provider_override=root_route["provider_override"],
                    reasoning_effort=root_route["reasoning_effort"],
                    execution_tier=root_route["execution_tier"],
                    model_policy_lock=root_route["model_policy_lock"],
                    receipt_owned=True,
                )
                root = kanban_db.get_task(kconn, root_task_id)
                if (
                    root is None
                    or root.project_id != canonical_project_id
                    or root.idempotency_key != root_key
                    or root.title != request_title
                    or root.body != root_body
                    or root.assignee != root_assignee
                    or root.model_policy_lock != root_route["model_policy_lock"]
                    or root.model_override != root_route["model_override"]
                    or root.provider_override != root_route["provider_override"]
                    or root.reasoning_effort != root_route["reasoning_effort"]
                    or root.execution_tier != root_route["execution_tier"]
                ):
                    raise OwnerWorkspaceError(
                        "crash_recovery_failed",
                        "the task-graph root does not match the approved proposal",
                    )

            # Deterministic from this receipt's own identity, so a replay that
            # recovers already-committed children activates exactly the rows
            # this receipt parked — never a task the owner postponed since.
            graph_park_generation = kanban_db.park_generation(
                actor=ctx.actor,
                profile=ctx.profile,
                idempotency_key=idempotency_key,
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
                        # Parked in the same insert and never auto-promoted:
                        # the approved graph stays non-claimable until this
                        # receipt's terminal result is durable.
                        auto_promote=False,
                        receipt_owned=True,
                        parked=True,
                        park_generation=graph_park_generation,
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
                "anchor_task_id": anchor_task_id,
                "root_task_id": root_task_id,
                "root_status": root.status,
                "task_ids": child_ids,
                "task_statuses": [task.status for task in task_rows],
                "task_count": len(child_ids),
                "parked_task_ids": list(child_ids),
                "park_generation": graph_park_generation,
            }
        finally:
            kconn.close()

        _update_progress(
            pconn, ctx, idempotency_key, token, task_id=root_task_id,
        )
        _finalize_receipt(
            pconn, ctx, idempotency_key, token, status="committed", result=result,
        )
        # Everything above is parked and non-claimable. Only now — with the
        # terminal receipt durable — does the approved work become runnable, in
        # one idempotent recoverable transition a replay can finish on its own.
        _activate_committed_owner_work(result)
        return result
    finally:
        pconn.close()


def _activate_committed_owner_work(result: Any) -> None:
    """Make one committed owner approval's parked work runnable.

    The single activation transition, driven entirely by the committed receipt
    so a replay after a crash performs exactly the same one. Idempotent on both
    halves: :func:`kanban_db.activate_owner_work` only moves rows still sitting
    in the parked column, and the dispatch-state write preserves an explicit
    owner pause. Never called before the terminal receipt is durable.

    ``parked_dependents`` carries the work a plan's archives newly unblocked,
    each with the exact column it must go back to, so releasing it is the same
    receipt-driven transition rather than a readiness recompute that could have
    run before this receipt existed.

    ``park_generation`` is the receipt's own parking identity, and it is what
    makes replay exact: the parked column is shared with work the owner
    deliberately postponed, so a receipt with no recorded generation releases
    NOTHING rather than matching on the column and un-postponing it.
    """
    if not isinstance(result, dict) or result.get("ok") is not True:
        return
    board_slug = str(result.get("board") or "").strip()
    generation = str(result.get("park_generation") or "").strip()
    if not board_slug:
        return
    parked = result.get("parked_task_ids")
    task_ids = [
        task_id for task_id in (parked if isinstance(parked, list) else [])
        if isinstance(task_id, str)
    ]
    restore_statuses: dict[str, str] = {}
    dependents = result.get("parked_dependents")
    for entry in dependents if isinstance(dependents, list) else []:
        if (
            isinstance(entry, (list, tuple))
            and len(entry) == 2
            and isinstance(entry[0], str)
            and isinstance(entry[1], str)
        ):
            task_ids.append(entry[0])
            restore_statuses[entry[0]] = entry[1]
    if task_ids and generation:
        with contextlib.closing(kanban_db.connect(board=board_slug)) as kconn:
            kanban_db.activate_owner_work(
                kconn,
                task_ids,
                generation=generation,
                restore_statuses=restore_statuses,
            )
    # One milestone approval is the execution authority. Activation is
    # idempotent and preserves an explicit owner pause, so a crash/retry
    # cannot silently resume a Project the owner stopped.
    _set_project_dispatch_state(board_slug, enabled=True)


def _receipt_owns_project(
    conn: sqlite3.Connection, ctx: OwnerContext, project_id: str,
) -> bool:
    """Return whether this trusted owner has a committed Project receipt.

    Fails closed on unreadable authority. Swallowing a malformed committed
    receipt and answering ``False`` turned "this Project's ownership cannot be
    read" into "you do not own this Project" — a 404 on a Project that exists,
    immediately before a mutation. Every candidate row therefore goes through
    the one strict validator, and a row that cannot be read raises
    ``snapshot_unavailable`` instead of being skipped.
    """
    rows = conn.execute(
        "SELECT operation, result_json FROM owner_workspace_receipts "
        "WHERE actor = ? AND profile = ? AND status = 'committed' "
        "AND operation IN ('owner_workspace_bootstrap', 'owner_task_graph_commit')",
        (ctx.actor, ctx.profile),
    ).fetchall()
    for row in rows:
        try:
            committed = decode_committed_owner_receipt(
                row["result_json"], str(row["operation"]),
            )
        except OwnerReceiptUnreadable as exc:
            raise OwnerWorkspaceError(
                "snapshot_unavailable",
                "the Project ownership record could not be read",
            ) from exc
        if committed["project_id"].strip() == project_id:
            return True
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


OWNER_PROJECT_LIFECYCLE_REVISION_CAPABILITY = "lifecycle_revision"
_OWNER_PROJECT_LIFECYCLE_ACTIONS = ("archive", "restore", "pause", "resume")


def _project_lifecycle_revision(
    conn: sqlite3.Connection, ctx: OwnerContext, project_id: str,
) -> int:
    """Return the monotonic effective lifecycle generation for one Project.

    A confirmation timeout is silence rather than an owner decision.  It is
    intentionally retryable under the same idempotency key, so it must not
    advance the optimistic revision that guards that exact retry.  Reading
    and classifying the small receipt set also handles rows migrated from an
    earlier schema where every terminal denial was initially marked as a
    generation.
    """
    columns = {
        str(row[1])
        for row in conn.execute("PRAGMA table_info(owner_workspace_receipts)")
    }
    if "terminal_generation" in columns:
        generation_filter = "AND terminal_generation > 0"
    else:
        # A read-only Workspace request can be the first request after an
        # upgrade, before any mutation has opened projects.db for migration.
        # Mirror _ensure_schema() by treating legacy terminal lifecycle rows as
        # generations while keeping this projection read-only. The first later
        # mutation will add the durable marker normally.
        generation_filter = "AND status IN ('committed', 'denied')"
    legacy_digests = tuple(
        _digest({"project_id": project_id, "action": action})
        for action in _OWNER_PROJECT_LIFECYCLE_ACTIONS
    )
    legacy_digest_placeholders = ", ".join("?" for _ in legacy_digests)
    rows = conn.execute(
        "SELECT * FROM owner_workspace_receipts "
        "WHERE actor = ? AND profile = ? "
        "AND (project_id = ? OR (project_id IS NULL AND request_digest IN ("
        f"{legacy_digest_placeholders}))) "
        "AND operation = 'owner_project_lifecycle' "
        f"{generation_filter}",
        (ctx.actor, ctx.profile, project_id, *legacy_digests),
    ).fetchall()
    return sum(1 for row in rows if not _is_retryable_confirmation_timeout(row))


def list_committed_projects(
    ctx: OwnerContext, *, lifecycle_revision: bool = False,
) -> list[dict]:
    """Read-only projection of projects proven by committed owner receipts.

    Empty means EMPTY. An absent ``projects.db`` is the only state that
    projects as "there are no Projects yet"; a database that exists but cannot
    be opened, is missing this projection's tables, holds a committed receipt
    whose result cannot be read, names a Project that is not there, or names a
    board whose ownership cannot be verified all raise ``snapshot_unavailable``.
    Answering any of those with an empty list told the owner their Projects
    were gone during an authority outage, and invited a caller to plan new work
    as though nothing existed.
    """
    path = projects_db.projects_db_path()
    if not path.is_file():
        return []

    conn = _open_read_only_sqlite(path, label="projects.db")

    try:
        try:
            tables = {
                row["name"]
                for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'table' "
                    "AND name IN ('projects', 'owner_workspace_receipts')"
                )
            }
            if "owner_workspace_receipts" not in tables:
                # The receipt table is created by the first owner operation
                # (see ``_ensure_schema``), so its absence really is "no owner
                # authority has ever been written here" — genuinely empty.
                return []
            if "projects" not in tables:
                raise OwnerWorkspaceError(
                    "snapshot_unavailable", "projects.db schema is unavailable"
                )
            receipts = conn.execute(
                "SELECT operation, result_json FROM owner_workspace_receipts "
                "WHERE actor = ? AND profile = ? AND status = 'committed' "
                "AND operation IN ('owner_workspace_bootstrap', 'owner_task_graph_commit') "
                "ORDER BY updated_at ASC",
                (ctx.actor, ctx.profile),
            ).fetchall()
        except sqlite3.Error as exc:
            raise OwnerWorkspaceError(
                "snapshot_unavailable", "the Project list could not be read"
            ) from exc

        project_ids: list[str] = []
        for receipt in receipts:
            # One strict validator, shared with the ownership check and native
            # crash recovery. A committed receipt nobody can read — including
            # one whose ``project_id`` is missing or empty — is not evidence of
            # "no Project"; it is unreadable authority, and skipping it is what
            # let corrupt authority become an authoritative empty Project list.
            try:
                committed = decode_committed_owner_receipt(
                    receipt["result_json"], str(receipt["operation"]),
                )
            except OwnerReceiptUnreadable as exc:
                raise OwnerWorkspaceError(
                    "snapshot_unavailable",
                    "a committed Project record could not be read",
                ) from exc
            project_id = committed["project_id"].strip()
            if project_id not in project_ids:
                project_ids.append(project_id)

        projects: list[dict] = []
        for project_id in project_ids:
            try:
                row = conn.execute(
                    "SELECT id, slug, name, description, board_slug, archived "
                    "FROM projects WHERE id = ?",
                    (project_id,),
                ).fetchone()
            except sqlite3.Error as exc:
                raise OwnerWorkspaceError(
                    "snapshot_unavailable", "the Project could not be read"
                ) from exc
            if row is None or not row["board_slug"]:
                raise OwnerWorkspaceError(
                    "snapshot_unavailable",
                    "a committed Project is not in projects.db",
                )
            try:
                _assert_board_ownership(row["board_slug"], project_id)
            except OwnerWorkspaceError as exc:
                raise OwnerWorkspaceError(
                    "snapshot_unavailable",
                    "the Project board ownership could not be verified",
                ) from exc
            projection = {
                "project_id": row["id"],
                "slug": row["slug"],
                "name": owner_project_name(row["name"]),
                "description": row["description"],
                "board": row["board_slug"],
                "archived": bool(row["archived"]),
            }
            if lifecycle_revision:
                # Opt-in keeps the legacy closed projection stable during a
                # rolling deploy. The durable marker never disappears when a
                # timed-out receipt is retried, so this generation cannot move
                # backwards while another owner decision is pending.
                projection["lifecycle_revision"] = _project_lifecycle_revision(
                    conn, ctx, project_id,
                )
            projects.append(projection)
        return projects
    finally:
        conn.close()


_OWNER_TASK_RECEIPT_OPERATIONS = (
    "owner_workspace_bootstrap",
    "owner_task_graph_commit",
    "owner_project_plan_commit",
)
# The exact result keys under which each committed receipt records the tasks IT
# owns. Nothing else is owner-kernel work: a card a human added to the same
# board appears in no receipt and is deliberately left alone.
_OWNER_TASK_RECEIPT_KEYS = (
    "task_id",
    "root_task_id",
    "task_ids",
    "created_task_ids",
    "affected_task_ids",
)
# Which of those keys a SUCCESSFUL receipt of each operation must carry, per
# operation. A committed successful receipt that cannot prove the work it owns
# is not evidence of "no owner work" — it is an unreadable authority, so the
# caller fails closed. An empty list is fine (a plan that only edited existing
# tasks created none); a MISSING or malformed key is not.
_OWNER_TASK_RECEIPT_REQUIRED_KEYS = {
    "owner_workspace_bootstrap": ("task_id",),
    "owner_task_graph_commit": ("root_task_id", "task_ids"),
    "owner_project_plan_commit": ("created_task_ids", "affected_task_ids"),
}
# Explicit terminal non-success errors that legitimately own no tasks. Only
# these may contribute nothing.
_OWNER_RECEIPT_EMPTY_ERRORS = frozenset({"conflict", "noop", "no_op", "denied"})


def owner_executor_home():
    """The canonical owner-executor HERMES_HOME, ignoring any profile override.

    Owner receipts are committed by the sole owner-facing coordinator under its
    own ``projects.db``. A route change for a NAMED role scopes HERMES_HOME to
    that role's profile home, so resolving the receipt store from the ambient
    home would scan a different (usually empty) database and silently fence
    nothing. ``get_profile_dir('default')`` resolves against the hermes root
    rather than the context-local override, which is exactly the invariance
    needed here.
    """
    from hermes_cli.profiles import get_profile_dir

    return get_profile_dir("default")


@contextlib.contextmanager
def _owner_executor_scope():
    """Scope reads to the canonical owner-executor store for this block."""
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    token = set_hermes_home_override(str(owner_executor_home()))
    try:
        yield
    finally:
        reset_hermes_home_override(token)


def _receipt_unreadable(detail: str) -> OwnerWorkspaceError:
    return OwnerWorkspaceError("execution_state_busy", detail)


def _owner_receipt_owned_ids(operation: str, result: Any) -> Optional[set[str]]:
    """Return the task ids one committed receipt proves it owns.

    ``None`` means "this receipt is an explicit terminal no-op/conflict and
    owns nothing", which is the only legitimate way for a committed receipt to
    contribute no tasks. Every other shape must prove, per operation, exactly
    which work it created; anything that cannot raises, because treating it as
    empty would let a settings change run over work it cannot see.
    """
    if not isinstance(result, dict) or "ok" not in result:
        raise _receipt_unreadable(
            "an owner work receipt does not record a readable result"
        )
    if result["ok"] is not True:
        error = str(result.get("error") or "").strip().lower()
        if result["ok"] is False and error in _OWNER_RECEIPT_EMPTY_ERRORS:
            return None
        raise _receipt_unreadable(
            "an owner work receipt does not record a readable outcome"
        )
    required = _OWNER_TASK_RECEIPT_REQUIRED_KEYS.get(operation)
    if required is None:
        raise _receipt_unreadable(
            "an owner work receipt names an operation with no known shape"
        )
    owned: set[str] = set()
    for key in _OWNER_TASK_RECEIPT_KEYS:
        if key not in result:
            if key in required:
                raise _receipt_unreadable(
                    "an owner work receipt does not record the work it created"
                )
            continue
        value = result[key]
        if isinstance(value, str) and value.strip():
            owned.add(value.strip())
        elif isinstance(value, list) and all(
            isinstance(item, str) and item.strip() for item in value
        ):
            owned.update(item.strip() for item in value)
        else:
            raise _receipt_unreadable(
                "an owner work receipt records unreadable work identity"
            )
    return owned


def _owner_receipt_task_ids() -> dict[str, set[str]]:
    """Map board slug -> exact task ids proven created by committed receipts.

    Derived from the durable receipts themselves rather than from any public
    id or key prefix: a task belongs to the owner kernel only because a
    committed receipt records having created it. Read from the canonical
    owner-executor store (see :func:`owner_executor_home`), never from whatever
    profile the caller happens to be scoped to.

    Fails closed on every ambiguity — an unreadable store, a malformed
    successful receipt, a receipt whose project cannot be resolved, and a
    project whose board ownership cannot be proven all raise rather than
    reporting "no owner work", because that answer would let a settings change
    proceed over work it cannot see.
    """
    with _owner_executor_scope():
        path = projects_db.projects_db_path()
        if not path.is_file():
            # No projects.db at all: there are provably no owner Projects yet.
            return {}
        try:
            conn = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)
            conn.row_factory = sqlite3.Row
        except sqlite3.Error as exc:
            raise _receipt_unreadable(
                "existing owner work could not be read before the change"
            ) from exc
        try:
            tables = {
                row["name"]
                for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'table' "
                    "AND name IN ('projects', 'owner_workspace_receipts')"
                )
            }
            if not tables & {"owner_workspace_receipts"}:
                # No receipt table means no committed owner receipt can exist.
                return {}
            if tables != {"projects", "owner_workspace_receipts"}:
                # Receipts exist but their Projects cannot be resolved.
                raise _receipt_unreadable(
                    "existing owner work could not be read before the change"
                )
            placeholders = ",".join("?" for _ in _OWNER_TASK_RECEIPT_OPERATIONS)
            by_project: dict[str, set[str]] = {}
            for receipt in conn.execute(
                "SELECT operation, project_id, result_json FROM owner_workspace_receipts "
                f"WHERE status = 'committed' AND operation IN ({placeholders})",
                _OWNER_TASK_RECEIPT_OPERATIONS,
            ):
                try:
                    result = json.loads(receipt["result_json"] or "null")
                except (TypeError, ValueError) as exc:
                    raise _receipt_unreadable(
                        "an owner work receipt could not be read before the change"
                    ) from exc
                owned = _owner_receipt_owned_ids(
                    str(receipt["operation"]), result
                )
                if owned is None:
                    continue
                project_id = str(result.get("project_id") or receipt["project_id"] or "")
                if not project_id:
                    raise _receipt_unreadable(
                        "an owner work receipt does not record which Project it belongs to"
                    )
                by_project.setdefault(project_id, set()).update(owned)

            boards: dict[str, set[str]] = {}
            for project_id, task_ids in by_project.items():
                row = conn.execute(
                    "SELECT board_slug FROM projects WHERE id = ?", (project_id,)
                ).fetchone()
                if row is None or not row["board_slug"]:
                    raise _receipt_unreadable(
                        "an owner work receipt does not resolve to a board"
                    )
                board_slug = str(row["board_slug"])
                # Ownership must be provable. An unprovable binding is not
                # "someone else's problem": it means this fence cannot tell
                # whose work these tasks are, so it must not proceed.
                _assert_board_ownership(board_slug, project_id)
                boards.setdefault(board_slug, set()).update(task_ids)
            return boards
        except sqlite3.Error as exc:
            raise _receipt_unreadable(
                "existing owner work could not be read before the change"
            ) from exc
        finally:
            conn.close()


# How long the fence waits for a board's dispatch lock before failing closed.
# Matches the owner pause/resume wait: long enough to ride out one in-flight
# tick's critical section, short enough that a settings change never hangs.
_FENCE_LOCK_WAIT_SECONDS = 5.0


@contextlib.contextmanager
def _fence_board_connection(board_slug: str):
    """Open one board for fencing while holding its native dispatch lock.

    The lock is the board's own dispatcher guard, so for as long as this is
    held no dispatcher tick on this board can be inside its critical section:
    no new claim starts and no worker spawns. Discovery and the pin therefore
    see the same rows, instead of a claim landing between "these are unlocked"
    and "these are now pinned".

    Acquired UNDER the caller's profile route lock, in sorted board order, and
    never the other way around, so the guards cannot deadlock. Fails closed on
    a busy lock — the caller must not write a route change it could not fence.
    """
    with contextlib.ExitStack() as stack:
        try:
            stack.enter_context(
                kanban_db.board_dispatch_lock(
                    board_slug, wait_seconds=_FENCE_LOCK_WAIT_SECONDS
                )
            )
            conn = stack.enter_context(
                contextlib.closing(kanban_db.connect(board=board_slug))
            )
        except (OSError, TimeoutError, sqlite3.Error) as exc:
            raise OwnerWorkspaceError(
                "execution_state_busy",
                "existing owner work could not be pinned before the change",
            ) from exc
        yield conn


def _owner_tasks_for_role(
    conn: sqlite3.Connection, task_ids: set[str], assignee: str
) -> list[str]:
    """Narrow receipt-created ids to the ones this role currently holds."""
    if not task_ids:
        return []
    ordered = sorted(task_ids)
    placeholders = ",".join("?" for _ in ordered)
    rows = conn.execute(
        f"SELECT id FROM tasks WHERE id IN ({placeholders}) "
        "AND assignee = ? AND task_kind = 'work'",
        (*ordered, assignee),
    ).fetchall()
    return [str(row["id"]) for row in rows]


def fence_effective_task_routes(profile: str) -> list[str]:
    """Freeze owner work on an exact approved route, before a route change.

    Provider selection rewrites a role's native route, which is part of the
    *effective* route of every owner task that does not already carry an exact
    owner-approved lock. Called immediately ahead of that write — under the same
    serialization as the write (see :func:`profile_route_lock`) — this pins each
    exposed task's route so a settings change can only ever affect newly
    approved work.

    EVERY receipt-owned executable task this role holds is covered, including
    one that already names a model, provider and effort but carries no lock:
    "fully specified" is not the same as "owner-approved", and an unlocked
    executable owner task must never be dispatchable as ordinary work. A task
    whose completed route is not an admitted authority cannot be pinned at all,
    so it is parked in the native blocked column with a re-approval
    requirement rather than left runnable.

    Only tasks a committed owner receipt records having created, and that this
    role currently holds, are touched; manual and non-Raphael Kanban cards are
    never mutated, so a role with no such work is a no-op and needs no route
    read at all. When there IS such work it fails closed: if the role's current
    route cannot be read and validated the caller must not write, because those
    tasks would otherwise silently inherit the new route. Returns the pinned
    ids.

    Work that is ALREADY claimed or running cannot be fenced at all: a worker
    is executing under the authority the row carries right now, and rewriting
    its route columns underneath it would neither stop that process nor
    re-authorize it. The whole change is refused (``execution_state_busy``) so
    a stale claim can never launch or continue under changed authority.

    Every board's native dispatch lock is held — in sorted slug order, under
    the caller's profile route lock — across discovery, the active-work check,
    the route read and the pin, so no dispatcher tick can claim or spawn in
    between. Any failure propagates before the caller's config write, so the
    profile is left untouched and the caller's own rollback still applies.
    """
    boards = _owner_receipt_task_ids()
    if not boards:
        return []

    pinned: list[str] = []
    with contextlib.ExitStack() as stack:
        exposed: list[tuple[sqlite3.Connection, list[str]]] = []
        for board_slug in sorted(boards):
            conn = stack.enter_context(_fence_board_connection(board_slug))
            try:
                candidates = _owner_tasks_for_role(conn, boards[board_slug], profile)
                if not kanban_db.count_unpinned_owner_tasks(conn, task_ids=candidates):
                    continue
                active = kanban_db.list_active_unpinned_owner_tasks(
                    conn, task_ids=candidates,
                )
            except sqlite3.Error as exc:
                raise OwnerWorkspaceError(
                    "execution_state_busy",
                    "existing owner work could not be read before the change",
                ) from exc
            if active:
                raise OwnerWorkspaceError(
                    "execution_state_busy",
                    "some of this role's existing work is running right now; "
                    "try this change again once it has finished",
                )
            exposed.append((conn, candidates))
        if not exposed:
            return []

        try:
            route = configured_assignment_for(profile)
        except (ValueError, OSError) as exc:
            raise OwnerWorkspaceError(
                "invalid_model_route",
                "the role's current route could not be confirmed before the change",
            ) from exc

        for conn, candidates in exposed:
            try:
                pinned.extend(
                    kanban_db.pin_effective_task_routes(
                        conn,
                        task_ids=candidates,
                        model=route.model,
                        provider=route.provider,
                        reasoning_effort=route.reasoning_effort,
                    )
                )
            except (ValueError, sqlite3.Error) as exc:
                raise OwnerWorkspaceError(
                    "invalid_model_route",
                    "existing owner work could not be pinned before the change",
                ) from exc
    return pinned


@contextlib.contextmanager
def profile_route_lock(profile: str):
    """Hold the cross-process lock that orders one profile's route changes.

    Every surface that can move a role's model, provider, effort or fallback
    policy — the Models endpoint, ``/api/model/set``, the config editors, the
    CLI, ``/model``, ``/reasoning``, the TUI and setup — takes this lock, so
    the pre-read, the compare-and-swap, the policy validation, the legacy fence
    and the write itself cannot interleave with another writer's. It is a
    kernel-held file lock keyed on the profile, not an in-process mutex, so
    concurrent dashboard, CLI and worker processes are ordered too. Fails
    closed: with no lock there is no write.

    The caller owns the ordering INSIDE the lock (see
    ``hermes_cli.config.profile_route_write``), because a compare-and-swap has
    to see the same state the validation and the fence do.
    """
    canonical = str(profile or "").strip().lower() or "default"
    with _global_guard(f"profile-route-{canonical}", label="profile route"):
        yield


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


_OWNER_TASK_PIN_COLUMNS = (
    "assignee",
    "provider_override",
    "model_override",
    "reasoning_effort",
    "execution_tier",
    "model_policy_lock",
)


def owner_task_pin_select(conn: sqlite3.Connection, alias: str) -> str:
    """Return the pin columns this database actually has, as a SELECT tail.

    Read-only owner projections must never assume a board has been migrated:
    an older ``kanban.db`` genuinely cannot carry a lock, so omitting the
    missing columns projects those tasks as unlocked — without a schema write
    hidden inside a GET.

    Two things are NOT tolerated. A schema read that fails is not evidence that
    a board is old: it means this projection cannot know whether a lock exists,
    so it raises ``snapshot_unavailable``. And a board that HAS
    ``model_policy_lock`` but is missing any column the lock's digest binds
    cannot prove a pin either, so the lock column is still projected and the
    pin reads back as invalid/unknown rather than as unlocked.
    """
    try:
        present = {
            str(row[1])
            for row in conn.execute("PRAGMA table_info(tasks)")
        }
    except sqlite3.Error as exc:
        raise OwnerWorkspaceError(
            "snapshot_unavailable",
            "this Project's work could not be read right now",
        ) from exc
    if "model_policy_lock" not in present:
        # Without the lock column there is no pin to project at all.
        return ""
    selected = [
        column for column in _OWNER_TASK_PIN_COLUMNS if column in present
    ]
    return "".join(f", {alias}.{column}" for column in selected)


@dataclass(frozen=True)
class OwnerTaskRoutePin:
    """One task's persisted route authority, or the fact that it is broken.

    ``valid`` False is deliberately NOT the same as "no pin": a task carrying a
    lock this build cannot validate has unknown routing, so every projection
    must report unknown rather than fall back to the looser role-level check
    that an unlocked task legitimately uses.
    """

    valid: bool
    profile: str = ""
    provider: str = ""
    model: str = ""
    reasoning_effort: str = ""


def owner_task_route_pin(row: Any) -> Optional[OwnerTaskRoutePin]:
    """Return one task's persisted route authority, or None when unlocked.

    ``None`` only for a row that cannot carry a lock at all (a pre-lock schema,
    projected without the lock column) or that carries none — an ordinary,
    manual, or pre-lock task, which keeps the older role-level check. A lock
    that does not validate against this build's policy for this exact
    assignee/provider/model/effort/tier returns an INVALID pin, and so does a
    lock whose bound columns are not all present: a partially migrated row
    proves nothing, and must never read as unlocked.
    """
    try:
        lock = row["model_policy_lock"]
    except (IndexError, KeyError, TypeError):
        # The row was projected without the lock column at all: this schema
        # cannot carry a lock, so the task is genuinely unlocked.
        return None
    if not lock:
        return None
    try:
        profile = str(row["assignee"] or "").strip()
        provider = str(row["provider_override"] or "").strip()
        model = str(row["model_override"] or "").strip()
        effort = str(row["reasoning_effort"] or "").strip().lower()
    except (IndexError, KeyError, TypeError):
        # Locked, but the digest's bound columns are not all here: unprovable.
        return OwnerTaskRoutePin(valid=False)
    if kanban_db.task_policy_lock_error(row):
        return OwnerTaskRoutePin(valid=False)
    return OwnerTaskRoutePin(
        valid=True,
        profile=profile,
        provider=provider,
        model=model,
        reasoning_effort=effort,
    )


def _owner_project_runtime_and_cost(
    run: kanban_db.Run, task_pin: Optional[OwnerTaskRoutePin],
) -> tuple[dict, dict]:
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
    if task_pin is not None:
        # A policy-locked task is answerable to its OWN pinned authority, not
        # to any route the role happens to admit: a run that used an admitted
        # but different model is exactly the silent switch this receipt must
        # catch. An invalid lock proves nothing at all, so it reads as unknown.
        if not task_pin.valid or (profile, provider, model, effort.lower()) != (
            task_pin.profile,
            task_pin.provider,
            task_pin.model,
            task_pin.reasoning_effort,
        ):
            return dict(_OWNER_UNKNOWN_RUNTIME), dict(_OWNER_UNKNOWN_COST)
    else:
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


def _owner_project_run_receipt(
    run: kanban_db.Run, task_pin: Optional[OwnerTaskRoutePin],
) -> dict:
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
    runtime, cost = _owner_project_runtime_and_cost(run, task_pin)
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
    run: kanban_db.Run,
    task_title: Any,
    *,
    task_pin: Optional[OwnerTaskRoutePin],
    has_newer_run: bool,
    run_context: bool,
) -> dict:
    """Project one run for the owner, carrying its retry fact but never its id.

    ``task_pin`` is the run's own task's persisted owner-approved route, so a
    recorded run that drifted off it reads as unknown rather than confirmed.

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
        "receipt": _owner_project_run_receipt(run, task_pin),
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

        worker_rows = kanban_db.verified_active_worker_rows(
            conn, project_id=project_id,
        )[:_OWNER_PROJECT_MAX_WORKERS + 1]

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

        # The task's own pinned route travels with its runs: a receipt is
        # checked against the route approved for THAT task, not against
        # whatever the role currently admits. Selected schema-aware because
        # this is a read-only connection — a board whose kanban.db predates the
        # pin columns must project as "unlocked", never fail and never migrate.
        run_rows = conn.execute(
            "SELECT r.*, t.title AS task_title"
            + owner_task_pin_select(conn, "t")
            + " FROM task_runs r JOIN tasks t ON t.id = r.task_id "
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
                    task_pin=owner_task_route_pin(row),
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

    # The authority row exists from here on, so 404 is off the table: a
    # storage or integrity failure is not proof that the attachment is not
    # there, and telling the owner "not found" for one hides a real fault and
    # invites them to re-upload a file that already exists.
    root = (
        kanban_db.kanban_home() / "kanban" / "attachments"
        if board_slug == kanban_db.DEFAULT_BOARD
        else kanban_db.board_dir(board_slug) / "attachments"
    ).resolve()
    try:
        stored = Path(str(row["stored_path"])).resolve()
        stored.relative_to(root)
    except (OSError, ValueError) as exc:
        # The row points outside this board's attachment store: that is broken
        # authority, not an absent file.
        raise OwnerWorkspaceError(
            "snapshot_unavailable", "the attachment record could not be verified"
        ) from exc

    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        fd = os.open(stored, flags)
    except OSError as exc:
        raise OwnerWorkspaceError(
            "snapshot_unavailable", "the attachment could not be read"
        ) from exc
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
            raise OwnerWorkspaceError(
                "snapshot_unavailable",
                "the stored attachment does not match its record",
            )
        chunks: list[bytes] = []
        remaining = size
        while remaining:
            chunk = os.read(fd, min(remaining, 1024 * 1024))
            if not chunk:
                raise OwnerWorkspaceError(
                    "snapshot_unavailable",
                    "the stored attachment does not match its record",
                )
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(fd, 1):
            raise OwnerWorkspaceError(
                "snapshot_unavailable",
                "the stored attachment does not match its record",
            )
    except OSError as exc:
        raise OwnerWorkspaceError(
            "snapshot_unavailable", "the attachment could not be read"
        ) from exc
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


def list_owner_decisions(ctx: OwnerContext) -> dict:
    """Project pending native gates into one owner-safe read-only inbox.

    Returns ``{"data": [...], "truncated": bool}``. The window is bounded both
    per Project and globally, and ``truncated`` says whether anything was left
    out — an owner who cannot tell a full inbox from a clipped one can believe
    they have answered everything Raphael is waiting on.

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
    truncated = False
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

        # ``LIMIT`` is one more than the window, so a full page proves there
        # is at least one more decision this Project could not carry.
        if len(rows) > _OWNER_DECISIONS_LIMIT:
            truncated = True
            rows = rows[:_OWNER_DECISIONS_LIMIT]
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
    return {
        "data": decisions[:_OWNER_DECISIONS_LIMIT],
        "truncated": truncated or len(decisions) > _OWNER_DECISIONS_LIMIT,
    }


def set_project_archived(
    ctx: OwnerContext,
    *,
    idempotency_key: str,
    project_id: str,
    expected_revision: Any,
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
    if type(expected_revision) is not int or expected_revision < 0:
        raise OwnerWorkspaceError(
            "invalid_argument", "expected_revision must be a non-negative integer",
        )
    action = str(action or "").strip().lower()
    if action not in _OWNER_PROJECT_LIFECYCLE_ACTIONS:
        raise OwnerWorkspaceError(
            "invalid_argument",
            "action must be 'archive', 'restore', 'pause', or 'resume'",
        )
    operation = "owner_project_lifecycle"
    _require_owner_run_authority(
        ctx,
        operation=operation,
        idempotency_key=idempotency_key,
        payload={
            "idempotency_key": idempotency_key,
            "project_id": project_id,
            "expected_revision": expected_revision,
            "action": action,
        },
    )
    payload = {
        "project_id": project_id,
        "expected_revision": expected_revision,
        "action": action,
    }
    digest = _digest(payload)
    target_archived = action == "archive"

    pconn = projects_db.connect()
    try:
        _ensure_schema(pconn)
        replay = _terminal_replay(
            pconn, ctx, idempotency_key, operation, digest,
        )
        if replay is not None:
            return replay
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
        if _project_lifecycle_revision(pconn, ctx, project_id) != expected_revision:
            raise OwnerWorkspaceError(
                "stale_revision",
                "the Project changed after it was shown; refresh before trying again",
            )

        state, row, token = _acquire_or_replay(
            pconn, ctx, idempotency_key, operation, digest,
        )
        if state == "terminal":
            return json.loads(row["result_json"])

        # Bind this receipt to the Project before either denial or mutation.
        # The read-only Project projection can then expose a monotonic terminal
        # lifecycle generation without parsing result JSON or counting an
        # in-flight receipt that must retain its idempotent identity.
        _update_progress(
            pconn, ctx, idempotency_key, token,
            project_id=project_id, board_slug=project.board_slug,
        )

        approval = _confirm(
            ctx,
            operation=operation,
            digest=digest,
            description=f"{action.title()} Project {owner_project_name(project.name)!r}",
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

        # Serialize every lifecycle writer on the Project's actual board from
        # the final compare-and-swap through durable receipt finalization. A
        # second approved action may already be waiting here, but it cannot
        # observe the old revision after the first mutation has landed.
        with _global_board_guard(project.board_slug):
            with write_txn(pconn):
                _assert_owns_lease(pconn, ctx, idempotency_key, token)
                current = projects_db.get_project(pconn, project_id)
                if current is None or not current.board_slug:
                    raise OwnerWorkspaceError(
                        "project_not_found", "the Project is unavailable",
                    )
                metadata = kanban_db.read_board_metadata(current.board_slug)
                current_revision = _project_lifecycle_revision(
                    pconn, ctx, project_id,
                )
                if current_revision != expected_revision:
                    result = {
                        "ok": False,
                        "error": "conflict",
                        "archived": bool(current.archived),
                        "execution_paused": bool(
                            metadata.get("dispatch_paused_by_owner")
                        ),
                    }
                elif action in {"pause", "resume"} and current.archived:
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
    r"^(?:(?:R(?:0[1-9]|1\d|2[0-5])|B(?:0[1-9]|1[0-2]))\s*(?:[—–:-])\s*)+",
    re.IGNORECASE,
)
_OWNER_TITLE_LIMIT = 240
_OWNER_PROJECT_NAME_LIMIT = 160
# What a projection returns when the input canonicalizes to nothing. Read-side
# placeholders for absent text, never a value to test against: an owner may
# legitimately write either of these exact strings, so the native write boundary
# reads emptiness from the canonical text itself instead of comparing to these.
_UNTITLED_WORK_ITEM = "Untitled work item"
_UNTITLED_PROJECT = "Untitled Project"

# LONE SURROGATES (U+D800-U+DFFF) are not Unicode scalar values at all. They
# survive inside a ``str`` — ``json.loads('"\\ud800"')`` produces one from a
# request body, and ``os.fsdecode`` produces U+DC80-U+DCFF for every byte of a
# path that is not valid UTF-8 — but encoding one raises
# ``UnicodeEncodeError``, so a name carrying one turns an owner read into a 500
# from FastAPI's JSON encoder instead of a projected name.
#
# Default-ignorable code points are removed by the shared Unicode 17.0 helper
# in tools.ansi_strip. This boundary adds only invalid lone surrogates.
_OWNER_DISPLAY_STRIP_RE = re.compile("[\ud800-\udfff]")
_OWNER_PRIVATE_WORK_ITEM_PATTERNS = (
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"\bgh[opsu]_[A-Za-z0-9]{20,}\b"),
    re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b"),
    re.compile(r"\bsk-[A-Za-z0-9_-]{16,}\b"),
    re.compile(r"\bBearer\s+[A-Za-z0-9._-]{16,}", re.IGNORECASE),
    re.compile(
        r"\b(?:api[_-]?key|secret|password|passwd|access[_-]?token)"
        r"\s*[:=]\s*\S+",
        re.IGNORECASE,
    ),
    re.compile(r"\b[0-9a-f]{64}\b", re.IGNORECASE),
    re.compile(
        r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-"
        r"[0-9a-f]{12}\b",
        re.IGNORECASE,
    ),
    re.compile(r"\b(?:task|run|resp|claim)[_-][0-9a-z]{6,}\b", re.IGNORECASE),
    re.compile(r"\braphael-[a-z0-9][a-z0-9_-]{2,}\b", re.IGNORECASE),
    re.compile(r"(?:^|\s)(?:/[A-Za-z0-9._-]+){2,}(?:\s|$)"),
    re.compile(r"\b[A-Za-z]:\\(?:[^\\\s]+\\)+[^\\\s]+"),
    re.compile(
        r"\bclaude\s+(?:opus|sonnet|haiku)(?:\s*\d+(?:\.\d+)*)?\b",
        re.IGNORECASE,
    ),
    re.compile(r"\bgpt(?:[-\s]?\d[A-Za-z0-9._-]*)\b", re.IGNORECASE),
    re.compile(
        r"\b(?:gpt|claude|gemini|llama|mistral|deepseek|qwen|grok)-"
        r"[A-Za-z0-9._-]*\d[A-Za-z0-9._-]*\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:openai|anthropic|google|aws|azure|bedrock)\s*[:/]\s*"
        r"[A-Za-z0-9._-]+\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:openai|anthropic|claude|gpt|gemini|llama|mistral|deepseek|"
        r"qwen|grok|bedrock|opus|sonnet|haiku)\b",
        re.IGNORECASE,
    ),
)
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
    annotation frame. :data:`_OWNER_DISPLAY_STRIP_RE` adds the removals that
    belong to THIS boundary and not to a global threat scan — the
    default-ignorable controls, the plain bidi marks and the lone surrogates.
    Doing all of it before redaction means a credential cannot survive by
    splitting its own token, or a query key its own name, across a character
    the owner cannot see; doing it at all means the text cannot reorder or
    hide what the owner is shown, cannot spend the display bound on nothing,
    and cannot break the JSON encoding of the response carrying it. Safe human
    Unicode is left exactly as written. Whitespace collapse (which also folds
    U+2028/U+2029), the optional internal-prefix strip, and the ``limit``
    Unicode code point bound then apply to already-sanitized,
    already-redacted text.

    ``strip_unicode_tags`` runs once more AFTER that bound, and the order
    matters: the first pass preserves the three pinned RGI subdivision flags
    whole, so the code point slice can land inside one and leave dangling
    invisible tag characters behind a visible U+1F3F4 base — exactly the
    smuggling frame the boundary exists to remove, and what the Workspace
    correctly rejects. Re-running the same pinned sanitizer on the bounded
    text keeps an intact flag intact and reduces a cut one to its visible
    black-flag base, with no grapheme parsing of our own.

    The final ``strip`` closes that tail: the bound is a raw code point slice,
    so it can cut mid-word and leave the whitespace before it stranded at the
    end. Without the strip a projected string would not survive being
    projected again — the second pass's whitespace collapse would shorten it —
    and every surface comparing a response against the canonical projection
    treats that disagreement as a rejection.
    """
    from agent.redact import redact_sensitive_text
    from tools.ansi_strip import (
        sanitize_display_text,
        strip_default_ignorables,
        strip_unicode_tags,
    )
    from tools.threat_patterns import INVISIBLE_CHARS

    text = strip_default_ignorables(sanitize_display_text(str(value or "")))
    text = _OWNER_DISPLAY_STRIP_RE.sub("", text)
    text = "".join(char for char in text if char not in INVISIBLE_CHARS)
    text = redact_sensitive_text(text, force=True, redact_url_credentials=True)
    text = " ".join(text.split())
    if strip_internal_prefix:
        text = _INTERNAL_TITLE_PREFIX.sub("", text).strip()
    return strip_unicode_tags(text[:limit]).strip()


def owner_title(value: Any) -> str:
    """Return the canonical owner-safe, single-line work-item title.

    See :func:`_owner_display_text` for the egress contract. A work-item
    title additionally loses the internal ``B03 — ``-style dispatcher prefix,
    which is Raphael's own bookkeeping and means nothing to the owner.
    """
    title = _owner_display_text(
        value, limit=_OWNER_TITLE_LIMIT, strip_internal_prefix=True
    )
    if not title or any(
        pattern.search(title) for pattern in _OWNER_PRIVATE_WORK_ITEM_PATTERNS
    ):
        return _UNTITLED_WORK_ITEM
    return title


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
    ) or _UNTITLED_PROJECT


def _native_owner_title(value: Any, field: str) -> str:
    """Canonicalize one work-item title Hermes itself is about to write.

    Hermes is not only a projection surface: ``owner_task_graph_commit`` and
    ``owner_project_plan_commit`` write task titles straight into
    ``kanban.db``, and every owner surface reads them back through
    :func:`owner_title`. Canonicalizing at input — BEFORE the request digest,
    the owner's approval description, persistence and the replay comparison
    all bind to this exact string — is what keeps the stored title and its
    projection the same text, so a caller that reaches this kernel directly
    cannot store a title that only becomes safe on the way back out.

    The ``limit`` is unchanged (:data:`_OWNER_TITLE_LIMIT`, still enforced by
    :func:`_bounded_text` as a rejection, not a truncation, of raw input);
    :func:`_owner_display_text` — the one core :func:`owner_title` projects
    through — supplies the rest of the contract, including the same secret
    redaction and the internal-prefix strip.

    It is called directly rather than through :func:`owner_title` so that
    emptiness is read from the canonical text itself. A value that
    canonicalizes to nothing carries no owner-visible text at all — it was
    invisible characters, or nothing but an internal dispatcher prefix — and
    storing the read-side placeholder would record a title the owner never
    wrote, so it is rejected. Testing the projection against that placeholder
    instead would also reject ``Untitled work item`` as a literal title, which
    is owner-visible text like any other and persists as written.
    """
    title = _owner_display_text(
        _bounded_text(value, field, limit=_OWNER_TITLE_LIMIT),
        limit=_OWNER_TITLE_LIMIT,
        strip_internal_prefix=True,
    )
    if not title:
        raise OwnerWorkspaceError(
            "invalid_argument", f"{field} has no owner-visible text",
        )
    if any(pattern.search(title) for pattern in _OWNER_PRIVATE_WORK_ITEM_PATTERNS):
        raise OwnerWorkspaceError(
            "invalid_argument", f"{field} contains private operational detail",
        )
    return title


def _native_owner_project_name(value: Any, field: str) -> str:
    """Canonicalize one Project name Hermes itself is about to write.

    Same native-write contract as :func:`_native_owner_title`, against the
    core :func:`owner_project_name` projects through: the 160 code point bound,
    no internal-prefix strip, and canonical emptiness rejected rather than
    stored as the ``Untitled Project`` placeholder. A Project an owner really
    named ``Untitled Project`` is owner text and is written as it stands.
    """
    name = _owner_display_text(
        _bounded_text(value, field, limit=_OWNER_PROJECT_NAME_LIMIT),
        limit=_OWNER_PROJECT_NAME_LIMIT,
    )
    if not name:
        raise OwnerWorkspaceError(
            "invalid_argument", f"{field} has no owner-visible text",
        )
    return name


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
    if not items:
        # Only the Project's non-executable control anchor exists: nothing has
        # been approved yet, so there is no work to be complete.
        execution_state = "waiting_for_approval"
        execution_summary = (
            "Raphael is waiting for an approved milestone before starting work."
        )
    elif not active_rows and not decision_rows and not attention_rows:
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
    required = {
        "title", "body", "assignee", "execution_tier",
    } | ({"parents"} if parent_limit is not None else set())
    allowed = required | {"responsibility"}
    if not isinstance(value, dict) or not required.issubset(value) or not set(value).issubset(allowed):
        raise OwnerWorkspaceError(
            "invalid_argument",
            f"{field} must contain {sorted(required)} and only optional responsibility",
        )
    raw = value
    from agent.redact import redact_sensitive_text

    result = {
        "title": _native_owner_title(raw["title"], f"{field}.title"),
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
    result.update(
        _resolved_route_pin(
            result["assignee"], raw["execution_tier"], f"{field}.execution_tier",
        )
    )
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
                "action", "reason", "title", "body", "assignee", "execution_tier",
                "existing_parents", "new_parents",
            }
            allowed = required | {"responsibility"}
            if not required.issubset(item) or not set(item).issubset(allowed):
                raise OwnerWorkspaceError(
                    "invalid_argument",
                    f"{field} must contain {sorted(required)} and only optional "
                    "responsibility",
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
                {
                    **{
                        key: raw[key]
                        for key in ("title", "body", "assignee", "execution_tier")
                    },
                    "responsibility": raw.get("responsibility"),
                },
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


def _resolve_existing_project_board(
    pconn: sqlite3.Connection, project_id: str, *, allow_archived: bool = False,
):
    project = projects_db.get_project(pconn, project_id)
    if (
        project is None
        or (project.archived and not allow_archived)
        or not project.board_slug
    ):
        raise OwnerWorkspaceError("project_not_found", f"project {project_id!r} is unavailable")
    if not kanban_db.board_exists(project.board_slug):
        raise OwnerWorkspaceError("project_not_found", "the Project board is unavailable")
    _assert_board_ownership(project.board_slug, project_id)
    return project, project.board_slug


_OWNER_ANCHOR_RECEIPT_FIELDS = {
    "owner_workspace_bootstrap": "task_id",
    "owner_task_graph_commit": "anchor_task_id",
    "owner_project_plan_commit": "anchor_task_id",
}
_LEGACY_GRAPH_RESULT_FIELDS = frozenset({
    "ok", "mode", "project_id", "project_slug", "board",
    "root_task_id", "root_status", "task_ids", "task_statuses", "task_count",
})
_LEGACY_PLAN_RESULT_FIELDS = frozenset({
    "ok", "project_id", "project_slug", "board", "risk_level", "applied",
    "change_count", "created_task_ids", "affected_task_ids",
    "executable_task_count",
})
_LEGACY_PLAN_RISK_LEVELS = frozenset({"standard", "significant_removal"})
_LEGACY_ANCHOR_MIGRATION_SALT = "legacy-project-anchor-v1"


@dataclass(frozen=True)
class _ProjectAnchorResolution:
    task_id: Optional[str]
    migration_key: Optional[str] = None


def _receipt_named_anchors(
    pconn: sqlite3.Connection, ctx: OwnerContext, project_id: str,
) -> set[str]:
    anchors: set[str] = set()
    placeholders = ", ".join("?" for _ in _OWNER_ANCHOR_RECEIPT_FIELDS)
    rows = pconn.execute(
        "SELECT operation, result_json FROM owner_workspace_receipts "
        "WHERE actor = ? AND profile = ? AND status = 'committed' "
        f"AND operation IN ({placeholders})",
        (ctx.actor, ctx.profile, *_OWNER_ANCHOR_RECEIPT_FIELDS),
    ).fetchall()
    for row in rows:
        try:
            result = json.loads(row["result_json"] or "null")
        except (TypeError, ValueError) as exc:
            raise OwnerWorkspaceError(
                "crash_recovery_failed",
                "an owner Project receipt could not be read",
            ) from exc
        if not isinstance(result, dict) or result.get("project_id") != project_id:
            continue
        operation = str(row["operation"])
        task_id = result.get(_OWNER_ANCHOR_RECEIPT_FIELDS[operation])
        if not isinstance(task_id, str) or not task_id.strip():
            if operation == "owner_workspace_bootstrap":
                raise OwnerWorkspaceError(
                    "crash_recovery_failed",
                    "an owner Project receipt does not record its Project anchor",
                )
            continue
        anchors.add(task_id.strip())
    return anchors


def _board_control_rows(kconn: sqlite3.Connection, project_id: str) -> list[str]:
    return [
        str(row["id"])
        for row in kconn.execute(
            "SELECT id FROM tasks WHERE project_id = ? AND task_kind = 'control' "
            "ORDER BY id",
            (project_id,),
        ).fetchall()
    ]


def _receipt_bound_control_anchor(
    pconn: sqlite3.Connection,
    kconn: sqlite3.Connection,
    ctx: OwnerContext,
    project_id: str,
) -> str:
    """Resolve the Project's one canonical, receipt-bound control anchor."""
    anchors = _receipt_named_anchors(pconn, ctx, project_id)
    if len(anchors) != 1:
        raise OwnerWorkspaceError(
            "project_not_owned",
            "this Project has no single receipt-bound anchor to apply a plan to",
        )
    anchor_task_id = next(iter(anchors))
    if _board_control_rows(kconn, project_id) != [anchor_task_id]:
        raise OwnerWorkspaceError(
            "project_not_owned",
            "this Project's anchor could not be confirmed on its board",
        )
    return anchor_task_id


def _legacy_anchor_migration_key(
    pconn: sqlite3.Connection,
    kconn: sqlite3.Connection,
    ctx: OwnerContext,
    project_id: str,
    *,
    project_slug: str,
    board_slug: str,
) -> Optional[str]:
    """Return a stable migration key only for the known pre-anchor receipt shape."""
    graphs: list[tuple[str, dict]] = []
    for row in pconn.execute(
        "SELECT idempotency_key, operation, project_id, board_slug, task_id, "
        "result_json FROM owner_workspace_receipts "
        "WHERE actor = ? AND profile = ? AND status = 'committed' "
        "AND operation IN ('owner_task_graph_commit', 'owner_project_plan_commit')",
        (ctx.actor, ctx.profile),
    ).fetchall():
        try:
            result = json.loads(row["result_json"] or "null")
        except (TypeError, ValueError) as exc:
            raise OwnerWorkspaceError(
                "crash_recovery_failed",
                "an owner Project receipt could not be read",
            ) from exc
        if not isinstance(result, dict):
            raise OwnerWorkspaceError(
                "crash_recovery_failed",
                "an owner Project receipt could not be read",
            )
        result_project_id = result.get("project_id")
        row_project_id = row["project_id"]
        result_project_id = (
            result_project_id.strip()
            if isinstance(result_project_id, str)
            else ""
        )
        row_project_id = row_project_id.strip() if isinstance(row_project_id, str) else ""
        if result_project_id != project_id and row_project_id != project_id:
            continue
        if (
            result_project_id != project_id
            or row_project_id != project_id
            or row["board_slug"] != board_slug
            or result.get("ok") is not True
        ):
            return None

        operation = str(row["operation"])
        if operation == "owner_task_graph_commit":
            task_ids = result.get("task_ids")
            task_statuses = result.get("task_statuses")
            task_count = result.get("task_count")
            if (
                set(result) != _LEGACY_GRAPH_RESULT_FIELDS
                or result.get("mode") not in {"new", "existing"}
                or result.get("project_slug") != project_slug
                or result.get("board") != board_slug
                or not isinstance(result.get("root_task_id"), str)
                or not result["root_task_id"].strip()
                or row["task_id"] != result["root_task_id"]
                or not isinstance(result.get("root_status"), str)
                or not result["root_status"].strip()
                or not isinstance(task_ids, list)
                or not all(isinstance(task_id, str) and task_id for task_id in task_ids)
                or not isinstance(task_statuses, list)
                or not all(
                    isinstance(status, str) and status for status in task_statuses
                )
                or len(task_statuses) != len(task_ids)
                or isinstance(task_count, bool)
                or not isinstance(task_count, int)
                or task_count != len(task_ids)
            ):
                return None
            graphs.append((str(row["idempotency_key"]), result))
            continue

        created_task_ids = result.get("created_task_ids")
        affected_task_ids = result.get("affected_task_ids")
        change_count = result.get("change_count")
        executable_count = result.get("executable_task_count")
        if (
            set(result) != _LEGACY_PLAN_RESULT_FIELDS
            or result.get("project_slug") != project_slug
            or result.get("board") != board_slug
            or result.get("applied") is not True
            or result.get("risk_level") not in _LEGACY_PLAN_RISK_LEVELS
            or not isinstance(row["task_id"], str)
            or not row["task_id"].strip()
            or isinstance(change_count, bool)
            or not isinstance(change_count, int)
            or change_count < 0
            or isinstance(executable_count, bool)
            or not isinstance(executable_count, int)
            or executable_count < 0
            or not isinstance(created_task_ids, list)
            or not all(
                isinstance(task_id, str) and task_id for task_id in created_task_ids
            )
            or not isinstance(affected_task_ids, list)
            or not all(
                isinstance(task_id, str) and task_id for task_id in affected_task_ids
            )
        ):
            return None
    if len(graphs) != 1:
        return None
    graph_key, graph = graphs[0]
    if graph["mode"] == "new":
        if project_id != "p_" + _derive_id(ctx, graph_key, "graph-project"):
            return None
    else:
        root = kconn.execute(
            "SELECT project_id, task_kind, created_by, idempotency_key, "
            "owner_receipt_bound FROM tasks WHERE id = ?",
            (graph["root_task_id"],),
        ).fetchone()
        expected_root_key = "owgraph_" + _derive_id(ctx, graph_key, "graph-root")
        if (
            root is None
            or root["project_id"] != project_id
            or root["task_kind"] != "work"
            or root["created_by"] != ctx.actor
            or root["idempotency_key"] != expected_root_key
            or root["owner_receipt_bound"] != 1
        ):
            return None
    return "owanchor_" + _derive_id(
        ctx, project_id, _LEGACY_ANCHOR_MIGRATION_SALT,
    )


def _is_legacy_migration_anchor(
    kconn: sqlite3.Connection,
    ctx: OwnerContext,
    task_id: str,
    *,
    project_id: str,
    migration_key: str,
) -> bool:
    anchor = kanban_db.get_control_task(kconn, task_id)
    return (
        anchor is not None
        and anchor.project_id == project_id
        and anchor.idempotency_key == migration_key
        and anchor.created_by == ctx.actor
        and anchor.assignee is None
        and anchor.execution_tier is None
        and anchor.model_policy_lock is None
        and anchor.status == "triage"
    )


def _classify_project_anchor(
    pconn: sqlite3.Connection,
    kconn: sqlite3.Connection,
    ctx: OwnerContext,
    project_id: str,
    *,
    project_slug: str,
    board_slug: str,
) -> _ProjectAnchorResolution:
    """Classify the strict modern or known legacy shape without mutating it."""
    if not _receipt_named_anchors(pconn, ctx, project_id):
        migration_key = _legacy_anchor_migration_key(
            pconn,
            kconn,
            ctx,
            project_id,
            project_slug=project_slug,
            board_slug=board_slug,
        )
        controls = _board_control_rows(kconn, project_id)
        if migration_key is not None and not controls:
            return _ProjectAnchorResolution(None, migration_key)
        if (
            migration_key is not None
            and len(controls) == 1
            and _is_legacy_migration_anchor(
                kconn,
                ctx,
                controls[0],
                project_id=project_id,
                migration_key=migration_key,
            )
        ):
            return _ProjectAnchorResolution(controls[0], migration_key)
    return _ProjectAnchorResolution(
        _receipt_bound_control_anchor(pconn, kconn, ctx, project_id)
    )


def _migrate_legacy_project_anchor(
    kconn: sqlite3.Connection,
    ctx: OwnerContext,
    *,
    project_id: str,
    board_slug: str,
    migration_key: str,
    name: str,
    description: Optional[str],
) -> str:
    anchor_id = kanban_db.create_task(
        kconn,
        title=name,
        body=description,
        created_by=ctx.actor,
        triage=True,
        control=True,
        board=board_slug,
        project_id=project_id,
        idempotency_key=migration_key,
    )
    if (
        _board_control_rows(kconn, project_id) != [anchor_id]
        or not _is_legacy_migration_anchor(
            kconn,
            ctx,
            anchor_id,
            project_id=project_id,
            migration_key=migration_key,
        )
    ):
        raise OwnerWorkspaceError(
            "crash_recovery_failed",
            "the legacy Project anchor does not match its migration identity",
        )
    return anchor_id


def _confirmed_project_anchor(
    pconn: sqlite3.Connection,
    kconn: sqlite3.Connection,
    ctx: OwnerContext,
    project_id: str,
    *,
    project,
    board_slug: str,
    classified: _ProjectAnchorResolution,
) -> str:
    current = _classify_project_anchor(
        pconn,
        kconn,
        ctx,
        project_id,
        project_slug=project.slug,
        board_slug=board_slug,
    )
    if classified.migration_key is None:
        if (
            current.migration_key is not None
            or current.task_id is None
            or current.task_id != classified.task_id
        ):
            raise OwnerWorkspaceError(
                "crash_recovery_failed",
                "this Project's anchor changed after owner confirmation",
            )
        return current.task_id
    if current.migration_key is None:
        if (
            current.task_id is None
            or not _is_legacy_migration_anchor(
                kconn,
                ctx,
                current.task_id,
                project_id=project_id,
                migration_key=classified.migration_key,
            )
        ):
            raise OwnerWorkspaceError(
                "crash_recovery_failed",
                "this Project's anchor changed after owner confirmation",
            )
        return current.task_id
    if current.migration_key != classified.migration_key:
        raise OwnerWorkspaceError(
            "crash_recovery_failed",
            "this Project's anchor migration changed after owner confirmation",
        )
    return _migrate_legacy_project_anchor(
        kconn,
        ctx,
        project_id=project_id,
        board_slug=board_slug,
        migration_key=current.migration_key,
        name=project.name,
        description=project.description,
    )


def _committed_project_plan_result(
    kconn: sqlite3.Connection, *, anchor_task_id: str, digest: str,
    idempotency_key: str, ctx: OwnerContext,
) -> Optional[dict]:
    for event in reversed(
        kanban_db.list_events(kconn, anchor_task_id, include_control=True)
    ):
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
    trigger: str,
    request_title: str,
    summary: str,
    specification: str,
    current_milestone: str,
    owner_visible_result: str,
    later_milestones: Any,
    changes: Any,
) -> dict:
    """Commit one approved Project Steward plan to the existing native board.

    The Project's control anchor is resolved internally from committed receipts.
    A strictly proven pre-anchor Project may create its one hidden anchor only
    after this plan's owner confirmation; callers can never name that row.
    """
    from agent.redact import redact_sensitive_text

    idempotency_key = _bounded_text(idempotency_key, "idempotency_key", limit=200)
    project_id = _bounded_text(project_id, "project_id", limit=100)
    trigger = str(trigger or "").strip()
    if trigger not in _PROJECT_PLAN_TRIGGERS:
        raise OwnerWorkspaceError("invalid_argument", "trigger is invalid")
    operation = "owner_project_plan_commit"
    _require_owner_run_authority(
        ctx,
        operation=operation,
        idempotency_key=idempotency_key,
        payload={
            "idempotency_key": idempotency_key,
            "project_id": project_id,
            "trigger": trigger,
            "request_title": request_title,
            "summary": summary,
            "specification": specification,
            "current_milestone": current_milestone,
            "owner_visible_result": owner_visible_result,
            "later_milestones": later_milestones,
            "changes": changes,
        },
    )
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

    pconn = projects_db.connect()
    try:
        _ensure_schema(pconn)
        if not _receipt_owns_project(pconn, ctx, project_id):
            raise OwnerWorkspaceError(
                "project_not_owned",
                "the Project is not owned by this trusted owner receipt",
            )
        state, row, token = _acquire_or_replay(
            pconn, ctx, idempotency_key, operation, digest,
        )
        if state == "terminal":
            committed = json.loads(row["result_json"])
            _activate_committed_owner_work(committed)
            return committed
        project, board_slug = _resolve_existing_project_board(pconn, project_id)
        kconn = kanban_db.connect(board=board_slug)
        try:
            anchor = _classify_project_anchor(
                pconn,
                kconn,
                ctx,
                project_id,
                project_slug=project.slug,
                board_slug=board_slug,
            )
            anchor_task_id = anchor.task_id
            recovered = (
                _committed_project_plan_result(
                    kconn,
                    anchor_task_id=anchor_task_id,
                    digest=digest,
                    idempotency_key=idempotency_key,
                    ctx=ctx,
                )
                if anchor_task_id is not None
                else None
            )
            if recovered is not None:
                result = {
                    "ok": True,
                    "project_id": project_id,
                    "project_slug": project.slug,
                    "board": board_slug,
                    "anchor_task_id": anchor_task_id,
                    "risk_level": risk_level,
                    **recovered,
                }
                _finalize_receipt(
                    pconn, ctx, idempotency_key, token, status="committed", result=result,
                )
                _activate_committed_owner_work(result)
                return result

            approval = _confirm(
                ctx,
                operation=operation,
                digest=digest,
                description=(
                    f"Apply {len(normalized_changes)} approved Project change(s) "
                    f"to {owner_project_name(project.name)!r}"
                ),
            )
            if not approval.get("approved"):
                result = {"ok": False, "error": "confirmation_denied", "reason": approval.get("reason")}
                _finalize_receipt(
                    pconn, ctx, idempotency_key, token, status="denied", result=result,
                )
                return result

            # Lifecycle changes and owner board writes share this cross-process
            # guard.  Re-read the Project only after the approval wait and
            # keep the guard through receipt finalization, so an approval for
            # stale active state can neither mutate an archived Project nor
            # leave a committed board change without its durable receipt.
            with _global_board_guard(board_slug):
                with write_txn(pconn):
                    _assert_owns_lease(pconn, ctx, idempotency_key, token)
                    if not _receipt_owns_project(pconn, ctx, project_id):
                        raise OwnerWorkspaceError(
                            "project_not_owned",
                            "the Project ownership receipt changed before commit",
                        )
                    current_project = projects_db.get_project(pconn, project_id)
                    if (
                        current_project is None
                        or current_project.archived
                        or current_project.board_slug != board_slug
                    ):
                        applied = {"applied": False}
                    else:
                        _assert_board_ownership(board_slug, project_id)
                        anchor_task_id = _confirmed_project_anchor(
                            pconn,
                            kconn,
                            ctx,
                            project_id,
                            project=current_project,
                            board_slug=board_slug,
                            classified=anchor,
                        )
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

                if not applied["applied"]:
                    result = {
                        "ok": False,
                        "error": "conflict",
                        "project_id": project_id,
                        "project_slug": project.slug,
                        "change_count": 0,
                    }
                    if anchor_task_id is not None:
                        result["anchor_task_id"] = anchor_task_id
                else:
                    result = {
                        "ok": True,
                        "project_id": project_id,
                        "project_slug": project.slug,
                        "board": board_slug,
                        "anchor_task_id": anchor_task_id,
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
            # Outside the ownership guard, and only after the terminal receipt:
            # the plan's new and reactivated work is parked until here, so a
            # failure anywhere above leaves nothing runnable.
            _activate_committed_owner_work(result)
            return result
        finally:
            kconn.close()
    finally:
        pconn.close()



# ---------------------------------------------------------------------------
# owner_task_move
# ---------------------------------------------------------------------------

_UNSAFE_STATUSES = {"running"}


def _resolve_receipt_owned_task(
    pconn: sqlite3.Connection,
    ctx: OwnerContext,
    project_id: str,
    task_id: str,
    *,
    allow_archived: bool = False,
):
    if not _receipt_owns_project(pconn, ctx, project_id):
        raise OwnerWorkspaceError(
            "project_not_owned",
            "the Project is not owned by this trusted owner receipt",
        )
    project, board_slug = _resolve_existing_project_board(
        pconn, project_id, allow_archived=allow_archived,
    )
    kconn = kanban_db.connect(board=board_slug)
    try:
        # The owner kernel also owns the Project's non-executable control
        # anchor, so it resolves that kind here too; no executable path can.
        task = kanban_db.get_task(kconn, task_id, include_control=True)
    finally:
        kconn.close()
    if task is None or task.project_id != project_id:
        raise OwnerWorkspaceError(
            "task_not_found",
            "the task is not part of this receipt-owned Project",
        )
    return project, board_slug, task


def move_task(
    ctx: OwnerContext,
    *,
    idempotency_key: str,
    task_id: str,
    to_status: str,
    expected_status: str,
    expected_revision: Any,
    project_id: str,
) -> dict:
    """Optimistic compare-and-swap task move inside the existing Kanban
    write transaction. See ``kanban_db.cas_transition_task``.

    Durability boundary: a move into ``done``/``archived`` satisfies its
    dependents' dependency the instant the Kanban transaction commits, which
    is BEFORE this receipt's terminal result is durable. Recomputing readiness
    there — or letting a live dispatcher tick land in that gap — would hand a
    newly enabled dependent to a worker for a move that may still fail to
    finalize. So the transition parks exactly the dependents it newly enables
    in its OWN transaction (see ``kanban_db.cas_transition_task``'s
    ``park_newly_enabled_dependents``), records that exact set and each
    dependent's restore column in the ``owner_move`` event and in the terminal
    result, and only once the receipt is committed does
    :func:`_activate_committed_owner_move` release them and recompute
    readiness. Sticky blocked/circuit-breaker holds are preserved: a parked
    dependent goes back to the column it came from, so the readiness guards
    apply exactly as they would have.

    Crash-safe readiness repair: if the CAS status move + event commit
    succeeds but a crash follows before activation (or before
    finalization), a replay adopting the dead claim recognizes — by the
    committed event's payload carrying THIS receipt's full identity (actor,
    profile, idempotency_key, AND the requested to_status/expected_status
    transition) at this task's very next revision after ``expected_revision``
    (see ``kanban_db.get_next_event_after`` — ``task_events.id`` is a
    board-wide sequence, not contiguous per task, so it is not necessarily
    ``expected_revision + 1``) — that it already performed the transition.
    It does not re-run the CAS (which would misread the already-advanced
    status/revision as an unrelated conflict); it rebuilds the same success
    snapshot — including the parked set that event recorded — finalizes the
    original success, and then completes the same idempotent activation.
    Genuinely unrelated status/revision
    drift still reports a conflict — including a DIFFERENT actor/profile
    validly reusing the same idempotency_key text on the same task (receipts
    are scoped per actor/profile/key, but the task's event log is
    board-wide, so a bare idempotency_key match on the next event would
    otherwise let one receipt fabricate a success snapshot for a mutation an
    entirely different receipt performed).

    Lease-fenced: the lease check, replay recognition, CAS, and dependency
    parking all run inside one held ``projects.db`` write lock (see
    :func:`_assert_owns_lease`) so a takeover cannot land between validating
    the lease and committing the move.
    """
    idempotency_key = _require_str(idempotency_key, "idempotency_key")
    project_id = _bounded_text(project_id, "project_id", limit=100)
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

    payload = {
        "project_id": project_id, "task_id": task_id, "to_status": to_status,
        "expected_status": expected_status, "expected_revision": expected_revision,
    }
    digest = _digest(payload)
    operation = "owner_task_move"

    pconn = projects_db.connect()
    try:
        _ensure_schema(pconn)
        replay = _terminal_replay(
            pconn, ctx, idempotency_key, operation, digest,
        )
        if replay is not None:
            # A crash between the terminal receipt and activation leaves this
            # move's newly enabled dependents parked. Replay finishes exactly
            # the same activation from the committed receipt.
            _activate_committed_owner_move(replay)
            return replay
        pending = _get_receipt(pconn, ctx, idempotency_key)
        recovery_pending = bool(
            pending is not None
            and pending["status"] == "in_progress"
            and pending["operation"] == operation
            and pending["request_digest"] == digest
        )
        _project, board_slug, task = _resolve_receipt_owned_task(
            pconn, ctx, project_id, task_id,
            allow_archived=recovery_pending,
        )
        state, row, token = _acquire_or_replay(pconn, ctx, idempotency_key, operation, digest)
        if state == "terminal":
            committed = json.loads(row["result_json"])
            _activate_committed_owner_move(committed)
            return committed

        approval = _confirm(
            ctx, operation=operation, digest=digest,
            description=(
                f"Move {owner_title(task.title)!r} to {to_status!r} "
                f"(from {expected_status!r})"
            ),
        )
        if not approval.get("approved"):
            result = {"ok": False, "error": "confirmation_denied", "reason": approval.get("reason")}
            _finalize_receipt(pconn, ctx, idempotency_key, token, status="denied", result=result)
            return result

        # Deterministic from this receipt's own identity, so a replay that
        # recognizes its own already-committed transition releases exactly the
        # dependents it parked.
        move_park_generation = kanban_db.park_generation(
            actor=ctx.actor, profile=ctx.profile, idempotency_key=idempotency_key,
        )
        kconn = kanban_db.connect(board=board_slug)
        try:
            if kanban_db.get_task(kconn, task_id, include_control=True) is None:
                raise OwnerWorkspaceError("task_not_found", f"no such task {task_id}")

            # Serialize the active-state recheck, board write, and receipt
            # finalization against archive/restore.  The approval may have
            # taken minutes; only state observed under this guard is allowed
            # to authorize the mutation.
            with _global_board_guard(board_slug):
                with write_txn(pconn):
                    _assert_owns_lease(pconn, ctx, idempotency_key, token)
                    if not _receipt_owns_project(pconn, ctx, project_id):
                        raise OwnerWorkspaceError(
                            "project_not_owned",
                            "the Project ownership receipt changed before commit",
                        )
                    current_project = projects_db.get_project(pconn, project_id)
                    current_task = kanban_db.get_task(
                        kconn, task_id, include_control=True
                    )
                    snapshot = None
                    if row is not None and current_task is not None:
                        # Adopting a dead claim: recognize only the exact
                        # event identity this receipt could have emitted.
                        already = kanban_db.get_next_event_after(
                            kconn, task_id, expected_revision,
                        )
                        if (
                            already is not None
                            and already.kind == "owner_move"
                            and (already.payload or {}).get("idempotency_key") == idempotency_key
                            and (already.payload or {}).get("actor") == ctx.actor
                            and (already.payload or {}).get("profile") == ctx.profile
                            and (already.payload or {}).get("to_status") == to_status
                            and (already.payload or {}).get("expected_status") == expected_status
                        ):
                            snapshot = {
                                "moved": True,
                                "status": to_status,
                                "revision": already.id,
                                # The exact set that transition parked, read
                                # back from its own committed event — never
                                # re-derived from a board that has since moved.
                                "parked_dependents": _decode_parked_dependents(
                                    (already.payload or {}).get(
                                        "parked_dependents"
                                    )
                                ),
                            }
                    if (
                        current_project is None
                        or current_project.archived
                        or current_project.board_slug != board_slug
                    ):
                        if snapshot is None:
                            snapshot = {
                                "moved": False,
                                "status": current_task.status if current_task else task.status,
                                "revision": (
                                    kanban_db.task_event_revision(kconn, task_id)
                                    if current_task else expected_revision
                                ),
                            }
                    else:
                        _assert_board_ownership(board_slug, project_id)
                        if current_task is None or current_task.project_id != project_id:
                            raise OwnerWorkspaceError(
                                "task_not_found",
                                "the task is no longer part of this receipt-owned Project",
                            )

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
                                    "to_status": to_status,
                                    "expected_status": expected_status,
                                },
                                # A terminal move releases its dependents in
                                # this very transaction — parked, not
                                # promoted, so nothing is claimable until the
                                # receipt below is durable.
                                park_newly_enabled_dependents=True,
                                park_generation=move_park_generation,
                            )

                if snapshot["moved"]:
                    result = {
                        "ok": True,
                        "task_id": task_id,
                        "status": snapshot["status"],
                        "revision": snapshot["revision"],
                        # Activation authority, driven entirely by the durable
                        # receipt so a replay performs exactly the same one.
                        "board": board_slug,
                        "parked_dependents": _decode_parked_dependents(
                            snapshot.get("parked_dependents")
                        ),
                        "park_generation": move_park_generation,
                    }
                else:
                    result = {
                        "ok": False, "error": "conflict", "task_id": task_id,
                        "current_status": snapshot["status"],
                        "current_revision": snapshot["revision"],
                    }
                _finalize_receipt(
                    pconn, ctx, idempotency_key, token,
                    status="committed", result=result,
                )
                # Everything this move enabled is parked and non-claimable.
                # Only now — with the terminal receipt durable — does it
                # become runnable, in one idempotent recoverable transition a
                # replay can finish on its own.
                _activate_committed_owner_move(result)
                return result
        finally:
            kconn.close()
    finally:
        pconn.close()


def _decode_parked_dependents(raw: Any) -> list[list[str]]:
    """Normalize a durable ``[[task_id, restore_status], ...]`` parked set.

    Malformed or absent data decodes to "nothing was parked", which is the
    safe reading: activation then releases nothing and only recomputes
    readiness, which proves each task's own gating before promoting it.
    """
    return [
        [entry[0], entry[1]]
        for entry in (raw if isinstance(raw, list) else [])
        if (
            isinstance(entry, (list, tuple))
            and len(entry) == 2
            and isinstance(entry[0], str)
            and isinstance(entry[1], str)
        )
    ]


def _activate_committed_owner_move(result: Any) -> None:
    """Make one committed owner move's parked dependents runnable.

    The single activation transition a terminal move performs AFTER its
    receipt is durable, driven entirely by that receipt so a replay after a
    crash performs exactly the same one. Idempotent on both halves:
    :func:`kanban_db.activate_owner_work` only moves rows still sitting in the
    parked column, and the readiness recompute it ends with is safe to repeat.

    A move that enabled nothing still recomputes readiness — the same
    crash-safe repair the pre-parking code performed, now on the safe side of
    the receipt. A non-terminal move, a conflict and a denial all release
    nothing, because none of them can have satisfied a dependency.
    """
    if not isinstance(result, dict) or result.get("ok") is not True:
        return
    if result.get("status") not in ("done", "archived"):
        return
    board_slug = str(result.get("board") or "").strip()
    generation = str(result.get("park_generation") or "").strip()
    if not board_slug:
        return
    restore = {
        task_id: status
        for task_id, status in _decode_parked_dependents(
            result.get("parked_dependents")
        )
    }
    with contextlib.closing(kanban_db.connect(board=board_slug)) as kconn:
        if restore and generation:
            kanban_db.activate_owner_work(
                kconn,
                list(restore),
                generation=generation,
                restore_statuses=restore,
            )
        else:
            kanban_db.recompute_ready(kconn)


# ---------------------------------------------------------------------------
# owner_task_comment
# ---------------------------------------------------------------------------


def comment_task(
    ctx: OwnerContext,
    *,
    idempotency_key: str,
    project_id: str,
    task_id: str,
    body: str,
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
    project_id = _bounded_text(project_id, "project_id", limit=100)
    task_id = _require_str(task_id, "task_id")
    from agent.redact import redact_sensitive_text

    body = redact_sensitive_text(
        _bounded_text(body, "body", limit=12_000), force=True,
    )

    payload = {"project_id": project_id, "task_id": task_id, "body": body}
    digest = _digest(payload)
    operation = "owner_task_comment"

    pconn = projects_db.connect()
    try:
        _ensure_schema(pconn)
        replay = _terminal_replay(
            pconn, ctx, idempotency_key, operation, digest,
        )
        if replay is not None:
            return replay
        pending = _get_receipt(pconn, ctx, idempotency_key)
        recovery_pending = bool(
            pending is not None
            and pending["status"] == "in_progress"
            and pending["operation"] == operation
            and pending["request_digest"] == digest
        )
        _project, board_slug, task = _resolve_receipt_owned_task(
            pconn, ctx, project_id, task_id,
            allow_archived=recovery_pending,
        )
        state, row, token = _acquire_or_replay(pconn, ctx, idempotency_key, operation, digest)
        if state == "terminal":
            return json.loads(row["result_json"])

        approval = _confirm(
            ctx, operation=operation, digest=digest,
            description=f"Comment on {owner_title(task.title)!r}",
        )
        if not approval.get("approved"):
            result = {"ok": False, "error": "confirmation_denied", "reason": approval.get("reason")}
            _finalize_receipt(pconn, ctx, idempotency_key, token, status="denied", result=result)
            return result

        operation_key = f"{ctx.actor}:{ctx.profile}:{idempotency_key}"
        kconn = kanban_db.connect(board=board_slug)
        try:
            with _global_board_guard(board_slug):
                with write_txn(pconn):
                    _assert_owns_lease(pconn, ctx, idempotency_key, token)
                    if not _receipt_owns_project(pconn, ctx, project_id):
                        raise OwnerWorkspaceError(
                            "project_not_owned",
                            "the Project ownership receipt changed before commit",
                        )
                    current_project = projects_db.get_project(pconn, project_id)
                    current_task = kanban_db.get_task(
                        kconn, task_id, include_control=True
                    )
                    if (
                        current_project is None
                        or current_project.archived
                        or current_project.board_slug != board_slug
                    ):
                        existing_comment = (
                            kconn.execute(
                                "SELECT id, author, body FROM task_comments "
                                "WHERE task_id = ? AND operation_key = ?",
                                (task_id, operation_key),
                            ).fetchone()
                            if row is not None and current_task is not None
                            else None
                        )
                        if existing_comment is not None and (
                            existing_comment["author"] != ctx.actor
                            or existing_comment["body"] != body
                        ):
                            raise OwnerWorkspaceError(
                                "crash_recovery_failed",
                                "the committed comment does not match this receipt",
                            )
                        comment_id = (
                            int(existing_comment["id"])
                            if existing_comment is not None else None
                        )
                        revision = (
                            kanban_db.task_event_revision(kconn, task_id)
                            if current_task else 0
                        )
                    else:
                        _assert_board_ownership(board_slug, project_id)
                        if current_task is None or current_task.project_id != project_id:
                            raise OwnerWorkspaceError(
                                "task_not_found",
                                "the task is no longer part of this receipt-owned Project",
                            )
                        # Author comes only from the trusted context.
                        comment_id = kanban_db.add_comment(
                            kconn,
                            task_id,
                            author=ctx.actor,
                            body=body,
                            operation_key=operation_key,
                            include_control=True,
                        )
                        task = kanban_db.get_task(
                            kconn, task_id, include_control=True
                        )
                        revision = kanban_db.task_event_revision(kconn, task_id)

                if comment_id is None:
                    result = {
                        "ok": False,
                        "error": "conflict",
                        "task_id": task_id,
                        "current_revision": revision,
                    }
                else:
                    result = {
                        "ok": True, "task_id": task_id, "comment_id": comment_id,
                        "status": task.status, "revision": revision,
                    }
                _finalize_receipt(
                    pconn, ctx, idempotency_key, token,
                    status="committed", result=result,
                )
                return result
        finally:
            kconn.close()
    finally:
        pconn.close()
