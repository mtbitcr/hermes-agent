"""SQLite-backed Kanban board for multi-profile, multi-project collaboration.

In a fresh install the board lives at ``<root>/kanban.db`` where
``<root>`` is the **shared Hermes root** (the parent of any active
profile). Profiles intentionally collapse onto a shared board: it IS
the cross-profile coordination primitive. A worker spawned with
``hermes -p <profile>`` joins the same board as the dispatcher that
claimed the task. The same applies to ``<root>/kanban/workspaces/`` and
``<root>/kanban/logs/``.

**Multiple boards (projects):** users can create additional boards to
separate unrelated streams of work (e.g. one per project / repo / domain).
Each board is a directory under ``<root>/kanban/boards/<slug>/`` with
its own ``kanban.db``, ``workspaces/``, and ``logs/``. All boards share
the profile's Hermes home but are otherwise isolated: a worker spawned
for a task on board ``atm10-server`` sees only that board's tasks,
cannot enumerate other boards, and its dispatcher ticks don't touch
other boards' DBs.

The first (and for single-project users, only) board is ``default``.
For back-compat its on-disk DB is ``<root>/kanban.db`` (not
``boards/default/kanban.db``), so installs that predate the boards
feature keep working with zero migration. See :func:`kanban_db_path`.

Board resolution order (highest precedence first, all optional):

* ``board=`` argument passed directly to :func:`connect` / :func:`init_db`
  (explicit — used by the CLI ``--board`` flag and the dashboard
  ``?board=...`` query param).
* ``HERMES_KANBAN_BOARD`` env var (used by the dispatcher to pin workers
  to the board their task lives on — workers cannot see other boards).
* ``HERMES_KANBAN_DB`` env var (pins the DB file path directly — legacy
  override still honoured; highest precedence when the file path itself
  is what the caller wants to force).
* ``<root>/kanban/current`` — a one-line text file holding the slug of
  the "currently selected" board. Written by ``hermes kanban boards
  switch <slug>``. When absent, the active board is ``default``.

In standard installs ``<root>`` is ``~/.hermes``. In Docker / custom
deployments where ``HERMES_HOME`` points outside ``~/.hermes`` (e.g.
``/opt/hermes``), ``<root>`` is ``HERMES_HOME``. Legacy env-var
overrides still work:

* ``HERMES_KANBAN_DB`` — pin the database file path directly.
* ``HERMES_KANBAN_WORKSPACES_ROOT`` — pin the workspaces root directly.
* ``HERMES_KANBAN_HOME`` — pin the umbrella root that anchors kanban
  paths. Useful for tests and unusual deployments.

The dispatcher injects ``HERMES_KANBAN_DB``,
``HERMES_KANBAN_WORKSPACES_ROOT``, and ``HERMES_KANBAN_BOARD`` into
worker subprocess env so workers converge on the exact DB the
dispatcher used to claim their task — even under unusual symlink or
Docker layouts.

Schema is intentionally small: tasks, task_links, task_comments,
task_events.  The ``workspace_kind`` field decouples coordination from git
worktrees so that research / ops / digital-twin workloads work alongside
coding workloads.  See ``docs/hermes-kanban-v1-spec.pdf`` for the full
design specification.

Concurrency strategy: WAL mode + ``BEGIN IMMEDIATE`` for write
transactions + compare-and-swap (CAS) updates on ``tasks.status`` and
``tasks.claim_lock``.  SQLite serializes writers via its WAL lock, so at
most one claimer can win any given task.  Losers observe zero affected
rows and move on -- no retry loops, no distributed-lock machinery.
The CAS coordination is **per-board** — each board is a separate DB,
so multi-board installs get the same atomicity guarantees without any
new locking.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import re
import random
import secrets
import shutil
import sqlite3
import subprocess
import sys
import threading
import logging
import time
import unicodedata
from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Optional

from hermes_cli.sqlite_util import (
    add_column_if_missing as _add_column_if_missing,
    cross_process_init_lock as _shared_cross_process_init_lock,
)
from toolsets import get_toolset_names

_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VALID_STATUSES = {"triage", "todo", "scheduled", "ready", "running", "blocked", "review", "done", "archived"}
VALID_INITIAL_STATUSES = {"running", "blocked"}

# Typed block reasons. Distinguishes the two fundamentally different things a
# worker (or human) means by "blocked", so each can be routed differently
# instead of all landing in one undifferentiated ``blocked`` bucket that a cron
# unblocks → worker re-blocks → cron unblocks … forever.
#
#   * ``dependency``   — can't proceed until another task finishes. Routed to
#                        ``todo`` (NOT ``blocked``) so the existing
#                        parent-gating / ``recompute_ready`` machinery promotes
#                        it automatically once parents are done. No human, no
#                        cron, no retry storm.
#   * ``needs_input``  — needs a human decision/answer it cannot derive.
#   * ``capability``   — hit a hard wall (no access, missing creds, an action no
#                        AI agent can perform). Genuinely human-only.
#   * ``transient``    — a flaky/temporary failure that may clear on retry.
#
# ``needs_input`` and ``capability`` are "truly blocked": they go to ``blocked``
# for a human, and the unblock-loop breaker (see ``block_task`` /
# ``BLOCK_RECURRENCE_LIMIT``) escalates them to ``triage`` if a cron keeps
# unblocking them only to have the worker re-block for the same reason.
# ``None`` = legacy/un-typed block (treated as a generic human blocker).
VALID_BLOCK_KINDS = {"dependency", "needs_input", "capability", "transient"}

# After a task has been blocked, unblocked, and re-blocked this many times for
# the same (truly-blocked) reason, the unblock-loop breaker stops trusting the
# unblocker (usually a cron) and routes the task to ``triage`` instead of back
# to ``blocked`` — breaking the infinite unblock↔re-block loop and forcing a
# human-in-the-loop decision. Mirrors the dispatcher's ``DEFAULT_FAILURE_LIMIT``
# spirit (default 2) but counts a different signal: manual unblock recurrences,
# not dispatcher spawn/crash/timeout failures.
BLOCK_RECURRENCE_LIMIT = 2
VALID_WORKSPACE_KINDS = {"scratch", "worktree", "dir"}

# The six owner-reviewed (never auto-applied) advice kinds
# ``create_recommendation`` supports.
VALID_RECOMMENDATION_KINDS = {
    "skill",
    "permission",
    "connection",
    "pipeline",
    "provider_model_policy",
    "profile_setting",
}
VALID_RECOMMENDATION_DECISIONS = {"pending", "deferred", "rejected", "accepted"}
VALID_RECOMMENDATION_AUTHORITIES = {"preauthorized_non_widening", "owner_approved"}
VALID_RECOMMENDATION_EFFECTIVE_STATES = {
    "none",
    "staged",
    "canary_running",
    "verified",
    "promoted",
    "rolled_back",
    "revoked",
}
RECOMMENDATION_SCOPE_FLAGS = (
    "credential_access",
    "connector_access",
    "data_access",
    "network_access",
    "external_write",
    "paid_route",
    "production_effect",
    "permission_widening",
)

# Bounds on the opaque recommendation display fields shown in the owner review UI.
_RECOMMENDATION_SUBJECT_ID_MAX_LEN = 200
_RECOMMENDATION_LABEL_MAX_LEN = 200
_RECOMMENDATION_RATIONALE_MAX_LEN = 4000
_RECOMMENDATION_PROVENANCE_AUTHORITY_MAX_LEN = 200
_RECOMMENDATION_PROVENANCE_REF_MAX_LEN = 500
_RECOMMENDATION_EVIDENCE_TEXT_MAX_LEN = 2000
_RECOMMENDATION_EVIDENCE_MAX_BYTES = 12000
_RECOMMENDATION_LIFECYCLE_TEXT_MAX_LEN = 1000
_RECOMMENDATION_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_.:-]{0,199}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

# Fixed non-semantic title; recommendation cards are read via recommendation_* columns, never title/body.
_RECOMMENDATION_TASK_TITLE = "hermes recommendation"

# ITEM31BH: version tag for the recommendation dedup identity stored in
# ``tasks.idempotency_key``. Bump only if the identity tuple changes meaning —
# a bump intentionally re-opens one new card per previously-deduped identity.
_RECOMMENDATION_IDENTITY_VERSION = "rec1"


def normalize_reasoning_effort(effort: Optional[str]) -> Optional[str]:
    """Normalize a per-task reasoning effort into a storable level.

    Accepts any level in ``hermes_constants.VALID_REASONING_EFFORTS`` plus
    ``"none"`` (thinking disabled), case-insensitively. Empty / None means
    "inherit the worker profile's own ``agent.reasoning_effort``" and stores
    NULL. Anything else is rejected rather than silently dropped — a typo'd
    level must not quietly hand the task back to the profile default.
    """
    from hermes_constants import VALID_REASONING_EFFORTS

    value = str(effort or "").strip().lower()
    if not value:
        return None
    if value == "none" or value in VALID_REASONING_EFFORTS:
        return value
    allowed = ", ".join(("none", *VALID_REASONING_EFFORTS))
    raise ValueError(
        f"reasoning_effort must be one of {allowed}, got {effort!r}"
    )


# Durable owner-approved model-route lock. ``tasks.model_policy_lock`` holds a
# versioned, digest-bound authority string minted by the canonical model
# policy; the digest covers the task's assignee, provider, model, effort and
# execution tier together, so no single one of those columns can be edited
# without the lock ceasing to validate. A locked task is immutable: the route
# mutators refuse it, role transitions must re-derive a separately approved
# route, and the dispatcher runs it with fallbacks off. NULL (every
# pre-existing, manual, and CLI task) keeps the historical mutable behaviour;
# a NON-NULL value that does not validate is never equivalent to NULL — it
# fails closed everywhere.


def _model_policy():
    """The single canonical model-policy owner (imported lazily, cached)."""
    from plugins.dashboard_auth.raphael_workspace import model_policy

    return model_policy


def mint_policy_lock(
    assignee: Optional[str],
    provider: Optional[str],
    model: Optional[str],
    effort: Optional[str],
    execution_tier: Optional[str],
) -> str:
    """Mint the durable lock for one exact admitted route (raises otherwise)."""
    return _model_policy().mint_policy_lock(
        assignee, provider, model, effort, execution_tier
    )


def policy_lock_error(
    lock: Optional[str],
    assignee: Optional[str],
    provider: Optional[str],
    model: Optional[str],
    effort: Optional[str],
    execution_tier: Optional[str],
) -> Optional[str]:
    """Return why ``lock`` does not authorize this exact route, else None.

    Checked before a pin is persisted, again on every route/role mutation, and
    again at dispatch, so a hand-edited, partially migrated, or stale row can
    never run.
    """
    return _model_policy().policy_lock_error(
        lock, assignee, provider, model, effort, execution_tier
    )


def _task_kinds(include_control: bool) -> str:
    """SQL tuple of the task kinds a reader is willing to resolve.

    ``('work')`` — the default everywhere — is what keeps a non-executable
    control anchor invisible to every executable path. Only the owner-workspace
    kernel, which owns anchors, widens it.
    """
    return "('work', 'control')" if include_control else "('work')"


# Which kinds count when asking whether a task's PARENTS are satisfied. A
# control anchor is never executable, but it is a real dependency gate: an
# owner-approved plan hangs new work under its Project's anchor, and that work
# must stay parked until the owner moves the anchor to done/archived. Excluding
# the anchor here would promote its children the moment they are created.
# Recommendation rows are still never dependencies.
_DEPENDENCY_PARENT_KINDS = "('work', 'control')"


def assert_claimable_route(conn: sqlite3.Connection, task_id: str) -> None:
    """Refuse to start a run on a task whose route authority does not hold.

    Called from every claim path, so a row that carries a lock this build
    cannot validate against its own assignee/provider/model/effort/tier — a
    hand-edited column, a partially migrated schema, a foreign or stale
    authority — never starts a run, whichever surface pulls it. An unlocked
    ordinary/manual/CLI task is unaffected: those legitimately carry no lock.

    A row that is owner-GOVERNED (it carries an execution tier or a lock) but
    unlocked is refused too. The readiness guard
    (:func:`authorize_executable_transition`) mints or parks such a row before
    it can reach an executable column; this is the backstop for a row that got
    there under an older build, so no path at all can start a run on
    receipt-owned work whose route nobody approved.
    """
    row = conn.execute(
        "SELECT assignee, provider_override, model_override, reasoning_effort, "
        "execution_tier, model_policy_lock, owner_receipt_bound FROM tasks "
        "WHERE id = ? AND task_kind = 'work'",
        (task_id,),
    ).fetchone()
    if row is None:
        return
    error = task_policy_lock_error(row)
    if not error and task_is_policy_governed(row) and not row["model_policy_lock"]:
        error = (
            "owner-governed task carries no approved route lock; it must be "
            "approved again before it can run"
        )
    if error:
        raise RuntimeError(f"task {task_id}: {error}")


def task_policy_lock_error(row: Any) -> Optional[str]:
    """Return why a locked task row is invalid, else None (incl. unlocked).

    A row projected without the lock column cannot carry a lock and is
    genuinely unlocked. A row that HAS a lock but is missing any column the
    digest binds is unprovable, so it fails closed.
    """
    try:
        lock = row["model_policy_lock"]
    except (IndexError, KeyError, TypeError):
        return None
    if not lock:
        return None
    try:
        bound = tuple(
            row[column]
            for column in (
                "assignee",
                "provider_override",
                "model_override",
                "reasoning_effort",
                "execution_tier",
            )
        )
    except (IndexError, KeyError, TypeError):
        return "policy lock cannot be checked against an incomplete task row"
    return policy_lock_error(lock, *bound)


KNOWN_TOOLSET_NAMES = frozenset(name.casefold() for name in get_toolset_names())
_IS_WINDOWS = sys.platform == "win32"
KANBAN_ATTACHMENT_MAX_BYTES = 25 * 1024 * 1024


def _assert_not_delegated_child_mutation() -> None:
    """Reject Kanban state mutations from ``delegate_task`` child contexts.

    The structured kanban tools and CLI dispatch layer both have fast-fail
    guards for better UX, but neither is a trust boundary: a delegated child can
    still shell out to the CLI or import this module directly. The actual
    invariant belongs at the DB/filesystem mutation layer so every public
    mutator that uses ``write_txn`` (tasks, runs, comments, attachments,
    dispatcher claims, repair events, subscriptions, GC, etc.) and every board
    metadata mutator fails closed before touching durable state.
    """
    try:
        from agent.delegation_context import is_delegated_child_process_context

        delegated = is_delegated_child_process_context()
    except Exception:
        delegated = bool(os.environ.get("HERMES_DELEGATED_CHILD_CONTEXT"))
    if delegated:
        raise PermissionError(
            "delegate_task child contexts cannot mutate Kanban tasks or boards"
        )


def _fire_kanban_lifecycle_hook(event: str, task_id: str, **fields: Any) -> None:
    """Fire a kanban lifecycle plugin hook, fully best-effort.

    Called by the claim/complete/block transitions AFTER their write txn has
    committed, so plugin code never runs while a SQLite write lock is held and
    always observes durable board state. Any failure (plugins unavailable,
    a plugin raising, import error) is swallowed — a misbehaving observer must
    never break a board state transition.

    ``profile_name`` is resolved from the active HERMES_HOME so dispatcher- and
    worker-side hooks both carry the right profile without the caller plumbing
    it through.
    """
    try:
        from hermes_cli.lifecycle import invoke_hook
        from hermes_cli.profiles import get_active_profile_name
        try:
            profile_name = get_active_profile_name()
        except Exception:
            profile_name = "default"
        invoke_hook(event, task_id=task_id, profile_name=profile_name, **fields)
    except Exception as exc:  # pragma: no cover - defensive
        _log.debug("kanban lifecycle hook %s failed: %s", event, exc)


def _kanban_observer_consumed(event: str) -> bool:
    """Return whether any first-party observer or plugin consumes *event*.

    Hot-path short-circuit for the worker-lifecycle / task-mutation /
    dispatch-tick observers (RFC #58548): those fire on every dispatcher
    tick and every task write, so call sites skip payload assembly entirely
    when nothing subscribes. Best-effort — if inspection fails the event is
    treated as unconsumed (the invoke path would fail the same way, and
    these are observers, so dropping is always safe).
    """
    try:
        from hermes_cli.lifecycle import has_hook

        return has_hook(event)
    except Exception:  # pragma: no cover - defensive
        return False


def _fire_worker_spawned_hook(
    conn: sqlite3.Connection,
    task: "Task",
    workspace_path: str,
    pid: Optional[int],
    *,
    board: Optional[str] = None,
) -> None:
    """Fire ``on_kanban_worker_spawned`` for one dispatched spawn.

    Called by the dispatch loop AFTER ``spawn_fn`` returned and the worker
    PID (when one was reported) has been durably persisted — the RFC #58548
    timing contract. Fully best-effort: any failure is swallowed so a
    misbehaving observer can never break the dispatch loop.
    """
    if not _kanban_observer_consumed("on_kanban_worker_spawned"):
        return
    try:
        _fire_kanban_lifecycle_hook(
            "on_kanban_worker_spawned",
            task.id,
            board=board or get_current_board(),
            assignee=task.assignee,
            run_id=_current_run_id(conn, task.id),
            worker_pid=int(pid) if pid else None,
            workspace_path=str(workspace_path),
        )
    except Exception as exc:  # pragma: no cover - defensive
        _log.debug("kanban worker spawned hook failed: %s", exc)


def notify_task_updated(
    conn: sqlite3.Connection,
    task_id: str,
    changed_fields: Iterable[str],
    *,
    board: Optional[str] = None,
) -> None:
    """Fire ``on_kanban_task_updated`` for a committed task-row mutation.

    Task-mutation boundary primitive from RFC #58548: a surface that mutates
    a task row outside the claim/complete/block lifecycle calls this AFTER
    its write txn has committed — including surfaces that write with direct
    SQL and bypass every ``kanban_db`` mutator (the dashboard plugin API's
    priority/title/body editors). ``changed_fields`` carries field NAMES
    only, never values. Observer-only and fully best-effort: it can never
    fail a task mutation, and it costs one ``has_hook`` probe when nothing
    subscribes.
    """
    if not _kanban_observer_consumed("on_kanban_task_updated"):
        return
    try:
        row = conn.execute(
            "SELECT assignee, current_run_id FROM tasks "
            "WHERE id = ? AND task_kind = 'work'",
            (task_id,),
        ).fetchone()
        _fire_kanban_lifecycle_hook(
            "on_kanban_task_updated",
            task_id,
            board=board or get_current_board(),
            assignee=row["assignee"] if row else None,
            run_id=row["current_run_id"] if row else None,
            changed_fields=list(changed_fields),
        )
    except Exception as exc:  # pragma: no cover - defensive
        _log.debug("kanban task updated hook failed: %s", exc)


def _fire_dispatch_tick_hook(
    result: "DispatchResult",
    *,
    board: Optional[str] = None,
    dry_run: bool = False,
) -> None:
    """Fire ``on_kanban_dispatch_tick`` after one dispatcher tick.

    Re-port of PR #56066 per the #64231 batch disposition: renamed to the
    taxonomy form and called by ``dispatch_once`` strictly AFTER
    ``_dispatch_tick_lock`` has been released — the original fired inside
    the lock, so a slow subscriber could extend the single-writer critical
    section and stall a sibling dispatcher's tick. Observer-only and fully
    best-effort: any subscriber failure is swallowed.
    """
    if not _kanban_observer_consumed("on_kanban_dispatch_tick"):
        return
    try:
        from hermes_cli.lifecycle import invoke_hook
        from hermes_cli.profiles import get_active_profile_name

        try:
            profile_name = get_active_profile_name()
        except Exception:
            profile_name = "default"
        if board is None:
            try:
                board = get_current_board()
            except Exception:
                board = None
        outcome = "ok"
        if result.skipped_locked:
            outcome = "skipped_locked"
        elif result.skipped_inactive:
            outcome = "skipped_inactive"
        elif not any((
            result.spawned,
            result.reclaimed,
            result.promoted,
            result.reconciled_orphans,
            result.crashed,
            result.stale,
            result.timed_out,
            result.auto_blocked,
            result.rate_limited,
            result.auto_assigned_default,
            result.respawn_guarded,
            result.skipped_per_profile_capped,
            result.skipped_file_scope_conflict,
            result.skipped_unassigned,
            result.skipped_nonspawnable,
        )):
            outcome = "idle"
        invoke_hook(
            "on_kanban_dispatch_tick",
            board=board,
            profile_name=profile_name,
            dry_run=bool(dry_run),
            outcome=outcome,
            result=result,
        )
    except Exception as exc:  # pragma: no cover - defensive
        _log.debug("kanban dispatch tick hook failed: %s", exc)


# A running task's claim is valid for 15 minutes by default; after that the
# next dispatcher tick reclaims it. Workers that outlive this window should
# call ``heartbeat_claim(task_id)`` periodically. In practice most kanban
# workloads either finish within 15m, set a longer claim explicitly, or use
# ``HERMES_KANBAN_CLAIM_TTL_SECONDS`` to raise the default claim window for
# long single-call MCP workflows.
DEFAULT_CLAIM_TTL_SECONDS = 15 * 60

# If a worker's PID is still alive but its ``last_heartbeat_at`` is
# older than this when ``release_stale_claims`` runs, treat the worker
# as wedged and reclaim regardless of PID liveness (#29747 gap 3).
# This catches the logic-loop case where the process is technically
# running but not making observable progress.  ``_touch_activity``
# bridges chunk-level liveness into ``last_heartbeat_at`` via #31752,
# so any genuinely active worker keeps its heartbeat fresh as a side
# effect of normal API traffic.
DEFAULT_CLAIM_HEARTBEAT_MAX_STALE_SECONDS = 60 * 60

# Grace added to a claim when a reclaim is deferred because the previous
# host-local worker is still alive after a termination attempt. Releasing the
# claim in that state would spawn a duplicate alongside the surviving worker —
# the runaway seen when a cgroup memory.high throttle parks a worker in
# uninterruptible (D) state, where a pending SIGKILL cannot be delivered until
# the throttle lifts. Holding the claim a short grace and retrying next tick
# stops the duplication; once no duplicate is spawned the pressure eases, the
# signal lands, and the following tick reclaims cleanly.
RECLAIM_DEFER_GRACE_SECONDS = 120


def _resolve_claim_ttl_seconds(ttl_seconds: Optional[int] = None) -> int:
    """Return the effective claim TTL, honoring the kanban env override.

    Explicit call-site values win. Otherwise a positive integer from
    ``HERMES_KANBAN_CLAIM_TTL_SECONDS`` overrides the built-in default.
    Invalid or non-positive env values fall back silently so existing
    installs keep working.
    """
    if ttl_seconds is not None:
        return max(1, int(ttl_seconds))

    raw = os.environ.get("HERMES_KANBAN_CLAIM_TTL_SECONDS", "").strip()
    if raw:
        try:
            parsed = int(raw)
        except ValueError:
            parsed = 0
        if parsed > 0:
            return parsed

    return DEFAULT_CLAIM_TTL_SECONDS


# Grace period after a task transitions to ``running`` during which
# ``detect_crashed_workers`` skips the ``_pid_alive`` check. Covers the
# fork() → /proc-visibility window where liveness can transiently report
# False for a freshly-spawned worker. The 15-minute claim TTL still
# catches genuinely-crashed workers; this only suppresses false positives
# during the launch window.
DEFAULT_CRASH_GRACE_SECONDS = 30


# Sentinel exit code a kanban worker uses to signal "I bailed because the
# provider rate-limited / exhausted quota, not because the task failed."
# The dispatcher's reap classifier maps this to a ``rate_limited`` exit kind
# so ``detect_crashed_workers`` can release the task back to ``ready``
# WITHOUT counting a failure (the circuit breaker must never trip on a
# transient throttle). 75 == BSD ``EX_TEMPFAIL`` (sysexits.h) — the
# conventional "temporary failure, retry later" code, and well clear of the
# 0/1/2 codes the worker uses for success / generic failure / usage error.
KANBAN_RATE_LIMIT_EXIT_CODE = 75


def _resolve_crash_grace_seconds() -> int:
    """Return the crash-detection grace period in seconds.

    Reads ``HERMES_KANBAN_CRASH_GRACE_SECONDS`` from the environment;
    falls back to ``DEFAULT_CRASH_GRACE_SECONDS`` when absent, empty,
    non-integer, or negative. A value of 0 restores immediate-reclaim
    behaviour (useful for tests).
    """
    raw = os.environ.get("HERMES_KANBAN_CRASH_GRACE_SECONDS", "").strip()
    if raw:
        try:
            parsed = int(raw)
        except ValueError:
            parsed = -1
        if parsed >= 0:
            return parsed
    return DEFAULT_CRASH_GRACE_SECONDS


def _resolve_rate_limit_cooldown_seconds() -> int:
    """Return the rate-limit requeue cooldown in seconds.

    Reads ``HERMES_KANBAN_RATE_LIMIT_COOLDOWN_SECONDS`` from the environment;
    falls back to ``DEFAULT_RATE_LIMIT_COOLDOWN_SECONDS`` when absent, empty,
    non-integer, or negative. A value of 0 disables the cooldown (re-spawn on
    the next tick) — useful for tests that want to assert the task becomes
    spawnable again immediately.
    """
    raw = os.environ.get(
        "HERMES_KANBAN_RATE_LIMIT_COOLDOWN_SECONDS", ""
    ).strip()
    if raw:
        try:
            parsed = int(raw)
        except ValueError:
            parsed = -1
        if parsed >= 0:
            return parsed
    return DEFAULT_RATE_LIMIT_COOLDOWN_SECONDS


# Worker-context caps so build_worker_context() stays bounded on
# pathological boards (retry-heavy tasks, comment storms, giant
# summaries). Values chosen to fit a typical 100k-char LLM prompt with
# plenty of headroom. Each constant is tuned independently so users
# who need to relax one don't have to relax all of them.
_CTX_MAX_PRIOR_ATTEMPTS = 10      # most recent N prior runs shown in full
_CTX_MAX_COMMENTS       = 30      # most recent N comments shown in full
_CTX_MAX_FIELD_BYTES    = 4 * 1024   # 4 KB per summary/error/metadata/result
_CTX_MAX_BODY_BYTES     = 8 * 1024   # 8 KB per task.body (opening post)
_CTX_MAX_COMMENT_BYTES  = 2 * 1024   # 2 KB per comment


def _relative_age(ts: Optional[int], now: Optional[int] = None) -> str:
    """Render the age of an epoch-seconds timestamp as a coarse, human-
    readable string like ``just now``, ``18h ago``, ``3d ago``.

    Workers read parent handoffs, comments, and prior-attempt summaries as
    if they describe *current* state. A bare absolute timestamp
    (``2026-06-25 14:30``) doesn't make an LLM reason about staleness — it
    reads the content as fact regardless of how old it is. A relative age
    ("18h ago") is the signal that prompts the worker to re-verify against
    the live source before acting on stale sibling work. Returns an empty
    string for missing/invalid timestamps so callers can append
    unconditionally.
    """
    if ts is None:
        return ""
    try:
        ts = int(ts)
    except (TypeError, ValueError):
        return ""
    if now is None:
        now = int(time.time())
    delta = now - ts
    if delta < 0:
        # Clock skew across machines/profiles — don't claim "in the future".
        return "just now"
    if delta < 60:
        return "just now"
    if delta < 3600:
        m = delta // 60
        return f"{m}m ago"
    if delta < 86400:
        h = delta // 3600
        return f"{h}h ago"
    d = delta // 86400
    return f"{d}d ago"


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

DEFAULT_BOARD = "default"
_CURRENT_BOARD_OVERRIDE: ContextVar[str | None] = ContextVar(
    "hermes_kanban_current_board_override",
    default=None,
)


@contextlib.contextmanager
def scoped_current_board(slug: str):
    """Temporarily pin the active board for the current context only."""
    token: Token[str | None] = _CURRENT_BOARD_OVERRIDE.set(slug)
    try:
        yield
    finally:
        _CURRENT_BOARD_OVERRIDE.reset(token)

# Slug validator: lowercase alphanumerics, digits, hyphens; 1–64 chars.
# Strict enough to stop traversal (`..`) and embedded path separators, loose
# enough that kebab-case names like ``atm10-server`` or ``hermes-agent``
# pass without fuss. Board names with display formatting (spaces, emoji)
# live in ``board.json``; the slug is just the directory name.
_BOARD_SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9\-_]{0,63}$")


def _normalize_board_slug(slug: Optional[str]) -> Optional[str]:
    """Lowercase + strip a slug; validate; return ``None`` for empty."""
    if slug is None:
        return None
    s = str(slug).strip().lower()
    if not s:
        return None
    if not _BOARD_SLUG_RE.match(s):
        raise ValueError(
            f"invalid board slug {slug!r}: must be 1-64 chars, lowercase "
            f"alphanumerics / hyphens / underscores, not starting with '-' or '_'"
        )
    return s


def kanban_home() -> Path:
    """Return the shared Hermes root that anchors the kanban board.

    Resolution order:

    1. ``HERMES_KANBAN_HOME`` env var when set and non-empty (explicit
       override for tests and unusual deployments).
    2. ``get_default_hermes_root()``, which already returns ``<root>``
       when ``HERMES_HOME`` is ``<root>/profiles/<name>``, and returns
       ``HERMES_HOME`` directly for Docker / custom deployments.

    The kanban board is shared across profiles **by design** (see the
    module docstring). Resolving the kanban paths through the active
    profile's ``HERMES_HOME`` would silently fork the board per profile,
    which breaks the dispatcher / worker handoff.
    """
    override = os.environ.get("HERMES_KANBAN_HOME", "").strip()
    if override:
        return Path(override).expanduser()
    from hermes_constants import get_default_hermes_root
    return get_default_hermes_root()


def boards_root() -> Path:
    """Return ``<root>/kanban/boards`` — the parent of non-default board dirs.

    ``default`` is intentionally NOT under this directory — its DB lives at
    ``<root>/kanban.db`` for back-compat with pre-boards installs. This
    function returns the directory where *additional* named boards live,
    used by :func:`list_boards` to enumerate them.
    """
    return kanban_home() / "kanban" / "boards"


def current_board_path() -> Path:
    """Return the path to ``<root>/kanban/current``.

    One-line text file written by ``hermes kanban boards switch <slug>``
    to persist the user's board selection across CLI invocations. Absent
    by default (meaning: active board is ``default``).
    """
    return kanban_home() / "kanban" / "current"


def get_current_board() -> str:
    """Return the active board slug, honouring the resolution chain.

    Order (highest precedence first):

    1. ``HERMES_KANBAN_BOARD`` env var (set by the dispatcher on worker
       spawn, or manually for ad-hoc overrides).
    2. ``<root>/kanban/current`` on disk (set by ``hermes kanban boards
       switch``), but only when that board still exists.
    3. ``DEFAULT_BOARD`` (``"default"``).

    A malformed or stale slug at any step falls through to the next layer
    with a best-effort warning — the dispatcher must never crash because a
    user hand-edited a file or removed a board directory.
    """
    scoped = (_CURRENT_BOARD_OVERRIDE.get() or "").strip()
    if scoped:
        try:
            normed = _normalize_board_slug(scoped)
            if normed and board_exists(normed):
                return normed
        except ValueError:
            pass

    env = os.environ.get("HERMES_KANBAN_BOARD", "").strip()
    if env:
        try:
            normed = _normalize_board_slug(env)
            if normed and board_exists(normed):
                return normed
        except ValueError:
            pass
    try:
        f = current_board_path()
        if f.exists():
            val = f.read_text(encoding="utf-8").strip()
            if val:
                try:
                    normed = _normalize_board_slug(val)
                    if normed and board_exists(normed):
                        return normed
                except ValueError:
                    pass
    except OSError:
        pass
    return DEFAULT_BOARD


def set_current_board(slug: str) -> Path:
    """Persist ``slug`` as the active board. Returns the file written.

    Writes ``<root>/kanban/current``. The caller should validate the slug
    exists first (via :func:`board_exists`) — this function does not —
    so that ``hermes kanban boards switch <typo>`` returns an error
    instead of silently pointing at nothing.
    """
    _assert_not_delegated_child_mutation()
    normed = _normalize_board_slug(slug)
    if not normed:
        raise ValueError("board slug is required")
    path = current_board_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(normed + "\n", encoding="utf-8")
    return path


def clear_current_board() -> None:
    """Remove ``<root>/kanban/current`` so the active board reverts to ``default``."""
    _assert_not_delegated_child_mutation()
    try:
        current_board_path().unlink()
    except FileNotFoundError:
        pass


def board_dir(board: Optional[str] = None) -> Path:
    """Return the on-disk directory for ``board``.

    ``default`` is ``<root>/kanban/boards/default/`` **for metadata only**
    (board.json + workspaces/ + logs/). Its DB file stays at
    ``<root>/kanban.db`` for back-compat — see :func:`kanban_db_path`.

    All other boards live at ``<root>/kanban/boards/<slug>/`` with
    everything inside that directory including the ``kanban.db``.
    """
    slug = _normalize_board_slug(board) or DEFAULT_BOARD
    return boards_root() / slug


def board_exists(board: Optional[str] = None) -> bool:
    """Return True if the board has persisted metadata or a DB on disk.

    ``default`` is considered to always exist — its DB is created
    on first :func:`connect` and there's no way for it to be missing
    in a configuration where the kanban feature is usable at all.
    """
    slug = _normalize_board_slug(board) or DEFAULT_BOARD
    if slug == DEFAULT_BOARD:
        return True
    d = board_dir(slug)
    return (d / "board.json").exists() or (d / "kanban.db").exists()


def kanban_db_path(board: Optional[str] = None) -> Path:
    """Return the path to the ``kanban.db`` for ``board``.

    Resolution (highest precedence first):

    1. ``HERMES_KANBAN_DB`` env var — pins the path directly. Honoured for
       back-compat and for the dispatcher→worker handoff (defense in
       depth: dispatcher injects this into worker env so workers are
       immune to any path-resolution disagreement).
    2. When ``board`` arg is None, the active board from
       :func:`get_current_board` is used.
    3. Board ``default`` → ``<root>/kanban.db`` (back-compat path).
       Other boards → ``<root>/kanban/boards/<slug>/kanban.db``.
    """
    override = os.environ.get("HERMES_KANBAN_DB", "").strip()
    if override:
        return Path(override).expanduser()
    slug = _normalize_board_slug(board)
    if slug is None:
        slug = get_current_board()
    if slug == DEFAULT_BOARD:
        return kanban_home() / "kanban.db"
    return board_dir(slug) / "kanban.db"


def workspaces_root(board: Optional[str] = None) -> Path:
    """Return the directory under which ``scratch`` workspaces are created.

    Anchored per-board so workspaces don't leak between projects.
    ``HERMES_KANBAN_WORKSPACES_ROOT`` pins the path directly (highest
    precedence) — the dispatcher injects this into worker env.

    ``default`` keeps the legacy path ``<root>/kanban/workspaces/`` so
    that existing scratch workspaces from before the boards feature are
    preserved. Other boards use ``<root>/kanban/boards/<slug>/workspaces/``.
    """
    override = os.environ.get("HERMES_KANBAN_WORKSPACES_ROOT", "").strip()
    if override:
        return Path(override).expanduser()
    slug = _normalize_board_slug(board)
    if slug is None:
        slug = get_current_board()
    if slug == DEFAULT_BOARD:
        return kanban_home() / "kanban" / "workspaces"
    return board_dir(slug) / "workspaces"


def attachments_root(board: Optional[str] = None) -> Path:
    """Return the directory under which task file attachments are stored.

    Mirrors :func:`worker_logs_dir` / :func:`workspaces_root`: anchored
    per-board so attachments don't leak between projects. Each task gets
    its own ``<root>/.../attachments/<task_id>/`` subdirectory.

    ``HERMES_KANBAN_ATTACHMENTS_ROOT`` pins the path directly (highest
    precedence) for tests and unusual deployments.

    ``default`` uses ``<root>/kanban/attachments/``; other boards use
    ``<root>/kanban/boards/<slug>/attachments/``.

    Workers (which run with full file-tool access) read attached files
    by the absolute path surfaced in :func:`build_worker_context`. On the
    local terminal backend — the default for kanban — that path resolves
    directly. Remote backends (Docker/Modal) need this directory mounted;
    see the kanban docs.
    """
    override = os.environ.get("HERMES_KANBAN_ATTACHMENTS_ROOT", "").strip()
    if override:
        return Path(override).expanduser()
    slug = _normalize_board_slug(board)
    if slug is None:
        slug = get_current_board()
    if slug == DEFAULT_BOARD:
        return kanban_home() / "kanban" / "attachments"
    return board_dir(slug) / "attachments"


def task_attachments_dir(task_id: str, board: Optional[str] = None) -> Path:
    """Return the per-task attachment directory ``<root>/<task_id>/``."""
    return attachments_root(board=board) / task_id


def worker_logs_dir(board: Optional[str] = None) -> Path:
    """Return the directory under which per-task worker logs are written.

    ``default`` keeps the legacy path ``<root>/kanban/logs/``. Other
    boards use ``<root>/kanban/boards/<slug>/logs/``. Logs follow the
    board — makes ``hermes kanban log`` unambiguous even when multiple
    boards have tasks with the same id.
    """
    slug = _normalize_board_slug(board)
    if slug is None:
        slug = get_current_board()
    if slug == DEFAULT_BOARD:
        return kanban_home() / "kanban" / "logs"
    return board_dir(slug) / "logs"


def board_metadata_path(board: Optional[str] = None) -> Path:
    """Return the path to ``board.json`` for ``board``.

    Stores display metadata (display name, description, icon, color,
    created_at). The on-disk slug is the canonical identity; this file
    is purely for presentation in the CLI / dashboard.
    """
    slug = _normalize_board_slug(board) or DEFAULT_BOARD
    return board_dir(slug) / "board.json"


def _default_board_display_name(slug: str) -> str:
    """Turn a slug into a reasonable default display name.

    ``atm10-server`` → ``Atm10 Server``. Users can override via
    ``board.json`` but the default should look presentable in the
    dashboard without any follow-up editing.
    """
    return " ".join(part.capitalize() for part in slug.replace("_", "-").split("-") if part) or slug


# The ownership/admission fields dispatch and every machine read decide on,
# and the exact shape each one must have to be believed. A ``board.json`` is a
# durable authority document, so a field of the wrong TYPE is as unreadable as
# a file that will not parse: ``project_id: 123`` names no Project this build
# can resolve, and ``dispatch_enabled: "yes"`` is not the owner's admission bit.
# Validated after the file's own fields are merged, so the file cannot smuggle
# a malformed authority value past the defaults it overrides.
def _board_authority_fields_valid(meta: Mapping[str, Any]) -> bool:
    project_id = meta.get("project_id")
    if project_id is not None and (
        not isinstance(project_id, str) or not project_id.strip()
    ):
        return False
    for field in ("dispatch_enabled", "dispatch_paused_by_owner", "archived"):
        if not isinstance(meta.get(field), bool):
            return False
    return True


def read_board_metadata(board: Optional[str] = None) -> dict:
    """Return ``board.json`` contents (or synthesized defaults).

    Never raises — a missing / malformed ``board.json`` falls back to a
    synthesised entry so the dashboard always has something to render.
    Includes the canonical ``slug`` and ``db_path`` so the caller
    doesn't need to reconstruct them.

    It also reports, as ``ownership_verified``, whether the ownership metadata
    in that entry was actually READ or merely defaulted. The two are not the
    same fact: an absent ``board.json`` genuinely publishes no owner, while one
    that exists and cannot be parsed may publish any owner at all — and a
    synthesised ``project_id: None`` for the second case is what let a legacy
    owner board's work stay unbound and claimable with no route authority. The
    flag is written after the file's own fields are merged, so the file can
    never claim its own metadata is verified.

    "Could not be parsed" is not the only way ownership goes unread. A file that
    IS valid JSON can still carry an authority field of a shape this build
    cannot act on, and merging it left ``ownership_verified: True`` on metadata
    whose owner nobody could resolve — the same unbound-legacy-work and
    unproven-dispatch-admission failure, reached through a well-formed file. So
    every authority field is type-checked too (see
    :func:`_board_authority_fields_valid`).
    """
    slug = _normalize_board_slug(board) or DEFAULT_BOARD
    meta: dict[str, Any] = {
        "slug": slug,
        "name": _default_board_display_name(slug),
        "description": "",
        "icon": "",
        "color": "",
        "default_workdir": None,
        # Optional first-class Project this board is scoped to. When set, new
        # tasks inherit it (deterministic worktree + branch under the project's
        # primary repo) and ``default_workdir`` mirrors the project's primary
        # path so the persistent-workspace inheritance path keeps working.
        "project_id": None,
        # Owner-approved project execution is opt-in. Upstream/manual boards
        # keep their historical behavior unless the dispatcher is explicitly
        # configured to require this admission bit.
        "dispatch_enabled": False,
        "dispatch_paused_by_owner": False,
        "created_at": None,
        "archived": False,
    }
    verified = True
    try:
        p = board_metadata_path(slug)
        if p.exists():
            raw = json.loads(p.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                # Never let the metadata file claim a different slug than
                # its directory — trust the filesystem.
                raw["slug"] = slug
                meta.update(raw)
            else:
                verified = False
    except (OSError, json.JSONDecodeError):
        verified = False
    if verified and not _board_authority_fields_valid(meta):
        verified = False
    meta["db_path"] = str(kanban_db_path(slug))
    meta["ownership_verified"] = verified
    return meta


def write_board_metadata(
    board: Optional[str],
    *,
    name: Optional[str] = None,
    description: Optional[str] = None,
    icon: Optional[str] = None,
    color: Optional[str] = None,
    archived: Optional[bool] = None,
    default_workdir: Optional[str] = None,
    project_id: Optional[str] = None,
    dispatch_enabled: Optional[bool] = None,
    dispatch_paused_by_owner: Optional[bool] = None,
) -> dict:
    """Create / update ``board.json`` for ``board``.

    Preserves any existing fields not mentioned in the call. Sets
    ``created_at`` on first write. Returns the resulting metadata dict.

    ``project_id``: ``None`` leaves it unchanged; empty string clears the
    project scope; a value sets it (not validated here — the caller resolves
    it against ``projects_db``).
    """
    _assert_not_delegated_child_mutation()
    slug = _normalize_board_slug(board) or DEFAULT_BOARD
    meta = read_board_metadata(slug)
    if meta.pop("ownership_verified", True) is False:
        # Rewriting a board.json this process could not read would overwrite
        # whatever ownership it published with a synthesised "no owner". A
        # well-formed file carrying a malformed authority field is the same
        # hazard: this call would preserve that value verbatim for every field
        # it was not asked to change.
        raise ValueError(
            f"board {slug!r} has unreadable metadata; refusing to overwrite it"
        )
    # Preserve existing DB-derived fields — they get re-computed each
    # read but shouldn't be written into board.json.
    meta.pop("db_path", None)
    if name is not None:
        meta["name"] = str(name).strip() or _default_board_display_name(slug)
    if description is not None:
        meta["description"] = str(description)
    if icon is not None:
        meta["icon"] = str(icon)
    if color is not None:
        meta["color"] = str(color)
    if archived is not None:
        meta["archived"] = bool(archived)
    if default_workdir is not None:
        meta["default_workdir"] = str(default_workdir) if default_workdir else None
    if project_id is not None:
        meta["project_id"] = str(project_id) if project_id else None
    if dispatch_enabled is not None:
        meta["dispatch_enabled"] = bool(dispatch_enabled)
    if dispatch_paused_by_owner is not None:
        meta["dispatch_paused_by_owner"] = bool(dispatch_paused_by_owner)
    if not meta.get("created_at"):
        meta["created_at"] = int(time.time())
    path = board_metadata_path(slug)
    path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write_text(path, json.dumps(meta, indent=2, ensure_ascii=False) + "\n")
    meta["db_path"] = str(kanban_db_path(slug))
    return meta


def board_dispatch_allowed(metadata: Mapping[str, Any]) -> bool:
    """Return whether strict owner-project dispatch may claim this board."""
    return (
        metadata.get("dispatch_enabled") is True
        and metadata.get("dispatch_paused_by_owner") is not True
        and metadata.get("archived") is not True
        and board_ownership_verified(metadata)
    )


def board_ownership_verified(metadata: Mapping[str, Any]) -> bool:
    """Whether this board's published ownership was read rather than guessed.

    ``False`` for a ``board.json`` that exists and could not be parsed, and for
    one that parsed but carries an authority field of a shape this build cannot
    act on. Unreadable ownership metadata cannot be shown to be absent, so it
    cannot be shown NOT to name an owner Project — and owner work whose route
    nobody can prove must not be dispatched. A board with no metadata file at
    all is verified: it genuinely publishes no owner.
    """
    return metadata.get("ownership_verified") is not False


@contextlib.contextmanager
def board_dispatch_lock(board: Optional[str], *, wait_seconds: float = 5.0):
    """Hold one board's native dispatch lock, failing closed when it is busy.

    The public form of the guard the dispatcher tick takes, for the owner-side
    operations that must be ordered against a claim: while this is held no
    dispatcher tick on this board can be inside its critical section, so no new
    claim can start and no spawn can launch. Raises ``TimeoutError`` rather than
    proceeding unlocked — a caller that answers the owner about execution state
    must not race a claim and then report a state that was never true.
    """
    slug = _normalize_board_slug(board) or DEFAULT_BOARD
    with _dispatch_tick_lock(
        kanban_db_path(board=slug),
        wait_seconds=wait_seconds,
        fail_open=False,
    ) as held:
        if not held:
            raise TimeoutError(f"board {slug!r} dispatch lock is busy")
        yield slug


def write_board_dispatch_state(
    board: Optional[str],
    *,
    dispatch_enabled: Optional[bool] = None,
    dispatch_paused_by_owner: Optional[bool] = None,
    wait_seconds: float = 5.0,
) -> dict:
    """Atomically publish owner execution state against the dispatch lock.

    Once a pause returns, every earlier dispatcher tick has left its critical
    section and every later tick must observe the new metadata before it may
    claim work. Running workers are deliberately untouched.
    """
    with board_dispatch_lock(board, wait_seconds=wait_seconds) as slug:
        return write_board_metadata(
            slug,
            dispatch_enabled=dispatch_enabled,
            dispatch_paused_by_owner=dispatch_paused_by_owner,
        )


def _atomic_write_text(path: Path, text: str) -> None:
    """Write ``text`` to ``path`` via same-directory tmp file + fsync + ``os.replace``.

    ``board.json`` is a durable publication point crash-recovery flows (e.g.
    owner-workspace bootstrap) reread to decide whether to reuse an already-
    published board instead of creating another. A bare ``write_text`` can
    leave a truncated/partial file behind if the process dies mid-write;
    ``os.replace`` on the same filesystem is atomic, so readers only ever see
    the fully-old or fully-new content, never a torn write.
    """
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}-{secrets.token_hex(4)}")
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        with contextlib.suppress(OSError):
            tmp.unlink()


def create_board(
    slug: str,
    *,
    name: Optional[str] = None,
    description: Optional[str] = None,
    icon: Optional[str] = None,
    color: Optional[str] = None,
    default_workdir: Optional[str] = None,
    project_id: Optional[str] = None,
) -> dict:
    """Create a new board directory + DB + metadata. Idempotent.

    Returns the resulting metadata. Raises :class:`ValueError` for a
    malformed slug; returns the existing metadata (not an error) if the
    board already exists — matching ``mkdir -p`` semantics.
    """
    normed = _normalize_board_slug(slug)
    if not normed:
        raise ValueError("board slug is required")
    meta = write_board_metadata(
        normed,
        name=name,
        description=description,
        icon=icon,
        color=color,
        default_workdir=default_workdir,
        project_id=project_id,
    )
    # Touch the DB so list_boards() sees it immediately.
    init_db(board=normed)
    return meta


def list_boards(*, include_archived: bool = True) -> list[dict]:
    """Enumerate all boards that exist on disk.

    Always includes ``default`` (even when the ``boards/default/``
    metadata dir doesn't exist, because its DB is at the legacy path).
    Other boards are discovered by scanning ``boards/`` for subdirectories
    that either contain a ``kanban.db`` or a ``board.json``.

    Returns a list of metadata dicts, sorted with ``default`` first and
    the rest alphabetically.
    """
    entries: list[dict] = []
    seen: set[str] = set()

    # Default board is always first.
    entries.append(read_board_metadata(DEFAULT_BOARD))
    seen.add(DEFAULT_BOARD)

    root = boards_root()
    if root.is_dir():
        for child in sorted(root.iterdir(), key=lambda p: p.name.lower()):
            if not child.is_dir():
                continue
            slug = child.name
            # Keep slug normalisation soft for discovery — but skip dirs
            # that don't parse as valid slugs so we don't surface junk.
            try:
                normed = _normalize_board_slug(slug)
            except ValueError:
                continue
            if not normed or normed in seen:
                continue
            has_db = (child / "kanban.db").exists()
            has_meta = (child / "board.json").exists()
            if not (has_db or has_meta):
                continue
            meta = read_board_metadata(normed)
            if meta.get("archived") and not include_archived:
                continue
            entries.append(meta)
            seen.add(normed)
    return entries


def remove_board(slug: str, *, archive: bool = True) -> dict:
    """Remove or archive a board.

    ``archive=True`` (default) moves the board's directory to
    ``<root>/kanban/boards/_archived/<slug>-<timestamp>/`` so the data
    is recoverable. ``archive=False`` deletes the directory outright.

    The ``default`` board cannot be removed — raises :class:`ValueError`.
    Returns a summary dict describing what happened (``{"slug", "action",
    "new_path"}``).
    """
    _assert_not_delegated_child_mutation()
    normed = _normalize_board_slug(slug)
    if not normed:
        raise ValueError("board slug is required")
    if normed == DEFAULT_BOARD:
        raise ValueError("the 'default' board cannot be removed")
    d = board_dir(normed)
    if not d.exists():
        raise ValueError(f"board {normed!r} does not exist")

    # If the user removed the currently-active board, revert to default.
    if get_current_board() == normed:
        clear_current_board()

    # A concurrent connect(board=normed) after the rename/delete recreates
    # an empty sqlite file via mkdir(exist_ok=True); the cache entry must be
    # dropped first so the schema init pass re-runs on that fresh file.
    _INITIALIZED_PATHS.discard(str((d / "kanban.db").resolve()))

    if archive:
        archive_root = boards_root() / "_archived"
        archive_root.mkdir(parents=True, exist_ok=True)
        ts = int(time.time())
        target = archive_root / f"{normed}-{ts}"
        # Avoid collision on rapid double-archives.
        suffix = 1
        while target.exists():
            target = archive_root / f"{normed}-{ts}-{suffix}"
            suffix += 1
        d.rename(target)
        return {"slug": normed, "action": "archived", "new_path": str(target)}
    else:
        import shutil
        shutil.rmtree(d)
        return {"slug": normed, "action": "deleted", "new_path": ""}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class Task:
    """In-memory view of a row from the ``tasks`` table."""

    id: str
    title: str
    body: Optional[str]
    assignee: Optional[str]
    status: str
    priority: int
    created_by: Optional[str]
    created_at: int
    started_at: Optional[int]
    completed_at: Optional[int]
    workspace_kind: str
    workspace_path: Optional[str]
    claim_lock: Optional[str]
    claim_expires: Optional[int]
    tenant: Optional[str]
    responsibility: Optional[str] = None
    branch_name: Optional[str] = None
    project_id: Optional[str] = None
    # Declared repository write ownership for this task. ``None`` means a
    # legacy/unscoped task and is treated as exclusive access to the whole
    # Project. ``[]`` is explicitly read-only. Relative POSIX paths own that
    # file or subtree; ``["."]`` is explicit whole-repository ownership.
    owned_paths: Optional[list[str]] = None
    # Explicit integration contract: when true, completion must prove that
    # every same-Project parent git head is an ancestor of this task's head.
    integrates_parent_heads: bool = False
    # Exact git revision receipts for project worktrees. ``base_commit`` is
    # captured before the worker starts and never rewritten; ``head_commit``
    # is derived by the kernel only after a clean, in-scope completion.
    base_commit: Optional[str] = None
    head_commit: Optional[str] = None
    result: Optional[str] = None
    idempotency_key: Optional[str] = None
    # Unified non-success counter. Incremented on any of:
    #   * spawn failure (dispatcher couldn't launch the worker)
    #   * timed_out outcome (worker exceeded max_runtime_seconds)
    #   * crashed outcome (worker PID vanished)
    # Reset to 0 only on a successful completion. See
    # ``_record_task_failure`` for the circuit-breaker trip rule.
    # (Pre-rename column: ``spawn_failures``.)
    consecutive_failures: int = 0
    worker_pid: Optional[int] = None
    # Short excerpt of the last failure's error text (any outcome, not
    # just spawn). Pre-rename column: ``last_spawn_error``.
    last_failure_error: Optional[str] = None
    max_runtime_seconds: Optional[int] = None
    last_heartbeat_at: Optional[int] = None
    current_run_id: Optional[int] = None
    workflow_template_id: Optional[str] = None
    current_step_key: Optional[str] = None
    # Force-loaded skills for the worker on this task (passed via
    # --skills). Stored as a JSON array of skill names. None = use only
    # the defaults; empty list = explicitly no extra skills.
    skills: Optional[list] = None
    model_override: Optional[str] = None
    # Provider that ``model_override`` belongs to. When set, the dispatcher
    # passes ``--provider <name>`` alongside ``-m <model>`` so the worker
    # resolves the model against the right backend instead of the profile's
    # configured provider. NULL = worker profile's provider resolves the
    # model (pre-existing behaviour). Solves the "model from provider A,
    # profile configured for provider B" mismatch class.
    provider_override: Optional[str] = None
    # Per-task reasoning effort for the worker (one of
    # ``hermes_constants.VALID_REASONING_EFFORTS``, or ``"none"`` for thinking
    # off). When set, the dispatcher passes ``--reasoning <level>`` so the
    # worker runs at that depth regardless of the profile's
    # ``agent.reasoning_effort``. NULL = the worker profile's own setting.
    reasoning_effort: Optional[str] = None
    # Owner-approved semantic task class (``routine``/``deep``) the model
    # policy resolves the route from. NULL for every ordinary/manual task.
    execution_tier: Optional[str] = None
    # Versioned, digest-bound owner-approved model-route lock, or NULL for
    # every ordinary/manual task. When set, the assignee, the three fields
    # above and this task's tier are the immutable authority the owner
    # approved; see ``policy_lock_error``.
    model_policy_lock: Optional[str] = None
    # Per-task override for the consecutive-failure circuit breaker.
    # The value is the failure count at which the breaker trips — e.g.
    # ``max_retries=1`` blocks on the first failure (zero retries),
    # ``max_retries=3`` blocks on the third (two retries allowed).
    # ``None`` (the common case) falls through to the dispatcher-level
    # ``kanban.failure_limit`` config, and then to ``DEFAULT_FAILURE_LIMIT``.
    # Name matches the ``--max-retries`` CLI flag on ``kanban create``.
    max_retries: Optional[int] = None
    # When True, the dispatched worker runs in a Ralph-style goal loop
    # (the same engine behind the ``/goal`` slash command): after each
    # turn an auxiliary judge model evaluates the worker's response
    # against this card's title/body (treated as the goal). If the judge
    # says "not done" and budget remains, the worker is fed a
    # continuation prompt IN THE SAME SESSION and keeps working until the
    # judge agrees, the goal-turn budget is exhausted (→ kanban_block),
    # or the worker explicitly blocks/completes. ``False`` (default) =
    # the classic single-shot worker. ``goal_max_turns`` bounds the loop.
    goal_mode: bool = False
    # Goal-loop turn budget for ``goal_mode`` workers. ``None`` falls
    # through to the goals engine default (``goals.DEFAULT_MAX_TURNS``).
    goal_max_turns: Optional[int] = None
    # Originating chat/agent session id, when the task was created from
    # within an agent loop that propagated ``HERMES_SESSION_ID``. NULL for
    # tasks created from the CLI, the dashboard, or any path that doesn't
    # set the env var. Lets clients render a per-session board without
    # relying on tenant + time-window heuristics.
    session_id: Optional[str] = None
    # Typed block reason (one of VALID_BLOCK_KINDS) or None for legacy/un-typed
    # blocks. Set by ``block_task``; preserved across unblock so a re-block for
    # the same kind is recognisable as an unblock↔re-block loop.
    block_kind: Optional[str] = None
    # Unblock-loop counter. See the column comment in SCHEMA_SQL and
    # ``BLOCK_RECURRENCE_LIMIT``. Reset only on successful completion.
    block_recurrences: int = 0

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "Task":
        keys = set(row.keys())
        # Parse skills JSON blob if present
        skills_value: Optional[list] = None
        if "skills" in keys and row["skills"]:
            try:
                parsed = json.loads(row["skills"])
                if isinstance(parsed, list):
                    skills_value = [str(s) for s in parsed if s]
            except Exception:
                skills_value = None
        owned_paths_value: Optional[list[str]] = None
        if "owned_paths" in keys and row["owned_paths"] is not None:
            try:
                parsed = json.loads(row["owned_paths"])
                if isinstance(parsed, list):
                    owned_paths_value = normalize_owned_paths(parsed)
            except Exception:
                owned_paths_value = None
        return cls(
            id=row["id"],
            title=row["title"],
            body=row["body"],
            assignee=row["assignee"],
            status=row["status"],
            priority=row["priority"],
            created_by=row["created_by"],
            created_at=row["created_at"],
            started_at=row["started_at"],
            completed_at=row["completed_at"],
            workspace_kind=row["workspace_kind"],
            workspace_path=row["workspace_path"],
            branch_name=row["branch_name"] if "branch_name" in keys else None,
            project_id=row["project_id"] if "project_id" in keys else None,
            owned_paths=owned_paths_value,
            integrates_parent_heads=(
                bool(row["integrates_parent_heads"])
                if "integrates_parent_heads" in keys
                else False
            ),
            base_commit=(
                row["base_commit"] if "base_commit" in keys and row["base_commit"] else None
            ),
            head_commit=(
                row["head_commit"] if "head_commit" in keys and row["head_commit"] else None
            ),
            claim_lock=row["claim_lock"],
            claim_expires=row["claim_expires"],
            tenant=row["tenant"] if "tenant" in keys else None,
            responsibility=(
                row["responsibility"] if "responsibility" in keys else None
            ),
            result=row["result"] if "result" in keys else None,
            idempotency_key=row["idempotency_key"] if "idempotency_key" in keys else None,
            consecutive_failures=(
                row["consecutive_failures"] if "consecutive_failures" in keys
                # Pre-migration fallback: ``_migrate_add_optional_columns`` always
                # adds ``consecutive_failures`` now, so this branch is only reachable
                # on a DB that was never opened since pre-#20410 code ran. Keep for
                # belt-and-suspenders safety; in practice it is dead code post-migration.
                else (row["spawn_failures"] if "spawn_failures" in keys else 0)
            ),
            worker_pid=row["worker_pid"] if "worker_pid" in keys else None,
            last_failure_error=(
                row["last_failure_error"] if "last_failure_error" in keys
                # Same belt-and-suspenders fallback as consecutive_failures above.
                else (row["last_spawn_error"] if "last_spawn_error" in keys else None)
            ),
            max_runtime_seconds=(
                row["max_runtime_seconds"] if "max_runtime_seconds" in keys else None
            ),
            last_heartbeat_at=(
                row["last_heartbeat_at"] if "last_heartbeat_at" in keys else None
            ),
            current_run_id=(
                row["current_run_id"] if "current_run_id" in keys else None
            ),
            workflow_template_id=(
                row["workflow_template_id"] if "workflow_template_id" in keys else None
            ),
            current_step_key=(
                row["current_step_key"] if "current_step_key" in keys else None
            ),
            skills=skills_value,
            model_override=row["model_override"] if "model_override" in keys and row["model_override"] else None,
            provider_override=(
                row["provider_override"]
                if "provider_override" in keys and row["provider_override"]
                else None
            ),
            reasoning_effort=(
                row["reasoning_effort"]
                if "reasoning_effort" in keys and row["reasoning_effort"]
                else None
            ),
            execution_tier=(
                row["execution_tier"]
                if "execution_tier" in keys and row["execution_tier"]
                else None
            ),
            model_policy_lock=(
                row["model_policy_lock"]
                if "model_policy_lock" in keys and row["model_policy_lock"]
                else None
            ),
            max_retries=(
                row["max_retries"] if "max_retries" in keys else None
            ),
            goal_mode=(
                bool(row["goal_mode"]) if "goal_mode" in keys and row["goal_mode"] else False
            ),
            goal_max_turns=(
                row["goal_max_turns"] if "goal_max_turns" in keys and row["goal_max_turns"] else None
            ),
            session_id=(
                row["session_id"] if "session_id" in keys else None
            ),
            block_kind=(
                row["block_kind"] if "block_kind" in keys and row["block_kind"] else None
            ),
            block_recurrences=(
                int(row["block_recurrences"])
                if "block_recurrences" in keys and row["block_recurrences"] is not None
                else 0
            ),
        )


@dataclass
class Run:
    """In-memory view of a ``task_runs`` row.

    A run is one attempt to execute a task — created on claim, closed
    on complete/block/crash/timeout/spawn_failure/reclaim. Multiple runs
    per task when retries happen. Carries the claim machinery, PID,
    heartbeat, and the structured handoff summary that downstream workers
    read via ``build_worker_context``.
    """

    id: int
    task_id: str
    profile: Optional[str]
    step_key: Optional[str]
    status: str
    claim_lock: Optional[str]
    claim_expires: Optional[int]
    worker_pid: Optional[int]
    max_runtime_seconds: Optional[int]
    last_heartbeat_at: Optional[int]
    started_at: int
    ended_at: Optional[int]
    outcome: Optional[str]
    summary: Optional[str]
    metadata: Optional[dict]
    error: Optional[str]

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "Run":
        try:
            meta = json.loads(row["metadata"]) if row["metadata"] else None
        except Exception:
            meta = None
        return cls(
            id=int(row["id"]),
            task_id=row["task_id"],
            profile=row["profile"],
            step_key=row["step_key"],
            status=row["status"],
            claim_lock=row["claim_lock"],
            claim_expires=row["claim_expires"],
            worker_pid=row["worker_pid"],
            max_runtime_seconds=row["max_runtime_seconds"],
            last_heartbeat_at=row["last_heartbeat_at"],
            started_at=int(row["started_at"]),
            ended_at=(int(row["ended_at"]) if row["ended_at"] is not None else None),
            outcome=row["outcome"],
            summary=row["summary"],
            metadata=meta,
            error=row["error"],
        )


@dataclass
class Comment:
    id: int
    task_id: str
    author: str
    body: str
    created_at: int


@dataclass
class Attachment:
    """In-memory view of a row from the ``task_attachments`` table."""

    id: int
    task_id: str
    filename: str
    stored_path: str
    content_type: Optional[str]
    size: int
    uploaded_by: Optional[str]
    created_at: int


@dataclass
class Event:
    id: int
    task_id: str
    kind: str
    payload: Optional[dict]
    created_at: int
    run_id: Optional[int] = None


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS tasks (
    id                   TEXT PRIMARY KEY,
    title                TEXT NOT NULL,
    body                 TEXT,
    assignee             TEXT,
    -- Optional stable logical responsibility. This describes what the work is
    -- for; assignee remains the runtime profile that executes it.
    responsibility       TEXT,
    status               TEXT NOT NULL,
    priority             INTEGER DEFAULT 0,
    created_by           TEXT,
    created_at           INTEGER NOT NULL,
    started_at           INTEGER,
    completed_at         INTEGER,
    workspace_kind       TEXT NOT NULL DEFAULT 'scratch',
    workspace_path       TEXT,
    branch_name          TEXT,
    -- Optional link to a first-class Project (hermes_cli/projects_db). When set,
    -- the task's worktree is anchored under the project's primary repo with a
    -- deterministic branch name instead of a random wt/<task-id> fallback.
    project_id           TEXT,
    -- Repository write ownership used by bounded parallel execution. NULL is
    -- legacy fail-closed whole-repo ownership; [] is explicitly read-only;
    -- relative POSIX paths own one file/subtree; ["."] owns the whole repo.
    owned_paths          TEXT,
    integrates_parent_heads INTEGER NOT NULL DEFAULT 0,
    -- Kernel-derived exact git revision receipts for worktree execution.
    base_commit          TEXT,
    head_commit          TEXT,
    claim_lock           TEXT,
    claim_expires        INTEGER,
    tenant               TEXT,
    result               TEXT,
    idempotency_key      TEXT,
    -- Unified consecutive-failure counter. Incremented on spawn
    -- failure, timeout, or crash; reset only on successful completion.
    -- The circuit breaker in _record_task_failure trips when this
    -- exceeds DEFAULT_FAILURE_LIMIT consecutive non-successes.
    consecutive_failures INTEGER NOT NULL DEFAULT 0,
    worker_pid           INTEGER,
    -- Short excerpt of the most recent failure's error text.
    last_failure_error   TEXT,
    max_runtime_seconds  INTEGER,
    last_heartbeat_at    INTEGER,
    -- Pointer into task_runs for the currently-active run (NULL if no
    -- run is in-flight). Denormalised for cheap reads.
    current_run_id       INTEGER,
    -- Forward-compat for v2 workflow routing. In v1 the kernel writes
    -- these when the task is opted into a template but otherwise ignores
    -- them; the dispatcher doesn't consult them for routing yet.
    workflow_template_id TEXT,
    current_step_key     TEXT,
    -- Force-loaded skills for the worker on this task, stored as JSON.
    -- Passed to the worker via `--skills`. NULL or empty array = no extras.
    skills               TEXT,
    -- Per-task model override. When set, the dispatcher passes -m <model>
    -- to the worker, overriding the profile's default model. NULL = use
    -- the profile default.
    model_override       TEXT,
    -- Provider the model override belongs to. When set (alongside
    -- model_override), the dispatcher passes --provider <name> so the
    -- worker resolves the model against the right backend instead of the
    -- profile's configured provider. NULL = profile provider.
    provider_override    TEXT,
    -- Per-task reasoning effort for the worker (minimal|low|medium|high|
    -- xhigh|max|ultra, or 'none' for thinking off). When set, the dispatcher
    -- passes --reasoning <level> so the worker runs at that depth regardless
    -- of the profile's agent.reasoning_effort. NULL = profile setting.
    reasoning_effort     TEXT,
    -- Owner-approved semantic task class (routine/deep) the model policy
    -- resolves this task's route from. NULL for ordinary/manual tasks.
    execution_tier       TEXT,
    -- Versioned, digest-bound owner-approved model-route lock, or NULL for
    -- every ordinary/manual task. When set, the assignee, the three columns
    -- above and execution_tier are the immutable authority approved for this
    -- exact task: the route mutators refuse it, a role transition must
    -- re-derive a separately approved route, and the dispatcher disables
    -- fallbacks for it. A non-NULL value that fails policy_lock_error() is
    -- never treated as NULL — it fails closed.
    model_policy_lock    TEXT,
    -- 1 when a committed owner receipt owns this work row. Receipt ownership
    -- is the durable fact that makes a task's route the owner's to approve,
    -- so it must live on the row itself: a legacy owner task carries NULL
    -- execution_tier AND NULL model_policy_lock, and without this column it
    -- is indistinguishable from an ordinary manual card and would dispatch on
    -- whatever route the profile happens to hold. 0 (the default) is every
    -- ordinary/manual/CLI task, which keeps its historical behaviour.
    owner_receipt_bound  INTEGER NOT NULL DEFAULT 0,
    -- Which parking operation put this row in ``scheduled``, when one did.
    -- ``scheduled`` is a shared column: an owner approval parks work there
    -- until its receipt is durable, and an owner ALSO postpones work there
    -- deliberately. Activation therefore cannot match on the column — it
    -- compare-and-swaps this exact generation, so replaying a receipt whose
    -- work was already released (and has since been postponed again) matches
    -- nothing. Cleared back to NULL the moment the row is released.
    park_generation      TEXT,
    -- Per-task override for the consecutive-failure circuit breaker.
    -- The value is the failure count at which the breaker trips — e.g.
    -- ``max_retries=1`` blocks on the first failure. NULL (the common
    -- case) falls through to the dispatcher-level ``kanban.failure_limit``
    -- config and then ``DEFAULT_FAILURE_LIMIT``.
    max_retries          INTEGER,
    -- When 1, the dispatched worker runs in a Ralph-style goal loop: an
    -- auxiliary judge re-evaluates the worker's response against the
    -- card title/body after each turn and feeds a continuation prompt
    -- back into the SAME session until the judge agrees the work is done
    -- or ``goal_max_turns`` is exhausted. NULL/0 = classic single-shot
    -- worker (the default).
    goal_mode            INTEGER NOT NULL DEFAULT 0,
    -- Goal-loop turn budget for ``goal_mode`` workers. NULL = use the
    -- goals-engine default.
    goal_max_turns       INTEGER,
    -- Originating chat/agent session id when the task was created from
    -- inside an agent loop that propagated ``HERMES_SESSION_ID``. NULL
    -- for tasks created from the CLI, dashboard, or any path that doesn't
    -- set the env var. Indexed so per-session list queries stay cheap on
    -- larger boards.
    session_id           TEXT,
    -- Typed block reason set by ``block_task`` (one of VALID_BLOCK_KINDS, or
    -- NULL for legacy/un-typed blocks). Drives routing: ``dependency`` never
    -- sits in ``blocked`` (goes to ``todo`` for parent-gating); the others go
    -- to ``blocked`` for a human. Preserved across unblock so a re-block for
    -- the SAME kind can be recognised as a loop.
    block_kind           TEXT,
    -- Unblock-loop counter. Incremented each time a task is re-blocked for the
    -- same truly-blocked reason after having been unblocked. When it reaches
    -- BLOCK_RECURRENCE_LIMIT the task is routed to ``triage`` instead of
    -- ``blocked`` so a cron can't spin it forever. Reset to 0 only on a
    -- successful completion — NOT on unblock (resetting on unblock is exactly
    -- the amnesia that let the loop run unbounded).
    block_recurrences    INTEGER NOT NULL DEFAULT 0,
    -- Discriminates ordinary work tasks ('work', the create_task default) from native
    -- recommendation cards (see create_recommendation). recommendation_* / target_profile /
    -- review_policy / provenance_* are populated only when task_kind='recommendation' (NULL otherwise).
    task_kind             TEXT NOT NULL DEFAULT 'work',
    recommendation_kind        TEXT,
    recommendation_subject_id  TEXT,
    recommendation_label       TEXT,
    recommendation_rationale   TEXT,
    target_profile             TEXT,
    review_policy              TEXT,
    provenance_authority       TEXT,
    provenance_ref             TEXT,
    provenance_observed_at     INTEGER,
    recommendation_evidence          TEXT,
    recommendation_decision          TEXT,
    recommendation_effective_state   TEXT,
    recommendation_lifecycle_version INTEGER
);

CREATE TABLE IF NOT EXISTS task_links (
    parent_id  TEXT NOT NULL,
    child_id   TEXT NOT NULL,
    PRIMARY KEY (parent_id, child_id)
);

CREATE TABLE IF NOT EXISTS task_comments (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id    TEXT NOT NULL,
    author     TEXT NOT NULL,
    body       TEXT NOT NULL,
    created_at INTEGER NOT NULL,
    -- Durable operation identity for idempotent replay (e.g. the
    -- owner-workspace mutation kernel's actor/profile/idempotency-key
    -- scoped key). NULL for ordinary comments that don't need dedup.
    -- See add_comment()'s operation_key parameter.
    operation_key TEXT
);

CREATE TABLE IF NOT EXISTS task_events (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id    TEXT NOT NULL,
    run_id     INTEGER,
    kind       TEXT NOT NULL,
    payload    TEXT,
    created_at INTEGER NOT NULL
);

-- Historical attempt record. Each time the dispatcher claims a task, a
-- new row is created here; claim state, PID, heartbeat, runtime cap,
-- and structured summary all live on the run, not the task. Multiple
-- rows per task id when the task was retried after crash/timeout/block.
-- v2 of the kanban schema will use ``step_key`` to drive per-stage
-- workflow routing; in v1 the column is nullable and unused (kernel
-- ignores it).
CREATE TABLE IF NOT EXISTS task_runs (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id             TEXT NOT NULL,
    profile             TEXT,
    step_key            TEXT,
    status              TEXT NOT NULL,
    -- status: running | done | blocked | crashed | timed_out | failed | released
    claim_lock          TEXT,
    claim_expires       INTEGER,
    worker_pid          INTEGER,
    max_runtime_seconds INTEGER,
    last_heartbeat_at   INTEGER,
    started_at          INTEGER NOT NULL,
    ended_at            INTEGER,
    outcome             TEXT,
    -- outcome: completed | blocked | crashed | timed_out | spawn_failed |
    --          gave_up | reclaimed | (null while still running)
    summary             TEXT,
    metadata            TEXT,
    error               TEXT
);

-- Files attached to a task (PDFs, images, source documents). The blob
-- lives on disk under ``attachments_root(board)/<task_id>/<stored_name>``;
-- this row carries metadata + the absolute ``stored_path`` so the
-- dashboard can list/download and ``build_worker_context`` can surface
-- the absolute path to the worker (which has full file-tool access). See
-- #35338.
CREATE TABLE IF NOT EXISTS task_attachments (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id      TEXT NOT NULL,
    filename     TEXT NOT NULL,
    stored_path  TEXT NOT NULL,
    content_type TEXT,
    size         INTEGER NOT NULL DEFAULT 0,
    uploaded_by  TEXT,
    created_at   INTEGER NOT NULL
);

-- Subscription from a gateway source (platform + chat + thread) to a
-- task. The gateway's kanban-notifier watcher tails task_events and
-- pushes ``completed`` / ``blocked`` / ``spawn_auto_blocked`` events to
-- the original requester so human-in-the-loop workflows close the loop.
CREATE TABLE IF NOT EXISTS kanban_notify_subs (
    task_id       TEXT NOT NULL,
    platform      TEXT NOT NULL,
    chat_id       TEXT NOT NULL,
    thread_id     TEXT NOT NULL DEFAULT '',
    user_id       TEXT,
    user_id_alt   TEXT,
    chat_type     TEXT,
    notifier_profile TEXT,
    delivery_mode TEXT NOT NULL DEFAULT 'notify',
    delivery_metadata TEXT,
    created_at    INTEGER NOT NULL,
    last_event_id INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (task_id, platform, chat_id, thread_id)
);

CREATE INDEX IF NOT EXISTS idx_tasks_assignee_status ON tasks(assignee, status);
CREATE INDEX IF NOT EXISTS idx_tasks_status          ON tasks(status);
CREATE INDEX IF NOT EXISTS idx_links_child           ON task_links(child_id);
CREATE INDEX IF NOT EXISTS idx_links_parent          ON task_links(parent_id);
CREATE INDEX IF NOT EXISTS idx_comments_task         ON task_comments(task_id, created_at);
CREATE INDEX IF NOT EXISTS idx_events_task           ON task_events(task_id, created_at);
CREATE INDEX IF NOT EXISTS idx_runs_task             ON task_runs(task_id, started_at);
CREATE INDEX IF NOT EXISTS idx_runs_status           ON task_runs(status);
CREATE INDEX IF NOT EXISTS idx_attachments_task      ON task_attachments(task_id, created_at);
CREATE INDEX IF NOT EXISTS idx_notify_task           ON kanban_notify_subs(task_id);
"""


# ---------------------------------------------------------------------------
# Connection helpers
# ---------------------------------------------------------------------------

_INITIALIZED_PATHS: set[str] = set()
_INIT_LOCK = threading.RLock()
_SQLITE_HEADER = b"SQLite format 3\x00"
DEFAULT_BUSY_TIMEOUT_MS = 120_000

# Maximum number of ``<db>.corrupt.<hash>.bak`` quarantine files retained per
# board DB. Content-addressing already dedupes identical corrupt bytes, but
# repeatedly-mutating corruption (partial repairs, further damage between
# dispatcher retries) mints a new fingerprint each time; without a cap a user
# accumulated 124 backups. Oldest-by-mtime files beyond the cap are pruned
# right after each new backup is created.
_CORRUPT_BACKUP_RETENTION = 10

# Bounded acquire for the cross-process init lock (#36644). The original bare
# blocking flock had no timeout, so a wedged holder blocked the dispatcher's
# next-tick connect forever. We retry a non-blocking acquire up to this
# deadline, polling at this interval, then proceed without the cross-process
# lock (the in-process _INIT_LOCK + idempotent init remain the backstop).
_INIT_LOCK_TIMEOUT_SECONDS = 10.0
_INIT_LOCK_POLL_SECONDS = 0.05


def _resolve_busy_timeout_ms() -> int:
    """Return the SQLite busy timeout for Kanban connections.

    Kanban is the shared cross-profile dispatch bus, so worker stampedes are
    expected.  A long busy timeout lets SQLite serialize writers via WAL rather
    than surfacing transient ``database is locked`` failures during bursts.
    """
    raw = os.environ.get("HERMES_KANBAN_BUSY_TIMEOUT_MS", "").strip()
    if raw:
        try:
            parsed = int(raw)
        except ValueError:
            parsed = 0
        if parsed > 0:
            return parsed
    return DEFAULT_BUSY_TIMEOUT_MS


def _sqlite_connect(path: Path) -> sqlite3.Connection:
    """Open a Kanban SQLite connection with consistent lock waiting.

    Uses ``connect_tracked`` so the live-connection registry knows this file
    is open: while it is, byte-level probes of the same file are refused,
    because an ``open()``/``close()`` would cancel this process's POSIX
    advisory locks on the database (see ``hermes_cli.sqlite_safe_read``).
    The registration is released automatically when the connection closes.
    """
    from hermes_cli.sqlite_safe_read import connect_tracked

    busy_timeout_ms = _resolve_busy_timeout_ms()
    conn = connect_tracked(
        path,
        connect_fn=sqlite3.connect,
        isolation_level=None,
        timeout=busy_timeout_ms / 1000.0,
    )
    # ``sqlite3.connect(timeout=...)`` normally maps to busy_timeout, but set
    # the PRAGMA explicitly so it is observable and survives future wrapper
    # changes. Parameter binding is not supported for PRAGMA assignments.
    conn.execute(f"PRAGMA busy_timeout={busy_timeout_ms}")
    return conn


@contextlib.contextmanager
def _cross_process_init_lock(path: Path):
    """Serialize first-connect WAL/schema/integrity setup across processes.

    ``_INIT_LOCK`` only protects threads inside one Python process. During a
    dispatcher burst, many worker processes can all hit a fresh/legacy board at
    once and each process has an empty ``_INITIALIZED_PATHS`` cache. This file
    lock keeps header validation, integrity probing, WAL activation, and
    additive migrations single-file/single-writer across the whole host while
    leaving normal post-init DB usage concurrent under SQLite WAL.

    The acquire is **bounded** (issue #36644): the original bare blocking
    ``flock(LOCK_EX)`` had no timeout, so a single process stalled inside the
    critical section (or a stale lock held by a wedged worker) blocked every
    other ``connect()`` — including the long-lived gateway dispatcher's
    next-tick connect — forever, with no traceback and no recovery short of a
    restart. We now retry a non-blocking acquire up to a deadline; on timeout
    we log a WARNING and proceed WITHOUT the cross-process lock. That is safe:
    the in-process ``_INIT_LOCK`` still serializes same-process threads, and
    the init work itself is idempotent (``CREATE TABLE IF NOT EXISTS`` +
    additive migrations), so the worst case of two processes racing first-init
    is redundant work, not corruption. A bounded "proceed anyway" beats an
    unbounded hang that silently stops the board.
    """
    def _warn(lock_path, timeout_seconds):
        _log.warning(
            "kanban init lock for %s not acquired within %.0fs — proceeding "
            "without the cross-process lock (in-process lock + idempotent "
            "init are the correctness backstop). A stuck holder is no longer "
            "able to block this connect indefinitely (#36644).",
            lock_path, timeout_seconds,
        )

    with _shared_cross_process_init_lock(
        path,
        timeout_seconds=_INIT_LOCK_TIMEOUT_SECONDS,
        poll_seconds=_INIT_LOCK_POLL_SECONDS,
        on_timeout=_warn,
    ):
        yield


@contextlib.contextmanager
def _dispatch_tick_lock(
    db_path: Path,
    *,
    wait_seconds: float = 0.0,
    fail_open: bool = True,
):
    """Bounded single-writer guard around one dispatcher tick or state change.

    Yields ``True`` when this process holds the board's dispatch lock and
    may proceed with the tick, or ``False`` when another process already
    holds it (the caller should skip the tick this round).

    Motivation (issue #35240): a ``hermes gateway run --replace`` /
    ``gateway restart`` invoked from a shell on a systemd/launchd host can
    leave an orphan gateway whose dispatcher escapes the service cgroup,
    survives ``systemctl restart``, and becomes a *second* long-lived
    writer on the same ``kanban.db``. Two dispatchers that each believe
    they own the file both pass SQLite ``busy_timeout`` and then race on
    WAL frames — the documented root cause of multi-writer corruption.
    The startup guard (``_guard_supervised_gateway_conflict``) blocks the
    common way an orphan is born, but this lock is the defense-in-depth
    that prevents two dispatchers from ever writing concurrently
    *regardless of how the second one got there*.

    Dispatcher calls keep the default non-blocking behavior: the gateway's
    async watcher never stalls on a held lock. Owner pause/resume calls may
    provide a small ``wait_seconds`` bound and ``fail_open=False`` so the
    metadata change is serialized with the final in-flight dispatch claim.

    Board-scoped: the lock file is a ``.dispatch.lock`` sibling of the
    board's ``kanban.db``, so unrelated boards tick independently. On
    platforms without ``fcntl``/``msvcrt`` the guard degrades to a no-op
    (yields ``True``) — single-writer enforcement is best-effort and the
    orphan-dispatcher scenario is specific to POSIX service managers.
    """
    wait_seconds = max(float(wait_seconds or 0.0), 0.0)
    deadline = time.monotonic() + wait_seconds
    lock_path = db_path.with_name(db_path.name + ".dispatch.lock")
    handle = None
    acquired = False
    try:
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        handle = lock_path.open("a+b")
        if _IS_WINDOWS:
            try:
                import msvcrt

                handle.seek(0)
                locking = getattr(msvcrt, "locking")
                # LK_NBLCK = non-blocking exclusive byte-range lock.
                nb_lock = getattr(msvcrt, "LK_NBLCK")
                while True:
                    try:
                        handle.seek(0)
                        locking(handle.fileno(), nb_lock, 1)
                        acquired = True
                        break
                    except OSError:
                        if time.monotonic() >= deadline:
                            break
                        time.sleep(_INIT_LOCK_POLL_SECONDS)
            except AttributeError:
                acquired = False
        else:
            try:
                import fcntl

                while True:
                    try:
                        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                        acquired = True
                        break
                    except (BlockingIOError, OSError):
                        if time.monotonic() >= deadline:
                            break
                        time.sleep(_INIT_LOCK_POLL_SECONDS)
            except ImportError:  # pragma: no cover - non-POSIX fallback
                acquired = fail_open
    except OSError:
        # Dispatch preserves the historical fail-open fallback. An owner
        # state mutation fails closed because returning "paused" without a
        # lock would race a claim and lie to the owner.
        acquired = fail_open
        handle = None
    try:
        yield acquired
    finally:
        if handle is not None:
            try:
                if acquired:
                    if _IS_WINDOWS:
                        import msvcrt

                        handle.seek(0)
                        locking = getattr(msvcrt, "locking")
                        unlock_mode = getattr(msvcrt, "LK_UNLCK")
                        locking(handle.fileno(), unlock_mode, 1)
                    else:
                        import fcntl

                        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            except (OSError, AttributeError):
                pass
            finally:
                handle.close()


# Periodic WAL checkpoint state for the dispatcher tick path. The kanban
# connections run with ``wal_autocheckpoint=100``, but a passive
# autocheckpoint can be starved on a busy multi-process board (any reader
# with an open snapshot blocks the WAL reset), letting the -wal file grow
# between gateway restarts. Once per coarse interval the dispatcher issues
# an explicit ``wal_checkpoint(PASSIVE)``.
#
# PASSIVE, not TRUNCATE (same class fix as the state.db checkpoints,
# #45383/#80255/#44795): the dispatch flock only makes the dispatcher the
# sole *dispatcher* — CLI kanban commands in other processes write to the
# same board without taking that flock, so a TRUNCATE here races live
# writers exactly like the state.db close() path did. PASSIVE never takes
# the exclusive checkpoint lock; the WAL file size is instead bounded by
# ``journal_size_limit`` (set at connection init) which truncates the file
# on the writer's natural post-checkpoint reset.
# Best-effort: a busy/locked checkpoint is logged at DEBUG and retried next
# interval. Keyed per resolved DB path so multi-board dispatchers checkpoint
# each board on its own clock.
_WAL_CHECKPOINT_INTERVAL_SECONDS = 300.0
_LAST_WAL_CHECKPOINT: dict[str, float] = {}
_WAL_CHECKPOINT_LOCK = threading.Lock()


def _maybe_checkpoint_wal(conn: sqlite3.Connection, db_path: Path) -> None:
    """Run ``PRAGMA wal_checkpoint(PASSIVE)`` at a coarse interval.

    Called from the dispatcher tick while the board's dispatch lock is
    held. No-ops (cheaply) until ``_WAL_CHECKPOINT_INTERVAL_SECONDS`` has
    elapsed since this process last checkpointed this board. Never raises:
    the checkpoint is pure hygiene and must not fail a dispatch tick.
    """
    try:
        key = str(db_path.resolve())
    except OSError:
        key = str(db_path)
    now = time.monotonic()
    with _WAL_CHECKPOINT_LOCK:
        last = _LAST_WAL_CHECKPOINT.get(key)
        if last is not None and (now - last) < _WAL_CHECKPOINT_INTERVAL_SECONDS:
            return
        # Claim the slot before doing the work so concurrent ticks (other
        # threads in this process) don't double-checkpoint on the boundary.
        _LAST_WAL_CHECKPOINT[key] = now
    try:
        row = conn.execute("PRAGMA wal_checkpoint(PASSIVE)").fetchone()
        _log.debug(
            "kanban WAL checkpoint (PASSIVE) on %s -> %s "
            "(busy, wal_frames, checkpointed_frames)",
            key, tuple(row) if row is not None else None,
        )
    except sqlite3.Error as exc:
        _log.debug("kanban WAL checkpoint on %s skipped: %s", key, exc)


def _looks_like_tls_record_at(data: bytes, offset: int) -> bool:
    """Return True for a TLS record header at ``data[offset:]``."""
    if len(data) < offset + 5:
        return False
    content_type = data[offset]
    major = data[offset + 1]
    minor = data[offset + 2]
    length = int.from_bytes(data[offset + 3:offset + 5], "big")
    return (
        content_type in {0x14, 0x15, 0x16, 0x17}
        and major == 0x03
        and minor in {0x00, 0x01, 0x02, 0x03, 0x04}
        and 0 < length <= 18432
    )


def _validate_sqlite_header(path: Path) -> None:
    """Fail early with an actionable error for non-SQLite Kanban DB files.

    ``sqlite3.connect()`` creates missing and zero-byte files, so those are
    allowed. Existing non-empty files must have the SQLite header before we
    hand them to SQLite/WAL setup. This keeps corrupted page-0 failures from
    being collapsed into a generic PRAGMA error and lets the gateway's corrupt
    board handling identify the board by fingerprint.
    """
    try:
        stat = path.stat()
    except FileNotFoundError:
        return
    except OSError:
        return
    if stat.st_size == 0:
        return
    # Byte-level probe, so it must run BEFORE any connection to this path
    # exists (connect() calls it under the init lock, ahead of _sqlite_connect).
    # read_header_bytes_preopen refuses once a connection is live, because the
    # close() would cancel this process's POSIX locks on the file.
    from hermes_cli.sqlite_safe_read import read_header_bytes_preopen

    head = read_header_bytes_preopen(path, length=64)
    if head is None:
        return
    if head.startswith(_SQLITE_HEADER):
        return
    signature = ""
    if head.startswith(b"SQLit") and _looks_like_tls_record_at(head, 5):
        signature = " (TLS record header detected at byte offset 5)"
    elif _looks_like_tls_record_at(head, 0):
        signature = " (TLS record header detected at byte offset 0)"
    raise sqlite3.DatabaseError(
        "file is not a database: invalid SQLite header for "
        f"{path}{signature}; first_32={head[:32].hex(' ')}"
    )


class KanbanDbCorruptError(RuntimeError):
    """Raised when an existing kanban DB file fails integrity checks.

    Fail-closed guard against silent recreation of a corrupt board file,
    which would otherwise destroy the user's tasks. Carries both the
    original path and the timestamped backup we made before refusing.
    """

    def __init__(self, db_path: Path, backup_path: Optional[Path], reason: str):
        self.db_path = db_path
        self.backup_path = backup_path
        self.reason = reason
        backup_str = str(backup_path) if backup_path is not None else "<backup failed>"
        super().__init__(
            f"Refusing to open corrupt kanban DB at {db_path}: {reason}. "
            f"Original preserved; backup at {backup_str}."
        )


def _prune_corrupt_backups(
    parent: Path, base_name: str, keep: Optional[Path] = None,
) -> None:
    """Cap the number of retained ``<db>.corrupt.<hash>.bak`` files.

    Content-addressed backups dedupe identical corrupt bytes, but a board
    whose file keeps changing between corruption events (partial repairs,
    ongoing damage, fleets of retrying dispatchers) can still accumulate
    backups without bound — a user reported 124 of them. After creating a
    new backup we keep only the ``_CORRUPT_BACKUP_RETENTION`` most recent
    (by mtime) and delete the rest, including their copied ``-wal``/``-shm``
    sidecars. ``keep`` (the just-created backup) is never pruned regardless
    of its mtime — ``shutil.copy2`` preserves the source file's timestamp,
    which may be older than existing backups. Best-effort: prune failures
    never mask the corruption error the caller is about to raise.
    """
    try:
        backups = [
            candidate
            for candidate in parent.glob(f"{base_name}.corrupt.*.bak")
            if candidate.is_file() and candidate != keep
        ]
    except OSError:
        return
    budget = _CORRUPT_BACKUP_RETENTION - (1 if keep is not None else 0)
    budget = max(budget, 0)
    if len(backups) <= budget:
        return

    def _mtime(item: Path) -> float:
        try:
            return item.stat().st_mtime
        except OSError:
            return 0.0

    backups.sort(key=_mtime, reverse=True)
    for stale in backups[budget:]:
        for victim in (
            stale,
            stale.with_name(stale.name + "-wal"),
            stale.with_name(stale.name + "-shm"),
        ):
            try:
                victim.unlink(missing_ok=True)
            except OSError:
                pass


def _backup_corrupt_db(path: Path) -> Optional[Path]:
    """Copy a corrupt DB (and its WAL/SHM sidecars) to a content-addressed backup.

    The backup filename is deterministic in the main DB's sha256, so repeated
    quarantines of the same corrupt bytes (gateway restarts, dispatcher retries,
    multi-profile fleets all hitting the same shared DB) reuse one backup
    instead of amplifying disk usage by N. If the corrupt bytes actually
    change between attempts — e.g. a partial repair or further damage — the
    fingerprint changes and a separate backup is preserved.

    Returns the backup path of the main DB file, or ``None`` if the copy
    itself failed (the caller still raises loudly in that case).

    Writes are confined to the original DB's parent directory. The backup
    basename is derived purely from ``path.name`` and a content hash, never
    from caller-supplied directory segments — no traversal is possible.
    """
    # Resolve once and pin the parent so subsequent path operations cannot
    # escape it. ``Path.resolve()`` collapses any ``..`` segments and
    # symlinks, and we only ever write inside ``parent``.
    resolved = path.resolve()
    parent = resolved.parent
    base_name = resolved.name  # basename only
    # This reads the whole DB file to fingerprint it. That is a close()-on-a-
    # database-file hazard (it cancels this process's POSIX advisory locks --
    # see hermes_cli.sqlite_safe_read), so it must only run once the board has
    # been taken out of service. Every caller reaches here on the corrupt/
    # quarantine path after closing its probe connection, but another
    # SessionDB/kanban connection elsewhere in the process would still be at
    # risk -- so REFUSE rather than warn-and-proceed. Losing a forensic copy
    # is strictly better than corrupting the live database we are trying to
    # rescue.
    from hermes_cli.sqlite_safe_read import has_live_connection

    if has_live_connection(resolved):
        _log.error(
            "refusing to quarantine %s: a connection to it is still open in "
            "this process, and fingerprinting the file would cancel that "
            "connection's POSIX locks. Close all connections first.",
            resolved,
        )
        return None
    digest = hashlib.sha256()
    try:
        with resolved.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError:
        return None
    token = digest.hexdigest()[:16]
    candidate = parent / f"{base_name}.corrupt.{token}.bak"
    # Defensive: candidate must still be inside parent after construction.
    if candidate.parent != parent:
        return None
    if not candidate.exists():
        try:
            shutil.copy2(resolved, candidate)
        except OSError:
            return None
        # A NEW backup landed on disk — enforce the retention cap so
        # mutating-corruption loops can't accumulate quarantines forever.
        _prune_corrupt_backups(parent, base_name, keep=candidate)
    for suffix in ("-wal", "-shm"):
        sidecar = parent / (base_name + suffix)
        if sidecar.parent != parent or not sidecar.exists():
            continue
        sidecar_backup = parent / (candidate.name + suffix)
        if sidecar_backup.parent != parent or sidecar_backup.exists():
            continue
        try:
            shutil.copy2(sidecar, sidecar_backup)
        except OSError:
            pass
    return candidate


# Repairable integrity_check error classes. Both shapes are *index-scoped*:
# the table b-tree is intact and only a secondary index disagrees with it,
# which REINDEX rebuilds losslessly from the table data. The index name is
# parsed generically from the message — no hardcoded index list. Any other
# integrity_check message (page corruption, "database disk image is
# malformed", freelist damage, …) is NOT repairable this way and keeps the
# fail-closed behavior.
_REPAIRABLE_INDEX_ERROR_PATTERNS = (
    re.compile(r"^wrong # of entries in index (?P<index>.+)$"),
    re.compile(r"^row \d+ missing from index (?P<index>.+)$"),
)


def _integrity_messages_ok(messages: list[str]) -> bool:
    """True iff ``PRAGMA integrity_check`` output is the single ``ok`` row."""
    return len(messages) == 1 and messages[0].strip().lower() == "ok"


def _run_integrity_check(conn: sqlite3.Connection) -> list[str]:
    """Return all ``PRAGMA integrity_check`` message rows as strings."""
    rows = conn.execute("PRAGMA integrity_check").fetchall()
    return [str(row[0]) for row in rows if row is not None and row[0] is not None]


def _repairable_index_names(messages: list[str]) -> Optional[list[str]]:
    """Return the distinct index names iff EVERY message is index-repairable.

    ``None`` when any line falls outside the repairable index-class errors
    (or when there are no messages at all) — the caller must then fail
    closed exactly as before. Order of first appearance is preserved so the
    REINDEX pass is deterministic.
    """
    names: list[str] = []
    saw_any = False
    for raw in messages:
        message = (raw or "").strip()
        if not message:
            continue
        for pattern in _REPAIRABLE_INDEX_ERROR_PATTERNS:
            match = pattern.match(message)
            if match:
                break
        else:
            return None
        saw_any = True
        name = match.group("index").strip()
        if name and name not in names:
            names.append(name)
    if not saw_any or not names:
        return None
    return names


def _attempt_index_reindex_repair(
    path: Path, index_names: list[str],
) -> tuple[bool, list[str]]:
    """REINDEX the named indexes, then re-run ``PRAGMA integrity_check``.

    Tries a per-index ``REINDEX "<name>"`` first (cheapest, most targeted);
    if any per-index statement fails — e.g. the parsed name does not resolve
    because integrity_check reported an internal/auto index — falls back to
    a bare ``REINDEX`` of the whole database. Returns
    ``(clean, post_repair_messages)``; never raises. Callers must hold the
    board's cross-process init flock so no other process connects mid-repair.
    """
    try:
        conn = _sqlite_connect(path)
    except sqlite3.Error as exc:
        return False, [f"could not reopen for REINDEX: {exc}"]
    try:
        try:
            for name in index_names:
                escaped = name.replace('"', '""')
                conn.execute(f'REINDEX "{escaped}"')
        except sqlite3.Error:
            # Per-index rebuild failed (unresolvable parsed name, auto
            # index, …) — bare REINDEX rebuilds every index in the DB.
            conn.execute("REINDEX")
        messages = _run_integrity_check(conn)
    except sqlite3.Error as exc:
        return False, [f"REINDEX failed: {exc}"]
    finally:
        conn.close()
    return _integrity_messages_ok(messages), messages


def _guard_existing_db_is_healthy(path: Path) -> None:
    """Run ``PRAGMA integrity_check`` on an existing non-empty DB file.

    Opens the probe in read/write mode so SQLite can recover or
    checkpoint a healthy WAL/hot-journal DB before we declare it
    corrupt.

    **Narrow auto-repair:** when the integrity failure consists *only* of
    index-scoped errors (``wrong # of entries in index <name>`` / ``row N
    missing from index <name>``), the table b-trees are intact and REINDEX
    rebuilds the damaged indexes losslessly. In that case we take the
    corrupt backup FIRST (same content-addressed quarantine as the
    fail-closed path), run REINDEX under the caller-held init flock,
    re-run ``integrity_check``, and proceed only if it comes back clean.
    Anything else — page corruption, ``malformed`` images, a REINDEX that
    does not produce a clean re-check — fails closed exactly as before:
    copy the file (and any WAL/SHM sidecars) to a backup and raise
    :class:`KanbanDbCorruptError` so callers cannot silently recreate the
    schema on top of a damaged DB.

    Transient lock/busy errors (``sqlite3.OperationalError``) are NOT
    treated as corruption; they propagate raw so the caller sees a
    normal lock failure and no spurious ``.corrupt`` backup is made.

    No-op for missing files, zero-byte files (treated as fresh), and
    paths already proven healthy this process (cache hit).

    Path-trust note: ``path`` arrives via :func:`connect`, which itself
    resolves it from an explicit ``db_path`` argument, the
    :func:`kanban_db_path` env-var chain, or the kanban-home default —
    all sources Hermes treats as user-controlled-but-trusted on the
    user's own machine. We additionally resolve the path here and
    confine all filesystem writes to its parent directory so any
    accidental ``..`` segments are collapsed before any I/O happens.
    """
    # Resolve before any I/O. ``Path.resolve()`` normalizes ``..`` and
    # symlinks, giving us a canonical path whose parent dir we can pin.
    try:
        resolved = path.resolve()
    except OSError:
        return
    try:
        if not resolved.exists() or resolved.stat().st_size == 0:
            return
    except OSError:
        return
    if str(resolved) in _INITIALIZED_PATHS:
        return
    reason: Optional[str] = None
    messages: list[str] = []
    try:
        probe = _sqlite_connect(resolved)
        try:
            messages = _run_integrity_check(probe)
        finally:
            probe.close()
        if not _integrity_messages_ok(messages):
            reason = (
                f"integrity_check returned "
                f"{messages[0] if messages else '<no row>'!r}"
            )
    except sqlite3.OperationalError:
        # Lock contention, busy, transient IO — not corruption. Let it propagate.
        raise
    except sqlite3.DatabaseError as exc:
        reason = f"sqlite refused to open file: {exc}"
    if reason is None:
        return
    # Quarantine FIRST — both the repair path and the fail-closed path
    # preserve the pre-touch bytes before anything mutates the file.
    backup = _backup_corrupt_db(resolved)
    index_names = _repairable_index_names(messages)
    if index_names:
        _log.warning(
            "kanban DB %s failed integrity_check with index-only errors "
            "(%s); pre-repair backup at %s — attempting REINDEX auto-repair.",
            resolved, ", ".join(index_names),
            backup if backup is not None else "<backup failed>",
        )
        repaired, post = _attempt_index_reindex_repair(resolved, index_names)
        if repaired:
            _log.warning(
                "kanban DB %s auto-repaired via REINDEX (%s); "
                "integrity_check now clean. Pre-repair copy kept at %s.",
                resolved, ", ".join(index_names),
                backup if backup is not None else "<backup failed>",
            )
            return
        reason = (
            f"{reason}; REINDEX auto-repair attempted but integrity_check "
            f"still returned {post[0] if post else '<no row>'!r}"
        )
    raise KanbanDbCorruptError(resolved, backup, reason)


@dataclass
class RepairResult:
    """Outcome of :func:`repair_db` for CLI/status reporting.

    ``status`` is one of:

    * ``"ok"``        — integrity_check was already clean; nothing done.
    * ``"repaired"``  — index-only errors found, REINDEX applied, re-check
      clean. ``backup_path`` holds the pre-repair quarantine copy.
    * ``"corrupt"``   — still corrupt: either a non-index error class
      (fail-closed, no repair attempted) or a REINDEX whose re-check did
      not come back clean.
    * ``"missing"``   — no DB file (or zero-byte placeholder); nothing to do.
    """

    status: str
    db_path: Path
    messages: list[str] = field(default_factory=list)
    post_repair_messages: list[str] = field(default_factory=list)
    backup_path: Optional[Path] = None
    reindexed: list[str] = field(default_factory=list)


def repair_db(
    db_path: Optional[Path] = None,
    *,
    board: Optional[str] = None,
) -> RepairResult:
    """Probe a kanban DB and apply the narrow index-REINDEX repair if needed.

    Shares the exact policy of :func:`_guard_existing_db_is_healthy`: only
    integrity failures composed *entirely* of index-scoped errors are
    repairable; the corrupt bytes are quarantined via
    :func:`_backup_corrupt_db` BEFORE any mutation; the REINDEX runs under
    the board's cross-process init flock; and anything else stays corrupt
    (fail-closed) for the caller to surface. Unlike the guard this never
    raises :class:`KanbanDbCorruptError` — it returns a structured
    :class:`RepairResult` so ``hermes kanban repair`` can report and choose
    its own exit code.

    Transient ``sqlite3.OperationalError`` (locked/busy) still propagates
    raw, exactly like the guard: a locked healthy DB is not corruption and
    must not be quarantined.
    """
    if db_path is not None:
        path = db_path
    else:
        path = kanban_db_path(board=board)
    try:
        resolved = path.resolve()
    except OSError:
        resolved = path
    try:
        if not resolved.exists() or resolved.stat().st_size == 0:
            return RepairResult(status="missing", db_path=resolved)
    except OSError:
        return RepairResult(status="missing", db_path=resolved)

    with _cross_process_init_lock(resolved):
        messages: list[str] = []
        try:
            probe = _sqlite_connect(resolved)
            try:
                messages = _run_integrity_check(probe)
            finally:
                probe.close()
        except sqlite3.OperationalError:
            # Locked/busy — not corruption; let the caller report it raw.
            raise
        except sqlite3.DatabaseError as exc:
            # Same quarantine the connect-time guard takes for a file
            # sqlite refuses to open at all (e.g. malformed page 1).
            return RepairResult(
                status="corrupt",
                db_path=resolved,
                messages=[f"sqlite refused to open file: {exc}"],
                backup_path=_backup_corrupt_db(resolved),
            )
        if _integrity_messages_ok(messages):
            return RepairResult(status="ok", db_path=resolved, messages=messages)

        # Quarantine FIRST — identical policy to the connect-time guard.
        backup = _backup_corrupt_db(resolved)
        index_names = _repairable_index_names(messages)
        if not index_names:
            return RepairResult(
                status="corrupt",
                db_path=resolved,
                messages=messages,
                backup_path=backup,
            )
        repaired, post = _attempt_index_reindex_repair(resolved, index_names)
        # The file changed on disk; force the next connect() in this process
        # to re-probe instead of trusting the stale healthy-path cache.
        with _INIT_LOCK:
            _INITIALIZED_PATHS.discard(str(resolved))
        return RepairResult(
            status="repaired" if repaired else "corrupt",
            db_path=resolved,
            messages=messages,
            post_repair_messages=post,
            backup_path=backup,
            reindexed=index_names,
        )


def _schema_is_present(conn: sqlite3.Connection) -> bool:
    """Whether an open connection actually sees the kanban schema.

    ``tasks`` is the sentinel: :data:`SCHEMA_SQL` always creates it, and
    SQLite loses tables all-or-nothing (a file is either the one we
    initialized or a fresh one created by this very open), so one
    ``sqlite_master`` lookup on the already-resident page 1 is enough. Cheap
    by design — it runs on every steady-state :func:`connect`.
    """
    try:
        row = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='tasks' LIMIT 1"
        ).fetchone()
    except sqlite3.DatabaseError:
        # Unreadable schema table is not this guard's call — let the full init
        # path's header/integrity probes classify and quarantine it.
        return False
    return row is not None


def connect(
    db_path: Optional[Path] = None,
    *,
    board: Optional[str] = None,
) -> sqlite3.Connection:
    """Open (and initialize if needed) the kanban DB.

    WAL mode is enabled on every connection; it's a no-op after the first
    time but keeps the code robust if the DB file is ever re-created.

    The first connection to a given path auto-runs :func:`init_db` so
    fresh installs and test harnesses that construct `connect()`
    directly don't have to remember a separate init step. Subsequent
    connections skip the schema check via a module-level path cache.

    Path resolution:

    * ``db_path`` explicit → used as-is (legacy callers, tests).
    * ``board`` explicit → resolves to that board's DB.
    * Neither → :func:`kanban_db_path` resolves via
      ``HERMES_KANBAN_DB`` env → ``HERMES_KANBAN_BOARD`` env →
      ``<root>/kanban/current`` → ``default``.
    """
    if db_path is not None:
        path = db_path
    else:
        path = kanban_db_path(board=board)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Fast path: once THIS process has initialized this path, the expensive
    # first-open work (header validation, integrity probe, schema + additive
    # migrations) is already done and cached in _INITIALIZED_PATHS. Acquiring
    # the cross-process init lock on every connect is what let a single stalled
    # holder (e.g. an external `hermes kanban list` mid-integrity-probe) block
    # the long-lived gateway dispatcher's next-tick connect() forever — an
    # unbounded flock with no timeout, no LOCK_NB, no recovery (#36644). On the
    # steady-state path there is nothing for the cross-process lock to protect
    # (no schema/migration writes run), so skip it entirely and just open the
    # connection with WAL/pragmas under the cheap in-process _INIT_LOCK.
    resolved = str(path.resolve())
    if resolved in _INITIALIZED_PATHS:
        conn = _sqlite_connect(path)
        try:
            conn.row_factory = sqlite3.Row
            with _INIT_LOCK:
                from hermes_state import apply_wal_with_fallback
                apply_wal_with_fallback(conn, db_label=f"kanban.db ({path.name})")
                conn.execute("PRAGMA synchronous=FULL")
                conn.execute("PRAGMA wal_autocheckpoint=100")
                # Bound the WAL file size now that the periodic explicit
                # checkpoint is PASSIVE (never truncates): on the writer's
                # natural post-checkpoint reset SQLite trims the -wal file
                # to this limit. 8 MiB is generous for a kanban board.
                conn.execute("PRAGMA journal_size_limit=8388608")
                conn.execute("PRAGMA foreign_keys=ON")
                conn.execute("PRAGMA secure_delete=ON")
                conn.execute("PRAGMA cell_size_check=ON")
                schema_present = _schema_is_present(conn)
        except Exception:
            conn.close()
            raise
        if schema_present:
            return conn
        # The cache says "initialized", the file says otherwise: it was deleted
        # or replaced under a live process, and the open above silently
        # recreated an empty DB. Left alone, every query on this path fails
        # with "no such table: tasks" for the rest of the process's life and
        # the board just renders empty (#83445). Drop the stale cache entry and
        # fall through to the full init path, which re-runs the header and
        # integrity probes and the schema script under the cross-process lock.
        conn.close()
        with _INIT_LOCK:
            _INITIALIZED_PATHS.discard(resolved)
        _log.warning(
            "kanban DB %s lost its schema after this process initialized it "
            "(deleted or replaced externally); re-initializing.",
            path,
        )

    with _cross_process_init_lock(path):
        # Read-only file/sidecar preflight (port of kilocode#12508) —
        # repair-or-refuse before the header/integrity probes so a stray
        # read-only kanban.db fails with an actionable message instead of
        # "attempt to write a readonly database" mid-init.
        from hermes_state import preflight_db_writability
        preflight_db_writability(path, db_label=f"kanban.db ({path.name})")
        # Cheap byte-level check first — catches the #29507 TLS-overwrite shape
        # and other invalid-header cases without opening a sqlite connection.
        _validate_sqlite_header(path)
        # Full integrity probe — catches corruption past the header (malformed
        # pages, broken internal metadata). Cached per-path after first success
        # via _INITIALIZED_PATHS so it only runs once per process per path.
        _guard_existing_db_is_healthy(path)
        resolved = str(path.resolve())
        conn = _sqlite_connect(path)
        try:
            conn.row_factory = sqlite3.Row
            with _INIT_LOCK:
                # WAL activation can take an exclusive lock while SQLite creates the
                # sidecar files for a fresh database. Keep it in the same process-local
                # critical section as schema initialization so concurrent gateway
                # startup threads do not race before _INITIALIZED_PATHS is populated.
                # WAL doesn't work on network filesystems (NFS/SMB/FUSE). Shared helper
                # falls back to DELETE with one ERROR log so kanban stays usable there.
                # See hermes_state._WAL_INCOMPAT_MARKERS for detection logic.
                from hermes_state import apply_wal_with_fallback
                apply_wal_with_fallback(conn, db_label=f"kanban.db ({path.name})")
                # FULL (was NORMAL): fsync before each checkpoint to narrow the
                # crash window that can leave a b-tree page header torn.
                conn.execute("PRAGMA synchronous=FULL")
                conn.execute("PRAGMA wal_autocheckpoint=100")
                # Bound the WAL file size now that the periodic explicit
                # checkpoint is PASSIVE (never truncates): on the writer's
                # natural post-checkpoint reset SQLite trims the -wal file
                # to this limit. 8 MiB is generous for a kanban board.
                conn.execute("PRAGMA journal_size_limit=8388608")
                conn.execute("PRAGMA foreign_keys=ON")
                # Zero freed pages so a later torn write cannot expose stale
                # cell content; persisted in the DB header for new DBs.
                conn.execute("PRAGMA secure_delete=ON")
                # Surface corrupt cells as read errors instead of silent
                # wrong-data returns.
                conn.execute("PRAGMA cell_size_check=ON")
                needs_init = resolved not in _INITIALIZED_PATHS
                if needs_init:
                    # Idempotent: runs CREATE TABLE IF NOT EXISTS + the additive
                    # migrations. Cached so subsequent connect() calls in the same
                    # process are cheap. The lock prevents same-process dispatcher
                    # threads from racing through the additive ALTER TABLE pass with
                    # stale PRAGMA snapshots during gateway startup.
                    conn.executescript(SCHEMA_SQL)
                    # Which board this file IS, so the migration can read its
                    # published owner metadata. An explicit ``db_path`` with no
                    # board named resolves to None, which simply skips that
                    # extra proof rather than guessing a slug.
                    _migrate_add_optional_columns(
                        conn,
                        _normalize_board_slug(board)
                        or (get_current_board() if db_path is None else None),
                    )
                    _INITIALIZED_PATHS.add(resolved)
        except Exception:
            conn.close()
            raise
    return conn


@contextlib.contextmanager
def connect_closing(
    db_path: Optional[Path] = None,
    *,
    board: Optional[str] = None,
):
    """Open a kanban DB connection and guarantee it is closed on exit.

    Use this instead of ``with kb.connect() as conn:`` — sqlite3's
    built-in connection context manager only commits/rollbacks the
    transaction; it does NOT close the file descriptor. In long-lived
    processes (gateway, dashboard) that route every kanban operation
    through ``connect()`` (e.g. ``run_slash`` dispatching ``/kanban …``
    commands, ``decompose_task_endpoint`` calling
    ``kanban_decompose.decompose_task``), the unclosed connections
    accumulate as open FDs to ``kanban.db`` and ``kanban.db-wal``. After
    enough operations the process hits the kernel FD limit and dies
    with ``[Errno 24] Too many open files``.

    See #33159 for the production incident.

    The ``connect()`` function itself remains unchanged so callers that
    intentionally manage the connection lifetime (tests, long-lived
    callers) continue to work.
    """
    conn = connect(db_path=db_path, board=board)
    try:
        yield conn
    finally:
        try:
            conn.close()
        except Exception:
            pass


def init_db(
    db_path: Optional[Path] = None,
    *,
    board: Optional[str] = None,
) -> Path:
    """Create the schema if it doesn't exist; return the path used.

    Kept as a public entry point so CLI ``hermes kanban init`` and the
    daemon have something explicit to call. Unlike :func:`connect`'s
    first-time auto-init (which caches by path), ``init_db`` always
    re-runs the migration pass. Callers that know the on-disk schema
    may have drifted — tests that write legacy event kinds directly,
    external tools that upgrade an old DB file — can call this to
    force re-migration.
    """
    if db_path is not None:
        path = db_path
    else:
        path = kanban_db_path(board=board)
    path.parent.mkdir(parents=True, exist_ok=True)
    resolved = str(path.resolve())
    # Clear the cache entry so the underlying connect() re-runs the
    # schema + migration pass unconditionally.
    with _INIT_LOCK:
        _INITIALIZED_PATHS.discard(resolved)
    # Carry the board through: the migration reads this board's published
    # owner metadata to recover receipt ownership on a pre-upgrade board.
    with contextlib.closing(connect(path, board=board)):
        pass
    return path


def _migrate_add_optional_columns(
    conn: sqlite3.Connection, board_slug: Optional[str] = None,
) -> None:
    """Add columns that were introduced after v1 release to legacy DBs.

    Called by ``init_db`` so opening an old DB is always safe. ``board_slug``
    lets the receipt-ownership reconciliation below read THIS board's published
    owner metadata; omitted, it falls back to the control-anchor proof alone.
    """
    cols = {row["name"] for row in conn.execute("PRAGMA table_info(tasks)")}
    if "tenant" not in cols:
        _add_column_if_missing(conn, "tasks", "tenant", "tenant TEXT")
    if "result" not in cols:
        _add_column_if_missing(conn, "tasks", "result", "result TEXT")
    if "branch_name" not in cols:
        _add_column_if_missing(conn, "tasks", "branch_name", "branch_name TEXT")
    if "project_id" not in cols:
        _add_column_if_missing(conn, "tasks", "project_id", "project_id TEXT")
    if "owned_paths" not in cols:
        _add_column_if_missing(conn, "tasks", "owned_paths", "owned_paths TEXT")
    if "integrates_parent_heads" not in cols:
        _add_column_if_missing(
            conn,
            "tasks",
            "integrates_parent_heads",
            "integrates_parent_heads INTEGER NOT NULL DEFAULT 0",
        )
    if "base_commit" not in cols:
        _add_column_if_missing(conn, "tasks", "base_commit", "base_commit TEXT")
    if "head_commit" not in cols:
        _add_column_if_missing(conn, "tasks", "head_commit", "head_commit TEXT")
    if "idempotency_key" not in cols:
        _add_column_if_missing(
            conn, "tasks", "idempotency_key", "idempotency_key TEXT"
        )
    # ``idx_tasks_idempotency`` is created unconditionally below alongside
    # the other additive-column indexes — see the block after the
    # legacy-column migration. Creating it here too would be redundant.

    # Refresh after early additive migrations above. Some existing DBs were
    # partially migrated in older releases and can already contain the later
    # columns (for example ``consecutive_failures``) even when this function's
    # initial snapshot did not. Re-snapshot here so the legacy-column migration
    # below is truly idempotent and never re-adds columns that already exist.
    cols = {row["name"] for row in conn.execute("PRAGMA table_info(tasks)")}

    # Legacy column migration: ``spawn_failures`` → ``consecutive_failures``
    # and ``last_spawn_error`` → ``last_failure_error``.
    #
    # Avoid ``ALTER TABLE ... RENAME COLUMN`` for two reasons:
    #   1. Primary: very old DBs may never have had ``spawn_failures`` at
    #      all, so RENAME raises OperationalError: no such column (the crash
    #      reported in issue #20842 after the #20410 update).
    #   2. Secondary: SQLite reparses the whole schema on any RENAME, which
    #      fails if related objects (views, triggers) reference the old name.
    #
    # ADD-first-then-copy is tolerant of both shapes and preserves
    # historical counter values when the legacy columns do exist.
    if "consecutive_failures" not in cols:
        added = _add_column_if_missing(
            conn,
            "tasks",
            "consecutive_failures",
            "consecutive_failures INTEGER NOT NULL DEFAULT 0",
        )
        if added and "spawn_failures" in cols:
            conn.execute(
                "UPDATE tasks SET consecutive_failures = COALESCE(spawn_failures, 0)"
            )
    if "worker_pid" not in cols:
        _add_column_if_missing(conn, "tasks", "worker_pid", "worker_pid INTEGER")
    if "last_failure_error" not in cols:
        added = _add_column_if_missing(
            conn, "tasks", "last_failure_error", "last_failure_error TEXT"
        )
        if added and "last_spawn_error" in cols:
            conn.execute(
                "UPDATE tasks SET last_failure_error = last_spawn_error"
            )
    if "max_runtime_seconds" not in cols:
        _add_column_if_missing(
            conn, "tasks", "max_runtime_seconds", "max_runtime_seconds INTEGER"
        )
    if "last_heartbeat_at" not in cols:
        _add_column_if_missing(
            conn, "tasks", "last_heartbeat_at", "last_heartbeat_at INTEGER"
        )
    if "current_run_id" not in cols:
        _add_column_if_missing(
            conn, "tasks", "current_run_id", "current_run_id INTEGER"
        )
    if "workflow_template_id" not in cols:
        _add_column_if_missing(
            conn, "tasks", "workflow_template_id", "workflow_template_id TEXT"
        )
    if "current_step_key" not in cols:
        _add_column_if_missing(
            conn, "tasks", "current_step_key", "current_step_key TEXT"
        )
    if "responsibility" not in cols:
        _add_column_if_missing(
            conn, "tasks", "responsibility", "responsibility TEXT"
        )
    if "skills" not in cols:
        # JSON array of skill names the dispatcher force-loads into the
        # worker via --skills. NULL is fine for existing rows.
        _add_column_if_missing(conn, "tasks", "skills", "skills TEXT")

    if "max_retries" not in cols:
        # Per-task override for the consecutive-failure circuit breaker.
        # NULL = fall through to the dispatcher-level ``kanban.failure_limit``
        # config, then ``DEFAULT_FAILURE_LIMIT``. Existing rows get NULL,
        # which is the correct default (they keep the global behaviour
        # they were getting before the column existed).
        _add_column_if_missing(conn, "tasks", "max_retries", "max_retries INTEGER")

    if "model_override" not in cols:
        conn.execute("ALTER TABLE tasks ADD COLUMN model_override TEXT")

    if "provider_override" not in cols:
        # Provider the model_override belongs to. NULL = worker profile's
        # provider resolves the model (the behaviour existing rows had).
        _add_column_if_missing(
            conn, "tasks", "provider_override", "provider_override TEXT"
        )

    if "reasoning_effort" not in cols:
        # Per-task thinking depth for the worker. NULL = the worker profile's
        # own agent.reasoning_effort, which is what existing rows were getting.
        _add_column_if_missing(
            conn, "tasks", "reasoning_effort", "reasoning_effort TEXT"
        )

    if "execution_tier" not in cols:
        # Owner-approved semantic task class. NULL on every existing row: those
        # tasks were never classified, so no route can be re-derived for them.
        _add_column_if_missing(
            conn, "tasks", "execution_tier", "execution_tier TEXT"
        )

    if "model_policy_lock" not in cols:
        # Owner-approved model-route lock. NULL on every existing row, which
        # is the correct default: those tasks keep the mutable, profile-driven
        # routing they already had.
        _add_column_if_missing(
            conn, "tasks", "model_policy_lock", "model_policy_lock TEXT"
        )

    if "owner_receipt_bound" not in cols:
        # Durable receipt ownership. 0 for every pre-existing row; the
        # reconciliation below re-derives it from the board's own control
        # anchors, so legacy owner work created before this column existed
        # still becomes governed instead of dispatching as ordinary work.
        _add_column_if_missing(
            conn,
            "tasks",
            "owner_receipt_bound",
            "owner_receipt_bound INTEGER NOT NULL DEFAULT 0",
        )

    if "park_generation" not in cols:
        # Which parking operation owns a row sitting in ``scheduled``. NULL on
        # every existing row, which is correct: nothing already there was
        # parked by a receipt this build can identify, so no activation may
        # claim it.
        _add_column_if_missing(
            conn, "tasks", "park_generation", "park_generation TEXT"
        )

    if "goal_mode" not in cols:
        # Ralph-style goal loop toggle for the dispatched worker. 0 (the
        # default) = classic single-shot worker, preserving the behaviour
        # existing rows had before the column existed.
        _add_column_if_missing(
            conn, "tasks", "goal_mode", "goal_mode INTEGER NOT NULL DEFAULT 0"
        )

    if "goal_max_turns" not in cols:
        # Per-task goal-loop turn budget. NULL = goals-engine default.
        _add_column_if_missing(
            conn, "tasks", "goal_max_turns", "goal_max_turns INTEGER"
        )

    if "session_id" not in cols:
        # Originating agent/chat session id, populated when the task is
        # created from within an agent loop that propagated
        # ``HERMES_SESSION_ID`` (e.g. ACP). NULL on legacy rows and on any
        # creation path that doesn't set the env var (CLI, dashboard).
        _add_column_if_missing(
            conn, "tasks", "session_id", "session_id TEXT"
        )

    if "block_kind" not in cols:
        # Typed block reason (VALID_BLOCK_KINDS) or NULL for legacy/un-typed
        # blocks. Existing blocked rows get NULL, which is treated as a
        # generic human blocker — same behaviour they had before the column.
        _add_column_if_missing(conn, "tasks", "block_kind", "block_kind TEXT")

    if "block_recurrences" not in cols:
        # Unblock-loop counter. Existing rows start at 0, so the loop breaker
        # only begins counting from the first re-block after this migration.
        _add_column_if_missing(
            conn,
            "tasks",
            "block_recurrences",
            "block_recurrences INTEGER NOT NULL DEFAULT 0",
        )

    if "task_kind" not in cols:
        _add_column_if_missing(
            conn, "tasks", "task_kind", "task_kind TEXT NOT NULL DEFAULT 'work'"
        )
    if "recommendation_kind" not in cols:
        _add_column_if_missing(
            conn, "tasks", "recommendation_kind", "recommendation_kind TEXT"
        )
    if "recommendation_subject_id" not in cols:
        _add_column_if_missing(
            conn,
            "tasks",
            "recommendation_subject_id",
            "recommendation_subject_id TEXT",
        )
    if "recommendation_label" not in cols:
        _add_column_if_missing(
            conn, "tasks", "recommendation_label", "recommendation_label TEXT"
        )
    if "recommendation_rationale" not in cols:
        _add_column_if_missing(
            conn,
            "tasks",
            "recommendation_rationale",
            "recommendation_rationale TEXT",
        )
    if "target_profile" not in cols:
        _add_column_if_missing(conn, "tasks", "target_profile", "target_profile TEXT")
    if "review_policy" not in cols:
        _add_column_if_missing(conn, "tasks", "review_policy", "review_policy TEXT")
    if "provenance_authority" not in cols:
        _add_column_if_missing(
            conn, "tasks", "provenance_authority", "provenance_authority TEXT"
        )
    if "provenance_ref" not in cols:
        _add_column_if_missing(conn, "tasks", "provenance_ref", "provenance_ref TEXT")
    if "provenance_observed_at" not in cols:
        _add_column_if_missing(
            conn,
            "tasks",
            "provenance_observed_at",
            "provenance_observed_at INTEGER",
        )
    if "recommendation_evidence" not in cols:
        _add_column_if_missing(
            conn, "tasks", "recommendation_evidence", "recommendation_evidence TEXT"
        )
    if "recommendation_decision" not in cols:
        _add_column_if_missing(
            conn, "tasks", "recommendation_decision", "recommendation_decision TEXT"
        )
    if "recommendation_effective_state" not in cols:
        _add_column_if_missing(
            conn, "tasks", "recommendation_effective_state", "recommendation_effective_state TEXT"
        )
    if "recommendation_lifecycle_version" not in cols:
        _add_column_if_missing(
            conn,
            "tasks",
            "recommendation_lifecycle_version",
            "recommendation_lifecycle_version INTEGER",
        )

    # Indexes over additive ``tasks`` columns must be created after the
    # columns exist. Keeping them in SCHEMA_SQL breaks legacy boards: SQLite
    # parses each statement in ``executescript`` against the live schema, so a
    # ``CREATE INDEX`` over a missing column aborts initialization before the
    # additive ``ALTER TABLE`` migrations below can run. Re-running them here
    # is cheap thanks to ``IF NOT EXISTS`` and stays correct on fresh DBs
    # (where the columns already exist from SCHEMA_SQL).
    conn.execute("CREATE INDEX IF NOT EXISTS idx_tasks_tenant ON tasks(tenant)")
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_tasks_idempotency ON tasks(idempotency_key)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_tasks_session_id ON tasks(session_id)"
    )
    if {"project_id", "status"}.issubset(cols):
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_tasks_project_status "
            "ON tasks(project_id, status)"
        )
    # Matches the recommendation GET's exact equality-filter + order shape.
    if {"status", "created_at", "id"}.issubset(cols):
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_tasks_recommendation_scope "
            "ON tasks(task_kind, review_policy, status, project_id, "
            "target_profile, created_at, id)"
        )

    # task_events gained a run_id column; back-fill it as NULL for
    # historical events (they predate runs and can't be attributed).
    ev_cols = {row["name"] for row in conn.execute("PRAGMA table_info(task_events)")}
    if "run_id" not in ev_cols:
        _add_column_if_missing(conn, "task_events", "run_id", "run_id INTEGER")

    # Durable operation identity for idempotent comment replay (see
    # add_comment()'s operation_key parameter). Partial unique index: NULL
    # values (ordinary comments) are exempt, only two rows sharing the same
    # (task_id, operation_key) collide — which is exactly what dedup wants.
    # Guarded by a table-existence check (same pattern as kanban_notify_subs
    # below) so callers that migrate a minimal/partial schema (e.g. a legacy
    # DB with only `tasks` + `task_events`) don't hit "no such table".
    comments_table_exists = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='task_comments'"
    ).fetchone() is not None
    if comments_table_exists:
        comment_cols = {row["name"] for row in conn.execute("PRAGMA table_info(task_comments)")}
        if "operation_key" not in comment_cols:
            _add_column_if_missing(conn, "task_comments", "operation_key", "operation_key TEXT")
        conn.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_comments_operation_key "
            "ON task_comments(task_id, operation_key) WHERE operation_key IS NOT NULL"
        )

    # Same ordering rule as the additive ``tasks`` indexes above: create the
    # index after the additive column migration so legacy ``task_events``
    # tables don't fail during SCHEMA_SQL execution before ``run_id`` exists.
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_events_run "
        "ON task_events(run_id, id)"
    )

    notify_table_exists = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='kanban_notify_subs'"
    ).fetchone() is not None
    if notify_table_exists:
        notify_cols = {
            row["name"] for row in conn.execute("PRAGMA table_info(kanban_notify_subs)")
        }
        if "notifier_profile" not in notify_cols:
            _add_column_if_missing(
                conn, "kanban_notify_subs", "notifier_profile", "notifier_profile TEXT"
            )
        if "delivery_mode" not in notify_cols:
            _add_column_if_missing(
                conn,
                "kanban_notify_subs",
                "delivery_mode",
                "delivery_mode TEXT NOT NULL DEFAULT 'notify'",
            )
            # Backfill: before this column existed, the notifier woke the
            # originating session unconditionally whenever the task carried a
            # session_id — every pre-existing gateway subscription had de
            # facto active wake. Defaulting them to plain 'notify' would
            # silently disable that behavior on upgrade. TUI/CLI rows keep
            # 'notify' (matching _maybe_auto_subscribe, which only requests
            # 'notify+wake' for gateway sessions). Runs ONLY on first-add of
            # the column, so a user's later explicit downgrade is never
            # overwritten by a re-migration.
            conn.execute(
                "UPDATE kanban_notify_subs SET delivery_mode = 'notify+wake' "
                "WHERE platform != 'tui'"
            )
        if "chat_type" not in notify_cols:
            _add_column_if_missing(
                conn,
                "kanban_notify_subs",
                "chat_type",
                "chat_type TEXT",
            )
        if "user_id_alt" not in notify_cols:
            # Records the originating source's platform-specific stable alt ID
            # (Signal UUID, Feishu union_id, ...) alongside ``user_id`` so an
            # active-wake replay reconstructs the SAME ``build_session_key`` as
            # the original event. ``build_session_key`` prefers ``user_id_alt``
            # over ``user_id`` when both are present (gateway/session.py); a
            # wake that only replayed ``user_id`` would key to a different,
            # context-less session whenever the two diverge. Legacy rows
            # default to NULL, which is inert: ``user_id_alt or user_id`` falls
            # back to the already-persisted ``user_id``.
            _add_column_if_missing(
                conn, "kanban_notify_subs", "user_id_alt", "user_id_alt TEXT"
            )
        if "delivery_metadata" not in notify_cols:
            _add_column_if_missing(
                conn, "kanban_notify_subs", "delivery_metadata", "delivery_metadata TEXT"
            )

    # One-shot backfill: any task that is 'running' before runs existed
    # had its claim_lock / claim_expires / worker_pid on the task row.
    # Synthesize a matching task_runs row so subsequent end-run / heartbeat
    # calls have something to write to. Wrapped in write_txn to serialize
    # against any concurrent dispatcher, and the per-row UPDATE uses
    # ``current_run_id IS NULL`` as a CAS guard so a racing claim can't
    # produce an orphaned row if it interleaves with the backfill pass.
    runs_exist = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='task_runs'"
    ).fetchone() is not None
    if runs_exist:
        with write_txn(conn):
            inflight = conn.execute(
                "SELECT id, assignee, claim_lock, claim_expires, worker_pid, "
                "       max_runtime_seconds, last_heartbeat_at, started_at "
                "FROM tasks "
                "WHERE status = 'running' AND current_run_id IS NULL"
            ).fetchall()
            for row in inflight:
                started = row["started_at"] or int(time.time())
                cur = conn.execute(
                    """
                    INSERT INTO task_runs (
                        task_id, profile, status,
                        claim_lock, claim_expires, worker_pid,
                        max_runtime_seconds, last_heartbeat_at,
                        started_at
                    ) VALUES (?, ?, 'running', ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        row["id"], row["assignee"], row["claim_lock"],
                        row["claim_expires"], row["worker_pid"],
                        row["max_runtime_seconds"], row["last_heartbeat_at"],
                        started,
                    ),
                )
                # CAS: only install the pointer if nothing else claimed
                # the task between our SELECT and here (shouldn't happen
                # under the write_txn, but belt-and-suspenders). If the
                # CAS fails we've got an orphan run_row — mark it
                # reclaimed so it doesn't look in-flight.
                upd = conn.execute(
                    "UPDATE tasks SET current_run_id = ? "
                    "WHERE id = ? AND current_run_id IS NULL",
                    (cur.lastrowid, row["id"]),
                )
                if upd.rowcount != 1:
                    conn.execute(
                        "UPDATE task_runs SET status = 'reclaimed', "
                        "    outcome = 'reclaimed', ended_at = ? "
                        "WHERE id = ?",
                        (int(time.time()), cur.lastrowid),
                    )

    # One-shot event-kind rename pass. The old names ("ready", "priority",
    # "spawn_auto_blocked") still worked but were awkward on the wire;
    # rename them in-place so existing DBs migrate cleanly. Fires once
    # per DB because after the UPDATE no rows match the old kinds.
    _EVENT_RENAMES = (
        # (old, new)
        ("ready",              "promoted"),
        ("priority",           "reprioritized"),
        ("spawn_auto_blocked", "gave_up"),
    )
    for old, new in _EVENT_RENAMES:
        conn.execute(
            "UPDATE task_events SET kind = ? WHERE kind = ?",
            (new, old),
        )

    _rebuild_drifted_tables(conn)
    _reconcile_receipt_owned_tasks(conn, board_slug)


def _board_owner_project_id(board_slug: Optional[str]) -> Optional[str]:
    """The Project id this board itself publishes as its owner, or None.

    ``board.json``'s ``project_id`` is written only by the owner kernel (see
    ``owner_workspace._assert_board_ownership``), and it has been written that
    way since before this build added ``task_kind='control'``. It is therefore
    the one piece of durable, board-local, owner-exclusive evidence available on
    a board created by an OLDER build, whose Project anchor is still an ordinary
    work row.
    """
    if not board_slug:
        return None
    try:
        metadata = read_board_metadata(board_slug)
    except Exception:  # pragma: no cover - read_board_metadata never raises
        return None
    if not board_ownership_verified(metadata):
        # Unverifiable ownership binds nothing HERE — a guess would either
        # claim another Project's work or leave real owner work unbound. The
        # fail-closed half of this is in ``board_dispatch_allowed``, which
        # refuses to dispatch such a board at all.
        _log.warning(
            "kanban board %s: board.json could not be read; its published "
            "Project ownership cannot be verified and dispatch stays closed",
            board_slug,
        )
        return None
    published = metadata.get("project_id")
    if not isinstance(published, str):
        return None
    return published.strip() or None


def _reconcile_receipt_owned_tasks(
    conn: sqlite3.Connection, board_slug: Optional[str] = None,
) -> None:
    """Bind receipt ownership onto legacy owner work, then park what it cannot prove.

    The owner kernel stamps ``owner_receipt_bound`` on every task it creates,
    but a board upgraded from an older build carries owner work with the
    column defaulted to 0, NULL ``execution_tier`` and NULL
    ``model_policy_lock`` — indistinguishable, to every dispatch path, from an
    ordinary manual card. Two durable proofs of owner ownership are recovered
    here, because neither covers the other's boards:

    * the Project's non-executable control anchor — only the owner kernel
      creates ``task_kind='control'`` rows, and it creates exactly one per
      Project, so every work row sharing a Project with an anchor is
      receipt-owned;
    * the board's own published ``project_id`` (:func:`_board_owner_project_id`)
      — a board created BEFORE this build has no control row at all (its anchor
      is a work row), so the anchor test alone would bind nothing there and its
      owner work would keep dispatching as ordinary work.

    Either way the fact is persisted here rather than re-derived on each read.

    Anything the binding catches that is ALREADY sitting in an executable
    column is then re-proved through the ordinary readiness guard, which mints
    the exact admitted lock for a row genuinely on an approved route and parks
    the rest for re-approval. Upgrading a board therefore cannot leave
    receipt-owned work runnable on a route nobody approved.

    Binding and reconciliation are ONE transaction, and the reconciliation
    scan runs on every migration pass rather than only when this pass bound a
    row. Committing the binding first and then skipping the scan whenever a
    later startup bound nothing left work stranded for good: a crash in that
    gap produced rows that are receipt-bound, unlocked and already sitting in
    an executable column, which every claim path refuses and no later pass
    ever looked at again.

    Idempotent, and a no-op (two cheap indexed reads) on every board that has
    no owner Project at all.

    ``allow_nested`` because this runs at the tail of the additive migration
    pass, which its callers legitimately drive from inside their own open
    transaction; the savepoint keeps the binding and the reconciliation
    inseparable either way.
    """
    columns = {
        str(row[1]) for row in conn.execute("PRAGMA table_info(tasks)")
    }
    if not {
        "status", "task_kind", "project_id", "owner_receipt_bound",
        "model_policy_lock",
    } <= columns:
        # A tasks table that cannot express a status, a kind or receipt
        # ownership cannot hold owner work in an executable column, so there
        # is provably nothing to bind or re-prove. This is genuine absence,
        # not a skipped scan.
        return
    with write_txn(conn, allow_nested=True):
        conn.execute(
            "UPDATE tasks SET owner_receipt_bound = 1 "
            "WHERE task_kind = 'work' AND owner_receipt_bound = 0 "
            "AND project_id IS NOT NULL AND (project_id IN ("
            "  SELECT project_id FROM tasks "
            "  WHERE task_kind = 'control' AND project_id IS NOT NULL) "
            "OR project_id = ?)",
            (_board_owner_project_id(board_slug),),
        )
        unproven = [
            str(row["id"])
            for row in conn.execute(
                "SELECT id FROM tasks WHERE task_kind = 'work' "
                "AND owner_receipt_bound = 1 AND model_policy_lock IS NULL "
                f"AND status IN ({', '.join('?' * len(EXECUTABLE_STATUSES))})",
                tuple(sorted(EXECUTABLE_STATUSES)),
            ).fetchall()
        ]
        for task_id in unproven:
            authorize_executable_transition(conn, task_id)


# Legacy DBs defined these tables with a ``TEXT PRIMARY KEY`` id (or, for
# ``kanban_notify_subs``, a nullable ``TEXT last_event_id``). The current
# schema uses ``INTEGER PRIMARY KEY AUTOINCREMENT`` / ``INTEGER NOT NULL
# DEFAULT 0``. ``CREATE TABLE IF NOT EXISTS`` skips existing tables
# regardless of schema and ``_add_column_if_missing`` only adds columns, so
# neither can fix a drifted column type — the table must be rebuilt. See
# #35096.
#
# Each entry pairs the canonical CREATE TABLE with the CREATE INDEX
# statements that DROP TABLE would otherwise take down with it (including
# ``idx_events_run``, added by the additive pass above). To guard against
# this list drifting from SCHEMA_SQL, ``test_rebuilt_schema_matches_fresh``
# asserts a rebuilt legacy DB is byte-identical to a fresh one.
_REBUILD_SPECS = {
    "task_events": (
        "CREATE TABLE task_events ("
        " id INTEGER PRIMARY KEY AUTOINCREMENT,"
        " task_id TEXT NOT NULL, run_id INTEGER, kind TEXT NOT NULL,"
        " payload TEXT, created_at INTEGER NOT NULL)",
        (
            "CREATE INDEX idx_events_task ON task_events(task_id, created_at)",
            "CREATE INDEX idx_events_run ON task_events(run_id, id)",
        ),
    ),
    "task_comments": (
        "CREATE TABLE task_comments ("
        " id INTEGER PRIMARY KEY AUTOINCREMENT,"
        " task_id TEXT NOT NULL, author TEXT NOT NULL, body TEXT NOT NULL,"
        " created_at INTEGER NOT NULL)",
        ("CREATE INDEX idx_comments_task ON task_comments(task_id, created_at)",),
    ),
    "task_runs": (
        "CREATE TABLE task_runs ("
        " id INTEGER PRIMARY KEY AUTOINCREMENT,"
        " task_id TEXT NOT NULL, profile TEXT, step_key TEXT,"
        " status TEXT NOT NULL, claim_lock TEXT, claim_expires INTEGER,"
        " worker_pid INTEGER, max_runtime_seconds INTEGER,"
        " last_heartbeat_at INTEGER, started_at INTEGER NOT NULL,"
        " ended_at INTEGER, outcome TEXT, summary TEXT, metadata TEXT,"
        " error TEXT)",
        (
            "CREATE INDEX idx_runs_task ON task_runs(task_id, started_at)",
            "CREATE INDEX idx_runs_status ON task_runs(status)",
        ),
    ),
    "kanban_notify_subs": (
        "CREATE TABLE kanban_notify_subs ("
        " task_id TEXT NOT NULL, platform TEXT NOT NULL, chat_id TEXT NOT NULL,"
        " thread_id TEXT NOT NULL DEFAULT '', user_id TEXT, user_id_alt TEXT,"
        " chat_type TEXT,"
        " notifier_profile TEXT, delivery_mode TEXT NOT NULL DEFAULT 'notify',"
        " delivery_metadata TEXT, created_at INTEGER NOT NULL,"
        " last_event_id INTEGER NOT NULL DEFAULT 0,"
        " PRIMARY KEY (task_id, platform, chat_id, thread_id))",
        ("CREATE INDEX idx_notify_task ON kanban_notify_subs(task_id)",),
    ),
}


def _table_has_drifted(conn: sqlite3.Connection, table: str) -> bool:
    """True when ``table`` still carries the legacy (pre-AUTOINCREMENT) shape."""
    info = conn.execute(f"PRAGMA table_info({table})").fetchall()
    if not info:
        return False  # table absent — nothing to rebuild
    if table == "kanban_notify_subs":
        lei = next((c for c in info if c["name"] == "last_event_id"), None)
        return lei is not None and (lei["type"] or "").upper() != "INTEGER"
    # task_events / task_comments / task_runs: id must be INTEGER and a PK.
    id_col = next((c for c in info if c["name"] == "id"), None)
    if id_col is None:
        return False
    return not ((id_col["type"] or "").upper() == "INTEGER" and id_col["pk"])


def _rebuild_drifted_tables(conn: sqlite3.Connection) -> None:
    """Rebuild any kanban table whose column types drifted from SCHEMA_SQL.

    Old boards crash the gateway notifier (``int(None)`` on a NULL id in
    ``unseen_events_for_sub``) and never match the ``id > cursor`` filter, so
    every kanban notification is silently lost (#35096). Each affected table is
    rebuilt with the standard SQLite pattern — CREATE new → INSERT shared
    columns → DROP old → RENAME — recreating its indexes too (DROP TABLE takes
    them down). The legacy TEXT ids are dropped (they aren't valid integers);
    AUTOINCREMENT assigns fresh ones and ``last_event_id`` cursors reset to 0,
    so the first post-migration tick replays a task's event history once —
    the safe failure mode for a feature that was already fully broken.

    The whole pass runs in one transaction so an interruption can't leave a
    table half-renamed, and under ``connect()``'s init locks so nothing races
    it. Idempotent: a correctly-typed DB skips every table and returns without
    opening a transaction.
    """
    drifted = [t for t in _REBUILD_SPECS if _table_has_drifted(conn, t)]
    if not drifted:
        return

    conn.execute("BEGIN IMMEDIATE")
    try:
        for table in drifted:
            create_sql, index_sqls = _REBUILD_SPECS[table]
            old_cols = [c["name"] for c in conn.execute(f"PRAGMA table_info({table})")]
            _log.info("kanban migration: rebuilding %s to match current schema", table)
            conn.execute(f"ALTER TABLE {table} RENAME TO {table}_legacy")
            conn.execute(create_sql)
            new_cols = {c["name"] for c in conn.execute(f"PRAGMA table_info({table})")}
            if table == "kanban_notify_subs":
                # Cast the legacy TEXT cursor to INTEGER; NULL / non-numeric → 0.
                shared = [c for c in old_cols if c in new_cols and c != "last_event_id"]
                cols_csv = ", ".join(shared)
                conn.execute(
                    f"INSERT INTO {table} ({cols_csv}, last_event_id) "
                    f"SELECT {cols_csv}, COALESCE(CAST(last_event_id AS INTEGER), 0) "
                    f"FROM {table}_legacy"
                )
            else:
                # Drop the legacy TEXT id; AUTOINCREMENT reassigns it.
                shared = [c for c in old_cols if c in new_cols and c != "id"]
                cols_csv = ", ".join(shared)
                conn.execute(
                    f"INSERT INTO {table} ({cols_csv}) "
                    f"SELECT {cols_csv} FROM {table}_legacy"
                )
            conn.execute(f"DROP TABLE {table}_legacy")
            for index_sql in index_sqls:
                conn.execute(index_sql)
        conn.execute("COMMIT")
    except Exception:
        try:
            conn.execute("ROLLBACK")
        except sqlite3.OperationalError:
            pass
        raise


def _check_file_length_invariant(conn: sqlite3.Connection) -> None:
    """Compare SQLite's own page accounting against the file size on disk.

    Raises sqlite3.DatabaseError if the file is shorter than the header claims
    (torn-extend corruption).

    Both sides are read WITHOUT opening the database file. The header side
    comes from ``PRAGMA page_count`` over the existing connection; the on-disk
    side from ``stat()``. An earlier version read the header field with a bare
    ``open(path,"rb")`` -- but ``close()`` cancels every POSIX advisory lock
    this process holds on the file, so that probe silently dropped the locks
    of concurrent writers (and of a running VACUUM) and let other processes
    write into a database a writer still believed it owned. That is the
    documented corruption route in sqlite.org/howtocorrupt.html section 2.2.
    """
    from hermes_cli.sqlite_safe_read import file_length_matches_header

    # In WAL mode a just-committed page can still live in the -wal file, so
    # the main file legitimately lags its page count. Only enforce the
    # invariant under a rollback journal, where every committed page must
    # already be in the main file.
    try:
        row = conn.execute("PRAGMA journal_mode").fetchone()
        journal_mode = str(row[0]).lower() if row and row[0] is not None else ""
    except sqlite3.Error:
        return
    if journal_mode == "wal":
        return

    ok = file_length_matches_header(conn)
    if ok is False:
        raise sqlite3.DatabaseError(
            "torn-extend detected: the database file is shorter than its "
            "header page count claims"
        )


# SQLite's own busy_timeout uses a near-deterministic backoff, so concurrent
# writers re-collide in lockstep under a stampede. A jittered retry on the
# transaction boundary breaks that convoy. Mirrors state.db's _execute_write:
# a fixed 20-150ms jitter band (a 20ms floor prevents a near-zero retry from
# busy-spinning back into the collision). Only BEGIN IMMEDIATE and COMMIT are
# retried -- both are idempotent re-issues that touch no transaction body, so a
# CAS inside write_txn is never replayed. kanban keeps fewer retries than
# state.db (5 vs 15) because its 120s busy_timeout already absorbs most waits;
# the retry is the backstop for the tail SQLite returns BUSY on immediately.
_BUSY_MAX_RETRIES = 5
_BUSY_RETRY_MIN_S = 0.020  # 20ms
_BUSY_RETRY_MAX_S = 0.150  # 150ms


def _is_busy_error(exc: BaseException) -> bool:
    return isinstance(exc, sqlite3.OperationalError) and (
        "database is locked" in str(exc).lower()
        or "database is busy" in str(exc).lower()
    )


def _execute_boundary_with_retry(conn: sqlite3.Connection, sql: str) -> None:
    for attempt in range(_BUSY_MAX_RETRIES + 1):
        try:
            conn.execute(sql)
            return
        except sqlite3.OperationalError as exc:
            if not _is_busy_error(exc) or attempt == _BUSY_MAX_RETRIES:
                raise
            time.sleep(random.uniform(_BUSY_RETRY_MIN_S, _BUSY_RETRY_MAX_S))


@contextlib.contextmanager
def write_txn(conn: sqlite3.Connection, *, allow_nested: bool = False):
    """Context manager for an IMMEDIATE write transaction.

    Use for any multi-statement write (creating a task + link, claiming a
    task + recording an event, etc.). A claim CAS inside this context is
    atomic -- at most one concurrent writer can succeed.

    Nesting is an explicit opt-in: a caller already inside a transaction
    gets a loud ``RuntimeError`` unless it passes ``allow_nested=True``,
    in which case a SQLite savepoint is used instead of a second
    ``BEGIN IMMEDIATE``. Only composition primitives that graph builders
    deliberately run under one outer commit (``create_task``,
    ``add_comment``) opt in — helpers with post-commit side effects
    (``complete_task`` & co.) must never run under an open outer
    transaction, because their side effects (workspace cleanup, ready
    recomputation, failure-counter clears) would fire while the outer
    transaction can still roll back.

    The explicit ROLLBACK on exception is wrapped in try/except so that
    a SQLite auto-rollback (which leaves no active transaction) does not
    shadow the original exception with a spurious rollback error.
    """
    _assert_not_delegated_child_mutation()
    if getattr(conn, "in_transaction", False):
        if not allow_nested:
            raise RuntimeError(
                "write_txn: already inside a transaction. Nested composition "
                "must opt in explicitly with write_txn(conn, allow_nested=True) "
                "(savepoint semantics; the inner RELEASE is not durable until "
                "the outer transaction commits)."
            )
        savepoint = f"hermes_nested_{secrets.token_hex(8)}"
        conn.execute(f"SAVEPOINT {savepoint}")
        try:
            yield conn
        except Exception:
            try:
                conn.execute(f"ROLLBACK TO {savepoint}")
                conn.execute(f"RELEASE {savepoint}")
            except sqlite3.OperationalError:
                pass
            raise
        else:
            conn.execute(f"RELEASE {savepoint}")
        return

    _execute_boundary_with_retry(conn, "BEGIN IMMEDIATE")
    try:
        yield conn
    except Exception:
        try:
            conn.execute("ROLLBACK")
        except sqlite3.OperationalError:
            # SQLite has already auto-rolled-back the transaction (typical
            # under EIO, lock contention, or corruption). Nothing to undo;
            # do not let this secondary failure shadow the real one.
            pass
        raise
    else:
        try:
            _execute_boundary_with_retry(conn, "COMMIT")
        except Exception:
            # COMMIT exhausted retries with the txn still open; roll back so the
            # connection isn't poisoned for the next BEGIN IMMEDIATE.
            try:
                conn.execute("ROLLBACK")
            except sqlite3.OperationalError:
                pass
            raise
        # Post-commit file-length check: header page_count must match actual file pages.
        # A discrepancy means a torn-extend — raise now rather than silently corrupt.
        _check_file_length_invariant(conn)


# ---------------------------------------------------------------------------
# ID generation
# ---------------------------------------------------------------------------

def _new_task_id() -> str:
    """Generate a short, URL-safe task id.

    4 hex bytes = ~4.3B possibilities. At 10k tasks the collision
    probability is ~1.2e-5; at 100k it's ~1.2e-3. Previously we used 2
    hex bytes (65k possibilities) which hit the birthday paradox hard:
    ~5% collision probability at 1k tasks, ~50% at 10k. Callers that
    care about idempotency should pass ``idempotency_key`` to
    :func:`create_task` rather than rely on id uniqueness.
    """
    return "t_" + secrets.token_hex(4)


def _claimer_id() -> str:
    """Return a ``host:pid`` string that identifies this claimer."""
    import socket
    try:
        host = socket.gethostname() or "unknown"
    except Exception:
        host = "unknown"
    return f"{host}:{os.getpid()}"


# ---------------------------------------------------------------------------
# Task creation / mutation
# ---------------------------------------------------------------------------

_RESPONSIBILITY_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_.:-]{0,63}$")
_OWNED_PATH_FORBIDDEN_RE = re.compile(r"[\\:*?\[\]\x00-\x1f\x7f]")
_MAX_OWNED_PATHS = 64
_MAX_OWNED_PATH_LENGTH = 512
_MAX_RECEIPT_CHANGED_PATHS = 256


def normalize_responsibility(value: Optional[str]) -> Optional[str]:
    """Normalize an optional stable logical responsibility identifier."""
    if value is None:
        return None
    responsibility = str(value).strip() or None
    if responsibility is not None and not _RESPONSIBILITY_RE.fullmatch(responsibility):
        raise ValueError(
            "responsibility must be a 1-64 character stable identifier"
        )
    return responsibility


def normalize_owned_paths(value: Optional[Iterable[str]]) -> Optional[list[str]]:
    """Canonicalise a task's declared repository ownership.

    ``None`` is a legacy/unscoped task and therefore exclusive. ``[]`` is
    explicitly read-only, ``["."]`` owns the whole repository, and every
    other value is one exact relative POSIX file/subtree prefix. Globs are
    deliberately unsupported so ownership cannot silently change as files
    are added to the repository.
    """
    if value is None:
        return None
    if isinstance(value, (str, bytes)):
        raise ValueError("owned_paths must be a list of repository-relative paths")
    try:
        raw_paths = list(value)
    except TypeError as exc:
        raise ValueError(
            "owned_paths must be a list of repository-relative paths"
        ) from exc
    if len(raw_paths) > _MAX_OWNED_PATHS:
        raise ValueError(f"owned_paths may contain at most {_MAX_OWNED_PATHS} paths")

    cleaned: list[str] = []
    seen: set[str] = set()
    for index, raw in enumerate(raw_paths):
        if not isinstance(raw, str):
            raise ValueError(f"owned_paths[{index}] must be a string")
        path = raw.strip()
        if not path:
            raise ValueError(f"owned_paths[{index}] must not be blank")
        if len(path) > _MAX_OWNED_PATH_LENGTH:
            raise ValueError(
                f"owned_paths[{index}] must be at most {_MAX_OWNED_PATH_LENGTH} characters"
            )
        if path == ".":
            canonical = path
        else:
            if path.startswith("/") or _OWNED_PATH_FORBIDDEN_RE.search(path):
                raise ValueError(
                    f"owned_paths[{index}] must be a literal relative POSIX path"
                )
            candidate = PurePosixPath(path)
            if any(part in {"", ".", ".."} for part in candidate.parts):
                raise ValueError(
                    f"owned_paths[{index}] must not contain '.', '..', or empty segments"
                )
            canonical = candidate.as_posix()
            if canonical != path or canonical == ".git" or canonical.startswith(".git/"):
                raise ValueError(
                    f"owned_paths[{index}] must be a canonical repository path"
                )
        if canonical not in seen:
            seen.add(canonical)
            cleaned.append(canonical)

    if "." in seen and len(cleaned) != 1:
        raise ValueError("owned_paths '.' cannot be combined with narrower paths")
    return cleaned


def _canonical_assignee(assignee: Optional[str]) -> Optional[str]:
    """Lowercase-assignee normalization for Kanban rows (dashboard/CLI parity)."""
    if assignee is None:
        return None
    from hermes_cli.profiles import normalize_profile_name

    return normalize_profile_name(assignee)


def create_task(
    conn: sqlite3.Connection,
    *,
    title: str,
    body: Optional[str] = None,
    assignee: Optional[str] = None,
    responsibility: Optional[str] = None,
    created_by: Optional[str] = None,
    workspace_kind: str = "scratch",
    workspace_path: Optional[str] = None,
    branch_name: Optional[str] = None,
    tenant: Optional[str] = None,
    priority: int = 0,
    parents: Iterable[str] = (),
    triage: bool = False,
    idempotency_key: Optional[str] = None,
    max_runtime_seconds: Optional[int] = None,
    skills: Optional[Iterable[str]] = None,
    max_retries: Optional[int] = None,
    model_override: Optional[str] = None,
    provider_override: Optional[str] = None,
    reasoning_effort: Optional[str] = None,
    execution_tier: Optional[str] = None,
    model_policy_lock: Optional[str] = None,
    goal_mode: bool = False,
    goal_max_turns: Optional[int] = None,
    initial_status: str = "running",
    session_id: Optional[str] = None,
    board: Optional[str] = None,
    project_id: Optional[str] = None,
    project_source_task_id: Optional[str] = None,
    owned_paths: Optional[Iterable[str]] = None,
    integrates_parent_heads: bool = False,
    control: bool = False,
    receipt_owned: bool = False,
) -> str:
    """Create a new task and optionally link it under parent tasks.

    Returns the new task id.  Status is ``ready`` when there are no
    parents (or all parents already ``done``), otherwise ``todo``.
    If ``triage=True``, status is forced to ``triage`` regardless of
    parents — a specifier/triager is expected to promote the task to
    ``todo`` once the spec is fleshed out.

    ``control=True`` creates the row with ``task_kind='control'`` instead of
    ``'work'``: a permanently NON-EXECUTABLE anchor. Every dispatcher,
    claim, specify, decompose and route-mutation query in this module positively
    requires ``task_kind = 'work'``, so a control row can never be promoted,
    assigned, claimed or spawned by any of them — only owner-approved graph
    creation makes executable tasks, and those carry exact route authority.

    ``receipt_owned=True`` records that a committed owner receipt owns this
    work row. Only the owner-workspace kernel passes it, and it is what keeps
    the row governed even if its route columns are ever cleared — see
    :func:`task_is_policy_governed`.

    If ``idempotency_key`` is provided and a non-archived task with the
    same key already exists, returns the existing task's id instead of
    creating a duplicate. Useful for retried webhooks / automation that
    should not double-write.

    ``max_runtime_seconds`` caps how long a worker may run before the
    dispatcher SIGTERMs (then SIGKILLs after a grace window) and
    re-queues the task. ``None`` means no cap (default).

    ``skills`` is an optional list of skill names to force-load into
    the worker when dispatched. Stored as JSON; the dispatcher passes
    each name to ``hermes --skills ...``. Use this to pin a task to a
    specialist skill (e.g. ``skills=["translation"]`` so the worker loads the
    translation skill regardless of the profile's default config).

    ``model_override`` / ``provider_override`` pin the worker to a specific
    model (and optionally its provider) without touching the profile's
    config — passed to the worker as ``-m <model> [--provider <name>]``.
    ``provider_override`` requires ``model_override``.

    ``reasoning_effort`` pins the worker's thinking depth for this task
    (``minimal``…``ultra``, or ``none`` to disable thinking), passed as
    ``--reasoning <level>``. It is independent of ``model_override``: a task
    can run the profile's own model at a different depth.

    ``model_policy_lock`` marks the assignee, the three fields above and
    ``execution_tier`` as an owner-approved immutable route: the route mutators
    refuse the task, a role transition must re-derive a separately approved
    route, and the dispatcher runs it with fallbacks disabled. It must be a
    lock this build's model policy still admits for exactly that authority.

    ``project_source_task_id`` is an internal cross-profile fallback for a
    worker-created child. When the active profile cannot resolve ``project_id``
    in its own projects.db, a matching canonical project-linked task in this
    board can supply the repo and branch convention. Its literal worktree is
    never reused; the new task still gets its own task-id-keyed path.

    ``owned_paths`` declares repository write ownership for bounded parallel
    execution: ``None`` is legacy fail-closed whole-repository ownership,
    ``[]`` is read-only, ``["."]`` is explicit whole-repository ownership, and
    other entries are canonical relative file/subtree prefixes. An integration
    task sets ``integrates_parent_heads`` so completion must contain the exact
    current git receipt of every mutating same-Project parent.
    """
    model_override = (model_override or "").strip() or None
    provider_override = (provider_override or "").strip() or None
    reasoning_effort = normalize_reasoning_effort(reasoning_effort)
    if provider_override and not model_override:
        raise ValueError("provider_override requires a model_override")
    execution_tier = (execution_tier or "").strip().lower() or None
    model_policy_lock = (model_policy_lock or "").strip() or None
    assignee = _canonical_assignee(assignee)
    if model_policy_lock is not None:
        route_error = policy_lock_error(
            model_policy_lock,
            assignee,
            provider_override,
            model_override,
            reasoning_effort,
            execution_tier,
        )
        if route_error:
            raise ValueError(route_error)
    if control and (assignee or model_policy_lock or execution_tier):
        # A control anchor is non-executable by construction, so it must not
        # carry any of the fields that only mean something for executable work.
        raise ValueError(
            "a control task cannot carry an assignee or a route"
        )
    responsibility = normalize_responsibility(responsibility)
    owned_paths_list = normalize_owned_paths(owned_paths)
    if not isinstance(integrates_parent_heads, bool):
        raise ValueError("integrates_parent_heads must be a boolean")
    if integrates_parent_heads and not owned_paths_list:
        raise ValueError(
            "integrates_parent_heads requires a mutating owned_paths scope"
        )
    if not title or not title.strip():
        raise ValueError("title is required")
    if initial_status not in VALID_INITIAL_STATUSES:
        raise ValueError(
            f"initial_status must be one of {sorted(VALID_INITIAL_STATUSES)}"
        )
    if workspace_kind not in VALID_WORKSPACE_KINDS:
        raise ValueError(
            f"workspace_kind must be one of {sorted(VALID_WORKSPACE_KINDS)}, "
            f"got {workspace_kind!r}"
        )
    if branch_name is not None:
        branch_name = str(branch_name).strip() or None
    if branch_name and workspace_kind != "worktree":
        raise ValueError("branch_name is only valid for worktree workspaces")

    # Inherit the board's scoped project when the caller didn't name one, so a
    # project-scoped board anchors every new task to that project's repo
    # (deterministic worktree + branch) without each surface repeating it.
    if project_id is None:
        try:
            _bmeta = read_board_metadata(board if board else get_current_board())
            _board_project = (_bmeta.get("project_id") or "").strip()
            if _board_project:
                project_id = _board_project
        except Exception:
            pass

    # Resolve an optional first-class Project link. A project-linked task is
    # anchored to the project's primary repo as a git worktree, so its branch
    # can be named deterministically (project slug + task id) instead of the
    # random ``wt/<task-id>`` fallback the worker skill applies when no branch
    # is set. Projects live in the creator's per-profile projects.db; the repo
    # path is absolute (profile-independent) and the branch name is pure, so the
    # cross-profile dispatcher needs no projects.db access at dispatch time.
    project_obj = None
    # Primary repo of a project-linked worktree task whose path we still need to
    # derive (a fresh worktree dir under the repo, computed once task_id exists).
    project_repo: Optional[str] = None
    if project_id is not None:
        project_id = str(project_id).strip() or None
    if project_id:
        from hermes_cli import projects_db as _pdb

        try:
            with _pdb.connect_closing() as _pconn:
                project_obj = _pdb.get_project(_pconn, project_id)
        except Exception:
            project_obj = None
        if project_obj is None and project_source_task_id:
            # Worker profiles have their own projects.db, while the Kanban DB is
            # intentionally shared. Recover routing only from a canonical
            # project-linked source task in this same board. This carries the
            # repo + project branch convention forward without copying or
            # opening the creator profile's project store, and without reusing
            # the source task's literal worktree path.
            source_task = get_task(conn, str(project_source_task_id))
            if (
                source_task is not None
                and source_task.project_id == project_id
                and source_task.workspace_kind == "worktree"
                and source_task.workspace_path
            ):
                source_path = Path(source_task.workspace_path)
                if (
                    source_path.is_absolute()
                    and source_path.name == source_task.id
                    and source_path.parent.name == ".worktrees"
                ):
                    project_slug = None
                    if source_task.branch_name:
                        prefix, separator, leaf = source_task.branch_name.partition("/")
                        if separator and (
                            leaf == source_task.id
                            or leaf.startswith(f"{source_task.id}-")
                        ):
                            try:
                                project_slug = _pdb.normalize_slug(prefix)
                            except ValueError:
                                project_slug = None
                    if project_slug is None:
                        try:
                            project_slug = _pdb.normalize_slug(project_id)
                        except ValueError:
                            project_slug = None
                    if project_slug:
                        project_repo = str(source_path.parent.parent)
                        project_obj = _pdb.Project(
                            id=project_id,
                            slug=project_slug,
                            name=project_slug,
                            created_at=0,
                            primary_path=project_repo,
                        )
                        if workspace_kind == "scratch":
                            workspace_kind = "worktree"

        if project_obj is None:
            # A project id/slug that doesn't resolve must not crash task
            # creation or persist a dangling reference — drop the link and
            # create the task as an ordinary (scratch) task.
            project_id = None
        else:
            # Canonicalise (a slug may have been passed) and anchor the
            # worktree under the project's primary repo.
            project_id = project_obj.id
            if workspace_kind == "scratch" and project_obj.primary_path:
                workspace_kind = "worktree"
            if (
                workspace_kind == "worktree"
                and workspace_path is None
                and project_obj.primary_path
            ):
                # Defer the concrete path to the insert loop: it's a fresh
                # ``<repo>/.worktrees/<task-id>`` dir keyed on the new task id.
                project_repo = str(project_obj.primary_path)

    if owned_paths_list and workspace_kind != "worktree":
        raise ValueError(
            "mutating owned_paths require workspace_kind='worktree'"
        )
    if owned_paths_list == [] and workspace_kind == "dir":
        raise ValueError(
            "read-only owned_paths require an isolated scratch or worktree workspace"
        )

    parents = tuple(p for p in parents if p)
    if control and parents:
        # Same asymmetry the link path enforces (see _assert_dependency_child):
        # an anchor gates work, it is never itself gated, so it may not be
        # created as a dependency child either.
        raise ValueError("a control task cannot depend on another task")

    # Normalise + validate skills: strip whitespace, drop empties, dedupe
    # (preserving order). Refuse commas inside a single name so we don't
    # invisibly splatter a comma-joined string into one argv slot — the
    # `hermes --skills X,Y` comma syntax is handled in the dispatcher,
    # not here.
    skills_list: Optional[list[str]] = None
    if skills is not None:
        cleaned: list[str] = []
        seen: set[str] = set()
        # Collect all toolset-name confusions up front so the user sees the
        # whole list at once. Raising on the first hit is friendly when the
        # input has one mistake, but agents that confuse skills with toolsets
        # usually pass several at once (`skills=["web", "browser", "terminal"]`)
        # and serial-correcting one per failure round-trips wastes tokens.
        toolset_typos: list[str] = []
        for s in skills:
            if not s:
                continue
            name = str(s).strip()
            if not name:
                continue
            if "," in name:
                raise ValueError(
                    f"skill name cannot contain comma: {name!r} "
                    f"(pass a list of separate names instead of a comma-joined string)"
                )
            if name.casefold() in KNOWN_TOOLSET_NAMES:
                toolset_typos.append(name)
                continue
            if name in seen:
                continue
            seen.add(name)
            cleaned.append(name)
        if toolset_typos:
            quoted = ", ".join(repr(n) for n in toolset_typos)
            noun = "is a toolset name" if len(toolset_typos) == 1 else "are toolset names"
            raise ValueError(
                f"{quoted} {noun}, not skill name(s). "
                "Put toolsets in the assignee profile's `toolsets:` config "
                "instead of per-task skills. Skills are named skill bundles "
                "(e.g. `blogwatcher`, `github-code-review`); toolsets are runtime "
                "capabilities (e.g. `web`, `browser`, `terminal`)."
            )
        skills_list = cleaned

    # Idempotency check — return the existing task instead of creating a
    # duplicate. Done BEFORE entering write_txn to keep the fast path fast
    # and to avoid holding a write lock during the lookup. Race is
    # acceptable: two concurrent creators with the same key might both
    # insert, at which point both rows exist but the next lookup stabilises.
    if idempotency_key:
        row = conn.execute(
            "SELECT id FROM tasks WHERE idempotency_key = ? "
            "AND status != 'archived' AND task_kind = ? "
            "ORDER BY created_at DESC LIMIT 1",
            (idempotency_key, "control" if control else "work"),
        ).fetchone()
        if row:
            return row["id"]

    now = int(time.time())

    # Resolve workspace_path from board-level default_workdir when the
    # caller did not specify one explicitly. Board defaults represent
    # persistent project checkouts, so only persistent workspace kinds may
    # inherit them. Scratch workspaces are auto-deleted on completion and
    # must stay under the per-board scratch root created by
    # ``resolve_workspace``; inheriting ``default_workdir`` for a scratch
    # task would point cleanup at the user's source tree (#28818). The
    # containment guard in ``_cleanup_workspace`` is the safety rail, but
    # we also stop the bad state from being created in the first place.
    if (
        workspace_path is None
        and project_repo is None
        and workspace_kind in {"dir", "worktree"}
    ):
        board_slug = board if board else get_current_board()
        board_meta = read_board_metadata(board_slug)
        board_default = board_meta.get("default_workdir")
        if board_default:
            workspace_path = str(board_default)

    # Retry once on the extremely unlikely id collision.
    for attempt in range(2):
        task_id = _new_task_id()
        try:
            # ``allow_nested=True``: graph builders (kanban_swarm.create_swarm)
            # compose create_task calls under one outer commit so the
            # dispatcher can never observe a partially constructed graph.
            with write_txn(conn, allow_nested=True):
                # Determine task status from parent status, unless the caller
                # parks it directly in blocked for human-ops review or in
                # triage for a specifier.
                if initial_status == "blocked":
                    task_status = "blocked"
                    if parents:
                        missing = _find_missing_parents(conn, parents)
                        if missing:
                            raise ValueError(f"unknown parent task(s): {', '.join(missing)}")
                elif triage:
                    task_status = "triage"
                else:
                    task_status = "ready"
                    if parents:
                        missing = _find_missing_parents(conn, parents)
                        if missing:
                            raise ValueError(f"unknown parent task(s): {', '.join(missing)}")
                        # If any parent is not yet done, we're todo.
                        rows = conn.execute(
                            "SELECT status FROM tasks WHERE id IN "
                            "(" + ",".join("?" * len(parents)) + ") "
                            f"AND task_kind IN {_DEPENDENCY_PARENT_KINDS}",
                            parents,
                        ).fetchall()
                        if any(r["status"] != "done" for r in rows):
                            task_status = "todo"
                # Even in triage mode we still need to validate parent ids
                # so the eventual link rows don't dangle.
                if triage and parents:
                    missing = _find_missing_parents(conn, parents)
                    if missing:
                        raise ValueError(f"unknown parent task(s): {', '.join(missing)}")

                # Project-linked worktree: a fresh worktree dir under the repo
                # plus a deterministic branch (project slug + task id). Together
                # these kill the random ``wt/<task-id>`` worker fallback and the
                # unanchored ``.worktrees/<id>`` under the dispatcher's cwd.
                if project_obj is not None and workspace_kind == "worktree":
                    if project_repo and not workspace_path:
                        workspace_path = os.path.join(
                            project_repo, ".worktrees", task_id
                        )
                    if not branch_name:
                        # _pdb was imported above when project_obj was resolved.
                        try:
                            branch_name = _pdb.branch_name_for(
                                project_obj, task_id, title=title or ""
                            )
                        except Exception:
                            branch_name = None

                conn.execute(
                    """
                    INSERT INTO tasks (
                        id, title, body, assignee, responsibility, status, priority,
                        created_by, created_at, workspace_kind, workspace_path,
                        branch_name, project_id, owned_paths, integrates_parent_heads,
                        tenant, idempotency_key,
                        max_runtime_seconds,
                        skills, max_retries, model_override, provider_override,
                        reasoning_effort, execution_tier, model_policy_lock,
                        goal_mode, goal_max_turns, session_id, task_kind,
                        owner_receipt_bound
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        task_id,
                        title.strip(),
                        body,
                        assignee,
                        responsibility,
                        task_status,
                        priority,
                        created_by,
                        now,
                        workspace_kind,
                        workspace_path,
                        branch_name,
                        project_id,
                        json.dumps(owned_paths_list) if owned_paths_list is not None else None,
                        1 if integrates_parent_heads else 0,
                        tenant,
                        idempotency_key,
                        int(max_runtime_seconds) if max_runtime_seconds is not None else None,
                        json.dumps(skills_list) if skills_list is not None else None,
                        int(max_retries) if max_retries is not None else None,
                        model_override,
                        provider_override,
                        reasoning_effort,
                        execution_tier,
                        model_policy_lock,
                        1 if goal_mode else 0,
                        int(goal_max_turns) if goal_max_turns is not None else None,
                        session_id,
                        "control" if control else "work",
                        1 if receipt_owned else 0,
                    ),
                )
                for pid in parents:
                    conn.execute(
                        "INSERT OR IGNORE INTO task_links (parent_id, child_id) VALUES (?, ?)",
                        (pid, task_id),
                    )
                # Notify-sub inheritance (ACK-edge: the originating channel
                # still hears about a child that BLOCKs, not just the final
                # fan-in) is handled by the single-owner helper below —
                # _inherit_notify_subs copies every routing/delivery column.
                _append_event(
                    conn,
                    task_id,
                    "created",
                    {
                        "assignee": assignee,
                        "status": task_status,
                        "parents": list(parents),
                        "tenant": tenant,
                        "workspace_kind": workspace_kind,
                        "workspace_path": workspace_path,
                        "branch_name": branch_name,
                        "project_id": project_id,
                        "owned_paths": owned_paths_list,
                        "integrates_parent_heads": integrates_parent_heads or None,
                        "skills": list(skills_list) if skills_list else None,
                        "goal_mode": bool(goal_mode) or None,
                        "model_override": model_override,
                        "provider_override": provider_override,
                        "execution_tier": execution_tier,
                        "model_route_pinned": bool(model_policy_lock) or None,
                    },
                )
                _inherit_notify_subs(conn, task_id, parents, created_at=now)
            return task_id
        except sqlite3.IntegrityError:
            if attempt == 1:
                raise
            # Retry with a fresh id.
            continue
    raise RuntimeError("unreachable")


def _bounded_display_field(
    value: Optional[str],
    *,
    name: str,
    max_len: int,
    required: bool,
) -> Optional[str]:
    """Validate, bound, and redact-before-durability an opaque recommendation display field.

    Raises ``ValueError`` when ``required`` and blank/missing, when the type isn't ``str``,
    or when the stripped value exceeds ``max_len``. Returns ``None`` for an optional blank
    field, otherwise the redacted, stripped string.
    """
    if value is None:
        if required:
            raise ValueError(f"{name} is required")
        return None
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string")
    cleaned = value.strip()
    if not cleaned:
        if required:
            raise ValueError(f"{name} is required")
        return None
    if len(cleaned) > max_len:
        raise ValueError(f"{name} exceeds {max_len} characters")
    return redact_review_value(cleaned)


def normalize_recommendation_evidence(value: Any) -> tuple[dict[str, Any], str]:
    """Validate, redact, and canonically serialize recommendation evidence."""
    if not isinstance(value, dict):
        raise ValueError("recommendation_evidence must be an object")
    expected = {
        "schema_version",
        "need",
        "expected_benefit",
        "requested_scope",
        "risks",
        "cost",
        "rollback",
    }
    if set(value) != expected:
        raise ValueError(
            "recommendation_evidence must contain exactly "
            + ", ".join(sorted(expected))
        )
    if value["schema_version"] != 1 or isinstance(value["schema_version"], bool):
        raise ValueError("recommendation_evidence.schema_version must be 1")
    scope = value["requested_scope"]
    if not isinstance(scope, dict) or set(scope) != set(RECOMMENDATION_SCOPE_FLAGS):
        raise ValueError(
            "recommendation_evidence.requested_scope must contain exactly "
            + ", ".join(RECOMMENDATION_SCOPE_FLAGS)
        )
    normalized_scope: dict[str, bool] = {}
    for flag in RECOMMENDATION_SCOPE_FLAGS:
        flag_value = scope[flag]
        if not isinstance(flag_value, bool):
            raise ValueError(
                f"recommendation_evidence.requested_scope.{flag} must be boolean"
            )
        normalized_scope[flag] = flag_value
    normalized: dict[str, Any] = {
        "schema_version": 1,
        "need": _bounded_display_field(
            value["need"],
            name="recommendation_evidence.need",
            max_len=_RECOMMENDATION_EVIDENCE_TEXT_MAX_LEN,
            required=True,
        ),
        "expected_benefit": _bounded_display_field(
            value["expected_benefit"],
            name="recommendation_evidence.expected_benefit",
            max_len=_RECOMMENDATION_EVIDENCE_TEXT_MAX_LEN,
            required=True,
        ),
        "requested_scope": normalized_scope,
        "risks": _bounded_display_field(
            value["risks"],
            name="recommendation_evidence.risks",
            max_len=_RECOMMENDATION_EVIDENCE_TEXT_MAX_LEN,
            required=True,
        ),
        "cost": _bounded_display_field(
            value["cost"],
            name="recommendation_evidence.cost",
            max_len=_RECOMMENDATION_EVIDENCE_TEXT_MAX_LEN,
            required=True,
        ),
        "rollback": _bounded_display_field(
            value["rollback"],
            name="recommendation_evidence.rollback",
            max_len=_RECOMMENDATION_EVIDENCE_TEXT_MAX_LEN,
            required=True,
        ),
    }
    serialized = json.dumps(
        normalized, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    if len(serialized.encode("utf-8")) > _RECOMMENDATION_EVIDENCE_MAX_BYTES:
        raise ValueError(
            f"recommendation_evidence exceeds {_RECOMMENDATION_EVIDENCE_MAX_BYTES} bytes"
        )
    return normalized, serialized


def parse_recommendation_evidence(value: Optional[str]) -> Optional[dict[str, Any]]:
    """Return validated evidence, or None for a legacy recommendation."""
    if value is None:
        return None
    try:
        decoded = json.loads(value)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError("stored recommendation_evidence is invalid JSON") from exc
    normalized, _ = normalize_recommendation_evidence(decoded)
    return normalized


def _recommendation_identity_key(
    *,
    project_id: str,
    target_profile: str,
    recommendation_kind: str,
    provenance_authority: str,
    recommendation_subject_id: str,
) -> str:
    """Digest the safe scope identity of a recommendation (ITEM31BH).

    A recommendation is advice a human owner reviews, so re-publishing the same
    advice — reworded, observed later, or noticed while working a different task
    — must not create a second nag. Identity is therefore the *scope* only:
    project, target profile, kind, provenance authority, and subject id. Label,
    rationale, observed time, and provenance ref are deliberately excluded.

    Callers must pass the already redacted/normalized values, so a secret that
    leaked into a display field cannot fork the identity (it is ``[REDACTED]``
    by the time it gets here) and never becomes durable state: only the digest
    is stored, never the digest input.
    """
    payload = json.dumps(
        {
            "project_id": project_id,
            "target_profile": target_profile,
            "recommendation_kind": recommendation_kind,
            "provenance_authority": provenance_authority,
            "recommendation_subject_id": recommendation_subject_id,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return f"{_RECOMMENDATION_IDENTITY_VERSION}:{digest}"


def create_recommendation(
    conn: sqlite3.Connection,
    *,
    project_id: str,
    target_profile: str,
    recommendation_kind: str,
    recommendation_subject_id: str,
    recommendation_label: str,
    recommendation_rationale: Optional[str] = None,
    recommendation_evidence: dict[str, Any],
    provenance_authority: str,
    provenance_ref: Optional[str] = None,
    provenance_observed_at: Optional[int] = None,
) -> str:
    """Create a native recommendation card for owner review.

    Unlike :func:`create_task`, this is a dedicated seam for non-actionable advice for a
    human owner to accept or reject, never dispatched to a worker: it writes the row
    directly (no assignee, workspace/branch, or run; ``status='review'``,
    ``review_policy='owner'`` unconditionally). Returns the new task id.

    Creation is idempotent on the safe scope identity (see
    :func:`_recommendation_identity_key`): a repeat publish returns the existing
    card's id and writes no second row and no second event. The lookup runs
    inside the same ``BEGIN IMMEDIATE`` write transaction as the insert, so
    concurrent callers (including separate connections/processes) collapse to
    one row. A resolved or archived card stays the dedup authority — owner
    review is a decision, and re-nagging past it is exactly what this prevents.
    """
    if recommendation_kind not in VALID_RECOMMENDATION_KINDS:
        raise ValueError(
            "recommendation_kind must be one of "
            f"{sorted(VALID_RECOMMENDATION_KINDS)}, got {recommendation_kind!r}"
        )

    if not isinstance(project_id, str):
        raise ValueError("project_id must be a string")
    project_id = project_id.strip()
    if not project_id:
        raise ValueError("project_id is required")

    from hermes_cli.profiles import normalize_profile_name

    target_profile = normalize_profile_name(target_profile)

    recommendation_subject_id = _bounded_display_field(
        recommendation_subject_id,
        name="recommendation_subject_id",
        max_len=_RECOMMENDATION_SUBJECT_ID_MAX_LEN,
        required=True,
    )
    recommendation_label = _bounded_display_field(
        recommendation_label,
        name="recommendation_label",
        max_len=_RECOMMENDATION_LABEL_MAX_LEN,
        required=True,
    )
    recommendation_rationale = _bounded_display_field(
        recommendation_rationale,
        name="recommendation_rationale",
        max_len=_RECOMMENDATION_RATIONALE_MAX_LEN,
        required=False,
    )
    provenance_authority = _bounded_display_field(
        provenance_authority,
        name="provenance_authority",
        max_len=_RECOMMENDATION_PROVENANCE_AUTHORITY_MAX_LEN,
        required=True,
    )
    provenance_ref = _bounded_display_field(
        provenance_ref,
        name="provenance_ref",
        max_len=_RECOMMENDATION_PROVENANCE_REF_MAX_LEN,
        required=False,
    )
    if provenance_observed_at is not None and (
        isinstance(provenance_observed_at, bool)
        or not isinstance(provenance_observed_at, int)
    ):
        raise ValueError("provenance_observed_at must be an integer unix timestamp")
    normalized_evidence, serialized_evidence = normalize_recommendation_evidence(
        recommendation_evidence
    )
    requested_scope = normalized_evidence["requested_scope"]
    if recommendation_kind == "permission" and not requested_scope[
        "permission_widening"
    ]:
        raise ValueError(
            "permission recommendations must declare requested_scope.permission_widening"
        )
    if recommendation_kind == "connection" and not requested_scope[
        "connector_access"
    ]:
        raise ValueError(
            "connection recommendations must declare requested_scope.connector_access"
        )

    identity_key = _recommendation_identity_key(
        project_id=project_id,
        target_profile=target_profile,
        recommendation_kind=recommendation_kind,
        provenance_authority=provenance_authority,
        recommendation_subject_id=recommendation_subject_id,
    )

    now = int(time.time())
    for attempt in range(2):
        task_id = _new_task_id()
        try:
            with write_txn(conn, allow_nested=True):
                # Inside the write transaction on purpose: BEGIN IMMEDIATE
                # already serializes writers, so a concurrent publisher either
                # loses the race and reads the winner's row here, or wins and
                # is the row everyone else finds. No status filter — an
                # archived/resolved card is still the dedup authority.
                existing = conn.execute(
                    "SELECT id, recommendation_evidence, recommendation_decision, "
                    "recommendation_effective_state, recommendation_lifecycle_version "
                    "FROM tasks WHERE idempotency_key = ? "
                    "AND task_kind = 'recommendation' "
                    "ORDER BY created_at ASC, id ASC LIMIT 1",
                    (identity_key,),
                ).fetchone()
                if existing is not None:
                    if existing["recommendation_evidence"] is None:
                        decision = existing["recommendation_decision"]
                        effective_state = existing["recommendation_effective_state"]
                        lifecycle_version = existing[
                            "recommendation_lifecycle_version"
                        ]
                        if (
                            decision not in {None, "pending"}
                            or effective_state not in {None, "none"}
                            or lifecycle_version not in {None, 0}
                        ):
                            raise ValueError(
                                "legacy recommendation lifecycle cannot be backfilled"
                            )
                        conn.execute(
                            "UPDATE tasks SET recommendation_evidence = ?, "
                            "recommendation_decision = 'pending', "
                            "recommendation_effective_state = 'none', "
                            "recommendation_lifecycle_version = 0 "
                            "WHERE id = ? AND recommendation_evidence IS NULL",
                            (serialized_evidence, existing["id"]),
                        )
                        _append_event(
                            conn,
                            existing["id"],
                            "recommendation_evidence_added",
                            {"evidence_schema_version": 1, "lifecycle_version": 0},
                        )
                    return existing["id"]
                conn.execute(
                    """
                    INSERT INTO tasks (
                        id, title, body, status, created_at, project_id,
                        task_kind, recommendation_kind, recommendation_subject_id,
                        recommendation_label, recommendation_rationale,
                        target_profile, review_policy,
                        provenance_authority, provenance_ref, provenance_observed_at,
                        recommendation_evidence, recommendation_decision,
                        recommendation_effective_state, recommendation_lifecycle_version,
                        idempotency_key
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        task_id,
                        _RECOMMENDATION_TASK_TITLE,
                        None,
                        "review",
                        now,
                        project_id,
                        "recommendation",
                        recommendation_kind,
                        recommendation_subject_id,
                        recommendation_label,
                        recommendation_rationale,
                        target_profile,
                        "owner",
                        provenance_authority,
                        provenance_ref,
                        provenance_observed_at,
                        serialized_evidence,
                        "pending",
                        "none",
                        0,
                        identity_key,
                    ),
                )
                _append_event(
                    conn,
                    task_id,
                    "recommendation_created",
                    {
                        "project_id": project_id,
                        "target_profile": target_profile,
                        "recommendation_kind": recommendation_kind,
                        "recommendation_subject_id": recommendation_subject_id,
                        "recommendation_label": recommendation_label,
                        "provenance_authority": provenance_authority,
                        "provenance_ref": provenance_ref,
                        "provenance_observed_at": provenance_observed_at,
                        "evidence_schema_version": 1,
                    },
                )
            return task_id
        except sqlite3.IntegrityError:
            if attempt == 1:
                raise
            # Retry with a fresh id.
            continue
    raise RuntimeError("unreachable")


def _assert_recommendation_lifecycle_operator() -> None:
    """Keep decision/apply evidence outside dispatcher-owned agent work."""
    if os.environ.get("HERMES_KANBAN_TASK"):
        raise PermissionError(
            "recommendation decisions and transitions are operator-only"
        )


def _recommendation_row(conn: sqlite3.Connection, task_id: str) -> sqlite3.Row:
    row = conn.execute(
        "SELECT * FROM tasks WHERE id = ? AND task_kind = 'recommendation' "
        "AND review_policy = 'owner'",
        (task_id,),
    ).fetchone()
    if row is None:
        raise ValueError(f"recommendation {task_id!r} not found")
    return row


def recommendation_lifecycle_snapshot(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "recommendation_id": row["id"],
        "decision": row["recommendation_decision"] or "pending",
        "effective_state": row["recommendation_effective_state"] or "none",
        "lifecycle_version": int(row["recommendation_lifecycle_version"] or 0),
    }


def _bounded_lifecycle_text(value: Any, *, name: str, required: bool = True) -> Optional[str]:
    return _bounded_display_field(
        value,
        name=name,
        max_len=_RECOMMENDATION_LIFECYCLE_TEXT_MAX_LEN,
        required=required,
    )


def _lifecycle_reference(value: Any, *, name: str) -> str:
    value = _bounded_lifecycle_text(value, name=name)
    if value is None or not _RECOMMENDATION_ID_RE.fullmatch(value):
        raise ValueError(
            f"{name} must be a lowercase non-secret identifier using letters, "
            "digits, dot, underscore, colon, or hyphen"
        )
    return value


def _sha256_identity(value: Any, *, name: str, required: bool = True) -> Optional[str]:
    if value is None and not required:
        return None
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise ValueError(f"{name} must be a lowercase sha256 digest")
    return value


def _completed_work_run_evidence(
    conn: sqlite3.Connection,
    *,
    recommendation: sqlite3.Row,
    task_id: Any,
    run_id: Any,
    role: str,
    not_task_ids: set[str],
    not_run_ids: set[int],
    min_started_at: int,
) -> dict[str, int | str]:
    task_id = _lifecycle_reference(task_id, name=f"{role}_task_id")
    if isinstance(run_id, bool) or not isinstance(run_id, int) or run_id < 1:
        raise ValueError(f"{role}_run_id must be a positive integer")
    if task_id == recommendation["id"] or task_id in not_task_ids:
        raise ValueError(f"{role} task must be distinct")
    if run_id in not_run_ids:
        raise ValueError(f"{role} run must be distinct")
    row = conn.execute(
        "SELECT t.status AS task_status, t.project_id, r.started_at, r.ended_at, "
        "r.outcome, (SELECT MAX(r2.id) FROM task_runs r2 "
        "WHERE r2.task_id = t.id) AS latest_run_id "
        "FROM tasks t JOIN task_runs r ON r.task_id = t.id "
        "WHERE t.id = ? AND t.task_kind = 'work' AND r.id = ?",
        (task_id, run_id),
    ).fetchone()
    if row is None:
        raise ValueError(f"{role} Task/run evidence was not found")
    if (
        row["task_status"] != "done"
        or row["outcome"] != "completed"
        or row["ended_at"] is None
    ):
        raise ValueError(f"{role} Task/run must be completed")
    if int(row["latest_run_id"]) != run_id:
        raise ValueError(f"{role} Task/run must be the latest run")
    if recommendation["project_id"] and row["project_id"] != recommendation["project_id"]:
        raise ValueError(f"{role} Task must belong to the recommendation project")
    observed_at = row["started_at"] if role in {"canary", "verifier"} else row["ended_at"]
    if int(observed_at) < int(min_started_at):
        raise ValueError(f"{role} Task/run is stale for this lifecycle transition")
    return {
        "task_id": task_id,
        "run_id": run_id,
        "started_at": int(row["started_at"]),
        "ended_at": int(row["ended_at"]),
    }


def _active_canary_run_evidence(
    conn: sqlite3.Connection,
    *,
    recommendation: sqlite3.Row,
    task_id: Any,
    run_id: Any,
    not_task_ids: set[str],
    not_run_ids: set[int],
    min_started_at: int,
) -> dict[str, int | str]:
    task_id = _lifecycle_reference(task_id, name="canary_task_id")
    if isinstance(run_id, bool) or not isinstance(run_id, int) or run_id < 1:
        raise ValueError("canary_run_id must be a positive integer")
    if task_id == recommendation["id"] or task_id in not_task_ids:
        raise ValueError("canary task must be distinct")
    if run_id in not_run_ids:
        raise ValueError("canary run must be distinct")
    row = conn.execute(
        "SELECT t.status AS task_status, t.project_id, t.current_run_id, "
        "r.started_at, r.ended_at, r.outcome FROM tasks t "
        "JOIN task_runs r ON r.task_id = t.id "
        "WHERE t.id = ? AND t.task_kind = 'work' AND r.id = ?",
        (task_id, run_id),
    ).fetchone()
    if row is None:
        raise ValueError("canary Task/run evidence was not found")
    if (
        row["task_status"] != "running"
        or row["current_run_id"] != run_id
        or row["ended_at"] is not None
        or row["outcome"] is not None
    ):
        raise ValueError("canary Task/run must be the active run")
    if recommendation["project_id"] and row["project_id"] != recommendation["project_id"]:
        raise ValueError("canary Task must belong to the recommendation project")
    if int(row["started_at"]) < int(min_started_at):
        raise ValueError("canary Task/run is stale for this lifecycle transition")
    return {"task_id": task_id, "run_id": run_id, "started_at": int(row["started_at"])}


def _recommendation_lifecycle_event(
    conn: sqlite3.Connection,
    task_id: str,
    *,
    effective_state: Optional[str] = None,
) -> Optional[tuple[dict[str, Any], int]]:
    rows = conn.execute(
        "SELECT payload, created_at FROM task_events WHERE task_id = ? "
        "AND kind IN ('recommendation_decided', 'recommendation_transitioned') "
        "ORDER BY id DESC",
        (task_id,),
    ).fetchall()
    for row in rows:
        try:
            payload = json.loads(row["payload"]) if row["payload"] else {}
        except (TypeError, json.JSONDecodeError):
            continue
        if effective_state is None or payload.get("effective_state") == effective_state:
            return payload, int(row["created_at"])
    return None


def _recommendation_lifecycle_references(
    conn: sqlite3.Connection, task_id: str
) -> tuple[set[str], set[int], int]:
    """Return prior Task/run evidence and the latest lifecycle timestamp."""
    rows = conn.execute(
        "SELECT payload, created_at FROM task_events WHERE task_id = ? "
        "AND kind IN ('recommendation_created', 'recommendation_evidence_added', "
        "'recommendation_decided', 'recommendation_transitioned') ORDER BY id ASC",
        (task_id,),
    ).fetchall()
    task_ids: set[str] = set()
    run_ids: set[int] = set()
    latest_at = 0
    for row in rows:
        latest_at = max(latest_at, int(row["created_at"]))
        try:
            payload = json.loads(row["payload"]) if row["payload"] else {}
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError("invalid recommendation lifecycle event") from exc
        if not isinstance(payload, dict):
            raise ValueError("invalid recommendation lifecycle event")
        for role in ("governance", "canary", "verifier"):
            prior_task = payload.get(f"{role}_task_id")
            prior_run = payload.get(f"{role}_run_id")
            if isinstance(prior_task, str):
                task_ids.add(prior_task)
            if isinstance(prior_run, int) and not isinstance(prior_run, bool):
                run_ids.add(prior_run)
    return task_ids, run_ids, latest_at


def decide_recommendation(
    conn: sqlite3.Connection,
    task_id: str,
    *,
    decision: str,
    authority: str,
    gate_ref: Optional[str],
    reason: str,
    actor: str,
    governance_task_id: str,
    governance_run_id: int,
    expected_lifecycle_version: int,
) -> dict[str, Any]:
    """Record an owner-governed decision; never apply configuration."""
    _assert_recommendation_lifecycle_operator()
    if decision not in {"deferred", "rejected", "accepted"}:
        raise ValueError("decision must be deferred, rejected, or accepted")
    if authority not in VALID_RECOMMENDATION_AUTHORITIES:
        raise ValueError(
            "authority must be preauthorized_non_widening or owner_approved"
        )
    if (
        isinstance(expected_lifecycle_version, bool)
        or not isinstance(expected_lifecycle_version, int)
        or expected_lifecycle_version < 0
    ):
        raise ValueError("expected_lifecycle_version must be a non-negative integer")
    reason = _bounded_lifecycle_text(reason, name="reason")
    actor = _bounded_lifecycle_text(actor, name="actor")
    gate_ref = _bounded_lifecycle_text(gate_ref, name="gate_ref", required=False)
    if gate_ref is not None:
        gate_ref = _lifecycle_reference(gate_ref, name="gate_ref")

    with write_txn(conn):
        row = _recommendation_row(conn, task_id)
        snapshot = recommendation_lifecycle_snapshot(row)
        if snapshot["lifecycle_version"] != expected_lifecycle_version:
            raise ValueError(
                "recommendation lifecycle version mismatch: "
                f"expected {expected_lifecycle_version}, found "
                f"{snapshot['lifecycle_version']}"
            )
        current = snapshot["decision"]
        allowed = {
            "pending": {"deferred", "rejected", "accepted"},
            "deferred": {"rejected", "accepted"},
        }
        if decision not in allowed.get(current, set()):
            raise ValueError(f"illegal recommendation decision transition {current} -> {decision}")

        evidence = parse_recommendation_evidence(row["recommendation_evidence"])
        if decision == "accepted":
            if evidence is None:
                raise ValueError("legacy recommendation without evidence cannot be accepted")
            if not gate_ref:
                raise ValueError("accepted recommendation requires gate_ref")
            widens = any(evidence["requested_scope"].values())
            if widens and authority != "owner_approved":
                raise ValueError(
                    "scope-widening recommendation requires owner_approved authority"
                )
            if authority == "preauthorized_non_widening" and widens:
                raise ValueError(
                    "preauthorized_non_widening requires every requested_scope flag false"
                )

        prior_task_ids, prior_run_ids, latest_at = (
            _recommendation_lifecycle_references(conn, task_id)
        )
        governance = _completed_work_run_evidence(
            conn,
            recommendation=row,
            task_id=governance_task_id,
            run_id=governance_run_id,
            role="governance",
            not_task_ids=prior_task_ids,
            not_run_ids=prior_run_ids,
            min_started_at=max(int(row["created_at"]), latest_at),
        )
        next_version = expected_lifecycle_version + 1
        cur = conn.execute(
            "UPDATE tasks SET recommendation_decision = ?, "
            "recommendation_effective_state = 'none', "
            "recommendation_lifecycle_version = ? "
            "WHERE id = ? AND task_kind = 'recommendation' "
            "AND COALESCE(recommendation_lifecycle_version, 0) = ?",
            (decision, next_version, task_id, expected_lifecycle_version),
        )
        if cur.rowcount != 1:
            raise ValueError("recommendation lifecycle version changed concurrently")
        payload = {
            "lifecycle_version": next_version,
            "decision": decision,
            "effective_state": "none",
            "authority": authority,
            "gate_ref": gate_ref,
            "reason": reason,
            "actor": actor,
            "governance_task_id": governance["task_id"],
            "governance_run_id": governance["run_id"],
        }
        _append_event(conn, task_id, "recommendation_decided", redact_review_value(payload))
        return {
            "recommendation_id": task_id,
            "decision": decision,
            "effective_state": "none",
            "lifecycle_version": next_version,
        }


_RECOMMENDATION_EFFECTIVE_TRANSITIONS = {
    "none": {"staged"},
    "staged": {"canary_running", "rolled_back"},
    "canary_running": {"verified", "rolled_back"},
    "verified": {"promoted", "rolled_back"},
    "promoted": {"revoked"},
}


def transition_recommendation(
    conn: sqlite3.Connection,
    task_id: str,
    *,
    effective_state: str,
    reason: str,
    actor: str,
    governance_task_id: str,
    governance_run_id: int,
    expected_lifecycle_version: int,
    native_surface: Optional[str] = None,
    config_identity: Optional[str] = None,
    rollback_identity: Optional[str] = None,
    readback_identity: Optional[str] = None,
    canary_task_id: Optional[str] = None,
    canary_run_id: Optional[int] = None,
    verifier_task_id: Optional[str] = None,
    verifier_run_id: Optional[int] = None,
) -> dict[str, Any]:
    """Record governed configuration evidence; never apply configuration."""
    _assert_recommendation_lifecycle_operator()
    if effective_state not in VALID_RECOMMENDATION_EFFECTIVE_STATES - {"none"}:
        raise ValueError("invalid recommendation effective_state")
    if (
        isinstance(expected_lifecycle_version, bool)
        or not isinstance(expected_lifecycle_version, int)
        or expected_lifecycle_version < 0
    ):
        raise ValueError("expected_lifecycle_version must be a non-negative integer")
    reason = _bounded_lifecycle_text(reason, name="reason")
    actor = _bounded_lifecycle_text(actor, name="actor")

    with write_txn(conn):
        row = _recommendation_row(conn, task_id)
        snapshot = recommendation_lifecycle_snapshot(row)
        if snapshot["lifecycle_version"] != expected_lifecycle_version:
            raise ValueError(
                "recommendation lifecycle version mismatch: "
                f"expected {expected_lifecycle_version}, found "
                f"{snapshot['lifecycle_version']}"
            )
        if snapshot["decision"] != "accepted":
            raise ValueError("only an accepted recommendation can enter configuration lifecycle")
        current = snapshot["effective_state"]
        if effective_state not in _RECOMMENDATION_EFFECTIVE_TRANSITIONS.get(current, set()):
            raise ValueError(
                f"illegal recommendation effective transition {current} -> {effective_state}"
            )

        prior_task_ids, prior_run_ids, latest_at = (
            _recommendation_lifecycle_references(conn, task_id)
        )
        floor = max(int(row["created_at"]), latest_at)
        governance = _completed_work_run_evidence(
            conn,
            recommendation=row,
            task_id=governance_task_id,
            run_id=governance_run_id,
            role="governance",
            not_task_ids=prior_task_ids,
            not_run_ids=prior_run_ids,
            min_started_at=floor,
        )
        payload: dict[str, Any] = {
            "lifecycle_version": expected_lifecycle_version + 1,
            "decision": "accepted",
            "effective_state": effective_state,
            "reason": reason,
            "actor": actor,
            "governance_task_id": governance["task_id"],
            "governance_run_id": governance["run_id"],
        }

        if effective_state == "staged":
            payload["native_surface"] = _lifecycle_reference(
                native_surface, name="native_surface"
            )
            config = _sha256_identity(config_identity, name="config_identity")
            rollback = _sha256_identity(
                rollback_identity, name="rollback_identity"
            )
            if config == rollback:
                raise ValueError(
                    "config_identity and rollback_identity must be distinct"
                )
            payload["config_identity"] = config
            payload["rollback_identity"] = rollback
        elif effective_state == "canary_running":
            canary = _active_canary_run_evidence(
                conn,
                recommendation=row,
                task_id=canary_task_id,
                run_id=canary_run_id,
                not_task_ids=prior_task_ids | {str(governance["task_id"])},
                not_run_ids=prior_run_ids | {int(governance["run_id"])},
                min_started_at=floor,
            )
            payload["canary_task_id"] = canary["task_id"]
            payload["canary_run_id"] = canary["run_id"]
        elif effective_state == "verified":
            canary_event = _recommendation_lifecycle_event(
                conn, task_id, effective_state="canary_running"
            )
            staged_event = _recommendation_lifecycle_event(
                conn, task_id, effective_state="staged"
            )
            if canary_event is None or staged_event is None:
                raise ValueError("verified state requires staged and canary_running evidence")
            canary_payload, _ = canary_event
            if (
                canary_task_id != canary_payload.get("canary_task_id")
                or canary_run_id != canary_payload.get("canary_run_id")
            ):
                raise ValueError("verified state must reference the recorded canary Task/run")
            canary = _completed_work_run_evidence(
                conn,
                recommendation=row,
                task_id=canary_task_id,
                run_id=canary_run_id,
                role="canary",
                not_task_ids={str(governance["task_id"])},
                not_run_ids={int(governance["run_id"])},
                min_started_at=staged_event[1],
            )
            if int(governance["ended_at"]) < int(canary["ended_at"]):
                raise ValueError(
                    "verified governance Task/run must end after the canary run"
                )
            verifier = _completed_work_run_evidence(
                conn,
                recommendation=row,
                task_id=verifier_task_id,
                run_id=verifier_run_id,
                role="verifier",
                not_task_ids=prior_task_ids
                | {
                    str(governance["task_id"]),
                    str(canary["task_id"]),
                },
                not_run_ids=prior_run_ids
                | {
                    int(governance["run_id"]),
                    int(canary["run_id"]),
                },
                min_started_at=max(
                    int(canary["ended_at"]), int(governance["ended_at"])
                ),
            )
            staged_payload, _ = staged_event
            readback = _sha256_identity(readback_identity, name="readback_identity")
            if readback != staged_payload.get("config_identity"):
                raise ValueError("verified readback_identity must equal config_identity")
            payload.update(
                {
                    "canary_task_id": canary["task_id"],
                    "canary_run_id": canary["run_id"],
                    "verifier_task_id": verifier["task_id"],
                    "verifier_run_id": verifier["run_id"],
                    "readback_identity": readback,
                }
            )
        elif effective_state in {"rolled_back", "revoked"}:
            staged_event = _recommendation_lifecycle_event(
                conn, task_id, effective_state="staged"
            )
            if staged_event is None:
                raise ValueError(f"{effective_state} requires staged evidence")
            staged_payload, _ = staged_event
            if current == "canary_running":
                canary_event = _recommendation_lifecycle_event(
                    conn, task_id, effective_state="canary_running"
                )
                if canary_event is None:
                    raise ValueError("rolled_back requires canary_running evidence")
                canary_payload, _ = canary_event
                canary = _completed_work_run_evidence(
                    conn,
                    recommendation=row,
                    task_id=canary_payload.get("canary_task_id"),
                    run_id=canary_payload.get("canary_run_id"),
                    role="canary",
                    not_task_ids={str(governance["task_id"])},
                    not_run_ids={int(governance["run_id"])},
                    min_started_at=staged_event[1],
                )
                if int(governance["ended_at"]) < int(canary["ended_at"]):
                    raise ValueError(
                        "rollback governance Task/run must end after the canary run"
                    )
            readback = _sha256_identity(readback_identity, name="readback_identity")
            if readback != staged_payload.get("rollback_identity"):
                raise ValueError(
                    f"{effective_state} readback_identity must equal rollback_identity"
                )
            verifier = _completed_work_run_evidence(
                conn,
                recommendation=row,
                task_id=verifier_task_id,
                run_id=verifier_run_id,
                role="verifier",
                not_task_ids=prior_task_ids | {str(governance["task_id"])},
                not_run_ids=prior_run_ids | {int(governance["run_id"])},
                min_started_at=int(governance["ended_at"]),
            )
            payload.update(
                {
                    "verifier_task_id": verifier["task_id"],
                    "verifier_run_id": verifier["run_id"],
                    "readback_identity": readback,
                }
            )

        next_version = expected_lifecycle_version + 1
        cur = conn.execute(
            "UPDATE tasks SET recommendation_effective_state = ?, "
            "recommendation_lifecycle_version = ? "
            "WHERE id = ? AND task_kind = 'recommendation' "
            "AND COALESCE(recommendation_lifecycle_version, 0) = ?",
            (effective_state, next_version, task_id, expected_lifecycle_version),
        )
        if cur.rowcount != 1:
            raise ValueError("recommendation lifecycle version changed concurrently")
        _append_event(
            conn,
            task_id,
            "recommendation_transitioned",
            redact_review_value(payload),
        )
        return {
            "recommendation_id": task_id,
            "decision": "accepted",
            "effective_state": effective_state,
            "lifecycle_version": next_version,
        }


def _find_missing_parents(conn: sqlite3.Connection, parents: Iterable[str]) -> list[str]:
    parents = list(parents)
    if not parents:
        return []
    placeholders = ",".join("?" * len(parents))
    rows = conn.execute(
        f"SELECT id FROM tasks WHERE id IN ({placeholders}) "
        f"AND task_kind IN {_DEPENDENCY_PARENT_KINDS}",
        parents,
    ).fetchall()
    present = {r["id"] for r in rows}
    return [p for p in parents if p not in present]


def _assert_dependency_child(conn: sqlite3.Connection, child_id: str) -> None:
    """A dependency edge's CHILD must be an executable work row.

    The two ends of an edge are not symmetric. A control anchor is a legitimate
    dependency PARENT — owner-approved work hangs under its Project's anchor
    and stays parked until the owner moves the anchor — but it is
    non-executable by construction, so it can never be the thing a parent
    releases. Validating both ends with the parent predicate admitted
    work-to-control and control-to-control edges, which would attach
    ``linked`` events and inherited notification subscriptions to a row no
    executable path is allowed to see. Checked BEFORE any row, event or
    subscription is written.
    """
    row = conn.execute(
        "SELECT task_kind FROM tasks WHERE id = ?", (child_id,)
    ).fetchone()
    if row is None:
        raise ValueError(f"unknown task(s): {child_id}")
    if row["task_kind"] != "work":
        raise ValueError(
            f"task {child_id} is a {row['task_kind']} row and cannot depend on "
            "another task; only executable work can be a dependency child"
        )


def _inherit_notify_subs(
    conn: sqlite3.Connection,
    child_id: str,
    parents: Iterable[str],
    *,
    created_at: Optional[int] = None,
) -> None:
    """Copy gateway notification subscriptions from parent tasks to a child.

    The inherited subscription starts caught up to the child's current event
    cursor. This makes manual `link_tasks(parent, existing_child)` safe: the
    parent chat receives future child terminal events without replaying the
    child's pre-link history.

    Copies EVERY routing/delivery column (chat_type, user_id_alt,
    delivery_mode, delivery_metadata included) — this helper is the single
    owner of subscription inheritance for create_task, link_tasks, and triage
    decomposition. Omitting columns here silently degrades routing: a
    DM-originated child completion falls back to chat_type='group' and wakes
    a fresh group-scoped session instead of the originating DM (issue #73030).
    """
    parent_ids = tuple(dict.fromkeys(p for p in parents if p))
    if not parent_ids:
        return
    row = conn.execute(
        "SELECT COALESCE(MAX(id), 0) AS cursor FROM task_events WHERE task_id = ?",
        (child_id,),
    ).fetchone()
    cursor = int(row["cursor"] if row is not None else 0)
    placeholders = ",".join("?" * len(parent_ids))
    conn.execute(
        f"""
        INSERT OR IGNORE INTO kanban_notify_subs
            (task_id, platform, chat_id, thread_id, user_id, user_id_alt,
             chat_type, notifier_profile, delivery_mode, delivery_metadata,
             created_at, last_event_id)
        SELECT ?, platform, chat_id, thread_id, user_id, user_id_alt,
               COALESCE(chat_type, 'dm'), notifier_profile,
               COALESCE(delivery_mode, 'notify'), delivery_metadata, ?, ?
          FROM kanban_notify_subs
         WHERE task_id IN ({placeholders})
        """,
        (
            child_id,
            int(created_at if created_at is not None else time.time()),
            cursor,
            *parent_ids,
        ),
    )


def get_task(
    conn: sqlite3.Connection, task_id: str, *, include_control: bool = False
) -> Optional[Task]:
    """Read one task. Executable ``work`` rows only, unless asked otherwise.

    ``include_control=True`` also resolves a non-executable ``control`` anchor.
    Only the owner-workspace kernel passes it, because only that kernel owns
    anchors; every executable path (dispatch, claim, spawn, specify, decompose,
    reassign) uses the default and therefore cannot see one.
    """
    row = conn.execute(
        f"SELECT * FROM tasks WHERE id = ? AND task_kind IN {_task_kinds(include_control)}",
        (task_id,),
    ).fetchone()
    return Task.from_row(row) if row else None


def get_control_task(conn: sqlite3.Connection, task_id: str) -> Optional[Task]:
    """Read a non-executable control anchor (``task_kind='control'``) only."""
    row = conn.execute(
        "SELECT * FROM tasks WHERE id = ? AND task_kind = 'control'", (task_id,)
    ).fetchone()
    return Task.from_row(row) if row else None


# Canonical sort-order mappings for ``hermes kanban list --sort``.
# Each value is a raw SQL fragment appended after ``ORDER BY``.
VALID_SORT_ORDERS: dict[str, str] = {
    "created": "created_at ASC, id ASC",
    "created-desc": "created_at DESC, id DESC",
    "priority": "priority DESC, created_at ASC",
    "priority-desc": "priority ASC, created_at ASC",
    "status": "status ASC, created_at ASC",
    "assignee": "assignee ASC, created_at ASC",
    "title": "title ASC, id ASC",
    "updated": "started_at DESC NULLS LAST, created_at DESC",
}


def list_tasks(
    conn: sqlite3.Connection,
    *,
    assignee: Optional[str] = None,
    status: Optional[str] = None,
    tenant: Optional[str] = None,
    session_id: Optional[str] = None,
    include_archived: bool = False,
    limit: Optional[int] = None,
    order_by: Optional[str] = None,
    workflow_template_id: Optional[str] = None,
    current_step_key: Optional[str] = None,
) -> list[Task]:
    query = "SELECT * FROM tasks WHERE task_kind = 'work'"
    params: list[Any] = []
    if assignee is not None:
        query += " AND assignee = ?"
        params.append(_canonical_assignee(assignee))
    if status is not None:
        if status not in VALID_STATUSES:
            raise ValueError(f"status must be one of {sorted(VALID_STATUSES)}")
        query += " AND status = ?"
        params.append(status)
    if tenant is not None:
        query += " AND tenant = ?"
        params.append(tenant)
    if session_id is not None:
        query += " AND session_id = ?"
        params.append(session_id)
    if workflow_template_id is not None:
        query += " AND workflow_template_id = ?"
        params.append(workflow_template_id)
    if current_step_key is not None:
        query += " AND current_step_key = ?"
        params.append(current_step_key)
    if not include_archived and status != "archived":
        query += " AND status != 'archived'"
    if order_by is not None:
        order_by = order_by.strip().lower()
        if order_by not in VALID_SORT_ORDERS:
            raise ValueError(
                f"order_by must be one of {sorted(VALID_SORT_ORDERS.keys())}"
            )
        query += f" ORDER BY {VALID_SORT_ORDERS[order_by]}"
    else:
        query += " ORDER BY priority DESC, created_at ASC"
    if limit:
        query += f" LIMIT {int(limit)}"
    rows = conn.execute(query, params).fetchall()
    return [Task.from_row(r) for r in rows]


def role_transition_route(
    conn: sqlite3.Connection,
    task_id: str,
    new_assignee: Optional[str],
    *,
    approved_route: Optional[dict] = None,
) -> tuple[list[tuple[str, Any]], Optional[dict]]:
    """Authorize one assignee write on a possibly policy-locked task.

    Every path that writes ``assignee`` — direct reassignment, unassignment,
    review handoff, rework handback (``request_changes``), specify, decompose,
    and the dispatcher's default assignee — goes through here rather than
    writing the column on its own, because a locked task's route authority is
    bound to the role that holds it. Centralising it is the point: a direct
    ``UPDATE tasks SET assignee`` elsewhere would mint nothing and leave the
    lock describing a role the task no longer has.

    Returns ``(assignments, repin)``: the extra ``(column, value)`` pairs the
    caller must include in the SAME ``UPDATE`` as its own ``assignee`` write,
    and the event payload to record when they are non-empty. For an unlocked
    task (every pre-existing, manual and CLI task) both are empty and nothing
    changes.

    For a LOCKED task every assignee change is refused. A task's approved
    assignee, provider, model, effort and tier are immutable for its whole run:
    re-deriving a route for whichever role happens to receive it would be
    exactly the silent re-pin the owner approval exists to prevent — including
    the internal review handoff and rework handback, which must be represented
    as separately approved review work instead. Unassigning a locked task is
    refused for the same reason: it would leave the lock bound to a role the
    row no longer names.

    An owner-GOVERNED row that carries no lock is refused just as hard. That is
    migrated owner work — receipt-owned on a board upgraded from a build with
    no route columns — and it can sit in ``scheduled``/``todo``, where the
    readiness guard and the rollout fence (both of which only look at
    executable rows) never reach it. Treating "no lock" as "ordinary card"
    there would let it be reassigned, unassigned or rerouted before anyone
    approved it again. Only ``approved_route`` gets it moving, and that
    installs its exact lock in the same write.

    ``approved_route`` is the one exception, and it is not a re-derivation: a
    fresh owner-approved mutation supplies the exact replacement
    ``{"assignee", "provider", "model", "reasoning_effort", "execution_tier",
    "model_policy_lock"}``, which is validated against the policy and installed
    atomically with the assignee write.

    Fails closed by raising ``RuntimeError`` on an unreadable, foreign, stale
    or incomplete lock, and on any replacement route that is not exactly
    authorized.
    """
    row = conn.execute(
        "SELECT assignee, execution_tier, model_policy_lock, model_override, "
        "provider_override, reasoning_effort, owner_receipt_bound FROM tasks "
        "WHERE id = ? AND task_kind = 'work'",
        (task_id,),
    ).fetchone()
    if row is None or not task_is_policy_governed(row):
        return [], None
    if not row["model_policy_lock"]:
        target = _canonical_assignee(new_assignee)
        if target == row["assignee"]:
            return [], None
        if approved_route is not None:
            return _approved_route_assignments(task_id, target, approved_route)
        raise RuntimeError(
            f"cannot change the role of owner-governed task {task_id}: it "
            "carries no approved route lock, so it must be approved again "
            "before any role change. A new owner-approved mutation must supply "
            "the exact replacement route."
        )
    lock_error = task_policy_lock_error(row)
    if lock_error:
        raise RuntimeError(f"task {task_id}: {lock_error}")
    target = _canonical_assignee(new_assignee)
    if target == row["assignee"]:
        return [], None

    if approved_route is not None:
        return _approved_route_assignments(task_id, target, approved_route)

    if target is None:
        raise RuntimeError(
            f"cannot unassign policy-locked task {task_id}: its owner-approved "
            "route names that role, so removing it would strand the lock. "
            "Approve a replacement route instead."
        )
    raise RuntimeError(
        f"cannot move policy-locked task {task_id} from "
        f"{row['assignee']!r} to {target!r}: the owner approved that exact "
        "assignee/provider/model/effort/tier for this task's whole run. A new "
        "owner-approved mutation must supply the exact replacement route, or "
        "the work must be represented as a separately approved task."
    )


def _approved_route_assignments(
    task_id: str, target: Optional[str], approved_route: dict
) -> tuple[list[tuple[str, Any]], Optional[dict]]:
    """Validate one owner-approved replacement route for a locked task."""
    required = {
        "assignee", "provider", "model", "reasoning_effort", "execution_tier",
        "model_policy_lock",
    }
    if not isinstance(approved_route, dict) or set(approved_route) != required:
        raise RuntimeError(
            f"task {task_id}: a replacement route must state exactly "
            f"{sorted(required)}"
        )
    approved_assignee = _canonical_assignee(approved_route["assignee"])
    if approved_assignee is None or approved_assignee != target:
        raise RuntimeError(
            f"task {task_id}: the replacement route names assignee "
            f"{approved_route['assignee']!r}, not {target!r}"
        )
    error = policy_lock_error(
        approved_route["model_policy_lock"],
        approved_assignee,
        approved_route["provider"],
        approved_route["model"],
        approved_route["reasoning_effort"],
        approved_route["execution_tier"],
    )
    if error:
        raise RuntimeError(f"task {task_id}: {error}")
    return (
        [
            ("model_override", approved_route["model"]),
            ("provider_override", approved_route["provider"]),
            ("reasoning_effort", approved_route["reasoning_effort"]),
            ("execution_tier", approved_route["execution_tier"]),
            ("model_policy_lock", approved_route["model_policy_lock"]),
        ],
        {
            "assignee": approved_assignee,
            "model": approved_route["model"],
            "provider": approved_route["provider"],
            "reasoning_effort": approved_route["reasoning_effort"],
            "execution_tier": approved_route["execution_tier"],
            "source": "owner_approved_replacement_route",
        },
    )


def _route_assignment_sql(
    assignments: list[tuple[str, Any]]
) -> tuple[str, tuple[Any, ...]]:
    """Render :func:`role_transition_route` pairs as a trailing SET fragment."""
    return (
        "".join(f", {column} = ?" for column, _ in assignments),
        tuple(value for _, value in assignments),
    )


def assign_task(
    conn: sqlite3.Connection,
    task_id: str,
    profile: Optional[str],
    *,
    approved_route: Optional[dict] = None,
) -> bool:
    """Assign or reassign a task.  Returns True on success.

    Refuses to reassign a task that's currently running (claim_lock set).
    Reassign after the current run completes if needed.

    A policy-locked task refuses every assignee change unless
    ``approved_route`` carries the exact owner-approved replacement route and
    lock, which are then installed in the SAME write — see
    :func:`role_transition_route`.
    """
    profile = _canonical_assignee(profile)
    with write_txn(conn):
        row = conn.execute(
            "SELECT status, claim_lock, assignee FROM tasks "
            "WHERE id = ? AND task_kind = 'work'",
            (task_id,),
        ).fetchone()
        if not row:
            return False
        if row["claim_lock"] is not None and row["status"] == "running":
            raise RuntimeError(
                f"cannot reassign {task_id}: currently running (claimed). "
                "Wait for completion or reclaim the stale lock first."
            )
        assignments, repin = role_transition_route(
            conn, task_id, profile, approved_route=approved_route
        )
        route_sql, route_params = _route_assignment_sql(assignments)
        if row["assignee"] != profile:
            # The retry guard is scoped to the task/profile combination. A
            # human reassigning the task is an explicit recovery action, so the
            # new profile should not inherit the previous profile's streak.
            conn.execute(
                "UPDATE tasks SET assignee = ?, consecutive_failures = 0, "
                "last_failure_error = NULL" + route_sql + " WHERE id = ?",
                (profile, *route_params, task_id),
            )
        else:
            conn.execute(
                "UPDATE tasks SET assignee = ?" + route_sql + " WHERE id = ?",
                (profile, *route_params, task_id),
            )
        _append_event(conn, task_id, "assigned", {"assignee": profile})
        if repin is not None:
            _append_event(conn, task_id, "model_route_repinned", repin)
    # Task-mutation observer (RFC #58548), fired AFTER the assignment txn
    # has committed so subscribers always observe durable board state.
    notify_task_updated(conn, task_id, ("assignee",))
    return True


def set_model_override(
    conn: sqlite3.Connection,
    task_id: str,
    model: Optional[str],
    provider: Optional[str] = None,
) -> bool:
    """Set (or clear) the per-task model/provider override.

    ``model=None`` (or empty) clears BOTH overrides — the worker falls back
    to its profile's configured model. ``provider`` without ``model`` is
    rejected: a bare provider switch has no defined meaning for the worker
    spawn (``--provider`` alone would re-resolve the profile's model name
    against a different backend, which is exactly the mismatch class this
    feature exists to kill).

    Allowed on any non-archived task, including ``running`` ones — the
    override only takes effect on the NEXT dispatch, so setting it on a
    running task that's about to be reclaimed/retried is the primary
    rate-limit-recovery flow. Returns True on success.

    An owner-governed task is refused: its route was approved by the owner for
    that exact task and must never change under it. That covers a task carrying
    ``model_policy_lock`` and migrated owner work that carries none — rerouting
    the latter would silently redefine the route its re-approval is supposed to
    install.
    """
    model = (model or "").strip() or None
    provider = (provider or "").strip() or None
    if provider and not model:
        raise ValueError("provider_override requires a model_override")
    if not model:
        provider = None
    with write_txn(conn):
        row = conn.execute(
            "SELECT status, execution_tier, model_policy_lock, owner_receipt_bound "
            "FROM tasks WHERE id = ? AND task_kind = 'work'",
            (task_id,),
        ).fetchone()
        if not row:
            return False
        if row["status"] == "archived":
            raise RuntimeError(f"cannot set model override on archived task {task_id}")
        if task_is_policy_governed(row):
            raise RuntimeError(
                f"cannot change the model override of owner-governed task {task_id}"
            )
        conn.execute(
            "UPDATE tasks SET model_override = ?, provider_override = ? WHERE id = ?",
            (model, provider, task_id),
        )
        _append_event(
            conn, task_id, "model_override_set",
            {"model": model, "provider": provider},
        )
    # Task-mutation observer (RFC #58548), fired AFTER the txn commits.
    notify_task_updated(conn, task_id, ("model_override", "provider_override"))
    return True


def set_reasoning_effort(
    conn: sqlite3.Connection,
    task_id: str,
    effort: Optional[str],
) -> bool:
    """Set (or clear) the per-task reasoning effort.

    ``effort=None`` (or empty) clears the override — the worker falls back to
    its profile's own ``agent.reasoning_effort``. ``"none"`` is a real value,
    not a clear: it pins thinking OFF for this task.

    Deliberately independent of :func:`set_model_override`: a task may run the
    profile's own model at a different depth, and clearing a model override
    must not silently reset the depth the operator chose. Like the model
    override, it takes effect on the NEXT dispatch, so it is settable on a
    running task. Returns True on success.

    Like :func:`set_model_override`, an owner-governed task is refused: the
    owner approved that exact depth alongside the model.
    """
    effort = normalize_reasoning_effort(effort)
    with write_txn(conn):
        row = conn.execute(
            "SELECT status, execution_tier, model_policy_lock, owner_receipt_bound "
            "FROM tasks WHERE id = ? AND task_kind = 'work'",
            (task_id,),
        ).fetchone()
        if not row:
            return False
        if row["status"] == "archived":
            raise RuntimeError(
                f"cannot set reasoning effort on archived task {task_id}"
            )
        if task_is_policy_governed(row):
            raise RuntimeError(
                f"cannot change the reasoning effort of owner-governed task {task_id}"
            )
        conn.execute(
            "UPDATE tasks SET reasoning_effort = ? WHERE id = ?",
            (effort, task_id),
        )
        _append_event(
            conn, task_id, "reasoning_effort_set", {"reasoning_effort": effort}
        )
    # Task-mutation observer (RFC #58548), fired AFTER the txn commits.
    notify_task_updated(conn, task_id, ("reasoning_effort",))
    return True


def _exposed_task_filter(task_ids: Iterable[str]) -> Optional[tuple[str, tuple]]:
    """WHERE clause for named executable tasks that carry no route lock.

    Every executable owner task without a lock is exposed, whether or not it
    already names a model, provider and effort: naming all three makes a route
    fixed, but not owner-approved, and an unlocked owner task must never be
    dispatchable as ordinary work. Already locked, finished, archived and
    non-``work`` rows are excluded, so the caller's candidate list can only
    ever narrow.
    """
    ids = [str(task_id) for task_id in task_ids if str(task_id or "").strip()]
    if not ids:
        return None
    placeholders = ",".join("?" for _ in ids)
    return (
        f"id IN ({placeholders}) AND task_kind = 'work' "
        "AND status NOT IN ('done', 'archived') "
        "AND model_policy_lock IS NULL",
        tuple(ids),
    )


def count_unpinned_owner_tasks(
    conn: sqlite3.Connection, *, task_ids: Iterable[str]
) -> int:
    """Count named executable tasks that carry no owner-approved route lock."""
    selector = _exposed_task_filter(task_ids)
    if selector is None:
        return 0
    predicate, params = selector
    row = conn.execute(
        f"SELECT COUNT(*) AS n FROM tasks WHERE {predicate}", params
    ).fetchone()
    return int(row["n"]) if row else 0


def list_active_unpinned_owner_tasks(
    conn: sqlite3.Connection, *, task_ids: Iterable[str]
) -> list[str]:
    """Exposed owner tasks that are already claimed, running, or mid-run.

    A settings change cannot safely re-pin one of these: a worker is already
    executing (or about to execute) under the authority the row carries right
    now, and neither rewriting its route columns underneath it nor releasing
    its claim would stop the process that is already running. The fence
    therefore refuses the change while any of them is active, rather than
    letting a stale claim continue under changed authority.

    Deliberately conservative — an expired-but-unreleased claim lock and a
    dangling ``current_run_id`` both count, because either one can still be
    adopted or reported against.
    """
    selector = _exposed_task_filter(task_ids)
    if selector is None:
        return []
    predicate, params = selector
    rows = conn.execute(
        f"SELECT id FROM tasks WHERE {predicate} AND ("
        "status = 'running' OR claim_lock IS NOT NULL "
        "OR worker_pid IS NOT NULL OR current_run_id IS NOT NULL) "
        "ORDER BY id",
        params,
    ).fetchall()
    return [str(row["id"]) for row in rows]


# The block reason a fenced-but-unpinnable owner task lands on, and the comment
# the board shows for it. ``needs_input`` is the native kind that means "a human
# decision is required before this can run again" — exactly the state of a task
# whose route nobody can prove was approved.
_UNPINNABLE_BLOCK_KIND = "needs_input"
_UNPINNABLE_BLOCK_REASON = (
    "This work is paused because its approved model route cannot be confirmed. "
    "It needs to be approved again before it can run."
)


def pin_effective_task_routes(
    conn: sqlite3.Connection,
    *,
    task_ids: Iterable[str],
    model: str,
    provider: str,
    reasoning_effort: str,
) -> list[str]:
    """Pin exposed owner tasks onto an exact owner-approved route, or pause them.

    For each exposed task the effective route is completed from the profile's
    current route only where the task itself declares nothing: an explicit
    model, provider or effort the operator already set is preserved exactly.
    The result is written per task, so a settings change afterwards cannot
    move any of them.

    A task whose completed route IS an admitted authority for its assignee and
    tier gets the durable policy lock. A task whose route the policy does not
    admit — a legacy row that was never classified, a partial pin, a hand-set
    override, or a forbidden route — cannot be given authority at all, so it is
    parked in the native ``blocked`` column with ``block_kind='needs_input'``
    and a plain-English re-approval requirement. Leaving it runnable would let
    receipt-owned work dispatch as ordinary work on a route nobody approved.

    Returns the ids that were pinned (locked); paused ids are reported through
    the task's own state and audit trail.
    """
    selector = _exposed_task_filter(task_ids)
    if selector is None:
        return []
    predicate, params = selector
    profile_effort = normalize_reasoning_effort(reasoning_effort)
    profile_model = (model or "").strip()
    profile_provider = (provider or "").strip()
    if not profile_model or not profile_provider or not profile_effort:
        raise ValueError(
            "cannot fence tasks onto an incomplete profile route "
            f"(model={profile_model or None!r}, provider={profile_provider or None!r}, "
            f"reasoning_effort={profile_effort or None!r})"
        )

    pinned: list[str] = []
    paused: list[str] = []
    with write_txn(conn):
        rows = conn.execute(
            "SELECT id, status, assignee, execution_tier, model_override, "
            f"provider_override, reasoning_effort FROM tasks WHERE {predicate}",
            params,
        ).fetchall()
        for row in rows:
            task_id = str(row["id"])
            effective_model = row["model_override"] or profile_model
            effective_provider = row["provider_override"] or profile_provider
            effective_effort = row["reasoning_effort"] or profile_effort
            try:
                lock = mint_policy_lock(
                    row["assignee"],
                    effective_provider,
                    effective_model,
                    effective_effort,
                    row["execution_tier"],
                )
            except ValueError as exc:
                _pause_unpinnable_task(conn, task_id, str(exc))
                paused.append(task_id)
                continue
            conn.execute(
                "UPDATE tasks SET model_override = ?, provider_override = ?, "
                "reasoning_effort = ?, model_policy_lock = ? WHERE id = ?",
                (
                    effective_model,
                    effective_provider,
                    effective_effort,
                    lock,
                    task_id,
                ),
            )
            pinned.append(task_id)
            _append_event(
                conn,
                task_id,
                "model_route_pinned",
                {
                    "model": effective_model,
                    "provider": effective_provider,
                    "reasoning_effort": effective_effort,
                    "execution_tier": row["execution_tier"],
                    "policy_locked": True,
                    "source": "effective_route_rollout_fence",
                },
            )
    for task_id in pinned:
        notify_task_updated(
            conn,
            task_id,
            ("model_override", "provider_override", "reasoning_effort"),
        )
    for task_id in paused:
        notify_task_updated(conn, task_id, ("status",))
    return pinned


def _pause_unpinnable_task(
    conn: sqlite3.Connection, task_id: str, detail: str
) -> None:
    """Park one receipt-owned task that has no provable approved route.

    Writes the native blocked state directly (rather than through
    :func:`block_task`) because the fence must be able to pause a task in ANY
    pre-run column — ``triage``, ``todo``, ``ready`` — while it already holds
    the fence's write transaction, and because the run bookkeeping
    ``block_task`` performs only applies to a task that was actually running.
    A running task's claim is released so its worker cannot report back into a
    route that was never approved.
    """
    conn.execute(
        "UPDATE tasks SET status = 'blocked', block_kind = ?, "
        "claim_lock = NULL, claim_expires = NULL, worker_pid = NULL "
        "WHERE id = ? AND task_kind = 'work'",
        (_UNPINNABLE_BLOCK_KIND, task_id),
    )
    payload = {
        "kind": _UNPINNABLE_BLOCK_KIND,
        "reason": _UNPINNABLE_BLOCK_REASON,
        "reapproval_required": True,
        "source": "effective_route_rollout_fence",
    }
    # A ``blocked`` event is what makes the pause STICKY: ``recompute_ready``
    # refuses to auto-promote a task whose latest block event is an explicit
    # one (see ``_has_sticky_block``), so nothing puts this back in the work
    # pool until the route is approved again.
    _append_event(conn, task_id, "blocked", payload)
    _append_event(
        conn, task_id, "model_route_unapproved", {**payload, "detail": detail},
    )


# Columns the readiness guard needs to prove one task's route authority.
_ROUTE_AUTHORITY_SELECT = (
    "SELECT id, assignee, provider_override, model_override, reasoning_effort, "
    "execution_tier, model_policy_lock, owner_receipt_bound FROM tasks "
    "WHERE id = ? AND task_kind = 'work'"
)

# Statuses from which the dispatcher will actually start a run.
EXECUTABLE_STATUSES = frozenset({"ready", "review"})

# Where work waits while an owner approval is still writing its durable
# receipt. ``scheduled`` is the one native non-terminal column that neither
# ``recompute_ready`` (which only ever scans ``todo``/``blocked``) nor any
# claim path will move on its own, so a row parked here cannot be promoted or
# claimed by a live dispatcher tick, and no new state machine is needed.
PARKED_STATUS = "scheduled"


def park_generation(*, actor: str, profile: str, idempotency_key: str) -> str:
    """The durable identity of one owner operation's parking.

    A pure function of the operation's own identity, so a replay of the exact
    same receipt recomputes the exact same generation and can finish an
    activation that was interrupted. Two different operations — including two
    parkings of the same task — never share one.
    """
    raw = f"{actor}\0{profile}\0{idempotency_key}".encode("utf-8")
    return "owpark_" + hashlib.sha256(raw).hexdigest()[:40]


def _require_park_generation(generation: Any) -> str:
    """Reject a parking generation that could match the wrong rows."""
    text = str(generation or "").strip()
    if not text or len(text) > 64:
        raise ValueError("a parking generation is required")
    return text


def activate_owner_work(
    conn: sqlite3.Connection,
    task_ids: Iterable[str],
    *,
    generation: str,
    restore_statuses: Optional[dict[str, str]] = None,
) -> list[str]:
    """Release exactly this receipt's parked work, then recompute readiness.

    The single transition an owner approval performs AFTER its terminal
    receipt is durable. It is driven by the exact task ids the committed
    receipt records AND by ``generation`` — the durable identity of the
    parking that receipt performed (:func:`park_generation`).

    The generation is what makes replay exact rather than merely idempotent.
    :data:`PARKED_STATUS` is a SHARED column: an owner also postpones work
    there deliberately. Matching on the column alone meant that replaying an
    already-completed receipt after the owner postponed one of its tasks
    reactivated that task — and, through ``recompute_ready``, could enable its
    dependents. Activation clears the generation as it releases the row, so a
    replay compare-and-swaps a generation no row carries any more and changes
    nothing.

    Landing on ``todo`` rather than ``ready`` is deliberate: promotion into an
    executable column must go through :func:`recompute_ready`, which proves
    each task's route authority and honours dependency order, instead of this
    function writing an executable status directly.

    ``restore_statuses`` names the exact column a task must return to when
    ``todo`` is not where it came from. A dependent that this plan parked out
    of ``blocked`` goes back to ``blocked``, so ``recompute_ready`` re-applies
    the sticky-block and circuit-breaker guards it would have applied all
    along; parking must never launder a blocked row into the work pool.
    """
    generation = _require_park_generation(generation)
    ids = [str(task_id) for task_id in task_ids if str(task_id or "").strip()]
    if not ids:
        return []
    restore = dict(restore_statuses or {})
    released: list[str] = []
    with write_txn(conn):
        for task_id in ids:
            target = restore.get(task_id, "todo")
            if target not in VALID_STATUSES or target == PARKED_STATUS:
                raise ValueError(f"cannot restore {task_id} to status {target!r}")
            cur = conn.execute(
                "UPDATE tasks SET status = ?, park_generation = NULL "
                "WHERE id = ? AND task_kind = 'work' AND status = ? "
                "AND park_generation = ?",
                (target, task_id, PARKED_STATUS, generation),
            )
            if cur.rowcount == 1:
                released.append(task_id)
                _append_event(conn, task_id, "owner_work_activated", None)
    recompute_ready(conn)
    return released


def _park_newly_enabled_dependents(
    conn: sqlite3.Connection,
    *,
    parent_ids_made_terminal: Iterable[str],
    already_parked: set[str],
    generation: str,
) -> list[list[str]]:
    """Park the work a plan just unblocked, inside the plan's own transaction.

    Archiving or merging a parent satisfies its children's dependency the
    instant the plan transaction commits — which is BEFORE the owner receipt
    that authorizes the plan is durable. Recomputing readiness there, or a
    dispatcher tick landing in that gap, would promote an already pinned child
    into a claimable column for a plan whose receipt may still fail to
    finalize. So every dependent the plan newly enables is moved into
    :data:`PARKED_STATUS` here, in the same transaction as the archive, and
    reported so the committed receipt can name it.

    Only a child whose parents are now ALL terminal is touched: one still
    waiting on the plan's own parked replacements is not newly enabled and is
    left exactly as it is. A sticky-blocked child is left alone too — nothing
    auto-promotes it, and parking would drop an explicit operator hold.

    Returns ``[[task_id, status_to_restore], ...]``, JSON-shaped so it round
    trips through the receipt unchanged. ``generation`` is stamped on each
    parked row so only this operation's own activation can release it (see
    :func:`activate_owner_work`).
    """
    generation = _require_park_generation(generation)
    parked: dict[str, str] = {}
    for parent_id in dict.fromkeys(str(pid) for pid in parent_ids_made_terminal):
        for child_id in child_ids(conn, parent_id):
            if child_id in already_parked or child_id in parked:
                continue
            row = conn.execute(
                "SELECT status FROM tasks WHERE id = ? AND task_kind = 'work'",
                (child_id,),
            ).fetchone()
            if row is None or row["status"] not in ("todo", "blocked"):
                continue
            if not _parents_satisfied(conn, child_id):
                continue
            if row["status"] == "blocked" and _has_sticky_block(conn, child_id):
                continue
            conn.execute(
                "UPDATE tasks SET status = ?, park_generation = ? "
                "WHERE id = ? AND task_kind = 'work' AND status = ?",
                (PARKED_STATUS, generation, child_id, row["status"]),
            )
            _append_event(
                conn, child_id, "owner_work_parked",
                {"restore_status": row["status"], "unblocked_by": parent_id},
            )
            parked[child_id] = row["status"]
    return [[task_id, status] for task_id, status in sorted(parked.items())]


def task_is_policy_governed(row: Any) -> bool:
    """Whether this row's route is owned by the owner model policy.

    ``execution_tier`` and ``model_policy_lock`` are written only by the
    owner-approved creation paths and by the route fence; an ordinary, manual
    or CLI task carries neither, which is why it keeps its historical
    unrestricted behaviour. A row that carries one but not a valid other is a
    partially pinned owner row — governed, but not yet provable.

    ``owner_receipt_bound`` is the third, and the one that closes the
    pre-upgrade hole: work a committed owner receipt owns is governed by that
    ownership alone, even when it carries NEITHER a tier nor a lock because it
    was created before those columns existed. Inferring "safe to run as
    ordinary work" from those NULLs is exactly what must not happen.
    """
    for column in ("execution_tier", "model_policy_lock", "owner_receipt_bound"):
        try:
            if row[column]:
                return True
        except (IndexError, KeyError, TypeError):
            continue
    return False


def route_authority_error(
    conn: sqlite3.Connection, task_id: str
) -> Optional[str]:
    """Why this task's approved route cannot be proven, or ``None``.

    The read-only half of :func:`authorize_executable_transition`: it answers
    the same question against the same five bound columns, but writes nothing
    — no minted lock, no parking, no event. That is what a dry run and a
    pre-claim screen need, so a caller can report or defer a task whose route
    authority does not hold instead of discovering it by having the real
    operation raise mid-tick.
    """
    row = conn.execute(_ROUTE_AUTHORITY_SELECT, (task_id,)).fetchone()
    if row is None or not task_is_policy_governed(row):
        return None
    if row["model_policy_lock"]:
        return task_policy_lock_error(row) or None
    try:
        mint_policy_lock(
            row["assignee"],
            row["provider_override"],
            row["model_override"],
            row["reasoning_effort"],
            row["execution_tier"],
        )
    except ValueError as exc:
        return str(exc)
    return None


def authorize_executable_transition(
    conn: sqlite3.Connection, task_id: str, *, park: bool = True
) -> bool:
    """Prove a task's exact admitted route before it becomes runnable.

    An owner-governed work row must be EXACTLY locked before it can sit in a
    column the dispatcher spawns from. Reaching ``ready``/``review`` without a
    lock — a legacy row created before locks existed, a partial pin, a row the
    route fence never saw because no settings change happened — would let
    receipt-owned work run as ordinary work on a route nobody approved.

    So, inside the caller's own write transaction (atomic with the status
    write it is about to make):

    * an ordinary/manual/CLI task, which carries no owner route fields at all,
      is allowed through unchanged;
    * a locked row is re-validated against this build's policy, and passes only
      when the lock still binds its exact assignee/provider/model/effort/tier;
    * an unlocked governed row has its exact admitted lock MINTED from its own
      completed route, so a legacy row that is genuinely on an approved route
      simply becomes provable (replaying this is a no-op: the digest is a pure
      function of the same five fields);
    * anything that cannot be proven or minted is refused, and — when ``park``
      is set — parked in the native ``blocked`` column with
      ``block_kind='needs_input'`` and a plain-English re-approval requirement,
      exactly as the route fence parks an unpinnable task.

    ``park=False`` is for a caller that reports a conflict snapshot and may
    itself abort the transaction; parking there would be rolled back anyway.
    Returns whether the transition may proceed.
    """
    row = conn.execute(_ROUTE_AUTHORITY_SELECT, (task_id,)).fetchone()
    if row is None or not task_is_policy_governed(row):
        return True
    if row["model_policy_lock"]:
        error = task_policy_lock_error(row)
        if not error:
            return True
    else:
        try:
            lock = mint_policy_lock(
                row["assignee"],
                row["provider_override"],
                row["model_override"],
                row["reasoning_effort"],
                row["execution_tier"],
            )
        except ValueError as exc:
            error = str(exc)
        else:
            conn.execute(
                "UPDATE tasks SET model_policy_lock = ? "
                "WHERE id = ? AND task_kind = 'work' AND model_policy_lock IS NULL",
                (lock, task_id),
            )
            _append_event(
                conn,
                task_id,
                "model_route_pinned",
                {
                    "model": row["model_override"],
                    "provider": row["provider_override"],
                    "reasoning_effort": row["reasoning_effort"],
                    "execution_tier": row["execution_tier"],
                    "policy_locked": True,
                    "source": "executable_readiness_guard",
                },
            )
            return True
    if park:
        _pause_unpinnable_task(conn, task_id, error)
    return False


# ---------------------------------------------------------------------------
# Links
# ---------------------------------------------------------------------------

def link_tasks(conn: sqlite3.Connection, parent_id: str, child_id: str) -> None:
    with write_txn(conn):
        _link_tasks_in_txn(conn, parent_id, child_id)


def _link_tasks_in_txn(conn: sqlite3.Connection, parent_id: str, child_id: str) -> bool:
    """Link two tasks while the caller already owns the write transaction.

    Returns ``True`` only when a new edge was inserted. Keeping the mutation
    in one helper lets bounded graph builders compose several dependency
    changes atomically without bypassing cycle checks, ready-state demotion,
    audit events, or notification inheritance.
    """
    if parent_id == child_id:
        raise ValueError("a task cannot depend on itself")
    missing = _find_missing_parents(conn, [parent_id])
    if missing:
        raise ValueError(f"unknown task(s): {', '.join(missing)}")
    _assert_dependency_child(conn, child_id)
    if _would_cycle(conn, parent_id, child_id):
        raise ValueError(
            f"linking {parent_id} -> {child_id} would create a cycle"
        )
    cur = conn.execute(
        "INSERT OR IGNORE INTO task_links (parent_id, child_id) VALUES (?, ?)",
        (parent_id, child_id),
    )
    if cur.rowcount != 1:
        return False
    # If child was ready but parent is not yet done, demote child to todo.
    parent_status = conn.execute(
        f"SELECT status FROM tasks WHERE id = ? AND task_kind IN {_DEPENDENCY_PARENT_KINDS}",
        (parent_id,),
    ).fetchone()["status"]
    if parent_status not in {"done", "archived"}:
        conn.execute(
            "UPDATE tasks SET status = 'todo' "
            "WHERE id = ? AND status = 'ready' AND task_kind = 'work'",
            (child_id,),
        )
    _append_event(
        conn, child_id, "linked",
        {"parent": parent_id, "child": child_id},
    )
    _inherit_notify_subs(conn, child_id, (parent_id,))
    return True


def _would_cycle(conn: sqlite3.Connection, parent_id: str, child_id: str) -> bool:
    """Return True if adding parent->child creates a cycle.

    A cycle exists iff ``parent_id`` is already a descendant of
    ``child_id`` via existing parent->child links.  We walk downward
    from ``child_id`` and check whether we reach ``parent_id``.
    """
    seen = set()
    stack = [child_id]
    while stack:
        node = stack.pop()
        if node == parent_id:
            return True
        if node in seen:
            continue
        seen.add(node)
        rows = conn.execute(
            "SELECT child_id FROM task_links WHERE parent_id = ?", (node,)
        ).fetchall()
        stack.extend(r["child_id"] for r in rows)
    return False


def unlink_tasks(conn: sqlite3.Connection, parent_id: str, child_id: str) -> bool:
    with write_txn(conn):
        cur = conn.execute(
            "DELETE FROM task_links WHERE parent_id = ? AND child_id = ?",
            (parent_id, child_id),
        )
        if cur.rowcount:
            _append_event(
                conn, child_id, "unlinked",
                {"parent": parent_id, "child": child_id},
            )
        removed = cur.rowcount > 0
    if removed:
        # Dependency edge removed — re-evaluate promotion eligibility for the
        # child immediately.  Matches the contract of complete_task and
        # unblock_task; without this the child stays stuck in todo until the
        # next dispatcher tick or a manual `hermes kanban recompute` (issue #22459).
        recompute_ready(conn)
    return removed


def parent_ids(conn: sqlite3.Connection, task_id: str) -> list[str]:
    rows = conn.execute(
        "SELECT parent_id FROM task_links WHERE child_id = ? ORDER BY parent_id",
        (task_id,),
    ).fetchall()
    return [r["parent_id"] for r in rows]


def child_ids(conn: sqlite3.Connection, task_id: str) -> list[str]:
    rows = conn.execute(
        "SELECT child_id FROM task_links WHERE parent_id = ? ORDER BY child_id",
        (task_id,),
    ).fetchall()
    return [r["child_id"] for r in rows]


def task_graph_contexts(
    conn: sqlite3.Connection, task_ids: Iterable[str]
) -> dict[str, dict]:
    """Bulk-load compact direct graph state for graph-aware diagnostics."""
    ordered_ids = list(dict.fromkeys(str(task_id) for task_id in task_ids if task_id))
    contexts = {
        task_id: {"parents": [], "children": []}
        for task_id in ordered_ids
    }
    if not ordered_ids:
        return contexts

    placeholders = ",".join("?" for _ in ordered_ids)
    for row in conn.execute(
        "SELECT l.child_id AS owner_id, t.id, t.title, t.status "
        "FROM task_links l JOIN tasks t ON t.id = l.parent_id "
        f"WHERE l.child_id IN ({placeholders}) AND t.task_kind = 'work' "
        "ORDER BY l.child_id, t.id",
        tuple(ordered_ids),
    ).fetchall():
        contexts[row["owner_id"]]["parents"].append({
            "id": row["id"],
            "title": row["title"],
            "status": row["status"],
        })
    for row in conn.execute(
        "SELECT l.parent_id AS owner_id, t.id, t.title, t.status "
        "FROM task_links l JOIN tasks t ON t.id = l.child_id "
        f"WHERE l.parent_id IN ({placeholders}) AND t.task_kind = 'work' "
        "ORDER BY l.parent_id, t.id",
        tuple(ordered_ids),
    ).fetchall():
        contexts[row["owner_id"]]["children"].append({
            "id": row["id"],
            "title": row["title"],
            "status": row["status"],
        })
    return contexts


def task_graph_context(conn: sqlite3.Connection, task_id: str) -> dict:
    """Return compact direct parent/child state for one task."""
    return task_graph_contexts(conn, [task_id])[task_id]


def parent_results(conn: sqlite3.Connection, task_id: str) -> list[tuple[str, Optional[str]]]:
    """Return ``(parent_id, result)`` for every done parent of ``task_id``."""
    rows = conn.execute(
        """
        SELECT t.id AS id, t.result AS result
        FROM tasks t
        JOIN task_links l ON l.parent_id = t.id
        WHERE l.child_id = ? AND t.status = 'done' AND t.task_kind = 'work'
        ORDER BY t.completed_at ASC
        """,
        (task_id,),
    ).fetchall()
    return [(r["id"], r["result"]) for r in rows]


# ---------------------------------------------------------------------------
# Comments & events
# ---------------------------------------------------------------------------

def add_comment(
    conn: sqlite3.Connection, task_id: str, author: str, body: str,
    *, operation_key: Optional[str] = None, include_control: bool = False,
) -> int:
    """Append a comment (+ a ``"commented"`` event) and return the comment id.

    ``operation_key``, when given, makes the call idempotent across a
    crash-and-retry: a prior call with the SAME ``(task_id, operation_key)``
    returns the ORIGINAL comment's id without inserting a second comment or
    appending a second event (enforced by a partial unique index — see
    ``idx_comments_operation_key`` in ``_migrate_add_optional_columns``). A
    retry whose ``author``/``body`` disagree with the original fails closed
    instead of silently returning a mismatched comment — callers (e.g. the
    owner-workspace kernel) that need conflict detection across the exact
    payload rely on this; their own idempotency-key digest check already
    rejects a differing payload before ever reaching here, so this is
    defense in depth, not the primary guard.
    """
    if not body or not body.strip():
        raise ValueError("comment body is required")
    if not author or not author.strip():
        raise ValueError("comment author is required")
    author = author.strip()
    body = body.strip()
    now = int(time.time())
    # ``allow_nested=True``: graph builders (kanban_swarm blackboard seeding)
    # compose comment writes under one outer commit.
    with write_txn(conn, allow_nested=True):
        if not conn.execute(
            "SELECT 1 FROM tasks WHERE id = ? AND task_kind IN "
            f"{_task_kinds(include_control)}",
            (task_id,),
        ).fetchone():
            raise ValueError(f"unknown task {task_id}")
        if operation_key:
            existing = conn.execute(
                "SELECT id, author, body FROM task_comments "
                "WHERE task_id = ? AND operation_key = ?",
                (task_id, operation_key),
            ).fetchone()
            if existing is not None:
                if existing["author"] != author or existing["body"] != body:
                    raise ValueError(
                        f"operation_key {operation_key!r} was already used with a "
                        "different author/body"
                    )
                return int(existing["id"])
        cur = conn.execute(
            "INSERT INTO task_comments (task_id, author, body, created_at, operation_key) "
            "VALUES (?, ?, ?, ?, ?)",
            (task_id, author, body, now, operation_key),
        )
        _append_event(conn, task_id, "commented", {"author": author, "len": len(body)})
        return int(cur.lastrowid or 0)


def list_comments(
    conn: sqlite3.Connection, task_id: str, *, include_control: bool = False
) -> list[Comment]:
    """List a task's comments. Executable ``work`` rows only, unless asked.

    ``include_control=True`` mirrors :func:`add_comment` / :func:`list_events`:
    the owner-workspace kernel comments on a Project's control anchor, so the
    same kernel (and its tests) must be able to read that thread back. Every
    executable surface — the dashboard, the CLI, the worker bridge — uses the
    default and therefore never sees an anchor's comments.
    """
    rows = conn.execute(
        "SELECT * FROM task_comments WHERE task_id = ? "
        "AND task_id IN (SELECT id FROM tasks WHERE task_kind IN "
        f"{_task_kinds(include_control)}) "
        "ORDER BY created_at ASC",
        (task_id,),
    ).fetchall()
    return [
        Comment(
            id=r["id"],
            task_id=r["task_id"],
            author=r["author"],
            body=r["body"],
            created_at=r["created_at"],
        )
        for r in rows
    ]


def list_comments_after(
    conn: sqlite3.Connection, task_id: str, *, after_id: int = 0
) -> list[Comment]:
    """Return comments on ``task_id`` with ``id > after_id`` (ascending).

    Keyed on the monotonic rowid rather than ``created_at`` so a same-second
    burst can't be skipped. Used by the live worker bridge to fold new
    operator notes into a running task without a restart (see
    ``tools.kanban_tools.inject_new_comments_from_env``).
    """
    rows = conn.execute(
        "SELECT id, task_id, author, body, created_at FROM task_comments "
        "WHERE task_id = ? AND id > ? "
        "AND task_id IN (SELECT id FROM tasks WHERE task_kind = 'work') "
        "ORDER BY id ASC",
        (task_id, int(after_id)),
    ).fetchall()
    return [
        Comment(
            id=r["id"],
            task_id=r["task_id"],
            author=r["author"],
            body=r["body"],
            created_at=r["created_at"],
        )
        for r in rows
    ]


# ---------------------------------------------------------------------------
# Attachments
# ---------------------------------------------------------------------------

# The attachment size cap is the module-level ``KANBAN_ATTACHMENT_MAX_BYTES``
# (defined near the top of this file) — one constant shared by the dashboard
# HTTP endpoint, the agent toolset, and the CLI so the limit cannot drift
# between surfaces.


class AttachmentTooLarge(ValueError):
    """Raised when an attachment exceeds the configured size cap.

    Subclasses :class:`ValueError` so generic ``except ValueError`` handlers
    (e.g. the dashboard's 400 fallback) still catch it, while callers that
    want a distinct user-facing message (the tool/CLI 413-equivalent) can
    catch it specifically.
    """


def _safe_attachment_name(raw: str) -> str:
    """Reduce a client-supplied filename to a safe basename.

    Strips any directory components (both separators) so a malicious
    ``../../etc/passwd`` or ``C:\\x`` collapses to its leaf. Drops control
    chars and leading dots so we never write a dotfile or a name with
    embedded NULs/newlines. Rejects empty / dotfile-only names. The result
    is only ever joined under the per-task attachments dir, never used
    verbatim as a path from the client.

    Raises :class:`ValueError` on an unusable name; HTTP callers map that
    to a 400.
    """
    name = (raw or "").replace("\\", "/").split("/")[-1].strip()
    name = "".join(ch for ch in name if ch.isprintable() and ch not in "\x00").strip()
    name = name.lstrip(".").strip()
    if not name:
        raise ValueError("invalid attachment filename")
    return name[:200]


def _collision_free_path(dest_dir: Path, safe_name: str) -> Path:
    """Return a path under ``dest_dir`` that doesn't clobber an existing file.

    ``foo.pdf`` → ``foo.pdf``, then ``foo (1).pdf``, ``foo (2).pdf``, …
    ``safe_name`` must already be sanitised via :func:`_safe_attachment_name`.
    """
    stem, dot, ext = safe_name.partition(".")
    candidate = safe_name
    n = 1
    while (dest_dir / candidate).exists():
        candidate = f"{stem} ({n}){dot}{ext}"
        n += 1
    return dest_dir / candidate


def store_attachment_bytes(
    conn: sqlite3.Connection,
    task_id: str,
    filename: str,
    data: bytes,
    *,
    content_type: Optional[str] = None,
    uploaded_by: Optional[str] = None,
    board: Optional[str] = None,
    max_bytes: Optional[int] = None,
) -> int:
    """Validate, size-check, persist a blob, and record its metadata row.

    This is the single write path shared by the dashboard endpoint, the
    agent toolset (``kanban_attach`` / ``kanban_attach_url``), and the CLI
    (``hermes kanban attach``) so name-sanitisation, the size cap, and the
    collision-resolution all behave identically everywhere.

    Steps: enforce ``max_bytes``, sanitise ``filename`` to a safe basename,
    write the bytes under :func:`task_attachments_dir` with a
    collision-free name, then insert the ``task_attachments`` row via
    :func:`add_attachment`. Returns the new attachment id.

    Raises :class:`AttachmentTooLarge` when ``data`` exceeds ``max_bytes``,
    or :class:`ValueError` for a bad filename / unknown task. On any failure
    after the blob is written (e.g. the task disappeared) the orphaned blob
    is removed before re-raising.
    """
    if max_bytes is None:
        max_bytes = KANBAN_ATTACHMENT_MAX_BYTES
    if len(data) > max_bytes:
        raise AttachmentTooLarge(
            f"attachment exceeds {max_bytes // (1024 * 1024)} MB limit"
        )
    safe_name = _safe_attachment_name(filename)
    dest_dir = task_attachments_dir(task_id, board=board)
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = _collision_free_path(dest_dir, safe_name)
    dest_path.write_bytes(data)
    try:
        return add_attachment(
            conn,
            task_id,
            filename=dest_path.name,
            stored_path=str(dest_path.resolve()),
            content_type=content_type,
            size=len(data),
            uploaded_by=uploaded_by,
        )
    except Exception:
        # Don't leave an orphan blob if the metadata insert fails (most
        # commonly: the task id doesn't exist).
        try:
            dest_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def add_attachment(
    conn: sqlite3.Connection,
    task_id: str,
    *,
    filename: str,
    stored_path: str,
    content_type: Optional[str] = None,
    size: int = 0,
    uploaded_by: Optional[str] = None,
) -> int:
    """Record a file attachment for a task. Returns the new attachment id.

    The caller is responsible for writing the blob to ``stored_path``
    first (under :func:`task_attachments_dir`); this only persists the
    metadata row and appends an ``attached`` event.
    """
    if not filename or not filename.strip():
        raise ValueError("attachment filename is required")
    if not stored_path or not stored_path.strip():
        raise ValueError("attachment stored_path is required")
    now = int(time.time())
    with write_txn(conn):
        if not conn.execute(
            "SELECT 1 FROM tasks WHERE id = ? AND task_kind = 'work'", (task_id,)
        ).fetchone():
            raise ValueError(f"unknown task {task_id}")
        cur = conn.execute(
            "INSERT INTO task_attachments "
            "(task_id, filename, stored_path, content_type, size, uploaded_by, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                task_id,
                filename.strip(),
                stored_path,
                content_type,
                int(size),
                uploaded_by,
                now,
            ),
        )
        _append_event(
            conn,
            task_id,
            "attached",
            {"filename": filename.strip(), "size": int(size), "by": uploaded_by},
        )
        return int(cur.lastrowid or 0)


def list_attachments(conn: sqlite3.Connection, task_id: str) -> list[Attachment]:
    rows = conn.execute(
        "SELECT * FROM task_attachments WHERE task_id = ? "
        "AND task_id IN (SELECT id FROM tasks WHERE task_kind = 'work') "
        "ORDER BY created_at ASC, id ASC",
        (task_id,),
    ).fetchall()
    return [
        Attachment(
            id=r["id"],
            task_id=r["task_id"],
            filename=r["filename"],
            stored_path=r["stored_path"],
            content_type=r["content_type"],
            size=r["size"] or 0,
            uploaded_by=r["uploaded_by"],
            created_at=r["created_at"],
        )
        for r in rows
    ]


def get_attachment(conn: sqlite3.Connection, attachment_id: int) -> Optional[Attachment]:
    r = conn.execute(
        "SELECT * FROM task_attachments WHERE id = ? "
        "AND task_id IN (SELECT id FROM tasks WHERE task_kind = 'work')",
        (attachment_id,),
    ).fetchone()
    if r is None:
        return None
    return Attachment(
        id=r["id"],
        task_id=r["task_id"],
        filename=r["filename"],
        stored_path=r["stored_path"],
        content_type=r["content_type"],
        size=r["size"] or 0,
        uploaded_by=r["uploaded_by"],
        created_at=r["created_at"],
    )


def delete_attachment(conn: sqlite3.Connection, attachment_id: int) -> Optional[Attachment]:
    """Delete an attachment row and its on-disk blob. Returns the removed row.

    Returns ``None`` when no row matched. The blob is removed best-effort
    (a missing file is not an error); the metadata row is the source of
    truth for whether an attachment "exists".
    """
    with write_txn(conn):
        att = get_attachment(conn, attachment_id)
        if att is None:
            return None
        conn.execute("DELETE FROM task_attachments WHERE id = ?", (attachment_id,))
        _append_event(
            conn, att.task_id, "attachment_removed", {"filename": att.filename}
        )
    try:
        p = Path(att.stored_path)
        if p.is_file():
            p.unlink()
    except OSError:
        pass
    return att


def list_events(
    conn: sqlite3.Connection, task_id: str, *, include_control: bool = False
) -> list[Event]:
    # Ordinary public accessor: positively requires task_kind='work' so a
    # recommendation's typed audit event is never exposed here. Inspecting
    # it is test-only, via direct SQL against task_events. ``include_control``
    # additionally admits an owner-workspace control anchor's own audit trail,
    # which that kernel needs for crash-safe replay recognition.
    rows = conn.execute(
        "SELECT * FROM task_events WHERE task_id = ? "
        "AND task_id IN (SELECT id FROM tasks WHERE task_kind IN "
        f"{_task_kinds(include_control)}) "
        "ORDER BY created_at ASC, id ASC",
        (task_id,),
    ).fetchall()
    out = []
    for r in rows:
        try:
            payload = json.loads(r["payload"]) if r["payload"] else None
        except Exception:
            payload = None
        out.append(
            Event(
                id=r["id"],
                task_id=r["task_id"],
                kind=r["kind"],
                payload=payload,
                created_at=r["created_at"],
                run_id=(int(r["run_id"]) if "run_id" in r.keys() and r["run_id"] is not None else None),
            )
        )
    return out


def get_next_event_after(conn: sqlite3.Connection, task_id: str, after_id: int) -> Optional[Event]:
    """The earliest event for ``task_id`` with ``id > after_id``, if any.

    ``task_events.id`` is a single AUTOINCREMENT sequence shared by every
    task on the board, so it is NOT contiguous per task (another task's
    event insert between two of THIS task's events consumes an id without
    advancing this task's own revision). A replay caller holding
    ``expected_revision`` (this task's own last-seen revision) therefore
    cannot assume its own next event landed at exactly
    ``expected_revision + 1`` — it must ask for "whatever this task's very
    next event actually was", which is what this returns. Used by
    idempotent-replay callers (e.g. the owner-workspace kernel) to recognize
    whether a specific past mutation on this task already committed.
    """
    row = conn.execute(
        "SELECT * FROM task_events WHERE task_id = ? AND id > ? ORDER BY id ASC LIMIT 1",
        (task_id, after_id),
    ).fetchone()
    if row is None:
        return None
    try:
        payload = json.loads(row["payload"]) if row["payload"] else None
    except Exception:
        payload = None
    return Event(
        id=row["id"],
        task_id=row["task_id"],
        kind=row["kind"],
        payload=payload,
        created_at=row["created_at"],
        run_id=(int(row["run_id"]) if "run_id" in row.keys() and row["run_id"] is not None else None),
    )


def _append_event(
    conn: sqlite3.Connection,
    task_id: str,
    kind: str,
    payload: Optional[dict] = None,
    *,
    run_id: Optional[int] = None,
) -> None:
    """Record an event row.  Called from within an already-open txn.

    ``run_id`` is optional: pass the current run id so UIs can group
    events by attempt. For events that aren't scoped to a single run
    (task created/edited/archived, dependency promotion) leave it None
    and the row carries NULL.
    """
    now = int(time.time())
    pl = json.dumps(payload, ensure_ascii=False) if payload else None
    conn.execute(
        "INSERT INTO task_events (task_id, run_id, kind, payload, created_at) "
        "VALUES (?, ?, ?, ?, ?)",
        (task_id, run_id, kind, pl, now),
    )


def _end_run(
    conn: sqlite3.Connection,
    task_id: str,
    *,
    outcome: str,
    summary: Optional[str] = None,
    error: Optional[str] = None,
    metadata: Optional[dict] = None,
    status: Optional[str] = None,
) -> Optional[int]:
    """Close the currently-active run for ``task_id`` and clear the pointer.

    ``outcome`` is the semantic result (completed / blocked / crashed /
    timed_out / spawn_failed / gave_up / reclaimed). ``status`` is the
    run-row status (usually just ``outcome``, but callers can pass it
    explicitly). Returns the closed run_id or ``None`` if no active run
    existed (e.g. a CLI user calling ``hermes kanban complete`` on a
    task that was never claimed).
    """
    now = int(time.time())
    row = conn.execute(
        "SELECT current_run_id FROM tasks WHERE id = ? AND task_kind = 'work'",
        (task_id,),
    ).fetchone()
    if not row or not row["current_run_id"]:
        return None
    run_id = int(row["current_run_id"])
    conn.execute(
        """
        UPDATE task_runs
           SET status        = ?,
               outcome       = ?,
               summary       = ?,
               error         = ?,
               metadata      = ?,
               ended_at      = ?,
               claim_lock    = NULL,
               claim_expires = NULL,
               worker_pid    = NULL
         WHERE id = ?
           AND ended_at IS NULL
        """,
        (
            status or outcome,
            outcome,
            summary,
            error,
            json.dumps(metadata, ensure_ascii=False) if metadata else None,
            now,
            run_id,
        ),
    )
    conn.execute(
        "UPDATE tasks SET current_run_id = NULL "
        "WHERE id = ? AND task_kind = 'work'",
        (task_id,),
    )
    return run_id


def _current_run_id(conn: sqlite3.Connection, task_id: str) -> Optional[int]:
    row = conn.execute(
        "SELECT current_run_id FROM tasks WHERE id = ? AND task_kind = 'work'",
        (task_id,),
    ).fetchone()
    return int(row["current_run_id"]) if row and row["current_run_id"] else None


def _synthesize_ended_run(
    conn: sqlite3.Connection,
    task_id: str,
    *,
    outcome: str,
    summary: Optional[str] = None,
    error: Optional[str] = None,
    metadata: Optional[dict] = None,
) -> int:
    """Insert a zero-duration, already-closed run row.

    Used when a terminal transition happens on a task that was never
    claimed (CLI user calling ``hermes kanban complete <ready-task>
    --summary X``, or dashboard "mark done" on a ready task). Without
    this, the handoff fields (summary / metadata / error) would be
    silently dropped: ``_end_run`` is a no-op because there's no
    current run.

    The synthetic run has ``started_at == ended_at == now`` so it
    shows up in attempt history as "instant" and doesn't skew elapsed
    stats. Caller is responsible for leaving ``current_run_id`` NULL
    (or for clearing it elsewhere in the same txn) since this
    function does NOT touch the tasks row.
    """
    now = int(time.time())
    trow = conn.execute(
        "SELECT assignee, current_step_key FROM tasks "
        "WHERE id = ? AND task_kind = 'work'",
        (task_id,),
    ).fetchone()
    profile = trow["assignee"] if trow else None
    step_key = trow["current_step_key"] if trow else None
    cur = conn.execute(
        """
        INSERT INTO task_runs (
            task_id, profile, step_key,
            status, outcome,
            summary, error, metadata,
            started_at, ended_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            task_id, profile, step_key,
            outcome, outcome,
            summary, error,
            json.dumps(metadata, ensure_ascii=False) if metadata else None,
            now, now,
        ),
    )
    return int(cur.lastrowid or 0)


# ---------------------------------------------------------------------------
# Dependency resolution (todo -> ready)
# ---------------------------------------------------------------------------

def _has_sticky_block(conn: sqlite3.Connection, task_id: str) -> bool:
    """Return True when ``task_id`` is sticky-blocked by an explicit
    worker/operator ``kanban_block`` call (#28712).

    A ``blocked`` status can come from two very different sources:

    * **Worker- or operator-initiated** — a worker called
      ``kanban_block(reason="review-required: ...")`` (or somebody ran
      ``hermes kanban block <id>``).  This is a deliberate handoff that
      should stay blocked until an operator unblocks it.  The block tool
      emits a ``"blocked"`` event row in ``task_events``.

    * **Circuit-breaker** — ``_record_task_failure`` tripped after
      repeated crashes / spawn failures / timeouts.  This emits
      ``"gave_up"``, *not* ``"blocked"``, and is meant to recover
      automatically once the underlying conditions change (e.g. parents
      finish, transient infra error clears).

    The cheapest signal that distinguishes the two is the most recent
    ``"blocked"`` / ``"unblocked"`` event for the task.  If the most
    recent one is ``"blocked"`` (or there is a ``"blocked"`` event and
    no ``"unblocked"`` event has fired since), the task is sticky and
    ``recompute_ready`` must *not* auto-promote it.

    Returns ``False`` when there is no such event at all (e.g. the task
    was set to ``status='blocked'`` by the circuit breaker or by direct
    DB manipulation) — preserves the pre-#28712 auto-recover semantics
    for that path.
    """
    row = conn.execute(
        "SELECT kind FROM task_events "
        "WHERE task_id = ? AND kind IN ('blocked', 'unblocked') "
        "ORDER BY id DESC LIMIT 1",
        (task_id,),
    ).fetchone()
    return bool(row) and row["kind"] == "blocked"


def _resume_status_from_events(conn: sqlite3.Connection, task_id: str) -> str:
    """Return the durable phase a blocked/dependency-wait task should resume.

    Events written by review workers carry ``source_status``/``retry_status``;
    an explicit unblock that must wait for parents carries ``resume_status``.
    Legacy events omit these fields and therefore retain the historical
    ``ready`` behavior.
    """
    row = conn.execute(
        "SELECT payload FROM task_events "
        "WHERE task_id = ? AND kind IN ("
        "'blocked', 'block_loop_detected', 'dependency_wait', 'gave_up', "
        "'unblocked', 'changes_requested', 'review_reopened', 'status', 'reclaimed', "
        "'stale', 'timed_out', 'crashed', 'spawn_failed', 'rate_limited'"
        ") ORDER BY id DESC LIMIT 1",
        (task_id,),
    ).fetchone()
    try:
        payload = json.loads(row["payload"]) if row and row["payload"] else {}
    except (json.JSONDecodeError, TypeError):
        payload = {}
    if not isinstance(payload, dict):
        payload = {}
    for key in ("resume_status", "retry_status", "source_status"):
        if payload.get(key) == "review":
            return "review"
    return "ready"


def recompute_ready(
    conn: sqlite3.Connection, failure_limit: int = None,
) -> int:
    """Promote ``todo`` tasks to ``ready`` when all parents are ``done`` or ``archived``.

    Returns the number of tasks promoted.  Opens its own IMMEDIATE txn, so it
    MUST be called OUTSIDE any open write transaction (plain ``write_txn``
    raises on nesting); call it after the enclosing txn commits.

    ``blocked`` tasks are also considered for promotion (so a task
    blocked purely by a parent dependency unblocks itself when the
    parent completes), *except* in two cases:

    1. The most recent block event was a worker-initiated
       ``kanban_block`` — those stay blocked until an explicit
       ``kanban_unblock`` (#28712).

    2. The task's ``consecutive_failures`` has reached the effective
       failure limit.  This prevents infinite retry loops when a task
       repeatedly exhausts its iteration budget: without this guard the
       counter would reset on every recovery cycle and the circuit
       breaker could never trip (#35072).

    The effective failure limit resolves in the same order as the
    circuit breaker in ``_record_task_failure`` so the two never
    disagree about when a task is permanently blocked:

      1. per-task ``max_retries`` if set
      2. caller-supplied ``failure_limit`` (the dispatcher passes the
         ``kanban.failure_limit`` config value through ``dispatch_once``)
      3. ``DEFAULT_FAILURE_LIMIT``
    """
    if failure_limit is None:
        failure_limit = DEFAULT_FAILURE_LIMIT
    promoted = 0
    with write_txn(conn):
        todo_rows = conn.execute(
            "SELECT id, status, consecutive_failures, max_retries "
            "FROM tasks WHERE status IN ('todo', 'blocked') AND task_kind = 'work'"
        ).fetchall()
        for row in todo_rows:
            task_id = row["id"]
            cur_status = row["status"]
            if cur_status == "blocked" and _has_sticky_block(conn, task_id):
                # Worker / operator asked for explicit human intervention — do not
                # silently auto-recover.  ``unblock_task`` is the only
                # legitimate exit (it emits ``"unblocked"`` which flips
                # this predicate back).
                continue
            parents = conn.execute(
                "SELECT t.status FROM tasks t "
                "JOIN task_links l ON l.parent_id = t.id "
                f"WHERE l.child_id = ? AND t.task_kind IN {_DEPENDENCY_PARENT_KINDS}",
                (task_id,),
            ).fetchall()
            if all(p["status"] in ("done", "archived") for p in parents):
                # Automatic promotion is a transition INTO an executable
                # column, so it must prove this task's route authority first;
                # an owner-governed row that cannot be proven is parked here
                # instead of joining the work pool.
                if not authorize_executable_transition(conn, task_id):
                    continue
                resume_status = _resume_status_from_events(conn, task_id)
                if cur_status == "blocked":
                    # Don't auto-recover tasks that have hit the
                    # circuit-breaker failure limit.  Without this
                    # guard, a task that repeatedly exhausts its
                    # iteration budget would cycle forever:
                    # block → auto-recover → respawn → budget
                    # exhausted → block → …  The counter must also
                    # be preserved so the breaker can accumulate
                    # across recovery cycles.
                    failures = int(row["consecutive_failures"] or 0)
                    task_limit = row["max_retries"]
                    effective_limit = (
                        int(task_limit) if task_limit is not None
                        else int(failure_limit)
                    )
                    if failures >= effective_limit:
                        continue
                    conn.execute(
                        "UPDATE tasks SET status = ? "
                        "WHERE id = ? AND status = 'blocked' AND task_kind = 'work'",
                        (resume_status, task_id),
                    )
                else:
                    conn.execute(
                        "UPDATE tasks SET status = ? "
                        "WHERE id = ? AND status = 'todo' AND task_kind = 'work'",
                        (resume_status, task_id),
                    )
                _append_event(
                    conn, task_id, "promoted",
                    {"status": resume_status} if resume_status != "ready" else None,
                )
                promoted += 1
    return promoted


# ---------------------------------------------------------------------------
# Claim / complete / block
# ---------------------------------------------------------------------------

def _parents_satisfied(conn: sqlite3.Connection, task_id: str) -> bool:
    """Return whether every direct parent is terminal for dependency gating."""
    return conn.execute(
        "SELECT 1 FROM task_links l "
        "JOIN tasks p ON p.id = l.parent_id "
        "WHERE l.child_id = ? "
        f"AND p.task_kind IN {_DEPENDENCY_PARENT_KINDS} "
        "AND p.status NOT IN ('done', 'archived') LIMIT 1",
        (task_id,),
    ).fetchone() is None


def _decode_owned_paths(raw: Any) -> Optional[list[str]]:
    """Decode a durable ownership value; malformed data fails closed."""
    if raw is None:
        return None
    try:
        value = json.loads(raw) if isinstance(raw, str) else raw
        if not isinstance(value, list):
            return None
        return normalize_owned_paths(value)
    except (TypeError, ValueError, json.JSONDecodeError):
        return None


def _owned_path_scopes_overlap(
    left: Optional[list[str]], right: Optional[list[str]]
) -> bool:
    """Return whether two normalized write scopes can touch the same path."""
    # NULL is legacy/unknown, therefore whole-repository and exclusive.
    if left is None or right is None:
        return True
    # Explicitly read-only tasks never contend for repository writes.
    if not left or not right:
        return False
    if "." in left or "." in right:
        return True
    # Git paths are case-sensitive, but the checkout may not be (the common
    # macOS/Windows case). Treat Unicode/case-equivalent prefixes as colliding
    # everywhere. This can serialize two genuinely distinct Linux paths, but
    # it can never let two workers race on one physical path.
    left_keys = [unicodedata.normalize("NFC", path).casefold() for path in left]
    right_keys = [unicodedata.normalize("NFC", path).casefold() for path in right]
    return any(
        a == b or a.startswith(f"{b}/") or b.startswith(f"{a}/")
        for a in left_keys
        for b in right_keys
    )


def _project_repository_key(row: sqlite3.Row) -> Optional[str]:
    """Return a proven canonical Project repo identity from a task row."""
    if row["workspace_kind"] != "worktree" or not row["project_id"]:
        return None
    raw_path = str(row["workspace_path"] or "").strip()
    if not raw_path:
        return None
    path = Path(raw_path).expanduser()
    if not path.is_absolute() or path.parent.name != ".worktrees":
        return None
    repository = path.parent.parent.resolve(strict=False).as_posix()
    return unicodedata.normalize("NFC", repository).casefold()


def file_scope_conflicts(conn: sqlite3.Connection, task_id: str) -> list[str]:
    """Return running tasks that prevent ``task_id`` from claiming.

    Scratch tasks are already isolated in task-specific ephemeral directories
    and do not participate in repository locking. Otherwise, only explicit
    read-only work may run beside a non-worktree mutator. Two mutating tasks may
    run together only when both use isolated worktrees and their explicit
    ownership prefixes are disjoint. This deliberately treats every
    legacy/unparseable repository scope as whole-repository ownership. Known,
    different Projects are independent; an unlinked repository task serializes
    with all mutators because it carries no repository-identity proof.
    """
    candidate = conn.execute(
        "SELECT id, project_id, workspace_kind, workspace_path, owned_paths FROM tasks "
        "WHERE id = ? AND task_kind = 'work'",
        (task_id,),
    ).fetchone()
    if candidate is None:
        return []
    if candidate["workspace_kind"] == "scratch":
        return []
    candidate_paths = _decode_owned_paths(candidate["owned_paths"])
    if candidate_paths == [] and candidate["workspace_kind"] == "worktree":
        return []

    conflicts: list[str] = []
    running_rows = conn.execute(
        "SELECT id, project_id, workspace_kind, workspace_path, owned_paths FROM tasks "
        "WHERE id != ? AND status = 'running' AND task_kind = 'work' "
        "ORDER BY started_at, id",
        (task_id,),
    ).fetchall()
    for running in running_rows:
        if running["workspace_kind"] == "scratch":
            continue
        running_paths = _decode_owned_paths(running["owned_paths"])
        if running_paths == [] and running["workspace_kind"] == "worktree":
            continue
        # Explicitly different Projects have separate primary repositories by
        # default. When both rows carry canonical Project worktree paths, use
        # the repository itself as stronger evidence: deliberately duplicated
        # Projects pointing at one repo must still contend. An unlinked task
        # has no such proof, so it serializes with every mutator instead of
        # becoming an escape hatch after the global cap rises above one.
        candidate_project = str(candidate["project_id"] or "").strip()
        running_project = str(running["project_id"] or "").strip()
        if candidate_project and running_project and candidate_project != running_project:
            candidate_repo = _project_repository_key(candidate)
            running_repo = _project_repository_key(running)
            if not candidate_repo or not running_repo or candidate_repo != running_repo:
                continue
        if not candidate_project or not running_project:
            conflicts.append(str(running["id"]))
            continue
        isolated = (
            candidate["workspace_kind"] == "worktree"
            and running["workspace_kind"] == "worktree"
        )
        if not isolated or _owned_path_scopes_overlap(candidate_paths, running_paths):
            conflicts.append(str(running["id"]))
    return conflicts


def _record_file_scope_deferral(
    conn: sqlite3.Connection, task_id: str, conflicts: list[str], *, now: int
) -> None:
    """Emit at most one identical scope-deferral event per minute."""
    latest = conn.execute(
        "SELECT created_at, payload FROM task_events "
        "WHERE task_id = ? AND kind = 'claim_deferred_file_scope' "
        "ORDER BY id DESC LIMIT 1",
        (task_id,),
    ).fetchone()
    if latest is not None and int(latest["created_at"] or 0) >= now - 60:
        try:
            payload = json.loads(latest["payload"] or "{}")
        except (TypeError, json.JSONDecodeError):
            payload = {}
        if payload.get("blocking_task_ids") == conflicts:
            return
    _append_event(
        conn,
        task_id,
        "claim_deferred_file_scope",
        {"blocking_task_ids": conflicts},
    )


def claim_task(
    conn: sqlite3.Connection,
    task_id: str,
    *,
    ttl_seconds: Optional[int] = None,
    claimer: Optional[str] = None,
) -> Optional[Task]:
    """Atomically transition ``ready -> running``.

    Returns the claimed ``Task`` on success, ``None`` if the task was
    already claimed (or is not in ``ready`` status).
    """
    now = int(time.time())
    lock = claimer or _claimer_id()
    expires = now + _resolve_claim_ttl_seconds(ttl_seconds)
    with write_txn(conn):
        assert_claimable_route(conn, task_id)
        # Structural invariant: never transition ready -> running while any
        # parent is not yet 'done'. This is the single enforcement point
        # regardless of which writer (create_task, link_tasks, unblock_task,
        # release_stale_claims, manual SQL) set status='ready'. If a racy
        # writer promoted a task with undone parents, demote it back to
        # 'todo' here — recompute_ready will re-promote when the parents
        # actually finish. See RCA at
        # kanban/boards/cookai/workspaces/t_a6acd07d/root-cause.md.
        undone = conn.execute(
            "SELECT 1 FROM task_links l "
            "JOIN tasks p ON p.id = l.parent_id "
            f"WHERE l.child_id = ? AND p.task_kind IN {_DEPENDENCY_PARENT_KINDS} "
            "AND p.status NOT IN ('done', 'archived') LIMIT 1",
            (task_id,),
        ).fetchone()
        if undone:
            conn.execute(
                "UPDATE tasks SET status = 'todo' "
                "WHERE id = ? AND status = 'ready' AND task_kind = 'work'",
                (task_id,),
            )
            _append_event(
                conn, task_id, "claim_rejected",
                {"reason": "parents_not_done"},
            )
            return None
        conflicts = file_scope_conflicts(conn, task_id)
        if conflicts:
            _record_file_scope_deferral(conn, task_id, conflicts, now=now)
            return None
        # Defensive: if a prior run somehow leaked (invariant violation from
        # an unknown code path), close it as 'reclaimed' so we don't strand
        # it when the CAS resets the pointer below. No-op when the invariant
        # holds (the common case).
        stale = conn.execute(
            "SELECT current_run_id FROM tasks "
            "WHERE id = ? AND status = 'ready' AND task_kind = 'work'",
            (task_id,),
        ).fetchone()
        if stale and stale["current_run_id"]:
            conn.execute(
                """
                UPDATE task_runs
                   SET status = 'reclaimed', outcome = 'reclaimed',
                       summary = COALESCE(summary, 'invariant recovery on re-claim'),
                       ended_at = ?,
                       claim_lock = NULL, claim_expires = NULL, worker_pid = NULL
                 WHERE id = ? AND ended_at IS NULL
                """,
                (now, int(stale["current_run_id"])),
            )
        cur = conn.execute(
            """
            UPDATE tasks
               SET status        = 'running',
                   claim_lock    = ?,
                   claim_expires = ?,
                   started_at    = COALESCE(started_at, ?)
             WHERE id = ?
               AND status = 'ready'
               AND claim_lock IS NULL
               AND task_kind = 'work'
            """,
            (lock, expires, now, task_id),
        )
        if cur.rowcount != 1:
            return None
        # Look up the current task row so we can populate the run with
        # its assignee / step / runtime cap.
        trow = conn.execute(
            "SELECT assignee, max_runtime_seconds, current_step_key "
            "FROM tasks WHERE id = ?",
            (task_id,),
        ).fetchone()
        run_cur = conn.execute(
            """
            INSERT INTO task_runs (
                task_id, profile, step_key, status,
                claim_lock, claim_expires, max_runtime_seconds,
                started_at
            ) VALUES (?, ?, ?, 'running', ?, ?, ?, ?)
            """,
            (
                task_id,
                trow["assignee"] if trow else None,
                trow["current_step_key"] if trow else None,
                lock,
                expires,
                trow["max_runtime_seconds"] if trow else None,
                now,
            ),
        )
        run_id = run_cur.lastrowid
        conn.execute(
            "UPDATE tasks SET current_run_id = ? WHERE id = ?",
            (run_id, task_id),
        )
        _append_event(
            conn, task_id, "claimed",
            {"lock": lock, "expires": expires, "run_id": run_id},
            run_id=run_id,
        )
        claimed = get_task(conn, task_id)
    _fire_kanban_lifecycle_hook(
        "kanban_task_claimed",
        task_id,
        board=get_current_board(),
        assignee=claimed.assignee if claimed else None,
        run_id=run_id,
    )
    return claimed


def claim_review_task(
    conn: sqlite3.Connection,
    task_id: str,
    *,
    ttl_seconds: Optional[int] = None,
    claimer: Optional[str] = None,
) -> Optional[Task]:
    """Atomically transition ``review -> running``.

    Returns the claimed ``Task`` on success, ``None`` if the task was
    already claimed (or is not in ``review`` status).

    Parent dependencies are re-checked because a previously completed parent
    may have been reopened while this task waited in review.

    Creates a new run entry so the review agent's lifecycle is tracked
    independently from the original worker run.
    """
    now = int(time.time())
    lock = claimer or _claimer_id()
    expires = now + _resolve_claim_ttl_seconds(ttl_seconds)
    with write_txn(conn):
        assert_claimable_route(conn, task_id)
        if not _parents_satisfied(conn, task_id):
            demoted = conn.execute(
                "UPDATE tasks SET status = 'todo' "
                "WHERE id = ? AND status = 'review' AND claim_lock IS NULL "
                "AND task_kind = 'work'",
                (task_id,),
            )
            if demoted.rowcount == 1:
                _append_event(
                    conn,
                    task_id,
                    "dependency_wait",
                    {
                        "reason": "parent_reopened",
                        "source_status": "review",
                    },
                )
            return None
        conflicts = file_scope_conflicts(conn, task_id)
        if conflicts:
            _record_file_scope_deferral(conn, task_id, conflicts, now=now)
            return None
        cur = conn.execute(
            """
            UPDATE tasks
               SET status        = 'running',
                   claim_lock    = ?,
                   claim_expires = ?,
                   started_at    = COALESCE(started_at, ?)
             WHERE id = ?
               AND status = 'review'
               AND claim_lock IS NULL
               AND task_kind = 'work'
            """,
            (lock, expires, now, task_id),
        )
        if cur.rowcount != 1:
            return None
        trow = conn.execute(
            "SELECT assignee, max_runtime_seconds, current_step_key "
            "FROM tasks WHERE id = ?",
            (task_id,),
        ).fetchone()
        run_cur = conn.execute(
            """
            INSERT INTO task_runs (
                task_id, profile, step_key, status,
                claim_lock, claim_expires, max_runtime_seconds,
                started_at
            ) VALUES (?, ?, ?, 'running', ?, ?, ?, ?)
            """,
            (
                task_id,
                trow["assignee"] if trow else None,
                trow["current_step_key"] if trow else None,
                lock,
                expires,
                trow["max_runtime_seconds"] if trow else None,
                now,
            ),
        )
        run_id = run_cur.lastrowid
        conn.execute(
            "UPDATE tasks SET current_run_id = ? WHERE id = ?",
            (run_id, task_id),
        )
        _append_event(
            conn, task_id, "claimed",
            {"lock": lock, "expires": expires, "run_id": run_id,
             "source_status": "review"},
            run_id=run_id,
        )
        return get_task(conn, task_id)


def _retry_status_for_run(
    conn: sqlite3.Connection,
    task_id: str,
    run_id: Optional[int] = None,
) -> str:
    """Return the non-running phase an interrupted run must resume from.

    Review claims record ``source_status=review`` on their claimed event. All
    other and legacy runs retry from ``ready``. Keeping this decision in one
    place prevents crash/timeout/reclaim paths from silently converting a
    reviewer run into an implementation run.
    """
    if run_id is None:
        row = conn.execute(
            "SELECT current_run_id FROM tasks WHERE id = ?",
            (task_id,),
        ).fetchone()
        run_id = row["current_run_id"] if row else None
    if run_id is None:
        return "ready"
    event = conn.execute(
        "SELECT payload FROM task_events "
        "WHERE task_id = ? AND run_id = ? AND kind = 'claimed' "
        "ORDER BY id DESC LIMIT 1",
        (task_id, int(run_id)),
    ).fetchone()
    try:
        payload = json.loads(event["payload"]) if event and event["payload"] else {}
    except (json.JSONDecodeError, TypeError):
        payload = {}
    if not isinstance(payload, dict):
        payload = {}
    return "review" if payload.get("source_status") == "review" else "ready"


def goal_run_status(
    conn: sqlite3.Connection,
    task_id: str,
    expected_run_id: Optional[int] = None,
) -> Optional[str]:
    """Resolve lifecycle status from the perspective of one worker run.

    A successor may claim the task immediately after this run hands it off.
    Returning the task's live ``running`` status in that case lets the old goal
    loop mutate the successor.  Bind terminal handoffs to the original run and
    report any other ownership loss as ``superseded``.
    """
    task = get_task(conn, task_id)
    if task is None:
        return None
    if expected_run_id is not None:
        row = conn.execute(
            "SELECT outcome FROM task_runs WHERE id = ? AND task_id = ?",
            (int(expected_run_id), task_id),
        ).fetchone()
        outcome = (
            str(row["outcome"])
            if row and row["outcome"] is not None
            else None
        )
        terminal_status = (
            {
                "completed": "done",
                "review_requested": "review",
                "changes_requested": "changes_requested",
                "blocked": "blocked",
                "dependency_wait": "blocked",
            }.get(outcome)
            if outcome is not None
            else None
        )
        if terminal_status is not None:
            return terminal_status
        if outcome is not None or task.current_run_id != int(expected_run_id):
            return "superseded"
    if task.status in {"ready", "todo"}:
        event = conn.execute(
            "SELECT kind FROM task_events WHERE task_id = ? "
            "ORDER BY id DESC LIMIT 1",
            (task_id,),
        ).fetchone()
        if event and event["kind"] == "changes_requested":
            return "changes_requested"
    return task.status


def heartbeat_claim(
    conn: sqlite3.Connection,
    task_id: str,
    *,
    ttl_seconds: Optional[int] = None,
    claimer: Optional[str] = None,
) -> bool:
    """Extend a running claim.  Returns True if we still own it.

    Workers that know they'll exceed 15 minutes should call this every
    few minutes to keep ownership.
    """
    expires = int(time.time()) + _resolve_claim_ttl_seconds(ttl_seconds)
    lock = claimer or _claimer_id()
    with write_txn(conn):
        cur = conn.execute(
            "UPDATE tasks SET claim_expires = ? "
            "WHERE id = ? AND status = 'running' AND claim_lock = ? "
            "AND task_kind = 'work'",
            (expires, task_id, lock),
        )
        if cur.rowcount == 1:
            run_id = _current_run_id(conn, task_id)
            if run_id is not None:
                conn.execute(
                    "UPDATE task_runs SET claim_expires = ? WHERE id = ?",
                    (expires, run_id),
                )
            return True
        return False


def release_stale_claims(
    conn: sqlite3.Connection,
    *,
    signal_fn=None,
) -> int:
    """Reset any ``running`` task whose claim has expired.

    A stale-by-TTL claim whose host-local worker PID is still alive is
    *extended* (with a ``claim_extended`` event) instead of being
    reclaimed. Reclaiming a live worker mid-flight produces the spawn-
    then-immediately-reclaim loop seen on slow models that spend longer
    than ``DEFAULT_CLAIM_TTL_SECONDS`` inside a single tool-free LLM
    call (#23025): no tool calls means no ``kanban_heartbeat``, even
    though the subprocess is healthy.

    Backstop (#29747 gap 3): if the worker's PID is still alive but its
    ``last_heartbeat_at`` is stale by more than
    ``DEFAULT_CLAIM_HEARTBEAT_MAX_STALE_SECONDS`` (1h), the worker has
    been making no observable progress and we reclaim anyway — even if
    ``_pid_alive`` is still true. This catches the wedged-in-a-logic-loop
    case where the process is technically running but accomplishing
    nothing. ``_touch_activity`` (run_agent.py) bridges chunk-level
    liveness into ``last_heartbeat_at`` via #31752, so any genuinely
    active worker keeps its heartbeat fresh as a side effect of normal
    API traffic. ``enforce_max_runtime`` and ``detect_crashed_workers``
    remain the upper bounds for genuinely wedged or dead workers.

    Returns the number of stale claims actually reclaimed (live-pid
    extensions don't count). Safe to call often.
    """
    now = int(time.time())
    reclaimed = 0
    host_prefix = f"{_claimer_id().split(':', 1)[0]}:"
    stale = conn.execute(
        "SELECT id, claim_lock, worker_pid, claim_expires, last_heartbeat_at, "
        "       assignee "
        "FROM tasks "
        "WHERE status = 'running' AND claim_expires IS NOT NULL "
        "  AND claim_expires < ? AND task_kind = 'work'",
        (now,),
    ).fetchall()
    for row in stale:
        lock = row["claim_lock"] or ""
        host_local = lock.startswith(host_prefix)
        hb = row["last_heartbeat_at"]
        # Heartbeat staleness backstop: if we have a heartbeat at all
        # and it's older than the max-stale threshold, the worker is
        # not making observable progress.  Reclaim instead of extending,
        # even if the PID is still alive (it's likely in a logic loop).
        heartbeat_stale = (
            hb is not None
            and (now - int(hb)) > DEFAULT_CLAIM_HEARTBEAT_MAX_STALE_SECONDS
        )
        if (
            host_local
            and row["worker_pid"]
            and _pid_alive(row["worker_pid"])
            and not heartbeat_stale
        ):
            new_expires = now + _resolve_claim_ttl_seconds()
            with write_txn(conn):
                cur = conn.execute(
                    "UPDATE tasks SET claim_expires = ? "
                    "WHERE id = ? AND status = 'running' "
                    "  AND claim_lock IS ? "
                    "  AND claim_expires IS NOT NULL "
                    "  AND claim_expires < ?",
                    (new_expires, row["id"], row["claim_lock"], now),
                )
                if cur.rowcount != 1:
                    continue
                run_id = _current_run_id(conn, row["id"])
                if run_id is not None:
                    conn.execute(
                        "UPDATE task_runs SET claim_expires = ? WHERE id = ?",
                        (new_expires, run_id),
                    )
                _append_event(
                    conn, row["id"], "claim_extended",
                    {
                        "reason": "pid_alive",
                        "worker_pid": int(row["worker_pid"]),
                        "claim_lock": row["claim_lock"],
                        "claim_expires_was": int(row["claim_expires"]),
                        "claim_expires_now": new_expires,
                        "last_heartbeat_at": (
                            int(row["last_heartbeat_at"])
                            if row["last_heartbeat_at"] is not None
                            else None
                        ),
                    },
                    run_id=run_id,
                )
            continue

        termination = _terminate_reclaimed_worker(
            row["worker_pid"], row["claim_lock"], signal_fn=signal_fn,
        )
        # Never release a claim while our own worker is still alive: that would
        # spawn a duplicate beside it. Hold the claim and retry next tick.
        if _worker_survived_termination(termination):
            _defer_reclaim_for_live_worker(
                conn, row["id"], row["claim_lock"], now, termination,
                reason="ttl_expired_worker_alive",
            )
            continue
        with write_txn(conn):
            retry_status = _retry_status_for_run(conn, row["id"])
            cur = conn.execute(
                "UPDATE tasks SET status = ?, claim_lock = NULL, "
                "claim_expires = NULL, worker_pid = NULL "
                "WHERE id = ? AND status = 'running' AND claim_lock IS ? "
                "AND claim_expires IS NOT NULL AND claim_expires < ?",
                (retry_status, row["id"], row["claim_lock"], now),
            )
            if cur.rowcount != 1:
                continue
            run_id = _end_run(
                conn, row["id"],
                outcome="reclaimed", status="reclaimed",
                error=f"stale_lock={row['claim_lock']}",
                metadata=termination,
            )
            payload = {
                "stale_lock": row["claim_lock"],
                "worker_pid": (
                    int(row["worker_pid"])
                    if row["worker_pid"] is not None else None
                ),
                "claim_expires": int(row["claim_expires"]),
                "last_heartbeat_at": (
                    int(row["last_heartbeat_at"])
                    if row["last_heartbeat_at"] is not None else None
                ),
                "now": now,
                "host_local": host_local,
                "heartbeat_stale": bool(heartbeat_stale),
                "retry_status": retry_status,
            }
            payload.update(termination)
            _append_event(
                conn, row["id"], "reclaimed",
                payload,
                run_id=run_id,
            )
            reclaimed += 1
        # Worker-lifecycle observer (RFC #58548): the reclaim txn above has
        # committed. The ``continue`` branches (rowcount mismatch, claim
        # extension, deferred reclaim) never reach this point, so only a
        # genuinely reclaimed stale claim fires.
        if _kanban_observer_consumed("on_kanban_worker_stale_claim"):
            _fire_kanban_lifecycle_hook(
                "on_kanban_worker_stale_claim",
                row["id"],
                board=get_current_board(),
                assignee=row["assignee"],
                run_id=run_id,
                worker_pid=(
                    int(row["worker_pid"])
                    if row["worker_pid"] is not None else None
                ),
                heartbeat_stale=bool(heartbeat_stale),
                retry_status=retry_status,
            )
    return reclaimed


def reclaim_task(
    conn: sqlite3.Connection,
    task_id: str,
    *,
    reason: Optional[str] = None,
    signal_fn=None,
) -> bool:
    """Operator-driven reclaim: release the claim and restore its source phase.

    Unlike :func:`release_stale_claims` which only acts on tasks whose
    ``claim_expires`` has passed, this function reclaims immediately
    regardless of TTL. Intended for the dashboard/CLI recovery flow
    when an operator wants to abort a running worker without waiting
    for the TTL to expire (e.g. after seeing a hallucination warning).

    Returns True if a reclaim happened, False if the task isn't in a
    reclaimable state (not running, or doesn't exist).
    """
    row = conn.execute(
        "SELECT status, claim_lock, worker_pid FROM tasks "
        "WHERE id = ? AND task_kind = 'work'",
        (task_id,),
    ).fetchone()
    if not row:
        return False
    if row["status"] != "running" and row["claim_lock"] is None:
        # Nothing to reclaim — already ready / blocked / done.
        return False
    prev_lock = row["claim_lock"]
    termination = _terminate_reclaimed_worker(
        row["worker_pid"], prev_lock, signal_fn=signal_fn,
    )
    with write_txn(conn):
        retry_status = _retry_status_for_run(conn, task_id)
        cur = conn.execute(
            "UPDATE tasks SET status = ?, claim_lock = NULL, "
            "claim_expires = NULL, worker_pid = NULL "
            "WHERE id = ? AND status IN ('running', 'ready', 'blocked') "
            "AND claim_lock IS ? AND task_kind = 'work'",
            (retry_status, task_id, prev_lock),
        )
        if cur.rowcount != 1:
            return False
        run_id = _end_run(
            conn, task_id,
            outcome="reclaimed", status="reclaimed",
            error=(
                f"manual_reclaim: {reason}" if reason
                else f"manual_reclaim lock={prev_lock}"
            ),
            metadata=termination,
        )
        payload = {
            "manual": True,
            "reason": reason,
            "prev_lock": prev_lock,
            "retry_status": retry_status,
        }
        payload.update(termination)
        _append_event(
            conn, task_id, "reclaimed",
            payload,
            run_id=run_id,
        )
    # Operator intervention — they've looked at the task, so the
    # consecutive-failures counter is now stale. Give the next retry
    # a fresh budget. (_clear_failure_counter opens its own write_txn,
    # so it runs after the enclosing one commits.)
    _clear_failure_counter(conn, task_id)
    return True


def reassign_task(
    conn: sqlite3.Connection,
    task_id: str,
    profile: Optional[str],
    *,
    reclaim_first: bool = False,
    reason: Optional[str] = None,
) -> bool:
    """Reassign a task, optionally reclaiming a stuck running worker first.

    This is the recovery path for "this profile's model is broken, try
    a different one". If ``reclaim_first`` is True, any active claim is
    released (via :func:`reclaim_task`) before the reassign happens;
    otherwise the function refuses to reassign a currently-running task
    and returns False (caller can retry with ``reclaim_first=True``).

    Returns True if the reassign landed. ``profile`` may be ``None`` to
    unassign entirely.
    """
    if reclaim_first:
        # Safe to call even if nothing to reclaim.
        reclaim_task(conn, task_id, reason=reason or "reassign")
    # assign_task handles its own txn + the still-running guard.
    try:
        return assign_task(conn, task_id, profile)
    except RuntimeError:
        # Task is still running and reclaim_first was False; caller
        # needs to decide whether to retry with reclaim.
        return False


def _verify_created_cards(
    conn: sqlite3.Connection,
    completing_task_id: str,
    claimed_ids: Iterable[str],
) -> tuple[list[str], list[str]]:
    """Partition ``claimed_ids`` into (verified, phantom).

    A card is "verified" iff a row exists in ``tasks`` AND at least one
    of the following holds:

    * ``created_by`` matches the completing task's ``assignee`` profile
      (the common case: worker A spawns a card via ``kanban_create``,
      which stamps ``created_by=A``).
    * ``created_by`` matches the completing task's id (edge case where
      a worker passed its own task id as the ``created_by`` value).
    * The card is linked as a ``task_links.child`` of the completing
      task — i.e. the worker explicitly called ``kanban_create`` with
      ``parents=[<current_task>]``. This accepts cards created through
      the dashboard/CLI by a different principal but then attached to
      the completing task by the worker.

    ``phantom`` returns ids that either don't exist at all, or exist
    but don't satisfy any of the three trust conditions. The caller
    decides what to do with each bucket; this helper never mutates.
    """
    claimed = [str(x).strip() for x in (claimed_ids or []) if str(x).strip()]
    if not claimed:
        return [], []
    # Dedupe while preserving order.
    seen: set[str] = set()
    ordered: list[str] = []
    for cid in claimed:
        if cid not in seen:
            seen.add(cid)
            ordered.append(cid)

    row = conn.execute(
        "SELECT assignee FROM tasks WHERE id = ? AND task_kind = 'work'",
        (completing_task_id,),
    ).fetchone()
    if row is None:
        # Completing task not found — nothing resolves.
        return [], ordered
    completing_assignee = row["assignee"]

    # Batch-fetch existence + created_by in one query.
    placeholders = ",".join(["?"] * len(ordered))
    rows = conn.execute(
        f"SELECT id, created_by FROM tasks WHERE id IN ({placeholders}) "
        "AND task_kind = 'work'",
        tuple(ordered),
    ).fetchall()
    found = {r["id"]: r["created_by"] for r in rows}

    # Pull the set of cards linked as children of the completing task.
    # Cheap: one query, indexed on parent_id.
    linked_children: set[str] = set(child_ids(conn, completing_task_id))

    verified: list[str] = []
    phantom: list[str] = []
    for cid in ordered:
        created_by = found.get(cid)
        if created_by is None:
            phantom.append(cid)
            continue
        # Accept if any of the three trust conditions holds.
        if completing_assignee and created_by == completing_assignee:
            verified.append(cid)
        elif created_by == completing_task_id:
            verified.append(cid)
        elif cid in linked_children:
            verified.append(cid)
        else:
            phantom.append(cid)
    return verified, phantom


# Task-id pattern used both by ``kanban_create`` (``t_<12 hex>``) and
# ``_new_task_id`` below. Kept permissive on length for forward compat:
# accept 8+ hex chars after the ``t_`` prefix.
_TASK_ID_PROSE_RE = re.compile(r"\bt_[a-f0-9]{8,}\b")


def _scan_prose_for_phantom_ids(
    conn: sqlite3.Connection,
    text: str,
) -> list[str]:
    """Regex-scan free-form text for ``t_<hex>`` references; return the
    ones that don't exist in ``tasks``.

    Used as a non-blocking advisory check on completion summaries. An
    empty return means "no suspicious references found" — either the
    text had no IDs at all, or every ID it mentioned resolves to a real
    task. Duplicates are deduped.
    """
    if not text:
        return []
    matches = _TASK_ID_PROSE_RE.findall(text)
    if not matches:
        return []
    # Dedupe preserving order.
    seen: set[str] = set()
    unique: list[str] = []
    for m in matches:
        if m not in seen:
            seen.add(m)
            unique.append(m)
    placeholders = ",".join(["?"] * len(unique))
    rows = conn.execute(
        f"SELECT id FROM tasks WHERE id IN ({placeholders}) AND task_kind = 'work'",
        tuple(unique),
    ).fetchall()
    existing = {r["id"] for r in rows}
    return [m for m in unique if m not in existing]


class HallucinatedCardsError(ValueError):
    """Raised by ``complete_task`` when ``created_cards`` contains ids
    that don't exist or weren't created by the completing worker.

    The phantom list is attached as ``.phantom`` for callers that want
    structured access. Kept as ``ValueError`` subclass so existing
    tool-error handlers treat it as a recoverable user error.
    """

    def __init__(self, phantom: list[str], completing_task_id: str):
        self.phantom = list(phantom)
        self.completing_task_id = completing_task_id
        super().__init__(
            f"completion blocked: claimed created_cards that do not exist "
            f"or were not created by this worker: {', '.join(phantom)}"
        )


class ArtifactPreservationError(RuntimeError):
    """Raised when a declared scratch deliverable cannot be preserved."""


class WorktreeScopeError(ValueError):
    """Raised when scoped git work cannot be proven clean and in-bounds."""


def complete_task(
    conn: sqlite3.Connection,
    task_id: str,
    *,
    result: Optional[str] = None,
    summary: Optional[str] = None,
    metadata: Optional[dict] = None,
    created_cards: Optional[Iterable[str]] = None,
    patch_attachment_id: Optional[int] = None,
    merge_parent_heads: bool = False,
    expected_run_id: Optional[int] = None,
    fire_lifecycle_hook: bool = True,
) -> bool:
    """Transition ``running|ready|blocked|review -> done`` and record ``result``.

    Accepts a task that is merely ``ready`` too, so a manual CLI
    completion (``hermes kanban complete <id>``) works without requiring
    a claim/start/complete sequence. ``review`` is accepted so a human
    (or reviewer) can approve a task parked in the review lane by
    :func:`request_review` — even when it has no active run
    (``current_run_id IS NULL``), the handoff fields are preserved via
    :func:`_synthesize_ended_run`.

    ``summary`` and ``metadata`` are stored on the closing run (if any)
    and surfaced to downstream children via :func:`build_worker_context`.
    When ``summary`` is omitted we fall back to ``result`` so single-run
    callers do not have to pass both. ``metadata`` is a free-form dict
    (e.g. ``{"changed_files": [...], "tests_run": [...]}``) — workers
    are encouraged to use it for structured handoff facts.

    ``created_cards`` is an optional list of task ids the completing
    worker claims to have created. Each id is verified against
    ``tasks.created_by``. If any id is phantom (does not exist or was
    not created by this worker's assignee profile), completion is blocked
    with a ``HallucinatedCardsError`` and a
    ``completion_blocked_hallucination`` event is emitted so the rejected
    attempt is auditable. When all ids verify, they are recorded on the
    ``completed`` event payload.

    ``patch_attachment_id`` is the narrow handoff for a remote sandbox that
    cannot see the host worktree. The trusted kanban kernel validates an
    agent-uploaded ``.patch`` from the task's current run, applies it inside
    the already-scoped worktree, checks declared path ownership, and commits
    it before deriving the normal execution receipt. ``merge_parent_heads``
    asks the same kernel to merge every exact mutating parent receipt first;
    it is accepted only for a task declared with
    ``integrates_parent_heads=true``. Neither option grants the worker host
    filesystem or shell access.

    After a successful completion, ``summary`` and ``result`` are scanned
    for prose references like ``t_deadbeefcafe`` that do not resolve.
    Any suspected phantom references are recorded as a
    ``suspected_hallucinated_references`` event. This pass is advisory
    and never blocks.
    """
    now = int(time.time())
    # Preserve the mutator isolation contract: recommendation rows and unknown
    # ids are not completable work tasks.  Check this before deriving scoped
    # git evidence so those ids retain the historical ``False`` result instead
    # of being upgraded into a worktree-scope error.
    if get_task(conn, task_id) is None:
        return False
    # Fail before validating cards or staging artifacts; re-check inside the
    # final write transaction below to close the parent-reopen race.
    if not _parents_satisfied(conn, task_id):
        return False

    # Gate: verify created_cards BEFORE the main write txn. A rejected
    # completion still needs an auditable event, so we emit it in a
    # tiny dedicated txn, then raise. The caller is responsible for
    # surfacing HallucinatedCardsError to the worker; this function
    # never mutates task state on a phantom-card rejection.
    if created_cards:
        verified_cards, phantom_cards = _verify_created_cards(
            conn, task_id, created_cards
        )
        if phantom_cards:
            with write_txn(conn):
                _append_event(
                    conn, task_id, "completion_blocked_hallucination",
                    {
                        "phantom_cards": phantom_cards,
                        "verified_cards": verified_cards,
                        "summary_preview": (
                            (summary or result or "").strip().splitlines()[0][:200]
                            if (summary or result)
                            else None
                        ),
                    },
                )
            raise HallucinatedCardsError(phantom_cards, task_id)
    else:
        verified_cards = []

    # Exact git evidence is derived by the kernel, never trusted from worker
    # prose/metadata. Legacy tasks (owned_paths=NULL) retain their historical
    # completion behaviour; explicitly scoped tasks fail closed.
    materialization_receipt: Optional[dict[str, Any]] = None
    materialization_start: Optional[str] = None
    try:
        if patch_attachment_id is not None or merge_parent_heads:
            materialization_receipt, materialization_start = (
                _materialize_remote_worktree_handoff(
                    conn,
                    task_id,
                    patch_attachment_id=patch_attachment_id,
                    merge_parent_heads=merge_parent_heads,
                    expected_run_id=expected_run_id,
                )
            )
        execution_receipt = _verify_scoped_worktree_completion(conn, task_id)
    except WorktreeScopeError as exc:
        if materialization_start is not None:
            _rollback_worktree_materialization(conn, task_id, materialization_start)
        with write_txn(conn):
            _append_event(
                conn,
                task_id,
                "completion_blocked_file_scope",
                {"reason": str(exc)[:800]},
            )
        raise
    if execution_receipt is not None:
        metadata = dict(metadata or {})
        metadata["execution_receipt"] = execution_receipt
    if materialization_receipt is not None:
        metadata = dict(metadata or {})
        metadata["worktree_materialization"] = materialization_receipt

    metadata = _merge_completion_prose_artifacts(
        conn, task_id, metadata, summary=summary, result=result,
    )
    with write_txn(conn):
        # Parent completion is a hard invariant even for direct human review
        # approval. A parent may have been reopened after this task entered
        # ``review`` or ``running``.
        if not _parents_satisfied(conn, task_id):
            if materialization_start is not None:
                _rollback_worktree_materialization(
                    conn, task_id, materialization_start
                )
            return False
        prior = conn.execute(
            "SELECT status FROM tasks WHERE id = ? AND task_kind = 'work'",
            (task_id,),
        ).fetchone()
        prior_status = prior["status"] if prior else None
        if expected_run_id is None:
            cur = conn.execute(
                """
                UPDATE tasks
                   SET status       = 'done',
                       result       = ?,
                       completed_at = ?,
                       claim_lock   = NULL,
                       claim_expires= NULL,
                       worker_pid   = NULL,
                       block_kind   = NULL,
                       block_recurrences = 0,
                       head_commit   = COALESCE(?, head_commit)
                 WHERE id = ?
                   AND status IN ('running', 'ready', 'blocked', 'review')
                   AND task_kind = 'work'
                """,
                (
                    result,
                    now,
                    execution_receipt.get("head_commit") if execution_receipt else None,
                    task_id,
                ),
            )
        else:
            cur = conn.execute(
                """
                UPDATE tasks
                   SET status       = 'done',
                       result       = ?,
                       completed_at = ?,
                       claim_lock   = NULL,
                       claim_expires= NULL,
                       worker_pid   = NULL,
                       block_kind   = NULL,
                       block_recurrences = 0,
                       head_commit   = COALESCE(?, head_commit)
                 WHERE id = ?
                   AND status IN ('running', 'ready', 'blocked', 'review')
                   AND current_run_id = ?
                   AND task_kind = 'work'
                """,
                (
                    result,
                    now,
                    execution_receipt.get("head_commit") if execution_receipt else None,
                    task_id,
                    int(expected_run_id),
                ),
            )
        if cur.rowcount != 1:
            if materialization_start is not None:
                _rollback_worktree_materialization(
                    conn, task_id, materialization_start
                )
            return False
        if isinstance(metadata, dict):
            _persist_scratch_completion_artifacts(conn, task_id, metadata)
            for stored_path in metadata.pop("_staged_artifacts", []):
                path = Path(stored_path)
                _insert_completion_attachment(
                    conn,
                    task_id,
                    filename=path.name,
                    stored_path=str(path),
                    size=path.stat().st_size,
                    created_at=now,
                )
        run_id = _end_run(
            conn, task_id,
            outcome="completed", status="done",
            summary=summary if summary is not None else result,
            metadata=metadata,
        )
        # If complete_task was called on a never-claimed task (ready or
        # blocked → done with no run in flight), synthesize a
        # zero-duration run so the handoff fields are persisted in
        # attempt history instead of silently lost.
        if run_id is None and (
            summary or metadata or result or prior_status == "review"
        ):
            synth_summary = summary if summary is not None else result
            synth_metadata = metadata
            if prior_status == "review" and not synth_summary and not synth_metadata:
                synth_summary = "Review approved without additional evidence."
                synth_metadata = {
                    "source_status": "review",
                    "approval": "manual",
                }
            run_id = _synthesize_ended_run(
                conn, task_id,
                outcome="completed",
                summary=synth_summary,
                metadata=synth_metadata,
            )
        # Carry the handoff summary in the event payload so gateway
        # notifiers and dashboard WS consumers can render it without a
        # second SQL round-trip. First line only, 400 char cap — the
        # full summary stays on the run row.
        event_summary = summary if summary is not None else result
        if prior_status == "review" and not event_summary:
            event_summary = "Review approved without additional evidence."
        _ev_lines = (event_summary or "").strip().splitlines()
        ev_summary = _ev_lines[0][:400] if _ev_lines else ""
        completed_payload: dict = {
            "result_len": len(result) if result else 0,
            "summary": ev_summary or None,
        }
        if verified_cards:
            completed_payload["verified_cards"] = verified_cards
        # Carry artifact paths in the event payload so the gateway
        # notifier can upload them as native attachments alongside the
        # completion message. Workers pass these via
        # ``kanban_complete(artifacts=[...])`` which stashes the list in
        # ``metadata["artifacts"]`` — we promote it onto the event so
        # consumers don't have to fetch the run row to find it.
        if isinstance(metadata, dict):
            md_artifacts = metadata.get("artifacts")
            if isinstance(md_artifacts, (list, tuple)):
                cleaned_artifacts = [
                    str(p).strip() for p in md_artifacts if isinstance(p, str) and str(p).strip()
                ]
                if cleaned_artifacts:
                    completed_payload["artifacts"] = cleaned_artifacts
        _append_event(
            conn, task_id, "completed",
            completed_payload,
            run_id=run_id,
        )
    # Prose-scan the summary + result for t_<hex> references that do
    # not resolve. Advisory — does not block the completion. Runs in
    # its own txn so the completion itself is already durable by the
    # time we emit the warning.
    scan_text = " ".join(filter(None, [summary, result]))
    if scan_text:
        phantom_refs = _scan_prose_for_phantom_ids(conn, scan_text)
        # Drop any phantom refs that were already flagged as verified
        # above (shouldn't happen — verified means they exist — but
        # belt-and-suspenders).
        phantom_refs = [p for p in phantom_refs if p not in set(verified_cards)]
        if phantom_refs:
            with write_txn(conn):
                _append_event(
                    conn, task_id, "suspected_hallucinated_references",
                    {
                        "phantom_refs": phantom_refs,
                        "source": "completion_summary",
                    },
                    run_id=run_id,
                )
    # Successful completion — wipe the consecutive-failures counter.
    # Failure history stays on the event log for audit; the counter
    # just tracks "is there a current pathology the breaker should
    # care about", and a success resets that question.
    _clear_failure_counter(conn, task_id)
    # Recompute ready status for dependents (separate txn so children see done).
    recompute_ready(conn)
    # Clean up the scratch workspace and any stale tmux session for the worker.
    _cleanup_workspace(conn, task_id)
    _done_task = get_task(conn, task_id)
    if fire_lifecycle_hook:
        _fire_kanban_lifecycle_hook(
            "kanban_task_completed",
            task_id,
            board=get_current_board(),
            assignee=_done_task.assignee if _done_task else None,
            run_id=run_id,
            summary=(summary if summary is not None else result),
        )
    return True


# ---------------------------------------------------------------------------
# Workspace / tmux cleanup
# ---------------------------------------------------------------------------


def _merge_completion_prose_artifacts(
    conn: sqlite3.Connection,
    task_id: str,
    metadata: Optional[dict],
    *,
    summary: Optional[str],
    result: Optional[str],
) -> Optional[dict]:
    """Promote existing scratch files named in legacy completion prose.

    ``artifacts=[...]`` is preferred. Older workers only wrote an absolute
    deliverable path in ``summary``/``result``; discover it while scratch still
    exists so cleanup cannot erase the file the user was promised.
    """
    row = conn.execute(
        "SELECT workspace_kind, workspace_path FROM tasks "
        "WHERE id = ? AND task_kind = 'work'",
        (task_id,),
    ).fetchone()
    if not row or row["workspace_kind"] != "scratch" or not row["workspace_path"]:
        return metadata
    workspace = Path(row["workspace_path"]).expanduser()
    if not _is_managed_scratch_path(workspace):
        return metadata
    text = "\n".join(part for part in (summary, result) if part)
    if not text:
        return metadata
    prefix = re.escape(str(workspace))
    discovered: list[str] = []
    for match in re.finditer(prefix + r"(?:[/\\][^\s`\"'<>]+)", text):
        raw = match.group(0).rstrip(".,;:!?)]}")
        candidate = Path(raw)
        if candidate.is_file():
            discovered.append(str(candidate))
    if not discovered:
        return metadata
    updated = dict(metadata) if isinstance(metadata, dict) else {}
    existing = updated.get("artifacts")
    merged = list(existing) if isinstance(existing, (list, tuple)) else []
    seen = {str(path) for path in merged}
    for path in discovered:
        if path not in seen:
            merged.append(path)
            seen.add(path)
    updated["artifacts"] = merged
    return updated


def _persist_scratch_completion_artifacts(
    conn: sqlite3.Connection,
    task_id: str,
    metadata: dict,
) -> None:
    """Copy scratch-workspace completion artifacts before cleanup removes them."""
    raw_artifacts = metadata.get("artifacts")
    if not isinstance(raw_artifacts, (list, tuple)):
        return

    row = conn.execute(
        "SELECT workspace_kind, workspace_path FROM tasks "
        "WHERE id = ? AND task_kind = 'work'",
        (task_id,),
    ).fetchone()
    if not row or row["workspace_kind"] != "scratch" or not row["workspace_path"]:
        return

    workspace = Path(row["workspace_path"]).expanduser()
    is_managed, board = _managed_scratch_path_info(workspace)
    if not is_managed:
        return

    try:
        workspace_root = workspace.resolve()
    except OSError:
        return

    attachment_dir = task_attachments_dir(task_id, board=board)
    persisted: list[str] = []
    used_destinations: set[Path] = set()
    changed = False

    def _discard_copies() -> None:
        for copied in used_destinations:
            try:
                copied.unlink(missing_ok=True)
            except OSError:
                pass
        try:
            attachment_dir.rmdir()
        except OSError:
            pass

    for item in raw_artifacts:
        artifact = str(item).strip() if isinstance(item, str) else ""
        if not artifact:
            continue
        src = Path(artifact).expanduser()
        try:
            resolved_src = src.resolve()
        except OSError:
            persisted.append(artifact)
            continue

        if not resolved_src.is_relative_to(workspace_root):
            persisted.append(artifact)
            continue

        if not src.is_file():
            _discard_copies()
            raise ArtifactPreservationError(
                f"declared scratch artifact is unavailable or not a regular file: {artifact}"
            )

        size = resolved_src.stat().st_size
        if size > KANBAN_ATTACHMENT_MAX_BYTES:
            _discard_copies()
            raise ArtifactPreservationError(
                f"declared scratch artifact exceeds the "
                f"{KANBAN_ATTACHMENT_MAX_BYTES}-byte limit: {artifact}"
            )

        dest: Optional[Path] = None
        try:
            attachment_dir.mkdir(parents=True, exist_ok=True)
            dest = _unique_attachment_path(attachment_dir, resolved_src.name, used_destinations)
            with resolved_src.open("rb") as source_file, dest.open("xb") as destination_file:
                copied = 0
                while chunk := source_file.read(1024 * 1024):
                    copied += len(chunk)
                    if copied > KANBAN_ATTACHMENT_MAX_BYTES:
                        raise ArtifactPreservationError(
                            f"declared scratch artifact grew beyond the size limit: {artifact}"
                        )
                    destination_file.write(chunk)
        except Exception as exc:
            if dest is not None:
                try:
                    dest.unlink(missing_ok=True)
                except OSError:
                    pass
            _discard_copies()
            if isinstance(exc, ArtifactPreservationError):
                raise
            raise ArtifactPreservationError(
                f"could not preserve declared scratch artifact {artifact}: {exc}"
            ) from exc

        used_destinations.add(dest)
        persisted.append(str(dest.resolve()))
        changed = True

    if changed:
        metadata["artifacts"] = persisted
        metadata["_staged_artifacts"] = [
            path for path in persisted if path.startswith(str(attachment_dir.resolve()))
        ]


def _insert_completion_attachment(
    conn: sqlite3.Connection,
    task_id: str,
    *,
    filename: str,
    stored_path: str,
    size: int,
    created_at: int,
) -> None:
    """Record a worker-produced artifact in the existing attachment table."""
    conn.execute(
        "INSERT INTO task_attachments "
        "(task_id, filename, stored_path, content_type, size, uploaded_by, created_at) "
        "VALUES (?, ?, ?, NULL, ?, 'kanban_complete', ?)",
        (task_id, filename, stored_path, size, created_at),
    )
    _append_event(
        conn,
        task_id,
        "attached",
        {"filename": filename, "size": size, "by": "kanban_complete"},
    )


def _unique_attachment_path(directory: Path, filename: str, used: set[Path]) -> Path:
    """Return a non-conflicting path under ``directory`` for ``filename``."""
    safe_name = Path(filename).name or "artifact"
    candidate = directory / safe_name
    if candidate not in used and not candidate.exists():
        return candidate

    stem = Path(safe_name).stem or "artifact"
    suffix = Path(safe_name).suffix
    idx = 1
    while True:
        candidate = directory / f"{stem}_{idx}{suffix}"
        if candidate not in used and not candidate.exists():
            return candidate
        idx += 1


def _managed_scratch_path_info(p: Path) -> tuple[bool, Optional[str]]:
    """Return whether *p* is managed scratch storage and the matching board."""
    try:
        p_abs = p.resolve(strict=False)
    except OSError:
        return False, None
    roots: list[tuple[Path, Optional[str]]] = []
    override = os.environ.get("HERMES_KANBAN_WORKSPACES_ROOT", "").strip()
    if override:
        try:
            roots.append((Path(override).expanduser().resolve(strict=False), None))
        except OSError:
            pass
    try:
        home = kanban_home()
    except OSError:
        home = None
    if home is not None:
        try:
            roots.append(((home / "kanban" / "workspaces").resolve(strict=False), DEFAULT_BOARD))
        except OSError:
            pass
        try:
            boards_parent = (home / "kanban" / "boards").resolve(strict=False)
        except OSError:
            boards_parent = None
        if boards_parent is not None:
            try:
                entries = list(boards_parent.iterdir())
            except OSError:
                entries = []
            for entry in entries:
                try:
                    if not entry.is_dir():
                        continue
                except OSError:
                    continue
                try:
                    roots.append(((entry / "workspaces").resolve(strict=False), entry.name))
                except OSError:
                    continue
    for root, board in roots:
        if p_abs == root:
            continue
        try:
            if p_abs.is_relative_to(root):
                return True, board
        except ValueError:
            continue
    return False, None


def _is_managed_scratch_path(p: Path) -> bool:
    """Return True iff *p* is a strict descendant of a kanban-managed scratch root.

    A managed root is exclusively a ``workspaces/`` directory — never the
    broader kanban home, a board root, or sibling subtrees like ``logs/`` or
    ``boards/<slug>/`` itself. Allowed roots:

    * ``HERMES_KANBAN_WORKSPACES_ROOT`` when set (worker-side override
      injected by the dispatcher).
    * ``<kanban_home>/kanban/workspaces`` — legacy default-board scratch root.
    * ``<kanban_home>/kanban/boards/<slug>/workspaces`` for each board slug
      that currently exists on disk.

    The check requires strict descendancy: a path equal to one of these
    roots is NOT managed (deleting the workspaces root would wipe every
    task's scratch dir at once), and a path that resolves to ``<kanban_home>
    /kanban`` itself, ``<kanban_home>/kanban/logs``, or
    ``<kanban_home>/kanban/boards/<slug>`` is rejected because those
    subtrees hold Hermes' own DB, metadata, and logs, not task workspaces.

    Used by :func:`_cleanup_workspace` to refuse to ``shutil.rmtree`` paths
    outside Hermes-managed storage. A board ``default_workdir`` pointing at a
    real source tree can otherwise pair with ``workspace_kind='scratch'`` and
    cause task completion to delete user data (#28818).
    """
    is_managed, _board = _managed_scratch_path_info(p)
    return is_managed


def _cleanup_workspace(conn: sqlite3.Connection, task_id: str) -> None:
    """Remove a task's scratch workspace dir and kill its stale tmux session.

    Called from :func:`complete_task` after the DB transaction commits.
    Best-effort — any error is swallowed so cleanup never blocks task completion.
    ``scratch`` workspaces are removed; ``worktree`` workspaces are removed only
    when provably free of work (clean tree, every commit reachable from a
    remote-tracking ref); ``dir`` workspaces are intentionally preserved.
    """
    try:
        row = conn.execute(
            "SELECT workspace_kind, workspace_path, branch_name FROM tasks WHERE id = ?",
            (task_id,),
        ).fetchone()
        if not row:
            return
        kind: Optional[str] = row["workspace_kind"]
        path: Optional[str] = row["workspace_path"]
        if kind not in ("scratch", "worktree") or not path:
            # This task's own workspace isn't a removable scratch dir, but its
            # completion may still unblock a deferred parent scratch cleanup
            # (e.g. a 'dir' child whose scratch parent was waiting on it). #33774
            _try_cleanup_parent_workspaces(conn, task_id)
            return
        # Check if this task has children that still need the workspace.
        # If any child is not yet done/archived, defer cleanup so the
        # child can read handoff artifacts from the workspace (#33774).
        _active_children = conn.execute(
            "SELECT 1 FROM task_links l "
            "JOIN tasks t ON t.id = l.child_id "
            "WHERE l.parent_id = ? AND t.status NOT IN ('done', 'archived', 'failed', 'cancelled') "
            "LIMIT 1",
            (task_id,),
        ).fetchone()
        if _active_children:
            _log.debug(
                "Deferring %s workspace cleanup for task %s: "
                "active children still need workspace at %s",
                kind, task_id, path,
            )
            return
        if kind == "worktree":
            # Kill the (dead) tmux worker session BEFORE removing the
            # worktree so a lingering worker never has its cwd deleted out
            # from under it. Both steps stay best-effort.
            _cleanup_worker_tmux(conn, task_id)
            _cleanup_worktree_workspace(task_id, path, row["branch_name"])
            _try_cleanup_parent_workspaces(conn, task_id)
            return
        import shutil
        wp = Path(path)
        if wp.is_dir():
            # Containment guard (#28818): a board's ``default_workdir`` can
            # pair ``workspace_kind='scratch'`` with a user-supplied path
            # pointing at a real source tree. Without this check, task
            # completion would unconditionally ``shutil.rmtree`` that path
            # and silently delete the user's source data.
            if _is_managed_scratch_path(wp):
                shutil.rmtree(wp, ignore_errors=True)
                _log.debug("Removed scratch workspace: %s", wp)
            else:
                _log.warning(
                    "Refusing to remove out-of-scratch workspace for task %s: %s "
                    "(workspace_kind='scratch' but path is outside any "
                    "kanban-managed workspaces root)",
                    task_id, wp,
                )
        # Also kill the tmux session for the worker that owned this task,
        # if the tmux session is now dead (worker process exited).
        _cleanup_worker_tmux(conn, task_id)
        # After cleaning up this task's workspace, check if any parent
        # tasks now have all children done — their deferred cleanup can
        # proceed (#33774).
        _try_cleanup_parent_workspaces(conn, task_id)
    except Exception:
        pass  # best-effort — never block completion


def _cleanup_worktree_workspace(
    task_id: str, path: str, branch_name: Optional[str] = None
) -> None:
    """Remove a finished task's linked git worktree when it holds no work.

    Mirrors the safety judgment of the CLI startup pruner
    (``cli._prune_stale_worktrees``): removal requires a clean working tree
    AND every commit reachable from a remote-tracking ref. Any doubt — dirty
    files, unpushed commits, unresolvable repo, failing git — preserves the
    worktree. The task's auto-generated ``wt/<task-id>`` branch is deleted
    with it; custom branches are kept. Best-effort like the scratch path.
    """
    try:
        from cli import _worktree_has_unpushed_commits, _worktree_is_dirty
    except Exception:
        return  # CLI safety predicates unavailable — preserve
    try:
        wp = Path(path).expanduser()
        if not wp.is_dir():
            return
        common = _git_common_dir(wp)
        if common is None or common.name != ".git":
            return  # not a linked worktree of a normal repo — never guess
        repo_root = common.parent
        if wp.resolve(strict=False) == repo_root.resolve(strict=False):
            return  # never remove the main checkout
        if _worktree_is_dirty(str(wp)) or _worktree_has_unpushed_commits(str(wp)):
            _log.info(
                "Preserving worktree for task %s: dirty or unpushed work at %s",
                task_id, wp,
            )
            return
        # No --force: the dirty/unpushed checks above run before removal, so
        # git's own dirty guard re-verifies at removal time. If the tree
        # became dirty between our check and the removal (TOCTOU), removal
        # fails safe and the worktree is preserved.
        result = subprocess.run(
            ["git", "-C", str(repo_root), "worktree", "remove", str(wp)],
            capture_output=True,
            text=True, encoding='utf-8', errors='replace',
            timeout=60,
            check=False,
        )
        if result.returncode != 0:
            _log.warning(
                "git worktree remove failed for task %s at %s: %s",
                task_id, wp, (result.stderr or result.stdout or "").strip(),
            )
            return
        _log.debug("Removed worktree workspace: %s", wp)
        branch = (branch_name or "").strip() or f"wt/{task_id}"
        if branch.startswith("wt/"):
            subprocess.run(
                ["git", "-C", str(repo_root), "branch", "-D", branch],
                capture_output=True,
                text=True, encoding='utf-8', errors='replace',
                timeout=30,
                check=False,
            )
    except Exception:
        pass  # best-effort — never block completion


def _try_cleanup_parent_workspaces(conn: sqlite3.Connection, task_id: str) -> None:
    """Clean up parent scratch workspaces now that *task_id* completed.

    When a parent task's cleanup was deferred because it had active children,
    this function is called after each child completes.  If all children of a
    parent are now done/archived/failed/cancelled, the parent's scratch
    workspace is removed (#33774).
    """
    try:
        parents = conn.execute(
            "SELECT parent_id FROM task_links WHERE child_id = ?",
            (task_id,),
        ).fetchall()
        for (parent_id,) in parents:
            row = conn.execute(
                "SELECT workspace_kind, workspace_path, branch_name FROM tasks WHERE id = ?",
                (parent_id,),
            ).fetchone()
            if (
                not row
                or row["workspace_kind"] not in ("scratch", "worktree")
                or not row["workspace_path"]
            ):
                continue
            # Check if ALL children of this parent are terminal
            active = conn.execute(
                "SELECT 1 FROM task_links l "
                "JOIN tasks t ON t.id = l.child_id "
                "WHERE l.parent_id = ? AND t.status NOT IN ('done', 'archived', 'failed', 'cancelled') "
                "LIMIT 1",
                (parent_id,),
            ).fetchone()
            if active:
                continue  # still has active children
            # All children done — safe to clean up parent workspace
            if row["workspace_kind"] == "worktree":
                _cleanup_worktree_workspace(
                    parent_id, row["workspace_path"], row["branch_name"]
                )
                continue
            import shutil
            wp = Path(row["workspace_path"])
            if wp.is_dir() and _is_managed_scratch_path(wp):
                shutil.rmtree(wp, ignore_errors=True)
                _log.debug("Deferred cleanup: removed parent %s scratch workspace: %s", parent_id, wp)
    except Exception:
        pass  # best-effort


def _cleanup_worker_tmux(conn: sqlite3.Connection, task_id: str) -> None:
    """Kill the tmux session associated with a task's assignee, if dead."""
    try:
        row = conn.execute(
            "SELECT assignee FROM tasks WHERE id = ?", (task_id,)
        ).fetchone()
        if not row or not row["assignee"]:
            return
        assignee: str = row["assignee"]
        # Workers named swarm1-12 use tmux sessions named swarm-swarm1 etc.
        session = f"swarm-{assignee}"
        # Check if session exists and pane is dead before killing
        out = subprocess.run(
            ["tmux", "list-panes", "-t", session, "-F", "#{pane_dead}"],
            capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=5,
        )
        if out.stdout.strip() == "1":
            subprocess.run(
                ["tmux", "kill-session", "-t", session],
                capture_output=True, timeout=5,
            )
            _log.debug("Killed stale tmux session: %s", session)
    except Exception:
        pass  # best-effort — never block completion


# ---------------------------------------------------------------------------
# First-use tip for scratch workspaces
# ---------------------------------------------------------------------------
#
# Scratch workspaces are intentionally ephemeral — ``_cleanup_workspace``
# removes them as soon as ``complete_task`` runs.  New users often don't
# realize that and lose worker output (community report, May 2026).  The
# behavior is right; the lack of warning is the bug.
#
# On the FIRST scratch workspace materialization across the whole install
# we:
#   1. Log a warning line on the dispatcher logger.
#   2. Append a ``tip_scratch_workspace`` event on the task so it's visible
#      via ``hermes kanban show <id>`` and the dashboard.
#   3. Touch a sentinel file under ``kanban_home() / '.scratch_tip_shown'``
#      so we don't repeat the tip — once you know, you know.
#
# Scope is per-install, not per-board: a user creating a second board
# already learned the lesson on board #1.

_SCRATCH_TIP_SENTINEL_NAME = ".scratch_tip_shown"

_SCRATCH_TIP_MESSAGE = (
    "scratch workspaces are ephemeral — they're deleted when the task "
    "completes. Use --workspace worktree: (git worktree) or "
    "--workspace dir:/abs/path (existing dir) to preserve worker output."
)


def _scratch_tip_sentinel_path() -> Path:
    """Path to the per-install scratch-workspace-tip sentinel file."""
    return kanban_home() / _SCRATCH_TIP_SENTINEL_NAME


def _scratch_tip_shown() -> bool:
    """True iff the scratch-workspace tip has already been emitted on this
    install. Best-effort — any error means we re-emit, which is the safer
    failure mode for a help message."""
    try:
        return _scratch_tip_sentinel_path().exists()
    except OSError:
        return False


def _mark_scratch_tip_shown() -> None:
    """Touch the sentinel so future scratch workspaces stay silent.

    Best-effort: a failure here just means the tip might appear once more,
    which is preferable to crashing dispatch over a help message.
    """
    try:
        path = _scratch_tip_sentinel_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch(exist_ok=True)
    except OSError:
        pass


def _maybe_emit_scratch_tip(
    conn: sqlite3.Connection,
    task_id: str,
    workspace_kind: Optional[str],
) -> None:
    """Emit the first-use scratch-workspace tip exactly once per install.

    Called from the dispatcher right after a scratch workspace is
    materialized. No-op for ``worktree`` / ``dir`` workspaces (they're
    preserved by design) and no-op after the sentinel exists.
    """
    if (workspace_kind or "scratch") != "scratch":
        return
    if _scratch_tip_shown():
        return
    try:
        _log.warning("kanban: %s (task %s)", _SCRATCH_TIP_MESSAGE, task_id)
        with write_txn(conn):
            _append_event(
                conn, task_id, "tip_scratch_workspace",
                {"message": _SCRATCH_TIP_MESSAGE},
            )
    except Exception:
        # Best-effort — never block the spawn loop over a help message.
        pass
    finally:
        _mark_scratch_tip_shown()


def edit_completed_task_result(
    conn: sqlite3.Connection,
    task_id: str,
    *,
    result: str,
    summary: Optional[str] = None,
    metadata: Optional[dict] = None,
) -> bool:
    """Backfill the user-visible result for an already completed task."""
    handoff_summary = summary if summary is not None else result
    with write_txn(conn):
        row = conn.execute(
            "SELECT status FROM tasks WHERE id = ? AND task_kind = 'work'",
            (task_id,),
        ).fetchone()
        if not row or row["status"] != "done":
            return False
        conn.execute(
            "UPDATE tasks SET result = ? WHERE id = ? AND task_kind = 'work'",
            (result, task_id),
        )
        run = conn.execute(
            """
            SELECT id FROM task_runs
             WHERE task_id = ?
               AND outcome = 'completed'
             ORDER BY COALESCE(ended_at, started_at, 0) DESC, id DESC
             LIMIT 1
            """,
            (task_id,),
        ).fetchone()
        run_id = int(run["id"]) if run else None
        if run_id is None:
            run_id = _synthesize_ended_run(
                conn, task_id,
                outcome="completed",
                summary=handoff_summary,
                metadata=metadata,
            )
        else:
            conn.execute(
                "UPDATE task_runs SET summary = ? WHERE id = ?",
                (handoff_summary, run_id),
            )
            if metadata is not None:
                conn.execute(
                    "UPDATE task_runs SET metadata = ? WHERE id = ?",
                    (json.dumps(metadata, ensure_ascii=False), run_id),
                )
        _ev_lines = (handoff_summary or "").strip().splitlines()
        ev_summary = _ev_lines[0][:400] if _ev_lines else ""
        _append_event(
            conn, task_id, "edited",
            {
                "fields": (
                    ["result", "summary"]
                    + (["metadata"] if metadata is not None else [])
                ),
                "result_len": len(result) if result else 0,
                "summary": ev_summary or None,
            },
            run_id=run_id,
        )
    return True


def block_task(
    conn: sqlite3.Connection,
    task_id: str,
    *,
    reason: Optional[str] = None,
    kind: Optional[str] = None,
    expected_run_id: Optional[int] = None,
) -> bool:
    """Transition ``running``/``ready`` → ``blocked`` (or route elsewhere).

    ``kind`` (one of :data:`VALID_BLOCK_KINDS`, or ``None`` for a legacy
    un-typed block) drives routing instead of every block landing in one
    undifferentiated ``blocked`` bucket:

    * ``dependency`` — the task is only waiting on another task. It does NOT
      sit in ``blocked`` (where a cron would keep "unblocking" it); it goes to
      ``todo`` so the existing parent-gating / ``recompute_ready`` machinery
      promotes it automatically once its parents finish. No human, no cron, no
      retry storm. This is Dale's "Type 2 — dependency blocked".

    * ``needs_input`` / ``capability`` / ``None`` — "truly blocked" (Dale's
      "Type 1"). Lands in ``blocked`` for a human. BUT: each time such a task
      is re-blocked for the SAME kind after having been unblocked, the
      unblock-loop counter (``block_recurrences``) increments. When it reaches
      :data:`BLOCK_RECURRENCE_LIMIT`, the task is routed to ``triage`` instead
      of ``blocked`` — breaking the cron-unblock ↔ worker-re-block loop and
      forcing a human-in-the-loop triage decision.

    * ``transient`` — treated like a generic block for routing, but a worker
      can use it to signal "this might clear on its own"; it still participates
      in the loop breaker so a forever-flaky task eventually escalates.

    Returns True on any successful transition (to ``blocked``, ``todo``, or
    ``triage``), False when the task wasn't in a blockable state.
    """
    if kind is not None and kind not in VALID_BLOCK_KINDS:
        raise ValueError(
            f"block kind must be one of {sorted(VALID_BLOCK_KINDS)} or None"
        )
    recurrences = 0
    with write_txn(conn):
        cur_row = conn.execute(
            "SELECT status, block_kind, block_recurrences FROM tasks "
            "WHERE id = ? AND task_kind = 'work'",
            (task_id,),
        ).fetchone()
        if cur_row is None:
            return False
        source_status = (
            _retry_status_for_run(conn, task_id)
            if cur_row["status"] == "running"
            else "ready"
        )
        prev_kind = cur_row["block_kind"] if "block_kind" in cur_row.keys() else None
        prev_recurrences = (
            int(cur_row["block_recurrences"])
            if "block_recurrences" in cur_row.keys()
            and cur_row["block_recurrences"] is not None
            else 0
        )

        # Dependency blocks never enter the human ``blocked`` bucket — they
        # wait in ``todo`` and let ``recompute_ready`` gate on parents. Routing
        # here (rather than ``blocked``) is what keeps a cron from ever seeing
        # a dependency-wait as something to "unblock".
        if kind == "dependency":
            cur = conn.execute(
                """
                UPDATE tasks
                   SET status        = 'todo',
                       claim_lock    = NULL,
                       claim_expires = NULL,
                       worker_pid    = NULL,
                       block_kind    = ?
                 WHERE id = ?
                   AND status IN ('running', 'ready')
                """ + ("" if expected_run_id is None else " AND current_run_id = ?"),
                (kind, task_id) if expected_run_id is None
                else (kind, task_id, int(expected_run_id)),
            )
            if cur.rowcount != 1:
                return False
            run_id = _end_run(
                conn, task_id,
                outcome="blocked", status="blocked",
                summary=reason,
            )
            if run_id is None and reason:
                run_id = _synthesize_ended_run(
                    conn, task_id, outcome="blocked", summary=reason,
                )
            _append_event(
                conn, task_id, "dependency_wait",
                {
                    "reason": reason,
                    "kind": kind,
                    "source_status": source_status,
                },
                run_id=run_id,
            )
            _blocked_task = get_task(conn, task_id)
            _fire_kanban_lifecycle_hook(
                "kanban_task_blocked",
                task_id,
                board=get_current_board(),
                assignee=_blocked_task.assignee if _blocked_task else None,
                run_id=run_id,
                reason=reason,
            )
            return True

        # Truly-blocked kinds. Increment the unblock-loop counter when this is a
        # re-block for the SAME reason after a prior unblock. block_task only
        # fires from running/ready (i.e. AFTER an unblock returned the task to
        # the work pool), so a stored block_kind that matches the incoming kind
        # means: blocked → unblocked → about-to-re-block for the same cause.
        # An un-typed (None) block compares as "same" to a prior un-typed block.
        same_cause = prev_kind == kind
        recurrences = prev_recurrences + 1 if same_cause else 1

        if recurrences >= BLOCK_RECURRENCE_LIMIT:
            # Loop detected — stop letting the unblocker spin this task. Route
            # to triage for a human-in-the-loop decision instead of blocked.
            cur = conn.execute(
                """
                UPDATE tasks
                   SET status        = 'triage',
                       claim_lock    = NULL,
                       claim_expires = NULL,
                       worker_pid    = NULL,
                       block_kind    = ?,
                       block_recurrences = ?
                 WHERE id = ?
                   AND status IN ('running', 'ready')
                """ + ("" if expected_run_id is None else " AND current_run_id = ?"),
                (kind, recurrences, task_id) if expected_run_id is None
                else (kind, recurrences, task_id, int(expected_run_id)),
            )
            if cur.rowcount != 1:
                return False
            run_id = _end_run(
                conn, task_id,
                outcome="blocked", status="blocked",
                summary=reason,
            )
            if run_id is None and reason:
                run_id = _synthesize_ended_run(
                    conn, task_id, outcome="blocked", summary=reason,
                )
            _append_event(
                conn, task_id, "block_loop_detected",
                {
                    "reason": reason,
                    "kind": kind,
                    "recurrences": recurrences,
                    "limit": BLOCK_RECURRENCE_LIMIT,
                    "source_status": source_status,
                },
                run_id=run_id,
            )
        else:
            if expected_run_id is None:
                cur = conn.execute(
                    """
                    UPDATE tasks
                       SET status        = 'blocked',
                           claim_lock    = NULL,
                           claim_expires = NULL,
                           worker_pid    = NULL,
                           block_kind    = ?,
                           block_recurrences = ?
                     WHERE id = ?
                       AND status IN ('running', 'ready')
                    """,
                    (kind, recurrences, task_id),
                )
            else:
                cur = conn.execute(
                    """
                    UPDATE tasks
                       SET status        = 'blocked',
                           claim_lock    = NULL,
                           claim_expires = NULL,
                           worker_pid    = NULL,
                           block_kind    = ?,
                           block_recurrences = ?
                     WHERE id = ?
                       AND status IN ('running', 'ready')
                       AND current_run_id = ?
                    """,
                    (kind, recurrences, task_id, int(expected_run_id)),
                )
            if cur.rowcount != 1:
                return False
            run_id = _end_run(
                conn, task_id,
                outcome="blocked", status="blocked",
                summary=reason,
            )
            # Synthesize a run when blocking a never-claimed task so the
            # reason is preserved in attempt history.
            if run_id is None and reason:
                run_id = _synthesize_ended_run(
                    conn, task_id,
                    outcome="blocked",
                    summary=reason,
                )
            _append_event(
                conn, task_id, "blocked",
                {
                    "reason": reason,
                    "kind": kind,
                    "recurrences": recurrences,
                    "source_status": source_status,
                },
                run_id=run_id,
            )
        _blocked_task = get_task(conn, task_id)
    _fire_kanban_lifecycle_hook(
        "kanban_task_blocked",
        task_id,
        board=get_current_board(),
        assignee=_blocked_task.assignee if _blocked_task else None,
        run_id=run_id,
        reason=reason,
    )
    return True



def redact_review_value(value: Any) -> Any:
    """Redact secrets at the domain boundary for durable review handoffs."""
    if isinstance(value, str):
        from agent.redact import redact_sensitive_text

        return redact_sensitive_text(value, force=True)
    if isinstance(value, dict):
        return {key: redact_review_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [redact_review_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(redact_review_value(item) for item in value)
    return value


def request_review(
    conn: sqlite3.Connection,
    task_id: str,
    *,
    summary: Optional[str] = None,
    metadata: Optional[dict] = None,
    reviewer: Optional[str] = None,
    expected_run_id: Optional[int] = None,
    force: bool = False,
    with_reason: bool = False,
):
    """Transition implementation work into the first-class review phase.

    Unlike :func:`block_task`, this transition never touches block recurrence
    accounting.  The current implementer and resolved reviewer are recorded on
    the event so an autonomous reviewer can route requested changes back to the
    right profile.  Supplying ``reviewer`` reassigns the task before it is
    exposed to the review dispatcher.  On re-review, omitting it reuses the
    reviewer provenance persisted by the latest ``changes_requested`` event.

    When the task is ``running`` under a live claim, a caller that supplies no
    ``expected_run_id`` must pass ``force=True`` (explicit human/CLI override)
    — otherwise the request is refused instead of silently clearing the live
    worker's ``claim_lock``/``worker_pid``. Workers prove ownership by passing
    their own run id as ``expected_run_id`` (unchanged).

    Returns ``bool`` by default. With ``with_reason=True`` returns
    ``(ok, reason)`` mirroring :func:`request_changes` — ``reason`` is a
    diagnostic string on failure, ``None`` on success.
    """

    def _ret(ok: bool, reason: Optional[str] = None):
        return (ok, reason) if with_reason else ok

    summary = redact_review_value(summary)
    metadata = redact_review_value(metadata)
    with write_txn(conn):
        if not _parents_satisfied(conn, task_id):
            return _ret(False, "parent dependencies are not satisfied")
        trow = conn.execute(
            "SELECT assignee, status, claim_lock, current_run_id "
            "FROM tasks WHERE id = ? AND task_kind = 'work'", (task_id,),
        ).fetchone()
        if trow is None:
            return _ret(False, "task not found")
        # Refuse to clear a live worker's claim without proof of ownership
        # (expected_run_id) or an explicit human override (force=True).
        if (
            expected_run_id is None
            and not force
            and trow["status"] == "running"
            and trow["claim_lock"] is not None
        ):
            return _ret(
                False,
                "task is running under a live claim; pass expected_run_id "
                "(worker ownership) or force=True (explicit operator "
                "override) instead of clearing the live run's claim",
            )
        implementer = trow["assignee"]
        if reviewer is None:
            changes_run = conn.execute(
                "SELECT id FROM task_runs "
                "WHERE task_id = ? AND outcome = 'changes_requested' "
                "ORDER BY id DESC LIMIT 1",
                (task_id,),
            ).fetchone()
            changes_event = None
            if changes_run is not None:
                changes_event = conn.execute(
                    "SELECT payload FROM task_events "
                    "WHERE task_id = ? AND run_id = ? "
                    "AND kind = 'changes_requested' "
                    "ORDER BY id DESC LIMIT 1",
                    (task_id, int(changes_run["id"])),
                ).fetchone()
            try:
                changes_payload = (
                    json.loads(changes_event["payload"])
                    if changes_event and changes_event["payload"]
                    else {}
                )
            except (json.JSONDecodeError, TypeError):
                changes_payload = {}
            prior_reviewer = (
                changes_payload.get("reviewer")
                if isinstance(changes_payload, dict)
                else None
            )
            if changes_run is not None:
                if not isinstance(prior_reviewer, str) or not prior_reviewer.strip():
                    return _ret(
                        False,
                        "re-review has no durable reviewer provenance (the "
                        "latest changes_requested event is missing or "
                        "malformed); pass reviewer= explicitly",
                    )
                reviewer = prior_reviewer
        reviewer = _canonical_assignee(reviewer) if reviewer is not None else None
        # The independent reviewer is a different role, so this is a role
        # transition and goes through the one authority helper. A policy-locked
        # task refuses the handoff: its route was approved for one assignee for
        # its whole run, so independent review has to be separately approved
        # work rather than a silent re-pin of this task.
        role_transition_route(conn, task_id, reviewer)
        assignee_sql = ", assignee = ?" if reviewer is not None else ""
        lead: tuple[Any, ...] = (reviewer,) if reviewer is not None else ()
        params: tuple[Any, ...]
        if expected_run_id is None:
            params = (*lead, task_id)
            run_guard = ""
        else:
            params = (*lead, task_id, int(expected_run_id))
            run_guard = " AND current_run_id = ?"
        cur = conn.execute(
            """
            UPDATE tasks
               SET status        = 'review',
                   claim_lock    = NULL,
                   claim_expires = NULL,
                   worker_pid    = NULL
            """ + assignee_sql + """
             WHERE id = ?
               AND status IN ('running', 'ready')
            """ + run_guard,
            params,
        )
        if cur.rowcount != 1:
            return _ret(
                False,
                "task is not in running/ready (or expected_run_id did not "
                "match the current run)",
            )
        run_id = _end_run(
            conn,
            task_id,
            outcome="review_requested",
            status="review",
            summary=summary,
            metadata=metadata,
        )
        if run_id is None and (summary or metadata):
            run_id = _synthesize_ended_run(
                conn,
                task_id,
                outcome="review_requested",
                summary=summary,
                metadata=metadata,
            )
        lines = (summary or "").strip().splitlines()
        event_summary = lines[0][:400] if lines else ""
        _append_event(
            conn,
            task_id,
            "review_requested",
            {
                "summary": event_summary or None,
                "implementer": implementer,
                "reviewer": reviewer,
            },
            run_id=run_id,
        )
    return _ret(True)


def request_changes(
    conn: sqlite3.Connection,
    task_id: str,
    *,
    reason: str,
    expected_run_id: Optional[int] = None,
) -> tuple[bool, Optional[str]]:
    """Finish an active review run and route the task back for rework.

    The transition is valid only for a run claimed from ``review``.  It closes
    that reviewer run, restores the implementer recorded by the latest
    ``review_requested`` event, reapplies parent gating, and emits an auditable
    ``changes_requested`` event.  The second tuple item is the implementer on
    success or a diagnostic reason on failure.
    """
    reason = str(redact_review_value(reason or "")).strip()
    if not reason:
        return False, "reason is required"

    with write_txn(conn):
        task_row = conn.execute(
            "SELECT status, assignee, current_run_id FROM tasks "
            "WHERE id = ? AND task_kind = 'work'",
            (task_id,),
        ).fetchone()
        if task_row is None:
            return False, "task not found"
        current_run_id = task_row["current_run_id"]
        if task_row["status"] != "running" or current_run_id is None:
            return False, "task is not in an active review run"
        if expected_run_id is not None and int(current_run_id) != int(expected_run_id):
            return False, "run_id mismatch"

        claimed_event = conn.execute(
            "SELECT payload FROM task_events "
            "WHERE task_id = ? AND run_id = ? AND kind = 'claimed' "
            "ORDER BY id DESC LIMIT 1",
            (task_id, int(current_run_id)),
        ).fetchone()
        try:
            claimed_payload = (
                json.loads(claimed_event["payload"])
                if claimed_event and claimed_event["payload"]
                else {}
            )
        except (json.JSONDecodeError, TypeError):
            claimed_payload = {}
        if not isinstance(claimed_payload, dict):
            claimed_payload = {}
        if claimed_payload.get("source_status") != "review":
            return False, "active run was not claimed from review"

        requested_event = conn.execute(
            "SELECT payload FROM task_events "
            "WHERE task_id = ? AND kind = 'review_requested' "
            "ORDER BY id DESC LIMIT 1",
            (task_id,),
        ).fetchone()
        if requested_event is None:
            return False, "no prior review_requested event"
        try:
            requested_payload = (
                json.loads(requested_event["payload"])
                if requested_event["payload"]
                else {}
            )
        except (json.JSONDecodeError, TypeError):
            requested_payload = {}
        if not isinstance(requested_payload, dict):
            requested_payload = {}
        implementer = requested_payload.get("implementer")
        if not isinstance(implementer, str) or not implementer.strip():
            return False, "review handoff has no valid implementer provenance"
        reviewer = task_row["assignee"]
        if isinstance(reviewer, str) and reviewer.strip():
            reviewer = _canonical_assignee(reviewer)
        else:
            reviewer = None

        new_status = _landing_status_after_parents(conn, task_id)
        # Handing the work back to the implementer is a role transition, so it
        # goes through the one authority helper rather than writing `assignee`
        # itself. A policy-locked task refuses the handback: its route was
        # approved for one assignee for its whole run, and rework has to be
        # separately approved work.
        role_transition_route(conn, task_id, implementer)
        # NOTE: consecutive_failures is deliberately PRESERVED (neither
        # reset nor incremented). Review transitions are not evidence the
        # pathology cleared — only complete_task's success path resets the
        # breaker counter (mirrors unblock_task, #35072).
        cur = conn.execute(
            """
            UPDATE tasks
               SET status = ?,
                   assignee = COALESCE(?, assignee),
                   claim_lock = NULL,
                   claim_expires = NULL,
                   worker_pid = NULL
             WHERE id = ? AND status = 'running' AND current_run_id = ?
            """,
            (new_status, implementer, task_id, int(current_run_id)),
        )
        if cur.rowcount != 1:
            return False, "task changed during review handoff"
        run_id = _end_run(
            conn,
            task_id,
            outcome="changes_requested",
            status=new_status,
            summary=reason,
        )
        _append_event(
            conn,
            task_id,
            "changes_requested",
            {
                "reason": reason,
                "implementer": implementer,
                "reviewer": reviewer,
                "status": new_status,
            },
            run_id=run_id,
        )
    return True, implementer


def promote_task(
    conn: sqlite3.Connection,
    task_id: str,
    *,
    actor: str,
    reason: Optional[str] = None,
    force: bool = False,
    dry_run: bool = False,
) -> tuple[bool, Optional[str]]:
    """Manually promote a `todo` or `blocked` task to `ready`.

    Mirrors the automatic promotion done by ``recompute_ready`` but
    drives it from a deliberate operator action with an audit-trail
    entry. Refuses to promote if any parent dep is not in a terminal
    state (`done`/`archived`) unless ``force=True``. Does NOT change
    assignee or claim state. Returns ``(True, None)`` on success and
    ``(False, reason)`` if refused. ``dry_run=True`` validates the
    promotion would succeed without mutating state.
    """
    row = conn.execute(
        "SELECT status FROM tasks WHERE id = ? AND task_kind = 'work'", (task_id,)
    ).fetchone()
    if row is None:
        return False, f"task {task_id} not found"

    cur_status = row["status"]
    if cur_status not in ("todo", "blocked"):
        return False, (
            f"task {task_id} is {cur_status!r}; promote only applies to "
            f"'todo' or 'blocked'"
        )

    if not force:
        parents = conn.execute(
            "SELECT t.id, t.status FROM tasks t "
            "JOIN task_links l ON l.parent_id = t.id "
            f"WHERE l.child_id = ? AND t.task_kind IN {_DEPENDENCY_PARENT_KINDS}",
            (task_id,),
        ).fetchall()
        unsatisfied = [
            p["id"] for p in parents
            if p["status"] not in ("done", "archived")
        ]
        if unsatisfied:
            return False, (
                f"unsatisfied parent dependencies: "
                f"{', '.join(unsatisfied)} (use --force to override)"
            )

    if dry_run:
        # The same authority question the real promotion asks, answered
        # without minting a lock, parking the row or writing an event. Without
        # it a dry run reported an unpinnable owner task as promotable while
        # the real operation refuses it.
        if route_authority_error(conn, task_id) is not None:
            return False, (
                f"task {task_id} cannot be promoted: its approved model route "
                "cannot be confirmed, so it is parked for re-approval"
            )
        return True, None

    with write_txn(conn):
        if not authorize_executable_transition(conn, task_id):
            return False, (
                f"task {task_id} cannot be promoted: its approved model route "
                "cannot be confirmed, so it is parked for re-approval"
            )
        upd = conn.execute(
            "UPDATE tasks SET status = 'ready' "
            "WHERE id = ? AND status IN ('todo', 'blocked') AND task_kind = 'work'",
            (task_id,),
        )
        if upd.rowcount != 1:
            return False, f"task {task_id} status changed during promotion"
        _append_event(
            conn,
            task_id,
            "promoted_manual",
            {"actor": actor, "reason": reason, "forced": force},
        )

    return True, None


def _reclaim_dangling_run(
    conn: sqlite3.Connection, task_id: str, *, statuses, now: int, note: str,
) -> None:
    """Close a leaked ``current_run_id`` (run row still open) before a status
    flip, preserving the runs invariant (``current_run_id IS NULL`` ⇔ run row
    terminal). No-op in the common path where the prior transition already
    closed the run. Shared by :func:`unblock_task` and
    :func:`reopen_review_task` so the recovery can't drift.
    """
    placeholders = ", ".join("?" for _ in statuses)
    stale = conn.execute(
        f"SELECT current_run_id FROM tasks WHERE id = ? AND status IN ({placeholders}) "
        "AND task_kind = 'work'",
        (task_id, *statuses),
    ).fetchone()
    if stale and stale["current_run_id"]:
        conn.execute(
            """
            UPDATE task_runs
               SET status = 'reclaimed', outcome = 'reclaimed',
                   summary = COALESCE(summary, ?),
                   ended_at = ?,
                   claim_lock = NULL, claim_expires = NULL, worker_pid = NULL
             WHERE id = ? AND ended_at IS NULL
            """,
            (note, now, int(stale["current_run_id"])),
        )


def _landing_status_after_parents(conn: sqlite3.Connection, task_id: str) -> str:
    """Return ``'todo'`` if any parent isn't ``done`` yet, else ``'ready'``.

    The parent-completion re-gate shared by :func:`unblock_task` and
    :func:`reopen_review_task`: flipping straight to ``ready`` would bypass the
    parent-completion invariant the dispatcher trusts (it would spawn a child
    whose upstream work isn't finished). If parents are still in progress the
    task waits in ``todo`` until ``recompute_ready`` picks it up. RCA: Bug 2 at
    kanban/boards/cookai/workspaces/t_a6acd07d/root-cause.md. Kept in one place
    so the two transitions can't drift.
    """
    undone_parents = conn.execute(
        "SELECT 1 FROM task_links l "
        "JOIN tasks p ON p.id = l.parent_id "
        "WHERE l.child_id = ? "
        f"AND p.task_kind IN {_DEPENDENCY_PARENT_KINDS} "
        "AND p.status NOT IN ('done', 'archived') LIMIT 1",
        (task_id,),
    ).fetchone()
    return "todo" if undone_parents else "ready"


def unblock_task(conn: sqlite3.Connection, task_id: str) -> bool:
    """Transition ``blocked``/``scheduled`` to its safe resumable phase.

    Defensively closes any stale ``current_run_id`` pointer before flipping
    status. In the common path (``block_task`` closed the run already) this
    is a no-op. If a future or external write left the pointer dangling,
    the leaked run is closed as ``reclaimed`` inside the same txn so the
    runs invariant (``current_run_id IS NULL`` ⇔ run row in terminal
    state) holds for the rest of this function's lifetime.
    """
    now = int(time.time())
    with write_txn(conn):
        current = conn.execute(
            "SELECT status FROM tasks WHERE id = ? AND task_kind = 'work'",
            (task_id,),
        ).fetchone()
        resume_status = (
            _resume_status_from_events(conn, task_id)
            if current and current["status"] == "blocked"
            else "ready"
        )
        _reclaim_dangling_run(
            conn, task_id, statuses=("blocked", "scheduled"), now=now,
            note="invariant recovery on unblock",
        )
        # Re-gate on parent completion before restoring the source phase.
        landing_status = _landing_status_after_parents(conn, task_id)
        new_status = (
            "review"
            if landing_status == "ready" and resume_status == "review"
            else landing_status
        )
        if new_status in EXECUTABLE_STATUSES and not authorize_executable_transition(
            conn, task_id
        ):
            # Parked for re-approval instead of unblocked into the work pool.
            return False
        # NOTE: deliberately does NOT touch ``block_recurrences`` or
        # ``block_kind``. Resetting the recurrence counter on unblock is exactly
        # the amnesia that let a cron unblock → worker re-block loop run
        # unbounded (Dale's report). The counter survives the unblock so that a
        # subsequent same-cause ``block_task`` can detect the loop and route to
        # triage at ``BLOCK_RECURRENCE_LIMIT``. It is reset to 0 only on a
        # successful completion (see ``complete_task``). ``consecutive_failures``
        # (the *dispatcher* spawn/crash/timeout counter — a different signal) is
        # still reset here, which is correct: a deliberate unblock is a fresh
        # start for the dispatcher's retry budget.
        cur = conn.execute(
            "UPDATE tasks SET status = ?, current_run_id = NULL, "
            "consecutive_failures = 0, last_failure_error = NULL "
            "WHERE id = ? AND status IN ('blocked', 'scheduled') "
            "AND task_kind = 'work'",
            (new_status, task_id),
        )
        if cur.rowcount != 1:
            return False
        _append_event(
            conn, task_id, "unblocked",
            (
                {"status": new_status, "resume_status": resume_status}
                if new_status != "ready" or resume_status != "ready"
                else None
            ),
        )
        return True


def reopen_review_task(conn: sqlite3.Connection, task_id: str) -> bool:
    """Transition ``review`` -> ready (or todo) so the implementer re-runs.

    The "changes requested" counterpart of :func:`request_review`: sends the
    task back out of the review lane so the dispatcher re-runs the implementer
    on the new comments. Mirrors :func:`unblock_task` (parent re-gating,
    defensive stale-run close, ``consecutive_failures`` preserved) and emits a
    ``review_reopened`` event.

    Deliberately does NOT touch ``block_recurrences``/``block_kind``: review is
    not a block, so there is no loop counter to reset. (A stale counter from a
    genuine block *before* review is left intact — only :func:`complete_task`
    clears it.) Returns False when the task is missing or not in ``review``.
    """
    now = int(time.time())
    with write_txn(conn):
        _reclaim_dangling_run(
            conn, task_id, statuses=("review",), now=now,
            note="invariant recovery on review reopen",
        )
        new_status = _landing_status_after_parents(conn, task_id)
        review_event = conn.execute(
            "SELECT payload FROM task_events "
            "WHERE task_id = ? AND kind = 'review_requested' "
            "ORDER BY id DESC LIMIT 1",
            (task_id,),
        ).fetchone()
        try:
            handoff = (
                json.loads(review_event["payload"])
                if review_event and review_event["payload"]
                else {}
            )
        except (json.JSONDecodeError, TypeError):
            handoff = {}
        implementer = handoff.get("implementer")
        if not isinstance(implementer, str) or not implementer.strip():
            implementer = None
        # Handing the work back to the implementer is a role transition too, so
        # it goes through the same authority helper; a policy-locked task
        # refuses it rather than being silently re-pinned.
        role_transition_route(conn, task_id, implementer)
        if new_status in EXECUTABLE_STATUSES and not authorize_executable_transition(
            conn, task_id
        ):
            # Parked for re-approval rather than handed back into the pool.
            return False
        assignee_sql = ", assignee = ?" if implementer else ""
        params: tuple[Any, ...] = (
            (new_status, implementer, task_id)
            if implementer
            else (new_status, task_id)
        )
        cur = conn.execute(
            "UPDATE tasks SET status = ?, current_run_id = NULL, "
            "claim_lock = NULL, claim_expires = NULL, worker_pid = NULL "
            # consecutive_failures deliberately PRESERVED: review reopen is
            # not a success signal; only complete_task resets the breaker
            # counter (mirrors unblock_task, #35072).
            + assignee_sql
            + " WHERE id = ? AND status = 'review' AND task_kind = 'work'",
            params,
        )
        if cur.rowcount != 1:
            return False
        payload: dict[str, Any] = {"status": new_status}
        if implementer:
            payload["implementer"] = implementer
        _append_event(
            conn,
            task_id,
            "review_reopened",
            payload if payload != {"status": "ready"} else None,
        )
        return True


def invalidate_descendants_for_parent_reopen(
    conn: sqlite3.Connection,
    task_id: str,
    *,
    author: str,
) -> dict[str, Any]:
    """Retract every dispatchable/completed descendant of a reopened ancestor.

    THE single domain implementation of done-reopen descendant invalidation.
    When a ``done`` (or ``archived``) ancestor is reopened, every descendant
    whose state assumed the ancestor's result — ``ready``, ``review``,
    ``running`` or ``done`` — is building on a retracted premise, so it is
    demoted to ``todo`` and re-gated on the graph. The CLI deliberately has
    NO done-reopen verb on this branch (``reopen-review`` only handles the
    review-phase transition via :func:`reopen_review_task`), so every surface
    that reopens a done task (dashboard drag-drop / PATCH — single and bulk —
    via ``_set_status_direct``) must route through this function; keeping the
    implementation here means a future CLI or tool reopen verb inherits
    identical semantics for free.

    Transactionality: composes under the caller's already-open transaction
    via ``write_txn(conn, allow_nested=True)`` — the dashboard's status
    writer must commit the ancestor's status flip and the descendant
    retractions atomically (a crash between the two would leave stale done
    descendants claiming a premise that no longer holds). Called standalone
    it opens its own transaction. All SQL is inline per this file's txn
    conventions (no calls into other txn-opening helpers).

    Non-silent contract: every invalidated descendant gets
    * a ``descendant_invalidated`` event with ``{ancestor, prior_status,
      new_status}`` (plus ``resume_status``) for board/notifier surfaces,
    * the legacy ``status`` event (``reason=ancestor_reopened``) the live
      feed already renders, and
    * a ``task_comments`` row naming the reopened ancestor, so operators see
      WHY a card moved instead of watching it silently teleport.

    Live ``running`` descendants keep the termination behavior (a running
    child building on a retracted premise is wasted spend): their run is
    closed ``reclaimed`` and their worker is killed via
    :func:`_terminate_reclaimed_worker` — the same helper the reclaim paths
    use. Events/comments are written inside the transaction and the kill
    happens strictly post-commit, so the audit trail exists BEFORE the
    worker dies. When this function opened its own transaction it performs
    the terminations itself after commit; when composing under a caller's
    transaction the caller MUST drain the returned ``terminations`` list
    with ``_terminate_reclaimed_worker`` after its own commit.

    ``consecutive_failures`` is reset to 0 on every invalidated descendant:
    ancestor reopen is a deliberate operator action, so demoted work gets a
    fresh start with the breaker (a previously auto-blocked-then-completed
    descendant should not re-enter the queue one failure from the breaker).
    This is deliberately the OPPOSITE of the review-transition rule
    (:func:`reopen_review_task` / #35072 preserves the counter) because the
    autonomous review loop must not be able to launder its own failure
    streak, while an operator invalidating a subtree is an explicit reset
    signal.

    Returns ``{"invalidated": [...], "terminations": [...]}`` where each
    invalidated entry is ``{id, prior_status, new_status, resume_status}``
    and each termination is a ``(worker_pid, claim_lock)`` tuple.
    """
    caller_owns_txn = bool(getattr(conn, "in_transaction", False))
    now = int(time.time())
    invalidated: list[dict[str, Any]] = []
    terminations: list[tuple[Optional[int], Optional[str]]] = []
    with write_txn(conn, allow_nested=True):
        rows = conn.execute(
            """
            WITH RECURSIVE descendants(id) AS (
                SELECT child_id FROM task_links WHERE parent_id = ?
                UNION
                SELECT l.child_id
                FROM task_links l
                JOIN descendants d ON d.id = l.parent_id
            )
            SELECT t.id, t.status, t.current_run_id, t.worker_pid, t.claim_lock
            FROM descendants d
            JOIN tasks t ON t.id = d.id
            WHERE t.task_kind = 'work'
            ORDER BY t.id
            """,
            (task_id,),
        ).fetchall()
        for row in rows:
            previous_status = row["status"]
            if previous_status not in {"ready", "review", "running", "done"}:
                continue
            resume_status = "ready"
            run_id = None
            if previous_status == "review":
                resume_status = "review"
            elif previous_status == "running":
                resume_status = _retry_status_for_run(
                    conn, row["id"], row["current_run_id"]
                )
                terminations.append((row["worker_pid"], row["claim_lock"]))
                run_id = _end_run(
                    conn,
                    row["id"],
                    outcome="reclaimed",
                    status="todo",
                    summary=f"ancestor {task_id} reopened",
                )
            # consecutive_failures = 0: deliberate operator reset — see
            # docstring for why this diverges from reopen_review_task.
            conn.execute(
                "UPDATE tasks SET status = 'todo', completed_at = NULL, "
                "claim_lock = NULL, claim_expires = NULL, worker_pid = NULL, "
                "current_run_id = NULL, consecutive_failures = 0 WHERE id = ?",
                (row["id"],),
            )
            _append_event(
                conn,
                row["id"],
                "descendant_invalidated",
                {
                    "ancestor": task_id,
                    "prior_status": previous_status,
                    "new_status": "todo",
                    "resume_status": resume_status,
                },
                run_id=run_id,
            )
            # Legacy 'status' event kept so existing live-feed consumers
            # still see the move without learning the new event kind.
            _append_event(
                conn,
                row["id"],
                "status",
                {
                    "status": "todo",
                    "reason": "ancestor_reopened",
                    "parent": task_id,
                    "previous_status": previous_status,
                    "resume_status": resume_status,
                },
                run_id=run_id,
            )
            # Inline comment insert (not add_comment: no txn-opening helper
            # calls inside a txn per file convention).
            conn.execute(
                "INSERT INTO task_comments (task_id, author, body, created_at) "
                "VALUES (?, ?, ?, ?)",
                (
                    row["id"],
                    author,
                    (
                        f"Invalidated: ancestor {task_id} was reopened; "
                        f"retracted from '{previous_status}' to 'todo' "
                        f"(will resume via '{resume_status}')."
                    ),
                    now,
                ),
            )
            invalidated.append(
                {
                    "id": row["id"],
                    "prior_status": previous_status,
                    "new_status": "todo",
                    "resume_status": resume_status,
                }
            )
    if not caller_owns_txn:
        # Standalone call: we committed above, so the audit trail is durable
        # — safe to kill workers now. Composed calls leave this to the
        # caller (post-commit), preserving events-before-termination.
        for pid, claim_lock in terminations:
            _terminate_reclaimed_worker(pid, claim_lock)
    return {"invalidated": invalidated, "terminations": terminations}


def specify_triage_task(
    conn: sqlite3.Connection,
    task_id: str,
    *,
    title: Optional[str] = None,
    body: Optional[str] = None,
    assignee: Optional[str] = None,
    author: Optional[str] = None,
) -> bool:
    """Flesh out a triage task and promote it to ``todo``.

    Atomically updates ``title`` / ``body`` / ``assignee`` (when provided)
    and transitions ``status: triage -> todo`` in a single write txn. Returns
    False when the task is missing or not in the ``triage`` column — callers
    should surface that as "nothing to specify" rather than an error.

    ``todo`` (not ``ready``) is the correct landing column: ``recompute_ready``
    promotes parent-free / parent-done todos to ``ready`` on the next
    dispatcher tick, which keeps the normal parent-gating behaviour intact
    for specified tasks that happen to have open parents.

    ``author`` is recorded on an audit comment only when at least one of
    ``title`` / ``body`` / ``assignee`` actually changed — avoids noisy
    comment spam for status-only promotions.
    """
    if title is not None and not title.strip():
        raise ValueError("title cannot be blank")
    assignee = _canonical_assignee(assignee)
    with write_txn(conn):
        existing = conn.execute(
            "SELECT title, body, assignee FROM tasks "
            "WHERE id = ? AND status = 'triage' AND task_kind = 'work'",
            (task_id,),
        ).fetchone()
        if existing is None:
            return False
        sets: list[str] = ["status = 'todo'"]
        params: list[Any] = []
        changed_fields: list[str] = []
        if title is not None and title.strip() != (existing["title"] or ""):
            sets.append("title = ?")
            params.append(title.strip())
            changed_fields.append("title")
        if body is not None and (body or "") != (existing["body"] or ""):
            sets.append("body = ?")
            params.append(body)
            changed_fields.append("body")
        if assignee is not None and assignee != (existing["assignee"] or None):
            # Specifying a triage task can move it to a different role, which is
            # a role transition: the authority helper refuses it outright on a
            # policy-locked task.
            role_transition_route(conn, task_id, assignee)
            sets.append("assignee = ?")
            params.append(assignee)
            changed_fields.append("assignee")
        params.append(task_id)
        cur = conn.execute(
            f"UPDATE tasks SET {', '.join(sets)} "
            f"WHERE id = ? AND status = 'triage' AND task_kind = 'work'",
            tuple(params),
        )
        if cur.rowcount != 1:
            return False
        if changed_fields and author and author.strip():
            # Inline INSERT (rather than ``add_comment``) because we're
            # already inside this function's write_txn — nested BEGIN
            # IMMEDIATE would raise OperationalError. We also skip the
            # 'commented' event that ``add_comment`` emits, since the
            # 'specified' event below already records the change.
            conn.execute(
                "INSERT INTO task_comments (task_id, author, body, created_at) "
                "VALUES (?, ?, ?, ?)",
                (
                    task_id,
                    author.strip(),
                    "Specified — updated "
                    + ", ".join(changed_fields)
                    + " and promoted to todo.",
                    int(time.time()),
                ),
            )
        _append_event(
            conn,
            task_id,
            "specified",
            {"changed_fields": changed_fields} if changed_fields else None,
        )
    # Outside the write_txn above, so we don't nest BEGIN IMMEDIATE — the
    # ready-promotion pass opens its own IMMEDIATE txn. This runs the same
    # logic the dispatcher would on its next tick, so a specified task
    # with no open parents flips straight to 'ready' here instead of
    # idling in 'todo' until the next sweep.
    recompute_ready(conn)
    return True


def decompose_triage_task(
    conn: sqlite3.Connection,
    task_id: str,
    *,
    root_assignee: Optional[str],
    children: list[dict],
    author: Optional[str] = None,
    auto_promote: bool = True,
    event_metadata: Optional[dict] = None,
    receipt_owned: bool = False,
    parked: bool = False,
    park_generation: Optional[str] = None,
) -> Optional[list[str]]:
    """Fan a triage task out into child tasks and promote the root to ``todo``.

    The root task stays alive and becomes the parent of every child —
    when all children reach ``done``, the root promotes to ``ready`` and
    its assignee (typically the orchestrator profile) wakes back up to
    judge completion or spawn more work.

    ``children`` is a list of dicts, each shaped like::

        {
            "title": "...",
            "body": "...",                     # optional
            "assignee": "profile-name",        # optional, None -> default fallback
            "parents": [0, 2],                 # indices into this same children list
            "model_override": "...",           # optional immutable task route
            "provider_override": "...",        # requires model_override
            "reasoning_effort": "max",          # optional task effort
            "execution_tier": "deep",          # semantic class the route came from
            "model_policy_lock": "raphael:v1:<digest>",   # owner-approved pin
        }

    Returns the list of created child task ids (in input order) on
    success. Returns ``None`` when:
      - The root task does not exist
      - The root task is not in ``triage``
      - A cycle would result (caller built a bad graph)

    ``event_metadata`` lets a trusted caller attach operation identity to the
    root's ``decomposed`` audit event for crash-safe replay recognition. The
    canonical ``child_ids`` and ``root_assignee`` keys always win, so metadata
    cannot falsify the graph that actually committed.

    ``receipt_owned`` marks the children as owned by a committed owner receipt
    (see :func:`create_task`). ``parked`` lands them in :data:`PARKED_STATUS`
    instead of ``todo``, in the SAME insert — so an approval that is still
    writing its durable receipt can never leave claimable work behind, and no
    dispatcher tick can promote them in a gap. ``park_generation`` — required
    whenever ``parked`` is set — is the durable identity
    :func:`activate_owner_work` compare-and-swaps against, so only this
    operation's own activation releases them afterwards.

    Validation of titles/assignees happens inside the same write_txn as
    the inserts so a malformed entry aborts the whole decomposition
    cleanly (no orphan children).
    """
    if not children:
        return None
    generation = _require_park_generation(park_generation) if parked else None
    if root_assignee is not None:
        root_assignee = _canonical_assignee(root_assignee)
    if event_metadata is not None and not isinstance(event_metadata, dict):
        raise ValueError("event_metadata must be a dict")

    # Pre-validate the children list shape outside the txn. Cheap checks
    # that don't need DB access. Bad input aborts before we touch the DB.
    for idx, child in enumerate(children):
        if not isinstance(child, dict):
            raise ValueError(f"child[{idx}] is not a dict")
        title = child.get("title")
        if not isinstance(title, str) or not title.strip():
            raise ValueError(f"child[{idx}].title is required")
        parents_idx = child.get("parents") or []
        if not isinstance(parents_idx, list):
            raise ValueError(f"child[{idx}].parents must be a list")
        for p in parents_idx:
            if not isinstance(p, int) or p < 0 or p >= len(children):
                raise ValueError(
                    f"child[{idx}].parents[{p}] is not a valid index into children"
                )
            if p == idx:
                raise ValueError(f"child[{idx}] cannot list itself as a parent")
        normalize_owned_paths(child.get("owned_paths"))
        if not isinstance(child.get("integrates_parent_heads", False), bool):
            raise ValueError(f"child[{idx}].integrates_parent_heads must be a boolean")
        if child.get("integrates_parent_heads") and not normalize_owned_paths(
            child.get("owned_paths")
        ):
            raise ValueError(
                f"child[{idx}].integrates_parent_heads requires mutating owned_paths"
            )
        model_override = str(child.get("model_override") or "").strip() or None
        provider_override = str(child.get("provider_override") or "").strip() or None
        effort = normalize_reasoning_effort(child.get("reasoning_effort"))
        if provider_override and not model_override:
            raise ValueError(
                f"child[{idx}].provider_override requires model_override"
            )
        policy_lock = str(child.get("model_policy_lock") or "").strip() or None
        if policy_lock is not None:
            route_error = policy_lock_error(
                policy_lock,
                _canonical_assignee(child.get("assignee")),
                provider_override,
                model_override,
                effort,
                child.get("execution_tier"),
            )
            if route_error:
                raise ValueError(f"child[{idx}]: {route_error}")

    # Detect cycles in the sibling parent graph (Kahn's topological sort).
    # link_tasks() calls _would_cycle() for every new edge; here we check
    # the entire sibling graph before touching the DB.  A cycle silently
    # deadlocks every involved child in 'todo' because recompute_ready()
    # can never promote them.
    _in_deg = [0] * len(children)
    _adj: list[list[int]] = [[] for _ in range(len(children))]
    for _i, _c in enumerate(children):
        for _p in (_c.get("parents") or []):
            _adj[_p].append(_i)
            _in_deg[_i] += 1
    _queue = [_i for _i in range(len(children)) if _in_deg[_i] == 0]
    _seen = 0
    while _queue:
        _node = _queue.pop()
        _seen += 1
        for _nb in _adj[_node]:
            _in_deg[_nb] -= 1
            if _in_deg[_nb] == 0:
                _queue.append(_nb)
    if _seen != len(children):
        raise ValueError("cyclic dependency detected in decomposed children list")

    # We do the full decomposition in a SINGLE write_txn so it's
    # atomic: either every child is created AND the root flips to
    # ``todo``, or nothing changes. We deliberately do NOT call any
    # kb helper that opens its own write_txn (create_task, link_tasks,
    # add_comment) from inside this block — see architecture.md
    # write_txn pitfalls. Instead we inline the INSERTs and
    # _append_event calls.
    now = int(time.time())
    child_ids: list[str] = []
    with write_txn(conn):
        root_row = conn.execute(
            "SELECT id, status, tenant, workspace_kind, workspace_path, project_id "
            "FROM tasks WHERE id = ? AND task_kind = 'work'",
            (task_id,),
        ).fetchone()
        if root_row is None:
            return None
        if root_row["status"] != "triage":
            return None
        tenant = root_row["tenant"]
        project_id = root_row["project_id"]
        # Children inherit the root's workspace by default so a fan-out
        # of a code-gen task lands in the parent's project dir/worktree
        # rather than throwaway scratch tmp dirs. A child dict can still
        # override with its own 'workspace_kind' / 'workspace_path'.
        root_ws_kind = root_row["workspace_kind"] or "scratch"
        root_ws_path = root_row["workspace_path"]

        # Create children. Status is 'todo' regardless of parents — we
        # link them under the root AFTER creation so the dispatcher
        # sees a coherent state, and recompute_ready() at the end
        # promotes parent-free children to 'ready'.
        for idx, child in enumerate(children):
            new_id = _new_task_id()
            title = child["title"].strip()
            body = child.get("body")
            assignee = _canonical_assignee(child.get("assignee"))
            responsibility = normalize_responsibility(child.get("responsibility"))
            model_override = str(child.get("model_override") or "").strip() or None
            provider_override = str(child.get("provider_override") or "").strip() or None
            reasoning_effort = normalize_reasoning_effort(
                child.get("reasoning_effort")
            )
            execution_tier = (
                str(child.get("execution_tier") or "").strip().lower() or None
            )
            model_policy_lock = (
                str(child.get("model_policy_lock") or "").strip() or None
            )
            owned_paths = normalize_owned_paths(child.get("owned_paths"))
            integrates_parent_heads = child.get("integrates_parent_heads", False)
            # Per-child override wins; otherwise inherit the root's
            # workspace. A child that sets workspace_kind without a path
            # falls back to the root path only when kinds match (so a
            # child can't accidentally point a 'dir' at the root's
            # worktree path or vice versa).
            child_ws_kind = child.get("workspace_kind") or root_ws_kind
            if owned_paths and child_ws_kind != "worktree":
                raise ValueError(
                    f"child[{idx}] mutating owned_paths require workspace_kind='worktree'"
                )
            if owned_paths == [] and child_ws_kind == "dir":
                raise ValueError(
                    f"child[{idx}] read-only owned_paths require an isolated "
                    "scratch or worktree workspace"
                )
            if child.get("workspace_path"):
                child_ws_path = child.get("workspace_path")
            elif child_ws_kind == "worktree":
                # A canonical Project task lives at
                # ``<project-repo>/.worktrees/<root-id>``. Preserve that
                # repository identity while still giving every sibling its
                # own checkout. Falling back to ``None`` is safe for legacy
                # roots whose path does not prove this convention; dispatch
                # then uses the board anchor as before.
                root_path = Path(root_ws_path) if root_ws_path else None
                if (
                    root_path is not None
                    and root_path.name == task_id
                    and root_path.parent.name == ".worktrees"
                ):
                    child_ws_path = str(root_path.parent / new_id)
                else:
                    child_ws_path = None
            elif child_ws_kind == root_ws_kind:
                child_ws_path = root_ws_path
            else:
                child_ws_path = None
            conn.execute(
                "INSERT INTO tasks "
                "(id, title, body, assignee, responsibility, status, workspace_kind, "
                " workspace_path, tenant, project_id, owned_paths, "
                " integrates_parent_heads, model_override, provider_override, "
                " reasoning_effort, execution_tier, model_policy_lock, "
                " owner_receipt_bound, park_generation, created_at, created_by) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    new_id,
                    title,
                    body if isinstance(body, str) else None,
                    assignee,
                    responsibility,
                    PARKED_STATUS if parked else "todo",
                    child_ws_kind,
                    child_ws_path,
                    tenant,
                    project_id,
                    json.dumps(owned_paths) if owned_paths is not None else None,
                    1 if integrates_parent_heads else 0,
                    model_override,
                    provider_override,
                    reasoning_effort,
                    execution_tier,
                    model_policy_lock,
                    1 if receipt_owned else 0,
                    generation,
                    now,
                    (author or "decomposer"),
                ),
            )
            _append_event(
                conn, new_id, "created",
                {
                    "by": author or "decomposer",
                    "from_decompose_of": task_id,
                    "owned_paths": owned_paths,
                    "integrates_parent_heads": integrates_parent_heads or None,
                    "execution_tier": execution_tier,
                    "model_route_pinned": bool(model_policy_lock),
                    "parked_for_activation": True if parked else None,
                },
            )
            _inherit_notify_subs(conn, new_id, (task_id,), created_at=now)
            child_ids.append(new_id)

        # Link children to their sibling parents (within the decomposed graph).
        for idx, child in enumerate(children):
            for p_idx in child.get("parents") or []:
                parent_id = child_ids[p_idx]
                child_id = child_ids[idx]
                conn.execute(
                    "INSERT OR IGNORE INTO task_links (parent_id, child_id) "
                    "VALUES (?, ?)",
                    (parent_id, child_id),
                )
                _append_event(
                    conn, child_id, "linked",
                    {"parent": parent_id, "child": child_id},
                )

        # Link the ROOT task as a child of every leaf child — i.e. the
        # root waits for the whole graph. Simpler than computing leaves:
        # link root under every child. Cycle-free because the root is
        # only ever a child here, never a parent of children.
        for cid in child_ids:
            conn.execute(
                "INSERT OR IGNORE INTO task_links (parent_id, child_id) "
                "VALUES (?, ?)",
                (cid, task_id),
            )

        # Flip the root: triage -> todo, set assignee to the orchestrator.
        # Moving the root to a different role is a role transition, so it goes
        # through the authority helper; a policy-locked root refuses it.
        sets = ["status = 'todo'"]
        params: list[Any] = []
        if root_assignee is not None:
            role_transition_route(conn, task_id, root_assignee)
            sets.append("assignee = ?")
            params.append(root_assignee)
        params.append(task_id)
        conn.execute(
            f"UPDATE tasks SET {', '.join(sets)} WHERE id = ?",
            tuple(params),
        )

        # Audit comment + event on the root so the timeline shows the fan-out.
        if author and author.strip():
            conn.execute(
                "INSERT INTO task_comments (task_id, author, body, created_at) "
                "VALUES (?, ?, ?, ?)",
                (
                    task_id,
                    author.strip(),
                    "Decomposed into "
                    + ", ".join(child_ids)
                    + ". Root will wake when all children complete.",
                    now,
                ),
            )
        event_payload = dict(event_metadata or {})
        event_payload.update({
            "child_ids": child_ids,
            "root_assignee": root_assignee,
        })
        _append_event(conn, task_id, "decomposed", event_payload)

    # Outside the write_txn: promote parent-free children to 'ready'
    # so the dispatcher picks them up on its next tick. Same pattern
    # specify_triage_task uses.  When auto_promote is False children
    # stay in 'todo' until the user manually promotes them — useful
    # for manual-review-first workflows.
    if auto_promote:
        recompute_ready(conn)
    return child_ids


def task_event_revision(conn: sqlite3.Connection, task_id: str) -> int:
    """Monotonic per-task revision: the highest ``task_events.id`` recorded
    for ``task_id`` (0 if the task has no events yet — cannot happen for a
    task that exists, since ``create_task`` always appends a ``"created"``
    event in the same transaction).  Callers needing optimistic
    concurrency control (compare-and-swap on status AND "nothing else about
    this task changed since I last read it") pass this back as
    ``expected_revision``; see :func:`cas_transition_task`.
    """
    row = conn.execute(
        "SELECT COALESCE(MAX(id), 0) AS rev FROM task_events WHERE task_id = ?",
        (task_id,),
    ).fetchone()
    return int(row["rev"]) if row else 0


def cas_transition_task(
    conn: sqlite3.Connection,
    task_id: str,
    *,
    expected_status: str,
    expected_revision: int,
    to_status: str,
    event_kind: str = "cas_transition",
    event_payload: Optional[dict] = None,
    park_newly_enabled_dependents: bool = False,
    park_generation: Optional[str] = None,
) -> dict:
    """Generic compare-and-swap status move, guarded by BOTH status and
    event-revision, in the existing write transaction.

    Unlike every other transition helper in this module (``claim_task``,
    ``complete_task``, ``archive_task``, ...) — which each encode one fixed,
    known-safe status pair — this is the ONE place a caller may request an
    arbitrary ``expected_status -> to_status`` move. It exists so a generic
    caller (the owner-workspace kernel) never has to reach past this module's
    public surface into ``_append_event``/``_end_run`` to build its own
    ad-hoc status-writing path — that would be exactly the kind of private
    direct-status bypass this module's CAS discipline (see ``write_txn``'s
    docstring) exists to prevent.

    Returns a snapshot dict: ``{"moved": bool, "status": str, "revision": int}``.
    ``moved`` is False when the current ``(status, revision)`` didn't match
    the expectation — a conflict — in which case ``status``/``revision``
    reflect the CURRENT row so the caller can hand back an authoritative
    snapshot instead of guessing. Never raises for a plain conflict; raises
    ``ValueError`` only for an unknown task id.

    ``park_newly_enabled_dependents`` makes a move INTO a terminal column
    carry its own dependency release: every dependent this transition newly
    unblocks is moved to :data:`PARKED_STATUS` in the SAME transaction and
    reported back — and recorded in the event payload — as
    ``parked_dependents``. An owner caller whose authority is only durable
    once its receipt is written needs exactly that, because the instant the
    parent commits as terminal a live dispatcher tick could otherwise claim a
    dependent for an operation that may still fail to finalize.
    ``park_generation`` — required whenever parking is requested — is the
    durable identity :func:`activate_owner_work` compare-and-swaps against, so
    only this operation's own activation can release what it parked.
    """
    with write_txn(conn):
        return _cas_transition_task_in_txn(
            conn,
            task_id,
            expected_status=expected_status,
            expected_revision=expected_revision,
            to_status=to_status,
            event_kind=event_kind,
            event_payload=event_payload,
            park_newly_enabled_dependents=park_newly_enabled_dependents,
            park_generation=park_generation,
        )


def _cas_transition_task_in_txn(
    conn: sqlite3.Connection,
    task_id: str,
    *,
    expected_status: str,
    expected_revision: int,
    to_status: str,
    event_kind: str,
    event_payload: Optional[dict],
    park_newly_enabled_dependents: bool = False,
    park_generation: Optional[str] = None,
) -> dict:
    """CAS helper for callers already holding the Kanban write transaction."""
    row = conn.execute(
        "SELECT status FROM tasks WHERE id = ?", (task_id,),
    ).fetchone()
    if row is None:
        raise ValueError(f"unknown task {task_id}")
    current_status = row["status"]
    current_revision = task_event_revision(conn, task_id)
    if current_status != expected_status or current_revision != expected_revision:
        return {
            "moved": False,
            "status": current_status,
            "revision": current_revision,
        }
    if to_status in EXECUTABLE_STATUSES and not authorize_executable_transition(
        conn, task_id, park=False
    ):
        # Generic CAS callers report an authoritative snapshot and may abort
        # the whole transaction, so this refuses without parking (which would
        # be rolled back) — the row keeps its current, non-executable state.
        return {
            "moved": False,
            "status": current_status,
            "revision": current_revision,
        }
    # Any move THROUGH this generic path — including an owner's deliberate
    # postpone into ``scheduled`` — drops whatever parking generation the row
    # carried: it is no longer sitting where some earlier receipt left it, so
    # that receipt's activation must not be able to pick it up again.
    if to_status == "archived":
        cur = conn.execute(
            "UPDATE tasks SET status = ?, park_generation = NULL, "
            "claim_lock = NULL, claim_expires = NULL, "
            "worker_pid = NULL WHERE id = ? AND status = ?",
            (to_status, task_id, expected_status),
        )
    else:
        cur = conn.execute(
            "UPDATE tasks SET status = ?, park_generation = NULL "
            "WHERE id = ? AND status = ?",
            (to_status, task_id, expected_status),
        )
    if cur.rowcount != 1:
        return {
            "moved": False,
            "status": conn.execute(
                "SELECT status FROM tasks WHERE id = ?", (task_id,),
            ).fetchone()["status"],
            "revision": task_event_revision(conn, task_id),
        }
    run_id = None
    if to_status == "archived":
        run_id = _end_run(
            conn, task_id,
            outcome="reclaimed", status="reclaimed",
            summary=(event_payload or {}).get("summary")
            or "task archived via cas_transition_task",
        )
    parked_dependents: list[list[str]] = []
    if park_newly_enabled_dependents and to_status in ("done", "archived"):
        parked_dependents = _park_newly_enabled_dependents(
            conn,
            parent_ids_made_terminal=[task_id],
            already_parked=set(),
            generation=park_generation,
        )
        # The exact parked set belongs in the transition's OWN event, so a
        # replay that recognizes this event as its own can reconstruct the
        # release without re-deriving it from a board that has since moved.
        event_payload = {
            **(event_payload or {}), "parked_dependents": parked_dependents,
        }
    _append_event(conn, task_id, event_kind, event_payload, run_id=run_id)
    return {
        "moved": True,
        "status": to_status,
        "revision": task_event_revision(conn, task_id),
        "parked_dependents": parked_dependents,
    }


def _owner_plan_task_key(
    *, actor: str, profile: str, idempotency_key: str, change_index: int,
    replacement_index: int,
) -> str:
    raw = (
        f"{actor}\0{profile}\0{idempotency_key}\0"
        f"{change_index}\0{replacement_index}"
    ).encode("utf-8")
    return "owplan_" + hashlib.sha256(raw).hexdigest()[:40]


def apply_owner_project_plan(
    conn: sqlite3.Connection,
    *,
    project_id: str,
    anchor_task_id: str,
    changes: list[dict],
    actor: str,
    profile: str,
    idempotency_key: str,
    request_digest: str,
    trigger: str,
    plan_summary: str,
    current_milestone: str,
    later_milestones: list[str],
    board: Optional[str] = None,
    parked: bool = True,
) -> dict:
    """Atomically apply one bounded, already-normalized Project Steward plan.

    This composition primitive is not a model tool. It verifies every
    optimistic snapshot before the first write. Any drift returns a conflict
    with zero changes. Replacements preserve source rows and old links, add
    the new blocking edges, and archive superseded rows instead of deleting
    history.

    ``parked`` (the default) lands every task this plan creates — and every
    task it reactivates — in :data:`PARKED_STATUS` inside the same
    transaction, and reports them under ``parked_task_ids``. The dependents an
    archive or a merge newly unblocks are parked in that same transaction and
    reported under ``parked_dependents`` as ``[task_id, status_to_restore]``
    pairs. The owner kernel releases both with :func:`activate_owner_work`
    only after its terminal receipt is durable, so a plan whose later durable
    writes fail can never have left claimable work behind. Deliberately
    postponed tasks also sit in ``scheduled`` and are NOT reported, and every
    row this plan parks carries this plan's own parking generation, so
    activation can only ever release what this exact receipt parked.
    """
    generation = park_generation(
        actor=actor, profile=profile, idempotency_key=idempotency_key,
    )
    expected: dict[str, tuple[str, int]] = {}
    mutation_targets: set[str] = set()

    def remember(ref: dict, *, mutating: bool) -> None:
        task_id = ref["task_id"]
        snapshot = (ref["expected_status"], int(ref["expected_revision"]))
        prior = expected.get(task_id)
        if prior is not None and prior != snapshot:
            raise ValueError(f"conflicting snapshots for task {task_id}")
        expected[task_id] = snapshot
        if mutating:
            if task_id in mutation_targets:
                raise ValueError(f"task {task_id} is changed more than once")
            mutation_targets.add(task_id)

    for change in changes:
        action = change["action"]
        if action == "add":
            for ref in change["existing_parents"]:
                remember(ref, mutating=False)
        elif action == "merge":
            for ref in change["targets"]:
                remember(ref, mutating=True)
        else:
            remember(change["target"], mutating=True)

    with write_txn(conn):
        # The anchor is the Project's non-executable control row, never a work
        # task: it carries no assignee and no approved route, so admitting a
        # 'work' anchor here would let a plan hang owner work off a task that
        # could itself be dispatched.
        anchor = conn.execute(
            "SELECT project_id, task_kind FROM tasks WHERE id = ?",
            (anchor_task_id,),
        ).fetchone()
        if (
            anchor is None
            or anchor["task_kind"] != "control"
            or anchor["project_id"] != project_id
        ):
            raise ValueError("project plan anchor is outside the bound Project")

        conflicts: list[dict] = []
        for task_id, (wanted_status, wanted_revision) in expected.items():
            row = conn.execute(
                "SELECT status, project_id, task_kind FROM tasks WHERE id = ?",
                (task_id,),
            ).fetchone()
            # A plan may hang new work under the Project's control anchor, so
            # a referenced task may be either kind. Mutating one is separately
            # rejected below: only 'work' rows are ever changed.
            if (
                row is None
                or row["task_kind"] not in ("work", "control")
                or row["project_id"] != project_id
            ):
                conflicts.append({"task_id": task_id, "status": None, "revision": None})
                continue
            if task_id in mutation_targets and row["task_kind"] != "work":
                conflicts.append({"task_id": task_id, "status": None, "revision": None})
                continue
            revision = task_event_revision(conn, task_id)
            if row["status"] != wanted_status or revision != wanted_revision:
                conflicts.append(
                    {"task_id": task_id, "status": row["status"], "revision": revision}
                )
        if conflicts:
            return {"applied": False, "conflicts": conflicts}

        created_task_ids: list[str] = []
        archived_task_ids: list[str] = []
        affected_task_ids: set[str] = set()
        parked_task_ids: list[str] = []
        add_output_by_change: dict[int, str] = {}
        internal_targets: dict[str, tuple[str, int]] = {}

        def expected_target(ref: dict) -> tuple[str, int]:
            """Include only state changes made earlier by this transaction."""
            return internal_targets.get(
                ref["task_id"], (ref["expected_status"], ref["expected_revision"]),
            )

        def record_internal_link(task_id: str) -> None:
            if task_id in mutation_targets:
                row = conn.execute(
                    "SELECT status FROM tasks WHERE id = ?", (task_id,),
                ).fetchone()
                internal_targets[task_id] = (
                    row["status"], task_event_revision(conn, task_id),
                )

        def create_planned_task(
            spec: dict,
            *,
            parent_task_ids: list[str],
            change_index: int,
            replacement_index: int,
        ) -> str:
            task_id = create_task(
                conn,
                title=spec["title"],
                body=spec["body"],
                assignee=spec["assignee"],
                responsibility=spec["responsibility"],
                model_override=spec.get("model_override"),
                provider_override=spec.get("provider_override"),
                reasoning_effort=spec.get("reasoning_effort"),
                execution_tier=spec.get("execution_tier"),
                model_policy_lock=spec.get("model_policy_lock"),
                # Absent means legacy fail-closed whole-repository ownership,
                # exactly as before this key existed; a present value is the
                # owner-approved explicit write boundary and forces the
                # project-anchored worktree inside create_task.
                owned_paths=spec.get("owned_paths"),
                created_by=actor,
                parents=parent_task_ids,
                idempotency_key=_owner_plan_task_key(
                    actor=actor,
                    profile=profile,
                    idempotency_key=idempotency_key,
                    change_index=change_index,
                    replacement_index=replacement_index,
                ),
                board=board,
                project_id=project_id,
                receipt_owned=True,
            )
            created_task_ids.append(task_id)
            affected_task_ids.add(task_id)
            if parked:
                # Same transaction as the insert: there is no instant at which
                # a freshly approved task exists in a claimable column.
                conn.execute(
                    "UPDATE tasks SET status = ?, park_generation = ? "
                    "WHERE id = ? AND task_kind = 'work' AND status != ?",
                    (PARKED_STATUS, generation, task_id, PARKED_STATUS),
                )
                parked_task_ids.append(task_id)
            return task_id

        for change_index, change in enumerate(changes):
            action = change["action"]
            reason = change["reason"]
            event_base = {
                "actor": actor,
                "profile": profile,
                "idempotency_key": idempotency_key,
                "request_digest": request_digest,
                "action": action,
                "reason": reason,
            }

            if action == "add":
                parent_task_ids = [
                    ref["task_id"] for ref in change["existing_parents"]
                ]
                for parent_change_index in change["new_parents"]:
                    try:
                        parent_task_ids.append(add_output_by_change[parent_change_index])
                    except KeyError as exc:
                        raise ValueError(
                            "new_parents may reference only an earlier add change"
                        ) from exc
                task_id = create_planned_task(
                    change,
                    parent_task_ids=parent_task_ids,
                    change_index=change_index,
                    replacement_index=0,
                )
                add_output_by_change[change_index] = task_id
                continue

            if action == "split":
                source_id = change["target"]["task_id"]
                target_status, target_revision = expected_target(change["target"])
                source_parents = parent_ids(conn, source_id)
                source_children = child_ids(conn, source_id)
                replacement_ids: list[str] = []
                referenced_replacements: set[int] = set()
                for replacement_index, replacement in enumerate(change["replacements"]):
                    internal_parents = replacement["parents"]
                    referenced_replacements.update(internal_parents)
                    replacement_parents = (
                        [replacement_ids[index] for index in internal_parents]
                        if internal_parents
                        else source_parents
                    )
                    replacement_ids.append(
                        create_planned_task(
                            replacement,
                            parent_task_ids=replacement_parents,
                            change_index=change_index,
                            replacement_index=replacement_index,
                        )
                    )
                leaf_ids = [
                    task_id
                    for index, task_id in enumerate(replacement_ids)
                    if index not in referenced_replacements
                ]
                for child_id in source_children:
                    for leaf_id in leaf_ids:
                        if _link_tasks_in_txn(conn, leaf_id, child_id):
                            affected_task_ids.add(child_id)
                            record_internal_link(child_id)
                snapshot = _cas_transition_task_in_txn(
                    conn,
                    source_id,
                    expected_status=target_status,
                    expected_revision=target_revision,
                    to_status="archived",
                    event_kind="owner_project_plan_change",
                    event_payload={**event_base, "replacement_task_ids": replacement_ids},
                )
                if not snapshot["moved"]:
                    raise RuntimeError("preflighted split target changed inside transaction")
                affected_task_ids.add(source_id)
                archived_task_ids.append(source_id)
                continue

            if action == "replace":
                source_id = change["target"]["task_id"]
                target_status, target_revision = expected_target(change["target"])
                replacement_id = create_planned_task(
                    change["replacement"],
                    parent_task_ids=parent_ids(conn, source_id),
                    change_index=change_index,
                    replacement_index=0,
                )
                for child_id in child_ids(conn, source_id):
                    if _link_tasks_in_txn(conn, replacement_id, child_id):
                        affected_task_ids.add(child_id)
                        record_internal_link(child_id)
                snapshot = _cas_transition_task_in_txn(
                    conn,
                    source_id,
                    expected_status=target_status,
                    expected_revision=target_revision,
                    to_status="archived",
                    event_kind="owner_project_plan_change",
                    event_payload={**event_base, "replacement_task_id": replacement_id},
                )
                if not snapshot["moved"]:
                    raise RuntimeError(
                        "preflighted replacement target changed inside transaction"
                    )
                affected_task_ids.add(source_id)
                archived_task_ids.append(source_id)
                continue

            if action == "merge":
                source_ids = [ref["task_id"] for ref in change["targets"]]
                source_set = set(source_ids)
                merged_parents = sorted(
                    {
                        parent_id
                        for source_id in source_ids
                        for parent_id in parent_ids(conn, source_id)
                        if parent_id not in source_set
                    }
                )
                merged_children = sorted(
                    {
                        child_id
                        for source_id in source_ids
                        for child_id in child_ids(conn, source_id)
                        if child_id not in source_set
                    }
                )
                replacement_id = create_planned_task(
                    change["replacement"],
                    parent_task_ids=merged_parents,
                    change_index=change_index,
                    replacement_index=0,
                )
                for child_id in merged_children:
                    if _link_tasks_in_txn(conn, replacement_id, child_id):
                        affected_task_ids.add(child_id)
                        record_internal_link(child_id)
                for ref in change["targets"]:
                    target_status, target_revision = expected_target(ref)
                    snapshot = _cas_transition_task_in_txn(
                        conn,
                        ref["task_id"],
                        expected_status=target_status,
                        expected_revision=target_revision,
                        to_status="archived",
                        event_kind="owner_project_plan_change",
                        event_payload={**event_base, "replacement_task_id": replacement_id},
                    )
                    if not snapshot["moved"]:
                        raise RuntimeError("preflighted merge target changed inside transaction")
                    affected_task_ids.add(ref["task_id"])
                    archived_task_ids.append(ref["task_id"])
                continue

            ref = change["target"]
            target_status, target_revision = expected_target(ref)
            if action == "move" and change["to_status"] == "ready":
                if not _parents_satisfied(conn, ref["task_id"]):
                    raise ValueError(
                        "cannot move a task to ready while a parent is unfinished"
                    )
                # Refuse the WHOLE plan rather than park inside a transaction
                # that is about to be rolled back: nothing changes, and the
                # task keeps its current non-executable state.
                if not authorize_executable_transition(
                    conn, ref["task_id"], park=False
                ):
                    raise ValueError(
                        "cannot move a task to ready while its approved model "
                        "route cannot be confirmed"
                    )
            to_status = (
                change["to_status"]
                if action == "move"
                else "scheduled"
                if action == "postpone"
                else "archived"
            )
            reactivating = action == "move" and to_status == "ready"
            if reactivating and parked:
                to_status = PARKED_STATUS
                parked_task_ids.append(ref["task_id"])
            snapshot = _cas_transition_task_in_txn(
                conn,
                ref["task_id"],
                expected_status=target_status,
                expected_revision=target_revision,
                to_status=to_status,
                event_kind="owner_project_plan_change",
                event_payload=event_base,
            )
            if not snapshot["moved"]:
                raise RuntimeError(
                    "preflighted Project Steward target changed inside transaction"
                )
            if reactivating and parked:
                # The generic CAS above cleared any stale generation; stamp
                # this plan's own so only its activation releases the row.
                conn.execute(
                    "UPDATE tasks SET park_generation = ? "
                    "WHERE id = ? AND task_kind = 'work' AND status = ?",
                    (generation, ref["task_id"], PARKED_STATUS),
                )
            if reactivating:
                # A needs-input worker receives task comments, not raw plan
                # events. Persist the approved owner answer as an idempotent
                # comment so the resumed specialist can actually use it.
                add_comment(
                    conn,
                    ref["task_id"],
                    actor,
                    reason,
                    operation_key=(
                        f"owner-plan:{idempotency_key}:change:{change_index}:resume"
                    ),
                )
            affected_task_ids.add(ref["task_id"])
            if to_status == "archived":
                archived_task_ids.append(ref["task_id"])

        # Archiving a parent is what releases its dependents, so those are
        # parked here too — in the same transaction, never by a later readiness
        # recompute that could land before this plan's receipt is durable.
        parked_dependents = (
            _park_newly_enabled_dependents(
                conn,
                parent_ids_made_terminal=archived_task_ids,
                already_parked=set(parked_task_ids),
                generation=generation,
            )
            if parked and archived_task_ids
            else []
        )
        affected_task_ids.update(task_id for task_id, _ in parked_dependents)

        # Deliberately NO project-wide executable total is enforced here. A
        # large Project may hold an evidence-backed backlog and later
        # milestones, and refusing an approved plan because the Project already
        # holds enough planned work failed the owner's approval for an internal
        # slot count they can neither see nor manage. What must stay bounded is
        # SIMULTANEOUS work, and every bound on that already lives where the
        # work actually becomes runnable: one plan's own size (the owner
        # kernel's ``_MAX_GRAPH_TASKS``), the dependency edges written above (an
        # unfinished parent keeps its child out of ``ready``), the parking this
        # function performs until the receipt is durable, and the dispatcher's
        # claim budget (``max_in_progress`` / ``max_in_progress_per_profile`` /
        # ``max_spawn`` in :func:`dispatch_once`).
        result = {
            "applied": True,
            "change_count": len(changes),
            "created_task_ids": created_task_ids,
            "affected_task_ids": sorted(affected_task_ids),
            "parked_task_ids": sorted(set(parked_task_ids)),
            "parked_dependents": parked_dependents,
            # The receipt carries the generation so a replay activates exactly
            # what this plan parked and nothing the owner parked since.
            "park_generation": generation,
        }
        _append_event(
            conn,
            anchor_task_id,
            "owner_project_plan_applied",
            {
                "actor": actor,
                "profile": profile,
                "idempotency_key": idempotency_key,
                "request_digest": request_digest,
                "trigger": trigger,
                "plan_summary": plan_summary,
                "current_milestone": current_milestone,
                "later_milestones": later_milestones,
                "result": result,
            },
        )

    # No readiness recompute here when the plan parked its work: everything
    # this plan enabled — its own new/reactivated tasks AND the dependents the
    # archives released — is sitting in ``scheduled``, where neither this call
    # nor a concurrent dispatcher tick can promote it. ``activate_owner_work``
    # runs the recompute once the terminal receipt is durable.
    if not parked:
        recompute_ready(conn)
    for task_id in archived_task_ids:
        _cleanup_workspace(conn, task_id)
    return result


def archive_task(conn: sqlite3.Connection, task_id: str) -> bool:
    with write_txn(conn):
        cur = conn.execute(
            "UPDATE tasks SET status = 'archived', "
            "    claim_lock = NULL, claim_expires = NULL, worker_pid = NULL "
            "WHERE id = ? AND status != 'archived' AND task_kind = 'work'",
            (task_id,),
        )
        if cur.rowcount != 1:
            return False
        # If archive happened while a run was still in flight (e.g. user
        # archived a running task from the dashboard), close that run with
        # outcome='reclaimed' so attempt history isn't orphaned.
        run_id = _end_run(
            conn, task_id,
            outcome="reclaimed", status="reclaimed",
            summary="task archived with run still active",
        )
        _append_event(conn, task_id, "archived", None, run_id=run_id)
    # ``archived`` parents no longer block children, same as ``done``.
    # Promote newly-unblocked dependents immediately instead of waiting
    # for a later dispatcher tick.
    recompute_ready(conn)
    # Reap the workspace on archive too — tasks archived without ever
    # completing previously kept their scratch dir / worktree forever.
    _cleanup_workspace(conn, task_id)
    return True


def delete_archived_task(conn: sqlite3.Connection, task_id: str) -> bool:
    """Permanently remove an already-archived task and its related rows.

    Safety guard: only archived tasks can be deleted. Active / blocked / done
    tasks must be explicitly archived first so accidental data loss requires a
    second deliberate action.
    """
    with write_txn(conn):
        row = conn.execute(
            "SELECT status FROM tasks WHERE id = ? AND task_kind = 'work'",
            (task_id,),
        ).fetchone()
        if not row or row["status"] != "archived":
            return False
        conn.execute(
            "DELETE FROM task_links WHERE parent_id = ? OR child_id = ?",
            (task_id, task_id),
        )
        conn.execute("DELETE FROM task_comments WHERE task_id = ?", (task_id,))
        conn.execute("DELETE FROM task_events WHERE task_id = ?", (task_id,))
        conn.execute("DELETE FROM task_runs WHERE task_id = ?", (task_id,))
        conn.execute("DELETE FROM kanban_notify_subs WHERE task_id = ?", (task_id,))
        cur = conn.execute(
            "DELETE FROM tasks WHERE id = ? AND task_kind = 'work'", (task_id,)
        )
        return cur.rowcount == 1


def delete_task(conn: sqlite3.Connection, task_id: str) -> bool:
    """Hard-delete a task and cascade to all related rows.

    Because the schema does not use ``ON DELETE CASCADE`` foreign keys,
    we explicitly delete from child tables first, then the task row.
    This keeps the operation atomic (single ``write_txn``).

    Returns ``True`` if the task existed and was deleted, ``False``
    if the task was not found.
    """
    with write_txn(conn):
        cur = conn.execute(
            "DELETE FROM tasks WHERE id = ? AND task_kind = 'work'", (task_id,)
        )
        if cur.rowcount != 1:
            return False
        conn.execute("DELETE FROM task_links WHERE parent_id = ? OR child_id = ?", (task_id, task_id))
        conn.execute("DELETE FROM task_comments WHERE task_id = ?", (task_id,))
        conn.execute("DELETE FROM task_events WHERE task_id = ?", (task_id,))
        conn.execute("DELETE FROM task_runs WHERE task_id = ?", (task_id,))
        conn.execute("DELETE FROM kanban_notify_subs WHERE task_id = ?", (task_id,))
    recompute_ready(conn)
    return True


# ---------------------------------------------------------------------------
# Workspace resolution
# ---------------------------------------------------------------------------

def _git_toplevel(path: Path) -> Optional[Path]:
    """Return the git toplevel containing ``path``, or ``None`` if not in a repo."""
    try:
        result = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "--show-toplevel"],
            capture_output=True,
            text=True, encoding='utf-8', errors='replace',
            timeout=30,
            check=False,
        )
    except Exception:
        return None
    if result.returncode != 0:
        return None
    out = (result.stdout or "").strip()
    if not out:
        return None
    try:
        return Path(out).expanduser().resolve()
    except Exception:
        return Path(out).expanduser()


def _git_branch_exists(repo_root: Path, branch_name: str) -> bool:
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_root), "show-ref", "--verify", f"refs/heads/{branch_name}"],
            capture_output=True,
            text=True, encoding='utf-8', errors='replace',
            timeout=30,
            check=False,
        )
    except Exception:
        return False
    return result.returncode == 0


def _git_common_dir(path: Path) -> Optional[Path]:
    try:
        result = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "--path-format=absolute", "--git-common-dir"],
            capture_output=True,
            text=True, encoding='utf-8', errors='replace',
            timeout=30,
            check=False,
        )
    except Exception:
        return None
    if result.returncode != 0:
        return None
    out = (result.stdout or "").strip()
    if not out:
        return None
    return Path(out).expanduser().resolve(strict=False)


def _git_dir(path: Path) -> Optional[Path]:
    try:
        result = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "--path-format=absolute", "--git-dir"],
            capture_output=True,
            text=True, encoding='utf-8', errors='replace',
            timeout=30,
            check=False,
        )
    except Exception:
        return None
    if result.returncode != 0:
        return None
    out = (result.stdout or "").strip()
    if not out:
        return None
    return Path(out).expanduser().resolve(strict=False)


def _git_current_branch(path: Path) -> Optional[str]:
    try:
        result = subprocess.run(
            ["git", "-C", str(path), "branch", "--show-current"],
            capture_output=True,
            text=True, encoding='utf-8', errors='replace',
            timeout=30,
            check=False,
        )
    except Exception:
        return None
    if result.returncode != 0:
        return None
    branch = (result.stdout or "").strip()
    return branch or None


def _is_linked_worktree_checkout(path: Path) -> bool:
    git_dir = _git_dir(path)
    common_dir = _git_common_dir(path)
    if git_dir is None or common_dir is None:
        return False
    return git_dir != common_dir


def _nearest_existing_path(path: Path) -> Path:
    current = path
    while not current.exists() and current != current.parent:
        current = current.parent
    return current


def _repo_root_for_worktree_target(path: Path) -> Optional[Path]:
    current = _nearest_existing_path(path).resolve(strict=False)
    while True:
        repo_root = _git_toplevel(current)
        if repo_root is not None:
            return repo_root
        if current == current.parent:
            return None
        current = current.parent


def _ensure_git_worktree(repo_root: Path, target: Path, branch_name: str) -> None:
    """Materialize ``target`` as a linked git worktree under ``repo_root``."""
    target = target.expanduser()
    repo_common = _git_common_dir(repo_root)
    if target.exists() and repo_common is not None:
        target_common = _git_common_dir(target)
        if target_common == repo_common:
            return
    target.parent.mkdir(parents=True, exist_ok=True)
    if _git_branch_exists(repo_root, branch_name):
        cmd = ["git", "-C", str(repo_root), "worktree", "add", str(target), branch_name]
    else:
        cmd = [
            "git", "-C", str(repo_root), "worktree", "add", "-b", branch_name,
            str(target), "HEAD",
        ]
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True, encoding='utf-8', errors='replace',
        timeout=60,
        check=False,
    )
    if result.returncode != 0:
        stderr = (result.stderr or result.stdout or "").strip()
        raise RuntimeError(
            f"git worktree add failed for {target} on branch {branch_name}: {stderr}"
        )


def _resolve_worktree_workspace(
    task: Task, *, board: Optional[str] = None
) -> tuple[Path, str]:
    """Resolve + materialize a linked git worktree for ``task``.

    When ``task.workspace_path`` is unset, the anchor is the board's
    ``default_workdir`` (a persistent project checkout). This keeps every
    worktree task under a meaningful, board-owned repo — ``<repo>/.worktrees/
    <task-id>`` — instead of silently landing under the dispatcher's current
    working directory (which is whatever directory the gateway happened to be
    launched from, e.g. the Hermes checkout). If no anchor is configured
    anywhere, we fail loudly rather than guess.
    """
    branch_name = (task.branch_name or "").strip() or f"wt/{task.id}"
    if not task.workspace_path:
        # Anchor on the board's configured default_workdir, not Path.cwd().
        # The dispatcher's CWD is incidental (gateway launch dir) and using it
        # scatters worktrees under whatever repo the gateway started in.
        board_slug = board if board else get_current_board()
        board_default = (read_board_metadata(board_slug).get("default_workdir") or "").strip()
        if not board_default:
            raise ValueError(
                f"task {task.id} has workspace_kind=worktree but no workspace_path, "
                f"and board {board_slug!r} has no default_workdir set. Set a board "
                "default workdir (a git repo) or create the task with "
                "--workspace worktree:<absolute-repo-path>."
            )
        anchor = Path(board_default).expanduser()
        if not anchor.is_absolute():
            raise ValueError(
                f"board {board_slug!r} default_workdir {board_default!r} is not "
                "absolute; use an absolute path to a git repo"
            )
        repo_root = _git_toplevel(anchor)
        if repo_root is None:
            raise ValueError(
                f"task {task.id} has workspace_kind=worktree but board "
                f"{board_slug!r} default_workdir {board_default!r} is not inside a git repo"
            )
        target = repo_root / ".worktrees" / task.id
        _ensure_git_worktree(repo_root, target, branch_name)
        return target, branch_name

    requested = Path(task.workspace_path).expanduser()
    if not requested.is_absolute():
        raise ValueError(
            f"task {task.id} has non-absolute worktree path "
            f"{task.workspace_path!r}; use an absolute path"
        )
    requested_resolved = requested.resolve(strict=False)

    if requested.exists() and _is_linked_worktree_checkout(requested):
        actual_branch = _git_current_branch(requested)
        if actual_branch == branch_name:
            return requested_resolved, actual_branch
        # The requested path is an existing checkout of a DIFFERENT
        # task's branch. Decompose children inherit the root's
        # workspace_path verbatim, so siblings all point here; reusing
        # the checkout as-is would run this task on the other task's
        # branch — silent cross-task provenance corruption, and unsafe
        # when siblings run concurrently. Fall back to a fresh worktree
        # of our own under the same repo.
        fallback_root = _repo_root_for_worktree_target(requested.parent)
        if fallback_root is not None:
            fallback = fallback_root / ".worktrees" / task.id
            if fallback.resolve(strict=False) != requested_resolved:
                _ensure_git_worktree(fallback_root, fallback, branch_name)
                return fallback.resolve(strict=False), branch_name
        # No repo to anchor a fallback on (or the occupied path IS this
        # task's own canonical worktree): keep the legacy reuse rather
        # than failing dispatch.
        return requested_resolved, actual_branch or branch_name

    repo_root = _git_toplevel(requested)
    if repo_root is not None and requested_resolved == repo_root:
        target = repo_root / ".worktrees" / task.id
        _ensure_git_worktree(repo_root, target, branch_name)
        return target, branch_name

    repo_root = _repo_root_for_worktree_target(requested.parent)
    if repo_root is None:
        raise ValueError(
            f"task {task.id} worktree path {task.workspace_path!r} is not inside a git repo "
            "and does not point at a git repo root"
        )
    _ensure_git_worktree(repo_root, requested, branch_name)
    return requested, branch_name


def resolve_workspace(task: Task, *, board: Optional[str] = None) -> Path:
    """Resolve (and create if needed) the workspace for a task.

    - ``scratch``: a fresh dir under ``<board-root>/workspaces/<id>/``,
      where ``<board-root>`` is the active board's root. The path is the
      same for the dispatcher and every profile worker, so handoff is
      path-stable.
    - ``dir:<path>``: the path stored in ``workspace_path``.  Created
      if missing.  MUST be absolute — relative paths are rejected to
      prevent confused-deputy traversal where ``../../../tmp/attacker``
      resolves against the dispatcher's CWD instead of a meaningful
      root.  Users who want a kanban-root-relative workspace should
      compute the absolute path themselves.
    - ``worktree``: a real linked git worktree. If ``workspace_path`` names
      a repo root, Hermes treats it as an anchor and materializes a linked
      worktree at ``<repo>/.worktrees/<task-id>``. If ``workspace_path`` names
      a concrete target path, Hermes creates/reuses that linked worktree. With
      no ``workspace_path``, Hermes anchors on the board's ``default_workdir``
      and materializes ``<repo>/.worktrees/<task-id>`` per task; if no
      ``default_workdir`` is configured it raises rather than guessing from the
      dispatcher's CWD. When ``branch_name`` is empty, Hermes uses
      ``wt/<task-id>``.

    Persist the resolved path back to the task row via ``set_workspace_path``
    so subsequent runs reuse the same directory.
    """
    kind = task.workspace_kind or "scratch"
    if kind == "scratch":
        if task.workspace_path:
            # Legacy scratch tasks that were set to an explicit path get the
            # same absolute-path guard as dir: — consistent with the
            # threat model.
            p = Path(task.workspace_path).expanduser()
            if not p.is_absolute():
                raise ValueError(
                    f"task {task.id} has non-absolute workspace_path "
                    f"{task.workspace_path!r}; workspace paths must be absolute"
                )
        else:
            p = workspaces_root(board=board) / task.id
        p.mkdir(parents=True, exist_ok=True)
        return p
    if kind == "dir":
        if not task.workspace_path:
            raise ValueError(
                f"task {task.id} has workspace_kind=dir but no workspace_path"
            )
        p = Path(task.workspace_path).expanduser()
        if not p.is_absolute():
            raise ValueError(
                f"task {task.id} has non-absolute workspace_path "
                f"{task.workspace_path!r}; use an absolute path "
                f"(relative paths are ambiguous against the dispatcher's CWD)"
            )
        p.mkdir(parents=True, exist_ok=True)
        return p
    if kind == "worktree":
        p, _branch_name = _resolve_worktree_workspace(task, board=board)
        return p
    raise ValueError(f"unknown workspace_kind: {kind}")


def set_workspace_path(
    conn: sqlite3.Connection, task_id: str, path: Path | str
) -> None:
    with write_txn(conn):
        conn.execute(
            "UPDATE tasks SET workspace_path = ? WHERE id = ? AND task_kind = 'work'",
            (str(path), task_id),
        )


def set_branch_name(
    conn: sqlite3.Connection, task_id: str, branch_name: str
) -> None:
    with write_txn(conn):
        conn.execute(
            "UPDATE tasks SET branch_name = ? WHERE id = ? AND task_kind = 'work'",
            (str(branch_name), task_id),
        )


def _git_output(path: Path, *args: str, binary: bool = False) -> str | bytes:
    """Run one bounded read-only git command or raise a scope error."""
    result = subprocess.run(
        ["git", "-C", str(path), *args],
        capture_output=True,
        text=not binary,
        encoding=None if binary else "utf-8",
        errors=None if binary else "replace",
        timeout=30,
        check=False,
    )
    if result.returncode != 0:
        detail = result.stderr if binary else (result.stderr or result.stdout or "")
        if isinstance(detail, bytes):
            detail = detail.decode("utf-8", errors="replace")
        raise WorktreeScopeError(
            f"git evidence command failed ({' '.join(args)}): {str(detail).strip()[:300]}"
        )
    return result.stdout


def _git_mutation(path: Path, *args: str) -> str:
    """Run one bounded kernel-owned git mutation or fail closed."""
    result = subprocess.run(
        [
            "git",
            "-C",
            str(path),
            "-c",
            "user.name=Hermes Kanban",
            "-c",
            "user.email=kanban@localhost",
            "-c",
            "commit.gpgsign=false",
            *args,
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=60,
        check=False,
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip()
        raise WorktreeScopeError(
            f"git materialization command failed ({' '.join(args[:3])}): "
            f"{detail[:500]}"
        )
    return result.stdout


def _rollback_worktree_materialization(
    conn: sqlite3.Connection, task_id: str, original_head: str
) -> None:
    """Restore only the isolated task worktree changed by this kernel call."""
    task = get_task(conn, task_id)
    if task is None or not task.workspace_path:
        raise WorktreeScopeError("cannot restore an unavailable task worktree")
    workspace = Path(task.workspace_path).expanduser()
    try:
        subprocess.run(
            ["git", "-C", str(workspace), "merge", "--abort"],
            capture_output=True,
            timeout=30,
            check=False,
        )
        _git_mutation(workspace, "reset", "--hard", original_head)
        dirty = _git_output(
            workspace, "status", "--porcelain=v1", "-z", binary=True
        )
        if dirty:
            raise WorktreeScopeError(
                "isolated worktree remained dirty after kernel rollback"
            )
    except WorktreeScopeError:
        raise
    except Exception as exc:
        raise WorktreeScopeError(
            f"could not restore isolated worktree after failed handoff: {exc}"
        ) from exc


def _materialize_remote_worktree_handoff(
    conn: sqlite3.Connection,
    task_id: str,
    *,
    patch_attachment_id: Optional[int],
    merge_parent_heads: bool,
    expected_run_id: Optional[int],
) -> tuple[dict[str, Any], str]:
    """Materialize a sandbox artifact without exposing the host worktree.

    The attachment row and file are untrusted input. Only a bounded
    agent-uploaded patch from the current run is eligible. Git owns parsing;
    the kernel checks the staged path manifest before committing. Parent heads
    come from durable task receipts, never worker prose.
    """
    task = get_task(conn, task_id)
    if task is None:
        raise WorktreeScopeError(f"unknown task {task_id}")
    if not isinstance(merge_parent_heads, bool):
        raise WorktreeScopeError("merge_parent_heads must be a boolean")
    if task.status != "running" or task.current_run_id is None:
        raise WorktreeScopeError(
            "remote worktree handoff requires an active dispatcher run"
        )
    if expected_run_id is not None and int(expected_run_id) != int(task.current_run_id):
        raise WorktreeScopeError("remote worktree handoff run id is stale")
    if task.workspace_kind != "worktree" or not task.workspace_path:
        raise WorktreeScopeError(
            "remote worktree handoff requires an isolated git worktree"
        )
    if not task.base_commit or task.owned_paths is None:
        raise WorktreeScopeError(
            "remote worktree handoff requires a persisted base and owned_paths"
        )
    if merge_parent_heads and not task.integrates_parent_heads:
        raise WorktreeScopeError(
            "merge_parent_heads requires integrates_parent_heads=true"
        )

    workspace = Path(task.workspace_path).expanduser()
    if not workspace.is_absolute() or not workspace.is_dir():
        raise WorktreeScopeError("scoped worktree path is unavailable")
    actual_branch = str(_git_output(workspace, "branch", "--show-current")).strip()
    if not actual_branch or actual_branch != (task.branch_name or "").strip():
        raise WorktreeScopeError("worktree branch does not match the task branch")
    if _git_output(workspace, "status", "--porcelain=v1", "-z", binary=True):
        raise WorktreeScopeError(
            "worktree is dirty before remote handoff; refusing to overwrite it"
        )
    original_head = str(
        _git_output(workspace, "rev-parse", "--verify", "HEAD")
    ).strip()

    attachment: Optional[Attachment] = None
    patch_path: Optional[Path] = None
    patch_sha256: Optional[str] = None
    if patch_attachment_id is not None:
        if isinstance(patch_attachment_id, bool):
            raise WorktreeScopeError("patch_attachment_id must be an integer")
        try:
            attachment_id = int(patch_attachment_id)
        except (TypeError, ValueError) as exc:
            raise WorktreeScopeError("patch_attachment_id must be an integer") from exc
        attachment = get_attachment(conn, attachment_id)
        if attachment is None or attachment.task_id != task_id:
            raise WorktreeScopeError(
                "patch attachment does not belong to the completing task"
            )
        if attachment.uploaded_by != "agent":
            raise WorktreeScopeError(
                "patch attachment must be uploaded by the active agent"
            )
        if not attachment.filename.lower().endswith(".patch"):
            raise WorktreeScopeError("remote code handoff must be a .patch attachment")
        run = conn.execute(
            "SELECT started_at, status, ended_at FROM task_runs "
            "WHERE id = ? AND task_id = ?",
            (int(task.current_run_id), task_id),
        ).fetchone()
        if (
            run is None
            or run["status"] != "running"
            or run["ended_at"] is not None
            or int(attachment.created_at) < int(run["started_at"])
        ):
            raise WorktreeScopeError(
                "patch attachment was not produced by the current active run"
            )
        raw_path = Path(attachment.stored_path).expanduser()
        if raw_path.is_symlink():
            raise WorktreeScopeError("patch attachment cannot be a symlink")
        try:
            patch_path = raw_path.resolve(strict=True)
        except OSError as exc:
            raise WorktreeScopeError("patch attachment file is unavailable") from exc
        if (
            not patch_path.is_file()
            or patch_path.parent.name != task_id
            or patch_path.name != attachment.filename
        ):
            raise WorktreeScopeError("patch attachment storage path is invalid")
        patch_size = patch_path.stat().st_size
        if (
            patch_size <= 0
            or patch_size != int(attachment.size)
            or patch_size > KANBAN_ATTACHMENT_MAX_BYTES
        ):
            raise WorktreeScopeError("patch attachment size does not match its receipt")
        patch_sha256 = hashlib.sha256(patch_path.read_bytes()).hexdigest()

    merged_parent_heads: list[dict[str, str]] = []
    parent_heads_already_integrated = False
    patch_already_materialized = False
    try:
        if merge_parent_heads:
            if not task.project_id:
                raise WorktreeScopeError(
                    "parent-head integration requires a Project-linked task"
                )
            rows = conn.execute(
                "SELECT p.id, p.head_commit, p.owned_paths FROM task_links l "
                "JOIN tasks p ON p.id = l.parent_id "
                "WHERE l.child_id = ? AND p.project_id = ? "
                "AND p.task_kind = 'work' ORDER BY p.id",
                (task_id, task.project_id),
            ).fetchall()
            mutating_rows = [
                row for row in rows if _decode_owned_paths(row["owned_paths"]) != []
            ]
            if not mutating_rows:
                raise WorktreeScopeError(
                    "parent-head integration requires at least one parent git receipt"
                )
            merged_any = False
            for row in mutating_rows:
                parent_head = str(row["head_commit"] or "").strip()
                if not re.fullmatch(r"[0-9a-fA-F]{40,64}", parent_head):
                    raise WorktreeScopeError(
                        f"mutating parent {row['id']} is missing a valid git head receipt"
                    )
                _git_output(workspace, "cat-file", "-e", f"{parent_head}^{{commit}}")
                contains = subprocess.run(
                    [
                        "git", "-C", str(workspace), "merge-base", "--is-ancestor",
                        parent_head, "HEAD",
                    ],
                    capture_output=True,
                    timeout=30,
                    check=False,
                )
                if contains.returncode != 0:
                    _git_mutation(
                        workspace, "merge", "--no-ff", "--no-edit", parent_head
                    )
                    merged_any = True
                merged_parent_heads.append(
                    {"task_id": str(row["id"]), "head_commit": parent_head}
                )
            parent_heads_already_integrated = (
                not merged_any and original_head != task.base_commit
            )

        if attachment is not None and patch_path is not None and patch_sha256 is not None:
            prior_messages = str(
                _git_output(
                    workspace,
                    "log",
                    "--format=%B",
                    f"{task.base_commit}..HEAD",
                )
            )
            attachment_marker = f"Hermes-Patch-Attachment: {attachment.id}"
            hash_marker = f"Hermes-Patch-SHA256: {patch_sha256}"
            patch_already_materialized = (
                attachment_marker in prior_messages and hash_marker in prior_messages
            )
            if not patch_already_materialized:
                _git_output(
                    workspace,
                    "apply",
                    "--check",
                    "--index",
                    "--whitespace=error-all",
                    str(patch_path),
                )
                _git_mutation(
                    workspace,
                    "apply",
                    "--index",
                    "--whitespace=error-all",
                    str(patch_path),
                )
                raw_staged = _git_output(
                    workspace,
                    "diff",
                    "--cached",
                    "--name-only",
                    "--no-renames",
                    "-z",
                    binary=True,
                )
                assert isinstance(raw_staged, bytes)
                staged_paths = [
                    item.decode("utf-8", errors="backslashreplace")
                    for item in raw_staged.split(b"\0")
                    if item
                ]
                if not staged_paths:
                    raise WorktreeScopeError("patch attachment produced no staged changes")
                outside = [
                    path
                    for path in staged_paths
                    if not _path_is_owned(path, task.owned_paths)
                ]
                if outside:
                    preview = ", ".join(outside[:8])
                    raise WorktreeScopeError(
                        f"patch changes paths outside declared ownership: {preview}"
                    )
                message = (
                    f"kanban: materialize remote artifact for {task_id}\n\n"
                    f"Hermes-Kanban-Task: {task_id}\n"
                    f"Hermes-Patch-Attachment: {attachment.id}\n"
                    f"Hermes-Patch-SHA256: {patch_sha256}"
                )
                _git_mutation(workspace, "commit", "--no-gpg-sign", "-m", message)

        materialized_head = str(
            _git_output(workspace, "rev-parse", "--verify", "HEAD")
        ).strip()
        if (
            materialized_head == original_head
            and not patch_already_materialized
            and not parent_heads_already_integrated
        ):
            raise WorktreeScopeError("remote worktree handoff produced no new commit")
        receipt: dict[str, Any] = {
            "kind": "kernel_worktree_materialization_v1",
            "materialized_head": materialized_head,
            "merged_parent_heads": merged_parent_heads,
        }
        if attachment is not None and patch_sha256 is not None:
            receipt.update(
                {
                    "patch_attachment_id": attachment.id,
                    "patch_filename": attachment.filename,
                    "patch_sha256": patch_sha256,
                }
            )
        return receipt, original_head
    except Exception as exc:
        try:
            _rollback_worktree_materialization(conn, task_id, original_head)
        except WorktreeScopeError as rollback_exc:
            raise WorktreeScopeError(
                f"{exc}; rollback also failed: {rollback_exc}"
            ) from exc
        if isinstance(exc, WorktreeScopeError):
            raise
        raise WorktreeScopeError(f"remote worktree handoff failed: {exc}") from exc


def record_worktree_base(
    conn: sqlite3.Connection, task_id: str, workspace_path: Path | str
) -> str:
    """Capture the exact pre-worker HEAD once and persist it durably."""
    workspace = Path(workspace_path).expanduser()
    head = str(_git_output(workspace, "rev-parse", "--verify", "HEAD")).strip()
    if not re.fullmatch(r"[0-9a-fA-F]{40,64}", head):
        raise WorktreeScopeError("worktree HEAD is not an exact git commit")
    with write_txn(conn):
        row = conn.execute(
            "SELECT base_commit FROM tasks WHERE id = ? AND task_kind = 'work'",
            (task_id,),
        ).fetchone()
        if row is None:
            raise WorktreeScopeError(f"unknown task {task_id}")
        existing = str(row["base_commit"] or "").strip()
        if existing:
            return existing
        cur = conn.execute(
            "UPDATE tasks SET base_commit = ? "
            "WHERE id = ? AND base_commit IS NULL AND task_kind = 'work'",
            (head, task_id),
        )
        if cur.rowcount == 1:
            _append_event(conn, task_id, "worktree_base_recorded", {"base_commit": head})
            return head
        row = conn.execute(
            "SELECT base_commit FROM tasks WHERE id = ? AND task_kind = 'work'",
            (task_id,),
        ).fetchone()
        existing = str(row["base_commit"] or "").strip() if row else ""
        if not existing:
            raise WorktreeScopeError("could not persist the worktree base commit")
        return existing


# ---------------------------------------------------------------------------
# Per-run remote sandbox reservation (native authority, no side store)
# ---------------------------------------------------------------------------
#
# A run that hands work to a remote sandbox (see
# :func:`_materialize_remote_worktree_handoff`) must own exactly ONE sandbox:
# two concurrent provisioning retries inside one worker turn would otherwise
# leave a live machine nobody tracks. The reservation therefore lives in the
# board's append-only ``task_events`` log, folded per run — not in a side
# JSON file:
#
#   * ``task_events`` is append-only and never rewritten, unlike
#     ``task_runs.metadata`` (which every ``UPDATE task_runs SET metadata``
#     writer replaces wholesale, so a reservation there could be silently
#     dropped mid-run and a duplicate machine created).
#   * ``write_txn`` is ``BEGIN IMMEDIATE``, so folding the log and appending
#     the next transition in one transaction IS the compare-and-swap. A
#     generation number makes the CAS explicit and visible in the history.
#   * The record is board-scoped, so it resolves identically from every
#     profile home, and it shows up in the ordinary task history an operator
#     already reads.

#: The transition kinds this reservation appends, in ``task_events.kind``.
RUN_SANDBOX_EVENTS = (
    "sandbox_reserved",
    "sandbox_provisioned",
    "sandbox_released",
)
_RUN_SANDBOX_STATES = {
    "sandbox_reserved": "reserved",
    "sandbox_provisioned": "active",
    "sandbox_released": "released",
}
#: Which prior states each transition may follow. ``reserved`` opens a new
#: generation; the other two settle the generation already open.
_RUN_SANDBOX_FROM = {
    "sandbox_reserved": {"absent", "released"},
    "sandbox_provisioned": {"reserved"},
    "sandbox_released": {"reserved", "active"},
}
_RUN_SANDBOX_ID_RE = re.compile(r"\A[A-Za-z0-9][A-Za-z0-9._:-]{0,127}\Z")
_MAX_RUN_SANDBOX_RECEIPT_KEYS = 32


class RunSandboxConflict(Exception):
    """Another attempt advanced this run's sandbox reservation first."""


def _run_sandbox_receipt(value: Any) -> dict:
    """Admit only a flat, printable, JSON-safe receipt.

    The receipt is read back by a model-facing tool, so the shapes that can
    be persisted here are deliberately narrow: no nested containers beyond
    one level, no floats, and no non-string keys. A malformed or hostile
    payload is refused at write time rather than surfacing later.
    """
    if not isinstance(value, dict) or not value:
        raise ValueError("sandbox receipt must be a non-empty object")
    if len(value) > _MAX_RUN_SANDBOX_RECEIPT_KEYS:
        raise ValueError("sandbox receipt has too many fields")

    def _scalar(item: Any) -> bool:
        return isinstance(item, (str, bool)) or (
            isinstance(item, int) and not isinstance(item, bool)
        )

    for key, item in value.items():
        if not isinstance(key, str) or not key:
            raise ValueError("sandbox receipt keys must be non-empty strings")
        if _scalar(item):
            continue
        if isinstance(item, list) and all(isinstance(entry, str) for entry in item):
            continue
        if isinstance(item, dict) and all(
            isinstance(entry_key, str) and _scalar(entry)
            for entry_key, entry in item.items()
        ):
            continue
        raise ValueError(f"sandbox receipt field {key!r} has an unsupported shape")
    return json.loads(json.dumps(value))


def read_run_sandbox(
    conn: sqlite3.Connection, task_id: str, *, run_id: int
) -> dict:
    """Fold one run's sandbox events into its current reservation.

    Returns ``{"generation", "state", "sandbox_id", "receipt"}``. ``state``
    is ``absent`` before the first reservation, then ``reserved`` (being
    provisioned), ``active`` (a machine exists and its receipt is durable),
    or ``released`` (the generation was abandoned or retired). Only events
    carrying this exact ``run_id`` are folded, so a later run always starts
    from ``absent`` and can never adopt a previous run's machine.
    """
    run_id = int(run_id)
    record = {
        "generation": 0,
        "state": "absent",
        "sandbox_id": None,
        "receipt": None,
    }
    placeholders = ", ".join("?" for _ in RUN_SANDBOX_EVENTS)
    rows = conn.execute(
        f"SELECT kind, payload FROM task_events WHERE task_id = ? AND run_id = ? "
        f"AND kind IN ({placeholders}) ORDER BY id ASC",
        (task_id, run_id, *RUN_SANDBOX_EVENTS),
    ).fetchall()
    for row in rows:
        kind = str(row["kind"])
        try:
            payload = json.loads(row["payload"]) if row["payload"] else {}
        except ValueError:
            payload = {}
        if not isinstance(payload, dict):
            payload = {}
        generation = payload.get("generation")
        if not isinstance(generation, int) or isinstance(generation, bool):
            # An unparseable transition is not evidence of a live machine;
            # fail closed by treating the reservation as still open.
            record["state"] = "reserved"
            continue
        record["generation"] = generation
        record["state"] = _RUN_SANDBOX_STATES[kind]
        if kind == "sandbox_provisioned":
            sandbox_id = payload.get("sandbox_id")
            receipt = payload.get("receipt")
            record["sandbox_id"] = (
                sandbox_id if isinstance(sandbox_id, str) and sandbox_id else None
            )
            record["receipt"] = receipt if isinstance(receipt, dict) else None
            if record["sandbox_id"] is None or record["receipt"] is None:
                # A provisioned event we cannot read is not reusable, but it
                # does mean a machine may exist: keep the generation open so
                # the caller must retire it before creating another.
                record["state"] = "reserved"
        else:
            record["sandbox_id"] = None
            record["receipt"] = None
    return record


def advance_run_sandbox(
    conn: sqlite3.Connection,
    task_id: str,
    *,
    run_id: int,
    transition: str,
    expected_generation: int,
    sandbox_id: Optional[str] = None,
    receipt: Optional[dict] = None,
    reason: Optional[str] = None,
) -> dict:
    """Append one sandbox reservation transition, or refuse on drift.

    The fold and the append share one ``BEGIN IMMEDIATE`` transaction, so at
    most one concurrent caller can move a given generation forward.
    ``expected_generation`` is the generation the caller last observed;
    ``RunSandboxConflict`` means another attempt already advanced it and the
    caller must re-read instead of creating a second machine.

    ``sandbox_reserved`` opens ``expected_generation + 1``.
    ``sandbox_provisioned`` records the durable receipt for the open
    generation. ``sandbox_released`` closes it (an abandoned attempt, or a
    machine whose liveness evidence says it is gone).
    """
    if transition not in _RUN_SANDBOX_STATES:
        raise ValueError(f"unknown sandbox transition {transition!r}")
    run_id = int(run_id)
    if isinstance(expected_generation, bool) or not isinstance(
        expected_generation, int
    ):
        raise ValueError("expected_generation must be an integer")
    payload: dict[str, Any] = {}
    if transition == "sandbox_provisioned":
        if not isinstance(sandbox_id, str) or not _RUN_SANDBOX_ID_RE.fullmatch(
            sandbox_id
        ):
            raise ValueError("sandbox_id must be a printable sandbox identifier")
        payload["sandbox_id"] = sandbox_id
        payload["receipt"] = _run_sandbox_receipt(receipt)
    elif sandbox_id is not None or receipt is not None:
        raise ValueError(f"{transition} does not carry a sandbox_id or receipt")
    if reason is not None:
        cleaned = str(reason).strip()
        if not cleaned or len(cleaned) > 100:
            raise ValueError("reason must be a short non-empty code")
        payload["reason"] = cleaned

    with write_txn(conn):
        row = conn.execute(
            "SELECT status, current_run_id FROM tasks "
            "WHERE id = ? AND task_kind = 'work'",
            (task_id,),
        ).fetchone()
        if row is None:
            raise RunSandboxConflict(f"unknown task {task_id}")
        if row["status"] != "running" or int(row["current_run_id"] or 0) != run_id:
            # The reservation is only meaningful for the board's own active
            # run; a stale worker must not keep writing to it.
            raise RunSandboxConflict("this run is no longer the task's active run")
        current = read_run_sandbox(conn, task_id, run_id=run_id)
        if current["generation"] != expected_generation:
            raise RunSandboxConflict(
                "this run's sandbox reservation advanced concurrently"
            )
        if current["state"] not in _RUN_SANDBOX_FROM[transition]:
            raise RunSandboxConflict(
                f"cannot {transition} from state {current['state']!r}"
            )
        generation = (
            expected_generation + 1
            if transition == "sandbox_reserved"
            else expected_generation
        )
        payload["generation"] = generation
        _append_event(conn, task_id, transition, payload, run_id=run_id)
        return read_run_sandbox(conn, task_id, run_id=run_id)


def _path_is_owned(path: str, owned_paths: list[str]) -> bool:
    if "." in owned_paths:
        return True
    return any(path == owner or path.startswith(f"{owner}/") for owner in owned_paths)


def _verify_scoped_worktree_completion(
    conn: sqlite3.Connection, task_id: str
) -> Optional[dict[str, Any]]:
    """Derive an exact completion receipt for an explicitly scoped task."""
    task = get_task(conn, task_id)
    if task is None:
        raise WorktreeScopeError(f"unknown task {task_id}")
    owned_paths = task.owned_paths
    if owned_paths is None:
        return None
    if task.workspace_kind != "worktree":
        if owned_paths == []:
            return None
        raise WorktreeScopeError(
            "mutating owned_paths require an isolated git worktree"
        )
    if not task.workspace_path or not task.base_commit:
        raise WorktreeScopeError(
            "scoped worktree is missing its persisted workspace/base commit"
        )
    workspace = Path(task.workspace_path).expanduser()
    if not workspace.is_absolute() or not workspace.is_dir():
        raise WorktreeScopeError("scoped worktree path is unavailable")

    actual_branch = str(_git_output(workspace, "branch", "--show-current")).strip()
    if not actual_branch or actual_branch != (task.branch_name or "").strip():
        raise WorktreeScopeError("worktree branch does not match the task branch")
    dirty = _git_output(workspace, "status", "--porcelain=v1", "-z", binary=True)
    if dirty:
        raise WorktreeScopeError(
            "worktree is dirty; commit or remove every change before completion"
        )
    head = str(_git_output(workspace, "rev-parse", "--verify", "HEAD")).strip()
    if not re.fullmatch(r"[0-9a-fA-F]{40,64}", head):
        raise WorktreeScopeError("worktree HEAD is not an exact git commit")
    ancestor = subprocess.run(
        ["git", "-C", str(workspace), "merge-base", "--is-ancestor", task.base_commit, head],
        capture_output=True,
        timeout=30,
        check=False,
    )
    if ancestor.returncode != 0:
        raise WorktreeScopeError("task base commit is not an ancestor of worktree HEAD")
    raw_names = _git_output(
        workspace,
        "diff",
        "--name-only",
        "--no-renames",
        "-z",
        f"{task.base_commit}..{head}",
        binary=True,
    )
    assert isinstance(raw_names, bytes)
    changed_paths = [
        item.decode("utf-8", errors="backslashreplace")
        for item in raw_names.split(b"\0")
        if item
    ]
    outside = [path for path in changed_paths if not _path_is_owned(path, owned_paths)]
    if outside:
        preview = ", ".join(outside[:8])
        suffix = " …" if len(outside) > 8 else ""
        raise WorktreeScopeError(
            f"commit changes paths outside declared ownership: {preview}{suffix}"
        )

    # A mutating downstream task in the same Project must actually contain
    # every exact code head it depends on. This turns the parent handoff from
    # advisory prose into a kernel-checked integration invariant. Read-only
    # verification tasks are intentionally excluded: they inspect the exact
    # parent head without incorporating it into their own checkout.
    parent_heads: list[dict[str, str]] = []
    if task.integrates_parent_heads:
        if not task.project_id:
            raise WorktreeScopeError(
                "parent-head integration requires a Project-linked task"
            )
        rows = conn.execute(
            "SELECT p.id, p.head_commit, p.owned_paths FROM task_links l "
            "JOIN tasks p ON p.id = l.parent_id "
            "WHERE l.child_id = ? AND p.project_id = ? "
            "AND p.task_kind = 'work' "
            "ORDER BY p.id",
            (task_id, task.project_id),
        ).fetchall()
        receipt_rows = []
        for row in rows:
            parent_scope = _decode_owned_paths(row["owned_paths"])
            if parent_scope == []:
                continue
            parent_head = str(row["head_commit"] or "").strip()
            if not parent_head:
                raise WorktreeScopeError(
                    f"mutating parent {row['id']} is missing its git head receipt"
                )
            receipt_rows.append(row)
        if not receipt_rows:
            raise WorktreeScopeError(
                "parent-head integration requires at least one parent git receipt"
            )
        for row in receipt_rows:
            parent_head = str(row["head_commit"] or "").strip()
            if not re.fullmatch(r"[0-9a-fA-F]{40,64}", parent_head):
                raise WorktreeScopeError(
                    f"parent {row['id']} has an invalid git head receipt"
                )
            contains_parent = subprocess.run(
                [
                    "git",
                    "-C",
                    str(workspace),
                    "merge-base",
                    "--is-ancestor",
                    parent_head,
                    head,
                ],
                capture_output=True,
                timeout=30,
                check=False,
            )
            if contains_parent.returncode != 0:
                raise WorktreeScopeError(
                    f"completed head does not contain parent head from {row['id']}"
                )
            parent_heads.append({"task_id": str(row["id"]), "head_commit": parent_head})
    if owned_paths and not changed_paths:
        raise WorktreeScopeError(
            "mutating completion has no committed changes"
        )
    receipt_changed_paths = changed_paths[:_MAX_RECEIPT_CHANGED_PATHS]
    receipt: dict[str, Any] = {
        "kind": "scoped_worktree_v1",
        "base_commit": task.base_commit,
        "head_commit": head,
        "branch": actual_branch,
        "owned_paths": list(owned_paths),
        "changed_paths": receipt_changed_paths,
        "parent_heads": parent_heads,
    }
    if len(changed_paths) > _MAX_RECEIPT_CHANGED_PATHS:
        receipt.update(
            {
                "changed_path_count": len(changed_paths),
                "changed_paths_truncated": True,
                "changed_paths_sha256": hashlib.sha256(raw_names).hexdigest(),
            }
        )
    return receipt


# ---------------------------------------------------------------------------
def schedule_task(
    conn: sqlite3.Connection,
    task_id: str,
    *,
    reason: Optional[str] = None,
    expected_run_id: Optional[int] = None,
) -> bool:
    """Park a task in ``scheduled`` so it is waiting on time, not human input.

    ``scheduled`` tasks are intentionally not dispatchable; an external cron,
    human action, or automation can later call ``unblock_task`` to re-gate them
    to ``ready`` (or ``todo`` if parents are still incomplete).
    """
    with write_txn(conn):
        params: list[Any] = [task_id]
        sql = """
            UPDATE tasks
               SET status       = 'scheduled',
                   claim_lock   = NULL,
                   claim_expires= NULL,
                   worker_pid   = NULL
             WHERE id = ?
               AND status IN ('todo', 'ready', 'running', 'blocked')
               AND task_kind = 'work'
        """
        if expected_run_id is not None:
            sql += " AND current_run_id = ?"
            params.append(int(expected_run_id))
        cur = conn.execute(sql, params)
        if cur.rowcount != 1:
            return False
        run_id = _end_run(
            conn, task_id,
            outcome="scheduled", status="scheduled",
            summary=reason,
        )
        if run_id is None and reason:
            run_id = _synthesize_ended_run(
                conn, task_id,
                outcome="scheduled",
                summary=reason,
            )
        _append_event(conn, task_id, "scheduled", {"reason": reason}, run_id=run_id)
        return True


# Dispatcher (one-shot pass)
# ---------------------------------------------------------------------------

# After this many consecutive non-success attempts on a task/profile, the
# dispatcher stops retrying and parks the task in ``blocked`` with a reason so
# a human can investigate. Prevents retry storms when a worker repeatedly times
# out, crashes, or cannot spawn.
DEFAULT_FAILURE_LIMIT = 2
# Legacy alias — callers / tests still reference the old name.
DEFAULT_SPAWN_FAILURE_LIMIT = DEFAULT_FAILURE_LIMIT

# Max bytes to keep in a single worker log file. The dispatcher truncates
# and rotates on spawn if the file is larger than this at spawn time.
DEFAULT_LOG_ROTATE_BYTES = 2 * 1024 * 1024   # 2 MiB
DEFAULT_LOG_BACKUP_COUNT = 1

# Keep a little wall-clock budget for the worker to observe a terminal timeout
# and call kanban_block/kanban_complete before max_runtime_seconds kills it.
KANBAN_TERMINAL_TIMEOUT_GRACE_SECONDS = 30

# ---------------------------------------------------------------------------
# Respawn guard constants
# ---------------------------------------------------------------------------

# Patterns in last_failure_error that indicate a quota / auth blocker.
# These errors won't resolve by retrying immediately — auto-block instead.
_RESPAWN_BLOCKER_RE = re.compile(
    r"\b(quota|rate[\s_\-]?limit|429|403|auth\w*|"
    r"unauthorized|forbidden|billing|subscription|"
    r"access[\s_]denied|permission[\s_]denied|"
    r"invalid[\s_]api[\s_]key)\b",
    re.IGNORECASE,
)

# Within this window a completed run counts as "recent proof"; don't re-spawn.
_RESPAWN_GUARD_SUCCESS_WINDOW = 3600  # 1 hour

# Cooldown after a rate-limited (quota-wall) requeue before the dispatcher
# re-spawns the worker. Without this, a task released by the rate-limit path
# would be re-spawned on the very next tick and immediately bounce off the
# same quota wall, burning a worker slot every tick for hours. The cooldown
# spaces retries out so the board keeps cheaply probing whether quota is back
# without thrashing. Overridable via ``HERMES_KANBAN_RATE_LIMIT_COOLDOWN_SECONDS``
# for operators who want a tighter/looser probe cadence.
DEFAULT_RATE_LIMIT_COOLDOWN_SECONDS = 300  # 5 minutes

# Within this window a GitHub PR URL in a comment blocks re-spawn.
_RESPAWN_GUARD_PR_WINDOW = 86400  # 24 hours

# Pattern matching a GitHub PR URL in task comments.
_RESPAWN_GUARD_PR_URL_RE = re.compile(
    r"https?://github\.com/[^/\s]+/[^/\s]+/pull/\d+",
    re.IGNORECASE,
)


@dataclass
class DispatchResult:
    """Outcome of a single ``dispatch`` pass."""

    reclaimed: int = 0
    promoted: int = 0
    reconciled_orphans: list[str] = field(default_factory=list)
    """Task ids requeued by :func:`reconcile_orphaned_running` this tick —
    ``running`` cards whose claim bookkeeping was broken (no valid claim,
    dead/gone worker). See the reconciliation pass for details."""
    spawned: list[tuple[str, str, str]] = field(default_factory=list)
    """List of ``(task_id, assignee, workspace_path)`` triples."""
    skipped_unassigned: list[str] = field(default_factory=list)
    """Ready task ids skipped because they have no assignee at all.
    Operator-actionable — usually a misfiled task waiting for routing."""
    auto_assigned_default: list[str] = field(default_factory=list)
    """Task ids that were unassigned in the DB and had
    ``kanban.default_assignee`` applied this tick before spawning (#27145).
    Surfaces the auto-assignment to telemetry / CLI / dashboard so the
    operator can see when the dispatcher is acting on the fallback rule
    rather than on explicit per-task assignments."""
    skipped_nonspawnable: list[str] = field(default_factory=list)
    """Ready task ids skipped because their assignee names a control-plane
    lane (a Claude Code terminal like ``orion-cc``) rather than a Hermes
    profile. Expected steady-state on multi-lane setups; NOT an
    operator-actionable failure. Tracked separately so health telemetry
    can distinguish "real stuck" (nothing spawned but spawnable work
    available) from "correctly idle" (nothing spawnable in the queue)."""
    skipped_per_profile_capped: list[tuple[str, str, int]] = field(default_factory=list)
    """Tasks deferred this tick because their assignee is already at
    ``kanban.max_in_progress_per_profile`` (#21582). Each entry is
    ``(task_id, assignee, current_running_count)``. NOT an
    operator-actionable failure — the task will be picked up on a
    subsequent tick when the assignee has capacity. Separate bucket so
    telemetry / dashboards can show "this profile is busy" vs
    "task is genuinely stuck"."""
    skipped_file_scope_conflict: list[tuple[str, list[str]]] = field(default_factory=list)
    """Tasks deferred because a running task in the same Project owns an
    overlapping repository path (or either task lacks an explicit safe
    scope). Each entry is ``(task_id, blocking_task_ids)``. Deferred work is
    preserved in its lane and retries automatically after the lock clears."""
    crashed: list[str] = field(default_factory=list)
    """Task ids reclaimed because their worker PID disappeared."""
    auto_blocked: list[str] = field(default_factory=list)
    """Task ids auto-blocked by the spawn-failure circuit breaker."""
    timed_out: list[str] = field(default_factory=list)
    """Task ids whose workers exceeded ``max_runtime_seconds``."""
    stale: list[str] = field(default_factory=list)
    """Task ids reclaimed because no progress (heartbeat) was seen
    within ``dispatch_stale_timeout_seconds``."""
    respawn_guarded: list[tuple[str, str]] = field(default_factory=list)
    """Tasks skipped by the respawn guard, as ``(task_id, reason)`` pairs.

    Reasons: ``"blocker_auth"`` (quota/auth error — also auto-blocked),
    ``"recent_success"`` (completed run within guard window),
    ``"active_pr"`` (GitHub PR URL in a recent comment)."""
    rate_limited: list[str] = field(default_factory=list)
    """Task ids whose workers bailed on a provider rate-limit / quota wall
    (EX_TEMPFAIL sentinel exit) and were released back to ``ready`` WITHOUT
    counting a failure. These never trip the circuit breaker — a long quota
    window just makes the task bounce cheaply until the window clears."""
    skipped_route_unproven: list[tuple[str, str]] = field(default_factory=list)
    """Tasks this tick refused to start because their approved model route
    could not be proven, as ``(task_id, reason)`` pairs.

    These are owner-governed rows whose lock cannot be validated or minted.
    Each is parked for re-approval and skipped INDIVIDUALLY: the route check
    used to run inside ``claim_task``, outside every per-task boundary, so one
    such row raised out of the whole tick and starved every unrelated task
    behind it. Reported in dry runs too, where the same rows used to come back
    as spawnable."""
    skipped_locked: bool = False
    """True when this tick was skipped because another process already held
    the board's dispatch lock (issue #35240). A losing dispatcher does no
    DB writes this tick — the lock holder is making progress on the same
    board. This is the steady-state signal that a single-writer guard is
    actively preventing two dispatchers from racing on ``kanban.db``."""
    skipped_inactive: bool = False
    """True when strict owner-project admission is enabled and this board is
    not active. No reclaim, promotion, claim, or spawn write occurred."""
    memory_pressure: Optional[str] = None
    """System memory pressure observed at spawn time when the memory guard
    restricted this tick (OOF-30/OOF-77): ``"critical"`` — no new workers
    were spawned this tick; ``"elevated"`` — at most one new worker was
    spawned. ``None`` when memory was fine/unknown and the guard imposed
    no restriction. Reclaim/promotion bookkeeping still ran either way;
    deferred tasks stay queued for the next tick."""


# Bounded registry of recently-reaped worker child exits, populated by the
# reap loop at the top of ``dispatch_once`` and consulted by
# ``detect_crashed_workers`` to classify a dead-pid task.
#
# Entry: ``pid -> (raw_wait_status, reaped_at_epoch)``. We keep raw status
# so both ``os.WIFEXITED`` / ``os.WEXITSTATUS`` and ``os.WIFSIGNALED`` can
# be consulted. Entries are trimmed by age (and total size cap as a
# belt-and-braces against unbounded growth on exotic platforms).
_RECENT_WORKER_EXIT_TTL_SECONDS = 600
_RECENT_WORKER_EXITS_MAX = 4096
_recent_worker_exits: "dict[int, tuple[int, float]]" = {}


def _record_worker_exit(pid: int, raw_status: int) -> None:
    """Record a reaped child's exit status for later classification.

    Called from the reap loop in ``dispatch_once``. Safe to call many
    times; duplicate pids overwrite (pids can cycle, latest wins).
    """
    if not pid or pid <= 0:
        return
    now = time.time()
    _recent_worker_exits[int(pid)] = (int(raw_status), now)
    # Age-based trim: drop entries older than the TTL.
    if len(_recent_worker_exits) > _RECENT_WORKER_EXITS_MAX // 2:
        cutoff = now - _RECENT_WORKER_EXIT_TTL_SECONDS
        for _pid in [p for p, (_s, t) in _recent_worker_exits.items() if t < cutoff]:
            _recent_worker_exits.pop(_pid, None)
    # Size cap as a final guard.
    if len(_recent_worker_exits) > _RECENT_WORKER_EXITS_MAX:
        # Drop oldest half.
        ordered = sorted(_recent_worker_exits.items(), key=lambda kv: kv[1][1])
        for _pid, _ in ordered[: len(ordered) // 2]:
            _recent_worker_exits.pop(_pid, None)


def _classify_worker_exit(pid: int) -> "tuple[str, Optional[int]]":
    """Classify a recently-reaped worker by pid.

    Returns ``(kind, code)`` where ``kind`` is one of:

    * ``"clean_exit"`` — ``WIFEXITED`` with ``WEXITSTATUS == 0``. When the
      task is still ``running`` in the DB, this is a protocol violation
      (worker exited without calling ``kanban_complete`` / ``kanban_block``)
      and should be auto-blocked immediately — retrying will just loop.
    * ``"rate_limited"`` — ``WIFEXITED`` with status
      ``KANBAN_RATE_LIMIT_EXIT_CODE``. The worker bailed because the
      provider rate-limited / exhausted quota, NOT because the task failed.
      ``detect_crashed_workers`` releases the task back to ``ready`` without
      counting a failure, so a long quota window can't trip the breaker.
    * ``"nonzero_exit"`` — ``WIFEXITED`` with non-zero status. Real error.
    * ``"signaled"`` — ``WIFSIGNALED`` (OOM killer, SIGKILL, etc). Real crash.
    * ``"unknown"`` — pid was not in the reap registry (either reaped by
      something else, or died between reap tick and liveness check). Fall
      back to existing crashed-counter behavior.

    ``code`` is the exit status (for ``clean_exit`` / ``rate_limited`` /
    ``nonzero_exit``) or the signal number (for ``signaled``), or ``None``
    for ``unknown``.
    """
    entry = _recent_worker_exits.get(int(pid))
    if entry is None:
        return ("unknown", None)
    raw, _ = entry
    try:
        if os.WIFEXITED(raw):
            code = os.WEXITSTATUS(raw)
            if code == 0:
                return ("clean_exit", 0)
            if code == KANBAN_RATE_LIMIT_EXIT_CODE:
                return ("rate_limited", code)
            return ("nonzero_exit", code)
        if os.WIFSIGNALED(raw):
            return ("signaled", os.WTERMSIG(raw))
    except Exception:
        pass
    return ("unknown", None)


def reap_worker_zombies() -> "list[int]":
    """Reap all zombie children of this process without blocking.

    Returns the list of reaped PIDs. Safe to call when there are no
    children (returns []). No-op on Windows.
    """
    reaped: "list[int]" = []
    if os.name != "nt":
        try:
            while True:
                try:
                    pid, status = os.waitpid(-1, os.WNOHANG)
                except ChildProcessError:
                    break
                if pid == 0:
                    break
                _record_worker_exit(pid, status)
                reaped.append(pid)
        except Exception:
            pass
    return reaped


def _pid_alive(pid: Optional[int]) -> bool:
    """Return True if ``pid`` is still running on this host.

    Cross-platform: uses ``OpenProcess`` + ``WaitForSingleObject`` on
    Windows (via ``gateway.status._pid_exists``) and ``os.kill(pid, 0)``
    on POSIX. Returns False for falsy PIDs or on any OS error.

    **DO NOT** use ``os.kill(pid, 0)`` directly on Windows — Python's
    Windows ``os.kill`` treats ``sig=0`` as ``CTRL_C_EVENT`` (bpo-14484)
    and will broadcast it to the target's console group, potentially
    killing unrelated processes.

    **Zombie handling:** the existence check succeeds against zombie
    processes (post-exit, pre-reap) because the process table entry
    still exists. A worker that exits without being reaped by its
    parent would stay "alive" to the dispatcher forever. Dispatcher
    workers are started via ``start_new_session=True`` + intentional
    Popen handle abandonment, so init reaps them quickly — but during
    the window between exit and reap, we'd otherwise see stale "alive"
    signals. On Linux we peek at ``/proc/<pid>/status`` and treat
    ``State: Z`` as dead. On macOS we ask ``ps`` for the BSD ``stat``
    field and treat values containing ``Z`` as dead.
    """
    if not pid or pid <= 0:
        return False
    from gateway.status import _pid_exists
    if not _pid_exists(int(pid)):
        return False
    # Still here → process exists. Check for zombie on platforms
    # where we have a cheap, deterministic process-state probe.
    if sys.platform == "linux":
        try:
            with open(f"/proc/{int(pid)}/status", "r", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("State:"):
                        # "State:\tZ (zombie)" → dead
                        if "Z" in line.split(":", 1)[1]:
                            return False
                        break
        except (FileNotFoundError, PermissionError, OSError):
            # proc entry gone → already reaped; treat as dead.
            # PermissionError shouldn't happen for our own children but
            # be defensive.
            pass
    elif sys.platform == "darwin":
        try:
            proc = subprocess.run(
                ["ps", "-o", "stat=", "-p", str(int(pid))],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True, encoding='utf-8', errors='replace',
                timeout=1,
                check=False,
            )
            if proc.returncode != 0:
                return False
            if "Z" in (proc.stdout or "").strip():
                return False
        except (OSError, subprocess.SubprocessError, TimeoutError):
            # If the secondary probe fails, keep the kill(0) answer.
            pass
    return True


def verified_active_worker_rows(
    conn: sqlite3.Connection,
    *,
    project_id: Optional[str] = None,
    now: Optional[int] = None,
) -> list[sqlite3.Row]:
    """Return only running workers whose native claim is live on this host.

    A persisted ``running`` row is historical state, not proof that a process
    is doing work now.  Owner-facing surfaces may say "Working now" only when
    the task and run still point at each other, their claim/PID agree, the
    claim has not expired, the heartbeat is not stale, and that exact local
    PID is alive.  A remote or otherwise unverifiable worker is omitted rather
    than upgraded from recorded state to a live claim.
    """
    observed_at = int(time.time()) if now is None else int(now)
    where = [
        "t.task_kind = 'work'",
        "t.status = 'running'",
        "r.status = 'running'",
        "r.ended_at IS NULL",
        "r.profile IS NOT NULL",
        "r.worker_pid IS NOT NULL",
        "t.current_run_id = r.id",
        "t.worker_pid = r.worker_pid",
        "t.claim_lock = r.claim_lock",
        "t.claim_expires = r.claim_expires",
    ]
    params: list[Any] = []
    if project_id is not None:
        where.append("t.project_id = ?")
        params.append(str(project_id))
    rows = conn.execute(
        "SELECT r.id AS run_id, r.profile, t.title AS task_title, "
        "r.started_at, r.worker_pid, r.claim_lock, r.claim_expires, "
        "r.last_heartbeat_at AS run_heartbeat, "
        "t.last_heartbeat_at AS task_heartbeat "
        "FROM task_runs r JOIN tasks t ON t.id = r.task_id WHERE "
        + " AND ".join(where)
        + " ORDER BY r.started_at ASC, r.id ASC",
        params,
    ).fetchall()
    host_prefix = f"{_claimer_id().split(':', 1)[0]}:"
    verified: list[sqlite3.Row] = []
    for row in rows:
        claim_lock = str(row["claim_lock"] or "")
        claim_expires = row["claim_expires"]
        if (
            not claim_lock.startswith(host_prefix)
            or claim_expires is None
            or int(claim_expires) < observed_at
            or not _pid_alive(row["worker_pid"])
        ):
            continue
        task_heartbeat = row["task_heartbeat"]
        run_heartbeat = row["run_heartbeat"]
        if task_heartbeat != run_heartbeat:
            continue
        if task_heartbeat is None:
            # A worker can be visible before its first activity callback, but
            # only for a short launch window.  Beyond that, PID state alone is
            # not evidence of useful work.
            if observed_at - int(row["started_at"]) > 120:
                continue
        else:
            heartbeat_age = observed_at - int(task_heartbeat)
            if (
                heartbeat_age < 0
                or heartbeat_age > DEFAULT_CLAIM_HEARTBEAT_MAX_STALE_SECONDS
            ):
                continue
        verified.append(row)
    return verified


def _terminate_reclaimed_worker(
    pid: Optional[int],
    claim_lock: Optional[str],
    *,
    signal_fn=None,
) -> dict[str, Any]:
    """Best-effort host-local worker termination for reclaim paths."""
    import signal

    info: dict[str, Any] = {
        "prev_pid": int(pid) if pid else None,
        "host_local": False,
        "termination_attempted": False,
        "terminated": False,
        "sigkill": False,
    }
    if not pid or pid <= 0 or not claim_lock:
        return info

    host_prefix = f"{_claimer_id().split(':', 1)[0]}:"
    if not str(claim_lock).startswith(host_prefix):
        return info
    info["host_local"] = True

    kill = signal_fn if signal_fn is not None else (
        os.kill if hasattr(os, "kill") else None
    )
    if kill is None:
        return info

    info["termination_attempted"] = True
    try:
        kill(int(pid), signal.SIGTERM)
    except ProcessLookupError:
        # Process is already gone — that's a successful termination, not a
        # survival. Leaving terminated=False here would make the reclaim guard
        # misread a dead worker as still-alive and defer forever.
        info["terminated"] = True
        return info
    except OSError:
        return info

    for _ in range(10):
        if not _pid_alive(pid):
            info["terminated"] = True
            return info
        time.sleep(0.5)

    if _pid_alive(pid):
        try:
            # signal.SIGKILL doesn't exist on Windows; fall back to SIGTERM
            # (which maps to TerminateProcess via the stdlib shim).
            _sigkill = getattr(signal, "SIGKILL", signal.SIGTERM)
            kill(int(pid), _sigkill)
            info["sigkill"] = True
        except (ProcessLookupError, OSError):
            return info

    info["terminated"] = not _pid_alive(pid)
    return info


def _worker_survived_termination(termination: dict) -> bool:
    """True when we tried to kill our own host-local worker and it is still alive.

    Reclaiming in this state would release the claim and let the dispatcher
    spawn a second worker while the first is still running — the duplication
    loop. Only host-local workers we actually signalled count: a non-local
    claim lock or a no-op attempt (no ``os.kill`` available) must fall through
    to the normal release path, since we cannot manage that worker anyway.
    """
    return bool(
        termination.get("termination_attempted")
        and termination.get("host_local")
        and not termination.get("terminated")
    )


def _defer_reclaim_for_live_worker(
    conn: sqlite3.Connection,
    task_id: str,
    claim_lock: Optional[str],
    now: int,
    termination: dict,
    *,
    reason: str,
) -> None:
    """Hold a claim whose worker survived termination instead of releasing it.

    Extends ``claim_expires`` by ``RECLAIM_DEFER_GRACE_SECONDS`` so the task
    stays ``running`` (no duplicate spawn) and records a ``reclaim_deferred``
    event so the hold is visible in ``hermes kanban tail``. The next dispatch
    tick retries the kill; this is self-correcting because not spawning a
    duplicate is what lets the throttled worker finally die.
    """
    grace = now + RECLAIM_DEFER_GRACE_SECONDS
    with write_txn(conn):
        cur = conn.execute(
            "UPDATE tasks SET claim_expires = ? "
            "WHERE id = ? AND status = 'running' AND claim_lock IS ?",
            (grace, task_id, claim_lock),
        )
        if cur.rowcount != 1:
            return
        run_id = _current_run_id(conn, task_id)
        if run_id is not None:
            conn.execute(
                "UPDATE task_runs SET claim_expires = ? WHERE id = ?",
                (grace, run_id),
            )
        payload = {
            "reason": reason,
            "claim_lock": claim_lock,
            "claim_expires_now": grace,
        }
        payload.update(termination)
        _append_event(conn, task_id, "reclaim_deferred", payload, run_id=run_id)


def heartbeat_worker(
    conn: sqlite3.Connection,
    task_id: str,
    *,
    note: Optional[str] = None,
    expected_run_id: Optional[int] = None,
) -> bool:
    """Record a ``heartbeat`` event + touch ``last_heartbeat_at``.

    Called by long-running workers as a liveness signal orthogonal to
    the PID check. A worker that forks a long-lived child (train loop,
    video encode, web crawl) can have its Python still alive while the
    actual work process is stuck; periodic heartbeats catch that.

    Returns True on success, False if the task is not in a state that
    should be heartbeating (not running, or claim expired).
    """
    now = int(time.time())
    with write_txn(conn):
        if expected_run_id is None:
            cur = conn.execute(
                "UPDATE tasks SET last_heartbeat_at = ? "
                "WHERE id = ? AND status = 'running'",
                (now, task_id),
            )
        else:
            cur = conn.execute(
                "UPDATE tasks SET last_heartbeat_at = ? "
                "WHERE id = ? AND status = 'running' AND current_run_id = ?",
                (now, task_id, int(expected_run_id)),
            )
        if cur.rowcount != 1:
            return False
        run_id = (
            int(expected_run_id)
            if expected_run_id is not None
            else _current_run_id(conn, task_id)
        )
        if run_id is not None:
            conn.execute(
                "UPDATE task_runs SET last_heartbeat_at = ? WHERE id = ?",
                (now, run_id),
            )
        _append_event(
            conn, task_id, "heartbeat",
            {"note": note} if note else None,
            run_id=run_id,
        )
    return True


def enforce_max_runtime(
    conn: sqlite3.Connection,
    *,
    signal_fn=None,
) -> list[str]:
    """Terminate workers whose per-task ``max_runtime_seconds`` has elapsed.

    Sends SIGTERM, waits a short grace window, then SIGKILL. Emits a
    ``timed_out`` event and restores the task's source phase so the next
    dispatcher tick re-spawns the same kind of worker — unless the circuit
    breaker has already given up, in which case the task stays blocked
    where ``_record_spawn_failure`` parked it.

    Runs host-local: only tasks claimed by this host are candidates
    (same reasoning as ``detect_crashed_workers``). ``signal_fn`` is a
    test hook; defaults to ``os.kill`` on POSIX.
    """
    import signal
    timed_out: list[str] = []
    now = int(time.time())
    host_prefix = f"{_claimer_id().split(':', 1)[0]}:"

    rows = conn.execute(
        "SELECT t.id, t.worker_pid, "
        "       COALESCE(r.started_at, t.started_at) AS active_started_at, "
        "       t.max_runtime_seconds, t.claim_lock "
        "FROM tasks t "
        "LEFT JOIN task_runs r ON r.id = t.current_run_id "
        "WHERE t.status = 'running' AND t.max_runtime_seconds IS NOT NULL "
        "  AND COALESCE(r.started_at, t.started_at) IS NOT NULL "
        "  AND t.worker_pid IS NOT NULL AND t.task_kind = 'work'"
    ).fetchall()
    for row in rows:
        lock = row["claim_lock"] or ""
        if not lock.startswith(host_prefix):
            continue
        # Runtime is per attempt, not lifetime-of-task. ``tasks.started_at``
        # intentionally records the first time a task ever started, so retries
        # must be measured from the active task_runs row when present.
        elapsed = now - int(row["active_started_at"])
        if elapsed < int(row["max_runtime_seconds"]):
            continue

        pid = int(row["worker_pid"])
        tid = row["id"]
        # SIGTERM then SIGKILL. Keep it simple: 5 s grace. Workers that
        # want a cleaner shutdown can install their own SIGTERM handler
        # before the grace expires.
        killed = False
        kill = signal_fn if signal_fn is not None else (
            os.kill if hasattr(os, "kill") else None
        )
        if kill is not None:
            try:
                kill(pid, signal.SIGTERM)
            except (ProcessLookupError, OSError):
                pass
            # Short polling wait — no time.sleep on the write txn.
            for _ in range(10):
                if not _pid_alive(pid):
                    break
                time.sleep(0.5)
            if _pid_alive(pid):
                try:
                    # signal.SIGKILL doesn't exist on Windows.
                    _sigkill = getattr(signal, "SIGKILL", signal.SIGTERM)
                    kill(pid, _sigkill)
                    killed = True
                except (ProcessLookupError, OSError):
                    pass

        with write_txn(conn):
            retry_status = _retry_status_for_run(conn, tid)
            cur = conn.execute(
                "UPDATE tasks SET status = ?, claim_lock = NULL, "
                "claim_expires = NULL, worker_pid = NULL, "
                "last_heartbeat_at = NULL "
                "WHERE id = ? AND status = 'running' "
                "  AND worker_pid = ? AND claim_lock IS ? AND task_kind = 'work'",
                (retry_status, tid, pid, row["claim_lock"]),
            )
            if cur.rowcount == 1:
                payload = {
                    "pid": pid,
                    "elapsed_seconds": int(elapsed),
                    "limit_seconds": int(row["max_runtime_seconds"]),
                    "sigkill": killed,
                    "retry_status": retry_status,
                }
                run_id = _end_run(
                    conn, tid,
                    outcome="timed_out", status="timed_out",
                    error=f"elapsed {int(elapsed)}s > limit {int(row['max_runtime_seconds'])}s",
                    metadata=payload,
                )
                _append_event(
                    conn, tid, "timed_out", payload, run_id=run_id,
                )
                timed_out.append(tid)
        # Increment the unified failure counter. Outside the write_txn
        # above because ``_record_task_failure`` opens its own. If the
        # breaker trips, this flips the retried task to ``blocked`` and
        # emits a ``gave_up`` event on top of the ``timed_out`` we
        # already emitted.
        if cur.rowcount == 1:
            _record_task_failure(
                conn, tid,
                error=f"elapsed {int(elapsed)}s > limit {int(row['max_runtime_seconds'])}s",
                outcome="timed_out",
                release_claim=False,
                end_run=False,
                event_payload_extra={
                    "pid": pid,
                    "sigkill": killed,
                    "retry_status": retry_status,
                },
            )
    return timed_out


# Heartbeat staleness heartbeat gap — if a running task hasn't sent a
# heartbeat in this many seconds it's considered inactive regardless of
# the ``dispatch_stale_timeout_seconds`` threshold.  Hardcoded at 1 hour
# to match the original spec (">4h started + no commits in 1h").
_STALE_HEARTBEAT_GAP_SECONDS = 3600


def detect_stale_running(
    conn: sqlite3.Connection,
    *,
    stale_timeout_seconds: int = 0,
    signal_fn=None,
) -> list[str]:
    """Reclaim ``running`` tasks that show no progress (heartbeat) within the
    staleness window.

    A task is considered stale when BOTH of these hold:

    1. It has been running for longer than ``stale_timeout_seconds``
       (measured from the active run's ``started_at``, falling back to
       ``tasks.started_at`` on older runs).
    2. Its ``last_heartbeat_at`` is older than
       ``_STALE_HEARTBEAT_GAP_SECONDS`` (or NULL — never sent a heartbeat).

    On reclaim the task is restored to its source phase, the run is closed with
    ``outcome='stale'``, and the host-local worker (if still running) is
    terminated.

    Only considers ``status='running'`` tasks. Blocked tasks are never
    candidates.  Returns the list of reclaimed task IDs.

    ``stale_timeout_seconds=0`` disables the check entirely (returns ``[]``
    immediately).  ``signal_fn`` is a test hook; defaults to ``os.kill``
    on POSIX.
    """
    if stale_timeout_seconds <= 0:
        return []


    now = int(time.time())
    reclaimed: list[str] = []

    rows = conn.execute(
        "SELECT t.id, t.worker_pid, t.last_heartbeat_at, t.claim_lock, "
        "       COALESCE(r.started_at, t.started_at) AS active_started_at "
        "FROM tasks t "
        "LEFT JOIN task_runs r ON r.id = t.current_run_id "
        "WHERE t.status = 'running' AND t.task_kind = 'work'"
    ).fetchall()

    for row in rows:
        # Skip if no started_at (shouldn't happen for running, but be safe).
        if row["active_started_at"] is None:
            continue

        elapsed = now - int(row["active_started_at"])
        if elapsed < stale_timeout_seconds:
            continue  # not old enough to check

        last_hb = row["last_heartbeat_at"]
        hb_age = (now - int(last_hb)) if last_hb is not None else None
        if hb_age is not None and hb_age < _STALE_HEARTBEAT_GAP_SECONDS:
            continue  # recent heartbeat → still alive

        pid = row["worker_pid"]
        tid = row["id"]
        lock = row["claim_lock"] or ""

        # Terminate the worker if it's still host-local.
        termination = _terminate_reclaimed_worker(
            pid, lock, signal_fn=signal_fn,
        )

        # Never release a claim while our own worker is still alive: that would
        # spawn a duplicate beside it. Hold the claim and retry next tick.
        if _worker_survived_termination(termination):
            _defer_reclaim_for_live_worker(
                conn, tid, lock, now, termination,
                reason="heartbeat_stale_worker_alive",
            )
            continue

        with write_txn(conn):
            retry_status = _retry_status_for_run(conn, tid)
            cur = conn.execute(
                "UPDATE tasks SET status = ?, claim_lock = NULL, "
                "claim_expires = NULL, worker_pid = NULL, "
                "last_heartbeat_at = NULL "
                "WHERE id = ? AND status = 'running' "
                "  AND claim_lock IS ?",
                (retry_status, tid, row["claim_lock"]),
            )
            if cur.rowcount != 1:
                continue

            payload = {
                "elapsed_seconds": int(elapsed),
                "last_heartbeat_at": (
                    int(last_hb) if last_hb is not None else None
                ),
                "heartbeat_age_seconds": (
                    int(hb_age) if hb_age is not None else None
                ),
                "timeout_seconds": stale_timeout_seconds,
                "pid": int(pid) if pid else None,
                "retry_status": retry_status,
            }
            payload.update(termination)

            run_id = _end_run(
                conn, tid,
                outcome="stale", status="stale",
                error=(
                    f"no heartbeat for {int(hb_age)}s "
                    if hb_age is not None
                    else "no heartbeat ever"
                ) + f" after {int(elapsed)}s running",
                metadata=payload,
            )
            _append_event(
                conn, tid, "stale", payload, run_id=run_id,
            )
            reclaimed.append(tid)

        # Intentionally NOT calling _record_task_failure here. Stale reclaim
        # is dispatcher-side detection of an absent heartbeat; the task is
        # going straight back to its source phase for re-dispatch. Counting it as
        # a worker failure would let two legitimately-long-running tasks
        # (>4h without explicit heartbeat) trip the circuit breaker and
        # auto-block, even though no worker actually failed. The 'stale'
        # event already lives in task_events for auditability; that's the
        # right surface for "this happened" without conflating with the
        # spawn_failed / timed_out / crashed counters.

    return reclaimed


def reconcile_orphaned_running(
    conn: sqlite3.Connection,
) -> list[str]:
    """Reconcile ``running`` cards whose claim bookkeeping is broken.

    Tracked-state vs. reality divergence: a task can sit in
    ``status='running'`` with ``claim_lock IS NULL`` or ``claim_expires IS
    NULL`` (crash mid-claim, manual SQL, DB restore). None of the other
    recovery paths ever touch such a card — ``release_stale_claims``
    requires a non-NULL ``claim_expires``, ``detect_crashed_workers``
    requires a host-local claim_lock + worker_pid, and
    ``detect_stale_running`` is disabled by default — so the card shows
    Running forever (a zombie).

    This pass finds those orphans, requeues them to ``ready`` with an
    explanatory comment, closes any leaked run, and appends a
    ``reconciled`` event. If the orphan row still records a live PID on
    this host, requeueing is deferred to a later tick so we never spawn a
    duplicate beside a possibly-alive worker.

    Returns the list of reconciled task ids. Safe to call every tick.

    Idea from openai/symphony's tracker reconciliation (Apache-2.0).
    """
    now = int(time.time())
    reconciled: list[str] = []
    rows = conn.execute(
        "SELECT id, claim_lock, claim_expires, worker_pid FROM tasks "
        "WHERE status = 'running' "
        "  AND (claim_lock IS NULL OR claim_expires IS NULL) AND task_kind = 'work'"
    ).fetchall()
    for row in rows:
        tid = row["id"]
        pid = row["worker_pid"]
        if pid and _pid_alive(pid):
            # The recorded worker may still be doing real work — never
            # requeue beside a live process. Retry next tick.
            _log.debug(
                "kanban reconcile: task %s has broken claim bookkeeping but "
                "pid %s is alive on this host — deferring", tid, pid,
            )
            continue
        with write_txn(conn):
            cur = conn.execute(
                "UPDATE tasks SET status = 'ready', claim_lock = NULL, "
                "claim_expires = NULL, worker_pid = NULL, "
                "last_heartbeat_at = NULL "
                "WHERE id = ? AND status = 'running' "
                "  AND claim_lock IS ? AND claim_expires IS ?",
                (tid, row["claim_lock"], row["claim_expires"]),
            )
            if cur.rowcount != 1:
                continue
            payload = {
                "reason": "orphaned_running",
                "claim_lock": row["claim_lock"],
                "claim_expires": (
                    int(row["claim_expires"])
                    if row["claim_expires"] is not None else None
                ),
                "worker_pid": int(pid) if pid else None,
                "now": now,
            }
            run_id = _end_run(
                conn, tid,
                outcome="reclaimed", status="reclaimed",
                error="orphaned running card (broken claim bookkeeping)",
                metadata=payload,
            )
            # Inline comment INSERT — add_comment opens its own write_txn
            # and would raise on nesting (see write_txn pitfalls).
            conn.execute(
                "INSERT INTO task_comments (task_id, author, body, created_at) "
                "VALUES (?, ?, ?, ?)",
                (
                    tid, "dispatcher",
                    "reconciliation: card was 'running' with no valid claim "
                    "(dead/gone worker) — requeued to ready",
                    now,
                ),
            )
            _append_event(conn, tid, "reconciled", payload, run_id=run_id)
            reconciled.append(tid)
        _log.info(
            "kanban reconcile: requeued orphaned running task %s "
            "(claim_lock=%r, worker_pid=%r)", tid, row["claim_lock"], pid,
        )
    return reconciled


def _error_fingerprint(error_text: str) -> str:
    """Normalize an error message for grouping identical failures.

    Strips host-specific details (PIDs, timestamps) so that errors
    with the same root cause produce the same fingerprint.
    """
    fp = re.sub(r'\bpid \d+\b', 'pid N', error_text[:80])
    fp = re.sub(r'\b\d{10,}\b', '<TS>', fp)
    return fp.lower().strip()


# Empirically ~96% of "clean exit without a terminal tool call" tasks complete
# on a later run (a goal-mode finalize nudge, or the model simply emitting the
# tool call next time), so a protocol violation is NOT deterministic — give it a
# bounded retry before the breaker trips instead of blocking on the first hit.
#
# The budget is a violation-only STREAK, not a share of the unified
# ``consecutive_failures`` counter: it counts consecutive clean-exit protocol
# violations (derived from run history by ``_protocol_violation_streak``), so
# earlier timeouts / nonzero exits neither consume nor extend it, and a
# below-budget violation does not tick the unified counter either. A per-task
# ``max_retries`` overrides this bound — the same "task override wins"
# precedence ``_record_task_failure`` documents for every other failure kind.
_PROTOCOL_VIOLATION_FAILURE_LIMIT = 3

# How far back to walk a task's closed runs when counting the violation
# streak. The streak trips at a handful of violations, so anything beyond a
# few dozen rows (violations interleaved with neutral rate-limited requeues)
# can only mean "way past the bound" anyway.
_PROTOCOL_VIOLATION_SCAN_LIMIT = 50


def _protocol_violation_streak(conn: sqlite3.Connection, task_id: str) -> int:
    """Count the task's trailing run of clean-exit protocol violations.

    Walks the task's closed runs newest-first — including the violation run
    ``detect_crashed_workers`` just closed — and counts how many in a row were
    clean-exit protocol violations:

    * ``rate_limited`` runs are neutral and skipped: a quota wall says nothing
      about the task, exactly as it is neutral for the unified
      ``consecutive_failures`` counter.
    * Any other closed run (completed, plain crash, timeout, spawn failure,
      reclaim, …) breaks the streak, so the bounded retry budget counts ONLY
      protocol violations — mixed failure kinds can neither consume nor
      extend it.

    Violation runs are recognized by the ``protocol_violation`` marker that
    ``detect_crashed_workers`` stamps into the run metadata; the violation
    error text is matched as a fallback for runs recorded before the marker
    existed.
    """
    streak = 0
    rows = conn.execute(
        "SELECT outcome, error, metadata FROM task_runs "
        "WHERE task_id = ? AND ended_at IS NOT NULL "
        "ORDER BY id DESC LIMIT ?",
        (task_id, _PROTOCOL_VIOLATION_SCAN_LIMIT),
    ).fetchall()
    for row in rows:
        outcome = row["outcome"] or ""
        if outcome == "rate_limited":
            continue
        if outcome == "crashed":
            is_violation = False
            raw_meta = row["metadata"]
            if raw_meta:
                try:
                    is_violation = bool(
                        json.loads(raw_meta).get("protocol_violation")
                    )
                except (ValueError, TypeError):
                    is_violation = False
            if not is_violation:
                is_violation = "protocol violation" in (row["error"] or "")
            if is_violation:
                streak += 1
                continue
        break
    return streak


def detect_crashed_workers(conn: sqlite3.Connection) -> list[str]:
    """Reclaim ``running`` tasks whose worker PID is no longer alive.

    Appends a ``crashed`` event and restores the task's source phase.
    Different from ``release_stale_claims``: this checks liveness
    immediately rather than waiting for the claim TTL.

    Only considers tasks claimed by *this host* — PIDs from other hosts
    are meaningless here. The host-local check is enough because
    ``_default_spawn`` always runs the worker on the same host as the
    dispatcher (the whole design is single-host).

    When the reap registry shows the worker exited cleanly (rc=0) but
    the task was still ``running`` in the DB, treat it as a protocol
    violation (worker answered conversationally without calling
    ``kanban_complete`` / ``kanban_block``) and trip the circuit breaker
    on the first occurrence — retrying a worker whose CLI keeps
    returning 0 without a terminal transition just loops forever.

    When the reap registry shows the worker exited with the rate-limit
    sentinel (``KANBAN_RATE_LIMIT_EXIT_CODE``), the worker bailed on a
    provider quota wall, NOT a task failure. Such tasks are released back
    to its source phase WITHOUT counting a failure (so a long quota window can't
    trip the breaker) and stamped with a quota-blocker error so
    ``check_respawn_guard`` defers their respawn until the window clears.
    The ids are returned via the ``_last_rate_limited`` function attribute
    (the public return stays the crashed-only ``list[str]``).
    """
    crashed: list[str] = []
    rate_limited: list[str] = []
    # Per-crash details collected inside the main txn, used after it
    # closes to run ``_record_task_failure`` (which needs its own
    # write_txn so can't nest). ``protocol_violation`` flags the
    # clean-exit-but-still-running case, which is accounted against its
    # own bounded violation streak instead of the unified failure
    # counter (see the post-txn loop below).
    crash_details: list[tuple[str, int, str, bool, str]] = []
    # (task_id, pid, claimer, protocol_violation, error_text)
    # Worker-exit observer payloads (RFC #58548), collected inside the main
    # txn and fired only after every reclaim/accounting txn has committed.
    exited_hook_payloads: list[dict] = []
    with write_txn(conn):
        rows = conn.execute(
            "SELECT id, worker_pid, claim_lock, started_at, assignee "
            "FROM tasks "
            "WHERE status = 'running' AND worker_pid IS NOT NULL "
            "AND task_kind = 'work'"
        ).fetchall()
        host_prefix = f"{_claimer_id().split(':', 1)[0]}:"
        for row in rows:
            # Only check liveness for claims owned by this host.
            lock = row["claim_lock"] or ""
            if not lock.startswith(host_prefix):
                continue
            # Skip liveness check inside the launch-window grace period
            # so a freshly-spawned worker isn't reclaimed before its PID
            # is visible on /proc.
            started_at = row["started_at"] if "started_at" in row.keys() else None
            if started_at is not None:
                grace = _resolve_crash_grace_seconds()
                if time.time() - started_at < grace:
                    continue
            if _pid_alive(row["worker_pid"]):
                continue

            pid = int(row["worker_pid"])
            kind, code = _classify_worker_exit(pid)
            rate_limited_exit = False
            if kind == "clean_exit":
                # Worker subprocess returned 0 but its task is still
                # ``running`` in the DB — it exited without calling
                # ``kanban_complete`` / ``kanban_block``. Overwhelmingly the
                # work itself succeeded and only the paperwork was skipped, so
                # a retry usually completes; the corrective sentence below is
                # surfaced to the retry worker via the prior-attempt error in
                # ``build_worker_context`` (guidance approach from #61817).
                protocol_violation = True
                error_text = (
                    "worker exited cleanly (rc=0) without calling "
                    "kanban_complete or kanban_block — protocol violation. "
                    "If the prior run already did the work, verify it and "
                    "report the result via kanban_complete; a run that ends "
                    "without a terminal kanban call counts as failed no "
                    "matter what it did."
                )
                event_kind = "protocol_violation"
                event_payload = {
                    "pid": pid,
                    "claimer": row["claim_lock"],
                    "exit_code": code,
                    # Durable marker for _protocol_violation_streak: _end_run
                    # copies this payload into the run metadata, which is how
                    # the violation-only retry budget is derived later.
                    "protocol_violation": True,
                }
            elif kind == "rate_limited":
                # Worker bailed because the provider rate-limited / exhausted
                # quota (EX_TEMPFAIL sentinel). This is NOT a task failure —
                # the task is fine, the account just hit a wall. Release it
                # back to its source phase so the respawn guard defers it until the
                # quota window clears, and crucially do NOT count a failure
                # (skip ``_record_task_failure``) so a long quota window can't
                # trip the circuit breaker and permanently block the card.
                protocol_violation = False
                rate_limited_exit = True
                error_text = (
                    f"pid {pid} exited rate-limited (quota wall) — "
                    f"requeued without counting a failure"
                )
                event_kind = "rate_limited"
                event_payload = {
                    "pid": pid,
                    "claimer": row["claim_lock"],
                    "exit_code": code,
                }
            else:
                protocol_violation = False
                if kind == "nonzero_exit":
                    error_text = f"pid {pid} exited with code {code}"
                elif kind == "signaled":
                    error_text = f"pid {pid} killed by signal {code}"
                else:
                    error_text = f"pid {pid} not alive"
                event_kind = "crashed"
                event_payload = {"pid": pid, "claimer": row["claim_lock"]}
                if code is not None and kind != "unknown":
                    event_payload["exit_kind"] = kind
                    event_payload["exit_code"] = code

            retry_status = _retry_status_for_run(conn, row["id"])
            event_payload["retry_status"] = retry_status
            cur = conn.execute(
                "UPDATE tasks SET status = ?, claim_lock = NULL, "
                "claim_expires = NULL, worker_pid = NULL "
                "WHERE id = ? AND status = 'running' "
                "  AND worker_pid = ? AND claim_lock IS ?",
                (retry_status, row["id"], pid, row["claim_lock"]),
            )
            if cur.rowcount == 1:
                # Rate-limited requeues are a clean release, not a crash —
                # record the run outcome as ``rate_limited`` so the board
                # history doesn't show a phantom crash for a quota wall.
                _run_outcome = "rate_limited" if rate_limited_exit else "crashed"
                run_id = _end_run(
                    conn, row["id"],
                    outcome=_run_outcome, status=_run_outcome,
                    error=error_text,
                    metadata=dict(event_payload),
                )
                _append_event(
                    conn, row["id"], event_kind,
                    event_payload,
                    run_id=run_id,
                )
                exited_hook_payloads.append({
                    "task_id": row["id"],
                    "assignee": row["assignee"],
                    "run_id": run_id,
                    "worker_pid": pid,
                    "exit_kind": kind,
                    "exit_code": code,
                    "outcome": _run_outcome,
                    "retry_status": retry_status,
                })
                if rate_limited_exit:
                    # Stamp the failure-error column so ``check_respawn_guard``
                    # recognizes this as a quota blocker and defers the
                    # respawn until the window clears — WITHOUT touching
                    # ``consecutive_failures`` (that's the whole point: no
                    # breaker trip on a throttle).
                    conn.execute(
                        "UPDATE tasks SET last_failure_error = ? WHERE id = ?",
                        (error_text[:500], row["id"]),
                    )
                    rate_limited.append(row["id"])
                else:
                    if protocol_violation:
                        # Stamp the failure error now: a below-budget
                        # violation never reaches ``_record_task_failure``
                        # (which stamps this column for every other failure
                        # kind), yet the board UI and the retry worker's
                        # context still need the violation message + the
                        # corrective guidance it carries.
                        conn.execute(
                            "UPDATE tasks SET last_failure_error = ? "
                            "WHERE id = ?",
                            (error_text[:500], row["id"]),
                        )
                    crashed.append(row["id"])
                    crash_details.append(
                        (row["id"], pid, row["claim_lock"],
                         protocol_violation, error_text)
                    )
    # Outside the main txn: account each crashed task and maybe trip the
    # breaker (the retried task transitions to blocked with a ``gave_up`` event
    # on top of the event we already emitted).
    #
    # Protocol-violation crashes (clean exit, no terminal tool call) get a
    # BOUNDED retry, not an immediate trip: empirically ~96% of these tasks
    # complete on a later run (a goal-mode finalize nudge, or the model simply
    # emitting kanban_complete/kanban_block next time), so blocking on the first
    # occurrence just churned them through the respawn cycle. The retry budget
    # is a violation-only streak (``_protocol_violation_streak``): earlier
    # timeouts / nonzero exits neither consume nor extend it, and a
    # below-budget violation does not tick the unified
    # ``consecutive_failures`` counter, so the two budgets stay independent.
    # A per-task ``max_retries`` overrides the violation bound with the same
    # top precedence it has for every other failure kind. Systemic same-error
    # crashes still trip immediately.
    auto_blocked: list[str] = []
    if crash_details:
        # Fingerprint errors to detect systemic failures.
        _fp_counts: dict[str, int] = {}
        for _, _, _, _, err_text in crash_details:
            fp = _error_fingerprint(err_text)
            _fp_counts[fp] = _fp_counts.get(fp, 0) + 1
        for tid, pid, claimer, protocol_violation, error_text in crash_details:
            if protocol_violation:
                streak = _protocol_violation_streak(conn, tid)
                trow = conn.execute(
                    "SELECT max_retries FROM tasks WHERE id = ?", (tid,),
                ).fetchone()
                if trow is None:
                    continue  # task deleted mid-loop
                task_override = (
                    trow["max_retries"] if "max_retries" in trow.keys() else None
                )
                violation_limit = (
                    int(task_override)
                    if task_override is not None
                    else _PROTOCOL_VIOLATION_FAILURE_LIMIT
                )
                if streak < violation_limit:
                    # Below budget: the task is already back at ``ready``
                    # (respawn allowed) with ``last_failure_error`` stamped.
                    # Deliberately no ``_record_task_failure`` call — a
                    # below-budget violation must not consume the unified
                    # failure budget, just as other failure kinds don't
                    # consume this one.
                    continue
                # Streak reached the bound: trip the breaker. ``force_trip``
                # skips the threshold resolution inside
                # ``_record_task_failure`` because the decision — including
                # the per-task ``max_retries`` override — was already made
                # against the violation streak above.
                tripped = _record_task_failure(
                    conn, tid,
                    error=error_text,
                    outcome="crashed",
                    failure_limit=violation_limit,
                    force_trip=True,
                    release_claim=False,
                    end_run=False,
                    event_payload_extra={
                        "pid": pid,
                        "claimer": claimer,
                        "protocol_violations": streak,
                        "protocol_violation_limit": violation_limit,
                    },
                )
                if tripped:
                    auto_blocked.append(tid)
                continue
            fp = _error_fingerprint(error_text)
            is_systemic = _fp_counts.get(fp, 0) >= 3
            tripped = _record_task_failure(
                conn, tid,
                error=error_text,
                outcome="crashed",
                failure_limit=1 if is_systemic else None,
                release_claim=False,
                end_run=False,
                event_payload_extra={"pid": pid, "claimer": claimer},
            )
            if tripped:
                auto_blocked.append(tid)
    # Stash auto-blocked ids on the function for the dispatch loop to pick up.
    # Keeps the public return type (``list[str]``) stable for direct callers
    # and tests that destructure the result; ``dispatch_once`` reads this
    # side-channel attribute to populate ``DispatchResult.auto_blocked``.
    detect_crashed_workers._last_auto_blocked = auto_blocked  # type: ignore[attr-defined]
    # Same side-channel for rate-limited requeues — these did NOT count a
    # failure and are NOT crashes, so they stay out of the ``crashed`` return.
    detect_crashed_workers._last_rate_limited = rate_limited  # type: ignore[attr-defined]
    # Worker-lifecycle observer (RFC #58548): exit events are tick-derived
    # from this reclaim pass — fired only now, after the main reclaim txn
    # AND the breaker accounting above have committed, so subscribers always
    # observe fully durable board state.
    if exited_hook_payloads and _kanban_observer_consumed("on_kanban_worker_exited"):
        _board = get_current_board()
        for hook_fields in exited_hook_payloads:
            hook_fields = dict(hook_fields)
            _fire_kanban_lifecycle_hook(
                "on_kanban_worker_exited",
                hook_fields.pop("task_id"),
                board=_board,
                **hook_fields,
            )
    return crashed


def _record_task_failure(
    conn: sqlite3.Connection,
    task_id: str,
    error: str,
    *,
    outcome: str,
    failure_limit: int = None,
    force_trip: bool = False,
    release_claim: bool = False,
    end_run: bool = False,
    event_payload_extra: Optional[dict] = None,
) -> bool:
    """Record a non-success outcome (spawn_failed / crashed / timed_out)
    and maybe trip the circuit breaker.

    Unified replacement for the old spawn-only ``_record_spawn_failure``.
    Every path that ends a task with a non-success outcome funnels
    through here so the ``consecutive_failures`` counter and the
    auto-block threshold stay consistent.

    Returns True when the task was auto-blocked (counter reached
    ``failure_limit``), False when it was just updated in place.

    Modes:

    * ``release_claim=True, end_run=True`` — spawn-failure path.
      Caller has a running task with an open run; this transitions
      it back to its source phase (or ``blocked`` when the breaker trips),
      releases the claim, and closes the run with ``outcome=<outcome>``.

    * ``release_claim=False, end_run=False`` — timeout/crash path.
      Caller has ALREADY restored the task's source phase and closed the
      run with the appropriate outcome. This just increments the
      counter; if the breaker trips, the task is re-transitioned
      into ``blocked`` and a ``gave_up`` event is emitted.

    ``event_payload_extra`` merges into the ``gave_up`` event payload
    when the breaker trips, so callers can include outcome-specific
    context (e.g. pid on crash, elapsed on timeout).

    Resolution order for the effective threshold:
      1. per-task ``max_retries`` if set (nothing else overrides)
      2. caller-supplied ``failure_limit`` (gateway passes the config
         value from ``kanban.failure_limit``; tests pass fixed values)
      3. ``DEFAULT_FAILURE_LIMIT``

    ``force_trip=True`` trips the breaker unconditionally, skipping the
    counter-vs-threshold comparison (the resolution order above is then
    only reported in the ``gave_up`` payload, not re-evaluated). Callers
    use it when they have already applied their own bounded-retry policy
    — e.g. the clean-exit protocol-violation streak in
    ``detect_crashed_workers``, which resolves the per-task
    ``max_retries`` override against the violation streak itself. The
    failure is still counted into ``consecutive_failures``.
    """
    if failure_limit is None:
        failure_limit = DEFAULT_FAILURE_LIMIT
    blocked = False
    with write_txn(conn):
        row = conn.execute(
            "SELECT consecutive_failures, status, max_retries, current_run_id "
            "FROM tasks WHERE id = ?", (task_id,),
        ).fetchone()
        if row is None:
            return False
        retry_status = (
            _retry_status_for_run(conn, task_id, row["current_run_id"])
            if release_claim
            else ("review" if row["status"] == "review" else "ready")
        )
        failures = int(row["consecutive_failures"]) + 1

        # Per-task override wins over both caller-supplied and default
        # thresholds. None (the common case) falls through.
        task_override = (
            row["max_retries"] if "max_retries" in row.keys() else None
        )
        if task_override is not None:
            effective_limit = int(task_override)
            limit_source = "task"
        else:
            effective_limit = int(failure_limit)
            limit_source = "dispatcher"

        if force_trip or failures >= effective_limit:
            # Trip the breaker.
            if release_claim:
                # Spawn path: still running, also clear claim state.
                conn.execute(
                    "UPDATE tasks SET status = 'blocked', claim_lock = NULL, "
                    "claim_expires = NULL, worker_pid = NULL, "
                    "consecutive_failures = ?, last_failure_error = ? "
                    "WHERE id = ? AND status IN ('running', 'ready', 'review')",
                    (failures, error[:500], task_id),
                )
            else:
                # Timeout/crash path: source phase already restored with claim
                # cleared; just flip to blocked + update
                # counter fields.
                conn.execute(
                    "UPDATE tasks SET status = 'blocked', "
                    "consecutive_failures = ?, last_failure_error = ? "
                    "WHERE id = ? AND status IN ('ready', 'review', 'running')",
                    (failures, error[:500], task_id),
                )
            run_id = None
            if end_run:
                # Only the spawn path has an open run to close.
                run_id = _end_run(
                    conn, task_id,
                    outcome="gave_up", status="gave_up",
                    error=error[:500],
                    metadata={
                        "failures": failures,
                        "trigger_outcome": outcome,
                        "effective_limit": effective_limit,
                        "limit_source": limit_source,
                        "retry_status": retry_status,
                    },
                )
            payload = {
                "failures": failures,
                "effective_limit": effective_limit,
                "limit_source": limit_source,
                "error": error[:500],
                "trigger_outcome": outcome,
                "retry_status": retry_status,
            }
            if event_payload_extra:
                payload.update(event_payload_extra)
            _append_event(
                conn, task_id, "gave_up", payload, run_id=run_id,
            )
            blocked = True
        else:
            # Below threshold.
            if release_claim:
                # Spawn path: restore the claimed source phase + clear claim.
                conn.execute(
                    "UPDATE tasks SET status = ?, claim_lock = NULL, "
                    "claim_expires = NULL, worker_pid = NULL, "
                    "consecutive_failures = ?, last_failure_error = ? "
                    "WHERE id = ? AND status = 'running'",
                    (retry_status, failures, error[:500], task_id),
                )
            else:
                # Timeout/crash path: caller already restored the source phase.
                conn.execute(
                    "UPDATE tasks SET consecutive_failures = ?, "
                    "last_failure_error = ? WHERE id = ?",
                    (failures, error[:500], task_id),
                )
            if end_run:
                # Spawn path: close the open run with outcome.
                run_id = _end_run(
                    conn, task_id,
                    outcome=outcome, status=outcome,
                    error=error[:500],
                    metadata={
                        "failures": failures,
                        "retry_status": retry_status,
                    },
                )
                _append_event(
                    conn, task_id, outcome,
                    {
                        "error": error[:500],
                        "failures": failures,
                        "retry_status": retry_status,
                    },
                    run_id=run_id,
                )
            # Timeout/crash path's caller already emitted its own event.
    return blocked


# Backward-compat alias. Old name is referenced from tests and possibly
# third-party callers. New code should call ``_record_task_failure``.
def _record_spawn_failure(
    conn: sqlite3.Connection,
    task_id: str,
    error: str,
    *,
    failure_limit: int = None,
) -> bool:
    return _record_task_failure(
        conn, task_id, error,
        outcome="spawn_failed",
        failure_limit=failure_limit,
        release_claim=True,
        end_run=True,
    )


def _set_worker_pid(conn: sqlite3.Connection, task_id: str, pid: int) -> None:
    """Record the spawned child's pid + emit a ``spawned`` event.

    The event's payload carries the pid so a human reading ``hermes kanban
    tail`` can correlate log lines with OS-level traces without opening
    the drawer.
    """
    with write_txn(conn):
        conn.execute(
            "UPDATE tasks SET worker_pid = ? WHERE id = ?",
            (int(pid), task_id),
        )
        run_id = _current_run_id(conn, task_id)
        if run_id is not None:
            conn.execute(
                "UPDATE task_runs SET worker_pid = ? WHERE id = ?",
                (int(pid), run_id),
            )
        _append_event(conn, task_id, "spawned", {"pid": int(pid)}, run_id=run_id)


def _clear_failure_counter(conn: sqlite3.Connection, task_id: str) -> None:
    """Reset the unified consecutive-failures counter.

    Called from ``complete_task`` on successful completion — a fresh
    success means the task + profile combination is working and any
    past failures are history. NOT called on spawn success anymore:
    a successful spawn proves the worker could start but says nothing
    about whether the run will succeed, so we need to let timeouts and
    crashes accumulate across spawn boundaries.
    """
    with write_txn(conn):
        conn.execute(
            "UPDATE tasks SET consecutive_failures = 0, "
            "last_failure_error = NULL WHERE id = ?",
            (task_id,),
        )


# Legacy alias for test-code and anything else that still imports it.
_clear_spawn_failures = _clear_failure_counter


def check_respawn_guard(
    conn: sqlite3.Connection, task_id: str, *, lane: str = "ready",
) -> Optional[str]:
    """Return a guard reason if ``task_id`` should NOT be re-spawned, else None.

    Called per ready/review task in ``dispatch_once`` before any claim attempt.
    Returning a reason defers the spawn this tick; the task stays in its
    source phase and gets another chance on the next dispatcher tick.

    ``lane`` names the dispatch column the task is being spawned from
    (``"ready"`` or ``"review"``). In the review lane the
    ``recent_success`` and ``active_pr`` rules are skipped: a recent PR
    URL comment (and often a recent completed run) is the *precondition*
    of the canonical review handoff — a worker opened a PR and requested
    review — not a duplicate-work signal. Rate-limit cooldown and the
    auth-blocker check still apply in every lane.

    Checks in priority order:

    ``"rate_limit_cooldown"``
        The task's most recent run ended with the ``rate_limited`` outcome
        (a worker bailed on a provider quota wall via the EX_TEMPFAIL
        sentinel) within ``_resolve_rate_limit_cooldown_seconds()``. The
        quota almost certainly hasn't reset yet, so defer the respawn until
        the cooldown elapses — then allow a cheap probe. This is checked
        BEFORE ``blocker_auth`` because the rate-limit requeue stamps a
        quota-flavored ``last_failure_error`` that would otherwise match the
        auth-blocker regex and park the task forever (the rate-limit path
        never increments ``consecutive_failures``, so the breaker can't free
        it). Once the cooldown elapses the task falls through and respawns.

    ``"blocker_auth"``
        The task's last failure error matches a quota / authentication
        pattern. Retrying immediately is unlikely to help (rate limits
        reset on a timer; auth needs human action), so we defer to the
        next tick. The existing ``consecutive_failures`` counter still
        trips the auto-block circuit breaker after ``failure_limit``
        consecutive failures, so a persistent auth error eventually
        blocks via the normal path — but a transient 429 gets a few
        ticks of recovery first.

    ``"recent_success"``
        A completed run exists within ``_RESPAWN_GUARD_SUCCESS_WINDOW``
        seconds. Useful work already succeeded for this task; wait for an
        explicit re-queue rather than immediately re-spawning. Bypassed when an
        explicit re-queue event (status change, promote, unblock, reclaim)
        arrives AFTER that completion — that's a deliberate re-run request.

    ``"active_pr"``
        A GitHub PR URL appears in a recent task comment (within
        ``_RESPAWN_GUARD_PR_WINDOW`` seconds).  A prior worker already
        opened a PR; re-spawning risks a duplicate PR on the same task.

    Stale / dead claim locks are NOT a guard reason — they are handled
    by ``release_stale_claims`` and ``detect_crashed_workers`` which
    reset the task to ``ready`` only after verifying the lock is
    genuinely dead (no live PID on this host).
    """
    row = conn.execute(
        "SELECT last_failure_error FROM tasks WHERE id = ?",
        (task_id,),
    ).fetchone()
    if row is None:
        return None

    now = int(time.time())

    # 1. Rate-limit cooldown. The most recent run ended ``rate_limited``
    #    (quota wall) — defer while inside the cooldown window, then allow a
    #    cheap probe. Must run BEFORE the blocker_auth regex check, because a
    #    rate-limit requeue stamps a quota-flavored last_failure_error that
    #    the regex would otherwise match → defer forever (no failure counter
    #    increment on this path means the breaker can never free it).
    #
    #    We look at the LATEST run only (ORDER BY ended_at DESC LIMIT 1): if a
    #    newer crash/completion superseded the rate-limit run, this guard
    #    no longer applies and the normal paths take over.
    rl_cooldown = _resolve_rate_limit_cooldown_seconds()
    latest_run = conn.execute(
        "SELECT outcome, ended_at FROM task_runs "
        "WHERE task_id = ? AND ended_at IS NOT NULL "
        "ORDER BY ended_at DESC LIMIT 1",
        (task_id,),
    ).fetchone()
    if (
        latest_run is not None
        and latest_run["outcome"] == "rate_limited"
    ):
        if rl_cooldown <= 0:
            # Cooldown disabled — respawn immediately, and skip the
            # blocker_auth regex so the stamped rate-limit text doesn't
            # re-trap the task.
            return None
        ended_at = latest_run["ended_at"]
        if ended_at is not None and (now - int(ended_at)) < rl_cooldown:
            return "rate_limit_cooldown"
        # Cooldown elapsed — allow the respawn. Return early so the
        # blocker_auth check below doesn't catch the rate-limit text we
        # stamped on the task; this path intentionally retries forever
        # (cheaply, spaced by the cooldown) until quota returns or a real
        # crash/completion supersedes it.
        return None

    # 2. Quota / auth blocker: retrying immediately will not help.
    err = row["last_failure_error"]
    if err and _RESPAWN_BLOCKER_RE.search(err):
        return "blocker_auth"

    # Review-lane spawns stop here: a recent completed run and a fresh PR
    # URL comment are the canonical *inputs* to a review handoff (worker
    # opened a PR, then requested review), not signals of duplicate work.
    if lane == "review":
        return None

    # 3. Completed run within guard window — proof of recent success.
    #    Exception: an explicit re-queue AFTER that success (an operator
    #    dragging done→ready, a dependency re-promotion, an unblock, a
    #    reclaim) is a deliberate "run it again" — honor it instead of
    #    deferring. Without this, a manual done→ready just sits there,
    #    silently held by the guard, until the window elapses.
    cutoff = now - _RESPAWN_GUARD_SUCCESS_WINDOW
    recent_completed = conn.execute(
        "SELECT ended_at FROM task_runs "
        "WHERE task_id = ? AND outcome = 'completed' AND ended_at >= ? "
        "ORDER BY ended_at DESC LIMIT 1",
        (task_id, cutoff),
    ).fetchone()
    if recent_completed:
        completed_at = int(recent_completed["ended_at"] or 0)
        requeued_after = conn.execute(
            "SELECT 1 FROM task_events "
            "WHERE task_id = ? AND created_at >= ? "
            "AND kind IN ('status', 'promoted', 'unblocked', 'reclaimed') "
            "LIMIT 1",
            (task_id, completed_at),
        ).fetchone()
        if not requeued_after:
            return "recent_success"

    # 4. GitHub PR URL in a recent comment — prior worker already opened a PR.
    pr_cutoff = now - _RESPAWN_GUARD_PR_WINDOW
    for c in conn.execute(
        "SELECT body FROM task_comments WHERE task_id = ? AND created_at >= ?",
        (task_id, pr_cutoff),
    ).fetchall():
        if c["body"] and _RESPAWN_GUARD_PR_URL_RE.search(c["body"]):
            return "active_pr"

    return None


def has_spawnable_ready(conn: sqlite3.Connection) -> bool:
    """Return True iff there is at least one ready+assigned+unclaimed task
    whose assignee maps to a real Hermes profile.

    Used by the gateway- and CLI-embedded dispatchers' health telemetry to
    decide whether ``0 spawned`` is a "stuck" condition (real spawnable
    work waiting) or a "correctly idle" condition (only control-plane
    lanes like ``orion-cc`` / ``orion-research`` waiting on terminals
    that pull tasks via ``claim_task`` directly).

    Falls back to "any ready+assigned" if ``profile_exists`` is not
    importable (e.g. partial install) — preserves the old behavior so
    the warning still fires in degraded environments.
    """
    rows = conn.execute(
        "SELECT id, assignee FROM tasks "
        "WHERE status = 'ready' AND assignee IS NOT NULL "
        "    AND claim_lock IS NULL AND task_kind = 'work' ORDER BY id"
    ).fetchall()
    if not rows:
        return False
    try:
        from hermes_cli.profiles import profile_exists  # local import: avoids cycle
    except Exception:
        # Can't introspect — assume spawnable, preserve legacy behavior.
        return True
    for row in rows:
        if profile_exists(row["assignee"]) and not file_scope_conflicts(
            conn, row["id"]
        ):
            return True
    return False


def has_spawnable_review(conn: sqlite3.Connection) -> bool:
    """Return True iff there is at least one review+assigned+unclaimed task
    whose assignee maps to a real Hermes profile.

    Mirror of :func:`has_spawnable_ready` for the review column —
    used by the health telemetry to decide whether the dispatcher
    should have spawned a review agent.
    """
    rows = conn.execute(
        "SELECT id, assignee FROM tasks "
        "WHERE status = 'review' AND assignee IS NOT NULL "
        "    AND claim_lock IS NULL AND task_kind = 'work' ORDER BY id"
    ).fetchall()
    if not rows:
        return False
    try:
        from hermes_cli.profiles import profile_exists  # local import: avoids cycle
    except Exception:
        return True
    for row in rows:
        if profile_exists(row["assignee"]) and not file_scope_conflicts(
            conn, row["id"]
        ):
            return True
    return False


def review_dispatch_enabled() -> bool:
    """Return whether first-class review tasks should dispatch automatically.

    The default is true because Hermes ships the ``sdlc-review`` skill and the
    review lifecycle includes a supported reviewer-owned changes-requested
    transition. Operators can disable it for human-only review boards.
    """
    try:
        from hermes_cli.config import load_config
        return bool(
            (load_config() or {}).get("kanban", {}).get("review_dispatch", True)
        )
    except Exception:
        return True


# ---------------------------------------------------------------------------
# Memory-aware dispatch guard (OOF-30 / OOF-77)
#
# Two production incidents ("larrikin-lollies", "synclare-task-manager")
# followed the same shape: no ``kanban.max_in_progress`` configured, a busy
# board, and a 1 GiB VM — the dispatcher fanned out 26-31 concurrent workers,
# the host went into swap-thrash/OOM, and the dashboard (and everything else
# on the machine) became unreachable. Two complementary safeguards:
#
#   1. A memory-DERIVED default concurrency cap when the operator never set
#      ``kanban.max_in_progress`` (``resolve_max_in_progress``) — sized from
#      MemTotal so a 1 GiB VM defaults to 2 workers, not unlimited.
#   2. A live memory-PRESSURE guard inside the dispatch tick itself
#      (``_memory_pressure_level``) — even a correctly-sized static cap can't
#      see other tenants of the box, so under real observed pressure the
#      dispatcher stops adding workers regardless of configured caps.
#
# Both fail open: on non-Linux hosts or any read error the sample is empty,
# the derived default is None (no cap — unchanged behaviour), and the
# pressure level is "unknown" (no spawn restriction).
# ---------------------------------------------------------------------------

# Assumed per-worker memory footprint for the derived default cap. Hermes
# workers are full agent processes (Python + model client + tool subprocesses);
# ~512 MiB is a deliberately conservative planning number so the derived cap
# errs toward fewer workers on small VMs.
MEMORY_GUARD_MB_PER_WORKER = 512
# Bounds for the derived default: never below 2 (a board must still make
# progress on the smallest hosted VM) and never above 8 (operators who want
# more fan-out on big iron should say so explicitly in config).
DERIVED_MAX_IN_PROGRESS_FLOOR = 2
DERIVED_MAX_IN_PROGRESS_CEILING = 8


def _system_memory_sample() -> dict:
    """Best-effort system memory snapshot (KiB values), ``{}`` when unknown.

    Delegates to :func:`gateway.lifecycle_ledger.sample_memory` (pure /proc
    reads, Linux-only, never raises). Local import keeps ``kanban_db``
    importable in stripped-down environments without the gateway package.
    Module-level indirection is also the test seam — the shared conftest
    patches this to ``{}`` so suite results don't depend on the CI runner's
    live memory state.
    """
    try:
        from gateway.lifecycle_ledger import sample_memory
        return sample_memory() or {}
    except Exception:
        return {}


def derive_default_max_in_progress(sample: Optional[Mapping[str, Any]] = None) -> Optional[int]:
    """Memory-derived default for ``kanban.max_in_progress`` when unset.

    ``clamp(MemTotal / MEMORY_GUARD_MB_PER_WORKER, FLOOR, CEILING)`` — e.g.
    a 1 GiB VM derives 2, a 4 GiB VM derives 8. Returns ``None`` (no cap,
    pre-fix behaviour) when total memory can't be determined, so dev
    machines on macOS/Windows are unaffected.
    """
    if sample is None:
        sample = _system_memory_sample()
    total_kib = sample.get("mem_total_kib")
    if isinstance(total_kib, bool) or not isinstance(total_kib, int) or total_kib <= 0:
        return None
    workers = (total_kib // 1024) // MEMORY_GUARD_MB_PER_WORKER
    return max(
        DERIVED_MAX_IN_PROGRESS_FLOOR,
        min(workers, DERIVED_MAX_IN_PROGRESS_CEILING),
    )


def resolve_max_in_progress(configured: Optional[int]) -> Optional[int]:
    """Return the effective global concurrency cap for a dispatch tick.

    An explicit operator-configured value always wins. When unset, fall back
    to the memory-derived default (see :func:`derive_default_max_in_progress`).
    Callers that parse config (gateway dispatcher, ``hermes kanban dispatch``)
    should route through this so both paths agree.
    """
    if configured is not None:
        return configured
    return derive_default_max_in_progress()


def configured_max_in_progress() -> Optional[int]:
    """Read ``kanban.max_in_progress`` from config, or None when unset/invalid.

    Small shared parser so every dispatch entry point (gateway watcher, CLI
    dispatch, standalone daemon) agrees on what "explicitly configured"
    means: a positive integer wins, anything else falls through to the
    memory-derived default via :func:`resolve_max_in_progress`.
    """
    try:
        from hermes_cli.config import load_config_readonly
        raw = (load_config_readonly() or {}).get("kanban", {}).get(
            "max_in_progress"
        )
    except Exception:
        return None
    if raw is None:
        return None
    try:
        ival = int(raw)
    except (TypeError, ValueError):
        return None
    return ival if ival >= 1 else None


def count_running_tasks(conn: sqlite3.Connection) -> int:
    """Return the number of tasks currently in ``status='running'``.

    Used by the gateway's multi-board sweep to account for workers on
    OTHER boards against the host-level concurrency budget (OOF-30): the
    memory-derived cap bounds the machine, so each board's tick must see
    the machine's total, not just its own. Fails open to 0 — a broken
    board must not brick dispatch on healthy ones (corruption is handled
    separately by the watcher's quarantine logic).
    """
    try:
        return int(
            conn.execute(
                "SELECT COUNT(*) FROM tasks "
                "WHERE status = 'running' AND task_kind = 'work'"
            ).fetchone()[0]
        )
    except Exception:
        return 0


def count_running_tasks_other_boards(board: Optional[str] = None) -> int:
    """Total ``running`` tasks across every board EXCEPT ``board``.

    The concurrency caps bound the HOST (workers are OS processes sharing
    one machine's memory), but each board's dispatch tick only sees its own
    DB. Without this, a memory-derived cap of N gets multiplied by the
    number of active boards — reproduced in review of OOF-30: two boards
    each spawned N workers on a derived N-worker host budget.

    Boards are matched by resolved DB path, so the ``HERMES_KANBAN_DB``
    override (which pins every board to one file) naturally yields 0.
    Fails open per board: one broken/corrupt board must not brick dispatch
    on the healthy ones.
    """
    try:
        current_path = str(kanban_db_path(board=board).expanduser().resolve())
    except Exception:
        current_path = None
    try:
        boards = list_boards(include_archived=False)
    except Exception:
        return 0
    total = 0
    for meta in boards:
        slug = meta.get("slug") or DEFAULT_BOARD
        try:
            path = kanban_db_path(board=slug).expanduser()
            resolved = str(path.resolve())
            if current_path is not None and resolved == current_path:
                continue
            if not path.exists():
                continue
            other = connect(board=slug)
            try:
                total += count_running_tasks(other)
            finally:
                try:
                    other.close()
                except Exception:
                    pass
        except Exception:
            continue
    return total


def _memory_pressure_level(sample: Optional[Mapping[str, Any]] = None) -> str:
    """Classify current system memory pressure: ok/elevated/critical/unknown.

    Reuses :func:`gateway.memory_status.classify_pressure` so the dispatcher's
    idea of "critical" matches the memory banner users see on the dashboard
    and the lifecycle ledger's OOM-suspicion heuristics (NS-608/NS-656).
    ``unknown`` (non-Linux, read failure) imposes no restriction — the guard
    must never brick dispatch on hosts where /proc isn't available.
    """
    if sample is None:
        sample = _system_memory_sample()
    if not sample:
        return "unknown"
    try:
        from gateway.memory_status import classify_pressure
        return classify_pressure(
            sample.get("mem_available_kib"), sample.get("mem_total_kib")
        )
    except Exception:
        return "unknown"


def dispatch_once(
    conn: sqlite3.Connection,
    *,
    spawn_fn=None,
    ttl_seconds: Optional[int] = None,
    dry_run: bool = False,
    max_spawn: Optional[int] = None,
    max_in_progress: Optional[int] = None,
    failure_limit: int = DEFAULT_SPAWN_FAILURE_LIMIT,
    stale_timeout_seconds: int = 0,
    board: Optional[str] = None,
    default_assignee: Optional[str] = None,
    max_in_progress_per_profile: Optional[int] = None,
    reconcile_orphans: bool = True,
    require_board_activation: bool = False,
) -> DispatchResult:
    """Run one dispatcher tick under the board's single-writer lock.

    Thin wrapper around :func:`_dispatch_once_locked`. It acquires a
    non-blocking, board-scoped dispatch lock (issue #35240) so that two
    dispatchers pointed at the same ``kanban.db`` — e.g. the service-
    managed gateway and a shell-spawned orphan that escaped the service
    cgroup — can never run a reclaim/spawn/write tick concurrently and
    race on WAL frames. The losing dispatcher returns an empty
    ``DispatchResult`` with ``skipped_locked=True`` and does no DB writes;
    the holder is already making progress on the same board.

    The lock is keyed off the board's resolved DB path, so unrelated
    boards tick in parallel. See :func:`_dispatch_tick_lock` for the
    cross-process / cross-platform mechanics.
    """
    try:
        db_path = kanban_db_path(board=board)
    except Exception:
        if require_board_activation:
            result = DispatchResult(skipped_inactive=True)
            _fire_dispatch_tick_hook(result, board=board, dry_run=dry_run)
            return result
        # Path resolution should never fail, but if it somehow does we
        # must not lose the tick — fall through to an unguarded dispatch
        # rather than dropping work.
        result = _dispatch_once_locked(
            conn,
            spawn_fn=spawn_fn,
            ttl_seconds=ttl_seconds,
            dry_run=dry_run,
            max_spawn=max_spawn,
            max_in_progress=max_in_progress,
            failure_limit=failure_limit,
            stale_timeout_seconds=stale_timeout_seconds,
            board=board,
            default_assignee=default_assignee,
            max_in_progress_per_profile=max_in_progress_per_profile,
            reconcile_orphans=reconcile_orphans,
        )
        _fire_dispatch_tick_hook(result, board=board, dry_run=dry_run)
        return result
    with _dispatch_tick_lock(db_path) as held:
        board_metadata = read_board_metadata(board)
        if not held:
            result = DispatchResult(skipped_locked=True)
        elif not board_ownership_verified(board_metadata):
            # Independent of ``require_board_activation``: a board whose
            # metadata cannot be read cannot be shown NOT to be an owner
            # Project's board, and owner work whose route authority nobody can
            # prove must never be claimed.
            _log.warning(
                "kanban dispatch: board %s has unreadable board.json; skipping "
                "the tick until its ownership can be verified", board,
            )
            result = DispatchResult(skipped_inactive=True)
        elif require_board_activation and not board_dispatch_allowed(
            board_metadata
        ):
            result = DispatchResult(skipped_inactive=True)
        else:
            result = _dispatch_once_locked(
                conn,
                spawn_fn=spawn_fn,
                ttl_seconds=ttl_seconds,
                dry_run=dry_run,
                max_spawn=max_spawn,
                max_in_progress=max_in_progress,
                failure_limit=failure_limit,
                stale_timeout_seconds=stale_timeout_seconds,
                board=board,
                default_assignee=default_assignee,
                max_in_progress_per_profile=max_in_progress_per_profile,
                reconcile_orphans=reconcile_orphans,
            )
            # Still under the dispatch lock: run the periodic PASSIVE WAL
            # checkpoint (see _maybe_checkpoint_wal; the -wal file size is
            # bounded by journal_size_limit on the writer's natural reset).
            _maybe_checkpoint_wal(conn, db_path)
    # The dispatch lock has been released here. Fire the tick observer
    # strictly OUTSIDE the single-writer critical section (#56066 sweeper
    # finding / #64231 disposition): a slow subscriber must never extend
    # the lock hold and stall a sibling dispatcher's tick.
    _fire_dispatch_tick_hook(result, board=board, dry_run=dry_run)
    return result


def _dispatch_once_locked(
    conn: sqlite3.Connection,
    *,
    spawn_fn=None,
    ttl_seconds: Optional[int] = None,
    dry_run: bool = False,
    max_spawn: Optional[int] = None,
    max_in_progress: Optional[int] = None,
    failure_limit: int = DEFAULT_SPAWN_FAILURE_LIMIT,
    stale_timeout_seconds: int = 0,
    board: Optional[str] = None,
    default_assignee: Optional[str] = None,
    max_in_progress_per_profile: Optional[int] = None,
    reconcile_orphans: bool = True,
) -> DispatchResult:
    """Run one dispatcher tick.

    Steps:
      1. Reclaim stale running tasks (TTL expired).
      2. Reclaim stale running tasks (no recent heartbeat).
      3. Reclaim crashed running tasks (host-local PID no longer alive).
      3. Promote todo -> ready where all parents are done.
      4. For each ready task with an assignee, atomically claim and call
         ``spawn_fn(task, workspace_path, board) -> Optional[int]``. The
         return value (if any) is recorded as ``worker_pid`` so subsequent
         ticks can detect crashes before the TTL expires.

    Spawn failures are counted per-task. After ``failure_limit`` consecutive
    failures the task is auto-blocked with the last error as its reason —
    prevents the dispatcher from thrashing forever on an unfixable task.

    ``max_spawn`` is a **live concurrency cap**, not a per-tick spawn budget:
    it counts tasks already in ``status='running'`` plus this tick's spawns
    against the limit. So ``max_spawn=4`` means "at most 4 workers running
    at any time across the whole board" — matching the gateway's stated
    intent ("limit concurrent kanban tasks"). With a per-tick interpretation
    a 60-second tick interval could grow concurrency by N every minute on a
    busy board and accumulate without bound.

    ``max_in_progress`` is a **host-level** concurrency cap (OOF-30): it
    counts running tasks on every active board — not just this one — plus
    this tick's spawns. Workers are OS processes sharing one machine's
    memory, so a per-board interpretation would multiply the cap by the
    number of active boards. ``max_spawn`` retains its historical per-board
    semantics.

    ``spawn_fn`` defaults to ``_default_spawn``. Tests pass a stub.
    ``board`` pins workspace/log/db resolution for this tick to a specific
    board. When omitted, the current-board resolution chain is used.
    """
    # Reap zombie children from previously spawned workers. See
    # reap_worker_zombies() for the full rationale.
    reap_worker_zombies()

    result = DispatchResult()
    result.reclaimed = release_stale_claims(conn)
    if reconcile_orphans:
        # Orphaned-card reconciliation: requeue 'running' cards whose claim
        # bookkeeping is broken (no valid claim, dead/gone worker) that the
        # TTL/crash/stale paths can never see. See reconcile_orphaned_running.
        result.reconciled_orphans = reconcile_orphaned_running(conn)
    result.stale = detect_stale_running(
        conn, stale_timeout_seconds=stale_timeout_seconds,
    )
    result.crashed = detect_crashed_workers(conn)
    # detect_crashed_workers stashes protocol-violation auto-blocks on
    # itself so the public list-return stays stable. Pull them into the
    # DispatchResult here so telemetry / tests see the trip.
    _crash_auto_blocked = getattr(
        detect_crashed_workers, "_last_auto_blocked", []
    )
    if _crash_auto_blocked:
        result.auto_blocked.extend(_crash_auto_blocked)
    # Rate-limited requeues (quota wall, no failure counted) — surface for
    # telemetry / tests. These tasks went back to ``ready`` and the respawn
    # guard will defer them until the quota window clears.
    _crash_rate_limited = getattr(
        detect_crashed_workers, "_last_rate_limited", []
    )
    if _crash_rate_limited:
        result.rate_limited.extend(_crash_rate_limited)
    result.timed_out = enforce_max_runtime(conn)
    result.promoted = recompute_ready(conn, failure_limit=failure_limit)

    # Count tasks already running so max_spawn enforces concurrency rather
    # than a per-tick spawn budget. See the docstring above for the full
    # rationale; the short version is that a 60-second tick interval with a
    # per-tick budget of N would grow concurrency by N every tick on a busy
    # board, since "running" tasks aren't reclaimed by completion alone —
    # they sit in status='running' until the worker calls
    # kanban_complete/kanban_block (or the dispatcher TTL-reclaims them).
    running_count = 0
    spawn_budget: Optional[int] = None
    if max_spawn is not None or max_in_progress is not None:
        running_count = count_running_tasks(conn)

    # Convert any concurrency caps into a shared additional-spawns budget
    # for this tick. Both ready and review loops consume from the same
    # budget so the total number of new workers stays bounded.
    if max_spawn is not None:
        if running_count >= max_spawn:
            return result
        spawn_budget = max_spawn - running_count

    # Honour kanban.max_in_progress across both ready and review queues: if
    # the board already has enough running tasks, skip this tick entirely.
    # When there is room left, intersect the remaining in-progress budget
    # with any explicit max_spawn cap above.
    #
    # max_in_progress is a HOST-level cap, not a per-board one (OOF-30):
    # workers are OS processes sharing one machine's memory, so running
    # workers on every other board count against the same budget. Without
    # this, N active boards multiply the cap by N — exactly the fan-out
    # the memory-derived default exists to prevent.
    if max_in_progress is not None:
        total_running = running_count + count_running_tasks_other_boards(board)
        if total_running >= max_in_progress:
            return result
        remaining = max_in_progress - total_running
        if spawn_budget is None or spawn_budget > remaining:
            spawn_budget = remaining

    # Memory-pressure guard (OOF-30/OOF-77): even a well-chosen static cap
    # can't see the host's actual memory state (other tenants, bloated
    # long-lived workers, dashboard growth). Under observed pressure the
    # dispatcher stops adding load: critical -> spawn nothing this tick;
    # elevated -> at most one new worker. Reclaim/promotion above already
    # ran, so board bookkeeping stays live either way, and deferred tasks
    # simply wait for a later tick. "unknown" imposes no restriction.
    pressure = _memory_pressure_level()
    if pressure == "critical":
        result.memory_pressure = pressure
        _log.warning(
            "kanban dispatch: system memory pressure is critical; "
            "spawning no new workers this tick (deferred, not dropped)"
        )
        return result
    if pressure == "elevated":
        result.memory_pressure = pressure
        if spawn_budget is None or spawn_budget > 1:
            _log.warning(
                "kanban dispatch: system memory pressure is elevated; "
                "limiting to at most 1 new worker this tick"
            )
            spawn_budget = 1

    ready_rows = conn.execute(
        "SELECT id, assignee FROM tasks "
        "WHERE status = 'ready' AND claim_lock IS NULL AND task_kind = 'work' "
        "ORDER BY priority DESC, created_at ASC"
    ).fetchall()
    # Review rows are enumerated up front (not after the ready loop) so the
    # budget split below can see whether review work exists at all.
    review_rows = []
    if review_dispatch_enabled():
        review_rows = conn.execute(
            "SELECT id, assignee FROM tasks "
            "WHERE status = 'review' AND claim_lock IS NULL AND task_kind = 'work' "
            "ORDER BY priority DESC, created_at ASC"
        ).fetchall()
    # Review-lane reservation (OOF-30 review finding): the ready loop runs
    # first and used to consume the ENTIRE shared budget, so a sustained
    # ready backlog permanently starved autonomous reviews — completed work
    # sat in 'review' forever while new work kept spawning. When spawnable
    # review work exists and the tick has any budget, hold one slot back
    # from the ready loop so the review lane always gets a spawn
    # opportunity. The reservation is per-tick and self-releasing: with no
    # spawnable review work (or no cap at all) the ready loop keeps the
    # full budget. "Spawnable" mirrors the review loop's own gate
    # (assigned + real profile) so a review column full of human-pulled
    # control-plane lanes doesn't permanently tax ready throughput.
    def _any_spawnable_review() -> bool:
        if not review_rows:
            return False
        try:
            from hermes_cli.profiles import profile_exists as _rpe
        except Exception:
            # Profiles module unavailable (test stubs, exotic envs) —
            # assume spawnable, matching the review loop's own fallback.
            return any(row["assignee"] for row in review_rows)
        return any(
            row["assignee"] and _rpe(row["assignee"]) for row in review_rows
        )

    ready_budget = spawn_budget
    if spawn_budget is not None and spawn_budget > 0 and _any_spawnable_review():
        ready_budget = max(spawn_budget - 1, 0)
    spawned = 0
    # Per-profile concurrency cap (#21582): when set, track how many
    # workers each assignee already has in flight, and refuse to spawn
    # when this would push that assignee past the cap. Prevents
    # fan-out workloads from melting a single profile's local model /
    # API quota / browser pool while leaving other profiles idle.
    # Tasks blocked this way go to skipped_per_profile_capped (not
    # skipped_unassigned — the operator-actionable signal is different:
    # "this profile is busy, try again later" not "this needs routing").
    _per_profile_cap = max_in_progress_per_profile if (
        isinstance(max_in_progress_per_profile, int)
        and max_in_progress_per_profile > 0
    ) else None
    _per_profile_running: dict[str, int] = {}
    if _per_profile_cap is not None:
        for prow in conn.execute(
            "SELECT assignee, COUNT(*) AS n FROM tasks "
            "WHERE status = 'running' AND assignee IS NOT NULL "
            "AND task_kind = 'work' GROUP BY assignee"
        ):
            _per_profile_running[prow["assignee"]] = int(prow["n"])
    # Normalize default_assignee once: empty/whitespace string → None so the
    # rest of the loop can use ``if default_assignee:`` as a single check.
    # We also resolve profile_exists once here for the same reason.
    _default_assignee = (default_assignee or "").strip() or None
    _default_assignee_resolved = False
    if _default_assignee:
        try:
            from hermes_cli.profiles import profile_exists as _pe
            _default_assignee_resolved = bool(_pe(_default_assignee))
        except Exception:
            # Profiles module not importable (test stubs, exotic envs).
            # Trust the operator's config and try the assignment; the
            # downstream profile_exists check on the assigned row will
            # bucket it as nonspawnable if the profile genuinely isn't
            # there, with the existing diagnostic.
            _default_assignee_resolved = True
    for row in ready_rows:
        if ready_budget is not None and spawned >= ready_budget:
            break
        row_assignee = row["assignee"]
        if not row_assignee:
            # Honour kanban.default_assignee: when the dispatcher hits an
            # unassigned ready task and an operator-configured fallback
            # exists, persist the assignment and proceed. This removes the
            # dashboard footgun where a task created without an assignee
            # parks in 'ready' forever even though the operator's intent
            # ("default") was perfectly clear (#27145). Mutating the row
            # (not just the in-memory view) keeps diagnostics and the
            # board state consistent: the task is now legitimately owned
            # by ``kanban.default_assignee``, not "unassigned but secretly
            # routed".
            if _default_assignee and _default_assignee_resolved:
                # Dry-run: show what WOULD happen (auto-assign + spawn) without
                # mutating the DB. Real run: mutate the row + emit the
                # 'assigned' event so the board state matches what just happened.
                if not dry_run:
                    try:
                        with write_txn(conn):
                            # The default assignee is a role transition like any
                            # other, so it goes through the authority helper: a
                            # policy-locked task refuses to be adopted onto a
                            # route the owner did not approve for it.
                            role_transition_route(
                                conn, row["id"], _default_assignee
                            )
                            conn.execute(
                                "UPDATE tasks SET assignee = ?"
                                " WHERE id = ? "
                                "AND (assignee IS NULL OR assignee = '')",
                                (_default_assignee, row["id"]),
                            )
                            _append_event(
                                conn, row["id"], "assigned",
                                {
                                    "assignee": _default_assignee,
                                    "source": "kanban.default_assignee",
                                },
                            )
                    except Exception:
                        _log.debug(
                            "kanban dispatch: failed to apply default_assignee=%r "
                            "to task %s",
                            _default_assignee, row["id"], exc_info=True,
                        )
                        result.skipped_unassigned.append(row["id"])
                        continue
                row_assignee = _default_assignee
                result.auto_assigned_default.append(row["id"])
            else:
                result.skipped_unassigned.append(row["id"])
                continue
        # Skip ready tasks whose assignee is not a real Hermes profile.
        # `_default_spawn` invokes ``hermes -p <assignee>`` which fails
        # with "Profile 'X' does not exist" when the assignee names a
        # control-plane lane (e.g. an interactive Claude Code terminal
        # like ``orion-cc`` / ``orion-research``) rather than a Hermes
        # profile. Those task lanes are pulled by terminals via
        # ``claim_task`` directly and should NEVER auto-spawn — the
        # subprocess would crash on startup, get reaped as a zombie,
        # the task would loop back to ``ready`` on next tick, and we'd
        # burn CPU forever (#kanban-dispatcher-crash-loop 2026-05-05).
        try:
            from hermes_cli.profiles import profile_exists  # local import: avoids cycle
        except Exception:
            profile_exists = None  # type: ignore[assignment]
        if profile_exists is not None and not profile_exists(row_assignee):
            # Bucket separately from skipped_unassigned: the operator
            # cannot fix this by assigning a profile (the assignee IS the
            # intended owner — a terminal lane). Health telemetry uses
            # this distinction to suppress spurious "stuck" warnings on
            # multi-lane setups where the ready queue is steadily full
            # of human-pulled work.
            result.skipped_nonspawnable.append(row["id"])
            continue
        # Per-profile concurrency cap (#21582): even if there's global
        # headroom, refuse to spawn for an assignee that's already at
        # its in-flight cap. Prevents one profile's local model / API
        # quota / browser pool from being overwhelmed by a fan-out
        # while the global max_in_progress / max_spawn caps still allow
        # work on OTHER profiles.
        if _per_profile_cap is not None:
            current = _per_profile_running.get(row_assignee, 0)
            if current >= _per_profile_cap:
                result.skipped_per_profile_capped.append(
                    (row["id"], row_assignee, current)
                )
                continue
        scope_conflicts = file_scope_conflicts(conn, row["id"])
        if scope_conflicts:
            result.skipped_file_scope_conflict.append((row["id"], scope_conflicts))
            if not dry_run:
                with write_txn(conn):
                    _record_file_scope_deferral(
                        conn, row["id"], scope_conflicts, now=int(time.time())
                    )
            continue
        # Respawn guard: refuse to re-spawn when useful work is already
        # in-flight/recent, or when the last failure is a deterministic
        # blocker (quota / auth). The guard defers the spawn this tick so
        # the task gets a chance to clear (rate limits often reset in
        # seconds-to-minutes); the existing consecutive_failures counter
        # still trips the auto-block circuit breaker after failure_limit
        # consecutive failures, so a persistent auth error eventually
        # blocks via the normal path rather than on first occurrence.
        guard_reason = check_respawn_guard(conn, row["id"])
        if guard_reason is not None:
            result.respawn_guarded.append((row["id"], guard_reason))
            # Emit an event so operators can see why the task was
            # skipped when reading `hermes kanban tail` — without
            # this the task appears stuck in ready with no diagnosis.
            if not dry_run:
                with write_txn(conn):
                    _append_event(
                        conn, row["id"], "respawn_guarded",
                        {"reason": guard_reason},
                    )
            continue
        # Route authority, proved per task and BEFORE the claim. The claim
        # path re-checks it and raises; letting that raise reach here aborted
        # the whole tick over one unpinnable row.
        route_error = route_authority_error(conn, row["id"])
        if route_error is not None:
            result.skipped_route_unproven.append((row["id"], route_error))
            if not dry_run:
                with write_txn(conn):
                    authorize_executable_transition(conn, row["id"])
            continue
        if dry_run:
            result.spawned.append((row["id"], row_assignee, ""))
            spawned += 1
            # Increment per-profile counter even in dry_run so the cap
            # check sees the would-be spawn on subsequent iterations.
            # Without this, dry_run reports every task as spawnable and
            # under-reports the capped subset (#21582).
            if _per_profile_cap is not None and row_assignee:
                _per_profile_running[row_assignee] = (
                    _per_profile_running.get(row_assignee, 0) + 1
                )
            continue
        try:
            claimed = claim_task(conn, row["id"], ttl_seconds=ttl_seconds)
        except RuntimeError as exc:
            # A route the screen above could not see (a concurrent edit, a
            # schema this build cannot validate). One task's problem stays one
            # task's problem.
            result.skipped_route_unproven.append((row["id"], str(exc)))
            continue
        if claimed is None:
            continue
        try:
            resolved_branch_name = None
            if claimed.workspace_kind == "worktree":
                workspace, resolved_branch_name = _resolve_worktree_workspace(claimed, board=board)
            else:
                workspace = resolve_workspace(claimed, board=board)
        except Exception as exc:
            auto = _record_spawn_failure(
                conn, claimed.id, f"workspace: {exc}",
                failure_limit=failure_limit,
            )
            if auto:
                result.auto_blocked.append(claimed.id)
            continue
        # Persist the resolved workspace path so the worker can cd there.
        set_workspace_path(conn, claimed.id, str(workspace))
        if claimed.workspace_kind == "worktree":
            set_branch_name(conn, claimed.id, resolved_branch_name or (claimed.branch_name or "").strip() or f"wt/{claimed.id}")
            try:
                record_worktree_base(conn, claimed.id, workspace)
            except Exception as exc:
                auto = _record_spawn_failure(
                    conn, claimed.id, f"worktree base: {exc}",
                    failure_limit=failure_limit,
                )
                if auto:
                    result.auto_blocked.append(claimed.id)
                continue
        _maybe_emit_scratch_tip(conn, claimed.id, claimed.workspace_kind)
        _spawn = spawn_fn if spawn_fn is not None else _default_spawn
        try:
            # Back-compat: older spawn_fn signatures accept only
            # (task, workspace). Test stubs in the suite rely on that.
            # Introspect the callable and pass `board` only when supported.
            import inspect
            try:
                sig = inspect.signature(_spawn)
                if "board" in sig.parameters:
                    pid = _spawn(claimed, str(workspace), board=board)
                else:
                    pid = _spawn(claimed, str(workspace))
            except (TypeError, ValueError):
                pid = _spawn(claimed, str(workspace))
            if pid:
                _set_worker_pid(conn, claimed.id, int(pid))
            # Worker-lifecycle observer (RFC #58548): fires AFTER spawn_fn
            # returned and the PID (when reported) is durably persisted,
            # per the RFC timing contract. Best-effort — can never break
            # the dispatch loop.
            _fire_worker_spawned_hook(
                conn, claimed, str(workspace), pid, board=board,
            )
            # NOTE: we intentionally do NOT reset consecutive_failures
            # here. A successful spawn proves the worker can start but
            # doesn't prove the run will succeed. Under unified
            # failure counting, resetting on spawn would let a task
            # that keeps timing out after spawn loop forever. The
            # counter is cleared only on successful completion (see
            # complete_task).
            result.spawned.append((claimed.id, claimed.assignee or "", str(workspace)))
            spawned += 1
            # Track the new in-flight count for this profile so later
            # iterations in this same tick respect the per-profile cap
            # (#21582). Subsequent ticks re-query from the DB.
            if _per_profile_cap is not None and claimed.assignee:
                _per_profile_running[claimed.assignee] = (
                    _per_profile_running.get(claimed.assignee, 0) + 1
                )
        except Exception as exc:
            auto = _record_spawn_failure(
                conn, claimed.id, str(exc),
                failure_limit=failure_limit,
            )
            if auto:
                result.auto_blocked.append(claimed.id)

    # ---- review column dispatch ----
    # Review tasks are tasks that a worker moved to 'review' after
    # creating a PR.  The dispatcher spawns a review agent (loading
    # sdlc-review skill) that verifies the candidate and either approves
    # (→ done) or requests changes (→ ready/todo for the implementer).
    #
    # Same concurrency model as ready dispatch: review spawns count
    # against max_spawn alongside ready tasks, so the total number of
    # running workers stays bounded.
    # Auto-dispatch is enabled by default because Hermes bundles the
    # ``sdlc-review`` skill and reviewer workers can now approve, request
    # changes without block-loop accounting, or escalate a genuine blocker.
    # Human-only boards can disable it with ``kanban.review_dispatch``.
    #
    # ``review_rows`` was enumerated before the ready loop; when it is
    # non-empty the ready loop ran against ``ready_budget`` (one slot held
    # back) so this lane cannot be permanently starved by a sustained
    # ready backlog. The review loop itself still checks the FULL shared
    # ``spawn_budget`` — the reservation caps the ready lane, it does not
    # grant the review lane extra capacity.
    for row in review_rows:
        if spawn_budget is not None and spawned >= spawn_budget:
            break
        if not row["assignee"]:
            result.skipped_unassigned.append(row["id"])
            continue
        try:
            from hermes_cli.profiles import profile_exists
        except Exception:
            profile_exists = None  # type: ignore[assignment]
        if profile_exists is not None and not profile_exists(row["assignee"]):
            result.skipped_nonspawnable.append(row["id"])
            continue
        if _per_profile_cap is not None:
            current = _per_profile_running.get(row["assignee"], 0)
            if current >= _per_profile_cap:
                result.skipped_per_profile_capped.append(
                    (row["id"], row["assignee"], current)
                )
                continue
        scope_conflicts = file_scope_conflicts(conn, row["id"])
        if scope_conflicts:
            result.skipped_file_scope_conflict.append((row["id"], scope_conflicts))
            if not dry_run:
                with write_txn(conn):
                    _record_file_scope_deferral(
                        conn, row["id"], scope_conflicts, now=int(time.time())
                    )
            continue
        guard_reason = check_respawn_guard(conn, row["id"], lane="review")
        if guard_reason is not None:
            result.respawn_guarded.append((row["id"], guard_reason))
            if not dry_run:
                with write_txn(conn):
                    _append_event(
                        conn, row["id"], "respawn_guarded",
                        {"reason": guard_reason},
                    )
            continue
        route_error = route_authority_error(conn, row["id"])
        if route_error is not None:
            result.skipped_route_unproven.append((row["id"], route_error))
            if not dry_run:
                with write_txn(conn):
                    authorize_executable_transition(conn, row["id"])
            continue
        if dry_run:
            result.spawned.append((row["id"], row["assignee"], ""))
            spawned += 1
            if _per_profile_cap is not None:
                _per_profile_running[row["assignee"]] = (
                    _per_profile_running.get(row["assignee"], 0) + 1
                )
            continue
        try:
            claimed = claim_review_task(conn, row["id"], ttl_seconds=ttl_seconds)
        except RuntimeError as exc:
            result.skipped_route_unproven.append((row["id"], str(exc)))
            continue
        if claimed is None:
            continue
        try:
            resolved_branch_name = None
            if claimed.workspace_kind == "worktree":
                workspace, resolved_branch_name = _resolve_worktree_workspace(claimed, board=board)
            else:
                workspace = resolve_workspace(claimed, board=board)
        except Exception as exc:
            auto = _record_spawn_failure(
                conn, claimed.id, f"workspace: {exc}",
                failure_limit=failure_limit,
            )
            if auto:
                result.auto_blocked.append(claimed.id)
            continue
        # Persist the resolved workspace path so the worker can cd there.
        set_workspace_path(conn, claimed.id, str(workspace))
        if claimed.workspace_kind == "worktree":
            set_branch_name(conn, claimed.id, resolved_branch_name or (claimed.branch_name or "").strip() or f"wt/{claimed.id}")
            try:
                record_worktree_base(conn, claimed.id, workspace)
            except Exception as exc:
                auto = _record_spawn_failure(
                    conn, claimed.id, f"worktree base: {exc}",
                    failure_limit=failure_limit,
                )
                if auto:
                    result.auto_blocked.append(claimed.id)
                continue
        _maybe_emit_scratch_tip(conn, claimed.id, claimed.workspace_kind)
        # Force-load the sdlc-review skill for review agents — it carries
        # the review logic (AC verification, merge, etc.). The mandatory
        # kanban lifecycle is already injected into every worker's system
        # prompt via KANBAN_GUIDANCE, so this is the only extra skill the
        # review agent needs.
        claimed.skills = list(
            dict.fromkeys([*(claimed.skills or []), "sdlc-review"])
        )
        _spawn = spawn_fn if spawn_fn is not None else _default_spawn
        try:
            import inspect
            try:
                sig = inspect.signature(_spawn)
                if "board" in sig.parameters:
                    pid = _spawn(claimed, str(workspace), board=board)
                else:
                    pid = _spawn(claimed, str(workspace))
            except (TypeError, ValueError):
                pid = _spawn(claimed, str(workspace))
            if pid:
                _set_worker_pid(conn, claimed.id, int(pid))
            # Worker-lifecycle observer (RFC #58548): same contract as the
            # ready-lane fire above — after spawn + PID persistence.
            _fire_worker_spawned_hook(
                conn, claimed, str(workspace), pid, board=board,
            )
            result.spawned.append((claimed.id, claimed.assignee or "", str(workspace)))
            spawned += 1
            if _per_profile_cap is not None and claimed.assignee:
                _per_profile_running[claimed.assignee] = (
                    _per_profile_running.get(claimed.assignee, 0) + 1
                )
        except Exception as exc:
            auto = _record_spawn_failure(
                conn, claimed.id, str(exc),
                failure_limit=failure_limit,
            )
            if auto:
                result.auto_blocked.append(claimed.id)
    return result


def _positive_int(value: Any, default: int, *, minimum: int = 1) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed >= minimum else default


def worker_log_rotation_config(kanban_cfg: Optional[dict] = None) -> tuple[int, int]:
    """Return ``(rotate_bytes, backup_count)`` for worker log rotation.

    Defaults preserve the historical behavior: rotate at 2 MiB and keep one
    backup generation (``.log.1``). Operators with long-running workers can
    raise either value from ``config.yaml`` without changing dispatcher code.
    """
    if kanban_cfg is None:
        try:
            from hermes_cli.config import load_config

            kanban_cfg = (load_config().get("kanban") or {})
        except Exception:
            kanban_cfg = {}
    max_bytes = _positive_int(
        (kanban_cfg or {}).get("worker_log_rotate_bytes"),
        DEFAULT_LOG_ROTATE_BYTES,
        minimum=1,
    )
    backup_count = _positive_int(
        (kanban_cfg or {}).get("worker_log_backup_count"),
        DEFAULT_LOG_BACKUP_COUNT,
        minimum=0,
    )
    return max_bytes, backup_count


def _rotated_log_path(log_path: Path, generation: int) -> Path:
    return log_path.with_suffix(log_path.suffix + f".{generation}")


def _rotate_worker_log(
    log_path: Path,
    max_bytes: int,
    backup_count: int = DEFAULT_LOG_BACKUP_COUNT,
) -> None:
    """Rotate ``<log>`` when it exceeds ``max_bytes``.

    ``backup_count=1`` preserves the legacy single-generation behavior:
    ``<log>`` moves to ``<log>.1`` and any previous ``.1`` is replaced.
    Higher values shift older generations up to ``backup_count``.
    """
    try:
        if not log_path.exists():
            return
        if log_path.stat().st_size <= max_bytes:
            return
        backup_count = _positive_int(
            backup_count,
            DEFAULT_LOG_BACKUP_COUNT,
            minimum=0,
        )
        if backup_count == 0:
            log_path.unlink()
            return
        oldest = _rotated_log_path(log_path, backup_count)
        try:
            if oldest.exists():
                oldest.unlink()
        except OSError:
            pass
        for generation in range(backup_count - 1, 0, -1):
            src = _rotated_log_path(log_path, generation)
            if not src.exists():
                continue
            try:
                src.rename(_rotated_log_path(log_path, generation + 1))
            except OSError:
                pass
        log_path.rename(_rotated_log_path(log_path, 1))
    except OSError:
        pass


def _module_hermes_argv() -> list[str]:
    """Return the interpreter-bound Hermes CLI invocation."""
    # ``hermes_cli.main`` is the console-script target declared in
    # pyproject.toml, NOT a top-level ``hermes`` package — there is no
    # ``hermes`` package to import.
    return [sys.executable, "-m", "hermes_cli.main"]


def _absolute_hermes_path(path: str) -> str:
    """Return an absolute filesystem path for a resolved Hermes shim."""
    expanded = os.path.expanduser(path)
    return expanded if os.path.isabs(expanded) else os.path.abspath(expanded)


def _looks_like_path(value: str) -> bool:
    """Return true when a command override is an explicit path, not a name."""
    expanded = os.path.expanduser(value)
    return (
        expanded.startswith("~")
        or os.path.isabs(expanded)
        or bool(os.path.dirname(expanded))
        or "\\" in expanded
        or bool(re.match(r"^[A-Za-z]:", expanded))
    )


def _is_windows_batch_shim(path: str) -> bool:
    """Return true for Windows shell/batch shims that should not be argv[0]."""
    return path.lower().endswith((".cmd", ".bat"))


def _path_search_names(command: str) -> list[str]:
    """Return executable names to try for an unqualified command."""
    if not _IS_WINDOWS or os.path.splitext(command)[1]:
        return [command]
    raw = os.environ.get("PATHEXT") or ".COM;.EXE;.BAT;.CMD"
    exts = [ext for ext in raw.split(";") if ext]
    return [command + ext for ext in exts]


def _safe_which_no_cwd(command: str) -> Optional[str]:
    """Resolve a bare command from PATH without implicit current-dir search.

    ``shutil.which`` follows platform search behavior. On Windows that can
    include the current directory before PATH for bare names, which is not a
    safe dispatcher primitive. This resolver only considers explicit PATH
    entries and skips empty / ``.`` entries.
    """
    path_env = os.environ.get("PATH", "")
    for raw_dir in path_env.split(os.pathsep):
        if not raw_dir or raw_dir == ".":
            continue
        directory = os.path.expanduser(raw_dir)
        for name in _path_search_names(command):
            candidate = os.path.join(directory, name)
            if not os.path.isfile(candidate):
                continue
            if _IS_WINDOWS or os.access(candidate, os.X_OK):
                return candidate
    return None


def _hermes_path_argv(path: str) -> list[str]:
    """Return argv for a resolved Hermes executable path.

    Windows batch shims (`.cmd` / `.bat`) are not safe as argv[0] for
    worker launches because the argument vector includes task-derived
    values. Prefer the interpreter-bound module form whenever the resolved
    executable is only a shell shim.
    """
    if _IS_WINDOWS and _is_windows_batch_shim(path):
        return _module_hermes_argv()
    return [_absolute_hermes_path(path)]


def _resolve_hermes_argv() -> list[str]:
    """Resolve the ``hermes`` invocation as argv parts for ``Popen``.

    Tries in order:

    1. ``$HERMES_BIN`` — explicit operator override. Path-like values are
       normalized to absolute paths; bare command names keep normal PATH
       semantics and never prefer a same-directory file before ``PATH``.
    2. ``shutil.which("hermes")`` — the console-script shim, normalized to
       an absolute path. On Windows, ``which`` can return a relative
       ``.\\hermes.CMD`` when the current directory is on ``PATH``; directly
       launching batch shims is also unsafe with task-derived argv. The
       dispatcher therefore falls back to the interpreter-bound module form
       for implicit ``.cmd`` / ``.bat`` shims.
    3. ``sys.executable -m hermes_cli.main`` — fallback for setups where
       Hermes is launched from a venv and the ``hermes`` shim is not on
       the dispatcher's ``$PATH`` (cron, systemd ``User=`` services,
       launchd jobs, detached processes, etc.). Goes through the running
       interpreter so the result is independent of ``$PATH``.

    Mirrors ``gateway.run._resolve_hermes_bin`` for the same reason. Kept
    local (not imported from gateway) because ``hermes_cli`` sits below
    ``gateway`` in the dependency order.
    """
    import shutil

    env_bin = os.environ.get("HERMES_BIN", "").strip()
    if env_bin:
        if _looks_like_path(env_bin):
            return _hermes_path_argv(env_bin)
        resolved_env_bin = _safe_which_no_cwd(env_bin)
        if resolved_env_bin:
            return _hermes_path_argv(resolved_env_bin)
        return _module_hermes_argv()

    hermes_bin = _safe_which_no_cwd("hermes") if _IS_WINDOWS else shutil.which("hermes")
    if hermes_bin:
        return _hermes_path_argv(hermes_bin)
    return _module_hermes_argv()


def _worker_terminal_timeout_env(
    max_runtime_seconds: Optional[int],
    current_timeout: Optional[str],
) -> Optional[str]:
    """Return a worker-scoped TERMINAL_TIMEOUT override, if needed.

    Kanban's ``max_runtime_seconds`` bounds the whole worker attempt. The
    terminal tool has its own default timeout via ``TERMINAL_TIMEOUT``; when
    the worker runtime is longer, raise only the child process default so a
    long command is not killed by the generic terminal default first.
    """
    if max_runtime_seconds is None:
        return None
    try:
        runtime = int(max_runtime_seconds)
    except (TypeError, ValueError):
        return None
    if runtime <= 0:
        return None

    desired = max(1, runtime - KANBAN_TERMINAL_TIMEOUT_GRACE_SECONDS)
    try:
        existing = int(str(current_timeout).strip()) if current_timeout else 0
    except (TypeError, ValueError):
        existing = 0
    if existing >= desired:
        return None
    return str(desired)


def _resolve_worker_cli_toolsets(hermes_home: Optional[str]) -> Optional[list[str]]:
    """Return the assigned profile's effective CLI toolsets for a worker.

    Dispatcher-spawned workers are launched from a long-lived gateway process,
    then the child re-enters the CLI with ``-p <assignee>``. Resolve the
    assignee profile's CLI tool surface at dispatch time and pass it as an
    explicit ``--toolsets`` pin so worker startup cannot fall back to a stale
    root/active-profile config or a profile whose top-level ``toolsets`` entry
    is only the kanban orchestrator surface. ``model_tools`` still appends the
    task-scoped kanban lifecycle tools when ``HERMES_KANBAN_TASK`` is set.
    """
    if not hermes_home:
        return None
    try:
        from hermes_constants import reset_hermes_home_override, set_hermes_home_override
        from hermes_cli.config import load_config
        from hermes_cli.tools_config import _get_platform_tools

        token = set_hermes_home_override(hermes_home)
        try:
            cfg = load_config()
            toolsets = sorted(_get_platform_tools(cfg, "cli"))
        finally:
            reset_hermes_home_override(token)
        return toolsets or None
    except Exception as exc:
        _log.debug(
            "kanban worker: could not resolve CLI toolsets for HERMES_HOME=%r (%s)",
            hermes_home,
            exc,
        )
        return None


_retagged_workspace_roots: set[str] = set()


def _retag_legacy_worker_sessions(workspaces_root_path: str) -> None:
    """Reclaim pre-tag worker rows in state.db so they leave the session lists.

    Best-effort and gated — the durable ``state_meta`` gate lives in
    ``retag_kanban_worker_sessions``; the in-process set keeps a busy
    dispatcher from reopening state.db on every spawn just to read it. A
    dispatcher tick must never fail because a session DB was busy or missing.
    """
    if workspaces_root_path in _retagged_workspace_roots:
        return
    try:
        from hermes_state import SessionDB

        db = SessionDB()
        try:
            db.retag_kanban_worker_sessions(workspaces_root_path)
        finally:
            db.close()
        _retagged_workspace_roots.add(workspaces_root_path)
    except Exception as exc:
        _log.debug("kanban worker: legacy session retag skipped (%s)", exc)


def _default_spawn(
    task: Task,
    workspace: str,
    *,
    board: Optional[str] = None,
) -> Optional[int]:
    """Fire-and-forget ``hermes -p <profile> chat -q ...`` subprocess.

    Returns the spawned child's PID so the dispatcher can detect crashes
    before the claim TTL expires. The child's completion is still observed
    via the ``complete`` / ``block`` transitions the worker writes itself;
    the PID check is a safety net for crashes, OOM kills, and Ctrl+C.

    ``board`` pins the child's kanban context to that board: the child's
    ``HERMES_KANBAN_DB`` / ``HERMES_KANBAN_BOARD`` / workspaces_root env
    vars all resolve to the same board the dispatcher claimed the task
    from. Workers cannot accidentally see other boards.
    """
    import subprocess
    if not task.assignee:
        raise ValueError(f"task {task.id} has no assignee")

    # A policy-locked task may only run on the exact authority the owner
    # approved for it: this assignee, provider, model, effort and tier, under a
    # lock version this build still mints. Anything else — a corrupt or foreign
    # authority string, a stale version, an incomplete row, a route the policy
    # no longer admits, or a single hand-edited column — refuses the spawn
    # rather than letting the worker resolve anything from the profile. The
    # failure is recorded on the task and the circuit breaker pauses it, so the
    # board shows why.
    if task.model_policy_lock:
        route_error = policy_lock_error(
            task.model_policy_lock,
            task.assignee,
            task.provider_override,
            task.model_override,
            task.reasoning_effort,
            task.execution_tier,
        )
        if route_error:
            raise RuntimeError(f"task {task.id}: {route_error}")

    from hermes_cli.profiles import normalize_profile_name

    profile_arg = normalize_profile_name(task.assignee)

    prompt = f"work kanban task {task.id}"
    env = dict(os.environ)
    # The dispatcher is detached from every conversation. Its worker must never
    # inherit routing mirrored by a previous gateway turn, even before the first
    # session binds ContextVars in this process.
    from gateway.session_context import _VAR_MAP
    for key in _VAR_MAP:
        env.pop(key, None)

    # Inject HERMES_HOME so the worker reads the profile-scoped config.yaml
    # (fallback_providers, toolsets, agent settings, etc.) instead of the root
    # config.  Without this, `env = dict(os.environ)` copies only the parent's
    # env, and when the child process starts `hermes -p <name>` the
    # _apply_profile_override() runs *before* hermes_constants is imported.
    # If HERMES_HOME is absent from the child's env, get_hermes_home() falls
    # back to Path.home() / ".hermes" (the DEFAULT profile root), ignoring the
    # profile-specific config entirely.  Fixes profile-scoped fallback_providers
    # being invisible to kanban workers.
    from hermes_cli.profiles import resolve_profile_env
    try:
        env["HERMES_HOME"] = resolve_profile_env(profile_arg)
    except FileNotFoundError:
        # Profile dir doesn't exist — defer resolution to the CLI's
        # _apply_profile_override() via HERMES_PROFILE (set below).
        # This only happens in test fixtures where the isolated
        # HERMES_HOME never had profiles created.
        pass
    if task.tenant:
        env["HERMES_TENANT"] = task.tenant
    env["HERMES_KANBAN_TASK"] = task.id
    env["HERMES_KANBAN_WORKSPACE"] = workspace
    # Tag the worker's session so it lands in state.db as `kanban`, not as an
    # untitled `cli` row. A worker is a dispatcher-owned run whose transcript is
    # read on the board and in `hermes kanban log` — it is not a conversation
    # the user started, so every session-browsing surface (desktop sidebar, TUI
    # resume picker, session_search) filters it out by source. Without this the
    # sidebar renders one row per attempt, labeled with the worker's own prompt
    # ("work kanban task t_…").
    env["HERMES_SESSION_SOURCE"] = "kanban"
    # Pin TERMINAL_CWD to the task's workspace so the worker's file tools and
    # context-file loader anchor on the workspace, not whatever cwd the
    # dispatching gateway happened to export. The worker subprocess is already
    # launched with cwd=workspace, but TERMINAL_CWD takes precedence over the
    # process cwd in both file_tools._resolve_base_dir (#41312 — relative
    # write_file paths were landing in the gateway user's home) and
    # build_context_files_prompt (#34619 — workers loaded the dispatching
    # gateway's AGENTS.md instead of the task's). Setting it to the workspace
    # fixes both: the workspace is where the task's work actually happens.
    # Only pin a real, absolute directory — file_tools rejects relative /
    # sentinel TERMINAL_CWD values, so a non-dir workspace must NOT be set
    # here (leave the inherited value rather than write a meaningless one).
    if workspace and os.path.isabs(workspace) and os.path.isdir(workspace):
        env["TERMINAL_CWD"] = workspace
    if task.branch_name:
        env["HERMES_KANBAN_BRANCH"] = task.branch_name
    if task.current_run_id is not None:
        env["HERMES_KANBAN_RUN_ID"] = str(task.current_run_id)
    if task.claim_lock:
        env["HERMES_KANBAN_CLAIM_LOCK"] = task.claim_lock
    # Goal-loop mode: the worker reads these and wraps its run in the
    # Ralph-style /goal judge loop (see cli.py quiet-mode path). Only set
    # when enabled so non-goal tasks keep a clean env.
    if task.goal_mode:
        env["HERMES_KANBAN_GOAL_MODE"] = "1"
        if task.goal_max_turns is not None:
            env["HERMES_KANBAN_GOAL_MAX_TURNS"] = str(int(task.goal_max_turns))
    # Owner-approved route: disable the fallback chain for this worker. The env
    # var is set so anything the worker itself spawns inherits the policy, but
    # it is NOT the authority — the profile's own ``.env`` is loaded with
    # override=True during startup and could reset it. The authority is the
    # ``--no-fallbacks`` flag added to the worker's argv below, which the CLI
    # latches into process state after all dotenv loading.
    if task.model_policy_lock:
        from hermes_cli.fallback_config import FALLBACKS_DISABLED_ENV

        env[FALLBACKS_DISABLED_ENV] = "1"
    terminal_timeout = _worker_terminal_timeout_env(
        task.max_runtime_seconds,
        env.get("TERMINAL_TIMEOUT"),
    )
    if terminal_timeout is not None:
        env["TERMINAL_TIMEOUT"] = terminal_timeout
    foreground_timeout = _worker_terminal_timeout_env(
        task.max_runtime_seconds,
        env.get("TERMINAL_MAX_FOREGROUND_TIMEOUT"),
    )
    if foreground_timeout is not None:
        env["TERMINAL_MAX_FOREGROUND_TIMEOUT"] = foreground_timeout
    # Pin the shared board + workspaces root the dispatcher resolved, so
    # that even when the worker activates a profile (`hermes -p <name>`
    # rewrites HERMES_HOME), its kanban paths still match the
    # dispatcher's. Belt-and-braces with the `get_default_hermes_root()`
    # resolution in `kanban_home()` — symmetric resolution is the norm,
    # but unusual symlink / Docker layouts are caught here too.
    env["HERMES_KANBAN_DB"] = str(kanban_db_path(board=board))
    env["HERMES_KANBAN_WORKSPACES_ROOT"] = str(workspaces_root(board=board))
    _retag_legacy_worker_sessions(env["HERMES_KANBAN_WORKSPACES_ROOT"])
    # Board slug — the final defense-in-depth pin. If the worker ever
    # resolves kanban paths without the DB / workspaces env vars, the
    # board slug still forces it to the right directory.
    resolved_board = _normalize_board_slug(board) or get_current_board()
    env["HERMES_KANBAN_BOARD"] = resolved_board
    # HERMES_PROFILE is the author the kanban_comment tool defaults to.
    # `hermes -p <assignee>` activates the profile, but the env var is
    # what the tool reads — set it explicitly here so comments are
    # attributed correctly regardless of how the child loads config.
    env["HERMES_PROFILE"] = profile_arg

    # A worker must NEVER boot the interactive TUI: an inherited HERMES_TUI=1
    # or a `display.interface: tui` in the profile's config would send the
    # quiet chat run into the Ink TUI, whose no-TTY bail-out exits 0 without
    # doing the task → "protocol violation" on every attempt. `--cli` is the
    # highest-precedence interface override; dropping the env var covers
    # older hermes builds on PATH that predate the flag's precedence.
    env.pop("HERMES_TUI", None)

    cmd = [
        *_resolve_hermes_argv(),
        "-p", profile_arg,
        "--cli",
        # Worker subprocesses switch to a profile-scoped HERMES_HOME above,
        # so they see that profile's shell-hook allowlist instead of the
        # dispatcher's root allowlist. Pass --accept-hooks explicitly so
        # profile-local worker sessions still register configured hooks.
        "--accept-hooks",
    ]
    if task.model_policy_lock:
        # The non-user-overridable channel for this task's no-fallback
        # authority: argv, latched after dotenv, so neither the profile's .env
        # nor its config.yaml can restore a fallback chain under a pinned run.
        from hermes_cli.fallback_config import NO_FALLBACK_FLAG

        cmd.append(NO_FALLBACK_FLAG)
    # Per-task force-loaded skills. Each name goes in its own
    # `--skills X` pair rather than a single comma-joined arg: the CLI
    # accepts both forms (action='append' + comma-split), but
    # per-name pairs are easier to read in `ps` output and avoid any
    # quoting ambiguity if a skill name ever contains unusual chars.
    if task.skills:
        for sk in task.skills:
            if sk:
                cmd.extend(["--skills", sk])
    if task.model_override:
        cmd.extend(["-m", task.model_override])
        # Pin the provider too when the override names one, so the worker
        # resolves the model against the intended backend instead of the
        # profile's configured provider (mixing model X with provider Y is
        # the classic mis-set that stalls a board).
        if task.provider_override:
            cmd.extend(["--provider", task.provider_override])
    # Per-task thinking depth. Independent of the model override — a task can
    # run the profile's own model at a different depth — so this is its own
    # branch, not a nested one.
    if task.reasoning_effort:
        cmd.extend(["--reasoning", task.reasoning_effort])
    worker_toolsets = _resolve_worker_cli_toolsets(env.get("HERMES_HOME"))
    if worker_toolsets:
        cmd.extend(["--toolsets", ",".join(worker_toolsets)])
    cmd.extend([
        "chat",
        "-q", prompt,
    ])
    if task.goal_mode:
        # Goal-mode workers must take the fully-quiet single-query path:
        # the kanban goal-loop hook (_run_kanban_goal_loop_q) only runs in
        # cli.py's quiet branch. Without -Q the worker gets exactly one
        # turn, prints text, exits rc=0, and the dispatcher records a
        # protocol violation (incident 2026-06-09 t_d9cbe312).
        cmd.append("-Q")
    # Redirect output to a per-task log under <board-root>/logs/.
    # Anchored at the board root (not the shared kanban root), so
    # `hermes kanban log` on a specific board reads its own file and
    # logs don't collide across boards that happen to share task ids.
    log_dir = worker_logs_dir(board=board)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{task.id}.log"
    rotate_bytes, backup_count = worker_log_rotation_config()
    _rotate_worker_log(log_path, rotate_bytes, backup_count)

    # Use 'a' so a re-run on unblock appends rather than overwrites.
    log_f = open(log_path, "ab")
    try:
        proc = subprocess.Popen(  # noqa: S603 -- argv is a fixed list built above
            cmd,
            cwd=workspace if os.path.isdir(workspace) else None,
            stdin=subprocess.DEVNULL,
            stdout=log_f,
            stderr=subprocess.STDOUT,
            env=env,
            start_new_session=True,
            creationflags=subprocess.CREATE_NO_WINDOW if _IS_WINDOWS else 0,
        )
    except FileNotFoundError:
        log_f.close()
        raise RuntimeError(
            "`hermes` executable not found on PATH. "
            "Install Hermes Agent or activate its venv before running the kanban dispatcher."
        )
    # NOTE: we intentionally do NOT close log_f here — we want Popen's
    # child process to keep writing after this function returns.  The
    # handle is kept alive by the child's inheritance.  The parent's
    # reference goes out of scope and is GC'd, but the OS-level FD stays
    # open in the child until the child exits.
    return proc.pid


# ---------------------------------------------------------------------------
# Long-lived dispatcher daemon
# ---------------------------------------------------------------------------

def run_daemon(
    *,
    interval: float = 60.0,
    max_spawn: Optional[int] = None,
    failure_limit: int = DEFAULT_SPAWN_FAILURE_LIMIT,
    stop_event=None,
    on_tick=None,
) -> None:
    """Run the dispatcher in a loop until interrupted.

    Calls :func:`dispatch_once` every ``interval`` seconds. Exits cleanly
    on SIGINT / SIGTERM so ``hermes kanban daemon`` is systemd-friendly.
    ``stop_event`` (a :class:`threading.Event`) and ``on_tick`` (a
    callable receiving the :class:`DispatchResult`) are test hooks.

    Each tick resolves ``kanban.max_in_progress`` (explicit config, else
    the memory-derived default) exactly like the gateway-embedded
    dispatcher and ``hermes kanban dispatch`` — the standalone daemon must
    not be the one uncapped entry point (OOF-30).
    """
    import signal
    import threading

    if stop_event is None:
        stop_event = threading.Event()

    def _handle(_signum, _frame):
        stop_event.set()

    # Install handlers only when running on the main thread — tests call
    # this inline from worker threads and signal() would raise there.
    if threading.current_thread() is threading.main_thread():
        for sig_name in ("SIGINT", "SIGTERM"):
            sig = getattr(signal, sig_name, None)
            if sig is not None:
                try:
                    signal.signal(sig, _handle)
                except (ValueError, OSError):
                    pass

    while not stop_event.is_set():
        try:
            # Resolve the global concurrency cap the same way the gateway
            # dispatcher and `hermes kanban dispatch` do (OOF-30): explicit
            # kanban.max_in_progress wins, otherwise the memory-derived
            # default applies. The standalone daemon previously passed no
            # cap at all — the shipped systemd path could still fan out an
            # entire backlog in one tick even with the derived default in
            # place everywhere else. Re-resolved every tick (config load is
            # mtime-cached) so operator edits apply without a restart.
            max_in_progress = resolve_max_in_progress(
                configured_max_in_progress()
            )
            with contextlib.closing(connect()) as conn:
                res = dispatch_once(
                    conn,
                    max_spawn=max_spawn,
                    max_in_progress=max_in_progress,
                    failure_limit=failure_limit,
                )
            if on_tick is not None:
                try:
                    on_tick(res)
                except Exception:
                    pass
        except Exception:
            # Don't let any single tick kill the daemon.
            import traceback
            traceback.print_exc()
        stop_event.wait(timeout=interval)


# ---------------------------------------------------------------------------
# Worker context builder (what a spawned worker sees)
# ---------------------------------------------------------------------------

def build_worker_context(conn: sqlite3.Connection, task_id: str) -> str:
    """Return the full text a worker should read to understand its task.

    Order:
      1. Task title (mandatory).
      2. Task body (optional opening post, capped at 8 KB).
      3. Prior attempts on THIS task (most recent ``_CTX_MAX_PRIOR_ATTEMPTS``
         shown; older attempts collapsed into a one-line summary).
         Each attempt's ``summary`` / ``error`` / ``metadata`` capped at
         ``_CTX_MAX_FIELD_BYTES`` each.
      4. Structured handoff results of every done parent task. Prefers
         ``run.summary`` / ``run.metadata`` when the parent was executed
         via a run; falls back to ``task.result`` for older data. Same
         per-field cap.
      5. Cross-task role history for the assignee (most recent 5
         completed runs on other tasks).
      6. Comment thread (most recent ``_CTX_MAX_COMMENTS`` shown, older
         collapsed).

    All caps exist so worker prompts stay bounded even on pathological
    boards (retry-heavy tasks, comment storms). The per-field char cap
    prevents a single 1 MB summary from dominating context.
    """
    task = get_task(conn, task_id)
    if not task:
        raise ValueError(f"unknown task {task_id}")

    # Single clock reading shared by every relative-age stamp below, so all
    # ages in one rendering are consistent ("3h ago" / "3h ago", not drifting
    # by the seconds it takes to build the block).
    _now = int(time.time())

    def _cap(s: Optional[str], limit: int = _CTX_MAX_FIELD_BYTES) -> str:
        """Truncate a string to `limit` chars with a visible ellipsis."""
        if not s:
            return ""
        s = s.strip()
        if len(s) <= limit:
            return s
        return s[:limit] + f"… [truncated, {len(s) - limit} chars omitted]"

    lines: list[str] = []
    lines.append(f"# Kanban task {task.id}: {task.title}")
    lines.append("")
    lines.append(f"Assignee: {task.assignee or '(unassigned)'}")
    lines.append(f"Status:   {task.status}")
    if task.tenant:
        lines.append(f"Tenant:   {task.tenant}")
    lines.append(f"Workspace: {task.workspace_kind} @ {task.workspace_path or '(unresolved)'}")
    if task.max_runtime_seconds is not None:
        terminal_timeout = _worker_terminal_timeout_env(
            task.max_runtime_seconds,
            os.environ.get("TERMINAL_TIMEOUT"),
        )
        effective_terminal_timeout = terminal_timeout or os.environ.get("TERMINAL_TIMEOUT")
        lines.append(f"Max runtime: {task.max_runtime_seconds}s")
        if effective_terminal_timeout:
            lines.append(f"Terminal timeout: {effective_terminal_timeout}s")
    if task.branch_name:
        lines.append(f"Branch:   {task.branch_name}")
    if task.owned_paths is None:
        lines.append("Repository ownership: legacy exclusive whole repository")
    elif not task.owned_paths:
        lines.append("Repository ownership: read-only (no repository changes allowed)")
    else:
        lines.append("Repository ownership: " + ", ".join(task.owned_paths))
    if task.integrates_parent_heads:
        lines.append("Integration contract: include every same-Project parent git head")
    if task.base_commit:
        lines.append(f"Base commit: {task.base_commit}")
    lines.append("")

    if task.body and task.body.strip():
        lines.append("## Body")
        lines.append(_cap(task.body, _CTX_MAX_BODY_BYTES))
        lines.append("")

    # Attachments — files uploaded to this task (PDFs, source docs,
    # images). Surface the absolute on-disk path so the worker, which has
    # full file-tool access, can read them directly (read_file, terminal
    # `pdftotext`, etc.). On the local terminal backend the path resolves
    # as-is; remote backends need the kanban attachments dir mounted.
    attachments = list_attachments(conn, task_id)
    if attachments:
        lines.append("## Attachments")
        lines.append(
            "Files attached to this task. Read them with the file/terminal "
            "tools at the absolute paths below:"
        )
        for att in attachments:
            size_kb = max(1, (att.size + 1023) // 1024) if att.size else 0
            size_str = f", {size_kb} KB" if size_kb else ""
            ctype = f", {att.content_type}" if att.content_type else ""
            lines.append(f"- `{att.filename}`{ctype}{size_str} → `{att.stored_path}`")
        lines.append("")

    # Prior attempts — show closed runs so a retrying worker sees the
    # history. Skip the currently-active run (that's this worker).
    # Cap at _CTX_MAX_PRIOR_ATTEMPTS most-recent closed runs; older
    # attempts get collapsed into a one-line marker so the worker knows
    # more exist without bloating the prompt.
    all_prior = [r for r in list_runs(conn, task_id) if r.ended_at is not None]
    # list_runs returns ascending by started_at; "most recent" = last N
    if len(all_prior) > _CTX_MAX_PRIOR_ATTEMPTS:
        omitted = len(all_prior) - _CTX_MAX_PRIOR_ATTEMPTS
        shown = all_prior[-_CTX_MAX_PRIOR_ATTEMPTS:]
        first_shown_idx = omitted + 1
    else:
        omitted = 0
        shown = all_prior
        first_shown_idx = 1
    if shown:
        lines.append("## Prior attempts on this task")
        if omitted:
            lines.append(
                f"_({omitted} earlier attempt{'s' if omitted != 1 else ''} "
                f"omitted; showing most recent {len(shown)})_"
            )
        for offset, run in enumerate(shown):
            idx = first_shown_idx + offset
            ts = time.strftime("%Y-%m-%d %H:%M", time.localtime(run.started_at))
            age = _relative_age(run.started_at, _now)
            ts_disp = f"{ts}, {age}" if age else ts
            profile = run.profile or "(unknown)"
            outcome = run.outcome or run.status
            lines.append(f"### Attempt {idx} — {outcome} ({profile}, {ts_disp})")
            if run.summary and run.summary.strip():
                lines.append(_cap(run.summary))
            if run.error and run.error.strip():
                lines.append(f"_error_: {_cap(run.error)}")
            if run.metadata:
                try:
                    meta_str = json.dumps(run.metadata, ensure_ascii=False, sort_keys=True)
                    lines.append(f"_metadata_: `{_cap(meta_str)}`")
                except Exception:
                    pass
            lines.append("")

    # Parents: prefer the most-recent 'completed' run's summary + metadata,
    # fall back to ``task.result`` when no run rows exist (legacy DBs,
    # or tasks completed before the runs table landed).
    parent_rows = conn.execute(
        "SELECT parent_id FROM task_links WHERE child_id = ? ORDER BY parent_id",
        (task_id,),
    ).fetchall()
    parent_ids = [r["parent_id"] for r in parent_rows]

    if parent_ids:
        wrote_header = False
        for pid in parent_ids:
            pt = get_task(conn, pid)
            if not pt or pt.status != "done":
                continue
            runs = [r for r in list_runs(conn, pid) if r.outcome == "completed"]
            runs.sort(key=lambda r: r.started_at, reverse=True)
            run = runs[0] if runs else None

            if not wrote_header:
                lines.append("## Parent task results")
                lines.append(
                    "_Handoffs from upstream tasks, captured when each parent "
                    "completed (see age below). These are point-in-time "
                    "snapshots, not live state — if a result drives your "
                    "current work and it's not recent, re-verify against the "
                    "source before acting on it as current._"
                )
                wrote_header = True

            # When did this parent's result get produced? Prefer the
            # completed run's end time; fall back to the task's completed_at.
            done_ts = None
            if run is not None and getattr(run, "ended_at", None):
                done_ts = run.ended_at
            elif pt.completed_at:
                done_ts = pt.completed_at
            age = _relative_age(done_ts, _now)
            lines.append(f"### {pid}" + (f" (completed {age})" if age else ""))
            if pt.base_commit or pt.head_commit:
                lines.append(
                    "_git receipt_: "
                    f"base={pt.base_commit or '(missing)'}; "
                    f"head={pt.head_commit or '(missing)'}; "
                    f"branch={pt.branch_name or '(missing)'}"
                )

            body_lines: list[str] = []
            if run is not None and run.summary and run.summary.strip():
                body_lines.append(_cap(run.summary))
            elif pt.result:
                body_lines.append(_cap(pt.result))
            else:
                body_lines.append("(no result recorded)")

            if run is not None and run.metadata:
                try:
                    meta_str = json.dumps(run.metadata, ensure_ascii=False, sort_keys=True)
                    body_lines.append(f"_metadata_: `{_cap(meta_str)}`")
                except Exception:
                    pass
            lines.extend(body_lines)
            lines.append("")

    # Cross-task role history: what else has THIS assignee completed
    # recently? Gives the worker implicit continuity — "I'm the reviewer
    # and my last three reviews focused on security" — without forcing
    # the user to wire anything into SOUL.md / MEMORY.md. Bounded to the
    # most recent 5 completed runs, excluding this task so the retry
    # section above isn't duplicated. Safe on assignee=None (skipped).
    if task.assignee:
        role_rows = conn.execute(
            "SELECT t.id, t.title, r.summary, r.ended_at "
            "FROM task_runs r JOIN tasks t ON r.task_id = t.id "
            "WHERE r.profile = ? AND r.task_id != ? "
            "  AND r.outcome = 'completed' AND t.task_kind = 'work' "
            "ORDER BY r.ended_at DESC LIMIT 5",
            (task.assignee, task_id),
        ).fetchall()
        if role_rows:
            lines.append(f"## Recent work by @{task.assignee}")
            for row in role_rows:
                ts = time.strftime(
                    "%Y-%m-%d %H:%M", time.localtime(int(row["ended_at"]))
                )
                age = _relative_age(row["ended_at"], _now)
                ts_disp = f"{ts}, {age}" if age else ts
                s = (row["summary"] or "").strip().splitlines()
                first = s[0][:200] if s else "(no summary)"
                lines.append(f"- {row['id']} — {row['title']} ({ts_disp}): {first}")
            lines.append("")

    # Comments: cap at the most-recent _CTX_MAX_COMMENTS so
    # comment-storm tasks don't blow out the worker's prompt. Older
    # comments summarised in a one-line marker like prior attempts.
    all_comments = list_comments(conn, task_id)
    if len(all_comments) > _CTX_MAX_COMMENTS:
        omitted_c = len(all_comments) - _CTX_MAX_COMMENTS
        shown_c = all_comments[-_CTX_MAX_COMMENTS:]
    else:
        omitted_c = 0
        shown_c = all_comments
    if shown_c:
        lines.append("## Comment thread")
        if omitted_c:
            lines.append(
                f"_({omitted_c} earlier comment{'s' if omitted_c != 1 else ''} "
                f"omitted; showing most recent {len(shown_c)})_"
            )
        for c in shown_c:
            ts = time.strftime("%Y-%m-%d %H:%M", time.localtime(c.created_at))
            age = _relative_age(c.created_at, _now)
            ts_disp = f"{ts}, {age}" if age else ts
            # Render author with explicit "comment from worker" framing so
            # operator-controlled HERMES_PROFILE values like "hermes-system"
            # or "operator" can't be misread by the next worker as a system
            # directive above the (attacker-influenceable) comment body.
            # Defense-in-depth — the LLM-controlled author-forgery surface
            # was already closed in #22435. See #22452.
            safe_author = (c.author or "").replace("`", "")
            lines.append(f"comment from worker `{safe_author}` at {ts_disp}:")
            lines.append(_cap(c.body, _CTX_MAX_COMMENT_BYTES))
            lines.append("")

    return "\n".join(lines).rstrip() + "\n"


# ---------------------------------------------------------------------------
# Stats + SLA helpers
# ---------------------------------------------------------------------------

def board_stats(conn: sqlite3.Connection) -> dict:
    """Per-status + per-assignee counts, plus the oldest ``ready`` age in
    seconds (the clearest staleness signal for a router or HUD).
    """
    by_status: dict[str, int] = {}
    for row in conn.execute(
        "SELECT status, COUNT(*) AS n FROM tasks "
        "WHERE status != 'archived' AND task_kind = 'work' GROUP BY status"
    ):
        by_status[row["status"]] = int(row["n"])

    by_assignee: dict[str, dict[str, int]] = {}
    for row in conn.execute(
        "SELECT assignee, status, COUNT(*) AS n FROM tasks "
        "WHERE status != 'archived' AND assignee IS NOT NULL AND task_kind = 'work' "
        "GROUP BY assignee, status"
    ):
        by_assignee.setdefault(row["assignee"], {})[row["status"]] = int(row["n"])

    oldest_row = conn.execute(
        "SELECT MIN(created_at) AS ts FROM tasks "
        "WHERE status = 'ready' AND task_kind = 'work'"
    ).fetchone()
    now = int(time.time())
    oldest_ready_age = (
        (now - int(oldest_row["ts"]))
        if oldest_row and oldest_row["ts"] is not None else None
    )

    return {
        "by_status": by_status,
        "by_assignee": by_assignee,
        "oldest_ready_age_seconds": oldest_ready_age,
        "now": now,
    }


def _to_epoch(val) -> Optional[int]:
    """Normalise a timestamp to unix epoch seconds.

    Accepts ints (pass-through), numeric strings, and ISO-8601 strings.
    Returns ``None`` for ``None`` / empty values.
    """
    if val is None:
        return None
    if isinstance(val, int):
        return val
    if isinstance(val, float):
        return int(val)
    s = str(val).strip()
    if not s:
        return None
    try:
        return int(s)
    except ValueError:
        pass
    # ISO-8601 fallback (e.g. '2026-05-10T15:00:00Z')
    try:
        from datetime import datetime
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
        return int(dt.timestamp())
    except (ValueError, OSError):
        return None


def task_age(task: Task) -> dict:
    """Return age metrics for a single task. All values are seconds or None."""
    now = int(time.time())
    _c = _to_epoch(task.created_at)
    _s = _to_epoch(task.started_at)
    _co = _to_epoch(task.completed_at)
    age_since_created = now - _c if _c is not None else None
    age_since_started = now - _s if _s is not None else None
    time_to_complete = (
        _co - (_s or _c) if _co is not None else None
    )
    return {
        "created_age_seconds": age_since_created,
        "started_age_seconds": age_since_started,
        "time_to_complete_seconds": time_to_complete,
    }


# ---------------------------------------------------------------------------
# Notification subscriptions (used by the gateway kanban-notifier)
# ---------------------------------------------------------------------------

# How the gateway kanban-notifier reacts to a terminal event for a
# subscription:
#   "notify"       -> passive ``adapter.send`` only (default)
#   "notify+wake"  -> passive send AND wake the destination gateway agent
#   "wake"         -> wake the agent only; no passive message is sent
_NOTIFY_DELIVERY_MODES = ("notify", "notify+wake", "wake")


def _encode_notify_delivery_metadata(
    metadata: Optional[Mapping[str, Any]],
) -> Optional[str]:
    """Serialize platform send metadata stored on notification subscriptions."""
    if not isinstance(metadata, Mapping):
        return None
    clean: dict[str, Any] = {}
    for key, value in metadata.items():
        if value is None:
            continue
        if isinstance(value, (str, int, float, bool)):
            clean[str(key)] = value
    if not clean:
        return None
    return json.dumps(clean, sort_keys=True, separators=(",", ":"))


def _decode_notify_delivery_metadata(raw: Any) -> dict[str, Any]:
    if isinstance(raw, Mapping):
        return dict(raw)
    if not raw:
        return {}
    try:
        data = json.loads(str(raw))
    except Exception:
        return {}
    if not isinstance(data, dict):
        return {}
    return {
        str(key): value
        for key, value in data.items()
        if isinstance(value, (str, int, float, bool))
    }


def add_notify_sub(
    conn: sqlite3.Connection,
    *,
    task_id: str,
    platform: str,
    chat_id: str,
    thread_id: Optional[str] = None,
    user_id: Optional[str] = None,
    user_id_alt: Optional[str] = None,
    chat_type: Optional[str] = None,
    notifier_profile: Optional[str] = None,
    delivery_mode: Optional[str] = None,
    delivery_metadata: Optional[Mapping[str, Any]] = None,
) -> None:
    """Register a gateway source that wants terminal-state notifications
    for ``task_id``. Idempotent on (task, platform, chat, thread).

    ``user_id_alt`` records the originating source's platform-specific stable
    alt ID (Signal UUID, Feishu union_id, ...) alongside ``user_id``. Active-wake
    replay must reproduce it so the woken turn's ``build_session_key`` matches
    the original event's — ``build_session_key`` prefers ``user_id_alt`` over
    ``user_id`` (gateway/session.py), so replaying only ``user_id`` would key a
    wake into a different session whenever the two diverge for this source.

    ``chat_type`` records the originating source's chat type; the active-wake
    delivery modes replay it so the woken turn resolves the operator's real
    channel. ``None`` keeps an existing row's value.

    ``delivery_mode`` (see ``_NOTIFY_DELIVERY_MODES``) selects how the
    kanban-notifier reacts to a terminal event for this subscription. ``None``
    leaves an existing row's mode untouched (and inserts the ``"notify"``
    default for a fresh row); an explicit value is last-write-wins, so an
    operator can intentionally re-subscribe to change the mode (e.g.
    ``notify`` -> ``wake``). An unknown value falls back to ``"notify"``.
    New subscriptions start "caught up": ``last_event_id`` snaps to the
    task's current ``MAX(task_events.id)`` at creation instead of the
    schema default 0. A cursor of 0 on an already-active task made the
    gateway notifier replay every historical terminal event on its next
    tick — and with many stale subs, a single boot-time burst of 100+
    messages (issue #29905). Subscribers only want events that occur
    AFTER they subscribe; the gateway/tool auto-subscribe paths run at
    task creation, where the snapshot is 0 anyway.
    """
    insert_mode = delivery_mode if delivery_mode in _NOTIFY_DELIVERY_MODES else (
        # api_server is stateless: the adapter has no send() — the wake
        # self-post IS the delivery on that path (see gateway/wake.py and
        # test_kanban_notifier_apiserver_wake). A plain-'notify' default
        # would leave those subscriptions with no delivery mechanism at
        # all, regressing the pre-delivery_mode behavior where a task
        # carrying a session_id always woke. Explicit modes still win.
        "notify+wake" if platform == "api_server" else "notify"
    )
    insert_chat_type = chat_type or "dm"
    now = int(time.time())
    metadata_json = _encode_notify_delivery_metadata(delivery_metadata)
    with write_txn(conn):
        conn.execute(
            """
            INSERT OR IGNORE INTO kanban_notify_subs
                (task_id, platform, chat_id, thread_id, user_id, user_id_alt,
                 chat_type, notifier_profile, delivery_mode, delivery_metadata,
                 created_at, last_event_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                    COALESCE((SELECT MAX(id) FROM task_events WHERE task_id = ?), 0))
            """,
            (
                task_id,
                platform,
                chat_id,
                thread_id or "",
                user_id,
                user_id_alt,
                insert_chat_type,
                notifier_profile,
                insert_mode,
                metadata_json,
                now,
                task_id,
            ),
        )
        if chat_type:
            # Explicit chat_type is last-write-wins on re-subscribe.
            conn.execute(
                """
                UPDATE kanban_notify_subs
                   SET chat_type = ?
                 WHERE task_id = ? AND platform = ? AND chat_id = ? AND thread_id = ?
                """,
                (chat_type, task_id, platform, chat_id, thread_id or ""),
            )
        if user_id_alt:
            # Self-heal legacy rows created before alternate IDs were tracked.
            conn.execute(
                """
                UPDATE kanban_notify_subs
                   SET user_id_alt = ?
                 WHERE task_id = ? AND platform = ? AND chat_id = ? AND thread_id = ?
                   AND (user_id_alt IS NULL OR user_id_alt = '')
                """,
                (user_id_alt, task_id, platform, chat_id, thread_id or ""),
            )
        if notifier_profile:
            # Self-heal legacy rows that predate notifier ownership.
            conn.execute(
                """
                UPDATE kanban_notify_subs
                   SET notifier_profile = ?
                 WHERE task_id = ? AND platform = ? AND chat_id = ? AND thread_id = ?
                   AND (notifier_profile IS NULL OR notifier_profile = '')
                """,
                (notifier_profile, task_id, platform, chat_id, thread_id or ""),
            )
        if delivery_mode in _NOTIFY_DELIVERY_MODES:
            # Explicit delivery_mode is last-write-wins on re-subscribe.
            conn.execute(
                """
                UPDATE kanban_notify_subs
                   SET delivery_mode = ?
                 WHERE task_id = ? AND platform = ? AND chat_id = ? AND thread_id = ?
                """,
                (delivery_mode, task_id, platform, chat_id, thread_id or ""),
            )
        if metadata_json:
            # Refresh the routing anchor for duplicate subscriptions.
            conn.execute(
                """
                UPDATE kanban_notify_subs
                   SET delivery_metadata = ?
                 WHERE task_id = ? AND platform = ? AND chat_id = ? AND thread_id = ?
                """,
                (metadata_json, task_id, platform, chat_id, thread_id or ""),
            )


def _notify_profile_filter(
    notifier_profiles: Optional[Iterable[str]],
    *,
    include_unowned: bool,
) -> tuple[str, list[str]]:
    """Build an optional SQL predicate for notification profile ownership."""
    if notifier_profiles is None:
        return "", []

    profiles = sorted(
        {
            str(profile).strip()
            for profile in notifier_profiles
            if str(profile).strip()
        }
    )
    clauses: list[str] = []
    params: list[str] = []
    if profiles:
        clauses.append(
            "notifier_profile IN (" + ",".join("?" for _ in profiles) + ")"
        )
        params.extend(profiles)
    if include_unowned:
        clauses.append("notifier_profile IS NULL OR notifier_profile = ''")
    if not clauses:
        return "0", []
    return "(" + ") OR (".join(clauses) + ")", params


def list_notify_subs(
    conn: sqlite3.Connection,
    task_id: Optional[str] = None,
    *,
    notifier_profiles: Optional[Iterable[str]] = None,
    include_unowned: bool = False,
) -> list[dict]:
    """List subscriptions, optionally restricted to notifier profile owners.

    Passing no ``notifier_profiles`` preserves the historical all-subscriptions
    result. Gateway notifier processes pass the profiles whose adapters they
    own so they cannot claim another gateway's events. ``include_unowned`` is
    used by the dispatch owner for legacy rows created before profile stamping.
    """
    owner_where, owner_params = _notify_profile_filter(
        notifier_profiles, include_unowned=include_unowned,
    )
    where: list[str] = []
    params: list[Any] = []
    if task_id is not None:
        where.append("task_id = ?")
        params.append(task_id)
    if owner_where:
        where.append(owner_where)
        params.extend(owner_params)
    sql = "SELECT * FROM kanban_notify_subs"
    if where:
        sql += " WHERE " + " AND ".join(f"({clause})" for clause in where)
    rows = conn.execute(sql, params).fetchall()
    out: list[dict] = []
    for row in rows:
        item = dict(row)
        if "delivery_metadata" in item:
            item["delivery_metadata"] = _decode_notify_delivery_metadata(
                item.get("delivery_metadata")
            )
        out.append(item)
    return out


def count_notify_subs(
    db_path: Optional[Path] = None,
    *,
    board: Optional[str] = None,
    notifier_profiles: Optional[Iterable[str]] = None,
    include_unowned: bool = False,
    platform: Optional[str] = None,
    chat_id: Optional[str] = None,
    thread_id: Optional[str] = None,
) -> int:
    """Count ``kanban_notify_subs`` rows via a read-only connection.

    Cheap probe for the gateway notifier's zero-subscription early exit:
    unlike :func:`connect`, this never creates the DB file, never runs
    schema init/migration, and never opens the database writable (no
    write locks, no checkpoints — though a read-only open of a WAL
    database may still create the ``-shm``/``-wal`` sidecars, it cannot
    write table content). Rows in a not-yet-checkpointed WAL are
    visible, so a freshly added subscription is never missed. A missing
    DB, or a legacy DB that predates the subscriptions table, counts as
    zero. When ``notifier_profiles`` is supplied, only subscriptions owned
    by those profiles are counted; ``include_unowned`` also includes legacy
    rows without an owner stamp. Optional platform/chat/thread filters narrow
    the probe to one notification owner without changing the unfiltered count.
    Platform matching is case-insensitive, matching notifier routing; chat and
    thread identifiers are exact. Path resolution matches :func:`connect`
    (explicit ``db_path``, else ``board`` via :func:`kanban_db_path`). Raises
    :class:`sqlite3.Error` when the DB exists but cannot be read
    (locked, corrupt); callers choose their own fallback.
    """
    path = db_path if db_path is not None else kanban_db_path(board=board)
    if not path.exists():
        return 0
    conn = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)
    try:
        try:
            owner_where, owner_params = _notify_profile_filter(
                notifier_profiles, include_unowned=include_unowned,
            )
            clauses: list[str] = []
            params: list[Any] = []
            if owner_where:
                clauses.append(f"({owner_where})")
                params.extend(owner_params)
            if platform is not None:
                clauses.append("LOWER(platform) = LOWER(?)")
                params.append(platform)
            if chat_id is not None:
                clauses.append("chat_id = ?")
                params.append(chat_id)
            if thread_id is not None:
                clauses.append("thread_id = ?")
                params.append(thread_id)
            query = "SELECT COUNT(*) FROM kanban_notify_subs"
            if clauses:
                query += " WHERE " + " AND ".join(clauses)
            row = conn.execute(query, params).fetchone()
        except sqlite3.OperationalError as exc:
            if "no such table" in str(exc).lower():
                return 0
            raise
        return int(row[0]) if row else 0
    finally:
        conn.close()


def remove_notify_sub(
    conn: sqlite3.Connection,
    *,
    task_id: str,
    platform: str,
    chat_id: str,
    thread_id: Optional[str] = None,
) -> bool:
    with write_txn(conn):
        cur = conn.execute(
            "DELETE FROM kanban_notify_subs WHERE task_id = ? "
            "AND platform = ? AND chat_id = ? AND thread_id = ?",
            (task_id, platform, chat_id, thread_id or ""),
        )
    return cur.rowcount > 0


def purge_stale_done_notify_subs(
    conn: sqlite3.Connection,
    *,
    max_age_days: int = 30,
) -> int:
    """Delete notify subscriptions whose task has sat in ``done`` untouched
    for longer than ``max_age_days``.

    The notifier keeps subscriptions alive through ``done`` because a
    completed task can be reopened (review corrections, continuation) and
    the reopened cycle must still notify its origin session. On boards
    that never archive, that retention would otherwise accumulate
    subscription rows forever — each one scanned every notifier tick.
    This GC bounds that: a task that has been ``done`` with no new events
    for the retention window is treated as settled and its subscriptions
    are purged. Age is measured from the task's most recent event
    (falling back to ``completed_at`` then ``created_at``), so ANY
    activity — including a reopen, which also moves the task off
    ``done`` — resets or exempts it.

    ``max_age_days <= 0`` disables the sweep entirely. Returns the number
    of subscription rows deleted.
    """
    try:
        days = int(max_age_days)
    except (TypeError, ValueError):
        days = 30
    if days <= 0:
        return 0
    cutoff = int(time.time()) - days * 86400
    with write_txn(conn):
        cur = conn.execute(
            "DELETE FROM kanban_notify_subs WHERE task_id IN ("
            " SELECT t.id FROM tasks t"
            " WHERE t.status = 'done'"
            " AND COALESCE("
            "  (SELECT MAX(e.created_at) FROM task_events e"
            "   WHERE e.task_id = t.id),"
            "  t.completed_at, t.created_at, 0"
            " ) < ?)",
            (cutoff,),
        )
    return int(cur.rowcount or 0)


def unseen_events_for_sub(
    conn: sqlite3.Connection,
    *,
    task_id: str,
    platform: str,
    chat_id: str,
    thread_id: Optional[str] = None,
    kinds: Optional[Iterable[str]] = None,
) -> tuple[int, list[Event]]:
    """Return ``(new_cursor, events)`` for a given subscription.

    Only events with ``id > last_event_id`` are returned. The subscription's
    cursor is NOT advanced here; call :func:`advance_notify_cursor` after
    the gateway has successfully delivered the notifications.
    """
    row = conn.execute(
        "SELECT last_event_id FROM kanban_notify_subs "
        "WHERE task_id = ? AND platform = ? AND chat_id = ? AND thread_id = ?",
        (task_id, platform, chat_id, thread_id or ""),
    ).fetchone()
    if row is None:
        return 0, []
    cursor = int(row["last_event_id"])
    kind_list = list(kinds) if kinds else None
    q = (
        "SELECT * FROM task_events WHERE task_id = ? AND id > ? "
        + ("AND kind IN (" + ",".join("?" * len(kind_list)) + ") " if kind_list else "")
        + "ORDER BY id ASC"
    )
    params: list[Any] = [task_id, cursor]
    if kind_list:
        params.extend(kind_list)
    rows = conn.execute(q, params).fetchall()
    out: list[Event] = []
    max_id = cursor
    for r in rows:
        try:
            payload = json.loads(r["payload"]) if r["payload"] else None
        except Exception:
            payload = None
        out.append(Event(
            id=r["id"], task_id=r["task_id"], kind=r["kind"],
            payload=payload, created_at=r["created_at"],
            run_id=(int(r["run_id"]) if "run_id" in r.keys() and r["run_id"] is not None else None),
        ))
        max_id = max(max_id, int(r["id"]))
    return max_id, out


def claim_unseen_events_for_sub(
    conn: sqlite3.Connection,
    *,
    task_id: str,
    platform: str,
    chat_id: str,
    thread_id: Optional[str] = None,
    kinds: Optional[Iterable[str]] = None,
) -> tuple[int, int, list[Event]]:
    """Atomically claim unseen notification events for one subscription.

    Returns ``(old_cursor, new_cursor, events)``. When events are returned,
    ``kanban_notify_subs.last_event_id`` has already been advanced to
    ``new_cursor`` inside a ``BEGIN IMMEDIATE`` transaction. That makes the
    notifier's read/claim step single-owner across multiple gateway watcher
    processes pointed at the same board DB: concurrent watchers serialize on
    SQLite's writer lock, and only the first process sees and claims a given
    event range.

    Callers should send the claimed events, then either leave the cursor at
    ``new_cursor`` on success or call :func:`rewind_notify_cursor` if delivery
    failed before any terminal unsubscribe removed the row.
    """
    with write_txn(conn):
        row = conn.execute(
            "SELECT last_event_id FROM kanban_notify_subs "
            "WHERE task_id = ? AND platform = ? AND chat_id = ? AND thread_id = ?",
            (task_id, platform, chat_id, thread_id or ""),
        ).fetchone()
        if row is None:
            return 0, 0, []
        old_cursor = int(row["last_event_id"])
        new_cursor, events = unseen_events_for_sub(
            conn,
            task_id=task_id,
            platform=platform,
            chat_id=chat_id,
            thread_id=thread_id,
            kinds=kinds,
        )
        if not events:
            return old_cursor, old_cursor, []
        conn.execute(
            "UPDATE kanban_notify_subs SET last_event_id = ? "
            "WHERE task_id = ? AND platform = ? AND chat_id = ? AND thread_id = ? "
            "AND last_event_id = ?",
            (int(new_cursor), task_id, platform, chat_id, thread_id or "", int(old_cursor)),
        )
        return old_cursor, new_cursor, events


def advance_notify_cursor(
    conn: sqlite3.Connection,
    *,
    task_id: str,
    platform: str,
    chat_id: str,
    thread_id: Optional[str] = None,
    new_cursor: int,
) -> None:
    with write_txn(conn):
        conn.execute(
            "UPDATE kanban_notify_subs SET last_event_id = ? "
            "WHERE task_id = ? AND platform = ? AND chat_id = ? AND thread_id = ?",
            (int(new_cursor), task_id, platform, chat_id, thread_id or ""),
        )


def rewind_notify_cursor(
    conn: sqlite3.Connection,
    *,
    task_id: str,
    platform: str,
    chat_id: str,
    thread_id: Optional[str] = None,
    claimed_cursor: int,
    old_cursor: int,
) -> bool:
    """Undo a notification claim when delivery fails.

    The CAS guard only rewinds if no later notifier advanced the row after our
    claim. This keeps retry behavior for transient send failures without
    clobbering newer progress.
    """
    with write_txn(conn):
        cur = conn.execute(
            "UPDATE kanban_notify_subs SET last_event_id = ? "
            "WHERE task_id = ? AND platform = ? AND chat_id = ? AND thread_id = ? "
            "AND last_event_id = ?",
            (
                int(old_cursor), task_id, platform, chat_id, thread_id or "",
                int(claimed_cursor),
            ),
        )
    return cur.rowcount > 0


# ---------------------------------------------------------------------------
# Retention + garbage collection
# ---------------------------------------------------------------------------

def gc_events(
    conn: sqlite3.Connection, *, older_than_seconds: int = 30 * 24 * 3600,
) -> int:
    """Delete task_events rows older than ``older_than_seconds`` for tasks
    in a terminal state (``done`` or ``archived``). Returns the number of
    rows deleted. Running / ready / blocked tasks keep their full event
    history."""
    cutoff = int(time.time()) - int(older_than_seconds)
    with write_txn(conn):
        cur = conn.execute(
            "DELETE FROM task_events WHERE created_at < ? AND task_id IN "
            "(SELECT id FROM tasks WHERE status IN ('done', 'archived'))",
            (cutoff,),
        )
    return int(cur.rowcount or 0)


def gc_worker_logs(
    *, older_than_seconds: int = 30 * 24 * 3600,
    board: Optional[str] = None,
) -> int:
    """Delete worker log files older than ``older_than_seconds``. Returns
    the number of files removed. Kept separate from ``gc_events`` because
    log files live on disk, not in SQLite. Scoped to ``board`` (defaults
    to the active board) — per-board isolation means deleting logs from
    board A cannot touch board B's logs."""
    log_dir = worker_logs_dir(board=board)
    if not log_dir.exists():
        return 0
    cutoff = time.time() - older_than_seconds
    removed = 0
    for p in log_dir.iterdir():
        try:
            if p.is_file() and p.stat().st_mtime < cutoff:
                p.unlink()
                removed += 1
        except OSError:
            continue
    return removed


# ---------------------------------------------------------------------------
# Worker log accessor
# ---------------------------------------------------------------------------

def worker_log_path(task_id: str, *, board: Optional[str] = None) -> Path:
    """Return the path to a worker's log file. The file may not exist
    (task never spawned, or log already GC'd).

    When ``board`` is None, resolves via the active board (env var →
    current-board file → default). The dispatcher always passes the
    board explicitly to avoid any resolution ambiguity when multiple
    boards exist."""
    return worker_logs_dir(board=board) / f"{task_id}.log"


def read_worker_log(
    task_id: str, *, tail_bytes: Optional[int] = None,
    board: Optional[str] = None,
) -> Optional[str]:
    """Read the worker log for ``task_id``. Returns None if the file
    doesn't exist. If ``tail_bytes`` is set, only the last N bytes are
    returned (useful for the dashboard drawer which shouldn't page megabytes)."""
    path = worker_log_path(task_id, board=board)
    if not path.exists():
        return None
    try:
        if tail_bytes is None:
            return path.read_text(encoding="utf-8", errors="replace")
        size = path.stat().st_size
        with open(path, "rb") as f:
            if size > tail_bytes:
                f.seek(size - tail_bytes)
                # Skip a partial line if we tailed mid-line. But if the
                # window has no newline at all (one giant log line),
                # readline() would eat everything — in that case don't
                # skip and return the raw tail.
                probe = f.tell()
                partial = f.readline()
                if not partial.endswith(b"\n") and f.tell() >= size:
                    f.seek(probe)
            data = f.read()
        return data.decode("utf-8", errors="replace")
    except OSError:
        return None


# ---------------------------------------------------------------------------
# Assignee enumeration (known profiles + per-profile board stats)
# ---------------------------------------------------------------------------

def list_profiles_on_disk() -> list[str]:
    """Return the set of assignee/profile names discovered on disk.

    Includes:
    - named profiles under ``<default-root>/profiles/<name>/config.yaml``
    - the implicit ``default`` profile when the default Hermes root exists

    Reads profile paths directly so this module has no import dependency on
    ``hermes_cli.profiles`` (which pulls in a large chunk of the CLI startup
    path).
    """
    try:
        from hermes_constants import get_default_hermes_root
        default_root = get_default_hermes_root()
        profiles_dir = default_root / "profiles"
    except Exception:
        return []

    names: set[str] = set()
    if default_root.exists():
        names.add("default")

    if profiles_dir.is_dir():
        try:
            for entry in sorted(profiles_dir.iterdir()):
                if not entry.is_dir():
                    continue
                if (entry / "config.yaml").is_file():
                    names.add(entry.name)
        except OSError:
            pass

    return sorted(names)


def known_assignees(conn: sqlite3.Connection) -> list[dict]:
    """Return every assignee name known to the board or on disk.

    Each entry is ``{"name": str, "on_disk": bool, "counts": {status: n}}``.
    A name is included when it's a configured profile on disk OR when
    any non-archived task has it as the assignee. Used by:

    - ``hermes kanban assignees`` for the terminal.
    - The dashboard assignee dropdown (so a fresh profile appears in
      the picker even before it's been given any task).
    - Router-profile heuristics ("who's overloaded?") without scanning
      the whole board.
    """
    on_disk = set(list_profiles_on_disk())

    # Count tasks per (assignee, status), excluding archived.
    counts: dict[str, dict[str, int]] = {}
    for row in conn.execute(
        "SELECT assignee, status, COUNT(*) AS n FROM tasks "
        "WHERE status != 'archived' AND assignee IS NOT NULL "
        "AND task_kind = 'work' GROUP BY assignee, status"
    ):
        counts.setdefault(row["assignee"], {})[row["status"]] = int(row["n"])

    names = sorted(on_disk | set(counts.keys()))
    return [
        {
            "name": name,
            "on_disk": name in on_disk,
            "counts": counts.get(name, {}),
        }
        for name in names
    ]


# ---------------------------------------------------------------------------
# Runs (attempt history on a task)
# ---------------------------------------------------------------------------

def list_runs(
    conn: sqlite3.Connection,
    task_id: str,
    *,
    include_active: bool = True,
    state_type: Optional[str] = None,
    state_name: Optional[str] = None,
) -> list[Run]:
    """Return all runs for ``task_id`` in start order.

    ``include_active=True`` (default) includes the currently-running
    attempt if any. Set False to return only closed runs (useful for
    "how many prior attempts have there been?" checks).

    When ``state_type`` and ``state_name`` are set, restrict to rows
    where that column equals ``state_name`` (``state_type`` is
    ``status`` or ``outcome``). Both must be passed together.
    """
    if (state_type is None) ^ (state_name is None):
        raise ValueError("state_type and state_name must both be set or both omitted")
    if state_type is not None:
        if state_type not in ("status", "outcome"):
            raise ValueError("state_type must be 'status' or 'outcome'")
    q = "SELECT * FROM task_runs WHERE task_id = ?"
    params: list[Any] = [task_id]
    if not include_active:
        q += " AND ended_at IS NOT NULL"
    if state_type is not None:
        q += f" AND {state_type} = ?"
        params.append(state_name)
    q += " ORDER BY started_at ASC, id ASC"
    rows = conn.execute(q, params).fetchall()
    return [Run.from_row(r) for r in rows]


def get_run(conn: sqlite3.Connection, run_id: int) -> Optional[Run]:
    row = conn.execute(
        "SELECT * FROM task_runs WHERE id = ?", (int(run_id),),
    ).fetchone()
    return Run.from_row(row) if row else None


def latest_run(conn: sqlite3.Connection, task_id: str) -> Optional[Run]:
    """Return the most recent run regardless of outcome (active or closed)."""
    row = conn.execute(
        "SELECT * FROM task_runs WHERE task_id = ? "
        "ORDER BY started_at DESC, id DESC LIMIT 1",
        (task_id,),
    ).fetchone()
    return Run.from_row(row) if row else None


def latest_summary(conn: sqlite3.Connection, task_id: str) -> Optional[str]:
    """Return the latest non-null ``task_runs.summary`` for ``task_id``.

    The worker writes its handoff to ``task_runs.summary``
    via ``complete_task(summary=...)``; ``tasks.result`` is left empty
    unless the caller passes ``result=`` explicitly. Dashboards and CLI
    "show" views need this value to surface what a worker actually did
    — without it, ``tasks.result`` is NULL and the task looks like a
    no-op even when the run completed.

    Picks the most recent run by ``ended_at`` (falling back to ``id``
    for ties or unfinished rows). Returns None if no run has a summary.
    """
    row = conn.execute(
        "SELECT summary FROM task_runs "
        "WHERE task_id = ? AND summary IS NOT NULL AND summary != '' "
        "ORDER BY COALESCE(ended_at, started_at) DESC, id DESC LIMIT 1",
        (task_id,),
    ).fetchone()
    return row["summary"] if row else None


def latest_summaries(
    conn: sqlite3.Connection, task_ids: Iterable[str]
) -> dict[str, str]:
    """Batch-fetch latest non-null summaries for a list of task ids.

    Used by the dashboard board endpoint to attach ``latest_summary`` to
    every card in a single SQL query, avoiding the N+1 pattern of
    calling :func:`latest_summary` per task. Returns a dict mapping
    ``task_id`` → summary string, omitting tasks with no summary.

    Approach: a window function picks the newest non-null-summary row
    per ``task_id``; works against SQLite ≥ 3.25 (default on every
    supported platform).
    """
    ids = list(task_ids)
    if not ids:
        return {}
    placeholders = ",".join("?" for _ in ids)
    rows = conn.execute(
        f"""
        SELECT task_id, summary FROM (
            SELECT task_id, summary,
                   ROW_NUMBER() OVER (
                       PARTITION BY task_id
                       ORDER BY COALESCE(ended_at, started_at) DESC, id DESC
                   ) AS rn
              FROM task_runs
             WHERE task_id IN ({placeholders})
               AND summary IS NOT NULL AND summary != ''
        ) WHERE rn = 1
        """,
        ids,
    ).fetchall()
    return {r["task_id"]: r["summary"] for r in rows}
