"""Work the AI provider stopped: hold the card once, with a believable resume time.

A worker stopped by the provider's usage limit exits ``KANBAN_RATE_LIMIT_EXIT_CODE``
and :func:`hermes_cli.kanban_db.detect_crashed_workers` books its run
``rate_limited``; :func:`hermes_cli.kanban_db.check_respawn_guard` then holds the
card. This module is what the kernel adds to that booking and that hold:

* Every such run carries a resume time. A believable reset the worker recorded
  wins; otherwise the kernel derives one: the run's end plus the cooldown,
  doubled for each consecutive rate-limited run on the card's provider, capped at
  six hours, and marked as kernel-derived.
* The card's provider is kept beside the reset, and a reset recorded for another
  provider than the card's current route no longer holds it. A reset stored
  before providers were kept keeps today's behaviour.
* A held card gets one ``respawn_guarded`` event, not one per dispatcher tick.
* Only a failed run's own error can name a sign-in problem.
* A worker whose provider refused the work (a ``content_policy_blocked`` turn)
  exits ``KANBAN_PROVIDER_REFUSED_EXIT_CODE``; its card is parked blocked as a
  capability wall, never retried unchanged, and no failure is counted.
"""

from __future__ import annotations

import json
import sqlite3
from typing import Any, Optional

# Exit code of a kanban worker whose provider refused the work. Unused by any
# other exit: 75 (EX_TEMPFAIL) is the usage-limit sentinel.
KANBAN_PROVIDER_REFUSED_EXIT_CODE = 77

# A turn result whose error starts with this is a provider refusal.
PROVIDER_REFUSED_ERROR_PREFIX = "content_policy_blocked"

# The run outcome such an exit is booked with.
PROVIDER_REFUSED_OUTCOME = "provider_refused"

# Upper bound of a kernel-derived hold, however long the streak.
RATE_LIMIT_HOLD_CAP_SECONDS = 6 * 3600

# Kept on a rate-limited run's metadata beside ``rate_limit_reset_at``: the
# card's provider when the stop was booked (``None`` when the card named none),
# and ``"kernel"`` when the kernel derived the reset itself.
RATE_LIMIT_RESET_PROVIDER_KEY = "rate_limit_reset_provider"
RATE_LIMIT_RESET_SOURCE_KEY = "rate_limit_reset_source"
KERNEL_RESET_SOURCE = "kernel"

# Outcomes of a run that failed: the failure recorder's crashed, timed_out and
# spawn_failed, and the breaker's gave_up.
FAILED_RUN_OUTCOMES = frozenset({"crashed", "timed_out", "spawn_failed", "gave_up"})

# How many earlier runs a streak is counted over; the cap is reached long before.
_STREAK_SCAN_LIMIT = 32


def is_provider_refusal(result: Any) -> bool:
    """True for a failed turn result whose error says the provider refused the work."""
    return (
        isinstance(result, dict)
        and bool(result.get("failed"))
        and str(result.get("error") or "").startswith(PROVIDER_REFUSED_ERROR_PREFIX)
    )


def _metadata(raw: Any) -> Optional[dict]:
    """A run's stored metadata as a dict: ``{}`` when empty, ``None`` when unreadable."""
    if not raw:
        return {}
    try:
        value = json.loads(raw)
    except (TypeError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def _provider(value: Any) -> Optional[str]:
    if value is None:
        return None
    return str(value).strip() or None


def _card_provider(conn: sqlite3.Connection, task_id: str) -> Optional[str]:
    row = conn.execute(
        "SELECT provider_override FROM tasks WHERE id = ?", (task_id,),
    ).fetchone()
    return _provider(row["provider_override"]) if row is not None else None


def _recorded_for(metadata: Any) -> tuple[bool, Optional[str]]:
    """``(kept, provider)``: whether a provider was kept beside the run's reset, and which."""
    if not isinstance(metadata, dict) or RATE_LIMIT_RESET_PROVIDER_KEY not in metadata:
        return False, None
    return True, _provider(metadata[RATE_LIMIT_RESET_PROVIDER_KEY])


def latest_ended_run(conn: sqlite3.Connection, task_id: str) -> Optional[sqlite3.Row]:
    """The task's most recently ended run, as the respawn guard reads it."""
    return conn.execute(
        "SELECT outcome, ended_at, metadata, error FROM task_runs "
        "WHERE task_id = ? AND ended_at IS NOT NULL "
        "ORDER BY ended_at DESC LIMIT 1",
        (task_id,),
    ).fetchone()


def _consecutive_stops(
    conn: sqlite3.Connection, task_id: str, run_id: int, provider: Optional[str],
) -> int:
    """Rate-limited runs in a row on ``provider``, ending with (and counting) ``run_id``."""
    stops = 1
    for row in conn.execute(
        "SELECT outcome, metadata FROM task_runs "
        "WHERE task_id = ? AND id < ? AND ended_at IS NOT NULL "
        "ORDER BY id DESC LIMIT ?",
        (task_id, run_id, _STREAK_SCAN_LIMIT),
    ):
        if row["outcome"] != "rate_limited":
            break
        if _recorded_for(_metadata(row["metadata"])) != (True, provider):
            break
        stops += 1
    return stops


def book_rate_limit_reset(conn: sqlite3.Connection, task_id: str, run_id: int) -> None:
    """Give a just-closed rate-limited run its resume time and the card's provider.

    Called inside the booking transaction, right after the run is closed. The
    worker's own reset stays when it is believable; otherwise, while the
    cooldown is on, the kernel writes ``ended_at + min(cooldown * 2**stops, cap)``
    where ``stops`` counts the consecutive rate-limited runs on this provider,
    this one included. Unreadable metadata is left as it is.
    """
    from hermes_cli import kanban_db as kb

    run = conn.execute(
        "SELECT ended_at, metadata FROM task_runs WHERE id = ? AND task_id = ?",
        (run_id, task_id),
    ).fetchone()
    if run is None or run["ended_at"] is None:
        return
    metadata = _metadata(run["metadata"])
    if metadata is None:
        return
    ended_at = int(run["ended_at"])
    provider = _card_provider(conn, task_id)
    metadata[RATE_LIMIT_RESET_PROVIDER_KEY] = provider
    if kb.recorded_rate_limit_reset(metadata, anchor=ended_at) is None:
        cooldown = kb._resolve_rate_limit_cooldown_seconds()
        if cooldown > 0:
            stops = _consecutive_stops(conn, task_id, run_id, provider)
            metadata[kb._RATE_LIMIT_RESET_KEY] = ended_at + min(
                cooldown * 2 ** stops, RATE_LIMIT_HOLD_CAP_SECONDS,
            )
            metadata[RATE_LIMIT_RESET_SOURCE_KEY] = KERNEL_RESET_SOURCE
    conn.execute(
        "UPDATE task_runs SET metadata = ? WHERE id = ?",
        (json.dumps(metadata, ensure_ascii=False), run_id),
    )


def rate_limit_resume_at(
    conn: sqlite3.Connection,
    task_id: str,
    run: Optional[sqlite3.Row] = None,
    *,
    cooldown: Optional[int] = None,
) -> Optional[int]:
    """When a card held after a rate-limited run may start again; ``None`` when it is not so held.

    ``run`` is the task's latest ended run (read when omitted). The hold lasts
    the cooldown, or until the run's recorded reset when that is later, unless
    the reset was kept for another provider than the card's current one.
    """
    from hermes_cli import kanban_db as kb

    if run is None:
        run = latest_ended_run(conn, task_id)
    if run is None or run["outcome"] != "rate_limited" or run["ended_at"] is None:
        return None
    if cooldown is None:
        cooldown = kb._resolve_rate_limit_cooldown_seconds()
    if cooldown <= 0:
        return None
    ended_at = int(run["ended_at"])
    metadata = _metadata(run["metadata"])
    resume_at = ended_at + cooldown
    reset = kb.recorded_rate_limit_reset(metadata, anchor=ended_at)
    if reset is not None:
        kept, provider = _recorded_for(metadata)
        if not kept or provider == _card_provider(conn, task_id):
            resume_at = max(resume_at, reset)
    return resume_at


def names_sign_in_blocker(
    latest_run: Optional[sqlite3.Row], last_failure_error: Optional[str], pattern: Any,
) -> bool:
    """Whether the latest ended run failed with an error naming a quota or sign-in wall.

    Only that run's own error counts, and only when it failed; a rate-limit stop
    or a review handback is never read as a sign-in problem. The card's
    ``last_failure_error`` must still be set: unblocking, reassigning and
    completing clear it, and each of those still frees the card.
    """
    if not last_failure_error or latest_run is None:
        return False
    if latest_run["outcome"] not in FAILED_RUN_OUTCOMES:
        return False
    return bool(latest_run["error"] and pattern.search(latest_run["error"]))


def record_respawn_guarded(conn: sqlite3.Connection, task_id: str, reason: str) -> bool:
    """Append the ``respawn_guarded`` event for a held card, once per hold.

    Written only when none exists since the task's latest run ended, or when
    the reason or the resume time changed; a card held tick after tick keeps its
    revision. Returns whether an event was written.
    """
    from hermes_cli import kanban_db as kb

    latest = latest_ended_run(conn, task_id)
    payload: dict = {"reason": reason}
    if reason == "rate_limit_cooldown":
        resume_at = rate_limit_resume_at(conn, task_id, latest)
        if resume_at is not None:
            payload["resume_at"] = resume_at
    previous = conn.execute(
        "SELECT created_at, payload FROM task_events "
        "WHERE task_id = ? AND kind = 'respawn_guarded' ORDER BY id DESC LIMIT 1",
        (task_id,),
    ).fetchone()
    if (
        previous is not None
        and (latest is None or previous["created_at"] >= latest["ended_at"])
        and _metadata(previous["payload"]) == payload
    ):
        return False
    kb._append_event(conn, task_id, "respawn_guarded", payload)
    return True


def park_provider_refusal(
    conn: sqlite3.Connection,
    *,
    task_id: str,
    assignee: Optional[str],
    pid: int,
    claim_lock: Optional[str],
    run_id: Optional[int],
    exit_code: Optional[int],
    evidence: dict,
    source_status: str,
) -> Optional[dict]:
    """Park a card whose provider refused the work, inside the crash sweep's transaction.

    The run closes ``provider_refused``; no failure is counted and no failure
    message is stamped. The card is blocked with ``block_kind='capability'`` and
    the ``blocked`` event that makes the block sticky, keeping the bookkeeping
    of ``_park_unreported_completion``, so it leaves only through
    ``unblock_task`` (the owner's retry) and is never started again unchanged.
    ``source_status`` is the lane the run was claimed from.

    Applied only while the task is still running under this worker, claim and
    run. Returns the worker-exited observer payload, or ``None`` (having written
    nothing) when the card has moved on.
    """
    from hermes_cli import kanban_db as kb

    prior = conn.execute(
        "SELECT block_kind, block_recurrences FROM tasks WHERE id = ?",
        (task_id,),
    ).fetchone()
    recurrences = (
        int(prior["block_recurrences"] or 0) + 1
        if prior is not None and prior["block_kind"] == "capability"
        else 1
    )
    cur = conn.execute(
        "UPDATE tasks SET status = 'blocked', block_kind = 'capability', "
        "block_recurrences = ?, "
        "claim_lock = NULL, claim_expires = NULL, worker_pid = NULL "
        "WHERE id = ? AND status = 'running' "
        "  AND worker_pid = ? AND claim_lock IS ? AND current_run_id IS ?",
        (recurrences, task_id, pid, claim_lock, run_id),
    )
    if cur.rowcount != 1:
        return None
    closed_run_id = kb._end_run(
        conn, task_id,
        outcome=PROVIDER_REFUSED_OUTCOME, status=PROVIDER_REFUSED_OUTCOME,
        error=(
            f"pid {pid} exited provider-refused ({PROVIDER_REFUSED_ERROR_PREFIX}) "
            "— parked without counting a failure"
        ),
        metadata={"pid": pid, "claimer": claim_lock, "exit_code": exit_code, **evidence},
    )
    kb._append_event(
        conn, task_id, "blocked",
        {
            "reason": (
                "The AI provider refused this work as worded; it is parked "
                "instead of being retried unchanged."
            ),
            "kind": "capability",
            "recurrences": recurrences,
            "source_status": source_status,
        },
        run_id=closed_run_id,
    )
    return {
        "task_id": task_id,
        "assignee": assignee,
        "run_id": closed_run_id,
        "worker_pid": pid,
        "exit_kind": PROVIDER_REFUSED_OUTCOME,
        "exit_code": exit_code,
        "outcome": PROVIDER_REFUSED_OUTCOME,
        "retry_status": "blocked",
    }
