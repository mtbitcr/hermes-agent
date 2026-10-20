"""Stop a running card's worker when someone other than that worker schedules it.

Finding 97: ``schedule_task`` clears the card's worker and claim and closes its
run. It captures the worker before its update and stops it after its commit,
the way the reclaim paths do.
"""

from __future__ import annotations

import sqlite3
from typing import Optional


def capture_worker(
    conn: sqlite3.Connection, task_id: str,
) -> Optional[tuple[int, Optional[str], Optional[int]]]:
    """The running card's recorded ``(pid, claim_lock, start time)``, or None."""
    # Late import: the facade imports this module at load time.
    from hermes_cli.kanban_db import _row_worker_start_time

    row = conn.execute(
        "SELECT claim_lock, worker_pid, worker_start_time FROM tasks "
        "WHERE id = ? AND status = 'running'",
        (task_id,),
    ).fetchone()
    if row is None or not row["worker_pid"]:
        return None
    return row["worker_pid"], row["claim_lock"], _row_worker_start_time(row)


def stop_after_commit(
    captured: Optional[tuple[int, Optional[str], Optional[int]]],
) -> None:
    """Stop the captured worker; a pid that now belongs to another process is not signalled."""
    if captured is None:
        return
    # Late import: the facade imports this module at load time.
    from hermes_cli.kanban_db import _terminate_reclaimed_worker

    pid, claim_lock, recorded_start_time = captured
    _terminate_reclaimed_worker(pid, claim_lock, recorded_start_time=recorded_start_time)
