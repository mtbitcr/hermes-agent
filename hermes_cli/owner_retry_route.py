"""The approved alternate route for one card after a provider limit."""

import hashlib
import json
import math
import os
import re
import sqlite3
import time
from dataclasses import dataclass
from typing import Any

from hermes_cli import kanban_db, kanban_provider_stops
from plugins.dashboard_auth.raphael_workspace import model_policy


_RETRY_LANES = frozenset({
    ("raphael-claude-worker", "deep"),
})
_PROVIDER = "openai-codex"
_PIN_COLUMNS = (
    "assignee", "provider_override", "model_override", "reasoning_effort",
    "execution_tier", "model_policy_lock",
)
# One row per run a switch found up to its held run: the route that run is
# proven to have run under, all NULL when unproven. ``kanban_db.gc_events``
# prunes the events the proof is read from, but no other table.
_RUN_PINS = "owner_retry_run_pins"
_ROUTE_CHANGES = ("model_route_pinned", "model_route_repinned", "model_route_unapproved")


@dataclass(frozen=True)
class RetryRoute:
    token: str
    previous: dict
    replacement: dict

    def view(self) -> dict:
        return {
            "token": self.token,
            "model": self.replacement["model"],
            "effort": self.replacement["reasoning_effort"],
        }


def resolve_retry_authority(value: Any) -> "dict[str, Any] | None":
    """Validate the closed transport shape for one owner retry run."""
    if value is None:
        return None
    if not isinstance(value, dict) or set(value) != {
        "operation", "idempotency_key", "payload",
    }:
        raise ValueError("invalid owner retry authority")
    operation = value.get("operation")
    idempotency_key = value.get("idempotency_key")
    payload = value.get("payload")
    required = {"idempotency_key", "project_id", "task_id", "reason"}
    if (
        operation != "owner_task_retry"
        or not isinstance(idempotency_key, str)
        or re.fullmatch(r"[A-Za-z0-9._:-]{1,200}", idempotency_key) is None
        or not isinstance(payload, dict)
        or not required <= set(payload) <= required | {"route_token"}
        or payload.get("idempotency_key") != idempotency_key
    ):
        raise ValueError("invalid owner retry authority")
    if "route_token" in payload and (
        not isinstance(payload["route_token"], str)
        or re.fullmatch(r"[0-9a-f]{64}", payload["route_token"]) is None
    ):
        raise ValueError("invalid owner retry route token")
    return {
        "operation": operation,
        "idempotency_key": idempotency_key,
        "payload": payload,
    }

def offer(task: kanban_db.Task, held: sqlite3.Row, *, revision: int) -> RetryRoute | None:
    role, tier = task.assignee, task.execution_tier
    # The gateway enables one approved card. The policy lock alone is insufficient.
    if (
        task.id != os.environ.get("HERMES_OPENAI_RETRY_TASK_ID")
        or task.status != "ready"
        or (role, tier) not in _RETRY_LANES
        or task.provider_override != "anthropic"
        or not task.model_policy_lock
        or any(value is not None for value in (task.claim_lock, task.current_run_id, task.worker_pid))
        or held["task_id"] != task.id
    ):
        return None
    previous = {column: getattr(task, column) for column in _PIN_COLUMNS}
    if kanban_db.task_policy_lock_error(previous):
        return None
    metadata = kanban_provider_stops._metadata(held["metadata"])
    if kanban_provider_stops._recorded_for(metadata) != (True, "anthropic"):
        return None

    try:
        assignment = model_policy.task_assignment_for(role, _PROVIDER, tier)
        lock = model_policy.mint_policy_lock(
            role, _PROVIDER, assignment.model, assignment.reasoning_effort, tier,
        )
    except ValueError:
        return None
    replacement = {
        "assignee": role,
        "provider": _PROVIDER,
        "model": assignment.model,
        "reasoning_effort": assignment.reasoning_effort,
        "execution_tier": tier,
        "model_policy_lock": lock,
    }
    # This token checks freshness. The native owner-run authority grants permission.
    # Bind the approval to the card version that the owner saw.
    binding = [task.project_id, task.id, held["id"], revision, task.model_policy_lock, lock]
    token = hashlib.sha256(json.dumps(binding, separators=(",", ":")).encode()).hexdigest()
    return RetryRoute(token, previous, replacement)


def openai_ready(role: str) -> bool:
    """Read the selected role's local sign-in expiry without a provider request."""
    from hermes_cli.auth_constants import _decode_jwt_claims
    from hermes_cli.model_update_check import _role_value, _runtime_token

    try:
        token = _role_value(role, lambda: _runtime_token(_PROVIDER))
        expires_at = _decode_jwt_claims(token).get("exp")
        return (
            type(expires_at) in (int, float)
            and time.time() < expires_at < math.inf
        )
    except (OSError, RuntimeError, TypeError, ValueError):
        return False


def keep_run_pins(conn: sqlite3.Connection, task_id: str, route: RetryRoute, run_id: int) -> None:
    """Keep each earlier run's own pin inside the caller's retry transaction, before :func:`repin`.

    The card is ready, so its event history is still complete. A run up to the
    held run ``run_id`` with no kept row ran under ``route.previous`` when it
    has a bound start event (its first) and no route change of the card follows
    that start; otherwise its pin is unproven. A kept row is never overwritten.
    """
    columns = ("run_id", "task_id", "held_run_id", *_PIN_COLUMNS, "created_at")
    conn.execute(
        f"CREATE TABLE IF NOT EXISTS {_RUN_PINS} (run_id INTEGER PRIMARY KEY, "
        "task_id TEXT NOT NULL, held_run_id INTEGER NOT NULL, "
        + "".join(f"{column} TEXT, " for column in _PIN_COLUMNS)
        + "created_at INTEGER NOT NULL)"
    )
    [changed] = conn.execute(
        "SELECT MAX(id) FROM task_events WHERE task_id = ? AND kind IN "
        f"({', '.join('?' for _ in _ROUTE_CHANGES)})",
        (task_id, *_ROUTE_CHANGES),
    ).fetchone()
    runs = conn.execute(
        "SELECT r.id, MIN(e.id) FROM task_runs r "
        "LEFT JOIN task_events e ON e.run_id = r.id AND e.task_id = r.task_id "
        f"WHERE r.task_id = ? AND r.id <= ? AND r.id NOT IN (SELECT run_id FROM {_RUN_PINS}) "
        "GROUP BY r.id",
        (task_id, run_id),
    ).fetchall()
    now = int(time.time())
    for earlier, start in runs:
        proven = start is not None and (changed is None or start > changed)
        conn.execute(
            f"INSERT INTO {_RUN_PINS} ({', '.join(columns)}) "
            f"VALUES ({', '.join('?' for _ in columns)})",
            (
                earlier, task_id, run_id,
                *(route.previous[column] if proven else None for column in _PIN_COLUMNS), now,
            ),
        )


def repin(conn: sqlite3.Connection, task_id: str, route: RetryRoute, run_id: int) -> None:
    """Change the next route inside the caller's retry transaction."""
    assignments, event = kanban_db._approved_route_assignments(
        task_id, route.previous["assignee"], route.replacement,
    )
    columns = ", ".join(f"{column} = ?" for column, _ in assignments)
    values = [value for _, value in assignments]
    conn.execute(f"UPDATE tasks SET {columns} WHERE id = ?", (*values, task_id))
    kanban_db._append_event(conn, task_id, "model_route_repinned", event, run_id=run_id)


def kept_run_pins(conn: sqlite3.Connection, run_ids: list) -> dict[int, sqlite3.Row | None]:
    """Each of ``run_ids`` up to its card's latest held run: its kept row, else None. Read-only.

    The joins re-prove that each kept row's run is that row's task's run. A run
    after its card's latest switch is absent, and so is every run on a board
    where no switch was ever written, which has no table.
    """
    if conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?", (_RUN_PINS,),
    ).fetchone() is None:
        return {}
    kept: dict[int, sqlite3.Row | None] = {}
    for row in conn.execute(
        f"SELECT r.id AS id, p.*, (SELECT MAX(q.held_run_id) FROM {_RUN_PINS} q "
        "JOIN task_runs s ON s.id = q.run_id AND s.task_id = q.task_id "
        "WHERE q.task_id = r.task_id) AS held "
        f"FROM task_runs r LEFT JOIN {_RUN_PINS} p ON p.run_id = r.id AND p.task_id = r.task_id "
        f"WHERE r.id IN ({', '.join('?' for _ in run_ids)})",
        tuple(run_ids),
    ):
        if row["run_id"] is not None:
            kept[int(row["id"])] = row
        elif row["held"] is not None and row["id"] <= row["held"]:
            kept[int(row["id"])] = None
    return kept
