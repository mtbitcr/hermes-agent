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


_PROVIDER = "openai-codex"
_PIN_COLUMNS = (
    "assignee", "provider_override", "model_override", "reasoning_effort",
    "execution_tier", "model_policy_lock",
)
# One switch record per card: its held run and the pin it had before the
# switch. ``kanban_db.gc_events`` prunes events, never this table.
_RUN_PINS = "owner_retry_run_pins"


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

    # Model policy stays the authority for both role-wide and task-only routes.
    try:
        assignment = model_policy.task_assignment_for(role, _PROVIDER, tier)
        lock = model_policy.mint_policy_lock(
            role, _PROVIDER, assignment.model, assignment.reasoning_effort, tier,
            owner_retry=True,
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


def keep_switch(conn: sqlite3.Connection, task_id: str, route: RetryRoute, held_run_id: int) -> None:
    """Write the card's one switch record inside the caller's retry transaction, before :func:`repin`.

    The record is the held run and ``route.previous``, the card's pin before
    the switch; a later approved switch of the same card replaces it.
    """
    columns = ("task_id", "held_run_id", *_PIN_COLUMNS)
    conn.execute(
        f"CREATE TABLE IF NOT EXISTS {_RUN_PINS} (task_id TEXT PRIMARY KEY, "
        "held_run_id INTEGER NOT NULL, "
        + ", ".join(f"{column} TEXT" for column in _PIN_COLUMNS) + ")"
    )
    conn.execute(
        f"INSERT INTO {_RUN_PINS} ({', '.join(columns)}) "
        f"VALUES ({', '.join('?' for _ in columns)}) ON CONFLICT(task_id) DO UPDATE SET "
        + ", ".join(f"{column} = excluded.{column}" for column in columns[1:]),
        (task_id, held_run_id, *(route.previous[column] for column in _PIN_COLUMNS)),
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
