"""Read approved plan history without migrating or changing either store."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sqlite3

from hermes_cli import owner_workspace

PLAN_RECEIPT_LIMIT = 20
# commit_project_plan() renders these four bounded fields and their separators into the stored plan text.
PLAN_TEXT_LIMIT = 240 + 2000 + 20000 + 1000 + len("Owner-visible result\n") + 6
# A merge can name six targets; a plan can contain twelve changes.
PLAN_CHANGE_LIMIT = owner_workspace._MAX_PROJECT_PLAN_CHANGES * owner_workspace._MAX_PROJECT_PLAN_REPLACEMENTS
SUPERSEDING_ACTIONS = {"replace", "split", "merge", "cancel"}


def _ref(identity: list) -> str:
    if len(identity) != 6 or any(not isinstance(item, str) or not item.strip() for item in identity[:4]):
        raise ValueError("review_plan_receipt_identity_invalid")
    return "plan_" + hashlib.sha256(
        json.dumps(identity[:4], ensure_ascii=True, separators=(",", ":")).encode(),
    ).hexdigest()


def _ids(value) -> list[str]:
    if not isinstance(value, list) or any(not isinstance(item, str) or not item for item in value):
        raise ValueError("review_plan_receipt_tasks_invalid")
    return value


def _changes(registry, project, identity, applied_event, receipt, max_bytes):
    rows = registry.execute(
        "WITH changes AS (SELECT e.id,e.task_id, "
        "CASE WHEN length(CAST(e.payload AS BLOB))<=? THEN CASE WHEN json_valid(e.payload) "
        "THEN e.payload END END AS payload FROM review_plan_board.task_events e "
        "JOIN review_plan_board.tasks t ON t.id=e.task_id WHERE t.project_id=? "
        "AND e.kind='owner_project_plan_change' AND e.id<?) "
        "SELECT * FROM changes WHERE json_extract(payload,'$.actor')=? "
        "AND json_extract(payload,'$.profile')=? AND json_extract(payload,'$.idempotency_key')=? "
        "AND json_extract(payload,'$.request_digest')=? ORDER BY id LIMIT ?",
        (max_bytes, project.id, applied_event, *identity[:4], PLAN_CHANGE_LIMIT + 1),
    ).fetchall()
    if len(rows) > PLAN_CHANGE_LIMIT or len({row['task_id'] for row in rows}) != len(rows):
        raise ValueError("review_plan_changes_invalid")
    changes = []
    for row in rows:
        value = json.loads(row["payload"])
        action = value.get("action")
        replacements = ([value.get("replacement_task_id")] if action in {"replace", "merge"}
                        else value.get("replacement_task_ids") if action == "split" else [])
        replacements = _ids(replacements)
        if (action not in SUPERSEDING_ACTIONS | {"move", "postpone"}
                or row["task_id"] not in receipt["affected_task_ids"]
                or not set(replacements).issubset(receipt["created_task_ids"])
                or action in {"replace", "merge"} and len(replacements) != 1
                or action == "split" and not 2 <= len(replacements) <= owner_workspace._MAX_PROJECT_PLAN_REPLACEMENTS):
            raise ValueError("review_plan_changes_invalid")
        changes.append({"event_id": row["id"], "task_id": row["task_id"], "action": action,
                        "replacement_task_ids": replacements})
    return changes


def read_plan_receipts(
    registry: sqlite3.Connection, project, board_path: Path | None, since: int | None, *, max_bytes: int,
) -> tuple[list[dict], int]:
    """Read one bounded window from the caller's read-only registry snapshot."""
    params = (project.id, since or 0)
    total = registry.execute(
        "SELECT COUNT(*) FROM owner_workspace_receipts WHERE project_id=? "
        "AND operation='owner_project_plan_commit' AND status='committed' AND updated_at>=?", params,
    ).fetchone()[0]
    if not total:
        return [], 0
    if board_path is None:
        raise ValueError("review_plan_board_unavailable")
    # The read-only URI prevents an absent board from becoming an empty database.
    registry.execute("ATTACH DATABASE ? AS review_plan_board", (board_path.resolve().as_uri() + "?mode=ro",))
    rows = registry.execute(
        "WITH plan_events AS ("
        " SELECT e.id, e.task_id, CASE WHEN length(CAST(e.payload AS BLOB))<=? "
        " THEN CASE WHEN json_valid(e.payload) THEN e.payload END END AS payload "
        " FROM review_plan_board.task_events e JOIN review_plan_board.tasks t ON t.id=e.task_id "
        " WHERE t.project_id=? AND e.kind='owner_project_plan_applied'"
        ") SELECT r.updated_at, "
        "CASE WHEN length(CAST(COALESCE(r.actor,'') || COALESCE(r.profile,'') || "
        " COALESCE(r.idempotency_key,'') || COALESCE(r.request_digest,'') || "
        " COALESCE(r.board_slug,'') || COALESCE(r.task_id,'') AS BLOB))<=? "
        " THEN json_array(r.actor,r.profile,r.idempotency_key,r.request_digest,r.board_slug,r.task_id) END AS identity_json, "
        "CASE WHEN length(CAST(r.result_json AS BLOB))<=? THEN r.result_json END AS result_json, "
        "e.id AS event_id,e.payload AS event_payload,COUNT(e.id) OVER (PARTITION BY r.rowid) AS event_count "
        "FROM owner_workspace_receipts r LEFT JOIN plan_events e ON e.task_id=r.task_id "
        "AND json_extract(e.payload,'$.actor')=r.actor AND json_extract(e.payload,'$.profile')=r.profile "
        "AND json_extract(e.payload,'$.idempotency_key')=r.idempotency_key "
        "AND json_extract(e.payload,'$.request_digest')=r.request_digest "
        "WHERE r.project_id=? AND r.operation='owner_project_plan_commit' AND r.status='committed' "
        "AND r.updated_at>=? ORDER BY r.updated_at DESC,e.id DESC,r.rowid DESC LIMIT ?",
        (max_bytes, project.id, max_bytes, max_bytes, *params, PLAN_RECEIPT_LIMIT),
    ).fetchall()
    result = []
    for row in rows:
        try:
            identity = json.loads(row["identity_json"])
            reference = _ref(identity)
            receipt = owner_workspace.decode_committed_owner_receipt(row["result_json"], "owner_project_plan_commit")
            if (identity[4] != project.board_slug or receipt.get("project_id") != project.id
                    or receipt.get("board") != project.board_slug or type(receipt.get("applied")) is not bool
                    or not isinstance(row["updated_at"], int) or row["updated_at"] < 0):
                raise ValueError("review_plan_receipt_scope_invalid")
            item = {"id": reference, "at": row["updated_at"],
                    "state": "applied" if receipt["applied"] else "not_applied"}
            if receipt["applied"]:
                if row["event_count"] != 1 or identity[5] != receipt.get("anchor_task_id"):
                    raise ValueError("review_plan_event_invalid")
                event = json.loads(row["event_payload"])
                applied = event.get("result")
                if (not isinstance(applied, dict) or applied.get("applied") is not True
                        or any(receipt.get(key) != value for key, value in applied.items())
                        or event.get("trigger") not in owner_workspace._PROJECT_PLAN_TRIGGERS):
                    raise ValueError("review_plan_event_invalid")
                for field, limit in (("plan_summary", PLAN_TEXT_LIMIT), ("current_milestone", 1000)):
                    if not isinstance(event.get(field), str) or not event[field].strip() or len(event[field]) > limit:
                        raise ValueError("review_plan_event_text_invalid")
                item.update({"event_id": row["event_id"], "summary": event["plan_summary"],
                             "current_milestone": event["current_milestone"], "trigger": event["trigger"],
                             "created_task_ids": _ids(receipt.get("created_task_ids")),
                             "affected_task_ids": _ids(receipt.get("affected_task_ids"))})
                item["changes"] = _changes(registry, project, identity, row["event_id"], receipt, max_bytes)
            elif row["event_count"]:
                raise ValueError("review_plan_event_invalid")
            result.append(item)
        except (ValueError, TypeError, KeyError, owner_workspace.OwnerReceiptUnreadable) as exc:
            raise ValueError("review_plan_receipt_unavailable") from exc
    return result, total


def verified_supersessions(registry, project, board_path, frozen_plans, *, max_bytes):
    """Match frozen change references to committed receipts before classification."""
    candidates = [item for item in frozen_plans if item.get("changes")]
    if not candidates:
        return {}
    # SQLite binds timestamps as signed 64-bit integers. Validate untrusted values before binding them.
    if any(type(item.get("at")) is not int or not 0 <= item["at"] <= 2**63 - 1 for item in candidates):
        raise ValueError("review_plan_evidence_changed")
    current, _total = read_plan_receipts(
        registry, project, board_path, min(item["at"] for item in candidates), max_bytes=max_bytes,
    )
    current = {item["id"]: item for item in current}
    proven = {}
    for item in candidates:
        live = current.get(item["id"])
        if (live is None or live.get("state") != "applied" or item.get("state") != "applied"
                or live.get("at") != item["at"]
                or live.get("event_id") != item.get("event_id") or live.get("changes") != item["changes"]):
            raise ValueError("review_plan_evidence_changed")
        for change in item["changes"]:
            if change["action"] in SUPERSEDING_ACTIONS:
                proven.setdefault(change["task_id"], item["id"])
    return proven
