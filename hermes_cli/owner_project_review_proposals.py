"""Validate plan advice without granting permission to change Project work."""

import json

from hermes_cli import kanban_db, owner_workspace
from hermes_cli.owner_text import check_owner_text


_TEXT_LIMITS = {
    "request_title": 240, "summary": 2000, "specification": 20000,
    "current_milestone": 1000, "owner_visible_result": 1000,
}
_CHAT_PREFIX = "Ask Raphael in the Project chat: "


def proposal_schema(*, limit):
    """Use the native argument schema as data, without adding a worker tool."""
    from tools import owner_workspace_tools

    native = owner_workspace_tools.registry.get_schema("owner_project_plan_commit")["parameters"]["properties"]
    names = list(_TEXT_LIMITS) + ["later_milestones", "changes"]
    properties = {name: dict(native[name]) for name in names}
    for name, maximum in _TEXT_LIMITS.items():
        properties[name]["maxLength"] = maximum
    # The chat instruction and its title share the existing next-step budget.
    properties["request_title"]["maxLength"] = min(_TEXT_LIMITS["request_title"], limit - len(_CHAT_PREFIX))
    return {"type": "object", "additionalProperties": False, "required": names, "properties": properties}


def _already_applied(action, ref, board, registry, project, now, max_bytes):
    event = board.execute(
        "SELECT created_at,CASE WHEN length(CAST(payload AS BLOB))<=? THEN payload END AS payload "
        "FROM task_events WHERE id=? AND task_id=? AND kind='owner_project_plan_change'",
        (max_bytes, ref["expected_revision"], ref["task_id"]),
    ).fetchone()
    if event is None or event["created_at"] < now - 30 * 86400:
        return False
    payload = json.loads(event["payload"]) if event["payload"] is not None else None
    if not isinstance(payload, dict):
        raise ValueError("review_plan_proposal_receipt_unavailable")
    if payload.get("action") != action:
        return False
    identity = [payload.get(name) for name in ("actor", "profile", "idempotency_key", "request_digest")]
    if any(not isinstance(item, str) or not item for item in identity):
        raise ValueError("review_plan_proposal_receipt_unavailable")
    row = registry.execute(
        "SELECT CASE WHEN length(CAST(result_json AS BLOB))<=? THEN result_json END AS result_json "
        "FROM owner_workspace_receipts WHERE actor=? AND profile=? AND idempotency_key=? AND request_digest=? "
        "AND project_id=? AND operation='owner_project_plan_commit' AND status='committed'",
        (max_bytes, *identity, project.id),
    ).fetchone()
    if row is None:
        return False
    try:
        receipt = owner_workspace.decode_committed_owner_receipt(row["result_json"], "owner_project_plan_commit")
    except owner_workspace.OwnerReceiptUnreadable as exc:
        raise ValueError("review_plan_proposal_receipt_unavailable") from exc
    affected = receipt.get("affected_task_ids")
    if (receipt.get("applied") is not True or receipt.get("project_id") != project.id
            or receipt.get("board") != project.board_slug or not isinstance(affected, list) or ref["task_id"] not in affected):
        raise ValueError("review_plan_proposal_receipt_unavailable")
    return True


def _equivalent_created_work(changes, board, project):
    # The content-link rule of the closures: token-set similarity of at least 0.8.
    from hermes_cli.owner_project_review import _same_work

    proposed_titles = []
    for change in changes:
        action = change["action"]
        specs = ([change] if action == "add" else change["replacements"] if action == "split"
                 else [change["replacement"]] if action in {"replace", "merge"} else [])
        excluded = {ref["task_id"] for ref in change.get("targets", [])}
        if "target" in change:
            excluded.add(change["target"]["task_id"])
        for spec in specs:
            if any(_same_work(spec["title"], title) for title in proposed_titles):
                return True
            proposed_titles.append(spec["title"])
            for row in board.execute(
                "SELECT id,task_kind,title,recommendation_label FROM tasks WHERE project_id=? "
                "AND status!='archived' AND task_kind IN ('work','recommendation')", (project.id,),
            ):
                label = row["recommendation_label"] if row["task_kind"] == "recommendation" else row["title"]
                if row["id"] not in excluded and _same_work(spec["title"], label):
                    return True
    return False


def proposal_hint(value, sources, board, registry, project, *, now, limit, max_bytes):
    """Render checked advice for the existing brief and summary fields."""
    if value is None:
        return None
    if not isinstance(value, dict) or set(value) != set(_TEXT_LIMITS) | {"later_milestones", "changes"}:
        raise ValueError("review_plan_proposal_invalid")
    try:
        texts = {name: owner_workspace._bounded_text(value[name], name, limit=size)
                 for name, size in _TEXT_LIMITS.items()}
        owner_workspace._normalize_later_milestones(value["later_milestones"])
        changes, _risk = owner_workspace._normalize_project_changes(value["changes"])
    except (owner_workspace.OwnerWorkspaceError, TypeError, KeyError, AttributeError) as exc:
        raise ValueError("review_plan_proposal_invalid") from exc
    frozen = {item["id"]: item for item in sources["tasks"]}
    mutations = []
    for change in changes:
        targets = list(change.get("targets", []))
        if "target" in change:
            targets.append(change["target"])
        references = list(change.get("existing_parents", [])) + targets
        mutations.extend((change["action"], ref) for ref in targets)
        for ref in references:
            item = frozen.get(ref["task_id"])
            live = board.execute(
                "SELECT status FROM tasks WHERE id=? AND project_id=? AND task_kind='work'",
                (ref["task_id"], project.id),
            ).fetchone()
            if (item is None or live is None or item["status"] != live["status"]
                    or ref.get("expected_status", item["status"]) != item["status"]
                    or ref.get("expected_revision", item["event_revision"]) != item["event_revision"]
                    or kanban_db.task_event_revision(board, ref["task_id"]) != item["event_revision"]):
                raise ValueError("review_plan_proposal_stale")
    # A later event is new evidence; without one, the approved target and action must not be proposed again.
    if any(_already_applied(action, ref, board, registry, project, now, max_bytes) for action, ref in mutations):
        return None
    if _equivalent_created_work(changes, board, project):
        return None
    hint = _CHAT_PREFIX + texts["request_title"]
    if len(hint) > limit or check_owner_text(hint, "planner_prose") != hint:
        raise ValueError("review_plan_proposal_text_invalid")
    return hint
