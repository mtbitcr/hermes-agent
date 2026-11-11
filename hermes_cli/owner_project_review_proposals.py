"""Validate plan advice without granting permission to change Project work."""

from hermes_cli import kanban_db, owner_workspace
from hermes_cli.owner_text import check_owner_text


_TEXT_LIMITS = {
    "request_title": 240, "summary": 2000, "specification": 20000,
    "current_milestone": 1000, "owner_visible_result": 1000,
}
_CHAT_PREFIX = "Ask Raphael in the Project chat: "
# Events an approved plan writes on its own work: they record that decision and are never new evidence against it.
_PLAN_EVENTS = {"owner_project_plan_change", "model_route_repinned", "owner_work_activated", "owner_work_parked",
                "promoted", "linked"}
# The owner's decision on work that has ended: a plan change, a direct move, an archive or a suggestion decision.
_ENDED_DECISIONS = ("owner_project_plan_change", "owner_move", "archived", "recommendation_decided")
_PHRASES = {"add": "add {new}", "replace": "replace {old} with {new}", "split": "split {old} into {new}",
            "merge": "merge {old} into {new}", "move": "move {old} to {status}", "postpone": "postpone {old}",
            "cancel": "cancel {old}"}


def proposal_schema(*, limit):
    """Use the native argument schema as data, without adding a worker tool."""
    from tools import owner_workspace_tools

    native = owner_workspace_tools.registry.get_schema("owner_project_plan_commit")["parameters"]["properties"]
    names = list(_TEXT_LIMITS) + ["later_milestones", "changes"]
    properties = {name: dict(native[name]) for name in names}
    for name, maximum in _TEXT_LIMITS.items():
        properties[name]["maxLength"] = maximum
    # The request title keeps its share of the next-step budget, so the schema in the review task stays as it was.
    properties["request_title"]["maxLength"] = min(_TEXT_LIMITS["request_title"], limit - len(_CHAT_PREFIX))
    return {"type": "object", "additionalProperties": False, "required": names, "properties": properties}


def _settled(board, task_id, events, *, ended):
    """Whether the owner's latest decision on this work has no newer event of the same work in the frozen input.

    On live work the decision is an applied plan change, of any action and age. Ended work without a recorded
    decision is settled as a whole.
    """
    kinds = _ENDED_DECISIONS if ended else _ENDED_DECISIONS[:1]
    decision = board.execute(
        "SELECT id,CASE WHEN json_valid(payload) THEN json_extract(payload,'$.idempotency_key') END AS plan "
        f"FROM task_events WHERE task_id=? AND kind IN ({','.join('?' * len(kinds))}) ORDER BY id DESC LIMIT 1",
        (task_id, *kinds),
    ).fetchone()
    if decision is None:
        return ended
    newer = [event["kind"] for event in events if event["id"] > decision["id"] and event["kind"] not in _PLAN_EVENTS]
    # A move back to ready also writes the approved reason as that plan's own comment, the first one after the change.
    prefix = f"owner-plan:{decision['plan']}:change:"
    if "commented" in newer and isinstance(decision["plan"], str) and board.execute(
        "SELECT 1 FROM task_comments WHERE task_id=? AND substr(operation_key,1,?)=? AND operation_key LIKE '%:resume'",
        (task_id, len(prefix), prefix),
    ).fetchone():
        newer.remove("commented")
    return not newer


def _equivalent_created_work(changes, sources, board, project):
    # The content-link rule of the closures: token-set similarity of at least 0.8.
    from hermes_cli.owner_project_review import _same_work

    frozen = {item["id"]: item.get("events", []) for item in sources["tasks"]}
    # The input holds the latest event of each suggestion, enough to show an event after its decision.
    frozen.update((item["id"], [{"id": item.get("event_revision", 0), "kind": "suggestion"}])
                  for item in sources.get("suggestions", []))
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
                "SELECT id,task_kind,status,title,recommendation_label FROM tasks WHERE project_id=? "
                "AND task_kind IN ('work','recommendation')", (project.id,),
            ):
                label = row["recommendation_label"] if row["task_kind"] == "recommendation" else row["title"]
                if row["id"] in excluded or not _same_work(spec["title"], label):
                    continue
                # Live work is a plain duplicate; archived work stays the owner's settled decision.
                if row["status"] != "archived" or _settled(board, row["id"], frozen.get(row["id"], []), ended=True):
                    return True
    return False


def _render(changes, sources, *, limit):
    """Name each checked change and its work: the owner title of the frozen input, or the proposed title."""
    titles = {item["id"]: item["title"] for item in sources["tasks"]}

    def names(values):
        shown = [title if len(title) <= 60 else title[:59].rstrip() + "…" for title in values]
        return ", ".join(shown[:-1]) + " and " + shown[-1] if len(shown) > 1 else "".join(shown)

    phrases = []
    for change in changes:
        refs = change["targets"] if "targets" in change else [change["target"]] if "target" in change else []
        specs = [change] if change["action"] == "add" else change.get("replacements", [change.get("replacement")])
        phrases.append(_PHRASES[change["action"]].format(
            old=names([titles[ref["task_id"]] for ref in refs]),
            new=names([spec["title"] for spec in specs if spec]), status=change.get("to_status")))
    # The longest run of whole changes that fits, then a count of the rest.
    for shown in range(len(phrases), 0, -1):
        rest = len(phrases) - shown
        more = f"; and {rest} more change{'s' if rest > 1 else ''}" if rest else ""
        hint = "Ask Raphael in the Project chat to " + "; ".join(phrases[:shown]) + more + "."
        if len(hint) <= limit:
            return hint
    count = f"the {len(phrases)} suggested plan changes" if len(phrases) > 1 else "the suggested plan change"
    return f"Ask Raphael in the Project chat to make {count}."


def proposal_hint(value, sources, board, registry, project, *, now, limit, max_bytes):
    """Render checked advice for the existing brief and summary fields."""
    if value is None:
        return None
    if not isinstance(value, dict) or set(value) != set(_TEXT_LIMITS) | {"later_milestones", "changes"}:
        raise ValueError("review_plan_proposal_invalid")
    try:
        for name, size in _TEXT_LIMITS.items():
            owner_workspace._bounded_text(value[name], name, limit=size)
        owner_workspace._normalize_later_milestones(value["later_milestones"])
        changes, _risk = owner_workspace._normalize_project_changes(value["changes"])
    except (owner_workspace.OwnerWorkspaceError, TypeError, KeyError, AttributeError) as exc:
        raise ValueError("review_plan_proposal_invalid") from exc
    frozen = {item["id"]: item for item in sources["tasks"]}
    mutated = []
    for change in changes:
        targets = list(change.get("targets", []))
        if "target" in change:
            targets.append(change["target"])
        references = list(change.get("existing_parents", [])) + targets
        mutated.extend(ref["task_id"] for ref in targets)
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
    # One gate: a settled decision on the affected work reopens only with a newer event of that work in the input.
    if any(_settled(board, task_id, frozen[task_id].get("events", []), ended=False) for task_id in mutated):
        return None
    if _equivalent_created_work(changes, sources, board, project):
        return None
    hint = _render(changes, sources, limit=limit)
    if len(hint) > limit or check_owner_text(hint, "planner_prose") != hint:
        raise ValueError("review_plan_proposal_text_invalid")
    return hint
