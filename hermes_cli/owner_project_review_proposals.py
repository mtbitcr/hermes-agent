"""Validate plan advice without granting permission to change Project work."""

from hermes_cli import kanban_db, owner_workspace
from hermes_cli.owner_text import check_owner_text


_TEXT_LIMITS = {
    "request_title": 240, "summary": 2000, "specification": 20000,
    "current_milestone": 1000, "owner_visible_result": 1000,
}
_CHAT_PREFIX = "Ask Raphael in the Project chat: "
# Two closed lists decide whether advice may reopen the owner's decision on work. A kind on neither is not evidence.
# The owner's decisions on a work item: an applied plan change, its creation by an approved plan (a creation counts only
# when that plan's applied record names the item, see _settled), a direct move, an archive and a suggestion decision.
_DECISION_EVENTS = ("owner_project_plan_change", "created", "owner_move", "archived", "recommendation_decided")
# Evidence about the work: a result of work or of a review, a delivery outcome, and a comment that is not the plan's own.
_EVIDENCE_EVENTS = ("completed", "blocked", "gave_up", "review_requested", "changes_requested", "review_reopened",
                    "delivery_merged", "merged", "released", "commented")
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
    """Whether the frozen input holds no evidence of this work newer than the owner's latest decision on it.

    The latest decision is the newest event of a decision kind, of any action and age. Ended work without a recorded
    decision is settled as a whole.
    """
    decision = board.execute(
        "SELECT id,kind,CASE WHEN json_valid(payload) THEN json_extract(payload,'$.idempotency_key') END AS plan "
        f"FROM task_events WHERE task_id=? AND kind IN ({','.join('?' * len(_DECISION_EVENTS))}) "
        # A creation is a decision only when an approved plan made the item: the plan's applied record on the Project's
        # control row names it among the work it created, alike for add and for replace, split and merge. A kernel's
        # rework item and earlier work have no such record. An activation is no proof: it also releases resumed work.
        "AND (kind<>'created' OR EXISTS (SELECT 1 FROM tasks work JOIN tasks anchor ON anchor.project_id=work.project_id "
        "AND anchor.task_kind='control' JOIN task_events applied ON applied.task_id=anchor.id "
        "AND applied.kind='owner_project_plan_applied' JOIN json_each(CASE WHEN json_valid(applied.payload) "
        "THEN applied.payload END,'$.result.created_task_ids') named ON named.value=work.id WHERE work.id=?)) "
        "ORDER BY id DESC LIMIT 1",
        (task_id, *_DECISION_EVENTS, task_id),
    ).fetchone()
    if decision is None and ended:
        return True
    latest = decision["id"] if decision is not None else 0
    own = None
    if decision is not None and decision["kind"] == "owner_project_plan_change" and isinstance(decision["plan"], str):
        # A move back to ready also writes the approved reason as that plan's own comment, in the change's transaction:
        # the first comment event after the change. Only that event is not evidence; any later comment is.
        prefix = f"owner-plan:{decision['plan']}:change:"
        if board.execute(
            "SELECT 1 FROM task_comments WHERE task_id=? AND substr(operation_key,1,?)=? AND operation_key LIKE '%:resume'",
            (task_id, len(prefix), prefix),
        ).fetchone():
            own = board.execute("SELECT MIN(id) FROM task_events WHERE task_id=? AND kind='commented' AND id>?",
                                (task_id, latest)).fetchone()[0]
    return not any(event["id"] > latest and event["kind"] in _EVIDENCE_EVENTS and event["id"] != own for event in events)


def _equivalent_created_work(changes, sources, board, project):
    # The content-link rule of the closures: token-set similarity of at least 0.8.
    from hermes_cli.owner_project_review import _same_work

    frozen = {item["id"]: item.get("events", []) for item in sources["tasks"]}
    # The input holds the latest event of each suggestion, enough to show an event after its decision. Its kind decides.
    for item in sources.get("suggestions", []):
        frozen[item["id"]] = [dict(row) for row in board.execute(
            "SELECT id,kind FROM task_events WHERE id=? AND task_id=?", (item.get("event_revision", 0), item["id"]))]
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

    def works(change):
        refs = change["targets"] if "targets" in change else [change["target"]] if "target" in change else []
        specs = [change] if change["action"] == "add" else change.get("replacements", [change.get("replacement")])
        return [titles[ref["task_id"]] for ref in refs], [spec["title"] for spec in specs if spec]

    def names(values, keep, width):
        shown = [title if len(title) <= width else title[:width - 1].rstrip() + "…" for title in values[:keep]]
        shown += [f"{len(values) - keep} more"] if len(values) > keep else []
        return ", ".join(shown[:-1]) + " and " + shown[-1] if len(shown) > 1 else "".join(shown)

    def phrase(change, keep=None, width=60):
        old, new = works(change)
        return _PHRASES[change["action"]].format(old=names(old, keep or len(old), width),
                                                 new=names(new, keep or len(new), width), status=change.get("to_status"))

    def hint(phrases, rest):
        more = f"; and {rest} more change{'s' if rest > 1 else ''}" if rest else ""
        return "Ask Raphael in the Project chat to " + "; ".join(phrases) + more + "."

    # The longest run of whole changes that fits, then a count of the rest.
    for shown in range(len(changes), 0, -1):
        text = hint([phrase(change) for change in changes[:shown]], len(changes) - shown)
        if len(text) <= limit:
            return text
    # The first change alone keeps its action: first its list of titles is cut to a count, then the title itself.
    first, rest = changes[0], len(changes) - 1
    size = max(map(len, works(first)))
    for keep, width in [(keep, 60) for keep in range(size - 1, 0, -1)] + [(1, width) for width in range(59, 1, -1)]:
        text = hint([phrase(first, keep, width)], rest)
        if len(text) <= limit:
            return text
    return text


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
