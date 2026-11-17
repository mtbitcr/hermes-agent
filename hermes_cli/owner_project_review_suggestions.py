"""The daily review's rules for suggestions: the closures a result proves for delivered suggestions, and the
suggestions its plan advice plans."""

from __future__ import annotations

from typing import Any

from hermes_cli import kanban_db


def check_closure_shapes(proposals: Any) -> None:
    """The closure list size, each proposal's keys and its reason text, without the evidence only the harvest checks."""
    from hermes_cli.owner_project_review import _CLOSURE_REASON_LIMIT, ReviewRefused, _review_texts

    if not isinstance(proposals, list) or len(proposals) > 3:
        raise ReviewRefused("review_closure_list_invalid")
    for index, proposal in enumerate(proposals):
        if (not isinstance(proposal, dict)
                or set(proposal) != {"recommendation_id", "card_id", "reason", "evidence"}
                or not isinstance(proposal["recommendation_id"], str)
                or not isinstance(proposal["card_id"], str)):
            raise ReviewRefused("review_closure_keys_invalid")
        _review_texts({"reason": proposal["reason"]}, {"reason": _CLOSURE_REASON_LIMIT}, "", f"proposed_closures[{index}]")


def _review_closures(board, project_id, sources, classification, proposals) -> list[dict]:
    """The result's closures its classification proves. Without a board, a stored review's closures are read again:
    the closure step then checks the live state in its own board transaction."""
    from hermes_cli.owner_project_review import ReviewRefused, _delivered_match

    check_closure_shapes(proposals)
    suggestions = {item["id"]: item for item in sources["suggestions"]}
    cards = {item["id"]: item for item in sources["tasks"]}
    classes = {item["item"]: item for item in classification}
    checked = []
    seen = set()
    for proposal in proposals:
        tid, card_id = proposal["recommendation_id"], proposal["card_id"]
        evidence = [f"suggestions:{tid}", f"tasks:{card_id}"]
        if (tid in seen or tid not in suggestions or card_id not in cards
                or not isinstance(proposal["evidence"], list) or len(proposal["evidence"]) != 2
                or not all(isinstance(ref, str) for ref in proposal["evidence"])
                or set(proposal["evidence"]) != set(evidence)
                or classes.get(tid, {}).get("class") != "superseded"
                or f"tasks:{card_id}" not in classes[tid]["evidence"]
                # A pending suggestion, or an accepted one whose own linked card is the proposed delivery.
                or suggestions[tid]["decision"] not in ("pending", "accepted")
                or (suggestions[tid]["decision"] == "accepted" and suggestions[tid].get("linked_card") != card_id)
                or (suggestions[tid].get("closure") and suggestions[tid]["closure"].get("undone_at") is None)):
            raise ReviewRefused("review_closure_evidence_required")
        delivery_state, lifecycle = cards[card_id].get("delivery_state"), suggestions[tid]
        if board is not None:
            delivery_state = _delivered_match(board, project_id, tid, cards[card_id])
            if delivery_state is None:
                raise ReviewRefused("review_closure_evidence_required")
            row = board.execute(
                "SELECT * FROM tasks WHERE id=? AND project_id=? AND task_kind='recommendation'", (tid, project_id),
            ).fetchone()
            lifecycle = kanban_db.recommendation_lifecycle_snapshot(row)
            if any(lifecycle[key] != suggestions[tid].get(key) for key in lifecycle):
                raise ReviewRefused("review_recommendation_state_changed")
        # check_closure_shapes passed this reason through _review_texts, which returns it stripped.
        reason = proposal["reason"].strip()
        seen.add(tid)
        checked.append({"recommendation_id": tid, "card_id": card_id, "reason": reason,
                        "evidence": evidence, "card_state": delivery_state, "expected_lifecycle_version": lifecycle["lifecycle_version"]})
    return checked


def check_plan_suggestions(value, sources, board, registry, project) -> None:
    """The result's plan_suggestions against the frozen input and the live Project board.

    Each pair names an add change of the checked plan_proposal and a suggestion of the frozen input that is accepted
    or pending, has no live closure and no linked card; no suggestion and no change appears twice, and a list without
    a plan is refused. A suggestion whose reviewed plan the owner denied is not planned again. The pairs are only the
    review's proposal: a link is written by the kernel, in the owner's exactly approved plan commit.
    """
    from hermes_cli.owner_project_review import ReviewRefused
    from hermes_cli.owner_project_review_origin import _denied_plan
    from hermes_cli.owner_project_review_record import _table

    if "plan_suggestions" not in value:
        return
    pairs, plan = value["plan_suggestions"], value.get("plan_proposal")
    changes = plan.get("changes") if isinstance(plan, dict) else None
    if (not isinstance(pairs, list) or len(pairs) > 3 or not isinstance(changes, list)
            or not all(isinstance(pair, dict) and set(pair) == {"change", "suggestion"} and type(pair["change"]) is int
                       and isinstance(pair["suggestion"], str) for pair in pairs)):
        raise ReviewRefused("review_plan_suggestions_invalid")
    if len({pair["change"] for pair in pairs}) != len(pairs) or len({pair["suggestion"] for pair in pairs}) != len(pairs):
        raise ReviewRefused("review_plan_suggestion_duplicate")
    suggestions = {item["id"]: item for item in sources["suggestions"]}
    for pair in pairs:
        index, tid = pair["change"], pair["suggestion"]
        if not 0 <= index < len(changes) or not isinstance(changes[index], dict) or changes[index].get("action") != "add":
            raise ReviewRefused("review_plan_suggestion_change_invalid")
        if tid not in suggestions:
            raise ReviewRefused("review_plan_suggestion_unknown")
        closure = suggestions[tid].get("closure")
        if (suggestions[tid]["decision"] not in ("pending", "accepted") or (closure and closure.get("undone_at") is None)
                or kanban_db.has_live_recommendation_closure(board, tid)):
            raise ReviewRefused("review_plan_suggestion_not_open")
        # The frozen lineage, and a card made from the suggestion since the input was frozen.
        if suggestions[tid].get("linked_card") is not None or (_table(board, "recommendation_made_from") and board.execute(
                "SELECT 1 FROM recommendation_made_from WHERE recommendation_id=?", (tid,)).fetchone() is not None):
            raise ReviewRefused("review_plan_suggestion_linked")
    if pairs and _denied_plan(registry, project.id, plan, {pair["suggestion"] for pair in pairs}):
        raise ReviewRefused("review_plan_suggestion_denied")
