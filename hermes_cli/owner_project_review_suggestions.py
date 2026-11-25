"""The daily review's rules for suggestions: the closures a result proves for delivered suggestions, and the
suggestions its plan advice plans."""

from __future__ import annotations

from typing import Any

from hermes_cli import kanban_db

_MAX_CLOSURE_PROPOSALS = 3


def check_closure_shapes(proposals: Any) -> None:
    """The closure list size, each proposal's keys and its reason text, without the evidence only the harvest checks."""
    from hermes_cli.owner_project_review import _CLOSURE_REASON_LIMIT, ReviewRefused, _review_texts

    if not isinstance(proposals, list) or len(proposals) > _MAX_CLOSURE_PROPOSALS:
        raise ReviewRefused("review_closure_list_invalid")
    for index, proposal in enumerate(proposals):
        if (not isinstance(proposal, dict)
                or set(proposal) != {"recommendation_id", "card_id", "reason", "evidence"}
                or not isinstance(proposal["recommendation_id"], str)
                or not isinstance(proposal["card_id"], str)):
            raise ReviewRefused("review_closure_keys_invalid")
        _review_texts({"reason": proposal["reason"]}, {"reason": _CLOSURE_REASON_LIMIT}, "", f"proposed_closures[{index}]")


def _names_input_items(sources, tid, refs) -> bool:
    """Whether refs is a list of at most 8 references that each name an item of the frozen input, by the check of the classification evidence."""
    from hermes_cli.owner_project_review import ReviewRefused, check_claims

    if not isinstance(refs, list) or len(refs) > 8:
        return False
    try:
        check_claims(sources, [{"item": tid, "class": "superseded", "evidence": refs}])
    except ReviewRefused as refused:
        return refused.code == "review_class_coverage_invalid"  # the references are checked before coverage, which one claim may miss
    return True


def _review_closures(board, project_id, sources, classification, proposals, frozen=()) -> list[dict]:
    """The result's closures its classification proves. Without a board, a stored review's closures are read again:
    the closure step then checks the live state in its own board transaction. A pair in ``frozen``, a required one
    whose proof changed since the freeze, gets only the frozen checks: the fenced read makes it stale (owner rule 4)."""
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
                or not _names_input_items(sources, tid, proposal["evidence"]) or not set(evidence) <= set(proposal["evidence"])
                or classes.get(tid, {}).get("class") != "superseded"
                or f"tasks:{card_id}" not in classes[tid]["evidence"]
                # A pending suggestion, or an accepted one whose own linked card is the proposed delivery.
                or suggestions[tid]["decision"] not in ("pending", "accepted")
                or (suggestions[tid]["decision"] == "accepted" and suggestions[tid].get("linked_card") != card_id)
                or (suggestions[tid].get("closure") and suggestions[tid]["closure"].get("undone_at") is None)):
            raise ReviewRefused("review_closure_evidence_required")
        delivery_state, lifecycle = cards[card_id].get("delivery_state"), suggestions[tid]
        if board is not None and (tid, card_id) not in frozen:
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


def closure_rule_published(body) -> bool:
    """Owner rule 3: whether the review task published step 7's rule sentence. Only the kernel's instruction, ahead of
    the data, counts."""
    from hermes_cli.owner_project_review import _CLOSURE_RULE

    return _CLOSURE_RULE in (body or "").partition("<<<BEGIN UNTRUSTED DATA")[0]


def check_required_proof_current(board, project_id, sources, proposals, *, checked=False) -> dict:
    """Owner rule 3, case two: the result's required proposals of a published rule whose pair's proof changed since the
    freeze, each pair with its stale code; such a result is stale, not the worker's fault. Owner rule 4: the parse
    judges these pairs on their frozen proof, and the fenced read makes the result stale after every ordinary check.
    Each other proposal, and each pair not proposed, keeps the existing checks.

    A raw proposal is the required one when the list passes its shape check, its pair is one the rule sentence requires
    of the frozen input, its suggestion appears once and its evidence passes the frozen checks of _review_closures;
    ``checked`` closures passed those checks at the parse and are selected by their pair."""
    from hermes_cli.owner_project_review import ReviewRefused
    from hermes_cli.owner_project_review_record import card_delivery_state

    suggestions = {item["id"]: item for item in sources["suggestions"]}
    cards = {item["id"]: item for item in sources["tasks"]}
    # The rule sentence's pairs; owner rule 1: no closure is required of a suggestion whose input item holds one.
    required = {(item["id"], item["linked_card"]) for item in sources["suggestions"]
                if item["decision"] == "accepted" and not item.get("closure") and item.get("linked_card") in cards
                and cards[item["linked_card"]].get("delivery_state") is not None}
    changed = {}
    if not checked:
        try:
            check_closure_shapes(proposals)
        except ReviewRefused:
            return changed
    tids = [proposal["recommendation_id"] for proposal in proposals]
    for proposal in proposals:
        tid, card_id = proposal["recommendation_id"], proposal["card_id"]
        if (tid, card_id) not in required or tids.count(tid) != 1:
            continue
        if not checked:
            try:
                # The class the kernel gives a required pair at the freeze.
                _review_closures(None, project_id, sources,
                                 [{"item": tid, "class": "superseded", "evidence": [f"tasks:{card_id}"]}], [proposal])
            except ReviewRefused:
                continue
        # The decision cursor covers only pending suggestions: an accepted one's close or Undo moves only its revision.
        if kanban_db.task_event_revision(board, tid) != suggestions[tid].get("event_revision"):
            changed[(tid, card_id)] = "review_recommendation_state_changed"
            continue
        # The frozen reader's own row scope, so an unchanged card with no Project is not called changed.
        row = board.execute(
            "SELECT status FROM tasks WHERE id=? AND task_kind='work' AND (project_id=? OR project_id IS NULL)",
            (card_id, project_id),
        ).fetchone()
        if (row is None or kanban_db.task_event_revision(board, card_id) != cards[card_id].get("event_revision")
                or card_delivery_state(board, card_id, row["status"]) != cards[card_id]["delivery_state"]):
            changed[(tid, card_id)] = "review_delivery_state_changed"
    return changed


def check_required_closures(board, project_id, sources, classification, closures, *, published=True,
                            frozen=()) -> None:
    """Refuse a result that leaves out a required closure. An eligible pair is an accepted suggestion of the frozen input
    with no closure, live or undone, and its linked card whose closure passes every check of _review_closures; the
    result's checked closures must hold one for each, or for as many as the cap allows. Owner rule 3, case one: when
    the task did not publish the rule (``published`` false), that missing closure makes the result stale instead.
    Owner rule 4: a proposed pair in ``frozen``, whose proof changed since the freeze, counts as at the freeze."""
    from hermes_cli.owner_project_review import ReviewRefused, ReviewStale

    eligible = set()
    for item in sources["suggestions"]:
        # Owner rule of 2026-11-25: after the owner's Undo a proposal stays possible, but none is required.
        if item["decision"] != "accepted" or not item.get("linked_card") or item.get("closure"):
            continue
        # The closure the review must propose for this pair, checked as a proposed one is.
        required = {"recommendation_id": item["id"], "card_id": item["linked_card"],
                    "reason": "The linked card delivered this suggestion.",
                    "evidence": [f"suggestions:{item['id']}", f"tasks:{item['linked_card']}"]}
        try:
            _review_closures(board, project_id, sources, classification, [required], frozen=frozen)
        except ReviewRefused:
            continue
        eligible.add(item["id"])
    proposed = {closure["recommendation_id"] for closure in closures}
    if len(eligible & proposed) < min(len(eligible), _MAX_CLOSURE_PROPOSALS):
        if not published:
            raise ReviewStale("review_closure_rule_unpublished")
        raise ReviewRefused("review_delivered_closure_missing")


def needs_no_card(board, project_id, item) -> bool:
    """Whether a frozen suggestion item needs no card: an accepted model update, which its own Update applies. The
    input's needs_card mark decides; an item with no mark, as in an input made before the mark, is decided from its
    board row, and a missing row needs a card."""
    from hermes_cli.owner_model_updates import parse_subject

    if "needs_card" in item:
        return item["needs_card"] is False
    if item.get("decision") != "accepted":
        return False
    row = board.execute(
        "SELECT * FROM tasks WHERE id=? AND project_id=? AND task_kind='recommendation'", (item["id"], project_id),
    ).fetchone()
    return row is not None and parse_subject(row["recommendation_kind"], row["recommendation_subject_id"]) is not None


def check_plan_suggestions(value, sources, board, registry, project) -> None:
    """The result's plan_suggestions against the frozen input and the live Project board.

    Each pair names an add change of the checked plan_proposal and a suggestion of the frozen input that is accepted
    or pending, has no live closure, no linked card and no needs_card false (an accepted model update, which its own
    Update applies; with no mark, its board row decides); no suggestion and no change appears twice, and a list without
    a plan is refused. A suggestion whose reviewed plan the owner denied is not planned again while the brief history
    holds that pair (origin._denied_plan). The pairs are only the review's proposal: a link is written by the kernel,
    in the owner's exactly approved plan commit.
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
        # An accepted model update, by the input's mark or, with no mark, its board row: its own Update applies it.
        if needs_no_card(board, project.id, suggestions[tid]):
            raise ReviewRefused("review_plan_suggestion_needs_no_card")
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
