"""Plan an accepted suggestion in the daily review, and link its card at the owner's plan commit."""

import contextlib
import json
import sqlite3
from datetime import datetime, timezone

import pytest
import yaml

from hermes_cli import kanban_db, owner_project_review as review, owner_workspace, projects_db
from hermes_cli import owner_project_review_record as record, owner_project_summary_events as events
from tests.hermes_cli.test_owner_project_review_plans import _approved_plan
from tests.hermes_cli.test_owner_project_review_record import _decide, _suggestion, _written_at, root  # noqa: F401
from tests.hermes_cli.test_owner_workspace import _bootstrap_board, _commit_project_plan, _with_approver, ctx  # noqa: F401

NOW = int(datetime(2026, 11, 10, 7, tzinfo=timezone.utc).timestamp())
ROOM = "Reserve a meeting room"
HALL = "Book the conference hall"
HINT = f"Ask Raphael in the Project chat to add {HALL} for your accepted suggestion {ROOM}."
# A title equivalent to the accepted suggestion's label under the content-link rule of the closures.
SAME = "Reserve the meeting room"


def _project_board(project):
    return kanban_db.connect_closing(kanban_db.kanban_db_path(board=project.board_slug), board=project.board_slug)


def _summary_board():
    return kanban_db.connect_closing(kanban_db.kanban_db_path(board="project-summaries"), board="project-summaries")


@pytest.fixture
def world(root, ctx):
    """One Project with its first daily review open: an accepted suggestion with no card, an accepted one whose
    made-from card is done, a rejected one, and a plan-made card with a newer owner comment."""
    with _written_at(NOW - 86400):
        setup = _bootstrap_board(ctx)
        [booked] = _approved_plan(ctx, setup)["created_task_ids"]
    with projects_db.connect_closing(root / "projects.db") as registry:
        project = projects_db.get_project(registry, setup["project_id"])
    with _written_at(NOW - 86400), _project_board(project) as board:
        kanban_db.add_comment(board, booked, "default", "The venue needs confirmation.")
        accepted = _suggestion(board, project, ROOM, "meeting-room")
        linked = _suggestion(board, project, "Arrange the catering", "catering")
        rejected = _suggestion(board, project, "Hire a string quartet", "music")
        _decide(board, accepted, "accepted", "The visit needs a room.")
        _decide(board, linked, "accepted", "The guests need lunch.")
        _decide(board, rejected, "rejected", "The visit is short.")
        card = kanban_db.create_task(board, title="Order lunch for the guests", project_id=project.id,
                                     made_from_recommendation=linked)
        assert kanban_db.complete_task(board, card, summary="Lunch is ordered.")
        target = {"task_id": booked, "expected_status": kanban_db.get_task(board, booked).status,
                  "expected_revision": kanban_db.task_event_revision(board, booked)}
    config = {"owner_review": {"enabled": True, "allowlist": [project.id], "closure": True}}
    (root / "config.yaml").write_text(yaml.safe_dump(config))
    assert review.review_tick(NOW) == 1
    with projects_db.connect_closing(root / "projects.db") as registry:
        review_task = registry.execute(
            "SELECT open_task_id FROM project_review_state WHERE project_id=?", (project.id,),
        ).fetchone()[0]
    return {"root": root, "project": project, "review_task": review_task, "booked": booked, "target": target,
            "card": card, "accepted": accepted, "linked": linked, "rejected": rejected}


def _add(title):
    return {"action": "add", "reason": "The owner accepted the suggestion.", "title": title,
            "body": f"{title} for the visit.", "assignee": "default", "execution_tier": "routine", "risk_tier": 0,
            "existing_parents": [], "new_parents": []}


def _plan(*changes):
    return {"request_title": "Plan the meeting room", "summary": "Book a conference hall for the visit.",
            "specification": "Add one card that books a conference hall.",
            "current_milestone": "Prepare the first visit.", "owner_visible_result": "The visit has a conference hall.",
            "later_milestones": [], "changes": list(changes) or [_add(HALL)]}


def _result(world, *, plan=None, pairs=None, classes=None, closures=()):
    """A review result: every suggestion is answered unless ``classes`` names its class and evidence."""
    claims = {tid: ("answered", [f"suggestions:{tid}"]) for tid in (world["accepted"], world["linked"], world["rejected"])}
    claims.update(classes or {})
    value = {
        "summary": {"outcome": "The meeting room suggestion is accepted.", "next_action": f"{HALL}.",
                    "owner_decision": None},
        "brief": {"headline": "The meeting room has no card yet.", "what_changed": f"{ROOM} is accepted and has no card.",
                  "next_step": f"{HALL}.", "needs_you": None},
        "classification": [{"item": tid, "class": name, "evidence": evidence} for tid, (name, evidence) in claims.items()],
        "proposed_closures": list(closures),
    }
    if plan is not None:
        value["plan_proposal"] = plan
    if pairs is not None:
        value["plan_suggestions"] = pairs
    return value


def _complete(world, value):
    with _summary_board() as board:
        assert kanban_db.claim_task(board, world["review_task"]) is not None
        assert kanban_db.complete_task(board, world["review_task"], summary="Reviewed the visit.",
                                       metadata={"project_review": value})


def _rewrite(world, value):
    """The completed run's result, replaced in place: the next judgement of the same frozen input."""
    with _summary_board() as board:
        run = kanban_db.latest_run(board, world["review_task"])
        with kanban_db.write_txn(board):
            board.execute("UPDATE task_runs SET metadata=? WHERE id=?", (json.dumps({"project_review": value}), run.id))


def _parse(world):
    with _summary_board() as board:
        return review.parse_review_result(board, world["review_task"], project_id=world["project"].id,
                                          now=NOW + 1, stale_days=14)


def _commit(ctx, world, plan, *, key, choice="once"):
    """The owner's normal plan commit from the Project chat, approved or denied."""
    approver = _with_approver(ctx.session, choice)
    try:
        return _commit_project_plan(ctx, idempotency_key=key, project_id=world["project"].id, trigger="owner_request",
                                    **plan)
    finally:
        approver.join()


def _silence(ctx, world, plan, *, key):
    """The owner's normal plan commit that ends with no answer of the owner: its approval times out."""
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(owner_workspace, "_confirm", lambda *args, **kwargs: {"approved": False, "reason": "timeout"})
        return _commit_project_plan(ctx, idempotency_key=key, project_id=world["project"].id, trigger="owner_request",
                                    **plan)


def _links(world):
    with _project_board(world["project"]) as board:
        return sorted(tuple(row) for row in board.execute("SELECT card_id,recommendation_id FROM recommendation_made_from"))


def _work_titles(world):
    with _project_board(world["project"]) as board:
        return sorted(row[0] for row in board.execute(
            "SELECT title FROM tasks WHERE project_id=? AND task_kind='work'", (world["project"].id,)))


def _trigger(world, card):
    proposals = record.build_review_input(world["project"], events.Cursors(0, 0, 0), None)["sources"]["proposals"]
    [plan] = [item for item in proposals if item["created_task_ids"] == [card]]
    return plan["trigger"]


def _paired_hint(title):
    return f"Ask Raphael in the Project chat to add {title} for your accepted suggestion {ROOM}."


def _brief(world):
    """The Project's latest stored brief, read back from the root registry: its next step and accepted source."""
    with projects_db.connect_closing(world["root"] / "projects.db") as registry:
        return tuple(registry.execute(
            "SELECT next_step,source_task_id,source_run_id,source_digest FROM project_review_briefs WHERE project_id=? "
            "ORDER BY review_date DESC LIMIT 1", (world["project"].id,)).fetchone())


def _next_review(world):
    """News opens the next day's review: its task."""
    with _written_at(NOW + 3600), _project_board(world["project"]) as board:
        kanban_db.add_comment(board, world["booked"], "default", "The hall must seat forty guests.")
    assert review.review_tick(NOW + 86400) == 1
    with projects_db.connect_closing(world["root"] / "projects.db") as registry:
        return registry.execute("SELECT open_task_id FROM project_review_state WHERE project_id=?",
                                (world["project"].id,)).fetchone()[0]


def _recheck(world, plan, task, **pairs):
    """The hint the review rechecks for ``plan`` against the frozen input of ``task``; without ``pairs`` as the
    accepted-advice read calls it."""
    from hermes_cli.owner_project_review_proposals import proposal_hint

    with _summary_board() as board:
        sources, _cut = review._frozen_sources(kanban_db.get_task(board, task).body)
    with review._review_evidence_board(world["project"].id) as (board, registry, live):
        return proposal_hint(plan, sources, board, registry, live, now=NOW + 86400, limit=review._BRIEF_LIMITS["next_step"],
                             max_bytes=record.TOTAL_INPUT_CAP_BYTES, **pairs)


def _request(project_id, plan):
    """The digest of the owner's exact approval of ``plan``, built the way the owner commit builds it."""
    changes, risk = owner_workspace._normalize_project_changes(plan["changes"])
    return owner_workspace._digest({
        "project_id": project_id, "trigger": "owner_request",
        **{name: plan[name] for name in ("request_title", "summary", "specification", "current_milestone",
                                         "owner_visible_result")},
        "later_milestones": owner_workspace._normalize_later_milestones(plan["later_milestones"]),
        "risk_level": risk, "changes": changes})


def test_t1_a_result_with_a_valid_plan_suggestions_pair_is_accepted(world):
    _complete(world, _result(world, plan=_plan(), pairs=[{"change": 0, "suggestion": world["accepted"]}]))

    result = _parse(world)

    # The kernel's own sentence names the stored label of the paired suggestion.
    assert result.brief["next_step"] == result.summary["next_action"] == HINT
    assert result.source_run_id is not None and result.source_digest is not None
    assert review.review_tick(NOW + 1) == 0
    with projects_db.connect_closing(world["root"] / "projects.db") as registry:
        stored = registry.execute(
            "SELECT next_step,source_digest FROM project_review_briefs WHERE project_id=?", (world["project"].id,),
        ).fetchone()
    assert tuple(stored) == (HINT, result.source_digest)


def test_t1_a_pair_after_two_long_changes_keeps_its_phrase_in_the_bounded_hint(world):
    from hermes_cli.owner_project_review_proposals import _render

    titles = ("Arrange comfortable seating for the guests in the reception room",
              "Prepare fresh refreshments for visitors before the evening reception",
              "Organize suitable transport for visitors after the evening reception")
    plan = _plan(*(_add(title) for title in titles))
    _complete(world, _result(world, plan=plan, pairs=[{"change": 2, "suggestion": world["accepted"]}]))

    result = _parse(world)

    assert review.review_tick(NOW + 1) == 0
    with projects_db.connect_closing(world["root"] / "projects.db") as registry:
        shown = projects_db.get_project_review_brief(registry, world["project"].id)["next_step"]
    # The paired change comes first, with the stored label; the count names the change the bounded hint leaves out.
    assert shown == result.brief["next_step"] == (
        "Ask Raphael in the Project chat to add Organize suitable transport for visitors after the evening… "
        f"for your accepted suggestion {ROOM}; add Arrange comfortable seating for the guests in the reception…; "
        "and 1 more change.")
    assert f" for your accepted suggestion {ROOM}" in shown and len(shown) <= 240
    # The same plan without a pair keeps its hint as it was: its own order, and all three changes fit.
    assert _render(plan["changes"], {"tasks": []}, limit=240) == (
        "Ask Raphael in the Project chat to add Arrange comfortable seating for the guests in the reception…; "
        "add Prepare fresh refreshments for visitors before the evening…; "
        "add Organize suitable transport for visitors after the evening….")


@pytest.mark.parametrize("title", [ROOM, SAME])
def test_t1_a_pair_titled_like_its_own_accepted_suggestion_keeps_its_hint_and_its_accepted_advice(world, title):
    from hermes_cli.owner_project_review_origin import accepted_review_source
    from hermes_cli.owner_project_review_view import project_review_advice_view

    project, plan = world["project"], _plan(_add(title))
    _complete(world, _result(world, plan=plan, pairs=[{"change": 0, "suggestion": world["accepted"]}]))

    assert review.review_tick(NOW + 1) == 0

    # The real harvest keeps the paired hint with the stored label, and stores the accepted source.
    next_step, task, run, digest = _brief(world)
    assert next_step == _paired_hint(title)
    assert (task, run is not None, digest is not None) == (world["review_task"], True, True)
    assert accepted_review_source(project.id) is not None
    # The accepted-advice read rechecks the same hint with the pairs the harvest checked and stored.
    assert project_review_advice_view(project.id) == {"review_date": "2026-11-10", "proposal": plan}
    assert _recheck(world, plan, world["review_task"]) == _paired_hint(title)


def test_t1_a_same_label_pair_after_two_unrelated_changes_keeps_its_phrase_first(world):
    from hermes_cli.owner_project_review_view import project_review_advice_view

    plan = _plan(_add("Print the visitor badges"), _add("Check the reception desk"), _add(ROOM))
    _complete(world, _result(world, plan=plan, pairs=[{"change": 2, "suggestion": world["accepted"]}]))

    assert review.review_tick(NOW + 1) == 0

    next_step, task, _run, _digest = _brief(world)
    assert next_step == (f"Ask Raphael in the Project chat to add {ROOM} for your accepted suggestion {ROOM}; "
                         "add Print the visitor badges; add Check the reception desk.")
    assert task == world["review_task"]
    assert project_review_advice_view(world["project"].id) == {"review_date": "2026-11-10", "proposal": plan}


@pytest.mark.parametrize("case", ["unpaired_plan", "unpaired_change", "other_suggestion", "work_card"])
def test_t1_a_duplicate_outside_its_checked_pair_still_gets_no_hint(world, case):
    """Controls: only the paired suggestion of its own add change is exempt. Actual work, any other suggestion, a
    plan with no pair and a change with no pair keep the duplicate check."""
    from hermes_cli.owner_project_review_view import project_review_advice_view

    plan, pairs = _plan(_add(ROOM)), [{"change": 0, "suggestion": world["accepted"]}]
    if case == "unpaired_plan":
        pairs = None
    elif case == "unpaired_change":
        plan = _plan(_add(HALL), _add(SAME))
    elif case == "other_suggestion":
        # The label of the rejected suggestion of the frozen input.
        plan = _plan(_add("Hire a string quartet"))
    elif case == "work_card":
        # The title of the done card of the frozen input.
        plan = _plan(_add("Order lunch for the guests"))
    _complete(world, _result(world, plan=plan, pairs=pairs))

    result = _parse(world)

    assert result.brief["next_step"] == result.summary["next_action"] == f"{HALL}."
    assert result.source_run_id is None
    assert review.review_tick(NOW + 1) == 0
    assert _brief(world)[:2] == (f"{HALL}.", None)
    assert project_review_advice_view(world["project"].id) is None


@pytest.mark.parametrize(("case", "code"), [
    ("unknown", "review_plan_suggestion_unknown"),
    ("rejected", "review_plan_suggestion_not_open"),
    ("not_add", "review_plan_suggestion_change_invalid"),
    ("twice", "review_plan_suggestion_duplicate"),
    ("linked", "review_plan_suggestion_linked"),
    ("no_plan", "review_plan_suggestions_invalid"),
])
def test_t2_each_wrong_pair_is_refused_with_its_own_code(world, case, code):
    plan, pairs = _plan(), [{"change": 0, "suggestion": world["accepted"]}]
    if case == "unknown":
        pairs[0]["suggestion"] = "t_not_in_the_input"
    elif case in ("rejected", "linked"):
        pairs[0]["suggestion"] = world[case]
    elif case == "not_add":
        plan = _plan(_add(HALL), {"action": "postpone", "reason": "The venue is not confirmed.", "target": world["target"]})
        pairs[0]["change"] = 1
    elif case == "twice":
        plan = _plan(_add(HALL), _add("Print the name badges"))
        pairs.append({"change": 1, "suggestion": world["accepted"]})
    value = _result(world, plan=plan, pairs=pairs)
    if case == "no_plan":
        value["plan_proposal"] = None
    _complete(world, value)

    with pytest.raises(review.ReviewRefused) as refused:
        _parse(world)

    assert refused.value.code == code


def test_t3_the_frozen_input_marks_each_suggestion_with_its_linked_card(world):
    with _summary_board() as board:
        body = kanban_db.get_task(board, world["review_task"]).body
    sources, _cut = review._frozen_sources(body)

    linked = {item["id"]: item.get("linked_card", "absent") for item in sources["suggestions"]}
    assert linked == {world["accepted"]: None, world["linked"]: world["card"], world["rejected"]: None}
    # The instruction ahead of the data names the pair list.
    assert "plan_suggestions" in body.split("<<<BEGIN UNTRUSTED DATA")[0]
    # A board from before the lineage table: every suggestion reads null.
    path = kanban_db.kanban_db_path(board=world["project"].board_slug)
    with contextlib.closing(sqlite3.connect(path)) as board:
        board.execute("DROP TABLE recommendation_made_from")
        board.commit()
    frozen = record.build_review_input(world["project"], events.Cursors(0, 0, 0), None)
    assert {item["id"]: item.get("linked_card", "absent") for item in frozen["sources"]["suggestions"]} == dict.fromkeys(linked)


def test_t3_the_instruction_requires_planning_an_accepted_suggestion_with_a_null_linked_card(world):
    with _summary_board() as board:
        body = kanban_db.get_task(board, world["review_task"]).body
    instruction = " ".join(body.split("<<<BEGIN UNTRUSTED DATA")[0].lower().split())

    assert "must plan" in instruction
    assert "plans three" in instruction
    assert "refuses" in instruction


def test_t4_a_delivered_linked_card_settles_its_accepted_suggestion_and_the_review_closes_it(world):
    linked, card = world["linked"], world["card"]
    proposal = {"recommendation_id": linked, "card_id": card, "reason": "Lunch for the guests is ordered.",
                "evidence": [f"suggestions:{linked}", f"tasks:{card}"]}
    _complete(world, _result(world, classes={linked: ("superseded", [f"tasks:{card}"])}, closures=[proposal]))

    result = _parse(world)

    classes = {item["item"]: (item["class"], item["evidence"]) for item in result.classification}
    assert classes[linked] == ("superseded", [f"tasks:{card}"])
    # Every other accepted suggestion stays answered.
    assert classes[world["accepted"]] == ("answered", [f"suggestions:{world['accepted']}"])
    [frozen] = [item for item in result.sources["suggestions"] if item["id"] == linked]
    assert result.proposed_closures == [{**proposal, "card_state": "done",
                                         "expected_lifecycle_version": frozen["lifecycle_version"]}]
    with _written_at(NOW + 1):
        review.review_tick(NOW + 1)
    with _project_board(world["project"]) as board:
        closure = board.execute(
            "SELECT card_id,card_state,closed_by,review_date FROM recommendation_closures WHERE recommendation_id=?",
            (linked,),
        ).fetchone()
        decision = board.execute("SELECT recommendation_decision FROM tasks WHERE id=?", (linked,)).fetchone()[0]
    assert tuple(closure) == (card, "done", "project-review", "2026-11-10")
    assert decision == "accepted"


@pytest.mark.parametrize("live", ["open", "linked_since_the_review"])
def test_t5_the_exact_reviewed_plan_links_its_card_in_the_card_transaction(world, ctx, live):
    plan = _plan()
    _complete(world, _result(world, plan=plan, pairs=[{"change": 0, "suggestion": world["accepted"]}]))
    review.review_tick(NOW + 1)
    before = _links(world)

    if live == "open":
        committed = _commit(ctx, world, plan, key="accepted-room-plan")
        [card] = committed["created_task_ids"]
        assert _links(world) == sorted(before + [(card, world["accepted"])])
        assert _trigger(world, card) == "scheduled_review"
        return
    # The kernel checks the live store in the card's transaction: a card linked since the review fails the whole plan.
    with _project_board(world["project"]) as board:
        other = kanban_db.create_task(board, title="Hold the meeting room", project_id=world["project"].id,
                                      made_from_recommendation=world["accepted"])
    with pytest.raises(owner_workspace.OwnerWorkspaceError) as failed:
        _commit(ctx, world, plan, key="accepted-room-plan")
    assert failed.value.code == "crash_recovery_failed"
    assert _links(world) == sorted(before + [(other, world["accepted"])])
    assert HALL not in _work_titles(world)


@pytest.mark.parametrize("title", [ROOM, SAME, HALL])
def test_t5_a_pair_titled_like_its_own_suggestion_links_its_card_at_the_exact_plan_commit(world, ctx, title):
    """The same-label pairs, and the differently titled control: the owner's exactly approved commit links the card."""
    from hermes_cli.owner_project_review_origin import confirmed_plan_links

    project, accepted, plan = world["project"], world["accepted"], _plan(_add(title))
    _complete(world, _result(world, plan=plan, pairs=[{"change": 0, "suggestion": accepted}]))
    assert review.review_tick(NOW + 1) == 0
    before = _links(world)
    next_step, task, run, digest = _brief(world)
    facts = {"hint": next_step, "source": (task, run is not None, digest is not None),
             "pairs": confirmed_plan_links(project.id, _request(project.id, plan))}

    committed = _commit(ctx, world, plan, key="same-label-room-plan")

    [card] = committed["created_task_ids"]
    assert title in _work_titles(world)
    # The provenance and the new made-from row of the created card, read back from the registry and the Project board.
    facts.update(trigger=_trigger(world, card), made_from=sorted(set(_links(world)) - set(before)))
    assert facts == {"hint": _paired_hint(title), "source": (world["review_task"], True, True),
                     "pairs": [(0, accepted)], "trigger": "scheduled_review", "made_from": [(card, accepted)]}


def test_t5_after_the_owner_denies_a_same_label_pair_the_next_review_gives_no_hint_and_refuses_it(world, ctx):
    """Control: the one denial rule (origin._denied_plan) still answers for a same-label pair."""
    from hermes_cli.owner_project_review_view import project_review_advice_view

    project, plan, pairs = world["project"], _plan(_add(ROOM)), [{"change": 0, "suggestion": world["accepted"]}]
    _complete(world, _result(world, plan=plan, pairs=pairs))
    review.review_tick(NOW + 1)
    assert _commit(ctx, world, plan, key="denied-same-label-plan", choice="deny")["error"] == "confirmation_denied"
    later = _next_review(world)
    _complete({**world, "review_task": later}, _result(world, plan=plan, pairs=pairs))

    assert _recheck(world, plan, later, pairs=pairs) is None
    review.review_tick(NOW + 86401)

    with projects_db.connect_closing(world["root"] / "projects.db") as registry:
        later_briefs = registry.execute("SELECT COUNT(*) FROM project_review_briefs WHERE source_task_id=?",
                                        (later,)).fetchone()[0]
        refusal = registry.execute("SELECT last_refusal FROM project_review_state WHERE project_id=?",
                                   (project.id,)).fetchone()[0]
    assert later_briefs == 0
    assert json.loads(refusal)["code"] == "review_plan_suggestion_denied"
    assert project_review_advice_view(project.id) is None


def test_t5_a_silence_on_a_same_label_pair_denies_nothing(world, ctx):
    """Control: a receipt with the reason timeout denies nothing; the next review gives the hint and accepts the pair."""
    from hermes_cli.owner_project_review_view import project_review_advice_view

    project, plan, pairs = world["project"], _plan(_add(ROOM)), [{"change": 0, "suggestion": world["accepted"]}]
    _complete(world, _result(world, plan=plan, pairs=pairs))
    review.review_tick(NOW + 1)
    assert _silence(ctx, world, plan, key="silent-same-label-plan")["reason"] == "timeout"
    later = _next_review(world)
    _complete({**world, "review_task": later}, _result(world, plan=plan, pairs=pairs))

    review.review_tick(NOW + 86401)

    with projects_db.connect_closing(world["root"] / "projects.db") as registry:
        refusal = registry.execute("SELECT last_refusal FROM project_review_state WHERE project_id=?",
                                   (project.id,)).fetchone()[0]
    assert (_brief(world)[:2], refusal) == ((_paired_hint(ROOM), later), None)
    assert project_review_advice_view(project.id) == {"review_date": "2026-11-11", "proposal": plan}


def test_t6_a_changed_plan_creates_its_card_and_writes_no_row(world, ctx):
    plan = _plan()
    _complete(world, _result(world, plan=plan, pairs=[{"change": 0, "suggestion": world["accepted"]}]))
    review.review_tick(NOW + 1)
    before = _links(world)

    committed = _commit(ctx, world, {**plan, "summary": "Book a conference hall for the guests."}, key="changed-room-plan")

    [card] = committed["created_task_ids"]
    assert HALL in _work_titles(world)
    assert _links(world) == before
    assert _trigger(world, card) == "owner_request"


def test_t7_a_denied_plan_writes_nothing_and_the_next_review_does_not_propose_it_again(world, ctx):
    plan = _plan()
    before = (_links(world), _work_titles(world))

    denied = _commit(ctx, world, plan, key="denied-room-plan", choice="deny")

    assert (denied["ok"], denied["error"]) == (False, "confirmation_denied")
    assert (_links(world), _work_titles(world)) == before
    # The next review proposes the same plan: no hint reaches the owner, and nothing is pinned as accepted advice.
    _complete(world, _result(world, plan=plan))
    result = _parse(world)
    assert result.brief["next_step"] == result.summary["next_action"] == f"{HALL}."
    assert result.source_run_id is None
    # The same pair is refused.
    _rewrite(world, _result(world, plan=plan, pairs=[{"change": 0, "suggestion": world["accepted"]}]))
    with pytest.raises(review.ReviewRefused) as refused:
        _parse(world)
    assert refused.value.code == "review_plan_suggestion_denied"


def test_t7_a_denied_reviewed_plan_is_not_proposed_again_for_its_suggestion_in_other_words(world, ctx):
    from hermes_cli.owner_project_review_proposals import proposal_hint
    from hermes_cli.owner_project_review_view import project_review_advice_view

    project, plan, pairs = world["project"], _plan(), [{"change": 0, "suggestion": world["accepted"]}]
    # A review pairs the accepted suggestion with a plan, and the harvest accepts it: the owner sees the hint.
    _complete(world, _result(world, plan=plan, pairs=pairs))
    assert _parse(world).brief["next_step"] == HINT
    assert review.review_tick(NOW + 1) == 0
    with projects_db.connect_closing(world["root"] / "projects.db") as registry:
        assert registry.execute("SELECT next_step FROM project_review_briefs WHERE project_id=?",
                                (project.id,)).fetchone()[0] == HINT
    assert project_review_advice_view(project.id) == {"review_date": "2026-11-10", "proposal": plan}

    # The owner denies exactly that reviewed plan in the Project chat.
    assert _commit(ctx, world, plan, key="denied-reviewed-plan", choice="deny")["error"] == "confirmation_denied"
    assert project_review_advice_view(project.id) is None

    # News opens the next day's review, which pairs the same suggestion with a reworded plan.
    with _written_at(NOW + 3600), _project_board(project) as board:
        kanban_db.add_comment(board, world["booked"], "default", "The hall must seat forty guests.")
    assert review.review_tick(NOW + 86400) == 1
    with projects_db.connect_closing(world["root"] / "projects.db") as registry:
        later = registry.execute("SELECT open_task_id FROM project_review_state WHERE project_id=?",
                                 (project.id,)).fetchone()[0]
    assert later != world["review_task"]
    reworded = {**_plan(_add("Reserve the large conference hall")), "summary": "Reserve a large hall for the visit."}
    _complete({**world, "review_task": later}, _result(world, plan=reworded, pairs=pairs))
    # Only the first plan's exact approval was denied: the reworded plan has a digest of its own.
    with projects_db.connect_closing(world["root"] / "projects.db") as registry:
        denials = [row[0] for row in registry.execute(
            "SELECT request_digest FROM owner_workspace_receipts WHERE status='denied' "
            "AND operation='owner_project_plan_commit'")]
    assert denials == [_request(project.id, plan)] and _request(project.id, reworded) not in denials

    with _summary_board() as board:
        with pytest.raises(review.ReviewRefused) as refused:
            review.parse_review_result(board, later, project_id=project.id, now=NOW + 86400, stale_days=14)
        sources, _cut = review._frozen_sources(kanban_db.get_task(board, later).body)
    assert refused.value.code == "review_plan_suggestion_denied"
    # No hint names the reworded plan for the suggestion whose reviewed plan the owner denied.
    with review._review_evidence_board(project.id) as (board, registry, live):
        assert proposal_hint(reworded, sources, board, registry, live, now=NOW + 86400, limit=240,
                             max_bytes=record.TOTAL_INPUT_CAP_BYTES, pairs=pairs) is None
    # The real harvest of that review, while the brief history holds the pair: no hint is stored, the pair is refused.
    review.review_tick(NOW + 86401)
    with projects_db.connect_closing(world["root"] / "projects.db") as registry:
        briefs = registry.execute("SELECT next_step,source_task_id FROM project_review_briefs WHERE project_id=?",
                                  (project.id,)).fetchall()
        refusal = registry.execute("SELECT last_refusal FROM project_review_state WHERE project_id=?",
                                   (project.id,)).fetchone()[0]
    assert [tuple(brief) for brief in briefs] == [(HINT, world["review_task"])]
    assert json.loads(refusal)["code"] == "review_plan_suggestion_denied"


def test_t7_a_silence_denies_nothing_and_the_next_review_gives_the_hint_and_accepts_the_pair(world, ctx):
    from hermes_cli.owner_project_review_view import project_review_advice_view

    project, plan, pairs = world["project"], _plan(), [{"change": 0, "suggestion": world["accepted"]}]
    _complete(world, _result(world, plan=plan, pairs=pairs))
    assert review.review_tick(NOW + 1) == 0
    before = (_links(world), _work_titles(world))

    # The owner gives no answer: the approval times out, and the receipt records the silence.
    silent = _silence(ctx, world, plan, key="silent-room-plan")

    assert silent == {"ok": False, "error": "confirmation_denied", "reason": "timeout"}
    assert (_links(world), _work_titles(world)) == before
    with projects_db.connect_closing(world["root"] / "projects.db") as registry:
        [receipt] = registry.execute("SELECT * FROM owner_workspace_receipts WHERE status='denied' "
                                     "AND operation='owner_project_plan_commit'").fetchall()
    assert receipt["request_digest"] == _request(project.id, plan)
    assert owner_workspace._is_retryable_confirmation_timeout(receipt)
    # News opens the next day's review, which pairs the same suggestion with the same plan.
    with _written_at(NOW + 3600), _project_board(project) as board:
        kanban_db.add_comment(board, world["booked"], "default", "The hall must seat forty guests.")
    assert review.review_tick(NOW + 86400) == 1
    with projects_db.connect_closing(world["root"] / "projects.db") as registry:
        later = registry.execute("SELECT open_task_id FROM project_review_state WHERE project_id=?",
                                 (project.id,)).fetchone()[0]
    _complete({**world, "review_task": later}, _result(world, plan=plan, pairs=pairs))

    review.review_tick(NOW + 86401)

    # The real harvest gives the hint and accepts the pair: the silence denied nothing.
    with projects_db.connect_closing(world["root"] / "projects.db") as registry:
        brief = registry.execute("SELECT next_step,source_task_id FROM project_review_briefs WHERE project_id=? "
                                 "ORDER BY review_date DESC LIMIT 1", (project.id,)).fetchone()
        refusal = registry.execute("SELECT last_refusal FROM project_review_state WHERE project_id=?",
                                   (project.id,)).fetchone()[0]
    assert (tuple(brief), refusal) == ((HINT, later), None)
    assert project_review_advice_view(project.id) == {"review_date": "2026-11-11", "proposal": plan}


def test_t7_after_its_pair_left_the_brief_history_the_review_gives_the_hint_again(world, ctx):
    """The accepted limit of the owner rule of 2026-11-17: the 90 briefs of the history remember a denial."""
    project, pairs = world["project"], [{"change": 0, "suggestion": world["accepted"]}]
    _complete(world, _result(world, plan=_plan(), pairs=pairs))
    assert review.review_tick(NOW + 1) == 0
    assert _commit(ctx, world, _plan(), key="denied-room-plan", choice="deny")["error"] == "confirmation_denied"
    # Daily review ticks until the normal retention drops the brief that paired the plan with the suggestion.
    for day in range(1, 92):
        review.review_tick(NOW + day * 86400)
        with projects_db.connect_closing(world["root"] / "projects.db") as registry:
            task = registry.execute("SELECT open_task_id FROM project_review_state WHERE project_id=?",
                                    (project.id,)).fetchone()[0]
        if task:
            _complete({**world, "review_task": task}, _result(world))
            review.review_tick(NOW + day * 86400 + 1)
    with projects_db.connect_closing(world["root"] / "projects.db") as registry:
        assert registry.execute("SELECT COUNT(*) FROM project_review_briefs WHERE source_task_id=?",
                                (world["review_task"],)).fetchone()[0] == 0
    when = NOW + 92 * 86400
    with _written_at(when - 3600), _project_board(project) as board:
        kanban_db.add_comment(board, world["booked"], "default", "The hall needs forty seats.")
    assert review.review_tick(when) == 1
    with projects_db.connect_closing(world["root"] / "projects.db") as registry:
        later = registry.execute("SELECT open_task_id FROM project_review_state WHERE project_id=?",
                                 (project.id,)).fetchone()[0]
    reworded = {**_plan(_add("Reserve the large conference hall")), "summary": "Reserve a larger hall for the visit."}
    _complete({**world, "review_task": later}, _result(world, plan=reworded, pairs=pairs))

    review.review_tick(when + 1)

    with projects_db.connect_closing(world["root"] / "projects.db") as registry:
        brief = registry.execute("SELECT next_step,source_task_id FROM project_review_briefs WHERE project_id=? "
                                 "ORDER BY review_date DESC LIMIT 1", (project.id,)).fetchone()
    assert tuple(brief) == (HINT.replace(HALL, "Reserve the large conference hall"), later)
