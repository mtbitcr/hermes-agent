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
