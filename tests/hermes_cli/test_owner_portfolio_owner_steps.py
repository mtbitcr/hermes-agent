"""The cross-project advice counts only owner steps and runs only when two or more Projects need the owner."""

import importlib
import json
import logging
import sqlite3
import time

import pytest
import yaml

from hermes_cli import kanban_db, owner_portfolio_review as portfolio, owner_project_review as review
from hermes_cli import owner_project_review_owner_line as owner_line, projects_db
from hermes_cli import owner_project_summary_events as events
from hermes_cli.owner_review_handover import handover_refusal
from tests.gateway.test_owner_heads_up import root as digest_root  # noqa: F401
from tests.gateway.test_owner_review_digest import NOTES_LINE, Adapter, _project, _ready, _runner
from tests.hermes_cli.test_owner_portfolio_review import (
    SUMMARIES, _brief, _finish_portfolio, _frozen, _news, _portfolio_body, _portfolio_on, _project_board, _summaries,
    _waiting_pair,
)
from tests.hermes_cli.test_owner_project_review_cycle import PLANNER, _at, _enter_planning_worker
from tests.hermes_cli.test_owner_project_review_owner_line import NAME, RID, _chat, _consume, _independent_count, _link
from tests.hermes_cli.test_owner_project_review_result import (  # noqa: F401
    NOW, _owner_turn, _quiet_project, _suggestion, root,
)
from tests.hermes_cli.test_owner_review_handover import _complete, _payload, _read, _task_with_key, _worker

DAY = "2026-11-09"
KEY = f"project-review:portfolio:{DAY}:0"
OLD_KEYS = ("assignee_load", "shared_assignees", "classification_counts", "ready_work")
WAITING_PLAN = {
    "latest_response_id": "resp_1", "proposal_consumed": False, "proposal_claimed": False, "conversation_closed": False,
}


def _work_card(project):
    with _project_board(project) as board:
        return board.execute("SELECT id FROM tasks WHERE task_kind='work' ORDER BY created_at,id").fetchone()[0]


def _ask_owner(project, *, at=NOW - 86400):
    """The Project's work card waits for the owner's answer, from ``at``."""
    card = _work_card(project)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(time, "time", lambda: at)
        with _project_board(project) as board:
            assert kanban_db.block_task(board, card, reason="Which room?", kind="needs_input")


def _park_work(project, assignee="worker"):
    """A parked (scheduled) card and a todo card on one assignee: work, but no owner step."""
    card = _work_card(project)
    with _project_board(project) as board:
        parked = kanban_db.create_task(board, title="Call the caterer", assignee=assignee, project_id=project.id)
        assert kanban_db.schedule_task(board, parked, reason="Wait for the date.")
        kanban_db.create_task(board, title="Print the signs", assignee=assignee, project_id=project.id, parents=[card])


def _blocked_cards(project):
    with _project_board(project) as board:
        return board.execute("SELECT COUNT(*) FROM tasks WHERE status='blocked'").fetchone()[0]


def _settle(root, *projects):
    """Move each Project's review cursor past its new events, so the Project stays quiet."""
    for project in projects:
        with _project_board(project) as board:
            cursor = board.execute("SELECT MAX(id) FROM task_events").fetchone()[0]
        with projects_db.connect_closing(root / "projects.db") as registry:
            with kanban_db.write_txn(registry):
                registry.execute(
                    "UPDATE project_review_state SET event_cursor=? WHERE project_id=?", (cursor, project.id),
                )


def test_two_projects_with_only_parked_and_todo_work_get_no_advice_and_no_task(root):
    first, second = _quiet_project(root, "Meeting Plan"), _quiet_project(root, "Guest Visit")
    for project in (first, second):
        _park_work(project)
    _settle(root, first, second)
    _portfolio_on(root, first, second)

    assert review.review_tick(NOW) == 0
    assert review.review_tick(NOW + 1) == 0

    with projects_db.connect_closing(root / "projects.db") as registry:
        brief = _brief(registry, DAY)
        assert [brief["conflicts"], brief["priorities"], brief["shared_blockers"]] == [[], [], []]
        assert sorted(brief["project_scope"]) == sorted([first.id, second.id])
        assert (brief["written_at"], brief["revision"]) == (NOW, 1)
        assert review.budget(registry, DAY) == (0, 0)
    assert not kanban_db.kanban_db_path(board=SUMMARIES).exists()


def test_the_frozen_input_holds_only_the_owner_steps_of_each_project(root):
    first, second = _quiet_project(root, "Meeting Plan"), _quiet_project(root, "Guest Visit")
    _ask_owner(first, at=NOW - 3 * 86400 - 60)
    _ask_owner(second, at=NOW - 7200)
    _settle(root, first, second)
    _portfolio_on(root, first, second)

    assert review.review_tick(NOW) == 1

    body = _portfolio_body()
    frozen = _frozen(body)
    assert set(frozen) == {"projects", "project_scope"}
    records = {item["name"]: item for item in frozen["projects"]}
    assert set(records) == {"Meeting Plan", "Guest Visit"}
    for item in records.values():
        assert set(item) == {"name", "review_state", "brief", "owner_steps", "oldest_answer_days"}
        assert set(item["brief"]) == {"needs_you"}
        assert item["owner_steps"] == {"answers": 1, "suggestions": 0, "plan_approvals": 0}
    assert records["Meeting Plan"]["oldest_answer_days"] == 3
    assert records["Guest Visit"]["oldest_answer_days"] == 0
    assert not any(key in body for key in OLD_KEYS)


@pytest.mark.parametrize("sentence", [
    "Three parked tasks wait for Monday.", "The default worker is preparing the room.",
], ids=["parked-count", "helper-name"])
def test_a_headline_a_worker_wrote_never_reaches_the_frozen_input(root, monkeypatch, sentence):
    first, second = _waiting_pair(root)
    _portfolio_on(root, first, second)
    monkeypatch.setattr(time, "time", lambda: NOW)
    _news(first)
    assert review.review_tick(NOW) == 1
    review_task = _task_with_key(root, f"project-review:{first.id}:{DAY}:0")

    card = _work_card(first)
    claim = {"item": card, "class": "still actionable", "evidence": [f"decisions:wait:{card}"]}
    payload = _payload(classification=[claim])
    payload["project_review"]["brief"]["headline"] = sentence
    with _worker(root, monkeypatch, review_task):
        accepted = json.loads(_complete({"summary": "Reviewed.", "metadata": payload}))
        assert accepted.get("ok") is True, accepted
    assert review.review_tick(NOW + 1) == 1

    # The harvest kept the worker's headline in the Project's own brief; the frozen input of the day holds no trace.
    brief = _read(root / "projects.db", "SELECT review_date, headline FROM project_review_briefs WHERE project_id=?",
                  first.id)
    assert brief == [(DAY, sentence)]
    body = _portfolio_body()
    records = {item["name"]: item for item in _frozen(body)["projects"]}
    assert set(records) == {"Meeting Plan", "Guest Visit"}
    assert records["Meeting Plan"]["review_state"] == "reviewed"
    assert all(item["owner_steps"]["answers"] == 1 for item in records.values())
    assert sentence not in body


@pytest.mark.parametrize("where", ["handover", "harvest"])
def test_a_line_naming_a_project_with_no_owner_step_is_refused(root, caplog, where):
    first, second, idle = (_quiet_project(root, name) for name in ("Meeting Plan", "Guest Visit", "Hotel Search"))
    for project in (first, second):
        _ask_owner(project)
    _settle(root, first, second, idle)
    _portfolio_on(root, first, second, idle)
    assert review.review_tick(NOW) == 1
    result = {"conflicts": [], "priorities": ["Hotel Search: Book the room."], "shared_blockers": []}

    if where == "handover":
        with _summaries() as board:
            task_id = board.execute("SELECT id FROM tasks WHERE idempotency_key=?", (KEY,)).fetchone()[0]
            assert kanban_db.claim_task(board, task_id) is not None
            task = kanban_db.get_task(board, task_id)
        refusal = handover_refusal(task, {"portfolio_review": result})
        assert refusal is not None
        assert refusal["code"] == "portfolio_result_text_invalid"
        assert refusal["detail"] == {"lines": [{"line": "priorities[0]", "rule": "no_owner_step"}]}
        assert "priorities line 1 names a Project that has nothing waiting for the owner." in refusal["message"]
        accepted = {"conflicts": [], "priorities": ["Meeting Plan: Book the room."], "shared_blockers": []}
        assert handover_refusal(task, {"portfolio_review": accepted}) is None
    else:
        _finish_portfolio(result)
        with caplog.at_level(logging.WARNING, logger="hermes_cli.owner_project_review"):
            assert review.review_tick(NOW + 1) == 0
        assert "portfolio_result_refused (portfolio_result_text_invalid)" in caplog.text
        assert "lines=priorities[0]:no_owner_step" in caplog.text
        with projects_db.connect_closing(root / "projects.db") as registry:
            assert _brief(registry, DAY) is None


def test_a_project_whose_only_owner_step_is_a_waiting_plan_proposal_counts_one_plan_approval(root, monkeypatch):
    planner, asker = _quiet_project(root, "Meeting Plan"), _quiet_project(root, "Guest Visit")
    _ask_owner(asker)
    _settle(root, planner, asker)
    _portfolio_on(root, planner, asker)
    monkeypatch.setattr(
        owner_line, "linked_chats",
        lambda registry, project_id, home: [("plan-chat", WAITING_PLAN)] if project_id == planner.id else [],
    )

    assert review.review_tick(NOW) == 1

    records = {item["name"]: item for item in _frozen(_portfolio_body())["projects"]}
    assert records["Meeting Plan"]["owner_steps"] == {"answers": 0, "suggestions": 0, "plan_approvals": 1}
    assert records["Meeting Plan"]["oldest_answer_days"] is None
    assert records["Guest Visit"]["owner_steps"] == {"answers": 1, "suggestions": 0, "plan_approvals": 0}


def test_the_portfolio_input_counts_owner_steps_from_the_root_stores_under_the_worker_environment(root, monkeypatch):
    first, second = _quiet_project(root, "Meeting Plan"), _quiet_project(root, "Guest Visit")
    monkeypatch.setattr(time, "time", lambda: NOW)
    for project in (first, second):
        _ask_owner(project)
    with _project_board(first) as board:
        _suggestion(board, first)
    names = ["raphael-owner-" + letter * 32 for letter in "abcde"]
    # In the first Project two proposals wait, one was applied and one chat ends with a question; in the second one waits.
    for index in (0, 1, 2, 4):
        _chat(root, names[index], f"{RID}_{index}")
    _consume(root, names[2], f"{RID}_2")
    _owner_turn(root, names[3], "resp_room_question", "Which room should be booked?", NOW)
    for name in names[:4]:
        _link(root, name, first.id)
    _link(root, names[4], second.id)
    blocked = [_blocked_cards(first), _blocked_cards(second)]

    _enter_planning_worker(monkeypatch, root, first.board_slug, _work_card(first))
    # A store in the worker's profile home that differs from the root store: there, the applied proposal waits.
    profile = root / "profiles" / PLANNER
    profile.mkdir(parents=True, exist_ok=True)
    source = sqlite3.connect(f"{(root / 'response_store.db').resolve().as_uri()}?mode=ro", uri=True)
    target = sqlite3.connect(profile / "response_store.db")
    try:
        source.backup(target)
        target.execute("UPDATE conversations SET consumed_response_id=NULL")
        target.commit()
    finally:
        source.close()
        target.close()
    registry_path = kanban_db.kanban_home() / "projects.db"
    assert registry_path.resolve() == (root / "projects.db").resolve()
    with events._read_only(registry_path) as registry:
        frozen = portfolio._input(registry, [first, second], DAY)

    steps = {item["name"]: item["owner_steps"] for item in frozen["projects"]}
    assert steps == {
        "Meeting Plan": {"answers": 1, "suggestions": 1, "plan_approvals": 2},
        "Guest Visit": {"answers": 1, "suggestions": 0, "plan_approvals": 1},
    }
    # The independent counts: the blocked cards of each board, and the root stores (the profile store says 3 for the first).
    assert blocked == [1, 1]
    assert _independent_count(root / "projects.db", root / "response_store.db", first.id) == 2
    assert _independent_count(root / "projects.db", root / "response_store.db", second.id) == 1
    assert _independent_count(root / "projects.db", profile / "response_store.db", first.id) == 3
    assert not (profile / "projects.db").exists()


def test_an_unknown_plan_approval_count_is_left_out_of_the_input_never_written_as_a_number(root):
    """Owner decision 1: a linked chat without a response store leaves the approvals unknown, so the key is absent."""
    unknown, known = _quiet_project(root, "Meeting Plan"), _quiet_project(root, "Guest Visit")
    for project in (unknown, known):
        _ask_owner(project)
    _link(root, NAME, unknown.id)
    assert not (root / "response_store.db").exists()

    with projects_db.connect_closing(root / "projects.db") as registry:
        frozen = portfolio._input(registry, [unknown, known], DAY)

    steps = {item["name"]: item["owner_steps"] for item in frozen["projects"]}
    assert steps["Meeting Plan"] == {"answers": 1, "suggestions": 0}
    assert steps["Guest Visit"] == {"answers": 1, "suggestions": 0, "plan_approvals": 0}


def test_a_project_whose_only_count_is_unknown_is_not_a_project_with_an_owner_step(root, monkeypatch):
    """Owner decision 1: an unknown count never makes a Project one with an owner step, so one real step is not enough."""
    asker, unknown = _quiet_project(root, "Meeting Plan"), _quiet_project(root, "Guest Visit")
    _ask_owner(asker)
    _settle(root, asker, unknown)
    _portfolio_on(root, asker, unknown)
    monkeypatch.setattr(
        owner_line, "linked_chats", lambda registry, project_id, home: None if project_id == unknown.id else [],
    )

    assert review.review_tick(NOW) == 0

    with projects_db.connect_closing(root / "projects.db") as registry:
        brief = _brief(registry, DAY)
        assert [brief["conflicts"], brief["priorities"], brief["shared_blockers"]] == [[], [], []]
        assert sorted(brief["project_scope"]) == sorted([asker.id, unknown.id])
        assert review.budget(registry, DAY) == (0, 0)
    assert not kanban_db.kanban_db_path(board=SUMMARIES).exists()


def test_a_plan_approval_read_that_raises_is_refused_by_the_existing_input_rule_and_takes_no_slot(root, caplog):
    """Owner decision 2: a chat that cannot be read is an input that cannot be read: the same warning, no second rule."""
    first, second = _quiet_project(root, "Meeting Plan"), _quiet_project(root, "Guest Visit")
    for project in (first, second):
        _ask_owner(project)
    _settle(root, first, second)
    review.review_tick(NOW)
    _portfolio_on(root, first, second)

    def unreadable(registry, project_id, home):
        raise ValueError("linked_conversation_unavailable")

    with pytest.MonkeyPatch.context() as fault, caplog.at_level(logging.WARNING, logger="hermes_cli.owner_project_review"):
        fault.setattr(owner_line, "linked_chats", unreadable)
        assert review.review_tick(NOW + 1) == 0

    assert "portfolio_input_refused" in caplog.text
    assert "portfolio_input_too_large" not in caplog.text
    with projects_db.connect_closing(root / "projects.db") as registry:
        assert _brief(registry, DAY) is None
        assert review.budget(registry, DAY) == (0, 0)
    assert not kanban_db.kanban_db_path(board=SUMMARIES).exists()

    assert review.review_tick(NOW + 2) == 1
    assert {item["name"] for item in _frozen(_portfolio_body())["projects"]} == {"Meeting Plan", "Guest Visit"}


# A morning message whose stored portfolio brief holds no advice says so; one line of advice keeps the notes line.
NO_ADVICE_LINE = "Raphael has no advice across your projects today."
LISTS = ("conflicts", "priorities", "shared_blockers")


# `root` here is the hermes_cli fixture, so the digest tests' `ready` is rebuilt on the gateway root.
@pytest.fixture
def digest_ready(digest_root, monkeypatch):
    now = _at(2026, 11, 10, 6, 30)
    monkeypatch.setattr(time, "time", lambda: now)
    project = _project(digest_root)
    config = yaml.safe_load((digest_root / "config.yaml").read_text())
    config["owner_review"] = {"enabled": True, "portfolio": True, "digest": True, "allowlist": [project["project_id"]]}
    (digest_root / "config.yaml").write_text(yaml.safe_dump(config))
    _ready(digest_root, project, "2026-11-10", now)
    return project


def _store_advice(root, **lists):
    with projects_db.connect_closing(root / "projects.db") as registry:
        for column in LISTS:
            registry.execute(f"UPDATE portfolio_review_briefs SET {column}=?", (json.dumps(lists.get(column, [])),))
        registry.commit()
        return projects_db.get_portfolio_review_brief(registry, "2026-11-10")


@pytest.mark.asyncio
async def test_a_stored_portfolio_brief_with_three_empty_lists_goes_out_with_the_no_advice_line(digest_root, digest_ready):
    digest = importlib.import_module("gateway.owner_review_digest")
    brief = _store_advice(digest_root)
    assert [brief[column] for column in LISTS] == [[], [], []]
    adapter = Adapter()

    assert await digest.tick(_runner(adapter)) == 1

    assert adapter.sent[0][1].splitlines()[1:] == [NO_ADVICE_LINE, "Meeting Plan: The room is booked."]


@pytest.mark.asyncio
@pytest.mark.parametrize("column", LISTS)
async def test_one_line_of_stored_advice_in_any_list_keeps_the_notes_line(digest_root, digest_ready, column):
    digest = importlib.import_module("gateway.owner_review_digest")
    advice = "Meeting Plan: Prepare the guest list."
    _store_advice(digest_root, **{column: [advice]})
    adapter = Adapter()

    assert await digest.tick(_runner(adapter)) == 1

    assert adapter.sent[0][1].splitlines()[1:] == [NOTES_LINE, "Meeting Plan: The room is booked.", advice]
