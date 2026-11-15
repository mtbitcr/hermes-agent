"""The owner line of the daily review: each open owner step in fixed kernel text, on a quiet day and on the first day.

An owner wait, a pending suggestion and a plan change that waits for the owner's approval in the Project chat are
named from records only. The approval count is read from the root response store, read-only, and frozen into the task
body; the harvest reads it again. A first review never refers to an earlier review. Everything runs through the real
review tick, the real hand-over and the root stores, which are read back with plain sqlite3.
"""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
import time

import pytest

from hermes_cli import kanban_db, owner_project_review as review, projects_db
from hermes_cli import owner_project_review_record as records, owner_project_summary_events as events
from tests.hermes_cli.test_owner_project_review_result import (  # noqa: F401
    NOW, _complete_review, _frozen_body, _open_review, _owner_turn, _quiet_project, _review_stores, _stale_outcome, root,
)
from tests.hermes_cli.test_owner_review_handover import (
    DAY, LOGGER, STILL_RUNNING, _assert_running, _complete, _read, _worker,
)

ANSWER = "An answer is waiting on Decisions."
APPROVAL = "A plan change waits for your approval in the Project chat."
QUIET = "Nothing changed since the last review."
FIRST = "Nothing has happened in this Project yet."
WAITS = "The work waits for you: see what needs you."
PLAN = "Follow the current plan."
EARLIER = "Book the room is unchanged since the last review."
SO_FAR = "Book the room is unchanged so far."
NAME = "raphael-owner-" + "a" * 32
RID = "resp_room_plan_change"

# An actionable Project change proposal, the final reply the Project chat stores as waiting for approval.
_PROPOSAL = {
    "schema_version": 6, "kind": "project_change_proposal", "mode": "existing",
    "request_title": "Add the approved milestone", "summary": "Add one owner-visible milestone.", "project_size": "small",
    "specification": "Add and verify the approved milestone.", "current_milestone": "Add the approved milestone",
    "owner_visible_result": "The owner can review the finished milestone.", "impact": ["Keeps completed work intact."],
    "later_milestones": [],
    "changes": [{
        "action": "add", "reason": "The approved milestone needs one task.", "title": "Prepare the milestone",
        "body": "Prepare the owner-visible milestone.", "assignee": "default", "responsibility": "B03",
        "execution_tier": "routine", "risk_tier": 1, "owned_paths": [], "existing_parent_refs": [], "new_parents": [],
    }],
}


def _chat(root, name, response_id, *, created_at=NOW):
    """One Project chat whose final reply is a plan change proposal that waits for the owner's approval."""
    from gateway.platforms.api_server import ResponseStore

    store = ResponseStore(db_path=str(root / "response_store.db"), default_profile="default")
    try:
        store.put(response_id, {
            "response": {"id": response_id, "created_at": created_at},
            "conversation_history": [
                {"role": "user", "content": "Plan the room change."},
                {"role": "assistant", "content": json.dumps(_PROPOSAL)},
            ],
        })
        assert store.set_conversation(name, response_id, owner_proposal=True)
    finally:
        store._conn.close()


def _consume(root, name, response_id):
    """The owner approves the proposal and the platform applies it; the chat head does not move."""
    from gateway.platforms.api_server import ResponseStore

    store = ResponseStore(db_path=str(root / "response_store.db"), default_profile="default")
    try:
        assert store.mark_owner_proposal_consumed("default", name, response_id)
    finally:
        store._conn.close()


def _link(root, name, project_id):
    with projects_db.connect_closing(root / "projects.db") as registry:
        assert projects_db.link_project_conversation(registry, name, project_id) == project_id


def _set_state(root, project, **values):
    with projects_db.connect_closing(root / "projects.db") as registry:
        with kanban_db.write_txn(registry):
            registry.execute(
                "UPDATE project_review_state SET " + ",".join(f"{key}=?" for key in values) + " WHERE project_id=?",
                (*values.values(), project.id),
            )


def _brief(root, project):
    with projects_db.connect_closing(root / "projects.db") as registry:
        return projects_db.get_project_review_brief(registry, project.id), projects_db.get_project_summary(registry, project.id)


def _comment(project):
    """An owner comment on the Project's open card: news for the next review."""
    with kanban_db.connect_closing(kanban_db.kanban_db_path(board=project.board_slug), board=project.board_slug) as board:
        card = board.execute("SELECT id FROM tasks WHERE task_kind='work'").fetchone()[0]
        kanban_db.add_comment(board, card, "default", "The room is chosen.")


def _result(what_changed):
    return {"project_review": {
        "summary": {"outcome": "The room plan is updated.", "next_action": "Book the room.", "owner_decision": None},
        "brief": {"headline": "The room plan is updated.", "what_changed": what_changed,
                  "next_step": "Book the room.", "needs_you": None},
        "classification": [], "proposed_closures": [],
    }}


def _store_state(root):
    """The response store's bytes and modification time, and every file of its name (a journal or WAL too)."""
    path = root / "response_store.db"
    return (hashlib.sha256(path.read_bytes()).hexdigest(), path.stat().st_mtime_ns,
            sorted(item.name for item in root.iterdir() if item.name.startswith("response_store.db")))


def _independent_count(registry, store, project_id):
    """The independent count: each linked chat whose proposal is neither applied, nor claimed, nor closed."""
    names = [name for (name,) in _read(
        registry, "SELECT conversation FROM project_conversation_links WHERE project_id=?", project_id,
    )]
    ((count,),) = _read(
        store,
        "SELECT COUNT(*) FROM conversations WHERE profile='default' AND name IN (SELECT value FROM json_each(?)) "
        "AND proposal_response_id IS NOT NULL AND consumed_response_id IS NOT proposal_response_id "
        "AND NOT (claimed_response_id IS proposal_response_id AND claim_state IS 'claimed') AND closed=0",
        json.dumps(names),
    )
    return count


def _first_review(root, monkeypatch):
    """A never-reviewed Project with news, and the review task the tick opens for it."""
    project = _quiet_project(root)
    _set_state(root, project, last_reviewed_date=None)
    monkeypatch.setattr(time, "time", lambda: NOW)
    _comment(project)
    assert review.review_tick(NOW) == 1
    task_id = _open_review(root, project)
    assert f"Review days: from the start to {DAY}" in _frozen_body(task_id).splitlines()
    return project, task_id


def test_t1_a_quiet_day_of_a_waiting_project_names_the_answer_and_that_the_work_waits(root):
    project = _quiet_project(root)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(time, "time", lambda: NOW - 86400)
        with kanban_db.connect_closing(kanban_db.kanban_db_path(board=project.board_slug), board=project.board_slug) as board:
            card = board.execute("SELECT id FROM tasks LIMIT 1").fetchone()[0]
            assert kanban_db.block_task(board, card, reason="Which room?", kind="needs_input")
            cursor = board.execute("SELECT MAX(id) FROM task_events").fetchone()[0]
    # The last review already read the wait, so the Project has no news today.
    _set_state(root, project, event_cursor=cursor)
    _, before = _brief(root, project)

    assert review.review_tick(NOW) == 0

    brief, after = _brief(root, project)
    assert (brief["needs_you"], brief["next_step"]) == (ANSWER, WAITS), brief
    assert (brief["headline"], brief["what_changed"], brief["quiet"]) == (QUIET, QUIET, True)
    # A quiet day never changes the summary.
    assert after == before


def test_t2_a_review_harvested_while_a_plan_change_waits_names_the_approval(root, monkeypatch):
    project = _quiet_project(root)
    monkeypatch.setattr(time, "time", lambda: NOW)
    _chat(root, NAME, RID)
    _link(root, NAME, project.id)
    # The new chat head is news, so the tick opens a review.
    assert review.review_tick(NOW) == 1
    _complete_review(_open_review(root, project), [])

    assert review.review_tick(NOW + 1) == 0

    brief, stored = _brief(root, project)
    assert (brief["needs_you"], stored["owner_decision"]) == (APPROVAL, APPROVAL), brief
    assert brief["quiet"] is False


def test_t3_a_proposal_applied_after_the_freeze_makes_the_result_stale_without_a_failure(root, monkeypatch, caplog):
    project = _quiet_project(root)
    monkeypatch.setattr(time, "time", lambda: NOW)
    _chat(root, NAME, RID)
    _link(root, NAME, project.id)
    assert review.review_tick(NOW) == 1
    review_task = _open_review(root, project)
    _complete_review(review_task, [])
    before = _review_stores(root, project)
    # The owner approves in the chat after the input was frozen. The chat head stays where it was.
    _consume(root, NAME, RID)

    with caplog.at_level(logging.WARNING, logger=LOGGER):
        opened = review.review_tick(NOW + 1)

    # No brief, no summary change and no failure; the one further review of the day opens at once.
    _stale_outcome(root, project, before, review_task)
    assert "review_decision_state_changed" in caplog.text
    assert opened == 1
    further = _open_review(root, project)
    assert further not in {None, review_task}
    assert "Plan approvals waiting: 1" in _frozen_body(review_task).splitlines()
    assert "Plan approvals waiting: 0" in _frozen_body(further).splitlines()


def test_t4_a_quiet_day_whose_only_owner_step_is_a_waiting_plan_change_names_the_approval(root, monkeypatch):
    project = _quiet_project(root)
    monkeypatch.setattr(time, "time", lambda: NOW)
    _chat(root, NAME, RID, created_at=NOW - 86400)
    _link(root, NAME, project.id)
    with projects_db.connect_closing(root / "projects.db") as registry:
        cursor = records.read_conversation_news(registry, project.id)[0]
    # The last review already read this chat head, so the Project has no news today.
    _set_state(root, project, conversation_cursor=cursor)

    assert review.review_tick(NOW) == 0

    brief, _ = _brief(root, project)
    assert (brief["needs_you"], brief["next_step"]) == (APPROVAL, WAITS), brief
    assert (brief["headline"], brief["what_changed"], brief["quiet"]) == (QUIET, QUIET, True)


def test_t5_a_quiet_first_review_says_that_nothing_has_happened_yet(root):
    project = _quiet_project(root)
    _set_state(root, project, last_reviewed_date=None)

    assert review.review_tick(NOW) == 0

    brief, _ = _brief(root, project)
    assert (brief["headline"], brief["what_changed"]) == (FIRST, FIRST), brief
    assert (brief["next_step"], brief["needs_you"], brief["covered_from"], brief["quiet"]) == (PLAN, None, None, True)


@pytest.mark.parametrize("where", ["harvest", "handover"])
def test_t6_a_first_review_that_refers_to_an_earlier_review_is_refused(root, monkeypatch, caplog, where):
    project, task_id = _first_review(root, monkeypatch)
    if where == "harvest":
        _complete_review(task_id, [], what_changed=EARLIER)
        with caplog.at_level(logging.WARNING, logger=LOGGER):
            assert review.review_tick(NOW + 1) == 0
        brief, _ = _brief(root, project)
        assert brief is None, brief
        [(reviewed, failures, refusal)] = _read(
            root / "projects.db", "SELECT last_reviewed_date, failures, last_refusal FROM project_review_state "
            "WHERE project_id=?", project.id,
        )
        assert (reviewed, failures) == (None, 1)
        assert json.loads(refusal) == {"date": DAY, "code": "review_first_review_wording", "detail": {"fields": [
            {"field": "brief.what_changed", "rule": "earlier_review"},
        ]}}
        [line] = [text for text in caplog.text.splitlines() if "refused Project" in text]
        assert f"(review_first_review_wording) day={DAY} failures=1 fields=brief.what_changed:earlier_review" in line
        assert EARLIER not in caplog.text
        return
    with _worker(root, monkeypatch, task_id) as task:
        with caplog.at_level(logging.WARNING, logger=LOGGER):
            raw = _complete({"summary": "Reviewed.", "metadata": _result(EARLIER)})
        refused = json.loads(raw)
        assert refused.get("refusal") == {"code": "review_first_review_wording", "detail": {"fields": [
            {"field": "brief.what_changed", "rule": "earlier_review"},
        ]}}, refused
        assert refused["still_running"] is True
        assert ("brief.what_changed refers to an earlier review, but this is the first review. Write so far instead."
                in refused["error"])
        assert refused["error"].endswith(STILL_RUNNING)
        assert EARLIER not in raw and EARLIER not in caplog.text
        [line] = [text for text in caplog.text.splitlines() if "review_handover_refused" in text]
        assert (f"review_handover_refused: Project {project.id} task {task.id} run {task.current_run_id} "
                "(review_first_review_wording) fields=brief.what_changed:earlier_review") in line
        _assert_running(root, task)
        accepted = json.loads(_complete({"summary": "Reviewed.", "metadata": _result(SO_FAR)}))
        assert accepted.get("ok") is True, accepted

    assert review.review_tick(NOW + 1) == 0
    brief, _ = _brief(root, project)
    assert (brief["what_changed"], brief["covered_from"]) == (SO_FAR, None)


def test_t7_a_later_review_may_still_refer_to_the_last_review(root, monkeypatch):
    project = _quiet_project(root)
    monkeypatch.setattr(time, "time", lambda: NOW)
    _comment(project)
    assert review.review_tick(NOW) == 1
    task_id = _open_review(root, project)
    assert f"Review days: from {DAY} to {DAY}" in _frozen_body(task_id).splitlines()
    with _worker(root, monkeypatch, task_id):
        accepted = json.loads(_complete({"summary": "Reviewed.", "metadata": _result(EARLIER)}))
        assert accepted.get("ok") is True, accepted

    assert review.review_tick(NOW + 1) == 0

    brief, _ = _brief(root, project)
    assert brief["what_changed"] == EARLIER


@pytest.mark.parametrize("counts, line", [
    ((0, 0, 0), None),
    ((1, 0, 0), "An answer is waiting on Decisions."),
    ((0, 2, 0), "2 suggestions are waiting on Decisions."),
    ((1, 1, 0), "2 items are waiting on Decisions."),
    ((0, 0, 1), "A plan change waits for your approval in the Project chat."),
    ((1, 0, 2), "An answer is waiting on Decisions. 2 plan changes wait for your approval in the Project chat."),
])
def test_t8_the_owner_line_names_each_owner_step_in_fixed_text(counts, line):
    from hermes_cli.owner_project_review_owner_line import owner_line

    assert owner_line(*counts) == line


def test_t9_the_freeze_counts_a_waiting_plan_change_without_changing_the_response_store(root, monkeypatch):
    project = _quiet_project(root)
    monkeypatch.setattr(time, "time", lambda: NOW)
    _chat(root, NAME, RID)
    _link(root, NAME, project.id)
    before = _store_state(root)

    assert review.review_tick(NOW) == 1

    assert _store_state(root) == before
    lines = _frozen_body(_open_review(root, project)).splitlines()
    cursor = next(i for i, text in enumerate(lines) if text.startswith("Owner decision cursor: "))
    assert lines[cursor + 1] == "Plan approvals waiting: 1"
    assert records.build_review_input(project, events.Cursors(), None)["plan_approvals"] == 1
    assert _store_state(root) == before


def test_t10_a_worker_pinned_count_reads_the_root_registry_and_the_root_response_store(root, monkeypatch):
    from hermes_cli.owner_project_review_owner_line import plan_approvals_waiting

    project = _quiet_project(root)
    with projects_db.connect_closing(root / "projects.db") as registry:
        garden = projects_db.create_project(registry, name="Garden Plan")
    monkeypatch.setattr(time, "time", lambda: NOW)
    names = ["raphael-owner-" + letter * 32 for letter in "abcde"]
    # Two proposals wait, one was applied, one chat ends with a question, and one waits in another Project's chat.
    for index in (0, 1, 2, 4):
        _chat(root, names[index], f"{RID}_{index}")
    _consume(root, names[2], f"{RID}_2")
    _owner_turn(root, names[3], "resp_room_question", "Which room should be booked?", NOW)
    for name in names[:4]:
        _link(root, name, project.id)
    _link(root, names[4], garden)
    assert review.review_tick(NOW) == 1
    task_id = _open_review(root, project)

    with _worker(root, monkeypatch, task_id) as task:
        # A store in the worker's profile home that differs from the root store: there, the applied proposal waits.
        profile = root / "profiles" / task.assignee
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
            counted = plan_approvals_waiting(registry, project.id)
        frozen = records.build_review_input(project, events.Cursors(), None)

    assert counted == frozen["plan_approvals"] == 2
    # The production proof, read against an independent count of the root stores.
    assert _independent_count(root / "projects.db", root / "response_store.db", project.id) == 2
    assert _independent_count(root / "projects.db", profile / "response_store.db", project.id) == 3
    # The worker reached the Project's own board through the root registry, and the tick froze the same count.
    assert "Book the room" in [item["title"] for item in frozen["sources"]["tasks"]]
    assert "Plan approvals waiting: 2" in _frozen_body(task_id).splitlines()
    assert not (profile / "projects.db").exists()
