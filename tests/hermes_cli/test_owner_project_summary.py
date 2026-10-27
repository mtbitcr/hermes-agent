"""Card A of the Project summary writer plan: Raphael writes the first summary of each Project.

The summary tick runs inside the gateway's dispatcher. For every live Project without a summary it opens one task for
the planning role on a board of its own. It stores the typed result of a finished task only after the result passes
every check, and a step that fails stores nothing, backs off, and stops after three failures in a row.

Each test imports the module itself, so each one fails on its own before the change.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import shutil
import sqlite3
import time
from pathlib import Path

import pytest

from agent import estop
from hermes_cli import kanban_db, kanban_risk_tier, owner_workspace as ow, projects_db
from hermes_cli.owner_text import check_owner_text
from plugins.dashboard_auth.raphael_workspace import model_policy

PLANNER = "raphael-planner"
BOARD = "project-summaries"
TITLE = "Write the Project summary: "
HOUR = 3600
T0 = 1_900_000_000

OUTCOME = "The room is booked for Tuesday at noon and the invitations are being written."
NEXT_ACTION = "Raphael sends the agenda to the team on Monday."
DECISION = "Please choose between the large room and the small room."
REFUSED = "I sent the plan to owner@example.test this morning."
REFUSED_NEXT_ACTION = "Claude Opus will send the agenda."
REFUSED_DECISION = "Call me on +1 415 555 0100."
LONG_TEXT = "The room is booked and the invitations are written. " * 9

# title -> (status, block_kind, seconds since completion)
_CARDS = {
    "Publish the weekly summary": ("done", None, 60),
    "Sort the old notes": ("done", None, 10 * 86_400),
    "Choose the launch date": ("blocked", "needs_input", None),
    "Draft the next milestone": ("ready", None, None),
}
_DESK_CARDS = {
    "Order the desk lamp": ("done", None, 60),
    "Pick the desk colour": ("blocked", "needs_input", None),
}


def _ops():
    from hermes_cli import owner_project_summary

    return owner_project_summary


class _ProcessDied(BaseException):
    """A gateway restart as the interrupted code sees it: nothing after this point runs."""


@pytest.fixture
def root(tmp_path, monkeypatch):
    """The shared root, entered as the default profile."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(
        model_policy, "configured_assignment_for",
        lambda profile: model_policy.assignment_for(profile, "anthropic"),
    )
    return home


@pytest.fixture
def owner(root):
    return ow.OwnerContext(actor="default", profile="default", session="run_owt")


def _clean(*texts: str) -> None:
    for text in texts:
        assert check_owner_text(text, "planner_prose") == text


def _valid(**changes) -> dict:
    summary = {"outcome": OUTCOME, "next_action": NEXT_ACTION, "owner_decision": DECISION}
    return {"project_summary": {**summary, **changes}}


def _rows(path: Path, sql: str, params=()) -> list:
    conn = sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True)
    try:
        return conn.execute(sql, params).fetchall()
    finally:
        conn.close()


def _board_db(root: Path, slug: str) -> Path:
    return root / "kanban" / "boards" / slug / "kanban.db"


@contextlib.contextmanager
def _board(root: Path, slug: str = BOARD):
    with kanban_db.connect_closing(_board_db(root, slug), board=slug) as conn:
        yield conn


def _board_meta(root: Path, slug: str = BOARD) -> dict:
    return json.loads((_board_db(root, slug).parent / "board.json").read_text(encoding="utf-8"))


def _fill_board(project: dict, cards: dict) -> None:
    now = int(time.time())
    with kanban_db.connect_closing(board=project["board"]) as conn:
        for title, (status, block_kind, done_ago) in cards.items():
            task_id = kanban_db.create_task(conn, title=title, project_id=project["project_id"])
            with kanban_db.write_txn(conn):
                conn.execute(
                    "UPDATE tasks SET status = ?, block_kind = ?, completed_at = ? WHERE id = ?",
                    (status, block_kind, None if done_ago is None else now - done_ago, task_id),
                )


def _project(owner, name: str, cards: dict | None = None) -> dict:
    """An owner Project with its own board, as the owner kernel creates it."""
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(ow, "_confirm", lambda *_args, **_kwargs: {"approved": True})
        project = ow.bootstrap(
            owner, idempotency_key="setup-" + name.lower().replace(" ", "-"), name=name,
        )
    assert project["ok"] is True
    project["slug"] = next(
        item["slug"] for item in ow.list_committed_projects(owner)
        if str(item["project_id"]) == project["project_id"]
    )
    if cards:
        _fill_board(project, cards)
    return project


def _registry_project(root: Path, name: str) -> str:
    """A Project that only the root registry knows: no board, so no steward snapshot."""
    with projects_db.connect_closing(root / "projects.db") as conn:
        return projects_db.create_project(conn, name=name)


def _summary_of(root: Path, project_id: str):
    with projects_db.connect_closing(root / "projects.db") as conn:
        return projects_db.get_project_summary(conn, project_id)


def _project_row(root: Path, project_id: str) -> list:
    return _rows(root / "projects.db", "SELECT * FROM projects WHERE id = ?", (project_id,))


_STATE_COLUMNS = ("open_task_id", "failures", "pending", "pending_urgent", "last_requested_at")


def _state(root: Path, project_id: str) -> dict | None:
    rows = _rows(
        root / "projects.db",
        f"SELECT {', '.join(_STATE_COLUMNS)} FROM project_summary_state WHERE project_id = ?",
        (project_id,),
    )
    return dict(zip(_STATE_COLUMNS, rows[0])) if rows else None


def _summary_tasks(root: Path, *, archived: bool = False) -> list:
    if not _board_db(root, BOARD).exists():
        return []
    with _board(root) as conn:
        ids = [
            row[0] for row in conn.execute(
                "SELECT id FROM tasks WHERE task_kind = 'work' ORDER BY created_at, rowid"
            )
        ]
        tasks = [kanban_db.get_task(conn, task_id) for task_id in ids]
    return [task for task in tasks if archived or task.status != "archived"]


def _finish(root: Path, task_id: str, metadata: dict | None, *, slug: str = BOARD) -> None:
    """What a dispatched worker does: claim the card, then complete it with metadata."""
    with _board(root, slug) as conn:
        assert kanban_db.claim_task(conn, task_id) is not None
        assert kanban_db.complete_task(
            conn, task_id, summary="Wrote the summary.", metadata=metadata,
        )


def _overrun(conn, task_id: str) -> None:
    assert kanban_db.claim_task(conn, task_id) is not None
    kanban_db._set_worker_pid(conn, task_id, os.getpid())
    started = int(time.time()) - 10_000
    with kanban_db.write_txn(conn):
        conn.execute("UPDATE tasks SET started_at = ? WHERE id = ?", (started, task_id))
        conn.execute(
            "UPDATE task_runs SET started_at = ? "
            "WHERE id = (SELECT current_run_id FROM tasks WHERE id = ?)",
            (started, task_id),
        )
    assert kanban_db.enforce_max_runtime(conn, signal_fn=lambda *_args: None) == [task_id]


def _fail(root: Path, monkeypatch, task_id: str, how: str) -> None:
    with _board(root) as conn:
        if how == "archived":
            assert kanban_db.archive_task(conn, task_id)
        elif how == "blocked":
            assert kanban_db.claim_task(conn, task_id) is not None
            assert kanban_db.block_task(conn, task_id, reason="The summary could not be written.")
        else:
            monkeypatch.setattr(kanban_db, "_pid_alive", lambda _pid: False)
            _overrun(conn, task_id)
            assert kanban_db.get_task(conn, task_id).status == "ready"
            _overrun(conn, task_id)
            assert kanban_db.get_task(conn, task_id).status == "blocked"


def test_new_project_gets_one_summary_task(root, owner):
    ops = _ops()
    project = _project(owner, "Garden Plan", _CARDS)
    project_id = project["project_id"]

    ops.summary_tick()
    ops.summary_tick()

    [task] = _summary_tasks(root, archived=True)
    assert task.title == "Write the Project summary: Garden Plan"
    assert task.assignee == PLANNER
    assert task.status == "ready"
    assert task.execution_tier == "routine"
    assert task.risk_tier == 0
    assert task.workspace_kind == "scratch"
    assert task.owned_paths == []
    assert task.project_id is None

    route = model_policy.task_assignment_for(PLANNER, "anthropic", "routine")
    assert (task.provider_override, task.model_override) == (route.provider, route.model)
    assert task.model_policy_lock
    assert kanban_db.policy_lock_error(
        task.model_policy_lock, task.assignee, task.provider_override, task.model_override,
        task.reasoning_effort, task.execution_tier,
    ) is None

    snapshot = ow.project_steward_snapshot(project_id=project_id)
    assert snapshot["progress"] and snapshot["decisions_needed"]
    for item in (*snapshot["progress"], *snapshot["decisions_needed"]):
        assert item["title"] in task.body
    assert "Publish the weekly summary" in task.body
    assert "Choose the launch date" in task.body
    assert "Sort the old notes" not in task.body

    meta = json.loads((_board_db(root, BOARD).parent / "board.json").read_text(encoding="utf-8"))
    assert meta["project_id"] is None
    assert meta["dispatch_enabled"] is True
    with _board(root, project["board"]) as conn:
        assert not [
            row for row in conn.execute("SELECT title FROM tasks") if row[0].startswith(TITLE)
        ]

    state = _state(root, project_id)
    assert state["open_task_id"] == task.id
    assert (state["pending"], state["pending_urgent"], state["failures"]) == (0, 0, 0)
    assert state["last_requested_at"] is not None


def test_valid_result_is_stored_on_the_next_tick(root, owner):
    ops = _ops()
    _clean(OUTCOME, NEXT_ACTION, DECISION)
    project = _project(owner, "Garden Plan", _CARDS)
    project_id = project["project_id"]
    ops.summary_tick()
    [task] = _summary_tasks(root)

    _finish(root, task.id, _valid())
    assert _summary_of(root, project_id) is None

    ops.summary_tick()

    stored = _summary_of(root, project_id)
    assert stored["revision"] == 1
    assert (stored["outcome"], stored["next_action"], stored["owner_decision"]) == (
        OUTCOME, NEXT_ACTION, DECISION,
    )
    shown = ow.read_project_snapshot(owner, project["slug"], project_summary=True)
    assert shown["raphael_summary"] == stored
    assert _summary_tasks(root) == []
    [archived] = _summary_tasks(root, archived=True)
    assert (archived.id, archived.status) == (task.id, "archived")
    assert _state(root, project_id)["open_task_id"] is None
    assert _state(root, project_id)["failures"] == 0

    ops.summary_tick()

    assert _summary_of(root, project_id)["revision"] == 1
    assert len(_summary_tasks(root, archived=True)) == 1


@pytest.mark.parametrize(
    ("given", "stored"),
    [
        ({"owner_decision": DECISION}, DECISION),
        ({"owner_decision": None}, None),
        ({"owner_decision": "   "}, None),
        ({}, None),
    ],
    ids=["text", "null", "blank", "omitted"],
)
def test_owner_decision_is_optional(root, owner, given, stored):
    ops = _ops()
    project_id = _project(owner, "Garden Plan")["project_id"]
    ops.summary_tick()
    [task] = _summary_tasks(root)
    summary = {"outcome": OUTCOME, "next_action": NEXT_ACTION, **given}

    _finish(root, task.id, {"project_summary": summary})
    ops.summary_tick()

    assert _summary_of(root, project_id)["owner_decision"] == stored


_INVALID = {
    "no_key": ({"notes": "The summary is in the comment."}, None),
    "not_an_object": ({"project_summary": [OUTCOME, NEXT_ACTION]}, None),
    "no_metadata": (None, None),
    "extra_key": (_valid(confidence="high"), None),
    "missing_outcome": ({"project_summary": {"next_action": NEXT_ACTION}}, None),
    "missing_next_action": ({"project_summary": {"outcome": OUTCOME}}, None),
    "outcome_too_long": (_valid(outcome=LONG_TEXT[:401]), None),
    "next_action_too_long": (_valid(next_action=LONG_TEXT[:241]), None),
    "decision_too_long": (_valid(owner_decision=LONG_TEXT[:241]), None),
    "outcome_blank": (_valid(outcome="   "), None),
    "next_action_empty": (_valid(next_action=""), None),
    "outcome_not_text": (_valid(outcome=42), None),
    "outcome_null": (_valid(outcome=None), None),
    "next_action_not_text": (_valid(next_action=["Send the agenda."]), None),
    "decision_not_text": (_valid(owner_decision={"text": DECISION}), None),
    "outcome_refused": (_valid(outcome=REFUSED), None),
    "next_action_refused": (_valid(next_action=REFUSED_NEXT_ACTION), None),
    "decision_refused": (_valid(owner_decision=REFUSED_DECISION), None),
    "reassigned_after_completion": (_valid(), "default"),
}


@pytest.mark.parametrize(("metadata", "reassigned_to"), _INVALID.values(), ids=list(_INVALID))
def test_invalid_result_stores_nothing(root, owner, metadata, reassigned_to):
    ops = _ops()
    for text in (REFUSED, REFUSED_NEXT_ACTION, REFUSED_DECISION):
        assert check_owner_text(text, "planner_prose") != text
    project_id = _project(owner, "Garden Plan")["project_id"]
    ops.summary_tick(now=T0)
    [task] = _summary_tasks(root)
    row_before = _project_row(root, project_id)

    _finish(root, task.id, metadata)
    if reassigned_to:
        with _board(root) as conn, kanban_db.write_txn(conn):
            conn.execute("UPDATE tasks SET assignee = ? WHERE id = ?", (reassigned_to, task.id))
    ops.summary_tick(now=T0 + 10)

    assert _summary_of(root, project_id) is None
    assert _project_row(root, project_id) == row_before
    state = _state(root, project_id)
    assert (state["open_task_id"], state["failures"]) == (None, 1)
    assert _summary_tasks(root) == []


@pytest.mark.parametrize("how", ["blocked", "timed_out", "archived"])
def test_failed_step_stores_nothing_backs_off_and_stops(root, owner, monkeypatch, how):
    ops = _ops()
    project_id = _project(owner, "Garden Plan")["project_id"]
    row_before = _project_row(root, project_id)

    def created() -> int:
        return len(_summary_tasks(root, archived=True))

    def check(failures: int, tasks: int) -> None:
        assert _summary_of(root, project_id) is None
        assert _project_row(root, project_id) == row_before
        state = _state(root, project_id)
        assert (state["open_task_id"], state["failures"]) == (None, failures)
        assert created() == tasks
        assert _summary_tasks(root) == []

    ops.summary_tick(now=T0)
    [first] = _summary_tasks(root)
    _fail(root, monkeypatch, first.id, how)

    first_failed = T0 + 10
    ops.summary_tick(now=first_failed)
    check(failures=1, tasks=1)
    ops.summary_tick(now=first_failed + HOUR - 1)
    check(failures=1, tasks=1)

    second_at = first_failed + HOUR
    ops.summary_tick(now=second_at)
    assert created() == 2
    [second] = _summary_tasks(root)
    assert _state(root, project_id)["open_task_id"] == second.id
    _fail(root, monkeypatch, second.id, how)

    second_failed = second_at + 10
    ops.summary_tick(now=second_failed)
    check(failures=2, tasks=2)
    ops.summary_tick(now=second_failed + 2 * HOUR - 1)
    check(failures=2, tasks=2)

    third_at = second_failed + 2 * HOUR
    ops.summary_tick(now=third_at)
    assert created() == 3
    [third] = _summary_tasks(root)
    _fail(root, monkeypatch, third.id, how)

    third_failed = third_at + 10
    ops.summary_tick(now=third_failed)
    check(failures=3, tasks=3)
    ops.summary_tick(now=third_failed + 30 * 86_400)
    check(failures=3, tasks=3)


def test_timed_out_run_that_is_requeued_is_not_yet_a_failure(root, owner, monkeypatch):
    ops = _ops()
    project_id = _project(owner, "Garden Plan")["project_id"]
    ops.summary_tick(now=T0)
    [task] = _summary_tasks(root)
    monkeypatch.setattr(kanban_db, "_pid_alive", lambda _pid: False)
    with _board(root) as conn:
        _overrun(conn, task.id)

    ops.summary_tick(now=T0 + HOUR * 5)

    state = _state(root, project_id)
    assert (state["open_task_id"], state["failures"]) == (task.id, 0)
    assert len(_summary_tasks(root, archived=True)) == 1


def test_task_with_the_same_title_made_by_anyone_else_is_never_harvested(root, owner):
    ops = _ops()
    own_outcome = "The garden beds are planted and the watering plan is set."
    other_outcome = "The room is booked for Wednesday and nothing else is planned."
    _clean(own_outcome, other_outcome)
    project = _project(owner, "Garden Plan", _CARDS)
    project_id = project["project_id"]
    title = TITLE + "Garden Plan"
    kanban_db.init_db()
    kanban_db.create_board(slug=BOARD, name="Project summaries")

    for slug in (BOARD, project["board"], kanban_db.DEFAULT_BOARD):
        with kanban_db.connect_closing(board=slug) as conn:
            other_id = kanban_db.create_task(
                conn, title=title, assignee=PLANNER, created_by="someone-else",
                project_id=project_id if slug == project["board"] else None,
            )
            assert kanban_db.complete_task(
                conn, other_id, summary="Wrote the summary.",
                metadata=_valid(outcome=other_outcome),
            )

    ops.summary_tick()
    ops.summary_tick()

    own = [task for task in _summary_tasks(root) if task.created_by != "someone-else"]
    [own_task] = own
    assert own_task.title == title
    assert _state(root, project_id)["open_task_id"] == own_task.id
    assert _summary_of(root, project_id) is None

    _finish(root, own_task.id, _valid(outcome=own_outcome))
    ops.summary_tick()

    assert _summary_of(root, project_id)["outcome"] == own_outcome
    assert _summary_of(root, project_id)["revision"] == 1


def test_at_most_three_tasks_per_tick_and_none_for_projects_that_are_gone(root, owner):
    ops = _ops()
    live = ["Alpha", "Bravo", "Charlie", "Delta", "Echo"]
    for name in live[:4]:
        _project(owner, name)
    cancelled = _project(owner, "Echo")["project_id"]
    archived = _project(owner, "Foxtrot")["project_id"]
    removing = _project(owner, "Golf")["project_id"]
    with projects_db.connect_closing(root / "projects.db") as conn:
        projects_db.archive_project(conn, archived)
        assert projects_db.record_removal_operation(
            conn, project_id=removing, idempotency_key="remove-golf", action="remove",
        )
        assert projects_db.record_removal_operation(
            conn, project_id=cancelled, idempotency_key="remove-echo", action="remove",
            phase="cancelled",
        )

    ops.summary_tick(now=T0)
    assert len(_summary_tasks(root)) == 3
    ops.summary_tick(now=T0 + 1)
    assert len(_summary_tasks(root)) == 5
    ops.summary_tick(now=T0 + 2)

    titles = sorted(task.title for task in _summary_tasks(root, archived=True))
    assert titles == sorted(TITLE + name for name in live)
    for project_id in (archived, removing):
        state = _state(root, project_id)
        assert not state or state["open_task_id"] is None
        assert _summary_of(root, project_id) is None


def test_project_the_owner_paused_still_gets_a_summary_task(root, owner):
    ops = _ops()
    project = _project(owner, "Garden Plan", _CARDS)
    kanban_db.write_board_dispatch_state(
        project["board"], dispatch_enabled=False, dispatch_paused_by_owner=True,
    )

    ops.summary_tick(now=T0)

    [task] = _summary_tasks(root)
    assert _state(root, project["project_id"])["open_task_id"] == task.id
    meta = kanban_db.read_board_metadata(project["board"])
    assert (meta["dispatch_enabled"], meta["dispatch_paused_by_owner"]) == (False, True)


def test_project_that_already_has_a_summary_is_not_asked_for_a_first_one(root):
    ops = _ops()
    project_id = _registry_project(root, "Garden Plan")
    with projects_db.connect_closing(root / "projects.db") as conn:
        projects_db.write_project_summary(conn, project_id, outcome=OUTCOME, next_action=NEXT_ACTION)

    ops.summary_tick(now=T0)

    assert _summary_tasks(root, archived=True) == []
    state = _state(root, project_id)
    assert (state["open_task_id"], state["pending"], state["pending_urgent"]) == (None, 0, 0)
    assert _summary_of(root, project_id)["revision"] == 1


def test_tick_that_loses_the_race_for_a_project_leaves_the_other_task_open(root, owner, monkeypatch):
    ops = _ops()
    project_id = _project(owner, "Garden Plan")["project_id"]
    create_task = kanban_db.create_task

    def create_then_lose(*args, **kwargs):
        task_id = create_task(*args, **kwargs)
        with projects_db.connect_closing(root / "projects.db") as conn, projects_db.write_txn(conn):
            conn.execute(
                "UPDATE project_summary_state SET open_task_id = 't_other', pending = 0 WHERE project_id = ?",
                (project_id,),
            )
        return task_id

    monkeypatch.setattr(kanban_db, "create_task", create_then_lose)

    ops.summary_tick(now=T0)

    assert _state(root, project_id)["open_task_id"] == "t_other"
    assert _summary_tasks(root) == []
    assert len(_summary_tasks(root, archived=True)) == 1


def test_planning_role_without_a_route_is_a_failure_that_backs_off(root, owner, monkeypatch):
    ops = _ops()
    project_id = _project(owner, "Garden Plan")["project_id"]
    working_route = model_policy.configured_assignment_for

    def no_route(_profile):
        raise ValueError("no route is configured")

    monkeypatch.setattr(model_policy, "configured_assignment_for", no_route)
    ops.summary_tick(now=T0)

    state = _state(root, project_id)
    assert (state["open_task_id"], state["failures"], state["pending"]) == (None, 1, 1)
    assert state["last_requested_at"] is None
    assert _summary_tasks(root, archived=True) == []

    monkeypatch.setattr(model_policy, "configured_assignment_for", working_route)
    ops.summary_tick(now=T0 + HOUR - 1)
    assert _summary_tasks(root, archived=True) == []
    ops.summary_tick(now=T0 + HOUR)

    [task] = _summary_tasks(root)
    assert _state(root, project_id)["open_task_id"] == task.id


def test_summary_board_that_is_gone_counts_as_a_failure(root, owner):
    ops = _ops()
    project_id = _project(owner, "Garden Plan")["project_id"]
    ops.summary_tick(now=T0)
    shutil.rmtree(_board_db(root, BOARD).parent)

    ops.summary_tick(now=T0 + 10)

    state = _state(root, project_id)
    assert (state["open_task_id"], state["failures"], state["pending"]) == (None, 1, 1)
    assert _summary_of(root, project_id) is None


def test_stored_runtime_cap_is_the_kernel_time_box_of_the_pinned_card(root, owner):
    ops = _ops()
    _project(owner, "Garden Plan", _CARDS)

    ops.summary_tick(now=T0)

    [task] = _summary_tasks(root)
    assert task.model_policy_lock
    kind = kanban_risk_tier.card_work_kind(PLANNER, [], "routine", False)
    stored = _rows(
        _board_db(root, BOARD), "SELECT max_runtime_seconds FROM tasks WHERE id = ?", (task.id,),
    )
    assert stored == [(kanban_risk_tier.pinned_time_box_seconds(kind),)]


def test_task_made_before_it_was_recorded_is_adopted_after_a_restart(root, owner, monkeypatch):
    ops = _ops()
    project_id = _project(owner, "Garden Plan", _CARDS)["project_id"]
    create_task = kanban_db.create_task

    def create_then_die(*args, **kwargs):
        create_task(*args, **kwargs)
        raise _ProcessDied

    monkeypatch.setattr(kanban_db, "create_task", create_then_die)
    with pytest.raises(_ProcessDied):
        ops.summary_tick(now=T0)
    monkeypatch.setattr(kanban_db, "create_task", create_task)
    [made] = _summary_tasks(root)
    assert _state(root, project_id)["open_task_id"] is None

    ops.summary_tick(now=T0 + 5)
    ops.summary_tick(now=T0 + 10)

    assert [task.id for task in _summary_tasks(root, archived=True)] == [made.id]
    assert _summary_tasks(root)[0].status == "ready"
    assert _state(root, project_id)["open_task_id"] == made.id


def test_two_ticks_racing_for_one_project_keep_the_one_task_they_share(root, owner, monkeypatch):
    ops = _ops()
    project_id = _project(owner, "Garden Plan", _CARDS)["project_id"]
    create_task = kanban_db.create_task
    made = []

    def create_while_the_other_tick_runs(*args, **kwargs):
        task_id = create_task(*args, **kwargs)
        if not made:
            made.append(task_id)
            ops.summary_tick(now=T0)
        return task_id

    monkeypatch.setattr(kanban_db, "create_task", create_while_the_other_tick_runs)

    ops.summary_tick(now=T0)

    assert [task.id for task in _summary_tasks(root, archived=True)] == made
    [task] = _summary_tasks(root)
    assert task.status == "ready"
    assert _state(root, project_id)["open_task_id"] == task.id


def test_board_made_but_never_switched_on_is_switched_on_after_a_restart(root, owner, monkeypatch):
    ops = _ops()
    project_id = _project(owner, "Garden Plan", _CARDS)["project_id"]
    switch_on = kanban_db.write_board_dispatch_state

    def die_before_the_switch(board, **_kwargs):
        raise _ProcessDied

    monkeypatch.setattr(kanban_db, "write_board_dispatch_state", die_before_the_switch)
    with pytest.raises(_ProcessDied):
        ops.summary_tick(now=T0)
    monkeypatch.setattr(kanban_db, "write_board_dispatch_state", switch_on)
    assert _board_db(root, BOARD).exists()
    assert _board_meta(root)["dispatch_enabled"] is False
    assert _summary_tasks(root) == []

    ops.summary_tick(now=T0 + 5)

    meta = _board_meta(root)
    assert (meta["dispatch_enabled"], meta["dispatch_paused_by_owner"], meta["archived"]) == (
        True, False, False,
    )
    assert meta["project_id"] is None
    [task] = _summary_tasks(root)
    assert _state(root, project_id)["open_task_id"] == task.id

    kanban_db.write_board_dispatch_state(BOARD, dispatch_enabled=False, dispatch_paused_by_owner=True)
    desk = _project(owner, "Planning Desk", _DESK_CARDS)
    ops.summary_tick(now=T0 + 10)

    meta = _board_meta(root)
    assert (meta["dispatch_enabled"], meta["dispatch_paused_by_owner"]) == (False, True)
    assert len(_summary_tasks(root)) == 2
    assert _state(root, desk["project_id"])["open_task_id"] is not None


def test_summary_board_that_belongs_to_a_project_is_refused_and_logged_once(root, owner, caplog):
    ops = _ops()
    colliding = _project(owner, "Project summaries")
    garden = _project(owner, "Garden Plan", _CARDS)
    assert colliding["board"] == BOARD
    meta_before = _board_meta(root)
    assert meta_before["project_id"] == colliding["project_id"]
    boards_before = sorted(path.name for path in (root / "kanban" / "boards").iterdir())

    with caplog.at_level(logging.WARNING):
        ops.summary_tick(now=T0)
        ops.summary_tick(now=T0 + HOUR)

    assert [
        task for task in _summary_tasks(root, archived=True) if task.title.startswith(TITLE)
    ] == []
    assert _board_meta(root) == meta_before
    assert sorted(path.name for path in (root / "kanban" / "boards").iterdir()) == boards_before
    for project_id in (colliding["project_id"], garden["project_id"]):
        state = _state(root, project_id)
        assert state is None or (state["open_task_id"], state["failures"]) == (None, 0)
    refusals = [
        record for record in caplog.records
        if record.name == ops.logger.name and record.levelno >= logging.WARNING
    ]
    assert len(refusals) == 1
    assert BOARD in refusals[0].getMessage()


def test_summary_board_whose_owner_cannot_be_read_is_refused(root, owner, caplog):
    ops = _ops()
    _project(owner, "Garden Plan", _CARDS)
    ops.summary_tick(now=T0)
    desk = _project(owner, "Planning Desk", _DESK_CARDS)["project_id"]
    (_board_db(root, BOARD).parent / "board.json").write_text("{not json", encoding="utf-8")

    with caplog.at_level(logging.WARNING):
        ops.summary_tick(now=T0 + 10)
        ops.summary_tick(now=T0 + HOUR)

    assert len(_summary_tasks(root, archived=True)) == 1
    state = _state(root, desk)
    assert state is None or (state["open_task_id"], state["failures"]) == (None, 0)
    refusals = [
        record for record in caplog.records
        if record.name == ops.logger.name and record.levelno >= logging.WARNING
    ]
    assert len(refusals) == 1


@pytest.mark.parametrize("delayed", [1, 2], ids=["first_failure", "second_failure"])
def test_backoff_waits_from_the_failure_not_from_the_request(root, owner, monkeypatch, delayed):
    ops = _ops()
    project_id = _project(owner, "Garden Plan", _CARDS)["project_id"]
    now = T0
    ops.summary_tick(now=now)

    for failure in range(1, delayed + 1):
        [task] = _summary_tasks(root)
        _fail(root, monkeypatch, task.id, "blocked")
        failed_at = now + 3 * HOUR
        ops.summary_tick(now=failed_at)

        assert _state(root, project_id)["failures"] == failure
        assert len(_summary_tasks(root, archived=True)) == failure
        wait = ops.backoff_seconds(failure)
        ops.summary_tick(now=failed_at + wait - 1)
        assert len(_summary_tasks(root, archived=True)) == failure
        ops.summary_tick(now=failed_at + wait)
        assert len(_summary_tasks(root, archived=True)) == failure + 1
        now = failed_at + wait


@pytest.mark.parametrize("how", ["unavailable", "no_board"])
def test_unreadable_steward_snapshot_is_a_failed_attempt_with_no_task(root, owner, monkeypatch, how):
    ops = _ops()
    if how == "unavailable":
        project_id = _project(owner, "Garden Plan", _CARDS)["project_id"]

        def refuse(**_kwargs):
            raise ow.OwnerWorkspaceError("snapshot_unavailable", "the Project board is unavailable")

        monkeypatch.setattr(ow, "project_steward_snapshot", refuse)
    else:
        project_id = _registry_project(root, "Garden Plan")
    row_before = _project_row(root, project_id)

    ops.summary_tick(now=T0)

    state = _state(root, project_id)
    assert (state["open_task_id"], state["failures"], state["pending"]) == (None, 1, 1)
    assert _summary_tasks(root, archived=True) == []
    ops.summary_tick(now=T0 + HOUR - 1)
    assert _state(root, project_id)["failures"] == 1
    ops.summary_tick(now=T0 + HOUR)
    assert _state(root, project_id)["failures"] == 2
    assert _summary_tasks(root, archived=True) == []
    assert _summary_of(root, project_id) is None
    assert _project_row(root, project_id) == row_before


def test_nothing_is_published_after_a_failed_snapshot_attempt(root, owner, monkeypatch):
    ops = _ops()
    project_id = _project(owner, "Garden Plan", _CARDS)["project_id"]

    def refuse(**_kwargs):
        raise ow.OwnerWorkspaceError("snapshot_unavailable", "the Project board is unavailable")

    monkeypatch.setattr(ow, "project_steward_snapshot", refuse)

    ops.summary_tick(now=T0)
    for task in _summary_tasks(root):
        _finish(root, task.id, _valid())
    ops.summary_tick(now=T0 + 10)

    assert _summary_of(root, project_id) is None
    assert _summary_tasks(root, archived=True) == []


def _leave(root: Path, project_id: str, how: str, *, undo: bool = False) -> None:
    with projects_db.connect_closing(root / "projects.db") as registry:
        if how == "archived":
            assert (projects_db.restore_project if undo else projects_db.archive_project)(registry, project_id)
        elif undo:
            assert projects_db.record_removal_operation(
                registry, project_id=project_id, idempotency_key="cancel-garden", action="remove",
                phase="cancelled",
            )
        else:
            assert projects_db.record_removal_operation(
                registry, project_id=project_id, idempotency_key="remove-garden", action="remove",
            )


@pytest.mark.parametrize("how", ["archived", "removal_started"])
def test_project_that_leaves_while_its_request_is_open_gets_no_summary(root, owner, how):
    ops = _ops()
    project_id = _project(owner, "Garden Plan", _CARDS)["project_id"]
    ops.summary_tick(now=T0)
    [task] = _summary_tasks(root)
    _finish(root, task.id, _valid())
    _leave(root, project_id, how)
    row_before = _project_row(root, project_id)

    ops.summary_tick(now=T0 + 10)
    ops.summary_tick(now=T0 + 5 * HOUR)

    assert _summary_of(root, project_id) is None
    assert _project_row(root, project_id) == row_before
    state = _state(root, project_id)
    assert (state["open_task_id"], state["failures"]) == (None, 0)
    assert _summary_tasks(root) == []
    assert len(_summary_tasks(root, archived=True)) == 1

    _leave(root, project_id, how, undo=True)
    ops.summary_tick(now=T0 + 6 * HOUR)

    [again] = _summary_tasks(root)
    assert again.id != task.id
    assert _state(root, project_id)["open_task_id"] == again.id
    assert _summary_of(root, project_id) is None


def test_project_archived_while_its_summary_is_being_stored_gets_none(root, owner, monkeypatch):
    ops = _ops()
    project_id = _project(owner, "Garden Plan", _CARDS)["project_id"]
    ops.summary_tick(now=T0)
    [task] = _summary_tasks(root)
    _finish(root, task.id, _valid())
    parse = ops.parse_summary_result

    def parse_then_archive(conn, task_id):
        result = parse(conn, task_id)
        _leave(root, project_id, "archived")
        return result

    monkeypatch.setattr(ops, "parse_summary_result", parse_then_archive)

    ops.summary_tick(now=T0 + 10)

    assert _summary_of(root, project_id) is None
    state = _state(root, project_id)
    assert (state["open_task_id"], state["failures"]) == (None, 0)
    assert _summary_tasks(root) == []


def test_project_whose_permanent_removal_has_finished_gets_no_summary(root, owner):
    ops = _ops()
    garden = _project(owner, "Garden Plan", _CARDS)["project_id"]
    desk = _project(owner, "Planning Desk", _DESK_CARDS)["project_id"]
    ops.summary_tick(now=T0)
    for task in _summary_tasks(root):
        _finish(root, task.id, _valid())
    with projects_db.connect_closing(root / "projects.db") as registry:
        assert projects_db.record_removal_operation(
            registry, project_id=garden, idempotency_key="remove-garden", action="start",
            phase="done", mode="permanent",
        )
        assert projects_db.get_active_removal_operation(registry, garden) is None

    ops.summary_tick(now=T0 + 10)
    ops.summary_tick(now=T0 + 5 * HOUR)

    assert _summary_of(root, garden) is None
    assert _summary_of(root, desk) is not None
    state = _state(root, garden)
    assert (state["open_task_id"], state["failures"]) == (None, 0)
    assert _summary_tasks(root) == []


def _drive_dispatcher(monkeypatch, *, ticks: int, after_dispatch=None) -> dict:
    """Run the gateway's embedded dispatcher loop for ``ticks`` iterations and count what each one runs.

    ``after_dispatch`` runs right after each dispatch pass returns, before the loop goes on.
    """
    from gateway.run import GatewayRunner
    import hermes_cli.config as cfg_mod

    kanban_db.init_db()
    kanban_db.create_board(slug=kanban_db.DEFAULT_BOARD, name="Default")
    runner = object.__new__(GatewayRunner)
    runner._running = True
    config = {"dispatch_in_gateway": True, "dispatch_interval_seconds": 1, "auto_decompose": False}
    monkeypatch.setattr(cfg_mod, "load_config", lambda: {"kanban": dict(config)})
    seen = {"reap_worker_zombies": 0, "_tick_once": 0, "_ready_nonempty": 0}

    async def _to_thread(fn, *args, **kwargs):
        name = getattr(fn, "__name__", "")
        if name in seen:
            seen[name] += 1
        if name == "reap_worker_zombies" and seen[name] >= ticks:
            runner._running = False
        result = fn(*args, **kwargs)
        if name == "_tick_once" and after_dispatch is not None:
            after_dispatch()
        return result

    async def _sleep(_delay):
        return None

    monkeypatch.setattr("gateway.run.asyncio.to_thread", _to_thread)
    monkeypatch.setattr("gateway.run.asyncio.sleep", _sleep)
    asyncio.run(asyncio.wait_for(runner._kanban_dispatcher_watcher(), timeout=60.0))
    assert seen["reap_worker_zombies"] == ticks
    return seen


def test_dispatcher_runs_the_summary_tick_once_per_tick(root, monkeypatch):
    ops = _ops()
    calls = []
    monkeypatch.setattr(ops, "summary_tick", lambda *args, **kwargs: calls.append((args, kwargs)))

    seen = _drive_dispatcher(monkeypatch, ticks=3)

    assert seen["_tick_once"] == 3
    assert len(calls) == 3


def test_dispatcher_does_not_run_the_summary_tick_while_paused(root, monkeypatch):
    ops = _ops()
    calls = []
    monkeypatch.setattr(ops, "summary_tick", lambda *args, **kwargs: calls.append((args, kwargs)))
    monkeypatch.setattr(estop, "_logged_components", set())
    estop.engage(reason="test")
    try:
        seen = _drive_dispatcher(monkeypatch, ticks=2)
    finally:
        estop.disengage()

    assert seen["_tick_once"] == 0
    assert calls == []


def test_dispatcher_does_not_run_the_summary_tick_when_pause_comes_during_dispatch(root, monkeypatch):
    ops = _ops()
    calls = []
    monkeypatch.setattr(ops, "summary_tick", lambda *args, **kwargs: calls.append((args, kwargs)))
    monkeypatch.setattr(estop, "_logged_components", set())
    try:
        seen = _drive_dispatcher(
            monkeypatch, ticks=1, after_dispatch=lambda: estop.engage(reason="test"),
        )
        assert estop.is_engaged()
    finally:
        estop.disengage()

    assert seen["_tick_once"] == 1
    assert calls == []


def test_summary_tick_that_raises_does_not_stop_dispatch(root, monkeypatch, caplog):
    ops = _ops()
    boom = RuntimeError("the summary tick broke")

    def _raise(*_args, **_kwargs):
        raise boom

    monkeypatch.setattr(ops, "summary_tick", _raise)

    with caplog.at_level(logging.WARNING):
        seen = _drive_dispatcher(monkeypatch, ticks=2)

    assert seen["_tick_once"] == 2
    assert seen["_ready_nonempty"] == 2
    assert [record for record in caplog.records if record.exc_info and record.exc_info[1] is boom]
    assert not [
        record for record in caplog.records if "unexpected watcher error" in record.getMessage()
    ]


def _enter_planning_worker(monkeypatch, root: Path, slug: str, task_id: str) -> None:
    """Take on the store pins kanban_db_dispatch._default_spawn gives a worker.

    The board pins are resolved first, in the dispatcher's own environment.
    """
    pins = {
        "HERMES_KANBAN_DB": str(kanban_db.kanban_db_path(board=slug)),
        "HERMES_KANBAN_WORKSPACES_ROOT": str(kanban_db.workspaces_root(board=slug)),
        "HERMES_KANBAN_BOARD": slug,
    }
    monkeypatch.setenv("HERMES_HOME", str(root / "profiles" / PLANNER))
    monkeypatch.setenv("HERMES_PROFILE", PLANNER)
    monkeypatch.setenv("HERMES_KANBAN_TASK", task_id)
    for name, value in pins.items():
        monkeypatch.setenv(name, value)


def test_tick_under_the_worker_environment_reaches_every_store(root, owner, monkeypatch):
    ops = _ops()
    garden_outcome = "The garden beds are planted and the watering plan is set."
    desk_outcome = "The desk lamp is ordered and the colour waits for a choice."
    _clean(garden_outcome, desk_outcome)
    garden = _project(owner, "Garden Plan", _CARDS)
    desk = _project(owner, "Planning Desk", _DESK_CARDS)
    spare = _project(owner, "Spare Room")
    attic = _project(owner, "Attic Annex")
    _leave(root, attic["project_id"], "archived")
    with _board(root, desk["board"]) as conn:
        worker_task = kanban_db.create_task(
            conn, title="Plan the next milestone", assignee=PLANNER,
            project_id=desk["project_id"],
        )
    profile_home = root / "profiles" / PLANNER
    profile_home.mkdir(parents=True)
    with projects_db.connect_closing(profile_home / "projects.db"):
        pass
    _enter_planning_worker(monkeypatch, root, desk["board"], worker_task)
    pinned = {name: os.environ[name] for name in ("HERMES_KANBAN_DB", "HERMES_KANBAN_BOARD")}
    own_board = _board_db(root, desk["board"])
    assert Path(pinned["HERMES_KANBAN_DB"]).resolve() == own_board.resolve()
    assert not _board_db(root, BOARD).exists()
    own_cards = _rows(own_board, "SELECT id, title, status FROM tasks ORDER BY id")

    ops.summary_tick()

    by_title = {task.title: task for task in _summary_tasks(root)}
    assert sorted(by_title) == [
        TITLE + "Garden Plan", TITLE + "Planning Desk", TITLE + "Spare Room",
    ]
    garden_task = by_title[TITLE + "Garden Plan"]
    desk_task = by_title[TITLE + "Planning Desk"]
    spare_task = by_title[TITLE + "Spare Room"]
    assert "Publish the weekly summary" in garden_task.body
    assert "Order the desk lamp" not in garden_task.body
    assert "Order the desk lamp" in desk_task.body
    assert "Publish the weekly summary" not in desk_task.body
    meta = _board_meta(root)
    assert (meta["project_id"], meta["dispatch_enabled"]) == (None, True)
    assert _rows(own_board, "SELECT id, title, status FROM tasks ORDER BY id") == own_cards
    assert {name: os.environ[name] for name in pinned} == pinned

    kanban_db.write_board_metadata(BOARD, dispatch_enabled=False)
    _leave(root, spare["project_id"], "archived")
    _leave(root, attic["project_id"], "archived", undo=True)
    _finish(root, garden_task.id, _valid(outcome=garden_outcome))
    _finish(root, desk_task.id, _valid(outcome=desk_outcome))
    ops.summary_tick()

    stored = _rows(
        root / "projects.db",
        "SELECT project_id, revision, outcome, next_action, owner_decision "
        "FROM project_summaries ORDER BY project_id",
    )
    assert stored == sorted([
        (garden["project_id"], 1, garden_outcome, NEXT_ACTION, DECISION),
        (desk["project_id"], 1, desk_outcome, NEXT_ACTION, DECISION),
    ])
    meta = _board_meta(root)
    assert (meta["project_id"], meta["dispatch_enabled"]) == (None, True)
    [attic_task] = _summary_tasks(root)
    assert attic_task.title == TITLE + "Attic Annex"
    assert _rows(
        root / "projects.db",
        "SELECT project_id, open_task_id FROM project_summary_state WHERE open_task_id IS NOT NULL",
    ) == [(attic["project_id"], attic_task.id)]
    assert _rows(
        root / "projects.db",
        "SELECT open_task_id, failures FROM project_summary_state WHERE project_id = ?",
        (spare["project_id"],),
    ) == [(None, 0)]
    assert _rows(_board_db(root, BOARD), "SELECT status FROM tasks WHERE id = ?", (spare_task.id,)) == [
        ("archived",),
    ]
    assert _rows(own_board, "SELECT id, title, status FROM tasks ORDER BY id") == own_cards
    assert _rows(profile_home / "projects.db", "SELECT count(*) FROM project_summaries") == [(0,)]
    assert _rows(profile_home / "projects.db", "SELECT count(*) FROM project_summary_state") == [(0,)]
    assert {name: os.environ[name] for name in pinned} == pinned


# Owner rule 1 of 2026-10-27: the one check. A task of the Project that is already on the summary board is the
# Project's open task; a tick records it before it reads anything else, so no failure can lose its identity.


def _crash_after_the_task_is_made(ops, monkeypatch, now: int) -> None:
    """Interrupt a tick right after create_task commits and before the state row records the task."""
    create_task = kanban_db.create_task

    def create_then_die(*args, **kwargs):
        create_task(*args, **kwargs)
        raise _ProcessDied

    monkeypatch.setattr(kanban_db, "create_task", create_then_die)
    try:
        with pytest.raises(_ProcessDied):
            ops.summary_tick(now=now)
    finally:
        monkeypatch.setattr(kanban_db, "create_task", create_task)


class _Reads:
    """The two reads a tick makes before it publishes: counts them, and refuses the one named in ``failing``."""

    def __init__(self, monkeypatch):
        self.calls: list = []
        self.failing: str | None = None
        route, snapshot = ow._resolved_route_pin, ow.project_steward_snapshot

        def read_route(*args, **kwargs):
            self.calls.append(("route", None))
            if self.failing == "route":
                raise ow.OwnerWorkspaceError("invalid_model_route", "the planning role has no usable route")
            return route(*args, **kwargs)

        def read_snapshot(**kwargs):
            self.calls.append(("snapshot", kwargs.get("project_id")))
            if self.failing == "snapshot":
                raise ow.OwnerWorkspaceError("snapshot_unavailable", "the Project board is unavailable")
            return snapshot(**kwargs)

        monkeypatch.setattr(ow, "_resolved_route_pin", read_route)
        monkeypatch.setattr(ow, "project_steward_snapshot", read_snapshot)


def _open_task_ids(root: Path, project_name: str) -> list:
    """The summary tasks of one Project that are neither done nor archived, counted on a separate connection."""
    return [
        row[0]
        for row in _rows(
            _board_db(root, BOARD),
            "SELECT id FROM tasks WHERE title = ? AND status NOT IN ('done', 'archived') ORDER BY created_at, id",
            (TITLE + project_name,),
        )
    ]


_WHOLE_STATE = (
    "event_cursor", "delivery_cursor", "release_cursor", "pending", "pending_urgent",
    "open_task_id", "last_requested_at", "failures", "updated_at",
)


def _whole_state(root: Path, project_id: str) -> dict | None:
    rows = _rows(
        root / "projects.db",
        f"SELECT {', '.join(_WHOLE_STATE)} FROM project_summary_state WHERE project_id = ?",
        (project_id,),
    )
    return dict(zip(_WHOLE_STATE, rows[0])) if rows else None


def _seed_state(root: Path, project_id: str, updated_at: int, *, failures: int = 0) -> None:
    """The state row an earlier tick left for a Project that waits for its first summary."""
    with projects_db.connect_closing(root / "projects.db") as conn, projects_db.write_txn(conn):
        conn.execute(
            "INSERT INTO project_summary_state (project_id, failures, updated_at) VALUES (?, ?, ?)",
            (project_id, failures, updated_at),
        )


def _summary_task_by_hand(
    project_id: str, project_name: str, key_tail: str, *, created_at: int | None = None
) -> str:
    """A summary task of the Project as an earlier run left it on the board, known only by its key."""
    kanban_db.init_db()
    if not kanban_db.board_exists(BOARD):
        kanban_db.create_board(slug=BOARD, name="Project summaries")
    with kanban_db.connect_closing(board=BOARD) as conn:
        task_id = kanban_db.create_task(
            conn, title=TITLE + project_name, assignee=PLANNER, created_by="project-summary",
            idempotency_key=f"project-summary:{project_id}:{key_tail}",
        )
        if created_at is not None:
            with kanban_db.write_txn(conn):
                conn.execute("UPDATE tasks SET created_at = ? WHERE id = ?", (created_at, task_id))
    return task_id


@pytest.mark.parametrize("failing", ["route", "snapshot"])
def test_task_made_before_a_restart_stays_the_only_open_task_when_a_read_fails_after_it(
    root, owner, monkeypatch, failing,
):
    ops = _ops()
    project_id = _project(owner, "Garden Plan", _CARDS)["project_id"]
    reads = _Reads(monkeypatch)
    _crash_after_the_task_is_made(ops, monkeypatch, T0)
    [made] = _summary_tasks(root)
    assert _state(root, project_id)["open_task_id"] is None
    reads.calls.clear()

    def only_the_made_task_is_open_and_recorded() -> None:
        assert _open_task_ids(root, "Garden Plan") == [made.id]
        state = _state(root, project_id)
        assert (state["open_task_id"], state["failures"]) == (made.id, 0)
        assert reads.calls == []

    reads.failing = failing
    ops.summary_tick(now=T0 + 5)
    only_the_made_task_is_open_and_recorded()

    reads.failing = None
    ops.summary_tick(now=T0 + 5 + HOUR)
    only_the_made_task_is_open_and_recorded()
    assert len(_summary_tasks(root, archived=True)) == 1


@pytest.mark.parametrize("has_row", [True, False], ids=["row_without_the_task", "no_row"])
def test_open_task_on_the_board_is_recorded_without_any_other_read(root, owner, monkeypatch, has_row):
    ops = _ops()
    project_id = _project(owner, "Garden Plan", _CARDS)["project_id"]
    reads = _Reads(monkeypatch)
    if has_row:
        _crash_after_the_task_is_made(ops, monkeypatch, T0)
        [made] = [task.id for task in _summary_tasks(root)]
    else:
        made = _summary_task_by_hand(project_id, "Garden Plan", f"0:{T0}")
    before = _state(root, project_id)
    assert before is None or before["open_task_id"] is None
    reads.calls.clear()

    assert ops.summary_tick(now=T0 + 5) == 0

    assert reads.calls == []
    assert _open_task_ids(root, "Garden Plan") == [made]
    recorded = {
        "event_cursor": 0, "delivery_cursor": 0, "release_cursor": 0, "pending": 0, "pending_urgent": 0,
        "open_task_id": made, "last_requested_at": T0 + 5, "failures": 0, "updated_at": T0 + 5,
    }
    assert _whole_state(root, project_id) == recorded

    assert ops.summary_tick(now=T0 + 10) == 0
    assert _whole_state(root, project_id) == recorded
    assert reads.calls == []


def test_oldest_open_task_of_the_project_is_the_one_recorded(root, owner, monkeypatch):
    ops = _ops()
    project_id = _project(owner, "Garden Plan", _CARDS)["project_id"]
    _crash_after_the_task_is_made(ops, monkeypatch, T0)
    [later] = _summary_tasks(root)
    older = _summary_task_by_hand(
        project_id, "Garden Plan", f"0:{T0 - 1}", created_at=later.created_at - 100,
    )
    assert _open_task_ids(root, "Garden Plan") == [older, later.id]

    ops.summary_tick(now=T0 + 5)

    assert _state(root, project_id)["open_task_id"] == older


def test_row_that_records_another_open_task_is_left_alone(root, owner, monkeypatch):
    ops = _ops()
    project_id = _project(owner, "Garden Plan", _CARDS)["project_id"]
    reads = _Reads(monkeypatch)
    _crash_after_the_task_is_made(ops, monkeypatch, T0)
    [recorded] = _summary_tasks(root)
    older = _summary_task_by_hand(
        project_id, "Garden Plan", f"0:{T0 - 1}", created_at=recorded.created_at - 100,
    )
    with projects_db.connect_closing(root / "projects.db") as conn, projects_db.write_txn(conn):
        conn.execute(
            "UPDATE project_summary_state SET open_task_id = ?, pending = 0, pending_urgent = 0, "
            "last_requested_at = ?, updated_at = ? WHERE project_id = ?",
            (recorded.id, T0, T0, project_id),
        )
    row_before = _whole_state(root, project_id)
    assert _open_task_ids(root, "Garden Plan") == [older, recorded.id]
    reads.calls.clear()

    ops.summary_tick(now=T0 + 5)

    assert _whole_state(root, project_id) == row_before
    assert reads.calls == []


@pytest.mark.parametrize("why_not_asked", ["stopped_after_three_failures", "removal_started"])
def test_open_task_on_the_board_is_recorded_though_no_new_one_could_be_asked_for(
    root, owner, monkeypatch, why_not_asked,
):
    ops = _ops()
    project_id = _project(owner, "Garden Plan", _CARDS)["project_id"]
    reads = _Reads(monkeypatch)
    if why_not_asked == "removal_started":
        _leave(root, project_id, "removal_started")
    else:
        _seed_state(root, project_id, updated_at=T0 - 100, failures=ops.FAILURE_STOP)
    made = _summary_task_by_hand(project_id, "Garden Plan", f"0:{T0}")
    before = _whole_state(root, project_id)

    assert ops.summary_tick(now=T0 + 5) == 0

    state = _whole_state(root, project_id)
    assert state["open_task_id"] == made
    assert (state["pending"], state["pending_urgent"], state["last_requested_at"]) == (0, 0, T0 + 5)
    assert state["failures"] == (0 if before is None else before["failures"])
    assert reads.calls == []

    ops.summary_tick(now=T0 + 10)

    assert reads.calls == []
    assert len(_summary_tasks(root, archived=True)) == 1
    if why_not_asked == "removal_started":
        assert _open_task_ids(root, "Garden Plan") == []
        assert _state(root, project_id)["open_task_id"] is None
    else:
        assert _open_task_ids(root, "Garden Plan") == [made]
        assert _state(root, project_id)["open_task_id"] == made


def test_finished_task_that_was_stored_but_not_archived_is_not_an_open_task(root, owner, monkeypatch):
    ops = _ops()
    _clean(OUTCOME, NEXT_ACTION, DECISION)
    project_id = _project(owner, "Garden Plan", _CARDS)["project_id"]
    reads = _Reads(monkeypatch)
    ops.summary_tick(now=T0)
    [task] = _summary_tasks(root)
    _finish(root, task.id, _valid())
    archive_task = kanban_db.archive_task

    def die_before_archiving(*_args, **_kwargs):
        raise _ProcessDied

    monkeypatch.setattr(kanban_db, "archive_task", die_before_archiving)
    with pytest.raises(_ProcessDied):
        ops.summary_tick(now=T0 + 10)
    monkeypatch.setattr(kanban_db, "archive_task", archive_task)
    assert _summary_of(root, project_id)["revision"] == 1
    assert _rows(_board_db(root, BOARD), "SELECT status FROM tasks WHERE id = ?", (task.id,)) == [("done",)]
    row_before = _whole_state(root, project_id)
    reads.calls.clear()

    assert ops.summary_tick(now=T0 + 20) == 0
    assert ops.summary_tick(now=T0 + 30) == 0

    assert _whole_state(root, project_id) == row_before
    assert _summary_of(root, project_id)["revision"] == 1
    assert reads.calls == []
    assert len(_summary_tasks(root, archived=True)) == 1


@pytest.mark.parametrize("failing", ["route", "snapshot"])
def test_failed_read_before_publishing_changes_only_failures_and_updated_at(root, owner, monkeypatch, failing):
    ops = _ops()
    project_id = _project(owner, "Garden Plan", _CARDS)["project_id"]
    _seed_state(root, project_id, updated_at=T0 - 100)
    row_before = _whole_state(root, project_id)
    reads = _Reads(monkeypatch)
    reads.failing = failing

    ops.summary_tick(now=T0)

    assert _whole_state(root, project_id) == {**row_before, "failures": 1, "updated_at": T0}
    assert _summary_tasks(root, archived=True) == []
    assert [name for name, _ in reads.calls] == (["route"] if failing == "route" else ["route", "snapshot"])

    reads.failing = None
    reads.calls.clear()
    ops.summary_tick(now=T0 + HOUR - 1)
    assert _summary_tasks(root, archived=True) == []
    assert reads.calls == []

    ops.summary_tick(now=T0 + HOUR)
    [task] = _summary_tasks(root)
    assert _state(root, project_id)["open_task_id"] == task.id


def test_reads_that_keep_failing_stop_after_three_attempts_with_no_request_ever_made(root, owner, monkeypatch):
    ops = _ops()
    project_id = _project(owner, "Garden Plan", _CARDS)["project_id"]
    reads = _Reads(monkeypatch)
    reads.failing = "snapshot"
    at = T0

    for failures in (1, 2, 3):
        ops.summary_tick(now=at)
        state = _state(root, project_id)
        assert (state["open_task_id"], state["failures"], state["last_requested_at"]) == (None, failures, None)
        assert len(reads.calls) == 2 * failures
        wait = ops.backoff_seconds(failures)
        ops.summary_tick(now=at + wait - 1)
        assert len(reads.calls) == 2 * failures
        at += wait

    ops.summary_tick(now=at + 30 * 86_400)
    assert len(reads.calls) == 6
    assert _summary_tasks(root, archived=True) == []


def test_project_ids_are_matched_literally_not_as_patterns(root):
    ops = _ops()
    with_underscore, lookalike = "p_aaaa1111", "pXaaaa1111"
    with projects_db.connect_closing(root / "projects.db") as conn:
        projects_db.create_project(conn, name="Garden Plan", id=with_underscore)
        projects_db.create_project(conn, name="Planning Desk", id=lookalike)
    lookalikes_task = _summary_task_by_hand(lookalike, "Planning Desk", f"0:{T0}")

    ops.summary_tick(now=T0 + 5)

    assert _state(root, lookalike)["open_task_id"] == lookalikes_task
    garden = _state(root, with_underscore)
    assert (garden["open_task_id"], garden["failures"]) == (None, 1)
    assert _open_task_ids(root, "Planning Desk") == [lookalikes_task]
    assert _open_task_ids(root, "Garden Plan") == []


class _LockedBoard:
    """A board connection that cannot be used: whatever is asked of it, the database is locked."""

    def __getattr__(self, name):
        raise sqlite3.OperationalError("database is locked")


def test_check_that_cannot_be_made_publishes_nothing_and_changes_nothing(root, owner, monkeypatch):
    ops = _ops()
    project_id = _project(owner, "Garden Plan", _CARDS)["project_id"]
    kanban_db.init_db()
    kanban_db.create_board(slug=BOARD, name="Project summaries")
    _seed_state(root, project_id, updated_at=T0 - 100)
    row_before = _whole_state(root, project_id)
    reads = _Reads(monkeypatch)
    connect_closing = kanban_db.connect_closing

    @contextlib.contextmanager
    def locked_summary_board(*args, **kwargs):
        if kwargs.get("board") != BOARD:
            with connect_closing(*args, **kwargs) as conn:
                yield conn
        else:
            yield _LockedBoard()

    monkeypatch.setattr(kanban_db, "connect_closing", locked_summary_board)
    with pytest.raises(sqlite3.OperationalError):
        ops.summary_tick(now=T0)
    monkeypatch.setattr(kanban_db, "connect_closing", connect_closing)

    assert reads.calls == []
    assert _summary_tasks(root, archived=True) == []
    assert _whole_state(root, project_id) == row_before

    assert ops.summary_tick(now=T0 + 10) == 1
    [task] = _summary_tasks(root)
    assert _state(root, project_id)["open_task_id"] == task.id


def test_open_task_is_recorded_under_the_worker_environment(root, owner, monkeypatch):
    ops = _ops()
    projects = {
        "Garden Plan": _project(owner, "Garden Plan", _CARDS),
        "Planning Desk": _project(owner, "Planning Desk", _DESK_CARDS),
    }
    desk = projects["Planning Desk"]
    reads = _Reads(monkeypatch)
    _crash_after_the_task_is_made(ops, monkeypatch, T0)
    [made] = _summary_tasks(root)
    crashed = made.title[len(TITLE):]
    other = next(name for name in projects if name != crashed)
    with _board(root, desk["board"]) as conn:
        worker_task = kanban_db.create_task(
            conn, title="Plan the next milestone", assignee=PLANNER, project_id=desk["project_id"],
        )
    profile_home = root / "profiles" / PLANNER
    profile_home.mkdir(parents=True)
    with projects_db.connect_closing(profile_home / "projects.db"):
        pass
    _enter_planning_worker(monkeypatch, root, desk["board"], worker_task)
    pinned = {name: os.environ[name] for name in ("HERMES_KANBAN_DB", "HERMES_KANBAN_BOARD")}
    own_board = _board_db(root, desk["board"])
    assert Path(pinned["HERMES_KANBAN_DB"]).resolve() == own_board.resolve()
    own_cards = _rows(own_board, "SELECT id, title, status FROM tasks ORDER BY id")
    reads.calls.clear()

    ops.summary_tick(now=T0 + 5)

    open_ids = {name: _open_task_ids(root, name) for name in projects}
    assert {name: len(ids) for name, ids in open_ids.items()} == {"Garden Plan": 1, "Planning Desk": 1}
    assert open_ids[crashed] == [made.id]
    recorded = _rows(
        root / "projects.db",
        "SELECT project_id, open_task_id FROM project_summary_state WHERE open_task_id IS NOT NULL",
    )
    assert dict(recorded) == {projects[name]["project_id"]: ids[0] for name, ids in open_ids.items()}
    assert [project for step, project in reads.calls if step == "snapshot"] == [projects[other]["project_id"]]
    assert {name: os.environ[name] for name in pinned} == pinned
    assert _rows(own_board, "SELECT id, title, status FROM tasks ORDER BY id") == own_cards
    assert _rows(profile_home / "projects.db", "SELECT count(*) FROM project_summary_state") == [(0,)]
