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
from hermes_cli import kanban_db, owner_workspace as ow, projects_db
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
def test_owner_decision_is_optional(root, given, stored):
    ops = _ops()
    project_id = _registry_project(root, "Garden Plan")
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
def test_invalid_result_stores_nothing(root, metadata, reassigned_to):
    ops = _ops()
    for text in (REFUSED, REFUSED_NEXT_ACTION, REFUSED_DECISION):
        assert check_owner_text(text, "planner_prose") != text
    project_id = _registry_project(root, "Garden Plan")
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
def test_failed_step_stores_nothing_backs_off_and_stops(root, monkeypatch, how):
    ops = _ops()
    project_id = _registry_project(root, "Garden Plan")
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

    ops.summary_tick(now=T0 + 10)
    check(failures=1, tasks=1)
    ops.summary_tick(now=T0 + HOUR - 1)
    check(failures=1, tasks=1)

    ops.summary_tick(now=T0 + HOUR)
    assert created() == 2
    [second] = _summary_tasks(root)
    assert _state(root, project_id)["open_task_id"] == second.id
    _fail(root, monkeypatch, second.id, how)

    second_at = T0 + HOUR
    ops.summary_tick(now=second_at + 10)
    check(failures=2, tasks=2)
    ops.summary_tick(now=second_at + 2 * HOUR - 1)
    check(failures=2, tasks=2)

    third_at = second_at + 2 * HOUR
    ops.summary_tick(now=third_at)
    assert created() == 3
    [third] = _summary_tasks(root)
    _fail(root, monkeypatch, third.id, how)

    ops.summary_tick(now=third_at + 10)
    check(failures=3, tasks=3)
    ops.summary_tick(now=third_at + 30 * 86_400)
    check(failures=3, tasks=3)


def test_timed_out_run_that_is_requeued_is_not_yet_a_failure(root, monkeypatch):
    ops = _ops()
    project_id = _registry_project(root, "Garden Plan")
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


def test_at_most_three_tasks_per_tick_and_none_for_projects_that_are_gone(root):
    ops = _ops()
    live = ["Alpha", "Bravo", "Charlie", "Delta", "Echo"]
    for name in live[:4]:
        _registry_project(root, name)
    cancelled = _registry_project(root, "Echo")
    archived = _registry_project(root, "Foxtrot")
    removing = _registry_project(root, "Golf")
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


def test_tick_that_loses_the_race_for_a_project_leaves_the_other_task_open(root, monkeypatch):
    ops = _ops()
    project_id = _registry_project(root, "Garden Plan")
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


def test_planning_role_without_a_route_is_a_failure_that_backs_off(root, monkeypatch):
    ops = _ops()
    project_id = _registry_project(root, "Garden Plan")
    working_route = model_policy.configured_assignment_for

    def no_route(_profile):
        raise ValueError("no route is configured")

    monkeypatch.setattr(model_policy, "configured_assignment_for", no_route)
    ops.summary_tick(now=T0)

    state = _state(root, project_id)
    assert (state["open_task_id"], state["failures"], state["pending"]) == (None, 1, 1)
    assert state["last_requested_at"] == T0
    assert _summary_tasks(root, archived=True) == []

    monkeypatch.setattr(model_policy, "configured_assignment_for", working_route)
    ops.summary_tick(now=T0 + HOUR - 1)
    assert _summary_tasks(root, archived=True) == []
    ops.summary_tick(now=T0 + HOUR)

    [task] = _summary_tasks(root)
    assert _state(root, project_id)["open_task_id"] == task.id


def test_summary_board_that_is_gone_counts_as_a_failure(root):
    ops = _ops()
    project_id = _registry_project(root, "Garden Plan")
    ops.summary_tick(now=T0)
    shutil.rmtree(_board_db(root, BOARD).parent)

    ops.summary_tick(now=T0 + 10)

    state = _state(root, project_id)
    assert (state["open_task_id"], state["failures"], state["pending"]) == (None, 1, 1)
    assert _summary_of(root, project_id) is None


def _drive_dispatcher(monkeypatch, *, ticks: int) -> dict:
    """Run the gateway's embedded dispatcher loop for ``ticks`` iterations and count what each one runs."""
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
        return fn(*args, **kwargs)

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
    assert sorted(by_title) == [TITLE + "Garden Plan", TITLE + "Planning Desk"]
    garden_task = by_title[TITLE + "Garden Plan"]
    desk_task = by_title[TITLE + "Planning Desk"]
    assert "Publish the weekly summary" in garden_task.body
    assert "Order the desk lamp" not in garden_task.body
    assert "Order the desk lamp" in desk_task.body
    assert "Publish the weekly summary" not in desk_task.body
    meta = json.loads((_board_db(root, BOARD).parent / "board.json").read_text(encoding="utf-8"))
    assert (meta["project_id"], meta["dispatch_enabled"]) == (None, True)
    assert _rows(own_board, "SELECT id, title, status FROM tasks ORDER BY id") == own_cards
    assert {name: os.environ[name] for name in pinned} == pinned

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
    assert _rows(
        root / "projects.db",
        "SELECT count(*) FROM project_summary_state WHERE open_task_id IS NOT NULL",
    ) == [(0,)]
    assert _rows(profile_home / "projects.db", "SELECT count(*) FROM project_summaries") == [(0,)]
    assert _rows(profile_home / "projects.db", "SELECT count(*) FROM project_summary_state") == [(0,)]
    assert {name: os.environ[name] for name in pinned} == pinned
