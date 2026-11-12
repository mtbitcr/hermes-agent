"""Dispatch a kernel-produced portfolio with the existing owner and tool limits."""

from pathlib import Path
from types import SimpleNamespace
import subprocess
import yaml

import pytest

from hermes_cli import kanban_db, owner_project_review as review, owner_workspace, projects_db
from tests.gateway.test_kanban_watchers_review_tick import _Loop, _drive, _switch_on
# Renamed: the watcher tests' root makes every assignee a profile, and the first test keeps the real profile check.
from tests.gateway.test_kanban_watchers_review_tick import root as watcher_root  # noqa: F401
from tests.hermes_cli.test_kanban_review_task_tool_allowlist import _profile
from tests.hermes_cli.test_owner_project_review_plans import _logical_state
from tests.hermes_cli.test_owner_project_review_result import NOW, _quiet_project, root  # noqa: F401

SUMMARIES = "project-summaries"
KEY = "project-review:portfolio:2026-11-09:0"
RESULT = {"conflicts": [], "priorities": ["Meeting Plan: Book the room."], "shared_blockers": []}


@pytest.mark.parametrize("paused", [False, True], ids=["active", "owner-paused"])
def test_a_generated_portfolio_keeps_native_dispatch_admission_and_tool_limits(root, tmp_path, monkeypatch, paused):
    project = _quiet_project(root)
    (root / "config.yaml").write_text(yaml.safe_dump({"owner_review": {
        "enabled": True, "allowlist": [project.id], "portfolio": True,
    }}))
    _profile(root, "raphael-planner")
    project_path = kanban_db.kanban_db_path(board=project.board_slug)
    before = _logical_state(project_path)
    assert review.review_tick(NOW) == 1
    board_name = "project-summaries"
    if paused:
        kanban_db.write_board_dispatch_state(board_name, dispatch_enabled=False, dispatch_paused_by_owner=True)
    captured = []

    def spawn(task, workspace):
        def popen(command, *args, **kwargs):
            kwargs["stdout"].close()
            captured.append((task.id, list(command), dict(kwargs["env"])))
            return SimpleNamespace(pid=4242)

        Path(workspace).mkdir(parents=True, exist_ok=True)
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(kanban_db, "_resolve_hermes_argv", lambda: ["hermes"])
            patch.setattr(subprocess, "Popen", popen)
            return kanban_db._default_spawn(task, workspace, board=board_name)

    with kanban_db.connect_closing(kanban_db.kanban_db_path(board=board_name), board=board_name) as board:
        task_id = board.execute("SELECT id FROM tasks WHERE idempotency_key=?", (
            "project-review:portfolio:2026-11-09:0",
        )).fetchone()[0]
        task_before = kanban_db.get_task(board, task_id)
        result = kanban_db.dispatch_once(board, board=board_name, spawn_fn=spawn,
                                        require_board_activation=True, max_in_progress=1)
        task = kanban_db.get_task(board, task_id)
        if paused:
            assert result.spawned == [] and captured == []
            assert task.status == task_before.status
            assert kanban_db.read_board_metadata(board_name)["dispatch_paused_by_owner"] is True
        else:
            assert [row[0] for row in result.spawned] == [task_id]
            assert task.status == "running" and task.current_run_id is not None
            assert len(captured) == 1 and captured[0][0] == task_id
            _task_id, command, environment = captured[0]
            assert command.count("--toolsets") == 1
            assert command[command.index("--toolsets") + 1] == "kanban"
            assert environment["HERMES_KANBAN_ALLOWED_TOOLS"] == "kanban_show,kanban_complete"
            assert environment["HERMES_KANBAN_TASK"] == task_id
            assert environment["HERMES_KANBAN_BOARD"] == board_name
            assert Path(environment["HERMES_KANBAN_DB"]).resolve() == kanban_db.kanban_db_path(board=board_name).resolve()
            ordinary = kanban_db.create_task(
                board, title="Plan the ordinary work", assignee="raphael-planner", created_by="owner",
                workspace_kind="scratch", owned_paths=[], board=board_name, risk_tier=2,
                **owner_workspace._resolved_route_pin("raphael-planner", "routine", "execution_tier"),
            )
            # The captured launcher has no OS child. Keep its live claim while testing the shared cap.
            held = kanban_db.dispatch_once(board, board=board_name, spawn_fn=spawn,
                                          require_board_activation=True, max_in_progress=1, reconcile_orphans=False)
            assert held.spawned == [] and len(captured) == 1
            assert kanban_db.get_task(board, ordinary).status != "running"
            assert kanban_db.complete_task(board, task_id, summary="Reviewed the Projects together.", metadata={
                "portfolio_review": {"conflicts": [], "priorities": [], "shared_blockers": []},
            })
            released = kanban_db.dispatch_once(board, board=board_name, spawn_fn=spawn,
                                              require_board_activation=True, max_in_progress=1, reconcile_orphans=False)
            assert [row[0] for row in released.spawned] == [ordinary]
            assert kanban_db.get_task(board, ordinary).status == "running"
            assert len(captured) == 2 and captured[1][0] == ordinary
    with projects_db.connect_closing(root / "projects.db") as registry:
        assert review.budget(registry, "2026-11-09") == (0, 1)
    assert _logical_state(project_path) == before


@pytest.fixture
def watcher_loop(watcher_root, monkeypatch):
    """The gateway dispatcher loop the watcher tests record, with a spawn stand-in that starts no worker process."""
    loop = _Loop()
    loop.record_spawns(monkeypatch)
    return loop


def _summaries():
    return kanban_db.connect_closing(kanban_db.kanban_db_path(board=SUMMARIES), board=SUMMARIES)


def _portfolios():
    """``(id, status, idempotency_key)`` of each portfolio task on the summary board, archived ones included."""
    if not kanban_db.kanban_db_path(board=SUMMARIES).exists():
        return []
    with _summaries() as board:
        return [tuple(row) for row in board.execute(
            "SELECT id,status,idempotency_key FROM tasks WHERE created_by='project-review' "
            "AND idempotency_key LIKE 'project-review:portfolio:%' ORDER BY created_at,id",
        )]


def _launched(loop):
    """``(tick, task id)`` of each portfolio worker the dispatcher launched."""
    portfolios = {task_id for task_id, _status, _key in _portfolios()}
    return [(tick, task_id) for tick, _board, task_id, *_rest in loop.spawned if task_id in portfolios]


def _worker_completes(result):
    """The pause between ticks, in which each running portfolio worker completes its task with ``result``."""

    def between(_tick):
        for task_id, status, _key in _portfolios():
            if status == "running":
                with _summaries() as board:
                    assert kanban_db.complete_task(board, task_id, summary="Reviewed the Projects together.",
                                                   metadata={"portfolio_review": result})

    return between


def _stored(home, day="2026-11-09"):
    """The day's ``(project_reviews, portfolio_reviews)`` budget and its stored portfolio brief."""
    with projects_db.connect_closing(home / "projects.db") as registry:
        return review.budget(registry, day), projects_db.get_portfolio_review_brief(registry, day)


def test_an_accepted_portfolio_day_starts_no_second_task_with_a_spare_slot(watcher_root, watcher_loop, monkeypatch):
    project = _quiet_project(watcher_root)
    _switch_on(watcher_root, project.id, portfolio=True, portfolio_cap=2)

    _drive(monkeypatch, watcher_loop, [NOW + n * 60 for n in range(5)], between=_worker_completes(RESULT))

    # Tick 0 publishes the day's portfolio and tick 1 launches its worker, which completes it. Tick 2 stores and
    # archives the accepted result; the spare slot of the cap starts nothing more that day.
    assert watcher_loop.made == [1, 0, 0, 0, 0]
    [(task_id, status, key)] = _portfolios()
    assert (status, key) == ("archived", KEY)
    assert _launched(watcher_loop) == [(1, task_id)]
    budget, brief = _stored(watcher_root)
    assert budget == (0, 1)
    assert (brief["revision"], brief["priorities"]) == (1, RESULT["priorities"])


def test_a_cap_raised_after_an_accepted_portfolio_starts_no_second_task_that_day(
    watcher_root, watcher_loop, monkeypatch
):
    project = _quiet_project(watcher_root)
    _switch_on(watcher_root, project.id, portfolio=True, portfolio_cap=1)
    finish = _worker_completes(RESULT)
    _drive(monkeypatch, watcher_loop, [NOW, NOW + 60, NOW + 120], between=finish)
    [(task_id, status, _key)] = _portfolios()
    assert (watcher_loop.made, status) == ([1, 0, 0], "archived")

    # The owner raises the cap on the same review day.
    _switch_on(watcher_root, project.id, portfolio=True, portfolio_cap=2)
    assert review.read_settings().portfolio_cap == 2
    _drive(monkeypatch, watcher_loop, [NOW + 180, NOW + 240, NOW + 300], between=finish)

    assert watcher_loop.made == [1, 0, 0, 0, 0, 0]
    assert _portfolios() == [(task_id, "archived", KEY)]
    assert _launched(watcher_loop) == [(1, task_id)]
    budget, brief = _stored(watcher_root)
    assert (budget, brief["revision"]) == ((0, 1), 1)


def test_the_next_review_day_still_gets_its_own_portfolio_with_a_spare_slot(watcher_root, watcher_loop, monkeypatch):
    project = _quiet_project(watcher_root)
    _switch_on(watcher_root, project.id, portfolio=True, portfolio_cap=2)
    finish = _worker_completes(RESULT)
    _drive(monkeypatch, watcher_loop, [NOW, NOW + 60, NOW + 120], between=finish)
    assert _stored(watcher_root)[1]["revision"] == 1

    _drive(monkeypatch, watcher_loop, [NOW + 86400, NOW + 86460], between=finish)

    # The next day's first tick publishes its own portfolio, and the tick after launches its worker.
    assert watcher_loop.made[3:] == [1, 0]
    [(task_id, status)] = [(task_id, status) for task_id, status, key in _portfolios()
                           if key == "project-review:portfolio:2026-11-10:0"]
    assert status == "running"
    assert [tick for tick, launched in _launched(watcher_loop) if launched == task_id] == [4]
    assert _stored(watcher_root, "2026-11-10") == ((0, 1), None)


def test_a_refused_portfolio_still_gets_its_retry_from_a_spare_slot(watcher_root, watcher_loop, monkeypatch):
    project = _quiet_project(watcher_root)
    _switch_on(watcher_root, project.id, portfolio=True, portfolio_cap=2)
    refused = {**RESULT, "changes": [{"action": "cancel"}]}

    _drive(monkeypatch, watcher_loop, [NOW + n * 60 for n in range(5)], between=_worker_completes(refused))

    # Tick 2 refuses and archives the result, so nothing is stored. Tick 3 publishes the day's retry from the spare
    # slot, and tick 4 launches its worker.
    assert watcher_loop.made == [1, 0, 0, 1, 0]
    [(first, first_status, first_key), (retry, retry_status, retry_key)] = _portfolios()
    assert (first_status, retry_status) == ("archived", "running")
    assert first_key == retry_key == KEY
    assert _launched(watcher_loop) == [(1, first), (4, retry)]
    assert _stored(watcher_root) == ((0, 2), None)
