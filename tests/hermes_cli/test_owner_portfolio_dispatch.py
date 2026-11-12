"""Dispatch a kernel-produced portfolio with the existing owner and tool limits."""

from pathlib import Path
from types import SimpleNamespace
import subprocess
import yaml

import pytest

from hermes_cli import kanban_db, owner_project_review as review, owner_workspace, projects_db
from tests.hermes_cli.test_kanban_review_task_tool_allowlist import _profile
from tests.hermes_cli.test_owner_project_review_plans import _logical_state
from tests.hermes_cli.test_owner_project_review_result import NOW, _quiet_project, root  # noqa: F401


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
