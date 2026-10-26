"""An owner plan ignores the heartbeats of a task it names.

A plan names each task it changes or depends on by the status and revision
the owner planning snapshot shows, and the kernel checks both again before it
changes anything. A running worker records a heartbeat every few seconds, so a
plan that named a running task, for example as the existing parent of a new
task, went out of date before the owner could approve it. Heartbeats no longer
count; any other event on a named task, a comment or a status change, still
makes the plan out of date.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import owner_workspace
from hermes_cli import projects_db


@pytest.fixture
def project(tmp_path, monkeypatch):
    """An open connection to an isolated board holding one Project's anchor."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    with projects_db.connect_closing() as pconn:
        project_id = projects_db.create_project(pconn, name="Website")
    conn = kb.connect()
    try:
        anchor_id = kb.create_task(
            conn, title="Website", control=True, project_id=project_id,
        )
        yield SimpleNamespace(conn=conn, id=project_id, anchor_id=anchor_id)
    finally:
        conn.close()


def _task(project, title: str) -> str:
    """A content task of the Project, ready to be claimed."""
    return kb.create_task(
        project.conn, title=title, assignee="default", project_id=project.id,
    )


def _running_task(project, title: str) -> str:
    """A content task of the Project that a worker has claimed."""
    task_id = _task(project, title)
    assert kb.claim_task(project.conn, task_id) is not None
    return task_id


def _ref(project, task_id: str) -> dict:
    """The snapshot a plan names *task_id* by, read from the owner planning
    snapshot that the owner Workspace builds its task references from."""
    # Only the task list matters here, not the column totals.
    context = owner_workspace.owner_project_planning_context(
        project.conn, project.id, Counter(),
    )
    [task] = [item for item in context["tasks"] if item["id"] == task_id]
    return {
        "task_id": task_id,
        "expected_status": task["status"],
        "expected_revision": task["event_revision"],
    }


def _spec(title: str) -> dict:
    return {
        "title": title,
        "body": "Produce the owner-visible website content.",
        "assignee": "default",
        "responsibility": None,
    }


def _recheck(parent: dict) -> dict:
    """An add whose new task waits for the task *parent* names."""
    return {
        "action": "add",
        "reason": "Re-check the documents once the content is final.",
        **_spec("Re-check the documents"),
        "existing_parents": [parent],
        "new_parents": [],
    }


def _replace(target: dict) -> dict:
    return {
        "action": "replace",
        "reason": "Rescope the content.",
        "target": target,
        "replacement": _spec("Rewrite the content"),
    }


def _apply(project, changes: list[dict]) -> dict:
    """Apply *changes* as one owner plan; the kernel's result."""
    return kb.apply_owner_project_plan(
        project.conn,
        project_id=project.id,
        anchor_task_id=project.anchor_id,
        changes=changes,
        actor="owner",
        profile="owner",
        idempotency_key="website-plan",
        request_digest="d" * 64,
        trigger="owner_request",
        plan_summary="Re-check the website content.",
        current_milestone="Launch the website",
        later_milestones=[],
    )


def _assert_out_of_date(project, result: dict, task_id: str) -> None:
    """The plan was refused on *task_id* alone, and the conflict reports what
    the owner planning snapshot now shows for that task."""
    current = _ref(project, task_id)
    assert result == {
        "applied": False,
        "conflicts": [{
            "task_id": task_id,
            "status": current["expected_status"],
            "revision": current["expected_revision"],
        }],
    }


def test_a_plan_naming_a_running_parent_still_applies_after_a_heartbeat(project):
    content = _running_task(project, "Write the content")
    ref = _ref(project, content)

    assert kb.heartbeat_worker(project.conn, content) is True
    # What the owner Workspace reads again when the owner approves the plan.
    approved = _ref(project, content)
    result = _apply(project, [_recheck(ref)])

    assert result["applied"] is True, result
    [recheck] = result["created_task_ids"]
    assert kb.parent_ids(project.conn, recheck) == [content]
    assert approved == ref


def test_the_same_plan_is_out_of_date_after_a_comment_on_the_running_parent(project):
    content = _running_task(project, "Write the content")
    ref = _ref(project, content)

    kb.add_comment(project.conn, content, "owner", "Use the new brand colours.")
    assert kb.heartbeat_worker(project.conn, content) is True
    result = _apply(project, [_recheck(ref)])

    _assert_out_of_date(project, result, content)
    assert kb.child_ids(project.conn, content) == []


def test_the_same_plan_is_out_of_date_after_a_status_change_of_the_running_parent(
    project,
):
    content = _running_task(project, "Write the content")
    ref = _ref(project, content)

    assert kb.heartbeat_worker(project.conn, content) is True
    assert kb.block_task(project.conn, content, reason="Waiting for the brand guide.")
    result = _apply(project, [_recheck(ref)])

    _assert_out_of_date(project, result, content)
    assert kb.get_task(project.conn, content).status != ref["expected_status"]
    assert kb.child_ids(project.conn, content) == []


def test_a_plan_targeting_a_task_is_out_of_date_after_a_real_event_on_it(project):
    content = _task(project, "Write the content")
    ref = _ref(project, content)

    kb.add_comment(project.conn, content, "owner", "Keep the current tone.")
    result = _apply(project, [_replace(ref)])

    _assert_out_of_date(project, result, content)
    assert kb.get_task(project.conn, content).status == ref["expected_status"]
