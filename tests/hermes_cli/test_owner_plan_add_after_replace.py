"""A later add of an owner plan waits for what replaced the task it names.

The kernel applies a plan's changes in order. When a replace, split or merge
of one plan archives a task and a later add of that same plan names the task
among its existing parents, the add waits for what replaced it. Linked to the
archived original instead, it would start at once: an archived parent with no
delivery row counts as satisfied. An add that comes before the replace stays
as it was, because the replace moves the original's children.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from hermes_cli import kanban_db as kb
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


def _task(project, title: str) -> tuple[str, dict]:
    """A content task of the Project, and the snapshot a plan names it by."""
    task_id = kb.create_task(
        project.conn, title=title, assignee="default", project_id=project.id,
    )
    return task_id, {
        "task_id": task_id,
        "expected_status": kb.get_task(project.conn, task_id).status,
        "expected_revision": kb.task_event_revision(project.conn, task_id),
    }


def _spec(title: str) -> dict:
    return {
        "title": title,
        "body": "Produce the owner-visible website content.",
        "assignee": "default",
        "responsibility": None,
    }


def _recheck(*existing_parents: dict) -> dict:
    return {
        "action": "add",
        "reason": "Re-check the documents once the content is final.",
        **_spec("Re-check the documents"),
        "existing_parents": list(existing_parents),
        "new_parents": [],
    }


def _replace(target: dict) -> dict:
    return {
        "action": "replace",
        "reason": "Rescope the content.",
        "target": target,
        "replacement": _spec("Rewrite the content"),
    }


def _apply(project, changes: list[dict]) -> list[str]:
    """Apply *changes* as one owner plan; the created task ids, in apply order."""
    result = kb.apply_owner_project_plan(
        project.conn,
        project_id=project.id,
        anchor_task_id=project.anchor_id,
        changes=changes,
        actor="owner",
        profile="owner",
        idempotency_key="website-plan",
        request_digest="d" * 64,
        trigger="owner_request",
        plan_summary="Rework the website content.",
        current_milestone="Launch the website",
        later_milestones=[],
    )
    assert result["applied"] is True
    return result["created_task_ids"]


def test_an_add_after_a_replace_waits_for_the_replacement(project):
    content, ref = _task(project, "Write the content")

    replacement, recheck = _apply(project, [_replace(ref), _recheck(ref)])

    assert kb.get_task(project.conn, content).status == "archived"
    assert kb.parent_ids(project.conn, recheck) == [replacement]
    assert not kb._parents_satisfied(project.conn, recheck)


def test_an_add_after_a_split_waits_for_each_replacement_no_later_one_names(project):
    """The review names the draft as its parent, so the add waits for the
    review and the translation, and through the review for the draft."""
    content, ref = _task(project, "Write the content")
    split = {
        "action": "split",
        "reason": "The content is too broad to verify safely.",
        "target": ref,
        "replacements": [
            {**_spec("Draft the documents"), "parents": []},
            {**_spec("Review the drafts"), "parents": [0]},
            {**_spec("Translate the documents"), "parents": []},
        ],
    }

    _draft, review, translation, recheck = _apply(project, [split, _recheck(ref)])

    assert kb.get_task(project.conn, content).status == "archived"
    assert sorted(kb.parent_ids(project.conn, recheck)) == sorted([review, translation])
    assert not kb._parents_satisfied(project.conn, recheck)


def test_an_add_after_a_merge_waits_for_the_merged_replacement(project):
    home, home_ref = _task(project, "Write the home page")
    about, about_ref = _task(project, "Write the about page")
    merge = {
        "action": "merge",
        "reason": "One coherent set of pages is easier to verify.",
        "targets": [home_ref, about_ref],
        "replacement": _spec("Write the pages"),
    }

    merged, recheck = _apply(project, [merge, _recheck(home_ref, about_ref)])

    assert kb.get_task(project.conn, home).status == "archived"
    assert kb.get_task(project.conn, about).status == "archived"
    assert kb.parent_ids(project.conn, recheck) == [merged]
    assert not kb._parents_satisfied(project.conn, recheck)


def test_an_add_before_a_replace_keeps_its_parent_and_waits_for_the_replacement(project):
    """The replace moves the original's children, the earlier add among them."""
    content, ref = _task(project, "Write the content")

    recheck, replacement = _apply(project, [_recheck(ref), _replace(ref)])

    assert kb.get_task(project.conn, content).status == "archived"
    assert sorted(kb.parent_ids(project.conn, recheck)) == sorted([content, replacement])
    assert not kb._parents_satisfied(project.conn, recheck)
