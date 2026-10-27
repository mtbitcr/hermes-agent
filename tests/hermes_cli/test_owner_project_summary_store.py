"""Card 4 of the Project operating home plan: the Raphael-written Project summary store.

The summary is three texts Raphael writes about a Project: the outcome now, the next action and, only when one
really waits, the owner's decision. It sits in its own table, so the owner's description is never touched, and
it reaches the owner app only for a reader that names ``project_summary``: the oldest owner reader validates a
closed snapshot schema, so a new key may not appear unasked.
"""

from __future__ import annotations

import contextlib
import sqlite3
import time

import pytest

from hermes_cli import owner_workspace as ow, projects_db
from hermes_cli.owner_text import check_owner_text

OUTCOME = "The room is booked for Tuesday at noon and the invitations are being written."
NEXT_ACTION = "Raphael sends the agenda to the team on Monday."
DECISION = "Please choose between the large room and the small room."
DESCRIPTION = "Plan the autumn harvest dinner for twelve guests."

# The registry as it was before the summary table: only these two tables, exactly as SCHEMA_SQL declared them.
_OLDER_REGISTRY_DDL = """
CREATE TABLE IF NOT EXISTS projects (
    id            TEXT PRIMARY KEY,
    slug          TEXT NOT NULL UNIQUE,
    name          TEXT NOT NULL,
    description   TEXT,
    icon          TEXT,
    color         TEXT,
    board_slug    TEXT,
    primary_path  TEXT,
    created_at    INTEGER NOT NULL,
    archived      INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS project_folders (
    project_id  TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    path        TEXT NOT NULL,
    label       TEXT,
    is_primary  INTEGER NOT NULL DEFAULT 0,
    added_at    INTEGER NOT NULL,
    PRIMARY KEY (project_id, path)
);
"""
# id, slug, name, description, created_at, archived
_OLDER_PROJECTS = (
    ("p_old00001", "alpha", "Alpha", "Alpha is the first of the older Projects.", 1_700_000_001, 0),
    ("p_old00002", "beta", "Beta", None, 1_700_000_002, 0),
    ("p_old00003", "gamma", "Gamma", "Gamma was archived last spring.", 1_700_000_003, 1),
)


@pytest.fixture
def owner() -> ow.OwnerContext:
    return ow.resolve_owner_context()


def _project(owner: ow.OwnerContext, name: str, description: str | None = None) -> dict:
    """Commit one owner Project, so the owner snapshot can read it."""
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(ow, "_confirm", lambda *_args, **_kwargs: {"approved": True})
        project = ow.bootstrap(
            owner, idempotency_key=f"setup-{name}", name=name, description=description,
        )
    assert project["ok"] is True
    project["slug"] = next(
        item["slug"] for item in ow.list_committed_projects(owner)
        if str(item["project_id"]) == project["project_id"]
    )
    return project


def _write(project_id: str, **texts) -> int:
    with projects_db.connect_closing() as conn:
        return projects_db.write_project_summary(conn, project_id, **texts)


def _stored(project_id: str):
    with projects_db.connect_closing() as conn:
        return projects_db.get_project_summary(conn, project_id)


def _shown(owner: ow.OwnerContext, project: dict):
    return ow.read_project_snapshot(owner, project["slug"], project_summary=True)["raphael_summary"]


def test_summary_appears_only_under_the_capability(owner):
    for text in (OUTCOME, NEXT_ACTION, DECISION):
        assert check_owner_text(text, "planner_prose") == text
    project = _project(owner, "Summary Project", DESCRIPTION)

    assert _shown(owner, project) is None

    _write(project["project_id"], outcome=OUTCOME, next_action=NEXT_ACTION, owner_decision=DECISION)

    asked = ow.read_project_snapshot(owner, project["slug"], project_summary=True)
    plain = ow.read_project_snapshot(owner, project["slug"])
    written_at = asked["raphael_summary"]["written_at"]
    assert isinstance(written_at, int)
    assert asked["raphael_summary"] == {
        "outcome": OUTCOME,
        "next_action": NEXT_ACTION,
        "owner_decision": DECISION,
        "written_at": written_at,
        "revision": 1,
    }
    assert "raphael_summary" not in plain
    assert "raphael_summary" not in plain["project"]
    assert set(asked) - set(plain) == {"raphael_summary"}
    assert set(plain) <= set(asked)
    assert asked["project"] == plain["project"]
    assert plain["project"]["description"] == DESCRIPTION


def test_text_the_policy_refuses_comes_back_as_the_fixed_sentence(owner):
    fixed = check_owner_text(None, "planner_prose")
    refused_outcome = "I sent the plan to owner@example.test this morning."
    refused_decision = "Claude Opus drafted the plan for you."
    assert check_owner_text(refused_outcome, "planner_prose") == fixed
    assert check_owner_text(refused_decision, "planner_prose") == fixed
    assert check_owner_text(NEXT_ACTION, "planner_prose") == NEXT_ACTION
    project = _project(owner, "Refused Project")

    _write(
        project["project_id"],
        outcome=refused_outcome, next_action=NEXT_ACTION, owner_decision=refused_decision,
    )

    shown = _shown(owner, project)
    assert shown["outcome"] == fixed
    assert shown["owner_decision"] == fixed
    assert shown["next_action"] == NEXT_ACTION
    stored = _stored(project["project_id"])
    assert stored["outcome"] == refused_outcome
    assert stored["owner_decision"] == refused_decision

    # No decision waits: that is absence, not text, so the policy never sees it.
    for nothing in (None, "", "   "):
        _write(project["project_id"], outcome=OUTCOME, next_action=NEXT_ACTION, owner_decision=nothing)
        assert _stored(project["project_id"])["owner_decision"] is None
        shown = _shown(owner, project)
        assert shown["owner_decision"] is None
        assert shown["outcome"] == OUTCOME


def test_revision_goes_up_by_one_on_each_write(owner):
    first = _project(owner, "First Project")
    second = _project(owner, "Second Project")
    drafts = (
        ("The first draft of the plan is written.", "Raphael reviews the draft on Monday."),
        ("Two cards are done and one is running.", "Raphael checks the running card tomorrow."),
        ("All three cards are done.", "Raphael hands the result over on Friday."),
    )

    before = int(time.time())
    revisions = [
        _write(first["project_id"], outcome=outcome, next_action=next_action)
        for outcome, next_action in drafts
    ]
    after = int(time.time())

    assert revisions == [1, 2, 3]
    stored = _stored(first["project_id"])
    assert set(stored) == {"outcome", "next_action", "owner_decision", "written_at", "revision"}
    assert (stored["outcome"], stored["next_action"]) == drafts[-1]
    assert stored["owner_decision"] is None
    assert stored["revision"] == 3
    assert before <= stored["written_at"] <= after
    assert _write(second["project_id"], outcome=OUTCOME, next_action=NEXT_ACTION) == 1
    assert _shown(owner, first)["revision"] == 3

    # A summary belongs to a Project the registry holds, and goes when that Project goes.
    with projects_db.connect_closing() as conn:
        with pytest.raises(sqlite3.IntegrityError):
            projects_db.write_project_summary(conn, "p_missing", outcome=OUTCOME, next_action=NEXT_ACTION)
        assert projects_db.get_project_summary(conn, "p_missing") is None
        assert projects_db.delete_project(conn, second["project_id"]) is True
        assert projects_db.get_project_summary(conn, second["project_id"]) is None
        assert projects_db.get_project_summary(conn, first["project_id"])["revision"] == 3


def test_older_registry_gains_the_summary_without_losing_a_project(tmp_path):
    path = tmp_path / "older" / "projects.db"
    path.parent.mkdir()
    with contextlib.closing(sqlite3.connect(path)) as older:
        older.executescript(_OLDER_REGISTRY_DDL)
        for project_id, slug, name, description, created_at, archived in _OLDER_PROJECTS:
            folder = str(path.parent / slug)
            older.execute(
                "INSERT INTO projects (id, slug, name, description, primary_path, created_at, archived) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (project_id, slug, name, description, folder, created_at, archived),
            )
            older.execute(
                "INSERT INTO project_folders (project_id, path, label, is_primary, added_at) "
                "VALUES (?, ?, NULL, 1, ?)",
                (project_id, folder, created_at),
            )
        older.commit()
    ids = [row[0] for row in _OLDER_PROJECTS]
    expected = {
        project_id: (name, description, bool(archived))
        for project_id, _slug, name, description, _created_at, archived in _OLDER_PROJECTS
    }

    # A read of the older file answers "nothing written" and leaves the file as it was.
    with contextlib.closing(sqlite3.connect(path)) as plain:
        plain.row_factory = sqlite3.Row
        assert [projects_db.get_project_summary(plain, project_id) for project_id in ids] == [None] * len(ids)
        tables = {row["name"] for row in plain.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
        assert "project_summaries" not in tables

    def listed(conn: sqlite3.Connection) -> dict:
        return {
            project.id: (project.name, project.description, project.archived)
            for project in projects_db.list_projects(conn, include_archived=True)
        }

    with contextlib.closing(projects_db.connect(path)) as conn:
        assert listed(conn) == expected
        assert [projects_db.get_project_summary(conn, project_id) for project_id in ids] == [None] * len(ids)

        assert projects_db.write_project_summary(
            conn, ids[0], outcome=OUTCOME, next_action=NEXT_ACTION, owner_decision=DECISION,
        ) == 1

        stored = projects_db.get_project_summary(conn, ids[0])
        assert (stored["outcome"], stored["next_action"], stored["owner_decision"]) == (
            OUTCOME, NEXT_ACTION, DECISION,
        )
        assert isinstance(stored["written_at"], int) and stored["written_at"] > 0
        assert stored["revision"] == 1
        assert [projects_db.get_project_summary(conn, project_id) for project_id in ids[1:]] == [None, None]
        assert listed(conn) == expected


def test_description_is_unchanged(owner):
    project = _project(owner, "Described Project", DESCRIPTION)

    snapshot = ow.read_project_snapshot(owner, project["slug"])
    listed = next(item for item in ow.list_committed_projects(owner) if item["slug"] == project["slug"])

    assert snapshot["project"]["description"] == DESCRIPTION
    assert "raphael_summary" not in snapshot
    assert listed["description"] == DESCRIPTION


def test_registry_works_in_a_dispatcher_worker_environment(tmp_path, monkeypatch):
    """The platform's writer runs inside a dispatched worker.

    A worker has its board pinned by environment and its HERMES_HOME on its own profile directory, and the
    registry follows HERMES_HOME alone.
    """
    root = tmp_path / "root"
    profile_home = root / "profiles" / "planner"
    monkeypatch.setenv("HERMES_HOME", str(profile_home))
    monkeypatch.setenv("HERMES_KANBAN_DB", str(root / "kanban" / "boards" / "worker-board" / "kanban.db"))
    monkeypatch.setenv("HERMES_KANBAN_BOARD", "worker-board")
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_worker0001")

    registry = projects_db.projects_db_path()
    assert registry.parent == profile_home

    with projects_db.connect_closing() as conn:
        project_id = projects_db.create_project(conn, name="Worker Project")
        assert projects_db.write_project_summary(
            conn, project_id, outcome=OUTCOME, next_action=NEXT_ACTION,
        ) == 1
    assert registry.is_file()

    with projects_db.connect_closing() as second:
        stored = projects_db.get_project_summary(second, project_id)
    assert stored is not None
    assert stored["revision"] == 1
    assert (stored["outcome"], stored["next_action"], stored["owner_decision"]) == (
        OUTCOME, NEXT_ACTION, None,
    )
