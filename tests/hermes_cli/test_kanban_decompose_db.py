"""Tests for kb.decompose_triage_task — the DB-layer atomic fan-out
from the triage column. LLM-free by design.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _create_triage(
    conn, title="rough idea", body=None, assignee=None, tenant=None, project_id=None,
):
    return kb.create_task(
        conn,
        title=title,
        body=body,
        assignee=assignee,
        tenant=tenant,
        project_id=project_id,
        triage=True,
    )


def test_decompose_creates_children_and_promotes_root(kanban_home):
    with kb.connect() as conn:
        tid = _create_triage(conn, title="ship a feature")
        assert kb.get_task(conn, tid).status == "triage"

    children = [
        {"title": "research", "body": "look at prior art", "assignee": "researcher", "parents": []},
        {"title": "build it", "body": "write code", "assignee": "engineer", "parents": [0]},
    ]
    with kb.connect() as conn:
        child_ids = kb.decompose_triage_task(
            conn,
            tid,
            root_assignee="orchestrator",
            children=children,
            author="decomposer",
        )
    assert child_ids is not None
    assert len(child_ids) == 2

    with kb.connect() as conn:
        root = kb.get_task(conn, tid)
        c0 = kb.get_task(conn, child_ids[0])
        c1 = kb.get_task(conn, child_ids[1])

    # Root flipped to todo with orchestrator assignee, gated by children.
    assert root.status == "todo"
    assert root.assignee == "orchestrator"
    # First child has no internal parents → ready on recompute_ready.
    assert c0.status == "ready"
    assert c0.assignee == "researcher"
    # Second child has parents=[0] → stays in todo until c0 completes.
    assert c1.status == "todo"
    assert c1.assignee == "engineer"


def test_decompose_records_audit_comment_and_event(kanban_home):
    with kb.connect() as conn:
        tid = _create_triage(conn)
        child_ids = kb.decompose_triage_task(
            conn,
            tid,
            root_assignee="orch",
            children=[{"title": "task A", "assignee": "researcher"}],
            author="alice",
            event_metadata={"operation_digest": "sha256-test", "child_ids": ["forged"]},
        )
    assert child_ids is not None

    with kb.connect() as conn:
        comments = kb.list_comments(conn, tid)
        events = kb.list_events(conn, tid)

    assert any("Decomposed into" in (c.body or "") for c in comments)
    event = next(ev for ev in events if ev.kind == "decomposed")
    assert event.payload["operation_digest"] == "sha256-test"
    assert event.payload["child_ids"] == child_ids


def test_decompose_children_inherit_root_project_id(kanban_home):
    with kb.connect() as conn:
        tid = _create_triage(conn)
        conn.execute("UPDATE tasks SET project_id = ? WHERE id = ?", ("p_owner", tid))
        child_ids = kb.decompose_triage_task(
            conn,
            tid,
            root_assignee="orch",
            children=[
                {"title": "task A", "assignee": "researcher"},
                {"title": "task B", "assignee": "engineer", "parents": [0]},
            ],
            author="alice",
        )

        assert child_ids is not None
        assert [kb.get_task(conn, child_id).project_id for child_id in child_ids] == [
            "p_owner",
            "p_owner",
        ]


def test_decompose_locks_each_preapproved_model_route(kanban_home):
    with kb.connect() as conn:
        tid = _create_triage(conn, title="ship a complex feature")
        child_ids = kb.decompose_triage_task(
            conn,
            tid,
            root_assignee="default",
            children=[{
                "title": "build the feature",
                "assignee": "raphael-claude-worker",
                "execution_tier": "deep",
                "model_override": "claude-opus-5",
                "provider_override": "anthropic",
                "reasoning_effort": "max",
                "model_policy_lock": kb.mint_policy_lock(
                    "raphael-claude-worker", "anthropic", "claude-opus-5",
                    "max", "deep",
                ),
            }],
            author="raphael",
        )

        assert child_ids is not None
        child = kb.get_task(conn, child_ids[0])
        created = next(
            event for event in kb.list_events(conn, child.id)
            if event.kind == "created"
        )

        assert child.model_override == "claude-opus-5"
        assert child.provider_override == "anthropic"
        assert child.reasoning_effort == "max"
        assert child.execution_tier == "deep"
        assert child.model_policy_lock == kb.mint_policy_lock(
            "raphael-claude-worker", "anthropic", "claude-opus-5", "max", "deep",
        )
        assert kb.task_policy_lock_error(
            kb.connect().execute(
                "SELECT * FROM tasks WHERE id = ?", (child.id,)
            ).fetchone()
        ) is None
        assert created.payload["execution_tier"] == "deep"
        assert created.payload["model_route_pinned"] is True

        # The lock is the whole point: no route mutator may move it.
        with pytest.raises(RuntimeError, match="owner-governed"):
            kb.set_model_override(conn, child.id, "claude-sonnet-5", provider="anthropic")
        with pytest.raises(RuntimeError, match="owner-governed"):
            kb.set_reasoning_effort(conn, child.id, "high")

        reread = kb.get_task(conn, child.id)
        assert (reread.model_override, reread.reasoning_effort) == (
            "claude-opus-5",
            "max",
        )


def test_decompose_rejects_a_forbidden_locked_route(kanban_home):
    # A forbidden route cannot even be minted into a lock...
    with pytest.raises(ValueError, match="forbidden model"):
        kb.mint_policy_lock(
            "raphael-claude-worker", "anthropic", "claude-fable-5", "max", "deep",
        )
    with pytest.raises(ValueError, match="forbidden reasoning effort"):
        kb.mint_policy_lock(
            "raphael-claude-worker", "anthropic", "claude-opus-5", "ultra", "deep",
        )
    # ...and a hand-written authority string never authorizes one either.
    real = kb.mint_policy_lock(
        "raphael-claude-worker", "anthropic", "claude-opus-5", "max", "deep",
    )
    with kb.connect() as conn:
        tid = _create_triage(conn, title="ship a complex feature")
        for model, effort, expected in (
            ("claude-fable-5", "max", "forbidden model"),
            ("claude-opus-5", "ultra", "forbidden reasoning effort"),
            ("claude-sonnet-5", "max", "not the admitted route"),
        ):
            with pytest.raises(ValueError, match=expected):
                kb.decompose_triage_task(
                    conn,
                    tid,
                    root_assignee="default",
                    children=[{
                        "title": "build the feature",
                        "assignee": "raphael-claude-worker",
                        "execution_tier": "deep",
                        "model_override": model,
                        "provider_override": "anthropic",
                        "reasoning_effort": effort,
                        "model_policy_lock": real,
                    }],
                    author="raphael",
                )


def test_unlocked_manual_task_routes_stay_mutable(kanban_home):
    """The lock must not leak onto ordinary Kanban work."""
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="manual card", assignee="engineer")
        assert kb.get_task(conn, tid).model_policy_lock is None
        assert kb.set_model_override(conn, tid, "claude-fable-5", provider="anthropic")
        assert kb.set_reasoning_effort(conn, tid, "ultra")

        task = kb.get_task(conn, tid)
        assert (task.model_override, task.reasoning_effort) == (
            "claude-fable-5",
            "ultra",
        )



