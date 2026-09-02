"""Contract tests for the owner-workspace mutation kernel (hermes_cli.owner_workspace)."""

from __future__ import annotations

import json
import inspect
import multiprocessing
import os
import sqlite3
import subprocess
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from hermes_cli import kanban_db, owner_workspace as ow, projects_db
from plugins.dashboard_auth.raphael_workspace import model_policy
from tools import approval


import contextlib


@contextlib.contextmanager
def _temporarily_patch(obj, attr: str, replacement):
    """Swap ``obj.attr`` for the duration of the block, then restore it.

    Deliberately NOT ``pytest``'s ``monkeypatch`` fixture: these crash-
    injection tests also use the ``ctx``/``monkeypatch``-sharing test's own
    ``monkeypatch`` would undo ALL patches made so far by ANY autouse
    fixture sharing that same function-scoped instance — including the
    ``HERMES_HOME`` env sandbox — silently repointing every DB lookup after
    the "crash" at a different, empty home. A plain save/restore has no such
    blast radius.
    """
    original = getattr(obj, attr)
    setattr(obj, attr, replacement)
    try:
        yield
    finally:
        setattr(obj, attr, original)


def _expire_lock(ctx, key: str) -> None:
    """Force a receipt's claim lock to look expired — simulates the passage
    of time past ``_LOCK_TTL_SECONDS`` after a crashed claimer, without
    actually sleeping."""
    with projects_db.connect_closing() as pconn:
        ow._ensure_schema(pconn)
        with ow.write_txn(pconn):
            pconn.execute(
                "UPDATE owner_workspace_receipts SET lock_expires = 0 "
                "WHERE actor = ? AND profile = ? AND idempotency_key = ?",
                (ctx.actor, ctx.profile, key),
            )


def test_receipt_lease_covers_the_full_approval_wait(monkeypatch):
    monkeypatch.setattr("tools.approval._get_approval_timeout", lambda: 7_200)
    assert ow._receipt_lease_seconds() == 7_260


def test_receipt_schema_first_use_is_serialized(tmp_path):
    db_path = tmp_path / "legacy-owner-receipts.db"
    conn = sqlite3.connect(db_path)
    conn.execute(
        """CREATE TABLE owner_workspace_receipts (
            actor TEXT NOT NULL,
            profile TEXT NOT NULL,
            idempotency_key TEXT NOT NULL,
            operation TEXT NOT NULL,
            request_digest TEXT NOT NULL,
            status TEXT NOT NULL,
            lock_token TEXT,
            lock_expires INTEGER,
            project_id TEXT,
            board_slug TEXT,
            task_id TEXT,
            result_json TEXT,
            created_at INTEGER NOT NULL,
            updated_at INTEGER NOT NULL,
            PRIMARY KEY (actor, profile, idempotency_key)
        )"""
    )
    conn.commit()
    conn.close()

    barrier = threading.Barrier(2)
    errors = []

    def migrate():
        worker = sqlite3.connect(db_path, timeout=3, check_same_thread=False)
        try:
            barrier.wait(timeout=1)
            ow._ensure_schema(worker)
        except Exception as exc:  # pragma: no cover - asserted below
            errors.append(exc)
        finally:
            worker.close()

    first = threading.Thread(target=migrate, daemon=True)
    second = threading.Thread(target=migrate, daemon=True)
    first.start()
    second.start()
    first.join(timeout=4)
    second.join(timeout=4)

    assert not first.is_alive()
    assert not second.is_alive()
    assert errors == []
    check = sqlite3.connect(db_path)
    try:
        columns = {
            row[1]
            for row in check.execute(
                "PRAGMA table_info(owner_workspace_receipts)"
            )
        }
        assert {"terminal_generation", "authority_digest"} <= columns
    finally:
        check.close()


@pytest.fixture
def ctx():
    return ow.OwnerContext(actor="default", profile="default", session="run_owt")


def _configured_anthropic(profile):
    """Stand in for the profile's on-disk config — never for the policy.

    Only the *provider selection* a role's config.yaml would carry is faked
    here; the admitted matrix, the tier resolution, and the durable lock
    minting all run as the real production code, so a lock these tests
    persist is one the dispatcher would actually accept. The end-to-end path
    through a real temporary HERMES_HOME is covered by
    ``test_real_profile_config_resolves_and_locks_owner_task_routes``.
    """
    return model_policy.assignment_for(profile, "anthropic")


def _configured_raphael_role(profile):
    provider = "openai-codex" if profile == "raphael-verifier" else "anthropic"
    return model_policy.assignment_for(profile, provider)


@pytest.fixture(autouse=True)
def _configured_provider():
    with _temporarily_patch(
        model_policy, "configured_assignment_for", _configured_anthropic
    ):
        yield


_RAW_COMMIT_TASK_GRAPH = ow.commit_task_graph
_RAW_COMMIT_PROJECT_PLAN = ow.commit_project_plan
_RAW_SET_PROJECT_LIFECYCLE = ow.set_project_archived


def _authorized_context(ctx, operation: str, function, kwargs: dict):
    bound = inspect.signature(function).bind(ctx, **kwargs)
    bound.apply_defaults()
    payload = dict(bound.arguments)
    payload.pop("ctx")
    return ow.OwnerContext(
        actor=ctx.actor,
        profile=ctx.profile,
        session=ctx.session,
        authority=ow.OwnerProposalAuthority(
            actor=ctx.actor,
            profile=ctx.profile,
            session=ctx.session,
            conversation="raphael-owner-" + "a" * 32,
            response_id="resp_" + "b" * 32,
            operation=operation,
            idempotency_key=payload["idempotency_key"],
            payload_digest=ow._digest(payload),
        ),
    )


def _commit_task_graph(ctx, **kwargs):
    return _RAW_COMMIT_TASK_GRAPH(
        _authorized_context(ctx, "owner_task_graph_commit", _RAW_COMMIT_TASK_GRAPH, kwargs),
        **kwargs,
    )


def _commit_project_plan(ctx, **kwargs):
    return _RAW_COMMIT_PROJECT_PLAN(
        _authorized_context(ctx, "owner_project_plan_commit", _RAW_COMMIT_PROJECT_PLAN, kwargs),
        **kwargs,
    )


def _set_project_archived(ctx, **kwargs):
    return _RAW_SET_PROJECT_LIFECYCLE(
        _authorized_context(
            ctx,
            "owner_project_lifecycle",
            _RAW_SET_PROJECT_LIFECYCLE,
            kwargs,
        ),
        **kwargs,
    )


def _auto_approve(session_key: str, choice: str = "once", timeout: float = 5.0) -> None:
    """Resolve the next queued approval for *session_key* by its approval_id."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        with approval._lock:
            queue = approval._gateway_queues.get(session_key)
            aid = queue[0].approval_id if queue else None
        if aid:
            approval.resolve_gateway_approval(session_key, choice, approval_id=aid)
            return
        time.sleep(0.005)
    raise TimeoutError("no approval was queued")


def _with_approver(session_key: str, choice: str = "once"):
    approval.register_gateway_notify(session_key, lambda data: None)
    t = threading.Thread(target=_auto_approve, args=(session_key, choice))
    t.start()
    return t


def test_bootstrap_denies_without_session(ctx):
    denied_ctx = ow.OwnerContext(actor="default", profile="default", session="")
    result = ow.bootstrap(denied_ctx, idempotency_key="k1", name="No Session")
    assert result == {"ok": False, "error": "confirmation_denied", "reason": "no_session"}


def test_bootstrap_creates_exactly_one_project_board_task(ctx):
    t = _with_approver(ctx.session)
    result = ow.bootstrap(ctx, idempotency_key="k1", name="Owner WS", description="d")
    t.join()
    assert result["ok"] is True

    with projects_db.connect_closing() as pconn:
        project = projects_db.get_project(pconn, result["project_id"])
        assert project is not None
        assert project.board_slug == result["board"]

    assert kanban_db.board_exists(result["board"])
    kconn = kanban_db.connect(board=result["board"])
    try:
        # The anchor is a non-executable control row, so it resolves only
        # through the control reader — never through the executable one.
        assert kanban_db.get_task(kconn, result["task_id"]) is None
        task = kanban_db.get_control_task(kconn, result["task_id"])
        assert task is not None
        assert task.title == "Owner WS"
    finally:
        kconn.close()


def test_bootstrap_exact_replay_returns_same_result(ctx):
    t = _with_approver(ctx.session)
    first = ow.bootstrap(ctx, idempotency_key="k2", name="Replay WS")
    t.join()

    # No approver registered for the replay — if it re-prompted, it would
    # hang/deny for lack of a session surface, proving no second decision
    # was requested.
    approval.unregister_gateway_notify(ctx.session)
    second = ow.bootstrap(ctx, idempotency_key="k2", name="Replay WS")
    assert second == first


def test_bootstrap_different_payload_same_key_conflicts_with_zero_changes(ctx):
    t = _with_approver(ctx.session)
    ow.bootstrap(ctx, idempotency_key="k3", name="Original")
    t.join()

    with projects_db.connect_closing() as pconn:
        before = len(projects_db.list_projects(pconn, include_archived=True))

    with pytest.raises(ow.OwnerWorkspaceError) as excinfo:
        ow.bootstrap(ctx, idempotency_key="k3", name="Different Name")
    assert excinfo.value.code == "idempotency_key_conflict"

    with projects_db.connect_closing() as pconn:
        after = len(projects_db.list_projects(pconn, include_archived=True))
    assert after == before


def test_concurrent_same_key_bootstrap_creates_exactly_one(ctx):
    t = _with_approver(ctx.session)
    results = [None, None]

    def _call(i):
        results[i] = ow.bootstrap(ctx, idempotency_key="concurrent-key", name="Concurrent WS")

    threads = [threading.Thread(target=_call, args=(i,)) for i in range(2)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    t.join()

    assert results[0] == results[1]
    assert results[0]["ok"] is True

    with projects_db.connect_closing() as pconn:
        projects = [p for p in projects_db.list_projects(pconn, include_archived=True) if p.name == "Concurrent WS"]
    assert len(projects) == 1


def test_bootstrap_denial_is_zero_change_and_replays_the_same_denial(ctx):
    t = _with_approver(ctx.session, choice="deny")
    result = ow.bootstrap(ctx, idempotency_key="k-deny", name="Denied WS")
    t.join()
    assert result["ok"] is False
    assert result["error"] == "confirmation_denied"

    with projects_db.connect_closing() as pconn:
        assert all(p.name != "Denied WS" for p in projects_db.list_projects(pconn, include_archived=True))

    approval.unregister_gateway_notify(ctx.session)
    replay = ow.bootstrap(ctx, idempotency_key="k-deny", name="Denied WS")
    assert replay == result



def _task_graph_args(**overrides):
    args = {
        "idempotency_key": "graph-1",
        "mode": "new",
        "project_name": "Launch Shop",
        "project_description": "A plain-English owner project.",
        "project_id": None,
        "request_title": "Launch the first useful version",
        "specification": "Build and verify the smallest owner-visible release.",
        "current_milestone": "Now: produce one working release.",
        "owner_visible_result": "The owner can open and verify the release.",
        "root_assignee": "default",
        "tasks": [
            {
                "title": "Prepare the release",
                "body": "Create the smallest complete release.",
                "assignee": "default",
                "responsibility": "B03",
                "execution_tier": "routine",
                "parents": [],
            },
            {
                "title": "Verify the release",
                "body": "Check the owner-visible result.",
                "assignee": "default",
                "responsibility": "R12",
                "execution_tier": "routine",
                "parents": [0],
            },
        ],
        "later_milestones": [
            "Next: learn from the first release.",
            "Later: expand only after evidence.",
        ],
    }
    args.update(overrides)
    return args


def test_task_graph_requires_authenticated_proposal_authority(ctx):
    with pytest.raises(ow.OwnerWorkspaceError) as excinfo:
        _RAW_COMMIT_TASK_GRAPH(ctx, **_task_graph_args())
    assert excinfo.value.code == "owner_run_authority_required"


def test_task_graph_authority_is_bound_to_the_exact_payload(ctx):
    args = _task_graph_args()
    authorized = _authorized_context(
        ctx, "owner_task_graph_commit", _RAW_COMMIT_TASK_GRAPH, args,
    )
    args["request_title"] = "A different milestone"
    with pytest.raises(ow.OwnerWorkspaceError) as excinfo:
        _RAW_COMMIT_TASK_GRAPH(authorized, **args)
    assert excinfo.value.code == "owner_run_authority_required"


def test_task_graph_commit_creates_native_project_and_atomic_graph(ctx):
    args = _task_graph_args()
    approver = _with_approver(ctx.session)
    result = _commit_task_graph(ctx, **args)
    approver.join()

    assert result["ok"] is True
    assert result["mode"] == "new"
    assert result["task_count"] == 2
    # The durable receipt records the state at receipt time: every approved
    # task parked and non-claimable. Activation happens strictly after it.
    assert result["task_statuses"] == ["scheduled", "scheduled"]
    assert result["parked_task_ids"] == result["task_ids"]

    with projects_db.connect_closing() as pconn:
        project = projects_db.get_project(pconn, result["project_id"])
        assert project is not None
        assert project.slug == result["project_slug"]
        assert project.board_slug == result["board"]

    with kanban_db.connect(board=result["board"]) as kconn:
        root = kanban_db.get_task(kconn, result["root_task_id"])
        first = kanban_db.get_task(kconn, result["task_ids"][0])
        second = kanban_db.get_task(kconn, result["task_ids"][1])
        anchors = [
            str(row["id"])
            for row in kconn.execute(
                "SELECT id FROM tasks WHERE project_id = ? AND task_kind = 'control'",
                (result["project_id"],),
            )
        ]
        second_parents = {
            row["parent_id"]
            for row in kconn.execute(
                "SELECT parent_id FROM task_links WHERE child_id = ?",
                (second.id,),
            )
        }

    # Exactly one hidden control anchor, receipt-bound, never a work row.
    assert anchors == [result["anchor_task_id"]]
    # ...and the approved work is live once the receipt committed.
    assert [first.status, second.status] == ["ready", "todo"]

    assert root.project_id == result["project_id"]
    assert root.status == "todo"
    assert root.assignee == "default"
    assert "Later roadmap" in root.body
    assert first.project_id == result["project_id"]
    assert first.responsibility == "B03"
    assert second.project_id == result["project_id"]
    assert second.responsibility == "R12"
    assert first.id in second_parents


def test_task_graph_resolves_and_locks_model_routes_before_approval(ctx):
    args = _task_graph_args(idempotency_key="graph-model-routes")
    args["tasks"][0]["execution_tier"] = "deep"
    args["tasks"][1]["execution_tier"] = "routine"

    resolved: list[tuple[str, str]] = []

    real_resolve = ow.resolve_task_assignment

    def resolved_route(profile, execution_tier):
        resolved.append((profile, execution_tier))
        return real_resolve(profile, execution_tier)

    with _temporarily_patch(ow, "resolve_task_assignment", resolved_route):
        approver = _with_approver(ctx.session)
        result = _commit_task_graph(ctx, **args)
        approver.join()

    # Every route resolved before the owner was asked to confirm.
    assert resolved == [
        ("default", "deep"), ("default", "routine"), ("default", "deep"),
    ]

    with kanban_db.connect(board=result["board"]) as conn:
        root = kanban_db.get_task(conn, result["root_task_id"])
        first = kanban_db.get_task(conn, result["task_ids"][0])
        second = kanban_db.get_task(conn, result["task_ids"][1])

    def route(task):
        return (
            task.provider_override,
            task.model_override,
            task.reasoning_effort,
            task.model_policy_lock,
        )

    def lock(assignee, model, tier):
        return kanban_db.mint_policy_lock(
            assignee, "anthropic", model, "max", tier,
        )

    # 'default'/anthropic admits claude-opus-5 on BOTH lanes, so the digest —
    # not the model — is what distinguishes the deep pin from the routine one.
    assert route(first) == (
        "anthropic", "claude-opus-5", "max", lock("default", "claude-opus-5", "deep"),
    )
    assert route(second) == (
        "anthropic", "claude-opus-5", "max",
        lock("default", "claude-opus-5", "routine"),
    )
    assert (first.execution_tier, second.execution_tier) == ("deep", "routine")
    # The executable root reviews a milestone containing deep work, so it is
    # pinned too — and on the deep lane.
    assert route(root) == (
        "anthropic", "claude-opus-5", "max", lock("default", "claude-opus-5", "deep"),
    )
    # Every persisted lock is one the dispatcher would actually accept.
    with kanban_db.connect(board=result["board"]) as conn:
        for task_id in (result["root_task_id"], *result["task_ids"]):
            row = conn.execute(
                "SELECT * FROM tasks WHERE id = ?", (task_id,)
            ).fetchone()
            assert kanban_db.task_policy_lock_error(row) is None


def test_task_graph_root_is_pinned_on_the_routine_lane_for_routine_work(ctx):
    approver = _with_approver(ctx.session)
    result = _commit_task_graph(ctx, **_task_graph_args(idempotency_key="graph-root-routine"))
    approver.join()

    with kanban_db.connect(board=result["board"]) as conn:
        root = kanban_db.get_task(conn, result["root_task_id"])

    assert (root.execution_tier, root.model_policy_lock) == (
        "routine",
        kanban_db.mint_policy_lock(
            "default", "anthropic", root.model_override, "max", "routine",
        ),
    )


@pytest.mark.parametrize("tier", [None, "", "ultra", "Ultracode", "complex", 1])
def test_task_graph_requires_an_admitted_execution_tier_before_approval(ctx, tier):
    args = _task_graph_args(idempotency_key=f"graph-tier-{tier!r}")
    if tier is None:
        del args["tasks"][0]["execution_tier"]
    else:
        args["tasks"][0]["execution_tier"] = tier

    with pytest.raises(ow.OwnerWorkspaceError) as excinfo:
        _commit_task_graph(ctx, **args)

    assert excinfo.value.code == "invalid_model_route"

    with projects_db.connect_closing() as pconn:
        assert all(
            p.name != "Launch Shop"
            for p in projects_db.list_projects(pconn, include_archived=True)
        )


def test_committed_owner_task_route_cannot_be_mutated_afterwards(ctx):
    approver = _with_approver(ctx.session)
    result = _commit_task_graph(ctx, **_task_graph_args(idempotency_key="graph-immutable"))
    approver.join()

    with kanban_db.connect(board=result["board"]) as conn:
        for task_id in (result["root_task_id"], *result["task_ids"]):
            with pytest.raises(RuntimeError, match="owner-governed"):
                kanban_db.set_model_override(
                    conn, task_id, "claude-fable-5", provider="anthropic",
                )
            with pytest.raises(RuntimeError, match="owner-governed"):
                kanban_db.set_reasoning_effort(conn, task_id, "ultra")
            task = kanban_db.get_task(conn, task_id)
            assert task.model_override == "claude-opus-5"
            assert task.reasoning_effort == "max"


def test_task_graph_rejects_invalid_responsibility_before_approval(ctx):
    args = _task_graph_args()
    args["tasks"][0]["responsibility"] = "role with spaces"

    with pytest.raises(ow.OwnerWorkspaceError) as excinfo:
        _commit_task_graph(ctx, **args)

    assert excinfo.value.code == "invalid_argument"


def test_task_graph_commit_existing_project_reuses_exact_native_board(ctx):
    setup = _bootstrap_board(ctx)
    args = _task_graph_args(
        idempotency_key="graph-existing",
        mode="existing",
        project_id=setup["project_id"],
        project_name=None,
        project_description=None,
        later_milestones=[],
    )

    approver = _with_approver(ctx.session)
    result = _commit_task_graph(ctx, **args)
    approver.join()

    assert result["ok"] is True
    assert result["mode"] == "existing"
    assert result["project_id"] == setup["project_id"]
    assert result["board"] == setup["board"]


def test_task_graph_exact_replay_creates_no_duplicate_tasks(ctx):
    args = _task_graph_args(idempotency_key="graph-replay")
    approver = _with_approver(ctx.session)
    first = _commit_task_graph(ctx, **args)
    approver.join()

    with kanban_db.connect(board=first["board"]) as kconn:
        before = kconn.execute(
            "SELECT COUNT(*) AS n FROM tasks WHERE project_id = ?",
            (first["project_id"],),
        ).fetchone()["n"]

    approval.unregister_gateway_notify(ctx.session)
    second = _commit_task_graph(ctx, **args)

    with kanban_db.connect(board=first["board"]) as kconn:
        after = kconn.execute(
            "SELECT COUNT(*) AS n FROM tasks WHERE project_id = ?",
            (first["project_id"],),
        ).fetchone()["n"]

    assert second == first
    # root + two children + the Project's one hidden control anchor
    assert before == after == 4


def test_task_graph_rejects_fake_whole_project_before_persistence(ctx):
    args = _task_graph_args(
        idempotency_key="graph-too-large",
        tasks=[
            {
                "title": f"Task {index}",
                "body": "Too much speculative work.",
                "assignee": "default",
                "execution_tier": "routine",
                "parents": [],
            }
            for index in range(13)
        ],
    )

    with pytest.raises(ow.OwnerWorkspaceError) as excinfo:
        _commit_task_graph(ctx, **args)

    assert excinfo.value.code == "milestone_too_large"
    with projects_db.connect_closing() as pconn:
        assert all(
            project.name != "Launch Shop"
            for project in projects_db.list_projects(pconn, include_archived=True)
        )


def test_task_graph_rejects_cycle_before_persistence(ctx):
    args = _task_graph_args(
        idempotency_key="graph-cycle",
        tasks=[
            {
                "title": "A", "body": "A", "assignee": "default",
                "execution_tier": "routine", "parents": [1],
            },
            {
                "title": "B", "body": "B", "assignee": "default",
                "execution_tier": "routine", "parents": [0],
            },
        ],
    )

    with pytest.raises(ow.OwnerWorkspaceError) as excinfo:
        _commit_task_graph(ctx, **args)

    assert excinfo.value.code == "invalid_graph"


def test_committed_project_projection_is_receipt_backed_and_read_only(ctx):
    args = _task_graph_args(idempotency_key="graph-list", project_name="Listed Project")
    approver = _with_approver(ctx.session)
    result = _commit_task_graph(ctx, **args)
    approver.join()

    before = projects_db.projects_db_path().stat().st_mtime_ns
    listed = ow.list_committed_projects(ctx)
    after = projects_db.projects_db_path().stat().st_mtime_ns

    assert before == after
    assert listed == [{
        "project_id": result["project_id"],
        "slug": result["project_slug"],
        "name": "Listed Project",
        "description": "A plain-English owner project.",
        "board": result["board"],
        "archived": False,
    }]
    assert ow.list_committed_projects(
        ctx, lifecycle_revision=True,
    )[0]["lifecycle_revision"] == 0


def test_project_snapshot_is_exact_receipt_backed_and_read_only(ctx):
    args = _task_graph_args(
        idempotency_key="graph-snapshot",
        project_name="Snapshot Project",
    )
    approver = _with_approver(ctx.session)
    result = _commit_task_graph(ctx, **args)
    approver.join()

    project_db = projects_db.projects_db_path()
    board_db = kanban_db.board_dir(result["board"]) / "kanban.db"
    before = (project_db.stat().st_mtime_ns, board_db.stat().st_mtime_ns)

    snapshot = ow.read_project_snapshot(ctx, result["project_slug"])

    after = (project_db.stat().st_mtime_ns, board_db.stat().st_mtime_ns)
    assert before == after
    assert snapshot["project"] == {
        "id": result["project_id"],
        "slug": result["project_slug"],
        "name": "Snapshot Project",
        "description": "A plain-English owner project.",
        "board": result["board"],
        "archived": False,
    }
    assert snapshot["board"]["slug"] == result["board"]
    assert snapshot["board"]["project_id"] == result["project_id"]
    assert snapshot["board"]["total"] == 3
    assert snapshot["board"]["counts"]["ready"] == 1
    assert snapshot["board"]["counts"]["todo"] == 2
    tasks = [task for column in snapshot["columns"] for task in column["tasks"]]
    assert len(tasks) == 3
    assert all(set(task) == {
        "id", "title", "assignee_name", "responsibility", "updated_at",
        "event_revision", "parent_ids", "child_ids",
    } for task in tasks)
    assert snapshot["workers"] == []
    assert snapshot["attachments"] == []
    assert snapshot["runs"] == []
    assert "planning_context" not in snapshot
    assert snapshot["steward"]["schema_version"] == 2
    assert snapshot["steward"]["execution"]["state"] == "working"
    assert snapshot["steward"]["execution"]["paused"] is False


def test_project_snapshot_planning_context_prioritizes_live_relations_and_recent_history(
    ctx,
):
    args = _task_graph_args(
        idempotency_key="graph-planning-context",
        project_name="Planning Context Project",
    )
    approver = _with_approver(ctx.session)
    result = _commit_task_graph(ctx, **args)
    approver.join()

    with kanban_db.connect(board=result["board"]) as conn:
        generic_done = [
            kanban_db.create_task(
                conn,
                title=f"Completed work {index}",
                project_id=result["project_id"],
            )
            for index in range(196)
        ]
        archived = [
            kanban_db.create_task(
                conn,
                title=f"Archived work {index}",
                project_id=result["project_id"],
            )
            for index in range(2)
        ]
        artifact_parent = kanban_db.create_task(
            conn,
            title="Produce the retained artifact",
            project_id=result["project_id"],
        )
        with kanban_db.write_txn(conn):
            conn.execute(
                "UPDATE tasks SET status = 'done' WHERE id IN ("
                + ",".join("?" for _ in (*generic_done, artifact_parent))
                + ")",
                (*generic_done, artifact_parent),
            )
            conn.execute(
                "UPDATE tasks SET status = 'archived' WHERE id IN (?, ?)",
                archived,
            )
            # The related artifact is deliberately oldest. It must still be
            # retained ahead of unrelated terminal history, while the recent
            # terminal window keeps the newest unrelated completion.
            for index, task_id in enumerate(generic_done):
                conn.execute(
                    "UPDATE task_events SET created_at = ? WHERE task_id = ?",
                    (10_000 + index, task_id),
                )
            conn.execute(
                "UPDATE task_events SET created_at = 1 WHERE task_id = ?",
                (artifact_parent,),
            )
        dependent = kanban_db.create_task(
            conn,
            title="Use the retained artifact",
            project_id=result["project_id"],
            parents=[artifact_parent],
        )

    snapshot = ow.read_project_snapshot(
        ctx,
        result["project_slug"],
        planning_context=True,
    )

    context = snapshot["planning_context"]
    assert set(context) == {
        "schema_version",
        "actionable_count",
        "omitted_terminal_count",
        "actionable_truncated",
        "relations_truncated",
        "tasks",
    }
    assert context["schema_version"] == 1
    assert context["actionable_count"] == 4
    assert context["actionable_truncated"] is False
    assert context["relations_truncated"] is False
    assert len(context["tasks"]) == ow._OWNER_PROJECT_MAX_TASKS
    assert context["omitted_terminal_count"] == 3
    assert (
        sum(snapshot["board"]["counts"][status] for status in ("done", "archived"))
        == context["omitted_terminal_count"]
        + sum(task["status"] in {"done", "archived"} for task in context["tasks"])
    )
    by_id = {task["id"]: task for task in context["tasks"]}
    assert artifact_parent in by_id
    assert dependent in by_id
    assert artifact_parent in by_id[dependent]["parent_ids"]
    assert by_id[dependent]["omitted_parent_count"] == 0
    assert by_id[dependent]["omitted_child_count"] == 0
    assert generic_done[0] not in by_id
    assert generic_done[-1] in by_id
    assert all(set(task) == {
        "id", "title", "status", "assignee_name", "responsibility",
        "updated_at", "event_revision", "parent_ids", "child_ids",
        "omitted_parent_count", "omitted_child_count",
    } for task in context["tasks"])


@pytest.mark.parametrize("unowned", [False, True])
def test_project_snapshot_marks_foreign_work_relations_without_disclosing_them(
    ctx,
    unowned,
):
    args = _task_graph_args(
        idempotency_key=f"graph-foreign-relation-{unowned}",
        project_name="Foreign Relation Project",
    )
    approver = _with_approver(ctx.session)
    result = _commit_task_graph(ctx, **args)
    approver.join()

    with kanban_db.connect(board=result["board"]) as conn:
        foreign_parent = kanban_db.create_task(
            conn,
            title="Foreign predecessor",
            project_id="p_foreign",
        )
        dependent = kanban_db.create_task(
            conn,
            title="Project-owned dependent",
            project_id=result["project_id"],
            parents=[foreign_parent],
        )
        if unowned:
            with kanban_db.write_txn(conn):
                conn.execute(
                    "UPDATE tasks SET project_id = NULL WHERE id = ?",
                    (foreign_parent,),
                )

    context = ow.read_project_snapshot(
        ctx,
        result["project_slug"],
        planning_context=True,
    )["planning_context"]

    assert context["relations_truncated"] is True
    by_id = {task["id"]: task for task in context["tasks"]}
    assert foreign_parent not in by_id
    assert by_id[dependent]["parent_ids"] == []
    assert by_id[dependent]["omitted_parent_count"] == 1
    assert foreign_parent not in json.dumps(context)


def test_owner_title_projection_vectors_are_shared_with_the_owner_workspace():
    """Pinned outputs the owner Workspace's own title projection must match.

    Both the receipt snapshot and the capability-gated native machine read
    call this public helper, so their owner-visible titles cannot drift.
    """
    assert ow.owner_title("B03 — Ship the thing") == "Ship the thing"
    assert ow.owner_title("R07: Fix the outage") == "Fix the outage"
    assert ow.owner_title("R07-Fix the outage") == "Fix the outage"
    assert ow.owner_title("R7-Fix the outage") == "R7-Fix the outage"
    assert ow.owner_title("B03 — R07: Ship the thing") == "Ship the thing"
    assert ow.owner_title(ow.owner_title("B03 — R07: Ship the thing")) == "Ship the thing"
    assert ow.owner_title("R07:   Fix   the\n  outage  ") == "Fix the outage"
    assert ow.owner_title("Rotate the shop key") == "Rotate the shop key"
    assert ow.owner_title("Version 12 of the plan") == "Version 12 of the plan"
    assert (
        ow.owner_title("Rotate ghp_ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 before Friday")
        == "Rotate ghp_AB...6789 before Friday"
    )
    assert ow.owner_title("Key sk-ABCDEFGHIJ rotated") == "Key *** rotated"
    assert (
        ow.owner_title("B03 — eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJvd25lciJ9")
        == "eyJhbG...ciJ9"
    )
    assert ow.owner_title("   ") == "Untitled work item"
    assert ow.owner_title("B03 —") == "Untitled work item"
    assert ow.owner_title(None) == "Untitled work item"
    # The cap counts Unicode code points, so an astral title keeps 240 of them.
    assert ow.owner_title("🚀" * 240) == "🚀" * 240
    assert ow.owner_title("🚀" * 241) == "🚀" * 240
    # The prefixed-and-redacted vector the Workspace parity test reads through
    # both a native board and a receipt snapshot.
    assert (
        ow.owner_title(
            "B03 —  Rotate ghp_ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789   before Friday"
        )
        == "Rotate ghp_AB...6789 before Friday"
    )


@pytest.mark.parametrize(
    "value",
    [
        "Use OpenAI for review",
        "Run GPT-5.6 for verification",
        "Open /srv/raphael/private/config.yaml",
        "Project 123e4567-e89b-12d3-a456-426614174000",
    ],
)
def test_owner_title_hides_private_operational_detail_but_project_name_keeps_owner_text(
    value,
):
    assert ow.owner_title(value) == "Untitled work item"
    assert ow.owner_project_name(value) == value


def test_owner_title_masks_url_credentials_at_a_non_navigation_egress():
    """A title is never followed as a link, so URL credentials are masked.

    Ordinary display redaction deliberately leaves OAuth callback codes,
    magic links and pre-signed URLs actionable. An owner-visible title is the
    opposite kind of boundary: the strict URL-credential mode has to strip
    credential-named query parameters, pre-signed signatures and
    ``user:password@`` userinfo before the title is projected.
    """
    assert (
        ow.owner_title("Retry https://api.example.com/v1/sync?token=OPAQUESECRET123&page=2")
        == "Retry https://api.example.com/v1/sync?token=***&page=2"
    )
    assert (
        ow.owner_title(
            "Upload https://bucket.s3.example.com/report.pdf"
            "?X-Amz-Expires=900&X-Amz-Signature=abcdef0123456789abcdef"
        )
        == "Upload https://bucket.s3.example.com/report.pdf"
        "?X-Amz-Expires=900&X-Amz-Signature=***"
    )
    # The whole userinfo is masked, username included: the credential is
    # routinely the USERNAME in these forms, so keeping it would publish the
    # token and mask only the constant marker beside it.
    assert (
        ow.owner_title("Mirror https://deploy:hunter2verylongpassword@git.example.com/repo.git")
        == "Mirror https://***@git.example.com/repo.git"
    )
    assert (
        ow.owner_title(
            "Clone https://opaque-placeholder-value:x-oauth-basic"
            "@github.com/example/repo.git"
        )
        == "Clone https://***@github.com/example/repo.git"
    )
    # A public parameter that merely resembles a credential name is not a
    # credential and must survive intact.
    assert (
        ow.owner_title("Open https://api.example.com/jobs?token_count=17&session_id=public")
        == "Open https://api.example.com/jobs?token_count=17&session_id=public"
    )


def test_owner_title_masks_the_common_signed_url_credential_keys():
    """A pre-signed URL's own credential keys are masked at this boundary.

    Ordinary display redaction leaves a pre-signed URL clickable on purpose.
    A title is never followed, so the three signed-URL families that carry
    bearer authority in a query parameter — the AWS SigV4 session token, the
    GCS V4 signature and the Azure SAS ``sig`` — are masked here, in their
    canonical and percent-encoded spellings alike. The public parameters that
    make the URL readable (expiry, algorithm, resource) are left alone.
    """
    assert (
        ow.owner_title(
            "Fetch https://bucket.s3.example.com/report.pdf"
            "?X-Amz-Algorithm=AWS4-HMAC-SHA256"
            "&X-Amz-Security-Token=placeholder-session-value"
        )
        == "Fetch https://bucket.s3.example.com/report.pdf"
        "?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Security-Token=***"
    )
    assert (
        ow.owner_title(
            "Fetch https://storage.example.com/o/report.pdf"
            "?X-Goog-Expires=900&X-Goog-Signature=deadbeefdeadbeefdeadbeef"
        )
        == "Fetch https://storage.example.com/o/report.pdf"
        "?X-Goog-Expires=900&X-Goog-Signature=***"
    )
    assert (
        ow.owner_title(
            "Fetch https://acct.blob.example.com/c/report.pdf"
            "?sp=r&sr=b&sig=placeholder-sas-value"
        )
        == "Fetch https://acct.blob.example.com/c/report.pdf?sp=r&sr=b&sig=***"
    )
    # The canonical form decodes the key, folds its case, and folds ``-`` to
    # ``_``, so an encoded or reshaped spelling of the same key matches too —
    # and the ORIGINAL spelling is what the owner still reads.
    for key in ("x_amz_security_token", "X%2DAmz%2DSecurity%2DToken", "SIG"):
        assert (
            ow.owner_title(
                f"Fetch https://acct.example.com/o?{key}=placeholder-value&sr=b"
            )
            == f"Fetch https://acct.example.com/o?{key}=***&sr=b"
        )
    assert (
        ow.owner_title(
            "Fetch https://acct.example.com/o?to%E2%80%8Bken=opaque-value&sr=b"
        )
        == "Fetch https://acct.example.com/o?to%E2%80%8Bken=***&sr=b"
    )
    assert (
        ow.owner_title(
            "Mirror https://public@credential-value@example.com/repo.git"
        )
        == "Mirror https://***@example.com/repo.git"
    )


def test_owner_title_removes_unsafe_control_and_display_characters():
    """Control/display characters are removed before anything else runs.

    A title reaches the owner through UIs that honour escape sequences and
    bidirectional formatting, so NUL, ESC sequences, 8-bit C1 controls,
    zero-width characters, bidi overrides/isolates and invisible plane-14 tag
    characters cannot be allowed to survive. Sanitizing FIRST is what stops a
    credential from hiding behind an invisible character that would otherwise
    split its token past the redactor.
    """
    zero_width, rl_override, pop_isolate = chr(0x200B), chr(0x202E), chr(0x2069)

    assert ow.owner_title("Ship\x00 the\x1b[31m thing\x9b1m") == "Ship the thing"
    assert (
        ow.owner_title(f"Sh{zero_width}ip the {rl_override}thing{pop_isolate}")
        == "Ship the thing"
    )
    assert (
        ow.owner_title("Ship the thing\U000e0041\U000e0042") == "Ship the thing"
    )
    assert ow.owner_title(f"\x00{zero_width}{rl_override}") == "Untitled work item"
    # Safe human Unicode is not what this removes.
    assert ow.owner_title("Café naïve 日本語 \U0001f680") == (
        "Café naïve 日本語 \U0001f680"
    )
    # A credential split by an invisible character is still a credential.
    assert (
        ow.owner_title(
            f"Rotate ghp_ABCDEF{zero_width}GHIJKLMNOPQRSTUVWXYZ0123456789 before Friday"
        )
        == "Rotate ghp_AB...6789 before Friday"
    )
    # The 240 code point bound is applied to already-sanitized text, so
    # padding a title with invisible characters cannot shorten what it says.
    assert ow.owner_title(f"a{zero_width}" * 300) == "a" * 240


def test_owner_title_removes_bidi_marks_and_deprecated_directional_controls():
    """A title loses more than the shared threat set covers.

    The deprecated directional controls (U+206A-U+206F) and U+2061 function
    application are invisible outright, so they are threats anywhere and live
    in ``INVISIBLE_CHARS``. The plain bidi marks — U+061C arabic letter mark,
    U+200E LRM, U+200F RLM — deliberately do NOT: correctly written Arabic and
    Hebrew prose contains them, so treating them as unconditional injection
    markers blocked legitimate multilingual memory and AGENTS content. A title
    is a short label rather than prose, so this boundary removes them from its
    own owner-scoped set instead.

    Either way they go before redaction, so none of them can split a
    credential past the redactor.

    Code points are written as ``chr(...)`` rather than pasted: a literal
    invisible character in this file would be unreviewable.
    """
    from tools.threat_patterns import INVISIBLE_CHARS

    shared_controls = (
        chr(0x2061),  # function application
        chr(0x206A),  # inhibit symmetric swapping
        chr(0x206B),  # activate symmetric swapping
        chr(0x206C),  # inhibit arabic form shaping
        chr(0x206D),  # activate arabic form shaping
        chr(0x206E),  # national digit shapes
        chr(0x206F),  # nominal digit shapes
    )
    owner_only_marks = (
        chr(0x061C),  # arabic letter mark
        chr(0x200E),  # left-to-right mark
        chr(0x200F),  # right-to-left mark
    )
    for mark in shared_controls:
        assert mark in INVISIBLE_CHARS
    for mark in owner_only_marks:
        assert mark not in INVISIBLE_CHARS

    for mark in (*shared_controls, *owner_only_marks):
        assert ow.owner_title(f"Sh{mark}ip the {mark}thing") == "Ship the thing"
        assert ow.owner_title(mark * 3) == "Untitled work item"
        assert ow.owner_project_name(f"Shoe{mark} Shop{mark}") == "Shoe Shop"

    # Removal runs BEFORE redaction, so a credential split across one of
    # these is still redacted rather than projected in full.
    assert (
        ow.owner_title(
            f"Rotate ghp_ABCDEF{chr(0x200F)}GHIJKLMNOPQRSTUVWXYZ0123456789 before Friday"
        )
        == "Rotate ghp_AB...6789 before Friday"
    )


def test_owner_projections_remove_default_ignorables_and_lone_surrogates():
    """Invisible-but-not-INVISIBLE_CHARS code points and invalid scalars.

    Three separate failures share one cause — a character the owner cannot
    see surviving into the projected string:

    * a credential or a query-parameter NAME split across one of them walks
      past the redactor, because neither the vendor prefix regexes nor the
      strict URL parameter pattern spans a foreign code point;
    * a run of them spends the display bound while showing nothing, so the
      visible text is silently truncated;
    * a lone surrogate is not a Unicode scalar value at all — it survives in
      a ``str`` but raises ``UnicodeEncodeError`` on the way out, turning an
      owner read into a 500 from FastAPI's JSON encoder.

    Code points are written as ``chr(...)`` rather than pasted: a literal
    invisible character in this file would be unreviewable.
    """
    ignorables = (
        chr(0x00AD),   # soft hyphen
        chr(0x034F),   # combining grapheme joiner
        chr(0x180E),   # mongolian vowel separator
        chr(0xFE00),   # variation selector-1
        chr(0xFE0F),   # variation selector-16
        chr(0xE0100),  # variation selector-17 (supplement)
        chr(0xE01EF),  # variation selector-256 (supplement)
        chr(0xD800),   # lone high surrogate
        chr(0xDC00),   # lone low surrogate
        chr(0xDFFF),   # lone low surrogate (top of range)
    )
    for char in ignorables:
        assert ow.owner_title(f"Sh{char}ip the {char}thing") == "Ship the thing"
        assert ow.owner_title(char * 3) == "Untitled work item"
        assert ow.owner_project_name(f"Shoe{char} Shop{char}") == "Shoe Shop"
        assert ow.owner_project_name(char * 3) == "Untitled Project"

        # A credential split across one of them is still a credential.
        assert (
            ow.owner_title(
                f"Rotate ghp_ABCDEF{char}GHIJKLMNOPQRSTUVWXYZ0123456789 now"
            )
            == "Rotate ghp_AB...6789 now"
        )
        # So is a query-parameter NAME split across one of them: the strict
        # URL pattern's key class stops at the foreign code point, so without
        # this removal the whole parameter goes unrecognised and the token is
        # projected verbatim.
        assert (
            ow.owner_title(
                f"Retry https://api.example.com/v1/sync?to{char}ken=OPAQUESECRET123&page=2"
            )
            == "Retry https://api.example.com/v1/sync?token=***&page=2"
        )
        # Invisible padding cannot eat the bound.
        assert ow.owner_title(f"a{char}" * 300) == "a" * 240
        assert ow.owner_project_name(f"a{char}" * 300) == "a" * 160

    # Every projection is UTF-8 encodable, which is what FastAPI's JSON
    # encoder needs and what a lone surrogate would otherwise break.
    surrogate_title = ow.owner_title(f"Ship{chr(0xD800)} the thing")
    assert surrogate_title == "Ship the thing"
    assert json.dumps(
        {"title": surrogate_title}, ensure_ascii=False
    ).encode("utf-8")

    # Emoji presentation is a variation selector, so it goes with them; the
    # base glyph the owner actually sees stays.
    assert ow.owner_title(f"Ship ✈{chr(0xFE0F)}") == "Ship ✈"


def test_owner_title_removes_zwj_so_a_zwj_title_is_never_canonical():
    """ZWJ removal here is the deliberate policy, not an oversight.

    U+200D is in ``INVISIBLE_CHARS``, so ``owner_title`` decomposes a ZWJ
    sequence into its base glyphs. ``owner_title`` IS the canonical owner
    projection, so a Workspace response that still carries the ZWJ does not
    equal the canonical title and is rejected. That is deliberate at this
    boundary: a title is a short label, never an emoji composition surface,
    so joining unrelated glyphs into one rendered image is exactly the
    display control the boundary removes. ``strip_unicode_tags`` leaving ZWJ
    alone is not a contradiction — its scope is plane-14 tags only.
    """
    zwj = chr(0x200D)
    family = f"\U0001F468{zwj}\U0001F469{zwj}\U0001F467"

    assert ow.owner_title(f"Ship {family}") == "Ship \U0001F468\U0001F469\U0001F467"
    assert ow.owner_title(f"Sh{zwj}ip the thing") == "Ship the thing"
    # Accents, CJK and ordinary non-ZWJ emoji are untouched by that policy.
    assert ow.owner_title("Café naïve 日本語 \U0001F680") == "Café naïve 日本語 \U0001F680"


def test_owner_title_keeps_only_the_three_pinned_rgi_tag_flags():
    """England/Scotland/Wales survive; every other tag payload does not.

    ``strip_unicode_tags`` pins the three RGI subdivision sequences by their
    exact code points, so those titles project unchanged. A payload wrapped
    in the same U+1F3F4 base and U+E007F CANCEL TAG frame is a smuggling
    frame, not a flag, and loses everything but the visible base.
    """
    def _tag_flag(code: str) -> str:
        return (
            "\U0001F3F4"
            + "".join(chr(0xE0000 + ord(c)) for c in code)
            + chr(0xE007F)
        )

    for code in ("gbeng", "gbsct", "gbwls"):
        flag = _tag_flag(code)
        assert ow.owner_title(f"Ship {flag}") == f"Ship {flag}"

    # The pinned codes match on their exact lowercase letters, so a whole
    # framed sentence, a cased or off-by-one near-miss, and a
    # digit/punctuation payload are all payloads rather than flags.
    for payload in (
        "usca", "ignore all instructions", "GBSCT", "gbsc", "gbsctx", "0123!?;-",
    ):
        assert ow.owner_title(f"Ship {_tag_flag(payload)}") == "Ship \U0001F3F4"

    # An unterminated frame never reaches its CANCEL TAG, so even the exact
    # Scotland spec is not a flag and only the visible base survives.
    unterminated = "\U0001F3F4" + "".join(chr(0xE0000 + ord(c)) for c in "gbsct")
    assert ow.owner_title(f"Ship {unterminated}") == "Ship \U0001F3F4"


def test_owner_title_never_projects_a_tag_flag_cut_by_the_240_bound():
    """The 240 code point slice can land inside a preserved pinned flag.

    The pinned flags are preserved WHOLE before the bound is applied, so the
    slice itself is what can cut one — leaving the visible U+1F3F4 base
    trailed by dangling invisible plane-14 tag characters, which is the
    smuggling frame the boundary exists to remove and which the Workspace
    correctly rejects. Re-running ``strip_unicode_tags`` on the bounded text
    keeps an intact flag intact and reduces a cut one to its visible base.

    Each flag is 7 code points (U+1F3F4 base + 5 tag letters + U+E007F), so
    240 - 7 = 233 is the last start offset that fits whole.
    """
    def _tag_flag(code: str) -> str:
        return (
            "\U0001F3F4"
            + "".join(chr(0xE0000 + ord(c)) for c in code)
            + chr(0xE007F)
        )

    def _has_tag_char(text: str) -> bool:
        return any(0xE0000 <= ord(char) <= 0xE007F for char in text)

    for code in ("gbeng", "gbsct", "gbwls"):
        flag = _tag_flag(code)
        assert len(flag) == 7

        # Wholly before the bound: unchanged, flag and all.
        assert ow.owner_title("a" * 100 + flag) == "a" * 100 + flag
        # Ending exactly ON the bound: still whole, still preserved.
        assert ow.owner_title("a" * 233 + flag) == "a" * 233 + flag

        # Cut by the bound — the CANCEL TAG, then the tag letters, then the
        # base itself fall outside 240. Every one keeps the visible base (or
        # nothing) and never a dangling tag character.
        for start, expected in (
            (234, "a" * 234 + "\U0001F3F4"),   # loses U+E007F only
            (238, "a" * 238 + "\U0001F3F4"),   # reviewer counterexample
            (239, "a" * 239 + "\U0001F3F4"),   # base is the 240th code point
        ):
            projected = ow.owner_title("a" * start + flag)
            assert projected == expected
            assert not _has_tag_char(projected)
            assert len(projected) <= 240

        # Wholly after the bound: the flag never reaches the owner at all.
        assert ow.owner_title("a" * 240 + flag) == "a" * 240
        assert ow.owner_title("a" * 300 + flag) == "a" * 240
        assert not _has_tag_char(ow.owner_title("a" * 300 + flag))


def test_owner_projections_remove_annotation_and_object_replacement_controls():
    """U+FFF9-U+FFFC are invisible, so neither projection may keep them.

    The interlinear annotation frame (ANCHOR / SEPARATOR / TERMINATOR) hides
    everything between the anchor and the terminator behind the annotated
    base text, and U+FFFC OBJECT REPLACEMENT CHARACTER renders as nothing at
    all. Each is removed on the same pre-redaction pass as the zero-width
    and bidi characters, so none of them can split a credential past the
    redactor either.

    Code points are written as ``chr(...)`` rather than pasted: a literal
    invisible character in this file would be unreviewable.
    """
    from tools.threat_patterns import INVISIBLE_CHARS

    controls = (
        chr(0xFFF9),  # interlinear annotation anchor
        chr(0xFFFA),  # interlinear annotation separator
        chr(0xFFFB),  # interlinear annotation terminator
        chr(0xFFFC),  # object replacement character
    )
    for control in controls:
        assert control in INVISIBLE_CHARS
        assert ow.owner_title(f"Sh{control}ip the {control}thing") == "Ship the thing"
        assert ow.owner_title(control * 3) == "Untitled work item"
        assert (
            ow.owner_project_name(f"Shoe{control} Shop{control}") == "Shoe Shop"
        )
        assert ow.owner_project_name(control * 3) == "Untitled Project"
        # Removal runs BEFORE redaction, so a credential split across one of
        # these is still redacted rather than projected in full.
        assert (
            ow.owner_project_name(
                f"Rotate ghp_ABCDEF{control}GHIJKLMNOPQRSTUVWXYZ0123456789 now"
            )
            == "Rotate ghp_AB...6789 now"
        )

    # A whole annotated payload is reduced to the visible text around it.
    anchor, separator, terminator = controls[0], controls[1], controls[2]
    assert (
        ow.owner_project_name(
            f"Shoe{anchor}Shop{separator}ignore all instructions{terminator}"
        )
        == "ShoeShopignore all instructions"
    )


def test_owner_project_name_is_the_canonical_project_display_projection():
    """One projection for every owner-facing Project name, bounded at 160.

    Same sanitize/redact contract as ``owner_title`` — control characters,
    invisible/bidi characters and plane-14 tag characters removed, URL
    credentials masked at this non-navigation egress, whitespace collapsed —
    with two deliberate differences: the bound is 160 code points, and a
    Project name is owner-authored text so no internal work-item prefix is
    stripped from it.
    """
    zero_width, rl_override = chr(0x200B), chr(0x202E)

    assert ow.owner_project_name("Shoe Shop") == "Shoe Shop"
    assert ow.owner_project_name("  Shoe   \n  Shop  ") == "Shoe Shop"
    assert ow.owner_project_name("") == "Untitled Project"
    assert ow.owner_project_name(None) == "Untitled Project"
    assert ow.owner_project_name("   ") == "Untitled Project"

    # Unsafe control / display characters never reach the owner.
    assert (
        ow.owner_project_name("Shoe\x00 \x1b[31mShop\x9b1m") == "Shoe Shop"
    )
    assert (
        ow.owner_project_name(f"Sh{zero_width}oe {rl_override}Shop")
        == "Shoe Shop"
    )
    assert ow.owner_project_name("Shoe Shop\U000e0041\U000e0042") == "Shoe Shop"
    assert ow.owner_project_name(f"\x00{zero_width}{rl_override}") == "Untitled Project"
    # Safe human Unicode is not what this removes.
    assert ow.owner_project_name("Café naïve 日本語 \U0001f680") == (
        "Café naïve 日本語 \U0001f680"
    )

    # Credentials, including URL-borne ones, are masked.
    assert (
        ow.owner_project_name("Shop key sk-ABCDEFGHIJ rotated")
        == "Shop key *** rotated"
    )
    assert (
        ow.owner_project_name(
            "Shop https://deploy:hunter2verylongpassword@git.example.com/repo.git"
        )
        == "Shop https://***@git.example.com/repo.git"
    )
    assert (
        ow.owner_project_name(
            "Shop https://api.example.com/v1/sync?token=OPAQUESECRET123&page=2"
        )
        == "Shop https://api.example.com/v1/sync?token=***&page=2"
    )
    assert (
        ow.owner_project_name(
            "Shop https://bucket.s3.example.com/report.pdf"
            "?X-Amz-Expires=900&X-Amz-Signature=abcdef0123456789abcdef"
        )
        == "Shop https://bucket.s3.example.com/report.pdf"
        "?X-Amz-Expires=900&X-Amz-Signature=***"
    )
    # A public parameter that merely resembles a credential name survives.
    assert (
        ow.owner_project_name("Shop https://api.example.com/jobs?token_count=17")
        == "Shop https://api.example.com/jobs?token_count=17"
    )

    # An internal-looking prefix is part of an owner-authored Project name.
    assert ow.owner_project_name("B03 — Shoe Shop") == "B03 — Shoe Shop"
    assert ow.owner_title("B03 — Shoe Shop") == "Shoe Shop"


def test_owner_project_name_bounds_at_160_unicode_code_points():
    """The bound counts code points and is applied to sanitized text."""
    zero_width = chr(0x200B)

    assert ow.owner_project_name("a" * 159) == "a" * 159
    assert ow.owner_project_name("a" * 160) == "a" * 160
    assert ow.owner_project_name("a" * 161) == "a" * 160
    # Astral characters are one code point each, not one UTF-16 pair.
    assert ow.owner_project_name("\U0001f680" * 160) == "\U0001f680" * 160
    assert ow.owner_project_name("\U0001f680" * 161) == "\U0001f680" * 160
    # Sanitizing first means invisible padding cannot shorten the name.
    assert ow.owner_project_name(f"a{zero_width}" * 200) == "a" * 160
    # A work-item title keeps its own, wider bound.
    assert ow.owner_title("a" * 200) == "a" * 200


@pytest.mark.parametrize(
    ("projector", "limit"),
    [
        (ow.owner_title, 240),
        (ow.owner_project_name, 160),
    ],
)
def test_owner_projection_is_idempotent_at_its_bound(projector, limit):
    """``projector(projector(x)) == projector(x)`` — the bound included.

    These projections ARE the canonical owner text: every surface that shows
    a name or title projects it, and the Workspace compares what it receives
    against its own projection of the same value. Anything that changes on a
    second pass therefore reads as a mismatch and is rejected.

    The bound is where that used to break. It is a raw code point slice, so it
    can cut mid-word and strand the whitespace in front of the cut at the end
    of the string — which the second pass's whitespace collapse would then
    remove. It can also cut a preserved RGI tag flag, which the trailing
    ``strip_unicode_tags`` reduces to its visible base. Both tails have to
    settle in one pass.
    """
    def _tag_flag(code: str) -> str:
        return (
            "\U0001F3F4"
            + "".join(chr(0xE0000 + ord(c)) for c in code)
            + chr(0xE007F)
        )

    vectors = [
        # Nothing to do.
        "Shoe Shop",
        "Café naïve 日本語 \U0001F680",
        # The empty-input fallbacks are themselves projections.
        "",
        "   ",
        # Cut exactly at, one before, and one after the bound.
        "a" * (limit - 1),
        "a" * limit,
        "a" * (limit + 1),
        # The cut lands on a space — the case the final strip exists for.
        "a" * (limit - 1) + " bcd",
        "a " * limit,
        # Astral characters are one code point each.
        "\U0001F680" * (limit + 5),
        # A preserved flag cut by the bound, at every offset that cuts it.
        *[
            "a" * (limit - offset) + _tag_flag(code)
            for code in ("gbeng", "gbsct", "gbwls")
            for offset in (7, 6, 2, 1, 0)
        ],
        # Already-redacted output must not be re-redacted into something else.
        "Rotate ghp_ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 before Friday",
        "Key sk-ABCDEFGHIJ rotated",
        "Mirror https://deploy:hunter2verylongpassword@git.example.com/repo.git",
        "Sync https://api.example.com/v1/sync?token=OPAQUEVALUE123&page=2",
        # Sanitized-away input, at and past the bound.
        f"a{chr(0x200B)}" * (limit + 40),
        f"Ship{chr(0xD800)} the {chr(0x00AD)}thing",
    ]
    for value in vectors:
        once = projector(value)
        assert projector(once) == once
        assert len(once) <= limit


def test_owner_run_projection_uses_only_native_runtime_and_cost_receipt():
    run = kanban_db.Run(
        id=1,
        task_id="task-1",
        profile="raphael-verifier",
        step_key=None,
        status="done",
        claim_lock=None,
        claim_expires=None,
        worker_pid=None,
        max_runtime_seconds=None,
        last_heartbeat_at=None,
        started_at=1_700_000_000,
        ended_at=1_700_000_100,
        outcome="completed",
        summary="ignored raw summary",
        metadata={
            "runtime_receipt": {
                "schema_version": 2,
                "engine": "hermes",
                "profile": "raphael-verifier",
                "provider": "openai-codex",
                "model": "gpt-5.6-sol",
                "reasoning_effort": "max",
                "route_evidence": "dominant-session-usage",
                "cost": {
                    "state": "estimated",
                    "currency": "USD",
                    "amount": 0.0123,
                    "source": "official_docs_snapshot",
                    "scope": "dominant-main-route",
                },
            }
        },
        error=None,
    )

    projection = ow._owner_project_run_projection(
        run,
        "B03 — Ship the thing",
        task_pin=None,
        has_newer_run=False,
        run_context=True,
    )
    assert projection["task_title"] == "Ship the thing"
    assert projection["has_newer_run"] is False
    receipt = projection["receipt"]

    assert receipt["runtime"] == {
        "state": "known",
        "engine": "hermes",
        "profile": "raphael-verifier",
        "provider": "openai-codex",
        "model": "gpt-5.6-sol",
        "reasoning_effort": "max",
    }
    assert receipt["cost"] == {
        "state": "estimated",
        "currency": "USD",
        "amount": 0.0123,
        "summary": "Estimated model usage for this recorded route.",
    }

    run.metadata["runtime_receipt"]["model"] = "unadmitted-model"
    rejected = ow._owner_project_run_projection(
        run,
        "B03 — Ship the thing",
        task_pin=None,
        has_newer_run=True,
        run_context=True,
    )["receipt"]
    assert rejected["runtime"] == ow._OWNER_UNKNOWN_RUNTIME
    assert rejected["cost"] == ow._OWNER_UNKNOWN_COST

    # Without the capability the same run projects to exactly the three keys
    # the first owner Workspace release accepts, so a Hermes-first rollout
    # cannot make a snapshot unreadable.
    assert set(ow._owner_project_run_projection(
        run,
        "B03 — Ship the thing",
        task_pin=None,
        has_newer_run=True,
        run_context=False,
    )) == {"started_at", "finished_at", "receipt"}


def test_run_receipt_is_checked_against_that_task_own_pinned_route(ctx):
    """An admitted route is not enough: it must be THIS task's pinned route."""
    run = kanban_db.Run(
        id=7,
        task_id="task-1",
        profile="raphael-verifier",
        step_key=None,
        status="done",
        claim_lock=None,
        claim_expires=None,
        worker_pid=None,
        max_runtime_seconds=None,
        last_heartbeat_at=None,
        started_at=1_700_000_000,
        ended_at=1_700_000_100,
        outcome="completed",
        summary="ignored raw summary",
        metadata={
            "runtime_receipt": {
                "schema_version": 2,
                "engine": "hermes",
                "profile": "raphael-verifier",
                "provider": "openai-codex",
                "model": "gpt-5.6-sol",
                "reasoning_effort": "max",
                "route_evidence": "session-row",
            }
        },
        error=None,
    )

    def runtime(pin):
        return ow._owner_project_run_receipt(run, pin)["runtime"]

    def pin(profile, provider, model, effort, *, valid=True):
        return ow.OwnerTaskRoutePin(
            valid=valid,
            profile=profile,
            provider=provider,
            model=model,
            reasoning_effort=effort,
        )

    assert runtime(
        pin("raphael-verifier", "openai-codex", "gpt-5.6-sol", "max")
    )["state"] == "known"

    # Independent verification exists to be independent OF the implementation
    # family, so there is no admitted Claude verifier route at all: the
    # role-level check itself refuses it, and the pinned check refuses the run.
    with pytest.raises(ValueError):
        ow.validate_raphael_model_assignment(
            "raphael-verifier", "anthropic", "claude-opus-5", "max",
            disable_fallbacks=True,
        )
    assert runtime(
        pin("raphael-verifier", "anthropic", "claude-opus-5", "max")
    ) == ow._OWNER_UNKNOWN_RUNTIME
    # Same model and provider, different approved depth.
    assert runtime(
        pin("raphael-verifier", "openai-codex", "gpt-5.6-sol", "high")
    ) == ow._OWNER_UNKNOWN_RUNTIME
    # Same route, different role: the pin binds the assignee too. ``default``
    # is independently admitted for this exact provider/model/effort, so only
    # the role binding can be what makes this unconfirmed.
    assert ow.validate_raphael_model_assignment(
        "default", "openai-codex", "gpt-5.6-sol", "max", disable_fallbacks=True,
    ).model == "gpt-5.6-sol"
    assert runtime(
        pin("default", "openai-codex", "gpt-5.6-sol", "max")
    ) == ow._OWNER_UNKNOWN_RUNTIME
    # An INVALID lock proves nothing — it must never fall back to the looser
    # role-level check that an unlocked task legitimately uses.
    assert runtime(
        pin("raphael-verifier", "openai-codex", "gpt-5.6-sol", "max", valid=False)
    ) == ow._OWNER_UNKNOWN_RUNTIME

    # The core claim on a role that really does have two admitted providers:
    # the run's route passes the role-level check, and is still unconfirmed
    # because it is not the route THIS task was pinned to.
    run.profile = "raphael-planner"
    run.metadata["runtime_receipt"]["profile"] = "raphael-planner"
    assert ow.validate_raphael_model_assignment(
        "raphael-planner", "anthropic", "claude-sonnet-5", "max",
        disable_fallbacks=True,
    ).model == "claude-sonnet-5"
    assert runtime(
        pin("raphael-planner", "openai-codex", "gpt-5.6-sol", "max")
    )["state"] == "known"
    assert runtime(
        pin("raphael-planner", "anthropic", "claude-sonnet-5", "max")
    ) == ow._OWNER_UNKNOWN_RUNTIME


def test_task_route_pin_distinguishes_unlocked_from_invalid():
    locked = {
        "assignee": "raphael-verifier",
        "provider_override": "openai-codex",
        "model_override": "gpt-5.6-sol",
        "reasoning_effort": "MAX",
        "execution_tier": "routine",
        "model_policy_lock": kanban_db.mint_policy_lock(
            "raphael-verifier", "openai-codex", "gpt-5.6-sol", "max", "routine",
        ),
    }
    assert ow.owner_task_route_pin(locked) == ow.OwnerTaskRoutePin(
        valid=True,
        profile="raphael-verifier",
        provider="openai-codex",
        model="gpt-5.6-sol",
        reasoning_effort="max",
    )
    # An unlocked (manual / pre-lock) task has no pin and keeps the older
    # role-level check.
    assert ow.owner_task_route_pin({**locked, "model_policy_lock": None}) is None
    # A row projected without the lock column at all cannot carry a lock.
    assert ow.owner_task_route_pin({"title": "manual card"}) is None
    # Every other break is INVALID, never "unlocked": an incomplete route, a
    # hand-edited component, an unknown authority, a stale version, and the
    # old truthy marker all report a pin that cannot confirm anything.
    for broken in (
        {"provider_override": ""},
        {"model_override": "gpt-5.6-terra"},
        {"execution_tier": None},
        # ``default`` is independently admitted for this exact route, so the
        # route still resolves and only the role-bound digest catches the edit.
        {"assignee": "default"},
        {"model_policy_lock": "bogus:v1:" + "a" * 64},
        {"model_policy_lock": "raphael:v99:" + "a" * 64},
        {"model_policy_lock": "raphael"},
    ):
        pin = ow.owner_task_route_pin({**locked, **broken})
        assert pin is not None and pin.valid is False, broken


def test_project_snapshot_run_projection_has_sanitized_task_title(ctx):
    args = _task_graph_args(
        idempotency_key="graph-run-title",
        project_name="Run Title Project",
    )
    approver = _with_approver(ctx.session)
    result = _commit_task_graph(ctx, **args)
    approver.join()

    now = int(time.time())
    with kanban_db.connect(board=result["board"]) as conn:
        task_id = kanban_db.create_task(
            conn,
            title="B03 — Ship the thing",
            assignee="default",
            project_id=result["project_id"],
        )
        with kanban_db.write_txn(conn):
            conn.execute(
                "INSERT INTO task_runs "
                "(task_id, profile, status, started_at, ended_at, outcome) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (task_id, "default", "done", now - 60, now, "completed"),
            )

    snapshot = ow.read_project_snapshot(
        ctx, result["project_slug"], run_context=True,
    )

    assert len(snapshot["runs"]) == 1
    run = snapshot["runs"][0]
    assert set(run) == {
        "task_title", "started_at", "finished_at", "has_newer_run", "receipt",
    }
    assert run["task_title"] == "Ship the thing"
    assert run["has_newer_run"] is False

    # The task id is legitimately present elsewhere in the snapshot (the
    # board task list); only the run projection/receipt must not leak it.
    run_payload = json.dumps(run)
    assert task_id not in run_payload

    # Same Project, same runs, no capability asked for: the default read is
    # exactly the shape a Workspace release that predates these keys accepts,
    # and it still carries the sanitized run facts that shape allows.
    default_read = ow.read_project_snapshot(ctx, result["project_slug"])
    assert len(default_read["runs"]) == 1
    assert set(default_read["runs"][0]) == {"started_at", "finished_at", "receipt"}
    assert default_read["runs"][0]["receipt"] == run["receipt"]
    assert default_read["runs"][0]["started_at"] == run["started_at"]
    # Every other key of the snapshot is unaffected by the capability, so the
    # two reads differ only in the run shape. The steward block is excluded
    # because it stamps its own read time.
    skipped = {"runs", "steward"}
    assert {k: v for k, v in default_read.items() if k not in skipped} == {
        k: v for k, v in snapshot.items() if k not in skipped
    }


def test_project_snapshot_run_retry_is_decided_per_exact_task_id(ctx):
    """Two tasks whose titles sanitize identically must not read as retries.

    Only a second run of the SAME exact task id is an older attempt at the
    same work; the run projection carries that as a boolean because the id
    itself never crosses the owner boundary.
    """
    args = _task_graph_args(
        idempotency_key="graph-run-retry",
        project_name="Run Retry Project",
    )
    approver = _with_approver(ctx.session)
    result = _commit_task_graph(ctx, **args)
    approver.join()

    now = int(time.time())
    with kanban_db.connect(board=result["board"]) as conn:
        retried_id = kanban_db.create_task(
            conn,
            title="B03 — Ship the thing",
            assignee="default",
            project_id=result["project_id"],
        )
        namesake_id = kanban_db.create_task(
            conn,
            title="B04 — Ship the thing",
            assignee="default",
            project_id=result["project_id"],
        )
        with kanban_db.write_txn(conn):
            for task_id, started, ended in (
                (retried_id, now - 300, now - 240),
                (retried_id, now - 120, now - 60),
                (namesake_id, now - 30, now),
            ):
                conn.execute(
                    "INSERT INTO task_runs "
                    "(task_id, profile, status, started_at, ended_at, outcome) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (task_id, "default", "done", started, ended, "completed"),
                )

    runs = ow.read_project_snapshot(
        ctx, result["project_slug"], run_context=True,
    )["runs"]

    assert all(
        set(run) == {
            "task_title", "started_at", "finished_at", "has_newer_run", "receipt",
        }
        for run in runs
    )
    # All three runs sanitize to one display title, so the title alone could
    # never separate them.
    assert [run["task_title"] for run in runs] == ["Ship the thing"] * 3
    # Newest first: the namesake task's only run, then the retried task's
    # newest run, then its older attempt — only the last has a newer run.
    assert [run["has_newer_run"] for run in runs] == [False, False, True]

    payload = json.dumps(runs)
    assert retried_id not in payload
    assert namesake_id not in payload

    # A reader that did not ask for the capability gets no retry fact at all,
    # rather than a fact it would reject or misread: the three runs keep the
    # legacy shape and stay in the same newest-first order.
    default_runs = ow.read_project_snapshot(ctx, result["project_slug"])["runs"]
    assert [set(run) for run in default_runs] == [
        {"started_at", "finished_at", "receipt"}
    ] * 3
    assert [run["started_at"] for run in default_runs] == [
        run["started_at"] for run in runs
    ]


def test_project_snapshot_hides_projects_without_owner_receipt(ctx):
    with projects_db.connect_closing() as conn:
        project_id = projects_db.create_project(conn, name="Foreign Snapshot")
        project = projects_db.get_project(conn, project_id)
        assert project is not None
        kanban_db.create_board(project.slug, name=project.name, project_id=project_id)
        projects_db.update_project(conn, project_id, board_slug=project.slug)

    with pytest.raises(ow.OwnerWorkspaceError) as excinfo:
        ow.read_project_snapshot(ctx, project.slug)

    assert excinfo.value.code == "project_not_found"


def test_project_attachment_is_exact_receipt_bound_and_bounded(ctx):
    approver = _with_approver(ctx.session)
    result = _commit_task_graph(
        ctx,
        **_task_graph_args(
            idempotency_key="graph-attachment",
            project_name="Attachment Project",
        ),
    )
    approver.join()
    with kanban_db.connect(board=result["board"]) as conn:
        attachment_id = kanban_db.store_attachment_bytes(
            conn,
            result["task_ids"][0],
            "../owner\"note.txt",
            b"safe owner bytes",
            content_type="text/plain\r\nX-Evil: yes",
            board=result["board"],
        )

    attachment = ow.read_project_attachment(
        ctx, result["project_slug"], str(attachment_id)
    )

    assert attachment == {
        "id": str(attachment_id),
        "filename": "owner_note.txt",
        "media_type": "application/octet-stream",
        "size": 16,
        "created_at": attachment["created_at"],
        "body": b"safe owner bytes",
    }

    with pytest.raises(ow.OwnerWorkspaceError) as excinfo:
        ow.read_project_attachment(ctx, result["project_slug"], "../1")
    assert excinfo.value.code == "attachment_not_found"


# ---------------------------------------------------------------------------
# Item 32TK: an absent store is empty; a broken one is a service failure
# ---------------------------------------------------------------------------


def test_no_projects_db_at_all_projects_as_genuinely_empty(ctx):
    projects_db.projects_db_path().unlink(missing_ok=True)
    assert ow.list_committed_projects(ctx) == []


def test_a_projects_db_with_no_owner_receipts_table_is_genuinely_empty(ctx):
    """The receipt table is created by the FIRST owner operation, so its
    absence really is "no owner authority was ever written here"."""
    with projects_db.connect_closing() as conn:
        projects_db.create_project(conn, name="Not owner work")
    with contextlib.closing(
        sqlite3.connect(projects_db.projects_db_path())
    ) as raw:
        raw.execute("DROP TABLE IF EXISTS owner_workspace_receipts")
        raw.commit()

    assert ow.list_committed_projects(ctx) == []


def _committed_project(ctx, *, key: str, name: str) -> dict:
    approver = _with_approver(ctx.session)
    result = _commit_task_graph(
        ctx, **_task_graph_args(idempotency_key=key, project_name=name),
    )
    approver.join()
    assert result["ok"] is True
    return result


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("project_id", ""),
        ("project_id", None),
        ("project_id", 12),
        ("ok", False),
        ("ok", "yes"),
    ],
)
def test_a_committed_receipt_missing_its_identifier_is_never_an_empty_list(
    ctx, field, value,
):
    """Skipping such a row let corrupt authority become an authoritative
    "you have no Projects"."""
    result = _committed_project(
        ctx, key="graph-receipt-identifier", name="Receipt Identifier",
    )
    assert [item["slug"] for item in ow.list_committed_projects(ctx)] == [
        result["project_slug"]
    ]
    path = projects_db.projects_db_path()
    with contextlib.closing(sqlite3.connect(path)) as raw:
        raw.row_factory = sqlite3.Row
        row = raw.execute(
            "SELECT idempotency_key, result_json FROM owner_workspace_receipts "
            "WHERE status = 'committed'"
        ).fetchone()
        stored = json.loads(row["result_json"])
        if value is None:
            stored.pop(field, None)
        else:
            stored[field] = value
        raw.execute(
            "UPDATE owner_workspace_receipts SET result_json = ? "
            "WHERE idempotency_key = ?",
            (json.dumps(stored), row["idempotency_key"]),
        )
        raw.commit()

    with pytest.raises(ow.OwnerWorkspaceError) as excinfo:
        ow.list_committed_projects(ctx)
    assert excinfo.value.code == "snapshot_unavailable"


def test_a_malformed_ownership_receipt_is_unavailable_not_not_owned(ctx):
    """Swallowing it and answering False turned "ownership cannot be read"
    into "you do not own this Project" — a 404 before a mutation."""
    result = _committed_project(
        ctx, key="graph-ownership-unreadable", name="Ownership Unreadable",
    )
    path = projects_db.projects_db_path()
    with contextlib.closing(sqlite3.connect(path)) as raw:
        raw.execute(
            "UPDATE owner_workspace_receipts SET result_json = '{not json' "
            "WHERE status = 'committed'"
        )
        raw.commit()

    with contextlib.closing(projects_db.connect()) as conn:
        ow._ensure_schema(conn)
        with pytest.raises(ow.OwnerWorkspaceError) as excinfo:
            ow._receipt_owns_project(conn, ctx, result["project_id"])
    assert excinfo.value.code == "snapshot_unavailable"


def test_a_native_run_receipt_is_absent_committed_or_unreadable(ctx):
    """Collapsing "unreadable" into "absent" put a committed external effect
    back on the retry path."""
    result = _committed_project(
        ctx, key="graph-native-receipt", name="Native Receipt",
    )
    with contextlib.closing(projects_db.connect()) as conn:
        row = conn.execute(
            "SELECT idempotency_key, operation, authority_digest FROM "
            "owner_workspace_receipts WHERE status = 'committed'"
        ).fetchone()
    digest = row["authority_digest"] or "a" * 64

    # Absent: nothing matching this authority ever committed.
    assert ow.read_committed_owner_run_receipt(
        profile=ctx.profile,
        idempotency_key="never-used",
        operation=row["operation"],
        authority_digest=digest,
    ) is None

    if row["authority_digest"]:
        # Committed: the exact receipt comes back.
        recovered = ow.read_committed_owner_run_receipt(
            profile=ctx.profile,
            idempotency_key=row["idempotency_key"],
            operation=row["operation"],
            authority_digest=digest,
        )
        assert recovered["project_id"] == result["project_id"]

    # Unreadable: a committed row whose result cannot be read is its own
    # answer, never "absent".
    path = projects_db.projects_db_path()
    with contextlib.closing(sqlite3.connect(path)) as raw:
        raw.execute(
            "UPDATE owner_workspace_receipts SET result_json = '{not json', "
            "authority_digest = ? WHERE idempotency_key = ?",
            (digest, row["idempotency_key"]),
        )
        raw.commit()
    with pytest.raises(ow.OwnerReceiptUnreadable):
        ow.read_committed_owner_run_receipt(
            profile=ctx.profile,
            idempotency_key=row["idempotency_key"],
            operation=row["operation"],
            authority_digest=digest,
        )


@pytest.mark.parametrize(
    "break_it",
    [
        "unreadable_db",
        "missing_projects_table",
        "malformed_receipt",
        "missing_project_row",
        "foreign_board_binding",
    ],
)
def test_a_broken_projects_store_is_never_projected_as_empty(ctx, break_it):
    """An authority outage must not read as "you have no Projects"."""
    result = _committed_project(ctx, key="graph-broken-store", name="Broken Store")
    assert [item["slug"] for item in ow.list_committed_projects(ctx)] == [
        result["project_slug"]
    ]

    path = projects_db.projects_db_path()
    if break_it == "unreadable_db":
        path.write_bytes(b"this is not a sqlite database")
    elif break_it == "missing_projects_table":
        with contextlib.closing(sqlite3.connect(path)) as raw:
            raw.execute("DROP TABLE projects")
            raw.commit()
    elif break_it == "malformed_receipt":
        with contextlib.closing(sqlite3.connect(path)) as raw:
            raw.execute(
                "UPDATE owner_workspace_receipts SET result_json = '{not json' "
                "WHERE status = 'committed'"
            )
            raw.commit()
    elif break_it == "missing_project_row":
        with contextlib.closing(sqlite3.connect(path)) as raw:
            raw.execute("DELETE FROM projects")
            raw.commit()
    else:
        kanban_db.write_board_metadata(
            result["board"], project_id="proj_someone_else",
        )

    with pytest.raises(ow.OwnerWorkspaceError) as excinfo:
        ow.list_committed_projects(ctx)
    assert excinfo.value.code == "snapshot_unavailable"


@pytest.mark.parametrize(
    "break_it", ["missing_file", "short_file", "trailing_bytes", "unreadable"],
)
def test_an_unreadable_attachment_is_not_reported_as_missing(ctx, break_it):
    """404 is for a missing authority row. A storage or integrity failure is
    not proof the attachment is not there."""
    result = _committed_project(
        ctx, key="graph-attachment-broken", name="Attachment Faults",
    )
    with kanban_db.connect(board=result["board"]) as conn:
        attachment_id = kanban_db.store_attachment_bytes(
            conn, result["task_ids"][0], "note.txt", b"owner bytes",
            content_type="text/plain", board=result["board"],
        )
        stored = Path(
            conn.execute(
                "SELECT stored_path FROM task_attachments WHERE id = ?",
                (attachment_id,),
            ).fetchone()["stored_path"]
        )

    if break_it == "missing_file":
        stored.unlink()
    elif break_it == "short_file":
        stored.write_bytes(b"owner")
    elif break_it == "trailing_bytes":
        stored.write_bytes(b"owner bytes and more")
    else:
        stored.chmod(0o000)

    try:
        with pytest.raises(ow.OwnerWorkspaceError) as excinfo:
            ow.read_project_attachment(
                ctx, result["project_slug"], str(attachment_id),
            )
    finally:
        if break_it == "unreadable":
            stored.chmod(0o600)
    assert excinfo.value.code == "snapshot_unavailable"


def test_an_attachment_id_with_no_authority_row_is_still_not_found(ctx):
    """The adjacent success path for the same boundary: proven absence."""
    result = _committed_project(
        ctx, key="graph-attachment-absent", name="Attachment Absent",
    )
    with pytest.raises(ow.OwnerWorkspaceError) as excinfo:
        ow.read_project_attachment(ctx, result["project_slug"], "987654")
    assert excinfo.value.code == "attachment_not_found"


def test_a_truncated_decision_inbox_says_so(ctx):
    """An owner who cannot tell a full inbox from a clipped one can believe
    they have answered everything Raphael is waiting on."""
    setup = _bootstrap_board(ctx)
    with kanban_db.connect(board=setup["board"]) as conn:
        for index in range(ow._OWNER_DECISIONS_LIMIT + 3):
            task_id = kanban_db.create_task(
                conn, title=f"Needs an answer {index}", assignee="default",
                project_id=setup["project_id"], board=setup["board"],
            )
            with kanban_db.write_txn(conn):
                conn.execute(
                    "UPDATE tasks SET status = 'blocked', "
                    "block_kind = 'needs_input' WHERE id = ?",
                    (task_id,),
                )

    projected = ow.list_owner_decisions(ctx)

    assert projected["truncated"] is True
    assert len(projected["data"]) == ow._OWNER_DECISIONS_LIMIT


def test_a_complete_decision_inbox_says_it_is_complete(ctx):
    setup = _bootstrap_board(ctx)
    with kanban_db.connect(board=setup["board"]) as conn:
        task_id = kanban_db.create_task(
            conn, title="Needs an answer", assignee="default",
            project_id=setup["project_id"], board=setup["board"],
        )
        with kanban_db.write_txn(conn):
            conn.execute(
                "UPDATE tasks SET status = 'blocked', "
                "block_kind = 'needs_input' WHERE id = ?",
                (task_id,),
            )

    projected = ow.list_owner_decisions(ctx)

    assert projected["truncated"] is False
    assert len(projected["data"]) == 1


def test_project_lifecycle_archives_and_restores_receipt_backed_project(ctx):
    args = _task_graph_args(
        idempotency_key="graph-lifecycle",
        project_name="Lifecycle Project",
    )
    approver = _with_approver(ctx.session)
    created = _commit_task_graph(ctx, **args)
    approver.join()
    assert kanban_db.board_dispatch_allowed(
        kanban_db.read_board_metadata(created["board"])
    ) is True

    archive_approver = _with_approver(ctx.session)
    archived = _set_project_archived(
        ctx,
        idempotency_key="project-lifecycle-archive",
        project_id=created["project_id"],
        expected_revision=0,
        action="archive",
    )
    archive_approver.join()
    assert archived == {
        "ok": True,
        "action": "archive",
        "project_slug": created["project_slug"],
        "archived": True,
        "execution_paused": True,
    }
    assert ow.list_committed_projects(ctx)[0]["archived"] is True
    assert ow.list_committed_projects(
        ctx, lifecycle_revision=True,
    )[0]["lifecycle_revision"] == 1

    approval.unregister_gateway_notify(ctx.session)
    assert _set_project_archived(
        ctx,
        idempotency_key="project-lifecycle-archive",
        project_id=created["project_id"],
        expected_revision=0,
        action="archive",
    ) == archived

    restore_approver = _with_approver(ctx.session)
    restored = _set_project_archived(
        ctx,
        idempotency_key="project-lifecycle-restore",
        project_id=created["project_id"],
        expected_revision=1,
        action="restore",
    )
    restore_approver.join()
    assert restored == {
        "ok": True,
        "action": "restore",
        "project_slug": created["project_slug"],
        "archived": False,
        "execution_paused": True,
    }
    assert ow.list_committed_projects(ctx)[0]["archived"] is False
    assert ow.list_committed_projects(
        ctx, lifecycle_revision=True,
    )[0]["lifecycle_revision"] == 2
    restored_meta = kanban_db.read_board_metadata(created["board"])
    assert restored_meta["dispatch_enabled"] is False
    assert restored_meta["dispatch_paused_by_owner"] is True

    resume_approver = _with_approver(ctx.session)
    resumed = _set_project_archived(
        ctx,
        idempotency_key="project-lifecycle-resume",
        project_id=created["project_id"],
        expected_revision=2,
        action="resume",
    )
    resume_approver.join()
    assert resumed["execution_paused"] is False
    assert kanban_db.board_dispatch_allowed(
        kanban_db.read_board_metadata(created["board"])
    ) is True

    pause_approver = _with_approver(ctx.session)
    paused = _set_project_archived(
        ctx,
        idempotency_key="project-lifecycle-pause",
        project_id=created["project_id"],
        expected_revision=3,
        action="pause",
    )
    pause_approver.join()
    assert paused["execution_paused"] is True
    assert kanban_db.board_dispatch_allowed(
        kanban_db.read_board_metadata(created["board"])
    ) is False
    assert ow.list_committed_projects(
        ctx, lifecycle_revision=True,
    )[0]["lifecycle_revision"] == 4

    # Returning to the same action after an intervening restore is a new
    # lifecycle generation, never a replay of the first archive receipt.
    second_archive_approver = _with_approver(ctx.session)
    second_archive = _set_project_archived(
        ctx,
        idempotency_key="project-lifecycle-archive-again",
        project_id=created["project_id"],
        expected_revision=4,
        action="archive",
    )
    second_archive_approver.join()
    assert second_archive["ok"] is True
    assert second_archive["archived"] is True
    assert ow.list_committed_projects(
        ctx, lifecycle_revision=True,
    )[0]["lifecycle_revision"] == 5


def test_project_lifecycle_projection_reads_pre_migration_receipts(ctx):
    approver = _with_approver(ctx.session)
    created = _commit_task_graph(
        ctx,
        **_task_graph_args(
            idempotency_key="graph-lifecycle-legacy-schema",
            project_name="Legacy Lifecycle Project",
        ),
    )
    approver.join()

    with projects_db.connect_closing() as conn:
        conn.execute(
            "ALTER TABLE owner_workspace_receipts "
            "RENAME TO owner_workspace_receipts_current"
        )
        conn.execute(
            """CREATE TABLE owner_workspace_receipts (
                actor            TEXT NOT NULL,
                profile          TEXT NOT NULL,
                idempotency_key  TEXT NOT NULL,
                operation        TEXT NOT NULL,
                request_digest   TEXT NOT NULL,
                status           TEXT NOT NULL,
                lock_token       TEXT,
                lock_expires     INTEGER,
                project_id       TEXT,
                board_slug       TEXT,
                task_id          TEXT,
                result_json      TEXT,
                created_at       INTEGER NOT NULL,
                updated_at       INTEGER NOT NULL,
                PRIMARY KEY (actor, profile, idempotency_key)
            )"""
        )
        conn.execute(
            "INSERT INTO owner_workspace_receipts "
            "(actor, profile, idempotency_key, operation, request_digest, "
            "status, lock_token, lock_expires, project_id, board_slug, task_id, "
            "result_json, created_at, updated_at) "
            "SELECT actor, profile, idempotency_key, operation, request_digest, "
            "status, lock_token, lock_expires, project_id, board_slug, task_id, "
            "result_json, created_at, updated_at "
            "FROM owner_workspace_receipts_current"
        )
        conn.execute(
            "INSERT INTO owner_workspace_receipts "
            "(actor, profile, idempotency_key, operation, request_digest, "
            "status, result_json, created_at, updated_at) "
            "VALUES (?, ?, ?, 'owner_project_lifecycle', ?, 'committed', ?, ?, ?)",
            (
                ctx.actor,
                ctx.profile,
                "legacy-lifecycle-receipt",
                ow._digest({
                    "project_id": created["project_id"],
                    "action": "archive",
                }),
                json.dumps({"ok": True, "action": "archive"}),
                1,
                1,
            ),
        )
        conn.execute("DROP TABLE owner_workspace_receipts_current")
        conn.commit()

    project = ow.list_committed_projects(ctx, lifecycle_revision=True)[0]
    assert project["project_id"] == created["project_id"]
    assert project["lifecycle_revision"] == 1
    with projects_db.connect_closing() as conn:
        columns = {
            str(row[1])
            for row in conn.execute(
                "PRAGMA table_info(owner_workspace_receipts)"
            )
        }
    assert "terminal_generation" not in columns

    with projects_db.connect_closing() as conn:
        ow._ensure_schema(conn)

    project = ow.list_committed_projects(ctx, lifecycle_revision=True)[0]
    assert project["lifecycle_revision"] == 1
    with projects_db.connect_closing() as conn:
        columns = {
            str(row[1])
            for row in conn.execute(
                "PRAGMA table_info(owner_workspace_receipts)"
            )
        }
    assert "terminal_generation" in columns


def test_project_lifecycle_requires_exact_authenticated_run_authority(ctx):
    approver = _with_approver(ctx.session)
    created = _commit_task_graph(
        ctx,
        **_task_graph_args(
            idempotency_key="graph-lifecycle-authority",
            project_name="Lifecycle authority Project",
        ),
    )
    approver.join()

    with pytest.raises(ow.OwnerWorkspaceError) as excinfo:
        _RAW_SET_PROJECT_LIFECYCLE(
            ctx,
            idempotency_key="project-lifecycle-without-authority",
            project_id=created["project_id"],
            expected_revision=0,
            action="archive",
        )

    assert excinfo.value.code == "owner_run_authority_required"
    project = ow.list_committed_projects(ctx, lifecycle_revision=True)[0]
    assert project["archived"] is False
    assert project["lifecycle_revision"] == 0


def test_project_lifecycle_rejects_a_stale_owner_revision_before_approval(ctx):
    approver = _with_approver(ctx.session)
    created = _commit_task_graph(
        ctx,
        **_task_graph_args(
            idempotency_key="graph-lifecycle-stale",
            project_name="Stale Lifecycle Project",
        ),
    )
    approver.join()

    approver = _with_approver(ctx.session)
    archived = _set_project_archived(
        ctx,
        idempotency_key="project-lifecycle-stale-archive",
        project_id=created["project_id"],
        expected_revision=0,
        action="archive",
    )
    approver.join()
    assert archived["ok"] is True

    approval.unregister_gateway_notify(ctx.session)
    with pytest.raises(ow.OwnerWorkspaceError) as excinfo:
        _set_project_archived(
            ctx,
            idempotency_key="project-lifecycle-stale-restore",
            project_id=created["project_id"],
            expected_revision=0,
            action="restore",
        )

    assert excinfo.value.code == "stale_revision"
    assert ow.list_committed_projects(ctx)[0]["archived"] is True
    assert ow.list_committed_projects(
        ctx, lifecycle_revision=True,
    )[0]["lifecycle_revision"] == 1


def test_project_lifecycle_timeout_does_not_consume_the_retry_revision(ctx):
    approver = _with_approver(ctx.session)
    created = _commit_task_graph(
        ctx,
        **_task_graph_args(
            idempotency_key="graph-lifecycle-timeout",
            project_name="Lifecycle Timeout Project",
        ),
    )
    approver.join()
    args = {
        "idempotency_key": "project-lifecycle-timeout-archive",
        "project_id": created["project_id"],
        "expected_revision": 0,
        "action": "archive",
    }

    with _temporarily_patch(
        ow,
        "_confirm",
        lambda *args, **kwargs: {"approved": False, "reason": "timeout"},
    ):
        timed_out = _set_project_archived(ctx, **args)

    assert timed_out == {
        "ok": False,
        "error": "confirmation_denied",
        "reason": "timeout",
    }
    assert ow.list_committed_projects(
        ctx, lifecycle_revision=True,
    )[0]["lifecycle_revision"] == 0

    with _temporarily_patch(
        ow,
        "_confirm",
        lambda *args, **kwargs: {"approved": True, "choice": "once"},
    ):
        retried = _set_project_archived(ctx, **args)

    assert retried["ok"] is True
    assert retried["archived"] is True
    assert ow.list_committed_projects(
        ctx, lifecycle_revision=True,
    )[0]["lifecycle_revision"] == 1


def test_project_lifecycle_rechecks_revision_after_approval(ctx, monkeypatch):
    approver = _with_approver(ctx.session)
    created = _commit_task_graph(
        ctx,
        **_task_graph_args(
            idempotency_key="graph-lifecycle-race",
            project_name="Lifecycle Race Project",
        ),
    )
    approver.join()

    revisions = iter((0, 1))
    monkeypatch.setattr(
        ow,
        "_project_lifecycle_revision",
        lambda _conn, _ctx, _project_id: next(revisions),
    )
    approver = _with_approver(ctx.session)
    result = _set_project_archived(
        ctx,
        idempotency_key="project-lifecycle-race-archive",
        project_id=created["project_id"],
        expected_revision=0,
        action="archive",
    )
    approver.join()

    assert result == {
        "ok": False,
        "error": "conflict",
        "archived": False,
        "execution_paused": False,
    }
    with projects_db.connect_closing() as conn:
        assert projects_db.get_project(conn, created["project_id"]).archived is False
    assert kanban_db.board_dispatch_allowed(
        kanban_db.read_board_metadata(created["board"])
    ) is True


def test_project_lifecycle_rejects_project_without_owner_receipt(ctx):
    with projects_db.connect_closing() as conn:
        project_id = projects_db.create_project(conn, name="Foreign Project")

    with pytest.raises(ow.OwnerWorkspaceError) as excinfo:
        _set_project_archived(
            ctx,
            idempotency_key="foreign-project-archive",
            project_id=project_id,
            expected_revision=0,
            action="archive",
        )
    assert excinfo.value.code == "project_not_owned"
    with projects_db.connect_closing() as conn:
        assert projects_db.get_project(conn, project_id).archived is False


# The bootstrap anchor is non-executable by construction (no assignee, no
# execution tier, no approved route), so it lands in ``triage``.
_ANCHOR_STATUS = "triage"


def _bootstrap_board(ctx):
    t = _with_approver(ctx.session)
    result = ow.bootstrap(ctx, idempotency_key=f"setup-{ctx.session}-{time.monotonic()}", name="Board Setup")
    t.join()
    return result


def test_project_steward_snapshot_is_bounded_owner_safe_and_read_only(ctx):
    setup = _bootstrap_board(ctx)
    now = int(time.time())
    with kanban_db.connect(board=setup["board"]) as conn:
        done_id = kanban_db.create_task(
            conn,
            title="B03 — Publish the weekly owner summary",
            assignee="raphael-claude-worker",
            project_id=setup["project_id"],
        )
        review_id = kanban_db.create_task(
            conn,
            title="B04 — Confirm the owner-visible result",
            assignee="raphael-verifier",
            project_id=setup["project_id"],
        )
        stale_id = kanban_db.create_task(
            conn,
            title="B05 — Prepare the next useful milestone",
            assignee="raphael-planner",
            project_id=setup["project_id"],
        )
        blocked_id = kanban_db.create_task(
            conn,
            title="B06 — Confirm the missing owner input",
            assignee="raphael-planner",
            project_id=setup["project_id"],
        )
        with kanban_db.write_txn(conn):
            conn.execute(
                "UPDATE tasks SET status = 'done', completed_at = ? WHERE id = ?",
                (now - 60, done_id),
            )
            conn.execute(
                "UPDATE tasks SET status = 'review' WHERE id = ?", (review_id,)
            )
            conn.execute(
                "UPDATE tasks SET status = 'ready', created_at = ? WHERE id = ?",
                (now - 9 * 86_400, stale_id),
            )
            conn.execute(
                "UPDATE tasks SET status = 'blocked', block_kind = 'needs_input' "
                "WHERE id = ?",
                (blocked_id,),
            )
    kanban_db.write_board_dispatch_state(
        setup["board"], dispatch_enabled=True,
    )

    project_db = projects_db.projects_db_path()
    board_db = kanban_db.board_dir(setup["board"]) / "kanban.db"
    before = (project_db.stat().st_mtime_ns, board_db.stat().st_mtime_ns)

    snapshot = ow.project_steward_snapshot(
        project_id=setup["project_id"], lookback_days=7
    )

    after = (project_db.stat().st_mtime_ns, board_db.stat().st_mtime_ns)
    assert after == before
    assert snapshot["project"] == {"name": "Board Setup"}
    assert snapshot["progress"][0]["title"] == "Publish the weekly owner summary"
    assert snapshot["schema_version"] == 2
    assert snapshot["execution"] == {
        "state": "waiting_for_you",
        "summary": "Raphael needs your answer before the plan can continue.",
        "paused": False,
    }
    assert snapshot["decisions_needed"] == [{
        "title": "Confirm the missing owner input",
        "state": "Waiting for your answer",
    }]
    assert any(
        item == {
            "title": "Confirm the owner-visible result",
            "state": "Being checked",
        }
        for item in snapshot["active_work"]
    )
    assert any(
        item["title"] == "Prepare the next useful milestone"
        for item in snapshot["stale_candidates"]
    )
    assert all(
        item["title"] != "Confirm the missing owner input"
        for item in snapshot["needs_attention"]
    )

    payload = json.dumps(snapshot)
    for forbidden_value in (
        done_id,
        review_id,
        stale_id,
        blocked_id,
        "raphael-claude-worker",
        "raphael-verifier",
        "raphael-planner",
    ):
        assert forbidden_value not in payload


def test_project_steward_does_not_offer_resume_for_an_unapproved_board(ctx):
    approver = _with_approver(ctx.session)
    setup = ow.bootstrap(
        ctx,
        idempotency_key="steward-unapproved-board",
        name="Unapproved Project",
    )
    approver.join()

    snapshot = ow.project_steward_snapshot(project_id=setup["project_id"])

    assert snapshot["execution"] == {
        "state": "waiting_for_approval",
        "summary": "Raphael is waiting for an approved milestone before starting work.",
        "paused": False,
    }

    forbidden_keys = {
        "task_id", "assignee", "body", "result", "error", "file_path",
    }

    def assert_safe_shape(value):
        if isinstance(value, dict):
            assert not (set(value) & forbidden_keys)
            for item in value.values():
                assert_safe_shape(item)
        elif isinstance(value, list):
            for item in value:
                assert_safe_shape(item)

    assert_safe_shape(snapshot)


@pytest.mark.parametrize("lookback_days", [True, 0, 31, "7"])
def test_project_steward_snapshot_rejects_invalid_lookback(ctx, lookback_days):
    setup = _bootstrap_board(ctx)
    with pytest.raises(ow.OwnerWorkspaceError) as excinfo:
        ow.project_steward_snapshot(
            project_id=setup["project_id"], lookback_days=lookback_days
        )
    assert excinfo.value.code == "invalid_argument"


def test_owner_decisions_projects_native_gates_without_writes_or_identifiers(ctx):
    setup = _bootstrap_board(ctx)
    with kanban_db.connect(board=setup["board"]) as conn:
        review_id = kanban_db.create_task(
            conn,
            title="Review the workshop outline",
            project_id=setup["project_id"],
        )
        input_id = kanban_db.create_task(
            conn,
            title="Choose the workshop date",
            project_id=setup["project_id"],
        )
        recommendation_id = kanban_db.create_recommendation(
            conn,
            project_id=setup["project_id"],
            target_profile="raphael-planner",
            recommendation_kind="skill",
            recommendation_subject_id="workshop-research",
            recommendation_label="Add workshop research support",
            recommendation_rationale="The current milestone needs public-source research.",
            recommendation_evidence={
                "schema_version": 1,
                "need": "The workshop outline needs current public evidence.",
                "expected_benefit": "Keep the owner-facing advice current.",
                "requested_scope": {
                    flag: False for flag in kanban_db.RECOMMENDATION_SCOPE_FLAGS
                },
                "risks": "Low",
                "cost": "No added cost",
                "rollback": "Remove the staged skill configuration.",
            },
            provenance_authority="project-steward",
            provenance_ref="private-native-reference",
        )
        with kanban_db.write_txn(conn):
            conn.execute(
                "UPDATE tasks SET status = 'review' WHERE id = ?", (review_id,)
            )
            conn.execute(
                "UPDATE tasks SET status = 'blocked', block_kind = 'needs_input' "
                "WHERE id = ?",
                (input_id,),
            )

    project_db = projects_db.projects_db_path()
    board_db = kanban_db.board_dir(setup["board"]) / "kanban.db"
    before = (project_db.stat().st_mtime_ns, board_db.stat().st_mtime_ns)

    projected = ow.list_owner_decisions(ctx)
    decisions = projected["data"]
    assert projected["truncated"] is False

    assert (project_db.stat().st_mtime_ns, board_db.stat().st_mtime_ns) == before
    assert {(item["authority"], item["kind"], item["title"]) for item in decisions} == {
        ("task", "owner_input", "Choose the workshop date"),
        ("recommendation", "capability", "Add workshop research support"),
    }
    assert all(item["project_slug"] == setup["board"] for item in decisions)
    assert all(item["project_name"] == "Board Setup" for item in decisions)
    assert all(item["decision_ref"].startswith("decision_") for item in decisions)
    assert len({item["decision_ref"] for item in decisions}) == 2

    payload = json.dumps(decisions)
    for forbidden in (
        review_id,
        input_id,
        recommendation_id,
        setup["project_id"],
        "raphael-planner",
        "private-native-reference",
        "workshop-research",
    ):
        assert forbidden not in payload

    archived_approver = _with_approver(ctx.session)
    _set_project_archived(
        ctx,
        idempotency_key="decisions-archive-project",
        project_id=setup["project_id"],
        expected_revision=0,
        action="archive",
    )
    archived_approver.join()
    assert ow.list_owner_decisions(ctx) == {"data": [], "truncated": False}


# A stored Project name Raphael never validated on the way in: an ANSI
# colour sequence, a NUL, a zero-width space, a right-to-left override, an
# interlinear annotation separator, and ``user:password@`` URL userinfo.
# The invisible code points are written as escapes, never pasted — a literal
# invisible character in this file would be unreviewable.
_ZERO_WIDTH_SPACE = chr(0x200B)
_RL_OVERRIDE = chr(0x202E)
_ANNOTATION_SEPARATOR = chr(0xFFFA)
_UNSAFE_STORED_PROJECT_NAME = (
    f"Shoe\x00 \x1b[31mShop{_ZERO_WIDTH_SPACE} {_RL_OVERRIDE}plan"
    f"{_ANNOTATION_SEPARATOR}"
    " https://deploy:hunter2verylongpassword@git.example.com/repo.git"
)
_PROJECTED_PROJECT_NAME = (
    "Shoe Shop plan https://***@git.example.com/repo.git"
)


def _rename_project(project_id: str, name: str) -> None:
    """Store a Project name directly, bypassing the commit path's checks."""
    with projects_db.connect_closing() as pconn:
        with ow.write_txn(pconn):
            pconn.execute(
                "UPDATE projects SET name = ? WHERE id = ?", (name, project_id)
            )


def test_unsafe_project_name_is_projected_on_every_owner_surface(ctx):
    """One unsafe stored name, every owner-facing Project surface.

    A Project name can reach the store without ever passing the commit
    path's checks — an older row, a native rename, a direct write. Whatever
    is stored, no owner surface may hand back its control characters,
    invisible characters or URL credentials, and all of them must agree on
    the same projected name.
    """
    setup = _bootstrap_board(ctx)
    with kanban_db.connect(board=setup["board"]) as conn:
        input_id = kanban_db.create_task(
            conn,
            title="Choose the workshop date",
            project_id=setup["project_id"],
        )
        with kanban_db.write_txn(conn):
            conn.execute(
                "UPDATE tasks SET status = 'blocked', block_kind = 'needs_input' "
                "WHERE id = ?",
                (input_id,),
            )
    _rename_project(setup["project_id"], _UNSAFE_STORED_PROJECT_NAME)
    # The board's own display name falls back to the Project name in the
    # snapshot, so it is the same owner-facing slot and gets the same test.
    kanban_db.write_board_metadata(
        setup["board"], name=_UNSAFE_STORED_PROJECT_NAME
    )

    listed = ow.list_committed_projects(ctx)
    snapshot = ow.read_project_snapshot(ctx, setup["board"])
    steward = ow.project_steward_snapshot(project_id=setup["project_id"])
    projected = ow.list_owner_decisions(ctx)
    decisions = projected["data"]
    assert projected["truncated"] is False

    projected = [
        *[item["name"] for item in listed],
        snapshot["project"]["name"],
        snapshot["board"]["name"],
        snapshot["steward"]["project"]["name"],
        steward["project"]["name"],
        *[item["project_name"] for item in decisions],
    ]
    # Six surfaces, one name — every one of them, and all agreeing.
    assert len(projected) == 6
    assert set(projected) == {_PROJECTED_PROJECT_NAME}
    # Asserted on the strings, not on ``json.dumps`` output: JSON renders a
    # NUL as a six-character escape, so a serialized membership check would
    # pass whether or not the control character actually survived.
    for name in projected:
        for forbidden in (
            "hunter2verylongpassword", "\x00", "\x1b",
            _ZERO_WIDTH_SPACE, _RL_OVERRIDE, _ANNOTATION_SEPARATOR,
        ):
            assert forbidden not in name


def test_owner_approval_descriptions_project_every_project_name(ctx):
    """The approval prompt is an owner egress like every other one.

    ``description`` reaches the gateway notify callback and the
    ``pre_approval_request`` plugin hook verbatim, and it is the text the
    owner reads before deciding. A raw name interpolated into it would carry
    an ESC sequence, an invisible reordering character or a URL-borne
    credential onto the one surface whose whole job is to be trustworthy — and
    unlike a snapshot field, that text is what authorises a mutation.

    All four exact-operation descriptions that name a Project are covered:
    bootstrap and task-graph name the REQUESTED name, lifecycle and
    project-plan name the STORED one.
    """
    notified: list[dict] = []
    hooked: list[dict] = []

    def _record_hook(hook_name, **kwargs):
        hooked.append({"hook": hook_name, **kwargs})

    def _approve_once() -> threading.Thread:
        approval.register_gateway_notify(ctx.session, notified.append)
        thread = threading.Thread(target=_auto_approve, args=(ctx.session,))
        thread.start()
        return thread

    with _temporarily_patch(approval, "_fire_approval_hook", _record_hook):
        approver = _approve_once()
        boot = ow.bootstrap(
            ctx,
            idempotency_key="approval-bootstrap",
            name=_UNSAFE_STORED_PROJECT_NAME,
        )
        approver.join()
        assert boot["ok"] is True

        approver = _approve_once()
        graph = _commit_task_graph(
            ctx,
            **_task_graph_args(
                idempotency_key="approval-graph",
                project_name=_UNSAFE_STORED_PROJECT_NAME,
            ),
        )
        approver.join()
        assert graph["ok"] is True

        approver = _approve_once()
        planned = _commit_project_plan(
            ctx,
            **_project_plan_args(
                boot,
                [{
                    "action": "add",
                    "reason": "Create one bounded owner-approved task.",
                    "title": "Prepare the approved deliverable",
                    "body": "Produce the owner-visible result.",
                    "assignee": "default",
                    "execution_tier": "routine",
                    "existing_parents": [],
                    "new_parents": [],
                }],
                idempotency_key="approval-plan",
            ),
        )
        approver.join()
        assert planned["ok"] is True

        approver = _approve_once()
        archived = _set_project_archived(
            ctx,
            idempotency_key="approval-archive",
            project_id=boot["project_id"],
            expected_revision=0,
            action="archive",
        )
        approver.join()
        assert archived["ok"] is True

    approval.unregister_gateway_notify(ctx.session)

    expected = [
        f"Bootstrap owner workspace project {_PROJECTED_PROJECT_NAME!r}",
        f"Create project {_PROJECTED_PROJECT_NAME!r} with 2 tasks",
        f"Apply 1 approved Project change(s) to {_PROJECTED_PROJECT_NAME!r}",
        f"Archive Project {_PROJECTED_PROJECT_NAME!r}",
    ]
    assert [item["description"] for item in notified] == expected
    # ``command`` is the same string on the same payload — the gateway
    # surfaces render one or the other, so neither may be the raw name.
    assert [item["command"] for item in notified] == expected

    # Observer plugins read the same text off the hook, before any UI does.
    pre_request = [
        item for item in hooked if item["hook"] == "pre_approval_request"
    ]
    assert [item["description"] for item in pre_request] == expected
    assert [item["command"] for item in pre_request] == expected

    # Asserted on the strings, not on ``json.dumps`` output: JSON renders a
    # NUL as a six-character escape, so a serialized membership check would
    # pass whether or not the control character actually survived.
    for text in expected:
        for forbidden in (
            "hunter2verylongpassword", "\x00", "\x1b",
            _ZERO_WIDTH_SPACE, _RL_OVERRIDE, _ANNOTATION_SEPARATOR,
        ):
            assert forbidden not in text


def test_owner_project_name_bound_is_applied_to_a_stored_name(ctx):
    """A stored name longer than the bound is cut, not rejected."""
    setup = _bootstrap_board(ctx)
    _rename_project(setup["project_id"], "n" * 400)

    assert ow.list_committed_projects(ctx)[0]["name"] == "n" * 160
    assert (
        ow.project_steward_snapshot(project_id=setup["project_id"])["project"]
        == {"name": "n" * 160}
    )


def _project_task_ref(conn, task_id: str) -> dict:
    # A plan can reference the Project's control anchor as a parent, so this
    # helper resolves either kind — exactly like the owner kernel does.
    task = kanban_db.get_task(conn, task_id, include_control=True)
    assert task is not None
    return {
        "task_id": task_id,
        "expected_status": task.status,
        "expected_revision": kanban_db.task_event_revision(conn, task_id),
    }


def _project_plan_args(setup: dict, changes: list[dict], **overrides) -> dict:
    args = {
        "idempotency_key": "steward-plan-1",
        "project_id": setup["project_id"],
        # No anchor: the Project's hidden control row is resolved inside the
        # kernel from its committed bootstrap receipt, never named by a caller.
        "trigger": "owner_request",
        "request_title": "Adapt the current plan",
        "summary": "Keep the active work small and current.",
        "specification": "Apply only the approved changes and preserve history.",
        "current_milestone": "Finish the revised milestone.",
        "owner_visible_result": "The owner sees the revised work on the same board.",
        "later_milestones": ["Reconsider the next milestone from live facts."],
        "changes": changes,
    }
    args.update(overrides)
    return args


def test_project_plan_timeout_can_retry_but_still_requires_approval(ctx):
    setup = _bootstrap_board(ctx)
    args = _project_plan_args(
        setup,
        [{
            "action": "add",
            "reason": "Create one bounded owner-approved task.",
            "title": "Prepare the approved deliverable",
            "body": "Produce the owner-visible result.",
            "assignee": "default",
            "execution_tier": "routine",
            "existing_parents": [],
            "new_parents": [],
        }],
        idempotency_key="steward-timeout-retry",
    )

    with _temporarily_patch(
        ow,
        "_confirm",
        lambda *args, **kwargs: {"approved": False, "reason": "timeout"},
    ):
        timed_out = _commit_project_plan(ctx, **args)

    assert timed_out == {
        "ok": False,
        "error": "confirmation_denied",
        "reason": "timeout",
    }
    with kanban_db.connect(board=setup["board"]) as conn:
        assert all(
            task.title != "Prepare the approved deliverable"
            for task in kanban_db.list_tasks(conn)
            if task.project_id == setup["project_id"]
        )

    with _temporarily_patch(
        ow,
        "_confirm",
        lambda *args, **kwargs: {"approved": True, "choice": "once"},
    ):
        retried = _commit_project_plan(ctx, **args)

    assert retried["ok"] is True
    assert retried["change_count"] == 1
    with kanban_db.connect(board=setup["board"]) as conn:
        matching = [
            task for task in kanban_db.list_tasks(conn)
            if task.project_id == setup["project_id"]
            and task.title == "Prepare the approved deliverable"
        ]
    assert len(matching) == 1


def test_project_plan_resolves_and_locks_a_new_task_model_route(ctx):
    setup = _bootstrap_board(ctx)
    args = _project_plan_args(
        setup,
        [{
            "action": "add",
            "reason": "Create one complex owner-approved deliverable.",
            "title": "Build the complex deliverable",
            "body": "Produce and verify the owner-visible result.",
            "assignee": "default",
            "execution_tier": "deep",
            "existing_parents": [],
            "new_parents": [],
        }],
        idempotency_key="steward-model-route",
    )

    real_resolve = ow.resolve_task_assignment

    def resolved_route(profile, execution_tier):
        assert (profile, execution_tier) == ("default", "deep")
        return real_resolve(profile, execution_tier)

    with _temporarily_patch(ow, "resolve_task_assignment", resolved_route):
        approver = _with_approver(ctx.session)
        result = _commit_project_plan(ctx, **args)
        approver.join()

    with kanban_db.connect(board=setup["board"]) as conn:
        task = kanban_db.get_task(conn, result["created_task_ids"][0])
        with pytest.raises(RuntimeError, match="owner-governed"):
            kanban_db.set_model_override(
                conn, task.id, "claude-sonnet-5", provider="anthropic",
            )

    assert (
        task.provider_override,
        task.model_override,
        task.reasoning_effort,
        task.execution_tier,
        task.model_policy_lock,
    ) == (
        "anthropic", "claude-opus-5", "max", "deep",
        kanban_db.mint_policy_lock(
            "default", "anthropic", "claude-opus-5", "max", "deep",
        ),
    )


@pytest.mark.parametrize("action", ["add", "replace", "split", "merge"])
def test_project_plan_requires_an_execution_tier_for_every_created_task(ctx, action):
    setup = _bootstrap_board(ctx)
    with kanban_db.connect(board=setup["board"]) as conn:
        target = _project_task_ref(conn, setup["task_id"])
        sibling = _project_task_ref(
            conn,
            kanban_db.create_task(
                conn,
                title="Second half",
                assignee="default",
                project_id=setup["project_id"],
            ),
        )

    spec = {
        "title": "Untiered work",
        "body": "This must not reach an owner decision.",
        "assignee": "default",
        "responsibility": "B04",
    }
    if action == "add":
        change = {
            "action": "add",
            "reason": "Create untiered work.",
            **spec,
            "existing_parents": [],
            "new_parents": [],
        }
    elif action == "replace":
        change = {
            "action": "replace",
            "reason": "Replace with untiered work.",
            "target": target,
            "replacement": spec,
        }
    elif action == "split":
        change = {
            "action": "split",
            "reason": "Split into untiered work.",
            "target": target,
            "replacements": [
                {**spec, "parents": []},
                {**spec, "title": "Second untiered work", "parents": [0]},
            ],
        }
    else:
        change = {
            "action": "merge",
            "reason": "Merge into untiered work.",
            "targets": [target, sibling],
            "replacement": spec,
        }

    with pytest.raises(ow.OwnerWorkspaceError) as excinfo:
        _commit_project_plan(
            ctx,
            **_project_plan_args(
                setup, [change], idempotency_key=f"steward-untiered-{action}",
            ),
        )

    assert excinfo.value.code == "invalid_argument"


def test_effective_route_fence_pins_exposed_owner_work_and_nothing_else(ctx, monkeypatch):
    """The rollout fence freezes pre-lock owner work and nothing else."""
    approver = _with_approver(ctx.session)
    result = _commit_task_graph(ctx, **_task_graph_args(idempotency_key="graph-fence"))
    approver.join()

    approved_lock = kanban_db.mint_policy_lock(
        "default", "anthropic", "claude-opus-5", "max", "routine",
    )
    routeless_id, partial_id = result["task_ids"][0], result["root_task_id"]
    with kanban_db.connect(board=result["board"]) as conn:
        # Rewind one task to the pre-change shape: an owner task whose whole
        # route still comes from its profile.
        with kanban_db.write_txn(conn):
            conn.execute(
                "UPDATE tasks SET model_override = NULL, provider_override = NULL, "
                "reasoning_effort = NULL, model_policy_lock = NULL WHERE id = ?",
                (routeless_id,),
            )
            # And rewind another to the shape the old fence SKIPPED: an
            # explicit model with an inherited provider and an explicit
            # non-default effort. It is exposed too — and its completed route
            # is not one the policy admits, so it must be parked rather than
            # left runnable.
            conn.execute(
                "UPDATE tasks SET model_override = 'claude-sonnet-5', "
                "provider_override = NULL, reasoning_effort = 'high', "
                "model_policy_lock = NULL WHERE id = ?",
                (partial_id,),
            )
        # A human card on the SAME owner board and role: it inherits the
        # board's project but the kernel did not create it.
        manual_id = kanban_db.create_task(
            conn, title="manual card", assignee="default", board=result["board"],
        )
        manual_in_project_id = kanban_db.create_task(
            conn,
            title="manual card inside the Project",
            assignee="default",
            board=result["board"],
            project_id=result["project_id"],
        )
        other_role_id = kanban_db.create_task(
            conn,
            title="another role's card",
            assignee="raphael-verifier",
            board=result["board"],
            project_id=result["project_id"],
        )

    monkeypatch.setattr(
        ow,
        "configured_assignment_for",
        lambda profile: SimpleNamespace(
            provider="anthropic", model="claude-opus-5", reasoning_effort="max",
        ),
    )
    # Only the task the policy can actually authorize is pinned.
    assert ow.fence_effective_task_routes("default") == [routeless_id]

    with kanban_db.connect(board=result["board"]) as conn:
        fully_inherited = kanban_db.get_task(conn, routeless_id)
        assert (
            fully_inherited.provider_override,
            fully_inherited.model_override,
            fully_inherited.reasoning_effort,
            fully_inherited.model_policy_lock,
        ) == ("anthropic", "claude-opus-5", "max", approved_lock)

        # Its explicit components are preserved exactly — the fence never
        # rewrites an operator's route. Completing it from the profile still
        # yields a route the policy does not admit, and an unlocked owner task
        # must never stay dispatchable, so it is parked for re-approval.
        partial = kanban_db.get_task(conn, partial_id)
        assert (
            partial.provider_override,
            partial.model_override,
            partial.reasoning_effort,
            partial.model_policy_lock,
        ) == (None, "claude-sonnet-5", "high", None)
        assert (partial.status, partial.block_kind) == ("blocked", "needs_input")
        assert any(
            event.kind == "model_route_unapproved"
            and event.payload["reapproval_required"] is True
            for event in kanban_db.list_events(conn, partial_id)
        )

        # Manual cards — inside the Project or not — and another role's card
        # are all untouched: only ids a committed receipt records creating,
        # and that this role holds, are fenced.
        for untouched_id in (manual_id, manual_in_project_id, other_role_id):
            untouched = kanban_db.get_task(conn, untouched_id)
            assert untouched.model_policy_lock is None
            assert untouched.model_override is None

        # Already-locked owner tasks are left exactly as approved.
        assert (
            kanban_db.get_task(conn, result["task_ids"][1]).model_policy_lock
            == approved_lock
        )
    # A second pass pins nothing new: the pinned task now carries a lock and
    # is no longer exposed, and the unpinnable one is already parked.
    assert ow.fence_effective_task_routes("default") == []
    with kanban_db.connect(board=result["board"]) as conn:
        assert kanban_db.get_task(conn, partial_id).status == "blocked"


def test_effective_route_fence_fails_closed_on_unprovable_receipts(ctx, monkeypatch):
    """An unreadable receipt store must never read as "no owner work"."""
    _bootstrap_board(ctx)  # the receipt store now exists and has work in it

    def _boom(*_args, **_kwargs):
        raise sqlite3.Error("receipt store unavailable")

    monkeypatch.setattr(ow.sqlite3, "connect", _boom)
    with pytest.raises(ow.OwnerWorkspaceError) as excinfo:
        ow.fence_effective_task_routes("default")
    assert excinfo.value.code == "execution_state_busy"


def test_effective_route_fence_needs_no_route_when_nothing_is_exposed(monkeypatch):
    """A role with no routeless owner work must not require a route read."""
    def _never(profile):  # pragma: no cover - asserted by not being called
        raise AssertionError("current route must not be read with nothing to fence")

    monkeypatch.setattr(ow, "configured_assignment_for", _never)
    assert ow.fence_effective_task_routes("raphael-builder") == []


def test_project_plan_approval_cannot_mutate_a_project_archived_while_waiting(ctx):
    setup = _bootstrap_board(ctx)
    args = _project_plan_args(
        setup,
        [{
            "action": "add",
            "reason": "Create one bounded owner-approved task.",
            "title": "Must not land after archive",
            "body": "This stale approval must fail closed.",
            "assignee": "default",
            "execution_tier": "routine",
            "existing_parents": [],
            "new_parents": [],
        }],
        idempotency_key="steward-archive-race",
    )

    def archive_then_approve(*_args, **_kwargs):
        with projects_db.connect_closing() as pconn:
            with ow.write_txn(pconn):
                pconn.execute(
                    "UPDATE projects SET archived = 1 WHERE id = ?",
                    (setup["project_id"],),
                )
        return {"approved": True, "choice": "once"}

    with _temporarily_patch(ow, "_confirm", archive_then_approve):
        result = _commit_project_plan(ctx, **args)

    assert result["ok"] is False
    assert result["error"] == "conflict"
    with kanban_db.connect(board=setup["board"]) as conn:
        assert all(
            task.title != "Must not land after archive"
            for task in kanban_db.list_tasks(conn)
        )


def test_project_plan_replace_preserves_history_edges_and_replays(ctx):
    setup = _bootstrap_board(ctx)
    with kanban_db.connect(board=setup["board"]) as conn:
        source_id = kanban_db.create_task(
            conn,
            title="Stopped implementation",
            assignee="default",
            parents=[setup["task_id"]],
            project_id=setup["project_id"],
        )
        downstream_id = kanban_db.create_task(
            conn,
            title="Review the repaired implementation",
            assignee="default",
            parents=[source_id],
            project_id=setup["project_id"],
        )
        source_ref = _project_task_ref(conn, source_id)

    args = _project_plan_args(
        setup,
        [{
            "action": "replace",
            "reason": "The earlier attempt stopped before producing a result.",
            "target": source_ref,
            "replacement": {
                "title": "Repair the stopped implementation",
                "body": "Preserve the findings and continue the same bounded delivery chain.",
                "assignee": "default",
                "execution_tier": "deep",
                "responsibility": "R07",
            },
        }],
        idempotency_key="steward-replace-stopped-work",
    )
    approver = _with_approver(ctx.session)
    first = _commit_project_plan(ctx, **args)
    approver.join()

    assert first["ok"] is True
    assert first["change_count"] == 1
    assert len(first["created_task_ids"]) == 1
    replacement_id = first["created_task_ids"][0]

    with kanban_db.connect(board=setup["board"]) as conn:
        assert kanban_db.get_task(conn, source_id).status == "archived"
        assert kanban_db.parent_ids(conn, replacement_id) == [setup["task_id"]]
        assert replacement_id in kanban_db.parent_ids(conn, downstream_id)
        replacement = kanban_db.get_task(conn, replacement_id)
        assert replacement.responsibility == "R07"
        assert replacement.execution_tier == "deep"
        event_count = conn.execute(
            "SELECT COUNT(*) FROM task_events WHERE task_id IN (?, ?)",
            (source_id, replacement_id),
        ).fetchone()[0]

    replay = _commit_project_plan(ctx, **args)
    assert replay == first
    with kanban_db.connect(board=setup["board"]) as conn:
        assert conn.execute(
            "SELECT COUNT(*) FROM task_events WHERE task_id IN (?, ?)",
            (source_id, replacement_id),
        ).fetchone()[0] == event_count


def test_project_plan_replaces_a_dependency_chain_atomically(ctx):
    """Internal relinking must not invalidate later targets in the same plan."""
    setup = _bootstrap_board(ctx)
    with kanban_db.connect(board=setup["board"]) as conn:
        stage_id = kanban_db.create_task(
            conn, title="Stage candidate", assignee="default",
            parents=[setup["task_id"]], project_id=setup["project_id"],
        )
        verify_id = kanban_db.create_task(
            conn, title="Verify candidate", assignee="default",
            parents=[stage_id], project_id=setup["project_id"],
        )
        release_id = kanban_db.create_task(
            conn, title="Release candidate", assignee="default",
            parents=[verify_id], project_id=setup["project_id"],
        )
        conn.execute(
            "UPDATE tasks SET status = 'ready' WHERE id IN (?, ?)",
            (verify_id, release_id),
        )
        refs = {
            task_id: _project_task_ref(conn, task_id)
            for task_id in (stage_id, verify_id, release_id)
        }

    changes = []
    for task_id, title, responsibility in (
        (stage_id, "Stage exact candidate", "R17"),
        (verify_id, "Verify exact candidate", "R21"),
        (release_id, "Release exact candidate", "R17"),
    ):
        changes.append({
            "action": "replace",
            "reason": "Retry the same step with the exact accepted candidate.",
            "target": refs[task_id],
            "replacement": {
                "title": title,
                "body": "Preserve the existing dependency chain and evidence.",
                "assignee": "default",
                "execution_tier": "deep",
                "responsibility": responsibility,
            },
        })

    args = _project_plan_args(
        setup, changes, idempotency_key="steward-replace-dependency-chain",
    )
    approver = _with_approver(ctx.session)
    result = _commit_project_plan(ctx, **args)
    approver.join()

    assert result["ok"] is True
    assert result["change_count"] == 3
    assert len(result["created_task_ids"]) == 3
    new_stage, new_verify, new_release = result["created_task_ids"]
    with kanban_db.connect(board=setup["board"]) as conn:
        for task_id in (stage_id, verify_id, release_id):
            assert kanban_db.get_task(conn, task_id).status == "archived"
        assert new_stage in kanban_db.parent_ids(conn, new_verify)
        assert new_verify in kanban_db.parent_ids(conn, new_release)

    assert _commit_project_plan(ctx, **args) == result


# ---------------------------------------------------------------------------
# Explicit repository ownership scope on owner-created tasks
# ---------------------------------------------------------------------------


def _project_repo(tmp_path: Path, ctx, project_id: str) -> Path:
    """Attach a real git repo as the Project's primary folder."""
    repo = tmp_path / "owner-repo"
    repo.mkdir()
    subprocess.run(
        ["git", "init", "-b", "main", str(repo)],
        check=True, capture_output=True, text=True,
    )
    (repo / "README.md").write_text("owner\n", encoding="utf-8")
    subprocess.run(
        ["git", "-C", str(repo), "-c", "user.name=T", "-c", "user.email=t@e.x",
         "-c", "commit.gpgsign=false", "add", "."],
        check=True, capture_output=True, text=True,
    )
    subprocess.run(
        ["git", "-C", str(repo), "-c", "user.name=T", "-c", "user.email=t@e.x",
         "-c", "commit.gpgsign=false", "commit", "-m", "init"],
        check=True, capture_output=True, text=True,
    )
    with projects_db.connect_closing() as pconn:
        projects_db.add_folder(pconn, project_id, str(repo), is_primary=True)
    return repo


def test_project_plan_replace_carries_an_explicit_ownership_scope(ctx, tmp_path):
    """An owner-approved replace persists the scope, worktree, and locks."""
    setup = _bootstrap_board(ctx)
    repo = _project_repo(tmp_path, ctx, setup["project_id"])
    with kanban_db.connect(board=setup["board"]) as conn:
        source_id = kanban_db.create_task(
            conn,
            title="Unscoped implementation",
            assignee="default",
            parents=[setup["task_id"]],
            project_id=setup["project_id"],
        )
        downstream_id = kanban_db.create_task(
            conn,
            title="Review the scoped implementation",
            assignee="default",
            parents=[source_id],
            project_id=setup["project_id"],
        )
        source_ref = _project_task_ref(conn, source_id)
        source_events = kanban_db.task_event_revision(conn, source_id)

    args = _project_plan_args(
        setup,
        [{
            "action": "replace",
            "reason": "Bound the retry to exactly the files it may change.",
            "target": source_ref,
            "replacement": {
                "title": "Scoped retry of the implementation",
                "body": "Change only the owned subtree and its tests.",
                "assignee": "default",
                "execution_tier": "deep",
                "responsibility": "R09",
                "owned_paths": ["src/api", "tests/api"],
            },
        }],
        idempotency_key="steward-replace-scoped",
    )
    approver = _with_approver(ctx.session)
    result = _commit_project_plan(ctx, **args)
    approver.join()

    assert result["ok"] is True
    replacement_id = result["created_task_ids"][0]
    with kanban_db.connect(board=setup["board"]) as conn:
        replacement = kanban_db.get_task(conn, replacement_id)
        # The explicit boundary, exactly as approved and canonicalised.
        assert replacement.owned_paths == ["src/api", "tests/api"]
        # A mutating scope must land in an isolated project worktree.
        assert replacement.workspace_kind == "worktree"
        assert replacement.workspace_path == str(
            repo / ".worktrees" / replacement_id
        )
        # The route lock still binds the whole approved route tuple.
        assert replacement.model_policy_lock == kanban_db.mint_policy_lock(
            "default", "anthropic", "claude-opus-5", "max", "deep",
        )
        assert replacement.responsibility == "R09"
        with pytest.raises(RuntimeError, match="owner-governed"):
            kanban_db.set_model_override(
                conn, replacement_id, "claude-sonnet-5", provider="anthropic",
            )
        # History and dependency edges survive the replacement.
        assert kanban_db.get_task(conn, source_id).status == "archived"
        assert kanban_db.task_event_revision(conn, source_id) > source_events
        assert kanban_db.parent_ids(conn, replacement_id) == [setup["task_id"]]
        assert replacement_id in kanban_db.parent_ids(conn, downstream_id)

    replay = _commit_project_plan(ctx, **args)
    assert replay == result


@pytest.mark.parametrize(
    ("case", "source_scope"),
    [("mutating", ["src/api", "tests/api"]), ("read-only", [])],
    ids=["mutating", "read-only"],
)
def test_project_plan_replace_inherits_an_explicit_source_boundary(
    ctx, tmp_path, case, source_scope
):
    """A replacement that states no scope inherits the source's exact one.

    The source already carries an owner-approved canonical boundary, so the
    one-to-one successor continues its work under precisely that boundary —
    neither narrowed nor widened. In particular an explicitly read-only
    ``[]`` source stays read-only: it is a stated boundary, not the absent
    (``None``) legacy spelling that resolves to whole-repository ``["."]``.
    """
    setup = _bootstrap_board(ctx)
    repo = _project_repo(tmp_path, ctx, setup["project_id"])
    with kanban_db.connect(board=setup["board"]) as conn:
        source_id = kanban_db.create_task(
            conn,
            title="Bounded implementation",
            assignee="default",
            parents=[setup["task_id"]],
            project_id=setup["project_id"],
            owned_paths=source_scope,
        )
        source = kanban_db.get_task(conn, source_id)
        # The canonical boundary as the board actually stored it, which is
        # what the successor has to end up holding.
        stored_scope = source.owned_paths
        assert stored_scope == source_scope
        assert source.workspace_kind == "worktree"
        source_ref = _project_task_ref(conn, source_id)

    args = _project_plan_args(
        setup,
        [{
            "action": "replace",
            "reason": "Retry the same bounded work after it stalled.",
            "target": source_ref,
            "replacement": {
                "title": "Retry of the bounded implementation",
                "body": "Continue within the boundary already approved.",
                "assignee": "default",
                "execution_tier": "deep",
                "responsibility": "R09",
            },
        }],
        idempotency_key=f"steward-replace-inherit-{case}",
    )
    approver = _with_approver(ctx.session)
    result = _commit_project_plan(ctx, **args)
    approver.join()

    assert result["ok"] is True
    replacement_id = result["created_task_ids"][0]
    with kanban_db.connect(board=setup["board"]) as conn:
        replacement = kanban_db.get_task(conn, replacement_id)
        # Exactly the source's boundary, with no fail-closed widening to
        # whole-repository ownership and no silent unscoping.
        assert replacement.owned_paths == stored_scope
        assert replacement.owned_paths != ["."]
        assert replacement.workspace_kind == "worktree"
        assert replacement.workspace_path == str(
            repo / ".worktrees" / replacement_id
        )
        assert kanban_db.get_task(conn, source_id).status == "archived"

    replay = _commit_project_plan(ctx, **args)
    assert replay == result
    with kanban_db.connect(board=setup["board"]) as conn:
        # Replay neither re-inherits into a second task nor rewrites the scope.
        assert kanban_db.get_task(conn, replacement_id).owned_paths == stored_scope


def test_project_plan_replace_inherits_a_legacy_worktree_boundary(ctx, tmp_path):
    """A one-to-one replacement of a legacy task keeps that task's boundary.

    ``owned_paths IS NULL`` on a repository worktree task already means
    exclusive whole-repository ownership, but only the literal ``["."]`` can
    be provisioned: the trusted sandbox provisioner refuses to infer a
    boundary. Omitting the scope on the replacement therefore normalizes the
    representation of the boundary the source task already had — it does not
    widen it.
    """
    setup = _bootstrap_board(ctx)
    repo = _project_repo(tmp_path, ctx, setup["project_id"])
    with kanban_db.connect(board=setup["board"]) as conn:
        source_id = kanban_db.create_task(
            conn,
            title="Legacy unscoped implementation",
            assignee="default",
            parents=[setup["task_id"]],
            project_id=setup["project_id"],
        )
        downstream_id = kanban_db.create_task(
            conn,
            title="Review the repaired implementation",
            assignee="default",
            parents=[source_id],
            project_id=setup["project_id"],
        )
        source = kanban_db.get_task(conn, source_id)
        # The reproduced failure's exact source shape.
        assert source.owned_paths is None
        assert source.workspace_kind == "worktree"
        source_ref = _project_task_ref(conn, source_id)

    args = _project_plan_args(
        setup,
        [{
            "action": "replace",
            "reason": "The earlier attempt stopped before producing a result.",
            "target": source_ref,
            "replacement": {
                "title": "Repair the stopped implementation",
                "body": "Continue the same bounded delivery chain.",
                "assignee": "default",
                "execution_tier": "deep",
                "responsibility": "R07",
            },
        }],
        idempotency_key="steward-replace-legacy-scope",
    )
    approver = _with_approver(ctx.session)
    result = _commit_project_plan(ctx, **args)
    approver.join()

    assert result["ok"] is True
    replacement_id = result["created_task_ids"][0]
    with kanban_db.connect(board=setup["board"]) as conn:
        replacement = kanban_db.get_task(conn, replacement_id)
        # The source's own exclusive whole-repository boundary, made literal.
        assert replacement.owned_paths == ["."]
        assert replacement.workspace_kind == "worktree"
        assert replacement.workspace_path == str(
            repo / ".worktrees" / replacement_id
        )
        # History and dependency edges survive the one-to-one replacement.
        assert kanban_db.get_task(conn, source_id).status == "archived"
        assert kanban_db.parent_ids(conn, replacement_id) == [setup["task_id"]]
        assert replacement_id in kanban_db.parent_ids(conn, downstream_id)

    replay = _commit_project_plan(ctx, **args)
    assert replay == result
    with kanban_db.connect(board=setup["board"]) as conn:
        assert kanban_db.get_task(conn, replacement_id).owned_paths == ["."]


def test_project_plan_replace_of_a_scratch_task_invents_no_boundary(ctx):
    """An unscoped non-repository source stays unscoped — no repo authority."""
    setup = _bootstrap_board(ctx)  # bootstrap attaches no repository folder
    with kanban_db.connect(board=setup["board"]) as conn:
        source_id = kanban_db.create_task(
            conn,
            title="Scratch investigation",
            assignee="default",
            parents=[setup["task_id"]],
            project_id=setup["project_id"],
        )
        source = kanban_db.get_task(conn, source_id)
        assert source.owned_paths is None
        assert source.workspace_kind == "scratch"
        source_ref = _project_task_ref(conn, source_id)

    approver = _with_approver(ctx.session)
    result = _commit_project_plan(
        ctx, **_project_plan_args(
            setup,
            [{
                "action": "replace",
                "reason": "Retry the investigation with a sharper question.",
                "target": source_ref,
                "replacement": {
                    "title": "Retry the investigation",
                    "body": "Report findings; there is no repository here.",
                    "assignee": "default",
                    "execution_tier": "routine",
                    "responsibility": "R07",
                },
            }],
            idempotency_key="steward-replace-scratch-scope",
        )
    )
    approver.join()

    assert result["ok"] is True
    with kanban_db.connect(board=setup["board"]) as conn:
        replacement = kanban_db.get_task(conn, result["created_task_ids"][0])
        assert replacement.owned_paths is None
        assert replacement.workspace_kind == "scratch"


@pytest.mark.parametrize("action", ["split", "merge"])
def test_only_a_one_to_one_replace_inherits_a_source_boundary(ctx, tmp_path, action):
    """Split and merge are not one-to-one, so nothing is inherited for them."""
    setup = _bootstrap_board(ctx)
    _project_repo(tmp_path, ctx, setup["project_id"])
    spec = {
        "title": "Successor work item",
        "body": "No boundary is declared for this successor.",
        "assignee": "default",
        "execution_tier": "routine",
        "responsibility": "R07",
    }
    with kanban_db.connect(board=setup["board"]) as conn:
        target = _project_task_ref(
            conn,
            kanban_db.create_task(
                conn,
                title="First legacy item",
                assignee="default",
                parents=[setup["task_id"]],
                project_id=setup["project_id"],
            ),
        )
        sibling = _project_task_ref(
            conn,
            kanban_db.create_task(
                conn,
                title="Second legacy item",
                assignee="default",
                parents=[setup["task_id"]],
                project_id=setup["project_id"],
            ),
        )
    if action == "split":
        change = {
            "action": "split", "reason": "Split the legacy item in two.",
            "target": target,
            "replacements": [
                {**spec, "parents": []},
                {**spec, "parents": []},
            ],
        }
    else:
        change = {
            "action": "merge", "reason": "Merge the legacy items into one.",
            "targets": [target, sibling], "replacement": spec,
        }

    approver = _with_approver(ctx.session)
    result = _commit_project_plan(
        ctx, **_project_plan_args(
            setup, [change], idempotency_key=f"steward-no-inherit-{action}",
        )
    )
    approver.join()

    assert result["ok"] is True
    with kanban_db.connect(board=setup["board"]) as conn:
        scopes = [
            kanban_db.get_task(conn, task_id).owned_paths
            for task_id in result["created_task_ids"]
        ]
    assert scopes == [None] * len(scopes)


@pytest.mark.parametrize("action", ["add", "replace", "split", "merge"])
def test_every_plan_action_that_creates_a_task_can_carry_a_scope(
    ctx, tmp_path, action
):
    setup = _bootstrap_board(ctx)
    _project_repo(tmp_path, ctx, setup["project_id"])
    spec = {
        "title": "Scoped work item",
        "body": "Change only the owned subtree.",
        "assignee": "default",
        "execution_tier": "routine",
        "responsibility": "R10",
        "owned_paths": ["docs"],
    }
    with kanban_db.connect(board=setup["board"]) as conn:
        # replace/split/merge mutate their targets, so a target must be a work
        # task: the Project's control anchor is never mutable.
        target = _project_task_ref(
            conn,
            kanban_db.create_task(
                conn,
                title="First mergeable item",
                assignee="default",
                parents=[setup["task_id"]],
                project_id=setup["project_id"],
            ),
        )
        sibling = _project_task_ref(
            conn,
            kanban_db.create_task(
                conn,
                title="Second mergeable item",
                assignee="default",
                parents=[setup["task_id"]],
                project_id=setup["project_id"],
            ),
        )
    if action == "add":
        change = {
            "action": "add", "reason": "Add one scoped task.", **spec,
            "existing_parents": [], "new_parents": [],
        }
    elif action == "replace":
        change = {
            "action": "replace", "reason": "Rescope the stalled task.",
            "target": target, "replacement": spec,
        }
    elif action == "split":
        change = {
            "action": "split", "reason": "Split into two disjoint scopes.",
            "target": target,
            "replacements": [
                {**spec, "owned_paths": ["docs"], "parents": []},
                {**spec, "owned_paths": ["src"], "parents": []},
            ],
        }
    else:
        change = {
            "action": "merge", "reason": "Merge into one scoped task.",
            "targets": [target, sibling], "replacement": spec,
        }

    approver = _with_approver(ctx.session)
    result = _commit_project_plan(
        ctx, **_project_plan_args(
            setup, [change], idempotency_key=f"steward-scope-{action}",
        )
    )
    approver.join()

    assert result["ok"] is True
    with kanban_db.connect(board=setup["board"]) as conn:
        scopes = [
            kanban_db.get_task(conn, task_id).owned_paths
            for task_id in result["created_task_ids"]
        ]
    assert all(scope for scope in scopes), scopes
    assert scopes[0] == ["docs"]


def test_a_plan_without_a_scope_keeps_the_default_boundary(ctx, tmp_path):
    """Backward compatibility: omitting the field changes nothing."""
    setup = _bootstrap_board(ctx)
    _project_repo(tmp_path, ctx, setup["project_id"])
    approver = _with_approver(ctx.session)
    result = _commit_project_plan(
        ctx, **_project_plan_args(
            setup,
            [{
                "action": "add",
                "reason": "Add one ordinary task.",
                "title": "Ordinary work item",
                "body": "No explicit boundary is declared.",
                "assignee": "default",
                "execution_tier": "routine",
                "existing_parents": [],
                "new_parents": [],
            }],
            idempotency_key="steward-scope-absent",
        )
    )
    approver.join()
    with kanban_db.connect(board=setup["board"]) as conn:
        assert kanban_db.get_task(
            conn, result["created_task_ids"][0]
        ).owned_paths is None


@pytest.mark.parametrize(
    "scope",
    [
        "src",
        ["/etc/passwd"],
        ["../escape"],
        ["src/*"],
        [".git"],
        [".git/config"],
        ["src/../../etc"],
        [".", "src"],
        [""],
        [None],
        ["src" + "/deep" * 200],
    ],
)
def test_a_malformed_scope_is_refused_before_approval(ctx, tmp_path, scope):
    setup = _bootstrap_board(ctx)
    _project_repo(tmp_path, ctx, setup["project_id"])
    args = _project_plan_args(
        setup,
        [{
            "action": "add",
            "reason": "Attempt an unsafe boundary.",
            "title": "Unsafe work item",
            "body": "This must never reach an approval prompt.",
            "assignee": "default",
            "execution_tier": "routine",
            "existing_parents": [],
            "new_parents": [],
            "owned_paths": scope,
        }],
        idempotency_key="steward-scope-malformed",
    )
    with pytest.raises(ow.OwnerWorkspaceError) as excinfo:
        _commit_project_plan(ctx, **args)
    assert excinfo.value.code == "invalid_ownership_scope"
    with kanban_db.connect(board=setup["board"]) as conn:
        assert conn.execute(
            "SELECT COUNT(*) FROM tasks WHERE title = 'Unsafe work item'"
        ).fetchone()[0] == 0


def test_a_mutating_scope_needs_a_project_repository(ctx):
    setup = _bootstrap_board(ctx)  # bootstrap attaches no folders
    args = _project_plan_args(
        setup,
        [{
            "action": "add",
            "reason": "Declare a boundary with no repository behind it.",
            "title": "Repo-less scoped item",
            "body": "There is nothing to scope a worktree in.",
            "assignee": "default",
            "execution_tier": "routine",
            "existing_parents": [],
            "new_parents": [],
            "owned_paths": ["src"],
        }],
        idempotency_key="steward-scope-no-repo",
    )
    with pytest.raises(ow.OwnerWorkspaceError) as excinfo:
        _commit_project_plan(ctx, **args)
    assert excinfo.value.code == "ownership_scope_unavailable"


def test_a_read_only_scope_needs_no_project_repository(ctx):
    setup = _bootstrap_board(ctx)
    approver = _with_approver(ctx.session)
    result = _commit_project_plan(
        ctx, **_project_plan_args(
            setup,
            [{
                "action": "add",
                "reason": "Add one explicitly read-only task.",
                "title": "Read-only investigation",
                "body": "Report findings; change nothing.",
                "assignee": "default",
                "execution_tier": "routine",
                "existing_parents": [],
                "new_parents": [],
                "owned_paths": [],
            }],
            idempotency_key="steward-scope-readonly",
        )
    )
    approver.join()
    assert result["ok"] is True
    with kanban_db.connect(board=setup["board"]) as conn:
        assert kanban_db.get_task(
            conn, result["created_task_ids"][0]
        ).owned_paths == []


def test_verifier_project_work_defaults_to_explicit_read_only(ctx):
    _install_profiles("raphael-verifier")
    setup = _bootstrap_board(ctx)
    approver = _with_approver(ctx.session)
    with _temporarily_patch(
        model_policy, "configured_assignment_for", _configured_raphael_role,
    ):
        result = _commit_project_plan(
            ctx, **_project_plan_args(
                setup,
                [{
                    "action": "add",
                    "reason": "Run one independent read-only review.",
                    "title": "Verify the release",
                    "body": "Inspect and report; change nothing.",
                    "assignee": "raphael-verifier",
                    "responsibility": "R12",
                    "execution_tier": "deep",
                    "existing_parents": [],
                    "new_parents": [],
                }],
                idempotency_key="verifier-default-readonly",
            )
        )
    approver.join()

    assert result["ok"] is True
    with kanban_db.connect(board=setup["board"]) as conn:
        task = kanban_db.get_task(conn, result["created_task_ids"][0])
        assert task.assignee == "raphael-verifier"
        assert task.owned_paths == []


def test_verifier_project_work_rejects_a_mutating_scope(ctx):
    _install_profiles("raphael-verifier")
    setup = _bootstrap_board(ctx)
    args = _project_plan_args(
        setup,
        [{
            "action": "add",
            "reason": "A reviewer must not own source writes.",
            "title": "Unsafe verifier work",
            "body": "This must be refused before approval.",
            "assignee": "raphael-verifier",
            "responsibility": "R12",
            "execution_tier": "deep",
            "existing_parents": [],
            "new_parents": [],
            "owned_paths": ["src"],
        }],
        idempotency_key="verifier-mutating-scope-refused",
    )

    with pytest.raises(ow.OwnerWorkspaceError) as excinfo:
        _commit_project_plan(ctx, **args)
    assert excinfo.value.code == "invalid_ownership_scope"


def test_task_graph_carries_explicit_scopes_into_committed_children(ctx, tmp_path):
    setup = _bootstrap_board(ctx)
    repo = _project_repo(tmp_path, ctx, setup["project_id"])
    approver = _with_approver(ctx.session)
    result = _commit_task_graph(
        ctx,
        idempotency_key="graph-scoped-milestone",
        mode="existing",
        project_id=setup["project_id"],
        request_title="Deliver the scoped milestone",
        specification="Two workers own disjoint subtrees.",
        current_milestone="Split the delivery by ownership.",
        owner_visible_result="Both subtrees change without conflicting.",
        root_assignee="default",
        tasks=[
            {
                "title": "Own the api subtree",
                "body": "Change only src/api.",
                "assignee": "default",
                "responsibility": "R11",
                "execution_tier": "routine",
                "parents": [],
                "owned_paths": ["src/api"],
            },
            {
                "title": "Own the web subtree",
                "body": "Change only src/web.",
                "assignee": "default",
                "responsibility": "R12",
                "execution_tier": "routine",
                "parents": [],
                "owned_paths": ["src/web"],
            },
        ],
    )
    approver.join()

    assert result["ok"] is True
    with kanban_db.connect(board=setup["board"]) as conn:
        children = [
            kanban_db.get_task(conn, task_id) for task_id in result["task_ids"]
        ]
    assert [child.owned_paths for child in children] == [["src/api"], ["src/web"]]
    for child in children:
        assert child.workspace_kind == "worktree"
        assert child.workspace_path == str(repo / ".worktrees" / child.id)
        assert child.model_policy_lock


def test_task_graph_defaults_verifier_children_to_read_only(ctx, tmp_path):
    _install_profiles("raphael-verifier")
    setup = _bootstrap_board(ctx)
    _project_repo(tmp_path, ctx, setup["project_id"])
    approver = _with_approver(ctx.session)
    with _temporarily_patch(
        model_policy, "configured_assignment_for", _configured_raphael_role,
    ):
        result = _commit_task_graph(
            ctx,
            idempotency_key="graph-verifier-readonly",
            mode="existing",
            project_id=setup["project_id"],
            request_title="Verify the milestone",
            specification="One independent reviewer reads the result.",
            current_milestone="Review without source ownership.",
            owner_visible_result="The reviewer reports a verdict.",
            root_assignee="default",
            tasks=[{
                "title": "Review the exact result",
                "body": "Inspect, test and report without importing changes.",
                "assignee": "raphael-verifier",
                "responsibility": "R12",
                "execution_tier": "deep",
                "parents": [],
            }],
        )
    approver.join()

    with kanban_db.connect(board=setup["board"]) as conn:
        task = kanban_db.get_task(conn, result["task_ids"][0])
        assert task.assignee == "raphael-verifier"
        assert task.owned_paths == []


def test_task_graph_scope_is_never_inferred_from_the_assignee(ctx, tmp_path):
    # The remote coding worker is the profile most likely to have a boundary
    # guessed for it, so it is the one this asserts nothing is guessed from.
    _install_profiles("raphael-claude-worker")
    setup = _bootstrap_board(ctx)
    _project_repo(tmp_path, ctx, setup["project_id"])
    approver = _with_approver(ctx.session)
    result = _commit_task_graph(
        ctx,
        idempotency_key="graph-unscoped-milestone",
        mode="existing",
        project_id=setup["project_id"],
        request_title="Deliver the unscoped milestone",
        specification="No boundary is declared for either task.",
        current_milestone="Keep the historical boundary.",
        owner_visible_result="The tasks behave exactly as before.",
        root_assignee="default",
        tasks=[{
            "title": "Do the work",
            "body": "No boundary is declared.",
            "assignee": "raphael-claude-worker",
            "responsibility": "R13",
            "execution_tier": "routine",
            "parents": [],
        }],
    )
    approver.join()
    with kanban_db.connect(board=setup["board"]) as conn:
        assert kanban_db.get_task(
            conn, result["task_ids"][0]
        ).owned_paths is None


def test_a_new_project_cannot_declare_a_mutating_scope(ctx):
    with pytest.raises(ow.OwnerWorkspaceError) as excinfo:
        _commit_task_graph(
            ctx,
            idempotency_key="graph-new-scoped",
            mode="new",
            project_name="Brand New Project",
            request_title="Deliver something scoped",
            specification="A new Project has no repository yet.",
            current_milestone="Refuse before asking the owner.",
            owner_visible_result="Nothing is created.",
            root_assignee="default",
            tasks=[{
                "title": "Scoped work in a repo-less project",
                "body": "There is no repository to scope.",
                "assignee": "default",
                "responsibility": "R14",
                "execution_tier": "routine",
                "parents": [],
                "owned_paths": ["src"],
            }],
        )
    assert excinfo.value.code == "ownership_scope_unavailable"


def test_project_plan_split_is_atomic_preserves_history_and_replays(ctx):
    setup = _bootstrap_board(ctx)
    with kanban_db.connect(board=setup["board"]) as conn:
        source_id = kanban_db.create_task(
            conn, title="Broad task", assignee="default", project_id=setup["project_id"],
        )
        downstream_id = kanban_db.create_task(
            conn,
            title="Verify the outcome",
            assignee="default",
            parents=[source_id],
            project_id=setup["project_id"],
        )
        source_ref = _project_task_ref(conn, source_id)

    args = _project_plan_args(
        setup,
        [{
            "action": "split",
            "reason": "The current task is too broad to verify safely.",
            "target": source_ref,
            "replacements": [
                {
                    "title": "Build the bounded change",
                    "body": "Produce one owner-visible outcome.",
                    "assignee": "default",
                    "execution_tier": "routine",
                    "responsibility": "B04",
                    "parents": [],
                },
                {
                    "title": "Check the bounded change",
                    "body": "Verify the outcome before downstream work continues.",
                    "assignee": "default",
                    "execution_tier": "routine",
                    "responsibility": "R12",
                    "parents": [0],
                },
            ],
        }],
    )
    approver = _with_approver(ctx.session)
    first = _commit_project_plan(ctx, **args)
    approver.join()

    assert first["ok"] is True
    assert first["change_count"] == 1
    assert len(first["created_task_ids"]) == 2
    first_id, leaf_id = first["created_task_ids"]

    with kanban_db.connect(board=setup["board"]) as conn:
        assert kanban_db.get_task(conn, source_id).status == "archived"
        assert kanban_db.get_task(conn, source_id) is not None
        assert kanban_db.parent_ids(conn, leaf_id) == [first_id]
        assert leaf_id in kanban_db.parent_ids(conn, downstream_id)
        assert kanban_db.get_task(conn, first_id).responsibility == "B04"
        assert kanban_db.get_task(conn, leaf_id).responsibility == "R12"
        before = conn.execute(
            "SELECT COUNT(*) AS n FROM tasks WHERE project_id = ?",
            (setup["project_id"],),
        ).fetchone()["n"]

    approval.unregister_gateway_notify(ctx.session)
    replay = _commit_project_plan(ctx, **args)
    assert replay == first

    with kanban_db.connect(board=setup["board"]) as conn:
        after = conn.execute(
            "SELECT COUNT(*) AS n FROM tasks WHERE project_id = ?",
            (setup["project_id"],),
        ).fetchone()["n"]
    assert after == before


def test_project_plan_add_move_and_postpone_apply_together(ctx):
    setup = _bootstrap_board(ctx)
    with kanban_db.connect(board=setup["board"]) as conn:
        parent_ref = _project_task_ref(conn, setup["task_id"])
        move_id = kanban_db.create_task(
            conn, title="Move me", assignee="default", project_id=setup["project_id"],
        )
        move_revision = kanban_db.task_event_revision(conn, move_id)
        moved = kanban_db.cas_transition_task(
            conn,
            move_id,
            expected_status="ready",
            expected_revision=move_revision,
            to_status="blocked",
            event_kind="test_block",
        )
        assert moved["moved"] is True
        postpone_id = kanban_db.create_task(
            conn, title="Postpone me", assignee="default", project_id=setup["project_id"],
        )
        move_ref = _project_task_ref(conn, move_id)
        postpone_ref = _project_task_ref(conn, postpone_id)

    args = _project_plan_args(
        setup,
        [
            {
                "action": "add",
                "reason": "Create the first bounded deliverable.",
                "title": "Prepare the deliverable",
                "body": "Produce the owner-visible input.",
                "assignee": "default",
                "execution_tier": "routine",
                "existing_parents": [parent_ref],
                "new_parents": [],
            },
            {
                "action": "add",
                "reason": "Verify the new deliverable.",
                "title": "Verify the deliverable",
                "body": "Check the preceding result.",
                "assignee": "default",
                "execution_tier": "routine",
                "existing_parents": [],
                "new_parents": [0],
            },
            {
                "action": "move",
                "reason": "Return this work to the planned queue.",
                "target": move_ref,
                "to_status": "ready",
            },
            {
                "action": "postpone",
                "reason": "Keep future work outside the executable milestone.",
                "target": postpone_ref,
            },
        ],
        idempotency_key="steward-standard-actions",
    )
    approver = _with_approver(ctx.session)
    result = _commit_project_plan(ctx, **args)
    approver.join()

    assert result["ok"] is True
    assert result["change_count"] == 4
    assert len(result["created_task_ids"]) == 2
    assert kanban_db.board_dispatch_allowed(
        kanban_db.read_board_metadata(setup["board"])
    ) is True
    first_id, second_id = result["created_task_ids"]
    with kanban_db.connect(board=setup["board"]) as conn:
        assert kanban_db.parent_ids(conn, first_id) == [setup["task_id"]]
        assert kanban_db.parent_ids(conn, second_id) == [first_id]
        assert kanban_db.get_task(conn, move_id).status == "ready"
        assert any(
            comment.body == "Return this work to the planned queue."
            for comment in kanban_db.list_comments(conn, move_id)
        )
        assert kanban_db.get_task(conn, postpone_id).status == "scheduled"
        assert any(
            event.kind == "owner_project_plan_applied"
            for event in kanban_db.list_events(
                conn, setup["task_id"], include_control=True,
            )
        )


def test_project_plan_merge_archives_sources_and_preserves_links(ctx):
    setup = _bootstrap_board(ctx)
    with kanban_db.connect(board=setup["board"]) as conn:
        left_id = kanban_db.create_task(
            conn, title="Left half", assignee="default", project_id=setup["project_id"],
        )
        right_id = kanban_db.create_task(
            conn, title="Right half", assignee="default", project_id=setup["project_id"],
        )
        child_id = kanban_db.create_task(
            conn,
            title="Dependent work",
            assignee="default",
            parents=[left_id, right_id],
            project_id=setup["project_id"],
        )
        left_ref = _project_task_ref(conn, left_id)
        right_ref = _project_task_ref(conn, right_id)

    args = _project_plan_args(
        setup,
        [{
            "action": "merge",
            "reason": "One coherent deliverable is easier to own and verify.",
            "targets": [left_ref, right_ref],
            "replacement": {
                "title": "Combined deliverable",
                "body": "Replace both overlapping work items without deleting history.",
                "assignee": "default",
                "execution_tier": "routine",
            },
        }],
        idempotency_key="steward-merge",
    )
    approver = _with_approver(ctx.session)
    result = _commit_project_plan(ctx, **args)
    approver.join()

    assert result["ok"] is True
    assert result["risk_level"] == "significant_removal"
    replacement_id = result["created_task_ids"][0]
    with kanban_db.connect(board=setup["board"]) as conn:
        assert kanban_db.get_task(conn, left_id).status == "archived"
        assert kanban_db.get_task(conn, right_id).status == "archived"
        assert set(kanban_db.parent_ids(conn, child_id)) == {
            left_id,
            right_id,
            replacement_id,
        }


def test_project_plan_cancel_archives_instead_of_deleting(ctx):
    setup = _bootstrap_board(ctx)
    with kanban_db.connect(board=setup["board"]) as conn:
        target_id = kanban_db.create_task(
            conn, title="Obsolete work", assignee="default", project_id=setup["project_id"],
        )
        target_ref = _project_task_ref(conn, target_id)

    args = _project_plan_args(
        setup,
        [{
            "action": "cancel",
            "reason": "The owner explicitly removed this outcome.",
            "target": target_ref,
        }],
        idempotency_key="steward-cancel",
    )
    approver = _with_approver(ctx.session)
    result = _commit_project_plan(ctx, **args)
    approver.join()

    assert result["ok"] is True
    assert result["risk_level"] == "significant_removal"
    with kanban_db.connect(board=setup["board"]) as conn:
        task = kanban_db.get_task(conn, target_id)
        assert task is not None
        assert task.status == "archived"


def test_project_plan_stale_snapshot_changes_nothing(ctx):
    setup = _bootstrap_board(ctx)
    with kanban_db.connect(board=setup["board"]) as conn:
        target_id = kanban_db.create_task(
            conn, title="Current task", assignee="default", project_id=setup["project_id"],
        )
        stale_ref = _project_task_ref(conn, target_id)
        kanban_db.add_comment(conn, target_id, "default", "State changed after planning.")
        before_events = conn.execute("SELECT COUNT(*) AS n FROM task_events").fetchone()["n"]

    args = _project_plan_args(
        setup,
        [{
            "action": "postpone",
            "reason": "Wait for the owner-visible dependency.",
            "target": stale_ref,
        }],
        idempotency_key="steward-stale",
    )
    approver = _with_approver(ctx.session)
    result = _commit_project_plan(ctx, **args)
    approver.join()

    assert result == {
        "ok": False,
        "error": "conflict",
        "project_id": setup["project_id"],
        "project_slug": "board-setup",
        "change_count": 0,
        "anchor_task_id": setup["task_id"],
    }
    with kanban_db.connect(board=setup["board"]) as conn:
        assert kanban_db.get_task(conn, target_id).status == "ready"
        assert conn.execute("SELECT COUNT(*) AS n FROM task_events").fetchone()["n"] == before_events


def test_project_plan_merge_or_cancel_requires_its_own_owner_decision(ctx):
    setup = _bootstrap_board(ctx)
    with kanban_db.connect(board=setup["board"]) as conn:
        target = _project_task_ref(conn, setup["task_id"])
    args = _project_plan_args(
        setup,
        [
            {"action": "cancel", "reason": "No longer useful.", "target": target},
            {
                "action": "add",
                "reason": "Unrelated work.",
                "title": "Another task",
                "body": "This must be approved separately.",
                "assignee": "default",
                "execution_tier": "routine",
                "existing_parents": [],
                "new_parents": [],
            },
        ],
        idempotency_key="steward-mixed-removal",
    )
    with pytest.raises(ow.OwnerWorkspaceError) as excinfo:
        _commit_project_plan(ctx, **args)
    assert excinfo.value.code == "separate_owner_decision"


def test_project_plan_rejects_ready_when_parent_is_unfinished_and_rolls_back(ctx):
    setup = _bootstrap_board(ctx)
    with kanban_db.connect(board=setup["board"]) as conn:
        child_id = kanban_db.create_task(
            conn,
            title="Waiting child",
            assignee="default",
            parents=[setup["task_id"]],
            project_id=setup["project_id"],
        )
        child_ref = _project_task_ref(conn, child_id)
        before_events = conn.execute("SELECT COUNT(*) AS n FROM task_events").fetchone()["n"]
    args = _project_plan_args(
        setup,
        [{
            "action": "move",
            "reason": "Try to start early.",
            "target": child_ref,
            "to_status": "ready",
        }],
        idempotency_key="steward-parent-gate",
    )
    approver = _with_approver(ctx.session)
    with pytest.raises(ValueError, match="parent is unfinished"):
        _commit_project_plan(ctx, **args)
    approver.join()
    with kanban_db.connect(board=setup["board"]) as conn:
        assert kanban_db.get_task(conn, child_id).status == "todo"
        assert conn.execute("SELECT COUNT(*) AS n FROM task_events").fetchone()["n"] == before_events


def test_move_task_cas_conflict_returns_snapshot_with_zero_changes(ctx):
    setup = _bootstrap_board(ctx)
    board = setup["board"]
    task_id = setup["task_id"]

    kconn = kanban_db.connect(board=board)
    try:
        rev = kanban_db.task_event_revision(kconn, task_id)
    finally:
        kconn.close()

    t = _with_approver(ctx.session)
    result = ow.move_task(
        ctx, idempotency_key="move-1", task_id=task_id,
        to_status="blocked", expected_status="WRONG", expected_revision=rev, project_id=setup["project_id"],
    )
    t.join()
    assert result["ok"] is False
    assert result["error"] == "conflict"
    assert result["current_status"] == _ANCHOR_STATUS

    kconn = kanban_db.connect(board=board)
    try:
        task = kanban_db.get_control_task(kconn, task_id)
    finally:
        kconn.close()
    assert task.status == _ANCHOR_STATUS


def test_move_task_rejects_running_target(ctx):
    setup = _bootstrap_board(ctx)
    with pytest.raises(ow.OwnerWorkspaceError) as excinfo:
        ow.move_task(
            ctx, idempotency_key="move-running", task_id=setup["task_id"],
            to_status="running", expected_status=_ANCHOR_STATUS, expected_revision=1, project_id=setup["project_id"],
        )
    assert excinfo.value.code == "unsafe_transition"


def test_move_task_success_and_archived_parent_satisfies_child_readiness(ctx):
    setup = _bootstrap_board(ctx)
    board = setup["board"]
    parent_id = setup["task_id"]

    kconn = kanban_db.connect(board=board)
    try:
        child_id = kanban_db.create_task(kconn, title="child", parents=[parent_id])
        assert kanban_db.get_task(kconn, child_id).status == "todo"
        rev = kanban_db.task_event_revision(kconn, parent_id)
    finally:
        kconn.close()

    t = _with_approver(ctx.session)
    result = ow.move_task(
        ctx, idempotency_key="move-archive", task_id=parent_id,
        to_status="archived", expected_status=_ANCHOR_STATUS, expected_revision=rev,
        project_id=setup["project_id"],
    )
    t.join()
    assert result["ok"] is True
    assert result["status"] == "archived"

    kconn = kanban_db.connect(board=board)
    try:
        assert kanban_db.get_task(kconn, child_id).status == "ready"
    finally:
        kconn.close()


def test_move_task_approval_cannot_mutate_a_project_archived_while_waiting(ctx):
    setup = _bootstrap_board(ctx)
    with kanban_db.connect(board=setup["board"]) as conn:
        revision = kanban_db.task_event_revision(conn, setup["task_id"])

    def archive_then_approve(*_args, **_kwargs):
        with projects_db.connect_closing() as pconn:
            with ow.write_txn(pconn):
                pconn.execute(
                    "UPDATE projects SET archived = 1 WHERE id = ?",
                    (setup["project_id"],),
                )
        return {"approved": True, "choice": "once"}

    with _temporarily_patch(ow, "_confirm", archive_then_approve):
        result = ow.move_task(
            ctx,
            idempotency_key="move-archive-race",
            task_id=setup["task_id"],
            to_status="blocked",
            expected_status=_ANCHOR_STATUS,
            expected_revision=revision,
            project_id=setup["project_id"],
        )

    assert result["ok"] is False
    assert result["error"] == "conflict"
    with kanban_db.connect(board=setup["board"]) as conn:
        assert kanban_db.get_control_task(
            conn, setup["task_id"]
        ).status == _ANCHOR_STATUS


def test_comment_author_is_trusted_context_not_caller_supplied(ctx):
    setup = _bootstrap_board(ctx)
    board = setup["board"]
    task_id = setup["task_id"]

    t = _with_approver(ctx.session)
    result = ow.comment_task(ctx, idempotency_key="comment-1", task_id=task_id, body="hello", project_id=setup["project_id"])
    t.join()
    assert result["ok"] is True

    kconn = kanban_db.connect(board=board)
    try:
        comments = kanban_db.list_comments(kconn, task_id, include_control=True)
    finally:
        kconn.close()
    assert len(comments) == 1
    assert comments[0].author == ctx.actor
    assert comments[0].body == "hello"


def test_comment_approval_cannot_write_to_a_project_archived_while_waiting(ctx):
    setup = _bootstrap_board(ctx)

    def archive_then_approve(*_args, **_kwargs):
        with projects_db.connect_closing() as pconn:
            with ow.write_txn(pconn):
                pconn.execute(
                    "UPDATE projects SET archived = 1 WHERE id = ?",
                    (setup["project_id"],),
                )
        return {"approved": True, "choice": "once"}

    with _temporarily_patch(ow, "_confirm", archive_then_approve):
        result = ow.comment_task(
            ctx,
            idempotency_key="comment-archive-race",
            task_id=setup["task_id"],
            body="must not be added",
            project_id=setup["project_id"],
        )

    assert result["ok"] is False
    assert result["error"] == "conflict"
    with kanban_db.connect(board=setup["board"]) as conn:
        assert all(
            comment.body != "must not be added"
            for comment in kanban_db.list_comments(
                conn, setup["task_id"], include_control=True,
            )
        )


def test_comment_exact_replay_creates_no_duplicate(ctx):
    setup = _bootstrap_board(ctx)
    board = setup["board"]
    task_id = setup["task_id"]

    t = _with_approver(ctx.session)
    first = ow.comment_task(ctx, idempotency_key="comment-replay", task_id=task_id, body="once", project_id=setup["project_id"])
    t.join()

    approval.unregister_gateway_notify(ctx.session)
    second = ow.comment_task(ctx, idempotency_key="comment-replay", task_id=task_id, body="once", project_id=setup["project_id"])
    assert second == first

    kconn = kanban_db.connect(board=board)
    try:
        comments = kanban_db.list_comments(kconn, task_id, include_control=True)
    finally:
        kconn.close()
    assert len(comments) == 1


def test_multiplexed_profile_binds_request_profile_not_process_default(tmp_path, monkeypatch):
    """resolve_owner_context() must reflect a per-request HERMES_HOME override,
    not the process's default profile — the exact property multiplexed
    /p/<profile>/ requests depend on.
    """
    from hermes_constants import set_hermes_home_override, reset_hermes_home_override

    other_home = tmp_path / "other_profile_home"
    other_home.mkdir()
    profiles_root = other_home.parent / "profiles"
    profiles_root.mkdir(parents=True, exist_ok=True)
    named = profiles_root / "coder"
    named.mkdir()
    monkeypatch.setattr("hermes_cli.profiles._get_profiles_root", lambda: profiles_root)

    token = set_hermes_home_override(str(named))
    try:
        bound = ow.resolve_owner_context()
    finally:
        reset_hermes_home_override(token)

    assert bound.profile == "coder"
    assert bound.actor == "coder"


# ---------------------------------------------------------------------------
# Enforced receipt lease ownership (lock_token)
# ---------------------------------------------------------------------------


def test_lease_takeover_old_token_cannot_mutate_progress_or_finalize(ctx):
    """The verifier's takeover regression: once an expired lease is adopted
    under a NEW token, the OLD claimant's stale token must fail closed on
    every subsequent write — it can never overwrite progress or finalize on
    the new claimant's behalf."""
    key = "takeover-1"
    digest = ow._digest({"name": "Takeover WS", "description": None})
    operation = "owner_workspace_bootstrap"

    with projects_db.connect_closing() as pconn:
        ow._ensure_schema(pconn)

        state, row, old_token = ow._claim_or_wait(pconn, ctx, key, operation, digest)
        assert state == "own"
        assert row is None
        assert old_token

        # Simulate the old claimant crashing: its lease's TTL has passed.
        _expire_lock(ctx, key)

        # A second claimant adopts the dead lock — mints a NEW token.
        state2, row2, new_token = ow._claim_or_wait(pconn, ctx, key, operation, digest)
        assert state2 == "own"
        assert new_token != old_token

        # The OLD claimant — still holding its now-stale token — must be
        # refused everywhere: pre-mutation lease check, progress write, and
        # finalization.
        with pytest.raises(ow.OwnerWorkspaceError) as excinfo:
            ow._assert_owns_lease(pconn, ctx, key, old_token)
        assert excinfo.value.code == "lease_lost"

        with pytest.raises(ow.OwnerWorkspaceError) as excinfo:
            ow._update_progress(pconn, ctx, key, old_token, project_id="p_stolen")
        assert excinfo.value.code == "lease_lost"

        with pytest.raises(ow.OwnerWorkspaceError) as excinfo:
            ow._finalize_receipt(
                pconn, ctx, key, old_token, status="committed",
                result={"ok": True, "via": "old_claimant"},
            )
        assert excinfo.value.code == "lease_lost"

        # The row must be untouched by any of the old claimant's attempts —
        # still in_progress, still owned by the NEW token.
        row3 = ow._get_receipt(pconn, ctx, key)
        assert row3["status"] == "in_progress"
        assert row3["lock_token"] == new_token

        # The NEW claimant's token still works normally.
        ow._assert_owns_lease(pconn, ctx, key, new_token)
        ow._finalize_receipt(
            pconn, ctx, key, new_token, status="committed",
            result={"ok": True, "via": "new_claimant"},
        )
        row4 = ow._get_receipt(pconn, ctx, key)
        assert row4["status"] == "committed"
        assert json.loads(row4["result_json"])["via"] == "new_claimant"


# ---------------------------------------------------------------------------
# Real lease fencing: no check-then-act gap between validation and mutation
# ---------------------------------------------------------------------------


def test_move_task_fence_blocks_concurrent_claim_attempt_during_mutation(ctx):
    """Two-connection/thread interleaving regression for the TOCTOU the fence
    closes: a claimant that has already passed `_assert_owns_lease` and is
    mid-mutation must not leave a window where a second connection can act on
    the SAME receipt row. The fence (`with write_txn(pconn):` wrapping both
    the lease check and the domain mutation) holds `projects.db`'s write lock
    for the whole span, so a concurrent claim attempt against the same row
    (which needs that same lock) is provably blocked until this mutation
    either finishes (proving "the fenced mutation completes before takeover")
    or the fence's own transaction were to lose the race and see a stale
    token first (proving "the stale claimant fails before mutation" — see
    `test_lease_takeover_old_token_cannot_mutate_progress_or_finalize` for
    that half, exercised sequentially since a real DB mutex makes the two
    outcomes mutually exclusive by construction).
    """
    setup = _bootstrap_board(ctx)
    board, task_id = setup["board"], setup["task_id"]

    kconn = kanban_db.connect(board=board)
    try:
        rev = kanban_db.task_event_revision(kconn, task_id)
    finally:
        kconn.close()

    key = "fence-race-1"
    validated = threading.Event()
    release = threading.Event()
    real_cas = kanban_db.cas_transition_task

    def paused_cas(*a, **kw):
        # Reached only AFTER `_assert_owns_lease` succeeded, and only while
        # still inside the fence's `with write_txn(pconn):` block — i.e.
        # still holding projects.db's write lock. This is the exact
        # validate-then-pause window the bug report describes.
        validated.set()
        release.wait(timeout=5.0)
        return real_cas(*a, **kw)

    old_result = {}

    def run_old_claimant():
        t = _with_approver(ctx.session)
        with _temporarily_patch(ow.kanban_db, "cas_transition_task", paused_cas):
            old_result["value"] = ow.move_task(
                ctx, idempotency_key=key, task_id=task_id, to_status="blocked",
                expected_status=_ANCHOR_STATUS, expected_revision=rev,
                project_id=setup["project_id"],
            )
        t.join()

    old_thread = threading.Thread(target=run_old_claimant)
    old_thread.start()
    assert validated.wait(timeout=5.0), "old claimant never reached the paused mutation"

    # A second, independent connection now races the still-open fence for
    # the SAME receipt row — exactly what an expired-lease adopter would do.
    takeover_state = {}

    def attempt_takeover():
        with projects_db.connect_closing() as pconn2:
            ow._ensure_schema(pconn2)
            digest = ow._digest({
                "project_id": setup["project_id"], "task_id": task_id, "to_status": "blocked",
                "expected_status": _ANCHOR_STATUS, "expected_revision": rev,
            })
            # A real competing caller goes through the bounded-poll wrapper
            # (not the raw single-shot primitive) — it may transiently
            # observe "wait" (a live, not-yet-finalized claim) right after
            # the fence releases and before finalization lands, but must
            # never observe "own" (a successful adoption): that would mean
            # it stole the lease mid-mutation.
            state, _row, _token = ow._acquire_or_replay(pconn2, ctx, key, "owner_task_move", digest)
            takeover_state["state"] = state

    takeover_thread = threading.Thread(target=attempt_takeover)
    takeover_thread.start()

    # A generous, deterministic bound (not a tight race): the takeover
    # attempt's own `write_txn` needs the exact write lock the paused fence
    # holds, so it MUST still be blocked after a real 2-second wait.
    takeover_thread.join(timeout=2.0)
    assert takeover_thread.is_alive(), (
        "a concurrent claim attempt must block on projects.db's write lock "
        "while the fenced mutation is in flight — it must not be able to "
        "observe or act on the row until the fence releases"
    )

    # Let the old (fence-holding) claimant finish its mutation and finalize.
    release.set()
    old_thread.join(timeout=5.0)
    takeover_thread.join(timeout=5.0)
    assert not takeover_thread.is_alive()

    assert old_result["value"]["ok"] is True
    assert old_result["value"]["status"] == "blocked"

    # The takeover attempt — unblocked only after the fence released — must
    # see the row already terminal (committed by the fenced mutation), never
    # a live claim to adopt: the fenced mutation provably completed before
    # any competitor could act on the row.
    assert takeover_state["state"] == "terminal"

    kconn = kanban_db.connect(board=board)
    try:
        task = kanban_db.get_control_task(kconn, task_id)
    finally:
        kconn.close()
    assert task.status == "blocked"


def test_comment_fence_blocks_concurrent_claim_attempt_during_mutation(ctx):
    """Same fencing property as the move-task regression above, exercised
    against `comment_task`'s fence (the lease check + `add_comment` insert)."""
    setup = _bootstrap_board(ctx)
    board, task_id = setup["board"], setup["task_id"]

    key = "fence-race-comment-1"
    validated = threading.Event()
    release = threading.Event()
    real_add_comment = kanban_db.add_comment

    def paused_add_comment(*a, **kw):
        validated.set()
        release.wait(timeout=5.0)
        return real_add_comment(*a, **kw)

    old_result = {}

    def run_old_claimant():
        t = _with_approver(ctx.session)
        with _temporarily_patch(ow.kanban_db, "add_comment", paused_add_comment):
            old_result["value"] = ow.comment_task(
                ctx, idempotency_key=key, task_id=task_id, body="hi", project_id=setup["project_id"],
            )
        t.join()

    old_thread = threading.Thread(target=run_old_claimant)
    old_thread.start()
    assert validated.wait(timeout=5.0), "old claimant never reached the paused mutation"

    takeover_state = {}

    def attempt_takeover():
        with projects_db.connect_closing() as pconn2:
            ow._ensure_schema(pconn2)
            digest = ow._digest({
                "project_id": setup["project_id"], "task_id": task_id, "body": "hi",
            })
            # See the move-task fence regression above for why this uses the
            # bounded-poll wrapper rather than the raw single-shot primitive.
            state, _row, _token = ow._acquire_or_replay(pconn2, ctx, key, "owner_task_comment", digest)
            takeover_state["state"] = state

    takeover_thread = threading.Thread(target=attempt_takeover)
    takeover_thread.start()
    takeover_thread.join(timeout=2.0)
    assert takeover_thread.is_alive(), (
        "a concurrent claim attempt must block while the comment's fenced "
        "mutation is in flight"
    )

    release.set()
    old_thread.join(timeout=5.0)
    takeover_thread.join(timeout=5.0)
    assert not takeover_thread.is_alive()

    assert old_result["value"]["ok"] is True
    assert takeover_state["state"] == "terminal"

    kconn = kanban_db.connect(board=board)
    try:
        comments = kanban_db.list_comments(kconn, task_id, include_control=True)
    finally:
        kconn.close()
    assert len(comments) == 1


# ---------------------------------------------------------------------------
# Crash-safe bootstrap roll-forward (deterministic identities)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("crash_point", ["after_project", "after_board", "after_task"])
def test_bootstrap_crash_at_each_boundary_yields_exactly_one_project_board_task(
    ctx, crash_point,
):
    key = f"crash-{crash_point}"
    name = f"Crash WS {crash_point}"

    def boom(*a, **kw):
        raise RuntimeError(f"simulated crash: {crash_point}")

    patched_attr = {
        "after_project": "create_board",
        "after_board": "create_task",
        "after_task": "task_event_revision",
    }[crash_point]

    with _temporarily_patch(ow.kanban_db, patched_attr, boom):
        t = _with_approver(ctx.session)
        with pytest.raises(RuntimeError):
            ow.bootstrap(ctx, idempotency_key=key, name=name)
        t.join()

    _expire_lock(ctx, key)

    t2 = _with_approver(ctx.session)
    result = ow.bootstrap(ctx, idempotency_key=key, name=name)
    t2.join()
    assert result["ok"] is True

    with projects_db.connect_closing() as pconn:
        matching_projects = [
            p for p in projects_db.list_projects(pconn, include_archived=True) if p.name == name
        ]
    assert len(matching_projects) == 1
    assert matching_projects[0].id == result["project_id"]

    matching_boards = [b for b in kanban_db.list_boards() if b.get("name") == name]
    assert len(matching_boards) == 1

    task_idempotency_key = "owtask_" + ow._derive_id(ctx, key, "task")
    kconn = kanban_db.connect(board=result["board"])
    try:
        matching_tasks = kconn.execute(
            "SELECT id FROM tasks WHERE idempotency_key = ?", (task_idempotency_key,),
        ).fetchall()
    finally:
        kconn.close()
    assert len(matching_tasks) == 1
    assert matching_tasks[0]["id"] == result["task_id"]


def test_bootstrap_replay_fails_closed_on_foreign_board_ownership(ctx):
    """A board slug that already exists but is owned by a DIFFERENT project
    (foreign/ambiguous ownership metadata) must never be silently adopted."""
    key = "foreign-board-1"
    name = "Foreign Board WS"

    def boom(*a, **kw):
        raise RuntimeError("simulated crash before project row commits its board_slug")

    with _temporarily_patch(ow.kanban_db, "create_task", boom):
        t = _with_approver(ctx.session)
        with pytest.raises(RuntimeError):
            ow.bootstrap(ctx, idempotency_key=key, name=name)
        t.join()

    project_id = "p_" + ow._derive_id(ctx, key, "project")
    with projects_db.connect_closing() as pconn:
        project = projects_db.get_project(pconn, project_id)
    board_slug = project.slug

    # Corrupt the board's metadata to claim a foreign owner.
    kanban_db.write_board_metadata(board_slug, project_id="p_someone_else")

    _expire_lock(ctx, key)
    t2 = _with_approver(ctx.session)
    with pytest.raises(ow.OwnerWorkspaceError) as excinfo:
        ow.bootstrap(ctx, idempotency_key=key, name=name)
    t2.join()
    assert excinfo.value.code == "crash_recovery_failed"


def test_bootstrap_replay_fails_closed_on_missing_board_ownership(ctx):
    """A board slug that already exists but carries NO ownership metadata
    (never linked to any project) must fail closed exactly like a foreign
    owner. This flow's own boards always carry ``project_id`` — an existing
    board without one can never be safely attributed to this receipt."""
    key = "missing-board-owner-1"
    name = "Missing Board Owner WS"

    def boom(*a, **kw):
        raise RuntimeError("simulated crash before project row commits its board_slug")

    with _temporarily_patch(ow.kanban_db, "create_task", boom):
        t = _with_approver(ctx.session)
        with pytest.raises(RuntimeError):
            ow.bootstrap(ctx, idempotency_key=key, name=name)
        t.join()

    project_id = "p_" + ow._derive_id(ctx, key, "project")
    with projects_db.connect_closing() as pconn:
        project = projects_db.get_project(pconn, project_id)
    board_slug = project.slug

    # Wipe the board's ownership metadata (missing, not foreign).
    kanban_db.write_board_metadata(board_slug, project_id="")
    assert kanban_db.read_board_metadata(board_slug).get("project_id") is None

    _expire_lock(ctx, key)
    t2 = _with_approver(ctx.session)
    with pytest.raises(ow.OwnerWorkspaceError) as excinfo:
        ow.bootstrap(ctx, idempotency_key=key, name=name)
    t2.join()
    assert excinfo.value.code == "crash_recovery_failed"


def test_bootstrap_replay_fails_closed_on_missing_task_ownership(ctx):
    """A pre-existing task resolved via the deterministic idempotency key but
    carrying NO ``project_id`` must fail closed exactly like a foreign owner.
    This flow's own tasks always carry ``project_id`` — an existing task
    without one can never be safely attributed to this receipt."""
    key = "missing-task-owner-1"
    name = "Missing Task Owner WS"

    def boom(*a, **kw):
        raise RuntimeError("simulated crash before the task is created")

    with _temporarily_patch(ow.kanban_db, "create_task", boom):
        t = _with_approver(ctx.session)
        with pytest.raises(RuntimeError):
            ow.bootstrap(ctx, idempotency_key=key, name=name)
        t.join()

    project_id = "p_" + ow._derive_id(ctx, key, "project")
    with projects_db.connect_closing() as pconn:
        project = projects_db.get_project(pconn, project_id)
    board_slug = project.slug
    task_idempotency_key = "owtask_" + ow._derive_id(ctx, key, "task")

    # A CONTROL anchor with the SAME deterministic idempotency key already
    # exists but was never linked to a project (missing ownership metadata) —
    # pass project_id="" explicitly so it does NOT inherit the board's own
    # project_id (create_task's board-inheritance only fires when the
    # caller omits project_id entirely). It has to be the same kind the replay
    # creates: create_task's idempotency lookup is scoped per task_kind, so a
    # 'work' row carrying this key is a different row the anchor path can
    # never resolve — and therefore never adopt — at all.
    kconn = kanban_db.connect(board=board_slug)
    try:
        precreated_task_id = kanban_db.create_task(
            kconn, title=name, board=board_slug, idempotency_key=task_idempotency_key,
            project_id="", triage=True, control=True,
        )
        assert kanban_db.get_control_task(kconn, precreated_task_id).project_id is None
    finally:
        kconn.close()

    _expire_lock(ctx, key)
    t2 = _with_approver(ctx.session)
    with pytest.raises(ow.OwnerWorkspaceError) as excinfo:
        ow.bootstrap(ctx, idempotency_key=key, name=name)
    t2.join()
    assert excinfo.value.code == "crash_recovery_failed"


# ---------------------------------------------------------------------------
# Idempotent comment across crash gaps
# ---------------------------------------------------------------------------


def test_comment_crash_before_finalize_replay_creates_no_duplicate(ctx):
    setup = _bootstrap_board(ctx)
    board, task_id = setup["board"], setup["task_id"]
    key = "comment-crash-1"

    def boom(*a, **kw):
        raise RuntimeError("simulated crash after add_comment, before finalize")

    with _temporarily_patch(ow.kanban_db, "task_event_revision", boom):
        t = _with_approver(ctx.session)
        with pytest.raises(RuntimeError):
            ow.comment_task(ctx, idempotency_key=key, task_id=task_id, body="hello", project_id=setup["project_id"])
        t.join()

    _expire_lock(ctx, key)
    with projects_db.connect_closing() as pconn:
        assert projects_db.archive_project(pconn, setup["project_id"]) is True

    t2 = _with_approver(ctx.session)
    result = ow.comment_task(ctx, idempotency_key=key, task_id=task_id, body="hello", project_id=setup["project_id"])
    t2.join()
    assert result["ok"] is True

    kconn = kanban_db.connect(board=board)
    try:
        comments = kanban_db.list_comments(kconn, task_id, include_control=True)
    finally:
        kconn.close()
    assert len(comments) == 1
    assert comments[0].body == "hello"


def test_comment_replay_conflicting_payload_fails_closed_at_board_layer(ctx):
    """Defense in depth: even calling kanban_db.add_comment directly (below
    the kernel's own idempotency-key digest guard) with the same
    operation_key but a different body/author must fail closed, not silently
    return the mismatched original."""
    setup = _bootstrap_board(ctx)
    board, task_id = setup["board"], setup["task_id"]

    kconn = kanban_db.connect(board=board)
    try:
        kanban_db.add_comment(
            kconn, task_id, author="default", body="first",
            operation_key="opk-1", include_control=True,
        )
        with pytest.raises(ValueError):
            kanban_db.add_comment(
                kconn, task_id, author="default", body="different",
                operation_key="opk-1", include_control=True,
            )
        comments = kanban_db.list_comments(kconn, task_id, include_control=True)
    finally:
        kconn.close()
    assert len(comments) == 1
    assert comments[0].body == "first"


# ---------------------------------------------------------------------------
# Task-move readiness repair after crash
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("to_status", ["done", "archived"])
@pytest.mark.parametrize("archive_before_replay", [False, True])
def test_move_task_crash_before_recompute_ready_replay_repairs_child_readiness(
    ctx, to_status, archive_before_replay,
):
    setup = _bootstrap_board(ctx)
    board = setup["board"]
    parent_id = setup["task_id"]

    kconn = kanban_db.connect(board=board)
    try:
        child_id = kanban_db.create_task(kconn, title="child", parents=[parent_id])
        assert kanban_db.get_task(kconn, child_id).status == "todo"
        rev = kanban_db.task_event_revision(kconn, parent_id)
    finally:
        kconn.close()

    key = f"move-crash-{to_status}"

    def boom(*a, **kw):
        raise RuntimeError("simulated crash after CAS commit, before recompute_ready")

    with _temporarily_patch(ow.kanban_db, "recompute_ready", boom):
        t = _with_approver(ctx.session)
        with pytest.raises(RuntimeError):
            ow.move_task(
                ctx, idempotency_key=key, task_id=parent_id, to_status=to_status,
                expected_status=_ANCHOR_STATUS, expected_revision=rev,
                project_id=setup["project_id"],
            )
        t.join()

    _expire_lock(ctx, key)

    # The premise of the bug: the CAS status + event commit already
    # succeeded before the crash — only recompute_ready failed.
    kconn = kanban_db.connect(board=board)
    try:
        assert kanban_db.get_control_task(kconn, parent_id).status == to_status
    finally:
        kconn.close()

    if archive_before_replay:
        with projects_db.connect_closing() as pconn:
            assert projects_db.archive_project(pconn, setup["project_id"]) is True

    t2 = _with_approver(ctx.session)
    result = ow.move_task(
        ctx, idempotency_key=key, task_id=parent_id, to_status=to_status,
        expected_status=_ANCHOR_STATUS, expected_revision=rev,
        project_id=setup["project_id"],
    )
    t2.join()
    assert result["ok"] is True
    assert result["status"] == to_status

    kconn = kanban_db.connect(board=board)
    try:
        assert kanban_db.get_task(kconn, child_id).status == "ready"
    finally:
        kconn.close()


def test_move_task_unrelated_drift_after_crash_still_conflicts(ctx):
    """A replay must not paper over a genuinely unrelated status change that
    happened to land at the same revision slot — only THIS receipt's own
    recorded owner_move event short-circuits the CAS."""
    setup = _bootstrap_board(ctx)
    board, task_id = setup["board"], setup["task_id"]

    kconn = kanban_db.connect(board=board)
    try:
        rev = kanban_db.task_event_revision(kconn, task_id)
    finally:
        kconn.close()

    key = "move-drift-1"

    def boom(*a, **kw):
        raise RuntimeError("simulated crash before the CAS ever runs")

    with _temporarily_patch(ow.kanban_db, "cas_transition_task", boom):
        t = _with_approver(ctx.session)
        with pytest.raises(RuntimeError):
            ow.move_task(
                ctx, idempotency_key=key, task_id=task_id, to_status="blocked",
                expected_status=_ANCHOR_STATUS, expected_revision=rev,
                project_id=setup["project_id"],
            )
        t.join()

    _expire_lock(ctx, key)

    # An unrelated actor moves the task in the meantime (not via this
    # receipt) — a genuine conflict for the queued replay.
    kconn = kanban_db.connect(board=board)
    try:
        kanban_db.cas_transition_task(
            kconn, task_id, expected_status=_ANCHOR_STATUS, expected_revision=rev,
            to_status="review", event_kind="unrelated_move",
        )
    finally:
        kconn.close()

    t2 = _with_approver(ctx.session)
    result = ow.move_task(
        ctx, idempotency_key=key, task_id=task_id, to_status="blocked",
        expected_status=_ANCHOR_STATUS, expected_revision=rev,
        project_id=setup["project_id"],
    )
    t2.join()
    assert result["ok"] is False
    assert result["error"] == "conflict"
    assert result["current_status"] == "review"


def test_move_task_exact_replay_recognizes_own_committed_event_by_full_identity(ctx):
    """Positive case: a replay after a crash-after-CAS-commit is recognized as
    the receipt's OWN event once identity binding covers actor, profile,
    idempotency_key, AND the requested transition — not just a bare
    idempotency_key string."""
    setup = _bootstrap_board(ctx)
    board, task_id = setup["board"], setup["task_id"]

    kconn = kanban_db.connect(board=board)
    try:
        rev = kanban_db.task_event_revision(kconn, task_id)
    finally:
        kconn.close()

    key = "exact-replay-1"

    def boom(*a, **kw):
        raise RuntimeError("simulated crash after CAS commit, before finalize")

    with _temporarily_patch(ow, "_finalize_receipt", boom):
        t = _with_approver(ctx.session)
        with pytest.raises(RuntimeError):
            ow.move_task(
                ctx, idempotency_key=key, task_id=task_id, to_status="blocked",
                expected_status=_ANCHOR_STATUS, expected_revision=rev,
                project_id=setup["project_id"],
            )
        t.join()

    # The CAS really did commit before the simulated crash.
    kconn = kanban_db.connect(board=board)
    try:
        committed_task = kanban_db.get_control_task(kconn, task_id)
        committed_revision = kanban_db.task_event_revision(kconn, task_id)
    finally:
        kconn.close()
    assert committed_task.status == "blocked"

    with projects_db.connect_closing() as pconn:
        assert projects_db.archive_project(pconn, setup["project_id"]) is True
    _expire_lock(ctx, key)
    t2 = _with_approver(ctx.session)
    result = ow.move_task(
        ctx, idempotency_key=key, task_id=task_id, to_status="blocked",
        expected_status=_ANCHOR_STATUS, expected_revision=rev,
        project_id=setup["project_id"],
    )
    t2.join()
    assert result["ok"] is True
    assert result["status"] == "blocked"
    assert result["revision"] == committed_revision

    # No second event/CAS was run — revision did not advance further.
    kconn = kanban_db.connect(board=board)
    try:
        assert kanban_db.task_event_revision(kconn, task_id) == committed_revision
    finally:
        kconn.close()


# ---------------------------------------------------------------------------
# A terminal move's dependency release is bounded by receipt durability
# ---------------------------------------------------------------------------


def _dependent_setup(ctx):
    """One bootstrapped Project whose anchor task gates one dependent."""
    setup = _bootstrap_board(ctx)
    with kanban_db.connect(board=setup["board"]) as kconn:
        child_id = kanban_db.create_task(
            kconn, title="child", parents=[setup["task_id"]],
        )
        assert kanban_db.get_task(kconn, child_id).status == "todo"
        setup["child_id"] = child_id
        setup["revision"] = kanban_db.task_event_revision(kconn, setup["task_id"])
    return setup


def _status_of(board: str, task_id: str) -> str:
    with kanban_db.connect(board=board) as kconn:
        return kanban_db.get_task(kconn, task_id).status


@pytest.mark.parametrize("to_status", ["done", "archived"])
def test_move_task_dependent_is_unclaimable_until_the_receipt_is_durable(
    ctx, to_status,
):
    """A dispatcher tick in the crash window cannot claim the dependent.

    The parent really is terminal at that moment — its dependency is
    satisfied — so the ONLY thing keeping the dependent out of the work pool
    is that the same transaction parked it. A readiness recompute run from an
    independent connection right before the receipt commits stands in for the
    live dispatcher tick.
    """
    setup = _dependent_setup(ctx)
    board, child_id = setup["board"], setup["child_id"]
    observed: dict = {}
    real_finalize = ow._finalize_receipt

    def _observe_then_finalize(pconn, ctx_, key, token, *, status, result):
        with kanban_db.connect(board=board) as tick:
            observed["parent_status"] = kanban_db.get_task(
                tick, setup["task_id"], include_control=True,
            ).status
            observed["promoted"] = kanban_db.recompute_ready(tick)
            observed["child_status"] = kanban_db.get_task(tick, child_id).status
        return real_finalize(
            pconn, ctx_, key, token, status=status, result=result,
        )

    with _temporarily_patch(ow, "_finalize_receipt", _observe_then_finalize):
        t = _with_approver(ctx.session)
        result = ow.move_task(
            ctx, idempotency_key=f"park-window-{to_status}",
            task_id=setup["task_id"], to_status=to_status,
            expected_status=_ANCHOR_STATUS, expected_revision=setup["revision"],
            project_id=setup["project_id"],
        )
        t.join()

    assert result["ok"] is True
    assert observed["parent_status"] == to_status
    assert observed["child_status"] == kanban_db.PARKED_STATUS
    assert observed["promoted"] == 0
    assert observed["child_status"] not in kanban_db.EXECUTABLE_STATUSES

    # The receipt is durable now, so the same dependent is runnable.
    assert result["parked_dependents"] == [[child_id, "todo"]]
    assert _status_of(board, child_id) == "ready"


@pytest.mark.parametrize("to_status", ["done", "archived"])
def test_move_task_crash_before_activation_replays_the_exact_parked_set(
    ctx, to_status,
):
    """A crash after the receipt commits leaves the dependent parked, and the
    replay releases exactly the set that receipt records."""
    setup = _dependent_setup(ctx)
    board, child_id = setup["board"], setup["child_id"]
    key = f"park-crash-{to_status}"

    def boom(*_a, **_kw):
        raise RuntimeError("crash after terminal receipt, before activation")

    with _temporarily_patch(ow, "_activate_committed_owner_move", boom):
        t = _with_approver(ctx.session)
        with pytest.raises(RuntimeError):
            ow.move_task(
                ctx, idempotency_key=key, task_id=setup["task_id"],
                to_status=to_status, expected_status=_ANCHOR_STATUS,
                expected_revision=setup["revision"],
                project_id=setup["project_id"],
            )
        t.join()

    # Premise: the receipt committed, and the dependent is parked — not
    # claimable, and not lost either.
    with projects_db.connect_closing() as pconn:
        row = ow._get_receipt(pconn, ctx, key)
        assert row["status"] == "committed"
        committed = json.loads(row["result_json"])
    assert committed["parked_dependents"] == [[child_id, "todo"]]
    assert _status_of(board, child_id) == kanban_db.PARKED_STATUS

    replay = ow.move_task(
        ctx, idempotency_key=key, task_id=setup["task_id"], to_status=to_status,
        expected_status=_ANCHOR_STATUS, expected_revision=setup["revision"],
        project_id=setup["project_id"],
    )
    assert replay == committed
    assert _status_of(board, child_id) == "ready"

    # Activation is idempotent: replaying again changes nothing.
    assert ow.move_task(
        ctx, idempotency_key=key, task_id=setup["task_id"], to_status=to_status,
        expected_status=_ANCHOR_STATUS, expected_revision=setup["revision"],
        project_id=setup["project_id"],
    ) == committed
    assert _status_of(board, child_id) == "ready"


def test_move_task_dead_claim_recovery_reconstructs_the_parked_set(ctx):
    """A crash between the CAS commit and finalization is recovered from the
    committed ``owner_move`` event, parked set included."""
    setup = _dependent_setup(ctx)
    board, child_id = setup["board"], setup["child_id"]
    key = "park-dead-claim"

    def boom(*_a, **_kw):
        raise RuntimeError("crash after CAS commit, before finalize")

    with _temporarily_patch(ow, "_finalize_receipt", boom):
        t = _with_approver(ctx.session)
        with pytest.raises(RuntimeError):
            ow.move_task(
                ctx, idempotency_key=key, task_id=setup["task_id"],
                to_status="done", expected_status=_ANCHOR_STATUS,
                expected_revision=setup["revision"],
                project_id=setup["project_id"],
            )
        t.join()

    assert _status_of(board, child_id) == kanban_db.PARKED_STATUS
    with kanban_db.connect(board=board) as kconn:
        event = kanban_db.get_next_event_after(
            kconn, setup["task_id"], setup["revision"],
        )
    assert event.kind == "owner_move"
    assert event.payload["parked_dependents"] == [[child_id, "todo"]]

    _expire_lock(ctx, key)
    t2 = _with_approver(ctx.session)
    result = ow.move_task(
        ctx, idempotency_key=key, task_id=setup["task_id"], to_status="done",
        expected_status=_ANCHOR_STATUS, expected_revision=setup["revision"],
        project_id=setup["project_id"],
    )
    t2.join()
    assert result["ok"] is True
    assert result["parked_dependents"] == [[child_id, "todo"]]
    assert _status_of(board, child_id) == "ready"


def test_move_task_never_parks_or_promotes_a_sticky_blocked_dependent(ctx):
    """An explicit operator hold survives the parent going terminal."""
    setup = _dependent_setup(ctx)
    board, child_id = setup["board"], setup["child_id"]
    with kanban_db.connect(board=board) as kconn:
        # An explicit ``blocked`` event is what makes the hold sticky.
        assert kanban_db.cas_transition_task(
            kconn, child_id, expected_status="todo",
            expected_revision=kanban_db.task_event_revision(kconn, child_id),
            to_status="blocked", event_kind="blocked",
            event_payload={"reason": "review-required: owner input"},
        )["moved"] is True
        assert kanban_db._has_sticky_block(kconn, child_id) is True

    t = _with_approver(ctx.session)
    result = ow.move_task(
        ctx, idempotency_key="park-sticky", task_id=setup["task_id"],
        to_status="done", expected_status=_ANCHOR_STATUS,
        expected_revision=setup["revision"], project_id=setup["project_id"],
    )
    t.join()

    assert result["ok"] is True
    assert result["parked_dependents"] == []
    assert _status_of(board, child_id) == "blocked"
    with kanban_db.connect(board=board) as kconn:
        kinds = [
            row["kind"] for row in kconn.execute(
                "SELECT kind FROM task_events WHERE task_id = ?", (child_id,),
            )
        ]
    assert "owner_work_parked" not in kinds
    assert "owner_work_activated" not in kinds


def test_move_task_non_terminal_move_parks_nothing(ctx):
    """An ordinary move satisfies no dependency, so it releases nothing."""
    setup = _dependent_setup(ctx)
    board, child_id = setup["board"], setup["child_id"]

    t = _with_approver(ctx.session)
    result = ow.move_task(
        ctx, idempotency_key="park-ordinary", task_id=setup["task_id"],
        to_status="blocked", expected_status=_ANCHOR_STATUS,
        expected_revision=setup["revision"], project_id=setup["project_id"],
    )
    t.join()

    assert result["ok"] is True
    assert result["parked_dependents"] == []
    assert _status_of(board, child_id) == "todo"
    with kanban_db.connect(board=board) as kconn:
        event = kanban_db.get_next_event_after(
            kconn, setup["task_id"], setup["revision"],
        )
    # A move that cannot satisfy a dependency records no release at all, so
    # its event payload is exactly what it always was.
    assert "parked_dependents" not in event.payload


def test_move_task_leaves_a_dependent_with_another_open_parent_alone(ctx):
    """Only work this transition NEWLY enables is parked."""
    setup = _dependent_setup(ctx)
    board, child_id = setup["board"], setup["child_id"]
    with kanban_db.connect(board=board) as kconn:
        other_parent = kanban_db.create_task(
            kconn, title="other parent", project_id=setup["project_id"],
        )
        kanban_db.link_tasks(kconn, other_parent, child_id)
        revision = kanban_db.task_event_revision(kconn, setup["task_id"])

    t = _with_approver(ctx.session)
    result = ow.move_task(
        ctx, idempotency_key="park-not-enabled", task_id=setup["task_id"],
        to_status="done", expected_status=_ANCHOR_STATUS,
        expected_revision=revision, project_id=setup["project_id"],
    )
    t.join()

    assert result["ok"] is True
    assert result["parked_dependents"] == []
    assert _status_of(board, child_id) == "todo"


def test_move_task_dependent_is_released_once_under_concurrent_replays(ctx):
    """Concurrent callers of the same key release the dependent exactly once."""
    setup = _dependent_setup(ctx)
    board, child_id = setup["board"], setup["child_id"]
    key = "park-concurrent"

    t = _with_approver(ctx.session)
    first = ow.move_task(
        ctx, idempotency_key=key, task_id=setup["task_id"], to_status="done",
        expected_status=_ANCHOR_STATUS, expected_revision=setup["revision"],
        project_id=setup["project_id"],
    )
    t.join()
    assert first["ok"] is True

    results: list = []
    errors: list = []

    def _replay():
        try:
            results.append(
                ow.move_task(
                    ctx, idempotency_key=key, task_id=setup["task_id"],
                    to_status="done", expected_status=_ANCHOR_STATUS,
                    expected_revision=setup["revision"],
                    project_id=setup["project_id"],
                )
            )
        except Exception as exc:  # pragma: no cover - surfaced by the assert
            errors.append(exc)

    threads = [threading.Thread(target=_replay) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert errors == []
    assert all(result == first for result in results)
    assert _status_of(board, child_id) == "ready"
    with kanban_db.connect(board=board) as kconn:
        activations = kconn.execute(
            "SELECT COUNT(*) AS n FROM task_events "
            "WHERE task_id = ? AND kind = 'owner_work_activated'",
            (child_id,),
        ).fetchone()["n"]
    assert activations == 1


def test_move_task_requires_the_receipt_owner_before_any_transition(ctx):
    """A caller cannot choose another actor's Project by board or task id."""
    setup = _bootstrap_board(ctx)
    board, task_id = setup["board"], setup["task_id"]

    kconn = kanban_db.connect(board=board)
    try:
        rev0 = kanban_db.task_event_revision(kconn, task_id)
    finally:
        kconn.close()

    shared_key = "shared-cross-actor-key"
    other_ctx = ow.OwnerContext(actor="other-actor", profile="other-actor", session="run_owt_other")

    with pytest.raises(ow.OwnerWorkspaceError) as excinfo:
        ow.move_task(
            other_ctx, idempotency_key=shared_key, task_id=task_id,
            to_status="archived", expected_status=_ANCHOR_STATUS,
            expected_revision=rev0, project_id=setup["project_id"],
        )
    assert excinfo.value.code == "project_not_owned"

    # The original ("default") actor independently and legitimately moves
    # the SAME task using the SAME idempotency_key text — a completely
    # separate, unrelated receipt row (primary-keyed by actor+profile+key).
    t2 = _with_approver(ctx.session)
    a_result = ow.move_task(
        ctx, idempotency_key=shared_key, task_id=task_id,
        to_status="blocked", expected_status=_ANCHOR_STATUS,
        expected_revision=rev0, project_id=setup["project_id"],
    )
    t2.join()
    assert a_result["ok"] is True
    assert a_result["status"] == "blocked"

    kconn = kanban_db.connect(board=board)
    try:
        task = kanban_db.get_control_task(kconn, task_id)
    finally:
        kconn.close()
    assert task.status == "blocked"


# ---------------------------------------------------------------------------
# Public optimistic revision (status/revision from bootstrap + comment feed
# directly into owner_task_move)
# ---------------------------------------------------------------------------


def test_bootstrap_status_and_revision_feed_directly_into_move(ctx):
    t = _with_approver(ctx.session)
    setup = ow.bootstrap(ctx, idempotency_key="pubrev-bootstrap-1", name="Pub Rev WS")
    t.join()
    assert setup["ok"] is True
    assert setup["status"] == _ANCHOR_STATUS
    assert setup["revision"] == 1

    t2 = _with_approver(ctx.session)
    result = ow.move_task(
        ctx, idempotency_key="pubrev-bootstrap-1-move", task_id=setup["task_id"],
        to_status="blocked", expected_status=setup["status"],
        expected_revision=setup["revision"], project_id=setup["project_id"],
    )
    t2.join()
    assert result["ok"] is True
    assert result["status"] == "blocked"


def test_comment_status_and_revision_feed_directly_into_move(ctx):
    setup = _bootstrap_board(ctx)
    board, task_id = setup["board"], setup["task_id"]

    t = _with_approver(ctx.session)
    comment_result = ow.comment_task(
        ctx, idempotency_key="pubrev-comment-1", task_id=task_id, body="hi", project_id=setup["project_id"],
    )
    t.join()
    assert comment_result["ok"] is True
    assert comment_result["status"] == _ANCHOR_STATUS
    assert comment_result["revision"] == 2  # task "created" + "commented"

    t2 = _with_approver(ctx.session)
    move_result = ow.move_task(
        ctx, idempotency_key="pubrev-comment-1-move", task_id=task_id,
        to_status="blocked", expected_status=comment_result["status"],
        expected_revision=comment_result["revision"], project_id=setup["project_id"],
    )
    t2.join()
    assert move_result["ok"] is True
    assert move_result["status"] == "blocked"


# ---------------------------------------------------------------------------
# Cross-profile board ownership — profile-local projects.db, GLOBAL kanban root
#
# Real multiprocessing, no mocked locking: the receipt lease only orders
# claimants that share one projects.db, so only the kernel-held global board
# guard can order two same-name bootstraps started from different profiles.
# ---------------------------------------------------------------------------


# "fork" keeps the already-imported, already-sandboxed interpreter state (the
# children only need to repoint HERMES_HOME); "spawn" is a correct fallback
# where fork is unavailable.
_MP = multiprocessing.get_context(
    "fork" if "fork" in multiprocessing.get_all_start_methods() else "spawn"
)

_XPROFILE_KEY = "xprofile-bootstrap-1"
_XPROFILE_NAME = "Shared Owner Board"
_XPROFILE_PROFILES = ("alpha", "beta")


def _xprofile_ctx(profile: str):
    return ow.OwnerContext(actor="owner", profile=profile, session=f"run_owt_{profile}")


def _expected_project_id(profile: str) -> str:
    """The deterministic id bootstrap() will derive for this profile."""
    return "p_" + ow._derive_id(_xprofile_ctx(profile), _XPROFILE_KEY, "project")


def _enter_profile(root: str, profile: str) -> None:
    """Repoint this process at ``<root>/profiles/<profile>``.

    Gives each process its own profile-local ``projects.db`` while
    ``kanban_db.kanban_home()`` still resolves back to the shared ``<root>``
    — the exact production layout the global board guard exists for.
    """
    home = Path(root) / "profiles" / profile
    home.mkdir(parents=True, exist_ok=True)
    os.environ["HERMES_HOME"] = str(home)
    assert kanban_db.kanban_home() == Path(root)


def _xprofile_bootstrap_worker(root, profile, barrier, queue):
    """Child process: bootstrap the same board name from its own profile."""
    try:
        _enter_profile(root, profile)
        child_ctx = _xprofile_ctx(profile)
        barrier.wait(timeout=60)
        approver = _with_approver(child_ctx.session)
        try:
            result = ow.bootstrap(
                child_ctx, idempotency_key=_XPROFILE_KEY, name=_XPROFILE_NAME,
            )
        finally:
            approver.join(timeout=30)
        queue.put((profile, result))
    except ow.OwnerWorkspaceError as exc:
        queue.put((profile, {"ok": False, "error": exc.code}))
    except BaseException as exc:  # reported as a failure, never a silent hang
        queue.put((profile, {"ok": False, "error": "worker_crashed", "detail": repr(exc)}))


def _guard_holder_worker(root, slug, acquired, queue):
    _enter_profile(root, "alpha")
    with ow._global_board_guard(slug):
        acquired.set()
        time.sleep(0.5)
        queue.put(("holder_released_at", time.monotonic()))


def _guard_waiter_worker(root, slug, acquired, queue):
    _enter_profile(root, "beta")
    acquired.wait(timeout=60)
    started = time.monotonic()
    with ow._global_board_guard(slug):
        queue.put(("waiter_acquired_at", time.monotonic()))
    queue.put(("waiter_waited", time.monotonic() - started))


def _run_workers(procs, queue, expected_messages):
    """Start, drain and join child processes with bounded waits."""
    for proc in procs:
        proc.start()
    try:
        messages = [queue.get(timeout=120) for _ in range(expected_messages)]
    finally:
        for proc in procs:
            proc.join(timeout=120)
            if proc.is_alive():  # pragma: no cover - only on a real hang
                proc.terminate()
                proc.join(timeout=30)
    for proc in procs:
        assert proc.exitcode == 0, f"child exited {proc.exitcode}"
    return messages


def test_global_board_guard_serializes_across_processes(tmp_path):
    """The guard is an OS lock on the GLOBAL board namespace, so two processes
    in different profiles cannot hold the same board slug at once."""
    root = tmp_path / "xprofile_root"
    (root / "profiles").mkdir(parents=True)
    acquired = _MP.Event()
    queue = _MP.Queue()
    procs = [
        _MP.Process(target=_guard_holder_worker, args=(str(root), "shared-board", acquired, queue)),
        _MP.Process(target=_guard_waiter_worker, args=(str(root), "shared-board", acquired, queue)),
    ]

    events = dict(_run_workers(procs, queue, 3))

    assert events["waiter_acquired_at"] > events["holder_released_at"]
    assert events["waiter_waited"] >= 0.4


def test_cross_profile_bootstrap_of_same_board_elects_exactly_one_owner(tmp_path, monkeypatch):
    """Two profiles concurrently bootstrap the SAME board name with DIFFERENT
    deterministic project ids. Their receipts live in separate projects.db
    files, so nothing but the global board guard orders them: exactly one may
    create and own the board, the other must fail closed with the ownership
    conflict instead of overwriting board.json's project_id."""
    root = tmp_path / "xprofile_root"
    (root / "profiles").mkdir(parents=True)
    barrier = _MP.Barrier(len(_XPROFILE_PROFILES))
    queue = _MP.Queue()
    procs = [
        _MP.Process(target=_xprofile_bootstrap_worker, args=(str(root), profile, barrier, queue))
        for profile in _XPROFILE_PROFILES
    ]

    results = dict(_run_workers(procs, queue, len(_XPROFILE_PROFILES)))

    project_ids = {p: _expected_project_id(p) for p in _XPROFILE_PROFILES}
    assert len(set(project_ids.values())) == len(_XPROFILE_PROFILES)

    winners = [p for p, r in results.items() if r.get("ok") is True]
    losers = [p for p, r in results.items() if r.get("ok") is not True]
    assert len(winners) == 1, f"expected exactly one owner, got {results}"
    assert len(losers) == 1

    winner, loser = winners[0], losers[0]
    assert results[winner]["project_id"] == project_ids[winner]
    assert results[loser]["error"] == "crash_recovery_failed", results[loser]

    board = results[winner]["board"]

    # The published board is owned by the winner, and the loser never got to
    # restamp it.
    monkeypatch.setenv("HERMES_HOME", str(root / "profiles" / winner))
    assert kanban_db.board_exists(board)
    assert kanban_db.read_board_metadata(board)["project_id"] == project_ids[winner]

    # The loser's own profile-local project row exists but was never bound to
    # the contested board.
    monkeypatch.setenv("HERMES_HOME", str(root / "profiles" / loser))
    with projects_db.connect_closing() as pconn:
        loser_project = projects_db.get_project(pconn, project_ids[loser])
    assert loser_project is not None
    assert loser_project.board_slug != board


# ---------------------------------------------------------------------------
# Native titles and names at the DIRECT Hermes mutation boundary.
#
# The Workspace client enforces the owner-title contract on what it sends, but
# these kernel functions are reachable without it: the API-server toolset is a
# thin pass-through, and an in-process caller skips the tool layer entirely.
# Whatever they write into projects.db/kanban.db is read back by every owner
# surface through ``owner_title``/``owner_project_name``, so the stored string
# has to already BE that projection — established before the request digest,
# the owner's approval description, persistence and the replay comparison all
# bind to it.
#
# Code points are written as ``chr(...)``: a literal invisible character in
# this file would be unreviewable.
# ---------------------------------------------------------------------------

_ENGLAND_FLAG = (
    "\U0001F3F4"
    + "".join(chr(0xE0000 + ord(char)) for char in "gbeng")
    + chr(0xE007F)
)

# One value carrying every class this boundary must resolve: an internal
# dispatcher prefix, an INVISIBLE_CHARS zero-width space, a default-ignorable
# soft hyphen, a bidi override, a lone surrogate, a URL credential — and one
# pinned RGI subdivision flag, which is legitimate Unicode and must survive.
_UNSAFE_NATIVE_TITLE = (
    f"B03 — Ship{chr(0x200B)} the{chr(0x00AD)} {chr(0x202E)}release{chr(0xD800)}"
    f" {_ENGLAND_FLAG}"
    " https://deploy:hunter2verylongpassword@git.example.com/repo.git"
)
_UNSAFE_NATIVE_PROJECT_NAME = (
    f"B03 — Shoe{chr(0x200B)} Shop{chr(0x00AD)} {chr(0x202E)}plan{chr(0xD800)}"
    f" {_ENGLAND_FLAG} key sk-ABCDEFGHIJ"
)
# Non-blank on arrival — none of these is Python whitespace, so ``strip()``
# keeps them — and nothing at all after canonicalization.
_HIDDEN_ONLY = f"{chr(0x200B)}{chr(0x00AD)}{chr(0x202E)}{chr(0xD800)}"
_PREFIX_ONLY_TITLE = "B03 —"


def _assert_canonical_stored_title(stored: str) -> None:
    """The stored title IS its own projection, and carries nothing hidden."""
    assert stored == ow.owner_title(stored)
    assert stored != ow._UNTITLED_WORK_ITEM
    for forbidden in (
        "B03 —", "B04 —", "R12 —",
        chr(0x200B), chr(0x00AD), chr(0x202E), chr(0xD800),
        "hunter2verylongpassword", "sk-ABCDEFGHIJ",
    ):
        assert forbidden not in stored
    # A lone surrogate raises here — the owner-read 500 this boundary exists
    # to make impossible.
    stored.encode("utf-8")


def test_task_graph_persists_canonical_native_titles_and_project_name(ctx):
    """Root title, child titles and the new Project name, all canonical."""
    args = _task_graph_args(
        idempotency_key="graph-canonical-titles",
        project_name=_UNSAFE_NATIVE_PROJECT_NAME,
        request_title=_UNSAFE_NATIVE_TITLE,
        tasks=[
            {
                "title": f"B04 — Prepare{chr(0x200B)} the release{chr(0xD800)}",
                "body": "Create the smallest complete release.",
                "assignee": "default",
                "execution_tier": "routine",
                "parents": [],
            },
            {
                "title": f"R12 — Verify the {chr(0x202E)}release{chr(0x00AD)}",
                "body": "Check the owner-visible result.",
                "assignee": "default",
                "execution_tier": "routine",
                "parents": [0],
            },
        ],
    )

    approver = _with_approver(ctx.session)
    result = _commit_task_graph(ctx, **args)
    approver.join()
    assert result["ok"] is True

    with kanban_db.connect(board=result["board"]) as kconn:
        root = kanban_db.get_task(kconn, result["root_task_id"])
        children = [
            kanban_db.get_task(kconn, task_id) for task_id in result["task_ids"]
        ]
    with projects_db.connect_closing() as pconn:
        project = projects_db.get_project(pconn, result["project_id"])

    assert root.title == ow.owner_title(_UNSAFE_NATIVE_TITLE)
    _assert_canonical_stored_title(root.title)
    # The pinned subdivision flag survives whole; the credential does not.
    assert _ENGLAND_FLAG in root.title
    assert "https://***@git.example.com/repo.git" in root.title

    assert [child.title for child in children] == [
        "Prepare the release",
        "Verify the release",
    ]
    for child in children:
        _assert_canonical_stored_title(child.title)

    # A Project name gets the same sanitizing contract with its own two
    # pinned differences: the 160 bound, and NO internal-prefix strip, because
    # an owner-authored name that happens to start like a dispatcher label is
    # still the owner's words.
    assert project.name == ow.owner_project_name(_UNSAFE_NATIVE_PROJECT_NAME)
    assert project.name == ow.owner_project_name(project.name)
    assert project.name.startswith("B03 — Shoe Shop plan")
    assert "sk-ABCDEFGHIJ" not in project.name
    assert chr(0xD800) not in project.name
    project.name.encode("utf-8")

    # The read-side projection now has nothing left to change.
    listed = [
        entry for entry in ow.list_committed_projects(ctx)
        if entry["project_id"] == result["project_id"]
    ]
    assert [entry["name"] for entry in listed] == [project.name]


def test_task_graph_replay_matches_on_the_canonical_title_not_its_spelling(ctx):
    """Canonicalization precedes the digest, so an equivalent spelling is the
    same operation — one approval, one root task, no conflict.

    Both halves of the boundary need it: the receipt digest is computed on the
    canonical payload, and the post-create check compares the stored root
    title against ``request_title``. Had either kept the raw spelling, a retry
    differing only in characters the owner cannot see would fail closed as an
    ``idempotency_key_conflict`` over what reads as the same words.
    """
    args = _task_graph_args(
        idempotency_key="graph-canonical-replay",
        project_name=_UNSAFE_NATIVE_PROJECT_NAME,
        request_title=_UNSAFE_NATIVE_TITLE,
    )
    approver = _with_approver(ctx.session)
    first = _commit_task_graph(ctx, **args)
    approver.join()
    assert first["ok"] is True

    with kanban_db.connect(board=first["board"]) as kconn:
        before = kconn.execute(
            "SELECT COUNT(*) AS n FROM tasks WHERE project_id = ?",
            (first["project_id"],),
        ).fetchone()["n"]

    # No approver from here on: a second decision request would have nothing
    # to resolve it, proving neither replay asked for one.
    approval.unregister_gateway_notify(ctx.session)
    assert _commit_task_graph(ctx, **args) == first

    zero_width = chr(0x200B)
    respelled = _task_graph_args(
        idempotency_key="graph-canonical-replay",
        project_name=_UNSAFE_NATIVE_PROJECT_NAME + zero_width,
        request_title=_UNSAFE_NATIVE_TITLE.replace("Ship", f"Ship{zero_width}"),
    )
    assert _commit_task_graph(ctx, **respelled) == first

    with kanban_db.connect(board=first["board"]) as kconn:
        after = kconn.execute(
            "SELECT COUNT(*) AS n FROM tasks WHERE project_id = ?",
            (first["project_id"],),
        ).fetchone()["n"]
    # root + two children + the Project's one hidden control anchor
    assert before == after == 4


def test_task_graph_rejects_titles_that_canonicalize_to_nothing(ctx):
    """The ``Untitled`` fallback is a read-side placeholder, never a write.

    ``owner_title``/``owner_project_name`` substitute it for empty input,
    which is right when projecting an already-stored value and wrong when it
    would BE the stored value: the owner never wrote those words. Each input
    below arrives non-blank and canonicalizes to nothing, so the whole request
    fails before approval or persistence.

    What rejects is the canonical emptiness, never the wording — the literal
    strings are owner text and persist, as the next test pins.
    """
    for index, overrides in enumerate((
        {"request_title": _HIDDEN_ONLY},
        {"request_title": _PREFIX_ONLY_TITLE},
        {"project_name": _HIDDEN_ONLY},
        {"tasks": [{
            "title": _PREFIX_ONLY_TITLE,
            "body": "Create the smallest complete release.",
            "assignee": "default",
            "execution_tier": "routine",
            "parents": [],
        }]},
    )):
        args = _task_graph_args(idempotency_key=f"graph-blank-{index}", **overrides)
        with pytest.raises(ow.OwnerWorkspaceError) as excinfo:
            _commit_task_graph(ctx, **args)
        assert excinfo.value.code == "invalid_argument"

    with projects_db.connect_closing() as pconn:
        names = {
            project.name
            for project in projects_db.list_projects(pconn, include_archived=True)
        }
    assert names.isdisjoint({"Launch Shop", ow._UNTITLED_PROJECT, _HIDDEN_ONLY})


def test_task_graph_persists_literal_untitled_values_the_owner_wrote(ctx):
    """The placeholder wording is not itself a rejection.

    ``Untitled work item`` and ``Untitled Project`` canonicalize to themselves,
    so they carry owner-visible text and are stored as written, on every native
    write path: root title, child title and new Project name. A boundary that
    recognized emptiness by comparing a projection against the fallback would
    refuse exactly the owner who really used those words.
    """
    args = _task_graph_args(
        idempotency_key="graph-literal-untitled",
        project_name=ow._UNTITLED_PROJECT,
        request_title=ow._UNTITLED_WORK_ITEM,
        tasks=[
            {
                "title": ow._UNTITLED_WORK_ITEM,
                "body": "Create the smallest complete release.",
                "assignee": "default",
                "execution_tier": "routine",
                "parents": [],
            },
        ],
    )

    approver = _with_approver(ctx.session)
    result = _commit_task_graph(ctx, **args)
    approver.join()
    assert result["ok"] is True

    with kanban_db.connect(board=result["board"]) as kconn:
        root = kanban_db.get_task(kconn, result["root_task_id"])
        child = kanban_db.get_task(kconn, result["task_ids"][0])
    with projects_db.connect_closing() as pconn:
        project = projects_db.get_project(pconn, result["project_id"])

    assert root.title == child.title == ow._UNTITLED_WORK_ITEM
    assert project.name == ow._UNTITLED_PROJECT
    # Stored as its own projection, exactly like any other owner text.
    assert ow.owner_title(root.title) == root.title
    assert ow.owner_project_name(project.name) == project.name


def test_project_plan_persists_canonical_titles_for_every_created_task(ctx):
    """Added and replacement tasks are native writes too — same contract.

    The plan's own ``request_title`` is NOT one of them: on an existing
    Project it is prose inside the recorded plan, not a work-item title, so it
    keeps its wording — internal-looking prefix included.
    """
    setup = _bootstrap_board(ctx)
    with kanban_db.connect(board=setup["board"]) as conn:
        broad_id = kanban_db.create_task(
            conn, title="Broad task", assignee="default", project_id=setup["project_id"],
        )
        broad_ref = _project_task_ref(conn, broad_id)

    prose_request_title = "B03 — Adapt the current plan"
    args = _project_plan_args(
        setup,
        [
            {
                "action": "add",
                "reason": "Create one bounded owner-approved task.",
                "title": _UNSAFE_NATIVE_TITLE,
                "body": "Produce the owner-visible result.",
                "assignee": "default",
                "execution_tier": "routine",
                "existing_parents": [],
                "new_parents": [],
            },
            {
                "action": "split",
                "reason": "The current task is too broad to verify safely.",
                "target": broad_ref,
                "replacements": [
                    {
                        "title": f"B04 — Build{chr(0x200B)} the change{chr(0xD800)}",
                        "body": "Produce one owner-visible outcome.",
                        "assignee": "default",
                        "execution_tier": "routine",
                        "parents": [],
                    },
                    {
                        "title": f"R12 — Check the {chr(0x202E)}change{chr(0x00AD)}",
                        "body": "Verify the outcome before downstream work continues.",
                        "assignee": "default",
                        "execution_tier": "routine",
                        "parents": [0],
                    },
                ],
            },
        ],
        idempotency_key="steward-canonical-titles",
        request_title=prose_request_title,
    )

    approver = _with_approver(ctx.session)
    result = _commit_project_plan(ctx, **args)
    approver.join()

    assert result["ok"] is True
    assert len(result["created_task_ids"]) == 3
    with kanban_db.connect(board=setup["board"]) as conn:
        titles = [
            kanban_db.get_task(conn, task_id).title
            for task_id in result["created_task_ids"]
        ]
        applied = [
            event
            for event in kanban_db.list_events(
                conn, setup["task_id"], include_control=True,
            )
            if event.kind == "owner_project_plan_applied"
        ]

    assert set(titles) == {
        ow.owner_title(_UNSAFE_NATIVE_TITLE),
        "Build the change",
        "Check the change",
    }
    for title in titles:
        _assert_canonical_stored_title(title)

    # Prose, unchanged: the plan record keeps the request title as written.
    assert len(applied) == 1
    assert applied[0].payload["plan_summary"].startswith(prose_request_title)

    # Same canonical payload under a different spelling is the same operation.
    approval.unregister_gateway_notify(ctx.session)
    respelled = dict(args)
    respelled["request_title"] = prose_request_title
    respelled["changes"] = json.loads(json.dumps(args["changes"]))
    respelled["changes"][0]["title"] = _UNSAFE_NATIVE_TITLE.replace(
        "Ship", f"Ship{chr(0x200B)}"
    )
    assert _commit_project_plan(ctx, **respelled) == result

    with kanban_db.connect(board=setup["board"]) as conn:
        assert len([
            task for task in kanban_db.list_tasks(conn)
            if task.title in titles
        ]) == 3


def test_project_plan_rejects_a_created_title_that_canonicalizes_to_nothing(ctx):
    setup = _bootstrap_board(ctx)
    args = _project_plan_args(
        setup,
        [{
            "action": "add",
            "reason": "Create one bounded owner-approved task.",
            "title": _HIDDEN_ONLY,
            "body": "Produce the owner-visible result.",
            "assignee": "default",
            "execution_tier": "routine",
            "existing_parents": [],
            "new_parents": [],
        }],
        idempotency_key="steward-blank-title",
    )

    with pytest.raises(ow.OwnerWorkspaceError) as excinfo:
        _commit_project_plan(ctx, **args)
    assert excinfo.value.code == "invalid_argument"

    with kanban_db.connect(board=setup["board"]) as conn:
        assert all(
            task.title not in {_HIDDEN_ONLY, ow._UNTITLED_WORK_ITEM}
            for task in kanban_db.list_tasks(conn)
        )


# ---------------------------------------------------------------------------
# Real production-path integration (no stubbed resolver at all)
# ---------------------------------------------------------------------------


def _write_real_profile_config(profile: str, provider: str, model: str, effort: str):
    """Materialise a real profile config.yaml under the sandboxed HERMES_HOME."""
    from hermes_cli.profiles import get_profile_dir

    directory = Path(get_profile_dir(profile))
    directory.mkdir(parents=True, exist_ok=True)
    directory.joinpath("config.yaml").write_text(
        "model:\n"
        f"  provider: {provider}\n"
        f"  default: {model}\n"
        "agent:\n"
        f"  reasoning_effort: {effort}\n"
        "fallback_providers: []\n",
        encoding="utf-8",
    )
    return directory


@pytest.fixture
def real_resolver():
    """Run the genuine config-reading resolver, defeating the autouse stub."""
    with _temporarily_patch(
        model_policy,
        "configured_assignment_for",
        model_policy.configured_assignment_for.__wrapped__
        if hasattr(model_policy.configured_assignment_for, "__wrapped__")
        else _REAL_CONFIGURED_ASSIGNMENT_FOR,
    ):
        yield


_REAL_CONFIGURED_ASSIGNMENT_FOR = model_policy.configured_assignment_for


def test_real_profile_config_resolves_and_locks_owner_task_routes(ctx, real_resolver):
    """End-to-end: a real config.yaml drives the pin the dispatcher accepts.

    Nothing about the route is stubbed here — the profile's provider is read
    off disk by the production resolver, the admitted matrix decides the model
    and effort, and the lock is minted, persisted and then re-validated exactly
    as ``_default_spawn`` does.
    """
    _write_real_profile_config(
        "raphael-builder", "anthropic", "claude-sonnet-5", "max"
    )
    args = _task_graph_args(
        idempotency_key="graph-real-config",
        project_name="Real Config Project",
        root_assignee="raphael-builder",
    )
    for task in args["tasks"]:
        task["assignee"] = "raphael-builder"
        task["responsibility"] = "B03"
    args["tasks"][0]["execution_tier"] = "deep"
    args["tasks"][1]["execution_tier"] = "routine"

    approver = _with_approver(ctx.session)
    result = _commit_task_graph(ctx, **args)
    approver.join()

    with kanban_db.connect(board=result["board"]) as conn:
        rows = {
            task_id: conn.execute(
                "SELECT * FROM tasks WHERE id = ?", (task_id,)
            ).fetchone()
            for task_id in (result["root_task_id"], *result["task_ids"])
        }

    deep_row = rows[result["task_ids"][0]]
    routine_row = rows[result["task_ids"][1]]
    # The matrix — not the test — decides the lanes: raphael-builder on
    # anthropic is Sonnet for routine work and Opus for deep work.
    assert (deep_row["execution_tier"], deep_row["model_override"]) == (
        "deep", "claude-opus-5",
    )
    assert (routine_row["execution_tier"], routine_row["model_override"]) == (
        "routine", "claude-sonnet-5",
    )
    # Every persisted lock is one the dispatcher would honour.
    for task_id, row in rows.items():
        assert row["model_policy_lock"], task_id
        assert kanban_db.task_policy_lock_error(row) is None, task_id

    # Selecting the other provider afterwards changes only NEW work: the fence
    # runs first, and the already-pinned rows keep their approved route.
    _write_real_profile_config(
        "raphael-builder", "openai-codex", "gpt-5.6-terra", "max"
    )
    assert ow.fence_effective_task_routes("raphael-builder") == []
    with kanban_db.connect(board=result["board"]) as conn:
        for task_id, row in rows.items():
            after = conn.execute(
                "SELECT * FROM tasks WHERE id = ?", (task_id,)
            ).fetchone()
            assert after["model_override"] == row["model_override"]
            assert after["model_policy_lock"] == row["model_policy_lock"]
            assert kanban_db.task_policy_lock_error(after) is None


def test_real_profile_config_with_fallbacks_enabled_cannot_pin_a_route(
    ctx, real_resolver
):
    """A role that can still fall back has no confirmable route to approve."""
    directory = _write_real_profile_config(
        "raphael-builder", "anthropic", "claude-sonnet-5", "max"
    )
    directory.joinpath("config.yaml").write_text(
        "model:\n"
        "  provider: anthropic\n"
        "  default: claude-sonnet-5\n"
        "agent:\n"
        "  reasoning_effort: max\n"
        "fallback_providers:\n"
        "  - provider: openai\n"
        "    model: gpt-x\n",
        encoding="utf-8",
    )
    args = _task_graph_args(
        idempotency_key="graph-real-fallbacks", root_assignee="raphael-builder",
    )
    for task in args["tasks"]:
        task["assignee"] = "raphael-builder"
        task["responsibility"] = "B03"

    with pytest.raises(ow.OwnerWorkspaceError) as excinfo:
        _commit_task_graph(ctx, **args)
    assert excinfo.value.code == "invalid_model_route"


def _install_profiles(*names):
    from hermes_cli.profiles import get_profile_dir

    for name in names:
        Path(get_profile_dir(name)).mkdir(parents=True, exist_ok=True)


def test_locked_review_handoff_is_refused_rather_than_silently_repinned(ctx):
    """Independent review is separately approved work, never a re-pin.

    Handing a locked task to a reviewer would mint a new route for a task whose
    assignee/provider/model/effort/tier the owner approved for its whole run.
    That is exactly the silent switch the lock exists to prevent, so the
    handoff is refused and the task stays put.
    """
    _install_profiles("raphael-builder", "raphael-verifier")
    args = _task_graph_args(idempotency_key="graph-review-handoff")
    args["tasks"][0]["assignee"] = "raphael-builder"
    args["tasks"][0]["responsibility"] = "B03"
    args["tasks"][0]["execution_tier"] = "routine"
    approver = _with_approver(ctx.session)
    result = _commit_task_graph(ctx, **args)
    approver.join()

    task_id = result["task_ids"][0]
    with kanban_db.connect(board=result["board"]) as conn:
        before = conn.execute(
            "SELECT * FROM tasks WHERE id = ?", (task_id,)
        ).fetchone()
        assert before["model_override"] == "claude-sonnet-5"
        with kanban_db.write_txn(conn):
            conn.execute(
                "UPDATE tasks SET status = 'running' WHERE id = ?", (task_id,)
            )
        with pytest.raises(RuntimeError, match="the owner approved that exact"):
            kanban_db.request_review(
                conn, task_id, reviewer="raphael-verifier", force=True,
            )
        after = conn.execute(
            "SELECT * FROM tasks WHERE id = ?", (task_id,)
        ).fetchone()

    for column in (
        "assignee", "model_override", "provider_override", "reasoning_effort",
        "execution_tier", "model_policy_lock",
    ):
        assert after[column] == before[column]
    assert kanban_db.task_policy_lock_error(after) is None


def test_locked_task_transition_to_any_other_role_is_refused(ctx):
    """No silent repin — not even to a role that IS separately approved."""
    _install_profiles("raphael-verifier")
    args = _task_graph_args(idempotency_key="graph-refuse-role")
    approver = _with_approver(ctx.session)
    result = _commit_task_graph(ctx, **args)
    approver.join()

    task_id = result["task_ids"][0]
    with kanban_db.connect(board=result["board"]) as conn:
        kanban_db.create_task(conn, title="unrelated", assignee="default")
        before = conn.execute(
            "SELECT * FROM tasks WHERE id = ?", (task_id,)
        ).fetchone()
        # raphael-verifier has its own admitted route for this tier, so this is
        # not "unroutable" — it is simply not this task's approved authority.
        with pytest.raises(RuntimeError, match="the owner approved that exact"):
            kanban_db.assign_task(conn, task_id, "raphael-verifier")
        # Unassignment would strand the lock, so it is refused too.
        with pytest.raises(RuntimeError, match="would strand the lock"):
            kanban_db.assign_task(conn, task_id, None)
        after = conn.execute(
            "SELECT * FROM tasks WHERE id = ?", (task_id,)
        ).fetchone()

    assert after["assignee"] == before["assignee"]
    assert after["model_policy_lock"] == before["model_policy_lock"]


def test_invalid_lock_blocks_every_role_transition(ctx):
    """A corrupt lock is not "unlocked": it fails closed on transition too."""
    _install_profiles("raphael-verifier")
    args = _task_graph_args(idempotency_key="graph-corrupt-lock")
    approver = _with_approver(ctx.session)
    result = _commit_task_graph(ctx, **args)
    approver.join()

    task_id = result["task_ids"][0]
    with kanban_db.connect(board=result["board"]) as conn:
        with kanban_db.write_txn(conn):
            conn.execute(
                "UPDATE tasks SET model_policy_lock = 'raphael' WHERE id = ?",
                (task_id,),
            )
        with pytest.raises(RuntimeError, match="provenance is unreadable"):
            kanban_db.assign_task(conn, task_id, "raphael-verifier")
        with pytest.raises(RuntimeError, match="owner-governed"):
            kanban_db.set_model_override(
                conn, task_id, "claude-sonnet-5", provider="anthropic",
            )


# ---------------------------------------------------------------------------
# Durable before activation: nothing is claimable until the receipt is terminal
# ---------------------------------------------------------------------------


class _CrashInjected(RuntimeError):
    """The simulated crash, distinguishable from a real defect."""


def _receipt_row(ctx, key: str):
    with projects_db.connect_closing() as pconn:
        ow._ensure_schema(pconn)
        return pconn.execute(
            "SELECT status, result_json FROM owner_workspace_receipts "
            "WHERE actor = ? AND profile = ? AND idempotency_key = ?",
            (ctx.actor, ctx.profile, key),
        ).fetchone()


def _live_dispatch_probe(board: str) -> dict:
    """Run one REAL dispatcher tick and report what it was able to take.

    Called from inside the injected crash, so it observes the board at exactly
    the instant the commit died — the window a live dispatcher would actually
    have hit.
    """
    spawned: list = []

    def fake_spawn(task, workspace, board=None):
        spawned.append(task.id)
        return 9999

    with contextlib.closing(kanban_db.connect(board=board)) as conn:
        kanban_db.recompute_ready(conn)
        kanban_db.dispatch_once(conn, spawn_fn=fake_spawn, board=board)
        rows = [
            (str(row["id"]), row["status"], row["claim_lock"])
            for row in conn.execute(
                "SELECT id, status, claim_lock FROM tasks "
                "WHERE task_kind = 'work' ORDER BY id"
            )
        ]
    return {"spawned": spawned, "rows": rows}


def _work_rows(board: str, project_id: str) -> dict:
    with contextlib.closing(kanban_db.connect(board=board)) as conn:
        return {
            str(row["id"]): row["status"]
            for row in conn.execute(
                "SELECT id, status FROM tasks "
                "WHERE task_kind = 'work' AND project_id = ?",
                (project_id,),
            )
        }


# Where the commit is made to die, and which internal step is interrupted.
_PRE_TERMINAL_CRASH_POINTS = {
    "after_graph_mutation": "_update_progress",
    "after_progress_write": "_finalize_receipt",
}


@pytest.mark.parametrize("crash_point", sorted(_PRE_TERMINAL_CRASH_POINTS))
def test_no_owner_work_is_claimable_before_the_terminal_receipt(ctx, crash_point):
    """A live dispatcher tick at the crash instant must find nothing to take."""
    key = f"graph-crash-{crash_point}"
    args = _task_graph_args(idempotency_key=key, project_name=f"Crash {crash_point}")
    probes: list = []
    target = _PRE_TERMINAL_CRASH_POINTS[crash_point]
    real = getattr(ow, target)

    def crash(pconn, c, idem, token, **kwargs):
        if crash_point == "after_graph_mutation" and "task_id" not in kwargs:
            # The FIRST progress write happens before the graph exists; the
            # one that records the root is the post-mutation step to die on.
            return real(pconn, c, idem, token, **kwargs)
        probes.append(_live_dispatch_probe(_board_for_project(args["project_name"])))
        raise _CrashInjected(crash_point)

    approver = _with_approver(ctx.session)
    with _temporarily_patch(ow, target, crash):
        with pytest.raises(_CrashInjected):
            _commit_task_graph(ctx, **args)
    approver.join()
    assert real is getattr(ow, target)

    # The dispatcher saw the board mid-commit and could take nothing: every
    # approved child was parked in the same insert that created it, and the
    # root it hangs under never reached an executable column.
    assert len(probes) == 1
    assert probes[0]["rows"], "the graph really was written before the crash"
    assert probes[0]["spawned"] == []
    assert all(lock is None for _id, _status, lock in probes[0]["rows"])
    statuses_at_crash = {status for _id, status, _lock in probes[0]["rows"]}
    assert not statuses_at_crash & {"ready", "review", "running"}
    assert "scheduled" in statuses_at_crash
    # ...and no terminal receipt exists to replay from.
    assert _receipt_row(ctx, key)["status"] not in ("committed", "denied")

    # Replay finishes the SAME approved graph — no second copy of anything.
    _expire_lock(ctx, key)
    approver = _with_approver(ctx.session)
    replayed = _commit_task_graph(ctx, **args)
    approver.join()

    assert replayed["ok"] is True
    assert replayed["task_count"] == 2
    statuses = _work_rows(replayed["board"], replayed["project_id"])
    # root + exactly two children, activated, with nothing duplicated.
    assert len(statuses) == 3
    assert "scheduled" not in statuses.values()


def _board_for_project(name: str) -> str:
    """The board an in-flight commit is writing to, resolved from projects.db.

    Deliberately not from the receipt: the receipt only learns its board on the
    progress write, which is one of the steps these tests crash before.
    """
    with projects_db.connect_closing() as pconn:
        for project in projects_db.list_projects(pconn, include_archived=True):
            if project.name == name:
                return project.board_slug
    return kanban_db.DEFAULT_BOARD


@pytest.mark.parametrize("crash_point", ["during_activation", "after_activation"])
def test_replay_from_a_committed_receipt_activates_once_without_reapproval(
    ctx, crash_point
):
    """The terminal receipt IS the authority; replay finishes the same transition."""
    key = f"graph-activation-{crash_point}"
    args = _task_graph_args(idempotency_key=key, project_name=f"Activate {crash_point}")
    probes: list = []

    real_activate = kanban_db.activate_owner_work
    real_dispatch_state = ow._set_project_dispatch_state

    def crash_during(conn, task_ids, **kwargs):
        probes.append(list(task_ids))
        raise _CrashInjected("during_activation")

    def crash_after(board_slug, *, enabled):
        probes.append(board_slug)
        raise _CrashInjected("after_activation")

    approver = _with_approver(ctx.session)
    if crash_point == "during_activation":
        patch_target = (kanban_db, "activate_owner_work", crash_during)
    else:
        patch_target = (ow, "_set_project_dispatch_state", crash_after)
    with _temporarily_patch(*patch_target):
        with pytest.raises(_CrashInjected):
            _commit_task_graph(ctx, **args)
    approver.join()

    # The receipt committed BEFORE activation was attempted.
    receipt = _receipt_row(ctx, key)
    assert receipt["status"] == "committed"
    committed = json.loads(receipt["result_json"])
    assert probes
    if crash_point == "during_activation":
        # Still parked: the transition never ran.
        assert set(_work_rows(committed["board"], committed["project_id"]).values()) <= {
            "scheduled", "triage", "todo",
        }
        assert kanban_db.board_dispatch_allowed(
            kanban_db.read_board_metadata(committed["board"])
        ) is False

    # Replay: no approver registered at all, and asking for one is a failure.
    approval.unregister_gateway_notify(ctx.session)
    _expire_lock(ctx, key)
    with _temporarily_patch(
        ow, "_confirm",
        lambda *a, **k: pytest.fail("replay asked the owner to approve again"),
    ):
        replayed = _commit_task_graph(ctx, **args)
        # Exactly once: replaying again changes nothing further.
        again = _commit_task_graph(ctx, **args)

    assert replayed == committed == again
    statuses = _work_rows(replayed["board"], replayed["project_id"])
    assert len(statuses) == 3
    assert "scheduled" not in statuses.values()
    assert kanban_db.board_dispatch_allowed(
        kanban_db.read_board_metadata(replayed["board"])
    ) is True
    assert real_activate is kanban_db.activate_owner_work
    assert real_dispatch_state is ow._set_project_dispatch_state


def test_an_owner_pause_survives_replayed_activation(ctx):
    """Activation is idempotent AND preserves an explicit owner stop."""
    args = _task_graph_args(idempotency_key="graph-pause", project_name="Paused Work")
    approver = _with_approver(ctx.session)
    result = _commit_task_graph(ctx, **args)
    approver.join()

    kanban_db.write_board_dispatch_state(
        result["board"], dispatch_enabled=False, dispatch_paused_by_owner=True,
    )
    ow._activate_committed_owner_work(result)

    assert kanban_db.board_dispatch_allowed(
        kanban_db.read_board_metadata(result["board"])
    ) is False


def test_project_plan_work_is_parked_until_its_receipt_commits(ctx):
    """The same contract on the plan path, with a live dispatcher at the crash."""
    setup = _bootstrap_board(ctx)
    args = _project_plan_args(
        setup,
        [{
            "action": "add",
            "reason": "Create one bounded owner-approved task.",
            "title": "Prepare the approved deliverable",
            "body": "Produce the owner-visible result.",
            "assignee": "default",
            "execution_tier": "routine",
            "existing_parents": [],
            "new_parents": [],
        }],
        idempotency_key="steward-plan-crash",
    )
    probes: list = []

    def crash(*a, **k):
        probes.append(_live_dispatch_probe(setup["board"]))
        raise _CrashInjected("plan_finalize")

    approver = _with_approver(ctx.session)
    with _temporarily_patch(ow, "_finalize_receipt", crash):
        with pytest.raises(_CrashInjected):
            _commit_project_plan(ctx, **args)
    approver.join()

    assert probes and probes[0]["spawned"] == []
    created = [
        (task_id, status)
        for task_id, status, _lock in probes[0]["rows"]
        if status == "scheduled"
    ]
    assert created, "the approved task should exist, parked and unclaimable"

    _expire_lock(ctx, "steward-plan-crash")
    approver = _with_approver(ctx.session)
    replayed = _commit_project_plan(ctx, **args)
    approver.join()

    assert replayed["ok"] is True
    with contextlib.closing(kanban_db.connect(board=setup["board"])) as conn:
        matching = [
            task for task in kanban_db.list_tasks(conn)
            if task.title == "Prepare the approved deliverable"
        ]
    # One task, activated exactly once — the crashed attempt left no duplicate.
    assert len(matching) == 1
    assert matching[0].status != kanban_db.PARKED_STATUS


# ---------------------------------------------------------------------------
# Cancelling a parent releases its dependents — but not before the receipt
# ---------------------------------------------------------------------------


def _cancel_plan_with_dependent(ctx, *, key: str, child_setup=None):
    """One Project holding a parent, its waiting child, and a cancel plan."""
    setup = _bootstrap_board(ctx)
    with kanban_db.connect(board=setup["board"]) as conn:
        parent_id = kanban_db.create_task(
            conn, title="Obsolete parent", assignee="default",
            project_id=setup["project_id"],
        )
        child_id = kanban_db.create_task(
            conn, title="Waiting dependent", assignee="default",
            parents=[parent_id], project_id=setup["project_id"],
        )
        if child_setup is not None:
            child_setup(conn, child_id)
        parent_ref = _project_task_ref(conn, parent_id)
        assert kanban_db.get_task(conn, child_id).status in ("todo", "blocked")
    args = _project_plan_args(
        setup,
        [{
            "action": "cancel",
            "reason": "The owner explicitly removed this outcome.",
            "target": parent_ref,
        }],
        idempotency_key=key,
    )
    return setup, parent_id, child_id, args


def test_a_cancelled_parent_never_releases_its_dependent_before_the_receipt(ctx):
    """A live dispatcher tick at the crash instant must find nothing to take."""
    setup, parent_id, child_id, args = _cancel_plan_with_dependent(
        ctx, key="steward-cancel-dependent-crash",
    )
    probes: list = []

    def crash(*a, **k):
        probes.append(_live_dispatch_probe(setup["board"]))
        raise _CrashInjected("plan_finalize")

    approver = _with_approver(ctx.session)
    with _temporarily_patch(ow, "_finalize_receipt", crash):
        with pytest.raises(_CrashInjected):
            _commit_project_plan(ctx, **args)
    approver.join()

    # The dispatcher ran a real tick against the board mid-commit: the parent
    # was already archived, and the child it unblocked was still unclaimable.
    assert probes and probes[0]["spawned"] == []
    statuses = dict((task_id, status) for task_id, status, _ in probes[0]["rows"])
    assert statuses[parent_id] == "archived"
    assert statuses[child_id] == kanban_db.PARKED_STATUS

    # Replay finishes the SAME approved plan and releases the dependent once.
    _expire_lock(ctx, args["idempotency_key"])
    approver = _with_approver(ctx.session)
    replayed = _commit_project_plan(ctx, **args)
    approver.join()

    assert replayed["ok"] is True
    assert replayed["parked_dependents"] == [[child_id, "todo"]]
    with contextlib.closing(kanban_db.connect(board=setup["board"])) as conn:
        assert kanban_db.get_task(conn, child_id).status != kanban_db.PARKED_STATUS

    # Exactly once: replaying the committed activation again changes nothing.
    before = _work_rows(setup["board"], setup["project_id"])
    ow._activate_committed_owner_work(replayed)
    assert _work_rows(setup["board"], setup["project_id"]) == before


def test_a_cancelled_parent_records_and_then_releases_its_dependent(ctx):
    setup, _parent_id, child_id, args = _cancel_plan_with_dependent(
        ctx, key="steward-cancel-dependent-ok",
    )
    approver = _with_approver(ctx.session)
    result = _commit_project_plan(ctx, **args)
    approver.join()

    assert result["ok"] is True
    # The committed receipt names the dependent it released, with the exact
    # column to restore — that record is what makes the replay recoverable.
    assert result["parked_dependents"] == [[child_id, "todo"]]
    assert child_id in result["affected_task_ids"]
    with contextlib.closing(kanban_db.connect(board=setup["board"])) as conn:
        assert kanban_db.get_task(conn, child_id).status != kanban_db.PARKED_STATUS


def test_replaying_a_receipt_never_un_postpones_work_the_owner_parked_since(ctx):
    """``scheduled`` is a shared column, so activation matches a generation.

    The dependent this plan released is later postponed by the owner — back
    into the very same column. Replaying the committed receipt (which every
    crash-recovery path does) must not read that as its own parked work and
    reactivate it, nor enable anything waiting on it.
    """
    setup, _parent_id, child_id, args = _cancel_plan_with_dependent(
        ctx, key="steward-replay-after-postpone",
    )
    approver = _with_approver(ctx.session)
    committed = _commit_project_plan(ctx, **args)
    approver.join()

    assert committed["parked_dependents"] == [[child_id, "todo"]]
    assert committed["park_generation"]

    # The owner then deliberately postpones that same task, which lands it in
    # the parked column again — this time as the owner's own decision.
    with contextlib.closing(kanban_db.connect(board=setup["board"])) as conn:
        released = kanban_db.get_task(conn, child_id)
        assert released.status != kanban_db.PARKED_STATUS
        postponed = kanban_db.cas_transition_task(
            conn,
            child_id,
            expected_status=released.status,
            expected_revision=kanban_db.task_event_revision(conn, child_id),
            to_status=kanban_db.PARKED_STATUS,
            event_kind="owner_move",
        )
        assert postponed["moved"] is True

    # The exact replay every crash-recovery path performs.
    ow._activate_committed_owner_work(committed)
    ow._activate_committed_owner_move(committed)

    with contextlib.closing(kanban_db.connect(board=setup["board"])) as conn:
        assert kanban_db.get_task(conn, child_id).status == kanban_db.PARKED_STATUS


def test_a_receipt_with_no_recorded_generation_releases_nothing(ctx):
    """Fail closed: an unidentifiable parking is never claimed by activation."""
    setup, _parent_id, child_id, args = _cancel_plan_with_dependent(
        ctx, key="steward-generation-missing",
    )
    approver = _with_approver(ctx.session)
    committed = _commit_project_plan(ctx, **args)
    approver.join()

    # Re-park the dependent and then replay a receipt whose generation is gone.
    with contextlib.closing(kanban_db.connect(board=setup["board"])) as conn:
        with kanban_db.write_txn(conn):
            conn.execute(
                "UPDATE tasks SET status = ? WHERE id = ?",
                (kanban_db.PARKED_STATUS, child_id),
            )
    ow._activate_committed_owner_work({**committed, "park_generation": None})

    with contextlib.closing(kanban_db.connect(board=setup["board"])) as conn:
        assert kanban_db.get_task(conn, child_id).status == kanban_db.PARKED_STATUS


def _circuit_broken(conn, child_id: str) -> None:
    """A dependent the breaker stopped: blocked, at its retry limit."""
    with kanban_db.write_txn(conn):
        conn.execute(
            "UPDATE tasks SET status = 'blocked', consecutive_failures = ?, "
            "max_retries = 1 WHERE id = ?",
            (kanban_db.DEFAULT_FAILURE_LIMIT + 1, child_id),
        )
    kanban_db._append_event(conn, child_id, "gave_up", None)
    conn.commit()


def test_parking_a_blocked_dependent_restores_it_blocked_with_its_guards(ctx):
    """Parking must never launder a stopped dependent into the work pool."""
    setup, _parent_id, child_id, args = _cancel_plan_with_dependent(
        ctx, key="steward-cancel-blocked-dependent", child_setup=_circuit_broken,
    )
    approver = _with_approver(ctx.session)
    result = _commit_project_plan(ctx, **args)
    approver.join()

    assert result["parked_dependents"] == [[child_id, "blocked"]]
    with contextlib.closing(kanban_db.connect(board=setup["board"])) as conn:
        assert kanban_db.get_task(conn, child_id).status == "blocked"


def _sticky_blocked(conn, child_id: str) -> None:
    """An explicit operator hold: ``blocked`` with a ``blocked`` event."""
    with kanban_db.write_txn(conn):
        conn.execute(
            "UPDATE tasks SET status = 'blocked' WHERE id = ?", (child_id,)
        )
        kanban_db._append_event(
            conn, child_id, "blocked",
            {"reason": "review-required: owner input", "block_kind": "needs_input"},
        )


def test_an_explicitly_blocked_dependent_is_left_exactly_as_it_is(ctx):
    setup, _parent_id, child_id, args = _cancel_plan_with_dependent(
        ctx, key="steward-cancel-sticky-dependent", child_setup=_sticky_blocked,
    )
    approver = _with_approver(ctx.session)
    result = _commit_project_plan(ctx, **args)
    approver.join()

    assert result["parked_dependents"] == []
    with contextlib.closing(kanban_db.connect(board=setup["board"])) as conn:
        assert kanban_db.get_task(conn, child_id).status == "blocked"


def test_a_merged_parents_dependent_waits_on_the_parked_replacement(ctx):
    """Merge relinks every dependent under the replacement, which is parked."""
    setup = _bootstrap_board(ctx)
    with kanban_db.connect(board=setup["board"]) as conn:
        left_id = kanban_db.create_task(
            conn, title="Left half", assignee="default", project_id=setup["project_id"],
        )
        right_id = kanban_db.create_task(
            conn, title="Right half", assignee="default", project_id=setup["project_id"],
        )
        child_id = kanban_db.create_task(
            conn, title="Dependent work", assignee="default",
            parents=[left_id, right_id], project_id=setup["project_id"],
        )
        left_ref = _project_task_ref(conn, left_id)
        right_ref = _project_task_ref(conn, right_id)

    args = _project_plan_args(
        setup,
        [{
            "action": "merge",
            "reason": "One coherent deliverable is easier to own and verify.",
            "targets": [left_ref, right_ref],
            "replacement": {
                "title": "Combined deliverable",
                "body": "Replace both overlapping work items without deleting history.",
                "assignee": "default",
                "execution_tier": "routine",
            },
        }],
        idempotency_key="steward-merge-dependent-crash",
    )
    probes: list = []

    def crash(*a, **k):
        probes.append(_live_dispatch_probe(setup["board"]))
        raise _CrashInjected("plan_finalize")

    approver = _with_approver(ctx.session)
    with _temporarily_patch(ow, "_finalize_receipt", crash):
        with pytest.raises(_CrashInjected):
            _commit_project_plan(ctx, **args)
    approver.join()

    assert probes and probes[0]["spawned"] == []
    statuses = dict((task_id, status) for task_id, status, _ in probes[0]["rows"])
    assert statuses[left_id] == statuses[right_id] == "archived"
    assert statuses[child_id] not in kanban_db.EXECUTABLE_STATUSES


# ---------------------------------------------------------------------------
# A graph-created Project can accept a LATER plan, through its own anchor
# ---------------------------------------------------------------------------


def _control_rows(board: str, project_id: str) -> list:
    with contextlib.closing(kanban_db.connect(board=board)) as conn:
        return [
            str(row["id"])
            for row in conn.execute(
                "SELECT id FROM tasks WHERE project_id = ? AND task_kind = 'control' "
                "ORDER BY id",
                (project_id,),
            )
        ]


def _graph_created_project(ctx, key="graph-followup"):
    args = _task_graph_args(idempotency_key=key, project_name="Follow-up Project")
    approver = _with_approver(ctx.session)
    result = _commit_task_graph(ctx, **args)
    approver.join()
    assert result["ok"] is True
    return args, result


def test_a_graph_created_project_accepts_a_later_existing_project_plan(ctx):
    """The whole follow-up path: create, read, prepare, commit, verify."""
    _args, created = _graph_created_project(ctx)

    # The owner reads the Project back exactly as the Workspace would.
    snapshot = ow.read_project_snapshot(ctx, created["project_slug"])
    before_ids = {
        task["id"] for column in snapshot["columns"] for task in column["tasks"]
    }
    assert before_ids == {created["root_task_id"], *created["task_ids"]}

    # A later approved plan hangs one new task under the Project's own hidden
    # anchor — an id the caller never sees and never supplies.
    plan = _project_plan_args(
        created,
        [{
            "action": "add",
            "reason": "The owner approved one more bounded step.",
            "title": "Ship the follow-up step",
            "body": "Deliver the newly approved milestone step.",
            "assignee": "default",
            "execution_tier": "deep",
            "existing_parents": [],
            "new_parents": [],
        }],
        idempotency_key="graph-followup-plan",
    )
    approver = _with_approver(ctx.session)
    applied = _commit_project_plan(ctx, **plan)
    approver.join()

    assert applied["ok"] is True
    assert applied["change_count"] == 1
    assert len(applied["created_task_ids"]) == 1
    new_task_id = applied["created_task_ids"][0]

    # The exact changed graph: the original three rows, plus exactly one new
    # task, carrying the deep lane's route pinned under an owner-approved lock.
    after = ow.read_project_snapshot(ctx, created["project_slug"])
    after_ids = {
        task["id"] for column in after["columns"] for task in column["tasks"]
    }
    assert after_ids == before_ids | {new_task_id}
    deep = model_policy.task_assignment_for("default", "anthropic", "deep")
    with contextlib.closing(kanban_db.connect(board=created["board"])) as conn:
        added = kanban_db.get_task(conn, new_task_id)
    assert (added.model_override, added.provider_override) == (deep.model, deep.provider)
    assert (added.reasoning_effort, added.execution_tier) == (
        deep.reasoning_effort, "deep",
    )
    assert kanban_db.policy_lock_error(
        added.model_policy_lock, added.assignee, added.provider_override,
        added.model_override, added.reasoning_effort, added.execution_tier,
    ) is None
    # Activated by the plan's own terminal receipt, never left parked.
    assert added.status != kanban_db.PARKED_STATUS

    # Still exactly one hidden anchor, and it is the one the graph receipt
    # recorded creating.
    assert _control_rows(created["board"], created["project_id"]) == [
        created["anchor_task_id"]
    ]


def test_replaying_the_graph_commit_mints_no_second_anchor_or_task(ctx):
    args, created = _graph_created_project(ctx, key="graph-anchor-replay")

    replayed = _commit_task_graph(ctx, **args)
    assert replayed == created
    # Re-open the board from scratch (a fresh process would too).
    assert _control_rows(created["board"], created["project_id"]) == [
        created["anchor_task_id"]
    ]
    with contextlib.closing(kanban_db.connect(board=created["board"])) as conn:
        work = [
            str(row["id"])
            for row in conn.execute(
                "SELECT id FROM tasks WHERE project_id = ? AND task_kind = 'work'",
                (created["project_id"],),
            )
        ]
    assert sorted(work) == sorted([created["root_task_id"], *created["task_ids"]])


def test_a_project_whose_anchor_is_missing_refuses_a_plan(ctx):
    _args, created = _graph_created_project(ctx, key="graph-anchor-missing")
    with contextlib.closing(kanban_db.connect(board=created["board"])) as conn:
        conn.execute(
            "DELETE FROM tasks WHERE id = ?", (created["anchor_task_id"],)
        )
        conn.commit()

    plan = _project_plan_args(
        created,
        [{
            "action": "add",
            "reason": "This must never land.",
            "title": "Unanchored work",
            "body": "Work with no proven anchor.",
            "assignee": "default",
            "execution_tier": "routine",
            "existing_parents": [],
            "new_parents": [],
        }],
        idempotency_key="graph-anchor-missing-plan",
    )
    with _temporarily_patch(
        ow, "_confirm",
        lambda *a, **k: pytest.fail("an unprovable anchor reached the owner"),
    ):
        with pytest.raises(ow.OwnerWorkspaceError) as excinfo:
            _commit_project_plan(ctx, **plan)
    assert excinfo.value.code == "project_not_owned"


def test_a_project_holding_two_anchors_refuses_a_plan(ctx):
    _args, created = _graph_created_project(ctx, key="graph-anchor-ambiguous")
    with contextlib.closing(kanban_db.connect(board=created["board"])) as conn:
        kanban_db.create_task(
            conn,
            title="a second anchor nobody approved",
            control=True,
            triage=True,
            board=created["board"],
            project_id=created["project_id"],
        )

    plan = _project_plan_args(
        created,
        [{
            "action": "add",
            "reason": "This must never land either.",
            "title": "Ambiguously anchored work",
            "body": "Work whose anchor cannot be proven.",
            "assignee": "default",
            "execution_tier": "routine",
            "existing_parents": [],
            "new_parents": [],
        }],
        idempotency_key="graph-anchor-ambiguous-plan",
    )
    with _temporarily_patch(
        ow, "_confirm",
        lambda *a, **k: pytest.fail("an ambiguous anchor reached the owner"),
    ):
        with pytest.raises(ow.OwnerWorkspaceError) as excinfo:
            _commit_project_plan(ctx, **plan)
    assert excinfo.value.code == "project_not_owned"


def test_the_control_anchor_never_appears_in_any_owner_or_task_only_surface(ctx):
    _args, created = _graph_created_project(ctx, key="graph-anchor-hidden")
    anchor = created["anchor_task_id"]

    snapshot = ow.read_project_snapshot(ctx, created["project_slug"])
    owner_ids = {
        task["id"] for column in snapshot["columns"] for task in column["tasks"]
    }
    assert anchor not in owner_ids
    assert snapshot["board"]["total"] == len(owner_ids)

    steward = ow.project_steward_snapshot(project_id=created["project_id"])
    assert anchor not in json.dumps(steward)

    with contextlib.closing(kanban_db.connect(board=created["board"])) as conn:
        assert anchor not in {task.id for task in kanban_db.list_tasks(conn)}
        assert kanban_db.get_task(conn, anchor) is None
        assert kanban_db.get_task(conn, anchor, include_control=True) is not None


# ---------------------------------------------------------------------------
# A Project created before Projects carried an anchor is migrated, once,
# and only on an owner-approved plan
# ---------------------------------------------------------------------------


_LEGACY_PLAN_CHANGE = {
    "action": "add",
    "reason": "The owner approved one more bounded step.",
    "title": "Ship the migrated step",
    "body": "Deliver the newly approved milestone step.",
    "assignee": "default",
    "execution_tier": "routine",
    "existing_parents": [],
    "new_parents": [],
}


def _receipt_result(ctx, key: str) -> dict:
    with projects_db.connect_closing() as pconn:
        row = pconn.execute(
            "SELECT result_json FROM owner_workspace_receipts "
            "WHERE actor = ? AND profile = ? AND idempotency_key = ?",
            (ctx.actor, ctx.profile, key),
        ).fetchone()
    assert row is not None
    return json.loads(row["result_json"])


def _insert_legacy_receipt(
    ctx,
    *,
    key: str,
    operation: str,
    result: dict,
    project_id: str,
    board: str,
    task_id: str,
) -> str:
    """Insert the exact durable row shape written before anchor authority."""
    digest = ow._digest({"legacy_receipt": key})
    now = int(time.time())
    with projects_db.connect_closing() as pconn:
        ow._ensure_schema(pconn)
        with ow.write_txn(pconn):
            pconn.execute(
                "INSERT INTO owner_workspace_receipts "
                "(actor, profile, idempotency_key, operation, request_digest, "
                "status, project_id, board_slug, task_id, result_json, "
                "created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, 'committed', ?, ?, ?, ?, ?, ?)",
                (
                    ctx.actor,
                    ctx.profile,
                    key,
                    operation,
                    digest,
                    project_id,
                    board,
                    task_id,
                    json.dumps(result),
                    now,
                    now,
                ),
            )
    return digest


def _legacy_graph_project(
    ctx, *, key: str, plan_key: str, graph_mode: str = "new",
) -> dict:
    """Build the exact persisted shape left by a pre-anchor release."""
    assert graph_mode in {"new", "existing"}
    project_salt = "graph-project" if graph_mode == "new" else "existing-project"
    project_id = "p_" + ow._derive_id(ctx, key, project_salt)
    board = "legacy-" + ow._derive_id(ctx, key, "board")[:24]
    name = "Pre-anchor Project"
    description = "Built by the release immediately before anchor authority."

    with projects_db.connect_closing() as pconn:
        projects_db.create_project(
            pconn,
            id=project_id,
            slug=board,
            board_slug=board,
            name=name,
            description=description,
        )
        project = projects_db.get_project(pconn, project_id)
    assert project is not None
    kanban_db.create_board(
        board,
        name=name,
        description=description,
        project_id=project_id,
    )

    with contextlib.closing(kanban_db.connect(board=board)) as conn:
        root_id = kanban_db.create_task(
            conn,
            title="Plan the first milestone",
            body="The original owner-visible milestone.",
            created_by=ctx.actor,
            triage=True,
            idempotency_key="owgraph_" + ow._derive_id(ctx, key, "graph-root"),
            board=board,
            project_id=project_id,
            receipt_owned=(graph_mode == "existing"),
        )
        first_id = kanban_db.create_task(
            conn,
            title="Prepare the first deliverable",
            body="The first pre-anchor task.",
            created_by=ctx.actor,
            board=board,
            project_id=project_id,
        )
        second_id = kanban_db.create_task(
            conn,
            title="Verify the first deliverable",
            body="The second pre-anchor task.",
            created_by=ctx.actor,
            parents=[first_id],
            board=board,
            project_id=project_id,
        )
        earlier_id = kanban_db.create_task(
            conn,
            title="Ship the earlier step",
            body="A task added by the old Project plan path.",
            created_by=ctx.actor,
            idempotency_key="owplan_" + ow._derive_id(ctx, plan_key, "plan-task"),
            board=board,
            project_id=project_id,
        )
        with ow.write_txn(conn):
            conn.execute("UPDATE tasks SET status = 'todo' WHERE id = ?", (root_id,))
            conn.execute("UPDATE tasks SET status = 'ready' WHERE id = ?", (first_id,))
            conn.execute("UPDATE tasks SET status = 'todo' WHERE id = ?", (second_id,))
            conn.execute("UPDATE tasks SET status = 'ready' WHERE id = ?", (earlier_id,))
            for child_id in (first_id, second_id):
                conn.execute(
                    "INSERT OR IGNORE INTO task_links (parent_id, child_id) "
                    "VALUES (?, ?)",
                    (child_id, root_id),
                )

        root = kanban_db.get_task(conn, root_id)
        children = [kanban_db.get_task(conn, first_id), kanban_db.get_task(conn, second_id)]
        assert root is not None and all(child is not None for child in children)
        graph_result = {
            "ok": True,
            "mode": graph_mode,
            "project_id": project_id,
            "project_slug": project.slug,
            "board": board,
            "root_task_id": root_id,
            "root_status": root.status,
            "task_ids": [first_id, second_id],
            "task_statuses": [child.status for child in children],
            "task_count": 2,
        }
        assert set(graph_result) == set(ow._LEGACY_GRAPH_RESULT_FIELDS)
        graph_digest = _insert_legacy_receipt(
            ctx,
            key=key,
            operation="owner_task_graph_commit",
            result=graph_result,
            project_id=project_id,
            board=board,
            task_id=root_id,
        )
        assert graph_digest

        executable_count = conn.execute(
            "SELECT COUNT(*) AS n FROM tasks "
            "WHERE project_id = ? AND task_kind = 'work' "
            "AND status IN ('triage', 'todo', 'ready', 'running', 'blocked', 'review')",
            (project_id,),
        ).fetchone()["n"]
        plan_result = {
            "applied": True,
            "change_count": 1,
            "created_task_ids": [earlier_id],
            "affected_task_ids": [earlier_id],
            "executable_task_count": int(executable_count),
        }
        plan_receipt = {
            "ok": True,
            "project_id": project_id,
            "project_slug": project.slug,
            "board": board,
            "risk_level": "standard",
            **plan_result,
        }
        assert set(plan_receipt) == set(ow._LEGACY_PLAN_RESULT_FIELDS)
        plan_digest = _insert_legacy_receipt(
            ctx,
            key=plan_key,
            operation="owner_project_plan_commit",
            result=plan_receipt,
            project_id=project_id,
            board=board,
            task_id=root_id,
        )
        with ow.write_txn(conn):
            kanban_db._append_event(
                conn,
                root_id,
                "owner_project_plan_applied",
                {
                    "actor": ctx.actor,
                    "profile": ctx.profile,
                    "idempotency_key": plan_key,
                    "request_digest": plan_digest,
                    "trigger": "owner_request",
                    "plan_summary": "The approved pre-anchor plan.",
                    "current_milestone": "Ship the earlier step.",
                    "later_milestones": [],
                    "result": plan_result,
                },
            )

    created = {
        "project_id": project_id,
        "project_slug": project.slug,
        "board": board,
        "root_task_id": root_id,
        "task_ids": [first_id, second_id],
    }
    assert _control_rows(board, project_id) == []
    return created


def test_a_legacy_project_migrates_its_anchor_on_one_approved_plan(ctx):
    created = _legacy_graph_project(
        ctx, key="legacy-anchor", plan_key="legacy-anchor-old-plan",
    )
    before = ow.read_project_snapshot(ctx, created["project_slug"])
    before_ids = {
        task["id"] for column in before["columns"] for task in column["tasks"]
    }

    plan = _project_plan_args(
        created, [dict(_LEGACY_PLAN_CHANGE)], idempotency_key="legacy-anchor-plan",
    )
    approver = _with_approver(ctx.session)
    applied = _commit_project_plan(ctx, **plan)
    approver.join()

    assert applied["ok"] is True
    assert applied["change_count"] == 1
    assert _control_rows(created["board"], created["project_id"]) == [
        applied["anchor_task_id"]
    ]

    after = ow.read_project_snapshot(ctx, created["project_slug"])
    after_ids = {
        task["id"] for column in after["columns"] for task in column["tasks"]
    }
    assert after_ids == before_ids | set(applied["created_task_ids"])
    assert applied["anchor_task_id"] not in after_ids
    with contextlib.closing(kanban_db.connect(board=created["board"])) as conn:
        anchor = kanban_db.get_task(
            conn, applied["anchor_task_id"], include_control=True,
        )
        assert kanban_db.get_task(conn, applied["anchor_task_id"]) is None
        added = kanban_db.get_task(conn, applied["created_task_ids"][0])
    assert anchor.status == _ANCHOR_STATUS
    assert (anchor.assignee, anchor.execution_tier, anchor.model_policy_lock) == (
        None, None, None,
    )
    assert added.status != kanban_db.PARKED_STATUS


def test_a_legacy_existing_project_migrates_from_its_receipt_bound_root(ctx):
    created = _legacy_graph_project(
        ctx,
        key="legacy-existing-anchor",
        plan_key="legacy-existing-old-plan",
        graph_mode="existing",
    )

    plan = _project_plan_args(
        created,
        [dict(_LEGACY_PLAN_CHANGE)],
        idempotency_key="legacy-existing-plan",
    )
    with contextlib.closing(kanban_db.connect(board=created["board"])) as conn:
        root = conn.execute(
            "SELECT project_id, task_kind, created_by, idempotency_key, "
            "owner_receipt_bound FROM tasks WHERE id = ?",
            (created["root_task_id"],),
        ).fetchone()
    assert root is not None
    assert dict(root) == {
        "project_id": created["project_id"],
        "task_kind": "work",
        "created_by": ctx.actor,
        "idempotency_key": "owgraph_"
        + ow._derive_id(ctx, "legacy-existing-anchor", "graph-root"),
        "owner_receipt_bound": 1,
    }
    approver = _with_approver(ctx.session)
    applied = _commit_project_plan(ctx, **plan)
    approver.join()

    assert applied["ok"] is True
    assert _control_rows(created["board"], created["project_id"]) == [
        applied["anchor_task_id"]
    ]


def test_a_legacy_existing_project_rejects_an_unrelated_root_identity(ctx):
    created = _legacy_graph_project(
        ctx,
        key="legacy-existing-wrong-root",
        plan_key="legacy-existing-wrong-root-old-plan",
        graph_mode="existing",
    )
    with contextlib.closing(kanban_db.connect(board=created["board"])) as conn:
        with ow.write_txn(conn):
            conn.execute(
                "UPDATE tasks SET idempotency_key = ? WHERE id = ?",
                ("unrelated-owner-root", created["root_task_id"]),
            )

    with pytest.raises(ow.OwnerWorkspaceError) as excinfo:
        _commit_project_plan(
            ctx,
            **_project_plan_args(
                created,
                [dict(_LEGACY_PLAN_CHANGE)],
                idempotency_key="legacy-existing-wrong-root-plan",
            ),
        )
    assert excinfo.value.code == "project_not_owned"
    assert _control_rows(created["board"], created["project_id"]) == []


def test_a_denied_plan_never_migrates_a_legacy_anchor(ctx):
    created = _legacy_graph_project(
        ctx, key="legacy-denied", plan_key="legacy-denied-old-plan",
    )
    board, project_id = created["board"], created["project_id"]
    with contextlib.closing(kanban_db.connect(board=board)) as conn:
        before = (
            conn.execute("SELECT COUNT(*) AS n FROM tasks").fetchone()["n"],
            conn.execute("SELECT COUNT(*) AS n FROM task_events").fetchone()["n"],
        )

    plan = _project_plan_args(
        created, [dict(_LEGACY_PLAN_CHANGE)], idempotency_key="legacy-denied-plan",
    )
    with _temporarily_patch(
        ow, "_confirm", lambda *a, **k: {"approved": False, "reason": "denied"},
    ):
        denied = _commit_project_plan(ctx, **plan)

    assert denied == {
        "ok": False, "error": "confirmation_denied", "reason": "denied",
    }
    assert _control_rows(board, project_id) == []
    with contextlib.closing(kanban_db.connect(board=board)) as conn:
        assert before == (
            conn.execute("SELECT COUNT(*) AS n FROM tasks").fetchone()["n"],
            conn.execute("SELECT COUNT(*) AS n FROM task_events").fetchone()["n"],
        )


@pytest.mark.parametrize("crash_point", ["before_apply", "before_finalize"])
def test_a_crashed_legacy_migration_adopts_its_staged_anchor(ctx, crash_point):
    created = _legacy_graph_project(
        ctx,
        key=f"legacy-crash-{crash_point}",
        plan_key=f"legacy-crash-old-plan-{crash_point}",
    )
    plan_key = f"legacy-crash-plan-{crash_point}"
    plan = _project_plan_args(
        created, [dict(_LEGACY_PLAN_CHANGE)], idempotency_key=plan_key,
    )

    def crash(*_args, **_kwargs):
        raise _CrashInjected(crash_point)

    target, attr = (
        (kanban_db, "apply_owner_project_plan")
        if crash_point == "before_apply"
        else (ow, "_finalize_receipt")
    )
    approver = _with_approver(ctx.session)
    with _temporarily_patch(target, attr, crash):
        with pytest.raises(_CrashInjected):
            _commit_project_plan(ctx, **plan)
    approver.join()

    staged = _control_rows(created["board"], created["project_id"])
    assert len(staged) == 1

    _expire_lock(ctx, plan_key)
    if crash_point == "before_apply":
        approver = _with_approver(ctx.session)
        replayed = _commit_project_plan(ctx, **plan)
        approver.join()
    else:
        with _temporarily_patch(
            ow, "_confirm",
            lambda *a, **k: pytest.fail("a recovered plan asked again"),
        ):
            replayed = _commit_project_plan(ctx, **plan)

    assert replayed["ok"] is True
    assert replayed["anchor_task_id"] == staged[0]
    assert _control_rows(created["board"], created["project_id"]) == staged
    with contextlib.closing(kanban_db.connect(board=created["board"])) as conn:
        matching = [
            task for task in kanban_db.list_tasks(conn)
            if task.title == _LEGACY_PLAN_CHANGE["title"]
        ]
    assert len(matching) == 1
    assert matching[0].status != kanban_db.PARKED_STATUS


def test_a_migrated_anchor_is_durable_authority_for_every_later_plan(ctx):
    created = _legacy_graph_project(
        ctx, key="legacy-durable", plan_key="legacy-durable-old-plan",
    )
    approver = _with_approver(ctx.session)
    migrated = _commit_project_plan(
        ctx,
        **_project_plan_args(
            created,
            [dict(_LEGACY_PLAN_CHANGE)],
            idempotency_key="legacy-durable-plan-1",
        ),
    )
    approver.join()
    assert migrated["ok"] is True

    assert "anchor_task_id" not in _receipt_result(ctx, "legacy-durable")
    assert "anchor_task_id" not in _receipt_result(
        ctx, "legacy-durable-old-plan",
    )

    approver = _with_approver(ctx.session)
    later = _commit_project_plan(
        ctx,
        **_project_plan_args(
            created,
            [dict(_LEGACY_PLAN_CHANGE, title="Ship the next step")],
            idempotency_key="legacy-durable-plan-2",
        ),
    )
    approver.join()

    assert later["ok"] is True
    assert later["anchor_task_id"] == migrated["anchor_task_id"]
    assert _control_rows(created["board"], created["project_id"]) == [
        migrated["anchor_task_id"]
    ]


@pytest.mark.parametrize("controls", [1, 2])
def test_a_legacy_project_with_unaccounted_controls_refuses_a_plan(ctx, controls):
    created = _legacy_graph_project(
        ctx,
        key=f"legacy-foreign-{controls}",
        plan_key=f"legacy-foreign-old-plan-{controls}",
    )
    with contextlib.closing(kanban_db.connect(board=created["board"])) as conn:
        for index in range(controls):
            kanban_db.create_task(
                conn,
                title=f"a control row nobody approved {index}",
                control=True,
                triage=True,
                board=created["board"],
                project_id=created["project_id"],
            )

    plan = _project_plan_args(
        created,
        [dict(_LEGACY_PLAN_CHANGE)],
        idempotency_key=f"legacy-foreign-plan-{controls}",
    )
    with _temporarily_patch(
        ow, "_confirm",
        lambda *a, **k: pytest.fail("an unprovable anchor reached the owner"),
    ):
        with pytest.raises(ow.OwnerWorkspaceError) as excinfo:
            _commit_project_plan(ctx, **plan)
    assert excinfo.value.code == "project_not_owned"
    assert len(_control_rows(created["board"], created["project_id"])) == controls


def test_a_corrupted_modern_receipt_is_never_treated_as_legacy(ctx):
    _args, created = _graph_created_project(ctx, key="modern-anchor-corrupt")
    with contextlib.closing(kanban_db.connect(board=created["board"])) as conn:
        conn.execute("DELETE FROM tasks WHERE id = ?", (created["anchor_task_id"],))
        conn.commit()
    result = _receipt_result(ctx, "modern-anchor-corrupt")
    result.pop("anchor_task_id")
    assert "park_generation" in result and "parked_task_ids" in result
    with projects_db.connect_closing() as pconn:
        with ow.write_txn(pconn):
            pconn.execute(
                "UPDATE owner_workspace_receipts SET result_json = ? "
                "WHERE actor = ? AND profile = ? AND idempotency_key = ?",
                (
                    json.dumps(result),
                    ctx.actor,
                    ctx.profile,
                    "modern-anchor-corrupt",
                ),
            )

    plan = _project_plan_args(
        created,
        [dict(_LEGACY_PLAN_CHANGE)],
        idempotency_key="modern-anchor-corrupt-plan",
    )
    with _temporarily_patch(
        ow, "_confirm",
        lambda *a, **k: pytest.fail("a corrupted modern receipt reached the owner"),
    ):
        with pytest.raises(ow.OwnerWorkspaceError) as excinfo:
            _commit_project_plan(ctx, **plan)
    assert excinfo.value.code == "project_not_owned"
    assert _control_rows(created["board"], created["project_id"]) == []


@pytest.mark.parametrize(
    "corruption",
    [
        "modern_marker",
        "missing_field",
        "wrong_slug",
        "wrong_board",
        "invalid_risk",
        "not_applied",
    ],
)
def test_every_legacy_plan_receipt_must_match_the_exact_old_contract(
    ctx, corruption,
):
    graph_key = f"legacy-plan-shape-{corruption}"
    old_plan_key = f"legacy-plan-shape-old-{corruption}"
    created = _legacy_graph_project(ctx, key=graph_key, plan_key=old_plan_key)
    receipt = _receipt_result(ctx, old_plan_key)
    if corruption == "modern_marker":
        receipt["anchor_task_id"] = "t_unproven"
    elif corruption == "missing_field":
        receipt.pop("executable_task_count")
    elif corruption == "wrong_slug":
        receipt["project_slug"] = "another-project"
    elif corruption == "wrong_board":
        receipt["board"] = "another-board"
    elif corruption == "invalid_risk":
        receipt["risk_level"] = "low"
    else:
        receipt["applied"] = False
    with projects_db.connect_closing() as pconn:
        with ow.write_txn(pconn):
            pconn.execute(
                "UPDATE owner_workspace_receipts SET result_json = ? "
                "WHERE actor = ? AND profile = ? AND idempotency_key = ?",
                (json.dumps(receipt), ctx.actor, ctx.profile, old_plan_key),
            )

    plan = _project_plan_args(
        created,
        [dict(_LEGACY_PLAN_CHANGE)],
        idempotency_key=f"legacy-plan-shape-new-{corruption}",
    )
    with _temporarily_patch(
        ow,
        "_confirm",
        lambda *a, **k: pytest.fail("an unproven legacy Project reached approval"),
    ):
        with pytest.raises(ow.OwnerWorkspaceError) as excinfo:
            _commit_project_plan(ctx, **plan)
    assert excinfo.value.code == "project_not_owned"
    assert _control_rows(created["board"], created["project_id"]) == []


def test_a_concurrent_identical_legacy_migration_adopts_one_anchor(ctx):
    created = _legacy_graph_project(
        ctx,
        key="legacy-concurrent",
        plan_key="legacy-concurrent-old-plan",
    )
    migration_key = "owanchor_" + ow._derive_id(
        ctx,
        created["project_id"],
        ow._LEGACY_ANCHOR_MIGRATION_SALT,
    )
    raced: dict[str, str] = {}

    def approve_after_identical_migration(*_args, **_kwargs):
        with contextlib.closing(kanban_db.connect(board=created["board"])) as conn:
            raced["anchor"] = kanban_db.create_task(
                conn,
                title="Pre-anchor Project",
                body="Built by the release immediately before anchor authority.",
                created_by=ctx.actor,
                triage=True,
                control=True,
                idempotency_key=migration_key,
                board=created["board"],
                project_id=created["project_id"],
            )
        return {"approved": True}

    with _temporarily_patch(ow, "_confirm", approve_after_identical_migration):
        applied = _commit_project_plan(
            ctx,
            **_project_plan_args(
                created,
                [dict(_LEGACY_PLAN_CHANGE)],
                idempotency_key="legacy-concurrent-new-plan",
            ),
        )

    assert applied["ok"] is True
    assert applied["anchor_task_id"] == raced["anchor"]
    assert _control_rows(created["board"], created["project_id"]) == [
        raced["anchor"]
    ]


def test_a_different_anchor_committed_during_confirmation_is_rejected(ctx):
    created = _legacy_graph_project(
        ctx,
        key="legacy-authority-race",
        plan_key="legacy-authority-race-old-plan",
    )
    raced: dict[str, str] = {}

    def approve_after_authority_changed(*_args, **_kwargs):
        with contextlib.closing(kanban_db.connect(board=created["board"])) as conn:
            raced["anchor"] = kanban_db.create_task(
                conn,
                title="A different control authority",
                created_by=ctx.actor,
                triage=True,
                control=True,
                idempotency_key="owanchor_foreign",
                board=created["board"],
                project_id=created["project_id"],
            )
        receipt = {
            "ok": True,
            "project_id": created["project_id"],
            "project_slug": created["project_slug"],
            "board": created["board"],
            "anchor_task_id": raced["anchor"],
            "risk_level": "standard",
            "applied": True,
            "change_count": 0,
            "created_task_ids": [],
            "affected_task_ids": [],
            "parked_task_ids": [],
            "parked_dependents": [],
            "park_generation": "race-generation",
        }
        _insert_legacy_receipt(
            ctx,
            key="legacy-authority-race-other-plan",
            operation="owner_project_plan_commit",
            result=receipt,
            project_id=created["project_id"],
            board=created["board"],
            task_id=raced["anchor"],
        )
        return {"approved": True}

    with _temporarily_patch(ow, "_confirm", approve_after_authority_changed):
        with pytest.raises(ow.OwnerWorkspaceError) as excinfo:
            _commit_project_plan(
                ctx,
                **_project_plan_args(
                    created,
                    [dict(_LEGACY_PLAN_CHANGE)],
                    idempotency_key="legacy-authority-race-new-plan",
                ),
            )
    assert excinfo.value.code == "crash_recovery_failed"
    assert _control_rows(created["board"], created["project_id"]) == [
        raced["anchor"]
    ]


def test_a_modern_anchor_cannot_change_during_confirmation(ctx):
    graph_key = "modern-authority-race"
    _args, created = _graph_created_project(ctx, key=graph_key)
    original_anchor = created["anchor_task_id"]
    raced: dict[str, str] = {}

    def approve_after_authority_changed(*_args, **_kwargs):
        with contextlib.closing(kanban_db.connect(board=created["board"])) as conn:
            conn.execute("DELETE FROM tasks WHERE id = ?", (original_anchor,))
            conn.commit()
            raced["anchor"] = kanban_db.create_task(
                conn,
                title="A replacement control authority",
                created_by=ctx.actor,
                triage=True,
                control=True,
                idempotency_key="owanchor_modern_foreign",
                board=created["board"],
                project_id=created["project_id"],
            )
        receipt = _receipt_result(ctx, graph_key)
        receipt["anchor_task_id"] = raced["anchor"]
        with projects_db.connect_closing() as pconn:
            with ow.write_txn(pconn):
                pconn.execute(
                    "UPDATE owner_workspace_receipts SET result_json = ?, task_id = ? "
                    "WHERE actor = ? AND profile = ? AND idempotency_key = ?",
                    (
                        json.dumps(receipt),
                        raced["anchor"],
                        ctx.actor,
                        ctx.profile,
                        graph_key,
                    ),
                )
        return {"approved": True}

    with _temporarily_patch(ow, "_confirm", approve_after_authority_changed):
        with pytest.raises(ow.OwnerWorkspaceError) as excinfo:
            _commit_project_plan(
                ctx,
                **_project_plan_args(
                    created,
                    [dict(_LEGACY_PLAN_CHANGE)],
                    idempotency_key="modern-authority-race-new-plan",
                ),
            )
    assert excinfo.value.code == "crash_recovery_failed"
    assert _control_rows(created["board"], created["project_id"]) == [
        raced["anchor"]
    ]


def test_a_legacy_migration_that_loses_its_lease_mints_no_anchor(ctx):
    created = _legacy_graph_project(
        ctx,
        key="legacy-lease-loss",
        plan_key="legacy-lease-loss-old-plan",
    )
    new_plan_key = "legacy-lease-loss-new-plan"
    with contextlib.closing(kanban_db.connect(board=created["board"])) as conn:
        before = (
            conn.execute("SELECT COUNT(*) AS n FROM tasks").fetchone()["n"],
            conn.execute("SELECT COUNT(*) AS n FROM task_events").fetchone()["n"],
        )

    def approve_after_lease_loss(*_args, **_kwargs):
        with projects_db.connect_closing() as pconn:
            with ow.write_txn(pconn):
                pconn.execute(
                    "UPDATE owner_workspace_receipts SET lock_token = ? "
                    "WHERE actor = ? AND profile = ? AND idempotency_key = ?",
                    ("stolen", ctx.actor, ctx.profile, new_plan_key),
                )
        return {"approved": True}

    with _temporarily_patch(ow, "_confirm", approve_after_lease_loss):
        with pytest.raises(ow.OwnerWorkspaceError) as excinfo:
            _commit_project_plan(
                ctx,
                **_project_plan_args(
                    created,
                    [dict(_LEGACY_PLAN_CHANGE)],
                    idempotency_key=new_plan_key,
                ),
            )
    assert excinfo.value.code == "lease_lost"
    assert _control_rows(created["board"], created["project_id"]) == []
    with contextlib.closing(kanban_db.connect(board=created["board"])) as conn:
        assert before == (
            conn.execute("SELECT COUNT(*) AS n FROM tasks").fetchone()["n"],
            conn.execute("SELECT COUNT(*) AS n FROM task_events").fetchone()["n"],
        )
