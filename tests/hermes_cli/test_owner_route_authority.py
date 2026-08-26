"""Route authority invariants for owner-kernel work.

Covers the facts the owner's approval actually rests on:

* receipt provenance resolves from the canonical owner-executor store, not from
  whatever profile a route change happens to be scoped to;
* a malformed committed receipt fails the route change closed;
* every receipt-owned executable task ends up with an exact lock or paused;
* a locked task's assignee is immutable for its whole run;
* the bootstrap anchor is structurally non-executable;
* a partially migrated board projects an unknown pin, never "unlocked".
"""

from __future__ import annotations

import contextlib
import json
import sqlite3
import threading
import time

import pytest

from hermes_cli import config as hermes_config
from hermes_cli import kanban_db, owner_workspace as ow, projects_db
from hermes_cli.profiles import get_profile_dir
from hermes_constants import reset_hermes_home_override, set_hermes_home_override
from plugins.dashboard_auth.raphael_workspace import model_policy
from tools import approval

_OWNER_PROFILE = "default"
_OWNER_ROUTE = {
    "model": {"provider": "anthropic", "default": "claude-opus-5"},
    "agent": {"reasoning_effort": "max"},
    "fallback_providers": [],
}
# A NAMED role that genuinely has two admitted providers, so its route can be
# moved from one to the other without inventing an unadmitted lane. The
# verifier deliberately has only the OpenAI route and the builder only the
# Claude one, so neither can stand in for this.
_NAMED_ROLE = "raphael-planner"
_NAMED_ROLE_ROUTE = {
    "model": {"provider": "openai-codex", "default": "gpt-5.6-sol"},
    "agent": {"reasoning_effort": "max"},
    "fallback_providers": [],
}


def _write_profile_route(profile: str, config: dict) -> None:
    directory = get_profile_dir(profile)
    directory.mkdir(parents=True, exist_ok=True)
    token = set_hermes_home_override(str(directory))
    try:
        hermes_config.save_config(dict(config))
    finally:
        reset_hermes_home_override(token)


def _auto_approve(session_key: str, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        with approval._lock:
            queue = approval._gateway_queues.get(session_key)
            aid = queue[0].approval_id if queue else None
        if aid:
            approval.resolve_gateway_approval(session_key, "once", approval_id=aid)
            return
        time.sleep(0.005)
    raise TimeoutError("no approval was queued")


def _with_approver(session_key: str) -> threading.Thread:
    approval.register_gateway_notify(session_key, lambda data: None)
    thread = threading.Thread(target=_auto_approve, args=(session_key,))
    thread.start()
    return thread


@pytest.fixture
def ctx():
    return ow.OwnerContext(
        actor=_OWNER_PROFILE, profile=_OWNER_PROFILE, session="run_route"
    )


@pytest.fixture
def bootstrapped(ctx):
    """A real committed bootstrap receipt in the owner-executor projects.db."""
    thread = _with_approver(ctx.session)
    result = ow.bootstrap(ctx, idempotency_key="route-1", name="Route Project")
    thread.join()
    assert result["ok"] is True
    return result


# ---------------------------------------------------------------------------
# The bootstrap anchor is structurally non-executable
# ---------------------------------------------------------------------------


def test_bootstrap_anchor_is_a_non_executable_control_task(bootstrapped):
    with kanban_db.connect(board=bootstrapped["board"]) as conn:
        kind = conn.execute(
            "SELECT task_kind, assignee, model_policy_lock FROM tasks WHERE id = ?",
            (bootstrapped["task_id"],),
        ).fetchone()
        assert kind["task_kind"] == "control"
        assert kind["assignee"] is None
        assert kind["model_policy_lock"] is None
        # Invisible to every executable reader.
        assert kanban_db.get_task(conn, bootstrapped["task_id"]) is None
        assert kanban_db.get_control_task(conn, bootstrapped["task_id"]) is not None


def test_generic_specify_and_decompose_cannot_promote_the_anchor(bootstrapped):
    anchor = bootstrapped["task_id"]
    with kanban_db.connect(board=bootstrapped["board"]) as conn:
        assert kanban_db.specify_triage_task(
            conn, anchor, assignee="raphael-builder"
        ) is False
        assert kanban_db.decompose_triage_task(
            conn,
            anchor,
            root_assignee="raphael-builder",
            children=[{"title": "sneak in executable work"}],
        ) is None
        assert kanban_db.assign_task(conn, anchor, "raphael-builder") is False
        assert kanban_db.claim_task(conn, anchor) is None
        row = conn.execute(
            "SELECT status, assignee FROM tasks WHERE id = ?", (anchor,)
        ).fetchone()
        assert (row["status"], row["assignee"]) == ("triage", None)


def test_the_dispatcher_never_sees_the_anchor(bootstrapped):
    anchor = bootstrapped["task_id"]
    with kanban_db.connect(board=bootstrapped["board"]) as conn:
        # Even parked directly in the dispatcher's own queue column.
        with kanban_db.write_txn(conn):
            conn.execute(
                "UPDATE tasks SET status = 'ready' WHERE id = ?", (anchor,)
            )
        rows = conn.execute(
            "SELECT id FROM tasks WHERE status = 'ready' AND task_kind = 'work'"
        ).fetchall()
        assert [row["id"] for row in rows] == []


def test_a_control_task_cannot_be_created_with_a_route():
    with kanban_db.connect() as conn:
        with pytest.raises(ValueError, match="control task cannot carry"):
            kanban_db.create_task(
                conn, title="anchor", control=True, assignee="raphael-builder",
            )


# ---------------------------------------------------------------------------
# Receipt provenance resolves from the canonical owner-executor store
# ---------------------------------------------------------------------------


def test_receipt_provenance_ignores_a_named_profile_override(bootstrapped):
    """A named-role route change must still see the default profile's receipts.

    The receipt was committed under the owner-executor (default) profile's
    ``projects.db``. Scoping HERMES_HOME to a NAMED profile — which is exactly
    what a named-role route write does — must not make that receipt invisible.
    Nothing is monkeypatched here: this is the real receipt store.
    """
    boards = ow._owner_receipt_task_ids()
    assert bootstrapped["board"] in boards
    assert bootstrapped["task_id"] in boards[bootstrapped["board"]]

    token = set_hermes_home_override(str(get_profile_dir(_NAMED_ROLE)))
    try:
        # The named profile's own projects.db is empty/absent...
        assert not projects_db.projects_db_path().is_file()
        # ...and the receipt is still resolved from the canonical store.
        scoped = ow._owner_receipt_task_ids()
    finally:
        reset_hermes_home_override(token)
    assert scoped == boards


def test_named_role_route_change_fences_default_receipt_owned_work(bootstrapped):
    """Integration: real default-profile receipt, real named-profile write.

    No monkeypatched receipt set. A named role's task created by a committed
    owner graph receipt must be pinned before that role's own route moves.
    """
    _write_profile_route(_NAMED_ROLE, _NAMED_ROLE_ROUTE)
    model_policy.enroll_profile(_NAMED_ROLE)

    # Real receipt-owned executable work held by the named role, classified but
    # not yet locked (the state a pre-lock graph commit leaves behind).
    with kanban_db.connect(board=bootstrapped["board"]) as conn:
        task_id = kanban_db.create_task(
            conn,
            title="check the work",
            assignee=_NAMED_ROLE,
            execution_tier="routine",
            project_id=bootstrapped["project_id"],
        )
    _record_graph_receipt(
        bootstrapped["project_id"], bootstrapped["board"], [task_id]
    )

    # The named-role route write scopes HERMES_HOME to that profile, and moves
    # it to that role's OTHER admitted provider.
    _write_profile_route(_NAMED_ROLE, {
        "model": {"provider": "anthropic", "default": "claude-sonnet-5"},
        "agent": {"reasoning_effort": "max"},
        "fallback_providers": [],
    })

    with kanban_db.connect(board=bootstrapped["board"]) as conn:
        pinned = kanban_db.get_task(conn, task_id)
    # Frozen on the route it was already approved for, not the new selection.
    assert (pinned.provider_override, pinned.model_override) == (
        "openai-codex", "gpt-5.6-sol",
    )
    assert pinned.model_policy_lock
    assert kanban_db.policy_lock_error(
        pinned.model_policy_lock,
        _NAMED_ROLE,
        "openai-codex",
        "gpt-5.6-sol",
        "max",
        "routine",
    ) is None


def _record_graph_receipt(project_id: str, board: str, task_ids: list[str]) -> None:
    """Commit a real owner_task_graph_commit receipt row for these tasks."""
    with projects_db.connect_closing() as pconn:
        ow._ensure_schema(pconn)
        now = int(time.time())
        with ow.write_txn(pconn):
            pconn.execute(
                "INSERT INTO owner_workspace_receipts ("
                " actor, profile, idempotency_key, operation, request_digest,"
                " status, project_id, board_slug, result_json, created_at, updated_at"
                ") VALUES (?, ?, ?, ?, ?, 'committed', ?, ?, ?, ?, ?)",
                (
                    _OWNER_PROFILE, _OWNER_PROFILE, f"graph-{task_ids[0]}",
                    "owner_task_graph_commit", "d" * 64,
                    project_id, board,
                    json.dumps({
                        "ok": True,
                        "project_id": project_id,
                        "board": board,
                        "root_task_id": task_ids[0],
                        "task_ids": task_ids[1:],
                    }),
                    now, now,
                ),
            )


# ---------------------------------------------------------------------------
# Malformed / unprovable receipts fail the route change closed
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "result",
    [
        # Not a readable result at all.
        "not-a-dict",
        {},
        # Successful, but proves nothing about the work it created.
        {"ok": True, "project_id": "p", "root_task_id": "t_1"},
        {"ok": True, "project_id": "p", "task_ids": ["t_1"]},
        # Unreadable work identity.
        {"ok": True, "project_id": "p", "root_task_id": "t_1", "task_ids": [1, 2]},
        # A non-success that is not an explicit terminal no-op/conflict.
        {"ok": False, "project_id": "p", "error": "something went wrong"},
        {"ok": False, "project_id": "p"},
        # Successful, but no Project to bind the work to.
        {"ok": True, "root_task_id": "t_1", "task_ids": []},
    ],
)
def test_a_malformed_committed_receipt_fails_the_route_change_closed(result):
    _seed_receipt("owner_task_graph_commit", result, project_id=None)
    with pytest.raises(ow.OwnerWorkspaceError) as excinfo:
        ow._owner_receipt_task_ids()
    assert excinfo.value.code == "execution_state_busy"


@pytest.mark.parametrize("error", ["conflict", "noop", "denied"])
def test_an_explicit_terminal_no_op_receipt_may_own_nothing(error):
    _seed_receipt(
        "owner_project_plan_commit",
        {"ok": False, "error": error, "project_id": "p_none", "change_count": 0},
        project_id="p_none",
    )
    assert ow._owner_receipt_task_ids() == {}


def test_a_receipt_whose_project_has_no_board_fails_closed():
    _seed_receipt(
        "owner_task_graph_commit",
        {"ok": True, "project_id": "p_ghost", "root_task_id": "t_1", "task_ids": []},
        project_id="p_ghost",
    )
    with pytest.raises(ow.OwnerWorkspaceError) as excinfo:
        ow._owner_receipt_task_ids()
    assert excinfo.value.code == "execution_state_busy"


def test_a_receipt_whose_board_ownership_is_unprovable_fails_closed(bootstrapped):
    """An unprovable binding is not someone else's problem — it is fail-closed."""
    kanban_db.write_board_metadata(
        bootstrapped["board"], project_id="p_someone_else",
    )
    with pytest.raises(ow.OwnerWorkspaceError) as excinfo:
        ow._owner_receipt_task_ids()
    assert excinfo.value.code == "crash_recovery_failed"


def _seed_receipt(operation: str, result, *, project_id) -> None:
    with projects_db.connect_closing() as pconn:
        ow._ensure_schema(pconn)
        if project_id:
            projects_db.create_project(pconn, id=project_id, name="Ghost")
        now = int(time.time())
        with ow.write_txn(pconn):
            pconn.execute(
                "INSERT INTO owner_workspace_receipts ("
                " actor, profile, idempotency_key, operation, request_digest,"
                " status, project_id, result_json, created_at, updated_at"
                ") VALUES (?, ?, ?, ?, ?, 'committed', ?, ?, ?, ?)",
                (
                    _OWNER_PROFILE, _OWNER_PROFILE, f"bad-{operation}-{now}",
                    operation, "d" * 64, project_id,
                    json.dumps(result), now, now,
                ),
            )


# ---------------------------------------------------------------------------
# Fencing covers EVERY receipt-owned executable task
# ---------------------------------------------------------------------------


def test_a_fully_specified_but_unlocked_task_is_not_skipped(monkeypatch):
    """Naming all three route fields is not the same as being approved."""
    _write_profile_route(_OWNER_PROFILE, _OWNER_ROUTE)
    with kanban_db.connect() as conn:
        task_id = kanban_db.create_task(
            conn,
            title="fully specified owner work",
            assignee=_OWNER_PROFILE,
            model_override="claude-opus-5",
            provider_override="anthropic",
            reasoning_effort="max",
            execution_tier="routine",
        )
        assert kanban_db.count_unpinned_owner_tasks(conn, task_ids=[task_id]) == 1
    monkeypatch.setattr(
        ow, "_owner_receipt_task_ids", lambda: {kanban_db.DEFAULT_BOARD: {task_id}},
    )

    assert ow.fence_effective_task_routes(_OWNER_PROFILE) == [task_id]
    with kanban_db.connect() as conn:
        task = kanban_db.get_task(conn, task_id)
    assert task.model_policy_lock
    assert task.status != "blocked"


def test_an_unpinnable_receipt_owned_task_is_paused_not_left_runnable(monkeypatch):
    """A route the policy cannot admit is paused with a reapproval requirement."""
    _write_profile_route(_OWNER_PROFILE, _OWNER_ROUTE)
    with kanban_db.connect() as conn:
        # No execution tier at all: the policy can mint no authority for it.
        legacy_id = kanban_db.create_task(
            conn, title="legacy owner work", assignee=_OWNER_PROFILE,
        )
        with kanban_db.write_txn(conn):
            conn.execute(
                "UPDATE tasks SET status = 'ready' WHERE id = ?", (legacy_id,)
            )
    monkeypatch.setattr(
        ow, "_owner_receipt_task_ids", lambda: {kanban_db.DEFAULT_BOARD: {legacy_id}},
    )

    assert ow.fence_effective_task_routes(_OWNER_PROFILE) == []
    with kanban_db.connect() as conn:
        task = kanban_db.get_task(conn, legacy_id)
        events = kanban_db.list_events(conn, legacy_id)
        # Refused by every dispatcher path: it is not in the ready queue...
        ready = conn.execute(
            "SELECT id FROM tasks WHERE status = 'ready' AND task_kind = 'work'"
        ).fetchall()
        assert legacy_id not in [row["id"] for row in ready]
        # ...and cannot be claimed even directly.
        assert kanban_db.claim_task(conn, legacy_id) is None
    assert task.status == "blocked"
    assert task.model_policy_lock is None
    unapproved = [e for e in events if e.kind == "model_route_unapproved"]
    assert unapproved and unapproved[-1].payload["reapproval_required"] is True
    # Sticky: the readiness sweep must not put it back in the work pool.
    with kanban_db.connect() as conn:
        kanban_db.recompute_ready(conn)
        assert kanban_db.get_task(conn, legacy_id).status == "blocked"
    # Owner-facing copy: plain English, no ids, paths, models or agent names.
    reason = unapproved[-1].payload["reason"]
    assert "approved again" in reason
    for forbidden in (legacy_id, _OWNER_PROFILE, "claude", "anthropic", "/"):
        assert forbidden not in reason


def test_an_ordinary_non_receipt_task_is_never_fenced(monkeypatch):
    _write_profile_route(_OWNER_PROFILE, _OWNER_ROUTE)
    with kanban_db.connect() as conn:
        manual_id = kanban_db.create_task(
            conn, title="manual card", assignee=_OWNER_PROFILE,
        )
    monkeypatch.setattr(ow, "_owner_receipt_task_ids", lambda: {})
    assert ow.fence_effective_task_routes(_OWNER_PROFILE) == []
    with kanban_db.connect() as conn:
        task = kanban_db.get_task(conn, manual_id)
    assert task.model_policy_lock is None
    assert task.status != "blocked"
    assert task.model_override is None


# ---------------------------------------------------------------------------
# The fence holds every board's native dispatch lock
# ---------------------------------------------------------------------------


def _exposed_owner_task(monkeypatch, **columns) -> str:
    """One unlocked receipt-owned task the fence is told to cover."""
    _write_profile_route(_OWNER_PROFILE, _OWNER_ROUTE)
    with kanban_db.connect() as conn:
        task_id = kanban_db.create_task(
            conn,
            title="receipt-owned owner work",
            assignee=_OWNER_PROFILE,
            execution_tier="routine",
        )
        if columns:
            assignments = ", ".join(f"{name} = ?" for name in columns)
            with kanban_db.write_txn(conn):
                conn.execute(
                    f"UPDATE tasks SET {assignments} WHERE id = ?",
                    (*columns.values(), task_id),
                )
    monkeypatch.setattr(
        ow, "_owner_receipt_task_ids", lambda: {kanban_db.DEFAULT_BOARD: {task_id}},
    )
    return task_id


def _dispatch_lock_is_held(board: str) -> bool:
    """Whether the board's dispatch lock is currently taken by someone."""
    try:
        with kanban_db.board_dispatch_lock(board, wait_seconds=0):
            return False
    except TimeoutError:
        return True


@pytest.mark.parametrize(
    "column", ["status", "claim_lock", "worker_pid", "current_run_id"],
)
def test_active_receipt_owned_work_refuses_the_route_change(monkeypatch, column):
    """A worker already executing cannot be re-authorized underneath itself.

    Rewriting the row's route columns would neither stop that process nor
    approve what it is doing, so the whole change is refused instead. An
    expired-but-unreleased claim and a dangling run id count too: either can
    still be adopted or reported against.
    """
    value = "running" if column == "status" else (
        4242 if column in {"worker_pid", "current_run_id"} else "held"
    )
    task_id = _exposed_owner_task(monkeypatch, **{column: value})

    with pytest.raises(ow.OwnerWorkspaceError) as excinfo:
        ow.fence_effective_task_routes(_OWNER_PROFILE)
    assert excinfo.value.code == "execution_state_busy"

    with kanban_db.connect() as conn:
        after = kanban_db.get_task(conn, task_id)
    # Nothing was pinned and nothing was parked: the row a worker holds is
    # exactly as it was, and the caller must not go on to write its config.
    assert after.model_policy_lock is None
    assert after.model_override is None
    assert after.status != "blocked"


def test_the_fence_holds_the_dispatch_lock_from_discovery_through_pinning(monkeypatch):
    """No dispatcher tick can claim between "unlocked" and "now pinned"."""
    task_id = _exposed_owner_task(monkeypatch)
    assert _dispatch_lock_is_held(kanban_db.DEFAULT_BOARD) is False

    observed: list[bool] = []
    real_pin = kanban_db.pin_effective_task_routes

    def observing_pin(conn, **kwargs):
        # Exactly the instant a dispatcher tick would try to enter its own
        # critical section: after discovery, before the pin lands.
        observed.append(_dispatch_lock_is_held(kanban_db.DEFAULT_BOARD))
        return real_pin(conn, **kwargs)

    monkeypatch.setattr(kanban_db, "pin_effective_task_routes", observing_pin)
    assert ow.fence_effective_task_routes(_OWNER_PROFILE) == [task_id]

    assert observed == [True]
    # ...and it is released again once the change is fenced.
    assert _dispatch_lock_is_held(kanban_db.DEFAULT_BOARD) is False


def test_a_busy_dispatch_lock_fails_the_route_change_closed(monkeypatch):
    """An unfenceable board is refused, never fenced without the lock."""
    task_id = _exposed_owner_task(monkeypatch)

    with kanban_db.board_dispatch_lock(kanban_db.DEFAULT_BOARD, wait_seconds=0):
        with pytest.raises(ow.OwnerWorkspaceError) as excinfo:
            ow.fence_effective_task_routes(_OWNER_PROFILE)
    assert excinfo.value.code == "execution_state_busy"

    with kanban_db.connect() as conn:
        assert kanban_db.get_task(conn, task_id).model_policy_lock is None


def test_every_board_lock_is_taken_in_one_deterministic_order(monkeypatch):
    """Sorted acquisition is what makes two concurrent fences undeadlockable.

    Two fences that took the same boards in different orders could each hold
    one and wait for the other. A single global order removes that shape, so
    the invariant asserted here is the ordering itself.
    """
    _write_profile_route(_OWNER_PROFILE, _OWNER_ROUTE)
    boards = ["zzz-second-board", "aaa-first-board"]
    owned: dict[str, set[str]] = {}
    for board in boards:
        with kanban_db.connect(board=board) as conn:
            owned[board] = {kanban_db.create_task(
                conn,
                title="receipt-owned owner work",
                assignee=_OWNER_PROFILE,
                execution_tier="routine",
            )}
    monkeypatch.setattr(ow, "_owner_receipt_task_ids", lambda: dict(owned))

    order: list[str] = []
    real_lock = kanban_db.board_dispatch_lock

    @contextlib.contextmanager
    def recording_lock(board, **kwargs):
        with real_lock(board, **kwargs) as slug:
            order.append(slug)
            yield slug

    monkeypatch.setattr(kanban_db, "board_dispatch_lock", recording_lock)
    pinned = ow.fence_effective_task_routes(_OWNER_PROFILE)

    assert order == sorted(boards)
    assert set(pinned) == set().union(*owned.values())


def test_two_concurrent_fences_over_the_same_boards_both_complete():
    """Liveness, not just ordering: neither fence is left waiting forever."""
    boards = ["shared-a", "shared-b"]
    roles = [_OWNER_PROFILE, _NAMED_ROLE]
    _write_profile_route(_OWNER_PROFILE, _OWNER_ROUTE)
    _write_profile_route(_NAMED_ROLE, _NAMED_ROLE_ROUTE)

    owned: dict[str, set[str]] = {board: set() for board in boards}
    expected: dict[str, set[str]] = {role: set() for role in roles}
    for board in boards:
        with kanban_db.connect(board=board) as conn:
            for role in roles:
                task_id = kanban_db.create_task(
                    conn,
                    title="receipt-owned owner work",
                    assignee=role,
                    execution_tier="routine",
                )
                owned[board].add(task_id)
                expected[role].add(task_id)

    start = threading.Barrier(len(roles))
    results: dict[str, object] = {}

    def _fence(role: str) -> None:
        try:
            start.wait(timeout=30)
            results[role] = set(ow.fence_effective_task_routes(role))
        except BaseException as exc:  # reported as a failure, never a hang
            results[role] = exc

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(ow, "_owner_receipt_task_ids", lambda: dict(owned))
        threads = [
            threading.Thread(target=_fence, args=(role,), daemon=True)
            for role in roles
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=60)
        assert all(not thread.is_alive() for thread in threads)

    assert results == expected


# ---------------------------------------------------------------------------
# A dependency edge's two ends are not symmetric
# ---------------------------------------------------------------------------


def test_only_executable_work_can_be_a_dependency_child(bootstrapped):
    """A control row is a legitimate PARENT and never a legitimate CHILD.

    Owner-approved work hangs under its Project's anchor, so control-as-parent
    has to work. The reverse would attach ``linked`` events and inherited
    notification subscriptions to a row no executable path is allowed to see,
    so every control-child direction is refused BEFORE any of that is written.
    """
    board, anchor = bootstrapped["board"], bootstrapped["task_id"]
    with kanban_db.connect(board=board) as conn:
        other_anchor = kanban_db.create_task(
            conn, title="second anchor", control=True,
        )
        work_id = kanban_db.create_task(
            conn, title="executable work", assignee=_OWNER_PROFILE,
        )
        # Give both would-be parents something a child WOULD inherit, so the
        # "no subscription reached the control row" assertion is not vacuous.
        with kanban_db.write_txn(conn):
            for parent_id in (work_id, anchor):
                conn.execute(
                    "INSERT INTO kanban_notify_subs "
                    "(task_id, platform, chat_id, created_at) VALUES (?, ?, ?, ?)",
                    (parent_id, "cli", "owner-chat", int(time.time())),
                )

        # Control as PARENT of executable work: the shape the anchor exists for.
        kanban_db.link_tasks(conn, anchor, work_id)
        assert [
            event.kind for event in kanban_db.list_events(conn, work_id)
        ].count("linked") == 1

        for parent_id, child_id in (
            (work_id, anchor),          # work -> control
            (anchor, other_anchor),     # control -> control
        ):
            with pytest.raises(ValueError, match="only executable work"):
                kanban_db.link_tasks(conn, parent_id, child_id)

        # Refused before anything was written: no edge, and no event or
        # inherited subscription on the control row it was aimed at.
        edges = conn.execute(
            "SELECT parent_id, child_id FROM task_links ORDER BY parent_id"
        ).fetchall()
        assert [(row["parent_id"], row["child_id"]) for row in edges] == [
            (anchor, work_id)
        ]
        for control_id in (anchor, other_anchor):
            assert [
                event.kind
                for event in kanban_db.list_events(
                    conn, control_id, include_control=True
                )
            ].count("linked") == 0
        # ``other_anchor`` had no subscription of its own, so anything here
        # could only have been inherited through the edge that was refused.
        assert conn.execute(
            "SELECT COUNT(*) AS n FROM kanban_notify_subs WHERE task_id = ?",
            (other_anchor,),
        ).fetchone()["n"] == 0


# ---------------------------------------------------------------------------
# A locked task's assignee is immutable for its whole run
# ---------------------------------------------------------------------------


def _locked_task(conn, *, assignee: str = "raphael-builder") -> str:
    route = model_policy.task_assignment_for(assignee, "anthropic", "deep")
    return kanban_db.create_task(
        conn,
        title="approved owner work",
        assignee=assignee,
        model_override=route.model,
        provider_override=route.provider,
        reasoning_effort=route.reasoning_effort,
        execution_tier="deep",
        model_policy_lock=kanban_db.mint_policy_lock(
            assignee, route.provider, route.model, route.reasoning_effort, "deep",
        ),
    )


def test_reassigning_a_locked_task_is_refused():
    with kanban_db.connect() as conn:
        task_id = _locked_task(conn)
        before = kanban_db.get_task(conn, task_id)
        with pytest.raises(RuntimeError, match="the owner approved that exact"):
            kanban_db.assign_task(conn, task_id, "raphael-verifier")
        after = kanban_db.get_task(conn, task_id)
    assert (after.assignee, after.model_override, after.model_policy_lock) == (
        before.assignee, before.model_override, before.model_policy_lock,
    )


def test_unassigning_a_locked_task_is_refused_so_no_stale_lock_survives():
    with kanban_db.connect() as conn:
        task_id = _locked_task(conn)
        with pytest.raises(RuntimeError, match="would strand the lock"):
            kanban_db.assign_task(conn, task_id, None)
        after = kanban_db.get_task(conn, task_id)
    assert after.assignee == "raphael-builder"
    assert after.model_policy_lock


def test_only_an_exact_owner_approved_replacement_route_can_move_a_locked_task():
    with kanban_db.connect() as conn:
        task_id = _locked_task(conn)
        # The verifier's one admitted lane — independent of the Claude family
        # that wrote the code, which is exactly why it is the replacement here.
        approved = model_policy.task_assignment_for(
            "raphael-verifier", "openai-codex", "deep"
        )
        replacement = {
            "assignee": "raphael-verifier",
            "provider": approved.provider,
            "model": approved.model,
            "reasoning_effort": approved.reasoning_effort,
            "execution_tier": "deep",
            "model_policy_lock": kanban_db.mint_policy_lock(
                "raphael-verifier", approved.provider, approved.model,
                approved.reasoning_effort, "deep",
            ),
        }
        # A replacement that does not name the target role is refused.
        with pytest.raises(RuntimeError, match="not 'raphael-designer'"):
            kanban_db.assign_task(
                conn, task_id, "raphael-designer", approved_route=replacement,
            )
        # A replacement whose lock does not bind its own route is refused.
        with pytest.raises(RuntimeError, match="does not bind this route"):
            kanban_db.assign_task(
                conn, task_id, "raphael-verifier",
                approved_route={**replacement, "model_policy_lock": (
                    "raphael:v1:" + "a" * 64
                )},
            )
        # A partial replacement is refused.
        with pytest.raises(RuntimeError, match="must state exactly"):
            kanban_db.assign_task(
                conn, task_id, "raphael-verifier",
                approved_route={"assignee": "raphael-verifier"},
            )
        # The exact approved replacement lands atomically.
        assert kanban_db.assign_task(
            conn, task_id, "raphael-verifier", approved_route=replacement,
        )
        moved = kanban_db.get_task(conn, task_id)
    assert moved.assignee == "raphael-verifier"
    assert moved.model_override == approved.model
    assert moved.model_policy_lock == replacement["model_policy_lock"]
    assert kanban_db.policy_lock_error(
        moved.model_policy_lock, moved.assignee, moved.provider_override,
        moved.model_override, moved.reasoning_effort, moved.execution_tier,
    ) is None


def test_internal_review_and_rework_do_not_silently_repin_a_locked_task():
    """Review/rework must be separately approved work, never a silent re-pin."""
    with kanban_db.connect() as conn:
        task_id = _locked_task(conn)
        with kanban_db.write_txn(conn):
            conn.execute(
                "UPDATE tasks SET status = 'ready' WHERE id = ?", (task_id,)
            )
        claimed = kanban_db.claim_task(conn, task_id)
        assert claimed is not None
        # The worker states its OWN run id, so the live-claim guard is
        # satisfied legitimately and the handoff reaches the route authority
        # instead of being refused one layer earlier.
        with pytest.raises(RuntimeError, match="the owner approved that exact"):
            kanban_db.request_review(
                conn,
                task_id,
                reviewer="raphael-verifier",
                summary="please check",
                expected_run_id=claimed.current_run_id,
            )
        after = kanban_db.get_task(conn, task_id)
    # Refused inside the transaction: the live run is still the builder's.
    assert after.assignee == "raphael-builder"
    assert after.model_policy_lock
    assert after.status == "running"


def test_request_changes_cannot_silently_repin_a_locked_task():
    """The rework handback is the same authority boundary as the handoff."""
    builder_route = model_policy.task_assignment_for(
        "raphael-builder", "anthropic", "deep"
    )
    verifier_route = model_policy.task_assignment_for(
        "raphael-verifier", "openai-codex", "deep"
    )
    with kanban_db.connect() as conn:
        # An ordinary unlocked card, so the historical review handoff really
        # runs and leaves the implementer provenance request_changes reads
        # back. It carries no execution tier and no lock, so it is not yet
        # owner-governed — an owner-governed row without a lock cannot be
        # claimed at all, which is a different guard tested elsewhere.
        task_id = kanban_db.create_task(
            conn,
            title="work under review",
            assignee="raphael-builder",
            model_override=builder_route.model,
            provider_override=builder_route.provider,
            reasoning_effort=builder_route.reasoning_effort,
        )
        claimed = kanban_db.claim_task(conn, task_id)
        assert claimed is not None
        assert kanban_db.request_review(
            conn,
            task_id,
            reviewer="raphael-verifier",
            summary="please check",
            expected_run_id=claimed.current_run_id,
        ) is True
        # The owner then approves that review work for the verifier: the task
        # is now locked to the role holding it.
        with kanban_db.write_txn(conn):
            conn.execute(
                "UPDATE tasks SET model_override = ?, provider_override = ?, "
                "reasoning_effort = ?, execution_tier = 'deep', "
                "model_policy_lock = ? WHERE id = ?",
                (
                    verifier_route.model,
                    verifier_route.provider,
                    verifier_route.reasoning_effort,
                    kanban_db.mint_policy_lock(
                        "raphael-verifier",
                        verifier_route.provider,
                        verifier_route.model,
                        verifier_route.reasoning_effort,
                        "deep",
                    ),
                    task_id,
                ),
            )
        assert kanban_db.claim_review_task(conn, task_id) is not None
        with pytest.raises(RuntimeError, match="the owner approved that exact"):
            kanban_db.request_changes(conn, task_id, reason="needs another pass")
        after = kanban_db.get_task(conn, task_id)
    # The handback never landed: the lock still names the role that holds it.
    assert after.assignee == "raphael-verifier"
    assert after.model_policy_lock
    assert kanban_db.policy_lock_error(
        after.model_policy_lock, after.assignee, after.provider_override,
        after.model_override, after.reasoning_effort, after.execution_tier,
    ) is None


def test_the_dispatcher_default_assignee_cannot_adopt_a_locked_task():
    with kanban_db.connect() as conn:
        task_id = _locked_task(conn)
        with kanban_db.write_txn(conn):
            conn.execute(
                "UPDATE tasks SET assignee = NULL WHERE id = ?", (task_id,)
            )
        with pytest.raises(RuntimeError):
            kanban_db.role_transition_route(conn, task_id, "raphael-verifier")


# ---------------------------------------------------------------------------
# Migrated owner work: receipt-owned, but carrying no route columns at all
# ---------------------------------------------------------------------------


def _migrated_owner_task(conn, *, status: str = "scheduled") -> str:
    """Owner work as a board upgraded from a pre-route build carries it.

    ``owner_receipt_bound`` is the ONLY proof of owner ownership on such a row:
    no execution tier, no lock, and a status the readiness guard and the
    rollout fence never scan.
    """
    task_id = kanban_db.create_task(
        conn, title="migrated owner work", assignee="raphael-builder",
    )
    with kanban_db.write_txn(conn):
        conn.execute(
            "UPDATE tasks SET owner_receipt_bound = 1, execution_tier = NULL, "
            "model_policy_lock = NULL, model_override = NULL, "
            "provider_override = NULL, reasoning_effort = NULL, status = ? "
            "WHERE id = ?",
            (status, task_id),
        )
    return task_id


@pytest.mark.parametrize("status", ["scheduled", "todo"])
def test_migrated_owner_work_refuses_every_reassignment_surface(status):
    with kanban_db.connect() as conn:
        task_id = _migrated_owner_task(conn, status=status)
        before = kanban_db.get_task(conn, task_id)

        with pytest.raises(RuntimeError, match="approved again"):
            kanban_db.assign_task(conn, task_id, "raphael-verifier")
        with pytest.raises(RuntimeError, match="approved again"):
            kanban_db.assign_task(conn, task_id, None)
        with pytest.raises(RuntimeError, match="owner-governed"):
            kanban_db.set_model_override(conn, task_id, "claude-opus-5", "anthropic")
        with pytest.raises(RuntimeError, match="owner-governed"):
            kanban_db.set_reasoning_effort(conn, task_id, "low")

        after = kanban_db.get_task(conn, task_id)
    assert (
        after.assignee,
        after.model_override,
        after.provider_override,
        after.reasoning_effort,
        after.model_policy_lock,
        after.status,
    ) == (
        before.assignee, None, None, None, None, status,
    )


def test_migrated_owner_work_cannot_be_rerouted_then_promoted_into_a_run():
    """The reroute surfaces are what make the readiness guard's mint safe.

    If an operator could name any model on migrated owner work, the guard would
    mint an "exact admitted lock" for a route nobody approved. With the reroute
    refused, the row has nothing to mint from and stays out of the work pool.
    """
    with kanban_db.connect() as conn:
        task_id = _migrated_owner_task(conn, status="todo")
        with pytest.raises(RuntimeError):
            kanban_db.set_model_override(conn, task_id, "claude-opus-5", "anthropic")
        kanban_db.recompute_ready(conn)
        parked = kanban_db.get_task(conn, task_id)
        with pytest.raises(RuntimeError, match="approved route lock"):
            kanban_db.assert_claimable_route(conn, task_id)
        with pytest.raises(RuntimeError, match="approved route lock"):
            kanban_db.claim_task(conn, task_id)
    assert parked.status not in kanban_db.EXECUTABLE_STATUSES
    assert parked.model_policy_lock is None


def test_migrated_owner_work_refuses_a_triage_specify_role_change():
    with kanban_db.connect() as conn:
        task_id = _migrated_owner_task(conn, status="triage")
        with pytest.raises(RuntimeError, match="approved again"):
            kanban_db.specify_triage_task(
                conn, task_id, assignee="raphael-verifier", author="operator",
            )
        after = kanban_db.get_task(conn, task_id)
    assert (after.assignee, after.status) == ("raphael-builder", "triage")


def test_one_exact_reapproved_route_moves_migrated_owner_work_atomically():
    with kanban_db.connect() as conn:
        task_id = _migrated_owner_task(conn)
        approved = model_policy.task_assignment_for(
            "raphael-verifier", "openai-codex", "deep"
        )
        replacement = {
            "assignee": "raphael-verifier",
            "provider": approved.provider,
            "model": approved.model,
            "reasoning_effort": approved.reasoning_effort,
            "execution_tier": "deep",
            "model_policy_lock": kanban_db.mint_policy_lock(
                "raphael-verifier", approved.provider, approved.model,
                approved.reasoning_effort, "deep",
            ),
        }
        # A replacement that is not exactly authorized changes nothing.
        with pytest.raises(RuntimeError, match="must state exactly"):
            kanban_db.assign_task(
                conn, task_id, "raphael-verifier",
                approved_route={"assignee": "raphael-verifier"},
            )
        assert kanban_db.get_task(conn, task_id).model_policy_lock is None

        assert kanban_db.assign_task(
            conn, task_id, "raphael-verifier", approved_route=replacement,
        )
        moved = kanban_db.get_task(conn, task_id)
    assert moved.assignee == "raphael-verifier"
    assert moved.model_policy_lock == replacement["model_policy_lock"]
    assert kanban_db.policy_lock_error(
        moved.model_policy_lock, moved.assignee, moved.provider_override,
        moved.model_override, moved.reasoning_effort, moved.execution_tier,
    ) is None


def test_an_ordinary_manual_task_keeps_every_route_mutation():
    """Nothing here narrows a card no owner receipt owns."""
    with kanban_db.connect() as conn:
        task_id = kanban_db.create_task(
            conn, title="manual card", assignee="engineer",
        )
        assert kanban_db.set_model_override(conn, task_id, "some-model", "some-provider")
        assert kanban_db.set_reasoning_effort(conn, task_id, "low")
        assert kanban_db.assign_task(conn, task_id, "someone-else")
        after = kanban_db.get_task(conn, task_id)
    assert (after.assignee, after.model_override, after.reasoning_effort) == (
        "someone-else", "some-model", "low",
    )


def test_an_unlocked_task_keeps_its_ordinary_reassignment_behaviour():
    with kanban_db.connect() as conn:
        task_id = kanban_db.create_task(
            conn, title="manual card", assignee="engineer",
        )
        assert kanban_db.assign_task(conn, task_id, "someone-else")
        assert kanban_db.get_task(conn, task_id).assignee == "someone-else"
        assert kanban_db.assign_task(conn, task_id, None)
        assert kanban_db.get_task(conn, task_id).assignee is None


# ---------------------------------------------------------------------------
# Partially migrated / unreadable pin schemas
# ---------------------------------------------------------------------------


def test_a_schema_read_failure_raises_snapshot_unavailable():
    class Broken:
        def execute(self, *args, **kwargs):
            raise sqlite3.OperationalError("database is locked")

    with pytest.raises(ow.OwnerWorkspaceError) as excinfo:
        ow.owner_task_pin_select(Broken(), "t")
    assert excinfo.value.code == "snapshot_unavailable"


def test_a_lock_column_without_its_bound_columns_projects_an_invalid_pin(tmp_path):
    """A partial lock schema is unknown routing, never "unlocked"."""
    conn = sqlite3.connect(tmp_path / "partial.db")
    conn.row_factory = sqlite3.Row
    conn.execute(
        "CREATE TABLE tasks (id TEXT, assignee TEXT, model_policy_lock TEXT)"
    )
    conn.execute(
        "INSERT INTO tasks VALUES ('t_1', 'raphael-builder', 'raphael:v1:"
        + "a" * 64 + "')"
    )
    try:
        tail = ow.owner_task_pin_select(conn, "t")
        # The lock column is still projected, so the pin is checkable...
        assert "model_policy_lock" in tail
        row = conn.execute(
            f"SELECT t.id{tail} FROM tasks t WHERE t.id = 't_1'"
        ).fetchone()
        pin = ow.owner_task_route_pin(row)
    finally:
        conn.close()
    # ...and it reads as INVALID, not as None/unlocked.
    assert pin is not None
    assert pin.valid is False


def test_a_pre_lock_schema_still_projects_as_genuinely_unlocked(tmp_path):
    conn = sqlite3.connect(tmp_path / "legacy.db")
    conn.row_factory = sqlite3.Row
    conn.execute("CREATE TABLE tasks (id TEXT, assignee TEXT)")
    conn.execute("INSERT INTO tasks VALUES ('t_1', 'raphael-builder')")
    try:
        assert ow.owner_task_pin_select(conn, "t") == ""
        row = conn.execute("SELECT t.id FROM tasks t").fetchone()
        assert ow.owner_task_route_pin(row) is None
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# All-or-nothing multi-role batch
# ---------------------------------------------------------------------------


# Only these roles have an admitted route on BOTH providers, so only these can
# be flipped from one provider to the other without naming a forbidden lane.
_TWO_PROVIDER_ROLES = ["default", "raphael-business", "raphael-planner"]


def test_a_batch_applies_every_role_or_none():
    roles = list(_TWO_PROVIDER_ROLES)
    # The role that fails is the last one the deterministic order reaches, so
    # the two that already succeeded are the ones that have to be rolled back.
    failing = sorted(roles)[-1]
    for role in roles:
        _write_profile_route(role, _batch_target(role, "anthropic"))
    before = {role: _raw_config(role) for role in roles}

    calls: list[str] = []

    def _apply(profile, payload):
        calls.append(profile)
        if profile == failing:
            raise RuntimeError("this role's provider is at its limit")
        _write_profile_route(profile, payload)
        return "r" * 32

    changes = [(role, _batch_target(role, "openai-codex")) for role in roles]
    with pytest.raises(hermes_config.ProfileRouteBatchError) as excinfo:
        hermes_config.apply_profile_route_batch(changes, apply_one=_apply)

    assert excinfo.value.profile == failing
    # Deterministic lock/apply order, and every profile restored byte for byte.
    assert calls == sorted(roles)
    assert {role: _raw_config(role) for role in roles} == before


def test_a_batch_that_succeeds_reports_every_revision():
    roles = ["default", "raphael-business"]
    for role in roles:
        _write_profile_route(role, _batch_target(role, "anthropic"))

    def _apply(profile, payload):
        _write_profile_route(profile, payload)
        return f"rev-{profile}"

    revisions = hermes_config.apply_profile_route_batch(
        [(role, _batch_target(role, "openai-codex")) for role in roles],
        apply_one=_apply,
    )
    assert revisions == {role: f"rev-{role}" for role in roles}


def test_a_batch_refuses_a_duplicated_profile():
    with pytest.raises(hermes_config.ProfileRouteBatchError, match="twice"):
        hermes_config.apply_profile_route_batch(
            [("default", {}), ("default", {})], apply_one=lambda *a: "x",
        )


def test_a_batch_and_a_single_write_never_deadlock_against_each_other(monkeypatch):
    """One global lock order.

    The batch used to take every profile lock first and only enter
    ``_CONFIG_LOCK`` inside ``apply_one``, while an ordinary single-profile
    writer took them the other way round. The window below is exactly the one
    that inversion needed: a single writer already holds ``_CONFIG_LOCK`` and
    is about to take a profile lock the batch wants. Before the fix the two
    waited on each other forever, with no timeout and no recovery.
    """
    roles = ["default", "raphael-business"]
    # The single writer contends for a profile the batch also holds — without
    # that overlap there is no inversion to reproduce.
    contended = "raphael-business"
    for role in roles:
        _write_profile_route(role, _batch_target(role, "anthropic"))

    real_lock = ow.profile_route_lock
    holding = threading.Event()
    release = threading.Event()
    single_done = threading.Event()

    @contextlib.contextmanager
    def _paused_lock(profile):
        if profile == contended and not holding.is_set():
            holding.set()
            assert release.wait(timeout=20), "the batch never reached its locks"
        with real_lock(profile):
            yield

    monkeypatch.setattr(ow, "profile_route_lock", _paused_lock)

    def _single_writer():
        _write_profile_route(contended, _batch_target(contended, "anthropic"))
        single_done.set()

    def _apply(profile, payload):
        _write_profile_route(profile, payload)
        return f"rev-{profile}"

    writer = threading.Thread(target=_single_writer, daemon=True)
    writer.start()
    try:
        assert holding.wait(timeout=20), "the single writer never started"
        # Daemon threads: a regression here is a real deadlock, and the test
        # process must still be able to exit and report it.
        batch = threading.Thread(
            target=lambda: hermes_config.apply_profile_route_batch(
                [(role, _batch_target(role, "openai-codex")) for role in roles],
                apply_one=_apply,
            ),
            daemon=True,
        )
        batch.start()
        # The batch is now waiting on _CONFIG_LOCK, which the paused single
        # writer holds. Releasing the writer must let BOTH finish.
        time.sleep(0.1)
        release.set()
        batch.join(timeout=20)
        assert not batch.is_alive(), "the batch deadlocked"
    finally:
        release.set()
        writer.join(timeout=20)

    assert single_done.is_set(), "the single-profile write deadlocked"
    for role in roles:
        assert _batch_target(role, "openai-codex")["model"]["default"] in _raw_config(role)


def _journal_path():
    return hermes_config._route_batch_journal_path()


def test_a_batch_journals_every_role_before_it_writes_anything():
    """Power loss is not a caught exception, so the undo record has to be on
    disk before the first live config is touched."""
    roles = ["default", "raphael-business"]
    for role in roles:
        _write_profile_route(role, _batch_target(role, "anthropic"))
    before = {role: _raw_config(role) for role in roles}
    seen: list = []

    def _apply(profile, payload):
        # What a crash at this instant would leave behind on disk.
        seen.append(json.loads(_journal_path().read_text(encoding="utf-8")))
        _write_profile_route(profile, payload)
        return f"rev-{profile}"

    hermes_config.apply_profile_route_batch(
        [(role, _batch_target(role, "openai-codex")) for role in roles],
        apply_one=_apply,
    )

    assert seen, "apply_one never ran"
    journalled = {
        entry["profile"]: entry["text"] for entry in seen[0]["roles"]
    }
    assert journalled == before
    # Retired once the whole operation owns its outcome.
    assert not _journal_path().exists()


def test_an_interrupted_batch_is_put_back_at_startup():
    """The exact crash the journal exists for: some roles written, the
    enrollment/audit half never reached, and no in-process anything left."""
    roles = ["default", "raphael-business"]
    for role in roles:
        _write_profile_route(role, _batch_target(role, "anthropic"))
    before = {role: _raw_config(role) for role in roles}

    class _PowerLoss(BaseException):
        """Not an ``Exception``: in-process rollback must not see it."""

    def _apply(profile, payload):
        _write_profile_route(profile, payload)
        if profile == sorted(roles)[-1]:
            raise _PowerLoss()
        return f"rev-{profile}"

    with pytest.raises(_PowerLoss):
        hermes_config.apply_profile_route_batch(
            [(role, _batch_target(role, "openai-codex")) for role in roles],
            apply_one=_apply,
        )

    # Split across two routes, with a journal naming what to put back.
    assert {role: _raw_config(role) for role in roles} != before
    assert _journal_path().exists()

    restored = hermes_config.reconcile_profile_route_batch_journal()

    assert sorted(restored) == sorted(roles)
    assert {role: _raw_config(role) for role in roles} == before
    assert not _journal_path().exists()
    # Idempotent: a second start finds nothing to do.
    assert hermes_config.reconcile_profile_route_batch_journal() == []


@pytest.fixture(autouse=True)
def _clear_route_journal_latch():
    """A blocked journal is process-wide state; never leak it between tests."""
    hermes_config._clear_route_journal_block()
    yield
    hermes_config._clear_route_journal_block()


def test_an_unreadable_journal_is_fatal_and_blocks_every_route_write():
    """Returning normally let startup continue and let the NEXT batch truncate
    the only record of what to put back."""
    path = _journal_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{not json", encoding="utf-8")

    with pytest.raises(hermes_config.RouteJournalUnreconciled):
        hermes_config.reconcile_profile_route_batch_journal()
    # Left in place for an operator: it is the only rollback evidence.
    assert path.exists()
    # And nothing may read or write a route until it is resolved.
    with pytest.raises(hermes_config.RouteJournalUnreconciled):
        hermes_config.assert_route_journal_reconciled()
    with pytest.raises(hermes_config.RouteJournalUnreconciled):
        hermes_config.current_route_revision()
    with pytest.raises(hermes_config.RouteJournalUnreconciled):
        _write_profile_route("default", _batch_target("default", "anthropic"))
    with pytest.raises(hermes_config.RouteJournalUnreconciled):
        hermes_config.apply_profile_route_batch(
            [("default", _batch_target("default", "anthropic"))],
            apply_one=lambda *a: "x" * 32,
        )


def test_a_journal_shape_this_build_never_wrote_is_fatal():
    path = _journal_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"version": 2, "roles": "nope"}), encoding="utf-8")

    with pytest.raises(hermes_config.RouteJournalUnreconciled):
        hermes_config.reconcile_profile_route_batch_journal()
    assert path.exists()


def test_a_journal_that_cannot_be_retired_is_fatal(monkeypatch):
    """A journal that outlives its batch is read by the next start as an
    interrupted one, so swallowing the unlink failure was not an option."""
    roles = ["default", "raphael-business"]
    for role in roles:
        _write_profile_route(role, _batch_target(role, "anthropic"))

    real_unlink = hermes_config.Path.unlink

    def _refuse(self, *args, **kwargs):
        if self == _journal_path():
            raise OSError("cannot unlink")
        return real_unlink(self, *args, **kwargs)

    journal = _journal_path()
    monkeypatch.setattr(hermes_config.Path, "unlink", _refuse)
    with pytest.raises(hermes_config.RouteJournalUnreconciled):
        hermes_config.apply_profile_route_batch(
            [(role, _batch_target(role, "openai-codex")) for role in roles],
            apply_one=lambda profile, payload: (
                _write_profile_route(profile, payload) or f"rev-{profile}"
            ),
        )
    assert journal.exists()
    with pytest.raises(hermes_config.RouteJournalUnreconciled):
        hermes_config.assert_route_journal_reconciled()


@pytest.mark.parametrize("fault", ["short_write", "fsync"])
def test_a_journal_that_cannot_be_made_durable_refuses_the_batch(monkeypatch, fault):
    """A journal that is not on disk when the power goes is no journal at all,
    so the batch is refused before the first live config is touched."""
    roles = ["default", "raphael-business"]
    for role in roles:
        _write_profile_route(role, _batch_target(role, "anthropic"))
    before = {role: _raw_config(role) for role in roles}

    real_write, real_fsync = hermes_config.os.write, hermes_config.os.fsync
    if fault == "short_write":
        monkeypatch.setattr(
            hermes_config.os, "write",
            lambda fd, data: real_write(fd, data[: len(data) // 2]),
        )
    else:
        def _fail_fsync(fd):
            raise OSError("fsync failed")

        monkeypatch.setattr(hermes_config.os, "fsync", _fail_fsync)

    with pytest.raises(hermes_config.ProfileRouteBatchError, match="undo"):
        hermes_config.apply_profile_route_batch(
            [(role, _batch_target(role, "openai-codex")) for role in roles],
            apply_one=lambda profile, payload: pytest.fail(
                "a batch wrote a live config with no durable journal"
            ),
        )
    monkeypatch.setattr(hermes_config.os, "write", real_write)
    monkeypatch.setattr(hermes_config.os, "fsync", real_fsync)

    # Nothing was changed, and nothing was left half-journalled.
    assert {role: _raw_config(role) for role in roles} == before
    assert not _journal_path().exists()
    hermes_config.assert_route_journal_reconciled()


def test_a_reconciled_machine_accepts_the_next_batch_again():
    """The block is until reconciliation completes, not forever."""
    roles = ["default"]
    _write_profile_route("default", _batch_target("default", "anthropic"))
    before = _raw_config("default")

    class _PowerLoss(BaseException):
        pass

    def _apply(profile, payload):
        _write_profile_route(profile, payload)
        raise _PowerLoss()

    with pytest.raises(_PowerLoss):
        hermes_config.apply_profile_route_batch(
            [(role, _batch_target(role, "openai-codex")) for role in roles],
            apply_one=_apply,
        )
    assert _journal_path().exists()

    assert hermes_config.reconcile_profile_route_batch_journal() == ["default"]
    assert _raw_config("default") == before

    revisions = hermes_config.apply_profile_route_batch(
        [(role, _batch_target(role, "openai-codex")) for role in roles],
        apply_one=lambda profile, payload: (
            _write_profile_route(profile, payload) or f"rev-{profile}"
        ),
    )
    assert revisions == {"default": "rev-default"}
    assert not _journal_path().exists()


def test_a_committed_batch_is_rolled_forward_not_back():
    """The exact split the phase exists for: the audit record is committed, so
    reverting the configs would leave a durable claim of a change that is gone."""
    roles = ["default", "raphael-business"]
    for role in roles:
        _write_profile_route(role, _batch_target(role, "anthropic"))
    after = {}

    def _apply(profile, payload):
        _write_profile_route(profile, payload)
        return f"rev-{profile}"

    def _after_write(_revisions):
        hermes_config.mark_route_batch_enrollment_written(None)
        hermes_config.mark_route_batch_audit_committed()
        after.update({role: _raw_config(role) for role in roles})
        raise RuntimeError("the audit writer reported an unprovable rollback")

    with pytest.raises(RuntimeError, match="unprovable rollback"):
        hermes_config.apply_profile_route_batch(
            [(role, _batch_target(role, "openai-codex")) for role in roles],
            apply_one=_apply,
            after_write=_after_write,
        )

    # Rolled FORWARD: the new routes stand and the journal is retired.
    assert {role: _raw_config(role) for role in roles} == after
    assert not _journal_path().exists()
    hermes_config.assert_route_journal_reconciled()


def test_a_crash_after_the_audit_commit_is_rolled_forward_at_startup():
    roles = ["default", "raphael-business"]
    for role in roles:
        _write_profile_route(role, _batch_target(role, "anthropic"))
    before = {role: _raw_config(role) for role in roles}

    # Exactly what a crash between the audit commit and the journal's
    # retirement leaves on disk.
    _write_committed_journal(roles)
    for role in roles:
        _write_profile_route(role, _batch_target(role, "openai-codex"))
    live = {role: _raw_config(role) for role in roles}
    assert live != before

    assert hermes_config.reconcile_profile_route_batch_journal() == []
    assert {role: _raw_config(role) for role in roles} == live
    assert not _journal_path().exists()


def test_an_uncommitted_crash_restores_the_exact_prior_enrollment():
    """A crash after enrollment but before the audit commit has to put BOTH the
    configs and the exact prior enrollment document back."""
    roles = ["default"]
    _write_profile_route("default", _batch_target("default", "anthropic"))
    before = _raw_config("default")
    enrollment = model_policy.enrollment_path()
    assert not enrollment.exists()

    class _PowerLoss(BaseException):
        pass

    def _after_write(_revisions):
        snapshot = model_policy.enroll_profiles(["default"])
        hermes_config.mark_route_batch_enrollment_written(snapshot)
        raise _PowerLoss()

    with pytest.raises(_PowerLoss):
        hermes_config.apply_profile_route_batch(
            [(role, _batch_target(role, "openai-codex")) for role in roles],
            apply_one=lambda profile, payload: (
                _write_profile_route(profile, payload) or f"rev-{profile}"
            ),
            after_write=_after_write,
        )

    assert _journal_path().exists()
    assert enrollment.exists()

    assert hermes_config.reconcile_profile_route_batch_journal() == ["default"]
    assert _raw_config("default") == before
    # The registry is back to "no record at all", exactly as it was.
    assert not enrollment.exists()
    assert not _journal_path().exists()


def test_a_second_process_cannot_reconcile_a_live_batch():
    """Reconciliation used to run before any interprocess lock, so a second
    process could restore and delete the journal of a batch still writing."""
    reached = threading.Event()
    release = threading.Event()
    observed: list = []

    def _apply(profile, payload):
        _write_profile_route(profile, payload)
        reached.set()
        assert release.wait(timeout=20), "the sibling never ran"
        return f"rev-{profile}"

    def _sibling():
        # Blocks on the machine-wide journal lock until the batch retires its
        # journal, so it can only ever observe a reconciled machine.
        observed.append(hermes_config.reconcile_profile_route_batch_journal())

    _write_profile_route("default", _batch_target("default", "anthropic"))
    sibling = threading.Thread(target=_sibling, daemon=True)
    batch = threading.Thread(
        target=lambda: hermes_config.apply_profile_route_batch(
            [("default", _batch_target("default", "openai-codex"))],
            apply_one=_apply,
        ),
        daemon=True,
    )
    batch.start()
    assert reached.wait(timeout=20), "the batch never wrote anything"
    sibling.start()
    time.sleep(0.2)
    # Still blocked: the journal is untouched and the batch's config stands.
    assert observed == []
    release.set()
    batch.join(timeout=20)
    sibling.join(timeout=20)
    assert observed == [[]]
    assert not _journal_path().exists()
    assert (
        _batch_target("default", "openai-codex")["model"]["default"]
        in _raw_config("default")
    )


def _write_committed_journal(roles: list) -> None:
    """Hand-write the journal a committed batch leaves before retirement."""
    record = {
        "version": 2,
        "created_at": 1.0,
        "phase": "audit-committed",
        "roles": [
            {
                "profile": role,
                "path": str(get_profile_dir(role) / "config.yaml"),
                "text": _raw_config(role),
                "mode": 0o600,
            }
            for role in roles
        ],
    }
    path = _journal_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")


def _batch_target(profile: str, provider: str) -> dict:
    assignment = model_policy.assignment_for(profile, provider)
    return {
        "model": {"provider": assignment.provider, "default": assignment.model},
        "agent": {"reasoning_effort": assignment.reasoning_effort},
        "fallback_providers": [],
    }


def _raw_config(profile: str) -> str:
    path = get_profile_dir(profile) / "config.yaml"
    return path.read_text(encoding="utf-8") if path.is_file() else ""


# ---------------------------------------------------------------------------
# A genuinely pre-upgrade board: receipt-owned work that predates the columns
# ---------------------------------------------------------------------------
#
# The previous build had no ``execution_tier``, ``model_policy_lock`` or
# ``owner_receipt_bound`` columns AND no ``task_kind='control'`` rows — a
# Project's anchor was an ordinary work row. Opening such a board through the
# current migrations must not leave its owner work dispatchable as ordinary
# work on whatever route the profile happens to hold.

_PRE_UPGRADE_ADDED_COLUMNS = (
    "execution_tier", "model_policy_lock", "owner_receipt_bound",
)
_LEGACY_BOARD = "legacy-upgrade"
_LEGACY_PROJECT = "proj_legacy_owner"
_LEGACY_ROLE = "raphael-builder"


def _insert_pre_upgrade_task(
    raw: sqlite3.Connection,
    task_id: str,
    *,
    status: str,
    project_id=None,
    assignee=None,
) -> str:
    raw.execute(
        "INSERT INTO tasks (id, title, status, assignee, project_id, "
        "task_kind, created_at, created_by) "
        "VALUES (?, ?, ?, ?, ?, 'work', ?, 'legacy')",
        (task_id, task_id, status, assignee, project_id, time.time()),
    )
    return task_id


@pytest.fixture
def pre_upgrade_board():
    """A board file in the exact shape the previous build left behind.

    The three columns this build added are really dropped, there is no control
    row anywhere, and the board carries the ``project_id`` the owner kernel
    published when it created it — which is the only owner-exclusive evidence
    that survives from before ``task_kind='control'`` existed.
    """
    path = kanban_db.kanban_db_path(board=_LEGACY_BOARD)
    with contextlib.closing(kanban_db.connect(board=_LEGACY_BOARD)):
        pass
    kanban_db.write_board_metadata(
        _LEGACY_BOARD, project_id=_LEGACY_PROJECT, dispatch_enabled=True,
    )
    raw = sqlite3.connect(str(path))
    try:
        for column in _PRE_UPGRADE_ADDED_COLUMNS:
            raw.execute(f"ALTER TABLE tasks DROP COLUMN {column}")
        rows = {
            # Owner work the previous build created: no tier, no route, no lock.
            "owner_unassigned_ready": _insert_pre_upgrade_task(
                raw, "owner_unassigned_ready",
                status="ready", project_id=_LEGACY_PROJECT,
            ),
            "owner_todo": _insert_pre_upgrade_task(
                raw, "owner_todo", status="todo",
                project_id=_LEGACY_PROJECT, assignee=_LEGACY_ROLE,
            ),
            # A hand-made Kanban card: never owner work, must keep working.
            "manual_card": _insert_pre_upgrade_task(
                raw, "manual_card", status="ready", assignee=_LEGACY_ROLE,
            ),
            # `hermes kanban create --project other` — a project_id that is NOT
            # this board's published owner.
            "other_project_card": _insert_pre_upgrade_task(
                raw, "other_project_card", status="ready",
                project_id="proj_someone_else", assignee=_LEGACY_ROLE,
            ),
        }
        raw.commit()
        present = {row[1] for row in raw.execute("PRAGMA table_info(tasks)")}
        assert not present & set(_PRE_UPGRADE_ADDED_COLUMNS)
    finally:
        raw.close()
    # Upgrade: open through the CURRENT migrations, changing no route at all.
    kanban_db.init_db(board=_LEGACY_BOARD)
    return rows


def _legacy_row(conn, task_id: str):
    return conn.execute(
        "SELECT status, assignee, block_kind, owner_receipt_bound, "
        "execution_tier, model_policy_lock FROM tasks WHERE id = ?",
        (task_id,),
    ).fetchone()


def test_upgrading_a_pre_upgrade_board_binds_and_parks_legacy_owner_work(
    pre_upgrade_board,
):
    with contextlib.closing(kanban_db.connect(board=_LEGACY_BOARD)) as conn:
        owner = _legacy_row(conn, pre_upgrade_board["owner_unassigned_ready"])
        # Bound by the board's own published ownership, with nothing to prove
        # its route: parked for re-approval rather than left in the work pool.
        assert owner["owner_receipt_bound"] == 1
        assert (owner["execution_tier"], owner["model_policy_lock"]) == (None, None)
        assert (owner["status"], owner["block_kind"]) == ("blocked", "needs_input")

        todo = _legacy_row(conn, pre_upgrade_board["owner_todo"])
        assert todo["owner_receipt_bound"] == 1
        # Not in an executable column, so nothing to park — but still governed.
        assert todo["status"] == "todo"

        for key in ("manual_card", "other_project_card"):
            ordinary = _legacy_row(conn, pre_upgrade_board[key])
            assert ordinary["owner_receipt_bound"] == 0
            assert ordinary["status"] == "ready"


def test_legacy_owner_work_can_no_longer_be_promoted_or_claimed(
    pre_upgrade_board,
):
    owner_todo = pre_upgrade_board["owner_todo"]
    owner_ready = pre_upgrade_board["owner_unassigned_ready"]
    with contextlib.closing(kanban_db.connect(board=_LEGACY_BOARD)) as conn:
        moved, reason = kanban_db.promote_task(conn, owner_todo, actor="tester")
        assert moved is False and "approved model route" in reason
        assert _legacy_row(conn, owner_todo)["status"] == "blocked"

        # Automatic promotion refuses it too, so nothing puts it back.
        kanban_db.recompute_ready(conn)
        assert _legacy_row(conn, owner_todo)["status"] == "blocked"

        # Backstop: even forced straight back into the dispatcher's own queue
        # column, no claim path will start a run on it.
        conn.execute(
            "UPDATE tasks SET status = 'ready', block_kind = NULL WHERE id = ?",
            (owner_ready,),
        )
        conn.commit()
        with pytest.raises(RuntimeError, match="approved again"):
            kanban_db.claim_task(conn, owner_ready)
        with pytest.raises(RuntimeError, match="approved again"):
            kanban_db.claim_review_task(conn, owner_ready)

        # ...and an ordinary manual card is completely unaffected.
        assert kanban_db.claim_task(conn, pre_upgrade_board["manual_card"])


def test_the_dispatcher_never_default_assigns_or_spawns_legacy_owner_work(
    pre_upgrade_board, all_assignees_spawnable,
):
    spawned: list = []

    def fake_spawn(task, workspace, board=None):
        spawned.append(task.id)
        return 4321

    with contextlib.closing(kanban_db.connect(board=_LEGACY_BOARD)) as conn:
        kanban_db.dispatch_once(
            conn,
            spawn_fn=fake_spawn,
            board=_LEGACY_BOARD,
            default_assignee=_LEGACY_ROLE,
        )
        owner = _legacy_row(conn, pre_upgrade_board["owner_unassigned_ready"])

    # Only the two genuinely non-owner cards ran.
    assert sorted(spawned) == [
        pre_upgrade_board["manual_card"],
        pre_upgrade_board["other_project_card"],
    ]
    # The parked owner row was never adopted onto the default role either.
    assert owner["assignee"] is None
    assert owner["status"] == "blocked"


def test_fully_admitted_owner_work_still_runs_on_the_same_upgraded_board(
    pre_upgrade_board, all_assignees_spawnable,
):
    """The upgrade parks what it cannot prove — not everything owner-owned."""
    spawned: list = []

    def fake_spawn(task, workspace, board=None):
        spawned.append((task.id, task.model_override, task.reasoning_effort))
        return 4322

    admitted = model_policy.task_assignment_for(_LEGACY_ROLE, "anthropic", "deep")
    with contextlib.closing(kanban_db.connect(board=_LEGACY_BOARD)) as conn:
        task_id = kanban_db.create_task(
            conn,
            title="approved on this build",
            assignee=_LEGACY_ROLE,
            project_id=_LEGACY_PROJECT,
            board=_LEGACY_BOARD,
            model_override=admitted.model,
            provider_override=admitted.provider,
            reasoning_effort=admitted.reasoning_effort,
            execution_tier="deep",
            model_policy_lock=kanban_db.mint_policy_lock(
                _LEGACY_ROLE, admitted.provider, admitted.model,
                admitted.reasoning_effort, "deep",
            ),
            receipt_owned=True,
        )
        kanban_db.dispatch_once(
            conn, spawn_fn=fake_spawn, board=_LEGACY_BOARD,
        )

    assert (task_id, admitted.model, admitted.reasoning_effort) in spawned


# ---------------------------------------------------------------------------
# Item 32TK: one unpinnable row never starves the tick, and unreadable board
# ownership never dispatches
# ---------------------------------------------------------------------------


def _unpinnable_owner_task(conn, *, status: str = "ready") -> str:
    """A receipt-owned row carrying a lock this build cannot validate."""
    task_id = kanban_db.create_task(
        conn,
        title="owner work with an unprovable route",
        assignee=_LEGACY_ROLE,
        project_id=_LEGACY_PROJECT,
        board=_LEGACY_BOARD,
        receipt_owned=True,
    )
    with kanban_db.write_txn(conn):
        conn.execute(
            "UPDATE tasks SET status = ?, model_policy_lock = 'not-a-real-lock' "
            "WHERE id = ?",
            (status, task_id),
        )
    return task_id


def test_one_unpinnable_task_never_starves_the_rest_of_the_tick(
    pre_upgrade_board, all_assignees_spawnable,
):
    """The route check used to raise out of ``claim_task`` — outside every
    per-task boundary — so one bad row aborted the whole dispatcher tick."""
    spawned: list = []

    def fake_spawn(task, workspace, board=None):
        spawned.append(task.id)
        return 9001

    with contextlib.closing(kanban_db.connect(board=_LEGACY_BOARD)) as conn:
        unpinnable = _unpinnable_owner_task(conn)
        result = kanban_db.dispatch_once(
            conn, spawn_fn=fake_spawn, board=_LEGACY_BOARD,
        )
        parked = _legacy_row(conn, unpinnable)

    # Unrelated ready work still ran, and the bad row is reported once.
    assert pre_upgrade_board["manual_card"] in spawned
    assert unpinnable not in spawned
    assert [task_id for task_id, _reason in result.skipped_route_unproven] == [
        unpinnable
    ]
    # And it is parked for re-approval rather than left in the queue.
    assert (parked["status"], parked["block_kind"]) == ("blocked", "needs_input")


def test_a_dry_run_never_reports_an_unpinnable_task_as_spawnable(
    pre_upgrade_board, all_assignees_spawnable,
):
    with contextlib.closing(kanban_db.connect(board=_LEGACY_BOARD)) as conn:
        unpinnable = _unpinnable_owner_task(conn)
        result = kanban_db.dispatch_once(
            conn, spawn_fn=lambda *a, **k: 1, board=_LEGACY_BOARD, dry_run=True,
        )
        # A dry run mutates nothing, so the row keeps its column.
        assert _legacy_row(conn, unpinnable)["status"] == "ready"

    assert unpinnable not in [task_id for task_id, _a, _w in result.spawned]
    assert [task_id for task_id, _reason in result.skipped_route_unproven] == [
        unpinnable
    ]


def test_a_dry_run_promotion_refuses_what_the_real_one_refuses(
    pre_upgrade_board,
):
    """``promote_task(dry_run=True)`` used to answer before the authority
    check, so it reported an unpinnable owner task as promotable."""
    with contextlib.closing(kanban_db.connect(board=_LEGACY_BOARD)) as conn:
        unpinnable = _unpinnable_owner_task(conn, status="todo")

        dry_ok, dry_reason = kanban_db.promote_task(
            conn, unpinnable, actor="tester", dry_run=True,
        )
        # Still non-mutating: the dry run neither parked nor moved the row.
        assert _legacy_row(conn, unpinnable)["status"] == "todo"

        real_ok, real_reason = kanban_db.promote_task(
            conn, unpinnable, actor="tester",
        )

    assert (dry_ok, real_ok) == (False, False)
    assert "approved model route" in dry_reason
    assert dry_reason == real_reason


def test_a_dry_run_promotion_still_succeeds_for_an_ordinary_task(
    pre_upgrade_board,
):
    with contextlib.closing(kanban_db.connect(board=_LEGACY_BOARD)) as conn:
        ordinary = kanban_db.create_task(
            conn, title="ordinary work", assignee=_LEGACY_ROLE,
            board=_LEGACY_BOARD,
        )
        with kanban_db.write_txn(conn):
            conn.execute(
                "UPDATE tasks SET status = 'todo' WHERE id = ?", (ordinary,)
            )
        assert kanban_db.promote_task(
            conn, ordinary, actor="tester", dry_run=True,
        ) == (True, None)
        assert _legacy_row(conn, ordinary)["status"] == "todo"


def _corrupt_board_metadata() -> None:
    kanban_db.board_metadata_path(_LEGACY_BOARD).write_text(
        "{not json at all", encoding="utf-8",
    )


def test_unreadable_board_ownership_never_dispatches(
    pre_upgrade_board, all_assignees_spawnable,
):
    """Metadata that cannot be read cannot be shown not to name an owner."""
    _corrupt_board_metadata()
    metadata = kanban_db.read_board_metadata(_LEGACY_BOARD)
    assert metadata["ownership_verified"] is False
    assert kanban_db.board_dispatch_allowed(metadata) is False

    spawned: list = []
    with contextlib.closing(kanban_db.connect(board=_LEGACY_BOARD)) as conn:
        result = kanban_db.dispatch_once(
            conn,
            spawn_fn=lambda task, workspace, board=None: spawned.append(task.id),
            board=_LEGACY_BOARD,
        )

    assert spawned == []
    assert result.skipped_inactive is True


def test_unreadable_board_ownership_is_never_overwritten(pre_upgrade_board):
    """A rewrite would replace the published owner with a synthesised none."""
    _corrupt_board_metadata()
    with pytest.raises(ValueError, match="unreadable metadata"):
        kanban_db.write_board_metadata(_LEGACY_BOARD, dispatch_enabled=False)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("project_id", ""),
        ("project_id", 12),
        ("project_id", {"id": "p1"}),
        ("dispatch_enabled", "yes"),
        ("dispatch_enabled", 1),
        ("dispatch_paused_by_owner", "no"),
        ("archived", "false"),
    ],
)
def test_a_malformed_authority_field_never_dispatches(
    pre_upgrade_board, all_assignees_spawnable, field, value,
):
    """A ``board.json`` that PARSES can still publish an authority value this
    build cannot act on. Marking it verified let dispatch activate on metadata
    whose owner nobody could resolve."""
    path = kanban_db.board_metadata_path(_LEGACY_BOARD)
    meta = json.loads(path.read_text(encoding="utf-8"))
    meta.setdefault("dispatch_enabled", True)
    meta.setdefault("dispatch_paused_by_owner", False)
    meta.setdefault("archived", False)
    meta[field] = value
    path.write_text(json.dumps(meta), encoding="utf-8")

    metadata = kanban_db.read_board_metadata(_LEGACY_BOARD)
    assert metadata["ownership_verified"] is False
    assert kanban_db.board_dispatch_allowed(metadata) is False

    spawned: list = []
    with contextlib.closing(kanban_db.connect(board=_LEGACY_BOARD)) as conn:
        result = kanban_db.dispatch_once(
            conn,
            spawn_fn=lambda task, workspace, board=None: spawned.append(task.id),
            board=_LEGACY_BOARD,
        )

    assert spawned == []
    assert result.skipped_inactive is True
    # And it is never overwritten with a synthesised "no owner" either.
    with pytest.raises(ValueError, match="unreadable metadata"):
        kanban_db.write_board_metadata(_LEGACY_BOARD, dispatch_enabled=False)


def test_a_well_formed_authority_field_is_still_verified(pre_upgrade_board):
    """The adjacent success path: an absent optional owner is not malformed."""
    path = kanban_db.board_metadata_path(_LEGACY_BOARD)
    meta = json.loads(path.read_text(encoding="utf-8"))
    meta.update({
        "project_id": None,
        "dispatch_enabled": False,
        "dispatch_paused_by_owner": False,
        "archived": False,
    })
    path.write_text(json.dumps(meta), encoding="utf-8")
    assert kanban_db.read_board_metadata(_LEGACY_BOARD)["ownership_verified"] is True


def test_a_board_with_no_metadata_file_still_dispatches(all_assignees_spawnable):
    """Absent is not unreadable: a board that publishes no owner is verified."""
    board = "no-metadata-board"
    with contextlib.closing(kanban_db.connect(board=board)) as conn:
        task_id = kanban_db.create_task(
            conn, title="plain card", assignee=_LEGACY_ROLE, board=board,
        )
        with kanban_db.write_txn(conn):
            conn.execute(
                "UPDATE tasks SET status = 'ready' WHERE id = ?", (task_id,)
            )
        kanban_db.board_metadata_path(board).unlink(missing_ok=True)
        assert kanban_db.read_board_metadata(board)["ownership_verified"] is True

        spawned: list = []
        kanban_db.dispatch_once(
            conn,
            spawn_fn=lambda task, workspace, board=None: spawned.append(task.id),
            board=board,
        )

    assert spawned == [task_id]


def test_every_migration_pass_reconciles_bound_executable_work(
    pre_upgrade_board,
):
    """The reconciliation scan used to be skipped whenever a pass bound zero
    rows, so a crash between the binding commit and the scan stranded work."""
    stranded = "stranded_bound_row"
    path = kanban_db.kanban_db_path(board=_LEGACY_BOARD)
    raw = sqlite3.connect(str(path))
    try:
        # Exactly the state that gap leaves: bound, unlocked, and already in an
        # executable column, with nothing left for a later pass to bind.
        raw.execute(
            "INSERT INTO tasks (id, title, status, assignee, project_id, "
            "task_kind, owner_receipt_bound, created_at, created_by) "
            "VALUES (?, ?, 'ready', ?, ?, 'work', 1, ?, 'legacy')",
            (stranded, stranded, _LEGACY_ROLE, _LEGACY_PROJECT, time.time()),
        )
        raw.commit()
    finally:
        raw.close()

    kanban_db.init_db(board=_LEGACY_BOARD)

    with contextlib.closing(kanban_db.connect(board=_LEGACY_BOARD)) as conn:
        row = _legacy_row(conn, stranded)
    assert (row["status"], row["block_kind"]) == ("blocked", "needs_input")
