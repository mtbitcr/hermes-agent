"""Contract tests for the owner-workspace mutation kernel (hermes_cli.owner_workspace)."""

from __future__ import annotations

import json
import multiprocessing
import os
import threading
import time
from pathlib import Path

import pytest

from hermes_cli import kanban_db, owner_workspace as ow, projects_db
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


@pytest.fixture
def ctx():
    return ow.OwnerContext(actor="default", profile="default", session="run_owt")


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
        task = kanban_db.get_task(kconn, result["task_id"])
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


def _bootstrap_board(ctx):
    t = _with_approver(ctx.session)
    result = ow.bootstrap(ctx, idempotency_key=f"setup-{ctx.session}-{time.monotonic()}", name="Board Setup")
    t.join()
    return result


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
        to_status="blocked", expected_status="WRONG", expected_revision=rev, board=board,
    )
    t.join()
    assert result["ok"] is False
    assert result["error"] == "conflict"
    assert result["current_status"] == "ready"

    kconn = kanban_db.connect(board=board)
    try:
        task = kanban_db.get_task(kconn, task_id)
    finally:
        kconn.close()
    assert task.status == "ready"


def test_move_task_rejects_running_target(ctx):
    setup = _bootstrap_board(ctx)
    with pytest.raises(ow.OwnerWorkspaceError) as excinfo:
        ow.move_task(
            ctx, idempotency_key="move-running", task_id=setup["task_id"],
            to_status="running", expected_status="ready", expected_revision=1, board=setup["board"],
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
        to_status="archived", expected_status="ready", expected_revision=rev, board=board,
    )
    t.join()
    assert result["ok"] is True
    assert result["status"] == "archived"

    kconn = kanban_db.connect(board=board)
    try:
        assert kanban_db.get_task(kconn, child_id).status == "ready"
    finally:
        kconn.close()


def test_comment_author_is_trusted_context_not_caller_supplied(ctx):
    setup = _bootstrap_board(ctx)
    board = setup["board"]
    task_id = setup["task_id"]

    t = _with_approver(ctx.session)
    result = ow.comment_task(ctx, idempotency_key="comment-1", task_id=task_id, body="hello", board=board)
    t.join()
    assert result["ok"] is True

    kconn = kanban_db.connect(board=board)
    try:
        comments = kanban_db.list_comments(kconn, task_id)
    finally:
        kconn.close()
    assert len(comments) == 1
    assert comments[0].author == ctx.actor
    assert comments[0].body == "hello"


def test_comment_exact_replay_creates_no_duplicate(ctx):
    setup = _bootstrap_board(ctx)
    board = setup["board"]
    task_id = setup["task_id"]

    t = _with_approver(ctx.session)
    first = ow.comment_task(ctx, idempotency_key="comment-replay", task_id=task_id, body="once", board=board)
    t.join()

    approval.unregister_gateway_notify(ctx.session)
    second = ow.comment_task(ctx, idempotency_key="comment-replay", task_id=task_id, body="once", board=board)
    assert second == first

    kconn = kanban_db.connect(board=board)
    try:
        comments = kanban_db.list_comments(kconn, task_id)
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
                expected_status="ready", expected_revision=rev, board=board,
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
                "task_id": task_id, "board": board, "to_status": "blocked",
                "expected_status": "ready", "expected_revision": rev,
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
        task = kanban_db.get_task(kconn, task_id)
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
                ctx, idempotency_key=key, task_id=task_id, body="hi", board=board,
            )
        t.join()

    old_thread = threading.Thread(target=run_old_claimant)
    old_thread.start()
    assert validated.wait(timeout=5.0), "old claimant never reached the paused mutation"

    takeover_state = {}

    def attempt_takeover():
        with projects_db.connect_closing() as pconn2:
            ow._ensure_schema(pconn2)
            digest = ow._digest({"task_id": task_id, "board": board, "body": "hi"})
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
        comments = kanban_db.list_comments(kconn, task_id)
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

    # A task with the SAME deterministic idempotency key already exists but
    # was never linked to a project (missing ownership metadata) — pass
    # project_id="" explicitly so it does NOT inherit the board's own
    # project_id (create_task's board-inheritance only fires when the
    # caller omits project_id entirely).
    kconn = kanban_db.connect(board=board_slug)
    try:
        precreated_task_id = kanban_db.create_task(
            kconn, title=name, board=board_slug, idempotency_key=task_idempotency_key,
            project_id="",
        )
        assert kanban_db.get_task(kconn, precreated_task_id).project_id is None
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
            ow.comment_task(ctx, idempotency_key=key, task_id=task_id, body="hello", board=board)
        t.join()

    _expire_lock(ctx, key)

    t2 = _with_approver(ctx.session)
    result = ow.comment_task(ctx, idempotency_key=key, task_id=task_id, body="hello", board=board)
    t2.join()
    assert result["ok"] is True

    kconn = kanban_db.connect(board=board)
    try:
        comments = kanban_db.list_comments(kconn, task_id)
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
        kanban_db.add_comment(kconn, task_id, author="default", body="first", operation_key="opk-1")
        with pytest.raises(ValueError):
            kanban_db.add_comment(kconn, task_id, author="default", body="different", operation_key="opk-1")
        comments = kanban_db.list_comments(kconn, task_id)
    finally:
        kconn.close()
    assert len(comments) == 1
    assert comments[0].body == "first"


# ---------------------------------------------------------------------------
# Task-move readiness repair after crash
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("to_status", ["done", "archived"])
def test_move_task_crash_before_recompute_ready_replay_repairs_child_readiness(
    ctx, to_status,
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
                expected_status="ready", expected_revision=rev, board=board,
            )
        t.join()

    _expire_lock(ctx, key)

    # The premise of the bug: the CAS status + event commit already
    # succeeded before the crash — only recompute_ready failed.
    kconn = kanban_db.connect(board=board)
    try:
        assert kanban_db.get_task(kconn, parent_id).status == to_status
    finally:
        kconn.close()

    t2 = _with_approver(ctx.session)
    result = ow.move_task(
        ctx, idempotency_key=key, task_id=parent_id, to_status=to_status,
        expected_status="ready", expected_revision=rev, board=board,
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
                expected_status="ready", expected_revision=rev, board=board,
            )
        t.join()

    _expire_lock(ctx, key)

    # An unrelated actor moves the task in the meantime (not via this
    # receipt) — a genuine conflict for the queued replay.
    kconn = kanban_db.connect(board=board)
    try:
        kanban_db.cas_transition_task(
            kconn, task_id, expected_status="ready", expected_revision=rev,
            to_status="review", event_kind="unrelated_move",
        )
    finally:
        kconn.close()

    t2 = _with_approver(ctx.session)
    result = ow.move_task(
        ctx, idempotency_key=key, task_id=task_id, to_status="blocked",
        expected_status="ready", expected_revision=rev, board=board,
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
                expected_status="ready", expected_revision=rev, board=board,
            )
        t.join()

    # The CAS really did commit before the simulated crash.
    kconn = kanban_db.connect(board=board)
    try:
        committed_task = kanban_db.get_task(kconn, task_id)
        committed_revision = kanban_db.task_event_revision(kconn, task_id)
    finally:
        kconn.close()
    assert committed_task.status == "blocked"

    _expire_lock(ctx, key)
    t2 = _with_approver(ctx.session)
    result = ow.move_task(
        ctx, idempotency_key=key, task_id=task_id, to_status="blocked",
        expected_status="ready", expected_revision=rev, board=board,
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


def test_move_task_replay_recognition_requires_matching_actor_and_transition(ctx):
    """Two different actors may validly reuse the same idempotency_key text
    on the same task — receipts are scoped by (actor, profile,
    idempotency_key), but the task's own event log is shared board-wide.
    Adopting a dead claim must never mistake ANOTHER actor's already-
    committed owner_move event (which happens to carry a matching
    idempotency_key string) for this receipt's own — that would fabricate a
    success snapshot (and could spuriously trigger readiness repair) for a
    transition this receipt never performed. The real outcome must be an
    unrelated CAS conflict against the actual current state."""
    setup = _bootstrap_board(ctx)
    board, task_id = setup["board"], setup["task_id"]

    kconn = kanban_db.connect(board=board)
    try:
        rev0 = kanban_db.task_event_revision(kconn, task_id)
    finally:
        kconn.close()

    shared_key = "shared-cross-actor-key"
    other_ctx = ow.OwnerContext(actor="other-actor", profile="other-actor", session="run_owt_other")

    # "Other" actor starts a move with the shared key but crashes before its
    # own CAS runs — its receipt is claimed (in_progress) with no event yet.
    def boom(*a, **kw):
        raise RuntimeError("simulated crash before the other actor's own CAS runs")

    with _temporarily_patch(ow.kanban_db, "cas_transition_task", boom):
        t = _with_approver(other_ctx.session)
        with pytest.raises(RuntimeError):
            ow.move_task(
                other_ctx, idempotency_key=shared_key, task_id=task_id,
                to_status="archived", expected_status="ready",
                expected_revision=rev0, board=board,
            )
        t.join()

    # The original ("default") actor independently and legitimately moves
    # the SAME task using the SAME idempotency_key text — a completely
    # separate, unrelated receipt row (primary-keyed by actor+profile+key).
    t2 = _with_approver(ctx.session)
    a_result = ow.move_task(
        ctx, idempotency_key=shared_key, task_id=task_id,
        to_status="blocked", expected_status="ready",
        expected_revision=rev0, board=board,
    )
    t2.join()
    assert a_result["ok"] is True
    assert a_result["status"] == "blocked"

    # The other actor's lease now looks expired (as if it had crashed) —
    # it resumes/replays.
    _expire_lock(other_ctx, shared_key)
    t3 = _with_approver(other_ctx.session)
    b_result = ow.move_task(
        other_ctx, idempotency_key=shared_key, task_id=task_id,
        to_status="archived", expected_status="ready",
        expected_revision=rev0, board=board,
    )
    t3.join()

    # The other actor must NOT be told its (never-performed) move to
    # "archived" succeeded just because an unrelated event happens to carry
    # the same idempotency_key text — it must see the real current state as
    # a conflict, never a fabricated success.
    assert b_result["ok"] is False
    assert b_result["error"] == "conflict"
    assert b_result["current_status"] == "blocked"

    kconn = kanban_db.connect(board=board)
    try:
        task = kanban_db.get_task(kconn, task_id)
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
    assert setup["status"] == "ready"
    assert setup["revision"] == 1

    t2 = _with_approver(ctx.session)
    result = ow.move_task(
        ctx, idempotency_key="pubrev-bootstrap-1-move", task_id=setup["task_id"],
        to_status="blocked", expected_status=setup["status"],
        expected_revision=setup["revision"], board=setup["board"],
    )
    t2.join()
    assert result["ok"] is True
    assert result["status"] == "blocked"


def test_comment_status_and_revision_feed_directly_into_move(ctx):
    setup = _bootstrap_board(ctx)
    board, task_id = setup["board"], setup["task_id"]

    t = _with_approver(ctx.session)
    comment_result = ow.comment_task(
        ctx, idempotency_key="pubrev-comment-1", task_id=task_id, body="hi", board=board,
    )
    t.join()
    assert comment_result["ok"] is True
    assert comment_result["status"] == "ready"
    assert comment_result["revision"] == 2  # task "created" + "commented"

    t2 = _with_approver(ctx.session)
    move_result = ow.move_task(
        ctx, idempotency_key="pubrev-comment-1-move", task_id=task_id,
        to_status="blocked", expected_status=comment_result["status"],
        expected_revision=comment_result["revision"], board=board,
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
