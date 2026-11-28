"""The owner can try work that waits for the AI provider again now.

A ready card held after a rate-limited run is accepted by the owner retry,
which records ``owner_retry`` against that exact run and writes no profile's
credential pool; the next tick may spawn it, and a limit that still holds books
a new reset and a new hold without counting a failure. The card's own worker,
when it next starts, clears its own pool's limit only for ``anthropic``,
whichever endpoint it calls, before its first model request resolves
credentials, and only when its card's latest ended run ended ``rate_limited``
and the owner's retry is bound to that run; nothing else clears a pool. A reader that names
``provider_wait_v1`` is told plainly why stopped work waits; one that does not
gets today's payload, compared whole against the one captured at the start
commit. A card reads as waiting only while the dispatcher still holds it.

Every stop is booked through the real reap path (the dispatcher spawns the
card, its worker exits, the sweep books the run), every owner call runs under
exactly the environment the dispatcher builds for a claimed worker, and every
worker start under exactly the one it builds for the card's own worker. The
root, each profile home, each pool and each board live in the test's own
temporary folder; every card, profile, custom provider, key, token and time is
made up. A built-in provider is named by its real id: a name, not a credential.
"""

from __future__ import annotations

import contextlib
import errno
import json
import logging
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

import cli
from agent import credential_pool, models_dev
from hermes_cli import kanban_db as kb, kanban_provider_stops, owner_workspace as ow
from hermes_cli.auth import PROVIDER_REGISTRY
from hermes_cli.cli_agent_setup_mixin import CLIAgentSetupMixin
from hermes_cli.config import ensure_hermes_home
from hermes_cli.kanban_provider_stops import KANBAN_PROVIDER_REFUSED_EXIT_CODE

ASSIGNEE = "worker"
STEWARD = "steward"
OTHER = "other"
# Made up, so a custom provider.
PROVIDER = "placeholder-alpha"
# Two built-in providers, by their real ids.
BUILT_IN = "deepseek"
SECOND_PROVIDER = "zai"
# A named custom provider and the legacy pool key its worker draws from.
VENDOR = "placeholder-vendor"
VENDOR_POOL = "custom:placeholder-vendor"
VENDOR_URL = "https://vendor.placeholder.invalid/v1"
COOLDOWN = 300
T0 = 1_900_000_000
REASON = "The placeholder quota was raised, so please try again now."
PROVIDER_WAIT = "provider_wait_v1"
WAITING_LINE = "Waiting for the AI provider. Work resumes by itself after {time} UTC."
REFUSED_LINE = "The AI provider declined this work as worded. It waits for you."
_STATUS_KEYS = (
    "last_status", "last_status_at", "last_error_code",
    "last_error_reason", "last_error_message", "last_error_reset_at",
)


@pytest.fixture
def clock(monkeypatch):
    now = {"t": T0}
    monkeypatch.setattr(kb.time, "time", lambda: now["t"])
    return now


@pytest.fixture
def root(tmp_path, monkeypatch, clock):
    """The shared root with three profile homes, the kanban root initialised."""
    home = tmp_path / ".hermes"
    for profile in (ASSIGNEE, STEWARD, OTHER):
        (home / "profiles" / profile).mkdir(parents=True)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    for key in [key for key in os.environ if key.startswith("HERMES_KANBAN_")]:
        monkeypatch.delenv(key)
    monkeypatch.setenv("HERMES_KANBAN_RATE_LIMIT_COOLDOWN_SECONDS", str(COOLDOWN))
    monkeypatch.setenv("HERMES_KANBAN_CRASH_GRACE_SECONDS", "0")
    monkeypatch.setattr(kb, "_pid_alive", lambda _pid: False)
    kb._INITIALIZED_PATHS.clear()
    kb.init_db()
    return home


@pytest.fixture
def owner(root, monkeypatch) -> ow.OwnerContext:
    """The owner runs as the steward profile, whose home keeps its projects."""
    monkeypatch.setenv("HERMES_HOME", str(root / "profiles" / STEWARD))
    return ow.resolve_owner_context()


def _project(owner: ow.OwnerContext, name: str) -> dict:
    """Commit one owner Project whose board may dispatch."""
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(ow, "_confirm", lambda *_args, **_kwargs: {"approved": True})
        project = ow.bootstrap(owner, idempotency_key=f"setup-{name}", name=name)
    assert project["ok"] is True
    kb.write_board_dispatch_state(project["board"], dispatch_enabled=True)
    project["slug"] = next(
        item["slug"] for item in ow.list_committed_projects(owner)
        if str(item["project_id"]) == project["project_id"]
    )
    return project


def _board(board: str):
    return contextlib.closing(kb.connect(board=board))


def _card(conn, project: dict, title: str, *, provider: str = PROVIDER) -> str:
    return kb.create_task(
        conn, title=title, assignee=ASSIGNEE, project_id=project["project_id"],
        provider_override=provider, model_override="placeholder-model",
    )


def _tick(conn, board: str, pids: dict):
    """One dispatcher tick; a spawned card's worker gets the pid ``pids`` names."""
    spawned = []

    def spawn(task, _workspace, **_kwargs):
        spawned.append(task.id)
        return pids[task.id]

    result = kb.dispatch_once(conn, spawn_fn=spawn, board=board)
    return result, spawned


def _reaped(conn, board: str, task_id: str, pid: int, code: int):
    """The dispatcher spawns the card, its worker exits ``code`` and the sweep books the run."""
    _result, spawned = _tick(conn, board, {task_id: pid})
    assert spawned == [task_id]
    run_id = kb.get_task(conn, task_id).current_run_id
    kb._record_worker_exit(pid, code << 8)
    kb.detect_crashed_workers(conn)
    return kb.get_run(conn, run_id)


def _held(conn, board: str, task_id: str, pid: int):
    run = _reaped(conn, board, task_id, pid, kb.KANBAN_RATE_LIMIT_EXIT_CODE)
    assert run.outcome == "rate_limited"
    assert kb.get_task(conn, task_id).status == "ready"
    return run


def _resume_at(run) -> int:
    return kb.recorded_rate_limit_reset(run.metadata, anchor=run.ended_at)


def _steward_env(root: Path, board: str) -> dict:
    """Exactly the environment the dispatcher builds for a claimed steward worker on ``board``."""
    with _board(board) as conn:
        task_id = kb.create_task(conn, title="Steward the placeholder work", assignee=STEWARD)
        assert kb.claim_task(conn, task_id, ttl_seconds=86_400) is not None
        task = kb.get_task(conn, task_id)
    launched = []

    def fake_popen(_cmd, *_args, **kwargs):
        launched.append(dict(kwargs["env"]))
        return SimpleNamespace(pid=0)

    with pytest.MonkeyPatch.context() as dispatcher:
        dispatcher.setattr(kb, "_resolve_hermes_argv", lambda: ["hermes"])
        dispatcher.setattr(subprocess, "Popen", fake_popen)
        kb._default_spawn(task, str(kb.resolve_workspace(task, board=board)), board=board)
    env = launched[0]
    assert env["HERMES_HOME"] == str(root / "profiles" / STEWARD)
    assert env["HERMES_KANBAN_TASK"] == task_id
    assert env["HERMES_KANBAN_DB"] == str(kb.kanban_db_path(board=board))
    assert env["HERMES_KANBAN_BOARD"] == board
    return env


@contextlib.contextmanager
def _as_worker(monkeypatch, env: dict):
    """Run the block with exactly ``env`` as the process environment."""
    with monkeypatch.context() as worker:
        for key in [key for key in os.environ if key not in env]:
            worker.delenv(key)
        for key, value in env.items():
            worker.setenv(key, value)
        yield


def _retry(
    owner: ow.OwnerContext, project: dict, task_id: str, key: str, *, route_token: str | None = None,
) -> dict:
    """The owner's retry, bound to an owner run the way the API server mints it."""
    payload = ow.canonical_owner_retry_payload(
        idempotency_key=key, project_id=project["project_id"],
        task_id=task_id, reason=REASON, route_token=route_token,
    )
    ctx = ow.OwnerContext(
        actor=owner.actor, profile=owner.profile, session=owner.session,
        authority=ow.OwnerProposalAuthority(
            actor=owner.actor, profile=owner.profile, session=owner.session,
            conversation="raphael-owner-" + "c" * 32,
            response_id="resp_" + "d" * 32,
            operation="owner_task_retry",
            idempotency_key=key,
            payload_digest=ow._digest(payload),
        ),
    )
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(ow, "_confirm", lambda *_args, **_kwargs: {"approved": True})
        return ow.retry_task(
            ctx, idempotency_key=key, project_id=project["project_id"],
            task_id=task_id, reason=REASON, route_token=route_token,
        )


def _events(conn, task_id: str, kind: str) -> list:
    return [event for event in kb.list_events(conn, task_id) if event.kind == kind]


def _exhausted(entry_id: str) -> dict:
    return {
        "id": entry_id,
        "label": f"placeholder {entry_id}",
        "auth_type": "api_key",
        "priority": 0,
        "source": "manual",
        "access_token": f"placeholder-token-{entry_id}",
        "last_status": "exhausted",
        "last_status_at": T0 - 30,
        "last_error_code": 429,
        "last_error_reason": "placeholder_quota",
        "last_error_message": "placeholder quota reached",
        "last_error_reset_at": T0 + 3_600,
        "failure_reason": "placeholder_quota",
    }


def _write_pool(home: Path, pool: dict) -> Path:
    path = home / "auth.json"
    path.write_text(json.dumps({"version": 1, "credential_pool": pool}, indent=2) + "\n")
    return path


def test_a_retry_makes_a_held_ready_card_spawnable_on_the_next_tick(root, owner, clock, monkeypatch):
    project = _project(owner, "Retry Pilot")
    board = project["board"]
    with _board(board) as conn:
        task_id = _card(conn, project, "Draft the placeholder brief")
        run = _held(conn, board, task_id, 4201)
        clock["t"] = run.ended_at + 60
        assert clock["t"] < _resume_at(run)
        result, spawned = _tick(conn, board, {task_id: 4202})
        assert (task_id, "rate_limit_cooldown") in result.respawn_guarded
        assert spawned == []

    env = _steward_env(root, board)
    with _as_worker(monkeypatch, env):
        retried = _retry(owner, project, task_id, "retry-held-ready")
    assert retried["ok"] is True, retried

    with _board(board) as conn:
        [event] = _events(conn, task_id, "owner_retry")
        assert retried == {
            "ok": True, "task_id": task_id, "status": "ready",
            "revision": event.id, "retry_reason": REASON,
        }
        assert event.run_id == run.id
        assert event.payload["reason"] == REASON
        assert kb.get_task(conn, task_id).status == "ready"
        assert kb.check_respawn_guard(conn, task_id) is None
        _result, spawned = _tick(conn, board, {task_id: 4203})
        assert spawned == [task_id]


@pytest.mark.parametrize(
    ("role", "tier"),
    [("raphael-claude-worker", "deep"), ("raphael-builder", "routine")],
)
def test_alternate_retry_requires_a_fresh_choice_after_a_card_change(root, owner, clock, monkeypatch, role, tier):
    from hermes_cli import owner_retry_route
    from plugins.dashboard_auth.raphael_workspace import model_policy

    import sqlite3

    profile_home = Path(os.environ["HERMES_HOME"])
    with monkeypatch.context() as shared_root:
        shared_root.setenv("HERMES_HOME", str(root))
        project = _project(owner, "Alternate Route Retry")
        with contextlib.closing(sqlite3.connect(root / "projects.db")) as source:
            with contextlib.closing(sqlite3.connect(profile_home / "projects.db")) as target:
                source.backup(target)
    board = project["board"]
    (root / "profiles" / role).mkdir()
    assignment = model_policy.task_assignment_for(role, "anthropic", tier)
    lock = model_policy.mint_policy_lock(
        role, "anthropic", assignment.model, assignment.reasoning_effort, tier,
    )
    with _board(board) as conn:
        task_id = kb.create_task(
            conn, title="Draft the placeholder brief", assignee=role,
            project_id=project["project_id"], provider_override="anthropic",
            model_override=assignment.model, reasoning_effort=assignment.reasoning_effort,
            execution_tier=tier, model_policy_lock=lock, requires_review=True,
        )
        held = _held(conn, board, task_id, 4211)
    clock["t"] = held.ended_at + 60
    monkeypatch.setenv("HERMES_OPENAI_RETRY_TASK_ID", task_id)
    monkeypatch.setattr(owner_retry_route, "openai_ready", lambda _role: True)

    def read_choice():
        snapshot = ow.read_project_snapshot(owner, project["slug"], provider_wait=True)
        return next(
            task["retry_route"]
            for column in snapshot["columns"] for task in column["tasks"] if task["id"] == task_id
        )

    env = _steward_env(root, board)
    with _as_worker(monkeypatch, env):
        old_choice = read_choice()

    with _board(board) as conn:
        earlier_revision = kb.task_event_revision(conn, task_id)
        kb.add_comment(conn, task_id, "owner", "Update the brief instructions.")
        assert kb.task_event_revision(conn, task_id) > earlier_revision

    with _as_worker(monkeypatch, env):
        stale = _retry(owner, project, task_id, "retry-stale", route_token=old_choice["token"])
    assert stale["ok"] is False, stale

    with _board(board) as conn:
        assert kb.get_task(conn, task_id).provider_override == "anthropic"
        assert kb.check_respawn_guard(conn, task_id) == "rate_limit_cooldown"
        assert not _events(conn, task_id, "owner_retry")

    with _as_worker(monkeypatch, env):
        fresh = read_choice()
        assert fresh["token"] != old_choice["token"]
        accepted = _retry(owner, project, task_id, "retry-fresh", route_token=fresh["token"])
    assert accepted["ok"] is True, accepted

    with _board(board) as conn:
        [event] = _events(conn, task_id, "owner_retry")
        assert event.run_id == held.id
        assert kb.get_run(conn, held.id).outcome == "rate_limited"
        task = kb.get_task(conn, task_id)
        assert task.provider_override == "openai-codex"
        assert task.requires_review
        _result, spawned = _tick(conn, board, {task_id: 4212})
        assert spawned == [task_id]


def _alternate_scene(
    root: Path, owner: ow.OwnerContext, clock, monkeypatch, name: str, *lanes: tuple,
) -> SimpleNamespace:
    """One Project with a card held by the provider limit on its Claude route for each (role, tier) of ``lanes``.

    ``worker_env(setting)`` is the environment the dispatcher gives a worker and
    ``read(setting)`` the owner's page read with the capability under it: the
    server setting is left out for ``None`` and is that text otherwise.
    """
    import sqlite3

    from hermes_cli import owner_retry_route
    from plugins.dashboard_auth.raphael_workspace import model_policy

    profile_home = Path(os.environ["HERMES_HOME"])
    with monkeypatch.context() as shared_root:
        shared_root.setenv("HERMES_HOME", str(root))
        project = _project(owner, name)
        with contextlib.closing(sqlite3.connect(root / "projects.db")) as source:
            with contextlib.closing(sqlite3.connect(profile_home / "projects.db")) as target:
                source.backup(target)
    board = project["board"]
    task_ids = []
    with _board(board) as conn:
        for role, tier in lanes:
            (root / "profiles" / role).mkdir(exist_ok=True)
            assignment = model_policy.task_assignment_for(role, "anthropic", tier)
            lock = model_policy.mint_policy_lock(
                role, "anthropic", assignment.model, assignment.reasoning_effort, tier,
            )
            task_ids.append(kb.create_task(
                conn, title=f"Draft the placeholder brief {len(task_ids) + 1}", assignee=role,
                project_id=project["project_id"], provider_override="anthropic",
                model_override=assignment.model, reasoning_effort=assignment.reasoning_effort,
                execution_tier=tier, model_policy_lock=lock, requires_review=True,
            ))
            held = _held(conn, board, task_ids[-1], 4800 + len(task_ids))
    clock["t"] = held.ended_at + 60
    monkeypatch.delenv("HERMES_OPENAI_RETRY_TASK_ID", raising=False)
    monkeypatch.setattr(owner_retry_route, "openai_ready", lambda _role: True)
    env = _steward_env(root, board)
    assert "HERMES_OPENAI_RETRY_TASK_ID" not in env

    def worker_env(setting: str | None) -> dict:
        return env if setting is None else {**env, "HERMES_OPENAI_RETRY_TASK_ID": setting}

    def read(setting: str | None) -> dict:
        with _as_worker(monkeypatch, worker_env(setting)):
            return ow.read_project_snapshot(owner, project["slug"], provider_wait=True)

    return SimpleNamespace(
        project=project, board=board, task_ids=task_ids, env=env, worker_env=worker_env, read=read,
    )


@pytest.mark.parametrize("setting", [None, "", "tâche-中文"], ids=["unset", "empty", "multibyte"])
def test_alternate_retry_is_offered_to_no_card_while_the_setting_names_none(
    root, owner, clock, monkeypatch, setting,
):
    scene = _alternate_scene(
        root, owner, clock, monkeypatch, "Setting Names None",
        ("raphael-claude-worker", "deep"), ("raphael-builder", "routine"),
    )
    page = scene.read(setting)
    assert [
        task["id"] for column in page["columns"] for task in column["tasks"] if "retry_route" in task
    ] == []
    # Named, a card is offered its route and the rest of the page reads as it did.
    for task_id in scene.task_ids:
        named = scene.read(task_id)
        assert set(_task(named, task_id)["retry_route"]) == {"token", "model", "effort"}
        del _task(named, task_id)["retry_route"]
        assert named == page


@pytest.mark.parametrize(
    ("role", "tier"),
    [
        ("raphael-claude-worker", "routine"),
        ("raphael-builder", "deep"),
        ("raphael-planner", "routine"),
        ("raphael-planner", "deep"),
    ],
)
def test_alternate_retry_is_not_offered_for_a_named_card_outside_the_two_lanes(
    root, owner, clock, monkeypatch, role, tier,
):
    scene = _alternate_scene(root, owner, clock, monkeypatch, "Outside The Two Lanes", (role, tier))
    [task_id] = scene.task_ids
    with _board(scene.board) as conn:
        kb.assert_claimable_route(conn, task_id)
    card = _task(scene.read(task_id), task_id)
    assert card["stopped_work"] == "provider_wait"
    assert "retry_route" not in card


@pytest.mark.parametrize(
    ("role", "tier"),
    [("raphael-claude-worker", "deep"), ("raphael-builder", "routine")],
)
@pytest.mark.parametrize("setting", [None, "", "placeholder-other-card"], ids=["unset", "empty", "other_card"])
def test_alternate_retry_cannot_repin_a_card_the_setting_does_not_name(
    root, owner, clock, monkeypatch, role, tier, setting,
):
    scene = _alternate_scene(root, owner, clock, monkeypatch, "Setting Does Not Name It", (role, tier))
    [task_id], board = scene.task_ids, scene.board
    token = _task(scene.read(task_id), task_id)["retry_route"]["token"]
    with _board(board) as conn:
        before = kb.get_task(conn, task_id)
    assert before.provider_override == "anthropic"

    with _as_worker(monkeypatch, scene.worker_env(setting)):
        refused = _retry(owner, scene.project, task_id, "retry-unnamed", route_token=token)
    assert refused["ok"] is False
    assert refused["reason"] == "This retry choice is no longer available. Reload this Project and check the work."

    with _board(board) as conn:
        assert kb.get_task(conn, task_id) == before
        assert not _events(conn, task_id, "owner_retry")
        assert not _events(conn, task_id, "model_route_repinned")
        assert not conn.execute(
            "SELECT name FROM sqlite_master WHERE name = 'owner_retry_run_pins'"
        ).fetchall()
        assert kb.check_respawn_guard(conn, task_id) == "rate_limit_cooldown"
    # The setting naming the card is the one thing missing: the same choice is offered again.
    assert _task(scene.read(task_id), task_id)["retry_route"]["token"] == token
    with _board(board) as conn:
        result, spawned = _tick(conn, board, {task_id: 4821})
        assert (task_id, "rate_limit_cooldown") in result.respawn_guarded
        assert spawned == []


@pytest.mark.parametrize(
    ("lock", "refusal", "reported"),
    [
        pytest.param(None, "approved route lock", "approved only by the owner's retry", id="no_lock"),
        pytest.param("", "approved route lock", "approved only by the owner's retry", id="empty_lock"),
        pytest.param(
            ("raphael-claude-worker", "openai-codex", "gpt-6.2-sol", "max", "deep"),
            "does not bind this route", "does not bind this route", id="deep_worker_lock",
        ),
        pytest.param(
            ("raphael-builder", "anthropic", "claude-opus-5-5", "max", "deep"),
            "does not bind this route", "does not bind this route", id="deep_builder_lock",
        ),
    ],
)
def test_a_routine_builder_card_on_the_alternate_route_never_starts_without_its_exact_lock(
    root, owner, clock, monkeypatch, lock, refusal, reported,
):
    from plugins.dashboard_auth.raphael_workspace import model_policy

    scene = _alternate_scene(
        root, owner, clock, monkeypatch, "Exact Lock To Start",
        ("raphael-builder", "routine"), ("raphael-builder", "routine"),
    )
    control, refused = scene.task_ids
    for key, task_id in (("control", control), ("refused", refused)):
        switched = _switch(owner, scene.project, task_id, scene.env, monkeypatch, f"switch-{key}")
        assert switched["ok"] is True, switched
    route = model_policy.task_assignment_for("raphael-builder", "openai-codex", "routine")
    exact = model_policy.mint_policy_lock(
        "raphael-builder", "openai-codex", route.model, route.reasoning_effort, "routine",
        owner_retry=True,
    )
    with _board(scene.board) as conn:
        for task_id in (control, refused):
            task = kb.get_task(conn, task_id)
            assert (
                task.assignee, task.execution_tier, task.provider_override,
                task.model_override, task.reasoning_effort, task.model_policy_lock, task.status,
            ) == (
                "raphael-builder", "routine", "openai-codex",
                route.model, route.reasoning_effort, exact, "ready",
            )
        with kb.write_txn(conn):
            conn.execute(
                "UPDATE tasks SET model_policy_lock = ? WHERE id = ?",
                (model_policy.mint_policy_lock(*lock) if isinstance(lock, tuple) else lock, refused),
            )
        runs = "SELECT id, status FROM task_runs WHERE task_id = ? ORDER BY id"
        refused_runs = [tuple(row) for row in conn.execute(runs, (refused,))]
        with pytest.raises(RuntimeError, match=refusal):
            kb.claim_task(conn, refused)

        result, spawned = _tick(conn, scene.board, {control: 4831, refused: 4832})
        # The card with its exact lock starts; the other one is turned away and starts nothing.
        assert spawned == [control]
        [(skipped, reason)] = result.skipped_route_unproven
        assert skipped == refused
        assert reported in reason
        task = kb.get_task(conn, refused)
        assert task.status != "running"
        assert (task.current_run_id, task.claim_lock, task.worker_pid) == (None, None, None)
        assert [tuple(row) for row in conn.execute(runs, (refused,))] == refused_runs
        assert "running" not in [status for _run_id, status in refused_runs]
        assert kb.get_task(conn, control).status == "running"
        assert [status for _run_id, status in conn.execute(runs, (control,))][-1] == "running"


def test_the_kernel_never_mints_the_routine_builder_alternate_lock_from_a_stored_route(root):
    from plugins.dashboard_auth.raphael_workspace import model_policy

    builder = model_policy.task_assignment_for("raphael-builder", "openai-codex", "routine")
    worker = model_policy.task_assignment_for("raphael-claude-worker", "openai-codex", "deep")
    # The kernel's wrapper passes no retry flag, so it is an ordinary caller.
    with pytest.raises(ValueError, match="retry"):
        kb.mint_policy_lock(
            "raphael-builder", "openai-codex", builder.model, builder.reasoning_effort, "routine",
        )
    deep_lock = kb.mint_policy_lock(
        "raphael-claude-worker", "openai-codex", worker.model, worker.reasoning_effort, "deep",
    )
    assert kb.policy_lock_error(
        deep_lock, "raphael-claude-worker", "openai-codex", worker.model, worker.reasoning_effort, "deep",
    ) is None

    cards = {}
    with kb.connect_closing() as conn:
        for role, tier, route in (
            ("raphael-builder", "routine", builder), ("raphael-claude-worker", "deep", worker),
        ):
            (root / "profiles" / role).mkdir(exist_ok=True)
            cards[role] = kb.create_task(conn, title=f"Stored on the alternate route: {role}", assignee=role)
            with kb.write_txn(conn):
                conn.execute(
                    "UPDATE tasks SET execution_tier = ?, provider_override = ?, model_override = ?, "
                    "reasoning_effort = ?, model_policy_lock = NULL WHERE id = ?",
                    (tier, "openai-codex", route.model, route.reasoning_effort, cards[role]),
                )
        with kb.write_txn(conn):
            allowed = {role: kb.authorize_executable_transition(conn, task_id) for role, task_id in cards.items()}
        rows = {
            role: conn.execute(
                "SELECT status, block_kind, model_policy_lock FROM tasks WHERE id = ?", (task_id,),
            ).fetchone()
            for role, task_id in cards.items()
        }
        refusals = {role: kb.route_authority_error(conn, task_id) for role, task_id in cards.items()}
    # The builder card parks for re-approval with no lock written; the deep Worker card is proven as before.
    assert allowed == {"raphael-builder": False, "raphael-claude-worker": True}
    assert tuple(rows["raphael-builder"]) == ("blocked", "needs_input", None)
    assert "retry" in refusals["raphael-builder"]
    assert (rows["raphael-claude-worker"]["status"], rows["raphael-claude-worker"]["model_policy_lock"]) == (
        "ready", deep_lock,
    )
    assert refusals["raphael-claude-worker"] is None


def _git_in(repo: Path, *args: str) -> str:
    return subprocess.run(
        [
            "git", "-C", str(repo), "-c", "user.name=Placeholder", "-c", "user.email=placeholder@example.invalid",
            "-c", "commit.gpgsign=false", *args,
        ],
        check=True, capture_output=True, text=True,
    ).stdout.strip()


def _delivery_repository(tmp_path: Path) -> tuple[Path, str]:
    """A git repository of one commit that the registry names as the primary folder of the Project ``delivery-project``."""
    from hermes_cli import projects_db

    repo = tmp_path / "delivery-repository"
    repo.mkdir()
    _git_in(repo, "init", "-q")
    (repo / "app.py").write_text("placeholder = 1\n", encoding="utf-8")
    _git_in(repo, "add", "app.py")
    _git_in(repo, "commit", "-q", "-m", "placeholder")
    with projects_db.connect_closing() as registry:
        projects_db.create_project(
            registry, name="delivery-project", id="delivery-project", primary_path=str(repo),
        )
    return repo, _git_in(repo, "rev-parse", "HEAD")


def _finished_builder_card(conn, repo: Path, head: str, risk_tier: int, receipt: tuple | None) -> str:
    """A done routine Builder card of the Project, governed on its Claude route, whose accepted handover names the
    Builder and whose latest run carries the runtime receipt of ``receipt`` (provider, model, effort), or none."""
    from hermes_cli.kanban_risk_tier import pinned_reasoning_effort
    from plugins.dashboard_auth.raphael_workspace import model_policy

    claude = model_policy.task_assignment_for("raphael-builder", "anthropic", "routine")
    effort = pinned_reasoning_effort(risk_tier, None)
    source = kb.create_task(
        conn, title="Build the placeholder feature", assignee="raphael-builder", project_id="delivery-project",
        provider_override="anthropic", model_override=claude.model, reasoning_effort=effort,
        execution_tier="routine", risk_tier=risk_tier, owned_paths=["app.py"], requires_review=True,
        model_policy_lock=kb.mint_policy_lock("raphael-builder", "anthropic", claude.model, effort, "routine"),
    )
    metadata = {"worker_session_id": f"session-{source}"}
    if receipt is not None:
        metadata["runtime_receipt"] = {
            "schema_version": 3, "engine": "hermes", "profile": "raphael-builder",
            "route_evidence": "dominant-session-usage",
            **dict(zip(("provider", "model", "reasoning_effort"), receipt)),
        }
    with kb.write_txn(conn):
        kb._append_event(conn, source, "review_requested", {"implementer": "raphael-builder", "reviewer": "raphael-verifier"})
        conn.execute(
            "INSERT INTO task_runs (task_id, profile, status, outcome, started_at, ended_at, metadata) "
            "VALUES (?, 'raphael-builder', 'done', 'completed', ?, ?, ?)",
            (source, T0, T0 + 1, json.dumps(metadata)),
        )
        conn.execute("UPDATE tasks SET status = 'done' WHERE id = ?", (source,))
    _git_in(repo, "branch", f"delivery-return/{source}-{head[:12]}", head)
    return source


def _returned_for_a_failed_check(conn, repo: Path, source: str, head: str) -> dict:
    """The failed required check's return of ``source`` at ``head``, written by the delivery's one continuation writer."""
    from hermes_cli import kanban_delivery as kd

    row = {"source_task_id": source, "pull_request_head": head, "pull_request_number": 7}
    body = kd._red_body(row, "A required check failed.", ["tests/test_app.py::test_placeholder"])
    with kb.write_txn(conn):
        return kd._continuation(conn, kb.kanban_db_path(), row, body, (repo, f"delivery-return/{source}-{head[:12]}"))


@pytest.mark.parametrize("risk_tier", [1, 2])
def test_a_continuation_without_its_own_approval_never_starts_on_the_alternate_builder_route(
    root, tmp_path, monkeypatch, risk_tier,
):
    from plugins.dashboard_auth.raphael_workspace import model_policy

    monkeypatch.delenv("HERMES_OPENAI_RETRY_TASK_ID", raising=False)
    (root / "profiles" / "raphael-builder").mkdir(exist_ok=True)
    repo, head = _delivery_repository(tmp_path)
    claude = model_policy.task_assignment_for("raphael-builder", "anthropic", "routine")
    alternate = model_policy.task_assignment_for("raphael-builder", "openai-codex", "routine")
    pinned = {1: "high", 2: "max"}[risk_tier]
    receipts = {
        "claude": ("anthropic", claude.model, pinned),
        "no_receipt": None,
        "alternate": ("openai-codex", alternate.model, alternate.reasoning_effort),
    }
    with kb.connect_closing() as conn:
        # The alternate receipt first and alone: no other card is ready, so no concurrency limit can be why
        # the card does not start. The card was never named by the owner, so it parks for re-approval.
        source = _finished_builder_card(conn, repo, head, risk_tier, receipts["alternate"])
        returned = _returned_for_a_failed_check(conn, repo, source, head)
        item = returned["rework"]
        _result, spawned = _tick(conn, None, {item: 4901})
        task = kb.get_task(conn, item)
        seen = {
            "parked": "retry" in returned.get("parked", ""),
            "status": task.status,
            "block_kind": conn.execute("SELECT block_kind FROM tasks WHERE id = ?", (item,)).fetchone()[0],
            "lock": task.model_policy_lock,
            "claim": (task.current_run_id, task.claim_lock, task.worker_pid),
            "runs": conn.execute("SELECT COUNT(*) FROM task_runs WHERE task_id = ?", (item,)).fetchone()[0],
            "spawned": spawned,
        }
        assert seen == {
            "parked": True, "status": "blocked", "block_kind": "needs_input", "lock": None,
            "claim": (None, None, None), "runs": 0, "spawned": [],
        }
        assert kb.route_authority_error(conn, item)
        assert kb.policy_lock_error(
            task.model_policy_lock, task.assignee, task.provider_override, task.model_override,
            task.reasoning_effort, task.execution_tier,
        )

        # The Claude receipt, and a run with no receipt at all, continue as they did: ready on the
        # Claude route with a lock that verifies, and the dispatcher starts one of them.
        controls = {}
        for name in ("claude", "no_receipt"):
            source = _finished_builder_card(conn, repo, head, risk_tier, receipts[name])
            returned = _returned_for_a_failed_check(conn, repo, source, head)
            assert "parked" not in returned, name
            controls[name] = returned["rework"]
            task = kb.get_task(conn, controls[name])
            assert (task.status, task.provider_override, task.model_override, task.reasoning_effort) == (
                "ready", "anthropic", claude.model, pinned,
            ), name
            assert kb.policy_lock_error(
                task.model_policy_lock, task.assignee, task.provider_override, task.model_override,
                task.reasoning_effort, task.execution_tier,
            ) is None, name
            assert kb.route_authority_error(conn, controls[name]) is None, name
        _result, spawned = _tick(conn, None, {task_id: 4902 + number for number, task_id in enumerate(controls.values())})
        assert spawned and set(spawned) <= set(controls.values())
        assert kb.get_task(conn, item).status == "blocked"


@pytest.mark.parametrize("repinned_between", [False, True], ids=["two_stops", "repinned_between"])
def test_after_an_alternate_route_retry_each_earlier_stop_keeps_the_receipt_its_route_history_proves(
    repinned_between, root, owner, clock, monkeypatch,
):
    from hermes_cli import owner_retry_route
    from plugins.dashboard_auth.raphael_workspace import model_policy

    import sqlite3

    profile_home = Path(os.environ["HERMES_HOME"])
    with monkeypatch.context() as shared_root:
        shared_root.setenv("HERMES_HOME", str(root))
        project = _project(owner, "Deep Worker Receipts")
        with contextlib.closing(sqlite3.connect(root / "projects.db")) as source:
            with contextlib.closing(sqlite3.connect(profile_home / "projects.db")) as target:
                source.backup(target)
    board = project["board"]
    role = "raphael-claude-worker"
    (root / "profiles" / role).mkdir()

    def approved(profile: str) -> dict:
        assignment = model_policy.task_assignment_for(profile, "anthropic", "deep")
        return {
            "assignee": profile, "provider": "anthropic", "model": assignment.model,
            "reasoning_effort": assignment.reasoning_effort, "execution_tier": "deep",
            "model_policy_lock": model_policy.mint_policy_lock(
                profile, "anthropic", assignment.model, assignment.reasoning_effort, "deep",
            ),
        }

    route = approved(role)
    receipt = {
        "schema_version": 3, "engine": "hermes", "profile": role, "provider": "anthropic",
        "model": route["model"], "reasoning_effort": route["reasoning_effort"],
        "route_evidence": "session-row",
        "cost": {"state": "estimated", "currency": "USD", "scope": "dominant-main-route", "amount": 0.25},
    }
    monkeypatch.setattr(
        kb, "_trusted_runtime_receipt",
        lambda session_id, _profile: dict(receipt) if session_id == _SESSION else None,
    )

    def stop(conn, task_id: str, pid: int):
        """The dispatcher spawns the card; its worker links its session and stops at the limit."""
        _result, spawned = _tick(conn, board, {task_id: pid})
        assert spawned == [task_id]
        run_id = kb.get_task(conn, task_id).current_run_id
        assert kb.heartbeat_worker(conn, task_id, expected_run_id=run_id, session_id=_SESSION)
        kb._record_worker_exit(pid, kb.KANBAN_RATE_LIMIT_EXIT_CODE << 8)
        kb.detect_crashed_workers(conn)
        run = kb.get_run(conn, run_id)
        assert run.outcome == "rate_limited"
        assert kanban_provider_stops._recorded_for(run.metadata) == (True, "anthropic")
        assert kb.get_task(conn, task_id).status == "ready"
        return run

    with _board(board) as conn:
        task_id = kb.create_task(
            conn, title="Draft the placeholder brief", assignee=role,
            project_id=project["project_id"], provider_override="anthropic",
            model_override=route["model"], reasoning_effort=route["reasoning_effort"],
            execution_tier="deep", model_policy_lock=route["model_policy_lock"],
            requires_review=True,
        )
        first = stop(conn, task_id, 4221)
        if repinned_between:
            # The card moves to another role and back: the same route again, but
            # its history records two route changes after the first attempt.
            for profile in ("raphael-builder", role):
                assert kb.assign_task(conn, task_id, profile, approved_route=approved(profile))
            assert len(_events(conn, task_id, "model_route_repinned")) == 2
        # The cooldown ends and the dispatcher tries the original route again.
        clock["t"] = _resume_at(first) + 1
        second = stop(conn, task_id, 4222)
    clock["t"] = second.ended_at + 60
    monkeypatch.setenv("HERMES_OPENAI_RETRY_TASK_ID", task_id)
    monkeypatch.setattr(owner_retry_route, "openai_ready", lambda _role: True)
    env = _steward_env(root, board)

    def read():
        snapshot = ow.read_project_snapshot(owner, project["slug"], provider_wait=True)
        receipts = [
            (run["receipt"]["runtime"]["state"], run["receipt"]["cost"]["state"])
            for run in snapshot["runs"]
        ]
        return snapshot, receipts

    with _as_worker(monkeypatch, env):
        snapshot, before = read()
        choice = next(
            task["retry_route"]
            for column in snapshot["columns"] for task in column["tasks"] if task["id"] == task_id
        )
        accepted = _retry(owner, project, task_id, "retry-alternate", route_token=choice["token"])
        assert accepted["ok"] is True, accepted
        _snapshot, after = read()

    with _board(board) as conn:
        assert kb.get_task(conn, task_id).provider_override == "openai-codex"
        [event] = _events(conn, task_id, "owner_retry")
        assert event.run_id == second.id
    known = ("known", "estimated")
    # Newest first: the stop the owner retried, then the attempt before it,
    # which reads unknown after the switch (rule of 2026-11-27, point 4a).
    assert before == [known, known]
    assert after == [known, ("unknown", "unknown")]


@pytest.mark.parametrize("pruned", ["first_run_events", "retry_event"])
def test_event_cleanup_after_an_alternate_route_retry_changes_no_earlier_stop_receipt(
    pruned, root, owner, clock, monkeypatch,
):
    from hermes_cli import owner_retry_route
    from plugins.dashboard_auth.raphael_workspace import model_policy

    import sqlite3

    profile_home = Path(os.environ["HERMES_HOME"])
    with monkeypatch.context() as shared_root:
        shared_root.setenv("HERMES_HOME", str(root))
        project = _project(owner, "Deep Worker Retention")
        with contextlib.closing(sqlite3.connect(root / "projects.db")) as source:
            with contextlib.closing(sqlite3.connect(profile_home / "projects.db")) as target:
                source.backup(target)
    board = project["board"]
    role = "raphael-claude-worker"
    (root / "profiles" / role).mkdir()
    own = model_policy.task_assignment_for(role, "anthropic", "deep")
    future = model_policy.task_assignment_for(role, "openai-codex", "deep")

    def receipt(provider: str, assignment) -> dict:
        return {
            "schema_version": 3, "engine": "hermes", "profile": role, "provider": provider,
            "model": assignment.model, "reasoning_effort": assignment.reasoning_effort,
            "route_evidence": "session-row",
            "cost": {"state": "estimated", "currency": "USD", "scope": "dominant-main-route", "amount": 0.25},
        }

    # The first stop's session names the route the card only gets later: its
    # receipt matches that future pin and never its own.
    mismatched = "placeholder-session-0002"
    receipts = {mismatched: receipt("openai-codex", future), _SESSION: receipt("anthropic", own)}
    monkeypatch.setattr(
        kb, "_trusted_runtime_receipt",
        lambda session_id, _profile: dict(receipts[session_id]) if session_id in receipts else None,
    )

    def stop(conn, task_id: str, pid: int, session_id: str):
        """The dispatcher spawns the card; its worker links its session and stops at the limit.

        The worker records its own provider beside its reset right before it
        exits, so the stop is booked at Anthropic whatever its receipt names.
        """
        _result, spawned = _tick(conn, board, {task_id: pid})
        assert spawned == [task_id]
        run_id = kb.get_task(conn, task_id).current_run_id
        assert kb.heartbeat_worker(conn, task_id, expected_run_id=run_id, session_id=session_id)
        with monkeypatch.context() as worker:
            worker.setenv("HERMES_KANBAN_TASK", task_id)
            worker.setenv("HERMES_KANBAN_RUN_ID", str(run_id))
            assert kb.record_run_rate_limit_reset(
                conn, task_id, run_id=run_id, reset_at=clock["t"] + COOLDOWN, provider="anthropic",
            )
        kb._record_worker_exit(pid, kb.KANBAN_RATE_LIMIT_EXIT_CODE << 8)
        kb.detect_crashed_workers(conn)
        run = kb.get_run(conn, run_id)
        assert run.outcome == "rate_limited"
        assert kanban_provider_stops._recorded_for(run.metadata) == (True, "anthropic")
        assert kb.get_task(conn, task_id).status == "ready"
        return run

    with _board(board) as conn:
        task_id = kb.create_task(
            conn, title="Draft the placeholder brief", assignee=role,
            project_id=project["project_id"], provider_override="anthropic",
            model_override=own.model, reasoning_effort=own.reasoning_effort,
            execution_tier="deep", requires_review=True,
            model_policy_lock=model_policy.mint_policy_lock(
                role, "anthropic", own.model, own.reasoning_effort, "deep",
            ),
        )
        first = stop(conn, task_id, 4231, mismatched)
        # The cooldown ends and the dispatcher tries the original route again.
        clock["t"] = _resume_at(first) + 1
        second = stop(conn, task_id, 4232, _SESSION)
    clock["t"] = second.ended_at + 60
    monkeypatch.setenv("HERMES_OPENAI_RETRY_TASK_ID", task_id)
    monkeypatch.setattr(owner_retry_route, "openai_ready", lambda _role: True)
    env = _steward_env(root, board)

    def stored_runs() -> list:
        with _board(board) as conn:
            return [
                tuple(row)
                for row in conn.execute("SELECT * FROM task_runs WHERE task_id = ? ORDER BY id", (task_id,))
            ]

    def read() -> tuple:
        snapshot = ow.read_project_snapshot(owner, project["slug"], provider_wait=True)
        receipts = [
            (run["receipt"]["runtime"]["state"], run["receipt"]["cost"]["state"])
            for run in snapshot["runs"]
        ]
        return snapshot, receipts

    runs = stored_runs()
    assert len(runs) == 2
    with _as_worker(monkeypatch, env):
        snapshot, before = read()
        choice = next(
            task["retry_route"]
            for column in snapshot["columns"] for task in column["tasks"] if task["id"] == task_id
        )
        accepted = _retry(owner, project, task_id, "retry-alternate", route_token=choice["token"])
        assert accepted["ok"] is True, accepted
        _snapshot, after = read()
    assert stored_runs() == runs

    with _board(board) as conn:
        assert kb.get_task(conn, task_id).provider_override == "openai-codex"
        [retry] = _events(conn, task_id, "owner_retry")
        assert retry.run_id == second.id
        clock["t"] = retry.created_at + 60
        assert kb.archive_task(conn, task_id)
        # Normal 30-day cleanup: its cutoff lies just past the first run's
        # events, or just past the retry event itself.
        cutoff = second.started_at if pruned == "first_run_events" else retry.created_at + 1
        clock["t"] = cutoff + 30 * 24 * 3600
        assert kb.gc_events(conn) > 0
        events = kb.list_events(conn, task_id)
    assert stored_runs() == runs
    assert not [event for event in events if event.run_id == first.id]
    assert [event.run_id for event in events if event.kind == "owner_retry"] == (
        [second.id] if pruned == "first_run_events" else []
    )
    assert bool([event for event in events if event.kind == "model_route_repinned"]) == (
        pruned == "first_run_events"
    )
    assert events[-1].kind == "archived"

    with _as_worker(monkeypatch, env):
        _snapshot, cleaned = read()
    unknown = ("unknown", "unknown")
    # Newest first: the stop the owner retried, then the attempt before it.
    assert before == [("known", "estimated"), unknown]
    assert after == before
    assert cleaned == before


# The owner's rule of 2026-11-27: an approved switch writes one switch record
# for its card, and each run of that card reads by the table of point 4, the
# same in the project snapshot and in the run detail.
_ROLE = "raphael-claude-worker"
_ANTHROPIC_SESSION = "placeholder-session-0101"
_OPENAI_SESSION = "placeholder-session-0102"
_KNOWN, _UNKNOWN = ("known", "estimated"), ("unknown", "unknown")
_SWITCH_REFUSAL = (
    "This retry choice could not be saved, so nothing was retried. This work is still waiting."
)


def _deep_route(provider: str) -> dict:
    from plugins.dashboard_auth.raphael_workspace import model_policy

    assignment = model_policy.task_assignment_for(_ROLE, provider, "deep")
    return {
        "provider_override": provider, "model_override": assignment.model,
        "reasoning_effort": assignment.reasoning_effort, "execution_tier": "deep",
        "model_policy_lock": model_policy.mint_policy_lock(
            _ROLE, provider, assignment.model, assignment.reasoning_effort, "deep",
        ),
    }


def _deep_project(root: Path, owner: ow.OwnerContext, name: str, monkeypatch) -> dict:
    """A Project the owner reads; each made-up session's receipt names one deep worker route."""
    import sqlite3

    profile_home = Path(os.environ["HERMES_HOME"])
    with monkeypatch.context() as shared_root:
        shared_root.setenv("HERMES_HOME", str(root))
        project = _project(owner, name)
        with contextlib.closing(sqlite3.connect(root / "projects.db")) as source:
            with contextlib.closing(sqlite3.connect(profile_home / "projects.db")) as target:
                source.backup(target)
    (root / "profiles" / _ROLE).mkdir(exist_ok=True)
    routes = {_ANTHROPIC_SESSION: _deep_route("anthropic"), _OPENAI_SESSION: _deep_route("openai-codex")}

    def receipt(session_id: str, _profile: str):
        route = routes.get(session_id)
        return None if route is None else {
            "schema_version": 3, "engine": "hermes", "profile": _ROLE,
            "provider": route["provider_override"], "model": route["model_override"],
            "reasoning_effort": route["reasoning_effort"], "route_evidence": "session-row",
            "cost": {"state": "estimated", "currency": "USD", "scope": "dominant-main-route", "amount": 0.25},
        }

    monkeypatch.setattr(kb, "_trusted_runtime_receipt", receipt)
    return project


def _deep_card(conn, project: dict, title: str) -> str:
    return kb.create_task(
        conn, title=title, assignee=_ROLE, project_id=project["project_id"],
        requires_review=True, **_deep_route("anthropic"),
    )


def _deep_stop(conn, board: str, task_id: str, pid: int, session_id: str, clock, monkeypatch, *, provider: str = "anthropic"):
    """The dispatcher spawns the card; its worker links ``session_id`` and records its stop at ``provider``."""
    _result, spawned = _tick(conn, board, {task_id: pid})
    assert spawned == [task_id]
    run_id = kb.get_task(conn, task_id).current_run_id
    assert kb.heartbeat_worker(conn, task_id, expected_run_id=run_id, session_id=session_id)
    with monkeypatch.context() as worker:
        worker.setenv("HERMES_KANBAN_TASK", task_id)
        worker.setenv("HERMES_KANBAN_RUN_ID", str(run_id))
        assert kb.record_run_rate_limit_reset(
            conn, task_id, run_id=run_id, reset_at=clock["t"] + COOLDOWN, provider=provider,
        )
    kb._record_worker_exit(pid, kb.KANBAN_RATE_LIMIT_EXIT_CODE << 8)
    kb.detect_crashed_workers(conn)
    run = kb.get_run(conn, run_id)
    assert run.outcome == "rate_limited"
    assert kanban_provider_stops._recorded_for(run.metadata) == (True, provider)
    return run


def _switch(owner: ow.OwnerContext, project: dict, task_id: str, env: dict, monkeypatch, key: str) -> dict:
    """The owner retries the held card on the route its snapshot offers, with that offer's token."""
    with _as_worker(monkeypatch, {**env, "HERMES_OPENAI_RETRY_TASK_ID": task_id}):
        snapshot = ow.read_project_snapshot(owner, project["slug"], provider_wait=True)
        choice = next(
            task["retry_route"]
            for column in snapshot["columns"] for task in column["tasks"] if task["id"] == task_id
        )
        return _retry(owner, project, task_id, key, route_token=choice["token"])


def _both_readers(owner: ow.OwnerContext, project: dict, board: str, env: dict, monkeypatch) -> dict:
    """Each run's receipt by run id, read by the project snapshot and equal in the run detail."""
    from plugins.kanban.dashboard import plugin_api

    with _as_worker(monkeypatch, env):
        snapshot = ow.read_project_snapshot(owner, project["slug"])
    with _board(board) as conn:
        run_ids = [row[0] for row in conn.execute(
            "SELECT r.id FROM task_runs r JOIN tasks t ON t.id = r.task_id "
            "WHERE t.project_id = ? AND t.task_kind = 'work' ORDER BY r.started_at DESC, r.id DESC",
            (project["project_id"],),
        )]
    monkeypatch.setattr(plugin_api, "WORKSPACE_BOARD", board)
    receipts = {}
    for run_id, run in zip(run_ids, snapshot["runs"], strict=True):
        detail = plugin_api._workspace_run_response(run_id)["run"]["receipt"]
        # On main the run detail never carries the owner's retry reason.
        assert {key: value for key, value in run["receipt"].items() if key != "owner_retry"} == detail
        receipts[run_id] = detail
    return receipts


def _states(receipts: dict, *run_ids: int) -> list:
    return [(receipts[run_id]["runtime"]["state"], receipts[run_id]["cost"]["state"]) for run_id in run_ids]


def _switch_scene(root: Path, owner: ow.OwnerContext, clock, monkeypatch) -> SimpleNamespace:
    """Three deep worker cards on one board, read through both readers at each step.

    Card ``m`` stops three times on its own Anthropic route and the owner
    switches it on the third; it then stops on its present OpenAI route and on
    the old one. Card ``x`` stops once on a receipt naming the OpenAI route
    before it has it, and the owner switches it. Card ``n`` is never switched.
    """
    from hermes_cli import owner_retry_route

    project = _deep_project(root, owner, "Switch Records", monkeypatch)
    board = project["board"]
    monkeypatch.setattr(owner_retry_route, "openai_ready", lambda _role: True)
    scene = SimpleNamespace(project=project, board=board, env=_steward_env(root, board))
    with _board(board) as conn:
        scene.m = _deep_card(conn, project, "Draft the placeholder brief")
        first = _deep_stop(conn, board, scene.m, 4301, _ANTHROPIC_SESSION, clock, monkeypatch)
        clock["t"] = _resume_at(first) + 1
        second = _deep_stop(conn, board, scene.m, 4302, _ANTHROPIC_SESSION, clock, monkeypatch)
        clock["t"] = _resume_at(second) + 1
        held = _deep_stop(conn, board, scene.m, 4303, _ANTHROPIC_SESSION, clock, monkeypatch)
    scene.earlier, scene.held = [first.id, second.id], held.id
    clock["t"] = held.ended_at + 60
    scene.before = _both_readers(owner, project, board, scene.env, monkeypatch)
    assert _switch(owner, project, scene.m, scene.env, monkeypatch, "switch-m")["ok"] is True
    scene.switched = _both_readers(owner, project, board, scene.env, monkeypatch)
    with _board(board) as conn:
        present = _deep_stop(
            conn, board, scene.m, 4304, _OPENAI_SESSION, clock, monkeypatch, provider="openai-codex",
        )
        clock["t"] = _resume_at(present) + 1
        stale = _deep_stop(
            conn, board, scene.m, 4305, _ANTHROPIC_SESSION, clock, monkeypatch, provider="openai-codex",
        )
        scene.later = [present.id, stale.id]
        scene.x = _deep_card(conn, project, "Check the placeholder figures")
        scene.x_held = _deep_stop(conn, board, scene.x, 4306, _OPENAI_SESSION, clock, monkeypatch).id
        scene.n = _deep_card(conn, project, "Sort the placeholder notes")
        scene.n_run = _deep_stop(conn, board, scene.n, 4307, _ANTHROPIC_SESSION, clock, monkeypatch).id
    clock["t"] += 60
    scene.x_before = _both_readers(owner, project, board, scene.env, monkeypatch)
    assert _switch(owner, project, scene.x, scene.env, monkeypatch, "switch-x")["ok"] is True
    scene.final = _both_readers(owner, project, board, scene.env, monkeypatch)
    return scene


def test_after_a_switch_two_runs_before_the_held_run_read_unknown_in_both_readers(root, owner, clock, monkeypatch):
    scene = _switch_scene(root, owner, clock, monkeypatch)
    assert _states(scene.before, *scene.earlier) == [_KNOWN, _KNOWN]
    assert _states(scene.switched, *scene.earlier) == [_UNKNOWN, _UNKNOWN]
    assert _states(scene.final, *scene.earlier) == [_UNKNOWN, _UNKNOWN]


def test_after_a_switch_a_held_run_matching_the_record_reads_as_main_reads_a_matching_run(
    root, owner, clock, monkeypatch,
):
    scene = _switch_scene(root, owner, clock, monkeypatch)
    assert _states(scene.before, scene.held) == [_KNOWN]
    assert scene.switched[scene.held] == scene.before[scene.held]
    assert scene.final[scene.held] == scene.before[scene.held]


def test_after_a_switch_a_held_run_not_matching_the_record_reads_unknown_in_both_readers(
    root, owner, clock, monkeypatch,
):
    scene = _switch_scene(root, owner, clock, monkeypatch)
    # Its receipt names the OpenAI route, never the Anthropic pin the record keeps.
    assert _states(scene.x_before, scene.x_held) == [_UNKNOWN]
    assert _states(scene.final, scene.x_held) == [_UNKNOWN]


def test_after_a_switch_a_later_run_is_checked_against_the_present_pin_in_both_readers(
    root, owner, clock, monkeypatch,
):
    scene = _switch_scene(root, owner, clock, monkeypatch)
    # The first names the card's present OpenAI route, the second its old one.
    assert _states(scene.final, *scene.later) == [_KNOWN, _UNKNOWN]
    assert _states(scene.final, scene.held) == [_KNOWN]


@pytest.mark.parametrize("cleanup", ["every_event", "claimed_and_spawned"])
def test_event_cleanup_after_a_switch_changes_no_receipt_in_either_reader(
    cleanup, root, owner, clock, monkeypatch,
):
    scene = _switch_scene(root, owner, clock, monkeypatch)
    cards = (scene.m, scene.x, scene.n)
    with _board(scene.board) as conn:
        if cleanup == "every_event":
            for task_id in cards:
                assert kb.archive_task(conn, task_id)
            # Normal 30-day cleanup, its cutoff past every event of the cards.
            clock["t"] += 30 * 24 * 3600 + 1
            assert kb.gc_events(conn) > 0
        else:
            slots = ", ".join("?" for _ in scene.earlier)
            assert conn.execute(
                "DELETE FROM task_events WHERE kind IN ('claimed', 'spawned') "
                f"AND run_id IN ({slots})",
                scene.earlier,
            ).rowcount > 0
        earlier = [
            event for event in kb.list_events(conn, scene.m) if event.run_id in scene.earlier
        ]
    if cleanup == "every_event":
        assert not earlier
    else:
        assert earlier
        assert not [event for event in earlier if event.kind in ("claimed", "spawned")]
    cleaned = _both_readers(owner, scene.project, scene.board, scene.env, monkeypatch)
    assert cleaned == scene.final
    assert _states(cleaned, *scene.earlier, scene.held, *scene.later, scene.x_held, scene.n_run) == [
        _UNKNOWN, _UNKNOWN, _KNOWN, _KNOWN, _UNKNOWN, _UNKNOWN, _KNOWN,
    ]


def test_a_card_with_no_switch_record_reads_as_on_main_in_both_readers(root, owner, clock, monkeypatch):
    scene = _switch_scene(root, owner, clock, monkeypatch)
    route = _deep_route("anthropic")
    # Before any switch the board has no store; each run reads as main reads it.
    assert _states(scene.before, *scene.earlier, scene.held) == [_KNOWN, _KNOWN, _KNOWN]
    # Beside two cards with a record, the card with none still does.
    assert scene.final[scene.n_run] == scene.x_before[scene.n_run]
    assert scene.final[scene.n_run]["runtime"] == {
        "state": "known", "engine": "hermes", "profile": _ROLE, "provider": "anthropic",
        "model": route["model_override"], "reasoning_effort": route["reasoning_effort"],
    }
    assert _states(scene.final, scene.n_run) == [_KNOWN]


def test_a_switch_whose_record_cannot_be_written_is_refused_and_the_card_keeps_its_route(
    root, owner, clock, monkeypatch,
):
    from hermes_cli import owner_retry_route

    project = _deep_project(root, owner, "Unsaved Switch", monkeypatch)
    board = project["board"]
    monkeypatch.setattr(owner_retry_route, "openai_ready", lambda _role: True)
    env = _steward_env(root, board)
    with _board(board) as conn:
        task_id = _deep_card(conn, project, "Draft the placeholder brief")
        held = _deep_stop(conn, board, task_id, 4311, _ANTHROPIC_SESSION, clock, monkeypatch)
        # A store of the record's own shape that refuses every record.
        conn.execute(
            "CREATE TABLE owner_retry_run_pins (task_id TEXT PRIMARY KEY, "
            "held_run_id INTEGER NOT NULL, assignee TEXT, provider_override TEXT, "
            "model_override TEXT, reasoning_effort TEXT, execution_tier TEXT, model_policy_lock TEXT)"
        )
        conn.execute(
            "CREATE TRIGGER owner_retry_run_pins_refused BEFORE INSERT ON owner_retry_run_pins "
            "BEGIN SELECT RAISE(ABORT, 'placeholder refusal'); END"
        )
    clock["t"] = held.ended_at + 60

    def board_state() -> tuple:
        with _board(board) as conn:
            return (
                tuple(conn.execute(
                    "SELECT assignee, provider_override, model_override, reasoning_effort, "
                    "execution_tier, model_policy_lock FROM tasks WHERE id = ?", (task_id,),
                ).fetchone()),
                [(event.id, event.kind, event.run_id) for event in kb.list_events(conn, task_id)],
                [tuple(row) for row in conn.execute("SELECT * FROM owner_retry_run_pins")],
                [tuple(row) for row in conn.execute(
                    "SELECT type, name, sql FROM sqlite_master "
                    "WHERE tbl_name = 'owner_retry_run_pins' ORDER BY name"
                )],
                kb.check_respawn_guard(conn, task_id),
            )

    before = board_state()
    assert before[0][1] == "anthropic"
    assert before[2] == []
    assert before[-1] == "rate_limit_cooldown"
    refused = _switch(owner, project, task_id, env, monkeypatch, "switch-unsaved")
    assert refused["ok"] is False, refused
    assert refused["reason"] == _SWITCH_REFUSAL
    assert board_state() == before
    with _board(board) as conn:
        assert not _events(conn, task_id, "model_route_repinned")
        assert not _events(conn, task_id, "owner_retry")


def test_on_a_board_with_no_switch_store_the_run_detail_reads_as_main_and_creates_nothing(
    root, owner, clock, monkeypatch,
):
    import sqlite3

    from plugins.kanban.dashboard import plugin_api

    project = _deep_project(root, owner, "No Switch Store", monkeypatch)
    board = project["board"]
    env = _steward_env(root, board)
    with _board(board) as conn:
        task_id = _deep_card(conn, project, "Draft the placeholder brief")
        run = _deep_stop(conn, board, task_id, 4321, _ANTHROPIC_SESSION, clock, monkeypatch)
    [receipt] = _both_readers(owner, project, board, env, monkeypatch).values()
    route = _deep_route("anthropic")
    assert receipt["runtime"] == {
        "state": "known", "engine": "hermes", "profile": _ROLE, "provider": "anthropic",
        "model": route["model_override"], "reasoning_effort": route["reasoning_effort"],
    }
    assert receipt["cost"]["state"] == "estimated"
    path = kb.kanban_db_path(board=board)
    stored = path.read_bytes()
    assert plugin_api._workspace_run_response(run.id)["run"]["receipt"] == receipt
    assert path.read_bytes() == stored
    with contextlib.closing(sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True)) as conn:
        assert conn.execute(
            "SELECT name FROM sqlite_master WHERE name = 'owner_retry_run_pins'"
        ).fetchall() == []


@pytest.mark.parametrize("step", [
    "owner_workspace._owner_board_counts",
    "kanban_delivery.unfinished_delivery_task_ids",
    "kanban_delivery_outcome.delivery_outcomes",
    "owner_retry_route.offer",
    "kanban_db.verified_active_worker_rows",
    "owner_workspace._owner_run_retry_reasons",
    "kanban_db.run_retry_origins",
])
def test_another_connection_commits_at_once_while_the_snapshot_reads_past_the_offer_inputs(
    step, root, owner, clock, monkeypatch,
):
    """On a rollback-journal board a read left open blocks every other connection's commit.

    The snapshot reads the cards, their revisions, the provider waits and the
    held runs, which the retry offer binds, from one snapshot and keeps no read
    open after them, so a commit beside each later step goes through at once.
    """
    import importlib
    import sqlite3

    project = _deep_project(root, owner, "Read Beside A Commit", monkeypatch)
    board = project["board"]
    env = _steward_env(root, board)
    with _board(board) as conn:
        task_id = _deep_card(conn, project, "Draft the placeholder brief")
        held = _deep_stop(conn, board, task_id, 4331, _ANTHROPIC_SESSION, clock, monkeypatch)
    clock["t"] = held.ended_at + 60
    path = kb.kanban_db_path(board=board)
    with contextlib.closing(sqlite3.connect(path)) as conn:
        # Set on the file, so the test does not depend on the SQLite build.
        assert conn.execute("PRAGMA journal_mode=DELETE").fetchone()[0] == "delete"
        [priority] = conn.execute("SELECT priority FROM tasks WHERE id = ?", (task_id,)).fetchone()
    module_name, name = step.split(".")
    module = importlib.import_module(f"hermes_cli.{module_name}")
    real, commits = getattr(module, name), []

    def step_then_commit(*args, **kwargs):
        result = real(*args, **kwargs)
        try:
            # No busy wait: a commit the reader blocks fails at once.
            with contextlib.closing(sqlite3.connect(path, timeout=0)) as other:
                with other:
                    other.execute("UPDATE tasks SET priority = priority + 1 WHERE id = ?", (task_id,))
        except sqlite3.OperationalError as exc:
            commits.append(str(exc))
        else:
            commits.append("committed")
        return result

    monkeypatch.setattr(module, name, step_then_commit)
    with _as_worker(monkeypatch, {**env, "HERMES_OPENAI_RETRY_TASK_ID": task_id}):
        snapshot = ow.read_project_snapshot(
            owner, project["slug"], run_context=True, owner_waiting=True, provider_wait=True,
            delivery_state=True, delivery_outcome=True, work_item=True,
        )

    assert commits and set(commits) == {"committed"}, commits
    with contextlib.closing(sqlite3.connect(path)) as conn:
        assert conn.execute(
            "SELECT priority FROM tasks WHERE id = ?", (task_id,),
        ).fetchone() == (priority + len(commits),)
    card = _task(snapshot, task_id)
    assert card["stopped_work"] == "provider_wait"
    assert set(card["retry_route"]) == {"token", "model", "effort"}
    assert (card["delivery"], card["delivery_outcome"]) == (None, None)
    [run] = snapshot["runs"]
    assert run["receipt"]["outcome"] == "waiting"


def test_a_retry_while_the_limit_holds_leaves_the_card_waiting_with_a_new_reset_and_no_breaker_trip(
    root, owner, clock, monkeypatch,
):
    project = _project(owner, "Still Limited")
    board = project["board"]
    with _board(board) as conn:
        task_id = _card(conn, project, "Draft the placeholder summary")
        first = _held(conn, board, task_id, 4301)
        clock["t"] = first.ended_at + 60

    env = _steward_env(root, board)
    with _as_worker(monkeypatch, env):
        retried = _retry(owner, project, task_id, "retry-still-limited")
    assert retried["ok"] is True, retried

    with _board(board) as conn:
        clock["t"] += 30
        second = _held(conn, board, task_id, 4302)
        assert second.id != first.id
        assert _resume_at(second) == second.ended_at + 4 * COOLDOWN
        assert _resume_at(second) > _resume_at(first)

        task = kb.get_task(conn, task_id)
        assert task.status == "ready"
        assert task.consecutive_failures == 0
        assert _events(conn, task_id, "gave_up") == []

        # The retry was bound to the first run; the second run's hold stands.
        assert [event.run_id for event in _events(conn, task_id, "owner_retry")] == [first.id]
        before = len(_events(conn, task_id, "respawn_guarded"))
        clock["t"] = second.ended_at + 60
        for _ in range(3):
            assert kb.check_respawn_guard(conn, task_id) == "rate_limit_cooldown"
            result, spawned = _tick(conn, board, {task_id: 4303})
            assert (task_id, "rate_limit_cooldown") in result.respawn_guarded
            assert spawned == []
        assert len(_events(conn, task_id, "respawn_guarded")) == before + 1
        assert kb.get_task(conn, task_id).consecutive_failures == 0


def _unparsed_store(shape: str) -> bytes:
    """The assignee's own store, holding an exhausted entry, in a shape no store is read from."""
    store = {"version": 1, "credential_pool": {BUILT_IN: [_exhausted("own-a")]}}
    text = json.dumps(store, indent=2)
    return {
        "truncated": text[: len(text) // 2].encode(),
        "not_utf8": b"\xff" + text.encode(),
        "not_an_object": json.dumps([store], indent=2).encode(),
    }[shape]


@pytest.mark.parametrize("shape", ["truncated", "not_utf8", "not_an_object"])
def test_the_retry_goes_ahead_and_writes_nothing_beside_an_own_store_that_does_not_parse(
    shape, root, owner, clock, monkeypatch, caplog,
):
    project = _project(owner, "Broken Store")
    board = project["board"]
    with _board(board) as conn:
        task_id = _card(conn, project, "Draft the placeholder card", provider=BUILT_IN)
        run = _held(conn, board, task_id, 4901)
        clock["t"] = run.ended_at + 60

    home = root / "profiles" / ASSIGNEE
    own_pool = home / "auth.json"
    own_pool.write_bytes(_unparsed_store(shape))
    raw, names = own_pool.read_bytes(), set(os.listdir(home))

    env = _steward_env(root, board)
    caplog.set_level("DEBUG")
    with _as_worker(monkeypatch, env):
        retried = _retry(owner, project, task_id, "retry-broken-store")
    assert retried["ok"] is True, retried

    assert own_pool.read_bytes() == raw
    # Only the lock the clear reads under may leave its file: no copy, no temporary file.
    assert set(os.listdir(home)) - names <= {own_pool.with_suffix(".lock").name}
    with _board(board) as conn:
        [event] = _events(conn, task_id, "owner_retry")
        assert event.run_id == run.id
        assert kb.check_respawn_guard(conn, task_id) is None
        events = json.dumps([event.payload for event in kb.list_events(conn, task_id)])
    for leaked in (json.dumps(retried), events, caplog.text):
        assert "placeholder-token" not in leaked


@pytest.mark.parametrize("state", ["missing", "unreadable", "self_referential"])
def test_the_retry_goes_ahead_and_changes_no_pool_when_the_own_pool_path_is_unusable(
    state, root, owner, clock, monkeypatch, caplog,
):
    project = _project(owner, "Unusable Pool")
    board = project["board"]
    with _board(board) as conn:
        task_id = _card(conn, project, "Draft the placeholder report", provider=BUILT_IN)
        run = _held(conn, board, task_id, 5001)
        clock["t"] = run.ended_at + 60

    root_pool = _write_pool(root, {BUILT_IN: [_exhausted("root-a")]})
    steward_pool = _write_pool(root / "profiles" / STEWARD, {BUILT_IN: [_exhausted("steward-a")]})
    other_pool = _write_pool(root / "profiles" / OTHER, {BUILT_IN: [_exhausted("other-a")]})
    pools = (root_pool, steward_pool, other_pool)
    untouched = {path: path.read_bytes() for path in pools}
    before = _exhausted_entries(pools)

    home = root / "profiles" / ASSIGNEE
    own_pool = home / "auth.json"
    if state == "unreadable":
        # A valid store holding an exhausted entry that no one, root included,
        # may read: only this exact path is refused, every other read is real.
        _write_pool(home, {BUILT_IN: [_exhausted("own-a")]})
        read_text = Path.read_text

        def refused(path, *args, **kwargs):
            if path == own_pool:
                raise PermissionError(errno.EACCES, os.strerror(errno.EACCES), str(path))
            return read_text(path, *args, **kwargs)

        monkeypatch.setattr(Path, "read_text", refused)
    elif state == "self_referential":
        own_pool.symlink_to(own_pool.name)
    own_raw = own_pool.read_bytes() if state == "unreadable" else None
    names = set(os.listdir(home))

    env = _steward_env(root, board)
    caplog.set_level("DEBUG")
    with _as_worker(monkeypatch, env):
        retried = _retry(owner, project, task_id, "retry-unusable-pool")
    assert retried["ok"] is True, retried

    for path, raw in untouched.items():
        assert path.read_bytes() == raw
    assert _exhausted_entries(pools) == before
    if state == "missing":
        assert not os.path.lexists(own_pool)
    elif state == "unreadable":
        assert own_pool.read_bytes() == own_raw
    else:
        assert os.readlink(own_pool) == own_pool.name
    # The owner never reaches a pool: no lock, no copy, no temporary file.
    assert set(os.listdir(home)) == names

    with _board(board) as conn:
        [event] = _events(conn, task_id, "owner_retry")
        assert event.run_id == run.id
        assert kb.get_task(conn, task_id).status == "ready"
        assert kb.check_respawn_guard(conn, task_id) is None
        events = json.dumps([event.payload for event in kb.list_events(conn, task_id)])
    records = [record for record in caplog.records if record.name == "agent.credential_pool"]
    # Nothing tried to clear the pool, so nothing warns that it could not.
    assert [record for record in records if record.levelno >= logging.WARNING] == []
    logged = "\n".join(logging.Formatter().format(record) for record in records)
    for leaked in (json.dumps(retried), events, logged):
        for value in ("placeholder-token", str(own_pool), str(home)):
            assert value not in leaked


def _waiting_and_refused(root: Path, owner: ow.OwnerContext) -> dict:
    """A Project with one card held at the provider's limit and one it refused.

    Both stops are reaped by the real sweep. The steward's own claimed card
    sits on a board of its own, so its worker env is pinned away from the
    Project's board.
    """
    import sqlite3
    profile_home = Path(os.environ["HERMES_HOME"])
    with pytest.MonkeyPatch.context() as shared_root:
        shared_root.setenv("HERMES_HOME", str(root))
        project = _project(owner, "Wait Pilot")
        # The Project lives in the root registry. Card creation and the owner's
        # calls still use the steward's own registry, so the steward holds a copy.
        with contextlib.closing(sqlite3.connect(root / "projects.db")) as source:
            with contextlib.closing(sqlite3.connect(profile_home / "projects.db")) as target:
                source.backup(target)
    board = project["board"]
    with _board(board) as conn:
        waiting = _card(conn, project, "Draft the placeholder brief")
        held = _held(conn, board, waiting, 4501)
        refused = _card(conn, project, "Word the placeholder notice")
        declined = _reaped(conn, board, refused, 4502, KANBAN_PROVIDER_REFUSED_EXIT_CODE)
        assert declined.outcome == "provider_refused"
        assert kb.get_task(conn, refused).status == "blocked"
    kb.create_board("steward-desk")
    return {
        "project": project, "waiting": waiting, "refused": refused,
        "held": held, "env": _steward_env(root, "steward-desk"),
    }


def _normalized(payload: dict, scene: dict) -> dict:
    """The payload with this run's generated ids replaced by fixed names."""
    project = scene["project"]
    text = json.dumps(payload, sort_keys=True)
    names = {
        scene["waiting"]: "<waiting-task>",
        scene["refused"]: "<refused-task>",
        project["project_id"]: "<project-id>",
        project["board"]: "<board>",
        project["slug"]: "<slug>",
    }
    for value in sorted(names, key=len, reverse=True):
        text = text.replace(value, names[value])
    return json.loads(text)


def test_with_the_capability_one_receipt_carries_the_time_and_another_the_refused_line(
    root, owner, clock, monkeypatch,
):
    scene = _waiting_and_refused(root, owner)
    project, held = scene["project"], scene["held"]
    resume_at = held.ended_at + 2 * COOLDOWN
    with _board(project["board"]) as conn:
        clock["t"] = resume_at - 1
        assert kb.check_respawn_guard(conn, scene["waiting"]) == "rate_limit_cooldown"
        clock["t"] = resume_at
        assert kb.check_respawn_guard(conn, scene["waiting"]) is None
    clock["t"] = held.ended_at + 60

    with _as_worker(monkeypatch, scene["env"]):
        snapshot = ow.read_project_snapshot(owner, project["slug"], provider_wait=True)
        plain = ow.read_project_snapshot(owner, project["slug"])
        # Pinned to its own board, the worker still reaches the root.
        assert kb.kanban_home() == root
        assert kb.register_db_path() == root / "kanban" / "board_register.db"
        entry = kb.get_register_entry(project["board"])
        assert entry is not None and entry.lifecycle is kb.BoardLifecycle.LIVE

    when = datetime.fromtimestamp(resume_at, timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    receipts = [run["receipt"] for run in snapshot["runs"]]
    assert {"outcome": "waiting", "summary": WAITING_LINE.format(time=when)} in [
        {"outcome": receipt["outcome"], "summary": receipt["summary"]} for receipt in receipts
    ]
    assert REFUSED_LINE in [receipt["summary"] for receipt in receipts]
    stopped = {
        task["id"]: task["stopped_work"]
        for column in snapshot["columns"] for task in column["tasks"]
    }
    assert stopped[scene["waiting"]] == "provider_wait"
    assert stopped[scene["refused"]] == plain_stopped(plain)[scene["refused"]]

    # The independent count, straight from the Project board's own rows.
    path = kb.board_dir(project["board"]) / "kanban.db"
    with contextlib.closing(kb.connect(db_path=path)) as conn:
        counts = {
            outcome: conn.execute(
                "SELECT COUNT(*) AS n FROM task_runs r JOIN tasks t ON t.id = r.task_id "
                "WHERE t.project_id = ? AND r.outcome = ?",
                (project["project_id"], outcome),
            ).fetchone()["n"]
            for outcome in ("rate_limited", "provider_refused")
        }
    assert counts == {"rate_limited": 1, "provider_refused": 1}
    assert sum(receipt["outcome"] == "waiting" for receipt in receipts) == counts["rate_limited"]
    assert sum(receipt["summary"] == REFUSED_LINE for receipt in receipts) == counts["provider_refused"]
    assert sum(state == "provider_wait" for state in stopped.values()) == counts["rate_limited"]
    for receipt in receipts:
        assert PROVIDER not in receipt["summary"]


def plain_stopped(snapshot: dict) -> dict:
    return {
        task["id"]: task["stopped_work"]
        for column in snapshot["columns"] for task in column["tasks"]
    }


def test_with_the_capability_the_waiting_time_is_the_instant_the_guard_releases_the_card(
    root, owner, clock, monkeypatch,
):
    scene = _waiting_and_refused(root, owner)
    project, held, waiting = scene["project"], scene["held"], scene["waiting"]
    clock["t"] = held.ended_at + 60
    with _board(project["board"]) as conn:
        assert kb.check_respawn_guard(conn, waiting) == "rate_limit_cooldown"
    with _as_worker(monkeypatch, scene["env"]):
        snapshot = ow.read_project_snapshot(owner, project["slug"], provider_wait=True)

    [summary] = [
        run["receipt"]["summary"] for run in snapshot["runs"]
        if run["receipt"]["outcome"] == "waiting"
    ]
    lead, tail = WAITING_LINE.split("{time}")
    assert summary.startswith(lead) and summary.endswith(tail), summary
    shown = datetime.fromisoformat(summary[len(lead):-len(tail)]).replace(tzinfo=timezone.utc)

    # The guard's own release instant, read straight from the Project board's rows.
    path = kb.board_dir(project["board"]) / "kanban.db"
    with contextlib.closing(kb.connect(db_path=path)) as conn:
        row = conn.execute(
            "SELECT outcome, ended_at, metadata FROM task_runs WHERE id = ? AND task_id = ?",
            (held.id, waiting),
        ).fetchone()
        release = kanban_provider_stops.rate_limit_resume_at(conn, waiting, row)
        assert release is not None and release % 60 != 0
        assert shown.timestamp() == release
        clock["t"] = release - 1
        assert kb.check_respawn_guard(conn, waiting) == "rate_limit_cooldown"
        clock["t"] = release
        assert kb.check_respawn_guard(conn, waiting) is None


# The whole snapshot served without ``provider_wait_v1``, captured at the start
# commit from the same waiting and refused runs, ids replaced by fixed names.
_SNAPSHOT_AT_START = '''
{
 "attachments": [],
 "board": {
  "counts": {
   "archived": 0,
   "blocked": 1,
   "done": 0,
   "ready": 1,
   "review": 0,
   "running": 0,
   "scheduled": 0,
   "todo": 0,
   "triage": 0
  },
  "name": "Wait Pilot",
  "project_id": "<project-id>",
  "slug": "<slug>",
  "total": 2
 },
 "columns": [
  {
   "name": "triage",
   "tasks": []
  },
  {
   "name": "todo",
   "tasks": []
  },
  {
   "name": "scheduled",
   "tasks": []
  },
  {
   "name": "ready",
   "tasks": [
    {
     "assignee_name": "worker",
     "child_ids": [],
     "event_revision": 8,
     "id": "<waiting-task>",
     "parent_ids": [],
     "responsibility": null,
     "review_state": "none",
     "stopped_work": "none",
     "title": "Draft the placeholder brief",
     "updated_at": "2030-03-17T17:46:40Z"
    }
   ]
  },
  {
   "name": "running",
   "tasks": []
  },
  {
   "name": "blocked",
   "tasks": [
    {
     "assignee_name": "worker",
     "child_ids": [],
     "event_revision": 11,
     "id": "<refused-task>",
     "parent_ids": [],
     "responsibility": null,
     "review_state": "none",
     "stopped_work": "capability",
     "title": "Word the placeholder notice",
     "updated_at": "2030-03-17T17:46:40Z"
    }
   ]
  },
  {
   "name": "review",
   "tasks": []
  },
  {
   "name": "done",
   "tasks": []
  }
 ],
 "project": {
  "archived": false,
  "board": "<slug>",
  "description": null,
  "id": "<project-id>",
  "name": "Wait Pilot",
  "slug": "<slug>"
 },
 "runs": [
  {
   "finished_at": "2030-03-17T17:46:40Z",
   "receipt": {
    "cost": {
     "state": "unknown",
     "summary": "This record does not contain an authoritative cost."
    },
    "evidence": {
     "kind": "project_activity",
     "state": "available"
    },
    "external_effect": {
     "state": "unknown",
     "summary": "This record does not confirm whether an external service changed."
    },
    "outcome": "unknown",
    "runtime": {
     "state": "unknown",
     "summary": "This record does not contain an authoritative model route."
    },
    "summary": "The final outcome could not be confirmed."
   },
   "started_at": "2030-03-17T17:46:40Z"
  },
  {
   "finished_at": "2030-03-17T17:46:40Z",
   "receipt": {
    "cost": {
     "state": "unknown",
     "summary": "This record does not contain an authoritative cost."
    },
    "evidence": {
     "kind": "project_activity",
     "state": "available"
    },
    "external_effect": {
     "state": "unknown",
     "summary": "This record does not confirm whether an external service changed."
    },
    "outcome": "attention",
    "runtime": {
     "state": "unknown",
     "summary": "This record does not contain an authoritative model route."
    },
    "summary": "Work stopped at the AI provider's usage limit and will start again by itself after 2030-03-17 17:56 UTC."
   },
   "started_at": "2030-03-17T17:46:40Z"
  }
 ],
 "steward": {
  "active_work": [
   {
    "state": "Ready",
    "title": "Draft the placeholder brief"
   }
  ],
  "counts": {
   "awaiting_review": 0,
   "completed_in_window": 0,
   "needs_attention": 1,
   "open": 2
  },
  "decisions_needed": [],
  "execution": {
   "paused": false,
   "state": "needs_attention",
   "summary": "Raphael found a problem and is preparing the safest next step."
  },
  "generated_at": "2030-03-17T17:47:40Z",
  "lookback_days": 7,
  "needs_attention": [
   {
    "reason": "A required capability is not available",
    "state": "Blocked",
    "title": "Word the placeholder notice"
   }
  ],
  "progress": [],
  "project": {
   "name": "Wait Pilot"
  },
  "schema_version": 2,
  "stale_candidates": [],
  "truncated": {
   "active_work": false,
   "decisions_needed": false,
   "needs_attention": false,
   "progress": false,
   "stale_candidates": false
  }
 },
 "truncated": {
  "attachments": false,
  "runs": false,
  "tasks": false,
  "workers": false
 },
 "workers": []
}
'''


def test_without_the_capability_the_snapshot_is_unchanged(root, owner, clock, monkeypatch):
    scene = _waiting_and_refused(root, owner)
    clock["t"] = scene["held"].ended_at + 60
    with _as_worker(monkeypatch, scene["env"]):
        snapshot = ow.read_project_snapshot(owner, scene["project"]["slug"])
    normalized = _normalized(snapshot, scene)
    for column in normalized["columns"]:
        for task in column["tasks"]:
            tier = task.pop("risk_tier")
            assert tier is None or (type(tier) is int and tier in (0, 1, 2))
            assert type(task.pop("risk_tier_raised")) is bool
    assert normalized == json.loads(_SNAPSHOT_AT_START)


def _task(snapshot: dict, task_id: str) -> dict:
    return next(
        task for column in snapshot["columns"] for task in column["tasks"]
        if task["id"] == task_id
    )


@pytest.mark.parametrize("ended_by", ["owner_retry", "expiry", "disabled_cooldown"])
def test_with_the_capability_a_card_whose_hold_has_ended_reads_exactly_as_without_it(
    ended_by, root, owner, clock, monkeypatch,
):
    scene = _waiting_and_refused(root, owner)
    project, held, waiting = scene["project"], scene["held"], scene["waiting"]
    board, env = project["board"], scene["env"]
    clock["t"] = held.ended_at + 60
    with _board(board) as conn:
        assert kb.check_respawn_guard(conn, waiting) == "rate_limit_cooldown"
    if ended_by == "owner_retry":
        with _as_worker(monkeypatch, _steward_env(root, board)):
            retried = _retry(owner, project, waiting, "retry-ends-the-hold")
        assert retried["ok"] is True, retried
    elif ended_by == "expiry":
        with _board(board) as conn:
            clock["t"] = _resume_at(held) - 1
            assert kb.check_respawn_guard(conn, waiting) == "rate_limit_cooldown"
        clock["t"] = _resume_at(held)
    else:
        # Booked while the cooldown was on; the dispatcher now runs with it off.
        monkeypatch.setenv("HERMES_KANBAN_RATE_LIMIT_COOLDOWN_SECONDS", "0")
        env = _steward_env(root, "steward-desk")
        assert env["HERMES_KANBAN_RATE_LIMIT_COOLDOWN_SECONDS"] == "0"

    with _as_worker(monkeypatch, env):
        snapshot = ow.read_project_snapshot(owner, project["slug"], provider_wait=True)
        plain = ow.read_project_snapshot(owner, project["slug"])

    # The card and the run it waited on read exactly as they do without the capability.
    [at] = [
        index for index, run in enumerate(plain["runs"])
        if run["receipt"]["outcome"] == "attention"
    ]
    assert (_task(snapshot, waiting), snapshot["runs"][at]) == (
        _task(plain, waiting), plain["runs"][at],
    )
    assert _task(snapshot, waiting)["stopped_work"] == "none"

    # The independent count, straight from the Project board's own rows: the
    # dispatcher holds nothing, so nothing reads as waiting.
    path = kb.board_dir(board) / "kanban.db"
    with contextlib.closing(kb.connect(db_path=path)) as conn:
        held_now = [
            task_id for task_id in (waiting, scene["refused"])
            if kb.check_respawn_guard(conn, task_id) == "rate_limit_cooldown"
        ]
        retries = conn.execute(
            "SELECT COUNT(*) AS n FROM task_events "
            "WHERE task_id = ? AND kind = 'owner_retry' AND run_id = ?",
            (waiting, held.id),
        ).fetchone()["n"]
    assert held_now == []
    assert retries == (1 if ended_by == "owner_retry" else 0)
    stopped = plain_stopped(snapshot)
    receipts = [run["receipt"] for run in snapshot["runs"]]
    assert sum(state == "provider_wait" for state in stopped.values()) == len(held_now)
    assert sum(receipt["outcome"] == "waiting" for receipt in receipts) == len(held_now)


_SESSION = "placeholder-session-0001"


def _booked_before_the_card_named_its_provider(conn, board: str, project: dict, _monkeypatch):
    """Nothing named a provider when the stop was booked; the card names one afterwards."""
    task_id = kb.create_task(
        conn, title="Draft the placeholder memo", assignee=ASSIGNEE,
        project_id=project["project_id"],
    )
    run = _held(conn, board, task_id, 4601)
    assert run.metadata[kanban_provider_stops.RATE_LIMIT_RESET_PROVIDER_KEY] is None
    assert kb.set_model_override(conn, task_id, "placeholder-model", provider=BUILT_IN)
    return task_id, run


def _booked_on_a_receipt_naming_its_provider(conn, board: str, project: dict, monkeypatch):
    """The card names no provider; the run's own accounting names the one it used.

    The worker links its session the way it does at its first heartbeat; only
    the kernel's receipt source is stubbed, so the real closing path stamps the
    receipt and the real booking keeps the provider's identity.
    """
    task_id = kb.create_task(
        conn, title="Draft the placeholder digest", assignee=ASSIGNEE,
        project_id=project["project_id"],
    )
    _result, spawned = _tick(conn, board, {task_id: 4701})
    assert spawned == [task_id]
    run_id = kb.get_task(conn, task_id).current_run_id
    assert kb.heartbeat_worker(conn, task_id, expected_run_id=run_id, session_id=_SESSION)
    receipt = {
        "schema_version": 3, "engine": "hermes", "profile": ASSIGNEE,
        "provider": BUILT_IN, "model": "placeholder-model",
        "reasoning_effort": "provider-default", "route_evidence": "session-row",
    }
    with monkeypatch.context() as accounting:
        accounting.setattr(
            kb, "_trusted_runtime_receipt",
            lambda session_id, _profile: dict(receipt) if session_id == _SESSION else None,
        )
        kb._record_worker_exit(4701, kb.KANBAN_RATE_LIMIT_EXIT_CODE << 8)
        kb.detect_crashed_workers(conn)
    run = kb.get_run(conn, run_id)
    assert run.outcome == "rate_limited"
    assert kb.get_task(conn, task_id).status == "ready"
    assert kb.get_task(conn, task_id).provider_override is None
    assert run.metadata["runtime_receipt"]["provider"] == BUILT_IN
    assert run.metadata[kanban_provider_stops.RATE_LIMIT_RESET_PROVIDER_KEY] == (
        kanban_provider_stops._identity(BUILT_IN)
    )
    return task_id, run


def _exhausted_entries(paths) -> int:
    """Exhausted entries across ``paths``, counted straight from the files."""
    return sum(
        entry.get("last_status") == "exhausted"
        for path in paths
        for entries in json.loads(path.read_text())["credential_pool"].values()
        for entry in entries
    )


@pytest.mark.parametrize(
    "book",
    [_booked_before_the_card_named_its_provider, _booked_on_a_receipt_naming_its_provider],
    ids=["card_provider", "run_receipt_provider"],
)
def test_the_owner_retry_writes_no_pool_file_whichever_provider_the_card_or_its_run_records(
    book, root, owner, clock, monkeypatch, caplog,
):
    project = _project(owner, "Recorded Route")
    board = project["board"]
    with _board(board) as conn:
        task_id, run = book(conn, board, project, monkeypatch)
        clock["t"] = run.ended_at + 60
        assert kb.check_respawn_guard(conn, task_id) == "rate_limit_cooldown"

    # The assignee's own canonical pool holds the provider's exhausted entry.
    _write_pool(root, {BUILT_IN: [_exhausted("root-a")]})
    _write_pool(root / "profiles" / ASSIGNEE, {
        BUILT_IN: [_exhausted("own-a")],
        SECOND_PROVIDER: [_exhausted("own-b")],
    })
    _write_pool(root / "profiles" / STEWARD, {BUILT_IN: [_exhausted("steward-a")]})
    _write_pool(root / "profiles" / OTHER, {BUILT_IN: [_exhausted("other-a")]})
    before = _exhausted_entries(_stores(root))

    _retried_changing_no_store(root, owner, project, task_id, run, monkeypatch, caplog)

    # The independent count, straight from the files: every exhausted entry still is.
    assert _exhausted_entries(_stores(root)) == before == len(_stores(root)) + 1


def _stores(root: Path) -> list:
    """The root pool and each profile's pool, spelled the way its worker's home names it."""
    return [root / "auth.json"] + [
        root / "profiles" / profile / "auth.json" for profile in (ASSIGNEE, STEWARD, OTHER)
    ]


def _retried_changing_no_store(root, owner, project: dict, task_id: str, run, monkeypatch, caplog):
    """The retry goes ahead under a claimed worker's env; no store changes and no file appears."""
    board = project["board"]
    env = _steward_env(root, board)
    stores = _stores(root)
    folders = (root, root / "profiles" / ASSIGNEE, root / "profiles" / OTHER)
    raw = {path: path.read_bytes() for path in stores}
    names = {folder: set(os.listdir(folder)) for folder in folders}

    caplog.set_level("DEBUG")
    with _as_worker(monkeypatch, env):
        retried = _retry(owner, project, task_id, "retry-changes-no-store")
    assert retried["ok"] is True, retried

    for path in stores:
        assert path.read_bytes() == raw[path], path
    # No lock, no temporary file, no copy: nothing was written beside any pool.
    for folder in folders:
        assert set(os.listdir(folder)) == names[folder], folder

    with _board(board) as conn:
        [event] = _events(conn, task_id, "owner_retry")
        assert retried == {
            "ok": True, "task_id": task_id, "status": "ready",
            "revision": event.id, "retry_reason": REASON,
        }
        assert event.run_id == run.id
        assert kb.get_task(conn, task_id).status == "ready"
        assert kb.check_respawn_guard(conn, task_id) is None
        events = json.dumps([event.payload for event in kb.list_events(conn, task_id)])
    # The independent count, straight from the Project board's own rows.
    path = kb.board_dir(board) / "kanban.db"
    with contextlib.closing(kb.connect(db_path=path)) as conn:
        bound = conn.execute(
            "SELECT COUNT(*) AS n FROM task_events "
            "WHERE task_id = ? AND kind = 'owner_retry' AND run_id = ?",
            (task_id, run.id),
        ).fetchone()["n"]
    assert bound == 1
    spelled = {str(path) for path in stores} | {os.path.realpath(path) for path in stores}
    for leaked in (json.dumps(retried), events, caplog.text):
        for value in ("placeholder-token", *spelled):
            assert value not in leaked


def _alias_the_own_pool(root: Path, alias: str) -> None:
    """Make the assignee's pool path reach a store it does not own alone.

    Every store path the test reads then holds an exhausted built-in entry.
    """
    own_home, other_home = root / "profiles" / ASSIGNEE, root / "profiles" / OTHER
    own_pool, other_pool = own_home / "auth.json", other_home / "auth.json"
    if alias == "another_home_links_to_the_own_home":
        _write_pool(own_home, {BUILT_IN: [_exhausted("own-a")]})
        other_home.rmdir()
        other_home.symlink_to(own_home, target_is_directory=True)
        return
    _write_pool(other_home, {BUILT_IN: [_exhausted("other-a")]})
    if alias == "own_home_links_to_another_home":
        own_home.rmdir()
        own_home.symlink_to(other_home, target_is_directory=True)
    elif alias == "own_pool_links_to_another_pool":
        own_pool.symlink_to(other_pool)
    elif alias == "own_pool_links_to_the_root_pool":
        own_pool.symlink_to(root / "auth.json")
    else:
        os.link(other_pool, own_pool)


@pytest.mark.parametrize("alias", [
    "own_home_links_to_another_home",
    "own_pool_links_to_another_pool",
    "own_pool_links_to_the_root_pool",
    "another_home_links_to_the_own_home",
    "own_pool_hard_links_another_pool",
])
def test_the_retry_goes_ahead_and_changes_no_store_when_the_own_pool_path_aliases_another(
    alias, root, owner, clock, monkeypatch, caplog,
):
    project = _project(owner, "Aliased Pool")
    board = project["board"]
    with _board(board) as conn:
        task_id = _card(conn, project, "Draft the placeholder plan", provider=BUILT_IN)
        run = _held(conn, board, task_id, 5101)
        clock["t"] = run.ended_at + 60
        assert kb.check_respawn_guard(conn, task_id) == "rate_limit_cooldown"

    _write_pool(root, {BUILT_IN: [_exhausted("root-a")]})
    _write_pool(root / "profiles" / STEWARD, {BUILT_IN: [_exhausted("steward-a")]})
    _alias_the_own_pool(root, alias)
    assert _exhausted_entries(_stores(root)) == len(_stores(root))

    _retried_changing_no_store(root, owner, project, task_id, run, monkeypatch, caplog)


@pytest.mark.parametrize("named", [VENDOR, VENDOR_POOL], ids=["custom_name", "custom_pool_key"])
def test_the_retry_goes_ahead_and_changes_no_store_for_a_custom_provider(
    named, root, owner, clock, monkeypatch, caplog,
):
    project = _project(owner, "Vendor Route")
    board = project["board"]
    with _board(board) as conn:
        task_id = _card(conn, project, "Draft the placeholder quote", provider=named)
        run = _held(conn, board, task_id, 5201)
        clock["t"] = run.ended_at + 60
        assert kb.check_respawn_guard(conn, task_id) == "rate_limit_cooldown"

    # The assignee's own config names the vendor, so its worker draws from the
    # legacy pool keyed ``custom:<name>``; that pool is the assignee's own.
    config = root / "profiles" / ASSIGNEE / "config.yaml"
    config.write_text(f"custom_providers:\n  - name: {VENDOR}\n    base_url: {VENDOR_URL}\n")
    [(name, entry)] = credential_pool._iter_custom_providers(yaml.safe_load(config.read_text()))
    assert credential_pool._pool_keys_for_custom_entry(name, entry) == [VENDOR_POOL]
    for path, label in zip(_stores(root), ("root", "own", "steward", "other")):
        _write_pool(path.parent, {
            VENDOR: [_exhausted(f"{label}-v")],
            VENDOR_POOL: [_exhausted(f"{label}-c")],
            BUILT_IN: [_exhausted(f"{label}-a")],
        })

    _retried_changing_no_store(root, owner, project, task_id, run, monkeypatch, caplog)


_FILE_SYNC, _FOLDER_SYNC = "file", "folder"


@contextlib.contextmanager
def _failing_sync(monkeypatch, pool: Path, fails):
    """``os.fsync`` raises EIO for one kind of descriptor of ``pool``'s own save only.

    ``os.open`` notes each descriptor opened for ``pool``'s temporary file or
    for its folder with its ``os.fstat`` identity, so a reused number never
    matches; only a noted descriptor of the kind ``fails`` names raises, and
    every other sync, the board's included, is forwarded unchanged. Yields the
    kinds synced for that save, in order.
    """
    real_open, real_fsync = os.open, os.fsync
    folder = os.path.realpath(pool.parent)
    noted, synced = {}, []

    def identity(fd: int) -> tuple:
        status = os.fstat(fd)
        return fd, status.st_dev, status.st_ino

    def opening(path, flags, *args, **kwargs):
        fd = real_open(path, flags, *args, **kwargs)
        name = os.fsdecode(path)
        if os.path.realpath(name) == folder:
            noted[identity(fd)] = _FOLDER_SYNC
        elif (
            os.path.realpath(os.path.dirname(name)) == folder
            and os.path.basename(name).startswith(pool.name + ".tmp.")
        ):
            noted[identity(fd)] = _FILE_SYNC
        return fd

    def syncing(fd):
        kind = noted.get(identity(fd if isinstance(fd, int) else fd.fileno()))
        if kind is not None:
            synced.append(kind)
            if kind == fails:
                raise OSError(errno.EIO, os.strerror(errno.EIO))
        return real_fsync(fd)

    with monkeypatch.context() as patch:
        patch.setattr(os, "open", opening)
        patch.setattr(os, "fsync", syncing)
        yield synced


# The card's own worker, at its next start.

ROUTE_URL = "https://deepseek.placeholder.invalid/v1"
DESK = "steward-desk"


def _routed(entry_id: str) -> dict:
    """An exhausted entry that names its own endpoint, so once clear it is drawn as it is."""
    return dict(_exhausted(entry_id), base_url=ROUTE_URL)


def _worker_pools(root: Path) -> dict:
    """Every store holds an exhausted entry for the built-in provider; the own one another provider's too."""
    return {
        "root": _write_pool(root, {BUILT_IN: [_routed("root-a")]}),
        "own": _write_pool(root / "profiles" / ASSIGNEE, {
            BUILT_IN: [_routed("own-a")], SECOND_PROVIDER: [_routed("own-b")],
        }),
        "steward": _write_pool(root / "profiles" / STEWARD, {BUILT_IN: [_routed("steward-a")]}),
        "other": _write_pool(root / "profiles" / OTHER, {BUILT_IN: [_routed("other-a")]}),
    }


def _marks(entry: dict) -> dict:
    return {key: entry.get(key) for key in _STATUS_KEYS}


def _scene(root: Path, owner: ow.OwnerContext, clock, name: str, pid: int, *, provider: str = BUILT_IN) -> dict:
    """A card held at the provider's limit, and a card of the steward's on a board of its own."""
    project = _project(owner, name)
    board = project["board"]
    with _board(board) as conn:
        task_id = _card(conn, project, "Draft the placeholder outline", provider=provider)
        run = _held(conn, board, task_id, pid)
        clock["t"] = run.ended_at + 60
    kb.create_board(DESK)
    with _board(DESK) as conn:
        desk_task = kb.create_task(conn, title="Steward the placeholder desk", assignee=STEWARD)
    return {
        "project": project, "board": board, "task_id": task_id, "run": run,
        "boards": {board: task_id, DESK: desk_task},
    }


def _owner_retried(
    root: Path, owner: ow.OwnerContext, clock, monkeypatch, name: str, pid: int, *, provider: str = BUILT_IN,
) -> dict:
    scene = _scene(root, owner, clock, name, pid, provider=provider)
    with _as_worker(monkeypatch, _steward_env(root, scene["board"])):
        retried = _retry(owner, scene["project"], scene["task_id"], f"retry-{pid}")
    assert retried["ok"] is True, retried
    return scene


def _worker_start(root: Path, board: str, task_id: str, pid: int) -> tuple:
    """The dispatcher's next tick spawns the card: exactly the argv and env it builds for its worker."""
    launched = []

    def fake_popen(cmd, *_args, **kwargs):
        launched.append((list(cmd), dict(kwargs["env"])))
        return SimpleNamespace(pid=pid)

    with _board(board) as conn, pytest.MonkeyPatch.context() as dispatcher:
        dispatcher.setattr(kb, "_resolve_hermes_argv", lambda: ["hermes"])
        dispatcher.setattr(subprocess, "Popen", fake_popen)
        kb.dispatch_once(conn, board=board)
        run_id = kb.get_task(conn, task_id).current_run_id
    [(argv, env)] = launched
    assert env["HERMES_HOME"] == str(root / "profiles" / ASSIGNEE)
    assert env["HERMES_KANBAN_TASK"] == task_id
    assert env["HERMES_KANBAN_RUN_ID"] == str(run_id)
    assert env["HERMES_KANBAN_DB"] == str(kb.kanban_db_path(board=board))
    assert env["HERMES_KANBAN_BOARD"] == board
    return argv, env


def _reaches_every_board_and_the_root(root: Path, boards: dict) -> None:
    """Pinned to the card's board, the worker still reaches every other board and the root registry."""
    assert kb.kanban_home() == root
    assert kb.register_db_path() == root / "kanban" / "board_register.db"
    for board, task_id in boards.items():
        entry = kb.get_register_entry(board)
        assert entry is not None and entry.lifecycle is kb.BoardLifecycle.LIVE
        with contextlib.closing(kb.connect(db_path=kb.board_dir(board) / "kanban.db")) as conn:
            assert kb.get_task(conn, task_id) is not None


class _Worker(CLIAgentSetupMixin):
    """The card's own worker, as the real ``-q`` entry drives it.

    Credentials come from the real runtime resolution, for the provider and
    model the dispatcher put on the worker's command line. Only the agent and
    its model request are stubbed; ``chat`` resolves credentials before it
    builds the agent, as the real chat turn does.
    """

    def __init__(self, argv: list):
        self.requested_provider = argv[argv.index("--provider") + 1]
        self.model = argv[argv.index("-m") + 1]
        self._explicit_api_key = self._explicit_base_url = None
        self.provider, self.api_mode, self.acp_command, self.acp_args = None, "chat_completions", None, []
        self.api_key = self.base_url = self._credential_pool = self._provider_source = None
        self._fallback_model = []
        self.agent = None
        self._active_agent_route_signature = "route"
        self.session_id = "placeholder-session-0002"
        self.conversation_history = []
        self.console = SimpleNamespace(print=lambda *_args, **_kwargs: None)
        self.resolved, self.requests = [], []

    def _ensure_runtime_credentials(self):
        # The own pool's marks for the provider, straight from the file, as the resolution starts.
        store = json.loads((Path(os.environ["HERMES_HOME"]) / "auth.json").read_text())
        self.resolved.append({
            entry["id"]: _marks(entry) for entry in store["credential_pool"][self.requested_provider]
        })
        return super()._ensure_runtime_credentials()

    def _maybe_print_free_tier_available_notice(self):
        pass

    def _normalize_model_for_provider(self, _provider):
        return False

    def _claim_active_session(self, _surface, *, stderr=False):
        return True

    def _show_security_advisories(self):
        pass

    def _print_exit_summary(self, clear_screen=True):
        pass

    def _resolve_turn_agent_config(self, _query):
        return {"signature": "route", "model": None, "runtime": None}

    def _init_agent(self, **_kwargs):
        self.agent = SimpleNamespace(session_id=self.session_id, run_conversation=self._request)
        return True

    def _request(self, **_kwargs):
        """The model request, stubbed: it notes the credential it would send."""
        self.requests.append((self.api_key, self.base_url, getattr(self._credential_pool, "provider", None)))
        return {"final_response": "done", "completed": True}

    def chat(self, query, images=None):
        if not self._ensure_runtime_credentials():
            return None
        route = self._resolve_turn_agent_config(query)
        if not self._init_agent(model_override=route["model"], runtime_override=route["runtime"]):
            return None
        self._last_turn_result = self.agent.run_conversation(
            user_message=query, conversation_history=self.conversation_history,
        )
        return self._last_turn_result["final_response"]


@pytest.fixture
def one_shot(monkeypatch):
    """The real ``-q`` entry, with its image and session plumbing out of the way."""
    # The entry sets this marker on os.environ itself; registering it here undoes that.
    monkeypatch.setenv("HERMES_SINGLE_QUERY_SESSION", "1")
    monkeypatch.setattr(cli, "_should_seed_interactive", lambda *_args: False)
    monkeypatch.setattr(cli, "_collect_query_images", lambda query, _image: (query, []))
    monkeypatch.setattr(cli, "_collect_kanban_task_images", lambda _images: [])
    monkeypatch.setattr(cli, "_finalize_single_query", lambda _worker: None)

    def start(worker: _Worker, argv: list, *, quiet: bool = False) -> None:
        """The worker runs the prompt the dispatcher gave it, and exits 0."""
        query = argv[argv.index("-q") + 1]
        if not quiet:
            cli._run_single_query_mode(worker, query, None, False, False)
            return
        with pytest.raises(SystemExit) as exited:
            cli._run_single_query_mode(worker, query, None, True, False)
        assert exited.value.code == 0

    return start


def _board_rows(board: str, task_id: str, run_id: int) -> tuple:
    """The independent count, straight from the card's own board rows.

    The latest ended run's outcome, and the owner retries bound to ``run_id``.
    """
    with contextlib.closing(kb.connect(db_path=kb.board_dir(board) / "kanban.db")) as conn:
        latest = conn.execute(
            "SELECT id, outcome FROM task_runs WHERE task_id = ? AND ended_at IS NOT NULL "
            "ORDER BY ended_at DESC, id DESC LIMIT 1",
            (task_id,),
        ).fetchone()
        bound = conn.execute(
            "SELECT COUNT(*) AS n FROM task_events "
            "WHERE task_id = ? AND kind = 'owner_retry' AND run_id = ?",
            (task_id, run_id),
        ).fetchone()["n"]
    return (latest["id"], latest["outcome"]), bound


def _names_only_the_provider(record: logging.LogRecord, root: Path, task_id: str, provider: str = BUILT_IN) -> None:
    assert provider in record.getMessage()
    assert record.exc_info is None
    shown = logging.Formatter().format(record)
    for value in (
        str(root), os.path.realpath(root), "auth.json", ROUTE_URL, VENDOR_URL, ASSIGNEE, STEWARD, OTHER,
        task_id, os.strerror(errno.EIO), "Errno", "placeholder-token", "Traceback",
    ):
        assert value not in shown, value


def _cli_lines(caplog) -> list:
    return [record for record in caplog.records if record.name == "cli" and record.levelno >= logging.INFO]


# The only provider whose limit a worker start resets, and its default endpoint.
ANTHROPIC = "anthropic"
ANTHROPIC_URL = PROVIDER_REGISTRY[ANTHROPIC].inference_base_url
PROXY_URL = "https://proxy.placeholder.invalid/v1"


def _spare(entry_id: str) -> dict:
    """An entry that holds no limit, drawn only after the provider's other entries."""
    entry = {key: value for key, value in _exhausted(entry_id).items() if key not in (*_STATUS_KEYS, "failure_reason")}
    return dict(entry, priority=1)


def _anthropic_pools(root: Path) -> dict:
    """Every store holds an exhausted anthropic entry; the own one a spare after it, and another provider's."""
    return {
        "root": _write_pool(root, {ANTHROPIC: [_exhausted("root-s")]}),
        "own": _write_pool(root / "profiles" / ASSIGNEE, {
            ANTHROPIC: [_exhausted("own-s"), _spare("own-t")], SECOND_PROVIDER: [_routed("own-b")],
        }),
        "steward": _write_pool(root / "profiles" / STEWARD, {ANTHROPIC: [_exhausted("steward-s")]}),
        "other": _write_pool(root / "profiles" / OTHER, {ANTHROPIC: [_exhausted("other-s")]}),
    }


@pytest.mark.parametrize("quiet", [False, True], ids=["one_shot", "quiet_one_shot"])
def test_after_an_owner_retry_the_worker_clears_its_own_provider_limit_before_its_first_request(
    quiet, root, owner, clock, monkeypatch, caplog, one_shot,
):
    scene = _owner_retried(root, owner, clock, monkeypatch, "Worker Start", 5401, provider=ANTHROPIC)
    board, task_id, run = scene["board"], scene["task_id"], scene["run"]
    pools = _anthropic_pools(root)
    own_pool = pools["own"]
    untouched = {path: path.read_bytes() for key, path in pools.items() if key != "own"}
    before = _exhausted_entries(pools.values())

    argv, env = _worker_start(root, board, task_id, 5402)
    worker = _Worker(argv)
    caplog.set_level("DEBUG")
    with _as_worker(monkeypatch, env):
        _reaches_every_board_and_the_root(root, scene["boards"])
        one_shot(worker, argv, quiet=quiet)

    # When the worker first resolves its credentials, its own exhausted entry is already clear,
    # so the request draws that entry from its own pool, at the provider's default endpoint.
    assert worker.resolved == [{"own-s": dict.fromkeys(_STATUS_KEYS), "own-t": dict.fromkeys(_STATUS_KEYS)}]
    assert worker.requests == [("placeholder-token-own-s", ANTHROPIC_URL, ANTHROPIC)]

    for path, raw in untouched.items():
        assert path.read_bytes() == raw, path
    stored = json.loads(own_pool.read_text())["credential_pool"]
    assert [(entry["id"], _marks(entry)) for entry in stored[ANTHROPIC]] == [
        ("own-s", dict.fromkeys(_STATUS_KEYS)), ("own-t", dict.fromkeys(_STATUS_KEYS)),
    ]
    assert stored[SECOND_PROVIDER] == [_routed("own-b")]
    # The independent counts, straight from the files and from the card's own board rows.
    assert _exhausted_entries(pools.values()) == before - 1 == len(pools)
    assert _board_rows(board, task_id, run.id) == ((run.id, "rate_limited"), 1)

    [line] = _cli_lines(caplog)
    assert line.levelno == logging.INFO
    _names_only_the_provider(line, root, task_id, ANTHROPIC)


@pytest.mark.parametrize("retried", ["no_owner_retry", "owner_retry_bound_to_an_earlier_run"])
def test_the_worker_start_clears_nothing_without_an_owner_retry_bound_to_the_latest_ended_run(
    retried, root, owner, clock, monkeypatch, caplog,
):
    scene = _scene(root, owner, clock, "Plain Resume", 5501)
    project, board, task_id, run = scene["project"], scene["board"], scene["task_id"], scene["run"]
    if retried == "owner_retry_bound_to_an_earlier_run":
        with _as_worker(monkeypatch, _steward_env(root, board)):
            assert _retry(owner, project, task_id, "retry-earlier-run")["ok"] is True
        with _board(board) as conn:
            clock["t"] += 30
            run = _held(conn, board, task_id, 5502)
    with _board(board) as conn:
        # The hold ends by itself; the latest ended run is still the rate-limited one.
        clock["t"] = _resume_at(run)
        assert kb.check_respawn_guard(conn, task_id) is None

    pools = _worker_pools(root)
    argv, env = _worker_start(root, board, task_id, 5503)
    raw = {path: path.read_bytes() for path in pools.values()}
    names = {path.parent: set(os.listdir(path.parent)) for path in pools.values()}
    caplog.set_level("DEBUG")
    with _as_worker(monkeypatch, env):
        _reaches_every_board_and_the_root(root, scene["boards"])
        cli._clear_owner_retried_provider_limit(_Worker(argv))

    for path, before in raw.items():
        assert path.read_bytes() == before, path
    for folder, before in names.items():
        assert set(os.listdir(folder)) == before, folder
    # The independent counts, straight from the files and from the card's own board rows.
    assert _exhausted_entries(pools.values()) == len(pools) + 1
    assert _board_rows(board, task_id, run.id) == ((run.id, "rate_limited"), 0)
    assert _cli_lines(caplog) == []


def test_the_worker_start_clears_nothing_when_the_latest_ended_run_did_not_end_rate_limited(
    root, owner, clock, monkeypatch, caplog,
):
    scene = _owner_retried(root, owner, clock, monkeypatch, "Crashed Resume", 5601)
    board, task_id, held = scene["board"], scene["task_id"], scene["run"]
    with _board(board) as conn:
        clock["t"] += 30
        crashed = _reaped(conn, board, task_id, 5602, 1)
        assert crashed.outcome != "rate_limited"
        assert kb.get_task(conn, task_id).status == "ready"
        clock["t"] = crashed.ended_at + 3_600
        assert kb.check_respawn_guard(conn, task_id) is None

    pools = _worker_pools(root)
    argv, env = _worker_start(root, board, task_id, 5603)
    raw = {path: path.read_bytes() for path in pools.values()}
    names = {path.parent: set(os.listdir(path.parent)) for path in pools.values()}
    caplog.set_level("DEBUG")
    with _as_worker(monkeypatch, env):
        _reaches_every_board_and_the_root(root, scene["boards"])
        cli._clear_owner_retried_provider_limit(_Worker(argv))

    for path, before in raw.items():
        assert path.read_bytes() == before, path
    for folder, before in names.items():
        assert set(os.listdir(folder)) == before, folder
    # The independent counts: the owner's retry is bound to the held run, which no longer ended last.
    assert _exhausted_entries(pools.values()) == len(pools) + 1
    assert _board_rows(board, task_id, held.id) == ((crashed.id, crashed.outcome), 1)
    assert _cli_lines(caplog) == []


def test_a_failed_clear_logs_one_provider_only_warning_and_the_worker_start_goes_on(
    root, owner, clock, monkeypatch, caplog, one_shot,
):
    scene = _owner_retried(root, owner, clock, monkeypatch, "Failed Clear", 5701, provider=ANTHROPIC)
    board, task_id, run = scene["board"], scene["task_id"], scene["run"]
    pools = _anthropic_pools(root)
    own_pool = pools["own"]
    raw = {path: path.read_bytes() for path in pools.values()}

    argv, env = _worker_start(root, board, task_id, 5702)
    names = set(os.listdir(own_pool.parent))
    worker = _Worker(argv)
    caplog.set_level("DEBUG")
    # Only the worker's own start counts: the dispatcher's spawn above logged in this process too.
    caplog.clear()
    with _as_worker(monkeypatch, env), _failing_sync(monkeypatch, own_pool, _FILE_SYNC) as synced:
        _reaches_every_board_and_the_root(root, scene["boards"])
        one_shot(worker, argv)

    # The clear's save failed before it replaced the store: every store is as it was.
    assert synced == [_FILE_SYNC]
    for path, before in raw.items():
        assert path.read_bytes() == before, path
    # No temporary file remains: only the lock the save ran under may appear, and the note
    # loading the provider's pool keeps of its heal check, as every resolution leaves it.
    assert set(os.listdir(own_pool.parent)) - names <= {own_pool.with_suffix(".lock").name, "cache"}
    assert set(os.listdir(own_pool.parent / "cache")) == {"oauth_heal_clean.json"}
    # The independent counts, straight from the files and from the card's own board rows.
    assert _exhausted_entries(pools.values()) == len(pools) + 1
    assert _board_rows(board, task_id, run.id) == ((run.id, "rate_limited"), 1)

    # The start went on: the worker resolved its credentials and made its request.
    assert worker.resolved == [{"own-s": _marks(_exhausted("own-s")), "own-t": dict.fromkeys(_STATUS_KEYS)}]
    assert len(worker.requests) == 1
    [warning] = [record for record in caplog.records if record.levelno >= logging.WARNING]
    assert warning.name == "cli"
    _names_only_the_provider(warning, root, task_id, ANTHROPIC)


NOTHING_CLEARED_LINE = "nothing was cleared for the {provider} provider after an owner retry"


def _start_clears_nothing(root: Path, scene: dict, argv: list, env: dict, monkeypatch, caplog, provider: str) -> None:
    """The worker starts under ``env``: no store changes, no file appears, one line says nothing was cleared."""
    stores = _stores(root)
    folders = [root] + [root / "profiles" / profile for profile in (ASSIGNEE, STEWARD, OTHER)]
    caplog.set_level("DEBUG")
    with _as_worker(monkeypatch, env):
        # Any process sets its own home up long before this check: only what the check adds counts.
        ensure_hermes_home()
        raw = {path: path.read_bytes() for path in stores}
        names = {folder: set(os.listdir(folder)) for folder in folders}
        before = _exhausted_entries(stores)
        _reaches_every_board_and_the_root(root, scene["boards"])
        cli._clear_owner_retried_provider_limit(_Worker(argv))

    for path in stores:
        assert path.read_bytes() == raw[path], path
    for folder in folders:
        assert set(os.listdir(folder)) == names[folder], folder
    # The independent counts, straight from the files and from the card's own board rows.
    assert _exhausted_entries(stores) == before
    run = scene["run"]
    assert _board_rows(scene["board"], scene["task_id"], run.id) == ((run.id, "rate_limited"), 1)
    [line] = _cli_lines(caplog)
    assert (line.levelno, line.getMessage()) == (logging.INFO, NOTHING_CLEARED_LINE.format(provider=provider))
    _names_only_the_provider(line, root, scene["task_id"], provider)


@pytest.mark.parametrize("provider", ["anthropic", "openai-codex", "xai-oauth"])
def test_the_worker_start_leaves_root_rows_its_profile_reads_byte_for_byte(
    provider, root, owner, clock, monkeypatch, caplog,
):
    scene = _owner_retried(root, owner, clock, monkeypatch, "Borrowed Rows", 5801, provider=provider)
    # The worker's own store holds no row for the provider, so the worker reads the root's.
    _write_pool(root, {provider: [_exhausted("root-s")]})
    _write_pool(root / "profiles" / ASSIGNEE, {BUILT_IN: [_routed("own-a")]})
    _write_pool(root / "profiles" / STEWARD, {provider: [_exhausted("steward-s")]})
    _write_pool(root / "profiles" / OTHER, {provider: [_exhausted("other-s")]})
    assert _exhausted_entries(_stores(root)) == len(_stores(root))

    argv, env = _worker_start(root, scene["board"], scene["task_id"], 5802)
    _start_clears_nothing(root, scene, argv, env, monkeypatch, caplog, provider)


_VENDOR_CONFIGS = {
    "custom_providers": f"custom_providers:\n  - name: {VENDOR}\n    base_url: {VENDOR_URL}\n",
    "providers": f"providers:\n  {VENDOR}:\n    base_url: {VENDOR_URL}\n",
}


@pytest.mark.parametrize("config", sorted(_VENDOR_CONFIGS))
def test_the_worker_start_resets_nothing_for_a_custom_provider(config, root, owner, clock, monkeypatch, caplog):
    scene = _owner_retried(root, owner, clock, monkeypatch, "Vendor Start", 5811, provider=VENDOR)
    # The worker's own config names the vendor; every store holds its exhausted rows under both pool keys.
    (root / "profiles" / ASSIGNEE / "config.yaml").write_text(_VENDOR_CONFIGS[config])
    for path, label in zip(_stores(root), ("root", "own", "steward", "other")):
        _write_pool(path.parent, {VENDOR: [_exhausted(f"{label}-v")], VENDOR_POOL: [_exhausted(f"{label}-c")]})

    argv, env = _worker_start(root, scene["board"], scene["task_id"], 5812)
    _start_clears_nothing(root, scene, argv, env, monkeypatch, caplog, VENDOR)


def test_a_failed_task_read_logs_one_provider_only_warning_and_the_worker_start_goes_on(
    root, owner, clock, monkeypatch, caplog, one_shot,
):
    scene = _owner_retried(root, owner, clock, monkeypatch, "Unread Task", 5821)
    board, task_id, run = scene["board"], scene["task_id"], scene["run"]
    pools = _worker_pools(root)
    detail = f"placeholder private detail under {root}"

    def unreadable(*_args, **_kwargs):
        raise OSError(errno.EIO, detail)

    argv, env = _worker_start(root, board, task_id, 5822)
    raw = {path: path.read_bytes() for path in pools.values()}
    names = {path.parent: set(os.listdir(path.parent)) for path in pools.values()}
    worker = _Worker(argv)
    caplog.set_level("DEBUG")
    # Only the worker's own start counts: the dispatcher's spawn above logged in this process too.
    caplog.clear()
    offline = models_dev.get_provider_info
    with _as_worker(monkeypatch, env), monkeypatch.context() as reading:
        reading.setattr(kb, "owner_retried_provider_limit", unreadable)
        # With its own entry still exhausted, the worker's resolution looks its provider up: offline only.
        reading.setattr(models_dev, "get_provider_info", lambda name, **_kwargs: offline(name, allow_network=False))
        _reaches_every_board_and_the_root(root, scene["boards"])
        one_shot(worker, argv)

    for path, before in raw.items():
        assert path.read_bytes() == before, path
    for folder, before in names.items():
        assert set(os.listdir(folder)) == before, folder
    # The independent counts, straight from the files and from the card's own board rows.
    assert _exhausted_entries(pools.values()) == len(pools) + 1
    assert _board_rows(board, task_id, run.id) == ((run.id, "rate_limited"), 1)

    # The start went on: the worker resolved its credentials and made its request.
    assert worker.resolved == [{"own-a": _marks(_routed("own-a"))}]
    assert len(worker.requests) == 1
    [warning] = [record for record in caplog.records if record.levelno >= logging.WARNING]
    assert warning.name == "cli"
    _names_only_the_provider(warning, root, task_id)
    # No traceback and no exception text, at any level.
    assert [record for record in caplog.records if record.name == "cli" and record.exc_info] == []
    assert detail not in caplog.text


@pytest.mark.parametrize("home", [OTHER, "root"], ids=["another_profile", "the_root"])
def test_the_worker_start_resets_nothing_under_the_home_of_another_profile(
    home, root, owner, clock, monkeypatch, caplog,
):
    scene = _owner_retried(root, owner, clock, monkeypatch, "Other Home", 5831)
    _worker_pools(root)
    argv, env = _worker_start(root, scene["board"], scene["task_id"], 5832)
    # Only the profile home differs from what the dispatcher built.
    env["HERMES_HOME"] = str(root if home == "root" else root / "profiles" / home)
    _start_clears_nothing(root, scene, argv, env, monkeypatch, caplog, BUILT_IN)


@pytest.mark.parametrize("run_id", ["held_run", "unknown_run"])
def test_the_worker_start_resets_nothing_for_a_run_that_is_not_the_task_current_running_run(
    run_id, root, owner, clock, monkeypatch, caplog,
):
    scene = _owner_retried(root, owner, clock, monkeypatch, "Stale Run", 5841)
    _worker_pools(root)
    argv, env = _worker_start(root, scene["board"], scene["task_id"], 5842)
    # Only the run id differs: the held run, which has ended, or one the board never had.
    current = int(env["HERMES_KANBAN_RUN_ID"])
    env["HERMES_KANBAN_RUN_ID"] = str(scene["run"].id if run_id == "held_run" else current + 1)
    _start_clears_nothing(root, scene, argv, env, monkeypatch, caplog, BUILT_IN)


def test_the_worker_start_says_nothing_was_cleared_when_its_own_rows_hold_no_limit(
    root, owner, clock, monkeypatch, caplog,
):
    scene = _owner_retried(root, owner, clock, monkeypatch, "Clear Rows", 5851)
    pools = _worker_pools(root)
    # The worker's own rows for the provider carry no limit; every other store is as before.
    clear = {key: value for key, value in _routed("own-a").items() if key not in (*_STATUS_KEYS, "failure_reason")}
    _write_pool(pools["own"].parent, {BUILT_IN: [clear], SECOND_PROVIDER: [_routed("own-b")]})

    argv, env = _worker_start(root, scene["board"], scene["task_id"], 5852)
    _start_clears_nothing(root, scene, argv, env, monkeypatch, caplog, BUILT_IN)


@pytest.mark.parametrize("provider", [BUILT_IN, SECOND_PROVIDER])
def test_the_worker_start_resets_nothing_for_a_provider_other_than_anthropic(
    provider, root, owner, clock, monkeypatch, caplog,
):
    scene = _owner_retried(root, owner, clock, monkeypatch, "Other Provider", 5861, provider=provider)
    # Every store, the worker's own included, holds the provider's exhausted rows at its default endpoint.
    for path, label in zip(_stores(root), ("root", "own", "steward", "other")):
        _write_pool(path.parent, {provider: [_exhausted(f"{label}-a")]})

    argv, env = _worker_start(root, scene["board"], scene["task_id"], 5862)
    _start_clears_nothing(root, scene, argv, env, monkeypatch, caplog, provider)


CLEARED_LINE = "cleared the {provider} provider limit after an owner retry"


def _limited_anthropic_pools(root: Path) -> None:
    """Every own anthropic row at its limit, beside another provider's; no other store holds an anthropic row."""
    _write_pool(root, {BUILT_IN: [_exhausted("root-a")]})
    _write_pool(root / "profiles" / ASSIGNEE, {
        ANTHROPIC: [_exhausted("own-s"), dict(_exhausted("own-t"), priority=1)],
        SECOND_PROVIDER: [_routed("own-b")],
    })
    for profile in (STEWARD, OTHER):
        _write_pool(root / "profiles" / profile, {BUILT_IN: [_exhausted(f"{profile}-a")]})


def _start_resets_the_own_rows(root: Path, scene: dict, argv: list, env: dict, monkeypatch, caplog) -> None:
    """The worker starts under ``env``: its own anthropic rows are reset, no other store changes, one line says so."""
    stores = _stores(root)
    own_pool = root / "profiles" / ASSIGNEE / "auth.json"
    others = [path for path in stores if path != own_pool]
    caplog.set_level("DEBUG")
    with _as_worker(monkeypatch, env):
        # Any process sets its own home up long before this check: only what the check adds counts.
        ensure_hermes_home()
        raw = {path: path.read_bytes() for path in others}
        names = {path.parent: set(os.listdir(path.parent)) for path in others}
        before = _exhausted_entries(stores)
        _reaches_every_board_and_the_root(root, scene["boards"])
        cli._clear_owner_retried_provider_limit(_Worker(argv))

    for path in others:
        assert path.read_bytes() == raw[path], path
    for folder, listed in names.items():
        # Nothing appears or goes beside another store, but the root store's lock, which any load
        # of the anthropic pool may take for its heal check.
        assert listed <= set(os.listdir(folder)) <= listed | ({"auth.lock"} if folder == root else set()), folder
    stored = json.loads(own_pool.read_text())["credential_pool"]
    assert [(entry["id"], _marks(entry)) for entry in stored[ANTHROPIC]] == [
        ("own-s", dict.fromkeys(_STATUS_KEYS)), ("own-t", dict.fromkeys(_STATUS_KEYS)),
    ]
    assert stored[SECOND_PROVIDER] == [_routed("own-b")]
    # The independent counts, straight from the files and from the card's own board rows.
    assert _exhausted_entries(stores) == before - 2 == len(stores)
    run = scene["run"]
    assert _board_rows(scene["board"], scene["task_id"], run.id) == ((run.id, "rate_limited"), 1)
    [line] = _cli_lines(caplog)
    assert (line.levelno, line.getMessage()) == (logging.INFO, CLEARED_LINE.format(provider=ANTHROPIC))
    _names_only_the_provider(line, root, scene["task_id"], ANTHROPIC)


def test_the_worker_start_resets_its_own_anthropic_rows_when_every_one_is_at_its_limit(
    root, owner, clock, monkeypatch, caplog,
):
    scene = _owner_retried(root, owner, clock, monkeypatch, "Limited Rows", 5866, provider=ANTHROPIC)
    _limited_anthropic_pools(root)
    argv, env = _worker_start(root, scene["board"], scene["task_id"], 5867)
    _start_resets_the_own_rows(root, scene, argv, env, monkeypatch, caplog)


def test_a_base_url_in_the_environment_does_not_change_the_reset(root, owner, clock, monkeypatch, caplog):
    scene = _owner_retried(root, owner, clock, monkeypatch, "Env Endpoint", 5871, provider=ANTHROPIC)
    _limited_anthropic_pools(root)
    # The dispatcher's own environment names an endpoint, so the worker's carries it too.
    monkeypatch.setenv(PROVIDER_REGISTRY[ANTHROPIC].base_url_env_var, PROXY_URL)
    argv, env = _worker_start(root, scene["board"], scene["task_id"], 5872)
    assert env[PROVIDER_REGISTRY[ANTHROPIC].base_url_env_var] == PROXY_URL
    _start_resets_the_own_rows(root, scene, argv, env, monkeypatch, caplog)
    assert PROXY_URL not in caplog.text


# One endpoint the runtime resolution would send the request to, and one it would pass over.
_CONFIG_URLS = {
    "honoured_by_the_resolution": "https://proxy.placeholder.invalid/anthropic",
    "ignored_by_the_resolution": PROXY_URL,
}


@pytest.mark.parametrize("url", sorted(_CONFIG_URLS))
def test_a_base_url_in_config_does_not_change_the_reset(url, root, owner, clock, monkeypatch, caplog):
    scene = _owner_retried(root, owner, clock, monkeypatch, "Config Endpoint", 5881, provider=ANTHROPIC)
    _limited_anthropic_pools(root)
    (root / "profiles" / ASSIGNEE / "config.yaml").write_text(
        f"model:\n  provider: {ANTHROPIC}\n  base_url: {_CONFIG_URLS[url]}\n"
    )
    argv, env = _worker_start(root, scene["board"], scene["task_id"], 5882)
    _start_resets_the_own_rows(root, scene, argv, env, monkeypatch, caplog)
    assert _CONFIG_URLS[url] not in caplog.text


def test_an_unreadable_own_store_logs_one_provider_only_warning_and_writes_nothing(
    root, owner, clock, monkeypatch, caplog,
):
    scene = _owner_retried(root, owner, clock, monkeypatch, "Unread Store", 5891, provider=ANTHROPIC)
    own_pool = _anthropic_pools(root)["own"]
    detail = f"placeholder private detail under {root}"
    read_text = Path.read_text

    def unreadable(path, *args, **kwargs):
        # Only the worker's own store: every other file reads as it is.
        if os.path.realpath(path) == os.path.realpath(own_pool):
            raise OSError(errno.EIO, detail)
        return read_text(path, *args, **kwargs)

    argv, env = _worker_start(root, scene["board"], scene["task_id"], 5892)
    stores = _stores(root)
    folders = [root] + [root / "profiles" / profile for profile in (ASSIGNEE, STEWARD, OTHER)]
    caplog.set_level("DEBUG")
    with _as_worker(monkeypatch, env):
        # Any process sets its own home up long before this check: only what the check adds counts.
        ensure_hermes_home()
        raw = {path: path.read_bytes() for path in stores}
        names = {folder: set(os.listdir(folder)) for folder in folders}
        # Only the worker's own start counts: the dispatcher's spawn above logged in this process too.
        caplog.clear()
        with monkeypatch.context() as reading:
            reading.setattr(Path, "read_text", unreadable)
            cli._clear_owner_retried_provider_limit(_Worker(argv))

    for path in stores:
        assert path.read_bytes() == raw[path], path
    for folder in folders:
        assert set(os.listdir(folder)) == names[folder], folder
    run = scene["run"]
    assert _board_rows(scene["board"], scene["task_id"], run.id) == ((run.id, "rate_limited"), 1)

    _one_provider_only_warning(caplog, root, scene["task_id"])


def _one_provider_only_warning(caplog, root: Path, task_id: str) -> None:
    """The check reads the store silently: the helper's one warning, naming only the provider, is the
    only record at that level from any logger, and no record carries a traceback."""
    [warning] = [record for record in caplog.records if record.levelno >= logging.WARNING]
    assert warning.name == "cli"
    _names_only_the_provider(warning, root, task_id, ANTHROPIC)
    assert [record for record in caplog.records if record.exc_info] == []


def test_an_own_store_that_does_not_parse_logs_one_provider_only_warning_and_writes_nothing(
    root, owner, clock, monkeypatch, caplog,
):
    scene = _owner_retried(root, owner, clock, monkeypatch, "Torn Store", 5895, provider=ANTHROPIC)
    own_pool = _anthropic_pools(root)["own"]
    argv, env = _worker_start(root, scene["board"], scene["task_id"], 5896)
    stores = _stores(root)
    folders = [root] + [root / "profiles" / profile for profile in (ASSIGNEE, STEWARD, OTHER)]
    caplog.set_level("DEBUG")
    with _as_worker(monkeypatch, env):
        # Any process sets its own home up long before this check: only what the check adds counts.
        ensure_hermes_home()
        own_pool.write_bytes(b'{"credential_pool": {"anthropic": [')
        raw = {path: path.read_bytes() for path in stores}
        names = {folder: set(os.listdir(folder)) for folder in folders}
        caplog.clear()
        cli._clear_owner_retried_provider_limit(_Worker(argv))

    for path in stores:
        assert path.read_bytes() == raw[path], path
    for folder in folders:
        # Nothing is repaired or preserved beside a store: no copy of the torn file appears.
        assert set(os.listdir(folder)) == names[folder], folder
    run = scene["run"]
    assert _board_rows(scene["board"], scene["task_id"], run.id) == ((run.id, "rate_limited"), 1)
    _one_provider_only_warning(caplog, root, scene["task_id"])
