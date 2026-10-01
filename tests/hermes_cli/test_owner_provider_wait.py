"""The owner can try work that waits for the AI provider again now.

A ready card held after a rate-limited run is accepted by the owner retry,
which records ``owner_retry`` against that exact run and writes no profile's
credential pool; the next tick may spawn it, and a limit that still holds books
a new reset and a new hold without counting a failure. The card's own worker,
when it next starts, clears its own pool's limit only for ``anthropic`` at that
provider's default base URL, before its first model request resolves
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


def _retry(owner: ow.OwnerContext, project: dict, task_id: str, key: str) -> dict:
    """The owner's retry, bound to an owner run the way the API server mints it."""
    payload = ow.canonical_owner_retry_payload(
        idempotency_key=key, project_id=project["project_id"],
        task_id=task_id, reason=REASON,
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
            task_id=task_id, reason=REASON,
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
    project = _project(owner, "Wait Pilot")
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
    assert _normalized(snapshot, scene) == json.loads(_SNAPSHOT_AT_START)


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


# The only provider whose limit a worker start resets, and the default endpoint it resets it at.
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


def test_the_worker_start_resets_nothing_when_the_environment_sets_a_base_url(
    root, owner, clock, monkeypatch, caplog,
):
    scene = _owner_retried(root, owner, clock, monkeypatch, "Env Endpoint", 5871, provider=ANTHROPIC)
    _anthropic_pools(root)
    # The dispatcher's own environment names an endpoint, so the worker's carries it too.
    monkeypatch.setenv(PROVIDER_REGISTRY[ANTHROPIC].base_url_env_var, PROXY_URL)
    argv, env = _worker_start(root, scene["board"], scene["task_id"], 5872)
    assert env[PROVIDER_REGISTRY[ANTHROPIC].base_url_env_var] == PROXY_URL
    _start_clears_nothing(root, scene, argv, env, monkeypatch, caplog, ANTHROPIC)
    assert PROXY_URL not in caplog.text


# One endpoint the runtime resolution would send the request to, and one it would pass over.
_CONFIG_URLS = {
    "honoured_by_the_resolution": "https://proxy.placeholder.invalid/anthropic",
    "ignored_by_the_resolution": PROXY_URL,
}


@pytest.mark.parametrize("url", sorted(_CONFIG_URLS))
def test_the_worker_start_resets_nothing_when_the_config_sets_a_base_url(url, root, owner, clock, monkeypatch, caplog):
    scene = _owner_retried(root, owner, clock, monkeypatch, "Config Endpoint", 5881, provider=ANTHROPIC)
    _anthropic_pools(root)
    (root / "profiles" / ASSIGNEE / "config.yaml").write_text(
        f"model:\n  provider: {ANTHROPIC}\n  base_url: {_CONFIG_URLS[url]}\n"
    )
    argv, env = _worker_start(root, scene["board"], scene["task_id"], 5882)
    _start_clears_nothing(root, scene, argv, env, monkeypatch, caplog, ANTHROPIC)
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

    # The helper's one line is a warning naming only the provider: no traceback, no exception text.
    [warning] = [record for record in caplog.records if record.name == "cli"]
    assert warning.levelno == logging.WARNING
    _names_only_the_provider(warning, root, scene["task_id"], ANTHROPIC)
    # The store's own reader still reports the failure in its own record, as it always does.
    assert [record for record in caplog.records if record.name == "hermes_cli.auth" and record.exc_info]


def test_the_worker_start_resets_nothing_when_an_own_row_names_an_endpoint_of_its_own(
    root, owner, clock, monkeypatch, caplog,
):
    scene = _owner_retried(root, owner, clock, monkeypatch, "Row Endpoint", 5901, provider=ANTHROPIC)
    pools = _anthropic_pools(root)
    # Once clear, the exhausted own row would be drawn first, at its own endpoint; the spare row,
    # which a resolution draws while the limit holds, names none.
    _write_pool(pools["own"].parent, {
        ANTHROPIC: [dict(_exhausted("own-s"), base_url=PROXY_URL), _spare("own-t")],
        SECOND_PROVIDER: [_routed("own-b")],
    })
    argv, env = _worker_start(root, scene["board"], scene["task_id"], 5902)
    _start_clears_nothing(root, scene, argv, env, monkeypatch, caplog, ANTHROPIC)
    assert PROXY_URL not in caplog.text
