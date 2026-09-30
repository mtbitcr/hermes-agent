"""The owner can try work that waits for the AI provider again now.

A ready card held after a rate-limited run is accepted by the owner retry,
which records ``owner_retry`` against that exact run and clears the provider's
exhaustion mark in the assignee profile's own pool only; the next tick may
spawn it, and a limit that still holds books a new reset and a new hold
without counting a failure. A reader that names ``provider_wait_v1`` is told
plainly why stopped work waits; one that does not gets today's payload,
compared whole against the one captured at the start commit.

Every stop is booked through the real reap path (the dispatcher spawns the
card, its worker exits, the sweep books the run), and every owner call runs
under exactly the environment the dispatcher builds for a claimed worker. The
root, each profile home, each pool and each board live in the test's own
temporary folder; every card, profile, provider, key and time is made up.
"""

from __future__ import annotations

import contextlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from agent.credential_pool import CredentialPool
from hermes_cli import kanban_db as kb, owner_workspace as ow
from hermes_cli.kanban_provider_stops import KANBAN_PROVIDER_REFUSED_EXIT_CODE

ASSIGNEE = "worker"
STEWARD = "steward"
OTHER = "other"
PROVIDER = "placeholder-alpha"
SECOND_PROVIDER = "placeholder-beta"
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


def _card(conn, project: dict, title: str) -> str:
    return kb.create_task(
        conn, title=title, assignee=ASSIGNEE, project_id=project["project_id"],
        provider_override=PROVIDER, model_override="placeholder-model",
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


def test_the_pool_clear_reaches_only_the_assignees_own_pool_for_that_provider(
    root, owner, clock, monkeypatch, caplog,
):
    project = _project(owner, "Pool Pilot")
    board = project["board"]
    with _board(board) as conn:
        task_id = _card(conn, project, "Draft the placeholder letter")
        run = _held(conn, board, task_id, 4401)
        clock["t"] = run.ended_at + 60

    root_pool = _write_pool(root, {PROVIDER: [_exhausted("root-a")]})
    own_pool = _write_pool(root / "profiles" / ASSIGNEE, {
        PROVIDER: [_exhausted("own-a")],
        SECOND_PROVIDER: [_exhausted("own-b")],
    })
    other_pool = _write_pool(root / "profiles" / STEWARD, {PROVIDER: [_exhausted("other-a")]})
    untouched = {path: path.read_bytes() for path in (root_pool, other_pool)}

    env = _steward_env(root, board)
    caplog.set_level("DEBUG")
    with _as_worker(monkeypatch, env):
        retried = _retry(owner, project, task_id, "retry-pool")
    assert retried["ok"] is True, retried

    for path, raw in untouched.items():
        assert path.read_bytes() == raw
    pool = json.loads(own_pool.read_text())["credential_pool"]
    cleared = dict(_exhausted("own-a"))
    cleared.pop("failure_reason")
    cleared.update({key: None for key in _STATUS_KEYS})
    assert pool[PROVIDER] == [cleared]
    assert pool[SECOND_PROVIDER] == [_exhausted("own-b")]

    with _board(board) as conn:
        events = json.dumps([event.payload for event in kb.list_events(conn, task_id)])
    for leaked in (json.dumps(retried), events, caplog.text):
        assert "placeholder-token" not in leaked


def test_the_pool_clear_writes_nothing_without_a_provider_an_own_entry_or_off_the_root(root):
    root_pool = _write_pool(root, {PROVIDER: [_exhausted("root-a")]})
    own_pool = _write_pool(root / "profiles" / ASSIGNEE, {SECOND_PROVIDER: [_exhausted("own-b")]})
    before = {path: path.read_bytes() for path in (root_pool, own_pool)}

    assert CredentialPool.clear_profile_provider_exhaustion(ASSIGNEE, None) == 0
    assert CredentialPool.clear_profile_provider_exhaustion(ASSIGNEE, PROVIDER) == 0
    assert CredentialPool.clear_profile_provider_exhaustion(OTHER, PROVIDER) == 0
    assert CredentialPool.clear_profile_provider_exhaustion("default", PROVIDER) == 0

    for path, raw in before.items():
        assert path.read_bytes() == raw
    assert not (root / "profiles" / OTHER / "auth.json").exists()


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

    when = datetime.fromtimestamp(resume_at, timezone.utc).strftime("%Y-%m-%d %H:%M")
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
