"""The owner can try work that waits for the AI provider again now.

A ready card held after a rate-limited run is accepted by the owner retry,
which records ``owner_retry`` against that exact run and clears the provider's
exhaustion mark in the assignee profile's own pool only; the next tick may
spawn it, and a limit that still holds books a new reset and a new hold
without counting a failure. A reader that names ``provider_wait_v1`` is told
plainly why stopped work waits; one that does not gets today's payload,
compared whole against the one captured at the start commit. A card reads as
waiting only while the dispatcher still holds it, and the pool clear finds the
provider the card or its run records.

Every stop is booked through the real reap path (the dispatcher spawns the
card, its worker exits, the sweep books the run), and every owner call runs
under exactly the environment the dispatcher builds for a claimed worker. The
root, each profile home, each pool and each board live in the test's own
temporary folder; every card, profile, provider, key and time is made up.
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

from agent.credential_pool import CredentialPool
from hermes_cli import auth, auth_nous
from hermes_cli import kanban_db as kb, kanban_provider_stops, owner_workspace as ow
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


_LEGACY_USER = "placeholder-user"
_LEGACY_PASSWORD = "placeholder-pass"
_LEGACY_HOST = "legacy.placeholder.invalid"
_PORTAL = "https://portal.placeholder.invalid"
_STAMPS = ("version", "updated_at")


def _legacy_store(pool: dict) -> dict:
    """A store whose other provider keeps legacy state the shared loader migrates and logs."""
    return {
        "version": 1,
        "active_provider": "nous",
        "updated_at": "2030-03-17T17:00:00+00:00",
        "providers": {
            "nous": {
                "portal_base_url": f"https://{_LEGACY_USER}:{_LEGACY_PASSWORD}@{_LEGACY_HOST}/",
                "access_token": "placeholder-token-nous",
            },
        },
        "credential_pool": pool,
    }


def test_the_pool_clear_keeps_another_providers_legacy_state_and_logs_no_credential(
    root, owner, clock, monkeypatch, caplog, tmp_path,
):
    project = _project(owner, "Legacy Pilot")
    board = project["board"]
    with _board(board) as conn:
        task_id = _card(conn, project, "Draft the placeholder note")
        run = _held(conn, board, task_id, 4801)
        clock["t"] = run.ended_at + 60

    # The shared loader would move the portal URL kept for this host, and log it.
    monkeypatch.setattr(auth_nous, "_NOUS_STALE_PORTAL_HOSTS", frozenset({_LEGACY_HOST}))
    monkeypatch.setattr(auth_nous, "DEFAULT_NOUS_PORTAL_URL", _PORTAL)
    root_pool = _write_pool(root, {PROVIDER: [_exhausted("root-a")]})
    own_pool = root / "profiles" / ASSIGNEE / "auth.json"
    own_pool.write_text(json.dumps(_legacy_store({
        PROVIDER: [_exhausted("own-a")],
        SECOND_PROVIDER: [_exhausted("own-b")],
    }), indent=2) + "\n")
    steward_pool = _write_pool(root / "profiles" / STEWARD, {PROVIDER: [_exhausted("steward-a")]})
    other_pool = _write_pool(root / "profiles" / OTHER, {PROVIDER: [_exhausted("other-a")]})
    pools = (root_pool, own_pool, steward_pool, other_pool)
    untouched = {path: path.read_bytes() for path in (root_pool, steward_pool, other_pool)}
    before = _exhausted_entries(pools)

    env = _steward_env(root, board)
    caplog.set_level("DEBUG")
    caplog.set_level("DEBUG", logger="hermes_cli.auth")
    with _as_worker(monkeypatch, env):
        retried = _retry(owner, project, task_id, "retry-legacy-state")
    assert retried["ok"] is True, retried

    for path, raw in untouched.items():
        assert path.read_bytes() == raw
    cleared = dict(_exhausted("own-a"))
    cleared.pop("failure_reason")
    cleared.update({key: None for key in _STATUS_KEYS})
    expected = _legacy_store({PROVIDER: [cleared], SECOND_PROVIDER: [_exhausted("own-b")]})
    stored = json.loads(own_pool.read_text())
    assert {key: value for key, value in stored.items() if key not in _STAMPS} == {
        key: value for key, value in expected.items() if key not in _STAMPS
    }
    assert _exhausted_entries(pools) == before - 1

    # The capture is live: it hears the shared loader move a URL on that host.
    probe = tmp_path / "probe.json"
    probe.write_text(json.dumps({
        "providers": {"nous": {"portal_base_url": f"https://{_LEGACY_HOST}/"}},
    }))
    heard = len(caplog.records)
    assert auth._load_auth_store(probe)["providers"]["nous"]["portal_base_url"] == _PORTAL
    assert "hermes_cli.auth" in {record.name for record in caplog.records[heard:]}

    with _board(board) as conn:
        [event] = _events(conn, task_id, "owner_retry")
        assert event.run_id == run.id
        events = json.dumps([event.payload for event in kb.list_events(conn, task_id)])
    for leaked in (json.dumps(retried), events, caplog.text):
        for secret in (_LEGACY_USER, _LEGACY_PASSWORD, "placeholder-token"):
            assert secret not in leaked


def _unparsed_store(shape: str) -> bytes:
    """The assignee's own store, holding an exhausted entry, in a shape no store is read from."""
    store = {"version": 1, "credential_pool": {PROVIDER: [_exhausted("own-a")]}}
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
        task_id = _card(conn, project, "Draft the placeholder card")
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
        task_id = _card(conn, project, "Draft the placeholder report")
        run = _held(conn, board, task_id, 5001)
        clock["t"] = run.ended_at + 60

    root_pool = _write_pool(root, {PROVIDER: [_exhausted("root-a")]})
    steward_pool = _write_pool(root / "profiles" / STEWARD, {PROVIDER: [_exhausted("steward-a")]})
    other_pool = _write_pool(root / "profiles" / OTHER, {PROVIDER: [_exhausted("other-a")]})
    pools = (root_pool, steward_pool, other_pool)
    untouched = {path: path.read_bytes() for path in pools}
    before = _exhausted_entries(pools)

    home = root / "profiles" / ASSIGNEE
    own_pool = home / "auth.json"
    if state == "unreadable":
        # A valid store holding an exhausted entry that no one, root included,
        # may read: only this exact path is refused, every other read is real.
        _write_pool(home, {PROVIDER: [_exhausted("own-a")]})
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
    # Only the lock the clear reads under may leave its file: no copy, no temporary file.
    assert set(os.listdir(home)) - names <= {own_pool.with_suffix(".lock").name}

    with _board(board) as conn:
        [event] = _events(conn, task_id, "owner_retry")
        assert event.run_id == run.id
        assert kb.get_task(conn, task_id).status == "ready"
        assert kb.check_respawn_guard(conn, task_id) is None
        events = json.dumps([event.payload for event in kb.list_events(conn, task_id)])
    records = [record for record in caplog.records if record.name == "agent.credential_pool"]
    if state == "unreadable":
        # Refused on every Python version. A link to itself makes resolve()
        # raise only on some, so that case is not required to be heard.
        [warning] = [record for record in records if record.levelno == logging.WARNING]
        assert PROVIDER in warning.getMessage()
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
    assert kb.set_model_override(conn, task_id, "placeholder-model", provider=PROVIDER)
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
        "provider": PROVIDER, "model": "placeholder-model",
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
    assert run.metadata["runtime_receipt"]["provider"] == PROVIDER
    assert run.metadata[kanban_provider_stops.RATE_LIMIT_RESET_PROVIDER_KEY] == (
        kanban_provider_stops._identity(PROVIDER)
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
def test_the_retry_clears_the_assignees_own_entry_for_the_provider_the_card_or_its_run_records(
    book, root, owner, clock, monkeypatch, caplog,
):
    project = _project(owner, "Recorded Route")
    board = project["board"]
    with _board(board) as conn:
        task_id, run = book(conn, board, project, monkeypatch)
        clock["t"] = run.ended_at + 60
        assert kb.check_respawn_guard(conn, task_id) == "rate_limit_cooldown"

    root_pool = _write_pool(root, {PROVIDER: [_exhausted("root-a")]})
    own_pool = _write_pool(root / "profiles" / ASSIGNEE, {
        PROVIDER: [_exhausted("own-a")],
        SECOND_PROVIDER: [_exhausted("own-b")],
    })
    steward_pool = _write_pool(root / "profiles" / STEWARD, {PROVIDER: [_exhausted("steward-a")]})
    other_pool = _write_pool(root / "profiles" / OTHER, {PROVIDER: [_exhausted("other-a")]})
    pools = (root_pool, own_pool, steward_pool, other_pool)
    untouched = {path: path.read_bytes() for path in (root_pool, steward_pool, other_pool)}
    before = _exhausted_entries(pools)

    env = _steward_env(root, board)
    caplog.set_level("DEBUG")
    with _as_worker(monkeypatch, env):
        retried = _retry(owner, project, task_id, "retry-recorded-route")
    assert retried["ok"] is True, retried

    for path, raw in untouched.items():
        assert path.read_bytes() == raw
    cleared = dict(_exhausted("own-a"))
    cleared.pop("failure_reason")
    cleared.update({key: None for key in _STATUS_KEYS})
    assert json.loads(own_pool.read_text())["credential_pool"] == {
        PROVIDER: [cleared], SECOND_PROVIDER: [_exhausted("own-b")],
    }
    assert _exhausted_entries(pools) == before - 1

    with _board(board) as conn:
        [event] = _events(conn, task_id, "owner_retry")
        assert event.run_id == run.id
        events = json.dumps([event.payload for event in kb.list_events(conn, task_id)])
    for leaked in (json.dumps(retried), events, caplog.text):
        assert "placeholder-token" not in leaked
