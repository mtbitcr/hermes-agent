"""Update, Later and Keep current on a model update suggestion, applied once.

A model update suggestion is a native recommendation of kind
``provider_model_policy`` whose subject starts with ``model-update:``. A request
that names the ``model_update_v1`` capability sees it as a ``model_update``
item; Update goes through its own route and moves every role on the current
model in one batch, Later and Keep current reuse defer and reject. These tests
drive the adapter's real routes over a real board, real role configs and the
real model policy; only the provider's live model list is stubbed.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import threading
import time
import urllib.error

import pytest
import yaml
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import PlatformConfig
from gateway.platforms.api_server import APIServerAdapter
from hermes_cli import config as hermes_config
from hermes_cli import kanban_db, owner_workspace as ow, projects_db
from hermes_cli import kanban_db_dispatch as kbd
from hermes_cli.config import get_config_path
from hermes_cli.profiles import get_profile_dir
from hermes_constants import (
    get_default_hermes_root,
    get_hermes_home,
    reset_hermes_home_override,
    set_hermes_home_override,
)
from plugins.dashboard_auth.raphael_workspace import model_policy

_PROVIDER = "anthropic"
_CURRENT = "claude-sonnet-5-5"
_NEWER = "claude-sonnet-5-6"
_SUBJECT = f"model-update:{_PROVIDER}:{_CURRENT}:{_NEWER}:2026-11"
_PRACTICE_SUBJECT = f"model-update:{_PROVIDER}:{_CURRENT}:{_CURRENT}:2026-11"
_FEED = "/v1/owner-workspace/decisions"
_CAPABLE_FEED = "/v1/owner-workspace/decisions?capabilities=model_update_v1"
# Two roles on the current model, and one role on another model that Update
# must leave alone.
_ROUTES = {
    "raphael-builder": (_PROVIDER, _CURRENT, "max"),
    "raphael-business": (_PROVIDER, _CURRENT, "high"),
    "raphael-planner": (_PROVIDER, "claude-opus-5-5", "max"),
}
_ON_CURRENT = ("raphael-builder", "raphael-business")

_TITLE = "A newer version of an AI model is ready for part of your team."
_REASON = (
    "Raphael checked that your account can use it, that it works with Raphael, "
    "and that it did one sample piece of work correctly. Update uses it for new "
    "work only. Work already started or waiting keeps its current setting. You "
    "can undo it in Settings."
)
_PRACTICE = "This is a practice run. Clicking Update changes nothing."
_DONE = "Done. New work will use the newer version."
_NOT_AVAILABLE = "This update is no longer available, so nothing changed."
_NOT_APPLIED = "Raphael could not apply this update, so nothing changed."
_NOT_WAITING = "This choice is no longer waiting."
# A placeholder sign-in: it must reach the provider read and nothing else.
_SIGN_IN = "placeholder-sign-in-not-a-live-key"


def _write_owner_workspace_config(*, enabled: bool) -> None:
    path = get_config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    config = {"gateway": {"api_server": {"owner_workspace": {"enabled": enabled}}}}
    path.write_text(yaml.safe_dump(config), encoding="utf-8")


def _write_route(profile: str, provider: str, model: str, effort: str) -> None:
    directory = get_profile_dir(profile)
    directory.mkdir(parents=True, exist_ok=True)
    token = set_hermes_home_override(str(directory))
    try:
        hermes_config.save_config({
            "model": {"provider": provider, "default": model},
            "agent": {"reasoning_effort": effort},
            "fallback_providers": [],
        })
    finally:
        reset_hermes_home_override(token)


@pytest.fixture
def owner(monkeypatch) -> ow.OwnerContext:
    """The owner the routes resolve for themselves, with the workspace enabled."""
    # gateway.run pins its config home at import; read this test's home instead.
    monkeypatch.setattr("gateway.run._hermes_home", get_hermes_home())
    _write_owner_workspace_config(enabled=True)
    return ow.resolve_owner_context()


@pytest.fixture
def roles(owner) -> None:
    """Enrolled roles with real native routes in the per-test home."""
    for profile, route in _ROUTES.items():
        _write_route(profile, *route)
        model_policy.enroll_profile(profile)


@pytest.fixture
def live_models(monkeypatch) -> dict[str, list[str] | None]:
    """The account's own model list, the one outside read Update makes; None is a failed read.

    Only the provider's reply is stubbed, so the sign-in lookup and the
    account reader above it run as deployed and no picker list can stand in.
    """
    listed: dict[str, list[str] | None] = {_PROVIDER: [_CURRENT, _NEWER, "claude-opus-5-5"]}
    monkeypatch.setenv("ANTHROPIC_API_KEY", _SIGN_IN)

    def get_json(url, **kwargs):
        if listed[_PROVIDER] is None:
            raise urllib.error.URLError("the provider could not be reached")
        return {"data": [{"id": model} for model in listed[_PROVIDER]]}

    monkeypatch.setattr("hermes_cli.models._get_json", get_json)
    return listed


def _project(owner: ow.OwnerContext, name: str = "Team settings") -> dict:
    """Commit one owner Project; its bootstrap confirmation is not under test."""
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(ow, "_confirm", lambda *_args, **_kwargs: {"approved": True})
        project = ow.bootstrap(owner, idempotency_key=f"setup-{name}", name=name)
    assert project["ok"] is True
    return project


def _suggestion(project: dict, subject: str = _SUBJECT) -> str:
    """File the suggestion exactly as the check card does (contract part 1)."""
    practice = subject.split(":")[2] == subject.split(":")[3]
    scope = {flag: False for flag in kanban_db.RECOMMENDATION_SCOPE_FLAGS}
    scope["production_effect"] = not practice
    with contextlib.closing(kanban_db.connect(board=project["board"])) as conn:
        return kanban_db.create_recommendation(
            conn,
            project_id=project["project_id"],
            target_profile="raphael-builder",
            recommendation_kind="provider_model_policy",
            recommendation_subject_id=subject,
            recommendation_label="Use the newer version for part of the team",
            recommendation_rationale="It passed the account, fit and sample checks.",
            recommendation_evidence={
                "schema_version": 1,
                "need": "A newer version is ready for part of the team.",
                "expected_benefit": "New work uses the newer version.",
                "requested_scope": scope,
                "risks": "Low",
                "cost": "No added cost",
                "rollback": "Undo it in Settings.",
            },
            provenance_authority="project-steward",
        )


def _app() -> web.Application:
    """The adapter's own Decisions routes, registered as ``connect()`` does."""
    adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={}))
    app = web.Application()
    for method, path, handler in adapter._http_route_table():
        if path.startswith("/v1/owner-workspace/decisions"):
            app.router.add_route(method, path, handler)
    return app


async def _feed(client: TestClient, path: str = _CAPABLE_FEED) -> list[dict]:
    resp = await client.get(path)
    assert resp.status == 200
    return (await resp.json())["data"]


async def _model_update(client: TestClient) -> dict:
    [item] = [item for item in await _feed(client) if item["kind"] == "model_update"]
    return item


async def _update(client: TestClient, ref: str):
    return await client.post(
        f"/v1/owner-workspace/decisions/{ref}/update", json={"reason": "Update"}
    )


def _events(project: dict, suggestion_id: str, kind: str) -> list[dict]:
    with contextlib.closing(kanban_db.connect(board=project["board"])) as conn:
        return [
            json.loads(event["payload"])
            for event in conn.execute(
                "SELECT payload FROM task_events WHERE task_id = ? AND kind = ? "
                "ORDER BY id",
                (suggestion_id, kind),
            )
        ]


def _receipts(project: dict, suggestion_id: str) -> list[dict]:
    return _events(project, suggestion_id, "model_update_applied") + _events(
        project, suggestion_id, "model_update_not_applied"
    )


def _decision(project: dict, suggestion_id: str) -> dict:
    with contextlib.closing(kanban_db.connect(board=project["board"])) as conn:
        row = conn.execute(
            "SELECT * FROM tasks WHERE id = ?", (suggestion_id,)
        ).fetchone()
    return kanban_db.recommendation_lifecycle_snapshot(row)


def _config_bytes() -> dict[str, bytes]:
    return {
        profile: (get_profile_dir(profile) / "config.yaml").read_bytes()
        for profile in _ROUTES
    }


def _record_bytes() -> bytes | None:
    path = get_default_hermes_root() / "raphael" / "model_policy_updates.json"
    return path.read_bytes() if path.exists() else None


def _configured_model(profile: str) -> str:
    config = yaml.safe_load((get_profile_dir(profile) / "config.yaml").read_text())
    return config["model"]["default"]


def _revision(profile: str) -> str:
    return hermes_config.profile_route_revision(get_profile_dir(profile))


def _record_graph_receipt(owner: ow.OwnerContext, project: dict, task_ids: list[str]) -> None:
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
                    owner.actor, owner.profile, f"graph-{task_ids[0]}",
                    "owner_task_graph_commit", "d" * 64,
                    project["project_id"], project["board"],
                    json.dumps({
                        "ok": True,
                        "project_id": project["project_id"],
                        "board": project["board"],
                        "root_task_id": task_ids[0],
                        "task_ids": task_ids[1:],
                    }),
                    now, now,
                ),
            )


def _locked_task(project: dict, title: str) -> str:
    """Owner work sealed on the builder's current route."""
    lock = kanban_db.mint_policy_lock(
        "raphael-builder", _PROVIDER, _CURRENT, "max", "routine"
    )
    with contextlib.closing(kanban_db.connect(board=project["board"])) as conn:
        return kanban_db.create_task(
            conn,
            title=title,
            assignee="raphael-builder",
            execution_tier="routine",
            project_id=project["project_id"],
            model_override=_CURRENT,
            provider_override=_PROVIDER,
            reasoning_effort="max",
            model_policy_lock=lock,
        )


# 1. Two Updates at the same time: one success, one not found, written once.
@pytest.mark.asyncio
async def test_two_updates_at_once_apply_the_update_once(owner, roles, live_models):
    project = _project(owner)
    suggestion_id = _suggestion(project)
    prior = {profile: _revision(profile) for profile in _ROUTES}
    # Revisions name content, not how often it was written, so the writes are
    # counted where configuration reaches disk, with the real write path running.
    config_files = {str(get_profile_dir(profile) / "config.yaml"): profile for profile in _ROUTES}
    written: list[str] = []
    batches: list[list] = []
    real_write = hermes_config.atomic_yaml_write
    real_batch = hermes_config.apply_profile_route_batch

    def counted_write(path, *args, **kwargs):
        result = real_write(path, *args, **kwargs)
        written.append(config_files.get(str(path), str(path)))
        return result

    def counted_batch(changes, **kwargs):
        batches.append(list(changes))
        return real_batch(changes, **kwargs)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(hermes_config, "atomic_yaml_write", counted_write)
        patch.setattr(hermes_config, "apply_profile_route_batch", counted_batch)
        async with TestClient(TestServer(_app())) as client:
            ref = (await _model_update(client))["decision_ref"]
            answers = await asyncio.gather(_update(client, ref), _update(client, ref))
            bodies = [(resp.status, await resp.json()) for resp in answers]

    # One successful write per affected role, none for the planner, one batch.
    assert sorted(written) == sorted(_ON_CURRENT)
    assert len(batches) == 1
    bodies.sort(key=lambda answer: answer[0])
    assert [status for status, _ in bodies] == [200, 404]
    assert bodies[0][1]["data"]["message"] == _DONE
    assert bodies[1][1]["error"]["code"] == "decision_not_found"

    [receipt] = _receipts(project, suggestion_id)
    assert sorted(receipt["roles"]) == sorted(_ON_CURRENT)
    assert receipt["practice"] is False
    for profile in _ON_CURRENT:
        assert _configured_model(profile) == _NEWER
        # Written once: what is on disk is the one write the receipt records.
        assert receipt["revisions"][profile] == _revision(profile)
        assert receipt["prior_revisions"][profile] == prior[profile]
    assert _configured_model("raphael-planner") == "claude-opus-5-5"
    assert _revision("raphael-planner") == prior["raphael-planner"]
    assert [
        (entry["from_model"], entry["to_model"], entry["state"])
        for entry in model_policy.read_model_updates()
    ] == [(_CURRENT, _NEWER, "active")]
    assert _decision(project, suggestion_id)["decision"] == "accepted"


# 1, after the routes moved: the slower click reads the routes only once the winner
# has moved them, and is still told the choice is gone rather than unavailable.
@pytest.mark.asyncio
async def test_concurrent_loser_returns_not_waiting_even_after_routes_move(
    owner, roles, live_models, monkeypatch
):
    from hermes_cli import owner_model_updates as updates

    project = _project(owner)
    suggestion_id = _suggestion(project)
    real_roles_on = updates.roles_on
    first_reading, second_reading, first_done = (
        threading.Event(), threading.Event(), threading.Event()
    )

    def roles_on(*args):
        if not first_reading.is_set():
            first_reading.set()
            assert second_reading.wait(30), "the second click never read the routes"
        else:
            second_reading.set()
            assert first_done.wait(30), "the first click never finished"
        return real_roles_on(*args)

    async with TestClient(TestServer(_app())) as client:
        ref = (await _model_update(client))["decision_ref"]
        monkeypatch.setattr(updates, "roles_on", roles_on)
        first = asyncio.create_task(_update(client, ref))
        assert await asyncio.to_thread(first_reading.wait, 30)
        second = asyncio.create_task(_update(client, ref))
        try:
            winner = await first
        finally:
            first_done.set()
        loser = await second
        bodies = [(resp.status, await resp.json()) for resp in (winner, loser)]

    assert [status for status, _ in bodies] == [200, 404]
    assert bodies[0][1]["data"]["message"] == _DONE
    assert bodies[1][1]["error"]["code"] == "decision_not_found"
    assert bodies[1][1]["error"]["message"] == _NOT_WAITING
    [receipt] = _receipts(project, suggestion_id)
    assert sorted(receipt["roles"]) == sorted(_ON_CURRENT)
    assert [_configured_model(profile) for profile in _ON_CURRENT] == [_NEWER, _NEWER]


# 2. Update in one tab and Later in another: exactly one outcome.
@pytest.mark.asyncio
async def test_update_and_later_in_two_tabs_give_one_outcome(owner, roles, live_models):
    project = _project(owner)
    suggestion_id = _suggestion(project)
    configs = _config_bytes()

    async with TestClient(TestServer(_app())) as client:
        ref = (await _model_update(client))["decision_ref"]
        update, later = await asyncio.gather(
            _update(client, ref),
            client.post(
                f"/v1/owner-workspace/decisions/{ref}/defer",
                json={"reason": "Later"},
            ),
        )
        statuses = sorted([update.status, later.status])
        loser = update if update.status == 404 else later
        loser_code = (await loser.json())["error"]["code"]

    assert statuses == [200, 404]
    assert loser_code == "decision_not_found"
    decided = _decision(project, suggestion_id)["decision"]
    if update.status == 200:
        assert decided == "accepted"
        assert [_configured_model(p) for p in _ON_CURRENT] == [_NEWER, _NEWER]
        assert len(_events(project, suggestion_id, "model_update_applied")) == 1
    else:
        assert decided == "deferred"
        assert _config_bytes() == configs
        assert _record_bytes() is None
        assert _receipts(project, suggestion_id) == []


# 3. A reload after Update shows nothing pending.
@pytest.mark.asyncio
async def test_a_reload_after_update_shows_nothing_pending(owner, roles, live_models):
    project = _project(owner)
    _suggestion(project)

    async with TestClient(TestServer(_app())) as client:
        ref = (await _model_update(client))["decision_ref"]
        assert (await _update(client, ref)).status == 200
        capable = await _feed(client)
        generic = await _feed(client, _FEED)

    assert [item for item in capable if item["decision_ref"] == ref] == []
    assert [item for item in generic if item["decision_ref"] == ref] == []


# 4. A moved revision or a busy fence: not applied, record restored, config unchanged.
@pytest.mark.parametrize("failure", ["moved_revision", "busy_fence"])
@pytest.mark.asyncio
async def test_a_failed_batch_restores_the_record_and_changes_nothing(
    owner, roles, live_models, monkeypatch, failure
):
    project = _project(owner)
    suggestion_id = _suggestion(project)
    record = _record_bytes()
    configs = _config_bytes()

    if failure == "busy_fence":
        # Receipt-owned owner work of a role on the current model, unsealed and
        # running right now: the fence refuses the route change.
        with contextlib.closing(kanban_db.connect(board=project["board"])) as conn:
            running = kanban_db.create_task(
                conn,
                title="work in progress",
                assignee="raphael-business",
                execution_tier="routine",
                project_id=project["project_id"],
            )
            conn.execute("UPDATE tasks SET status = 'running' WHERE id = ?", (running,))
            conn.commit()
        _record_graph_receipt(owner, project, [running])
    else:
        # Settings moves one role while the owner's click is being checked, so
        # the revision Update read is no longer the one on disk.
        def get_json(url, **kwargs):
            _write_route("raphael-business", "openai-codex", "gpt-5.6-terra", "max")
            configs.update(_config_bytes())
            return {"data": [{"id": _CURRENT}, {"id": _NEWER}]}

        monkeypatch.setattr("hermes_cli.models._get_json", get_json)

    async with TestClient(TestServer(_app())) as client:
        ref = (await _model_update(client))["decision_ref"]
        resp = await _update(client, ref)
        body = await resp.json()
        again = await _update(client, ref)

    assert resp.status == 409
    assert body["error"]["code"] == "model_update_not_applied"
    assert body["error"]["message"] == _NOT_APPLIED
    assert again.status == 404
    assert _record_bytes() == record
    assert _config_bytes() == configs
    assert _events(project, suggestion_id, "model_update_applied") == []
    [not_applied] = _events(project, suggestion_id, "model_update_not_applied")
    assert not_applied["to_model"] == _NEWER
    assert _decision(project, suggestion_id)["decision"] == "accepted"


# 4, beside another Update: a failed Update puts back only what it changed, so an
# Update that started while it was paused keeps its record and its routes agree.
@pytest.mark.asyncio
async def test_failed_update_cannot_erase_another_updates_record(
    owner, roles, live_models, monkeypatch
):
    project = _project(owner)
    sonnet = _suggestion(project)
    opus = _suggestion(project, "model-update:anthropic:claude-opus-5-5:claude-opus-5-6:2026-11")
    live_models[_PROVIDER].append("claude-opus-5-6")
    # The business role is busy, so Update A (the sonnet roles) fails at the fence.
    with contextlib.closing(kanban_db.connect(board=project["board"])) as conn:
        running = kanban_db.create_task(
            conn,
            title="work in progress",
            assignee="raphael-business",
            execution_tier="routine",
            project_id=project["project_id"],
        )
        conn.execute("UPDATE tasks SET status = 'running' WHERE id = ?", (running,))
        conn.commit()
    _record_graph_receipt(owner, project, [running])

    # A pauses just after writing its record until Update B (the planner) has
    # started; B then waits at the route lock until A has failed and restored.
    a_thread: list[int] = []
    a_recorded, b_started, a_done = threading.Event(), threading.Event(), threading.Event()
    real_record = model_policy.apply_model_update_record
    real_lock = hermes_config._route_batch_journal_lock

    def record(update):
        snapshot = real_record(update)
        if update["from_model"] == _CURRENT:
            a_thread.append(threading.get_ident())
            a_recorded.set()
            assert b_started.wait(30), "Update B never started"
        return snapshot

    @contextlib.contextmanager
    def route_lock():
        ident = threading.get_ident()
        if a_recorded.is_set() and ident not in a_thread and not b_started.is_set():
            b_started.set()
            assert a_done.wait(30), "Update A never finished"
        with real_lock():
            yield

    monkeypatch.setattr(model_policy, "apply_model_update_record", record)
    monkeypatch.setattr(hermes_config, "_route_batch_journal_lock", route_lock)
    a_ref, b_ref = (
        ow._owner_decision_ref(owner, authority="recommendation", native_id=native_id)
        for native_id in (sonnet, opus)
    )
    async with TestClient(TestServer(_app())) as client:
        a_task = asyncio.create_task(_update(client, a_ref))
        assert await asyncio.to_thread(a_recorded.wait, 30)
        b_task = asyncio.create_task(_update(client, b_ref))
        try:
            a = await a_task
            a_body = await a.json()
        finally:
            a_done.set()
        b = await b_task

    assert a.status == 409
    assert a_body["error"]["code"] == "model_update_not_applied"
    assert b.status == 200
    assert [
        (entry["from_model"], entry["to_model"], entry["state"])
        for entry in model_policy.read_model_updates()
    ] == [("claude-opus-5-5", "claude-opus-5-6", "active")]
    assert _configured_model("raphael-planner") == "claude-opus-5-6"
    assert [_configured_model(profile) for profile in _ON_CURRENT] == [_CURRENT, _CURRENT]
    for profile in _ROUTES:
        assert _configured_model(profile) == model_policy.assignment_for(profile, _PROVIDER).model


# 5. A running locked task keeps its model, and its lock still validates.
@pytest.mark.asyncio
async def test_a_running_locked_task_keeps_its_model(owner, roles, live_models):
    project = _project(owner)
    _suggestion(project)
    task_id = _locked_task(project, "sealed work in progress")
    _record_graph_receipt(owner, project, [task_id])
    with contextlib.closing(kanban_db.connect(board=project["board"])) as conn:
        assert kanban_db.claim_task(conn, task_id) is not None

    async with TestClient(TestServer(_app())) as client:
        ref = (await _model_update(client))["decision_ref"]
        assert (await _update(client, ref)).status == 200

    assert _configured_model("raphael-builder") == _NEWER
    with contextlib.closing(kanban_db.connect(board=project["board"])) as conn:
        row = conn.execute("SELECT * FROM tasks WHERE id = ?", (task_id,)).fetchone()
    assert row["status"] == "running"
    assert (row["model_override"], row["provider_override"]) == (_CURRENT, _PROVIDER)
    assert kanban_db.task_policy_lock_error(row) is None


# 6. A queued locked card and a retry start with the old model.
@pytest.mark.asyncio
async def test_a_queued_card_and_a_retry_start_with_the_old_model(
    owner, roles, live_models, monkeypatch, tmp_path
):
    project = _project(owner)
    _suggestion(project)
    queued = _locked_task(project, "sealed work waiting")
    retried = _locked_task(project, "sealed work tried again")
    _record_graph_receipt(owner, project, [queued, retried])
    with contextlib.closing(kanban_db.connect(board=project["board"])) as conn:
        assert kanban_db.claim_task(conn, retried) is not None
        assert kanban_db.block_task(conn, retried, reason="Stopped to try again")
        assert kanban_db.unblock_task(conn, retried)

    async with TestClient(TestServer(_app())) as client:
        ref = (await _model_update(client))["decision_ref"]
        assert (await _update(client, ref)).status == 200
    assert _configured_model("raphael-builder") == _NEWER

    launched: list[list[str]] = []

    class _Process:
        pid = 4242

    monkeypatch.setattr(kbd, "_resolve_hermes_argv", lambda: ["hermes"])
    monkeypatch.setattr(
        "tools.process_registry._is_supervised_gateway_process", lambda: False
    )
    monkeypatch.setattr(
        "subprocess.Popen", lambda cmd, **_kwargs: launched.append(list(cmd)) or _Process()
    )
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    with contextlib.closing(kanban_db.connect(board=project["board"])) as conn:
        for task_id in (queued, retried):
            task = kanban_db.claim_task(conn, task_id)
            assert task is not None
            kbd._default_spawn(task, str(workspace), board=project["board"])

    assert len(launched) == 2
    for argv in launched:
        model_at = argv.index("-m")
        assert argv[model_at:model_at + 4] == ["-m", _CURRENT, "--provider", _PROVIDER]


# 7. A new model that is not available at the click is refused, nothing written.
@pytest.mark.parametrize("listing", ["missing", "empty", "failed"])
@pytest.mark.asyncio
async def test_an_unavailable_new_model_is_refused_before_any_write(
    owner, roles, live_models, listing
):
    live_models[_PROVIDER] = {
        "missing": [_CURRENT, "claude-opus-5-5"],
        "empty": [],
        "failed": None,
    }[listing]
    project = _project(owner)
    suggestion_id = _suggestion(project)
    configs = _config_bytes()
    pending = _decision(project, suggestion_id)

    async with TestClient(TestServer(_app())) as client:
        ref = (await _model_update(client))["decision_ref"]
        resp = await _update(client, ref)
        body = await resp.json()

    assert resp.status == 409
    assert body["error"]["code"] == "model_update_unavailable"
    assert body["error"]["message"] == _NOT_AVAILABLE
    assert _config_bytes() == configs
    assert _record_bytes() is None
    assert _receipts(project, suggestion_id) == []
    assert _decision(project, suggestion_id) == pending


# 7, with a picker that already offers the new version: only the account's list counts.
@pytest.mark.parametrize("listing", ["missing", "empty", "failed"])
@pytest.mark.asyncio
async def test_fresh_account_list_not_picker_fallback(
    owner, roles, live_models, monkeypatch, listing
):
    from hermes_cli import models

    monkeypatch.setitem(models._PROVIDER_MODELS, _PROVIDER, [_CURRENT, _NEWER])
    await test_an_unavailable_new_model_is_refused_before_any_write(
        owner, roles, live_models, listing
    )


# 7, on Codex: the offline catalog an empty or failed account answer falls back to
# is not the account's list, and the account's own list is what admits the version.
@pytest.mark.parametrize("status", [200, 503])
@pytest.mark.asyncio
async def test_codex_fallback_is_not_account_availability(
    owner, roles, monkeypatch, caplog, status
):
    from hermes_cli import codex_models

    caplog.set_level(logging.DEBUG)
    old, new = "gpt-5.6-terra", "gpt-5.7-terra"
    _write_route("raphael-business", "openai-codex", old, "max")
    monkeypatch.setattr(
        codex_models, "DEFAULT_CODEX_MODELS", [*codex_models.DEFAULT_CODEX_MODELS, new]
    )
    monkeypatch.setattr(
        "hermes_cli.auth.resolve_codex_runtime_credentials",
        lambda **kwargs: {"api_key": _SIGN_IN},
    )
    reply = {"status": status, "models": []}
    signed = []

    class Reply:
        def __init__(self):
            self.status_code = reply["status"]

        def json(self):
            return {"models": reply["models"]}

    def get(url, *, headers, **kwargs):
        signed.append(headers["Authorization"])
        return Reply()

    monkeypatch.setattr("httpx.get", get)
    project = _project(owner)
    suggestion_id = _suggestion(project, f"model-update:openai-codex:{old}:{new}:2026-11")
    configs = _config_bytes()

    async with TestClient(TestServer(_app())) as client:
        ref = (await _model_update(client))["decision_ref"]
        refused = await _update(client, ref)
        refused_body = await refused.json()
        assert refused.status == 409
        assert refused_body["error"]["code"] == "model_update_unavailable"
        assert _config_bytes() == configs
        assert _record_bytes() is None
        assert _decision(project, suggestion_id)["decision"] == "pending"

        reply.update(status=200, models=[{"slug": new, "priority": 1}])
        applied = await _update(client, ref)
        applied_body = await applied.json()

    assert applied.status == 200
    assert _configured_model("raphael-business") == new
    # The runtime's sign-in reached the provider read, and nothing else kept it.
    assert signed and set(signed) == {f"Bearer {_SIGN_IN}"}
    for kept in (
        json.dumps([refused_body, applied_body]),
        _record_bytes().decode(),
        json.dumps(_receipts(project, suggestion_id)),
        caplog.text,
    ):
        assert _SIGN_IN not in kept


# 8. A request that does not name the capability gets the generic suggestion.
@pytest.mark.asyncio
async def test_without_the_capability_the_row_stays_a_generic_suggestion(
    owner, roles, live_models
):
    from hermes_cli.owner_text import check_owner_text

    project = _project(owner)
    _suggestion(project)

    async with TestClient(TestServer(_app())) as client:
        [generic] = await _feed(client, _FEED)
        [capable] = await _feed(client)

    assert generic["kind"] == "capability"
    assert "actions" not in generic and "practice" not in generic
    assert capable == {
        "decision_ref": generic["decision_ref"],
        "kind": "model_update",
        "title": _TITLE,
        "reason": _REASON,
        "actions": ["update", "later", "keep_current"],
        "practice": False,
    }
    for text in (capable["title"], capable["reason"]):
        for name in ("claude", "sonnet", "anthropic", _CURRENT, _NEWER):
            assert name not in text.lower()
    assert check_owner_text(capable["title"], "title") == _TITLE
    assert check_owner_text(capable["reason"], "stop_reason_decision") == _REASON


# 9. An accept through the generic route applies nothing.
@pytest.mark.asyncio
async def test_accept_through_the_generic_route_applies_nothing(owner, roles, live_models):
    project = _project(owner)
    suggestion_id = _suggestion(project)
    configs = _config_bytes()

    async with TestClient(TestServer(_app())) as client:
        ref = (await _model_update(client))["decision_ref"]
        accepted = await client.post(
            f"/v1/owner-workspace/decisions/{ref}/accept", json={"reason": "Yes"}
        )
        assert accepted.status == 200
        assert (await _update(client, ref)).status == 404

    assert _decision(project, suggestion_id)["decision"] == "accepted"
    assert _config_bytes() == configs
    assert _record_bytes() is None
    assert _receipts(project, suggestion_id) == []


# 10. A practice Update changes no byte of the configuration and makes one receipt.
@pytest.mark.asyncio
async def test_a_practice_update_changes_nothing_and_makes_one_receipt(
    owner, roles, live_models
):
    project = _project(owner)
    suggestion_id = _suggestion(project, _PRACTICE_SUBJECT)
    configs = _config_bytes()
    revisions = {profile: _revision(profile) for profile in _ROUTES}

    async with TestClient(TestServer(_app())) as client:
        item = await _model_update(client)
        assert (item["practice"], item["reason"]) == (True, _PRACTICE)
        resp = await _update(client, item["decision_ref"])
        body = await resp.json()
        again = await _update(client, item["decision_ref"])

    assert resp.status == 200
    assert body["data"]["message"] == _PRACTICE
    assert again.status == 404
    assert _config_bytes() == configs
    assert {profile: _revision(profile) for profile in _ROUTES} == revisions
    assert _record_bytes() is None
    [receipt] = _receipts(project, suggestion_id)
    assert receipt["practice"] is True
    assert (receipt["from_model"], receipt["to_model"]) == (_CURRENT, _CURRENT)


@pytest.mark.asyncio
async def test_keep_current_reuses_reject_and_applies_nothing(owner, roles, live_models):
    project = _project(owner)
    suggestion_id = _suggestion(project)
    configs = _config_bytes()

    async with TestClient(TestServer(_app())) as client:
        ref = (await _model_update(client))["decision_ref"]
        kept = await client.post(
            f"/v1/owner-workspace/decisions/{ref}/reject", json={"reason": "Keep current"}
        )
        assert kept.status == 200
        assert [item for item in await _feed(client) if item["decision_ref"] == ref] == []

    assert _decision(project, suggestion_id)["decision"] == "rejected"
    assert _config_bytes() == configs
    assert _receipts(project, suggestion_id) == []


def test_every_owner_text_is_fixed_and_names_no_model():
    from hermes_cli import owner_model_updates as updates
    from hermes_cli.owner_text import check_owner_text

    texts = {
        "title": updates.TITLE,
        "reason": updates.REASON,
        "practice": updates.PRACTICE,
        "done": updates.DONE,
        "not_available": updates.NOT_AVAILABLE,
        "not_applied": updates.NOT_APPLIED,
    }
    assert texts == {
        "title": _TITLE,
        "reason": _REASON,
        "practice": _PRACTICE,
        "done": _DONE,
        "not_available": _NOT_AVAILABLE,
        "not_applied": _NOT_APPLIED,
    }
    assert check_owner_text(_TITLE, "title") == _TITLE
    for text in texts.values():
        assert check_owner_text(text, "stop_reason_decision") == text
        for name in ("claude", "sonnet", "opus", "gpt", "anthropic", "openai"):
            assert name not in text.lower()
