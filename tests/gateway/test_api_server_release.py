"""Release card 8: the owner's release decision, served by the API server.

Each test runs on its own Hermes root. The release units the user manager lists
and every child process the gateway starts are fakes, so no test starts a real
release unit or a real ``hermes release start``.
"""

from __future__ import annotations

import asyncio
import hashlib
import sqlite3
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import PlatformConfig
from gateway.platforms.api_server import APIServerAdapter
from hermes_cli import release_ledger as ledger
from hermes_cli import release_unit
from hermes_cli.kanban_risk_tier import RISK_NOT_RECORDED

KEY = "sk-owner-secret"
VIEW = "/v1/owner-workspace/release"
PAGE_REF = "decisions-page:release"


@pytest.fixture
def root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A Hermes root at ``<tmp>/.hermes`` that serves the owner workspace."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    _owner_workspace(home, monkeypatch, enabled=True)
    return home


@pytest.fixture(autouse=True)
def units(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """Fake release units, and a fake child for every process the gateway starts.

    A child that exits 0 leaves its release unit running, as a real start does.
    """
    fake = SimpleNamespace(busy=[], exits=[], started=[])

    async def child(*argv, **_options):
        fake.started.append(argv)
        code = fake.exits.pop(0) if fake.exits else 0
        if code == 0:
            fake.busy.append((release_unit.unit_name(int(argv[-1])), "active"))
        return SimpleNamespace(returncode=code, communicate=_no_output)

    def in_process(_batch_id):
        raise AssertionError("the release start must run as a child process of the gateway")

    monkeypatch.setattr(release_unit, "busy_units", lambda: list(fake.busy))
    monkeypatch.setattr(asyncio, "create_subprocess_exec", child)
    monkeypatch.setattr("hermes_cli.release_cmd.start", in_process)
    return fake


async def _no_output():
    return b"", None


def _owner_workspace(home: Path, monkeypatch: pytest.MonkeyPatch, *, enabled: bool) -> None:
    monkeypatch.setattr("gateway.run._hermes_home", home)
    config = {"gateway": {"api_server": {"owner_workspace": {"enabled": enabled}}}}
    (home / "config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")


def _client(key: str = "") -> TestClient:
    adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={"key": key} if key else {}))
    app = web.Application()
    for method, path, handler in adapter._http_route_table():
        if path.startswith(VIEW):
            app.router.add_route(method, path, handler)
    return TestClient(TestServer(app))


def _commit(label: str) -> str:
    return hashlib.sha1(label.encode()).hexdigest()


def _ledger(step, *args, **kwargs):
    """One step on the release record, taken as the merge step and the release run take it."""
    conn = ledger.connect()
    try:
        return step(conn, *args, **kwargs)
    finally:
        conn.close()


def _merge(label: str, title: str, *, tier=1) -> dict:
    """Record one merged change; returns its batch."""
    return _ledger(
        ledger.record_merge,
        merge_commit=_commit(label),
        pr_url="https://github.com/mtbitcr/hermes-agent/pull/41",
        reviewed_base=_commit(f"{label}-base"),
        reviewed_head=_commit(f"{label}-head"),
        reviewed_tree=_commit(f"{label}-tree"),
        tier=tier,
        card_id=f"t_{_commit(label)[:8]}",
        title=title,
        on_main=True,
    )["batch"]


def _accepted(batch: dict) -> dict:
    """The batch accepted outside the gateway, for a test that starts from there."""
    return _ledger(
        ledger.decide_release,
        batch["batch_id"],
        decision="accepted",
        shown_digest=batch["digest"],
        expected_version=batch["version"],
        decision_ref=PAGE_REF,
    )


def _store(home: Path, sql: str, *params) -> list:
    """Rows read straight from the one root store, independent of the handlers."""
    store = sqlite3.connect((home / "kanban" / "release_ledger.db").as_uri() + "?mode=ro", uri=True)
    try:
        return store.execute(sql, params).fetchall()
    finally:
        store.close()


def _answers(home: Path) -> int:
    return _store(home, "SELECT COUNT(*) FROM release_events WHERE kind = 'release_decided'")[0][0]


def _bytes(home: Path) -> dict:
    """Every name under the home, with each file's bytes."""
    return {str(path): path.read_bytes() if path.is_file() else None for path in home.rglob("*")}


def _shown(batch: dict) -> dict:
    return {"version": batch["version"], "digest": batch["digest"]}


def _start_argv(batch_id: int) -> tuple:
    return (sys.executable, "-m", "hermes_cli.main", "release", "start", str(batch_id))


@pytest.mark.asyncio
async def test_release_decision_lists_the_waiting_batch_in_plain_words(root):
    async with _client() as client:
        empty = await (await client.get(VIEW)).json()
        assert (empty["waiting"], empty["release"]) == (None, None)
        assert not (root / "kanban").exists()  # the read made no store
        _merge("a", "Show the release decision", tier=0)
        _merge("b", "Keep the owner page fast", tier=1)
        batch = _merge(
            "c",
            "Fix the start (https://github.com/mtbitcr/hermes-agent/pull/9) for t_0badc0de"
            " at 1a2b3c4d5e in /srv/hermes/start.py",
            tier=None,
        )
        before = _bytes(root)
        response = await client.get(VIEW)
        text = await response.text()
        body = await response.json()

    assert response.status == 200
    assert _bytes(root) == before  # a read changes no byte of the live home
    assert body["waiting"] == {
        "batch_id": batch["batch_id"],
        "status": "Waiting for your decision.",
        "titles": [
            "Show the release decision",
            "Keep the owner page fast",
            "Fix the start … for … at … in …",
        ],
        "count": 3,
        "tier": 2,
        "tier_sentence": f"{RISK_NOT_RECORDED} Tier 2, high risk: the release goes forward,"
        " rehearses the way back, then goes forward again.",
        "version": batch["version"],
        "digest": batch["digest"],
        "actions": ["accept", "defer"],
    }
    assert body["release"] is None
    hidden = [str(root), "github.com", "t_0badc0de", "1a2b3c4d5e", "start.py"]
    for member in batch["members"]:
        hidden += [member["card_id"], member["pr_url"]]
        commits = ("merge_commit", "reviewed_base", "reviewed_head", "reviewed_tree")
        hidden += [member[name][:7] for name in commits]
    assert [item for item in hidden if item in text] == []
    assert "/" not in text


@pytest.mark.asyncio
async def test_accept_with_the_shown_version_and_digest_starts_the_release(root, units):
    _merge("a", "Show the release decision")
    async with _client() as client:
        shown = (await (await client.get(VIEW)).json())["waiting"]
        accept = f"{VIEW}/{shown['batch_id']}/accept"
        response = await client.post(accept, json=_shown(shown))
        body = await response.json()
        repeated = await client.post(accept, json=_shown(shown))

    assert response.status == 200
    assert (body["answer"], body["message"]) == ("releasing", "Accepted. The release is running.")
    assert _answers(root) == 1
    assert _store(root, "SELECT state FROM release_batches") == [("accepted",)]
    # One unit start: the release start command, run as a child process.
    assert units.started == [_start_argv(shown["batch_id"])]
    release = body["decision"]["release"]
    assert (release["batch_id"], release["unit_running"]) == (shown["batch_id"], True)
    assert release["actions"] == []
    assert body["decision"]["waiting"] is None
    # A repeated accept records no second answer and starts nothing.
    assert repeated.status == 409
    assert _answers(root) == 1
    assert len(units.started) == 1


@pytest.mark.asyncio
async def test_stale_accept_changes_nothing_and_returns_the_new_list(root, units):
    _merge("a", "Show the release decision")
    async with _client() as client:
        shown = (await (await client.get(VIEW)).json())["waiting"]
        current = _merge("b", "Keep the owner page fast")  # lands after the page was drawn
        accept = f"{VIEW}/{shown['batch_id']}/accept"
        before = _bytes(root)
        stale = await client.post(accept, json=_shown(shown))
        body = await stale.json()
        forged = await client.post(accept, json={"version": current["version"], "digest": "0" * 64})

    assert stale.status == forged.status == 409
    assert body["error"]["code"] == "release_changed"
    assert _bytes(root) == before
    assert units.started == []
    waiting = body["decision"]["waiting"]
    assert waiting["titles"] == ["Show the release decision", "Keep the owner page fast"]
    assert _shown(waiting) == _shown(current)


@pytest.mark.asyncio
async def test_put_off_keeps_the_batch_waiting(root, units):
    _merge("a", "Show the release decision")
    async with _client() as client:
        shown = (await (await client.get(VIEW)).json())["waiting"]
        defer = f"{VIEW}/{shown['batch_id']}/defer"
        response = await client.post(defer, json=_shown(shown))
        body = await response.json()
        repeated = await client.post(defer, json=_shown(shown))
        again = await client.post(defer, json=_shown(body["decision"]["waiting"]))
        _merge("b", "Keep the owner page fast")
        later = (await (await client.get(VIEW)).json())["waiting"]

    assert response.status == 200
    assert body["answer"] == "put_off"
    # An answer is recorded once: neither the repeat nor a second put off records another.
    assert repeated.status == again.status == 409
    assert _answers(root) == 1
    assert units.started == []
    assert _store(root, "SELECT state FROM release_batches") == [("deferred",)]
    assert (later["batch_id"], later["count"]) == (shown["batch_id"], 2)
    assert later["status"] == "Put off by you. New changes still join it."
    assert later["actions"] == ["accept"]


@pytest.mark.asyncio
async def test_failed_unit_start_keeps_the_batch_accepted_and_offers_start_again(root, units):
    _merge("a", "Show the release decision")
    units.exits = [1]  # the first start cannot start the release unit
    async with _client() as client:
        shown = (await (await client.get(VIEW)).json())["waiting"]
        response = await client.post(f"{VIEW}/{shown['batch_id']}/accept", json=_shown(shown))
        body = await response.json()
        restarted = await client.post(f"{VIEW}/{shown['batch_id']}/start")
        answer = await restarted.json()

    assert response.status == 200
    assert body["answer"] == "could_not_start"
    assert _store(root, "SELECT state FROM release_batches") == [("accepted",)]
    release = body["decision"]["release"]
    assert (release["batch_id"], release["state"]) == (shown["batch_id"], "accepted")
    assert (release["unit_running"], release["actions"]) == (False, ["start"])
    assert restarted.status == 200
    assert answer["answer"] == "releasing"
    assert units.started == [_start_argv(shown["batch_id"])] * 2
    assert _answers(root) == 1


@pytest.mark.asyncio
async def test_start_again_is_refused_while_a_release_unit_runs(root, units):
    batch = _accepted(_merge("a", "Show the release decision"))
    new = batch["members"][-1]["merge_commit"]
    _ledger(ledger.begin_release, batch["batch_id"], prev=_commit("live"), new=new)
    units.busy = [(release_unit.unit_name(batch["batch_id"]), "active")]
    start = f"{VIEW}/{batch['batch_id']}/start"
    async with _client() as client:
        running = (await (await client.get(VIEW)).json())["release"]
        refused = await client.post(start)
        body = await refused.json()
        units.busy = []  # the run stopped midway: still releasing, and no unit runs
        stopped = (await (await client.get(VIEW)).json())["release"]
        _ledger(ledger.finish_release, batch["batch_id"], outcome="released")
        released = (await (await client.get(VIEW)).json())["release"]
        ended = await client.post(start)

    assert (running["state"], running["unit_running"], running["actions"]) == ("releasing", True, [])
    assert refused.status == 409
    assert body["error"]["code"] == "release_unit_running"
    assert (stopped["unit_running"], stopped["actions"]) == (False, ["start"])
    assert (released["outcome_text"], released["actions"]) == ("It was released.", [])
    assert ended.status == 409
    assert units.started == []


@pytest.mark.asyncio
async def test_routes_refuse_without_owner_authentication(root, units, monkeypatch):
    batch = _merge("a", "Show the release decision")
    routes = [("GET", VIEW)] + [
        ("POST", f"{VIEW}/{batch['batch_id']}/{action}") for action in ("accept", "defer", "start")
    ]
    owner = {"Authorization": f"Bearer {KEY}"}
    client = _client(KEY)
    before = _bytes(root)
    async with client:
        for method, path in routes:
            for headers in ({}, {"Authorization": "Bearer sk-not-the-owner"}):
                response = await client.request(method, path, json=_shown(batch), headers=headers)
                assert response.status == 401, (method, path)
        assert _bytes(root) == before
        assert (await client.get(VIEW, headers=owner)).status == 200
        # There is no reject action.
        reject = f"{VIEW}/{batch['batch_id']}/reject"
        assert (await client.post(reject, json=_shown(batch), headers=owner)).status == 404
        _owner_workspace(root, monkeypatch, enabled=False)
        before = _bytes(root)
        for method, path in routes:
            response = await client.request(method, path, json=_shown(batch), headers=owner)
            assert response.status == 404, (method, path)
            assert (await response.json())["error"]["code"] == "owner_workspace_not_enabled"

    assert _bytes(root) == before
    assert _answers(root) == 0
    assert units.started == []


@pytest.mark.asyncio
async def test_routes_reach_the_root_release_store_under_a_worker_env(root, units, monkeypatch):
    first = _accepted(_merge("from-operator-1", "Show the release decision"))
    _merge("from-operator-2", "Keep the owner page fast")
    waiting = _merge("from-operator-3", "Hold the release unit")

    # Exactly what the dispatcher injects into a worker it spawns.
    profile_home = root / "profiles" / "worker"
    board_dir = root / "kanban" / "boards" / "worker-board"
    profile_home.mkdir(parents=True)
    board_dir.mkdir(parents=True)
    _owner_workspace(profile_home, monkeypatch, enabled=True)
    monkeypatch.setenv("HERMES_HOME", str(profile_home))
    monkeypatch.setenv("HERMES_PROFILE", "worker")
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_0badc0de")
    monkeypatch.setenv("HERMES_KANBAN_DB", str(board_dir / "kanban.db"))
    monkeypatch.setenv("HERMES_KANBAN_BOARD", "worker-board")
    monkeypatch.setenv("HERMES_KANBAN_WORKSPACES_ROOT", str(board_dir / "workspaces"))

    client = _client()
    before = _bytes(root)
    async with client:
        view = await (await client.get(VIEW)).json()
        accept = await client.post(f"{VIEW}/{waiting['batch_id']}/accept", json=_shown(waiting))
        started = await client.post(f"{VIEW}/{first['batch_id']}/start")
        answer = await started.json()

    # Independent count, read straight from the one root store.
    sql = "SELECT COUNT(*) FROM release_members WHERE batch_id = ?"
    (stored,) = _store(root, sql, waiting["batch_id"])[0]
    assert view["waiting"]["count"] == stored == 2
    assert _shown(view["waiting"]) == _shown(waiting)
    assert (view["release"]["batch_id"], view["release"]["actions"]) == (first["batch_id"], ["start"])
    # A worker never decides a release; the start still reaches the root record.
    assert accept.status == 403
    assert _answers(root) == 1
    assert started.status == 200
    assert answer["answer"] == "releasing"
    assert units.started == [_start_argv(first["batch_id"])]
    # No byte changed under the root, so no second store sits in the profile home or the board.
    assert _bytes(root) == before
