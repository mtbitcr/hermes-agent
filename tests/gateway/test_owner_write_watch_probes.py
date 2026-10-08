"""Plan tests 10 to 12 of the hourly owner write check, against the API server app built as the
listener builds it: ``APIServerAdapter.connect()``, with its middleware in its order and its route
table with each ``/p/{profile}`` mirror, serving a real loopback port from this test's temporary
home. Every probe is the check's own request, sent by its own sender to the address its own
listener reader gives, with each serving profile's own key from that profile's own .env.

Each control applies a test's own check to a case where it must fail: a real write through the
same listener, a permitted route with the right key, and a job that does not keep its state.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import socket
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

pytest.importorskip("aiohttp")

from gateway.config import PlatformConfig  # noqa: E402
from gateway.platforms.api_server import APIServerAdapter  # noqa: E402
from hermes_cli import kanban_db, owner_workspace as ow, owner_write_watch as watch  # noqa: E402
from hermes_constants import (  # noqa: E402
    get_hermes_home,
    reset_hermes_home_override,
    set_hermes_home_override,
)

START = datetime(2026, 10, 5, 9, 0, tzinfo=timezone.utc)
HOUR = timedelta(hours=1)
PLANNER = "raphael-planner"
# Each serving profile's own key, only ever a request's bearer.
DEFAULT_KEY = "probe-default-" + "7c41e9a2" * 4
PLANNER_KEY = "probe-planner-" + "d2b85f06" * 4
WRONG_KEY = "probe-wrong-" + "39e0a7c4" * 4
DECISION_ROUTES = tuple(
    f"POST /v1/owner-workspace/decisions/{{decision_ref}}/{action}"
    for action in ("accept", "reject", "defer")
)
RELEASE_ROUTES = tuple(
    f"POST /v1/owner-workspace/release/{{batch_id}}/{action}" for action in ("accept", "defer", "start")
)
# The default profile's other probed routes, and the Decisions feed the owner app reads.
OTHER_ROUTES = (
    "GET /v1/owner-workspace/decisions", "POST /v1/runs", "POST /v1/runs/{run_id}/approval",
) + RELEASE_ROUTES
[DECISIONS] = [probe for probe in watch.PROBES["default"] if probe.name == "decisions"]
RELEASE = [probe for probe in watch.PROBES["default"] if "/owner-workspace/release/" in probe.path]
# The plan's stores, as paths in the home; each board is every kanban.db found there.
STORES = ("projects.db", "response_store.db", "cron/jobs.json", "cron/executions.db",
          "cron/delivery_records.db")
_EVIDENCE = {
    "schema_version": 1,
    "need": "The workshop outline needs current public evidence.",
    "expected_benefit": "Keep the owner-facing advice current.",
    "requested_scope": {flag: False for flag in kanban_db.RECOMMENDATION_SCOPE_FLAGS},
    "risks": "Low",
    "cost": "No added cost",
    "rollback": "Remove the staged skill configuration.",
}
_SQLITE = b"SQLite format 3\x00"


def _free_port() -> int:
    with contextlib.closing(socket.socket()) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _settings(root: Path, port: int, permitted=None) -> None:
    """The root profile's own config.yaml: the listener's port, the owner workspace switched on and,
    when given, the permitted-route list."""
    api_server = {"enabled": True, "port": port, "owner_workspace": {"enabled": True}}
    if permitted is not None:
        api_server["allowed_routes"] = list(permitted)
    (root / "config.yaml").write_text(yaml.safe_dump({"gateway": {"api_server": api_server}}),
                                      encoding="utf-8")


@pytest.fixture
def home(tmp_path, monkeypatch):
    """The temporary home: the root (default) profile with its port, its own key and the owner
    workspace switched on; the planner profile with its own key; one owner Project with one pending
    suggestion; an empty scheduled-jobs store; one execution with its delivery record."""
    from cron import delivery_record, executions, jobs

    root = get_hermes_home()
    for name in ("API_SERVER_HOST", "API_SERVER_PORT"):
        monkeypatch.delenv(name, raising=False)
    # gateway.run pins its config home at import; read this test's home instead.
    monkeypatch.setattr("gateway.run._hermes_home", root)
    port = _free_port()
    _settings(root, port)
    (root / ".env").write_text(f"API_SERVER_KEY={DEFAULT_KEY}\n", encoding="utf-8")
    planner = root / "profiles" / PLANNER
    planner.mkdir(parents=True)
    (planner / ".env").write_text(f"API_SERVER_KEY={PLANNER_KEY}\n", encoding="utf-8")

    owner = ow.resolve_owner_context()
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(ow, "_confirm", lambda *_args, **_kwargs: {"approved": True})
        project = ow.bootstrap(owner, idempotency_key="setup-probes", name="Workshop Pilot")
    assert project["ok"] is True
    with contextlib.closing(kanban_db.connect(board=project["board"])) as conn:
        suggestion = kanban_db.create_recommendation(
            conn,
            project_id=project["project_id"],
            target_profile=PLANNER,
            recommendation_kind="skill",
            recommendation_subject_id="workshop-research",
            recommendation_label="Add workshop research support",
            recommendation_rationale="The current milestone needs public-source research.",
            recommendation_evidence=_EVIDENCE,
            provenance_authority="project-steward",
        )

    jobs.save_jobs([])
    row = executions.create_execution("0" * 12, source="schedule")
    executions.finish_execution(row["id"], success=True)
    with delivery_record.recording(row["id"], "0" * 12):
        recorder = delivery_record.active_recorder()
        recorder.begin("Earlier report", [], [{"platform": "telegram", "chat_id": "-100100000001"}])
        recorder.next_target()
        recorder.note_sent(True)

    page_check = tmp_path / "page-check.sh"
    page_check.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    return SimpleNamespace(root=root, planner=planner, port=port, owner=owner, project=project,
                           suggestion=suggestion, page_check=page_check)


@contextlib.asynccontextmanager
async def listening(home):
    """The API server built and served as the listener builds it: one adapter bound to the gateway's
    multiplexing runner (so /p/<profile> enters that profile), then connect()."""
    adapter = APIServerAdapter(PlatformConfig(
        enabled=True, extra={"host": "127.0.0.1", "port": home.port, "key": DEFAULT_KEY}))
    adapter.gateway_runner = SimpleNamespace(
        config=SimpleNamespace(multiplex_profiles=True, multiplex_profile_allowlist=None),
        adapters={}, _profile_adapters={}, _draining=False, _external_drain_active=False,
    )
    try:
        assert await adapter.connect()
        yield adapter
    finally:
        await adapter.disconnect()


def _registered(method, path):
    """The deployed route policy with every Automations, Connections and Models route registered
    for its scope: these indirect signals send no request and are not under test here."""
    scopes = {(signal.method, signal.path): signal.scope for signal in watch.SIGNALS}
    return SimpleNamespace(required_scope=scopes.get((method, path)))


def _job(profile, profile_home, page_check, moment):
    """One hourly run of ``profile``'s own job in that profile's own home: its key from its own
    .env, its own state file, the address from the root profile's listener settings, and real
    requests by the check's own sender. Each message counts as delivered."""
    token = set_hermes_home_override(profile_home)
    try:
        return watch.run(str(page_check), profile=profile, run_page_check=lambda _path: watch.PASS,
                         route_policy=_registered, delivery_state=lambda _announced: "delivered",
                         now=lambda: moment)
    finally:
        reset_hermes_home_override(token)


def _send(probe, key):
    """The check's own request for ``probe``, by its own sender, to its own listener address."""
    return watch._probe(probe, watch._listener_base(), key, watch._http_send, lambda: START)


def _rows(path: Path):
    """One SQLite store's version, schema and rows, read through a read-only connection."""
    with contextlib.closing(sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True)) as conn:
        schema = conn.execute("SELECT type, name, sql FROM sqlite_master ORDER BY type, name").fetchall()
        rows = {name: sorted(map(repr, conn.execute(f'SELECT * FROM "{name}"')))
                for kind, name, sql in schema
                if kind == "table" and not (sql or "").upper().startswith("CREATE VIRTUAL")}
        return conn.execute("PRAGMA user_version").fetchone()[0], schema, rows


def _stores(root: Path) -> dict:
    """Every file in the home: each SQLite store as its rows, every other file as its bytes. Left
    out: SQLite's own side files, which a read alone may create or change, and the check's own
    state and report files."""
    snapshot = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name.endswith(("-wal", "-shm", "-journal")) \
                or path.name in (watch.STATE_FILE, watch.REPORT_FILE):
            continue
        content = path.read_bytes()
        is_store = path.suffix == ".db" or content.startswith(_SQLITE)
        snapshot[path.relative_to(root).as_posix()] = _rows(path) if is_store else content
    return snapshot


def _changed(before: dict, after: dict) -> list:
    return sorted(name for name in before.keys() | after.keys() if before.get(name) != after.get(name))


def _pending(home) -> tuple:
    """The suggestion's lifecycle and decision records read afresh, and the owner's Decisions feed."""
    with contextlib.closing(kanban_db.connect(board=home.project["board"])) as conn:
        row = conn.execute("SELECT * FROM tasks WHERE id = ?", (home.suggestion,)).fetchone()
        decided = conn.execute(
            "SELECT COUNT(*) FROM task_events WHERE task_id = ? AND kind = 'recommendation_decided'",
            (home.suggestion,)).fetchone()[0]
    feed = [item["kind"] for item in ow.list_owner_decisions(home.owner)["data"]]
    return kanban_db.recommendation_lifecycle_snapshot(row)["decision"], decided, feed


def _answers(*homes) -> dict:
    """Each probe's outcome, status and code, from each job's own report."""
    return {check["name"]: (check["outcome"], check["status"], check["code"])
            for profile_home in homes
            for check in json.loads((profile_home / watch.REPORT_FILE).read_text())["checks"]
            if check["method"] == "POST"}


async def _probe_every_route(home) -> list:
    """Both serving profiles' jobs, one hourly run each: every probe the check has."""
    return [await asyncio.to_thread(_job, profile, profile_home, home.page_check, START)
            for profile, profile_home in (("default", home.root), (PLANNER, home.planner))]


@pytest.mark.asyncio
async def test_10_no_probe_changes_state(home):
    """10. No probe changes state: build the API server app exactly as the listener does (middleware
    order and route table), in a temporary home with the owner workspace switched on and one pending
    suggestion. Run every probe and prove every store is unchanged (projects, each board, response
    store, scheduled jobs, delivery records) and the suggestion is still pending."""
    async with listening(home) as adapter:
        adapter._response_store.put("resp_kept", {"id": "resp_kept", "object": "response", "output": []})
        before = _stores(home.root)
        outcomes = await _probe_every_route(home)
        after = _stores(home.root)

    # Every probe ran and got its exact healthy answer, from behind the key check.
    assert [(outcome.code, outcome.kind) for outcome in outcomes] == [(0, "active"), (0, "active")]
    assert _answers(home.root, home.planner) == {
        probe.name: (watch.PASS, probe.status, probe.code)
        for table in watch.PROBES.values() for probe in table
    }
    boards = [name for name in before if name.endswith("kanban.db")]
    assert kanban_db.kanban_db_path(board=home.project["board"]).relative_to(home.root).as_posix() in boards
    assert set(STORES) <= set(before)
    assert _changed(before, after) == []
    assert _pending(home) == ("pending", 0, ["capability"])


@pytest.mark.asyncio
async def test_10_control_a_real_write_is_caught(home):
    """Control for 10: after every probe, one real write through the same listener (the probe's own
    request with the Decisions feed's real key) is caught by the same comparison, and the suggestion
    is no longer pending, so both checks of test 10 fail on a write."""
    async with listening(home) as adapter:
        adapter._response_store.put("resp_kept", {"id": "resp_kept", "object": "response", "output": []})
        before = _stores(home.root)
        await _probe_every_route(home)
        [ref] = [item["decision_ref"] for item in ow.list_owner_decisions(home.owner)["data"]]
        base = watch._listener_base()
        status, _ = await asyncio.to_thread(
            watch._http_send, DECISIONS.method, base + DECISIONS.path.replace(watch._DECISION, ref),
            DECISIONS.body, DEFAULT_KEY)
        after = _stores(home.root)

    assert status == 200
    board = kanban_db.kanban_db_path(board=home.project["board"]).relative_to(home.root).as_posix()
    assert _changed(before, after) == [board]
    assert _pending(home) == ("deferred", 1, [])


def _spy_on_the_lookup(monkeypatch) -> list:
    """Record each entry into the suggestion lookup, which then runs as before."""
    entered, lookup = [], ow._owner_pending_suggestion

    def recorded(ctx, decision_ref):
        entered.append(decision_ref)
        return lookup(ctx, decision_ref)

    monkeypatch.setattr(ow, "_owner_pending_suggestion", recorded)
    return entered


@pytest.mark.asyncio
async def test_11_the_refusal_runs_before_the_lookup(home, monkeypatch):
    """11. The refusal runs before the lookup: with the decision routes left out of the permitted
    list, the probe gets 403 route_not_allowed and the suggestion lookup is never entered; with a
    wrong key it gets 401 and no lookup."""
    entered = _spy_on_the_lookup(monkeypatch)
    async with listening(home):
        _settings(home.root, home.port, OTHER_ROUTES)
        refused = await asyncio.to_thread(_send, DECISIONS, DEFAULT_KEY)
        assert (refused.status, refused.code, refused.outcome) == (403, "route_not_allowed", watch.FAIL)
        assert entered == []

        _settings(home.root, home.port, OTHER_ROUTES + DECISION_ROUTES)
        wrong_key = await asyncio.to_thread(_send, DECISIONS, WRONG_KEY)
        assert (wrong_key.status, wrong_key.code, wrong_key.outcome) == (401, "gateway_auth_failed", watch.FAIL)
        assert entered == []
    assert _pending(home) == ("pending", 0, ["capability"])


@pytest.mark.asyncio
async def test_11_control_a_permitted_route_reaches_the_lookup(home, monkeypatch):
    """Control for 11: with the decision routes permitted and the right key, the same probe enters
    the suggestion lookup and gets its healthy answer, so the spy of test 11 sees a lookup that
    runs."""
    entered = _spy_on_the_lookup(monkeypatch)
    async with listening(home):
        _settings(home.root, home.port, OTHER_ROUTES + DECISION_ROUTES)
        healthy = await asyncio.to_thread(_send, DECISIONS, DEFAULT_KEY)

    assert (healthy.status, healthy.code, healthy.outcome) == (404, "decision_not_found", watch.PASS)
    assert entered == [watch._DECISION]


@pytest.mark.asyncio
async def test_release_probes_pass_against_the_real_release_routes(home):
    """Each release probe gets 400 invalid_argument from the real release route, which refuses the
    batch id 0 before any read, reservation or action."""
    assert len(RELEASE) == 3
    async with listening(home):
        _settings(home.root, home.port, OTHER_ROUTES + DECISION_ROUTES)
        sent = [await asyncio.to_thread(_send, probe, DEFAULT_KEY) for probe in RELEASE]

    assert [(r.status, r.code, r.outcome) for r in sent] == [(400, "invalid_argument", watch.PASS)] * 3


@pytest.mark.asyncio
async def test_a_release_route_missing_from_the_allowlist_fails_its_probe(home):
    """With one release route left out of the permitted list, that route's probe gets 403
    route_not_allowed and fails, and the other two release probes still pass."""
    assert len(RELEASE) == 3
    async with listening(home):
        for left_out, probe in zip(RELEASE_ROUTES, RELEASE):
            permitted = tuple(route for route in OTHER_ROUTES if route != left_out)
            _settings(home.root, home.port, permitted + DECISION_ROUTES)
            sent = {p.name: await asyncio.to_thread(_send, p, DEFAULT_KEY) for p in RELEASE}
            assert (sent[probe.name].status, sent[probe.name].code, sent[probe.name].outcome) == (
                403, "route_not_allowed", watch.FAIL)
            assert [r.outcome for name, r in sent.items() if name != probe.name] == [watch.PASS] * 2


async def _replay(home, *, keep_state=True) -> list:
    """The default profile's job, hourly: one run with the decision routes permitted, then 44 runs
    with a permitted list without them, then the list restored and one more run. ``(hour, outcome)``
    with hour 0 the first run after the change."""
    outcomes = []
    for hour in range(-1, 45):
        _settings(home.root, home.port, OTHER_ROUTES if 0 <= hour < 44 else OTHER_ROUTES + DECISION_ROUTES)
        if not keep_state:
            watch.state_path().unlink(missing_ok=True)
        outcome = await asyncio.to_thread(_job, "default", home.root, home.page_check, START + hour * HOUR)
        outcomes.append((hour, outcome))
    return outcomes


def _assert_one_alert_then_one_recovery(outcomes) -> None:
    assert all(outcome.code == 0 for _, outcome in outcomes)
    (_, ready), *replay = outcomes
    assert ready.kind == "active"
    assert [(hour, outcome.kind) for hour, outcome in replay if outcome.text] == [(0, "alert"), (44, "recovery")]
    alert, recovery = replay[0][1].text, replay[44][1].text
    assert alert.splitlines()[1:] == [f"- {DECISIONS.label}: answered 403 route_not_allowed"]
    assert "The failure lasted 1 d 20 h (2026-10-05 09:00 UTC to 2026-10-07 05:00 UTC)." in recovery


@pytest.mark.asyncio
async def test_12_the_44_hour_replay(home):
    """12. The 44-hour replay: a permitted list without the three decision routes for 44 simulated
    hours, then restored, gives exactly one alert at the first run and one recovery at the first run
    after the fix."""
    async with listening(home):
        outcomes = await _replay(home)
    _assert_one_alert_then_one_recovery(outcomes)


@pytest.mark.asyncio
async def test_12_control_a_job_that_forgets_its_state_alerts_every_hour(home):
    """Control for 12: the same replay by a job whose state file is gone before each run alerts at
    each of the 44 runs and never sends a recovery, so the check of test 12 fails on it."""
    async with listening(home):
        outcomes = await _replay(home, keep_state=False)
    assert [outcome.kind for _, outcome in outcomes] == ["active"] + ["alert"] * 44 + ["active"]
    with pytest.raises(AssertionError):
        _assert_one_alert_then_one_recovery(outcomes)
