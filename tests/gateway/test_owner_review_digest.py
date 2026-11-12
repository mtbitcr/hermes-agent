"""Deliver one daily review through the existing owner channel boundary."""

import importlib
import asyncio
import hashlib
import json
import logging
import multiprocessing
import sqlite3
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from gateway import owner_channel_pause
from gateway.config import Platform
from hermes_cli import owner_workspace as ow, projects_db
from hermes_cli.owner_text import check_owner_text
from tests.gateway.test_owner_heads_up import HOME_CHAT, root  # noqa: F401
from tests.gateway.test_owner_heads_up import _waiting
from tests.hermes_cli.test_owner_project_review_cycle import _at
from tools import approval


class Adapter:
    def __init__(self, outcomes=()):
        self.sent = []
        self.outcomes = list(outcomes)

    async def send(self, chat, text):
        self.sent.append((chat, text))
        result = self.outcomes.pop(0) if self.outcomes else SimpleNamespace(success=True)
        if isinstance(result, Exception):
            raise result
        return result


def _project(root, name="Meeting Plan"):
    ctx = ow.OwnerContext(actor="default", profile="default", session="daily-digest-fixture")

    def approve(data):
        assert approval.resolve_gateway_approval(ctx.session, "once", approval_id=data["approval_id"]) == 1

    approval.register_gateway_notify(ctx.session, approve)
    try:
        made = ow.bootstrap(ctx, idempotency_key=f"daily-digest-{name}", name=name)
        assert made["ok"]
        return next(p for p in ow.list_committed_projects(ctx) if p["project_id"] == made["project_id"])
    finally:
        approval.unregister_gateway_notify(ctx.session)


def _headline(root, project, day, now, headline):
    with projects_db.connect_closing(root / "projects.db") as registry:
        projects_db.write_project_review_brief(
            registry, project["project_id"], review_date=day, covered_from=None, covered_to=day,
            headline=headline, what_changed="The room booking is confirmed.",
            next_step="Prepare the guest list.", needs_you=None, ready_work=[], cut_note=None,
            quiet=False, written_at=now,
        )


def _ready(root, project, day, now):
    _headline(root, project, day, now, "The room is booked.")
    with projects_db.connect_closing(root / "projects.db") as registry:
        projects_db.write_portfolio_review_brief(
            registry, day, conflicts=[], priorities=["Meeting Plan: Prepare the guest list."],
            shared_blockers=[], written_at=now, project_scope=[project["project_id"]],
        )


def _scope(root, *projects):
    """Allowlist ``projects`` and widen today's portfolio scope to them."""
    ids = [project["project_id"] for project in projects]
    config = yaml.safe_load((root / "config.yaml").read_text())
    config["owner_review"]["allowlist"] = ids
    (root / "config.yaml").write_text(yaml.safe_dump(config))
    with projects_db.connect_closing(root / "projects.db") as registry:
        registry.execute("UPDATE portfolio_review_briefs SET project_scope=?", (json.dumps(ids),))
        registry.commit()


def _runner(adapter):
    return SimpleNamespace(_active_profile_name=lambda: "default", adapters={Platform.TELEGRAM: adapter})


def _clock(monkeypatch):
    clock = [int(time.time())]
    monkeypatch.setattr(time, "time", lambda: clock[0])
    return clock


@pytest.fixture
def ready(root, monkeypatch):
    now = _at(2026, 11, 10, 6, 30)
    monkeypatch.setattr(time, "time", lambda: now)
    project = _project(root)
    config = yaml.safe_load((root / "config.yaml").read_text())
    config["owner_review"] = {"enabled": True, "portfolio": True, "digest": True, "allowlist": [project["project_id"]]}
    (root / "config.yaml").write_text(yaml.safe_dump(config))
    _ready(root, project, "2026-11-10", now)
    return project


NOTES_LINE = "These are Raphael's notes from today's review. To act on any of them, open the app."
QUESTION = "Which room should we book?"
LONG_QUESTION = (
    "Which room should we book for the spring meeting of the whole team, the large hall on the ground floor "
    "near the main door or the small room next to the kitchen with a view of the garden?"
)


@pytest.mark.asyncio
@pytest.mark.parametrize("announced", [False, True])
@pytest.mark.parametrize("question,quote", [
    (QUESTION, "WHICH ROOM  SHOULD WE BOOK?"),
    (QUESTION, "Which room should we book"),
    (QUESTION, "Which room, should we book?"),
    (QUESTION, "Which room should we book？"),
    (QUESTION, QUESTION.translate(str.maketrans("oei", "\u043e\u0435\u0456"))),  # Cyrillic look-alikes
    (LONG_QUESTION, LONG_QUESTION[:LONG_QUESTION.rfind(" ", 0, 120)]),
], ids=["case_and_spaces", "no_question_mark", "inserted_comma", "full_width_mark", "look_alike_letters",
        "longer_than_the_line"])
async def test_review_prose_cannot_repeat_a_live_wait_question(root, ready, announced, question, quote):
    from gateway import owner_heads_up

    # The policy admits each quote, so only the digest's own guard can keep it out.
    assert check_owner_text(quote, "planner_prose") == quote and len(quote) <= 120
    _waiting(ready, title="Choose a room", reason=question)
    adapter = Adapter()
    runner = _runner(adapter)
    if announced:
        assert await owner_heads_up.tick(runner) == 1
        adapter.sent.clear()
    with projects_db.connect_closing(root / "projects.db") as registry:
        registry.execute("UPDATE project_review_briefs SET headline=?", (quote,))
        registry.execute("UPDATE portfolio_review_briefs SET priorities=?", (json.dumps(["Meeting Plan: " + quote]),))
        registry.commit()
    digest = importlib.import_module("gateway.owner_review_digest")
    assert await digest.tick(runner) == (0 if announced else 1)
    if not announced:
        assert adapter.sent[0][1].splitlines()[1:] == [NOTES_LINE, "Meeting Plan: 1 answer waiting on Decisions"]
    else:
        assert adapter.sent == []


@pytest.mark.asyncio
@pytest.mark.parametrize("outside", ["portfolio_scope", "allowlist"])
async def test_a_project_outside_the_portfolio_scope_or_the_allowlist_adds_nothing(root, ready, outside):
    other = _project(root, "Garden Party")
    if outside == "portfolio_scope":
        config = yaml.safe_load((root / "config.yaml").read_text())
        config["owner_review"]["allowlist"].append(other["project_id"])
        (root / "config.yaml").write_text(yaml.safe_dump(config))
    _headline(root, other, "2026-11-10", int(time.time()), "The garden is ready.")
    garden_question = "Which caterer should we hire for the garden?"
    _waiting(other, title="Choose a caterer", reason=garden_question)
    with projects_db.connect_closing(root / "projects.db") as registry:
        # Review prose of the visible Project quotes the other Project's live question.
        registry.execute("UPDATE project_review_briefs SET headline=? WHERE project_id=?",
                         (garden_question, ready["project_id"]))
        registry.commit()
    adapter = Adapter()
    digest = importlib.import_module("gateway.owner_review_digest")
    assert await digest.tick(_runner(adapter)) == 1
    assert adapter.sent[0][1].splitlines()[1:] == [NOTES_LINE, "Meeting Plan: Prepare the guest list."]


@pytest.mark.asyncio
@pytest.mark.parametrize("fault", ["written_later", "other_day", "changed_between_reads", "board_unread"])
async def test_an_unproven_portfolio_or_an_unread_board_leaves_the_day_unsent(root, ready, monkeypatch, fault):
    from gateway import owner_heads_up

    digest = importlib.import_module("gateway.owner_review_digest")
    _waiting(ready, title="Choose a room", reason=QUESTION)
    unreadable = [fault == "board_unread"]
    board_waits = owner_heads_up._board_waits

    def board_read(project):
        if unreadable[0]:
            raise sqlite3.OperationalError("disk I/O error")
        return board_waits(project)

    monkeypatch.setattr(owner_heads_up, "_board_waits", board_read)
    if fault == "changed_between_reads":
        view = digest._portfolio_view

        def rewritten_after_the_view(projects):
            result = view(projects)
            with projects_db.connect_closing(root / "projects.db") as registry:
                registry.execute("UPDATE portfolio_review_briefs SET priorities=?",
                                 (json.dumps(["Meeting Plan: Book the hall."]),))
                registry.commit()
            return result

        monkeypatch.setattr(digest, "_portfolio_view", rewritten_after_the_view)
    elif fault != "board_unread":
        with projects_db.connect_closing(root / "projects.db") as registry:
            change = "written_at=written_at+60" if fault == "written_later" else "review_date='2026-11-09'"
            registry.execute(f"UPDATE portfolio_review_briefs SET {change}")
            registry.commit()
    adapter = Adapter()
    assert await digest.tick(_runner(adapter)) == 0
    assert adapter.sent == [] and not (root / "owner_review_digest.json").exists()
    if fault == "board_unread":
        unreadable[0] = False
        assert await digest.tick(_runner(adapter)) == 1
        assert "Meeting Plan: 1 answer waiting on Decisions" in adapter.sent[0][1]


@pytest.mark.asyncio
async def test_one_digest_a_day_and_a_told_day_reads_no_board_until_the_next_review_day(root, ready, monkeypatch):
    from gateway import owner_heads_up

    clock = _clock(monkeypatch)
    reads = []
    board_waits = owner_heads_up._board_waits
    monkeypatch.setattr(owner_heads_up, "_board_waits",
                        lambda project: reads.append(project["project_id"]) or board_waits(project))
    adapter = Adapter()
    runner = _runner(adapter)
    digest = importlib.import_module("gateway.owner_review_digest")
    assert await digest.tick(runner) == 1
    for _ in range(3):
        clock[0] += 30
        assert await digest.tick(runner) == 0
    assert len(adapter.sent) == 1 and reads == [ready["project_id"]]
    clock[0] = _at(2026, 11, 11, 6, 30)
    _ready(root, ready, "2026-11-11", clock[0])
    assert await digest.tick(runner) == 1
    assert len(adapter.sent) == 2 and adapter.sent[1][1].startswith("Your daily project review (2026-11-11)")


@pytest.mark.asyncio
async def test_a_retryable_refusal_waits_before_each_next_attempt(root, ready, monkeypatch):
    clock = _clock(monkeypatch)
    start = clock[0]
    refused = SimpleNamespace(success=False, retryable=True)
    adapter = Adapter([refused, refused])
    runner = _runner(adapter)
    digest = importlib.import_module("gateway.owner_review_digest")
    attempts = []
    for seconds in (0, 30, 299, 300, 330, 1199, 1200, 1230):
        clock[0] = start + seconds
        await digest.tick(runner)
        attempts.append(len(adapter.sent))
    assert attempts == [1, 1, 1, 2, 2, 2, 3, 3]
    assert json.loads((root / "owner_review_digest.json").read_text())["told"]["2026-11-10"]["state"] == "sent"


@pytest.mark.asyncio
async def test_wait_counts_lead_after_the_notes_line_and_survive_the_cut(root, ready):
    projects = [ready] + [_project(root, f"Project {number:02d}") for number in range(1, 12)]
    _scope(root, *projects)
    # Twelve headlines of 117 characters already pass 1,500 characters on their own.
    headline = ("The team finished the first draft, checked every page and sent the plan to the printer "
                "for a proof on Friday morning.")
    for project in projects:
        with projects_db.connect_closing(root / "projects.db") as registry:
            registry.execute("DELETE FROM project_review_briefs WHERE project_id=?", (project["project_id"],))
            registry.commit()
        _headline(root, project, "2026-11-10", int(time.time()), headline)
    _waiting(projects[-1], title="Choose a room", reason=QUESTION)
    adapter = Adapter()
    digest = importlib.import_module("gateway.owner_review_digest")
    assert await digest.tick(_runner(adapter)) == 1
    text = adapter.sent[0][1]
    assert len(text) == 1500 and text.endswith("…")
    assert text.splitlines()[1:3] == [NOTES_LINE, "Project 11: 1 answer waiting on Decisions"]
    assert text.splitlines()[3] == "Meeting Plan: " + headline


@pytest.mark.asyncio
async def test_a_failing_tick_warns_once_a_day_with_the_error_type_only(root, ready, monkeypatch, caplog):
    clock = _clock(monkeypatch)

    def unreadable_registry(_ctx):
        raise sqlite3.DatabaseError("synthetic registry detail")

    monkeypatch.setattr(ow, "list_committed_projects", unreadable_registry)
    caplog.set_level(logging.DEBUG, logger="gateway.owner_review_digest")
    adapter = Adapter()
    # A fresh module, so a failure warned earlier today in this process cannot hide the first warning.
    digest = importlib.reload(importlib.import_module("gateway.owner_review_digest"))
    for _ in range(3):
        assert await digest.tick(_runner(adapter)) == 0
        clock[0] += 30
    clock[0] = _at(2026, 11, 11, 6, 30)
    assert await digest.tick(_runner(adapter)) == 0
    warnings = [record.getMessage() for record in caplog.records
                if record.name == "gateway.owner_review_digest" and record.levelno >= logging.WARNING]
    assert len(warnings) == 2 and all("DatabaseError" in message for message in warnings)
    assert "synthetic registry detail" not in caplog.text and adapter.sent == []


@pytest.mark.asyncio
async def test_gateway_start_registers_the_digest_and_its_real_loop_retries_a_refusal(root, ready, monkeypatch):
    from gateway import run_watchers
    from gateway.run import GatewayRunner

    clock = _clock(monkeypatch)
    start = clock[0]

    async def minute_passes(_runner, _seconds):
        clock[0] += 60
        await asyncio.sleep(0)

    monkeypatch.setattr(run_watchers, "_interruptible_sleep", minute_passes)
    runner = GatewayRunner.__new__(GatewayRunner)
    spawned = {}
    runner._spawn_supervised = lambda factory, name, **_kwargs: spawned.setdefault(name, factory)
    runner._failed_platforms = {}
    runner._spawn_reconnect_watcher = lambda: None
    runner._scale_to_zero_should_arm = lambda: False
    runner._log_scale_to_zero_not_armed_reason = lambda: None
    runner._active_profile_name = lambda: "default"
    runner._running = True
    runner._start_spawn_background_watchers()
    assert "owner_review_digest_watcher" in spawned

    class StopAfterRetry(Adapter):
        async def send(self, chat, text):
            result = await super().send(chat, text)
            sends.append(clock[0] - start)
            if len(self.sent) == 2:
                runner._running = False
            return result

    sends = []
    adapter = StopAfterRetry([SimpleNamespace(success=False, retryable=True)])
    runner.adapters = {Platform.TELEGRAM: adapter}
    await asyncio.wait_for(spawned["owner_review_digest_watcher"](interval=0), 15)
    assert len(adapter.sent) == 2 and all(chat == HOME_CHAT for chat, _text in adapter.sent)
    assert sends == [0, 300]  # the retry waits five minutes of one-minute passes


@pytest.mark.asyncio
async def test_the_combined_digest_stays_within_1500_characters(root, ready):
    advice = "Meeting Plan: " + "Prepare the guest list. " * 7
    assert len(advice) <= 200
    with projects_db.connect_closing(root / "projects.db") as registry:
        registry.execute("UPDATE portfolio_review_briefs SET conflicts=?,priorities=?,shared_blockers=?",
                         (json.dumps([advice] * 5),) * 3)
        registry.commit()
    adapter = Adapter()
    runner = SimpleNamespace(_active_profile_name=lambda: "default", adapters={Platform.TELEGRAM: adapter})
    digest = importlib.import_module("gateway.owner_review_digest")
    assert await digest.tick(runner) == 1
    assert len(adapter.sent[0][1]) == 1500 and adapter.sent[0][1].endswith("…")


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", [
    TimeoutError("synthetic transport detail"), SimpleNamespace(success=False, retryable=False),
    SimpleNamespace(success=False, retryable="true"), SimpleNamespace(success=None, retryable=True),
    SimpleNamespace(success=1, retryable=True), None,
])
async def test_uncertain_transport_results_never_replay_after_reload(root, ready, outcome, caplog):
    adapter = Adapter([outcome])
    runner = SimpleNamespace(_active_profile_name=lambda: "default", adapters={Platform.TELEGRAM: adapter})
    digest = importlib.import_module("gateway.owner_review_digest")
    assert await digest.tick(runner) == 0
    digest = importlib.reload(digest)
    assert await digest.tick(runner) == 0
    assert len(adapter.sent) == 1
    assert "synthetic transport detail" not in caplog.text


@pytest.mark.asyncio
@pytest.mark.parametrize("state", ["sending", "sent", "paused", "unknown", "failed"])
async def test_a_durable_told_day_or_exhausted_retry_stays_unsent(root, ready, state):
    record = root / "owner_review_digest.json"
    content = json.dumps({"told": {"2026-11-10": {"state": state, "attempts": 3, "at": int(time.time())}}})
    record.write_text(content)
    adapter = Adapter()
    runner = SimpleNamespace(_active_profile_name=lambda: "default", adapters={Platform.TELEGRAM: adapter})
    digest = importlib.import_module("gateway.owner_review_digest")
    assert await digest.tick(runner) == 0
    assert adapter.sent == [] and record.read_text() == content


@pytest.mark.asyncio
@pytest.mark.parametrize("name", ["owner_review_digest.json", "owner_heads_up.json"])
async def test_an_unreadable_record_blocks_delivery_and_keeps_its_bytes(root, ready, name, monkeypatch):
    record = root / name
    record.write_text('{"told":{}}')
    read_text = Path.read_text

    def refused_read(path, *args, **kwargs):
        # A chmod 0 file stays readable for root, so the refused read is patched in.
        if path.name == name:
            raise PermissionError(13, "Permission denied", str(path))
        return read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", refused_read)
    adapter = Adapter()
    digest = importlib.import_module("gateway.owner_review_digest")
    assert await digest.tick(_runner(adapter)) == 0
    assert adapter.sent == []
    assert record.read_bytes() == b'{"told":{}}'
    assert name == "owner_review_digest.json" or not (root / "owner_review_digest.json").exists()


@pytest.mark.asyncio
async def test_sending_is_durable_before_transport_and_a_lost_receipt_never_replays(root, ready, monkeypatch):
    from gateway import owner_heads_up

    record = root / "owner_review_digest.json"
    observed = []

    class Observing(Adapter):
        async def send(self, chat, text):
            observed.append(json.loads(record.read_text())["told"]["2026-11-10"]["state"])
            return await super().send(chat, text)

    write_record = owner_heads_up._write_record
    writes = []

    def receipt_lost(path, told):
        # A read-only home does not stop root, so the second write (the receipt) fails through a patch.
        writes.append(dict(told))
        if len(writes) == 2:
            raise OSError("synthetic write failure")
        write_record(path, told)

    monkeypatch.setattr(owner_heads_up, "_write_record", receipt_lost)
    adapter = Observing()
    runner = _runner(adapter)
    digest = importlib.import_module("gateway.owner_review_digest")
    await digest.tick(runner)
    assert observed == ["sending"] and len(adapter.sent) == 1 and len(writes) == 2
    assert json.loads(record.read_text())["told"]["2026-11-10"]["state"] == "sending"
    monkeypatch.setattr(owner_heads_up, "_write_record", write_record)
    assert await importlib.reload(digest).tick(runner) == 0
    assert len(adapter.sent) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("reason", ["quiet", "single_project_only", "wrong_scope"])
async def test_nothing_to_report_or_unproven_portfolio_scope_sends_nothing(root, ready, reason):
    with projects_db.connect_closing(root / "projects.db") as registry:
        if reason == "single_project_only":
            registry.execute("DELETE FROM portfolio_review_briefs")
        elif reason == "wrong_scope":
            registry.execute("UPDATE portfolio_review_briefs SET project_scope=?", (json.dumps(["unowned-project"]),))
        else:
            registry.execute("UPDATE project_review_briefs SET quiet=1")
            registry.execute("UPDATE portfolio_review_briefs SET priorities='[]'")
        registry.commit()
    adapter = Adapter()
    runner = SimpleNamespace(_active_profile_name=lambda: "default", adapters={Platform.TELEGRAM: adapter})
    digest = importlib.import_module("gateway.owner_review_digest")
    assert await digest.tick(runner) == 0
    assert adapter.sent == [] and not (root / "owner_review_digest.json").exists()


@pytest.mark.asyncio
async def test_delivery_preserves_project_data_schema_main_files_and_configuration(root, ready):
    def snapshot():
        result = {}
        for path in root.rglob("*"):
            if not path.is_file() or path.name.endswith(("-wal", "-shm")) or path.name.startswith("owner_review_digest."):
                continue
            entry = [hashlib.sha256(path.read_bytes()).hexdigest(), path.stat().st_mtime_ns, path.stat().st_size]
            if path.suffix == ".db":
                conn = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)
                try:
                    entry.append(hashlib.sha256("\n".join(conn.iterdump()).encode()).hexdigest())
                finally:
                    conn.close()
            result[str(path.relative_to(root))] = entry
        return result

    before = snapshot()
    adapter = Adapter()
    runner = SimpleNamespace(_active_profile_name=lambda: "default", adapters={Platform.TELEGRAM: adapter})
    digest = importlib.import_module("gateway.owner_review_digest")
    assert await digest.tick(runner) == 1
    assert snapshot() == before


def _contending_sender(start, release, reports):
    class HoldingAdapter(Adapter):
        async def send(self, chat, text):
            reports.put(("sending", 0))
            assert release.wait(15)
            return await super().send(chat, text)

    assert start.wait(15)
    runner = SimpleNamespace(_active_profile_name=lambda: "default", adapters={Platform.TELEGRAM: HoldingAdapter()})
    digest = importlib.import_module("gateway.owner_review_digest")
    reports.put(("done", asyncio.run(digest.tick(runner))))


def test_two_gateway_processes_share_one_daily_delivery_lock(root, ready):
    ctx = multiprocessing.get_context("fork")
    start, release, reports = ctx.Event(), ctx.Event(), ctx.Queue()
    workers = [ctx.Process(target=_contending_sender, args=(start, release, reports)) for _ in range(2)]
    try:
        for worker in workers:
            worker.start()
        start.set()
        first = [reports.get(timeout=15), reports.get(timeout=15)]
        assert sorted(first) == [("done", 0), ("sending", 0)]
        release.set()
        assert reports.get(timeout=15) == ("done", 1)
        for worker in workers:
            worker.join(15)
            assert worker.exitcode == 0
    finally:
        release.set()
        for worker in workers:
            if worker.is_alive():
                worker.terminate()
            worker.join(5)
        reports.close()


@pytest.mark.asyncio
async def test_default_owner_delivery_uses_root_stores_and_home_under_worker_pins(root, ready, monkeypatch):
    worker = root / "profiles" / "planner"
    worker.mkdir(parents=True)
    worker_config = yaml.safe_dump({"gateway": {"api_server": {"owner_workspace": {"enabled": False}}},
                                   "owner_review": {"digest": False},
                                   "platforms": {"telegram": {"home_channel": {"chat_id": "5550000000002"}}}})
    (worker / "config.yaml").write_text(worker_config)
    monkeypatch.setenv("HERMES_HOME", str(worker))
    monkeypatch.setenv("HERMES_KANBAN_DB", str(worker / "decoy.db"))
    monkeypatch.setenv("HERMES_KANBAN_BOARD", "decoy")
    monkeypatch.setenv("TELEGRAM_HOME_CHANNEL", "5550000000003")
    adapter = Adapter()
    runner = SimpleNamespace(_active_profile_name=lambda: "default", adapters={Platform.TELEGRAM: adapter})
    digest = importlib.import_module("gateway.owner_review_digest")
    assert await digest.tick(runner) == 1
    assert len(adapter.sent) == 1 and adapter.sent[0][0] == HOME_CHAT
    assert (root / "owner_review_digest.json").is_file()
    assert sorted(path.name for path in worker.iterdir()) == ["config.yaml"]
    assert (worker / "config.yaml").read_text() == worker_config


@pytest.mark.asyncio
@pytest.mark.parametrize("content", [
    "not json", "[]", '{"told":[]}', '{"told":{"2026-11-10":{}}}',
    '{"told":{"not-a-day":{"state":"sent","attempts":0,"at":1794292200}}}',
    '{"told":{"2026-11-10":{"state":"failed","attempts":-1,"at":1794292200}}}',
    '{"told":{"2026-11-09":{"state":"failed","attempts":4,"at":1794292200}}}',
    '{"told":{"2026-11-09":{"state":"sent","attempts":0,"at":NaN}}}',
])
async def test_a_malformed_durable_record_never_becomes_an_empty_send_history(root, ready, content):
    record = root / "owner_review_digest.json"
    record.write_text(content)
    adapter = Adapter()
    runner = SimpleNamespace(_active_profile_name=lambda: "default", adapters={Platform.TELEGRAM: adapter})
    digest = importlib.import_module("gateway.owner_review_digest")
    assert await digest.tick(runner) == 0
    assert adapter.sent == []
    assert record.read_text() == content


@pytest.mark.asyncio
@pytest.mark.parametrize("closed", [
    "absent", "disabled", "portfolio_off", "digest_off", "string_flag", "empty_allowlist", "other_allowlist",
    "workspace_off", "worker", "no_home", "no_adapter",
])
async def test_closed_delivery_gates_leave_the_day_available_without_a_send(root, ready, closed):
    config = yaml.safe_load((root / "config.yaml").read_text())
    if closed == "absent":
        del config["owner_review"]
    elif closed in ("disabled", "portfolio_off", "digest_off", "string_flag"):
        key = {"disabled": "enabled", "portfolio_off": "portfolio"}.get(closed, "digest")
        config["owner_review"][key] = "true" if closed == "string_flag" else False
    elif closed.endswith("allowlist"):
        config["owner_review"]["allowlist"] = [] if closed == "empty_allowlist" else ["another-project"]
    elif closed == "workspace_off":
        config["gateway"]["api_server"]["owner_workspace"]["enabled"] = False
    elif closed == "no_home":
        config["platforms"]["telegram"] = {}
    (root / "config.yaml").write_text(yaml.safe_dump(config))
    adapter = Adapter()
    runner = SimpleNamespace(_active_profile_name=lambda: "worker" if closed == "worker" else "default",
                             adapters={} if closed == "no_adapter" else {Platform.TELEGRAM: adapter})
    digest = importlib.import_module("gateway.owner_review_digest")
    assert await digest.tick(runner) == 0
    assert adapter.sent == []
    assert not (root / "owner_review_digest.json").exists()


@pytest.mark.asyncio
async def test_the_digest_adds_headlines_and_only_unannounced_wait_counts_without_questions(root, ready):
    from gateway import owner_heads_up

    _waiting(ready, title="Choose a room", reason="Which room should we book?")
    heads_adapter = Adapter()
    runner = SimpleNamespace(_active_profile_name=lambda: "default", adapters={Platform.TELEGRAM: heads_adapter})
    assert await owner_heads_up.tick(runner) == 1
    _waiting(ready, title="Choose a meal", reason="Which meal should we serve?")
    adapter = Adapter()
    runner.adapters[Platform.TELEGRAM] = adapter
    digest = importlib.import_module("gateway.owner_review_digest")
    assert await digest.tick(runner) == 1
    text = adapter.sent[0][1]
    assert "Meeting Plan: The room is booked." in text
    assert "1 answer waiting on Decisions" in text
    assert "Which room" not in text and "Which meal" not in text


@pytest.mark.asyncio
@pytest.mark.parametrize("refusals", [1, 3])
async def test_only_explicit_retryable_refusals_get_at_most_three_attempts(root, ready, refusals, monkeypatch):
    clock = _clock(monkeypatch)
    digest = importlib.import_module("gateway.owner_review_digest")
    adapter = Adapter([SimpleNamespace(success=False, retryable=True)] * refusals)
    runner = SimpleNamespace(_active_profile_name=lambda: "default", adapters={Platform.TELEGRAM: adapter})
    sent = 0
    for _ in range(6):
        sent += await digest.tick(runner)
        # Reload the module to prove that transport attempts do not depend on process-local memory.
        digest = importlib.reload(digest)
        clock[0] += 15 * 60  # past every retry wait
    assert len(adapter.sent) == (2 if refusals == 1 else 3)
    assert sent == (1 if refusals == 1 else 0)


@pytest.mark.asyncio
async def test_t8_a_paused_daily_digest_stays_unsent_after_resume_but_the_next_day_is_delivered(root, monkeypatch):
    digest = importlib.import_module("gateway.owner_review_digest")
    clock = [_at(2026, 11, 9, 5)]
    monkeypatch.setattr(time, "time", lambda: clock[0])
    monkeypatch.setattr(time, "sleep", lambda seconds: clock.__setitem__(0, clock[0] + seconds))
    project = _project(root)
    config = yaml.safe_load((root / "config.yaml").read_text())
    config["owner_review"] = {"enabled": True, "portfolio": True, "digest": True, "allowlist": [project["project_id"]]}
    (root / "config.yaml").write_text(yaml.safe_dump(config))
    adapter = Adapter()
    runner = SimpleNamespace(_active_profile_name=lambda: "default", adapters={Platform.TELEGRAM: adapter})
    owner_channel_pause.pause()
    clock[0] = _at(2026, 11, 9, 6, 30)
    _ready(root, project, "2026-11-09", clock[0])
    assert await digest.tick(runner) == 0
    assert adapter.sent == []
    clock[0] = _at(2026, 11, 9, 9)
    owner_channel_pause.resume()
    assert await digest.tick(runner) == 0
    assert adapter.sent == []
    clock[0] = _at(2026, 11, 10, 6, 30)
    _ready(root, project, "2026-11-10", clock[0])
    assert await digest.tick(runner) == 1
    assert len(adapter.sent) == 1 and adapter.sent[0][0] == HOME_CHAT
    assert "Meeting Plan" in adapter.sent[0][1] and "Prepare the guest list." in adapter.sent[0][1]
    assert len(adapter.sent[0][1]) <= 1500
    assert await digest.tick(runner) == 0
    assert len(adapter.sent) == 1
