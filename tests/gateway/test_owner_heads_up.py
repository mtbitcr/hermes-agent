"""One Telegram heads-up to the owner when a card starts waiting (gateway/owner_heads_up.py).

Each test builds a real temporary root Hermes home whose config.yaml names HOME_CHAT as the
Telegram home and turns the owner workspace on, one real owner Project through
``owner_workspace.bootstrap``, and real cards on its board through the real kernel calls. A fake
Telegram adapter keeps every message it is handed (one test runs the real Telegram adapter over a
stub bot instead, so nothing contacts Telegram); each tick runs through
``asyncio.run(owner_heads_up.tick(runner))``. The module is imported inside each test, so on a tree
without it each test fails on its own. Chat ids, names and texts are invented placeholders.
"""

from __future__ import annotations

import asyncio
import contextlib
import importlib
import json
import re
import shutil
import sqlite3
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from gateway import owner_channel_pause
from gateway.config import Platform
from gateway.platforms.base import SendResult
from gateway.status import _release_file_lock, _try_acquire_file_lock
from hermes_cli import kanban_db as kb, owner_design_choices as odc, owner_workspace as ow, projects_db

HOME_CHAT = "5550000000001"
PROJECT = "Garden journal"
TITLE = "Write the welcome text"
REASON = "Which colour should the buttons be?"
QUESTION = "Which home page should Raphael build?"
SENTENCES = (
    "A calm layout with a large photo at the top.",
    "A bold layout with a big title and two buttons.",
)
ALTS = (
    "The home page with a wide photo above a short welcome.",
    "The home page with a large title and two buttons below it.",
)
PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 24


def _configure(home, *, enabled=True, chat_id=HOME_CHAT):
    """Write the root config.yaml: ``chat_id`` as the Telegram home (none when None) and the owner
    workspace switch."""
    telegram = {"home_channel": {"platform": "telegram", "chat_id": chat_id, "name": "Home"}}
    config = {
        "platforms": {"telegram": telegram if chat_id is not None else {}},
        "gateway": {"api_server": {"owner_workspace": {"enabled": enabled}}},
    }
    (home / "config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")


@pytest.fixture
def root(tmp_path, monkeypatch):
    """A real temporary root Hermes home served as the default profile."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    for key in (
        "HERMES_KANBAN_TASK", "HERMES_KANBAN_RUN_ID", "HERMES_KANBAN_DB",
        "HERMES_KANBAN_BOARD", "HERMES_KANBAN_ATTACHMENTS_ROOT", "HERMES_KANBAN_HOME",
    ):
        monkeypatch.delenv(key, raising=False)
    kb._INITIALIZED_PATHS.clear()
    _configure(home)
    return home


@pytest.fixture
def heads_up(root):
    return importlib.import_module("gateway.owner_heads_up")


@pytest.fixture
def owner(root):
    return ow.resolve_owner_context()


def _project(owner, name=PROJECT):
    """Commit one real owner Project."""
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(ow, "_confirm", lambda *_args, **_kwargs: {"approved": True})
        made = ow.bootstrap(owner, idempotency_key=f"setup-{name[:40]}", name=name)
    assert made["ok"] is True
    return made


@pytest.fixture
def project(owner):
    return _project(owner)


class _Telegram:
    """A Telegram adapter that keeps every message it is handed and returns ``result``, or raises ``error``."""

    def __init__(self, result=None, error=None):
        self.sent = []
        self.result = SendResult(success=True) if result is None else result
        self.error = error

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        self.sent.append((str(chat_id), content))
        if self.error is not None:
            raise self.error
        return self.result


@pytest.fixture
def telegram():
    return _Telegram()


def _runner(adapter):
    return SimpleNamespace(adapters={Platform.TELEGRAM: adapter}, _active_profile_name=lambda: "default")


@pytest.fixture
def runner(telegram):
    return _runner(telegram)


def _tick(heads_up, runner):
    return asyncio.run(heads_up.tick(runner))


def _board(project):
    return contextlib.closing(kb.connect(board=project["board"]))


def _block(conn, task_id, *, reason=REASON, kind="needs_input"):
    claimed = kb.claim_task(conn, task_id)
    assert claimed is not None
    assert kb.block_task(conn, task_id, reason=reason, kind=kind, expected_run_id=claimed.current_run_id)


def _waiting(project, title=TITLE, *, reason=REASON, kind="needs_input"):
    """A card of ``project`` that was claimed and then stopped with ``kind``."""
    with _board(project) as conn:
        task_id = kb.create_task(conn, title=title, assignee="writer", project_id=project["project_id"])
        _block(conn, task_id, reason=reason, kind=kind)
    return task_id


def _designer(project, title="Design the home page"):
    """A designer card of ``project`` that offered two real pictures as options, so it waits for a choice."""
    board = project["board"]
    with _board(project) as conn:
        task_id = kb.create_task(conn, title=title, assignee="designer", project_id=project["project_id"])
        claimed = kb.claim_task(conn, task_id)
        assert claimed is not None
        ids = [
            str(kb.store_attachment_bytes(
                conn, task_id, f"option-{n}.png", PNG, content_type="image/png", uploaded_by="agent", board=board,
            ))
            for n in (1, 2)
        ]
        options = [
            {"n": n, "sentence": SENTENCES[n - 1], "pictures": [{"attachment_id": ids[n - 1], "alt": ALTS[n - 1]}]}
            for n in (1, 2)
        ]
        assert odc.declare_design_options(
            conn, task_id, QUESTION, options, board=board, expected_run_id=claimed.current_run_id,
        )
    return task_id


def _key(project, task_id):
    """The record key of the card's current wait: Project id, card id and its newest blocked event id."""
    with _board(project) as conn:
        (event_id,) = conn.execute(
            "SELECT MAX(id) FROM task_events WHERE task_id = ? AND kind = 'blocked'", (task_id,),
        ).fetchone()
    return f"{project['project_id']}:{task_id}:{event_id}"


def _record(root):
    return root / "owner_heads_up.json"


def _told(root):
    return json.loads(_record(root).read_text(encoding="utf-8"))["told"]


def _states(root):
    return {key: entry["state"] for key, entry in _told(root).items()}


def _questions(telegram):
    return [text.split("\n")[1] for _chat, text in telegram.sent]


# --- what is told ----------------------------------------------------------------------------------

def test_a_design_choice_sends_one_heads_up(root, heads_up, project, telegram, runner):
    task = _designer(project)

    assert _tick(heads_up, runner) == 1

    assert telegram.sent == [(HOME_CHAT, (
        f"Raphael needs you to choose a design in {PROJECT}.\n"
        f"{QUESTION}\n"
        "The options are waiting for you on Decisions in the app."
    ))]
    assert _states(root) == {_key(project, task): "sent"}


def test_a_question_sends_one_heads_up(root, heads_up, project, telegram, runner):
    task = _waiting(project)

    assert _tick(heads_up, runner) == 1

    assert telegram.sent == [(HOME_CHAT, (
        f"Raphael needs your answer in {PROJECT}.\n"
        f"{REASON}\n"
        "Please answer on Decisions in the app."
    ))]
    entry = _told(root)[_key(project, task)]
    assert entry["state"] == "sent"
    assert abs(entry["at"] - time.time()) < 60


def test_a_capability_stop_sends_one_and_a_loose_card_or_a_transient_block_sends_nothing(
    root, heads_up, project, telegram, runner,
):
    _waiting(project, "Connect the shop", reason="Raphael needs access to the shop account.", kind="capability")
    with _board(project) as conn:
        loose = kb.create_task(conn, title="A card of no Project", assignee="writer")
        _block(conn, loose)
        brief = kb.create_task(conn, title="Retry the upload", assignee="writer", project_id=project["project_id"])
        _block(conn, brief, reason="The network dropped for a moment.", kind="transient")

    assert _tick(heads_up, runner) == 1

    assert _questions(telegram) == ["Raphael needs access to the shop account."]


def test_every_wait_found_at_the_first_start_gets_its_own_heads_up_once(root, heads_up, project, telegram, runner):
    reasons = ["Which colour should the buttons be?", "Which font should the title use?", "Is the logo final?"]
    for n, reason in enumerate(reasons):
        _waiting(project, f"Card {n}", reason=reason)
    assert not _record(root).exists()

    assert _tick(heads_up, runner) == 3
    assert sorted(_questions(telegram)) == sorted(reasons)

    assert _tick(importlib.reload(heads_up), runner) == 0
    assert len(telegram.sent) == 3


# --- the pause -------------------------------------------------------------------------------------

def test_nothing_while_paused_not_even_after_the_resume_and_a_new_wait_after_it_is_told(
    root, heads_up, project, telegram, runner,
):
    owner_channel_pause.pause()
    first = _waiting(project)

    assert _tick(heads_up, runner) == 0
    assert telegram.sent == []
    assert _states(root) == {_key(project, first): "paused"}

    owner_channel_pause.resume()
    assert _tick(heads_up, runner) == 0
    assert telegram.sent == []

    second = _waiting(project, "Write the about page", reason="Which photo goes on the about page?")
    assert _tick(heads_up, runner) == 1
    assert _questions(telegram) == ["Which photo goes on the about page?"]
    assert _states(root) == {_key(project, first): "paused", _key(project, second): "sent"}


def test_a_wait_that_started_inside_a_pause_window_is_never_told(root, heads_up, project, telegram, runner):
    owner_channel_pause.pause()
    task = _waiting(project)
    owner_channel_pause.resume()

    assert _tick(heads_up, runner) == 0
    assert _tick(heads_up, runner) == 0

    assert telegram.sent == []
    assert _states(root) == {_key(project, task): "paused"}


def test_every_wait_found_while_paused_is_recorded_paused_however_many(
    root, heads_up, project, telegram, runner, monkeypatch,
):
    class _Clock:
        """The kernel's own clock, an hour behind, so every wait starts before the pause."""

        def time(self):
            return time.time() - 3_600

        def __getattr__(self, name):
            return getattr(time, name)

    with monkeypatch.context() as patch:
        patch.setattr(kb, "time", _Clock())
        tasks = [_waiting(project, f"Card {n}") for n in range(12)]
    assert len(tasks) > heads_up.MAX_PER_TICK
    owner_channel_pause.pause()

    assert _tick(heads_up, runner) == 0
    assert _states(root) == {_key(project, task): "paused" for task in tasks}

    owner_channel_pause.resume()
    assert _tick(heads_up, runner) == 0
    assert telegram.sent == []


# --- no repeat -------------------------------------------------------------------------------------

def test_no_repeat_across_ticks_or_a_reload_of_the_module(root, heads_up, project, telegram, runner):
    _waiting(project)

    assert _tick(heads_up, runner) == 1
    assert _tick(heads_up, runner) == 0
    assert _tick(importlib.reload(heads_up), runner) == 0

    assert len(telegram.sent) == 1


def test_a_new_wait_after_an_answer_is_told_once_and_the_answered_one_is_dropped(
    root, heads_up, project, telegram, runner,
):
    task = _waiting(project)
    assert _tick(heads_up, runner) == 1
    answered = _key(project, task)

    with _board(project) as conn:
        assert kb.unblock_task(conn, task)
        _block(conn, task, reason="Which font should the title use?")
    assert _tick(heads_up, runner) == 1
    assert _tick(heads_up, runner) == 0

    assert _key(project, task) != answered
    assert _states(root) == {_key(project, task): "sent"}
    assert _questions(telegram) == [REASON, "Which font should the title use?"]


def test_a_new_wait_after_a_design_choice_is_told_once(root, heads_up, owner, project, telegram, runner):
    task = _designer(project)
    assert _tick(heads_up, runner) == 1

    with _board(project) as conn:
        assert odc.accept_choice(conn, task, 1, actor=owner.actor) == "released"
        _block(conn, task, reason="Which photo goes on the home page?")
    assert _tick(heads_up, runner) == 1
    assert _tick(heads_up, runner) == 0

    assert _questions(telegram) == [QUESTION, "Which photo goes on the home page?"]
    assert _states(root) == {_key(project, task): "sent"}


def test_a_wait_found_sending_counts_as_told(root, heads_up, project, telegram, runner):
    task = _waiting(project)
    key = _key(project, task)
    _record(root).write_text(json.dumps({"told": {key: {"state": "sending", "attempts": 0, "at": 1}}}), encoding="utf-8")

    assert _tick(heads_up, runner) == 0

    assert telegram.sent == []
    assert _states(root) == {key: "sending"}


def test_another_sender_holding_the_lock_stops_the_tick(root, heads_up, project, telegram, runner):
    _waiting(project)

    with open(root / "owner_heads_up.lock", "a+", encoding="utf-8") as other:
        assert _try_acquire_file_lock(other)
        try:
            assert _tick(heads_up, runner) == 0
        finally:
            _release_file_lock(other)
    assert telegram.sent == []
    assert not _record(root).exists()

    assert _tick(heads_up, runner) == 1


def test_at_most_ten_a_tick_oldest_first(root, heads_up, project, telegram, runner, monkeypatch):
    class _Clock:
        """The kernel's own clock, pinned so each wait starts at a chosen second."""

        now = int(time.time()) - 3_600

        def time(self):
            return float(self.now)

        def __getattr__(self, name):
            return getattr(time, name)

    clock = _Clock()
    monkeypatch.setattr(kb, "time", clock)
    start = clock.now
    reasons = [f"Which picture goes on page {n}?" for n in range(12)]
    for n, reason in enumerate(reasons):
        clock.now = start + 100 - n  # each later card started waiting one second earlier
        _waiting(project, f"Card {n}", reason=reason)

    assert _tick(heads_up, runner) == heads_up.MAX_PER_TICK == 10
    assert _questions(telegram) == reasons[::-1][:10]

    assert _tick(heads_up, runner) == 2
    assert _questions(telegram) == reasons[::-1]


def test_an_archived_project_is_not_told(root, heads_up, project, telegram, runner):
    _waiting(project)
    with projects_db.connect_closing() as conn:
        assert projects_db.archive_project(conn, project["project_id"])

    assert _tick(heads_up, runner) == 0

    assert telegram.sent == []


def test_a_board_gone_with_a_completed_removal_is_not_read_and_keeps_its_keys(
    root, heads_up, owner, project, telegram, runner, monkeypatch,
):
    task = _waiting(project)
    assert _tick(heads_up, runner) == 1
    key = _key(project, task)
    shutil.rmtree(kb.board_dir(project["board"]))
    monkeypatch.setattr(
        ow, "_completed_removal_state",
        lambda project_id: {"phase": "done"} if project_id == project["project_id"] else None,
    )
    other = _project(owner, "Recipe book")
    _waiting(other, reason="Which recipe comes first?")
    opened = []
    real = ow._open_read_only_sqlite

    def watched(path, *, label):
        opened.append(Path(path))
        return real(path, label=label)

    monkeypatch.setattr(ow, "_open_read_only_sqlite", watched)

    assert _tick(heads_up, runner) == 1

    assert [path for path in opened if path.name == "kanban.db"] == [kb.board_dir(other["board"]) / "kanban.db"]
    assert _questions(telegram) == [REASON, "Which recipe comes first?"]
    assert _states(root)[key] == "sent"


def test_a_board_that_fails_to_read_keeps_its_keys_and_the_other_boards_are_still_told(
    root, heads_up, owner, project, telegram, runner, monkeypatch,
):
    task = _waiting(project)
    assert _tick(heads_up, runner) == 1
    key = _key(project, task)
    with _board(project) as conn:
        assert kb.unblock_task(conn, task)
    other = _project(owner, "Recipe book")
    _waiting(other, reason="Which recipe comes first?")

    real = ow._open_read_only_sqlite

    def first_board_unreadable(path, *, label):
        if Path(path).parent == kb.board_dir(project["board"]):
            raise sqlite3.OperationalError("unable to open database file")
        return real(path, label=label)

    monkeypatch.setattr(ow, "_open_read_only_sqlite", first_board_unreadable)
    assert _tick(heads_up, runner) == 1
    assert _questions(telegram) == [REASON, "Which recipe comes first?"]
    assert _states(root)[key] == "sent"

    monkeypatch.setattr(ow, "_open_read_only_sqlite", real)
    assert _tick(heads_up, runner) == 0
    assert key not in _told(root)
    assert len(telegram.sent) == 2


def test_a_failed_event_read_skips_the_board_and_keeps_its_told_keys(
    root, heads_up, project, telegram, runner, monkeypatch,
):
    task = _waiting(project)
    assert _tick(heads_up, runner) == 1
    key = _key(project, task)
    real = ow._open_read_only_sqlite
    denied = []

    def event_payloads_denied(path, *, label):
        """The real board connection, with SQLite itself refusing every read of task_events.payload."""
        conn = real(path, label=label)
        if Path(path).name == "kanban.db":
            def authorize(action, table, column, database, trigger):
                if action == sqlite3.SQLITE_READ and (table, column) == ("task_events", "payload"):
                    denied.append((table, column))
                    return sqlite3.SQLITE_DENY
                return sqlite3.SQLITE_OK

            conn.set_authorizer(authorize)
        return conn

    with monkeypatch.context() as patch:
        patch.setattr(ow, "_open_read_only_sqlite", event_payloads_denied)
        assert _tick(heads_up, runner) == 0
    assert denied
    assert _states(root) == {key: "sent"}

    assert _tick(heads_up, runner) == 0
    assert len(telegram.sent) == 1
    assert _states(root) == {key: "sent"}


@pytest.mark.parametrize("fails_in", ["execute", "fetch"])
def test_a_failed_nested_wait_read_skips_the_board_rather_than_telling_a_design_wait_as_a_question(
    root, heads_up, project, telegram, runner, monkeypatch, fails_in,
):
    task = _designer(project)
    real = ow._open_read_only_sqlite
    wait_reads = []

    class _Rows:
        """The real cursor, except that its rows cannot be fetched."""

        def __init__(self, cursor):
            self.cursor = cursor

        def __getattr__(self, name):
            return getattr(self.cursor, name)

        def fetchall(self):
            raise sqlite3.OperationalError("disk I/O error")

    class _Board:
        """The real board connection, except that the second read of the wait projection, the one
        decision_design_options makes inside, fails in ``execute`` or while its rows are fetched."""

        def __init__(self, conn):
            self.conn = conn

        def __getattr__(self, name):
            return getattr(self.conn, name)

        def execute(self, sql, *args):
            if sql.startswith("SELECT task_id, id, kind, created_at, payload FROM task_events"):
                wait_reads.append(sql)
                if len(wait_reads) == 2:
                    if fails_in == "execute":
                        raise sqlite3.OperationalError("disk I/O error")
                    return _Rows(self.conn.execute(sql, *args))
            return self.conn.execute(sql, *args)

    def second_wait_read_fails(path, *, label):
        conn = real(path, label=label)
        return _Board(conn) if Path(path).name == "kanban.db" else conn

    with monkeypatch.context() as patch:
        patch.setattr(ow, "_open_read_only_sqlite", second_wait_read_fails)
        _tick(heads_up, runner)
    assert len(wait_reads) == 2
    assert telegram.sent == []
    assert not _record(root).exists()

    assert _tick(heads_up, runner) == 1
    assert telegram.sent == [(HOME_CHAT, (
        f"Raphael needs you to choose a design in {PROJECT}.\n"
        f"{QUESTION}\n"
        "The options are waiting for you on Decisions in the app."
    ))]
    assert _states(root) == {_key(project, task): "sent"}


def test_a_projects_list_that_fails_stops_the_tick(root, heads_up, project, telegram, runner, monkeypatch):
    _waiting(project)
    real = ow.list_committed_projects

    def failing(ctx):
        raise ow.OwnerWorkspaceError("projects_unavailable", "The Projects list cannot be read.")

    monkeypatch.setattr(ow, "list_committed_projects", failing)
    assert _tick(heads_up, runner) == 0
    assert telegram.sent == []
    assert not _record(root).exists()

    monkeypatch.setattr(ow, "list_committed_projects", real)
    assert _tick(heads_up, runner) == 1


# --- content safety --------------------------------------------------------------------------------

def test_the_heads_up_is_short_and_carries_no_internal_detail(root, heads_up, owner, telegram, runner):
    name = "Garden journal " + "g" * 145
    assert len(name) == 160
    project = _project(owner, name)
    reason = ("Please choose the colour of the buttons on the welcome page. " * 90)[:5000]
    task = _waiting(project, reason=reason)

    assert _tick(heads_up, runner) == 1

    [(_chat, text)] = telegram.sent
    first, _question, last = text.split("\n")
    assert len(text) < 4000
    assert first == f"Raphael needs your answer in {name}."
    assert last == "Please answer on Decisions in the app."
    with _board(project) as conn:
        runs = [str(row[0]) for row in conn.execute("SELECT id FROM task_runs WHERE task_id = ?", (task,))]
    assert runs
    for detail in (task, project["project_id"], project["board"], str(root), "hermes", *[f"run {r}" for r in runs]):
        assert detail not in text
    assert not re.search(r"\d|/|\\|https?:|www\.", text)


def test_an_overlong_question_is_cut_to_fit_and_keeps_the_name_and_decisions(
    root, heads_up, project, telegram, runner, monkeypatch,
):
    monkeypatch.setattr(ow, "owner_stop_reason", lambda value, **_kwargs: "word " * 1000)
    _waiting(project)

    assert _tick(heads_up, runner) == 1

    [(_chat, text)] = telegram.sent
    first, question, last = text.split("\n")
    assert len(text) <= heads_up.MAX_MESSAGE == 4000
    assert first == f"Raphael needs your answer in {PROJECT}."
    assert question.endswith("…") and ("word " * 1000).startswith(question[:-1])
    assert last == "Please answer on Decisions in the app."


# --- send failures ---------------------------------------------------------------------------------

def test_a_send_that_raises_changes_no_card_and_is_never_tried_again(root, heads_up, owner, project):
    telegram = _Telegram(error=ConnectionError("Telegram is unreachable"))
    runner = _runner(telegram)
    task = _waiting(project)
    with _board(project) as conn:
        (events,) = conn.execute("SELECT COUNT(*) FROM task_events").fetchone()

    assert _tick(heads_up, runner) == 0
    assert _tick(heads_up, runner) == 0
    assert _tick(heads_up, runner) == 0

    assert len(telegram.sent) == 1
    assert _states(root) == {_key(project, task): "unknown"}
    with _board(project) as conn:
        assert conn.execute("SELECT status FROM tasks WHERE id = ?", (task,)).fetchone()[0] == "blocked"
        assert conn.execute("SELECT COUNT(*) FROM task_events").fetchone()[0] == events
    assert TITLE in [item["title"] for item in ow.list_owner_decisions(owner)["data"]]


def test_the_record_says_sending_before_the_message_goes_out(root, heads_up, project):
    seen = []

    class _Watching(_Telegram):
        async def send(self, chat_id, content, reply_to=None, metadata=None):
            seen.append(_states(root))
            return await super().send(chat_id, content, reply_to, metadata)

    task = _waiting(project)

    assert _tick(heads_up, _runner(_Watching())) == 1

    assert seen == [{_key(project, task): "sending"}]
    assert _states(root) == {_key(project, task): "sent"}


def test_a_refused_send_is_tried_at_most_three_times(root, heads_up, project):
    telegram = _Telegram(result=SendResult(success=False, error="Bad Request: chat not found", retryable=True))
    runner = _runner(telegram)
    task = _waiting(project)

    assert [_tick(heads_up, runner) for _ in range(5)] == [0] * 5

    assert len(telegram.sent) == 3
    entry = _told(root)[_key(project, task)]
    assert (entry["state"], entry["attempts"]) == ("failed", 3)


def test_a_refused_send_that_then_goes_through_is_sent(root, heads_up, project):
    telegram = _Telegram(result=SendResult(success=False, error="Too Many Requests", retryable=True))
    runner = _runner(telegram)
    task = _waiting(project)
    assert _tick(heads_up, runner) == 0

    telegram.result = SendResult(success=True)
    assert _tick(heads_up, runner) == 1
    assert _tick(heads_up, runner) == 0

    assert len(telegram.sent) == 2
    assert _states(root) == {_key(project, task): "sent"}


def test_a_refused_send_that_is_not_retryable_is_never_tried_again(root, heads_up, project):
    telegram = _Telegram(result=SendResult(success=False, error="Timed out", retryable=False, error_kind="unknown"))
    runner = _runner(telegram)
    task = _waiting(project)

    assert [_tick(heads_up, runner) for _ in range(3)] == [0] * 3

    assert len(telegram.sent) == 1
    assert _states(root) == {_key(project, task): "unknown"}


def test_a_telegram_timeout_that_may_have_arrived_is_never_sent_again(root, heads_up, project):
    from unittest.mock import AsyncMock, MagicMock

    from telegram.error import TimedOut

    from gateway.config import PlatformConfig
    from plugins.platforms.telegram.adapter import TelegramAdapter

    task = _waiting(project)
    accepted = []

    async def accepted_then_the_answer_is_lost(**kwargs):
        accepted.append(kwargs)
        if len(accepted) == 1:
            raise TimedOut("timed out")
        return SimpleNamespace(message_id=42)

    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="placeholder-token"))
    adapter._rich_send_disabled = True
    adapter._retrigger_typing = AsyncMock()
    adapter._bot = MagicMock(send_message=AsyncMock(side_effect=accepted_then_the_answer_is_lost))
    results = []
    real_send = adapter.send

    async def send(*args, **kwargs):
        results.append(await real_send(*args, **kwargs))
        return results[-1]

    adapter.send = send
    runner = _runner(adapter)

    ticks = [_tick(heads_up, runner) for _ in range(3)]

    assert len(accepted) == 1
    assert [(result.success, result.retryable) for result in results] == [(False, False)]
    assert _states(root) == {_key(project, task): "unknown"}
    assert ticks == [0, 0, 0]


# --- gates and the record --------------------------------------------------------------------------

_GATES = {
    "profile not default": lambda root, runner: setattr(runner, "_active_profile_name", lambda: "coder"),
    "owner workspace off": lambda root, runner: _configure(root, enabled=False),
    "owner workspace the string true": lambda root, runner: _configure(root, enabled="true"),
    "no home chat": lambda root, runner: _configure(root, chat_id=None),
    "blank home chat": lambda root, runner: _configure(root, chat_id="   "),
    "no Telegram adapter": lambda root, runner: runner.adapters.clear(),
}


@pytest.mark.parametrize("close_gate", list(_GATES.values()), ids=list(_GATES))
def test_a_closed_gate_sends_nothing_and_writes_no_record(root, heads_up, project, telegram, runner, close_gate):
    _waiting(project)
    close_gate(root, runner)

    assert _tick(heads_up, runner) == 0

    assert telegram.sent == []
    assert not _record(root).exists()


@pytest.mark.parametrize("content", [
    '{"told": ',
    "[]",
    '{"told": []}',
    '{"told": {"a:b:1": "sent"}}',
    '{"told": {"a:b:1": {"state": "told", "attempts": 0, "at": 1}}}',
])
def test_a_malformed_record_sends_nothing_and_is_left_as_it_is(root, heads_up, project, telegram, runner, content):
    _waiting(project)
    _record(root).write_text(content, encoding="utf-8")

    assert _tick(heads_up, runner) == 0

    assert telegram.sent == []
    assert _record(root).read_text(encoding="utf-8") == content


def test_a_record_that_cannot_be_read_sends_nothing(root, heads_up, project, telegram, runner):
    _waiting(project)
    _record(root).mkdir()

    assert _tick(heads_up, runner) == 0

    assert telegram.sent == []
    assert _record(root).is_dir()


def test_a_missing_record_is_a_normal_first_send(root, heads_up, project, telegram, runner):
    _waiting(project)
    assert not _record(root).exists()

    assert _tick(heads_up, runner) == 1
    assert len(telegram.sent) == 1


# --- the watcher -----------------------------------------------------------------------------------

def test_the_watcher_calls_tick_again_after_it_raises(heads_up, monkeypatch):
    from gateway.run_watchers import GatewaySessionWatchersMixin

    runner = SimpleNamespace(_running=True)
    calls = []

    async def tick(passed):
        calls.append(passed)
        if len(calls) == 1:
            raise RuntimeError("placeholder failure")
        passed._running = False
        return 0

    monkeypatch.setattr(heads_up, "tick", tick)

    asyncio.run(asyncio.wait_for(GatewaySessionWatchersMixin._owner_heads_up_watcher(runner, interval=0), 10))

    assert calls == [runner, runner]


def test_the_gateway_starts_the_watcher_supervised():
    from gateway.run import GatewayRunner

    runner = GatewayRunner.__new__(GatewayRunner)
    spawned = {}
    runner._spawn_supervised = lambda factory, name, **_kwargs: spawned.setdefault(name, factory)
    runner._failed_platforms = {}
    runner._spawn_reconnect_watcher = lambda: None
    runner._scale_to_zero_should_arm = lambda: False
    runner._log_scale_to_zero_not_armed_reason = lambda: None

    runner._start_spawn_background_watchers()

    assert spawned["owner_heads_up_watcher"] == runner._owner_heads_up_watcher
