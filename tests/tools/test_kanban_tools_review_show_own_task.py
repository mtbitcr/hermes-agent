"""A review or portfolio task's ``kanban_show`` answers only for its own task.

The summary board holds every project's review tasks and frozen records, so a
review worker must not read another card, on its board or on another board.
Real board DB files under ``tmp_path``, ``HERMES_HOME`` isolated, calls through
the registered handler under the env the dispatcher injects.
"""

from __future__ import annotations

import json
import os
import sqlite3
from contextlib import closing
from pathlib import Path

import pytest

REVIEW_BOARD = "daily-project-review"
OTHER_BOARD = "alpha-board"
SECRET = "alpha confidential frozen record"


@pytest.fixture
def world(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    for key in [k for k in os.environ if k.startswith("HERMES_KANBAN_")]:
        monkeypatch.delenv(key)
    monkeypatch.delenv("HERMES_PROFILE", raising=False)
    monkeypatch.delenv("HERMES_SESSION_ID", raising=False)

    from hermes_cli import kanban_db as kb
    import tools.kanban_tools  # noqa: F401 - registers the tools

    kb._INITIALIZED_PATHS.clear()
    kb.create_board(REVIEW_BOARD)
    kb.create_board(OTHER_BOARD)

    def make(board, title, created_by, key=None):
        with kb.connect_closing(board=board) as conn:
            tid = kb.create_task(
                conn, title=title, body=SECRET, assignee="raphael",
                created_by=created_by, idempotency_key=key, board=board,
            )
            kb.add_comment(conn, tid, "someone", f"note on {title}")
            return tid

    ids = {
        "review": make(REVIEW_BOARD, "review of alpha", kb.REVIEW_TASK_CREATOR, "project-review:prj_alpha:2026-11-09:0"),
        "review_peer": make(REVIEW_BOARD, "review of beta", kb.REVIEW_TASK_CREATOR, "project-review:prj_beta:2026-11-09:0"),
        "plain": make(REVIEW_BOARD, "plain card", "default"),
        "other_board": make(OTHER_BOARD, "alpha work card", "default"),
        "plain_other": make(OTHER_BOARD, "alpha second card", "default"),
    }
    return kb, ids


def _become_worker(monkeypatch, kb, tid, board=REVIEW_BOARD, *, pin_db=True):
    monkeypatch.setenv("HERMES_KANBAN_TASK", tid)
    monkeypatch.setenv("HERMES_KANBAN_BOARD", board)
    if pin_db:
        monkeypatch.setenv("HERMES_KANBAN_DB", str(kb.board_dir(board) / "kanban.db"))


def _show(**args) -> dict:
    from tools.registry import registry

    return json.loads(registry.get_entry("kanban_show").handler(args))


def _assert_refused(out: dict, *hidden: str) -> None:
    error = out.get("error") or ""
    assert out.get("ok") is not True and "task" not in out, out
    assert error.startswith("kanban_show") and "refused" in error and "operator rule S1" in error, out
    for text in hidden:
        assert text not in json.dumps(out), out


def _all_rows(root: Path) -> dict:
    rows = {}
    for db in sorted(root.rglob("kanban.db")):
        with closing(sqlite3.connect(f"{db.resolve().as_uri()}?mode=ro", uri=True)) as conn:
            tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table' ORDER BY name")]
            for t in tables:
                rows[(str(db), t)] = sorted(repr(r) for r in conn.execute(f'SELECT * FROM "{t}"'))
    assert rows
    return rows


PINNED = pytest.mark.parametrize("pin_db", [True, False], ids=["db-pinned", "db-unpinned"])


@PINNED
@pytest.mark.parametrize("explicit", [False, True], ids=["implicit-id", "explicit-id"])
def test_review_task_reads_its_own_card(world, monkeypatch, pin_db, explicit):
    kb, ids = world
    _become_worker(monkeypatch, kb, ids["review"], pin_db=pin_db)
    out = _show(task_id=ids["review"]) if explicit else _show()
    assert out["task"]["id"] == ids["review"], out
    assert out["task"]["title"] == "review of alpha"


@PINNED
@pytest.mark.parametrize("peer", ["review_peer", "plain"])
def test_review_task_is_refused_for_another_card_on_its_board(world, monkeypatch, pin_db, peer):
    kb, ids = world
    _become_worker(monkeypatch, kb, ids["review"], pin_db=pin_db)
    out = _show(task_id=ids[peer])
    _assert_refused(out, ids[peer], "review of beta", "plain card", SECRET)


@PINNED
def test_review_task_is_refused_for_a_card_on_another_board(world, monkeypatch, pin_db):
    kb, ids = world
    _become_worker(monkeypatch, kb, ids["review"], pin_db=pin_db)
    out = _show(task_id=ids["other_board"], board=OTHER_BOARD)
    _assert_refused(out, ids["other_board"], "alpha work card", SECRET)


@PINNED
@pytest.mark.parametrize("board", [REVIEW_BOARD, OTHER_BOARD])
@pytest.mark.parametrize("explicit", [False, True], ids=["implicit-id", "explicit-id"])
def test_review_task_is_refused_for_its_own_card_with_a_board_argument(world, monkeypatch, pin_db, board, explicit):
    kb, ids = world
    _become_worker(monkeypatch, kb, ids["review"], pin_db=pin_db)
    args = {"task_id": ids["review"]} if explicit else {}
    out = _show(board=board, **args)
    _assert_refused(out, "review of alpha", SECRET)


def test_refusal_is_identical_for_an_existing_and_a_missing_card(world, monkeypatch):
    kb, ids = world
    _become_worker(monkeypatch, kb, ids["review"])
    existing = _show(task_id=ids["plain"])
    missing = _show(task_id="t_does_not_exist")
    other_missing = _show(task_id="t_does_not_exist", board="no-such-board")
    _assert_refused(existing)
    assert existing == missing == other_missing


def test_portfolio_task_is_held_the_same_way(world, monkeypatch):
    kb, ids = world
    with kb.connect_closing(board=REVIEW_BOARD) as conn:
        portfolio = kb.create_task(
            conn, title="portfolio", assignee="raphael", created_by=kb.REVIEW_TASK_CREATOR,
            idempotency_key="project-review:portfolio:2026-11-09:0", board=REVIEW_BOARD,
        )
    _become_worker(monkeypatch, kb, portfolio)
    assert _show()["task"]["id"] == portfolio
    _assert_refused(_show(task_id=ids["review"]), ids["review"])


@PINNED
def test_normal_worker_reads_other_cards_and_boards_as_before(world, monkeypatch, pin_db):
    kb, ids = world
    _become_worker(monkeypatch, kb, ids["plain"], pin_db=pin_db)
    assert _show()["task"]["id"] == ids["plain"]
    assert _show(task_id=ids["review"])["task"]["title"] == "review of alpha"
    assert _show(task_id=ids["review_peer"])["task"]["id"] == ids["review_peer"]
    assert _show(task_id=ids["plain"], board=REVIEW_BOARD)["task"]["id"] == ids["plain"]
    if not pin_db:
        # With the path pinned, kb.connect ignores the board argument, as it always did.
        other = _show(task_id=ids["other_board"], board=OTHER_BOARD)
        assert other["task"]["title"] == "alpha work card", other
    missing = _show(task_id="t_does_not_exist")
    assert missing["error"] == "task t_does_not_exist not found"


def test_call_without_a_worker_task_behaves_as_before(world, monkeypatch):
    kb, ids = world
    monkeypatch.setenv("HERMES_KANBAN_BOARD", REVIEW_BOARD)
    assert _show(task_id=ids["review"])["task"]["id"] == ids["review"]
    assert _show(task_id=ids["review_peer"], board=REVIEW_BOARD)["task"]["id"] == ids["review_peer"]
    assert _show(task_id=ids["other_board"], board=OTHER_BOARD)["task"]["title"] == "alpha work card"
    assert "task_id is required" in _show()["error"]
    assert _show(task_id="t_does_not_exist")["error"] == "task t_does_not_exist not found"


@PINNED
def test_refusal_writes_nothing(world, monkeypatch, tmp_path, pin_db):
    kb, ids = world
    _become_worker(monkeypatch, kb, ids["review"], pin_db=pin_db)
    root = tmp_path / ".hermes"
    before = _all_rows(root)
    assert any(t == "task_events" for _db, t in before) or any(t == "events" for _db, t in before)
    _assert_refused(_show(task_id=ids["plain"]))
    _assert_refused(_show(task_id=ids["other_board"], board=OTHER_BOARD))
    _assert_refused(_show(board=REVIEW_BOARD))
    _assert_refused(_show(task_id="t_does_not_exist"))
    assert _all_rows(root) == before
