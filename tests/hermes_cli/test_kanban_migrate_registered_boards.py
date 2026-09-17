"""Unit tests for kanban_db.migrate_registered_boards()."""
import sqlite3
import pytest
from hermes_cli import kanban_db as kb


@pytest.fixture
def fresh_home(tmp_path, monkeypatch):
    home = tmp_path / "hermes_home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    for v in "HERMES_KANBAN_DB HERMES_KANBAN_WORKSPACES_ROOT HERMES_KANBAN_HOME HERMES_KANBAN_BOARD".split():
        monkeypatch.delenv(v, raising=False)
    try: import hermes_constants; hermes_constants._cached_default_hermes_root = None
    except Exception: pass
    kb._INITIALIZED_PATHS.clear()
    yield home
    kb._INITIALIZED_PATHS.clear()


def _drop_wst(db_path):
    c = sqlite3.connect(str(db_path))
    for t in ("tasks", "task_runs"):
        if "worker_start_time" in {r[1] for r in c.execute(f"PRAGMA table_info({t})")}:
            c.execute(f"ALTER TABLE {t} DROP COLUMN worker_start_time")
    c.commit(); c.close()


def _has_wst(db_path):
    c = sqlite3.connect(str(db_path))
    v = "worker_start_time" in {r[1] for r in c.execute("PRAGMA table_info(tasks)")}
    c.close(); return v


def test_migrates_all(fresh_home):
    for slug in ("board-a", "board-b"):
        kb.create_board(slug)
        p = kb.kanban_db_path(board=slug)
        _drop_wst(p); kb._INITIALIZED_PATHS.discard(str(p.resolve()))
    assert not _has_wst(kb.kanban_db_path(board="board-a"))
    migrated = kb.migrate_registered_boards()
    assert "board-a" in migrated and "board-b" in migrated
    for slug in ("board-a", "board-b"):
        assert _has_wst(kb.kanban_db_path(board=slug))


def test_skips_and_never_raises(fresh_home):
    kb.create_board("board-good"); kb.create_board("board-bad")
    db_good = kb.kanban_db_path(board="board-good")
    db_bad = kb.kanban_db_path(board="board-bad")
    _drop_wst(db_good); kb._INITIALIZED_PATHS.discard(str(db_good.resolve()))
    db_bad.unlink(); kb._INITIALIZED_PATHS.discard(str(db_bad.resolve()))
    # e: skips unopenable, still migrates remaining
    migrated = kb.migrate_registered_boards()
    assert "board-good" in migrated and "board-bad" not in migrated
    assert _has_wst(db_good)
    # f: never raises when all boards fail
    db_good.unlink(); kb._INITIALIZED_PATHS.discard(str(db_good.resolve()))
    assert "board-good" not in kb.migrate_registered_boards()


def test_raises_on_missing_column(fresh_home):
    kb.create_board("board-contract")
    db_path = kb.kanban_db_path(board="board-contract")
    _drop_wst(db_path)
    conn = sqlite3.connect(db_path.resolve().as_uri() + "?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    with pytest.raises(sqlite3.OperationalError, match="no such column"):
        kb.verified_active_worker_rows(conn)
    conn.close()
