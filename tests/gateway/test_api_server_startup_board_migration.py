"""Regression: connect() migrates boards lacking worker_start_time before serving."""
import json, pytest, sqlite3
pytest.importorskip("aiohttp")
from hermes_cli import kanban_db, owner_workspace as ow, projects_db  # noqa: E402
from gateway.platforms.api_server import APIServerAdapter  # noqa: E402
from gateway.config import PlatformConfig  # noqa: E402

_KEY = "migration-test-key-abcdef1234567890"
_n = [0]


def _setup(monkeypatch):
    _n[0] += 1; s = f"migrate_test_{_n[0]}"
    kanban_db.create_board(s)
    p = kanban_db.kanban_db_path(board=s)
    with projects_db.connect_closing() as pc:
        pid = projects_db.create_project(pc, name=f"MT{_n[0]}", slug=s, primary_path=f"/tmp/mt{_n[0]}")
        projects_db.update_project(pc, pid, board_slug=s)
        ow._ensure_schema(pc)
        pc.execute("INSERT INTO owner_workspace_receipts VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",("default","default",f"boot_{_n[0]}","owner_workspace_bootstrap","d","committed",None,None,None,None,None,json.dumps({"ok":True,"project_id":str(pid)}),None,0,0,0))
        pc.commit()
    kanban_db.write_board_metadata(s, project_id=pid)
    raw = sqlite3.connect(str(p))
    for t in ("tasks", "task_runs"):
        if "worker_start_time" in {r[1] for r in raw.execute(f"PRAGMA table_info({t})")}:
            raw.execute(f"ALTER TABLE {t} DROP COLUMN worker_start_time")
    raw.commit(); raw.close()
    kanban_db._INITIALIZED_PATHS.discard(str(p.resolve()))
    return ow.OwnerContext(actor="default", profile="default", session="migration_test"), s


@pytest.mark.asyncio
async def test_startup_migrates_board(monkeypatch):
    ctx, slug = _setup(monkeypatch)
    p = kanban_db.kanban_db_path(board=slug)
    uri = p.resolve().as_uri() + "?mode=ro"
    # c: pre-state genuinely broken before connect() runs
    ro = sqlite3.connect(uri, uri=True); ro.row_factory = sqlite3.Row
    with pytest.raises(sqlite3.OperationalError, match="no such column"):
        kanban_db.verified_active_worker_rows(ro)
    ro.close()
    monkeypatch.setattr(
        "gateway.platforms.api_server.APIServerAdapter._recover_orphaned_owner_jobs",
        lambda self: None,
    )
    monkeypatch.setattr(ow, "_assert_board_ownership", lambda *a, **kw: None)
    monkeypatch.setenv("API_SERVER_KEY", _KEY)
    adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={"host": "127.0.0.1", "port": 0, "key": _KEY}))
    try:  # g: finally tears down socket
        assert await adapter.connect()
        # b: both read paths succeed after migration
        ro2 = sqlite3.connect(uri, uri=True); ro2.row_factory = sqlite3.Row
        assert isinstance(kanban_db.verified_active_worker_rows(ro2), list)
        ro2.close()
        assert "project" in ow.read_project_snapshot(ctx, slug)
    finally:
        await adapter.disconnect()
