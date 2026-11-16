"""The Decisions route lists one service access item, only for a reader that names it.

Card A of the service access expiry plan. A reader that names
``service_access_v1`` gets one standing item at the front of Decisions once a
key of Raphael's managed credentials ends within 14 days; every other reader
keeps today's answer, byte for byte. The item copies nothing of a key record
but its end date, and no state of the registry can break Decisions.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import sqlite3
import uuid
from pathlib import Path

import pytest
from aiohttp.test_utils import TestClient, TestServer

from hermes_cli import kanban_db
from plugins.dashboard_auth.raphael_workspace import token_store
from tests.gateway.test_api_server_owner_waiting import (  # noqa: F401
    _app,
    _project,
    _query,
    _stopped,
    owner,
)

DAY = 86400
SURFACES = ("workspace", "connections", "automations", "models")
PERMITTED_KEYS = [
    "decision_ref", "authority", "kind", "project_slug", "project_name", "title", "reason", "created_at",
]


def _issue(out_dir: Path, surface: str, *, days: int) -> tuple[token_store.TokenRecord, str]:
    """One key of ``surface`` as the operator issues it, and its value."""
    out_dir.mkdir(mode=0o700, exist_ok=True)
    out_path = out_dir / f"{uuid.uuid4().hex}.token"
    record = token_store.issue(out_path=out_path, ttl_seconds=days * DAY, surface=surface)
    return record, out_path.read_text(encoding="utf-8").strip()


def _owner_decision(owner) -> None:
    """One owner decision for the item to come before."""
    project = _project(owner, "Garden Plan")
    with contextlib.closing(kanban_db.connect(board=project["board"])) as conn:
        _stopped(
            conn, project, "Order the seeds",
            kind="needs_input", reason="Which seeds should we order?",
        )


async def _body(client: TestClient, *values: str) -> bytes:
    """One Decisions GET's raw body; ``values`` are the capabilities parameters."""
    path = "/v1/owner-workspace/decisions"
    resp = await client.get(f"{path}?{_query(*values)}" if values else path)
    assert resp.status == 200, await resp.text()
    return await resp.read()


def _service_items(body: bytes) -> list[dict]:
    return [item for item in json.loads(body)["data"] if item["kind"] == "service_access"]


@pytest.mark.asyncio
async def test_without_the_capability_the_answer_is_unchanged(owner, tmp_path):
    _owner_decision(owner)
    async with TestClient(TestServer(_app())) as client:
        today = await _body(client)
        waiting_today = await _body(client, "owner_waiting_v1")
        for surface in SURFACES:
            _issue(tmp_path / "issued", surface, days=10)

        # Every reader that does not name service_access_v1 gets today's answer.
        assert await _body(client) == today
        assert await _body(client, "service_access_v2") == today
        assert await _body(client, "owner_waiting_v1") == waiting_today
        named = json.loads(await _body(client, "service_access_v1"))
        both = json.loads(await _body(client, "owner_waiting_v1,service_access_v1"))

    # A reader that names it gets one item, at the front, and nothing else changes.
    item = named["data"][0]
    assert list(item) == PERMITTED_KEYS
    assert item["decision_ref"] == "decision_" + hashlib.sha256(
        f"{owner.profile}\x00service_access".encode("utf-8")
    ).hexdigest()[:32]
    assert (item["authority"], item["kind"], item["project_slug"], item["project_name"]) == (
        "service", "service_access", None, "Raphael",
    )
    assert item["title"].startswith("Raphael's access ends on ")
    today_answer = json.loads(today)
    assert today_answer["data"]
    assert named == {**today_answer, "data": [item, *today_answer["data"]]}
    waiting_answer = json.loads(waiting_today)
    assert both == {**waiting_answer, "data": [item, *waiting_answer["data"]]}


@pytest.mark.asyncio
async def test_service_access_item_carries_no_token_metadata(owner, tmp_path):
    issued = [_issue(tmp_path / "issued", surface, days=10) for surface in SURFACES]
    registry = token_store.store_path()
    async with TestClient(TestServer(_app())) as client:
        raw = await _body(client, "service_access_v1")

    body = raw.decode("utf-8")
    [item] = _service_items(raw)
    for record, value in issued:
        secret = value.split(".", 1)[1]
        for leak in (value, secret, record.token_id, record.digest, record.principal, record.grant):
            assert leak not in body
    # No file name or path: the registry, its folder, the issued files, the home.
    for leak in (registry.name, registry.parent.name, str(tmp_path)):
        assert leak not in body
    words = f"{item['title']} {item['reason']}".lower()
    assert "token" not in words and "key file" not in words


def _rows() -> dict:
    """The row count of every table of every board on disk."""
    counts = {}
    for board in kanban_db.list_boards():
        path = kanban_db.kanban_db_path(board["slug"])
        if not path.exists():
            continue
        with contextlib.closing(sqlite3.connect(path)) as db:
            tables = [name for (name,) in db.execute("SELECT name FROM sqlite_master WHERE type = 'table'")]
            for table in tables:
                counts[(board["slug"], table)] = db.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0]
    return counts


@pytest.mark.asyncio
async def test_service_access_writes_no_kanban_row(owner, tmp_path):
    _owner_decision(owner)
    for surface in SURFACES:
        _issue(tmp_path / "issued", surface, days=10)
    before = _rows()
    assert any(table == "tasks" and count for (_board, table), count in before.items())

    async with TestClient(TestServer(_app())) as client:
        for _ in range(2):
            assert len(_service_items(await _body(client, "service_access_v1"))) == 1

    assert _rows() == before


def _too_deep(path: Path, _monkeypatch) -> None:
    path.write_text("[" * 100_000 + "]" * 100_000, encoding="utf-8")


def _a_folder(path: Path, _monkeypatch) -> None:
    path.unlink()
    path.mkdir(mode=0o700)


def _cannot_enter(path: Path, monkeypatch) -> None:
    """The registry's folder belongs to another account. Root may enter any
    folder, so the refusal is raised where the store first looks inside."""
    real_lstat = Path.lstat

    def refused(self, *args, **kwargs):
        if self.parent == path.parent:
            raise PermissionError(13, "Permission denied", str(self))
        return real_lstat(self, *args, **kwargs)

    monkeypatch.setattr(Path, "lstat", refused)


BROKEN_REGISTRIES = {
    "missing": lambda path, _monkeypatch: path.unlink(),
    "not json": lambda path, _monkeypatch: path.write_text("{", encoding="utf-8"),
    "nested too deep": _too_deep,
    "a folder in its place": _a_folder,
    "a folder the gateway cannot enter": _cannot_enter,
}


@pytest.mark.asyncio
@pytest.mark.parametrize("breakage", list(BROKEN_REGISTRIES))
async def test_a_broken_missing_or_unreadable_registry_keeps_decisions(owner, tmp_path, monkeypatch, breakage):
    _owner_decision(owner)
    _issue(tmp_path / "issued", "workspace", days=10)
    async with TestClient(TestServer(_app())) as client:
        today = await _body(client)
        assert len(_service_items(await _body(client, "service_access_v1"))) == 1

        BROKEN_REGISTRIES[breakage](token_store.store_path(), monkeypatch)

        # The answer still returns: today's, without the item.
        assert await _body(client, "service_access_v1") == today
