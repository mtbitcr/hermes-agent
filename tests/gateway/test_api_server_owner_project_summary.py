"""The owner snapshot route passes ``project_summary`` to the kernel so the Overview shows Raphael's summary.

Only a reader that names ``project_summary`` gets ``raphael_summary``; the kernel stand-in below follows the same
contract as ``read_project_snapshot``: the key is present (the stored text, or None) only when asked for.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import PlatformConfig
from gateway.platforms.api_server import APIServerAdapter

_SNAPSHOT = "/v1/owner-workspace/projects/{project_slug}/snapshot"
_STORED = {"outcome": "The room is booked for Tuesday at noon.", "next_action": "Send the agenda on Monday."}


def _app() -> web.Application:
    adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={}))
    app = web.Application()
    for method, path, handler in adapter._http_route_table():
        if method == "GET" and path == _SNAPSHOT:
            app.router.add_route(method, path, handler)
    return app


async def _get_snapshot(query: str, stored: dict | None) -> dict:
    def fake_read(_ctx, _slug, *, project_summary=False, **_kwargs):
        snapshot = {"project": {"slug": "shoe-shop"}}
        if project_summary:
            snapshot["raphael_summary"] = stored
        return snapshot

    config = {"gateway": {"api_server": {"owner_workspace": {"enabled": True}}}}
    with (
        patch("gateway.run._load_gateway_config", return_value=config),
        patch("hermes_cli.owner_workspace.resolve_owner_context", return_value=object()),
        patch("hermes_cli.owner_workspace.read_project_snapshot", side_effect=fake_read),
    ):
        async with TestClient(TestServer(_app())) as client:
            resp = await client.get(f"/v1/owner-workspace/projects/shoe-shop/snapshot{query}")
            assert resp.status == 200
            return (await resp.json())["data"]


@pytest.mark.asyncio
async def test_a_stored_summary_is_returned_when_project_summary_is_named():
    body = await _get_snapshot("?capabilities=project_summary", _STORED)

    assert body["raphael_summary"] == _STORED


@pytest.mark.asyncio
async def test_the_summary_is_null_when_none_is_stored():
    body = await _get_snapshot("?capabilities=project_summary", None)

    assert "raphael_summary" in body
    assert body["raphael_summary"] is None


@pytest.mark.asyncio
async def test_no_summary_key_without_the_capability():
    body = await _get_snapshot("", _STORED)

    assert "raphael_summary" not in body
