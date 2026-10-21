"""Card C5 (finding 144): the snapshot route negotiates ``delivery_state_v1`` the existing way.

Only a reader that names ``delivery_state_v1`` in ONE ``capabilities`` parameter makes the kernel call with
``delivery_state=True``; every other request makes exactly today's call.
"""

from __future__ import annotations

from unittest.mock import ANY, patch

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import PlatformConfig
from gateway.platforms.api_server import APIServerAdapter

_SNAPSHOT = "/v1/owner-workspace/projects/{project_slug}/snapshot"


def _app() -> web.Application:
    """The adapter's own snapshot route, as ``connect()`` adds it."""
    adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={}))
    app = web.Application()
    for method, path, handler in adapter._http_route_table():
        if method == "GET" and path == _SNAPSHOT:
            app.router.add_route(method, path, handler)
    return app


@pytest.mark.asyncio
@pytest.mark.parametrize("query, run_context, forwarded", [
    ("", False, {}),
    ("?capabilities=delivery_state_v1", False, {"delivery_state": True}),
    ("?capabilities=run_task_context,delivery_state_v1", True, {"delivery_state": True}),
    ("?capabilities=owner_waiting_v1,delivery_state_v1", False, {"owner_waiting": True, "delivery_state": True}),
    ("?capabilities=owner_waiting_v1", False, {"owner_waiting": True}),
    ("?capabilities=delivery_state_v2", False, {}),
    ("?capabilities=delivery_state", False, {}),
    ("?capabilities=delivery_state_v1&capabilities=delivery_state_v1", False, {}),
    ("?capabilities=run_task_context&capabilities=delivery_state_v1", False, {}),
])
async def test_the_snapshot_forwards_delivery_state_only_when_asked(query, run_context, forwarded):
    config = {"gateway": {"api_server": {"owner_workspace": {"enabled": True}}}}
    with (
        patch("gateway.run._load_gateway_config", return_value=config),
        patch("hermes_cli.owner_workspace.resolve_owner_context", return_value=object()),
        patch("hermes_cli.owner_workspace.read_project_snapshot", return_value={}) as read,
    ):
        async with TestClient(TestServer(_app())) as client:
            resp = await client.get(f"/v1/owner-workspace/projects/shoe-shop/snapshot{query}")

    assert resp.status == 200
    read.assert_called_once_with(ANY, "shoe-shop", run_context=run_context, planning_context=False, **forwarded)
