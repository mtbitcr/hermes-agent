"""The RFC 9207 issuer (``iss``) must reach the MCP SDK from the dashboard callback."""

import asyncio

import pytest

ISSUER = "https://issuer.example/oauth/"
STATE = "state-iss"


def _waiting_flow(tmp_path, server_name="reports"):
    from tools.mcp_dashboard_oauth import DashboardOAuthFlow

    flow = DashboardOAuthFlow(
        flow_id=f"flow-iss-{server_name}",
        server_name=server_name,
        profile=None,
        hermes_home=str(tmp_path),
        redirect_uri=f"https://agent.example/api/mcp/oauth/callback/{server_name}",
    )
    asyncio.run(
        flow.publish_authorization_url(f"https://idp.example/authorize?state={STATE}")
    )
    return flow


def _sdk_result(flow):
    """What the dashboard waiter hands the SDK once the flow has its callback."""
    from tools.mcp_dashboard_oauth import dashboard_oauth_flow
    from tools.mcp_oauth import _make_callback_waiter

    # The waiter blocks for its whole timeout if no callback was ever delivered.
    assert flow.snapshot()["status"] == "exchanging"
    with dashboard_oauth_flow(flow):
        return asyncio.run(_make_callback_waiter(0)())


@pytest.fixture
def hosted_callback(monkeypatch):
    """GET the real callback route the way the browser redirect does."""
    from starlette.testclient import TestClient

    from hermes_cli import web_server

    monkeypatch.setattr(web_server.app.state, "auth_required", True, raising=False)
    client = TestClient(web_server.app)

    def deliver(flow, **query):
        monkeypatch.setitem(web_server._mcp_oauth_flows, flow.flow_id, flow)
        return client.get(
            f"/api/mcp/oauth/callback/{flow.server_name}", params=query
        )

    return deliver


def test_dashboard_flow_hands_the_sdk_the_callback_issuer(tmp_path):
    flow = _waiting_flow(tmp_path)

    flow.deliver_callback(code="code-1", state=STATE, error=None, iss=ISSUER)

    result = _sdk_result(flow)
    assert result.iss == ISSUER
    assert (result.code, result.state) == ("code-1", STATE)


def test_callback_route_passes_the_issuer_query_value_to_the_flow(
    tmp_path, hosted_callback
):
    flow = _waiting_flow(tmp_path)

    response = hosted_callback(flow, code="code-2", state=STATE, iss=ISSUER)

    assert response.status_code == 200
    assert _sdk_result(flow).iss == ISSUER


def test_callback_without_issuer_hands_the_sdk_none(tmp_path, hosted_callback):
    redirected = _waiting_flow(tmp_path, "redirected")
    assert hosted_callback(redirected, code="code-3", state=STATE).status_code == 200

    pasted = _waiting_flow(tmp_path, "pasted")
    pasted.deliver_callback(code="code-4", state=STATE, error=None)

    assert _sdk_result(redirected).iss is None
    assert _sdk_result(pasted).iss is None
