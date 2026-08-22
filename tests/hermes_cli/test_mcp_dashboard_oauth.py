"""Dashboard HTTP contract for hosted MCP OAuth."""

from unittest.mock import patch

import pytest


def _client():
    from starlette.testclient import TestClient

    from hermes_cli.web_server import app, _SESSION_HEADER_NAME, _SESSION_TOKEN

    client = TestClient(app)
    client.headers[_SESSION_HEADER_NAME] = _SESSION_TOKEN
    return client


@pytest.fixture(autouse=True)
def _clear_flows():
    from hermes_cli import web_server

    web_server._mcp_oauth_flows.clear()
    web_server.app.state.auth_required = False
    yield
    web_server._mcp_oauth_flows.clear()
    web_server.app.state.auth_required = False


def test_hosted_auth_start_returns_public_authorization_url(monkeypatch):
    from hermes_cli import web_server

    client = _client()
    client.post(
        "/api/mcp/servers",
        json={"name": "reports", "url": "https://mcp.example/mcp", "auth": "oauth"},
    )

    def fake_worker(flow, cfg):
        import asyncio

        asyncio.run(flow.publish_authorization_url("https://idp.example/authorize?state=s1"))

    monkeypatch.setattr(web_server, "_run_dashboard_mcp_oauth", fake_worker)
    with patch(
        "hermes_cli.dashboard_auth.prefix.resolve_public_url",
        return_value="https://agent.example",
    ):
        response = client.post("/api/mcp/servers/reports/auth")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "authorization_required"
    assert body["authorization_url"] == "https://idp.example/authorize?state=s1"
    flow = web_server._mcp_oauth_flows[body["flow_id"]]
    assert flow.redirect_uri == "https://agent.example/api/mcp/oauth/callback/reports"


def test_hosted_auth_retry_reuses_the_waiting_flow(monkeypatch):
    import asyncio
    import threading

    from hermes_cli import web_server

    client = _client()
    client.post(
        "/api/mcp/servers",
        json={"name": "reports", "url": "https://mcp.example/mcp", "auth": "oauth"},
    )
    release = threading.Event()
    worker_calls = 0

    def fake_worker(flow, _cfg):
        nonlocal worker_calls
        worker_calls += 1
        asyncio.run(flow.publish_authorization_url("https://idp.example/authorize?state=s1"))
        release.wait(2)
        flow.mark_worker_done()

    monkeypatch.setattr(web_server, "_run_dashboard_mcp_oauth", fake_worker)
    first = client.post("/api/mcp/servers/reports/auth")
    second = client.post("/api/mcp/servers/reports/auth")
    release.set()

    assert first.status_code == 200
    assert second.status_code == 200
    assert second.json()["flow_id"] == first.json()["flow_id"]
    assert second.json()["authorization_url"] == first.json()["authorization_url"]
    assert worker_calls == 1
    assert len(web_server._mcp_oauth_flows) == 1


def test_claude_design_uses_manual_first_party_code_flow(monkeypatch):
    import asyncio

    from hermes_constants import get_hermes_home
    from hermes_cli import web_server
    from tools.claude_design_oauth import CLAUDE_DESIGN_MCP_URL

    profile_home = get_hermes_home() / "profiles" / "raphael-designer"
    profile_home.mkdir(parents=True)
    (profile_home / "config.yaml").write_text("{}\n", encoding="utf-8")
    response = _client().post(
        "/api/mcp/catalog/install",
        json={"name": "claude-design", "profile": "raphael-designer"},
    )
    assert response.status_code == 200, response.text

    def fake_worker(flow, cfg):
        assert cfg["url"] == CLAUDE_DESIGN_MCP_URL
        asyncio.run(
            flow.publish_authorization_url(
                "https://claude.com/cai/oauth/authorize?state=design-state"
            )
        )
        flow.mark_worker_done()

    monkeypatch.setattr(web_server, "_run_dashboard_mcp_oauth", fake_worker)
    response = _client().post(
        "/api/mcp/servers/claude-design/auth?profile=raphael-designer",
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["status"] == "authorization_required"
    assert body["flow"] == "browser_code"
    assert body["expires_in"] == 900
    parsed = __import__("urllib.parse", fromlist=["urlparse"]).urlparse(
        body["authorization_url"]
    )
    assert (parsed.scheme, parsed.netloc, parsed.path) == (
        "https",
        "claude.com",
        "/cai/oauth/authorize",
    )
    flow = web_server._mcp_oauth_flows[body["flow_id"]]
    assert flow.profile == "raphael-designer"
    assert flow.redirect_uri == "https://platform.claude.com/oauth/code/callback"


def test_claude_design_does_not_accept_the_generic_callback(monkeypatch):
    import asyncio

    from hermes_constants import get_hermes_home
    from hermes_cli import web_server

    profile_home = get_hermes_home() / "profiles" / "raphael-designer"
    profile_home.mkdir(parents=True)
    (profile_home / "config.yaml").write_text("{}\n", encoding="utf-8")
    client = _client()
    assert client.post(
        "/api/mcp/catalog/install",
        json={"name": "claude-design", "profile": "raphael-designer"},
    ).status_code == 200

    def fake_worker(flow, _cfg):
        asyncio.run(
            flow.publish_authorization_url(
                "https://claude.com/cai/oauth/authorize?state=design-state"
            )
        )
        flow._callback_ready.wait(2)
        flow.mark_worker_done()

    monkeypatch.setattr(web_server, "_run_dashboard_mcp_oauth", fake_worker)
    start = client.post(
        "/api/mcp/servers/claude-design/auth?profile=raphael-designer",
    ).json()
    flow = web_server._mcp_oauth_flows[start["flow_id"]]

    response = client.get(
        "/api/mcp/oauth/callback/claude-design",
        params={"code": "must-not-be-consumed", "state": flow.expected_state},
    )

    assert response.status_code == 404
    assert flow.status == "authorization_required"
    flow.mark_error("test cleanup")


def test_claude_design_replaces_only_a_waiting_flow(monkeypatch):
    import asyncio
    import threading

    from hermes_constants import get_hermes_home
    from hermes_cli import web_server

    profile_home = get_hermes_home() / "profiles" / "raphael-designer"
    profile_home.mkdir(parents=True)
    (profile_home / "config.yaml").write_text("{}\n", encoding="utf-8")
    client = _client()
    assert client.post(
        "/api/mcp/catalog/install",
        json={"name": "claude-design", "profile": "raphael-designer"},
    ).status_code == 200

    exchange_started = threading.Event()
    release_exchange = threading.Event()

    def fake_worker(flow, _cfg):
        asyncio.run(
            flow.publish_authorization_url(
                f"https://claude.com/cai/oauth/authorize?state=state-{flow.flow_id}"
            )
        )
        try:
            asyncio.run(flow.wait_for_callback(timeout=2))
            exchange_started.set()
            release_exchange.wait(2)
            flow.mark_approved()
        except Exception as exc:
            flow.mark_error(str(exc))
        finally:
            flow.mark_worker_done()

    monkeypatch.setattr(web_server, "_run_dashboard_mcp_oauth", fake_worker)

    first = client.post(
        "/api/mcp/servers/claude-design/auth?profile=raphael-designer",
    ).json()
    first_flow = web_server._mcp_oauth_flows[first["flow_id"]]
    second_response = client.post(
        "/api/mcp/servers/claude-design/auth?profile=raphael-designer",
    )
    assert second_response.status_code == 200
    assert first_flow.status == "error"

    second_flow = web_server._mcp_oauth_flows[second_response.json()["flow_id"]]
    submit = client.post(
        f"/api/mcp/oauth/flows/{second_flow.flow_id}/submit",
        json={"code": f"approved-code#{second_flow.expected_state}"},
    )
    assert submit.status_code == 200
    assert submit.json() == {"ok": True, "status": "verifying"}
    assert exchange_started.wait(1)
    third_response = client.post(
        "/api/mcp/servers/claude-design/auth?profile=raphael-designer",
    )
    assert third_response.status_code == 409
    assert second_flow.status == "exchanging"
    release_exchange.set()


def test_claude_design_manual_code_completes_and_probes_exact_profile(monkeypatch):
    import asyncio
    import time

    from hermes_constants import get_hermes_home
    from hermes_cli import web_server

    profile_home = get_hermes_home() / "profiles" / "raphael-designer"
    profile_home.mkdir(parents=True)
    (profile_home / "config.yaml").write_text("{}\n", encoding="utf-8")
    client = _client()
    assert client.post(
        "/api/mcp/catalog/install",
        json={"name": "claude-design", "profile": "raphael-designer"},
    ).status_code == 200

    def fake_worker(flow, config):
        assert config["url"] == "https://api.anthropic.com/v1/design/mcp"
        assert flow.hermes_home == str(profile_home)
        asyncio.run(
            flow.publish_authorization_url(
                "https://claude.com/cai/oauth/authorize?state=expected-state"
            )
        )
        try:
            code, state = asyncio.run(flow.wait_for_callback(timeout=2))
            assert (code, state) == ("approved-code", "expected-state")
            flow.tools = [
                {"name": "design_review", "description": "Review a design"}
            ]
            flow.mark_approved()
        except Exception as exc:
            flow.mark_error(str(exc))
        finally:
            flow.mark_worker_done()

    monkeypatch.setattr(web_server, "_run_dashboard_mcp_oauth", fake_worker)
    start = client.post(
        "/api/mcp/servers/claude-design/auth?profile=raphael-designer",
    ).json()
    flow = web_server._mcp_oauth_flows[start["flow_id"]]
    response = client.post(
        f"/api/mcp/oauth/flows/{start['flow_id']}/submit",
        json={"code": f"approved-code#{flow.expected_state}"},
    )

    assert response.status_code == 200, response.text
    assert response.json() == {"ok": True, "status": "verifying"}
    deadline = time.monotonic() + 1
    status = {}
    while time.monotonic() < deadline:
        status_response = client.get(f"/api/mcp/oauth/flows/{flow.flow_id}")
        assert status_response.status_code == 200
        status = status_response.json()
        if status["status"] == "approved":
            break
        time.sleep(0.01)
    assert status["status"] == "approved"
    assert status["tools"] == [
        {"name": "design_review", "description": "Review a design"}
    ]


def test_claude_design_rejects_mismatched_or_reused_manual_code(monkeypatch):
    import asyncio

    from hermes_constants import get_hermes_home
    from hermes_cli import web_server

    profile_home = get_hermes_home() / "profiles" / "raphael-designer"
    profile_home.mkdir(parents=True)
    (profile_home / "config.yaml").write_text("{}\n", encoding="utf-8")
    client = _client()
    assert client.post(
        "/api/mcp/catalog/install",
        json={"name": "claude-design", "profile": "raphael-designer"},
    ).status_code == 200

    def fake_worker(flow, _cfg):
        asyncio.run(
            flow.publish_authorization_url(
                "https://claude.com/cai/oauth/authorize?state=expected-state"
            )
        )
        flow._callback_ready.wait(2)
        flow.mark_worker_done()

    monkeypatch.setattr(web_server, "_run_dashboard_mcp_oauth", fake_worker)
    start = client.post(
        "/api/mcp/servers/claude-design/auth?profile=raphael-designer",
    ).json()
    flow = web_server._mcp_oauth_flows[start["flow_id"]]

    mismatch = client.post(
        f"/api/mcp/oauth/flows/{flow.flow_id}/submit",
        json={"code": "approved-code#wrong-state"},
    )
    accepted = client.post(
        f"/api/mcp/oauth/flows/{flow.flow_id}/submit",
        json={"code": f"approved-code#{flow.expected_state}"},
    )
    reused = client.post(
        f"/api/mcp/oauth/flows/{flow.flow_id}/submit",
        json={"code": f"second-code#{flow.expected_state}"},
    )

    assert mismatch.status_code == 400
    assert accepted.status_code == 200
    assert reused.status_code == 409


def test_hosted_auth_accepts_exact_https_service_callback(monkeypatch):
    from hermes_cli import web_server

    client = _client()
    client.post(
        "/api/mcp/servers",
        json={"name": "reports", "url": "https://mcp.example/mcp", "auth": "oauth"},
    )

    def fake_worker(flow, cfg):
        import asyncio

        asyncio.run(flow.publish_authorization_url("https://idp.example/authorize?state=s1"))

    monkeypatch.setattr(web_server, "_run_dashboard_mcp_oauth", fake_worker)
    response = client.post(
        "/api/mcp/servers/reports/auth",
        json={
            "redirect_uri": "https://workspace.example/api/mcp/oauth/callback/reports"
        },
    )

    assert response.status_code == 200
    flow = web_server._mcp_oauth_flows[response.json()["flow_id"]]
    assert flow.redirect_uri == (
        "https://workspace.example/api/mcp/oauth/callback/reports"
    )


@pytest.mark.parametrize(
    "redirect_uri",
    [
        "http://workspace.example/api/mcp/oauth/callback/reports",
        "https://workspace.example/other/callback",
        "https://user:pass@workspace.example/api/mcp/oauth/callback/reports",
        "https://workspace.example/api/mcp/oauth/callback/reports?next=evil",
    ],
)
def test_hosted_auth_rejects_untrusted_service_callback(redirect_uri):
    client = _client()
    client.post(
        "/api/mcp/servers",
        json={"name": "reports", "url": "https://mcp.example/mcp", "auth": "oauth"},
    )

    response = client.post(
        "/api/mcp/servers/reports/auth",
        json={"redirect_uri": redirect_uri},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Invalid OAuth redirect URI"


def test_hosted_callback_bypasses_gated_cookie_auth(monkeypatch):
    import asyncio

    from starlette.testclient import TestClient

    from hermes_cli import web_server
    from tools.mcp_dashboard_oauth import DashboardOAuthFlow

    flow = DashboardOAuthFlow(
        flow_id="flow-gated",
        server_name="reports",
        profile=None,
        hermes_home="/tmp/hermes-test",
        redirect_uri="https://agent.example/api/mcp/oauth/callback/reports",
    )
    asyncio.run(
        flow.publish_authorization_url(
            "https://idp.example/authorize?state=expected"
        )
    )
    web_server._mcp_oauth_flows[flow.flow_id] = flow
    monkeypatch.setattr(web_server.app.state, "auth_required", True, raising=False)

    response = TestClient(web_server.app).get(
        "/api/mcp/oauth/callback/reports?code=abc&state=expected"
    )

    assert response.status_code == 200
    assert flow._callback == ("abc", "expected")


def test_hosted_auth_allows_same_server_name_in_different_profiles(tmp_path, monkeypatch):
    from hermes_cli import web_server
    from tools.mcp_dashboard_oauth import DashboardOAuthFlow

    profile_home = tmp_path / "profiles" / "work"
    profile_home.mkdir(parents=True)
    monkeypatch.setattr(web_server, "_resolve_profile_dir", lambda _name: profile_home)

    existing = DashboardOAuthFlow(
        flow_id="existing-default",
        server_name="reports",
        profile=None,
        hermes_home=str(tmp_path / "default"),
        redirect_uri="https://agent.example/callback/existing",
    )
    web_server._mcp_oauth_flows[existing.flow_id] = existing

    def fake_worker(flow, cfg):
        import asyncio

        asyncio.run(flow.publish_authorization_url("https://idp.example/authorize?state=work"))

    with patch("hermes_cli.mcp_config._get_mcp_servers", return_value={"reports": {"url": "https://mcp.example"}}), \
         patch.object(web_server, "_run_dashboard_mcp_oauth", fake_worker):
        response = _client().post("/api/mcp/servers/reports/auth?profile=work")

    assert response.status_code != 409


def test_delete_revokes_inflight_oauth_without_late_state_resurrection(monkeypatch):
    import asyncio
    import threading

    from hermes_cli import web_server
    from hermes_cli.mcp_config import _get_mcp_servers
    from tools.mcp_dashboard_oauth import dashboard_oauth_flow
    from tools.mcp_oauth import HermesTokenStorage

    client = _client()
    assert client.post(
        "/api/mcp/servers",
        json={"name": "reports", "url": "https://mcp.example/mcp", "auth": "oauth"},
    ).status_code == 200

    release_late_write = threading.Event()

    class LateClientInfo:
        def model_dump(self, **_kwargs):
            return {"client_id": "late-client"}

    def fake_worker(flow, _cfg):
        storage = HermesTokenStorage("reports", hermes_home=flow.hermes_home)
        with dashboard_oauth_flow(flow):
            asyncio.run(
                flow.publish_authorization_url(
                    "https://idp.example/authorize?state=late-write"
                )
            )
            release_late_write.wait(2)
            try:
                asyncio.run(storage.set_client_info(LateClientInfo()))
            except RuntimeError:
                pass
            finally:
                flow.mark_worker_done()

    monkeypatch.setattr(web_server, "_run_dashboard_mcp_oauth", fake_worker)
    started = client.post("/api/mcp/servers/reports/auth")
    assert started.status_code == 200
    flow = web_server._mcp_oauth_flows[started.json()["flow_id"]]
    storage = HermesTokenStorage("reports", hermes_home=flow.hermes_home)

    removed = client.delete("/api/mcp/servers/reports")
    release_late_write.set()
    assert flow._worker_done.wait(2)

    assert removed.status_code == 200
    assert flow.snapshot()["status"] == "error"
    assert "reports" not in _get_mcp_servers()
    assert not storage._client_info_path().exists()


def test_oauth_start_cannot_resurrect_connection_removed_before_registration(
    monkeypatch,
):
    import threading

    from hermes_cli.mcp_config import _get_mcp_servers
    from tools import mcp_dashboard_oauth
    from tools.mcp_oauth import HermesTokenStorage

    client = _client()
    assert client.post(
        "/api/mcp/servers",
        json={"name": "reports", "url": "https://mcp.example/mcp", "auth": "oauth"},
    ).status_code == 200

    flow_constructed = threading.Event()
    release_registration = threading.Event()
    auth_result = {}
    flow_class = mcp_dashboard_oauth.DashboardOAuthFlow

    def blocking_flow(*args, **kwargs):
        flow = flow_class(*args, **kwargs)
        flow_constructed.set()
        assert release_registration.wait(2)
        return flow

    monkeypatch.setattr(mcp_dashboard_oauth, "DashboardOAuthFlow", blocking_flow)

    def start_auth():
        auth_result["response"] = _client().post("/api/mcp/servers/reports/auth")

    auth_thread = threading.Thread(target=start_auth)
    auth_thread.start()
    assert flow_constructed.wait(2)

    removed = client.delete("/api/mcp/servers/reports")
    release_registration.set()
    auth_thread.join(2)

    assert not auth_thread.is_alive()
    assert removed.status_code == 200
    assert auth_result["response"].status_code == 200
    assert auth_result["response"].json()["status"] == "error"
    assert "reports" not in _get_mcp_servers()
    assert not HermesTokenStorage("reports")._client_info_path().exists()


def test_flow_status_does_not_expose_authorization_code():
    from hermes_cli import web_server
    from tools.mcp_dashboard_oauth import DashboardOAuthFlow

    flow = DashboardOAuthFlow(
        flow_id="flow-status",
        server_name="reports",
        profile=None,
        hermes_home="/tmp/hermes-test",
        redirect_uri="https://agent.example/api/mcp/oauth/callback/flow-status",
    )
    flow.authorization_url = "https://idp.example/authorize"
    flow.status = "approved"
    flow._callback = ("secret-code", "secret-state")
    web_server._mcp_oauth_flows[flow.flow_id] = flow

    response = _client().get("/api/mcp/oauth/flows/flow-status")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "approved"
    assert "secret-code" not in response.text
    assert "secret-state" not in response.text


def test_dashboard_reauth_authorizes_before_anonymous_discovery(tmp_path, monkeypatch):
    """Hosted re-auth must explicitly authorize before probing Drive-like MCPs."""
    from hermes_cli import web_server
    from tools.mcp_dashboard_oauth import DashboardOAuthFlow

    flow = DashboardOAuthFlow(
        flow_id="flow-anonymous-discovery",
        server_name="drive",
        profile=None,
        hermes_home=str(tmp_path),
        redirect_uri="https://agent.example/api/mcp/oauth/callback/drive",
    )
    cfg = {
        "url": "https://drivemcp.googleapis.com/mcp/v1",
        "auth": "oauth",
    }
    calls = []
    authorized = {"value": False}

    class FakeManager:
        def remove(self, *_args, **_kwargs):
            calls.append("remove")
            return None

        def restore_entry(self, *_args, **_kwargs):
            raise AssertionError("successful authorization must not roll back")

    def fake_authorize(name, config, *, connect_timeout):
        assert name == "drive"
        assert config is cfg
        assert connect_timeout >= 315
        calls.append("authorize")
        authorized["value"] = True

    def fake_probe(name, config, connect_timeout):
        assert authorized["value"] is True
        calls.append("probe")
        return [("search_files", "Search Drive")]

    monkeypatch.setattr(
        "hermes_cli.mcp_config._get_mcp_servers",
        lambda: {"drive": cfg},
    )
    monkeypatch.setattr(
        "hermes_cli.mcp_config._authorize_oauth_server",
        fake_authorize,
    )
    monkeypatch.setattr(
        "hermes_cli.mcp_config._probe_single_server",
        fake_probe,
    )
    monkeypatch.setattr(
        "hermes_cli.mcp_config._oauth_tokens_present",
        lambda _name: authorized["value"],
    )
    monkeypatch.setattr(
        "hermes_cli.mcp_config._save_mcp_server",
        lambda *_args: calls.append("save") or True,
    )
    monkeypatch.setattr(
        "tools.mcp_oauth_manager.get_manager",
        lambda: FakeManager(),
    )

    web_server._run_dashboard_mcp_oauth(flow, cfg)

    assert flow.status == "approved"
    assert flow.tools == [{"name": "search_files", "description": "Search Drive"}]
    assert calls == ["remove", "authorize", "probe", "save"]
