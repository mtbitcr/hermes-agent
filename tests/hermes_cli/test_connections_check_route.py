"""The owner app may check its own connections (B5a).

The owner app's connections credential reaches the existing connection check,
``POST /api/mcp/servers/{name}/test``, through the real token seam. The
credential is a recorded fake principal: no token value appears anywhere.
"""
import json
from types import SimpleNamespace

import pytest

from plugins.dashboard_auth.raphael_workspace import (
    CONNECTIONS_GRANT,
    CONNECTIONS_PRINCIPAL,
    CONNECTIONS_SCOPE,
    CONNECTIONS_TOKEN_PREFIX,
)

_ROLES = ("default", "raphael-designer", "raphael-verifier")


def _address(role, name):
    return f"https://{role}.mcp.example/{name}"


@pytest.fixture
def owner_check(tmp_path, monkeypatch, _isolate_hermes_home):
    """POST the check as the owner app's connections credential.

    Every home lists the same entries at addresses that name the home, so the
    recorded probes show which profile's entry was read.
    """
    try:
        from starlette.testclient import TestClient
    except ImportError:
        pytest.skip("fastapi/starlette not installed")

    from fastapi import FastAPI

    import hermes_cli.mcp_config as mcp_config
    import hermes_cli.web_server  # noqa: F401 — the router's late-bound helpers
    from hermes_cli import profiles
    from hermes_cli.dashboard_auth import audit, token_auth
    from hermes_cli.dashboard_auth.base import TokenPrincipal
    from hermes_cli.web_routers import mcp as mcp_routes
    from hermes_constants import get_hermes_home

    default_home = get_hermes_home()
    profiles_root = default_home / "profiles"
    for role in _ROLES:
        home = default_home if role == "default" else profiles_root / role
        home.mkdir(parents=True, exist_ok=True)
        (home / "config.yaml").write_text(
            "mcp_servers:\n"
            f"  work-mail:\n    url: {_address(role, 'work-mail')}\n"
            f"  work-calendar:\n    url: {_address(role, 'work-calendar')}\n"
            "    auth: oauth\n"
            f"  claude-design:\n    url: {_address(role, 'claude-design')}\n",
            encoding="utf-8",
        )
    monkeypatch.setattr(profiles, "_get_default_hermes_home", lambda: default_home)
    monkeypatch.setattr(profiles, "_get_profiles_root", lambda: profiles_root)

    state = SimpleNamespace(
        probes=[], fail=False, log_path=tmp_path / "logs" / "dashboard-auth.log"
    )

    def fake_probe(name, config, connect_timeout=30, details=None):
        state.probes.append((name, config["url"]))
        if state.fail:
            raise ConnectionError(f"cannot reach {config['url']}")
        if details is not None:
            details.update(schema_chars={"list_messages": 420}, prompts=2, resources=1)
        return [
            ("list_messages", f"Lists the mail at {config['url']}"),
            ("read_message", "Reads one message"),
        ]

    monkeypatch.setattr(mcp_config, "_probe_single_server", fake_probe)
    monkeypatch.setattr(mcp_config, "_oauth_tokens_present", lambda name: False)
    monkeypatch.setattr(audit, "_resolve_log_path", lambda: state.log_path)

    # Recorded fake of the credential: the seam's classification and provider
    # verification are replaced; route policy, scope, profile and strict-audit
    # handling stay real.
    principal = TokenPrincipal(
        principal=CONNECTIONS_PRINCIPAL,
        provider="raphael-connections-token",
        scopes=(CONNECTIONS_SCOPE,),
    )
    family = token_auth.MachineTokenFamily(CONNECTIONS_TOKEN_PREFIX, strict_audit=True)
    monkeypatch.setattr(token_auth, "get_machine_token_family", lambda _bearer: family)
    monkeypatch.setattr(
        token_auth, "authenticate_token", lambda _request: (principal, None)
    )
    mcp_routes._register_connections_machine_routes()

    app = FastAPI()
    app.include_router(mcp_routes.router)

    @app.middleware("http")
    async def machine_auth(request, call_next):
        return await token_auth.token_auth_middleware(request, call_next)

    state.client = TestClient(app)
    return state


def test_owner_credential_checks_root_entries_and_claude_design_only_as_designer(
    owner_check,
):
    """A root entry is checked in the root profile and Claude Design only in
    the Designer profile; every other pairing is refused before any probe."""
    client = owner_check.client

    root = client.post("/api/mcp/servers/work-mail/test")
    assert root.status_code == 200, root.text
    assert root.json()["ok"] is True
    design = client.post(
        "/api/mcp/servers/claude-design/test", params={"profile": "raphael-designer"}
    )
    assert design.status_code == 200, design.text
    assert design.json()["ok"] is True
    assert owner_check.probes == [
        ("work-mail", _address("default", "work-mail")),
        ("claude-design", _address("raphael-designer", "claude-design")),
    ]

    for name, profile in (
        ("claude-design", None),
        ("work-mail", "raphael-designer"),
        ("work-mail", "raphael-verifier"),
        ("claude-design", "raphael-verifier"),
    ):
        params = {"profile": profile} if profile else {}
        resp = client.post(f"/api/mcp/servers/{name}/test", params=params)
        assert resp.status_code == 403, (name, profile)
    assert len(owner_check.probes) == 2


def test_owner_credential_answer_holds_tool_names_only(owner_check):
    """No description, schema size or other detail; a failed check carries no
    error text, so no address can reach the owner app."""
    client = owner_check.client
    address = _address("default", "work-mail")

    working = client.post("/api/mcp/servers/work-mail/test")
    assert working.status_code == 200, working.text
    assert working.json() == {
        "ok": True,
        "tools": [{"name": "list_messages"}, {"name": "read_message"}],
    }
    assert address not in working.text

    owner_check.fail = True
    failed = client.post("/api/mcp/servers/work-mail/test")
    assert failed.status_code == 200, failed.text
    assert failed.json() == {"ok": False, "tools": []}
    assert address not in failed.text

    owner_check.fail = False
    unauthorized = client.post("/api/mcp/servers/work-calendar/test")
    assert unauthorized.status_code == 200, unauthorized.text
    assert unauthorized.json() == {"ok": False, "tools": []}


def test_owner_credential_check_is_audited(owner_check):
    """Each check writes one allow record naming the connection, a failed
    probe too: the strict seam refuses a 200 the route did not audit."""
    from hermes_cli.dashboard_auth.audit import AuditEvent

    client = owner_check.client
    assert client.post("/api/mcp/servers/work-mail/test").status_code == 200
    owner_check.fail = True
    assert client.post("/api/mcp/servers/work-mail/test").status_code == 200

    raw = owner_check.log_path.read_text(encoding="utf-8")
    records = [json.loads(line) for line in raw.splitlines()]
    allowed = [r for r in records if r["event"] == AuditEvent.TOKEN_AUTH_SUCCESS.value]
    assert len(allowed) == 2
    for record in allowed:
        assert record["principal"] == CONNECTIONS_PRINCIPAL
        assert record["grant"] == CONNECTIONS_GRANT
        assert record["method"] == "POST"
        assert record["route_template"] == "/api/mcp/servers/{name}/test"
        assert record["connection"] == "work-mail"
        assert record["decision"] == "allow"
        assert record["status"] == 200
    assert _address("default", "work-mail") not in raw
