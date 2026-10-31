"""Connection rows: GET /api/mcp/connection-rows, one read-only row per connection.

The owner app's connections credential reads one row per saved connection of a
fixed list of profiles: default (owner), raphael-designer (designer) and
raphael-builder, raphael-planner, raphael-verifier and raphael-claude-worker
(agent). A row holds names and states only, never an address, command,
arguments, environment values or headers, and the read changes nothing. Every
other caller is refused. Every token is minted here under tmp_path, every
saved sign-in is an empty recorded fake and every address, command and value
is synthetic.
"""

import hashlib
import importlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

ROUTER = "hermes_cli.web_routers.connection_rows"
PATH = "/api/mcp/connection-rows"
ROW_FIELDS = [
    "name",
    "profile",
    "profile_role",
    "transport",
    "sign_in_kind",
    "status",
    "enabled",
    "read_only",
    "catalogue",
]
ROLES = {
    "default": "owner",
    "raphael-designer": "designer",
    "raphael-builder": "agent",
    "raphael-planner": "agent",
    "raphael-verifier": "agent",
    "raphael-claude-worker": "agent",
}
STATUSES = {"connected", "paused", "needs_authorization", "configured", "unknown"}
#: A real profile that is not on the list.
OUTSIDER = "raphael-business"
#: Owner decision of 2026-10-31: removed from the root profile, never a root row.
REMOVED_FROM_ROOT = ("microsoft-personal", "github-bitcredit", "google-calendar-personal")


def _fake(label):
    """A synthetic value no 8-character run of which can occur in a row."""
    return "SYNTHETIC" + hashlib.sha256(label.encode("utf-8")).hexdigest()[:24]


def _http(label, *, auth="oauth", enabled=True, headers=None):
    cfg = {"url": f"https://{_fake(label + '-host')}.invalid/mcp/{_fake(label + '-path')}"}
    if auth:
        cfg["auth"] = auth
    if headers:
        cfg["headers"] = headers
    if not enabled:
        cfg["enabled"] = False
    return cfg


def _stdio(label):
    return {
        "command": f"/opt/{_fake(label + '-dir')}/run",
        "args": ["--" + _fake(label + "-flag"), _fake(label + "-arg")],
        "env": {_fake(label + "-env-key"): _fake(label + "-env-value")},
    }


def _servers():
    """What each profile saves, the outsider included."""
    agent_github = {
        "github-agent-factory": _stdio("factory"),
        "github-agent-factory-org": _stdio("factory-org"),
    }
    return {
        "default": {
            "google-gmail-work": _http("gmail"),
            "google-calendar-work": _http("calendar", enabled=False),
            "google-drive-work": _http("drive", headers={"X-Trace-" + _fake("drive-hname"): _fake("drive-hvalue")}),
            "microsoft-personal": _stdio("microsoft"),
            "github-bitcredit": _http("bitcredit", auth=None, headers={"Authorization": "Bearer " + _fake("bitcredit")}),
            "google-calendar-personal": _http("calendar-personal", enabled=False),
        },
        "raphael-designer": {
            "claude-design": _http("claude-design", enabled=False),
            "github-agent-factory": _stdio("designer-factory"),
        },
        "raphael-builder": {
            **agent_github,
            "github-agent-factory-provisioner": _stdio("provisioner"),
        },
        "raphael-planner": dict(agent_github),
        "raphael-verifier": {
            **agent_github,
            "github-independent-reviewer": _stdio("reviewer"),
        },
        "raphael-claude-worker": {
            "context7": _http("context7", auth=None, headers={"Authorization": "Bearer " + _fake("context7")}),
        },
        OUTSIDER: {"business-crm": _http("business-crm")},
    }


#: Saved sign-ins (recorded fakes: the file exists, its body holds nothing).
SIGNED_IN = {
    "default": ("google-drive-work", "microsoft-personal"),
    "raphael-designer": ("claude-design",),
    OUTSIDER: ("business-crm",),
}


def _secrets(servers):
    """Every saved address, command, argument, environment key and value and header value."""
    found = []
    for by_name in servers.values():
        for cfg in by_name.values():
            found += [cfg[key] for key in ("url", "command") if key in cfg]
            found += cfg.get("args", [])
            for key, value in cfg.get("env", {}).items():
                found += [key, value]
            for key, value in cfg.get("headers", {}).items():
                found += [value] + ([key] if key != "Authorization" else [])
    return found


class _Quiet(SimpleNamespace):
    """A namespace whose repr names no value, so failure output never prints a token or fake."""

    def __repr__(self):
        return "<quiet>"


def _holds_part_of(text, value, size=8):
    """True when ``text`` holds ``value`` or any ``size``-character run of it."""
    if not value:
        return False
    if len(value) <= size:
        return value in text
    return any(value[start : start + size] in text for start in range(len(value) - size + 1))


def _router():
    """The route's module, asserted inside a test so its absence fails that test."""
    assert importlib.util.find_spec(ROUTER) is not None, f"{ROUTER} does not exist"
    module = importlib.import_module(ROUTER)
    module._register_connection_rows_route()
    return module


def _write_home(home, servers):
    home.mkdir(parents=True, exist_ok=True)
    (home / "config.yaml").write_text(yaml.safe_dump({"mcp_servers": servers}), encoding="utf-8")


def _save_sign_in(home, name):
    tokens = home / "mcp-tokens"
    tokens.mkdir(parents=True, exist_ok=True)
    (tokens / f"{name}.json").write_text("{}", encoding="utf-8")


def _saved_state(root):
    """Every saved connection setting and sign-in under the root home: path -> bytes."""
    state = {}
    for path in sorted(root.rglob("*")):
        rel = path.relative_to(root)
        if path.is_dir() and path.name == "mcp-tokens":
            state[f"{rel}/"] = b""
        elif path.is_file() and (path.name in {"config.yaml", ".env"} or "mcp-tokens" in rel.parts):
            state[str(rel)] = path.read_bytes()
    return state


@pytest.fixture()
def home(tmp_path, monkeypatch):
    """The root Hermes home the dashboard runs with."""
    root = tmp_path / "hermes"
    root.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(root))
    return root


@pytest.fixture()
def saved(home):
    """The saved connections and sign-ins of every profile, the outsider's included."""
    servers = _servers()
    homes = {name: home if name == "default" else home / "profiles" / name for name in servers}
    for name, by_name in servers.items():
        _write_home(homes[name], by_name)
        for connection in SIGNED_IN.get(name, ()):
            _save_sign_in(homes[name], connection)
    return _Quiet(root=home, homes=homes, servers=servers, secrets=_secrets(servers))


def _restore_state(state, saved_attrs):
    for attr, previous in saved_attrs.items():
        if previous is None:
            if hasattr(state, attr):
                delattr(state, attr)
        else:
            setattr(state, attr, previous)


@pytest.fixture()
def dash(home, saved, tmp_path):
    """The real dashboard app, an owner session and a token for every machine surface."""
    from fastapi.testclient import TestClient

    from hermes_cli import web_server
    from hermes_cli.dashboard_auth import register_provider
    from hermes_cli.dashboard_auth.registry import restore_registration, snapshot_registration
    from hermes_cli.web_routers.mcp import _register_connections_machine_routes
    from plugins.dashboard_auth.raphael_workspace import (
        AutomationsManageTokenProvider,
        ConnectionsManageTokenProvider,
        ModelsManageTokenProvider,
        RecommendationsReadTokenProvider,
        WorkspaceReadTokenProvider,
        token_store,
    )
    from plugins.dashboard_auth.raphael_workspace.model_policy import register_models_machine_routes

    register_models_machine_routes()
    _register_connections_machine_routes()
    token_dir = tmp_path / "machine-tokens"
    token_dir.mkdir(mode=0o700)

    def issue(name, surface):
        path = token_dir / f"{name}.token"
        record = token_store.issue(out_path=path, surface=surface)
        return record, path.read_text(encoding="utf-8").strip()

    connections_record, connections = issue("connections", token_store.CONNECTIONS_SURFACE)
    revoked_record, revoked = issue("connections-revoked", token_store.CONNECTIONS_SURFACE)
    token_store.revoke(revoked_record.token_id)
    bearers = {
        "connections": connections,
        "connections_revoked": revoked,
        "connections_unknown": token_store.CONNECTIONS_TOKEN_PREFIX + "SyntheticUnknownId.placeholder-secret",
        "connections_wrong_secret": f"{connections_record.token_id}.wrong-secret",
        "workspace": issue("workspace", token_store.WORKSPACE_SURFACE)[1],
        "recommendations": issue("recommendations", token_store.RECOMMENDATIONS_SURFACE)[1],
        "automations": issue("automations", token_store.AUTOMATIONS_SURFACE)[1],
        "models": issue("models", token_store.MODELS_SURFACE)[1],
        "session_as_bearer": web_server._SESSION_TOKEN,
    }
    headers = {who: {"Authorization": f"Bearer {bearer}"} for who, bearer in bearers.items()}
    headers["owner"] = {web_server._SESSION_HEADER_NAME: web_server._SESSION_TOKEN}
    headers["none"] = {}

    providers = [
        WorkspaceReadTokenProvider(),
        RecommendationsReadTokenProvider(),
        ConnectionsManageTokenProvider(),
        AutomationsManageTokenProvider(),
        ModelsManageTokenProvider(),
    ]
    added = [provider for provider in providers if snapshot_registration(provider.name) is None]
    for provider in added:
        register_provider(provider)

    state = web_server.app.state
    saved_attrs = {attr: getattr(state, attr, None) for attr in ("auth_required", "bound_host")}
    state.auth_required = False
    state.bound_host = None
    client = TestClient(web_server.app)

    def call(who, **kwargs):
        return client.get(PATH, headers=headers[who], **kwargs)

    def holds_token(text):
        """True when ``text`` holds the connections token or any run of its secret."""
        return connections in text or _holds_part_of(text, connections.split(".", 1)[1])

    try:
        yield _Quiet(call=call, holds_token=holds_token, web_server=web_server)
    finally:
        client.close()
        _restore_state(state, saved_attrs)
        for provider in added:
            restore_registration(provider.name, provider, None)


def _rows(dash):
    response = dash.call("connections")
    assert response.status_code == 200, response.text
    return response.json()["rows"]


def _audit_text():
    from hermes_cli.dashboard_auth import audit as audit_mod

    path = audit_mod._resolve_log_path()
    return path.read_text(encoding="utf-8") if path.exists() else ""


def _successes(method, path):
    """The token success entries in the audit log for one route."""
    entries = []
    for raw in _audit_text().splitlines():
        try:
            entry = json.loads(raw)
        except ValueError:
            continue
        if (entry.get("event"), entry.get("method"), entry.get("route_template")) == (
            "token_auth_success",
            method,
            path,
        ):
            entries.append(entry)
    return entries


# ---------------------------------------------------------------------------
# 1. Only the row fields
# ---------------------------------------------------------------------------


def test_rows_hold_only_the_row_fields_never_an_address_command_args_env_or_headers(dash, saved):
    _router()
    from hermes_cli.mcp_catalog import list_catalog

    response = dash.call("connections")
    assert response.status_code == 200, response.text
    body = response.json()
    assert list(body) == ["rows"]
    for row in body["rows"]:
        assert list(row) == ROW_FIELDS, row
        assert not {"url", "command", "args", "env", "headers"} & set(row)
        assert row["profile_role"] == ROLES[row["profile"]]
        assert row["status"] in STATUSES
        assert all(isinstance(row[key], bool) for key in ("enabled", "read_only", "catalogue"))
    leaked = [index for index, value in enumerate(saved.secrets) if _holds_part_of(response.text, value)]
    assert leaked == [], "a row holds part of a saved address, command, argument, environment value or header"

    catalogue = {entry.name for entry in list_catalog()}
    assert {"claude-design", "google-drive-work"} <= catalogue
    assert all(row["catalogue"] == (row["name"] in catalogue) for row in body["rows"])
    assert [(row["profile"], row["name"], row["transport"], row["sign_in_kind"]) for row in body["rows"]] == [
        ("default", "google-calendar-work", "http", "oauth"),
        ("default", "google-drive-work", "http", "oauth"),
        ("default", "google-gmail-work", "http", "oauth"),
        ("raphael-designer", "claude-design", "http", "code_sign_in"),
        ("raphael-designer", "github-agent-factory", "stdio", "local_command"),
        ("raphael-builder", "github-agent-factory", "stdio", "local_command"),
        ("raphael-builder", "github-agent-factory-org", "stdio", "local_command"),
        ("raphael-builder", "github-agent-factory-provisioner", "stdio", "local_command"),
        ("raphael-planner", "github-agent-factory", "stdio", "local_command"),
        ("raphael-planner", "github-agent-factory-org", "stdio", "local_command"),
        ("raphael-verifier", "github-agent-factory", "stdio", "local_command"),
        ("raphael-verifier", "github-agent-factory-org", "stdio", "local_command"),
        ("raphael-verifier", "github-independent-reviewer", "stdio", "local_command"),
        ("raphael-claude-worker", "context7", "http", "header_key"),
    ]


# ---------------------------------------------------------------------------
# 2. read_only
# ---------------------------------------------------------------------------


def test_agent_rows_are_read_only_and_owner_rows_are_not(dash, saved):
    _router()
    rows = _rows(dash)

    agent_rows = [row for row in rows if row["profile_role"] == "agent"]
    assert {row["profile"] for row in agent_rows} == {name for name, role in ROLES.items() if role == "agent"}
    assert all(row["read_only"] is True for row in agent_rows)

    owner_rows = [row for row in rows if row["profile_role"] == "owner"]
    # Owner decision: the root holds only Gmail, Calendar and Google Drive Work.
    assert sorted(row["name"] for row in owner_rows) == ["google-calendar-work", "google-drive-work", "google-gmail-work"]
    assert all(row["read_only"] is False for row in owner_rows)
    assert not {row["name"] for row in rows} & set(REMOVED_FROM_ROOT)

    designer = {row["name"]: row["read_only"] for row in rows if row["profile_role"] == "designer"}
    assert designer == {"claude-design": False, "github-agent-factory": True}


# ---------------------------------------------------------------------------
# 3. Status from the saved sign-in and the enabled flag
# ---------------------------------------------------------------------------


def test_an_oauth_row_without_a_saved_sign_in_reads_needs_authorization(dash, saved):
    _router()
    before = _saved_state(saved.root)
    rows = _rows(dash)
    assert _saved_state(saved.root) == before, "the read changed a saved connection or sign-in"

    status = {(row["profile"], row["name"]): row["status"] for row in rows}
    assert status[("default", "google-gmail-work")] == "needs_authorization"
    assert status[("default", "google-calendar-work")] == "paused"
    assert status[("default", "google-drive-work")] == "configured"
    assert status[("raphael-designer", "claude-design")] == "paused"
    assert status[("raphael-designer", "github-agent-factory")] == "configured"
    assert status[("raphael-claude-worker", "context7")] == "configured"
    assert "connected" not in status.values()

    # On and with its sign-in saved, as Claude Design looks on disk while the gateway parks it.
    designer = saved.homes["raphael-designer"]
    servers = saved.servers["raphael-designer"]
    servers["claude-design"].pop("enabled")
    _write_home(designer, servers)
    claude_design = {row["name"]: row for row in _rows(dash) if row["profile"] == "raphael-designer"}["claude-design"]
    assert (claude_design["enabled"], claude_design["status"]) == (True, "configured")

    (designer / "mcp-tokens" / "claude-design.json").unlink()
    before = _saved_state(saved.root)
    claude_design = {row["name"]: row for row in _rows(dash) if row["profile"] == "raphael-designer"}["claude-design"]
    assert claude_design["status"] == "needs_authorization"
    assert _saved_state(saved.root) == before, "the read changed a saved connection or sign-in"


# ---------------------------------------------------------------------------
# 4. Only the connections credential
# ---------------------------------------------------------------------------


def test_only_the_connections_credential_is_admitted(dash, saved):
    _router()
    assert dash.call("connections").status_code == 200

    for who in ("workspace", "recommendations", "automations", "models"):
        response = dash.call(who)
        assert response.status_code == 403, (who, response.status_code, response.text)
        assert "rows" not in response.json()
    for who in (
        "none",
        "owner",
        "session_as_bearer",
        "connections_unknown",
        "connections_wrong_secret",
        "connections_revoked",
    ):
        response = dash.call(who)
        assert response.status_code == 401, (who, response.status_code, response.text)
        assert "rows" not in response.json()

    from hermes_cli.dashboard_auth.token_auth import get_token_route_policy
    from plugins.dashboard_auth.raphael_workspace import CONNECTIONS_SCOPE

    policy = get_token_route_policy("GET", PATH)
    assert (policy.required_scope, policy.optional, policy.strict_audit) == (CONNECTIONS_SCOPE, False, True)


# ---------------------------------------------------------------------------
# 5. A profile off the list is never read
# ---------------------------------------------------------------------------


def test_a_profile_off_the_list_is_never_read(dash, saved, monkeypatch):
    _router()
    from hermes_cli import mcp_config
    from hermes_constants import get_hermes_home

    scoped, read_homes = [], []
    real_scope = dash.web_server._profile_scope
    real_servers = mcp_config._get_mcp_servers
    real_signed_in = mcp_config._oauth_tokens_present

    def scope(profile):
        scoped.append(profile)
        return real_scope(profile)

    def servers(*args, **kwargs):
        read_homes.append(Path(get_hermes_home()).resolve())
        return real_servers(*args, **kwargs)

    def signed_in(name):
        read_homes.append(Path(get_hermes_home()).resolve())
        return real_signed_in(name)

    monkeypatch.setattr(dash.web_server, "_profile_scope", scope)
    monkeypatch.setattr(mcp_config, "_get_mcp_servers", servers)
    monkeypatch.setattr(mcp_config, "_oauth_tokens_present", signed_in)

    plain = dash.call("connections")
    asked = dash.call("connections", params={"profile": OUTSIDER})
    assert (plain.status_code, asked.status_code) == (200, 200)
    assert asked.json() == plain.json()
    assert OUTSIDER not in {row["profile"] for row in plain.json()["rows"]}
    assert "business-crm" not in plain.text

    assert set(scoped) == set(ROLES)
    assert set(read_homes) == {saved.homes[name].resolve() for name in ROLES}


# ---------------------------------------------------------------------------
# 6. Audit
# ---------------------------------------------------------------------------


def test_a_read_writes_one_audit_row(dash, saved):
    _router()
    from plugins.dashboard_auth.raphael_workspace import CONNECTIONS_GRANT, CONNECTIONS_PRINCIPAL

    assert dash.call("workspace").status_code == 403
    assert _successes("GET", PATH) == []

    assert dash.call("connections").status_code == 200
    entries = _successes("GET", PATH)
    assert len(entries) == 1
    entry = entries[0]
    assert (entry["principal"], entry["grant"], entry["source"]) == (
        CONNECTIONS_PRINCIPAL,
        CONNECTIONS_GRANT,
        "raphael-connections",
    )
    assert (entry["decision"], entry["status"]) == ("allow", 200)
    leaked = dash.holds_token(_audit_text())
    assert not leaked, "the audit log holds part of the connections token"
