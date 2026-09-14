"""Two multiplexed profiles that name the same MCP server with different credentials are two
connections (#106005, #91654): the ledgers in ``tools.mcp_tool`` are keyed per owning profile
scope, and an owner's scoped reload re-registers the profiles that had adopted its connection."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from hermes_constants import hermes_home_key, reset_hermes_home_override, set_hermes_home_override


def _tool(name="t"):
    return SimpleNamespace(name=name, description="d", inputSchema={"type": "object", "properties": {}},
                           annotations=None)


def _server(name, cfg, tool_name="t"):
    return SimpleNamespace(name=name, session=object(), _config=cfg, _tools=[_tool(tool_name)], tool_timeout=30,
                           initialize_result=None, _registered_tool_names=[], _sampling=None)


def _register(name, cfg, tool_name="t"):
    from tools import mcp_tool_discovery as disc, mcp_tool_registration as reg

    disc._select_new_servers({name: cfg})
    server = _server(name, cfg, tool_name)
    disc._adopt_server(name, server)
    server._registered_tool_names = reg._register_server_tools(name, server, cfg)
    return server


@pytest.fixture
def two_profiles(tmp_path, monkeypatch):
    """Multiplex on, clean MCP ledgers, a scope switcher for homes A and B; restores everything."""
    import tools.mcp_tool as core
    from tools import mcp_tool_config as _config
    from tools.registry import registry

    homes = {k: tmp_path / "profiles" / k for k in ("a", "b")}
    for home in homes.values():
        home.mkdir(parents=True)
    monkeypatch.setattr("agent.secret_scope.is_multiplex_active", lambda: True)
    monkeypatch.setattr(core, "_ensure_mcp_sdk", lambda: True)
    monkeypatch.setattr(_config, "_filter_suspicious_mcp_servers", lambda servers: servers)
    ledgers = ("_servers", "_server_scope_keys", "_server_tool_scopes", "_server_connecting",
               "_server_connect_errors", "_server_connect_retry_after", "_server_connect_failures",
               "_server_error_counts", "_server_breaker_opened_at", "_lazy_server_configs",
               "_mcp_tool_server_names", "_orphaned_adopters", "_parallel_safe_servers",
               "_server_trust_levels", "_tool_read_only_hints")
    saved = {n: type(getattr(core, n))(getattr(core, n)) for n in ledgers}
    for n in ledgers:
        getattr(core, n).clear()
    tokens = []

    def enter(which):
        tokens.append(set_hermes_home_override(homes[which]))
        return hermes_home_key(homes[which])

    yield enter
    for home in homes.values():
        scope = hermes_home_key(home)
        for entry in registry._snapshot_state(scope)[0]:
            if entry.toolset.startswith("mcp-"):
                registry.deregister(entry.name, scope=scope)
    for token in reversed(tokens):
        reset_hermes_home_override(token)
    for n in ledgers:
        getattr(core, n).clear()
        getattr(core, n).update(saved[n])


def test_same_named_server_with_other_credentials_is_a_separate_connection(two_profiles):
    import tools.mcp_tool as core
    from tools import mcp_tool_discovery as disc, mcp_tool_handlers as handlers
    from tools import mcp_tool_registration as reg
    from tools.registry import registry
    import toolsets

    cfg_a = {"url": "https://mcp.example/x", "headers": {"Authorization": "Bearer A"}}
    cfg_b = {"url": "https://mcp.example/x", "headers": {"Authorization": "Bearer B"}}

    two_profiles("a")
    srv_a = _server("x", cfg_a)
    disc._adopt_server("x", srv_a)
    srv_a._registered_tool_names = reg._register_server_tools("x", srv_a, cfg_a)
    assert toolsets.resolve_toolset("mcp-x") == ["mcp__x__t"]
    for _ in range(core._CIRCUIT_BREAKER_THRESHOLD):
        core._bump_server_error("x")
    disc._note_connect_failure("y", RuntimeError("boom"))

    two_profiles("b")
    # B's own view: no tools yet, its memo is not A's, and A's connection is not "connected" for B.
    assert registry.get_tool_names_for_toolset("mcp-x") == []
    assert toolsets.resolve_toolset("mcp-x") == []
    assert disc.get_mcp_status({"x": cfg_b})[0]["status"] == "configured"
    # B's differently-authenticated 'x' is a connect candidate, not shadowed by A's ledger entries.
    assert "x" in disc._select_new_servers({"x": cfg_b})
    assert not disc._connect_cooldown_active("y")
    assert handlers._check_circuit_breaker("x") is None


def test_oauth_server_is_not_adopted_across_profiles(two_profiles):
    from tools import mcp_tool_discovery as disc
    from tools import mcp_tool_registration as reg
    from tools.registry import registry

    cfg = {"url": "https://mcp.example/x", "auth": "oauth"}

    two_profiles("a")
    srv_a = _server("x", cfg)
    disc._adopt_server("x", srv_a)
    srv_a._registered_tool_names = reg._register_server_tools("x", srv_a, cfg)
    assert reg.register_connected_into_current_scope({"x": dict(cfg)}) == 0
    assert registry.get_tool_names_for_toolset("mcp-x") == ["mcp__x__t"]

    two_profiles("b")
    assert reg.register_connected_into_current_scope({"x": dict(cfg)}) == 0
    assert registry.get_tool_names_for_toolset("mcp-x") == []
    assert "x" in disc._select_new_servers({"x": dict(cfg)})


def test_same_named_server_with_other_mtls_identity_is_a_separate_connection(two_profiles):
    from tools import mcp_tool_discovery as disc
    from tools import mcp_tool_registration as reg

    cfg_a = {
        "url": "https://mcp.example/x",
        "client_cert": "/certs/profile-a.pem",
        "client_key": "/certs/profile-a.key",
    }
    cfg_b = {
        "url": "https://mcp.example/x",
        "client_cert": "/certs/profile-b.pem",
        "client_key": "/certs/profile-b.key",
    }

    two_profiles("a")
    srv_a = _server("x", cfg_a)
    disc._adopt_server("x", srv_a)
    srv_a._registered_tool_names = reg._register_server_tools("x", srv_a, cfg_a)

    two_profiles("b")
    reg.register_connected_into_current_scope({"x": cfg_b})
    assert "x" in disc._select_new_servers({"x": cfg_b})


def test_owner_reload_reregisters_profiles_that_adopted_its_connection(two_profiles):
    import tools.mcp_tool as core
    from tools import mcp_tool_discovery as disc, mcp_tool_lifecycle as lifecycle
    from tools import mcp_tool_registration as reg
    from tools.registry import registry

    cfg = {"url": "https://mcp.example/x", "headers": {"Authorization": "Bearer shared"}}
    scope_a = two_profiles("a")
    srv_a = _server("x", cfg)
    disc._adopt_server("x", srv_a)
    srv_a._registered_tool_names = reg._register_server_tools("x", srv_a, cfg)

    two_profiles("b")
    assert reg.register_connected_into_current_scope({"x": cfg}) == 1
    assert registry.get_tool_names_for_toolset("mcp-x") == ["mcp__x__t"]

    # Owner A: scoped shutdown (no MCP loop here, so emulate the task teardown), then rediscovery.
    two_profiles("a")
    with patch.object(lifecycle._loop, "_stop_mcp_loop", lambda **_kw: False):
        lifecycle.shutdown_mcp_servers(scope=scope_a)
    for tool_name in list(srv_a._registered_tool_names):
        reg._deregister_mcp_tool_all_scopes(srv_a, tool_name)
    with core._lock:
        for key in [k for k, v in core._servers.items() if v is srv_a]:
            core._servers.pop(key)
            core._server_scope_keys.pop(key, None)
            core._server_tool_scopes.pop(key, None)

    def fake_pass(new_servers):
        for name, config in new_servers.items():
            srv = _server(name, config)
            disc._adopt_server(name, srv)
            srv._registered_tool_names = reg._register_server_tools(name, srv, config)

    with patch.object(disc, "_run_discovery_pass", fake_pass), \
            patch.object(disc._loop, "_ensure_mcp_loop", lambda: None), \
            patch("tools.mcp_tool_config._load_mcp_config", lambda: {"x": cfg}):
        disc.register_mcp_servers({"x": cfg})

    # B never reloaded, yet has its tools back on the owner's new identical connection.
    two_profiles("b")
    assert registry.get_tool_names_for_toolset("mcp-x") == ["mcp__x__t"]
    assert disc.get_mcp_status({"x": cfg})[0]["status"] == "connected"


def test_untrusted_adopter_of_a_full_profiles_connection_keeps_its_own_trust_gate(two_profiles, monkeypatch):
    """Trust is the consuming profile's policy: adopting A's ``trust: full`` connection must not let
    B's ``trust: untrusted`` write-capable call skip approval."""
    from tools import mcp_tool_discovery as disc, mcp_tool_handlers as handlers
    from tools import mcp_tool_registration as reg
    import tools.approval as approval_prompt

    route = {"url": "https://mcp.example/x", "headers": {"Authorization": "Bearer shared"}}
    cfg_a, cfg_b = dict(route, trust="full"), dict(route, trust="untrusted")
    asked = []
    monkeypatch.setattr(approval_prompt, "request_elicitation_consent",
                        lambda *a, **k: asked.append(a) or "deny")

    two_profiles("a")
    srv_a = _server("x", cfg_a)
    disc._adopt_server("x", srv_a)
    srv_a._registered_tool_names = reg._register_server_tools("x", srv_a, cfg_a)

    two_profiles("b")
    assert reg.register_connected_into_current_scope({"x": cfg_b}) == 1
    assert handlers._trust_gate_check("x", "t") is not None and asked

    two_profiles("a")
    assert handlers._trust_gate_check("x", "t") is None and len(asked) == 1


@pytest.mark.parametrize("order", [("a", "b"), ("b", "a")])
@pytest.mark.parametrize("parallel_a", [False, True])
def test_parallel_safe_opt_in_is_per_profile(two_profiles, order, parallel_a):
    """B's ``supports_parallel_tool_calls`` on its own same-named server never makes A's serial
    server's tool parallel-safe (the batch planner would run two A calls concurrently)."""
    from tools import mcp_tool_discovery as disc

    policies = {"a": parallel_a, "b": not parallel_a}
    for profile in order:
        two_profiles(profile)
        _register("x", {"url": "https://mcp.example/x", "headers": {"Authorization": profile},
                        "supports_parallel_tool_calls": policies[profile]})
    for profile, expected in policies.items():
        two_profiles(profile)
        assert disc.is_mcp_tool_parallel_safe("mcp__x__t") is expected


@pytest.mark.parametrize("extra_a,extra_b", [
    ({}, {"ssl_verify": False}),
    ({}, {"strict_redirect_headers": True}),
    ({"identity_header": {"name": "X-Identity", "value": "a"}},
     {"identity_header": {"name": "X-Identity", "value": "b"}}),
])
def test_different_connection_inputs_are_not_adopted(two_profiles, extra_a, extra_b):
    from tools import mcp_tool_discovery as disc, mcp_tool_registration as reg
    from tools.registry import registry

    cfg_a = {"url": "https://mcp.example/x", **extra_a}
    cfg_b = {"url": "https://mcp.example/x", **extra_b}
    two_profiles("a")
    _register("x", cfg_a)
    two_profiles("b")
    assert reg.register_connected_into_current_scope({"x": cfg_b}) == 0
    assert registry.get_tool_names_for_toolset("mcp-x") == []
    assert "x" in disc._select_new_servers({"x": cfg_b})


@pytest.mark.parametrize("cfg", [
    {"command": "python3", "args": ["server.py"]},
    {"url": "https://mcp.example/x", "identity_header": {"name": "X-Profile", "value_from": "profile"}},
])
def test_implicit_profile_authority_is_not_shared(two_profiles, cfg):
    from tools import mcp_tool_discovery as disc, mcp_tool_registration as reg
    from tools.registry import registry

    two_profiles("a")
    _register("x", cfg)
    two_profiles("b")
    assert reg.register_connected_into_current_scope({"x": cfg}) == 0
    assert registry.get_tool_names_for_toolset("mcp-x") == []
    assert "x" in disc._select_new_servers({"x": cfg})


def test_stdio_cwd_change_does_not_reuse_a_shared_process(two_profiles):
    from tools import mcp_tool_discovery as disc, mcp_tool_registration as reg

    cfg = {"command": "python3", "args": ["server.py"], "cwd": "/profile-a"}
    two_profiles("a")
    _register("x", cfg)
    two_profiles("b")
    changed = dict(cfg, cwd="/profile-b")
    assert reg.register_connected_into_current_scope({"x": changed}) == 0
    assert "x" in disc._select_new_servers({"x": changed})


def test_tool_provenance_cannot_be_rebound_by_a_sibling_profile(two_profiles):
    from tools import mcp_tool_discovery as disc

    two_profiles("a")
    _register("foo-bar", {"url": "https://mcp.example/serial"})
    _register("foo_bar", {"url": "https://mcp.example/parallel", "supports_parallel_tool_calls": True}, "other")
    assert disc.is_mcp_tool_parallel_safe("mcp__foo_bar__t") is False
    assert disc.is_mcp_tool_parallel_safe("mcp__foo_bar__other") is True
    two_profiles("b")
    _register("foo_bar", {"url": "https://mcp.example/b", "supports_parallel_tool_calls": True})
    assert disc.is_mcp_tool_parallel_safe("mcp__foo_bar__t") is True
    two_profiles("a")
    assert disc.is_mcp_tool_parallel_safe("mcp__foo_bar__t") is False
    assert disc.is_mcp_tool_parallel_safe("mcp__foo_bar__other") is True


def test_adopter_shutdown_clears_only_its_policy_and_provenance(two_profiles, monkeypatch):
    import tools.mcp_tool as core
    from tools import mcp_tool_discovery as disc, mcp_tool_lifecycle as lifecycle
    from tools import mcp_tool_registration as reg
    from tools.registry import registry

    cfg = {"url": "https://mcp.example/x", "supports_parallel_tool_calls": True}
    scope_a = two_profiles("a")
    owner = _register("x", cfg)
    scope_b = two_profiles("b")
    assert reg.register_connected_into_current_scope({"x": cfg}) == 1
    disc._select_new_servers({"x": cfg})
    assert disc.is_mcp_tool_parallel_safe("mcp__x__t") is True
    two_profiles("a")
    monkeypatch.setattr(lifecycle._loop, "_stop_mcp_loop", lambda **kw: False)
    lifecycle.shutdown_mcp_servers(scope=scope_b)
    assert registry.snapshot_registration("mcp__x__t", scope=scope_b) is None
    assert registry.snapshot_registration("mcp__x__t", scope=scope_a) is not None
    assert core._servers[(scope_a, "x")] is owner
    assert disc.is_mcp_tool_parallel_safe("mcp__x__t") is True
    two_profiles("b")
    assert disc.is_mcp_tool_parallel_safe("mcp__x__t") is False


@pytest.mark.live_system_guard_bypass
def test_real_stdio_calls_use_each_profiles_external_secret(tmp_path, monkeypatch):
    """Identical config must not reuse a process launched with a sibling's scoped secret."""
    import sys
    from contextlib import contextmanager

    from agent.secret_scope import (build_profile_secret_scope, is_multiplex_active,
                                    reset_secret_scope, set_multiplex_active, set_secret_scope)
    from tools import mcp_tool_config as config, mcp_tool_discovery as disc
    from tools import mcp_tool_lifecycle as lifecycle
    from tools.registry import registry

    server = tmp_path / "scope_server.py"
    server.write_text('''import asyncio, os
import mcp.types as types
from mcp.server.lowlevel import Server
from mcp.server.stdio import stdio_server
async def tools(context, params):
    tool = types.Tool.model_validate({"name":"identity","description":"Read the test marker","inputSchema":{"type":"object","properties":{}}})
    return types.ListToolsResult(tools=[tool])
async def call(context, params):
    return types.CallToolResult(content=[types.TextContent(type="text", text=os.environ.get("MCP_SCOPE_PROBE", "missing"))])
server = Server("scope-probe", on_list_tools=tools, on_call_tool=call)
async def main():
    async with stdio_server() as (reader, writer):
        await server.run(reader, writer, server.create_initialization_options())
asyncio.run(main())
''')
    homes = {name: tmp_path / "profiles" / name for name in ("a", "b")}
    for name, home in homes.items():
        home.mkdir(parents=True)
        (home / ".env").write_text(f"MCP_SCOPE_PROBE=scope-{name}-private-marker\n")

    @contextmanager
    def profile(name):
        home_token = set_hermes_home_override(homes[name])
        secret_token = set_secret_scope(build_profile_secret_scope(homes[name]))
        try:
            yield
        finally:
            reset_secret_scope(secret_token)
            reset_hermes_home_override(home_token)

    previous_multiplex = is_multiplex_active()
    set_multiplex_active(True)
    # Simulate only the external secret source's name inventory; actual scoped
    # lookup, subprocess launch, MCP wire calls and shutdown stay real.
    monkeypatch.setattr("hermes_cli.env_loader.secret_source_names", lambda: {"MCP_SCOPE_PROBE"})
    monkeypatch.setattr(config, "_filter_suspicious_mcp_servers", lambda servers: servers)
    cfg = {"command": sys.executable, "args": [str(server)], "connect_timeout": 15}
    tool_name = "mcp__scopeprobe__identity"
    try:
        for name in homes:
            with profile(name):
                registered = disc.register_mcp_servers({"scopeprobe": cfg})
                assert tool_name in registered, (homes[name] / "logs/mcp-stderr.log").read_text()[-3000:]
                result = registry.get_entry(tool_name).handler({})
                assert f"scope-{name}-private-marker" in result, result
        with profile("b"):
            lifecycle.shutdown_mcp_servers(scope=hermes_home_key(homes["a"]))
            result = registry.get_entry(tool_name).handler({})
            assert "scope-b-private-marker" in result, result
    finally:
        lifecycle.shutdown_mcp_servers()
        set_multiplex_active(previous_multiplex)
        for home in homes.values():
            handle = config._mcp_stderr_log_fh.pop(hermes_home_key(home), None)
            if handle is not None and handle not in (sys.stderr, sys.stdout):
                handle.close()
