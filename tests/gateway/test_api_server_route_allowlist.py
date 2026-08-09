"""Tests for the API-server route allowlist boundary
(gateway/platforms/api_server.py's ``gateway.api_server.allowed_routes``
least-privilege gate and the sibling ``owner_workspace`` admission flag that
lives in the same boundary section).

Covers:
  - ``_resolve_api_server_allowed_routes()`` — real config.yaml resolution
    (E2E against an isolated HERMES_HOME, no mocking of the config loader)
    for every documented mode: unrestricted, deny_all, allowlist.
  - ``_route_matches_any()`` — segment-aware prefix matching.
  - ``_owner_workspace_toolset_enabled()`` — default-off, fail-closed flag.
  - The admission boundary: owner_workspace is reachable ONLY through the
    dedicated api_server flag, never via generic ``platform_toolsets``
    naming on any platform (including api_server itself), and toggling the
    flag has zero effect on any other toolset or on unrelated route
    patterns.
"""
from __future__ import annotations

import yaml
import pytest

from gateway.platforms.api_server import (
    _owner_workspace_toolset_enabled,
    _resolve_api_server_allowed_routes,
    _route_matches_any,
    _ROUTES_ALLOWLIST,
    _ROUTES_DENY_ALL,
    _ROUTES_UNRESTRICTED,
)
from hermes_cli.config import get_config_path
from hermes_cli.tools_config import _get_platform_tools


def _write_config(data: dict) -> None:
    path = get_config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data), encoding="utf-8")


def _write_raw_yaml(text: str) -> None:
    path = get_config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


# ---------------------------------------------------------------------------
# _resolve_api_server_allowed_routes — real config.yaml, isolated HERMES_HOME
# ---------------------------------------------------------------------------


class TestResolveApiServerAllowedRoutesUnrestricted:
    def test_no_config_file_at_all(self):
        assert _resolve_api_server_allowed_routes() == (_ROUTES_UNRESTRICTED, [])

    def test_no_gateway_key(self):
        _write_config({"model": {"default": "x"}})
        assert _resolve_api_server_allowed_routes() == (_ROUTES_UNRESTRICTED, [])

    def test_no_api_server_key(self):
        _write_config({"gateway": {"multiplex_profiles": True}})
        assert _resolve_api_server_allowed_routes() == (_ROUTES_UNRESTRICTED, [])

    def test_no_allowed_routes_key(self):
        _write_config({"gateway": {"api_server": {"port": 8642}}})
        assert _resolve_api_server_allowed_routes() == (_ROUTES_UNRESTRICTED, [])

    def test_allowed_routes_explicit_null(self):
        _write_config({"gateway": {"api_server": {"allowed_routes": None}}})
        assert _resolve_api_server_allowed_routes() == (_ROUTES_UNRESTRICTED, [])

    def test_allowed_routes_empty_list(self):
        _write_config({"gateway": {"api_server": {"allowed_routes": []}}})
        assert _resolve_api_server_allowed_routes() == (_ROUTES_UNRESTRICTED, [])


class TestResolveApiServerAllowedRoutesAllowlist:
    def test_list_of_patterns(self):
        _write_config({
            "gateway": {"api_server": {"allowed_routes": ["/v1/chat/completions", "/v1/models"]}},
        })
        mode, patterns = _resolve_api_server_allowed_routes()
        assert mode == _ROUTES_ALLOWLIST
        assert patterns == ["/v1/chat/completions", "/v1/models"]

    def test_single_string_pattern(self):
        _write_config({"gateway": {"api_server": {"allowed_routes": "/v1/models"}}})
        assert _resolve_api_server_allowed_routes() == (_ROUTES_ALLOWLIST, ["/v1/models"])

    def test_patterns_are_stripped(self):
        _write_config({"gateway": {"api_server": {"allowed_routes": ["  /v1/models  "]}}})
        assert _resolve_api_server_allowed_routes() == (_ROUTES_ALLOWLIST, ["/v1/models"])


class TestResolveApiServerAllowedRoutesDenyAll:
    def test_empty_string_is_malformed(self):
        _write_config({"gateway": {"api_server": {"allowed_routes": ""}}})
        assert _resolve_api_server_allowed_routes() == (_ROUTES_DENY_ALL, [])

    def test_bool_true_is_malformed(self):
        _write_config({"gateway": {"api_server": {"allowed_routes": True}}})
        assert _resolve_api_server_allowed_routes() == (_ROUTES_DENY_ALL, [])

    def test_bool_false_is_malformed(self):
        _write_config({"gateway": {"api_server": {"allowed_routes": False}}})
        assert _resolve_api_server_allowed_routes() == (_ROUTES_DENY_ALL, [])

    def test_int_is_malformed(self):
        _write_config({"gateway": {"api_server": {"allowed_routes": 0}}})
        assert _resolve_api_server_allowed_routes() == (_ROUTES_DENY_ALL, [])

    def test_dict_is_malformed(self):
        _write_config({"gateway": {"api_server": {"allowed_routes": {}}}})
        assert _resolve_api_server_allowed_routes() == (_ROUTES_DENY_ALL, [])

    def test_list_with_non_string_entry(self):
        _write_config({"gateway": {"api_server": {"allowed_routes": ["/v1/models", 5]}}})
        assert _resolve_api_server_allowed_routes() == (_ROUTES_DENY_ALL, [])

    def test_list_with_empty_string_entry(self):
        _write_config({"gateway": {"api_server": {"allowed_routes": ["", "/v1/models"]}}})
        assert _resolve_api_server_allowed_routes() == (_ROUTES_DENY_ALL, [])

    def test_gateway_not_a_dict(self):
        _write_config({"gateway": "oops"})
        assert _resolve_api_server_allowed_routes() == (_ROUTES_DENY_ALL, [])

    def test_api_server_not_a_dict(self):
        _write_config({"gateway": {"api_server": "oops"}})
        assert _resolve_api_server_allowed_routes() == (_ROUTES_DENY_ALL, [])

    @pytest.mark.parametrize(
        "raw_yaml",
        [
            pytest.param("- just\n- a\n- list\n", id="list-root"),
            pytest.param("just-a-string\n", id="string-root"),
            pytest.param("42\n", id="int-root"),
            pytest.param("3.14\n", id="float-root"),
            pytest.param("true\n", id="bool-root"),
            pytest.param("null\n", id="explicit-null-root"),
            pytest.param("~\n", id="tilde-null-root"),
            pytest.param("- gateway:\n    api_server:\n      allowed_routes: /v1/models\n",
                         id="list-root-wrapping-real-keys"),
        ],
    )
    def test_non_mapping_yaml_root_denies_all(self, raw_yaml):
        """A config.yaml that PARSED FINE but whose root is not a mapping is
        malformed, not unconfigured. Collapsing it to ``{}`` (as the generic
        raw-config reader does) made ``gateway`` look merely absent and
        resolved the boundary to UNRESTRICTED — a real file could silently
        disable the least-privilege gate. Every non-mapping root must deny."""
        _write_raw_yaml(raw_yaml)
        assert _resolve_api_server_allowed_routes() == (_ROUTES_DENY_ALL, [])

    def test_unparseable_yaml_fails_closed(self):
        _write_raw_yaml("gateway: {api_server: [unterminated\n")
        assert _resolve_api_server_allowed_routes() == (_ROUTES_DENY_ALL, [])

    def test_unreadable_config_fails_closed(self, monkeypatch):
        """A config that cannot be read at all is a load failure, and a load
        failure is never allowed to read as "nothing configured"."""
        import gateway.platforms.api_server as api_server_mod

        def _raise():
            raise OSError("disk on fire")

        monkeypatch.setattr(api_server_mod, "_read_raw_config_root", _raise)
        assert _resolve_api_server_allowed_routes() == (_ROUTES_DENY_ALL, [])


# ---------------------------------------------------------------------------
# _route_matches_any
# ---------------------------------------------------------------------------


class TestRouteMatchesAny:
    def test_exact_match(self):
        assert _route_matches_any("/v1/runs", ["/v1/runs"]) is True

    def test_descendant_matches(self):
        assert _route_matches_any("/v1/runs/abc123", ["/v1/runs"]) is True

    def test_sibling_with_shared_string_prefix_does_not_match(self):
        assert _route_matches_any("/v1/runs-evil", ["/v1/runs"]) is False

    def test_unrelated_path_does_not_match(self):
        assert _route_matches_any("/api/sessions", ["/v1/runs"]) is False

    def test_trailing_slash_on_pattern_is_normalized(self):
        assert _route_matches_any("/v1/runs", ["/v1/runs/"]) is True

    def test_no_patterns_never_matches(self):
        assert _route_matches_any("/v1/runs", []) is False

    def test_matches_if_any_pattern_matches(self):
        assert _route_matches_any("/v1/models", ["/v1/runs", "/v1/models"]) is True


# ---------------------------------------------------------------------------
# _owner_workspace_toolset_enabled — default-off, fail-closed
# ---------------------------------------------------------------------------


class TestOwnerWorkspaceToolsetEnabledFlag:
    def test_default_disabled_on_empty_config(self):
        assert _owner_workspace_toolset_enabled({}) is False

    def test_default_disabled_when_api_server_section_present_but_key_absent(self):
        assert _owner_workspace_toolset_enabled({"gateway": {"api_server": {}}}) is False

    def test_explicit_true_enables(self):
        cfg = {"gateway": {"api_server": {"owner_workspace": {"enabled": True}}}}
        assert _owner_workspace_toolset_enabled(cfg) is True

    def test_explicit_false_stays_disabled(self):
        cfg = {"gateway": {"api_server": {"owner_workspace": {"enabled": False}}}}
        assert _owner_workspace_toolset_enabled(cfg) is False

    @pytest.mark.parametrize(
        "value",
        [
            pytest.param("true", id="str-true"),
            pytest.param("True", id="str-True"),
            pytest.param("yes", id="str-yes"),
            pytest.param("on", id="str-on"),
            pytest.param("false", id="str-false"),
            pytest.param("0", id="str-zero"),
            pytest.param(1, id="int-one"),
            pytest.param(1.0, id="float-one"),
            pytest.param(["owner_workspace"], id="non-empty-list"),
            pytest.param({"enabled": True}, id="non-empty-mapping"),
            pytest.param(None, id="explicit-null"),
        ],
    )
    def test_malformed_truthy_values_stay_disabled(self, value):
        """Only the literal boolean ``True`` opens this owner-mutation
        surface. A ``bool()`` coercion would turn ordinary YAML quoting slips
        (``enabled: "false"`` — a non-empty, therefore truthy, string) into a
        live mutation surface, so every non-``True`` value must stay off."""
        cfg = {"gateway": {"api_server": {"owner_workspace": {"enabled": value}}}}
        assert _owner_workspace_toolset_enabled(cfg) is False

    def test_resolution_error_fails_closed_to_disabled(self, monkeypatch):
        import hermes_cli.config as config_mod

        def _raise(*args, **kwargs):
            raise RuntimeError("boom")

        monkeypatch.setattr(config_mod, "cfg_get", _raise)
        cfg = {"gateway": {"api_server": {"owner_workspace": {"enabled": True}}}}
        assert _owner_workspace_toolset_enabled(cfg) is False


# ---------------------------------------------------------------------------
# Admission boundary: owner_workspace is reachable ONLY via the dedicated
# api_server flag — never via platform_toolsets, on any platform.
# ---------------------------------------------------------------------------


class TestOwnerWorkspaceAdmissionBoundary:
    @pytest.mark.parametrize("platform", ["cli", "telegram", "cron"])
    def test_flag_enabled_has_no_effect_on_other_platforms(self, platform):
        """The owner_workspace.enabled flag is read ONLY by api_server's own
        _create_agent(); _get_platform_tools() (shared by every platform,
        including cli/telegram/cron) never looks at it and always strips
        the kernel-gated toolset regardless of platform_toolsets naming it."""
        config = {
            "gateway": {"api_server": {"owner_workspace": {"enabled": True}}},
            "platform_toolsets": {platform: [f"hermes-{platform}", "owner_workspace"]},
        }
        enabled = _get_platform_tools(config, platform)
        assert "owner_workspace" not in enabled
        assert "owner_workspace_bootstrap" not in enabled
        assert "owner_task_move" not in enabled
        assert "owner_task_comment" not in enabled

    def test_api_server_generic_resolution_never_admits_it_either(self):
        """_get_platform_tools() alone — without the dedicated union
        _create_agent() applies afterward — never admits owner_workspace,
        even for api_server itself and even when explicitly named."""
        config = {
            "gateway": {"api_server": {"owner_workspace": {"enabled": True}}},
            "platform_toolsets": {"api_server": ["hermes-api-server", "owner_workspace"]},
        }
        enabled = _get_platform_tools(config, "api_server")
        assert "owner_workspace" not in enabled

    def test_default_is_disabled_for_api_server_too(self):
        config = {"platform_toolsets": {"api_server": ["hermes-api-server"]}}
        assert _owner_workspace_toolset_enabled(config) is False
        assert "owner_workspace" not in _get_platform_tools(config, "api_server")

    @pytest.mark.parametrize("platform", ["cli", "telegram", "cron"])
    def test_unrelated_toolsets_are_unaffected_by_the_flag(self, platform):
        """Toggling owner_workspace.enabled must not change ANY other
        toolset's admission for ANY platform — it is read by exactly one
        call site (api_server's _create_agent), nowhere else."""
        base_config = {"platform_toolsets": {platform: [f"hermes-{platform}", "terminal"]}}
        flagged_config = {
            **base_config,
            "gateway": {"api_server": {"owner_workspace": {"enabled": True}}},
        }
        assert _get_platform_tools(base_config, platform) == _get_platform_tools(flagged_config, platform)

    def test_unrelated_route_patterns_are_unaffected_by_owner_workspace_flag(self):
        """The allowed_routes gate and the owner_workspace flag are
        independent axes of the same config section — setting one must not
        perturb resolution of the other."""
        _write_config({
            "gateway": {
                "api_server": {
                    "allowed_routes": ["/v1/models", "/health"],
                    "owner_workspace": {"enabled": True},
                },
            },
        })
        mode, patterns = _resolve_api_server_allowed_routes()
        assert mode == _ROUTES_ALLOWLIST
        assert patterns == ["/v1/models", "/health"]
