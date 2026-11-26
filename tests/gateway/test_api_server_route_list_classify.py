"""``_classify_allowed_routes`` reads a role's route list exactly as the gateway does.

For every configuration case of the resolver tests in
test_api_server_route_allowlist.py, the pure classifier given the parsed YAML
root (or ``_CONFIG_ROOT_ABSENT``) returns what ``_resolve_api_server_allowed_routes()``
returns with that config.yaml written, and both equal the expected
``(mode, patterns)``.
"""
from __future__ import annotations

import copy

import pytest
import yaml

from gateway.platforms.api_server import (
    _CONFIG_ROOT_ABSENT,
    _ROUTES_ALLOWLIST,
    _ROUTES_DENY_ALL,
    _ROUTES_UNRESTRICTED,
    _resolve_api_server_allowed_routes,
)
from gateway.platforms.api_server_route_list import _classify_allowed_routes
from hermes_cli.config import get_config_path


def _no_content(case_id, text, expected):
    """No config at all: no file (``text`` is None), or a file with no YAML content."""
    return pytest.param(text, _CONFIG_ROOT_ABSENT, expected, id=case_id)


def _written(case_id, text, expected):
    """A config.yaml holding ``text``; the classifier gets what YAML parses from it."""
    return pytest.param(text, yaml.safe_load(text), expected, id=case_id)


def _allowed_routes(value):
    """The text the resolver tests write for one ``allowed_routes`` value."""
    return yaml.safe_dump({"gateway": {"api_server": {"allowed_routes": value}}})


_CASES = [
    # Unrestricted: nothing configured.
    _no_content("absent", None, (_ROUTES_UNRESTRICTED, [])),
    _no_content("empty-file", "", (_ROUTES_UNRESTRICTED, [])),
    _no_content("blank-lines", "\n   \n", (_ROUTES_UNRESTRICTED, [])),
    _no_content(
        "comments-only",
        "# gateway:\n#   api_server:\n\n#     allowed_routes: []\n",
        (_ROUTES_UNRESTRICTED, []),
    ),
    _written("no-gateway", yaml.safe_dump({"model": {"default": "x"}}), (_ROUTES_UNRESTRICTED, [])),
    _written(
        "no-api-server",
        yaml.safe_dump({"gateway": {"multiplex_profiles": True}}),
        (_ROUTES_UNRESTRICTED, []),
    ),
    _written(
        "no-allowed-routes-key",
        yaml.safe_dump({"gateway": {"api_server": {"port": 8642}}}),
        (_ROUTES_UNRESTRICTED, []),
    ),
    _written("null", _allowed_routes(None), (_ROUTES_UNRESTRICTED, [])),
    _written("empty-list", _allowed_routes([]), (_ROUTES_UNRESTRICTED, [])),
    # Allowlist: a non-empty string or a list of non-empty strings, normalised.
    _written(
        "list",
        _allowed_routes(["/v1/chat/completions", "/v1/models"]),
        (_ROUTES_ALLOWLIST, ["/v1/chat/completions", "/v1/models"]),
    ),
    _written("single-string", _allowed_routes("/v1/models"), (_ROUTES_ALLOWLIST, ["/v1/models"])),
    _written("stripped-entry", _allowed_routes(["  /v1/models  "]), (_ROUTES_ALLOWLIST, ["/v1/models"])),
    _written(
        "list-beside-owner-workspace",
        yaml.safe_dump({
            "gateway": {
                "api_server": {
                    "allowed_routes": ["/v1/models", "/health"],
                    "owner_workspace": {"enabled": True},
                },
            },
        }),
        (_ROUTES_ALLOWLIST, ["/v1/models", "/health"]),
    ),
    _written(
        "method-rules-normalized",
        _allowed_routes([" post   /v1/responses/ ", "GET /v1/responses/{response_id}"]),
        (_ROUTES_ALLOWLIST, ["POST /v1/responses", "GET /v1/responses/{response_id}"]),
    ),
    # Deny-all: every malformed value or entry.
    _written("empty-string", _allowed_routes(""), (_ROUTES_DENY_ALL, [])),
    _written("bool-true", _allowed_routes(True), (_ROUTES_DENY_ALL, [])),
    _written("bool-false", _allowed_routes(False), (_ROUTES_DENY_ALL, [])),
    _written("int-zero", _allowed_routes(0), (_ROUTES_DENY_ALL, [])),
    _written("empty-mapping", _allowed_routes({}), (_ROUTES_DENY_ALL, [])),
    _written("list-with-non-string-entry", _allowed_routes(["/v1/models", 5]), (_ROUTES_DENY_ALL, [])),
    _written("list-with-empty-string-entry", _allowed_routes(["", "/v1/models"]), (_ROUTES_DENY_ALL, [])),
    _written("gateway-not-a-mapping", yaml.safe_dump({"gateway": "oops"}), (_ROUTES_DENY_ALL, [])),
    _written(
        "api-server-not-a-mapping",
        yaml.safe_dump({"gateway": {"api_server": "oops"}}),
        (_ROUTES_DENY_ALL, []),
    ),
    _written("unknown-method", _allowed_routes(["TRACE /v1/responses"]), (_ROUTES_DENY_ALL, [])),
    _written("relative-template", _allowed_routes(["POST relative/path"]), (_ROUTES_DENY_ALL, [])),
    _written("query-string", _allowed_routes(["POST /v1/responses?x=1"]), (_ROUTES_DENY_ALL, [])),
    _written("space-in-template", _allowed_routes(["POST /v1/responses fragment"]), (_ROUTES_DENY_ALL, [])),
    # Deny-all: a root that parsed but is not a mapping.
    _written("list-root", "- just\n- a\n- list\n", (_ROUTES_DENY_ALL, [])),
    _written("string-root", "just-a-string\n", (_ROUTES_DENY_ALL, [])),
    _written("int-root", "42\n", (_ROUTES_DENY_ALL, [])),
    _written("float-root", "3.14\n", (_ROUTES_DENY_ALL, [])),
    _written("bool-root", "true\n", (_ROUTES_DENY_ALL, [])),
    _written("explicit-null-root", "null\n", (_ROUTES_DENY_ALL, [])),
    _written("tilde-null-root", "~\n", (_ROUTES_DENY_ALL, [])),
    _written(
        "list-root-wrapping-real-keys",
        "- gateway:\n    api_server:\n      allowed_routes: /v1/models\n",
        (_ROUTES_DENY_ALL, []),
    ),
]


@pytest.mark.parametrize(("text", "parsed", "expected"), _CASES)
def test_classifier_matches_the_resolver_with_that_file_written(text, parsed, expected):
    path = get_config_path()
    if text is None:
        assert not path.exists()
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")

    assert _classify_allowed_routes(parsed) == _resolve_api_server_allowed_routes() == expected


def test_classifier_reads_no_file_and_leaves_its_input_unchanged(monkeypatch):
    import gateway.platforms.api_server as api_server_mod
    import hermes_cli.config as config_mod

    def _no_read(*_args, **_kwargs):
        raise AssertionError("the classifier must not read a config file")

    monkeypatch.setattr(api_server_mod, "_read_raw_config_root", _no_read)
    monkeypatch.setattr(config_mod, "get_config_path", _no_read)
    raw = {"gateway": {"api_server": {"allowed_routes": [" get  /v1/models/ ", "/v1/chat"]}}}
    before = copy.deepcopy(raw)

    first = _classify_allowed_routes(raw)
    second = _classify_allowed_routes(raw)

    assert first == second == (_ROUTES_ALLOWLIST, ["GET /v1/models", "/v1/chat"])
    assert raw == before
    assert first[1] is not second[1]
