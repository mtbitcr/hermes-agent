"""The route-roles file declares, for every registered API route, which roles serve it."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import gateway.platforms.api_server as api_server
from gateway.config import PlatformConfig
from gateway.platforms.api_server import APIServerAdapter, _normalize_api_route_rule
from hermes_cli.profiles import _PROFILE_ID_RE

ROUTE_PROFILES_PATH = (
    Path(api_server.__file__).resolve().parent / "api_server_route_profiles.json"
)


def _load_route_profiles() -> dict:
    def no_duplicate_keys(pairs):
        keys = [key for key, _value in pairs]
        duplicates = sorted({key for key in keys if keys.count(key) > 1})
        assert not duplicates, f"routes listed more than once: {duplicates}"
        return dict(pairs)

    return json.loads(
        ROUTE_PROFILES_PATH.read_text(encoding="utf-8"),
        object_pairs_hook=no_duplicate_keys,
    )


def _route_table_keys(adapter: APIServerAdapter) -> set:
    return {f"{method} {path}" for method, path, _handler in adapter._http_route_table()}


def _compare_routes(declared: set, registered: set) -> "list[str]":
    """Name every route that is registered but not declared, or declared but not registered."""
    return [f"registered but not in the file: {route}" for route in sorted(registered - declared)] + [
        f"in the file but not registered: {route}" for route in sorted(declared - registered)
    ]


@pytest.fixture
def adapter(monkeypatch):
    monkeypatch.setattr(api_server, "_CRON_AVAILABLE", True)
    return APIServerAdapter(PlatformConfig(enabled=True))


def test_file_keys_equal_route_table(adapter):
    problems = _compare_routes(set(_load_route_profiles()), _route_table_keys(adapter))
    assert not problems, "\n".join(problems)


def test_comparison_reports_a_route_added_to_the_table(adapter, monkeypatch):
    original = APIServerAdapter._http_route_table
    extra = ("GET", "/v1/an-added-route/{item}", lambda request: None)
    monkeypatch.setattr(
        APIServerAdapter, "_http_route_table", lambda self: original(self) + [extra]
    )
    problems = _compare_routes(set(_load_route_profiles()), _route_table_keys(adapter))
    assert problems == ["registered but not in the file: GET /v1/an-added-route/{item}"]


def test_comparison_reports_a_route_removed_from_the_table(adapter, monkeypatch):
    original = APIServerAdapter._http_route_table
    removed_method, removed_path, _handler = original(adapter)[0]
    monkeypatch.setattr(
        APIServerAdapter, "_http_route_table", lambda self: original(self)[1:]
    )
    problems = _compare_routes(set(_load_route_profiles()), _route_table_keys(adapter))
    assert problems == [f"in the file but not registered: {removed_method} {removed_path}"]


def test_every_key_is_a_normalized_route_rule():
    not_fixed = [
        key for key in _load_route_profiles() if _normalize_api_route_rule(key) != key
    ]
    assert not not_fixed, f"keys not written as _normalize_api_route_rule writes them: {not_fixed}"


def test_every_value_is_a_sorted_unique_list_of_role_names():
    bad = {}
    for route, roles in _load_route_profiles().items():
        valid = (
            isinstance(roles, list)
            and all(
                isinstance(role, str) and (role == "default" or _PROFILE_ID_RE.match(role))
                for role in roles
            )
            and roles == sorted(set(roles))
        )
        if not valid:
            bad[route] = roles
    assert not bad, f"roles must be a sorted list without duplicates of 'default' or profile names: {bad}"


def test_file_is_plain_json_with_one_object():
    assert isinstance(_load_route_profiles(), dict)
