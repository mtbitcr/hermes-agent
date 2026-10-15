"""Strict managed-scope reads: a managed config.yaml that is present but cannot be applied.

By default the loaders stay fail-open, so a broken administrator file can never brick startup.
A caller that passes ``strict=True`` gets ``ManagedScopeError`` instead of a config that
silently lacks, or only half carries, the administrator's values.
"""

import copy

import pytest
import yaml

from gateway import config_loader
from hermes_cli import managed_scope

USER_CONFIG = {"gateway": {"api_server": {"port": 8650}}}

# Parses as a mapping, but its anchor refers to itself, so applying its values never ends.
APPLY_FAILING = "gateway: &loop {api_server: {port: 9000}, loop: *loop}\n"
MALFORMED = "gateway: {api_server: {port: 9000}\n"
NOT_A_MAPPING = "- gateway\n- api_server\n"
# Documents whose root is not a mapping. The falsey ones must not pass for an empty layer.
NON_MAPPING_ROOTS = [
    pytest.param(NOT_A_MAPPING, id="not-a-mapping"),
    pytest.param("[]\n", id="empty-list"),
    pytest.param("false\n", id="false"),
    pytest.param("0\n", id="zero"),
    pytest.param("0.0\n", id="zero-float"),
    pytest.param("''\n", id="empty-string"),
    pytest.param("null\n", id="null"),
    pytest.param("~\n", id="tilde"),
    pytest.param("!!set {}\n", id="empty-set"),
    pytest.param('!!binary ""\n', id="empty-binary"),
]


@pytest.fixture()
def managed_dir(tmp_path, monkeypatch):
    """The managed scope's directory under tmp_path, absent until a test writes into it."""
    path = tmp_path / "managed"
    monkeypatch.setenv("HERMES_MANAGED_DIR", str(path))
    managed_scope.invalidate_managed_cache()
    yield path
    managed_scope.invalidate_managed_cache()


@pytest.fixture(params=["apply_managed_overlay", "read_yaml_layers"])
def load(request, tmp_path):
    """The user's config with the managed overlay, through each entry point that takes ``strict``."""
    if request.param == "apply_managed_overlay":
        return lambda **kwargs: managed_scope.apply_managed_overlay(copy.deepcopy(USER_CONFIG), **kwargs)
    home = tmp_path / "home"
    home.mkdir()
    (home / "config.yaml").write_text(yaml.safe_dump(USER_CONFIG), encoding="utf-8")
    return lambda **kwargs: config_loader.read_yaml_layers(home, **kwargs)


def _write_managed(managed_dir, text):
    managed_dir.mkdir()
    (managed_dir / "config.yaml").write_text(text, encoding="utf-8")


def test_managed_file_that_parses_but_fails_while_applied_raises_in_strict_mode(managed_dir, load):
    _write_managed(managed_dir, APPLY_FAILING)

    with pytest.raises(managed_scope.ManagedScopeError):
        load(strict=True)


@pytest.mark.parametrize("text", [pytest.param(MALFORMED, id="malformed"), *NON_MAPPING_ROOTS])
def test_malformed_or_non_mapping_managed_file_raises_in_strict_mode(managed_dir, load, text):
    _write_managed(managed_dir, text)

    with pytest.raises(managed_scope.ManagedScopeError):
        load(strict=True)


@pytest.mark.parametrize("text", NON_MAPPING_ROOTS)
def test_non_mapping_managed_file_raises_in_strict_mode_after_a_default_read(managed_dir, load, text):
    _write_managed(managed_dir, text)
    assert load() == USER_CONFIG  # so the strict read meets the file already parsed and cached

    with pytest.raises(managed_scope.ManagedScopeError):
        load(strict=True)
    assert load() == USER_CONFIG


@pytest.mark.parametrize(
    "text",
    [
        pytest.param(APPLY_FAILING, id="apply-failing"),
        pytest.param(MALFORMED, id="malformed"),
        *NON_MAPPING_ROOTS,
    ],
)
def test_without_strict_each_such_managed_file_leaves_the_config_unchanged(managed_dir, load, text):
    _write_managed(managed_dir, text)

    assert load() == USER_CONFIG


@pytest.mark.parametrize(
    "text",
    [
        pytest.param("", id="zero-byte"),
        pytest.param("  \n\n", id="whitespace-only"),
        pytest.param("# placeholder: no managed values yet\n", id="comment-only"),
        pytest.param("{}\n", id="empty-mapping"),
    ],
)
def test_managed_file_holding_no_values_raises_nothing_in_strict_mode(managed_dir, load, text):
    _write_managed(managed_dir, text)

    assert load(strict=True) == USER_CONFIG
    assert load() == USER_CONFIG


def test_absent_managed_directory_raises_nothing_in_strict_mode(managed_dir, load):
    assert load(strict=True) == USER_CONFIG


def test_absent_managed_config_in_a_present_directory_raises_nothing_in_strict_mode(managed_dir, load):
    managed_dir.mkdir()

    assert load(strict=True) == USER_CONFIG
