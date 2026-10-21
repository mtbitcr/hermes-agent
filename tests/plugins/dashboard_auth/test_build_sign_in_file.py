"""The one protected file build machines read their sign-in from first.

Covers what reads as no file at all (a missing file, a symlink, a mode other
than 0600, another owner, broken content), the owner-only write that replaces
an earlier file in one step, and a worker running under its profile
``HERMES_HOME`` that must still reach the one file in the root Hermes home.
Every token here is synthetic, and none may reach a log record or an exception.
"""
from __future__ import annotations

import json
import logging
import os
import stat
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from plugins.dashboard_auth.raphael_workspace import build_sign_in_file as bsf
from plugins.dashboard_auth.raphael_workspace import sandbox_dispatch as sd

BUILD_TOKEN = "sk-ant-oat01-SYNTHETIC-build-sign-in-file-token"
NEWER_TOKEN = "sk-ant-oat01-SYNTHETIC-newer-build-sign-in-file-token"
PROFILE_TOKEN = "sk-ant-oat01-SYNTHETIC-file-under-the-profile-home"
ENV_TOKEN = "sk-ant-oat01-SYNTHETIC-setup-token-from-the-environment"
SYNTHETIC_TOKENS = (BUILD_TOKEN, NEWER_TOKEN, PROFILE_TOKEN, ENV_TOKEN)

#: Where the file lives under a root Hermes home, computed here on its own.
FILE_PARTS = ("secrets", "build-sign-in.json")


@pytest.fixture(autouse=True)
def _no_token_in_any_log_record(caplog):
    caplog.set_level(logging.DEBUG)
    yield
    _assert_not_exposed(caplog)


@pytest.fixture
def root(tmp_path, monkeypatch):
    """The root Hermes home the Server 1 gateway runs with."""
    home = tmp_path / "hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    return home


def _assert_not_exposed(caplog, *texts: str) -> None:
    blobs = [caplog.text, *(record.getMessage() for record in caplog.records), *texts]
    for token in SYNTHETIC_TOKENS:
        for blob in blobs:
            assert token not in blob


def _made(days_ago: float, *, token: str = BUILD_TOKEN) -> dict:
    made_at = datetime.now(timezone.utc) - timedelta(days=days_ago)
    return {"token": token, "made_at": made_at.isoformat()}


def _place(path: Path, body, *, mode: int = 0o600) -> Path:
    """Put a sign-in file in place the way an operator would."""
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = body if isinstance(body, bytes) else json.dumps(body).encode("utf-8")
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "wb") as handle:
        handle.write(raw)
    os.chmod(path, mode)
    return path


def _token_at(path) -> str:
    return json.loads(Path(path).read_text(encoding="utf-8"))["token"]


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------


def test_a_valid_file_gives_the_token_and_its_end_a_year_after_it_was_made(root):
    record = _made(10)
    _place(root.joinpath(*FILE_PARTS), record)

    token, ends_at = bsf.read_build_token()

    assert token == BUILD_TOKEN
    assert ends_at == datetime.fromisoformat(record["made_at"]) + timedelta(days=365)


def test_a_missing_file_reads_as_no_file(root):
    assert bsf.read_build_token() is None


@pytest.mark.parametrize("mode", [0o644, 0o640, 0o400])
def test_a_mode_other_than_0600_reads_as_no_file(root, mode):
    _place(root.joinpath(*FILE_PARTS), _made(10), mode=mode)
    assert bsf.read_build_token() is None


def test_a_symlink_reads_as_no_file(root, tmp_path):
    target = _place(tmp_path / "elsewhere" / FILE_PARTS[-1], _made(10))
    link = root.joinpath(*FILE_PARTS)
    link.parent.mkdir(parents=True)
    link.symlink_to(target)

    assert bsf.read_build_token() is None


def test_a_file_owned_by_another_user_reads_as_no_file(root, monkeypatch):
    path = _place(root.joinpath(*FILE_PARTS), _made(10))
    # Tests run as root, so the other owner is faked through the uid lookup.
    monkeypatch.setattr(bsf, "_current_uid", lambda: path.stat().st_uid + 1)

    assert bsf.read_build_token() is None


_MADE_AT = _made(10)["made_at"]


@pytest.mark.parametrize(
    "body",
    [
        pytest.param(b"", id="empty"),
        pytest.param(b"not json at all", id="not-json"),
        pytest.param(json.dumps(_made(10)).encode("utf-8")[:-12], id="cut-short"),
        pytest.param(b"\xff\xfe\x00not utf-8", id="not-utf-8"),
        pytest.param(b"[" * 60000, id="nested-too-deep"),
        pytest.param([BUILD_TOKEN, _MADE_AT], id="not-an-object"),
        pytest.param({"made_at": _MADE_AT}, id="no-token"),
        pytest.param({"token": "  ", "made_at": _MADE_AT}, id="blank-token"),
        pytest.param({"token": 12345, "made_at": _MADE_AT}, id="token-not-text"),
        pytest.param({"token": BUILD_TOKEN}, id="no-made-time"),
        pytest.param({"token": BUILD_TOKEN, "made_at": "last tuesday"}, id="made-time-unreadable"),
        pytest.param({"token": BUILD_TOKEN, "made_at": 1760000000}, id="made-time-not-text"),
        pytest.param(
            {"token": BUILD_TOKEN, "made_at": "2026-10-11T10:00:00"}, id="made-time-without-utc-offset"
        ),
        pytest.param(
            {"token": BUILD_TOKEN, "made_at": "9999-12-31T00:00:00+00:00"}, id="made-time-out-of-range"
        ),
        pytest.param(
            {"token": BUILD_TOKEN, "made_at": _MADE_AT, "padding": "x" * (1024 * 1024)},
            id="far-too-large",
        ),
    ],
)
def test_broken_content_reads_as_no_file(root, body):
    _place(root.joinpath(*FILE_PARTS), body)
    assert bsf.read_build_token() is None


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------


def test_a_write_is_0600_and_a_second_write_replaces_the_first_in_one_step(root, monkeypatch):
    bsf.write_build_token(BUILD_TOKEN)
    path = root.joinpath(*FILE_PARTS)
    assert stat.S_IMODE(path.lstat().st_mode) == 0o600
    assert bsf.read_build_token()[0] == BUILD_TOKEN

    real_replace = os.replace
    swaps = []

    def _watch_replace(src, dst):
        # The instant before the swap: the old file is still whole, and the new
        # one is already complete and owner-only.
        swaps.append((_token_at(dst), _token_at(src), stat.S_IMODE(os.lstat(src).st_mode)))
        real_replace(src, dst)

    monkeypatch.setattr(os, "replace", _watch_replace)
    bsf.write_build_token(NEWER_TOKEN)

    assert swaps == [(BUILD_TOKEN, NEWER_TOKEN, 0o600)]
    assert stat.S_IMODE(path.lstat().st_mode) == 0o600
    token, ends_at = bsf.read_build_token()
    assert token == NEWER_TOKEN
    assert abs(ends_at - (datetime.now(timezone.utc) + timedelta(days=365))) < timedelta(minutes=1)
    assert [entry.name for entry in path.parent.iterdir()] == [path.name]


def test_a_blank_token_is_refused_and_the_earlier_file_stays(root, caplog):
    bsf.write_build_token(BUILD_TOKEN)

    with pytest.raises(ValueError) as excinfo:
        bsf.write_build_token("   ")

    assert bsf.read_build_token()[0] == BUILD_TOKEN
    _assert_not_exposed(caplog, str(excinfo.value))


def test_a_failed_write_names_no_token(root, caplog):
    # The secrets folder cannot be made: a plain file already holds its name.
    root.joinpath(FILE_PARTS[0]).write_text("not a folder\n", encoding="utf-8")

    with pytest.raises(OSError) as excinfo:
        bsf.write_build_token(BUILD_TOKEN)

    _assert_not_exposed(caplog, str(excinfo.value), repr(excinfo.value))


# ---------------------------------------------------------------------------
# A worker reaches the one root file from its profile home
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("layout", ["home-dot-hermes", "custom-root"])
def test_a_worker_under_its_profile_home_reaches_the_one_root_file(
    tmp_path, monkeypatch, layout
):
    import agent.anthropic_credentials as credentials

    if layout == "home-dot-hermes":
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        root = tmp_path / ".hermes"
    else:
        root = tmp_path / "srv" / "hermes"
    profile = root / "profiles" / "raphael-claude-worker"
    profile.mkdir(parents=True)
    # Exactly what the kanban dispatcher injects into a worker it spawns.
    monkeypatch.setenv("HERMES_HOME", str(profile))
    monkeypatch.setenv("HERMES_PROFILE", "raphael-claude-worker")
    monkeypatch.setenv("HERMES_KANBAN_DB", str(root / "kanban" / "boards" / "raphael" / "kanban.db"))
    monkeypatch.setenv("HERMES_KANBAN_BOARD", "raphael")
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_build_sign_in")
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", ENV_TOKEN)
    monkeypatch.setattr(credentials, "resolve_anthropic_token", lambda: "")

    # A file under the profile home is never the one builds read.
    _place(profile.joinpath(*FILE_PARTS), _made(10, token=PROFILE_TOKEN))
    assert bsf.read_build_token() is None
    assert sd._resolve_host_credential() == ENV_TOKEN

    _place(root.joinpath(*FILE_PARTS), _made(10))
    assert bsf.read_build_token()[0] == BUILD_TOKEN
    assert sd._resolve_host_credential() == BUILD_TOKEN

    # A write from the worker lands in the same one root file.
    bsf.write_build_token(NEWER_TOKEN)
    assert _token_at(root.joinpath(*FILE_PARTS)) == NEWER_TOKEN
    assert _token_at(profile.joinpath(*FILE_PARTS)) == PROFILE_TOKEN
