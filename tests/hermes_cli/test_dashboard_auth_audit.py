"""Audit log for dashboard-auth events.

Profile-aware location: ``$HERMES_HOME/logs/dashboard-auth.log``.
Format: one JSON object per line. Token-like kwargs are dropped before
serialisation so we never leak refresh tokens or JWTs to disk.
"""
from __future__ import annotations

import json
import stat
import pytest

from hermes_cli.dashboard_auth import audit as audit_module
from hermes_cli.dashboard_auth.audit import AuditEvent, AuditWriteError, audit_log


@pytest.fixture
def profile_home(tmp_path, monkeypatch):
    """Redirect $HERMES_HOME and ~ to a tmp dir for the duration of the test."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    # Some code paths fall back to Path.home() — patch that too.
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    return home


def test_audit_writes_jsonlines(profile_home):
    audit_log(AuditEvent.LOGIN_START, provider="nous", ip="1.2.3.4")
    audit_log(
        AuditEvent.LOGIN_SUCCESS,
        provider="nous", user_id="u1",
        email="a@b.com", ip="1.2.3.4",
    )

    path = profile_home / "logs" / "dashboard-auth.log"
    assert path.exists(), f"audit log not created at {path}"
    lines = path.read_text().strip().splitlines()
    assert len(lines) == 2

    second = json.loads(lines[1])
    assert second["event"] == "login_success"
    assert second["provider"] == "nous"
    assert second["user_id"] == "u1"
    assert second["email"] == "a@b.com"
    assert "ts" in second  # ISO-8601 timestamp


def test_audit_redacts_token_like_fields(profile_home):
    audit_log(
        AuditEvent.LOGIN_SUCCESS,
        provider="nous", access_token="should-not-appear",
        refresh_token="also-not", code="not-this", state="nope",
    )
    raw = (profile_home / "logs" / "dashboard-auth.log").read_text()
    for forbidden in ("should-not-appear", "also-not", "not-this", "nope"):
        assert forbidden not in raw, f"token-like value leaked into audit log: {forbidden}"


# --------------------------------------------------------------------------
# Item 32G-A: strict=True fail-closed audit writes
# --------------------------------------------------------------------------


def test_strict_false_default_never_raises_on_write_failure(profile_home, monkeypatch):
    def _boom(*a, **k):
        raise OSError("disk full")

    monkeypatch.setattr("builtins.open", _boom)
    # Every existing call site relies on this: a broken logger must not
    # break auth.
    audit_log(AuditEvent.LOGIN_START, provider="nous")


def test_strict_true_raises_audit_write_error_on_failure(profile_home, monkeypatch):
    def _boom(*a, **k):
        raise OSError("disk full")

    monkeypatch.setattr(audit_module.os, "open", _boom)
    with pytest.raises(AuditWriteError):
        audit_log(AuditEvent.TOKEN_AUTH_SUCCESS, strict=True, provider="raphael-workspace-token")


def test_strict_true_succeeds_silently_when_write_succeeds(profile_home):
    audit_log(AuditEvent.TOKEN_AUTH_SUCCESS, strict=True, provider="raphael-workspace-token")
    path = profile_home / "logs" / "dashboard-auth.log"
    assert path.exists()
    assert json.loads(path.read_text().strip().splitlines()[-1])["event"] == "token_auth_success"


@pytest.mark.skipif(audit_module.os.name == "nt", reason="POSIX modes only")
def test_audit_tightens_owned_log_and_parent_permissions(profile_home):
    logs = profile_home / "logs"
    logs.mkdir(mode=0o755)
    path = logs / "dashboard-auth.log"
    path.write_text("")
    path.chmod(0o644)

    audit_log(AuditEvent.TOKEN_AUTH_SUCCESS, strict=True, provider="workspace")

    assert stat.S_IMODE(logs.stat().st_mode) == 0o700
    assert stat.S_IMODE(path.stat().st_mode) == 0o600


