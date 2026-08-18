"""Item 32G-A: the revocable, expiring Raphael Workspace read credential.

Covers the on-disk token registry (issue/list/revoke/expiry/restart
persistence, exclusive/no-follow/mode/path failures for the plaintext
output, secret non-disclosure) and the provider (wrong/revoked/expired/
other-provider tokens, protocol compliance, non-interactive surface).
"""
from __future__ import annotations

import json
import stat
import sys
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from hermes_cli.dashboard_auth import TokenPrincipal, assert_protocol_compliance
from hermes_cli.dashboard_auth.base import ProviderError

from plugins.dashboard_auth.raphael_workspace import token_store
from plugins.dashboard_auth import raphael_workspace
from plugins.dashboard_auth.raphael_workspace import WorkspaceReadTokenProvider


pytestmark_posix = pytest.mark.skipif(
    sys.platform.startswith("win"), reason="POSIX mode bits not enforced on Windows"
)


@pytest.fixture
def workspace_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    return home


def _issue(workspace_home: Path, **kwargs):
    out_dir = workspace_home / "plaintext-output"
    out_dir.mkdir(mode=0o700, exist_ok=True)
    out_path = out_dir / f"{uuid.uuid4().hex}.token"
    record = token_store.issue(out_path=out_path, **kwargs)
    return record, out_path.read_text(encoding="utf-8").strip()


# ---------------------------------------------------------------------------
# issue / list / revoke / expiry / restart persistence
# ---------------------------------------------------------------------------


class TestLifecycle:
    def test_issue_returns_record_and_plaintext_with_secret_not_in_record(self, workspace_home):
        record, plaintext = _issue(workspace_home)
        assert record.token_id in plaintext
        assert record.principal == "raphael-workspace"
        assert record.scope == "kanban.read"
        assert record.project == "raphael-workspace"
        assert record.board == "raphael-workspace"
        assert record.grant == "raphael-workspace-kanban-read-v1"
        assert record.revoked_at is None
        # The record (what gets persisted) never carries the raw secret.
        secret = plaintext.split(".", 1)[1]
        assert secret not in record.to_dict().values()
        assert secret != record.digest

    def test_issued_token_verifies(self, workspace_home):
        record, plaintext = _issue(workspace_home)
        token_id, secret = plaintext.split(".", 1)
        assert token_store.verify(token_id, secret) is not None
        assert token_store.verify(token_id, secret + "x") is None
        assert token_store.verify("nope", secret) is None

    def test_list_shows_every_issued_token(self, workspace_home):
        r1, _ = _issue(workspace_home)
        r2, _ = _issue(workspace_home)
        ids = {r.token_id for r in token_store.load_records()}
        assert ids == {r1.token_id, r2.token_id}

    def test_revoke_takes_effect_immediately(self, workspace_home):
        record, plaintext = _issue(workspace_home)
        token_id, secret = plaintext.split(".", 1)
        assert token_store.verify(token_id, secret) is not None
        token_store.revoke(token_id)
        assert token_store.verify(token_id, secret) is None

    def test_revoke_is_idempotent(self, workspace_home):
        record, _ = _issue(workspace_home)
        first = token_store.revoke(record.token_id)
        second = token_store.revoke(record.token_id)
        assert first.revoked_at == second.revoked_at

    def test_revoke_unknown_token_raises(self, workspace_home):
        with pytest.raises(token_store.UnknownTokenError):
            token_store.revoke("does-not-exist")

    def test_expired_token_does_not_verify(self, workspace_home, monkeypatch):
        record, plaintext = _issue(
            workspace_home, ttl_seconds=token_store.MIN_TTL_SECONDS
        )
        token_id, secret = plaintext.split(".", 1)
        monkeypatch.setattr(token_store.time, "time", lambda: record.expires_at)
        assert token_store.verify(token_id, secret) is None

    def test_ttl_bounds_enforced(self, workspace_home):
        with pytest.raises(ValueError):
            token_store.issue(
                out_path=workspace_home / "too-short.token",
                ttl_seconds=token_store.MIN_TTL_SECONDS - 1,
            )
        with pytest.raises(ValueError):
            token_store.issue(
                out_path=workspace_home / "too-long.token",
                ttl_seconds=token_store.MAX_TTL_SECONDS + 1,
            )

    def test_restart_persistence(self, workspace_home):
        """A fresh call with no in-memory state still sees a prior issue/revoke."""
        record, plaintext = _issue(workspace_home)
        token_id, secret = plaintext.split(".", 1)
        # Nothing is cached in module state — re-reading from disk is the
        # only path load_records/verify ever take, so this simulates a
        # process restart without needing to spawn one.
        assert token_store.verify(token_id, secret) is not None
        token_store.revoke(token_id)
        assert token_store.verify(token_id, secret) is None

    def test_store_created_with_owner_only_permissions(self, workspace_home):
        _issue(workspace_home)
        path = token_store.store_path()
        if not sys.platform.startswith("win"):
            assert stat.S_IMODE(path.stat().st_mode) == 0o600
            assert stat.S_IMODE(path.parent.stat().st_mode) == 0o700

    def test_parallel_issuance_preserves_every_record(self, workspace_home):
        with ThreadPoolExecutor(max_workers=8) as pool:
            issued = list(pool.map(lambda _: _issue(workspace_home)[0], range(12)))
        assert {r.token_id for r in token_store.load_records()} == {
            r.token_id for r in issued
        }

    def test_failed_strict_audit_leaves_no_token_or_plaintext(
        self, workspace_home, monkeypatch
    ):
        from hermes_cli.dashboard_auth.audit import AuditWriteError

        out_dir = workspace_home / "manual-output"
        out_dir.mkdir(mode=0o700)
        out_path = out_dir / "token"
        monkeypatch.setattr(
            token_store,
            "audit_log",
            lambda *args, **kwargs: (_ for _ in ()).throw(AuditWriteError("full")),
        )
        with pytest.raises(AuditWriteError):
            token_store.issue(out_path=out_path)
        assert not out_path.exists()
        assert token_store.load_records() == []

    def test_replacement_requires_an_active_token(self, workspace_home):
        current, _ = _issue(workspace_home)
        replacement, _ = _issue(
            workspace_home, replaces_token_id=current.token_id
        )
        assert replacement.status == "active"
        assert token_store.find(current.token_id).status == "active"
        token_store.revoke(current.token_id)
        with pytest.raises(token_store.UnknownTokenError):
            _issue(workspace_home, replaces_token_id=current.token_id)


# ---------------------------------------------------------------------------
# Fail-closed on malformed/unsafe on-disk state
# ---------------------------------------------------------------------------


class TestMalformedState:
    def test_malformed_json_raises(self, workspace_home):
        path = token_store.store_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("not json")
        path.chmod(0o600)
        with pytest.raises(token_store.TokenStoreError):
            token_store.load_records()

    def test_wrong_shape_raises(self, workspace_home):
        path = token_store.store_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"version": 1, "tokens": "not-a-list"}))
        path.chmod(0o600)
        with pytest.raises(token_store.TokenStoreError):
            token_store.load_records()

    def test_missing_field_in_record_raises(self, workspace_home):
        path = token_store.store_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"version": 1, "tokens": [{"token_id": "x"}]}))
        path.chmod(0o600)
        with pytest.raises(token_store.TokenStoreError):
            token_store.load_records()

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("issued_at", True),
            ("expires_at", 1.5),
            ("principal", 123),
            ("expires_at", token_store.MAX_TTL_SECONDS * 10),
        ],
    )
    def test_coerced_or_unbounded_metadata_is_rejected(
        self, workspace_home, field, value
    ):
        _issue(workspace_home)
        path = token_store.store_path()
        raw = json.loads(path.read_text(encoding="utf-8"))
        raw["tokens"][0][field] = value
        path.write_text(json.dumps(raw), encoding="utf-8")
        path.chmod(0o600)
        with pytest.raises(token_store.TokenStoreError):
            token_store.load_records()

    def test_unknown_metadata_field_is_rejected(self, workspace_home):
        _issue(workspace_home)
        path = token_store.store_path()
        raw = json.loads(path.read_text(encoding="utf-8"))
        raw["tokens"][0]["unexpected"] = "value"
        path.write_text(json.dumps(raw), encoding="utf-8")
        path.chmod(0o600)
        with pytest.raises(token_store.TokenStoreError):
            token_store.load_records()

    @pytestmark_posix
    def test_group_readable_store_raises(self, workspace_home):
        record, plaintext = _issue(workspace_home)
        path = token_store.store_path()
        path.chmod(0o640)
        with pytest.raises(token_store.TokenStoreError):
            token_store.load_records()

    def test_missing_store_is_empty_not_an_error(self, workspace_home):
        assert token_store.load_records() == []


# ---------------------------------------------------------------------------
# Plaintext output: exclusive / no-follow / mode / path failures
# ---------------------------------------------------------------------------


@pytestmark_posix
class TestPlaintextOutput:
    def test_writes_once_at_0600(self, tmp_path):
        out_dir = tmp_path / "secrets"
        out_dir.mkdir(mode=0o700)
        out_path = out_dir / "token.txt"
        token_store.write_plaintext_once(out_path, "abc.def\n")
        assert out_path.read_text() == "abc.def\n"
        assert stat.S_IMODE(out_path.stat().st_mode) == 0o600

    def test_refuses_to_overwrite_existing_file(self, tmp_path):
        out_dir = tmp_path / "secrets"
        out_dir.mkdir(mode=0o700)
        out_path = out_dir / "token.txt"
        out_path.write_text("pre-existing")
        with pytest.raises(token_store.PlaintextOutputError):
            token_store.write_plaintext_once(out_path, "abc.def\n")
        assert out_path.read_text() == "pre-existing"

    def test_refuses_to_follow_a_symlink(self, tmp_path):
        out_dir = tmp_path / "secrets"
        out_dir.mkdir(mode=0o700)
        real_target = tmp_path / "outside.txt"
        link_path = out_dir / "token.txt"
        link_path.symlink_to(real_target)
        with pytest.raises(token_store.PlaintextOutputError):
            token_store.write_plaintext_once(link_path, "abc.def\n")
        assert not real_target.exists()

    def test_refuses_a_missing_parent_directory(self, tmp_path):
        out_path = tmp_path / "does-not-exist" / "token.txt"
        with pytest.raises(token_store.PlaintextOutputError):
            token_store.write_plaintext_once(out_path, "abc.def\n")

    def test_refuses_a_group_writable_parent_directory(self, tmp_path):
        out_dir = tmp_path / "secrets"
        out_dir.mkdir(mode=0o750)
        out_path = out_dir / "token.txt"
        with pytest.raises(token_store.PlaintextOutputError):
            token_store.write_plaintext_once(out_path, "abc.def\n")


# ---------------------------------------------------------------------------
# Provider: wrong / revoked / expired / other-provider tokens
# ---------------------------------------------------------------------------


class TestProvider:
    def test_plugin_registers_provider_and_cli(self):
        ctx = MagicMock()
        raphael_workspace.register(ctx)
        ctx.register_dashboard_auth_provider.assert_called_once()
        assert isinstance(
            ctx.register_dashboard_auth_provider.call_args.args[0],
            WorkspaceReadTokenProvider,
        )
        assert ctx.register_cli_command.call_args.kwargs["name"] == (
            "kanban-workspace-token"
        )

    def test_protocol_compliance(self):
        assert_protocol_compliance(WorkspaceReadTokenProvider)

    def test_supports_token_only(self):
        p = WorkspaceReadTokenProvider()
        assert p.supports_token is True
        assert p.supports_session is False

    def test_verify_token_accepts_issued_token(self, workspace_home):
        record, plaintext = _issue(workspace_home)
        p = WorkspaceReadTokenProvider()
        principal = p.verify_token(token=plaintext)
        assert isinstance(principal, TokenPrincipal)
        assert principal.principal == "raphael-workspace"
        assert principal.scopes == ("kanban.read",)
        assert principal.credential_id == record.token_id

    @pytest.mark.parametrize("token", ["", "no-dot-at-all", ".", "a.", ".b", "unknown.secret"])
    def test_verify_token_rejects_malformed_or_unknown(self, workspace_home, token):
        p = WorkspaceReadTokenProvider()
        assert p.verify_token(token=token) is None

    def test_verify_token_rejects_wrong_secret(self, workspace_home):
        record, plaintext = _issue(workspace_home)
        p = WorkspaceReadTokenProvider()
        assert p.verify_token(token=f"{record.token_id}.wrong-secret") is None

    def test_verify_token_rejects_revoked(self, workspace_home):
        record, plaintext = _issue(workspace_home)
        token_store.revoke(record.token_id)
        p = WorkspaceReadTokenProvider()
        assert p.verify_token(token=plaintext) is None

    def test_verify_token_rejects_expired(self, workspace_home, monkeypatch):
        record, plaintext = _issue(workspace_home)
        monkeypatch.setattr(token_store.time, "time", lambda: record.expires_at)
        p = WorkspaceReadTokenProvider()
        assert p.verify_token(token=plaintext) is None

    def test_verify_token_rejects_another_providers_token_shape(self, workspace_home):
        # A drain/recommendations-style flat secret (no "." separator) never
        # matches this provider's <token_id>.<secret> shape.
        p = WorkspaceReadTokenProvider()
        assert p.verify_token(token="some-flat-static-secret-value") is None

    def test_malformed_store_raises_provider_error_not_401(self, workspace_home):
        path = token_store.store_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("not json")
        path.chmod(0o600)
        p = WorkspaceReadTokenProvider()
        with pytest.raises(ProviderError):
            p.verify_token(token=f"{token_store.TOKEN_PREFIX}invalid.secret")

    def test_interactive_methods_raise_and_session_methods_are_inert(self):
        p = WorkspaceReadTokenProvider()
        with pytest.raises(NotImplementedError):
            p.start_login(redirect_uri="r")
        with pytest.raises(NotImplementedError):
            p.complete_login(code="c", state="s", code_verifier="v", redirect_uri="r")
        with pytest.raises(NotImplementedError):
            p.refresh_session(refresh_token="r")
        assert p.verify_session(access_token="anything") is None
        assert p.revoke_session(refresh_token="anything") is None
