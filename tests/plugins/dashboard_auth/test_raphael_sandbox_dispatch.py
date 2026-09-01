"""Trusted Server 2 sandbox provisioning for the Raphael Claude coding worker.

Covers the fail-closed gates (profile, dispatcher-owned run, git-backed and
scoped source, digest-pinned image, host connection/credential), the exact
policy payload handed to the official OpenSandbox SDK (immutable image digest,
bounded resources, default-deny egress, CredentialProxy, loopback-tunnel
transport, placeholder token in the sandbox + real credential only in
Credential Vault, bound to HTTPS + host + ``/v1/*`` + GET/POST only), the
credential preflight that keeps a short-lived Claude account token from dying
mid-task — including a concurrent rotation this process lost — the fixed
second extraction whose write bits are removed so the worker can generate an
exact recursive patch (a diff aid, never claimed read-only or immutable, since
work on the pinned image runs as uid 0), the sanitized receipt, and — through
the native per-run reservation in ``task_events``, which is the only authority
— idempotent reuse, SDK-proved liveness before reuse, single-replacement on a
dead machine, and the compare-and-swap that stops a concurrent caller from
creating a second machine.
"""
from __future__ import annotations

import dataclasses
import json
import hashlib
import logging
import os
import subprocess
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path, PurePosixPath
from types import SimpleNamespace

import pytest

from hermes_cli import kanban_db as kb
from plugins.dashboard_auth import raphael_workspace
from plugins.dashboard_auth.raphael_workspace import sandbox_dispatch as sd


REAL_SECRET = "sk-ant-api03-REAL-host-owned-anthropic-credential-value"
REAL_OAUTH = "sk-ant-oat01-REAL-host-owned-claude-code-oauth-token"
FRESH_OAUTH = "sk-ant-oat01-FRESH-rotated-claude-code-oauth-token"
#: What another Server 1 process persisted by winning the same rotation.
WINNER_OAUTH = "sk-ant-oat01-WINNER-concurrently-rotated-claude-code-token"
REFRESH_TOKEN = "sk-ant-ort01-REFRESH-token-that-must-never-leave-server-1"
CONNECTION_SECRET = "os-api-key-value"
IMAGE_DIGEST = "sha256:" + "ab" * 32
IMAGE_REF = f"registry.internal.example/raphael/claude-worker@{IMAGE_DIGEST}"

DAY_MS = 24 * 60 * 60 * 1000


def _expires_in(seconds: float) -> int:
    return int((time.time() + seconds) * 1000)


# ---------------------------------------------------------------------------
# Fake Server 2 (only the network-touching SandboxSync class is faked; every
# request/response model below is the real SDK pydantic model, so an invalid
# payload — or a field this module reads that the SDK does not actually
# publish — fails here exactly as it would against a live control plane).
# ---------------------------------------------------------------------------


class FakeVault:
    def __init__(self, box: "FakeSandbox") -> None:
        self._box = box

    def create(self, *, credentials, bindings):
        self._box.vault_calls.append({"credentials": credentials, "bindings": bindings})
        if self._box.behavior.get("vault_fails"):
            raise RuntimeError(f"vault rejected token {REAL_SECRET}")
        return SimpleNamespace(revision=1, credentials=[], bindings=[])


class FakeFiles:
    """Models the two measured, asymmetric facts the real SDK exposes.

    ``write_file``/``read_bytes(_stream)`` FOLLOW every symlink in the path,
    including intermediate directory components — exactly like any ordinary
    filesystem call. ``get_file_info`` lstats: for a requested path it
    resolves every component *except the last* (a symlinked ancestor still
    redirects it there, matching the real kernel), then reports whatever sits
    at that exact spot without dereferencing it — a symlink is reported as
    ``"symlink"``, never silently followed.
    """

    def __init__(self, box: "FakeSandbox") -> None:
        self._box = box

    def _resolve(self, path, *, follow_leaf):
        parts = PurePosixPath(path).parts
        resolved = PurePosixPath(parts[0])
        last_index = len(parts) - 1
        for index, part in enumerate(parts[1:], start=1):
            resolved = resolved / part
            if index == last_index and not follow_leaf:
                continue
            seen = set()
            while str(resolved) in self._box.symlinks:
                if str(resolved) in seen:
                    raise RuntimeError("symlink loop")
                seen.add(str(resolved))
                resolved = PurePosixPath(self._box.symlinks[str(resolved)])
        return str(resolved)

    def write_file(self, path, data, **kwargs):
        if self._box.behavior.get("upload_fails"):
            raise RuntimeError("upload failed")
        resolved = self._resolve(path, follow_leaf=True)
        self._box.uploads.append((path, data))
        self._box.file_data[resolved] = data if isinstance(data, bytes) else data.encode()

    def read_bytes(self, path, **kwargs):
        return self._box.file_data[self._resolve(path, follow_leaf=True)]

    def read_bytes_stream(self, path, **kwargs):
        yield self.read_bytes(path)

    def create_directories(self, entries):
        for entry in entries:
            node = PurePosixPath(self._resolve(entry.path, follow_leaf=True))
            while True:
                self._box.directories.add(str(node))
                if node == node.parent:
                    break
                node = node.parent

    def get_file_info(self, paths):
        from opensandbox.models.filesystem import EntryInfo

        now = datetime.now(timezone.utc)
        info = {}
        for requested in paths:
            resolved = self._resolve(requested, follow_leaf=False)
            if resolved in self._box.symlinks:
                entry_type = "symlink"
            elif resolved in self._box.directories:
                entry_type = "directory"
            elif resolved in self._box.file_data:
                entry_type = "file"
            else:
                continue
            info[requested] = EntryInfo(
                path=requested, entry_type=entry_type, mode=0o644,
                owner="root", group="root",
                size=len(self._box.file_data.get(resolved, b"")),
                modified_at=now, created_at=now,
            )
        return info


class FakeCommands:
    def __init__(self, box: "FakeSandbox") -> None:
        self._box = box

    def run(self, command, **kwargs):
        self._box.commands_log.append(command)
        if self._box.behavior.get("extract_fails"):
            return SimpleNamespace(exit_code=2, logs=SimpleNamespace(stderr=[]))
        return SimpleNamespace(exit_code=0, logs=SimpleNamespace(stderr=[]))


class FakeSandbox:
    """Stands in for ``SandboxSync`` — the only network-touching SDK class."""

    created: list["FakeSandbox"] = []
    live: dict = {}
    connect_calls: list = []
    behavior: dict = {}
    counter = 0

    def __init__(self, kwargs: dict) -> None:
        type(self).counter += 1
        self.id = f"sbx-{type(self).counter:03d}"
        self.create_kwargs = kwargs
        self.vault_calls: list[dict] = []
        self.uploads: list[tuple] = []
        self.file_data = {}
        # Directories the seeding shell step and prior imports would already
        # have created on a real machine — pre-seeded here since FakeCommands
        # doesn't touch this registry. path -> symlink target for planted links.
        self.directories = {"/", "/workspace", "/tmp", sd.SANDBOX_WORKSPACE, sd.SANDBOX_BASELINE}
        self.symlinks: dict = {}
        self.commands_log: list[str] = []
        self.connect_kwargs: dict = {}
        self.killed = False
        self.closed = False
        self.behavior = type(self).behavior
        self.credential_vault = FakeVault(self)
        self.files = FakeFiles(self)
        self.commands = FakeCommands(self)
        # Server-side facts a later liveness check reads back.
        self.state = "RUNNING"
        self.healthy = True
        self.expires_in_seconds: float | None = sd.SANDBOX_TIMEOUT_SECONDS
        self.reported_image = kwargs.get("image").image
        self.reported_metadata = dict(kwargs.get("metadata") or {})
        self.reported_id = self.id
        self.info_fails = False

    @classmethod
    def create(cls, image=None, **kwargs):
        if cls.behavior.get("create_fails"):
            raise RuntimeError("control plane unavailable")
        box = cls({"image": image, **kwargs})
        cls.created.append(box)
        cls.live[box.id] = box
        hook = cls.behavior.get("after_create")
        if hook is not None:
            hook(box)
        return box

    @classmethod
    def connect(cls, sandbox_id, **kwargs):
        cls.connect_calls.append({"sandbox_id": sandbox_id, **kwargs})
        box = cls.live.get(sandbox_id)
        if box is None or box.killed:
            raise RuntimeError(f"sandbox {sandbox_id} does not exist")
        box.connect_kwargs = kwargs
        return box

    def get_info(self):
        from opensandbox.models.sandboxes import (
            SandboxImageSpec,
            SandboxInfo,
            SandboxStatus,
        )

        if self.info_fails:
            raise RuntimeError("info unavailable")
        expires_at = None
        if self.expires_in_seconds is not None:
            expires_at = datetime.now(timezone.utc) + timedelta(
                seconds=self.expires_in_seconds
            )
        return SandboxInfo(
            id=self.reported_id,
            status=SandboxStatus(state=self.state),
            entrypoint=[],
            created_at=datetime.now(timezone.utc),
            expires_at=expires_at,
            image=SandboxImageSpec(image=self.reported_image),
            metadata=dict(self.reported_metadata),
        )

    def is_healthy(self):
        return self.healthy

    def kill(self):
        self.killed = True
        type(self).live.pop(self.id, None)

    def close(self):
        self.closed = True


@pytest.fixture(autouse=True)
def _reset_fake():
    FakeSandbox.created = []
    FakeSandbox.live = {}
    FakeSandbox.connect_calls = []
    FakeSandbox.behavior = {}
    FakeSandbox.counter = 0
    yield
    FakeSandbox.created = []
    FakeSandbox.live = {}
    FakeSandbox.connect_calls = []
    FakeSandbox.behavior = {}


@pytest.fixture
def sdk(monkeypatch):
    """Real SDK request models + a fake sandbox class."""
    fake = dataclasses.replace(sd._load_sdk(), sandbox=FakeSandbox)
    monkeypatch.setattr(sd, "_load_sdk", lambda: fake)
    return fake


# ---------------------------------------------------------------------------
# Host fixtures
# ---------------------------------------------------------------------------


def _git(cwd: Path, *args: str) -> str:
    return subprocess.run(
        [
            "git", "-C", str(cwd),
            "-c", "user.name=Test User",
            "-c", "user.email=test@example.com",
            "-c", "commit.gpgsign=false",
            *args,
        ],
        check=True, capture_output=True, text=True,
    ).stdout.strip()


@pytest.fixture
def host(tmp_path, monkeypatch):
    """A Server 1 control plane: HERMES_HOME, board, active run, git worktree."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-b", "main", str(repo)],
                   check=True, capture_output=True, text=True)
    (repo / "README.md").write_text("hermes\n", encoding="utf-8")
    (repo / "app.py").write_text("print('hi')\n", encoding="utf-8")
    # Ignored so a test can drop an untracked file in the worktree without
    # making the tree dirty — the point being that ``git archive HEAD`` still
    # never packages it.
    (repo / ".gitignore").write_text("untracked-secret.txt\n", encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "init")

    kb.init_db()
    with kb.connect() as conn:
        task_id = kb.create_task(
            conn,
            title="ship the founder feature",
            assignee="raphael-claude-worker",
            workspace_kind="worktree",
            owned_paths=["."],
        )
        worktree = repo / ".worktrees" / task_id
        _git(repo, "worktree", "add", str(worktree), "-b", f"wt/{task_id}", "HEAD")
        with kb.write_txn(conn):
            conn.execute(
                "INSERT INTO task_runs (task_id, status, started_at) "
                "VALUES (?, 'running', 0)",
                (task_id,),
            )
            run_id = int(
                conn.execute("SELECT last_insert_rowid() AS id").fetchone()["id"]
            )
            conn.execute(
                "UPDATE tasks SET workspace_path = ?, status = 'running', "
                "current_run_id = ? WHERE id = ?",
                (str(worktree), run_id, task_id),
            )
        head = _git(worktree, "rev-parse", "HEAD")

    monkeypatch.setenv("HERMES_PROFILE", "raphael-claude-worker")
    monkeypatch.setenv("HERMES_KANBAN_TASK", task_id)
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(run_id))
    monkeypatch.setenv("HERMES_KANBAN_WORKSPACE", str(worktree))
    monkeypatch.setenv("HERMES_KANBAN_BOARD", "default")

    # Trusted host config: digest-pinned image + already-configured tunnel.
    monkeypatch.setattr(
        sd, "_load_host_config",
        lambda: {"image": IMAGE_REF, "domain": "sandbox.internal:8080",
                 "protocol": "https"},
    )
    monkeypatch.setattr(
        sd, "_load_connection_secret_from_env", lambda: CONNECTION_SECRET
    )
    monkeypatch.setattr(sd, "_resolve_host_credential", lambda: REAL_SECRET)
    monkeypatch.setattr(sd, "_claude_account_credentials", lambda: None)

    return SimpleNamespace(
        home=home, repo=repo, worktree=worktree, task_id=task_id,
        run_id=run_id, head=head,
    )


def _provision(args=None):
    return json.loads(sd.handle_provision(args or {}))


def _reservation(host):
    with kb.connect_closing() as conn:
        return kb.read_run_sandbox(conn, host.task_id, run_id=host.run_id)


def _event_kinds(host):
    with kb.connect_closing() as conn:
        rows = conn.execute(
            "SELECT kind FROM task_events WHERE task_id = ? AND run_id = ? "
            "AND kind LIKE 'sandbox_%' ORDER BY id ASC",
            (host.task_id, host.run_id),
        ).fetchall()
    return [str(row["kind"]) for row in rows]


# ---------------------------------------------------------------------------
# 1. Model-call surface
# ---------------------------------------------------------------------------


class TestSchema:
    def test_schema_exposes_no_free_form_parameters(self):
        params = sd.PROVISION_SCHEMA["parameters"]
        assert params["type"] == "object"
        assert params.get("properties") == {}
        assert params.get("required", []) == []
        assert params.get("additionalProperties") is False

    def test_schema_text_never_names_a_configurable_surface(self):
        blob = json.dumps(sd.PROVISION_SCHEMA).lower()
        for banned in ("image", "endpoint", "secret", "credential", "token",
                       "api_key", "network", "egress", "shell", "command",
                       "source_path", "workspace_path", "host"):
            assert banned not in blob, banned

    def test_extra_model_supplied_arguments_are_rejected(self, host, sdk):
        out = _provision({"image": "python:3.11", "source_path": "/etc"})
        assert "error" in out
        assert not FakeSandbox.created


# ---------------------------------------------------------------------------
# 2. Fail-closed gates
# ---------------------------------------------------------------------------


class TestGates:
    def test_check_fn_true_only_for_the_coding_worker(self, host):
        assert sd.check_provision_available() is True

    def test_check_fn_false_for_coordinator(self, host, monkeypatch):
        monkeypatch.setenv("HERMES_PROFILE", "default")
        assert sd.check_provision_available() is False

    def test_check_fn_false_without_kanban_run(self, host, monkeypatch):
        monkeypatch.delenv("HERMES_KANBAN_RUN_ID")
        assert sd.check_provision_available() is False

    def test_check_fn_false_for_delegated_child(self, host):
        from agent.delegation_context import delegated_child_context

        with delegated_child_context("s1"):
            assert sd.check_provision_available() is False

    def test_handler_refuses_another_profile(self, host, sdk, monkeypatch):
        monkeypatch.setenv("HERMES_PROFILE", "default")
        out = _provision()
        assert "scoped engineering" in out["error"]
        assert not FakeSandbox.created

    def test_handler_refuses_when_run_id_does_not_match_the_board(
        self, host, sdk, monkeypatch
    ):
        monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(host.run_id + 90))
        out = _provision()
        assert "error" in out
        assert not FakeSandbox.created

    def test_handler_refuses_scratch_workspace(self, host, sdk, monkeypatch):
        scratch = host.home / "scratch"
        scratch.mkdir()
        with kb.connect() as conn:
            conn.execute(
                "UPDATE tasks SET workspace_kind='scratch', workspace_path=? "
                "WHERE id = ?", (str(scratch), host.task_id),
            )
            conn.commit()
        monkeypatch.setenv("HERMES_KANBAN_WORKSPACE", str(scratch))
        out = _provision()
        assert "error" in out
        assert not FakeSandbox.created

    def test_handler_refuses_unscoped_ownership(self, host, sdk):
        with kb.connect() as conn:
            conn.execute(
                "UPDATE tasks SET owned_paths = NULL WHERE id = ?", (host.task_id,)
            )
            conn.commit()
        out = _provision()
        assert "error" in out
        assert not FakeSandbox.created

    def test_handler_refuses_dirty_source(self, host, sdk):
        (host.worktree / "app.py").write_text("print('dirty')\n", encoding="utf-8")
        out = _provision()
        assert "error" in out
        assert not FakeSandbox.created

    def test_handler_refuses_workspace_that_disagrees_with_the_board(
        self, host, sdk, monkeypatch
    ):
        monkeypatch.setenv("HERMES_KANBAN_WORKSPACE", str(host.repo))
        out = _provision()
        assert "error" in out
        assert not FakeSandbox.created

    def test_handler_refuses_a_mutable_image_tag(self, host, sdk, monkeypatch):
        monkeypatch.setattr(
            sd, "_load_host_config",
            lambda: {"image": "registry.internal.example/claude-worker:latest",
                     "domain": "sandbox.internal:8080", "protocol": "https"},
        )
        out = _provision()
        assert "error" in out
        assert not FakeSandbox.created

    def test_handler_refuses_without_a_configured_tunnel(self, host, sdk, monkeypatch):
        monkeypatch.setattr(sd, "_load_connection_secret_from_env", lambda: "")
        out = _provision()
        assert "error" in out
        assert not FakeSandbox.created

    def test_handler_refuses_without_a_host_credential(self, host, sdk, monkeypatch):
        monkeypatch.setattr(sd, "_resolve_host_credential", lambda: "")
        out = _provision()
        assert "error" in out
        assert not FakeSandbox.created

    def test_handler_refuses_when_the_sdk_is_absent(self, host, monkeypatch):
        def _missing():
            raise sd.SandboxDispatchError(
                "the OpenSandbox Python SDK is not installed"
            )

        monkeypatch.setattr(sd, "_load_sdk", _missing)
        out = _provision()
        assert "OpenSandbox" in out["error"]


# ---------------------------------------------------------------------------
# 3. The provisioned sandbox
# ---------------------------------------------------------------------------


class TestProvision:
    def test_receipt_is_sanitized_and_complete(self, host, sdk):
        out = _provision()
        assert set(out) == {
            "sandbox_id", "image_digest", "source_commit", "source_digest",
            "workspace", "baseline", "ownership_scope", "policy",
        }
        assert out["sandbox_id"] == "sbx-001"
        assert out["image_digest"] == IMAGE_DIGEST
        assert out["source_commit"] == host.head
        assert out["source_digest"].startswith("sha256:")
        assert out["workspace"] == sd.SANDBOX_WORKSPACE
        assert out["baseline"] == sd.SANDBOX_BASELINE
        assert out["ownership_scope"] == ["."]
        assert out["policy"] == {
            "credential_vault": True,
            "credential_proxy": True,
            "egress_default_deny": True,
            "sandbox_token_is_placeholder": True,
            "source_is_tracked_head_only": True,
            "source_tree_clean": True,
            "baseline_write_bits_removed": True,
            "idempotent_reuse": False,
        }

        blob = json.dumps(out)
        assert REAL_SECRET not in blob
        assert CONNECTION_SECRET not in blob
        assert str(host.worktree) not in blob
        assert str(host.home) not in blob
        assert "sandbox.internal" not in blob

    def test_sandbox_is_created_from_the_immutable_digest_with_bounded_limits(
        self, host, sdk
    ):
        _provision()
        box = FakeSandbox.created[0]
        assert box.create_kwargs["image"].image == IMAGE_REF
        assert box.create_kwargs["resource"] == sd.SANDBOX_RESOURCES
        assert box.create_kwargs["timeout"].total_seconds() == sd.SANDBOX_TIMEOUT_SECONDS

    def test_egress_defaults_to_deny_with_only_the_minimum_allowlist(self, host, sdk):
        _provision()
        policy = FakeSandbox.created[0].create_kwargs["network_policy"]
        assert policy.default_action == "deny"
        assert [r.action for r in policy.egress] == ["allow"] * len(policy.egress)
        assert [r.target for r in policy.egress] == list(sd.EGRESS_ALLOWLIST)
        assert "api.anthropic.com" in sd.EGRESS_ALLOWLIST

    def test_credential_proxy_is_enabled_and_only_a_placeholder_is_in_the_env(
        self, host, sdk
    ):
        _provision()
        kwargs = FakeSandbox.created[0].create_kwargs
        assert kwargs["credential_proxy"].enabled is True
        env = kwargs["env"]
        assert env["ANTHROPIC_API_KEY"] == sd.PLACEHOLDER_TOKEN
        assert REAL_SECRET not in json.dumps(env)
        assert CONNECTION_SECRET not in json.dumps(env)

    def test_real_credential_goes_only_to_the_vault_with_a_minimum_v1_binding(
        self, host, sdk
    ):
        _provision()
        call = FakeSandbox.created[0].vault_calls[0]
        (cred,) = call["credentials"]
        assert cred.name == sd.VAULT_CREDENTIAL_NAME
        assert cred.source.value == REAL_SECRET
        (binding,) = call["bindings"]
        assert binding.match.hosts == ["api.anthropic.com"]
        assert binding.match.schemes == ["https"]
        assert binding.match.paths == ["/v1/*"]
        assert binding.auth.type == "apiKey"
        assert binding.auth.name == "x-api-key"
        assert binding.auth.credential == sd.VAULT_CREDENTIAL_NAME

    def test_the_vault_binding_admits_only_get_and_post(self, host, sdk):
        """The verb contour is exact, so no mutating verb is ever signed."""
        _provision()
        (binding,) = FakeSandbox.created[0].vault_calls[0]["bindings"]
        assert list(sd.VAULT_MATCH_METHODS) == ["GET", "POST"]
        assert binding.match.methods == ["GET", "POST"]
        admitted = list(binding.match.methods or [])
        for refused in ("DELETE", "PUT", "PATCH", "HEAD", "OPTIONS", "*"):
            assert refused not in admitted, refused
        # The rest of the contour is unchanged by the narrowing.
        assert binding.match.schemes == ["https"]
        assert binding.match.hosts == [sd.VAULT_MATCH_HOST]
        assert binding.match.paths == [sd.VAULT_MATCH_PATH]

    def test_oauth_host_credential_binds_as_bearer(self, host, sdk, monkeypatch):
        monkeypatch.setattr(sd, "_resolve_host_credential", lambda: REAL_OAUTH)
        _provision()
        box = FakeSandbox.created[0]
        assert box.create_kwargs["env"]["ANTHROPIC_AUTH_TOKEN"] == sd.PLACEHOLDER_TOKEN
        assert "ANTHROPIC_API_KEY" not in box.create_kwargs["env"]
        (binding,) = box.vault_calls[0]["bindings"]
        assert binding.auth.type == "bearer"
        assert binding.auth.credential == sd.VAULT_CREDENTIAL_NAME

    def test_only_the_tracked_head_is_packaged_and_extracted_to_the_fixed_path(
        self, host, sdk
    ):
        (host.worktree / "untracked-secret.txt").write_text("nope\n", encoding="utf-8")
        out = _provision()
        box = FakeSandbox.created[0]
        (path, data) = box.uploads[0]
        assert isinstance(data, (bytes, bytearray))
        names = subprocess.run(
            ["tar", "-tf", "-"], input=bytes(data), capture_output=True, check=True,
        ).stdout.decode()
        assert "app.py" in names
        assert "README.md" in names
        assert "untracked-secret.txt" not in names
        assert sd.SANDBOX_WORKSPACE in box.commands_log[0]
        assert path in box.commands_log[0]
        assert out["policy"]["source_is_tracked_head_only"] is True

    def test_no_host_path_or_secret_reaches_the_sandbox(self, host, sdk):
        _provision()
        box = FakeSandbox.created[0]
        blob = json.dumps(
            {"env": box.create_kwargs["env"], "meta": box.create_kwargs.get("metadata"),
             "cmds": box.commands_log}
        )
        assert str(host.worktree) not in blob
        assert str(host.repo) not in blob
        assert REAL_SECRET not in blob

    def test_host_temporary_material_is_removed(self, host, sdk):
        _provision()
        leftovers = list((host.home / "raphael").glob("**/*.tar"))
        assert leftovers == []


# ---------------------------------------------------------------------------
# 3b. The fixed second extraction the worker's recursive patch is taken against
# ---------------------------------------------------------------------------


def _command_steps(box: FakeSandbox) -> list:
    """The seeding command split into its ordered, individual steps."""
    (command,) = box.commands_log
    return [segment.strip() for segment in command.split(";") if segment.strip()]


class TestReviewBaseline:
    """The baseline is a convenience for generating the patch, nothing more.

    A live Server 2 probe against the pinned image showed the maintained MCP
    ``command_run`` running as uid 0, with no uid/gid parameter in its
    published schema, and uid 0 rewrote a baseline whose write bits had already
    been removed. So these tests fix what the seeding command actually does —
    two extractions of one archive, then ``chmod -R a-w`` on the second — and
    refuse any read-only or immutability claim built on top of it. Acceptance
    belongs to the independent Reviewer, which reapplies the produced patch to
    the receipt's source commit and verifies that candidate.
    """

    def test_the_two_trees_are_fixed_distinct_and_not_nested(self):
        assert sd.SANDBOX_WORKSPACE == "/workspace/task"
        assert sd.SANDBOX_BASELINE == "/workspace/baseline"
        # Neither may sit inside the other, or a recursive diff of the work
        # tree would walk into the baseline and report itself.
        assert not sd.SANDBOX_BASELINE.startswith(sd.SANDBOX_WORKSPACE + "/")
        assert not sd.SANDBOX_WORKSPACE.startswith(sd.SANDBOX_BASELINE + "/")

    def test_both_trees_are_seeded_from_the_one_transferred_archive(self, host, sdk):
        _provision()
        box = FakeSandbox.created[0]
        # One transfer, extracted twice: the trees start byte-identical, and
        # nothing is packaged, uploaded or copied a second time.
        assert [path for path, _ in box.uploads] == [sd.SANDBOX_ARCHIVE_PATH]
        steps = _command_steps(box)
        assert [step for step in steps if step.startswith("tar -xf")] == [
            f"tar -xf '{sd.SANDBOX_ARCHIVE_PATH}' -C '{sd.SANDBOX_WORKSPACE}'",
            f"tar -xf '{sd.SANDBOX_ARCHIVE_PATH}' -C '{sd.SANDBOX_BASELINE}'",
        ]
        assert f"mkdir -p '{sd.SANDBOX_WORKSPACE}' '{sd.SANDBOX_BASELINE}'" in steps

    def test_the_baseline_write_bits_are_removed_once_it_holds_the_tree(
        self, host, sdk
    ):
        _provision()
        steps = _command_steps(FakeSandbox.created[0])
        chmod = f"chmod -R a-w '{sd.SANDBOX_BASELINE}'"
        assert chmod in steps
        assert steps.index(chmod) > steps.index(
            f"tar -xf '{sd.SANDBOX_ARCHIVE_PATH}' -C '{sd.SANDBOX_BASELINE}'"
        )
        # Only the baseline loses its write bits; the worker's own tree stays
        # writable. Neither mode change makes a tree unwritable to uid 0.
        assert f"chmod -R a-w '{sd.SANDBOX_WORKSPACE}'" not in steps
        # And the transferred archive leaves no third copy behind.
        assert f"rm -f '{sd.SANDBOX_ARCHIVE_PATH}'" in steps

    def test_the_receipt_names_the_baseline_and_its_policy_fact(self, host, sdk):
        out = _provision()
        assert out["baseline"] == sd.SANDBOX_BASELINE
        assert out["workspace"] == sd.SANDBOX_WORKSPACE
        assert out["policy"]["baseline_write_bits_removed"] is True
        # A reused machine hands back the same handoff facts.
        reused = _provision()
        assert reused["baseline"] == sd.SANDBOX_BASELINE
        assert reused["policy"]["baseline_write_bits_removed"] is True
        assert reused["policy"]["idempotent_reuse"] is True

    def test_the_receipt_claims_no_read_only_or_immutable_baseline(self, host, sdk):
        """uid 0 rewrote a chmod-ed baseline on this image, so nothing may.

        The receipt states only what the seeding command did. The retired
        ``baseline_is_read_only`` fact was false — removing write bits stops an
        accidental non-root write, not the worker itself.
        """
        out = _provision()
        reused = _provision()
        for policy in (out["policy"], reused["policy"]):
            assert policy["baseline_write_bits_removed"] is True
            assert "baseline_is_read_only" not in policy
            for name in policy:
                assert "read_only" not in name, name
                assert "immutable" not in name, name

    def test_the_tool_prose_offers_the_baseline_as_a_diff_aid_only(self):
        """What the worker is told about the baseline must match the probe."""
        blob = json.dumps(sd.PROVISION_SCHEMA).lower()
        for claimed in ("read-only", "read only", "immutable", "unwritable",
                        "frozen", "tamper"):
            assert claimed not in blob, claimed
        assert "write bits removed" in blob
        assert "diff aid" in blob
        # And it points acceptance at the patch, not at either sandbox tree.
        assert "reapplying your patch" in blob

    def test_the_baseline_step_carries_no_host_path_or_secret(self, host, sdk):
        _provision()
        (command,) = FakeSandbox.created[0].commands_log
        for hidden in (str(host.worktree), str(host.repo), str(host.home),
                       host.task_id, REAL_SECRET, CONNECTION_SECRET):
            assert hidden not in command
        # No version-control metadata is seeded either: the diff is between
        # two extracted trees, so the sandbox never needs a repository.
        assert "git" not in command
        assert ".git" not in json.dumps(
            [path for path, _ in FakeSandbox.created[0].uploads]
        )


# ---------------------------------------------------------------------------
# 4. The Server 1 → Server 2 transport
# ---------------------------------------------------------------------------


class TestTunnelTransport:
    def test_creation_goes_through_the_loopback_tunnel_proxy(self, host, sdk):
        _provision()
        config = FakeSandbox.created[0].create_kwargs["connection_config"]
        assert isinstance(config, sdk.connection_config)
        assert config.use_server_proxy is True
        assert config.domain == "sandbox.internal:8080"
        assert config.protocol == "https"
        assert config.api_key == CONNECTION_SECRET
        assert (
            config.request_timeout.total_seconds()
            == sd.SANDBOX_REQUEST_TIMEOUT_SECONDS
        )

    def test_liveness_reattachment_also_uses_the_tunnel_proxy(self, host, sdk):
        _provision()
        _provision()
        assert FakeSandbox.connect_calls, "expected a liveness re-attachment"
        config = FakeSandbox.connect_calls[0]["connection_config"]
        assert config.use_server_proxy is True

    def test_the_proxy_is_not_operator_tunable(self, host, sdk, monkeypatch):
        monkeypatch.setattr(
            sd, "_load_host_config",
            lambda: {"image": IMAGE_REF, "domain": "sandbox.internal:8080",
                     "protocol": "https", "use_server_proxy": False},
        )
        _provision()
        config = FakeSandbox.created[0].create_kwargs["connection_config"]
        assert config.use_server_proxy is True


# ---------------------------------------------------------------------------
# 5. The Server 2 API key: .env first, then a trusted operator file
# ---------------------------------------------------------------------------


def _write_secret_file(path: Path, body: str, mode: int = 0o600) -> Path:
    path.write_text(body, encoding="utf-8")
    os.chmod(path, mode)
    return path


class TestConnectionSecretFile:
    @pytest.fixture
    def no_env_secret(self, host, monkeypatch):
        monkeypatch.setattr(sd, "_load_connection_secret_from_env", lambda: "")
        return host

    def _use_file(self, monkeypatch, path):
        monkeypatch.setattr(
            sd, "_load_host_config",
            lambda: {"image": IMAGE_REF, "domain": "sandbox.internal:8080",
                     "protocol": "https", "api_key_file": str(path)},
        )

    def test_env_wins_over_the_file(self, host, tmp_path, monkeypatch):
        path = _write_secret_file(
            tmp_path / "creds.env", "OPEN_SANDBOX_API_KEY=from-the-file\n"
        )
        self._use_file(monkeypatch, path)
        assert sd._load_connection_secret() == CONNECTION_SECRET

    def test_a_valid_owner_only_file_supplies_the_key(
        self, no_env_secret, tmp_path, monkeypatch
    ):
        path = _write_secret_file(
            tmp_path / "creds.env",
            "# server 2\nexport OPEN_SANDBOX_API_KEY=\"file-key-value\"\n",
        )
        self._use_file(monkeypatch, path)
        assert sd._load_connection_secret() == "file-key-value"

    def test_a_valid_file_reaches_the_sdk_connection(
        self, no_env_secret, sdk, tmp_path, monkeypatch
    ):
        path = _write_secret_file(
            tmp_path / "creds.env", "OPEN_SANDBOX_API_KEY=file-key-value\n"
        )
        self._use_file(monkeypatch, path)
        out = _provision()
        assert out["sandbox_id"] == "sbx-001"
        config = FakeSandbox.created[0].create_kwargs["connection_config"]
        assert config.api_key == "file-key-value"
        assert "file-key-value" not in json.dumps(out)

    def test_no_configured_file_is_not_an_error_just_no_fallback(
        self, no_env_secret, sdk
    ):
        assert sd._load_connection_secret() == ""
        out = _provision()
        assert "error" in out
        assert not FakeSandbox.created

    def test_a_missing_file_refuses(self, no_env_secret, tmp_path, monkeypatch):
        self._use_file(monkeypatch, tmp_path / "absent.env")
        with pytest.raises(sd.SandboxDispatchError) as excinfo:
            sd._load_connection_secret()
        assert excinfo.value.code == "connection_secret_file"

    def test_a_relative_path_refuses(self, no_env_secret, monkeypatch):
        self._use_file(monkeypatch, Path("creds.env"))
        with pytest.raises(sd.SandboxDispatchError):
            sd._load_connection_secret()

    @pytest.mark.parametrize(
        "body",
        [
            "OTHER_KEY=value\n",
            "OPEN_SANDBOX_API_KEY=\n",
            "OPEN_SANDBOX_API_KEY=has a space\n",
            "OPEN_SANDBOX_API_KEYX=value\n",
            "not an assignment at all\n",
            "",
        ],
    )
    def test_a_malformed_file_refuses(
        self, no_env_secret, tmp_path, monkeypatch, body
    ):
        path = _write_secret_file(tmp_path / "creds.env", body)
        self._use_file(monkeypatch, path)
        with pytest.raises(sd.SandboxDispatchError) as excinfo:
            sd._load_connection_secret()
        assert excinfo.value.code == "connection_secret_file"

    @pytest.mark.skipif(os.name == "nt", reason="POSIX permission bits")
    @pytest.mark.parametrize("mode", [0o644, 0o660, 0o606])
    def test_a_loose_mode_file_refuses(
        self, no_env_secret, tmp_path, monkeypatch, mode
    ):
        path = _write_secret_file(
            tmp_path / "creds.env", "OPEN_SANDBOX_API_KEY=file-key-value\n", mode=mode
        )
        self._use_file(monkeypatch, path)
        with pytest.raises(sd.SandboxDispatchError) as excinfo:
            sd._load_connection_secret()
        assert excinfo.value.code == "connection_secret_file"

    @pytest.mark.skipif(not hasattr(os, "symlink"), reason="needs symlinks")
    def test_a_symlink_refuses_even_when_the_target_is_safe(
        self, no_env_secret, tmp_path, monkeypatch
    ):
        target = _write_secret_file(
            tmp_path / "real.env", "OPEN_SANDBOX_API_KEY=file-key-value\n"
        )
        link = tmp_path / "link.env"
        link.symlink_to(target)
        self._use_file(monkeypatch, link)
        with pytest.raises(sd.SandboxDispatchError) as excinfo:
            sd._load_connection_secret()
        assert excinfo.value.code == "connection_secret_file"

    def test_a_directory_refuses(self, no_env_secret, tmp_path, monkeypatch):
        directory = tmp_path / "creds.d"
        directory.mkdir(mode=0o700)
        self._use_file(monkeypatch, directory)
        with pytest.raises(sd.SandboxDispatchError):
            sd._load_connection_secret()

    def test_the_value_is_read_literally_and_no_shell_ever_runs(
        self, no_env_secret, sdk, tmp_path, monkeypatch
    ):
        marker = tmp_path / "pwned"
        path = _write_secret_file(
            tmp_path / "creds.env",
            f"OPEN_SANDBOX_API_KEY=$(touch;{marker})\n",
        )
        self._use_file(monkeypatch, path)
        assert sd._load_connection_secret() == f"$(touch;{marker})"
        assert not marker.exists()

    def test_the_last_assignment_wins(self, no_env_secret, tmp_path, monkeypatch):
        path = _write_secret_file(
            tmp_path / "creds.env",
            "OPEN_SANDBOX_API_KEY=first\nOPEN_SANDBOX_API_KEY='second'\n",
        )
        self._use_file(monkeypatch, path)
        assert sd._load_connection_secret() == "second"

    def test_a_refusal_never_names_the_path_or_the_value(
        self, no_env_secret, sdk, tmp_path, monkeypatch, caplog
    ):
        path = _write_secret_file(
            tmp_path / "very-private-creds.env",
            "OPEN_SANDBOX_API_KEY=file-key-value\n",
            mode=0o644,
        )
        self._use_file(monkeypatch, path)
        with caplog.at_level(logging.DEBUG):
            out = _provision()
        assert "error" in out
        for blob in (json.dumps(out), caplog.text):
            assert str(path) not in blob
            assert "very-private-creds" not in blob
            assert "file-key-value" not in blob


# ---------------------------------------------------------------------------
# 6. Credential preflight: a Claude account token must outlive the sandbox
# ---------------------------------------------------------------------------


@pytest.fixture
def refresh_spy(monkeypatch):
    """Replace the host refresh path and record how it was called."""
    import agent.anthropic_adapter as adapter

    calls: list = []

    def _fake_refresh(creds):
        calls.append(dict(creds))
        return calls_result["value"]

    calls_result = {"value": FRESH_OAUTH}
    monkeypatch.setattr(adapter, "_refresh_oauth_token", _fake_refresh)
    return SimpleNamespace(calls=calls, result=calls_result)


def _account_record(token: str, expires_at_ms: int) -> dict:
    return {
        "accessToken": token,
        "refreshToken": REFRESH_TOKEN,
        "expiresAt": expires_at_ms,
        "source": "claude_code_credentials_file",
    }


class TestCredentialPreflight:
    def test_a_short_lived_account_token_is_refreshed_before_the_vault_write(
        self, host, sdk, monkeypatch, refresh_spy
    ):
        records = [_account_record(REAL_OAUTH, _expires_in(600))]
        monkeypatch.setattr(sd, "_resolve_host_credential", lambda: REAL_OAUTH)
        monkeypatch.setattr(sd, "_claude_account_credentials", lambda: records[-1])

        def _refresh(creds):
            refresh_spy.calls.append(dict(creds))
            records.append(_account_record(FRESH_OAUTH, _expires_in(8 * 3600)))
            return FRESH_OAUTH

        import agent.anthropic_adapter as adapter

        monkeypatch.setattr(adapter, "_refresh_oauth_token", _refresh)

        out = _provision()
        assert out["sandbox_id"] == "sbx-001"
        assert len(refresh_spy.calls) == 1
        (cred,) = FakeSandbox.created[0].vault_calls[0]["credentials"]
        assert cred.source.value == FRESH_OAUTH

    def test_a_long_lived_account_token_is_not_refreshed(
        self, host, sdk, monkeypatch, refresh_spy
    ):
        monkeypatch.setattr(sd, "_resolve_host_credential", lambda: REAL_OAUTH)
        monkeypatch.setattr(
            sd, "_claude_account_credentials",
            lambda: _account_record(REAL_OAUTH, _expires_in(8 * 3600)),
        )
        _provision()
        assert refresh_spy.calls == []
        (cred,) = FakeSandbox.created[0].vault_calls[0]["credentials"]
        assert cred.source.value == REAL_OAUTH

    def test_a_setup_token_contour_with_no_declared_expiry_is_untouched(
        self, host, sdk, monkeypatch, refresh_spy
    ):
        monkeypatch.setattr(sd, "_resolve_host_credential", lambda: REAL_OAUTH)
        monkeypatch.setattr(
            sd, "_claude_account_credentials",
            lambda: _account_record(REAL_OAUTH, 0),
        )
        _provision()
        assert refresh_spy.calls == []
        (cred,) = FakeSandbox.created[0].vault_calls[0]["credentials"]
        assert cred.source.value == REAL_OAUTH

    def test_an_api_key_is_never_refreshed(
        self, host, sdk, monkeypatch, refresh_spy
    ):
        monkeypatch.setattr(
            sd, "_claude_account_credentials",
            lambda: _account_record(REAL_SECRET, _expires_in(60)),
        )
        _provision()
        assert refresh_spy.calls == []
        (cred,) = FakeSandbox.created[0].vault_calls[0]["credentials"]
        assert cred.source.value == REAL_SECRET

    def test_a_token_that_is_not_the_account_credential_is_untouched(
        self, host, sdk, monkeypatch, refresh_spy
    ):
        monkeypatch.setattr(sd, "_resolve_host_credential", lambda: REAL_OAUTH)
        monkeypatch.setattr(
            sd, "_claude_account_credentials",
            lambda: _account_record("sk-ant-oat01-some-other-token", _expires_in(60)),
        )
        _provision()
        assert refresh_spy.calls == []

    def test_an_unrefreshable_short_lived_token_fails_closed(
        self, host, sdk, monkeypatch, refresh_spy
    ):
        monkeypatch.setattr(sd, "_resolve_host_credential", lambda: REAL_OAUTH)
        monkeypatch.setattr(
            sd, "_claude_account_credentials",
            lambda: _account_record(REAL_OAUTH, _expires_in(600)),
        )
        refresh_spy.result["value"] = None
        out = _provision()
        assert "error" in out
        assert "never ask a person to paste a key" in out["error"]
        assert not FakeSandbox.created
        assert _reservation(host)["state"] == "released"

    def test_a_refresh_that_is_still_too_short_fails_closed(
        self, host, sdk, monkeypatch, refresh_spy
    ):
        records = [_account_record(REAL_OAUTH, _expires_in(600))]
        monkeypatch.setattr(sd, "_resolve_host_credential", lambda: REAL_OAUTH)
        monkeypatch.setattr(sd, "_claude_account_credentials", lambda: records[-1])

        def _refresh(creds):
            refresh_spy.calls.append(dict(creds))
            records.append(_account_record(FRESH_OAUTH, _expires_in(900)))
            return FRESH_OAUTH

        import agent.anthropic_adapter as adapter

        monkeypatch.setattr(adapter, "_refresh_oauth_token", _refresh)

        out = _provision()
        assert "error" in out
        assert not FakeSandbox.created

    def test_a_concurrent_rotation_winner_is_adopted_when_it_is_provably_usable(
        self, host, sdk, monkeypatch, refresh_spy, caplog
    ):
        """Another process won the rotation: trust the record, not this result.

        An Anthropic refresh token is single-use, so once a concurrent Server 1
        process has rotated the pair, the token this call was handed may
        already be dead. The current record's token is adopted — but only
        because it is present, still an account token, and outlives the whole
        sandbox lease plus the buffer.
        """
        records = [_account_record(REAL_OAUTH, _expires_in(600))]
        monkeypatch.setattr(sd, "_resolve_host_credential", lambda: REAL_OAUTH)
        monkeypatch.setattr(sd, "_claude_account_credentials", lambda: records[-1])

        def _refresh(creds):
            refresh_spy.calls.append(dict(creds))
            records.append(_account_record(WINNER_OAUTH, _expires_in(8 * 3600)))
            return FRESH_OAUTH

        import agent.anthropic_adapter as adapter

        monkeypatch.setattr(adapter, "_refresh_oauth_token", _refresh)

        with caplog.at_level(logging.DEBUG):
            out = _provision()

        assert out["sandbox_id"] == "sbx-001"
        box = FakeSandbox.created[0]
        (cred,) = box.vault_calls[0]["credentials"]
        assert cred.source.value == WINNER_OAUTH
        # This call's own superseded result never reaches the machine.
        assert FRESH_OAUTH != cred.source.value
        assert FRESH_OAUTH not in json.dumps(box.create_kwargs["env"])
        for hidden in (WINNER_OAUTH, FRESH_OAUTH, REAL_OAUTH, REFRESH_TOKEN):
            assert hidden not in json.dumps(out)
            assert hidden not in json.dumps(box.create_kwargs["env"])
            assert hidden not in caplog.text

    @pytest.mark.parametrize(
        "token,expires_in,case",
        [
            (WINNER_OAUTH, 600, "the winner's token dies inside the lease"),
            (WINNER_OAUTH, None, "the winner's record declares no expiry"),
            ("", 8 * 3600, "the record no longer holds an access token"),
            ("not-an-anthropic-token", 8 * 3600, "the record holds an unusable value"),
        ],
    )
    def test_an_unusable_concurrent_rotation_result_fails_closed(
        self, host, sdk, monkeypatch, refresh_spy, caplog, token, expires_in, case
    ):
        """A record that moved on is never itself a reason to hand out a token."""
        records = [_account_record(REAL_OAUTH, _expires_in(600))]
        monkeypatch.setattr(sd, "_resolve_host_credential", lambda: REAL_OAUTH)
        monkeypatch.setattr(sd, "_claude_account_credentials", lambda: records[-1])

        def _refresh(creds):
            refresh_spy.calls.append(dict(creds))
            records.append(
                _account_record(
                    token,
                    _expires_in(expires_in) if expires_in is not None else 0,
                )
            )
            return FRESH_OAUTH

        import agent.anthropic_adapter as adapter

        monkeypatch.setattr(adapter, "_refresh_oauth_token", _refresh)

        with caplog.at_level(logging.DEBUG):
            out = _provision()

        assert "error" in out, case
        assert "never ask a person to paste a key" in out["error"]
        assert not FakeSandbox.created
        # The generation this attempt owned is closed, so a corrected retry is
        # not wedged behind it.
        assert _reservation(host)["state"] == "released"
        for hidden in (WINNER_OAUTH, FRESH_OAUTH, REAL_OAUTH, REFRESH_TOKEN):
            assert hidden not in out["error"]
            assert hidden not in caplog.text

    def test_the_refresh_token_never_leaves_server_1(
        self, host, sdk, monkeypatch, refresh_spy, caplog
    ):
        records = [_account_record(REAL_OAUTH, _expires_in(600))]
        monkeypatch.setattr(sd, "_resolve_host_credential", lambda: REAL_OAUTH)
        monkeypatch.setattr(sd, "_claude_account_credentials", lambda: records[-1])

        def _refresh(creds):
            records.append(_account_record(FRESH_OAUTH, _expires_in(8 * 3600)))
            return FRESH_OAUTH

        import agent.anthropic_adapter as adapter

        monkeypatch.setattr(adapter, "_refresh_oauth_token", _refresh)

        with caplog.at_level(logging.DEBUG):
            out = _provision()
        box = FakeSandbox.created[0]
        exposed = json.dumps(
            {
                "env": box.create_kwargs["env"],
                "metadata": box.create_kwargs.get("metadata"),
                "commands": box.commands_log,
                "receipt": out,
                "vault": [c.source.value for c in box.vault_calls[0]["credentials"]],
            }
        )
        assert REFRESH_TOKEN not in exposed
        assert REFRESH_TOKEN not in caplog.text
        assert REAL_OAUTH not in json.dumps(box.create_kwargs["env"])

    def test_the_safety_buffer_is_clear_of_the_sandbox_lifetime(self):
        assert sd.CREDENTIAL_SAFETY_BUFFER_SECONDS > 0
        assert (
            sd.SANDBOX_TIMEOUT_SECONDS + sd.CREDENTIAL_SAFETY_BUFFER_SECONDS
            > sd.SANDBOX_TIMEOUT_SECONDS
        )


# ---------------------------------------------------------------------------
# 7. Failure handling and client-handle lifecycle
# ---------------------------------------------------------------------------


class TestFailureCleanup:
    def test_success_closes_the_client_handle_without_killing_the_machine(
        self, host, sdk
    ):
        _provision()
        box = FakeSandbox.created[0]
        assert box.closed is True
        assert box.killed is False
        assert FakeSandbox.live[box.id] is box

    def test_vault_failure_kills_and_closes_and_hides_the_secret(self, host, sdk):
        FakeSandbox.behavior = {"vault_fails": True}
        out = _provision()
        assert "error" in out
        assert REAL_SECRET not in json.dumps(out)
        assert FakeSandbox.created[0].killed is True
        assert FakeSandbox.created[0].closed is True

    def test_extraction_failure_kills_the_sandbox(self, host, sdk):
        FakeSandbox.behavior = {"extract_fails": True}
        out = _provision()
        assert "error" in out
        assert FakeSandbox.created[0].killed is True
        assert FakeSandbox.created[0].closed is True

    def test_upload_failure_kills_the_sandbox(self, host, sdk):
        FakeSandbox.behavior = {"upload_fails": True}
        out = _provision()
        assert "error" in out
        assert FakeSandbox.created[0].killed is True

    def test_an_unexpected_failure_after_creation_still_kills_the_sandbox(
        self, host, sdk, monkeypatch
    ):
        def _boom(*_args, **_kwargs):
            raise RuntimeError(f"unexpected fault carrying {REAL_SECRET}")

        monkeypatch.setattr(sd, "_vault_payload", _boom)
        out = _provision()
        assert "error" in out
        assert REAL_SECRET not in json.dumps(out)
        assert FakeSandbox.created[0].killed is True
        assert FakeSandbox.created[0].closed is True

    def test_a_failed_attempt_leaves_no_live_sandbox_recorded(self, host, sdk):
        FakeSandbox.behavior = {"vault_fails": True}
        _provision()
        assert _reservation(host)["state"] == "released"
        FakeSandbox.behavior = {}
        out = _provision()
        assert out.get("sandbox_id") == "sbx-002"
        assert len(FakeSandbox.created) == 2
        assert _reservation(host)["generation"] == 2


class TestUnexpectedFailureEvent:
    def test_the_unexpected_path_logs_only_the_exception_type(
        self, host, sdk, monkeypatch, caplog
    ):
        def _boom(*_args, **_kwargs):
            raise ZeroDivisionError(f"host fault leaking {REAL_SECRET}")

        monkeypatch.setattr(sd, "_resolve_source", _boom)
        with caplog.at_level(logging.DEBUG):
            out = _provision()

        assert "unexpected host fault" in out["error"]
        assert REAL_SECRET not in caplog.text
        assert "leaking" not in caplog.text
        assert "Traceback" not in caplog.text
        assert all(record.exc_info is None for record in caplog.records)
        events = [
            (record, json.loads(record.message.split(": ", 1)[1]))
            for record in caplog.records
            if record.message.startswith("raphael sandbox dispatch: {")
        ]
        ((record, failure),) = [e for e in events if e[1].get("outcome") == "failed"]
        assert record.levelno == logging.ERROR
        assert failure["reason"] == "internal_error"
        assert failure["exception_type"] == "ZeroDivisionError"
        # The type name is the only exception-derived field in the event.
        assert set(failure) == {
            "outcome", "profile", "task_id", "run_id", "board",
            "reason", "exception_type",
        }


# ---------------------------------------------------------------------------
# 8. One native authority: no second state store
# ---------------------------------------------------------------------------


class TestNativeAuthority:
    def test_the_receipt_is_recorded_in_the_task_event_log(self, host, sdk):
        out = _provision()
        record = _reservation(host)
        assert record["state"] == "active"
        assert record["generation"] == 1
        assert record["sandbox_id"] == "sbx-001"
        assert record["receipt"] == out
        assert _event_kinds(host) == ["sandbox_reserved", "sandbox_provisioned"]

    def test_no_side_json_ledger_is_written(self, host, sdk):
        _provision()
        assert not (host.home / "raphael" / "sandbox_dispatch.json").exists()
        assert list((host.home / "raphael").glob("*.json")) == []
        for removed in ("read_audit_records", "_ledger_path", "_locked_ledger",
                        "_read_ledger", "_write_ledger", "_reserve", "_finalize",
                        "_release", "_audit", "_idempotency_key"):
            assert not hasattr(sd, removed), removed

    def test_a_failed_attempt_releases_the_exact_generation(self, host, sdk):
        FakeSandbox.behavior = {"vault_fails": True}
        _provision()
        record = _reservation(host)
        assert record == {
            "generation": 1,
            "state": "released",
            "sandbox_id": None,
            "receipt": None,
        }
        assert _event_kinds(host) == ["sandbox_reserved", "sandbox_released"]

    def test_the_board_is_the_only_thing_a_new_process_needs(self, host, sdk):
        first = _provision()
        # Nothing in this module's memory carries the reservation forward.
        second = _provision()
        assert second["sandbox_id"] == first["sandbox_id"]
        assert len(FakeSandbox.created) == 1


# ---------------------------------------------------------------------------
# 9. Idempotent reuse, proved live before it is offered
# ---------------------------------------------------------------------------


class TestIdempotentReuse:
    def test_retry_returns_the_same_sandbox_without_creating_another(self, host, sdk):
        first = _provision()
        second = _provision()
        assert len(FakeSandbox.created) == 1
        assert second["sandbox_id"] == first["sandbox_id"]
        assert second["source_commit"] == first["source_commit"]
        assert second["policy"]["idempotent_reuse"] is True
        assert first["policy"]["idempotent_reuse"] is False
        assert _event_kinds(host) == ["sandbox_reserved", "sandbox_provisioned"]

    def test_reuse_proves_liveness_over_the_trusted_connection(self, host, sdk):
        _provision()
        _provision()
        assert [c["sandbox_id"] for c in FakeSandbox.connect_calls] == ["sbx-001"]

    def test_reuse_closes_the_verification_handle_without_killing(self, host, sdk):
        _provision()
        box = FakeSandbox.created[0]
        box.closed = False
        _provision()
        assert box.closed is True
        assert box.killed is False

    def test_a_new_run_gets_a_new_sandbox(self, host, sdk, monkeypatch):
        _provision()
        with kb.connect() as conn:
            with kb.write_txn(conn):
                conn.execute(
                    "INSERT INTO task_runs (task_id, status, started_at) "
                    "VALUES (?, 'running', 0)",
                    (host.task_id,),
                )
                next_run = int(
                    conn.execute("SELECT last_insert_rowid() AS id").fetchone()["id"]
                )
                conn.execute(
                    "UPDATE tasks SET current_run_id = ? WHERE id = ?",
                    (next_run, host.task_id),
                )
        monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(next_run))
        second = _provision()
        assert len(FakeSandbox.created) == 2
        assert second["sandbox_id"] == "sbx-002"


# ---------------------------------------------------------------------------
# 10. A recorded machine that is no longer usable is replaced exactly once
# ---------------------------------------------------------------------------


class TestReplacement:
    def _break(self, host, **attrs):
        box = FakeSandbox.created[0]
        for key, value in attrs.items():
            setattr(box, key, value)
        return box

    def _assert_replaced(self, host, out):
        assert out["sandbox_id"] == "sbx-002"
        assert len(FakeSandbox.created) == 2
        assert _event_kinds(host) == [
            "sandbox_reserved", "sandbox_provisioned",
            "sandbox_released", "sandbox_reserved", "sandbox_provisioned",
        ]
        record = _reservation(host)
        assert record["generation"] == 2
        assert record["state"] == "active"
        assert record["sandbox_id"] == "sbx-002"

    def test_a_vanished_machine_is_replaced(self, host, sdk):
        _provision()
        FakeSandbox.live.clear()
        self._assert_replaced(host, _provision())

    def test_a_stopped_machine_is_replaced(self, host, sdk):
        _provision()
        self._break(host, state="TERMINATED")
        self._assert_replaced(host, _provision())

    def test_an_unhealthy_machine_is_replaced(self, host, sdk):
        _provision()
        self._break(host, healthy=False)
        self._assert_replaced(host, _provision())

    def test_an_expiring_machine_is_replaced(self, host, sdk):
        _provision()
        self._break(host, expires_in_seconds=sd.SANDBOX_MIN_REMAINING_SECONDS - 30)
        self._assert_replaced(host, _provision())

    def test_a_machine_running_another_image_is_replaced(self, host, sdk):
        _provision()
        self._break(
            host,
            reported_image="registry.internal.example/other@sha256:" + "cd" * 32,
        )
        self._assert_replaced(host, _provision())

    def test_a_machine_whose_info_is_unreadable_is_replaced(self, host, sdk):
        _provision()
        self._break(host, info_fails=True)
        self._assert_replaced(host, _provision())

    def test_a_stopped_machine_is_killed_because_it_is_provably_this_runs(
        self, host, sdk
    ):
        _provision()
        box = self._break(host, state="TERMINATED")
        _provision()
        assert box.killed is True

    def test_a_machine_that_belongs_to_someone_else_is_never_killed(self, host, sdk):
        _provision()
        box = self._break(
            host, reported_metadata={"hermes_task": "t_other", "hermes_run": "99"}
        )
        out = _provision()
        assert box.killed is False
        assert box.closed is True
        self._assert_replaced(host, out)

    def test_a_malformed_recorded_receipt_is_never_reused(self, host, sdk):
        _provision()
        box = FakeSandbox.created[0]
        for receipt in (None, "not-a-receipt", {"sandbox_id": "sbx-999"}):
            box.killed = False
            assert (
                sd._receipt_still_holds(
                    box,
                    sd._worker_context(),
                    {"sandbox_id": box.id, "receipt": receipt},
                )
                == "retire"
            )

    def test_a_receipt_without_an_image_digest_is_never_reused(self, host, sdk):
        _provision()
        box = FakeSandbox.created[0]
        assert (
            sd._receipt_still_holds(
                box,
                sd._worker_context(),
                {"sandbox_id": box.id, "receipt": {"sandbox_id": box.id}},
            )
            == "retire"
        )

    def test_exactly_one_replacement_is_created(self, host, sdk):
        _provision()
        FakeSandbox.live.clear()
        _provision()
        assert len(FakeSandbox.created) == 2
        FakeSandbox.live.clear()
        _provision()
        assert len(FakeSandbox.created) == 3
        assert _reservation(host)["generation"] == 3


# ---------------------------------------------------------------------------
# 11. Concurrency: the generation CAS stops a second machine
# ---------------------------------------------------------------------------


class TestConcurrency:
    def test_a_caller_that_lost_the_reservation_creates_nothing(self, host, sdk):
        with kb.connect_closing() as conn:
            kb.advance_run_sandbox(
                conn, host.task_id, run_id=host.run_id,
                transition="sandbox_reserved", expected_generation=0,
            )
        out = _provision()
        assert "error" in out
        assert "already being prepared" in out["error"]
        assert not FakeSandbox.created
        assert _reservation(host)["state"] == "reserved"

    def test_a_caller_that_loses_after_creating_discards_its_own_machine(
        self, host, sdk
    ):
        def _winner_settles_first(box):
            with kb.connect_closing() as conn:
                kb.advance_run_sandbox(
                    conn, host.task_id, run_id=host.run_id,
                    transition="sandbox_provisioned", expected_generation=1,
                    sandbox_id="sbx-winner",
                    receipt={"sandbox_id": "sbx-winner", "image_digest": IMAGE_DIGEST},
                )

        FakeSandbox.behavior = {"after_create": _winner_settles_first}
        out = _provision()
        assert "error" in out
        assert len(FakeSandbox.created) == 1
        assert FakeSandbox.created[0].killed is True
        record = _reservation(host)
        assert record["state"] == "active"
        assert record["sandbox_id"] == "sbx-winner"

    def test_a_lost_race_does_not_release_the_winners_reservation(self, host, sdk):
        def _winner_settles_first(box):
            with kb.connect_closing() as conn:
                kb.advance_run_sandbox(
                    conn, host.task_id, run_id=host.run_id,
                    transition="sandbox_provisioned", expected_generation=1,
                    sandbox_id="sbx-winner",
                    receipt={"sandbox_id": "sbx-winner", "image_digest": IMAGE_DIGEST},
                )

        FakeSandbox.behavior = {"after_create": _winner_settles_first}
        _provision()
        assert _event_kinds(host) == ["sandbox_reserved", "sandbox_provisioned"]


# ---------------------------------------------------------------------------
# 12. The real SDK surface this module depends on
# ---------------------------------------------------------------------------


class TestRealSdkSurface:
    """Guard against SDK drift: the fake above may only fake what exists."""

    def test_the_official_sync_sandbox_publishes_every_method_used(self):
        from opensandbox.sync.sandbox import SandboxSync

        for name in ("create", "connect", "get_info", "is_healthy", "kill", "close"):
            assert callable(getattr(SandboxSync, name)), name

    def test_the_official_signatures_accept_every_argument_passed(self):
        import inspect

        from opensandbox.sync.sandbox import SandboxSync

        create = inspect.signature(SandboxSync.create).parameters
        for name in ("image", "timeout", "ready_timeout", "resource", "env",
                     "metadata", "network_policy", "credential_proxy",
                     "connection_config"):
            assert name in create, name
        connect = inspect.signature(SandboxSync.connect).parameters
        for name in ("sandbox_id", "connection_config", "connect_timeout"):
            assert name in connect, name

    def test_the_official_connection_config_publishes_the_proxy_switch(self, sdk):
        config = sd._connection_config(
            sdk,
            sd._Connection(
                image_ref=IMAGE_REF, image_digest=IMAGE_DIGEST,
                domain="sandbox.internal:8080", protocol="https",
                api_key=CONNECTION_SECRET,
            ),
        )
        from opensandbox.config.connection_sync import ConnectionConfigSync

        assert isinstance(config, ConnectionConfigSync)
        assert "use_server_proxy" in ConnectionConfigSync.model_fields
        assert config.use_server_proxy is True
        assert config.disable_metrics is True
        assert config.get_base_url().startswith("https://sandbox.internal:8080/")

    def test_sandbox_info_publishes_every_field_the_liveness_check_reads(self):
        from opensandbox.models.sandboxes import SandboxInfo

        for field in ("id", "status", "expires_at", "image", "metadata"):
            assert field in SandboxInfo.model_fields, field

    def test_the_official_credential_binding_publishes_the_method_contour(self, sdk):
        """The verb narrowing must be a field the SDK really honours.

        If the official match model ever stopped publishing ``methods``, a
        silently dropped key would widen the binding back to every verb — so
        this asserts the real model carries it, not just that the value was
        passed.
        """
        _credentials, bindings = sd._vault_payload(sdk, REAL_SECRET)
        (binding,) = bindings
        assert "methods" in type(binding.match).model_fields
        assert binding.match.methods == list(sd.VAULT_MATCH_METHODS)


# ---------------------------------------------------------------------------
# 13. Plugin edge
# ---------------------------------------------------------------------------


def _manifest() -> dict:
    import yaml

    return yaml.safe_load(
        (Path(raphael_workspace.__file__).parent / "plugin.yaml").read_text(
            encoding="utf-8"
        )
    )


class TestPluginRegistration:
    def test_register_wires_the_tool_through_the_existing_plugin_edge(self, host):
        calls = []
        hooks = []

        class Ctx:
            def register_dashboard_auth_provider(self, provider):
                return None

            def register_cli_command(self, **kwargs):
                return None

            def register_tool(self, **kwargs):
                calls.append(kwargs)
                return None

            def register_hook(self, name, callback):
                hooks.append((name, callback))

        raphael_workspace.register(Ctx())
        (tool,) = [c for c in calls if c["name"] == sd.TOOL_NAME]
        assert tool["toolset"] == sd.TOOLSET
        assert tool["schema"] is sd.PROVISION_SCHEMA
        assert tool["handler"] is sd.handle_provision
        assert tool["check_fn"] is sd.check_provision_available
        assert not tool.get("override")
        assert hooks == [
            ("pre_runtime_turn", sd.enforce_sandbox_runtime),
            ("pre_tool_call", sd.maintain_sandbox_lease),
        ]

    def test_manifest_declares_a_bounded_plugin_dependency(self):
        manifest = _manifest()
        deps = manifest["python_dependencies"]
        (dep,) = [d for d in deps if d.startswith("opensandbox")]
        assert ">=" in dep and "<" in dep
        assert sd.TOOL_NAME in manifest["provides_tools"]

    def test_manifest_admits_exactly_the_proven_installed_sdk_as_its_floor(self):
        """The admitted floor is the SDK this wrapper was proved against.

        0.1.15 is what the host has installed and what the live run exercised,
        and it publishes every API used here. 0.1.16 is inside the same bound —
        so a host that already has it keeps working — but it is newer than the
        repository's intentional 14-day exclude-newer window, so it must not be
        required. The upper bound keeps the resolver off the 0.2 API break.
        """
        import importlib.metadata

        from packaging.specifiers import SpecifierSet
        from packaging.version import Version

        (dep,) = [
            d for d in _manifest()["python_dependencies"]
            if d.startswith("opensandbox")
        ]
        assert dep == "opensandbox>=0.1.15,<0.2"

        spec = SpecifierSet(dep[len("opensandbox"):])
        assert Version("0.1.15") in spec
        assert Version("0.1.16") in spec
        assert Version("0.1.14") not in spec
        assert Version("0.2.0") not in spec
        # The floor is not aspirational: the SDK actually imported by the
        # module under test satisfies it.
        assert Version(importlib.metadata.version("opensandbox")) in spec


def test_artifact_round_trip_moves_real_bytes_without_model_text(host, sdk):
    from plugins.dashboard_auth.raphael_workspace import sandbox_artifacts as sa

    _provision()
    box = FakeSandbox.created[-1]
    data = bytes(range(256)) * 137
    digest = hashlib.sha256(data).hexdigest()
    box.files.write_file("/workspace/task/result.patch", data)
    exported = json.loads(sa.handle_artifact({
        "direction": "export", "path": "/workspace/task/result.patch",
        "expected_sha256": digest,
    }))
    assert exported.get("sha256") == digest, exported
    assert exported["size"] == len(data)
    replay = json.loads(sa.handle_artifact({
        "direction": "export", "path": "/workspace/task/result.patch",
        "expected_sha256": digest,
    }))
    assert replay["attachment_id"] == exported["attachment_id"]
    with kb.connect_closing() as conn:
        stored = kb.get_attachment(conn, exported["attachment_id"])
        assert Path(stored.stored_path).read_bytes() == data
    imported = json.loads(sa.handle_artifact({
        "direction": "import", "attachment_id": exported["attachment_id"],
        "expected_sha256": digest,
    }))
    assert imported.get("sha256") == digest, imported
    assert box.files.read_bytes(imported["path"]) == data


def test_verification_role_gets_same_scoped_sandbox_without_coding_credentials(host, sdk, monkeypatch):
    monkeypatch.setenv("HERMES_PROFILE", "raphael-verifier")
    with kb.connect_closing() as conn:
        conn.execute("UPDATE tasks SET assignee=? WHERE id=?", ("raphael-verifier", host.task_id))
        conn.commit()
    out = _provision()
    assert "sandbox_id" in out, out
    box = FakeSandbox.created[-1]
    assert box.reported_metadata["hermes_profile"] == "raphael-verifier"
    assert box.create_kwargs["env"] == {}
    assert box.vault_calls == []


def test_active_sandbox_is_renewed_before_the_lease_expires(host, sdk):
    _provision()
    box = FakeSandbox.created[-1]
    box.expires_in_seconds = 120
    box.renewals = []
    box.renew = lambda timeout: box.renewals.append(timeout.total_seconds())
    box.credential_vault.patch = lambda **kwargs: None
    result = sd.maintain_sandbox_lease(
        tool_name="mcp_opensandbox_agent_factory_sandbox_command_run",
        args={"sandbox_id": box.id},
    )
    assert result is None, result
    assert box.renewals == [sd.SANDBOX_TIMEOUT_SECONDS]


@pytest.mark.parametrize("tool_name", [
    "mcp__opensandbox_agent_factory__sandbox_command_run",
    "mcp_opensandbox_agent_factory_sandbox_command_run",
    "mcp__sandbox__command_run",
    "mcp__sandbox__sandbox_command_run",
    "mcp_sandbox_command_run",
    "mcp__server2__file_read",
])
def test_generic_opensandbox_requires_this_runs_active_sandbox(host, tool_name):
    result = sd.maintain_sandbox_lease(
        tool_name=tool_name,
        args={"sandbox_id": "sbx-unrecorded"},
    )

    assert result and result["action"] == "block"
    assert FakeSandbox.connect_calls == []


@pytest.mark.parametrize("server_alias", ["opensandbox_agent_factory", "sandbox"])
@pytest.mark.parametrize("operation", ["sandbox_create", "sandbox_list"])
def test_generic_opensandbox_cannot_manage_sandboxes(host, server_alias, operation):
    result = sd.maintain_sandbox_lease(
        tool_name=f"mcp__{server_alias}__{operation}", args={},
    )

    assert result and result["action"] == "block"
    assert "raphael_sandbox_provision" in result["message"]
    assert FakeSandbox.connect_calls == []


def test_generic_opensandbox_fails_closed_for_delegated_raphael_worker(host):
    from agent.delegation_context import delegated_child_context

    with delegated_child_context("sandbox-boundary-negative"):
        result = sd.maintain_sandbox_lease(
            tool_name="mcp__opensandbox_agent_factory__sandbox_command_run",
            args={"sandbox_id": "sbx-unrecorded"},
        )

    assert result and result["action"] == "block"
    assert FakeSandbox.connect_calls == []


def test_generic_opensandbox_hook_leaves_non_raphael_profiles_unchanged(
    host, monkeypatch,
):
    monkeypatch.setenv("HERMES_PROFILE", "default")

    result = sd.maintain_sandbox_lease(
        tool_name="mcp__opensandbox_agent_factory__sandbox_create", args={},
    )

    assert result is None


def test_codex_app_server_runtime_is_blocked_before_tool_dispatch(host):
    from agent.conversation_loop import _runtime_turn_block_message

    result = sd.enforce_sandbox_runtime(api_mode="codex_app_server")
    message = _runtime_turn_block_message(
        SimpleNamespace(
            api_mode="codex_app_server",
            model="gpt-5.6-sol",
            session_id="runtime-boundary",
            platform="api_server",
        ),
        host.task_id,
    )

    assert result and result["action"] == "block"
    assert message == result["message"]
    with kb.connect_closing() as conn:
        task = kb.get_task(conn, host.task_id)
        assert task.status == "blocked"
        assert task.block_kind == "capability"


def test_default_runtime_remains_available_to_sandbox_profiles(host):
    assert sd.enforce_sandbox_runtime(api_mode="codex_responses") is None
    with kb.connect_closing() as conn:
        assert kb.get_task(conn, host.task_id).status == "running"


@pytest.mark.parametrize("bad_args", [
    {},
    {"sandbox_id": 7},
    {"sandbox": {"sandbox_id": "sbx-001"}},
    {"sandbox_id": "sbx-foreign"},
])
@pytest.mark.parametrize("tool_name", [
    "mcp__opensandbox_agent_factory__sandbox_command_run",
    "mcp__sandbox__command_run",
])
def test_generic_opensandbox_rejects_missing_malformed_and_foreign_ids_before_throttle(
    host, sdk, bad_args, tool_name,
):
    _provision()
    box = FakeSandbox.created[-1]
    own = sd.maintain_sandbox_lease(
        tool_name="mcp__opensandbox_agent_factory__sandbox_command_run",
        args={"sandbox_id": box.id},
    )
    assert own is None
    FakeSandbox.connect_calls.clear()

    result = sd.maintain_sandbox_lease(
        tool_name=tool_name,
        args=bad_args,
    )

    assert result and result["action"] == "block"
    assert FakeSandbox.connect_calls == []


@pytest.mark.parametrize("bad_input", [
    {"direction": "export", "path": "/etc/passwd", "expected_sha256": "0" * 64},
    {"direction": "export", "path": "/workspace/task/../secret", "expected_sha256": "0" * 64},
    {"direction": "import", "attachment_id": True, "expected_sha256": "0" * 64},
])
def test_artifact_refuses_unscoped_inputs(host, sdk, bad_input):
    from plugins.dashboard_auth.raphael_workspace import sandbox_artifacts as sa
    _provision()
    result = json.loads(sa.handle_artifact(bad_input))
    assert "error" in result
    with kb.connect_closing() as conn:
        assert kb.list_attachments(conn, host.task_id) == []


def test_artifact_refuses_wrong_checksum_and_stale_run(host, sdk):
    from plugins.dashboard_auth.raphael_workspace import sandbox_artifacts as sa
    _provision()
    box = FakeSandbox.created[-1]
    box.files.write_file("/workspace/task/result.patch", b"real bytes")
    args = {"direction": "export", "path": "/workspace/task/result.patch",
            "expected_sha256": "0" * 64}
    assert "error" in json.loads(sa.handle_artifact(args))
    with kb.connect_closing() as conn:
        assert kb.list_attachments(conn, host.task_id) == []
        conn.execute("UPDATE tasks SET status='blocked' WHERE id=?", (host.task_id,))
        conn.commit()
    args["expected_sha256"] = hashlib.sha256(b"real bytes").hexdigest()
    assert "error" in json.loads(sa.handle_artifact(args))


def test_artifact_refuses_another_tasks_unrelated_input(host, sdk):
    from plugins.dashboard_auth.raphael_workspace import sandbox_artifacts as sa
    _provision()
    data = b"other task data"
    with kb.connect_closing() as conn:
        foreign = kb.create_task(conn, title="Unrelated work", assignee="raphael-claude-worker")
        aid = kb.store_attachment_bytes(conn, foreign, "foreign.patch", data, uploaded_by="agent")
    result = json.loads(sa.handle_artifact({
        "direction": "import", "attachment_id": aid,
        "expected_sha256": hashlib.sha256(data).hexdigest(),
    }))
    assert "error" in result
    assert not any(path.startswith("/workspace/inputs/") for path in FakeSandbox.created[-1].file_data)


def test_worker_link_added_after_claim_cannot_authorize_same_project_artifact(
    host, monkeypatch,
):
    from plugins.dashboard_auth.raphael_workspace import sandbox_artifacts as sa

    _, _, _, current, _ = _replacement_artifact_context(
        host, monkeypatch, "raphael-builder", "R17",
    )
    data = b"unrelated same-project private input"
    with kb.connect_closing() as conn:
        unrelated = kb.create_task(
            conn,
            title="Unrelated completed work",
            project_id=current.project_id,
            project_source_task_id=current.id,
            initial_status="blocked",
        )
        aid = kb.store_attachment_bytes(
            conn, unrelated, "private.txt", data, uploaded_by="operator",
        )
        assert kb.complete_task(
            conn, unrelated, summary="Done", fire_lifecycle_hook=False,
        )
        kb.link_tasks(conn, unrelated, current.id)
        assert unrelated in kb.parent_ids(conn, current.id)

    result = json.loads(sa.handle_artifact({
        "direction": "inspect", "attachment_id": aid,
    }))

    assert "error" in result
    assert data.decode() not in json.dumps(result)


@pytest.mark.parametrize("direction", ["inspect", "copy"])
def test_preclaim_dependency_snapshot_preserves_authorized_artifact_access(
    host, monkeypatch, direction,
):
    from hermes_cli import projects_db
    from plugins.dashboard_auth.raphael_workspace import sandbox_artifacts as sa

    with projects_db.connect_closing() as conn:
        project_id = projects_db.create_project(
            conn, name="Frozen artifact inputs", primary_path=str(host.repo),
        )
    data = b"approved dependency input"
    with kb.connect_closing() as conn:
        parent = kb.create_task(
            conn, title="Completed dependency", project_id=project_id,
            initial_status="blocked",
        )
        aid = kb.store_attachment_bytes(
            conn, parent, "input.txt", data, uploaded_by="operator",
        )
        assert kb.complete_task(
            conn, parent, summary="Done", fire_lifecycle_hook=False,
        )
        child = kb.create_task(
            conn,
            title="Consume approved input",
            assignee=sd.WORKER_PROFILE,
            project_id=project_id,
            parents=[parent],
            owned_paths=[],
        )
        current = kb.claim_task(conn, child)
        assert current is not None

    monkeypatch.setenv("HERMES_KANBAN_TASK", current.id)
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(current.current_run_id))
    monkeypatch.setenv("HERMES_KANBAN_WORKSPACE", current.workspace_path)
    args = {"direction": direction, "attachment_id": aid}
    if direction == "copy":
        args["expected_sha256"] = hashlib.sha256(data).hexdigest()

    result = json.loads(sa.handle_artifact(args))

    assert "error" not in result, result
    assert result["source_task_id"] == parent
    assert result["sha256"] == hashlib.sha256(data).hexdigest()


# ---------------------------------------------------------------------------
# Remote symlink containment — get_file_info lstats and never follows, while
# read_bytes_stream/write_file both follow every symlink in a path, including
# an intermediate ancestor. A lexical-only check (absolute, no "..", inside
# the root) cannot see that, so every ancestor must be proven a real
# directory, and the leaf proven the right thing, with metadata before any
# byte moves.
# ---------------------------------------------------------------------------


class TestArtifactRemoteSymlinkContainment:
    def test_export_refuses_a_leaf_symlink_even_when_the_target_holds_the_expected_bytes(
        self, host, sdk
    ):
        from plugins.dashboard_auth.raphael_workspace import sandbox_artifacts as sa

        _provision()
        box = FakeSandbox.created[-1]
        secret = b"outside-the-workspace secret bytes"
        box.file_data["/outside/secret.txt"] = secret
        box.symlinks["/workspace/task/result.patch"] = "/outside/secret.txt"
        result = json.loads(sa.handle_artifact({
            "direction": "export", "path": "/workspace/task/result.patch",
            "expected_sha256": hashlib.sha256(secret).hexdigest(),
        }))
        assert "error" in result
        assert box.file_data["/outside/secret.txt"] == secret
        with kb.connect_closing() as conn:
            assert kb.list_attachments(conn, host.task_id) == []

    def test_export_refuses_when_an_intermediate_directory_is_a_symlink(self, host, sdk):
        from plugins.dashboard_auth.raphael_workspace import sandbox_artifacts as sa

        _provision()
        box = FakeSandbox.created[-1]
        secret = b"reached only by walking through the redirected directory"
        box.file_data["/outside/dir/result.patch"] = secret
        box.symlinks["/workspace/task/sub"] = "/outside/dir"
        result = json.loads(sa.handle_artifact({
            "direction": "export", "path": "/workspace/task/sub/result.patch",
            "expected_sha256": hashlib.sha256(secret).hexdigest(),
        }))
        assert "error" in result
        assert box.file_data["/outside/dir/result.patch"] == secret
        with kb.connect_closing() as conn:
            assert kb.list_attachments(conn, host.task_id) == []

    def test_export_still_succeeds_for_an_ordinary_regular_file(self, host, sdk):
        from plugins.dashboard_auth.raphael_workspace import sandbox_artifacts as sa

        _provision()
        box = FakeSandbox.created[-1]
        data = b"an ordinary file with no symlinks anywhere in its path"
        box.files.write_file("/workspace/task/result.patch", data)
        result = json.loads(sa.handle_artifact({
            "direction": "export", "path": "/workspace/task/result.patch",
            "expected_sha256": hashlib.sha256(data).hexdigest(),
        }))
        assert "error" not in result, result
        with kb.connect_closing() as conn:
            stored = kb.get_attachment(conn, result["attachment_id"])
            assert kb.read_attachment_bytes(stored) == data

    def test_import_refuses_a_pre_existing_symlink_at_the_destination(self, host, sdk):
        from plugins.dashboard_auth.raphael_workspace import sandbox_artifacts as sa

        _provision()
        box = FakeSandbox.created[-1]
        own_data = b"bytes this task is entitled to move to the sandbox"
        with kb.connect_closing() as conn:
            aid = kb.store_attachment_bytes(
                conn, host.task_id, "own.bin", own_data, uploaded_by="agent",
            )
        untouched = b"whatever this external file already held"
        box.file_data["/outside/target.bin"] = untouched
        box.symlinks[f"/workspace/inputs/{aid}/own.bin"] = "/outside/target.bin"
        result = json.loads(sa.handle_artifact({
            "direction": "import", "attachment_id": aid,
            "expected_sha256": hashlib.sha256(own_data).hexdigest(),
        }))
        assert "error" in result
        assert box.file_data["/outside/target.bin"] == untouched
        assert box.symlinks[f"/workspace/inputs/{aid}/own.bin"] == "/outside/target.bin"

    def test_import_refuses_when_a_destination_ancestor_is_a_symlink(self, host, sdk):
        from plugins.dashboard_auth.raphael_workspace import sandbox_artifacts as sa

        _provision()
        box = FakeSandbox.created[-1]
        own_data = b"bytes redirected by a symlinked inputs root"
        with kb.connect_closing() as conn:
            aid = kb.store_attachment_bytes(
                conn, host.task_id, "own.bin", own_data, uploaded_by="agent",
            )
        box.symlinks["/workspace/inputs"] = "/outside/inputs"
        result = json.loads(sa.handle_artifact({
            "direction": "import", "attachment_id": aid,
            "expected_sha256": hashlib.sha256(own_data).hexdigest(),
        }))
        assert "error" in result
        assert not any(path.startswith("/outside/inputs") for path in box.file_data)

    def test_import_still_succeeds_on_a_clean_destination(self, host, sdk):
        from plugins.dashboard_auth.raphael_workspace import sandbox_artifacts as sa

        _provision()
        box = FakeSandbox.created[-1]
        own_data = b"bytes with nothing pre-planted at the destination"
        with kb.connect_closing() as conn:
            aid = kb.store_attachment_bytes(
                conn, host.task_id, "own.bin", own_data, uploaded_by="agent",
            )
        result = json.loads(sa.handle_artifact({
            "direction": "import", "attachment_id": aid,
            "expected_sha256": hashlib.sha256(own_data).hexdigest(),
        }))
        assert "error" not in result, result
        assert box.file_data[result["path"]] == own_data

    def test_source_staging_write_refuses_a_pre_existing_symlink_at_the_archive_path(
        self, host, sdk
    ):
        untouched = b"whatever already lived at the redirected archive target"

        def _plant_symlink(box):
            box.file_data["/outside/evil.tar"] = untouched
            box.symlinks[sd.SANDBOX_ARCHIVE_PATH] = "/outside/evil.tar"

        FakeSandbox.behavior = {"after_create": _plant_symlink}
        out = _provision()
        assert "error" in out
        box = FakeSandbox.created[0]
        assert box.file_data["/outside/evil.tar"] == untouched
        assert box.killed is True


def test_inspect_native_artifact_without_a_sandbox(host):
    from plugins.dashboard_auth.raphael_workspace import sandbox_artifacts as sa

    data = json.dumps({"patch_sha256": "a" * 64}).encode("utf-8")
    with kb.connect_closing() as conn:
        aid = kb.store_attachment_bytes(
            conn, host.task_id, "manifest.json", data,
            content_type="application/json", uploaded_by="operator",
        )

    result = json.loads(sa.handle_artifact({
        "direction": "inspect", "attachment_id": aid,
    }))

    assert "error" not in result, result
    assert result["sha256"] == hashlib.sha256(data).hexdigest()
    assert result["content_text"] == data.decode("utf-8")
    assert result["source_uploaded_by"] == "operator"
    assert _reservation(host)["state"] == "absent"

def test_copy_native_artifact_preserves_bytes_and_retry_identity(host):
    from plugins.dashboard_auth.raphael_workspace import sandbox_artifacts as sa

    data = bytes(range(256)) * 8192
    with kb.connect_closing() as conn:
        aid = kb.store_attachment_bytes(
            conn, host.task_id, "recovered.patch", data,
            content_type="application/octet-stream", uploaded_by="operator",
        )
    args = {"direction": "copy", "attachment_id": aid,
            "expected_sha256": hashlib.sha256(data).hexdigest()}

    result = json.loads(sa.handle_artifact(args))

    assert "error" not in result, result
    assert result["source_attachment_id"] == aid
    assert result["source_uploaded_by"] == "operator"
    assert result["attachment_id"] != aid
    with kb.connect_closing() as conn:
        copied = kb.get_attachment(conn, result["attachment_id"])
        assert copied.uploaded_by == "agent"
        assert kb.read_attachment_bytes(copied) == data
    assert json.loads(sa.handle_artifact(args))["attachment_id"] == result["attachment_id"]
    assert _reservation(host)["state"] == "absent"

def _replacement_artifact_context(host, monkeypatch, profile, responsibility):
    from hermes_cli import projects_db
    from plugins.dashboard_auth.raphael_workspace.model_policy import task_assignment_for

    with projects_db.connect_closing() as conn:
        project_id = projects_db.create_project(
            conn, name="Artifact recovery", primary_path=str(host.repo),
        )
    provider = "openai-codex" if profile == "raphael-verifier" else "anthropic"
    route = task_assignment_for(profile, provider, "deep")
    spec = {
        "title": "Continue the saved artifact",
        "body": "Reuse the exact saved input without reauthoring it.",
        "assignee": profile, "responsibility": responsibility,
        "execution_tier": "deep", "model_override": route.model,
        "provider_override": route.provider, "reasoning_effort": route.reasoning_effort,
        "model_policy_lock": kb.mint_policy_lock(
            profile, route.provider, route.model, route.reasoning_effort, "deep",
        ),
        "owned_paths": ["."],
    }
    data = json.dumps({"original_input": "preserved exactly"}).encode()
    with kb.connect_closing() as conn:
        kb.block_task(conn, host.task_id, reason="Prepare the isolated replacement fixture.")
        anchor = kb.create_task(conn, title="Fixture anchor", project_id=project_id, control=True)
        source = kb.create_task(
            conn, title="Stopped work", assignee=sd.WORKER_PROFILE,
            project_id=project_id, initial_status="blocked",
        )
        aid = kb.store_attachment_bytes(
            conn, source, "saved.json", data,
            content_type="application/json", uploaded_by="operator",
        )
        plan = kb.apply_owner_project_plan(
            conn, project_id=project_id, anchor_task_id=anchor,
            changes=[{
                "action": "replace", "reason": "Continue the saved work.",
                "target": {"task_id": source, "expected_status": "blocked",
                           "expected_revision": kb.task_event_revision(conn, source)},
                "replacement": spec,
            }],
            actor="default", profile="default", idempotency_key="artifact-replacement",
            request_digest=hashlib.sha256(b"artifact replacement").hexdigest(),
            trigger="Continue", plan_summary="Recover the saved input.",
            current_milestone="Verify artifact recovery", later_milestones=[], parked=False,
        )
        assert plan["applied"], plan
        current = kb.claim_task(conn, plan["created_task_ids"][0])
        assert current is not None
    monkeypatch.setenv("HERMES_PROFILE", profile)
    monkeypatch.setenv("HERMES_KANBAN_TASK", current.id)
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(current.current_run_id))
    monkeypatch.setenv("HERMES_KANBAN_WORKSPACE", current.workspace_path)
    return source, aid, data, current, anchor


@pytest.mark.parametrize(("profile", "responsibility"), [
    ("raphael-builder", "R17"), ("raphael-verifier", "R13"),
])
def test_replacement_worker_recovers_native_predecessor_artifact(
    host, monkeypatch, profile, responsibility,
):
    from plugins.dashboard_auth.raphael_workspace import sandbox_artifacts as sa

    source, aid, data, current, _ = _replacement_artifact_context(
        host, monkeypatch, profile, responsibility,
    )
    from model_tools import handle_function_call
    from tools.registry import invalidate_check_fn_cache, registry

    invalidate_check_fn_cache()
    definitions = registry.get_definitions({sa.TOOL_NAME}, quiet=True)
    enabled = [definition["function"]["name"] for definition in definitions]
    assert sa.TOOL_NAME in enabled
    inspected = json.loads(handle_function_call(
        sa.TOOL_NAME, {"direction": "inspect", "attachment_id": aid},
        task_id=current.id, enabled_tools=enabled, enabled_toolsets=[sd.TOOLSET],
    ))
    assert "error" not in inspected, inspected
    copied = json.loads(handle_function_call(
        sa.TOOL_NAME,
        {"direction": "copy", "attachment_id": aid, "expected_sha256": inspected["sha256"]},
        task_id=current.id, enabled_tools=enabled, enabled_toolsets=[sd.TOOLSET],
    ))
    assert "error" not in copied, copied
    assert copied["source_task_id"] == source
    with kb.connect_closing() as conn:
        assert kb.get_task(conn, source).status == "archived"
        attachment = kb.get_attachment(conn, copied["attachment_id"])
        assert attachment.task_id == current.id
        assert kb.read_attachment_bytes(attachment) == data

@pytest.mark.parametrize("direction", ["inspect", "copy"])
def test_native_artifact_recovery_does_not_require_a_live_sandbox(host, sdk, direction):
    from plugins.dashboard_auth.raphael_workspace import sandbox_artifacts as sa

    _provision()
    FakeSandbox.created[-1].expires_in_seconds = -1
    data = b"saved before expiry"
    with kb.connect_closing() as conn:
        aid = kb.store_attachment_bytes(
            conn, host.task_id, "saved.txt", data, content_type="text/plain",
            uploaded_by="operator",
        )
    args = {"direction": direction, "attachment_id": aid}
    if direction == "copy":
        args["expected_sha256"] = hashlib.sha256(data).hexdigest()

    gate = sd.maintain_sandbox_lease(tool_name="raphael_sandbox_artifact", args=args)

    assert gate is None, gate
    result = json.loads(sa.handle_artifact(args))
    assert "error" not in result, result
    assert result["sha256"] == hashlib.sha256(data).hexdigest()

@pytest.mark.parametrize("direction", ["inspect", "copy"])
def test_native_artifact_rejects_cancelled_source_and_forged_comment(host, monkeypatch, direction):
    from plugins.dashboard_auth.raphael_workspace import sandbox_artifacts as sa

    _, _, _, current, anchor = _replacement_artifact_context(
        host, monkeypatch, "raphael-builder", "R17",
    )
    data = b"unrelated private input"
    with kb.connect_closing() as conn:
        other = kb.create_task(
            conn, title="Cancelled unrelated work", project_id=current.project_id,
            assignee=sd.WORKER_PROFILE, initial_status="blocked",
        )
        aid = kb.store_attachment_bytes(conn, other, "private.txt", data, uploaded_by="operator")
        kb.add_comment(conn, other, "operator", json.dumps({
            "kind": "owner_project_plan_change", "action": "replace",
            "replacement_task_id": current.id,
        }))
        result = kb.apply_owner_project_plan(
            conn, project_id=current.project_id, anchor_task_id=anchor,
            changes=[{"action": "cancel", "reason": "Cancel unrelated work.",
                      "target": {"task_id": other, "expected_status": "blocked",
                                 "expected_revision": kb.task_event_revision(conn, other)}}],
            actor="default", profile="default", idempotency_key="cancel-unrelated-input",
            request_digest=hashlib.sha256(b"cancel unrelated input").hexdigest(),
            trigger="Cancel", plan_summary="Cancel only this work.",
            current_milestone="Verify input boundaries", later_milestones=[], parked=False,
        )
        assert result["applied"]
    args = {"direction": direction, "attachment_id": aid}
    if direction == "copy":
        args["expected_sha256"] = hashlib.sha256(data).hexdigest()
    result = json.loads(sa.handle_artifact(args))
    assert "error" in result
    assert data.decode() not in json.dumps(result)
    with kb.connect_closing() as conn:
        assert kb.get_task(conn, other).status == "archived"
        assert kb.list_attachments(conn, current.id) == []


@pytest.mark.parametrize("direction", ["inspect", "copy"])
def test_native_artifact_rejects_other_project_and_stale_run(host, monkeypatch, direction):
    from hermes_cli import projects_db
    from plugins.dashboard_auth.raphael_workspace import sandbox_artifacts as sa

    _, own_id, own_data, current, _ = _replacement_artifact_context(
        host, monkeypatch, "raphael-verifier", "R13",
    )
    with projects_db.connect_closing() as conn:
        foreign_project = projects_db.create_project(conn, name="Other project")
    with kb.connect_closing() as conn:
        other = kb.create_task(conn, title="Other project's work", project_id=foreign_project)
        aid = kb.store_attachment_bytes(conn, other, "foreign.txt", b"foreign data", uploaded_by="operator")
        assert kb.complete_task(conn, other, summary="Done", fire_lifecycle_hook=False)
    args = {"direction": direction, "attachment_id": aid}
    if direction == "copy":
        args["expected_sha256"] = hashlib.sha256(b"foreign data").hexdigest()
    assert "error" in json.loads(sa.handle_artifact(args))
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(current.current_run_id + 1))
    args["attachment_id"] = own_id
    if direction == "copy":
        args["expected_sha256"] = hashlib.sha256(own_data).hexdigest()
    assert "error" in json.loads(sa.handle_artifact(args))
    with kb.connect_closing() as conn:
        assert kb.list_attachments(conn, current.id) == []


@pytest.mark.parametrize("direction", ["inspect", "copy"])
@pytest.mark.parametrize("caller", ["coordinator", "delegated", "scheduled"])
def test_native_artifact_rejects_non_worker_callers(host, monkeypatch, direction, caller):
    from contextlib import nullcontext
    from agent.delegation_context import delegated_child_context, non_dispatcher_owned_context
    from plugins.dashboard_auth.raphael_workspace import sandbox_artifacts as sa

    with kb.connect_closing() as conn:
        aid = kb.store_attachment_bytes(conn, host.task_id, "input.txt", b"private input", uploaded_by="operator")
    args = {"direction": direction, "attachment_id": aid}
    if direction == "copy":
        args["expected_sha256"] = hashlib.sha256(b"private input").hexdigest()
    if caller == "coordinator":
        monkeypatch.setenv("HERMES_PROFILE", "default")
        scope = nullcontext()
    elif caller == "delegated":
        scope = delegated_child_context("artifact-negative")
    else:
        scope = non_dispatcher_owned_context()
    with scope:
        result = json.loads(sa.handle_artifact(args))
    assert "error" in result
    assert "private input" not in json.dumps(result)
    with kb.connect_closing() as conn:
        assert [a.id for a in kb.list_attachments(conn, host.task_id)] == [aid]


def test_native_copy_rejects_checksum_mismatch_without_writing(host):
    from plugins.dashboard_auth.raphael_workspace import sandbox_artifacts as sa

    with kb.connect_closing() as conn:
        aid = kb.store_attachment_bytes(conn, host.task_id, "input.txt", b"exact", uploaded_by="operator")
    result = json.loads(sa.handle_artifact({
        "direction": "copy", "attachment_id": aid, "expected_sha256": "0" * 64,
    }))
    assert "error" in result
    with kb.connect_closing() as conn:
        assert [a.id for a in kb.list_attachments(conn, host.task_id)] == [aid]


@pytest.mark.parametrize(("data", "media_type"), [
    (b"x" * 8193, "text/plain"), (b"binary", "application/octet-stream"),
])
def test_native_inspection_never_returns_partial_or_binary_text(host, data, media_type):
    from plugins.dashboard_auth.raphael_workspace import sandbox_artifacts as sa

    with kb.connect_closing() as conn:
        aid = kb.store_attachment_bytes(
            conn, host.task_id, "input.dat", data, content_type=media_type, uploaded_by="operator",
        )
    result = json.loads(sa.handle_artifact({"direction": "inspect", "attachment_id": aid}))
    assert "error" not in result
    assert result["content_text_included"] is False
    assert "content_text" not in result
    assert result["sha256"] == hashlib.sha256(data).hexdigest()


def test_active_run_retry_reuses_a_collision_renamed_export(host):
    with kb.connect_closing() as conn:
        kb.store_attachment_bytes(conn, host.task_id, "result.patch", b"old", uploaded_by="operator")
        first = kb.store_attachment_bytes(
            conn, host.task_id, "result.patch", b"new", uploaded_by="agent", expected_run_id=host.run_id,
        )
        again = kb.store_attachment_bytes(
            conn, host.task_id, "result.patch", b"new", uploaded_by="agent", expected_run_id=host.run_id,
        )
        assert first == again
        assert len(kb.list_attachments(conn, host.task_id)) == 2


def test_copied_operator_patch_can_complete_the_native_worktree_handoff(host):
    from plugins.dashboard_auth.raphael_workspace import sandbox_artifacts as sa

    patch = b"--- a/app.py\n+++ b/app.py\n@@ -1 +1 @@\n-print('hi')\n+print('recovered')\n"
    with kb.connect_closing() as conn:
        kb.set_branch_name(conn, host.task_id, _git(host.worktree, "branch", "--show-current"))
        kb.record_worktree_base(conn, host.task_id, host.worktree)
        aid = kb.store_attachment_bytes(conn, host.task_id, "recovered.patch", patch, uploaded_by="operator")
    copied = json.loads(sa.handle_artifact({
        "direction": "copy", "attachment_id": aid, "expected_sha256": hashlib.sha256(patch).hexdigest(),
    }))
    assert "error" not in copied, copied
    with kb.connect_closing() as conn:
        assert kb.complete_task(
            conn, host.task_id, summary="Materialized the verified recovered input.",
            patch_attachment_id=copied["attachment_id"], expected_run_id=host.run_id,
            fire_lifecycle_hook=False,
        )
        task = kb.get_task(conn, host.task_id)
        assert task.status == "done"
        assert task.head_commit != task.base_commit
    assert _git(host.repo, "show", f"{task.head_commit}:app.py") == "print('recovered')"
