"""Trusted Server 2 sandbox provisioning for scoped Raphael workers.

Why this exists
---------------
Server 1 is the control plane and secret boundary; Server 2 is the only place
specialist coding, building, testing, or browsing may happen. The maintained
OpenSandbox MCP server gives a worker lifecycle/command/file tools, but its
generic ``sandbox_create`` accepts a mutable image and can seed neither
Credential Vault nor an exact task source — so a Claude worker that used it got
a sandbox with no usable Anthropic credential and no source tree, and had no
choice left but to ask a non-technical owner for an API key.

This module adds the one missing capability as a single, argument-free worker
tool: *create one sandbox for my current task and seed it*. Everything the
generic path let the model choose — source, image, endpoint, resources, egress
policy, credentials — is resolved here from host-owned state and frozen
constants. Ordinary command, file, info, health, and endpoint work stays on the
maintained official MCP tools. Generic create/list/renew/kill management is
blocked: this plugin owns creation, fixed-duration renewal, and cleanup.

The coding profile receives a scoped credential-vault binding. Builder and
verification profiles receive no coding credentials. Reference-only inspection
and copying of saved artifacts live in the sibling artifact module and do not
require a live execution machine.

Frozen policy contract
----------------------
The egress allowlist, resource limits, task timeout, in-sandbox workspace and
diff baseline, Credential Vault names and route contour, and the placeholder
token below are constants, not operator-editable configuration —
mirroring the fixed grants in :mod:`token_store`. Only two facts are
operator-owned, and neither is reachable from a model argument: the
digest-pinned worker image and the already-configured OpenSandbox connection
(``config.yaml`` for the non-secret settings; ``.env`` for the API key,
falling back to a trusted operator credential file whose path is itself
``config.yaml``).

A diff aid, not an authority
----------------------------
The same tracked archive is extracted twice, into two fixed paths: the
editable workspace the worker builds in, and a baseline whose write bits are
then removed. The worker can therefore generate an exact recursive patch of
its own change with no git metadata inside the sandbox and nothing for a
person to copy in or out.

Removing the write bits is a convenience, not a boundary, and the baseline is
not immutable. A live Server 2 probe against the pinned image showed the
maintained MCP ``command_run`` running as uid 0 — its published schema has no
uid or gid parameter — and uid 0 rewrote a baseline whose write bits had
already been removed. What the ``chmod`` still buys is real but narrow: an
accidental non-root write cannot silently corrupt the diff.

So acceptance never rests on either tree here. The independent Reviewer
reapplies the patch this worker produced to the exact source commit named in
the receipt and verifies *that* candidate; it never treats the mutable sandbox
workspace, or the baseline beside it, as acceptance evidence.

One authority for "does this run already have a machine"
--------------------------------------------------------
The reservation is *not* kept here. ``kanban_db`` owns it as append-only
``task_events`` transitions folded per run
(:func:`hermes_cli.kanban_db.read_run_sandbox` /
:func:`~hermes_cli.kanban_db.advance_run_sandbox`), with an explicit
generation compare-and-swap. This module only reads that record, advances it,
and emits ordinary sanitized log lines — it deliberately keeps no second state
store, so there is exactly one answer to "is a machine already live for this
run" and it is the one an operator already reads in the task history.

The same transaction also mirrors the current created or provisioned generation
into ``run_sandbox_cleanup_intents``. That table grants no machine authority:
it is a bounded durable scheduler for ended-run cleanup, deleted by the exact
``sandbox_released`` transition. Exhausted automatic cleanup remains visible
and deletion-protecting until an operator resolves it. The event fold remains
the liveness proof. A physical allocation that loses the post-create CAS is
not another candidate authority; its ID is retained separately in
``run_sandbox_orphan_cleanup_intents`` only so bounded cleanup can kill that
loser without replacing or releasing the canonical winner.

Fail-closed everywhere
----------------------
Wrong profile, missing/foreign Kanban run, scratch or ambiguous or unscoped
source, a dirty tree, a mutable image tag, an unconfigured connection, a
missing host credential, a Claude account token that would expire mid-task and
cannot be refreshed, or a missing SDK all refuse before anything is created. A
failure after creation records the remote ID before later setup, then either
confirms cleanup or retains durable cleanup authority. Host temporary material
is removed and the caller receives one actionable sanitized error. Exception
bodies are never logged or returned — they can carry protected data.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import shutil
import stat
import subprocess
import sys
import time
import threading
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

from hermes_constants import get_default_hermes_root, secure_parent_dir

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Frozen policy contract — see the module docstring. Not configuration.
# ---------------------------------------------------------------------------

TOOL_NAME = "raphael_sandbox_provision"
TOOLSET = "raphael_sandbox"
WORKER_PROFILE = "raphael-claude-worker"
SANDBOX_PROFILES = frozenset({
    WORKER_PROFILE, "raphael-builder", "raphael-verifier", "raphael-planner",
})
#: Profiles whose runs only ever read the exact source: no patch authority,
#: no coding credential. The planner joins the verifier here so code
#: evidence for planning work is gathered on Server 2, never on the host.
READ_ONLY_SOURCE_PROFILES = frozenset({"raphael-verifier", "raphael-planner"})

#: Where the task source is extracted inside the sandbox. Fixed so no caller
#: — model or operator — picks a path on Server 2.
SANDBOX_WORKSPACE = "/workspace/task"
#: A second extraction of the *same* archive, beside the editable one, with
#: its write bits removed afterwards. It exists so the worker can generate an
#: exact recursive patch of what it changed — ``diff -ru`` between these two
#: fixed paths — without git metadata inside the sandbox and without a person
#: copying a tree in or out. It is a diff aid only, and not immutable: the
#: worker's own MCP ``command_run`` on this pinned image runs as uid 0 and can
#: rewrite it, so the Reviewer verifies the produced patch against the source
#: commit rather than this tree. Fixed for the same reason as the workspace.
SANDBOX_BASELINE = "/workspace/baseline"
#: Staging path for the transferred archive; removed by the extraction step.
SANDBOX_ARCHIVE_PATH = "/tmp/hermes-task-source.tar"

SANDBOX_RESOURCES = {"cpu": "2", "memory": "4Gi"}
SANDBOX_TIMEOUT_SECONDS = 3600
SANDBOX_READY_TIMEOUT_SECONDS = 180
SANDBOX_REQUEST_TIMEOUT_SECONDS = 60
#: How long to wait when re-attaching to a recorded machine to prove it is
#: still alive. Short: an unreachable machine must be replaced, not waited on.
SANDBOX_VERIFY_TIMEOUT_SECONDS = 30
SANDBOX_CLEANUP_REQUEST_TIMEOUT_SECONDS = 3
SANDBOX_CLEANUP_CONNECT_TIMEOUT_SECONDS = 3
#: A recorded machine with less than this left on its own server-side lease is
#: treated as expired and replaced, rather than handed back for a task that
#: would then lose it mid-build.
SANDBOX_MIN_REMAINING_SECONDS = 300

#: Default-deny egress with the minimum Claude + package endpoints a Claude
#: Code worker needs to authenticate, install itself, and install task deps.
EGRESS_ALLOWLIST = (
    "api.anthropic.com",
    "registry.npmjs.org",
    "pypi.org",
    "files.pythonhosted.org",
)

VAULT_CREDENTIAL_NAME = "anthropic"
VAULT_BINDING_NAME = "anthropic-v1"
VAULT_MATCH_HOST = "api.anthropic.com"
VAULT_MATCH_PATH = "/v1/*"
#: The only two verbs a coding worker's Anthropic traffic needs: POST to
#: create a message, GET to read model/usage metadata. Naming them keeps the
#: binding from lending the host credential to a mutating verb (DELETE, PUT,
#: PATCH) that no task requires, so an unexpected request is refused by the
#: contour rather than signed by it.
VAULT_MATCH_METHODS = ("GET", "POST")

#: The only Anthropic value that ever enters the sandbox. Shaped like a real
#: key so Claude Code's client-side format check passes; the egress sidecar
#: substitutes the vault credential on the way out.
PLACEHOLDER_TOKEN = "sk-ant-api03-hermes-credential-proxy-placeholder-not-a-secret"

#: A Claude *account* access token is refreshable but short-lived, so one that
#: is valid now can still die inside a bounded task. Require it to outlive the
#: whole sandbox lease plus this buffer, or refresh it before the vault write.
CREDENTIAL_SAFETY_BUFFER_SECONDS = 900

#: How many times the reservation loop may re-read after losing a
#: compare-and-swap before refusing. Bounded so a pathological contender can
#: never spin, and generous enough for the one legitimate two-step path
#: (retire an unverifiable generation, then open the next).
MAX_RESERVATION_ATTEMPTS = 4

_IMAGE_DIGEST_RE = re.compile(
    r"\A(?P<repo>[A-Za-z0-9][A-Za-z0-9._\-/:]*)@(?P<digest>sha256:[0-9a-f]{64})\Z"
)
_COMMIT_RE = re.compile(r"\A[0-9a-f]{40}\Z")
_GIT_TIMEOUT_SECONDS = 120

#: The only assignment form honoured in an operator credential file, and the
#: only value shape accepted from it. Deliberately literal: no interpolation,
#: no command substitution, no shell.
_SECRET_FILE_ASSIGNMENT_RE = re.compile(
    r"\A(?:export[ \t]+)?OPEN_SANDBOX_API_KEY[ \t]*=[ \t]*(?P<value>.*?)[ \t]*\Z"
)
_SECRET_VALUE_RE = re.compile(r"\A[\x21-\x7e]{1,512}\Z")
_MAX_SECRET_FILE_BYTES = 64 * 1024
#: One refusal for every unusable-credential-file case. It never names the
#: path or the value, because both are protected host material.
_SECRET_FILE_REFUSAL = (
    "this host's configured build-service credential file is unusable, so no "
    "machine can be reached. Ask the operator to check it — never ask a "
    "person to paste a key."
)


PROVISION_SCHEMA = {
    "name": TOOL_NAME,
    "description": (
        "Provision the ready remote build machine for the task you are "
        "already assigned, and return a receipt proving what was provisioned. "
        "Takes no arguments: the source tree, machine specification, outbound "
        "policy, and sign-in material are all resolved by Hermes from trusted "
        "state you neither see nor choose. Call this once before doing any "
        "build, test, or browser work, then drive the machine with the "
        "maintained OpenSandbox tools. The receipt also names a fixed second "
        "copy of the starting tree beside your editable one, with its write "
        "bits removed, so you can produce an exact recursive patch of your "
        "own work for review without version-control metadata. Treat that "
        "copy as a diff aid only, never as protected or authoritative: you "
        "run as root on the machine and can still overwrite it, and the "
        "reviewer accepts your work by reapplying your patch to the revision "
        "named in this receipt, not by reading either tree on the machine. "
        "An empty ownership_scope permits observation and temporary test "
        "scratch inside this disposable machine but authorizes no source "
        "patch import back to the Project. Do not create, list, or kill sandboxes with "
        "generic management tools: Hermes destroys this exact machine "
        "automatically when the run ends. "
        "Never ask a person for sign-in material — if this refuses, report "
        "its message verbatim to the operator and stop."
    ),
    "parameters": {
        "type": "object",
        "properties": {},
        "required": [],
        "additionalProperties": False,
    },
}


class SandboxDispatchError(Exception):
    """One safe, actionable, already-sanitized refusal reason."""

    def __init__(self, message: str, *, code: str = "refused") -> None:
        super().__init__(message)
        self.message = message
        self.code = code


# ---------------------------------------------------------------------------
# Official SDK (never a re-implementation of its HTTP API)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Sdk:
    """The exact official SDK surface this module uses."""

    sandbox: Any
    image_spec: Any
    network_policy: Any
    network_rule: Any
    credential_proxy: Any
    credential: Any
    credential_binding: Any
    connection_config: Any


def _load_sdk() -> _Sdk:
    try:
        from opensandbox.config.connection_sync import ConnectionConfigSync
        from opensandbox.models.sandboxes import (
            Credential,
            CredentialBinding,
            CredentialProxyConfig,
            NetworkPolicy,
            NetworkRule,
            SandboxImageSpec,
        )
        from opensandbox.sync.sandbox import SandboxSync
    except ImportError as exc:
        raise SandboxDispatchError(
            "the OpenSandbox Python SDK is not installed on this host, so no "
            "build machine can be provisioned. Ask the operator to install "
            "the plugin's declared dependency and retry.",
            code="sdk_unavailable",
        ) from exc
    return _Sdk(
        sandbox=SandboxSync,
        image_spec=SandboxImageSpec,
        network_policy=NetworkPolicy,
        network_rule=NetworkRule,
        credential_proxy=CredentialProxyConfig,
        credential=Credential,
        credential_binding=CredentialBinding,
        connection_config=ConnectionConfigSync,
    )


# ---------------------------------------------------------------------------
# Worker identity (host-owned; never a tool argument)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _WorkerContext:
    task_id: str
    run_id: int
    board: Optional[str]
    workspace_env: str
    profile: str = WORKER_PROFILE


def _active_profile() -> str:
    """The profile this process is actually running as."""
    name = (os.environ.get("HERMES_PROFILE") or "").strip()
    if name:
        return name
    try:
        from hermes_cli.profiles import get_active_profile_name

        return get_active_profile_name()
    except Exception:
        return ""


def _dispatcher_owned() -> bool:
    """False for delegate_task children and worker-fired cron jobs."""
    try:
        from agent.delegation_context import (
            is_delegated_child_context,
            is_dispatcher_owned_worker_context,
        )
    except Exception:
        return False
    return not is_delegated_child_context() and is_dispatcher_owned_worker_context()


def check_provision_available() -> bool:
    """Registry gate for dispatcher-owned engineering and verification workers.

    Deliberately cheap and env-only (the registry TTL-caches ``check_fn``
    results process-wide). Every gate here is re-checked, against host-owned
    board state, inside :func:`handle_provision`.
    """
    if not _dispatcher_owned():
        return False
    if _active_profile() not in SANDBOX_PROFILES:
        return False
    if not (os.environ.get("HERMES_KANBAN_TASK") or "").strip():
        return False
    return (os.environ.get("HERMES_KANBAN_RUN_ID") or "").strip().isdigit()


def _worker_context() -> _WorkerContext:
    if not _dispatcher_owned():
        raise SandboxDispatchError(
            "only the Kanban worker that owns this run may provision a build "
            "machine; delegated and scheduled work cannot.",
            code="not_dispatcher_owned",
        )
    profile = _active_profile()
    if profile not in SANDBOX_PROFILES:
        raise SandboxDispatchError(
            "this capability belongs to scoped engineering and verification workers; "
            f"this session is running as {profile or 'an unnamed profile'}.",
            code="wrong_profile",
        )
    task_id = (os.environ.get("HERMES_KANBAN_TASK") or "").strip()
    run_raw = (os.environ.get("HERMES_KANBAN_RUN_ID") or "").strip()
    if not task_id or not run_raw.isdigit():
        raise SandboxDispatchError(
            "no active Kanban run is attached to this session, so there is no "
            "task to provision a build machine for.",
            code="no_run_context",
        )
    return _WorkerContext(
        task_id=task_id,
        run_id=int(run_raw),
        board=(os.environ.get("HERMES_KANBAN_BOARD") or "").strip() or None,
        workspace_env=(os.environ.get("HERMES_KANBAN_WORKSPACE") or "").strip(),
        profile=profile,
    )


# ---------------------------------------------------------------------------
# Source identity (clean, git-backed, exactly scoped — or refuse)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Source:
    root: Path
    commit: str
    ownership_scope: list


def _git(root: Path, *args: str) -> str:
    """Run one read-only git query. Never surfaces git's own stderr."""
    try:
        result = subprocess.run(
            ["git", "-C", str(root), *args],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=_GIT_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise SandboxDispatchError(
            "git is not usable on this host, so the task source cannot be "
            "packaged.",
            code="git_unavailable",
        ) from exc
    if result.returncode != 0:
        raise SandboxDispatchError(
            "the task workspace is not a readable git checkout, so there is "
            "no exact source revision to send.",
            code="source_not_git",
        )
    return result.stdout.strip()


def _run_requires_read_only_source(ctx: _WorkerContext) -> bool:
    """Resolve review authority from the exact durable run, not task scope."""
    if ctx.profile in READ_ONLY_SOURCE_PROFILES:
        return True
    kb = _kanban()
    try:
        with kb.connect_closing(board=ctx.board) as conn:
            return kb.run_claimed_from_review(conn, ctx.task_id, ctx.run_id)
    except Exception as exc:
        raise SandboxDispatchError(
            "this run's review authority could not be read, so no source "
            "write boundary can be granted.",
            code="review_authority_unavailable",
        ) from exc


def _resolve_source(task: Any, ctx: _WorkerContext) -> _Source:
    if task.workspace_kind != "worktree" or not task.workspace_path:
        raise SandboxDispatchError(
            "this task has no git-backed source workspace (its workspace is "
            "scratch or unset), so there is no exact revision to send. Ask "
            "the operator to give the task a git worktree workspace.",
            code="source_not_worktree",
        )
    root = Path(task.workspace_path)
    if not root.is_dir():
        raise SandboxDispatchError(
            "the task's recorded source workspace does not exist on this "
            "host, so nothing can be packaged.",
            code="source_missing",
        )
    if not ctx.workspace_env:
        raise SandboxDispatchError(
            "this session has no resolved task workspace, so the source to "
            "send is ambiguous.",
            code="workspace_unresolved",
        )
    if Path(ctx.workspace_env).resolve(strict=False) != root.resolve(strict=False):
        raise SandboxDispatchError(
            "this session's workspace disagrees with the workspace recorded "
            "for the task on the board; refusing to guess which source is "
            "the task's.",
            code="workspace_mismatch",
        )
    read_only = _run_requires_read_only_source(ctx)
    if task.owned_paths is None and not read_only:
        raise SandboxDispatchError(
            "this task declares no repository ownership scope, so the source "
            "cannot be sent with a provable write boundary. Ask the operator "
            "to set the task's owned paths.",
            code="ownership_unscoped",
        )

    toplevel = _git(root, "rev-parse", "--show-toplevel")
    if not toplevel or Path(toplevel).resolve(strict=False) != root.resolve(
        strict=False
    ):
        raise SandboxDispatchError(
            "the task workspace is not the root of its own git checkout, so "
            "the source boundary is ambiguous.",
            code="source_not_root",
        )
    if _git(root, "status", "--porcelain"):
        raise SandboxDispatchError(
            "the task workspace has uncommitted changes; commit them so the "
            "build machine receives an exact revision.",
            code="source_not_clean",
        )
    commit = _git(root, "rev-parse", "HEAD")
    if not _COMMIT_RE.fullmatch(commit):
        raise SandboxDispatchError(
            "the task workspace has no resolvable HEAD commit to send.",
            code="source_no_head",
        )
    return _Source(
        root=root,
        commit=commit,
        ownership_scope=[] if read_only else list(task.owned_paths),
    )


def _package_source(source: _Source, staging: Path) -> tuple:
    """Write a tar of exactly the tracked tree at HEAD; return (path, digest)."""
    archive = staging / "source.tar"
    try:
        with open(archive, "wb") as handle:
            result = subprocess.run(
                ["git", "-C", str(source.root), "archive", "--format=tar",
                 source.commit],
                stdout=handle,
                stderr=subprocess.DEVNULL,
                timeout=_GIT_TIMEOUT_SECONDS,
            )
    except (OSError, subprocess.SubprocessError) as exc:
        raise SandboxDispatchError(
            "the task source could not be packaged on this host.",
            code="archive_failed",
        ) from exc
    if result.returncode != 0 or not archive.is_file() or archive.stat().st_size == 0:
        raise SandboxDispatchError(
            "the task source could not be packaged on this host.",
            code="archive_failed",
        )
    digest = hashlib.sha256()
    with open(archive, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return archive, f"sha256:{digest.hexdigest()}"


# ---------------------------------------------------------------------------
# Trusted host configuration (config.yaml settings + .env secret)
# ---------------------------------------------------------------------------


def _load_host_config() -> dict:
    """Read machine-wide ``raphael.sandbox`` settings from the shared root.

    Kanban workers run with profile-scoped ``HERMES_HOME`` values, but the
    Server 2 connection describes this host rather than one model profile.
    Keep one authority at ``<root>/config.yaml``.  The active-profile lookup
    remains a compatibility fallback until existing installations migrate.
    """
    try:
        from hermes_cli.config import cfg_get, load_config
        from hermes_constants import (
            get_default_hermes_root,
            reset_hermes_home_override,
            set_hermes_home_override,
        )

        legacy_config = load_config()
        token = set_hermes_home_override(get_default_hermes_root())
        try:
            shared_config = load_config()
        finally:
            reset_hermes_home_override(token)

        missing = object()
        section = cfg_get(
            shared_config, "raphael", "sandbox", default=missing,
        )
        if section is missing:
            section = cfg_get(
                legacy_config, "raphael", "sandbox", default={},
            )
    except Exception:
        section = None
    return dict(section) if isinstance(section, dict) else {}


def _load_connection_secret_from_env() -> str:
    """Read the already-configured OpenSandbox API key from ``.env``/env."""
    try:
        from hermes_cli.config import get_env_value_prefer_dotenv

        return (get_env_value_prefer_dotenv("OPEN_SANDBOX_API_KEY") or "").strip()
    except Exception:
        return ""


def _read_connection_secret_file(raw_path: str) -> str:
    """Read ``OPEN_SANDBOX_API_KEY`` out of a trusted operator-owned file.

    Some operators keep the Server 2 key in a systemd/agent credential file
    rather than in ``.env``. That file is *host* configuration: its path comes
    from ``config.yaml`` and can never be a tool argument, so a model cannot
    point this at an arbitrary file to have its contents read.

    The file must be an absolute path to a regular, non-symlink, owner-owned
    file with no group or other permission bits, opened ``O_NOFOLLOW`` so the
    check and the read cannot disagree. Only a literal
    ``OPEN_SANDBOX_API_KEY=<value>`` assignment is honoured (optionally
    ``export``-prefixed and optionally quoted); nothing is expanded and no
    shell ever runs. Neither the path nor the value is returned in an error or
    written to a log.
    """
    path = Path(raw_path)
    if not path.is_absolute():
        raise SandboxDispatchError(_SECRET_FILE_REFUSAL, code="connection_secret_file")
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    if hasattr(os, "O_NOCTTY"):
        flags |= os.O_NOCTTY
    try:
        fd = os.open(str(path), flags)
    except OSError as exc:
        raise SandboxDispatchError(
            _SECRET_FILE_REFUSAL, code="connection_secret_file"
        ) from exc
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode):
            raise SandboxDispatchError(
                _SECRET_FILE_REFUSAL, code="connection_secret_file"
            )
        if not sys.platform.startswith("win"):
            if info.st_uid != os.geteuid():  # windows-footgun: ok -- POSIX-gated above
                raise SandboxDispatchError(
                    _SECRET_FILE_REFUSAL, code="connection_secret_file"
                )
            if stat.S_IMODE(info.st_mode) & 0o077:
                raise SandboxDispatchError(
                    _SECRET_FILE_REFUSAL, code="connection_secret_file"
                )
        if info.st_size > _MAX_SECRET_FILE_BYTES:
            raise SandboxDispatchError(
                _SECRET_FILE_REFUSAL, code="connection_secret_file"
            )
        try:
            # Read to EOF rather than trusting one ``os.read``: a short read
            # would silently truncate the last line and turn a valid file into
            # a "malformed" refusal.
            chunks: list[bytes] = []
            remaining = _MAX_SECRET_FILE_BYTES
            while remaining > 0:
                chunk = os.read(fd, remaining)
                if not chunk:
                    break
                chunks.append(chunk)
                remaining -= len(chunk)
            body = b"".join(chunks).decode("utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            raise SandboxDispatchError(
                _SECRET_FILE_REFUSAL, code="connection_secret_file"
            ) from exc
    finally:
        os.close(fd)

    value = ""
    for line in body.splitlines():
        candidate = line.strip()
        if not candidate or candidate.startswith("#"):
            continue
        match = _SECRET_FILE_ASSIGNMENT_RE.fullmatch(candidate)
        if match is None:
            continue
        # Last assignment wins, matching how both ``.env`` loaders and a
        # sourced shell file resolve a repeated key.
        value = _unquote_secret(match.group("value"))
    if not _SECRET_VALUE_RE.fullmatch(value):
        raise SandboxDispatchError(_SECRET_FILE_REFUSAL, code="connection_secret_file")
    return value


def _unquote_secret(value: str) -> str:
    """Strip one matched pair of surrounding quotes. Never expands anything."""
    if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
        return value[1:-1]
    return value


def _load_connection_secret() -> str:
    """The Server 2 API key: ``.env``/env first, then the operator's file.

    Precedence is deliberate. ``.env`` is the documented place and the one an
    operator edits to rotate a key mid-session; the credential file is the
    fallback for hosts that keep the key outside ``.env`` entirely. If neither
    holds it, the caller's ``connection_unconfigured`` refusal applies.
    """
    value = _load_connection_secret_from_env()
    if value:
        return value
    raw_path = str(_load_host_config().get("api_key_file") or "").strip()
    if not raw_path:
        return ""
    return _read_connection_secret_file(raw_path)


def _resolve_host_credential() -> str:
    """The existing host-owned Anthropic resolver — never a tool argument."""
    try:
        from agent.anthropic_adapter import resolve_anthropic_token

        return (resolve_anthropic_token() or "").strip()
    except Exception:
        return ""


def _credential_is_oauth(secret: str) -> bool:
    try:
        from agent.anthropic_adapter import _is_oauth_token

        return bool(_is_oauth_token(secret))
    except Exception:
        # Unknown shape → the x-api-key contour, which is the narrower of the
        # two (one header, one credential) rather than a bearer grant.
        return False


def _claude_account_credentials() -> Optional[dict]:
    """The host's refreshable Claude *account* credential record, if any."""
    try:
        from agent.anthropic_adapter import read_claude_code_credentials

        record = read_claude_code_credentials()
    except Exception:
        return None
    return record if isinstance(record, dict) else None


def _credential_remaining_seconds(record: dict) -> Optional[float]:
    """Seconds of life left, or ``None`` when the record declares no expiry."""
    expires_at_ms = record.get("expiresAt") or 0
    if not isinstance(expires_at_ms, (int, float)) or expires_at_ms <= 0:
        return None
    return (float(expires_at_ms) / 1000.0) - time.time()


def _prepare_host_credential() -> str:
    """Resolve the Anthropic credential and prove it outlives the sandbox.

    A Claude account access token is refreshable but short-lived: one that is
    valid at provisioning time can still expire partway through a bounded
    task, and the worker inside the sandbox has no refresh token (it must
    never have one) and so cannot recover. So before the vault is written,
    inspect the credential's own metadata and, when its remaining lifetime is
    shorter than the sandbox lease plus a safety buffer, refresh it through
    the existing host refresh path — which rotates the pair on disk on Server
    1, where the refresh token stays.

    Two contours are deliberately left alone because they carry no such
    expiry: a Console API key (``sk-ant-api…``), and a long-lived
    setup-token/managed credential, whose record declares no ``expiresAt``.

    A freshly issued token's lifetime is re-checked against the same bound, so
    a successful preflight means the credential provably outlives the sandbox
    — which is also why a later reuse of the same machine needs no re-write,
    and why no background refresh loop is warranted. If it cannot be made to
    outlive the sandbox, this refuses rather than handing out a machine that
    will lose its sign-in mid-build.

    That re-check is unconditional, including when another Server 1 process
    rotated the pair concurrently: see :func:`_adopt_rotation_winner`. A
    disagreement between this call's refresh result and the persisted record
    is never itself a reason to hand out a token.
    """
    secret = _resolve_host_credential()
    if not secret:
        raise SandboxDispatchError(
            "this host holds no Anthropic sign-in material, so the build "
            "machine cannot be signed in. Ask the operator to connect the "
            "Anthropic account — never ask a person to paste a key.",
            code="host_credential_missing",
        )
    if not _credential_is_oauth(secret):
        return secret

    record = _claude_account_credentials()
    if record is None or record.get("accessToken") != secret:
        # Not the refreshable account credential (an operator-set env token or
        # a credential-pool entry): no expiry metadata to act on.
        return secret
    required = SANDBOX_TIMEOUT_SECONDS + CREDENTIAL_SAFETY_BUFFER_SECONDS
    remaining = _credential_remaining_seconds(record)
    if remaining is None or remaining >= required:
        return secret

    try:
        from agent.anthropic_adapter import _refresh_oauth_token

        refreshed = (_refresh_oauth_token(record) or "").strip()
    except Exception:
        refreshed = ""
    if not refreshed:
        raise SandboxDispatchError(
            "this host's Anthropic sign-in expires before a build could "
            "finish and could not be renewed, so no machine was started. Ask "
            "the operator to reconnect the Anthropic account — never ask a "
            "person to paste a key.",
            code="host_credential_expiring",
        )
    # Re-read the record the refresh just rotated. Either it still names the
    # token this call was handed — in which case that token's own lifetime is
    # what must clear the bound — or another Server 1 process won the rotation,
    # in which case this call's result is already superseded and only the
    # current record can be trusted.
    after = _claude_account_credentials() or {}
    if str(after.get("accessToken") or "").strip() != refreshed:
        return _adopt_rotation_winner(after, required)
    renewed = _credential_remaining_seconds(after)
    if renewed is not None and renewed < required:
        raise SandboxDispatchError(
            "this host's Anthropic sign-in was renewed but still expires "
            "before a build could finish, so no machine was started. Ask "
            "the operator to reconnect the Anthropic account.",
            code="host_credential_expiring",
        )
    return refreshed


def _adopt_rotation_winner(record: dict, required: float) -> str:
    """Adopt a concurrent winner's rotated access token, or fail closed.

    Reached only when the persisted account record no longer names the token
    this call's own refresh returned: another Server 1 process rotated the
    pair first, and an Anthropic refresh token is single-use, so this call's
    result may already be dead. The disagreement itself proves nothing about
    either token's remaining life, so nothing is handed out on the strength of
    it — the winner's token is adopted only on positive proof that it is
    present, still an account access token, and declares the whole sandbox
    lease plus the safety buffer left.

    Every other shape refuses: an absent or unreadable record, a value that is
    not an account token, and a token with no declared expiry or too little
    life. A worker inside the sandbox holds no refresh token by design and so
    cannot recover from a sign-in that dies mid-build; failing closed here
    costs one retry, while guessing costs the whole task. Neither the access
    token nor the refresh token appears in the refusal or in any log line.
    """
    current = str(record.get("accessToken") or "").strip()
    remaining = _credential_remaining_seconds(record) if current else None
    if (
        not current
        or not _credential_is_oauth(current)
        or remaining is None
        or remaining < required
    ):
        raise SandboxDispatchError(
            "this host's Anthropic sign-in was renewed by another process "
            "while this machine was being prepared, and the sign-in now on "
            "this host cannot be proven to outlive a build, so no machine was "
            "started. Retry once; if it persists, ask the operator to "
            "reconnect the Anthropic account — never ask a person to paste a "
            "key.",
            code="host_credential_rotated",
        )
    return current


@dataclass(frozen=True)
class _Connection:
    image_ref: str
    image_digest: str
    domain: str
    protocol: str
    api_key: str


def _resolve_connection() -> _Connection:
    config = _load_host_config()
    image = str(config.get("image") or "").strip()
    if not image:
        raise SandboxDispatchError(
            "no build-machine image is configured on this host. Ask the "
            "operator to pin one before remote work can start.",
            code="image_unconfigured",
        )
    match = _IMAGE_DIGEST_RE.fullmatch(image)
    if match is None:
        raise SandboxDispatchError(
            "the configured build-machine image is not pinned to an immutable "
            "digest, so the machine's contents cannot be proven. Ask the "
            "operator to pin it by digest.",
            code="image_not_pinned",
        )
    domain = str(config.get("domain") or "").strip()
    protocol = str(config.get("protocol") or "https").strip().lower()
    api_key = _load_connection_secret()
    if not domain or protocol not in {"http", "https"} or not api_key:
        raise SandboxDispatchError(
            "the remote build service is not configured on this host, so no "
            "machine can be reached. Ask the operator to finish connecting "
            "it.",
            code="connection_unconfigured",
        )
    return _Connection(
        image_ref=image,
        image_digest=match.group("digest"),
        domain=domain,
        protocol=protocol,
        api_key=api_key,
    )


# ---------------------------------------------------------------------------
# Native per-run reservation (kanban_db is the only authority)
# ---------------------------------------------------------------------------


def _kanban():
    from hermes_cli import kanban_db as kb

    return kb


def _temp_root() -> Path:
    """Host-owned staging root for packaged source, cleaned after every run."""
    return get_default_hermes_root() / "raphael" / "sandbox-dispatch-tmp"


def _log_event(
    ctx: Optional[_WorkerContext],
    outcome: str,
    *,
    level: int = logging.INFO,
    **fields: Any,
) -> None:
    """Emit one sanitized line to agent.log. Never a state store.

    The durable record lives in ``task_events`` (see :func:`_read_reservation`);
    this is only the operator-readable trace of what this process decided.
    """
    entry: dict[str, Any] = {"outcome": outcome, "profile": ctx.profile if ctx else _active_profile()}
    if ctx is not None:
        entry["task_id"] = ctx.task_id
        entry["run_id"] = ctx.run_id
        entry["board"] = ctx.board or "default"
    entry.update(fields)
    logger.log(level, "raphael sandbox dispatch: %s", json.dumps(entry, sort_keys=True))


def _read_reservation(ctx: _WorkerContext) -> dict:
    """Fold this run's sandbox transitions out of the board."""
    kb = _kanban()
    try:
        with kb.connect_closing(board=ctx.board) as conn:
            return kb.read_run_sandbox(conn, ctx.task_id, run_id=ctx.run_id)
    except Exception as exc:
        raise SandboxDispatchError(
            "the task board could not be read on this host, so a duplicate "
            "build machine cannot be ruled out.",
            code="reservation_unavailable",
        ) from exc


def _advance_reservation(
    ctx: _WorkerContext,
    transition: str,
    expected_generation: int,
    **fields: Any,
) -> Optional[dict]:
    """Append one native transition, or return ``None`` if another call won.

    ``None`` means the generation compare-and-swap failed: some other attempt
    for this exact run moved the reservation first. The caller must re-read
    and adapt — never create a second machine.
    """
    kb = _kanban()
    try:
        with kb.connect_closing(board=ctx.board) as conn:
            return kb.advance_run_sandbox(
                conn,
                ctx.task_id,
                run_id=ctx.run_id,
                transition=transition,
                expected_generation=expected_generation,
                **fields,
            )
    except kb.RunSandboxConflict:
        return None
    except Exception as exc:
        raise SandboxDispatchError(
            "this run's build-machine reservation could not be recorded on "
            "the board, so no machine was started.",
            code="reservation_unavailable",
        ) from exc


def _release_reservation(ctx: _WorkerContext, generation: int, reason: str) -> None:
    """Close the generation this call owns so a corrected retry is not wedged."""
    if not _record_cleaned_release(ctx, generation, reason):
        logger.warning(
            "raphael sandbox dispatch: reservation generation %s could not be "
            "released for task=%s run=%s",
            generation,
            ctx.task_id,
            ctx.run_id,
        )


# ---------------------------------------------------------------------------
# Provisioning
# ---------------------------------------------------------------------------


def _resolve_task(ctx: _WorkerContext):
    kb = _kanban()
    try:
        with kb.connect_closing(board=ctx.board) as conn:
            task = kb.get_task(conn, ctx.task_id)
    except Exception as exc:
        raise SandboxDispatchError(
            "the task board could not be read on this host, so this run's "
            "identity cannot be confirmed.",
            code="board_unavailable",
        ) from exc
    if task is None:
        raise SandboxDispatchError(
            "this session's task does not exist on the board, so there is "
            "nothing to provision for.",
            code="task_unknown",
        )
    if (task.assignee or "").strip() != ctx.profile:
        raise SandboxDispatchError(
            "this task is not assigned to this worker profile, so it "
            "may not provision a build machine.",
            code="task_not_assigned",
        )
    if task.status != "running" or task.current_run_id != ctx.run_id:
        raise SandboxDispatchError(
            "this session is not the board's active run for the task, so it "
            "may not provision a build machine.",
            code="run_not_active",
        )
    return task


def _sandbox_env(secret: str) -> dict:
    """Only a placeholder ever enters the sandbox environment."""
    if _credential_is_oauth(secret):
        return {"ANTHROPIC_AUTH_TOKEN": PLACEHOLDER_TOKEN}
    return {"ANTHROPIC_API_KEY": PLACEHOLDER_TOKEN}


def _vault_payload(sdk: _Sdk, secret: str) -> tuple:
    if _credential_is_oauth(secret):
        auth = {"type": "bearer", "credential": VAULT_CREDENTIAL_NAME}
    else:
        auth = {
            "type": "apiKey",
            "name": "x-api-key",
            "credential": VAULT_CREDENTIAL_NAME,
        }
    credentials = [
        sdk.credential(name=VAULT_CREDENTIAL_NAME, source={"value": secret})
    ]
    bindings = [
        sdk.credential_binding(
            name=VAULT_BINDING_NAME,
            match={
                "schemes": ["https"],
                "hosts": [VAULT_MATCH_HOST],
                "paths": [VAULT_MATCH_PATH],
                "methods": list(VAULT_MATCH_METHODS),
            },
            auth=auth,
        )
    ]
    return credentials, bindings


def _extract_command() -> str:
    """The one fixed seeding command; nothing here is caller-supplied.

    Extracts the *same* transferred archive twice — once into the editable
    workspace, once into the baseline — then deletes the archive and removes
    every write bit from the baseline, so the two trees start byte-identical
    and a stray non-root write cannot silently corrupt the ``diff -ru``.

    The ``chmod`` is a convenience for generating that recursive patch, not a
    guarantee about the baseline: the maintained MCP ``command_run`` on this
    pinned image runs as uid 0 and takes no uid/gid parameter, and uid 0
    rewrites the baseline whatever its mode bits say. The exactness the
    Reviewer relies on comes from reapplying the produced patch to the
    receipt's source commit, never from trusting this tree.
    """
    return (
        f"set -eu; mkdir -p '{SANDBOX_WORKSPACE}' '{SANDBOX_BASELINE}'; "
        f"tar -xf '{SANDBOX_ARCHIVE_PATH}' -C '{SANDBOX_WORKSPACE}'; "
        f"tar -xf '{SANDBOX_ARCHIVE_PATH}' -C '{SANDBOX_BASELINE}'; "
        f"rm -f '{SANDBOX_ARCHIVE_PATH}'; "
        f"chmod -R a-w '{SANDBOX_BASELINE}'"
    )


def _connection_config(sdk: _Sdk, connection: _Connection):
    """The one trusted Server 1 → Server 2 connection this module ever opens.

    ``use_server_proxy=True`` is not tunable. The live path from Server 1 to
    Server 2 is a loopback tunnel: only the management API is reachable, and
    the per-sandbox execd/egress endpoints the control plane advertises are
    not routable from here. Proxying execd traffic through the sandbox server
    is therefore the only way command/file/health calls reach the machine —
    with direct endpoints they hang until the request timeout and every
    provision fails after the machine already exists.
    """
    return sdk.connection_config(
        api_key=connection.api_key,
        domain=connection.domain,
        protocol=connection.protocol,
        request_timeout=timedelta(seconds=SANDBOX_REQUEST_TIMEOUT_SECONDS),
        use_server_proxy=True,
        disable_metrics=True,
    )


def _cleanup_connection_config(sdk: _Sdk, connection: _Connection):
    """Official SDK connection with a strict one-shot cleanup timeout."""
    return sdk.connection_config(
        api_key=connection.api_key,
        domain=connection.domain,
        protocol=connection.protocol,
        request_timeout=timedelta(
            seconds=SANDBOX_CLEANUP_REQUEST_TIMEOUT_SECONDS,
        ),
        use_server_proxy=True,
        disable_metrics=True,
    )


def _discard_confirmed(sandbox: Any) -> bool:
    """Kill one held machine and report only confirmed remote absence."""
    confirmed = False
    try:
        sandbox.kill()
        confirmed = True
    except Exception as exc:
        confirmed = _sandbox_is_confirmed_absent(exc)
        if not confirmed:
            logger.warning(
                "raphael sandbox dispatch: kill failed while unwinding a failed "
                "provision"
            )
    _close_quietly(sandbox)
    return confirmed


def _close_quietly(sandbox: Any) -> None:
    """Release only this process's client handle; the machine keeps running.

    ``close()`` shuts the SDK's local httpx transport, nothing on Server 2, so
    the worker can keep driving the same machine through the maintained
    OpenSandbox MCP command/file/info/health/kill tools afterwards.
    """
    try:
        sandbox.close()
    except Exception:
        logger.warning(
            "raphael sandbox dispatch: releasing the client handle failed"
        )


def _sandbox_is_confirmed_absent(exc: Exception) -> bool:
    """Use the official SDK's typed HTTP status as absence authority."""
    try:
        from opensandbox.exceptions import SandboxApiException
    except ImportError:
        return False
    return isinstance(exc, SandboxApiException) and exc.status_code == 404


def _receipt_still_holds(
    sandbox: Any, ctx: _WorkerContext, record: dict, *,
    minimum_remaining: int = SANDBOX_MIN_REMAINING_SECONDS,
) -> str:
    """Compare a live machine against the immutable receipt already recorded.

    Returns ``"ok"`` when the machine exists, is running, is healthy, has
    enough lease left to be worth handing back, and matches the receipt
    exactly; ``"retire"`` when it is provably this run's machine but no longer
    usable; ``"absent"`` when the official API proves it is gone;
    ``"foreign"`` when this process cannot prove the machine belongs to this
    run at all (so it must never be killed from here).
    """
    recorded_id = str(record.get("sandbox_id") or "")
    try:
        info = sandbox.get_info()
    except Exception as exc:
        return "absent" if _sandbox_is_confirmed_absent(exc) else "foreign"

    metadata = getattr(info, "metadata", None) or {}
    if (
        metadata.get("hermes_task") != ctx.task_id
        or metadata.get("hermes_run") != str(ctx.run_id)
        or metadata.get("hermes_profile") != ctx.profile
        or str(getattr(info, "id", "") or "") != recorded_id
    ):
        return "foreign"

    # From here the machine is provably this run's own, so an unusable one is
    # retired (killed) rather than left orphaned.
    receipt = record.get("receipt")
    if not isinstance(receipt, dict):
        return "retire"
    if str(receipt.get("sandbox_id") or "") != recorded_id:
        return "retire"
    state = str(getattr(getattr(info, "status", None), "state", "") or "").upper()
    if state != "RUNNING":
        return "retire"
    expires_at = getattr(info, "expires_at", None)
    if isinstance(expires_at, datetime):
        if expires_at.tzinfo is None:
            expires_at = expires_at.replace(tzinfo=timezone.utc)
        remaining = (expires_at - datetime.now(timezone.utc)).total_seconds()
        if remaining < minimum_remaining:
            return "retire"
    digest = str(receipt.get("image_digest") or "")
    image = str(getattr(getattr(info, "image", None), "image", "") or "")
    if not digest or not image.endswith(f"@{digest}"):
        return "retire"
    try:
        if not sandbox.is_healthy():
            return "retire"
    except Exception:
        return "retire"
    return "ok"


def _verify_recorded_sandbox(
    sdk: _Sdk, connection: _Connection, ctx: _WorkerContext, record: dict
) -> tuple[str, Optional[Any]]:
    """Return an explicit reuse/retirement disposition for one recorded machine.

    A receipt in ``task_events`` is durable, but the machine it names is not:
    Server 2 may have expired, evicted, or never finished it. So a reuse is
    only offered after the official SDK proves, over the same trusted
    connection, that the machine is still there and still the one the receipt
    describes.
    """
    sandbox_id = record.get("sandbox_id")
    if not isinstance(sandbox_id, str) or not sandbox_id:
        return "blocked", None
    try:
        sandbox = sdk.sandbox.connect(
            sandbox_id,
            connection_config=_connection_config(sdk, connection),
            connect_timeout=timedelta(seconds=SANDBOX_VERIFY_TIMEOUT_SECONDS),
        )
    except Exception as exc:
        return (
            ("confirmed_absent", None)
            if _sandbox_is_confirmed_absent(exc)
            else ("blocked", None)
        )
    verdict = _receipt_still_holds(sandbox, ctx, record)
    if verdict == "ok":
        return "reuse", sandbox
    if verdict == "absent":
        _close_quietly(sandbox)
        return "confirmed_absent", None
    if verdict == "retire":
        return (
            ("cleaned", None)
            if _discard_confirmed(sandbox)
            else ("blocked", None)
        )
    _close_quietly(sandbox)
    return "blocked", None


def _record_cleaned_release(
    ctx: _WorkerContext, generation: int, reason: str,
) -> bool:
    """Settle one killed sandbox in either the active or ended run phase."""
    kb = _kanban()
    try:
        with kb.connect_closing(board=ctx.board) as conn:
            try:
                settled = kb.advance_run_sandbox(
                    conn,
                    ctx.task_id,
                    run_id=ctx.run_id,
                    transition="sandbox_released",
                    expected_generation=generation,
                    reason=reason,
                )
            except kb.RunSandboxConflict:
                try:
                    settled = kb.release_ended_run_sandbox(
                        conn,
                        ctx.task_id,
                        run_id=ctx.run_id,
                        expected_generation=generation,
                        reason=reason,
                    )
                except kb.RunSandboxConflict:
                    return kb.read_run_sandbox(
                        conn, ctx.task_id, run_id=ctx.run_id,
                    ).get("state") == "released"
    except Exception:
        logger.warning(
            "raphael sandbox cleanup receipt failed for task=%s run=%s",
            ctx.task_id,
            ctx.run_id,
        )
        return False
    return settled.get("state") == "released"


def _record_orphan_cleanup(
    ctx: _WorkerContext,
    generation: int,
    sandbox_id: str,
    reason: str,
) -> bool:
    """Persist a losing remote allocation without touching the winner."""
    kb = _kanban()
    try:
        with kb.connect_closing(board=ctx.board) as conn:
            return kb.record_run_sandbox_orphan_cleanup(
                conn,
                ctx.task_id,
                run_id=ctx.run_id,
                generation=generation,
                sandbox_id=sandbox_id,
                reason=reason,
            ) or conn.execute(
                "SELECT 1 FROM run_sandbox_orphan_cleanup_intents "
                "WHERE task_id = ? AND run_id = ? AND sandbox_id = ?",
                (ctx.task_id, ctx.run_id, sandbox_id),
            ).fetchone() is not None
    except Exception:
        logger.warning(
            "raphael orphan sandbox cleanup authority could not be recorded "
            "for task=%s run=%s",
            ctx.task_id,
            ctx.run_id,
        )
        return False


def _record_orphan_release(
    ctx: _WorkerContext,
    sandbox_id: str,
    reason: str,
) -> bool:
    kb = _kanban()
    try:
        with kb.connect_closing(board=ctx.board) as conn:
            return kb.release_run_sandbox_orphan_cleanup(
                conn,
                ctx.task_id,
                ctx.run_id,
                sandbox_id,
                reason=reason,
            )
    except Exception:
        logger.warning(
            "raphael orphan sandbox cleanup receipt failed for task=%s run=%s",
            ctx.task_id,
            ctx.run_id,
        )
        return False


def _cleanup_orphan_sandbox(
    ctx: _WorkerContext,
    item: dict,
    *,
    reason: str,
) -> bool:
    """Kill one exact losing allocation without reading the winner's record."""
    sandbox_id = item.get("sandbox_id")
    generation = item.get("generation")
    if (
        not isinstance(sandbox_id, str)
        or not sandbox_id
        or isinstance(generation, bool)
        or not isinstance(generation, int)
    ):
        return False
    try:
        sdk = _load_sdk()
        connection = _resolve_connection()
    except Exception:
        return False
    sandbox = None
    try:
        sandbox = sdk.sandbox.connect(
            sandbox_id,
            connection_config=_cleanup_connection_config(sdk, connection),
            connect_timeout=timedelta(
                seconds=SANDBOX_CLEANUP_CONNECT_TIMEOUT_SECONDS,
            ),
            skip_health_check=True,
        )
    except Exception as exc:
        if _sandbox_is_confirmed_absent(exc):
            return _record_orphan_release(ctx, sandbox_id, "already_absent")
        return False
    verdict = _receipt_still_holds(
        sandbox,
        ctx,
        {
            "sandbox_id": sandbox_id,
            "receipt": {
                "sandbox_id": sandbox_id,
                "image_digest": connection.image_digest,
            },
        },
        minimum_remaining=0,
    )
    if verdict == "absent":
        _close_quietly(sandbox)
        return _record_orphan_release(ctx, sandbox_id, "already_absent")
    if verdict == "foreign":
        _close_quietly(sandbox)
        _log_event(
            ctx,
            "orphan_cleanup_refused",
            level=logging.WARNING,
            generation=generation,
            reason="ownership_unverified",
        )
        return False
    try:
        sandbox.kill()
    except Exception as exc:
        if not _sandbox_is_confirmed_absent(exc):
            _close_quietly(sandbox)
            return False
    _close_quietly(sandbox)
    return _record_orphan_release(ctx, sandbox_id, reason)


def _cleanup_run_sandbox(ctx: _WorkerContext, *, reason: str) -> bool:
    """Kill only this exact run's recorded sandbox and persist the cleanup."""
    try:
        record = _read_reservation(ctx)
    except SandboxDispatchError:
        return False
    if record.get("state") not in {"created", "active"}:
        return True
    sandbox_id = record.get("sandbox_id")
    generation = record.get("generation")
    if (
        not isinstance(sandbox_id, str)
        or not sandbox_id
        or isinstance(generation, bool)
        or not isinstance(generation, int)
    ):
        return False

    try:
        sdk = _load_sdk()
        connection = _resolve_connection()
    except Exception as exc:
        _log_event(
            ctx, "cleanup_pending", level=logging.WARNING,
            generation=generation, reason=type(exc).__name__,
        )
        return False

    sandbox = None
    try:
        sandbox = sdk.sandbox.connect(
            sandbox_id,
            connection_config=_cleanup_connection_config(sdk, connection),
            connect_timeout=timedelta(
                seconds=SANDBOX_CLEANUP_CONNECT_TIMEOUT_SECONDS,
            ),
            skip_health_check=True,
        )
    except Exception as exc:
        if _sandbox_is_confirmed_absent(exc):
            if _record_cleaned_release(ctx, generation, "already_absent"):
                _log_event(
                    ctx, "cleaned", generation=generation,
                    reason="already_absent",
                )
                return True
        _log_event(
            ctx, "cleanup_pending", level=logging.WARNING,
            generation=generation, reason=type(exc).__name__,
        )
        return False

    verdict = _receipt_still_holds(
        sandbox, ctx, record, minimum_remaining=0,
    )
    if verdict == "absent":
        _close_quietly(sandbox)
        if _record_cleaned_release(ctx, generation, "already_absent"):
            _log_event(
                ctx, "cleaned", generation=generation,
                reason="already_absent",
            )
            return True
        return False
    if verdict == "foreign":
        _close_quietly(sandbox)
        _log_event(
            ctx, "cleanup_refused", level=logging.WARNING,
            generation=generation, reason="ownership_unverified",
        )
        return False
    try:
        sandbox.kill()
    except Exception as exc:
        if not _sandbox_is_confirmed_absent(exc):
            _close_quietly(sandbox)
            _log_event(
                ctx, "cleanup_pending", level=logging.WARNING,
                generation=generation, reason=type(exc).__name__,
            )
            return False
    _close_quietly(sandbox)

    if not _record_cleaned_release(ctx, generation, reason):
        _log_event(
            ctx, "cleanup_pending", level=logging.WARNING,
            generation=generation, reason="release_record_unavailable",
        )
        return False
    _log_event(ctx, "cleaned", generation=generation, reason=reason)
    return True


def _hook_context(
    *,
    task_id: Any,
    run_id: Any,
    board: Any,
    assignee: Any,
    profile_name: Any,
) -> Optional[_WorkerContext]:
    profile = str(assignee or profile_name or "").strip()
    if (
        profile not in SANDBOX_PROFILES
        or not isinstance(task_id, str)
        or not task_id.strip()
        or isinstance(run_id, bool)
        or not isinstance(run_id, int)
        or run_id <= 0
    ):
        return None
    return _WorkerContext(
        task_id=task_id.strip(),
        run_id=run_id,
        board=str(board).strip() if board else None,
        workspace_env="",
        profile=profile,
    )


def cleanup_sandbox_on_session_end(**_kwargs: Any) -> None:
    """Clean a normally finalizing Kanban worker before it reports terminal."""
    try:
        ctx = _worker_context()
    except SandboxDispatchError:
        return
    _cleanup_run_sandbox(ctx, reason="session_end")


def _cleanup_sandbox_from_lifecycle(reason: str, **kwargs: Any) -> None:
    ctx = _hook_context(
        task_id=kwargs.get("task_id"),
        run_id=kwargs.get("run_id"),
        board=kwargs.get("board"),
        assignee=kwargs.get("assignee"),
        profile_name=kwargs.get("profile_name"),
    )
    if ctx is not None:
        _cleanup_run_sandbox(ctx, reason=reason)


def cleanup_sandbox_on_task_completed(**kwargs: Any) -> None:
    _cleanup_sandbox_from_lifecycle("task_completed", **kwargs)


def cleanup_sandbox_on_worker_exited(**kwargs: Any) -> None:
    _cleanup_sandbox_from_lifecycle("worker_exited", **kwargs)


def cleanup_sandbox_on_stale_claim(**kwargs: Any) -> None:
    _cleanup_sandbox_from_lifecycle("stale_claim", **kwargs)


_CLEANUP_RETRY_LEASE_SECONDS = 30


def _defer_cleanup_retry(
    ctx: _WorkerContext, *, reason: str,
) -> None:
    kb = _kanban()
    try:
        with kb.connect_closing(board=ctx.board) as conn:
            kb.defer_run_sandbox_cleanup(
                conn,
                ctx.task_id,
                ctx.run_id,
                reason=reason,
            )
    except Exception:
        logger.warning(
            "raphael sandbox cleanup retry schedule failed for task=%s run=%s",
            ctx.task_id,
            ctx.run_id,
        )


def _defer_orphan_cleanup_retry(
    ctx: _WorkerContext,
    sandbox_id: str,
    *,
    reason: str,
) -> None:
    kb = _kanban()
    try:
        with kb.connect_closing(board=ctx.board) as conn:
            kb.defer_run_sandbox_orphan_cleanup(
                conn,
                ctx.task_id,
                ctx.run_id,
                sandbox_id,
                reason=reason,
            )
    except Exception:
        logger.warning(
            "raphael orphan sandbox cleanup retry schedule failed "
            "for task=%s run=%s",
            ctx.task_id,
            ctx.run_id,
        )


def retry_ended_sandbox_cleanup(
    *, board: Any = None, dry_run: bool = False, **_kwargs: Any,
) -> None:
    """Complete one durably leased retry for persistent or one-shot dispatch."""
    if dry_run:
        return
    board_name = str(board).strip() if board else None
    kb = _kanban()
    try:
        with kb.connect_closing(board=board_name) as conn:
            pending = kb.claim_next_run_sandbox_cleanup(
                conn, lease_seconds=_CLEANUP_RETRY_LEASE_SECONDS,
            )
    except Exception:
        logger.warning(
            "raphael sandbox cleanup retry could not read board=%s",
            board_name or "default",
        )
        return
    if not pending:
        return
    item = pending[0]
    orphan = bool(item["orphan"])
    ctx = _WorkerContext(
        task_id=item["task_id"],
        run_id=item["run_id"],
        board=board_name,
        workspace_env="",
        profile=item.get("profile") or "",
    )
    if ctx.profile not in SANDBOX_PROFILES:
        logger.warning(
            "raphael sandbox cleanup retry has no admitted profile "
            "for task=%s run=%s",
            ctx.task_id,
            ctx.run_id,
        )
        if orphan:
            _defer_orphan_cleanup_retry(
                ctx, item["sandbox_id"], reason="profile_unavailable",
            )
        else:
            _defer_cleanup_retry(ctx, reason="profile_unavailable")
        return
    if orphan:
        if not _cleanup_orphan_sandbox(ctx, item, reason="dispatcher_retry"):
            _defer_orphan_cleanup_retry(
                ctx, item["sandbox_id"], reason="cleanup_retry_failed",
            )
    elif not _cleanup_run_sandbox(ctx, reason="dispatcher_retry"):
        _defer_cleanup_retry(ctx, reason="cleanup_retry_failed")


def _acquire(
    sdk: _Sdk, connection: _Connection, ctx: _WorkerContext
) -> tuple[str, Any]:
    """Adopt this run's live machine, or win the right to create exactly one.

    Returns ``("reuse", receipt)`` or ``("create", generation)``. Every
    decision is a compare-and-swap against the native record, so of two
    concurrent callers exactly one reaches ``create``; the loser re-reads and
    either reuses the winner's machine or refuses — it never creates a second.
    """
    for _ in range(MAX_RESERVATION_ATTEMPTS):
        record = _read_reservation(ctx)
        generation = record["generation"]
        state = record["state"]
        if state == "active":
            disposition, sandbox = _verify_recorded_sandbox(
                sdk, connection, ctx, record,
            )
            if disposition == "reuse" and sandbox is not None:
                # Proof is all this needed the handle for; the machine keeps
                # running for the worker's own MCP tools.
                _close_quietly(sandbox)
                return "reuse", record["receipt"]
            if disposition in {"confirmed_absent", "cleaned"}:
                _advance_reservation(
                    ctx,
                    "sandbox_released",
                    generation,
                    reason=disposition,
                )
                _log_event(
                    ctx,
                    "retired",
                    generation=generation,
                    reason=disposition,
                )
                continue
            raise SandboxDispatchError(
                "the recorded build machine could not be proved absent or "
                "safely retired, so its cleanup authority was retained and no "
                "replacement was started.",
                code="retirement_unconfirmed",
            )
        if state == "created":
            if _cleanup_run_sandbox(ctx, reason="provision_retry"):
                continue
            raise SandboxDispatchError(
                "a previously created build machine still needs cleanup before "
                "this run can start another.",
                code="created_cleanup_pending",
            )
        if state == "reserved":
            # Another attempt for this same run holds the open generation.
            # Refusing is the whole point of the CAS: the loser must wait
            # rather than start a second machine. A crashed attempt cannot
            # wedge the task, because the record is folded per run and the
            # next run starts from 'absent'.
            raise SandboxDispatchError(
                "a build machine for this task is already being prepared; "
                "wait for it instead of starting another.",
                code="reservation_in_flight",
            )
        opened = _advance_reservation(ctx, "sandbox_reserved", generation)
        if opened is None:
            continue
        return "create", opened["generation"]
    raise SandboxDispatchError(
        "this run's build-machine reservation kept changing underneath this "
        "attempt, so nothing was started. Retry once.",
        code="reservation_conflict",
    )


def _provision(
    sdk: _Sdk,
    connection: _Connection,
    ctx: _WorkerContext,
    source: _Source,
    generation: int,
) -> dict:
    """Create, seed, and durably record exactly one machine for *generation*."""
    staging = _temp_root() / uuid.uuid4().hex
    staging.mkdir(parents=True, exist_ok=True)
    secure_parent_dir(staging)
    sandbox = None
    created_recorded = False
    orphan_recorded = False
    provisioned = False
    try:
        secret = (
            _prepare_host_credential()
            if ctx.profile == WORKER_PROFILE
            else None
        )
        archive, source_digest = _package_source(source, staging)

        config = _connection_config(sdk, connection)
        network_policy = sdk.network_policy(
            default_action="deny",
            egress=[
                sdk.network_rule(action="allow", target=target)
                for target in EGRESS_ALLOWLIST
            ],
        )
        try:
            sandbox = sdk.sandbox.create(
                sdk.image_spec(image=connection.image_ref),
                timeout=timedelta(seconds=SANDBOX_TIMEOUT_SECONDS),
                ready_timeout=timedelta(seconds=SANDBOX_READY_TIMEOUT_SECONDS),
                resource=dict(SANDBOX_RESOURCES),
                env=_sandbox_env(secret) if secret is not None else {},
                metadata={
                    "hermes_task": ctx.task_id,
                    "hermes_run": str(ctx.run_id),
                    "hermes_profile": ctx.profile,
                },
                network_policy=network_policy,
                credential_proxy=(sdk.credential_proxy(enabled=True) if secret is not None else None),
                connection_config=config,
            )
        except Exception as exc:
            raise SandboxDispatchError(
                "the remote build service refused to start a machine. Ask the "
                "operator to check the connection and retry.",
                code="create_failed",
            ) from exc

        created = _advance_reservation(
            ctx,
            "sandbox_created",
            generation,
            sandbox_id=str(sandbox.id),
        )
        if created is None:
            # Persist the losing physical allocation BEFORE attempting kill.
            # kill() is a remote call and can hang, crash the process or fail;
            # the next dispatcher must already know this exact ID.
            orphan_recorded = _record_orphan_cleanup(
                ctx,
                generation,
                str(sandbox.id),
                "post_create_reservation_lost",
            )
            raise SandboxDispatchError(
                "another attempt advanced this task's build-machine reservation, "
                "so this newly created machine could not be adopted.",
                code="reservation_lost",
            )
        created_recorded = True
        # ``sandbox_created`` may be admitted solely to retain cleanup
        # authority after this run ended during remote create. Re-check the
        # exact native run before any credential, source, or command side
        # effect; cleanup below still owns the already-recorded exact ID.
        _resolve_task(ctx)
        _log_event(
            ctx,
            "created",
            generation=generation,
            sandbox_id=str(sandbox.id),
        )

        if secret is not None:
            credentials, bindings = _vault_payload(sdk, secret)
            try:
                sandbox.credential_vault.create(
                    credentials=credentials, bindings=bindings
                )
            except Exception as exc:
                raise SandboxDispatchError(
                    "the build machine's sign-in vault could not be written, so "
                    "the machine was discarded. Ask the operator to check the "
                    "remote build service and retry.",
                    code="vault_failed",
                ) from exc

        try:
            # Local imports: sandbox_artifacts imports this module at load
            # time, so importing it back at module scope here would be
            # circular.
            from pathlib import PurePosixPath

            from plugins.dashboard_auth.raphael_workspace import sandbox_artifacts as sa

            sa._verify_remote_containment(
                sandbox, PurePosixPath(SANDBOX_ARCHIVE_PATH), "/tmp", must_be_new=True,
                message="the archive staging path is not safe to write",
            )
            with open(archive, "rb") as handle:
                sandbox.files.write_file(SANDBOX_ARCHIVE_PATH, handle.read())
        except Exception as exc:
            raise SandboxDispatchError(
                "the task source could not be transferred to the build "
                "machine, so the machine was discarded. Retry once; if it "
                "persists, ask the operator to check the connection.",
                code="transfer_failed",
            ) from exc

        try:
            execution = sandbox.commands.run(_extract_command())
            exit_code = getattr(execution, "exit_code", None)
        except Exception as exc:
            raise SandboxDispatchError(
                "the task source could not be unpacked on the build machine, "
                "so the machine was discarded.",
                code="extract_failed",
            ) from exc
        if exit_code != 0:
            raise SandboxDispatchError(
                "the task source could not be unpacked on the build machine, "
                "so the machine was discarded.",
                code="extract_failed",
            )

        receipt = {
            "sandbox_id": str(sandbox.id),
            "image_digest": connection.image_digest,
            "source_commit": source.commit,
            "source_digest": source_digest,
            "workspace": SANDBOX_WORKSPACE,
            "baseline": SANDBOX_BASELINE,
            "ownership_scope": list(source.ownership_scope),
            "policy": {
                "credential_vault": secret is not None,
                "credential_proxy": secret is not None,
                "egress_default_deny": True,
                "sandbox_token_is_placeholder": secret is not None,
                "source_is_tracked_head_only": True,
                "source_tree_clean": True,
                "host_patch_import_authorized": bool(source.ownership_scope),
                "automatic_run_cleanup": True,
                "manual_sandbox_management_blocked": True,
                # What the seeding command provably did — not a claim that the
                # baseline is protected. uid 0 can still rewrite it; the
                # Reviewer verifies the produced patch against
                # ``source_commit`` instead.
                "baseline_write_bits_removed": True,
                "idempotent_reuse": False,
            },
        }
        settled = _advance_reservation(
            ctx,
            "sandbox_provisioned",
            generation,
            sandbox_id=str(sandbox.id),
            receipt=receipt,
        )
        if settled is None:
            # Another attempt settled this run's reservation while this one was
            # building. Only one machine may survive, and it is not this one.
            raise SandboxDispatchError(
                "another attempt finished this task's build machine first, so "
                "this one was discarded. Retry to pick up the existing "
                "machine.",
                code="reservation_lost",
            )
        provisioned = True
        # Success: release only this process's client handle. The machine
        # stays up for the worker's maintained MCP command/file/kill tools.
        _close_quietly(sandbox)
        _log_event(ctx, "provisioned", generation=generation, **_event_view(receipt))
        return receipt
    finally:
        # Any exit before the receipt is durable must either prove the remote
        # machine absent and close the generation, or retain the created ID and
        # cleanup intent for the dispatcher. Never turn an unconfirmed kill
        # into release authority.
        if not provisioned:
            if sandbox is not None and not created_recorded and not orphan_recorded:
                orphan_recorded = _record_orphan_cleanup(
                    ctx,
                    generation,
                    str(getattr(sandbox, "id", "") or ""),
                    "allocation_record_unavailable",
                )
            cleaned = sandbox is None
            if sandbox is not None:
                cleaned = _discard_confirmed(sandbox)
            if orphan_recorded:
                sandbox_id = str(getattr(sandbox, "id", "") or "")
                released = cleaned and _record_orphan_release(
                    ctx, sandbox_id, "provision_failed",
                )
                _log_event(
                    ctx,
                    "orphan_cleaned" if released else "orphan_cleanup_pending",
                    level=logging.INFO if released else logging.WARNING,
                    generation=generation,
                    reason=(
                        "provision_failed"
                        if released
                        else "provision_cleanup_unconfirmed"
                    ),
                )
            elif cleaned and created_recorded:
                _release_reservation(ctx, generation, "provision_failed")
            elif cleaned and sandbox is None:
                _release_reservation(ctx, generation, "provision_failed")
            elif cleaned:
                # A concurrent winner may have advanced this generation while
                # this process was inside remote create. Only close a still-
                # unallocated reservation; never release the winner's active
                # machine after killing our own unrecorded loser.
                try:
                    current = _read_reservation(ctx)
                except SandboxDispatchError:
                    current = {}
                if (
                    current.get("generation") == generation
                    and current.get("state") == "reserved"
                ):
                    _release_reservation(ctx, generation, "provision_failed")
            elif created_recorded:
                _log_event(
                    ctx,
                    "cleanup_pending",
                    level=logging.WARNING,
                    generation=generation,
                    reason="provision_cleanup_unconfirmed",
                )
            else:
                _log_event(
                    ctx,
                    "cleanup_untracked",
                    level=logging.ERROR,
                    generation=generation,
                    reason="orphan_authority_unavailable",
                )
        shutil.rmtree(staging, ignore_errors=True)


def _event_view(receipt: dict) -> dict:
    """The receipt's provable facts, flattened for the sanitized log line.

    Read defensively: a diagnostic log line must never be the thing that
    denies a worker a machine this call has already proved is good.
    """
    return {
        "sandbox_id": receipt.get("sandbox_id"),
        "image_digest": receipt.get("image_digest"),
        "source_commit": receipt.get("source_commit"),
        "source_digest": receipt.get("source_digest"),
        "workspace": receipt.get("workspace"),
        "baseline": receipt.get("baseline"),
        "ownership_scope": receipt.get("ownership_scope"),
        "policy": dict(receipt.get("policy") or {}),
    }


def handle_provision(args: dict, **_kwargs) -> str:
    """Create and seed exactly one Server 2 sandbox for this worker's run."""
    from tools.registry import tool_error, tool_result

    ctx = None
    owned_generation = None
    try:
        if args:
            raise SandboxDispatchError(
                "this capability takes no arguments; the build machine's "
                "source, specification, and sign-in are chosen by Hermes.",
                code="unexpected_arguments",
            )
        ctx = _worker_context()
        task = _resolve_task(ctx)
        source = _resolve_source(task, ctx)
        sdk = _load_sdk()
        connection = _resolve_connection()

        decision, payload = _acquire(sdk, connection, ctx)
        if decision == "reuse":
            receipt = dict(payload)
            receipt["policy"] = {**receipt.get("policy", {}), "idempotent_reuse": True}
            _log_event(ctx, "reused", **_event_view(receipt))
            return tool_result(receipt)

        generation = payload
        # From this point _provision owns the reserved generation and either
        # releases it after confirmed absence or leaves durable cleanup
        # authority. The outer generic error path must not guess.
        owned_generation = None
        return tool_result(
            _provision(sdk, connection, ctx, source, generation)
        )
    except SandboxDispatchError as exc:
        if ctx is not None:
            if owned_generation is not None and exc.code != "reservation_lost":
                _release_reservation(ctx, owned_generation, exc.code)
            _log_event(ctx, "failed", reason=exc.code)
        else:
            logger.warning("raphael sandbox dispatch refused: %s", exc.code)
        return tool_error(exc.message)
    except Exception as exc:
        # Never surface or log an unexpected exception body — it can carry
        # protected data. Emit one stable event whose only exception-derived
        # field is the type name: no message, no traceback, no exc_info.
        if ctx is not None:
            if owned_generation is not None:
                _release_reservation(ctx, owned_generation, "internal_error")
            _log_event(
                ctx,
                "failed",
                level=logging.ERROR,
                reason="internal_error",
                exception_type=type(exc).__name__,
            )
        else:
            logger.error(
                "raphael sandbox dispatch: unexpected failure (%s)",
                type(exc).__name__,
            )
        return tool_error(
            "the build machine could not be provisioned because of an "
            "unexpected host fault. Ask the operator to check the Hermes "
            "logs; do not ask a person for sign-in material."
        )


# Native tool hooks maintain a live run's lease without model polling.
# This is an in-process throttle, not a second sandbox authority.
_lease_checked: dict[tuple, float] = {}
_lease_lock = threading.Lock()

_OPENSANDBOX_MCP_OPERATIONS = frozenset({
    "sandbox_create",
    "sandbox_connect",
    "sandbox_kill",
    "sandbox_get_info",
    "sandbox_list",
    "sandbox_renew",
    "sandbox_get_metrics",
    "sandbox_healthcheck",
    "sandbox_get_endpoint",
    "command_run",
    "sandbox_command_run",
    "command_interrupt",
    "file_read",
    "file_write",
    "file_delete",
    "file_search",
    "file_create_directories",
    "file_delete_directories",
    "file_move",
    "file_replace_contents",
})


def _opensandbox_mcp_operation(tool_name: str) -> Optional[str]:
    """Resolve an OpenSandbox operation without trusting its server alias."""
    if not isinstance(tool_name, str):
        return None
    normalized = tool_name.strip().lower()
    if not normalized.startswith("mcp"):
        return None
    if normalized.startswith("mcp__") and "__" in normalized[5:]:
        operation = normalized.rsplit("__", 1)[-1]
        if operation in _OPENSANDBOX_MCP_OPERATIONS or operation.startswith(
            ("sandbox_", "command_", "file_")
        ):
            return operation
    for operation in _OPENSANDBOX_MCP_OPERATIONS:
        if normalized.endswith(f"_{operation}"):
            return operation
    if "opensandbox" in normalized:
        return normalized.rsplit("_", 1)[-1]
    return None


def enforce_sandbox_runtime(api_mode: str = "", **_kwargs):
    """Block runtimes that bypass this plugin's pre-tool authorization."""
    if api_mode != "codex_app_server" or _active_profile() not in SANDBOX_PROFILES:
        return None
    message = (
        "This Raphael profile cannot use codex_app_server because that runtime "
        "bypasses the scoped Server 2 tool authorization. Use the default "
        "OpenAI runtime for this profile."
    )
    try:
        ctx = _worker_context()
        _resolve_task(ctx)
        kb = _kanban()
        with kb.connect_closing(board=ctx.board) as conn:
            kb.block_task(
                conn,
                ctx.task_id,
                reason=message,
                kind="capability",
                expected_run_id=ctx.run_id,
            )
    except SandboxDispatchError:
        pass
    except Exception as exc:
        logger.warning(
            "sandbox runtime admission could not record blocker (%s)",
            type(exc).__name__,
        )
    return {"action": "block", "message": message}


def maintain_sandbox_lease(tool_name: str = "", args: Optional[dict] = None, **_kwargs):
    # Saved native artifacts remain usable after the execution machine expires.
    # Their handler still enforces the active task and input-authority checks.
    if (
        tool_name == "raphael_sandbox_artifact"
        and isinstance(args, dict)
        and args.get("direction") in {"inspect", "copy"}
    ):
        return None
    opensandbox_operation = _opensandbox_mcp_operation(tool_name)
    generic_opensandbox = opensandbox_operation is not None
    if generic_opensandbox and _active_profile() not in SANDBOX_PROFILES:
        return None
    if not generic_opensandbox and (
        tool_name not in {"kanban_heartbeat", "raphael_sandbox_artifact"}
        or not check_provision_available()
    ):
        return None
    sandbox = None
    with _lease_lock:
        try:
            ctx = _worker_context()
            _resolve_task(ctx)
            if opensandbox_operation in {
                "sandbox_create", "sandbox_list", "sandbox_kill", "sandbox_renew",
            }:
                raise SandboxDispatchError(
                    "Generic sandbox creation, listing, renewal, and manual killing are "
                    "unavailable to this worker. Use raphael_sandbox_provision "
                    "for the current run; lease renewal and cleanup are automatic.",
                    code="generic_sandbox_management",
                )
            record = _read_reservation(ctx)
            if generic_opensandbox:
                if record.get("state") != "active":
                    raise SandboxDispatchError(
                        "This run has no recorded active sandbox. Use "
                        "raphael_sandbox_provision before sandbox tools.",
                        code="sandbox_not_active",
                    )
                requested_id = args.get("sandbox_id") if isinstance(args, dict) else None
                if (
                    not isinstance(requested_id, str)
                    or not requested_id.strip()
                    or requested_id != record.get("sandbox_id")
                ):
                    raise SandboxDispatchError(
                        "This sandbox tool call does not target the current run's "
                        "recorded sandbox.",
                        code="sandbox_id_mismatch",
                    )
            elif record.get("state") != "active":
                return None
            key = (ctx.board, ctx.task_id, ctx.run_id, record["sandbox_id"])
            now = time.monotonic()
            if now - _lease_checked.get(key, float("-inf")) < 60:
                return None
            sdk = _load_sdk()
            connection = _resolve_connection()
            sandbox = sdk.sandbox.connect(
                record["sandbox_id"],
                connection_config=_connection_config(sdk, connection),
                connect_timeout=timedelta(seconds=SANDBOX_VERIFY_TIMEOUT_SECONDS),
            )
            if _receipt_still_holds(sandbox, ctx, record, minimum_remaining=0) != "ok":
                raise SandboxDispatchError(
                    "this task's sandbox is no longer available; preserve any "
                    "recorded artifacts and report the failure instead of restarting blindly."
                )
            expiry = sandbox.get_info().expires_at
            if isinstance(expiry, datetime):
                if expiry.tzinfo is None:
                    expiry = expiry.replace(tzinfo=timezone.utc)
                remaining = (expiry - datetime.now(timezone.utc)).total_seconds()
                if remaining < 2 * SANDBOX_MIN_REMAINING_SECONDS:
                    _resolve_task(ctx)
                    if ctx.profile == WORKER_PROFILE:
                        secret = _prepare_host_credential()
                        credentials, _bindings = _vault_payload(sdk, secret)
                        sandbox.credential_vault.patch(credentials={"replace": credentials})
                    sandbox.renew(timedelta(seconds=SANDBOX_TIMEOUT_SECONDS))
                    _log_event(ctx, "lease_renewed")
            _lease_checked[key] = now
            return None
        except SandboxDispatchError as exc:
            return {"action": "block", "message": exc.message}
        except Exception as exc:
            logger.warning("sandbox lease check failed (%s)", type(exc).__name__)
            return {
                "action": "block",
                "message": "The current build machine could not be checked. "
                "Record a recoverable blocker; do not create another attempt blindly.",
            }
        finally:
            if sandbox is not None:
                _close_quietly(sandbox)


# ---------------------------------------------------------------------------
# Plugin edge
# ---------------------------------------------------------------------------


def register_sandbox_tool(ctx) -> None:
    """Register the bounded provisioner, artifact operations and lease hook."""
    from plugins.dashboard_auth.raphael_workspace.sandbox_artifacts import register_artifact_tool

    register_artifact_tool(ctx)
    ctx.register_hook("pre_runtime_turn", enforce_sandbox_runtime)
    ctx.register_hook("pre_tool_call", maintain_sandbox_lease)
    ctx.register_hook("on_session_end", cleanup_sandbox_on_session_end)
    ctx.register_hook("kanban_task_completed", cleanup_sandbox_on_task_completed)
    ctx.register_hook("on_kanban_worker_exited", cleanup_sandbox_on_worker_exited)
    ctx.register_hook("on_kanban_worker_stale_claim", cleanup_sandbox_on_stale_claim)
    ctx.register_hook("on_kanban_dispatch_tick", retry_ended_sandbox_cleanup)
    ctx.register_tool(
        name=TOOL_NAME,
        toolset=TOOLSET,
        schema=PROVISION_SCHEMA,
        handler=handle_provision,
        check_fn=check_provision_available,
        description=(
            "Provision and seed a scoped Raphael worker's Server 2 sandbox "
            "from host-owned task state."
        ),
        emoji="📦",
    )
