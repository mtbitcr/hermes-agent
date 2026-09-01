"""
OpenAI-compatible API server platform adapter.

Exposes an HTTP server with endpoints:
- POST /v1/chat/completions        — OpenAI Chat Completions format (stateless; opt-in session continuity via X-Hermes-Session-Id header; opt-in long-term memory scoping via X-Hermes-Session-Key header)
- POST /v1/responses               — OpenAI Responses API format (stateful via previous_response_id; X-Hermes-Session-Key supported)
- GET  /v1/responses/{response_id} — Retrieve a stored response
- DELETE /v1/responses/{response_id} — Delete a stored response
- GET  /v1/models                  — lists hermes-agent and any configured model_routes aliases
- GET  /v1/capabilities            — machine-readable API capabilities for external UIs
- GET  /api/sessions               — list client-visible Hermes sessions
- POST /api/sessions               — create an empty Hermes session
- GET/PATCH/DELETE /api/sessions/{session_id} — read/update/delete a session
- GET  /api/sessions/{session_id}/messages — read session message history
- POST /api/sessions/{session_id}/fork — branch a session using SessionDB lineage
- POST /api/sessions/{session_id}/chat[/stream] — chat with a persisted session
- GET  /v1/owner-workspace/projects — list receipt-backed owner Projects
- GET  /v1/owner-workspace/projects/{project_slug}/snapshot — exact read-only Project surface
- GET  /v1/owner-workspace/projects/{project_slug}/attachments/{attachment_id} — exact owner attachment
- GET  /v1/owner-workspace/decisions — project pending native owner gates
- POST /v1/runs                    — start a run, returns run_id immediately (202)
- GET  /v1/runs/{run_id}           — retrieve current run status
- GET  /v1/runs/{run_id}/events    — SSE stream of structured lifecycle events
- POST /v1/runs/{run_id}/approval — resolve a pending run approval
- POST /v1/runs/{run_id}/steer      — inject guidance into a running agent
- POST /v1/runs/{run_id}/stop       — interrupt a running agent
- GET  /health                     — health check
- GET  /health/detailed            — rich status for cross-container dashboard probing

Any OpenAI-compatible frontend (Open WebUI, LobeChat, LibreChat,
AnythingLLM, NextChat, ChatBox, etc.) can connect to hermes-agent
through this adapter by pointing at http://localhost:8642/v1 and
authenticating with API_SERVER_KEY.

When ``gateway.multiplex_profiles`` is on, the default profile owns this
listener and secondary profiles are reached via a URL prefix — same contract
as the webhook adapter:

    GET  /p/<profile>/v1/models
    POST /p/<profile>/v1/chat/completions
    ...

Requires:
- aiohttp (already available in the gateway)
"""

import asyncio
import copy
import base64
import errno
import hashlib
import hmac
import itertools
import json
import math
from contextlib import contextmanager, nullcontext, suppress
from contextvars import ContextVar
from functools import wraps
import logging
import os
import re
import sqlite3
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

# Sentinel returned by _resolve_request_profile when a /p/<profile>/ prefix
# names a profile this gateway does not serve (→ 404). Distinct from None
# (no prefix / multiplexing off → handle as the default profile).
_PROFILE_REJECTED = object()

# Profile selected by the /p/<profile>/ URL prefix for the current request.
# Set by the profile-prefix middleware; read by handlers / _run_agent.
_api_request_profile: ContextVar[Optional[str]] = ContextVar(
    "api_server_request_profile", default=None
)

def _approval_event_choices(*, smart_denied: bool, allow_permanent: bool) -> list[str]:
    if smart_denied:
        return ["once", "deny"]
    return ["once", "session", "always", "deny"] if allow_permanent else ["once", "session", "deny"]


# ---------------------------------------------------------------------------
# Per-profile ``gateway.api_server.allowed_routes`` least-privilege gate.
#
# Resolved fresh per request (never cached) from the profile-scoped
# config.yaml so a multiplexed /p/<profile>/ request is gated by ITS OWN
# profile's setting, not the listener-owning default profile's.
# ---------------------------------------------------------------------------

_ROUTES_UNRESTRICTED = "unrestricted"
_ROUTES_DENY_ALL = "deny_all"
_ROUTES_ALLOWLIST = "allowlist"

_HTTP_ROUTE_METHODS = frozenset({"DELETE", "GET", "HEAD", "OPTIONS", "PATCH", "POST", "PUT"})

# Always reachable regardless of ``allowed_routes`` — the liveness probe.
_ALWAYS_ALLOWED_PATHS = ("/health",)

# Returned by _read_raw_config_root() when there is genuinely NO config to
# read: no file on disk, or a file with no YAML content at all (blank /
# comments only). Distinct from a successfully parsed root that simply isn't a
# mapping (``- a``, ``42``, ``null``), which is malformed and denies closed.
_CONFIG_ROOT_ABSENT = object()


def _read_raw_config_root() -> Any:
    """Return the CURRENT profile's ``config.yaml`` YAML root, unnormalized.

    Deliberately NOT ``read_user_config_raw()``: that helper collapses a
    non-mapping root to ``{}``, making ``- a\\n- b`` (or ``42``, or ``null``)
    indistinguishable from a missing file — which would resolve a malformed
    config to "key simply not present" and therefore UNRESTRICTED. This
    security boundary must tell those apart, so it reads the same
    profile-aware path (``get_config_path()``) and returns exactly what YAML
    parsed, plus a dedicated :data:`_CONFIG_ROOT_ABSENT` sentinel for "no
    config at all".

    Raises on unparseable YAML / unreadable file (callers deny closed).
    """
    from hermes_cli.config import get_config_path
    from utils import fast_safe_load

    path = get_config_path()
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return _CONFIG_ROOT_ABSENT
    # A file that carries no YAML content (empty, whitespace, or only
    # comments) parses to None but means "nothing configured" — treat it like
    # a missing file rather than a malformed root, so an empty/commented-out
    # config.yaml keeps its long-standing unrestricted behavior. An EXPLICIT
    # ``null``/``~`` root is content, and stays malformed.
    if not any(
        line.strip() and not line.lstrip().startswith("#")
        for line in text.splitlines()
    ):
        return _CONFIG_ROOT_ABSENT
    return fast_safe_load(text)


def _resolve_api_server_allowed_routes() -> "tuple[str, list[str]]":
    """Resolve ``gateway.api_server.allowed_routes`` for the CURRENT profile.

    Returns ``(mode, patterns)``:
      - ``(_ROUTES_UNRESTRICTED, [])`` — no config at all (missing/blank
        file), key omitted/``None``, OR the intentionally-supported empty
        list/tuple. No gating.
      - ``(_ROUTES_DENY_ALL, [])`` — an explicit malformed falsey value
        (``False``, ``0``, ``{}``, ``""``), any other malformed type/entry,
        a non-mapping YAML root (list/scalar/``null``), or a config-load
        failure. Only ``/health`` is reachable.
      - ``(_ROUTES_ALLOWLIST, patterns)`` — a non-empty string (single
        pattern) or list/tuple of non-empty strings.

    Uses ``_read_raw_config_root()`` (profile-aware via ``get_config_path()``
    → ``get_hermes_home()``, which the profile-prefix middleware has already
    scoped by the time this runs) because it RAISES on unparseable YAML —
    unlike the gateway's usual fail-open loader — and because it reports a
    successfully parsed NON-MAPPING root as itself instead of normalizing it
    to ``{}``. Both a load failure and a malformed root are therefore
    distinguishable from "key simply not present" and denied closed, per
    contract.
    """
    try:
        raw = _read_raw_config_root()
    except Exception:
        return (_ROUTES_DENY_ALL, [])
    if raw is _CONFIG_ROOT_ABSENT:
        # No config file / no YAML content — nothing configured, not malformed.
        return (_ROUTES_UNRESTRICTED, [])
    if not isinstance(raw, dict):
        # Parsed fine, but the root is a list/scalar/None — malformed config,
        # never "unconfigured".
        return (_ROUTES_DENY_ALL, [])

    gateway_cfg = raw.get("gateway")
    if gateway_cfg is None:
        return (_ROUTES_UNRESTRICTED, [])
    if not isinstance(gateway_cfg, dict):
        return (_ROUTES_DENY_ALL, [])

    api_server_cfg = gateway_cfg.get("api_server")
    if api_server_cfg is None:
        return (_ROUTES_UNRESTRICTED, [])
    if not isinstance(api_server_cfg, dict):
        return (_ROUTES_DENY_ALL, [])

    if "allowed_routes" not in api_server_cfg:
        return (_ROUTES_UNRESTRICTED, [])
    value = api_server_cfg["allowed_routes"]

    if value is None:
        return (_ROUTES_UNRESTRICTED, [])
    if isinstance(value, bool):
        # bool is an int subclass — check before the general int/str/etc.
        # fallthrough so True/False both land here (True is not a valid
        # list/string either — malformed, deny-all).
        return (_ROUTES_DENY_ALL, [])
    if isinstance(value, (list, tuple)):
        if len(value) == 0:
            return (_ROUTES_UNRESTRICTED, [])
        patterns: list[str] = []
        for entry in value:
            if not isinstance(entry, str) or not entry.strip():
                return (_ROUTES_DENY_ALL, [])
            normalized = _normalize_api_route_rule(entry)
            if normalized is None:
                return (_ROUTES_DENY_ALL, [])
            patterns.append(normalized)
        return (_ROUTES_ALLOWLIST, patterns)
    if isinstance(value, str):
        if value == "":
            return (_ROUTES_DENY_ALL, [])
        normalized = _normalize_api_route_rule(value)
        if normalized is None:
            return (_ROUTES_DENY_ALL, [])
        return (_ROUTES_ALLOWLIST, [normalized])
    # int (incl. 0), float, dict (incl. {}), or any other type: malformed.
    return (_ROUTES_DENY_ALL, [])


def _owner_workspace_toolset_enabled(user_config: dict) -> bool:
    """``gateway.api_server.owner_workspace.enabled`` — default OFF.

    Enables ONLY on the literal boolean ``True``. Anything else — including
    the truthy strings ``"true"``/``"yes"``/``"false"``, numbers, lists,
    mappings, ``None``, and every other malformed value — leaves the owner
    mutation surface disabled. A ``bool()`` coercion here would turn
    ``enabled: "false"`` (a very ordinary YAML quoting slip) into a live
    owner-mutation surface, so this admission gate demands the exact value
    its documentation promises rather than guessing at intent.

    Fail-closed to False on any resolution error: an owner-mutation surface
    must never appear because a config read hiccuped.
    """
    try:
        from hermes_cli.config import cfg_get

        value = cfg_get(user_config, "gateway", "api_server", "owner_workspace", "enabled", default=False)
    except Exception:
        return False
    return value is True


_OWNER_WORKSPACE_CAPABILITIES_MAX_LENGTH = 256


def _owner_workspace_capability_requested(
    request: "web.Request", capability: str,
) -> bool:
    """Whether this reader explicitly asked for one named response capability.

    The ``/v1`` owner-workspace surface keeps serving the exact response
    shape its oldest reader validates as a closed schema, so an added field
    is opt-in: a reader that understands one names it in
    ``?capabilities=a,b``. That keeps either side of a rolling deployment
    readable by the other, because an older Hermes ignores the parameter and
    an older reader never sends it.

    Anything else grants nothing: a missing, repeated, oversized, or unknown
    value is the same as not asking. The key must occur EXACTLY once — a
    request that sends it twice has no single negotiated answer, and reading
    one of the two (``query.get`` returns the first) would make the grant
    depend on the order a caller happened to write them in. Both orders
    therefore fail closed to the legacy shape.
    """
    raw_values = request.query.getall("capabilities", [])
    if len(raw_values) != 1:
        return False
    raw = raw_values[0]
    if len(raw) > _OWNER_WORKSPACE_CAPABILITIES_MAX_LENGTH:
        return False
    return capability in {token.strip() for token in raw.split(",")}


_OWNER_PROJECT_NAME_MAX_LENGTH = 160


def _resolve_owner_workspace_run_context(value: Any) -> "dict[str, str | None] | None":
    """Validate optional owner routing metadata against native Project state.

    The context never grants mutation authority and never accepts a native id.
    It is retained only with the in-memory Run status so the read-only
    Decisions inbox can route an active approval back to its originating
    owner surface. Existing Projects are resolved from receipt-backed native
    state, including archived Projects that need a restore approval; a
    not-yet-created Project may carry only its bounded display name.
    """
    if value is None:
        return None
    if not isinstance(value, dict) or set(value) != {
        "mode", "project_slug", "project_name",
    }:
        raise ValueError("invalid owner workspace context")

    from gateway.run import _load_gateway_config

    if not _owner_workspace_toolset_enabled(_load_gateway_config()):
        raise ValueError("owner workspace is disabled")

    mode = value.get("mode")
    project_slug = value.get("project_slug")
    project_name = value.get("project_name")
    if mode not in {"new", "existing"}:
        raise ValueError("invalid owner workspace mode")

    from hermes_cli.owner_workspace import (
        OwnerWorkspaceError,
        _native_owner_project_name,
        list_committed_projects,
        owner_project_name,
        resolve_owner_context,
    )

    owner = resolve_owner_context()
    if mode == "existing":
        if (
            not isinstance(project_slug, str)
            or re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", project_slug) is None
        ):
            raise ValueError("invalid owner Project slug")
        matches = [
            project for project in list_committed_projects(owner)
            if project.get("slug") == project_slug
        ]
        if len(matches) != 1:
            raise ValueError("owner Project is unavailable")
        project_name = matches[0].get("name")
    else:
        if project_slug is not None:
            raise ValueError("new Project cannot have a slug")
        if not isinstance(project_name, str):
            raise ValueError("new Project name is required")
        # A client-supplied new name is still rejected fail-fast at the same
        # 160-code-point bound ``commit_task_graph`` enforces when the Project
        # is first written. Letting the projection below shorten it instead
        # would admit a request the create path refuses and route the approval
        # under a name the client never asked for. Only a *new* name is a
        # client claim about a Project that does not exist yet; an existing
        # name resolved from receipt-backed state was already storable, so it
        # only ever needs projecting on read.
        try:
            project_name = _native_owner_project_name(
                project_name, "project_name",
            )
        except OwnerWorkspaceError as exc:
            raise ValueError("invalid owner Project name") from exc

    # The retained name is owner-facing display text on the Decisions inbox,
    # so it leaves here through the one canonical Project-name projection —
    # which sanitizes and redacts URL credentials — rather than through a
    # sanitizer this boundary maintains itself.
    project_name = owner_project_name(project_name)
    profile = str(getattr(owner, "profile", "") or "").strip()
    if not profile:
        raise ValueError("owner profile is unavailable")
    return {
        "mode": str(mode),
        "project_slug": project_slug if mode == "existing" else None,
        "project_name": project_name,
        "profile": profile,
    }


def _resolve_owner_proposal_run_authority(value: Any) -> "dict[str, Any] | None":
    """Validate hidden proposal authority on one authenticated owner run."""
    if value is None:
        return None
    expected = {
        "proposal_profile", "conversation", "response_id", "claim_id",
        "operation", "idempotency_key", "payload",
    }
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError("invalid owner proposal authority")
    proposal_profile = value.get("proposal_profile")
    conversation = value.get("conversation")
    response_id = value.get("response_id")
    claim_id = value.get("claim_id")
    operation = value.get("operation")
    idempotency_key = value.get("idempotency_key")
    payload = value.get("payload")
    if (
        not isinstance(proposal_profile, str)
        or _OWNER_PROFILE_RE.fullmatch(proposal_profile) is None
        or not isinstance(conversation, str)
        or _OWNER_CONVERSATION_RE.fullmatch(conversation) is None
        or not isinstance(response_id, str)
        or _OWNER_RESPONSE_RE.fullmatch(response_id) is None
        or not isinstance(claim_id, str)
        or _OWNER_CLAIM_RE.fullmatch(claim_id) is None
        or operation not in {
            "owner_task_graph_commit", "owner_project_plan_commit",
        }
        or not isinstance(idempotency_key, str)
        or re.fullmatch(r"[A-Za-z0-9._:-]{1,200}", idempotency_key) is None
        or not isinstance(payload, dict)
        or payload.get("idempotency_key") != idempotency_key
    ):
        raise ValueError("invalid owner proposal authority")
    return {
        "proposal_profile": proposal_profile,
        "conversation": conversation,
        "response_id": response_id,
        "claim_id": claim_id,
        "operation": operation,
        "idempotency_key": idempotency_key,
        "payload": payload,
    }


def _resolve_owner_lifecycle_run_authority(value: Any) -> "dict[str, Any] | None":
    """Validate the closed transport shape for one Project lifecycle run."""
    if value is None:
        return None
    if not isinstance(value, dict) or set(value) != {
        "operation", "idempotency_key", "payload",
    }:
        raise ValueError("invalid owner lifecycle authority")
    operation = value.get("operation")
    idempotency_key = value.get("idempotency_key")
    payload = value.get("payload")
    if (
        operation != "owner_project_lifecycle"
        or not isinstance(idempotency_key, str)
        or re.fullmatch(r"[A-Za-z0-9._:-]{1,200}", idempotency_key) is None
        or not isinstance(payload, dict)
        or set(payload) != {
            "idempotency_key", "project_id", "expected_revision", "action",
        }
        or payload.get("idempotency_key") != idempotency_key
    ):
        raise ValueError("invalid owner lifecycle authority")
    return {
        "operation": operation,
        "idempotency_key": idempotency_key,
        "payload": payload,
    }


def _normalize_api_route_rule(value: str) -> "str | None":
    """Normalize one legacy path prefix or exact METHOD route-template."""
    rule = value.strip()
    if not rule:
        return None
    if rule.startswith("/"):
        return rule
    parts = rule.split(None, 1)
    if len(parts) != 2:
        return None
    method, template = parts[0].upper(), parts[1].strip()
    if method not in _HTTP_ROUTE_METHODS:
        return None
    if not template.startswith("/") or any(ch.isspace() for ch in template):
        return None
    if "?" in template or "#" in template:
        return None
    normalized_template = template.rstrip("/") or "/"
    return f"{method} {normalized_template}"


def _route_matches_any(
    path: str,
    patterns: "list[str]",
    *,
    method: "str | None" = None,
    route_template: "str | None" = None,
) -> bool:
    """Match legacy path prefixes or exact method + route-template rules."""
    canonical = (route_template or path).rstrip("/") or "/"
    for pattern in patterns:
        if not pattern.startswith("/"):
            rule_method, rule_template = pattern.split(" ", 1)
            if method and method.upper() == rule_method and canonical == rule_template:
                return True
            continue
        pat = pattern.rstrip("/") or "/"
        if path == pat or path.startswith(pat + "/"):
            return True
    return False


def _resolve_api_server_allowed_toolsets(user_config: dict) -> "list[str] | None":
    """Return an exact API-server toolset allowlist, or None for legacy mode.

    A present malformed value denies all. An explicit empty list is therefore
    a stable zero-tool policy: newly installed plugins or MCP servers cannot
    widen an API profile behind the operator's back.
    """
    try:
        gateway_cfg = user_config.get("gateway")
        if not isinstance(gateway_cfg, dict):
            return None if gateway_cfg is None else []
        api_server_cfg = gateway_cfg.get("api_server")
        if not isinstance(api_server_cfg, dict):
            return None if api_server_cfg is None else []
        if "allowed_toolsets" not in api_server_cfg:
            return None
        value = api_server_cfg["allowed_toolsets"]
        if not isinstance(value, list):
            return []

        from toolsets import validate_toolset

        result: list[str] = []
        for entry in value:
            if not isinstance(entry, str) or not entry.strip():
                return []
            name = entry.strip()
            if not validate_toolset(name):
                return []
            if name not in result:
                result.append(name)
        return result
    except Exception:
        return []


def _resolve_api_server_agent_toolsets(user_config: dict) -> list[str]:
    """Resolve the exact tool authority for one API-server request profile."""
    exact = _resolve_api_server_allowed_toolsets(user_config)
    if exact is None:
        from hermes_cli.tools_config import _get_platform_tools

        enabled = set(_get_platform_tools(user_config, "api_server"))
        if _owner_workspace_toolset_enabled(user_config):
            enabled.add("owner_workspace")
        return sorted(enabled)

    from toolsets import get_kernel_gated_toolsets

    enabled = set(exact)
    if not _owner_workspace_toolset_enabled(user_config):
        enabled -= get_kernel_gated_toolsets()
    return sorted(enabled)


try:
    from aiohttp import web
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False
    web = None  # type: ignore[assignment]

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import (
    MEDIA_TAG_CLEANUP_RE,
    BasePlatformAdapter,
    SendResult,
    is_network_accessible,
    validate_media_delivery_path,
)
from agent.redact import redact_sensitive_text
from agent.interrupt_compat import request_hard_interrupt
from agent.message_metadata import PERSISTENCE_ONLY_MESSAGE_FIELDS
from gateway.readiness import collect_runtime_readiness

from agent.secret_scope import UnscopedSecretError as _UnscopedSecretError
from agent.secret_scope import get_secret as _scoped_get_secret


def _get_scoped_secret(name, default=None):
    """Scope-aware credential read with the default-profile startup fallback.

    Secondary profiles construct their adapters under a profile secret
    scope -- the scope is authoritative and a scoped miss returns ``default``
    (no cross-profile borrow from ``os.environ``, which may hold another
    profile's value). The DEFAULT profile's adapter constructs and sends
    *unscoped* under multiplexing, where a bare ``get_secret`` would raise
    ``UnscopedSecretError`` and crash this path; there ``os.environ`` is that
    profile's own value, so fall back to it. Same pattern as the Slack
    ``SLACK_APP_TOKEN`` read (#59739) and
    ``gateway/platforms/whatsapp_common.py::_get_wsecret``.
    """
    try:
        val = _scoped_get_secret(name, default)
    except _UnscopedSecretError:
        val = os.getenv(name)
    return val if val is not None else default


logger = logging.getLogger(__name__)


def _hermes_version() -> str:
    """Return the canonical Hermes Agent version string.

    ``hermes_cli.__version__`` is the runtime source of truth used by the CLI,
    dashboard, portal tags, and release script. Prefer it over installed
    distribution metadata because editable/source checkouts can retain stale
    ``hermes_agent-*.dist-info`` after a source update until the environment is
    reinstalled. Never raises — a version probe must not be able to break the
    health endpoint.
    """
    try:
        from hermes_cli import __version__

        return __version__
    except Exception:
        pass
    try:
        from importlib.metadata import version

        return version("hermes-agent")
    except Exception:
        return "dev"


# Default settings
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8642
MAX_STORED_RESPONSES = 100
MAX_REQUEST_BYTES = 10_000_000  # 10 MB — accommodates long agent conversations with tool calls
CHAT_COMPLETIONS_SSE_KEEPALIVE_SECONDS = 30.0
MAX_NORMALIZED_TEXT_LENGTH = 65_536  # 64 KB cap for normalized content parts
MAX_CONTENT_LIST_SIZE = 1_000  # Max items when content is an array
RESPONSES_AUTO_TRUNCATION_HISTORY_LIMIT = 100
_COMPRESSED_SUMMARY_METADATA_KEY = "_compressed_summary"


class ThreadSafeAsyncQueue(asyncio.Queue):
    """An ``asyncio.Queue`` that a non-loop thread can push into safely.

    The SSE writers' streaming loops used to bridge a plain ``queue.Queue``
    into the event loop via ``await loop.run_in_executor(None, lambda:
    stream_q.get(timeout=0.5))`` inside a ``while True`` poll — a thread-pool
    round trip on every 0.5s tick even when idle, plus up to 500ms of tail
    latency between a delta landing in the queue and it reaching the
    response. ``run_conversation`` itself runs on a worker thread (via
    ``loop.run_in_executor``), so its ``stream_delta_callback`` closures
    (``_on_delta`` etc.) call ``put_threadsafe`` from off the loop thread;
    the consumer side just does a plain ``await queue.get()``/
    ``asyncio.wait_for(queue.get(), timeout=...)``, woken immediately by
    ``call_soon_threadsafe`` instead of polling.
    """

    def put_threadsafe(self, item, *, loop: asyncio.AbstractEventLoop = None) -> None:
        (loop or self._loop_ref).call_soon_threadsafe(self.put_nowait, item)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Always constructed inside a running async handler (the SSE
        # request handlers below), so get_running_loop() is safe here.
        self._loop_ref = asyncio.get_running_loop()


def _sse_frame(data: Any, *, event: str = None, ensure_ascii: bool = True) -> bytes:
    """Encode one SSE frame: optional ``event:`` line, then ``data: <json>\n\n``.

    The single source of truth for SSE frame serialization across every
    streaming writer in this module — ``_write_sse_chat_completion`` (the
    five call sites it was first extracted from), ``_write_sse_responses``'s
    inner ``_write_event`` closure, and the ``/v1/runs`` event stream.  All
    three used the identical ``json.dumps(data)`` / ``json.dumps(...,
    ensure_ascii=False)`` + ``"\\ndata: ...\\n\\n"`` shape; routing them all
    through here keeps the on-the-wire format in exactly one place.

    ``ensure_ascii`` defaults to ``True``, byte-identical to a bare
    ``json.dumps(data)``.  Callers that must preserve raw non-ASCII bytes on
    the wire (the Responses-API writer historically used
    ``ensure_ascii=False``) pass ``ensure_ascii=False`` explicitly — the
    option exists so every writer shares one helper without changing any
    existing byte stream.
    """
    prefix = f"event: {event}\n" if event else ""
    return f"{prefix}data: {json.dumps(data, ensure_ascii=ensure_ascii)}\n\n".encode()


def _coerce_port(value: Any, default: int = DEFAULT_PORT) -> int:
    """Parse a listen port without letting malformed env/config values crash startup."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


_TRUE_REQUEST_BOOL_STRINGS = frozenset({"1", "true", "yes", "on"})
_FALSE_REQUEST_BOOL_STRINGS = frozenset({"0", "false", "no", "off"})


def _coerce_request_bool(value: Any, default: bool = False) -> bool:
    """Normalize boolean-like API payload values.

    External clients should send real JSON booleans, but some OpenAI-compatible
    frontends and middleware serialize flags like ``stream`` as strings.  Using
    Python truthiness on those values misroutes requests because ``"false"`` is
    still truthy.  Treat only explicit bool-ish scalars as booleans; everything
    else falls back to the caller's default.
    """
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in _TRUE_REQUEST_BOOL_STRINGS:
            return True
        if normalized in _FALSE_REQUEST_BOOL_STRINGS:
            return False
        return default
    if isinstance(value, (int, float)):
        return bool(value)
    return default


_REQUEST_OPTION_MISSING = object()
_REASONING_EFFORTS = frozenset({"none", "minimal", "low", "medium", "high", "xhigh", "max"})
_RUNTIME_AGENT_OVERRIDE_KEYS = (
    "api_key",
    "base_url",
    "provider",
    "api_mode",
    "command",
    "args",
    "credential_pool",
    "max_tokens",
)


def _clean_request_string(value: Any) -> Optional[str]:
    """Return a stripped request string, or None for absent/non-string values."""
    if not isinstance(value, str):
        return None
    cleaned = value.strip()
    return cleaned or None


def _request_reasoning_config(model_options: Any) -> Optional[Dict[str, Any]]:
    """Translate browser/API model_options into AIAgent reasoning_config.

    The browser extension sends both a structured ``reasoning`` object and a
    compatibility ``reasoning_effort`` scalar.  Keep this parser permissive so
    older clients can send either shape, but ignore unknown effort values rather
    than raising on a chat request.
    """
    if not isinstance(model_options, dict):
        return None

    reasoning = model_options.get("reasoning")
    enabled: Any = None
    effort: Any = model_options.get("reasoning_effort")
    if isinstance(reasoning, dict):
        enabled = reasoning.get("enabled")
        effort = reasoning.get("effort", effort)

    effort_norm = str(effort).strip().lower() if effort is not None else ""
    if enabled is False or effort_norm == "none":
        return {"enabled": False}
    if effort_norm in _REASONING_EFFORTS and effort_norm != "none":
        return {"enabled": True, "effort": effort_norm}
    if enabled is True:
        return {"enabled": True}
    return None


def _request_service_tier(model_options: Any) -> Any:
    """Return a per-request service_tier override or _REQUEST_OPTION_MISSING."""
    if not isinstance(model_options, dict):
        return _REQUEST_OPTION_MISSING
    if "service_tier" in model_options:
        raw_tier = model_options.get("service_tier")
        if raw_tier is None:
            return None
        if isinstance(raw_tier, str):
            return raw_tier.strip() or None
        return raw_tier
    if "fast" in model_options:
        return "priority" if _coerce_request_bool(model_options.get("fast"), default=False) else None
    return _REQUEST_OPTION_MISSING


def _apply_runtime_agent_overrides(
    runtime_kwargs: Dict[str, Any], overrides: Optional[Dict[str, Any]]
) -> Dict[str, Any]:
    """Merge resolved provider/runtime fields into ``runtime_kwargs`` in place."""
    if not isinstance(overrides, dict):
        return runtime_kwargs
    for key in _RUNTIME_AGENT_OVERRIDE_KEYS:
        if key not in overrides:
            continue
        value = overrides.get(key)
        if value is None:
            continue
        runtime_kwargs[key] = list(value) if key == "args" and isinstance(value, (list, tuple)) else value
    return runtime_kwargs


def _resolve_request_runtime_agent_kwargs(provider: str, target_model: Optional[str] = None) -> Dict[str, Any]:
    """Resolve runtime kwargs for a one-request provider override.

    This mirrors gateway.run._resolve_runtime_agent_kwargs(), but accepts an
    explicit provider/model so an API caller can use the same authenticated
    provider catalog as the TUI without mutating config.yaml.
    """
    from hermes_cli.runtime_provider import resolve_runtime_provider, format_runtime_provider_error, _get_model_config

    try:
        runtime = resolve_runtime_provider(requested=provider, target_model=target_model)
    except Exception as exc:
        raise RuntimeError(format_runtime_provider_error(exc)) from exc

    model_cfg = _get_model_config()
    max_tokens = None
    env_max_tokens = os.environ.get("HERMES_MAX_TOKENS")
    if env_max_tokens:
        try:
            max_tokens = int(env_max_tokens)
        except (ValueError, TypeError):
            max_tokens = None
    elif isinstance(model_cfg, dict):
        cfg_max_tokens = model_cfg.get("max_tokens")
        if isinstance(cfg_max_tokens, int):
            max_tokens = cfg_max_tokens
    if max_tokens is None:
        runtime_max_tokens = runtime.get("max_output_tokens")
        if isinstance(runtime_max_tokens, int) and runtime_max_tokens > 0:
            max_tokens = runtime_max_tokens

    return {
        "api_key": runtime.get("api_key"),
        "base_url": runtime.get("base_url"),
        "provider": runtime.get("provider"),
        "api_mode": runtime.get("api_mode"),
        "command": runtime.get("command"),
        "args": list(runtime.get("args") or []),
        "credential_pool": runtime.get("credential_pool"),
        "max_tokens": max_tokens,
    }


def _request_agent_overrides(
    body: Any,
    *,
    virtual_model: Optional[str] = None,
    allow_bare_model: bool = True,
) -> Dict[str, Any]:
    """Extract per-request model/provider/options for _run_agent.

    ``/v1/models`` advertises a stable virtual model (usually ``hermes-agent``)
    for OpenAI-compatible clients.  Treat that alias as "use the gateway
    default"; real model picker selections from the browser extension send the
    raw provider model id plus a provider slug and should override this turn.

    ``allow_bare_model`` controls whether a ``model`` value WITHOUT an
    accompanying ``provider`` is honored.  Generic OpenAI clients routinely
    hardcode model names ("gpt-4o", ...), and existing deployments rely on
    those falling back to the gateway default on the OpenAI-compatible
    surfaces — so those handlers pass the opt-in
    ``direct_model_requests`` config value here, while Hermes-native
    endpoints (session chat, /v1/runs) always allow it.  A request that
    sends an explicit ``provider`` is unambiguously Hermes-aware and is
    always honored.
    """
    if not isinstance(body, dict):
        return {}

    overrides: Dict[str, Any] = {}
    provider = _clean_request_string(body.get("provider"))
    if provider:
        overrides["requested_provider"] = provider

    model = _clean_request_string(body.get("model"))
    if model and model != virtual_model and (provider or allow_bare_model):
        overrides["requested_model"] = model

    model_options = body.get("model_options")
    if isinstance(model_options, dict):
        overrides["model_options"] = dict(model_options)
    return overrides


def _message_text_prefix(content: Any) -> str:
    if isinstance(content, str):
        return content[:128]
    if not isinstance(content, list):
        return ""
    parts: List[str] = []
    for item in content[:4]:
        if isinstance(item, str):
            parts.append(item)
        elif isinstance(item, dict):
            text = item.get("text")
            if isinstance(text, str):
                parts.append(text)
        if sum(len(part) for part in parts) >= 128:
            break
    return "\n".join(parts)[:128]


def _is_compressed_summary_message(message: Any) -> bool:
    if not isinstance(message, dict):
        return False
    if message.get(_COMPRESSED_SUMMARY_METADATA_KEY):
        return True
    prefix = _message_text_prefix(message.get("content"))
    return prefix.startswith("[CONTEXT COMPACTION") or prefix.startswith("[CONTEXT SUMMARY]:")


def _auto_truncate_response_history(
    conversation_history: List[Dict[str, Any]],
    *,
    limit: int = RESPONSES_AUTO_TRUNCATION_HISTORY_LIMIT,
) -> List[Dict[str, Any]]:
    """Keep recent Responses history without dropping the compaction handoff.

    Compaction summaries are preserved wherever they sit in the history —
    the gateway /compress path can leave them after a retained system head
    (see ``context_compressor`` force-user-leading handling), so a
    leading-block-only scan would silently drop them.
    """
    if limit <= 0 or len(conversation_history) <= limit:
        return conversation_history

    summary_indices = [
        index
        for index, message in enumerate(conversation_history)
        if _is_compressed_summary_message(message)
    ]
    if not summary_indices:
        return conversation_history[-limit:]

    kept_indices = set(summary_indices[:limit])
    remaining = limit - len(kept_indices)
    if remaining > 0:
        summary_index_set = set(summary_indices)
        for index in range(len(conversation_history) - 1, -1, -1):
            if index in summary_index_set:
                continue
            kept_indices.add(index)
            remaining -= 1
            if remaining <= 0:
                break

    return [conversation_history[index] for index in sorted(kept_indices)]


def _normalize_chat_content(
    content: Any, *, _max_depth: int = 10, _depth: int = 0,
) -> str:
    """Normalize OpenAI chat message content into a plain text string.

    Some clients (Open WebUI, LobeChat, etc.) send content as an array of
    typed parts instead of a plain string::

        [{"type": "text", "text": "hello"}, {"type": "input_text", "text": "..."}]

    This function flattens those into a single string so the agent pipeline
    (which expects strings) doesn't choke.

    Defensive limits prevent abuse: recursion depth, list size, and output
    length are all bounded.
    """
    if _depth > _max_depth:
        return ""
    if content is None:
        return ""
    if isinstance(content, str):
        return content[:MAX_NORMALIZED_TEXT_LENGTH] if len(content) > MAX_NORMALIZED_TEXT_LENGTH else content

    if isinstance(content, list):
        parts: List[str] = []
        total_len = 0
        items = content[:MAX_CONTENT_LIST_SIZE] if len(content) > MAX_CONTENT_LIST_SIZE else content
        for item in items:
            if isinstance(item, str):
                if item:
                    part = item[:MAX_NORMALIZED_TEXT_LENGTH]
                    parts.append(part)
                    total_len += len(part)
            elif isinstance(item, dict):
                item_type = str(item.get("type") or "").strip().lower()
                if item_type in {"text", "input_text", "output_text"}:
                    text = item.get("text", "")
                    if text:
                        try:
                            part = str(text)[:MAX_NORMALIZED_TEXT_LENGTH]
                            parts.append(part)
                            total_len += len(part)
                        except Exception:
                            pass
                # Silently skip image_url / other non-text parts
            elif isinstance(item, list):
                nested = _normalize_chat_content(item, _max_depth=_max_depth, _depth=_depth + 1)
                if nested:
                    parts.append(nested)
                    total_len += len(nested)
            # Check accumulated size
            if total_len >= MAX_NORMALIZED_TEXT_LENGTH:
                break
        result = "\n".join(parts)
        return result[:MAX_NORMALIZED_TEXT_LENGTH] if len(result) > MAX_NORMALIZED_TEXT_LENGTH else result

    # Fallback for unexpected types (int, float, bool, etc.)
    try:
        result = str(content)
        return result[:MAX_NORMALIZED_TEXT_LENGTH] if len(result) > MAX_NORMALIZED_TEXT_LENGTH else result
    except Exception:
        return ""


# Content part type aliases used by the OpenAI Chat Completions and Responses
# APIs.  We accept both spellings on input and emit a single canonical internal
# shape (``{"type": "text", ...}`` / ``{"type": "image_url", ...}``) that the
# rest of the agent pipeline already understands.
_TEXT_PART_TYPES = frozenset({"text", "input_text", "output_text"})
_IMAGE_PART_TYPES = frozenset({"image_url", "input_image"})
_FILE_PART_TYPES = frozenset({"file", "input_file"})


def _normalize_multimodal_content(content: Any) -> Any:
    """Validate and normalize multimodal content for the API server.

    Returns a plain string when the content is text-only, or a list of
    ``{"type": "text"|"image_url", ...}`` parts when images are present.
    The output shape is the native OpenAI Chat Completions vision format,
    which the agent pipeline accepts verbatim (OpenAI-wire providers) or
    converts (``_preprocess_anthropic_content`` for Anthropic).

    Raises ``ValueError`` with an OpenAI-style code on invalid input:
      * ``unsupported_content_type`` — file/input_file/file_id parts, or
        non-image ``data:`` URLs.
      * ``invalid_image_url`` — missing URL or unsupported scheme.
      * ``invalid_content_part`` — malformed text/image objects.

    Callers translate the ValueError into a 400 response.
    """
    # Scalar passthrough mirrors ``_normalize_chat_content``.
    if content is None:
        return ""
    if isinstance(content, str):
        return content[:MAX_NORMALIZED_TEXT_LENGTH] if len(content) > MAX_NORMALIZED_TEXT_LENGTH else content
    if not isinstance(content, list):
        # Mirror the legacy text-normalizer's fallback so callers that
        # pre-existed image support still get a string back.
        return _normalize_chat_content(content)

    items = content[:MAX_CONTENT_LIST_SIZE] if len(content) > MAX_CONTENT_LIST_SIZE else content
    normalized_parts: List[Dict[str, Any]] = []
    text_accum_len = 0

    for part in items:
        if isinstance(part, str):
            if part:
                trimmed = part[:MAX_NORMALIZED_TEXT_LENGTH]
                normalized_parts.append({"type": "text", "text": trimmed})
                text_accum_len += len(trimmed)
            continue

        if not isinstance(part, dict):
            # Ignore unknown scalars for forward compatibility with future
            # Responses API additions (e.g. ``refusal``).  The same policy
            # the text normalizer applies.
            continue

        raw_type = part.get("type")
        part_type = str(raw_type or "").strip().lower()

        if part_type in _TEXT_PART_TYPES:
            text = part.get("text")
            if text is None:
                continue
            if not isinstance(text, str):
                text = str(text)
            if text:
                trimmed = text[:MAX_NORMALIZED_TEXT_LENGTH]
                normalized_parts.append({"type": "text", "text": trimmed})
                text_accum_len += len(trimmed)
            continue

        if part_type in _IMAGE_PART_TYPES:
            detail = part.get("detail")
            image_ref = part.get("image_url")
            # OpenAI Responses sends ``input_image`` with a top-level
            # ``image_url`` string; Chat Completions sends ``image_url`` as
            # ``{"url": "...", "detail": "..."}``.  Support both.
            if isinstance(image_ref, dict):
                url_value = image_ref.get("url")
                detail = image_ref.get("detail", detail)
            else:
                url_value = image_ref
            if not isinstance(url_value, str) or not url_value.strip():
                raise ValueError("invalid_image_url:Image parts must include a non-empty image URL.")
            url_value = url_value.strip()
            lowered = url_value.lower()
            if lowered.startswith("data:"):
                if not lowered.startswith("data:image/") or "," not in url_value:
                    raise ValueError(
                        "unsupported_content_type:Only image data URLs are supported. "
                        "Non-image data payloads are not supported."
                    )
            elif not (lowered.startswith("http://") or lowered.startswith("https://")):
                raise ValueError(
                    "invalid_image_url:Image inputs must use http(s) URLs or data:image/... URLs."
                )
            image_part: Dict[str, Any] = {"type": "image_url", "image_url": {"url": url_value}}
            if detail is not None:
                if not isinstance(detail, str) or not detail.strip():
                    raise ValueError("invalid_content_part:Image detail must be a non-empty string when provided.")
                image_part["image_url"]["detail"] = detail.strip()
            normalized_parts.append(image_part)
            continue

        if part_type in _FILE_PART_TYPES:
            raise ValueError(
                "unsupported_content_type:Inline image inputs are supported, "
                "but uploaded files and document inputs are not supported on this endpoint."
            )

        # Unknown part type — reject explicitly so clients get a clear error
        # instead of a silently dropped turn.
        raise ValueError(
            f"unsupported_content_type:Unsupported content part type {raw_type!r}. "
            "Only text and image_url/input_image parts are supported."
        )

    if not normalized_parts:
        return ""

    # Text-only: collapse to a plain string so downstream logging/trajectory
    # code sees the native shape and prompt caching on text-only turns is
    # unaffected.
    if all(p.get("type") == "text" for p in normalized_parts):
        return "\n".join(p["text"] for p in normalized_parts if p.get("text"))

    return normalized_parts


def _content_has_visible_payload(content: Any) -> bool:
    """True when content has any text or image attachment.  Used to reject empty turns."""
    if isinstance(content, str):
        return bool(content.strip())
    if isinstance(content, list):
        for part in content:
            if isinstance(part, dict):
                ptype = str(part.get("type") or "").strip().lower()
                if ptype in _TEXT_PART_TYPES and str(part.get("text") or "").strip():
                    return True
                if ptype in _IMAGE_PART_TYPES:
                    return True
    return False


def _multimodal_validation_error(exc: ValueError, *, param: str) -> "web.Response":
    """Translate a ``_normalize_multimodal_content`` ValueError into a 400 response."""
    raw = str(exc)
    code, _, message = raw.partition(":")
    if not message:
        code, message = "invalid_content_part", raw
    return web.json_response(
        _openai_error(message, code=code, param=param),
        status=400,
    )


def _reap_disconnected_agent_processes(
    agent: Any, *, source: str = "api_server_sse_disconnect"
) -> None:
    """Reap background processes an abandoned API-server turn created.

    Mirrors the gateway-turn cleanup in ``gateway/run.py`` (#76115) for this
    API-server surface, which runs its own agent lifecycle via ``_run_agent``
    and never passes through ``TurnRunner`` — so it needs its own trigger for
    the same baseline-diff reap. Fire-and-forget on a daemon thread so the
    SSE handler's own cleanup isn't blocked on process-tree teardown.

    Reaping is epoch-gated: client-provided session IDs are conversation
    scopes, and multiple concurrent runs can intentionally share one (see
    ``_handle_runs``). Without the gate, run A disconnecting could kill a
    process a still-live run B (same task_id) spawned after A's baseline
    snapshot — the same stale-reaper bug class the gateway path gates via
    ``run_generation``. The epoch closure skips the reap when a newer run
    has since claimed the task_id; that newer run's own baseline covers its
    eventual cleanup.
    """
    process_task_id = getattr(agent, "_gateway_turn_process_task_id", "")
    process_baseline = getattr(agent, "_gateway_turn_process_baseline", None)
    if not process_task_id or process_baseline is None:
        return
    epoch = getattr(agent, "_gateway_turn_process_epoch", None)
    is_still_current: Optional[Any] = None
    if epoch is not None:
        def _epoch_still_current(_task_id=process_task_id, _epoch=epoch):
            # Skip only when a NEWER run has claimed this task_id. A missing
            # entry means the abandoned run's own clear pruned it (worker
            # returned after the interrupt) — no newer claimant exists, so
            # the reap must still proceed or the leak survives. This matches
            # the gateway gate's semantics: worker completion does not bump
            # run_generation either.
            with _TURN_PROCESS_EPOCH_LOCK:
                current = _TURN_PROCESS_EPOCHS.get(_task_id)
            return current is None or current == _epoch

        is_still_current = _epoch_still_current

    from gateway.run import _reap_gateway_turn_processes

    threading.Thread(
        target=_reap_gateway_turn_processes,
        args=(process_task_id, process_baseline),
        kwargs={"source": source, "is_still_current": is_still_current},
        name=f"api-turn-reaper-{process_task_id[:12]}",
        daemon=True,
    ).start()


# Per-task-id run epochs for the reap gate above. task_id is a conversation
# scope shared by concurrent API runs, so each run that claims it bumps the
# epoch; a reaper holding a stale epoch declines to kill. Epochs come from a
# single monotonic counter (never reused), so pruning an entry and later
# re-claiming the task_id can never resurrect a stale reaper's claim.
# Entries are pruned on clear when still current, bounding the dict to
# in-flight runs.
_TURN_PROCESS_EPOCHS: Dict[str, int] = {}
_TURN_PROCESS_EPOCH_LOCK = threading.Lock()
_TURN_PROCESS_EPOCH_COUNTER = itertools.count(1)


def _publish_turn_process_ownership(agent: Any, task_id: str) -> None:
    """Snapshot the process baseline and claim the task_id's current epoch.

    Single place all API-server agent lifecycles (chat/responses ``_run_agent``
    and ``/v1/runs``) record turn ownership, so the marker attribute names and
    epoch bookkeeping cannot drift between surfaces.
    """
    from tools.process_registry import process_registry

    with _TURN_PROCESS_EPOCH_LOCK:
        epoch = next(_TURN_PROCESS_EPOCH_COUNTER)
        _TURN_PROCESS_EPOCHS[task_id] = epoch
    agent._gateway_turn_process_task_id = task_id
    agent._gateway_turn_process_baseline = process_registry.snapshot_running_ids(
        task_id
    )
    agent._gateway_turn_process_epoch = epoch


def _clear_turn_process_ownership(agent: Any) -> None:
    """Clear turn ownership the moment the turn finishes (success or crash).

    A disconnect/cancel landing after this point must not reap background
    work the turn deliberately left running — mirrors the same race-window
    guard in ``gateway/run.py``'s ``_run_sync_with_timeout_lifecycle``.
    """
    task_id = getattr(agent, "_gateway_turn_process_task_id", "")
    epoch = getattr(agent, "_gateway_turn_process_epoch", None)
    if task_id and epoch is not None:
        with _TURN_PROCESS_EPOCH_LOCK:
            # Prune only when this run is still the current claimant; a
            # newer concurrent run owns the entry otherwise.
            if _TURN_PROCESS_EPOCHS.get(task_id) == epoch:
                del _TURN_PROCESS_EPOCHS[task_id]
    agent._gateway_turn_process_task_id = ""
    agent._gateway_turn_process_baseline = frozenset()
    agent._gateway_turn_process_epoch = None


def _session_chat_user_message(body: Dict[str, Any], *, param: str = "message") -> tuple[Any, Optional["web.Response"]]:
    """Parse and normalize session chat ``message`` / ``input`` like chat completions."""
    user_message = body.get("message") or body.get("input")
    if not _content_has_visible_payload(user_message):
        return None, web.json_response(
            _openai_error("Missing 'message' field", code="missing_message"),
            status=400,
        )
    try:
        return _normalize_multimodal_content(user_message), None
    except ValueError as exc:
        return None, _multimodal_validation_error(exc, param=param)


def check_api_server_requirements() -> bool:
    """Check if API server dependencies are available."""
    return AIOHTTP_AVAILABLE


# Keep the owner-history projection aligned with the Workspace conversation contract.
_OWNER_HISTORY_OWNER_MAX_CHARS = 12_000
_OWNER_HISTORY_RAPHAEL_MAX_CHARS = 50_000
_OWNER_CONVERSATION_RE = re.compile(
    r"raphael-owner-[a-f0-9]{32}(?:-[a-f0-9]{32})?"
)
_OWNER_RESPONSE_RE = re.compile(r"resp_[A-Za-z0-9_-]{8,128}")
# "The caller asserted nothing about the predecessor", which is distinct from
# asserting ``None`` ("this conversation must still have no turn").
_UNSTATED: Any = object()
_OWNER_CLAIM_RE = re.compile(r"claim_[A-Za-z0-9_-]{8,128}")
_OWNER_RUN_RE = re.compile(r"run_[a-f0-9]{32}")
_OWNER_PROFILE_RE = re.compile(r"[a-z0-9][a-z0-9_-]{0,63}")
_OWNER_SESSION_INDEX_LIMIT = 100
# This process's identity as the executor of durably queued owner work. A
# queued response or run lives in the store, but its only executor is an
# in-memory asyncio task, so a restart leaves the durable state with nobody
# driving it. The job row names the executor that owns it and carries a
# renewable lease; recovery reclaims a row whose lease has expired (or whose
# named process is provably gone, which is only a faster path to the same
# answer). Fencing on the lease rather than on pid liveness is what stops a
# recycled pid from stranding queued owner work forever.
_EXECUTOR_ID = uuid.uuid4().hex
# Jobs younger than this are never reaped, so a sibling gateway process that
# just reserved one is not mistaken for a dead executor on a pid-reuse hit.
_OWNER_JOB_REAP_MIN_AGE_SECONDS = 30.0
# How long one executor's claim on a job stands without a heartbeat. Renewed on
# every 60s sweep while the work is still running, so the margin is wide enough
# that a busy process is never mistaken for a dead one.
_OWNER_JOB_LEASE_SECONDS = 300.0
# Plain-English, non-technical: this text reaches the owner.
_OWNER_ORPHAN_RUN_MESSAGE = (
    "This stopped before it finished because Raphael restarted. Nothing was "
    "changed. You can ask for it again."
)
_OWNER_ORPHAN_RESPONSE_MESSAGE = (
    "This stopped before it finished because Raphael restarted. Please send "
    "your message again."
)
_OWNER_INTERRUPTED_TURN_MESSAGE = (
    "Raphael could not prepare a safe plan for this request. Nothing was "
    "changed. You can send it again."
)
_OWNER_RUN_STOPPED_MESSAGE = (
    "This stopped before it finished. Nothing was changed. You can ask for it "
    "again."
)
# The terminal states a run's durable row may record, and the full set a poller
# may read back from it.
_TERMINAL_RUN_STATUSES = frozenset({"completed", "failed", "cancelled"})
_DURABLE_RUN_STATUSES = frozenset({"queued"}) | _TERMINAL_RUN_STATUSES
# How many owner-visible turns one history projection carries. The reader
# fetches this many and reports whether more exist, so a caller is never handed
# a silently shortened transcript.
_OWNER_HISTORY_TURN_LIMIT = 40
_OWNER_CLAIM_LEASE_SECONDS = 300
_OWNER_CONVERSATION_RESERVATION_LEASE_SECONDS = 300
_OWNER_CONVERSATION_RESERVATION_RENEW_SECONDS = 60
# An interrupted request is recoverable until its caller acknowledges it, and
# for no shorter a time than that. Nothing else can tell the owner what became
# of what they sent: the turn was never published, and the browser holds no
# handle to it. A bounded lease meant an owner who came back late — or after a
# restart — found the request gone instead of answered, so the record simply
# does not expire. ``owner_conversation_recovery.expires_at`` is a NOT NULL
# column an older schema left behind: it is written so those rows stay valid,
# never read, and never used to hide or purge a record.
_OWNER_CONVERSATION_RECOVERY_NO_EXPIRY = float("inf")
_OWNER_PROPOSAL_MAX_MUTATIONS = 12
# The exact proposal schema versions that carry approval authority. Every
# created task must now name its ``execution_tier``, so only these versions can
# be committed: an older stored proposal stays readable (see
# ``_OWNER_HISTORY_SCHEMA_VERSIONS``) but is no longer actionable, because
# committing it would leave the native kernel resolving a route from a class
# the planner never stated.
_OWNER_NEW_PROPOSAL_SCHEMA = 3
_OWNER_EXISTING_PROPOSAL_SCHEMA = 4
# Every schema version whose structured assistant replies remain projectable in
# owner conversation history, including the pre-tier ones.
_OWNER_HISTORY_SCHEMA_VERSIONS = frozenset({1, 2, 3, 4})
_OWNER_NEW_PROPOSAL_KEYS = frozenset({
    "schema_version", "kind", "mode", "project_name",
    "project_description", "request_title", "summary", "project_size",
    "specification", "current_milestone", "owner_visible_result",
    "impact", "later_milestones", "tasks",
})
_OWNER_EXISTING_PROPOSAL_KEYS = frozenset({
    "schema_version", "kind", "mode", "request_title", "summary",
    "project_size", "specification", "current_milestone",
    "owner_visible_result", "impact", "later_milestones", "changes",
})
# Exact shape of one ``add`` change inside an actionable existing-project
# proposal. ``execution_tier`` is part of the authority: it is what the native
# kernel resolves the task's immutable route from, so a change object without
# it does not authorize anything.
_OWNER_PROPOSAL_ADD_KEYS = frozenset({
    "action", "reason", "title", "body", "assignee", "responsibility",
    "execution_tier", "existing_parent_refs", "new_parents",
})


class OwnerAuthorityBroken(RuntimeError):
    """The stored owner authority exists but cannot be read as authority.

    A conversation that never existed and a conversation whose stored response
    row is missing, whose JSON is corrupt, or whose transcript is malformed are
    not the same answer. The first is genuinely empty; the second is a service
    failure, and projecting it as "no history" invites a caller to plan a first
    turn against a conversation that already has turns nobody can read.
    """


class OwnerTurnNotRecoverable(RuntimeError):
    """One ending would have left neither a published turn nor a way back.

    A turn that stops without publishing has exactly two honest endings: it
    took the conversation's head, or its fence became the record carrying the
    owner's request. An ending that can do neither is not an ending at all —
    committing it would retire the durable job that says somebody must finish
    this response while destroying the only account of what was sent. The whole
    transaction rolls back instead, so the work stays queued and recoverable.
    """


class _OwnerNativeReceiptUnreadable(RuntimeError):
    """Whether one run's native mutation committed could not be decided.

    Distinct from "it committed nothing". A run whose external effect is
    undecided must not be reported failed and must not have its approval
    released, because both invite the owner to run the same mutation twice.
    """


class OwnerAuthorityUnavailable(RuntimeError):
    """Owner-authoritative state has no durable store, so nothing may proceed.

    An owner proposal claim, conversation closure, run attachment or run
    idempotency record IS authority: losing it does not merely lose history, it
    re-opens an approval that was already spent and lets the same owner
    mutation run twice. So when the configured durable store cannot be opened,
    those operations fail closed with one stable error rather than silently
    continuing against ephemeral in-memory state.
    """


def _response_store_locked(method):
    """Serialize runtime access to ResponseStore's shared SQLite connection."""

    @wraps(method)
    def locked(self, *args, **kwargs):
        with self._conversation_lock:
            return method(self, *args, **kwargs)

    return locked


def _owner_authority(method):
    """Gate one owner-authoritative store operation on durable storage."""

    @wraps(method)
    def guarded(self, *args, **kwargs):
        self._require_durable_owner_authority()
        return method(self, *args, **kwargs)

    return guarded


def _active_owner_profile() -> str:
    """Resolve the request-selected profile without accepting caller data."""
    profile = _api_request_profile.get()
    if not profile:
        try:
            from hermes_cli.profiles import get_active_profile_name

            profile = get_active_profile_name()
        except Exception:
            profile = "default"
    profile = str(profile or "default").strip().lower()
    return profile if _OWNER_PROFILE_RE.fullmatch(profile) else "default"


def _owner_conversation_group(name: Any) -> "tuple[Optional[str], Optional[str]]":
    """Split one owner conversation name into ``(group, session)``.

    ``session`` is ``None`` for the group's own (legacy) conversation. Returns
    ``(None, None)`` for anything that is not an owner conversation name.
    """
    if not isinstance(name, str) or _OWNER_CONVERSATION_RE.fullmatch(name) is None:
        return None, None
    parts = name.split("-")
    if len(parts) == 3:
        return name, None
    return "-".join(parts[:3]), parts[3]


def _owner_final_proposal(history: Any) -> "dict[str, Any] | None":
    """Return the exact final actionable assistant object, never an earlier draft."""
    if not isinstance(history, list):
        return None
    for message in reversed(history):
        if not isinstance(message, dict) or message.get("role") != "assistant":
            continue
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            continue
        try:
            candidate = json.loads(content)
        except (json.JSONDecodeError, TypeError):
            return None
        if not isinstance(candidate, dict):
            return None
        if (
            candidate.get("schema_version") == _OWNER_NEW_PROPOSAL_SCHEMA
            and candidate.get("kind") == "proposal"
            and candidate.get("mode") == "new"
            and set(candidate) == _OWNER_NEW_PROPOSAL_KEYS
            and isinstance(candidate.get("tasks"), list)
            and 1 <= len(candidate["tasks"]) <= _OWNER_PROPOSAL_MAX_MUTATIONS
        ) or (
            candidate.get("schema_version") == _OWNER_EXISTING_PROPOSAL_SCHEMA
            and candidate.get("kind") == "project_change_proposal"
            and candidate.get("mode") == "existing"
            and set(candidate) == _OWNER_EXISTING_PROPOSAL_KEYS
            and isinstance(candidate.get("changes"), list)
            and 1 <= len(candidate["changes"]) <= _OWNER_PROPOSAL_MAX_MUTATIONS
        ):
            return candidate
        return None
    return None


def _owner_proposal_digest(candidate: Any) -> "str | None":
    if not isinstance(candidate, dict):
        return None
    canonical = json.dumps(
        candidate, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _owner_history_has_actionable_final_proposal(
    history: List[Dict[str, Any]],
) -> bool:
    """Return whether the exact final assistant reply grants approval authority."""
    return _owner_final_proposal(history) is not None


def _pending_owner_message(value: Any) -> Optional[str]:
    """The plain-text request a reservation may carry, or nothing.

    Held to exactly the bound the projected owner turns are held to, so a
    request that is recorded here can also be shown. Anything else — a
    multimodal input, an empty message, one too long to project — records
    nothing rather than a partial or unprojectable request: the fence itself
    must never be refused over this, and a pending projection that is missing
    is honest, while one that is truncated is not.
    """
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text or len(text) > _OWNER_HISTORY_OWNER_MAX_CHARS:
        return None
    return text


def _owner_failure_reply_is_projectable(value: Any) -> bool:
    """Whether one structured failure is safe to show as a complete turn."""
    return (
        isinstance(value, dict)
        and set(value) == {"schema_version", "kind", "message"}
        and value.get("schema_version") == 1
        and value.get("kind") == "failure"
        and isinstance(value.get("message"), str)
        and bool(value["message"].strip())
        and len(value["message"]) <= 240
    )


class ResponseStore:
    """
    SQLite-backed LRU store for Responses API state.

    Each stored response includes the full internal conversation history
    (with tool calls and results) so it can be reconstructed on subsequent
    requests via previous_response_id.

    Persists across gateway restarts.  Falls back to in-memory SQLite
    if the on-disk path is unavailable.
    """

    def __init__(
        self,
        max_size: int = MAX_STORED_RESPONSES,
        db_path: str = None,
        *,
        default_profile: Optional[str] = None,
    ):
        self._max_size = max_size
        selected_profile = str(default_profile or _active_owner_profile()).strip().lower()
        self._default_profile = (
            selected_profile
            if _OWNER_PROFILE_RE.fullmatch(selected_profile)
            else "default"
        )
        if db_path is None:
            try:
                from hermes_cli.config import get_hermes_home
                db_path = str(get_hermes_home() / "response_store.db")
            except Exception:
                logger.error(
                    "Response store path could not be resolved; owner workspace "
                    "authority is unavailable in this process",
                    exc_info=True,
                )
                db_path = ":memory:"
        self._db_path: Optional[str] = db_path if db_path != ":memory:" else None
        try:
            self._conn = sqlite3.connect(db_path, check_same_thread=False)
        except Exception:
            # Generic Responses traffic may still be served from memory — a lost
            # cache entry there cannot authorize or duplicate anything. Owner
            # authority may not: ``_require_durable_owner_authority`` refuses
            # every owner operation for the life of this store instead of
            # letting a claim, closure or run identity live somewhere that
            # disappears on restart.
            logger.error(
                "Response store %s could not be opened; owner workspace "
                "authority is unavailable in this process", db_path,
                exc_info=True,
            )
            self._conn = sqlite3.connect(":memory:", check_same_thread=False)
            self._db_path = None
        self._conversation_lock = threading.RLock()
        # Use shared WAL-fallback helper so response_store.db degrades
        # gracefully on NFS/SMB/FUSE-mounted HERMES_HOME (same filesystem
        # issue addressed for state.db/kanban.db — see
        # hermes_state._WAL_INCOMPAT_MARKERS).
        from hermes_state import apply_wal_with_fallback
        apply_wal_with_fallback(self._conn, db_label="response_store.db")
        self._initialize_schema()
        # response_store.db contains conversation history (tool payloads,
        # prompts, results). Tighten to owner-only after creation so other
        # local users on a shared box can't read it. Run once at __init__
        # rather than after every commit — chmod-on-every-write is wasted
        # syscalls on a hot path.
        self._tighten_file_permissions()

    def _initialize_schema(self) -> None:
        """Serialize transactional schema upgrades across gateway processes."""
        lock_markers = (
            "database is locked",
            "database table is locked",
            "database schema is locked",
        )
        for attempt in range(5):
            try:
                self._conn.execute("BEGIN EXCLUSIVE")
                self._initialize_schema_locked()
                self._conn.commit()
                return
            except sqlite3.OperationalError as exc:
                self._conn.rollback()
                if (
                    attempt == 4
                    or not any(marker in str(exc).lower() for marker in lock_markers)
                ):
                    raise
                time.sleep(0.05 * (attempt + 1))
            except Exception:
                self._conn.rollback()
                raise
        raise RuntimeError("response store schema migration did not complete")

    def _initialize_schema_locked(self) -> None:
        """Create and migrate every ResponseStore table under one DB fence."""
        self._conn.execute(
            """CREATE TABLE IF NOT EXISTS responses (
                profile TEXT NOT NULL,
                response_id TEXT NOT NULL,
                data TEXT NOT NULL,
                accessed_at REAL NOT NULL,
                PRIMARY KEY (profile, response_id)
            )"""
        )
        response_columns = {
            str(row[1]) for row in self._conn.execute("PRAGMA table_info(responses)")
        }
        if "profile" not in response_columns:
            self._conn.execute("ALTER TABLE responses ADD COLUMN profile TEXT")
            self._conn.execute(
                "UPDATE responses SET profile = ? WHERE profile IS NULL OR profile = ''",
                (self._default_profile,),
            )
        self._ensure_response_schema()
        self._ensure_conversation_schema()
        self._ensure_closure_schema()
        self._conn.execute(
            """CREATE TABLE IF NOT EXISTS owner_conversation_reservations (
                profile TEXT NOT NULL,
                name TEXT NOT NULL,
                response_id TEXT NOT NULL,
                created_at REAL NOT NULL,
                expires_at REAL NOT NULL,
                PRIMARY KEY (profile, name)
            )"""
        )
        reservation_columns = {
            str(row[1])
            for row in self._conn.execute(
                "PRAGMA table_info(owner_conversation_reservations)"
            )
        }
        if "expires_at" not in reservation_columns:
            self._conn.execute(
                "ALTER TABLE owner_conversation_reservations ADD COLUMN expires_at REAL"
            )
            self._conn.execute(
                "UPDATE owner_conversation_reservations SET expires_at = created_at + ? "
                "WHERE expires_at IS NULL",
                (_OWNER_CONVERSATION_RESERVATION_LEASE_SECONDS,),
            )
        if "owner_message" not in reservation_columns:
            # The request this reservation is planning, durable from before any
            # model runs. A row written by an older build carries none, which
            # reads as "not projectable" rather than as an empty request.
            self._conn.execute(
                "ALTER TABLE owner_conversation_reservations "
                "ADD COLUMN owner_message TEXT"
            )
        # What is left of a turn that ended without ever publishing one. The
        # reservation above is released by that ending, so it cannot be what a
        # caller recovers from; this row replaces it in the same write and
        # stands until the caller says it holds both the request and the
        # outcome. One row per conversation, because while it stands the fence
        # refuses to start another turn here at all: a second request must not
        # be able to bury the one that was never answered.
        #
        # ``expires_at`` is legacy. It is still written so a database created by
        # an older build keeps satisfying its NOT NULL, and it is read nowhere:
        # nothing about this record's age may hide or purge it.
        self._conn.execute(
            """CREATE TABLE IF NOT EXISTS owner_conversation_recovery (
                profile TEXT NOT NULL,
                name TEXT NOT NULL,
                response_id TEXT NOT NULL,
                owner_message TEXT,
                created_at REAL NOT NULL,
                expires_at REAL NOT NULL,
                PRIMARY KEY (profile, name)
            )"""
        )
        self._conn.execute(
            """CREATE TABLE IF NOT EXISTS run_idempotency (
                profile TEXT NOT NULL,
                session_scope TEXT NOT NULL,
                idempotency_key TEXT NOT NULL,
                fingerprint TEXT NOT NULL,
                run_id TEXT NOT NULL,
                created_at REAL NOT NULL,
                status_json TEXT,
                terminal_json TEXT,
                PRIMARY KEY (profile, session_scope, idempotency_key)
            )"""
        )
        run_idempotency_columns = {
            str(row[1])
            for row in self._conn.execute("PRAGMA table_info(run_idempotency)")
        }
        if "terminal_json" not in run_idempotency_columns:
            self._conn.execute(
                "ALTER TABLE run_idempotency ADD COLUMN terminal_json TEXT"
            )
        if "status_json" not in run_idempotency_columns:
            self._conn.execute(
                "ALTER TABLE run_idempotency ADD COLUMN status_json TEXT"
            )
        self._conn.execute(
            """CREATE TABLE IF NOT EXISTS owner_response_idempotency (
                profile TEXT NOT NULL,
                session_scope TEXT NOT NULL,
                idempotency_key TEXT NOT NULL,
                fingerprint TEXT NOT NULL,
                conversation TEXT NOT NULL,
                response_id TEXT NOT NULL,
                state TEXT NOT NULL,
                replay_json TEXT,
                session_id TEXT,
                created_at REAL NOT NULL,
                PRIMARY KEY (profile, session_scope, idempotency_key)
            )"""
        )
        self._conn.execute(
            """CREATE TABLE IF NOT EXISTS owner_conversation_sessions (
                profile TEXT NOT NULL,
                group_name TEXT NOT NULL,
                name TEXT NOT NULL,
                seq INTEGER NOT NULL,
                created_at REAL NOT NULL,
                PRIMARY KEY (profile, name)
            )"""
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS owner_conversation_sessions_group "
            "ON owner_conversation_sessions (profile, group_name, seq)"
        )
        self._backfill_owner_conversation_sessions()
        self._conn.execute(
            """CREATE TABLE IF NOT EXISTS owner_executor_jobs (
                kind TEXT NOT NULL,
                job_key TEXT NOT NULL,
                profile TEXT NOT NULL,
                executor_id TEXT NOT NULL,
                executor_pid INTEGER NOT NULL,
                payload TEXT NOT NULL,
                created_at REAL NOT NULL,
                PRIMARY KEY (kind, job_key)
            )"""
        )
        job_columns = {
            str(row[1])
            for row in self._conn.execute("PRAGMA table_info(owner_executor_jobs)")
        }
        if "lease_expires_at" not in job_columns:
            # A renewable lease, not PID liveness, is what fences an executor.
            # A recycled PID looks alive forever, so a job whose owner died
            # could be stranded permanently; a lease expires on its own. Rows
            # written by an older build start with no lease, which reads as
            # already expired — correct, since that process is long gone.
            self._conn.execute(
                "ALTER TABLE owner_executor_jobs ADD COLUMN lease_expires_at REAL"
            )
        self._backfill_owner_proposals()

    def _ensure_response_schema(self) -> None:
        """Migrate stored responses to an exact profile-scoped identity."""
        info = self._conn.execute("PRAGMA table_info(responses)").fetchall()
        pk_columns = [
            str(row[1]) for row in sorted(info, key=lambda row: int(row[5]))
            if int(row[5]) > 0
        ]
        if pk_columns == ["profile", "response_id"]:
            return
        self._conn.execute("DROP TABLE IF EXISTS responses_profile_migration")
        self._conn.execute(
            """CREATE TABLE responses_profile_migration (
                profile TEXT NOT NULL,
                response_id TEXT NOT NULL,
                data TEXT NOT NULL,
                accessed_at REAL NOT NULL,
                PRIMARY KEY (profile, response_id)
            )"""
        )
        for response_id, profile, data, accessed_at in self._conn.execute(
            "SELECT response_id, profile, data, accessed_at FROM responses"
        ).fetchall():
            selected = str(profile or self._default_profile).strip().lower()
            if _OWNER_PROFILE_RE.fullmatch(selected) is None:
                selected = self._default_profile
            self._conn.execute(
                "INSERT OR IGNORE INTO responses_profile_migration "
                "(profile, response_id, data, accessed_at) VALUES (?, ?, ?, ?)",
                (selected, response_id, data, accessed_at),
            )
        self._conn.execute("DROP TABLE responses")
        self._conn.execute(
            "ALTER TABLE responses_profile_migration RENAME TO responses"
        )

    def _ensure_conversation_schema(self) -> None:
        """Migrate conversation authority to a profile-scoped composite key."""
        desired = """CREATE TABLE {table} (
            profile TEXT NOT NULL,
            name TEXT NOT NULL,
            response_id TEXT NOT NULL,
            proposal_response_id TEXT,
            proposal_digest TEXT,
            consumed_response_id TEXT,
            claimed_response_id TEXT,
            claim_id TEXT,
            claim_expires_at REAL,
            owner_run_id TEXT,
            bound_operation TEXT,
            bound_payload_digest TEXT,
            claim_state TEXT,
            closed INTEGER NOT NULL DEFAULT 0,
            PRIMARY KEY (profile, name)
        )"""
        info = self._conn.execute("PRAGMA table_info(conversations)").fetchall()
        if not info:
            self._conn.execute(desired.format(table="conversations"))
            return
        columns = {str(row[1]) for row in info}
        pk_columns = [
            str(row[1]) for row in sorted(info, key=lambda row: int(row[5]))
            if int(row[5]) > 0
        ]
        if "profile" not in columns or pk_columns != ["profile", "name"]:
            self._conn.execute("DROP TABLE IF EXISTS conversations_profile_migration")
            self._conn.execute(desired.format(table="conversations_profile_migration"))
            selectable = [
                name for name in (
                    "name", "response_id", "proposal_response_id", "consumed_response_id",
                    "claimed_response_id", "claim_id", "owner_run_id", "claim_state", "closed",
                ) if name in columns
            ]
            for row in self._conn.execute(
                f"SELECT {', '.join(selectable)} FROM conversations"
            ).fetchall():
                record = dict(zip(selectable, row))
                self._conn.execute(
                    "INSERT OR IGNORE INTO conversations_profile_migration ("
                    "profile, name, response_id, proposal_response_id, consumed_response_id, "
                    "claimed_response_id, claim_id, owner_run_id, claim_state, closed"
                    ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        self._default_profile,
                        record.get("name"),
                        record.get("response_id"),
                        record.get("proposal_response_id"),
                        record.get("consumed_response_id"),
                        record.get("claimed_response_id"),
                        record.get("claim_id"),
                        record.get("owner_run_id"),
                        record.get("claim_state"),
                        int(bool(record.get("closed"))),
                    ),
                )
            self._conn.execute("DROP TABLE conversations")
            self._conn.execute(
                "ALTER TABLE conversations_profile_migration RENAME TO conversations"
            )
            return
        for column, declaration in (
            ("proposal_response_id", "TEXT"),
            ("proposal_digest", "TEXT"),
            ("consumed_response_id", "TEXT"),
            ("claimed_response_id", "TEXT"),
            ("claim_id", "TEXT"),
            ("claim_expires_at", "REAL"),
            ("owner_run_id", "TEXT"),
            ("bound_operation", "TEXT"),
            ("bound_payload_digest", "TEXT"),
            ("claim_state", "TEXT"),
            ("closed", "INTEGER NOT NULL DEFAULT 0"),
        ):
            if column not in columns:
                self._conn.execute(
                    f"ALTER TABLE conversations ADD COLUMN {column} {declaration}"
                )

    def _ensure_closure_schema(self) -> None:
        desired = """CREATE TABLE {table} (
            profile TEXT NOT NULL,
            name TEXT NOT NULL,
            closed_at REAL NOT NULL,
            PRIMARY KEY (profile, name)
        )"""
        info = self._conn.execute(
            "PRAGMA table_info(owner_conversation_closures)"
        ).fetchall()
        if not info:
            self._conn.execute(desired.format(table="owner_conversation_closures"))
            return
        columns = {str(row[1]) for row in info}
        pk_columns = [
            str(row[1]) for row in sorted(info, key=lambda row: int(row[5]))
            if int(row[5]) > 0
        ]
        if "profile" in columns and pk_columns == ["profile", "name"]:
            return
        self._conn.execute("DROP TABLE IF EXISTS owner_closures_profile_migration")
        self._conn.execute(desired.format(table="owner_closures_profile_migration"))
        for name, closed_at in self._conn.execute(
            "SELECT name, closed_at FROM owner_conversation_closures"
        ).fetchall():
            self._conn.execute(
                "INSERT OR IGNORE INTO owner_closures_profile_migration "
                "(profile, name, closed_at) VALUES (?, ?, ?)",
                (self._default_profile, name, closed_at),
            )
        self._conn.execute("DROP TABLE owner_conversation_closures")
        self._conn.execute(
            "ALTER TABLE owner_closures_profile_migration "
            "RENAME TO owner_conversation_closures"
        )

    def _backfill_owner_conversation_sessions(self) -> None:
        """Give every already-mapped owner conversation its durable sequence.

        A one-time seed for stores written before this table existed. The seed
        order is the only signal such a store has — the group's own conversation
        first (it is by construction the oldest), then each sibling by the LRU
        access time of its mapped response. That timestamp is exactly what this
        sequence exists to stop depending on, which is why it is read ONCE here
        and never again: from this point the sequence is immutable, so a later
        read of an old session cannot reorder anything.
        """
        rows = self._conn.execute(
            "SELECT c.profile, c.name, r.accessed_at FROM conversations c "
            "LEFT JOIN responses r ON r.profile = c.profile "
            "AND r.response_id = c.response_id "
            "WHERE c.name NOT IN ("
            "  SELECT name FROM owner_conversation_sessions s "
            "  WHERE s.profile = c.profile"
            ")"
        ).fetchall()
        seeded: List[tuple] = []
        for profile, name, accessed_at in rows:
            group, session = _owner_conversation_group(name)
            if group is None:
                continue
            order = (
                float(accessed_at)
                if isinstance(accessed_at, (int, float))
                and not isinstance(accessed_at, bool)
                and math.isfinite(float(accessed_at))
                else 0.0
            )
            seeded.append((str(profile), group, str(name), session is None, order))
        # Within each group: its legacy (group-named) conversation first — it is
        # by construction the oldest — then oldest access first, so the newest
        # sibling ends up with the highest sequence.
        seeded.sort(
            key=lambda item: (
                item[0], item[1], 0 if item[3] else 1, item[4], item[2],
            )
        )
        for profile, group, name, _is_legacy, _order in seeded:
            self._record_owner_session_locked(profile, group, name)

    def _record_owner_session_locked(
        self, profile: str, group: str, name: str,
    ) -> None:
        """Assign this owner conversation its immutable place in its group.

        Assigned once, when the conversation is first mapped, and never changed.
        That is what makes "which session is current" a durable fact rather than
        a side effect of which response was read most recently: an owner opening
        an old change no longer promotes it, and a bounded read can only ever cut
        the OLDEST sessions.
        """
        self._conn.execute(
            "INSERT INTO owner_conversation_sessions "
            "(profile, group_name, name, seq, created_at) "
            "SELECT ?, ?, ?, COALESCE(MAX(seq), 0) + 1, ? "
            "FROM owner_conversation_sessions WHERE profile = ? AND group_name = ? "
            "ON CONFLICT(profile, name) DO NOTHING",
            (profile, group, name, time.time(), profile, group),
        )

    def _backfill_owner_proposals(self) -> None:
        """Recover only structurally actionable final proposals from old rows."""
        rows = self._conn.execute(
            "SELECT profile, name, response_id FROM conversations "
            "WHERE proposal_response_id IS NULL"
        ).fetchall()
        for profile, name, response_id in rows:
            if not isinstance(name, str) or _OWNER_CONVERSATION_RE.fullmatch(name) is None:
                continue
            stored = self._conn.execute(
                "SELECT data FROM responses WHERE response_id = ? AND profile = ?",
                (response_id, profile),
            ).fetchone()
            if stored is None:
                continue
            try:
                raw = json.loads(stored[0])
            except (json.JSONDecodeError, TypeError):
                continue
            candidate = _owner_final_proposal(
                raw.get("conversation_history") if isinstance(raw, dict) else None
            )
            digest = _owner_proposal_digest(candidate)
            if digest is None:
                continue
            self._conn.execute(
                "UPDATE conversations SET proposal_response_id = ?, proposal_digest = ? "
                "WHERE profile = ? AND name = ? AND response_id = ? "
                "AND proposal_response_id IS NULL",
                (response_id, digest, profile, name, response_id),
            )

    def _tighten_file_permissions(self) -> None:
        """Force owner-only permissions on the DB and SQLite sidecars."""
        if not self._db_path:
            return
        for candidate in (
            Path(self._db_path),
            Path(f"{self._db_path}-wal"),
            Path(f"{self._db_path}-shm"),
        ):
            try:
                if candidate.exists():
                    candidate.chmod(0o600)
            except OSError:
                logger.debug(
                    "Failed to restrict response store permissions for %s",
                    candidate,
                    exc_info=True,
                )

    def _require_durable_owner_authority(self) -> None:
        """Fail closed unless owner authority is backed by durable storage.

        ``_db_path`` is None only when the configured on-disk store could not
        be opened (or was never resolved) and the constructor fell back to an
        in-memory connection. That fallback is fine for the generic Responses
        cache; for owner state it would silently drop proposal claims,
        conversation closures and run idempotency on restart, so every owner
        operation refuses here instead. One stable error, checked per call, so
        a store that starts durable and is later replaced does not need a
        second code path.

        Checked per call for a second reason: a store that WAS durable can stop
        being so while the process runs. If the file is unlinked underneath us,
        SQLite keeps serving the open inode for reads — and vanishes at the next
        restart. An owner claim, closure or run identity recorded there is not
        durable, so the same refusal applies.
        """
        if self._db_path is None:
            raise OwnerAuthorityUnavailable(
                "the owner workspace store is unavailable"
            )
        if not os.path.exists(self._db_path):
            logger.error(
                "Response store %s is no longer on disk; owner workspace "
                "authority is unavailable in this process", self._db_path,
            )
            self._demote_to_memory()
            raise OwnerAuthorityUnavailable(
                "the owner workspace store is unavailable"
            )

    def _demote_to_memory(self) -> None:
        """Enter the same no-durable-authority state the constructor falls back to.

        The connection is still attached to the vanished inode, and SQLite
        refuses to write a database file it has seen move or disappear
        (``SQLITE_READONLY_DBMOVED`` — "attempt to write a readonly database").
        Left as-is that breaks the generic Responses cache too, which is not
        authority and has no reason to fail. Rebinding to ``:memory:`` keeps
        that traffic serving and makes the owner refusal permanent for this
        store: a path that has already proven it does not persist never regains
        authority just because a new file appears there.
        """
        with self._conversation_lock:
            if self._db_path is None:
                return
            self._db_path = None
            try:
                self._conn.close()
            except Exception:
                logger.debug(
                    "Failed to close the vanished response store connection",
                    exc_info=True,
                )
            self._conn = sqlite3.connect(":memory:", check_same_thread=False)
            self._initialize_schema()

    def _profile(self, profile: Optional[str]) -> str:
        selected = str(profile or self._default_profile).strip().lower()
        if _OWNER_PROFILE_RE.fullmatch(selected) is None:
            raise ValueError("invalid profile scope")
        return selected

    @_response_store_locked
    def get(
        self, response_id: str, *, profile: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Retrieve a stored response by ID (updates access time for LRU)."""
        profile = self._profile(profile)
        row = self._conn.execute(
            "SELECT data FROM responses WHERE response_id = ? AND profile = ?",
            (response_id, profile),
        ).fetchone()
        if row is None:
            return None
        self._conn.execute(
            "UPDATE responses SET accessed_at = ? WHERE response_id = ? AND profile = ?",
            (time.time(), response_id, profile),
        )
        self._conn.commit()
        try:
            return json.loads(row[0])
        except (json.JSONDecodeError, TypeError):
            logger.warning(
                "Corrupted JSON in response store for id=%s, evicting entry",
                response_id,
            )
            self._conn.execute(
                "DELETE FROM responses WHERE response_id = ? AND profile = ?",
                (response_id, profile),
            )
            self._conn.commit()
            return None

    @_response_store_locked
    def put(
        self, response_id: str, data: Dict[str, Any], *, profile: Optional[str] = None,
    ) -> None:
        """Store a response, evicting the oldest if at capacity."""
        self._put_response_locked(response_id, data, self._profile(profile))
        self._conn.commit()

    def _put_response_locked(
        self, response_id: str, data: Dict[str, Any], profile: str,
    ) -> None:
        """Write one response row and its eviction, WITHOUT committing.

        Split out so an owner turn can publish its response, its conversation
        head, its reservation release and its durable replay record inside ONE
        transaction (see :meth:`publish_owner_turn`). As four separate commits,
        a crash between any two left a turn that had happened but could not be
        replayed, or a head with no response behind it.
        """
        self._conn.execute(
            "INSERT OR REPLACE INTO responses "
            "(response_id, profile, data, accessed_at) VALUES (?, ?, ?, ?)",
            (response_id, profile, json.dumps(data, default=str), time.time()),
        )
        # Evict oldest entries beyond max_size
        count = self._conn.execute(
            "SELECT COUNT(*) FROM responses WHERE profile = ?", (profile,),
        ).fetchone()[0]
        if count > self._max_size:
            # A current owner conversation is durable workflow authority, not
            # an LRU cache entry. Keep its latest response (and the response
            # being inserted, which may be mapped immediately after put()) and
            # evict only ordinary responses. If durable owner sessions alone
            # exceed the cache bound, correctness wins over the soft LRU cap.
            owner_response_ids = {
                mapped_response_id
                for mapped_response_id, name in self._conn.execute(
                    "SELECT response_id, name FROM conversations WHERE profile = ?",
                    (profile,),
                ).fetchall()
                if isinstance(name, str)
                and _OWNER_CONVERSATION_RE.fullmatch(name) is not None
            }
            evict_ids = []
            for (candidate_id,) in self._conn.execute(
                "SELECT response_id FROM responses WHERE profile = ? "
                "ORDER BY accessed_at ASC",
                (profile,),
            ).fetchall():
                if candidate_id == response_id or candidate_id in owner_response_ids:
                    continue
                evict_ids.append(candidate_id)
                if len(evict_ids) >= count - self._max_size:
                    break
            if evict_ids:
                placeholders = ",".join("?" for _ in evict_ids)
                # Clear conversation mappings pointing to evicted responses
                self._conn.execute(
                    f"DELETE FROM conversations WHERE profile = ? "
                    f"AND response_id IN ({placeholders})",
                    [profile, *evict_ids],
                )
                # Delete evicted responses
                self._conn.execute(
                    f"DELETE FROM responses WHERE profile = ? "
                    f"AND response_id IN ({placeholders})",
                    [profile, *evict_ids],
                )

    @_response_store_locked
    def delete(self, response_id: str, *, profile: Optional[str] = None) -> bool:
        """Remove a response from the store. Returns True if found and deleted."""
        profile = self._profile(profile)
        # Deleting the latest response of an owner conversation would also
        # delete its approval/consumption fence. Those rows are durable native
        # authority and may only move through the exact owner lifecycle.
        if self.owner_response_is_current(response_id, profile=profile):
            return False
        # Clear conversation mappings pointing to this response
        self._conn.execute(
            "DELETE FROM conversations WHERE response_id = ? AND profile = ?",
            (response_id, profile),
        )
        cursor = self._conn.execute(
            "DELETE FROM responses WHERE response_id = ? AND profile = ?",
            (response_id, profile),
        )
        self._conn.commit()
        return cursor.rowcount > 0

    @_response_store_locked
    def owner_response_is_current(
        self, response_id: str, *, profile: Optional[str] = None,
    ) -> bool:
        """Return whether a response currently anchors exact owner authority."""
        profile = self._profile(profile)
        rows = self._conn.execute(
            "SELECT name FROM conversations WHERE response_id = ? AND profile = ?",
            (response_id, profile),
        ).fetchall()
        return any(
            isinstance(row[0], str)
            and _OWNER_CONVERSATION_RE.fullmatch(row[0]) is not None
            for row in rows
        )

    @_response_store_locked
    def get_conversation(
        self, name: str, *, profile: Optional[str] = None,
    ) -> Optional[str]:
        """Get the latest response_id for a conversation name."""
        profile = self._profile(profile)
        row = self._conn.execute(
            "SELECT response_id FROM conversations WHERE profile = ? AND name = ?",
            (profile, name),
        ).fetchone()
        return row[0] if row else None

    @_owner_authority
    @_response_store_locked
    def owner_history_snapshot(
        self, name: str, *, profile: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Project one conversation into owner-safe turns and its recovery handle.

        The stored transcript remains the native Responses authority.  This
        projection never returns system instructions, tools, intermediate
        assistant text, usage, sessions, or private reasoning.  The opaque
        handle stays service-to-service so callers can re-read and validate
        the stored response before granting any approval authority.

        Absent and BROKEN authority are answered differently. A conversation
        with no mapping is the empty snapshot; one whose mapped response row is
        missing, whose stored JSON is corrupt, whose transcript is not a list,
        or which carries a malformed authoritative turn raises
        :class:`OwnerAuthorityBroken`, which the endpoint reports as a service
        failure. Returning the same empty snapshot for both invited a caller to
        plan a first turn against a conversation that already had turns.
        """
        empty: Dict[str, Any] = {
            "head_response_id": None,
            "latest_response_id": None,
            "proposal_consumed": False,
            "proposal_claimed": False,
            "active_run_id": None,
            "completed_run_id": None,
            "conversation_closed": False,
            "truncated": False,
            "incomplete": False,
            "pending": None,
            "recovery": None,
            "data": [],
        }
        profile = self._profile(profile)
        if (
            not isinstance(name, str)
            or _OWNER_CONVERSATION_RE.fullmatch(name) is None
        ):
            return empty
        # The turn that is being planned right now, if one is, and the last one
        # that ended without ever becoming a turn. Read first and for every
        # outcome below — including the empty snapshot, which is the very first
        # turn of a new Project and therefore exactly the case whose lost accept
        # response left the owner with nothing.
        pending = self._pending_owner_turn(profile, name)
        recovery = self._owner_conversation_recovery(profile, name)
        empty["pending"] = pending
        empty["recovery"] = recovery
        mapping = self._conn.execute(
            "SELECT response_id FROM conversations WHERE profile = ? AND name = ?",
            (profile, name),
        ).fetchone()
        row = self._conn.execute(
            "SELECT c.response_id, c.proposal_response_id, c.consumed_response_id, "
            "c.claimed_response_id, c.owner_run_id, c.claim_state, c.closed, r.data "
            "FROM conversations c "
            "JOIN responses r ON r.response_id = c.response_id "
            "WHERE c.profile = ? AND c.name = ? AND r.profile = c.profile",
            (profile, name),
        ).fetchone()
        if row is None:
            if mapping is not None:
                # The conversation IS mapped, but the response it names is gone.
                # That is a broken authority, not an empty conversation.
                raise OwnerAuthorityBroken(
                    f"owner conversation {name} maps a response that is missing"
                )
            if self._conn.execute(
                "SELECT 1 FROM owner_conversation_closures "
                "WHERE profile = ? AND name = ?",
                (profile, name),
            ).fetchone() is not None:
                empty["conversation_closed"] = True
            return empty
        try:
            stored = json.loads(row[7])
        except (json.JSONDecodeError, TypeError) as exc:
            raise OwnerAuthorityBroken(
                f"owner conversation {name} has an unreadable stored response"
            ) from exc
        history = stored.get("conversation_history") if isinstance(stored, dict) else None
        if not isinstance(history, list):
            raise OwnerAuthorityBroken(
                f"owner conversation {name} has no readable transcript"
            )

        from agent.redact import redact_sensitive_text

        turns: List[Dict[str, str]] = []
        owner_text: Optional[str] = None
        raphael_text: Optional[str] = None
        incomplete = False
        owner_turn_truncated = False
        # Whether the owner turn being read has assistant output that is not a
        # Raphael reply at all. Reset at every owner boundary and read exactly
        # once, for the trailing turn, below the loop.
        unreadable_reply = False
        substituted_failure_turn = False

        def _flush() -> None:
            nonlocal owner_text, raphael_text, incomplete
            if owner_text is not None and raphael_text is not None:
                turns.append({"owner": owner_text, "raphael": raphael_text})
            elif owner_text is not None:
                # An owner turn with no projectable reply before the next turn
                # boundary — consecutive owner messages, or a trailing one. The
                # turn really happened, so it is NOT silently dropped: the
                # projection is marked incomplete, which is what stops a caller
                # presenting the remaining turns as the whole conversation.
                incomplete = True
            owner_text = None
            raphael_text = None

        # An OWNER message is a turn BOUNDARY, so it is authoritative in a way
        # an assistant message is not: dropping one silently merges two owner
        # turns into one and shows the owner a transcript that never happened.
        # Every one of them therefore has to be projectable, or the whole
        # projection fails. The assistant side keeps selecting the last
        # structured reply and skipping everything else — tool traffic and
        # intermediate text are legitimately not owner-visible, and a reply
        # kind this projection deliberately excludes (an Automations
        # ``automation_proposal``) is not a defect either.
        for message in history:
            if not isinstance(message, dict):
                # A stored transcript is written by this service; a record that
                # is not an object at all is corrupt authority, not a message
                # to skip past. Skipping it could merge two owner turns.
                raise OwnerAuthorityBroken(
                    f"owner conversation {name} has an unreadable transcript record"
                )
            role = message.get("role")
            content = message.get("content")
            if role == "user":
                if not isinstance(content, str) or not content.strip():
                    raise OwnerAuthorityBroken(
                        f"owner conversation {name} has an unreadable owner turn"
                    )
                text = content.strip()
                _flush()
                unreadable_reply = False
                if len(text) > _OWNER_HISTORY_OWNER_MAX_CHARS:
                    text = (
                        "[Earlier owner message omitted from this view because it "
                        "exceeded the safe display limit. The native record remains "
                        "unchanged.]"
                    )
                    owner_turn_truncated = True
                owner_text = redact_sensitive_text(text, force=True)
                continue
            if role != "assistant" or owner_text is None:
                continue
            if not isinstance(content, str):
                # Output that is not text at all still answered this turn with
                # something no reader can turn into an outcome.
                unreadable_reply = unreadable_reply or bool(content)
                continue
            text = content.strip()
            if not text:
                continue
            try:
                candidate = json.loads(text)
            except (json.JSONDecodeError, TypeError, ValueError, RecursionError):
                # Raw model prose — a plain-text imitation of a tool call is the
                # shape that stranded the Workspace. Only the FACT that it
                # happened is carried forward; the text itself never is.
                unreadable_reply = True
                continue
            if (
                not isinstance(candidate, dict)
                or candidate.get("schema_version")
                not in _OWNER_HISTORY_SCHEMA_VERSIONS
                or not isinstance(candidate.get("kind"), str)
            ):
                # Structured, but not a versioned Raphael reply: it names no
                # kind this service ever writes, so it carries no outcome.
                unreadable_reply = True
                continue
            if (
                candidate["kind"]
                not in {"question", "proposal", "project_change_proposal"}
                and not _owner_failure_reply_is_projectable(candidate)
            ):
                # A real reply of a kind this projection deliberately excludes
                # (an Automations ``automation_proposal``). Recognised
                # authority, just not owner-visible — not a turn to complete.
                continue
            # Last valid structured assistant message before the next owner
            # turn is the authoritative final reply for that turn.
            from hermes_cli.kanban_db import redact_review_value

            projected_text = json.dumps(
                redact_review_value(candidate), ensure_ascii=False,
            )
            if (
                len(text) > _OWNER_HISTORY_RAPHAEL_MAX_CHARS
                or len(projected_text) > _OWNER_HISTORY_RAPHAEL_MAX_CHARS
            ):
                # This IS the turn's authoritative structured reply. Skipping it
                # would leave an EARLIER reply standing as final, so the whole
                # projection fails instead of quietly showing the wrong answer.
                raise OwnerAuthorityBroken(
                    f"owner conversation {name} has a reply that cannot be "
                    "projected"
                )
            raphael_text = projected_text

        # The conversation ENDS on assistant output that carries no outcome. The
        # turn was real, so dropping it hid the owner's own latest message and
        # left the Workspace waiting on something that was already over, with no
        # control to recover from. Complete it with the SAME structured failure
        # an interrupted turn already gets, built from the fixed constant alone:
        # the stored text is never shown, so a plain-text imitation of a tool
        # call can neither reach the owner nor read as if it had run. Only the
        # TRAILING turn — an interior one is already bounded by the owner turn
        # after it, so nothing there is stranded — and only on the read side:
        # the stored row keeps saying exactly what it said.
        #
        # ``incomplete`` is therefore left as the earlier turns set it. This
        # turn is no longer a HIDDEN one: it is shown, with a terminal outcome
        # the owner can act on, so reporting it as missing would tell the caller
        # to warn about a turn it is already rendering.
        if owner_text is not None and raphael_text is None and unreadable_reply:
            raphael_text = json.dumps(
                {
                    "schema_version": 1,
                    "kind": "failure",
                    "message": _OWNER_INTERRUPTED_TURN_MESSAGE,
                },
                ensure_ascii=False,
            )
            substituted_failure_turn = True
        _flush()
        truncated = (
            owner_turn_truncated or len(turns) > _OWNER_HISTORY_TURN_LIMIT
        )
        data = turns[-_OWNER_HISTORY_TURN_LIMIT:]
        # ``latest_response_id`` below hands out an outstanding PROPOSAL handle.
        # The substituted turn is a terminal failure and grants no approval
        # authority, so a projection holding nothing else must not be what makes
        # an older proposal approvable again.
        proposal_bearing_turns = data[:-1] if substituted_failure_turn else data
        proposal_response_id = row[1]
        proposal_consumed = (
            proposal_response_id is not None and row[2] == proposal_response_id
        )
        proposal_claimed = (
            proposal_response_id is not None
            and not proposal_consumed
            and row[3] == proposal_response_id
            and row[5] == "claimed"
        )
        completed_run_id = None
        if (
            proposal_consumed
            and row[5] == "completed"
            and isinstance(row[4], str)
            and _OWNER_RUN_RE.fullmatch(row[4]) is not None
            # The opaque run handle is useful only while its exact terminal
            # receipt still exists. This is what lets another browser tab
            # recover the same founder-safe completion without trusting a
            # browser cookie or replaying the mutation.
            and self._bound_owner_run_completion(row[4]) is not None
        ):
            completed_run_id = row[4]
        return {
            # The exact turn this conversation currently ends at, whatever it
            # was. Distinct from ``latest_response_id``, which names the
            # outstanding PROPOSAL and is null after an ordinary question — a
            # caller that has to state the turn it planned against needs the
            # head, not the proposal. Service-to-service only, like every other
            # opaque handle in this projection.
            # Independent of the projected turns. The projection deliberately
            # excludes some reply kinds (an Automations ``automation_proposal``
            # is one), so a conversation with a real durable head can project
            # zero turns — and nulling the head there made every later turn on
            # that conversation either conflict as stale or replay the old
            # proposal.
            "head_response_id": row[0],
            "latest_response_id": (
                proposal_response_id if proposal_bearing_turns else None
            ),
            "proposal_consumed": proposal_consumed,
            "proposal_claimed": proposal_claimed,
            "active_run_id": row[4] if proposal_claimed else None,
            "completed_run_id": completed_run_id,
            "conversation_closed": bool(row[6]),
            # Whether older owner-visible turns exist beyond the window this
            # projection carries, so a caller renders "there is more" instead
            # of presenting a shortened transcript as the whole conversation.
            "truncated": truncated,
            # Whether an owner turn inside this window could not be projected as
            # a turn at all — a reply kind this projection excludes, or a
            # transcript that does not alternate. Reported for the same reason
            # ``truncated`` is: a caller must not present what is left as the
            # complete conversation.
            "incomplete": incomplete,
            # The request this conversation is planning right now: the owner's
            # own words and the response id planning them, both durable since
            # before the model started. Null once the turn is published, at
            # which point ``data`` carries the same words.
            "pending": pending,
            # The last request that ended without becoming a turn at all, and
            # the response id that decided it. It outlives the fence it was
            # made from, and only an explicit acknowledgement retires it.
            "recovery": recovery,
            "data": data,
        }

    @_owner_authority
    @_response_store_locked
    def mark_owner_proposal_consumed(
        self, profile: str, name: str, response_id: str,
    ) -> bool:
        """Durably prevent one applied proposal from regaining approval authority."""
        profile = self._profile(profile)
        if (
            not isinstance(name, str)
            or _OWNER_CONVERSATION_RE.fullmatch(name) is None
            or not isinstance(response_id, str)
            or _OWNER_RESPONSE_RE.fullmatch(response_id) is None
        ):
            return False
        cursor = self._conn.execute(
            "UPDATE conversations SET consumed_response_id = ? "
            "WHERE profile = ? AND name = ? AND proposal_response_id = ? AND ("
            "claimed_response_id IS NULL OR claimed_response_id != proposal_response_id "
            "OR claim_state IS NULL OR claim_state != 'claimed'"
            ")",
            (response_id, profile, name, response_id),
        )
        self._conn.commit()
        return cursor.rowcount == 1

    @staticmethod
    def _valid_owner_authority_ids(
        profile: str,
        name: str,
        response_id: str,
        claim_id: Optional[str] = None,
        run_id: Optional[str] = None,
    ) -> bool:
        return (
            isinstance(profile, str)
            and _OWNER_PROFILE_RE.fullmatch(profile) is not None
            and isinstance(name, str)
            and _OWNER_CONVERSATION_RE.fullmatch(name) is not None
            and isinstance(response_id, str)
            and _OWNER_RESPONSE_RE.fullmatch(response_id) is not None
            and (
                claim_id is None
                or isinstance(claim_id, str)
                and _OWNER_CLAIM_RE.fullmatch(claim_id) is not None
            )
            and (
                run_id is None
                or isinstance(run_id, str)
                and _OWNER_RUN_RE.fullmatch(run_id) is not None
            )
        )

    def _pending_owner_turn(
        self, profile: str, name: str,
    ) -> "Optional[Dict[str, str]]":
        """Project the request one live reservation is still planning.

        Read-only, and deliberately non-destructive: an expired row is simply
        not pending — nothing is planning it any more, and saying otherwise
        would tell the owner their words are still being worked on when they
        are not. Sweeping it belongs to the fence, which holds the write lock.

        The response id is service-to-service, like every other opaque handle
        in this projection: it is what lets a caller resume THIS turn instead of
        planning the same words again.
        """
        row = self._conn.execute(
            "SELECT response_id, owner_message FROM "
            "owner_conversation_reservations "
            "WHERE profile = ? AND name = ? AND expires_at > ?",
            (profile, name, time.time()),
        ).fetchone()
        if row is None:
            return None
        message = _pending_owner_message(row[1])
        if message is None:
            return None

        from agent.redact import redact_sensitive_text

        return {
            "owner": redact_sensitive_text(message, force=True),
            "response_id": str(row[0]),
        }

    def _owner_conversation_recovery(
        self, profile: str, name: str,
    ) -> "Optional[Dict[str, Optional[str]]]":
        """Project the interrupted request this conversation has not answered.

        Read-only and repeatable, for the same reason the record exists at all:
        the answer carrying it can be lost exactly like the one that started
        this. Only :meth:`acknowledge_owner_conversation_recovery` retires it.

        ``owner`` is null when the request was never projectable. That still
        matters: the response id alone is what lets a caller read the outcome
        this conversation ended a turn on, which is the fact a browser holding
        no handle at all would otherwise never learn.

        Age is deliberately not consulted. A record that has waited a week, or
        that a restart carried across, is still an unanswered request.
        """
        row = self._conn.execute(
            "SELECT response_id, owner_message FROM owner_conversation_recovery "
            "WHERE profile = ? AND name = ?",
            (profile, name),
        ).fetchone()
        if row is None:
            return None
        message = _pending_owner_message(row[1])
        if message is None:
            return {"owner": None, "response_id": str(row[0])}

        from agent.redact import redact_sensitive_text

        return {
            "owner": redact_sensitive_text(message, force=True),
            "response_id": str(row[0]),
        }

    def _convert_owner_reservation_to_recovery_locked(
        self, profile: str, name: str, response_id: str,
    ) -> bool:
        """Replace one turn's fence with the way back to its outcome.

        The caller holds ``_conversation_lock`` and an open transaction, so the
        release and the record are ONE write, and the record is minted only
        FROM a live fence. A turn that published released its reservation inside
        the transaction that took the head, so a failure handler arriving
        afterwards finds nothing to convert and writes nothing: this can never
        become a second, contradicting account of a completed turn.
        """
        reserved = self._conn.execute(
            "SELECT owner_message FROM owner_conversation_reservations "
            "WHERE profile = ? AND name = ? AND response_id = ?",
            (profile, name, response_id),
        ).fetchone()
        if reserved is None:
            return False
        self._conn.execute(
            "DELETE FROM owner_conversation_reservations "
            "WHERE profile = ? AND name = ? AND response_id = ?",
            (profile, name, response_id),
        )
        now = time.time()
        self._conn.execute(
            "INSERT OR REPLACE INTO owner_conversation_recovery "
            "(profile, name, response_id, owner_message, created_at, "
            "expires_at) VALUES (?, ?, ?, ?, ?, ?)",
            (
                profile, name, response_id, reserved[0], now,
                _OWNER_CONVERSATION_RECOVERY_NO_EXPIRY,
            ),
        )
        return True

    @_owner_authority
    def acknowledge_owner_conversation_recovery(
        self, profile: str, name: str, response_id: str,
    ) -> str:
        """Seal one recovery as a durable failure turn, then retire its fence.

        Exact and idempotent: it seals THIS turn and nothing else, and a caller
        that says so twice is not in error — its first answer may have been the
        one that never arrived.  The response becomes the conversation head in
        the same transaction that removes recovery, so a reload cannot erase
        the owner's words or reveal an older proposal as approvable again.

        Says which of three things happened, because a caller cannot verify an
        answer that is the same whatever it did. ``"retired"`` sealed this
        exact record as a non-actionable failure turn. ``"absent"`` found
        nothing outstanding on this
        conversation at all, which is what a repeated acknowledgement sees.
        ``"mismatch"`` found a DIFFERENT unanswered request, and wrote nothing:
        reporting that as success would tell a caller a request it has never
        shown anyone had been dealt with.
        """
        profile = self._profile(profile)
        if not self._valid_owner_authority_ids(profile, name, response_id):
            return "mismatch"
        with self._conversation_lock:
            try:
                self._conn.execute("BEGIN IMMEDIATE")
                recovery = self._conn.execute(
                    "SELECT response_id, owner_message "
                    "FROM owner_conversation_recovery "
                    "WHERE profile = ? AND name = ?",
                    (profile, name),
                ).fetchone()
                if recovery is None:
                    self._conn.commit()
                    return "absent"
                if recovery[0] != response_id:
                    self._conn.rollback()
                    return "mismatch"

                owner_message = _pending_owner_message(recovery[1])
                if owner_message is not None:
                    terminal_row = self._conn.execute(
                        "SELECT data FROM responses "
                        "WHERE profile = ? AND response_id = ?",
                        (profile, response_id),
                    ).fetchone()
                    if terminal_row is None:
                        raise OwnerAuthorityBroken(
                            "owner recovery maps a response that is missing"
                        )
                    try:
                        terminal_data = json.loads(terminal_row[0])
                    except (json.JSONDecodeError, TypeError) as exc:
                        raise OwnerAuthorityBroken(
                            "owner recovery maps an unreadable response"
                        ) from exc
                    terminal_response = (
                        terminal_data.get("response")
                        if isinstance(terminal_data, dict) else None
                    )
                    if (
                        not isinstance(terminal_response, dict)
                        or terminal_response.get("status")
                        not in {"failed", "incomplete"}
                    ):
                        raise OwnerAuthorityBroken(
                            "owner recovery maps a response that is not terminal"
                        )

                    head = self._conn.execute(
                        "SELECT response_id FROM conversations "
                        "WHERE profile = ? AND name = ?",
                        (profile, name),
                    ).fetchone()
                    previous_response_id = head[0] if head is not None else None
                    previous_history: List[Dict[str, Any]] = []
                    if previous_response_id is not None:
                        previous_row = self._conn.execute(
                            "SELECT data FROM responses "
                            "WHERE profile = ? AND response_id = ?",
                            (profile, previous_response_id),
                        ).fetchone()
                        if previous_row is None:
                            raise OwnerAuthorityBroken(
                                "owner conversation maps a response that is missing"
                            )
                        try:
                            previous_data = json.loads(previous_row[0])
                        except (json.JSONDecodeError, TypeError) as exc:
                            raise OwnerAuthorityBroken(
                                "owner conversation maps an unreadable response"
                            ) from exc
                        previous_history = (
                            previous_data.get("conversation_history")
                            if isinstance(previous_data, dict) else None
                        )
                        if not isinstance(previous_history, list):
                            raise OwnerAuthorityBroken(
                                "owner conversation has no readable transcript"
                            )

                    failure_reply = json.dumps(
                        {
                            "schema_version": 1,
                            "kind": "failure",
                            "message": _OWNER_INTERRUPTED_TURN_MESSAGE,
                        },
                        ensure_ascii=False,
                    )
                    terminal_data["conversation_history"] = [
                        *previous_history,
                        {"role": "user", "content": owner_message},
                        {"role": "assistant", "content": failure_reply},
                    ]
                    self._put_response_locked(
                        response_id, terminal_data, profile,
                    )
                    if not self._set_conversation_locked(
                        name,
                        response_id,
                        owner_proposal=False,
                        profile=profile,
                        reservation_id=None,
                        expected_previous_response_id=previous_response_id,
                    ):
                        raise OwnerAuthorityBroken(
                            "owner recovery could not seal its conversation turn"
                        )

                cursor = self._conn.execute(
                    "DELETE FROM owner_conversation_recovery "
                    "WHERE profile = ? AND name = ? AND response_id = ?",
                    (profile, name, response_id),
                )
                if cursor.rowcount != 1:
                    raise OwnerAuthorityBroken(
                        "owner recovery changed while it was being sealed"
                    )
                self._conn.commit()
                return "retired"
            except Exception:
                self._conn.rollback()
                raise

    def _owner_response_work_is_unresolved_locked(
        self, profile: str, response_id: str,
    ) -> bool:
        """Whether a durable job row still says somebody must finish this.

        The job row is dropped only by the transaction that makes the response
        terminal, so while it exists the work behind that response has not been
        resolved either way.
        """
        return self._conn.execute(
            "SELECT 1 FROM owner_executor_jobs "
            "WHERE kind = 'response' AND job_key = ? AND profile = ?",
            (response_id, profile),
        ).fetchone() is not None

    def _active_owner_conversation_reservation_locked(
        self, profile: str, name: str,
    ) -> "sqlite3.Row | None":
        """Return the fence on this conversation, discarding only dead ones.

        Callers hold ``_conversation_lock`` and an active ``BEGIN IMMEDIATE``
        transaction, so expiry and the following authority decision are one
        atomic fence.

        An expired lease says the executor stopped renewing it, NOT that the
        work it fenced is over. While that response's durable job row still
        says somebody must finish it, the reservation is kept and still
        returned: discarding it left the ending that eventually arrives with
        nothing to convert into the record carrying the owner's request, and
        let another caller take or close the conversation in the meantime.
        """
        now = time.time()
        self._conn.execute(
            "DELETE FROM owner_conversation_reservations "
            "WHERE profile = ? AND name = ? "
            "AND (expires_at IS NULL OR expires_at <= ?) "
            "AND response_id NOT IN ("
            "SELECT job_key FROM owner_executor_jobs "
            "WHERE kind = 'response' AND profile = ?)",
            (profile, name, now, profile),
        )
        return self._conn.execute(
            "SELECT response_id, expires_at FROM owner_conversation_reservations "
            "WHERE profile = ? AND name = ?",
            (profile, name),
        ).fetchone()

    def _unanswered_owner_request_locked(self, profile: str, name: str) -> bool:
        """Whether a request that ended without a turn is still standing here.

        Callers hold ``_conversation_lock`` and an active ``BEGIN IMMEDIATE``,
        so this and the authority decision that follows are one atomic fence.

        Deliberately separate from the reservation above. The reservation is
        released the moment a plan ENDS, so a conversation carrying an
        interrupted request looks completely idle — which is how an older
        proposal on it could still be claimed and run while the later request
        stood unanswered behind it, consuming that proposal in a way no
        acknowledgement can undo. Only an acknowledgement retires this.
        """
        return self._conn.execute(
            "SELECT 1 FROM owner_conversation_recovery "
            "WHERE profile = ? AND name = ?",
            (profile, name),
        ).fetchone() is not None

    @_owner_authority
    @_response_store_locked
    def owner_request_is_unanswered(self, profile: str, name: str) -> bool:
        """The same fence, for a caller deciding BEFORE it opens a transaction.

        Exactly :meth:`_unanswered_owner_request_locked`, so a request-level
        refusal and the transaction that would otherwise mutate this
        conversation can never drift apart. It is a read: a caller that acts on
        it still meets the transactional fence, which is the one that decides.
        """
        profile = self._profile(profile)
        if (
            not isinstance(name, str)
            or _OWNER_CONVERSATION_RE.fullmatch(name) is None
        ):
            return False
        return self._unanswered_owner_request_locked(profile, name)

    @_owner_authority
    def claim_owner_proposal(
        self, profile: str, name: str, response_id: str, claim_id: str,
    ) -> bool:
        """Atomically reserve the exact current proposal for one owner approval.

        Refused while an interrupted request is standing unanswered on this
        conversation: an approval is not an answer to it, and the run behind
        one consumes the proposal for good.
        """
        profile = self._profile(profile)
        if not self._valid_owner_authority_ids(profile, name, response_id, claim_id):
            return False
        with self._conversation_lock:
            try:
                self._conn.execute("BEGIN IMMEDIATE")
                row = self._conn.execute(
                    "SELECT proposal_response_id, consumed_response_id, claimed_response_id, "
                    "claim_id, claim_state, closed, owner_run_id, claim_expires_at "
                    "FROM conversations WHERE profile = ? AND name = ?",
                    (profile, name),
                ).fetchone()
                reserved = self._active_owner_conversation_reservation_locked(
                    profile, name,
                )
                if (
                    row is None
                    or row[0] != response_id
                    or row[1] == response_id
                    or bool(row[5])
                    or reserved is not None
                    or self._unanswered_owner_request_locked(profile, name)
                ):
                    self._conn.rollback()
                    return False
                if row[2] == response_id and row[4] == "claimed":
                    same_claim = row[3] == claim_id
                    lease_live = row[6] is not None or (
                        isinstance(row[7], (int, float)) and row[7] > time.time()
                    )
                    if lease_live:
                        self._conn.rollback()
                        return same_claim
                self._conn.execute(
                    "UPDATE conversations SET claimed_response_id = ?, claim_id = ?, "
                    "claim_expires_at = ?, owner_run_id = NULL, bound_operation = NULL, "
                    "bound_payload_digest = NULL, claim_state = 'claimed' "
                    "WHERE profile = ? AND name = ? AND proposal_response_id = ?",
                    (
                        response_id, claim_id, time.time() + _OWNER_CLAIM_LEASE_SECONDS,
                        profile, name, response_id,
                    ),
                )
                self._conn.commit()
                return True
            except Exception:
                self._conn.rollback()
                raise

    @_owner_authority
    def attach_owner_run(
        self,
        profile: str,
        name: str,
        response_id: str,
        claim_id: str,
        run_id: str,
        *,
        operation: Optional[str] = None,
        payload_digest: Optional[str] = None,
    ) -> bool:
        """Bind one native run to the exact proposal claim, idempotently."""
        profile = self._profile(profile)
        if not self._valid_owner_authority_ids(
            profile, name, response_id, claim_id, run_id,
        ):
            return False
        with self._conversation_lock:
            cursor = self._conn.execute(
                "UPDATE conversations SET owner_run_id = ?, claim_expires_at = NULL, "
                "bound_operation = COALESCE(?, bound_operation), "
                "bound_payload_digest = COALESCE(?, bound_payload_digest) "
                "WHERE profile = ? AND name = ? AND proposal_response_id = ? "
                "AND consumed_response_id IS NOT ? "
                "AND claimed_response_id = ? AND claim_id = ? AND claim_state = 'claimed' "
                "AND (owner_run_id IS NULL OR owner_run_id = ?) "
                "AND (bound_operation IS NULL OR bound_operation IS ?) "
                "AND (bound_payload_digest IS NULL OR bound_payload_digest IS ?)",
                (
                    run_id, operation, payload_digest, profile, name, response_id,
                    response_id, response_id, claim_id, run_id, operation, payload_digest,
                ),
            )
            self._conn.commit()
            return cursor.rowcount == 1

    @_owner_authority
    def claim_and_attach_owner_run(
        self,
        profile: str,
        name: str,
        response_id: str,
        claim_id: str,
        run_id: str,
        *,
        operation: str,
        payload_digest: str,
        job_payload: "Optional[Dict[str, Any]]" = None,
        job_profile: Optional[str] = None,
    ) -> bool:
        """Atomically bind one validated proposal, claim, payload, and run.

        ``job_payload`` puts the durable executor recovery job in the SAME
        transaction as the claim, so a crash between them can never leave a
        proposal claimed by a run nobody is driving.

        Refused, like :meth:`claim_owner_proposal`, while an interrupted
        request is standing unanswered on this conversation.
        """
        profile = self._profile(profile)
        if (
            not self._valid_owner_authority_ids(
                profile, name, response_id, claim_id, run_id,
            )
            or operation not in {
                "owner_task_graph_commit", "owner_project_plan_commit",
            }
            or re.fullmatch(r"[a-f0-9]{64}", payload_digest) is None
        ):
            return False
        with self._conversation_lock:
            try:
                self._conn.execute("BEGIN IMMEDIATE")
                row = self._conn.execute(
                    "SELECT proposal_response_id, consumed_response_id, "
                    "claimed_response_id, claim_id, owner_run_id, claim_state, closed, "
                    "claim_expires_at "
                    "FROM conversations WHERE profile = ? AND name = ?",
                    (profile, name),
                ).fetchone()
                reserved = self._active_owner_conversation_reservation_locked(
                    profile, name,
                )
                if (
                    row is None
                    or row[0] != response_id
                    or row[1] == response_id
                    or bool(row[6])
                    or reserved is not None
                    or self._unanswered_owner_request_locked(profile, name)
                ):
                    self._conn.rollback()
                    return False
                if row[5] == "claimed" and row[2] == response_id:
                    stale_unattached = (
                        row[4] is None
                        and isinstance(row[7], (int, float))
                        and row[7] <= time.time()
                    )
                    if not stale_unattached and (
                        row[3] != claim_id or row[4] not in {None, run_id}
                    ):
                        self._conn.rollback()
                        return False
                self._conn.execute(
                    "UPDATE conversations SET claimed_response_id = ?, claim_id = ?, "
                    "claim_expires_at = NULL, owner_run_id = ?, bound_operation = ?, "
                    "bound_payload_digest = ?, claim_state = 'claimed' "
                    "WHERE profile = ? AND name = ? AND proposal_response_id = ?",
                    (
                        response_id, claim_id, run_id, operation, payload_digest,
                        profile, name, response_id,
                    ),
                )
                if job_payload is not None:
                    self._reserve_owner_job_locked(
                        "run", run_id, self._profile(job_profile), job_payload,
                    )
                self._conn.commit()
                return True
            except Exception:
                self._conn.rollback()
                raise

    @_owner_authority
    @_response_store_locked
    def owner_claim_is_completed(
        self, profile: str, name: str, response_id: str, claim_id: str, run_id: str,
    ) -> bool:
        profile = self._profile(profile)
        if not self._valid_owner_authority_ids(
            profile, name, response_id, claim_id, run_id,
        ):
            return False
        row = self._conn.execute(
            "SELECT 1 FROM conversations WHERE profile = ? AND name = ? "
            "AND proposal_response_id = ? AND consumed_response_id = ? "
            "AND claimed_response_id = ? AND claim_id = ? AND owner_run_id = ? "
            "AND claim_state = 'completed'",
            (profile, name, response_id, response_id, response_id, claim_id, run_id),
        ).fetchone()
        return row is not None

    @_owner_authority
    @_response_store_locked
    def owner_claim_is_released(
        self, profile: str, name: str, response_id: str, claim_id: str, run_id: str,
    ) -> bool:
        """Verify the server-finalized release of one exact failed owner run."""
        profile = self._profile(profile)
        if not self._valid_owner_authority_ids(
            profile, name, response_id, claim_id, run_id,
        ):
            return False
        row = self._conn.execute(
            "SELECT 1 FROM conversations WHERE profile = ? AND name = ? "
            "AND proposal_response_id = ? AND consumed_response_id IS NOT ? "
            "AND claimed_response_id = ? AND claim_id = ? AND owner_run_id = ? "
            "AND claim_state = 'released'",
            (profile, name, response_id, response_id, response_id, claim_id, run_id),
        ).fetchone()
        return row is not None

    @_owner_authority
    def abandon_unattached_owner_claim(
        self, profile: str, name: str, response_id: str, claim_id: str,
    ) -> bool:
        """Release only the exact legacy claim that never acquired a run."""
        profile = self._profile(profile)
        if not self._valid_owner_authority_ids(
            profile, name, response_id, claim_id,
        ):
            return False
        with self._conversation_lock:
            cursor = self._conn.execute(
                "UPDATE conversations SET claim_state = 'released', "
                "claim_expires_at = NULL WHERE profile = ? AND name = ? "
                "AND proposal_response_id = ? AND consumed_response_id IS NOT ? "
                "AND claimed_response_id = ? AND claim_id = ? "
                "AND owner_run_id IS NULL AND claim_state IN ('claimed', 'released')",
                (profile, name, response_id, response_id, response_id, claim_id),
            )
            self._conn.commit()
            return cursor.rowcount == 1

    @_owner_authority
    @_response_store_locked
    def owner_proposal_record(
        self, profile: str, name: str, response_id: str,
    ) -> "tuple[dict[str, Any], str] | None":
        """Read the exact current stored proposal and its persisted digest."""
        profile = self._profile(profile)
        if not self._valid_owner_authority_ids(profile, name, response_id):
            return None
        row = self._conn.execute(
            "SELECT c.proposal_digest, c.consumed_response_id, c.closed, r.data "
            "FROM conversations c JOIN responses r ON r.response_id = c.response_id "
            "AND r.profile = c.profile WHERE c.profile = ? AND c.name = ? "
            "AND c.response_id = ? AND c.proposal_response_id = ?",
            (profile, name, response_id, response_id),
        ).fetchone()
        if row is None or row[1] == response_id or bool(row[2]):
            return None
        try:
            stored = json.loads(row[3])
        except (json.JSONDecodeError, TypeError):
            return None
        candidate = _owner_final_proposal(
            stored.get("conversation_history") if isinstance(stored, dict) else None
        )
        digest = _owner_proposal_digest(candidate)
        if candidate is None or digest is None or row[0] != digest:
            return None
        return candidate, digest

    @_owner_authority
    @_response_store_locked
    def owner_run_is_attached(
        self, profile: str, name: str, response_id: str, claim_id: str, run_id: str,
    ) -> bool:
        """Verify the exact run binding created by the native run endpoint."""
        profile = self._profile(profile)
        if not self._valid_owner_authority_ids(
            profile, name, response_id, claim_id, run_id,
        ):
            return False
        row = self._conn.execute(
            "SELECT 1 FROM conversations WHERE profile = ? AND name = ? "
            "AND proposal_response_id = ? "
            "AND consumed_response_id IS NOT ? AND claimed_response_id = ? "
            "AND claim_id = ? AND owner_run_id = ? AND claim_state = 'claimed'",
            (profile, name, response_id, response_id, response_id, claim_id, run_id),
        ).fetchone()
        return row is not None

    @_owner_authority
    def complete_owner_claim(
        self, profile: str, name: str, response_id: str, claim_id: str, run_id: str,
    ) -> bool:
        """Consume the exact proposal only after its exact claimed run completes."""
        profile = self._profile(profile)
        if not self._valid_owner_authority_ids(
            profile, name, response_id, claim_id, run_id,
        ):
            return False
        with self._conversation_lock:
            cursor = self._conn.execute(
                "UPDATE conversations SET consumed_response_id = ?, claim_state = 'completed' "
                "WHERE profile = ? AND name = ? AND proposal_response_id = ? "
                "AND claimed_response_id = ? "
                "AND claim_id = ? AND owner_run_id = ? "
                "AND claim_state IN ('claimed', 'completed')",
                (response_id, profile, name, response_id, response_id, claim_id, run_id),
            )
            self._conn.commit()
            return cursor.rowcount == 1

    @_owner_authority
    def release_owner_claim(
        self, profile: str, name: str, response_id: str, claim_id: str, run_id: str,
    ) -> bool:
        """Release an exact failed run without reopening any other proposal."""
        profile = self._profile(profile)
        if not self._valid_owner_authority_ids(
            profile, name, response_id, claim_id, run_id,
        ):
            return False
        with self._conversation_lock:
            cursor = self._conn.execute(
                "UPDATE conversations SET claim_state = 'released', claim_expires_at = NULL "
                "WHERE profile = ? AND name = ? AND proposal_response_id = ? "
                "AND consumed_response_id IS NOT ? "
                "AND claimed_response_id = ? AND claim_id = ? AND owner_run_id = ? "
                "AND claim_state IN ('claimed', 'released')",
                (profile, name, response_id, response_id, response_id, claim_id, run_id),
            )
            self._conn.commit()
            return cursor.rowcount == 1

    @_owner_authority
    def close_owner_conversation(
        self,
        profile: str,
        name: str,
        response_id: Optional[str],
        *,
        expected_head_response_id: Any = _UNSTATED,
        next_session_id: Any = None,
    ) -> bool:
        """Close one exact conversation only while it has no unanswered work.

        ``expected_head_response_id`` is the caller's assertion about the turn
        this conversation currently ENDS at, compared inside the closing
        transaction. Comparing only the outstanding proposal could not see a
        concurrent question landing: an ordinary question leaves
        ``proposal_response_id`` untouched, so a stale tab still matched and
        closed the conversation — hiding a newer turn the owner had already
        been shown. ``None`` asserts "no turn yet"; :data:`_UNSTATED` preserves
        the previous behaviour for a caller that states no head.

        An unacknowledged interrupted request refuses this outright. The fence
        is released the moment a plan ends, so a live reservation cannot speak
        for a request that already ended without publishing; closing there left
        that record orphaned on a conversation nothing would read again.

        ``next_session_id`` is the change session this group moves on to, given
        its durable place in the group inside the SAME transaction as the
        close. Without it the group's current-session pointer still named the
        conversation just retired, so a browser holding no cookie came back to
        it. A malformed one is refused rather than ignored, since a caller that
        believes it moved the group on must not be told it did.
        """
        profile = self._profile(profile)
        if (
            not isinstance(name, str)
            or _OWNER_CONVERSATION_RE.fullmatch(name) is None
            or (
                response_id is not None
                and (
                    not isinstance(response_id, str)
                    or _OWNER_RESPONSE_RE.fullmatch(response_id) is None
                )
            )
            or (
                expected_head_response_id is not _UNSTATED
                and expected_head_response_id is not None
                and (
                    not isinstance(expected_head_response_id, str)
                    or _OWNER_RESPONSE_RE.fullmatch(expected_head_response_id) is None
                )
            )
            or (
                next_session_id is not None
                and (
                    not isinstance(next_session_id, str)
                    or re.fullmatch(r"[a-f0-9]{32}", next_session_id) is None
                )
            )
        ):
            return False
        group, _session = _owner_conversation_group(name)
        if next_session_id is not None and group is None:
            return False
        with self._conversation_lock:

            def _commit_retired() -> bool:
                """Record the closure and where this group goes next, together."""
                self._conn.execute(
                    "INSERT INTO owner_conversation_closures "
                    "(profile, name, closed_at) VALUES (?, ?, ?) "
                    "ON CONFLICT(profile, name) DO NOTHING",
                    (profile, name, time.time()),
                )
                if next_session_id is not None and group is not None:
                    self._record_owner_session_locked(
                        profile, group, f"{group}-{next_session_id}",
                    )
                self._conn.commit()
                return True

            try:
                self._conn.execute("BEGIN IMMEDIATE")
                if self._active_owner_conversation_reservation_locked(
                    profile, name,
                ) is not None:
                    self._conn.rollback()
                    return False
                if self._conn.execute(
                    "SELECT 1 FROM owner_conversation_recovery "
                    "WHERE profile = ? AND name = ?",
                    (profile, name),
                ).fetchone() is not None:
                    self._conn.rollback()
                    return False
                row = self._conn.execute(
                    "SELECT proposal_response_id, consumed_response_id, claimed_response_id, "
                    "claim_state, closed, response_id FROM conversations "
                    "WHERE profile = ? AND name = ?",
                    (profile, name),
                ).fetchone()
                if row is None:
                    if response_id is not None or (
                        expected_head_response_id is not _UNSTATED
                        and expected_head_response_id is not None
                    ):
                        self._conn.rollback()
                        return False
                    return _commit_retired()
                if row[0] != response_id:
                    self._conn.rollback()
                    return False
                if (
                    expected_head_response_id is not _UNSTATED
                    and row[5] != expected_head_response_id
                ):
                    self._conn.rollback()
                    return False
                if bool(row[4]):
                    return _commit_retired()
                if row[2] == row[0] and row[3] == "claimed" and row[1] != row[0]:
                    self._conn.rollback()
                    return False
                cursor = self._conn.execute(
                    "UPDATE conversations SET closed = 1 "
                    "WHERE profile = ? AND name = ? AND proposal_response_id IS ? "
                    "AND response_id IS ?",
                    (profile, name, response_id, row[5]),
                )
                if cursor.rowcount != 1:
                    self._conn.rollback()
                    return False
                return _commit_retired()
            except Exception:
                self._conn.rollback()
                raise

    @_owner_authority
    @_response_store_locked
    def owner_session_index(
        self, profile: str, group: str,
    ) -> Dict[str, Any]:
        """List owner-safe change-session metadata for one conversation group.

        Conversation transcripts remain the native authority.  This bounded
        projection exposes no response handles, system prompts, tool output,
        internal reasoning, or sibling project history.

        Ordered by each session's own immutable sequence (see
        :meth:`_record_owner_session_locked`), newest FIRST, so the group's
        current session is always the first entry and a bounded read can only
        ever cut the oldest ones. Ordering by the mapped response's LRU access
        time instead meant reading an old change promoted it, and could push the
        real current session past the bound before anything validated it.

        Nothing is silently dropped. A session whose mapped response row is
        missing, whose stored JSON is unreadable, or whose transcript cannot be
        projected is listed as explicitly unavailable, and a session with a
        valid head but no owner-visible turns yet is listed with a zero turn
        count — dropping either made a caller select an older session, or a
        legacy one, as if the real one did not exist.
        """
        profile = self._profile(profile)
        if (
            not isinstance(group, str)
            or re.fullmatch(r"raphael-owner-[a-f0-9]{32}", group) is None
        ):
            return {"data": [], "truncated": False, "current_session_id": None}

        # Resolved from the durable sequence alone, independently of the bound
        # below, so "which session is current" is one unambiguous fact even for
        # a group whose list is truncated. A caller must never infer it from a
        # cookie it cannot find in the list.
        current_row = self._conn.execute(
            "SELECT name FROM owner_conversation_sessions "
            "WHERE profile = ? AND group_name = ? ORDER BY seq DESC LIMIT 1",
            (profile, group),
        ).fetchone()
        current_session_id: Optional[str] = None
        if current_row is not None and isinstance(current_row[0], str):
            current_session_id = (
                "legacy" if current_row[0] == group
                else current_row[0].removeprefix(f"{group}-")
            )
        rows = self._conn.execute(
            "SELECT s.name, s.created_at, c.response_id, r.data "
            "FROM owner_conversation_sessions s "
            "JOIN conversations c ON c.profile = s.profile AND c.name = s.name "
            "LEFT JOIN responses r ON r.profile = c.profile "
            "AND r.response_id = c.response_id "
            "WHERE s.profile = ? AND s.group_name = ? "
            "ORDER BY s.seq DESC LIMIT ?",
            (profile, group, _OWNER_SESSION_INDEX_LIMIT + 1),
        ).fetchall()
        truncated = len(rows) > _OWNER_SESSION_INDEX_LIMIT
        sessions: List[Dict[str, Any]] = []
        expected = re.compile(re.escape(group) + r"(?:-[a-f0-9]{32})?")
        for name, session_created_at, _mapped_response_id, raw in rows[
            :_OWNER_SESSION_INDEX_LIMIT
        ]:
            if not isinstance(name, str) or expected.fullmatch(name) is None:
                continue
            session_id = (
                "legacy" if name == group else name.removeprefix(f"{group}-")
            )
            fallback_at = (
                int(session_created_at)
                if isinstance(session_created_at, (int, float))
                and not isinstance(session_created_at, bool)
                and math.isfinite(float(session_created_at))
                and session_created_at >= 0
                else 0
            )

            def _unavailable() -> Dict[str, Any]:
                return {
                    "session_id": session_id,
                    "updated_at": fallback_at,
                    "preview": "",
                    "visible_turn_count": 0,
                    "available": False,
                }

            updated_at: Any = None
            if raw is not None:
                try:
                    stored = json.loads(raw)
                except (json.JSONDecodeError, TypeError):
                    stored = None
                response = (
                    stored.get("response") if isinstance(stored, dict) else None
                )
                updated_at = (
                    response.get("created_at") if isinstance(response, dict) else None
                )
            if (
                raw is None
                or isinstance(updated_at, bool)
                or not isinstance(updated_at, (int, float))
                or not math.isfinite(updated_at)
                or updated_at < 0
            ):
                sessions.append(_unavailable())
                continue
            try:
                snapshot = self.owner_history_snapshot(name, profile=profile)
            except OwnerAuthorityBroken:
                sessions.append(_unavailable())
                continue
            history = snapshot["data"]
            preview = history[0]["owner"] if history else ""
            if len(preview) > 180:
                preview = preview[:177].rstrip() + "..."
            sessions.append({
                "session_id": session_id,
                "updated_at": int(updated_at),
                "preview": preview,
                "visible_turn_count": len(history),
                "available": True,
            })
        return {
            "data": sessions,
            "truncated": truncated,
            "current_session_id": current_session_id,
        }

    @_owner_authority
    def owner_history(
        self, name: str, *, profile: Optional[str] = None,
    ) -> List[Dict[str, str]]:
        """Return the backward-compatible owner-safe turn list."""
        return self.owner_history_snapshot(name, profile=profile)["data"]

    @_owner_authority
    def reserve_owner_conversation(
        self,
        profile: str,
        name: str,
        response_id: str,
        *,
        expected_previous_response_id: Any = _UNSTATED,
        owner_message: Optional[str] = None,
    ) -> bool:
        """Fence one owner turn before any model or tool is allowed to run.

        ``expected_previous_response_id`` is the caller's assertion about the
        turn this one follows, compared HERE — inside the same transaction that
        takes the fence — so a delayed request planned against predecessor A
        cannot take the fence after a newer request has already committed
        predecessor B. ``None`` asserts "no turn yet"; :data:`_UNSTATED` (the
        default) asserts nothing and preserves the previous behaviour for every
        caller that does not state one.

        ``owner_message`` is the request this turn is planning, recorded in the
        same write that takes the fence — that is, before any model runs. It is
        what :meth:`owner_history_snapshot` projects as ``pending``, so a browser
        that never received the accept response can still show the owner exactly
        what they sent and resume THIS response id instead of planning the same
        words a second time.

        An unacknowledged interrupted request refuses this outright. Deleting it
        to make room for a newer turn destroyed the only account of what the
        owner sent, and it took nothing more than a second browser being used to
        do it. Nothing here plans anything until that record has been
        acknowledged.

        So does the draft that CREATED a Project: its proposal is consumed, the
        run that consumed it is still bound to it, and the receipt naming that
        Project is replayed from exactly that pair until a browser says it holds
        it. A turn taken here moves the head that pair is read from, so the
        receipt and the redirect to the Project became unreachable while the
        Project itself sat there — and a stale tab was all it took. Only the
        acknowledgement closes this conversation, and the group moves on to a
        successor in the same write.
        """
        profile = self._profile(profile)
        if not self._valid_owner_authority_ids(profile, name, response_id):
            return False
        with self._conversation_lock:
            try:
                self._conn.execute("BEGIN IMMEDIATE")
                if self._conn.execute(
                    "SELECT 1 FROM owner_conversation_closures "
                    "WHERE profile = ? AND name = ?",
                    (profile, name),
                ).fetchone() is not None:
                    self._conn.rollback()
                    return False
                if self._conn.execute(
                    "SELECT 1 FROM owner_conversation_recovery "
                    "WHERE profile = ? AND name = ?",
                    (profile, name),
                ).fetchone() is not None:
                    self._conn.rollback()
                    return False
                row = self._conn.execute(
                    "SELECT proposal_response_id, consumed_response_id, "
                    "claimed_response_id, claim_state, closed, response_id, "
                    "owner_run_id, bound_operation "
                    "FROM conversations WHERE profile = ? AND name = ?",
                    (profile, name),
                ).fetchone()
                if expected_previous_response_id is not _UNSTATED and (
                    (row[5] if row is not None else None)
                    != expected_previous_response_id
                ):
                    self._conn.rollback()
                    return False
                if row is not None and (
                    bool(row[4])
                    or (
                        row[0] is not None
                        and row[2] == row[0]
                        and row[3] == "claimed"
                        and row[1] != row[0]
                    )
                    # The spent New Project draft, named by the one durable fact
                    # that distinguishes it from a Project's change session: the
                    # operation the consuming run was bound to. A change session
                    # is not spent by the run it started and goes on taking
                    # turns; a draft that created a Project owes exactly one
                    # receipt and then retires.
                    or (
                        row[0] is not None
                        and row[1] == row[0]
                        and row[6] is not None
                        and row[7] == "owner_task_graph_commit"
                    )
                ):
                    self._conn.rollback()
                    return False
                existing = self._active_owner_conversation_reservation_locked(
                    profile, name,
                )
                if existing is not None:
                    if existing[0] == response_id:
                        self._conn.execute(
                            "UPDATE owner_conversation_reservations "
                            "SET expires_at = ? WHERE profile = ? AND name = ? "
                            "AND response_id = ?",
                            (
                                time.time()
                                + _OWNER_CONVERSATION_RESERVATION_LEASE_SECONDS,
                                profile, name, response_id,
                            ),
                        )
                        self._conn.commit()
                        return True
                    self._conn.rollback()
                    return False
                now = time.time()
                self._conn.execute(
                    "INSERT INTO owner_conversation_reservations "
                    "(profile, name, response_id, created_at, expires_at, "
                    "owner_message) VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        profile, name, response_id, now,
                        now + _OWNER_CONVERSATION_RESERVATION_LEASE_SECONDS,
                        _pending_owner_message(owner_message),
                    ),
                )
                self._conn.commit()
                return True
            except Exception:
                self._conn.rollback()
                raise

    @_owner_authority
    def release_owner_conversation_reservation(
        self, profile: str, name: str, response_id: str,
    ) -> None:
        """Drop this turn's fence, unless its work is still unresolved.

        Called when a turn is over, including from a task-done callback that
        runs whatever the turn did. A response whose durable job row still says
        somebody must finish it is NOT over: dropping its fence there would
        leave the recovery pass nothing to convert, so the request itself could
        never be recovered.
        """
        profile = self._profile(profile)
        with self._conversation_lock:
            if self._owner_response_work_is_unresolved_locked(profile, response_id):
                return
            self._conn.execute(
                "DELETE FROM owner_conversation_reservations "
                "WHERE profile = ? AND name = ? AND response_id = ?",
                (profile, name, response_id),
            )
            self._conn.commit()

    @_owner_authority
    def renew_owner_conversation_reservation(
        self, profile: str, name: str, response_id: str,
    ) -> bool:
        """Extend one still-live exact reservation without reviving a lost one."""
        profile = self._profile(profile)
        if not self._valid_owner_authority_ids(profile, name, response_id):
            return False
        now = time.time()
        with self._conversation_lock:
            cursor = self._conn.execute(
                "UPDATE owner_conversation_reservations SET expires_at = ? "
                "WHERE profile = ? AND name = ? AND response_id = ? "
                "AND expires_at > ?",
                (
                    now + _OWNER_CONVERSATION_RESERVATION_LEASE_SECONDS,
                    profile, name, response_id, now,
                ),
            )
            self._conn.commit()
            return cursor.rowcount == 1

    @_owner_authority
    def reserve_run_idempotency(
        self,
        profile: str,
        session_scope: str,
        idempotency_key: str,
        fingerprint: str,
        run_id: str,
        *,
        owner: "dict[str, str] | None" = None,
        job_payload: "Optional[Dict[str, Any]]" = None,
    ) -> "tuple[str, Optional[str]]":
        """Persist one scoped run identity before execution can be scheduled.

        ``job_payload`` makes the durable executor recovery job part of THIS
        transaction. The run's idempotency row and its owner proposal claim used
        to commit here while the job was reserved much later, so a crash in that
        gap left a durable queued run and a claimed proposal with no executor:
        polling reported working forever and the owner could not approve the same
        proposal again.

        Both owner branches below — the fresh claim and the retry that re-binds
        a released one — are refused while an interrupted request is standing
        unanswered on that conversation, exactly as
        :meth:`claim_and_attach_owner_run` is. The reservation they already
        checked is released the moment a plan ENDS, so it cannot speak for a
        request that ended without publishing: binding a run there consumed the
        older proposal behind the standing record, which no acknowledgement can
        undo. The refusal is decided before either branch writes anything.
        """
        profile = self._profile(profile)
        with self._conversation_lock:
            try:
                self._conn.execute("BEGIN IMMEDIATE")
                existing = self._conn.execute(
                    "SELECT fingerprint, run_id FROM run_idempotency "
                    "WHERE profile = ? AND session_scope = ? AND idempotency_key = ?",
                    (profile, session_scope, idempotency_key),
                ).fetchone()
                owner_profile = (
                    self._profile(owner["proposal_profile"])
                    if owner is not None else profile
                )
                if existing is not None:
                    if existing[0] != fingerprint:
                        self._conn.rollback()
                        return "conflict", None
                    existing_run_id = str(existing[1])
                    if owner is not None:
                        row = self._conn.execute(
                            "SELECT proposal_response_id, consumed_response_id, "
                            "claimed_response_id, claim_id, owner_run_id, claim_state, closed, "
                            "bound_operation, bound_payload_digest "
                            "FROM conversations WHERE profile = ? AND name = ?",
                            (owner_profile, owner["conversation"]),
                        ).fetchone()
                        reserved = self._active_owner_conversation_reservation_locked(
                            owner_profile, owner["conversation"],
                        )
                        if (
                            row is not None
                            and row[0] == owner["response_id"]
                            and row[1] != owner["response_id"]
                            and row[2] == owner["response_id"]
                            and row[3] == owner["claim_id"]
                            and row[4] == existing_run_id
                            and row[5] == "released"
                            and not bool(row[6])
                            and row[7] == owner["operation"]
                            and row[8] == owner["payload_digest"]
                            and reserved is None
                        ):
                            if self._unanswered_owner_request_locked(
                                owner_profile, owner["conversation"],
                            ):
                                self._conn.rollback()
                                return "authority_conflict", None
                            created_at = time.time()
                            queued_json = self._queued_run_status_json(
                                run_id, created_at,
                            )
                            self._conn.execute(
                                "UPDATE conversations SET owner_run_id = ?, "
                                "claim_state = 'claimed' WHERE profile = ? AND name = ? "
                                "AND proposal_response_id = ? AND owner_run_id = ? "
                                "AND claim_state = 'released'",
                                (
                                    run_id, owner_profile, owner["conversation"],
                                    owner["response_id"], existing_run_id,
                                ),
                            )
                            self._conn.execute(
                                "UPDATE run_idempotency SET run_id = ?, created_at = ?, "
                                "status_json = ?, terminal_json = NULL "
                                "WHERE profile = ? AND session_scope = ? "
                                "AND idempotency_key = ? AND fingerprint = ? AND run_id = ?",
                                (
                                    run_id, created_at, queued_json,
                                    profile, session_scope,
                                    idempotency_key, fingerprint, existing_run_id,
                                ),
                            )
                            if job_payload is not None:
                                self._reserve_owner_job_locked(
                                    "run", run_id, profile, job_payload,
                                )
                            self._conn.commit()
                            return "new", run_id
                    self._conn.rollback()
                    return "existing", existing_run_id
                if owner is not None:
                    row = self._conn.execute(
                        "SELECT proposal_response_id, consumed_response_id, "
                        "claimed_response_id, claim_id, owner_run_id, claim_state, closed, "
                        "claim_expires_at "
                        "FROM conversations WHERE profile = ? AND name = ?",
                        (owner_profile, owner["conversation"]),
                    ).fetchone()
                    reserved = self._active_owner_conversation_reservation_locked(
                        owner_profile, owner["conversation"],
                    )
                    if (
                        row is None
                        or row[0] != owner["response_id"]
                        or row[1] == owner["response_id"]
                        or bool(row[6])
                        or reserved is not None
                        or self._unanswered_owner_request_locked(
                            owner_profile, owner["conversation"],
                        )
                        or (
                            row[5] == "claimed"
                            and not (
                                row[4] is None
                                and isinstance(row[7], (int, float))
                                and row[7] <= time.time()
                            )
                            and (
                                row[2] != owner["response_id"]
                                or row[3] != owner["claim_id"]
                                or row[4] not in {None, run_id}
                            )
                        )
                    ):
                        self._conn.rollback()
                        return "authority_conflict", None
                    self._conn.execute(
                        "UPDATE conversations SET claimed_response_id = ?, claim_id = ?, "
                        "claim_expires_at = NULL, owner_run_id = ?, bound_operation = ?, "
                        "bound_payload_digest = ?, claim_state = 'claimed' "
                        "WHERE profile = ? AND name = ? AND proposal_response_id = ?",
                        (
                            owner["response_id"], owner["claim_id"], run_id,
                            owner["operation"], owner["payload_digest"],
                            owner_profile, owner["conversation"], owner["response_id"],
                        ),
                    )
                created_at = time.time()
                queued_json = self._queued_run_status_json(run_id, created_at)
                self._conn.execute(
                    "INSERT INTO run_idempotency (profile, session_scope, idempotency_key, "
                    "fingerprint, run_id, created_at, status_json) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (
                        profile, session_scope, idempotency_key,
                        fingerprint, run_id, created_at, queued_json,
                    ),
                )
                if job_payload is not None:
                    self._reserve_owner_job_locked(
                        "run", run_id, profile, job_payload,
                    )
                self._conn.commit()
                return "new", run_id
            except Exception:
                self._conn.rollback()
                raise

    @_owner_authority
    def reserve_owner_response(
        self,
        profile: str,
        session_scope: str,
        idempotency_key: str,
        fingerprint: str,
        conversation: str,
        response_id: str,
    ) -> "tuple[str, Optional[dict], Optional[str]]":
        """Claim one owner turn's idempotency key durably, or replay it.

        The five-minute in-process cache cannot own this. An owner turn mints a
        proposal whose response id later carries approval authority, so an
        exact retry after a restart — or after that cache expired — must replay
        the FIRST attempt rather than plan a second one. The key, its request
        fingerprint, the response id it minted and the body to replay therefore
        live here, next to the conversation state they authorize.

        Returns ``(outcome, replay_body, session_id)``:

        * ``"new"`` — this key is now reserved for ``response_id``; proceed.
        * ``"replay"`` — the original attempt's exact stored body.
        * ``"conflict"`` — the same key was used for a different request.
        * ``"incomplete"`` — a previous attempt reserved the key, minted a
          response and then died before recording a body. It is NOT re-run:
          replaying its stored response is the caller's only safe move, and a
          second proposal is never minted under a key that already produced one.

        A reservation whose response never reached the response store at all
        (a crash between reserving and storing) minted nothing, so this adopts
        the key for the new attempt instead of stranding it forever.
        """
        profile = self._profile(profile)
        with self._conversation_lock:
            try:
                self._conn.execute("BEGIN IMMEDIATE")
                outcome, replay, session, row = self._owner_response_state_locked(
                    profile, session_scope, idempotency_key, fingerprint,
                    conversation,
                )
                if outcome != "new":
                    self._conn.rollback()
                    return outcome, replay, session
                if row is not None:
                    self._conn.execute(
                        "UPDATE owner_response_idempotency SET response_id = ?, "
                        "state = 'reserved', replay_json = NULL, session_id = NULL, "
                        "created_at = ? WHERE profile = ? AND session_scope = ? "
                        "AND idempotency_key = ?",
                        (
                            response_id, time.time(),
                            profile, session_scope, idempotency_key,
                        ),
                    )
                    self._conn.commit()
                    return "new", None, None
                self._conn.execute(
                    "INSERT INTO owner_response_idempotency ("
                    "profile, session_scope, idempotency_key, fingerprint, "
                    "conversation, response_id, state, created_at"
                    ") VALUES (?, ?, ?, ?, ?, ?, 'reserved', ?)",
                    (
                        profile, session_scope, idempotency_key, fingerprint,
                        conversation, response_id, time.time(),
                    ),
                )
                self._conn.commit()
                return "new", None, None
            except Exception:
                self._conn.rollback()
                raise

    def _owner_response_state_locked(
        self,
        profile: str,
        session_scope: str,
        idempotency_key: str,
        fingerprint: str,
        conversation: str,
    ) -> "tuple[str, Optional[dict], Optional[str], Any]":
        """Classify one owner idempotency key, and return its row if any.

        Shared by the read-only lookup and the reservation so both answer from
        exactly the same rules; only the reservation writes.
        """
        row = self._conn.execute(
            "SELECT fingerprint, conversation, response_id, state, "
            "replay_json, session_id FROM owner_response_idempotency "
            "WHERE profile = ? AND session_scope = ? AND idempotency_key = ?",
            (profile, session_scope, idempotency_key),
        ).fetchone()
        if row is None:
            return "new", None, None, None
        if row[0] != fingerprint or row[1] != conversation:
            return "conflict", None, None, row
        if row[3] == "completed" and row[4]:
            try:
                replay = json.loads(row[4])
            except (json.JSONDecodeError, TypeError):
                replay = None
            if isinstance(replay, dict):
                return "replay", replay, row[5], row
        minted = self._conn.execute(
            "SELECT 1 FROM responses WHERE profile = ? AND response_id = ?",
            (profile, str(row[2])),
        ).fetchone()
        if minted is not None:
            return "incomplete", None, row[5], row
        # Reserved, but nothing was ever minted under it: adopting the key is
        # safe and is the only thing that keeps a crashed attempt from locking
        # this exact request out forever.
        return "new", None, None, row

    @_owner_authority
    @_response_store_locked
    def lookup_owner_response(
        self,
        profile: str,
        session_scope: str,
        idempotency_key: str,
        fingerprint: str,
        conversation: str,
    ) -> "tuple[str, Optional[dict], Optional[str]]":
        """Answer an exact retry before this request is treated as a new turn.

        A replay is not a new turn, so it must be recognized BEFORE the
        predecessor assertion is compared: the conversation has legitimately
        moved on since the original attempt — by that attempt's own reply — and
        refusing the retry as stale would hide the answer the owner already
        has. Read-only; the reservation stays the atomic authority.
        """
        outcome, replay, session, _row = self._owner_response_state_locked(
            self._profile(profile), session_scope, idempotency_key, fingerprint,
            conversation,
        )
        return outcome, replay, session

    @_owner_authority
    @_response_store_locked
    def complete_owner_response(
        self,
        profile: str,
        session_scope: str,
        idempotency_key: str,
        response_id: str,
        replay: dict,
        session_id: Optional[str],
    ) -> None:
        """Record the exact body an exact retry of this owner turn replays."""
        self._complete_owner_response_locked(
            self._profile(profile), session_scope, idempotency_key, response_id,
            replay, session_id,
        )
        self._conn.commit()

    def _complete_owner_response_locked(
        self,
        profile: str,
        session_scope: str,
        idempotency_key: str,
        response_id: str,
        replay: dict,
        session_id: Optional[str],
    ) -> None:
        """The replay record's write, WITHOUT committing (see
        :meth:`publish_owner_turn`)."""
        self._conn.execute(
            "UPDATE owner_response_idempotency SET state = 'completed', "
            "replay_json = ?, session_id = ? "
            "WHERE profile = ? AND session_scope = ? AND idempotency_key = ? "
            "AND response_id = ?",
            (
                json.dumps(replay, sort_keys=True, separators=(",", ":")),
                session_id, profile, session_scope, idempotency_key, response_id,
            ),
        )

    @_owner_authority
    @_response_store_locked
    def release_owner_response(
        self,
        profile: str,
        session_scope: str,
        idempotency_key: str,
        response_id: str,
    ) -> None:
        """Drop a reservation whose turn never minted anything.

        Only ever removes a row still in ``reserved`` naming this exact
        response id, so a completed record — the durable replay authority — is
        never dropped, and neither is a newer attempt's reservation.
        """
        profile = self._profile(profile)
        self._conn.execute(
            "DELETE FROM owner_response_idempotency "
            "WHERE profile = ? AND session_scope = ? AND idempotency_key = ? "
            "AND response_id = ? AND state = 'reserved'",
            (profile, session_scope, idempotency_key, response_id),
        )
        self._conn.commit()

    @_response_store_locked
    def purge_owner_response_idempotency(self, older_than: float) -> None:
        """Bound retention without evicting live owner authority.

        A record is only dropped once it is old AND its response is no longer
        the conversation's head, its outstanding proposal, or a claim — i.e.
        once replaying it could not grant authority over anything. An owner who
        comes back to a still-open proposal therefore still replays it.
        """
        self._conn.execute(
            "DELETE FROM owner_response_idempotency WHERE created_at < ? "
            "AND response_id NOT IN ("
            "  SELECT response_id FROM conversations "
            "  WHERE profile = owner_response_idempotency.profile "
            "  UNION SELECT proposal_response_id FROM conversations "
            "  WHERE profile = owner_response_idempotency.profile "
            "  UNION SELECT claimed_response_id FROM conversations "
            "  WHERE profile = owner_response_idempotency.profile"
            ")",
            (older_than,),
        )
        self._conn.commit()

    @staticmethod
    def _queued_run_status_json(run_id: str, created_at: float) -> str:
        return json.dumps(
            {
                "object": "hermes.run",
                "run_id": run_id,
                "status": "queued",
                "created_at": created_at,
                "updated_at": created_at,
            },
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        )

    @_owner_authority
    @_response_store_locked
    def lookup_run_idempotency(
        self,
        profile: str,
        session_scope: str,
        idempotency_key: str,
        fingerprint: str,
    ) -> "tuple[str, Optional[str]]":
        profile = self._profile(profile)
        row = self._conn.execute(
            "SELECT fingerprint, run_id FROM run_idempotency "
            "WHERE profile = ? AND session_scope = ? AND idempotency_key = ?",
            (profile, session_scope, idempotency_key),
        ).fetchone()
        if row is None:
            return "missing", None
        return (
            ("existing", str(row[1]))
            if row[0] == fingerprint
            else ("conflict", None)
        )

    @_owner_authority
    @_response_store_locked
    def run_idempotency_created_at(
        self,
        profile: str,
        session_scope: str,
        idempotency_key: str,
        run_id: str,
    ) -> "float | None":
        """Read the immutable creation time for one exact persisted run."""
        profile = self._profile(profile)
        row = self._conn.execute(
            "SELECT created_at FROM run_idempotency "
            "WHERE profile = ? AND session_scope = ? "
            "AND idempotency_key = ? AND run_id = ?",
            (profile, session_scope, idempotency_key, run_id),
        ).fetchone()
        if (
            row is None
            or isinstance(row[0], bool)
            or not isinstance(row[0], (int, float))
            or not math.isfinite(float(row[0]))
        ):
            return None
        return float(row[0])

    @_owner_authority
    @_response_store_locked
    def run_idempotency_status(
        self, profile: str, run_id: str,
    ) -> "Dict[str, Any] | None":
        """Read the durable queued — or terminal — state of a run.

        Every terminal state is admitted alongside ``queued``, because every one
        of them is persisted (see :meth:`persist_terminal_run_status`). A run
        that ended has to STOP reporting working, and this durable status is the
        one thing a poller can still read after the process that owned the run
        is gone; leaving the row saying ``queued`` made an ordinary failure,
        cancellation or completion look like work still in progress forever.
        """
        profile = self._profile(profile)
        rows = self._conn.execute(
            "SELECT status_json FROM run_idempotency "
            "WHERE profile = ? AND run_id = ? AND status_json IS NOT NULL LIMIT 2",
            (profile, run_id),
        ).fetchall()
        if len(rows) != 1:
            return None
        try:
            value = json.loads(rows[0][0])
        except (json.JSONDecodeError, TypeError, ValueError):
            return None
        allowed = {"object", "run_id", "status", "created_at", "updated_at"}
        if value.get("status") == "failed" if isinstance(value, dict) else False:
            allowed = allowed | {"error"}
        if (
            not isinstance(value, dict)
            or set(value) != allowed
            or value.get("object") != "hermes.run"
            or value.get("run_id") != run_id
            or _OWNER_RUN_RE.fullmatch(run_id) is None
            or value.get("status") not in _DURABLE_RUN_STATUSES
            or (
                value.get("status") == "failed"
                and not isinstance(value.get("error"), str)
            )
            or not all(
                isinstance(value.get(field), (int, float))
                and not isinstance(value.get(field), bool)
                and math.isfinite(float(value[field]))
                for field in ("created_at", "updated_at")
            )
        ):
            return None
        return value

    @_owner_authority
    @_response_store_locked
    def persist_terminal_run_status(
        self, profile: str, run_id: str, status: str,
    ) -> None:
        """Persist a run's terminal status and retire its job, in ONE transaction.

        The job row is the only record that says "somebody must still finish
        this run". Deleting it separately from the terminal status meant a
        restart found a durable row still saying ``queued`` with no executor and
        no recovery authority, so polling reported working forever.

        A failed run records one plain, non-diagnostic sentence. The exception
        text stays in the log and in this process's transport status; it is not
        copied into a durable row an owner surface can read.

        A run that already persisted a terminal receipt is left exactly as it
        is: that receipt outranks any transport-level status.
        """
        profile = self._profile(profile)
        if status not in _TERMINAL_RUN_STATUSES:
            raise ValueError("not a terminal run status")
        try:
            self._conn.execute("BEGIN IMMEDIATE")
            now = time.time()
            for row in self._conn.execute(
                "SELECT profile, session_scope, idempotency_key, created_at "
                "FROM run_idempotency WHERE run_id = ? AND terminal_json IS NULL",
                (run_id,),
            ).fetchall():
                record: Dict[str, Any] = {
                    "object": "hermes.run",
                    "run_id": run_id,
                    "status": status,
                    "created_at": (
                        float(row[3])
                        if isinstance(row[3], (int, float))
                        and not isinstance(row[3], bool)
                        and math.isfinite(float(row[3]))
                        else now
                    ),
                    "updated_at": now,
                }
                if status == "failed":
                    record["error"] = _OWNER_RUN_STOPPED_MESSAGE
                self._conn.execute(
                    "UPDATE run_idempotency SET status_json = ? "
                    "WHERE profile = ? AND session_scope = ? AND idempotency_key = ? "
                    "AND run_id = ? AND terminal_json IS NULL",
                    (
                        json.dumps(
                            record, sort_keys=True, separators=(",", ":"),
                            ensure_ascii=True,
                        ),
                        row[0], row[1], row[2], run_id,
                    ),
                )
            self._release_owner_job_locked("run", run_id)
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise

    @_owner_authority
    def persist_owner_run_completion(
        self,
        profile: str,
        session_scope: str,
        idempotency_key: str,
        run_id: str,
        receipt: str,
        *,
        created_at: float,
        owner: "dict[str, str] | None" = None,
    ) -> Dict[str, Any]:
        """Atomically persist the native receipt and consume its proposal."""
        profile = self._profile(profile)
        if (
            _OWNER_RUN_RE.fullmatch(run_id) is None
            or re.fullmatch(r"[A-Za-z0-9._:-]{1,200}", idempotency_key) is None
            or not isinstance(created_at, (int, float))
            or isinstance(created_at, bool)
            or not math.isfinite(float(created_at))
        ):
            raise ValueError("invalid owner run completion")
        try:
            receipt_value = json.loads(receipt)
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            raise ValueError("invalid owner run receipt") from exc
        if not isinstance(receipt_value, dict) or receipt_value.get("ok") is not True:
            raise ValueError("invalid owner run receipt")

        owner_profile = profile
        if owner is not None:
            required_owner_keys = {
                "proposal_profile", "conversation", "response_id", "claim_id",
                "operation", "payload_digest",
            }
            if not required_owner_keys.issubset(owner):
                raise ValueError("invalid owner proposal completion")
            owner_profile = self._profile(owner["proposal_profile"])
            if (
                not self._valid_owner_authority_ids(
                    owner_profile,
                    owner["conversation"],
                    owner["response_id"],
                    owner["claim_id"],
                    run_id,
                )
                or owner["operation"] not in {
                    "owner_task_graph_commit", "owner_project_plan_commit",
                }
                or re.fullmatch(r"[a-f0-9]{64}", owner["payload_digest"]) is None
            ):
                raise ValueError("invalid owner proposal completion")
        now = time.time()
        terminal = {
            "object": "hermes.run",
            "run_id": run_id,
            "status": "completed",
            "created_at": float(created_at),
            "updated_at": now,
            "output": receipt,
            "usage": {},
            "owner_mutation_committed": True,
            "last_event": "run.completed",
        }
        encoded = json.dumps(
            terminal, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        )
        with self._conversation_lock:
            try:
                self._conn.execute("BEGIN IMMEDIATE")
                if owner is not None:
                    proposal_cursor = self._conn.execute(
                        "UPDATE conversations SET consumed_response_id = ?, "
                        "claim_state = 'completed' "
                        "WHERE profile = ? AND name = ? AND proposal_response_id = ? "
                        "AND claimed_response_id = ? AND claim_id = ? AND owner_run_id = ? "
                        "AND bound_operation = ? AND bound_payload_digest = ? "
                        "AND claim_state IN ('claimed', 'completed')",
                        (
                            owner["response_id"], owner_profile,
                            owner["conversation"], owner["response_id"],
                            owner["response_id"], owner["claim_id"], run_id,
                            owner["operation"], owner["payload_digest"],
                        ),
                    )
                    if proposal_cursor.rowcount != 1:
                        self._conn.rollback()
                        raise RuntimeError("owner proposal completion is unavailable")
                cur = self._conn.execute(
                    "UPDATE run_idempotency SET terminal_json = ?, status_json = NULL "
                    "WHERE profile = ? AND session_scope = ? "
                    "AND idempotency_key = ? AND run_id = ?",
                    (
                        encoded, profile, session_scope, idempotency_key, run_id,
                    ),
                )
                if cur.rowcount != 1:
                    self._conn.rollback()
                    raise RuntimeError("owner run idempotency record is unavailable")
                # Same transaction as the terminal receipt: the recovery job row
                # is retired exactly when — and only when — this run becomes
                # durably terminal.
                self._release_owner_job_locked("run", run_id)
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise
        return terminal

    @_owner_authority
    @_response_store_locked
    def owner_run_completion(
        self, profile: str, run_id: str,
    ) -> "Dict[str, Any] | None":
        """Read one closed persisted owner completion, failing closed on drift."""
        profile = self._profile(profile)
        rows = self._conn.execute(
            "SELECT terminal_json FROM run_idempotency "
            "WHERE profile = ? AND run_id = ? AND terminal_json IS NOT NULL "
            "LIMIT 2",
            (profile, run_id),
        ).fetchall()
        if len(rows) != 1:
            return None
        try:
            value = json.loads(rows[0][0])
            output = value.get("output") if isinstance(value, dict) else None
            receipt = json.loads(output) if isinstance(output, str) else None
        except (json.JSONDecodeError, TypeError, ValueError):
            return None
        expected_keys = {
            "object", "run_id", "status", "created_at", "updated_at",
            "output", "usage", "owner_mutation_committed", "last_event",
        }
        if (
            not isinstance(value, dict)
            or set(value) != expected_keys
            or value.get("object") != "hermes.run"
            or value.get("run_id") != run_id
            or _OWNER_RUN_RE.fullmatch(run_id) is None
            or value.get("status") != "completed"
            or value.get("owner_mutation_committed") is not True
            or value.get("last_event") != "run.completed"
            or value.get("usage") != {}
            or not isinstance(receipt, dict)
            or receipt.get("ok") is not True
            or not all(
                isinstance(value.get(field), (int, float))
                and math.isfinite(float(value[field]))
                for field in ("created_at", "updated_at")
            )
        ):
            return None
        return value

    @_response_store_locked
    def _bound_owner_run_completion(
        self, run_id: str,
    ) -> "Dict[str, Any] | None":
        """Read the one terminal receipt bound by a conversation's opaque run id.

        Owner conversations live under the planner profile while their approved
        mutations execute under the executor profile. The conversation already
        supplies the unguessable bound run id; this lookup additionally requires
        that exactly one profile owns it, then delegates every receipt-shape check
        to the canonical reader above.
        """
        if not isinstance(run_id, str) or _OWNER_RUN_RE.fullmatch(run_id) is None:
            return None
        rows = self._conn.execute(
            "SELECT profile FROM run_idempotency "
            "WHERE run_id = ? AND terminal_json IS NOT NULL LIMIT 2",
            (run_id,),
        ).fetchall()
        if len(rows) != 1 or not isinstance(rows[0][0], str):
            return None
        return self.owner_run_completion(rows[0][0], run_id)

    @_owner_authority
    @_response_store_locked
    def reserve_owner_job(
        self, kind: str, job_key: str, profile: str, payload: Dict[str, Any],
    ) -> None:
        """Persist that THIS process is the executor of one queued owner job.

        Written before the in-memory executor exists, so the durable queued
        state a caller is about to be told about (a 202) is never the only
        record of work with nobody driving it. Released only in the same
        transaction that persists the work's terminal state.

        Used standalone only for work that has no other durable authority to
        commit alongside; owner responses and owner runs reserve their job
        INSIDE the transaction that commits that authority (see
        :meth:`accept_owner_background_response`,
        :meth:`reserve_run_idempotency`, :meth:`claim_and_attach_owner_run`).
        """
        self._reserve_owner_job_locked(kind, job_key, self._profile(profile), payload)
        self._conn.commit()

    def _reserve_owner_job_locked(
        self, kind: str, job_key: str, profile: str, payload: Dict[str, Any],
    ) -> None:
        """Write one job row and its lease, WITHOUT committing."""
        now = time.time()
        self._conn.execute(
            "INSERT INTO owner_executor_jobs ("
            "kind, job_key, profile, executor_id, executor_pid, payload, "
            "created_at, lease_expires_at"
            ") VALUES (?, ?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(kind, job_key) DO UPDATE SET "
            "executor_id = excluded.executor_id, "
            "executor_pid = excluded.executor_pid, "
            "payload = excluded.payload, created_at = excluded.created_at, "
            "lease_expires_at = excluded.lease_expires_at",
            (
                kind, job_key, profile, _EXECUTOR_ID, os.getpid(),
                json.dumps(payload, sort_keys=True, separators=(",", ":")),
                now, now + _OWNER_JOB_LEASE_SECONDS,
            ),
        )

    def _release_owner_job_locked(self, kind: str, job_key: str) -> None:
        """Drop one job row, WITHOUT committing.

        Only ever called from inside the transaction that persists the job's
        terminal state, so the row and the terminal fact appear (or fail to
        appear) together. A separate delete meant a publication that raised
        still dropped the recovery authority and left the work queued forever.
        """
        self._conn.execute(
            "DELETE FROM owner_executor_jobs WHERE kind = ? AND job_key = ?",
            (kind, job_key),
        )

    @_response_store_locked
    def renew_owner_job_leases(self, kind: str, job_keys: "Iterable[str]") -> None:
        """Heartbeat: extend this executor's lease on jobs it is still driving.

        The lease — not PID liveness — is what proves an executor is still
        there, so it has to be renewed while the work runs. Only rows this
        process actually owns are touched, so a heartbeat can never extend a
        dead sibling's claim.
        """
        keys = [str(key) for key in job_keys]
        if not keys:
            return
        expires = time.time() + _OWNER_JOB_LEASE_SECONDS
        self._conn.executemany(
            "UPDATE owner_executor_jobs SET lease_expires_at = ? "
            "WHERE kind = ? AND job_key = ? AND executor_id = ?",
            [(expires, kind, key, _EXECUTOR_ID) for key in keys],
        )
        self._conn.commit()

    @_response_store_locked
    def claim_orphaned_owner_jobs(
        self, kind: str, driving: "Iterable[str]" = (),
    ) -> "list[Dict[str, Any]]":
        """Lease every job of ``kind`` whose executor is gone, WITHOUT deleting it.

        Atomic: each returned row is re-leased to THIS process in the same
        transaction that selects it, so two gateway processes starting together
        cannot both recover the same job. The row itself survives until the
        transaction that makes its work terminal deletes it — deleting first
        meant a crash, a malformed payload, or an exception during
        terminalization permanently lost the only record of what to recover.

        ``driving`` names the jobs THIS process still has a live executor for.
        Everything else is reclaimable once it is old enough that a sibling
        reserving one right now is never mistaken for a dead executor:

        * one of this process's OWN rows with no live executor left — the task
          died without recording a terminal state, so nobody is coming back for
          it and only this process can tell;
        * a sibling's row whose lease has expired, or whose named process is
          provably gone. The lease is the load-bearing half: a recycled pid
          looks alive forever, so pid liveness alone could strand queued owner
          work permanently. The liveness check only makes recovery from a crash
          faster than the lease TTL.
        """
        live = {str(key) for key in driving}
        orphans: list[Dict[str, Any]] = []
        try:
            self._conn.execute("BEGIN IMMEDIATE")
            now = time.time()
            cutoff = now - _OWNER_JOB_REAP_MIN_AGE_SECONDS
            for row in self._conn.execute(
                "SELECT job_key, profile, executor_id, executor_pid, payload, "
                "lease_expires_at "
                "FROM owner_executor_jobs WHERE kind = ? AND created_at < ?",
                (kind, cutoff),
            ).fetchall():
                if str(row[0]) in live:
                    continue
                lease = row[5]
                leased = (
                    isinstance(lease, (int, float))
                    and not isinstance(lease, bool)
                    and float(lease) > now
                )
                if row[2] != _EXECUTOR_ID and leased and _process_is_alive(row[3]):
                    continue
                try:
                    payload = json.loads(row[4])
                except (json.JSONDecodeError, TypeError):
                    payload = None
                orphans.append({
                    "job_key": str(row[0]),
                    "profile": str(row[1]),
                    "payload": payload if isinstance(payload, dict) else {},
                })
            for orphan in orphans:
                self._conn.execute(
                    "UPDATE owner_executor_jobs SET executor_id = ?, "
                    "executor_pid = ?, lease_expires_at = ? "
                    "WHERE kind = ? AND job_key = ?",
                    (
                        _EXECUTOR_ID, os.getpid(),
                        now + _OWNER_JOB_LEASE_SECONDS, kind, orphan["job_key"],
                    ),
                )
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise
        return orphans

    @_owner_authority
    @_response_store_locked
    def fail_orphaned_owner_response(
        self, profile: str, conversation: str, response_id: str, message: str,
    ) -> bool:
        """Terminally fail one queued owner response nobody is executing.

        One transaction: the stored response becomes ``failed`` so polling
        stops, this turn's conversation reservation is dropped so the
        conversation is not locked, this exact turn's terminal failure becomes
        the durable body its own idempotency key replays, and the recovery job
        row is retired. The conversation head is deliberately left alone — a
        turn that never completed never owned it — and nothing is touched at all
        if this response DID become the head.

        The idempotency record is UPDATED, never deleted. An owner turn's key is
        immutable authority: it already minted this response id, so deleting the
        record let an exact retry mint a SECOND turn for the same submission
        instead of being told what happened to the first. Recording the terminal
        failure under the original key is what makes the retry replay it.
        """
        profile = self._profile(profile)
        try:
            self._conn.execute("BEGIN IMMEDIATE")
            head = self._conn.execute(
                "SELECT response_id FROM conversations WHERE profile = ? AND name = ?",
                (profile, conversation),
            ).fetchone()
            if head is not None and head[0] == response_id:
                # It completed after all; its publication is the authority.
                self._conn.rollback()
                return False
            stored = self._conn.execute(
                "SELECT data FROM responses WHERE profile = ? AND response_id = ?",
                (profile, response_id),
            ).fetchone()
            record: Dict[str, Any] = {}
            if stored is not None:
                try:
                    parsed = json.loads(stored[0])
                except (json.JSONDecodeError, TypeError):
                    parsed = None
                if isinstance(parsed, dict):
                    record = parsed
            response = record.get("response")
            failed = dict(response) if isinstance(response, dict) else {
                "id": response_id, "object": "response",
            }
            if failed.get("status") in {"completed", "failed", "incomplete"}:
                self._conn.rollback()
                return False
            failed.update({
                "status": "failed",
                "output": [],
                "error": {"code": "server_error", "message": message},
            })
            record["response"] = failed
            self._put_response_locked(response_id, record, profile)
            self._retire_interrupted_owner_turn_locked(
                profile, conversation, response_id, failed,
            )
            self._release_owner_job_locked("response", response_id)
            self._conn.commit()
            return True
        except OwnerTurnNotRecoverable:
            # Not an error, and not a decision either: this pass cannot end the
            # turn without losing the request behind it. Everything rolls back,
            # the job row stays, and the next sweep tries again.
            self._conn.rollback()
            return False
        except Exception:
            self._conn.rollback()
            raise

    def _retire_interrupted_owner_turn_locked(
        self,
        profile: str,
        conversation: str,
        response_id: str,
        terminal_response: Dict[str, Any],
    ) -> bool:
        """Leave one turn that ended without publishing fully recoverable.

        Inside the CALLER's transaction, so this lands with the terminal body it
        describes and never separately from it: the fence becomes the record
        carrying the owner's request, and this exact terminal outcome becomes
        what the turn's own idempotency key replays.

        A turn that DID take the head is answered, and nothing here may
        contradict that: it returns ``False`` having written nothing.

        Otherwise this ending MUST leave a way back. The conversion is what
        produces it, and a conversion that finds no fence produces nothing — so
        unless this exact request is already recorded as recoverable, the whole
        transaction is refused with :class:`OwnerTurnNotRecoverable`. Ignoring
        that let the terminal body, the replay record and the job retirement
        land while the owner's request itself vanished.

        The idempotency record is UPDATED, never deleted. An owner turn's key is
        immutable authority: it already minted this response id, so deleting the
        record let an exact retry mint a SECOND turn for the same submission
        instead of being told what happened to the first.
        """
        head = self._conn.execute(
            "SELECT response_id FROM conversations WHERE profile = ? AND name = ?",
            (profile, conversation),
        ).fetchone()
        if head is not None and head[0] == response_id:
            return False
        if not self._convert_owner_reservation_to_recovery_locked(
            profile, conversation, response_id,
        ) and self._conn.execute(
            "SELECT 1 FROM owner_conversation_recovery "
            "WHERE profile = ? AND name = ? AND response_id = ?",
            (profile, conversation, response_id),
        ).fetchone() is None:
            raise OwnerTurnNotRecoverable(
                "owner turn would end with neither a published turn nor a "
                "recovery record",
            )
        self._conn.execute(
            "UPDATE owner_response_idempotency SET state = 'completed', "
            "replay_json = ? "
            "WHERE profile = ? AND conversation = ? AND response_id = ?",
            (
                json.dumps(
                    terminal_response, sort_keys=True, separators=(",", ":"),
                ),
                profile, conversation, response_id,
            ),
        )
        return True

    @_owner_authority
    @_response_store_locked
    def fail_orphaned_owner_run(
        self, profile: str, run_id: str, owner: "Optional[Dict[str, str]]",
    ) -> bool:
        """Terminally fail one queued owner run nobody is executing.

        One transaction: the run's durable status becomes ``failed`` so polling
        stops reporting working forever, its owner proposal claim is released so
        the owner can approve the same proposal again, and its recovery job row
        is retired. A run that already persisted its terminal receipt is left
        exactly as it is — and its job row stays, so nothing is lost if this
        recovery pass cannot decide.
        """
        profile = self._profile(profile)
        if _OWNER_RUN_RE.fullmatch(run_id) is None:
            return False
        try:
            self._conn.execute("BEGIN IMMEDIATE")
            rows = self._conn.execute(
                "SELECT profile, session_scope, idempotency_key, terminal_json "
                "FROM run_idempotency WHERE run_id = ?",
                (run_id,),
            ).fetchall()
            if any(row[3] is not None for row in rows):
                self._conn.rollback()
                return False
            now = time.time()
            for row in rows:
                self._conn.execute(
                    "UPDATE run_idempotency SET status_json = ? "
                    "WHERE profile = ? AND session_scope = ? AND idempotency_key = ? "
                    "AND run_id = ? AND terminal_json IS NULL",
                    (
                        _failed_run_status_json(run_id, now),
                        row[0], row[1], row[2], run_id,
                    ),
                )
            if owner:
                self._conn.execute(
                    "UPDATE conversations SET claim_state = 'released', "
                    "claim_expires_at = NULL WHERE profile = ? AND name = ? "
                    "AND proposal_response_id = ? AND consumed_response_id IS NOT ? "
                    "AND claimed_response_id = ? AND claim_id = ? AND owner_run_id = ? "
                    "AND claim_state = 'claimed'",
                    (
                        self._profile(owner.get("proposal_profile")),
                        owner.get("conversation"),
                        owner.get("response_id"), owner.get("response_id"),
                        owner.get("response_id"), owner.get("claim_id"), run_id,
                    ),
                )
            self._release_owner_job_locked("run", run_id)
            self._conn.commit()
            return True
        except Exception:
            self._conn.rollback()
            raise

    def purge_run_idempotency(self, older_than: float) -> None:
        with self._conversation_lock:
            self._conn.execute(
                "DELETE FROM run_idempotency WHERE created_at < ?",
                (older_than,),
            )
            self._conn.commit()

    def set_conversation(
        self,
        name: str,
        response_id: str,
        *,
        owner_proposal: bool = False,
        profile: Optional[str] = None,
        reservation_id: Optional[str] = None,
        expected_previous_response_id: Any = _UNSTATED,
    ) -> bool:
        """Map a conversation unless approved work or an explicit close owns it.

        ``expected_previous_response_id`` is the same assertion the turn's
        reservation compared, re-compared here in the mapping transaction, so a
        turn can only ever append to the exact predecessor it planned against —
        never overwrite a newer one that appeared while it was running.
        """
        profile = self._profile(profile)
        is_owner = (
            isinstance(name, str)
            and _OWNER_CONVERSATION_RE.fullmatch(name) is not None
        )
        if is_owner:
            # Mapping an owner conversation publishes the proposal handle that
            # a later approval is granted against, so it is authority too.
            # Generic conversation names are unaffected.
            self._require_durable_owner_authority()
        with self._conversation_lock:
            try:
                self._conn.execute("BEGIN IMMEDIATE")
                mapped = self._set_conversation_locked(
                    name,
                    response_id,
                    owner_proposal=owner_proposal,
                    profile=profile,
                    reservation_id=reservation_id,
                    expected_previous_response_id=expected_previous_response_id,
                )
                if not mapped:
                    self._conn.rollback()
                    return False
                self._conn.commit()
                return True
            except Exception:
                self._conn.rollback()
                raise

    @_owner_authority
    @_response_store_locked
    def accept_owner_background_response(
        self,
        *,
        profile: Optional[str],
        response_id: str,
        data: Dict[str, Any],
        conversation: Optional[str],
        replay: Optional[Dict[str, Any]] = None,
        session_scope: Optional[str] = None,
        idempotency_key: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> None:
        """Commit everything a background 202 promises, as ONE transaction.

        The queued response body, the durable executor recovery job, and the
        replay record an exact retry answers from are three facts about the same
        acceptance. Committing them separately meant a crash between any two
        left an accepted owner response with no executor to drive it — queued
        forever — or a reserved idempotency key that had minted a response but
        could never record a body, so the key was permanently unusable and its
        exact retry could only ever replay a queued turn.
        """
        profile = self._profile(profile)
        try:
            self._conn.execute("BEGIN IMMEDIATE")
            self._put_response_locked(response_id, data, profile)
            if conversation is not None:
                self._reserve_owner_job_locked(
                    "response", response_id, profile,
                    {"conversation": str(conversation)},
                )
            if (
                replay is not None
                and session_scope is not None
                and idempotency_key is not None
            ):
                self._complete_owner_response_locked(
                    profile, session_scope, idempotency_key, response_id,
                    replay, session_id,
                )
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise

    @_owner_authority
    @_response_store_locked
    def store_terminal_owner_response(
        self,
        *,
        profile: Optional[str],
        response_id: str,
        data: Dict[str, Any],
        release_job: bool,
        conversation: Optional[str] = None,
        interrupted: bool = False,
    ) -> None:
        """Persist one terminal background response and everything it ends.

        The job row is the ONLY record that says "somebody must finish this
        response". It may therefore be dropped only in the same transaction that
        makes the response terminal: if this write fails the row survives and a
        later sweep recovers the work, instead of the response being left queued
        forever with its recovery authority already deleted.

        ``interrupted`` marks the endings that produced a terminal response and
        no turn. For those, this turn's fence becoming the record that carries
        the owner's request — and this outcome becoming what its idempotency key
        replays — are facts about the SAME ending, committed here with it. They
        were a second transaction, and a crash in between left a failed turn
        with nothing left to say what the owner had sent.
        """
        profile = self._profile(profile)
        try:
            self._conn.execute("BEGIN IMMEDIATE")
            self._put_response_locked(response_id, data, profile)
            if interrupted and conversation is not None:
                self._retire_interrupted_owner_turn_locked(
                    profile, conversation, response_id, data.get("response") or {},
                )
            if release_job:
                self._release_owner_job_locked("response", response_id)
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise

    @_owner_authority
    def publish_owner_turn(
        self,
        *,
        profile: Optional[str],
        conversation: str,
        response_id: str,
        data: Dict[str, Any],
        owner_proposal: bool,
        reservation_id: Optional[str],
        expected_previous_response_id: Any,
        replay: Optional[Dict[str, Any]] = None,
        session_scope: Optional[str] = None,
        idempotency_key: Optional[str] = None,
        session_id: Optional[str] = None,
        release_job: bool = False,
    ) -> bool:
        """Commit one owner turn's whole publication as ONE transaction.

        The response body, the conversation head compare-and-swap, the release
        of this turn's reservation and the durable replay record an exact retry
        answers from are four facts about the same event. Committing them
        separately meant a crash between any two left a turn that had really
        happened but that an exact retry could not replay — it would plan a
        second one — or a published head with no durable replay behind it.

        Returns ``False`` (having written nothing) when the head is no longer
        this turn's to take, exactly like :meth:`set_conversation`.
        """
        profile = self._profile(profile)
        with self._conversation_lock:
            try:
                self._conn.execute("BEGIN IMMEDIATE")
                self._put_response_locked(response_id, data, profile)
                if not self._set_conversation_locked(
                    conversation,
                    response_id,
                    owner_proposal=owner_proposal,
                    profile=profile,
                    reservation_id=reservation_id,
                    expected_previous_response_id=expected_previous_response_id,
                ):
                    self._conn.rollback()
                    return False
                if (
                    replay is not None
                    and session_scope is not None
                    and idempotency_key is not None
                ):
                    self._complete_owner_response_locked(
                        profile, session_scope, idempotency_key, response_id,
                        replay, session_id,
                    )
                if release_job:
                    # Same transaction as the publication it proves. The job row
                    # is retired only once this turn is durably terminal.
                    self._release_owner_job_locked("response", response_id)
                self._conn.commit()
                return True
            except Exception:
                self._conn.rollback()
                raise

    def _set_conversation_locked(
        self,
        name: str,
        response_id: str,
        *,
        owner_proposal: bool,
        profile: str,
        reservation_id: Optional[str],
        expected_previous_response_id: Any,
    ) -> bool:
        """The mapping decision and write, inside the CALLER's transaction.

        Split out of :meth:`set_conversation` so an owner turn can commit its
        response, this head compare-and-swap, its reservation release and its
        durable replay record together (see :meth:`publish_owner_turn`).
        Returns ``False`` without writing when the head is not this turn's to
        take; the caller owns the rollback.
        """
        is_owner = (
            isinstance(name, str)
            and _OWNER_CONVERSATION_RE.fullmatch(name) is not None
        )
        if (
            is_owner
            and self._conn.execute(
                "SELECT 1 FROM owner_conversation_closures "
                "WHERE profile = ? AND name = ?",
                (profile, name),
            ).fetchone() is not None
        ):
            return False
        if is_owner:
            reservation = self._active_owner_conversation_reservation_locked(
                profile, name,
            )
            if (
                reservation_id is not None
                and (
                    reservation is None
                    or reservation[0] != reservation_id
                )
            ) or (
                reservation_id is None and reservation is not None
            ):
                return False
        row = self._conn.execute(
            "SELECT response_id, proposal_response_id, consumed_response_id, claimed_response_id, "
            "claim_state, closed FROM conversations WHERE profile = ? AND name = ?",
            (profile, name),
        ).fetchone()
        if expected_previous_response_id is not _UNSTATED and (
            (row[0] if row is not None else None)
            not in (expected_previous_response_id, response_id)
        ):
            # Anything but the stated predecessor (or this same turn,
            # already mapped) means a newer turn owns the conversation.
            return False
        if (
            row is not None
            and row[0] != response_id
            and is_owner
            and (
                bool(row[5])
                or (
                    row[1] is not None
                    and row[3] == row[1]
                    and row[4] == "claimed"
                    and row[2] != row[1]
                )
            )
        ):
            return False
        proposal_digest = None
        if owner_proposal:
            stored = self._conn.execute(
                "SELECT data FROM responses WHERE response_id = ? AND profile = ?",
                (response_id, profile),
            ).fetchone()
            if stored is None:
                return False
            try:
                raw = json.loads(stored[0])
            except (json.JSONDecodeError, TypeError):
                return False
            proposal_digest = _owner_proposal_digest(
                _owner_final_proposal(
                    raw.get("conversation_history")
                    if isinstance(raw, dict) else None
                )
            )
            if proposal_digest is None:
                return False
        self._conn.execute(
            "INSERT INTO conversations ("
            "profile, name, response_id, proposal_response_id, proposal_digest, "
            "consumed_response_id, claimed_response_id, claim_id, claim_expires_at, "
            "owner_run_id, bound_operation, bound_payload_digest, claim_state, closed"
            ") VALUES (?, ?, ?, ?, ?, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, 0) "
            "ON CONFLICT(profile, name) DO UPDATE SET "
            "response_id = excluded.response_id, "
            "proposal_response_id = CASE "
            "WHEN conversations.response_id = excluded.response_id "
            "THEN COALESCE(excluded.proposal_response_id, conversations.proposal_response_id) "
            "ELSE excluded.proposal_response_id END, "
            "proposal_digest = CASE WHEN conversations.response_id = excluded.response_id "
            "THEN COALESCE(excluded.proposal_digest, conversations.proposal_digest) "
            "ELSE excluded.proposal_digest END, "
            "consumed_response_id = conversations.consumed_response_id, "
            "claimed_response_id = CASE "
            "WHEN conversations.response_id = excluded.response_id "
            "THEN conversations.claimed_response_id ELSE NULL END, "
            "claim_id = CASE WHEN conversations.response_id = excluded.response_id "
            "THEN conversations.claim_id ELSE NULL END, "
            "claim_expires_at = CASE WHEN conversations.response_id = excluded.response_id "
            "THEN conversations.claim_expires_at ELSE NULL END, "
            "owner_run_id = CASE WHEN conversations.response_id = excluded.response_id "
            "THEN conversations.owner_run_id ELSE NULL END, "
            "bound_operation = CASE WHEN conversations.response_id = excluded.response_id "
            "THEN conversations.bound_operation ELSE NULL END, "
            "bound_payload_digest = CASE WHEN conversations.response_id = excluded.response_id "
            "THEN conversations.bound_payload_digest ELSE NULL END, "
            "claim_state = CASE WHEN conversations.response_id = excluded.response_id "
            "THEN conversations.claim_state ELSE NULL END, "
            "closed = CASE WHEN conversations.response_id = excluded.response_id "
            "THEN conversations.closed ELSE 0 END",
            (
                profile, name, response_id,
                response_id if owner_proposal else None,
                proposal_digest,
            ),
        )
        if is_owner:
            # Same transaction as the mapping: a conversation that exists must
            # have its immutable place in its group, or "which session is
            # current" would depend on read order again.
            group, _session = _owner_conversation_group(name)
            if group is not None:
                self._record_owner_session_locked(profile, group, name)
        if reservation_id is not None:
            self._conn.execute(
                "DELETE FROM owner_conversation_reservations "
                "WHERE profile = ? AND name = ? AND response_id = ?",
                (profile, name, reservation_id),
            )
        return True
    @_response_store_locked
    def close(self) -> None:
        """Close the database connection."""
        try:
            self._conn.close()
        except Exception:
            pass

    @_response_store_locked
    def __len__(self) -> int:
        row = self._conn.execute("SELECT COUNT(*) FROM responses").fetchone()
        return row[0] if row else 0


# ---------------------------------------------------------------------------
# CORS middleware
# ---------------------------------------------------------------------------

_CORS_HEADERS = {
    "Access-Control-Allow-Methods": "GET, POST, DELETE, OPTIONS",
    "Access-Control-Allow-Headers": "Authorization, Content-Type, Idempotency-Key",
}


if AIOHTTP_AVAILABLE:
    @web.middleware
    async def cors_middleware(request, handler):
        """Add CORS headers for explicitly allowed origins; handle OPTIONS preflight."""
        adapter = request.app.get("api_server_adapter")
        origin = request.headers.get("Origin", "")
        cors_headers = None
        if adapter is not None:
            if not adapter._origin_allowed(origin):
                return web.Response(status=403)
            cors_headers = adapter._cors_headers_for_origin(origin)

        if request.method == "OPTIONS":
            if cors_headers is None:
                return web.Response(status=403)
            return web.Response(status=200, headers=cors_headers)

        response = await handler(request)
        if cors_headers is not None:
            response.headers.update(cors_headers)
        return response
else:
    cors_middleware = None  # type: ignore[assignment]


_MEDIA_IMG_EXT = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"}
_MEDIA_MIME = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
    ".bmp": "image/bmp",
}
_MEDIA_DATA_URL_MAX_BYTES = 5 * 1024 * 1024  # skip images larger than 5MB


def _resolve_media_to_data_urls(text: str) -> str:
    """Replace ``MEDIA:<path>`` image tags with inline base64 data URLs.

    Remote OpenAI-compatible frontends can't read local file paths, so
    ``MEDIA:`` tags referencing images on the server are useless to them.
    Inline small local images as markdown data URLs; non-image or unreadable
    paths are left untouched.

    Uses the same anchored ``MEDIA_TAG_CLEANUP_RE`` matcher and
    ``validate_media_delivery_path`` safety check every other platform
    adapter's media delivery already goes through (gateway/platforms/base.py)
    — an absolute-path anchor plus a known-extension requirement, and a
    resolved-path check against the credential/system-path denylist. The
    prior pattern here matched any bare token after ``MEDIA:`` (including a
    relative/traversal path like ``../../etc/passwd.png``) and read the file
    directly with no denylist, so any image-suffixed, readable file the
    process could see was base64-exfiltrated to the API caller if its path
    merely appeared in the model's own final reply text.
    """
    if not text or "MEDIA:" not in text:
        return text
    import base64

    def _to_data_url(path_str: str) -> Optional[str]:
        # validate_media_delivery_path() strips wrapping quotes/backticks
        # and trailing punctuation internally, same as MEDIA_TAG_CLEANUP_RE's
        # other callers (extract_media / _strip_media_tag_directives) rely on.
        safe_path = validate_media_delivery_path(path_str)
        if not safe_path:
            return None
        p = Path(safe_path)
        suffix = p.suffix.lower()
        if suffix not in _MEDIA_IMG_EXT:
            return None
        try:
            if p.stat().st_size > _MEDIA_DATA_URL_MAX_BYTES:
                return None
            b64 = base64.b64encode(p.read_bytes()).decode()
        except OSError:
            return None
        return f"![image](data:{_MEDIA_MIME[suffix]};base64,{b64})"

    def _repl(m: "re.Match[str]") -> str:
        return _to_data_url(m.group("path")) or m.group(0)

    try:
        return MEDIA_TAG_CLEANUP_RE.sub(_repl, text)
    except Exception:
        return text


def _redact_api_error_text(value: Any, *, limit: int | None = None) -> str:
    """Redact API-bound error text before it crosses the HTTP boundary."""
    redacted = redact_sensitive_text(str(value), force=True)
    if limit is not None:
        return redacted[:limit]
    return redacted


def _process_is_alive(pid: Any) -> bool:
    """Whether ``pid`` names a live process on this host.

    Uses the shared :func:`gateway.status._pid_exists`, which is the one
    cross-platform liveness check in this tree that does not kill the target
    (``os.kill(pid, 0)`` sends Ctrl+C to the whole console process group on
    Windows — see that function's own note).

    ``True`` when liveness cannot be decided at all, because reaping another
    process's LIVE job is worse than leaving an orphan for the next start.
    """
    try:
        value = int(pid)
    except (TypeError, ValueError):
        return False
    if value <= 0:
        return False
    try:
        from gateway.status import _pid_exists

        return _pid_exists(value)
    except Exception:
        return True


def _failed_run_status_json(run_id: str, updated_at: float) -> str:
    """The terminal status a recovered orphan run reports to a poller."""
    return json.dumps(
        {
            "object": "hermes.run",
            "run_id": run_id,
            "status": "failed",
            "created_at": updated_at,
            "updated_at": updated_at,
            "error": _OWNER_ORPHAN_RUN_MESSAGE,
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )


def _openai_error(message: str, err_type: str = "invalid_request_error", param: str = None, code: str = None) -> Dict[str, Any]:
    """OpenAI-style error envelope."""
    return {
        "error": {
            "message": _redact_api_error_text(message),
            "type": err_type,
            "param": param,
            "code": code,
        }
    }


_api_agent_request_reservation: ContextVar[Optional[dict[str, bool]]] = ContextVar(
    "api_agent_request_reservation", default=None
)


def _admit_api_agent_request(handler):
    """Reserve an authenticated API turn before its handler first awaits.

    Gateway shutdown and aiohttp requests share an event loop. Keeping the
    drain check and reservation in one non-awaiting block prevents a request
    admitted immediately before shutdown from becoming invisible while it is
    still parsing its body or resolving session state. The mutable reservation
    is intentionally shared with child tasks so agent/task bookkeeping releases
    this one slot exactly once.
    """
    @wraps(handler)
    async def _wrapped(self, request, *args, **kwargs):
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err
        draining = self._draining_response()
        if draining is not None:
            return draining
        reservation = {"active": True}
        token = _api_agent_request_reservation.set(reservation)
        self._pending_agent_requests += 1
        try:
            return await handler(self, request, *args, **kwargs)
        finally:
            if reservation["active"]:
                reservation["active"] = False
                self._pending_agent_requests = max(0, self._pending_agent_requests - 1)
            _api_agent_request_reservation.reset(token)

    return _wrapped


def _release_pending_api_work(adapter, reservation: dict[str, bool]) -> None:
    """Release a pending-work reservation exactly once."""
    if reservation["active"]:
        reservation["active"] = False
        adapter._pending_agent_requests = max(0, adapter._pending_agent_requests - 1)


@contextmanager
def _reserve_pending_api_work(adapter):
    """Keep externally-triggered background work visible across awaits.

    A handler can detach the reservation to an asyncio task; its done callback
    then owns release so shutdown cannot miss the handoff to background work.
    """
    reservation = {"active": True, "detached": False}
    adapter._pending_agent_requests += 1
    try:
        yield reservation
    finally:
        if not reservation["detached"]:
            _release_pending_api_work(adapter, reservation)


if AIOHTTP_AVAILABLE:
    @web.middleware
    async def body_limit_middleware(request, handler):
        """Reject overly large request bodies early based on Content-Length."""
        if request.method in {"POST", "PUT", "PATCH"}:
            cl = request.headers.get("Content-Length")
            if cl is not None:
                try:
                    if int(cl) > MAX_REQUEST_BYTES:
                        return web.json_response(_openai_error("Request body too large.", code="body_too_large"), status=413)
                except ValueError:
                    return web.json_response(_openai_error("Invalid Content-Length header.", code="invalid_content_length"), status=400)
        try:
            return await handler(request)
        except web.HTTPRequestEntityTooLarge:
            # aiohttp's client_max_size tripped mid-read (chunked bodies carry
            # no Content-Length) — return a proper 413 instead of letting the
            # handler's broad JSON except turn it into 400 "Invalid JSON".
            return web.json_response(
                _openai_error("Request body too large.", code="body_too_large"),
                status=413,
            )
else:
    body_limit_middleware = None  # type: ignore[assignment]

_SECURITY_HEADERS = {
    "Content-Security-Policy": "default-src 'none'; frame-ancestors 'none'",
    "Permissions-Policy": "camera=(), microphone=(), geolocation=()",
    "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "X-XSS-Protection": "0",
    "Referrer-Policy": "no-referrer",
}


if AIOHTTP_AVAILABLE:
    @web.middleware
    async def security_headers_middleware(request, handler):
        """Add security headers to all responses (including errors)."""
        response = await handler(request)
        for k, v in _SECURITY_HEADERS.items():
            response.headers.setdefault(k, v)
        return response
else:
    security_headers_middleware = None  # type: ignore[assignment]


class _IdempotencyConflict(Exception):
    """One idempotency key was reused for a different request."""


class _OwnerConversationReservationChanged(Exception):
    """The exact owner turn lost its durable conversation reservation."""


class _IdempotencyCache:
    """In-memory idempotency cache with TTL and basic LRU semantics."""
    def __init__(self, max_items: int = 1000, ttl_seconds: int = 300):
        from collections import OrderedDict
        self._store = OrderedDict()
        self._inflight: Dict[tuple[str, str], "asyncio.Task[Any]"] = {}
        self._ttl = ttl_seconds
        self._max = max_items

    def _purge(self):
        now = time.time()
        expired = [k for k, v in self._store.items() if now - v["ts"] > self._ttl]
        for k in expired:
            self._store.pop(k, None)
        while len(self._store) > self._max:
            self._store.popitem(last=False)

    async def get_existing(self, key: str, fingerprint: str):
        """Replay a completed or in-flight exact request without recomputing."""
        self._purge()
        item = self._store.get(key)
        if item:
            if item["fp"] != fingerprint:
                raise _IdempotencyConflict
            return item["resp"]
        inflight_key = (key, fingerprint)
        task = self._inflight.get(inflight_key)
        if task is not None:
            return await asyncio.shield(task)
        if any(stored_key == key for stored_key, _ in self._inflight):
            raise _IdempotencyConflict
        return None

    async def get_or_set(self, key: str, fingerprint: str, compute_coro):
        self._purge()
        item = self._store.get(key)
        if item:
            if item["fp"] != fingerprint:
                raise _IdempotencyConflict
            return item["resp"]

        inflight_key = (key, fingerprint)
        if any(stored_key == key for stored_key, _ in self._inflight):
            if inflight_key not in self._inflight:
                raise _IdempotencyConflict
        task = self._inflight.get(inflight_key)
        if task is None:
            async def _compute_and_store():
                resp = await compute_coro()
                import time as _t
                self._store[key] = {"resp": resp, "fp": fingerprint, "ts": _t.time()}
                self._purge()
                return resp

            task = asyncio.create_task(_compute_and_store())
            self._inflight[inflight_key] = task

            def _clear_inflight(done_task: "asyncio.Task[Any]") -> None:
                if self._inflight.get(inflight_key) is done_task:
                    self._inflight.pop(inflight_key, None)

            task.add_done_callback(_clear_inflight)

        return await asyncio.shield(task)


_idem_cache = _IdempotencyCache()


def _make_request_fingerprint(body: Dict[str, Any], keys: List[str]) -> str:
    from hashlib import sha256
    subset = {k: body.get(k) for k in keys}
    canonical = json.dumps(
        subset, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
    )
    return sha256(canonical.encode("utf-8")).hexdigest()


def _derive_chat_session_id(
    system_prompt: Optional[str],
    first_user_message: str,
) -> str:
    """Derive a stable session ID from the conversation's first user message.

    OpenAI-compatible frontends (Open WebUI, LibreChat, etc.) send the full
    conversation history with every request.  The system prompt and first user
    message are constant across all turns of the same conversation, so hashing
    them produces a deterministic session ID that lets the API server reuse
    the same Hermes session (and therefore the same Docker container sandbox
    directory) across turns.
    """
    seed = f"{system_prompt or ''}\n{first_user_message}"
    digest = hashlib.sha256(seed.encode("utf-8")).hexdigest()[:16]
    return f"api-{digest}"


_CRON_AVAILABLE = False
try:
    from cron.jobs import (
        list_jobs as _cron_list,
        get_job as _cron_get,
        update_job as _cron_update,
        remove_job as _cron_remove,
        pause_job as _cron_pause,
        resume_job as _cron_resume,
        trigger_job as _cron_trigger,
    )
    from cron.scheduler import (
        CronSchedulerRegistrationError as _CronSchedulerRegistrationError,
        create_job_with_scheduler_registration as _cron_create,
    )
    _CRON_AVAILABLE = True
except ImportError:
    _cron_list = None
    _cron_get = None
    _cron_create = None
    _cron_update = None
    _cron_remove = None
    _cron_pause = None
    _cron_resume = None
    _cron_trigger = None

    class _CronSchedulerRegistrationError(RuntimeError):
        pass


def _notify_cron_provider_jobs_changed() -> None:
    """Tell the active cron scheduler provider the job set changed after a REST
    mutation (no-op for the built-in). Best-effort — never breaks the handler."""
    try:
        from cron.scheduler import _notify_provider_jobs_changed
        _notify_provider_jobs_changed()
    except Exception:
        pass

# Defense-in-depth: mirror the agent-facing cronjob tool, which scans the
# user-supplied prompt for exfiltration/injection payloads at create/update
# time (tools/cronjob_tools.py).  The REST cron endpoints are authenticated
# (every handler runs _check_auth, and connect() refuses to start without
# API_SERVER_KEY), so this is not the trust boundary — it's parity with the
# tool path so a malicious prompt is rejected the same way regardless of
# which surface created the job.  Imported defensively: a missing scanner
# must not disable the cron REST API.
try:
    from tools.cronjob_tools import _scan_cron_prompt as _scan_cron_prompt
except Exception:  # pragma: no cover - scanner is optional hardening
    _scan_cron_prompt = None


class _ProviderAuthResolutionError(RuntimeError):
    """Raised only when gateway.run._resolve_runtime_agent_kwargs() fails
    to resolve provider credentials.

    That function is the sole raiser of RuntimeError(format_runtime_
    provider_error(...)) anywhere in _create_agent()'s call graph.
    Re-raising it as this dedicated subclass -- instead of catching bare
    RuntimeError around the much wider _create_agent()+run_conversation()
    span -- lets callers distinguish "provider auth/credential failure"
    from any other RuntimeError a provider adapter or run_conversation()
    might legitimately raise (e.g. run_agent.py's "Failed to recreate
    closed OpenAI client"), which a bare `except RuntimeError` there would
    otherwise mislabel as an auth failure.
    """


class _AgentTurnInactivityTimeout(RuntimeError):
    """One API-server turn the inactivity watchdog abandoned.

    Raised by ``_run_agent`` when ``gateway.run``'s existing
    ``agent.gateway_timeout`` watchdog fired and the executor worker still has
    not returned.  Without it the ``await`` on that worker is unbounded: a
    wedged ``run_conversation`` parks the background owner-response task
    forever, so the response stays ``in_progress``, its job lease keeps being
    heartbeated, and only a restart sweep ever makes it terminal.  Surfacing
    the abandonment as an ordinary exception is what lets the background task
    reach a terminal state on its own.
    """


class APIServerAdapter(BasePlatformAdapter):
    """
    OpenAI-compatible HTTP API server adapter.

    Runs an aiohttp web server that accepts OpenAI-format requests
    and routes them through hermes-agent's AIAgent.
    """

    # Stateless request/response: every route (the OpenAI-spec
    # /v1/chat/completions and /v1/responses, and the proprietary /v1/runs SSE
    # stream) tears down its channel when the turn ends. There is no persistent
    # outbound channel to push a background completion to a client that already
    # received its response, and ``send()`` is a no-op stub. So async-delivery
    # tools (terminal notify_on_complete / watch_patterns, delegate_task
    # background=True) must NOT promise delivery on this path — see
    # ``async_delivery_supported()``.
    supports_async_delivery: bool = False

    # Same statelessness applies to the startup auto-resume prompt: no client
    # is waiting to answer "session restored — what next?", so a resumed turn
    # should complete the interrupted work rather than acknowledge (#57056).
    interactive_resume: bool = False

    def __init__(self, config: PlatformConfig):
        super().__init__(config, Platform.API_SERVER)
        extra = config.extra or {}
        self._host: str = extra.get("host", os.getenv("API_SERVER_HOST", DEFAULT_HOST))
        raw_port = extra.get("port")
        if raw_port is None:
            raw_port = os.getenv("API_SERVER_PORT", str(DEFAULT_PORT))
        self._port: int = _coerce_port(raw_port, DEFAULT_PORT)
        self._api_key: str = extra.get("key", _get_scoped_secret("API_SERVER_KEY", ""))
        self._cors_origins: tuple[str, ...] = self._parse_cors_origins(
            extra.get("cors_origins", os.getenv("API_SERVER_CORS_ORIGINS", "")),
        )
        self._model_name: str = self._resolve_model_name(
            extra.get("model_name", os.getenv("API_SERVER_MODEL_NAME", "")),
        )
        # model_routes: maps incoming ``model`` field values to specific
        # provider/model configs so one API server instance can serve
        # multiple clients on different backends.
        #
        # Config format (platforms.api_server.extra in the gateway config):
        #   model_routes:
        #     minimax-m2:          # alias the client sends as the "model" field
        #       model: "minimax/minimax-m1"
        #       provider: "openrouter"   # optional — resolved via the provider
        #                                # credential chain when set
        #       api_key: "sk-…"          # optional — per-route UPSTREAM provider
        #                                # key override (NOT caller auth; never logged)
        #       base_url: "https://…"    # optional — per-route base URL override
        self._model_routes: Dict[str, Dict[str, Any]] = self._parse_model_routes(
            extra.get("model_routes"),
        )
        # direct_model_requests: opt-in passthrough for a bare ``model`` value
        # (no ``provider``) on the OpenAI-compatible surfaces
        # (/v1/chat/completions, /v1/responses).  Off by default: generic
        # OpenAI clients routinely hardcode model names ("gpt-4o", ...), and
        # existing deployments rely on those falling back to the gateway
        # default rather than switching the executing model.  Requests that
        # send an explicit ``provider`` — and the Hermes-native session-chat
        # and /v1/runs endpoints — are always honored regardless of this flag.
        # (Idea credit: PR #22825 by @mssteuer.)
        self._direct_model_requests: bool = _coerce_request_bool(
            extra.get("direct_model_requests"), default=False
        )
        self._app: Optional["web.Application"] = None
        self._runner: Optional["web.AppRunner"] = None
        self._site: Optional["web.TCPSite"] = None
        self._response_store = ResponseStore()
        # Active run streams: run_id -> asyncio.Queue of SSE event dicts
        self._run_streams: Dict[str, "asyncio.Queue[Optional[Dict]]"] = {}
        # Creation timestamps for orphaned-run TTL sweep
        self._run_streams_created: Dict[str, float] = {}
        # Runs with a connected SSE consumer; their queue is actively draining.
        self._run_stream_subscribers: set[str] = set()
        # Active run agent/task references for stop support
        self._active_run_agents: Dict[str, Any] = {}
        self._active_run_tasks: Dict[str, "asyncio.Task"] = {}
        # Stop is cooperative: the executor thread may outlive the HTTP request.
        self._stopping_run_ids: set[str] = set()
        # Background owner responses this process is still driving. Their durable
        # recovery job is fenced by a renewable lease, so a turn that legitimately
        # takes longer than the lease has to keep saying it is alive — otherwise a
        # sibling gateway would reclaim live work.
        self._owner_response_jobs: set[str] = set()
        # Pollable run status for dashboards and external control-plane UIs.
        self._run_statuses: Dict[str, Dict[str, Any]] = {}
        # Active approval session key for each run_id.  The approval core
        # resolves requests by session key, while API clients address the
        # in-flight run by run_id.
        self._run_approval_sessions: Dict[str, str] = {}
        self._session_db: Optional[Any] = None  # Lazy-init SessionDB for session continuity
        self._session_dbs: Dict[str, Any] = {}
        self._session_db_cache_lock = threading.Lock()
        self._session_db_cache_closed = False
        # Last-known-good resolved model per session (keyed by gateway_session_key
        # ONLY — never session_id, which rotates/is ephemeral for one-off API
        # server requests; "*" is the process-wide fallback), mirroring
        # GatewayRunner._last_resolved_model in run.py — recovers from a
        # transient empty model resolution (#35314) instead of building an
        # agent with model="" that 400s every call until manual retry.
        self._last_resolved_model: Dict[str, str] = {}
        self._session_db_lock: Optional[asyncio.Lock] = None  # Single-flight for lazy init
        # Concurrency cap shared across all agent-serving endpoints
        # (/v1/chat/completions, /v1/responses, /v1/runs). Read from
        # config.yaml gateway.api_server.max_concurrent_runs; 0 disables
        # the cap. Bounds CPU / memory / upstream-LLM-quota exhaustion
        # from a request flood (#7483).
        self._max_concurrent_runs: int = self._resolve_max_concurrent_runs()
        # Number of in-flight runs on the non-streaming chat/responses paths
        # (the /v1/runs path tracks its own in-flight set via
        # _active_run_tasks).
        self._inflight_agent_runs: int = 0
        # Every agent currently inside _run_agent(), i.e. exactly the turns
        # counted by _inflight_agent_runs above.  Shutdown needs the whole
        # adapter-owned set, so this is deliberately NOT _active_run_agents:
        # that one is run_id-keyed and scoped to the public /v1/runs stop API,
        # and only /v1/runs has a run_id at all.  Keyed by id() because the
        # other six agent-entry paths have no stable identifier of their own;
        # the dict holds a strong reference for the life of the turn, so an
        # id() can never be recycled while it is still registered.
        self._shutdown_interruptible_agents: Dict[int, Any] = {}
        # Back-reference to the owning GatewayRunner (set by gateway/run.py)
        # so /api/platforms/{platform}/events can resolve sibling adapters.
        # BasePlatformAdapter declares the class-level default of None.
        self.gateway_runner: Optional[Any] = None
        # Requests admitted before their handler reaches agent bookkeeping.
        # Shutdown counts this reservation so the request cannot slip through
        # the drain between its first await and _run_agent()/task registration.
        self._pending_agent_requests: int = 0

    def active_agent_work_count(self) -> int:
        """Return all live agent work owned by this API adapter.

        ``/v1/runs`` registers an asyncio task before it constructs and stores
        its agent, so ``_active_run_agents`` has a real queued-before-agent gap.
        Reuse the task-based accounting used by the concurrent-run limit: it
        covers that gap and excludes completed tasks retained until cleanup.
        """
        try:
            return (
                int(getattr(self, "_pending_agent_requests", 0))
                + int(self._inflight_agent_runs)
                + sum(not task.done() for task in self._active_run_tasks.values())
            )
        except Exception:
            return 0

    def interrupt_active_runs(self, reason: str) -> int:
        """Cooperatively interrupt every adapter-owned agent during shutdown.

        The gateway drain accounts for API-server work through
        ``active_agent_work_count()``, but those agents are owned by this
        adapter rather than ``GatewayRunner._running_agents``, so
        ``GatewayRunner._interrupt_running_agents()`` never reaches them: the
        turn runs to the drain timeout with no cooperative interrupt and is
        then amputated by the post-interrupt tool-subprocess kill.

        Cover the same set the drain waits on, so accounting and interrupt
        agree:

        * ``_active_run_agents`` — the ``/v1/runs`` agents counted through
          ``_active_run_tasks``.
        * ``_shutdown_interruptible_agents`` — every ``_run_agent()`` turn
          counted through ``_inflight_agent_runs``, i.e. both session-chat
          routes, ``/v1/chat/completions`` and ``/v1/responses`` in their
          streaming and non-streaming forms.

        ``_pending_agent_requests`` is intentionally not covered: it counts
        admitted requests that have not constructed an agent yet, so there is
        no object to interrupt.

        Returns the number of agents that accepted an interrupt.
        """
        agents: Dict[int, Any] = {}
        for agent in list(self._active_run_agents.values()):
            if agent is not None:
                agents[id(agent)] = agent
        for agent in list(self._shutdown_interruptible_agents.values()):
            if agent is not None:
                # Dedupe by object identity — the two registries are disjoint
                # today (/v1/runs runs its own lifecycle, not _run_agent), but
                # an agent published to both must still be interrupted once.
                agents[id(agent)] = agent

        interrupted = 0
        for agent in agents.values():
            try:
                if request_hard_interrupt(agent, reason):
                    interrupted += 1
            except Exception as exc:
                logger.debug("[api_server] failed interrupting active agent: %s", exc)
        return interrupted

    @staticmethod
    def _gateway_is_draining() -> bool:
        """Whether the owning gateway currently refuses new agent turns."""
        try:
            from gateway.run import _gateway_runner_ref

            runner = _gateway_runner_ref()
            return bool(
                runner
                and (
                    getattr(runner, "_draining", False)
                    or getattr(runner, "_external_drain_active", False)
                )
            )
        except Exception:
            return False

    def _draining_response(self) -> Optional["web.Response"]:
        """Return a retryable response while the gateway drains existing work."""
        if not self._gateway_is_draining():
            return None
        return web.json_response(
            _openai_error(
                "Gateway is draining existing work; retry shortly.",
                code="gateway_draining",
            ),
            status=503,
            headers={"Retry-After": "1"},
        )

    def _activate_admitted_request(self) -> None:
        """Transfer this request's drain reservation to agent bookkeeping."""
        reservation = _api_agent_request_reservation.get()
        if reservation and reservation["active"]:
            reservation["active"] = False
            self._pending_agent_requests = max(0, self._pending_agent_requests - 1)

    def _readiness_work_counts(self) -> tuple[int, int, int]:
        """Return bounded work counts from each subsystem's public state."""
        active_api_runs = sum(
            1
            for status in self._run_statuses.values()
            # "stopping" (set by _handle_stop_run) is not terminal: the run
            # stays in this state, doing real executor-thread work, until the
            # agent actually notices the interrupt and the task settles to
            # "cancelled" — an unbounded window, not the old ~5s hard-timeout
            # wait. Excluding it here undercounts active_api_runs for the
            # whole duration of a cooperative stop.
            if status.get("status") in {"queued", "running", "waiting_for_approval", "stopping"}
        )
        process_depth = 0
        active_delegations = 0
        try:
            from tools.process_registry import process_registry

            process_depth = process_registry.completion_queue.qsize()
        except Exception:
            pass
        try:
            from tools.async_delegation import active_count

            active_delegations = active_count()
        except Exception:
            pass
        return active_api_runs, process_depth, active_delegations

    @staticmethod
    def _parse_cors_origins(value: Any) -> tuple[str, ...]:
        """Normalize configured CORS origins into a stable tuple."""
        if not value:
            return ()

        if isinstance(value, str):
            items = value.split(",")
        elif isinstance(value, (list, tuple, set)):
            items = value
        else:
            items = [str(value)]

        return tuple(str(item).strip() for item in items if str(item).strip())

    @staticmethod
    def _resolve_max_concurrent_runs() -> int:
        """Read the concurrent-run cap from config.yaml (0 disables).

        gateway.api_server.max_concurrent_runs. Falls back to the historical
        default of 10 when unset or malformed. Negative values are clamped
        to 0 (disabled).
        """
        default = 10
        try:
            from hermes_cli.config import cfg_get, load_config

            raw = cfg_get(
                load_config(),
                "gateway",
                "api_server",
                "max_concurrent_runs",
                default=default,
            )
            value = int(raw)
        except Exception:
            return default
        return max(0, value)

    @staticmethod
    def _resolve_model_name(explicit: str) -> str:
        """Derive the advertised model name for /v1/models.

        Priority:
        1. Explicit override (config extra or API_SERVER_MODEL_NAME env var)
        2. Active profile name (so each profile advertises a distinct model)
        3. Fallback: "hermes-agent"

        Delegates the tiered fallthrough to
        :func:`hermes_cli.model_switch.resolve_effective_model` (the shared
        override > mid-tier > default precedence owner).
        """
        from hermes_cli.model_switch import resolve_effective_model

        profile_name = ""
        try:
            from hermes_cli.profiles import get_active_profile_name
            profile = get_active_profile_name()
            if profile and profile not in {"default", "custom"}:
                profile_name = profile
        except Exception:
            pass
        return resolve_effective_model(explicit, profile_name, "hermes-agent")

    def _cors_headers_for_origin(self, origin: str) -> Optional[Dict[str, str]]:
        """Return CORS headers for an allowed browser origin."""
        if not origin or not self._cors_origins:
            return None

        if "*" in self._cors_origins:
            headers = dict(_CORS_HEADERS)
            headers["Access-Control-Allow-Origin"] = "*"
            headers["Access-Control-Max-Age"] = "600"
            return headers

        if origin not in self._cors_origins:
            return None

        headers = dict(_CORS_HEADERS)
        headers["Access-Control-Allow-Origin"] = origin
        headers["Vary"] = "Origin"
        headers["Access-Control-Max-Age"] = "600"
        return headers

    def _origin_allowed(self, origin: str) -> bool:
        """Allow non-browser clients and explicitly configured browser origins."""
        if not origin:
            return True

        if not self._cors_origins:
            return False

        return "*" in self._cors_origins or origin in self._cors_origins

    @staticmethod
    def _clean_log_value(value: Any, *, max_len: int = 200) -> str:
        """Sanitize request metadata before it reaches security logs."""
        if value is None:
            return ""
        text = str(value).replace("\r", " ").replace("\n", " ").strip()
        return text[:max_len]

    def _request_audit_context(self, request: "web.Request") -> Dict[str, str]:
        """Return non-secret source metadata for security/audit warnings."""
        peer_ip = ""
        try:
            peer = request.transport.get_extra_info("peername") if request.transport else None
            if isinstance(peer, (tuple, list)) and peer:
                peer_ip = str(peer[0])
        except Exception:
            peer_ip = ""

        return {
            "remote": self._clean_log_value(getattr(request, "remote", "") or peer_ip),
            "peer_ip": self._clean_log_value(peer_ip),
            "forwarded_for": self._clean_log_value(request.headers.get("X-Forwarded-For", "")),
            "real_ip": self._clean_log_value(request.headers.get("X-Real-IP", "")),
            "method": self._clean_log_value(request.method, max_len=16),
            "path": self._clean_log_value(request.path_qs, max_len=500),
            "user_agent": self._clean_log_value(request.headers.get("User-Agent", ""), max_len=300),
        }

    def _request_audit_log_suffix(self, request: "web.Request") -> str:
        ctx = self._request_audit_context(request)
        fields = [f"{key}={value!r}" for key, value in ctx.items() if value]
        return " ".join(fields) if fields else "source='unknown'"

    def _cron_origin_from_request(self, request: "web.Request") -> Dict[str, str]:
        """Persist safe API source metadata on cron jobs created over HTTP."""
        ctx = self._request_audit_context(request)
        origin = {
            "platform": "api_server",
            "chat_id": "api",
        }
        if ctx.get("remote"):
            origin["source_ip"] = ctx["remote"]
        if ctx.get("peer_ip"):
            origin["peer_ip"] = ctx["peer_ip"]
        if ctx.get("forwarded_for"):
            origin["forwarded_for"] = ctx["forwarded_for"]
        if ctx.get("real_ip"):
            origin["real_ip"] = ctx["real_ip"]
        if ctx.get("user_agent"):
            origin["user_agent"] = ctx["user_agent"]
        return origin

    # ------------------------------------------------------------------
    # Auth helper
    # ------------------------------------------------------------------

    def _expected_api_key(self) -> str:
        """Return the API key authorized for the URL-selected profile."""
        profile = _api_request_profile.get()
        if not profile or profile == "default":
            return self._api_key

        try:
            from agent.secret_scope import get_secret
            from hermes_cli.auth import has_usable_secret

            key = get_secret("API_SERVER_KEY", "") or ""
            if not has_usable_secret(key, min_length=16):
                return ""
            return key
        except Exception as exc:
            # Fail closed if the profile scope or strength guard cannot resolve
            # the credential. Do not log the key or exception text.
            logger.warning(
                "Failed to resolve a usable profile-scoped API_SERVER_KEY for %r: %s",
                profile,
                type(exc).__name__,
            )
            return ""

    def _check_auth(self, request: "web.Request") -> Optional["web.Response"]:
        """
        Validate Bearer token from Authorization header.

        Returns None if auth is OK, or a 401 web.Response on failure.
        connect() refuses to start the API server without API_SERVER_KEY, so
        the no-key branch only exists for tests or unsupported manual wiring.
        """
        profile = _api_request_profile.get()
        is_named_profile = bool(profile and profile != "default")
        expected_key = self._expected_api_key()
        if not expected_key:
            # Preserve the historical no-key test/manual-wiring behavior only
            # for the default listener. Named profiles must fail closed rather
            # than inherit the listener owner's key.
            if not is_named_profile:
                return None
            logger.warning(
                "API server rejected request for profile %r: no profile-scoped "
                "API_SERVER_KEY is configured; %s",
                profile,
                self._request_audit_log_suffix(request),
            )
            return web.json_response(
                {
                    "error": {
                        "message": "Invalid gateway API key (API_SERVER_KEY)",
                        "type": "gateway_auth_error",
                        "code": "gateway_auth_failed",
                    }
                },
                status=401,
            )

        auth_header = request.headers.get("Authorization", "")
        if auth_header.startswith("Bearer "):
            token = auth_header[7:].strip()
            # Compare as bytes: ``hmac.compare_digest`` raises TypeError on a
            # str containing non-ASCII characters, and ``token`` is the raw
            # client-supplied header. A stray non-ASCII byte in the key would
            # otherwise crash this handler (500) instead of returning a clean
            # 401. Encoding both sides keeps the timing-safe comparison and
            # matches web_server.py's dashboard-token check.
            if hmac.compare_digest(token.encode(), expected_key.encode()):
                return None  # Auth OK

        logger.warning(
            "API server rejected invalid API key: %s",
            self._request_audit_log_suffix(request),
        )
        return web.json_response(
            {"error": {"message": "Invalid gateway API key (API_SERVER_KEY)", "type": "gateway_auth_error", "code": "gateway_auth_failed"}},
            status=401,
        )

    @staticmethod
    def _normalize_callback_platform(value: str) -> str:
        normalized = (value or "").strip().lower().replace("-", "_")
        if not re.fullmatch(r"[a-z0-9_]+", normalized):
            return ""
        return normalized

    def _get_platform_callback_adapter(
        self,
        request: "web.Request",
        platform_name: str,
    ) -> Optional[Any]:
        injected = request.app.get("platform_event_adapters")
        if isinstance(injected, dict):
            adapter = injected.get(platform_name)
            if adapter is not None:
                return adapter

        adapter = request.app.get(f"{platform_name}_adapter")
        if adapter is not None:
            return adapter

        runner = self.gateway_runner or request.app.get("gateway_runner")
        adapters = getattr(runner, "adapters", None)
        if not adapters:
            return None

        try:
            from gateway.config import Platform as _Platform
            return adapters.get(_Platform(platform_name))
        except Exception:
            for platform, candidate in adapters.items():
                if getattr(platform, "value", platform) == platform_name:
                    return candidate
        return None

    async def _handle_platform_event_callback(self, request: "web.Request") -> "web.Response":
        platform_name = self._normalize_callback_platform(
            request.match_info.get("platform", "")
        )
        if not platform_name:
            return web.json_response(
                _openai_error(
                    "Invalid platform name",
                    code="invalid_platform",
                ),
                status=400,
            )

        adapter = self._get_platform_callback_adapter(request, platform_name)
        if adapter is None:
            return web.json_response(
                _openai_error(
                    "Platform adapter is not connected",
                    code="platform_unavailable",
                ),
                status=503,
            )

        verifier = getattr(adapter, "verify_http_event_request", None)
        dispatcher = getattr(adapter, "dispatch_http_event", None)
        if verifier is None or dispatcher is None:
            return web.json_response(
                _openai_error(
                    "Platform adapter does not support HTTP events",
                    code="platform_http_events_unsupported",
                ),
                status=503,
            )

        auth_header = request.headers.get("Authorization", "")
        try:
            if asyncio.iscoroutinefunction(verifier):
                ok, code = await verifier(auth_header)
            else:
                # Platform verifiers may do blocking network I/O (e.g. Google
                # signing-cert fetches) — keep that off the event loop.
                ok, code = await asyncio.to_thread(verifier, auth_header)
        except Exception:
            # Fail closed: a crashing verifier must never admit the event.
            logger.exception(
                "Platform HTTP event verifier failed for %s", platform_name
            )
            ok, code = False, "platform_event_verifier_error"
        if not ok:
            return web.json_response(
                _openai_error(
                    "Invalid platform event authorization",
                    code=code or "invalid_platform_event_authorization",
                ),
                status=401,
            )

        try:
            payload = await request.json()
        except Exception:
            return web.json_response(
                _openai_error("Invalid JSON in platform event", code="invalid_json"),
                status=400,
            )

        if not isinstance(payload, dict):
            return web.json_response(
                _openai_error(
                    "Platform event must be a JSON object",
                    code="invalid_request",
                ),
                status=400,
            )

        try:
            result = await dispatcher(payload)
        except Exception:
            logger.exception("Platform HTTP event dispatch failed for %s", platform_name)
            return web.json_response(
                _openai_error(
                    "Platform event dispatch failed",
                    err_type="server_error",
                    code="platform_event_dispatch_failed",
                ),
                status=500,
            )

        return web.json_response(result if isinstance(result, dict) else {})

    # ------------------------------------------------------------------
    # Multi-profile multiplexing (/p/<profile>/…)
    # ------------------------------------------------------------------

    def _resolve_request_profile(self, request: "web.Request"):
        """Resolve + validate the /p/<profile>/ URL prefix on an API request.

        Returns:
          - ``None`` when no profile prefix is present, or multiplexing is off
            (the prefix is ignored; request handled as the default profile).
          - the profile name (str) when present, multiplexing is on, and the
            profile is one this gateway serves.
          - ``_PROFILE_REJECTED`` when a prefix is present but the profile is
            unknown/unconfigured (handler/middleware returns 404).
        """
        profile = (request.match_info.get("profile") or "").strip()
        if not profile:
            return None
        runner = getattr(self, "gateway_runner", None)
        cfg = getattr(runner, "config", None)
        if not getattr(cfg, "multiplex_profiles", False):
            # Prefix supplied but multiplexing is off — ignore it, behave as
            # the single-profile gateway (don't 404 a would-be valid route).
            return None
        try:
            from hermes_cli.profiles import profiles_to_serve

            served = {
                name
                for name, _ in profiles_to_serve(
                    multiplex=True,
                    profile_allowlist=getattr(
                        cfg, "multiplex_profile_allowlist", None
                    ),
                )
            }
        except Exception:
            return _PROFILE_REJECTED
        if profile not in served:
            return _PROFILE_REJECTED
        return profile

    @staticmethod
    def _profile_scope(profile: Optional[str]):
        """Enter the multiplex profile runtime scope, or a no-op when unset.

        When no ``/p/<profile>/`` prefix was given AND multiplexing is active,
        enter the DEFAULT profile's scope instead of a no-op: api_server is a
        port-binding platform that lives on the default profile, and with
        multiplex fail-closed ``get_secret`` active, an unscoped agent run
        raises ``UnscopedSecretError`` on its first credential read (#61276).
        Single-profile gateways keep the no-op — ``get_secret`` falls through
        to ``os.environ`` there, unchanged.
        """
        if not profile:
            try:
                from agent.secret_scope import is_multiplex_active

                if is_multiplex_active():
                    from gateway.run import _profile_runtime_scope
                    from hermes_constants import get_hermes_home

                    return _profile_runtime_scope(get_hermes_home())
            except Exception:
                pass
            return nullcontext()
        from gateway.run import _profile_runtime_scope
        from hermes_cli.profiles import get_profile_dir

        return _profile_runtime_scope(get_profile_dir(profile))

    def _make_profile_prefix_middleware(self):
        """Reject unknown /p/<profile>/ prefixes and scope the request home."""

        @web.middleware
        async def profile_prefix_middleware(request: "web.Request", handler):
            profile = self._resolve_request_profile(request)
            if profile is _PROFILE_REJECTED:
                return web.json_response(
                    {"error": "Unknown or unconfigured profile"},
                    status=404,
                )
            token = _api_request_profile.set(profile)
            try:
                with self._profile_scope(profile):
                    return await handler(request)
            finally:
                _api_request_profile.reset(token)

        return profile_prefix_middleware

    def _make_route_allowlist_middleware(self):
        """Enforce ``gateway.api_server.allowed_routes`` for least privilege.

        Runs AFTER the profile-prefix middleware (later in ``mws``) so
        ``get_hermes_home()`` — and therefore config resolution — is already
        scoped to the request's profile for a multiplexed ``/p/<profile>/``
        call. Exactly one multiplex prefix is stripped before matching, per
        contract: a route is gated the same way whether reached natively or
        via its ``/p/<profile>/`` mirror.
        """

        @web.middleware
        async def route_allowlist_middleware(request: "web.Request", handler):
            path = request.path
            prefix_profile = request.match_info.get("profile")
            if prefix_profile is not None:
                stripped = path[len(f"/p/{prefix_profile}"):]
                path = stripped or "/"

            route = getattr(request.match_info, "route", None)
            resource = getattr(route, "resource", None)
            route_template = getattr(resource, "canonical", None)
            if not isinstance(route_template, str) or not route_template.startswith("/"):
                route_template = path
            elif prefix_profile is not None:
                template_prefix = "/p/{profile}"
                if route_template.startswith(template_prefix):
                    route_template = route_template[len(template_prefix):] or "/"

            if path in _ALWAYS_ALLOWED_PATHS:
                return await handler(request)

            mode, patterns = _resolve_api_server_allowed_routes()
            if mode == _ROUTES_UNRESTRICTED:
                return await handler(request)
            if mode == _ROUTES_ALLOWLIST and _route_matches_any(
                path,
                patterns,
                method=request.method,
                route_template=route_template,
            ):
                return await handler(request)

            return web.json_response(
                _openai_error(
                    "Route not permitted for this profile",
                    code="route_not_allowed",
                ),
                status=403,
            )

        return route_allowlist_middleware

    def _make_owner_authority_middleware(self):
        """Turn a missing durable owner store into one stable, safe refusal.

        :class:`OwnerAuthorityUnavailable` is raised by the response store
        before any owner state is read or written, so reaching here means the
        request performed no owner mutation and started no run. The reply is
        deliberately uninformative about the storage failure — the operator
        gets the details in ``errors.log``, the caller gets a retryable 503.
        """

        @web.middleware
        async def owner_authority_middleware(request: "web.Request", handler):
            try:
                return await handler(request)
            except OwnerAuthorityUnavailable:
                logger.error(
                    "Owner workspace storage is unavailable; refused %s %s",
                    request.method, request.path,
                )
                return web.json_response(
                    _openai_error(
                        "The workspace is unavailable right now. Nothing was "
                        "changed. Please try again shortly.",
                        err_type="server_error",
                        code="owner_workspace_unavailable",
                    ),
                    status=503,
                )

        return owner_authority_middleware

    def _http_route_table(self) -> List[tuple]:
        """Return (method, path, handler) rows registered by ``connect()``.

        Kept as a method so multiplex tests can assert the /p/<profile>/
        mirrors without starting a real aiohttp listener.
        """
        routes: List[tuple] = [
            ("GET", "/health", self._handle_health),
            ("GET", "/health/detailed", self._handle_health_detailed),
            ("GET", "/v1/health", self._handle_health),
            ("GET", "/v1/models", self._handle_models),
            ("GET", "/api/model/options", self._handle_model_options),
            ("GET", "/v1/capabilities", self._handle_capabilities),
            ("GET", "/v1/skills", self._handle_skills),
            ("GET", "/v1/toolsets", self._handle_toolsets),
            ("GET", "/api/sessions", self._handle_list_sessions),
            ("POST", "/api/sessions", self._handle_create_session),
            ("GET", "/api/sessions/{session_id}", self._handle_get_session),
            ("PATCH", "/api/sessions/{session_id}", self._handle_patch_session),
            ("DELETE", "/api/sessions/{session_id}", self._handle_delete_session),
            ("GET", "/api/sessions/{session_id}/messages", self._handle_session_messages),
            ("POST", "/api/sessions/{session_id}/fork", self._handle_fork_session),
            ("POST", "/api/sessions/{session_id}/chat", self._handle_session_chat),
            ("POST", "/api/sessions/{session_id}/chat/stream", self._handle_session_chat_stream),
            ("POST", "/api/sessions/{session_id}/model", self._handle_session_model_lock),
            ("POST", "/v1/chat/completions", self._handle_chat_completions),
            ("POST", "/v1/responses", self._handle_responses),
            (
                "GET", "/v1/responses/conversations/{conversation}",
                self._handle_owner_conversation_history,
            ),
            (
                "POST", "/v1/responses/conversations/{conversation}/consume",
                self._handle_consume_owner_proposal,
            ),
            (
                "POST", "/v1/responses/conversations/{conversation}/authority",
                self._handle_owner_conversation_authority,
            ),
            (
                "POST", "/v1/responses/conversations/{conversation}/recovery",
                self._handle_acknowledge_owner_recovery,
            ),
            ("GET", "/v1/responses/{response_id}", self._handle_get_response),
            ("DELETE", "/v1/responses/{response_id}", self._handle_delete_response),
            # Generic platform HTTP event callback ingress. Authenticated by
            # the target adapter's own verifier (platform-signed bearer), NOT
            # API_SERVER_KEY — external platforms hold no API server key.
            ("POST", "/api/platforms/{platform}/events", self._handle_platform_event_callback),
            ("GET", "/api/jobs", self._handle_list_jobs),
            ("POST", "/api/jobs", self._handle_create_job),
            ("GET", "/api/jobs/{job_id}", self._handle_get_job),
            ("PATCH", "/api/jobs/{job_id}", self._handle_update_job),
            ("DELETE", "/api/jobs/{job_id}", self._handle_delete_job),
            ("POST", "/api/jobs/{job_id}/pause", self._handle_pause_job),
            ("POST", "/api/jobs/{job_id}/resume", self._handle_resume_job),
            ("POST", "/api/jobs/{job_id}/run", self._handle_run_job),
            ("GET", "/v1/owner-workspace/projects", self._handle_owner_workspace_projects),
            (
                "GET",
                "/v1/owner-workspace/projects/{project_slug}/snapshot",
                self._handle_owner_workspace_project_snapshot,
            ),
            (
                "GET",
                "/v1/owner-workspace/projects/{project_slug}/attachments/{attachment_id}",
                self._handle_owner_workspace_project_attachment,
            ),
            ("GET", "/v1/owner-workspace/decisions", self._handle_owner_workspace_decisions),
            ("POST", "/v1/runs", self._handle_runs),
            ("GET", "/v1/runs/{run_id}", self._handle_get_run),
            ("GET", "/v1/runs/{run_id}/events", self._handle_run_events),
            ("POST", "/v1/runs/{run_id}/approval", self._handle_run_approval),
            ("POST", "/v1/runs/{run_id}/steer", self._handle_steer_run),
            ("POST", "/v1/runs/{run_id}/stop", self._handle_stop_run),
        ]
        if _CRON_AVAILABLE:
            # Chronos managed-cron fire webhook (NAS → agent). Authenticated
            # by a NAS-minted JWT (NOT API_SERVER_KEY).
            routes.append(("POST", "/api/cron/fire", self._handle_cron_fire))
        return routes

    # ------------------------------------------------------------------
    # Session header helpers
    # ------------------------------------------------------------------

    # Soft length cap for session identifiers.  Headers are bounded in
    # aggregate by aiohttp (``client_max_size`` / default 8 KiB per
    # header), but we impose a tighter limit on the session headers so a
    # caller can't burn memory by passing a multi-kilobyte "session key".
    # 256 chars is well above any realistic stable channel identifier
    # (e.g. ``agent:main:webui:dm:user-42``) while staying small enough
    # that the sanitized form is safe to pass into Honcho / state.db.
    _MAX_SESSION_HEADER_LEN = 256

    def _parse_session_key_header(
        self, request: "web.Request"
    ) -> tuple[Optional[str], Optional["web.Response"]]:
        """Extract and validate the ``X-Hermes-Session-Key`` header.

        The session key is a stable per-channel identifier that scopes
        long-term memory (e.g. Honcho sessions) across transcripts.  It
        is independent of ``X-Hermes-Session-Id``: callers may send
        either, both, or neither.

        Returns ``(session_key, None)`` on success (with an empty/absent
        header yielding ``None`` for the key), or ``(None, error_response)``
        on validation failure.

        Security: like session continuation, accepting a caller-supplied
        memory scope requires API-key authentication so that an
        unauthenticated client on a local-only server can't inject itself
        into another user's long-term memory scope by guessing a key.
        """
        raw = request.headers.get("X-Hermes-Session-Key", "").strip()
        if not raw:
            return None, None

        if not self._api_key:
            logger.warning(
                "X-Hermes-Session-Key rejected: no API key configured. "
                "Set API_SERVER_KEY to enable long-term memory scoping."
            )
            return None, web.json_response(
                _openai_error(
                    "X-Hermes-Session-Key requires API key authentication. "
                    "Configure API_SERVER_KEY to enable this feature."
                ),
                status=403,
            )

        # Reject control characters that could enable header injection on
        # the echo path.
        if re.search(r'[\r\n\x00]', raw):
            return None, web.json_response(
                {"error": {"message": "Invalid session key", "type": "invalid_request_error"}},
                status=400,
            )

        if len(raw) > self._MAX_SESSION_HEADER_LEN:
            return None, web.json_response(
                {"error": {"message": "Session key too long", "type": "invalid_request_error"}},
                status=400,
            )

        return raw, None

    # ------------------------------------------------------------------
    # Session DB helper
    # ------------------------------------------------------------------

    def _open_and_cache_session_db(self, home) -> Optional[Any]:
        """Sync core: return the cached SessionDB for ``home``, opening it once.

        Shared by the sync (``_ensure_session_db``) and async
        (``_ensure_session_db_async``) entry points so both honor the same
        per-profile cache. Deliberately does NOT write into ``self._session_db``
        — that stays reserved for an explicit test/manual override, so the first
        profile served can't pin every later request to its DB.
        """
        from hermes_state import SessionDB

        key = str(home)
        with self._session_db_cache_lock:
            if self._session_db_cache_closed:
                return None
            db = self._session_dbs.get(key)
            if db is None:
                db = SessionDB(db_path=home / "state.db")
                self._session_dbs[key] = db
            return db

    def _close_cached_session_dbs(self) -> None:
        """Close SessionDB handles owned by this adapter's profile cache."""
        with self._session_db_cache_lock:
            self._session_db_cache_closed = True
            cached = list(self._session_dbs.values())
            self._session_dbs.clear()
        shared_db = getattr(self, "_session_db", None)
        for db in cached:
            if db is shared_db:
                continue
            try:
                db.close()
            except Exception:
                logger.debug("Failed to close API-server SessionDB", exc_info=True)

    def _ensure_session_db(self):
        """Lazily initialise and return the SessionDB for the active profile home.

        Sessions are persisted to ``state.db`` so that ``hermes sessions list``
        shows API-server conversations alongside CLI and gateway ones.

        Under multiplex ``/p/<profile>/`` requests the profile runtime scope
        redirects ``get_hermes_home()``, so each profile gets its own DB —
        never the default profile's file. Synchronous: used by ``_create_agent``
        (itself sync, and run in both loop and worker contexts). Request
        handlers use ``_ensure_session_db_async`` to keep the SQLite open off
        the event loop.
        """
        # Explicit override (tests / manual wiring) wins.
        if self._session_db is not None:
            return self._session_db
        try:
            from hermes_constants import get_hermes_home

            return self._open_and_cache_session_db(get_hermes_home())
        except Exception as e:
            logger.debug("SessionDB unavailable for API server: %s", e)
            return None

    async def _ensure_session_db_async(self):
        """Async variant for request handlers: offload the SQLite open/schema
        init off the single aiohttp event-loop thread.

        The active profile home is captured on the loop thread (its runtime
        scope is not visible inside ``asyncio.to_thread``); only the blocking
        construction runs in the worker. A single-flight lock prevents duplicate
        concurrent construction for the same home.
        """
        if self._session_db is not None:
            return self._session_db
        try:
            from hermes_constants import get_hermes_home

            home = get_hermes_home()
            key = str(home)
            with self._session_db_cache_lock:
                cached = self._session_dbs.get(key)
            if cached is not None:
                return cached
            if self._session_db_lock is None:
                self._session_db_lock = asyncio.Lock()
            async with self._session_db_lock:
                with self._session_db_cache_lock:
                    cached = self._session_dbs.get(key)
                if cached is not None:
                    return cached
                return await asyncio.to_thread(self._open_and_cache_session_db, home)
        except Exception as e:
            logger.debug("SessionDB unavailable for API server: %s", e)
            return None

    # ------------------------------------------------------------------
    # Agent creation helper
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_model_routes(raw: Any) -> Dict[str, Dict[str, Any]]:
        """Validate and normalize the ``model_routes`` config block.

        Accepts a mapping of ``alias -> {model, provider?, api_key?, base_url?}``.
        Invalid shapes are dropped (never raised) so a config typo can't take
        the whole API server down.  Route values are coerced to strings.

        Security: per-route ``api_key`` values are UPSTREAM provider
        credentials (used to call the routed model's backend), not caller
        authentication — callers still authenticate with the global
        API_SERVER_KEY bearer token via ``_check_auth``.  Route api_keys must
        never be logged; only alias names and non-secret fields may appear in
        logs.
        """
        if not isinstance(raw, dict):
            if raw:
                logger.warning(
                    "api_server model_routes ignored: expected a mapping, got %s",
                    type(raw).__name__,
                )
            return {}

        allowed_keys = ("model", "provider", "api_key", "base_url")
        routes: Dict[str, Dict[str, Any]] = {}
        for alias, cfg in raw.items():
            alias_str = str(alias).strip()
            if not alias_str or not isinstance(cfg, dict):
                logger.warning(
                    "api_server model_routes: dropping invalid route entry %r", alias_str or alias
                )
                continue
            route = {
                key: str(cfg[key]).strip()
                for key in allowed_keys
                if cfg.get(key) is not None and str(cfg[key]).strip()
            }
            if not route.get("model"):
                logger.warning(
                    "api_server model_routes: route %r has no 'model'; dropping", alias_str
                )
                continue
            routes[alias_str] = route
        return routes

    def _resolve_route(self, model_alias: Any) -> Optional[Dict[str, Any]]:
        """Return the model_routes entry for *model_alias*, or None."""
        if not self._model_routes or not isinstance(model_alias, str):
            return None
        return self._model_routes.get(model_alias)

    @staticmethod
    def _clean_runtime_id(value: Any, *, max_len: int = 200) -> str:
        if value is None:
            return ""
        text = str(value).strip()
        if not text or len(text) > max_len:
            return ""
        if re.search(r"[\r\n\x00]", text):
            return ""
        return text

    @classmethod
    def _split_provider_prefixed_model(cls, model: str) -> tuple[str, str]:
        text = cls._clean_runtime_id(model)
        if "::" in text:
            provider, raw = text.split("::", 1)
            if re.match(r"^[a-zA-Z0-9_.-]{2,64}$", provider) and raw.strip():
                return provider, raw.strip()
        return "", text

    @classmethod
    def _runtime_options_from_model_options(cls, model_options: Any) -> Dict[str, Any]:
        if not isinstance(model_options, dict):
            return {}
        runtime_options: Dict[str, Any] = {}
        reasoning = model_options.get("reasoning")
        if isinstance(reasoning, dict):
            enabled = reasoning.get("enabled")
            effort = cls._clean_runtime_id(reasoning.get("effort"), max_len=32)
            if enabled is False:
                runtime_options["reasoning_config"] = {"enabled": False}
            elif effort:
                runtime_options["reasoning_config"] = {"enabled": True, "effort": effort}
            elif enabled is True:
                runtime_options["reasoning_config"] = {"enabled": True}
        service_tier = cls._clean_runtime_id(model_options.get("service_tier"), max_len=32)
        if service_tier:
            runtime_options["service_tier"] = service_tier
        elif _coerce_request_bool(model_options.get("fast"), default=False):
            runtime_options["service_tier"] = "priority"
        return runtime_options

    def _session_runtime_request_from_body(self, body: Dict[str, Any]) -> Dict[str, Any]:
        raw_model = self._clean_runtime_id(body.get("model") or body.get("model_id"))
        raw_provider = self._clean_runtime_id(body.get("provider") or body.get("provider_id"), max_len=80)
        prefixed_provider, split_model = self._split_provider_prefixed_model(raw_model)
        provider = raw_provider or prefixed_provider
        model = split_model or raw_model
        alias_route = self._resolve_route(raw_model) or self._resolve_route(model)
        route = dict(alias_route) if isinstance(alias_route, dict) else None
        route_source = "model_routes" if route else "global"
        if not route and model and model != self._model_name:
            route = {"model": model}
            if provider:
                route["provider"] = provider
            route_source = "raw_request"
        elif not route and provider and model:
            route = {"model": model, "provider": provider}
            route_source = "raw_request"
        runtime_options = self._runtime_options_from_model_options(body.get("model_options"))
        requested = {"provider": provider, "model": model, "raw_model": raw_model}
        return {
            "requested": requested,
            "route": route,
            "route_source": route_source,
            "runtime_options": runtime_options,
            "require_model_lock": _coerce_request_bool(body.get("require_model_lock"), default=False),
            "model_options": body.get("model_options") if isinstance(body.get("model_options"), dict) else {},
        }

    def _runtime_lock_error(self, runtime_request: Dict[str, Any]) -> Optional["web.Response"]:
        if not runtime_request.get("require_model_lock"):
            return None
        requested = runtime_request.get("requested") or {}
        model = self._clean_runtime_id(requested.get("model"))
        provider = self._clean_runtime_id(requested.get("provider"), max_len=80)
        route = runtime_request.get("route")
        if not model and not provider:
            return web.json_response(
                _openai_error("require_model_lock was set but no model/provider was provided", code="missing_model"),
                status=400,
            )
        if not route or runtime_request.get("route_source") == "global":
            return web.json_response(
                _openai_error("Requested Browser model lock cannot be routed; refusing silent global fallback", code="model_lock_unavailable"),
                status=409,
            )
        return None

    def _persist_session_runtime_lock(self, session_id: str, runtime_request: Dict[str, Any]) -> bool:
        # Persist only a newly confirmed lock. Reusing a stored lock should not
        # rewrite its timestamp/prompt state on every turn, and an ordinary
        # one-off request override must not erase a previously confirmed lock.
        if runtime_request.get("persisted_lock") or not runtime_request.get("require_model_lock"):
            return True
        requested = runtime_request.get("requested") or {}
        model = self._clean_runtime_id(requested.get("model"))
        provider = self._clean_runtime_id(requested.get("provider"), max_len=80)
        if not model and not provider:
            return False
        db = self._ensure_session_db()
        if db is None:
            return False
        try:
            db.update_session_runtime_lock(
                session_id,
                model=model or None,
                provider=provider or None,
                model_options=runtime_request.get("model_options") or {},
                route_source=runtime_request.get("route_source") or "",
                confirmed=bool(runtime_request.get("require_model_lock")),
            )
            return True
        except Exception:
            logger.warning("[%s] failed to persist session runtime lock for %s", self.name, session_id, exc_info=True)
            return False

    @staticmethod
    def _parse_session_model_config(raw: Any) -> Dict[str, Any]:
        if isinstance(raw, dict):
            return dict(raw)
        if isinstance(raw, str) and raw.strip():
            try:
                parsed = json.loads(raw)
            except Exception:
                return {}
            if isinstance(parsed, dict):
                return parsed
        return {}

    def _runtime_request_from_persisted_session_lock(
        self,
        session: Optional[Dict[str, Any]],
        body: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        if not isinstance(session, dict):
            return None
        model_config = self._parse_session_model_config(session.get("model_config"))
        lock = model_config.get("browser_model_lock")
        if not isinstance(lock, dict) or not _coerce_request_bool(lock.get("confirmed"), default=False):
            return None
        model = self._clean_runtime_id(lock.get("model"))
        provider = self._clean_runtime_id(lock.get("provider"), max_len=80)
        if not model and not provider:
            return None
        persisted_route_source = self._clean_runtime_id(
            lock.get("route_source"),
            max_len=64,
        ).lower()
        route: Optional[Dict[str, Any]] = None
        if persisted_route_source == "model_routes":
            route = self._resolve_route(model) if model else None
        else:
            route = {"model": model} if model else {}
            if provider:
                route["provider"] = provider
        model_options = (
            body.get("model_options")
            if isinstance(body.get("model_options"), dict)
            else lock.get("model_options")
        )
        return {
            "requested": {
                "provider": provider,
                "model": model,
                "raw_model": model,
            },
            "route": route or None,
            "route_source": "session_model_lock",
            "runtime_options": self._runtime_options_from_model_options(model_options),
            "require_model_lock": True,
            "model_options": model_options if isinstance(model_options, dict) else {},
            "persisted_lock": True,
        }

    def _effective_session_runtime_request(
        self,
        *,
        session: Optional[Dict[str, Any]],
        body: Dict[str, Any],
    ) -> Dict[str, Any]:
        runtime_request = self._session_runtime_request_from_body(body)
        requested = runtime_request.get("requested") or {}
        if requested.get("model") or requested.get("provider"):
            return runtime_request
        persisted = self._runtime_request_from_persisted_session_lock(session, body)
        return persisted or runtime_request

    @classmethod
    def _sanitize_runtime_metadata(
        cls,
        *,
        runtime: Optional[Dict[str, Any]] = None,
        requested_runtime: Optional[Dict[str, Any]] = None,
        route_source: str = "global",
        model_lock: str = "",
    ) -> Dict[str, Any]:
        payload = dict(runtime or {})
        provider = cls._clean_runtime_id(
            payload.get("provider") or payload.get("provider_id") or payload.get("effective_provider"),
            max_len=80,
        )
        model = cls._clean_runtime_id(payload.get("model") or payload.get("model_id") or payload.get("effective_model"))
        result: Dict[str, Any] = {
            "provider": provider,
            "model": model,
            "route_source": cls._clean_runtime_id(payload.get("route_source") or route_source, max_len=64) or "global",
        }
        if requested_runtime or payload.get("requested"):
            req = requested_runtime or payload.get("requested") or {}
            result["requested"] = {
                "provider": cls._clean_runtime_id(req.get("provider"), max_len=80),
                "model": cls._clean_runtime_id(req.get("model")),
            }
        if model_lock or payload.get("model_lock"):
            result["model_lock"] = cls._clean_runtime_id(model_lock or payload.get("model_lock"), max_len=32)
        return result

    @staticmethod
    def _normalize_session_source(value: Any) -> str:
        text = str(value or "").strip().lower()
        allowed = {"api_server", "hermes_browser", "browser", "cli", "telegram", "discord", "slack", "desktop", "dashboard"}
        if text in allowed:
            return "hermes_browser" if text == "browser" else text
        return "api_server"

    def _session_model_override_for(self, session_key: Optional[str]) -> Optional[Dict[str, Any]]:
        """Return the gateway's session ``/model`` override for *session_key*, if any.

        The gateway tracks per-session ``/model`` switches in
        ``GatewayRunner._session_model_overrides``.  API-server requests that
        share such a session key must keep honouring the explicit session
        override even when the request's ``model`` field matches a configured
        route — a user-issued ``/model`` always wins over static config.
        """
        if not session_key:
            return None
        try:
            from gateway.run import _gateway_runner_ref
            runner = _gateway_runner_ref()
            if runner is None:
                return None
            try:
                rehydrate = getattr(runner, "_rehydrate_session_model_override", None)
                if callable(rehydrate):
                    rehydrate(session_key)
            except Exception:
                logger.debug(
                    "api_server failed to rehydrate session /model override for %s",
                    session_key,
                    exc_info=True,
                )
            override = runner._session_model_overrides.get(session_key)
            return dict(override) if isinstance(override, dict) else None
        except Exception:
            return None

    def _request_route_conflict_error(
        self,
        *,
        session_id: Optional[str],
        gateway_session_key: Optional[str],
        requested_model: Optional[str],
        requested_provider: Optional[str],
        route: Optional[Dict[str, Any]],
    ) -> Optional[str]:
        """Return a 400-worthy conflict string for ambiguous route/provider mixes."""
        request_provider = _clean_request_string(requested_provider)
        if not request_provider or not isinstance(route, dict):
            return None
        if self._session_model_override_for(gateway_session_key or session_id):
            # Session /model wins over both the route and the request override, so
            # there is no ambiguity to reject on this request path.
            return None

        route_provider = _clean_request_string(route.get("provider"))
        route_api_key = _clean_request_string(route.get("api_key"))
        route_base_url = _clean_request_string(route.get("base_url"))
        route_alias = _clean_request_string(requested_model) or "requested model"

        if route_provider and request_provider != route_provider:
            return (
                f"Model route '{route_alias}' is pinned to provider '{route_provider}'. "
                f"Remove 'provider' or use '{route_provider}'."
            )
        if not route_provider and (route_api_key or route_base_url):
            return (
                f"Model route '{route_alias}' pins route credentials/base_url. "
                "Do not combine it with an explicit 'provider'."
            )
        return None

    def _create_agent(
        self,
        ephemeral_system_prompt: Optional[str] = None,
        session_id: Optional[str] = None,
        stream_delta_callback=None,
        tool_progress_callback=None,
        tool_start_callback=None,
        tool_complete_callback=None,
        gateway_session_key: Optional[str] = None,
        requested_model: Optional[str] = None,
        requested_provider: Optional[str] = None,
        model_options: Optional[Dict[str, Any]] = None,
        route: Optional[Dict[str, Any]] = None,
        session_model: Optional[str] = None,
        confirmed_runtime_lock: bool = False,
    ) -> Any:
        """
        Create an AIAgent instance using the gateway's runtime config.

        Uses _resolve_runtime_agent_kwargs() to pick up model, api_key,
        base_url, etc. from config.yaml / env vars.  Toolsets are resolved
        from config.yaml platform_toolsets.api_server (same as all other
        gateway platforms), falling back to the hermes-api-server default.

        ``gateway_session_key`` is a stable per-channel identifier supplied
        by the client (via ``X-Hermes-Session-Key``).  Unlike ``session_id``
        which scopes the short-term transcript and rotates on /new, this
        key is meant to persist across transcripts so long-term memory
        providers (e.g. Honcho) can scope their per-chat state correctly
        — matching the semantics of the native gateway's ``session_key``.

        ``route`` is an optional ``model_routes`` entry (per-client model
        routing).  When set — and no session ``/model`` override exists for
        this session — its model/provider/api_key/base_url override the
        global defaults for this agent instance only.

        ``session_model`` is the raw model persisted on a native API session
        row at creation time (``POST /api/sessions {"model": ...}``) when
        that value does not resolve to a ``model_routes`` alias.  Session-chat
        handlers pass either ``route`` (alias hit) or ``session_model`` (raw
        model), never both.  Precedence: session ``/model`` override →
        ``session_model`` → route alias / per-request selection → global.

        ``confirmed_runtime_lock`` marks a backend-acknowledged Browser model
        lock (POST /api/sessions/{id}/model).  A confirmed lock beats the
        session ``/model`` override, disables the global fallback model
        chain, and fails closed if the locked provider's credentials cannot
        be resolved.
        """
        from run_agent import AIAgent
        from gateway.run import (
            _checkpoint_agent_kwargs,
            _current_max_iterations,
            _resolve_runtime_agent_kwargs,
            _resolve_gateway_model,
            _load_gateway_config,
            GatewayRunner,
        )
        from hermes_cli.tools_config import _get_platform_tools

        # Catch RuntimeError ONLY around this call, not the wider
        # _create_agent()+run_conversation() span --
        # _resolve_runtime_agent_kwargs() is the sole raiser of
        # RuntimeError(format_runtime_provider_error(...)) for provider
        # auth/credential failure.  Re-raising as
        # _ProviderAuthResolutionError lets _run_agent() (and
        # _handle_runs()) distinguish this from an unrelated RuntimeError
        # elsewhere in the call graph.
        try:
            runtime_kwargs = _resolve_runtime_agent_kwargs()
        except RuntimeError as exc:
            raise _ProviderAuthResolutionError(str(exc)) from exc
        model = _resolve_gateway_model()

        # When the primary provider's auth fails (expired token / 429 quota
        # cap), _resolve_runtime_agent_kwargs() falls through to the fallback
        # provider chain, whose runtime dict carries its own ``model`` key.
        # Pop it and let it override the config model, mirroring the native
        # gateway path (_resolve_session_agent_runtime in run.py). Otherwise
        # the explicit ``model=model`` below collides with the ``**runtime_kwargs``
        # spread → "got multiple values for keyword argument 'model'", 500ing
        # every /v1/chat/completions request while a fallback is active.
        runtime_model = runtime_kwargs.pop("model", None)
        if runtime_model:
            model = runtime_model

        request_reasoning_config = _request_reasoning_config(model_options)
        request_service_tier = _request_service_tier(model_options)

        request_model = _clean_request_string(requested_model)
        request_provider = _clean_request_string(requested_provider)
        route_model = _clean_request_string(route.get("model")) if isinstance(route, dict) else None
        route_provider = _clean_request_string(route.get("provider")) if isinstance(route, dict) else None
        route_api_key = _clean_request_string(route.get("api_key")) if isinstance(route, dict) else None
        route_base_url = _clean_request_string(route.get("base_url")) if isinstance(route, dict) else None

        def _resolve_provider_runtime(
            provider: Optional[str],
            *,
            target_model: Optional[str],
            required: bool,
        ) -> Optional[Dict[str, Any]]:
            provider_name = _clean_request_string(provider)
            if not provider_name:
                return None
            try:
                return _resolve_request_runtime_agent_kwargs(
                    provider_name,
                    target_model=target_model or None,
                )
            except Exception as exc:
                try:
                    from gateway.run import _resolve_runtime_agent_kwargs_for_provider

                    return _resolve_runtime_agent_kwargs_for_provider(provider_name)
                except Exception:
                    pass
                if required:
                    # Surface as the typed provider-auth failure so
                    # _run_agent()/_handle_runs() return the controlled
                    # response shape instead of a raw 500.
                    raise _ProviderAuthResolutionError(str(exc)) from exc
                logger.debug(
                    "api_server provider-runtime refresh failed for provider=%s model=%s",
                    provider_name,
                    target_model or "",
                    exc_info=True,
                )
                return None

        # Final precedence mirrors the gateway contract:
        # confirmed Browser model lock → session /model override →
        # session-persisted model (POST /api/sessions {"model": ...}) →
        # model_routes mapping selected by the request model alias → direct
        # per-request provider/model → global defaults.  model_options stay
        # request-scoped regardless of which selection wins.  A confirmed
        # lock is an execution contract: it bypasses the session /model
        # override and fails closed (never reuses global credentials) if
        # its provider cannot be resolved.
        session_key = gateway_session_key or session_id
        session_row_model = _clean_request_string(session_model)
        session_override = None
        if not confirmed_runtime_lock:
            session_override = self._session_model_override_for(session_key)
        # Model-string precedence delegates to the shared owner
        # hermes_cli.model_switch.resolve_effective_model (session /model
        # override > session-persisted model > global) — the rule 7dd00bb47d
        # had to re-fix here after it diverged from gateway/run.py.
        from hermes_cli.model_switch import resolve_effective_model
        if session_override:
            override_model = resolve_effective_model(session_override, None, model)
            session_provider = _clean_request_string(session_override.get("provider"))
            current_provider = _clean_request_string(runtime_kwargs.get("provider"))
            provider_runtime = _resolve_provider_runtime(
                session_provider or current_provider,
                target_model=override_model,
                required=False,
            )
            if provider_runtime:
                _apply_runtime_agent_overrides(runtime_kwargs, provider_runtime)
            _apply_runtime_agent_overrides(runtime_kwargs, session_override)
            model = override_model
            if route or request_model or request_provider:
                logger.debug(
                    "api_server request selection skipped: session /model override wins for %s",
                    session_key or "",
                )
        elif session_row_model and not confirmed_runtime_lock:
            # Session-persisted model (raw string that resolved to no route
            # alias).  Pins this session's turns ahead of per-request body
            # values — a session's chosen model is a standing selection,
            # matching the native gateway's session-model semantics.
            current_provider = _clean_request_string(runtime_kwargs.get("provider"))
            provider_runtime = _resolve_provider_runtime(
                current_provider,
                target_model=session_row_model,
                required=False,
            )
            if provider_runtime:
                _apply_runtime_agent_overrides(runtime_kwargs, provider_runtime)
            model = resolve_effective_model(None, session_row_model, model)
            if request_model or request_provider:
                logger.debug(
                    "api_server request selection skipped: session-persisted model wins for %s",
                    session_key or "",
                )
        else:
            if route is not None:
                # The request's ``model`` field selected this route, so its
                # value is the route ALIAS — never usable as a model name.
                # A route with no ``model`` key keeps the global default
                # (pre-existing model_routes behavior).
                effective_model = route_model or model
            else:
                effective_model = request_model or model
            current_provider = _clean_request_string(runtime_kwargs.get("provider"))
            effective_provider = request_provider or route_provider or current_provider
            provider_runtime = None
            if effective_provider and (
                bool(request_provider or route_provider) or effective_model != model
            ):
                provider_runtime = _resolve_provider_runtime(
                    effective_provider,
                    target_model=effective_model,
                    # A confirmed Browser lock fails closed: if the locked
                    # provider cannot be resolved, never fall through to
                    # the previous global provider's credentials.
                    required=bool(request_provider) or confirmed_runtime_lock,
                )
            if provider_runtime:
                _apply_runtime_agent_overrides(runtime_kwargs, provider_runtime)
            elif effective_provider and effective_provider != current_provider:
                runtime_kwargs["provider"] = effective_provider
            model = effective_model
            # Per-route explicit transport secrets/base URLs win within the
            # route contract after provider resolution.
            if route_api_key:
                runtime_kwargs["api_key"] = route_api_key
            if route_base_url:
                runtime_kwargs["base_url"] = route_base_url
            if route:
                logger.debug(
                    "api_server request selection applied: model=%s provider=%s route_provider=%s request_provider=%s",
                    model,
                    runtime_kwargs.get("provider"),
                    route_provider or "",
                    request_provider or "",
                )

        # When the config has no model.default but a provider was resolved
        # (e.g. user ran `hermes auth add openai-codex` without `hermes model`),
        # fall back to the provider's first catalog model so the API call
        # doesn't fail with "model must be a non-empty string". Mirrors
        # run.py::_resolve_session_agent_runtime. Runs after the selection
        # block above so a route/session/request override that already
        # resolved a model is never treated as "empty" here.
        if not model and runtime_kwargs.get("provider"):
            try:
                from hermes_cli.models import get_default_model_for_provider
                model = get_default_model_for_provider(runtime_kwargs["provider"])
                if model:
                    logger.info(
                        "No model configured — defaulting to %s for provider %s",
                        model, runtime_kwargs["provider"],
                    )
            except Exception:
                pass

        # Final safety net (#35314): if resolution still produced an empty
        # model — e.g. a transient config-cache miss — reuse the last model
        # successfully resolved for this session (or, failing that, the most
        # recent one resolved process-wide). Building an agent with model=""
        # makes every API call fail HTTP 400 until a manual retry. Mirrors
        # run.py::_resolve_session_agent_runtime.
        #
        # Cache key is gateway_session_key ONLY, never session_id — unlike
        # run.py's native gateway (stable, long-lived chat scopes), the API
        # server hands out a fresh UUID session_id per one-off request
        # (/v1/responses, /v1/runs when no explicit session is supplied).
        # Keying on session_id would leave one permanent dict entry per
        # stateless request, growing unbounded for the life of the process.
        _resolved_key = gateway_session_key or ""
        if not model:
            _recovered = (self._last_resolved_model.get(_resolved_key)
                          or self._last_resolved_model.get("*"))
            if _recovered:
                logger.warning(
                    "Empty model resolved for session=%s — recovering "
                    "last-known-good model %s (config read likely returned "
                    "empty; see #35314)",
                    _resolved_key, _recovered,
                )
                model = _recovered
        elif model:
            if _resolved_key:
                self._last_resolved_model[_resolved_key] = model
            self._last_resolved_model["*"] = model

        user_config = _load_gateway_config()
        enabled_toolsets = _resolve_api_server_agent_toolsets(user_config)

        max_iterations = _current_max_iterations()

        # Load fallback provider chain so the API server platform has the
        # same fallback behaviour as Telegram/Discord/Slack (fixes #4954).
        fallback_model = (
            None
            if confirmed_runtime_lock
            else GatewayRunner._load_fallback_model()
        )

        # Resolve reasoning against the model this request will actually
        # run. Per-model ``agent.reasoning_overrides`` key off that model,
        # and it is only settled after the precedence chain above (browser
        # lock -> session /model -> session row -> route -> per-request ->
        # defaults). Resolving at function entry keyed them off
        # ``model.default`` instead — the defect e81d18dfb removed from the
        # native gateway paths. An explicit per-request reasoning parameter
        # still wins over config.
        reasoning_config = (
            request_reasoning_config
            if request_reasoning_config is not None
            else GatewayRunner._load_reasoning_config(model)
        )

        agent_kwargs = {
            "model": model,
            **runtime_kwargs,
            **_checkpoint_agent_kwargs(user_config),
            "max_iterations": max_iterations,
            "quiet_mode": True,
            "verbose_logging": False,
            "ephemeral_system_prompt": ephemeral_system_prompt or None,
            "enabled_toolsets": enabled_toolsets,
            "session_id": session_id,
            "platform": "api_server",
            "stream_delta_callback": stream_delta_callback,
            "tool_progress_callback": tool_progress_callback,
            "tool_start_callback": tool_start_callback,
            "tool_complete_callback": tool_complete_callback,
            "session_db": self._ensure_session_db(),
            "fallback_model": fallback_model,
            "reasoning_config": reasoning_config,
            "gateway_session_key": gateway_session_key,
        }
        if request_service_tier is not _REQUEST_OPTION_MISSING:
            agent_kwargs["service_tier"] = request_service_tier

        resolved_effort = ""
        if isinstance(reasoning_config, dict):
            if reasoning_config.get("enabled") is False:
                resolved_effort = "none"
            else:
                raw_effort = reasoning_config.get("effort")
                if isinstance(raw_effort, str):
                    resolved_effort = raw_effort.strip().lower()

        agent = AIAgent(**agent_kwargs)
        agent._hermes_api_runtime = {
            "provider": runtime_kwargs.get("provider") or getattr(agent, "provider", "") or "",
            "model": getattr(agent, "model", None) or model,
            "effort": resolved_effort,
            "engine": "native-hermes",
            "route_source": (
                "session_model_lock"
                if confirmed_runtime_lock
                else "session_model_override"
                if session_override
                else "raw_request"
                if route or request_model or request_provider
                else "global"
            ),
        }
        return agent

    # ------------------------------------------------------------------
    # HTTP Handlers
    # ------------------------------------------------------------------

    async def _handle_health(self, request: "web.Request") -> "web.Response":
        """GET /health — simple health check."""
        return web.json_response(
            {"status": "ok", "platform": "hermes-agent", "version": _hermes_version()}
        )

    async def _handle_health_detailed(self, request: "web.Request") -> "web.Response":
        """GET /health/detailed — rich status for cross-container dashboard probing.

        Returns gateway state, connected platforms, PID, and uptime so the
        dashboard can display full status without needing a shared PID file or
        /proc access.  Requires the same Bearer auth as other API routes.
        """
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err

        from gateway.status import (
            derive_gateway_busy,
            derive_gateway_drainable,
            normalize_updated_at,
            parse_active_agents,
            read_runtime_status,
        )

        runtime = read_runtime_status() or {}
        gw_state = runtime.get("gateway_state")
        gw_active = parse_active_agents(runtime.get("active_agents", 0))
        # This endpoint is served BY the gateway process, so it is by definition
        # alive — gateway_running is True. Derive busy/drainable from the same
        # shared contract /api/status uses so the two surfaces never disagree.
        active_api_runs, process_depth, active_delegations = self._readiness_work_counts()
        from gateway.run import _resolve_gateway_model

        readiness = collect_runtime_readiness(
            configured_model=_resolve_gateway_model(),
            runtime_status=runtime,
            active_api_runs=active_api_runs,
            process_completion_queue_depth=process_depth,
            active_delegations=active_delegations,
        )
        return web.json_response({
            "status": readiness["status"],
            "readiness": readiness,
            "platform": "hermes-agent",
            "version": _hermes_version(),
            "gateway_state": gw_state,
            "platforms": runtime.get("platforms", {}),
            "active_agents": gw_active,
            "gateway_busy": derive_gateway_busy(
                gateway_running=True,
                gateway_state=gw_state,
                active_agents=gw_active,
            ),
            "gateway_drainable": derive_gateway_drainable(
                gateway_running=True,
                gateway_state=gw_state,
            ),
            "exit_reason": runtime.get("exit_reason"),
            # Contract: updated_at is RFC3339 string | null, never a number —
            # the state file may carry legacy epoch floats or hand-edited junk.
            "updated_at": normalize_updated_at(runtime.get("updated_at")),
            "pid": os.getpid(),
        })

    async def _handle_models(self, request: "web.Request") -> "web.Response":
        """GET /v1/models — list hermes-agent and any configured model_routes aliases.

        Under ``/p/<profile>/v1/models`` (multiplex on) the advertised primary
        model id follows that profile's name/config, not the default adapter's
        cached ``_model_name``.
        """
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err

        now = int(time.time())
        # Middleware already entered the profile runtime scope when a /p/
        # prefix was present, so get_active_profile_name() resolves correctly.
        model_name = (
            self._resolve_model_name("")
            if _api_request_profile.get()
            else self._model_name
        )
        models = [
            {
                "id": model_name,
                "object": "model",
                "created": now,
                "owned_by": "hermes",
                "permission": [],
                "root": model_name,
                "parent": None,
            }
        ]
        # Expose configured model route aliases so clients can discover them.
        # Only the alias and resolved model name are exposed — never provider
        # credentials.
        for alias, route_cfg in self._model_routes.items():
            if alias == model_name:
                continue  # already listed above
            models.append({
                "id": alias,
                "object": "model",
                "created": now,
                "owned_by": "hermes",
                "permission": [],
                "root": route_cfg.get("model", alias),
                "parent": model_name,
            })

        return web.json_response({"object": "list", "data": models})

    async def _handle_model_options(self, request: "web.Request") -> "web.Response":
        """GET /api/model/options — return Hermes provider/model inventory.

        This mirrors the dashboard/TUI model picker inventory endpoint so
        external clients using the API server can sync to the user's configured
        Hermes provider catalog instead of scraping the single OpenAI-compatible
        `/v1/models` alias.
        """
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err

        refresh = _coerce_request_bool(request.query.get("refresh"), default=False)
        try:
            from hermes_cli.inventory import build_model_options_payload, load_picker_context

            def _build_payload() -> Dict[str, Any]:
                return build_model_options_payload(
                    load_picker_context(),
                    include_unconfigured=True,
                    refresh=refresh,
                )

            # Inventory enrichment can fetch pricing and provider catalogs.
            # Keep all synchronous picker work off aiohttp's event loop.
            payload = await asyncio.to_thread(_build_payload)
            return web.json_response(payload)
        except Exception:
            logger.exception("[%s] GET /api/model/options failed", self.name)
            return web.json_response(
                _openai_error(
                    "Failed to list model options.",
                    code="model_options_failed",
                ),
                status=500,
            )

    async def _handle_capabilities(self, request: "web.Request") -> "web.Response":
        """GET /v1/capabilities — advertise the stable API surface.

        External UIs and orchestrators use this endpoint to discover the API
        server's plugin-safe contract without scraping docs or assuming that
        every Hermes version exposes the same endpoints.
        """
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err

        return web.json_response({
            "object": "hermes.api_server.capabilities",
            "platform": "hermes-agent",
            "model": self._model_name,
            "auth": {
                "type": "bearer",
                "required": bool(self._api_key),
            },
            "runtime": {
                "mode": "server_agent",
                "tool_execution": "server",
                "split_runtime": False,
                "description": (
                    "The API server creates a server-side Hermes AIAgent; "
                    "tools execute on the API-server host unless a future "
                    "explicit split-runtime mode is enabled."
                ),
            },
            "features": {
                "chat_completions": True,
                "chat_completions_streaming": True,
                "responses_api": True,
                "responses_streaming": True,
                "run_submission": True,
                "run_status": True,
                "run_events_sse": True,
                "run_stop": True,
                "run_steer": True,
                "run_approval_response": True,
                "tool_progress_events": True,
                "approval_events": True,
                "session_resources": True,
                "model_options": True,
                "session_chat": True,
                "session_chat_streaming": True,
                "session_fork": True,
                "session_model_lock": True,
                "admin_config_rw": False,
                "jobs_admin": False,
                "memory_write_api": False,
                "skills_api": True,
                "audio_api": False,
                "realtime_voice": False,
                "session_continuity_header": "X-Hermes-Session-Id",
                "session_key_header": "X-Hermes-Session-Key",
                "cors": bool(self._cors_origins),
            },
            "endpoints": {
                "health": {"method": "GET", "path": "/health"},
                "health_detailed": {"method": "GET", "path": "/health/detailed"},
                "models": {"method": "GET", "path": "/v1/models"},
                "model_options": {"method": "GET", "path": "/api/model/options"},
                "chat_completions": {"method": "POST", "path": "/v1/chat/completions"},
                "responses": {"method": "POST", "path": "/v1/responses"},
                "runs": {"method": "POST", "path": "/v1/runs"},
                "run_status": {"method": "GET", "path": "/v1/runs/{run_id}"},
                "run_events": {"method": "GET", "path": "/v1/runs/{run_id}/events"},
                "run_approval": {"method": "POST", "path": "/v1/runs/{run_id}/approval"},
                "run_steer": {"method": "POST", "path": "/v1/runs/{run_id}/steer"},
                "run_stop": {"method": "POST", "path": "/v1/runs/{run_id}/stop"},
                "skills": {"method": "GET", "path": "/v1/skills"},
                "toolsets": {"method": "GET", "path": "/v1/toolsets"},
                "sessions": {"method": "GET", "path": "/api/sessions"},
                "session_create": {"method": "POST", "path": "/api/sessions"},
                "session": {"method": "GET", "path": "/api/sessions/{session_id}"},
                "session_update": {"method": "PATCH", "path": "/api/sessions/{session_id}"},
                "session_delete": {"method": "DELETE", "path": "/api/sessions/{session_id}"},
                "session_messages": {"method": "GET", "path": "/api/sessions/{session_id}/messages"},
                "session_fork": {"method": "POST", "path": "/api/sessions/{session_id}/fork"},
                "session_chat": {"method": "POST", "path": "/api/sessions/{session_id}/chat"},
                "session_chat_stream": {"method": "POST", "path": "/api/sessions/{session_id}/chat/stream"},
                "session_model_lock": {"method": "POST", "path": "/api/sessions/{session_id}/model"},
            },
        })

    async def _handle_skills(self, request: "web.Request") -> "web.Response":
        """GET /v1/skills — list installed skills visible to the API-server agent.

        Read-only listing intended for external clients that need to know
        which skills are available without sending a chat message and asking
        the model. Mirrors what the gateway/CLI surfaces through
        ``/skills list``, but as a deterministic JSON payload.

        Returns the same skill metadata (name, description, category) the
        skills hub uses internally. Disabled skills are excluded so the
        listing matches what the agent actually loads.
        """
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err

        try:
            from tools.skills_tool import _find_all_skills, _sort_skills
            skills = _sort_skills(_find_all_skills(skip_disabled=False))
        except Exception:
            logger.exception("GET /v1/skills failed")
            return web.json_response(
                _openai_error("Failed to enumerate skills", err_type="server_error"),
                status=500,
            )

        return web.json_response({
            "object": "list",
            "data": skills,
        })

    async def _handle_toolsets(self, request: "web.Request") -> "web.Response":
        """GET /v1/toolsets — list toolsets and their resolved tools.

        Returns the toolset surface the api_server platform actually exposes
        to its agent: each toolset's enabled/configured state plus the
        concrete tool names it expands to. This is the deterministic
        equivalent of what a client would otherwise have to recover by
        asking the model what tools it can call.
        """
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err

        try:
            from hermes_cli.config import load_config
            from hermes_cli.tools_config import (
                _get_effective_configurable_toolsets,
                _get_platform_tools,
                _toolset_has_keys,
                get_nous_subscription_features,
            )
            from toolsets import resolve_toolset

            config = load_config()
            enabled_toolsets = _get_platform_tools(
                config,
                "api_server",
                include_default_mcp_servers=False,
            )
            features = get_nous_subscription_features(config)
            data: List[Dict[str, Any]] = []
            for name, label, desc in _get_effective_configurable_toolsets():
                try:
                    tools = sorted(set(resolve_toolset(name)))
                except Exception:
                    tools = []
                is_enabled = name in enabled_toolsets
                data.append({
                    "name": name,
                    "label": label,
                    "description": desc,
                    "enabled": is_enabled,
                    "configured": _toolset_has_keys(name, config, features=features),
                    "tools": tools,
                })
        except Exception:
            logger.exception("GET /v1/toolsets failed")
            return web.json_response(
                _openai_error("Failed to enumerate toolsets", err_type="server_error"),
                status=500,
            )

        return web.json_response({
            "object": "list",
            "platform": "api_server",
            "data": data,
        })

    # ------------------------------------------------------------------
    # /api/sessions — thin client/session resource API
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_nonnegative_int(value: Any, default: int, maximum: int) -> int:
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            return default
        if parsed < 0:
            return default
        return min(parsed, maximum)

    @staticmethod
    def _session_response(session: Dict[str, Any]) -> Dict[str, Any]:
        """Return a stable, client-safe session representation."""
        safe_keys = (
            "id", "source", "user_id", "model", "title", "started_at", "ended_at",
            "end_reason", "message_count", "tool_call_count", "input_tokens",
            "output_tokens", "cache_read_tokens", "cache_write_tokens",
            "reasoning_tokens", "estimated_cost_usd", "actual_cost_usd",
            "api_call_count", "parent_session_id", "last_active", "preview",
            "_lineage_root_id", "pinned", "archived", "hidden",
        )
        payload = {key: session.get(key) for key in safe_keys if key in session}
        # SQLite stores these as 0/1; clients reconcile against a real boolean.
        for flag in ("pinned", "archived", "hidden"):
            if flag in payload:
                payload[flag] = bool(payload[flag])
        # Avoid exposing full system prompts/model_config through the client API;
        # callers only need to know whether those snapshots exist.
        payload["has_system_prompt"] = bool(session.get("system_prompt"))
        payload["has_model_config"] = bool(session.get("model_config"))
        return payload

    @staticmethod
    def _message_response(message: Dict[str, Any]) -> Dict[str, Any]:
        safe_keys = (
            "id", "session_id", "role", "content", "tool_call_id", "tool_calls",
            "tool_name", "timestamp", "token_count", "finish_reason", "reasoning",
            "reasoning_content",
        )
        return {key: message.get(key) for key in safe_keys if key in message}

    async def _read_json_body(self, request: "web.Request") -> tuple[Dict[str, Any], Optional["web.Response"]]:
        try:
            body = await request.json()
        except Exception:
            return {}, web.json_response(_openai_error("Invalid JSON in request body"), status=400)
        if not isinstance(body, dict):
            return {}, web.json_response(_openai_error("Request body must be a JSON object"), status=400)
        return body, None

    async def _get_existing_session_or_404(self, session_id: str) -> tuple[Optional[Dict[str, Any]], Optional["web.Response"]]:
        db = await self._ensure_session_db_async()
        if db is None:
            return None, web.json_response(_openai_error("Session database unavailable", code="session_db_unavailable"), status=503)
        # Offload the blocking SQLite read off the event loop (CWE/perf: the
        # API server is single-threaded aiohttp; a sync SessionDB call here
        # freezes every in-flight request, see PR discussion on event-loop
        # blocking SQLite in the gateway surface).
        session = await asyncio.to_thread(db.get_session, session_id)
        if not session:
            return None, web.json_response(_openai_error(f"Session not found: {session_id}", code="session_not_found"), status=404)
        return session, None

    async def _conversation_history_for_session(self, session_id: str) -> List[Dict[str, Any]]:
        db = await self._ensure_session_db_async()
        if db is None:
            return []
        try:
            return await asyncio.to_thread(db.get_messages_as_conversation, session_id)
        except Exception as exc:
            logger.warning("Failed to load session history for %s: %s", session_id, exc)
            return []

    async def _handle_list_sessions(self, request: "web.Request") -> "web.Response":
        """GET /api/sessions — list persisted Hermes sessions."""
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err

        db = await self._ensure_session_db_async()
        if db is None:
            return web.json_response(_openai_error("Session database unavailable", code="session_db_unavailable"), status=503)

        limit = self._parse_nonnegative_int(request.query.get("limit"), default=50, maximum=200)
        offset = self._parse_nonnegative_int(request.query.get("offset"), default=0, maximum=1_000_000)
        source = request.query.get("source") or None
        include_children = _coerce_request_bool(request.query.get("include_children"), default=False)
        sessions = await asyncio.to_thread(db.list_sessions_rich,
            source=source,
            limit=limit,
            offset=offset,
            include_children=include_children,
            order_by_last_active=True,
            # A pin means "always reachable", so a pinned conversation that has
            # aged past the recency window is back-filled rather than dropped.
            include_pinned=True,
        )
        # Back-filled pins arrive PAST the limit, so counting them would report
        # another page that doesn't exist. Only the recency window decides.
        windowed = sum(1 for s in sessions if not s.get("pinned"))
        return web.json_response({
            "object": "list",
            "data": [self._session_response(s) for s in sessions],
            "limit": limit,
            "offset": offset,
            "has_more": windowed >= limit,
        })

    async def _handle_create_session(self, request: "web.Request") -> "web.Response":
        """POST /api/sessions -- create an empty Hermes session row.

        The existence check, insert, title handling, and invalid-title
        rollback run as a single off-loop operation to avoid a TOCTOU
        window between the duplicate check and the insert (concurrent
        same-ID creates could otherwise both pass the check and both
        return 201 via the ON CONFLICT enrichment upsert).
        """
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err
        body, err = await self._read_json_body(request)
        if err:
            return err

        db = await self._ensure_session_db_async()
        if db is None:
            return web.json_response(_openai_error("Session database unavailable", code="session_db_unavailable"), status=503)

        raw_id = body.get("id") or body.get("session_id")
        session_id = str(raw_id).strip() if raw_id else f"api_{int(time.time())}_{uuid.uuid4().hex[:8]}"
        from gateway.session import _is_path_unsafe
        if not session_id or re.search(r'[\r\n\x00]', session_id) or _is_path_unsafe(session_id):
            return web.json_response(_openai_error("Invalid session ID", code="invalid_session_id"), status=400)
        if len(session_id) > self._MAX_SESSION_HEADER_LEN:
            return web.json_response(_openai_error("Session ID too long", code="invalid_session_id"), status=400)

        model = body.get("model") or self._model_name
        system_prompt = body.get("system_prompt")
        if system_prompt is not None and not isinstance(system_prompt, str):
            return web.json_response(_openai_error("system_prompt must be a string", code="invalid_system_prompt"), status=400)
        source = self._normalize_session_source(body.get("source") or "api_server")
        runtime_request = self._session_runtime_request_from_body(body)
        lock_error = self._runtime_lock_error(runtime_request)
        if lock_error is not None:
            return lock_error
        requested = runtime_request.get("requested") or {}
        model_name = self._clean_runtime_id(requested.get("model")) or (str(model) if model else None)
        model_config = None
        if requested.get("model") or requested.get("provider"):
            model_config = {
                "browser_model_lock": {
                    "provider": requested.get("provider") or "",
                    "model": requested.get("model") or "",
                    "model_options": runtime_request.get("model_options") or {},
                    "route_source": runtime_request.get("route_source") or "",
                    "confirmed": bool(runtime_request.get("require_model_lock")),
                    "updated_at": time.time(),
                }
            }
        title = body.get("title")

        # Run the entire check-insert-title sequence inside a single
        # _execute_write call (BEGIN IMMEDIATE + commit) so the existence
        # check and the insert are atomic at the SQLite level.  Two
        # concurrent requests for the same ID serialize here: the second
        # one blocks on the write lock and sees the row the first inserted.
        def _do_create():
            def _atomic(conn):
                row = conn.execute(
                    "SELECT id FROM sessions WHERE id = ?", (session_id,)
                ).fetchone()
                if row:
                    return None, "exists"
                import time as _time
                conn.execute(
                    """INSERT INTO sessions (
                       id, source, model, model_config, system_prompt, started_at
                    ) VALUES (?, ?, ?, ?, ?, ?)""",
                    (
                        session_id,
                        source,
                        model_name,
                        json.dumps(model_config) if model_config else None,
                        system_prompt,
                        _time.time(),
                    ),
                )
                if title is not None:
                    clean_title = db.sanitize_title(str(title))
                    if clean_title:
                        conflict = conn.execute(
                            "SELECT id FROM sessions WHERE title = ? AND id != ?",
                            (clean_title, session_id),
                        ).fetchone()
                        if conflict:
                            conn.execute(
                                "DELETE FROM sessions WHERE id = ?", (session_id,)
                            )
                            return None, f"title:Title already in use by session {conflict['id']}"
                    conn.execute(
                        "UPDATE sessions SET title = ? WHERE id = ?",
                        (clean_title, session_id),
                    )
                session_row = conn.execute(
                    "SELECT * FROM sessions WHERE id = ?", (session_id,)
                ).fetchone()
                return (dict(session_row) if session_row else {
                    "id": session_id, "source": source,
                    "model": model_name, "title": title,
                }), None
            return db._execute_write(_atomic)

        session, err = await asyncio.to_thread(_do_create)
        if err == "exists":
            return web.json_response(_openai_error(f"Session already exists: {session_id}", code="session_exists"), status=409)
        if err and err.startswith("title:"):
            return web.json_response(_openai_error(err[len("title:"):], code="invalid_title"), status=400)
        return web.json_response({"object": "hermes.session", "session": self._session_response(session)}, status=201)

    async def _handle_get_session(self, request: "web.Request") -> "web.Response":
        """GET /api/sessions/{session_id}."""
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err
        session, err = await self._get_existing_session_or_404(request.match_info["session_id"])
        if err:
            return err
        return web.json_response({"object": "hermes.session", "session": self._session_response(session)})

    async def _handle_patch_session(self, request: "web.Request") -> "web.Response":
        """PATCH /api/sessions/{session_id} — update client-safe session metadata."""
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err
        session_id = request.match_info["session_id"]
        session, err = await self._get_existing_session_or_404(session_id)
        if err:
            return err
        body, err = await self._read_json_body(request)
        if err:
            return err
        # `pinned` and `archived` are durable per-session flags the desktop
        # sidebar owns (the "keep" flag exempts a chat from the auto-archive
        # sweep). Rejecting them here was silently 400ing every pin the desktop
        # made, so pins only ever lived in that one app's localStorage.
        # `unread` is the read-state watermark toggle (same desktop owner).
        allowed = {"title", "end_reason", "pinned", "archived", "hidden", "unread"}
        unknown = sorted(set(body) - allowed)
        if unknown:
            return web.json_response(_openai_error(f"Unsupported session fields: {', '.join(unknown)}", code="unsupported_session_field"), status=400)

        for flag in ("pinned", "archived", "hidden", "unread"):
            if flag in body and not isinstance(body[flag], bool):
                return web.json_response(_openai_error(f"'{flag}' must be a boolean", code="invalid_session_field"), status=400)

        db = await self._ensure_session_db_async()
        if db is None:
            return web.json_response(_openai_error("Session database unavailable", code="session_db_unavailable"), status=503)
        if "title" in body:
            try:
                await asyncio.to_thread(db.set_session_title, session_id, "" if body["title"] is None else str(body["title"]))
            except ValueError as exc:
                return web.json_response(_openai_error(str(exc), code="invalid_title"), status=400)
        if "pinned" in body:
            await asyncio.to_thread(db.set_session_pinned, session_id, body["pinned"])
        if "archived" in body:
            await asyncio.to_thread(db.set_session_archived, session_id, body["archived"])
        if "hidden" in body:
            await asyncio.to_thread(db.set_session_hidden, session_id, body["hidden"])
        if "unread" in body:
            await asyncio.to_thread(db.set_session_read, session_id, read=not body["unread"])
        if body.get("end_reason"):
            await asyncio.to_thread(db.end_session, session_id, str(body["end_reason"]))
        session = await asyncio.to_thread(db.get_session, session_id) or session
        return web.json_response({"object": "hermes.session", "session": self._session_response(session)})

    async def _handle_delete_session(self, request: "web.Request") -> "web.Response":
        """DELETE /api/sessions/{session_id}."""
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err
        session_id = request.match_info["session_id"]
        session, err = await self._get_existing_session_or_404(session_id)
        if err:
            return err
        db = await self._ensure_session_db_async()
        deleted = await asyncio.to_thread(db.delete_session, session_id)
        return web.json_response({"object": "hermes.session.deleted", "id": session_id, "deleted": bool(deleted)})

    async def _handle_session_messages(self, request: "web.Request") -> "web.Response":
        """GET /api/sessions/{session_id}/messages."""
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err
        session_id = request.match_info["session_id"]
        _, err = await self._get_existing_session_or_404(session_id)
        if err:
            return err
        db = await self._ensure_session_db_async()
        resolved_id = await asyncio.to_thread(db.resolve_resume_session_id, session_id)
        raw_limit = request.query.get("limit")
        raw_offset = request.query.get("offset", "0")
        order = request.query.get("order")
        if order not in (None, "oldest", "latest"):
            return web.json_response(
                _openai_error(
                    "order must be one of: oldest, latest",
                    code="invalid_pagination",
                ),
                status=400,
            )
        try:
            offset = int(raw_offset)
            requested_limit = None if raw_limit is None else int(raw_limit)
        except (TypeError, ValueError):
            offset = -1
            requested_limit = -1
        if offset < 0 or (requested_limit is not None and requested_limit < 0):
            return web.json_response(
                _openai_error(
                    "limit and offset must be non-negative integers",
                    code="invalid_pagination",
                ),
                status=400,
            )

        default_page = requested_limit is None
        latest_page = order == "latest" or (order is None and default_page)
        limit = 500 if default_page else min(requested_limit, 500)
        messages = await asyncio.to_thread(
            db.get_messages,
            resolved_id,
            limit=limit,
            offset=offset,
            latest=latest_page,
        )
        return web.json_response({
            "object": "list",
            "session_id": resolved_id,
            "data": [self._message_response(m) for m in messages],
            "pagination": {
                "limit": limit,
                "offset": offset,
                "order": order or ("latest" if default_page else "oldest"),
                "returned": len(messages),
            },
        })

    async def _handle_fork_session(self, request: "web.Request") -> "web.Response":
        """POST /api/sessions/{session_id}/fork — branch via current SessionDB primitives."""
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err
        source_id = request.match_info["session_id"]
        source, err = await self._get_existing_session_or_404(source_id)
        if err:
            return err
        body, err = await self._read_json_body(request)
        if err:
            return err
        db = await self._ensure_session_db_async()
        fork_id = str(body.get("id") or body.get("session_id") or f"api_{int(time.time())}_{uuid.uuid4().hex[:8]}").strip()
        if not fork_id or re.search(r'[\r\n\x00]', fork_id):
            return web.json_response(_openai_error("Invalid session ID", code="invalid_session_id"), status=400)
        if await asyncio.to_thread(db.get_session, fork_id):
            return web.json_response(_openai_error(f"Session already exists: {fork_id}", code="session_exists"), status=409)

        # Match the CLI /branch semantics: mark the original as branched, then
        # create a child session that carries the transcript forward. This uses
        # SessionDB's native parent_session_id/end_reason visibility model rather
        # than inventing a parallel fork store.
        await asyncio.to_thread(db.end_session, source_id, "branched")
        await asyncio.to_thread(db.create_session,
            fork_id,
            "api_server",
            model=source.get("model"),
            system_prompt=source.get("system_prompt"),
            parent_session_id=source_id,
        )
        messages = await asyncio.to_thread(db.get_messages, source_id)
        await asyncio.to_thread(db.replace_messages, fork_id, messages)
        title = body.get("title")
        if title is None:
            base = source.get("title") or "fork"
            try:
                title = await asyncio.to_thread(db.get_next_title_in_lineage, base)
            except Exception:
                title = f"{base} fork"
        try:
            await asyncio.to_thread(db.set_session_title, fork_id, str(title))
        except ValueError as exc:
            return web.json_response(_openai_error(str(exc), code="invalid_title"), status=400)
        fork = await asyncio.to_thread(db.get_session, fork_id) or {"id": fork_id, "parent_session_id": source_id}
        return web.json_response({"object": "hermes.session", "session": self._session_response(fork)}, status=201)

    @_admit_api_agent_request
    async def _handle_session_chat(self, request: "web.Request") -> "web.Response":
        """POST /api/sessions/{session_id}/chat — one synchronous agent turn."""
        gateway_session_key, key_err = self._parse_session_key_header(request)
        if key_err is not None:
            return key_err
        session_id = request.match_info["session_id"]
        session, err = await self._get_existing_session_or_404(session_id)
        if err:
            return err
        body, err = await self._read_json_body(request)
        if err:
            return err
        user_message, err = _session_chat_user_message(body)
        if err is not None:
            return err
        system_prompt = body.get("system_message") or body.get("instructions")
        if system_prompt is not None and not isinstance(system_prompt, str):
            return web.json_response(_openai_error("system_message must be a string", code="invalid_system_message"), status=400)
        # Runtime selection. A backend-acknowledged Browser model lock
        # (require_model_lock in the body, or a previously confirmed lock
        # persisted on the session row) is an execution contract and wins.
        # Otherwise: session-persisted model (POST /api/sessions
        # {"model": ...}) — previously fetched and discarded here — routes
        # through model_routes when it is an alias (route
        # provider/credentials come along) or threads through as
        # session_model when it is a raw string; per-request body values
        # come after that.
        runtime_request = self._effective_session_runtime_request(
            session=session,
            body=body,
        )
        lock_error = self._runtime_lock_error(runtime_request)
        if lock_error is not None:
            return lock_error
        if not self._persist_session_runtime_lock(session_id, runtime_request):
            return web.json_response(
                _openai_error(
                    "Could not persist the requested session model lock",
                    code="model_lock_persistence_failed",
                ),
                status=500,
            )
        lock_active = bool(runtime_request.get("require_model_lock"))
        if lock_active:
            route = runtime_request.get("route")
            session_model = None
            requested = runtime_request.get("requested") or {}
            agent_overrides: Dict[str, Any] = {}
            if requested.get("model"):
                agent_overrides["requested_model"] = requested["model"]
            if requested.get("provider"):
                agent_overrides["requested_provider"] = requested["provider"]
            if runtime_request.get("model_options"):
                agent_overrides["model_options"] = runtime_request["model_options"]
        else:
            stored_model = session.get("model") if isinstance(session, dict) else None
            stored_route = self._resolve_route(stored_model)
            route = stored_route or self._resolve_route(body.get("model"))
            session_model = stored_model if (stored_model and stored_route is None) else None
            agent_overrides = _request_agent_overrides(body, virtual_model=self._model_name)
            selection_error = self._request_route_conflict_error(
                session_id=session_id,
                gateway_session_key=gateway_session_key,
                requested_model=agent_overrides.get("requested_model"),
                requested_provider=agent_overrides.get("requested_provider"),
                route=route,
            )
            if selection_error:
                return web.json_response(_openai_error(selection_error), status=400)
        history = await self._conversation_history_for_session(session_id)
        result, usage = await self._run_agent(
            user_message=user_message,
            conversation_history=history,
            ephemeral_system_prompt=system_prompt,
            session_id=session_id,
            gateway_session_key=gateway_session_key,
            route=route,
            session_model=session_model,
            requested_runtime=runtime_request.get("requested") or {},
            route_source=runtime_request.get("route_source") or "global",
            confirmed_runtime_lock=lock_active,
            **agent_overrides,
        )
        effective_session_id = result.get("session_id") if isinstance(result, dict) else session_id
        final_response = _resolve_media_to_data_urls(result.get("final_response", "") if isinstance(result, dict) else "")
        headers = {"X-Hermes-Session-Id": effective_session_id or session_id}
        if gateway_session_key:
            headers["X-Hermes-Session-Key"] = gateway_session_key
        runtime = {}
        if isinstance(result, dict):
            runtime = result.get("runtime") or {}
        if not runtime and isinstance(usage, dict):
            runtime = usage.get("runtime") or {}
        runtime = self._sanitize_runtime_metadata(
            runtime=runtime,
            requested_runtime=runtime_request.get("requested"),
            route_source=runtime_request.get("route_source") or "global",
            model_lock=(
                "confirmed"
                if runtime and runtime_request.get("require_model_lock")
                else "accepted"
                if runtime_request.get("require_model_lock")
                else ""
            ),
        )
        return web.json_response(
            {
                "object": "hermes.session.chat.completion",
                "session_id": effective_session_id or session_id,
                "message": {"role": "assistant", "content": final_response},
                "usage": usage,
                "runtime": runtime,
            },
            headers=headers,
        )

    @_admit_api_agent_request
    async def _handle_session_chat_stream(self, request: "web.Request") -> "web.StreamResponse":
        """POST /api/sessions/{session_id}/chat/stream — SSE wrapper over _run_agent."""
        gateway_session_key, key_err = self._parse_session_key_header(request)
        if key_err is not None:
            return key_err
        session_id = request.match_info["session_id"]
        session, err = await self._get_existing_session_or_404(session_id)
        if err:
            return err
        body, err = await self._read_json_body(request)
        if err:
            return err
        user_message, err = _session_chat_user_message(body)
        if err is not None:
            return err
        system_prompt = body.get("system_message") or body.get("instructions")
        if system_prompt is not None and not isinstance(system_prompt, str):
            return web.json_response(_openai_error("system_message must be a string", code="invalid_system_message"), status=400)
        # Runtime selection — mirrors _handle_session_chat (lock wins,
        # otherwise session-persisted model then per-request values).
        runtime_request = self._effective_session_runtime_request(
            session=session,
            body=body,
        )
        lock_error = self._runtime_lock_error(runtime_request)
        if lock_error is not None:
            return lock_error
        if not self._persist_session_runtime_lock(session_id, runtime_request):
            return web.json_response(
                _openai_error(
                    "Could not persist the requested session model lock",
                    code="model_lock_persistence_failed",
                ),
                status=500,
            )
        lock_active = bool(runtime_request.get("require_model_lock"))
        if lock_active:
            route = runtime_request.get("route")
            session_model = None
            requested = runtime_request.get("requested") or {}
            agent_overrides: Dict[str, Any] = {}
            if requested.get("model"):
                agent_overrides["requested_model"] = requested["model"]
            if requested.get("provider"):
                agent_overrides["requested_provider"] = requested["provider"]
            if runtime_request.get("model_options"):
                agent_overrides["model_options"] = runtime_request["model_options"]
        else:
            stored_model = session.get("model") if isinstance(session, dict) else None
            stored_route = self._resolve_route(stored_model)
            route = stored_route or self._resolve_route(body.get("model"))
            session_model = stored_model if (stored_model and stored_route is None) else None
            agent_overrides = _request_agent_overrides(body, virtual_model=self._model_name)
            selection_error = self._request_route_conflict_error(
                session_id=session_id,
                gateway_session_key=gateway_session_key,
                requested_model=agent_overrides.get("requested_model"),
                requested_provider=agent_overrides.get("requested_provider"),
                route=route,
            )
            if selection_error:
                return web.json_response(_openai_error(selection_error), status=400)
        runtime_meta = self._sanitize_runtime_metadata(
            requested_runtime=runtime_request.get("requested"),
            route_source=runtime_request.get("route_source") or "global",
            model_lock=("accepted" if lock_active else ""),
        )

        loop = asyncio.get_running_loop()
        queue: "asyncio.Queue[Optional[tuple[str, Dict[str, Any]]]]" = asyncio.Queue()
        message_id = f"msg_{uuid.uuid4().hex}"
        run_id = f"run_{uuid.uuid4().hex}"
        self._set_run_status(
            run_id,
            "queued",
            session_id=session_id,
            model=body.get("model", self._model_name),
        )
        seq = 0

        def _event_payload(name: str, payload: Dict[str, Any]) -> tuple[str, Dict[str, Any]]:
            nonlocal seq
            seq += 1
            payload.setdefault("session_id", session_id)
            payload.setdefault("run_id", run_id)
            payload.setdefault("seq", seq)
            payload.setdefault("ts", time.time())
            return name, payload

        def _enqueue(name: str, payload: Dict[str, Any]) -> None:
            event = _event_payload(name, payload)
            try:
                running_loop = asyncio.get_running_loop()
            except RuntimeError:
                running_loop = None
            try:
                if running_loop is loop:
                    queue.put_nowait(event)
                else:
                    loop.call_soon_threadsafe(queue.put_nowait, event)
            except RuntimeError:
                pass

        def _delta(delta: str) -> None:
            if delta:
                _enqueue("assistant.delta", {"message_id": message_id, "delta": delta})

        def _tool_progress(event_type: str, tool_name: str = None, preview: str = None, args=None, **kwargs) -> None:
            if event_type == "reasoning.available":
                _enqueue("tool.progress", {"message_id": message_id, "tool_name": tool_name or "_thinking", "delta": preview or ""})
            elif event_type in {"tool.started", "tool.completed", "tool.failed"}:
                event_name = event_type.replace("tool.", "tool.")
                _enqueue(event_name, {"message_id": message_id, "tool_name": tool_name, "preview": preview, "args": args})

        async def _run_and_signal() -> None:
            try:
                await queue.put(_event_payload("run.started", {
                    "user_message": {"role": "user", "content": user_message},
                    "runtime": runtime_meta,
                }))
                self._set_run_status(run_id, "running", last_event="run.started")
                await queue.put(_event_payload("message.started", {"message": {"id": message_id, "role": "assistant"}}))
                history = await self._conversation_history_for_session(session_id)
                result, usage = await self._run_agent(
                    user_message=user_message,
                    conversation_history=history,
                    ephemeral_system_prompt=system_prompt,
                    session_id=session_id,
                    stream_delta_callback=_delta,
                    tool_progress_callback=_tool_progress,
                    active_run_id=run_id,
                    gateway_session_key=gateway_session_key,
                    route=route,
                    session_model=session_model,
                    requested_runtime=runtime_request.get("requested") or {},
                    route_source=runtime_request.get("route_source") or "global",
                    confirmed_runtime_lock=lock_active,
                    **agent_overrides,
                )
                final_response = _resolve_media_to_data_urls(result.get("final_response", "") if isinstance(result, dict) else "")
                effective_session_id = result.get("session_id", session_id) if isinstance(result, dict) else session_id
                turn_messages = self._turn_transcript_messages(history, user_message, result) if isinstance(result, dict) else []
                effective_runtime = {}
                if isinstance(result, dict):
                    effective_runtime = result.get("runtime") or {}
                if not effective_runtime and isinstance(usage, dict):
                    effective_runtime = usage.get("runtime") or {}
                effective_runtime = self._sanitize_runtime_metadata(
                    runtime=effective_runtime,
                    requested_runtime=runtime_request.get("requested"),
                    route_source=runtime_request.get("route_source") or "global",
                    model_lock=(
                        "confirmed"
                        if effective_runtime and runtime_request.get("require_model_lock")
                        else "accepted"
                        if runtime_request.get("require_model_lock")
                        else ""
                    ),
                )
                await queue.put(_event_payload("assistant.completed", {
                    "session_id": effective_session_id,
                    "message_id": message_id,
                    "content": final_response,
                    "completed": True,
                    "partial": False,
                    "interrupted": False,
                    "runtime": effective_runtime,
                }))
                # A steer accepted after the final assistant response is drained
                # into result["pending_steer"] by the turn finalizer instead of
                # being consumed; surface it so clients can replay it as the
                # next user turn rather than silently losing it.
                pending_steer = result.get("pending_steer") if isinstance(result, dict) else None
                completed_payload = {
                    "session_id": effective_session_id,
                    "message_id": message_id,
                    "completed": True,
                    "messages": turn_messages,
                    "usage": usage,
                    "runtime": effective_runtime,
                }
                if pending_steer:
                    completed_payload["pending_steer"] = pending_steer
                await queue.put(_event_payload("run.completed", completed_payload))
                self._set_run_status(
                    run_id,
                    "completed",
                    session_id=effective_session_id,
                    usage=usage,
                    last_event="run.completed",
                    **({"pending_steer": pending_steer} if pending_steer else {}),
                )
            except asyncio.CancelledError:
                self._set_run_status(run_id, "cancelled", last_event="run.cancelled")
                raise
            except Exception as exc:
                logger.exception("[api_server] session chat stream failed")
                self._set_run_status(
                    run_id,
                    "failed",
                    error=_redact_api_error_text(exc),
                    last_event="run.failed",
                )
                await queue.put(_event_payload("error", {"message": _redact_api_error_text(exc)}))
            finally:
                self._active_run_agents.pop(run_id, None)
                await queue.put(_event_payload("done", {}))
                await queue.put(None)

        # NOTE: deliberately NOT registered in _active_run_tasks — this turn
        # is already counted by active_agent_work_count() via
        # _inflight_agent_runs (_run_agent), and a second task-based entry
        # would double-count it in the shutdown drain. Run-scoped control
        # needs only the agent ref, registered by _run_agent(active_run_id).
        task = asyncio.create_task(_run_and_signal())
        try:
            self._background_tasks.add(task)
        except TypeError:
            pass
        if hasattr(task, "add_done_callback"):
            task.add_done_callback(self._background_tasks.discard)

        headers = {
            "Content-Type": "text/event-stream",
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "X-Hermes-Session-Id": session_id,
        }
        if gateway_session_key:
            headers["X-Hermes-Session-Key"] = gateway_session_key
        response = web.StreamResponse(status=200, headers=headers)
        await response.prepare(request)
        try:
            while True:
                try:
                    item = await asyncio.wait_for(queue.get(), timeout=CHAT_COMPLETIONS_SSE_KEEPALIVE_SECONDS)
                except asyncio.TimeoutError:
                    await response.write(b": keepalive\n\n")
                    continue
                if item is None:
                    break
                name, payload = item
                await response.write(_sse_frame(payload, event=name, ensure_ascii=False))
        except (ConnectionResetError, ConnectionAbortedError, BrokenPipeError, OSError):
            await self._drain_session_stream_task_on_disconnect(
                run_id, task, interrupt_message="SSE client disconnected", shield_wait=False
            )
            logger.info("Session SSE client disconnected; interrupted live run %s", run_id)
        except asyncio.CancelledError:
            await self._drain_session_stream_task_on_disconnect(
                run_id, task, interrupt_message="SSE task cancelled", shield_wait=True
            )
            logger.info("Session SSE task cancelled; drained live run %s", run_id)
            raise
        except Exception as exc:
            logger.debug("[api_server] session SSE stream error: %s", exc)
        return response

    async def _drain_session_stream_task_on_disconnect(
        self,
        run_id: str,
        task: "asyncio.Task",
        *,
        interrupt_message: str,
        shield_wait: bool,
    ) -> None:
        """Preserve live run control refs until the executor-backed turn actually exits."""
        agent = self._active_run_agents.get(run_id)
        if agent is None:
            if not task.done():
                task.cancel()
                with suppress(Exception):
                    await task
            return
        with suppress(Exception):
            agent.interrupt(interrupt_message)
        if not task.done():
            with suppress(Exception):
                await (asyncio.shield(task) if shield_wait else task)

    async def _handle_session_model_lock(self, request: "web.Request") -> "web.Response":
        """POST /api/sessions/{session_id}/model — backend-ack a Browser model lock."""
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err
        session_id = request.match_info["session_id"]
        _, err = await self._get_existing_session_or_404(session_id)
        if err:
            return err
        body, err = await self._read_json_body(request)
        if err:
            return err
        runtime_request = self._session_runtime_request_from_body(body)
        runtime_request["require_model_lock"] = True
        lock_error = self._runtime_lock_error(runtime_request)
        if lock_error is not None:
            return lock_error
        if not self._persist_session_runtime_lock(session_id, runtime_request):
            return web.json_response(
                _openai_error(
                    "Could not persist the requested session model lock",
                    code="model_lock_persistence_failed",
                ),
                status=500,
            )
        requested = runtime_request.get("requested") or {}
        route = runtime_request.get("route") or {}
        runtime = self._sanitize_runtime_metadata(
            runtime={
                "provider": route.get("provider") or requested.get("provider") or "",
                "model": route.get("model") or requested.get("model") or "",
                "route_source": runtime_request.get("route_source") or "raw_request",
            },
            requested_runtime=requested,
            route_source=runtime_request.get("route_source") or "raw_request",
            model_lock="accepted",
        )
        return web.json_response({
            "object": "hermes.session.model_lock",
            "session_id": session_id,
            "runtime": runtime,
        })
    @_admit_api_agent_request
    async def _handle_chat_completions(self, request: "web.Request") -> "web.Response":
        """POST /v1/chat/completions — OpenAI Chat Completions format."""
        # Bound total in-flight agent runs (configurable; #7483).
        limited = self._concurrency_limited_response()
        if limited is not None:
            return limited

        # Parse request body
        try:
            body = await request.json()
        except (json.JSONDecodeError, Exception):
            return web.json_response(_openai_error("Invalid JSON in request body"), status=400)

        messages = body.get("messages")
        if not messages or not isinstance(messages, list):
            return web.json_response(
                {"error": {"message": "Missing or invalid 'messages' field", "type": "invalid_request_error"}},
                status=400,
            )

        stream = _coerce_request_bool(body.get("stream"), default=False)

        # Extract system message (becomes ephemeral system prompt layered ON TOP of core)
        system_prompt = None
        conversation_messages: List[Dict[str, str]] = []

        for idx, msg in enumerate(messages):
            role = msg.get("role", "")
            raw_content = msg.get("content", "")
            if role == "system":
                # System messages don't support images (Anthropic rejects, OpenAI
                # text-model systems don't render them).  Flatten to text.
                content = _normalize_chat_content(raw_content)
                if system_prompt is None:
                    system_prompt = content
                else:
                    system_prompt = system_prompt + "\n" + content
            elif role in {"user", "assistant"}:
                try:
                    content = _normalize_multimodal_content(raw_content)
                except ValueError as exc:
                    return _multimodal_validation_error(exc, param=f"messages[{idx}].content")
                conversation_messages.append({"role": role, "content": content})

        # Extract the last user message as the primary input
        user_message: Any = ""
        history = []
        if conversation_messages:
            user_message = conversation_messages[-1].get("content", "")
            history = conversation_messages[:-1]

        if not _content_has_visible_payload(user_message):
            return web.json_response(
                {"error": {"message": "No user message found in messages", "type": "invalid_request_error"}},
                status=400,
            )

        # Allow caller to scope long-term memory (e.g. Honcho) with a
        # stable per-channel identifier via X-Hermes-Session-Key.  This
        # is independent of X-Hermes-Session-Id: the key persists across
        # transcripts while the id rotates when the caller starts a new
        # transcript (i.e. /new semantics).  See _parse_session_key_header.
        gateway_session_key, key_err = self._parse_session_key_header(request)
        if key_err is not None:
            return key_err

        # Allow caller to continue an existing session by passing X-Hermes-Session-Id.
        # When provided, history is loaded from state.db instead of from the request body.
        #
        # Security: session continuation exposes conversation history, so it is
        # only allowed when the API key is configured and the request is
        # authenticated.  Without this gate, any unauthenticated client could
        # read arbitrary session history by guessing/enumerating session IDs.
        provided_session_id = request.headers.get("X-Hermes-Session-Id", "").strip()
        if provided_session_id:
            if not self._api_key:
                logger.warning(
                    "Session continuation via X-Hermes-Session-Id rejected: "
                    "no API key configured.  Set API_SERVER_KEY to enable "
                    "session continuity."
                )
                return web.json_response(
                    _openai_error(
                        "Session continuation requires API key authentication. "
                        "Configure API_SERVER_KEY to enable this feature."
                    ),
                    status=403,
                )
            # Sanitize: reject control characters that could enable header
            # injection, and path-traversal-shaped IDs that would escape the
            # sessions directory when interpolated into on-disk artifact
            # filenames (session snapshots, request dumps). Mirrors the native
            # gateway's entry-boundary guard (gateway.session._is_path_unsafe).
            from gateway.session import _is_path_unsafe
            if re.search(r'[\r\n\x00]', provided_session_id) or _is_path_unsafe(provided_session_id):
                return web.json_response(
                    {"error": {"message": "Invalid session ID", "type": "invalid_request_error"}},
                    status=400,
                )
            if len(provided_session_id) > self._MAX_SESSION_HEADER_LEN:
                return web.json_response(
                    {"error": {"message": "Session ID too long", "type": "invalid_request_error"}},
                    status=400,
                )
            session_id = provided_session_id
            try:
                db = await self._ensure_session_db_async()
                if db is not None:
                    history = await asyncio.to_thread(db.get_messages_as_conversation, session_id)
            except Exception as e:
                logger.warning("Failed to load session history for %s: %s", session_id, e)
                history = []
        else:
            # Derive a stable session ID from the conversation fingerprint so
            # that consecutive messages from the same Open WebUI (or similar)
            # conversation map to the same Hermes session.  The first user
            # message + system prompt are constant across all turns.
            first_user = ""
            for cm in conversation_messages:
                if cm.get("role") == "user":
                    first_user = cm.get("content", "")
                    break
            session_id = _derive_chat_session_id(system_prompt, first_user)
            # history already set from request body above

        completion_id = f"chatcmpl-{uuid.uuid4().hex[:29]}"
        model_name = body.get("model", self._model_name)
        created = int(time.time())

        # Per-client model routing: if the requested model matches a
        # configured model_routes alias, this request's agent is created
        # with that route's model/provider instead of the global default.
        route = self._resolve_route(model_name)
        agent_overrides = _request_agent_overrides(
            body,
            virtual_model=self._model_name,
            allow_bare_model=self._direct_model_requests,
        )
        selection_error = self._request_route_conflict_error(
            session_id=session_id,
            gateway_session_key=gateway_session_key,
            requested_model=agent_overrides.get("requested_model"),
            requested_provider=agent_overrides.get("requested_provider"),
            route=route,
        )
        if selection_error:
            return web.json_response(_openai_error(selection_error), status=400)

        if stream:
            _stream_q = ThreadSafeAsyncQueue()

            def _on_delta(delta):
                # Filter out None — the agent fires stream_delta_callback(None)
                # to signal the CLI display to close its response box before
                # tool execution, but the SSE writer uses None as end-of-stream
                # sentinel.  Forwarding it would prematurely close the HTTP
                # response, causing Open WebUI (and similar frontends) to miss
                # the final answer after tool calls.  The SSE loop detects
                # completion via agent_task.done() instead.
                # Called from the worker thread running run_conversation —
                # put_threadsafe (not put_nowait) is required here.
                if delta is not None:
                    _stream_q.put_threadsafe(delta)

            # Track which tool_call_ids we've emitted a "running" lifecycle
            # event for, so a "completed" event without a matching "running"
            # (e.g. internal/filtered tools) is silently dropped instead of
            # producing an orphaned event clients can't correlate.
            _started_tool_call_ids: set[str] = set()

            def _on_tool_start(tool_call_id, function_name, function_args):
                """Emit ``hermes.tool.progress`` with ``status: running``.

                Replaces the old ``tool_progress_callback("tool.started",
                ...)`` emit so SSE consumers receive a single event per
                tool start, carrying both the legacy ``tool``/``emoji``/
                ``label`` payload (for #6972 frontends) and the new
                ``toolCallId``/``status`` correlation fields (#16588).

                Skips tools whose names start with ``_`` so internal
                events (``_thinking``, …) stay off the wire — matching
                the prior ``_on_tool_progress`` filter exactly.
                """
                if not tool_call_id or function_name.startswith("_"):
                    return
                _started_tool_call_ids.add(tool_call_id)
                from agent.display import build_tool_preview, get_tool_emoji
                label = build_tool_preview(function_name, function_args) or function_name
                _stream_q.put_threadsafe(("__tool_progress__", {
                    "tool": function_name,
                    "emoji": get_tool_emoji(function_name),
                    "label": label,
                    "toolCallId": tool_call_id,
                    "status": "running",
                }))

            def _on_tool_complete(tool_call_id, function_name, function_args, function_result):
                """Emit the matching ``status: completed`` event.

                Dropped if the start was filtered (internal tool, missing
                id, or never seen) so clients never get an orphaned
                ``completed`` they can't correlate to a prior ``running``.
                """
                if not tool_call_id or tool_call_id not in _started_tool_call_ids:
                    return
                _started_tool_call_ids.discard(tool_call_id)
                _stream_q.put_threadsafe(("__tool_progress__", {
                    "tool": function_name,
                    "toolCallId": tool_call_id,
                    "status": "completed",
                }))

            # Start agent in background.  agent_ref is a mutable container
            # so the SSE writer can interrupt the agent on client disconnect.
            #
            # ``tool_progress_callback`` is intentionally not wired here:
            # it would duplicate every emit because ``run_agent`` fires it
            # side-by-side with ``tool_start_callback``/``tool_complete_callback``.
            # The structured callbacks are strictly richer (they carry
            # the tool_call id), so they own the chat-completions SSE channel.
            agent_ref = [None]
            agent_task = asyncio.ensure_future(self._run_agent(
                user_message=user_message,
                conversation_history=history,
                ephemeral_system_prompt=system_prompt,
                session_id=session_id,
                stream_delta_callback=_on_delta,
                tool_start_callback=_on_tool_start,
                tool_complete_callback=_on_tool_complete,
                agent_ref=agent_ref,
                gateway_session_key=gateway_session_key,
                **agent_overrides,
                route=route,
            ))
            # Ensure SSE drain loops can terminate without relying on polling
            # agent_task.done(), which can race with queue timeout checks.
            agent_task.add_done_callback(lambda _fut: _stream_q.put_nowait(None))

            return await self._write_sse_chat_completion(
                request, completion_id, model_name, created, _stream_q,
                agent_task, agent_ref, session_id=session_id,
                gateway_session_key=gateway_session_key,
            )

        # Non-streaming: run the agent (with optional Idempotency-Key)
        async def _compute_completion():
            return await self._run_agent(
                user_message=user_message,
                conversation_history=history,
                ephemeral_system_prompt=system_prompt,
                session_id=session_id,
                gateway_session_key=gateway_session_key,
                **agent_overrides,
                route=route,
            )

        idempotency_key = request.headers.get("Idempotency-Key")
        if idempotency_key:
            if re.fullmatch(r"[A-Za-z0-9._:-]{1,200}", idempotency_key) is None:
                return web.json_response(
                    _openai_error("Invalid Idempotency-Key"), status=400,
                )
            fp = _make_request_fingerprint(
                body,
                keys=["model", "provider", "model_options", "messages", "tools", "tool_choice", "stream"],
            )
            cache_scope = hashlib.sha256(
                f"{gateway_session_key or ''}\0{session_id or ''}".encode("utf-8")
            ).hexdigest()
            cache_key = (
                f"chat:{_active_owner_profile()}:{cache_scope}:{idempotency_key}"
            )
            try:
                result, usage = await _idem_cache.get_or_set(
                    cache_key, fp, _compute_completion,
                )
            except _IdempotencyConflict:
                return web.json_response(
                    _openai_error(
                        "Idempotency-Key was already used for a different request",
                        code="idempotency_conflict",
                    ),
                    status=409,
                )
            except Exception as e:
                logger.error("Error running agent for chat completions: %s", e, exc_info=True)
                return web.json_response(
                    _openai_error(f"Internal server error: {e}", err_type="server_error"),
                    status=500,
                )
        else:
            try:
                result, usage = await _compute_completion()
            except Exception as e:
                logger.error("Error running agent for chat completions: %s", e, exc_info=True)
                return web.json_response(
                    _openai_error(f"Internal server error: {e}", err_type="server_error"),
                    status=500,
                )

        final_response = _resolve_media_to_data_urls(result.get("final_response") or "")
        is_partial = bool(result.get("partial"))
        is_failed = bool(result.get("failed"))
        completed = bool(result.get("completed", True))
        raw_err_msg = result.get("error")
        err_msg = _redact_api_error_text(raw_err_msg) if raw_err_msg else raw_err_msg

        # Decide finish_reason. OpenAI uses "length" for truncation, "stop"
        # for normal completion, and downstream SDKs accept "error" / custom
        # codes. See issue #22496.
        if is_partial and err_msg and "truncat" in err_msg.lower():
            finish_reason = "length"
        elif is_failed or (not completed and err_msg):
            finish_reason = "error"
        else:
            finish_reason = "stop"

        response_headers = {
            "X-Hermes-Session-Id": result.get("session_id", session_id),
        }
        if gateway_session_key:
            response_headers["X-Hermes-Session-Key"] = gateway_session_key

        # Hard-fail path: no usable assistant text AND a real failure → 5xx
        # with OpenAI-style error envelope so SDK clients raise instead of
        # silently rendering the internal failure string as message.content.
        if not final_response and (is_failed or is_partial):
            err_body = _openai_error(
                err_msg or "Agent run did not produce a response.",
                err_type="server_error",
                code="agent_incomplete",
            )
            err_body["error"]["hermes"] = {
                "completed": completed,
                "partial": is_partial,
                "failed": is_failed,
            }
            response_headers["X-Hermes-Completed"] = "false"
            response_headers["X-Hermes-Partial"] = "true" if is_partial else "false"
            return web.json_response(err_body, status=502, headers=response_headers)

        # Soft-partial path: we have *some* text but the run did not complete
        # (e.g. truncation with partial buffered output). Still 200 but signal
        # truncation via finish_reason="length" + Hermes-specific extras.
        response_data = {
            "id": completion_id,
            "object": "chat.completion",
            "created": created,
            "model": model_name,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": final_response,
                    },
                    "finish_reason": finish_reason,
                }
            ],
            "usage": {
                "prompt_tokens": usage.get("input_tokens", 0),
                "completion_tokens": usage.get("output_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0),
            },
        }
        if is_partial or is_failed or not completed:
            response_data["hermes"] = {
                "completed": completed,
                "partial": is_partial,
                "failed": is_failed,
                "error": err_msg,
                "error_code": "output_truncated" if finish_reason == "length" else "agent_error",
            }
            response_headers["X-Hermes-Completed"] = "false"
            response_headers["X-Hermes-Partial"] = "true" if is_partial else "false"
            if err_msg:
                response_headers["X-Hermes-Error"] = _redact_api_error_text(err_msg, limit=200)

        return web.json_response(response_data, headers=response_headers)

    async def _write_sse_chat_completion(
        self, request: "web.Request", completion_id: str, model: str,
        created: int, stream_q, agent_task, agent_ref=None, session_id: str = None,
        gateway_session_key: str = None,
    ) -> "web.StreamResponse":
        """Write real streaming SSE from agent's stream_delta_callback queue.

        If the client disconnects mid-stream (network drop, browser tab close),
        the agent is interrupted via ``agent.interrupt()`` so it stops making
        LLM API calls, and the asyncio task wrapper is cancelled.
        """
        sse_headers = {
            "Content-Type": "text/event-stream",
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        }
        # CORS middleware can't inject headers into StreamResponse after
        # prepare() flushes them, so resolve CORS headers up front.
        origin = request.headers.get("Origin", "")
        cors = self._cors_headers_for_origin(origin) if origin else None
        if cors:
            sse_headers.update(cors)
        if session_id:
            sse_headers["X-Hermes-Session-Id"] = session_id
        if gateway_session_key:
            sse_headers["X-Hermes-Session-Key"] = gateway_session_key
        response = web.StreamResponse(status=200, headers=sse_headers)
        await response.prepare(request)

        try:
            last_activity = time.monotonic()

            # Role chunk
            role_chunk = {
                "id": completion_id, "object": "chat.completion.chunk",
                "created": created, "model": model,
                "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
            }
            await response.write(_sse_frame(role_chunk))
            last_activity = time.monotonic()

            # Helper — route a queue item to the correct SSE event.
            async def _emit(item):
                """Write a single queue item to the SSE stream.

                Plain strings are sent as normal ``delta.content`` chunks.
                Tagged tuples ``("__tool_progress__", payload)`` are sent
                as a custom ``event: hermes.tool.progress`` SSE event so
                frontends can display them without storing the markers in
                conversation history.  See #6972 for the original event,
                #16588 for the ``toolCallId``/``status`` lifecycle fields.
                """
                if isinstance(item, tuple) and len(item) == 2 and item[0] == "__tool_progress__":
                    await response.write(_sse_frame(item[1], event="hermes.tool.progress"))
                else:
                    content_chunk = {
                        "id": completion_id, "object": "chat.completion.chunk",
                        "created": created, "model": model,
                        "choices": [{"index": 0, "delta": {"content": item}, "finish_reason": None}],
                    }
                    await response.write(_sse_frame(content_chunk))
                return time.monotonic()

            # Stream content chunks as they arrive from the agent. Woken
            # directly by put_threadsafe's call_soon_threadsafe — no
            # executor hop, no poll-interval latency (see
            # ThreadSafeAsyncQueue's docstring).
            while True:
                try:
                    delta = await asyncio.wait_for(stream_q.get(), timeout=0.5)
                except asyncio.TimeoutError:
                    if agent_task.done():
                        # Drain any remaining items
                        while True:
                            try:
                                delta = stream_q.get_nowait()
                                if delta is None:
                                    break
                                last_activity = await _emit(delta)
                            except asyncio.QueueEmpty:
                                break
                        break
                    if time.monotonic() - last_activity >= CHAT_COMPLETIONS_SSE_KEEPALIVE_SECONDS:
                        await response.write(b": keepalive\n\n")
                        last_activity = time.monotonic()
                    continue

                if delta is None:  # End of stream sentinel
                    break

                last_activity = await _emit(delta)

            # Get usage from completed agent. The agent can fail two ways
            # after the content queue terminates cleanly: (1) ``agent_task``
            # raises, or (2) it returns a ``result`` dict flagged
            # failed/partial/incomplete. Both previously fell through to a
            # ``finish_reason: "stop"`` chunk, so OpenAI-compatible clients
            # saw a fake success. Surface either as a non-"stop" finish so
            # the failure is detectable — mirroring the non-streaming path's
            # decision logic (see the finish_reason block above).
            usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
            result = None
            agent_error = None
            try:
                result, agent_usage = await agent_task
                usage = agent_usage or usage
            except Exception as exc:
                agent_error = exc
                logger.error(
                    "Agent task %s failed during SSE streaming: %s", completion_id, exc
                )

            # Inspect the result dict for a flagged (non-exception) failure.
            is_partial = bool(result.get("partial")) if isinstance(result, dict) else False
            is_failed = bool(result.get("failed")) if isinstance(result, dict) else False
            completed = bool(result.get("completed", True)) if isinstance(result, dict) else True
            err_msg = result.get("error") if isinstance(result, dict) else None
            if agent_error is not None:
                is_failed = True
                err_msg = err_msg or str(agent_error)

            # Decide finish_reason, matching the non-streaming logic: "length"
            # for truncation, "error" for failure, "stop" for normal completion.
            if is_partial and err_msg and "truncat" in err_msg.lower():
                finish_reason = "length"
            elif agent_error is not None or is_failed or (not completed and err_msg):
                finish_reason = "error"
            else:
                finish_reason = "stop"

            # Finish chunk
            finish_chunk = {
                "id": completion_id, "object": "chat.completion.chunk",
                "created": created, "model": model,
                "choices": [{"index": 0, "delta": {}, "finish_reason": finish_reason}],
                "usage": {
                    "prompt_tokens": usage.get("input_tokens", 0),
                    "completion_tokens": usage.get("output_tokens", 0),
                    "total_tokens": usage.get("total_tokens", 0),
                },
            }
            if finish_reason != "stop":
                finish_chunk["choices"][0]["delta"] = {}
                if err_msg:
                    finish_chunk["error"] = {
                        "message": err_msg,
                        "type": type(agent_error).__name__ if agent_error else "agent_error",
                    }
                finish_chunk["hermes"] = {
                    "completed": completed,
                    "partial": is_partial,
                    "failed": is_failed,
                    "error": err_msg,
                    "error_code": "output_truncated" if finish_reason == "length" else "agent_error",
                }
            await response.write(_sse_frame(finish_chunk))
            await response.write(b"data: [DONE]\n\n")
        except (ConnectionResetError, ConnectionAbortedError, BrokenPipeError, OSError):
            # Client disconnected mid-stream.  Interrupt the agent so it
            # stops making LLM API calls at the next loop iteration, then
            # cancel the asyncio task wrapper.
            agent = agent_ref[0] if agent_ref else None
            if agent is not None:
                try:
                    request_hard_interrupt(agent, "SSE client disconnected")
                except Exception:
                    pass
                _reap_disconnected_agent_processes(agent)
            if not agent_task.done():
                agent_task.cancel()
                try:
                    await agent_task
                except (asyncio.CancelledError, Exception):
                    pass
            logger.info("SSE client disconnected; interrupted agent task %s", completion_id)
        except Exception as _exc:
            # Agent crashed mid-stream.  Try to emit an error chunk
            # so the client gets a proper response instead of a
            # TransferEncodingError from incomplete chunked encoding.
            import traceback as _tb
            logger.error("Agent crashed mid-stream for %s: %s", completion_id, _tb.format_exc()[:300])
            try:
                error_chunk = {
                    "id": completion_id, "object": "chat.completion.chunk",
                    "created": created, "model": model,
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "error"}],
                }
                await response.write(_sse_frame(error_chunk))
                await response.write(b"data: [DONE]\n\n")
            except Exception:
                pass

        return response

    async def _write_sse_responses(
        self,
        request: "web.Request",
        response_id: str,
        model: str,
        created_at: int,
        stream_q,
        agent_task,
        agent_ref,
        conversation_history: List[Dict[str, str]],
        user_message: str,
        instructions: Optional[str],
        conversation: Optional[str],
        store: bool,
        session_id: str,
        gateway_session_key: Optional[str] = None,
        store_profile: Optional[str] = None,
        conversation_reservation_id: Optional[str] = None,
        expected_previous_response_id: Any = _UNSTATED,
        owner_publish: Optional[Any] = None,
    ) -> "web.StreamResponse":
        """Write an SSE stream for POST /v1/responses (OpenAI Responses API).

        Emits spec-compliant event types as the agent runs:

        - ``response.created`` — initial envelope (status=in_progress)
        - ``response.output_text.delta`` / ``response.output_text.done`` —
          streamed assistant text
        - ``response.output_item.added`` / ``response.output_item.done``
          with ``item.type == "function_call"`` — when the agent invokes a
          tool (both events fire; the ``done`` event carries the finalized
          ``arguments`` string)
        - ``response.output_item.added`` with
          ``item.type == "function_call_output"`` — tool result with
          ``{call_id, output, status}``
        - ``response.completed`` — terminal event carrying the full
          response object with all output items + usage (same payload
          shape as the non-streaming path for parity)
        - ``response.failed`` — terminal event on agent error

        If the client disconnects mid-stream, ``agent.interrupt()`` is
        called so the agent stops issuing upstream LLM calls, then the
        asyncio task is cancelled.  When ``store=True`` an initial
        ``in_progress`` snapshot is persisted immediately after
        ``response.created`` and disconnects update it to an
        ``incomplete`` snapshot so GET /v1/responses/{id} and
        ``previous_response_id`` chaining still have something to
        recover from.

        ``owner_publish`` commits an owner turn's terminal snapshot, its
        conversation head, its reservation release and its durable replay
        record as ONE transaction, and returns whether the head was still this
        turn's to take. That single commit is the moment the turn really
        happened, so it is the moment — and the only one — at which an exact
        retry may replay it instead of planning a second one. Nothing else
        about the stream changes, and a non-owner stream never passes one.
        """
        sse_headers = {
            "Content-Type": "text/event-stream",
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        }
        origin = request.headers.get("Origin", "")
        cors = self._cors_headers_for_origin(origin) if origin else None
        if cors:
            sse_headers.update(cors)
        if session_id:
            sse_headers["X-Hermes-Session-Id"] = session_id
        if gateway_session_key:
            sse_headers["X-Hermes-Session-Key"] = gateway_session_key
        response = web.StreamResponse(status=200, headers=sse_headers)
        await response.prepare(request)

        # State accumulated during the stream
        final_text_parts: List[str] = []
        # Track open function_call items by name so we can emit a matching
        # ``done`` event when the tool completes.  Order preserved.
        pending_tool_calls: List[Dict[str, Any]] = []
        # Output items we've emitted so far (used to build the terminal
        # response.completed payload).  Kept in the order they appeared.
        emitted_items: List[Dict[str, Any]] = []
        # Monotonic counter for output_index (spec requires it).
        output_index = 0
        # Monotonic counter for call_id generation if the agent doesn't
        # provide one (it doesn't, from tool_progress_callback).
        call_counter = 0
        # Canonical Responses SSE events include a monotonically increasing
        # sequence_number. Add it server-side for every emitted event so
        # clients that validate the OpenAI event schema can parse our stream.
        sequence_number = 0
        # Track the assistant message item id + content index for text
        # delta events — the spec ties deltas to a specific item.
        message_item_id = f"msg_{uuid.uuid4().hex[:24]}"
        message_output_index: Optional[int] = None
        message_opened = False

        async def _write_event(event_type: str, data: Dict[str, Any]) -> None:
            nonlocal sequence_number
            if "sequence_number" not in data:
                data["sequence_number"] = sequence_number
            sequence_number += 1
            await response.write(_sse_frame(data, event=event_type))

        def _envelope(status: str) -> Dict[str, Any]:
            env: Dict[str, Any] = {
                "id": response_id,
                "object": "response",
                "status": status,
                "created_at": created_at,
                "model": model,
            }
            return env

        final_response_text = ""
        agent_error: Optional[str] = None
        usage: Dict[str, int] = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
        terminal_snapshot_persisted = False

        def _persist_response_snapshot(
            response_env: Dict[str, Any],
            *,
            conversation_history_snapshot: Optional[List[Dict[str, Any]]] = None,
            session_id_snapshot: Optional[str] = None,
        ) -> None:
            if not store:
                return
            if conversation_history_snapshot is None:
                conversation_history_snapshot = list(conversation_history)
                conversation_history_snapshot.append({"role": "user", "content": user_message})
            status = response_env.get("status")
            if (
                conversation
                and conversation_reservation_id is not None
                and owner_publish is not None
                and status == "completed"
            ):
                # ONE commit for the response, the head, the reservation and the
                # replay record — the streaming path has the same contract as
                # the standard and background ones.
                if not owner_publish(
                    response_env,
                    conversation_history_snapshot,
                    session_id_snapshot or session_id,
                ):
                    raise RuntimeError("owner conversation reservation changed")
                return
            self._response_store.put(response_id, {
                "response": response_env,
                "conversation_history": conversation_history_snapshot,
                "instructions": instructions,
                "session_id": session_id_snapshot or session_id,
            }, profile=store_profile)
            if conversation:
                if conversation_reservation_id is not None:
                    if status in {"failed", "incomplete"}:
                        self._response_store.release_owner_conversation_reservation(
                            str(store_profile), conversation, conversation_reservation_id,
                        )
                else:
                    self._response_store.set_conversation(
                        conversation,
                        response_id,
                        owner_proposal=(
                            status == "completed"
                            and _owner_history_has_actionable_final_proposal(
                                conversation_history_snapshot
                            )
                        ),
                        profile=store_profile,
                    )

        def _persist_incomplete_if_needed() -> None:
            """Persist an ``incomplete`` snapshot if no terminal one was written.

            Called from both the client-disconnect (``ConnectionResetError``)
            and server-cancellation (``asyncio.CancelledError``) paths so
            GET /v1/responses/{id} and ``previous_response_id`` chaining keep
            working after abrupt stream termination.
            """
            if not store or terminal_snapshot_persisted:
                return
            incomplete_text = "".join(final_text_parts) or final_response_text
            incomplete_items: List[Dict[str, Any]] = list(emitted_items)
            if incomplete_text:
                incomplete_items.append({
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": incomplete_text}],
                })
            incomplete_env = _envelope("incomplete")
            incomplete_env["output"] = incomplete_items
            incomplete_env["usage"] = {
                "input_tokens": usage.get("input_tokens", 0),
                "output_tokens": usage.get("output_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0),
            }
            incomplete_history = list(conversation_history)
            incomplete_history.append({"role": "user", "content": user_message})
            if incomplete_text:
                incomplete_history.append({"role": "assistant", "content": incomplete_text})
            _persist_response_snapshot(
                incomplete_env,
                conversation_history_snapshot=incomplete_history,
            )

        try:
            # response.created — initial envelope, status=in_progress
            created_env = _envelope("in_progress")
            created_env["output"] = []
            await _write_event("response.created", {
                "type": "response.created",
                "response": created_env,
            })
            _persist_response_snapshot(created_env)
            last_activity = time.monotonic()

            async def _open_message_item() -> None:
                """Emit response.output_item.added for the assistant message
                the first time any text delta arrives."""
                nonlocal message_opened, message_output_index, output_index
                if message_opened:
                    return
                message_opened = True
                message_output_index = output_index
                output_index += 1
                item = {
                    "id": message_item_id,
                    "type": "message",
                    "status": "in_progress",
                    "role": "assistant",
                    "content": [],
                }
                await _write_event("response.output_item.added", {
                    "type": "response.output_item.added",
                    "output_index": message_output_index,
                    "item": item,
                })

            async def _emit_text_delta(delta_text: str) -> None:
                await _open_message_item()
                final_text_parts.append(delta_text)
                await _write_event("response.output_text.delta", {
                    "type": "response.output_text.delta",
                    "item_id": message_item_id,
                    "output_index": message_output_index,
                    "content_index": 0,
                    "delta": delta_text,
                    "logprobs": [],
                })

            async def _emit_tool_started(payload: Dict[str, Any]) -> str:
                """Emit response.output_item.added for a function_call.

                Returns the call_id so the matching completion event can
                reference it.  Prefer the real ``tool_call_id`` from the
                agent when available; fall back to a generated call id for
                safety in tests or older code paths.
                """
                nonlocal output_index, call_counter
                call_counter += 1
                call_id = payload.get("tool_call_id") or f"call_{response_id[5:]}_{call_counter}"
                args = payload.get("arguments", {})
                if isinstance(args, dict):
                    arguments_str = json.dumps(args)
                else:
                    arguments_str = str(args)
                item = {
                    "id": f"fc_{uuid.uuid4().hex[:24]}",
                    "type": "function_call",
                    "status": "in_progress",
                    "name": payload.get("name", ""),
                    "call_id": call_id,
                    "arguments": arguments_str,
                }
                idx = output_index
                output_index += 1
                pending_tool_calls.append({
                    "call_id": call_id,
                    "name": payload.get("name", ""),
                    "arguments": arguments_str,
                    "item_id": item["id"],
                    "output_index": idx,
                })
                emitted_items.append({
                    "type": "function_call",
                    "name": payload.get("name", ""),
                    "arguments": arguments_str,
                    "call_id": call_id,
                })
                await _write_event("response.output_item.added", {
                    "type": "response.output_item.added",
                    "output_index": idx,
                    "item": item,
                })
                return call_id

            async def _emit_tool_completed(payload: Dict[str, Any]) -> None:
                """Emit response.output_item.done (function_call) followed
                by response.output_item.added (function_call_output)."""
                nonlocal output_index
                call_id = payload.get("tool_call_id")
                result = payload.get("result", "")
                pending = None
                if call_id:
                    for i, p in enumerate(pending_tool_calls):
                        if p["call_id"] == call_id:
                            pending = pending_tool_calls.pop(i)
                            break
                if pending is None:
                    # Completion without a matching start — skip to avoid
                    # emitting orphaned done events.
                    return

                # function_call done
                done_item = {
                    "id": pending["item_id"],
                    "type": "function_call",
                    "status": "completed",
                    "name": pending["name"],
                    "call_id": pending["call_id"],
                    "arguments": pending["arguments"],
                }
                await _write_event("response.output_item.done", {
                    "type": "response.output_item.done",
                    "output_index": pending["output_index"],
                    "item": done_item,
                })

                # function_call_output added (result)
                result_str = result if isinstance(result, str) else json.dumps(result)
                output_parts = [{"type": "input_text", "text": result_str}]
                output_item = {
                    "id": f"fco_{uuid.uuid4().hex[:24]}",
                    "type": "function_call_output",
                    "call_id": pending["call_id"],
                    "output": output_parts,
                    "status": "completed",
                }
                idx = output_index
                output_index += 1
                emitted_items.append({
                    "type": "function_call_output",
                    "call_id": pending["call_id"],
                    "output": output_parts,
                })
                await _write_event("response.output_item.added", {
                    "type": "response.output_item.added",
                    "output_index": idx,
                    "item": output_item,
                })
                await _write_event("response.output_item.done", {
                    "type": "response.output_item.done",
                    "output_index": idx,
                    "item": output_item,
                })

            # Main drain loop — thread-safe queue fed by agent callbacks.
            async def _dispatch(it) -> None:
                """Route a queue item to the correct SSE emitter.

                Plain strings are text deltas — they are batched (50ms)
                to reduce Open WebUI re-render storms.  Tagged tuples
                with ``__tool_started__`` / ``__tool_completed__``
                prefixes are tool lifecycle events and flush the buffer
                before emitting.
                """
                nonlocal _batch_timer
                if isinstance(it, tuple) and len(it) == 2 and isinstance(it[0], str):
                    tag, payload = it
                    # Flush batched text before tool events
                    if _batch_buf:
                        await _flush_batch()
                    if tag == "__tool_started__":
                        await _emit_tool_started(payload)
                    elif tag == "__tool_completed__":
                        await _emit_tool_completed(payload)
                elif isinstance(it, str):
                    # Batch text deltas — append to buffer, flush on timer
                    _batch_buf.append(it)
                    if _batch_timer is None:
                        _batch_timer = asyncio.create_task(_batch_flush_after(0.05))
                # Other types are silently dropped.

            # ── Batching state ──
            _batch_buf: List[str] = []
            _batch_timer: Optional[asyncio.Task] = None
            _batch_lock = asyncio.Lock()

            async def _batch_flush_after(delay: float) -> None:
                """Wait delay seconds, then flush accumulated text deltas."""
                try:
                    await asyncio.sleep(delay)
                except asyncio.CancelledError:
                    return
                # Clear timer reference BEFORE flush so new deltas
                # can start a fresh timer while we emit
                nonlocal _batch_buf, _batch_timer
                _batch_timer = None
                await _flush_batch()

            async def _flush_batch() -> None:
                """Emit a single SSE delta for all accumulated text."""
                nonlocal _batch_buf
                async with _batch_lock:
                    if _batch_buf:
                        combined = "".join(_batch_buf)
                        _batch_buf = []
                        await _emit_text_delta(combined)

            while True:
                try:
                    item = await asyncio.wait_for(stream_q.get(), timeout=0.5)
                except asyncio.TimeoutError:
                    if agent_task.done():
                        # Drain remaining
                        while True:
                            try:
                                item = stream_q.get_nowait()
                                if item is None:
                                    break
                                await _dispatch(item)
                                last_activity = time.monotonic()
                            except asyncio.QueueEmpty:
                                break
                        break
                    if time.monotonic() - last_activity >= CHAT_COMPLETIONS_SSE_KEEPALIVE_SECONDS:
                        await response.write(b": keepalive\n\n")
                        last_activity = time.monotonic()
                    continue

                if item is None:  # EOS sentinel
                    # Cancel pending timer and flush remaining batched text
                    if _batch_timer and not _batch_timer.done():
                        _batch_timer.cancel()
                        _batch_timer = None
                    if _batch_buf:
                        await _flush_batch()
                    break

                await _dispatch(item)
                last_activity = time.monotonic()

            # Flush any final batched text before processing result
            if _batch_buf:
                await _flush_batch()

            # Pick up agent result + usage from the completed task
            try:
                result, agent_usage = await agent_task
                usage = agent_usage or usage
                # If the agent produced a final_response but no text
                # deltas were streamed (e.g. some providers only emit
                # the full response at the end), emit a single fallback
                # delta so Responses clients still receive a live text part.
                agent_final = result.get("final_response", "") if isinstance(result, dict) else ""
                if agent_final and not final_text_parts:
                    await _emit_text_delta(agent_final)
                if agent_final and not final_response_text:
                    final_response_text = agent_final
                if isinstance(result, dict) and result.get("error") and not final_response_text:
                    agent_error = _redact_api_error_text(result["error"])
            except Exception as e:  # noqa: BLE001
                logger.error("Error running agent for streaming responses: %s", e, exc_info=True)
                agent_error = _redact_api_error_text(e)

            # Close the message item if it was opened
            final_response_text = "".join(final_text_parts) or final_response_text
            if message_opened:
                await _write_event("response.output_text.done", {
                    "type": "response.output_text.done",
                    "item_id": message_item_id,
                    "output_index": message_output_index,
                    "content_index": 0,
                    "text": final_response_text,
                    "logprobs": [],
                })
                msg_done_item = {
                    "id": message_item_id,
                    "type": "message",
                    "status": "completed",
                    "role": "assistant",
                    "content": [
                        {"type": "output_text", "text": final_response_text}
                    ],
                }
                await _write_event("response.output_item.done", {
                    "type": "response.output_item.done",
                    "output_index": message_output_index,
                    "item": msg_done_item,
                })

            # Always append a final message item in the completed
            # response envelope so clients that only parse the terminal
            # payload still see the assistant text.  This mirrors the
            # shape produced by _extract_output_items in the batch path.
            final_items: List[Dict[str, Any]] = list(emitted_items)

            # Trim large content from tool call arguments to keep the
            # response.completed event under ~100KB.  Clients already
            # received full details via incremental events.
            for _item in final_items:
                if _item.get("type") == "function_call":
                    try:
                        _args = json.loads(_item.get("arguments", "{}")) if isinstance(_item.get("arguments"), str) else _item.get("arguments", {})
                        if isinstance(_args, dict):
                            for _k in ("content", "query", "pattern", "old_string", "new_string"):
                                if isinstance(_args.get(_k), str) and len(_args[_k]) > 500:
                                    _args[_k] = "[" + str(len(_args[_k])) + " chars — truncated for response.completed]"
                            _item["arguments"] = json.dumps(_args)
                    except Exception:
                        pass
                elif _item.get("type") == "function_call_output":
                    _output = _item.get("output", [])
                    if isinstance(_output, list) and _output:
                        _first = _output[0]
                        if isinstance(_first, dict) and _first.get("type") == "input_text":
                            _text = _first.get("text", "")
                            if len(_text) > 1000:
                                _first["text"] = _text[:500] + "...[" + str(len(_text) - 500) + " more chars]"
                                _item["output"] = [_first]

            final_items.append({
                "type": "message",
                "role": "assistant",
                "content": [
                    {"type": "output_text", "text": final_response_text or (_redact_api_error_text(agent_error) if agent_error else "")}
                ],
            })

            if agent_error:
                failed_env = _envelope("failed")
                failed_env["output"] = final_items
                failed_env["error"] = {"message": _redact_api_error_text(agent_error), "type": "server_error"}
                failed_env["usage"] = {
                    "input_tokens": usage.get("input_tokens", 0),
                    "output_tokens": usage.get("output_tokens", 0),
                    "total_tokens": usage.get("total_tokens", 0),
                }
                _failed_history = list(conversation_history)
                _failed_history.append({"role": "user", "content": user_message})
                if final_response_text or agent_error:
                    _failed_history.append({
                        "role": "assistant",
                        "content": final_response_text or _redact_api_error_text(agent_error),
                    })
                _persist_response_snapshot(
                    failed_env,
                    conversation_history_snapshot=_failed_history,
                )
                terminal_snapshot_persisted = True
                await _write_event("response.failed", {
                    "type": "response.failed",
                    "response": failed_env,
                })
            else:
                completed_env = _envelope("completed")
                completed_env["output"] = final_items
                completed_env["usage"] = {
                    "input_tokens": usage.get("input_tokens", 0),
                    "output_tokens": usage.get("output_tokens", 0),
                    "total_tokens": usage.get("total_tokens", 0),
                }
                full_history = self._build_response_conversation_history(
                    conversation_history,
                    user_message,
                    result,
                    final_response_text,
                )
                # Compression-aware transcript substitution happens inside
                # _build_response_conversation_history (result["_compressed"]);
                # here we only propagate a compression-rotated session_id so
                # previous_response_id chaining resumes the child session.
                _result_sid = result.get("session_id") if isinstance(result, dict) else None
                _persist_response_snapshot(
                    completed_env,
                    conversation_history_snapshot=full_history,
                    session_id_snapshot=_result_sid if isinstance(_result_sid, str) and _result_sid else None,
                )
                terminal_snapshot_persisted = True
                await _write_event("response.completed", {
                    "type": "response.completed",
                    "response": completed_env,
                })

        except (ConnectionResetError, ConnectionAbortedError, BrokenPipeError, OSError):
            _persist_incomplete_if_needed()
            # Client disconnected — interrupt the agent so it stops
            # making upstream LLM calls, then cancel the task.
            agent = agent_ref[0] if agent_ref else None
            if agent is not None:
                try:
                    request_hard_interrupt(agent, "SSE client disconnected")
                except Exception:
                    pass
                _reap_disconnected_agent_processes(agent)
            if not agent_task.done():
                agent_task.cancel()
                try:
                    await agent_task
                except (asyncio.CancelledError, Exception):
                    pass
            logger.info("SSE client disconnected; interrupted agent task %s", response_id)
        except asyncio.CancelledError:
            # Server-side cancellation (e.g. shutdown, request timeout) —
            # persist an incomplete snapshot so GET /v1/responses/{id} and
            # previous_response_id chaining still work, then re-raise so the
            # runtime's cancellation semantics are respected.
            _persist_incomplete_if_needed()
            agent = agent_ref[0] if agent_ref else None
            if agent is not None:
                try:
                    request_hard_interrupt(agent, "SSE task cancelled")
                except Exception:
                    pass
                # Same abandonment as a client disconnect: the run will never
                # be resumed, so reap the background processes it created
                # (#76115). Epoch-gated; no-op when the turn already
                # finished and cleared its markers.
                _reap_disconnected_agent_processes(
                    agent, source="api_server_sse_cancelled"
                )
            if not agent_task.done():
                agent_task.cancel()
            logger.info("SSE task cancelled; persisted incomplete snapshot for %s", response_id)
            raise
        except Exception as _exc:
            # Agent crashed with an unhandled error (e.g. model API error like
            # BadRequestError, AuthenticationError).  Emit a response.failed
            # event and properly terminate the SSE stream so the client doesn't
            # get a TransferEncodingError from incomplete chunked encoding.
            import traceback as _tb
            _persist_incomplete_if_needed()
            agent_error = _redact_api_error_text(_tb.format_exc())
            try:
                failed_env = _envelope("failed")
                failed_env["output"] = list(emitted_items)
                failed_env["error"] = {"message": _redact_api_error_text(_exc, limit=500), "type": "server_error"}
                failed_env["usage"] = {
                    "input_tokens": usage.get("input_tokens", 0),
                    "output_tokens": usage.get("output_tokens", 0),
                    "total_tokens": usage.get("total_tokens", 0),
                }
                await _write_event("response.failed", {
                    "type": "response.failed",
                    "response": failed_env,
                })
            except Exception:
                pass
            logger.error("Agent crashed mid-stream for %s: %s", response_id, str(agent_error)[:300])

        return response

    @staticmethod
    def _owner_idempotency_response(
        stored: "tuple[str, Optional[dict], Optional[str]]",
        gateway_session_key: Optional[str],
    ) -> "Optional[web.Response]":
        """Turn a durable owner idempotency verdict into its HTTP answer.

        ``None`` means "this really is a new turn; carry on".
        """
        outcome, replay, session = stored
        if outcome == "conflict":
            return web.json_response(
                _openai_error(
                    "Idempotency-Key was already used for a different request",
                    code="idempotency_conflict",
                ),
                status=409,
            )
        if outcome == "replay" and replay is not None:
            headers = {}
            if session:
                headers["X-Hermes-Session-Id"] = session
            if gateway_session_key:
                headers["X-Hermes-Session-Key"] = gateway_session_key
            return web.json_response(replay, headers=headers)
        if outcome == "incomplete":
            return web.json_response(
                _openai_error(
                    "That request is still being prepared. Reload and try again.",
                    code="owner_response_incomplete",
                ),
                status=409,
            )
        return None

    @_admit_api_agent_request
    async def _handle_responses(self, request: "web.Request") -> "web.Response":
        """POST /v1/responses — OpenAI Responses API format."""
        # Bound total in-flight agent runs (configurable; #7483).
        limited = self._concurrency_limited_response()
        if limited is not None:
            return limited

        # Long-term memory scope header (see chat_completions for details).
        gateway_session_key, key_err = self._parse_session_key_header(request)
        if key_err is not None:
            return key_err

        # Parse request body
        try:
            body = await request.json()
        except (json.JSONDecodeError, Exception):
            return web.json_response(
                {"error": {"message": "Invalid JSON in request body", "type": "invalid_request_error"}},
                status=400,
            )

        raw_input = body.get("input")
        if raw_input is None:
            return web.json_response(_openai_error("Missing 'input' field"), status=400)

        instructions = body.get("instructions")
        previous_response_id = body.get("previous_response_id")
        conversation = body.get("conversation")
        store = _coerce_request_bool(body.get("store"), default=True)
        background = _coerce_request_bool(body.get("background"), default=False)
        stream = _coerce_request_bool(body.get("stream"), default=False)

        if background and not store:
            return web.json_response(
                _openai_error("'background' requires 'store' to be true"),
                status=400,
            )
        if background and stream:
            return web.json_response(
                _openai_error("'background' cannot be combined with 'stream'"),
                status=400,
            )

        response_profile = _active_owner_profile()
        is_owner_conversation = (
            isinstance(conversation, str)
            and _OWNER_CONVERSATION_RE.fullmatch(conversation) is not None
        )
        if is_owner_conversation and not store:
            return web.json_response(
                _openai_error("Owner conversations require durable storage"),
                status=400,
            )
        # An owner conversation's history is the durable head, and only the
        # durable head. A caller-supplied transcript would let a direct caller
        # publish a fabricated or truncated conversation as the new authority,
        # and ``previous_response_id`` would let it plan against a turn the
        # conversation has moved past. Both are refused rather than ignored, so
        # a caller that sends one is told its request was not carried out.
        if is_owner_conversation and "conversation_history" in body:
            return web.json_response(
                _openai_error(
                    "Owner conversations derive history from the conversation "
                    "itself and cannot accept 'conversation_history'",
                    param="conversation_history",
                ),
                status=400,
            )
        if is_owner_conversation and body.get("previous_response_id") is not None:
            return web.json_response(
                _openai_error(
                    "Owner conversations cannot accept 'previous_response_id'",
                    param="previous_response_id",
                ),
                status=400,
            )

        # The exact turn this request was planned against, stated by the
        # caller. Machine-only: it never reaches the model and is never
        # projected to the owner. Present (including an explicit ``null`` for
        # "this conversation has no turn yet") means the caller is asserting
        # the predecessor, and the assertion is compared atomically both when
        # this turn reserves the conversation and when it maps its result.
        # Absent means no assertion, which is what every non-owner client and
        # every older caller sends.
        expected_predecessor_stated = "expected_previous_response_id" in body
        expected_predecessor = body.get("expected_previous_response_id")
        if expected_predecessor_stated and not (
            expected_predecessor is None
            or (
                isinstance(expected_predecessor, str)
                and _OWNER_RESPONSE_RE.fullmatch(expected_predecessor) is not None
            )
        ):
            return web.json_response(
                _openai_error("Invalid expected_previous_response_id"), status=400,
            )
        # Both contracts are MANDATORY on an owner conversation, including an
        # explicit ``null`` for the first turn. Optional, they let an older or
        # direct caller append to whatever the latest head happens to be and
        # duplicate an already-answered turn after a restart; there is nothing
        # to compare and nothing to replay.
        if is_owner_conversation and not expected_predecessor_stated:
            return web.json_response(
                _openai_error(
                    "Owner conversations must state "
                    "'expected_previous_response_id' (null for the first turn)",
                    param="expected_previous_response_id",
                ),
                status=400,
            )

        idempotency_key = request.headers.get("Idempotency-Key")
        if is_owner_conversation and idempotency_key is None:
            return web.json_response(
                _openai_error(
                    "Owner conversations require an Idempotency-Key header",
                    param="Idempotency-Key",
                ),
                status=400,
            )
        response_idempotency_fingerprint: Optional[str] = None
        response_idempotency_cache_key: Optional[str] = None
        if idempotency_key is not None:
            if re.fullmatch(r"[A-Za-z0-9._:-]{1,200}", idempotency_key) is None:
                return web.json_response(
                    _openai_error("Invalid Idempotency-Key"), status=400,
                )
            response_scope = hashlib.sha256(
                (gateway_session_key or "").encode("utf-8")
            ).hexdigest()
            fingerprint_body = dict(body)
            fingerprint_body["_session_scope"] = response_scope
            response_idempotency_fingerprint = _make_request_fingerprint(
                fingerprint_body, keys=sorted(fingerprint_body),
            )
            response_idempotency_cache_key = (
                f"responses:{response_profile}:{response_scope}:{idempotency_key}"
            )
            try:
                cached = await _idem_cache.get_existing(
                    response_idempotency_cache_key,
                    response_idempotency_fingerprint,
                )
            except _IdempotencyConflict:
                return web.json_response(
                    _openai_error(
                        "Idempotency-Key was already used for a different request",
                        code="idempotency_conflict",
                    ),
                    status=409,
                )
            if cached is not None:
                cached_response, cached_session_id = cached
                cached_headers = {}
                if cached_session_id:
                    cached_headers["X-Hermes-Session-Id"] = cached_session_id
                if gateway_session_key:
                    cached_headers["X-Hermes-Session-Key"] = gateway_session_key
                return web.json_response(cached_response, headers=cached_headers)
            if is_owner_conversation:
                # A durable replay is answered before anything treats this as a
                # new turn — including the predecessor comparison below, which
                # this conversation has legitimately moved past by virtue of
                # the very reply being replayed.
                replayed = self._owner_idempotency_response(
                    self._response_store.lookup_owner_response(
                        response_profile,
                        response_scope,
                        idempotency_key,
                        response_idempotency_fingerprint,
                        str(conversation),
                    ),
                    gateway_session_key,
                )
                if replayed is not None:
                    return replayed

        # conversation and previous_response_id are mutually exclusive
        if conversation and previous_response_id:
            return web.json_response(_openai_error("Cannot use both 'conversation' and 'previous_response_id'"), status=400)

        # Resolve conversation name to latest response_id
        if conversation:
            previous_response_id = self._response_store.get_conversation(
                conversation, profile=response_profile,
            )
            # No error if conversation doesn't exist yet — it's a new conversation
            if (
                expected_predecessor_stated
                and previous_response_id != expected_predecessor
            ):
                # The conversation moved on since the caller read it. Planning
                # this turn against the history it actually has would silently
                # answer a different question than the one that was asked, so
                # it is refused before any model is run.
                return web.json_response(
                    _openai_error(
                        "The conversation moved on before this request was planned",
                        code="owner_conversation_stale",
                    ),
                    status=409,
                )

        # Normalize input to message list
        input_messages: List[Dict[str, Any]] = []
        if isinstance(raw_input, str):
            input_messages = [{"role": "user", "content": raw_input}]
        elif isinstance(raw_input, list):
            for idx, item in enumerate(raw_input):
                if isinstance(item, str):
                    input_messages.append({"role": "user", "content": item})
                elif isinstance(item, dict):
                    role = item.get("role", "user")
                    try:
                        content = _normalize_multimodal_content(item.get("content", ""))
                    except ValueError as exc:
                        return _multimodal_validation_error(exc, param=f"input[{idx}].content")
                    input_messages.append({"role": role, "content": content})
        else:
            return web.json_response(_openai_error("'input' must be a string or array"), status=400)

        # Accept explicit conversation_history from the request body.
        # This lets stateless clients supply their own history instead of
        # relying on server-side response chaining via previous_response_id.
        # Precedence: explicit conversation_history > previous_response_id.
        conversation_history: List[Dict[str, Any]] = []
        raw_history = body.get("conversation_history")
        if raw_history:
            if not isinstance(raw_history, list):
                return web.json_response(
                    _openai_error("'conversation_history' must be an array of message objects"),
                    status=400,
                )
            for i, entry in enumerate(raw_history):
                if not isinstance(entry, dict) or "role" not in entry or "content" not in entry:
                    return web.json_response(
                        _openai_error(f"conversation_history[{i}] must have 'role' and 'content' fields"),
                        status=400,
                    )
                try:
                    entry_content = _normalize_multimodal_content(entry["content"])
                except ValueError as exc:
                    return _multimodal_validation_error(exc, param=f"conversation_history[{i}].content")
                conversation_history.append({"role": str(entry["role"]), "content": entry_content})
            if previous_response_id:
                logger.debug("Both conversation_history and previous_response_id provided; using conversation_history")

        stored_session_id = None
        if not conversation_history and previous_response_id:
            stored = self._response_store.get(
                previous_response_id, profile=response_profile,
            )
            if stored is None:
                return web.json_response(_openai_error(f"Previous response not found: {previous_response_id}"), status=404)
            conversation_history = list(stored.get("conversation_history", []))
            stored_session_id = stored.get("session_id")
            # If no instructions provided, carry forward from previous
            if instructions is None:
                instructions = stored.get("instructions")

        # Append new input messages to history (all but the last become history)
        for msg in input_messages[:-1]:
            conversation_history.append(msg)

        # Last input message is the user_message
        user_message: Any = input_messages[-1].get("content", "") if input_messages else ""
        if not _content_has_visible_payload(user_message):
            return web.json_response(_openai_error("No user message found in input"), status=400)

        # Truncation support
        if body.get("truncation") == "auto":
            conversation_history = _auto_truncate_response_history(conversation_history)

        # Reuse session from previous_response_id chain so the dashboard
        # groups the entire conversation under one session entry.
        session_id = stored_session_id or str(uuid.uuid4())

        route = self._resolve_route(body.get("model"))
        agent_overrides = _request_agent_overrides(
            body,
            virtual_model=self._model_name,
            allow_bare_model=self._direct_model_requests,
        )
        selection_error = self._request_route_conflict_error(
            session_id=session_id,
            gateway_session_key=gateway_session_key,
            requested_model=agent_overrides.get("requested_model"),
            requested_provider=agent_overrides.get("requested_provider"),
            route=route,
        )
        if selection_error:
            return web.json_response(_openai_error(selection_error), status=400)
        response_id = f"resp_{uuid.uuid4().hex[:28]}"
        conversation_reservation_id: Optional[str] = None
        owner_response_idempotency: Optional[tuple[str, str]] = None

        def _release_owner_response_reservation() -> None:
            nonlocal owner_response_idempotency
            reserved = owner_response_idempotency
            owner_response_idempotency = None
            if reserved is not None:
                self._response_store.release_owner_response(
                    response_profile, reserved[0], reserved[1], response_id,
                )

        def _publish_owner_turn(
            terminal: dict,
            history_snapshot: List[Dict[str, Any]],
            effective_session_id: Optional[str],
            *,
            release_job: bool = False,
        ) -> bool:
            """Commit this owner turn's whole publication, or take no head.

            The stored response, the conversation head compare-and-swap, this
            turn's reservation release and the durable replay record an exact
            retry answers from all land in ONE transaction. Clearing the
            in-memory reservation afterwards is what makes the release in the
            caller's ``finally`` a no-op, so a turn that really did complete is
            never demoted back to "nothing was minted" and replanned.
            """
            nonlocal owner_response_idempotency
            reserved = owner_response_idempotency
            published = self._response_store.publish_owner_turn(
                profile=response_profile,
                conversation=str(conversation),
                response_id=response_id,
                data={
                    "response": terminal,
                    "conversation_history": history_snapshot,
                    "instructions": instructions,
                    "session_id": effective_session_id,
                },
                owner_proposal=_owner_history_has_actionable_final_proposal(
                    history_snapshot
                ),
                reservation_id=conversation_reservation_id,
                expected_previous_response_id=(
                    expected_predecessor if expected_predecessor_stated
                    else _UNSTATED
                ),
                replay=terminal if reserved is not None else None,
                session_scope=reserved[0] if reserved is not None else None,
                idempotency_key=reserved[1] if reserved is not None else None,
                session_id=effective_session_id,
                release_job=release_job,
            )
            if published:
                owner_response_idempotency = None
            return published

        if is_owner_conversation:
            if (
                idempotency_key is not None
                and response_idempotency_fingerprint is not None
            ):
                # Durable, not process-local: this turn may mint a proposal
                # whose response id later carries approval authority, so an
                # exact retry after a restart or past the in-memory TTL has to
                # replay the first attempt rather than plan a second one. The
                # same classification the early lookup made, taken atomically.
                replayed = self._owner_idempotency_response(
                    self._response_store.reserve_owner_response(
                        response_profile,
                        response_scope,
                        idempotency_key,
                        response_idempotency_fingerprint,
                        str(conversation),
                        response_id,
                    ),
                    gateway_session_key,
                )
                if replayed is not None:
                    return replayed
                owner_response_idempotency = (response_scope, idempotency_key)
            if not self._response_store.reserve_owner_conversation(
                response_profile,
                str(conversation),
                response_id,
                expected_previous_response_id=(
                    expected_predecessor if expected_predecessor_stated else _UNSTATED
                ),
                # Recorded by the same write that takes the fence, so the
                # owner's own request is durable from before the model starts.
                # A caller whose accept response never arrived has no handle to
                # this turn at all; the pending projection is the only thing
                # that can give it back both the words and this response id.
                owner_message=user_message,
            ):
                _release_owner_response_reservation()
                if (
                    response_idempotency_cache_key is not None
                    and response_idempotency_fingerprint is not None
                ):
                    try:
                        cached = await _idem_cache.get_existing(
                            response_idempotency_cache_key,
                            response_idempotency_fingerprint,
                        )
                    except _IdempotencyConflict:
                        cached = None
                    if cached is not None:
                        cached_response, cached_session_id = cached
                        cached_headers = {}
                        if cached_session_id:
                            cached_headers["X-Hermes-Session-Id"] = cached_session_id
                        if gateway_session_key:
                            cached_headers["X-Hermes-Session-Key"] = gateway_session_key
                        return web.json_response(
                            cached_response, headers=cached_headers,
                        )
                return web.json_response(
                    _openai_error(
                        "Owner conversation is busy or closed",
                        code="owner_conversation_locked",
                    ),
                    status=409,
                )
            conversation_reservation_id = response_id
        conversation_agent_ref: list[Any] = [None]
        conversation_reservation_lost = threading.Event()
        reservation_heartbeat_task: "asyncio.Task[Any] | None" = None

        async def _renew_owner_conversation_reservation() -> None:
            try:
                while conversation_reservation_id is not None:
                    await asyncio.sleep(
                        _OWNER_CONVERSATION_RESERVATION_RENEW_SECONDS
                    )
                    if not self._response_store.renew_owner_conversation_reservation(
                        response_profile,
                        str(conversation),
                        conversation_reservation_id,
                    ):
                        conversation_reservation_lost.set()
                        agent = conversation_agent_ref[0]
                        if agent is not None:
                            request_hard_interrupt(
                                agent, "Owner conversation reservation was lost",
                            )
                        return
            except asyncio.CancelledError:
                return
            except Exception:
                conversation_reservation_lost.set()
                agent = conversation_agent_ref[0]
                if agent is not None:
                    try:
                        request_hard_interrupt(
                            agent, "Owner conversation reservation was lost",
                        )
                    except Exception:
                        pass
                logger.exception(
                    "[api_server] owner conversation reservation renewal failed"
                )

        def _stop_reservation_heartbeat() -> None:
            """Stop renewing a fence this turn has finished with."""
            nonlocal reservation_heartbeat_task
            heartbeat = reservation_heartbeat_task
            reservation_heartbeat_task = None
            if heartbeat is not None and not heartbeat.done():
                heartbeat.cancel()

        def _release_owner_conversation_reservation() -> None:
            """Drop this turn's fence, leaving nothing behind.

            For the endings that produced a terminal response and no turn, the
            fence is not dropped here at all: it becomes the record that carries
            the owner's request, inside the same transaction that stores the
            terminal body (see ``store_terminal_owner_response``). Doing it here
            as a second write meant a crash in between lost the request.
            """
            _stop_reservation_heartbeat()
            if conversation_reservation_id is None:
                return
            self._response_store.release_owner_conversation_reservation(
                response_profile,
                str(conversation),
                conversation_reservation_id,
            )

        if conversation_reservation_id is not None:
            reservation_heartbeat_task = asyncio.create_task(
                _renew_owner_conversation_reservation()
            )
            self._background_tasks.add(reservation_heartbeat_task)
            reservation_heartbeat_task.add_done_callback(
                self._background_tasks.discard
            )
        if stream:
            # Streaming branch — emit OpenAI Responses SSE events as the
            # agent runs so frontends can render text deltas and tool
            # calls in real time.  See _write_sse_responses for details.
            _stream_q = ThreadSafeAsyncQueue()

            def _on_delta(delta):
                # None from the agent is a CLI box-close signal, not EOS.
                # Forwarding would kill the SSE stream prematurely; the
                # SSE writer detects completion via agent_task.done().
                # Called from the worker thread running run_conversation —
                # put_threadsafe (not put_nowait) is required here.
                if delta is not None:
                    _stream_q.put_threadsafe(delta)

            def _on_tool_progress(event_type, name, preview, args, **kwargs):
                """Queue non-start tool progress events if needed in future.

                The structured Responses stream uses ``tool_start_callback``
                and ``tool_complete_callback`` for exact call-id correlation,
                so progress events are currently ignored here.
                """
                return

            def _on_tool_start(tool_call_id, function_name, function_args):
                """Queue a started tool for live function_call streaming."""
                _stream_q.put_threadsafe(("__tool_started__", {
                    "tool_call_id": tool_call_id,
                    "name": function_name,
                    "arguments": function_args or {},
                }))

            def _on_tool_complete(tool_call_id, function_name, function_args, function_result):
                """Queue a completed tool result for live function_call_output streaming."""
                _stream_q.put_threadsafe(("__tool_completed__", {
                    "tool_call_id": tool_call_id,
                    "name": function_name,
                    "arguments": function_args or {},
                    "result": function_result,
                }))

            agent_ref = conversation_agent_ref
            agent_task = asyncio.ensure_future(self._run_agent(
                user_message=user_message,
                conversation_history=conversation_history,
                ephemeral_system_prompt=instructions,
                session_id=session_id,
                stream_delta_callback=_on_delta,
                tool_progress_callback=_on_tool_progress,
                tool_start_callback=_on_tool_start,
                tool_complete_callback=_on_tool_complete,
                agent_ref=agent_ref,
                abort_event=conversation_reservation_lost,
                gateway_session_key=gateway_session_key,
                **agent_overrides,
                route=route,
            ))
            # Ensure SSE drain loops can terminate without relying on polling
            # agent_task.done(), which can race with queue timeout checks.
            agent_task.add_done_callback(lambda _fut: _stream_q.put_nowait(None))

            model_name = body.get("model", self._model_name)
            created_at = int(time.time())

            try:
                return await self._write_sse_responses(
                    request=request,
                    response_id=response_id,
                    model=model_name,
                    created_at=created_at,
                    stream_q=_stream_q,
                    agent_task=agent_task,
                    agent_ref=agent_ref,
                    conversation_history=conversation_history,
                    user_message=user_message,
                    instructions=instructions,
                    conversation=conversation,
                    store=store,
                    session_id=session_id,
                    gateway_session_key=gateway_session_key,
                    store_profile=response_profile,
                    conversation_reservation_id=conversation_reservation_id,
                    expected_previous_response_id=(
                        expected_predecessor if expected_predecessor_stated
                        else _UNSTATED
                    ),
                    owner_publish=_publish_owner_turn,
                )
            finally:
                # A stream that never produced a durably mapped response has
                # minted no authority, so its reservation is dropped and an
                # exact retry is free to plan the turn again. One that did has
                # already been completed above, and this is a no-op.
                _release_owner_conversation_reservation()
                _release_owner_response_reservation()

        async def _compute_response():
            result = await self._run_agent(
                user_message=user_message,
                conversation_history=conversation_history,
                ephemeral_system_prompt=instructions,
                session_id=session_id,
                gateway_session_key=gateway_session_key,
                agent_ref=conversation_agent_ref,
                abort_event=conversation_reservation_lost,
                **agent_overrides,
                route=route,
            )
            if conversation_reservation_lost.is_set():
                raise RuntimeError("owner conversation reservation changed")
            return result

        if background:
            created_at = int(time.time())
            model_name = body.get("model", self._model_name)
            queued_response = {
                "id": response_id,
                "object": "response",
                "status": "queued",
                "created_at": created_at,
                "model": model_name,
                "background": True,
                "output": [],
            }
            pending_store_data = {
                "response": queued_response,
                "conversation_history": list(conversation_history),
                "instructions": instructions,
                "session_id": session_id,
            }

            def _store_terminal_background(
                response: dict, *, interrupted: bool = False,
            ) -> None:
                """Persist a terminal background body and everything it ends.

                For an owner conversation the recovery job row is the only
                record that says this response still needs finishing, so it is
                dropped in the SAME transaction that makes the response
                terminal. A separate release ran even when the terminal write
                itself failed, which left the response queued forever with
                nobody left to recover it.

                ``interrupted`` says this ending produced no turn, so the same
                transaction also leaves behind the one thing a browser that
                never received the accept response has left to find: the
                request, and the response id whose outcome it can read.
                """
                if is_owner_conversation:
                    self._response_store.store_terminal_owner_response(
                        profile=response_profile,
                        response_id=response_id,
                        data={**pending_store_data, "response": response},
                        release_job=True,
                        conversation=str(conversation),
                        interrupted=interrupted,
                    )
                    _stop_reservation_heartbeat()
                    return
                self._response_store.put(response_id, {
                    **pending_store_data,
                    "response": response,
                }, profile=response_profile)

            def _terminalize_background(response: dict) -> None:
                """Record ONE interrupted terminal ending, or leave it recoverable.

                Every ending that produced no turn — the inactivity timeout, a
                cancellation, a failed compute, and a finalizer or publication
                that raised — funnels through the same single transaction, so a
                response can never be made terminal twice or half-terminal. If
                that transaction itself fails there is nothing more this task
                can safely do: the job row deliberately survives, and a later
                recovery sweep is what finishes the response. Raising instead
                would kill this task with the response still ``in_progress``,
                which is the very ending this guard exists to prevent.
                """
                try:
                    _store_terminal_background(response, interrupted=True)
                except Exception:
                    logger.exception(
                        "[api_server] could not store the terminal body for "
                        "background response %s; its recovery job survives for "
                        "a later sweep",
                        response_id,
                    )

            async def _run_background_response() -> None:
                in_progress = dict(queued_response)
                in_progress["status"] = "in_progress"
                self._response_store.put(response_id, {
                    **pending_store_data,
                    "response": in_progress,
                }, profile=response_profile)
                # Everything after the compute — finalizing the result,
                # publishing the owner turn, the non-owner head swap and the
                # unpublished handling — is inside the guard too. Unguarded, an
                # exception from any of them escaped this coroutine with no
                # terminal state recorded at all, leaving the response
                # ``in_progress`` forever: the same symptom as a wedged worker,
                # reached from the other side of the await.
                try:
                    result, usage = await _compute_response()
                    response_data, full_history, effective_session_id = (
                        self._finalize_response_result(
                            response_id=response_id,
                            created_at=created_at,
                            model=model_name,
                            conversation_history=conversation_history,
                            user_message=user_message,
                            session_id=session_id,
                            result=result,
                            usage=usage,
                            background=True,
                        )
                    )
                    published = True
                    if is_owner_conversation:
                        # ONE commit for the whole publication, exactly as the
                        # standard and streaming paths do — plus, on this path,
                        # the retirement of the recovery job the 202 reserved.
                        published = _publish_owner_turn(
                            response_data, full_history, effective_session_id,
                            release_job=True,
                        )
                    else:
                        self._response_store.put(response_id, {
                            "response": response_data,
                            "conversation_history": full_history,
                            "instructions": instructions,
                            "session_id": effective_session_id,
                        }, profile=response_profile)
                        if conversation:
                            published = self._response_store.set_conversation(
                                conversation,
                                response_id,
                                owner_proposal=(
                                    _owner_history_has_actionable_final_proposal(
                                        full_history
                                    )
                                ),
                                profile=response_profile,
                                reservation_id=conversation_reservation_id,
                                expected_previous_response_id=(
                                    expected_predecessor
                                    if expected_predecessor_stated
                                    else _UNSTATED
                                ),
                            )
                    if not published:
                        failed = dict(response_data)
                        failed.update({
                            "status": "failed",
                            "output": [],
                            "error": {
                                "code": "owner_conversation_locked",
                                "message": "Owner conversation changed before this response completed",
                            },
                        })
                        # Publication failure is still a terminal outcome for
                        # THIS response, so its job retires with the terminal
                        # body — and only with it. If this write fails the job
                        # survives and a later sweep recovers the response.
                        if is_owner_conversation:
                            self._response_store.store_terminal_owner_response(
                                profile=response_profile,
                                response_id=response_id,
                                data={
                                    "response": failed,
                                    "conversation_history": list(conversation_history),
                                    "instructions": instructions,
                                    "session_id": effective_session_id,
                                },
                                release_job=True,
                                conversation=str(conversation),
                                interrupted=True,
                            )
                            _stop_reservation_heartbeat()
                        else:
                            self._response_store.put(response_id, {
                                "response": failed,
                                "conversation_history": list(conversation_history),
                                "instructions": instructions,
                                "session_id": effective_session_id,
                            }, profile=response_profile)
                except asyncio.CancelledError:
                    incomplete = dict(queued_response)
                    incomplete.update({
                        "status": "incomplete",
                        "incomplete_details": {"reason": "cancelled"},
                    })
                    _terminalize_background(incomplete)
                    raise
                except Exception as exc:
                    safe_error = _redact_api_error_text(exc)
                    logger.error(
                        "Background response %s failed: %s",
                        response_id,
                        safe_error,
                    )
                    failed = dict(queued_response)
                    failed.update({
                        "status": "failed",
                        "error": {
                            "code": "server_error",
                            "message": safe_error,
                        },
                    })
                    _terminalize_background(failed)
                    return

            async def _start_background_response():
                if is_owner_conversation:
                    # ONE transaction for everything the 202 promises: the
                    # queued body, the durable executor recovery job, and the
                    # replay record an exact retry answers from — including
                    # after a restart, when the in-memory cache is gone. Three
                    # separate commits meant a crash between any two left an
                    # accepted owner turn with no executor, or an idempotency
                    # key that had minted a response and could never record one.
                    self._response_store.accept_owner_background_response(
                        profile=response_profile,
                        response_id=response_id,
                        data=pending_store_data,
                        conversation=str(conversation),
                        replay=(
                            queued_response
                            if owner_response_idempotency is not None else None
                        ),
                        session_scope=(
                            owner_response_idempotency[0]
                            if owner_response_idempotency is not None else None
                        ),
                        idempotency_key=(
                            owner_response_idempotency[1]
                            if owner_response_idempotency is not None else None
                        ),
                        session_id=session_id,
                    )
                else:
                    self._response_store.put(
                        response_id, pending_store_data, profile=response_profile,
                    )
                task = asyncio.create_task(_run_background_response())
                self._background_tasks.add(task)
                task.add_done_callback(self._background_tasks.discard)
                if is_owner_conversation:
                    # Marks this job as one THIS process is still driving, so
                    # its lease is heartbeated and no sibling reclaims it — and
                    # so a task that dies without recording a terminal state
                    # becomes reclaimable the moment it stops.
                    self._owner_response_jobs.add(response_id)
                    task.add_done_callback(
                        lambda _task: self._owner_response_jobs.discard(response_id)
                    )
                if conversation_reservation_id is not None:
                    task.add_done_callback(
                        lambda _task: _release_owner_conversation_reservation()
                    )
                await asyncio.sleep(0)
                return queued_response, session_id

            try:
                if (
                    response_idempotency_cache_key is not None
                    and response_idempotency_fingerprint is not None
                ):
                    cached_response, cached_session_id = await _idem_cache.get_or_set(
                        response_idempotency_cache_key,
                        response_idempotency_fingerprint,
                        _start_background_response,
                    )
                else:
                    cached_response, cached_session_id = (
                        await _start_background_response()
                    )
            except _IdempotencyConflict:
                _release_owner_conversation_reservation()
                _release_owner_response_reservation()
                return web.json_response(
                    _openai_error(
                        "Idempotency-Key was already used for a different request",
                        code="idempotency_conflict",
                    ),
                    status=409,
                )
            except Exception as exc:
                _release_owner_conversation_reservation()
                _release_owner_response_reservation()
                logger.error(
                    "Error starting background response: %s",
                    _redact_api_error_text(exc),
                    exc_info=True,
                )
                return web.json_response(
                    _openai_error("Internal server error", err_type="server_error"),
                    status=500,
                )
            if cached_response.get("id") != response_id:
                _release_owner_conversation_reservation()
            response_headers = {}
            if cached_session_id:
                response_headers["X-Hermes-Session-Id"] = cached_session_id
            if gateway_session_key:
                response_headers["X-Hermes-Session-Key"] = gateway_session_key
            return web.json_response(cached_response, headers=response_headers)

        async def _compute_finalized_response():
            result, usage = await _compute_response()
            created_at = int(time.time())
            response_data, full_history, effective_session_id = (
                self._finalize_response_result(
                    response_id=response_id,
                    created_at=created_at,
                    model=body.get("model", self._model_name),
                    conversation_history=conversation_history,
                    user_message=user_message,
                    session_id=session_id,
                    result=result,
                    usage=usage,
                    background=False,
                )
            )
            if store:
                if is_owner_conversation:
                    # ONE commit: the response, the head compare-and-swap, this
                    # turn's reservation release and the durable replay record
                    # an exact retry answers from.
                    if not _publish_owner_turn(
                        response_data, full_history, effective_session_id,
                    ):
                        raise _OwnerConversationReservationChanged
                    return response_data, effective_session_id
                self._response_store.put(response_id, {
                    "response": response_data,
                    "conversation_history": full_history,
                    "instructions": instructions,
                    "session_id": effective_session_id,
                }, profile=response_profile)
                if conversation and not self._response_store.set_conversation(
                    conversation,
                    response_id,
                    owner_proposal=_owner_history_has_actionable_final_proposal(
                        full_history
                    ),
                    profile=response_profile,
                    reservation_id=conversation_reservation_id,
                    expected_previous_response_id=(
                        expected_predecessor if expected_predecessor_stated
                        else _UNSTATED
                    ),
                ):
                    raise _OwnerConversationReservationChanged
            return response_data, effective_session_id

        try:
            if (
                response_idempotency_cache_key is not None
                and response_idempotency_fingerprint is not None
            ):
                response_data, _effective_session_id = await _idem_cache.get_or_set(
                    response_idempotency_cache_key,
                    response_idempotency_fingerprint,
                    _compute_finalized_response,
                )
            else:
                response_data, _effective_session_id = (
                    await _compute_finalized_response()
                )
        except _IdempotencyConflict:
            _release_owner_conversation_reservation()
            _release_owner_response_reservation()
            return web.json_response(
                _openai_error(
                    "Idempotency-Key was already used for a different request",
                    code="idempotency_conflict",
                ),
                status=409,
            )
        except _OwnerConversationReservationChanged:
            _release_owner_conversation_reservation()
            _release_owner_response_reservation()
            return web.json_response(
                _openai_error(
                    "Owner conversation changed before this response completed",
                    code="owner_conversation_locked",
                ),
                status=409,
            )
        except Exception as exc:
            logger.error(
                "Error running agent for responses: %s",
                _redact_api_error_text(exc),
                exc_info=True,
            )
            _release_owner_conversation_reservation()
            _release_owner_response_reservation()
            return web.json_response(
                _openai_error("Internal server error", err_type="server_error"),
                status=500,
            )

        response_headers = {"X-Hermes-Session-Id": _effective_session_id}
        if gateway_session_key:
            response_headers["X-Hermes-Session-Key"] = gateway_session_key
        _release_owner_conversation_reservation()
        return web.json_response(response_data, headers=response_headers)

    # ------------------------------------------------------------------
    # GET / DELETE response endpoints
    # ------------------------------------------------------------------

    async def _handle_owner_conversation_history(
        self, request: "web.Request",
    ) -> "web.Response":
        """Return the owner-safe projection of one stored conversation."""
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err

        conversation = request.match_info["conversation"]
        view = request.query.get("view")
        if request.query:
            if len(request.query) != 1 or view != "sessions":
                return web.json_response(
                    _openai_error("Invalid owner conversation view"),
                    status=400,
                )
            if re.fullmatch(r"raphael-owner-[a-f0-9]{32}", conversation) is None:
                return web.json_response(
                    _openai_error("Invalid owner conversation group"),
                    status=400,
                )
            index = self._response_store.owner_session_index(
                _active_owner_profile(), conversation,
            )
            return web.json_response({
                "object": "hermes.response.owner_sessions",
                **index,
            })

        try:
            snapshot = self._response_store.owner_history_snapshot(
                conversation, profile=_active_owner_profile(),
            )
        except OwnerAuthorityBroken as exc:
            logger.error("Owner conversation authority is unreadable: %s", exc)
            return web.json_response(
                _openai_error(
                    "Owner conversation history is unavailable",
                    err_type="server_error",
                    code="owner_history_unavailable",
                ),
                status=503,
            )
        return web.json_response({
            "object": "hermes.response.owner_history",
            **snapshot,
        })

    async def _handle_consume_owner_proposal(
        self, request: "web.Request",
    ) -> "web.Response":
        """Mark the exact applied proposal so it cannot regain approval authority."""
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err
        try:
            body = await request.json()
        except (json.JSONDecodeError, Exception):
            return web.json_response(_openai_error("Invalid JSON"), status=400)
        if (
            not isinstance(body, dict)
            or set(body) != {"response_id"}
            or not isinstance(body["response_id"], str)
        ):
            return web.json_response(
                _openai_error("Invalid owner proposal consumption request"),
                status=400,
            )
        consumed = self._response_store.mark_owner_proposal_consumed(
            _active_owner_profile(),
            request.match_info["conversation"],
            body["response_id"],
        )
        if not consumed:
            return web.json_response(
                _openai_error("Owner proposal is not current"),
                status=409,
            )
        return web.json_response({
            "object": "hermes.response.owner_proposal_consumption",
            "consumed": True,
        })

    async def _handle_acknowledge_owner_recovery(
        self, request: "web.Request",
    ) -> "web.Response":
        """Seal one interrupted request, once its caller holds what it said.

        Deliberately separate from the proposal authority endpoint: this grants
        no action. It records the request and its failure as the new durable,
        non-actionable turn, so any proposal from an earlier turn stays
        superseded, then says "I have the words and the outcome". Reading the
        projection must not be allowed to say that on the caller's behalf —
        that read can be lost too. Repeating it is not an error, for exactly
        the same reason.

        The answer says which of three things happened. Always reporting
        success made this endpoint unverifiable: a caller could not tell a
        record it really retired from one that was never here, and a request
        naming a DIFFERENT unanswered record was told it had been dealt with.
        """
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err
        try:
            body = await request.json()
        except Exception:
            return web.json_response(_openai_error("Invalid JSON"), status=400)
        if (
            not isinstance(body, dict)
            or set(body) != {"response_id"}
            or not isinstance(body["response_id"], str)
        ):
            return web.json_response(
                _openai_error("Invalid owner recovery acknowledgement"),
                status=400,
            )
        outcome = self._response_store.acknowledge_owner_conversation_recovery(
            _active_owner_profile(),
            request.match_info["conversation"],
            body["response_id"],
        )
        return web.json_response(
            {
                "object": "hermes.response.owner_recovery_acknowledgement",
                "acknowledged": outcome != "mismatch",
                "outcome": outcome,
            },
            status=409 if outcome == "mismatch" else 200,
        )

    async def _handle_owner_conversation_authority(
        self, request: "web.Request",
    ) -> "web.Response":
        """Apply one exact atomic owner-proposal authority transition."""
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err
        try:
            body = await request.json()
        except Exception:
            return web.json_response(_openai_error("Invalid JSON"), status=400)
        if not isinstance(body, dict) or not isinstance(body.get("action"), str):
            return web.json_response(
                _openai_error("Invalid owner proposal authority request"),
                status=400,
            )

        action = body["action"]
        expected_keys = {
            "claim": {"action", "response_id", "claim_id"},
            "abandon": {"action", "response_id", "claim_id"},
            "attach": {"action", "response_id", "claim_id", "run_id"},
            "complete": {"action", "response_id", "claim_id", "run_id"},
            "release": {"action", "response_id", "claim_id", "run_id"},
            "reconcile": {"action", "response_id", "claim_id", "run_id"},
            # ``head_response_id`` is optional so an older caller still works,
            # but a caller that states it gets its close compare-and-swapped
            # against the conversation's actual durable head.
            #
            # ``next_session_id`` is likewise optional, and names the change
            # session this group moves on to. Stating it makes the group's
            # durable current-session pointer move with the close.
            "close": (
                {"action", "response_id"},
                {"head_response_id", "next_session_id"},
            ),
        }
        if action not in expected_keys:
            return web.json_response(
                _openai_error("Invalid owner proposal authority request"),
                status=400,
            )
        shape = expected_keys[action]
        required, optional = shape if isinstance(shape, tuple) else (shape, set())
        if not required <= set(body) or not set(body) <= (required | optional):
            return web.json_response(
                _openai_error("Invalid owner proposal authority request"),
                status=400,
            )

        conversation = request.match_info["conversation"]
        response_id = body.get("response_id")
        profile = _active_owner_profile()
        if action == "claim":
            applied = self._response_store.claim_owner_proposal(
                profile, conversation, response_id, body.get("claim_id"),
            )
        elif action == "abandon":
            applied = self._response_store.abandon_unattached_owner_claim(
                profile, conversation, response_id, body.get("claim_id"),
            )
        elif action == "attach":
            run_id = body.get("run_id")
            # /v1/runs creates the binding before returning its run ID. This
            # service endpoint only verifies that exact native binding; it may
            # not attach an arbitrary caller-declared run to owner authority.
            applied = (
                run_id in self._run_statuses
                and self._response_store.owner_run_is_attached(
                    profile, conversation, response_id, body.get("claim_id"), run_id,
                )
            )
        elif action == "complete":
            run_id = body.get("run_id")
            # Verification only: the native tool callback already consumed
            # authority. A caller cannot turn generic run status into proof.
            applied = self._response_store.owner_claim_is_completed(
                profile, conversation, response_id, body.get("claim_id"), run_id,
            )
        elif action == "release":
            run_id = body.get("run_id")
            applied = self._response_store.owner_claim_is_released(
                profile, conversation, response_id, body.get("claim_id"), run_id,
            )
        elif action == "reconcile":
            run_id = body.get("run_id")
            # A caller may ask the server to recover authority after a restart,
            # but may not release a run the current process can still observe.
            # The exact durable proposal/claim/run tuple remains the final fence.
            applied = (
                run_id not in self._run_statuses
                and run_id not in self._active_run_tasks
                and self._response_store.release_owner_claim(
                    profile, conversation, response_id, body.get("claim_id"), run_id,
                )
            )
        else:
            applied = self._response_store.close_owner_conversation(
                profile, conversation, response_id,
                expected_head_response_id=(
                    body["head_response_id"] if "head_response_id" in body
                    else _UNSTATED
                ),
                next_session_id=body.get("next_session_id"),
            )
        if not applied:
            return web.json_response(
                _openai_error("Owner proposal authority changed"),
                status=409,
            )
        return web.json_response({
            "object": "hermes.response.owner_authority",
            "action": action,
            "applied": True,
        })

    async def _handle_get_response(self, request: "web.Request") -> "web.Response":
        """GET /v1/responses/{response_id} — retrieve a stored response."""
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err

        response_id = request.match_info["response_id"]
        profile = _active_owner_profile()
        stored = self._response_store.get(response_id, profile=profile)
        if stored is None:
            return web.json_response(_openai_error(f"Response not found: {response_id}"), status=404)

        return web.json_response(stored["response"])

    async def _handle_delete_response(self, request: "web.Request") -> "web.Response":
        """DELETE /v1/responses/{response_id} — delete a stored response."""
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err

        response_id = request.match_info["response_id"]
        profile = _active_owner_profile()
        if self._response_store.owner_response_is_current(
            response_id, profile=profile,
        ):
            return web.json_response(
                _openai_error(
                    "Response is retained by an owner conversation",
                    code="owner_conversation_active",
                ),
                status=409,
            )
        deleted = self._response_store.delete(response_id, profile=profile)
        if not deleted:
            return web.json_response(_openai_error(f"Response not found: {response_id}"), status=404)

        return web.json_response({
            "id": response_id,
            "object": "response",
            "deleted": True,
        })

    # ------------------------------------------------------------------
    # Cron jobs API
    # ------------------------------------------------------------------

    _JOB_ID_RE = __import__("re").compile(r"[a-f0-9]{12}")
    # Allowed fields for update — prevents clients injecting arbitrary keys
    _UPDATE_ALLOWED_FIELDS = {"name", "schedule", "prompt", "deliver", "skills", "skill", "repeat", "enabled"}
    _MAX_NAME_LENGTH = 200
    _MAX_PROMPT_LENGTH = 5000

    @staticmethod
    def _check_jobs_available() -> Optional["web.Response"]:
        """Return error response if cron module isn't available."""
        if not _CRON_AVAILABLE:
            return web.json_response(
                {"error": "Cron module not available"}, status=501,
            )
        return None

    def _check_job_id(self, request: "web.Request") -> tuple:
        """Validate and extract job_id. Returns (job_id, error_response)."""
        job_id = request.match_info["job_id"]
        if not self._JOB_ID_RE.fullmatch(job_id):
            logger.warning(
                "Cron jobs API rejected invalid job_id %r: %s",
                job_id,
                self._request_audit_log_suffix(request),
            )
            return job_id, web.json_response(
                {"error": "Invalid job ID format"}, status=400,
            )
        return job_id, None

    async def _handle_list_jobs(self, request: "web.Request") -> "web.Response":
        """GET /api/jobs — list all cron jobs."""
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err
        cron_err = self._check_jobs_available()
        if cron_err:
            return cron_err
        try:
            include_disabled = request.query.get("include_disabled", "").lower() in {"true", "1"}
            jobs = _cron_list(include_disabled=include_disabled)
            return web.json_response({"jobs": jobs})
        except Exception as e:
            return web.json_response({"error": _redact_api_error_text(e)}, status=500)

    async def _handle_create_job(self, request: "web.Request") -> "web.Response":
        """POST /api/jobs — create a new cron job."""
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err
        cron_err = self._check_jobs_available()
        if cron_err:
            return cron_err
        try:
            body = await request.json()
            name = (body.get("name") or "").strip()
            schedule = (body.get("schedule") or "").strip()
            prompt = body.get("prompt", "")
            deliver = body.get("deliver", "local")
            skills = body.get("skills")
            repeat = body.get("repeat")

            if not name:
                return web.json_response({"error": "Name is required"}, status=400)
            if len(name) > self._MAX_NAME_LENGTH:
                return web.json_response(
                    {"error": f"Name must be ≤ {self._MAX_NAME_LENGTH} characters"}, status=400,
                )
            if not schedule:
                return web.json_response({"error": "Schedule is required"}, status=400)
            if len(prompt) > self._MAX_PROMPT_LENGTH:
                return web.json_response(
                    {"error": f"Prompt must be ≤ {self._MAX_PROMPT_LENGTH} characters"}, status=400,
                )
            if prompt and _scan_cron_prompt is not None:
                scan_error = _scan_cron_prompt(prompt)
                if scan_error:
                    return web.json_response({"error": scan_error}, status=400)
            if repeat is not None and (not isinstance(repeat, int) or repeat < 1):
                return web.json_response({"error": "Repeat must be a positive integer"}, status=400)

            kwargs = {
                "prompt": prompt,
                "schedule": schedule,
                "name": name,
                "deliver": deliver,
                "origin": self._cron_origin_from_request(request),
            }
            if skills:
                kwargs["skills"] = skills
            if repeat is not None:
                kwargs["repeat"] = repeat

            job = _cron_create(**kwargs)
            return web.json_response({"job": job})
        except _CronSchedulerRegistrationError as e:
            return web.json_response(e.to_dict(), status=424)
        except Exception as e:
            return web.json_response({"error": _redact_api_error_text(e)}, status=500)

    async def _handle_get_job(self, request: "web.Request") -> "web.Response":
        """GET /api/jobs/{job_id} — get a single cron job."""
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err
        cron_err = self._check_jobs_available()
        if cron_err:
            return cron_err
        job_id, id_err = self._check_job_id(request)
        if id_err:
            return id_err
        try:
            job = _cron_get(job_id)
            if not job:
                return web.json_response({"error": "Job not found"}, status=404)
            return web.json_response({"job": job})
        except Exception as e:
            return web.json_response({"error": _redact_api_error_text(e)}, status=500)

    async def _handle_update_job(self, request: "web.Request") -> "web.Response":
        """PATCH /api/jobs/{job_id} — update a cron job."""
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err
        cron_err = self._check_jobs_available()
        if cron_err:
            return cron_err
        job_id, id_err = self._check_job_id(request)
        if id_err:
            return id_err
        try:
            body = await request.json()
            # Whitelist allowed fields to prevent arbitrary key injection
            sanitized = {k: v for k, v in body.items() if k in self._UPDATE_ALLOWED_FIELDS}
            if not sanitized:
                return web.json_response({"error": "No valid fields to update"}, status=400)
            # Validate lengths if present
            if "name" in sanitized and len(sanitized["name"]) > self._MAX_NAME_LENGTH:
                return web.json_response(
                    {"error": f"Name must be ≤ {self._MAX_NAME_LENGTH} characters"}, status=400,
                )
            if "prompt" in sanitized and len(sanitized["prompt"]) > self._MAX_PROMPT_LENGTH:
                return web.json_response(
                    {"error": f"Prompt must be ≤ {self._MAX_PROMPT_LENGTH} characters"}, status=400,
                )
            if sanitized.get("prompt") and _scan_cron_prompt is not None:
                scan_error = _scan_cron_prompt(sanitized["prompt"])
                if scan_error:
                    return web.json_response({"error": scan_error}, status=400)
            job = _cron_update(job_id, sanitized)
            if not job:
                return web.json_response({"error": "Job not found"}, status=404)
            _notify_cron_provider_jobs_changed()
            return web.json_response({"job": job})
        except Exception as e:
            return web.json_response({"error": _redact_api_error_text(e)}, status=500)

    async def _handle_delete_job(self, request: "web.Request") -> "web.Response":
        """DELETE /api/jobs/{job_id} — delete a cron job."""
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err
        cron_err = self._check_jobs_available()
        if cron_err:
            return cron_err
        job_id, id_err = self._check_job_id(request)
        if id_err:
            return id_err
        try:
            success = _cron_remove(job_id)
            if not success:
                return web.json_response({"error": "Job not found"}, status=404)
            _notify_cron_provider_jobs_changed()
            return web.json_response({"ok": True})
        except Exception as e:
            return web.json_response({"error": _redact_api_error_text(e)}, status=500)

    async def _handle_pause_job(self, request: "web.Request") -> "web.Response":
        """POST /api/jobs/{job_id}/pause — pause a cron job."""
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err
        cron_err = self._check_jobs_available()
        if cron_err:
            return cron_err
        job_id, id_err = self._check_job_id(request)
        if id_err:
            return id_err
        try:
            job = _cron_pause(job_id)
            if not job:
                return web.json_response({"error": "Job not found"}, status=404)
            _notify_cron_provider_jobs_changed()
            return web.json_response({"job": job})
        except Exception as e:
            return web.json_response({"error": _redact_api_error_text(e)}, status=500)

    async def _handle_resume_job(self, request: "web.Request") -> "web.Response":
        """POST /api/jobs/{job_id}/resume — resume a paused cron job."""
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err
        cron_err = self._check_jobs_available()
        if cron_err:
            return cron_err
        job_id, id_err = self._check_job_id(request)
        if id_err:
            return id_err
        try:
            job = _cron_resume(job_id)
            if not job:
                return web.json_response({"error": "Job not found"}, status=404)
            _notify_cron_provider_jobs_changed()
            return web.json_response({"job": job})
        except Exception as e:
            return web.json_response({"error": _redact_api_error_text(e)}, status=500)

    async def _handle_run_job(self, request: "web.Request") -> "web.Response":
        """POST /api/jobs/{job_id}/run — trigger immediate execution."""
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err
        draining = self._draining_response()
        if draining is not None:
            return draining
        cron_err = self._check_jobs_available()
        if cron_err:
            return cron_err
        job_id, id_err = self._check_job_id(request)
        if id_err:
            return id_err
        try:
            job = _cron_trigger(job_id)
            if not job:
                return web.json_response({"error": "Job not found"}, status=404)
            return web.json_response({"job": job})
        except Exception as e:
            return web.json_response({"error": _redact_api_error_text(e)}, status=500)

    async def _handle_cron_fire(self, request: "web.Request") -> "web.Response":
        """POST /api/cron/fire — Chronos managed-cron fire webhook (NAS → agent).

        Authenticated by a NAS-minted JWT (verified via the pluggable
        fire-verifier), NOT API_SERVER_KEY — NAS holds no API server key, and
        this is the only inbound that can trigger remote job execution, so it
        gets its own purpose-scoped token check.

        Returns 202 + runs the job in the background so a long agent turn never
        trips NAS's HTTP timeout. The store CAS claim inside fire_due guards
        against double-fire on a NAS/scheduler retry.
        """
        from hermes_cli.config import cfg_get, load_config
        from plugins.cron_providers.chronos.verify import get_fire_verifier

        auth = request.headers.get("Authorization", "")
        token = auth[7:].strip() if auth.startswith("Bearer ") else ""

        cfg = load_config()
        verifier = get_fire_verifier()
        verify_kwargs = dict(
            token=token,
            expected_audience=cfg_get(cfg, "cron", "chronos", "expected_audience", default=""),
            jwks_or_key=cfg_get(cfg, "cron", "chronos", "nas_jwks_url", default="") or None,
            issuer=cfg_get(cfg, "cron", "chronos", "portal_url", default="") or None,
        )
        try:
            if asyncio.iscoroutinefunction(verifier):
                claims = await verifier(**verify_kwargs)
            else:
                # The verifier resolves the NAS signing key from a JWKS URL,
                # which is a synchronous HTTP GET on a cache miss (cold client
                # or a rotated kid) — keep that blocking I/O off the event loop
                # so a slow or rate-limited portal can't stall every other
                # adapter sharing this loop. Same hardening the platform HTTP
                # event verifier already got.
                claims = await asyncio.to_thread(verifier, **verify_kwargs)
        except Exception:
            # Fail closed: a crashing verifier must never admit a fire — this
            # is the only inbound that can trigger remote job execution.
            logger.exception("cron fire: verifier crashed; rejecting token")
            claims = None
        if claims is None:
            logger.warning(
                "cron fire: rejected invalid token: %s",
                self._request_audit_log_suffix(request),
            )
            return web.json_response({"error": "invalid fire token"}, status=401)
        draining = self._draining_response()
        if draining is not None:
            return draining

        with _reserve_pending_api_work(self) as reservation:
            try:
                body = await request.json()
            except Exception:
                body = {}
            job_id = (body or {}).get("job_id")
            if not job_id:
                return web.json_response({"error": "missing job_id"}, status=400)

            from cron.scheduler_provider import (
                provider_supports_split_fire,
                resolve_cron_scheduler,
            )
            provider = resolve_cron_scheduler()

            loop = asyncio.get_running_loop()
            # Live adapters for delivery parity with the built-in ticker
            # (gateway/run.py passes runner.adapters to the in-process
            # scheduler). Without them, _deliver_result cannot resolve a live
            # transport, so E2EE platforms and relay-fronted logical platforms
            # (whose only send path IS the live relay adapter — no native
            # credential exists) fail with "platform 'X' not
            # configured/enabled" on every external-provider fire even though
            # the same job delivers fine under the built-in ticker.
            runner = self.gateway_runner or request.app.get("gateway_runner")
            if runner is None:
                try:
                    from gateway.run import _gateway_runner_ref

                    runner = _gateway_runner_ref()
                except Exception:
                    runner = None
            adapters = getattr(runner, "adapters", None) or None

            if not provider_supports_split_fire(provider):
                # Legacy single-phase provider: it overrides the documented
                # ``fire_due`` hook (custom claim/re-arm/telemetry) but
                # inherits the base ``claim_fire`` — driving it through the
                # split claim path would silently bypass that override.
                task = asyncio.create_task(
                    asyncio.to_thread(
                        provider.fire_due,
                        job_id,
                        adapters=adapters,
                        loop=loop,
                    )
                )
                reservation["detached"] = True
                task.add_done_callback(
                    lambda _task: _release_pending_api_work(self, reservation)
                )
                try:
                    self._background_tasks.add(task)
                    task.add_done_callback(self._background_tasks.discard)
                except (TypeError, AttributeError):
                    pass
                return web.json_response(
                    {"status": "accepted", "job_id": job_id}, status=202
                )

            # Persist the attempt and exact store owner before acknowledging NAS.
            # A failure here is retryable and the reservation remains attached.
            try:
                claimed_job = await asyncio.to_thread(provider.claim_fire, job_id)
            except Exception as exc:
                logger.error("cron fire admission failed for %s: %s", job_id, exc)
                return web.json_response(
                    {"error": "cron fire admission failed", "job_id": job_id},
                    status=503,
                )
            if claimed_job is None:
                return web.json_response(
                    {"status": "duplicate", "job_id": job_id},
                    status=200,
                )

            task = asyncio.create_task(
                asyncio.to_thread(
                    provider.fire_claimed,
                    claimed_job,
                    adapters=adapters,
                    loop=loop,
                )
            )
            reservation["detached"] = True
            task.add_done_callback(
                lambda _task: _release_pending_api_work(self, reservation)
            )
            try:
                self._background_tasks.add(task)
                task.add_done_callback(self._background_tasks.discard)
            except (TypeError, AttributeError):
                pass

            return web.json_response({"status": "accepted", "job_id": job_id}, status=202)


    # ------------------------------------------------------------------
    # Output extraction helper
    # ------------------------------------------------------------------

    @staticmethod
    def _build_response_conversation_history(
        conversation_history: List[Dict[str, Any]],
        user_message: Any,
        result: Dict[str, Any],
        final_response: Any,
    ) -> List[Dict[str, Any]]:
        """Build the stored Responses transcript without duplicating history.

        When context compression occurs during a turn the agent returns a
        compressed full transcript in ``result["messages"]`` (starting with a
        summary) and sets ``result["_compressed"] = True``.  Because the
        compressed transcript does not share the input ``conversation_history``
        prefix, the normal turn-start detection fails and old code would
        concatenate the uncompressed history on front, bloating the stored
        context and re-triggering compression on every subsequent request.
        """
        prior = list(conversation_history)
        current_user = {"role": "user", "content": user_message}
        agent_messages = result.get("messages") if isinstance(result, dict) else None

        if isinstance(agent_messages, list) and agent_messages:
            turn_start = APIServerAdapter._response_messages_turn_start_index(
                conversation_history,
                user_message,
                result,
            )
            if turn_start:
                return list(agent_messages)

            # turn_start == 0: agent_messages does not start with prior.
            # This can happen because compression rewrote the transcript
            # (summary prefix replaces original history), OR because
            # agent_messages only carries the current turn without prior.
            # The ``_compressed`` flag (set by _run_agent after compaction)
            # distinguishes — skip the concatenation and use the compressed
            # transcript directly.
            if result.get("_compressed"):
                return list(agent_messages)

            full_history = prior
            full_history.append(current_user)
            full_history.extend(agent_messages)
            return full_history

        full_history = prior
        full_history.append(current_user)
        full_history.append({"role": "assistant", "content": final_response})
        return full_history

    @staticmethod
    def _messages_open_with(
        agent_messages: List[Dict[str, Any]],
        prefix: List[Dict[str, Any]],
    ) -> bool:
        """Whether ``agent_messages`` opens with ``prefix``, message for message.

        A message's identity here is everything a provider would see — role,
        content, and every tool-call field such as ``tool_calls``,
        ``tool_call_id``, ``name``, ``refusal`` or ``reasoning``. Only the
        durable bookkeeping the live transcript stamps onto its own copies is
        ignored, and ``PERSISTENCE_ONLY_MESSAGE_FIELDS`` is what names it: those
        fields describe Hermes' record, not what was said.

        Comparing whole dicts instead meant this turn's own user message never
        equalled the one the request described, because the agent had stamped
        it in the meantime. Prefix detection then reported "no shared prefix"
        for an already-complete transcript and the caller concatenated it onto
        itself, storing the owner's message twice.
        """
        if len(agent_messages) < len(prefix):
            return False

        def identity(message: Any) -> Any:
            if not isinstance(message, dict):
                return message
            return {
                key: value
                for key, value in message.items()
                if key not in PERSISTENCE_ONLY_MESSAGE_FIELDS
            }

        return all(
            identity(actual) == identity(expected)
            for actual, expected in zip(agent_messages, prefix)
        )

    @staticmethod
    def _response_messages_turn_start_index(
        conversation_history: List[Dict[str, Any]],
        user_message: Any,
        result: Dict[str, Any],
    ) -> int:
        """Detect transcript-shaped result["messages"] and return turn start."""
        agent_messages = result.get("messages") if isinstance(result, dict) else None
        if not isinstance(agent_messages, list) or not agent_messages:
            return 0

        prior = list(conversation_history)
        current_user = {"role": "user", "content": user_message}
        expected_prefix = prior + [current_user]
        # Positional and greedy from index 0, so two legitimately identical
        # owner requests sent as two real turns each match their own position
        # and both survive.
        if APIServerAdapter._messages_open_with(agent_messages, expected_prefix):
            return len(expected_prefix)
        if prior and APIServerAdapter._messages_open_with(agent_messages, prior):
            return len(prior)
        return 0

    @classmethod
    def _turn_transcript_messages(
        cls,
        conversation_history: List[Dict[str, Any]],
        user_message: Any,
        result: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Return this turn's assistant/tool messages in client-safe shape.

        The streaming SSE contract delivers all assistant text as
        ``assistant.delta`` events under one ``message_id`` interleaved with
        ``tool.*`` events, and a single ``assistant.completed`` carrying only
        the final reply.  A client that accumulates deltas into one buffer
        cannot reconstruct *intermediate* assistant text segments that preceded
        tool calls — so when the page is re-opened mid/post-stream those
        segments appear lost, even though state.db persisted them correctly.

        Emitting the authoritative per-turn transcript on ``run.completed`` lets
        any SSE consumer reconcile its live view against ground truth without a
        separate ``GET /messages`` round-trip.  Purely additive: clients that
        ignore the field are unaffected.  Refs #34703.
        """
        agent_messages = result.get("messages") if isinstance(result, dict) else None
        if not isinstance(agent_messages, list) or not agent_messages:
            return []
        start = cls._response_messages_turn_start_index(
            conversation_history, user_message, result
        )
        turn = agent_messages[start:]
        out: List[Dict[str, Any]] = []
        for msg in turn:
            if not isinstance(msg, dict):
                continue
            if msg.get("role") not in {"assistant", "tool"}:
                continue
            out.append(cls._message_response(msg))
        return out

    @staticmethod
    def _extract_output_items(result: Dict[str, Any], start_index: int = 0) -> List[Dict[str, Any]]:
        """
        Build the output item array from the agent's messages.

        Walks *result["messages"]* starting at *start_index* and emits:
        - ``function_call`` items for each tool_call on assistant messages
        - ``function_call_output`` items for each tool-role message
        - a final ``message`` item with the assistant's text reply
        """
        items: List[Dict[str, Any]] = []
        messages = result.get("messages", [])
        if start_index > 0:
            messages = messages[start_index:]

        for msg in messages:
            role = msg.get("role")
            if role == "assistant" and msg.get("tool_calls"):
                for tc in msg["tool_calls"]:
                    func = tc.get("function", {})
                    items.append({
                        "id": f"fc_{uuid.uuid4().hex[:24]}",
                        "type": "function_call",
                        # These calls were already executed server-side by the
                        # Hermes agent; they are replayed for structured tool
                        # UI only.  Mark them completed (matching the SSE
                        # streaming path) so OpenAI clients don't interpret
                        # them as pending calls the client must execute.
                        "status": "completed",
                        "name": func.get("name", ""),
                        "arguments": func.get("arguments", ""),
                        "call_id": tc.get("id", ""),
                    })
            elif role == "tool":
                items.append({
                    "id": f"fco_{uuid.uuid4().hex[:24]}",
                    "type": "function_call_output",
                    "status": "completed",
                    "call_id": msg.get("tool_call_id", ""),
                    "output": msg.get("content", ""),
                })

        # Final assistant message
        final = result.get("final_response", "")
        if not final:
            final = _redact_api_error_text(result.get("error", "(No response generated)"))

        items.append({
            "type": "message",
            "role": "assistant",
            "content": [
                {
                    "type": "output_text",
                    "text": final,
                }
            ],
        })
        return items

    def _finalize_response_result(
        self,
        *,
        response_id: str,
        created_at: int,
        model: str,
        conversation_history: List[Dict[str, Any]],
        user_message: Any,
        session_id: str,
        result: Dict[str, Any],
        usage: Dict[str, Any],
        background: bool,
    ) -> tuple[Dict[str, Any], List[Dict[str, Any]], str]:
        """Build one completed Responses object and its durable transcript."""
        final_response = _resolve_media_to_data_urls(result.get("final_response", ""))
        if not final_response:
            final_response = _redact_api_error_text(
                result.get("error", "(No response generated)")
            )

        full_history = self._build_response_conversation_history(
            conversation_history,
            user_message,
            result,
            final_response,
        )

        effective_session_id = session_id
        result_session_id = result.get("session_id")
        if isinstance(result_session_id, str) and result_session_id:
            effective_session_id = result_session_id

        output_start_index = self._response_messages_turn_start_index(
            conversation_history,
            user_message,
            result,
        )
        output_items = self._extract_output_items(
            result,
            start_index=output_start_index,
        )

        response_data = {
            "id": response_id,
            "object": "response",
            "status": "completed",
            "created_at": created_at,
            "model": model,
            "output": output_items,
            "usage": {
                "input_tokens": usage.get("input_tokens", 0),
                "output_tokens": usage.get("output_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0),
            },
        }
        if background:
            response_data["background"] = True

        return response_data, full_history, effective_session_id

    # ------------------------------------------------------------------
    # Agent execution
    # ------------------------------------------------------------------

    def _concurrency_limited_response(self) -> Optional["web.Response"]:
        """Return a 429 response if the concurrent-run cap is reached, else None.

        The cap bounds total in-flight agent activity across every
        agent-serving endpoint. Reuse the same adapter-owned work count that
        shutdown draining uses, including an admitted request before it reaches
        agent/task bookkeeping. Stream queues are transport state and may
        disappear while their underlying run remains active, so they must not
        define run concurrency. A configured value of 0 disables the cap.
        """
        limit = self._max_concurrent_runs
        if limit <= 0:
            return None
        inflight = self.active_agent_work_count()
        # The current request owns one reservation until it hands off to
        # _run_agent() or /v1/runs task registration. It must not consume its
        # own last available slot; other admitted requests remain counted.
        reservation = _api_agent_request_reservation.get()
        if reservation and reservation["active"]:
            inflight -= 1
        if inflight >= limit:
            return web.json_response(
                _openai_error(
                    f"Too many concurrent runs (max {limit})",
                    err_type="rate_limit_error",
                    code="rate_limit_exceeded",
                ),
                status=429,
                headers={"Retry-After": "1"},
            )
        return None

    @staticmethod
    def _bind_api_server_session(
        *,
        chat_id: str = "",
        session_key: str = "",
        session_id: str = "",
    ) -> list:
        """Bind session contextvars for an API-server agent run.

        This is the SINGLE structural chokepoint every API-server agent-entry
        path must use to seed session context — it hardwires
        ``platform="api_server"`` and ``async_delivery=False`` so a new route
        physically cannot reintroduce the silent-no-op bug (#10760) by
        forgetting to mark the channel as non-delivering. There is no
        ``async_delivery`` parameter to get wrong; the stateless HTTP path can
        never wake the agent after the turn ends, on ANY route.

        Returns reset tokens; pass them to ``clear_session_vars`` in a
        ``finally`` block (the binding is request-scoped and must not outlive
        the turn — a session resumed later on a delivering interface, e.g. the
        CLI or a gateway platform, re-binds fresh and is NOT blocked).
        """
        from gateway.session_context import set_session_vars

        return set_session_vars(
            platform="api_server",
            chat_id=chat_id,
            session_key=session_key,
            session_id=session_id,
            async_delivery=False,
            cron_session="",
        )

    async def _run_agent(
        self,
        user_message: str,
        conversation_history: List[Dict[str, str]],
        ephemeral_system_prompt: Optional[str] = None,
        session_id: Optional[str] = None,
        stream_delta_callback=None,
        tool_progress_callback=None,
        tool_start_callback=None,
        tool_complete_callback=None,
        agent_ref: Optional[list] = None,
        abort_event: Optional[threading.Event] = None,
        active_run_id: Optional[str] = None,
        gateway_session_key: Optional[str] = None,
        requested_model: Optional[str] = None,
        requested_provider: Optional[str] = None,
        model_options: Optional[Dict[str, Any]] = None,
        route: Optional[Dict[str, Any]] = None,
        session_model: Optional[str] = None,
        requested_runtime: Optional[Dict[str, Any]] = None,
        route_source: str = "global",
        confirmed_runtime_lock: bool = False,
    ) -> tuple:
        """
        Create an agent and run a conversation in a thread executor.

        Returns ``(result_dict, usage_dict)`` where *usage_dict* contains
        ``input_tokens``, ``output_tokens`` and ``total_tokens``.

        *route* is an optional ``model_routes`` entry (resolved from the
        request's ``model`` field) that overrides the global model/provider
        for this specific request.

        *session_model* is a raw model persisted on a native API session
        row.  It is used only when the persisted value did not resolve to a
        ``model_routes`` alias — see ``_create_agent`` for precedence.

        *requested_runtime* / *route_source* / *confirmed_runtime_lock*
        carry the Browser model-lock contract: when a confirmed lock is
        active the completed agent's actual provider/model must match the
        locked selection or the turn fails, and the response carries
        sanitized ``runtime`` metadata reporting actual vs requested.

        If *agent_ref* is a one-element list, the AIAgent instance is stored
        at ``agent_ref[0]`` before ``run_conversation`` begins.  This allows
        callers (e.g. the SSE writer) to call ``agent.interrupt()`` from
        another thread to stop in-progress LLM calls.

        If *active_run_id* is supplied, the same live agent is registered in
        ``_active_run_agents`` while the turn is running so API clients can
        call run-scoped control endpoints such as ``/v1/runs/{run_id}/steer``.
        """
        loop = asyncio.get_running_loop()
        # Capture before hopping to the executor — ContextVars do not follow
        # run_in_executor threads, so the profile scope must be re-entered
        # inside _run() from this explicit value.
        request_profile = _api_request_profile.get()
        # Inactivity-watchdog lifecycle for this turn (see the tail of this
        # method).  The holders are filled by _run() the moment the agent
        # exists, because only two callers pass ``agent_ref`` and the watchdog
        # must be able to poll the live agent on every path.
        turn_agent_holder: list = [None]
        turn_process_epoch: list = [None]

        def _run():
            from gateway.session_context import clear_session_vars

            with self._profile_scope(request_profile):
                tokens = self._bind_api_server_session(
                    chat_id=session_id or "",
                    session_key=gateway_session_key or session_id or "",
                    session_id=session_id or "",
                )
                agent = None
                try:
                    agent = self._create_agent(
                        ephemeral_system_prompt=ephemeral_system_prompt,
                        session_id=session_id,
                        stream_delta_callback=stream_delta_callback,
                        tool_progress_callback=tool_progress_callback,
                        tool_start_callback=tool_start_callback,
                        tool_complete_callback=tool_complete_callback,
                        gateway_session_key=gateway_session_key,
                        requested_model=requested_model,
                        requested_provider=requested_provider,
                        model_options=model_options,
                        route=route,
                        session_model=session_model,
                        confirmed_runtime_lock=confirmed_runtime_lock,
                    )
                    # Publish the live agent to the inactivity watchdog before
                    # anything can block: it reads seconds_since_activity off
                    # this holder, and a turn that never publishes an agent can
                    # never be judged inactive.
                    turn_agent_holder[0] = agent
                    if agent_ref is not None:
                        agent_ref[0] = agent
                    if active_run_id:
                        self._active_run_agents[active_run_id] = agent
                    if abort_event is not None and abort_event.is_set():
                        request_hard_interrupt(
                            agent, "Owner conversation reservation was lost",
                        )
                        raise RuntimeError("owner conversation reservation changed")
                    effective_task_id = session_id or str(uuid.uuid4())
                    # Baseline for selective background-process reaping on
                    # SSE client disconnect — mirrors gateway/run.py's
                    # gateway-turn cleanup (#76115); this API-server surface
                    # runs its own agent lifecycle and doesn't go through
                    # TurnRunner, so it needs its own baseline.
                    _publish_turn_process_ownership(agent, effective_task_id)
                    # Same epoch gate the disconnect reaper uses: a watchdog
                    # reap that lands after a NEWER run claimed this task_id
                    # must decline rather than kill that newer run's process.
                    turn_process_epoch[0] = getattr(
                        agent, "_gateway_turn_process_epoch", None,
                    )
                    # Shutdown interrupt coverage (#63529).  Registering here,
                    # once, covers every _run_agent() caller — the same reason
                    # the _ProviderAuthResolutionError handler below lives here
                    # rather than in each route.  Only two callers pass
                    # ``agent_ref``, and only /v1/runs has a run_id, so neither
                    # is a usable hook for the rest.
                    self._shutdown_interruptible_agents[id(agent)] = agent
                    result = agent.run_conversation(
                        user_message=user_message,
                        conversation_history=conversation_history,
                        task_id=effective_task_id,
                    )
                    usage = {
                        "input_tokens": getattr(agent, "session_prompt_tokens", 0) or 0,
                        "output_tokens": getattr(agent, "session_completion_tokens", 0) or 0,
                        "total_tokens": getattr(agent, "session_total_tokens", 0) or 0,
                    }
                    # Include the effective session ID in the result so callers
                    # (e.g. X-Hermes-Session-Id header) can track compression-
                    # triggered session rotations. (#16938)
                    _eff_sid = getattr(agent, "session_id", session_id)
                    if isinstance(_eff_sid, str) and _eff_sid:
                        result["session_id"] = _eff_sid
                    # Signal whether context compression occurred during this turn
                    # so _build_response_conversation_history can skip the
                    # prior-concatenation path and store the compressed transcript
                    # directly.  Rotation mode changes agent.session_id; in-place
                    # mode sets _last_compaction_in_place (see #38763).
                    _compacted_in_place = bool(getattr(agent, "_last_compaction_in_place", False))
                    _session_rotated = (
                        isinstance(_eff_sid, str) and isinstance(session_id, str)
                        and _eff_sid != session_id
                    )
                    if _compacted_in_place or _session_rotated:
                        result["_compressed"] = True
                    include_runtime = bool(
                        requested_runtime
                        or route
                        or confirmed_runtime_lock
                        or (route_source and route_source != "global")
                    )
                    if include_runtime:
                        runtime = dict(getattr(agent, "_hermes_api_runtime", {}) or {})
                        raw_provider = getattr(agent, "provider", "")
                        raw_model = getattr(agent, "model", "")
                        actual_provider = (
                            self._clean_runtime_id(raw_provider, max_len=80)
                            if isinstance(raw_provider, str)
                            else ""
                        )
                        actual_model = (
                            self._clean_runtime_id(raw_model)
                            if isinstance(raw_model, str)
                            else ""
                        )
                        if actual_provider:
                            runtime["provider"] = actual_provider
                        else:
                            runtime.setdefault("provider", "")
                        if actual_model:
                            runtime["model"] = actual_model
                        else:
                            runtime.setdefault("model", "")
                        if confirmed_runtime_lock:
                            expected_provider = self._clean_runtime_id(
                                (route or {}).get("provider")
                                or (requested_runtime or {}).get("provider"),
                                max_len=80,
                            )
                            expected_model = self._clean_runtime_id(
                                (route or {}).get("model")
                                or (requested_runtime or {}).get("model")
                            )
                            mismatched = (
                                (expected_provider and actual_provider != expected_provider)
                                or (expected_model and actual_model != expected_model)
                            )
                            if mismatched:
                                raise RuntimeError(
                                    "confirmed model lock runtime mismatch: "
                                    f"expected provider={expected_provider or '<unspecified>'} "
                                    f"model={expected_model or '<unspecified>'}; "
                                    f"actual provider={actual_provider or '<unknown>'} "
                                    f"model={actual_model or '<unknown>'}"
                                )
                        if requested_runtime:
                            runtime["requested"] = {
                                "provider": self._clean_runtime_id((requested_runtime or {}).get("provider"), max_len=80),
                                "model": self._clean_runtime_id((requested_runtime or {}).get("model")),
                            }
                        runtime["route_source"] = route_source or runtime.get("route_source") or "global"
                        runtime = self._sanitize_runtime_metadata(
                            runtime=runtime,
                            requested_runtime=requested_runtime,
                            route_source=route_source or "global",
                            model_lock=("confirmed" if confirmed_runtime_lock else ""),
                        )
                        if isinstance(result, dict):
                            result["runtime"] = runtime
                        usage["runtime"] = runtime
                    return result, usage
                except _ProviderAuthResolutionError as exc:
                    # Only _ProviderAuthResolutionError — raised exclusively
                    # where _resolve_runtime_agent_kwargs() is called inside
                    # _create_agent() — means a provider auth/credential
                    # failure.  Catching bare RuntimeError here would
                    # mislabel unrelated RuntimeErrors from
                    # run_conversation() (e.g. "Failed to recreate closed
                    # OpenAI client") as auth failures.  Matches run.py's
                    # response shape (final_response text, no HTTP error).
                    # Previously this propagated unhandled:
                    # /v1/chat/completions caught it as an undifferentiated
                    # "Internal server error" 500, and
                    # /api/sessions/{id}/chat[/stream] didn't catch it at
                    # all (raw aiohttp 500, no JSON body).  Handling it
                    # here, once, covers every _run_agent() caller;
                    # /v1/runs has its own branch in its executor.
                    logger.warning("Provider authentication failed for session=%s: %s",
                                   session_id or "", exc)
                    return (
                        {
                            "final_response": f"⚠️ Provider authentication failed: {exc}",
                            "messages": [],
                            "api_calls": 0,
                            "tools": [],
                        },
                        {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
                    )
                finally:
                    # Turn finished (success, auth failure, or crash) — clear
                    # ownership markers so a disconnect landing after this
                    # point can't reap background work this turn left
                    # running on purpose. Mirrors the same race-window guard
                    # in gateway/run.py's _run_sync_with_timeout_lifecycle.
                    if active_run_id:
                        self._active_run_agents.pop(active_run_id, None)
                    if agent is not None:
                        _clear_turn_process_ownership(agent)
                        # Symmetric with the registration above: the turn is
                        # over, so it must not be interrupted by a later
                        # shutdown.  pop() is a no-op when _create_agent
                        # succeeded but the turn never reached registration.
                        self._shutdown_interruptible_agents.pop(id(agent), None)
                    clear_session_vars(tokens)

        # Bound the executor await with the gateway's EXISTING inactivity
        # lifecycle (agent.gateway_timeout / HERMES_AGENT_TIMEOUT).  Not a
        # wall-clock limit: a turn that is actively streaming or calling tools
        # runs as long as it likes; only genuine inactivity is abandoned.
        # Without this the await is unbounded, so a never-returning worker
        # parks _run_background_response at its `await _compute_response()`
        # forever and the owner response never becomes terminal.
        from gateway.run import _float_env, _watch_gateway_turn_inactivity
        from tools.process_registry import process_registry

        _raw_turn_timeout = _float_env("HERMES_AGENT_TIMEOUT", 1800)
        # <= 0 is the operator opting into unbounded turns; the watchdog is
        # disabled entirely, exactly as in gateway/run.py.
        turn_timeout = _raw_turn_timeout if _raw_turn_timeout > 0 else None
        # Session-scoped, like gateway/run.py's own turn task id. Blank for a
        # sessionless caller, which makes the reap a documented no-op rather
        # than a match against every empty-task process.
        turn_task_id = session_id or ""
        turn_worker_done = threading.Event()
        turn_timeout_fired = threading.Event()
        turn_cleanup_lock = threading.Lock()
        # A production timeout (1800s) polls at gateway/run.py's own 5s; a
        # small configured one has to poll proportionally faster or it could
        # not observe its own inactivity window at all.
        poll_interval = (
            min(5.0, max(0.02, turn_timeout / 4.0))
            if turn_timeout is not None else 5.0
        )
        # A background=true process intentionally survives a successful turn,
        # so only children this turn created may ever be reaped.
        turn_process_baseline = (
            process_registry.snapshot_running_ids(turn_task_id)
            if turn_timeout is not None else frozenset()
        )

        def _turn_is_still_current() -> bool:
            epoch = turn_process_epoch[0]
            if epoch is None:
                # The turn never claimed the task_id (it wedged before the
                # agent existed), so there is nothing of its own to protect.
                return True
            with _TURN_PROCESS_EPOCH_LOCK:
                current = _TURN_PROCESS_EPOCHS.get(turn_task_id)
            return current is None or current == epoch

        def _run_with_timeout_lifecycle():
            try:
                return _run()
            finally:
                # The instant real work is done, under the same
                # worker_done/timeout_fired/cleanup_lock protocol
                # _abandon_timed_out_gateway_turn arbitrates with: a worker
                # that finished can no longer be declared timed out.
                turn_worker_done.set()

        # Set from the watchdog thread once it has actually decided to abandon
        # this turn, so the await below wakes on a decided abandonment rather
        # than on a bare elapsed interval.
        timeout_signal = asyncio.Event()

        def _watch_turn() -> None:
            try:
                _watch_gateway_turn_inactivity(
                    agent_holder=turn_agent_holder,
                    task_id=turn_task_id,
                    process_baseline=turn_process_baseline,
                    timeout=turn_timeout,
                    worker_done=turn_worker_done,
                    timeout_fired=turn_timeout_fired,
                    cleanup_lock=turn_cleanup_lock,
                    poll_interval=poll_interval,
                    is_still_current=_turn_is_still_current,
                )
            finally:
                # Only a real fire wakes the awaiting turn. Losing the
                # worker_done tiebreak inside _abandon_timed_out_gateway_turn
                # leaves timeout_fired clear, and the worker keeps the floor.
                if turn_timeout_fired.is_set():
                    try:
                        loop.call_soon_threadsafe(timeout_signal.set)
                    except RuntimeError:  # pragma: no cover - loop closed
                        pass

        def _swallow_abandoned_result(future: "asyncio.Future") -> None:
            """Retire an abandoned worker's late outcome without publishing it.

            The turn already ended without this result, so it is fenced out by
            construction — nothing reads the future any more. Consuming it only
            keeps asyncio from reporting a never-retrieved exception for a turn
            that was deliberately abandoned.
            """
            try:
                future.result()
            except BaseException:
                pass
            if turn_timeout_fired.is_set():
                logger.warning(
                    "[api_server] worker for session=%s returned after the "
                    "inactivity watchdog abandoned its turn; the late result "
                    "is discarded",
                    session_id or "",
                )
            else:
                # Ordinary cancellation (an SSE client disconnecting, a
                # shutdown) also leaves a worker nobody reads any more.
                logger.debug(
                    "[api_server] worker for session=%s returned after its "
                    "turn was already over",
                    session_id or "",
                )

        watchdog: Optional[threading.Thread] = None
        worker_future: "Optional[asyncio.Future]" = None
        self._activate_admitted_request()
        self._inflight_agent_runs += 1
        try:
            if turn_timeout is not None:
                watchdog = threading.Thread(
                    target=_watch_turn,
                    name=f"api-turn-watchdog-{(turn_task_id or 'anon')[:12]}",
                    daemon=True,
                )
                watchdog.start()

            worker_future = loop.run_in_executor(None, _run_with_timeout_lifecycle)
            if turn_timeout is None:
                return await worker_future

            fired_waiter = asyncio.ensure_future(timeout_signal.wait())
            try:
                await asyncio.wait(
                    {worker_future, fired_waiter},
                    return_when=asyncio.FIRST_COMPLETED,
                )
            finally:
                fired_waiter.cancel()
            if worker_future.done():
                # Worker-result-wins: a real answer that landed in the same
                # window as the watchdog's decision is still the answer, and is
                # never thrown away for a racing timeout. This is the asyncio
                # side of _abandon_timed_out_gateway_turn's own tiebreak.
                return worker_future.result()
            # The watchdog fired and the cooperative interrupt did not free the
            # worker. Its thread cannot be killed, so abandon the future and
            # surface the abandonment: the caller must reach a terminal state
            # instead of waiting on a worker that may never return.
            _idle_seconds = turn_timeout
            _watchdog_agent = turn_agent_holder[0]
            if _watchdog_agent is not None:
                try:
                    _idle_seconds = float(
                        _watchdog_agent.get_activity_summary().get(
                            "seconds_since_activity", turn_timeout,
                        )
                    )
                except Exception:
                    pass
            logger.error(
                "[api_server] abandoning wedged turn for session=%s after "
                "%.0fs without activity (timeout %.0fs)",
                session_id or "", _idle_seconds, turn_timeout,
            )
            raise _AgentTurnInactivityTimeout(
                f"Agent was inactive for {int(turn_timeout)}s and the turn "
                "was interrupted"
            )
        finally:
            # Settle the watchdog on every path — normal return, timeout,
            # cancellation, crash — so no watchdog thread outlives its turn.
            # Setting worker_done also closes the door on a late second fire.
            turn_worker_done.set()
            if watchdog is not None:
                # It leaves worker_done.wait() the instant the event is set;
                # the bound only covers a reap still in flight, which must not
                # park the event loop.
                watchdog.join(timeout=0.1)
            if worker_future is not None and not worker_future.done():
                # Timed out or cancelled: this turn is over and nobody will
                # read the worker's eventual outcome again.
                worker_future.add_done_callback(_swallow_abandoned_result)
            # Exactly one decrement per increment above, on every path — an
            # abandoned worker thread is not a second run to account for.
            self._inflight_agent_runs -= 1

    # ------------------------------------------------------------------
    # /v1/runs — structured event streaming
    # ------------------------------------------------------------------

    _RUN_STREAM_TTL = 300  # seconds before orphaned runs are swept
    _RUN_STATUS_TTL = 3600  # seconds to retain terminal run status for polling

    def _set_run_status(self, run_id: str, status: str, **fields: Any) -> Dict[str, Any]:
        """Update pollable run status without exposing private agent objects."""
        now = time.time()
        current = self._run_statuses.get(run_id, {})
        current.update({
            "object": "hermes.run",
            "run_id": run_id,
            "status": status,
            "updated_at": now,
        })
        current.setdefault("created_at", fields.pop("created_at", now))
        current.update(fields)
        if status != "waiting_for_approval":
            current.pop("pending_approval", None)
        self._run_statuses[run_id] = current
        return current

    def _make_run_event_callback(self, run_id: str, loop: "asyncio.AbstractEventLoop"):
        """Return a tool_progress_callback that pushes structured events to the run's SSE queue."""
        def _push(event: Dict[str, Any]) -> None:
            self._set_run_status(
                run_id,
                self._run_statuses.get(run_id, {}).get("status", "running"),
                last_event=event.get("event"),
            )
            q = self._run_streams.get(run_id)
            if q is None:
                return
            try:
                loop.call_soon_threadsafe(q.put_nowait, event)
            except Exception:
                pass

        def _callback(event_type: str, tool_name: str = None, preview: str = None, args=None, **kwargs):
            ts = time.time()
            if event_type == "tool.started":
                _push({
                    "event": "tool.started",
                    "run_id": run_id,
                    "timestamp": ts,
                    "tool": tool_name,
                    "preview": preview,
                })
            elif event_type == "tool.completed":
                _push({
                    "event": "tool.completed",
                    "run_id": run_id,
                    "timestamp": ts,
                    "tool": tool_name,
                    "duration": round(kwargs.get("duration", 0), 3),
                    "error": kwargs.get("is_error", False),
                })
            elif event_type == "reasoning.available":
                _push({
                    "event": "reasoning.available",
                    "run_id": run_id,
                    "timestamp": ts,
                    "text": preview or "",
                })
            elif event_type in {"subagent.start", "subagent.complete"}:
                event = {
                    "event": event_type,
                    "run_id": run_id,
                    "timestamp": ts,
                }
                if preview is not None:
                    event["preview"] = redact_sensitive_text(
                        str(preview), force=True
                    )
                for key in (
                    "goal",
                    "task_count",
                    "task_index",
                    "subagent_id",
                    "child_session_id",
                    "parent_id",
                    "depth",
                    "model",
                    "tool_count",
                    "status",
                    "summary",
                    "duration_seconds",
                    "input_tokens",
                    "output_tokens",
                    "reasoning_tokens",
                    "api_calls",
                    "cost_usd",
                    "files_read",
                    "files_written",
                    "output_tail",
                ):
                    value = kwargs.get(key)
                    if value is None:
                        continue
                    # Free-text fields can carry child terminal/tool output —
                    # force the same secret redaction the API applies to error
                    # text before it leaves the process on a public stream.
                    if key in ("goal", "summary", "output_tail") and isinstance(
                        value, str
                    ):
                        value = redact_sensitive_text(value, force=True)
                    event[key] = value
                _push(event)
            # _thinking, subagent.tool, and subagent_progress are intentionally
            # not forwarded on the /v1/runs stream: they are high-volume UI
            # noise. Lifecycle boundaries (start/complete) still need to land
            # so clients can observe delegate_task timeouts and failures.

        return _callback


    async def _handle_owner_workspace_projects(
        self, request: "web.Request",
    ) -> "web.Response":
        """GET receipt-backed owner Projects without opening a mutation path."""
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err

        try:
            from gateway.run import _load_gateway_config

            if not _owner_workspace_toolset_enabled(_load_gateway_config()):
                return web.json_response(
                    _openai_error(
                        "Owner workspace is not enabled for this profile",
                        code="owner_workspace_not_enabled",
                    ),
                    status=404,
                )
            from hermes_cli.owner_workspace import (
                OWNER_PROJECT_LIFECYCLE_REVISION_CAPABILITY,
                list_committed_projects,
                resolve_owner_context,
            )

            projects = list_committed_projects(
                resolve_owner_context(),
                lifecycle_revision=_owner_workspace_capability_requested(
                    request, OWNER_PROJECT_LIFECYCLE_REVISION_CAPABILITY,
                ),
            )
        except Exception:
            logger.exception("[api_server] owner-workspace Project projection failed")
            return web.json_response(
                _openai_error(
                    "Owner workspace Project projection is unavailable",
                    code="owner_workspace_unavailable",
                ),
                status=500,
            )

        return web.json_response({
            "object": "hermes.owner_workspace.project_list",
            "data": projects,
        })
    async def _handle_owner_workspace_project_snapshot(
        self, request: "web.Request",
    ) -> "web.Response":
        """GET one exact receipt-backed Project through the executor boundary."""
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err
        from hermes_cli.owner_workspace import OwnerWorkspaceError

        try:
            from gateway.run import _load_gateway_config

            if not _owner_workspace_toolset_enabled(_load_gateway_config()):
                return web.json_response(
                    _openai_error(
                        "Owner workspace is not enabled for this profile",
                        code="owner_workspace_not_enabled",
                    ),
                    status=404,
                )
            from hermes_cli.owner_workspace import (
                OWNER_PROJECT_RUN_CONTEXT_CAPABILITY,
                read_project_snapshot,
                resolve_owner_context,
            )

            snapshot = read_project_snapshot(
                resolve_owner_context(),
                request.match_info.get("project_slug", ""),
                run_context=_owner_workspace_capability_requested(
                    request, OWNER_PROJECT_RUN_CONTEXT_CAPABILITY
                ),
            )
        except OwnerWorkspaceError as exc:
            if exc.code == "project_not_found":
                return web.json_response(
                    _openai_error(
                        "Owner workspace Project was not found",
                        code="project_not_found",
                    ),
                    status=404,
                )
            logger.warning(
                "[api_server] owner-workspace Project snapshot unavailable (%s)",
                exc.code,
            )
            return web.json_response(
                _openai_error(
                    "Owner workspace Project snapshot is unavailable",
                    code="owner_workspace_unavailable",
                ),
                status=503,
            )
        except Exception:
            logger.exception("[api_server] owner-workspace Project snapshot failed")
            return web.json_response(
                _openai_error(
                    "Owner workspace Project snapshot is unavailable",
                    code="owner_workspace_unavailable",
                ),
                status=500,
            )

        return web.json_response({
            "object": "hermes.owner_workspace.project_snapshot",
            "data": snapshot,
        })
    async def _handle_owner_workspace_project_attachment(
        self, request: "web.Request",
    ) -> "web.Response":
        """GET one bounded attachment from one exact receipt-backed Project."""
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err
        from hermes_cli.owner_workspace import OwnerWorkspaceError

        try:
            from gateway.run import _load_gateway_config

            if not _owner_workspace_toolset_enabled(_load_gateway_config()):
                return web.json_response(
                    _openai_error(
                        "Owner workspace is not enabled for this profile",
                        code="owner_workspace_not_enabled",
                    ),
                    status=404,
                )
            from hermes_cli.owner_workspace import (
                read_project_attachment,
                resolve_owner_context,
            )

            attachment = read_project_attachment(
                resolve_owner_context(),
                request.match_info.get("project_slug", ""),
                request.match_info.get("attachment_id", ""),
            )
        except OwnerWorkspaceError as exc:
            if exc.code in {"attachment_not_found", "project_not_found"}:
                return web.json_response(
                    _openai_error(
                        "Owner workspace attachment was not found",
                        code="attachment_not_found",
                    ),
                    status=404,
                )
            logger.warning(
                "[api_server] owner-workspace attachment unavailable (%s)", exc.code
            )
            return web.json_response(
                _openai_error(
                    "Owner workspace attachment is unavailable",
                    code="owner_workspace_unavailable",
                ),
                status=503,
            )
        except Exception:
            logger.exception("[api_server] owner-workspace attachment failed")
            return web.json_response(
                _openai_error(
                    "Owner workspace attachment is unavailable",
                    code="owner_workspace_unavailable",
                ),
                status=500,
            )

        return web.Response(
            body=attachment["body"],
            content_type=attachment["media_type"],
            headers={
                "Content-Disposition": f'attachment; filename="{attachment["filename"]}"',
                "Cache-Control": "no-store",
            },
        )
    async def _handle_owner_workspace_decisions(
        self, request: "web.Request",
    ) -> "web.Response":
        """GET owner-safe pending gates without creating decision authority."""
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err

        try:
            from gateway.run import _load_gateway_config

            if not _owner_workspace_toolset_enabled(_load_gateway_config()):
                return web.json_response(
                    _openai_error(
                        "Owner workspace is not enabled for this profile",
                        code="owner_workspace_not_enabled",
                    ),
                    status=404,
                )
            from hermes_cli.owner_workspace import (
                list_owner_decisions,
                resolve_owner_context,
            )

            owner = resolve_owner_context()
            projected = list_owner_decisions(owner)
            decisions = list(projected["data"])
            truncated = bool(projected["truncated"])
            owner_profile = str(owner.profile)
        except Exception:
            logger.exception("[api_server] owner-workspace decision projection failed")
            return web.json_response(
                _openai_error(
                    "Owner workspace decision projection is unavailable",
                    code="owner_workspace_unavailable",
                ),
                status=500,
            )

        operation_titles = {
            "owner_workspace_bootstrap": "Approve the new Project",
            "owner_task_graph_commit": "Approve the first Project milestone",
            "owner_project_plan_commit": "Approve Project changes",
            "owner_task_move": "Approve a work-state change",
            "owner_task_comment": "Approve an owner reply",
            "owner_project_lifecycle": "Approve the Project lifecycle change",
        }
        for run_id, status in self._run_statuses.items():
            context = status.get("owner_workspace_context")
            pending = status.get("pending_approval")
            if (
                status.get("status") != "waiting_for_approval"
                or not isinstance(context, dict)
                or context.get("profile") != owner_profile
                or not isinstance(pending, dict)
            ):
                continue
            operation = str(pending.get("operation") or "")
            title = operation_titles.get(operation)
            if title is None:
                continue
            created_at = status.get("created_at")
            try:
                created_iso = time.strftime(
                    "%Y-%m-%dT%H:%M:%SZ", time.gmtime(float(created_at))
                )
            except (OSError, OverflowError, TypeError, ValueError):
                created_iso = None
            decisions.append({
                "decision_ref": "decision_" + hashlib.sha256(
                    f"{owner_profile}\x00run\x00{run_id}".encode("utf-8")
                ).hexdigest()[:32],
                "authority": "run",
                "kind": "run_approval",
                "project_slug": context.get("project_slug"),
                "project_name": context.get("project_name"),
                "title": title,
                "reason": (
                    "Raphael is waiting for your confirmation before "
                    "changing this Project."
                ),
                "created_at": created_iso,
            })

        return web.json_response({
            "object": "hermes.owner_workspace.decision_list",
            # Truncation is reported, not hidden: an owner who cannot tell a
            # full inbox from a clipped one can believe they have answered
            # everything Raphael is waiting on.
            "truncated": truncated or len(decisions) > 100,
            "data": decisions[:100],
        })

    @staticmethod
    def _owner_authority_clean(value: Any) -> Any:
        """Mirror the Workspace proposal parser's semantic string trimming."""
        if isinstance(value, str):
            return value.strip()
        if isinstance(value, list):
            return [APIServerAdapter._owner_authority_clean(item) for item in value]
        if isinstance(value, dict):
            return {
                str(key): APIServerAdapter._owner_authority_clean(item)
                for key, item in value.items()
            }
        return value

    def _validated_owner_proposal_authority(
        self,
        authority: "dict[str, Any]",
        context: "dict[str, Any]",
        profile: str,
    ) -> "dict[str, Any]":
        """Derive mutation authority from the stored proposal and live Project.

        A conversation carrying a request that ended without a turn authorizes
        nothing here, which is the same fence the reservation transaction takes
        (see :meth:`ResponseStore.reserve_run_idempotency`). Refused at this end
        as well, before the stored proposal is read and long before anything is
        bound: an approval is not an answer to that request, and the run behind
        one consumes the proposal for good.
        """
        if context.get("profile") != profile:
            raise ValueError("owner profile mismatch")
        proposal_profile = authority["proposal_profile"]
        if self._response_store.owner_request_is_unanswered(
            proposal_profile, authority["conversation"],
        ):
            raise ValueError("an owner request on this conversation is unanswered")
        record = self._response_store.owner_proposal_record(
            proposal_profile, authority["conversation"], authority["response_id"],
        )
        if record is None:
            raise ValueError("owner proposal is not current")
        candidate, proposal_digest = record
        idempotency_key = (
            "conversation-"
            + hashlib.sha256(authority["response_id"].encode("utf-8")).hexdigest()
        )
        if authority["idempotency_key"] != idempotency_key:
            raise ValueError("owner idempotency key mismatch")

        clean = self._owner_authority_clean
        if candidate.get("kind") == "proposal":
            if (
                set(candidate) != _OWNER_NEW_PROPOSAL_KEYS
                or context.get("mode") != "new"
                or authority["operation"] != "owner_task_graph_commit"
                or context.get("project_slug") is not None
            ):
                raise ValueError("stored proposal does not authorize this operation")
            from hermes_cli.owner_workspace import (
                OwnerWorkspaceError,
                _native_owner_project_name,
                owner_project_name,
            )

            try:
                stored_project_name = owner_project_name(
                    _native_owner_project_name(
                        clean(candidate.get("project_name")), "project_name",
                    )
                )
            except OwnerWorkspaceError as exc:
                raise ValueError("stored proposal Project name is invalid") from exc
            if context.get("project_name") != stored_project_name:
                raise ValueError("owner Project context does not match the stored proposal")
            from hermes_cli.profiles import list_profiles

            profiles = [str(item.name) for item in list_profiles()]
            if not profiles:
                raise ValueError("no owner workspace profile is available")
            root_assignee = "default" if "default" in profiles else profiles[0]
            expected_payload: Dict[str, Any] = {
                "idempotency_key": idempotency_key,
                "mode": "new",
                "project_name": clean(candidate.get("project_name")),
                "project_description": clean(candidate.get("project_description")),
                "project_id": None,
                "request_title": clean(candidate.get("request_title")),
                "specification": clean(candidate.get("specification")),
                "current_milestone": clean(candidate.get("current_milestone")),
                "owner_visible_result": clean(candidate.get("owner_visible_result")),
                "root_assignee": root_assignee,
                "tasks": clean(candidate.get("tasks")),
                "later_milestones": clean(candidate.get("later_milestones")),
            }
        else:
            if (
                set(candidate) != _OWNER_EXISTING_PROPOSAL_KEYS
                or context.get("mode") != "existing"
                or authority["operation"] != "owner_project_plan_commit"
                or not isinstance(context.get("project_slug"), str)
            ):
                raise ValueError("stored proposal does not authorize this operation")
            from hermes_cli.owner_workspace import read_project_snapshot, resolve_owner_context

            snapshot = read_project_snapshot(
                resolve_owner_context(), str(context["project_slug"]),
            )
            project = snapshot.get("project")
            columns = snapshot.get("columns")
            if not isinstance(project, dict) or not isinstance(columns, list):
                raise ValueError("owner Project snapshot is unavailable")
            tasks = [
                task
                for column in columns if isinstance(column, dict)
                for task in column.get("tasks", []) if isinstance(task, dict)
            ]
            if not tasks:
                raise ValueError("owner Project has no task anchor")
            secret = self._expected_api_key()
            if not secret:
                raise ValueError("owner executor key is unavailable")
            native_by_ref: Dict[str, Dict[str, Any]] = {}
            for task in tasks:
                canonical = json.dumps(
                    {
                        "version": 1,
                        "project_id": project.get("id"),
                        "task_id": task.get("id"),
                        "title": task.get("title"),
                        "status": task.get("status") or next(
                            (
                                column.get("name") for column in columns
                                if isinstance(column, dict)
                                and task in column.get("tasks", [])
                            ),
                            None,
                        ),
                        "event_revision": task.get("event_revision"),
                        "parent_ids": sorted(task.get("parent_ids") or []),
                        "child_ids": sorted(task.get("child_ids") or []),
                    },
                    separators=(",", ":"),
                    ensure_ascii=False,
                )
                ref = "tr_" + base64.urlsafe_b64encode(
                    hmac.new(
                        secret.encode("utf-8"), canonical.encode("utf-8"), hashlib.sha256,
                    ).digest()
                ).decode("ascii").rstrip("=")
                native_by_ref[ref] = {
                    "task_id": task.get("id"),
                    "expected_status": json.loads(canonical)["status"],
                    "expected_revision": task.get("event_revision"),
                }

            def native(ref: Any) -> Dict[str, Any]:
                value = native_by_ref.get(str(ref))
                if value is None:
                    raise ValueError("proposal task reference is stale")
                return value

            changes: List[Dict[str, Any]] = []
            raw_changes = candidate.get("changes")
            if not isinstance(raw_changes, list) or not raw_changes:
                raise ValueError("stored proposal has no changes")
            for raw in raw_changes:
                if not isinstance(raw, dict):
                    raise ValueError("stored proposal change is invalid")
                action = raw.get("action")
                reason = clean(raw.get("reason"))
                if action == "add" and set(raw) == _OWNER_PROPOSAL_ADD_KEYS:
                    changes.append({
                        "action": "add", "reason": reason,
                        "title": clean(raw.get("title")), "body": clean(raw.get("body")),
                        "assignee": clean(raw.get("assignee")),
                        "responsibility": clean(raw.get("responsibility")),
                        "execution_tier": clean(raw.get("execution_tier")),
                        "existing_parents": [native(ref) for ref in raw["existing_parent_refs"]],
                        "new_parents": clean(raw.get("new_parents")),
                    })
                elif action == "replace" and set(raw) == {
                    "action", "reason", "target_ref", "replacement",
                }:
                    changes.append({
                        "action": "replace", "reason": reason,
                        "target": native(raw.get("target_ref")),
                        "replacement": clean(raw.get("replacement")),
                    })
                elif action == "split" and set(raw) == {
                    "action", "reason", "target_ref", "replacements",
                }:
                    changes.append({
                        "action": "split", "reason": reason,
                        "target": native(raw.get("target_ref")),
                        "replacements": clean(raw.get("replacements")),
                    })
                elif action == "merge" and set(raw) == {
                    "action", "reason", "target_refs", "replacement",
                }:
                    changes.append({
                        "action": "merge", "reason": reason,
                        "targets": [native(ref) for ref in raw["target_refs"]],
                        "replacement": clean(raw.get("replacement")),
                    })
                elif action == "move" and set(raw) == {
                    "action", "reason", "target_ref", "to_status",
                }:
                    changes.append({
                        "action": "move", "reason": reason,
                        "target": native(raw.get("target_ref")),
                        "to_status": clean(raw.get("to_status")),
                    })
                elif action in {"postpone", "cancel"} and set(raw) == {
                    "action", "reason", "target_ref",
                }:
                    changes.append({
                        "action": action, "reason": reason,
                        "target": native(raw.get("target_ref")),
                    })
                else:
                    raise ValueError("stored proposal change is invalid")
            # No anchor here: the Project's control anchor is resolved inside
            # Hermes from its committed bootstrap receipt. Deriving one from
            # this owner-visible task list could only ever name a work row.
            expected_payload = {
                "idempotency_key": idempotency_key,
                "project_id": project.get("id"),
                "trigger": "owner_request",
                "request_title": clean(candidate.get("request_title")),
                "summary": clean(candidate.get("summary")),
                "specification": clean(candidate.get("specification")),
                "current_milestone": clean(candidate.get("current_milestone")),
                "owner_visible_result": clean(candidate.get("owner_visible_result")),
                "later_milestones": clean(candidate.get("later_milestones")),
                "changes": changes,
            }

        if clean(authority["payload"]) != expected_payload:
            raise ValueError("run payload differs from the stored owner proposal")
        canonical = json.dumps(
            expected_payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        )
        return {
            "proposal_profile": proposal_profile,
            "conversation": authority["conversation"],
            "response_id": authority["response_id"],
            "claim_id": authority["claim_id"],
            "operation": authority["operation"],
            "idempotency_key": idempotency_key,
            "payload_digest": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
            "proposal_digest": proposal_digest,
            "payload": copy.deepcopy(expected_payload),
        }

    def _validated_owner_lifecycle_authority(
        self,
        authority: "dict[str, Any]",
        context: "dict[str, Any]",
        profile: str,
    ) -> "dict[str, Any]":
        """Bind one lifecycle run to exact receipt-backed Project state."""
        if (
            context.get("profile") != profile
            or context.get("mode") != "existing"
            or not isinstance(context.get("project_slug"), str)
        ):
            raise ValueError("owner lifecycle context mismatch")
        payload = authority["payload"]
        project_id = payload.get("project_id")
        expected_revision = payload.get("expected_revision")
        action = payload.get("action")
        if (
            not isinstance(project_id, str)
            or not project_id
            or isinstance(expected_revision, bool)
            or not isinstance(expected_revision, int)
            or expected_revision < 0
            or action not in {"archive", "restore", "pause", "resume"}
        ):
            raise ValueError("owner lifecycle payload is invalid")

        from hermes_cli.owner_workspace import (
            list_committed_projects,
            resolve_owner_context,
        )

        projects = list_committed_projects(
            resolve_owner_context(), lifecycle_revision=True,
        )
        matches = [
            project for project in projects
            if project.get("slug") == context["project_slug"]
            and project.get("project_id") == project_id
            and project.get("lifecycle_revision") == expected_revision
        ]
        if len(matches) != 1:
            raise ValueError("owner lifecycle Project state is stale")
        canonical = json.dumps(
            payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        )
        return {
            "operation": "owner_project_lifecycle",
            "payload": copy.deepcopy(payload),
            "idempotency_key": authority["idempotency_key"],
            "payload_digest": hashlib.sha256(
                canonical.encode("utf-8")
            ).hexdigest(),
        }

    def _owner_authority_digest_for_recovery(
        self, authority: "dict[str, Any]",
    ) -> str:
        """Rebuild the exact digest the validated run bound to its tool call."""
        payload = self._owner_authority_clean(authority["payload"])
        canonical = json.dumps(
            payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def _recover_native_owner_completion(
        self,
        *,
        profile: str,
        session_scope: str,
        run_id: str,
        authority: "dict[str, Any]",
        owner: "dict[str, str] | None",
        authority_digest: Optional[str] = None,
    ) -> "Dict[str, Any] | None":
        """Close the cross-database crash window from a bound native receipt.

        ``None`` means this exact run provably committed nothing. Anything that
        leaves the external effect UNDECIDED — a receipt that cannot be read,
        more than one committed receipt claiming the same run, a receipt whose
        minimal projection fails, or an error while persisting the recovered
        completion — raises :class:`_OwnerNativeReceiptUnreadable`. Answering
        "nothing committed" for those let a completed change be reported as
        failed and its approval released for a second run.
        """
        from hermes_cli.owner_workspace import (
            OwnerReceiptUnreadable,
            read_committed_owner_run_receipt,
        )

        try:
            if authority_digest is None:
                authority_digest = self._owner_authority_digest_for_recovery(
                    authority
                )
            receipt = read_committed_owner_run_receipt(
                profile=profile,
                idempotency_key=authority["idempotency_key"],
                operation=authority["operation"],
                authority_digest=authority_digest,
            )
        except OwnerReceiptUnreadable as exc:
            logger.error(
                "[api_server] native owner receipt for run=%s is unreadable: %s",
                run_id, exc,
            )
            raise _OwnerNativeReceiptUnreadable(str(exc)) from exc
        except Exception as exc:
            logger.exception(
                "[api_server] native owner receipt for run=%s could not be read",
                run_id,
            )
            raise _OwnerNativeReceiptUnreadable(
                "the native owner receipt could not be read"
            ) from exc
        if receipt is None:
            return None
        try:
            minimal_receipt = self._owner_mutation_receipt(
                authority["operation"], receipt,
            )
            if minimal_receipt is None:
                raise _OwnerNativeReceiptUnreadable(
                    "a committed native receipt could not be projected"
                )
            created_at = self._response_store.run_idempotency_created_at(
                profile,
                session_scope,
                authority["idempotency_key"],
                run_id,
            )
            if created_at is None:
                raise _OwnerNativeReceiptUnreadable(
                    "a committed native receipt has no run identity to bind to"
                )
            completion_owner = dict(owner) if owner is not None else None
            if completion_owner is not None:
                completion_owner["payload_digest"] = authority_digest
            return self._response_store.persist_owner_run_completion(
                profile,
                session_scope,
                authority["idempotency_key"],
                run_id,
                minimal_receipt,
                created_at=created_at,
                owner=completion_owner,
            )
        except _OwnerNativeReceiptUnreadable:
            raise
        except Exception as exc:
            logger.exception(
                "[api_server] native owner completion recovery failed for run=%s",
                run_id,
            )
            raise _OwnerNativeReceiptUnreadable(
                "a committed native receipt could not be recorded"
            ) from exc

    @staticmethod
    def _owner_mutation_receipt(
        operation: str, result: Any,
    ) -> "str | None":
        """Return the minimal receipt for one exact successful owner mutation.

        The tool callback observes the native tool result, not the model's
        later prose. Denied, conflicting, malformed, or merely asserted
        results cannot become successful owner receipts.
        """
        if isinstance(result, str):
            try:
                value = json.loads(result)
            except (json.JSONDecodeError, TypeError):
                return None
        elif isinstance(result, dict):
            value = result
        else:
            return None
        if not isinstance(value, dict) or value.get("ok") is not True:
            return None
        project_slug = value.get("project_slug")
        if (
            not isinstance(project_slug, str)
            or re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", project_slug) is None
        ):
            return None
        if operation == "owner_task_graph_commit":
            task_count = value.get("task_count")
            if (
                isinstance(task_count, bool)
                or not isinstance(task_count, int)
                or not 1 <= task_count <= 12
            ):
                return None
            receipt = {
                "ok": True,
                "project_slug": project_slug,
                "task_count": task_count,
            }
        elif operation == "owner_project_plan_commit":
            change_count = value.get("change_count")
            if (
                value.get("applied") is not True
                or isinstance(change_count, bool)
                or not isinstance(change_count, int)
                or not 1 <= change_count <= 12
            ):
                return None
            receipt = {
                "ok": True,
                "project_slug": project_slug,
                "applied": True,
                "change_count": change_count,
            }
        elif operation == "owner_project_lifecycle":
            action = value.get("action")
            archived = value.get("archived")
            execution_paused = value.get("execution_paused")
            expected_state = {
                "archive": (True, True),
                "restore": (False, True),
                "pause": (False, True),
                "resume": (False, False),
            }.get(action)
            if (
                expected_state is None
                or type(archived) is not bool
                or type(execution_paused) is not bool
                or (archived, execution_paused) != expected_state
            ):
                return None
            receipt = {
                "ok": True,
                "action": action,
                "project_slug": project_slug,
                "archived": archived,
                "execution_paused": execution_paused,
            }
        else:
            return None
        return json.dumps(receipt, sort_keys=True, separators=(",", ":"))


    @_admit_api_agent_request
    async def _handle_runs(self, request: "web.Request") -> "web.Response":
        """POST /v1/runs — start an agent run, return run_id immediately."""
        # Long-term memory scope header (see chat_completions for details).
        gateway_session_key, key_err = self._parse_session_key_header(request)
        if key_err is not None:
            return key_err

        try:
            body = await request.json()
        except Exception:
            return web.json_response(_openai_error("Invalid JSON"), status=400)

        try:
            owner_workspace_context = _resolve_owner_workspace_run_context(
                body.get("owner_workspace_context")
            )
            owner_proposal_authority = _resolve_owner_proposal_run_authority(
                body.get("owner_proposal_authority")
            )
            owner_lifecycle_authority = _resolve_owner_lifecycle_run_authority(
                body.get("owner_lifecycle_authority")
            )
        except ValueError:
            return web.json_response(
                _openai_error("Invalid owner-workspace authority"), status=400
            )
        if (
            owner_proposal_authority is not None
            and owner_lifecycle_authority is not None
        ):
            return web.json_response(
                _openai_error("Owner authorities cannot be combined"), status=400
            )
        if (
            (owner_proposal_authority is not None or owner_lifecycle_authority is not None)
            and owner_workspace_context is None
        ):
            return web.json_response(
                _openai_error(
                    "Owner authority requires owner workspace context"
                ),
                status=400,
            )
        request_owner_profile = _active_owner_profile()
        if owner_proposal_authority is not None:
            # A transport retry can arrive after the native tool callback has
            # already consumed the exact proposal. Replaying the persisted,
            # fingerprint-matched run is safe; re-validating it as fresh
            # authority would incorrectly reject the successful retry merely
            # because consumption did its job.
            retry_key = request.headers.get("Idempotency-Key")
            if retry_key == owner_proposal_authority["idempotency_key"]:
                retry_scope = hashlib.sha256(
                    (gateway_session_key or "").encode("utf-8")
                ).hexdigest()
                retry_body = dict(body)
                retry_body["_session_scope"] = retry_scope
                retry_fingerprint = _make_request_fingerprint(
                    retry_body, keys=sorted(retry_body),
                )
                retry_state, retry_run_id = (
                    self._response_store.lookup_run_idempotency(
                        request_owner_profile,
                        retry_scope,
                        retry_key,
                        retry_fingerprint,
                    )
                )
                retry_status = (
                    self._response_store.owner_run_completion(
                        request_owner_profile, retry_run_id,
                    )
                    if retry_run_id is not None else None
                ) or self._run_statuses.get(retry_run_id or "") or (
                    self._response_store.run_idempotency_status(
                        request_owner_profile, retry_run_id,
                    )
                    if retry_run_id is not None else None
                ) or {}
                if (
                    retry_state == "existing"
                    and retry_run_id is not None
                    and not (
                        retry_status.get("status") == "completed"
                        and retry_status.get("owner_mutation_committed") is True
                    )
                ):
                    try:
                        retry_status = self._recover_native_owner_completion(
                            profile=request_owner_profile,
                            session_scope=retry_scope,
                            run_id=retry_run_id,
                            authority=owner_proposal_authority,
                            owner={
                                "proposal_profile": owner_proposal_authority[
                                    "proposal_profile"
                                ],
                                "conversation": owner_proposal_authority["conversation"],
                                "response_id": owner_proposal_authority["response_id"],
                                "claim_id": owner_proposal_authority["claim_id"],
                                "operation": owner_proposal_authority["operation"],
                            },
                        ) or retry_status
                    except _OwnerNativeReceiptUnreadable:
                        # Whether the first attempt's change committed cannot be
                        # decided, so this retry may neither replay it as
                        # completed nor start a second one.
                        return web.json_response(
                            _openai_error(
                                "The earlier attempt's outcome could not be "
                                "confirmed",
                                err_type="server_error",
                                code="owner_run_outcome_unconfirmed",
                            ),
                            status=503,
                        )
                if (
                    retry_state == "existing"
                    and retry_run_id is not None
                    and retry_status.get("status") == "completed"
                    and retry_status.get("owner_mutation_committed") is True
                ):
                    response_headers = (
                        {"X-Hermes-Session-Key": gateway_session_key}
                        if gateway_session_key else {}
                    )
                    return web.json_response(
                        {"run_id": retry_run_id, "status": "started"},
                        status=202,
                        headers=response_headers,
                    )
            try:
                owner_proposal_authority = self._validated_owner_proposal_authority(
                    owner_proposal_authority,
                    owner_workspace_context,
                    request_owner_profile,
                )
            except ValueError:
                return web.json_response(
                    _openai_error(
                        "Owner proposal does not authorize this run",
                        code="owner_proposal_authority_conflict",
                    ),
                    status=409,
                )
        if owner_lifecycle_authority is not None:
            # A completed lifecycle command changes the revision it was bound
            # to. An exact transport retry must therefore resolve from the
            # persisted terminal receipt before fresh-state validation.
            retry_key = request.headers.get("Idempotency-Key")
            if retry_key == owner_lifecycle_authority["idempotency_key"]:
                retry_scope = hashlib.sha256(
                    (gateway_session_key or "").encode("utf-8")
                ).hexdigest()
                retry_body = dict(body)
                retry_body["_session_scope"] = retry_scope
                retry_fingerprint = _make_request_fingerprint(
                    retry_body, keys=sorted(retry_body),
                )
                retry_state, retry_run_id = (
                    self._response_store.lookup_run_idempotency(
                        request_owner_profile,
                        retry_scope,
                        retry_key,
                        retry_fingerprint,
                    )
                )
                retry_status = (
                    self._response_store.owner_run_completion(
                        request_owner_profile, retry_run_id,
                    )
                    if retry_run_id is not None else None
                ) or self._run_statuses.get(retry_run_id or "") or (
                    self._response_store.run_idempotency_status(
                        request_owner_profile, retry_run_id,
                    )
                    if retry_run_id is not None else None
                ) or {}
                if (
                    retry_state == "existing"
                    and retry_run_id is not None
                    and not (
                        retry_status.get("status") == "completed"
                        and retry_status.get("owner_mutation_committed") is True
                    )
                ):
                    try:
                        retry_status = self._recover_native_owner_completion(
                            profile=request_owner_profile,
                            session_scope=retry_scope,
                            run_id=retry_run_id,
                            authority=owner_lifecycle_authority,
                            owner=None,
                        ) or retry_status
                    except _OwnerNativeReceiptUnreadable:
                        return web.json_response(
                            _openai_error(
                                "The earlier attempt's outcome could not be "
                                "confirmed",
                                err_type="server_error",
                                code="owner_run_outcome_unconfirmed",
                            ),
                            status=503,
                        )
                if (
                    retry_state == "existing"
                    and retry_run_id is not None
                    and retry_status.get("status") == "completed"
                    and retry_status.get("owner_mutation_committed") is True
                ):
                    response_headers = (
                        {"X-Hermes-Session-Key": gateway_session_key}
                        if gateway_session_key else {}
                    )
                    return web.json_response(
                        {"run_id": retry_run_id, "status": "started"},
                        status=202,
                        headers=response_headers,
                    )
        if owner_lifecycle_authority is not None:
            try:
                owner_lifecycle_authority = (
                    self._validated_owner_lifecycle_authority(
                        owner_lifecycle_authority,
                        owner_workspace_context,
                        request_owner_profile,
                    )
                )
            except ValueError:
                return web.json_response(
                    _openai_error(
                        "Owner lifecycle state does not authorize this run",
                        code="owner_lifecycle_authority_conflict",
                    ),
                    status=409,
                )
        owner_mutation_authority = (
            owner_proposal_authority or owner_lifecycle_authority
        )

        raw_input = body.get("input")
        if not raw_input:
            return web.json_response(_openai_error("Missing 'input' field"), status=400)

        user_message = raw_input if isinstance(raw_input, str) else (raw_input[-1].get("content", "") if isinstance(raw_input, list) else "")
        if not user_message:
            return web.json_response(_openai_error("No user message found in input"), status=400)

        instructions = body.get("instructions")
        previous_response_id = body.get("previous_response_id")

        # Accept explicit conversation_history from the request body.
        # Precedence: explicit conversation_history > previous_response_id.
        conversation_history: List[Dict[str, str]] = []
        raw_history = body.get("conversation_history")
        if raw_history:
            if not isinstance(raw_history, list):
                return web.json_response(
                    _openai_error("'conversation_history' must be an array of message objects"),
                    status=400,
                )
            for i, entry in enumerate(raw_history):
                if not isinstance(entry, dict) or "role" not in entry or "content" not in entry:
                    return web.json_response(
                        _openai_error(f"conversation_history[{i}] must have 'role' and 'content' fields"),
                        status=400,
                    )
                conversation_history.append({"role": str(entry["role"]), "content": str(entry["content"])})
            if previous_response_id:
                logger.debug("Both conversation_history and previous_response_id provided; using conversation_history")

        stored_session_id = None
        if not conversation_history and previous_response_id:
            stored = self._response_store.get(
                previous_response_id, profile=request_owner_profile,
            )
            if stored:
                conversation_history = list(stored.get("conversation_history", []))
                stored_session_id = stored.get("session_id")
                if instructions is None:
                    instructions = stored.get("instructions")

        # When input is a multi-message array, extract all but the last
        # message as conversation history (the last becomes user_message).
        # Only fires when no explicit history was provided.
        if not conversation_history and isinstance(raw_input, list) and len(raw_input) > 1:
            for msg in raw_input[:-1]:
                if isinstance(msg, dict) and msg.get("role") and msg.get("content"):
                    content = msg["content"]
                    if isinstance(content, list):
                        # Flatten multi-part content blocks to text
                        content = " ".join(
                            part.get("text", "") for part in content
                            if isinstance(part, dict) and part.get("type") == "text"
                        )
                    conversation_history.append({"role": msg["role"], "content": str(content)})

        session_id = body.get("session_id") or stored_session_id
        route = self._resolve_route(body.get("model"))
        agent_overrides = _request_agent_overrides(body, virtual_model=self._model_name)
        selection_error = self._request_route_conflict_error(
            session_id=session_id,
            gateway_session_key=gateway_session_key,
            requested_model=agent_overrides.get("requested_model"),
            requested_provider=agent_overrides.get("requested_provider"),
            route=route,
        )
        if selection_error:
            return web.json_response(_openai_error(selection_error), status=400)

        idempotency_key = request.headers.get("Idempotency-Key")
        if idempotency_key is not None and re.fullmatch(
            r"[A-Za-z0-9._:-]{1,200}", idempotency_key,
        ) is None:
            return web.json_response(
                _openai_error("Invalid Idempotency-Key"), status=400,
            )
        if owner_mutation_authority is not None and (
            idempotency_key != owner_mutation_authority["idempotency_key"]
        ):
            return web.json_response(
                _openai_error("Owner authority does not match this run"),
                status=400,
            )
        idempotency_fingerprint = None
        idempotency_session_scope = hashlib.sha256(
            (gateway_session_key or "").encode("utf-8")
        ).hexdigest()
        if idempotency_key is not None:
            fingerprint_body = dict(body)
            fingerprint_body["_session_scope"] = idempotency_session_scope
            idempotency_fingerprint = _make_request_fingerprint(
                fingerprint_body, keys=sorted(fingerprint_body),
            )
            state, existing_run_id = self._response_store.lookup_run_idempotency(
                request_owner_profile,
                idempotency_session_scope,
                idempotency_key,
                idempotency_fingerprint,
            )
            if state != "missing":
                if state == "conflict":
                    return web.json_response(
                        _openai_error(
                            "Idempotency-Key was already used for a different run",
                            code="idempotency_conflict",
                        ),
                        status=409,
                    )
                persisted_status = (
                    self._response_store.owner_run_completion(
                        request_owner_profile, existing_run_id,
                    )
                    if existing_run_id is not None else None
                ) or (
                    self._response_store.run_idempotency_status(
                        request_owner_profile, existing_run_id,
                    )
                    if existing_run_id is not None else None
                )
                released_owner_retry = (
                    owner_proposal_authority is not None
                    and existing_run_id is not None
                    and self._response_store.owner_claim_is_released(
                        owner_proposal_authority["proposal_profile"],
                        owner_proposal_authority["conversation"],
                        owner_proposal_authority["response_id"],
                        owner_proposal_authority["claim_id"],
                        existing_run_id,
                    )
                )
                if (
                    not released_owner_retry
                    and existing_run_id not in self._run_statuses
                    and persisted_status is None
                ):
                    return web.json_response(
                        _openai_error(
                            "Idempotent run result has expired",
                            code="idempotency_expired",
                        ),
                        status=409,
                    )
                if owner_proposal_authority is not None and not released_owner_retry:
                    try:
                        owner_snapshot = (
                            self._response_store.owner_history_snapshot(
                                owner_proposal_authority["conversation"],
                                profile=owner_proposal_authority[
                                    "proposal_profile"
                                ],
                            )
                        )
                    except OwnerAuthorityBroken:
                        return web.json_response(
                            _openai_error(
                                "Owner proposal authority could not be read",
                                code="owner_proposal_authority_conflict",
                            ),
                            status=409,
                        )
                    existing_status = (
                        self._run_statuses.get(existing_run_id)
                        or persisted_status
                        or {}
                    )
                    same_completed_mutation = (
                        owner_snapshot.get("proposal_consumed") is True
                        and existing_status.get("status") == "completed"
                        and existing_status.get("owner_mutation_committed") is True
                    )
                    if (
                        owner_snapshot.get("latest_response_id")
                        != owner_proposal_authority["response_id"]
                        or (
                            owner_snapshot.get("active_run_id") != existing_run_id
                            and not same_completed_mutation
                        )
                    ):
                        return web.json_response(
                            _openai_error(
                                "Owner proposal authority no longer owns this run",
                                code="owner_proposal_authority_conflict",
                            ),
                            status=409,
                        )
                if not released_owner_retry:
                    response_headers = (
                        {"X-Hermes-Session-Key": gateway_session_key}
                        if gateway_session_key else {}
                    )
                    return web.json_response(
                        {"run_id": existing_run_id, "status": "started"},
                        status=202,
                        headers=response_headers,
                    )

        # A retry that already owns a run is returned above even while the
        # service is at capacity. New work still obeys the shared limit.
        limited = self._concurrency_limited_response()
        if limited is not None:
            return limited

        run_id = f"run_{uuid.uuid4().hex}"
        # Everything startup recovery needs to decide this run's fate WITHOUT
        # guessing: which proposal claim to release, and — before it declares
        # anything failed — the exact native mutation authority whose committed
        # receipt would prove the change actually landed. Reserved in the same
        # transaction as the run's own authority, never after it.
        recovery_job_payload: Dict[str, Any] = {}
        if owner_proposal_authority is not None:
            recovery_job_payload["owner"] = {
                "proposal_profile": owner_proposal_authority["proposal_profile"],
                "conversation": owner_proposal_authority["conversation"],
                "response_id": owner_proposal_authority["response_id"],
                "claim_id": owner_proposal_authority["claim_id"],
                "operation": owner_proposal_authority["operation"],
            }
        native_authority = owner_proposal_authority or owner_lifecycle_authority
        if native_authority is not None and idempotency_key is not None:
            recovery_job_payload["native"] = {
                "operation": native_authority["operation"],
                "idempotency_key": native_authority["idempotency_key"],
                "authority_digest": native_authority["payload_digest"],
                "session_scope": idempotency_session_scope,
            }
        authority_bound = True
        if idempotency_key is not None:
            reserve_state, reserved_run_id = self._response_store.reserve_run_idempotency(
                request_owner_profile,
                idempotency_session_scope,
                idempotency_key,
                str(idempotency_fingerprint),
                run_id,
                owner=owner_proposal_authority,
                job_payload=recovery_job_payload,
            )
            authority_bound = reserve_state != "authority_conflict"
            if reserve_state == "conflict":
                return web.json_response(
                    _openai_error(
                        "Idempotency-Key was already used for a different run",
                        code="idempotency_conflict",
                    ),
                    status=409,
                )
            if reserve_state == "existing":
                response_headers = (
                    {"X-Hermes-Session-Key": gateway_session_key}
                    if gateway_session_key else {}
                )
                return web.json_response(
                    {"run_id": reserved_run_id, "status": "started"},
                    status=202,
                    headers=response_headers,
                )
        elif owner_proposal_authority is not None:
            authority_bound = self._response_store.claim_and_attach_owner_run(
                owner_proposal_authority["proposal_profile"],
                owner_proposal_authority["conversation"],
                owner_proposal_authority["response_id"],
                owner_proposal_authority["claim_id"],
                run_id,
                operation=owner_proposal_authority["operation"],
                payload_digest=owner_proposal_authority["payload_digest"],
                job_payload=recovery_job_payload,
                job_profile=request_owner_profile,
            )
        else:
            # No idempotency key and no proposal authority: this run has no
            # other durable record, so its job row IS the recovery authority
            # and one write commits it.
            self._response_store.reserve_owner_job(
                "run", run_id, request_owner_profile, recovery_job_payload,
            )
        if not authority_bound:
            return web.json_response(
                _openai_error(
                    "Owner proposal authority is not current",
                    code="owner_proposal_authority_conflict",
                ),
                status=409,
            )
        session_id = session_id or run_id
        # Approval queues gate host-side tool execution and must be isolated
        # per API run.  Client-provided session IDs and memory session keys are
        # conversation/memory scopes, not authorization namespaces: multiple
        # concurrent runs can intentionally share them, and resolving an
        # approval for one run must not unblock another run's dangerous command.
        approval_session_key = run_id
        owner_authority_context = None
        if owner_mutation_authority is not None:
            from hermes_cli.owner_workspace import OwnerProposalAuthority

            owner_profile = str(owner_workspace_context["profile"])
            owner_authority_context = OwnerProposalAuthority(
                actor=owner_profile,
                profile=owner_profile,
                session=approval_session_key,
                conversation=(
                    owner_proposal_authority["conversation"]
                    if owner_proposal_authority is not None else ""
                ),
                response_id=(
                    owner_proposal_authority["response_id"]
                    if owner_proposal_authority is not None else ""
                ),
                operation=owner_mutation_authority["operation"],
                idempotency_key=owner_mutation_authority["idempotency_key"],
                payload_digest=owner_mutation_authority["payload_digest"],
            )
        ephemeral_system_prompt = instructions
        loop = asyncio.get_running_loop()
        q: "asyncio.Queue[Optional[Dict]]" = asyncio.Queue()
        created_at = time.time()
        self._run_streams[run_id] = q
        self._run_streams_created[run_id] = created_at
        self._run_approval_sessions[run_id] = approval_session_key

        event_cb = self._make_run_event_callback(run_id, loop)

        def _put_event_if_active(event: Optional[Dict]) -> None:
            """Enqueue only while this run still owns live transport state."""
            if self._run_streams.get(run_id) is q:
                q.put_nowait(event)

        # Also wire stream_delta_callback so message.delta events flow through.
        def _text_cb(delta: Optional[str]) -> None:
            if delta is None:
                return
            if run_id not in self._run_streams:
                return
            try:
                loop.call_soon_threadsafe(_put_event_if_active, {
                    "event": "message.delta",
                    "run_id": run_id,
                    "timestamp": time.time(),
                    "delta": delta,
                })
            except Exception:
                pass

        owner_receipt_lock = threading.Lock()
        owner_finalize_lock = threading.Lock()
        owner_receipt: List[Optional[str]] = [None]
        owner_authority_finalized = [False]
        owner_authority_completed = [False]
        owner_completion_persisted = [False]
        owner_worker_future: List[Optional["asyncio.Future[Any]"]] = [None]

        def _finalize_owner_authority() -> bool:
            """Release an uncommitted proposal; success closes atomically elsewhere."""
            if owner_proposal_authority is None:
                return False
            with owner_finalize_lock:
                if owner_authority_finalized[0]:
                    return owner_authority_completed[0]
                with owner_receipt_lock:
                    receipt = owner_receipt[0]
                if receipt is not None:
                    return False
                args = (
                    owner_proposal_authority["proposal_profile"],
                    owner_proposal_authority["conversation"],
                    owner_proposal_authority["response_id"],
                    owner_proposal_authority["claim_id"],
                    run_id,
                )
                try:
                    applied = self._response_store.release_owner_claim(*args)
                    applied = applied or self._response_store.owner_claim_is_released(
                        *args
                    )
                    if applied:
                        owner_authority_finalized[0] = True
                    return owner_authority_completed[0]
                except Exception:
                    logger.exception(
                        "[api_server] owner authority finalization failed for run=%s",
                        run_id,
                    )
                    return False

        def _persist_owner_completion(receipt: str) -> bool:
            """Close one native owner mutation and its retry record together."""
            if owner_mutation_authority is None or idempotency_key is None:
                return False
            with owner_finalize_lock:
                if owner_completion_persisted[0]:
                    return True
                if owner_authority_finalized[0] and not owner_authority_completed[0]:
                    return False
                try:
                    persisted_created_at = (
                        self._response_store.run_idempotency_created_at(
                            request_owner_profile,
                            idempotency_session_scope,
                            idempotency_key,
                            run_id,
                        )
                    )
                    if persisted_created_at is None:
                        return False
                    terminal = self._response_store.persist_owner_run_completion(
                        request_owner_profile,
                        idempotency_session_scope,
                        idempotency_key,
                        run_id,
                        receipt,
                        created_at=persisted_created_at,
                        owner=owner_proposal_authority,
                    )
                except Exception:
                    logger.exception(
                        "[api_server] owner completion persistence failed for run=%s",
                        run_id,
                    )
                    return False
                owner_completion_persisted[0] = True
                if owner_proposal_authority is not None:
                    owner_authority_finalized[0] = True
                    owner_authority_completed[0] = True
                self._run_statuses[run_id] = terminal
                return True

        def _owner_tool_complete(
            _tool_call_id: str,
            function_name: str,
            _function_args: Any,
            function_result: Any,
        ) -> None:
            """Record only the exact authority-bound native mutation receipt."""
            if (
                owner_mutation_authority is None
                or function_name != owner_mutation_authority["operation"]
            ):
                return
            receipt = self._owner_mutation_receipt(function_name, function_result)
            if receipt is None:
                return
            with owner_receipt_lock:
                if owner_receipt[0] is None:
                    owner_receipt[0] = receipt
                elif owner_receipt[0] != receipt:
                    logger.error(
                        "[api_server] inconsistent owner mutation receipts for run=%s",
                        run_id,
                    )
                    return
            # Close the crash window at the tool boundary. The native mutation
            # has committed and its canonical result is already observable;
            # later model prose is not execution authority.
            _persist_owner_completion(receipt)

        def _owner_completed_receipt() -> Optional[str]:
            if owner_mutation_authority is None:
                return None
            with owner_receipt_lock:
                receipt = owner_receipt[0]
            if receipt is None or not _persist_owner_completion(receipt):
                return None
            return receipt

        def _publish_owner_completion(usage: Optional[Dict[str, Any]] = None) -> bool:
            receipt = _owner_completed_receipt()
            if receipt is None:
                return False
            completed_event = {
                "event": "run.completed",
                "run_id": run_id,
                "timestamp": time.time(),
                "output": receipt,
                "usage": usage or {},
            }
            _put_event_if_active(completed_event)
            self._set_run_status(
                run_id,
                "completed",
                output=receipt,
                usage=usage or {},
                owner_mutation_committed=True,
                last_event="run.completed",
            )
            return True

        def _owner_worker_done(future: "asyncio.Future[Any]") -> None:
            # Retrieve an exception when the outer task was cancelled while
            # the executor thread kept running; this prevents a lost-future
            # warning without trusting that exception as mutation authority.
            try:
                future.exception()
            except (asyncio.CancelledError, Exception):
                pass
            if _owner_completed_receipt() is None:
                _finalize_owner_authority()
            status = self._run_statuses.get(run_id, {}).get("status")
            if status in {"cancelled", "failed"}:
                _publish_owner_completion()

        self._set_run_status(
            run_id,
            "queued",
            created_at=created_at,
            session_id=session_id,
            model=body.get("model", self._model_name),
            owner_workspace_context=owner_workspace_context,
        )

        # Background task outlives the HTTP response (and thus the middleware
        # profile scope). Capture now and re-enter inside the task/executor.
        request_profile = _api_request_profile.get()

        async def _run_and_close():
            try:
                self._set_run_status(run_id, "running")
                if run_id in self._stopping_run_ids:
                    _put_event_if_active({
                        "event": "run.cancelled",
                        "run_id": run_id,
                        "timestamp": time.time(),
                    })
                    self._set_run_status(
                        run_id,
                        "cancelled",
                        last_event="run.cancelled",
                    )
                    return
                agent = None
                if owner_mutation_authority is None:
                    with self._profile_scope(request_profile):
                        agent = self._create_agent(
                            ephemeral_system_prompt=ephemeral_system_prompt,
                            session_id=session_id,
                            stream_delta_callback=_text_cb,
                            tool_progress_callback=event_cb,
                            tool_complete_callback=_owner_tool_complete,
                            gateway_session_key=gateway_session_key,
                            requested_model=agent_overrides.get("requested_model"),
                            requested_provider=agent_overrides.get("requested_provider"),
                            model_options=agent_overrides.get("model_options"),
                            route=route,
                        )
                    self._active_run_agents[run_id] = agent
                raw_runtime = dict(
                    getattr(agent, "_hermes_api_runtime", {}) or {}
                )
                persisted_runtime = {
                    "provider": self._clean_runtime_id(
                        raw_runtime.get("provider"), max_len=80
                    ),
                    "model": self._clean_runtime_id(raw_runtime.get("model")),
                    "effort": self._clean_runtime_id(
                        raw_runtime.get("effort"), max_len=16
                    ),
                    "engine": self._clean_runtime_id(
                        raw_runtime.get("engine"), max_len=32
                    ),
                }
                self._set_run_status(
                    run_id, "running", runtime=persisted_runtime
                )

                def _approval_notify(approval_data: Dict[str, Any]) -> None:
                    event = dict(approval_data or {})
                    # Redact credentials from the command before it enters the
                    # SSE/API event stream — same egress bug as #48456, second
                    # transport: API/desktop clients would otherwise receive the
                    # raw command Tirith flagged. Reuse the gateway seam.
                    if "command" in event:
                        from gateway.run import _redact_approval_command

                        event["command"] = _redact_approval_command(event.get("command"))
                    choices = _approval_event_choices(
                        smart_denied=bool(event.get("smart_denied"))
                        or bool(event.get("exact_operation")),
                        allow_permanent=event.get("allow_permanent") is not False,
                    )
                    event.update({
                        "event": "approval.request",
                        "run_id": run_id,
                        "timestamp": time.time(),
                        "choices": choices,
                    })
                    pending_approval = {
                        "approval_id": str(event.get("approval_id") or ""),
                        "description": redact_sensitive_text(
                            str(event.get("description") or "Approve this operation"),
                            force=True,
                        ),
                        "choices": choices,
                    }
                    operation = str(event.get("operation") or "")
                    if event.get("exact_operation") and operation in {
                        "owner_workspace_bootstrap",
                        "owner_task_graph_commit",
                        "owner_project_plan_commit",
                        "owner_task_move",
                        "owner_task_comment", "owner_project_lifecycle",
                    }:
                        pending_approval["operation"] = operation
                    self._set_run_status(
                        run_id,
                        "waiting_for_approval",
                        last_event="approval.request",
                        pending_approval=pending_approval,
                    )
                    try:
                        loop.call_soon_threadsafe(q.put_nowait, event)
                    except Exception:
                        pass

                def _run_sync():
                    from gateway.session_context import clear_session_vars
                    from tools.approval import (
                        register_gateway_notify,
                        reset_current_session_key,
                        set_current_session_key,
                        unregister_gateway_notify,
                    )

                    effective_task_id = session_id or run_id
                    approval_token = None
                    owner_authority_token = None
                    session_tokens = []
                    with self._profile_scope(request_profile):
                        try:
                            # Bind approval/session identity for this API run via
                            # contextvars so concurrent runs do not share process
                            # environment state.
                            approval_token = set_current_session_key(approval_session_key)
                            if owner_authority_context is not None:
                                from hermes_cli.owner_workspace import (
                                    set_owner_proposal_authority,
                                )

                                owner_authority_token = set_owner_proposal_authority(
                                    owner_authority_context,
                                )
                            session_tokens = self._bind_api_server_session(
                                # chat_id carries the raw session id (the
                                # X-Hermes-Session-Id equivalent) exactly like
                                # the other agent-entry routes bind it via
                                # _run_agent(). Without it,
                                # tools.async_delegation reads an empty
                                # HERMES_SESSION_CHAT_ID on /v1/runs and
                                # background delegations stay forced-sync
                                # (no wake target).
                                chat_id=session_id or "",
                                session_key=approval_session_key,
                                session_id=session_id or "",
                            )
                            register_gateway_notify(approval_session_key, _approval_notify)
                            # /v1/runs runs its own agent lifecycle (no
                            # TurnRunner, no _run_agent) — record turn process
                            # ownership so stop/cancel can reap only the
                            # background processes this run created (#76115).
                            if owner_mutation_authority is not None:
                                # The authenticated proposal is already frozen.
                                # Apply its exact payload through the existing
                                # guarded kernel; never ask a model to copy it.
                                from tools import owner_workspace_tools

                                handlers = {
                                    "owner_task_graph_commit":
                                        owner_workspace_tools._handle_task_graph,
                                    "owner_project_plan_commit":
                                        owner_workspace_tools._handle_project_plan,
                                    "owner_project_lifecycle":
                                        owner_workspace_tools._handle_project_lifecycle,
                                }
                                operation = owner_mutation_authority["operation"]
                                arguments = copy.deepcopy(
                                    owner_mutation_authority["payload"]
                                )
                                result = handlers[operation](arguments)
                                _owner_tool_complete(
                                    run_id, operation, arguments, result,
                                )
                                r = {"final_response": result}
                            else:
                                _publish_turn_process_ownership(agent, effective_task_id)
                                r = agent.run_conversation(
                                    user_message=user_message,
                                    conversation_history=conversation_history,
                                    task_id=effective_task_id,
                                )
                        finally:
                            # Worker finished (interrupted or complete) —
                            # clear turn ownership immediately so a later
                            # stop/cancel can't reap background work this
                            # run deliberately left running (same race-window
                            # guard as gateway/run.py and _run_agent above).
                            if agent is not None:
                                _clear_turn_process_ownership(agent)
                            try:
                                unregister_gateway_notify(approval_session_key)
                            finally:
                                if approval_token is not None:
                                    try:
                                        reset_current_session_key(approval_token)
                                    except Exception:
                                        pass
                                if owner_authority_token is not None:
                                    try:
                                        from hermes_cli.owner_workspace import (
                                            reset_owner_proposal_authority,
                                        )

                                        reset_owner_proposal_authority(
                                            owner_authority_token,
                                        )
                                    except Exception:
                                        pass
                                if session_tokens:
                                    try:
                                        clear_session_vars(session_tokens)
                                    except Exception:
                                        pass
                        u = {
                            "input_tokens": getattr(agent, "session_prompt_tokens", 0) or 0,
                            "output_tokens": getattr(agent, "session_completion_tokens", 0) or 0,
                            "total_tokens": getattr(agent, "session_total_tokens", 0) or 0,
                        }
                        return r, u

                worker_future = asyncio.get_running_loop().run_in_executor(
                    None, _run_sync,
                )
                owner_worker_future[0] = worker_future
                if owner_mutation_authority is not None:
                    worker_future.add_done_callback(_owner_worker_done)
                result, usage = await asyncio.shield(worker_future)
                if _publish_owner_completion(usage):
                    pass
                elif run_id in self._stopping_run_ids:
                    _put_event_if_active({
                        "event": "run.cancelled",
                        "run_id": run_id,
                        "timestamp": time.time(),
                    })
                    self._set_run_status(
                        run_id,
                        "cancelled",
                        last_event="run.cancelled",
                    )
                # Check for structured failure (non-retryable client errors like
                # 401/400 return failed=True instead of raising, so the except
                # block below never fires — issue #15561).
                elif isinstance(result, dict) and result.get("failed"):
                    error_msg = _redact_api_error_text(result.get("error") or "agent run failed")
                    _put_event_if_active({
                        "event": "run.failed",
                        "run_id": run_id,
                        "timestamp": time.time(),
                        "error": error_msg,
                    })
                    self._set_run_status(
                        run_id,
                        "failed",
                        error=error_msg,
                        last_event="run.failed",
                    )
                elif owner_mutation_authority is not None:
                    error_msg = "The approved Project change was not committed"
                    _put_event_if_active({
                        "event": "run.failed",
                        "run_id": run_id,
                        "timestamp": time.time(),
                        "error": error_msg,
                    })
                    self._set_run_status(
                        run_id,
                        "failed",
                        error=error_msg,
                        owner_mutation_committed=False,
                        last_event="run.failed",
                    )
                else:
                    final_response = result.get("final_response", "") if isinstance(result, dict) else ""
                    # Undelivered steer text (accepted after the final response;
                    # see turn_finalizer) rides on the terminal event/status so
                    # the client can replay it as the next user turn.
                    pending_steer = result.get("pending_steer") if isinstance(result, dict) else None
                    completed_event = {
                        "event": "run.completed",
                        "run_id": run_id,
                        "timestamp": time.time(),
                        "output": final_response,
                        "usage": usage,
                    }
                    if pending_steer:
                        completed_event["pending_steer"] = pending_steer
                    _put_event_if_active(completed_event)
                    self._set_run_status(
                        run_id,
                        "completed",
                        output=final_response,
                        usage=usage,
                        last_event="run.completed",
                        **({"pending_steer": pending_steer} if pending_steer else {}),
                    )
            except asyncio.CancelledError:
                if not _publish_owner_completion():
                    self._set_run_status(
                        run_id,
                        "cancelled",
                        last_event="run.cancelled",
                    )
                    try:
                        _put_event_if_active({
                            "event": "run.cancelled",
                            "run_id": run_id,
                            "timestamp": time.time(),
                        })
                    except Exception:
                        pass
                raise
            except _ProviderAuthResolutionError as exc:
                # /v1/runs builds its own agent via _create_agent() and does
                # not route through _run_agent() (see that method's own
                # _ProviderAuthResolutionError branch), so it needs its own
                # handling to surface the same distinguished, controlled
                # message the other endpoints give a provider auth/credential
                # failure, instead of falling through to the generic
                # except-Exception branch below.
                logger.warning("Provider authentication failed for run=%s: %s", run_id, exc)
                error_msg = f"⚠️ Provider authentication failed: {exc}"
                if not _publish_owner_completion():
                    self._set_run_status(
                        run_id,
                        "failed",
                        error=error_msg,
                        last_event="run.failed",
                    )
                    try:
                        _put_event_if_active({
                            "event": "run.failed",
                            "run_id": run_id,
                            "timestamp": time.time(),
                            "error": error_msg,
                        })
                    except Exception:
                        pass
            except Exception as exc:
                logger.exception("[api_server] run %s failed", run_id)
                if not _publish_owner_completion():
                    self._set_run_status(
                        run_id,
                        "failed",
                        error=_redact_api_error_text(exc),
                        last_event="run.failed",
                    )
                    try:
                        _put_event_if_active({
                            "event": "run.failed",
                            "run_id": run_id,
                            "timestamp": time.time(),
                            "error": _redact_api_error_text(exc),
                        })
                    except Exception:
                        pass
            finally:
                # Before a worker exists, failure/cancellation is final and
                # can safely release the exact proposal. Once the executor
                # thread starts, its done callback owns finalization because
                # asyncio cancellation does not stop that thread.
                if (
                    owner_proposal_authority is not None
                    and owner_worker_future[0] is None
                ):
                    _finalize_owner_authority()
                # If the asyncio wrapper is cancelled (for example via
                # /stop), the executor thread can still be blocked waiting
                # on an approval Event.  Unregistering here releases those
                # waits immediately; the in-thread unregister is harmlessly
                # idempotent on normal completion.
                try:
                    from tools.approval import unregister_gateway_notify

                    unregister_gateway_notify(approval_session_key)
                except Exception:
                    pass
                # Sentinel: signal SSE stream to close
                try:
                    _put_event_if_active(None)
                except Exception:
                    pass
                self._active_run_agents.pop(run_id, None)
                self._active_run_tasks.pop(run_id, None)
                self._run_approval_sessions.pop(run_id, None)
                self._stopping_run_ids.discard(run_id)

        # The durable recovery job was reserved in the same transaction as this
        # run's own authority, above — never after it, because a crash in that
        # gap left a durable queued run and a claimed proposal with no executor.
        self._activate_admitted_request()
        task = asyncio.create_task(_run_and_close())
        task.add_done_callback(
            lambda _task: self._finalize_run_recovery_job(
                run_id, request_owner_profile,
            )
        )
        self._active_run_tasks[run_id] = task
        try:
            self._background_tasks.add(task)
        except TypeError:
            pass
        if hasattr(task, "add_done_callback"):
            task.add_done_callback(self._background_tasks.discard)

        response_headers = (
            {"X-Hermes-Session-Key": gateway_session_key} if gateway_session_key else {}
        )
        return web.json_response(
            {"run_id": run_id, "status": "started"},
            status=202,
            headers=response_headers,
        )

    async def _handle_get_run(self, request: "web.Request") -> "web.Response":
        """GET /v1/runs/{run_id} — return pollable run status for external UIs."""
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err

        run_id = request.match_info["run_id"]
        # A durable owner completion outranks in-memory transport state. A
        # restart recovery can close the cross-database receipt window while a
        # stale running/failed cache entry still exists for the same run.
        status = self._response_store.owner_run_completion(
            _active_owner_profile(), run_id,
        ) or self._run_statuses.get(run_id) or (
            self._response_store.run_idempotency_status(
                _active_owner_profile(), run_id,
            )
        )
        if status is None:
            return web.json_response(
                _openai_error(f"Run not found: {run_id}", code="run_not_found"),
                status=404,
            )
        return web.json_response(status)

    async def _handle_run_events(self, request: "web.Request") -> "web.StreamResponse":
        """GET /v1/runs/{run_id}/events — SSE stream of structured agent lifecycle events."""
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err

        run_id = request.match_info["run_id"]

        # Allow subscribing slightly before the run is registered (race condition window)
        for _ in range(20):
            if run_id in self._run_streams:
                break
            await asyncio.sleep(0.05)
        else:
            return web.json_response(_openai_error(f"Run not found: {run_id}", code="run_not_found"), status=404)

        q = self._run_streams[run_id]
        self._run_stream_subscribers.add(run_id)

        response = web.StreamResponse(
            status=200,
            headers={
                "Content-Type": "text/event-stream",
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )
        await response.prepare(request)

        try:
            while True:
                try:
                    event = await asyncio.wait_for(q.get(), timeout=30.0)
                except asyncio.TimeoutError:
                    await response.write(b": keepalive\n\n")
                    continue
                if event is None:
                    # Run finished — send final SSE comment and close
                    await response.write(b": stream closed\n\n")
                    break
                payload = _sse_frame(event)
                await response.write(payload)
        except Exception as exc:
            logger.debug("[api_server] SSE stream error for run %s: %s", run_id, exc)
        finally:
            self._run_stream_subscribers.discard(run_id)
            self._run_streams.pop(run_id, None)
            self._run_streams_created.pop(run_id, None)

        return response


    async def _handle_run_approval(self, request: "web.Request") -> "web.Response":
        """POST /v1/runs/{run_id}/approval — resolve a pending run approval."""
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err

        run_id = request.match_info["run_id"]
        status = self._run_statuses.get(run_id)
        if status is None:
            return web.json_response(
                _openai_error(f"Run not found: {run_id}", code="run_not_found"),
                status=404,
            )

        try:
            body = await request.json()
        except Exception:
            return web.json_response(_openai_error("Invalid JSON"), status=400)

        raw_choice = str(body.get("choice", "")).strip().lower()
        aliases = {"approve": "once", "approved": "once", "allow": "once"}
        choice = aliases.get(raw_choice, raw_choice)
        allowed = {"once", "session", "always", "deny"}
        if choice not in allowed:
            return web.json_response(
                _openai_error(
                    "Invalid approval choice; expected one of: once, session, always, deny",
                    code="invalid_approval_choice",
                ),
                status=400,
            )

        approval_session_key = self._run_approval_sessions.get(run_id)
        if not approval_session_key:
            return web.json_response(
                _openai_error(
                    f"Run has no active approval session: {run_id}",
                    code="approval_not_active",
                ),
                status=409,
            )

        resolve_all = (
            _coerce_request_bool(body.get("all"), default=False)
            or _coerce_request_bool(body.get("resolve_all"), default=False)
        )
        approval_id = body.get("approval_id")
        if approval_id is not None:
            approval_id = str(approval_id).strip() or None
        try:
            from tools.approval import resolve_gateway_approval

            resolved = resolve_gateway_approval(
                approval_session_key,
                choice,
                resolve_all=resolve_all,
                approval_id=approval_id,
            )
        except Exception as exc:
            logger.exception("[api_server] approval resolution failed for run %s", run_id)
            return web.json_response(_openai_error(str(exc)), status=500)

        if resolved <= 0:
            return web.json_response(
                _openai_error(
                    f"Run has no pending approval: {run_id}",
                    code="approval_not_pending",
                ),
                status=409,
            )

        self._set_run_status(run_id, "running", last_event="approval.responded")
        q = self._run_streams.get(run_id)
        if q is not None:
            try:
                q.put_nowait({
                    "event": "approval.responded",
                    "run_id": run_id,
                    "timestamp": time.time(),
                    "choice": choice,
                    "resolved": resolved,
                })
            except Exception:
                pass

        return web.json_response({
            "object": "hermes.run.approval_response",
            "run_id": run_id,
            "choice": choice,
            "resolved": resolved,
        })

    async def _handle_steer_run(self, request: "web.Request") -> "web.Response":
        """POST /v1/runs/{run_id}/steer — inject guidance into a running agent."""
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err

        run_id = request.match_info["run_id"]
        status = self._run_statuses.get(run_id)
        if status is None:
            return web.json_response(_openai_error(f"Run not found: {run_id}", code="run_not_found"), status=404)
        # Only genuinely running runs are steerable.  /stop retains agent/task
        # refs during cooperative shutdown, so the status gate (not the mere
        # presence of an agent ref) is what rejects stop-then-steer.
        agent = self._active_run_agents.get(run_id)
        if status.get("status") != "running" or not hasattr(agent, "steer"):
            return web.json_response(
                _openai_error(
                    f"Run is not currently accepting steer input: {run_id}",
                    code="run_not_accepting_steer",
                ),
                status=409,
            )

        body, err = await self._read_json_body(request)
        if err:
            return err
        raw_text = body.get("input") or body.get("message") or body.get("text") or ""
        steer_text = _normalize_chat_content(raw_text).strip()
        if not steer_text:
            return web.json_response(
                _openai_error(
                    "Missing non-empty steer text; expected 'input', 'message', or 'text'.",
                    code="invalid_steer_input",
                ),
                status=400,
            )

        try:
            accepted = bool(agent.steer(steer_text))
        except Exception as exc:
            logger.exception("[api_server] steer failed for run %s", run_id)
            return web.json_response(_openai_error(_redact_api_error_text(exc), code="steer_failed"), status=500)
        if not accepted:
            return web.json_response(
                _openai_error(f"Run did not accept steer text: {run_id}", code="steer_not_accepted"),
                status=409,
            )

        self._set_run_status(run_id, "running", last_event="run.steered")
        q = self._run_streams.get(run_id)
        if q is not None:
            with suppress(Exception):
                q.put_nowait({
                    "event": "run.steered",
                    "run_id": run_id,
                    "timestamp": time.time(),
                    "accepted": True,
                })
        return web.json_response({"object": "hermes.run.steer", "run_id": run_id, "accepted": True})

    async def _handle_stop_run(self, request: "web.Request") -> "web.Response":
        """POST /v1/runs/{run_id}/stop — interrupt a running agent."""
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err

        run_id = request.match_info["run_id"]
        agent = self._active_run_agents.get(run_id)
        task = self._active_run_tasks.get(run_id)

        if agent is None and task is None:
            return web.json_response(_openai_error(f"Run not found: {run_id}", code="run_not_found"), status=404)

        self._set_run_status(run_id, "stopping", last_event="run.stopping")
        self._stopping_run_ids.add(run_id)

        if agent is not None:
            try:
                request_hard_interrupt(agent, "Stop requested via API")
            except Exception:
                pass
            # The stopped run is abandoned — reap only the background
            # processes it created (#76115). Epoch-gated inside, so a
            # concurrent run sharing the same session_id keeps its own
            # processes; no-op if the run already finished and cleared
            # its ownership markers.
            _reap_disconnected_agent_processes(
                agent, source="api_server_run_stop"
            )

        return web.json_response({"run_id": run_id, "status": "stopping"})

    async def _sweep_orphaned_runs(self) -> None:
        """Periodically expire transport buffers and terminal status records."""
        while True:
            await asyncio.sleep(60)
            self._sweep_orphaned_runs_once(time.time())

    def _sweep_orphaned_runs_once(self, now: Optional[float] = None) -> None:
        """Expire old SSE buffers without treating transport age as run age."""
        if now is None:
            now = time.time()
        stale = [
            run_id
            for run_id, created_at in list(self._run_streams_created.items())
            if now - created_at > self._RUN_STREAM_TTL
            and run_id not in self._run_stream_subscribers
        ]
        for run_id in stale:
            logger.debug("[api_server] sweeping expired run transport %s", run_id)
            task = self._active_run_tasks.get(run_id)
            task_done = task is None or task.done()
            if task_done:
                try:
                    from tools.approval import unregister_gateway_notify

                    approval_session_key = self._run_approval_sessions.get(run_id)
                    if approval_session_key:
                        unregister_gateway_notify(approval_session_key)
                except Exception:
                    pass
            # The transport TTL always bounds buffering. Live control state is
            # independent and survives until the executor-backed task returns.
            self._run_streams.pop(run_id, None)
            self._run_streams_created.pop(run_id, None)
            if task_done:
                self._active_run_agents.pop(run_id, None)
                self._active_run_tasks.pop(run_id, None)
                self._run_approval_sessions.pop(run_id, None)
                self._stopping_run_ids.discard(run_id)

        stale_statuses = [
            run_id
            for run_id, status in list(self._run_statuses.items())
            if status.get("status") in {"completed", "failed", "cancelled"}
            and now - float(status.get("updated_at", 0) or 0) > self._RUN_STATUS_TTL
        ]
        for run_id in stale_statuses:
            self._run_statuses.pop(run_id, None)
        self._response_store.purge_run_idempotency(now - 24 * 60 * 60)
        self._response_store.purge_owner_response_idempotency(now - 24 * 60 * 60)
        # Heartbeat BEFORE recovery, so work this process is still driving can
        # never be reaped by its own sweep.
        self._heartbeat_owner_job_leases()
        self._recover_orphaned_owner_jobs()

    def _finalize_run_recovery_job(self, run_id: str, profile: str) -> None:
        """Persist this run's terminal status and retire its recovery job.

        Both in ONE transaction. Deleting the job row on its own left a durable
        row still saying ``queued`` after a restart — with no executor and no
        recovery authority — so polling reported working forever. A run that
        already persisted a terminal receipt keeps it; that receipt outranks any
        transport-level status.

        Nothing is deleted when the run did not actually reach a terminal state
        (the task died before recording one): the job row survives so a later
        sweep can recover it.
        """
        status = self._run_statuses.get(run_id) or {}
        state = str(status.get("status") or "")
        if state not in _TERMINAL_RUN_STATUSES:
            state = "failed"
        try:
            self._response_store.persist_terminal_run_status(profile, run_id, state)
        except OwnerAuthorityUnavailable:
            return
        except Exception:
            logger.exception(
                "[api_server] run %s terminal state could not be persisted", run_id,
            )

    def _driving_owner_jobs(self, kind: str) -> "list[str]":
        """The jobs of ``kind`` this process still has a live executor for."""
        if kind == "run":
            return [
                run_id for run_id, task in list(self._active_run_tasks.items())
                if not task.done()
            ]
        return list(self._owner_response_jobs)

    def _heartbeat_owner_job_leases(self) -> None:
        """Renew this executor's lease on the owner work it is still driving.

        The lease — not pid liveness — is what proves an executor is alive, so a
        run or response that is genuinely still working has to keep saying so,
        or a sibling process would reclaim live work once the lease ran out.
        """
        for kind in ("run", "response"):
            try:
                self._response_store.renew_owner_job_leases(
                    kind, self._driving_owner_jobs(kind),
                )
            except OwnerAuthorityUnavailable:
                return
            except Exception:
                logger.exception("[api_server] owner job lease heartbeat failed")

    def _recover_orphaned_owner_jobs(self) -> None:
        """Close out durably queued owner work whose executor is gone.

        A queued owner response or run is durable, but its only executor is an
        in-memory task in the process that started it. When that process dies
        the work cannot resume, so each orphan is made terminal exactly once:
        the response reports failed, frees its conversation and records that
        terminal failure under its own immutable idempotency key, and the run
        reports failed and releases its owner proposal claim so the owner can
        approve the same change again. Runs on every start and on every sweep,
        so a sibling process that dies later is reconciled too.

        Each orphan is CLAIMED under a fresh lease rather than deleted. Its row
        is removed only by the transaction that makes its work terminal, so a
        crash, a malformed payload, or an exception during terminalization
        leaves the recovery authority intact for the next sweep instead of
        destroying it.

        A mutation run is never declared failed before its exact native receipt
        has been reconciled: the gateway can die after the native database
        committed but before its own run terminal was stored, and reporting that
        completed change as failed — while releasing its proposal — would invite
        the owner to approve the very same mutation a second time.
        """
        try:
            responses = self._response_store.claim_orphaned_owner_jobs(
                "response", self._driving_owner_jobs("response"),
            )
            runs = self._response_store.claim_orphaned_owner_jobs(
                "run", self._driving_owner_jobs("run"),
            )
        except OwnerAuthorityUnavailable:
            return
        except Exception:
            logger.exception("[api_server] owner job recovery could not read state")
            return
        for job in responses:
            conversation = str(job["payload"].get("conversation") or "")
            if not conversation:
                continue
            try:
                if self._response_store.fail_orphaned_owner_response(
                    job["profile"], conversation, job["job_key"],
                    _OWNER_ORPHAN_RESPONSE_MESSAGE,
                ):
                    logger.warning(
                        "[api_server] recovered orphaned owner response %s",
                        job["job_key"],
                    )
            except Exception:
                logger.exception(
                    "[api_server] orphaned owner response %s could not be closed",
                    job["job_key"],
                )
        for job in runs:
            owner = job["payload"].get("owner")
            try:
                if self._reconcile_orphaned_native_run(job):
                    logger.warning(
                        "[api_server] recovered a committed native mutation for "
                        "orphaned owner run %s", job["job_key"],
                    )
                    continue
            except _OwnerNativeReceiptUnreadable:
                # The mutation may or may not have committed. Its job row keeps
                # its lease and a later sweep decides; failing it now could
                # report a completed change as failed and release its proposal.
                logger.error(
                    "[api_server] orphaned owner run %s left for another pass: "
                    "its native receipt could not be read", job["job_key"],
                )
                continue
            try:
                if self._response_store.fail_orphaned_owner_run(
                    job["profile"], job["job_key"],
                    owner if isinstance(owner, dict) else None,
                ):
                    logger.warning(
                        "[api_server] recovered orphaned owner run %s",
                        job["job_key"],
                    )
            except Exception:
                logger.exception(
                    "[api_server] orphaned owner run %s could not be closed",
                    job["job_key"],
                )

    def _reconcile_orphaned_native_run(self, job: "Dict[str, Any]") -> bool:
        """Persist an orphaned run's committed native receipt, if it has one.

        Returns True when the receipt was found and the run's terminal
        completion is now durable. Returns False only when this exact run
        provably committed NOTHING — the one state in which it may be reported
        failed. Raises :class:`_OwnerNativeReceiptUnreadable` when that cannot
        be decided.
        """
        native = job["payload"].get("native")
        if not isinstance(native, dict):
            # A run with no native mutation authority (an ordinary agent run)
            # has no external effect to reconcile.
            return False
        owner = job["payload"].get("owner")
        return self._recover_native_owner_completion(
            profile=str(job["profile"]),
            session_scope=str(native.get("session_scope") or ""),
            run_id=str(job["job_key"]),
            authority={
                "operation": str(native.get("operation") or ""),
                "idempotency_key": str(native.get("idempotency_key") or ""),
            },
            owner=owner if isinstance(owner, dict) else None,
            authority_digest=str(native.get("authority_digest") or ""),
        ) is not None

    # ------------------------------------------------------------------
    # BasePlatformAdapter interface
    # ------------------------------------------------------------------

    def _api_key_passes_startup_guard(self) -> bool:
        """Return True when API_SERVER_KEY is present and strong enough to start."""
        if not self._api_key:
            logger.error(
                "[%s] Refusing to start: API_SERVER_KEY is required for the API server, "
                "including loopback-only binds on %s.",
                self.name, self._host,
            )
            return False

        try:
            from hermes_cli.auth import has_usable_secret
        except Exception as exc:
            # Fail CLOSED. This guard is the only thing between a guessable
            # key and a terminal-capable endpoint, so "the check could not be
            # run" must not resolve to "start anyway" — the same posture
            # tools/credential_files.py takes when its deny-list cannot be
            # consulted.
            logger.error(
                "[%s] Refusing to start: API_SERVER_KEY strength could not be "
                "verified (%s: %s), and this endpoint dispatches "
                "terminal-capable agent work. Repair the installation before "
                "starting the API server on %s.",
                self.name, type(exc).__name__, exc, self._host,
            )
            return False

        if not has_usable_secret(self._api_key, min_length=16):
            logger.error(
                "[%s] Refusing to start: API_SERVER_KEY is a "
                "placeholder or too short (<16 chars). This endpoint "
                "dispatches terminal-capable agent work — a guessable "
                "key is remote code execution. Generate a strong secret "
                "(e.g. `openssl rand -hex 32`) and set API_SERVER_KEY "
                "before starting the API server on %s.",
                self.name, self._host,
            )
            return False
        return True

    async def connect(self, *, is_reconnect: bool = False) -> bool:
        """Start the aiohttp web server."""
        if not AIOHTTP_AVAILABLE:
            logger.warning("[%s] aiohttp not installed", self.name)
            return False

        with self._session_db_cache_lock:
            self._session_db_cache_closed = False

        if not self._api_key_passes_startup_guard():
            # A rejected API_SERVER_KEY is a configuration error, not a
            # transient blip — the key will not become valid on its own. A
            # bare ``return False`` makes the reconnect watcher in
            # gateway.run treat it as retryable and loop forever at the
            # backoff cap, re-instantiating the adapter (and its
            # ResponseStore sqlite connection) every retry (#38803: ~501
            # leaked connections / 1002 fds over 2.5 days until EMFILE took
            # the whole gateway down). Non-retryable drops it from the
            # reconnect queue — same treatment as the port-conflict guard
            # (api_server_port_in_use). The guard already logged the
            # specific rejection reason just above.
            self._set_fatal_error(
                "api_server_key_invalid",
                "API_SERVER_KEY was rejected by the startup guard (missing, "
                "placeholder/too short, or strength unverifiable — see the "
                "error logged above). Generate a strong secret (e.g. "
                "`openssl rand -hex 32`), set API_SERVER_KEY, then "
                "`/platform resume api_server`.",
                retryable=False,
            )
            return False

        try:
            mws = [
                mw
                for mw in (
                    self._make_profile_prefix_middleware(),
                    self._make_route_allowlist_middleware(),
                    self._make_owner_authority_middleware(),
                    cors_middleware,
                    body_limit_middleware,
                    security_headers_middleware,
                )
                if mw is not None
            ]
            self._app = web.Application(middlewares=mws, client_max_size=MAX_REQUEST_BYTES)
            assert self._app is not None
            # Native routes + multiplex /p/<profile>/… mirrors. Same handlers;
            # the profile-prefix middleware validates the prefix and scopes
            # config/credentials to that profile when multiplexing is on.
            for method, path, handler in self._http_route_table():
                self._app.router.add_route(method, path, handler)
                self._app.router.add_route(method, f"/p/{{profile}}{path}", handler)
            # Store the adapter after native routes are registered. Local Hermes-Relay
            # bootstrap shims use this key as a feature-detection hook; registering
            # native routes first lets those shims no-op instead of shadowing the
            # upstream session-control handlers.
            self._app["api_server_adapter"] = self
            if self.gateway_runner is not None:
                self._app["gateway_runner"] = self.gateway_runner

            # Durably queued owner work whose executor died with the previous
            # process cannot resume, so it is made terminal before this one
            # starts serving.
            self._recover_orphaned_owner_jobs()

            # Start background sweep to clean up orphaned (unconsumed) run streams
            sweep_task = asyncio.create_task(self._sweep_orphaned_runs())
            try:
                self._background_tasks.add(sweep_task)
            except TypeError:
                pass
            if hasattr(sweep_task, "add_done_callback"):
                sweep_task.add_done_callback(self._background_tasks.discard)

            # Loud warning when a network-accessible API server runs against an
            # unsandboxed local terminal backend. The API server can drive the
            # agent's terminal/file tools as the host user; on a public bind
            # that is the exact surface the hermes-0day campaign abused to write
            # ~/.hermes/config.yaml and plant persistence. Sandboxing (Docker /
            # remote backend) contains the blast radius. Warn, don't refuse —
            # the operator may have an external firewall / strong key.
            if is_network_accessible(self._host):
                try:
                    from hermes_cli.config import load_config as _load_cfg
                    _backend = (
                        ((_load_cfg() or {}).get("terminal") or {}).get(
                            "backend", "local"
                        )
                    )
                except Exception:
                    _backend = "local"
                if str(_backend).lower() == "local":
                    logger.warning(
                        "[%s] API server is network-accessible (%s) AND the "
                        "terminal backend is 'local' (unsandboxed). Agent work "
                        "dispatched through this endpoint runs as the host user "
                        "with full terminal/file access. Strongly consider a "
                        "sandboxed backend (terminal.backend: docker) and "
                        "firewalling this port to trusted networks only.",
                        self.name, self._host,
                    )

            self._runner = web.AppRunner(self._app)
            await self._runner.setup()
            # Bind directly instead of probing 127.0.0.1 first — the old
            # single-family pre-probe raced the real bind and reported a
            # TIME_WAIT socket as "in use" (#10297), failing gateway
            # restarts for up to ~60s.
            #
            # SO_REUSEADDR is platform-dependent (same rationale as the
            # webhook adapter, #65482):
            #   - macOS (BSD semantics): two sockets with SO_REUSEADDR can
            #     silently split traffic while both report success — disable.
            #   - Linux: SO_REUSEADDR only permits rebinding past TIME_WAIT
            #     (a second live listener needs SO_REUSEPORT, never set), so
            #     keep the default (enabled) for instant restart rebinds.
            self._site = web.TCPSite(
                self._runner,
                self._host,
                self._port,
                reuse_address=False if sys.platform == "darwin" else None,
            )
            try:
                await self._site.start()
            except OSError as exc:
                await self._runner.cleanup()
                self._runner = None
                self._site = None
                if getattr(exc, "errno", None) == errno.EADDRINUSE:
                    # A port conflict is a configuration error, not a
                    # transient blip — another process holds the port for
                    # its lifetime. A bare ``return False`` makes the
                    # reconnect watcher in gateway.run treat it as retryable
                    # and loop forever at the backoff cap (observed: 1568+
                    # retries over 5 days across multi-profile setups all
                    # defaulting to the same port, #52132), filling
                    # errors.log and leaking the adapter's ResponseStore
                    # fds each retry. Non-retryable drops it from the
                    # reconnect queue; the operator recovers with
                    # ``/platform resume api_server`` after changing the port.
                    self._set_fatal_error(
                        "api_server_port_in_use",
                        f"Port {self._port} already in use. Set "
                        f"platforms.api_server.port in config.yaml to a "
                        f"different value, then `/platform resume api_server`.",
                        retryable=False,
                    )
                logger.error(
                    "[%s] Could not bind %s:%d: %s. Set a different port in "
                    "config.yaml: platforms.api_server.port",
                    self.name, self._host, self._port, exc,
                )
                return False

            self._mark_connected()
            logger.info(
                "[%s] API server listening on http://%s:%d (model: %s)",
                self.name, self._host, self._port, self._model_name,
            )
            return True

        except Exception as e:
            logger.error("[%s] Failed to start API server: %s", self.name, e)
            return False

    async def disconnect(self) -> None:
        """Stop the aiohttp web server and release all owned resources.

        Closes the ResponseStore SQLite connection in addition to stopping
        the aiohttp web server. Without this, every adapter instance leaks
        2 file descriptors (the database file and its WAL sidecar) — the
        reconnect loop in ``gateway.run`` constructs a fresh adapter on
        every retry, so 2 fds/retry × 300s backoff cap ≈ 12 fds/hour, which
        exhausts the default 2560 fd limit after ~12h of failed reconnects
        and turns the whole gateway into a zombie
        (OSError: [Errno 24] Too many open files, #37011).
        """
        self._mark_disconnected()
        if self._response_store is not None:
            try:
                self._response_store.close()
            except Exception:
                logger.debug(
                    "Failed to close response store for %s", self.name, exc_info=True,
                )
        try:
            if self._site:
                await self._site.stop()
                self._site = None
            if self._runner:
                await self._runner.cleanup()
                self._runner = None
        finally:
            self._close_cached_session_dbs()
            self._app = None
        logger.info("[%s] API server stopped", self.name)

    async def send(
        self,
        chat_id: str,
        content: str,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """
        Not used — HTTP request/response cycle handles delivery directly.
        """
        return SendResult(success=False, error="API server uses HTTP request/response, not send()")

    async def get_chat_info(self, chat_id: str) -> Dict[str, Any]:
        """Return basic info about the API server."""
        return {
            "name": "API Server",
            "type": "api",
            "host": self._host,
            "port": self._port,
        }
