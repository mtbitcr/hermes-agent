"""Connection rows: one read-only row per saved connection, for the owner app.

``GET /api/mcp/connection-rows`` answers the owner app's connections
credential and no other caller. It reads the saved connections of a fixed list
of profiles and answers names and states only: never an address, command,
arguments, environment values or headers. It writes, refreshes, signs in,
enables and disables nothing.
"""

import asyncio
from typing import Any, Dict, List, Optional, Tuple

from fastapi import APIRouter, HTTPException, Request

from hermes_cli.dashboard_auth.token_auth import register_machine_token_family, register_token_route
from hermes_cli.web_deps import late
from hermes_cli.web_routers.mcp import (
    _CLAUDE_DESIGN_CONNECTION,
    _audit_connections_machine_success,
    _enforce_connections_machine_profile,
)
from plugins.dashboard_auth.raphael_workspace import (
    CONNECTIONS_PRINCIPAL,
    CONNECTIONS_SCOPE,
    CONNECTIONS_TOKEN_PREFIX,
)

router = APIRouter()

# Late-bound web_server helpers (resolved at call time; cycle-safe,
# monkeypatch-transparent).
_mcp_server_summary = late("_mcp_server_summary")
_profile_scope = late("_profile_scope")

_ROUTE = "/api/mcp/connection-rows"

#: The only profiles read, each with the role its rows carry. A fixed list on
#: purpose: ``_enforce_connections_machine_profile`` deliberately lets this
#: credential reach nothing wider, and it is not loosened for this read.
_ROW_PROFILES = {
    "default": "owner",
    "raphael-designer": "designer",
    "raphael-builder": "agent",
    "raphael-planner": "agent",
    "raphael-verifier": "agent",
    "raphael-claude-worker": "agent",
}

#: Owner decision of 2026-10-31: the root profile holds only Gmail, Calendar and
#: Google Drive Work. These were removed from it and never appear as root rows.
_REMOVED_ROOT_CONNECTIONS = frozenset({"microsoft-personal", "github-bitcredit", "google-calendar-personal"})

_SIGNS_IN_ONLINE = frozenset({"oauth", "code_sign_in"})


def _register_connection_rows_route() -> None:
    """Token-only (no session), and only for a token carrying the connections scope."""
    register_machine_token_family(CONNECTIONS_TOKEN_PREFIX, strict_audit=True)
    register_token_route(_ROUTE, method="GET", required_scope=CONNECTIONS_SCOPE, strict_audit=True)


_register_connection_rows_route()


def _require_connections_credential(request: Request) -> None:
    """The token seam already refuses every other caller; refuse them here too."""
    if not getattr(request.state, "token_authenticated", False):
        raise HTTPException(status_code=401, detail="Unauthorized")
    principal = getattr(request.state, "token_principal", None)
    if getattr(principal, "principal", None) != CONNECTIONS_PRINCIPAL:
        raise HTTPException(status_code=403, detail="Forbidden")


def _sign_in_kind(name: str, transport: str, auth: Optional[str]) -> str:
    if transport == "stdio":
        return "local_command"
    if transport != "http":
        return "unknown"
    if auth == "oauth":
        return "code_sign_in" if name == _CLAUDE_DESIGN_CONNECTION else "oauth"
    if auth == "header":
        return "header_key"
    return "none" if auth in (None, "", "none") else "unknown"


def _status(sign_in_kind: str, enabled: bool, signed_in: bool) -> str:
    """From the enabled flag and the saved sign-in alone, so never ``connected``.

    Neither tells a working connection from one the gateway has parked, so a
    saved sign-in reads ``configured``.
    """
    if not enabled:
        return "paused"
    if sign_in_kind in _SIGNS_IN_ONLINE:
        return "configured" if signed_in else "needs_authorization"
    return "unknown" if sign_in_kind == "unknown" else "configured"


def _read_profile(profile: str) -> List[Tuple[str, str, str, bool, bool]]:
    """``(name, transport, sign_in_kind, enabled, signed_in)`` per saved connection.

    Nothing for a profile that does not exist on this host.
    """
    from hermes_cli.mcp_catalog import server_enabled
    from hermes_cli.mcp_config import _get_mcp_servers, _oauth_tokens_present

    try:
        with _profile_scope(profile):
            saved = []
            for name, cfg in sorted(_get_mcp_servers().items()):
                if profile == "default" and name in _REMOVED_ROOT_CONNECTIONS:
                    continue
                summary = _mcp_server_summary(name, cfg)
                kind = _sign_in_kind(name, summary["transport"], summary["auth"])
                signed_in = kind in _SIGNS_IN_ONLINE and _oauth_tokens_present(name)
                saved.append((name, summary["transport"], kind, server_enabled(cfg), signed_in))
            return saved
    except HTTPException as exc:
        if exc.status_code == 404:
            return []
        raise


def _read_only(request: Request, profile: str, role: str, name: str) -> bool:
    """True for every agent row, and wherever the credential's own gate refuses a change."""
    if role == "agent":
        return True
    try:
        _enforce_connections_machine_profile(
            request, name=name, profile=None if profile == "default" else profile
        )
    except HTTPException:
        return True
    return False


@router.get(_ROUTE)
async def list_connection_rows(request: Request):
    """One row per saved connection of each listed profile: names and states only."""
    _require_connections_credential(request)
    from hermes_cli.mcp_catalog import list_catalog

    def _read() -> List[Dict[str, Any]]:
        catalogue = {entry.name for entry in list_catalog()}
        rows = []
        for profile, role in _ROW_PROFILES.items():
            for name, transport, kind, enabled, signed_in in _read_profile(profile):
                rows.append({
                    "name": name,
                    "profile": profile,
                    "profile_role": role,
                    "transport": transport,
                    "sign_in_kind": kind,
                    "status": _status(kind, enabled, signed_in),
                    "enabled": enabled,
                    "read_only": _read_only(request, profile, role, name),
                    "catalogue": name in catalogue,
                })
        return rows

    result = {"rows": await asyncio.to_thread(_read)}
    _audit_connections_machine_success(request)
    return result
