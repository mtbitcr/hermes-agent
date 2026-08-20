"""Fail-closed model policy for Raphael's owner-facing Models surface.

The native Hermes profile config remains the only runtime model authority.
This module adds no registry, router, credential store, or fallback engine: it
only narrows the existing OAuth/model/profile endpoints for Raphael's dedicated
machine credential and validates the four admitted role assignments.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional

from fastapi import HTTPException, Request

from hermes_cli.dashboard_auth.audit import AuditEvent, AuditWriteError, audit_log
from hermes_cli.dashboard_auth.token_auth import (
    register_machine_token_family,
    register_token_route,
    register_token_route_template,
    transport_peer_ip,
)
from plugins.dashboard_auth.raphael_workspace import token_store


@dataclass(frozen=True)
class ModelAssignment:
    profile: str
    provider: str
    model: str
    reasoning_effort: str


_ASSIGNMENTS = {
    ("raphael-planner", "anthropic"): ModelAssignment(
        "raphael-planner", "anthropic", "claude-sonnet-5", "max"
    ),
    ("default", "anthropic"): ModelAssignment(
        "default", "anthropic", "claude-opus-5", "max"
    ),
    ("raphael-planner", "openai-codex"): ModelAssignment(
        "raphael-planner", "openai-codex", "gpt-5.6-sol", "high"
    ),
    ("default", "openai-codex"): ModelAssignment(
        "default", "openai-codex", "gpt-5.6-sol", "xhigh"
    ),
}


def assignment_for(profile: str, provider: str) -> ModelAssignment:
    """Return the one admitted assignment for ``profile`` and ``provider``."""
    key = (str(profile or "").strip(), str(provider or "").strip())
    try:
        return _ASSIGNMENTS[key]
    except KeyError as exc:
        raise ValueError("unadmitted Raphael model assignment") from exc


def validate_assignment(
    profile: str,
    provider: str,
    model: str,
    reasoning_effort: str,
    *,
    disable_fallbacks: bool,
) -> ModelAssignment:
    """Reject anything except an exact, fallback-free admitted assignment."""
    expected = assignment_for(profile, provider)
    if (
        str(model or "").strip() != expected.model
        or str(reasoning_effort or "").strip().lower() != expected.reasoning_effort
        or disable_fallbacks is not True
    ):
        raise ValueError("unadmitted Raphael model assignment")
    return expected


def admitted_provider_ids() -> tuple[str, ...]:
    return ("anthropic", "openai-codex")


def model_machine_request(request: Any) -> bool:
    """True only for a verified bearer from the Models token provider."""
    state = getattr(request, "state", None)
    if not getattr(state, "token_authenticated", False):
        return False
    principal = getattr(state, "token_principal", None)
    return getattr(principal, "provider", None) == "raphael-models-token"


def require_models_machine_profile(
    request: Any,
    profile: Optional[str],
    *,
    allowed: tuple[Optional[str], ...],
) -> Optional[str]:
    """Bind Models-machine calls to the explicitly admitted profile set."""
    if not model_machine_request(request):
        return profile
    requested = str(profile or "").strip()
    canonical = None if not requested or requested.lower() == "current" else requested
    if canonical not in allowed:
        raise HTTPException(status_code=400, detail="Invalid profile")
    return canonical


def project_oauth_start(provider: str, payload: Mapping[str, Any]) -> dict:
    """Whitelist only fields required to finish the admitted native flow."""
    result = {
        "session_id": payload.get("session_id"),
        "flow": payload.get("flow"),
        "expires_in": payload.get("expires_in"),
    }
    if provider == "anthropic":
        result["auth_url"] = payload.get("auth_url")
    elif provider == "openai-codex":
        result.update({
            "user_code": payload.get("user_code"),
            "verification_url": payload.get("verification_url"),
            "poll_interval": payload.get("poll_interval"),
        })
    return result


def project_oauth_result(payload: Mapping[str, Any]) -> dict:
    """Return only the stable OAuth completion fields consumed by Workspace."""
    status = str(payload.get("status") or "")
    if status not in {"pending", "approved", "denied", "expired", "error"}:
        status = "error"
    return {"ok": payload.get("ok") is True, "status": status}


def project_oauth_poll(payload: Mapping[str, Any]) -> dict:
    """Remove provider error text and timing internals from machine polling."""
    status = str(payload.get("status") or "")
    if status not in {"pending", "approved", "denied", "expired", "error"}:
        status = "error"
    return {
        "session_id": str(payload.get("session_id") or ""),
        "status": status,
    }


def project_oauth_payload(payload: Mapping[str, Any]) -> dict:
    """Project the native provider list to non-secret admitted status only."""
    allowed = set(admitted_provider_ids())
    providers = []
    raw_rows = payload.get("providers")
    if isinstance(raw_rows, list):
        for raw in raw_rows:
            if not isinstance(raw, Mapping):
                continue
            provider_id = str(raw.get("id") or "")
            if provider_id not in allowed:
                continue
            status = raw.get("status")
            logged_in = bool(
                isinstance(status, Mapping) and status.get("logged_in") is True
            )
            providers.append({
                "id": provider_id,
                "name": "Anthropic" if provider_id == "anthropic" else "OpenAI",
                "flow": "pkce" if provider_id == "anthropic" else "device_code",
                "status": {"logged_in": logged_in},
            })
    return {"providers": providers}


def project_options_payload(payload: Mapping[str, Any]) -> dict:
    """Keep only models that are admitted for at least one Raphael role."""
    admitted = {
        provider: {
            assignment.model
            for assignment in _ASSIGNMENTS.values()
            if assignment.provider == provider
        }
        for provider in admitted_provider_ids()
    }
    rows = []
    raw_rows = payload.get("providers")
    if isinstance(raw_rows, list):
        for raw in raw_rows:
            if not isinstance(raw, Mapping):
                continue
            provider = str(raw.get("slug") or raw.get("id") or "")
            if provider not in admitted:
                continue
            models = raw.get("models")
            safe_models = []
            if isinstance(models, list):
                for item in models:
                    model_id = (
                        str(item.get("id") or item.get("model") or "")
                        if isinstance(item, Mapping)
                        else str(item or "")
                    )
                    if model_id in admitted[provider]:
                        safe_models.append(model_id)
            rows.append({
                "slug": provider,
                "authenticated": raw.get("authenticated") is True,
                "models": safe_models,
            })
    return {"providers": rows}


_LITERAL_ROUTES = (
    ("GET", "/api/providers/oauth"),
    ("POST", "/api/providers/oauth/anthropic/start"),
    ("POST", "/api/providers/oauth/openai-codex/start"),
    ("POST", "/api/providers/oauth/anthropic/submit"),
    ("DELETE", "/api/providers/oauth/anthropic"),
    ("DELETE", "/api/providers/oauth/openai-codex"),
    ("GET", "/api/model/options"),
    ("GET", "/api/model/info"),
    ("PUT", "/api/profiles/raphael-planner/model"),
    ("PUT", "/api/profiles/default/model"),
)
_TEMPLATE_ROUTES = (("GET", "/api/providers/oauth/openai-codex/poll/{session_id}"),)


def register_models_machine_routes() -> None:
    register_machine_token_family(token_store.MODELS_TOKEN_PREFIX, strict_audit=True)
    for method, path in _LITERAL_ROUTES:
        register_token_route(
            path,
            method=method,
            required_scope=token_store.MODELS_SCOPE,
            optional=True,
            strict_audit=True,
        )
    for method, path in _TEMPLATE_ROUTES:
        register_token_route_template(
            path,
            method=method,
            required_scope=token_store.MODELS_SCOPE,
            optional=True,
            strict_audit=True,
        )


def audit_models_machine_success(
    request: Request,
    *,
    action: str,
    profile: Optional[str] = None,
    provider: Optional[str] = None,
) -> None:
    """Durably audit a successful Models-machine request, or fail closed."""
    if not model_machine_request(request):
        return
    principal = request.state.token_principal
    try:
        audit_log(
            AuditEvent.TOKEN_AUTH_SUCCESS,
            strict=True,
            provider=principal.provider,
            principal=principal.principal,
            credential_id=principal.credential_id,
            grant=token_store.MODELS_GRANT,
            source="raphael-models",
            method=request.method,
            route_template=getattr(
                request.state, "token_route_template", request.url.path
            ),
            action=action,
            profile=profile,
            model_provider=provider,
            decision="allow",
            status=200,
            ip=transport_peer_ip(request),
        )
        request.state.token_route_audited = True
    except AuditWriteError as exc:
        raise HTTPException(status_code=503, detail="Service Unavailable") from exc
