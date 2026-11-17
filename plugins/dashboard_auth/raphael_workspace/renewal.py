"""Renewal routes: a still-valid Raphael service key asks for its own successor.

Owner decision of 16 November: while its current service key is still valid,
the owner app may get its own next one, and may retire the old one. For each
of the four fixed families (workspace, connections, automations, models) this
adds two token-only routes, each signed by a key of that family with its own
scope:

* ``POST /api/service-access/<surface>/renew`` answers
  ``{"token": "<id>.<secret>", "expires_at": int}``, never cached. The calling
  key stays valid. Refusals: ``409 {"error": "not_due"}`` and
  ``409 {"error": "already_renewed"}``.
* ``POST /api/service-access/<surface>/retire-older``, signed by the
  successor, revokes the older keys of its family: ``{"retired": int}``.

The checks live in :func:`token_store.renew` and :func:`token_store.retire_older`.
The new bearer appears only in the renew body; every other answer is a fixed
string, and nothing here logs or audits a token value or digest. Refusals of
the shared token middleware keep ``error`` and ``detail``; a refusal made here
carries ``error`` only. ``router`` carries the routes under
:data:`ROUTE_PREFIX`; ``hermes_cli/web_server.py`` mounts it.
"""

import logging
from typing import Optional

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from hermes_cli.dashboard_auth.audit import AuditEvent, AuditWriteError, audit_log
from hermes_cli.dashboard_auth.token_auth import (
    register_machine_token_family,
    register_token_route,
    transport_peer_ip,
)
from plugins.dashboard_auth.raphael_workspace import token_store

logger = logging.getLogger(__name__)

ROUTE_PREFIX = "/api/service-access"

#: The families that may renew themselves; a recommendations key never does.
RENEWAL_SURFACES = (
    token_store.WORKSPACE_SURFACE,
    token_store.CONNECTIONS_SURFACE,
    token_store.AUTOMATIONS_SURFACE,
    token_store.MODELS_SURFACE,
)

_NO_STORE = {"Cache-Control": "no-store"}

router = APIRouter(prefix=ROUTE_PREFIX)


def _paths(surface: str) -> "tuple[str, str]":
    """The renew and retire-older paths of one family, relative to ``ROUTE_PREFIX``."""
    return f"/{surface}/renew", f"/{surface}/retire-older"


def register_renewal_routes() -> None:
    """Token-only and strictly audited; each family reaches only its own two routes."""
    for surface in RENEWAL_SURFACES:
        policy = token_store.policy_for_surface(surface)
        register_machine_token_family(policy.token_prefix, strict_audit=True)
        for path in _paths(surface):
            register_token_route(
                ROUTE_PREFIX + path,
                method="POST",
                required_scope=policy.scope,
                strict_audit=True,
            )


def _answer(status_code: int, body: dict) -> JSONResponse:
    return JSONResponse(body, status_code=status_code, headers=_NO_STORE)


def _refusal(request: Request, policy: token_store.FixedTokenPolicy) -> Optional[JSONResponse]:
    """``None`` only for a key of this very family, verified by the token seam."""
    if not getattr(request.state, "token_authenticated", False):
        return _answer(401, {"error": "unauthenticated"})
    principal = getattr(request.state, "token_principal", None)
    credential_id = getattr(principal, "credential_id", None) or ""
    if (
        getattr(principal, "principal", None) != policy.principal
        or token_store.policy_for_token_id(credential_id) != policy
    ):
        return _answer(403, {"error": "forbidden"})
    return None


def _audit_success(request: Request, policy: token_store.FixedTokenPolicy, action: str) -> None:
    """Durably audit the answered request (ids only), as the strict seam requires."""
    principal = request.state.token_principal
    audit_log(
        AuditEvent.TOKEN_AUTH_SUCCESS,
        strict=True,
        provider=principal.provider,
        principal=principal.principal,
        credential_id=principal.credential_id,
        grant=policy.grant,
        source="raphael-credentials",
        action=action,
        method=request.method,
        route_template=getattr(request.state, "token_route_template", request.url.path),
        decision="allow",
        status=200,
        ip=transport_peer_ip(request),
    )
    request.state.token_route_audited = True


def _renew(request: Request, surface: str) -> JSONResponse:
    policy = token_store.policy_for_surface(surface)
    refusal = _refusal(request, policy)
    if refusal is not None:
        return refusal
    try:
        record, secret = token_store.renew(request.state.token_principal.credential_id)
        # A failure past this point leaves an undelivered successor; the
        # caller keeps its key, and the next renewal after 15 minutes
        # revokes that successor.
        _audit_success(request, policy, "renew")
    except token_store.NotDue:
        return _answer(409, {"error": "not_due"})
    except token_store.AlreadyRenewed:
        return _answer(409, {"error": "already_renewed"})
    except token_store.UnknownTokenError:
        return _answer(401, {"error": "unauthenticated"})
    except (token_store.TokenStoreError, AuditWriteError, OSError) as exc:
        logger.warning("raphael %s credential renewal failed: %s", surface, type(exc).__name__)
        return _answer(503, {"error": "unavailable"})
    return _answer(200, {"token": f"{record.token_id}.{secret}", "expires_at": record.expires_at})


def _retire_older(request: Request, surface: str) -> JSONResponse:
    policy = token_store.policy_for_surface(surface)
    refusal = _refusal(request, policy)
    if refusal is not None:
        return refusal
    try:
        retired = token_store.retire_older(request.state.token_principal.credential_id)
        _audit_success(request, policy, "retire-older")
    except token_store.UnknownTokenError:
        return _answer(401, {"error": "unauthenticated"})
    except (token_store.TokenStoreError, AuditWriteError, OSError) as exc:
        logger.warning("raphael %s credential retirement failed: %s", surface, type(exc).__name__)
        return _answer(503, {"error": "unavailable"})
    return _answer(200, {"retired": retired})


def _add_routes(surface: str) -> None:
    renew_path, retire_path = _paths(surface)

    def renew(request: Request) -> JSONResponse:
        return _renew(request, surface)

    def retire_older(request: Request) -> JSONResponse:
        return _retire_older(request, surface)

    router.add_api_route(
        renew_path, renew, methods=["POST"], name=f"raphael_{surface}_credential_renew"
    )
    router.add_api_route(
        retire_path, retire_older, methods=["POST"], name=f"raphael_{surface}_credential_retire_older"
    )


for _surface in RENEWAL_SURFACES:
    _add_routes(_surface)
