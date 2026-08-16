"""Route-agnostic non-interactive (bearer-token) auth seam for the dashboard.

This is the generic API-token capability (decisions.md Q-C): a reusable seam
that ANY service-to-service / machine-credential provider plugs into, NOT a
drain-specific hook. The drain bearer-secret plugin is merely the first
consumer.

How it fits the existing auth framework:

  * The interactive gate (``gated_auth_middleware``) authenticates a human
    via a session cookie on every non-public route. A service caller has no
    cookie — it presents a bearer token in the ``Authorization`` header on a
    single request. That is what this seam verifies.

  * A route opts in by registering its exact HTTP method + exact path + the
    single capability scope it demands, via :func:`register_token_route`.
    Only registered (method, path) pairs are token-authable; everything else
    is untouched, so this can never accidentally widen the auth surface of an
    existing route.

  * :func:`token_auth_middleware` runs OUTERMOST (installed last in
    ``web_server.py``). For a token route it fully owns the auth decision:
    authenticate via the stacked token providers, enforce the route's required
    scope on the returned principal, attach the verified
    :class:`~hermes_cli.dashboard_auth.base.TokenPrincipal` to
    ``request.state.token_principal`` + set ``request.state.token_authenticated``,
    and pass through; otherwise reject (401 unauthenticated, 403 authenticated
    but not authorised for this route, or 503 when a provider's backing store
    was unreachable). The downstream cookie/session gates honour
    ``token_authenticated`` and skip enforcement, so a token-authed service
    request is never bounced to ``/login``.

  * Fails closed: a token route with no registered token provider, no token,
    or an unrecognised token gets 401 — never an open pass-through.

Authority boundary (Item 31BI)
------------------------------
A route policy is (method, path, required_scope), never a bare path. Being
recognised by SOME provider is not authorisation: the principal must also
carry the route's exact required scope, otherwise the seam answers a generic
403 and the request never reaches the handler. That keeps each machine
credential confined to its own endpoint — the drain secret cannot read Kanban
recommendations and a recommendations secret cannot drive drain control — even
though both are verified by the same stack of token providers. The 403 body is
deliberately generic: it discloses neither the scopes the caller holds nor the
scope the route wants, and it never falls back to another provider or to
cookie/session auth.

Provider stacking mirrors ``verify_session``: each ``supports_token`` provider
is consulted in registration order until one returns a principal. A provider
that doesn't recognise the token returns ``None`` and the seam moves on; a
provider whose backing store is unreachable raises ``ProviderError``, which the
seam remembers and surfaces as 503 only if NO provider accepts the token.
"""
from __future__ import annotations

import logging
import re
import threading
from dataclasses import dataclass
from typing import Awaitable, Callable, Dict, Optional, Tuple

from fastapi import Request
from fastapi.responses import JSONResponse, Response

from hermes_cli.dashboard_auth import list_token_providers
from hermes_cli.dashboard_auth.audit import AuditEvent, audit_log
from hermes_cli.dashboard_auth.base import ProviderError, TokenPrincipal

_log = logging.getLogger(__name__)

# An HTTP method is a bare alphabetic token (GET, POST, DELETE, ...). Anything
# else is a registration bug, not a route.
_METHOD_RE = re.compile(r"^[A-Z]+$")


@dataclass(frozen=True)
class TokenRoutePolicy:
    """The immutable authority a single token route grants.

    ``method`` + ``path`` identify the route exactly (no prefix, no pattern,
    no method wildcard); ``required_scope`` is the one capability a verified
    principal must carry to be allowed through. All three are mandatory —
    there is no such thing as an unscoped token route.
    """

    method: str
    path: str
    required_scope: str


# Registered token routes, keyed by (METHOD, exact path). A route registers
# itself at import/startup; the seam only acts on registered routes.
_token_routes: Dict[Tuple[str, str], TokenRoutePolicy] = {}
_lock = threading.Lock()


def _normalize_method(method: str) -> str:
    """Uppercase + validate an HTTP method, or raise ``ValueError``."""
    normalized = (method or "").strip().upper()
    if not _METHOD_RE.match(normalized):
        raise ValueError(
            f"token route method must be a single HTTP method, got {method!r}"
        )
    return normalized


def register_token_route(path: str, *, method: str, required_scope: str) -> None:
    """Mark ``method`` + ``path`` (both exact) as token-authable under a scope.

    Call at module import / app setup so the seam knows which routes to guard.
    Registering a route does NOT make it public — it makes it authenticate by
    token, and only by a token whose principal carries ``required_scope``,
    instead of by session cookie.

    Thread-safe and idempotent: re-registering the identical policy is a no-op.
    Registering the same (method, path) with a DIFFERENT required scope raises
    ``ValueError`` rather than silently replacing the policy, so two callers
    can never race the route into a weaker authority.

    Raises ``ValueError`` for a blank/multi-token method, a blank path, or a
    blank required scope — a route with no scope would be exactly the
    any-recognised-token hole this seam exists to close.
    """
    normalized_method = _normalize_method(method)
    normalized_path = (path or "").strip()
    if not normalized_path:
        raise ValueError("token route path must be non-empty")
    scope = (required_scope or "").strip()
    if not scope:
        raise ValueError(
            f"token route {normalized_method} {normalized_path} needs a "
            "non-empty required scope"
        )
    policy = TokenRoutePolicy(
        method=normalized_method, path=normalized_path, required_scope=scope
    )
    key = (normalized_method, normalized_path)
    with _lock:
        existing = _token_routes.get(key)
        if existing is not None and existing != policy:
            raise ValueError(
                f"token route {normalized_method} {normalized_path} is already "
                f"registered with a different required scope"
            )
        _token_routes[key] = policy


def get_token_route_policy(method: str, path: str) -> Optional[TokenRoutePolicy]:
    """Return the policy registered for this exact method + path, else ``None``.

    A method mismatch is simply not a token route: the lookup misses and the
    seam passes the request through to the ordinary gates, which is strictly
    narrower than token auth — it can never broaden access.
    """
    try:
        key = (_normalize_method(method), (path or "").strip())
    except ValueError:
        return None
    with _lock:
        return _token_routes.get(key)


def is_token_route(path: str, *, method: str) -> bool:
    """True if this exact method + path was registered as token-authable."""
    return get_token_route_policy(method, path) is not None


def clear_token_routes() -> None:
    """Test-only: drop all registered token routes."""
    with _lock:
        _token_routes.clear()


def _client_ip(request: Request) -> str:
    fwd = request.headers.get("x-forwarded-for", "")
    if fwd:
        return fwd.split(",")[0].strip()
    return request.client.host if request.client else ""


def extract_bearer_token(request: Request) -> str:
    """Return the bearer token from the ``Authorization`` header, or "".

    Accepts ``<scheme> <token>`` where scheme is "bearer" (case-insensitive).
    Returns an empty string for a missing/malformed header or a non-bearer
    scheme — the caller treats "" as "no token presented".
    """
    auth = request.headers.get("authorization", "")
    parts = auth.split(" ", 1)
    if len(parts) == 2 and parts[0].strip().lower() == "bearer":
        return parts[1].strip()
    return ""


def authenticate_token(
    request: Request,
) -> Tuple[Optional[TokenPrincipal], Optional[str]]:
    """Try every token provider against the request's bearer token.

    Returns ``(principal, unreachable_provider_name)``:
      * ``(TokenPrincipal, None)`` — a provider recognised and accepted the token.
      * ``(None, None)`` — no token, or no provider recognised it (reject 401).
      * ``(None, name)`` — no provider accepted it AND at least one provider's
        backing store was unreachable (the caller surfaces 503, not 401, so a
        transient outage doesn't read as "bad credentials").

    Never raises: a provider ``ProviderError`` is caught and remembered.
    """
    token = extract_bearer_token(request)
    if not token:
        return None, None
    unreachable: Optional[str] = None
    for provider in list_token_providers():
        try:
            principal = provider.verify_token(token=token)
        except ProviderError as e:
            _log.warning(
                "dashboard-auth: token provider %r unreachable during verify: %s",
                provider.name, e,
            )
            if unreachable is None:
                unreachable = provider.name
            continue
        except Exception as e:  # noqa: BLE001 — a buggy provider must not 500 the gate
            _log.warning(
                "dashboard-auth: token provider %r raised during verify: %s",
                provider.name, e,
            )
            continue
        if principal is not None:
            return principal, None
    return None, unreachable


async def token_auth_middleware(
    request: Request,
    call_next: Callable[[Request], Awaitable[Response]],
) -> Response:
    """Outermost auth seam for token-authable routes.

    No-op pass-through for any request whose exact method + path was not
    registered via :func:`register_token_route`. For a registered route, token
    auth is the only accepted scheme:

      * valid token, has the route's required scope
                     → attach principal + ``token_authenticated`` flag, pass through.
      * valid token, missing that scope
                     → 403 (generic; no scope contents disclosed, no fallback
                       to another provider or to cookie/session auth).
      * unreachable  → 503 (provider backing store down; not "bad credentials").
      * otherwise    → 401 unauthenticated.

    The policy is resolved BEFORE any authentication happens, and the token is
    authenticated exactly once; the scope check runs immediately after a
    provider recognises it.

    Runs before the cookie/session gates (installed last in ``web_server.py``).
    The cookie gates honour ``request.state.token_authenticated`` and skip
    enforcement, so a token-authed request is never redirected to ``/login``.
    """
    path = request.url.path
    policy = get_token_route_policy(request.method, path)
    if policy is None:
        return await call_next(request)

    principal, unreachable = authenticate_token(request)
    if principal is not None:
        if policy.required_scope not in principal.scopes:
            # Recognised credential, wrong authority. Audit it as a token-auth
            # failure with identifiers only — never the presented credential,
            # the principal's scopes, or the scope the route wants.
            audit_log(
                AuditEvent.TOKEN_AUTH_FAILURE,
                provider=principal.provider,
                principal=principal.principal,
                reason="missing_required_scope",
                method=policy.method,
                path=path,
                ip=_client_ip(request),
            )
            return JSONResponse(
                {"error": "forbidden", "detail": "Forbidden"},
                status_code=403,
            )
        request.state.token_principal = principal
        request.state.token_authenticated = True
        return await call_next(request)

    if unreachable:
        audit_log(
            AuditEvent.TOKEN_AUTH_FAILURE,
            provider=unreachable,
            reason="provider_unreachable",
            method=policy.method,
            path=path,
            ip=_client_ip(request),
        )
        return JSONResponse(
            {"detail": f"Auth provider {unreachable!r} unreachable"},
            status_code=503,
        )

    audit_log(
        AuditEvent.TOKEN_AUTH_FAILURE,
        reason="no_provider_recognises_token",
        method=policy.method,
        path=path,
        ip=_client_ip(request),
    )
    return JSONResponse(
        {"error": "unauthenticated", "detail": "Unauthorized"},
        status_code=401,
    )
