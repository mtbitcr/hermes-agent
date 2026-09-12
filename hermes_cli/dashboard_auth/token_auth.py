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
    Only registered (method, path) pairs are token-authable. Ordinary requests
    to everything else are untouched; a bearer using a reserved machine-token
    family is instead denied on any unregistered path/method so redirects or
    method confusion cannot escape that family's fixed contour.

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
from hermes_cli.dashboard_auth.audit import AuditEvent, AuditWriteError, audit_log
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

    ``optional`` (default ``False``) marks a route shared with the interactive
    dashboard. On such a route only a bearer matching a registered machine
    token family enters this seam; missing bearer tokens and ordinary session
    bearers continue to the existing cookie/session gate unchanged.

      * ``False`` (drain, recommendations, and every route registered before
        Item 32G-A) — absence of a token is fail-closed 401, exactly as
        documented above. There is no interactive equivalent of these routes,
        so there is nothing to fall back to.
      * ``True`` — this route also serves interactive callers. A registered
        machine-family prefix selects machine auth; every other bearer remains
        available to the pre-existing session-bearer implementation.

    ``strict_audit`` is opt-in. When true, an audit write failure returns a
    generic 503 before the machine request can reach its handler.
    """

    method: str
    path: str
    required_scope: str
    optional: bool = False
    strict_audit: bool = False


@dataclass(frozen=True)
class MachineTokenFamily:
    """A syntactically identifiable machine credential family.

    Shared interactive routes need this discriminator so an ordinary
    dashboard session bearer is never mistaken for a failed machine token.
    Prefixes classify credentials only; providers still authenticate the full
    token and route policies still enforce scope.
    """

    prefix: str
    strict_audit: bool = False


# Registered token routes, keyed by (METHOD, exact path). A route registers
# itself at import/startup; the seam only acts on registered routes.
_token_routes: Dict[Tuple[str, str], TokenRoutePolicy] = {}
# Registered token route TEMPLATES, for routes with one or more named path
# segments (e.g. "/api/plugins/kanban/tasks/{task_id}") that a plain exact-path
# dict lookup cannot express. Populated by :func:`register_token_route_template`;
# consulted by :func:`get_token_route_policy` only when the exact-literal dict
# above misses, so every existing literal registration is completely unaffected.
# Entries are ``(METHOD, compiled anchored regex, policy)``.
_token_route_templates: list = []
# Keyed by (METHOD, raw template string) — mirrors ``_token_routes`` so
# re-registering the identical template is idempotent and a conflicting
# re-registration (different scope/optional) raises, exactly like the literal
# path registry.
_token_route_template_index: Dict[Tuple[str, str], TokenRoutePolicy] = {}
_machine_token_families: Dict[str, MachineTokenFamily] = {}
_lock = threading.Lock()

# A "{name}" path template segment: one named, non-empty, non-slash segment —
# mirrors FastAPI/Starlette's default (untyped) path-param converter. Anything
# outside this shape (prefixes, "*", regex metacharacters) is not a supported
# template; templates stay exact shapes with named holes, never wildcards.
_TEMPLATE_PARAM_RE = re.compile(r"\{([a-zA-Z_][a-zA-Z0-9_]*)\}")
_MACHINE_PREFIX_RE = re.compile(r"^[A-Za-z0-9_-]{4,32}$")


def _normalize_method(method: str) -> str:
    """Uppercase + validate an HTTP method, or raise ``ValueError``."""
    normalized = (method or "").strip().upper()
    if not _METHOD_RE.match(normalized):
        raise ValueError(
            f"token route method must be a single HTTP method, got {method!r}"
        )
    return normalized


def register_token_route(
    path: str,
    *,
    method: str,
    required_scope: str,
    optional: bool = False,
    strict_audit: bool = False,
) -> None:
    """Mark ``method`` + ``path`` (both exact) as token-authable under a scope.

    Call at module import / app setup so the seam knows which routes to guard.
    Registering a route does NOT make it public — it makes it authenticate by
    token, and only by a token whose principal carries ``required_scope``,
    instead of by session cookie.

    Thread-safe and idempotent: re-registering the identical policy is a no-op.
    Registering the same (method, path) with a DIFFERENT required scope (or a
    different ``optional``/``strict_audit``) raises ``ValueError`` rather than silently
    replacing the policy, so two callers can never race the route into a
    weaker authority.

    Raises ``ValueError`` for a blank/multi-token method, a blank path, or a
    blank required scope — a route with no scope would be exactly the
    any-recognised-token hole this seam exists to close.

    ``optional=True`` (default ``False``) is for a route that ALSO serves
    interactive session/cookie callers — see :class:`TokenRoutePolicy`. Every
    existing caller omits it and keeps the original pure-token-only, fail-
    closed-on-no-token behaviour unchanged.
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
        method=normalized_method,
        path=normalized_path,
        required_scope=scope,
        optional=bool(optional),
        strict_audit=bool(strict_audit),
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


def _compile_token_route_template(path_template: str) -> "re.Pattern[str]":
    """Compile an exact-match, fully-anchored regex from a ``{name}`` template.

    Each ``{name}`` segment matches exactly one non-empty, non-slash path
    segment (mirroring FastAPI/Starlette's default untyped path-param
    converter); every other character is matched literally
    (``re.escape``d). The result is anchored with ``^...$``, so a template is
    still an EXACT shape — no prefix, no suffix, no wildcard — just with named
    variable segments where a literal path has none.
    """
    parts: list = []
    last_end = 0
    for m in _TEMPLATE_PARAM_RE.finditer(path_template):
        parts.append(re.escape(path_template[last_end : m.start()]))
        parts.append("[^/]+")
        last_end = m.end()
    parts.append(re.escape(path_template[last_end:]))
    return re.compile("^" + "".join(parts) + "$")


def register_token_route_template(
    path_template: str,
    *,
    method: str,
    required_scope: str,
    optional: bool = False,
    strict_audit: bool = False,
) -> None:
    """Like :func:`register_token_route`, for a path with named segments.

    ``path_template`` uses ``{name}`` for a single dynamic path segment, e.g.
    ``"/api/plugins/kanban/tasks/{task_id}"``. Use this ONLY when the path
    genuinely has a variable segment; a literal path with no ``{...}`` must go
    through :func:`register_token_route` instead (this function rejects a
    template with no placeholder).

    Matching happens by exact-literal lookup first, this template list only on
    a miss — see :data:`_token_route_templates` — so no existing literal
    registration (drain, recommendations) is affected by this function
    existing at all, let alone by any particular template registered here.

    Same idempotency/conflict/validation contract as
    :func:`register_token_route`.
    """
    normalized_method = _normalize_method(method)
    template = (path_template or "").strip()
    if not template:
        raise ValueError("token route template must be non-empty")
    segments = template.split("/")
    placeholders = [
        segment for segment in segments if _TEMPLATE_PARAM_RE.fullmatch(segment)
    ]
    if not placeholders:
        raise ValueError(
            f"token route template {template!r} has no {{param}} segment; "
            "use register_token_route() for a literal exact path"
        )
    if any(
        ("{" in segment or "}" in segment)
        and _TEMPLATE_PARAM_RE.fullmatch(segment) is None
        for segment in segments
    ):
        raise ValueError("token route template placeholders must occupy a full segment")
    scope = (required_scope or "").strip()
    if not scope:
        raise ValueError(
            f"token route template {normalized_method} {template} needs a "
            "non-empty required scope"
        )
    policy = TokenRoutePolicy(
        method=normalized_method,
        path=template,
        required_scope=scope,
        optional=bool(optional),
        strict_audit=bool(strict_audit),
    )
    key = (normalized_method, template)
    compiled = _compile_token_route_template(template)
    with _lock:
        existing = _token_route_template_index.get(key)
        if existing is not None and existing != policy:
            raise ValueError(
                f"token route template {normalized_method} {template} is "
                "already registered with a different policy"
            )
        if existing is None:
            if any(
                method == normalized_method and pattern.pattern == compiled.pattern
                for method, pattern, _ in _token_route_templates
            ):
                raise ValueError(
                    f"token route template {normalized_method} {template} "
                    "duplicates an existing route shape"
                )
            _token_route_templates.append((normalized_method, compiled, policy))
        _token_route_template_index[key] = policy


def register_machine_token_family(prefix: str, *, strict_audit: bool = False) -> None:
    """Register a non-secret prefix that identifies one machine-token family.

    Prefixes are intentionally short, fixed protocol markers (not secrets).
    Overlapping prefixes are rejected so classification cannot depend on
    registration order. Registration is thread-safe and idempotent.
    """
    normalized = (prefix or "").strip()
    if not _MACHINE_PREFIX_RE.fullmatch(normalized):
        raise ValueError(
            "machine token prefix must be 4-32 ASCII letters, numbers, _ or -"
        )
    family = MachineTokenFamily(normalized, bool(strict_audit))
    with _lock:
        existing = _machine_token_families.get(normalized)
        if existing is not None and existing != family:
            raise ValueError(
                f"machine token family {normalized!r} is already registered "
                "with a different policy"
            )
        for other in _machine_token_families:
            if other != normalized and (
                other.startswith(normalized) or normalized.startswith(other)
            ):
                raise ValueError(
                    f"machine token prefix {normalized!r} overlaps {other!r}"
                )
        _machine_token_families[normalized] = family


def get_machine_token_family(token: str) -> Optional[MachineTokenFamily]:
    """Return the registered family whose prefix matches ``token``."""
    if not token:
        return None
    with _lock:
        for prefix, family in _machine_token_families.items():
            if token.startswith(prefix):
                return family
    return None


def get_token_route_policy(method: str, path: str) -> Optional[TokenRoutePolicy]:
    """Return the policy registered for this exact method + path, else ``None``.

    A method mismatch is simply not a token route: the lookup misses and the
    seam passes the request through to the ordinary gates, which is strictly
    narrower than token auth — it can never broaden access.

    Checks the exact-literal registry first; only on a miss does it try the
    registered path TEMPLATES (dynamic segments), in registration order. A
    literal registration therefore always wins and is never shadowed by a
    template, and a path that matches no literal and no template is not a
    token route at all.
    """
    try:
        normalized_method = _normalize_method(method)
    except ValueError:
        return None
    # Request paths are authority identifiers. Never trim a decoded trailing
    # space into a different registered route.
    normalized_path = path or ""
    with _lock:
        policy = _token_routes.get((normalized_method, normalized_path))
        if policy is not None:
            return policy
        for tmpl_method, pattern, tmpl_policy in _token_route_templates:
            if tmpl_method == normalized_method and pattern.match(normalized_path):
                return tmpl_policy
    return None


def is_token_route(path: str, *, method: str) -> bool:
    """True if this exact method + path was registered as token-authable."""
    return get_token_route_policy(method, path) is not None


def clear_token_routes() -> None:
    """Test-only: drop all token routes and machine-family classifiers."""
    with _lock:
        _token_routes.clear()
        _token_route_templates.clear()
        _token_route_template_index.clear()
        _machine_token_families.clear()


def transport_peer_ip(request: Request) -> str:
    """Return only the immediate transport peer.

    Forwarded headers are attacker-controlled unless a separately configured
    trusted-proxy boundary proves the immediate peer. Dashboard auth has no
    such policy, so machine audit must ignore them.
    """
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


def _audit_machine_failure(
    request: Request,
    *,
    reason: str,
    method: str,
    route_template: str,
    strict: bool,
    principal: Optional[TokenPrincipal] = None,
    provider: Optional[str] = None,
    status_code: int,
) -> Optional[Response]:
    """Write one bounded denial audit, returning 503 if strict audit failed."""
    def bounded(value: str, limit: int = 256) -> str:
        return "".join(
            char if " " <= char <= "~" else "?" for char in str(value)
        )[:limit]

    try:
        audit_log(
            AuditEvent.TOKEN_AUTH_FAILURE,
            strict=strict,
            provider=provider or (principal.provider if principal else None),
            principal=principal.principal if principal else None,
            credential_id=principal.credential_id if principal else None,
            reason=reason,
            decision="deny",
            status=status_code,
            method=bounded(method, 16),
            route_template=bounded(route_template),
            path=bounded(request.url.path),
            ip=transport_peer_ip(request),
        )
    except AuditWriteError:
        return JSONResponse(
            {"error": "unavailable", "detail": "Service Unavailable"},
            status_code=503,
        )
    return None


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
    bearer = extract_bearer_token(request)
    family = get_machine_token_family(bearer)
    policy = get_token_route_policy(request.method, path)

    # A recognised machine-family marker may never exploit Starlette redirects,
    # a method-override convention, or an unregistered path/method to fall into
    # interactive auth. Prefix classification is not authentication; it is only
    # a fail-closed dispatch decision for this reserved credential protocol.
    if family is not None and any(
        name in request.headers
        for name in ("x-http-method-override", "x-method-override", "x-http-method")
    ):
        unavailable = _audit_machine_failure(
            request,
            reason="method_override_not_allowed",
            method=request.method,
            route_template=policy.path if policy else path,
            strict=family.strict_audit,
            status_code=400,
        )
        if unavailable is not None:
            return unavailable
        return JSONResponse(
            {"error": "bad_request", "detail": "Bad Request"}, status_code=400
        )

    if policy is None:
        if family is not None:
            unavailable = _audit_machine_failure(
                request,
                reason="route_or_method_not_allowed",
                method=request.method,
                route_template=path,
                strict=family.strict_audit,
                status_code=403,
            )
            if unavailable is not None:
                return unavailable
            return JSONResponse(
                {"error": "forbidden", "detail": "Forbidden"}, status_code=403
            )
        return await call_next(request)

    if policy.optional and family is None:
        # Existing interactive cookie/session-bearer request. The machine seam
        # does not authenticate or reject it; downstream behavior is unchanged.
        return await call_next(request)

    strict_audit = policy.strict_audit or bool(family and family.strict_audit)

    principal, unreachable = authenticate_token(request)
    if principal is not None:
        if policy.required_scope not in principal.scopes:
            # Recognised credential, wrong authority. Audit it as a token-auth
            # failure with identifiers only — never the presented credential,
            # the principal's scopes, or the scope the route wants.
            unavailable = _audit_machine_failure(
                request,
                strict=strict_audit,
                status_code=403,
                principal=principal,
                reason="missing_required_scope",
                method=policy.method,
                route_template=policy.path,
            )
            if unavailable is not None:
                return unavailable
            return JSONResponse(
                {"error": "forbidden", "detail": "Forbidden"},
                status_code=403,
            )
        request.state.token_principal = principal
        request.state.token_authenticated = True
        request.state.token_route_template = policy.path
        request.state.token_audit_strict = strict_audit
        response = await call_next(request)
        if strict_audit and family is not None and not getattr(
            request.state, "token_route_audited", False
        ):
            # FastAPI can reject a typed path parameter (for example a
            # non-integer run id) before the registered endpoint function is
            # entered. Preserve the strict machine-audit contract for that
            # downstream denial instead of returning an unaudited 422. A
            # successful response without the endpoint's explicit audit mark
            # is an integration bug and therefore fails closed.
            reason = (
                "route_audit_missing"
                if response.status_code < 400
                else "downstream_rejected"
            )
            unavailable = _audit_machine_failure(
                request,
                strict=True,
                status_code=response.status_code,
                principal=principal,
                reason=reason,
                method=policy.method,
                route_template=policy.path,
            )
            if unavailable is not None or response.status_code < 400:
                return JSONResponse(
                    {"error": "unavailable", "detail": "Service Unavailable"},
                    status_code=503,
                )
        return response

    if unreachable:
        unavailable_response = _audit_machine_failure(
            request,
            strict=strict_audit,
            status_code=503,
            provider=unreachable,
            reason="provider_unreachable",
            method=policy.method,
            route_template=policy.path,
        )
        if unavailable_response is not None:
            return unavailable_response
        if strict_audit:
            return JSONResponse(
                {"error": "unavailable", "detail": "Service Unavailable"},
                status_code=503,
            )
        return JSONResponse(
            {"detail": f"Auth provider {unreachable!r} unreachable"},
            status_code=503,
        )

    unavailable = _audit_machine_failure(
        request,
        strict=strict_audit,
        status_code=401,
        reason="no_provider_recognises_token",
        method=policy.method,
        route_template=policy.path,
    )
    if unavailable is not None:
        return unavailable
    return JSONResponse(
        {"error": "unauthenticated", "detail": "Unauthorized"},
        status_code=401,
    )
