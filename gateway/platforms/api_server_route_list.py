"""Classify a role's ``gateway.api_server.allowed_routes`` from its parsed config.yaml.

A release must classify a saved configuration exactly as the gateway will, so
the reading is one pure function of the parsed YAML root. The gateway's
per-request resolver, ``_resolve_api_server_allowed_routes()`` in api_server.py,
reads the profile's config.yaml and calls it.
"""

from __future__ import annotations

from typing import Any

from gateway.platforms.api_server import (
    _CONFIG_ROOT_ABSENT,
    _ROUTES_ALLOWLIST,
    _ROUTES_DENY_ALL,
    _ROUTES_UNRESTRICTED,
    _normalize_api_route_rule,
)


def _classify_allowed_routes(raw: Any) -> "tuple[str, list[str]]":
    """Classify ``gateway.api_server.allowed_routes`` of one parsed config root.

    ``raw`` is the config.yaml YAML root exactly as parsed, or
    :data:`_CONFIG_ROOT_ABSENT` when there is no config at all (no file, or a
    file with no YAML content). Returns ``(mode, patterns)``:
      - ``(_ROUTES_UNRESTRICTED, [])`` — no config at all (missing/blank
        file), key omitted/``None``, OR the intentionally-supported empty
        list/tuple. No gating.
      - ``(_ROUTES_DENY_ALL, [])`` — an explicit malformed falsey value
        (``False``, ``0``, ``{}``, ``""``), any other malformed type/entry,
        or a non-mapping YAML root (list/scalar/``null``). Only ``/health``
        is reachable.
      - ``(_ROUTES_ALLOWLIST, patterns)`` — a non-empty string (single
        pattern) or list/tuple of non-empty strings, each normalized by
        ``_normalize_api_route_rule``.

    Pure: reads no file, environment or global mutable state, never mutates
    ``raw``, and returns a new list on every call.
    """
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
