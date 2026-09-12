"""Helpers for reading the effective fallback provider chain from config."""

from __future__ import annotations

import os
from typing import Any, Sequence

# Kill switch for the whole fallback chain, checked ahead of config so a caller
# that must not switch routes stays pinned even if config.yaml drifts later.
# Set by the kanban dispatcher for model-policy-locked tasks.
#
# The env var is the OUTER, inheritable channel — useful for anything the
# worker itself spawns — but it is NOT the authority, because a profile's
# ``.env`` is loaded with ``override=True`` and would happily reset it to 0
# during startup. The authority is ``NO_FALLBACK_FLAG`` on the worker's own
# argv, latched into this module's process state AFTER all dotenv loading has
# finished (see ``hermes_cli.main``). Nothing a user can put in config.yaml or
# .env can clear that latch.
FALLBACKS_DISABLED_ENV = "HERMES_DISABLE_FALLBACKS"
NO_FALLBACK_FLAG = "--no-fallbacks"

_ENV_TRUE = frozenset({"1", "true", "yes", "on"})

# Process-wide, set-once. Deliberately module state rather than an env var:
# ``load_dotenv(..., override=True)`` rewrites ``os.environ`` but cannot reach
# a Python global.
_fallbacks_disabled_for_process = False


def disable_fallbacks_for_process() -> None:
    """Latch this process into no-fallback mode. Never reversible."""
    global _fallbacks_disabled_for_process
    _fallbacks_disabled_for_process = True


def apply_process_fallback_policy(argv: Sequence[str] | None = None) -> bool:
    """Latch the no-fallback authority from this process's own argv.

    Called by the CLI entry point immediately AFTER dotenv loading, so a
    profile ``.env`` that sets ``HERMES_DISABLE_FALLBACKS=0`` cannot undo it.
    Also re-exports the env var so anything this process spawns inherits the
    same policy. Returns whether the latch is now set.
    """
    import sys

    args = list(sys.argv[1:] if argv is None else argv)
    if NO_FALLBACK_FLAG in args:
        disable_fallbacks_for_process()
        os.environ[FALLBACKS_DISABLED_ENV] = "1"
    return fallbacks_disabled()


def strip_no_fallback_flag(argv: list[str]) -> list[str]:
    """Remove the control flag so argparse never sees it."""
    return [arg for arg in argv if arg != NO_FALLBACK_FLAG]


def fallbacks_disabled() -> bool:
    """True when this process must not substitute any other provider/model."""
    if _fallbacks_disabled_for_process:
        return True
    return os.environ.get(FALLBACKS_DISABLED_ENV, "").strip().lower() in _ENV_TRUE


# Back-compat alias for the env-only question. Prefer ``fallbacks_disabled``:
# it also honours the non-overridable CLI latch.
fallbacks_disabled_by_env = fallbacks_disabled


def _normalized_base_url(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    return value.strip().rstrip("/")


def resolve_entry_api_key(entry: dict[str, Any] | None) -> str | None:
    """API key for one fallback entry: inline ``api_key``, else ``key_env``.

    Mirrors the custom-provider convention (``key_env`` names the env var
    holding the key; ``api_key_env`` accepted as an alias). Returns None when
    neither yields a non-empty value, letting ``resolve_runtime_provider``
    fall through to the provider's standard credential resolution.

    ``key_env`` is resolved through ``agent.secret_scope.get_secret`` rather
    than a raw ``os.getenv`` — in a multiplexed gateway a bare env read would
    ignore the active profile's scope and can return another profile's
    credential. ``get_secret`` already implements the right fallback: it
    reads ``os.environ`` when there's no active multiplexed scope (matching
    prior single-profile behavior), and fails closed only when multiplexing
    is active with no scope installed.
    """
    if not isinstance(entry, dict):
        return None
    inline = str(entry.get("api_key") or "").strip()
    if inline:
        return inline
    key_env = str(entry.get("key_env") or entry.get("api_key_env") or "").strip()
    if key_env:
        from agent.secret_scope import get_secret

        return (get_secret(key_env) or "").strip() or None
    return None


def _iter_fallback_entries(raw: Any) -> list[dict[str, Any]]:
    if isinstance(raw, dict):
        candidates = [raw]
    elif isinstance(raw, list):
        candidates = raw
    else:
        return []

    entries: list[dict[str, Any]] = []
    for entry in candidates:
        if not isinstance(entry, dict):
            continue
        provider = str(entry.get("provider") or "").strip()
        model = str(entry.get("model") or "").strip()
        if not provider or not model:
            continue

        normalized = dict(entry)
        normalized["provider"] = provider
        normalized["model"] = model

        base_url = _normalized_base_url(entry.get("base_url"))
        if base_url:
            normalized["base_url"] = base_url

        entries.append(normalized)
    return entries


def _entry_identity(entry: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(entry.get("provider") or "").strip().lower(),
        str(entry.get("model") or "").strip().lower(),
        _normalized_base_url(entry.get("base_url")).lower(),
    )


def get_fallback_chain(config: dict[str, Any] | None) -> list[dict[str, Any]]:
    """Return the effective fallback chain merged across old and new config keys.

    ``fallback_providers`` remains the primary source of truth and keeps its
    order. Legacy ``fallback_model`` entries are appended afterwards unless
    they target the same provider/model/base_url route as an earlier entry.
    The returned list always contains fresh dict copies.

    The no-fallback latch wins over both keys and yields an empty chain: a run
    pinned to one exact route must never switch, whatever the config happens to
    say by the time it starts.
    """

    if fallbacks_disabled():
        return []

    config = config or {}
    chain: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()

    for key in ("fallback_providers", "fallback_model"):
        for entry in _iter_fallback_entries(config.get(key)):
            identity = _entry_identity(entry)
            if identity in seen:
                continue
            seen.add(identity)
            chain.append(entry)

    return chain
