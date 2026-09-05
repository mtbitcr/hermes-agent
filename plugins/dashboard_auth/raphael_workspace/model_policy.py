"""Fail-closed model policy for Raphael's owner-facing Models surface.

The native Hermes profile config remains the only runtime model authority.
This module adds no registry, router, credential store, or fallback engine: it
only narrows the existing OAuth/model/profile endpoints for Raphael's dedicated
machine credential and validates the admitted profile assignments.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import re
import threading
from dataclasses import dataclass
from typing import Any, Mapping, Optional

from fastapi import HTTPException, Request

logger = logging.getLogger(__name__)

# fcntl is Unix-only; msvcrt is the Windows equivalent. One of the two is what
# makes the enrollment registry safe against a concurrent process, so absence
# of BOTH fails the write closed rather than degrading to an unlocked rewrite.
msvcrt = None
try:
    import fcntl
except ImportError:  # pragma: no cover - platform-specific fallback
    fcntl = None
    try:
        import msvcrt
    except ImportError:
        pass

from hermes_cli.dashboard_auth.audit import (
    AuditBatch,
    AuditEvent,
    AuditRollbackUncertain,
    AuditWriteError,
    audit_log_batch,
    begin_audit_batch,
    commit_audit_batch,
)
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
    model_label: str
    reasoning_effort: str
    recommended: bool = False


_PROFILE_IDS = (
    "default",
    "raphael-planner",
    "raphael-business",
    "raphael-designer",
    "raphael-claude-worker",
    "raphael-builder",
    "raphael-verifier",
)


_RECOMMENDED_PROVIDERS = {
    "default": "anthropic",
    "raphael-planner": "anthropic",
    "raphael-business": "anthropic",
    "raphael-designer": "anthropic",
    "raphael-claude-worker": "anthropic",
    "raphael-builder": "anthropic",
    "raphael-verifier": "openai-codex",
}


def _assignment(
    profile: str,
    provider: str,
    model: str,
    model_label: str,
    reasoning_effort: str,
) -> ModelAssignment:
    return ModelAssignment(
        profile=profile,
        provider=provider,
        model=model,
        model_label=model_label,
        reasoning_effort=reasoning_effort,
        recommended=_RECOMMENDED_PROVIDERS.get(profile) == provider,
    )


_ASSIGNMENTS = {
    ("raphael-planner", "anthropic"): _assignment(
        "raphael-planner", "anthropic", "claude-sonnet-5", "Claude Sonnet 5", "max"
    ),
    ("default", "anthropic"): _assignment(
        "default", "anthropic", "claude-opus-5", "Claude Opus 5", "max"
    ),
    ("raphael-planner", "openai-codex"): _assignment(
        "raphael-planner", "openai-codex", "gpt-5.6-sol", "GPT-5.6 Sol", "max"
    ),
    ("default", "openai-codex"): _assignment(
        "default", "openai-codex", "gpt-5.6-sol", "GPT-5.6 Sol", "max"
    ),
    ("raphael-business", "anthropic"): _assignment(
        "raphael-business", "anthropic", "claude-sonnet-5", "Claude Sonnet 5", "high"
    ),
    ("raphael-business", "openai-codex"): _assignment(
        "raphael-business", "openai-codex", "gpt-5.6-terra", "GPT-5.6 Terra", "max"
    ),
    ("raphael-designer", "anthropic"): _assignment(
        "raphael-designer", "anthropic", "claude-opus-5", "Claude Opus 5", "max"
    ),
    ("raphael-claude-worker", "anthropic"): _assignment(
        "raphael-claude-worker", "anthropic", "claude-sonnet-5", "Claude Sonnet 5 + Claude Code", "max"
    ),
    ("raphael-builder", "anthropic"): _assignment(
        "raphael-builder", "anthropic", "claude-sonnet-5", "Claude Sonnet 5", "max"
    ),
    # There is deliberately NO builder route on the OpenAI family. The builder
    # integrates verified work and operates infrastructure, so its deep lane is
    # high-risk coding, and high-risk coding may mint only Claude Opus 5 / max
    # (see _DEEP_ANTHROPIC_PROFILES). Admitting an OpenAI builder assignment
    # would make ``task_assignment_for(builder, openai-codex, 'deep')`` fall
    # through to that provider's base route, which is not a qualified deep
    # coding lane.
    ("raphael-verifier", "openai-codex"): _assignment(
        "raphael-verifier", "openai-codex", "gpt-5.6-sol", "GPT-5.6 Sol", "max"
    ),
    # Independent verification exists to be independent OF the implementation
    # family: Claude writes the code, so a Claude verifier is the same family
    # reviewing itself, and the OpenAI GPT-5.6 Sol / max lane stays the only
    # recommended verifier route. The Anthropic entry below is the named,
    # non-recommended "Claude Security" lane: security analysis runs on two
    # explicitly identified lanes (Codex Security on the recommended route,
    # Claude Security on this one), so a finding is never verified only by
    # the family that wrote the code. It is the strongest Claude model, so the
    # builder's Sonnet lane never reviews itself. Admitted, never recommended.
    ("raphael-verifier", "anthropic"): _assignment(
        "raphael-verifier", "anthropic", "claude-opus-5", "Claude Opus 5", "max"
    ),
}

_DEEP_ANTHROPIC_PROFILES = frozenset({
    "raphael-planner",
    "raphael-business",
    "raphael-claude-worker",
    "raphael-builder",
})

_EXECUTION_TIERS = frozenset({"routine", "deep"})


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


def normalize_execution_tier(value: Any) -> str:
    """Return one non-owner-facing task class, rejecting invented variants."""
    tier = str(value or "").strip().lower()
    if tier not in _EXECUTION_TIERS:
        raise ValueError("unadmitted Raphael execution tier")
    return tier


def task_assignment_for(
    profile: str, provider: str, execution_tier: str
) -> ModelAssignment:
    """Resolve one immutable task route from role, selected provider, and risk.

    The planner may classify work as ``routine`` or ``deep`` but never chooses a
    provider, model id, or effort.  Deep Claude planning, business, coding, and
    delivery work moves to the currently qualified Opus lane.  Other provider
    choices retain their admitted profile assignment until that provider has a
    separately qualified deep lane.
    """
    tier = normalize_execution_tier(execution_tier)
    base = assignment_for(profile, provider)
    if (
        tier == "deep"
        and provider == "anthropic"
        and profile in _DEEP_ANTHROPIC_PROFILES
    ):
        return _assignment(
            profile,
            provider,
            "claude-opus-5",
            "Claude Opus 5 + Claude Code"
            if profile == "raphael-claude-worker"
            else "Claude Opus 5",
            "max",
        )
    return base


def validate_runtime_assignment(
    profile: str,
    provider: str,
    model: str,
    reasoning_effort: str,
    *,
    disable_fallbacks: bool,
) -> ModelAssignment:
    """Accept a configured base route or an admitted task-specific route."""
    if disable_fallbacks is not True:
        raise ValueError("unadmitted Raphael model assignment")
    candidates = {
        task_assignment_for(profile, provider, "routine"),
        task_assignment_for(profile, provider, "deep"),
    }
    for candidate in candidates:
        if (
            str(model or "").strip() == candidate.model
            and str(reasoning_effort or "").strip().lower()
            == candidate.reasoning_effort
        ):
            return candidate
    raise ValueError("unadmitted Raphael model assignment")


def configured_assignment_for(profile: str) -> ModelAssignment:
    """Read and validate the profile's current native, fallback-free route."""
    from hermes_cli.config import load_config_readonly
    from hermes_cli.profiles import get_profile_dir
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    token = set_hermes_home_override(str(get_profile_dir(profile)))
    try:
        config = load_config_readonly()
    finally:
        reset_hermes_home_override(token)
    model_config = config.get("model")
    agent_config = config.get("agent")
    if not isinstance(model_config, Mapping) or not isinstance(agent_config, Mapping):
        raise ValueError("unadmitted Raphael model assignment")
    fallback_providers = config.get("fallback_providers")
    fallback_model = config.get("fallback_model")
    fallbacks_disabled = (
        isinstance(fallback_providers, list)
        and not fallback_providers
        and (
            fallback_model is None
            or (isinstance(fallback_model, str) and not fallback_model.strip())
        )
    )
    return validate_assignment(
        profile,
        str(model_config.get("provider") or ""),
        str(model_config.get("default") or model_config.get("model") or ""),
        str(agent_config.get("reasoning_effort") or ""),
        disable_fallbacks=fallbacks_disabled,
    )


def resolve_task_assignment(profile: str, execution_tier: str) -> ModelAssignment:
    """Resolve a new task against the provider selected for its role."""
    configured = configured_assignment_for(profile)
    return task_assignment_for(profile, configured.provider, execution_tier)


# ---------------------------------------------------------------------------
# Durable task-route lock provenance
# ---------------------------------------------------------------------------
#
# A locked task stores its authority as ``<authority>:v<version>:<digest>``.
# The digest binds the *whole* authority — assignee, provider, model, effort
# and execution tier — so no single field of a locked row can be edited (by
# hand, by a migration, or by a future code path) without the lock ceasing to
# validate.  Validation additionally re-derives the route from this module's
# matrix, so a lock minted under a policy that no longer admits that route is
# stale and therefore invalid.  There is exactly one authority name and one
# version: anything else is unknown provenance and fails closed.

POLICY_LOCK_AUTHORITY = "raphael"
POLICY_LOCK_VERSION = 1

# Never-admissible values, enforced independently of the matrix so a route
# can never be admitted by a future matrix edit alone.
_FORBIDDEN_MODEL_MARKERS = ("fable", "ultracode")
_FORBIDDEN_EFFORTS = frozenset({"ultra"})

_LOCK_RE = re.compile(r"\A([a-z][a-z0-9-]{0,31}):v(\d{1,4}):([0-9a-f]{64})\Z")


def _lock_digest(
    assignee: str, provider: str, model: str, reasoning_effort: str, execution_tier: str
) -> str:
    canonical = json.dumps(
        {
            "authority": POLICY_LOCK_AUTHORITY,
            "version": POLICY_LOCK_VERSION,
            "assignee": assignee,
            "provider": provider,
            "model": model,
            "reasoning_effort": reasoning_effort,
            "execution_tier": execution_tier,
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def mint_policy_lock(
    assignee: str,
    provider: str,
    model: str,
    reasoning_effort: str,
    execution_tier: str,
) -> str:
    """Return the durable lock for one exact owner-approved task route.

    Raises ``ValueError`` unless the five components are together an admitted
    route right now, so an unmintable route can never become a stored lock.
    """
    parts = _normalized_lock_parts(
        assignee, provider, model, reasoning_effort, execution_tier
    )
    error = _route_authority_error(*parts)
    if error:
        raise ValueError(error)
    return f"{POLICY_LOCK_AUTHORITY}:v{POLICY_LOCK_VERSION}:{_lock_digest(*parts)}"


def _normalized_lock_parts(
    assignee: Any, provider: Any, model: Any, reasoning_effort: Any, execution_tier: Any
) -> tuple[str, str, str, str, str]:
    return (
        str(assignee or "").strip(),
        str(provider or "").strip(),
        str(model or "").strip(),
        str(reasoning_effort or "").strip().lower(),
        str(execution_tier or "").strip().lower(),
    )


def _route_authority_error(
    assignee: str, provider: str, model: str, reasoning_effort: str, execution_tier: str
) -> Optional[str]:
    """Return why this five-tuple is not an admitted locked route, else None."""
    if not assignee or not provider or not model or not reasoning_effort:
        return (
            "policy-locked route is incomplete "
            f"(assignee={assignee or None!r}, provider={provider or None!r}, "
            f"model={model or None!r}, reasoning_effort={reasoning_effort or None!r})"
        )
    lowered = model.lower()
    for marker in _FORBIDDEN_MODEL_MARKERS:
        if marker in lowered:
            return f"policy-locked route names a forbidden model {model!r}"
    if reasoning_effort in _FORBIDDEN_EFFORTS:
        return (
            "policy-locked route names a forbidden reasoning effort "
            f"{reasoning_effort!r}"
        )
    try:
        expected = task_assignment_for(assignee, provider, execution_tier)
    except ValueError:
        return (
            "policy-locked route names no admitted authority for "
            f"{assignee!r}/{provider!r}/{execution_tier or None!r}"
        )
    if model != expected.model or reasoning_effort != expected.reasoning_effort:
        return (
            f"policy-locked route {model!r}/{reasoning_effort!r} is not the "
            f"admitted route for {assignee!r}/{provider!r}/{execution_tier!r}"
        )
    return None


def policy_lock_error(
    lock: Any,
    assignee: Any,
    provider: Any,
    model: Any,
    reasoning_effort: Any,
    execution_tier: Any,
) -> Optional[str]:
    """Return why ``lock`` does not authorize this exact route, else None.

    Fails closed on every ambiguity: an unparseable or foreign authority, a
    version this build does not mint, a digest that does not bind these exact
    five components, an incomplete route, and a route the current policy no
    longer admits are all invalid.  Never treats an invalid lock as absent —
    callers ask about a lock only after deciding one is present.
    """
    text = str(lock or "").strip()
    if not text:
        return "policy lock is missing"
    match = _LOCK_RE.match(text)
    if match is None:
        return "policy lock provenance is unreadable"
    authority, version, digest = match.group(1), int(match.group(2)), match.group(3)
    if authority != POLICY_LOCK_AUTHORITY:
        return f"policy lock names an unknown authority {authority!r}"
    if version != POLICY_LOCK_VERSION:
        return (
            f"policy lock is stale (authority version {version}, "
            f"current {POLICY_LOCK_VERSION})"
        )
    parts = _normalized_lock_parts(
        assignee, provider, model, reasoning_effort, execution_tier
    )
    error = _route_authority_error(*parts)
    if error:
        return error
    if digest != _lock_digest(*parts):
        return "policy lock digest does not bind this route"
    return None


def admitted_provider_ids() -> tuple[str, ...]:
    return ("anthropic", "openai-codex")


def admitted_profile_ids() -> tuple[str, ...]:
    """Return the exact native profiles visible in Advanced settings."""
    return _PROFILE_IDS


# ---------------------------------------------------------------------------
# Explicit Raphael policy enrollment
# ---------------------------------------------------------------------------
#
# Whether a profile is governed by this policy is a PERSISTED Hermes-owned
# fact, never an inference from the route the profile currently happens to
# run. Inferring it means an enrolled role that has drifted (or been
# hand-edited) silently stops being governed exactly when governance matters
# most, and an ordinary profile that coincidentally shares a name and model
# starts being governed when it should stay native and unrestricted.
#
# The record is written only by the trusted Models-machine mutation, only for
# admitted profile ids, and lives under the hermes root (NOT under any single
# profile home) so it resolves identically while a request is scoped to a
# named profile.

ENROLLMENT_SCHEMA_VERSION = 1
ENROLLMENT_AUTHORITY = "raphael-models-machine"


class EnrollmentUnavailable(RuntimeError):
    """The enrollment record could not be read or written; fail closed."""


def enrollment_path():
    """Where the Hermes-owned Raphael enrollment record lives."""
    from hermes_constants import get_default_hermes_root

    return get_default_hermes_root() / "raphael" / "model_policy_enrollment.json"


# Re-entrancy depth for the registry lock, per THREAD (never inherited by a
# new one). ``flock`` is per-descriptor: a second acquire from the same process
# on a second descriptor blocks forever, so a transaction that wants to hold
# the lock across enrollment AND the steps that must succeed for it to stand
# has to join the outer hold rather than take it again.
_enrollment_lock_depth = threading.local()


@contextlib.contextmanager
def _enrollment_registry_lock():
    """Hold the one cross-process lock that orders enrollment mutations.

    Enrollment is a read-modify-replace of a single shared JSON document, so
    two processes enrolling DIFFERENT profiles concurrently would each write
    back the document they read and the later writer would drop the other's
    role — silently un-governing a role that a Models-machine write believed it
    had enrolled. A kernel-held ``flock`` on a stable sibling lock file (the
    same shape ``kanban_db`` uses for its ``<db>.init.lock``) serializes the
    whole read/merge/replace, so no retry loop is needed and no temporary file
    is orphaned. Fails closed: with no lock there is no enrollment write.

    Re-entrant on the same thread (see :func:`enrollment_transaction`), so an
    outer holder can call :func:`enroll_profiles` / :func:`restore_enrollment`
    without deadlocking against its own descriptor.
    """
    from pathlib import Path

    path = Path(enrollment_path())
    if getattr(_enrollment_lock_depth, "value", 0):
        _enrollment_lock_depth.value += 1
        try:
            yield path
        finally:
            _enrollment_lock_depth.value -= 1
        return
    lock_path = path.with_name(path.name + ".lock")
    handle = None
    try:
        if fcntl is None and msvcrt is None:  # pragma: no cover - exotic platform
            raise RuntimeError("no OS file-locking primitive is available")
        path.parent.mkdir(parents=True, exist_ok=True)
        handle = lock_path.open("a+b")
        if fcntl is not None:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        else:  # pragma: no cover - Windows-only path
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
    except (OSError, RuntimeError) as exc:
        if handle is not None:
            handle.close()
        raise EnrollmentUnavailable(
            "the Raphael policy enrollment record could not be locked"
        ) from exc
    _enrollment_lock_depth.value = 1
    try:
        yield path
    finally:
        _enrollment_lock_depth.value = 0
        try:
            if fcntl is not None:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            else:  # pragma: no cover - Windows-only path
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        except OSError:
            pass
        handle.close()


@contextlib.contextmanager
def enrollment_transaction(profiles):
    """Enroll roles and hold the registry through everything that must follow.

    Enrollment and the strict audit record are one operation: neither is
    complete without the other. Holding the registry lock across BOTH makes
    them one registry transaction, which is what lets the rollback be a plain
    restore of the exact prior text.

    Without the outer hold, the rollback is unsafe under concurrency: this
    batch releases the lock after enrolling, a second batch enrolls a different
    role and commits, and then this batch's failure restores a whole-document
    snapshot that predates — and therefore erases — the other batch's
    successful enrollment. Serializing the two batches means the loser's
    snapshot already contains the winner's role, so restoring it preserves it.

    Ordering: always taken UNDER the caller's profile route locks (see
    ``hermes_cli.config.apply_profile_route_batch``), never the other way
    round, so the two lock families cannot deadlock.
    """
    with _enrollment_registry_lock():
        snapshot = enroll_profiles(profiles)
        try:
            yield snapshot
        except BaseException:
            restore_enrollment(snapshot)
            raise


def _load_enrollment(path) -> tuple:
    """Return ``(exact prior file text or None, validated document)``.

    The raw text is kept so a failed post-write step can restore the registry
    byte-for-byte — including "the file did not exist" — rather than a
    re-serialized equivalent.
    """
    try:
        if not path.is_file():
            return None, {"version": ENROLLMENT_SCHEMA_VERSION, "profiles": {}}
        text = path.read_text(encoding="utf-8")
        raw = json.loads(text)
    except (OSError, ValueError) as exc:
        raise EnrollmentUnavailable(
            "the Raphael policy enrollment record could not be read"
        ) from exc
    if (
        not isinstance(raw, dict)
        or raw.get("version") != ENROLLMENT_SCHEMA_VERSION
        or not isinstance(raw.get("profiles"), dict)
    ):
        raise EnrollmentUnavailable(
            "the Raphael policy enrollment record is not a shape this build wrote"
        )
    return text, raw


def _read_enrollment() -> dict:
    """Return the raw enrollment document, or raise when it is unusable."""
    from pathlib import Path

    return _load_enrollment(Path(enrollment_path()))[1]


def enrolled_profile_ids() -> frozenset[str]:
    """Return the profiles explicitly enrolled into Raphael policy.

    Raises :class:`EnrollmentUnavailable` when the record exists but cannot be
    read or is not this build's shape: a route-affecting write must refuse
    rather than assume "nothing is governed". Only admitted ids are returned,
    so a record naming a profile this build does not admit cannot widen
    governance either.
    """
    admitted = set(admitted_profile_ids())
    entries = _read_enrollment()["profiles"]
    return frozenset(
        name
        for name, entry in entries.items()
        if name in admitted
        and isinstance(entry, dict)
        and entry.get("authority") == ENROLLMENT_AUTHORITY
    )


def is_profile_enrolled(profile: str) -> bool:
    """Whether this exact profile is under Raphael policy right now."""
    return str(profile or "").strip() in enrolled_profile_ids()


def enroll_profiles(profiles) -> Optional[str]:
    """Enroll every named profile under one held registry lock.

    Returns the registry's EXACT prior file text (``None`` when there was no
    file), so a caller whose later step fails can restore the registry to the
    state it had before this call — see :func:`restore_enrollment`.

    Called by the trusted Models-machine mutation only, and only for admitted
    profile ids — an ordinary profile can never be pulled under the policy by
    any other surface. The document is re-read INSIDE the lock and merged, so a
    concurrent enrollment of a different profile is preserved rather than
    overwritten. Idempotent: enrolling an already-enrolled profile writes
    nothing.
    """
    from utils import atomic_json_write

    admitted = admitted_profile_ids()
    canonical = []
    for profile in profiles:
        name = str(profile or "").strip()
        if name not in admitted:
            raise ValueError("unadmitted Raphael profile id")
        canonical.append(name)

    with _enrollment_registry_lock() as path:
        prior_text, document = _load_enrollment(path)
        entries = document["profiles"]
        changed = False
        for name in canonical:
            entry = entries.get(name)
            if (
                isinstance(entry, dict)
                and entry.get("authority") == ENROLLMENT_AUTHORITY
            ):
                continue
            entries[name] = {"authority": ENROLLMENT_AUTHORITY}
            changed = True
        if changed:
            try:
                atomic_json_write(path, document)
            except OSError as exc:
                raise EnrollmentUnavailable(
                    "the Raphael policy enrollment record could not be written"
                ) from exc
        return prior_text


def enroll_profile(profile: str) -> None:
    """Record that ``profile`` is governed by Raphael policy from now on."""
    enroll_profiles([profile])


def restore_enrollment(snapshot: Optional[str]) -> None:
    """Put the registry back to the exact text :func:`enroll_profiles` saw.

    ``None`` means the record did not exist and is removed again. Restoring
    goes through the same lock as the mutation, and through the same atomic
    replace, so permissions, ownership and symlinks survive.
    """
    from utils import atomic_write_text

    with _enrollment_registry_lock() as path:
        try:
            if snapshot is None:
                path.unlink(missing_ok=True)
            else:
                atomic_write_text(path, snapshot, preserve_mode=True)
        except OSError as exc:
            raise EnrollmentUnavailable(
                "the Raphael policy enrollment record could not be restored"
            ) from exc


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


def _admitted_models_for(profile: Optional[str], provider: str) -> set:
    """Model ids ``profile`` may run on ``provider``, across every task class.

    With no profile named (a generic availability read) this stays the whole
    provider-wide admitted set, exactly as before.
    """
    key = str(profile or "").strip()
    if not key:
        return {
            assignment.model
            for assignment in _ASSIGNMENTS.values()
            if assignment.provider == provider
        }
    try:
        base = assignment_for(key, provider)
    except ValueError:
        return set()
    models = {base.model}
    for tier in _EXECUTION_TIERS:
        models.add(task_assignment_for(key, provider, tier).model)
    return models


def _assignment_payload(assignment: ModelAssignment) -> dict:
    return {
        "profile": assignment.profile,
        "provider": assignment.provider,
        "model": assignment.model,
        "model_label": assignment.model_label,
        "reasoning_effort": assignment.reasoning_effort,
        "recommended": assignment.recommended,
    }


def project_options_payload(
    payload: Mapping[str, Any],
    *,
    profile: Optional[str] = None,
    revision: Optional[str] = None,
) -> dict:
    """Project native availability plus Hermes' canonical role policy.

    When a ``profile`` is named the exposed model list is narrowed to the ids
    THAT role may actually run on each provider (its base route plus the route
    of every execution tier). A role with no admitted assignment for a provider
    therefore offers no models at all, so a lane this policy does not admit —
    an OpenAI builder, a Claude verifier — cannot be surfaced as a selectable
    option by a reader that only looks at the model list.
    """
    admitted = {
        provider: _admitted_models_for(profile, provider)
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
            base_assignment = _ASSIGNMENTS.get((str(profile or ""), provider))
            task_routes = None
            if base_assignment is not None:
                task_routes = {
                    tier: _assignment_payload(
                        task_assignment_for(base_assignment.profile, provider, tier)
                    )
                    for tier in sorted(_EXECUTION_TIERS)
                }
            rows.append({
                "slug": provider,
                "authenticated": raw.get("authenticated") is True,
                "models": safe_models,
                "assignment": (
                    _assignment_payload(base_assignment)
                    if base_assignment is not None
                    else None
                ),
                "task_routes": task_routes,
            })
    return {"profile": profile, "revision": revision, "providers": rows}


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
    ("PUT", "/api/profiles/raphael-business/model"),
    ("PUT", "/api/profiles/raphael-designer/model"),
    ("PUT", "/api/profiles/raphael-claude-worker/model"),
    ("PUT", "/api/profiles/raphael-builder/model"),
    ("PUT", "/api/profiles/raphael-verifier/model"),
    # One all-or-nothing multi-role selection. Workspace calls this instead of
    # emulating a transaction across several single-profile writes.
    ("POST", "/api/profiles/model-batch"),
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
    """Durably audit a successful Models-machine request, or fail closed.

    For a read, which has nothing to roll back: one line, one ``fsync``, 503 on
    an unwritable log. Route and enrollment changes never come through here —
    they go through :func:`journal_models_machine_batch_success`, whose record
    is only exposed once the change itself is durably committed.
    """
    entries = _models_machine_entries(request, action=action, roles=[(profile, provider)])
    if entries is None:
        return
    try:
        audit_log_batch(entries, strict=True)
        request.state.token_route_audited = True
    except AuditWriteError as exc:
        raise HTTPException(status_code=503, detail="Service Unavailable") from exc


def _models_machine_entries(
    request: Request,
    *,
    action: str,
    roles: "list[tuple[Optional[str], Optional[str]]]",
) -> "Optional[list[tuple[AuditEvent, dict]]]":
    """One record per role, or ``None`` when this is not a Models-machine call."""
    if not model_machine_request(request):
        return None
    principal = request.state.token_principal
    route_template = getattr(
        request.state, "token_route_template", request.url.path
    )
    ip = transport_peer_ip(request)
    return [
        (
            AuditEvent.TOKEN_AUTH_SUCCESS,
            {
                "provider": principal.provider,
                "principal": principal.principal,
                "credential_id": principal.credential_id,
                "grant": token_store.MODELS_GRANT,
                "source": "raphael-models",
                "method": request.method,
                "route_template": route_template,
                "action": action,
                "profile": profile,
                "model_provider": provider,
                "decision": "allow",
                "status": 200,
                "ip": ip,
            },
        )
        for profile, provider in roles
    ]


class _JournalledBatchAudit:
    """A journalled Models-machine batch awaiting its commit."""

    def __init__(self, batch: "Optional[AuditBatch]"):
        self._batch = batch

    def commit(self) -> None:
        if self._batch is None:
            return
        try:
            commit_audit_batch(self._batch)
        except AuditRollbackUncertain as exc:
            # A different fact from "nothing was recorded": the commit line
            # reached the log and could not be proven gone, so the log may
            # already be asserting this batch. Failing the request here made the
            # caller roll the whole operation back, which is the one outcome
            # this two-phase record exists to prevent — a durable
            # ``decision=allow`` for routes that were reverted.
            #
            # So the batch is treated as COMMITTED and rolled forward. Every
            # write it owed (the configs, the enrollment) had already landed
            # before this call; the only thing in doubt is whether the audit
            # line survives a power loss, and a log that may be missing a
            # record for a change that really happened is strictly weaker than
            # a log asserting one that did not. Reported once, loudly, for an
            # operator.
            logger.error(
                "Models batch audit commit could not be confirmed durable; the "
                "route change stands and the dashboard-auth log may be missing "
                "its record: %s",
                exc,
            )
            return
        except AuditWriteError as exc:
            raise HTTPException(
                status_code=503, detail="Service Unavailable"
            ) from exc


def journal_models_machine_batch_success(
    request: Request,
    *,
    action: str,
    roles: "list[tuple[Optional[str], Optional[str]]]",
) -> _JournalledBatchAudit:
    """Journal a whole Models-machine route operation, or fail closed.

    One record per role, and ONE durable batch for the operation, exposed only
    once the caller commits it. Auditing each role as it completed — or even
    auditing them together while the operation could still be reverted — would
    leave a rolled-back batch's roles recorded as ``decision="allow"``: a
    durable claim of success for a change that never took effect. A short
    write, a failed ``fsync`` after the bytes landed, a rollback, a restart, or
    a concurrent batch all leave this one uncommitted, and an uncommitted batch
    is not a record of anything.

    The caller commits as the LAST durable act of the operation, so what the
    log ends up asserting is what actually happened.
    """
    entries = _models_machine_entries(request, action=action, roles=roles)
    if entries is None:
        return _JournalledBatchAudit(None)
    try:
        batch = begin_audit_batch(entries)
    except AuditWriteError as exc:
        raise HTTPException(status_code=503, detail="Service Unavailable") from exc
    request.state.token_route_audited = True
    return _JournalledBatchAudit(batch)
