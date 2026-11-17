"""``hermes model update-check``: offer a newer model version, never apply it.

Card 2 of the model update flow. For every model a role uses now, the check
looks for a higher version of the same family in the live account lists and
runs three checks on it, cheapest first:

1. Account: every sign-in a role of that model uses lists the version.
2. Compatibility, local and free: the capability row supports each role's
   effort, and the context window is not smaller than the current one's.
3. Probe: one short coding question per sign-in, with fallbacks disabled.

The first ranked candidate that passes all three is filed as one suggestion
through Decisions (a ``provider_model_policy`` recommendation on the board of
the planning card's project) and the run writes one receipt, the task event
``model_update_checked``, on that card. One suggestion covers one new version
of one model for every role that uses it.

Proposal-only on purpose: the check never calls the route batch, never writes
configuration and never writes the approved update record. Only the Update
button on the Decisions screen does that, after the owner chose it. A practice
run (``--practice <role>``) files the role's current model as its own
candidate so the owner can try the buttons before a real release; Update then
changes nothing.

It runs inside the monthly planning card's worker, whose environment pins one
board. Past suggestions can sit on any board, so every board and the root
project registry are opened by their own paths, never through the pin.

No sign-in value is written anywhere: account lists and probes hold it only in
memory, and an error is recorded by its type, never its text.
"""

from __future__ import annotations

import logging
import os
import re
import sqlite3
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Optional

from hermes_cli import fallback_config
from hermes_cli import kanban_db as kb
from plugins.dashboard_auth.raphael_workspace import model_policy

logger = logging.getLogger(__name__)

RECEIPT_KIND = "model_update_checked"
SUBJECT_PREFIX = "model-update:"
RECOMMENDATION_KIND = "provider_model_policy"

PROBE_QUESTION = (
    "What does the Python expression sum(range(10)) evaluate to? "
    "Reply with the number only."
)
PROBE_ANSWER = "45"
# Room for a reasoning model to think briefly and still answer: the plan's
# estimate for one probe is about 2-8k tokens.
_PROBE_MAX_TOKENS = 8000
_PROBE_TIMEOUT_SECONDS = 120.0

OWNER_TITLE = "A newer version of an AI model is ready for part of your team."
_CHECKED = (
    "Raphael checked that your account can use it, that it works with Raphael, "
    "and that it did one sample piece of work correctly."
)
OWNER_REASON = (
    f"{_CHECKED} Update uses it for new work only. Work already started or "
    "waiting keeps its current setting. You can undo it in Settings."
)
PRACTICE_SENTENCE = "This is a practice run. Clicking Update changes nothing."
_PRACTICE_TITLE = "A practice model update is ready for part of your team."
_PRACTICE_REASON = f"{PRACTICE_SENTENCE} {_CHECKED}"

# Owner-visible through the Decisions feed, so no model or provider is named;
# the exact names live in the subject id and the receipt only.
_EVIDENCE = {
    "need": "A newer version of an AI model that part of your team uses is available.",
    "expected_benefit": "New work can use the newer version.",
    "risks": "The newer version may behave a little differently.",
    "cost": "Deciding costs nothing. The check used one short sample piece of work.",
    "rollback": "You can undo it in Settings.",
}
_PRACTICE_EVIDENCE = {
    "need": "You asked for a practice run of a model update.",
    "expected_benefit": "You can try the buttons before a real update arrives.",
    "risks": "None. Clicking Update changes nothing.",
    "cost": "Deciding costs nothing. The check used one short sample piece of work.",
    "rollback": "There is nothing to undo.",
}

_TIER_RANK = {"deep": 0, "routine": 1}
_NO_EFFORT = frozenset({"", "none"})
_VERSION_PART_RE = re.compile(r"[0-9]+(?:\.[0-9]+)*")


class UpdateCheckRefused(RuntimeError):
    """The check cannot run here: it needs the planning card's worker."""


@dataclass(frozen=True)
class Route:
    """One route a role runs now: its routine route or its deep lane."""

    role: str
    provider: str
    model: str
    effort: str
    tier: str


@dataclass(frozen=True)
class Capability:
    """The capability row of one model; ``context_window`` None when unknown."""

    reasoning: bool
    context_window: Optional[int]


@dataclass(frozen=True)
class ProbeReply:
    text: str
    input_tokens: int
    output_tokens: int


@dataclass(frozen=True)
class Seams:
    """Everything the check reads from outside; tests pass fakes."""

    routes: Callable[[str], tuple]
    sign_ins: Callable[[str, str], tuple]
    account_models: Callable[[str, str], tuple]
    capabilities: Callable[[str, str], Capability]
    probe: Callable[[str, str, str, str, str], ProbeReply]
    route_revision: Callable[[str], str]
    now: Callable[[], datetime]


@dataclass(frozen=True)
class _Card:
    task_id: str
    run_id: Optional[int]
    profile: str
    project_id: str
    # None when the card has no open project; the run then files nothing.
    project_board: Optional[str]


@dataclass
class _Model:
    """One model in use, with every role that uses it."""

    provider: str
    model: str
    rank: tuple
    roles: list
    efforts: dict
    sign_ins: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Production readers (the defaults when no seams are passed)
# ---------------------------------------------------------------------------


def _configured_routes(role: str) -> tuple:
    """The role's configured route and, when it differs, its deep lane.

    Read through the model policy, so an owner-approved update record is
    applied exactly as every other route read applies it.
    """
    routine = model_policy.configured_assignment_for(role)
    routes = [Route(role, routine.provider, routine.model, routine.reasoning_effort, "routine")]
    deep = model_policy.task_assignment_for(role, routine.provider, "deep")
    if (deep.model, deep.reasoning_effort) != (routine.model, routine.reasoning_effort):
        routes.append(Route(role, deep.provider, deep.model, deep.reasoning_effort, "deep"))
    return tuple(routes)


def _gateway_token() -> str:
    from agent.anthropic_credentials import resolve_anthropic_token

    return (resolve_anthropic_token() or "").strip()


def _build_token() -> str:
    from plugins.dashboard_auth.raphael_workspace.build_sign_in_file import read_build_token

    found = read_build_token()
    return found[0] if found is not None else ""


def _codex_token() -> str:
    """The account the openai-codex runtime selects (web_server.py:7087-7107)."""
    from agent.credential_pool import preview_runtime_selection

    entry, _borrowed = preview_runtime_selection("openai-codex")
    if entry is None:
        return ""
    return str(entry.runtime_api_key or entry.access_token or "").strip()


def _anthropic_ids(token: str):
    from hermes_cli.models import _fetch_anthropic_models

    return _fetch_anthropic_models(api_key=token)


def _codex_ids(token: str):
    """The live account list only.

    ``get_codex_model_ids`` falls back to a local cache and built-in defaults
    when the account cannot be read, which would pass the account check for a
    version the account does not offer; an empty live answer is unreadable.
    """
    from hermes_cli.codex_models import _fetch_models_from_api

    return _fetch_models_from_api(token) or None


# (provider, sign-in) -> (reader of its value, reader of its live model ids).
# The value stays in memory: it is only ever handed to the reader or the probe.
_SIGN_INS = {
    ("anthropic", "gateway"): (_gateway_token, _anthropic_ids),
    ("anthropic", "build"): (_build_token, _anthropic_ids),
    ("openai-codex", "codex"): (_codex_token, _codex_ids),
}


def _held_value(provider: str, sign_in: str) -> str:
    reader = _SIGN_INS.get((provider, sign_in))
    return reader[0]() if reader is not None else ""


def _sign_in_value(provider: str, sign_in: str) -> str:
    value = _held_value(provider, sign_in)
    if not value:
        raise LookupError("the sign-in holds no value")
    return value


def _role_sign_ins(role: str, provider: str) -> tuple:
    """Which sign-ins ``role`` uses for ``provider``: ``(names, "")`` or ``(None, why)``.

    The coding worker's builds sign in through the host resolver of the build
    machine, which prefers the build sign-in and can fall back to other
    credentials; which one it picks is proven by comparing values in memory.
    Every other role runs on the gateway's sign-in of its provider.
    """
    from plugins.dashboard_auth.raphael_workspace import sandbox_dispatch

    if provider == "openai-codex":
        return ("codex",), ""
    if provider != "anthropic":
        return None, "The check has no account list for this provider."
    if role != sandbox_dispatch.WORKER_PROFILE:
        return ("gateway",), ""
    used = sandbox_dispatch._resolve_host_credential()
    if not used:
        return None, "The worker's builds have no sign-in now, so the check cannot tell which account they use."
    for name in ("build", "gateway"):
        if used == _held_value(provider, name):
            return tuple(dict.fromkeys(("gateway", name))), ""
    return None, (
        "The worker's builds sign in with a credential that is neither the build "
        "sign-in nor the gateway sign-in, so the check cannot list its models."
    )


def _account_model_ids(provider: str, sign_in: str) -> tuple:
    listed = _SIGN_INS[(provider, sign_in)][1](_sign_in_value(provider, sign_in))
    if listed is None:
        raise LookupError("the account list could not be read")
    return tuple(listed)


def _capability_row(provider: str, model: str) -> Capability:
    """The capability row inventory.py:272 shows: reasoning defaults to True."""
    from agent.models_dev import get_model_capabilities

    meta = get_model_capabilities(provider, model)
    if meta is None:
        return Capability(reasoning=True, context_window=None)
    return Capability(reasoning=bool(meta.supports_reasoning), context_window=int(meta.context_window))


def _probe_call(provider: str, model: str, sign_in: str, effort: str, question: str) -> ProbeReply:
    """One short call on this sign-in and model; the caller disables fallbacks.

    An anthropic call uses exactly the key passed. An openai-codex call uses
    the account the codex runtime selects, the same one ``_codex_token`` reads.
    """
    from agent.auxiliary_client import call_llm

    response = call_llm(
        provider=provider,
        model=model,
        api_key=_sign_in_value(provider, sign_in),
        messages=[{"role": "user", "content": question}],
        max_tokens=_PROBE_MAX_TOKENS,
        reasoning_config=None if effort in _NO_EFFORT else {"enabled": True, "effort": effort},
        timeout=_PROBE_TIMEOUT_SECONDS,
    )
    usage = getattr(response, "usage", None)
    return ProbeReply(
        text=str(response.choices[0].message.content or ""),
        input_tokens=int(getattr(usage, "prompt_tokens", 0) or 0),
        output_tokens=int(getattr(usage, "completion_tokens", 0) or 0),
    )


def _route_revision(role: str) -> str:
    from hermes_cli.config import profile_route_revision
    from hermes_cli.profiles import get_profile_dir

    return profile_route_revision(get_profile_dir(role))


def default_seams() -> Seams:
    return Seams(
        routes=_configured_routes,
        sign_ins=_role_sign_ins,
        account_models=_account_model_ids,
        capabilities=_capability_row,
        probe=_probe_call,
        route_revision=_route_revision,
        now=lambda: datetime.now(timezone.utc),
    )


# ---------------------------------------------------------------------------
# Boards, the card and past suggestions
# ---------------------------------------------------------------------------


def _board_store(slug: str):
    """A board's own store. ``kanban_db_path`` would answer the worker's pin."""
    if slug == kb.DEFAULT_BOARD:
        return kb.kanban_home() / "kanban.db"
    return kb.board_dir(slug) / "kanban.db"


def _project_board(project_id: str) -> Optional[str]:
    """The board of ``project_id`` in the root registry, read-only.

    The registry sits beside the shared board at the root, never in the
    worker's profile home (see tools/kanban_tools.py).
    """
    from hermes_cli import projects_db

    path = kb.kanban_home() / "projects.db"
    if not path.exists():
        return None
    conn = sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True)
    try:
        conn.row_factory = sqlite3.Row
        project = projects_db.get_project(conn, project_id)
    finally:
        conn.close()
    if project is None or project.archived:
        return None
    return project.board_slug or project.slug


def _planning_card() -> _Card:
    """The card this worker was spawned for, read from its own pinned board."""
    from hermes_cli.profiles import get_active_profile_name, normalize_profile_name

    task_id = (os.environ.get("HERMES_KANBAN_TASK") or "").strip()
    if not task_id:
        raise UpdateCheckRefused(
            "the update check runs only inside the worker of a planning card"
        )
    with kb.connect_closing() as conn:
        task = kb.get_task(conn, task_id)
    if task is None:
        raise UpdateCheckRefused("the planning card was not found on its board")
    profile = normalize_profile_name(get_active_profile_name())
    if normalize_profile_name(task.assignee or "") != profile:
        raise UpdateCheckRefused("the planning card belongs to another profile")
    project_id = (task.project_id or "").strip()
    run_id = (os.environ.get("HERMES_KANBAN_RUN_ID") or "").strip()
    return _Card(
        task_id=task_id,
        run_id=int(run_id) if run_id.isdigit() else None,
        profile=profile,
        project_id=project_id,
        project_board=_project_board(project_id) if project_id else None,
    )


def _past_suggestions(receipt: dict) -> dict:
    """``(provider, from, to) -> [(month, decision)]`` from every board."""
    past: dict = {}
    for meta in kb.list_boards(include_archived=True):
        slug = meta.get("slug")
        store = _board_store(slug) if slug else None
        if store is None or not store.exists():
            continue
        with kb.connect_closing(db_path=store) as conn:
            rows = conn.execute(
                "SELECT recommendation_subject_id, recommendation_decision FROM tasks "
                "WHERE task_kind = 'recommendation' AND recommendation_subject_id LIKE ?",
                (SUBJECT_PREFIX + "%",),
            ).fetchall()
        receipt["boards_read"].append(slug)
        for subject, decision in rows:
            parts = str(subject).split(":")
            if len(parts) == 5:
                past.setdefault(tuple(parts[1:4]), []).append((parts[4], decision))
    return past


def _blocked_by_past(decisions: list, month: str, practice: bool) -> Optional[str]:
    """Why an earlier suggestion of the same pair stops filing it now, or None.

    A rejected or accepted pair is never asked again; a deferred pair is asked
    again only in a new month, which is why the month is part of the subject
    id. A practice pair is asked again whenever no practice of it is waiting
    or was filed this month.
    """
    seen = {decision for _month, decision in decisions}
    if not practice:
        for decision in ("rejected", "accepted"):
            if decision in seen:
                return f"{decision}_before"
    if "pending" in seen:
        return "waiting_for_owner"
    for past_month, decision in decisions:
        if past_month == month:
            return f"{decision}_this_month"
    return None


# ---------------------------------------------------------------------------
# Candidates and the three checks
# ---------------------------------------------------------------------------


def _version(model: str) -> tuple:
    """The version digits of ``model`` as numbers, in order."""
    return tuple(
        int(number)
        for part in model.split("-")
        if _VERSION_PART_RE.fullmatch(part)
        for number in part.split(".")
    )


def _is_newer(candidate: str, model: str) -> bool:
    """Same family and a higher version, by the update record's own family rule.

    The approved update record accepts only a pair whose ids differ in their
    version parts (``model_policy._model_family``), so the check uses that rule
    and no suggestion is filed that Update could not record. A same-family id
    is never a forbidden one, because the current model is admitted. A dated
    snapshot of the current id is the same version, not a newer one.
    """
    return (
        candidate != model
        and not candidate.startswith(model + "-")
        and model_policy._model_family(candidate) == model_policy._model_family(model)
        and _version(candidate) > _version(model)
    )


def _models_in_use(routes: list, roles: tuple) -> list:
    """One entry per model, ranked by role order and then deep before routine."""
    found: dict = {}
    for route in routes:
        rank = (roles.index(route.role), _TIER_RANK.get(route.tier, len(_TIER_RANK)))
        entry = found.setdefault(
            (route.provider, route.model),
            _Model(route.provider, route.model, rank, [], {}),
        )
        entry.rank = min(entry.rank, rank)
        if route.role not in entry.roles:
            entry.roles.append(route.role)
        entry.efforts.setdefault(route.role, route.effort)
    for entry in found.values():
        entry.roles.sort(key=roles.index)
    return sorted(found.values(), key=lambda entry: entry.rank)


@contextmanager
def _fallbacks_disabled():
    """Disable fallbacks for one probe and leave the worker's env as it was.

    ``disable_fallbacks_for_process`` is a one-way latch, so the probe uses the
    environment switch every model call honours and restores it afterwards.
    """
    name = fallback_config.FALLBACKS_DISABLED_ENV
    prior = os.environ.get(name)
    os.environ[name] = "1"
    try:
        yield
    finally:
        if prior is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = prior


def _answered(text: str) -> bool:
    """The known answer stands alone: "45" or "45." pass, "145" or "45.5" do not."""
    return re.search(rf"(?<![\w.]){re.escape(PROBE_ANSWER)}(?!\w|\.\d)", text or "") is not None


def _probe(seams: Seams, provider: str, model: str, sign_in: str, effort: str) -> dict:
    started = time.monotonic()
    result = {"sign_in": sign_in, "passed": False, "input_tokens": 0, "output_tokens": 0}
    try:
        with _fallbacks_disabled():
            reply = seams.probe(provider, model, sign_in, effort, PROBE_QUESTION)
    except Exception as exc:
        # The type only: an error text can carry what a sign-in returned.
        logger.warning("model update check: a probe failed (%s)", type(exc).__name__)
        result["error"] = type(exc).__name__
    else:
        result.update(
            passed=_answered(reply.text),
            input_tokens=int(reply.input_tokens),
            output_tokens=int(reply.output_tokens),
        )
    result["seconds"] = round(time.monotonic() - started, 3)
    return result


def _check(seams: Seams, model: _Model, to_model: str, sign_ins: list, listed: dict,
           blocked: Optional[str]) -> dict:
    """Run the three checks on one candidate, stopping at the first that fails."""
    provider = model.provider
    checked = {
        "provider": provider,
        "from_model": model.model,
        "to_model": to_model,
        "roles": list(model.roles),
        "result": None,
        "account": {name: to_model in (listed[name] or ()) for name in sign_ins},
        "compatibility": {"effort": None, "context": None},
        "probe": [],
        "route_revisions": {role: seams.route_revision(role) for role in model.roles},
    }
    if blocked is not None:
        checked["result"] = blocked
        return checked
    if not all(checked["account"].values()):
        checked["result"] = "not_listed"
        return checked
    new, old = seams.capabilities(provider, to_model), seams.capabilities(provider, model.model)
    effort_ok = all(effort in _NO_EFFORT or new.reasoning for effort in model.efforts.values())
    # A catalog silent on a new version does not block it; the probe still runs.
    context_ok = (
        new.context_window is None
        or old.context_window is None
        or new.context_window >= old.context_window
    )
    checked["compatibility"] = {"effort": effort_ok, "context": context_ok}
    if not effort_ok:
        checked["result"] = "effort_unsupported"
        return checked
    if not context_ok:
        checked["result"] = "context_smaller"
        return checked
    for name in sign_ins:
        role = next(role for role in model.roles if name in model.sign_ins[role])
        probed = _probe(seams, provider, to_model, name, model.efforts[role])
        checked["probe"].append(probed)
        if not probed["passed"]:
            checked["result"] = "probe_failed"
            return checked
    checked["result"] = "passed"
    return checked


def _list_ids(seams: Seams, provider: str, sign_in: str) -> Optional[frozenset]:
    try:
        return frozenset(seams.account_models(provider, sign_in))
    except Exception as exc:
        logger.warning(
            "model update check: the %s account list could not be read (%s)",
            sign_in, type(exc).__name__,
        )
        return None


def _file(card: _Card, checked: dict, month: str, practice: bool) -> dict:
    subject = SUBJECT_PREFIX + ":".join(
        (checked["provider"], checked["from_model"], checked["to_model"], month)
    )
    evidence = dict(_PRACTICE_EVIDENCE if practice else _EVIDENCE)
    # Accepting a real update changes the route new work runs on; a practice
    # one changes nothing.
    evidence["schema_version"] = 1
    evidence["requested_scope"] = {
        flag: flag == "production_effect" and not practice
        for flag in kb.RECOMMENDATION_SCOPE_FLAGS
    }
    with kb.connect_closing(db_path=_board_store(card.project_board)) as conn:
        task_id = kb.create_recommendation(
            conn,
            project_id=card.project_id,
            target_profile=checked["roles"][0],
            recommendation_kind=RECOMMENDATION_KIND,
            recommendation_subject_id=subject,
            recommendation_label=_PRACTICE_TITLE if practice else OWNER_TITLE,
            recommendation_rationale=_PRACTICE_REASON if practice else OWNER_REASON,
            recommendation_evidence=evidence,
            provenance_authority=f"hermes-profile:{card.profile}",
            provenance_ref=f"kanban-task:{card.task_id}",
            provenance_observed_at=int(time.time()),
        )
    return {
        "board": card.project_board,
        "task_id": task_id,
        "subject_id": subject,
        "target_profile": checked["roles"][0],
    }


def _write_receipt(card: _Card, receipt: dict) -> None:
    """One ``model_update_checked`` event on the card, under its own run."""
    with kb.connect_closing() as conn:
        with kb.write_txn(conn):
            kb._append_event(conn, card.task_id, RECEIPT_KIND, receipt, run_id=card.run_id)


# ---------------------------------------------------------------------------
# The run
# ---------------------------------------------------------------------------


def _candidates(model: _Model, sign_ins: list, listed: dict, practice: bool) -> list:
    if practice:
        return [model.model]
    seen = set().union(*(listed[name] or () for name in sign_ins))
    newer = [candidate for candidate in seen if _is_newer(candidate, model.model)]
    return sorted(newer, key=lambda candidate: (_version(candidate), candidate), reverse=True)


def _run(seams: Seams, card: _Card, practice_role: Optional[str], receipt: dict) -> None:
    month, practice = receipt["month"], practice_role is not None
    roles = model_policy.admitted_profile_ids()
    if card.project_board is None:
        receipt["reason"] = "The planning card has no open project to file a suggestion in."
        return
    if practice and practice_role not in roles:
        receipt["reason"] = "The practice role is not one of the team's roles."
        return
    routes = []
    for role in (practice_role,) if practice else roles:
        try:
            routes.extend(seams.routes(role))
        except (ValueError, OSError, LookupError, model_policy.EnrollmentUnavailable) as exc:
            receipt["unread_roles"].append({"role": role, "error": type(exc).__name__})
    if practice:
        if not routes:
            receipt["reason"] = "The practice role has no route the policy admits."
            return
        routes = [min(routes, key=lambda route: _TIER_RANK.get(route.tier, len(_TIER_RANK)))]
    try:
        approved = {
            (update["provider"], update["to_model"])
            for update in model_policy.read_model_updates()
            if update["state"] == "active"
        }
    except model_policy.EnrollmentUnavailable:
        receipt["reason"] = "The approved update record could not be read."
        return
    past = _past_suggestions(receipt)
    lists: dict = {}
    for model in _models_in_use(routes, roles):
        entry = {"provider": model.provider, "model": model.model, "roles": list(model.roles)}
        receipt["models"].append(entry)
        for role in model.roles:
            names, why = seams.sign_ins(role, model.provider)
            if names is None:
                entry.update(result="sign_in_unproven", reason=why, role=role)
                break
            model.sign_ins[role] = tuple(names)
        if entry.get("result"):
            continue
        sign_ins = list(dict.fromkeys(name for role in model.roles for name in model.sign_ins[role]))
        entry["sign_ins"] = sign_ins
        for name in sign_ins:
            if (model.provider, name) not in lists:
                lists[(model.provider, name)] = _list_ids(seams, model.provider, name)
        listed = {name: lists[(model.provider, name)] for name in sign_ins}
        entry["unreadable_lists"] = [name for name in sign_ins if listed[name] is None]
        candidates = _candidates(model, sign_ins, listed, practice)
        if candidates:
            entry["result"] = "candidates"
        else:
            entry["result"] = "list_unreadable" if entry["unreadable_lists"] else "no_newer_version"
        for to_model in candidates:
            blocked = _blocked_by_past(past.get((model.provider, model.model, to_model), []),
                                       month, practice)
            if blocked is None and not practice and (model.provider, to_model) in approved:
                blocked = "already_approved"
            checked = _check(seams, model, to_model, sign_ins, listed, blocked)
            receipt["candidates"].append(checked)
            if checked["result"] == "passed":
                receipt["filed"] = _file(card, checked, month, practice)
                receipt["result"] = "filed"
                receipt["reason"] = "One suggestion was filed."
                return
    if receipt["candidates"]:
        receipt["reason"] = "No newer version passed all three checks."
    elif any(entry.get("result") == "list_unreadable" for entry in receipt["models"]):
        receipt["reason"] = "An account list could not be read, and no newer version was found."
    else:
        receipt["reason"] = "No newer version was found."


def run_update_check(*, practice_role: Optional[str] = None,
                     seams: Optional[Seams] = None) -> dict[str, Any]:
    """Run one check, file at most one suggestion, and return its receipt.

    The receipt is also written on the planning card; it names exact models,
    which is why it is not owner-visible. ``practice_role`` files the role's
    current model as a practice suggestion; the probe still runs.
    """
    seams = seams or default_seams()
    card = _planning_card()
    receipt: dict[str, Any] = {
        "schema_version": 1,
        "month": seams.now().strftime("%Y-%m"),
        "practice": practice_role is not None,
        "practice_role": practice_role,
        "result": "nothing_filed",
        "reason": "",
        "boards_read": [],
        "unread_roles": [],
        "models": [],
        "candidates": [],
        "filed": None,
        "error": None,
    }
    try:
        _run(seams, card, practice_role, receipt)
    except Exception as exc:
        # The run still leaves its one receipt; the type only, as everywhere.
        receipt["error"] = type(exc).__name__
        receipt["reason"] = "The check stopped on an unexpected error."
        _write_receipt(card, receipt)
        raise
    _write_receipt(card, receipt)
    return receipt
