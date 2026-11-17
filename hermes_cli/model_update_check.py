"""``hermes model update-check``: offer a newer model version, never apply it.

Card 2 of the model update flow. For every model a role uses now, the check
looks for a higher version of the same family in the live account lists and
runs three checks on it, cheapest first:

1. Account: every sign-in a role of that model uses, read in that role's own
   profile, lists the version.
2. Compatibility, local and free: the capability row supports each role's
   effort, and the context window is not smaller than the current one's.
3. Probe: one short coding question per distinct sign-in, with fallbacks
   disabled.

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
import threading
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
# The narrow format the question asks for: the number only, then at most one
# full stop or exclamation mark; surrounding whitespace is ignored.
_ANSWER_RE = re.compile(rf"\s*{re.escape(PROBE_ANSWER)}[.!]?\s*")
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
    """What the catalog proves about one model; a field is None when it is unknown.

    ``efforts`` is the set of efforts the provider's route sends unchanged for
    this model, so a role's exact effort is checked, not reasoning in general.
    """

    reasoning: Optional[bool]
    context_window: Optional[int]
    efforts: Optional[frozenset] = None


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
    from agent import anthropic_credentials

    # The resolver refreshes an expired shared login, which writes it; the
    # check never asks then, so that sign-in stays unproven.
    login = anthropic_credentials.read_claude_code_credentials()
    if login and login.get("refreshToken") and not anthropic_credentials.is_claude_code_token_valid(login):
        return ""
    return (anthropic_credentials.resolve_anthropic_token() or "").strip()


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


class _SignIn:
    """One sign-in a role uses: its name, and its value in memory only.

    Two sign-ins are the same account when their values are equal, which is
    compared here and nowhere else. Showing, copying or pickling one gives its
    name only, so a value never reaches a receipt, a log or a suggestion.
    """

    __slots__ = ("name", "value")

    def __init__(self, name: str, value: str):
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "value", value)

    def __setattr__(self, _name, _value):
        raise AttributeError("a sign-in is read-only")

    def __eq__(self, other):
        return isinstance(other, _SignIn) and other.value == self.value

    def __hash__(self):
        return hash((_SignIn, self.value))

    def __repr__(self):
        return f"_SignIn({self.name!r})"

    def __str__(self):
        return self.name

    def __reduce__(self):
        return (str, (self.name,))


def _value_of(provider: str, sign_in) -> str:
    """The value of a proven sign-in, or of a named one in the current scope."""
    return sign_in.value if isinstance(sign_in, _SignIn) else _sign_in_value(provider, sign_in)


def _name_of(sign_in) -> str:
    return getattr(sign_in, "name", sign_in)


def _in_scope(home, secrets: dict, reader: Callable[[], str]) -> str:
    """``reader`` run in one profile's read-only scope: its home and its secrets."""
    from agent.secret_scope import reset_secret_scope, set_secret_scope
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    home_token = set_hermes_home_override(home)
    secret_token = set_secret_scope(secrets)
    try:
        return (reader() or "").strip()
    finally:
        reset_secret_scope(secret_token)
        reset_hermes_home_override(home_token)


def _role_value(role: str, reader: Callable[[], str]) -> str:
    """What ``reader`` resolves in ``role``'s own scope, or "" when that is not proven.

    A secret the role's own sources lack falls through to this process's
    environment, which holds the planning profile's values. So for any other
    role the answer must stay the same when every secret only the planning
    profile defines is blanked; one that changes was borrowed, never the
    role's own.
    """
    from agent.secret_scope import build_profile_secret_scope
    from hermes_cli.profiles import get_profile_dir
    from hermes_constants import get_hermes_home

    planner_home, home = get_hermes_home(), get_profile_dir(role)
    secrets = build_profile_secret_scope(home)
    value = _in_scope(home, secrets, reader)
    if home.resolve() == planner_home.resolve():
        return value
    masked = {name: "" for name in build_profile_secret_scope(planner_home) if name not in secrets}
    if not value or value != _in_scope(home, {**masked, **secrets}, reader):
        return ""
    return value


def _role_sign_ins(role: str, provider: str) -> tuple:
    """The sign-ins ``role`` itself uses for ``provider``: ``(sign-ins, "")`` or ``(None, why)``.

    Each is resolved in the role's own read-only scope and held as a
    ``_SignIn``. The coding worker's builds sign in through the host resolver
    of the build machine, which prefers the build sign-in and can fall back to
    other credentials; which one it picks is proven by comparing its value
    with the worker's own sign-ins in memory.
    """
    from plugins.dashboard_auth.raphael_workspace import sandbox_dispatch

    if provider == "openai-codex":
        value = _role_value(role, _codex_token)
        if not value:
            return None, "The role's own sign-in for this provider could not be proven."
        return (_SignIn("codex", value),), ""
    if provider != "anthropic":
        return None, "The check has no account list for this provider."
    gateway = _role_value(role, _gateway_token)
    if not gateway:
        return None, "The role's own sign-in for this provider could not be proven."
    if role != sandbox_dispatch.WORKER_PROFILE:
        return (_SignIn("gateway", gateway),), ""
    used = sandbox_dispatch._resolve_host_credential()
    if not used:
        return None, "The worker's builds have no sign-in now, so the check cannot tell which account they use."
    for name, value in (("build", _role_value(role, _build_token)), ("gateway", gateway)):
        if used == value:
            return tuple(dict.fromkeys((_SignIn("gateway", gateway), _SignIn(name, value)))), ""
    return None, (
        "The worker's builds sign in with a credential that is neither the build "
        "sign-in nor the gateway sign-in, so the check cannot list its models."
    )


_QUIET = threading.Lock()
_quiet = {"depth": 0, "prior": logging.NOTSET}


@contextmanager
def _no_log_records():
    """No log record is created anywhere while a shared helper holds a sign-in value.

    The shared account-list helpers log the raw text of a remote error
    (models.py:1798, codex_models.py:167), and that text can echo the sign-in
    the request carried. Records are switched off for the length of the call,
    so no handler, formatter or record factory ever sees it; the check then
    logs the error's type only.
    """
    with _QUIET:
        if _quiet["depth"] == 0:
            _quiet["prior"] = logging.root.manager.disable
            logging.disable(logging.CRITICAL)
        _quiet["depth"] += 1
    try:
        yield
    finally:
        with _QUIET:
            _quiet["depth"] -= 1
            if _quiet["depth"] == 0:
                logging.disable(_quiet["prior"])


def _account_model_ids(provider: str, sign_in) -> tuple:
    with _no_log_records():
        listed = _SIGN_INS[(provider, _name_of(sign_in))][1](_value_of(provider, sign_in))
    if listed is None:
        raise LookupError("the account list could not be read")
    return tuple(listed)


def _route_efforts(provider: str, model: str) -> Optional[frozenset]:
    """The efforts the provider's adapter sends unchanged for ``model``; None when unknown.

    The anthropic adapter turns an effort it cannot send into another one (xhigh
    into max, or a fixed budget), so only the efforts it keeps count.
    """
    if provider == "anthropic":
        from agent import anthropic_adapter

        if not anthropic_adapter._supports_adaptive_thinking(model):
            return frozenset(anthropic_adapter.THINKING_BUDGET)
        efforts = {"low", "medium", "high", "max"}
        if anthropic_adapter._supports_xhigh_effort(model):
            efforts.add("xhigh")
        return frozenset(efforts)
    if provider == "openai-codex":
        from agent.reasoning_effort import codex_supported_efforts

        return frozenset(codex_supported_efforts(model))
    return None


def _capability_row(provider: str, model: str) -> Capability:
    """The capability row inventory.py:272 shows, without its defaults.

    A model the catalog does not know is unknown, never capable; the context
    window comes only from a catalog or override value, because the row's own
    200000 is a default when the catalog has none.
    """
    from agent.models_dev import get_model_capabilities, lookup_models_dev_context

    meta = get_model_capabilities(provider, model)
    if meta is None:
        return Capability(reasoning=None, context_window=None, efforts=None)
    return Capability(
        reasoning=bool(meta.supports_reasoning),
        context_window=lookup_models_dev_context(provider, model),
        efforts=_route_efforts(provider, model) if meta.supports_reasoning else frozenset(),
    )


def _probe_call(provider: str, model: str, sign_in, effort: str, question: str) -> ProbeReply:
    """One short call on this sign-in and model; the caller disables fallbacks.

    The client is built for exactly this provider, model and sign-in value,
    not through ``call_llm``: with fallbacks disabled that pins a call to the
    planning profile's main route, which refuses the candidate or swaps its
    key. The request is sent once, with no retry, refresh, rotation or
    fallback, and anything else is refused before it is sent.
    """
    from agent import auxiliary_client as aux

    value = _value_of(provider, sign_in)
    if provider == "anthropic":
        client, final = aux._get_cached_client(provider, model, api_key=value)
    elif provider == "openai-codex":
        client = aux.CodexAuxiliaryClient(
            aux._create_openai_client(
                api_key=value,
                base_url=aux._CODEX_AUX_BASE_URL,
                default_headers=aux._codex_cloudflare_headers(value),
            ),
            model,
        )
        final = model
    else:
        raise LookupError("the check has no probe for this provider")
    if client is None or final != model:
        raise LookupError("no client for exactly this model and sign-in")
    kwargs = aux._build_call_kwargs(
        provider,
        model,
        [{"role": "user", "content": question}],
        max_tokens=_PROBE_MAX_TOKENS,
        timeout=_PROBE_TIMEOUT_SECONDS,
        reasoning_config=None if effort in _NO_EFFORT else {"enabled": True, "effort": effort},
        base_url=str(getattr(client, "base_url", "") or ""),
    )
    if kwargs.get("model") != model:
        raise LookupError("the request would not go to exactly this model")
    response = client.chat.completions.create(**kwargs)
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


def _provenance_ref(card: _Card) -> str:
    return f"kanban-task:{card.task_id}"


def _past_suggestions(receipt: dict, card: _Card) -> dict:
    """``(provider, from, to) -> [(month, decision)]`` from every board.

    A suggestion this planning card filed, in any run, is also noted in the
    receipt's budget: the card files at most one.
    """
    past: dict = {}
    for meta in kb.list_boards(include_archived=True):
        slug = meta.get("slug")
        store = _board_store(slug) if slug else None
        if store is None or not store.exists():
            continue
        with kb.connect_closing(db_path=store) as conn:
            rows = conn.execute(
                "SELECT id, recommendation_subject_id, recommendation_decision, provenance_ref "
                "FROM tasks WHERE task_kind = 'recommendation' AND recommendation_subject_id LIKE ? "
                "ORDER BY rowid",
                (SUBJECT_PREFIX + "%",),
            ).fetchall()
        receipt["boards_read"].append(slug)
        for task_id, subject, decision, ref in rows:
            if ref == _provenance_ref(card):
                receipt["budget"]["filed_before"].append(
                    {"board": slug, "task_id": task_id, "subject_id": subject}
                )
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
        # Every effort the role runs this model at: its routine and deep lanes.
        efforts = entry.efforts.setdefault(route.role, [])
        if route.effort not in efforts:
            efforts.append(route.effort)
    for entry in found.values():
        entry.roles.sort(key=roles.index)
    return sorted(found.values(), key=lambda entry: entry.rank)


_SWITCH = threading.Lock()
_switch: dict = {"depth": 0, "prior": None}


@contextmanager
def _fallbacks_disabled():
    """Disable fallbacks for one probe and leave the worker's env as it was.

    ``disable_fallbacks_for_process`` is a one-way latch, so the probe uses the
    environment switch every model call honours and restores it afterwards.
    """
    name = fallback_config.FALLBACKS_DISABLED_ENV
    with _SWITCH:
        if _switch["depth"] == 0:
            _switch["prior"] = os.environ.get(name)
            os.environ[name] = "1"
        _switch["depth"] += 1
    try:
        yield
    finally:
        with _SWITCH:
            _switch["depth"] -= 1
            if _switch["depth"] == 0:
                if _switch["prior"] is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = _switch["prior"]


def _answered(text: Optional[str]) -> bool:
    """The whole reply is the number the probe asks for, as it asks for it.

    "45", "45." or " 45 " pass. A reply that only contains the number, such as
    a contradiction, a negation or an explanation, fails, and so does a near
    match such as "-45" or "45.0".
    """
    return _ANSWER_RE.fullmatch(text or "") is not None


def _probe(seams: Seams, provider: str, model: str, sign_in, effort: str, label: str) -> dict:
    started = time.monotonic()
    result = {"sign_in": label, "passed": False, "input_tokens": 0, "output_tokens": 0}
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


def _compatibility(model: _Model, new: Capability, old: Capability) -> dict:
    """The local check: True, False or None (unknown) per property, and why it stops.

    Every exact effort an affected role runs this model at must be one the
    candidate's route accepts, and the candidate's context window must be
    known and not smaller than the current model's known one.
    """
    needed = sorted({effort for efforts in model.efforts.values() for effort in efforts
                     if effort not in _NO_EFFORT})
    found = {"effort": None, "context": None, "efforts": needed, "result": "compatible", "reason": ""}
    if new.reasoning is None:
        found.update(result="capability_unknown",
                     reason="The catalog has no capability row for the new version.")
        return found
    if needed and new.efforts is None:
        found.update(result="effort_unknown",
                     reason="The catalog does not show which efforts the new version accepts.")
        return found
    missing = [effort for effort in needed if effort not in (new.efforts or ())]
    found["effort"] = not missing
    if missing:
        found.update(result="effort_unsupported",
                     reason="The new version does not accept the effort a role uses: " + ", ".join(missing) + ".")
        return found
    if new.context_window is None or old.context_window is None:
        found.update(result="context_unknown",
                     reason="The catalog does not show the context window of the new or the current version.")
        return found
    found["context"] = new.context_window >= old.context_window
    if not found["context"]:
        found.update(result="context_smaller",
                     reason="The new version has a smaller context window than the current one.")
    return found


def _check(seams: Seams, model: _Model, to_model: str, accounts: list, listed: dict,
           blocked: Optional[str]) -> dict:
    """Run the three checks on one candidate, stopping at the first that fails."""
    provider = model.provider
    checked = {
        "provider": provider,
        "from_model": model.model,
        "to_model": to_model,
        "roles": list(model.roles),
        "result": None,
        "account": {label: to_model in (listed[label] or ()) for label, _, _ in accounts},
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
    compatibility = _compatibility(model, seams.capabilities(provider, to_model),
                                   seams.capabilities(provider, model.model))
    checked["compatibility"] = compatibility
    if compatibility["result"] != "compatible":
        # Unknown is not a pass: every property must be proven before the probe.
        checked["result"] = compatibility.pop("result")
        return checked
    compatibility.pop("result")
    for label, sign_in, roles in accounts:
        probed = _probe(seams, provider, to_model, sign_in, model.efforts[roles[0]][0], label)
        checked["probe"].append(probed)
        if not probed["passed"]:
            checked["result"] = "probe_failed"
            return checked
    checked["result"] = "passed"
    return checked


def _list_ids(seams: Seams, provider: str, sign_in, label: str) -> Optional[frozenset]:
    try:
        return frozenset(seams.account_models(provider, sign_in))
    except Exception as exc:
        logger.warning(
            "model update check: the %s account list could not be read (%s)",
            label, type(exc).__name__,
        )
        return None


def _accounts(model: _Model) -> list:
    """The distinct accounts the model's roles use, in role order: ``(label, sign-in, roles)``.

    Roles share an account only when their sign-ins are equal in memory. The
    label is the sign-in's name, with its first role added when distinct
    accounts of this model share that name.
    """
    groups: dict = {}
    for role in model.roles:
        for sign_in in model.sign_ins[role]:
            roles = groups.setdefault(sign_in, [])
            if role not in roles:
                roles.append(role)
    names = [_name_of(sign_in) for sign_in in groups]
    return [
        (name if names.count(name) == 1 else f"{name}:{roles[0]}", sign_in, roles)
        for name, (sign_in, roles) in zip(names, groups.items())
    ]


def _file(card: _Card, checked: dict, month: str, practice: bool) -> Optional[dict]:
    """File the one suggestion, or None when this planning card already filed one.

    The budget check and the new row share one write transaction on the
    project's board, so of two runs of the card at the same time only one
    files; the other sees the first one's row.
    """
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
    with kb.connect_closing(db_path=_board_store(card.project_board)) as conn, kb.write_txn(conn):
        spent = conn.execute(
            "SELECT 1 FROM tasks WHERE task_kind = 'recommendation' AND provenance_ref = ? "
            "AND recommendation_subject_id LIKE ? LIMIT 1",
            (_provenance_ref(card), SUBJECT_PREFIX + "%"),
        ).fetchone()
        if spent is not None:
            return None
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
            provenance_ref=_provenance_ref(card),
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


def _candidates(model: _Model, listed: dict, practice: bool) -> list:
    if practice:
        return [model.model]
    seen = set().union(*(ids or () for ids in listed.values()))
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
    past = _past_suggestions(receipt, card)
    if receipt["budget"]["filed_before"]:
        # Checked before any pair is chosen: the card's one suggestion is spent.
        receipt["reason"] = "This planning card already filed its one suggestion."
        return
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
        # Every role is proven: each distinct account is listed and probed once.
        accounts = _accounts(model)
        entry["sign_ins"] = [{"sign_in": label, "roles": roles} for label, _, roles in accounts]
        for label, sign_in, _ in accounts:
            if (model.provider, sign_in) not in lists:
                lists[(model.provider, sign_in)] = _list_ids(seams, model.provider, sign_in, label)
        listed = {label: lists[(model.provider, sign_in)] for label, sign_in, _ in accounts}
        entry["unreadable_lists"] = [label for label, ids in listed.items() if ids is None]
        candidates = _candidates(model, listed, practice)
        if candidates:
            entry["result"] = "candidates"
        else:
            entry["result"] = "list_unreadable" if entry["unreadable_lists"] else "no_newer_version"
        for to_model in candidates:
            blocked = _blocked_by_past(past.get((model.provider, model.model, to_model), []),
                                       month, practice)
            if blocked is None and not practice and (model.provider, to_model) in approved:
                blocked = "already_approved"
            checked = _check(seams, model, to_model, accounts, listed, blocked)
            receipt["candidates"].append(checked)
            if checked["result"] == "passed":
                filed = _file(card, checked, month, practice)
                if filed is None:
                    checked["result"] = "budget_spent"
                    receipt["reason"] = "Another run of this planning card filed its one suggestion first."
                    return
                receipt["filed"] = filed
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
        # The planning card files at most one suggestion over all its runs.
        "budget": {"planning_card": card.task_id, "filed_before": []},
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
