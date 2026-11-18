"""Update on a model update suggestion of the Decisions feed, applied once.

A model update suggestion is a native recommendation of kind
``provider_model_policy`` whose subject is
``model-update:<provider>:<current model>:<new model>:<month>``; a practice
suggestion names the same model twice. To a request that names
:data:`MODEL_UPDATE_CAPABILITY` the Decisions feed shows it as a
``model_update`` item. Later and Keep current are the existing defer and reject
of the suggestion; Update is :func:`apply_model_update`, behind its own route.

One Update moves every role that runs the current model, read at the click, in
one route batch with the revisions read at the click. The owner's decision is
taken first, with the lifecycle compare-and-swap, so a second click or another
tab finds nothing waiting. The board and the role configs are not one
transaction, so the receipt is its own event on the suggestion, appended right
after the batch.
"""

from __future__ import annotations

import contextlib
import functools
import logging
import time
from dataclasses import dataclass
from typing import Any, Optional

from hermes_cli import kanban_db
from hermes_cli.owner_workspace import (
    OwnerContext,
    OwnerWorkspaceError,
    _owner_pending_suggestion,
)

MODEL_UPDATE_CAPABILITY = "model_update_v1"

_log = logging.getLogger(__name__)

# The owner's fixed texts. None names a model or a provider.
TITLE = "A newer version of an AI model is ready for part of your team."
REASON = (
    "Raphael checked that your account can use it, that it works with Raphael, "
    "and that it did one sample piece of work correctly. Update uses it for new "
    "work only. Work already started or waiting keeps its current setting. You "
    "can undo it in Settings."
)
PRACTICE = "This is a practice run. Clicking Update changes nothing."
DONE = "Done. New work will use the newer version."
NOT_AVAILABLE = "This update is no longer available, so nothing changed."
NOT_APPLIED = "Raphael could not apply this update, so nothing changed."
NOT_WAITING = "This choice is no longer waiting."
# What the owner app says for each answer of the Settings undo and practice.
UNDO_TEXTS = {
    "undone": "The last model update was undone. New work uses the earlier version again.",
    "none": "There is no model update to undo.",
    "already_undone": "The last model update was already undone.",
    "busy": "Some work is running right now, so nothing changed. Try again when it has finished.",
    "not_applied": "Raphael could not undo this update, so nothing changed.",
}
PRACTICE_TEXTS = {
    "started": "A practice update is on its way. It will show in Decisions in a few minutes.",
    "already_started": "A practice update for this part of your team has already started this month.",
}
PRACTICE_CARD_TITLE = "Practice a model update for part of the team"

_ACTIONS = ("update", "later", "keep_current")
_KIND = "provider_model_policy"
_PREFIX = "model-update:"


@dataclass(frozen=True)
class ModelUpdate:
    provider: str
    from_model: str
    to_model: str
    month: str

    @property
    def practice(self) -> bool:
        return self.from_model == self.to_model


def parse_subject(kind: Any, subject: Any) -> Optional[ModelUpdate]:
    """The update a suggestion names, or None when it is no model update."""
    if kind != _KIND or not isinstance(subject, str) or not subject.startswith(_PREFIX):
        return None
    parts = subject.split(":")
    if len(parts) != 5 or not all(part and part == part.strip() for part in parts):
        return None
    return ModelUpdate(*parts[1:])


def feed_item(decision_ref: str, kind: Any, subject: Any) -> Optional[dict]:
    """The ``model_update`` item of one pending suggestion, or None."""
    update = parse_subject(kind, subject)
    if update is None:
        return None
    return {
        "decision_ref": decision_ref,
        "kind": "model_update",
        "title": TITLE,
        "reason": PRACTICE if update.practice else REASON,
        "actions": list(_ACTIONS),
        "practice": update.practice,
    }


def model_available(provider: str, model: str) -> bool:
    """Whether this account's fresh model list names ``model`` right now.

    The list is the provider's own answer for the sign-in the runtime resolves,
    never a picker catalog: a picker falls back to curated or default entries
    when that answer is empty or fails, and those say nothing about this
    account. No sign-in, a provider without such a list, an empty answer or a
    failed one is no. The sign-in goes to the provider read and nowhere else.
    """
    read = _ACCOUNT_MODEL_LISTS.get(provider)
    if read is None:
        return False
    try:
        listed = read()
    except Exception:
        return False  # e.g. no sign-in the runtime can resolve
    return bool(listed) and model in listed


def _anthropic_account_models() -> Optional[list[str]]:
    from agent.anthropic_credentials import resolve_anthropic_token
    from hermes_cli import models

    sign_in = resolve_anthropic_token()
    return models._fetch_anthropic_models(api_key=sign_in) if sign_in else None


def _codex_account_models() -> Optional[list[str]]:
    from hermes_cli import codex_models
    from hermes_cli.auth import resolve_codex_runtime_credentials

    sign_in = resolve_codex_runtime_credentials(refresh_if_expiring=True).get("api_key")
    return codex_models._fetch_models_from_api(sign_in) if sign_in else None


# The account model list of each provider an admitted route runs on.
_ACCOUNT_MODEL_LISTS = {
    "anthropic": _anthropic_account_models,
    "openai-codex": _codex_account_models,
}


def roles_on(provider: str, model: str) -> dict[str, str]:
    """Each role whose configured route runs ``model`` now, with its revision."""
    from hermes_cli.config import profile_route_revision
    from hermes_cli.profiles import get_profile_dir
    from plugins.dashboard_auth.raphael_workspace import model_policy

    roles: dict[str, str] = {}
    for profile in model_policy.admitted_profile_ids():
        try:
            route = model_policy.configured_assignment_for(profile)
        except Exception:
            continue  # A role without a readable approved route runs no model.
        if (route.provider, route.model) == (provider, model):
            roles[profile] = profile_route_revision(get_profile_dir(profile))
    return roles


def apply_model_update(ctx: OwnerContext, *, decision_ref: Any, reason: Any) -> dict:
    """Decide Update on one pending model update suggestion and apply it once.

    Refused with nothing written (``model_update_unavailable``) when no role
    runs the current model or the account does not list the new one now, and
    the suggestion is still waiting; if another click has taken it meanwhile,
    the answer is ``decision_not_found``. Otherwise the decision is taken by
    compare-and-swap before any write, so a lost race is ``decision_not_found``.
    A batch that fails after the decision leaves the record and every config as
    they were, appends one ``model_update_not_applied`` event and raises
    ``model_update_not_applied``.
    """
    if not isinstance(reason, str) or not reason.strip():
        raise OwnerWorkspaceError("invalid_argument", "a decision needs the owner's reason")
    reason = reason.strip()
    if len(reason) > kanban_db._RECOMMENDATION_LIFECYCLE_TEXT_MAX_LEN:
        raise OwnerWorkspaceError("invalid_argument", "the owner's reason is too long")

    board_slug, suggestion_id, version, update = _pending_update(ctx, decision_ref)
    roles = roles_on(update.provider, update.from_model)
    if not roles or not model_available(update.provider, update.to_model):
        # The click that took this decision meanwhile may be what moved the
        # routes; then the choice is gone, not unavailable.
        _pending_update(ctx, decision_ref)
        raise OwnerWorkspaceError("model_update_unavailable", NOT_AVAILABLE)

    with contextlib.closing(kanban_db.connect(board=board_slug)) as conn:
        try:
            kanban_db.decide_recommendation_on_owner_screen(
                conn,
                suggestion_id,
                decision="accepted",
                reason=reason,
                actor=ctx.actor,
                owner_screen_ref=f"owner-workspace:decisions:{decision_ref}",
                expected_lifecycle_version=version,
            )
        except ValueError as exc:
            raise OwnerWorkspaceError("decision_not_found", NOT_WAITING) from exc

    receipt = {
        "provider": update.provider,
        "from_model": update.from_model,
        "to_model": update.to_model,
        "roles": sorted(roles),
        "prior_revisions": dict(sorted(roles.items())),
        "practice": update.practice,
    }
    if update.practice:
        # A practice run writes no config and no update record.
        _append_receipt(board_slug, suggestion_id, "model_update_applied", receipt)
        return {"decision_ref": decision_ref, "decision": "accepted", "message": PRACTICE}
    try:
        revisions = _apply_routes(update, decision_ref, roles)
    except _NotApplied as exc:
        _append_receipt(
            board_slug, suggestion_id, "model_update_not_applied",
            {**receipt, "error": type(exc.__cause__).__name__},
        )
        raise OwnerWorkspaceError("model_update_not_applied", NOT_APPLIED) from exc
    _append_receipt(
        board_slug, suggestion_id, "model_update_applied",
        {**receipt, "revisions": dict(sorted(revisions.items()))},
    )
    return {"decision_ref": decision_ref, "decision": "accepted", "message": DONE}


def _pending_update(ctx: OwnerContext, decision_ref: Any) -> tuple[str, str, int, ModelUpdate]:
    """``(board, suggestion id, lifecycle version, update)`` of a waiting update."""
    found = _owner_pending_suggestion(ctx, decision_ref)
    if found is not None:
        board_slug, row = found
        with contextlib.closing(kanban_db.connect(board=board_slug)) as conn:
            named = conn.execute(
                "SELECT recommendation_kind, recommendation_subject_id FROM tasks "
                "WHERE id = ?",
                (str(row["id"]),),
            ).fetchone()
        update = named and parse_subject(
            named["recommendation_kind"], named["recommendation_subject_id"]
        )
        if update and not row["legacy"]:
            return (
                board_slug, str(row["id"]),
                int(row["recommendation_lifecycle_version"] or 0), update,
            )
    raise OwnerWorkspaceError("decision_not_found", NOT_WAITING)


class _NotApplied(Exception):
    """The update was refused with the record and every config as they were."""


def _apply_routes(update: ModelUpdate, decision_ref: str, roles: dict[str, str]) -> dict[str, str]:
    """Record the update, then move every role in one batch; undo the record on failure.

    The record is one document for every Update, and the undo puts back the
    whole text this Update read. So the record write, the batch and the undo
    run under the route batch journal lock, first in the route lock order (the
    batch takes it again inside): no other Update can commit an entry in
    between for the undo to erase from under that Update's live routes.

    A record that cannot be put back is not "nothing changed", so that error
    propagates as it is rather than as :class:`_NotApplied`.
    """
    from hermes_cli import config as hermes_config
    from plugins.dashboard_auth.raphael_workspace import model_policy

    committed: dict[str, str] = {}

    def enroll(revisions: dict[str, str]) -> None:
        with model_policy.enrollment_transaction(sorted(revisions)) as snapshot:
            hermes_config.mark_route_batch_enrollment_written(snapshot)
            hermes_config.mark_route_batch_audit_committed()
            committed.update(revisions)

    with contextlib.ExitStack() as held:
        try:
            held.enter_context(hermes_config._route_batch_journal_lock())
            prior = model_policy.apply_model_update_record({
                "provider": update.provider,
                "from_model": update.from_model,
                "to_model": update.to_model,
                "decision_ref": decision_ref,
                "approved_at": int(time.time()),
                "state": "active",
            })
        except Exception as exc:
            raise _NotApplied() from exc
        try:
            return hermes_config.apply_profile_route_batch(
                sorted(roles.items()),
                apply_one=functools.partial(_write_model, update.to_model),
                after_write=enroll,
            )
        except Exception as exc:
            if committed:
                return committed  # Rolled forward: the new routes are live.
            model_policy.restore_model_update_record(prior)
            raise _NotApplied() from exc


def _write_model(model: str, profile: str, expected_revision: str) -> str:
    """One role's guarded write: ``model`` on its own provider and effort, by CAS."""
    from hermes_cli.config import load_config, save_config
    from hermes_cli.profiles import get_profile_dir
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    token = set_hermes_home_override(str(get_profile_dir(profile)))
    try:
        cfg = load_config()
        route = cfg.get("model")
        route = dict(route) if isinstance(route, dict) else {}
        route["default"] = model
        route.pop("context_length", None)
        cfg["model"] = route
        return save_config(cfg, expected_revision=expected_revision)
    finally:
        reset_hermes_home_override(token)


def _append_receipt(board_slug: str, suggestion_id: str, kind: str, payload: dict) -> None:
    with contextlib.closing(kanban_db.connect(board=board_slug)) as conn:
        with kanban_db.write_txn(conn):
            kanban_db._append_event(conn, suggestion_id, kind, payload)


class Unavailable(Exception):
    """The record, the route journal, the audit or the board could not be used."""


def undo_last_model_update(*, journal) -> dict:
    """Roll the last applied model update back: its record entry and every role it moved.

    The last entry of the record is the last update; an entry already rolled
    back answers ``already_undone``, so an older update is never reached. The
    entry flips first, so the policy admits the earlier version again, and then
    every role on the newer version moves back in one batch with the revisions
    read at the start. ``journal(revisions, provider, model)`` writes the audit
    of the moved roles, each now on ``model``, inside the enrollment
    transaction. A batch that fails puts the record back, so nothing changed:
    ``busy`` when the fence refused, else ``not_applied``. Held under the route
    batch journal lock, as Update is. Raises :class:`Unavailable` for an
    unreadable record, an unreconciled journal or an audit that cannot be
    written; a record that cannot be put back raises as it is.
    """
    from hermes_cli import config as hermes_config
    from plugins.dashboard_auth.raphael_workspace import model_policy

    committed: dict[str, str] = {}

    def enroll(revisions: dict[str, str]) -> None:
        with model_policy.enrollment_transaction(sorted(revisions)) as snapshot:
            hermes_config.mark_route_batch_enrollment_written(snapshot)
            journal(revisions, last["provider"], last["from_model"])
            hermes_config.mark_route_batch_audit_committed()
            committed.update(revisions)

    with hermes_config._route_batch_journal_lock():
        try:
            entries = model_policy.read_model_updates()
            last = entries[-1] if entries else None
            roles = roles_on(last["provider"], last["to_model"]) if last else {}
        except (model_policy.EnrollmentUnavailable, hermes_config.RouteJournalUnreconciled) as exc:
            raise Unavailable() from exc
        if last is None:
            return {"state": "none"}
        update = {"from_model": last["from_model"], "to_model": last["to_model"]}
        if last["state"] != "active":
            return {"state": "already_undone", **update}
        try:
            prior = model_policy.apply_model_update_record({**last, "state": "rolled_back"})
        except model_policy.EnrollmentUnavailable as exc:
            raise Unavailable() from exc
        except ValueError:
            return {"state": "not_applied"}
        try:
            revisions = hermes_config.apply_profile_route_batch(
                sorted(roles.items()),
                apply_one=functools.partial(_write_model, last["from_model"]),
                after_write=enroll,
            ) if roles else {}
        except Exception as exc:
            if not committed:
                model_policy.restore_model_update_record(prior)
                return _undo_refused(exc, sorted(roles))
            revisions = committed  # Rolled forward: the earlier routes are live.
    try:
        _rolled_back_receipt(last, {
            **update, "provider": last["provider"], "roles": sorted(roles),
            "revisions": dict(sorted(revisions.items())), "prior_revisions": dict(sorted(roles.items())),
        })
    except Exception as exc:  # The undo stands; only its receipt is missing.
        _log.warning("model update undo: receipt not written (%s)", type(exc).__name__)
    return {"state": "undone", "roles": sorted(roles), **update}


def _undo_refused(exc: BaseException, roles: list[str]) -> dict:
    """The answer for a batch that changed nothing, by what refused it."""
    from hermes_cli.config import ProfileRouteBatchError, RouteJournalUnreconciled

    seen, cause = [], exc
    while cause is not None and cause not in seen:
        seen.append(cause)
        cause = cause.__cause__ or cause.__context__
    if any(isinstance(error, (RouteJournalUnreconciled, Unavailable)) for error in seen):
        raise Unavailable() from exc
    if any(getattr(error, "code", None) == "execution_state_busy" for error in seen):
        try:
            busy = _running_roles(roles)
        except Exception:
            busy = []
        refused = [error.profile for error in seen if isinstance(error, ProfileRouteBatchError) and error.profile]
        return {"state": "busy", "roles": busy or refused[:1]}
    return {"state": "not_applied"}


def _running_roles(roles: list[str]) -> list[str]:
    """Each of ``roles`` whose owner work the fence would refuse to move now; reads only."""
    from hermes_cli.owner_workspace import _owner_receipt_task_ids, _owner_tasks_for_role

    running = set()
    for board_slug, task_ids in sorted(_owner_receipt_task_ids().items()):
        with contextlib.closing(kanban_db.connect(board=board_slug)) as conn:
            for role in roles:
                held = _owner_tasks_for_role(conn, task_ids, role)
                if held and kanban_db.list_active_unpinned_owner_tasks(conn, task_ids=held):
                    running.add(role)
    return sorted(running)


def _rolled_back_receipt(entry: dict, receipt: dict) -> None:
    """One ``model_update_rolled_back`` event on the suggestion of ``entry``'s decision."""
    from hermes_cli.owner_workspace import _owner_decision_ref, list_committed_projects, resolve_owner_context

    ctx = resolve_owner_context()
    subject = f"{_PREFIX}{entry['provider']}:{entry['from_model']}:{entry['to_model']}:%"
    for project in list_committed_projects(ctx):
        with contextlib.closing(kanban_db.connect(board=project["board"])) as conn:
            rows = conn.execute(
                "SELECT id FROM tasks WHERE task_kind = 'recommendation' "
                "AND recommendation_kind = ? AND recommendation_subject_id LIKE ?",
                (_KIND, subject),
            ).fetchall()
        for row in rows:
            ref = _owner_decision_ref(ctx, authority="recommendation", native_id=str(row["id"]))
            if ref == entry["decision_ref"]:
                _append_receipt(project["board"], str(row["id"]), "model_update_rolled_back", receipt)
                return


def start_practice(project: dict, role: str) -> str:
    """File the practice card for ``role`` on ``project``'s board, once a month.

    The card is ordinary work of the planning role, sealed on that role's
    current route, and its worker runs the check card's command for ``role``.
    It is the only write. The look-up and the create run under the route batch
    journal lock, as Update and undo do, because the board's own idempotency
    look-up is not atomic. Answers ``started`` or ``already_started``.
    """
    from hermes_cli import config as hermes_config
    from hermes_cli.owner_workspace import _resolved_route_pin

    key = f"model-update-practice:{role}:{time.strftime('%Y-%m', time.gmtime())}"
    body = (
        f"Run `hermes model update-check --practice {role}` and finish with its receipt. "
        "It is a practice model update: it changes no setting."
    )
    with hermes_config._route_batch_journal_lock():
        with contextlib.closing(kanban_db.connect(board=project["board"])) as conn:
            found = conn.execute(
                "SELECT id FROM tasks WHERE idempotency_key = ? AND status != 'archived' "
                "AND task_kind = 'work'",
                (key,),
            ).fetchone()
            if found:
                return "already_started"
            kanban_db.create_task(
                conn, title=PRACTICE_CARD_TITLE, body=body, assignee="raphael-planner",
                project_id=project["project_id"], idempotency_key=key, board=project["board"],
                created_by="model-update-practice",
                **_resolved_route_pin("raphael-planner", "routine", "assignee"),
            )
    return "started"
