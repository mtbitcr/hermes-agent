"""``kanban_model_update_check``: the worker of a model update check card runs the check.

A check card is made by the practice route (``created_by``
``model-update-practice``) or by the monthly host job (``model-update-check``).
Only its creator makes it one: any role can choose an idempotency key. Its
worker runs ``hermes_cli.model_update_check.run_update_check`` once, in its own
process, with no command line. The tool has no parameter: the card is
``HERMES_KANBAN_TASK``, read from the worker's own board, and a practice card's
role comes from its key ``model-update-practice:<role>:<year-month>``; the
monthly card has no role.

Registered here, beside ``tools.kanban_tools``, in the ``kanban`` toolset; the
built-in tool discovery imports this module by name. The check writes its
receipt on the card and files at most one suggestion. The receipt names models,
providers and sign-ins, so the answer holds only the result word, the reason
sentence and whether a suggestion was filed, and a check that fails answers
with the error's kind only.
"""

from __future__ import annotations

import logging
import os
import re
import sqlite3
from contextlib import closing
from typing import Optional

from agent.delegation_context import is_dispatcher_owned_worker_context
from tools.kanban_tools import _ok, _reject_delegated_child_mutation
from tools.registry import no_cache_check_fn, registry, tool_error

logger = logging.getLogger(__name__)

TOOL_NAME = "kanban_model_update_check"
PRACTICE_CREATOR = "model-update-practice"  # owner_model_updates.start_practice
MONTHLY_CREATOR = "model-update-check"  # the monthly host job
CHECK_CARD_CREATORS = frozenset({PRACTICE_CREATOR, MONTHLY_CREATOR})
# Upper length of the answer's JSON text; the check's own reason sentences are far shorter.
ANSWER_MAX_CHARS = 300
_PRACTICE_KEY = re.compile(r"model-update-practice:([^:]+):[0-9]{4}-(?:0[1-9]|1[0-2])")

KANBAN_MODEL_UPDATE_CHECK_SCHEMA = {
    "name": TOOL_NAME,
    "description": (
        "Run the model update check of your own card, here in your own process. Use it "
        "when your card asks you to run `hermes model update-check` (with or without "
        "--practice): this tool is that command. It reads all it needs from your card, "
        "so it takes no arguments. Call it once, then finish the card with its answer: "
        "the result, the reason and whether a suggestion was filed. If it refuses or "
        "fails, block the card with its exact answer; there is no other way to run the check."
    ),
    "parameters": {"type": "object", "properties": {}, "required": []},
}


def _check_card() -> Optional[sqlite3.Row]:
    """The worker's own card when it is a check card this worker runs, else None.

    The run must be the dispatcher's: a dispatcher-owned context (no
    delegate_task child, no cron job fired in-process), ``HERMES_KANBAN_TASK``
    and ``HERMES_KANBAN_RUN_ID`` set, and that run the card's current one,
    claimed for work and not from the review lane. The card is read read-only
    from the worker's own board. Any error is None.
    """
    task_id = os.environ.get("HERMES_KANBAN_TASK") or ""
    run_id = os.environ.get("HERMES_KANBAN_RUN_ID") or ""
    if not (task_id and re.fullmatch(r"[0-9]+", run_id) and is_dispatcher_owned_worker_context()):
        return None
    try:
        from hermes_cli import kanban_db as kb

        path = kb.kanban_db_path()
        with closing(sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True)) as conn:
            conn.row_factory = sqlite3.Row
            card = conn.execute(
                "SELECT created_by, idempotency_key, current_run_id FROM tasks WHERE id = ?", (task_id,),
            ).fetchone()
            if (card is None or card["created_by"] not in CHECK_CARD_CREATORS
                    or card["current_run_id"] != int(run_id)
                    or kb.run_claimed_from_review(conn, task_id, int(run_id))):
                return None
            return card
    except Exception:
        return None


def _practice_role(key: Optional[str]) -> Optional[str]:
    """The team role a practice card's key names, or None."""
    from plugins.dashboard_auth.raphael_workspace import model_policy

    match = _PRACTICE_KEY.fullmatch(key or "")
    if match is None or match.group(1) not in model_policy.admitted_profile_ids():
        return None
    return match.group(1)


@no_cache_check_fn
def _check_model_update_check_mode() -> bool:
    """Listed only for the worker the dispatcher started on a check card.

    Uncached: the card's state on its board decides, which the registry's
    TTL cache would not see change. The handler repeats the check.
    """
    return _check_card() is not None


def _answer(receipt: dict) -> str:
    """The result word, the reason sentence and whether a suggestion was filed.

    Nothing else of the receipt, which names models, providers and sign-ins
    and stays on the card. A reason that would make the answer longer than
    ANSWER_MAX_CHARS is cut short.
    """
    result, reason, filed = str(receipt["result"]), str(receipt["reason"]), receipt["filed"] is not None
    answer = _ok(result=result, reason=reason, filed=filed)
    if len(answer) > ANSWER_MAX_CHARS:
        # Each character takes at least one in the JSON text, so no longer prefix fits.
        reason = reason[:ANSWER_MAX_CHARS]
        while reason and len(_ok(result=result, reason=reason + "...", filed=filed)) > ANSWER_MAX_CHARS:
            reason = reason[:-1]
        answer = _ok(result=result, reason=reason + "...", filed=filed)
    return answer


def _run_check() -> str:
    refusal = _reject_delegated_child_mutation(TOOL_NAME)
    if refusal:
        return refusal
    card = _check_card()
    if card is None:
        return tool_error(
            f"{TOOL_NAME} refused: only the worker the dispatcher started on a model "
            "update check card may run the check"
        )
    role = None
    if card["created_by"] == PRACTICE_CREATOR:
        role = _practice_role(card["idempotency_key"])
        if role is None:
            return tool_error(
                f"{TOOL_NAME} refused: the practice card's key does not name one of the team's roles"
            )
    from hermes_cli import model_update_check

    return _answer(model_update_check.run_update_check(practice_role=role))


def _handle_model_update_check(args: dict, **kw) -> str:
    """Run the check once for the worker's own check card. ``args`` is never read."""
    try:
        return _run_check()
    except Exception as exc:
        # The kind only: an error's text can carry a sign-in value, a model or a path.
        logger.warning("%s: the check stopped (%s)", TOOL_NAME, type(exc).__name__)
        return tool_error(f"{TOOL_NAME} failed: {type(exc).__name__}")


registry.register(
    name=TOOL_NAME,
    toolset="kanban",
    schema=KANBAN_MODEL_UPDATE_CHECK_SCHEMA,
    handler=_handle_model_update_check,
    check_fn=_check_model_update_check_mode,
    emoji="🔎",
)
