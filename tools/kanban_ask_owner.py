"""``kanban_ask_owner``: a worker that cannot go on asks the owner one question,
with up to four choices, on its own card, and the card waits.

Registered here, beside ``tools.kanban_tools``, and listed in the ``kanban``
toolset; the built-in tool discovery imports this module by name. The rules
and the one write transaction are ``owner_questions.declare_owner_question``;
this module decides only who may call it and on which card. Only a
dispatcher-owned worker sees or runs the tool, and only on the card the
dispatcher gave it (``HERMES_KANBAN_TASK``), on that card's own board.
"""

from __future__ import annotations

import logging
import os

from tools.kanban_tools import (
    _board_of_connection,
    _connect,
    _enforce_worker_task_ownership,
    _is_delegated_child_context,
    _is_dispatcher_owned_worker,
    _ok,
    _reject_delegated_child_mutation,
    _worker_run_id,
)
from tools.kanban_tools_schemas import KANBAN_ASK_OWNER_SCHEMA
from tools.registry import registry, tool_error

logger = logging.getLogger(__name__)

TOOL_NAME = "kanban_ask_owner"


def _check_ask_owner_mode() -> bool:
    """Listed only for a dispatcher-owned worker on its own card.

    The tool acts on ``HERMES_KANBAN_TASK`` alone, so an orchestrator
    profile, an ordinary chat, a delegate_task child and a cron job fired
    in-process from a worker never see it. The handler repeats every check
    at call time in case a registration is stale.
    """
    if _is_delegated_child_context():
        return False
    return bool(os.environ.get("HERMES_KANBAN_TASK")) and _is_dispatcher_owned_worker()


def _handle_ask_owner(args: dict, **kw) -> str:
    """Declare the question on the worker's own card and hold it for the owner."""
    from hermes_cli.owner_questions import declare_owner_question

    delegated_err = _reject_delegated_child_mutation(TOOL_NAME)
    if delegated_err:
        return delegated_err
    own_task = os.environ.get("HERMES_KANBAN_TASK")
    if not own_task or not _is_dispatcher_owned_worker():
        return tool_error(
            f"{TOOL_NAME} is only for the worker the dispatcher spawned on "
            "its own task"
        )
    # A stray task_id naming another card is refused; the worker's own card changes nothing.
    tid = args.get("task_id") or own_task
    ownership_err = _enforce_worker_task_ownership(tid)
    if ownership_err:
        return ownership_err
    try:
        kb, conn = _connect()
        try:
            declare_owner_question(
                conn,
                tid,
                args.get("stop", "question"),
                args.get("question"),
                args.get("choices"),
                args.get("words"),
                board=_board_of_connection(kb, conn) or kb.get_current_board(),
                expected_run_id=_worker_run_id(tid),
            )
            landed = kb.get_task(conn, tid)
            return _ok(
                task_id=tid,
                status=landed.status if landed else None,
                block_kind=landed.block_kind if landed else None,
            )
        finally:
            conn.close()
    except ValueError as e:
        return tool_error(f"{TOOL_NAME}: {e}")
    except Exception as e:
        logger.exception("%s failed", TOOL_NAME)
        return tool_error(f"{TOOL_NAME}: {e}")


registry.register(
    name=TOOL_NAME,
    toolset="kanban",
    schema=KANBAN_ASK_OWNER_SCHEMA,
    handler=_handle_ask_owner,
    check_fn=_check_ask_owner_mode,
    emoji="❓",
)
