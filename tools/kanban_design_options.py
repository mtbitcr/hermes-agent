"""``kanban_offer_design_options``: the designer offers the owner two or three
pictured options for its own card, and the card waits.

Registered here, beside ``tools.kanban_tools``, and listed in the ``kanban``
toolset; the built-in tool discovery imports this module by name. The rules
and the one write transaction are ``owner_design_choices.declare_design_options``;
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
from tools.kanban_tools_schemas import KANBAN_OFFER_DESIGN_OPTIONS_SCHEMA
from tools.registry import registry, tool_error

logger = logging.getLogger(__name__)

TOOL_NAME = "kanban_offer_design_options"


def _check_design_options_mode() -> bool:
    """Listed only for a dispatcher-owned worker on its own card.

    The tool acts on ``HERMES_KANBAN_TASK`` alone, so an orchestrator
    profile, an ordinary chat, a delegate_task child and a cron job fired
    in-process from a worker never see it. The handler repeats every check
    at call time in case a registration is stale.
    """
    if _is_delegated_child_context():
        return False
    return bool(os.environ.get("HERMES_KANBAN_TASK")) and _is_dispatcher_owned_worker()


def _handle_offer_design_options(args: dict, **kw) -> str:
    """Declare the options on the worker's own card and hold it for the owner."""
    from hermes_cli.owner_design_choices import declare_design_options

    delegated_err = _reject_delegated_child_mutation(TOOL_NAME)
    if delegated_err:
        return delegated_err
    own_task = os.environ.get("HERMES_KANBAN_TASK")
    if not own_task or not _is_dispatcher_owned_worker():
        return tool_error(
            f"{TOOL_NAME} is only for the worker the dispatcher spawned on "
            "its own task"
        )
    tid = args.get("task_id") or own_task
    ownership_err = _enforce_worker_task_ownership(tid)
    if ownership_err:
        return ownership_err
    try:
        kb, conn = _connect()
        try:
            declared = declare_design_options(
                conn,
                tid,
                args.get("question"),
                args.get("options"),
                board=_board_of_connection(kb, conn) or kb.get_current_board(),
                expected_run_id=_worker_run_id(tid),
            )
            landed = kb.get_task(conn, tid)
            return _ok(
                task_id=tid,
                status=landed.status if landed else None,
                block_kind=landed.block_kind if landed else None,
                declared=declared,
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
    schema=KANBAN_OFFER_DESIGN_OPTIONS_SCHEMA,
    handler=_handle_offer_design_options,
    check_fn=_check_design_options_mode,
    emoji="🎨",
)
