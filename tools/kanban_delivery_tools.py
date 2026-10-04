"""Kanban delivery tools: the kernel's GitHub delivery steps for an integration card.

Slice 4 adds T1 ``kanban_delivery_publish`` only. It takes no input: the
source card, its approved head, the repository, the branch and the pull
request all come from the delivery record the kernel wrote, and the decision,
the GitHub work and the record live in
:func:`hermes_cli.kanban_delivery.publish_delivery`.

The tool lives in the ``kanban_delivery`` toolset, which is not in the default
tool list, and is listed only for the current run of the integration card a
delivery record names, with ``kanban.delivery.enabled`` true and the
repository in the policy. Registry dispatch does not consult the check
function, so the handler repeats every admission check at call time.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Optional

from agent.redact import redact_sensitive_text
from tools.registry import no_cache_check_fn, registry, tool_error

logger = logging.getLogger(__name__)


def _caller() -> tuple[Optional[tuple[str, int]], Optional[str]]:
    """``((task_id, run_id), None)`` for the dispatcher-owned worker run this
    process is, else ``(None, refusal code)``. Fails closed on any lookup error."""
    try:
        from agent.delegation_context import (
            is_delegated_child_process_context,
            is_dispatcher_owned_worker_context,
        )

        delegated = is_delegated_child_process_context()
        owned = is_dispatcher_owned_worker_context()
    except Exception:
        delegated, owned = True, False
    if delegated:
        return None, "delegated_child"
    task_id = os.environ.get("HERMES_KANBAN_TASK") or ""
    try:
        run_id = int(os.environ.get("HERMES_KANBAN_RUN_ID") or "")
    except ValueError:
        run_id = None
    if not owned or not task_id or run_id is None:
        return None, "not_current_run"
    return (task_id, run_id), None


@no_cache_check_fn
def _check_kanban_delivery_publish_mode() -> bool:
    """``kanban_delivery_publish`` is listed only when ``kanban.delivery.enabled``
    is true, the caller is the current run of the integration card a delivery
    record names, and that delivery's repository is in the policy (C1, C2).
    This only shapes the schema; ``_handle_publish`` repeats every check at call
    time. Any error hides the tool."""
    try:
        caller, _ = _caller()
        if caller is None:
            return False
        from hermes_cli import kanban_delivery as kd

        return kd.publish_available(*caller)
    except Exception:
        return False


def _refusal(code: str, detail: str) -> str:
    return redact_sensitive_text(
        tool_error(f"kanban_delivery_publish refused: {code}", code=code, detail=detail),
        force=True,
    )


def _handle_publish(args: dict, **kw) -> str:
    """T1. Fixed output fields only, redacted (C6); every refusal stores nothing."""
    if args:
        return _refusal("unknown_arguments", "kanban_delivery_publish takes no arguments")
    caller, code = _caller()
    if caller is None:
        return _refusal(code, "only the current run of the integration card may publish")
    from hermes_cli import kanban_delivery as kd

    try:
        result = kd.publish_delivery(*caller)
    except kd.PublishRefused as refused:
        return _refusal(refused.code, refused.detail)
    except Exception:
        logger.exception("kanban_delivery_publish failed")
        return _refusal("publish_failed", "the publish step failed; nothing was stored")
    return redact_sensitive_text(json.dumps({
        "state": result["state"],
        "head": result["head"],
        "pull_request_number": result["pull_request_number"],
        "branch": result["branch"],
    }), force=True)


KANBAN_DELIVERY_PUBLISH_SCHEMA = {
    "name": "kanban_delivery_publish",
    "description": (
        "Delivery step T1 for an integration card: publish the approved head of "
        "this card's delivery as its one pull request. Takes no arguments; the "
        "source card, head, branch, repository and pull request all come from "
        "the kernel's delivery record. Returns state (created, already_published "
        "or fast_forwarded), head, pull_request_number and branch. Calling it "
        "again is safe. A refusal returns a code and changes nothing; block the "
        "card (needs_input) with that code."
    ),
    "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
}


registry.register(
    name="kanban_delivery_publish",
    toolset="kanban_delivery",
    schema=KANBAN_DELIVERY_PUBLISH_SCHEMA,
    handler=_handle_publish,
    check_fn=_check_kanban_delivery_publish_mode,
    emoji="🚚",
)
