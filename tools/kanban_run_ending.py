"""Mark the tool call that is ending a worker's run, so the run's receipt names it.

A run's receipt lists the tools read from the run's own session log. The call
that ends the run is still running while that receipt is built, so the log does
not hold its result row yet and the call would go unlisted. Each tool that ends
a run is decorated with ``ends_run``, and the kernel adds the marked name to
the receipt it stamps for the run (``with_ending_tool``).

The mark is bound to the run, not to a session (owner rule of 2026-10-15). It
holds the run the dispatcher launched the calling worker for, the injected
``HERMES_KANBAN_RUN_ID``, and names the call only in the receipt of that very
run, so a context compression that rebinds the worker to a child session
while the run stays linked to the parent changes nothing. A call from a
process with no injected run, such as an owner agent or an operator command,
marks nothing, so a run the kernel or another agent ends lists exactly its
session's tools.

The mark is made only in the worker's dispatcher-owned execution
(``agent.delegation_context.is_dispatcher_owned_worker_context``): a cron run
or a delegate_task child inside a worker process marks nothing, although it
inherits the worker's environment.
"""

from __future__ import annotations

import contextvars
import functools
import os
from typing import Callable, Optional

from agent.delegation_context import is_dispatcher_owned_worker_context

# The run the calling worker was launched for and the name of its call ending it.
_ENDING_TOOL: contextvars.ContextVar[Optional[tuple[int, str]]] = (
    contextvars.ContextVar("kanban_run_ending_tool", default=None)
)


def _injected_run_id() -> Optional[int]:
    """The run the dispatcher launched this worker process for, or ``None``.

    Taken only inside a worker (``HERMES_KANBAN_TASK`` set) and in its
    dispatcher-owned execution, from the ``HERMES_KANBAN_RUN_ID`` the
    dispatcher injects at the spawn.
    """
    if not os.environ.get("HERMES_KANBAN_TASK") or not is_dispatcher_owned_worker_context():
        return None
    try:
        return int(os.environ.get("HERMES_KANBAN_RUN_ID", ""))
    except ValueError:
        return None


def ends_run(tool_name: str) -> Callable[[Callable], Callable]:
    """Mark every call of the decorated handler as ``tool_name`` ending the run."""

    def decorate(handler: Callable) -> Callable:
        @functools.wraps(handler)
        def marked(*args, **kwargs):
            run_id = _injected_run_id()
            token = _ENDING_TOOL.set((run_id, tool_name) if run_id is not None else None)
            try:
                return handler(*args, **kwargs)
            finally:
                _ENDING_TOOL.reset(token)

        return marked

    return decorate


def with_ending_tool(receipt: dict, run_id: int) -> dict:
    """Return the receipt of run ``run_id``, naming the marked call that ends it.

    The name goes only into the receipt of the run the marked call's worker
    was launched for, and only into a capability read from the run's session
    log; any other receipt is returned unchanged. The tool list is bounded
    again, so it keeps the cap and the truncation flags of every receipt.
    """
    mark = _ENDING_TOOL.get()
    capability = receipt.get("capability") or {}
    if mark is None or mark[0] != run_id or capability.get("source") != "session-tool-calls":
        return receipt
    from tools.kanban_tools import _capability_bounded

    tools, cut = _capability_bounded({*capability["tools"], mark[1]})
    tools_truncated = capability["tools_truncated"] or cut
    return {
        **receipt,
        "capability": {
            **capability,
            "tools": tools,
            "tools_truncated": tools_truncated,
            "truncated": capability["truncated"] or tools_truncated,
        },
    }
