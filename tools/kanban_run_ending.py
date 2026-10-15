"""Mark the tool call that is ending a worker's run, so the run's receipt names it.

A run's receipt lists the tools read from the run's own session log. The call
that ends the run is still running while that receipt is built, so the log does
not hold its result row yet and the call would go unlisted. Each tool that ends
a run is decorated with ``ends_run``, and the receipt builder adds the marked
name. The mark is bound to the calling worker's own session, the one its
session stamp trusts, and is added only to a receipt read from that session.
Outside such a call, and for a call from a process that is not a worker,
nothing is marked, so a run the kernel or another agent ends lists exactly its
session's tools.
"""

from __future__ import annotations

import contextvars
import functools
import os
from typing import Callable, Optional

# The calling worker's own session and the name of its call ending the run.
_ENDING_TOOL: contextvars.ContextVar[Optional[tuple[str, str]]] = (
    contextvars.ContextVar("kanban_run_ending_tool", default=None)
)


def _worker_session_id() -> Optional[str]:
    """This process's own session, as the worker's session stamp trusts it.

    Taken only inside a worker (``HERMES_KANBAN_TASK`` set); ``None`` otherwise.
    """
    if not os.environ.get("HERMES_KANBAN_TASK"):
        return None
    return os.environ.get("HERMES_SESSION_ID") or None


def ends_run(tool_name: str) -> Callable[[Callable], Callable]:
    """Mark every call of the decorated handler as ``tool_name`` ending the run."""

    def decorate(handler: Callable) -> Callable:
        @functools.wraps(handler)
        def marked(*args, **kwargs):
            session_id = _worker_session_id()
            token = _ENDING_TOOL.set((session_id, tool_name) if session_id else None)
            try:
                return handler(*args, **kwargs)
            finally:
                _ENDING_TOOL.reset(token)

        return marked

    return decorate


def add_ending_tool(tools: set, session_id: str) -> None:
    """Add the name of the call ending this run to ``tools``; nothing when none is.

    The name goes only into a receipt read from the calling worker's own
    session, ``session_id``; a receipt read from any other session is untouched.
    """
    mark = _ENDING_TOOL.get()
    if mark is not None and mark[0] == session_id:
        tools.add(mark[1])
