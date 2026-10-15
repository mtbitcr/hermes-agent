"""Mark the tool call that is ending a worker's run, so the run's receipt names it.

A run's receipt lists the tools read from the run's own session log. The call
that ends the run is still running while that receipt is built, so the log does
not hold its result row yet and the call would go unlisted. Each tool that ends
a run is decorated with ``ends_run``, and the receipt builder adds the marked
name. Outside such a call nothing is marked, so a run the kernel ends from
another process lists exactly its session's tools.
"""

from __future__ import annotations

import contextvars
import functools
from typing import Callable, Optional

_ENDING_TOOL: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "kanban_run_ending_tool", default=None
)


def ends_run(tool_name: str) -> Callable[[Callable], Callable]:
    """Mark every call of the decorated handler as ``tool_name`` ending the run."""

    def decorate(handler: Callable) -> Callable:
        @functools.wraps(handler)
        def marked(*args, **kwargs):
            token = _ENDING_TOOL.set(tool_name)
            try:
                return handler(*args, **kwargs)
            finally:
                _ENDING_TOOL.reset(token)

        return marked

    return decorate


def add_ending_tool(tools: set) -> None:
    """Add the name of the call ending this run to ``tools``; nothing when none is."""
    name = _ENDING_TOOL.get()
    if name:
        tools.add(name)
