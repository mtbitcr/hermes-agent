"""Turn-end guard for kanban workers, which must end with ``kanban_complete`` or
``kanban_block``. Some models narrate the next step and stop with no tool calls;
Hermes treats that as a clean exit → ``rc=0`` → dispatcher ``protocol_violation``.
Policy-only: return a bounded synthetic nudge so the loop continues instead of exiting.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Iterable, Optional

from tools.tool_result_storage import PERSISTED_OUTPUT_CLOSING_TAG, PERSISTED_OUTPUT_TAG


_TERMINAL_KANBAN_TOOLS = frozenset({"kanban_complete", "kanban_block"})
_VERDICT_KANBAN_TOOLS = frozenset({"kanban_review_findings", "kanban_request_changes"})

_DEFAULT_MAX_ATTEMPTS = 2

_SUCCESS_START = '{"ok": true'
_FORM_ONE_HEAD = re.compile(
    re.escape(PERSISTED_OUTPUT_TAG) + r"\n"
    r"(?:(?!Full output saved to: )[^\n]+\n)*"
    r"Full output saved to: \S[^\n]*\n"
    r"(?:[^\n]+\n)*"
    r"\nPreview \(first \d+ chars\):\n"
)
_FORM_ONE_END = "\n" + PERSISTED_OUTPUT_CLOSING_TAG
_FORM_TWO_NOTICE = re.compile(
    r"\n\n\[Truncated: tool response was \d{1,3}(?:,\d{3})* chars\. "
    r"Full output could not be saved to sandbox\.\]\Z"
)


def kanban_stop_nudge_enabled() -> bool:
    """On when ``HERMES_KANBAN_TASK`` is set, unless ``HERMES_KANBAN_STOP_NUDGE`` disables it."""
    if (os.environ.get("HERMES_KANBAN_STOP_NUDGE") or "").strip().lower() in {"0", "false", "no", "off"}:
        return False
    return bool((os.environ.get("HERMES_KANBAN_TASK") or "").strip())


def _tool_call_name(tc: Any) -> str:
    """Tool name from a dict or object tool call (``function.name`` first, then ``name``)."""
    if isinstance(tc, dict):
        fn = tc.get("function")
        return str((fn.get("name") if isinstance(fn, dict) else tc.get("name")) or "")
    fn = getattr(tc, "function", None)
    return str((getattr(fn, "name", "") if fn is not None else getattr(tc, "name", "")) or "")


def _replaced_result_starts_with_success(content: str) -> bool:
    """True when ``content`` is a result the tool publisher replaced with a preview and the
    result's first characters are a success. Form one: tagged block with a saved-output line,
    then ``Preview (first N chars):`` after a blank line, ending with the closing tag. Form two:
    result start, then the publisher's truncation notice as the very end of the text."""
    head = _FORM_ONE_HEAD.match(content)
    if head is not None:
        return content.endswith(_FORM_ONE_END) and content.startswith(_SUCCESS_START, head.end())
    return content.startswith(_SUCCESS_START) and _FORM_TWO_NOTICE.search(content) is not None


def _verdict_result_succeeded(content: Any) -> bool:
    """True for the board tools' success form (``{"ok": true, ...}``), not their error form.
    A result over 100,000 chars is replaced by the tool publisher with a preview, which is not
    complete JSON; it counts as a success when the replaced form starts with the success form."""
    try:
        data = json.loads(content) if isinstance(content, str) else None
    except ValueError:
        return isinstance(content, str) and _replaced_result_starts_with_success(content)
    return isinstance(data, dict) and data.get("ok") is True and "error" not in data


def session_called_kanban_terminal(messages: Iterable[dict] | None) -> bool:
    """True if this conversation already invoked a terminal kanban tool, or a review
    verdict tool whose tool result is a success (a refused or unanswered verdict is not)."""
    for msg in filter(lambda m: isinstance(m, dict), messages or ()):
        role = msg.get("role")
        if role == "assistant" and any(
            _tool_call_name(tc) in _TERMINAL_KANBAN_TOOLS for tc in msg.get("tool_calls") or []
        ):
            return True
        if role == "tool":
            name = str(msg.get("name") or "")
            if name in _TERMINAL_KANBAN_TOOLS:
                return True
            if name in _VERDICT_KANBAN_TOOLS and _verdict_result_succeeded(msg.get("content")):
                return True
    return False


def build_kanban_stop_nudge(
    *,
    messages: Iterable[dict] | None = None,
    attempts: int = 0,
    max_attempts: int = _DEFAULT_MAX_ATTEMPTS,
    task_id: Optional[str] = None,
) -> Optional[str]:
    """Synthetic follow-up when a kanban worker exits without a terminal tool; ``None`` when
    the guard should not fire (not a kanban worker, already completed/blocked, budget exhausted)."""
    if (
        not kanban_stop_nudge_enabled()
        or attempts >= max_attempts
        or session_called_kanban_terminal(messages)
    ):
        return None

    tid = (task_id or os.environ.get("HERMES_KANBAN_TASK") or "").strip() or "this task"
    return (
        "[System: You are a Hermes kanban worker. A plain-text reply is NOT a "
        "terminal state for the board.\n\n"
        f"Task `{tid}` is still `running`. Ending now without a board tool "
        "causes a protocol violation (clean exit with no "
        "`kanban_complete` / `kanban_block`).\n\n"
        "Do this immediately in your next response — do not narrate intent:\n"
        "1. Finish any remaining deliverable (write the required file(s) now).\n"
        "2. Call `kanban_complete(summary=..., artifacts=[...])` if the work "
        "is done, OR `kanban_block(reason=...)` if you are blocked.\n\n"
        "Never end a turn with only a promise of future action. Repeated "
        "protocol violations will block this task and require manual intervention.]"
    )


__all__ = ["build_kanban_stop_nudge", "kanban_stop_nudge_enabled", "session_called_kanban_terminal"]
