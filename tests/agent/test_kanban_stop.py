"""Tests for the kanban worker turn-end stop guard."""

from __future__ import annotations

import pytest

from agent.kanban_stop import (
    build_kanban_stop_nudge,
    kanban_stop_nudge_enabled,
    session_called_kanban_terminal,
)


@pytest.fixture
def clear_kanban_env(monkeypatch):
    for var in ("HERMES_KANBAN_TASK", "HERMES_KANBAN_STOP_NUDGE"):
        monkeypatch.delenv(var, raising=False)
    return monkeypatch






def test_env_can_disable(clear_kanban_env):
    clear_kanban_env.setenv("HERMES_KANBAN_TASK", "t_abc")
    clear_kanban_env.setenv("HERMES_KANBAN_STOP_NUDGE", "0")
    assert kanban_stop_nudge_enabled() is False
    assert build_kanban_stop_nudge(messages=[]) is None


def test_nudge_when_no_terminal_tool(clear_kanban_env):
    clear_kanban_env.setenv("HERMES_KANBAN_TASK", "t_46be8aa5")
    messages = [
        {"role": "user", "content": "work kanban task"},
        {
            "role": "assistant",
            "content": "Let me write the comprehensive recipe.",
            "tool_calls": [
                {
                    "id": "1",
                    "type": "function",
                    "function": {"name": "kanban_heartbeat", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "name": "kanban_heartbeat", "tool_call_id": "1", "content": "ok"},
    ]
    nudge = build_kanban_stop_nudge(messages=messages, attempts=0)
    assert nudge is not None
    assert "kanban_complete" in nudge
    assert "kanban_block" in nudge
    assert "t_46be8aa5" in nudge
    assert "protocol violation" in nudge.lower() or "protocol" in nudge.lower()


def test_no_nudge_after_kanban_complete(clear_kanban_env):
    clear_kanban_env.setenv("HERMES_KANBAN_TASK", "t_abc")
    messages = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "1",
                    "type": "function",
                    "function": {"name": "kanban_complete", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "name": "kanban_complete", "tool_call_id": "1", "content": "done"},
    ]
    assert session_called_kanban_terminal(messages) is True
    assert build_kanban_stop_nudge(messages=messages) is None


def _verdict_call(name: str) -> dict:
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {"id": "1", "type": "function", "function": {"name": name, "arguments": "{}"}}
        ],
    }


def _verdict_result(name: str, content: str) -> dict:
    return {"role": "tool", "name": name, "tool_call_id": "1", "content": content}


def _refusal(name: str) -> str:
    from tools.kanban_tools import _handle_request_changes, _handle_review_findings

    if name == "kanban_request_changes":
        return _handle_request_changes({})
    return _handle_review_findings({"findings": []})


def test_no_nudge_after_successful_review_findings(clear_kanban_env):
    clear_kanban_env.setenv("HERMES_KANBAN_TASK", "t_abc")
    from tools.kanban_tools import _ok

    # Success form of _handle_review_findings: _ok(task_id=tid, **result).
    messages = [
        _verdict_call("kanban_review_findings"),
        _verdict_result(
            "kanban_review_findings",
            _ok(task_id="t_abc", outcome="passed", re_raised_fingerprints=[]),
        ),
    ]
    assert session_called_kanban_terminal(messages) is True
    assert build_kanban_stop_nudge(messages=messages) is None


def test_no_nudge_after_successful_request_changes(clear_kanban_env):
    clear_kanban_env.setenv("HERMES_KANBAN_TASK", "t_abc")
    from tools.kanban_tools import _ok

    # Success form of _handle_request_changes.
    messages = [
        _verdict_call("kanban_request_changes"),
        _verdict_result(
            "kanban_request_changes",
            _ok(task_id="t_abc", run_id=3, status="ready", implementer="builder"),
        ),
    ]
    assert session_called_kanban_terminal(messages) is True
    assert build_kanban_stop_nudge(messages=messages) is None


def test_nudge_after_refused_review_findings(clear_kanban_env):
    clear_kanban_env.setenv("HERMES_KANBAN_TASK", "t_abc")
    from tools.kanban_tools import _handle_review_findings

    # Real refusal: this process is not the dispatcher-owned reviewer run.
    refusal = _handle_review_findings({"findings": []})
    assert '"error"' in refusal
    messages = [
        _verdict_call("kanban_review_findings"),
        _verdict_result("kanban_review_findings", refusal),
    ]
    assert session_called_kanban_terminal(messages) is False
    assert build_kanban_stop_nudge(messages=messages) is not None


def test_nudge_after_refused_request_changes(clear_kanban_env):
    clear_kanban_env.setenv("HERMES_KANBAN_TASK", "t_abc")

    # Real refusal: _handle_request_changes rejects a call with no reason.
    refusal = _refusal("kanban_request_changes")
    assert '"error"' in refusal
    messages = [
        _verdict_call("kanban_request_changes"),
        _verdict_result("kanban_request_changes", refusal),
    ]
    assert session_called_kanban_terminal(messages) is False
    assert build_kanban_stop_nudge(messages=messages) is not None


@pytest.mark.parametrize("verdict", ["kanban_review_findings", "kanban_request_changes"])
def test_nudge_when_heartbeat_success_reuses_refused_verdict_call_id(clear_kanban_env, verdict):
    clear_kanban_env.setenv("HERMES_KANBAN_TASK", "t_abc")
    from tools.kanban_tools import _ok

    # Call ids are unique only within one assistant batch: the heartbeat in a
    # later batch reuses id "1" and its own success must not read as a verdict.
    messages = [
        _verdict_call(verdict),
        _verdict_result(verdict, _refusal(verdict)),
        _verdict_call("kanban_heartbeat"),
        _verdict_result("kanban_heartbeat", _ok(task_id="t_abc", run_id=3)),
    ]
    assert session_called_kanban_terminal(messages) is False
    assert build_kanban_stop_nudge(messages=messages) is not None


def test_nudge_after_verdict_call_without_tool_result(clear_kanban_env):
    clear_kanban_env.setenv("HERMES_KANBAN_TASK", "t_abc")
    messages = [_verdict_call("kanban_review_findings")]
    assert session_called_kanban_terminal(messages) is False
    assert build_kanban_stop_nudge(messages=messages) is not None





# ── Integration: agent nudge + dispatcher bounded retry ──────────────
# These tests verify the two layers compose correctly: the agent-side
# nudge fires first (up to 2 attempts), and if the worker still exits
# without a terminal call, the dispatcher's bounded retry (streak of 3)
# handles it.  See also tests/hermes_cli/test_kanban_core_functionality.py
# for the dispatcher-side streak tests.




