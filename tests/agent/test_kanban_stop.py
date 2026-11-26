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


# ── Verdict results the tool publisher replaced with a preview ───────

_VERDICT_TOOLS = ["kanban_review_findings", "kanban_request_changes"]
_SUCCESS_START = '{"ok": true, "task_id": "t_abc", "re_raised_fingerprints": ["f1", "f2"'
_ERROR_START = '{"error": "kanban_review_findings: refused", "detail": "x'


def _form_one(start: str) -> str:
    from tools.tool_result_storage import PERSISTED_OUTPUT_CLOSING_TAG, PERSISTED_OUTPUT_TAG

    return (
        f"{PERSISTED_OUTPUT_TAG}\n"
        "This tool result was too large (150,000 characters, 146.5 KB).\n"
        "Full output saved to: /tmp/spill/1.txt\n"
        "Use the read_file tool with offset and limit to access specific sections of this output.\n\n"
        f"Preview (first {len(start)} chars):\n"
        f"{start}\n...\n{PERSISTED_OUTPUT_CLOSING_TAG}"
    )


def _form_two(start: str) -> str:
    return (
        f"{start}\n\n[Truncated: tool response was 150,000 chars. "
        "Full output could not be saved to sandbox.]"
    )


_FORMS = [pytest.param(_form_one, id="form-one"), pytest.param(_form_two, id="form-two")]


def _nudge_for(name: str, content: str):
    return build_kanban_stop_nudge(messages=[_verdict_call(name), _verdict_result(name, content)])


@pytest.mark.parametrize("form", _FORMS)
@pytest.mark.parametrize("verdict", _VERDICT_TOOLS)
def test_no_nudge_when_replaced_verdict_result_starts_with_success(clear_kanban_env, verdict, form):
    clear_kanban_env.setenv("HERMES_KANBAN_TASK", "t_abc")
    messages = [_verdict_call(verdict), _verdict_result(verdict, form(_SUCCESS_START))]
    assert session_called_kanban_terminal(messages) is True
    assert build_kanban_stop_nudge(messages=messages) is None


@pytest.mark.parametrize("form", _FORMS)
@pytest.mark.parametrize("verdict", _VERDICT_TOOLS)
def test_nudge_when_replaced_verdict_result_starts_with_error(clear_kanban_env, verdict, form):
    clear_kanban_env.setenv("HERMES_KANBAN_TASK", "t_abc")
    messages = [_verdict_call(verdict), _verdict_result(verdict, form(_ERROR_START))]
    assert session_called_kanban_terminal(messages) is False
    assert build_kanban_stop_nudge(messages=messages) is not None


@pytest.mark.parametrize("form", _FORMS)
def test_nudge_when_replaced_result_is_of_a_non_verdict_tool(clear_kanban_env, form):
    clear_kanban_env.setenv("HERMES_KANBAN_TASK", "t_abc")
    for tool in ("kanban_heartbeat", "terminal"):
        messages = [_verdict_call(tool), _verdict_result(tool, form(_SUCCESS_START))]
        assert session_called_kanban_terminal(messages) is False
        assert build_kanban_stop_nudge(messages=messages) is not None


@pytest.mark.parametrize("verdict", _VERDICT_TOOLS)
def test_nudge_when_success_start_has_neither_form(clear_kanban_env, verdict):
    clear_kanban_env.setenv("HERMES_KANBAN_TASK", "t_abc")
    from tools.tool_result_storage import PERSISTED_OUTPUT_CLOSING_TAG, PERSISTED_OUTPUT_TAG

    texts = [
        _SUCCESS_START,
        _SUCCESS_START + "\n...",
        # Form one pieces missing or in the wrong place.
        f"{PERSISTED_OUTPUT_TAG}\nFull output saved to: /tmp/x\n{_SUCCESS_START}\n{PERSISTED_OUTPUT_CLOSING_TAG}",
        _form_one(_SUCCESS_START).replace(PERSISTED_OUTPUT_CLOSING_TAG, ""),
        _form_one(_SUCCESS_START).replace(PERSISTED_OUTPUT_TAG, "", 1),
        # Form two marker missing, or success start not first.
        _SUCCESS_START + "\n\nTruncated: tool response was 150,000 chars.",
        "note\n" + _form_two(_SUCCESS_START),
        "x " + _SUCCESS_START + "\n[Truncated: tool response was 1 chars.]",
    ]
    for text in texts:
        assert _nudge_for(verdict, text) is not None, text


@pytest.mark.parametrize("verdict", _VERDICT_TOOLS)
def test_nudge_when_form_one_preview_does_not_start_with_success(clear_kanban_env, verdict):
    clear_kanban_env.setenv("HERMES_KANBAN_TASK", "t_abc")
    # Success text sits in the header, not at the start of the preview.
    text = _form_one(_ERROR_START).replace("Full output saved to: /tmp/spill/1.txt", _SUCCESS_START)
    assert _nudge_for(verdict, text) is not None


@pytest.mark.parametrize("verdict", _VERDICT_TOOLS)
def test_complete_verdict_results_read_as_before(clear_kanban_env, verdict):
    clear_kanban_env.setenv("HERMES_KANBAN_TASK", "t_abc")
    assert _nudge_for(verdict, '{"ok": true, "task_id": "t_abc"}') is None
    assert _nudge_for(verdict, '{"ok": true, "error": "x"}') is not None
    assert _nudge_for(verdict, _refusal(verdict)) is not None
    assert _nudge_for(verdict, "") is not None
    assert build_kanban_stop_nudge(messages=[_verdict_call(verdict)]) is not None


@pytest.mark.parametrize("form", _FORMS)
@pytest.mark.parametrize("verdict", _VERDICT_TOOLS)
def test_guard_opens_no_saved_file_or_board_store(clear_kanban_env, monkeypatch, tmp_path, verdict, form):
    import builtins
    import io
    import pathlib

    clear_kanban_env.setenv("HERMES_KANBAN_TASK", "t_abc")

    def _forbidden(*args, **kwargs):
        raise AssertionError("guard must not open files or the board store")

    monkeypatch.setattr(builtins, "open", _forbidden)
    monkeypatch.setattr(io, "open", _forbidden)
    for attr in ("open", "read_text", "read_bytes"):
        monkeypatch.setattr(pathlib.Path, attr, _forbidden)
    monkeypatch.setattr("sqlite3.connect", _forbidden)

    saved = tmp_path / "1.txt"
    content = form(_SUCCESS_START).replace("/tmp/spill/1.txt", str(saved))
    assert _nudge_for(verdict, content) is None
    assert _nudge_for(verdict, form(_ERROR_START)) is not None
    assert not saved.exists()


def _publish(content: str, tool_name: str):
    from tools.tool_result_storage import maybe_persist_tool_result

    return maybe_persist_tool_result(content, tool_name, "call_1")


def _long_success() -> str:
    import json

    return json.dumps(
        {"ok": True, "task_id": "t_abc", "re_raised_fingerprints": [f"fp{i:06d}" for i in range(12000)]}
    )


@pytest.mark.parametrize("verdict", _VERDICT_TOOLS)
def test_publisher_form_one_reads_as_success(clear_kanban_env, monkeypatch, tmp_path, verdict):
    from tools.tool_result_storage import PERSISTED_OUTPUT_TAG

    clear_kanban_env.setenv("HERMES_KANBAN_TASK", "t_abc")
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    full = _long_success()
    assert len(full) > 100_000
    replaced = _publish(full, verdict)
    assert replaced.startswith(PERSISTED_OUTPUT_TAG)
    assert replaced != full
    messages = [_verdict_call(verdict), _verdict_result(verdict, replaced)]
    assert session_called_kanban_terminal(messages) is True
    assert build_kanban_stop_nudge(messages=messages) is None
    # The same replaced form of another tool's result is no verdict.
    other = [_verdict_call("kanban_heartbeat"), _verdict_result("kanban_heartbeat", replaced)]
    assert build_kanban_stop_nudge(messages=other) is not None
    # Tagged block with a success-looking preview and closing tag, but no saved-output line.
    lines = replaced.split("\n")
    no_path = "\n".join(ln for ln in lines if not ln.startswith("Full output saved to:"))
    assert no_path != replaced and no_path.startswith(PERSISTED_OUTPUT_TAG)
    missing = [_verdict_call(verdict), _verdict_result(verdict, no_path)]
    assert session_called_kanban_terminal(missing) is False
    assert build_kanban_stop_nudge(messages=missing) is not None


@pytest.mark.parametrize("verdict", _VERDICT_TOOLS)
def test_publisher_form_two_reads_as_success(clear_kanban_env, monkeypatch, tmp_path, verdict):
    from tools.tool_result_storage import PERSISTED_OUTPUT_TAG

    clear_kanban_env.setenv("HERMES_KANBAN_TASK", "t_abc")
    # HERMES_HOME is a file, so the publisher cannot create its spillover folder.
    blocker = tmp_path / "not_a_dir"
    blocker.write_text("x")
    monkeypatch.setenv("HERMES_HOME", str(blocker))
    full = _long_success()
    replaced = _publish(full, verdict)
    assert PERSISTED_OUTPUT_TAG not in replaced
    assert "\n[Truncated: tool response was" in replaced
    messages = [_verdict_call(verdict), _verdict_result(verdict, replaced)]
    assert session_called_kanban_terminal(messages) is True
    assert build_kanban_stop_nudge(messages=messages) is None
    other = [_verdict_call("kanban_heartbeat"), _verdict_result("kanban_heartbeat", replaced)]
    assert build_kanban_stop_nudge(messages=other) is not None


@pytest.mark.parametrize("verdict", _VERDICT_TOOLS)
def test_nudge_when_form_two_notice_is_cut_off_or_followed_by_text(clear_kanban_env, verdict):
    clear_kanban_env.setenv("HERMES_KANBAN_TASK", "t_abc")
    texts = [
        _SUCCESS_START + "\n[Truncated: tool response was",
        _SUCCESS_START + "\n\n[Truncated: tool response was",
        _form_two(_SUCCESS_START) + "\nunrelated text",
        _form_two(_SUCCESS_START) + "\n",
    ]
    for text in texts:
        assert _nudge_for(verdict, text) is not None, text
    assert _nudge_for(verdict, _form_two(_SUCCESS_START)) is None



# ── Integration: agent nudge + dispatcher bounded retry ──────────────
# These tests verify the two layers compose correctly: the agent-side
# nudge fires first (up to 2 attempts), and if the worker still exits
# without a terminal call, the dispatcher's bounded retry (streak of 3)
# handles it.  See also tests/hermes_cli/test_kanban_core_functionality.py
# for the dispatcher-side streak tests.




