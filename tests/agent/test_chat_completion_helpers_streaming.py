"""The Anthropic streaming worker must retry, not abort, when the vendor SDK's
strict re-parse of the accumulated tool-call JSON rejects a literal control
character (a raw newline or tab) that the provider streamed inside a string
argument. The SDK surfaces that as a plain ``ValueError`` ("control character
(\\u0000-\\u001F) found while parsing a string ..."); without this the turn was
classified as a local bug, the worker exited without reporting and the run was
lost (nine runs on 2026-09-20)."""
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest


CONTROL_CHARACTER_MESSAGE = (
    "control character (\\u0000-\\u001F) found while parsing a string at line 2 column 0"
)


def _make_anthropic_agent(**kwargs):
    from run_agent import AIAgent

    defaults = dict(
        api_key="test-key",
        base_url="https://example.com/v1",
        model="claude-opus-4-7",
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
    )
    defaults.update(kwargs)
    agent = AIAgent(**defaults)
    agent.api_mode = "anthropic_messages"
    agent._anthropic_client = MagicMock()
    agent._anthropic_api_key = "test-anthropic-key"
    agent._create_request_anthropic_client = lambda *a, **k: agent._anthropic_client
    return agent


def _stream_cm(final_message, events=()):
    cm = MagicMock()
    stream = MagicMock()
    stream.__iter__ = MagicMock(return_value=iter(list(events)))
    stream.get_final_message = MagicMock(return_value=final_message)
    cm.__enter__ = MagicMock(return_value=stream)
    cm.__exit__ = MagicMock(return_value=False)
    return cm


def _raising_stream_cm(error):
    """A stream whose first event raises, the way the SDK's accumulating
    iterator dies on the delta that carries the raw control character."""

    def _boom():
        raise error
        yield  # pragma: no cover - makes this a generator

    cm = MagicMock()
    stream = MagicMock()
    stream.__iter__ = MagicMock(side_effect=lambda: _boom())
    cm.__enter__ = MagicMock(return_value=stream)
    cm.__exit__ = MagicMock(return_value=False)
    return cm


def _done_message():
    done = MagicMock()
    done.content = [SimpleNamespace(type="tool_use", id="toolu_x", name="terminal",
                                    input={"command": "echo a\nb"})]
    done.stop_reason = "tool_use"
    done.usage = SimpleNamespace(input_tokens=10, output_tokens=5)
    return done


def test_control_character_parse_error_is_a_wire_error():
    from agent.chat_completion_helpers import _is_control_character_stream_error

    assert _is_control_character_stream_error(ValueError(CONTROL_CHARACTER_MESSAGE))
    assert not _is_control_character_stream_error(ValueError("expected ident at line 1"))
    assert not _is_control_character_stream_error(ValueError("something else"))


def test_control_character_parse_error_retries_the_stream():
    """First attempt dies on the SDK re-parse; the retry returns the message."""
    agent = _make_anthropic_agent()
    agent._anthropic_client.messages.stream = MagicMock(side_effect=[
        _raising_stream_cm(ValueError(CONTROL_CHARACTER_MESSAGE)),
        _stream_cm(_done_message()),
    ])

    response = agent._interruptible_streaming_api_call({"model": "claude-opus-4-7"})

    assert response.stop_reason == "tool_use"
    assert response.content[0].input == {"command": "echo a\nb"}
    assert agent._anthropic_client.messages.stream.call_count == 2


def test_other_value_errors_still_propagate():
    """An unrelated ValueError keeps today's path: no retry, error surfaces."""
    agent = _make_anthropic_agent()
    agent._anthropic_client.messages.stream = MagicMock(side_effect=[
        _raising_stream_cm(ValueError("something unrelated")),
        _stream_cm(_done_message()),
    ])

    with pytest.raises(ValueError, match="something unrelated"):
        agent._interruptible_streaming_api_call({"model": "claude-opus-4-7"})
    assert agent._anthropic_client.messages.stream.call_count == 1


def test_empty_stream_still_reports_the_empty_stream_error():
    from agent.chat_completion_helpers import EmptyStreamError

    empty = MagicMock()
    empty.content = []
    empty.stop_reason = None
    agent = _make_anthropic_agent()
    agent._anthropic_client.messages.stream = MagicMock(return_value=_stream_cm(empty))

    with pytest.raises(EmptyStreamError):
        agent._interruptible_streaming_api_call({"model": "claude-opus-4-7"})
