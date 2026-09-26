"""Owner-history projection of a turn the model provider declined as a safety matter.

The platform answers such a turn with its own fixed refusal notice
(``agent.turn_truncation.handle_content_policy_refusal``). Sending the same
request again unchanged would only be declined again, so a trailing turn that
ends on that notice tells the owner that rewording may help, instead of the
generic failure that invites a resend. Every other unreadable reply, and every
interior turn, projects exactly as before: those expected snapshots were pinned
from the projection at the starting commit (76f3c940).
"""

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from agent import turn_truncation
from gateway.platforms.api_server import ResponseStore

_NAME = "raphael-owner-" + "7" * 32
_RESPONSE_ID = "resp_owner_turn"
_EXPLANATION = "I can't help with disabling the office alarm."

_QUESTION_REPLY = (
    '{"schema_version": 1, "kind": "question", "message": "Which first?"}'
)
_UNREADABLE_TURN_REPLY = (
    '{"schema_version": 1, "kind": "failure", "message": "Raphael could not '
    'prepare a safe plan for this request. Nothing was changed. You can send '
    'it again."}'
)
_REFUSED_TURN_REPLY = (
    '{"schema_version": 1, "kind": "failure", "message": "Raphael could not '
    'work on this request as it is worded. Nothing was changed. Rewording the '
    'request may help."}'
)
_TOOL_CALL_IMITATION = json.dumps({
    "tool": "run_command",
    "arguments": {"command": "rm -rf /srv/workshop"},
})


def _written_notice():
    """The notice the real writer produces when the provider declines a turn."""
    agent = MagicMock()
    agent.api_mode = "chat_completions"
    agent._has_pending_fallback.return_value = False
    agent._try_activate_fallback.return_value = False
    agent._get_transport.return_value.normalize_response.return_value = (
        SimpleNamespace(content=_EXPLANATION)
    )
    verdict = turn_truncation.handle_content_policy_refusal(
        agent, SimpleNamespace(), None, thinking_spinner=None, messages=[],
        api_messages=[], api_kwargs={}, active_system_prompt=None,
        conversation_history=None, api_call_count=1, effective_task_id=None,
        turn_id=None, api_request_id=None, api_start_time=0.0, retry_count=0,
        max_retries=3,
    )
    assert verdict.action == "return"
    return verdict.result["final_response"]


def _owner_store(history):
    store = ResponseStore(max_size=10)
    store.put(_RESPONSE_ID, {
        "response": {"id": _RESPONSE_ID},
        "conversation_history": history,
    })
    assert store.set_conversation(_NAME, _RESPONSE_ID) is True
    return store


def _stored_state(store):
    """Everything the store holds, statement by statement."""
    return list(store._conn.iterdump())


def _expected_snapshot(*, incomplete, data):
    """A whole projection of a conversation with no proposal, run or recovery."""
    return {
        "head_response_id": _RESPONSE_ID,
        "latest_response_id": None,
        "proposal_consumed": False,
        "proposal_claimed": False,
        "active_run_id": None,
        "completed_run_id": None,
        "released_run_id": None,
        "conversation_closed": False,
        "truncated": False,
        "incomplete": incomplete,
        "pending": None,
        "recovery": None,
        "data": data,
    }


def test_a_trailing_refused_turn_is_shown_as_needing_rewording():
    """Resending a declined request unchanged is declined again, so the owner is
    told that rewording may help — and never shown the notice itself."""
    notice = _written_notice()
    store = _owner_store([
        {"role": "user", "content": "Plan it."},
        {"role": "assistant", "content": _QUESTION_REPLY},
        {"role": "user", "content": "The second one."},
        {"role": "assistant", "content": notice},
    ])
    stored_before = _stored_state(store)

    snapshot = store.owner_history_snapshot(_NAME)

    assert snapshot["data"][-1] == {
        "owner": "The second one.",
        "raphael": _REFUSED_TURN_REPLY,
    }
    # Everything else is what any completed trailing turn already gets.
    assert snapshot == _expected_snapshot(incomplete=False, data=[
        {"owner": "Plan it.", "raphael": _QUESTION_REPLY},
        {"owner": "The second one.", "raphael": _REFUSED_TURN_REPLY},
    ])
    # Only the projection changed: the stored row still holds the notice.
    assert _stored_state(store) == stored_before
    (stored,) = store._conn.execute(
        "SELECT data FROM responses WHERE response_id = ?", (_RESPONSE_ID,),
    ).fetchone()
    assert json.loads(stored)["conversation_history"][-1] == {
        "role": "assistant", "content": notice,
    }
    # Neither the model's explanation nor any part of the notice reaches the
    # owner.
    projected = json.dumps(snapshot, ensure_ascii=False)
    for part in (
        _EXPLANATION, *notice.split("\n\n"),
        "⚠", "declined", "refusal", "explanation", "fallback",
    ):
        assert part not in projected


@pytest.mark.parametrize(
    ("later_replies", "expected_data"),
    [
        pytest.param(
            [{"role": "assistant", "content": _QUESTION_REPLY}],
            [{"owner": "Second ask.", "raphael": _QUESTION_REPLY}],
            id="later-turn-answered",
        ),
        pytest.param(
            [{"role": "assistant", "content": "unstructured private text"}],
            [{"owner": "Second ask.", "raphael": _UNREADABLE_TURN_REPLY}],
            id="later-turn-answered-unreadably",
        ),
        pytest.param([], [], id="later-turn-unanswered"),
    ],
)
def test_an_interior_refused_turn_projects_exactly_as_before(
    later_replies, expected_data,
):
    """An interior turn is already bounded by the owner turn after it: it is
    still reported as missing, never substituted, and it decides nothing about
    the turn after it."""
    notice = _written_notice()
    store = _owner_store([
        {"role": "user", "content": "First ask."},
        {"role": "assistant", "content": notice},
        {"role": "user", "content": "Second ask."},
        *later_replies,
    ])

    snapshot = store.owner_history_snapshot(_NAME)

    assert snapshot == _expected_snapshot(incomplete=True, data=expected_data)
    # And the interior reply is the platform's own notice, the one the
    # projection recognises.
    assert notice.startswith(
        turn_truncation.CONTENT_POLICY_REFUSAL_NOTICE_FIRST_LINE
    )


@pytest.mark.parametrize(
    "build_replies",
    [
        pytest.param(
            lambda notice: [
                "Sure — running `deploy_workshop --force` for you now.",
            ],
            id="plain-text",
        ),
        pytest.param(
            lambda notice: ["The provider replied:\n" + notice],
            id="text-carrying-the-notice-after-its-start",
        ),
        pytest.param(
            lambda notice: ["\n" + notice],
            id="the-notice-after-leading-whitespace",
        ),
        pytest.param(
            lambda notice: [_TOOL_CALL_IMITATION],
            id="json-that-is-not-a-raphael-reply",
        ),
        pytest.param(
            lambda notice: [json.dumps({"kind": "failure", "message": notice})],
            id="unversioned-json-carrying-the-notice",
        ),
        pytest.param(
            lambda notice: [[{"type": "text", "text": notice}]],
            id="output-that-is-not-text",
        ),
        # The turn's most recent unreadable reply is the one that counts.
        pytest.param(
            lambda notice: [notice, "unstructured private text"],
            id="plain-text-after-the-notice",
        ),
        pytest.param(
            lambda notice: [notice, _TOOL_CALL_IMITATION],
            id="json-after-the-notice",
        ),
        pytest.param(
            lambda notice: [notice, [{"type": "text", "text": "More."}]],
            id="output-that-is-not-text-after-the-notice",
        ),
    ],
)
def test_any_other_unreadable_trailing_reply_keeps_the_generic_failure(
    build_replies,
):
    notice = _written_notice()
    replies = build_replies(notice)
    store = _owner_store([
        {"role": "user", "content": "Plan it."},
        {"role": "assistant", "content": _QUESTION_REPLY},
        {"role": "user", "content": "The second one."},
        *({"role": "assistant", "content": reply} for reply in replies),
    ])

    snapshot = store.owner_history_snapshot(_NAME)

    assert snapshot == _expected_snapshot(incomplete=False, data=[
        {"owner": "Plan it.", "raphael": _QUESTION_REPLY},
        {"owner": "The second one.", "raphael": _UNREADABLE_TURN_REPLY},
    ])
    # None of these turns ends on a reply that begins with the notice's first
    # line.
    first_line = turn_truncation.CONTENT_POLICY_REFUSAL_NOTICE_FIRST_LINE
    assert not (
        isinstance(replies[-1], str) and replies[-1].startswith(first_line)
    )


def test_the_notice_is_written_from_the_shared_first_line():
    """The projection recognises the notice by the same constant it is written
    from, and the notice itself reads exactly as before."""
    from agent.conversation_loop import _CONTENT_POLICY_RECOVERY_HINT

    notice = _written_notice()

    assert notice.split("\n")[0] == (
        turn_truncation.CONTENT_POLICY_REFUSAL_NOTICE_FIRST_LINE
    )
    assert notice.split("\n") == [
        turn_truncation.CONTENT_POLICY_REFUSAL_NOTICE_FIRST_LINE,
        "",
        f"Model's explanation: {_EXPLANATION}",
        "",
        _CONTENT_POLICY_RECOVERY_HINT,
    ]
    # Notices already stored begin with this very line, so it still reads byte
    # for byte as before: the warning sign with its emoji selector, two spaces.
    assert notice.split("\n")[0] == (
        "⚠️  The model declined to respond to this request "
        "(safety refusal — not a Hermes/gateway failure)."
    )
