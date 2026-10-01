"""Owner history of a turn the AI provider never answered.

When the agent's retries end on a provider failure, the turn's final text
starts with the platform's own fixed words "API call failed after"
(``agent.turn_recovery``). Sending the request again unchanged right away
would only fail again, so a trailing owner turn that ends on those words — and
a recovery sealed from a failed turn whose own stored reply starts with them —
says that the provider did not answer, instead of the generic interrupted
failure. The refusal notice, every other unreadable reply, every interior turn
and every other sealed failure read exactly as before.

Every stored reply and error here is a placeholder: no real provider reply and
no real error detail.
"""

import json

import pytest

from agent.turn_truncation import CONTENT_POLICY_REFUSAL_NOTICE_FIRST_LINE
from gateway.platforms import api_server
from gateway.platforms.api_server import ResponseStore

_NAME = "raphael-owner-" + "8" * 32
_RESPONSE_ID = "resp_owner_turn"
_EARLIER_RESPONSE_ID = "resp_earlier_turn"
_FAILED_RESPONSE_ID = "resp_failed_turn"
_OWNER_REQUEST = "Placeholder owner request."

_PLACEHOLDER_DETAIL = "placeholder provider detail"
_PROVIDER_FAILURE_REPLY = "API call failed after 3 retries: " + _PLACEHOLDER_DETAIL
_REFUSAL_NOTICE = (
    CONTENT_POLICY_REFUSAL_NOTICE_FIRST_LINE + "\n\nPlaceholder explanation."
)
_TOOL_CALL_IMITATION = json.dumps({
    "tool": "placeholder_tool",
    "arguments": {"placeholder": "value"},
})

# The new message's exact text, as the analysis fixes it.
_PROVIDER_UNAVAILABLE_TEXT = (
    "The AI provider did not answer, so nothing was prepared and nothing was "
    "changed. You can send it again in a few minutes."
)


def _failure_reply(message):
    return json.dumps(
        {"schema_version": 1, "kind": "failure", "message": message},
        ensure_ascii=False,
    )


_QUESTION_REPLY = (
    '{"schema_version": 1, "kind": "question", "message": "Placeholder question?"}'
)
_INTERRUPTED_TURN_REPLY = _failure_reply(
    "Raphael could not prepare a safe plan for this request. Nothing was "
    "changed. You can send it again."
)
_REFUSED_TURN_REPLY = _failure_reply(
    "Raphael could not work on this request as it is worded. Nothing was "
    "changed. Rewording the request may help."
)
_PROVIDER_UNAVAILABLE_TURN_REPLY = _failure_reply(_PROVIDER_UNAVAILABLE_TEXT)


def _new_message():
    """Read through the module, so a missing constant fails only its checks."""
    return api_server._OWNER_PROVIDER_UNAVAILABLE_TURN_MESSAGE


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


def _trailing_turn_store(replies):
    return _owner_store([
        {"role": "user", "content": "Placeholder first request."},
        {"role": "assistant", "content": _QUESTION_REPLY},
        {"role": "user", "content": "Placeholder second request."},
        *({"role": "assistant", "content": reply} for reply in replies),
    ])


def _trailing_turn_snapshot(raphael):
    return _expected_snapshot(incomplete=False, data=[
        {"owner": "Placeholder first request.", "raphael": _QUESTION_REPLY},
        {"owner": "Placeholder second request.", "raphael": raphael},
    ])


@pytest.mark.parametrize(
    "replies",
    [
        pytest.param([_PROVIDER_FAILURE_REPLY], id="the-words-alone"),
        pytest.param(
            [_PROVIDER_FAILURE_REPLY + "\n\nPlaceholder guidance."],
            id="the-words-with-guidance-after-them",
        ),
        # The turn's most recent unreadable reply is the one that counts.
        pytest.param(
            ["Placeholder raw text.", _PROVIDER_FAILURE_REPLY],
            id="the-words-after-other-raw-text",
        ),
        pytest.param(
            [_REFUSAL_NOTICE, _PROVIDER_FAILURE_REPLY],
            id="the-words-after-the-refusal-notice",
        ),
    ],
)
def test_a_trailing_provider_failure_reads_as_provider_unavailable(replies):
    store = _trailing_turn_store(replies)
    stored_before = _stored_state(store)

    snapshot = store.owner_history_snapshot(_NAME)

    assert snapshot == _trailing_turn_snapshot(_PROVIDER_UNAVAILABLE_TURN_REPLY)
    # Only the projection changed: the stored row still holds what it held.
    assert _stored_state(store) == stored_before
    # Neither the stored words nor any detail after them reaches the owner.
    projected = json.dumps(snapshot, ensure_ascii=False)
    for part in (_PLACEHOLDER_DETAIL, "API call failed", "retries"):
        assert part not in projected


@pytest.mark.parametrize(
    "replies",
    [
        pytest.param([_REFUSAL_NOTICE], id="the-notice-alone"),
        pytest.param(
            [_PROVIDER_FAILURE_REPLY, _REFUSAL_NOTICE],
            id="the-notice-after-the-words",
        ),
    ],
)
def test_a_trailing_refusal_notice_still_reads_as_refused(replies):
    store = _trailing_turn_store(replies)

    snapshot = store.owner_history_snapshot(_NAME)

    assert snapshot == _trailing_turn_snapshot(_REFUSED_TURN_REPLY)
    assert _new_message() not in json.dumps(snapshot, ensure_ascii=False)


@pytest.mark.parametrize(
    "replies",
    [
        pytest.param(["Placeholder raw text."], id="plain-text"),
        pytest.param(
            ["\n" + _PROVIDER_FAILURE_REPLY],
            id="the-words-after-leading-whitespace",
        ),
        pytest.param(
            ["Placeholder: " + _PROVIDER_FAILURE_REPLY],
            id="the-words-after-the-start",
        ),
        pytest.param(
            [_PROVIDER_FAILURE_REPLY.lower()], id="the-words-in-other-case",
        ),
        pytest.param(
            ["API call failed: " + _PLACEHOLDER_DETAIL],
            id="a-shorter-start",
        ),
        pytest.param([_TOOL_CALL_IMITATION], id="json-that-is-not-a-raphael-reply"),
        pytest.param(
            [json.dumps({"kind": "failure", "message": _PROVIDER_FAILURE_REPLY})],
            id="unversioned-json-carrying-the-words",
        ),
        pytest.param(
            [[{"type": "text", "text": _PROVIDER_FAILURE_REPLY}]],
            id="output-that-is-not-text",
        ),
        pytest.param(
            [_PROVIDER_FAILURE_REPLY, "Placeholder raw text."],
            id="plain-text-after-the-words",
        ),
        pytest.param(
            [_PROVIDER_FAILURE_REPLY, _TOOL_CALL_IMITATION],
            id="json-after-the-words",
        ),
        pytest.param(
            [_PROVIDER_FAILURE_REPLY, [{"type": "text", "text": "Placeholder."}]],
            id="output-that-is-not-text-after-the-words",
        ),
    ],
)
def test_other_raw_text_still_reads_as_interrupted(replies):
    store = _trailing_turn_store(replies)

    snapshot = store.owner_history_snapshot(_NAME)

    assert snapshot == _trailing_turn_snapshot(_INTERRUPTED_TURN_REPLY)
    assert _new_message() not in json.dumps(snapshot, ensure_ascii=False)


@pytest.mark.parametrize(
    ("later_replies", "expected_data"),
    [
        pytest.param(
            [{"role": "assistant", "content": _QUESTION_REPLY}],
            [{"owner": "Placeholder second request.", "raphael": _QUESTION_REPLY}],
            id="later-turn-answered",
        ),
        pytest.param(
            [{"role": "assistant", "content": "Placeholder raw text."}],
            [{
                "owner": "Placeholder second request.",
                "raphael": _INTERRUPTED_TURN_REPLY,
            }],
            id="later-turn-answered-unreadably",
        ),
        pytest.param([], [], id="later-turn-unanswered"),
    ],
)
def test_an_interior_provider_failure_turn_reads_exactly_as_before(
    later_replies, expected_data,
):
    """An interior turn is never substituted, and its words decide nothing
    about the turn after it."""
    store = _owner_store([
        {"role": "user", "content": "Placeholder first request."},
        {"role": "assistant", "content": _PROVIDER_FAILURE_REPLY},
        {"role": "user", "content": "Placeholder second request."},
        *later_replies,
    ])

    snapshot = store.owner_history_snapshot(_NAME)

    assert snapshot == _expected_snapshot(incomplete=True, data=expected_data)
    assert _new_message() not in json.dumps(snapshot, ensure_ascii=False)


def _sealed_snapshot(earlier, failed_history):
    """Seal one failed owner turn through the production entry points."""
    store = ResponseStore(max_size=10)
    if earlier:
        store.put(_EARLIER_RESPONSE_ID, {
            "response": {"id": _EARLIER_RESPONSE_ID},
            "conversation_history": earlier,
        })
        assert store.set_conversation(_NAME, _EARLIER_RESPONSE_ID) is True
    assert store.reserve_owner_conversation(
        "default", _NAME, _FAILED_RESPONSE_ID, owner_message=_OWNER_REQUEST,
    ) is True
    store.store_terminal_owner_response(
        profile="default",
        response_id=_FAILED_RESPONSE_ID,
        data={
            "response": {
                "id": _FAILED_RESPONSE_ID,
                "object": "response",
                "status": "failed",
                "output": [],
                "error": {"code": "server_error", "message": "placeholder error"},
            },
            "conversation_history": failed_history,
        },
        release_job=True,
        conversation=_NAME,
        interrupted=True,
    )
    assert store.acknowledge_owner_conversation_recovery(
        "default", _NAME, _FAILED_RESPONSE_ID,
    ) == "retired"
    snapshot = store.owner_history_snapshot(_NAME)
    store.close()
    assert snapshot["head_response_id"] == _FAILED_RESPONSE_ID
    assert snapshot["recovery"] is None
    return snapshot


_EARLIER_ANSWERED = [
    {"role": "user", "content": "Placeholder first request."},
    {"role": "assistant", "content": _QUESTION_REPLY},
]


def _own_turn(reply):
    return [
        {"role": "user", "content": _OWNER_REQUEST},
        {"role": "assistant", "content": reply},
    ]


@pytest.mark.parametrize(
    "earlier",
    [
        pytest.param([], id="first-turn"),
        pytest.param(_EARLIER_ANSWERED, id="after-an-answered-turn"),
    ],
)
def test_a_sealed_failed_turn_whose_own_reply_starts_with_the_words_reads_as_provider_unavailable(
    earlier,
):
    snapshot = _sealed_snapshot(
        earlier, [*earlier, *_own_turn(_PROVIDER_FAILURE_REPLY)],
    )

    assert snapshot["data"][-1] == {
        "owner": _OWNER_REQUEST,
        "raphael": _PROVIDER_UNAVAILABLE_TURN_REPLY,
    }
    projected = json.dumps(snapshot, ensure_ascii=False)
    for part in (_PLACEHOLDER_DETAIL, "placeholder error", "API call failed"):
        assert part not in projected


@pytest.mark.parametrize(
    ("earlier", "failed_history"),
    [
        pytest.param(_EARLIER_ANSWERED, [], id="no-reply-in-the-record"),
        pytest.param(
            _EARLIER_ANSWERED, list(_EARLIER_ANSWERED),
            id="only-the-inherited-transcript",
        ),
        # The record of a later turn carries the conversation's earlier
        # transcript; words an EARLIER turn ended on are not this turn's own,
        # even when the owner sent the very same request again.
        pytest.param(
            _own_turn(_PROVIDER_FAILURE_REPLY),
            _own_turn(_PROVIDER_FAILURE_REPLY),
            id="the-words-only-in-an-earlier-turn",
        ),
        # A record that does not continue the conversation's transcript cannot
        # show which of its words are this turn's own.
        pytest.param(
            _EARLIER_ANSWERED,
            [_EARLIER_ANSWERED[0], *_own_turn(_PROVIDER_FAILURE_REPLY)[1:]],
            id="the-words-in-a-transcript-that-does-not-continue-the-conversation",
        ),
        pytest.param(
            _EARLIER_ANSWERED,
            [*_EARLIER_ANSWERED, *_own_turn("Placeholder raw text.")],
            id="own-reply-is-other-raw-text",
        ),
        pytest.param(
            _EARLIER_ANSWERED,
            [*_EARLIER_ANSWERED, *_own_turn("\n" + _PROVIDER_FAILURE_REPLY)],
            id="own-reply-has-the-words-after-its-start",
        ),
        pytest.param(
            _EARLIER_ANSWERED,
            [
                *_EARLIER_ANSWERED,
                *_own_turn([{"type": "text", "text": _PROVIDER_FAILURE_REPLY}]),
            ],
            id="own-reply-is-not-text",
        ),
        pytest.param(
            _EARLIER_ANSWERED,
            [
                *_EARLIER_ANSWERED,
                *_own_turn(_PROVIDER_FAILURE_REPLY),
                {"role": "assistant", "content": "Placeholder raw text."},
            ],
            id="own-reply-after-the-words",
        ),
    ],
)
def test_a_sealed_failed_turn_without_the_words_keeps_the_interrupted_message(
    earlier, failed_history,
):
    snapshot = _sealed_snapshot(earlier, failed_history)

    assert snapshot["data"][-1] == {
        "owner": _OWNER_REQUEST,
        "raphael": _INTERRUPTED_TURN_REPLY,
    }
    assert _new_message() not in json.dumps(snapshot, ensure_ascii=False)


def test_the_new_message_reads_exactly_and_fits_the_owner_page():
    """The owner page accepts a failure message of at most 240 characters."""
    message = _new_message()

    assert message == _PROVIDER_UNAVAILABLE_TEXT
    assert len(message) <= 240
