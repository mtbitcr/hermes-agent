"""A running owner turn keeps its response when other responses are written.

ResponseStore trims ordinary responses to its cap after every write. It has
always kept the row being written and each owner conversation's current head,
but it counted every row when deciding how many to evict, so a profile holding
more owner heads than the cap evicted every other response on any write. That
included the queued or in-progress response of an owner turn still running:
GET /v1/responses/{response_id} answered 404 and the owner page reported the
request as expired, although the turn went on to publish its proposal.

A live row in owner_conversation_reservations (expires_at in the future) holds
the response it names, so eviction keeps it; the number to evict is taken over
the ordinary responses alone, so the cap still bounds them. "Readable" below
means ResponseStore.get, the read behind GET /v1/responses/{response_id}.
"""

import time

from gateway.platforms.api_server import ResponseStore

_PROFILE = "default"


def _conversation(index):
    return "raphael-owner-" + format(index, "032x")


def _response(response_id, status="completed"):
    return {
        "response": {"id": response_id, "object": "response", "status": status},
        "conversation_history": [{"role": "user", "content": "Plan the launch"}],
    }


def _store(tmp_path, max_size):
    return ResponseStore(
        max_size=max_size, db_path=str(tmp_path / "response_store.db"),
    )


def _publish_owner_heads(store, count):
    """Give ``count`` owner conversations a current head, as the planner has."""
    heads = []
    for index in range(count):
        response_id = "resp_owner_head_" + format(index, "04d")
        store.put(response_id, _response(response_id), profile=_PROFILE)
        assert store.set_conversation(
            _conversation(index), response_id, profile=_PROFILE,
        ) is True
        heads.append(response_id)
    return heads


def _reserve_owner_turn(store, conversation, response_id):
    """Take the reservation an owner turn holds while it runs."""
    assert store.reserve_owner_conversation(
        _PROFILE, conversation, response_id, owner_message="Plan the launch",
    ) is True


def _readable(store, response_id):
    return store.get(response_id, profile=_PROFILE) is not None


def test_a_running_owner_turn_keeps_its_response_when_another_turn_publishes(
    tmp_path,
):
    """The incident: more owner heads than the cap, and a turn still running."""
    store = _store(tmp_path, max_size=2)
    try:
        heads = _publish_owner_heads(store, 4)
        running = "resp_running_owner_turn"
        running_conversation = _conversation(100)
        _reserve_owner_turn(store, running_conversation, running)
        # What the gateway writes for a background owner turn: the queued body
        # it accepts, then the in-progress body once the turn starts.
        store.accept_owner_background_response(
            profile=_PROFILE,
            response_id=running,
            data=_response(running, status="queued"),
            conversation=running_conversation,
        )
        store.put(
            running, _response(running, status="in_progress"), profile=_PROFILE,
        )

        # Another owner turn publishes while this one is still running.
        other = "resp_other_owner_turn"
        other_conversation = _conversation(101)
        _reserve_owner_turn(store, other_conversation, other)
        assert store.publish_owner_turn(
            profile=_PROFILE,
            conversation=other_conversation,
            response_id=other,
            data=_response(other),
            owner_proposal=False,
            reservation_id=other,
            expected_previous_response_id=None,
        ) is True

        stored = store.get(running, profile=_PROFILE)
        assert stored is not None
        assert stored["response"]["status"] == "in_progress"
        assert all(_readable(store, head) for head in heads)
        assert _readable(store, other)
    finally:
        store.close()


def test_a_held_response_is_evicted_like_any_other_once_its_reservation_expires(
    tmp_path,
):
    """Only a live reservation holds a response; an expired one holds nothing."""
    store = _store(tmp_path, max_size=1)
    try:
        heads = _publish_owner_heads(store, 2)
        held = "resp_reserved_owner_turn"
        conversation = _conversation(100)
        _reserve_owner_turn(store, conversation, held)
        store.put(held, _response(held, status="in_progress"), profile=_PROFILE)
        older = "resp_ordinary_older"
        store.put(older, _response(older), profile=_PROFILE)

        # While the reservation is live the response is kept, although the one
        # ordinary slot the cap allows is already taken.
        assert _readable(store, held)

        # The lease runs out without being renewed.
        store._conn.execute(
            "UPDATE owner_conversation_reservations SET expires_at = ? "
            "WHERE profile = ? AND name = ?",
            (time.time() - 1, _PROFILE, conversation),
        )
        store._conn.commit()
        newer = "resp_ordinary_newer"
        store.put(newer, _response(newer), profile=_PROFILE)

        # Now it is an ordinary response: the cap evicts it together with the
        # older one and keeps only the newest.
        assert not _readable(store, held)
        assert not _readable(store, older)
        assert _readable(store, newer)
        assert all(_readable(store, head) for head in heads)
    finally:
        store.close()


def test_ordinary_responses_are_still_evicted_oldest_first_down_to_the_cap(
    tmp_path,
):
    """Protected rows do not count against the cap, and it still bounds the rest."""
    store = _store(tmp_path, max_size=2)
    try:
        heads = _publish_owner_heads(store, 3)
        held = "resp_running_owner_turn"
        _reserve_owner_turn(store, _conversation(100), held)
        store.put(held, _response(held, status="in_progress"), profile=_PROFILE)
        ordinary = ["resp_ordinary_" + str(index) for index in range(5)]
        for response_id in ordinary:
            store.put(response_id, _response(response_id), profile=_PROFILE)

        # Exactly the cap's worth of ordinary responses is left: the newest.
        assert [_readable(store, response_id) for response_id in ordinary] == [
            False, False, False, True, True,
        ]
        assert all(_readable(store, head) for head in heads)
        assert _readable(store, held)
        assert len(store) == len(heads) + 1 + 2
    finally:
        store.close()
