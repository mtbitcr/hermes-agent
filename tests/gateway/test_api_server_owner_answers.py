"""The owner answers the question of a stopped card, and the card resumes.

Card P3 of the plan for how the owner answers a stopped card. The Decisions list read with ``owner_answer_v1`` serves the
question each stopped card waits on (card P2). One owner route, ``POST /v1/owner-workspace/decisions/{ref}/answer``,
takes the owner's answer to it: ``accept_answer`` records it once as the event ``owner_answer_recorded``, comments on the
card as ``owner`` and resumes the card, in one write transaction. Each test takes the key and the stop reference of its
card from the list, drives the adapter's real routes into the real kernel inside the per-test ``HERMES_HOME``, and every
count it proves is read by a second, plain connection to the card's own store.
"""

from __future__ import annotations

import asyncio
import contextlib
import copy
import json
import os
import sqlite3
import threading
import time
from types import SimpleNamespace

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import PlatformConfig
from gateway.platforms.api_server import APIServerAdapter
from hermes_cli import kanban_db as kb
from hermes_cli import owner_questions as oq
from hermes_cli import owner_workspace as ow
from hermes_cli.owner_text import check_owner_text
from hermes_constants import get_hermes_home

# The owner the routes resolve for themselves, a committed owner Project and the workspace switch are reused verbatim
# from the suggestion decisions, the helpers of the choose route from its tests, and the worker, its question and its
# stop reference from the list tests of card P2.
from tests.gateway.test_api_server_owner_choices import (
    _assert_refused,
    _designer_card,
    _feed,
    _iso,
    _owner_move_to_ready,
    _store,
)
from tests.gateway.test_api_server_owner_suggestion_decisions import (  # noqa: F401  (owner is a fixture)
    _project,
    _write_owner_workspace_config,
    owner,
)
from tests.hermes_cli.test_owner_questions_feed import (
    _CHOICES,
    _PATH,
    _PLAIN,
    _QUESTION,
    _WORKER,
    _newest_blocked,
    _stop_ref,
)

_ROUTE = "/v1/owner-workspace/decisions/{ref}/answer"
_FEED = "/v1/owner-workspace/decisions"
_API_KEY = "owner-answers-" + "5c8e2a71" * 4
_IDEMPOTENCY_KEY = "answer-" + "9b4d1f62" * 4
# The facts' sentence for a question that no longer waits (point 26).
_NOT_WAITING = "This question is no longer waiting."
_STORED_FAILURE = check_owner_text(None, "stored_failure")
_OBJECT = "hermes.owner_workspace.owner_answer"
_REVIEWER = "raphael-reviewer"
_REVIEW_SUMMARY = "The reviewer asked twice for a shorter sign up form."
_CAPABILITY = "Raphael needs a browser to check the sign up page."
_WORDS = "Keep the button green."


def _app(key: str = "") -> web.Application:
    """The adapter's own Decisions routes, the answer route among them, as ``connect()`` adds them."""
    adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={"key": key} if key else {}))
    app = web.Application()
    rows = [row for row in adapter._http_route_table() if row[1].startswith(_FEED)]
    assert [row[:2] for row in rows].count(("POST", _ROUTE)) == 1, "the answer route is not registered"
    for method, path, handler in rows:
        app.router.add_route(method, path, handler)
    return app


def _card(
    project: dict, title: str, *, stop: str | None = "question", question: str = _QUESTION, choices=None,
    words: str = "optional", kind: str = "needs_input", reason: str = _PLAIN, inputs: list | None = None,
    review: bool = False,
) -> SimpleNamespace:
    """A work card of *project* that stops for the owner: its worker's run, or with *review* the run of a reviewer that
    claimed it after its worker asked for a review, declares its own question (card P1) when *stop* is given, and else
    holds the card *kind* with *reason*. The card first declares the named *inputs* it needs, when given."""
    board = project["board"]
    with contextlib.closing(kb.connect(board=board)) as conn:
        task_id = kb.create_task(conn, title=title, assignee=_WORKER, project_id=project["project_id"])
        run = kb.claim_task(conn, task_id)
        assert run is not None and run.status == "running"
        if review:
            assert kb.request_review(
                conn, task_id, summary="ready", reviewer=_REVIEWER, expected_run_id=run.current_run_id,
            )
            run = kb.claim_review_task(conn, task_id, claimer=f"{_REVIEWER}:1")
            assert run is not None and run.status == "running"
        if inputs:
            kb._append_event(conn, task_id, "named_inputs_declared", {"inputs": inputs})
        if stop is None:
            assert kb.block_task(conn, task_id, reason=reason, kind=kind) is True
        else:
            oq.declare_owner_question(
                conn, task_id, stop, question, copy.deepcopy(_CHOICES if choices is None else choices), words,
                board=board, expected_run_id=run.current_run_id,
            )
    return SimpleNamespace(
        board=board, store=_store(board), project_id=project["project_id"], task_id=task_id, title=title,
    )


def _observed(card) -> dict:
    """The card and its board as a second, plain connection to the card's own store sees them."""
    with contextlib.closing(sqlite3.connect(card.store)) as conn:
        status, block_kind = conn.execute(
            "SELECT status, block_kind FROM tasks WHERE id = ?", (card.task_id,),
        ).fetchone()
        return {
            "status": status,
            "block_kind": block_kind,
            "answers": [json.loads(payload) for (payload,) in conn.execute(
                "SELECT payload FROM task_events WHERE task_id = ? AND kind = 'owner_answer_recorded' ORDER BY id",
                (card.task_id,),
            )],
            "comments": conn.execute(
                "SELECT author, body FROM task_comments WHERE task_id = ? ORDER BY id", (card.task_id,),
            ).fetchall(),
            "unblocked": conn.execute(
                "SELECT COUNT(*) FROM task_events WHERE task_id = ? AND kind = 'unblocked'", (card.task_id,),
            ).fetchone()[0],
            "events": conn.execute("SELECT COUNT(*) FROM task_events").fetchone()[0],
        }


async def _listed(client: TestClient, card, capabilities: str | None = "owner_answer_v1") -> dict:
    """The item the owner's own Decisions list, read with ``owner_answer_v1``, holds for *card*."""
    _status, _raw, feed = await _feed(client, capabilities)
    (item,) = [item for item in feed["data"] if item["title"] == card.title]
    return item


async def _answer(client: TestClient, ref: str, body, headers: dict | None = None) -> tuple[int, dict]:
    data = body if isinstance(body, bytes) else json.dumps(body).encode("utf-8")
    response = await client.post(f"/v1/owner-workspace/decisions/{ref}/answer", data=data, headers=headers or {})
    return response.status, await response.json()


def _answered(ref: str, stop_ref: str, choice, resumed: bool) -> tuple[int, dict]:
    return 200, {"object": _OBJECT, "decision_ref": ref, "stop_ref": stop_ref, "choice": choice, "resumed": resumed}


# ---------------------------------------------------------------------------
# a, c, d, e: one answer, one event, one comment, and the card resumed
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_35_a_c_d_e_an_answer_with_a_choice_is_recorded_once_commented_and_resumes_the_card(owner):
    project = _project(owner, "Sign up Website")
    card = _card(project, "Build the sign up page")
    other = _card(project, "Build the price page")
    headers = {"Idempotency-Key": _IDEMPOTENCY_KEY}
    before = int(time.time())

    async with TestClient(TestServer(_app())) as client:
        item, bystander = await _listed(client, card), await _listed(client, other)
        ref, stop_ref = item["decision_ref"], item["question"]["stop_ref"]
        body = {"stop_ref": stop_ref, "choice": 2, "words": f"{_WORDS}\n"}
        first = await _answer(client, ref, body, headers)
        recorded = _observed(card)
        # c. A replay with the same key and body gets the first answer and writes nothing new.
        replay = await _answer(client, ref, body, headers)
        replayed = _observed(card)
        # d. The same key with another body.
        conflict = await _answer(client, ref, {**body, "choice": 1}, headers)
        # e. A new key after the success.
        late = await _answer(client, ref, body, {"Idempotency-Key": "answer-late-" + "6a0f3e58" * 4})
        _status, _raw, feed = await _feed(client, "owner_answer_v1")
        still = await _listed(client, other)

    assert first == _answered(ref, stop_ref, 2, True)
    assert replay == first and replayed == recorded
    _assert_refused(conflict, 409, _STORED_FAILURE, "idempotency_conflict")
    _assert_refused(late, 404, _NOT_WAITING, "decision_not_found")
    assert _observed(card) == recorded
    # One event with exactly its five keys, one comment of the author owner, and the card resumed.
    (answered,) = recorded["answers"]
    assert {key: value for key, value in answered.items() if key != "answered_at"} == {
        "stop_ref": stop_ref, "choice": 2, "words_length": len(body["words"]), "actor": owner.actor,
    }
    assert _iso(before) <= answered["answered_at"] <= _iso(time.time())
    assert recorded["comments"] == [
        ("owner", f"The owner answered: {_QUESTION}\nChoice 2: {_CHOICES[1]['sentence']}\n{_WORDS}"),
    ]
    # The comment carries the operation key of the stop it answers.
    with contextlib.closing(sqlite3.connect(card.store)) as conn:
        assert conn.execute(
            "SELECT operation_key FROM task_comments WHERE task_id = ?", (card.task_id,),
        ).fetchall() == [(f"owner_answer:{stop_ref}",)]
    assert (recorded["status"], recorded["unblocked"]) == ("ready", 1)
    # It no longer waits, so the list no longer lists it; the other card's question is served as before.
    assert [entry for entry in feed["data"] if entry["title"] == card.title] == []
    assert still == bystander


@pytest.mark.asyncio
async def test_35_a_an_answer_to_a_stop_that_came_in_a_review_resumes_the_card_to_review(owner):
    project = _project(owner, "Sign up Website")
    card = _card(
        project, "Build the sign up page", stop="review_limit", question=_REVIEW_SUMMARY, choices=[], review=True,
    )

    async with TestClient(TestServer(_app())) as client:
        item = await _listed(client, card)
        question = item["question"]
        answer = await _answer(client, item["decision_ref"], {
            "stop_ref": question["stop_ref"], "choice": 1, "words": None,
        })

    seen = _observed(card)
    assert question["stop"] == "review_limit"
    assert [choice["sentence"] for choice in question["choices"]] == list(oq.REVIEW_LIMIT_CHOICES)
    assert answer == _answered(item["decision_ref"], question["stop_ref"], 1, True)
    (answered,) = seen["answers"]
    assert (answered["stop_ref"], answered["choice"], answered["words_length"]) == (question["stop_ref"], 1, 0)
    assert seen["comments"] == [
        ("owner", f"The owner answered: {_REVIEW_SUMMARY}\nChoice 1: {oq.REVIEW_LIMIT_CHOICES[0]}"),
    ]
    assert (seen["status"], seen["unblocked"]) == ("review", 1)


# ---------------------------------------------------------------------------
# b: a failure inside the transaction leaves nothing
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_35_b_a_forced_failure_after_the_comment_leaves_no_event_no_comment_and_the_card_blocked(owner):
    project = _project(owner, "Sign up Website")
    card = _card(project, "Build the sign up page")
    written = []

    def fail_after_the_comment(conn, task_id, now, **_kwargs):
        written.append(tuple(conn.execute(
            "SELECT (SELECT COUNT(*) FROM task_events WHERE task_id = ? AND kind = 'owner_answer_recorded'), "
            "(SELECT COUNT(*) FROM task_comments WHERE task_id = ?)", (task_id, task_id),
        ).fetchone()))
        raise RuntimeError(f"forced failure of {task_id} in {card.store}")

    async with TestClient(TestServer(_app())) as client:
        item = await _listed(client, card)
        body = {"stop_ref": item["question"]["stop_ref"], "choice": 1, "words": None}
        before = _observed(card)
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(kb, "_unblock_task_within_txn", fail_after_the_comment)
            failed = await _answer(client, item["decision_ref"], body)
        after = _observed(card)
        again = await _listed(client, card)
        # Control: the same answer, with nothing forced, is taken.
        control = await _answer(client, item["decision_ref"], body)

    # The event and the comment were written when the failure came, and neither stayed.
    assert written == [(1, 1)]
    _assert_refused(failed, 503, _STORED_FAILURE, "owner_workspace_unavailable")
    assert "forced" not in json.dumps(failed[1]) and card.task_id not in json.dumps(failed[1])
    assert after == before
    assert (after["status"], after["block_kind"], after["answers"], after["comments"]) == (
        "blocked", "needs_input", [], [],
    )
    assert again == item
    assert control == _answered(item["decision_ref"], item["question"]["stop_ref"], 1, True)


# ---------------------------------------------------------------------------
# f, g: a stop that ended takes no answer
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_35_f_an_old_stop_ref_after_the_card_was_resumed_and_blocked_again_is_refused(owner):
    project = _project(owner, "Sign up Website")
    card = _card(project, "Build the sign up page")

    async with TestClient(TestServer(_app())) as client:
        item = await _listed(client, card)
        ref, old = item["decision_ref"], item["question"]["stop_ref"]
        assert await _answer(client, ref, {"stop_ref": old, "choice": 1, "words": None}) == _answered(
            ref, old, 1, True,
        )
        with contextlib.closing(kb.connect(board=card.board)) as conn:
            assert kb.claim_task(conn, card.task_id) is not None
            assert kb.block_task(conn, card.task_id, reason=_PLAIN, kind="needs_input")
        held = _observed(card)
        listed = await _listed(client, card)
        stale = await _answer(client, ref, {"stop_ref": old, "choice": None, "words": _WORDS})

    # The new stop is listed under the same key, with a stop reference of its own.
    assert listed["decision_ref"] == ref and listed["question"]["stop_ref"] != old
    assert (listed["question"]["choices"], listed["question"]["words"]) == ([], "required")
    _assert_refused(stale, 404, _NOT_WAITING, "decision_not_found")
    assert _observed(card) == held
    assert [answer["stop_ref"] for answer in held["answers"]] == [old]


@pytest.mark.asyncio
@pytest.mark.parametrize("release", ["unblock_task", "owner_move"])
async def test_35_g_a_card_a_person_resumed_first_is_refused_and_nothing_changes(owner, release):
    project = _project(owner, "Sign up Website")
    card = _card(project, "Build the sign up page")

    async with TestClient(TestServer(_app())) as client:
        item = await _listed(client, card)
        if release == "unblock_task":
            with contextlib.closing(kb.connect(board=card.board)) as conn:
                assert kb.unblock_task(conn, card.task_id)
        else:
            _owner_move_to_ready(owner, card)
        resumed = _observed(card)
        late = await _answer(client, item["decision_ref"], {
            "stop_ref": item["question"]["stop_ref"], "choice": 1, "words": None,
        })

    _assert_refused(late, 404, _NOT_WAITING, "decision_not_found")
    assert _observed(card) == resumed
    assert (resumed["status"], resumed["answers"], resumed["comments"]) == ("ready", [], [])


# ---------------------------------------------------------------------------
# h, i: the form of an answer and the body
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("choices, words, accepted, refused, not_offered", [
    # No choices, words required: the words must be given; every choice is one the question does not offer.
    ([], "required", {"choice": None, "words": _WORDS}, [{"choice": None, "words": None}], 1),
    # Choices, words optional: a choice must be given, the words can be null.
    (_CHOICES, "optional", {"choice": 2, "words": None}, [
        {"choice": None, "words": _WORDS}, {"choice": None, "words": None},
    ], 3),
    # Choices, words required: the choice can be null, the words must be given.
    (_CHOICES, "required", {"choice": None, "words": _WORDS}, [
        {"choice": 2, "words": None}, {"choice": None, "words": None},
    ], 3),
], ids=["no_choices_words_required", "choices_words_optional", "choices_words_required"])
async def test_35_h_each_row_of_the_form_table_takes_one_answer_and_refuses_each_other_form(
    owner, choices, words, accepted, refused, not_offered,
):
    project = _project(owner, "Sign up Website")
    card = _card(project, "Build the sign up page", choices=choices, words=words)

    async with TestClient(TestServer(_app())) as client:
        item = await _listed(client, card)
        ref, stop_ref = item["decision_ref"], item["question"]["stop_ref"]
        before = _observed(card)
        refusals = [await _answer(client, ref, {"stop_ref": stop_ref, **form}) for form in refused]
        missing = await _answer(client, ref, {"stop_ref": stop_ref, "choice": not_offered, "words": _WORDS})
        unchanged = _observed(card)
        taken = await _answer(client, ref, {"stop_ref": stop_ref, **accepted})

    assert (item["question"]["choices"], item["question"]["words"]) == (choices, words)
    for refusal in refusals:
        _assert_refused(refusal, 400, _STORED_FAILURE, "invalid_argument")
    _assert_refused(missing, 404, _NOT_WAITING, "decision_not_found")
    assert unchanged == before
    assert taken == _answered(ref, stop_ref, accepted["choice"], True)
    seen = _observed(card)
    (answered,) = seen["answers"]
    assert (answered["choice"], answered["words_length"]) == (accepted["choice"], len(accepted["words"] or ""))
    lines = [f"The owner answered: {_QUESTION}"]
    if accepted["choice"] is not None:
        lines.append(f"Choice {accepted['choice']}: {choices[accepted['choice'] - 1]['sentence']}")
    if accepted["words"] is not None:
        lines.append(accepted["words"])
    assert seen["comments"] == [("owner", "\n".join(lines))]
    assert (seen["status"], seen["unblocked"]) == ("ready", 1)


@pytest.mark.asyncio
async def test_35_i_each_refused_body_gets_400_and_the_card_is_not_looked_up(owner, monkeypatch):
    from gateway.platforms import api_server_owner_answers

    project = _project(owner, "Sign up Website")
    card = _card(project, "Build the sign up page")
    looked_up = []
    monkeypatch.setattr(
        api_server_owner_answers, "_waiting_card",
        lambda owner, ref, real=api_server_owner_answers._waiting_card: looked_up.append(ref) or real(owner, ref),
    )

    async with TestClient(TestServer(_app())) as client:
        item = await _listed(client, card)
        ref, stop_ref = item["decision_ref"], item["question"]["stop_ref"]
        good = {"stop_ref": stop_ref, "choice": 1, "words": _WORDS}
        before = _observed(card)
        bodies = [
            # Not one JSON object.
            b"", b"not json", b"{", b"\xff\xfe\x00", b"[" * 100_000, b"null", b'"answer"', b"1",
            json.dumps([stop_ref, 1, _WORDS]).encode("utf-8"),
            # Not exactly the three keys.
            {}, {"stop_ref": stop_ref, "choice": 1}, {"choice": 1, "words": _WORDS}, {**good, "note": _WORDS},
            # The stop reference.
            {**good, "stop_ref": None}, {**good, "stop_ref": 7}, {**good, "stop_ref": stop_ref.upper()},
            {**good, "stop_ref": stop_ref[:-1]}, {**good, "stop_ref": stop_ref + "0"},
            {**good, "stop_ref": stop_ref + "\n"}, {**good, "stop_ref": "stop-" + stop_ref[5:]},
            {**good, "stop_ref": "stop_" + "g" * 32},
            # The choice.
            *({**good, "choice": choice} for choice in (0, 5, -1, True, False, 1.0, "1", [1], 10 ** 30)),
            # The words.
            *({**good, "words": words} for words in (
                "", "   ", "\n\n", "　", "a" * 1001, "Keep\tit", "Keep it\r\n", "Keep it\x00", "Keep it\x7f",
                "Keep it\x85", 7, [_WORDS], {"words": _WORDS},
            )),
            # Neither a choice nor words.
            {**good, "choice": None, "words": None},
            # Larger than the server reads.
            {**good, "words": "a" * (1024 * 1024)},
        ]
        # One member given twice: a refused value, then a valid one; the other way round; and one valid value twice.
        repeats = [
            (json.dumps({**good, name: first})[:-1] + f", {json.dumps(name)}: {json.dumps(last)}}}").encode("utf-8")
            for name, wrong in (("stop_ref", stop_ref.upper()), ("choice", True), ("words", "Keep it\x00"))
            for first, last in ((wrong, good[name]), (good[name], wrong), (good[name], good[name]))
        ]
        refused = [await _answer(client, ref, body) for body in bodies]
        # The header is checked first, and a refused key never reaches the card either.
        bad_key = await _answer(client, ref, good, {"Idempotency-Key": "not a key"})
        # The repeated members are sent with a key of their own, and the cache keeps no answer for it.
        headers = {"Idempotency-Key": "answer-repeated-" + "c3e8a571" * 4}
        repeated = [await _answer(client, ref, body, headers) for body in repeats]
        unchanged = _observed(card)
        refused_lookups = list(looked_up)
        # Control: words of exactly 1000 code points, none of them in the basic plane, are taken, under that same key.
        taken = await _answer(client, ref, {**good, "words": "\U0001F600" * 1000}, headers)

    for answer in [*refused, bad_key, *repeated]:
        _assert_refused(answer, 400, _STORED_FAILURE, "invalid_argument")
    assert refused_lookups == [] and unchanged == before
    assert taken == _answered(ref, stop_ref, 1, True) and looked_up == [ref]
    (answered,) = _observed(card)["answers"]
    assert answered["words_length"] == 1000


# ---------------------------------------------------------------------------
# j, k: keys and authority
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_35_j_an_unknown_key_and_a_key_of_another_owner_are_refused_and_change_nothing(owner):
    project = _project(owner, "Sign up Website")
    card = _card(project, "Build the sign up page")
    stranger = ow.OwnerContext(actor="stranger", profile="stranger", session="")
    foreign = _card(_project(stranger, "Stranger Website"), "Build the stranger's page")
    # A key and a stop reference the stranger's own list shows: real, but not this owner's to answer.
    [foreign_item] = ow.list_owner_decisions(stranger, owner_answer=True)["data"]
    before = (_observed(card), _observed(foreign))

    async with TestClient(TestServer(_app())) as client:
        item = await _listed(client, card)
        own = item["question"]["stop_ref"]
        answers = [
            await _answer(client, key, {"stop_ref": stop_ref, "choice": 1, "words": None})
            for key, stop_ref in (
                ("decision_" + "0" * 32, own),
                (foreign_item["decision_ref"], foreign_item["question"]["stop_ref"]),
                (foreign_item["decision_ref"], own),
                # A native id is never a key, and a stop reference the card never had is never its stop.
                (card.task_id, own),
                (item["decision_ref"], "stop_" + "0" * 32),
                (item["decision_ref"], foreign_item["question"]["stop_ref"]),
            )
        ]

    for answer in answers:
        _assert_refused(answer, 404, _NOT_WAITING, "decision_not_found")
    assert (_observed(card), _observed(foreign)) == before


@pytest.mark.asyncio
async def test_35_k_without_owner_auth_401_and_with_the_workspace_off_404_and_nothing_changes(owner):
    project = _project(owner, "Sign up Website")
    card = _card(project, "Build the sign up page")
    (item,) = ow.list_owner_decisions(owner, owner_answer=True)["data"]
    body = {"stop_ref": item["question"]["stop_ref"], "choice": 1, "words": None}
    owner_auth = {"Authorization": f"Bearer {_API_KEY}"}
    before = _observed(card)

    async with TestClient(TestServer(_app(_API_KEY))) as client:
        missing = await _answer(client, item["decision_ref"], body)
        wrong = await _answer(client, item["decision_ref"], body, {"Authorization": "Bearer not-the-owner"})
        # The authority comes before the body.
        unread = await _answer(client, item["decision_ref"], b"not json")
        _write_owner_workspace_config(enabled=False)
        disabled = await _answer(client, item["decision_ref"], body, owner_auth)
        disabled_unread = await _answer(client, item["decision_ref"], b"not json", owner_auth)
        unchanged = _observed(card)
        _write_owner_workspace_config(enabled=True)
        # Control: the same request with the owner's authority, the workspace on, records the answer.
        control = await _answer(client, item["decision_ref"], body, owner_auth)

    for answer in (missing, wrong, unread):
        _assert_refused(answer, 401, _STORED_FAILURE, "gateway_auth_failed")
    for answer in (disabled, disabled_unread):
        _assert_refused(answer, 404, _STORED_FAILURE, "owner_workspace_not_enabled")
    assert unchanged == before
    assert control == _answered(item["decision_ref"], item["question"]["stop_ref"], 1, True)
    assert [answer["choice"] for answer in _observed(card)["answers"]] == [1]


# ---------------------------------------------------------------------------
# l, m: two writers, and a card parked for a new approval
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_35_l_two_answers_for_one_stop_in_two_threads_resume_the_card_once(owner, monkeypatch):
    from gateway.platforms import api_server_owner_answers

    project = _project(owner, "Sign up Website")
    card = _card(project, "Build the sign up page")
    # Both threads reach accept_answer together, and the first to hold the write transaction keeps it a moment, so
    # the second waits for it and then reads the card the first one left.
    together, threads = threading.Barrier(2, timeout=10), set()
    real_accept, real_unblock = api_server_owner_answers.accept_answer, kb._unblock_task_within_txn

    def accept(*args, **kwargs):
        threads.add(threading.get_ident())
        together.wait()
        return real_accept(*args, **kwargs)

    def unblock(*args, **kwargs):
        time.sleep(0.3)
        return real_unblock(*args, **kwargs)

    monkeypatch.setattr(api_server_owner_answers, "accept_answer", accept)
    monkeypatch.setattr(kb, "_unblock_task_within_txn", unblock)

    async with TestClient(TestServer(_app())) as client:
        item = await _listed(client, card)
        ref, stop_ref = item["decision_ref"], item["question"]["stop_ref"]
        answers = await asyncio.gather(
            _answer(client, ref, {"stop_ref": stop_ref, "choice": 1, "words": None}),
            _answer(client, ref, {"stop_ref": stop_ref, "choice": 2, "words": _WORDS}),
        )

    seen = _observed(card)
    assert len(threads) == 2
    (taken,) = [answer for answer in answers if answer[0] == 200]
    (refused,) = [answer for answer in answers if answer[0] != 200]
    assert taken == _answered(ref, stop_ref, taken[1]["choice"], True)
    _assert_refused(refused, 404, _NOT_WAITING, "decision_not_found")
    (answered,) = seen["answers"]
    assert answered["choice"] == taken[1]["choice"]
    assert len(seen["comments"]) == 1 and (seen["status"], seen["unblocked"]) == ("ready", 1)


@pytest.mark.asyncio
async def test_35_m_a_card_parked_for_a_new_approval_keeps_the_answer_and_takes_no_second_one(owner):
    project = _project(owner, "Sign up Website")
    card = _card(project, "Build the sign up page")

    async with TestClient(TestServer(_app())) as client:
        item = await _listed(client, card)
        ref, stop_ref = item["decision_ref"], item["question"]["stop_ref"]
        # The card's model route lock no longer holds, so its resume is parked for a new approval (point 9).
        with contextlib.closing(kb.connect(board=card.board)) as conn:
            with kb.write_txn(conn):
                conn.execute("UPDATE tasks SET model_policy_lock = 'not-a-real-lock' WHERE id = ?", (card.task_id,))
        body = {"stop_ref": stop_ref, "choice": 1, "words": None}
        first = await _answer(client, ref, body)
        parked = _observed(card)
        second = await _answer(client, ref, body)
        fresh = await _answer(client, ref, {**body, "choice": 2}, {"Idempotency-Key": "answer-parked-" + "4e7b0c93" * 4})
        listed = await _listed(client, card)
        today = await _listed(client, card, None)

    assert first == _answered(ref, stop_ref, 1, False)
    (answered,) = parked["answers"]
    assert (answered["stop_ref"], answered["choice"]) == (stop_ref, 1)
    assert parked["comments"] == [("owner", f"The owner answered: {_QUESTION}\nChoice 1: {_CHOICES[0]['sentence']}")]
    assert (parked["status"], parked["block_kind"], parked["unblocked"]) == ("blocked", "needs_input", 0)
    for answer in (second, fresh):
        _assert_refused(answer, 404, _NOT_WAITING, "decision_not_found")
    assert _observed(card) == parked
    # The list serves no question for the card: its item is today's item.
    assert listed["decision_ref"] == ref and "question" not in listed
    assert listed == today


@pytest.mark.asyncio
async def test_35_m_a_parked_card_promoted_by_hand_serves_its_next_question_and_takes_its_answer(owner):
    project = _project(owner, "Sign up Website")
    card = _card(project, "Build the sign up page")
    question = "Which colour should the sign up button have?"

    async with TestClient(TestServer(_app())) as client:
        item = await _listed(client, card)
        ref, old = item["decision_ref"], item["question"]["stop_ref"]
        # The first answer's resume is parked for a new approval, as in 35m.
        with contextlib.closing(kb.connect(board=card.board)) as conn:
            (lock,) = conn.execute("SELECT model_policy_lock FROM tasks WHERE id = ?", (card.task_id,)).fetchone()
            with kb.write_txn(conn):
                conn.execute("UPDATE tasks SET model_policy_lock = 'not-a-real-lock' WHERE id = ?", (card.task_id,))
        first = await _answer(client, ref, {"stop_ref": old, "choice": 1, "words": None})
        # The route is approved again, the operator promotes the card by hand (``promoted_manual``), and its worker
        # claims it and declares a new question.
        with contextlib.closing(kb.connect(board=card.board)) as conn:
            with kb.write_txn(conn):
                conn.execute("UPDATE tasks SET model_policy_lock = ? WHERE id = ?", (lock, card.task_id))
            assert kb.promote_task(conn, card.task_id, actor="operator") == (True, None)
            run = kb.claim_task(conn, card.task_id)
            assert run is not None and run.status == "running"
            oq.declare_owner_question(
                conn, card.task_id, "question", question, copy.deepcopy(_CHOICES), "optional",
                board=card.board, expected_run_id=run.current_run_id,
            )
            new = _stop_ref(owner, card.task_id, _newest_blocked(conn, card.task_id))
        held = _observed(card)
        listed = await _listed(client, card)
        stale = await _answer(client, ref, {"stop_ref": old, "choice": 1, "words": None})
        refused = _observed(card)
        taken = await _answer(client, ref, {"stop_ref": new, "choice": 2, "words": None})

    assert first == _answered(ref, old, 1, False)
    # The new stop is served under the same key, with a stop reference of its own.
    assert new != old and listed["decision_ref"] == ref
    assert listed.get("question") == {
        "object": oq.OBJECT, "stop": "question", "stop_ref": new, "text": question, "choices": _CHOICES,
        "words": "optional", "words_limit": oq.WORDS_LIMIT,
    }
    # The old stop still takes no answer, and nothing is written.
    _assert_refused(stale, 404, _NOT_WAITING, "decision_not_found")
    assert refused == held
    # The new stop takes its answer and the card resumes: one answer event for each of the two stops.
    assert taken == _answered(ref, new, 2, True)
    done = _observed(card)
    assert [(answer["stop_ref"], answer["choice"]) for answer in done["answers"]] == [(old, 1), (new, 2)]
    assert done["comments"] == [
        ("owner", f"The owner answered: {_QUESTION}\nChoice 1: {_CHOICES[0]['sentence']}"),
        ("owner", f"The owner answered: {question}\nChoice 2: {_CHOICES[1]['sentence']}"),
    ]
    assert (done["status"], done["unblocked"]) == ("ready", 1)


@pytest.mark.asyncio
async def test_35_m_a_parked_card_parked_again_without_a_release_has_a_new_stop(owner):
    project = _project(owner, "Sign up Website")
    card = _card(project, "Build the sign up page")

    async with TestClient(TestServer(_app())) as client:
        item = await _listed(client, card)
        ref, old = item["decision_ref"], item["question"]["stop_ref"]
        with contextlib.closing(kb.connect(board=card.board)) as conn:
            with kb.write_txn(conn):
                conn.execute("UPDATE tasks SET model_policy_lock = 'not-a-real-lock' WHERE id = ?", (card.task_id,))
        first = await _answer(client, ref, {"stop_ref": old, "choice": 1, "words": None})
        parked = await _listed(client, card)
        # A person promotes the card while its route still cannot be confirmed: the kernel parks it again, with a new
        # blocked event and no release.
        with contextlib.closing(kb.connect(board=card.board)) as conn:
            promoted, _reason = kb.promote_task(conn, card.task_id, actor="operator")
            new = _stop_ref(owner, card.task_id, _newest_blocked(conn, card.task_id))
        held = _observed(card)
        listed = await _listed(client, card)
        stale = await _answer(client, ref, {"stop_ref": old, "choice": 1, "words": None})
        refused = _observed(card)
        taken = await _answer(client, ref, {"stop_ref": new, "choice": None, "words": _WORDS})

    assert first == _answered(ref, old, 1, False) and "question" not in parked
    assert promoted is False
    assert (held["status"], held["block_kind"], held["unblocked"]) == ("blocked", "needs_input", 0)
    # The later block is a new stop: the list serves it as a plain stop, with a stop reference of its own.
    assert new != old and listed.get("question", {}).get("stop_ref") == new
    assert (listed["question"]["choices"], listed["question"]["words"]) == ([], "required")
    _assert_refused(stale, 404, _NOT_WAITING, "decision_not_found")
    assert refused == held
    # It takes its answer; the route still cannot be confirmed, so the card is parked once more.
    assert taken == _answered(ref, new, None, False)
    assert [answer["stop_ref"] for answer in _observed(card)["answers"]] == [old, new]


# ---------------------------------------------------------------------------
# n, p: which stops take an answer, and the text of the comment
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_35_n_a_capability_stop_takes_an_answer_and_cards_with_inputs_or_options_to_give_do_not(owner):
    project = _project(owner, "Sign up Website")
    capability = _card(project, "Check the sign up page", stop=None, kind="capability", reason=_CAPABILITY)
    needs_file = _card(project, "Load the price list", stop=None, inputs=[
        {"n": 1, "name": "price list", "type": "csv", "file": "price-list.csv"},
    ])
    designer = _designer_card(project, title="Design the sign up page")
    with contextlib.closing(kb.connect(board=project["board"])) as conn:
        # The stop reference each of the two would have, made as the list makes it.
        stop_refs = {card.task_id: _stop_ref(owner, card.task_id, _newest_blocked(conn, card.task_id))
                     for card in (needs_file, designer)}
    before = {card.task_id: _observed(card) for card in (needs_file, designer)}

    async with TestClient(TestServer(_app())) as client:
        items = {card.task_id: await _listed(client, card) for card in (capability, needs_file, designer)}
        question = items[capability.task_id]["question"]
        taken = await _answer(client, items[capability.task_id]["decision_ref"], {
            "stop_ref": question["stop_ref"], "choice": None, "words": _WORDS,
        })
        refused = [
            await _answer(client, items[card.task_id]["decision_ref"], {
                "stop_ref": stop_refs[card.task_id], "choice": choice, "words": _WORDS,
            })
            for card in (needs_file, designer) for choice in (None, 1)
        ]

    assert (question["stop"], question["choices"], question["words"]) == ("plain", [], "required")
    assert taken == _answered(items[capability.task_id]["decision_ref"], question["stop_ref"], None, True)
    seen = _observed(capability)
    assert seen["comments"] == [("owner", f"The owner answered: {question['text']}\n{_WORDS}")]
    assert (len(seen["answers"]), seen["status"], seen["unblocked"]) == (1, "ready", 1)
    # The list serves no question for the two, and an answer to either is refused and changes nothing.
    assert not [item for item in (items[needs_file.task_id], items[designer.task_id]) if "question" in item]
    for answer in refused:
        _assert_refused(answer, 404, _NOT_WAITING, "decision_not_found")
    for card in (needs_file, designer):
        assert _observed(card) | {"events": None} == before[card.task_id] | {"events": None}
        assert (_observed(card)["status"], _observed(card)["answers"]) == ("blocked", [])


@pytest.mark.asyncio
@pytest.mark.parametrize("reason", [_PLAIN, f"Raphael cannot read {_PATH} without you."], ids=["plain", "path"])
async def test_35_p_a_plain_stop_takes_words_and_the_comment_begins_with_the_text_the_list_serves(owner, reason):
    project = _project(owner, "Sign up Website")
    card = _card(project, "Build the sign up page", stop=None, reason=reason)
    words = "Use the prices from last year."

    async with TestClient(TestServer(_app())) as client:
        item = await _listed(client, card)
        question = item["question"]
        taken = await _answer(client, item["decision_ref"], {
            "stop_ref": question["stop_ref"], "choice": None, "words": words,
        })

    seen = _observed(card)
    assert (question["stop"], question["choices"], question["words"]) == ("plain", [], "required")
    assert taken == _answered(item["decision_ref"], question["stop_ref"], None, True)
    (answered,) = seen["answers"]
    assert {key: value for key, value in answered.items() if key != "answered_at"} == {
        "stop_ref": question["stop_ref"], "choice": None, "words_length": len(words), "actor": owner.actor,
    }
    ((author, body),) = seen["comments"]
    assert author == "owner" and body.splitlines() == [f"The owner answered: {question['text']}", words]
    assert _PATH not in body
    assert (seen["status"], seen["unblocked"]) == ("ready", 1)


# ---------------------------------------------------------------------------
# o: a worker's environment
# ---------------------------------------------------------------------------


def test_35_o_under_a_workers_env_the_answer_reaches_the_cards_own_board_and_the_root_register(owner, monkeypatch):
    """The answer resolves the card's own board by the store path the owner's list resolved: under exactly the
    environment the dispatcher injects into a worker of another board, the question is read and the answer written on
    the card's own board under the root, and the write transaction still reads the root board register for that
    board."""
    import subprocess

    from gateway.platforms import api_server_owner_answers

    project = _project(owner, "Sign up Website")
    card = _card(project, "Build the sign up page")
    (item,) = ow.list_owner_decisions(owner, owner_answer=True)["data"]
    # The route resolves the ref in the owner's own environment, the gateway's, before anything is recorded.
    board, store, task_id = api_server_owner_answers._waiting_card(owner, item["decision_ref"])
    root = get_hermes_home()
    profile_home, worker_board = root / "profiles" / "worker", root / "kanban" / "boards" / "worker-board"
    profile_home.mkdir(parents=True)
    kb.create_board("worker-board", name="Worker board")
    with contextlib.closing(kb.connect(board="worker-board")) as conn:
        worker = kb.claim_task(conn, kb.create_task(conn, title="Write the menu page", assignee="worker"))
    assert (board, task_id) == (card.board, card.task_id) and board != "worker-board" and worker is not None

    # Exactly the environment the dispatcher's own spawn injects into that worker.
    spawned = {}
    monkeypatch.setattr(subprocess, "Popen", lambda cmd, **kwargs: spawned.update(kwargs) or SimpleNamespace(pid=4242))
    kb._default_spawn(worker, str(worker_board / "workspaces" / worker.id), board="worker-board")
    for name in set(os.environ) - set(spawned["env"]):
        monkeypatch.delenv(name)
    for name, value in spawned["env"].items():
        monkeypatch.setenv(name, value)
    assert (
        os.environ["HERMES_KANBAN_DB"], os.environ["HERMES_KANBAN_BOARD"], os.environ["HERMES_KANBAN_TASK"],
        os.environ["HERMES_HOME"],
    ) == (str(worker_board / "kanban.db"), "worker-board", worker.id, str(profile_home))
    # The card's board still resolves to its own store under the root, never to the worker's pinned one.
    assert (kb.kanban_home() if board == kb.DEFAULT_BOARD else kb.board_dir(board)) / "kanban.db" == store
    in_profile_home = sorted(profile_home.rglob("*"))
    with contextlib.closing(sqlite3.connect(worker_board / "kanban.db")) as conn:
        worker_events = conn.execute("SELECT COUNT(*) FROM task_events").fetchone()
    registers = []
    monkeypatch.setattr(kb, "get_register_entry", lambda slug, real=kb.get_register_entry: (
        registers.append((slug, kb.register_db_path())) or real(slug)
    ))

    reader = ow._open_read_only_sqlite(store, label="kanban.db")
    try:
        served = oq.decision_questions(reader, [task_id], owner)
    finally:
        reader.close()
    with kb.connect_closing(store) as conn:
        outcome = oq.accept_answer(conn, task_id, item["question"]["stop_ref"], 1, None, owner=owner)

    # Independent count, read with plain sqlite under the root, never through the kernel.
    seen = _observed(card)
    assert served == {task_id: item["question"]} and outcome == "resumed"
    (answered,) = seen["answers"]
    assert {key: value for key, value in answered.items() if key != "answered_at"} == {
        "stop_ref": item["question"]["stop_ref"], "choice": 1, "words_length": 0, "actor": owner.actor,
    }
    assert seen["comments"] == [("owner", f"The owner answered: {_QUESTION}\nChoice 1: {_CHOICES[0]['sentence']}")]
    assert (seen["status"], seen["unblocked"]) == ("ready", 1)
    # The fence of the card's board was read from the root board register, where that board is live.
    assert registers and set(registers) == {(board, root / "kanban" / "board_register.db")}
    with contextlib.closing(sqlite3.connect(root / "kanban" / "board_register.db")) as conn:
        assert conn.execute(
            "SELECT lifecycle FROM board_register WHERE board_name = ?", (board,),
        ).fetchall() == [("live",)]
    # Nothing reached the worker's own board or the profile home.
    with contextlib.closing(sqlite3.connect(worker_board / "kanban.db")) as conn:
        assert conn.execute("SELECT COUNT(*) FROM task_events").fetchone() == worker_events
        assert conn.execute("SELECT COUNT(*) FROM task_comments").fetchone() == (0,)
    assert sorted(profile_home.rglob("*")) == in_profile_home
