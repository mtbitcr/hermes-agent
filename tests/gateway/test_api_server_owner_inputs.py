"""The owner gives a card held for a missing input the file it named (card C3 of the named inputs plan).

One owner route, ``POST /v1/owner-workspace/decisions/{ref}/inputs/{n}``, takes the raw file as its body.
Each test drives the adapter's real route into the real kernel inside the per-test ``HERMES_HOME``: every
card is made by the real authority and apply path and held by the real claim, and its ref is the one the
owner's own Decisions feed lists. Each refusal answers its fixed sentence from the plan and stores
nothing; an accepted file is stored once, as the owner's, under its derived name, and the last one
releases the card.
"""

from __future__ import annotations

import contextlib
import json
import os
import sqlite3
import uuid
from types import SimpleNamespace

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import PlatformConfig
from gateway.platforms.api_server import MAX_REQUEST_BYTES, APIServerAdapter, body_limit_middleware
from hermes_cli import kanban_db as kb
from hermes_cli import owner_named_inputs as oni
from hermes_cli import owner_workspace as ow
from hermes_cli.owner_text import check_owner_text
from hermes_constants import get_hermes_home

# The real authority and apply path of card C1, the hold of card C2 and the
# harnesses under them (temporary home, auto-approver, configured provider,
# the owner the gateway routes resolve) are reused verbatim.
from tests.gateway.test_api_server_owner_waiting import (  # noqa: F401
    _write_owner_workspace_config,
    owner,
)
from tests.hermes_cli.test_owner_input_hold import _MENU, _OPENING_HOURS, _PRICE_LIST, _card
from tests.hermes_cli.test_owner_workspace import (  # noqa: F401
    _configured_provider,
    ctx,
)

_ROUTE = "/v1/owner-workspace/decisions/{ref}/inputs/{n}"
_KEY = "owner-inputs-" + "5e0c7a93" * 4
# The plan's refusal sentences, word for word (section 4).
_TOO_BIG = "This file is larger than 8 MB. Please send a smaller file."
_EMPTY = "This file is empty. Please choose the file again."
_WRONG_TYPE = "Raphael can only take text files here: .txt, .md, .csv or .json."
_NOT_ASKED = "This is not the kind of file Raphael asked for. Please send the file it named."
_NOT_TEXT = "This file does not look like plain text. Please save it as text and try again."
_NOT_WAITING = "This request is no longer waiting for a file."
_STORED_FAILURE = check_owner_text(None, "stored_failure")
_PRICE_TABLE = {"name": "price table", "type": "json"}
_PRICES = "item,price\ncafé au lait,3\n".encode("utf-8")


def _app(key: str = "", middlewares: tuple = ()) -> web.Application:
    """The adapter's own upload route, as ``connect()`` adds it, behind *middlewares* and nothing else."""
    adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={"key": key} if key else {}))
    app = web.Application(middlewares=middlewares, client_max_size=MAX_REQUEST_BYTES)
    rows = [row for row in adapter._http_route_table() if row[:2] == ("POST", _ROUTE)]
    assert len(rows) == 1, "the upload route is not registered"
    for method, path, handler in rows:
        app.router.add_route(method, path, handler)
    return app


def _held(ctx, *inputs: dict) -> SimpleNamespace:
    """One card of the real authority and apply path, held by the real claim for its *inputs*, with
    its ref as the owner's own Decisions feed lists it."""
    setup, task_id = _card(ctx, *inputs)
    with contextlib.closing(kb.connect(board=setup["board"])) as conn:
        assert kb.claim_task(conn, task_id) is None
    (item,) = ow.list_owner_decisions(ctx, owner_input_upload=True)["data"]
    return SimpleNamespace(
        board=setup["board"], project_id=setup["project_id"], task_id=task_id,
        ref=item["decision_ref"],
    )


async def _post(client: TestClient, ref: str, n, body: bytes, content_type: str = "text/csv",
                headers: dict | None = None) -> tuple[int, dict]:
    response = await client.post(
        f"/v1/owner-workspace/decisions/{ref}/inputs/{n}", data=body,
        headers={"Content-Type": content_type, **(headers or {})},
    )
    return response.status, await response.json()


def _files(card) -> list[tuple]:
    """Each file on the card as (file name, uploader, content type, bytes), and the folder's files."""
    with contextlib.closing(kb.connect(board=card.board)) as conn:
        attachments = kb.list_attachments(conn, card.task_id)
        attached = [event for event in kb.list_events(conn, card.task_id) if event.kind == "attached"]
    assert len(attached) == len(attachments)
    folder = kb.task_attachments_dir(card.task_id, board=card.board)
    on_disk = sorted(path.name for path in folder.iterdir()) if folder.exists() else []
    assert on_disk == sorted(attachment.filename for attachment in attachments)
    return [
        (attachment.filename, attachment.uploaded_by, attachment.content_type,
         kb.read_attachment_bytes(attachment, board=card.board))
        for attachment in attachments
    ]


def _state(card) -> tuple:
    """The card's status and block kind."""
    with contextlib.closing(kb.connect(board=card.board)) as conn:
        row = conn.execute(
            "SELECT status, block_kind FROM tasks WHERE id = ?", (card.task_id,),
        ).fetchone()
    return tuple(row)


def _missing(ctx) -> dict:
    """Each listed owner-input card's still-missing input numbers, by title."""
    return {
        item["title"]: [entry["n"] for entry in item.get("missing_inputs", [])]
        for item in ow.list_owner_decisions(ctx, owner_input_upload=True)["data"]
    }


def _assert_refused(answer: tuple[int, dict], status: int, sentence: str) -> None:
    got, body = answer
    assert (got, body["error"]["message"]) == (status, sentence)


# ---------------------------------------------------------------------------
# Each refusal: its sentence, and nothing stored
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("n,body,content_type,status,sentence", [
    pytest.param(1, b"a" * (oni.MAX_INPUT_BYTES + 1), "text/csv", 413, _TOO_BIG, id="8-mib-and-one-byte"),
    pytest.param(1, b"", "text/csv", 400, _EMPTY, id="empty"),
    pytest.param(1, _PRICES, "image/png", 415, _WRONG_TYPE, id="image"),
    pytest.param(1, _PRICES, "application/octet-stream", 415, _WRONG_TYPE, id="octet-stream"),
    pytest.param(1, _PRICES, "text/html", 415, _WRONG_TYPE, id="html"),
    pytest.param(1, _PRICES, "application/json", 415, _NOT_ASKED, id="csv-sent-as-json"),
    pytest.param(1, _PRICES, "text/markdown", 415, _NOT_ASKED, id="csv-sent-as-markdown"),
    pytest.param(2, b"Monday to Friday\n", "text/csv", 415, _NOT_ASKED, id="txt-sent-as-csv"),
    pytest.param(4, b'{"tea": 3}', "text/plain", 415, _NOT_ASKED, id="json-sent-as-text"),
    pytest.param(1, b"item,price\n\xff\xfe,3\n", "text/csv", 400, _NOT_TEXT, id="invalid-utf-8"),
    pytest.param(1, b"item,price\n\xed\xa0\x80,3\n", "text/csv", 400, _NOT_TEXT, id="surrogate"),
    pytest.param(1, b"item,price\ntea\x00,3\n", "text/csv", 400, _NOT_TEXT, id="nul-byte"),
    pytest.param(4, b'{"tea": 3', "application/json", 400, _NOT_TEXT, id="bad-json"),
    pytest.param(4, b"tea", "application/json", 400, _NOT_TEXT, id="not-json"),
])
async def test_each_refusal_answers_its_sentence_and_stores_nothing(
    ctx, owner, n, body, content_type, status, sentence,
):
    card = _held(ctx, _PRICE_LIST, _OPENING_HOURS, _MENU, _PRICE_TABLE)

    async with TestClient(TestServer(_app())) as client:
        answer = await _post(client, card.ref, n, body, content_type)

    _assert_refused(answer, status, sentence)
    assert _files(card) == []
    assert _state(card) == ("blocked", "needs_input")
    assert _missing(ctx) == {"Publish the price page": [1, 2, 3, 4]}


@pytest.mark.asyncio
async def test_a_ref_of_another_owner_or_not_from_the_feed_is_refused(ctx, owner):
    card = _held(ctx, _PRICE_LIST)
    others = [
        # The same held card, keyed for another owner.
        ow._owner_decision_ref(
            ow.OwnerContext(actor="raphael-planner", profile="raphael-planner", session="run_other"),
            authority="task", native_id=card.task_id,
        ),
        # The same card keyed as a suggestion, its native id, and no ref at all.
        ow._owner_decision_ref(ctx, authority="recommendation", native_id=card.task_id),
        card.task_id,
        "decision_" + "0" * 32,
    ]

    async with TestClient(TestServer(_app())) as client:
        answers = [await _post(client, ref, 1, _PRICES) for ref in others]

    for answer in answers:
        _assert_refused(answer, 404, _NOT_WAITING)
    assert _files(card) == []
    assert _state(card) == ("blocked", "needs_input")


@pytest.mark.asyncio
async def test_a_card_held_on_a_board_outside_the_owners_projects_is_refused(ctx, owner):
    card = _held(ctx, _PRICE_LIST)
    kb.create_board("elsewhere", name="Elsewhere")
    with contextlib.closing(kb.connect(board="elsewhere")) as conn:
        task_id = kb.create_task(
            conn, title="Publish the price page", assignee="default", project_id=card.project_id,
        )
        with kb.write_txn(conn):
            kb._append_event(conn, task_id, oni.DECLARED_EVENT, oni._declaration([_PRICE_LIST]))
        assert kb.claim_task(conn, task_id) is None
    elsewhere = SimpleNamespace(board="elsewhere", task_id=task_id)
    assert _state(elsewhere) == ("blocked", "needs_input")
    # Keyed for this very owner, but the owner's feed never reads that board.
    ref = ow._owner_decision_ref(ctx, authority="task", native_id=task_id)

    async with TestClient(TestServer(_app())) as client:
        answer = await _post(client, ref, 1, _PRICES)

    _assert_refused(answer, 404, _NOT_WAITING)
    assert _files(elsewhere) == [] and _files(card) == []
    assert _state(elsewhere) == ("blocked", "needs_input")


@pytest.mark.asyncio
async def test_a_card_that_is_not_waiting_for_a_file_is_refused(ctx, owner):
    setup, ready = _card(ctx, _PRICE_LIST)
    with contextlib.closing(kb.connect(board=setup["board"])) as conn:
        # A question for the owner: blocked needs_input, but it names no file.
        question = kb.create_task(
            conn, title="Choose the venue", assignee="raphael-worker", project_id=setup["project_id"],
        )
        assert kb.claim_task(conn, question) is not None
        assert kb.block_task(conn, question, reason="Which venue should we book?", kind="needs_input") is True
    (item,) = ow.list_owner_decisions(ctx, owner_input_upload=True)["data"]
    assert item["title"] == "Choose the venue"
    cards = {
        # Declares a file but was never held: it is ready, not waiting.
        "ready": (ready, ow._owner_decision_ref(ctx, authority="task", native_id=ready)),
        "question": (question, item["decision_ref"]),
    }

    async with TestClient(TestServer(_app())) as client:
        answers = {name: await _post(client, ref, 1, _PRICES) for name, (_, ref) in cards.items()}

    for name, (task_id, _) in cards.items():
        _assert_refused(answers[name], 404, _NOT_WAITING)
        assert _files(SimpleNamespace(board=setup["board"], task_id=task_id)) == []


@pytest.mark.asyncio
async def test_an_input_that_is_not_declared_or_already_given_is_refused(ctx, owner):
    card = _held(ctx, _PRICE_LIST, _OPENING_HOURS)

    async with TestClient(TestServer(_app())) as client:
        undeclared = [await _post(client, card.ref, n, _PRICES) for n in ("3", "0", "01", "one", "-1", "1.0")]
        given = await _post(client, card.ref, 1, _PRICES)
        again = await _post(client, card.ref, 1, b"item,price\ntea,4\n")

    for answer in undeclared:
        _assert_refused(answer, 404, _NOT_WAITING)
    assert given[0] == 200
    _assert_refused(again, 404, _NOT_WAITING)
    assert _files(card) == [("price-list.csv", "owner", "text/csv", _PRICES)]
    assert _state(card) == ("blocked", "needs_input")


@pytest.mark.asyncio
@pytest.mark.parametrize("ref_ok,n,body,content_type,status,sentence", [
    pytest.param(False, "3", b"", "image/png", 404, _NOT_WAITING, id="ref-before-input"),
    pytest.param(True, "3", b"a" * (oni.MAX_INPUT_BYTES + 1), "image/png", 404, _NOT_WAITING,
                 id="input-before-size"),
    pytest.param(True, "1", b"", "image/png", 400, _EMPTY, id="empty-before-type"),
    pytest.param(True, "1", b"\x00" * (oni.MAX_INPUT_BYTES + 1), "image/png", 413, _TOO_BIG,
                 id="size-before-type-and-text"),
    pytest.param(True, "1", b"\xff\xfe", "image/png", 415, _WRONG_TYPE, id="type-before-text"),
    pytest.param(True, "1", b"tea\x00", "application/json", 415, _NOT_ASKED, id="match-before-text"),
])
async def test_the_checks_run_in_the_plans_order(ctx, owner, ref_ok, n, body, content_type, status, sentence):
    card = _held(ctx, _PRICE_LIST, _OPENING_HOURS)

    async with TestClient(TestServer(_app())) as client:
        answer = await _post(client, card.ref if ref_ok else "decision_" + "0" * 32, n, body, content_type)

    _assert_refused(answer, status, sentence)
    assert _files(card) == []


# ---------------------------------------------------------------------------
# Owner authority
# ---------------------------------------------------------------------------


def _workspace_read_token() -> str:
    """A real workspace read token (``hrw1_``), minted in this test's home."""
    from plugins.dashboard_auth.raphael_workspace import token_store

    folder = get_hermes_home() / "workspace-token"
    folder.mkdir(mode=0o700)
    token_store.issue(out_path=folder / "bearer")
    token = (folder / "bearer").read_text(encoding="utf-8").strip()
    assert token.startswith("hrw1_")
    return token


@pytest.mark.asyncio
async def test_a_request_without_owner_authority_is_refused(ctx, owner):
    card = _held(ctx, _PRICE_LIST)
    refused = [
        {},
        {"Authorization": f"Bearer {_workspace_read_token()}"},
        {"Authorization": "Bearer " + "0" * 40},
    ]

    async with TestClient(TestServer(_app(_KEY))) as client:
        answers = [await _post(client, card.ref, 1, _PRICES, headers=headers) for headers in refused]
        assert _files(card) == []
        # Control: the owner's own key is accepted.
        accepted = await _post(client, card.ref, 1, _PRICES, headers={"Authorization": f"Bearer {_KEY}"})

    for answer in answers:
        _assert_refused(answer, 401, _STORED_FAILURE)
    assert accepted[0] == 200
    assert _files(card) == [("price-list.csv", "owner", "text/csv", _PRICES)]


@pytest.mark.asyncio
async def test_with_the_owner_workspace_switched_off_nothing_is_stored(ctx, owner):
    card = _held(ctx, _PRICE_LIST)
    _write_owner_workspace_config(enabled=False)

    async with TestClient(TestServer(_app())) as client:
        answer = await _post(client, card.ref, 1, _PRICES)

    _assert_refused(answer, 404, _STORED_FAILURE)
    assert _files(card) == []


@pytest.mark.asyncio
async def test_a_failure_inside_answers_the_stored_failure_sentence_only(ctx, owner, monkeypatch):
    card = _held(ctx, _PRICE_LIST)

    def broken(*_args, **_kwargs):
        raise OSError("disk full under /srv/hermes/private")

    monkeypatch.setattr(kb, "store_owner_input", broken)
    async with TestClient(TestServer(_app())) as client:
        response = await client.post(
            f"/v1/owner-workspace/decisions/{card.ref}/inputs/1", data=_PRICES,
            headers={"Content-Type": "text/csv"},
        )
        status, text = response.status, await response.text()

    assert status == 503
    assert json.loads(text)["error"]["message"] == _STORED_FAILURE
    assert "disk" not in text and "/srv" not in text and "Traceback" not in text
    assert _files(card) == []
    assert _state(card) == ("blocked", "needs_input")


# ---------------------------------------------------------------------------
# Accepted uploads and the release
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("n,content_type,body,stored", [
    pytest.param(1, "text/csv", _PRICES, ("price-list.csv", "text/csv"), id="csv"),
    pytest.param(1, "text/plain", _PRICES, ("price-list.csv", "text/csv"), id="csv-as-text"),
    pytest.param(1, "text/csv; charset=utf-8", _PRICES, ("price-list.csv", "text/csv"), id="csv-charset"),
    pytest.param(2, "text/plain", b"Monday to Friday, 9 to 17\n", ("opening-hours.txt", "text/plain"),
                 id="txt"),
    pytest.param(3, "text/markdown", b"# Menu\n\n- Tea\n", ("menu.md", "text/markdown"), id="md"),
    pytest.param(3, "text/plain", b"# Menu\n\n- Tea\n", ("menu.md", "text/markdown"), id="md-as-text"),
    pytest.param(4, "application/json", b'{"tea": 3}\n', ("price-table.json", "application/json"),
                 id="json"),
])
async def test_an_accepted_file_is_stored_once_as_the_owners_under_its_derived_name(
    ctx, owner, n, content_type, body, stored,
):
    card = _held(ctx, _PRICE_LIST, _OPENING_HOURS, _MENU, _PRICE_TABLE)

    async with TestClient(TestServer(_app())) as client:
        status, answer = await _post(client, card.ref, n, body, content_type)

    assert status == 200
    assert answer == {
        "object": "hermes.owner_workspace.input_upload",
        "decision_ref": card.ref,
        "n": n,
        "released": False,
    }
    assert _files(card) == [(stored[0], oni.OWNER_UPLOADER, stored[1], body)]
    assert _state(card) == ("blocked", "needs_input")
    assert _missing(ctx) == {"Publish the price page": [i for i in (1, 2, 3, 4) if i != n]}


@pytest.mark.asyncio
async def test_a_file_of_exactly_8_mib_is_accepted(ctx, owner):
    card = _held(ctx, _OPENING_HOURS)
    body = b"a" * oni.MAX_INPUT_BYTES

    async with TestClient(TestServer(_app())) as client:
        status, _ = await _post(client, card.ref, 1, body, "text/plain")

    assert status == 200
    assert _files(card) == [("opening-hours.txt", "owner", "text/plain", body)]


@pytest.mark.asyncio
async def test_the_last_file_releases_the_card(ctx, owner):
    card = _held(ctx, _PRICE_LIST, _OPENING_HOURS)

    async with TestClient(TestServer(_app())) as client:
        first = await _post(client, card.ref, 2, b"Monday to Friday\n", "text/plain")
        assert _state(card) == ("blocked", "needs_input")
        last = await _post(client, card.ref, 1, _PRICES)
        after = await _post(client, card.ref, 1, _PRICES)

    assert (first[0], first[1]["released"]) == (200, False)
    assert (last[0], last[1]["released"]) == (200, True)
    _assert_refused(after, 404, _NOT_WAITING)
    assert _state(card)[0] == "ready"
    assert _missing(ctx) == {}
    assert [name for name, *_ in _files(card)] == ["opening-hours.txt", "price-list.csv"]
    with contextlib.closing(kb.connect(board=card.board)) as conn:
        claimed = kb.claim_task(conn, card.task_id)
    assert claimed is not None and claimed.status == "running"


@pytest.mark.asyncio
async def test_a_card_whose_parent_is_not_done_is_released_to_todo(ctx, owner):
    card = _held(ctx, _PRICE_LIST)
    with contextlib.closing(kb.connect(board=card.board)) as conn:
        parent = kb.create_task(
            conn, title="Collect the prices", assignee="raphael-worker", project_id=card.project_id,
        )
        kb.link_tasks(conn, parent, card.task_id)
        assert kb.get_task(conn, parent).status not in ("done", "archived")
    assert _state(card) == ("blocked", "needs_input")

    async with TestClient(TestServer(_app())) as client:
        status, answer = await _post(client, card.ref, 1, _PRICES)

    assert (status, answer["released"]) == (200, True)
    assert _state(card)[0] == "todo"
    assert _missing(ctx) == {}


# ---------------------------------------------------------------------------
# Idempotency-Key
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_retry_with_the_same_idempotency_key_stores_nothing_new(ctx, owner):
    card = _held(ctx, _PRICE_LIST, _OPENING_HOURS)
    first_key, last_key = (f"upload-{uuid.uuid4().hex}" for _ in range(2))
    hours = b"Monday to Friday\n"

    async with TestClient(TestServer(_app())) as client:
        first = await _post(client, card.ref, 2, hours, "text/plain", {"Idempotency-Key": first_key})
        first_retry = await _post(client, card.ref, 2, hours, "text/plain", {"Idempotency-Key": first_key})
        last = await _post(client, card.ref, 1, _PRICES, headers={"Idempotency-Key": last_key})
        # The card is released now, and the retry still gets the first answer.
        last_retry = await _post(client, card.ref, 1, _PRICES, headers={"Idempotency-Key": last_key})
        other_body = await _post(client, card.ref, 1, b"item,price\ntea,4\n",
                                 headers={"Idempotency-Key": last_key})
        without_key = await _post(client, card.ref, 1, _PRICES)

    assert first[0] == 200 and first_retry == first
    assert last[0] == 200 and last[1]["released"] is True and last_retry == last
    _assert_refused(other_body, 409, _STORED_FAILURE)
    _assert_refused(without_key, 404, _NOT_WAITING)
    assert _files(card) == [
        ("opening-hours.txt", "owner", "text/plain", hours),
        ("price-list.csv", "owner", "text/csv", _PRICES),
    ]
    with contextlib.closing(kb.connect(board=card.board)) as conn:
        assert [event.kind for event in kb.list_events(conn, card.task_id)].count("unblocked") == 1
    assert _state(card)[0] == "ready"


@pytest.mark.asyncio
async def test_an_unusable_idempotency_key_is_refused_before_anything_is_stored(ctx, owner):
    card = _held(ctx, _PRICE_LIST)

    async with TestClient(TestServer(_app())) as client:
        answer = await _post(client, card.ref, 1, _PRICES, headers={"Idempotency-Key": "no spaces allowed"})

    _assert_refused(answer, 400, _STORED_FAILURE)
    assert _files(card) == []


# ---------------------------------------------------------------------------
# Owner rule 1: one transaction, the server-wide limit, a worker's environment
# ---------------------------------------------------------------------------


def _locked(*_args, **_kwargs):
    raise sqlite3.OperationalError("database is locked")


def _failing_once_stored(real):
    """*real*, failing as soon as the card's new attachment row is visible: after the store."""

    def check(conn, task_id, *args, **kwargs):
        if kb.list_attachments(conn, task_id):
            _locked()
        return real(conn, task_id, *args, **kwargs)

    return check


@pytest.mark.asyncio
@pytest.mark.parametrize("step", ["missing-input-check", "release"])
async def test_a_failure_after_the_store_leaves_nothing_and_the_retry_releases_the_card(
    ctx, owner, monkeypatch, step,
):
    card = _held(ctx, _PRICE_LIST)
    headers = {"Idempotency-Key": f"upload-{uuid.uuid4().hex}"}

    with monkeypatch.context() as patch:
        if step == "release":
            for name in ("_unblock_task_within_txn", "unblock_task"):
                patch.setattr(kb, name, _locked, raising=False)
        else:
            for module in (kb, oni):
                patch.setattr(module, "missing_inputs", _failing_once_stored(oni.missing_inputs), raising=False)
        async with TestClient(TestServer(_app())) as client:
            failed = await _post(client, card.ref, 1, _PRICES, headers=headers)

    _assert_refused(failed, 503, _STORED_FAILURE)
    # No attachment row, no file and the card still held, still listing the input as missing.
    assert _files(card) == []
    folder = kb.task_attachments_dir(card.task_id, board=card.board)
    assert not folder.exists() or list(folder.iterdir()) == []
    assert _state(card) == ("blocked", "needs_input")
    assert _missing(ctx) == {"Publish the price page": [1]}

    async with TestClient(TestServer(_app())) as client:
        retried = await _post(client, card.ref, 1, _PRICES, headers=headers)

    assert (retried[0], retried[1]["released"]) == (200, True)
    assert _files(card) == [("price-list.csv", "owner", "text/csv", _PRICES)]
    assert _state(card)[0] == "ready"
    assert _missing(ctx) == {}
    with contextlib.closing(kb.connect(board=card.board)) as conn:
        assert [event.kind for event in kb.list_events(conn, card.task_id)].count("unblocked") == 1


@pytest.mark.asyncio
async def test_above_the_server_wide_limit_the_servers_own_refusal_answers_before_any_route_check(ctx, owner):
    """Through the production body-limit middleware: above 10,000,000 bytes its own 413 answers first, also for
    a ref that waits for nothing; from 8 MiB plus one byte up to that limit the plan's sentence answers."""
    card = _held(ctx, _PRICE_LIST)
    nowhere = "decision_" + "0" * 32

    async with TestClient(TestServer(_app(middlewares=(body_limit_middleware,)))) as client:
        above = [await _post(client, ref, 1, b"a" * (MAX_REQUEST_BYTES + 1)) for ref in (card.ref, nowhere)]
        below = [
            await _post(client, card.ref, 1, b"a" * size) for size in (oni.MAX_INPUT_BYTES + 1, MAX_REQUEST_BYTES)
        ]
        not_waiting = await _post(client, nowhere, 1, b"a" * MAX_REQUEST_BYTES)

    for status, body in above:
        assert (status, body["error"]["message"], body["error"]["code"]) == (
            413, "Request body too large.", "body_too_large",
        )
    for answer in below:
        _assert_refused(answer, 413, _TOO_BIG)
    _assert_refused(not_waiting, 404, _NOT_WAITING)
    assert _files(card) == []
    assert _state(card) == ("blocked", "needs_input")


def test_under_a_workers_env_the_kernel_still_reaches_the_cards_board_and_the_root_register(ctx, owner, monkeypatch):
    """The kernel function resolves the attachment store path: under exactly the environment the dispatcher
    injects into a worker of another board it still stores on the card's own board under the root, and its write
    transaction still reads the root board register for that board."""
    import subprocess

    from gateway.platforms import api_server_owner_inputs

    card = _held(ctx, _PRICE_LIST)
    # The route resolves the ref in the owner's own environment, the gateway's, before anything is stored.
    board, store, task_id = api_server_owner_inputs._waiting_card(owner, card.ref)
    root = get_hermes_home()
    profile_home, worker_board = root / "profiles" / "worker", root / "kanban" / "boards" / "worker-board"
    profile_home.mkdir(parents=True)
    kb.create_board("worker-board", name="Worker board")
    with contextlib.closing(kb.connect(board="worker-board")) as conn:
        worker = kb.claim_task(conn, kb.create_task(conn, title="Write the price page", assignee="worker"))
    assert board == card.board != "worker-board" and worker is not None

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
    in_profile_home = sorted(profile_home.rglob("*"))
    registers = []
    monkeypatch.setattr(kb, "get_register_entry", lambda slug, real=kb.get_register_entry: (
        registers.append((slug, kb.register_db_path())) or real(slug)
    ))

    with kb.connect_closing(store) as conn:
        assert oni.accept_upload(conn, task_id, 1, _PRICES, "text/csv", board=board) == "released"

    # Independent count, read with plain sqlite and the folder listing under the root, never through the kernel.
    folder = (root / "kanban" if board == kb.DEFAULT_BOARD else store.parent) / "attachments" / task_id
    with contextlib.closing(sqlite3.connect(store)) as conn:
        stored = conn.execute(
            "SELECT filename, uploaded_by, size, stored_path FROM task_attachments WHERE task_id = ?", (task_id,),
        ).fetchall()
        (status,) = conn.execute("SELECT status FROM tasks WHERE id = ?", (task_id,)).fetchone()
    assert stored == [("price-list.csv", "owner", len(_PRICES), str((folder / "price-list.csv").resolve()))]
    assert status == "ready"
    assert [(path.name, path.read_bytes()) for path in folder.iterdir()] == [("price-list.csv", _PRICES)]
    # The fence of the card's board was read from the root board register, where that board is live.
    assert registers == [(board, root / "kanban" / "board_register.db")]
    with contextlib.closing(sqlite3.connect(root / "kanban" / "board_register.db")) as conn:
        assert conn.execute(
            "SELECT lifecycle FROM board_register WHERE board_name = ?", (board,),
        ).fetchall() == [("live",)]
    # Nothing reached the worker's own board or the profile home.
    with contextlib.closing(sqlite3.connect(worker_board / "kanban.db")) as conn:
        assert conn.execute("SELECT COUNT(*) FROM task_attachments").fetchone() == (0,)
    assert not (worker_board / "attachments").exists()
    assert sorted(profile_home.rglob("*")) == in_profile_home
