"""The owner's design choice is listed, recorded and returned to the designer's card.

Card 2 of the design options plan, platform tests 4, 5 and 6 of section G. A designer card that declared two or
three pictured options (card 1) waits ``needs_input``. Only a Decisions reader that names ``design_options_v1``
sees its ``design_options`` and the record of past choices, ``recent_design_choices``. One owner route,
``POST /v1/owner-workspace/decisions/{ref}/choose/{n}``, records the owner's choice, comments on the card and releases
it, in one write transaction. Each test drives the adapter's real routes into the real kernel inside the per-test
``HERMES_HOME``, and every count it proves is read by a second, plain connection to the card's own store.
"""

from __future__ import annotations

import contextlib
import json
import os
import sqlite3
import time
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import PlatformConfig
from gateway.platforms.api_server import APIServerAdapter
from hermes_cli import kanban_db as kb
from hermes_cli import owner_design_choices as odc
from hermes_cli import owner_workspace as ow
from hermes_cli.owner_text import check_owner_text
from hermes_constants import get_hermes_home

# The owner the routes resolve for themselves, a committed owner Project and the workspace switch are reused
# verbatim from the suggestion decisions, and the designer's question, sentences and pictures from card 1.
from tests.gateway.test_api_server_owner_suggestion_decisions import (  # noqa: F401  (owner is a fixture)
    _project,
    _write_owner_workspace_config,
    owner,
)
from tests.hermes_cli.test_owner_design_choices import ALTS, JPEG, PNG, QUESTION, SENTENCES, WEBP
# The owner's approval of a move, as the owner move tests give it.
from tests.hermes_cli.test_owner_workspace import _with_approver

_ROUTE = "/v1/owner-workspace/decisions/{ref}/choose/{n}"
_FEED = "/v1/owner-workspace/decisions"
_API_KEY = "owner-choices-" + "3f9a1c64" * 4
_IDEMPOTENCY_KEY = "choice-" + "8d2e5b17" * 4
# The plan's sentence for a choice that no longer waits (section C, step 6).
_NOT_WAITING = "This choice is no longer waiting."
_STORED_FAILURE = check_owner_text(None, "stored_failure")
_PICTURES = ((PNG, "image/png", "png"), (JPEG, "image/jpeg", "jpg"), (WEBP, "image/webp", "webp"))


def _app(key: str = "") -> web.Application:
    """The adapter's own Decisions routes, the choose route among them, as ``connect()`` adds them."""
    adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={"key": key} if key else {}))
    app = web.Application()
    rows = [row for row in adapter._http_route_table() if row[1].startswith(_FEED)]
    assert [row[:2] for row in rows].count(("POST", _ROUTE)) == 1, "the choose route is not registered"
    for method, path, handler in rows:
        app.router.add_route(method, path, handler)
    return app


def _store(board: str):
    folder = kb.kanban_home() if board == kb.DEFAULT_BOARD else kb.board_dir(board)
    return folder / "kanban.db"


def _designer_card(project: dict, title: str = "Design the home page", count: int = 3) -> SimpleNamespace:
    """A designer card of *project* that attached *count* pictures to itself and offered them as options (card 1),
    so that it waits for the owner's choice."""
    board = project["board"]
    with contextlib.closing(kb.connect(board=board)) as conn:
        task_id = kb.create_task(conn, title=title, assignee="designer", project_id=project["project_id"])
        claimed = kb.claim_task(conn, task_id)
        assert claimed is not None and claimed.status == "running"
        ids = [
            str(kb.store_attachment_bytes(
                conn, task_id, f"option-{n}.{ext}", data, content_type=media_type, uploaded_by="agent", board=board,
            ))
            for n, (data, media_type, ext) in enumerate(_PICTURES[:count], start=1)
        ]
        options = [
            {"n": n, "sentence": SENTENCES[n - 1], "pictures": [{"attachment_id": picture, "alt": ALTS[n - 1]}]}
            for n, picture in enumerate(ids, start=1)
        ]
        assert odc.declare_design_options(
            conn, task_id, QUESTION, options, board=board, expected_run_id=claimed.current_run_id,
        )
    return SimpleNamespace(
        board=board, store=_store(board), project_id=project["project_id"], task_id=task_id, title=title,
        attachment_ids=ids, count=count,
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
            "declared": [json.loads(payload) for (payload,) in conn.execute(
                "SELECT payload FROM task_events WHERE task_id = ? AND kind = 'design_options_declared' ORDER BY id",
                (card.task_id,),
            )],
            "chosen": [json.loads(payload) for (payload,) in conn.execute(
                "SELECT payload FROM task_events WHERE task_id = ? AND kind = 'design_option_chosen' ORDER BY id",
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


async def _feed(client: TestClient, capabilities: str | None = None) -> tuple[int, bytes, dict]:
    response = await client.get(_FEED, params={"capabilities": capabilities} if capabilities else None)
    raw = await response.read()
    return response.status, raw, json.loads(raw)


async def _ref(client: TestClient, card) -> str:
    """The key the owner's own Decisions feed lists for *card*."""
    _status, _raw, feed = await _feed(client, "design_options_v1")
    (ref,) = [item["decision_ref"] for item in feed["data"] if item["title"] == card.title]
    return ref


async def _choose(client: TestClient, ref: str, n, headers: dict | None = None) -> tuple[int, dict]:
    response = await client.post(f"/v1/owner-workspace/decisions/{ref}/choose/{n}", headers=headers or {})
    return response.status, await response.json()


def _assert_refused(answer: tuple[int, dict], status: int, message: str, code: str) -> None:
    got, body = answer
    assert (got, body["error"]["message"], body["error"]["code"]) == (status, message, code)


def _iso(seconds: float) -> str:
    return datetime.fromtimestamp(int(seconds), timezone.utc).isoformat().replace("+00:00", "Z")


# ---------------------------------------------------------------------------
# Test 4: design_options, on the one waiting designer card, only when asked
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_4_the_feed_with_design_options_v1_adds_design_options_to_that_one_item_only(owner):
    project = _project(owner, "Bakery Website")
    card = _designer_card(project)
    with contextlib.closing(kb.connect(board=project["board"])) as conn:
        question = kb.create_task(
            conn, title="Choose the opening date", assignee="default", project_id=project["project_id"],
        )
        assert kb.block_task(conn, question, reason="Which date suits you?", kind="needs_input")

    async with TestClient(TestServer(_app())) as client:
        today = await _feed(client)
        upload_only = await _feed(client, "owner_input_upload_v1")
        granted = await _feed(client, "design_options_v1")

    (status, raw, plain), (_, upload_raw, _), (granted_status, _, listed) = today, upload_only, granted
    assert status == granted_status == 200
    assert {item["kind"] for item in plain["data"]} == {"owner_input"} and len(plain["data"]) == 2
    # The other capability names never bring the new fields.
    assert upload_raw == raw
    assert "recent_design_choices" not in plain
    assert not [item for item in plain["data"] if "design_options" in item]

    # Only the designer card's item carries design_options: exactly what the card declared.
    (offered,) = [item for item in listed["data"] if "design_options" in item]
    assert offered["title"] == card.title
    (declared,) = _observed(card)["declared"]
    assert offered["design_options"] == declared == {
        "object": "hermes.owner_workspace.design_options",
        "schema_version": 1,
        "question": QUESTION,
        "options": [
            {
                "n": n,
                "sentence": SENTENCES[n - 1],
                "pictures": [{
                    "attachment_id": card.attachment_ids[n - 1], "alt": ALTS[n - 1],
                    "media_type": _PICTURES[n - 1][1], "bytes": len(_PICTURES[n - 1][0]),
                }],
            }
            for n in (1, 2, 3)
        ],
    }
    assert listed["recent_design_choices"] == []
    # Without the two new fields the answer is byte for byte today's answer.
    stripped = {key: value for key, value in listed.items() if key != "recent_design_choices"}
    stripped["data"] = [
        {key: value for key, value in item.items() if key != "design_options"} for item in listed["data"]
    ]
    assert json.dumps(stripped).encode("utf-8") == raw


# ---------------------------------------------------------------------------
# Test 5: the choose route
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_5_the_choose_route_records_the_choice_comments_and_releases_the_card(owner):
    project = _project(owner, "Bakery Website")
    card = _designer_card(project)
    headers = {"Idempotency-Key": _IDEMPOTENCY_KEY}
    before = int(time.time())

    async with TestClient(TestServer(_app())) as client:
        ref = await _ref(client, card)
        first = await _choose(client, ref, 2, headers)
        recorded = _observed(card)
        # A replay with the same key gets the first answer and records nothing new.
        replay = await _choose(client, ref, 2, headers)
        replayed = _observed(card)
        # A second, different choice and a late choice of the same option.
        second = await _choose(client, ref, 1)
        late = await _choose(client, ref, 2)
        _status, _raw, feed = await _feed(client, "design_options_v1")

    assert first == (200, {
        "object": "hermes.owner_workspace.design_choice", "decision_ref": ref, "n": 2, "released": True,
    })
    assert replay == first and replayed == recorded
    for answer in (second, late):
        _assert_refused(answer, 404, _NOT_WAITING, "decision_not_found")
    assert _observed(card) == recorded
    # One record of the choice, one comment in plain words, and the card released as unblock_task releases it.
    (chosen,) = recorded["chosen"]
    assert {key: value for key, value in chosen.items() if key != "chosen_at"} == {
        "n": 2, "sentence": SENTENCES[1], "attachment_id": card.attachment_ids[1], "actor": owner.actor,
    }
    assert _iso(before) <= chosen["chosen_at"] <= _iso(time.time())
    assert recorded["comments"] == [(owner.actor, f"The owner chose option 2: {SENTENCES[1]}")]
    assert (recorded["status"], recorded["unblocked"]) == ("ready", 1)
    # It no longer waits, so the feed no longer lists it.
    assert feed["data"] == []


@pytest.mark.asyncio
async def test_5_a_card_held_again_after_the_choice_never_takes_a_second_choice(owner):
    """The first choice stands: held again for another answer, the card is listed as today's plain item, and a
    choice on its key is refused and changes nothing."""
    project = _project(owner, "Bakery Website")
    card = _designer_card(project)
    async with TestClient(TestServer(_app())) as client:
        ref = await _ref(client, card)
        assert (await _choose(client, ref, 3))[0] == 200
        with contextlib.closing(kb.connect(board=card.board)) as conn:
            assert kb.claim_task(conn, card.task_id) is not None
            assert kb.block_task(conn, card.task_id, reason="Which photo goes on top?", kind="needs_input")
        held = _observed(card)
        _status, _raw, feed = await _feed(client, "design_options_v1")
        again = await _choose(client, ref, 1)

    (item,) = feed["data"]
    assert item["decision_ref"] == ref and "design_options" not in item
    _assert_refused(again, 404, _NOT_WAITING, "decision_not_found")
    assert _observed(card) == held
    assert [entry["n"] for entry in held["chosen"]] == [3]


def _owner_move_to_ready(owner: ow.OwnerContext, card) -> None:
    """The owner's own move of the held *card* to ready, through the real owner kernel and its approval."""
    mover = ow.OwnerContext(actor=owner.actor, profile=owner.profile, session="run_owner_choices")
    with contextlib.closing(kb.connect(board=card.board)) as conn:
        revision = kb.task_event_revision(conn, card.task_id)
    approver = _with_approver(mover.session)
    moved = ow.move_task(
        mover, idempotency_key="move-blocked-to-ready", task_id=card.task_id, to_status="ready",
        expected_status="blocked", expected_revision=revision, project_id=card.project_id,
    )
    approver.join()
    assert (moved["ok"], moved["status"]) == (True, "ready")


@pytest.mark.asyncio
@pytest.mark.parametrize("release", ["unblock_task", "owner_move"])
async def test_5_a_card_released_and_held_again_for_another_question_no_longer_offers_its_options(owner, release):
    """The options belong to the hold their declaration made. Released by unblock_task or by the owner's own move to
    ready, claimed by a new run and held again for another answer, the card is listed as today's plain item, and a
    choice on its key is refused, with no key or a fresh one, and changes nothing."""
    project = _project(owner, "Bakery Website")
    card = _designer_card(project)
    async with TestClient(TestServer(_app())) as client:
        ref = await _ref(client, card)
        if release == "unblock_task":
            with contextlib.closing(kb.connect(board=card.board)) as conn:
                assert kb.unblock_task(conn, card.task_id)
        else:
            _owner_move_to_ready(owner, card)
        with contextlib.closing(kb.connect(board=card.board)) as conn:
            assert kb.claim_task(conn, card.task_id) is not None
            assert kb.block_task(conn, card.task_id, reason="Which date should we use?", kind="needs_input")
        held = _observed(card)
        _status, _raw, feed = await _feed(client, "design_options_v1")
        answers = [
            await _choose(client, ref, 2),
            await _choose(client, ref, 2, {"Idempotency-Key": f"choice-{release}-" + "4b7e0c93" * 4}),
        ]

    (item,) = feed["data"]
    assert (item["decision_ref"], item["kind"]) == (ref, "owner_input") and "design_options" not in item
    for answer in answers:
        _assert_refused(answer, 404, _NOT_WAITING, "decision_not_found")
    assert _observed(card) == held
    assert (held["status"], held["block_kind"], held["chosen"], held["comments"]) == ("blocked", "needs_input", [], [])


@pytest.mark.asyncio
async def test_5_a_comment_on_the_waiting_card_keeps_its_options_and_the_choice_releases_it(owner):
    """A comment written while the card waits does not end the hold its declaration made: the options stay listed,
    and the choice is recorded and releases the card."""
    project = _project(owner, "Bakery Website")
    card = _designer_card(project)
    async with TestClient(TestServer(_app())) as client:
        ref = await _ref(client, card)
        with contextlib.closing(kb.connect(board=card.board)) as conn:
            kb.add_comment(conn, card.task_id, owner.actor, "Please keep the logo at the top.")
        _status, _raw, feed = await _feed(client, "design_options_v1")
        chosen = await _choose(client, ref, 2)

    (item,) = feed["data"]
    seen = _observed(card)
    assert item["decision_ref"] == ref and item["design_options"] == seen["declared"][0]
    assert chosen == (200, {
        "object": "hermes.owner_workspace.design_choice", "decision_ref": ref, "n": 2, "released": True,
    })
    assert [entry["n"] for entry in seen["chosen"]] == [2]
    assert seen["comments"] == [
        (owner.actor, "Please keep the logo at the top."), (owner.actor, f"The owner chose option 2: {SENTENCES[1]}"),
    ]
    assert (seen["status"], seen["unblocked"]) == ("ready", 1)


@pytest.mark.asyncio
async def test_5_an_unknown_or_foreign_key_or_an_option_not_offered_is_refused_and_changes_nothing(owner):
    project = _project(owner, "Bakery Website")
    card = _designer_card(project, count=2)
    stranger = ow.OwnerContext(actor="stranger", profile="stranger", session="")
    foreign = _designer_card(_project(stranger, "Stranger Website"), title="Design the stranger's page")
    # A key the stranger's own feed shows: real, but not this owner's to choose.
    [foreign_item] = ow.list_owner_decisions(stranger)["data"]
    before = (_observed(card), _observed(foreign))

    async with TestClient(TestServer(_app())) as client:
        ref = await _ref(client, card)
        # Unknown, foreign, and a native id, which is never a key.
        answers = [
            await _choose(client, key, 1)
            for key in ("decision_" + "0" * 32, foreign_item["decision_ref"], card.task_id)
        ]
        # Options this card never offered.
        answers += [await _choose(client, ref, n) for n in (0, 3, "01", "x")]

    for answer in answers:
        _assert_refused(answer, 404, _NOT_WAITING, "decision_not_found")
    assert (_observed(card), _observed(foreign)) == before


@pytest.mark.asyncio
async def test_5_without_owner_auth_401_and_with_the_workspace_off_404_and_nothing_changes(owner):
    project = _project(owner, "Bakery Website")
    card = _designer_card(project)
    (item,) = ow.list_owner_decisions(owner)["data"]
    before = _observed(card)

    async with TestClient(TestServer(_app(_API_KEY))) as client:
        missing = await _choose(client, item["decision_ref"], 1)
        wrong = await _choose(client, item["decision_ref"], 1, {"Authorization": "Bearer not-the-owner"})
        _write_owner_workspace_config(enabled=False)
        disabled = await _choose(client, item["decision_ref"], 1, {"Authorization": f"Bearer {_API_KEY}"})
        unchanged = _observed(card)
        _write_owner_workspace_config(enabled=True)
        # Control: the same request with the owner's authority, the workspace on, records the choice.
        control = await _choose(client, item["decision_ref"], 1, {"Authorization": f"Bearer {_API_KEY}"})

    for answer in (missing, wrong):
        _assert_refused(answer, 401, _STORED_FAILURE, "gateway_auth_failed")
    _assert_refused(disabled, 404, _STORED_FAILURE, "owner_workspace_not_enabled")
    assert unchanged == before
    assert (control[0], control[1]["n"]) == (200, 1)
    assert [entry["n"] for entry in _observed(card)["chosen"]] == [1]


# ---------------------------------------------------------------------------
# Test 6: the record of past choices
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_6_recent_design_choices_lists_every_option_newest_first_capped_at_10(owner, monkeypatch):
    projects = [_project(owner, "Bakery Website"), _project(owner, "Bakery Menu")]
    slugs = {item["project_id"]: item["slug"] for item in ow.list_committed_projects(owner)}
    # Eleven designer cards across the two Projects, two or three options each.
    cards = [
        _designer_card(projects[index % 2], title=f"Design page {index + 1}", count=2 + index % 2)
        for index in range(11)
    ]
    start = int(time.time()) + 3600
    async with TestClient(TestServer(_app())) as client:
        refs = [await _ref(client, card) for card in cards]
        for index, (card, ref) in enumerate(zip(cards, refs)):
            with monkeypatch.context() as clock:
                clock.setattr(time, "time", lambda at=start + 60 * index: float(at))
                assert (await _choose(client, ref, 1 + index % card.count))[0] == 200
        _status, _raw, plain = await _feed(client)
        _status, _raw, listed = await _feed(client, "design_options_v1")

    assert "recent_design_choices" not in plain
    # Newest first and capped at 10, so the first choice is left out; each lists the chosen option and every
    # option not chosen, with its sentence and its picture.
    assert listed["recent_design_choices"] == [
        {
            "project_slug": slugs[card.project_id],
            "project_name": "Bakery Website" if index % 2 == 0 else "Bakery Menu",
            "question": QUESTION,
            "n": 1 + index % card.count,
            "chosen_at": _iso(start + 60 * index),
            "options": [
                {
                    "n": n,
                    "sentence": SENTENCES[n - 1],
                    "picture": {"attachment_id": card.attachment_ids[n - 1], "alt": ALTS[n - 1]},
                }
                for n in range(1, card.count + 1)
            ],
        }
        for index, card in reversed(list(enumerate(cards)))
    ][:10]
    assert len(listed["recent_design_choices"]) == 10


# ---------------------------------------------------------------------------
# Owner rule: a worker's environment
# ---------------------------------------------------------------------------


def test_under_a_workers_env_the_kernel_reaches_the_designer_cards_board_its_pictures_and_the_root_register(
    owner, monkeypatch,
):
    """The listed decision and the choice resolve the designer card's own board and its pictures by the store
    path the owner's feed resolved: under exactly the environment the dispatcher injects into a worker of another
    board they still read and write the card's own board under the root, the pictures they name are the card's own
    attachments there, and the write transaction still reads the root board register for that board."""
    import subprocess

    from gateway.platforms import api_server_owner_choices

    project = _project(owner, "Bakery Website")
    card = _designer_card(project)
    (listed_project,) = [item for item in ow.list_committed_projects(owner) if item["project_id"] == card.project_id]
    (item,) = ow.list_owner_decisions(owner, design_options=True)["data"]
    # The route resolves the ref in the owner's own environment, the gateway's, before anything is recorded.
    board, store, task_id = api_server_owner_choices._waiting_card(owner, item["decision_ref"])
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
        offered = odc.decision_design_options(reader, [task_id])
    finally:
        reader.close()
    with kb.connect_closing(store) as conn:
        assert odc.accept_choice(conn, task_id, 2, actor=owner.actor) == "released"
    reader = ow._open_read_only_sqlite(store, label="kanban.db")
    try:
        (entry,) = odc.recent_design_choices(reader, listed_project, [])
    finally:
        reader.close()

    # Independent count, read with plain sqlite and the folder listing under the root, never through the kernel.
    seen = _observed(card)
    assert offered == {task_id: item["design_options"]} and seen["declared"] == [item["design_options"]]
    folder = (root / "kanban" if board == kb.DEFAULT_BOARD else store.parent) / "attachments" / task_id
    with contextlib.closing(sqlite3.connect(store)) as conn:
        for option in item["design_options"]["options"]:
            (picture,) = option["pictures"]
            owned, stored_path, size = conn.execute(
                "SELECT task_id, stored_path, size FROM task_attachments WHERE id = ?", (int(picture["attachment_id"]),),
            ).fetchone()
            assert (owned, size) == (task_id, picture["bytes"])
            assert stored_path.startswith(str(folder.resolve())) and os.path.getsize(stored_path) == size
    assert [(entry["n"], entry["project_slug"]) for entry in [entry]] == [(2, listed_project["slug"])]
    assert [option["picture"]["attachment_id"] for option in entry["options"]] == card.attachment_ids
    assert [chosen["n"] for chosen in seen["chosen"]] == [2]
    assert seen["comments"] == [(owner.actor, f"The owner chose option 2: {SENTENCES[1]}")]
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
