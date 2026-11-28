"""The Decisions list serves the question of a stopped card to ``owner_answer_v1``.

Card P2 of the plan for how the owner answers a stopped card, with decisions
S2 to S4, S6 and S10 and section 5.1. ``list_owner_decisions(owner_answer=True)``
adds ``question`` to each owner-input item of a work card that waits on the
owner now, with no missing named input and no design options. A reader that
does not name the token gets exactly today's list. Each test builds a real
board in its own temporary home through the real kernel calls.
"""

from __future__ import annotations

import contextlib
import copy
import dataclasses
import hashlib
import json
import sqlite3
import threading
from urllib.parse import quote

import pytest
import yaml
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import PlatformConfig
from gateway.platforms.api_server import APIServerAdapter
from hermes_cli import owner_named_inputs as oni
from hermes_cli.config import get_config_path
from hermes_constants import get_hermes_home

from hermes_cli import kanban_db as kb
from hermes_cli import owner_questions as oq
from hermes_cli import owner_workspace as ow

_WORKER = "raphael-worker"
_FIXED = "Raphael needs your answer before this work can continue."
_QUESTION = "Which sign up page should Raphael build first?"
_CHOICES = [
    {"n": 1, "sentence": "Build the short form with one field."},
    {"n": 2, "sentence": "Build the long form with every field."},
]
_PLAIN = "Raphael needs your answer on the sign up page."
_PATH = "/srv/placeholder/config.yaml"


@pytest.fixture
def owner() -> ow.OwnerContext:
    return ow.resolve_owner_context()


def _project(owner: ow.OwnerContext, name: str = "Sign up") -> dict:
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(ow, "_confirm", lambda *_args, **_kwargs: {"approved": True})
        project = ow.bootstrap(owner, idempotency_key=f"setup-{name}", name=name)
    assert project["ok"] is True
    kb.write_board_dispatch_state(project["board"], dispatch_enabled=True)
    project["slug"] = next(
        item["slug"] for item in ow.list_committed_projects(owner)
        if str(item["project_id"]) == project["project_id"]
    )
    return project


def _board(project: dict):
    return contextlib.closing(kb.connect(board=project["board"]))


def _running(conn, project: dict, title: str) -> tuple[str, int]:
    task_id = kb.create_task(
        conn, title=title, assignee=_WORKER, project_id=project["project_id"],
    )
    claimed = kb.claim_task(conn, task_id)
    assert claimed is not None and claimed.status == "running"
    return task_id, claimed.current_run_id


def _stopped(conn, project: dict, title: str, *, kind: str = "needs_input", reason: str = _PLAIN) -> str:
    task_id, _run = _running(conn, project, title)
    assert kb.block_task(conn, task_id, reason=reason, kind=kind) is True
    return task_id


def _asked(
    conn, project: dict, title: str, *, stop: str = "question", question: str = _QUESTION,
    choices=None, words: str = "optional",
) -> str:
    task_id, run = _running(conn, project, title)
    oq.declare_owner_question(
        conn, task_id, stop, question, copy.deepcopy(_CHOICES if choices is None else choices), words,
        board=project["board"], expected_run_id=run,
    )
    return task_id


def _newest_blocked(conn, task_id: str) -> int:
    return max(event.id for event in kb.list_events(conn, task_id) if event.kind == "blocked")


def _stop_ref(owner: ow.OwnerContext, task_id: str, event_id: int) -> str:
    canonical = "\x00".join((owner.profile, task_id, str(event_id)))
    return "stop_" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:32]


def _item(owner, project: dict, task_id: str, **flags) -> dict:
    ref = ow._owner_decision_ref(owner, authority="task", native_id=task_id)
    return next(
        item for item in ow.list_owner_decisions(owner, **flags)["data"]
        if item["decision_ref"] == ref
    )


def _items(owner, project: dict, **flags) -> list[dict]:
    return [
        item for item in ow.list_owner_decisions(owner, **flags)["data"]
        if item["project_slug"] == project["slug"]
    ]


def _json(body: dict) -> str:
    return json.dumps(body, sort_keys=True)


def test_the_token_is_owner_answer_v1():
    assert ow.OWNER_ANSWER_CAPABILITY == "owner_answer_v1"


def test_without_the_token_the_list_equals_todays_on_the_whole_body(owner):
    project = _project(owner)
    with _board(project) as conn:
        plain = _stopped(conn, project, "Plain stop")
        asked = _asked(conn, project, "Asked stop")
    for flags in ({}, {"owner_waiting": True}):
        today = ow.list_owner_decisions(owner, **flags)
        answered = ow.list_owner_decisions(owner, owner_answer=True, **flags)
        assert "question" not in _json(today)
        stripped = copy.deepcopy(answered)
        for item, plain_item in zip(stripped["data"], today["data"]):
            if item.pop("question", None) is not None:
                # Its time is when the stop began, not the card's own (finding 2).
                item["created_at"] = plain_item["created_at"]
        # Same items, same order, same words, with only ``question`` and its time set apart.
        assert _json(stripped) == _json(today)
        assert [item["decision_ref"] for item in answered["data"]] == [
            item["decision_ref"] for item in today["data"]
        ]
    expected = {
        "decision_ref": ow._owner_decision_ref(owner, authority="task", native_id=plain),
        "authority": "task",
        "kind": "owner_input",
        "project_slug": project["slug"],
        "project_name": "Sign up",
        "title": "Plain stop",
        "reason": _FIXED,
        "created_at": _item(owner, project, plain)["created_at"],
    }
    assert _item(owner, project, plain) == expected
    assert _item(owner, project, asked)["reason"] == _FIXED


def test_a_declared_question_is_served_with_its_choices_and_stop_ref(owner):
    project = _project(owner)
    with _board(project) as conn:
        asked = _asked(conn, project, "Asked stop")
        limit = _asked(
            conn, project, "Limit stop", stop="review_limit", question="The reviewer asked for a shorter form.",
            choices=[], words="required",
        )
        event_id = _newest_blocked(conn, asked)
        limit_event = _newest_blocked(conn, limit)
    served = _item(owner, project, asked, owner_answer=True)
    assert served["question"] == {
        "object": "hermes.owner_workspace.owner_question",
        "stop": "question",
        "stop_ref": _stop_ref(owner, asked, event_id),
        "text": _QUESTION,
        "choices": _CHOICES,
        "words": "optional",
        "words_limit": 1000,
    }
    assert served["reason"] == _FIXED
    review = _item(owner, project, limit, owner_answer=True)["question"]
    assert review["stop"] == "review_limit"
    assert review["stop_ref"] == _stop_ref(owner, limit, limit_event)
    assert review["text"] == "The reviewer asked for a shorter form."
    assert review["choices"] == [
        {"n": 1, "sentence": "Repair it once more with the reviewer's corrections."},
        {"n": 2, "sentence": "Stop this work and release nothing."},
    ]
    assert review["words"] == "optional"
    assert served["question"]["stop_ref"].startswith("stop_")
    assert len(served["question"]["stop_ref"]) == 5 + 32


def test_a_plain_needs_input_stop_is_served_as_plain_with_words_required(owner):
    project = _project(owner)
    with _board(project) as conn:
        plain = _stopped(conn, project, "Plain stop")
        pathy = _stopped(conn, project, "Pathy stop", reason=f"Raphael cannot read {_PATH} yet.")
        capability = _stopped(conn, project, "Capability stop", kind="capability", reason=_PLAIN)
        event_id = _newest_blocked(conn, plain)
    served = _item(owner, project, plain, owner_answer=True)
    assert served["question"] == {
        "object": "hermes.owner_workspace.owner_question",
        "stop": "plain",
        "stop_ref": _stop_ref(owner, plain, event_id),
        "text": ow.owner_stop_reason(_PLAIN),
        "choices": [],
        "words": "required",
        "words_limit": 1000,
    }
    assert served["question"]["text"] == _PLAIN
    # The item's own words stay today's: only the question carries the reason.
    assert served["reason"] == _FIXED
    # A reason that fails the policy reads as the fixed sentence, as it does today.
    assert _item(owner, project, pathy, owner_answer=True)["question"]["text"] == _FIXED
    both = {"owner_answer": True, "owner_waiting": True}
    assert _item(owner, project, plain, **both)["question"]["text"] == _PLAIN
    assert _item(owner, project, plain, **both)["reason"] == _PLAIN
    current = _item(owner, project, capability, **both)
    assert current["question"]["stop"] == "plain"
    assert current["question"]["words"] == "required"
    assert current["question"]["text"] == _PLAIN


def test_a_stop_with_missing_inputs_or_design_options_gets_no_question(owner):
    project = _project(owner)
    with _board(project) as conn:
        needs_file, _run = _running(conn, project, "Needs a file")
        kb._append_event(conn, needs_file, "named_inputs_declared", {
            "inputs": [{"n": 1, "name": "price list", "type": "csv", "file": "price-list.csv"}],
        })
        assert kb.block_task(conn, needs_file, reason=_PLAIN, kind="needs_input") is True
        designing, _run = _running(conn, project, "Needs a choice")
        kb._append_event(conn, designing, "design_options_declared", {"question": _QUESTION, "options": []})
        assert kb.block_task(conn, designing, reason=_PLAIN, kind="needs_input") is True
        plain = _stopped(conn, project, "Plain stop")
    for flags in (
        {"owner_answer": True},
        {"owner_answer": True, "owner_input_upload": True, "design_options": True},
    ):
        assert "question" not in _item(owner, project, needs_file, **flags)
        assert "question" not in _item(owner, project, designing, **flags)
        assert _item(owner, project, plain, **flags)["question"]["stop"] == "plain"


def test_a_declaration_that_now_fails_the_policy_gets_no_question(owner):
    project = _project(owner)
    with _board(project) as conn:
        asked = _asked(conn, project, "Asked stop")
        assert "question" in _item(owner, project, asked, owner_answer=True)
        row = conn.execute(
            "SELECT id, payload FROM task_events WHERE task_id = ? AND kind = ?", (asked, oq.DECLARED_EVENT),
        ).fetchone()
        payload = json.loads(row["payload"])
        payload["question"] = f"Should Raphael edit {_PATH} first?"
        with kb.write_txn(conn):
            conn.execute("UPDATE task_events SET payload = ? WHERE id = ?", (json.dumps(payload), row["id"]))
    served = _item(owner, project, asked, owner_answer=True)
    # No question, and not a plain one either: the owner reads today's item.
    assert "question" not in served
    assert served["reason"] == _FIXED
    assert served == _item(owner, project, asked)


def test_a_declaration_with_a_changed_choice_set_gets_no_question(owner):
    project = _project(owner)
    with _board(project) as conn:
        asked = _asked(conn, project, "Asked stop", stop="review_limit", question="The reviewer asked for a fix.", choices=[])
        row = conn.execute(
            "SELECT id, payload FROM task_events WHERE task_id = ? AND kind = ?", (asked, oq.DECLARED_EVENT),
        ).fetchone()
        payload = json.loads(row["payload"])
        payload["choices"] = payload["choices"][:1]
        with kb.write_txn(conn):
            conn.execute("UPDATE task_events SET payload = ? WHERE id = ?", (json.dumps(payload), row["id"]))
    assert "question" not in _item(owner, project, asked, owner_answer=True)


def test_stop_ref_changes_when_the_card_is_blocked_again_and_differs_between_owners(owner):
    project = _project(owner)
    with _board(project) as conn:
        task_id = _stopped(conn, project, "Plain stop")
        first = _item(owner, project, task_id, owner_answer=True)["question"]["stop_ref"]
        assert _item(owner, project, task_id, owner_answer=True)["question"]["stop_ref"] == first
        assert kb.unblock_task(conn, task_id) is True
        claimed = kb.claim_task(conn, task_id)
        assert claimed is not None
        assert kb.block_task(conn, task_id, reason=_PLAIN, kind="needs_input") is True
    second = _item(owner, project, task_id, owner_answer=True)["question"]["stop_ref"]
    assert second != first
    other = dataclasses.replace(owner, profile=owner.profile + "-other")
    with _board(project) as conn:
        mine = oq.decision_questions(conn, [task_id], owner)[task_id]["stop_ref"]
        theirs = oq.decision_questions(conn, [task_id], other)[task_id]["stop_ref"]
        event_id = _newest_blocked(conn, task_id)
    assert mine == second
    assert theirs != second
    assert theirs == _stop_ref(other, task_id, event_id)


def test_a_resumed_card_has_no_question_until_it_stops_again(owner):
    project = _project(owner)
    with _board(project) as conn:
        asked = _asked(conn, project, "Asked stop")
        assert kb.unblock_task(conn, asked) is True
    assert ow._owner_decision_ref(owner, authority="task", native_id=asked) not in [
        item["decision_ref"] for item in _items(owner, project, owner_answer=True)
    ]
    with _board(project) as conn:
        claimed = kb.claim_task(conn, asked)
        assert claimed is not None
        assert kb.block_task(conn, asked, reason=_PLAIN, kind="needs_input") is True
    # The old declaration no longer counts: the new stop is a plain one.
    assert _item(owner, project, asked, owner_answer=True)["question"]["stop"] == "plain"


def test_many_stopped_cards_each_get_their_own_stop_ref(owner):
    project = _project(owner)
    with _board(project) as conn:
        ids = [_stopped(conn, project, f"Stop {n}") for n in range(6)]
    served = _items(owner, project, owner_answer=True)
    assert {item["title"] for item in served} == {f"Stop {n}" for n in range(6)}
    assert all(item["question"]["stop"] == "plain" for item in served)
    assert len({item["question"]["stop_ref"] for item in served}) == len(ids)


# The served route: the repair of the four review findings of card P2.

_OLD = "Raphael cannot reach the placeholder calendar yet."
_PRICES = b"sku,price\nA,1\n"
_LATER = 600


@pytest.fixture
def served(monkeypatch) -> ow.OwnerContext:
    monkeypatch.setattr("gateway.run._hermes_home", get_hermes_home())
    path = get_config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump({"gateway": {"api_server": {"owner_workspace": {"enabled": True}}}}), encoding="utf-8",
    )
    return ow.resolve_owner_context()


def _app() -> web.Application:
    adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={}))
    app = web.Application()
    for method, path, handler in adapter._http_route_table():
        if method == "GET" and path == "/v1/owner-workspace/decisions":
            app.router.add_route(method, path, handler)
    return app


async def _served(*tokens: str) -> list[dict]:
    """The Decisions list the route serves for one ``capabilities`` parameter."""
    url = "/v1/owner-workspace/decisions"
    if tokens:
        url += "?capabilities=" + quote(",".join(tokens), safe=",")
    async with TestClient(TestServer(_app())) as client:
        response = await client.get(url)
        assert response.status == 200, await response.text()
        return (await response.json())["data"]


def _served_item(data: list[dict], owner, task_id: str) -> dict | None:
    ref = ow._owner_decision_ref(owner, authority="task", native_id=task_id)
    return next((item for item in data if item["decision_ref"] == ref), None)


def _event_time(conn, event_id: int) -> int:
    return int(conn.execute("SELECT created_at FROM task_events WHERE id = ?", (event_id,)).fetchone()[0])


def _shift(conn, sql: str, *params) -> None:
    with kb.write_txn(conn):
        conn.execute(sql, params)


@pytest.mark.asyncio
async def test_a_current_capability_stop_is_served_with_only_owner_answer_v1(served):
    owner = served
    project = _project(owner)
    with _board(project) as conn:
        capability = _stopped(conn, project, "Capability stop", kind="capability", reason=_PLAIN)
        event_id = _newest_blocked(conn, capability)
        _shift(conn, "UPDATE tasks SET created_at = created_at - ? WHERE id = ?", _LATER, capability)
        stopped_at = _event_time(conn, event_id)
    item = _served_item(await _served("owner_answer_v1"), owner, capability)
    assert item is not None
    assert item["kind"] == "owner_input"
    assert item["reason"] == _FIXED
    assert item["created_at"] == ow._owner_timestamp(stopped_at)
    assert item["question"] == {
        "object": "hermes.owner_workspace.owner_question",
        "stop": "plain",
        "stop_ref": _stop_ref(owner, capability, event_id),
        "text": _PLAIN,
        "choices": [],
        "words": "required",
        "words_limit": 1000,
    }
    # Without the answer token a capability stop still joins only for owner_waiting_v1.
    for tokens in ((), ("owner_input_upload_v1",), ("design_options_v1",)):
        assert _served_item(await _served(*tokens), owner, capability) is None
    waiting = _served_item(await _served("owner_waiting_v1"), owner, capability)
    assert waiting is not None and "question" not in waiting and waiting["reason"] == _PLAIN


@pytest.mark.asyncio
async def test_created_at_of_a_question_is_when_the_stop_began_for_owner_answer_v1_alone(served):
    owner = served
    project = _project(owner)
    with _board(project) as conn:
        plain = _stopped(conn, project, "Plain stop")
        asked = _asked(conn, project, "Asked stop")
        stops = {task_id: _newest_blocked(conn, task_id) for task_id in (plain, asked)}
        for task_id in stops:
            _shift(conn, "UPDATE tasks SET created_at = created_at - ? WHERE id = ?", _LATER, task_id)
        began = {task_id: _event_time(conn, event_id) for task_id, event_id in stops.items()}
        created = {
            task_id: int(conn.execute("SELECT created_at FROM tasks WHERE id = ?", (task_id,)).fetchone()[0])
            for task_id in stops
        }
    for task_id in stops:
        assert created[task_id] != began[task_id]
        answered = _served_item(await _served("owner_answer_v1"), owner, task_id)
        assert answered["created_at"] == ow._owner_timestamp(began[task_id])
        both = _served_item(await _served("owner_answer_v1", "owner_waiting_v1"), owner, task_id)
        assert both["created_at"] == ow._owner_timestamp(began[task_id])
        # The legacy paths keep their own times.
        assert _served_item(await _served(), owner, task_id)["created_at"] == ow._owner_timestamp(created[task_id])
        waiting = _served_item(await _served("owner_waiting_v1"), owner, task_id)
        assert waiting["created_at"] == ow._owner_timestamp(began[task_id])
    # An item with no question keeps the card's creation time.
    with _board(project) as conn:
        row = conn.execute(
            "SELECT id, payload FROM task_events WHERE task_id = ? AND kind = ?", (asked, oq.DECLARED_EVENT),
        ).fetchone()
        payload = json.loads(row["payload"])
        payload["question"] = f"Should Raphael edit {_PATH} first?"
        _shift(conn, "UPDATE task_events SET payload = ? WHERE id = ?", json.dumps(payload), row["id"])
    unasked = _served_item(await _served("owner_answer_v1"), owner, asked)
    assert "question" not in unasked
    assert unasked["created_at"] == ow._owner_timestamp(created[asked])


@pytest.mark.asyncio
async def test_a_new_question_after_the_last_named_input_is_served(served):
    owner = served
    project = _project(owner)
    with _board(project) as conn:
        card, _run = _running(conn, project, "Needs a file")
        kb._append_event(conn, card, "named_inputs_declared", {
            "inputs": [{"n": 1, "name": "price list", "type": "csv", "file": "price-list.csv"}],
        })
        assert kb.block_task(conn, card, reason=_PLAIN, kind="needs_input") is True
    both = ("owner_answer_v1", "owner_input_upload_v1")
    held = _served_item(await _served(*both), owner, card)
    assert [entry["n"] for entry in held["missing_inputs"]] == [1] and "question" not in held
    with _board(project) as conn:
        assert oni.accept_upload(conn, card, 1, _PRICES, "text/csv", board=project["board"]) == "released"
        claimed = kb.claim_task(conn, card)
        assert claimed is not None
        oq.declare_owner_question(
            conn, card, "question", _QUESTION, copy.deepcopy(_CHOICES), "optional",
            board=project["board"], expected_run_id=claimed.current_run_id,
        )
        event_id = _newest_blocked(conn, card)
        assert oni.missing_inputs(conn, card) == []
    for tokens in (("owner_answer_v1",), both, ("owner_answer_v1", "design_options_v1")):
        item = _served_item(await _served(*tokens), owner, card)
        assert item["question"]["stop"] == "question"
        assert item["question"]["text"] == _QUESTION
        assert item["question"]["stop_ref"] == _stop_ref(owner, card, event_id)
    # The upload fields the reader already got stay as they were.
    assert _served_item(await _served(*both), owner, card)["missing_inputs"] == []
    assert "missing_inputs" not in _served_item(await _served("owner_answer_v1"), owner, card)


@pytest.mark.asyncio
async def test_one_item_never_mixes_two_stops_when_the_card_stops_again_between_reads(served, monkeypatch):
    owner = served
    project = _project(owner)
    with _board(project) as conn:
        card = _stopped(conn, project, "Plain stop", reason=_OLD)
        old_event = _newest_blocked(conn, card)
        _shift(conn, "UPDATE task_events SET created_at = created_at - ? WHERE id = ?", _LATER, old_event)
        old_at = _event_time(conn, old_event)
    fired: list[threading.Thread] = []
    failures: list[BaseException] = []
    real = ow.decision_missing_inputs

    def resume_claim_declare() -> None:
        try:
            with _board(project) as other:
                assert kb.unblock_task(other, card) is True
                claimed = kb.claim_task(other, card)
                assert claimed is not None
                oq.declare_owner_question(
                    other, card, "question", _QUESTION, copy.deepcopy(_CHOICES), "optional",
                    board=project["board"], expected_run_id=claimed.current_run_id,
                )
        except BaseException as exc:  # reported by the test thread
            failures.append(exc)

    def stops_again(conn, task_ids):
        if not fired:
            # Another connection commits between two reads of the request. Under a rollback
            # journal that commit waits for the request's read to end, so it runs on its own
            # thread: a read that is not one snapshot has seen it by the time this returns.
            writer = threading.Thread(target=resume_claim_declare)
            fired.append(writer)
            writer.start()
            writer.join(timeout=3)
        return real(conn, task_ids)

    monkeypatch.setattr(ow, "decision_missing_inputs", stops_again)
    tokens = ("owner_answer_v1", "owner_waiting_v1")
    item = _served_item(await _served(*tokens), owner, card)
    assert len(fired) == 1
    fired[0].join(timeout=60)
    assert not fired[0].is_alive() and failures == []
    with _board(project) as conn:
        new_event = _newest_blocked(conn, card)
        new_at = _event_time(conn, new_event)
    assert new_event != old_event and new_at != old_at
    old = {
        "reason": _OLD, "text": _OLD, "stop": "plain", "stop_ref": _stop_ref(owner, card, old_event),
        "created_at": ow._owner_timestamp(old_at),
    }
    new = {
        "reason": _QUESTION, "text": _QUESTION, "stop": "question", "stop_ref": _stop_ref(owner, card, new_event),
        "created_at": ow._owner_timestamp(new_at),
    }
    seen = {
        "reason": item["reason"], "text": item["question"]["text"], "stop": item["question"]["stop"],
        "stop_ref": item["question"]["stop_ref"], "created_at": item["created_at"],
    }
    assert seen in (old, new)
    # The next read sees the new stop whole.
    item = _served_item(await _served(*tokens), owner, card)
    assert (item["reason"], item["question"]["text"], item["question"]["stop_ref"]) == (
        new["reason"], new["text"], new["stop_ref"],
    )


# Card P5: the owner continues a briefly stopped card.

_BRIEF = "The network dropped for a moment."
_BRIEF_TEXT = "This work stopped for a short-lived reason. Should Raphael continue it?"
_CONTINUE = [{"n": 1, "sentence": "Continue the work."}]
_ANSWERING = ({"owner_answer": True}, {"owner_answer": True, "owner_waiting": True})
_NOT_ANSWERING = ({}, {"owner_waiting": True}, {"owner_input_upload": True}, {"design_options": True})
_NOT_BRIEF = {
    "released": lambda conn, card: kb.unblock_task(conn, card),
    "claimed again": lambda conn, card: (kb.unblock_task(conn, card), kb.claim_task(conn, card)),
    "triage": lambda conn, card: _shift(conn, "UPDATE tasks SET status = 'triage' WHERE id = ?", card),
    "done": lambda conn, card: _shift(conn, "UPDATE tasks SET status = 'done' WHERE id = ?", card),
    "control card": lambda conn, card: _shift(conn, "UPDATE tasks SET task_kind = 'control' WHERE id = ?", card),
    "recommendation card": lambda conn, card: _shift(
        conn, "UPDATE tasks SET task_kind = 'recommendation' WHERE id = ?", card,
    ),
    "no block kind": lambda conn, card: _shift(conn, "UPDATE tasks SET block_kind = NULL WHERE id = ?", card),
    "dependency": lambda conn, card: _shift(conn, "UPDATE tasks SET block_kind = 'dependency' WHERE id = ?", card),
    "no stop event": lambda conn, card: _shift(
        conn, "DELETE FROM task_events WHERE task_id = ? AND kind = 'blocked'", card,
    ),
}


def _briefly(conn, project: dict, title: str, *, reason: str = _BRIEF) -> str:
    return _stopped(conn, project, title, kind="transient", reason=reason)


def _gave_up(conn, project: dict, title: str) -> str:
    """A card the kernel blocked as transient and then gave up on: still blocked, its block kind kept."""
    task_id = _briefly(conn, project, title)
    assert kb.unblock_task(conn, task_id) is True
    assert kb.claim_task(conn, task_id) is not None
    kb._record_spawn_failure(conn, task_id, "spawn failed", failure_limit=1)
    status, block_kind = conn.execute("SELECT status, block_kind FROM tasks WHERE id = ?", (task_id,)).fetchone()
    assert (status, block_kind) == ("blocked", "transient")
    newest = conn.execute(
        "SELECT kind FROM task_events WHERE task_id = ? AND kind IN ('blocked', 'unblocked', 'gave_up') "
        "ORDER BY id DESC LIMIT 1", (task_id,),
    ).fetchone()[0]
    assert newest == "gave_up"
    return task_id


def _brief_question(owner, task_id: str, event_id: int) -> dict:
    return {
        "object": "hermes.owner_workspace.owner_question",
        "stop": "stopped_briefly",
        "stop_ref": _stop_ref(owner, task_id, event_id),
        "text": _BRIEF_TEXT,
        "choices": copy.deepcopy(_CONTINUE),
        "words": "optional",
        "words_limit": 1000,
    }


def _current_ref(owner, conn, task_id: str) -> str:
    return _stop_ref(owner, task_id, _newest_blocked(conn, task_id))


def _record(conn, task_id: str) -> dict:
    """What the board holds of one card: its stop, the answers and comments on it, and how often it was resumed."""
    status, block_kind = conn.execute("SELECT status, block_kind FROM tasks WHERE id = ?", (task_id,)).fetchone()
    return {
        "status": status,
        "block_kind": block_kind,
        "answers": [
            json.loads(payload) for (payload,) in conn.execute(
                "SELECT payload FROM task_events WHERE task_id = ? AND kind = ? ORDER BY id",
                (task_id, oq.ANSWERED_EVENT),
            ).fetchall()
        ],
        "comments": [
            (author, body) for author, body in conn.execute(
                "SELECT author, body FROM task_comments WHERE task_id = ? ORDER BY id", (task_id,),
            ).fetchall()
        ],
        "unblocked": conn.execute(
            "SELECT COUNT(*) FROM task_events WHERE task_id = ? AND kind = 'unblocked'", (task_id,),
        ).fetchone()[0],
    }


def _written(conn) -> tuple[int, int]:
    """How many events and how many comments the whole board holds."""
    return (
        conn.execute("SELECT COUNT(*) FROM task_events").fetchone()[0],
        conn.execute("SELECT COUNT(*) FROM task_comments").fetchone()[0],
    )


def _found(owner, task_id: str, **flags) -> dict | None:
    ref = ow._owner_decision_ref(owner, authority="task", native_id=task_id)
    return next((item for item in ow.list_owner_decisions(owner, **flags)["data"] if item["decision_ref"] == ref), None)


def test_a_current_brief_stop_is_served_with_the_question_to_continue_and_the_time_of_its_stop(owner):
    project = _project(owner)
    with _board(project) as conn:
        brief = _briefly(conn, project, "Brief stop")
        event_id = _newest_blocked(conn, brief)
        _shift(conn, "UPDATE tasks SET created_at = created_at - ? WHERE id = ?", _LATER, brief)
        stopped_at = _event_time(conn, event_id)
        created = int(conn.execute("SELECT created_at FROM tasks WHERE id = ?", (brief,)).fetchone()[0])
    assert created != stopped_at
    expected = {
        "decision_ref": ow._owner_decision_ref(owner, authority="task", native_id=brief),
        "authority": "task",
        "kind": "owner_input",
        "project_slug": project["slug"],
        "project_name": "Sign up",
        "title": "Brief stop",
        "reason": _FIXED,
        "created_at": ow._owner_timestamp(stopped_at),
        "question": _brief_question(owner, brief, event_id),
    }
    for flags in _ANSWERING:
        assert _found(owner, brief, **flags) == expected
        assert _found(owner, brief, **flags) == expected  # the same stop_ref on every read
        assert [item["title"] for item in _items(owner, project, **flags)].count("Brief stop") == 1
    assert expected["question"]["stop_ref"].startswith("stop_") and len(expected["question"]["stop_ref"]) == 5 + 32


def test_brief_stops_of_two_projects_are_each_served_once_under_their_own_project(owner):
    first, second = _project(owner), _project(owner, "Checkout")
    with _board(first) as conn:
        _briefly(conn, first, "Brief one")
    with _board(second) as conn:
        _briefly(conn, second, "Brief two")
        _briefly(conn, second, "Brief three")
    for flags in _ANSWERING:
        served = [item for item in ow.list_owner_decisions(owner, **flags)["data"] if item["title"].startswith("Brief")]
        assert sorted((item["title"], item["project_slug"]) for item in served) == sorted([
            ("Brief one", first["slug"]), ("Brief two", second["slug"]), ("Brief three", second["slug"]),
        ])
        assert all(item["question"]["stop"] == "stopped_briefly" for item in served)
        assert len({item["question"]["stop_ref"] for item in served}) == 3


def test_decision_questions_serves_a_brief_stop_beside_the_stops_it_served_before(owner):
    project = _project(owner)
    with _board(project) as conn:
        brief = _briefly(conn, project, "Brief stop")
        plain = _stopped(conn, project, "Plain stop")
        asked = _asked(conn, project, "Asked stop")
        capability = _stopped(conn, project, "Capability stop", kind="capability")
        cards = [brief, plain, asked, capability]
        refs = {card: _current_ref(owner, conn, card) for card in cards}
        brief_event = _newest_blocked(conn, brief)
        served = oq.decision_questions(conn, cards, owner)
        alone = oq.decision_questions(conn, [brief], owner)
        nothing = (oq.decision_questions(conn, [], owner), oq.decision_questions(conn, ["t_missing"], owner))
    assert {card: question["stop"] for card, question in served.items()} == {
        brief: "stopped_briefly", plain: "plain", asked: "question", capability: "plain",
    }
    assert {card: question["stop_ref"] for card, question in served.items()} == refs
    assert served[brief] == _brief_question(owner, brief, brief_event)
    assert alone == {brief: _brief_question(owner, brief, brief_event)}
    assert nothing == ({}, {})


def test_the_list_of_a_reader_that_does_not_name_owner_answer_v1_is_the_same_with_a_brief_stop(owner):
    project = _project(owner)
    with _board(project) as conn:
        _stopped(conn, project, "Plain stop")
        _stopped(conn, project, "Capability stop", kind="capability")
        brief, _run = _running(conn, project, "Brief stop")
    before = [_json(ow.list_owner_decisions(owner, **flags)) for flags in _NOT_ANSWERING]
    with _board(project) as conn:
        assert kb.block_task(conn, brief, reason=_BRIEF, kind="transient") is True
    after = [_json(ow.list_owner_decisions(owner, **flags)) for flags in _NOT_ANSWERING]
    assert after == before
    assert all("Brief stop" not in body for body in after)
    # Control: the same card, read with the token, is served.
    assert _found(owner, brief, owner_answer=True)["question"]["stop"] == "stopped_briefly"


@pytest.mark.parametrize("ended", ["unblocked", "gave_up", *kb._STOP_SUPERSEDING_EVENT_KINDS])
def test_a_brief_stop_that_a_later_event_ended_is_not_served_and_takes_no_answer(owner, ended):
    project = _project(owner)
    with _board(project) as conn:
        control = _briefly(conn, project, "Current stop")
        card = _briefly(conn, project, "Ended stop")
        kb._append_event(conn, card, ended)
        stale = _current_ref(owner, conn, card)
        before = (_record(conn, card), _written(conn))
        assert oq.decision_questions(conn, [card], owner) == {}
        assert oq.accept_answer(conn, card, stale, 1, None, owner=owner) == "not_waiting"
        assert (_record(conn, card), _written(conn)) == before
    for flags in _ANSWERING:
        titles = [item["title"] for item in _items(owner, project, **flags)]
        assert "Current stop" in titles
        assert "Ended stop" not in titles
    assert _found(owner, control, owner_answer=True)["question"]["stop"] == "stopped_briefly"


def test_a_brief_stop_the_kernel_gave_up_on_is_not_served_and_takes_no_answer(owner):
    project = _project(owner)
    with _board(project) as conn:
        control = _briefly(conn, project, "Current stop")
        gone = _gave_up(conn, project, "Given up")
        stale = _current_ref(owner, conn, gone)
        before = (_record(conn, gone), _written(conn))
        assert oq.decision_questions(conn, [gone], owner) == {}
        assert oq.accept_answer(conn, gone, stale, 1, None, owner=owner) == "not_waiting"
        assert (_record(conn, gone), _written(conn)) == before
    for flags in _ANSWERING:
        titles = [item["title"] for item in _items(owner, project, **flags)]
        assert "Given up" not in titles
        assert "Current stop" in titles
    assert _found(owner, control, owner_answer=True) is not None


@pytest.mark.parametrize("earlier", kb._STOP_SUPERSEDING_EVENT_KINDS)
def test_an_event_that_came_before_the_stop_does_not_end_it(owner, earlier):
    project = _project(owner)
    with _board(project) as conn:
        card, _run = _running(conn, project, "Brief stop")
        kb._append_event(conn, card, earlier)
        assert kb.block_task(conn, card, reason=_BRIEF, kind="transient") is True
        event_id = _newest_blocked(conn, card)
    for flags in _ANSWERING:
        assert _found(owner, card, **flags)["question"] == _brief_question(owner, card, event_id)


@pytest.mark.parametrize(
    "tail",
    [(), ("unblocked",), ("gave_up",), ("unblocked", "blocked"), ("p5_marker",),
     *((kind,) for kind in kb._STOP_SUPERSEDING_EVENT_KINDS)],
    ids=lambda tail: "+".join(tail) or "no-event",
)
def test_a_brief_stop_ends_exactly_when_the_kernel_ends_a_needs_input_stop(owner, monkeypatch, tail):
    monkeypatch.setattr(kb, "_STOP_SUPERSEDING_EVENT_KINDS", (*kb._STOP_SUPERSEDING_EVENT_KINDS, "p5_marker"))
    project = _project(owner)
    with _board(project) as conn:
        cards = {}
        for kind in ("needs_input", "transient"):
            cards[kind] = _stopped(conn, project, f"{kind} card", kind=kind)
            for event in tail:
                kb._append_event(conn, cards[kind], event)
        waiting = cards["needs_input"] in kb.task_owner_waits(conn, [cards["needs_input"]])
        brief = cards["transient"] in oq.decision_questions(conn, [cards["transient"]], owner)
    assert waiting == (tail in ((), ("unblocked", "blocked")))
    assert brief == waiting


@pytest.mark.parametrize("state", sorted(_NOT_BRIEF))
def test_a_card_that_is_not_a_brief_stop_is_never_served_as_one(owner, state):
    project = _project(owner)
    with _board(project) as conn:
        card = _briefly(conn, project, "Not a brief stop")
        stop_ref = _current_ref(owner, conn, card)
        _NOT_BRIEF[state](conn, card)
        before = (_record(conn, card), _written(conn))
        served = oq.decision_questions(conn, [card], owner).get(card)
        assert served is None or served["stop"] != "stopped_briefly"
        assert oq.accept_answer(conn, card, stop_ref, 1, None, owner=owner) == "not_waiting"
        assert (_record(conn, card), _written(conn)) == before
    for flags in _ANSWERING:
        item = _found(owner, card, **flags)
        assert item is None or item.get("question", {}).get("stop") != "stopped_briefly"


def test_choice_one_resumes_a_brief_stop_once_with_one_event_and_one_comment(owner):
    project = _project(owner)
    with _board(project) as conn:
        brief = _briefly(conn, project, "Brief stop")
        other = _briefly(conn, project, "Other stop")
        stop_ref = _current_ref(owner, conn, brief)
        before, other_before = _record(conn, brief), _record(conn, other)
        assert (before["status"], before["block_kind"], before["answers"], before["comments"], before["unblocked"]) == (
            "blocked", "transient", [], [], 0,
        )
        assert oq.accept_answer(conn, brief, stop_ref, 1, None, owner=owner) == "resumed"
        after = _record(conn, brief)
        assert (after["status"], after["unblocked"]) == ("ready", 1)
        (answer,) = after["answers"]
        assert set(answer) == {"stop_ref", "choice", "words_length", "actor", "answered_at"}
        assert (answer["stop_ref"], answer["choice"], answer["words_length"], answer["actor"]) == (
            stop_ref, 1, 0, owner.actor,
        )
        assert after["comments"] == [("owner", f"The owner answered: {_BRIEF_TEXT}\nChoice 1: Continue the work.")]
        assert _record(conn, other) == other_before


@pytest.mark.parametrize("failing, written", [("add_comment", (1, 0)), ("_unblock_task_within_txn", (1, 1))])
def test_a_write_that_fails_after_the_event_is_written_leaves_the_brief_stop_untouched(owner, failing, written):
    project = _project(owner)
    with _board(project) as conn:
        brief = _briefly(conn, project, "Brief stop")
        control = _briefly(conn, project, "Control stop")
        stop_ref = _current_ref(owner, conn, brief)
        control_ref = _current_ref(owner, conn, control)
        before = (_record(conn, brief), _written(conn))
    seen = []

    def broken(conn, *_args, **_kwargs):
        seen.append((
            conn.execute("SELECT COUNT(*) FROM task_events WHERE kind = ?", (oq.ANSWERED_EVENT,)).fetchone()[0],
            conn.execute("SELECT COUNT(*) FROM task_comments").fetchone()[0],
        ))
        raise RuntimeError("the write failed")

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(kb, failing, broken)
        with _board(project) as conn:
            with pytest.raises(RuntimeError, match="the write failed"):
                oq.accept_answer(conn, brief, stop_ref, 1, "Go on.", owner=owner)
            assert not conn.in_transaction
    # The event, and the comment for the second case, were in the transaction when it failed.
    assert seen == [written]
    with _board(project) as conn:
        assert (_record(conn, brief), _written(conn)) == before
    for flags in _ANSWERING:
        assert _found(owner, brief, **flags)["question"]["stop_ref"] == stop_ref
    with _board(project) as conn:
        assert oq.accept_answer(conn, control, control_ref, 1, None, owner=owner) == "resumed"
        assert oq.accept_answer(conn, brief, stop_ref, 1, "Go on.", owner=owner) == "resumed"
        record = _record(conn, brief)
    assert (len(record["answers"]), len(record["comments"]), record["unblocked"]) == (1, 1, 1)


def test_choice_one_with_words_is_accepted_and_the_words_are_in_the_comment(owner):
    words = "Bitte weiter — 続けてください 🙂\nZweite Zeile"
    project = _project(owner)
    with _board(project) as conn:
        brief = _briefly(conn, project, "Brief stop")
        stop_ref = _current_ref(owner, conn, brief)
        assert oq.accept_answer(conn, brief, stop_ref, 1, words, owner=owner) == "resumed"
        record = _record(conn, brief)
    assert record["comments"] == [
        ("owner", f"The owner answered: {_BRIEF_TEXT}\nChoice 1: Continue the work.\n{words}"),
    ]
    assert [answer["words_length"] for answer in record["answers"]] == [len(words)]
    assert (record["status"], record["unblocked"]) == ("ready", 1)


@pytest.mark.parametrize("words", [None, "Please go on.", "Bitte weiter 🙂 — 続けて\nZweite Zeile", "x" * 1000])
def test_words_with_no_choice_give_not_allowed_and_nothing_is_written(owner, words):
    project = _project(owner)
    with _board(project) as conn:
        brief = _briefly(conn, project, "Brief stop")
        stop_ref = _current_ref(owner, conn, brief)
        before = (_record(conn, brief), _written(conn))
        assert oq.accept_answer(conn, brief, stop_ref, None, words, owner=owner) == "not_allowed"
        assert (_record(conn, brief), _written(conn)) == before
    assert _found(owner, brief, owner_answer=True)["question"]["stop_ref"] == stop_ref


def test_a_wrong_stop_ref_a_choice_not_offered_and_a_second_answer_all_give_not_waiting(owner):
    project = _project(owner)
    with _board(project) as conn:
        brief = _briefly(conn, project, "Brief stop")
        other = _briefly(conn, project, "Other stop")
        event_id = _newest_blocked(conn, brief)
        right = _stop_ref(owner, brief, event_id)
        stranger = dataclasses.replace(owner, profile=owner.profile + "-other")
        wrong = [
            "stop_" + "0" * 32, "", _stop_ref(stranger, brief, event_id), _stop_ref(owner, brief, event_id + 1),
            _current_ref(owner, conn, other),
        ]
        before = (_record(conn, brief), _written(conn))
        for stop_ref in wrong:
            assert oq.accept_answer(conn, brief, stop_ref, 1, None, owner=owner) == "not_waiting"
            assert oq.accept_answer(conn, brief, stop_ref, None, "Please go on.", owner=owner) == "not_waiting"
        for choice in (2, 3, 4, 0, -1, 5):
            assert oq.accept_answer(conn, brief, right, choice, None, owner=owner) == "not_waiting"
            assert oq.accept_answer(conn, brief, right, choice, "Please go on.", owner=owner) == "not_waiting"
        assert oq.accept_answer(conn, "t_missing", right, 1, None, owner=owner) == "not_waiting"
        assert oq.accept_answer(conn, brief, right, 1, None, owner=dataclasses.replace(owner, profile="x")) == "not_waiting"
        assert (_record(conn, brief), _written(conn)) == before
        assert oq.accept_answer(conn, brief, right, 1, None, owner=owner) == "resumed"
        settled = (_record(conn, brief), _written(conn))
        assert oq.accept_answer(conn, brief, right, 1, "Again.", owner=owner) == "not_waiting"
        assert oq.accept_answer(conn, brief, right, 1, None, owner=owner) == "not_waiting"
        assert (_record(conn, brief), _written(conn)) == settled
        assert len(settled[0]["answers"]) == 1


def test_after_the_answer_the_brief_stop_is_not_served_again(owner):
    project = _project(owner)
    with _board(project) as conn:
        brief = _briefly(conn, project, "Brief stop")
        control = _briefly(conn, project, "Control stop")
        stop_ref = _current_ref(owner, conn, brief)
        assert _found(owner, brief, owner_answer=True) is not None
        assert oq.accept_answer(conn, brief, stop_ref, 1, None, owner=owner) == "resumed"
        assert oq.decision_questions(conn, [brief], owner) == {}
    for flags in _ANSWERING:
        assert _found(owner, brief, **flags) is None
        assert _found(owner, control, **flags) is not None
    for flags in _NOT_ANSWERING:
        assert _found(owner, brief, **flags) is None


def test_a_brief_stop_with_an_answer_on_record_is_left_out_while_the_card_stays_blocked(owner):
    project = _project(owner)
    with _board(project) as conn:
        brief = _briefly(conn, project, "Brief stop")
        control = _briefly(conn, project, "Control stop")
        stop_ref = _current_ref(owner, conn, brief)
        kb._append_event(conn, brief, oq.ANSWERED_EVENT, {
            "stop_ref": stop_ref, "choice": 1, "words_length": 0, "actor": owner.actor,
            "answered_at": ow._owner_timestamp(0),
        })
        before = (_record(conn, brief), _written(conn))
        assert oq.decision_questions(conn, [brief], owner) == {}
        assert oq.accept_answer(conn, brief, stop_ref, 1, None, owner=owner) == "not_waiting"
        assert (_record(conn, brief), _written(conn)) == before
        assert before[0]["status"] == "blocked"
    for flags in _ANSWERING:
        assert _found(owner, brief, **flags) is None
        assert _found(owner, control, **flags) is not None


def test_a_card_that_stops_briefly_again_after_its_resume_gets_a_new_stop_ref_and_is_served_again(owner):
    project = _project(owner)
    with _board(project) as conn:
        card = _briefly(conn, project, "Brief stop")
        first_event = _newest_blocked(conn, card)
        first_ref = _stop_ref(owner, card, first_event)
        assert oq.accept_answer(conn, card, first_ref, 1, None, owner=owner) == "resumed"
        # The kernel sends a card that stops twice for one kind in a row to triage, so another kind comes between.
        assert kb.claim_task(conn, card) is not None
        assert kb.block_task(conn, card, reason=_PLAIN, kind="needs_input") is True
        assert kb.unblock_task(conn, card) is True
        assert kb.claim_task(conn, card) is not None
        assert kb.block_task(conn, card, reason=_BRIEF, kind="transient") is True
        assert tuple(conn.execute("SELECT status, block_kind FROM tasks WHERE id = ?", (card,)).fetchone()) == (
            "blocked", "transient",
        )
        second_event = _newest_blocked(conn, card)
        _shift(conn, "UPDATE tasks SET created_at = created_at - ? WHERE id = ?", 2 * _LATER, card)
        _shift(conn, "UPDATE task_events SET created_at = created_at - ? WHERE id = ?", _LATER, first_event)
        second_at = _event_time(conn, second_event)
    second_ref = _stop_ref(owner, card, second_event)
    assert second_event != first_event and second_ref != first_ref
    for flags in _ANSWERING:
        item = _found(owner, card, **flags)
        assert item["question"] == _brief_question(owner, card, second_event)
        assert item["created_at"] == ow._owner_timestamp(second_at)
    with _board(project) as conn:
        before = (_record(conn, card), _written(conn))
        assert oq.accept_answer(conn, card, first_ref, 1, None, owner=owner) == "not_waiting"
        assert (_record(conn, card), _written(conn)) == before
        assert oq.accept_answer(conn, card, second_ref, 1, "Go on again.", owner=owner) == "resumed"
        record = _record(conn, card)
    assert [answer["stop_ref"] for answer in record["answers"]] == [first_ref, second_ref]
    assert [author for author, _body in record["comments"]] == ["owner", "owner"]
    assert record["status"] == "ready"
    for flags in _ANSWERING:
        assert _found(owner, card, **flags) is None


_PRICE_LIST = [{"n": 1, "name": "price list", "type": "csv", "file": "price-list.csv"}]


def _needing_a_file(conn, project: dict, title: str, *, kind: str) -> str:
    card, _run = _running(conn, project, title)
    kb._append_event(conn, card, "named_inputs_declared", {"inputs": copy.deepcopy(_PRICE_LIST)})
    assert kb.block_task(conn, card, reason=_BRIEF, kind=kind) is True
    return card


def test_a_brief_stop_with_a_missing_named_input_keeps_its_own_route_as_a_capability_stop_does(owner):
    project = _project(owner)
    with _board(project) as conn:
        brief = _needing_a_file(conn, project, "Brief stop with a file", kind="transient")
        capability = _needing_a_file(conn, project, "Capability stop with a file", kind="capability")
        control = _briefly(conn, project, "Plain brief stop")
        control_event = _newest_blocked(conn, control)
        assert len(oni.missing_inputs(conn, brief)) == 1 and len(oni.missing_inputs(conn, capability)) == 1
        for card in (brief, capability):
            _shift(conn, "UPDATE tasks SET created_at = created_at - ? WHERE id = ?", _LATER, card)
        created = {
            card: int(conn.execute("SELECT created_at FROM tasks WHERE id = ?", (card,)).fetchone()[0])
            for card in (brief, capability)
        }
        stopped_at = _event_time(conn, _newest_blocked(conn, brief))
        assert created[brief] != stopped_at
        stop_ref = _current_ref(owner, conn, brief)
        before = (_record(conn, brief), _written(conn))
        assert oq.accept_answer(conn, brief, stop_ref, 1, None, owner=owner) == "not_waiting"
        assert oq.accept_answer(conn, brief, stop_ref, 1, "Please go on.", owner=owner) == "not_waiting"
        assert (_record(conn, brief), _written(conn)) == before
    for flags in ({"owner_answer": True}, {"owner_answer": True, "owner_input_upload": True}):
        item, held = _found(owner, brief, **flags), _found(owner, capability, **flags)
        assert item is not None and "question" not in item
        assert item["reason"] == _FIXED and item["created_at"] == ow._owner_timestamp(stopped_at)
        # The upload route takes only a card held needs_input, so a brief stop is offered no upload.
        assert "missing_inputs" not in item
        assert ("missing_inputs" in held) is ("owner_input_upload" in flags)
        own = ("decision_ref", "title", "created_at", "missing_inputs")
        assert {key: value for key, value in item.items() if key not in own} == {
            key: value for key, value in held.items() if key not in own
        }
        assert _found(owner, control, **flags)["question"] == _brief_question(owner, control, control_event)


def test_a_brief_stop_that_holds_design_options_keeps_its_own_route(owner, monkeypatch):
    project = _project(owner)
    with _board(project) as conn:
        brief = _briefly(conn, project, "Brief stop with options")
        stop_ref = _current_ref(owner, conn, brief)

    def holding(_conn, task_ids):
        return {task_id: {} for task_id in task_ids if task_id == brief}

    # A design declaration waits only on a needs_input hold, so no real brief stop holds one: stand its reader in.
    monkeypatch.setattr(ow, "decision_design_options", holding)
    monkeypatch.setattr("hermes_cli.owner_design_choices.decision_design_options", holding)
    item = _found(owner, brief, owner_answer=True)
    assert item is not None and "question" not in item
    with _board(project) as conn:
        before = (_record(conn, brief), _written(conn))
        assert oq.accept_answer(conn, brief, stop_ref, 1, None, owner=owner) == "not_waiting"
        assert (_record(conn, brief), _written(conn)) == before


def test_a_worker_cannot_declare_a_brief_stop(owner):
    assert oq.STOPS == ("question", "review_limit")
    project = _project(owner)
    with _board(project) as conn:
        card, run = _running(conn, project, "Working card")
        before = (_record(conn, card), _written(conn))
        for stop, text, choices in (
            ("stopped_briefly", _BRIEF_TEXT, copy.deepcopy(_CONTINUE)),
            ("stopped_briefly", _QUESTION, copy.deepcopy(_CHOICES)),
            ("transient", _QUESTION, copy.deepcopy(_CHOICES)),
            ("", _QUESTION, copy.deepcopy(_CHOICES)),
            (None, _QUESTION, copy.deepcopy(_CHOICES)),
        ):
            with pytest.raises(ValueError):
                oq.declare_owner_question(
                    conn, card, stop, text, choices, "optional", board=project["board"], expected_run_id=run,
                )
        assert (_record(conn, card), _written(conn)) == before
        assert conn.execute("SELECT status FROM tasks WHERE id = ?", (card,)).fetchone()[0] == "running"


def test_a_declaration_forged_as_a_brief_stop_is_not_served_and_takes_no_answer(owner):
    project = _project(owner)
    with _board(project) as conn:
        asked = _asked(conn, project, "Asked stop")
        row = conn.execute(
            "SELECT id, payload FROM task_events WHERE task_id = ? AND kind = ?", (asked, oq.DECLARED_EVENT),
        ).fetchone()
        payload = json.loads(row["payload"])
        payload.update(stop="stopped_briefly", question=_BRIEF_TEXT, choices=copy.deepcopy(_CONTINUE))
        _shift(conn, "UPDATE task_events SET payload = ? WHERE id = ?", json.dumps(payload), row["id"])
        stop_ref = _current_ref(owner, conn, asked)
        before = (_record(conn, asked), _written(conn))
        assert oq.decision_questions(conn, [asked], owner) == {}
        assert oq.accept_answer(conn, asked, stop_ref, 1, None, owner=owner) == "not_waiting"
        assert (_record(conn, asked), _written(conn)) == before
    served = _found(owner, asked, owner_answer=True)
    # Today's item, with no question, as for any declaration that fails its checks.
    assert "question" not in served and served == _found(owner, asked)


def test_a_failed_read_of_the_brief_stops_fails_the_list_and_the_answer_and_no_other_reader_makes_it(owner):
    project = _project(owner)
    with _board(project) as conn:
        brief = _briefly(conn, project, "Brief stop")
        _stopped(conn, project, "Plain stop")
        stop_ref = _current_ref(owner, conn, brief)
    calls: list[list[str]] = []

    def locked(conn, task_ids, profile):
        calls.append(list(task_ids))
        raise sqlite3.OperationalError("database is locked")

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(oq, "brief_stops", locked)
        for flags in _ANSWERING:
            with pytest.raises(ow.OwnerWorkspaceError) as raised:
                ow.list_owner_decisions(owner, **flags)
            assert raised.value.code == "snapshot_unavailable"
        assert calls
        calls.clear()
        for flags in _NOT_ANSWERING:
            assert {item["title"] for item in _items(owner, project, **flags)} == {"Plain stop"}
        assert calls == []
        with _board(project) as conn:
            before = (_record(conn, brief), _written(conn))
            with pytest.raises(sqlite3.OperationalError, match="database is locked"):
                oq.accept_answer(conn, brief, stop_ref, 1, None, owner=owner)
            assert not conn.in_transaction
            assert (_record(conn, brief), _written(conn)) == before
    for flags in _ANSWERING:
        assert _found(owner, brief, **flags)["question"]["stop_ref"] == stop_ref
    with _board(project) as conn:
        assert oq.accept_answer(conn, brief, stop_ref, 1, None, owner=owner) == "resumed"


def _ranked(conn, cards: list[str]) -> None:
    """Give each card its own creation time, the first of *cards* the newest."""
    top = int(conn.execute("SELECT MAX(created_at) FROM tasks").fetchone()[0])
    for rank, card in enumerate(cards):
        _shift(conn, "UPDATE tasks SET created_at = ? WHERE id = ?", top + 100 - rank, card)


def test_an_old_current_brief_stop_is_read_though_newer_cards_that_are_not_brief_stops_fill_the_page(owner, monkeypatch):
    monkeypatch.setattr(ow, "_OWNER_DECISIONS_LIMIT", 3)
    project = _project(owner)
    with _board(project) as conn:
        gave_up = [_gave_up(conn, project, f"Gave up {n}") for n in range(4)]
        plain = [_stopped(conn, project, f"Plain {n}") for n in range(2)]
        current = _briefly(conn, project, "Current brief stop")
        _ranked(conn, [*gave_up, *plain, current])
    for flags in _ANSWERING:
        feed = ow.list_owner_decisions(owner, **flags)
        assert sorted(item["title"] for item in feed["data"]) == ["Current brief stop", "Plain 0", "Plain 1"]
        assert feed["truncated"] is False


def test_a_brief_stop_is_merged_into_the_order_of_its_project_before_the_page_is_cut(owner, monkeypatch):
    monkeypatch.setattr(ow, "_OWNER_DECISIONS_LIMIT", 3)
    project = _project(owner)
    with _board(project) as conn:
        plain = {n: _stopped(conn, project, f"Plain {n}") for n in (1, 2, 3)}
        brief = _briefly(conn, project, "Brief stop")
        _ranked(conn, [brief, plain[3], plain[2], plain[1]])
    for flags in _ANSWERING:
        feed = ow.list_owner_decisions(owner, **flags)
        assert sorted(item["title"] for item in feed["data"]) == ["Brief stop", "Plain 2", "Plain 3"]
        assert feed["truncated"] is True
    # A reader that does not name the token is not cut earlier for a card it never sees.
    for flags in _NOT_ANSWERING:
        feed = ow.list_owner_decisions(owner, **flags)
        assert sorted(item["title"] for item in feed["data"]) == ["Plain 1", "Plain 2", "Plain 3"]
        assert feed["truncated"] is False


def test_brief_stops_are_found_among_more_cards_than_one_query_can_carry(owner):
    project = _project(owner)
    with _board(project) as conn:
        briefs = [_briefly(conn, project, f"Brief stop {n}") for n in range(3)]
        events = {card: _newest_blocked(conn, card) for card in briefs}
        ids = [f"t_absent{n:04d}" for n in range(1200)]
        for position, card in zip((10, 700, 1100), briefs):
            ids[position] = card
        conn.setlimit(sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER, 999)
        served = oq.decision_questions(conn, ids, owner)
    assert served == {card: _brief_question(owner, card, events[card]) for card in briefs}


@pytest.mark.parametrize("owner_waiting", [False, True])
def test_the_steward_snapshot_lists_a_brief_stop_among_attention_rows_and_not_among_decisions(owner, owner_waiting):
    project = _project(owner)
    with _board(project) as conn:
        _briefly(conn, project, "Brief stop")
        _stopped(conn, project, "Asking stop")
    snapshot = ow.project_steward_snapshot(project_id=project["project_id"], owner_waiting=owner_waiting)
    attention = {
        "title": "Brief stop", "state": ow._OWNER_STATE_LABELS["blocked"],
        "reason": ow._OWNER_BLOCK_REASONS["transient"],
    }
    assert attention in snapshot["needs_attention"]
    assert [entry["title"] for entry in snapshot["decisions_needed"]] == ["Asking stop"]
    assert snapshot["counts"]["needs_attention"] == 1


@pytest.mark.asyncio
async def test_the_served_list_carries_a_brief_stop_only_for_a_reader_that_names_owner_answer_v1(served):
    owner = served
    project = _project(owner)
    with _board(project) as conn:
        brief = _briefly(conn, project, "Brief stop")
        event_id = _newest_blocked(conn, brief)
        _shift(conn, "UPDATE tasks SET created_at = created_at - ? WHERE id = ?", _LATER, brief)
        stopped_at = _event_time(conn, event_id)
    for tokens in (
        ("owner_answer_v1",),
        ("owner_answer_v1", "owner_waiting_v1"),
        ("owner_answer_v1", "owner_input_upload_v1", "design_options_v1"),
    ):
        item = _served_item(await _served(*tokens), owner, brief)
        assert item is not None, tokens
        assert (item["authority"], item["kind"], item["reason"]) == ("task", "owner_input", _FIXED)
        assert item["created_at"] == ow._owner_timestamp(stopped_at)
        assert item["question"] == _brief_question(owner, brief, event_id)
    for tokens in (
        (), ("owner_waiting_v1",), ("owner_input_upload_v1",), ("design_options_v1",),
        ("owner_waiting_v1", "owner_input_upload_v1"),
    ):
        assert _served_item(await _served(*tokens), owner, brief) is None, tokens


def test_a_brief_stop_whose_blocked_event_has_no_payload_is_served_and_answered_as_any_other(owner):
    project = _project(owner)
    with _board(project) as conn:
        brief = _briefly(conn, project, "Brief stop")
        event_id = _newest_blocked(conn, brief)
        _shift(conn, "UPDATE task_events SET payload = NULL, created_at = created_at - ? WHERE id = ?", _LATER, event_id)
        stopped_at = _event_time(conn, event_id)
    for flags in _ANSWERING:
        item = _found(owner, brief, **flags)
        assert item["question"] == _brief_question(owner, brief, event_id)
        assert item["created_at"] == ow._owner_timestamp(stopped_at)
    with _board(project) as conn:
        assert oq.accept_answer(conn, brief, _stop_ref(owner, brief, event_id), 1, None, owner=owner) == "resumed"


def test_a_card_answered_at_an_earlier_stop_is_served_at_its_new_brief_stop(owner):
    project = _project(owner)
    with _board(project) as conn:
        card = _asked(conn, project, "Asked stop")
        old_ref = _current_ref(owner, conn, card)
        assert oq.accept_answer(conn, card, old_ref, 1, None, owner=owner) == "resumed"
        assert kb.claim_task(conn, card) is not None
        assert kb.block_task(conn, card, reason=_BRIEF, kind="transient") is True
        new_event = _newest_blocked(conn, card)
    for flags in _ANSWERING:
        assert _found(owner, card, **flags)["question"] == _brief_question(owner, card, new_event)
    with _board(project) as conn:
        assert oq.accept_answer(conn, card, old_ref, 1, None, owner=owner) == "not_waiting"
        assert oq.accept_answer(conn, card, _stop_ref(owner, card, new_event), 1, None, owner=owner) == "resumed"
        assert len(_record(conn, card)["answers"]) == 2


def test_a_declaration_left_by_an_earlier_stop_does_not_change_the_question_of_a_brief_stop(owner):
    project = _project(owner)
    with _board(project) as conn:
        card = _asked(conn, project, "Asked stop")
        assert kb.unblock_task(conn, card) is True
        assert kb.claim_task(conn, card) is not None
        assert kb.block_task(conn, card, reason=_BRIEF, kind="transient") is True
        event_id = _newest_blocked(conn, card)
        assert conn.execute(
            "SELECT COUNT(*) FROM task_events WHERE task_id = ? AND kind = ?", (card, oq.DECLARED_EVENT),
        ).fetchone()[0] == 1
    for flags in _ANSWERING:
        assert _found(owner, card, **flags)["question"] == _brief_question(owner, card, event_id)


def test_the_reason_the_kernel_recorded_for_a_brief_stop_is_never_shown(owner):
    secret = "internal-detail-7731"
    project = _project(owner)
    with _board(project) as conn:
        brief = _briefly(conn, project, "Brief stop", reason=f"{secret}\x00 réseau interrompu 🙂 ネットワーク")
        stop_ref = _current_ref(owner, conn, brief)
    for flags in _ANSWERING:
        feed = ow.list_owner_decisions(owner, **flags)
        assert secret not in json.dumps(feed, ensure_ascii=False)
        assert _found(owner, brief, **flags)["reason"] == _FIXED
    with _board(project) as conn:
        assert oq.accept_answer(conn, brief, stop_ref, 1, None, owner=owner) == "resumed"
        comments = _record(conn, brief)["comments"]
    assert len(comments) == 1 and secret not in comments[0][1]


@pytest.mark.asyncio
@pytest.mark.parametrize("tokens", [("owner_answer_v1",), ("owner_answer_v1", "owner_waiting_v1")])
async def test_one_item_never_mixes_two_brief_stops_when_the_card_stops_again_between_reads(served, monkeypatch, tokens):
    owner = served
    project = _project(owner)
    with _board(project) as conn:
        card = _briefly(conn, project, "Brief stop")
        old_event = _newest_blocked(conn, card)
        _shift(conn, "UPDATE task_events SET created_at = created_at - ? WHERE id = ?", _LATER, old_event)
        old_at = _event_time(conn, old_event)
    fired: list[threading.Thread] = []
    failures: list[BaseException] = []
    real = ow.decision_missing_inputs

    def stops_briefly_again() -> None:
        try:
            with _board(project) as other:
                assert kb.unblock_task(other, card) is True
                assert kb.claim_task(other, card) is not None
                assert kb.block_task(other, card, reason=_PLAIN, kind="needs_input") is True
                assert kb.unblock_task(other, card) is True
                assert kb.claim_task(other, card) is not None
                assert kb.block_task(other, card, reason=_OLD, kind="transient") is True
        except BaseException as exc:  # reported by the test thread
            failures.append(exc)

    def between_reads(conn, task_ids):
        if not fired:
            # Another connection commits between two reads of the request; see the test above.
            writer = threading.Thread(target=stops_briefly_again)
            fired.append(writer)
            writer.start()
            writer.join(timeout=3)
        return real(conn, task_ids)

    monkeypatch.setattr(ow, "decision_missing_inputs", between_reads)
    item = _served_item(await _served(*tokens), owner, card)
    assert item is not None and len(fired) == 1
    fired[0].join(timeout=60)
    assert not fired[0].is_alive() and failures == []
    with _board(project) as conn:
        new_event = _newest_blocked(conn, card)
        new_at = _event_time(conn, new_event)
    assert new_event != old_event and new_at != old_at
    old = (_stop_ref(owner, card, old_event), ow._owner_timestamp(old_at))
    new = (_stop_ref(owner, card, new_event), ow._owner_timestamp(new_at))
    assert (item["question"]["stop_ref"], item["created_at"]) in (old, new)
    # The next read sees the new stop whole.
    item = _served_item(await _served(*tokens), owner, card)
    assert (item["question"]["stop_ref"], item["created_at"]) == new
