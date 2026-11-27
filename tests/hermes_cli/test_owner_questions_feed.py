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
