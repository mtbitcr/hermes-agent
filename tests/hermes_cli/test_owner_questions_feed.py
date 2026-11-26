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

import pytest

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
        capability = _stopped(conn, project, "Capability stop", kind="capability", reason=_PLAIN)
    for flags in ({}, {"owner_waiting": True}):
        today = ow.list_owner_decisions(owner, **flags)
        answered = ow.list_owner_decisions(owner, owner_answer=True, **flags)
        assert "question" not in _json(today)
        stripped = copy.deepcopy(answered)
        for item in stripped["data"]:
            item.pop("question", None)
        # Same items, same order, same words, with only ``question`` set apart.
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
    # A capability stop joins the list only for owner_waiting_v1, as before.
    assert ow._owner_decision_ref(owner, authority="task", native_id=capability) not in [
        item["decision_ref"] for item in ow.list_owner_decisions(owner, owner_answer=True)["data"]
    ]


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
