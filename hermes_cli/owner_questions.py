"""The owner question rules: one copy, used by a worker that asks the owner.

A worker whose card cannot go on without the owner records one question on
its own card (owner answer plan, decisions S1 to S3 and card P1): the
question in plain words, 0 to 4 choices of one line each, and whether the
owner's own words are optional or required in the answer. A reviewer at its
review limit records its summary of its corrections as the question instead,
and the platform fixes that stop's two choices.

:func:`declare_owner_question` records the question once, as the task event
``owner_question_declared`` (no schema migration), in the same write
transaction that holds the card ``needs_input`` with the question as its
reason. :func:`current_questions` reads back the question each card waits on
now. Nothing here is shown to the owner.
"""

from __future__ import annotations

import json
import sqlite3
from typing import Any

from hermes_cli.owner_text import check_owner_text

# The object tag and version of the declaration payload (S1).
OBJECT = "hermes.owner_workspace.owner_question"
SCHEMA_VERSION = 1
# The task event that records a card's question (S1).
DECLARED_EVENT = "owner_question_declared"
# The stops a worker declares: its own question, or a reviewer at its review
# limit (S3). A stop without a declaration is the reader's to serve (S4, S5).
STOPS = ("question", "review_limit")
# Whether the owner's answer may or must hold the owner's own words (S1).
WORDS = ("optional", "required")
# The longest question and the longest choice, in code points (S2).
MAX_QUESTION_LENGTH = 500
MAX_CHOICE_LENGTH = 160
MAX_CHOICES = 4
# The choices of a review limit stop, in order; the worker cannot change them (S3).
REVIEW_LIMIT_CHOICES = (
    "Repair it once more with the reviewer's corrections.",
    "Stop this work and release nothing.",
)


def _plain_text(value: Any, what: str, *, limit: int, sentence: bool) -> str:
    """*value* when it is 1 to *limit* code points of plain owner text.

    Without *sentence*, as for the question, it must hold a letter: marks,
    digits or spaces alone are not plain words. With *sentence*, as for a
    choice (S2), it must be one line that holds a letter or a digit: marks or
    spaces alone are refused. The sentences of a choice are not counted; the
    owner accepted that a choice can hold two sentences on one line.
    """
    if (
        not isinstance(value, str)
        or not any(char.isalpha() or (sentence and char.isdigit()) for char in value)
        or len(value) > limit
        or (sentence and value.splitlines() != [value])
        or check_owner_text(value, "planner_prose") != value
    ):
        shape = "one line of " if sentence else ""
        raise ValueError(
            f"{what} must be {shape}1 to {limit} characters in plain words, "
            "with no path, web address, id, secret or model name"
        )
    return value


def _declaration(stop: Any, question: Any, choices: Any, words: Any) -> dict:
    """The S1 payload of a worker's question, or ``ValueError`` (S2, S3).

    A review limit stop ignores the worker's *choices* and *words*: it offers
    exactly :data:`REVIEW_LIMIT_CHOICES`, and the owner's words are optional.
    """
    if stop not in STOPS:
        raise ValueError('stop must be "question" or "review_limit"')
    question = _plain_text(question, "the question", limit=MAX_QUESTION_LENGTH, sentence=False)
    if stop == "review_limit":
        choices = [
            {"n": n, "sentence": sentence}
            for n, sentence in enumerate(REVIEW_LIMIT_CHOICES, start=1)
        ]
        words = "optional"
    else:
        if not isinstance(choices, list) or len(choices) > MAX_CHOICES:
            raise ValueError(f"choices must be a list of 0 to {MAX_CHOICES} entries")
        for n, choice in enumerate(choices, start=1):
            if not isinstance(choice, dict) or set(choice) != {"n", "sentence"}:
                raise ValueError("each choice must be exactly an n and a sentence")
            if type(choice["n"]) is not int or choice["n"] != n:
                raise ValueError("choices must be numbered from 1 in order, with no gaps")
            _plain_text(
                choice["sentence"], f"the sentence of choice {n}",
                limit=MAX_CHOICE_LENGTH, sentence=True,
            )
        if words not in WORDS:
            raise ValueError('words must be "optional" or "required"')
        if not choices and words != "required":
            raise ValueError('a question with no choices must have words "required"')
        choices = [{"n": choice["n"], "sentence": choice["sentence"]} for choice in choices]
    return {
        "object": OBJECT,
        "schema_version": SCHEMA_VERSION,
        "stop": stop,
        "question": question,
        "choices": choices,
        "words": words,
    }


def declare_owner_question(
    conn: sqlite3.Connection,
    task_id: str,
    stop: Any,
    question: Any,
    choices: Any,
    words: Any,
    *,
    board: str,
    expected_run_id: int,
) -> None:
    """Record the worker's question on *task_id* and hold the card for the owner.

    For a worker's tool, on the worker's own current card of *board*. In one
    write transaction on *conn*: *question*, *choices* and *words* are checked
    by S2 and S3; one ``owner_question_declared`` event is written; and the
    card is held ``needs_input`` with the question as its reason through
    ``_block_task_within_txn``, only while *expected_run_id* is still the
    card's current run. ``kanban_task_blocked`` fires once the transaction
    committed, as ``block_task`` fires it.

    *expected_run_id* is required: the worker's run id from its dispatcher, a
    whole number of at least 1. Anything else, None or a bool included, is
    refused before the transaction opens, so a card that no worker claimed, or
    that a newer run claimed, is never held. *stop* is "question" or
    "review_limit"; *choices* is a list of 0 to 4 ``{"n", "sentence"}``
    numbered from 1 in order; *words* is "optional" or "required". A card held
    now cannot declare; once it was resumed it may declare again. A refused
    run id, text or shape, or a card that cannot be held, raises
    ``ValueError`` with a plain message, and then nothing was written.
    """
    from hermes_cli import kanban_db

    if type(expected_run_id) is not int or expected_run_id < 1:
        raise ValueError(
            "the worker's run id is required: the whole number of at least 1 "
            "its dispatcher gave it"
        )
    with kanban_db.write_txn(conn):
        payload = _declaration(stop, question, choices, words)
        kanban_db._append_event(conn, task_id, DECLARED_EVENT, payload)
        blocked = kanban_db._block_task_within_txn(
            conn, task_id, reason=payload["question"], kind="needs_input",
            expected_run_id=expected_run_id,
        )
        if blocked is None:
            raise ValueError(
                f"task {task_id!r} cannot wait for the owner's answer: it is not "
                "running, or this run is not its current run"
            )
    kanban_db._fire_kanban_lifecycle_hook(
        "kanban_task_blocked",
        task_id,
        board=board,
        assignee=blocked["assignee"],
        run_id=blocked["run_id"],
        reason=payload["question"],
    )


def current_questions(
    conn: sqlite3.Connection, task_ids: list[str],
) -> dict[str, tuple[int, dict]]:
    """The question each of *task_ids* waits on now, with its stop's event id.

    The rule of ``owner_design_choices._waiting_declarations``. A declaration
    counts exactly while both are true: the card is blocked now with block
    kind ``needs_input``, by the kernel's own reading of the status changes it
    records (``task_owner_waits``); and the card's newest ``blocked`` event is
    the one the declaration wrote in its own transaction, its first after the
    declaration. So a resume ends a question for good, and a card blocked
    again has no current question until it declares again. Every event that
    leaves the card's status as it is leaves the question as it is too.

    Each entry is ``(id of that blocked event, payload as declared)``: the
    stop reference is computed from that id (S6), and the reader checks the
    payload again before it shows it (S2). Batched: a few queries for all of
    *task_ids* together, never one per card. They all read one snapshot, so
    a resume, claim or declaration that commits meanwhile never mixes two
    states: inside a transaction the caller holds already, which stays open;
    otherwise inside a read transaction opened and ended here, so a read-only
    connection serves too.
    """
    from hermes_cli import kanban_db

    own_transaction = not conn.in_transaction
    if own_transaction:
        conn.execute("BEGIN")
    try:
        held = [
            task_id for task_id, wait in kanban_db.task_owner_waits(conn, task_ids).items()
            if wait["kind"] == "needs_input"
        ]
        if not held:
            return {}
        declared: dict[str, dict] = {}
        hold: dict[str, tuple[int, dict]] = {}
        newest: dict[str, int] = {}
        for task_id, event_id, kind, payload in conn.execute(
            "SELECT task_id, id, kind, payload FROM task_events WHERE kind IN (?, 'blocked') "
            "AND task_id IN (" + ", ".join("?" for _ in held) + ") ORDER BY id",
            (DECLARED_EVENT, *held),
        ).fetchall():
            task_id = str(task_id)
            if kind == DECLARED_EVENT:
                declared[task_id] = json.loads(payload)
                continue
            newest[task_id] = int(event_id)
            if task_id in declared:
                hold[task_id] = (int(event_id), declared.pop(task_id))
    finally:
        if own_transaction and conn.in_transaction:
            conn.execute("ROLLBACK")  # a read keeps nothing
    return {
        task_id: current for task_id, current in hold.items() if current[0] == newest[task_id]
    }
