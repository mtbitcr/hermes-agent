"""The owner question rules: one copy, used by a worker that asks the owner.

A worker whose card cannot go on without the owner records one question on
its own card (owner answer plan, decisions S1 to S3 and card P1): the
question in plain words, 0 to 4 choices of one sentence each, and whether the
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
import re
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
# The sentence rule of a choice (S2); :func:`_one_sentence` explains it.
# The marks that end a sentence: the ASCII marks, the ellipsis, and the full
# stops and question or exclamation marks of other scripts. The Greek question
# mark looks like a semicolon, so it is written as an escape.
_MARKS = ".!?…。！？．｡‼⁇⁈⁉؟۔।॥።။։\u037e"
# Each quote or bracket that can close right after a mark, with the one that
# opens it. A straight quote opens and closes alike.
_OPENERS = {
    "”": "“", "’": "‘", ")": "(", "]": "[", "」": "「", "』": "『", "）": "（", "】": "【",
    '"': '"', "'": "'",
}
# A run of marks, with the closing quotes or brackets in it or right after it.
_SENTENCE_END = re.compile(
    "[" + re.escape(_MARKS) + "][" + re.escape(_MARKS + "".join(_OPENERS)) + "]*"
)
# The punctuation a sentence goes on with: a comma, a semicolon, a colon, a
# hyphen or a dash.
_JOINERS = ",;:-–—"
# The Japanese particles that join a quotation to its sentence: と and って
# quote it, を makes it the object. No word starts with を.
_PARTICLES = ("と", "って", "を")
# A dotted abbreviation: two or more groups of one or two letters, each with
# its point, as in U.S., e.g., a.m. or Ph.D.
_DOTTED = re.compile(r"(?<![\w.])(?:[^\W\d_]{1,2}\.){2,}")
# The word right before a point, with any points inside it, as "etc" or "e.g".
_WORD = re.compile(r"(?<![\w.])[^\W\d_]+(?:\.[^\W\d_]+)*\Z")
# Titles before a name and lead-ins, whose point ends no sentence.
_TITLES = {"Dr", "Mr", "Mrs", "Ms", "Mx", "Prof", "St", "Mt"}
_LEAD_INS = {"e.g", "i.e", "vs", "cf", "approx", "incl"}
# Number lead-ins and months, whose point ends no sentence before a digit.
_NUMBER_LEAD_INS = {
    "No", "p", "pp", "Fig", "Vol",
    "Jan", "Feb", "Mar", "Apr", "Jun", "Jul", "Aug", "Sep", "Sept", "Oct", "Nov", "Dec",
}
# Abbreviations that can end a sentence, as every dotted one can; "al" is
# that of et al.
_CAN_END = {"etc", "al", "Inc", "Ltd", "Co", "Corp", "Jr", "Sr"}


def _one_sentence(text: str) -> bool:
    """Whether *text* is exactly one sentence: the S2 rule for a choice.

    A sentence holds at least one letter, of any script; marks, digits or
    spaces alone are no sentence. A sentence ends at a run of the marks
    . ! ? … or the full stops and question or exclamation marks of other
    scripts, with the quotes or brackets that close right after the marks. A
    quote or bracket closes there only if the text opened it before; a
    straight quote, only if the same quote came before. Any text after an end
    starts a second sentence, glued to the end or not, unless one of these
    holds:

    - the end is a point inside a number, as in 2.5 or 2.5.1, or before one,
      as in .5;
    - it is a point inside a dotted abbreviation, as in U.S., e.g., a.m. or
      Ph.D.;
    - it is the point of a title (Dr. Mr. Mrs. Ms. Mx. Prof. St. Mt.) before
      a name, that is before a letter, or of a lead-in (e.g. i.e. vs. cf.
      approx. incl.);
    - it is the point of a number lead-in or a month (No. p. pp. Fig. Vol.
      Jan. Feb. Mar. Apr. Jun. Jul. Aug. Sep. Sept. Oct. Nov. Dec.) before a
      digit;
    - it is the last point of an abbreviation that can end a sentence (etc.
      et al. Inc. Ltd. Co. Corp. Jr. Sr. and every dotted one), before a
      space and a lowercase letter, or before , ; : a hyphen or a dash;
    - it is an ellipsis (… or three points or more) before a lowercase letter;
    - a quote or bracket closes there, as in “Ready?” or (wait!), and a
      lowercase letter, , ; : a hyphen or a dash comes next, or a Japanese
      particle that joins the quotation to its sentence, と, って or を,
      comes right after it, as in “準備はいいですか？”と聞きます。.

    Abbreviations count only as written here, so "ms." or "no." is no title
    and no number lead-in; a lead-in may also start with a capital, but
    "VS." is no lead-in.

    Where the rule cannot tell an abbreviation from the end of a sentence,
    it refuses: an abbreviation that can end a sentence, as U.S. or etc.,
    before a capital or a numeral, glued or not ("U.S. Then", "etc. 2",
    "U.S.2"); a single capital letter before a capital ("option A. It");
    and a closing quote before a letter of a caseless script other than
    those particles (“はい。”次に進みます。). S2 prefers a refusal, which the
    worker rewrites with nothing written, to a second sentence stored. Only
    a title before a capital and と right after a quotation are read the
    other way, as one sentence, since a name and the quoting particle are
    what usually follow them. A semicolon never ends a sentence, as in
    English, even where Greek text uses it as a question mark; the separate
    Greek question mark character (U+037E) does.
    """
    if not any(char.isalpha() for char in text):
        return False
    for end in _SENTENCE_END.finditer(text):
        start, stop = end.span()
        # The end stops before a quote or bracket that nothing opened.
        stop = next(
            (
                at for at in range(start + 1, stop)
                if text[at] in _OPENERS and _OPENERS[text[at]] not in text[:at]
            ),
            stop,
        )
        if text[stop:].strip() and not _goes_on(text, start, stop):
            return False
    return True


def _goes_on(text: str, start: int, stop: int) -> bool:
    """Whether the sentence goes on after its end ``text[start:stop]``, which
    more text follows, by the exceptions of :func:`_one_sentence`."""
    run, rest = text[start:stop], text[stop:]
    after = rest.lstrip()
    follows, spaced = after[0], len(after) < len(rest)
    if any(char in _OPENERS for char in run):
        return (
            follows.islower()
            or follows in _JOINERS
            or (not spaced and after.startswith(_PARTICLES))
        )
    if set(run) <= set(".…") and (len(run) >= 3 or "…" in run):
        return follows.islower()
    if run != ".":
        return False
    before = text[start - 1] if start else " "
    if follows.isdigit() and not spaced and (before.isdigit() or before.isspace()):
        return True
    dotted = next(
        (match for match in _DOTTED.finditer(text) if match.start() < start < match.end()),
        None,
    )
    if dotted and start < dotted.end() - 1:
        return True
    word = _WORD.search(text[:start])
    word = word.group() if word else ""
    if word in _TITLES:
        return follows.isalpha()
    if word[:1].lower() + word[1:] in _LEAD_INS:
        return True
    if word in _NUMBER_LEAD_INS:
        return follows.isdigit()
    if word in _CAN_END or dotted:
        return (spaced and follows.islower()) or follows in _JOINERS
    return False


def _plain_text(value: Any, what: str, *, limit: int, sentence: bool) -> str:
    """*value* when it is 1 to *limit* code points of plain owner text.

    It must hold a letter: marks, digits or spaces alone are not plain words.
    With *sentence*, it must also be exactly one sentence on one line (S2).
    """
    if (
        not isinstance(value, str)
        or not any(char.isalpha() for char in value)
        or len(value) > limit
        or (sentence and (value.splitlines() != [value] or not _one_sentence(value)))
        or check_owner_text(value, "planner_prose") != value
    ):
        shape = "exactly one sentence of " if sentence else ""
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
