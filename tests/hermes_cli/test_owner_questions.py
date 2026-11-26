"""A worker asks the owner a question on its own card, and the card waits.

Card P1 of the plan for how the owner answers a stopped card, with decisions
S1 to S3 and S6. In one write transaction
``owner_questions.declare_owner_question`` checks the question and its
choices the way the owner text policy checks anything the owner reads,
records one ``owner_question_declared`` task event and holds the card
``needs_input`` with the question as its reason; ``kanban_task_blocked``
fires once that transaction committed. Every refusal writes nothing.
``owner_questions.current_questions`` reads back the question each card waits
on now. Nothing is shown to the owner yet.

Every count a test proves is read by a second, read-only connection to the
board's own store, independent of the connection that wrote.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from types import SimpleNamespace

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import owner_questions as oq
from hermes_cli.owner_text import check_owner_text
from hermes_cli.plugins import get_plugin_manager

QUESTION = "Which sign up page should Raphael build first?"
SENTENCES = (
    "Build the short form with one field.",
    "Build the long form with every field.",
    "Build both forms and let people pick.",
    "Wait until the designer has drawn both forms.",
    "Ask the designer for a third form.",
)
# A reviewer's summary of its corrections: two sentences on two lines.
REVIEW_SUMMARY = (
    "The reviewer asked for two corrections.\n"
    "The title must be clearer and the form must be shorter."
)
REVIEW_LIMIT_CHOICES = [
    {"n": 1, "sentence": "Repair it once more with the reviewer's corrections."},
    {"n": 2, "sentence": "Stop this work and release nothing."},
]
PLAIN_STOP = "Raphael needs your answer on the sign up page."


@pytest.fixture
def board(tmp_path, monkeypatch):
    """A real disposable default board under a temporary root."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    for key in (
        "HERMES_KANBAN_TASK", "HERMES_KANBAN_RUN_ID", "HERMES_KANBAN_DB",
        "HERMES_KANBAN_BOARD", "HERMES_KANBAN_ATTACHMENTS_ROOT",
    ):
        monkeypatch.delenv(key, raising=False)
    kb._INITIALIZED_PATHS.clear()
    kb.init_db()
    conn = kb.connect()
    try:
        yield conn
    finally:
        conn.close()


@pytest.fixture
def fired_hooks():
    """Each ``kanban_task_blocked`` call, with the card's status as a second,
    read-only connection to ``store`` (set by the test) saw it while the hook
    ran: a hook fired inside the transaction would still see the card running.
    """
    manager = get_plugin_manager()
    saved = {name: list(callbacks) for name, callbacks in manager._hooks.items()}
    seen = SimpleNamespace(store=None, fired=[])

    def capture(**fields) -> None:
        fields.pop("telemetry_schema_version", None)
        status = _observe(
            seen.store, "SELECT status FROM tasks WHERE id = ?", fields["task_id"],
        )
        seen.fired.append((fields, status))

    manager._hooks.setdefault("kanban_task_blocked", []).append(capture)
    try:
        yield seen
    finally:
        manager._hooks = saved


def _store_file(conn: sqlite3.Connection) -> str:
    return next(row[2] for row in conn.execute("PRAGMA database_list") if row[1] == "main")


def _observe(store_file: str, query: str, *params):
    """One value, as a second, read-only connection to the board sees it."""
    observer = sqlite3.connect(Path(store_file).as_uri() + "?mode=ro", uri=True)
    try:
        return observer.execute(query, params).fetchone()[0]
    finally:
        observer.close()


def _observed_state(store_file: str, task_id: str) -> dict:
    """What another connection sees of the card and the whole board."""
    return {
        "status": _observe(store_file, "SELECT status FROM tasks WHERE id = ?", task_id),
        "block_kind": _observe(
            store_file, "SELECT block_kind FROM tasks WHERE id = ?", task_id,
        ),
        "declarations": _observe(
            store_file,
            "SELECT COUNT(*) FROM task_events WHERE task_id = ? AND kind = ?",
            task_id, oq.DECLARED_EVENT,
        ),
        "events": _observe(store_file, "SELECT COUNT(*) FROM task_events"),
        "open_runs": _observe(
            store_file,
            "SELECT COUNT(*) FROM task_runs WHERE task_id = ? AND ended_at IS NULL",
            task_id,
        ),
    }


def _running_card(conn: sqlite3.Connection, title: str = "sign up page") -> tuple[str, int]:
    """(card, run) of a card a worker claimed and is working on."""
    task_id = kb.create_task(conn, title=title, assignee="builder")
    claimed = kb.claim_task(conn, task_id)
    assert claimed is not None and claimed.status == "running"
    return task_id, claimed.current_run_id


def _resumed_and_claimed(conn: sqlite3.Connection, task_id: str) -> int:
    """The new run of *task_id* once it was resumed and a worker claimed it again."""
    assert kb.unblock_task(conn, task_id) is True
    claimed = kb.claim_task(conn, task_id)
    assert claimed is not None and claimed.status == "running"
    return claimed.current_run_id


def _choices(count: int) -> list[dict]:
    return [{"n": n, "sentence": SENTENCES[n - 1]} for n in range(1, count + 1)]


def _question_of(length: int) -> str:
    """A plain question of exactly *length* code points, and more bytes than that."""
    text = "Should Raphael build the café sign up page" + " and then the café menu page" * 20
    return text[: length - 1] + "?"


def _sentence_of(length: int) -> str:
    """One plain sentence of exactly *length* code points."""
    text = "Build the sign up form with a short title" + " and a clear button" * 10
    return text[: length - 1] + "."


def _declare(conn, task_id, run_id, stop, question, choices, words):
    return oq.declare_owner_question(
        conn, task_id, stop, question, choices, words,
        board=kb.DEFAULT_BOARD, expected_run_id=run_id,
    )


def _declared_payloads(conn: sqlite3.Connection, task_id: str) -> list[dict]:
    return [e.payload for e in kb.list_events(conn, task_id) if e.kind == oq.DECLARED_EVENT]


def _blocked_event_ids(conn: sqlite3.Connection, task_id: str) -> list[int]:
    return [e.id for e in kb.list_events(conn, task_id) if e.kind == "blocked"]


def test_the_review_limit_choices_pass_the_owner_text_policy_unchanged():
    """S3 and S2: the two fixed choices, which the read-time check runs on again."""
    assert [
        {"n": n, "sentence": sentence}
        for n, sentence in enumerate(oq.REVIEW_LIMIT_CHOICES, start=1)
    ] == REVIEW_LIMIT_CHOICES
    for choice in REVIEW_LIMIT_CHOICES:
        assert check_owner_text(choice["sentence"], "planner_prose") == choice["sentence"]


def test_the_review_limit_choices_pass_the_module_s_own_choice_check():
    """S2 and S3: each fixed choice is exactly one sentence by the same check
    a worker's choice must pass."""
    for n, sentence in enumerate(oq.REVIEW_LIMIT_CHOICES, start=1):
        assert oq._plain_text(
            sentence, f"the sentence of choice {n}", limit=oq.MAX_CHOICE_LENGTH, sentence=True,
        ) == sentence


# ---------------------------------------------------------------------------
# Card P1 test (a): one event and the hold, in one transaction.
# ---------------------------------------------------------------------------


def test_a_declared_question_holds_the_card_needs_input_with_one_event_and_the_question_as_reason(
    board, monkeypatch, fired_hooks,
):
    """Card P1 test (a), S1: one ``owner_question_declared`` event and the
    ``needs_input`` hold with the question as its reason, written on one open
    transaction; ``kanban_task_blocked`` fires once, after the commit."""
    task_id, run_id = _running_card(board)
    store = fired_hooks.store = _store_file(board)
    before = _observed_state(store, task_id)
    hold = kb._block_task_within_txn
    seen: dict = {}

    def observed_hold(conn, held_task_id, **kwargs):
        blocked = hold(conn, held_task_id, **kwargs)
        seen["kwargs"] = kwargs
        seen["in_transaction"] = conn.in_transaction
        seen["writer"] = (
            conn.execute(
                "SELECT COUNT(*) FROM task_events WHERE task_id = ? AND kind = ?",
                (held_task_id, oq.DECLARED_EVENT),
            ).fetchone()[0],
            conn.execute("SELECT status FROM tasks WHERE id = ?", (held_task_id,)).fetchone()[0],
        )
        seen["observer"] = _observed_state(store, held_task_id)
        return blocked

    monkeypatch.setattr(kb, "_block_task_within_txn", observed_hold)

    _declare(board, task_id, run_id, "question", QUESTION, _choices(2), "optional")

    # The declaration and the hold are written on one open transaction that
    # no other connection can see until it commits.
    assert seen["in_transaction"] is True
    assert seen["kwargs"]["kind"] == "needs_input"
    assert seen["kwargs"]["reason"] == QUESTION
    assert seen["kwargs"]["expected_run_id"] == run_id
    assert seen["writer"] == (1, "blocked")
    assert seen["observer"] == before
    assert _observed_state(store, task_id) == {
        **before,
        "status": "blocked",
        "block_kind": "needs_input",
        "declarations": 1,
        "events": before["events"] + 2,  # the declaration and the hold's "blocked"
        "open_runs": 0,
    }
    (declared,) = [
        e for e in kb.list_events(board, task_id) if e.kind == "owner_question_declared"
    ]
    assert declared.payload == {
        "object": "hermes.owner_workspace.owner_question",
        "schema_version": 1,
        "stop": "question",
        "question": QUESTION,
        "choices": [
            {"n": 1, "sentence": SENTENCES[0]},
            {"n": 2, "sentence": SENTENCES[1]},
        ],
        "words": "optional",
    }
    (blocked,) = [e for e in kb.list_events(board, task_id) if e.kind == "blocked"]
    assert blocked.payload["reason"] == QUESTION
    assert blocked.payload["kind"] == "needs_input"
    assert declared.id < blocked.id
    assert _observe(store, "SELECT outcome FROM task_runs WHERE id = ?", run_id) == "blocked"
    # kanban_task_blocked fired once, after the commit.
    ((fields, status_seen),) = fired_hooks.fired
    assert status_seen == "blocked"
    assert fields["task_id"] == task_id
    assert fields["board"] == kb.DEFAULT_BOARD
    assert fields["reason"] == QUESTION
    assert fields["run_id"] == run_id
    # The question is the card's current one, from the hold it wrote (S6).
    assert oq.current_questions(board, [task_id]) == {task_id: (blocked.id, declared.payload)}


@pytest.mark.parametrize(
    "count, words", [(0, "required"), (1, "optional"), (2, "required"), (3, "optional"), (4, "optional")],
)
def test_zero_to_four_choices_are_recorded_as_given(board, count, words):
    """S1 and S2: a worker supplies 0 to 4 choices; the event holds exactly the S1 payload."""
    task_id, run_id = _running_card(board)

    _declare(board, task_id, run_id, "question", QUESTION, _choices(count), words)

    assert _declared_payloads(board, task_id) == [{
        "object": "hermes.owner_workspace.owner_question",
        "schema_version": 1,
        "stop": "question",
        "question": QUESTION,
        "choices": _choices(count),
        "words": words,
    }]


def test_a_question_of_500_code_points_and_a_choice_of_160_are_accepted(board):
    """S2: the limits count code points; a question of 500 code points is more bytes."""
    task_id, run_id = _running_card(board)
    question, sentence = _question_of(500), _sentence_of(160)
    assert len(question) == 500 < len(question.encode("utf-8")) and len(sentence) == 160

    _declare(board, task_id, run_id, "question", question, [{"n": 1, "sentence": sentence}], "optional")

    (payload,) = _declared_payloads(board, task_id)
    assert payload["question"] == question
    assert payload["choices"] == [{"n": 1, "sentence": sentence}]
    (blocked,) = [e for e in kb.list_events(board, task_id) if e.kind == "blocked"]
    assert blocked.payload["reason"] == question


ONE_SENTENCE_CHOICES = {
    "an English sentence": "Build the short form first.",
    "a Japanese sentence": "短いフォームを作ります。",
    "a decimal number": "Use 2.5 times the spacing.",
    "initials at the end": "Launch the page in the U.S.",
    "the first review limit choice": oq.REVIEW_LIMIT_CHOICES[0],
    "the second review limit choice": oq.REVIEW_LIMIT_CHOICES[1],
    # A point inside a number or of an abbreviation, an ellipsis before a
    # lowercase letter, and a quotation or bracket that closes inside the
    # sentence end no sentence.
    "a title before a name": "Ask Dr. Lee to review the form.",
    "a dotted abbreviation before a lowercase letter": "Use the U.S. spelling.",
    "a lead-in inside brackets": "Use the examples (e.g. the short form).",
    "a quotation that closes after a question mark": "Set the banner to “Ready?” before posting.",
    "a quotation that closes after a full stop": "Set the banner to “Done.” before posting.",
    "a quotation that closes before a comma": "Ask “Is it ready?”, then build the form.",
    "a bracket that closes after an exclamation mark": (
        "Build the short form (the long one can wait!) first."
    ),
    "two titles before names": "Ask Mr. Smith and Mrs. Jones to sign the form.",
    "the lead-in vs.": "Compare plan A vs. plan B first.",
    "p.m. before a lowercase letter": "Ship it by 5 p.m. on Friday.",
    "a dotted abbreviation before a noun": "Ask the U.S. office to review the form.",
    "etc. before a lowercase letter": "Bring the forms, pens, etc. to the review.",
    "No. before a digit": "Use form No. 2 for the review.",
    "No. glued to a digit": "Use form No.2 for the review.",
    "three points before a lowercase letter": "Wait for the review... then build the form.",
    "an ellipsis before a lowercase letter": "Wait for the review… then build the form.",
    "a version number": "Use version 2.5.1 of the form.",
    "a semicolon": "Build the short form; then build the long one.",
    "a question": "Is the short form enough?",
    "a straight quotation that closes after a question mark": (
        'Set the banner to "Ready?" before posting.'
    ),
    # The policy refuses the corner brackets 「」, so curly quotes stand in.
    "a Japanese quotation that closes after a question mark, before と": (
        "“準備はいいですか？”と聞きます。"
    ),
    "a Japanese quotation that closes after a question mark, before って": (
        "“準備はいいですか？”って聞きます。"
    ),
    "a Japanese quotation that closes after a full stop, before と": "“はい。”と言います。",
    "a Japanese quotation that closes after a question mark, before を": (
        "“保存しますか？”を表示します。"
    ),
    "the title Mx. before a name": "Ask Mx. Lee to sign the form.",
    "the title Mt. before a name": "Climb Mt. Fuji first.",
    "a dotted abbreviation before a hyphen": "Use the U.S.-based server.",
    "a month before a date": "Ship it by Jan. 5 at noon.",
    "et al. before a lowercase letter": "Ask Lee et al. to review it.",
    # The Greek question mark looks like a semicolon, so it is written as an escape.
    "a Greek question": "Είναι έτοιμη η φόρμα\u037e",
}


@pytest.mark.parametrize("case", sorted(ONE_SENTENCE_CHOICES))
def test_a_choice_of_one_sentence_is_declared_and_stored_as_given(board, case):
    """S2: a choice of one sentence, in any script, with a decimal number, an
    abbreviation, an ellipsis, an inline quotation or a final mark, passes the
    owner text policy unchanged and is not taken for two sentences or none."""
    task_id, run_id = _running_card(board)
    sentence = ONE_SENTENCE_CHOICES[case]
    assert check_owner_text(sentence, "planner_prose") == sentence

    _declare(board, task_id, run_id, "question", QUESTION, [{"n": 1, "sentence": sentence}], "optional")

    (payload,) = _declared_payloads(board, task_id)
    assert payload["choices"] == [{"n": 1, "sentence": sentence}]
    assert kb.get_task(board, task_id).status == "blocked"
    assert oq.current_questions(board, [task_id]) == {
        task_id: (_blocked_event_ids(board, task_id)[-1], payload),
    }


# ---------------------------------------------------------------------------
# Card P1 test (b): each refused text writes nothing.
# ---------------------------------------------------------------------------


def _with_choice(sentence: str) -> list[dict]:
    return [{"n": 1, "sentence": SENTENCES[0]}, {"n": 2, "sentence": sentence}]


# case -> (stop, question, choices, words) for the card with id task_id.
REFUSED_TEXTS = {
    "a path": lambda task_id: (
        "question", "Should Raphael copy /root/site/index.html into the new folder?",
        _choices(2), "optional",
    ),
    "a card id": lambda task_id: (
        "question", f"Should Raphael wait for {task_id} to finish first?", _choices(2), "optional",
    ),
    "a commit id": lambda task_id: (
        "question", "Should Raphael keep commit 0a1b2c3 as it is?", _choices(2), "optional",
    ),
    "a model name": lambda task_id: (
        "question", "Should Raphael ask GPT-5 to write the page?", _choices(2), "optional",
    ),
    "501 code points": lambda task_id: ("question", _question_of(501), _choices(2), "optional"),
    "two sentences in a choice": lambda task_id: (
        "question", QUESTION, _with_choice("Use the short form. Then add a title."), "optional",
    ),
    # Two sentences with no space between them, or after a mark that is not ASCII.
    "two sentences in a choice, joined by an exclamation mark": lambda task_id: (
        "question", QUESTION, _with_choice("Build the short form!Then build the long form."),
        "optional",
    ),
    "two Japanese sentences in a choice": lambda task_id: (
        "question", QUESTION, _with_choice("短いフォームを作ります。次に長いフォームを作ります。"),
        "optional",
    ),
    "two sentences in a choice, after a full-width question mark": lambda task_id: (
        "question", QUESTION, _with_choice("短いフォームでいいですか？次に長いフォームを作ります。"),
        "optional",
    ),
    "two sentences in a choice, a question mark before a lowercase letter": lambda task_id: (
        "question", QUESTION, _with_choice("Is the short form enough?then build the long form."),
        "optional",
    ),
    "two sentences in a choice, a full stop before an uppercase letter": lambda task_id: (
        "question", QUESTION, _with_choice("Build the short form.I will check it later."),
        "optional",
    ),
    "5 choices": lambda task_id: ("question", QUESTION, _choices(5), "optional"),
    "0 choices with words optional": lambda task_id: ("question", QUESTION, [], "optional"),
}


@pytest.mark.parametrize("case", sorted(REFUSED_TEXTS))
def test_card_p1_b_a_refused_text_writes_nothing(board, fired_hooks, case):
    """Card P1 test (b), S2: the declaration is refused with a plain error and nothing is written."""
    task_id, run_id = _running_card(board)
    stop, question, choices, words = REFUSED_TEXTS[case](task_id)
    store = fired_hooks.store = _store_file(board)
    before = _observed_state(store, task_id)

    with pytest.raises(ValueError):
        _declare(board, task_id, run_id, stop, question, choices, words)

    assert _observed_state(store, task_id) == before
    assert before["status"] == "running" and before["declarations"] == 0
    assert board.in_transaction is False
    assert fired_hooks.fired == []
    assert oq.current_questions(board, [task_id]) == {}


# Choices of two sentences, glued to a numeral or to opening punctuation or not,
# and choices of no sentence at all (S2). Each passes the owner text policy
# unchanged, so the sentence rule alone refuses it.
NOT_ONE_SENTENCE_CHOICES = {
    "a full stop glued to a numeral": "Use the short form.2 people can review it.",
    "an exclamation mark glued to a numeral": "Try the short form!2 people can review it.",
    "an exclamation mark glued to an opening bracket": (
        "Try the short form!(The longer form can wait.)"
    ),
    "a question mark glued to an opening quote": "Is the form ready?“Yes, it is.”",
    "a question mark alone": "?",
    "three points alone": "...",
    "two marks alone": "!?",
    "an ellipsis alone": "…",
    "a full stop glued to an opening bracket": "Use the short form.(The long form can wait.)",
    "a full stop glued to an opening quote": "Use the short form.“Then add a title.”",
    "a full stop before a numeral": "Use the short form. 2 people can review it.",
    # The policy refuses "form.then" as a host name, so "form.a" stands in.
    "a full stop glued to a lowercase letter": "Use the short form.a title comes next.",
    "a full stop before a lowercase letter": "Use the short form. then add a title.",
    "an exclamation mark before a lowercase letter": "Use the short form! then add a title.",
    "three points before a capital": "Wait for the review... Then build the form.",
    "a bracket that closes after a full stop, before a capital": (
        "Build the short form (it is faster.) Then build the long one."
    ),
    "a single capital letter before a capital": "Pick option A. It costs less.",
    "a dotted abbreviation before a capital": "Ship the form to the U.S. Then sign it.",
    "etc. before a capital": "Bring the forms, pens, etc. Then sign them.",
    "no. before a capital": "Answer no. Then stop the work.",
    # The policy refuses "form。Th" as a host name, so "form。A" stands in.
    "an ideographic full stop in English": "Build the short form。A title comes next.",
    # The policy refuses the corner brackets 「」, so curly quotes stand in.
    "a Japanese quotation that closes after a full stop, then more text": "“はい。”次に進みます。",
    "a dotted abbreviation glued to a numeral": "Ship the form to the U.S.2 people can review it.",
    "a bracket that nothing opened, before a lowercase letter": (
        "Use the short form.) then add a title."
    ),
    "a straight quote glued after a question mark that opens a sentence": (
        'Is the form ready?"Yes, it is."'
    ),
    "a title before a numeral": "Ask the Dr.2 people can review it.",
    "a title before an opening bracket": "Ask the Dr.(The long form can wait.)",
    "a Japanese quotation that closes after a question mark, then more text": (
        "“準備はいいですか？”次に進みます。"
    ),
    "etc. before a numeral": "Bring the forms, pens, etc. 2 people can review it.",
    "etc. glued to a lowercase letter": "Bring the forms, pens, etc.a title comes next.",
    # The Greek question mark looks like a semicolon, so it is written as an escape.
    "a Greek question mark before a capital": "Είναι έτοιμη η φόρμα\u037e Ναι, είναι.",
}


@pytest.mark.parametrize("case", sorted(NOT_ONE_SENTENCE_CHOICES))
def test_card_p1_b_a_choice_of_two_sentences_or_none_writes_nothing(board, fired_hooks, case):
    """Card P1 test (b), S2: a choice of two sentences, or of no sentence, is
    refused with a plain error and nothing is written, though the owner text
    policy lets it through unchanged."""
    sentence = NOT_ONE_SENTENCE_CHOICES[case]
    assert check_owner_text(sentence, "planner_prose") == sentence
    task_id, run_id = _running_card(board)
    store = fired_hooks.store = _store_file(board)
    before = _observed_state(store, task_id)

    with pytest.raises(ValueError, match="choice 2 must be exactly one sentence"):
        _declare(board, task_id, run_id, "question", QUESTION, _with_choice(sentence), "optional")

    assert _observed_state(store, task_id) == before
    assert before["status"] == "running" and before["declarations"] == 0
    assert board.in_transaction is False
    assert fired_hooks.fired == []
    assert oq.current_questions(board, [task_id]) == {}


# The other limits S1 and S2 set, and the stops only the reader serves (S4, S5).
REFUSED_SHAPES = {
    "a path in a choice": lambda task_id: (
        "question", QUESTION, _with_choice("Copy the file at /root/site/form.html first."), "optional",
    ),
    "a card id in a choice": lambda task_id: (
        "question", QUESTION, _with_choice(f"Wait for {task_id} to finish."), "optional",
    ),
    "an empty question": lambda task_id: ("question", "", _choices(2), "optional"),
    "a blank question": lambda task_id: ("question", "   ", _choices(2), "optional"),
    "a question of marks alone": lambda task_id: ("question", "?", _choices(2), "optional"),
    "a question that is not text": lambda task_id: ("question", None, _choices(2), "optional"),
    "a choice of 161 code points": lambda task_id: (
        "question", QUESTION, _with_choice(_sentence_of(161)), "optional",
    ),
    "an empty choice": lambda task_id: ("question", QUESTION, _with_choice(""), "optional"),
    "a choice over two lines": lambda task_id: (
        "question", QUESTION, _with_choice("Build the long form\nwith every field."), "optional",
    ),
    "a gap in the choice numbers": lambda task_id: (
        "question", QUESTION,
        [{"n": 1, "sentence": SENTENCES[0]}, {"n": 3, "sentence": SENTENCES[1]}], "optional",
    ),
    "a choice numbered with text": lambda task_id: (
        "question", QUESTION, [{"n": "1", "sentence": SENTENCES[0]}], "optional",
    ),
    "a choice with an unknown field": lambda task_id: (
        "question", QUESTION, [{"n": 1, "sentence": SENTENCES[0], "url": "https://example.com"}],
        "optional",
    ),
    "a choice that is only a sentence": lambda task_id: ("question", QUESTION, [SENTENCES[0]], "optional"),
    "choices that are not a list": lambda task_id: ("question", QUESTION, SENTENCES[0], "optional"),
    "no choices given": lambda task_id: ("question", QUESTION, None, "required"),
    "an unknown words value": lambda task_id: ("question", QUESTION, _choices(2), "maybe"),
    "no words value": lambda task_id: ("question", QUESTION, _choices(2), None),
    "a plain stop, which only the reader serves": lambda task_id: (
        "plain", QUESTION, [], "required",
    ),
    "a brief stop, which only the reader serves": lambda task_id: (
        "stopped_briefly", QUESTION, [{"n": 1, "sentence": "Continue the work."}], "optional",
    ),
    "an unknown stop": lambda task_id: ("design", QUESTION, _choices(2), "optional"),
    "a review limit summary that fails the owner text policy": lambda task_id: (
        "review_limit", "The reviewer found a fault in /srv/site/form.py twice.", [], "optional",
    ),
    "a review limit summary of 501 code points": lambda task_id: (
        "review_limit", _question_of(501), [], "optional",
    ),
}


@pytest.mark.parametrize("case", sorted(REFUSED_SHAPES))
def test_every_other_refused_declaration_writes_nothing(board, fired_hooks, case):
    """S1, S2 (and S4, S5 stops are served only): refused with a plain error, nothing written."""
    task_id, run_id = _running_card(board)
    stop, question, choices, words = REFUSED_SHAPES[case](task_id)
    store = fired_hooks.store = _store_file(board)
    before = _observed_state(store, task_id)

    with pytest.raises(ValueError):
        _declare(board, task_id, run_id, stop, question, choices, words)

    assert _observed_state(store, task_id) == before
    assert before["status"] == "running" and before["declarations"] == 0
    assert board.in_transaction is False
    assert fired_hooks.fired == []


# ---------------------------------------------------------------------------
# Card P1 test (c): a review limit stop has the platform's two choices.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "worker_choices",
    [
        [],
        _choices(3),
        _choices(5),
        [{"n": 1, "sentence": "Copy the file at /root/site/form.html first."}],
        None,
    ],
    ids=["none", "three", "five", "a refused one", "not a list"],
)
@pytest.mark.parametrize("worker_words", ["optional", "required"])
def test_card_p1_c_a_review_limit_stop_ignores_worker_choices_and_stores_the_two_fixed_sentences(
    board, worker_choices, worker_words,
):
    """Card P1 test (c), S3: the worker's choices are ignored; the event holds the
    two fixed sentences and optional words, and the summary is the question."""
    task_id, run_id = _running_card(board)

    _declare(board, task_id, run_id, "review_limit", REVIEW_SUMMARY, worker_choices, worker_words)

    assert _declared_payloads(board, task_id) == [{
        "object": "hermes.owner_workspace.owner_question",
        "schema_version": 1,
        "stop": "review_limit",
        "question": REVIEW_SUMMARY,
        "choices": REVIEW_LIMIT_CHOICES,
        "words": "optional",
    }]
    (blocked,) = [e for e in kb.list_events(board, task_id) if e.kind == "blocked"]
    assert blocked.payload["reason"] == REVIEW_SUMMARY
    assert blocked.payload["kind"] == "needs_input"


# ---------------------------------------------------------------------------
# Card P1 test (d): a question is current only while its own hold lasts.
# ---------------------------------------------------------------------------


def test_card_p1_d_a_resumed_and_reblocked_card_has_no_current_question_until_it_declares_again(
    board,
):
    """Card P1 test (d), the ``_waiting_declarations`` rule and S6: a resume ends
    the question for good, a later plain stop does not bring it back, and the
    card may declare again, unlike design options."""
    task_id, run_id = _running_card(board)
    _declare(board, task_id, run_id, "question", QUESTION, _choices(2), "optional")
    (first,) = _declared_payloads(board, task_id)
    first_hold = _blocked_event_ids(board, task_id)[-1]
    assert oq.current_questions(board, [task_id]) == {task_id: (first_hold, first)}

    # An event that leaves the card's status as it is leaves the question too.
    kb.add_comment(board, task_id, "builder", "The card waits for the owner.")
    assert oq.current_questions(board, [task_id]) == {task_id: (first_hold, first)}

    run_id = _resumed_and_claimed(board, task_id)
    assert oq.current_questions(board, [task_id]) == {}

    assert kb.block_task(
        board, task_id, reason=PLAIN_STOP, kind="needs_input", expected_run_id=run_id,
    ) is True
    assert kb.task_owner_waits(board, [task_id])[task_id]["kind"] == "needs_input"
    assert oq.current_questions(board, [task_id]) == {}

    run_id = _resumed_and_claimed(board, task_id)
    _declare(board, task_id, run_id, "question", "Which page should Raphael build next?", [], "required")

    assert _declared_payloads(board, task_id) == [first, {
        "object": "hermes.owner_workspace.owner_question",
        "schema_version": 1,
        "stop": "question",
        "question": "Which page should Raphael build next?",
        "choices": [],
        "words": "required",
    }]
    second_hold = _blocked_event_ids(board, task_id)[-1]
    assert second_hold > first_hold
    assert oq.current_questions(board, [task_id]) == {
        task_id: (second_hold, _declared_payloads(board, task_id)[-1]),
    }


def test_a_held_card_cannot_declare_again_until_it_is_resumed(board, fired_hooks):
    """Card P1: the hold refuses a card that is not running, so a second
    declaration, even the same one, is refused while the first one waits."""
    task_id, run_id = _running_card(board)
    store = fired_hooks.store = _store_file(board)
    _declare(board, task_id, run_id, "question", QUESTION, _choices(2), "optional")
    first = _observed_state(store, task_id)
    current = oq.current_questions(board, [task_id])

    with pytest.raises(ValueError):
        _declare(board, task_id, run_id, "question", "Which page should Raphael build next?", [], "required")
    with pytest.raises(ValueError):
        _declare(board, task_id, run_id, "question", QUESTION, _choices(2), "optional")

    assert _observed_state(store, task_id) == first
    assert oq.current_questions(board, [task_id]) == current
    assert len(fired_hooks.fired) == 1


def test_current_questions_reads_many_cards_at_once_and_only_current_questions(board):
    """Card P1 ``current_questions``: only a card held now by the hold its own
    declaration wrote is listed, with that ``blocked`` event's id (S6)."""
    asked, asked_run = _running_card(board, "sign up page")
    _declare(board, asked, asked_run, "question", QUESTION, _choices(2), "optional")
    reviewed, reviewed_run = _running_card(board, "sign up review")
    _declare(board, reviewed, reviewed_run, "review_limit", REVIEW_SUMMARY, [], "optional")
    plain, plain_run = _running_card(board, "price page")
    assert kb.block_task(
        board, plain, reason=PLAIN_STOP, kind="needs_input", expected_run_id=plain_run,
    ) is True
    running, _ = _running_card(board, "help page")
    resumed, resumed_run = _running_card(board, "about page")
    _declare(board, resumed, resumed_run, "question", QUESTION, _choices(3), "optional")
    assert kb.unblock_task(board, resumed) is True

    current = oq.current_questions(
        board, [asked, reviewed, plain, running, resumed, "t_00000000"],
    )

    assert current == {
        asked: (_blocked_event_ids(board, asked)[-1], _declared_payloads(board, asked)[-1]),
        reviewed: (
            _blocked_event_ids(board, reviewed)[-1], _declared_payloads(board, reviewed)[-1],
        ),
    }
    assert oq.current_questions(board, []) == {}


def test_current_questions_reads_the_held_cards_and_their_events_in_one_snapshot(
    board, monkeypatch,
):
    """Card P1 ``current_questions``: the wait check and the event lookup read
    one snapshot. Between the two, A is resumed, and B is resumed, claimed
    again and declares a question; the read still returns the one state it
    began in. It joins a transaction the caller holds and leaves it open, and
    it ends the one it opened itself.

    The test puts its board in WAL mode first: the interleaving needs a writer
    that can commit while the reader holds its snapshot, which only WAL allows.
    In rollback-journal mode the reader's open read transaction blocks that
    commit, so the read is consistent by locking."""
    # The writer must be able to commit while the reader holds its snapshot.
    assert board.execute("PRAGMA journal_mode=WAL").fetchone()[0] == "wal"
    store = _store_file(board)
    a, a_run = _running_card(board, "sign up page")
    _declare(board, a, a_run, "question", QUESTION, _choices(2), "optional")
    b, b_run = _running_card(board, "price page")
    assert kb.block_task(
        board, b, reason=PLAIN_STOP, kind="needs_input", expected_run_id=b_run,
    ) is True
    only_a = {a: (_blocked_event_ids(board, a)[-1], _declared_payloads(board, a)[-1])}
    # A read-only connection to the board, opened as the owner pages open one.
    reader = sqlite3.connect(Path(store).as_uri() + "?mode=ro", uri=True)
    reader.row_factory = sqlite3.Row
    reader.execute("PRAGMA query_only = ON")
    try:
        reader.execute("BEGIN")
        assert oq.current_questions(reader, [a, b]) == only_a
        assert reader.in_transaction is True
        reader.execute("ROLLBACK")
        with kb.write_txn(board):
            assert oq.current_questions(board, [a, b]) == only_a
            assert board.in_transaction is True
        assert oq.current_questions(reader, [a, b]) == only_a
        assert reader.in_transaction is False

        real_task_owner_waits = kb.task_owner_waits
        b_runs = []

        def wait_check_then_lifecycle(conn, task_ids):
            waits = real_task_owner_waits(conn, task_ids)
            if conn is reader and not b_runs:
                assert kb.unblock_task(board, a) is True
                b_runs.append(_resumed_and_claimed(board, b))
                _declare(
                    board, b, b_runs[0], "question",
                    "Which price page should Raphael build first?", _choices(3), "optional",
                )
            return waits

        monkeypatch.setattr(kb, "task_owner_waits", wait_check_then_lifecycle)
        assert oq.current_questions(reader, [a, b]) == only_a
        assert b_runs and reader.in_transaction is False
        assert oq.current_questions(reader, [a, b]) == {
            b: (_blocked_event_ids(board, b)[-1], _declared_payloads(board, b)[-1]),
        }
        assert _declared_payloads(board, b)[-1] != only_a[a][1]
    finally:
        reader.close()


# ---------------------------------------------------------------------------
# Card P1 test (e): only the worker's current run may declare.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("stale", ["a later run id", "an earlier run of the card"])
def test_card_p1_e_a_run_that_is_no_longer_current_writes_nothing(board, fired_hooks, stale):
    """Card P1 test (e): the hold compares the card's current run, so a run that
    is no longer current is refused and nothing is written."""
    task_id, run_id = _running_card(board)
    store = fired_hooks.store = _store_file(board)
    if stale == "a later run id":
        stale_run = run_id + 1
    else:
        stale_run = run_id
        assert kb.block_task(
            board, task_id, reason=PLAIN_STOP, kind="needs_input", expected_run_id=run_id,
        ) is True
        assert _resumed_and_claimed(board, task_id) != stale_run
    before = _observed_state(store, task_id)
    fired_before = list(fired_hooks.fired)

    with pytest.raises(ValueError):
        _declare(board, task_id, stale_run, "question", QUESTION, _choices(2), "optional")

    assert _observed_state(store, task_id) == before
    assert before["status"] == "running" and before["open_runs"] == 1
    assert before["declarations"] == 0
    assert board.in_transaction is False
    assert fired_hooks.fired == fired_before
    assert oq.current_questions(board, [task_id]) == {}


def _card_and_runs(store_file: str, task_id: str) -> tuple[list, list]:
    """The card's row and every run of it, as a second, read-only connection sees them."""
    observer = sqlite3.connect(Path(store_file).as_uri() + "?mode=ro", uri=True)
    try:
        return (
            observer.execute("SELECT * FROM tasks WHERE id = ?", (task_id,)).fetchall(),
            observer.execute(
                "SELECT * FROM task_runs WHERE task_id = ? ORDER BY id", (task_id,),
            ).fetchall(),
        )
    finally:
        observer.close()


def test_no_run_id_on_a_card_a_newer_run_claimed_writes_nothing(board, fired_hooks):
    """Card P1: a declaration must name the worker's run. Without a run id it
    is refused before anything is written, so the newer run of a card that was
    resumed and claimed again stays its current run, and stays open."""
    task_id, first_run = _running_card(board)
    assert kb.block_task(
        board, task_id, reason=PLAIN_STOP, kind="needs_input", expected_run_id=first_run,
    ) is True
    newer_run = _resumed_and_claimed(board, task_id)
    store = fired_hooks.store = _store_file(board)
    before, rows = _observed_state(store, task_id), _card_and_runs(store, task_id)
    fired_before = list(fired_hooks.fired)

    with pytest.raises(ValueError, match="run id is required"):
        _declare(board, task_id, None, "question", QUESTION, _choices(2), "optional")

    assert _card_and_runs(store, task_id) == rows
    assert _observed_state(store, task_id) == before
    assert _observe(store, "SELECT current_run_id FROM tasks WHERE id = ?", task_id) == newer_run
    assert _observe(store, "SELECT ended_at FROM task_runs WHERE id = ?", newer_run) is None
    assert before["status"] == "running" and before["open_runs"] == 1
    assert before["declarations"] == 0 and len(_blocked_event_ids(board, task_id)) == 1
    assert board.in_transaction is False
    assert fired_hooks.fired == fired_before
    assert oq.current_questions(board, [task_id]) == {}


@pytest.mark.parametrize("run", ["no run id", "the run id of another card"])
def test_a_ready_card_no_worker_claimed_cannot_declare(board, fired_hooks, run):
    """Card P1: a ready card has no current run, so no run may hold it. Without
    a run id the call is refused; with one, the hold's current run compare
    refuses it and the event is rolled back. Nothing is written."""
    _, other_run = _running_card(board, "price page")
    task_id = kb.create_task(board, title="sign up page", assignee="builder")
    store = fired_hooks.store = _store_file(board)
    before, rows = _observed_state(store, task_id), _card_and_runs(store, task_id)

    with pytest.raises(ValueError):
        _declare(
            board, task_id, None if run == "no run id" else other_run,
            "question", QUESTION, _choices(2), "optional",
        )

    assert _card_and_runs(store, task_id) == rows
    assert _observed_state(store, task_id) == before
    assert before["status"] == "ready" and before["open_runs"] == 0
    assert _observe(store, "SELECT current_run_id FROM tasks WHERE id = ?", task_id) is None
    assert before["declarations"] == 0 and _blocked_event_ids(board, task_id) == []
    assert board.in_transaction is False
    assert fired_hooks.fired == []
    assert oq.current_questions(board, [task_id]) == {}


@pytest.mark.parametrize(
    "run_id", [True, False, 0, -1, "1", 1.0], ids=["True", "False", "0", "-1", "'1'", "1.0"],
)
def test_a_run_id_that_is_not_a_whole_number_of_at_least_1_writes_nothing(
    board, fired_hooks, run_id,
):
    """Card P1: the run id is a whole number of at least 1. True, "1" and 1.0
    would name run 1 in SQL, and False, 0 and -1 no run; each is refused
    before anything is written."""
    task_id, current_run = _running_card(board)
    assert current_run == 1
    store = fired_hooks.store = _store_file(board)
    before, rows = _observed_state(store, task_id), _card_and_runs(store, task_id)

    with pytest.raises(ValueError, match="run id is required"):
        _declare(board, task_id, run_id, "question", QUESTION, _choices(2), "optional")

    assert _card_and_runs(store, task_id) == rows
    assert _observed_state(store, task_id) == before
    assert before["status"] == "running" and before["open_runs"] == 1
    assert before["declarations"] == 0
    assert board.in_transaction is False
    assert fired_hooks.fired == []
    assert oq.current_questions(board, [task_id]) == {}
