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
