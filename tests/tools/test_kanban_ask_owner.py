"""``kanban_ask_owner`` — real registry dispatch, real boards.

Every test runs under the exact environment the dispatcher injects into a
worker: ``HERMES_KANBAN_DB`` and ``HERMES_KANBAN_BOARD`` pinned to the worker's
own board, ``HERMES_KANBAN_TASK`` set to its card and ``HERMES_HOME`` pointing at
its profile home under the root, with two more boards beside its own. The
worker asks its one question through the registry. Each proof is read again
from the board store files by a separate read-only connection, never from the
tool's own answer. The tool is built as ``kanban_offer_design_options`` is
built, so a refusal is compared with that tool's refusal for the same caller,
and a refused value with the plain error of ``declare_owner_question`` itself.
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import pytest

from hermes_cli.owner_questions import (
    MAX_CHOICE_LENGTH,
    MAX_CHOICES,
    MAX_QUESTION_LENGTH,
    REVIEW_LIMIT_CHOICES,
    STOPS,
    WORDS,
)

TOOL = "kanban_ask_owner"
MODEL = "kanban_offer_design_options"
DECLARED_EVENT = "owner_question_declared"
QUESTION = "Which page should I build first, the home page or the shop page?"
CHOICES = [
    {"n": 1, "sentence": "Build the home page first."},
    {"n": 2, "sentence": "Build the shop page first."},
]
ASK = {"question": QUESTION, "choices": CHOICES, "words": "optional"}
BOARDS = ("default", "own-board", "other-board")
CALLERS = ("worker", "delegated child", "no card", "not spawned by the dispatcher", "review task")
REFUSED_CALLERS = CALLERS[1:]
LISTED = {
    "worker": True,
    "delegated child": False,
    "no card": False,
    "not spawned by the dispatcher": False,
    "review task": True,
}


# ---------------------------------------------------------------------------
# Copied verbatim from tests/tools/test_kanban_tools.py (copied, not imported,
# so this file stands on its own).
# ---------------------------------------------------------------------------

def _kanban_worker_schema_names(monkeypatch):
    """Tool names a dispatcher-spawned worker actually sees.

    Mirrors model_tools' worker path: the ``kanban`` toolset is folded in for
    a dispatcher-owned worker, then each entry's own check_fn decides.
    """
    import tools.kanban_tools  # noqa: F401  (ensure registered)
    from tools.registry import invalidate_check_fn_cache, registry
    from toolsets import resolve_toolset

    invalidate_check_fn_cache()
    schema = registry.get_definitions(set(resolve_toolset("kanban")), quiet=True)
    names = {s["function"].get("name") for s in schema if "function" in s}
    invalidate_check_fn_cache()
    return {n for n in names if n and n.startswith("kanban_")}


# ---------------------------------------------------------------------------
# Helpers for this file
# ---------------------------------------------------------------------------

def _store_path(slug):
    """The board's own store file, resolved from its slug alone."""
    from hermes_cli import kanban_db as kb

    if slug == kb.DEFAULT_BOARD:
        return kb.kanban_home() / "kanban.db"
    return kb.board_dir(slug) / "kanban.db"


def _claimed_card(slug, title):
    """One card on board ``slug``, claimed by the dispatcher and running."""
    from hermes_cli import kanban_db as kb

    conn = kb.connect(board=slug)
    try:
        task_id = kb.create_task(conn, title=title, assignee="builder", board=slug)
        claimed = kb.claim_task(conn, task_id)
    finally:
        conn.close()
    assert claimed is not None and claimed.status == "running"
    return claimed


def _read(store, query, *params):
    """Rows of ``query`` as a separate read-only connection to ``store`` sees them."""
    conn = sqlite3.connect(f"{Path(store).resolve().as_uri()}?mode=ro", uri=True)
    try:
        return conn.execute(query, params).fetchall()
    finally:
        conn.close()


def _board_state(stores):
    """Per board: each card's (id, status, block_kind) and its declaration events."""
    return {
        slug: (
            _read(store, "SELECT id, status, block_kind FROM tasks ORDER BY id"),
            _read(
                store, "SELECT task_id, payload FROM task_events WHERE kind = ? ORDER BY id",
                DECLARED_EVENT,
            ),
        )
        for slug, store in stores.items()
    }


def _every_row(stores):
    """Per board: every row of every table, so that no write of any kind goes unseen."""
    whole = {}
    for slug, store in stores.items():
        tables = [
            name for (name,) in _read(
                store,
                "SELECT name FROM sqlite_master WHERE type = 'table' "
                "AND name NOT LIKE 'sqlite_%' ORDER BY name",
            )
        ]
        whole[slug] = {
            name: sorted(repr(row) for row in _read(store, f'SELECT * FROM "{name}"'))
            for name in tables
        }
    return whole


def _dispatch(name, args):
    from tools.registry import registry

    return json.loads(registry.dispatch(name, args))


def _the_one_declaration(worker):
    """The card and the four values of the only declaration on any board."""
    state = _board_state(worker.stores)
    assert state["default"][1] == [] and state["other-board"][1] == []
    ((declared_on, payload),) = state["own-board"][1]
    recorded = json.loads(payload)
    return declared_on, (
        recorded["stop"], recorded["question"], recorded["choices"], recorded["words"],
    )


def _the_kernels_message(worker, args):
    """What ``declare_owner_question`` itself says to the same four values, on the
    worker's own card with the worker's own run."""
    from hermes_cli import kanban_db as kb
    from hermes_cli.owner_questions import declare_owner_question

    conn = kb.connect(board="own-board")
    try:
        with pytest.raises(ValueError) as refused:
            declare_owner_question(
                conn, worker.mine.id, args.get("stop", "question"), args.get("question"),
                args.get("choices"), args.get("words"),
                board="own-board", expected_run_id=worker.mine.current_run_id,
            )
    finally:
        conn.close()
    return str(refused.value)


def _text_of(length, lead=""):
    """Plain words of exactly ``length`` characters, ending on a letter."""
    unit = "more words "
    text = (lead + unit * (length // len(unit) + 2))[:length]
    return text if text[-1].isalpha() else text[:-1] + "e"


def _question_at_the_limit():
    """A question of exactly the longest length, over two lines."""
    text = _text_of(MAX_QUESTION_LENGTH)
    at = text.index(" ", 120)
    return text[:at] + "\n" + text[at + 1:]


def _choices_at_the_limit():
    return [
        {"n": n, "sentence": _text_of(MAX_CHOICE_LENGTH, f"Choice {n} is ")}
        for n in range(1, MAX_CHOICES + 1)
    ]


def _numbered(count):
    return [{"n": n, "sentence": f"Build page number {n} first."} for n in range(1, count + 1)]


def _become_the_worker_of(card, monkeypatch):
    """Point the dispatcher's env at ``card``, a claimed card on ``own-board``."""
    from hermes_cli import kanban_db as kb

    monkeypatch.setenv("HERMES_KANBAN_TASK", card.id)
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(card.current_run_id))
    monkeypatch.setenv("HERMES_KANBAN_CLAIM_LOCK", card.claim_lock)
    monkeypatch.setenv(
        "HERMES_KANBAN_WORKSPACE", str(kb.workspaces_root(board="own-board") / card.id),
    )


def _become_the_worker_of_a_review_task(monkeypatch):
    """The dispatcher's worker of a review task of a pull request, on ``own-board``."""
    from hermes_cli import kanban_db as kb

    conn = kb.connect(board="own-board")
    try:
        task_id = kb.create_task(
            conn, title="review of the home page", assignee="builder",
            created_by=kb.REVIEW_TASK_CREATOR,
            idempotency_key="project-review:prj_home:2026-11-27:0", board="own-board",
        )
        review = kb.claim_task(conn, task_id)
    finally:
        conn.close()
    assert review is not None and review.status == "running"
    _become_the_worker_of(review, monkeypatch)
    return review


@pytest.fixture
def worker(tmp_path, monkeypatch):
    """Three boards under one root and the env the dispatcher's spawn gives
    the builder for its card on ``own-board``.
    """
    from hermes_cli import kanban_db as kb
    from hermes_cli.profiles import get_active_profile_name, resolve_profile_env
    from tools.registry import discover_builtin_tools

    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    root = tmp_path / ".hermes"
    root.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(root))
    for key in (
        "HERMES_KANBAN_TASK", "HERMES_KANBAN_DB", "HERMES_KANBAN_BOARD",
        "HERMES_KANBAN_RUN_ID", "HERMES_KANBAN_CLAIM_LOCK",
        "HERMES_KANBAN_WORKSPACE", "HERMES_KANBAN_WORKSPACES_ROOT",
        "HERMES_KANBAN_ATTACHMENTS_ROOT", "HERMES_PROFILE", "HERMES_SESSION_ID",
    ):
        monkeypatch.delenv(key, raising=False)

    kb._INITIALIZED_PATHS.clear()
    kb.init_db()
    kb.create_board("own-board")
    kb.create_board("other-board")
    mine = _claimed_card("own-board", "home page build")
    sibling = _claimed_card("own-board", "menu page build")
    foreign = _claimed_card("other-board", "shop page build")
    default_card = _claimed_card(kb.DEFAULT_BOARD, "about page build")
    stores = {slug: _store_path(slug) for slug in BOARDS}
    workspaces = kb.workspaces_root(board="own-board")

    (root / "profiles" / "builder").mkdir(parents=True)
    profile_home = Path(resolve_profile_env("builder"))
    # The env the dispatcher's spawn gives a worker for a card on own-board.
    monkeypatch.setenv("HERMES_HOME", str(profile_home))
    monkeypatch.setenv("HERMES_KANBAN_TASK", mine.id)
    monkeypatch.setenv("HERMES_KANBAN_WORKSPACE", str(workspaces / mine.id))
    monkeypatch.setenv("HERMES_SESSION_SOURCE", "kanban")
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(mine.current_run_id))
    monkeypatch.setenv("HERMES_KANBAN_CLAIM_LOCK", mine.claim_lock)
    monkeypatch.setenv("HERMES_KANBAN_DB", str(stores["own-board"]))
    monkeypatch.setenv("HERMES_KANBAN_WORKSPACES_ROOT", str(workspaces))
    monkeypatch.setenv("HERMES_KANBAN_BOARD", "own-board")
    monkeypatch.setenv("HERMES_PROFILE", "builder")

    # Positive controls: the pin is live (the resolver hands back the own
    # store for any slug), the profile resolves from the env alone, the
    # built-in discovery imports the tool's module by name, and the worker's
    # schema lists the tool.
    assert kb.kanban_db_path(board="other-board") == stores["own-board"]
    assert get_active_profile_name() == "builder"
    assert "tools.kanban_ask_owner" in discover_builtin_tools()
    assert TOOL in _kanban_worker_schema_names(monkeypatch)
    return SimpleNamespace(
        root=root, profile_home=profile_home, stores=stores, mine=mine,
        sibling=sibling, foreign=foreign, default_card=default_card,
    )


@pytest.fixture
def caller(request, worker, monkeypatch):
    """The caller of the tool as one case of ``CALLERS``, for the whole test."""
    from agent.delegation_context import delegated_child_context, non_dispatcher_owned_context

    kind = request.param
    if kind == "no card":
        monkeypatch.delenv("HERMES_KANBAN_TASK")
    elif kind == "review task":
        _become_the_worker_of_a_review_task(monkeypatch)
    with contextlib.ExitStack() as contexts:
        if kind == "delegated child":
            contexts.enter_context(delegated_child_context())
        elif kind == "not spawned by the dispatcher":
            contexts.enter_context(non_dispatcher_owned_context())
        yield kind


# ---------------------------------------------------------------------------
# A worker on its own card
# ---------------------------------------------------------------------------

def test_the_worker_asks_the_owner_on_its_own_card_on_its_own_board(worker):
    before = _every_row(worker.stores)

    out = _dispatch(TOOL, ASK)

    assert out == {
        "ok": True,
        "task_id": worker.mine.id,
        "status": "blocked",
        "block_kind": "needs_input",
    }
    after = _every_row(worker.stores)
    # Only the worker's own board changed: its card alone is held, with one
    # declaration; every other card on every board is as it was.
    assert {slug: after[slug] for slug in ("default", "other-board")} == {
        slug: before[slug] for slug in ("default", "other-board")
    }
    own_cards, _ = _board_state(worker.stores)["own-board"]
    assert own_cards == sorted([
        (worker.mine.id, "blocked", "needs_input"),
        (worker.sibling.id, "running", None),
    ])
    declared_on, values = _the_one_declaration(worker)
    assert declared_on == worker.mine.id
    assert values == ("question", QUESTION, CHOICES, "optional")
    assert not (worker.profile_home / "kanban").exists()


def test_the_stop_is_question_when_it_is_not_given(worker):
    assert "stop" not in ASK

    out = _dispatch(TOOL, ASK)

    assert "error" not in out, out
    _, (stop, *_rest) = _the_one_declaration(worker)
    assert stop == "question"


def test_the_stop_review_limit_records_the_two_fixed_choices(worker):
    summary = (
        "The reviewer asked for three corrections: the title, the colours and "
        "the order of the menu."
    )

    out = _dispatch(TOOL, {
        "stop": "review_limit", "question": summary,
        "choices": CHOICES, "words": "required",
    })

    assert "error" not in out, out
    declared_on, values = _the_one_declaration(worker)
    assert declared_on == worker.mine.id
    # The worker's own choices and words are ignored; the two fixed choices
    # and optional words are recorded.
    assert values == (
        "review_limit",
        summary,
        [{"n": n, "sentence": sentence} for n, sentence in enumerate(REVIEW_LIMIT_CHOICES, start=1)],
        "optional",
    )
    own_cards, _ = _board_state(worker.stores)["own-board"]
    assert (worker.mine.id, "blocked", "needs_input") in own_cards


def test_a_question_with_no_choices_and_the_words_required_is_recorded(worker):
    out = _dispatch(TOOL, {"question": QUESTION, "choices": [], "words": "required"})

    assert "error" not in out, out
    declared_on, values = _the_one_declaration(worker)
    assert declared_on == worker.mine.id
    assert values == ("question", QUESTION, [], "required")
    own_cards, _ = _board_state(worker.stores)["own-board"]
    assert (worker.mine.id, "blocked", "needs_input") in own_cards


def test_the_recorded_question_is_exactly_the_four_values_the_worker_gave(worker):
    question, choices = _question_at_the_limit(), _choices_at_the_limit()
    assert len(question) == MAX_QUESTION_LENGTH and len(choices) == MAX_CHOICES
    assert {len(choice["sentence"]) for choice in choices} == {MAX_CHOICE_LENGTH}

    out = _dispatch(TOOL, {
        "stop": "question", "question": question, "choices": choices, "words": "required",
    })

    assert "error" not in out, out
    declared_on, values = _the_one_declaration(worker)
    assert declared_on == worker.mine.id
    assert values == ("question", question, choices, "required")


# Texts the kernel accepts and keeps as they come; a tool that tidies them up
# (strips, joins lines, squeezes spaces) records something the worker did not give.
VERBATIM = [
    pytest.param(f"  {QUESTION}  ", CHOICES, id="a-question-with-spaces-at-both-ends"),
    pytest.param(f"{QUESTION}\n", CHOICES, id="a-question-ending-on-a-new-line"),
    pytest.param("Quelle page faut-il construire d'abord ?", CHOICES, id="a-question-with-accents"),
    pytest.param(
        QUESTION,
        [
            {"n": 1, "sentence": "  Build the home page first.  "},
            {"n": 2, "sentence": "Build the shop  page first."},
        ],
        id="choices-with-extra-spaces",
    ),
]


@pytest.mark.parametrize(("question", "choices"), VERBATIM)
def test_a_text_the_kernel_keeps_as_it_comes_is_recorded_as_it_came(worker, question, choices):
    out = _dispatch(TOOL, {"question": question, "choices": choices, "words": "optional"})

    assert "error" not in out, out
    assert _the_one_declaration(worker) == (worker.mine.id, ("question", question, choices, "optional"))


def test_the_card_held_is_the_one_the_dispatcher_named_in_hermes_kanban_task(worker, monkeypatch):
    _become_the_worker_of(worker.sibling, monkeypatch)

    out = _dispatch(TOOL, ASK)

    assert "error" not in out, out
    own_cards, _ = _board_state(worker.stores)["own-board"]
    assert own_cards == sorted([
        (worker.mine.id, "running", None),
        (worker.sibling.id, "blocked", "needs_input"),
    ])
    assert _the_one_declaration(worker)[0] == worker.sibling.id


def test_under_the_dispatchers_env_the_question_lands_once_on_the_pinned_boards_own_card(
    worker, monkeypatch,
):
    from hermes_cli import kanban_db as kb

    # The exact env of a dispatcher's worker.
    assert os.environ["HERMES_KANBAN_DB"] == str(worker.stores["own-board"])
    assert os.environ["HERMES_KANBAN_BOARD"] == "own-board"
    assert os.environ["HERMES_KANBAN_TASK"] == worker.mine.id
    assert os.environ["HERMES_HOME"] == str(worker.profile_home)
    assert worker.profile_home.parent == worker.root / "profiles"
    hooks = []
    fire = kb._fire_kanban_lifecycle_hook

    def spy(event, task_id, **fields):
        hooks.append((event, task_id, fields.get("board")))
        return fire(event, task_id, **fields)

    monkeypatch.setattr(kb, "_fire_kanban_lifecycle_hook", spy)

    out = _dispatch(TOOL, ASK)

    assert "error" not in out, out
    # An independent count: every board store under the home, found by walking
    # the tree and not through the fixture's own list of stores.
    found = sorted(path.resolve() for path in worker.root.parent.rglob("kanban.db"))
    assert found == sorted(store.resolve() for store in worker.stores.values())
    declared = [
        (store, task_id)
        for store in found
        for (task_id,) in _read(
            store, "SELECT task_id FROM task_events WHERE kind = ?", DECLARED_EVENT,
        )
    ]
    assert declared == [(worker.stores["own-board"].resolve(), worker.mine.id)]
    assert not (worker.profile_home / "kanban").exists()
    # The owner's side is told the board the card is on.
    assert hooks == [("kanban_task_blocked", worker.mine.id, "own-board")]


# ---------------------------------------------------------------------------
# Callers and cards the tool refuses; nothing is written
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("caller", REFUSED_CALLERS, indirect=True)
def test_a_caller_the_design_options_tool_refuses_is_refused_the_same_way(worker, caller):
    before = _every_row(worker.stores)

    out = _dispatch(TOOL, ASK)
    model = _dispatch(MODEL, {})

    assert "error" in model and "ok" not in model  # the caller is one that tool refuses
    assert out == {"error": model["error"].replace(MODEL, TOOL)}
    assert _every_row(worker.stores) == before


@pytest.mark.parametrize("caller", CALLERS, indirect=True)
def test_the_checks_come_in_the_order_of_the_design_options_tool(worker, caller):
    # A stray task_id on top of the caller's own fault: the answer shows which check
    # comes first, and it is the one the design options tool gives for the same call.
    stray = worker.sibling.id
    before = _every_row(worker.stores)

    out = _dispatch(TOOL, {"task_id": stray, **ASK})
    model = _dispatch(MODEL, {"task_id": stray})

    assert "error" in model and "ok" not in model
    assert out == {"error": model["error"].replace(MODEL, TOOL)}
    assert _every_row(worker.stores) == before


def test_the_tool_takes_its_helpers_from_kanban_tools_as_the_design_options_tool_does():
    import tools.kanban_ask_owner as ask
    import tools.kanban_design_options as model
    import tools.kanban_tools as kanban_tools

    for name in (
        "_board_of_connection", "_connect", "_enforce_worker_task_ownership",
        "_is_delegated_child_context", "_is_dispatcher_owned_worker", "_ok",
        "_reject_delegated_child_mutation", "_worker_run_id",
    ):
        assert getattr(model, name) is getattr(kanban_tools, name), name
        assert getattr(ask, name) is getattr(kanban_tools, name), name


@pytest.mark.parametrize("target", ["sibling", "foreign", "default_card"])
def test_a_stray_task_id_naming_another_card_is_refused_and_nothing_is_written(worker, target):
    stray = {"task_id": getattr(worker, target).id, **ASK}
    before = _every_row(worker.stores)

    out = _dispatch(TOOL, stray)
    model = _dispatch(MODEL, {"task_id": stray["task_id"]})

    assert "error" in model and "ok" not in model
    assert out == {"error": model["error"].replace(MODEL, TOOL)}
    assert _every_row(worker.stores) == before


@pytest.mark.parametrize("run", ["stale", "absent"])
def test_a_run_that_is_not_the_cards_current_run_is_refused_and_nothing_is_written(
    worker, monkeypatch, run,
):
    if run == "absent":
        monkeypatch.delenv("HERMES_KANBAN_RUN_ID")
    else:
        monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(worker.mine.current_run_id + 1))
    before = _every_row(worker.stores)

    out = _dispatch(TOOL, ASK)

    assert "error" in out and "ok" not in out
    assert _every_row(worker.stores) == before


def test_a_second_question_on_a_held_card_is_refused_and_the_first_stays_alone(worker):
    first = _dispatch(TOOL, ASK)
    assert "error" not in first, first
    held = _every_row(worker.stores)

    out = _dispatch(TOOL, {
        "question": "Which colour should the buttons be?", "choices": [], "words": "required",
    })

    assert "error" in out and "ok" not in out
    assert _every_row(worker.stores) == held
    assert _the_one_declaration(worker) == (worker.mine.id, ("question", QUESTION, CHOICES, "optional"))


# ---------------------------------------------------------------------------
# Values the kernel refuses come back as its plain error; nothing is written
# ---------------------------------------------------------------------------

def test_five_choices_come_back_as_the_plain_error_and_nothing_is_written(worker):
    args = {**ASK, "choices": _numbered(MAX_CHOICES + 1)}
    before = _every_row(worker.stores)

    out = _dispatch(TOOL, args)

    assert _every_row(worker.stores) == before
    message = _the_kernels_message(worker, args)
    assert f"0 to {MAX_CHOICES}" in message
    assert out == {"error": f"{TOOL}: {message}"}
    assert _every_row(worker.stores) == before


REFUSED_VALUES = [
    pytest.param({**ASK, "stop": "pause"}, id="a-stop-that-is-not-offered"),
    pytest.param({"question": QUESTION, "choices": CHOICES}, id="no-words"),
    pytest.param({"question": QUESTION, "words": "required"}, id="no-choices-key"),
    pytest.param({**ASK, "choices": [], "words": "optional"}, id="no-choices-and-optional-words"),
    pytest.param(
        {**ASK, "choices": [{**CHOICES[0], "picture": "calm"}]}, id="a-choice-with-an-extra-key",
    ),
    pytest.param(
        {**ASK, "choices": [{"n": 2, "sentence": CHOICES[0]["sentence"]}]},
        id="choices-not-numbered-from-one",
    ),
    pytest.param(
        {**ASK, "choices": [{"n": 1, "sentence": "Build the home page.\nThen the shop page."}]},
        id="a-choice-over-two-lines",
    ),
    pytest.param({**ASK, "question": _text_of(MAX_QUESTION_LENGTH + 1)}, id="a-question-over-the-limit"),
    pytest.param(
        {**ASK, "question": "Please open /home/user/site/index.html and tell me what to fix first."},
        id="a-path-in-the-question",
    ),
]


@pytest.mark.parametrize("args", REFUSED_VALUES)
def test_a_value_the_kernel_refuses_comes_back_as_its_plain_error(worker, args):
    before = _every_row(worker.stores)

    out = _dispatch(TOOL, args)

    assert _every_row(worker.stores) == before
    assert out == {"error": f"{TOOL}: {_the_kernels_message(worker, args)}"}
    assert _every_row(worker.stores) == before


def test_an_unexpected_failure_comes_back_as_the_tool_error_and_nothing_is_written(
    worker, monkeypatch,
):
    import hermes_cli.owner_questions as owner_questions

    def unavailable(*args, **kwargs):
        raise RuntimeError("the board store is busy")

    monkeypatch.setattr(owner_questions, "declare_owner_question", unavailable)
    before = _every_row(worker.stores)

    out = _dispatch(TOOL, ASK)

    assert out == {"error": f"{TOOL}: the board store is busy"}
    assert _every_row(worker.stores) == before


# ---------------------------------------------------------------------------
# Listing, schema and toolsets
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("caller", CALLERS, indirect=True)
def test_the_tool_is_listed_to_the_same_callers_as_the_design_options_tool(
    worker, caller, monkeypatch,
):
    from tools.registry import registry

    ask, model = registry.get_entry(TOOL).check_fn, registry.get_entry(MODEL).check_fn
    assert bool(model()) is LISTED[caller]  # the case really is the case it names
    assert bool(ask()) == bool(model())
    names = _kanban_worker_schema_names(monkeypatch)
    assert (TOOL in names) == (MODEL in names) == LISTED[caller]


def _schema():
    from tools.kanban_tools_schemas import KANBAN_ASK_OWNER_SCHEMA

    return KANBAN_ASK_OWNER_SCHEMA


def _descriptions_of(schema):
    """Every description in the schema, joined, in lower case."""
    found = []

    def walk(node):
        if isinstance(node, dict):
            for key, value in node.items():
                if key == "description" and isinstance(value, str):
                    found.append(value)
                else:
                    walk(value)
        elif isinstance(node, (list, tuple)):
            for item in node:
                walk(item)

    walk(schema)
    return " ".join(found).lower()


def test_the_schema_names_the_four_parameters_and_no_other():
    schema = _schema()

    assert schema["name"] == TOOL
    parameters = schema["parameters"]
    properties = parameters["properties"]
    assert set(properties) == {"question", "choices", "words", "stop"}
    assert "task_id" not in properties and "board" not in properties
    assert set(parameters["required"]) == {"question", "choices", "words"}
    assert properties["question"]["type"] == "string"
    assert properties["choices"]["type"] == "array"
    entry = properties["choices"]["items"]
    assert set(entry["properties"]) == {"n", "sentence"}
    assert set(entry["required"]) == {"n", "sentence"}
    assert all(value in properties["words"]["description"] for value in WORDS)
    assert all(value in properties["stop"]["description"] for value in STOPS)


def test_the_schema_tells_the_worker_the_rules_of_the_owner():
    text = _descriptions_of(_schema())

    for phrase in (
        "plain words", "not a programmer",
        "file path", "card id", "model name",
        f"at most {MAX_QUESTION_LENGTH} characters",
        f"at most {MAX_CHOICES} choices",
        f"one line of at most {MAX_CHOICE_LENGTH} characters",
        "review_limit", "review round limit", "summary of the corrections",
        "held for the owner", "end the turn",
    ):
        assert phrase in text, phrase
    assert str(MAX_QUESTION_LENGTH) in re.findall(r"\d+", text)
    assert str(MAX_CHOICE_LENGTH) in re.findall(r"\d+", text)


def test_the_schema_says_one_line_and_not_one_sentence():
    text = _descriptions_of(_schema())

    assert "one line" in text
    assert "one sentence" not in text


def _lists_holding(name):
    import toolsets

    return sorted(
        [f"TOOLSETS[{key!r}]" for key, spec in toolsets.TOOLSETS.items() if name in spec.get("tools", ())]
        + [
            attr for attr in ("_HERMES_CORE_TOOLS", "_HERMES_WEBHOOK_SAFE_TOOLS")
            if name in getattr(toolsets, attr)
        ]
    )


def test_the_name_is_in_the_kanban_toolset_and_in_the_lists_of_the_design_options_tool():
    from tools.registry import registry
    from toolsets import resolve_toolset

    import tools.kanban_ask_owner  # noqa: F401  (ensure registered)

    assert registry.get_toolset_for_tool(TOOL) == "kanban"
    assert TOOL in resolve_toolset("kanban", include_registry=False)
    # The design options tool is in the core list and in the kanban toolset;
    # the new name is in those two lists and in no other.
    assert {"TOOLSETS['kanban']", "_HERMES_CORE_TOOLS"} <= set(_lists_holding(MODEL))
    assert _lists_holding(TOOL) == _lists_holding(MODEL)


@pytest.mark.usefixtures("worker")
@pytest.mark.parametrize("platform", ["cli", "cron"])
def test_a_saved_toolset_list_still_recovers_the_kanban_toolset(platform):
    from hermes_cli.tools_config import _get_platform_tools
    from toolsets import resolve_toolset

    # The worker's spawn pin and a cron run take their toolsets from the
    # platform resolution. With the tool in the kanban toolset, a profile
    # that saved its own list still gets kanban back, and every kanban tool with it.
    saved = {"platform_toolsets": {platform: ["terminal", "web"]}}

    enabled = _get_platform_tools(saved, platform)

    assert {"terminal", "web", "kanban"} <= enabled
    offered = {name for toolset in enabled for name in resolve_toolset(toolset)}
    assert {"kanban_complete", "kanban_block", TOOL} <= offered
