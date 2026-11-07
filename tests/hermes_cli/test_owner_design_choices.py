"""The designer offers two or three pictured options and its card waits.

Card 1 of the design options plan, platform tests 1 and 2 of section G. The
designer stores each picture as an ordinary attachment on its own card, then
declares one question and two or three options. In one write transaction
``owner_design_choices.declare_design_options`` checks every limit of section
A, records one ``design_options_declared`` task event and holds the card
``needs_input`` with the plain hold sentence; ``kanban_task_blocked`` fires
once that transaction committed. Every refusal writes nothing. Nothing is
shown to the owner yet.

Every count a test proves is read by a second, read-only connection to the
board's own store, independent of the connection that wrote.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from types import SimpleNamespace

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import owner_design_choices as odc
from hermes_cli.owner_text import check_owner_text
from hermes_cli.plugins import get_plugin_manager

QUESTION = "Which home page should Raphael build?"
SENTENCES = (
    "A calm layout with a large photo at the top.",
    "A bold layout with a big title and two buttons.",
    "A simple layout with one short welcome line.",
)
ALTS = (
    "The home page with a wide photo above a short welcome.",
    "The home page with a large title and two buttons below it.",
    "The home page with one line of text on a plain background.",
)
PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 24
JPEG = b"\xff\xd8\xff\xe0" + b"\x00" * 24
WEBP = b"RIFF" + (24).to_bytes(4, "little") + b"WEBPVP8 " + b"\x00" * 16
GIF = b"GIF89a" + b"\x00" * 24
SVG = b'<svg xmlns="http://www.w3.org/2000/svg" width="1" height="1"/>'
FOUR_MIB = 4 * 1024 * 1024


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
            task_id, odc.DECLARED_EVENT,
        ),
        "events": _observe(store_file, "SELECT COUNT(*) FROM task_events"),
        "open_runs": _observe(
            store_file,
            "SELECT COUNT(*) FROM task_runs WHERE task_id = ? AND ended_at IS NULL",
            task_id,
        ),
    }


def _running_card(conn: sqlite3.Connection, title: str = "home page design") -> tuple[str, int]:
    """(card, run) of a card the designer claimed and is working on."""
    task_id = kb.create_task(conn, title=title, assignee="designer")
    claimed = kb.claim_task(conn, task_id)
    assert claimed is not None and claimed.status == "running"
    return task_id, claimed.current_run_id


def _picture(
    conn: sqlite3.Connection,
    task_id: str,
    data: bytes = PNG,
    content_type: str | None = "image/png",
    filename: str = "option.png",
) -> str:
    """The id, as a decimal string, of a picture the designer attached."""
    return str(kb.store_attachment_bytes(
        conn, task_id, filename, data, content_type=content_type,
        uploaded_by="agent", board=kb.DEFAULT_BOARD,
    ))


def _options(*attachment_ids: str) -> list[dict]:
    return [
        {
            "n": n,
            "sentence": SENTENCES[n - 1],
            "pictures": [{"attachment_id": attachment_id, "alt": ALTS[n - 1]}],
        }
        for n, attachment_id in enumerate(attachment_ids, start=1)
    ]


def _declare(conn, task_id, run_id, question, options):
    return odc.declare_design_options(
        conn, task_id, question, options,
        board=kb.DEFAULT_BOARD, expected_run_id=run_id,
    )


def test_the_hold_sentence_passes_the_owner_text_policy_unchanged():
    assert odc.HOLD_SENTENCE == "Raphael needs you to choose one design."
    assert check_owner_text(odc.HOLD_SENTENCE, "stop_reason_decision") == odc.HOLD_SENTENCE


# ---------------------------------------------------------------------------
# Platform test 1: the declaration accepts 2 or 3 options and refuses the rest.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("count", [2, 3])
def test_a_declaration_of_two_or_three_options_is_accepted(board, count):
    task_id, run_id = _running_card(board)
    pictures = [
        (PNG, "image/png", "calm.png"),
        (JPEG, "image/jpeg", "bold.jpg"),
        (WEBP, "image/webp", "simple.webp"),
    ][:count]
    ids = [_picture(board, task_id, data, media, name) for data, media, name in pictures]

    assert _declare(board, task_id, run_id, QUESTION, _options(*ids)) is True

    (event,) = [e for e in kb.list_events(board, task_id) if e.kind == odc.DECLARED_EVENT]
    assert event.payload == {
        "object": "hermes.owner_workspace.design_options",
        "schema_version": 1,
        "question": QUESTION,
        "options": [
            {
                "n": n,
                "sentence": SENTENCES[n - 1],
                "pictures": [{
                    "attachment_id": ids[n - 1],
                    "alt": ALTS[n - 1],
                    "media_type": pictures[n - 1][1],
                    "bytes": len(pictures[n - 1][0]),
                }],
            }
            for n in range(1, count + 1)
        ],
    }


def test_a_picture_of_exactly_four_mib_is_accepted(board):
    task_id, run_id = _running_card(board)
    ids = [_picture(board, task_id, PNG + b"\x00" * (FOUR_MIB - len(PNG))), _picture(board, task_id)]

    assert _declare(board, task_id, run_id, QUESTION, _options(*ids)) is True
    assert _observed_state(_store_file(board), task_id)["declarations"] == 1


def _foreign_picture(conn, task_id, ids):
    other, _ = _running_card(conn, title="another card")
    return QUESTION, _options(_picture(conn, other), ids[1])


def _replaced(ids, position, **fields):
    options = _options(*ids)
    options[position - 1].update(fields)
    return QUESTION, options


def _picture_replaced(ids, position, **fields):
    options = _options(*ids)
    options[position - 1]["pictures"][0].update(fields)
    return QUESTION, options


def _without_alt(ids):
    options = _options(*ids)
    del options[0]["pictures"][0]["alt"]
    return QUESTION, options


# Each case: (conn, card, the ids of two good pictures on it) -> (question, options).
REFUSALS = {
    "one option": lambda conn, tid, ids: (QUESTION, _options(ids[0])),
    "four options": lambda conn, tid, ids: (
        QUESTION, _options(*ids, _picture(conn, tid)) + [{
            "n": 4,
            "sentence": SENTENCES[0],
            "pictures": [{"attachment_id": _picture(conn, tid), "alt": ALTS[0]}],
        }],
    ),
    "no options": lambda conn, tid, ids: (QUESTION, []),
    "an empty sentence": lambda conn, tid, ids: _replaced(ids, 1, sentence=""),
    "a blank sentence": lambda conn, tid, ids: _replaced(ids, 2, sentence="   "),
    "a two-sentence sentence": lambda conn, tid, ids: _replaced(
        ids, 1, sentence="A calm layout. It has a large photo at the top.",
    ),
    "a sentence over two lines": lambda conn, tid, ids: _replaced(
        ids, 2, sentence="A bold layout\nwith a big title",
    ),
    "a sentence that fails the owner text policy": lambda conn, tid, ids: _replaced(
        ids, 1, sentence="Copy the file at /root/designs/hero.png first.",
    ),
    "a sentence over 160 characters": lambda conn, tid, ids: _replaced(
        ids, 1, sentence="A calm layout " + "with a photo " * 12 + "at the top.",
    ),
    "a two-sentence question": lambda conn, tid, ids: (
        "Which home page is best? Raphael will build it.", _options(*ids),
    ),
    "a question that fails the owner text policy": lambda conn, tid, ids: (
        "Which layout from https://example.com should Raphael build?", _options(*ids),
    ),
    "a missing alternative text": lambda conn, tid, ids: _without_alt(ids),
    "an empty alternative text": lambda conn, tid, ids: _picture_replaced(ids, 2, alt=""),
    "an alternative text that fails the owner text policy": lambda conn, tid, ids: _picture_replaced(
        ids, 1, alt="A layout like the GPT-4 one.",
    ),
    "a gap in the option numbers": lambda conn, tid, ids: _replaced(ids, 2, n=3),
    "two pictures for one option": lambda conn, tid, ids: _replaced(
        ids, 1, pictures=[
            {"attachment_id": ids[0], "alt": ALTS[0]},
            {"attachment_id": ids[1], "alt": ALTS[1]},
        ],
    ),
    "an unknown option field": lambda conn, tid, ids: _replaced(ids, 1, url="https://example.com"),
    "an id that is not a decimal string": lambda conn, tid, ids: _picture_replaced(
        ids, 1, attachment_id="option.png",
    ),
    "an id of no attachment": lambda conn, tid, ids: _picture_replaced(
        ids, 1, attachment_id="999999",
    ),
    "an attachment from another card": _foreign_picture,
    "an SVG picture": lambda conn, tid, ids: _picture_replaced(
        ids, 1, attachment_id=_picture(conn, tid, SVG, "image/svg+xml", "option.svg"),
    ),
    "a GIF picture": lambda conn, tid, ids: _picture_replaced(
        ids, 2, attachment_id=_picture(conn, tid, GIF, "image/gif", "option.gif"),
    ),
    # No type given and none to guess from the name: PNG bytes, no stored type.
    "a picture without a stored type": lambda conn, tid, ids: _picture_replaced(
        ids, 1, attachment_id=_picture(conn, tid, PNG, None, "option"),
    ),
    "a file over 4 MiB": lambda conn, tid, ids: _picture_replaced(
        ids, 1, attachment_id=_picture(conn, tid, PNG + b"\x00" * (FOUR_MIB + 1 - len(PNG))),
    ),
    "JPEG bytes stored as PNG": lambda conn, tid, ids: _picture_replaced(
        ids, 1, attachment_id=_picture(conn, tid, JPEG, "image/png", "option.png"),
    ),
    "GIF bytes stored as WebP": lambda conn, tid, ids: _picture_replaced(
        ids, 2, attachment_id=_picture(conn, tid, GIF, "image/webp", "option.webp"),
    ),
    "text bytes stored as JPEG": lambda conn, tid, ids: _picture_replaced(
        ids, 1, attachment_id=_picture(conn, tid, b"not a picture\n", "image/jpeg", "option.jpg"),
    ),
}


@pytest.mark.parametrize("case", sorted(REFUSALS))
def test_a_declaration_outside_the_limits_is_refused_and_writes_nothing(board, case):
    task_id, run_id = _running_card(board)
    ids = [_picture(board, task_id, PNG, "image/png", "calm.png"),
           _picture(board, task_id, JPEG, "image/jpeg", "bold.jpg")]
    question, options = REFUSALS[case](board, task_id, ids)
    store = _store_file(board)
    before = _observed_state(store, task_id)

    with pytest.raises(ValueError):
        _declare(board, task_id, run_id, question, options)

    assert _observed_state(store, task_id) == before
    assert before["status"] == "running" and before["declarations"] == 0
    assert board.in_transaction is False


# ---------------------------------------------------------------------------
# Platform test 2: one event and the hold, in one transaction; replays.
# ---------------------------------------------------------------------------


def test_a_valid_declaration_writes_one_event_and_holds_the_card_in_one_transaction(
    board, monkeypatch, fired_hooks,
):
    task_id, run_id = _running_card(board)
    ids = [_picture(board, task_id), _picture(board, task_id, JPEG, "image/jpeg", "bold.jpg")]
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
                (held_task_id, odc.DECLARED_EVENT),
            ).fetchone()[0],
            conn.execute("SELECT status FROM tasks WHERE id = ?", (held_task_id,)).fetchone()[0],
        )
        seen["observer"] = _observed_state(store, held_task_id)
        return blocked

    monkeypatch.setattr(kb, "_block_task_within_txn", observed_hold)

    assert _declare(board, task_id, run_id, QUESTION, _options(*ids)) is True

    # The declaration and the hold are written on one open transaction that
    # no other connection can see until it commits.
    assert seen["in_transaction"] is True
    assert seen["kwargs"]["kind"] == "needs_input"
    assert seen["kwargs"]["reason"] == odc.HOLD_SENTENCE
    assert seen["writer"] == (1, "blocked")
    assert seen["observer"] == before
    after = _observed_state(store, task_id)
    assert after == {
        **before,
        "status": "blocked",
        "block_kind": "needs_input",
        "declarations": 1,
        "events": before["events"] + 2,  # the declaration and the hold's "blocked"
        "open_runs": 0,
    }
    (blocked,) = [e for e in kb.list_events(board, task_id) if e.kind == "blocked"]
    assert blocked.payload["reason"] == odc.HOLD_SENTENCE
    assert blocked.payload["kind"] == "needs_input"
    assert _observe(
        store, "SELECT outcome FROM task_runs WHERE id = ?", run_id,
    ) == "blocked"
    # kanban_task_blocked fired once, after the commit.
    ((fields, status_seen),) = fired_hooks.fired
    assert status_seen == "blocked"
    assert fields["task_id"] == task_id
    assert fields["reason"] == odc.HOLD_SENTENCE
    assert fields["run_id"] == run_id


def test_a_card_that_is_no_longer_its_run_is_refused_and_keeps_nothing(board, fired_hooks):
    task_id, run_id = _running_card(board)
    ids = [_picture(board, task_id), _picture(board, task_id)]
    store = fired_hooks.store = _store_file(board)
    before = _observed_state(store, task_id)

    with pytest.raises(ValueError):
        _declare(board, task_id, run_id + 1, QUESTION, _options(*ids))

    assert _observed_state(store, task_id) == before
    assert fired_hooks.fired == []


def test_a_second_identical_declaration_is_a_no_op(board, fired_hooks):
    task_id, run_id = _running_card(board)
    ids = [_picture(board, task_id), _picture(board, task_id, WEBP, "image/webp", "s.webp")]
    store = fired_hooks.store = _store_file(board)
    assert _declare(board, task_id, run_id, QUESTION, _options(*ids)) is True
    first = _observed_state(store, task_id)
    payload = [e.payload for e in kb.list_events(board, task_id) if e.kind == odc.DECLARED_EVENT]

    assert _declare(board, task_id, run_id, QUESTION, _options(*ids)) is False

    assert _observed_state(store, task_id) == first
    assert [
        e.payload for e in kb.list_events(board, task_id) if e.kind == odc.DECLARED_EVENT
    ] == payload
    assert len(fired_hooks.fired) == 1


@pytest.mark.parametrize("change", ["question", "sentence", "picture", "count"])
def test_a_different_second_declaration_is_refused_and_changes_nothing(
    board, fired_hooks, change,
):
    task_id, run_id = _running_card(board)
    ids = [_picture(board, task_id), _picture(board, task_id), _picture(board, task_id)]
    store = fired_hooks.store = _store_file(board)
    assert _declare(board, task_id, run_id, QUESTION, _options(*ids[:2])) is True
    first = _observed_state(store, task_id)
    payload = [e.payload for e in kb.list_events(board, task_id) if e.kind == odc.DECLARED_EVENT]
    question, options = QUESTION, _options(*ids[:2])
    if change == "question":
        question = "Which home page should Raphael build first?"
    elif change == "sentence":
        options[1]["sentence"] = SENTENCES[2]
    elif change == "picture":
        options[0]["pictures"][0]["attachment_id"] = ids[2]
    else:
        options = _options(*ids)

    with pytest.raises(ValueError):
        _declare(board, task_id, run_id, question, options)

    assert _observed_state(store, task_id) == first
    assert [
        e.payload for e in kb.list_events(board, task_id) if e.kind == odc.DECLARED_EVENT
    ] == payload
    assert len(fired_hooks.fired) == 1
