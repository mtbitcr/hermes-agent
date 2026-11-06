"""The design options rules: one copy, used by the designer's worker tool.

A designer card may offer the owner two or three pictured options and wait
for the owner to choose one (design options plan, section A and card 1).
The designer first stores each picture as an ordinary attachment on its own
card, then declares one question and its options with
:func:`declare_design_options`. Each option is one sentence and exactly one
picture: the id of that attachment as a decimal string, and its alternative
text.

The declaration is recorded once, as the task event
``design_options_declared`` (no schema migration), in the same write
transaction that holds the card ``needs_input`` with :data:`HOLD_SENTENCE`.
Its payload is the section A object; the platform adds each picture's media
type and size from the attachment it checked. Nothing here is shown to the
owner.

Card 2 lists and records the owner's choice. :func:`decision_design_options`
gives a waiting card's declaration to the Decisions feed;
:func:`accept_choice` records the owner's choice once, as the task event
``design_option_chosen``, comments on the card and releases it; and
:func:`recent_design_choices` reads the record of past choices back from both
events.
"""

from __future__ import annotations

import json
import re
import sqlite3
import time
from typing import Any

from hermes_cli.owner_text import check_owner_text

# The object tag and version of the declaration payload (section A).
OBJECT = "hermes.owner_workspace.design_options"
SCHEMA_VERSION = 1
# The task event that records a card's design options.
DECLARED_EVENT = "design_options_declared"
# The task event that records the owner's choice (card 2).
CHOSEN_EVENT = "design_option_chosen"
# How many decided design choices the Decisions list returns (section D).
RECENT_LIMIT = 10
# The one owner reason of the hold. It passes the owner text policy
# unchanged, so the shared policy copy never has to change.
HOLD_SENTENCE = "Raphael needs you to choose one design."
MIN_OPTIONS = 2
MAX_OPTIONS = 3
# The longest question, option sentence or alternative text, in characters.
MAX_TEXT_LENGTH = 160
# The largest picture an option may show: 4 MiB, below the 8 MiB ceiling of
# a file the owner gives (``owner_named_inputs.MAX_INPUT_BYTES``).
MAX_PICTURE_BYTES = 4 * 1024 * 1024
# The picture types an option may show. GIF can animate and SVG is never
# shown inline, so both are refused.
MEDIA_TYPES = ("image/png", "image/jpeg", "image/webp")

# An attachment id as SQLite stores one: a positive decimal number.
_ATTACHMENT_ID = re.compile(r"[1-9][0-9]{0,17}")
# A sentence end followed by more words: the text holds a second sentence.
_SENTENCE_END = re.compile(r"[.!?][\"')\]’”]*\s+\S")


def _plain_text(value: Any, what: str, *, sentence: bool) -> str:
    """*value* when it is 1 to 160 characters of plain owner text.

    With *sentence*, it must also be exactly one sentence on one line.
    """
    if (
        not isinstance(value, str)
        or not value.strip()
        or len(value) > MAX_TEXT_LENGTH
        or (sentence and (value.splitlines() != [value] or _SENTENCE_END.search(value)))
        or check_owner_text(value, "planner_prose") != value
    ):
        shape = "exactly one sentence of " if sentence else ""
        raise ValueError(
            f"{what} must be {shape}1 to {MAX_TEXT_LENGTH} characters in plain "
            "words, with no path, web address, id, secret or model name"
        )
    return value


def _first_bytes_match(media_type: str, data: bytes) -> bool:
    if media_type == "image/png":
        return data.startswith(b"\x89PNG\r\n\x1a\n")
    if media_type == "image/jpeg":
        return data.startswith(b"\xff\xd8\xff")
    return data[:4] == b"RIFF" and data[8:12] == b"WEBP"


def _picture(
    conn: sqlite3.Connection, task_id: str, n: int, picture: Any, *, board: str,
) -> dict:
    """The stored entry of option *n*'s picture, read and checked on *conn*.

    The attachment must be on *task_id* itself, stored as PNG, JPEG or WebP,
    1 byte to 4 MiB, and its first bytes must be those of its stored type.
    """
    from hermes_cli import kanban_db

    if not isinstance(picture, dict) or set(picture) != {"attachment_id", "alt"}:
        raise ValueError(f"the picture of option {n} must be exactly an attachment_id and an alt")
    attachment_id = picture["attachment_id"]
    if not isinstance(attachment_id, str) or not _ATTACHMENT_ID.fullmatch(attachment_id):
        raise ValueError(
            f"the attachment_id of option {n} must be an attachment id as a decimal string"
        )
    alt = _plain_text(picture["alt"], f"the alternative text of option {n}", sentence=False)
    attachment = kanban_db.get_attachment(conn, int(attachment_id))
    if attachment is None or attachment.task_id != task_id:
        raise ValueError(f"the picture of option {n} must be an attachment on this card")
    media_type = (attachment.content_type or "").split(";")[0].strip().lower()
    if media_type not in MEDIA_TYPES:
        raise ValueError(f"the picture of option {n} must be stored as PNG, JPEG or WebP")
    if not 1 <= attachment.size <= MAX_PICTURE_BYTES:
        raise ValueError(f"the picture of option {n} must be 1 byte to 4 MiB")
    try:
        data = kanban_db.read_attachment_bytes(attachment, board=board)
    except (OSError, ValueError) as exc:
        raise ValueError(f"the picture of option {n} cannot be read: {exc}") from None
    if not _first_bytes_match(media_type, data):
        raise ValueError(f"the picture of option {n} is not the {media_type} its stored type says")
    return {
        "attachment_id": attachment_id,
        "alt": alt,
        "media_type": media_type,
        "bytes": attachment.size,
    }


def _declaration(
    conn: sqlite3.Connection, task_id: str, question: Any, options: Any, *, board: str,
) -> dict:
    """The section A payload of *question* and *options*, or ``ValueError``."""
    question = _plain_text(question, "the question", sentence=True)
    if not isinstance(options, list) or not MIN_OPTIONS <= len(options) <= MAX_OPTIONS:
        raise ValueError(f"options must be a list of {MIN_OPTIONS} or {MAX_OPTIONS} entries")
    declared: list[dict] = []
    for n, option in enumerate(options, start=1):
        if not isinstance(option, dict) or set(option) != {"n", "sentence", "pictures"}:
            raise ValueError("each option must be exactly an n, a sentence and pictures")
        if type(option["n"]) is not int or option["n"] != n:
            raise ValueError("options must be numbered 1, 2 and 3 in order, with no gaps")
        sentence = _plain_text(option["sentence"], f"the sentence of option {n}", sentence=True)
        pictures = option["pictures"]
        if not isinstance(pictures, list) or len(pictures) != 1:
            raise ValueError(f"option {n} must have exactly one picture")
        declared.append({
            "n": n,
            "sentence": sentence,
            "pictures": [_picture(conn, task_id, n, pictures[0], board=board)],
        })
    return {
        "object": OBJECT,
        "schema_version": SCHEMA_VERSION,
        "question": question,
        "options": declared,
    }


def _declarations(events: list) -> list:
    return [event.payload for event in events if event.kind == DECLARED_EVENT]


def declare_design_options(
    conn: sqlite3.Connection,
    task_id: str,
    question: Any,
    options: Any,
    *,
    board: str,
    expected_run_id: int | None = None,
) -> bool:
    """Record the design options of *task_id* and hold the card for the owner.

    For the designer's worker tool, on the worker's own current card of
    *board*. In one write transaction on *conn*: every limit of section A is
    checked, each picture read back from the card's own attachments; one
    ``design_options_declared`` event is written; and the card is held
    ``needs_input`` with :data:`HOLD_SENTENCE` through
    ``_block_task_within_txn``, so with *expected_run_id* it must still be
    that run. ``kanban_task_blocked`` fires once the transaction committed,
    as ``block_task`` fires it.

    Returns True when declared. Idempotent: a card already holding exactly
    this declaration is left alone and False is returned. A card holding
    another declaration, a card that cannot be held, or any limit not met
    raises ``ValueError``, and then nothing was written.
    """
    from hermes_cli import kanban_db

    with kanban_db.write_txn(conn):
        payload = _declaration(conn, task_id, question, options, board=board)
        existing = _declarations(kanban_db.list_events(conn, task_id))
        if existing == [payload]:
            return False
        if existing:
            raise ValueError(f"task {task_id!r} already offers other design options")
        kanban_db._append_event(conn, task_id, DECLARED_EVENT, payload)
        blocked = kanban_db._block_task_within_txn(
            conn, task_id, reason=HOLD_SENTENCE, kind="needs_input",
            expected_run_id=expected_run_id,
        )
        if blocked is None:
            raise ValueError(
                f"task {task_id!r} cannot wait for a choice: it is not running "
                "or ready, or this is no longer its current run"
            )
    kanban_db._fire_kanban_lifecycle_hook(
        "kanban_task_blocked",
        task_id,
        board=board,
        assignee=blocked["assignee"],
        run_id=blocked["run_id"],
        reason=HOLD_SENTENCE,
    )
    return True


def _waiting_declarations(conn: sqlite3.Connection, task_ids: list[str]) -> dict[str, dict]:
    """The declaration of each of *task_ids* that still waits for the owner's choice.

    The one rule the feed and the choice both use. A declaration waits exactly
    while all three are true: the card is blocked now with block kind
    ``needs_input``, by the kernel's own reading of the status changes it
    records (``task_owner_waits``); the card's newest ``blocked`` event is the
    one the declaration wrote in its own transaction, its first after the
    declaration, so since then no status change and no later block; and no
    choice is recorded. Every event that leaves the card's status as it is
    leaves the hold as it is too. Batched: a few queries for all of
    *task_ids* together, never one per card.
    """
    from hermes_cli import kanban_db

    held = [
        task_id for task_id, wait in kanban_db.task_owner_waits(conn, task_ids).items()
        if wait["kind"] == "needs_input"
    ]
    if not held:
        return {}
    declared: dict[str, dict] = {}
    hold: dict[str, int] = {}
    newest: dict[str, int] = {}
    chosen: set[str] = set()
    for task_id, event_id, kind, payload in conn.execute(
        "SELECT task_id, id, kind, payload FROM task_events WHERE kind IN (?, 'blocked', ?) "
        "AND task_id IN (" + ", ".join("?" for _ in held) + ") ORDER BY id",
        (DECLARED_EVENT, CHOSEN_EVENT, *held),
    ).fetchall():
        task_id = str(task_id)
        if kind == DECLARED_EVENT:
            declared.setdefault(task_id, json.loads(payload))
        elif task_id not in declared:
            continue
        elif kind == CHOSEN_EVENT:
            chosen.add(task_id)
        else:
            hold.setdefault(task_id, int(event_id))
            newest[task_id] = int(event_id)
    return {
        task_id: payload for task_id, payload in declared.items()
        if task_id in hold and hold[task_id] == newest[task_id] and task_id not in chosen
    }


def decision_design_options(conn: sqlite3.Connection, task_ids: list[str]) -> dict[str, dict]:
    """The design options each of *task_ids* waits for the owner to choose from.

    For the Decisions feed read with ``design_options_v1`` (card 2), on a
    read-only connection to the cards' own store: the declaration payload of
    each card whose declaration still waits, the section A object as
    declared, so exactly the cards :func:`accept_choice` takes a choice for.
    Options are listed only while the ``needs_input`` hold their declaration
    made lasts: a change of the card's status or a later block ends it for
    good, and an event that leaves the status as it is, such as a comment, an
    attachment added or removed, an assignment, a link or an unlink, does
    not. A card without one is left out and stays today's item. Read from the
    events only: a picture that can no longer be read is the app's to show as
    such, and the owner can still choose by the option's sentence (section B).
    """
    return _waiting_declarations(conn, task_ids)


def accept_choice(conn: sqlite3.Connection, task_id: str, n: int, *, actor: str) -> str:
    """Record the owner's choice of option *n* on a waiting designer card.

    For the owner choose route (card 2), which resolved *task_id* through the
    owner's own Decisions feed. In one write transaction on *conn*: the
    card's declaration must still wait, by the one rule
    :func:`decision_design_options` lists by, so a card held again after the
    hold its declaration made ended never takes a choice; the declaration must
    offer option *n*; one ``design_option_chosen`` event
    ``{n, sentence, attachment_id, actor, chosen_at}`` is written; *actor*
    comments on the card in plain words; and the card is released as
    ``unblock_task`` releases it, to todo while a parent is not done.

    Returns "released", or "recorded" when the card is parked for
    re-approval instead; or "not_waiting", and then nothing was written. The
    first choice is never rewritten.
    """
    from hermes_cli import kanban_db
    from hermes_cli.owner_workspace import _owner_timestamp

    now = int(time.time())
    with kanban_db.write_txn(conn):
        declared = _waiting_declarations(conn, [task_id]).get(task_id)
        option = next(
            (option for option in declared["options"] if option["n"] == n), None,
        ) if declared else None
        if option is None:
            return "not_waiting"
        kanban_db._append_event(conn, task_id, CHOSEN_EVENT, {
            "n": n,
            "sentence": option["sentence"],
            "attachment_id": option["pictures"][0]["attachment_id"],
            "actor": actor,
            "chosen_at": _owner_timestamp(now),
        })
        kanban_db.add_comment(
            conn, task_id, actor, f"The owner chose option {n}: {option['sentence']}",
        )
        released = kanban_db._unblock_task_within_txn(conn, task_id, now)
    return "released" if released else "recorded"


def recent_design_choices(
    conn: sqlite3.Connection, project: dict, recent: list[dict],
) -> list[dict]:
    """*recent* with the choices made on *project*'s cards, newest first.

    For the Decisions list read with ``design_options_v1`` (section D), on a
    read-only connection to the Project's own store. Each entry is read from
    a card's ``design_options_declared`` and ``design_option_chosen`` events:
    the Project's name and slug, the question, the chosen ``n`` and
    ``chosen_at``, and every option with its sentence and its picture. At
    most :data:`RECENT_LIMIT` are returned.
    """
    from hermes_cli.owner_workspace import owner_project_name

    entries = list(recent)
    for task_id, payload in conn.execute(
        "SELECT e.task_id, e.payload FROM task_events e JOIN tasks t ON t.id = e.task_id "
        "WHERE t.project_id = ? AND e.kind = ? ORDER BY e.id DESC LIMIT ?",
        (str(project["project_id"]), CHOSEN_EVENT, RECENT_LIMIT),
    ).fetchall():
        declared = conn.execute(
            "SELECT payload FROM task_events WHERE task_id = ? AND kind = ? ORDER BY id LIMIT 1",
            (task_id, DECLARED_EVENT),
        ).fetchone()
        if declared is None:
            continue
        chosen, offered = json.loads(payload), json.loads(declared[0])
        entries.append({
            "project_slug": str(project["slug"]),
            "project_name": owner_project_name(project["name"]),
            "question": offered["question"],
            "n": chosen["n"],
            "chosen_at": chosen["chosen_at"],
            "options": [
                {
                    "n": option["n"],
                    "sentence": option["sentence"],
                    "picture": {
                        "attachment_id": option["pictures"][0]["attachment_id"],
                        "alt": option["pictures"][0]["alt"],
                    },
                }
                for option in offered["options"]
            ],
        })
    entries.sort(key=lambda entry: entry["chosen_at"], reverse=True)
    return entries[:RECENT_LIMIT]
