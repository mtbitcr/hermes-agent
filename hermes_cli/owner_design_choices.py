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
"""

from __future__ import annotations

import re
import sqlite3
from typing import Any

from hermes_cli.owner_text import check_owner_text

# The object tag and version of the declaration payload (section A).
OBJECT = "hermes.owner_workspace.design_options"
SCHEMA_VERSION = 1
# The task event that records a card's design options.
DECLARED_EVENT = "design_options_declared"
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
