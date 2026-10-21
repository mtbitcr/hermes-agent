"""The named input rules: one copy, shared by the authority, the apply path
and, later, the missing-input hold and the upload route.

An approved plan may name, on each task it creates, the text files that task
needs from the owner: the optional ``inputs`` field, a list of one to five
``{"name", "type"}`` entries. Absent means no inputs, so a proposal stored
before the field existed replays unchanged. The file the owner gives for an
input is stored under a name the platform derives from it, never one a model
chooses (named inputs plan, section 2 and card C1).

The apply path records a card's inputs once, as the task event
``named_inputs_declared`` (no schema migration), and :func:`missing_inputs`
reads that declaration back against the card's attachments.
"""

from __future__ import annotations

import re
import sqlite3
from typing import Any

from hermes_cli.owner_text import check_owner_text

INPUT_TYPES = ("txt", "md", "csv", "json")
MAX_INPUTS = 5
MAX_NAME_LENGTH = 40
# The task event that records a card's named inputs.
DECLARED_EVENT = "named_inputs_declared"
# Who uploaded an attachment the owner gave; only such a file fills an input.
OWNER_UPLOADER = "owner"
# The one owner reason of the missing-input hold (card C2). It passes the
# owner text policy unchanged, so the shared policy copy never has to change.
HOLD_SENTENCE = "Raphael needs a file from you before this work can start."

# Words of lowercase ASCII letters, digits and hyphens joined by single
# spaces, starting with a letter: no dot, slash or extension can appear.
_NAME = re.compile(r"[a-z][a-z0-9-]*(?: [a-z0-9-]+)*")


def parse_named_inputs(value: Any) -> list[dict]:
    """Return *value* as a task's named inputs, or raise ``ValueError``.

    *value* is the ``inputs`` field as stated; its absence is the caller's
    "no inputs". Every entry is exactly a ``name`` and a ``type``. A name must
    also read as plain words: one that ``check_owner_text`` would rewrite as a
    title (an id, a hex word, a model name) is refused. Two inputs on one task
    may not have the same name, whatever their types, nor derive the same
    file: "price list" and "price-list" are refused as two csv inputs and
    accepted as a txt and a json one.
    """
    if not isinstance(value, list) or not 1 <= len(value) <= MAX_INPUTS:
        raise ValueError(f"inputs must be a list of 1 to {MAX_INPUTS} entries")
    parsed: list[dict] = []
    names: set[str] = set()
    files: set[str] = set()
    for entry in value:
        if not isinstance(entry, dict) or set(entry) != {"name", "type"}:
            raise ValueError("each input must be exactly a name and a type")
        name, input_type = entry["name"], entry["type"]
        if (
            not isinstance(name, str)
            or len(name) > MAX_NAME_LENGTH
            or not _NAME.fullmatch(name)
            or check_owner_text(name, "title") != name
        ):
            raise ValueError(
                f"an input name must be 1 to {MAX_NAME_LENGTH} lowercase letters, "
                "digits, single spaces and hyphens, start with a letter and read "
                "as plain words"
            )
        if not isinstance(input_type, str) or input_type not in INPUT_TYPES:
            raise ValueError(f"an input type must be one of {', '.join(INPUT_TYPES)}")
        if name in names:
            raise ValueError("two inputs on one task may not have the same name")
        file_name = derived_file_name(name, input_type)
        if file_name in files:
            raise ValueError("two inputs on one task may not have the same file name")
        names.add(name)
        files.add(file_name)
        parsed.append({"name": name, "type": input_type})
    return parsed


def derived_file_name(name: str, input_type: str) -> str:
    """The file name an input is stored under, for example "price-list.csv".

    The name with its spaces turned into hyphens, a dot and the type. It
    passes the attachment store's safe-name rule unchanged.
    """
    return f"{name.replace(' ', '-')}.{input_type}"


def _declaration(inputs: list[dict]) -> dict:
    return {
        "inputs": [
            {
                "n": n,
                "name": entry["name"],
                "type": entry["type"],
                "file": derived_file_name(entry["name"], entry["type"]),
            }
            for n, entry in enumerate(inputs, start=1)
        ],
    }


def _declarations(events: list) -> list:
    return [event.payload for event in events if event.kind == DECLARED_EVENT]


def declare_named_inputs(
    conn: sqlite3.Connection, specs: list[dict], task_ids: list[str],
) -> None:
    """Record the named inputs of the cards one approved plan just created.

    *specs* are the approved task specifications in creation order and
    *task_ids* the cards made from them. A specification without ``inputs``
    declares nothing. Each declaring card gets exactly one
    ``named_inputs_declared`` event, written on *conn* inside the still open
    transaction that creates the parked card, so card and declaration commit
    or roll back together; this opens no transaction of its own. Idempotent:
    a card already holding exactly this declaration is left alone. A card
    holding another one, a card that is no longer parked, or a card whose
    title is not its specification's raises ``ValueError``, so no declaration
    is ever doubled, misplaced or silently dropped.
    """
    from hermes_cli import kanban_db

    declared = [
        (task_id, spec) for spec, task_id in zip(specs, task_ids) if "inputs" in spec
    ]
    for task_id, spec in declared:
        payload = _declaration(parse_named_inputs(spec["inputs"]))
        existing = _declarations(kanban_db.list_events(conn, task_id))
        if existing == [payload]:
            continue
        row = conn.execute(
            "SELECT title, status FROM tasks WHERE id = ? AND task_kind = 'work'",
            (task_id,),
        ).fetchone()
        if (
            existing
            or row is None
            or row["title"] != spec["title"]
            or row["status"] != kanban_db.PARKED_STATUS
        ):
            raise ValueError(
                f"the approved named inputs cannot be declared on task {task_id!r}"
            )
        kanban_db._append_event(conn, task_id, DECLARED_EVENT, payload)


def missing_inputs(conn: sqlite3.Connection, task_id: str) -> list[dict]:
    """The declared inputs of *task_id* that the owner has not attached yet.

    Each is the declared ``{"n", "name", "type", "file"}`` entry, in order; a
    card without a declaration misses nothing. Input n is present when an
    attachment the owner uploaded (:data:`OWNER_UPLOADER`) exists on the card
    under the input's derived file name. A file from anyone else, an agent
    included, never counts. The name the uploader asked for is read from the
    attachment's ``attached`` event, because the store keeps a file that
    would overwrite another under a new name ("price-list (1).csv").
    """
    from hermes_cli import kanban_db

    events = kanban_db.list_events(conn, task_id)
    declarations = _declarations(events)
    if not declarations:
        return []
    requested = {
        event.payload.get("attachment_id"): event.payload.get("requested_filename")
        for event in events
        if event.kind == "attached" and isinstance(event.payload, dict)
    }
    present = {
        requested.get(attachment.id) or attachment.filename
        for attachment in kanban_db.list_attachments(conn, task_id)
        if attachment.uploaded_by == OWNER_UPLOADER
    }
    return [
        dict(entry)
        for entry in declarations[-1]["inputs"]
        if entry["file"] not in present
    ]
