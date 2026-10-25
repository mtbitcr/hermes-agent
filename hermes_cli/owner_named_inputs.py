"""The named input rules: one copy, shared by the authority, the apply path,
the missing-input hold, the Decisions reader and the upload route.

An approved plan may name, on each task it creates, the text files that task
needs from the owner: the optional ``inputs`` field, a list of one to five
``{"name", "type"}`` entries. Absent means no inputs, so a proposal stored
before the field existed replays unchanged. The file the owner gives for an
input is stored under a name the platform derives from it, never one a model
chooses (named inputs plan, section 2 and card C1).

The apply path records a card's inputs once, as the task event
``named_inputs_declared`` (no schema migration), and :func:`missing_inputs`
reads that declaration back against the card's attachments. ``claim_task``
holds a card that still misses one (:class:`InputHold`), and the Decisions
feed names what is missing (:func:`decision_missing_inputs`, card C2). The
owner upload route stores each file the owner gives and releases the card
once none is missing (:func:`accept_upload`, card C3).
"""

from __future__ import annotations

import json
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
# The plain words the owner app shows for each input type (card C2). Each
# passes the owner text policy unchanged and is never a file name.
TYPE_LABELS = {
    "txt": "text file",
    "md": "formatted text (Markdown)",
    "csv": "spreadsheet (CSV)",
    "json": "data file (JSON)",
}
# The largest file the owner may give for one input: 8 MiB.
MAX_INPUT_BYTES = 8 * 1024 * 1024
# The Content-Type an upload for each input type may carry (card C3); the
# first is the declared one, which the stored file gets.
_MEDIA_TYPES = {
    "txt": ("text/plain",),
    "md": ("text/markdown", "text/plain"),
    "csv": ("text/csv", "text/plain"),
    "json": ("application/json",),
}

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


class InputHold:
    """The missing-input hold of ``claim_task`` (card C2).

    ``claim_task`` opens one around its write transaction and calls
    :meth:`hold` after its parents check. A ready work card whose declared
    inputs the owner has not all given is blocked ``needs_input`` with
    :data:`HOLD_SENTENCE` in that same transaction, so no run is made and the
    claim returns None. ``kanban_task_blocked`` fires once the transaction has
    committed, as ``block_task`` fires it, and never when it rolled back.
    """

    def __init__(self) -> None:
        self._held: tuple[str, dict] | None = None

    def __enter__(self) -> "InputHold":
        return self

    def hold(self, conn: sqlite3.Connection, task_id: str) -> bool:
        """Whether the card is held; *conn* is in the claim's transaction."""
        from hermes_cli import kanban_db

        declared = conn.execute(
            "SELECT 1 FROM tasks WHERE id = ? AND status = 'ready' "
            "AND task_kind = 'work' AND EXISTS (SELECT 1 FROM task_events "
            "WHERE task_id = tasks.id AND kind = ?)",
            (task_id, DECLARED_EVENT),
        ).fetchone()
        if declared is None or not missing_inputs(conn, task_id):
            return False
        blocked = kanban_db._block_task_within_txn(
            conn, task_id, reason=HOLD_SENTENCE, kind="needs_input",
            expected_run_id=None, synthesize_run=False,
        )
        if blocked is not None:
            self._held = (task_id, blocked)
        return True

    def __exit__(self, exc_type, exc, traceback) -> None:
        if exc_type is not None or self._held is None:
            return
        from hermes_cli import kanban_db

        task_id, blocked = self._held
        kanban_db._fire_kanban_lifecycle_hook(
            "kanban_task_blocked",
            task_id,
            board=kanban_db.get_current_board(),
            assignee=blocked["assignee"],
            run_id=blocked["run_id"],
            reason=HOLD_SENTENCE,
        )


def decision_missing_inputs(
    conn: sqlite3.Connection, task_ids: list[str],
) -> dict[str, list[dict]]:
    """The still-missing inputs of each declaring card among *task_ids*.

    For a Decisions reader that names ``owner_input_upload_v1`` (card C2).
    Each entry is ``{"n", "name", "type_label", "max_bytes"}``, in declaration
    order: the input's plain :data:`TYPE_LABELS` word, never its file name,
    and :data:`MAX_INPUT_BYTES`. A card without a declaration is left out.
    Read only; *task_ids* is one Project's bounded Decisions window.
    """
    if not task_ids:
        return {}
    declaring = conn.execute(
        "SELECT DISTINCT task_id FROM task_events WHERE kind = ? AND task_id IN "
        f"({', '.join('?' for _ in task_ids)})",
        (DECLARED_EVENT, *task_ids),
    ).fetchall()
    return {
        row[0]: [
            {
                "n": entry["n"],
                "name": entry["name"],
                "type_label": TYPE_LABELS[entry["type"]],
                "max_bytes": MAX_INPUT_BYTES,
            }
            for entry in missing_inputs(conn, row[0])
        ]
        for row in declaring
    }


def accept_upload(
    conn: sqlite3.Connection,
    task_id: str,
    n: int,
    data: bytes,
    content_type: str,
    *,
    board: str,
) -> str:
    """Store the file the owner gave for input *n* of a held card (card C3).

    For the owner upload route, which resolved *task_id* through the owner's
    own Decisions feed. Checks, in the plan's order and before any byte is
    stored: the card is still held ``needs_input`` and *n* is one of its
    still-missing inputs; *data* is not empty and at most
    :data:`MAX_INPUT_BYTES`; *content_type* is one the input's type takes; the
    file is strict UTF-8 without a NUL byte, and parses for type json. Then
    ``store_owner_input`` stores the file, uploaded by :data:`OWNER_UPLOADER`
    under its derived name and with the declared type, and once no input is
    missing releases the card as ``unblock_task`` does, to todo while a parent
    is not done, all in one write transaction that leaves no file behind when
    it fails.

    Returns "released" or "stored", or a refusal: "not_waiting", "empty",
    "too_big", "wrong_type", "type_mismatch" or "not_text", and then nothing
    was written.
    """
    from hermes_cli import kanban_db

    held = conn.execute(
        "SELECT 1 FROM tasks WHERE id = ? AND status = 'blocked' "
        "AND block_kind = 'needs_input' AND task_kind = 'work'",
        (task_id,),
    ).fetchone()
    entry = next(
        (entry for entry in missing_inputs(conn, task_id) if entry["n"] == n), None,
    ) if held else None
    if entry is None:
        return "not_waiting"
    if not data:
        return "empty"
    if len(data) > MAX_INPUT_BYTES:
        return "too_big"
    media_type = content_type.split(";")[0].strip().lower()
    if not any(media_type in types for types in _MEDIA_TYPES.values()):
        return "wrong_type"
    if media_type not in _MEDIA_TYPES[entry["type"]]:
        return "type_mismatch"
    try:
        text = data.decode("utf-8")
        if entry["type"] == "json":
            json.loads(text)
    except (ValueError, RecursionError):  # bad UTF-8 or JSON, or JSON nested too deep
        return "not_text"
    if "\x00" in text:
        return "not_text"
    released = kanban_db.store_owner_input(
        conn, task_id, entry["file"], data,
        content_type=_MEDIA_TYPES[entry["type"]][0], board=board,
    )
    if released is None:  # the card stopped waiting for this file meanwhile
        return "not_waiting"
    return "released" if released else "stored"
