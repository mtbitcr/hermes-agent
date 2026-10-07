"""``hermes release``: check a release before it runs (release card 4).

``prepare`` works out PREV and NEW (S-B), builds both live adapters from the ``release`` settings
(S-E) and asks the host every merged guard, printing each result in plain words. It writes
nothing to the release state: config.yaml is read as it is on disk, the release record is opened
read-only, and the guards only read. ``status`` prints the waiting batch and the last
outcome. The release itself, under the pause (S-A) and in its own unit (S-F), belongs to a later
card.
"""

from __future__ import annotations

import contextlib
import os
import sqlite3
import stat
from pathlib import Path
from typing import Any

from hermes_cli import release_ledger
from hermes_cli.config import (
    DEFAULT_CONFIG,
    _ENV_REF_RE,
    _deep_merge,
    _expand_env_vars,
    cfg_get,
    get_project_root,
    read_raw_config,
)
from hermes_cli.release_guards import GATEWAY_UNIT, GUARDS, SERVE_UNIT, Pins, RecordedMerge
from hermes_cli.release_host import LiveHostReader
from hermes_cli.release_host_actions import ReleaseHostActions
from hermes_cli.release_runner import SANDBOX_TUNNEL_UNIT
from hermes_constants import get_default_hermes_root, get_hermes_home

# The host's unit names and readback addresses are not in the repository, so an empty one refuses
# here, before anything runs, instead of failing a readback after the cutover.
_REQUIRED = {
    f"units.{GATEWAY_UNIT}": "the gateway unit",
    f"units.{SERVE_UNIT}": "the serve unit",
    f"units.{SANDBOX_TUNNEL_UNIT}": "the sandbox tunnel unit",
    "health_url": "the health address",
    "workspace_check_url": "the owner-page check address",
}
# A required value or snapshot_dir that keeps a reference whose variable is not set refuses too:
# its placeholder would reach the adapters, and G9 would look for NEW's state snapshot under it.
_RESOLVED = {**_REQUIRED, "snapshot_dir": "the snapshot directory"}
_WAITING = {"open": "waiting for the owner's decision", "deferred": "put off by the owner"}
_OUTCOMES = {
    "released": "was released",
    "restored": "was rolled back; its changes went back to the waiting decision",
    "refused": "was refused before anything changed; its changes went back to the waiting decision",
    "failed": "failed midway and is kept for recovery",
}
# G9 fails when NEW's state snapshot exists: a release of these very changes took it after the
# stop, and was then rolled back.
_ROLLED_BACK = (
    "These changes were already tried and rolled back: a new change is needed before they can be"
    " released again."
)
_LAST_FINISHED_SQL = (
    "SELECT batch_id FROM release_events WHERE kind = 'release_finished' ORDER BY id DESC LIMIT 1"
)
# How long a read of the release record waits while a writer keeps it out, before it refuses.
_BUSY_TIMEOUT_MS = 1000


class _UnreadableRecord(Exception):
    """The release record could not be read; the message is the plain reason."""


def cmd_release(args) -> int:
    """``hermes release prepare`` or ``hermes release status``."""
    return prepare() if args.release_command == "prepare" else status()


def prepare() -> int:
    """Ask the host every guard for the accepted batch and print each result; write nothing.

    Returns 0 when every guard passed, 1 when one failed or the release was refused before them.
    """
    root, home = get_default_hermes_root(), get_hermes_home()
    if home.resolve() != root.resolve():  # S-A
        return _refuse(f"a release runs only from the root Hermes home {root}, not from {home}")
    settings, config = _settings(), root / "config.yaml"
    reasons = []
    for key, name in _RESOLVED.items():
        value = str(cfg_get(settings, *key.split(".")) or "")
        if key in _REQUIRED and not value.strip():
            reasons.append(f"{name} is not set: set release.{key} in {config}")
        elif _ENV_REF_RE.search(value):
            reasons.append(
                f"{name} refers to a variable that is not set: set the variable, or change"
                f" release.{key} in {config}"
            )
    if reasons:
        return _refuse(*reasons)

    try:
        batches, live, _finished = _record()
    except _UnreadableRecord as error:
        return _refuse(str(error))
    accepted = [batch for batch in batches if batch["state"] == "accepted"]
    if not accepted:
        return _refuse("no batch is accepted, so there is nothing to release")
    batch = accepted[0]
    # What the guards read from the record is decoded before the adapters are built or anything
    # is printed: the batch's changes, NEW, the last one's merge commit, and PREV, the NEW of the
    # last released batch once one was released (S-B).
    try:
        merges = _merges(batch)
        if any(other["state"] == "released" for other in batches):
            live = _text(live, "the last released batch has no NEW")
    except _UnreadableRecord as error:
        return _refuse(str(error))
    new = merges[-1].commit
    # The guards only read; the actions are built from the same settings, as the release will.
    reader, _actions = _adapters(root, settings, new)
    try:
        prev = live or reader.checkout_head()
    except Exception as error:
        return _refuse(
            f"the checkout's head could not be read, so PREV is unknown"
            f" ({type(error).__name__}: {error})"
        )
    pins = Pins(new=new, prev=prev)

    print(f"Batch {batch['batch_id']}: PREV {prev}, NEW {new}")
    failed = 0
    for guard, rule, check in GUARDS:
        # Every guard is asked, as in release_guards.prepare. A read that raises fails its guard
        # and keeps its cause, as a readback does in the runner.
        try:
            ok, cause = bool(check(reader, pins, merges)), ""
        except Exception as error:
            ok, cause = False, f" (could not check: {type(error).__name__}: {error})"
        print(f"{guard} {'passed' if ok else 'failed'}: {rule}{cause}")
        if guard == "G9" and not ok and not cause:
            print(f"    {_ROLLED_BACK}")
        failed += not ok
    if failed:
        return _refuse(f"{failed} of {len(GUARDS)} release guards failed")
    print(f"All {len(GUARDS)} release guards passed. Nothing in the release state was written.")
    return 0


def status() -> int:
    """Print the waiting batch and the last outcome in plain words; write nothing."""
    # What is shown is decoded from the record before anything is printed.
    try:
        batches, _live, finished = _record()
        waiting = next((batch for batch in batches if batch["state"] in _WAITING), None)
        changes = [] if waiting is None else [_change(waiting, m) for m in waiting["members"]]
        last = next((batch for batch in batches if batch["batch_id"] == finished), None)
        if finished is not None and (last is None or last["outcome"] not in _OUTCOMES):
            raise _unreadable(
                f"batch {finished}, whose release finished last, has no known outcome"
            )
    except _UnreadableRecord as error:
        return _refuse(str(error))
    if waiting is None:
        print("No batch is waiting for a decision.")
    else:
        count = len(changes)
        print(
            f"Waiting batch {waiting['batch_id']}, {_WAITING[waiting['state']]}: "
            f"{count} change{'s' * (count != 1)}"
        )
        for change in changes:
            print(f"  - {change}")
    if last is None:
        print("Last outcome: no release has finished yet.")
    else:
        print(f"Last outcome: batch {last['batch_id']} {_OUTCOMES[last['outcome']]}.")
    return 0


def _refuse(*reasons: str) -> int:
    for reason in reasons:
        print(f"Refused: {reason}.")
    print("Nothing in the release state was written.")
    return 1


def _settings() -> dict[str, Any]:
    """The ``release`` settings over their defaults, from config.yaml as it is on disk: loading
    the full configuration would create the home's directories and files. Their ``${VAR}`` and
    ``${env:VAR}`` references are resolved as load_config resolves them; one whose variable is
    not set is kept as written."""
    raw = read_raw_config().get("release")
    return _expand_env_vars(
        _deep_merge(DEFAULT_CONFIG["release"], raw if isinstance(raw, dict) else {})
    )


def _adapters(
    root: Path, settings: dict[str, Any], new: str
) -> tuple[LiveHostReader, ReleaseHostActions]:
    """The live reader and actions, from explicit keyword values only (S-E).

    An empty ``snapshot_dir`` means release-snapshots under the root home, and the reader's
    named snapshot is NEW (S-C).
    """
    snapshot_dir = Path(str(settings.get("snapshot_dir") or "").strip() or "release-snapshots")
    shared = {
        "checkout": get_project_root(),
        "root_home": root,
        "units": settings["units"],
        "snapshot_root": root / snapshot_dir.expanduser(),
    }
    return (
        LiveHostReader(**shared, snapshot_name=new),
        ReleaseHostActions(
            **shared,
            health_url=settings["health_url"],
            workspace_check_url=settings["workspace_check_url"],
        ),
    )


def _record() -> tuple[list[dict[str, Any]], str | None, int | None]:
    """Every batch, the NEW of the last released batch and the batch whose release finished last,
    all from one read transaction on the release record.

    The record has two shapes. When none of its four names exists, the database file and its
    -wal, -shm and -journal files, it does not exist yet and reads as empty. When the database
    file is a regular file, not a link, SQLite opens it read-only, with a short busy timeout, and
    reads every row in one read transaction, so only what was committed is read, in the -wal too.
    SQLite keeps the side files as it does for any read-only reader and never writes the database
    file. Any other state, an error while the names are looked at, any SQLite error (a hot
    journal, a lock held past the timeout, a file it cannot read) and a batch the record's own
    snapshot cannot decode raise _UnreadableRecord.
    """
    path = release_ledger.ledger_path()
    try:
        # Each name is looked at with lstat, as os.path.lexists does, so a broken link exists.
        # lexists takes every error for a missing name, a folder that cannot be searched too;
        # here only a missing name is, and any other error refuses.
        found = {}
        for side in ("", "-wal", "-shm", "-journal"):
            with contextlib.suppress(FileNotFoundError):
                found[side] = os.lstat(f"{path}{side}").st_mode
        if not found:
            return [], None, None
        if "" not in found:
            beside = ", ".join(f"{path.name}{side}" for side in found)
            raise _unreadable(f"there is no database file beside {beside}")
        if not stat.S_ISREG(found[""]):
            link = "a symbolic link, " if stat.S_ISLNK(found[""]) else ""
            raise _unreadable(f"it is {link}not a regular file")
        conn = sqlite3.connect(
            f"{path.absolute().as_uri()}?mode=ro",
            uri=True,
            isolation_level=None,
            timeout=_BUSY_TIMEOUT_MS / 1000,
        )
        conn.row_factory = sqlite3.Row
        try:
            with release_ledger._read_txn(conn):
                ids = conn.execute("SELECT id FROM release_batches ORDER BY id").fetchall()
                live = release_ledger.last_released(conn)
                finished = conn.execute(_LAST_FINISHED_SQL).fetchone()
                return (
                    [_batch(conn, row["id"]) for row in ids],
                    live,
                    None if finished is None else finished["batch_id"],
                )
        finally:
            conn.close()
    except (OSError, sqlite3.Error) as error:
        raise _unreadable(f"{type(error).__name__}: {error}") from error


def _batch(conn: sqlite3.Connection, batch_id: int) -> dict[str, Any]:
    """A batch as the record's own snapshot reads it. One it cannot decode, a field missing or a
    commit that is not ASCII, cannot be read."""
    try:
        return release_ledger._snapshot(conn, batch_id)
    except (LookupError, ValueError) as error:
        raise _unreadable(
            f"batch {batch_id} could not be decoded: {type(error).__name__}: {error}"
        ) from error


def _merges(batch: dict[str, Any]) -> list[RecordedMerge]:
    """The accepted batch's changes in merge order, as the guards read them. A batch without
    changes, or a change without one of its four commits, cannot be read."""
    if not batch["members"]:
        raise _unreadable(f"accepted batch {batch['batch_id']} has no changes")
    missing = f"a change in batch {batch['batch_id']} has no"
    names = ("merge_commit", "reviewed_base", "reviewed_head", "reviewed_tree")
    return [
        RecordedMerge(*(_text(member.get(name), f"{missing} {name}") for name in names))
        for member in batch["members"]
    ]


def _change(batch: dict[str, Any], member: dict[str, Any]) -> str:
    """A waiting change as status shows it: its title, or its card id when the title is empty,
    and its pull request. A change without one of those it shows cannot be read; a card id that
    is not shown is not needed."""
    missing = f"a change in batch {batch['batch_id']} has no"
    title = _text(member.get("title"), f"{missing} title", empty=True)
    name = title or _text(member.get("card_id"), f"{missing} card_id")
    pr_url = _text(member.get("pr_url"), f"{missing} pr_url")
    return f"{name} ({pr_url})"


def _text(value: Any, reason: str, *, empty: bool = False) -> str:
    """value when it is text, and not empty unless empty is allowed; otherwise the record cannot
    be read, for reason."""
    if not isinstance(value, str) or not (value or empty):
        raise _unreadable(reason)
    return value


def _unreadable(reason: str) -> _UnreadableRecord:
    """The plain refusal of a release record that cannot be read, with its reason."""
    return _UnreadableRecord(
        f"the release record {release_ledger.ledger_path()} could not be read ({reason})"
    )
