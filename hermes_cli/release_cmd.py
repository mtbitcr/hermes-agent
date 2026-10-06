"""``hermes release``: check a release before it runs (release card 4).

``prepare`` works out PREV and NEW (S-B), builds both live adapters from the ``release`` settings
(S-E) and asks the host every merged guard, printing each result in plain words. It writes
nothing to the release state: config.yaml is read as it is on disk, the release record is read
from a private copy, and the guards only read. ``status`` prints the waiting batch and the last
outcome. The release itself, under the pause (S-A) and in its own unit (S-F), belongs to a later
card.
"""

from __future__ import annotations

import contextlib
import shutil
import sqlite3
import tempfile
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
# release_ledger.last_released's query. The ledger's readers refuse a connection to anything but
# the live store itself, so the copy is read with these queries and release_ledger._snapshot.
_LAST_RELEASED_SQL = (
    "SELECT new FROM release_batches JOIN release_events ON batch_id = release_batches.id "
    "WHERE state = 'released' AND kind = 'release_finished' ORDER BY release_events.id DESC LIMIT 1"
)


class _UnreadableRecord(Exception):
    """The copy of the release record could not be read; the message is the plain reason."""


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
    new = batch["members"][-1]["merge_commit"]
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
    merges = [
        RecordedMerge(
            m["merge_commit"], m["reviewed_base"], m["reviewed_head"], m["reviewed_tree"]
        )
        for m in batch["members"]
    ]

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
    try:
        batches, _live, finished = _record()
    except _UnreadableRecord as error:
        return _refuse(str(error))
    waiting = next((batch for batch in batches if batch["state"] in _WAITING), None)
    if waiting is None:
        print("No batch is waiting for a decision.")
    else:
        count = len(waiting["members"])
        print(
            f"Waiting batch {waiting['batch_id']}, {_WAITING[waiting['state']]}: "
            f"{count} change{'s' * (count != 1)}"
        )
        for member in waiting["members"]:
            print(f"  - {member['title'] or member['card_id']} ({member['pr_url']})")
    last = next((batch for batch in batches if batch["batch_id"] == finished), None)
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
    all from a private copy of the release record.

    The database file, and its -wal when one exists, are copied into a private temporary
    directory; the copy is opened there read-only, read and removed. SQLite never opens the live
    files, so it cannot create, change or delete their side files, and what was committed to the
    -wal is still read. A record that does not exist yet reads as empty; a copy SQLite cannot
    read raises _UnreadableRecord.
    """
    path = release_ledger.ledger_path()
    if not path.is_file():
        return [], None, None
    try:
        with tempfile.TemporaryDirectory(prefix="hermes-release-record-") as private:
            copy = Path(private) / path.name
            shutil.copyfile(path, copy)
            with contextlib.suppress(FileNotFoundError):
                shutil.copyfile(f"{path}-wal", f"{copy}-wal")
            conn = sqlite3.connect(f"{copy.as_uri()}?mode=ro", uri=True, isolation_level=None)
            conn.row_factory = sqlite3.Row
            try:
                ids = conn.execute("SELECT id FROM release_batches ORDER BY id").fetchall()
                live = conn.execute(_LAST_RELEASED_SQL).fetchone()
                finished = conn.execute(_LAST_FINISHED_SQL).fetchone()
                return (
                    [release_ledger._snapshot(conn, row["id"]) for row in ids],
                    None if live is None else live["new"],
                    None if finished is None else finished["batch_id"],
                )
            finally:
                conn.close()
    except (OSError, sqlite3.Error) as error:
        raise _UnreadableRecord(
            f"the release record {path} could not be read ({type(error).__name__}: {error})"
        ) from error
