"""``hermes release``: check a release before it runs (release card 4).

``prepare`` works out PREV and NEW (S-B), builds both live adapters from the ``release`` settings
(S-E) and asks the host every merged guard, printing each result in plain words. It writes
nothing: config.yaml is read as it is on disk, the release record is opened read-only, and the
guards only read. ``status`` prints the waiting batch and the last outcome. The release itself,
under the pause (S-A) and in its own unit (S-F), belongs to a later card.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

from hermes_cli import release_ledger
from hermes_cli.config import (
    DEFAULT_CONFIG,
    _deep_merge,
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
    settings = _settings()
    unset = [
        (key, name)
        for key, name in _REQUIRED.items()
        if not str(cfg_get(settings, *key.split(".")) or "").strip()
    ]
    if unset:
        config = root / "config.yaml"
        return _refuse(*(f"{name} is not set: set release.{key} in {config}" for key, name in unset))

    batches, live, _finished = _record()
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
    print(f"All {len(GUARDS)} release guards passed. Nothing was written.")
    return 0


def status() -> int:
    """Print the waiting batch and the last outcome in plain words; write nothing."""
    batches, _live, finished = _record()
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
    print("Nothing was written.")
    return 1


def _settings() -> dict[str, Any]:
    """The ``release`` settings over their defaults, from config.yaml as it is on disk: loading
    the full configuration would create the home's directories and files."""
    raw = read_raw_config().get("release")
    return _deep_merge(DEFAULT_CONFIG["release"], raw if isinstance(raw, dict) else {})


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
    all from one snapshot of the release record.

    The record is opened read-only, never through release_ledger.connect(), which creates and
    upgrades it; a record that does not exist yet reads as empty.
    """
    path = release_ledger.ledger_path()
    if not path.is_file():
        return [], None, None
    conn = sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True, isolation_level=None)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("BEGIN")
        finished = conn.execute(_LAST_FINISHED_SQL).fetchone()
        return (
            release_ledger.list_batches(conn),
            release_ledger.last_released(conn),
            None if finished is None else finished["batch_id"],
        )
    finally:
        conn.close()
