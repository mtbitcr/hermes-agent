"""``hermes release``: check a release before it runs (release card 4), and run it (card 5).

``prepare`` works out PREV and NEW (S-B), builds both live adapters from the ``release`` settings
(S-E) and asks the host every merged guard, printing each result in plain words. It writes
nothing to the release state: config.yaml is read as it is on disk, the release record is opened
read-only, and the guards only read. ``status`` prints the waiting batch and the last
outcome. ``run`` releases the accepted batch after the same preflight, under the pause (S-A), in
the calling process: SIGINT or SIGTERM refuses a release that has not cut over yet, and fails one
that has, which stays paused for its recovery (card 7). The release unit (S-F) belongs to card 6.

Design rule (K6), as in the runner: everything ``run`` needs is imported when this module loads,
so nothing is imported once the units stop and the checkout moves under the running process.
"""

from __future__ import annotations

import contextlib
import json
import os
import secrets
import signal
import sqlite3
import stat
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agent import estop
from hermes_cli import release_guards, release_ledger
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
from hermes_cli.release_runner import SANDBOX_TUNNEL_UNIT, run_release
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
    """``hermes release prepare``, ``hermes release status`` or ``hermes release run``."""
    if args.release_command == "run":
        return run()
    return prepare() if args.release_command == "prepare" else status()


def prepare() -> int:
    """Ask the host every guard for the accepted batch and print each result; write nothing.

    Returns 0 when every guard passed, 1 when one failed or the release was refused before them.
    """
    inputs = _inputs()
    if isinstance(inputs, int):  # refused before the guards
        return inputs
    batch, merges, pins, reader, _actions, _settings = inputs
    failed = _preflight(batch, merges, pins, reader)
    if failed:
        return _refuse(f"{len(failed)} of {len(GUARDS)} release guards failed")
    print(f"All {len(GUARDS)} release guards passed. Nothing in the release state was written.")
    return 0


def _inputs() -> int | tuple[
    dict[str, Any], list[RecordedMerge], Pins, LiveHostReader, ReleaseHostActions, dict[str, Any]
]:
    """What prepare and run work on: the accepted batch, its changes, both versions, the live
    reader and actions, and the release settings; or, once a refusal is printed, its exit code."""
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
    reader, actions = _adapters(root, settings, new)
    try:
        prev = live or reader.checkout_head()
    except Exception as error:
        return _refuse(
            f"the checkout's head could not be read, so PREV is unknown"
            f" ({type(error).__name__}: {error})"
        )
    pins = Pins(new=new, prev=prev)
    return batch, merges, pins, reader, actions, settings


def _preflight(
    batch: dict[str, Any], merges: list[RecordedMerge], pins: Pins, reader: LiveHostReader
) -> list[str]:
    """Card 4's preflight: print both versions, then ask the host every guard and print each
    result in plain words. Returns the guards that failed."""
    print(f"Batch {batch['batch_id']}: PREV {pins.prev}, NEW {pins.new}")
    failed = []
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
        if not ok:
            failed.append(guard)
    return failed


def run() -> int:
    """Release the accepted batch between a pause and a resume, in card 5's order.

    Mark the batch releasing with both versions, fetch NEW and save the named configuration
    snapshot, and run prepare's preflight. A pause already in place, or a failed guard other than
    G10, refuses the release and pauses nothing. Otherwise publish the pause (S-A), whole, with the
    release reason and a release token, wait until the merged G10 holds, and run the merged runner
    for the batch's tier: tier 2 when a change has tier 2 or no recorded tier, forward only
    otherwise. The pause is read back after its publication and again just before the runner,
    which asks every guard again before it stops anything. Then record the outcome and resume,
    except after a failed outcome: the platform stays paused for a person.

    SIGINT and SIGTERM refuse the release until the runner stops its first unit. From then on, a
    stop, or a runner result that never comes back, fails the release midway.

    Returns 0 when the batch was released, 1 otherwise, and 1 when the release's own pause is
    still in place after its removal.
    """
    inputs = _inputs()
    if isinstance(inputs, int):  # refused before anything was written
        return inputs
    batch, merges, pins, reader, actions, settings = inputs
    batch_id, reason = batch["batch_id"], f"release {batch['batch_id']}"
    token = secrets.token_hex(16)  # the release token: only the release's own pause carries it
    try:
        conn = release_ledger.connect()
    except (OSError, sqlite3.Error) as error:
        return _refuse(f"the release record could not be opened ({type(error).__name__}: {error})")
    with contextlib.closing(conn):
        try:
            release_ledger.begin_release(conn, batch_id, prev=pins.prev, new=pins.new)
        except (ValueError, sqlite3.Error) as error:
            return _refuse(f"batch {batch_id} could not begin its release ({error})")
        refusals, result, host = [], None, _Host(reader, actions)
        previous = {
            signum: signal.signal(signum, host.stop) for signum in (signal.SIGINT, signal.SIGTERM)
        }
        try:
            actions.fetch(pins.new)
            actions.save_config_snapshot(pins.new)
            failed = [guard for guard in _preflight(batch, merges, pins, reader) if guard != "G10"]
            if estop.is_engaged():
                refusals.append("the platform is already paused")
            if failed:
                refusals.append(_guards_failed(failed))
            # The publication refuses an owner's pause written since that read, too.
            if not refusals and not _pause(reason, token):
                refusals.append("the platform is already paused")
            if not refusals:
                _hold(token)
                _drain(reader, pins, merges, settings["drain_poll_seconds"])
                tier = max(m["tier"] if m["tier_recorded"] else 2 for m in batch["members"])
                _hold(token)
                result = run_release(host, pins, tier, merges)
        except _Stopped as stop:
            refusals.append(str(stop))
        except Exception as error:  # an action, or a read in the drain or the fresh guards (F5)
            refusals.append(f"a step before the cutover failed ({type(error).__name__}: {error})")
        finally:
            for signum, handler in previous.items():
                signal.signal(signum, handler or signal.SIG_DFL)
        if host.cut_over and (host.stopped or result is None):
            # From the first unit stop on, a stop, or a runner result that never came back, is
            # never a refusal: the release failed midway and stays paused for its recovery (card 7).
            outcome = "failed"
            print("The release failed after the cutover (stopped midway).")
        else:
            if result is not None and result.outcome == "refused":
                fresh = [check.guard for check in result.guards if not check.ok]
                refusals.append(f"{_guards_failed(fresh)} when asked again before the stop")
            outcome = "refused" if result is None else result.outcome
            for refusal in refusals:
                print(f"Refused: {refusal}.")
            if result is not None and result.error:
                print(f"The release failed after the cutover ({result.error}).")
                for error in result.restore_errors:
                    print(f"    Going back, a step failed too: {error}")
        release_ledger.finish_release(conn, batch_id, outcome=outcome)
    # S-A: lift only the release's own pause, the one with its token, unless the outcome failed;
    # an owner's pause stays. One written between the token's read and the removal goes with it,
    # by design: the owner page then shows the platform running, and the owner can pause again.
    # The removal is read back, and never tried again.
    state = estop.get_state()
    lift = outcome != "failed" and _token() == token
    if lift:
        estop.disengage()
    print(f"Batch {batch_id} {_OUTCOMES[outcome]}.")
    if state is not None and not lift:
        print(f"The platform stays paused ({state['reason'] or 'no reason given'}).")
    elif lift and _token() == token:  # the removal failed: the release's own pause is in place
        print(
            f"The platform is still paused ({reason}): the release could not remove its pause"
            f" at {estop.sentinel_path()}."
        )
        return 1
    return 0 if outcome == "released" else 1


def _guards_failed(guards: list[str]) -> str:
    return f"release guard{'s' * (len(guards) > 1)} {', '.join(guards)} failed"


def _drain(
    reader: LiveHostReader, pins: Pins, merges: list[RecordedMerge], seconds: float
) -> None:
    """Wait until the merged G10 holds, asking it every ``seconds``. The wait has no time limit
    of its own (decision B): a stop signal ends it."""
    [check] = [check for guard, _rule, check in release_guards.GUARDS if guard == "G10"]
    while not check(reader, pins, merges):
        time.sleep(seconds)


class _Host:
    """The live reader and actions as the one release host the runner asks. A name the actions
    have is theirs, so ``checkout`` is the move; every other name is the reader's.

    The host is also the run's SIGINT and SIGTERM handler. It marks the cutover just before the
    runner's first unit stop is handed over, so a stop in that very call comes after the cutover.
    """

    def __init__(self, reader: LiveHostReader, actions: ReleaseHostActions) -> None:
        self._reader, self._actions = reader, actions
        self.cut_over = self.stopped = False

    def __getattr__(self, name: str) -> Any:
        if name == "stop_units":
            self.cut_over = True
        return getattr(self._actions if hasattr(self._actions, name) else self._reader, name)

    def stop(self, signum: int, frame: Any) -> None:
        self.stopped = True
        raise _Stopped("stopped midway" if self.cut_over else "stopped before cutover")


class _Stopped(BaseException):
    """A stop signal, or the release's pause gone before the cutover, ended the run. Like an
    interrupt, no ``except Exception`` takes it."""


def _pause(reason: str, token: str) -> bool:
    """Publish the pause sentinel whole: estop's payload and the release token are written to a
    temporary file beside it, which is then linked to the sentinel's name. The link fails when a
    pause is in place, so an owner's pause is never written over: False then. Any other failure
    is left to the readback, which refuses. The temporary file is removed in every case."""
    path = estop.sentinel_path()
    temporary = path.with_name(f".{path.name}.{token}")
    payload = {"engaged_at": datetime.now(timezone.utc).isoformat(), "reason": reason}
    try:
        fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o666)
        with os.fdopen(fd, "w", encoding="utf-8") as sentinel:
            sentinel.write(json.dumps({**payload, "release_token": token}, indent=2) + "\n")
        os.link(temporary, path)
    except FileExistsError:
        return False
    except OSError:
        pass
    finally:
        with contextlib.suppress(OSError):
            os.unlink(temporary)
    return True


def _token() -> str | None:
    """The release token in the pause sentinel, or None when there is none."""
    with contextlib.suppress(OSError, ValueError, AttributeError):
        return json.loads(estop.sentinel_path().read_text(encoding="utf-8")).get("release_token")
    return None


def _hold(token: str) -> None:
    """Stop the release before the cutover unless its own pause, with ``token``, is in place."""
    if _token() != token:
        raise _Stopped("stopped before cutover")


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
