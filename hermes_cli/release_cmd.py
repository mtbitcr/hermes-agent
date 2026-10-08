"""``hermes release``: check a release before it runs (release card 4), run it (card 5), and start
its run as its own user service (card 6).

``prepare`` works out PREV and NEW (S-B), builds both live adapters from the ``release`` settings
(S-E) and asks the host every merged guard, printing each result in plain words. It writes
nothing to the release state: config.yaml is read as it is on disk, the release record is opened
read-only, and the guards only read. ``status`` prints the waiting batch and the last
outcome. ``run`` releases the accepted batch after the same preflight, under the pause (S-A), in
the calling process: SIGINT or SIGTERM refuses a release until its runner starts; after that, only
a hard kill ends it, and its recovery belongs to card 7. ``start BATCH`` runs ``release run BATCH``
as the release unit (S-F), the transient user service of release_unit, so the release keeps
running when the gateway stops; the start writes nothing to the release state either. ``recover
BATCH`` brings the platform back from a release that stopped midway, as ``run BATCH`` does for a
batch the record shows releasing or failed. ``run`` and ``recover`` share one host lock: while one
of them is at work, another refuses at once and writes nothing.

Design rule (K6), as in the runner: everything ``run`` and ``recover`` need, the host lock
included, is imported when this module loads, so nothing is imported once the units stop and the
checkout moves under the running process.
"""

from __future__ import annotations

import contextlib
import fcntl
import json
import os
import secrets
import signal
import sqlite3
import stat
import time
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agent import estop
from hermes_cli import release_guards, release_ledger, release_runner, release_unit
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
# The batches start runs: an accepted one, or one whose release began and has not ended (card 7).
_STARTABLE = ("accepted", "releasing", "failed")
# The batches whose release stopped midway, which recovery takes, at most this many times.
_UNFINISHED = ("releasing", "failed")
_RECOVERY_ATTEMPTS = 3
_CAPPED = "Batch {} stays failed after {} recovery attempts: a person must bring the platform back."
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
# The host lock that run and recover share, a file in the root home (see _locked).
_LOCK = ".release.lock"


class _UnreadableRecord(Exception):
    """The release record could not be read; the message is the plain reason."""


def cmd_release(args) -> int:
    """``hermes release prepare``, ``status``, ``run [BATCH]``, ``start BATCH`` or
    ``recover BATCH``."""
    if args.release_command == "start":
        return start(args.batch)
    if args.release_command == "recover":
        return recover(args.batch)
    if args.release_command == "run":
        return run() if args.batch is None else run(args.batch)
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


def _root_settings() -> int | tuple[Path, dict[str, Any]]:
    """The root home and its release settings, each required one set and resolved; or, once a
    refusal is printed, its exit code."""
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
    return root, settings


def _inputs(batch_id: int | None = None) -> int | tuple[
    dict[str, Any], list[RecordedMerge], Pins, LiveHostReader, ReleaseHostActions, dict[str, Any]
]:
    """What prepare and run work on: the accepted batch, its changes, both versions, the live
    reader and actions, and the release settings; or, once a refusal is printed, its exit code.
    Given ``batch_id``, a batch that is not the accepted batch taken first refuses."""
    found = _root_settings()
    if isinstance(found, int):
        return found
    root, settings = found

    try:
        batches, live, _finished = _record()
    except _UnreadableRecord as error:
        return _refuse(str(error))
    accepted = [batch for batch in batches if batch["state"] == "accepted"]
    if not accepted:
        return _refuse("no batch is accepted, so there is nothing to release")
    batch = accepted[0]
    if batch_id is not None and batch["batch_id"] != batch_id:
        return _refuse(
            f"batch {batch_id} is not the accepted batch a release run takes first"
            f" (batch {batch['batch_id']} is)"
        )
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


def run(batch_id: int | None = None) -> int:
    """Release the accepted batch between a pause and a resume, in card 5's order.

    First, before any other step, remove every HERMES_KANBAN variable from the run's environment,
    so no board context reaches the release or a process it starts. Given ``batch_id``, as the
    release unit gives it, the run releases only that batch, and refuses and writes nothing unless
    it is the accepted batch a run takes first: a restart of the unit never takes another batch.
    A batch the record shows releasing or failed, whose release stopped midway, goes straight to
    ``recover`` instead (card 7). The run holds the host lock from before its first read of the
    record until it ends (see _locked): while another run or recovery holds it, the run refuses
    at once and writes nothing.

    Mark the batch releasing with both versions, fetch NEW and save the named configuration
    snapshot, and run prepare's preflight. A pause already in place, or a failed guard other than
    G10, refuses the release and pauses nothing. Otherwise publish the pause (S-A), whole, with the
    release reason and a release token, wait until the merged G10 holds, and run the merged runner
    for the batch's tier: tier 2 when a change has tier 2 or no recorded tier, forward only
    otherwise. The pause is read back after its publication and again just before the runner,
    which asks every guard again before it stops anything. Then record the outcome and resume,
    except after a failed outcome: the platform stays paused for a person.

    SIGINT and SIGTERM only set a stop flag, read after the pause's publication, at every drain
    poll and just before the runner, where it refuses the release; from then on the runner's own
    outcome stands. The runner's first unit stop reads the pause again and refuses the release
    for good unless it is the release's own. A runner result that never comes back fails it midway.

    Returns 0 when the batch was released, 1 otherwise, and 1 when the release's own pause is
    still in place after its removal.
    """
    for name in [name for name in os.environ if name.startswith("HERMES_KANBAN")]:
        del os.environ[name]
    return _locked(_run, batch_id)


def _run(batch_id: int | None) -> int:
    """``run`` under the host lock."""
    if batch_id is not None and _unfinished(batch_id):
        return _recover(batch_id)
    inputs = _inputs(batch_id)
    if isinstance(inputs, int):  # refused before anything was written
        return inputs
    _batch, _merges, _pins, reader, actions, _settings = inputs
    token = secrets.token_hex(16)  # the release token: only the release's own pause carries it
    host = _Host(reader, actions, token)
    # The handlers only set the host's stop flag. They are in place before the begin commit, and
    # the caller's are back only once the outcome is recorded and the pause's removal read back.
    previous = {
        signum: signal.signal(signum, host.stop) for signum in (signal.SIGINT, signal.SIGTERM)
    }
    try:
        return _release(inputs, host, token)
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler or signal.SIG_DFL)


def _release(inputs: tuple[Any, ...], host: _Host, token: str) -> int:
    """``run`` from the begin commit to the readback of the pause's removal, under its handlers."""
    batch, merges, pins, reader, actions, settings = inputs
    batch_id, reason = batch["batch_id"], f"release {batch['batch_id']}"
    try:
        conn = release_ledger.connect()
    except (OSError, sqlite3.Error) as error:
        return _refuse(f"the release record could not be opened ({type(error).__name__}: {error})")
    with contextlib.closing(conn):
        try:
            release_ledger.begin_release(conn, batch_id, prev=pins.prev, new=pins.new)
        except (ValueError, sqlite3.Error) as error:
            return _refuse(f"batch {batch_id} could not begin its release ({error})")
        refusals, result = [], None
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
                _hold(token, host.stopped)
                _drain(reader, pins, merges, settings["drain_poll_seconds"], host)
                tier = max(m["tier"] if m["tier_recorded"] else 2 for m in batch["members"])
                _hold(token, host.stopped)
                result = run_release(host, pins, tier, merges)
                if result.outcome != "refused" and not host.cut_over:
                    # At its first unit stop the pause was not the release's own: nothing moved.
                    result = None
                    raise _Stopped("stopped before cutover")
        except _Stopped as stop:
            refusals.append(str(stop))
        except Exception as error:  # an action, or a read in the drain or the fresh guards (F5)
            refusals.append(f"a step before the cutover failed ({type(error).__name__}: {error})")
        if host.cut_over and result is None:
            # From the first unit stop on, a runner result that never came back is never a
            # refusal: the release failed midway and stays paused for its recovery (card 7).
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
    own = _final_token()
    lift = outcome != "failed" and own == token
    if lift:
        estop.disengage()
    print(f"Batch {batch_id} {_OUTCOMES[outcome]}.")
    after = _final_token() if lift else own
    if after is _UNREADABLE:  # the release cannot know whether its own pause is still in place
        print("The pause could not be read: the platform may still be paused by this release.")
        return 1
    if state is not None and not lift:
        print(f"The platform stays paused ({state['reason'] or 'no reason given'}).")
    elif lift and after == token:  # the removal failed: the release's own pause is in place
        print(
            f"The platform is still paused ({reason}): the release could not remove its pause"
            f" at {estop.sentinel_path()}."
        )
        return 1
    return 0 if outcome == "released" else 1


def _guards_failed(guards: list[str]) -> str:
    return f"release guard{'s' * (len(guards) > 1)} {', '.join(guards)} failed"


def _drain(
    reader: LiveHostReader, pins: Pins, merges: list[RecordedMerge], seconds: float, host: _Host
) -> None:
    """Wait until the merged G10 holds, asking it every ``seconds``. The wait has no time limit
    of its own (decision B): the host's stop flag, read at every poll, ends it."""
    [check] = [check for guard, _rule, check in release_guards.GUARDS if guard == "G10"]
    while not check(reader, pins, merges):
        if host.stopped:
            raise _Stopped("stopped before cutover")
        time.sleep(seconds)


class _Host:
    """The live reader and actions as the one release host the runner asks. A name the actions
    have is theirs, so ``checkout`` is the move; every other name is the reader's.

    The host also holds the run's stop flag, which its SIGINT and SIGTERM handler only sets. Until
    the cutover, it hands an action over only while the release's own pause is in place, read each
    time: the runner's first unit stop, after the fresh guards, reads it and marks the cutover. A
    pause that is not the release's own refuses that action, and the refusal is final: every later
    action is refused too, without reading the pause again, so the runner has no unit to stop and
    no move back.
    """

    def __init__(self, reader: LiveHostReader, actions: ReleaseHostActions, token: str) -> None:
        self._reader, self._actions, self._release_token = reader, actions, token
        self.cut_over = self.stopped = self.refused = False

    def __getattr__(self, name: str) -> Any:
        owner = self._actions if hasattr(self._actions, name) else self._reader
        if owner is self._actions and not self.cut_over:
            if self.refused or _token() != self._release_token:
                self.refused = True  # for the rest of the run: the pause is not read again
                raise _Stopped("stopped before cutover")
            self.cut_over = name == "stop_units"
        return getattr(owner, name)

    def stop(self, signum: int, frame: Any) -> None:
        self.stopped = True


class _Stopped(BaseException):
    """A stop signal, or the release's pause gone before the cutover, ended the run. Like an
    interrupt, no ``except Exception`` takes it."""


def _pause(reason: str, token: str) -> bool:
    """Publish the pause sentinel whole: estop's payload and the release token are written to a
    temporary file beside it, which is then linked to the sentinel's name. The link fails when a
    pause is in place, so an owner's pause is never written over: False then. Any other failure
    to publish is left to the readback, which refuses. Once created, the temporary file is
    removed; an error of that removal other than a missing file is raised, so the run refuses
    before the cutover, with its type and errno only: its file name and text carry the token."""
    path = estop.sentinel_path()
    temporary = path.with_name(f".{path.name}.{token}")
    payload = {"engaged_at": datetime.now(timezone.utc).isoformat(), "reason": reason}
    created = False
    try:
        fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o666)
        created = True
        with os.fdopen(fd, "w", encoding="utf-8") as sentinel:
            sentinel.write(json.dumps({**payload, "release_token": token}, indent=2) + "\n")
        os.link(temporary, path)
    except FileExistsError:
        return False
    except OSError:
        pass
    finally:
        if created:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass
            except OSError as error:  # its name and text carry the token: type and errno only
                step = "the temporary pause file could not be removed"
                raise type(error)(error.errno, step) from None
    return True


def _token() -> str | None:
    """The release token in the pause sentinel, or None when there is none."""
    with contextlib.suppress(OSError, ValueError, AttributeError):
        return json.loads(estop.sentinel_path().read_text(encoding="utf-8")).get("release_token")
    return None


_UNREADABLE = object()  # the pause sentinel may exist but cannot be read


def _final_token() -> object:
    """The release token for the final removal, which must tell an unreadable sentinel from an
    absent one: None when there is no sentinel or it carries no token (an owner's pause),
    _UNREADABLE when it cannot be read, otherwise the token."""
    try:
        text = estop.sentinel_path().read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    except OSError:
        return _UNREADABLE
    with contextlib.suppress(ValueError):
        data = json.loads(text)
        return data.get("release_token") if isinstance(data, dict) else None
    return None  # not the release's own sentinel, which is always written whole


def _hold(token: str, stopped: bool = False) -> None:
    """Stop the release before the cutover when ``stopped``, or unless its own pause, with
    ``token``, is in place."""
    if stopped or _token() != token:
        raise _Stopped("stopped before cutover")


def _unfinished(batch_id: int) -> bool:
    """Whether the record shows ``batch_id`` releasing or failed: its release stopped midway. A
    record that cannot be read is the run's to refuse."""
    with contextlib.suppress(_UnreadableRecord):
        batches, _live, _finished = _record()
        state = next((batch["state"] for batch in batches if batch["batch_id"] == batch_id), None)
        return state in _UNFINISHED
    return False


def recover(batch_id: int) -> int:
    """Bring the platform back from a release of ``batch_id`` that stopped midway (card 7), to
    exactly PREV or exactly NEW with a passing readback, and never forward.

    As the run does, first remove every HERMES_KANBAN variable, then take the host lock: while
    another run or recovery holds it, recovery refuses at once and writes nothing. A batch that is
    not releasing or failed refuses. Past three attempts the host is not asked, nor when the count
    begin_recovery returns is past three: the outcome stays failed and recovery exits cleanly, so
    the release unit is not started again. Otherwise count the attempt, run the runner's recover
    with the stored versions (S-B) and the saved configuration snapshot (S-C), and record the
    outcome. After a passing readback, lift only the release's own pause, the one with its reason
    and a token. An owner's pause stays; a failed recovery keeps any pause.

    Returns 0 when the batch was released, and once its third attempt failed; 1 otherwise, and 1
    when the pause cannot be read or the release's own pause is still in place after its removal.
    """
    for name in [name for name in os.environ if name.startswith("HERMES_KANBAN")]:
        del os.environ[name]
    return _locked(_recover, batch_id)


def _recover(batch_id: int) -> int:
    """``recover`` under the host lock, which ``run`` holds too when it recovers a batch."""
    found = _root_settings()
    if isinstance(found, int):
        return found
    root, settings = found
    try:
        batches, _live, _finished = _record()
        batch = next((batch for batch in batches if batch["batch_id"] == batch_id), None)
        if batch is None:
            return _refuse(f"there is no batch {batch_id} in the release record")
        if batch["state"] not in _UNFINISHED:
            return _refuse(f"batch {batch_id} is {batch['state']}, not releasing or failed")
        pins = Pins(
            new=_text(batch.get("new"), f"batch {batch_id} has no stored NEW"),
            prev=_text(batch.get("prev"), f"batch {batch_id} has no stored PREV"),
        )
    except _UnreadableRecord as error:
        return _refuse(str(error))
    try:
        conn = release_ledger.connect()
    except (OSError, sqlite3.Error) as error:
        return _refuse(f"the release record could not be opened ({type(error).__name__}: {error})")
    with contextlib.closing(conn):
        if batch["recovery_attempts"] >= _RECOVERY_ATTEMPTS:  # the host is not asked again
            if batch["state"] == "releasing":  # the last attempt ended before its outcome
                release_ledger.finish_release(conn, batch_id, outcome="failed")
            print(_CAPPED.format(batch_id, _RECOVERY_ATTEMPTS))
            return 0
        try:
            attempt = release_ledger.begin_recovery(conn, batch_id)["recovery_attempts"]
        except (ValueError, sqlite3.Error) as error:
            return _refuse(f"batch {batch_id} could not begin its recovery ({error})")
        if attempt > _RECOVERY_ATTEMPTS:  # the count its own write returned: the host is not asked
            release_ledger.finish_release(conn, batch_id, outcome="failed")
            print(_CAPPED.format(batch_id, _RECOVERY_ATTEMPTS))
            return 0
        print(
            f"Batch {batch_id} stopped midway: recovery attempt {attempt} of"
            f" {_RECOVERY_ATTEMPTS}, PREV {pins.prev}, NEW {pins.new}"
        )
        reader, actions = _adapters(root, settings, pins.new)
        actions.config_snapshot = pins.new  # the configuration snapshot the release saved (S-C)
        host = _Host(reader, actions, "")
        host.cut_over = True  # the release cut over before it stopped: every action goes through
        result = release_runner.recover(host, pins)
        if result.error:
            print(f"The checked-out version did not read back ({result.error}).")
            for error in result.restore_errors:
                print(f"    Going back, a step failed too: {error}")
        outcome = result.outcome
        release_ledger.finish_release(conn, batch_id, outcome=outcome)
    # S-A, as in the run: lift only the release's own pause unless the outcome failed. The removal
    # is read back, and never tried again.
    reason = f"release {batch_id}"
    state, own = estop.get_state(), _final_token()
    lift = outcome != "failed" and isinstance(own, str) and (state or {}).get("reason") == reason
    if lift:
        estop.disengage()
    print(f"Batch {batch_id} {_OUTCOMES[outcome]}.")
    after = _final_token() if lift else own
    if after is _UNREADABLE:  # recovery cannot know whether the release's own pause is in place
        print("The pause could not be read: the platform may still be paused by this release.")
        return 1
    if state is not None and not lift:
        print(f"The platform stays paused ({state['reason'] or 'no reason given'}).")
    elif lift and after == own:  # the removal failed: the release's own pause is in place
        print(
            f"The platform is still paused ({reason}): the recovery could not remove its pause"
            f" at {estop.sentinel_path()}."
        )
        return 1
    if outcome == "failed" and attempt >= _RECOVERY_ATTEMPTS:  # the unit is not started again
        print(_CAPPED.format(batch_id, _RECOVERY_ATTEMPTS))
        return 0
    return 0 if outcome == "released" else 1


def _locked(command: Callable[[Any], int], batch_id: int | None) -> int:
    """``command(batch_id)`` under the host lock that ``run`` and ``recover`` share: an exclusive
    flock on _LOCK in the root home, taken before the command reads or writes the release record
    and held until it returns; the process's exit lets go of it too. While another run or recovery
    holds it, the command refuses at once and writes nothing. The lock file is removed before the
    lock is let go, so the root home keeps no file of it; a command that locked a file removed in
    the meantime refuses too."""
    path = get_default_hermes_root() / _LOCK
    try:
        fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
    except OSError as error:
        return _refuse(
            f"the release lock {path} could not be opened ({type(error).__name__}: {error})"
        )
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        held = os.path.samestat(os.fstat(fd), os.stat(path))
    except (BlockingIOError, FileNotFoundError):
        held = False
    if not held:
        os.close(fd)
        return _refuse(f"another release run or recovery holds the release lock {path}")
    try:
        return command(batch_id)
    finally:
        with contextlib.suppress(OSError):
            os.unlink(path)
        os.close(fd)


def start(batch_id: int) -> int:
    """Run ``release run`` for ``batch_id`` as its release unit, the transient user service of
    S-F (card 6). The start writes nothing to the release state: the run in the unit begins the
    release.

    First, before any other step, remove every HERMES_KANBAN variable from the start's
    environment, as the run does, so its reads of the release record, before the launch and after
    a failed launch or readback, read the record of the root home the launch runs the unit from:
    the record the run in the unit reads.

    Every read comes first: systemd-run, the batches in the release record, and the release units
    the user manager has loaded. A missing systemd-run, a batch that is not accepted, releasing or
    failed, an accepted batch behind an older accepted one, which a release run releases first,
    and a release unit at work each refuse. Then one action: systemd-run asks the user manager to
    make the unit named for the batch and start it, in one call that also reserves the name. The
    unit is read back until the user manager answers active, for at most
    release_unit.READBACK_SECONDS; otherwise, or when the launch fails, the release could not
    start: the start reads the record again and prints the user manager's last answer and the
    batch's state as the record now shows it.

    Returns 0 once the unit answers active, 1 otherwise.
    """
    for name in [name for name in os.environ if name.startswith("HERMES_KANBAN")]:
        del os.environ[name]
    systemd_run = release_unit.find_systemd_run()
    if systemd_run is None:  # fails closed, as update_abort_recovery does
        return _refuse("systemd-run is missing, so the release cannot run as its own user service")
    try:
        batches, _live, _finished = _record()
    except _UnreadableRecord as error:
        return _refuse(str(error))
    state = next((batch["state"] for batch in batches if batch["batch_id"] == batch_id), None)
    accepted = [batch["batch_id"] for batch in batches if batch["state"] == "accepted"]
    if state is None:
        return _refuse(f"there is no batch {batch_id} in the release record")
    if state not in _STARTABLE:
        return _refuse(f"batch {batch_id} is {state}, not accepted, releasing or failed")
    if state == "accepted" and accepted[0] != batch_id:
        return _refuse(
            f"batch {batch_id} is accepted after batch {accepted[0]}, which a release run"
            " releases first"
        )
    try:
        busy = release_unit.busy_units()
    except release_unit.UnitError as error:
        return _refuse(f"the release units could not be read ({error})")
    if busy:
        return _refuse(*(f"a release is already at work: {unit} is {now}" for unit, now in busy))
    unit = release_unit.unit_name(batch_id)
    try:
        release_unit.launch(systemd_run, batch_id, get_default_hermes_root())
    except release_unit.UnitError as error:
        return _could_not_start(batch_id, str(error), state)
    answer = release_unit.read_back(batch_id)
    if answer != "active":
        return _could_not_start(
            batch_id,
            f"the user manager did not answer active for {unit} within"
            f" {release_unit.READBACK_SECONDS:g} seconds (its last answer: {answer})",
            state,
        )
    print(
        f"Started the release of batch {batch_id} as the user service {unit}: the user manager"
        " answers active."
    )
    print(f"Follow it with: journalctl --user -u {unit}")
    return 0


def _could_not_start(batch_id: int, reason: str, before: str) -> int:
    """Print why the release could not start, and what became of the batch, which was ``before``
    when the start read the record. The unit's run may have written to the record since: it is
    read again, and the batch was left as it was only when its state is the same."""
    print(f"The release of batch {batch_id} could not start: {reason.rstrip('.')}.")
    try:
        batches, _live, _finished = _record()
    except _UnreadableRecord as error:
        print(f"Batch {batch_id} was {before} before the start; its state now is unknown: {error}.")
        return 1
    now = next((batch["state"] for batch in batches if batch["batch_id"] == batch_id), "missing")
    if now == before:
        print(f"Batch {batch_id} was left as it was: the release record still shows it {now}.")
    else:
        print(
            f"Batch {batch_id} was {before} before the start; the release record now shows it"
            f" {now}."
        )
    return 1


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
