"""Shared SQLite primitives for the small per-profile / board stores.

The projects and kanban stores open WAL SQLite files with the same three
primitives — an idempotent column-add migration, an IMMEDIATE write
transaction, and a cross-process first-connect lock. One definition here keeps
the two stores from drifting.
"""

from __future__ import annotations

import contextlib
import sqlite3
import sys
import time
from pathlib import Path

_IS_WINDOWS = sys.platform == "win32"

# Bounded acquire for the cross-process init lock (issue #36644). A bare
# blocking lock had no timeout, so a wedged holder blocked every other
# ``connect()`` forever. We retry a non-blocking acquire up to this deadline,
# polling at this interval, then proceed without the cross-process lock — the
# caller's in-process lock plus idempotent init remain the backstop.
INIT_LOCK_TIMEOUT_SECONDS = 10.0
INIT_LOCK_POLL_SECONDS = 0.05


class InitLockUnavailable(RuntimeError):
    """The cross-process first-connect lock could not be taken in time.

    Raised only for a ``required=True`` caller, which has declared that running
    its init work unserialized is not an acceptable outcome.
    """


class InitLockDirectoryAbsent(FileNotFoundError):
    """A ``require_existing_directory=True`` caller's directory was not there.

    The lock file is a sibling of the database, so materialising it in an
    absent directory would ``mkdir`` that directory back into existence. A
    caller that declared the directory a precondition gets this instead —
    raised from the ``ENOENT`` the OPEN ITSELF returned, so it reports what
    the operating system decided at that instant rather than what a
    preceding ``exists()`` check believed.

    A :class:`FileNotFoundError` subclass so a caller that only wants "the
    lock could not be placed" can keep catching ``OSError``.
    """


@contextlib.contextmanager
def cross_process_init_lock(
    path: Path,
    *,
    timeout_seconds: float = INIT_LOCK_TIMEOUT_SECONDS,
    poll_seconds: float = INIT_LOCK_POLL_SECONDS,
    required: bool = False,
    on_timeout=None,
    require_existing_directory: bool = False,
):
    """Serialize one database's first-connect setup across processes.

    A thread lock only protects one Python process. During a burst — a
    dispatcher spawning workers, or several owner requests arriving at once —
    many processes can hit a fresh database simultaneously, each with an empty
    per-process "already initialized" cache. This file lock keeps WAL
    activation, schema creation and additive migrations single-writer across
    the whole host while leaving normal post-init usage concurrent under WAL.

    The lock file is a stable sibling of the database (``<db>.init.lock``), so
    nothing temporary is left behind and the same path is computed by every
    competing process.

    The acquire is always bounded; what happens at the deadline is the
    caller's explicit choice, because the two stores want opposite answers:

    * ``required=False`` (default) — best effort. ``on_timeout`` is called so
      the caller can warn, and the body then runs WITHOUT the cross-process
      lock. Correct for a store whose init work is purely idempotent and whose
      availability matters more, e.g. the kanban dispatch bus (issue #36644),
      where an unbounded hang silently stops the board.
    * ``required=True`` — fail closed. :class:`InitLockUnavailable` is raised
      and the body never runs, so no caller can be handed a connection whose
      first-open setup was never serialized. Correct wherever "this ran
      unserialized" is itself the failure, not merely redundant work.

    There is deliberately no third mode: a caller that neither passes
    ``required`` nor reads ``on_timeout`` gets the documented best-effort
    contract rather than a silent one.

    ``require_existing_directory`` decides who owns the containing directory:

    * ``False`` (default) — this helper materialises it
      (``mkdir(parents=True, exist_ok=True)``). Correct for the per-profile
      projects store and for the kanban board register's own lock directory,
      which are infrastructure paths no authority governs.
    * ``True`` — the directory is a PRECONDITION of the call. Nothing is
      created on the way to the lock, and an absent directory raises
      :class:`InitLockDirectoryAbsent`. Correct for a store whose existence
      some authority admits or refuses (the kanban boards), where a helper
      that quietly re-created the directory would resurrect a removed board's
      identity behind that authority's back.

      This is deliberately NOT ``if not path.parent.exists(): raise`` — that
      is the same check/use race one function further out. The open below is
      the only thing that touches the filesystem, it never creates a
      directory, and it is the KERNEL that decides whether the directory is
      still there. A removal landing at any instant, including between the
      last instruction and the ``open`` itself, therefore yields ``ENOENT``
      rather than a re-created directory with a lock file in it. Creating
      the lock FILE inside a directory that is still there is allowed and
      is not a resurrection: a fresh store and a legacy store both need one,
      and its parent is what carries the board's discoverable identity.
    """
    path = Path(path)
    lock_path = path.with_name(path.name + ".init.lock")
    if require_existing_directory:
        try:
            handle = lock_path.open("a+b")
        except (FileNotFoundError, NotADirectoryError) as exc:
            raise InitLockDirectoryAbsent(
                f"{lock_path.parent} does not exist: refusing to create it to "
                f"place {lock_path.name}"
            ) from exc
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        handle = lock_path.open("a+b")
    acquired = False
    try:
        deadline = time.monotonic() + timeout_seconds
        if _IS_WINDOWS:
            import msvcrt

            locking = getattr(msvcrt, "locking")
            nb_lock = getattr(msvcrt, "LK_NBLCK")
            while True:
                try:
                    handle.seek(0)
                    locking(handle.fileno(), nb_lock, 1)
                    acquired = True
                    break
                except OSError:
                    if time.monotonic() >= deadline:
                        break
                    time.sleep(poll_seconds)
        else:
            import fcntl

            while True:
                try:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    acquired = True
                    break
                except (BlockingIOError, OSError):
                    if time.monotonic() >= deadline:
                        break
                    time.sleep(poll_seconds)
        if not acquired:
            if required:
                raise InitLockUnavailable(
                    f"{lock_path} was not acquired within {timeout_seconds:g}s"
                )
            if on_timeout is not None:
                on_timeout(lock_path, timeout_seconds)
        yield
    finally:
        try:
            if acquired:
                if _IS_WINDOWS:
                    import msvcrt

                    handle.seek(0)
                    locking = getattr(msvcrt, "locking")
                    unlock_mode = getattr(msvcrt, "LK_UNLCK")
                    locking(handle.fileno(), unlock_mode, 1)
                else:
                    import fcntl

                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()


def add_column_if_missing(conn: sqlite3.Connection, table: str, column: str, ddl: str) -> bool:
    """``ALTER TABLE <table> ADD COLUMN <ddl>``, idempotent across races: True when this call added
    it, False on the ``duplicate column name`` a concurrent migrator caused.

    ``column`` is the human-readable name for the call site; ``ddl`` carries the actual definition. See
    #21708.
    """
    try:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {ddl}")
        return True
    except sqlite3.OperationalError as exc:
        if "duplicate column name" in str(exc).lower():
            return False
        raise


@contextlib.contextmanager
def write_txn(conn: sqlite3.Connection):
    """An IMMEDIATE write transaction: at most one concurrent writer wins.

    The explicit ROLLBACK is guarded so a SQLite auto-rollback (no active
    transaction left under EIO / lock contention / corruption) cannot shadow
    the original exception with a spurious rollback error.

    Reentrant on the same connection: if *conn* already has an open
    transaction (``conn.in_transaction``), this call joins it instead of
    issuing a second ``BEGIN`` (which SQLite rejects outright). This lets a
    caller hold one outer fence — e.g. a lease-ownership check that must stay
    valid across a mutation — around store functions (``create_project``,
    ``update_project``, ...) that each open their own ``write_txn`` for
    standalone use: nested under an outer fence, their statements land in the
    SAME atomic transaction instead of racing it. Only the outermost caller's
    ``with`` block actually commits/rolls back; an inner one is a no-op
    wrapper so a raised exception still propagates to the outer block, which
    is the one holding the real transaction.
    """
    if conn.in_transaction:
        yield conn
        return
    conn.execute("BEGIN IMMEDIATE")
    try:
        yield conn
    except Exception:
        try:
            conn.execute("ROLLBACK")
        except sqlite3.OperationalError:
            pass
        raise
    else:
        conn.execute("COMMIT")
