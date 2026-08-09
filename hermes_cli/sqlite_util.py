"""Shared SQLite primitives for the small per-profile / board stores.

The projects and kanban stores open WAL SQLite files with the same two
primitives — an idempotent column-add migration and an IMMEDIATE write
transaction. One definition here keeps the two stores from drifting.
"""

from __future__ import annotations

import contextlib
import sqlite3


def add_column_if_missing(conn: sqlite3.Connection, table: str, column: str, ddl: str) -> bool:
    """``ALTER TABLE <table> ADD COLUMN <ddl>``, idempotent across races.

    Returns ``True`` when this call added the column. Swallows the
    ``duplicate column name`` error a concurrent migrator may have run first
    (issue #21708). ``column`` is the human-readable name for the call site;
    ``ddl`` carries the actual definition.
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
