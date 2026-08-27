"""Tests for the per-profile Projects store (hermes_cli/projects_db)."""

from __future__ import annotations

import multiprocessing
import os
import time
from pathlib import Path

import pytest

from hermes_cli import projects_db as pdb, sqlite_util

# "fork" keeps the already-imported, already-sandboxed interpreter state (the
# children only need the explicit db path); "spawn" is a correct fallback where
# fork is unavailable.
_MP = multiprocessing.get_context(
    "fork" if "fork" in multiprocessing.get_all_start_methods() else "spawn"
)

_INIT_LOCK_NAME = "projects.db.init.lock"


@pytest.fixture
def conn(tmp_path):
    c = pdb.connect(db_path=tmp_path / "projects.db")
    try:
        yield c
    finally:
        c.close()


def _run_projects_workers(procs, queue, expected_messages):
    """Start, drain and join child processes with bounded waits."""
    for proc in procs:
        proc.start()
    try:
        messages = [queue.get(timeout=120) for _ in range(expected_messages)]
    finally:
        for proc in procs:
            proc.join(timeout=120)
            if proc.is_alive():  # pragma: no cover - only on a real hang
                proc.terminate()
                proc.join(timeout=30)
    for proc in procs:
        assert proc.exitcode == 0, f"child exited {proc.exitcode}"
    return messages


def _first_open_worker(db_path, barrier, lock, active, peak, queue):
    """Race a fresh Projects DB open and report the widest observed overlap."""
    import hermes_state

    real_apply = hermes_state.apply_wal_with_fallback

    def observed_apply(connection, **kwargs):
        with lock:
            active.value += 1
            peak.value = max(peak.value, active.value)
        try:
            # Widen the first-open window so an unserialized implementation
            # deterministically overlaps another process's WAL setup.
            time.sleep(0.05)
            return real_apply(connection, **kwargs)
        finally:
            with lock:
                active.value -= 1

    hermes_state.apply_wal_with_fallback = observed_apply
    connection = None
    try:
        barrier.wait(timeout=60)
        connection = pdb.connect(db_path=Path(db_path))
        queue.put(bool(connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='projects'"
        ).fetchone()))
    except BaseException as exc:  # reported as a failure, never a silent hang
        queue.put(repr(exc))
    finally:
        if connection is not None:
            connection.close()


def test_concurrent_first_open_serializes_wal_and_schema_across_processes(tmp_path):
    """First open is single-writer HOST-wide, not just inside one process.

    Every child has its own empty ``_INITIALIZED_PATHS``, so each one believes
    it is the first opener — exactly the burst the file lock exists for, and a
    burst an in-process thread lock cannot order. The barrier releases them
    together and no attempt is retried.
    """
    db_path = tmp_path / "concurrent" / "projects.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    pdb._INITIALIZED_PATHS.discard(str(db_path.resolve()))

    workers = 6
    barrier = _MP.Barrier(workers)
    lock = _MP.Lock()
    active = _MP.Value("i", 0)
    peak = _MP.Value("i", 0)
    queue = _MP.Queue()
    procs = [
        _MP.Process(
            target=_first_open_worker,
            args=(str(db_path), barrier, lock, active, peak, queue),
        )
        for _ in range(workers)
    ]

    results = _run_projects_workers(procs, queue, workers)

    assert results == [True] * workers
    assert peak.value == 1
    # One stable sibling lock file; nothing per-run is left behind.
    assert {p.name for p in db_path.parent.iterdir()} <= {
        "projects.db", "projects.db-wal", "projects.db-shm", _INIT_LOCK_NAME,
    }


def _init_lock_holder_worker(db_path, acquired, release, queue):
    """Hold the Projects init lock through the production helper."""
    from hermes_cli.sqlite_util import cross_process_init_lock

    try:
        with cross_process_init_lock(Path(db_path)):
            acquired.set()
            release.wait(timeout=120)
        queue.put("released")
    except BaseException as exc:  # pragma: no cover - reported below
        queue.put(repr(exc))


def test_first_open_fails_closed_when_the_init_lock_cannot_be_taken(
    tmp_path, monkeypatch
):
    """Reaching the deadline is an error, not a licence to skip the lock.

    The Projects store holds the owner receipts every route change and plan is
    proven against, so a connection whose first open was never serialized is
    itself the failure. The wait stays bounded — this must not hang behind a
    wedged holder either.
    """
    db_path = tmp_path / "fenced" / "projects.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    resolved = str(db_path.resolve())
    pdb._INITIALIZED_PATHS.discard(resolved)
    monkeypatch.setattr(pdb, "_INIT_LOCK_TIMEOUT_SECONDS", 0.2)

    acquired, release, queue = _MP.Event(), _MP.Event(), _MP.Queue()
    holder = _MP.Process(
        target=_init_lock_holder_worker,
        args=(str(db_path), acquired, release, queue),
    )
    holder.start()
    try:
        assert acquired.wait(timeout=60)
        started = time.monotonic()
        with pytest.raises(sqlite_util.InitLockUnavailable):
            pdb.connect(db_path=db_path)
        waited = time.monotonic() - started
    finally:
        release.set()
        holder.join(timeout=60)
        if holder.is_alive():  # pragma: no cover - only on a real hang
            holder.terminate()
            holder.join(timeout=30)

    assert queue.get(timeout=60) == "released"
    assert holder.exitcode == 0
    # It gave up on its own deadline instead of waiting the holder out.
    assert waited < 30
    # Fail closed: nothing was initialized behind a lock it never held.
    assert resolved not in pdb._INITIALIZED_PATHS
    # One stable sibling lock file; nothing per-run is left behind.
    assert {p.name for p in db_path.parent.iterdir()} <= {
        "projects.db", _INIT_LOCK_NAME,
    }



def test_discovery_policy_change_clears_only_discovered_rows(conn):
    project_id = pdb.create_project(conn, name="Explicit", folders=["/www/explicit"])
    pdb.record_discovered_repos(
        conn, [("/www/scanned", "scanned")], policy_key="policy-a"
    )

    assert pdb.reconcile_discovered_repos_policy(conn, "policy-b") is True
    assert pdb.list_discovered_repos(conn) == []
    assert pdb.get_project(conn, project_id) is not None
    assert pdb.get_discovery_policy_key(conn) == "policy-b"






def test_create_get_list(conn):
    pid = pdb.create_project(conn, name="Hermes Agent", folders=["/tmp/hermes"])
    proj = pdb.get_project(conn, pid)

    assert proj is not None
    assert proj.slug == "hermes-agent"
    assert proj.name == "Hermes Agent"
    # First folder becomes primary.
    assert proj.primary_path == "/tmp/hermes"
    assert [f.path for f in proj.folders] == ["/tmp/hermes"]
    assert proj.folders[0].is_primary is True

    # Lookup by slug too.
    assert pdb.get_project(conn, "hermes-agent").id == pid
    assert len(pdb.list_projects(conn)) == 1












def test_project_for_path_skips_archived(conn):
    pid = pdb.create_project(conn, name="P", folders=["/www/app"])
    pdb.archive_project(conn, pid)

    assert pdb.project_for_path(conn, "/www/app/src") is None
    # Archived hidden from the default list but visible with include_archived.
    assert pdb.list_projects(conn) == []
    assert len(pdb.list_projects(conn, include_archived=True)) == 1

    pdb.restore_project(conn, pid)
    assert pdb.project_for_path(conn, "/www/app/src").id == pid


def test_create_dedups_by_primary_path(conn):
    pid = pdb.create_project(conn, name="GeoTrace", folders=["/www/geotrace"])

    # Same folder again (any name): refused, existing project named in error.
    with pytest.raises(ValueError, match="already belongs to project 'geotrace'"):
        pdb.create_project(conn, name="GeoTrace", folders=["/www/geotrace"])
    with pytest.raises(ValueError, match="already belongs"):
        pdb.create_project(conn, name="Other Name", primary_path="/www/geotrace")

    # Trailing-separator spelling of the same folder is still a duplicate.
    with pytest.raises(ValueError, match="already belongs"):
        pdb.create_project(conn, name="GeoTrace", primary_path="/www/geotrace/")

    # Deliberate duplicates stay possible.
    dup = pdb.create_project(
        conn, name="GeoTrace", folders=["/www/geotrace"], allow_duplicate_path=True
    )
    assert dup != pid
    assert len(pdb.list_projects(conn)) == 2


def test_create_dedup_ignores_archived_and_other_paths(conn):
    pid = pdb.create_project(conn, name="App", folders=["/www/app"])
    pdb.archive_project(conn, pid)

    # Archived project no longer blocks the path.
    fresh = pdb.create_project(conn, name="App", folders=["/www/app"])
    assert fresh != pid

    # Different folder is never a collision; folder-less projects don't match.
    pdb.create_project(conn, name="Elsewhere", folders=["/www/other"])
    pdb.create_project(conn, name="No Folder")


def test_find_by_primary_path(conn):
    pid = pdb.create_project(conn, name="App", folders=["/www/app"])

    assert pdb.find_by_primary_path(conn, "/www/app").id == pid
    assert pdb.find_by_primary_path(conn, "/www/app/").id == pid
    assert pdb.find_by_primary_path(conn, "/www/nope") is None
    assert pdb.find_by_primary_path(conn, "") is None






def test_per_profile_isolation(tmp_path):
    # Two distinct DB paths stand in for two profiles' HERMES_HOME.
    a = pdb.connect(db_path=tmp_path / "a" / "projects.db")
    b = pdb.connect(db_path=tmp_path / "b" / "projects.db")
    try:
        pdb.create_project(a, name="Only In A", folders=["/a"])
        pdb.record_discovered_repos(a, [("/a/scanned", "scanned")])

        assert [p.slug for p in pdb.list_projects(a)] == ["only-in-a"]
        assert pdb.list_projects(b) == []
        assert [row["root"] for row in pdb.list_discovered_repos(a)] == [
            "/a/scanned"
        ]
        assert pdb.list_discovered_repos(b) == []
    finally:
        a.close()
        b.close()

