"""Card 3: the acting half of the release host, run against stand-ins in temporary directories.

Each test drives ReleaseHostActions over things the test owns: a stand-in systemctl on PATH,
temporary git repositories, temporary homes and databases, and a local HTTP server. Nothing here
touches the real host, its units, its checkout or its homes.
"""

from __future__ import annotations

import contextlib
import http.server
import json
import os
import shutil
import socket
import sqlite3
import stat
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from hermes_cli import backup, build_info, release_host_actions
from hermes_cli.release_host_actions import ReleaseHostActions
from hermes_constants import mark_named_profile_deleted
from plugins.plugin_storage import plugin_db

# Stand-in unit names, so no call could reach a real unit even by mistake.
UNITS = {
    "gateway": "stand-in-gateway",
    "serve": "stand-in-serve",
    "sandbox-tunnel": "stand-in-tunnel",
}


def make_actions(tmp_path: Path, **values) -> ReleaseHostActions:
    settings = {
        "checkout": tmp_path / "checkout",
        "root_home": tmp_path / "root",
        "units": UNITS,
        "snapshot_root": tmp_path / "snapshots",
        "health_url": "",
        "workspace_check_url": "",
    }
    return ReleaseHostActions(**{**settings, **values})


# The stand-in user manager records every call. "reloading" exits 0 like the real is-active, but
# it is not the answer active.
STAND_IN_SYSTEMCTL = """#!/bin/sh
printf '%s\\n' "$*" >> "$STAND_IN_CALLS"
case "$*" in
  "--user is-active stand-in-gateway") echo active ;;
  "--user is-active stand-in-serve") echo reloading ;;
  "--user is-active "*) echo inactive; exit 3 ;;
  "--user start stand-in-tunnel") echo "Job for stand-in-tunnel.service failed." >&2; exit 1 ;;
esac
"""


def test_units_are_driven_only_through_the_user_manager(tmp_path, monkeypatch):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "systemctl").write_text(STAND_IN_SYSTEMCTL)
    (bin_dir / "systemctl").chmod(0o755)
    calls = tmp_path / "calls"
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("STAND_IN_CALLS", str(calls))

    def recorded() -> list[str]:
        return calls.read_text().splitlines() if calls.exists() else []

    actions = make_actions(tmp_path)
    actions.stop_units(["gateway", "serve"])
    actions.start_units(["serve", "gateway"])
    assert recorded() == [
        "--user stop stand-in-gateway",
        "--user stop stand-in-serve",
        "--user start stand-in-serve",
        "--user start stand-in-gateway",
    ]

    with pytest.raises(RuntimeError):
        actions.start_units(["sandbox-tunnel"])

    calls.unlink()
    assert [actions.unit_active(role) for role in UNITS] == [True, False, False]
    assert recorded() == [f"--user is-active {name}" for name in UNITS.values()]

    # A role without a unit name never reaches the manager, and nothing else is driven first.
    calls.unlink()
    unset = make_actions(tmp_path, units={**UNITS, "serve": ""})
    with pytest.raises(ValueError):
        unset.stop_units(["gateway", "serve"])
    assert unset.unit_active("serve") is False
    assert recorded() == []


def git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args], check=True, capture_output=True, text=True
    ).stdout.strip()


def git_succeeds(repo: Path, *args: str) -> bool:
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True).returncode == 0


def commit_text(repo: Path, text: str) -> str:
    (repo / "app.txt").write_text(text)
    git(repo, "add", "app.txt")
    git(repo, "commit", "-q", "-m", text.strip())
    return git(repo, "rev-parse", "HEAD")


@pytest.fixture
def repos(tmp_path, monkeypatch):
    """An origin with two commits on main, and a clean checkout cloned when it had only the first."""
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", os.devnull)
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    for who in ("AUTHOR", "COMMITTER"):
        monkeypatch.setenv(f"GIT_{who}_NAME", "Release Test")
        monkeypatch.setenv(f"GIT_{who}_EMAIL", "release-test@example.invalid")
    work, origin, checkout = tmp_path / "work", tmp_path / "origin.git", tmp_path / "checkout"
    work.mkdir()
    git(work, "init", "-q", "-b", "main")
    first = commit_text(work, "first\n")
    git(tmp_path, "clone", "-q", "--bare", str(work), str(origin))
    git(tmp_path, "clone", "-q", str(origin), str(checkout))
    second = commit_text(work, "second\n")
    git(work, "push", "-q", str(origin), "main")
    return SimpleNamespace(checkout=checkout, first=first, second=second)


def test_fetch_brings_the_commit_without_touching_the_working_tree(repos, tmp_path):
    actions = make_actions(tmp_path, checkout=repos.checkout)

    def head() -> tuple[str, str]:
        return git(repos.checkout, "symbolic-ref", "HEAD"), git(repos.checkout, "rev-parse", "HEAD")

    before = head()
    assert not git_succeeds(repos.checkout, "cat-file", "-e", repos.second)

    actions.fetch(repos.second)

    assert git(repos.checkout, "show", f"{repos.second}:app.txt") == "second"
    assert head() == before
    assert git(repos.checkout, "status", "--porcelain") == ""
    assert (repos.checkout / "app.txt").read_text() == "first\n"


def test_checkout_moves_a_clean_checkout_to_the_exact_commit(repos, tmp_path):
    git(repos.checkout, "fetch", "-q", "origin")
    actions = make_actions(tmp_path, checkout=repos.checkout)

    actions.checkout(repos.second)

    assert git(repos.checkout, "rev-parse", "HEAD") == repos.second
    assert not git_succeeds(repos.checkout, "symbolic-ref", "-q", "HEAD")  # detached
    assert git(repos.checkout, "status", "--porcelain") == ""
    assert (repos.checkout / "app.txt").read_text() == "second\n"

    # An unknown commit fails in git; a shortened name is refused before git runs. Neither moves.
    for commit, error in (("f" * 40, RuntimeError), (repos.first[:12], ValueError)):
        with pytest.raises(error):
            actions.checkout(commit)
        assert git(repos.checkout, "rev-parse", "HEAD") == repos.second


def test_state_snapshot_copies_every_database_consistently(tmp_path):
    root = tmp_path / "root"
    (root / "profiles" / "coder").mkdir(parents=True)
    configs = {
        "config.yaml": b"model: root\n",
        ".env": b"ROOT_KEY=kept\n",
        "profiles/coder/config.yaml": b"model: coder\n",
    }
    for rel, data in configs.items():
        (root / rel).write_bytes(data)
    writer = sqlite3.connect(root / "state.db")
    writer.execute("PRAGMA journal_mode=WAL")
    writer.execute("PRAGMA wal_autocheckpoint=0")
    writer.execute("CREATE TABLE notes (body TEXT)")
    writer.executemany("INSERT INTO notes VALUES (?)", [(f"note {n}",) for n in range(100)])
    writer.commit()
    actions = make_actions(tmp_path)
    try:
        assert (root / "state.db-wal").stat().st_size > 0  # the rows are still in the log
        actions.take_snapshot("older")
        actions.take_snapshot("newer")
    finally:
        writer.close()

    state = tmp_path / "snapshots" / "state"
    with contextlib.closing(sqlite3.connect(state / "older" / "state.db")) as copy:
        assert copy.execute("SELECT count(*) FROM notes").fetchone() == (100,)
    assert {rel: (state / "older" / rel).read_bytes() for rel in configs} == configs

    with pytest.raises(FileExistsError):
        actions.take_snapshot("newer")
    actions.delete_snapshot("older")
    actions.delete_snapshot("older")  # already gone: nothing to do
    with pytest.raises(ValueError):
        actions.delete_snapshot("..")
    assert sorted(path.name for path in state.iterdir()) == ["newer"]
    assert (state / "newer" / "state.db").is_file()


def wal_database(
    path: Path, rows: int, writer: sqlite3.Connection | None = None
) -> sqlite3.Connection:
    """A database in write-ahead-log mode whose rows stay in its log while the writer is open.
    The writer may be one that the store's own code opened."""
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = sqlite3.connect(path) if writer is None else writer
    writer.execute("PRAGMA journal_mode=WAL")
    writer.execute("PRAGMA wal_autocheckpoint=0")
    writer.execute("CREATE TABLE notes (body TEXT)")
    writer.executemany("INSERT INTO notes VALUES (?)", [(f"note {n}",) for n in range(rows)])
    writer.commit()
    return writer


def make_store(root: Path, rel: str, rows: int, monkeypatch) -> sqlite3.Connection:
    """The store at rel with its rows still in its log. A plugin's store is opened by plugin
    storage itself (plugins/plugin_storage.py), in the home that holds it."""
    home, _, plugin = rel.partition("plugin-data/")
    if not plugin:
        return wal_database(root / rel, rows)
    with monkeypatch.context() as patch:
        patch.setenv("HERMES_HOME", str(root / home))
        return wal_database(root / rel, rows, plugin_db(*plugin.split("/")))


def published(snapshot: Path) -> dict[str, bytes]:
    """Every file of a snapshot with its bytes, by its path in it. Read it before a copy is
    opened: opening one makes the copy's own sidecars."""
    files = filter(Path.is_file, snapshot.rglob("*"))
    return {path.relative_to(snapshot).as_posix(): path.read_bytes() for path in files}


# Every application store of a root with two boards and one served profile, where the release
# record (release_ledger.py), the board registry, its removal archive and the boards (kanban_db.py),
# the cron queues (cron/), the hosted rooms (gateway/hosted_rooms.py), the metrics store
# (observability/shared_metrics.py), plugins' stores under any name, including plugins named like
# task payloads (plugins/plugin_storage.py), and the profile's own state keep them.
STORES = (
    "state.db",
    "shared-state.db",
    "cron/deliveries.db",
    "cron/notepad.db",
    "kanban/release_ledger.db",
    "kanban/board_register.db",
    "kanban/board_removal_archive.db",
    "kanban/boards/sample/kanban.db",
    "kanban/boards/other/kanban.db",
    "telemetry/shared_metrics/metrics.sqlite3",
    "plugin-data/ordinary/data.db",
    "plugin-data/ordinary/facts.sqlite",
    "plugin-data/attachments/data.db",
    "plugin-data/workspaces/data.db",
    "profiles/coder/state.db",
    "profiles/coder/cron/delivery_records.db",
    "profiles/coder/telemetry/shared_metrics/metrics.sqlite3",
)
# Databases inside the root that are no application store: the tasks' workspaces and attachments
# of the default board and of another board (kanban_db.py), the checkout, the release snapshots,
# and backup.py's artifacts and infrastructure, at the root or in a plugin's directory.
NOT_STORES = (
    "kanban/workspaces/t_root/app.db",
    "kanban/attachments/t_root/upload.sqlite3",
    "kanban/boards/sample/workspaces/t_worker/app.db",
    "kanban/boards/sample/attachments/t_worker/upload.sqlite3",
    "work/hermes/tests/fixture.db",
    "release-snapshots/state/OLD/state.db",
    "state.db.retired-wal-1-2/state.db",
    "state.db.pre-update-emergency-1.bak",
    "plugin-data/ordinary/node_modules/cache.db",
)


def test_state_snapshot_reaches_every_store_from_a_worker_environment(tmp_path, monkeypatch):
    root = tmp_path / "root"
    writers = [
        make_store(root, rel, rows, monkeypatch) for rows, rel in enumerate(STORES, start=1)
    ]
    for rel in NOT_STORES:
        wal_database(root / rel, 1).close()
    # What the dispatcher gives a worker: its own board, task and workspaces, and a profile home.
    board = root / "kanban" / "boards" / "sample"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(board / "kanban.db"))
    monkeypatch.setenv("HERMES_KANBAN_BOARD", "sample")
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_worker")
    monkeypatch.setenv("HERMES_KANBAN_WORKSPACES_ROOT", str(board / "workspaces"))
    monkeypatch.setenv("HERMES_HOME", str(root / "profiles" / "coder"))
    try:
        assert all((root / f"{rel}-wal").stat().st_size > 0 for rel in STORES)
        make_actions(
            tmp_path, checkout=root / "work" / "hermes", snapshot_root=root / "release-snapshots"
        ).take_snapshot("NEW")
    finally:
        for writer in writers:
            writer.close()

    saved = root / "release-snapshots" / "state" / "NEW"
    # Against the independent count of the stores made above: no other file and no sidecar.
    assert sorted(published(saved)) == sorted(STORES)
    for rows, rel in enumerate(STORES, start=1):
        with contextlib.closing(sqlite3.connect(saved / rel)) as copy:
            assert copy.execute("SELECT count(*) FROM notes").fetchone() == (rows,), rel


# A plugin may take any name and give its store any file name (plugins/plugin_storage.py): names
# that backup.py gives its own directories and artifacts, and file names that end like a sidecar.
# The ordinary names, plugin-data/ordinary/data.db among them, are the controls.
PLUGIN_NAMES = (
    "backups", "state-snapshots", "checkpoints", "browser-profile", "browser-profiles",
    "node_modules", "site-packages", "venv", "state.db.retired-wal-capture",
    "ordinary", "attachments", "workspaces", "hermes-agent",
)
STORE_NAMES = ("audit-wal", "audit-shm", "audit-journal", "facts.sqlite", "facts.sqlite3", "store")


@pytest.mark.parametrize(
    "rel",
    [f"plugin-data/{name}/data.db" for name in PLUGIN_NAMES]
    + [f"plugin-data/ordinary/{name}" for name in STORE_NAMES],
)
def test_state_snapshot_keeps_a_plugin_store_under_any_name(tmp_path, monkeypatch, rel):
    root = tmp_path / "root"
    writer = make_store(root, rel, 3, monkeypatch)
    try:
        assert (root / f"{rel}-wal").stat().st_size > 0
        make_actions(tmp_path).take_snapshot("NEW")
    finally:
        writer.close()

    saved = tmp_path / "snapshots" / "state" / "NEW"
    assert sorted(published(saved)) == [rel]  # none of the store's own sidecars
    with contextlib.closing(sqlite3.connect(saved / rel)) as copy:
        assert copy.execute("SELECT count(*) FROM notes").fetchone() == (3,)


# A store whose image is damaged, by a database name: the metrics store of the root and of a
# profile (observability/shared_metrics.py) and a plugin's store. A .db name is the control.
@pytest.mark.parametrize(
    "rel",
    [
        "telemetry/shared_metrics/metrics.sqlite3",
        "profiles/coder/telemetry/shared_metrics/metrics.sqlite3",
        "plugin-data/ordinary/facts.sqlite",
        "plugin-data/ordinary/data.db",
    ],
)
def test_state_snapshot_is_not_published_without_a_damaged_store(tmp_path, monkeypatch, rel):
    root = tmp_path / "root"
    make_store(root, rel, 3, monkeypatch).close()
    (root / rel).write_bytes(bytes(4096))  # the header and the first page overwritten
    writer = wal_database(root / "state.db", 2)
    try:
        with pytest.raises(RuntimeError, match=f"could not copy {rel}$"):
            make_actions(tmp_path).take_snapshot("NEW")
    finally:
        writer.close()
    snapshots = tmp_path / "snapshots"
    assert list(snapshots.rglob("*")) == [snapshots / "state"]  # nothing published or staged


@pytest.mark.parametrize(
    "setting, name", [("snapshot_root", "release-snapshots"), ("checkout", "checkout")]
)
@pytest.mark.parametrize("container", ["kanban/boards/example", "pairing", "platforms/pairing"])
def test_state_snapshot_leaves_out_the_snapshots_and_the_checkout(
    tmp_path, container, setting, name
):
    """Also inside a directory that a quick snapshot copies whole (backup.py): a board's
    directory and both pairing directories."""
    root, place = tmp_path / "root", tmp_path / "root" / container / name
    wal_database(place / "fixtures" / "example.db", 1).close()
    (place / "placeholder.txt").write_text("placeholder\n")
    (root / "config.yaml").write_bytes(b"model: root\n")
    actions = make_actions(tmp_path, **{setting: place})
    writer = wal_database(root / "state.db", 2)
    try:
        actions.save_config_snapshot("CONFIG")
        actions.take_snapshot("OLD")
        actions.take_snapshot("NEW")
    finally:
        writer.close()

    state, live = actions.snapshot_root / "state", ["config.yaml", "state.db"]
    assert {snapshot: sorted(published(state / snapshot)) for snapshot in ("OLD", "NEW")} == {
        "OLD": live,
        "NEW": live,
    }


def test_state_snapshot_is_not_published_when_a_file_is_not_copied(tmp_path, monkeypatch):
    root = tmp_path / "root"
    root.mkdir()
    (root / "config.yaml").write_bytes(b"model: root\n")
    wal_database(root / "state.db", 1).close()
    real_copy2 = shutil.copy2

    def refuse_config(src, dst, **options):
        if Path(src).name == "config.yaml":
            raise PermissionError(13, "Permission denied", str(src))
        return real_copy2(src, dst, **options)

    monkeypatch.setattr(shutil, "copy2", refuse_config)
    with pytest.raises(RuntimeError, match="config.yaml"):
        make_actions(tmp_path).take_snapshot("NEW")
    assert not (tmp_path / "snapshots" / "state" / "NEW").exists()


def test_state_snapshot_is_not_published_when_a_directory_cannot_be_listed(tmp_path, monkeypatch):
    state, native = tmp_path / "snapshots" / "state", tmp_path / "root" / "plugin-data" / "native"
    wal_database(native / "data.db", 3).close()
    actions = make_actions(tmp_path)
    actions.take_snapshot("OLD")  # the control: listed, the subtree's database is saved
    earlier = published(state / "OLD")
    assert list(earlier) == ["plugin-data/native/data.db"]
    denied, real_scandir = os.path.realpath(native), os.scandir

    def scandir(path="."):
        if not isinstance(path, int) and os.path.realpath(path) == denied:
            raise PermissionError(13, "Permission denied", os.fspath(path))
        return real_scandir(path)

    monkeypatch.setattr(os, "scandir", scandir)
    with pytest.raises(PermissionError, match="native"):
        actions.take_snapshot("NEW")
    assert not (state / "NEW").exists()
    assert published(state / "OLD") == earlier


def test_state_snapshot_never_pairs_a_safe_copy_with_later_sidecars(tmp_path, monkeypatch):
    board = tmp_path / "root" / "kanban" / "boards" / "example"
    board.mkdir(parents=True)
    live = board / "kanban.db"
    # A real WAL writer, the tables x and y on pages of their own: (1,1), then (2,2) at once.
    writer = sqlite3.connect(live, isolation_level=None)
    writer.executescript(
        "PRAGMA journal_mode=WAL; PRAGMA wal_autocheckpoint=0; CREATE TABLE x (v); CREATE TABLE"
        " y (v); BEGIN; INSERT INTO x VALUES (1); INSERT INTO y VALUES (1); COMMIT;"
        " BEGIN; UPDATE x SET v = 2; UPDATE y SET v = 2; COMMIT;"
    )
    notes = wal_database(board / "notes.sqlite3", 5)  # another database name, rows in its log
    real_copy, moved = backup._safe_copy_db, []

    def copy_then_write(src, dst, **options):
        done = real_copy(src, dst, **options)
        if not moved and Path(src).samefile(live):  # right after the safe copy of the board
            moved.append(src)
            writer.executescript(
                "BEGIN; UPDATE x SET v = 1; UPDATE y SET v = 1; COMMIT;"
                " PRAGMA wal_checkpoint(TRUNCATE); UPDATE x SET v = 3;"
            )
        return done

    for module in (release_host_actions, backup):
        monkeypatch.setattr(module, "_safe_copy_db", copy_then_write)
    try:
        make_actions(tmp_path).take_snapshot("NEW")
    finally:
        writer.close()
        notes.close()

    def read(path: Path) -> tuple:
        with contextlib.closing(sqlite3.connect(path)) as db:
            return db.execute("SELECT (SELECT v FROM x), (SELECT v FROM y)").fetchone()

    saved = tmp_path / "snapshots" / "state" / "NEW" / "kanban" / "boards" / "example"
    names = sorted(published(saved))  # before a copy is opened
    with contextlib.closing(sqlite3.connect(saved / "notes.sqlite3")) as copy:
        rows = copy.execute("SELECT count(*) FROM notes").fetchone()
    assert moved
    # The safe image alone, as it was copied, with every row of the other database's log.
    assert (names, read(saved / "kanban.db"), rows, read(live)) == (
        ["kanban.db", "notes.sqlite3"], (2, 2), (5,), (3, 1)
    )


@contextlib.contextmanager
def umask(mask: int):
    previous = os.umask(mask)
    try:
        yield
    finally:
        os.umask(previous)


def snapshot_beside_a_public_file(public: Path) -> Path:
    """Under umask 022, snapshot a private home into an open snapshot root that also holds a
    public placeholder file. Returns the state snapshot directory."""
    root, snapshots = public / "root", public / "snapshots"
    with umask(0o022):
        root.mkdir(mode=0o700)
        wal_database(root / "state.db", 1).close()
        (root / "state.db").chmod(0o600)
        snapshots.mkdir()
        (snapshots / "placeholder.txt").write_text("public placeholder\n")
        make_actions(public).take_snapshot("NEW")
    return snapshots / "state"


def open_to_others(state: Path) -> dict[str, str]:
    """Each snapshot directory or copy that its group or other users may reach, with its mode."""
    return {
        path.relative_to(state.parent).as_posix(): oct(stat.S_IMODE(path.stat().st_mode))
        for path in [state, *state.rglob("*")]
        if path.stat().st_mode & 0o077
    }


def test_state_snapshot_stays_private_under_an_open_umask_and_root(tmp_path):
    state = snapshot_beside_a_public_file(tmp_path)

    assert (state / "NEW" / "state.db").is_file()
    assert open_to_others(state) == {}


def nobody() -> tuple[int, int] | None:
    """The unprivileged user nobody and its group, when this process may switch to them."""
    try:
        import pwd

        entry = pwd.getpwnam("nobody")
    except (ImportError, KeyError):
        return None
    return (entry.pw_uid, entry.pw_gid) if os.geteuid() == 0 else None


@pytest.mark.skipif(nobody() is None, reason="needs a process that may switch to the user nobody")
def test_another_user_cannot_read_a_state_snapshot():
    uid, gid = nobody()

    def reads(path: Path) -> bool:
        attempt = subprocess.run(
            ["cat", str(path)], user=uid, group=gid, extra_groups=[], cwd="/", capture_output=True
        )
        return attempt.returncode == 0

    # Pytest's own temporary root may be closed to other users, so the public parent is made here.
    with tempfile.TemporaryDirectory(prefix="release-snapshot-") as name:
        public = Path(name)
        public.chmod(0o755)
        state = snapshot_beside_a_public_file(public)
        assert reads(public / "snapshots" / "placeholder.txt")  # the control: nobody gets this far
        assert not reads(public / "root" / "state.db")
        copies = sorted(state.rglob("*.db"))
        assert [copy.name for copy in copies] == ["state.db"]
        assert [copy.name for copy in copies if reads(copy)] == []
        assert open_to_others(state) == {}


def test_config_snapshot_round_trip_restores_exact_bytes(tmp_path):
    root = tmp_path / "root"
    saved = {
        "config.yaml": b"model: root\n",
        ".env": b"ROOT_KEY=saved\n",
        "profiles/coder/config.yaml": b"model: coder\r\n\x00\xff",
        "profiles/coder/.env": b"CODER_KEY=saved\n",
    }
    for rel, data in saved.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_bytes(data)
    # A deleted profile is not served, so its configuration is not part of the set.
    gone = root / "profiles" / "gone"
    gone.mkdir()
    (gone / "config.yaml").write_bytes(b"model: gone\n")
    mark_named_profile_deleted(gone)
    actions = make_actions(tmp_path)
    with pytest.raises(RuntimeError):
        actions.restore_config()  # nothing saved yet

    actions.save_config_snapshot("release")
    for rel in [*saved, "profiles/gone/config.yaml"]:
        (root / rel).write_bytes(b"changed\n")
    actions.restore_config()

    assert {rel: (root / rel).read_bytes() for rel in saved} == saved
    assert (gone / "config.yaml").read_bytes() == b"changed\n"


def test_config_restore_removes_files_created_after_the_save(tmp_path):
    root = tmp_path / "root"
    reviewer = root / "profiles" / "reviewer"
    reviewer.mkdir(parents=True)
    (root / "config.yaml").write_bytes(b"model: root\n")
    (reviewer / "config.yaml").write_bytes(b"model: reviewer\n")
    actions = make_actions(tmp_path)
    actions.save_config_snapshot("release")
    (root / ".env").write_bytes(b"CREATED=after the save\n")
    # A profile deleted after the save is no longer served: its files stay as they are.
    mark_named_profile_deleted(reviewer)
    for name in ("config.yaml", ".env"):
        (reviewer / name).write_bytes(b"changed\n")
    actions.restore_config()

    assert not (root / ".env").exists()
    assert (root / "config.yaml").read_bytes() == b"model: root\n"
    assert [(reviewer / name).read_bytes() for name in ("config.yaml", ".env")] == [b"changed\n"] * 2


def test_config_restore_recreates_a_removed_profile_directory(tmp_path):
    root = tmp_path / "root"
    coder = root / "profiles" / "coder"
    coder.mkdir(parents=True)
    saved = {"config.yaml": b"model: root\n", "profiles/coder/config.yaml": b"model: coder\n"}
    for rel, data in saved.items():
        (root / rel).write_bytes(data)
    actions = make_actions(tmp_path)
    actions.save_config_snapshot("release")
    shutil.rmtree(coder)
    actions.restore_config()

    assert {rel: (root / rel).read_bytes() for rel in saved} == saved
    assert stat.S_IMODE(coder.stat().st_mode) == 0o700


def test_a_failed_config_save_keeps_the_earlier_save(tmp_path, monkeypatch):
    root = tmp_path / "root"
    root.mkdir()
    (root / "config.yaml").write_bytes(b"model: saved\n")
    actions = make_actions(tmp_path)
    actions.save_config_snapshot("NEW")
    (root / "config.yaml").write_bytes(b"model: changed\n")
    saved, real_replace, refused = tmp_path / "snapshots" / "config" / "NEW", os.replace, []

    def publication_fails(src, dst):
        if Path(dst) == saved and not refused:  # only the publication of the second save
            refused.append(src)
            raise PermissionError(13, "Permission denied", str(dst))
        real_replace(src, dst)

    with monkeypatch.context() as patch:
        patch.setattr(os, "replace", publication_fails)
        with pytest.raises(PermissionError):
            actions.save_config_snapshot("NEW")
    assert refused
    assert (saved / "config.yaml").is_file(), "the earlier save is gone"
    assert (saved / "config.yaml").read_bytes() == b"model: saved\n"
    assert sorted(path.name for path in saved.parent.iterdir()) == ["NEW"]
    actions.restore_config()
    assert (root / "config.yaml").read_bytes() == b"model: saved\n"


SLOW_SECONDS = 2.5


class Answers(http.server.BaseHTTPRequestHandler):
    """/ok and /login answer 200, /check redirects to /login, /slow answers 200 only after the
    readback gave up, anything else 503."""

    def do_GET(self) -> None:
        if self.path == "/slow":
            time.sleep(SLOW_SECONDS)
        with contextlib.suppress(OSError):  # the reader may be gone by now
            if self.path == "/check":
                self.send_response(302)
                self.send_header("Location", "/login")
            else:
                self.send_response(200 if self.path in ("/ok", "/slow", "/login") else 503)
            self.end_headers()

    def log_message(self, *args) -> None:
        pass


@pytest.fixture
def server():
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Answers)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()
    httpd.server_close()


def refused_url() -> str:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    return f"http://127.0.0.1:{port}/ok"


def test_health_and_owner_page_readbacks_fail_closed(tmp_path, server, monkeypatch):
    monkeypatch.setattr(release_host_actions, "READBACK_TIMEOUT_SECONDS", SLOW_SECONDS / 5)

    def readbacks(url: str) -> tuple[bool, bool] | str:
        actions = make_actions(tmp_path, health_url=url, workspace_check_url=url)
        try:
            return actions.health_ok(), actions.workspace_reads_ok()
        except Exception as error:  # a readback never raises
            return repr(error)

    assert readbacks(f"{server}/ok") == (True, True)
    # So do a redirect to a login page that answers 200 and an address that is not HTTP.
    closed = (
        f"{server}/broken",
        f"{server}/slow",
        refused_url(),
        "",
        f"{server}/check",
        "data:text/plain,placeholder",
    )
    assert {url: readbacks(url) for url in closed} == dict.fromkeys(closed, (False, False))


def test_fleet_version_is_one_version_or_none(tmp_path, monkeypatch):
    root = tmp_path / "root"
    coder = root / "profiles" / "coder"
    coder.mkdir(parents=True)
    ended = subprocess.Popen([sys.executable, "-c", ""])
    ended.wait()
    live, new, old = os.getpid(), "1" * 40, "2" * 40
    # collect_fleet_versions reads the gateways under this process's root and compares each stamp
    # with the checkout's own code identity, which is NEW here.
    monkeypatch.setenv("HERMES_HOME", str(root))
    monkeypatch.setattr(build_info, "get_code_identity", lambda refresh=False: {"sha": new})
    cases = {
        "no gateway": [],
        "one live gateway stamped NEW": [(root, live, new)],
        "every gateway stamped NEW": [(root, live, new), (coder, live, new)],
        "mixed versions": [(root, live, new), (coder, live, old)],
        "a stale stamp": [(root, live, old)],
        "an unknown stamp": [(root, live, None)],
        "no live gateway": [(root, ended.pid, new)],
        "a number as the identity": [(root, live, 17)],
        "true as the identity": [(root, live, True)],
    }

    def answer(stamps, root_home: Path = root) -> str:
        for home in (root, coder):
            (home / "gateway_state.json").unlink(missing_ok=True)
        for home, pid, code_sha in stamps:
            record = {"pid": pid, "gateway_state": "running"}
            if code_sha is not None:
                record["code_sha"] = code_sha
            (home / "gateway_state.json").write_text(json.dumps(record))
        return make_actions(tmp_path, root_home=root_home).fleet_version()

    answers = {case: answer(stamps) for case, stamps in cases.items()}
    good = ("one live gateway stamped NEW", "every gateway stamped NEW")
    assert answers == {**dict.fromkeys(cases, ""), **dict.fromkeys(good, new)}
    # The gateways under this process's root are not another host root's answer.
    assert answer([(root, live, new)], root_home=tmp_path / "elsewhere") == ""
