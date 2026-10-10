"""Card 3: the acting half of the release host, run against stand-ins in temporary directories.

Each test drives ReleaseHostActions over things the test owns: a stand-in systemctl and a stand-in
docker on PATH, temporary git repositories, temporary homes and databases, and a local HTTP
server. Nothing here touches the real host, its units, its containers, its checkout or its homes.
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
        "workspace_container": "stand-in-workspace",
    }
    return ReleaseHostActions(**{**settings, **values})


# The stand-in user manager records every call. Only the unit named in STAND_IN_ACTIVE is active,
# so by default every unit is stopped. "reloading" exits 0 like the real is-active, but it is not
# the answer active.
STAND_IN_SYSTEMCTL = """#!/bin/sh
printf '%s\\n' "$*" >> "$STAND_IN_CALLS"
case "$*" in
  "--user is-active $STAND_IN_ACTIVE") echo active ;;
  "--user is-active $STAND_IN_RELOADING") echo reloading ;;
  "--user is-active "*) echo inactive; exit 3 ;;
  "--user start stand-in-tunnel") echo "Job for stand-in-tunnel.service failed." >&2; exit 1 ;;
esac
"""

# The stand-in docker records its arguments, each ended by a NUL, and runs the read as
# STAND_IN_READ says. All it writes, the access file's content and response content among it, must
# stay out of the journal: ok reads the projects list and the board, refused is the platform
# turning the access file away, fails is a board read that breaks after the projects list, and
# late answers only after five seconds.
STAND_IN_DOCKER = """#!/bin/sh
printf '%s\\0' "$@" >> "$STAND_IN_DOCKER_ARGS"
echo 'stand-in access file content' >&2
case "$STAND_IN_READ" in
  ok) echo '{"projects": [{"slug": "stand-in-board"}]}'; echo '{"columns": []}' ;;
  refused) echo '401 {"detail": "Unauthorized"}'; exit 1 ;;
  fails) echo '{"projects": [{"slug": "stand-in-board"}]}'; echo 'board: 500' >&2; exit 1 ;;
  late) echo '{"projects": [{"slug": "stand-in-board"}]}'; exec sleep 5 ;;
esac
"""


@pytest.fixture(autouse=True)
def calls(tmp_path, monkeypatch) -> Path:
    """Every test runs with the stand-in user manager and the stand-in docker first on PATH, so
    none can reach a real manager, unit or container: a state snapshot asks the manager first.
    Returns the file of the manager's recorded calls."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "systemctl").write_text(STAND_IN_SYSTEMCTL)
    (bin_dir / "systemctl").chmod(0o755)
    (bin_dir / "docker").write_text(STAND_IN_DOCKER)
    (bin_dir / "docker").chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("STAND_IN_CALLS", str(tmp_path / "calls"))
    monkeypatch.setenv("STAND_IN_DOCKER_ARGS", str(tmp_path / "docker-args"))
    return tmp_path / "calls"


def test_units_are_driven_only_through_the_user_manager(tmp_path, monkeypatch, calls):
    monkeypatch.setenv("STAND_IN_ACTIVE", "stand-in-gateway")
    monkeypatch.setenv("STAND_IN_RELOADING", "stand-in-serve")

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


def test_checkout_refuses_to_overwrite_an_ignored_local_file(repos, tmp_path):
    # An ignored file is not in git status, so a clean checkout can still hold one. When the new
    # commit tracks that file, the move must refuse instead of replacing the file's bytes.
    work = tmp_path / "work"
    (work / ".gitignore").write_text("local.txt\n")
    git(work, "add", ".gitignore")
    git(work, "commit", "-q", "-m", "ignore local.txt")
    prev = git(work, "rev-parse", "HEAD")
    (work / "local.txt").write_text("release\n")
    git(work, "add", "-f", "local.txt")
    git(work, "commit", "-q", "-m", "track local.txt")
    new = git(work, "rev-parse", "HEAD")
    git(work, "push", "-q", str(tmp_path / "origin.git"), "main")
    git(repos.checkout, "fetch", "-q", "origin")
    git(repos.checkout, "checkout", "-q", "--detach", prev)
    (repos.checkout / "local.txt").write_text("local\n")
    assert git(repos.checkout, "status", "--porcelain") == ""
    actions = make_actions(tmp_path, checkout=repos.checkout)

    with pytest.raises(RuntimeError):
        actions.checkout(new)

    assert git(repos.checkout, "rev-parse", "HEAD") == prev
    assert (repos.checkout / "local.txt").read_text() == "local\n"


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
    wal_database(root / "state.db", 100)
    assert (root / "state.db-wal").stat().st_size > 0  # the rows are still in the log
    actions = make_actions(tmp_path)
    actions.take_snapshot("older")
    actions.take_snapshot("newer")

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


# A writer of its own process that commits its rows and stops without closing, as a stopped unit
# leaves its store: no transaction is open anywhere, and the rows stay in the log beside it.
STOPPED_WRITER = """
import os, sqlite3, sys
from plugins.plugin_storage import plugin_db
path, rows, plugin = sys.argv[1], int(sys.argv[2]), sys.argv[3:]
writer = plugin_db(*plugin) if plugin else sqlite3.connect(path)
writer.execute("PRAGMA journal_mode=WAL")
writer.execute("PRAGMA wal_autocheckpoint=0")
writer.execute("CREATE TABLE notes (body TEXT)")
writer.executemany("INSERT INTO notes VALUES (?)", [(f"note {n}",) for n in range(rows)])
writer.commit()
os._exit(0)
"""


def wal_database(path: Path, rows: int, *plugin: str, home: Path | None = None) -> None:
    """A database in write-ahead-log mode whose stopped writer left its rows in the log. The
    writer may be the store's own code, plugin storage, in the home that holds the store."""
    path.parent.mkdir(parents=True, exist_ok=True)
    env = None if home is None else {**os.environ, "HERMES_HOME": str(home)}
    writer = [sys.executable, "-c", STOPPED_WRITER, str(path), str(rows), *plugin]
    subprocess.run(writer, check=True, cwd=Path(__file__).resolve().parents[2], env=env)


def make_store(root: Path, rel: str, rows: int) -> None:
    """The store at rel with its rows still in its log. A plugin's store in its own directory is
    opened by plugin storage itself (plugins/plugin_storage.py), in the home that holds it."""
    home, _, plugin = rel.partition("plugin-data/")
    if plugin.count("/") != 1:
        return wal_database(root / rel, rows)
    return wal_database(root / rel, rows, *plugin.split("/"), home=root / home)


def published(snapshot: Path) -> dict[str, bytes]:
    """Every file of a snapshot with its bytes, by its path in it. Read it before a copy is
    opened: opening one makes the copy's own sidecars."""
    files = filter(Path.is_file, snapshot.rglob("*"))
    return {path.relative_to(snapshot).as_posix(): path.read_bytes() for path in files}


def walked(root: Path, snapshot: Path) -> list[str]:
    """The full backup's own list for the root (backup.py), the snapshot directory as out_path."""
    return sorted(rel.as_posix() for _, rel in backup._iter_backup_files(root, snapshot))


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
# Plain files beside them: the configuration of the root and of the profile, and a log.
PLAIN_FILES = ("config.yaml", ".env", "profiles/coder/config.yaml", "logs/agent.log")


def test_state_snapshot_copies_the_walks_list_from_a_worker_environment(tmp_path, monkeypatch):
    root = tmp_path / "root"
    for rows, rel in enumerate(STORES, start=1):
        make_store(root, rel, rows)
    for rel in PLAIN_FILES:
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text(f"{rel}\n")
    # What the dispatcher gives a worker: its own board, task and workspaces, and a profile home.
    board = root / "kanban" / "boards" / "sample"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(board / "kanban.db"))
    monkeypatch.setenv("HERMES_KANBAN_BOARD", "sample")
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_worker")
    monkeypatch.setenv("HERMES_KANBAN_WORKSPACES_ROOT", str(board / "workspaces"))
    monkeypatch.setenv("HERMES_HOME", str(root / "profiles" / "coder"))
    saved = tmp_path / "snapshots" / "state" / "NEW"
    expected = walked(root, saved)
    assert all((root / f"{rel}-wal").stat().st_size > 0 for rel in STORES)
    make_actions(tmp_path).take_snapshot("NEW")

    # Against the independent count: the walk's own list, which holds every store and file made.
    assert set(STORES) | set(PLAIN_FILES) <= set(expected)
    files = published(saved)
    assert sorted(files) == expected
    assert [files[rel] for rel in PLAIN_FILES] == [f"{rel}\n".encode() for rel in PLAIN_FILES]
    for rows, rel in enumerate(STORES, start=1):
        with contextlib.closing(sqlite3.connect(saved / rel)) as copy:
            assert copy.execute("SELECT count(*) FROM notes").fetchone() == (rows,), rel


# Names that start like backup.py's artifacts, in a plugin's directory of the root and of a profile:
# stores inside directories by such names and a store file by one, which the walk yields, and last
# the control, a file name the walk leaves out.
ARTIFACT_LIKE = (
    "state.db.retired-wal-1-2/data.db",
    "state.db.pre-update-emergency-1/data.db",
    "state.db.pre-update-emergency.db",
    "state.db.pre-update-emergency-1.bak",
)


@pytest.mark.parametrize(
    "place", ["plugin-data/ordinary/", "profiles/coder/plugin-data/ordinary/"]
)
def test_state_snapshot_holds_a_store_named_like_an_artifact_when_the_walk_yields_it(
    tmp_path, place
):
    root, rels = tmp_path / "root", [place + name for name in ARTIFACT_LIKE]
    for rows, rel in enumerate(rels, start=1):
        make_store(root, rel, rows)
    saved = tmp_path / "snapshots" / "state" / "NEW"
    expected = walked(root, saved)
    assert all((root / f"{rel}-wal").stat().st_size > 0 for rel in rels)
    make_actions(tmp_path).take_snapshot("NEW")

    assert sorted(published(saved)) == expected
    assert [rel in expected for rel in rels] == [True, True, True, False]
    for rows, rel in enumerate(rels[:-1], start=1):
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
def test_state_snapshot_holds_a_plugin_store_exactly_when_the_walk_yields_it(
    tmp_path, rel
):
    root = tmp_path / "root"
    make_store(root, rel, 3)
    saved = tmp_path / "snapshots" / "state" / "NEW"
    expected = walked(root, saved)
    assert (root / f"{rel}-wal").stat().st_size > 0
    make_actions(tmp_path).take_snapshot("NEW")

    files = published(saved)
    assert sorted(files) == expected  # a store not named .db with its own log and index
    if f"{rel}-wal" in expected:  # the log its stopped writer left, as it is beside its image
        assert files[f"{rel}-wal"] == (root / f"{rel}-wal").read_bytes()
    if rel in expected:
        with contextlib.closing(sqlite3.connect(saved / rel)) as copy:
            assert copy.execute("SELECT count(*) FROM notes").fetchone() == (3,)


# A store whose image is damaged: the metrics store of the root and of a profile
# (observability/shared_metrics.py) and a plugin's stores, under any name.
@pytest.mark.parametrize(
    "rel",
    [
        "telemetry/shared_metrics/metrics.sqlite3",
        "profiles/coder/telemetry/shared_metrics/metrics.sqlite3",
        "plugin-data/ordinary/facts.sqlite",
        "plugin-data/ordinary/data.db",
    ],
)
def test_state_snapshot_refuses_a_damaged_database_and_copies_a_file_without_the_header(
    tmp_path, rel
):
    root, snapshots = tmp_path / "root", tmp_path / "snapshots"
    (root / rel).parent.mkdir(parents=True)  # no log beside it, which could hold its pages
    header = backup._SQLITE_HEADER
    (root / rel).write_bytes(header + bytes(4096 - len(header)))  # the header, then no database
    wal_database(root / "state.db", 2)
    actions = make_actions(tmp_path)
    with pytest.raises(RuntimeError, match=f"could not copy {rel}$"):
        actions.take_snapshot("NEW")
    assert list(snapshots.rglob("*")) == [snapshots / "state"]  # nothing published or staged
    (root / rel).write_bytes(bytes(4096))  # the header overwritten too: a file like any other
    actions.take_snapshot("NEW")
    assert published(snapshots / "state" / "NEW")[rel] == bytes(4096)


def test_state_snapshot_inside_the_root_copies_the_walks_list_from_before_its_first_copy(tmp_path):
    root = tmp_path / "root"
    wal_database(root / "state.db", 2)
    (root / "config.yaml").write_bytes(b"model: root\n")
    assert "kept-snapshots" not in backup._EXCLUDED_DIRS  # else the walk never enters it
    actions = make_actions(tmp_path, snapshot_root=root / "kept-snapshots")
    state = actions.snapshot_root / "state"
    # An earlier release snapshot in the root, which the walk must skip.
    make_actions(tmp_path, snapshot_root=root / "release-snapshots").take_snapshot("EARLIER")
    actions.save_config_snapshot("CONFIG")
    actions.take_snapshot("OLD")
    expected = walked(root, state / "NEW")
    actions.take_snapshot("NEW")

    files = sorted(published(state / "NEW"))
    assert files == expected
    assert "state.db" in files
    assert [rel for rel in files if "release-snapshots" in Path(rel).parts] == []
    # Nothing of its own staging or copy: the walk's list was taken before its first copy.
    own = ("kept-snapshots/.state-NEW", "kept-snapshots/state/NEW/")
    assert [rel for rel in files if rel.startswith(own)] == []


def test_state_snapshot_is_not_published_when_a_file_is_not_copied(tmp_path, monkeypatch):
    root = tmp_path / "root"
    root.mkdir()
    (root / "config.yaml").write_bytes(b"model: root\n")
    wal_database(root / "state.db", 1)
    real_copy2 = shutil.copy2

    def refuse_config(src, dst, **options):
        if Path(src).name == "config.yaml":
            raise PermissionError(13, "Permission denied", str(src))
        return real_copy2(src, dst, **options)

    monkeypatch.setattr(shutil, "copy2", refuse_config)
    with pytest.raises(RuntimeError, match="config.yaml"):
        make_actions(tmp_path).take_snapshot("NEW")
    assert not (tmp_path / "snapshots" / "state" / "NEW").exists()


def test_state_snapshot_of_a_missing_root_home_raises_and_publishes_nothing(tmp_path):
    snapshots = tmp_path / "snapshots"
    wal_database(tmp_path / "root" / "state.db", 1)
    actions = make_actions(tmp_path)
    actions.take_snapshot("OLD")
    shutil.rmtree(tmp_path / "root")  # the walk would yield nothing: an empty snapshot
    with pytest.raises(OSError):
        actions.take_snapshot("NEW")
    assert [path.name for path in snapshots.iterdir()] == ["state"]  # nothing staged either
    assert [path.name for path in (snapshots / "state").iterdir()] == ["OLD"]


@pytest.mark.parametrize("role", ["gateway", "serve"])
def test_state_snapshot_refuses_while_a_platform_unit_is_active(tmp_path, monkeypatch, role):
    root, snapshots = tmp_path / "root", tmp_path / "snapshots"
    wal_database(root / "state.db", 2)
    (root / "config.yaml").write_bytes(b"model: root\n")
    actions = make_actions(tmp_path)
    monkeypatch.setenv("STAND_IN_ACTIVE", UNITS[role])
    with pytest.raises(RuntimeError, match=f"the {role} unit is active"):
        actions.take_snapshot("NEW")
    assert not snapshots.exists()  # nothing copied, staged or even made

    # The sandbox tunnel is no platform unit: the release keeps it up.
    monkeypatch.setenv("STAND_IN_ACTIVE", UNITS["sandbox-tunnel"])
    expected = walked(root, snapshots / "state" / "NEW")
    actions.take_snapshot("NEW")
    assert sorted(published(snapshots / "state" / "NEW")) == expected == ["config.yaml", "state.db"]


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
        wal_database(root / "state.db", 1)
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


# The configuration set (S-D) of a root home with one profile home.
CONFIG_SET = ("config.yaml", ".env", "profiles/coder/config.yaml", "profiles/coder/.env")


def saved_then_changed(tmp_path: Path) -> tuple[ReleaseHostActions, Path, dict[str, bytes]]:
    """Save the configuration set, then change each of its files live, so that a restore shows in
    every one. Returns the actions, the save, and the live files with their changed bytes."""
    root = tmp_path / "root"
    for rel in CONFIG_SET:
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_bytes(f"saved {rel}\n".encode())
    actions = make_actions(tmp_path)
    actions.save_config_snapshot("release")
    changed = {rel: f"changed {rel}\n".encode() for rel in CONFIG_SET}
    for rel, data in changed.items():
        (root / rel).write_bytes(data)
    return actions, tmp_path / "snapshots" / "config" / "release", changed


# A save that does not hold the whole configuration set would remove the live files it lacks, so
# the restore refuses it before any live file changes: each one keeps its exact bytes.
def test_config_restore_refuses_an_empty_save_and_keeps_every_live_file(tmp_path):
    actions, saved, changed = saved_then_changed(tmp_path)
    shutil.rmtree(saved)
    saved.mkdir()
    with pytest.raises(FileNotFoundError, match="config.yaml"):
        actions.restore_config()
    assert published(tmp_path / "root") == changed


def test_config_restore_refuses_a_save_of_unrelated_files_and_keeps_every_live_file(tmp_path):
    actions, saved, changed = saved_then_changed(tmp_path)
    shutil.rmtree(saved)
    for rel in ("notes.txt", "state.db", "profiles/coder/memories/MEMORY.md"):
        (saved / rel).parent.mkdir(parents=True, exist_ok=True)
        (saved / rel).write_bytes(b"not configuration\n")
    with pytest.raises(ValueError, match="no configuration file"):
        actions.restore_config()
    assert published(tmp_path / "root") == changed


def test_config_restore_refuses_a_save_with_an_unreadable_file_and_keeps_every_live_file(
    tmp_path, monkeypatch
):
    actions, saved, changed = saved_then_changed(tmp_path)
    unreadable, read_bytes = saved / "profiles" / "coder" / "config.yaml", Path.read_bytes
    unreadable.chmod(0)  # sorted last: a restore that read as it went would change the rest first

    def read_or_deny(path):  # root reads any file: the denial is injected here
        if path == unreadable:
            raise PermissionError(13, "Permission denied", str(path))
        return read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", read_or_deny)
    with pytest.raises(PermissionError):
        actions.restore_config()
    assert published(tmp_path / "root") == changed


def test_config_restore_of_a_complete_save_puts_back_every_saved_file(tmp_path):
    actions, saved, changed = saved_then_changed(tmp_path)
    assert sorted(published(saved)) == sorted(changed)
    actions.restore_config()
    assert published(tmp_path / "root") == {rel: f"saved {rel}\n".encode() for rel in CONFIG_SET}


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


def test_health_readback_fails_closed(tmp_path, server, monkeypatch):
    monkeypatch.setattr(release_host_actions, "READBACK_TIMEOUT_SECONDS", SLOW_SECONDS / 5)

    def readback(url: str) -> bool | str:
        actions = make_actions(tmp_path, health_url=url)
        try:
            return actions.health_ok()
        except Exception as error:  # a readback never raises
            return repr(error)

    assert readback(f"{server}/ok") is True
    # So do a redirect to a login page that answers 200 and an address that is not HTTP.
    closed = (
        f"{server}/broken",
        f"{server}/slow",
        refused_url(),
        "",
        f"{server}/check",
        "data:text/plain,placeholder",
    )
    assert {url: readback(url) for url in closed} == dict.fromkeys(closed, False)


# R6 runs its two reads inside a container through the stand-in docker. The container's name holds
# a space and shell syntax, so it reaches docker intact only as one argument, never through a shell.
CONTAINER = "stand-in workspace; echo shell"


def read_in_container(tmp_path, monkeypatch, capfd, answer: str) -> tuple[bool | str, list, str]:
    """R6 with the stand-in docker's read answering ``answer``: what R6 read back, the arguments
    docker got, and all that reached this process's output, which is the release journal."""
    monkeypatch.setenv("STAND_IN_READ", answer)
    actions = make_actions(tmp_path, workspace_container=CONTAINER)
    try:
        held = actions.workspace_reads_ok()
    except Exception as error:  # a readback never raises
        held = repr(error)
    recorded = Path(os.environ["STAND_IN_DOCKER_ARGS"])
    args = recorded.read_text().split("\0")[:-1] if recorded.exists() else []
    out, err = capfd.readouterr()
    return held, args, out + err


def one_read() -> list[str]:
    """docker's arguments for R6's read: exec, the container's name, node and the fixed script."""
    return ["exec", CONTAINER, "node", "-e", release_host_actions._WORKSPACE_READS]


def test_r6_is_true_when_the_container_read_succeeds(tmp_path, monkeypatch, capfd):
    assert read_in_container(tmp_path, monkeypatch, capfd, "ok") == (True, one_read(), "")


def test_r6_is_false_when_the_container_read_is_refused(tmp_path, monkeypatch, capfd):
    assert read_in_container(tmp_path, monkeypatch, capfd, "refused") == (False, one_read(), "")


def test_r6_is_false_when_the_container_read_fails(tmp_path, monkeypatch, capfd):
    assert read_in_container(tmp_path, monkeypatch, capfd, "fails") == (False, one_read(), "")


def test_r6_is_false_when_the_container_read_is_late(tmp_path, monkeypatch, capfd):
    monkeypatch.setattr(release_host_actions, "READBACK_TIMEOUT_SECONDS", SLOW_SECONDS / 5)
    began = time.monotonic()
    assert read_in_container(tmp_path, monkeypatch, capfd, "late") == (False, one_read(), "")
    assert time.monotonic() - began < SLOW_SECONDS  # it gave up at the time limit


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
