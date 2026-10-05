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

from hermes_cli import build_info, release_host_actions
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


def wal_database(path: Path, rows: int) -> sqlite3.Connection:
    """A database in write-ahead-log mode whose rows stay in its log while the writer is open."""
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = sqlite3.connect(path)
    writer.execute("PRAGMA journal_mode=WAL")
    writer.execute("PRAGMA wal_autocheckpoint=0")
    writer.execute("CREATE TABLE notes (body TEXT)")
    writer.executemany("INSERT INTO notes VALUES (?)", [(f"note {n}",) for n in range(rows)])
    writer.commit()
    return writer


# Every application store of a root with two boards and one served profile, where the release
# record (release_ledger.py), the board registry, its removal archive and the boards (kanban_db.py),
# the cron queues (cron/), the hosted rooms (gateway/hosted_rooms.py), a plugin's data
# (plugins/plugin_storage.py) and the profile's own state keep them.
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
    "plugin-data/sample/data.db",
    "profiles/coder/state.db",
    "profiles/coder/cron/delivery_records.db",
)
# Databases inside the root that are no application store: a task's workspace, the checkout and
# the release snapshots.
NOT_STORES = (
    "kanban/boards/sample/workspaces/t_worker/app.db",
    "work/hermes/tests/fixture.db",
    "release-snapshots/state/OLD/state.db",
)


def test_state_snapshot_reaches_every_store_from_a_worker_environment(tmp_path, monkeypatch):
    root = tmp_path / "root"
    writers = [wal_database(root / rel, rows) for rows, rel in enumerate(STORES, start=1)]
    for rel in NOT_STORES:
        wal_database(root / rel, 1).close()
    # What the dispatcher gives a worker: its own board and task, and a profile home.
    monkeypatch.setenv("HERMES_KANBAN_DB", str(root / "kanban" / "boards" / "sample" / "kanban.db"))
    monkeypatch.setenv("HERMES_KANBAN_BOARD", "sample")
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_worker")
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
    copies = sorted(path.relative_to(saved).as_posix() for path in saved.rglob("*.db"))
    assert copies == sorted(STORES)  # against the independent count of the stores made above
    for rows, rel in enumerate(STORES, start=1):
        with contextlib.closing(sqlite3.connect(saved / rel)) as copy:
            assert copy.execute("SELECT count(*) FROM notes").fetchone() == (rows,), rel


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
