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
import socket
import sqlite3
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from hermes_cli import release_host_actions
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


SLOW_SECONDS = 2.5


class Answers(http.server.BaseHTTPRequestHandler):
    """/ok answers 200, /slow answers 200 only after the readback gave up, anything else 503."""

    def do_GET(self) -> None:
        if self.path == "/slow":
            time.sleep(SLOW_SECONDS)
        with contextlib.suppress(OSError):  # the reader may be gone by now
            self.send_response(200 if self.path in ("/ok", "/slow") else 503)
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

    def readbacks(url: str) -> tuple[bool, bool]:
        actions = make_actions(tmp_path, health_url=url, workspace_check_url=url)
        return actions.health_ok(), actions.workspace_reads_ok()

    assert readbacks(f"{server}/ok") == (True, True)
    for url in (f"{server}/broken", f"{server}/slow", refused_url(), ""):
        assert readbacks(url) == (False, False), url


def test_fleet_version_is_one_version_or_none(tmp_path):
    root = tmp_path / "root"
    coder = root / "profiles" / "coder"
    coder.mkdir(parents=True)
    ended = subprocess.Popen([sys.executable, "-c", ""])
    ended.wait()
    new, old = "1" * 40, "2" * 40

    def stamp(home: Path, pid: int, code_sha: str) -> None:
        record = {"pid": pid, "gateway_state": "running", "code_sha": code_sha}
        (home / "gateway_state.json").write_text(json.dumps(record))

    actions = make_actions(tmp_path)
    assert actions.fleet_version() == ""  # no gateway reports at all
    stamp(root, os.getpid(), new)
    stamp(coder, os.getpid(), new)
    assert actions.fleet_version() == new
    stamp(coder, os.getpid(), old)
    assert actions.fleet_version() == ""
    stamp(coder, ended.pid, old)  # a gateway that has exited no longer reports
    assert actions.fleet_version() == new
