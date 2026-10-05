"""Release card 2: the live-host reader, read against temporary stand-ins only.

Each test builds what the reader reads under tmp_path: a git repository with a bare remote, or a
partial clone of one; a stand-in systemctl first on PATH; boards made by the real kanban kernel;
and Hermes homes, with the state records of their gateways. Nothing here reaches the real host,
the real HERMES_HOME, systemd or the network. What a test expects is read back without the module
under test.
"""

from __future__ import annotations

import json
import logging
import os
import re
import sqlite3
import subprocess
import sys
import time
from contextlib import closing
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import release_host
from hermes_cli.release_guards import GUARDS, Pins, RecordedMerge, prepare
from hermes_cli.release_host import LiveHostReader
from tests.hermes_cli._kanban_fence_support import (
    DEFAULT_GIT_IDENTITY,
    create_fenced_board,
    git,
    make_git_repo,
    read_only,
    ready_task,
)
from tests.hermes_cli._kanban_worker_identity_support import dead_pid, live_child, pid_is_alive

REPO = Path(__file__).resolve().parents[2]
UNITS = {"gateway": "hermes-gateway", "serve": "hermes-serve", "sandbox-tunnel": "hermes-sandbox-tunnel"}
NEW = "1" * 40
SECRET = "sk-release-host-test-secret"
# S-D written out by hand: config.yaml and .env of the root and of each served profile home.
CONFIG_KEYS = {
    f"{home}{name}"
    for home in ("", "profiles/coder/", "profiles/writer/")
    for name in ("config.yaml", ".env")
}
# A beta board writer in a process of its own, since SQLite shares one -shm mapping between the
# connections of a process. It commits an open run whose worker is the given pid, then exits
# without closing the board, or idles with it open until its input ends and then closes it.
BOARD_WRITER = """
import os, sys
from hermes_cli import kanban_db as kb
from tests.hermes_cli._kanban_fence_support import ready_task
conn = kb.connect(board="beta")
task_id = ready_task(conn)
assert kb.claim_task(conn, task_id) is not None
kb._set_worker_pid(conn, task_id, int(sys.argv[1]))
if sys.argv[2] == "crash":
    os._exit(0)
print("idle", flush=True)
sys.stdin.readline()
conn.close()
"""
# A competing writer in a process of its own: it takes the board's write lock at once, or prints
# why it cannot, and commits a row naming itself.
COMPETITOR = """
import sqlite3, sys
conn = sqlite3.connect(sys.argv[1], timeout=0, isolation_level=None)
try:
    conn.execute("BEGIN IMMEDIATE")
    conn.execute("INSERT INTO writes VALUES (?)", (sys.argv[2],))
    conn.execute("COMMIT")
    print("COMMITTED")
except sqlite3.Error as exc:
    print(exc)
"""


@pytest.fixture(autouse=True)
def _no_host_git_config(monkeypatch):
    # The reader's own git calls must not read this machine's git configuration either.
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", os.devnull)
    monkeypatch.setenv("GIT_CONFIG_SYSTEM", os.devnull)


def _reader(tmp_path, root, *, name=NEW):
    return LiveHostReader(
        checkout=tmp_path / "checkout",
        root_home=root,
        units=UNITS,
        snapshot_root=tmp_path / "snapshots",
        snapshot_name=name,
    )


def _check(host, guard):
    check = next(check for name, _rule, check in GUARDS if name == guard)
    return check(host, Pins(new=NEW, prev=NEW), ())


def _merge(work, name):
    git(work, "checkout", "-b", name)
    (work / f"{name}.txt").write_text(f"{name}\n", encoding="utf-8")
    git(work, "add", f"{name}.txt")
    git(work, "commit", "-m", name, author=DEFAULT_GIT_IDENTITY)
    git(work, "checkout", "main")
    git(work, "merge", "--no-ff", "-m", f"merge {name}", name, author=DEFAULT_GIT_IDENTITY)
    return git(work, "rev-parse", "HEAD")


def _repository(tmp_path):
    """A checkout at PREV whose bare origin is two merges ahead. Returns (checkout, PREV, merges)."""
    checkout, remote, work = tmp_path / "checkout", tmp_path / "origin.git", tmp_path / "work"
    git(tmp_path, "init", "--bare", "--initial-branch=main", str(remote))
    make_git_repo(checkout)
    (checkout / ".gitignore").write_text("venv/\n", encoding="utf-8")
    (checkout / "hermes_cli").mkdir()
    (checkout / "hermes_cli" / "__init__.py").write_text("", encoding="utf-8")
    git(checkout, "add", ".gitignore", "hermes_cli/__init__.py")
    git(checkout, "commit", "-m", "prev", author=DEFAULT_GIT_IDENTITY)
    git(checkout, "remote", "add", "origin", str(remote))
    git(checkout, "push", "origin", "main")
    git(tmp_path, "clone", str(remote), str(work))
    merges = [_merge(work, "one"), _merge(work, "two")]
    git(work, "push", "origin", "main")
    # NEW's objects reach the checkout's store, as card 3's fetch brings them; origin/main stays.
    git(checkout, "fetch", str(remote), "main")
    return checkout, git(checkout, "rev-parse", "HEAD"), merges


def _systemctl(tmp_path, monkeypatch):
    """A stand-in systemctl first on PATH: it logs its arguments and answers `show` for KillSignal
    and KillMode; it prints nothing for any other property."""
    calls, script = tmp_path / "systemctl.calls", tmp_path / "bin" / "systemctl"
    script.parent.mkdir()
    script.write_text(
        "#!/bin/sh\n"
        f'echo "$*" >> "{calls}"\n'
        'case "$*" in\n'
        "  *--property=KillSignal*) echo KillSignal=2 ;;\n"
        "  *--property=KillMode*) echo KillMode=mixed ;;\n"
        "esac\n",
        encoding="utf-8",
    )
    script.chmod(0o755)
    monkeypatch.setenv("PATH", f"{script.parent}{os.pathsep}{os.environ['PATH']}")
    return calls


def _gateway_record(home, pid, code_sha=None):
    """The gateway_state.json of a running gateway of *home*, with *pid* and no start time, stamped
    with *code_sha* unless it is None. Returns the record."""
    record = {"pid": pid, "gateway_state": "running"}
    if code_sha is not None:
        record["code_sha"] = code_sha
    home.mkdir(parents=True, exist_ok=True)
    (home / "gateway_state.json").write_text(json.dumps(record), encoding="utf-8")
    return record


def _homes(root, *profiles):
    """The root home and named profile homes, each with a config.yaml and a .env holding SECRET."""
    for home in (root, *(root / "profiles" / name for name in profiles)):
        home.mkdir(parents=True, exist_ok=True)
        (home / "config.yaml").write_text(f"model: {home.name}\n", encoding="utf-8")
        (home / ".env").write_text(f"OPENAI_API_KEY={SECRET}\n", encoding="utf-8")


def _save_config_snapshot(root, snapshot_root, name):
    """The configuration snapshot as S-C and S-D lay it out: config/NAME/<path from the root>."""
    for key in CONFIG_KEYS:
        target = snapshot_root / "config" / name / key
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((root / key).read_bytes())


def _open_run(conn, pid):
    """A claimed task whose run records *pid* as its worker, through the kernel's own calls."""
    task_id = ready_task(conn)
    assert kb.claim_task(conn, task_id) is not None
    kb._set_worker_pid(conn, task_id, pid)
    return task_id


def _independent_open_runs(root):
    """Open runs read straight from each board file the root registry names, without kanban_db."""
    with read_only(root / "kanban" / "board_register.db") as conn:
        slugs = {"default", *(row[0] for row in conn.execute("SELECT board_name FROM board_register"))}
    runs = set()
    for slug in slugs:
        path = root / "kanban.db" if slug == "default" else root / "kanban" / "boards" / slug / "kanban.db"
        with read_only(path) as conn:
            rows = conn.execute(
                "SELECT id, worker_pid FROM task_runs WHERE status = 'running' AND ended_at IS NULL"
            ).fetchall()
        runs |= {(f"{slug}:{run_id}", pid_is_alive(pid)) for run_id, pid in rows}
    return runs


def _tree_bytes(*roots):
    """Every path under the roots: a file's bytes, a link's target, None for a directory."""
    tree = {}
    for root in roots:
        for path in (root, *root.rglob("*")):
            if path.is_symlink():
                tree[path] = os.readlink(path)
            else:
                tree[path] = None if path.is_dir() else path.read_bytes()
    return tree


def _before_board_open(monkeypatch, board, *hook):
    """Runs the *hook* lines once in the reader's board-reading process, just before SQLite opens *board*."""
    prelude = "\n".join((
        "import sqlite3",
        "def _connect(database, *args, _real=sqlite3.connect, **kwargs):",
        f"    if database.startswith({board.resolve().as_uri()!r}):",
        "        sqlite3.connect = _real",
        *(f"        {line}" for line in hook),
        "    return _real(database, *args, **kwargs)",
        "sqlite3.connect = _connect",
    ))
    monkeypatch.setattr(release_host, "_BOARD_READ", f"{prelude}\n{release_host._BOARD_READ}")


def _held_write(tmp_path, fence_home):
    """The beta board in rollback mode with a writes table, and a connection of this process holding
    an uncommitted write on it. Returns (board file, that connection, the reader)."""
    kb.init_db()
    create_fenced_board("beta")
    beta_db = fence_home / "kanban" / "boards" / "beta" / "kanban.db"
    first = sqlite3.connect(beta_db, isolation_level=None)
    first.execute("PRAGMA journal_mode=DELETE")
    first.execute("CREATE TABLE writes (writer TEXT)")
    first.execute("INSERT INTO writes VALUES ('seed')")
    first.execute("BEGIN IMMEDIATE")
    first.execute("UPDATE writes SET writer = 'first'")
    return beta_db, first, _reader(tmp_path, fence_home)


def _compete(beta_db, name):
    argv = [sys.executable, "-c", COMPETITOR, str(beta_db), name]
    return subprocess.run(argv, capture_output=True, text=True, timeout=120, check=True).stdout.strip()


def test_git_reads_answer_from_a_real_repository(tmp_path):
    checkout, prev, (m1, m2) = _repository(tmp_path)
    host = _reader(tmp_path, tmp_path / "root")

    assert host.checkout_head() == prev
    (checkout / "venv").mkdir()
    (checkout / "venv" / "pyvenv.cfg").write_text("home = /usr/bin\n", encoding="utf-8")
    assert host.checkout_is_clean() is True  # an ignored path is no change
    (checkout / "stray.txt").write_text("stray\n", encoding="utf-8")
    assert host.checkout_is_clean() is False
    # The local remote-tracking ref is stale; origin main is read from the remote itself (S-H).
    assert git(checkout, "rev-parse", "refs/remotes/origin/main") == prev
    assert host.origin_main() == m2
    assert list(host.first_parent_chain(prev, m2)) == [m2, m1]
    assert host.commit_parents(m2) == (m1, git(checkout, "rev-parse", f"{m2}^2"))
    assert host.commit_tree(m2) == git(checkout, "rev-parse", f"{m2}^{{tree}}")
    assert host.is_ancestor(prev, m2) is True
    assert host.is_ancestor(m2, prev) is False
    assert set(host.changed_paths(prev, m2)) == {"one.txt", "two.txt"}
    assert host.checkout_root() == str(checkout.resolve())


def test_git_reads_fetch_nothing_from_a_promisor_remote(tmp_path):
    checkout, remote, seed = tmp_path / "checkout", tmp_path / "origin.git", tmp_path / "seed"
    git(tmp_path, "init", "--bare", "--initial-branch=main", str(remote))
    git(remote, "config", "uploadpack.allowFilter", "true")
    make_git_repo(seed)
    git(seed, "push", str(remote), "main")
    # A partial clone: origin is its promisor remote, which serves each missing tree on demand.
    git(tmp_path, "clone", "--filter=tree:0", remote.as_uri(), str(checkout))
    prev = git(checkout, "rev-parse", "HEAD")
    (seed / "new.txt").write_text("new\n", encoding="utf-8")
    git(seed, "add", "new.txt")
    git(seed, "commit", "-m", "new", author=DEFAULT_GIT_IDENTITY)
    git(seed, "push", str(remote), "main")
    git(checkout, "fetch", "origin")  # NEW's commit only: its tree stays on the remote
    new = git(checkout, "rev-parse", "refs/remotes/origin/main")
    host, packs = _reader(tmp_path, tmp_path / "root"), checkout / ".git" / "objects" / "pack"

    before, pack_files = _tree_bytes(checkout), set(packs.iterdir())
    # A missing object is refused, not fetched: fetching is card 3's (S-H).
    with pytest.raises(subprocess.CalledProcessError):
        host.changed_paths(prev, new)
    with pytest.raises(subprocess.CalledProcessError):
        host.commit_tree(new)
    assert set(packs.iterdir()) == pack_files
    assert _tree_bytes(checkout) == before


def test_unit_property_returns_signal_names(tmp_path, monkeypatch):
    calls = _systemctl(tmp_path, monkeypatch)
    host = _reader(tmp_path, tmp_path / "root")

    assert host.unit_property("gateway", "KillSignal") == "SIGINT"
    assert host.unit_property("gateway", "KillMode") == "mixed"
    argvs = [line.split() for line in calls.read_text(encoding="utf-8").splitlines()]
    assert len(argvs) == 2
    for argv in argvs:  # only the user manager is asked, about the unit the role maps to
        assert "--user" in argv and "--system" not in argv
        assert argv[-1] == "hermes-gateway"


@pytest.mark.parametrize("case", ["current", "stale", "unknown", "none", "every"])
def test_venv_import_root_asks_the_live_gateways(tmp_path, fence_home, monkeypatch, case):
    checkout, prev, (m1, _m2) = _repository(tmp_path)
    worker, coder = fence_home / "profiles" / "worker", fence_home / "profiles" / "coder"
    worker.mkdir(parents=True)
    # A live gateway, in this process's pid, in the root home and in the coder profile; for none,
    # only a dead one in the root home.
    pid = dead_pid() if case == "none" else os.getpid()
    records = [_gateway_record(fence_home, pid, {"stale": m1, "unknown": None}.get(case, prev))]
    if case != "none":
        records.append(_gateway_record(coder, pid, m1 if case == "every" else prev))
    stamps = {record.get("code_sha") for record in records if pid_is_alive(record["pid"])}
    expected = str(checkout.resolve()) if stamps == {prev} else ""
    # The environment the dispatcher gives a worker of the beta board, in the worker profile.
    monkeypatch.setenv("HERMES_HOME", str(worker))
    monkeypatch.setenv("HERMES_KANBAN_DB", str(fence_home / "kanban" / "boards" / "beta" / "kanban.db"))
    monkeypatch.setenv("HERMES_KANBAN_BOARD", "beta")
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_0123abcd")
    calls = _systemctl(tmp_path, monkeypatch)  # whatever G6 asks, no real user manager answers
    host = _reader(tmp_path, fence_home)

    loaded = set(sys.modules)
    found = host.venv_import_root()
    imported = set(sys.modules) - loaded
    if case == "current":
        assert imported == set(), " ".join(sorted(imported))  # K6: a G6 read imports nothing
    assert found == expected
    assert _check(host, "G6") is (case == "current")
    assert not calls.exists()  # the gateways answer, not their unit


def test_open_runs_cover_every_board_under_a_worker_env(tmp_path, fence_home, monkeypatch):
    kb.init_db()
    create_fenced_board("beta")
    beta_db = fence_home / "kanban" / "boards" / "beta" / "kanban.db"
    profile = fence_home / "profiles" / "worker"
    profile.mkdir(parents=True)
    with live_child() as worker:
        with closing(kb.connect(board="default")) as default, closing(kb.connect(board="beta")) as beta:
            _open_run(default, worker.pid)
            task_id = _open_run(beta, dead_pid())
        # The environment the dispatcher gives a worker of the beta board.
        monkeypatch.setenv("HERMES_HOME", str(profile))
        monkeypatch.setenv("HERMES_KANBAN_DB", str(beta_db))
        monkeypatch.setenv("HERMES_KANBAN_BOARD", "beta")
        monkeypatch.setenv("HERMES_KANBAN_TASK", task_id)
        assert kb.kanban_db_path("default") == beta_db  # the pin hides every other board's file

        found = {(run.run_id, run.worker_alive) for run in _reader(tmp_path, fence_home).open_native_runs()}
        expected = _independent_open_runs(fence_home)
        assert found == expected
        assert sorted(alive for _run_id, alive in expected) == [False, True]
        with pytest.raises(RuntimeError):  # a reader built for another root refuses to answer
            _reader(tmp_path, profile).open_native_runs()


def test_unreadable_board_counts_as_an_open_live_run(tmp_path, fence_home):
    kb.init_db()
    create_fenced_board("beta")
    with closing(kb.connect(board="beta")) as conn:
        _open_run(conn, dead_pid())
    host = _reader(tmp_path, fence_home)
    assert [run.worker_alive for run in host.open_native_runs()] == [False]
    assert _check(host, "G10") is True
    assert kb.count_running_tasks_other_boards("default") == 1

    # Running as root, permissions cannot hide a file, so the board gets corrupt bytes instead.
    (fence_home / "kanban" / "boards" / "beta" / "kanban.db").write_bytes(b"not a database\n" * 512)
    assert [run.worker_alive for run in host.open_native_runs()] == [True]
    assert _check(host, "G10") is False  # the drain fails closed
    assert kb.count_running_tasks_other_boards("default") == 0  # the host-cap count fails open (F18)


def test_a_run_committed_between_wal_detection_and_open_fails_the_drain(tmp_path, fence_home, monkeypatch):
    kb.init_db()
    create_fenced_board("beta")
    beta_db = fence_home / "kanban" / "boards" / "beta" / "kanban.db"
    with closing(sqlite3.connect(beta_db)) as conn:  # a WAL board whose last connection closed
        conn.execute("PRAGMA journal_mode=WAL")
    assert not Path(f"{beta_db}-wal").exists()
    # The reader has looked for the -wal file and opens the board read-only. Just before, a writer
    # commits a run of this live process and exits without closing the board, so the run is in
    # the WAL only.
    writer = [sys.executable, "-c", BOARD_WRITER, str(os.getpid()), "crash"]
    run_writer = f"subprocess.run({writer!r}, cwd={str(REPO)!r}, check=True, timeout=120)"
    _before_board_open(monkeypatch, beta_db, "import subprocess", run_writer)
    assert _check(_reader(tmp_path, fence_home), "G10") is False
    assert Path(f"{beta_db}-wal").stat().st_size > 0
    assert [alive for _run_id, alive in _independent_open_runs(fence_home)] == [True]  # written once


def test_reading_boards_keeps_the_write_lock_of_this_process(tmp_path, fence_home):
    beta_db, first, host = _held_write(tmp_path, fence_home)
    with closing(first):
        assert _compete(beta_db, "before") == "database is locked"
        assert _check(host, "G10") is True
        # The read let go of none of the open transaction's locks.
        assert _compete(beta_db, "after") == "database is locked"
        assert first.in_transaction


def test_reading_boards_loses_no_committed_row(tmp_path, fence_home):
    beta_db, first, host = _held_write(tmp_path, fence_home)
    with closing(first):
        assert _check(host, "G10") is True
        committed = {"second"} if _compete(beta_db, "second") == "COMMITTED" else set()
        try:
            first.execute("COMMIT")
            committed.add("first")
        except sqlite3.Error:
            pass
    with read_only(beta_db) as conn:
        rows = {writer for (writer,) in conn.execute("SELECT writer FROM writes")}
    assert committed and committed <= rows  # every row whose COMMIT succeeded is there


@pytest.mark.parametrize("writer", ["crash", "idle", "closing", "this process", "empty", "corrupt"])
def test_reading_a_wal_board_changes_none_of_its_files(tmp_path, fence_home, monkeypatch, writer):
    kb.init_db()
    create_fenced_board("beta")
    beta_db = fence_home / "kanban" / "boards" / "beta" / "kanban.db"
    with closing(sqlite3.connect(beta_db)) as conn:
        conn.execute("PRAGMA journal_mode=WAL")
    host, unreadable = _reader(tmp_path, fence_home), {("beta:unreadable", True)}
    if writer == "this process":  # an idle read-write connection in the reader's own process
        with closing(kb.connect(board="beta")) as conn:
            _open_run(conn, os.getpid())
            before = _tree_bytes(fence_home)
            found = {(run.run_id, run.worker_alive) for run in host.open_native_runs()}
            assert _tree_bytes(fence_home) == before
            expected = _independent_open_runs(fence_home)
        assert [alive for _run_id, alive in expected] == [True]
        assert found in (expected, unreadable)  # the run stays visible, or the board fails closed
        return
    mode = {"closing": "idle", "empty": "crash", "corrupt": "crash"}.get(writer, writer)
    argv = [sys.executable, "-c", BOARD_WRITER, str(os.getpid()), mode]
    with subprocess.Popen(argv, cwd=REPO, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True) as proc:
        if writer == "idle" or writer == "closing":  # a live -shm, its writer idle with the board open
            assert proc.stdout.readline() == "idle\n"
        else:  # a committed WAL left by an unclean exit, without its -shm
            assert proc.wait(timeout=120) == 0
            Path(f"{beta_db}-shm").unlink()
        if writer == "closing":  # then, as the board's last connection, it closes after the -wal check
            close = f"os.write(os.open('/proc/{proc.pid}/fd/0', os.O_WRONLY), b'\\n')"
            gone = f"select.select([os.pidfd_open({proc.pid})], [], [], 120)"
            _before_board_open(monkeypatch, beta_db, "import os, select", close, gone)
        if writer in ("empty", "corrupt"):  # and the board file itself left empty, or corrupt
            beta_db.write_bytes(b"" if writer == "empty" else b"not a database\n" * 512)
        assert Path(f"{beta_db}-wal").stat().st_size > 0
        before = _tree_bytes(fence_home)
        found = {(run.run_id, run.worker_alive) for run in host.open_native_runs()}
        if writer in ("empty", "corrupt"):
            assert _check(host, "G10") is False
        assert _tree_bytes(fence_home) == before  # the leftover -wal keeps its bytes too
        expected = unreadable if writer in ("empty", "corrupt") else _independent_open_runs(fence_home)
    assert [alive for _run_id, alive in expected] == [True]
    # The committed run stays visible, or the board fails closed; either way the drain refuses.
    assert found in (expected, unreadable)


def test_config_digests_cover_root_and_profiles_and_match_the_named_snapshot(
    tmp_path, fence_home, monkeypatch, caplog
):
    caplog.set_level(logging.DEBUG)
    _homes(fence_home, "coder", "writer")
    _save_config_snapshot(fence_home, tmp_path / "snapshots", NEW)
    # The set is the root and every served profile, whichever profile is active.
    monkeypatch.setenv("HERMES_HOME", str(fence_home / "profiles" / "coder"))
    host = _reader(tmp_path, fence_home)

    live, named = host.live_config(), host.named_config_snapshot()
    assert set(live) == set(named) == CONFIG_KEYS
    assert live == named and _check(host, "G11") is True
    assert all(re.fullmatch(r"[0-9a-f]{64}", digest) for digest in live.values())
    env = fence_home / "profiles" / "writer" / ".env"
    content = bytearray(env.read_bytes())
    content[-2] ^= 1
    env.write_bytes(bytes(content))
    changed = host.live_config()
    assert {key for key in CONFIG_KEYS if changed[key] != named[key]} == {"profiles/writer/.env"}
    assert _check(host, "G11") is False
    assert SECRET not in repr((live, named, changed)) + caplog.text


def test_reader_writes_nothing(tmp_path, fence_home, monkeypatch):
    checkout, prev, (m1, m2) = _repository(tmp_path)
    # A tracked file whose stat no longer matches the index: a plain `git status` would rewrite it.
    os.utime(checkout / "README.md", (time.time() + 3600,) * 2)
    _systemctl(tmp_path, monkeypatch)
    _homes(fence_home, "coder", "writer")
    _gateway_record(fence_home, os.getpid(), prev)  # G6: the root home's live gateway runs PREV
    snapshot_root = tmp_path / "snapshots"
    _save_config_snapshot(fence_home, snapshot_root, m2)
    for age, name in ((200, "f" * 40), (100, "a" * 40)):
        (snapshot_root / "state" / name).mkdir(parents=True)
        os.utime(snapshot_root / "state" / name, (time.time() - age,) * 2)
    kb.init_db()
    create_fenced_board("beta")
    with closing(kb.connect(board="beta")) as conn:
        _open_run(conn, dead_pid())
    beta_db = fence_home / "kanban" / "boards" / "beta" / "kanban.db"
    with closing(sqlite3.connect(beta_db)) as conn:  # a WAL board whose last writer closed cleanly
        conn.execute("PRAGMA journal_mode=WAL")
    assert not Path(f"{beta_db}-wal").exists()
    merges = [
        RecordedMerge(merge, *git(checkout, "rev-parse", f"{merge}^1", f"{merge}^2", f"{merge}^{{tree}}").split())
        for merge in (m1, m2)
    ]
    host = _reader(tmp_path, fence_home, name=m2)

    before = _tree_bytes(checkout, fence_home, snapshot_root)
    results = prepare(host, Pins(new=m2, prev=prev), merges)
    assert _tree_bytes(checkout, fence_home, snapshot_root) == before
    # Every guard holds on this host, except G8 where this machine has under 4 GiB free.
    assert [result.guard for result in results if not result.ok] in ([], ["G8"])
    assert host.snapshots() == ["f" * 40, "a" * 40]  # oldest first
