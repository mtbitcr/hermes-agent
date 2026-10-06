"""Card 4 of the release re-cut: ``hermes release prepare`` and ``hermes release status``.

prepare runs against FakeHost, an in-memory release host shaped like the one in
test_release_sequence.py that stands in for both live adapters, so no real host command runs.
Each test imports the command through ``_command``, so a missing module fails that test on its
own instead of the whole file at collection.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import os
import sqlite3
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

import pytest
import yaml

from hermes_cli import release_ledger as ledger
from hermes_cli.release_guards import GUARDS


def _commit(label: str) -> str:
    """A 40-hex id shaped like a git commit."""
    return hashlib.sha1(label.encode()).hexdigest()


PREV, NEW, FEATURE, TREE = (_commit(label) for label in ("prev", "new", "feature", "tree"))
LATER = _commit("later")
WROTE_NOTHING = "Nothing in the release state was written."
CHECKOUT = "/srv/hermes/checkout"
CONFIG = {"config.yaml": "digest"}
PAGE_REF = "decisions-page:release"
# What the owner sets in config.yaml; the gateway and serve units keep their defaults.
SETTINGS = {
    "units": {"sandbox-tunnel": "hermes-sandbox-tunnel"},
    "health_url": "http://127.0.0.1:8642/health",
    "workspace_check_url": "http://127.0.0.1:8650/api/projects",
}
UNITS = {"gateway": "hermes-gateway", "serve": "hermes-serve", **SETTINGS["units"]}
_REPO = Path(__file__).resolve().parents[2]
# The real CLI as the hermes script runs it, with a FakeHost of the kill mode in argv[2] standing
# in for both live adapters, as in _command: there is no real release host here.
_CLI = (
    "import sys\n"
    "sys.path.insert(0, sys.argv.pop(1))\n"
    "from test_release_cmd import FakeHost\n"
    "from hermes_cli import release_cmd\n"
    "host = FakeHost(kill_mode=sys.argv.pop(1))\n"
    "release_cmd.LiveHostReader = release_cmd.ReleaseHostActions = host.build\n"
    "from hermes_cli.main import main\n"
    "sys.argv[0] = 'hermes'\n"
    "sys.exit(main())\n"
)
# A writer of the release record in its own process, as the merge step is. It turns the record to
# WAL mode, commits an accepted batch for NEW and a later change to the -wal alone, says ready and
# keeps its connection until its input closes. That -wal holds no frame of page 1, the header.
_WRITER = (
    "import sys\n"
    "sys.path.insert(0, sys.argv[1])\n"
    "import test_release_cmd as t\n"
    "conn = t.ledger.connect()\n"
    "conn.execute('PRAGMA journal_mode=WAL')\n"
    "conn.execute('PRAGMA wal_autocheckpoint=0')\n"
    "t._accept(conn, t._merge(conn, t.NEW))\n"
    "t._merge(conn, t.LATER, 2)\n"
    "print('ready', flush=True)\n"
    "sys.stdin.read()\n"
    "conn.close()\n"
)


@dataclass
class FakeHost:
    """A healthy release host in memory, at PREV, on which every guard passes for NEW.

    It stands in for both live adapters and keeps the keywords each was built with. Any other call
    is one of the host's moves or readbacks, which prepare never asks; it is only noted.
    """

    head: str = PREV
    snapshot_dirs: list[str] = field(default_factory=list)
    kill_mode: str = "mixed"
    runs_readable: bool = True
    built: list[dict] = field(default_factory=list)
    chains: list[tuple[str, str]] = field(default_factory=list)
    asked: list[str] = field(default_factory=list)

    def build(self, **settings):
        self.built.append(settings)
        return self

    def __getattr__(self, name):
        return lambda *args, **kwargs: self.asked.append(name)

    def checkout_head(self):
        return self.head

    def checkout_is_clean(self):
        return True

    def origin_main(self):
        return NEW

    def first_parent_chain(self, prev, new):
        self.chains.append((prev, new))
        return [new]

    def commit_parents(self, commit):
        return (PREV, FEATURE)

    def commit_tree(self, commit):
        return TREE

    def is_ancestor(self, ancestor, descendant):
        # As in git, a commit is its own ancestor.
        return ancestor == descendant or (ancestor, descendant) == (PREV, NEW)

    def changed_paths(self, prev, new):
        return {"hermes_cli/main.py"}

    def checkout_root(self):
        return CHECKOUT

    def venv_import_root(self):
        return CHECKOUT

    def unit_property(self, unit, name):
        return {"KillSignal": "SIGINT", "KillMode": self.kill_mode}[name]

    def free_disk_bytes(self):
        return 5 * 1024**3

    def snapshots(self):
        return list(self.snapshot_dirs)

    def open_native_runs(self):
        if not self.runs_readable:
            raise RuntimeError("the board listing could not be read")
        return []

    def live_config(self):
        return dict(CONFIG)

    def named_config_snapshot(self):
        return dict(CONFIG)


@pytest.fixture
def root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """The root Hermes home at ``<tmp>/.hermes``, with the release settings this host needs."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    (home / "config.yaml").write_text(yaml.safe_dump({"release": SETTINGS}))
    return home


def _command(monkeypatch: pytest.MonkeyPatch, host: FakeHost | None = None):
    """The release command, with ``host`` built in place of both live adapters."""
    from hermes_cli import release_cmd

    if host is not None:
        monkeypatch.setattr(release_cmd, "LiveHostReader", host.build)
        monkeypatch.setattr(release_cmd, "ReleaseHostActions", host.build)
    return release_cmd


def _merge(conn, commit: str, pr: int = 1, title: str = "") -> dict:
    """Record a merged change as FakeHost's history has it; returns the batch it joined."""
    return ledger.record_merge(
        conn,
        merge_commit=commit,
        pr_url=f"https://github.com/mtbitcr/hermes-agent/pull/{pr}",
        reviewed_base=PREV,
        reviewed_head=FEATURE,
        reviewed_tree=TREE,
        tier=1,
        card_id=f"t_{commit[:8]}",
        title=title,
        on_main=True,
    )["batch"]


def _accept(conn, batch: dict) -> dict:
    return ledger.decide_release(
        conn,
        batch["batch_id"],
        decision="accepted",
        shown_digest=batch["digest"],
        expected_version=batch["version"],
        decision_ref=PAGE_REF,
    )


def _release(conn, batch: dict, *, prev: str, outcome: str) -> None:
    """Begin and finish the batch's release in the record only (S-B, S-G)."""
    new = batch["members"][-1]["merge_commit"]
    ledger.begin_release(conn, batch["batch_id"], prev=prev, new=new)
    ledger.finish_release(conn, batch["batch_id"], outcome=outcome)


def _accepted_batch(*commits: str) -> dict:
    """Record the changes, accept the batch they joined and close the record again."""
    conn = ledger.connect()
    try:
        for commit in commits:
            batch = _merge(conn, commit)
        return _accept(conn, batch)
    finally:
        conn.close()


def _files(top: Path) -> dict[str, tuple[bytes, int]]:
    """Every path under ``top`` with its bytes and modification time."""
    return {
        str(path): (path.read_bytes() if path.is_file() else b"", path.stat().st_mtime_ns)
        for path in top.rglob("*")
    }


def _release_state(top: Path) -> dict[str, bytes | None]:
    """The release record, the snapshot directory and the homes' configuration files and
    databases under ``top``, each path with its bytes, None for a directory. The generic start of
    every hermes command (logs, caches, the persona file, locks) is no part of it."""
    return {
        str(path): path.read_bytes() if path.is_file() else None
        for path in top.rglob("*")
        if "release-snapshots" in path.relative_to(top).parts
        or path.name in ("config.yaml", ".env")
        or ".db" in path.name
    }


def _live_files(path: Path) -> dict[str, bytes | None]:
    """The live record's database file and side files, each with its bytes, or None if absent."""
    files = {side: Path(f"{path}{side}") for side in ("", "-wal", "-shm", "-journal")}
    return {side: file.read_bytes() if file.exists() else None for side, file in files.items()}


def _env(top: Path, home: Path) -> dict[str, str]:
    """The environment of a child process: HOME at ``top`` and HERMES_HOME at ``home``, both
    temporary, never a real home."""
    return {
        **os.environ,
        "HOME": str(top),
        "HERMES_HOME": str(home),
        "PYTHONPATH": str(_REPO),
        "PYTHONDONTWRITEBYTECODE": "1",
    }


def _cli(top: Path, home: Path, *argv: str, kill_mode: str = "mixed"):
    """``hermes *argv`` in a child process (see _CLI)."""
    return subprocess.run(
        [sys.executable, "-c", _CLI, str(Path(__file__).parent), kill_mode, *argv],
        cwd=_REPO,
        env=_env(top, home),
        capture_output=True,
        text=True,
        timeout=120,
    )


@contextlib.contextmanager
def _wal_writer(top: Path, home: Path):
    """_WRITER, from the moment its changes are committed until the block ends."""
    with subprocess.Popen(
        [sys.executable, "-c", _WRITER, str(Path(__file__).parent)],
        env=_env(top, home),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    ) as writer:
        assert writer.stdout.readline() == "ready\n"
        yield writer


def test_prepare_runs_every_guard_and_writes_nothing(root, tmp_path, monkeypatch, capsys):
    host = FakeHost()
    release_cmd = _command(monkeypatch, host)
    batch = _accepted_batch(NEW)
    before = _files(tmp_path)

    assert release_cmd.prepare() == 0
    out = capsys.readouterr().out
    assert f"Batch {batch['batch_id']}: PREV {PREV}, NEW {NEW}" in out
    for guard, rule, _check in GUARDS:
        assert f"{guard} passed: {rule}" in out
    # S-E: both live adapters are built from the settings, with explicit keywords only.
    shared = {
        "checkout": Path(release_cmd.__file__).resolve().parents[1],
        "root_home": root,
        "units": UNITS,
        "snapshot_root": root / "release-snapshots",
    }
    assert host.built == [
        {**shared, "snapshot_name": NEW},
        {**shared, "health_url": SETTINGS["health_url"],
         "workspace_check_url": SETTINGS["workspace_check_url"]},
    ]

    host.kill_mode, host.runs_readable = "control-group", False
    assert release_cmd.prepare() != 0
    out = capsys.readouterr().out
    # A failing guard does not stop the pass, and a read that raises fails its guard.
    verdicts = {line.split()[0]: line.split()[1] for line in out.splitlines() if line[0] == "G"}
    assert verdicts == {
        guard: "failed:" if guard in ("G7", "G10") else "passed:" for guard, _, _ in GUARDS
    }
    assert "RuntimeError: the board listing could not be read" in out
    # The record, the homes and the snapshot directory are unchanged; no move was asked.
    assert _files(tmp_path) == before
    assert host.asked == []


def test_pins_come_from_the_last_released_batch_or_the_checkout(root, monkeypatch, capsys):
    host = FakeHost(head=_commit("checkout"))
    release_cmd = _command(monkeypatch, host)
    first = _accepted_batch(_commit("a"), NEW)

    release_cmd.prepare()  # no batch was ever released: PREV is the checkout's head
    conn = ledger.connect()
    _release(conn, first, prev=host.head, outcome="released")
    conn.close()
    _accepted_batch(_commit("b"), _commit("c"))
    release_cmd.prepare()  # PREV is the NEW of the last released batch, wherever the head is

    out = capsys.readouterr().out
    assert f"PREV {_commit('checkout')}, NEW {NEW}" in out
    assert f"PREV {NEW}, NEW {_commit('c')}" in out
    # NEW is the merge commit of the accepted batch's last member, and the guards ask about
    # exactly these pins.
    assert host.chains == [(_commit("checkout"), NEW), (NEW, _commit("c"))]


def test_refusal_after_a_rollback_says_a_new_change_is_needed(root, monkeypatch, capsys):
    # The release that was tried took NEW's state snapshot after the stop, then rolled back.
    host = FakeHost(snapshot_dirs=[NEW])
    release_cmd = _command(monkeypatch, host)
    conn = ledger.connect()
    _release(conn, _accept(conn, _merge(conn, NEW)), prev=PREV, outcome="restored")
    [returned] = [batch for batch in ledger.list_batches(conn) if batch["state"] == "open"]
    _accept(conn, returned)  # the same changes, accepted again with no new one
    conn.close()

    assert release_cmd.prepare() != 0

    out = capsys.readouterr().out
    assert "G9 failed: this release's state snapshot does not exist yet" in out
    assert "already tried and rolled back" in out and "a new change is needed" in out


def test_prepare_refuses_under_a_profile_home(root, tmp_path, monkeypatch, capsys):
    host = FakeHost()
    release_cmd = _command(monkeypatch, host)
    _accepted_batch(NEW)
    profile = root / "profiles" / "coder"
    profile.mkdir(parents=True)
    (profile / "config.yaml").write_text(yaml.safe_dump({"release": SETTINGS}))
    monkeypatch.setenv("HERMES_HOME", str(profile))
    before = _files(tmp_path)

    assert release_cmd.prepare() != 0

    out = capsys.readouterr().out
    assert out.startswith("Refused:") and f"root Hermes home {root}" in out
    assert (host.built, _files(tmp_path)) == ([], before)


@pytest.mark.parametrize(
    ("settings", "reason"),
    [
        pytest.param(
            {**SETTINGS, "units": {}},
            "the sandbox tunnel unit is not set: set release.units.sandbox-tunnel",
            id="sandbox-tunnel-unit",
        ),
        pytest.param(
            {**SETTINGS, "health_url": ""},
            "the health address is not set: set release.health_url",
            id="health-address",
        ),
        pytest.param(
            {**SETTINGS, "workspace_check_url": ""},
            "the owner-page check address is not set: set release.workspace_check_url",
            id="owner-page-check-address",
        ),
    ],
)
def test_unset_required_settings_refuse(root, monkeypatch, capsys, settings, reason):
    host = FakeHost()
    release_cmd = _command(monkeypatch, host)
    (root / "config.yaml").write_text(yaml.safe_dump({"release": settings}))
    _accepted_batch(NEW)

    assert release_cmd.prepare() != 0

    assert f"Refused: {reason} in {root / 'config.yaml'}." in capsys.readouterr().out
    assert host.built == []


def test_status_prints_the_waiting_batch_and_last_outcome_in_plain_words(
    root, tmp_path, monkeypatch, capsys
):
    release_cmd = _command(monkeypatch)
    before = _files(tmp_path)

    assert release_cmd.status() == 0
    assert capsys.readouterr().out.splitlines() == [
        "No batch is waiting for a decision.",
        "Last outcome: no release has finished yet.",
    ]
    assert _files(tmp_path) == before  # a look creates no record

    conn = ledger.connect()
    released = _accept(conn, _merge(conn, _commit("a"), 1, "Faster boards"))
    _release(conn, released, prev=PREV, outcome="released")
    tried = _accept(conn, _merge(conn, _commit("b"), 2, "Quieter logs"))
    _release(conn, tried, prev=_commit("a"), outcome="restored")
    waiting = _merge(conn, _commit("c"), 3)  # joins the changes that came back
    conn.close()

    assert release_cmd.status() == 0
    assert capsys.readouterr().out.splitlines() == [
        f"Waiting batch {waiting['batch_id']}, waiting for the owner's decision: 2 changes",
        "  - Quieter logs (https://github.com/mtbitcr/hermes-agent/pull/2)",
        f"  - t_{_commit('c')[:8]} (https://github.com/mtbitcr/hermes-agent/pull/3)",
        f"Last outcome: batch {tried['batch_id']} was rolled back; its changes went back to the"
        " waiting decision.",
    ]


def test_release_is_registered_with_its_subcommands(monkeypatch):
    from hermes_cli import main

    monkeypatch.setattr(main, "_plugin_cli_discovery_needed", lambda: False)
    parser, subparsers = main._build_cli_parser()

    release = subparsers.choices["release"]
    [nested] = [a for a in release._actions if isinstance(a, argparse._SubParsersAction)]
    assert set(nested.choices) == {"prepare", "status"}
    # A built-in name, so `hermes release` skips plugin discovery and `hermes --help` lists it.
    assert "release" in main._BUILTIN_SUBCOMMANDS
    assert "Check a release before it runs" in parser.format_help()
    args = parser.parse_args(["release", "status"])
    assert args.release_command == "status" and callable(args.func)


@pytest.mark.parametrize(
    ("case", "code", "tail"),
    [
        ("passing prepare", 0, [f"All {len(GUARDS)} release guards passed. {WROTE_NOTHING}"]),
        ("failed guard", 1, [f"Refused: 1 of {len(GUARDS)} release guards failed.", WROTE_NOTHING]),
        (
            "unset settings",
            1,
            [
                "Refused: the sandbox tunnel unit is not set: set release.units.sandbox-tunnel in"
                " {root}/config.yaml.",
                WROTE_NOTHING,
            ],
        ),
        (
            "profile home",
            1,
            [
                "Refused: a release runs only from the root Hermes home {root}, not from"
                " {root}/profiles/coder.",
                WROTE_NOTHING,
            ],
        ),
        (
            "status",
            0,
            ["No batch is waiting for a decision.", "Last outcome: no release has finished yet."],
        ),
    ],
)
def test_the_real_command_writes_nothing_to_the_release_state(root, tmp_path, case, code, tail):
    # Through the CLI in a child process, its generic start included.
    profile = root / "profiles" / "coder"
    profile.mkdir(parents=True)
    for home in (root, profile):
        (home / "config.yaml").write_text(yaml.safe_dump({"release": SETTINGS}))
        (home / ".env").write_text("HERMES_RELEASE_TEST=kept\n")
        with contextlib.closing(sqlite3.connect(home / "state.db")) as db:
            db.execute("CREATE TABLE kept (value)")
    for kind in ("state", "config"):
        (root / "release-snapshots" / kind / PREV).mkdir(parents=True)
        (root / "release-snapshots" / kind / PREV / "kept").write_text(kind)
    if case == "unset settings":
        (root / "config.yaml").write_text(yaml.safe_dump({"release": {**SETTINGS, "units": {}}}))
    _accepted_batch(NEW)
    before = _release_state(tmp_path)

    done = _cli(
        tmp_path,
        profile if case == "profile home" else root,
        "release",
        "status" if case == "status" else "prepare",
        kill_mode="control-group" if case == "failed guard" else "mixed",
    )

    assert done.returncode == code, done.stderr
    assert done.stdout.splitlines()[-len(tail):] == [line.format(root=root) for line in tail]
    # The release record, the snapshot directory and the homes' configuration files and
    # databases are unchanged, byte for byte.
    assert _release_state(tmp_path) == before


@pytest.mark.parametrize("command", ["prepare", "status"])
@pytest.mark.parametrize("writing", [False, True], ids=["closed-wal-ledger", "live-writer"])
def test_the_release_record_is_read_from_a_private_copy(
    root, tmp_path, monkeypatch, capsys, command, writing
):
    release_cmd = _command(monkeypatch, FakeHost())
    path = ledger.ledger_path()
    with _wal_writer(tmp_path, root) as writer:
        if not writing:
            writer.stdin.close()
            writer.wait(timeout=30)  # it folds the -wal back and removes it and the -shm
        before = _live_files(path)
        assert before[""][18:20] == b"\2\2"  # the record is in WAL mode
        assert (before["-wal"] is not None, before["-shm"] is not None) == (writing, writing)

        assert getattr(release_cmd, command)() == 0

        assert _live_files(path) == before
    # Every committed change is read, those still only in the -wal too.
    out = capsys.readouterr().out
    if command == "prepare":
        assert f"PREV {PREV}, NEW {NEW}" in out
    else:
        assert "waiting for the owner's decision: 1 change" in out


@pytest.mark.parametrize("command", ["prepare", "status"])
@pytest.mark.parametrize(
    ("damage", "cause"),
    [
        pytest.param(
            "empty", "OperationalError: no such table: release_batches", id="empty-main-file"
        ),
        pytest.param("header", "DatabaseError: file is not a database", id="damaged-header"),
    ],
)
def test_a_record_copy_sqlite_cannot_read_fails_closed(
    root, tmp_path, monkeypatch, capsys, command, damage, cause
):
    host = FakeHost()
    release_cmd = _command(monkeypatch, host)
    path = ledger.ledger_path()
    with _wal_writer(tmp_path, root):
        wal = Path(f"{path}-wal").read_bytes()  # a valid -wal of this record
    main = path.read_bytes()
    path.write_bytes(b"" if damage == "empty" else bytes(100) + main[100:])
    Path(f"{path}-wal").write_bytes(wal)
    before = _live_files(path)

    assert getattr(release_cmd, command)() == 1

    assert capsys.readouterr().out.splitlines() == [
        f"Refused: the release record {path} could not be read ({cause}).",
        WROTE_NOTHING,
    ]
    assert (_live_files(path), host.built) == (before, [])
