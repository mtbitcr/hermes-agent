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
import stat
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import pytest
import yaml

from hermes_cli import release_ledger as ledger
from hermes_cli.release_guards import GUARDS
from hermes_cli.release_host import LiveHostReader


def _commit(label: str) -> str:
    """A 40-hex id shaped like a git commit."""
    return hashlib.sha1(label.encode()).hexdigest()


PREV, NEW, FEATURE, TREE = (_commit(label) for label in ("prev", "new", "feature", "tree"))
LATER = _commit("later")
WROTE_NOTHING = "Nothing in the release state was written."
CHECKOUT = "/srv/hermes/checkout"
CONFIG = {"config.yaml": "digest"}
PAGE_REF = "decisions-page:release"
# What the owner sets in config.yaml; the gateway and serve units and the Workspace container keep
# their defaults.
SETTINGS = {
    "units": {"sandbox-tunnel": "hermes-sandbox-tunnel"},
    "health_url": "http://127.0.0.1:8642/health",
}
UNITS = {"gateway": "hermes-gateway", "serve": "hermes-serve", **SETTINGS["units"]}
# Each required setting and the plain name its refusal gives it.
REQUIRED = [
    pytest.param("units.gateway", "the gateway unit", id="gateway-unit"),
    pytest.param("units.serve", "the serve unit", id="serve-unit"),
    pytest.param("units.sandbox-tunnel", "the sandbox tunnel unit", id="sandbox-tunnel-unit"),
    pytest.param("health_url", "the health address", id="health-address"),
]
# The two forms of a reference in config.yaml; a test puts its own variable's name for VAR.
REFERENCE_FORMS = [pytest.param("${VAR}", id="var"), pytest.param("${env:VAR}", id="env-var")]
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
# The first bytes of a rollback journal once it is synced, as it is before a writer puts any change
# into the database file. Until then they are zeros, and the journal is not hot.
_JOURNAL_MAGIC = bytes.fromhex("d9d505f920a163d7")


@dataclass
class FakeHost:
    """A healthy release host in memory, at PREV, on which every guard passes for NEW.

    It stands in for both live adapters and keeps the keywords each was built with. Any other call
    is one of the host's moves or readbacks, which prepare never asks; it is only noted. With
    live_snapshots, the state snapshots and their sizes are what the real LiveHostReader, built
    with the reader's keywords, finds on disk, instead of snapshot_dirs of 0 bytes each.
    """

    head: str = PREV
    snapshot_dirs: list[str] = field(default_factory=list)
    kill_mode: str = "mixed"
    runs_readable: bool = True
    live_snapshots: bool = False
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
        if self.live_snapshots:
            [reader] = [settings for settings in self.built if "snapshot_name" in settings]
            return LiveHostReader(**reader).snapshots()
        return list(self.snapshot_dirs)

    def snapshot_bytes(self, name):
        if self.live_snapshots:
            [reader] = [settings for settings in self.built if "snapshot_name" in settings]
            return LiveHostReader(**reader).snapshot_bytes(name)
        return 0

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
    return _root_at(tmp_path, monkeypatch)


def _root_at(top: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """The root Hermes home at ``<top>/.hermes``, with the release settings this host needs."""
    home = top / ".hermes"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: top)
    monkeypatch.setenv("HERMES_HOME", str(home))
    (home / "config.yaml").write_text(yaml.safe_dump({"release": SETTINGS}))
    return home


def _settings_with(key: str, value: str) -> dict:
    """SETTINGS with the dotted ``key`` set to ``value``."""
    section, _, leaf = key.partition(".")
    if leaf:
        return {**SETTINGS, section: {**SETTINGS[section], leaf: value}}
    return {**SETTINGS, key: value}


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


def _live_files(path: Path) -> dict[str, bytes | tuple[str, str | None] | None]:
    """The live record's database file and side files: a regular file with its bytes, any other
    name with its kind and permissions and a link's target, None if absent. A pipe is never opened
    and a link never followed."""

    def look(name: Path) -> bytes | tuple[str, str | None] | None:
        try:
            mode = name.lstat().st_mode
        except FileNotFoundError:
            return None
        if stat.S_ISREG(mode):
            return name.read_bytes()
        return stat.filemode(mode), (os.readlink(name) if stat.S_ISLNK(mode) else None)

    return {side: look(Path(f"{path}{side}")) for side in ("", "-wal", "-shm", "-journal")}


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


def _cli(
    top: Path, home: Path, *argv: str, kill_mode: str = "mixed", user: tuple[int, int] | None = None
):
    """``hermes *argv`` in a child process (see _CLI), as ``user``'s uid and gid when given."""
    as_user = {} if user is None else {"user": user[0], "group": user[1], "extra_groups": []}
    return subprocess.run(
        [sys.executable, "-c", _CLI, str(Path(__file__).parent), kill_mode, *argv],
        cwd=_REPO,
        env=_env(top, home),
        capture_output=True,
        text=True,
        timeout=120,
        **as_user,
    )


def _unprivileged() -> tuple[int, int] | None:
    """The user nobody and its group when the tests run as root, which searches any folder; None,
    the current user, otherwise."""
    if os.geteuid() != 0:
        return None
    import pwd

    try:
        entry = pwd.getpwnam("nobody")
    except KeyError:
        pytest.skip("the tests run as root and there is no user nobody to run a check as")
    return entry.pw_uid, entry.pw_gid


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


class _RolledBack(Exception):
    """Raised in place of a held COMMIT, so its transaction rolls back."""


def _accept_held_at_commit(batch: dict, cache_pages: int, at_commit) -> None:
    """Accept ``batch`` on a connection with ``cache_pages`` pages of cache, call ``at_commit``
    when the decision's COMMIT is due, and roll the decision back instead of committing it. With
    one or two pages the decision has spilled into the database file by then."""

    class HeldAtCommit(sqlite3.Connection):
        def execute(self, sql, parameters=(), /):
            if sql == "COMMIT":
                at_commit()
                raise _RolledBack
            return super().execute(sql, parameters)

    with contextlib.closing(
        sqlite3.connect(ledger.ledger_path(), isolation_level=None, factory=HeldAtCommit)
    ) as conn:
        conn.row_factory = sqlite3.Row
        conn.execute(f"PRAGMA cache_size={cache_pages}")
        with pytest.raises(_RolledBack):
            _accept(conn, batch)


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
         "workspace_container": "raphael-workspace"},
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


@pytest.mark.parametrize("form", REFERENCE_FORMS)
@pytest.mark.parametrize(("key", "name"), REQUIRED)
def test_a_reference_that_resolves_to_an_empty_value_refuses(
    root, monkeypatch, capsys, key, name, form
):
    host = FakeHost()
    release_cmd = _command(monkeypatch, host)
    monkeypatch.setenv("RELEASE_REF_TEST_EMPTY", "")
    reference = form.replace("VAR", "RELEASE_REF_TEST_EMPTY")
    (root / "config.yaml").write_text(yaml.safe_dump({"release": _settings_with(key, reference)}))
    _accepted_batch(NEW)

    assert release_cmd.prepare() != 0

    # The refusal of a literal empty value, before any adapter is built.
    assert capsys.readouterr().out.splitlines() == [
        f"Refused: {name} is not set: set release.{key} in {root / 'config.yaml'}.",
        WROTE_NOTHING,
    ]
    assert host.built == []


@pytest.mark.parametrize("form", REFERENCE_FORMS)
@pytest.mark.parametrize(
    ("key", "name"),
    [*REQUIRED, pytest.param("snapshot_dir", "the snapshot directory", id="snapshot-dir")],
)
def test_a_reference_to_a_variable_that_is_not_set_refuses(
    root, monkeypatch, capsys, key, name, form
):
    # The reference is kept as written; its placeholder must reach neither adapter.
    host = FakeHost()
    release_cmd = _command(monkeypatch, host)
    monkeypatch.delenv("RELEASE_REF_TEST_UNSET", raising=False)
    reference = form.replace("VAR", "RELEASE_REF_TEST_UNSET")
    (root / "config.yaml").write_text(yaml.safe_dump({"release": _settings_with(key, reference)}))
    _accepted_batch(NEW)

    assert release_cmd.prepare() != 0

    assert capsys.readouterr().out.splitlines() == [
        f"Refused: {name} refers to a variable that is not set: set the variable, or change"
        f" release.{key} in {root / 'config.yaml'}.",
        WROTE_NOTHING,
    ]
    assert host.built == []


@pytest.mark.parametrize("form", REFERENCE_FORMS)
def test_both_adapters_are_built_from_the_resolved_settings(root, monkeypatch, form):
    host = FakeHost()
    release_cmd = _command(monkeypatch, host)

    def ref(variable: str, value: str) -> str:
        monkeypatch.setenv(variable, value)
        return form.replace("VAR", variable)

    settings = {
        "units": {
            "gateway": ref("RELEASE_REF_TEST_GATEWAY", "hermes-gateway-blue"),
            "serve": ref("RELEASE_REF_TEST_SERVE", "hermes-serve-blue"),
            "sandbox-tunnel": ref("RELEASE_REF_TEST_TUNNEL", "hermes-sandbox-tunnel-blue"),
        },
        "health_url": ref("RELEASE_REF_TEST_HEALTH", "http://127.0.0.1:9642/health"),
        "workspace_container": ref("RELEASE_REF_TEST_CONTAINER", "raphael-workspace-blue"),
        # An empty snapshot_dir, here from a reference, means release-snapshots under the root.
        "snapshot_dir": ref("RELEASE_REF_TEST_EMPTY", ""),
    }
    (root / "config.yaml").write_text(yaml.safe_dump({"release": settings}))
    _accepted_batch(NEW)

    assert release_cmd.prepare() == 0

    shared = {
        "checkout": Path(release_cmd.__file__).resolve().parents[1],
        "root_home": root,
        "units": {
            "gateway": "hermes-gateway-blue",
            "serve": "hermes-serve-blue",
            "sandbox-tunnel": "hermes-sandbox-tunnel-blue",
        },
        "snapshot_root": root / "release-snapshots",
    }
    assert host.built == [
        {**shared, "snapshot_name": NEW},
        {
            **shared,
            "health_url": "http://127.0.0.1:9642/health",
            "workspace_container": "raphael-workspace-blue",
        },
    ]


@pytest.mark.parametrize("form", REFERENCE_FORMS)
def test_g9_looks_for_new_under_the_resolved_snapshot_dir(
    root, tmp_path, monkeypatch, capsys, form
):
    # A real directory, named by a reference, with an earlier release's state snapshot in it.
    snapshots = tmp_path / "resolved-snapshots"
    (snapshots / "state" / PREV).mkdir(parents=True)
    monkeypatch.setenv("RELEASE_REF_TEST_SNAPSHOTS", str(snapshots))
    reference = form.replace("VAR", "RELEASE_REF_TEST_SNAPSHOTS")
    (root / "config.yaml").write_text(
        yaml.safe_dump({"release": _settings_with("snapshot_dir", reference)})
    )
    _accepted_batch(NEW)

    host = FakeHost(live_snapshots=True)
    assert _command(monkeypatch, host).prepare() == 0
    assert "G9 passed: this release's state snapshot does not exist yet" in capsys.readouterr().out
    # Both adapters have the resolved root, and the real reader looked under it.
    assert [settings["snapshot_root"] for settings in host.built] == [snapshots, snapshots]

    # A release of these very changes took NEW's state snapshot there, then rolled back.
    (snapshots / "state" / NEW).mkdir()
    host = FakeHost(live_snapshots=True)
    assert _command(monkeypatch, host).prepare() != 0
    out = capsys.readouterr().out
    assert "G9 failed: this release's state snapshot does not exist yet" in out
    assert "already tried and rolled back" in out and "a new change is needed" in out


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
    assert set(nested.choices) == {"prepare", "status", "run", "start", "recover"}
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
def test_the_committed_changes_of_a_record_in_wal_mode_are_read(
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

        # The database file and the writer's -wal are as they were. SQLite may change the
        # -shm, and make it and an empty -wal where there were none.
        after = _live_files(path)
        assert (after[""], after["-wal"] or None) == (before[""], before["-wal"])
    # Every committed change is read, those still only in the -wal too.
    out = capsys.readouterr().out
    if command == "prepare":
        assert f"PREV {PREV}, NEW {NEW}" in out
    else:
        assert "waiting for the owner's decision: 1 change" in out


@pytest.mark.parametrize("cache_pages", [1, 2, 5])
def test_a_decision_that_was_never_committed_is_never_reported(root, tmp_path, cache_pages):
    # S1 of the final security review: the owner's Accept is held at its COMMIT while both
    # commands run, each in its own process, and is then rolled back. This process does not open
    # the database file meanwhile: closing it would drop the writer's locks on it.
    with contextlib.closing(ledger.connect()) as conn:
        batch = _merge(conn, NEW)
    path = ledger.ledger_path()
    before = _live_files(path)
    done = {}

    def run_both():
        if cache_pages < 5:  # the decision is in the database file already
            assert Path(f"{path}-journal").read_bytes().startswith(_JOURNAL_MAGIC)
        for command in ("prepare", "status"):
            done[command] = _cli(tmp_path, root, "release", command)

    _accept_held_at_commit(batch, cache_pages, run_both)

    assert _live_files(path) == before  # the decision is gone, and the reads changed nothing
    # Each command reads the committed record, in which the batch still waits, or refuses in plain
    # words; neither reports the acceptance, which never was.
    committed = {
        "prepare": (
            1,
            ["Refused: no batch is accepted, so there is nothing to release.", WROTE_NOTHING],
        ),
        "status": (
            0,
            [
                f"Waiting batch {batch['batch_id']}, waiting for the owner's decision: 1 change",
                f"  - t_{NEW[:8]} (https://github.com/mtbitcr/hermes-agent/pull/1)",
                "Last outcome: no release has finished yet.",
            ],
        ),
    }
    unreadable = f"Refused: the release record {path} could not be read ("
    seen, expected = {}, {}
    for command, result in done.items():
        out = result.stdout.splitlines()
        refusal = next((line for line in out if line.startswith(unreadable)), None)
        code, lines = committed[command] if refusal is None else (1, [refusal, WROTE_NOTHING])
        seen[command], expected[command] = (result.returncode, out[-len(lines):]), (code, lines)
    assert seen == expected


@pytest.mark.parametrize("command", ["prepare", "status"])
def test_a_record_with_a_hot_journal_fails_closed(root, monkeypatch, capsys, command):
    # A writer stopped at its COMMIT, after its Accept spilled into the database file, leaves a hot
    # journal: only a rollback, which a read-only reader cannot make, brings back what was
    # committed.
    host = FakeHost()
    release_cmd = _command(monkeypatch, host)
    with contextlib.closing(ledger.connect()) as conn:
        batch = _merge(conn, NEW)
    path = ledger.ledger_path()
    left = {}

    def stop():  # what the writer leaves when it stops there
        left.update((side, Path(f"{path}{side}").read_bytes()) for side in ("", "-journal"))

    _accept_held_at_commit(batch, 1, stop)
    for side, data in left.items():
        Path(f"{path}{side}").write_bytes(data)
    assert left["-journal"].startswith(_JOURNAL_MAGIC)
    before = _live_files(path)

    assert getattr(release_cmd, command)() == 1

    assert capsys.readouterr().out.splitlines() == [
        f"Refused: the release record {path} could not be read (OperationalError: attempt to"
        " write a readonly database).",
        WROTE_NOTHING,
    ]
    assert (_live_files(path), host.built) == (before, [])


@pytest.mark.parametrize(
    ("command", "code", "lines"),
    [
        (
            "prepare",
            1,
            ["Refused: no batch is accepted, so there is nothing to release.", WROTE_NOTHING],
        ),
        (
            "status",
            0,
            ["No batch is waiting for a decision.", "Last outcome: no release has finished yet."],
        ),
    ],
)
def test_a_record_none_of_whose_names_exists_reads_as_empty(
    root, monkeypatch, capsys, command, code, lines
):
    release_cmd = _command(monkeypatch, FakeHost())
    path = ledger.ledger_path()
    path.parent.mkdir(parents=True)  # the record's folder, with none of its four names in it

    assert getattr(release_cmd, command)() == code

    assert capsys.readouterr().out.splitlines() == lines
    assert set(_live_files(path).values()) == {None}  # and none was made


@pytest.mark.parametrize("command", ["prepare", "status"])
def test_a_record_whose_database_file_is_a_regular_file_is_read(
    root, monkeypatch, capsys, command
):
    release_cmd = _command(monkeypatch, FakeHost())
    path = ledger.ledger_path()
    conn = ledger.connect()
    accepted = _accept(conn, _merge(conn, NEW))
    waiting = _merge(conn, LATER, 2)
    conn.close()
    before = _live_files(path)

    assert getattr(release_cmd, command)() == 0

    assert capsys.readouterr().out.splitlines() == {
        "prepare": [
            f"Batch {accepted['batch_id']}: PREV {PREV}, NEW {NEW}",
            *(f"{guard} passed: {rule}" for guard, rule, _check in GUARDS),
            f"All {len(GUARDS)} release guards passed. {WROTE_NOTHING}",
        ],
        "status": [
            f"Waiting batch {waiting['batch_id']}, waiting for the owner's decision: 1 change",
            f"  - t_{LATER[:8]} (https://github.com/mtbitcr/hermes-agent/pull/2)",
            "Last outcome: no release has finished yet.",
        ],
    }[command]
    assert _live_files(path) == before


# Each state of the live record that is neither of its two shapes, and the cause its refusal gives.
@pytest.mark.parametrize("command", ["prepare", "status"])
@pytest.mark.parametrize(
    ("state", "cause"),
    [
        pytest.param("directory", "it is not a regular file", id="directory"),
        pytest.param("link", "it is a symbolic link, not a regular file", id="broken-link"),
        pytest.param("pipe", "it is not a regular file", id="pipe"),
        *(
            pytest.param(
                side,
                f"there is no database file beside release_ledger.db{side}",
                id=f"orphan{side}",
            )
            for side in ("-wal", "-shm", "-journal")
        ),
    ],
)
def test_every_other_state_of_the_record_fails_closed(
    root, monkeypatch, capsys, command, state, cause
):
    host = FakeHost()
    release_cmd = _command(monkeypatch, host)
    path = ledger.ledger_path()
    path.parent.mkdir(parents=True)
    if state == "directory":
        path.mkdir()
    elif state == "link":
        path.symlink_to(path.with_name("moved.db"))  # broken: nothing is there
    elif state == "pipe":
        os.mkfifo(path)  # a read of it would wait for a writer
    else:  # a side file without the database file
        Path(f"{path}{state}").write_bytes(b"left behind")
    before = _live_files(path)

    assert getattr(release_cmd, command)() == 1

    assert capsys.readouterr().out.splitlines() == [
        f"Refused: the release record {path} could not be read ({cause}).",
        WROTE_NOTHING,
    ]
    assert (_live_files(path), host.built) == (before, [])


@pytest.mark.parametrize("command", ["prepare", "status"])
def test_a_record_in_a_folder_that_cannot_be_searched_fails_closed(monkeypatch, command):
    # Through the CLI in a child process, as nobody when the tests run as root, since root searches
    # any folder. Pytest's own temporary root may be closed to nobody, so the public parent is made
    # here, and nobody is given the home as an ordinary user has their own.
    user = _unprivileged()
    with tempfile.TemporaryDirectory(prefix="release-record-") as name:
        top = Path(name)
        top.chmod(0o755)
        home = _root_at(top, monkeypatch)
        _accepted_batch(NEW)
        path = ledger.ledger_path()
        if user is not None:
            for item in [top, *top.rglob("*")]:
                os.chown(item, *user)
        before = _live_files(path)
        path.parent.chmod(0)
        try:
            done = _cli(top, home, "release", command, user=user)
        finally:
            path.parent.chmod(0o755)

        assert done.returncode == 1, done.stderr
        assert done.stdout.splitlines()[-2:] == [
            f"Refused: the release record {path} could not be read (PermissionError: [Errno 13]"
            f" Permission denied: '{path}').",
            WROTE_NOTHING,
        ]
        assert _live_files(path) == before


# Damage to a record that is otherwise well formed, and the plain reason each command that reads
# the damaged part refuses with; a command that does not read it runs as usual. The record holds
# a released batch, an accepted one and a waiting one (see the test below).
MALFORMED = [
    # Both commands read every batch as the record's own snapshot decodes it.
    pytest.param(
        "ALTER TABLE release_batches RENAME COLUMN version TO damaged_version",
        dict.fromkeys(
            ("prepare", "status"),
            "batch {released} could not be decoded: IndexError: No item with that key",
        ),
        id="missing-batch-version",
    ),
    *(
        pytest.param(
            f"ALTER TABLE release_members RENAME COLUMN {name} TO damaged_{name}",
            dict.fromkeys(
                ("prepare", "status"),
                f"batch {{released}} could not be decoded: KeyError: '{name}'",
            ),
            id=f"missing-member-{name}",
        )
        for name in ("tier_recorded", "merge_commit")
    ),
    pytest.param(
        "UPDATE release_members SET merge_commit = char(233) || substr(merge_commit, 2)"
        " WHERE batch_id = (SELECT id FROM release_batches WHERE state = 'open')",
        dict.fromkeys(
            ("prepare", "status"),
            "batch {waiting} could not be decoded: UnicodeEncodeError: 'ascii' codec can't encode"
            " character '\\xe9' in position 0: ordinal not in range(128)",
        ),
        id="merge-commit-not-ascii",
    ),
    # prepare reads NEW, PREV and the accepted batch's changes as the guards read them.
    pytest.param(
        "DELETE FROM release_members"
        " WHERE batch_id = (SELECT id FROM release_batches WHERE state = 'accepted')",
        {"prepare": "accepted batch {accepted} has no changes"},
        id="accepted-batch-without-changes",
    ),
    pytest.param(
        "UPDATE release_members SET merge_commit = ''"
        " WHERE batch_id = (SELECT id FROM release_batches WHERE state = 'accepted')",
        {"prepare": "a change in batch {accepted} has no merge_commit"},
        id="empty-new",
    ),
    *(
        pytest.param(
            f"ALTER TABLE release_members RENAME COLUMN {name} TO damaged_{name}",
            {"prepare": f"a change in batch {{accepted}} has no {name}"},
            id=f"missing-member-{name}",
        )
        for name in ("reviewed_base", "reviewed_head", "reviewed_tree")
    ),
    pytest.param(
        "UPDATE release_batches SET new = NULL WHERE state = 'released'",
        {"prepare": "the last released batch has no NEW"},
        id="released-batch-without-new",
    ),
    # status reads the waiting batch's changes as it shows them, and the last outcome. The waiting
    # change has no title, so its card id is shown.
    *(
        pytest.param(
            f"ALTER TABLE release_members RENAME COLUMN {name} TO damaged_{name}",
            {"status": f"a change in batch {{waiting}} has no {name}"},
            id=f"missing-member-{name}",
        )
        for name in ("title", "card_id", "pr_url")
    ),
    pytest.param(
        "UPDATE release_batches SET outcome = NULL WHERE state = 'released'",
        {"status": "batch {released}, whose release finished last, has no known outcome"},
        id="last-outcome-missing",
    ),
    pytest.param(
        "INSERT INTO release_events (batch_id, kind, payload, created_at)"
        " VALUES (99, 'release_finished', '{}', 0)",
        {"status": "batch 99, whose release finished last, has no known outcome"},
        id="last-finished-batch-missing",
    ),
]


@pytest.mark.parametrize(("damage", "refusals"), MALFORMED)
def test_a_malformed_record_refuses_in_plain_words(root, monkeypatch, capsys, damage, refusals):
    path = ledger.ledger_path()
    conn = ledger.connect()
    released = _accept(conn, _merge(conn, PREV))
    _release(conn, released, prev=_commit("older"), outcome="released")  # its NEW is PREV now
    accepted = _accept(conn, _merge(conn, NEW))
    waiting = _merge(conn, LATER, 2)
    conn.close()
    with contextlib.closing(sqlite3.connect(path)) as damaged:
        damaged.executescript(damage)
    before = _live_files(path)
    batches = {
        "released": released["batch_id"],
        "accepted": accepted["batch_id"],
        "waiting": waiting["batch_id"],
    }
    usual = {
        "prepare": [
            f"Batch {accepted['batch_id']}: PREV {PREV}, NEW {NEW}",
            *(f"{guard} passed: {rule}" for guard, rule, _check in GUARDS),
            f"All {len(GUARDS)} release guards passed. {WROTE_NOTHING}",
        ],
        "status": [
            f"Waiting batch {waiting['batch_id']}, waiting for the owner's decision: 1 change",
            f"  - t_{LATER[:8]} (https://github.com/mtbitcr/hermes-agent/pull/2)",
            f"Last outcome: batch {released['batch_id']} was released.",
        ],
    }

    for command in ("prepare", "status"):
        host = FakeHost()
        reason = refusals.get(command)
        assert getattr(_command(monkeypatch, host), command)() == (0 if reason is None else 1)
        out = capsys.readouterr().out.splitlines()
        if reason is None:  # it does not read the damaged part
            assert out == usual[command]
        else:  # the plain refusal alone: no other line came first and no adapter was built
            assert out == [
                f"Refused: the release record {path} could not be read"
                f" ({reason.format(**batches)}).",
                WROTE_NOTHING,
            ]
            assert host.built == []
    assert _live_files(path) == before


def test_status_needs_a_card_id_only_for_a_change_without_a_title(root, monkeypatch, capsys):
    # A change's card id is shown only when it has no title: while every waiting change has one,
    # status asks for no card id.
    path = ledger.ledger_path()
    conn = ledger.connect()
    waiting = _merge(conn, LATER, 2, "Later changes")
    conn.close()
    with contextlib.closing(sqlite3.connect(path)) as damaged:
        damaged.execute("ALTER TABLE release_members RENAME COLUMN card_id TO damaged_card_id")

    assert _command(monkeypatch).status() == 0

    assert capsys.readouterr().out.splitlines() == [
        f"Waiting batch {waiting['batch_id']}, waiting for the owner's decision: 1 change",
        "  - Later changes (https://github.com/mtbitcr/hermes-agent/pull/2)",
        "Last outcome: no release has finished yet.",
    ]
