"""The live host, read for the release guards (release card 2).

`LiveHostReader` is the real `HostReader` of `hermes_cli.release_guards`. It reads:
- the checkout and its history, with origin main read from the remote itself (S-H);
- systemd unit properties through the user manager, free disk and the state snapshot names;
- the open runs on every board;
- digests of the configuration set (S-D), live and in the named configuration snapshot (S-C).

Every method only reads. Moving the checkout and taking or restoring snapshots are actions, and
they belong to card 3. Reading also leaves nothing behind, which is what lets prepare mode keep
its promise to write nothing: git runs with --no-optional-locks, so `git status` does not
refresh the index; origin main comes from `git ls-remote`, which updates no local ref; a board
file is opened so that SQLite creates no side files; and the venv probe writes no bytecode.

Like the runner (K6), this module imports everything when it loads and no method imports
anything: the readbacks after the stop call the checkout, venv and configuration reads while
the checkout on disk is being swapped.
"""

from __future__ import annotations

import hashlib
import shutil
import signal
import sqlite3
import subprocess
from collections.abc import Collection, Mapping, Sequence
from contextlib import closing
from pathlib import Path

from hermes_cli.kanban_db import (
    DEFAULT_BOARD,
    HolderPresence,
    _worker_identity_presence,
    board_dir,
    kanban_home,
    list_boards,
)
from hermes_cli.managed_uv import _default_live_venv, _venv_python
from hermes_cli.profiles import profiles_to_serve
from hermes_cli.release_guards import OpenRun
from hermes_constants import get_default_hermes_root

# S-D: the configuration set is these files of the root home and of every served profile home.
CONFIG_FILES = ("config.yaml", ".env")

_ORIGIN_MAIN = "refs/heads/main"
_TIMEOUT_SECONDS = 60
# Builder note: an open run is a run row still marked running, with no end time.
_OPEN_RUNS_SQL = (
    "SELECT id, worker_pid, worker_start_time FROM task_runs "
    "WHERE status = 'running' AND ended_at IS NULL ORDER BY id"
)
# Bytes 18 and 19 of a SQLite file are its write and read format versions; 2 means WAL.
_WAL_FORMAT = b"\x02\x02"
_SIGNAL_NAMES = {sig.value: sig.name for sig in signal.Signals}
# Run by the live venv's interpreter: where it would import hermes_cli from, without importing it.
_IMPORT_ROOT_PROBE = (
    "import importlib.util, os; "
    "origin = importlib.util.find_spec('hermes_cli').origin; "
    "print(os.path.dirname(os.path.dirname(os.path.realpath(origin))))"
)


class LiveHostReader:
    """The release host as the guards and the readbacks see it. Every method only reads.

    The keywords are the ones S-E names, plus `snapshot_name`: the NAME of S-C, which is NEW,
    the merged `Pins.snapshot`. `named_config_snapshot` takes no argument, so the reader has to
    be told which configuration snapshot is the named one.
    """

    def __init__(
        self,
        *,
        checkout: Path,
        root_home: Path,
        units: Mapping[str, str],
        snapshot_root: Path,
        snapshot_name: str,
    ) -> None:
        self.checkout = Path(checkout)
        self.root_home = Path(root_home)
        self.units = dict(units)
        self.snapshot_root = Path(snapshot_root)
        self.snapshot_name = snapshot_name

    # The checkout and its history.

    def checkout_head(self) -> str:
        return self._git("rev-parse", "--verify", "HEAD").stdout.strip()

    def checkout_is_clean(self) -> bool:
        # Untracked files count; ignored ones, such as the live venv, do not.
        return self._git("status", "--porcelain", "--untracked-files=normal").stdout == ""

    def origin_main(self) -> str:
        # S-H: asked of the remote itself, so a stale origin/main cannot answer for it.
        for line in self._git("ls-remote", "origin", _ORIGIN_MAIN).stdout.splitlines():
            commit, _, ref = line.partition("\t")
            if ref == _ORIGIN_MAIN:
                return commit
        raise LookupError(f"origin has no {_ORIGIN_MAIN}")

    def first_parent_chain(self, prev: str, new: str) -> Sequence[str]:
        argv = ("rev-list", "--first-parent", "--end-of-options", new, f"^{prev}")
        return self._git(*argv).stdout.split()

    def commit_parents(self, commit: str) -> tuple[str, ...]:
        line = self._git("rev-list", "--parents", "--no-walk", "--end-of-options", commit).stdout
        return tuple(line.split()[1:])

    def commit_tree(self, commit: str) -> str:
        argv = ("rev-parse", "--verify", "--end-of-options", f"{commit}^{{tree}}")
        return self._git(*argv).stdout.strip()

    def is_ancestor(self, ancestor: str, descendant: str) -> bool:
        argv = ("merge-base", "--is-ancestor", "--end-of-options", ancestor, descendant)
        return self._git(*argv, accept=(0, 1)).returncode == 0

    def changed_paths(self, prev: str, new: str) -> Collection[str]:
        argv = ("diff-tree", "-r", "--name-only", "--no-renames", "-z", "--end-of-options", prev, new)
        return frozenset(path for path in self._git(*argv).stdout.split("\0") if path)

    def checkout_root(self) -> str:
        return str(Path(self._git("rev-parse", "--show-toplevel").stdout.strip()).resolve())

    def venv_import_root(self) -> str:
        # The live venv as runtime repair finds it: venv when it holds an interpreter, else
        # .venv. -I keeps this process's environment and working directory out of the answer.
        python = _venv_python(_default_live_venv(self.checkout))
        done = _run([str(python), "-I", "-B", "-c", _IMPORT_ROOT_PROBE])
        return str(Path(done.stdout.strip()).resolve())

    # Units, disk and snapshots.

    def unit_property(self, unit: str, name: str) -> str:
        # S-E: units are driven, and read, only through the user manager.
        argv = ["systemctl", "--user", "show", f"--property={name}", "--", self.units[unit]]
        key, _, value = _run(argv).stdout.strip().partition("=")
        if key != name:
            raise LookupError(f"systemctl reported no {name} for the {unit} unit")
        if name.endswith("Signal") and value.isdigit():
            return _SIGNAL_NAMES.get(int(value), value)
        return value

    def free_disk_bytes(self) -> int:
        # Before the first release the snapshot root may not exist yet; its filesystem is then
        # the one of the nearest directory above it that does.
        snapshots = next(
            path for path in (self.snapshot_root, *self.snapshot_root.parents) if path.exists()
        )
        return min(shutil.disk_usage(path).free for path in (self.checkout, snapshots))

    def snapshots(self) -> Sequence[str]:
        state = self.snapshot_root / "state"  # S-C
        if not state.is_dir():
            return []
        entries = sorted(state.iterdir(), key=lambda entry: (entry.lstat().st_mtime_ns, entry.name))
        return [entry.name for entry in entries]

    # Runs.

    def open_native_runs(self) -> Sequence[OpenRun]:
        home = self._require_root(kanban_home())
        runs: list[OpenRun] = []
        # Builder note: boards come from the kanban board listing.
        for board in list_boards(include_archived=True):
            slug = board["slug"]
            # Not board["db_path"] or kanban_db_path(): both honour HERMES_KANBAN_DB, which the
            # dispatcher pins to a worker's own board, so under a worker's environment they name
            # that one file for every board. The same rule as release_ledger.ledger_path.
            path = home / "kanban.db" if slug == DEFAULT_BOARD else board_dir(slug) / "kanban.db"
            runs.extend(_open_runs(slug, path))
        return runs

    # Configuration.

    def live_config(self) -> Mapping[str, str]:
        root = self._require_root(get_default_hermes_root())
        files = {}
        # S-D: the profile homes the gateway serves, as hermes_cli/AGENTS.md names them.
        for _profile, home in profiles_to_serve(multiplex=True):
            for name in CONFIG_FILES:
                files[(home / name).relative_to(root).as_posix()] = home / name
        return _digests(files)

    def named_config_snapshot(self) -> Mapping[str, str]:
        base = self.snapshot_root / "config" / self.snapshot_name  # S-C
        files = {
            path.relative_to(base).as_posix(): path
            for path in base.rglob("*")
            if path.name in CONFIG_FILES
        }
        return _digests(files)

    def _git(self, *argv: str, accept: Collection[int] = (0,)) -> subprocess.CompletedProcess:
        return _run(["git", "-C", str(self.checkout), "--no-optional-locks", *argv], accept=accept)

    def _require_root(self, found: Path) -> Path:
        # The board listing and the profile scan find the root through this process's
        # environment. A root other than the one this reader was built for would answer for
        # another platform, so it is refused.
        if found.resolve() != self.root_home.resolve():
            raise RuntimeError(f"this process resolves the Hermes root to {found}, not {self.root_home}")
        return found


def _run(argv: Sequence[str], *, accept: Collection[int] = (0,)) -> subprocess.CompletedProcess:
    done = subprocess.run(
        list(argv),
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="surrogateescape",
        timeout=_TIMEOUT_SECONDS,
        check=False,
    )
    if done.returncode not in accept:
        # stderr stays out: the runner keeps this message as release evidence, and a remote's
        # error text can carry a URL with a credential in it.
        raise subprocess.CalledProcessError(done.returncode, list(argv))
    return done


def _digests(files: Mapping[str, Path]) -> dict[str, str]:
    # S-D: only a file's SHA-256 leaves this module; its content is never logged or returned.
    return {
        key: hashlib.sha256(path.read_bytes()).hexdigest()
        for key, path in sorted(files.items())
        if path.is_file()
    }


def _open_runs(slug: str, path: Path) -> list[OpenRun]:
    if not path.exists():
        return []
    try:
        with closing(_read_only(path)) as conn:
            rows = conn.execute(_OPEN_RUNS_SQL).fetchall()
    except (OSError, sqlite3.DatabaseError):
        # F18: the dispatcher's host-cap count skips a board it cannot read and so fails open.
        # The drain fails closed: a board that cannot be read may hold a live run, so it counts
        # as one.
        return [OpenRun(f"{slug}:unreadable", True)]
    # Builder note: liveness is the kernel's own process check, which reads the recorded
    # (pid, start time) identity. Only a worker proven absent is dead; INDETERMINATE counts as
    # live, as the kernel itself treats it.
    return [
        OpenRun(
            f"{slug}:{run_id}",
            _worker_identity_presence(pid, start) is not HolderPresence.PROVABLY_ABSENT,
        )
        for run_id, pid, start in rows
    ]


def _read_only(path: Path) -> sqlite3.Connection:
    # A plain read-only open of a WAL database whose last writer closed cleanly creates its -wal
    # and -shm files and leaves them behind. With no -wal file there is nothing to replay, so
    # such a file is opened immutable, which creates nothing.
    with path.open("rb") as handle:
        header = handle.read(20)
    immutable = header[18:20] == _WAL_FORMAT and not Path(f"{path}-wal").exists()
    mode = "mode=ro&immutable=1" if immutable else "mode=ro"
    return sqlite3.connect(f"{path.resolve().as_uri()}?{mode}", uri=True)
