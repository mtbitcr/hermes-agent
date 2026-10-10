"""The live host, read for the release guards (release card 2).

`LiveHostReader` is the real `HostReader` of `hermes_cli.release_guards`. It reads:
- the checkout and its history, with origin main read from the remote itself (S-H);
- the code identity each live gateway stamps, for G6;
- systemd unit properties through the user manager, free disk, and the state snapshots' names
  and sizes;
- the open runs on every board;
- digests of the configuration set (S-D), live and in the named configuration snapshot (S-C).

Every method only reads. Moving the checkout, fetching objects and taking or restoring snapshots
are actions, and they belong to card 3. Reading also leaves nothing behind, which is what lets
prepare mode keep its promise to write nothing: git runs with --no-optional-locks, so `git status`
does not refresh the index; every git read but `git ls-remote` runs with no transport allowed, so
a partial clone fetches no missing object; origin main comes from `git ls-remote`, which updates
no local ref; and boards are read in a child interpreter of their own, so that SQLite creates no
file and changes none, its -shm included, and this process's own board connections keep their
locks.

Like the runner (K6), this module imports everything when it loads and no method imports
anything: the readbacks after the stop call the checkout, gateway and configuration reads while
the checkout on disk is being swapped. So the Hermes modules that the fleet scan of
`hermes_cli.update_receipt` imports inside its functions are imported here too.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import signal
import stat
import subprocess
import sys
from collections.abc import Collection, Mapping, Sequence
from pathlib import Path

# K6: the fleet scan imports these inside its functions.
import gateway.control_socket  # noqa: F401
import gateway.status  # noqa: F401
import hermes_cli.build_info  # noqa: F401
from hermes_cli.kanban_db import (
    DEFAULT_BOARD,
    HolderPresence,
    _worker_identity_presence,
    board_dir,
    kanban_home,
    list_boards,
)
from hermes_cli.profiles import profiles_to_serve
from hermes_cli.release_guards import OpenRun
from hermes_cli.update_receipt import collect_fleet_versions
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
# Boards are read by a child interpreter (-I -B, standard library only), given the query and the
# board files. A descriptor on a live board never opens in this process: closing it would drop
# every SQLite lock this process's own connections hold on the file, and a connection here would
# share their -shm and write its read marks. The child prints, per file, its rows, or null for a
# board it cannot read; a listed board without its file is one, as a missing file is no proof of
# an empty board. Before its header is read, a file is held under a read lock on the bytes
# SQLite's shared lock covers, an open file description lock kept until SQLite has closed it: it
# keeps every connection from the exclusive lock it needs to commit in rollback mode, to change
# the journal mode either way, and to fold a -wal back and remove it, so the header read holds for
# the whole read. A file whose header is not SQLite 3 in rollback or WAL format never reaches
# SQLite, which deletes a -wal beside an empty file. A WAL file with no -wal has nothing to replay
# and is opened immutable, which opens no side file; a -wal there after that read was created
# during it and fails the read. Otherwise readonly_shm reads the -wal's commits and writes to
# neither file, and a missing -shm fails the read. The one file SQLite opens must be the very file
# that was locked and checked, so the descriptors the child gains in opening it must name that
# file and no other.
_BOARD_READ = r"""
import fcntl, json, os, sqlite3, struct, sys
from pathlib import Path
LOCK = struct.pack("hhqqi4x", fcntl.F_RDLCK, os.SEEK_SET, 0x40000002, 510, 0)
def descriptors():
    found = {}
    for number in os.listdir("/proc/self/fd"):
        try:
            info = os.fstat(int(number))
        except OSError:
            continue
        found[number] = (info.st_dev, info.st_ino)
    return found
def rows(path):
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return None
    try:
        fcntl.fcntl(fd, getattr(fcntl, "F_OFD_SETLK", 37), LOCK)  # Python 3.11 has no name for it; 37 on Linux
        header, info = os.pread(fd, 100, 0), os.fstat(fd)
        if len(header) < 100 or header[:16] != b"SQLite format 3\0" or header[18:20] not in (b"\1\1", b"\2\2"):
            return None
        immutable = header[18:20] == b"\2\2" and not os.path.exists(path + "-wal")
        mode = "immutable=1" if immutable else "readonly_shm=1"
        before = descriptors()
        conn = sqlite3.connect(f"{Path(path).as_uri()}?mode=ro&{mode}", uri=True)
        try:
            gained = {file for number, file in descriptors().items() if before.get(number) != file}
            if gained != {(info.st_dev, info.st_ino)}:
                return None
            found = conn.execute(sys.argv[1]).fetchall()
        finally:
            conn.close()
        return None if immutable and os.path.exists(path + "-wal") else found
    except (OSError, sqlite3.Error):
        return None
    finally:
        os.close(fd)
print(json.dumps([rows(os.path.realpath(path)) for path in sys.argv[2:]]))
"""
_SIGNAL_NAMES = {sig.value: sig.name for sig in signal.Signals}


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
        # Owner rule for G6: the running gateways answer. The fleet scan reads each gateway's
        # control-socket identity, or its live gateway_state.json record, in the root home and in
        # every named profile home. The checkout is the live import root only when a gateway is
        # live and every live gateway is stamped with the checkout's head; no live gateway, or a
        # stale, unknown or missing stamp, gives "", so G6 fails closed. A row's own state is not
        # used: the scan computes it against this process's source tree, which need not be the
        # checkout.
        self._require_root(get_default_hermes_root())
        head, rows = self.checkout_head(), collect_fleet_versions()
        if rows and all(row.get("code_sha") == head for row in rows):
            return self.checkout_root()
        return ""

    # Units, disk and snapshots.

    def unit_property(self, unit: str, name: str) -> str:
        # S-E: units are driven, and read, only through the user manager.
        argv = ["systemctl", "--user", "show", f"--property={name}", "--", self.units[unit]]
        values = []
        for line in _run(argv).stdout.splitlines():
            key, _, value = line.partition("=")
            if key != name:
                raise LookupError(f"systemctl reported no {name} for the {unit} unit")
            values.append(value)
        if len(values) != 1:
            raise LookupError(f"systemctl reported no single {name} for the {unit} unit")
        if name.endswith("Signal") and values[0].isdigit():
            return _SIGNAL_NAMES.get(int(values[0]), values[0])
        return values[0]

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

    def snapshot_bytes(self, name: str) -> int:
        # Only regular files count. Every path, the snapshot's own included, is read with lstat, so
        # no link is followed, whether it points to a file or to a directory.
        total, paths = 0, [self.snapshot_root / "state" / name]  # S-C
        while paths:
            path = paths.pop()
            info = path.lstat()
            if stat.S_ISDIR(info.st_mode):
                paths.extend(path.iterdir())
            elif stat.S_ISREG(info.st_mode):
                total += info.st_size
        return total

    # Runs.

    def open_native_runs(self) -> Sequence[OpenRun]:
        home = self._require_root(kanban_home())
        slugs, paths = [], []
        # Builder note: boards come from the kanban board listing.
        for board in list_boards(include_archived=True):
            slugs.append(board["slug"])
            # Not board["db_path"] or kanban_db_path(): both honour HERMES_KANBAN_DB, which the
            # dispatcher pins to a worker's own board, so under a worker's environment they name
            # that one file for every board. The same rule as release_ledger.ledger_path.
            paths.append(home / "kanban.db" if slugs[-1] == DEFAULT_BOARD else board_dir(slugs[-1]) / "kanban.db")
        runs: list[OpenRun] = []
        for slug, rows in zip(slugs, _board_rows(paths)):
            runs.extend(_open_runs(slug, rows))
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
        # An empty GIT_ALLOW_PROTOCOL allows no transport. In a partial clone, a read that needs a
        # missing object then fails before it reaches the promisor remote, instead of fetching
        # the object: fetching is card 3's (S-H). ls-remote alone asks the remote, and it writes
        # no object and no ref.
        env = None if argv[0] == "ls-remote" else {**os.environ, "GIT_ALLOW_PROTOCOL": ""}
        return _run(["git", "-C", str(self.checkout), "--no-optional-locks", *argv], accept=accept, env=env)

    def _require_root(self, found: Path) -> Path:
        # The board listing, the profile scan and the fleet scan find the root through this
        # process's environment. A root other than the one this reader was built for would answer
        # for another platform, so it is refused.
        if found.resolve() != self.root_home.resolve():
            raise RuntimeError(f"this process resolves the Hermes root to {found}, not {self.root_home}")
        return found


def _run(
    argv: Sequence[str],
    *,
    accept: Collection[int] = (0,),
    env: Mapping[str, str] | None = None,
) -> subprocess.CompletedProcess:
    done = subprocess.run(
        list(argv),
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="surrogateescape",
        timeout=_TIMEOUT_SECONDS,
        check=False,
        env=env,
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


def _board_rows(paths: Sequence[Path]) -> list:
    # Any failure of the child, its timeout, its exit status or output that does not parse,
    # leaves every board unread.
    argv = [sys.executable, "-I", "-B", "-c", _BOARD_READ, _OPEN_RUNS_SQL, *map(str, paths)]
    try:
        found = json.loads(_run(argv).stdout)
        if len(found) == len(paths) and all(rows is None or all(len(row) == 3 for row in rows) for rows in found):
            return found
    except (OSError, TypeError, ValueError, subprocess.SubprocessError):
        pass
    return [None] * len(paths)


def _open_runs(slug: str, rows: list | None) -> list[OpenRun]:
    if rows is None:
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
