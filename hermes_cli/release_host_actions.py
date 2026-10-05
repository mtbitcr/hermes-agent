"""The acting half of the release host: the units, the checkout, snapshots and the readbacks.

The merged runner (hermes_cli/release_runner.py) drives a `ReleaseHost`. Card 2's reader answers
the guards' reads; this class takes the actions and runs the readbacks the runner adds on top.
Every move is one process call whose exit status is checked: `systemctl --user` for the units,
which only the user manager drives (S-E), and git for the checkout (S-H). Snapshots live under one
release snapshot directory, state snapshots as state/NAME and configuration snapshots as
config/NAME (S-C).

The actions raise when the host refuses them, so the runner's failure rule takes the host back.
The readbacks answer False instead, as the runner's protocol asks.

Design rule (K6, from the runner): every import happens when this module loads and no function
imports anything, because it runs while the checkout under it is being swapped.
"""

from __future__ import annotations

import contextlib
import encodings.idna  # noqa: F401  (K6: sockets load it at their first host name)
import http.client
import json
import os
import re
import shutil
import subprocess
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

import hermes_cli.sqlite_safe_read  # noqa: F401  (K6: backup.py loads it when a copy fails)
from hermes_cli.backup import _copy_quick_snapshot_files
from hermes_cli.profiles import _PROFILE_ID_RE
from hermes_constants import named_profile_is_deleted
from utils import atomic_replace

READBACK_TIMEOUT_SECONDS = 10.0
# S-D: the configuration files of every served home.
CONFIG_FILES = ("config.yaml", ".env")
# S-C: each kind of snapshot has its own directory under the snapshot root.
STATE, CONFIG = "state", "config"

_FULL_SHA = re.compile(r"[0-9a-f]{40}")
_SNAPSHOT_NAME = re.compile(r"[0-9A-Za-z][0-9A-Za-z._-]*")
# The readbacks ask the host's own addresses, so no proxy from the environment stands between.
_DIRECT = urllib.request.build_opener(urllib.request.ProxyHandler({}))


class ReleaseHostActions:
    """The actions and readbacks of `ReleaseHost` on the live host, from explicit settings (S-E).

    `units` maps the roles gateway, serve and sandbox-tunnel to unit names. A role without a unit
    name is never driven and reads back False, as does an empty address.
    """

    def __init__(
        self,
        *,
        checkout: str | Path,
        root_home: str | Path,
        units: Mapping[str, str],
        snapshot_root: str | Path,
        health_url: str,
        workspace_check_url: str,
    ) -> None:
        self.checkout_path = Path(checkout)  # `checkout` is the move below
        self.root_home = Path(root_home)
        self.units = dict(units)
        self.snapshot_root = Path(snapshot_root)
        self.health_url = health_url
        self.workspace_check_url = workspace_check_url
        # The configuration snapshot this release saved: the one the restore puts back.
        self.config_snapshot = ""

    # Units, only through the user manager.

    def stop_units(self, units: Sequence[str]) -> None:
        self._drive("stop", units)

    def start_units(self, units: Sequence[str]) -> None:
        self._drive("start", units)

    def _drive(self, verb: str, roles: Sequence[str]) -> None:
        names = [self._unit(role) for role in roles]  # an unset role refuses before any call
        for name in names:
            _run(["systemctl", "--user", verb, name])

    def _unit(self, role: str) -> str:
        if not self.units.get(role):
            raise ValueError(f"no unit name is set for the {role} role")
        return self.units[role]

    def unit_active(self, unit: str) -> bool:
        """The user manager answers active for the role's unit. Reloading or failed is not."""
        name = self.units.get(unit)
        if not name:
            return False
        try:
            answer = _run(
                ["systemctl", "--user", "is-active", name], timeout=READBACK_TIMEOUT_SECONDS
            )
        except (OSError, RuntimeError, subprocess.SubprocessError):
            return False  # every state but active and reloading exits nonzero
        return answer.stdout.strip() == "active"

    # The checkout (S-H).

    def fetch(self, commit: str) -> None:
        """Bring the commit's objects into the object store. The head and working tree stay."""
        self._git("fetch", "--no-tags", "origin", _full_sha(commit))

    def checkout(self, commit: str) -> None:
        """Move the checkout to exactly the commit, with a detached head [U14].

        Without --force no local change is ever dropped: git carries it over or refuses. The
        guards found the checkout clean before the stop.
        """
        self._git("checkout", "--detach", _full_sha(commit))

    def _git(self, *args: str) -> None:
        _run(["git", "-C", str(self.checkout_path), *args])

    # Snapshots (S-C).

    def take_snapshot(self, name: str) -> None:
        """Copy every served home's databases and files into state/NAME with backup.py's copier.

        Its online copy reads each database through its write-ahead log, so the copy holds rows
        not yet checkpointed. A database it cannot copy fails the snapshot.
        """

        def fill(staging: Path) -> None:
            failed: list[str] = []
            for home in self._served_homes():
                rel = home.relative_to(self.root_home)
                _, failed_dbs, _ = _copy_quick_snapshot_files(home, staging / rel, None)
                failed += [(rel / db).as_posix() for db in failed_dbs]
            if failed:
                raise RuntimeError(f"could not copy {', '.join(failed)}")

        self._write_snapshot(STATE, name, fill, replace=False)

    def delete_snapshot(self, name: str) -> None:
        with contextlib.suppress(FileNotFoundError):
            shutil.rmtree(self._named(STATE, name))

    def save_config_snapshot(self, name: str) -> None:
        """Save the configuration set (S-D) as config/NAME, replacing an earlier save of NAME."""

        def fill(staging: Path) -> None:
            for rel in self._config_files():
                (staging / rel).parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(self.root_home / rel, staging / rel)

        self._write_snapshot(CONFIG, name, fill, replace=True)
        self.config_snapshot = name

    def restore_config(self) -> None:
        """Put back each file of the saved configuration snapshot, each by one atomic replace.

        A live file the snapshot does not hold is left alone.
        """
        if not self.config_snapshot:
            raise RuntimeError("no configuration snapshot was saved")
        saved = self._named(CONFIG, self.config_snapshot)
        if not saved.is_dir():
            raise FileNotFoundError(f"the configuration snapshot {self.config_snapshot} is gone")
        for copy in sorted(filter(Path.is_file, saved.rglob("*"))):
            live = self.root_home / copy.relative_to(saved)
            temporary = live.with_name(f".{live.name}.release-restore")
            shutil.copy2(copy, temporary)
            atomic_replace(temporary, live)

    def _write_snapshot(
        self, kind: str, name: str, fill: Callable[[Path], None], *, replace: bool
    ) -> None:
        """Fill a staging directory, then rename it to the name, so no name holds half a snapshot.

        The staging directory sits outside state/ and config/, where no snapshot listing sees it.
        """
        target = self._named(kind, name)
        if target.exists() and not replace:
            raise FileExistsError(f"the {kind} snapshot {name} already exists")
        staging = self.snapshot_root / f".{kind}-{name}.partial"
        shutil.rmtree(staging, ignore_errors=True)
        staging.mkdir(parents=True)
        try:
            fill(staging)
            if replace:
                shutil.rmtree(target, ignore_errors=True)
            target.parent.mkdir(exist_ok=True)
            os.replace(staging, target)
        finally:
            shutil.rmtree(staging, ignore_errors=True)  # already gone once renamed

    def _named(self, kind: str, name: str) -> Path:
        if not _SNAPSHOT_NAME.fullmatch(name):
            raise ValueError(f"not a snapshot name: {name!r}")
        return self.snapshot_root / kind / name

    def _served_homes(self) -> list[Path]:
        """The root home and the profile homes the gateway serves: profiles_to_serve with
        multiplex on (S-D), read under this host's root rather than this process's home."""
        profiles = self.root_home / "profiles"
        named = sorted(profiles.iterdir()) if profiles.is_dir() else []
        return [self.root_home] + [
            home
            for home in named
            if home.is_dir()
            and home.name != "default"
            and _PROFILE_ID_RE.match(home.name)
            and not named_profile_is_deleted(home)
        ]

    def _config_files(self) -> list[str]:
        """The S-D key set: each configuration file's path relative to the root home."""
        return [
            (home / name).relative_to(self.root_home).as_posix()
            for home in self._served_homes()
            for name in CONFIG_FILES
            if (home / name).is_file()
        ]

    # Readbacks.

    def health_ok(self) -> bool:
        return _answers_ok(self.health_url)

    def workspace_reads_ok(self) -> bool:
        return _answers_ok(self.workspace_check_url)

    def fleet_version(self) -> str:
        """The one code version every live gateway stamped, or "" when none or several do.

        It reads each served home's gateway_state.json itself: the fleet helper in
        hermes_cli/update_receipt.py imports lazily, which K6 rules out here.
        """
        versions = {record.get("code_sha") or "" for record in self._live_gateways()}
        return versions.pop() if len(versions) == 1 else ""

    def _live_gateways(self) -> list[dict]:
        records = []
        for home in self._served_homes():
            try:
                record = json.loads((home / "gateway_state.json").read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue  # no gateway stamped this home, or the stamp is unreadable
            if isinstance(record, dict) and _alive(record.get("pid")):
                records.append(record)
        return records


def _run(argv: list[str], **options) -> subprocess.CompletedProcess[str]:
    """One process call. A nonzero exit raises with what the process wrote to stderr."""
    done = subprocess.run(
        argv, capture_output=True, encoding="utf-8", errors="replace", check=False, **options
    )
    if done.returncode:
        raise RuntimeError(f"{' '.join(argv)} exited {done.returncode}: {done.stderr.strip()}")
    return done


def _full_sha(commit: str) -> str:
    """Only a full commit name moves the host: a short or symbolic one could name another."""
    if not _FULL_SHA.fullmatch(commit):
        raise ValueError(f"not a full commit name: {commit!r}")
    return commit


def _answers_ok(url: str) -> bool:
    """A 2xx answer within the timeout. No address, an error status, a timeout or a refused
    connection reads False."""
    if not url:
        return False
    try:
        with _DIRECT.open(url, timeout=READBACK_TIMEOUT_SECONDS) as answer:
            return 200 <= answer.status < 300
    except (OSError, ValueError, http.client.HTTPException):
        return False


def _alive(pid: object) -> bool:
    """A positive process id that names a running process. Signal 0 only asks."""
    if type(pid) is not int or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except PermissionError:
        return True  # it runs, as another user
    except (OSError, OverflowError):
        return False
    return True
