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
import os
import re
import shutil
import subprocess
import tomllib  # noqa: F401  (K6: build_info loads it to read the version)
import urllib.parse
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

import psutil  # noqa: F401  (K6: gateway.status loads it to ask about a process)

import gateway.control_socket  # noqa: F401  (K6: update_receipt loads it to ask a gateway)
import gateway.status  # noqa: F401  (K6: update_receipt loads it to read a gateway's record)
import hermes_cli.build_info  # noqa: F401  (K6: update_receipt loads it for the code identity)
import hermes_cli.sqlite_safe_read  # noqa: F401  (K6: backup.py loads it when a copy fails)
from hermes_cli.backup import (
    _copy_quick_snapshot_files,
    _iter_backup_files,
    _quick_snapshot_candidates,
    _safe_copy_db,
)
from hermes_cli.profiles import _PROFILE_ID_RE
from hermes_cli.update_receipt import collect_fleet_versions
from hermes_constants import get_default_hermes_root, named_profile_is_deleted
from utils import atomic_replace

READBACK_TIMEOUT_SECONDS = 10.0
# S-D: the configuration files of every served home.
CONFIG_FILES = ("config.yaml", ".env")
# S-C: each kind of snapshot has its own directory under the snapshot root.
STATE, CONFIG = "state", "config"

_FULL_SHA = re.compile(r"[0-9a-f]{40}")
_SNAPSHOT_NAME = re.compile(r"[0-9A-Za-z][0-9A-Za-z._-]*")


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs) -> None:
        return None  # another page, such as a login page, is not the answer: an error status


# The readbacks ask the host's own addresses, so no proxy from the environment stands between,
# and the answer must come from the address itself.
_DIRECT = urllib.request.build_opener(urllib.request.ProxyHandler({}), _NoRedirect())


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
        """Copy every application database and file into state/NAME with backup.py's copiers.

        That is each served home's quick-snapshot set and every other database a full backup of
        the home holds, such as the release record, the board registry and the cron queues. The
        online copy reads each database through its write-ahead log, so the copy holds rows not
        yet checkpointed. A file that is not copied fails the snapshot before it is published, and
        only the owner can read the copies, whatever the umask or the snapshot root's access.
        """

        def fill(staging: Path) -> None:
            missing: list[str] = []
            for home in self._served_homes():
                rel = home.relative_to(self.root_home)
                wanted = [sub for _, sub, _ in _quick_snapshot_candidates(home)]
                copied, _, _ = _copy_quick_snapshot_files(home, staging / rel, None)
                missing += [(rel / sub).as_posix() for sub in wanted if sub not in copied]
                for store in self._other_databases(home, set(wanted)):
                    copy = staging / rel / store.relative_to(home)
                    copy.parent.mkdir(parents=True, exist_ok=True)
                    if not _safe_copy_db(store, copy):
                        missing.append(copy.relative_to(staging).as_posix())
            if missing:
                raise RuntimeError(f"could not copy {', '.join(missing)}")
            for path in staging.rglob("*"):
                path.chmod(0o700 if path.is_dir() else 0o600)

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
        """Put back the saved configuration set (S-D): the same files with the same bytes.

        Each saved file comes back by one atomic replace, into a private directory made again when
        it is gone. A configuration file of a served home that the save does not hold is removed.
        A profile deleted since the save, every other file and the databases stay as they are.
        """
        if not self.config_snapshot:
            raise RuntimeError("no configuration snapshot was saved")
        saved = self._named(CONFIG, self.config_snapshot)
        if not saved.is_dir():
            raise FileNotFoundError(f"the configuration snapshot {self.config_snapshot} is gone")
        kept = set()
        for copy in sorted(filter(Path.is_file, saved.rglob("*"))):
            live = self.root_home / copy.relative_to(saved)
            kept.add(live)
            if live.parent != self.root_home and named_profile_is_deleted(live.parent):
                continue
            live.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            temporary = live.with_name(f".{live.name}.release-restore")
            shutil.copy2(copy, temporary)
            atomic_replace(temporary, live)
        for rel in self._config_files():
            if self.root_home / rel not in kept:
                (self.root_home / rel).unlink(missing_ok=True)  # created after the save

    def _write_snapshot(
        self, kind: str, name: str, fill: Callable[[Path], None], *, replace: bool
    ) -> None:
        """Fill a staging directory, then rename it to the name, so no name holds half a snapshot.

        The staging directory sits outside state/ and config/, where no snapshot listing sees it,
        and is private before anything is copied into it. An earlier snapshot of the name is set
        aside beside it until the rename is done, and comes back when the rename fails.
        """
        target = self._named(kind, name)
        if target.exists() and not replace:
            raise FileExistsError(f"the {kind} snapshot {name} already exists")
        staging = self.snapshot_root / f".{kind}-{name}.partial"
        earlier = self.snapshot_root / f".{kind}-{name}.previous"
        shutil.rmtree(staging, ignore_errors=True)
        self.snapshot_root.mkdir(mode=0o700, parents=True, exist_ok=True)
        target.parent.mkdir(mode=0o700, exist_ok=True)
        staging.mkdir(mode=0o700)
        try:
            fill(staging)
            set_aside = replace and target.exists()
            if set_aside:
                shutil.rmtree(earlier, ignore_errors=True)
                os.replace(target, earlier)
            try:
                os.replace(staging, target)
            except OSError:
                if set_aside:
                    os.replace(earlier, target)
                raise
            shutil.rmtree(earlier, ignore_errors=True)
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

    def _other_databases(self, home: Path, quick: set[str]) -> list[Path]:
        """Every *.db that backup.py's full-backup walk keeps in the home, beyond its quick set.

        Left out are the profiles under the root, each served one walked as its own home, the task
        workspaces and attachments, which hold the tasks' files rather than the platform's stores,
        and the checkout and the release snapshots themselves.
        """
        apart = [self.snapshot_root.resolve(), self.checkout_path.resolve()]
        return sorted(
            path
            for path, rel in _iter_backup_files(home, self.snapshot_root)
            if path.suffix == ".db"
            and rel.as_posix() not in quick
            and not (home == self.root_home and rel.parts[0] == "profiles")
            and {"workspaces", "attachments"}.isdisjoint(rel.parts)
            and not any(path.resolve().is_relative_to(place) for place in apart)
        )

    # Readbacks.

    def health_ok(self) -> bool:
        return _answers_ok(self.health_url)

    def workspace_reads_ok(self) -> bool:
        return _answers_ok(self.workspace_check_url)

    def fleet_version(self) -> str:
        """The one code version every live gateway reports, or "" when that is not so.

        collect_fleet_versions (hermes_cli/update_receipt.py) asks the gateways of this process's
        root through their own records, the control socket or gateway_state.json, and calls one
        current only when its stamp is the checkout's own code identity. A stale, unknown or
        missing identity, no live gateway, or a root other than this host's reads "".
        """
        if get_default_hermes_root().resolve() != self.root_home.resolve():
            return ""
        versions = {
            row.get("code_sha") if row.get("state") == "current" else None
            for row in collect_fleet_versions()
        }
        version = versions.pop() if len(versions) == 1 else None
        return version if isinstance(version, str) and _FULL_SHA.fullmatch(version) else ""


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
    """A 2xx answer from the address itself within the timeout. No address or one that is not
    HTTP, a redirect, an error status, a timeout or a refused connection reads False."""
    if not url:
        return False
    try:
        if urllib.parse.urlsplit(url).scheme not in ("http", "https"):
            return False
        with _DIRECT.open(url, timeout=READBACK_TIMEOUT_SECONDS) as answer:
            return 200 <= answer.status < 300
    except (OSError, ValueError, http.client.HTTPException):
        return False
