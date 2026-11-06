"""The acting half of the release host: the units, the checkout, snapshots and the readbacks.

The merged runner (hermes_cli/release_runner.py) drives a `ReleaseHost`. Card 2's reader answers
the guards' reads; this class takes the actions and runs the readbacks the runner adds on top.
Every move is one process call whose exit status is checked: `systemctl --user` for the units,
which only the user manager drives (S-E), git for the checkout (S-H), and uv, from the Hermes
home's bin folder, for the platform environment. Snapshots live under one release snapshot
directory, state snapshots as state/NAME, configuration snapshots as config/NAME (S-C) and
environment snapshots as env/NAME. A state snapshot copies the files that the full backup's own
walk yields for the root home (hermes_cli/backup.py). The platform environment is a link to one
real folder (D2): its snapshot copies that folder, the sync changes it in place through the link,
and the restore moves the copy back into the same folder, so the link never changes.

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
import stat
import subprocess
import tomllib  # noqa: F401  (K6: build_info loads it to read the version)
import urllib.parse
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

import psutil  # noqa: F401  (K6: gateway.status loads it to ask about a process)
from packaging.requirements import Requirement
from packaging.version import Version

import gateway.control_socket  # noqa: F401  (K6: update_receipt loads it to ask a gateway)
import gateway.status  # noqa: F401  (K6: update_receipt loads it to read a gateway's record)
import hermes_cli.build_info  # noqa: F401  (K6: update_receipt loads it for the code identity)
from hermes_cli.backup import _SQLITE_HEADER, _iter_backup_files, _safe_copy_db
from hermes_cli.managed_uv import resolve_uv
from hermes_cli.profiles import _PROFILE_ID_RE
from hermes_cli.release_runner import PLATFORM_UNITS, EnvironmentUnreadable
from hermes_cli.sqlite_safe_read import has_live_connection, read_header_bytes_preopen
from hermes_cli.update_receipt import collect_fleet_versions
from hermes_constants import get_default_hermes_root, named_profile_is_deleted
from utils import atomic_replace

READBACK_TIMEOUT_SECONDS = 10.0
# S-D: the configuration files of every served home.
CONFIG_FILES = ("config.yaml", ".env")
# S-C: each kind of snapshot has its own directory under the snapshot root.
STATE, CONFIG, ENVIRONMENT = "state", "config", "env"
# D3: the extras the sync installs; the default groups never.
SYNC_EXTRAS = ("all",)
SYNC_TIMEOUT_SECONDS = 900
ENVIRONMENT_READ_TIMEOUT_SECONDS = 60

_FULL_SHA = re.compile(r"[0-9a-f]{40}")
_SNAPSHOT_NAME = re.compile(r"[0-9A-Za-z][0-9A-Za-z._-]*")


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs) -> None:
        return None  # another page, such as a login page, is not the answer: an error status


# The health readback asks the host's own address, so no proxy from the environment stands
# between, and the answer must come from the address itself.
_DIRECT = urllib.request.build_opener(urllib.request.ProxyHandler({}), _NoRedirect())
# R6's two reads, run by node inside the Workspace container as the manual release runs them: with
# the platform address and access file the container's environment names, the projects list, then
# the board of the first project. The script writes nothing, not even an error, so neither a
# response nor the access file's content can reach the journal, and it exits 0 only when both
# reads answer 2xx with JSON. A redirect is no answer, and the access file never follows one. It
# ends itself after the readback's ten seconds too: a docker exec killed at the time limit leaves
# its command running in the container.
_WORKSPACE_READS = r"""
const fs = require("fs");
setTimeout(() => process.exit(1), 10000);
(async () => {
  const base = process.env.HERMES_API_BASE_URL.replace(/\/+$/, "");
  const token = fs.readFileSync(process.env.HERMES_API_TOKEN_FILE, "utf8").trim();
  const read = async (path) => {
    const answer = await fetch(base + path, {
      headers: { Authorization: `Bearer ${token}` },
      redirect: "error",
    });
    if (!answer.ok) throw new Error();
    return answer.json();
  };
  const { projects } = await read("/api/plugins/kanban/projects");
  await read(`/api/plugins/kanban/board?board=${encodeURIComponent(projects[0].slug)}`);
})().then(() => process.exit(0), () => process.exit(1));
"""


class ReleaseHostActions:
    """The actions and readbacks of `ReleaseHost` on the live host, from explicit settings (S-E).

    `units` maps the roles gateway, serve and sandbox-tunnel to unit names. A role without a unit
    name is never driven and reads back False, as does an empty address. `venv_dir` is the
    platform environment's link, and `extras` the extras its sync installs.
    """

    def __init__(
        self,
        *,
        checkout: str | Path,
        root_home: str | Path,
        units: Mapping[str, str],
        snapshot_root: str | Path,
        health_url: str,
        workspace_container: str,
        venv_dir: str | Path | None = None,
        extras: Sequence[str] = SYNC_EXTRAS,
    ) -> None:
        self.checkout_path = Path(checkout)  # `checkout` is the move below
        self.root_home = Path(root_home)
        self.units = dict(units)
        self.snapshot_root = Path(snapshot_root)
        self.health_url = health_url
        self.workspace_container = workspace_container
        self.venv_dir = None if venv_dir is None else Path(venv_dir)
        self.extras = tuple(extras)
        # The configuration snapshot this release saved: the one the restore puts back.
        self.config_snapshot = ""
        # The environment snapshot this release took: the one its restore moves back.
        self.environment_snapshot = ""

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

        Without --force no local change is ever dropped: git carries it over or refuses. An
        ignored file that the commit tracks makes git refuse too (--no-overwrite-ignore), because
        git status does not show it. The guards found the checkout clean before the stop.
        """
        self._git("checkout", "--detach", "--no-overwrite-ignore", _full_sha(commit))

    def _git(self, *args: str) -> None:
        _run(["git", "-C", str(self.checkout_path), *args])

    # Snapshots (S-C).

    def take_snapshot(self, name: str) -> None:
        """Copy every file of the root home that the full backup holds into state/NAME.

        Those are the files backup.py's own walk (`_iter_backup_files`) yields for the root home,
        whose profile homes lie under it, with state/NAME as its output; its exclusions are the
        only ones, and its whole list is taken before the first copy. A file that starts with
        SQLite's header goes through backup.py's online copy, which reads it through its
        write-ahead log into one standalone image, and every other file is copied as it is. A
        root home that is not a directory, or a listed file that is not copied, fails the snapshot
        before it is published, and only the owner can read the copies, whatever the umask or the
        snapshot root's access.

        The runner takes it with the platform units stopped, so no platform writer is live beside
        the copy. While the user manager answers active for one of them, the snapshot is refused
        before anything is copied.
        """
        for role in PLATFORM_UNITS:
            if self.unit_active(role):
                raise RuntimeError(f"the {role} unit is active: the state snapshot is refused")
        target = self._named(STATE, name)
        if not self.root_home.is_dir():  # the walk would yield nothing: an empty snapshot
            raise NotADirectoryError(f"the root home {self.root_home} is not a directory")

        def fill(staging: Path) -> None:
            missing: list[str] = []
            # Sorted, so a database's image is made before its own -wal is copied beside it:
            # SQLite deletes a -wal it finds beside the empty file it starts the image in.
            for source, rel in sorted(_iter_backup_files(self.root_home, target)):
                copy = staging / rel
                copy.parent.mkdir(parents=True, exist_ok=True)
                database = _starts_with_header(source)
                if database is None or not (
                    _safe_copy_db(source, copy) if database else _copied(source, copy)
                ):
                    missing.append(rel.as_posix())
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

        The whole save is read and checked first. A save without the root home's config.yaml, or
        with a file that cannot be read or that is no configuration file of the root home or of a
        profile home, is refused before any live file changes: putting it back would remove live
        configuration that it does not hold. Each saved file comes back by one atomic replace,
        into a private directory made again when it is gone. A configuration file of a served home
        that the save does not hold is removed. A profile deleted since the save, every other file
        and the databases stay as they are.
        """
        if not self.config_snapshot:
            raise RuntimeError("no configuration snapshot was saved")
        saved = self._named(CONFIG, self.config_snapshot)
        if not saved.is_dir():
            raise FileNotFoundError(f"the configuration snapshot {self.config_snapshot} is gone")
        kept = set()
        for copy in _read_config_save(saved):
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

    # The platform environment (D2, D3).

    def sync_environment(self) -> None:
        """Sync the checkout's lock file into the environment, in place in the real folder the
        link names, through that environment's own interpreter.

        Frozen, so the lock file is used as it is and never resolved again; inexact, so packages
        installed on demand stay; the extras set, and no default groups. The locked dependencies
        only: the project itself is never built or installed again, because the platform's
        setup.py refuses a wheel build, and its installed copy keeps its current install mode,
        never editable. uv is the one in the Hermes home's bin folder, whatever PATH holds;
        without it the sync raises before any call.
        """
        uv, real = self._uv(), self._environment()
        extras = [argument for extra in self.extras for argument in ("--extra", extra)]
        _run(
            [uv, "sync", "--frozen", "--inexact", "--no-default-groups", "--no-editable",
             "--no-install-project", *extras,
             "--project", str(self.checkout_path), "--python", str(real / "bin" / "python")],
            env={**os.environ, "UV_PROJECT_ENVIRONMENT": str(real)},
            timeout=SYNC_TIMEOUT_SECONDS,
        )

    def environment_matches_lock(self) -> bool:
        """Every package the checkout's lock file names for this host, with the extras and no
        default groups, is installed at its locked version. A package outside the lock, such as
        one installed on demand, may stay. A comparison that cannot be read reads False."""
        try:
            return not self.environment_differences()
        except EnvironmentUnreadable:
            return False

    def environment_differences(self) -> tuple[str, ...]:
        """Each package the checkout's lock file names for this host, with the extras and no
        default groups, that is not installed at its locked version, by normalised name and in
        name order: "NAME expected LOCKED, installed VERSION", or "NAME expected LOCKED, missing".
        A package outside the lock, such as one installed on demand, is no difference. Both lists
        are read through uv, so nothing is imported. A comparison that cannot be read raises
        EnvironmentUnreadable with a short reason that names no path, address or setting."""
        try:
            uv = self._uv()
        except FileNotFoundError:
            raise EnvironmentUnreadable("uv is not in the Hermes home's bin folder") from None
        try:
            real = self._environment()
        except (OSError, RuntimeError, ValueError):
            raise EnvironmentUnreadable("the platform environment is not found") from None
        return lock_differences(uv, real, self.checkout_path, self.extras)

    def take_environment_snapshot(self, name: str) -> None:
        """Copy the real folder the environment's link names, all of it, into env/NAME.

        Taken with the platform units stopped, like the state snapshot, and refused before
        anything is copied while the user manager answers active for one of them.
        """
        for role in PLATFORM_UNITS:
            if self.unit_active(role):
                raise RuntimeError(f"the {role} unit is active: the environment snapshot is refused")
        real = self._environment()
        self._write_snapshot(ENVIRONMENT, name, lambda staging: _copy_environment(real, staging), replace=False)
        self.environment_snapshot = name

    def restore_environment(self) -> None:
        """Move the environment snapshot this release took back into the same real folder.

        With no snapshot taken, the release never synced, and nothing changes. The live folder is
        set aside beside itself and removed once the copy is in place; when the move fails, it
        comes back. The link is never touched. The copy is then taken again from the folder, so
        a later move back, as in tier 2, still has its way back.
        """
        if not self.environment_snapshot:
            return
        for role in PLATFORM_UNITS:
            if self.unit_active(role):
                raise RuntimeError(f"the {role} unit is active: the environment restore is refused")
        name = self.environment_snapshot
        saved = self._named(ENVIRONMENT, name)
        if not saved.is_dir():
            raise FileNotFoundError(f"the environment snapshot {name} is gone")
        if self.venv_dir is None:
            raise ValueError("release.venv_dir names no platform environment")
        real = self.venv_dir.resolve()  # the link's own target, even when the folder is gone
        aside = real.with_name(f".{real.name}.release-aside")
        shutil.rmtree(aside, ignore_errors=True)
        if os.path.lexists(real):
            os.replace(real, aside)
        try:
            shutil.move(saved, real)
        except BaseException:
            shutil.rmtree(real, ignore_errors=True)
            if os.path.lexists(aside):
                os.replace(aside, real)
            raise
        shutil.rmtree(aside, ignore_errors=True)
        self._write_snapshot(ENVIRONMENT, name, lambda staging: _copy_environment(real, staging), replace=True)

    def delete_environment_snapshot(self, name: str) -> None:
        with contextlib.suppress(FileNotFoundError):
            shutil.rmtree(self._named(ENVIRONMENT, name))

    def _environment(self) -> Path:
        """The real folder the venv_dir link names. Raises when it holds no environment."""
        if self.venv_dir is None:
            raise ValueError("release.venv_dir names no platform environment")
        real = self.venv_dir.resolve(strict=True)
        if not (real / "pyvenv.cfg").is_file():
            raise FileNotFoundError(f"{real} holds no environment")
        return real

    def _uv(self) -> str:
        uv = resolve_uv()
        if uv is None:
            raise FileNotFoundError("uv is not in the Hermes home's bin folder")
        return uv

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

    # Readbacks.

    def health_ok(self) -> bool:
        return _answers_ok(self.health_url)

    def workspace_reads_ok(self) -> bool:
        """R6: the projects list and one board read both succeed inside the Workspace container,
        within the time limit. What the read writes goes nowhere, so neither a response nor the
        access file can reach the journal. A refused, failed or late read reads False."""
        try:
            done = subprocess.run(
                ["docker", "exec", self.workspace_container, "node", "-e", _WORKSPACE_READS],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=READBACK_TIMEOUT_SECONDS,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            return False  # no docker, or the time limit passed
        return done.returncode == 0

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


def _read(what: str, argv: list[str]) -> str:
    """What one uv read of the environment's comparison prints. A read that fails raises
    EnvironmentUnreadable naming only what was read, as a failed process's error names paths."""
    try:
        return _run(argv, timeout=ENVIRONMENT_READ_TIMEOUT_SECONDS).stdout
    except subprocess.TimeoutExpired:
        raise EnvironmentUnreadable(f"{what} timed out") from None
    except (OSError, RuntimeError, subprocess.SubprocessError):
        raise EnvironmentUnreadable(f"{what} failed") from None


def lock_differences(uv: str, real: Path, checkout: Path, extras: Sequence[str]) -> tuple[str, ...]:
    """The comparison of `environment_differences`, with uv, the environment's real folder, the
    checkout and the extras given: the release guards ask it too, before the first stop (G12)."""
    arguments = [argument for extra in extras for argument in ("--extra", extra)]
    locked = _locked_here(_read(
        "the lock export",
        [uv, "export", "--frozen", "--format", "requirements.txt", "--no-hashes",
         "--no-header", "--no-annotate", "--no-emit-project", "--no-default-groups",
         *arguments, "--project", str(checkout)],
    ))
    listed = _read(
        "the installed list",
        [uv, "pip", "list", "--python", str(real / "bin" / "python"), "--format", "json"],
    )
    try:
        installed = {_package(entry["name"]): entry["version"] for entry in json.loads(listed)}
        versions = {name: Version(installed[name]) for name in locked if name in installed}
    except (ValueError, KeyError, TypeError):
        raise EnvironmentUnreadable("the installed list could not be read") from None
    return tuple(
        f"{name} expected {version}, "
        + (f"installed {versions[name]}" if name in versions else "missing")
        for name, version in sorted(locked.items())
        if versions.get(name) != version
    )


def _full_sha(commit: str) -> str:
    """Only a full commit name moves the host: a short or symbolic one could name another."""
    if not _FULL_SHA.fullmatch(commit):
        raise ValueError(f"not a full commit name: {commit!r}")
    return commit


def _starts_with_header(path: Path) -> bool | None:
    """Whether a listed file starts with SQLite's header, or None when its first bytes cannot be
    read: then it cannot be copied. A file this process holds a SQLite connection to is a database
    and is not opened raw, as closing a raw descriptor would drop the locks SQLite holds on it
    (sqlite_safe_read.py). A file that is not regular, such as a leftover socket or a pipe, is
    never opened."""
    try:
        if not stat.S_ISREG(os.lstat(path).st_mode):
            return None
    except OSError:
        return None
    if has_live_connection(path):
        return True
    head = read_header_bytes_preopen(path, length=len(_SQLITE_HEADER))
    return None if head is None else head == _SQLITE_HEADER


def _copy_environment(real: Path, staging: Path) -> None:
    """Copy the environment's real folder as it is: every file with its mode, every link as a
    link, so its interpreter link still names the same interpreter. Only its __pycache__ folders
    are left out: Python writes bytecode again, and a sweep can delete one while the copy runs."""
    shutil.copytree(
        real,
        staging,
        symlinks=True,
        dirs_exist_ok=True,
        ignore=shutil.ignore_patterns("__pycache__"),
    )


def _package(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _locked_here(exported: str) -> dict[str, Version]:
    """The packages a lock export pins for this host, by normalised name. A package whose marker
    does not hold here, such as one for another platform, is left out. An export that cannot be
    read raises EnvironmentUnreadable, naming at most the package, never the line itself."""
    locked = {}
    for line in exported.splitlines():
        line = line.strip()
        if not line or line.startswith(("#", "-")):
            continue
        try:
            requirement = Requirement(line)
            if requirement.marker is not None and not requirement.marker.evaluate():
                continue
            pins = list(requirement.specifier)  # an export pins each package with one ==
            name = _package(requirement.name)
            if len(pins) != 1 or pins[0].operator != "==":
                raise EnvironmentUnreadable(f"the lock export does not pin {name} with one ==")
            locked[name] = Version(pins[0].version)
        except ValueError:
            raise EnvironmentUnreadable("the lock export could not be read") from None
    return locked


def _copied(source: Path, copy: Path) -> bool:
    """Copy a file that does not start with SQLite's header as it is."""
    try:
        shutil.copy2(source, copy)
    except OSError:
        return False
    return True


def _read_config_save(saved: Path) -> list[Path]:
    """Every file of a configuration save, sorted, once each one was read (S-D).

    Each directory of the save is listed, and a listing that fails raises rather than leave its
    files out. So does a file that is not regular, such as a pipe, which is never opened, or one
    that cannot be read. A file other than CONFIG_FILES of the root home or of a profile home, by
    the names _served_homes takes, and a save without the root home's config.yaml raise too.
    """
    files, directories = [], [saved]
    while directories:
        for path in directories.pop().iterdir():
            if path.is_dir() and not path.is_symlink():
                directories.append(path)
                continue
            rel = path.relative_to(saved)
            home = rel.parent.parts  # () for the root home, ("profiles", NAME) for a profile's
            a_home = not home or (
                len(home) == 2
                and home[0] == "profiles"
                and home[1] != "default"
                and _PROFILE_ID_RE.match(home[1])
            )
            if rel.name not in CONFIG_FILES or not a_home:
                raise ValueError(
                    f"the configuration snapshot {saved.name} holds {rel}, no configuration file"
                )
            if not path.is_file():
                raise OSError(f"the configuration snapshot {saved.name} holds {rel}, not a file")
            path.read_bytes()  # only the read: the restore copies the file with its mode
            files.append(path)
    if saved / "config.yaml" not in files:
        raise FileNotFoundError(f"the configuration snapshot {saved.name} holds no config.yaml")
    return sorted(files)


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
