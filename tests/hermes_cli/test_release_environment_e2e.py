"""The release's environment sync, readback and rollback with real uv, inside temporary folders only.

This is the rehearsal of Build 1's environment step (plan, item 5). A temporary project stands in
for the checkout, and a temporary environment, linked from the checkout as on the release host,
for the platform environment. Its packages are tiny wheels made here and served from a local
folder, so nothing is fetched. The Hermes home is a temporary folder whose bin folder holds a link
to the real uv, and uv is taken off PATH, as for the release service. No unit, the live Hermes
home and its environment are never read or written.

The last test runs the registered ``hermes release run`` against the actual platform project: its
manifest, lock file and build guard, copied from this repository. Its locked packages come offline
from the test's own scratch uv cache, a temporary folder that the operator's scratch sync step,
run by the test in its scratch checkout before anything runs offline, fills from the package index.

Marked integration, so normal runs leave it out. The operator runs this file with exactly this
command, from the repository root, with ``-m integration`` because the addopts in pyproject.toml
deselect integration tests:

    scripts/run_tests.sh tests/hermes_cli/test_release_environment_e2e.py -m integration

It needs uv on PATH when it starts, and the package index reachable for the scratch sync step.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tomllib
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from packaging.requirements import Requirement
from packaging.version import Version

from hermes_cli import build_info, main, release_cmd, release_host_actions, release_runner
from hermes_cli.release_host_actions import ReleaseHostActions
from tests.hermes_cli.test_release_run import (
    HELD, NEW, PREV, SETTINGS, Boundary, _accepted, _batches, _readbacks,
    sent,  # noqa: F401  (autouse: the owner's send is replaced by a record of its lines)
)

pytestmark = pytest.mark.integration

PROJECT = "platform-e2e"
# The checkout's version A and version B. B raises the locked library; both lock the same extra.
LOCK_A = {"e2e-lib": "1.0", "e2e-extra": "1.0"}
LOCK_B = {"e2e-lib": "2.0", "e2e-extra": "1.0"}
WHEELS = (
    ("e2e-lib", "1.0"), ("e2e-lib", "2.0"), ("e2e-extra", "1.0"), ("e2e-dev", "1.0"),
    ("e2e-on-demand", "1.0"),
)

# The repository under test, whose platform project definition the last test copies.
REPOSITORY = Path(__file__).resolve().parents[2]
# The manifest, the lock file, the build guard and the files the manifest names.
PROJECT_FILES = ("pyproject.toml", "uv.lock", "setup.py", "README.md", "LICENSE")
# A small, explicitly labelled installed-distribution fixture made by _wheel: it stands for the
# platform package as the live host has it installed, at its version and not editable. It is no
# build of Hermes, and holds no platform code.
PLATFORM_FIXTURE = ("hermes-agent", "0.21.2")
# Stand-ins, made by _wheel, for the packages the live host installed on demand (D3).
ON_DEMAND = {
    "discord.py": "2.7.1", "python-telegram-bot": "22.8", "slack-sdk": "3.43.0",
    "brotlicffi": "1.2.0.2",
}
# The operator's scratch sync step, which the last test runs in its scratch checkout before its
# offline part: it fills the test's scratch uv cache with every locked package, and the test then
# installs them from it offline.
SCRATCH_SYNC = ("sync", "--frozen", "--no-install-project", "--extra", "all", "--extra", "dev")


def _wheel(index: Path, name: str, version: str) -> None:
    """A pure-Python wheel with one module that names its version."""
    module = name.replace("-", "_")
    dist = f"{module}-{version}.dist-info"
    files = {
        f"{module}/__init__.py": f"__version__ = {version!r}\n",
        f"{dist}/METADATA": f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n",
        f"{dist}/WHEEL": (
            "Wheel-Version: 1.0\nGenerator: e2e\nRoot-Is-Purelib: true\nTag: py3-none-any\n"
        ),
    }
    with zipfile.ZipFile(index / f"{module}-{version}-py3-none-any.whl", "w") as wheel:
        for path, text in files.items():
            wheel.writestr(path, text)
        wheel.writestr(f"{dist}/RECORD", "".join(f"{path},,\n" for path in [*files, f"{dist}/RECORD"]))


def _write_version(checkout: Path, index: Path, lock: dict[str, str], uv_version: str) -> None:
    """The project manifest of one version: the library in its dependencies, the extra in the
    all extra and a dev package in the default dev group. Its build backend is pinned to the
    running uv's own version, so uv builds it itself, offline, whichever uv runs the rehearsal."""
    (checkout / "pyproject.toml").write_text(
        f'[project]\nname = "{PROJECT}"\nversion = "0.1.0"\nrequires-python = ">=3.8"\n'
        f'dependencies = ["e2e-lib=={lock["e2e-lib"]}"]\n\n'
        f'[project.optional-dependencies]\nall = ["e2e-extra=={lock["e2e-extra"]}"]\n\n'
        '[dependency-groups]\ndev = ["e2e-dev==1.0"]\n\n'
        f'[build-system]\nrequires = ["uv_build=={uv_version}"]\nbuild-backend = "uv_build"\n\n'
        f'[tool.uv]\nno-index = true\nfind-links = ["{index}"]\n'
    )


def _uv(uv: str, *args: str) -> str:
    done = subprocess.run([uv, *args], capture_output=True, text=True, check=False)
    assert done.returncode == 0, done.stderr
    return done.stdout


def _installed(uv: str, real: Path) -> dict[str, str]:
    listed = json.loads(_uv(uv, "pip", "list", "--python", str(real / "bin" / "python"), "--format", "json"))
    return {entry["name"]: entry["version"] for entry in listed}


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _install_record(uv: str, real: Path, name: str) -> tuple[str, bool, dict[str, str]]:
    """How NAME is installed in the environment: its version, whether it is editable, and every
    file its install record lists, with the digest of its bytes."""
    listed = json.loads(_uv(uv, "pip", "list", "--python", str(real / "bin" / "python"), "--format", "json"))
    entry = next(entry for entry in listed if entry["name"] == name)
    site = next((real / "lib").glob("python3*/site-packages"))
    record = site / f"{name.replace('-', '_')}-{entry['version']}.dist-info" / "RECORD"
    files = [line.split(",")[0] for line in record.read_text().splitlines() if line]
    return entry["version"], "editable_project_location" in entry, {path: _digest(site / path) for path in files}


def _normal(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _exported(exported: str) -> tuple[set[str], dict[str, Version]]:
    """Every package a frozen export names, and the version it pins for this interpreter."""
    names, locked = set(), {}
    for line in filter(str.strip, exported.splitlines()):
        requirement = Requirement(line)
        names.add(_normal(requirement.name))
        if requirement.marker is None or requirement.marker.evaluate():
            (pin,) = requirement.specifier
            assert pin.operator == "==", line
            locked[_normal(requirement.name)] = Version(pin.version)
    return names, locked


class _Boundary(Boundary):
    """test_release_run's boundary, whose git checks out any commit: a tier-2 release's back move
    checks out PREV."""

    def run(self, argv, **options):
        if argv[0] == "git" and argv[3] == "checkout":
            self.head = argv[-1]
            self.moves.append(f"checkout {self.head}")
            return self._done("")
        return super().run(argv, **options)


@pytest.fixture
def rehearsal(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    uv = shutil.which("uv")
    assert uv, "the rehearsal needs uv on PATH when it starts"
    uv = os.path.realpath(uv)
    cache = tmp_path / "uv-cache"
    monkeypatch.setenv("UV_CACHE_DIR", str(cache))  # uv's cache stays in the temporary folder
    monkeypatch.setenv("UV_OFFLINE", "1")
    for variable in ("VIRTUAL_ENV", "UV_PROJECT_ENVIRONMENT", "UV_PYTHON"):
        monkeypatch.delenv(variable, raising=False)

    index = tmp_path / "wheels"
    index.mkdir()
    for name, version in WHEELS:
        _wheel(index, name, version)

    # The checkout: the project at version A and B, each with the lock file uv makes for it.
    checkout = tmp_path / "checkout"
    (checkout / "src" / "platform_e2e").mkdir(parents=True)
    (checkout / "src" / "platform_e2e" / "__init__.py").write_text('__version__ = "0.1.0"\n')
    uv_version = _uv(uv, "--version").split()[1]  # "uv 0.9.28 (...)"
    locks = {}
    for version, lock in (("B", LOCK_B), ("A", LOCK_A)):
        _write_version(checkout, index, lock, uv_version)
        _uv(uv, "lock", "--project", str(checkout))
        locks[version] = (checkout / "pyproject.toml").read_text(), (checkout / "uv.lock").read_text()

    # The platform environment: one real folder, linked from the checkout, with a package
    # installed on demand, outside the lock, and the project itself installed as on the live
    # host before any release: as a package from its own wheel, not editable.
    real = tmp_path / "environments" / "hermes-env"
    real.parent.mkdir()
    _uv(uv, "venv", "--quiet", "--python", sys.executable, str(real))
    _uv(uv, "pip", "install", "--python", str(real / "bin" / "python"), "--find-links", str(index),
        "--no-index", "e2e-on-demand==1.0")
    installed = tmp_path / "installed"
    installed.mkdir()
    _wheel(installed, PROJECT, "0.1.0")
    _uv(uv, "pip", "install", "--python", str(real / "bin" / "python"), "--find-links",
        str(installed), "--no-index", f"{PROJECT}==0.1.0")
    link = checkout / "venv"
    os.symlink(real, link)

    # The Hermes home holds uv in its bin folder; PATH no longer leads to any uv.
    home = tmp_path / "home"
    (home / "bin").mkdir(parents=True)
    os.symlink(uv, home / "bin" / "uv")
    monkeypatch.setenv("HERMES_HOME", str(home))
    path = [entry for entry in os.environ["PATH"].split(os.pathsep) if entry and not (Path(entry) / "uv").exists()]
    monkeypatch.setenv("PATH", os.pathsep.join(path))
    assert shutil.which("uv") is None

    def check_out(version: str) -> None:
        manifest, lock = locks[version]
        (checkout / "pyproject.toml").write_text(manifest)
        (checkout / "uv.lock").write_text(lock)

    actions = ReleaseHostActions(
        checkout=checkout,
        root_home=tmp_path / "root",
        units={},  # no unit name: no unit is ever asked about or driven
        snapshot_root=tmp_path / "snapshots",
        health_url="",
        workspace_container="",
        venv_dir=link,
    )
    return actions, check_out, uv, real, link, checkout


def test_sync_readback_and_rollback_with_real_uv(rehearsal):
    actions, check_out, uv, real, link, checkout = rehearsal
    on_demand = {"e2e-on-demand": "1.0"}
    # The project's install record before any sync: a package at its version, not editable.
    project = _install_record(uv, real, PROJECT)
    assert project[:2] == ("0.1.0", False)

    # 1. Version A synced, then the environment snapshot; then version B synced.
    check_out("A")
    lock_a = _digest(checkout / "uv.lock")
    actions.sync_environment()
    assert _digest(checkout / "uv.lock") == lock_a  # frozen: nothing was locked again
    assert actions.environment_matches_lock() is True
    installed_a = _installed(uv, real)
    assert installed_a.items() >= {**LOCK_A, **on_demand}.items()
    assert _install_record(uv, real, PROJECT) == project  # never built or installed again

    actions.take_environment_snapshot("new-version")

    check_out("B")
    lock_b = _digest(checkout / "uv.lock")
    actions.sync_environment()

    # 2. The readback shows B, and everything else stayed: the package installed on demand, the
    # link to the same real folder, the lock file, and the project's own install, unchanged.
    assert actions.environment_matches_lock() is True
    assert actions.environment_differences() == ()
    installed_b = _installed(uv, real)
    assert installed_b.items() >= {**LOCK_B, **on_demand}.items()
    assert "e2e-dev" not in installed_b  # no default groups
    assert _digest(checkout / "uv.lock") == lock_b
    assert link.is_symlink() and os.readlink(link) == str(real)
    assert _install_record(uv, real, PROJECT) == project

    # 3. A mismatch: one locked package off its version. The readback returns False, and the
    # difference names the package with the version B's lock expects and the one installed.
    _uv(uv, "pip", "install", "--python", str(real / "bin" / "python"), "--find-links",
        str(real.parents[1] / "wheels"), "--no-index", "e2e-lib==1.0")
    assert actions.environment_matches_lock() is False
    assert actions.environment_differences() == ("e2e-lib expected 2.0, installed 1.0",)

    # 4. The restore moves the copy back into the same real folder; with the checkout at A
    # again, the readback shows A, exactly as installed before.
    actions.restore_environment()
    check_out("A")
    assert actions.environment_matches_lock() is True
    assert actions.environment_differences() == ()
    assert _installed(uv, real) == installed_a
    assert link.is_symlink() and os.readlink(link) == str(real)
    assert _digest(checkout / "uv.lock") == lock_a
    assert _install_record(uv, real, PROJECT) == project


def test_a_registered_release_syncs_the_actual_platform_project_without_building_it(
    tmp_path, monkeypatch, capsys
):
    """``hermes release run``, parsed by the registered parser, releases a tier-2 batch onto the
    actual platform project: the release syncs its frozen lock with real uv into a real temporary
    environment and never builds the project, whose setup.py refuses a wheel build. Git, the user
    manager, docker, health, the fleet and the owner's notifications are the test boundary of
    test_release_run; only the managed uv runs for real."""
    uv = shutil.which("uv")
    assert uv, "this test needs uv on PATH when it starts"
    uv = os.path.realpath(uv)
    # The test's own scratch uv cache: every uv started here, the release's included, uses it.
    monkeypatch.setenv("UV_CACHE_DIR", str(tmp_path / "uv-cache"))
    (tmp_path / "tmp").mkdir()
    monkeypatch.setenv("TMPDIR", str(tmp_path / "tmp"))  # where uv's sync leaves its lock file
    for variable in ("VIRTUAL_ENV", "UV_PROJECT_ENVIRONMENT", "UV_PYTHON", "HERMES_NIX_BUILD"):
        monkeypatch.delenv(variable, raising=False)

    # The checkout: the platform project's own definition files, byte for byte.
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    for name in PROJECT_FILES:
        shutil.copyfile(REPOSITORY / name, checkout / name)
        assert (checkout / name).read_bytes() == (REPOSITORY / name).read_bytes()
    metadata = tomllib.loads((checkout / "pyproject.toml").read_text())["project"]
    assert {metadata["readme"], *metadata["license-files"]} <= set(PROJECT_FILES)
    definition = {name: _digest(checkout / name) for name in ("uv.lock", "pyproject.toml")}

    # The operator's scratch sync step, in the checkout, fills the scratch cache with every locked
    # package. From here on, everything runs offline.
    _uv(uv, *SCRATCH_SYNC, "--project", str(checkout))
    monkeypatch.setenv("UV_OFFLINE", "1")  # nothing is downloaded: every package is in the cache

    # The lock's frozen export for the all extra with no default groups: the on-demand packages
    # are none of its packages, or the inexact sync would no longer leave them as they are.
    names, locked = _exported(_uv(
        uv, "export", "--frozen", "--format", "requirements.txt", "--no-hashes", "--no-header",
        "--no-annotate", "--no-emit-project", "--no-default-groups", "--extra", "all",
        "--project", str(checkout),
    ))
    on_demand = {_normal(name): version for name, version in ON_DEMAND.items()}
    assert not names & set(on_demand), (
        f"the lock's frozen all export now holds {sorted(names & set(on_demand))}: stop and report it"
    )

    # The platform environment: one real folder made by uv with this Python, behind the link. It
    # holds the locked packages, from the cache only, the labelled hermes-agent fixture installed
    # as a package, and the on-demand stand-ins.
    real = tmp_path / "environments" / "hermes-env"
    real.parent.mkdir()
    _uv(uv, "venv", "--quiet", "--python", sys.executable, str(real))
    seeded = subprocess.run(
        [uv, "sync", "--frozen", "--inexact", "--no-default-groups", "--no-install-project",
         "--extra", "all", "--project", str(checkout), "--python", str(real / "bin" / "python")],
        env={**os.environ, "UV_PROJECT_ENVIRONMENT": str(real)},
        capture_output=True, text=True, check=False,
    )
    if seeded.returncode:
        pytest.fail(
            "The scratch uv cache that the scratch sync step filled cannot serve the locked"
            f" packages offline.\n{seeded.stderr}"
        )
    fixtures = tmp_path / "fixture-wheels"
    fixtures.mkdir()
    for name, version in (PLATFORM_FIXTURE, *ON_DEMAND.items()):
        _wheel(fixtures, name, version)
    _uv(uv, "pip", "install", "--python", str(real / "bin" / "python"), "--find-links",
        str(fixtures), "--no-index",
        *(f"{name}=={version}" for name, version in (PLATFORM_FIXTURE, *ON_DEMAND.items())))
    link = tmp_path / "venv"
    os.symlink(real, link)
    platform = _install_record(uv, real, PLATFORM_FIXTURE[0])
    assert platform[:2] == (PLATFORM_FIXTURE[1], False)
    installed = {_normal(name): version for name, version in _installed(uv, real).items()}
    assert {name: installed.get(name) for name in on_demand} == on_demand

    # The root Hermes home: the release settings, with the environment's link, and the real uv
    # in its bin folder; PATH no longer leads to any uv.
    home = tmp_path / ".hermes"
    (home / "bin").mkdir(parents=True)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    (home / "config.yaml").write_text(yaml.safe_dump({"release": {**SETTINGS, "venv_dir": str(link)}}))
    os.symlink(uv, home / "bin" / "uv")
    path = [entry for entry in os.environ["PATH"].split(os.pathsep) if entry and not (Path(entry) / "uv").exists()]
    monkeypatch.setenv("PATH", os.pathsep.join(path))
    assert shutil.which("uv") is None

    # Every process but the managed uv is the boundary's; the managed uv runs for real.
    boundary = _Boundary(home)
    boundary.stamp()  # the live gateway runs PREV
    managed, real_run = str(home / "bin" / "uv"), subprocess.run

    def run(argv, **options):
        if argv[0] == managed:
            boundary.moves.append(f"uv {argv[1]}")
            return real_run(argv, **options)
        return boundary.run(argv, **options)

    answer = SimpleNamespace(status=200)  # the health address answers
    monkeypatch.setattr(subprocess, "run", run)
    monkeypatch.setattr(release_cmd, "get_project_root", lambda: checkout)
    monkeypatch.setattr(build_info, "get_code_identity", lambda **_: {"sha": boundary.head})
    monkeypatch.setattr(
        release_host_actions, "_DIRECT",
        SimpleNamespace(open=lambda url, timeout: contextlib.nullcontext(answer)),
    )
    monkeypatch.setattr(shutil, "disk_usage", lambda path: SimpleNamespace(free=8 * 1024**3))
    monkeypatch.setattr(release_runner, "_wait", lambda seconds: None)
    monkeypatch.setattr(main, "_plugin_cli_discovery_needed", lambda: False)
    _accepted(2)
    parser, _subparsers = main._build_cli_parser()
    args = parser.parse_args(["release", "run"])

    code = args.func(args)

    out = capsys.readouterr().out
    assert code == 0, f"{out}\nOS and uv actions: {boundary.moves}"
    assert [batch["outcome"] for batch in _batches()] == ["released"]
    assert boundary.moves.count("uv sync") == 2  # each forward move synced NEW's lock
    # Forward, back and forward again, then the last start: every readback held, the
    # environment among its checks.
    assert _readbacks(out) == [
        f"Readback {number} of 4, expecting {commit}: held ({HELD})."
        for number, commit in enumerate((NEW, PREV, NEW, NEW), start=1)
    ]
    assert link.is_symlink() and os.readlink(link) == str(real)
    monkeypatch.setattr(subprocess, "run", real_run)  # the checks below read with the real uv
    # The platform package's install is as it was: its version, not editable, the same files.
    assert _install_record(uv, real, PLATFORM_FIXTURE[0]) == platform
    installed = {_normal(name): version for name, version in _installed(uv, real).items()}
    assert {name: installed.get(name) for name in on_demand} == on_demand
    assert {name: installed.get(name) and Version(installed[name]) for name in locked} == locked
    assert {name: _digest(checkout / name) for name in definition} == definition
