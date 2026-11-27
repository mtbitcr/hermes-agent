"""Each role skill reaches its own role at a release, and an exact copy is adopted.

Each test runs the real sync against a temporary release repository (``skills/``, set through
``HERMES_BUNDLED_SKILLS``, beside ``role-skills/``, set by replacing the one function that gives the
checkout's ``role-skills/``) and temporary homes, set through ``HERMES_HOME``; a role is a home under a
folder named ``profiles``. The release's own call, ``seed_profile_skills``, runs the sync in a child
from the checkout, which no replaced function reaches: its tests use this repository's
``role-skills/``, or a scratch checkout that links this repository's entries beside a fixture
``role-skills/``. A "release" is a change to the repository followed by the sync. The test names
carry the point of the card's facts they hold ("addition": the owner's addition after review round 1).
"""

import hashlib
import json
import os
import shutil
import sys
import threading
from contextlib import suppress
from pathlib import Path
from types import SimpleNamespace

import pytest

import hermes_cli.profiles as profiles
import tools.skills_sync as skills_sync
import tools.skills_sync_roles as skills_sync_roles
from hermes_cli.profiles import seed_profile_skills
from tools.skill_usage import _read_skill_name
from tools.skills_sync import NO_BUNDLED_SKILLS_MARKER, _dir_hash, _discover_bundled_skills, sync_skills

REPO = Path(__file__).resolve().parents[2]
PLANNER, BUILDER = "raphael-planner", "raphael-claude-worker"
OTHER = "raphael-verifier"  # a role with no folder under role-skills/: its run has no role pass
PLAN, BUILD = "team-operations/handing-build-work", "team-operations/running-build-work"
NAME = "handing-build-work"
ROLE_RECORD = ".role_skills_manifest"  # the role pass's own record, beside .bundled_manifest (own-record point 6)
LISTS = ("copied", "updated", "kept", "refused", "suppressed", "shadowed_by_external")
NO_LISTS = {key: [] for key in LISTS}
# What ``sync_skills`` returns today when the general catalogue is not there.
NO_CATALOGUE = {"copied": [], "updated": [], "skipped": 0, "user_modified": [], "cleaned": [], "suppressed": [],
                "total_bundled": 0, "not_seeded_by_marker": 0, "optional_provenance_backfilled": [],
                "kept": [], "refused": []}
OWNED = {"version": 1, "installed": {NAME: {  # a hub lock that owns the planner skill's place
    "source": "official", "identifier": f"official/{PLAN}", "install_path": PLAN}}}

needs_symlinks = pytest.mark.skipif(sys.platform == "win32", reason="symlinks need extra rights on Windows")
needs_fifo = pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="named pipes are POSIX")
# Root reads and writes whatever the mode, so a chmod-based denial only shows for other users.
needs_real_permissions = pytest.mark.skipif(
    sys.platform == "win32" or os.geteuid() == 0, reason="chmod-based access denial needs a non-root user")


@pytest.fixture
def release(tmp_path, monkeypatch):
    """A release repository with ``skills/`` and ``role-skills/``, and a root whose roles are under ``profiles/``."""
    repo = tmp_path / "repo"
    for path in (repo / "skills", repo / "role-skills"):
        path.mkdir(parents=True)
    monkeypatch.setenv("HERMES_BUNDLED_SKILLS", str(repo / "skills"))
    _use_role_skills(monkeypatch, repo / "role-skills")
    root = tmp_path / "root"
    return SimpleNamespace(skills=repo / "skills", roles=repo / "role-skills", root=root,
                           planner=root / "profiles" / PLANNER, builder=root / "profiles" / BUILDER)


@pytest.fixture
def checkout(tmp_path, monkeypatch):
    """A scratch released checkout for the release's own call: a link to each entry of this repository
    beside a fixture ``role-skills/``, which it returns. The child runs from the folder above
    ``hermes_cli/``."""
    root = tmp_path / "checkout"
    (root / "role-skills").mkdir(parents=True)
    for entry in REPO.iterdir():
        if entry.name not in ("role-skills", "skills"):
            (root / entry.name).symlink_to(entry, target_is_directory=entry.is_dir())
    monkeypatch.setattr(profiles, "__file__", str(root / "hermes_cli" / "profiles.py"))
    return root / "role-skills"


def _use_role_skills(monkeypatch, folder: Path) -> None:
    """The checkout's ``role-skills/``, replaced by ``folder`` in this process (addition point 5)."""
    monkeypatch.setattr(skills_sync_roles, "role_skills_dir", lambda: folder)


def _sync(monkeypatch, home: Path) -> dict:
    """One sync of ``home``, as the release runs it for each role."""
    monkeypatch.setenv("HERMES_HOME", str(home))
    return sync_skills(quiet=True)


def _seed(home: Path) -> dict:
    """One seeding of ``home`` through the release's own call: a child from the checkout, whole result."""
    result = seed_profile_skills(home, quiet=True, full_result=True)
    assert result is not None
    return result


def _write_skill(root: Path, rel: str, body: str, name: str = "") -> Path:
    folder = root / rel
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "SKILL.md").write_text(
        f"---\nname: {name or folder.name}\ndescription: test skill\n---\n{body}\n", encoding="utf-8")
    return folder


def _home(home: Path, *, marked: bool = True) -> Path:
    """A role home as the host holds one (facts 4 and 5): its ``skills/`` with a record file, and the marker."""
    (home / "skills").mkdir(parents=True)
    (home / "skills" / ".bundled_manifest").write_text("", encoding="utf-8")
    if marked:
        _mark(home)
    return home


def _edit(folder: Path) -> None:
    md = folder / "SKILL.md"
    md.write_text(md.read_text(encoding="utf-8") + "owner change\n", encoding="utf-8")


def _mark(home: Path) -> None:
    home.mkdir(parents=True, exist_ok=True)
    (home / NO_BUNDLED_SKILLS_MARKER).write_text("opted out\n", encoding="utf-8")


def _files(folder: Path) -> dict:
    """Every file under ``folder``: relative path -> bytes (byte-for-byte comparisons)."""
    return {p.relative_to(folder).as_posix(): p.read_bytes() for p in sorted(folder.rglob("*")) if p.is_file()}


def _manifest(home: Path) -> dict:
    path = home / "skills" / ".bundled_manifest"
    lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
    return dict(line.split(":", 1) for line in lines if line)


def _role_record(home: Path) -> dict:
    """The role pass's own record file, read as ``_manifest`` reads ``.bundled_manifest``."""
    path = home / "skills" / ROLE_RECORD
    lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
    return dict(line.split(":", 1) for line in lines if line)


def _names(result: dict) -> dict:
    """The skill names in each list of the result (``kept`` and ``refused`` hold ``{"name", "reason"}``)."""
    return {key: [item["name"] if isinstance(item, dict) else item for item in result[key]] for key in LISTS}


def _stat(path: Path) -> tuple:
    info = path.stat()
    return info.st_mtime_ns, info.st_ctime_ns, info.st_ino


def _tree(folder: Path) -> dict:
    """Every entry under ``folder``, no link followed: relative path -> its bytes, ("link", target) or "dir"."""
    tree = {}
    for top, dirs, files in os.walk(folder):
        for name in dirs + files:
            path = Path(top) / name
            tree[path.relative_to(folder).as_posix()] = (
                ("link", os.readlink(path)) if path.is_symlink() else "dir" if path.is_dir()
                else "fifo" if path.is_fifo() else path.read_bytes())
    return tree


def _without_record(home: Path) -> dict:
    """``_tree(home)`` but the role record, the one difference from a twin (``.bundled_manifest`` is equal)."""
    tree = _tree(home)
    tree.pop(f"skills/{ROLE_RECORD}", None)
    return tree


def _in_time(call, fifo: Path, seconds: float = 20):
    """``call()``, failed and not blocked when it opens ``fifo`` to read: it runs in a thread, and a reader
    still blocked after ``seconds`` is let go by a writer that opens and closes the FIFO."""
    out = {}
    thread = threading.Thread(target=lambda: out.update(result=call()), daemon=True)
    thread.start()
    thread.join(seconds)
    blocked = thread.is_alive()
    if blocked:
        with suppress(OSError):
            os.close(os.open(fifo, os.O_WRONLY | os.O_NONBLOCK))
        thread.join(seconds)
    assert not blocked, f"the call opened {fifo} to read"
    return out["result"]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _hub_lock(skills: Path, form) -> Path:
    """The hub lock of a role's ``skills/``: ``form`` is its JSON value, or its bytes as they are."""
    path = skills / ".hub" / "lock.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(form if isinstance(form, bytes) else json.dumps(form).encode("utf-8"))
    return path


def _twin(home: Path, twin_root: Path, role: str = "") -> Path:
    """A copy of ``home``'s tree, made before any sync, for the same run with no role pass (addition
    point 13): a role with no folder under ``role-skills/`` (``role``), or no role skills at all."""
    twin = twin_root / "profiles" / (role or home.name)
    shutil.copytree(home, twin, symlinks=True)
    return twin


def _general(monkeypatch, folder: Path) -> Path:
    """The general catalogue of the run, through ``HERMES_BUNDLED_SKILLS`` (the child gets it too)."""
    folder.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("HERMES_BUNDLED_SKILLS", str(folder))
    return folder


def _lists(result: dict) -> dict:
    return {key: result[key] for key in LISTS}


def _outside(home: Path, places) -> dict:
    """``_tree`` of ``skills/`` but ``.role_skills_manifest`` and each role skill's place (relative to
    ``skills/``): the place, each entry inside it and each folder on the way to it (own-record point 21)."""
    places = [Path(place) for place in places]
    return {rel: entry for rel, entry in _tree(home / "skills").items()
            if rel != ROLE_RECORD and not any(Path(rel) == place or place in Path(rel).parents
                                              or Path(rel) in place.parents for place in places)}


def _step(home: Path, control_root: Path, roles: Path, places, role: dict) -> dict:
    """One release step of a point-20 sequence with its control home (own-record point 21): a copy of
    ``home`` under ``control_root/profiles/`` is seeded with the checkout's ``role-skills/`` (``roles``)
    renamed aside, then ``home`` with it there. The home's result is the control's with ``role``, the
    role pass's entries, appended to its lists; ``.bundled_manifest`` and ``skills/`` outside ``places``
    and outside ``.role_skills_manifest`` are equal. Returns the home's result."""
    control = control_root / "profiles" / home.name
    shutil.copytree(home, control, symlinks=True)
    aside = roles.with_name("role-skills-aside")
    roles.rename(aside)
    try:
        expected = _seed(control)
    finally:
        aside.rename(roles)
    result = _seed(home)
    assert result == {key: value + role[key] if key in role else value for key, value in expected.items()}
    assert _manifest(home) == _manifest(control)
    assert _outside(home, places) == _outside(control, places)
    return result


# -- Point 10, the table ---------------------------------------------------------------------------

def test_10a_a_new_role_skill_is_copied_with_its_record_line(release, monkeypatch):
    src = _write_skill(release.roles / PLANNER, PLAN, "planner v1")

    result = _sync(monkeypatch, release.planner)

    assert result["copied"] == [NAME]
    assert _files(release.planner / "skills" / PLAN) == _files(src)
    assert _role_record(release.planner) == {NAME: _dir_hash(src)}
    assert _manifest(release.planner) == {}


def test_10b_a_deleted_role_copy_stays_deleted(release, monkeypatch):
    _write_skill(release.roles / PLANNER, PLAN, "planner v1")
    _sync(monkeypatch, release.planner)
    shutil.rmtree(release.planner / "skills" / PLAN)
    before = _files(release.planner / "skills")
    _write_skill(release.roles / PLANNER, PLAN, "planner v2")  # a later release changes it

    result = _sync(monkeypatch, release.planner)

    assert not os.path.lexists(release.planner / "skills" / PLAN)
    assert _files(release.planner / "skills") == before  # its line stays as it was
    assert not any(_names(result).values())


def test_10b_a_line_with_no_copy_at_the_place_writes_nothing_when_the_release_moves_the_skill(release, monkeypatch):
    """No folder at the place and a line: nothing is written, and no copy elsewhere is moved."""
    _write_skill(release.roles / PLANNER, PLAN, "planner v1")
    assert _sync(monkeypatch, release.planner)["copied"] == [NAME]
    shutil.move(str(release.roles / PLANNER / "team-operations"), str(release.roles / PLANNER / "build-loop"))
    before = _files(release.planner / "skills")

    result = _sync(monkeypatch, release.planner)

    assert _files(release.planner / "skills") == before
    assert not os.path.lexists(release.planner / "skills" / "build-loop")
    assert not any(_names(result).values())


@pytest.mark.parametrize("marked", [True, False], ids=["marker", "no-marker"])
def test_10c_an_exact_role_copy_is_adopted_and_no_byte_of_it_is_written(release, monkeypatch, marked):
    """As on the production host: the role holds an exact copy and no line for it."""
    src = _write_skill(release.roles / PLANNER, PLAN, "planner v1")
    if marked:
        _mark(release.planner)
    copy = release.planner / "skills" / PLAN
    shutil.copytree(src, copy)
    md = copy / "SKILL.md"
    os.utime(md, ns=(1_600_000_000_000_000_000, 1_600_000_000_000_000_000))  # a rewrite would show
    before = _files(copy), _stat(md), _stat(copy)

    result = _sync(monkeypatch, release.planner)

    assert (_files(copy), _stat(md), _stat(copy)) == before  # bytes, change times and inode
    assert _role_record(release.planner) == {NAME: _dir_hash(src)}
    assert _manifest(release.planner) == {}
    assert not any(_names(result).values())  # an adopted copy is in none of the lists


def test_10c_an_exact_copy_at_a_hub_install_path_is_kept_and_not_adopted(release, monkeypatch):
    src = _write_skill(release.roles / PLANNER, PLAN, "planner v1")
    copy = release.planner / "skills" / PLAN
    shutil.copytree(src, copy)
    (release.planner / "skills" / ".hub").mkdir()
    (release.planner / "skills" / ".hub" / "lock.json").write_text(json.dumps({"version": 1, "installed": {
        NAME: {"source": "official", "identifier": f"official/{PLAN}", "install_path": PLAN}}}), encoding="utf-8")
    before = _files(copy)

    result = _sync(monkeypatch, release.planner)

    assert result["kept"] == [{"name": NAME, "reason": "changed"}]
    assert _files(copy) == before
    assert NAME not in _manifest(release.planner) and NAME not in _role_record(release.planner)


@pytest.mark.parametrize("kind, reason", [
    ("edited", "changed"),
    pytest.param("linked", "link", marks=needs_symlinks),
])
def test_10d_a_changed_role_copy_is_kept_and_named_with_its_reason(release, monkeypatch, tmp_path, kind, reason):
    src = _write_skill(release.roles / PLANNER, PLAN, "planner v1")
    copy = release.planner / "skills" / PLAN
    if kind == "edited":
        shutil.copytree(src, copy)
        _edit(copy)
    else:  # the release's exact text, reached through a link
        shutil.copytree(src, tmp_path / "elsewhere")
        copy.parent.mkdir(parents=True)
        copy.symlink_to(tmp_path / "elsewhere", target_is_directory=True)
    before = _files(copy)

    result = _sync(monkeypatch, release.planner)

    assert result["kept"] == [{"name": NAME, "reason": reason}]
    assert result["copied"] == result["updated"] == result["refused"] == []
    assert _files(copy) == before
    assert NAME not in _manifest(release.planner) and NAME not in _role_record(release.planner)


@pytest.mark.parametrize("marked", [True, False], ids=["marker", "no-marker"])
def test_10e_an_adopted_copy_is_updated_when_the_released_text_changes(release, monkeypatch, marked):
    src = _write_skill(release.roles / PLANNER, PLAN, "planner v1")
    if marked:
        _mark(release.planner)
    shutil.copytree(src, release.planner / "skills" / PLAN)
    _sync(monkeypatch, release.planner)  # adopted
    _write_skill(release.roles / PLANNER, PLAN, "planner v2")

    result = _sync(monkeypatch, release.planner)

    assert result["updated"] == [NAME]
    assert _files(release.planner / "skills" / PLAN) == _files(src)
    assert _role_record(release.planner) == {NAME: _dir_hash(src)}
    assert _manifest(release.planner) == {}


def test_10e_a_recorded_role_copy_that_a_person_changed_is_kept_with_no_update(release, monkeypatch):
    _write_skill(release.roles / PLANNER, PLAN, "planner v1")
    _sync(monkeypatch, release.planner)
    _edit(release.planner / "skills" / PLAN)
    before, line = _files(release.planner / "skills" / PLAN), _role_record(release.planner)
    _write_skill(release.roles / PLANNER, PLAN, "planner v2")

    result = _sync(monkeypatch, release.planner)

    assert result["kept"] == [{"name": NAME, "reason": "changed"}] and result["updated"] == []
    assert _files(release.planner / "skills" / PLAN) == before
    assert _role_record(release.planner) == line != {}
    assert _manifest(release.planner) == {}


# -- Points 7 to 9, 11 and 12 ----------------------------------------------------------------------

def test_7_8_a_role_skill_reaches_only_its_own_role(release, monkeypatch):
    _write_skill(release.roles / PLANNER, PLAN, "planner")
    _write_skill(release.roles / BUILDER, BUILD, "builder")
    other = release.root / "profiles" / "raphael-verifier"

    assert _sync(monkeypatch, release.planner)["copied"] == [NAME]
    assert _sync(monkeypatch, release.builder)["copied"] == ["running-build-work"]
    assert _sync(monkeypatch, other)["copied"] == []
    assert sorted(_files(release.planner / "skills")) == [".bundled_manifest", ROLE_RECORD, f"{PLAN}/SKILL.md"]
    assert sorted(_files(release.builder / "skills")) == [".bundled_manifest", ROLE_RECORD, f"{BUILD}/SKILL.md"]
    assert sorted(_files(other / "skills")) == [".bundled_manifest"]


def test_7_the_root_home_and_a_home_outside_profiles_get_no_role_skill(release, monkeypatch, tmp_path):
    _write_skill(release.roles / PLANNER, PLAN, "planner")
    (release.root / "profiles").mkdir(parents=True)
    # The root home, the root home reached through profiles/.., and homes named as the role elsewhere.
    for home in (release.root, release.root / "profiles" / "..", tmp_path / "teams" / PLANNER, tmp_path / PLANNER):
        result = _sync(monkeypatch, home)

        assert not any(_names(result).values()), home
        assert not os.path.lexists(home / "skills" / "team-operations"), home
        assert _manifest(home) == {}, home
        assert not os.path.lexists(home / "skills" / ROLE_RECORD), home


def test_9_a_role_with_the_no_bundled_skills_marker_gets_its_role_skill(release, monkeypatch):
    _write_skill(release.skills, "software-development/spike", "general")
    _write_skill(release.roles / PLANNER, PLAN, "planner")
    _mark(release.planner)

    result = _sync(monkeypatch, release.planner)

    assert result["copied"] == [NAME]
    assert (release.planner / "skills" / PLAN / "SKILL.md").is_file()
    assert result["not_seeded_by_marker"] == 1  # the general skill is still stopped
    assert not os.path.lexists(release.planner / "skills" / "software-development")


@pytest.mark.parametrize("marked", [True, False], ids=["marker", "no-marker"])
def test_11_a_role_skill_with_the_name_of_a_general_skill_is_refused_and_nothing_is_written(
        release, monkeypatch, marked):
    general = _write_skill(release.skills, f"software-development/{NAME}", "general")
    src = _write_skill(release.roles / PLANNER, PLAN, "planner")
    if marked:
        _mark(release.planner)
    shutil.copytree(src, release.planner / "skills" / PLAN)  # even an exact copy is not adopted
    before = _files(release.planner / "skills" / PLAN)

    result = _sync(monkeypatch, release.planner)

    assert result["refused"] == [{"name": NAME, "reason": "name"}]
    assert _files(release.planner / "skills" / PLAN) == before
    # Only the general skill, at its own place, is seeded (a marked role stops it).
    assert result["copied"] == ([] if marked else [NAME])
    assert _manifest(release.planner) == ({} if marked else {NAME: _dir_hash(general)})
    assert not os.path.lexists(release.planner / "skills" / ROLE_RECORD)


def test_12_a_role_skill_record_line_survives_the_cleanup_in_a_role_with_no_marker(release, monkeypatch):
    src = _write_skill(release.roles / PLANNER, PLAN, "planner")
    _write_skill(release.skills, "software-development/spike", "general")
    _sync(monkeypatch, release.planner)
    shutil.rmtree(release.skills / "software-development")  # the release removes the general skill

    result = _sync(monkeypatch, release.planner)

    assert result["cleaned"] == ["spike"]  # the cleanup ran
    assert _role_record(release.planner) == {NAME: _dir_hash(src)}
    assert _manifest(release.planner) == {}  # the cleanup reads no role record (own-record point 19)
    assert _sync(monkeypatch, release.planner)["cleaned"] == []


def test_8_the_role_copy_lands_at_the_same_inner_path(release, monkeypatch):
    rel = "team-operations/build-loop/handing-build-work"
    src = _write_skill(release.roles / PLANNER, rel, "planner")
    (src / "references").mkdir()
    (src / "references" / "checklist.md").write_text("one\n", encoding="utf-8")

    _sync(monkeypatch, release.planner)

    assert sorted(_files(release.planner / "skills")) == [
        ".bundled_manifest", ROLE_RECORD, f"{rel}/SKILL.md", f"{rel}/references/checklist.md"]
    assert _files(release.planner / "skills" / rel) == _files(src)


# -- Points 13 to 15 -------------------------------------------------------------------------------

@pytest.mark.parametrize("case", [
    pytest.param("a link on the way", marks=needs_symlinks),
    "a suppressed name",
    "a name an external skills folder provides",
    "a failed copy",
])
def test_13_where_the_table_is_silent_a_role_skill_goes_as_a_general_skill_of_a_role_with_no_marker(
        tmp_path, monkeypatch, case):
    """The same skill, once general in a home with no marker and once a role skill of a marked role."""
    seen, copies = [], []

    def failing_copy(src, dest):
        copies.append(Path(dest).relative_to(Path(dest).parents[1]).as_posix())  # its place under skills/
        raise OSError("disk full")

    for kind in ("general", "role"):
        base = tmp_path / kind
        skills = base / "repo" / "skills"
        skills.mkdir(parents=True)
        monkeypatch.setenv("HERMES_BUNDLED_SKILLS", str(skills))
        _use_role_skills(monkeypatch, base / "repo" / "role-skills")
        if kind == "general":
            _write_skill(skills, PLAN, "v1")
            home = base / "home"
        else:
            _write_skill(base / "repo" / "role-skills" / PLANNER, PLAN, "v1")
            home = base / "root" / "profiles" / PLANNER
            _mark(home)
        (home / "skills").mkdir(parents=True)
        outside = base / "outside"
        outside.mkdir()
        if case == "a link on the way":
            (home / "skills" / "team-operations").symlink_to(outside, target_is_directory=True)
        elif case == "a suppressed name":
            (home / "skills" / ".curator_suppressed").write_text(f"{NAME}\n", encoding="utf-8")
        elif case == "a name an external skills folder provides":
            _write_skill(outside, f"devops/{NAME}", "external")
            (home / "config.yaml").write_text(f"skills:\n  external_dirs:\n    - {outside}\n", encoding="utf-8")
        else:
            monkeypatch.setattr(skills_sync, "_copy_dir", failing_copy)

        result = _sync(monkeypatch, home)

        seen.append(({key: result[key] for key in LISTS}, result["skipped"], _manifest(home), _role_record(home),
                     _files(outside)))
    if case == "a name an external skills folder provides":  # own-record point 19: the reason name
        assert seen[0][0] == {**NO_LISTS, "shadowed_by_external": [NAME]}
        assert seen[1][0] == {**NO_LISTS, "refused": [{"name": NAME, "reason": "name"}]}
        assert seen[1][1:] == (0, *seen[0][2:])
    else:
        assert seen[0] == seen[1]
    assert seen[1][2] == seen[1][3] == {}  # no line in any of these cases: the next sync tries again
    if case == "a failed copy":
        assert copies == [PLAN, PLAN] and not any(seen[1][0].values())
    else:
        assert any(seen[1][0].values())  # the case reached the skill


@pytest.mark.parametrize("marked", [True, False], ids=["marker", "no-marker"])
def test_14_the_result_keeps_its_keys_and_its_counts_leave_out_role_skills(release, monkeypatch, tmp_path, marked):
    _write_skill(release.skills, "software-development/spike", "general")
    _write_skill(release.roles / PLANNER, PLAN, "planner")
    other = tmp_path / "home"  # the same release for a home with no role
    if marked:
        _mark(release.planner)
        _mark(other)

    with_role, without = _sync(monkeypatch, release.planner), _sync(monkeypatch, other)

    assert set(with_role) == set(without)
    assert with_role["total_bundled"] == without["total_bundled"]
    assert with_role["not_seeded_by_marker"] == without["not_seeded_by_marker"]
    assert with_role["copied"] == without["copied"] + [NAME]


def _general_release(repo: Path, home: Path, *, marked: bool, role_skills: bool) -> list:
    """General skills only for this role: a first sync, then a release that adds, updates, keeps a changed
    copy, leaves a deleted copy deleted and removes a skill. Returns the two results."""
    skills = repo / "skills"
    _write_skill(skills, "autonomous-ai-agents/hermes-agent", "essential v1")
    for rel in ("software-development/spike", "media/ascii-art", "note-taking/obsidian", "research/arxiv"):
        _write_skill(skills, rel, "v1")
    (skills / "media" / "DESCRIPTION.md").write_text("Media skills\n", encoding="utf-8")
    if role_skills:  # read by the pass, but nothing for this role: another role's skill and a stray file
        _write_skill(repo / "role-skills" / BUILDER, BUILD, "builder")
        (repo / "role-skills" / PLANNER).mkdir(parents=True)
        (repo / "role-skills" / PLANNER / "README.md").write_text("no skill here\n", encoding="utf-8")
    results = [sync_skills(quiet=True)]
    if marked:
        _mark(home)
    _edit(home / "skills" / "note-taking" / "obsidian")
    shutil.rmtree(home / "skills" / "research" / "arxiv")
    for rel in ("media/ascii-art", "note-taking/obsidian", "media/gif-search"):
        _write_skill(skills, rel, "v2")
    shutil.rmtree(skills / "software-development")
    results.append(sync_skills(quiet=True))
    return results


@pytest.mark.parametrize("marked", [True, False], ids=["marker", "no-marker"])
def test_15_general_skills_give_the_same_result_and_bytes_with_the_role_pass_and_with_no_role_skills_folder(
        tmp_path, monkeypatch, marked):
    runs = []
    for role_skills in (True, False):
        repo = tmp_path / ("with" if role_skills else "without") / "repo"
        home = repo.parent / "root" / "profiles" / PLANNER
        (repo / "skills").mkdir(parents=True)
        monkeypatch.setenv("HERMES_BUNDLED_SKILLS", str(repo / "skills"))
        _use_role_skills(monkeypatch, repo / "role-skills")
        monkeypatch.setenv("HERMES_HOME", str(home))
        results = _general_release(repo, home, marked=marked, role_skills=role_skills)
        runs.append(([{key: sorted(value, key=repr) if isinstance(value, list) else value
                       for key, value in result.items()} for result in results], _files(home)))
    assert os.path.isdir(tmp_path / "with" / "repo" / "role-skills" / PLANNER)
    assert not os.path.lexists(tmp_path / "without" / "repo" / "role-skills")
    assert runs[0] == runs[1]
    second = runs[1][0][1]  # the fixture reaches the general rules
    assert second["updated"] == ["ascii-art"] and [item["name"] for item in second["kept"]] == ["obsidian"]
    assert second["copied"] == ([] if marked else ["gif-search"])
    assert second["cleaned"] == ([] if marked else ["spike"])


# -- Point 22, and the plan's acceptance through the release's own call ------------------------------

@pytest.mark.parametrize("role, rel", [(PLANNER, PLAN), (BUILDER, BUILD)])
def test_22_the_two_build_loop_skills_are_in_the_repository_with_their_names(role, rel):
    folder = REPO / "role-skills" / role / rel
    assert _read_skill_name(folder / "SKILL.md", "") == folder.name
    assert _discover_bundled_skills(REPO / "role-skills" / role) == [(folder.name, folder)]


def test_the_release_seeding_adopts_an_exact_planner_copy_and_gives_the_builder_no_planner_skill(
        release, monkeypatch):
    src = REPO / "role-skills" / PLANNER / PLAN  # the child reads the checkout's role-skills/ (addition 3)
    for home in (release.planner, release.builder):
        _mark(home)
    shutil.copytree(src, release.planner / "skills" / PLAN)
    md = release.planner / "skills" / PLAN / "SKILL.md"
    sha = hashlib.sha256(md.read_bytes()).hexdigest()

    result = seed_profile_skills(release.planner, quiet=True, full_result=True)

    assert result is not None
    assert not any(NAME in names for names in _names(result).values())
    assert _role_record(release.planner) == {NAME: _dir_hash(src)}
    assert (release.planner / "skills" / ".bundled_manifest").read_bytes() == b""  # as the general pass wrote it
    assert hashlib.sha256(md.read_bytes()).hexdigest() == sha
    assert seed_profile_skills(release.builder, quiet=True, full_result=True)["copied"] == ["running-build-work"]
    assert not os.path.lexists(release.builder / "skills" / PLAN)


# -- The addition after review round 1: the source (points 3 to 5) -----------------------------------

def test_addition_3_5_one_function_gives_the_role_skills_folder_beside_tools(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_BUNDLED_SKILLS", str(tmp_path / "skills"))  # it moves the general catalogue only
    folder = skills_sync_roles.role_skills_dir()
    assert folder == Path(skills_sync.__file__).parent.parent / "role-skills"
    assert folder.resolve() == REPO / "role-skills"


@pytest.mark.parametrize("marked", [True, False], ids=["marker", "no-marker"])
def test_addition_3_the_release_seeds_the_checkout_role_skills_whatever_hermes_bundled_skills_names(
        tmp_path, monkeypatch, marked):
    """Review finding 1, through the release's own call: another general catalogue, with role skills
    beside it, moves no role skill."""
    other = _general(monkeypatch, tmp_path / "other-release" / "skills").parent
    spike = _write_skill(other / "skills", "software-development/spike", "v1")
    for role, rel in ((PLANNER, PLAN), (BUILDER, BUILD)):
        for decoy in (rel, "decoy/decoy-skill"):
            _write_skill(other / "role-skills" / role, decoy, "not the release's")
    for role, rel in ((PLANNER, PLAN), (BUILDER, BUILD)):
        home = _home(tmp_path / "root" / "profiles" / role, marked=marked)
        src = REPO / "role-skills" / role / rel

        result = _seed(home)

        assert result["copied"] == ([] if marked else ["spike"]) + [src.name]
        assert result["kept"] == result["refused"] == []
        assert _files(home / "skills" / rel) == _files(src)
        assert _role_record(home) == {src.name: _dir_hash(src)}
        assert _manifest(home) == ({} if marked else {"spike": _dir_hash(spike)})
        assert not os.path.lexists(home / "skills" / "decoy")


@pytest.mark.parametrize("marked", [True, False], ids=["marker", "no-marker"])
def test_addition_4_the_role_pass_runs_when_the_general_catalogue_is_not_there(tmp_path, monkeypatch, marked):
    """Review finding 1, through the release's own call: a general catalogue that is not there stops
    the general pass only. The root home, and a role with no role skills, get today's result."""
    monkeypatch.setenv("HERMES_BUNDLED_SKILLS", str(tmp_path / "not-there"))
    root = _home(tmp_path / "root", marked=marked)
    planner = _home(root / "profiles" / PLANNER, marked=marked)
    for home in (root, _home(root / "profiles" / OTHER, marked=marked)):
        before = _tree(root)
        assert _seed(home) == NO_CATALOGUE
        assert _tree(root) == before
    src = REPO / "role-skills" / PLANNER / PLAN

    assert _seed(planner) == {**NO_CATALOGUE, "copied": [NAME]}
    assert _files(planner / "skills" / PLAN) == _files(src)
    assert _role_record(planner) == {NAME: _dir_hash(src)}
    assert _manifest(planner) == {}


# -- The addition: the plain shape of a role skill (points 6 and 7) ----------------------------------

@needs_symlinks
@pytest.mark.parametrize("case, refused", [pytest.param(case, refused, id=case) for case, refused in (
    ("the role's folder", [{"name": PLANNER, "reason": "link"}]),
    ("role-skills", [{"name": "role-skills", "reason": "link"}]),
    ("a category folder in the role's folder", [{"name": "team-operations", "reason": "link"}]),
    ("the skill folder", [{"name": NAME, "reason": "link"}]),
)])
def test_addition_6a_a_link_on_the_way_from_role_skills_gives_the_role_no_skill(
        checkout, tmp_path, monkeypatch, case, refused):
    """Review finding 2, through the release's own call: the planner's way to a skill passes a link, to
    the builder's folders or out of the checkout. Each link is refused under its own name, and nothing
    is read through it (review 2 finding 4)."""
    if case == "role-skills":
        _write_skill(tmp_path / "elsewhere" / PLANNER, PLAN, "not in the checkout")
        checkout.rmdir()
        checkout.symlink_to(tmp_path / "elsewhere", target_is_directory=True)
    else:
        builder = _write_skill(checkout / BUILDER, BUILD, "builder")
        link, target = {"the role's folder": (checkout / PLANNER, checkout / BUILDER),
                        "a category folder in the role's folder": (checkout / PLANNER / "team-operations",
                                                                   builder.parent),
                        "the skill folder": (checkout / PLANNER / PLAN, builder)}[case]
        link.parent.mkdir(parents=True, exist_ok=True)
        link.symlink_to(target, target_is_directory=True)
    _general(monkeypatch, tmp_path / "general")
    planner = _home(tmp_path / "root" / "profiles" / PLANNER)
    before = _tree(planner)

    result = _seed(planner)

    assert _lists(result) == {**NO_LISTS, "refused": refused}
    assert _tree(planner) == before


def test_addition_6a_a_skill_md_in_the_role_folder_itself_is_refused_with_each_skill_inside(release, monkeypatch):
    _write_skill(release.roles, PLANNER, "the whole role", name="role-root")
    _write_skill(release.roles / PLANNER, PLAN, "planner")
    home = _home(release.planner)
    before = _tree(home)

    result = _sync(monkeypatch, home)

    assert sorted(result["refused"], key=lambda item: item["name"]) == [
        {"name": NAME, "reason": "nested"}, {"name": "role-root", "reason": "link"}]
    assert _lists(result) == {**NO_LISTS, "refused": result["refused"]}
    assert _tree(home) == before


@needs_symlinks
@pytest.mark.parametrize("case, name", [pytest.param(case, name, id=case) for case, name in (
    ("a link to the other role's SKILL.md", NAME),  # its folder's own name: nothing is read through it
    ("a folder", NAME),
)])
def test_addition_6b_a_role_skill_whose_skill_md_is_no_regular_file_is_refused(
        checkout, tmp_path, monkeypatch, case, name):
    """Review finding 2, through the release's own call."""
    builder = _write_skill(checkout / BUILDER, BUILD, "builder")
    md = checkout / PLANNER / PLAN / "SKILL.md"
    md.parent.mkdir(parents=True)
    if case == "a folder":
        md.mkdir()
        (md / "notes.md").write_text("a note\n", encoding="utf-8")
    else:
        md.symlink_to(builder / "SKILL.md")
    _general(monkeypatch, tmp_path / "general")
    planner = _home(tmp_path / "root" / "profiles" / PLANNER)
    before = _tree(planner)

    result = _seed(planner)

    assert _lists(result) == {**NO_LISTS, "refused": [{"name": name, "reason": "link"}]}
    assert _tree(planner) == before


@needs_symlinks
@pytest.mark.parametrize("case", ["a file", "a folder"])
def test_addition_6c_a_role_skill_with_a_link_inside_is_refused(release, monkeypatch, case):
    builder = _write_skill(release.roles / BUILDER, BUILD, "builder")
    src = _write_skill(release.roles / PLANNER, PLAN, "planner")
    if case == "a file":
        (src / "references").mkdir()
        (src / "references" / "checklist.md").symlink_to(builder / "SKILL.md")
    else:
        (src / "references").symlink_to(builder, target_is_directory=True)
    home = _home(release.planner)
    before = _tree(home)

    result = _sync(monkeypatch, home)

    assert _lists(result) == {**NO_LISTS, "refused": [{"name": NAME, "reason": "link"}]}
    assert _tree(home) == before


@needs_symlinks
@pytest.mark.parametrize("marked", [True, False], ids=["marker", "no-marker"])
def test_addition_6d_a_role_skill_folder_that_holds_a_second_skill_is_refused_as_a_whole(
        checkout, tmp_path, monkeypatch, marked):
    """Review finding 4, through the release's own call: the inner skill has a general skill's name.
    The twin home is seeded in the same run by a role with no role skills."""
    general = _general(monkeypatch, tmp_path / "general")
    _write_skill(general, "software-development/shared-name", "general")
    _write_skill(checkout / PLANNER, PLAN, "planner")
    _write_skill(checkout / PLANNER, f"{PLAN}/inner/shared-name", "the role's own")
    planner = _home(tmp_path / "root" / "profiles" / PLANNER, marked=marked)
    twin = _twin(planner, tmp_path / "twin", OTHER)

    with_role, without = _seed(planner), _seed(twin)

    assert sorted(with_role["refused"], key=lambda item: item["name"]) == [
        {"name": NAME, "reason": "nested"}, {"name": "shared-name", "reason": "nested"}]
    assert with_role == {**without, "refused": with_role["refused"]} and without["refused"] == []
    assert without["copied"] == ([] if marked else ["shared-name"])
    assert _tree(planner) == _tree(twin)


# -- The addition: the place in the role (point 8) ---------------------------------------------------

@needs_symlinks
@pytest.mark.parametrize("catalogue", [True, False], ids=["general catalogue", "no general catalogue"])
@pytest.mark.parametrize("link", ["home", "skills"])
def test_addition_8_a_role_whose_home_or_skills_folder_is_a_link_gets_nothing_and_one_refusal(
        tmp_path, monkeypatch, link, catalogue):
    """Review finding 2, through the release's own call: the planner's home, or its skills/, is a link to
    the builder's, seeded first. The planner's seeding changes no byte of either."""
    if catalogue:
        _general(monkeypatch, tmp_path / "general")
    else:
        monkeypatch.setenv("HERMES_BUNDLED_SKILLS", str(tmp_path / "not-there"))
    roles = tmp_path / "root" / "profiles"
    builder = _home(roles / BUILDER)
    assert _seed(builder)["copied"] == ["running-build-work"]
    planner = roles / PLANNER
    if link == "home":
        planner.symlink_to(builder, target_is_directory=True)
    else:
        _mark(planner)
        (planner / "skills").symlink_to(builder / "skills", target_is_directory=True)
    before = _tree(roles)

    result = _seed(planner)

    refused = [{"name": PLANNER, "reason": "link"}]
    if catalogue:
        assert _lists(result) == {**NO_LISTS, "refused": refused}
    else:
        assert result == {**NO_CATALOGUE, "refused": refused}
    assert _tree(roles) == before


# -- The addition: the hub lock (points 10 and 11) ---------------------------------------------------

@pytest.mark.parametrize("case", ["no hub folder", "no lock file", "a lock of other paths",
                                  "a lock that owns its place"])
def test_addition_10_an_exact_copy_is_adopted_only_when_the_hub_lock_is_not_there_or_owns_other_paths(
        tmp_path, monkeypatch, case):
    """Review finding 3, through the release's own call: the planner home holds an exact copy."""
    _general(monkeypatch, tmp_path / "general")
    src = REPO / "role-skills" / PLANNER / PLAN
    planner = _home(tmp_path / "root" / "profiles" / PLANNER)
    shutil.copytree(src, planner / "skills" / PLAN)
    md = planner / "skills" / PLAN / "SKILL.md"
    if case == "no lock file":
        (planner / "skills" / ".hub").mkdir()
    elif case == "a lock of other paths":  # one is the start of the place's text, not a folder above it
        entry = OWNED["installed"][NAME]
        _hub_lock(planner / "skills", {"version": 1, "installed": {
            "handing-build": {**entry, "install_path": "team-operations/handing-build"},
            "other": {**entry, "install_path": "team-operations/other"}}})
    elif case == "a lock that owns its place":
        _hub_lock(planner / "skills", OWNED)
    owned = case == "a lock that owns its place"
    before, sha, stat = _tree(planner), _sha(md), _stat(md)

    result = _seed(planner)

    assert _lists(result) == {**NO_LISTS, "kept": [{"name": NAME, "reason": "changed"}] if owned else []}
    assert _role_record(planner) == ({} if owned else {NAME: _dir_hash(src)})
    assert _manifest(planner) == {}
    assert (_sha(md), _stat(md)) == (sha, stat)
    record = {} if owned else {f"skills/{ROLE_RECORD}": f"{NAME}:{_dir_hash(src)}\n".encode("utf-8")}
    assert _tree(planner) == {**before, **record}


@pytest.mark.parametrize("case", [
    pytest.param("unreadable", marks=needs_real_permissions),
    "a folder",  # the twin of "unreadable" that holds as root
    "truncated",
    "installed is no map",
])
def test_addition_11_an_exact_copy_is_kept_with_hub_lock_when_the_lock_cannot_be_read_or_has_no_valid_form(
        tmp_path, monkeypatch, case):
    """Review finding 3, through the release's own call: an exact copy at the place its lock names (a
    readable lock keeps it as "changed", test 10). The seeding, with the essential general skill, ends
    as for the twin home of a role with no role skills, in the same run."""
    general = _general(monkeypatch, tmp_path / "general")
    _write_skill(general, "autonomous-ai-agents/hermes-agent", "essential")
    planner = _home(tmp_path / "root" / "profiles" / PLANNER)
    shutil.copytree(REPO / "role-skills" / PLANNER / PLAN, planner / "skills" / PLAN)
    md = planner / "skills" / PLAN / "SKILL.md"
    if case == "a folder":
        (planner / "skills" / ".hub" / "lock.json").mkdir(parents=True)
    else:
        forms = {"truncated": json.dumps(OWNED)[:-5].encode("utf-8"),
                 "installed is no map": {**OWNED, "installed": list(OWNED["installed"].values())}}
        _hub_lock(planner / "skills", forms.get(case, OWNED))
    twin = _twin(planner, tmp_path / "twin", OTHER)
    locks = [home / "skills" / ".hub" / "lock.json" for home in (planner, twin)] if case == "unreadable" else []
    sha = _sha(md)
    try:
        for lock in locks:
            lock.chmod(0)
        with_role, without = _seed(planner), _seed(twin)
    finally:
        for lock in locks:
            lock.chmod(0o644)

    assert without["copied"] == ["hermes-agent"] and without["kept"] == []
    assert with_role == {**without, "kept": [{"name": NAME, "reason": "hub_lock"}]}  # no count changes
    assert _tree(planner) == _tree(twin)
    assert NAME not in _manifest(planner) and NAME not in _role_record(planner) and _sha(md) == sha


@pytest.mark.parametrize("form", [
    pytest.param(b"", id="empty"),
    pytest.param(b"\xff", id="no utf-8"),
    pytest.param([], id="no map"),
    pytest.param({"version": 1}, id="no installed map"),
    pytest.param({"version": 1, "installed": {NAME: "official"}}, id="an entry that is no map"),
    pytest.param({"version": 1, "installed": {"other": {"source": "official"}}}, id="an entry with no install path"),
    pytest.param({"version": 1, "installed": {NAME: {"install_path": 5}}}, id="an install path that is no text"),
    pytest.param({"version": 1, "installed": {NAME: {"install_path": "/"}}}, id="an empty install path"),
])
def test_addition_11_each_lock_form_that_is_not_valid_keeps_the_exact_copy_with_hub_lock(release, monkeypatch, form):
    src = _write_skill(release.roles / PLANNER, PLAN, "planner v1")
    home = _home(release.planner)
    shutil.copytree(src, home / "skills" / PLAN)
    _hub_lock(home / "skills", form)
    before = _tree(home)

    result = _sync(monkeypatch, home)

    assert _lists(result) == {**NO_LISTS, "kept": [{"name": NAME, "reason": "hub_lock"}]}
    assert _tree(home) == before


# -- The addition: a place that the general pass writes (points 12 and 13) ---------------------------

@pytest.mark.parametrize("marked", [True, False], ids=["marker", "no-marker"])
def test_addition_12_13_a_general_description_in_the_place_of_a_role_skill_refuses_it_where_the_general_pass_writes_it(
        tmp_path, monkeypatch, marked):
    """Review finding 5, through the release's own call: an exact copy of the planner skill, and a general
    description inside its place. Only a description the general pass writes in the run refuses the copy
    (review 2 finding 5): with the marker it writes none there, and the copy is adopted in the first
    release; with no marker the first release refuses the copy and writes the description into it, so
    the second finds a changed copy and keeps it. Two releases; in each, the twin home is seeded by a
    role with no role skills."""
    general = _general(monkeypatch, tmp_path / "general")
    _write_skill(general, "autonomous-ai-agents/hermes-agent", "essential")
    (general / PLAN).mkdir(parents=True)
    (general / PLAN / "DESCRIPTION.md").write_text("A general description\n", encoding="utf-8")
    src = REPO / "role-skills" / PLANNER / PLAN
    planner = _home(tmp_path / "root" / "profiles" / PLANNER, marked=marked)
    shutil.copytree(src, planner / "skills" / PLAN)
    twin = _twin(planner, tmp_path / "twin", OTHER)

    for release_no in range(2):  # the second release finds what the first one wrote
        with_role, without = _seed(planner), _seed(twin)

        # Own-record points 10 and 11: a description is no place of a general skill, so no overlap; with
        # no marker the general pass wrote it into the copy before the role pass, which keeps it (line d).
        if marked:
            assert with_role == without  # adopted: in no list, and no count changes
        else:
            assert with_role == {**without, "kept": [{"name": NAME, "reason": "changed"}]}
        assert without["refused"] == [] and _without_record(planner) == _without_record(twin)
        assert _manifest(planner) == _manifest(twin)
        assert _role_record(planner) == ({NAME: _dir_hash(src)} if marked else {})
    assert (planner / "skills" / PLAN / "DESCRIPTION.md").is_file() == (not marked)  # the general pass's own


@pytest.mark.parametrize("case", [
    "its own description where the general pass writes one",
    "a general skill inside its place",
    "its place inside a general skill",
    "a general skill at its place, by another name",
    "a general skill at its backup place",
    "the record file at its place",
])
def test_addition_12_13_a_role_skill_where_the_general_pass_writes_is_refused_and_the_general_pass_is_as_it_is(
        release, monkeypatch, tmp_path, case):
    """Review finding 5, in a fresh role with no marker, where the general pass writes the most. The twin
    home is synced in the same run with no role skills."""
    rel, name = (".bundled_manifest", "record") if case == "the record file at its place" else (PLAN, NAME)
    src = _write_skill(release.roles / PLANNER, rel, "planner", name=name)
    if case == "its own description where the general pass writes one":
        (src / "DESCRIPTION.md").write_text("The role's description\n", encoding="utf-8")
        (release.skills / PLAN).mkdir(parents=True)
        (release.skills / PLAN / "DESCRIPTION.md").write_text("The general description\n", encoding="utf-8")
    else:
        general_rel, general_name = {
            "a general skill inside its place": (f"{PLAN}/extra", "extra"),
            "its place inside a general skill": ("team-operations", "team-operations"),
            "a general skill at its place, by another name": (PLAN, "other-name"),
            "a general skill at its backup place": (f"{PLAN}.bak", "backup-name"),
            "the record file at its place": ("software-development/spike", "spike"),
        }[case]
        _write_skill(release.skills, general_rel, "general", name=general_name)
    home = _home(release.planner, marked=False)
    twin = _twin(home, tmp_path / "twin")

    with_role = _sync(monkeypatch, home)
    _use_role_skills(monkeypatch, tmp_path / "no-role-skills")
    without = _sync(monkeypatch, twin)

    # Own-record points 10 and 11: only a general skill's place gives overlap; a description, a backup
    # place or the record file does not, and the table decides (line d for what the general pass left).
    role = {"its own description where the general pass writes one": {"kept": [{"name": name, "reason": "changed"}]},
            "a general skill at its backup place": {"copied": [name]},
            "the record file at its place": {"kept": [{"name": name, "reason": "changed"}]},
            }.get(case, {"refused": [{"name": name, "reason": "overlap"}]})
    assert with_role == {key: value + role[key] if key in role else value for key, value in without.items()}
    assert _manifest(home) == _manifest(twin)
    if case == "a general skill at its backup place":
        assert _outside(home, [PLAN]) == _outside(twin, [PLAN])
        assert _files(home / "skills" / PLAN) == _files(src) and _role_record(home) == {name: _dir_hash(src)}
    else:
        assert _tree(home) == _tree(twin)


# -- Review round 2 -----------------------------------------------------------------------------------

@needs_symlinks
@pytest.mark.parametrize("catalogue", [True, False], ids=["general catalogue", "no general catalogue"])
@pytest.mark.parametrize("record", ["a link to the builder's record", "a folder"])
def test_review2_1_a_role_whose_record_file_is_a_link_or_no_regular_file_gets_nothing_and_one_refusal(
        tmp_path, monkeypatch, record, catalogue):
    """Review 2 finding 1, through the release's own call: the planner's record file is a link to the
    builder's, seeded first, or a folder. No role file and no role line is written, there or through it."""
    if catalogue:
        _general(monkeypatch, tmp_path / "general")
    else:
        monkeypatch.setenv("HERMES_BUNDLED_SKILLS", str(tmp_path / "not-there"))
    roles = tmp_path / "root" / "profiles"
    builder = _home(roles / BUILDER)
    assert _seed(builder)["copied"] == ["running-build-work"]
    planner = _home(roles / PLANNER)
    manifest = planner / "skills" / ".bundled_manifest"
    manifest.unlink()
    if record == "a folder":
        manifest.mkdir()
    else:
        manifest.symlink_to(builder / "skills" / ".bundled_manifest")
    before = _tree(roles)

    result = _seed(planner)

    refused = [{"name": PLANNER, "reason": "link"}]
    if catalogue:
        assert _lists(result) == {**NO_LISTS, "refused": refused}
    else:
        assert result == {**NO_CATALOGUE, "refused": refused}
    assert _tree(roles) == before


@needs_fifo
def test_review2_1_with_no_general_catalogue_a_record_file_that_is_a_fifo_is_refused_and_never_opened(
        release, monkeypatch, tmp_path):
    _write_skill(release.roles / PLANNER, PLAN, "planner")
    monkeypatch.setenv("HERMES_BUNDLED_SKILLS", str(tmp_path / "not-there"))
    home = _home(release.planner)
    manifest = home / "skills" / ".bundled_manifest"
    manifest.unlink()
    os.mkfifo(manifest)
    before = _tree(home)

    result = _in_time(lambda: _sync(monkeypatch, home), manifest)

    assert result == {**NO_CATALOGUE, "refused": [{"name": PLANNER, "reason": "link"}]}
    assert _tree(home) == before


@pytest.mark.parametrize("marked", [True, False], ids=["marker", "no-marker"])
def test_review2_2_a_role_record_is_no_evidence_for_a_later_general_skill_of_the_same_name(
        checkout, tmp_path, monkeypatch, marked):
    """Review 2 finding 2, through the release's own call, two releases: the first adopts an exact role
    copy; the second adds a general skill of the same name at another inner path. The twin home holds the
    same files and is seeded with no role skills: the general result and bytes are its own, the role
    copy stays at its place with its bytes, and the role skill is refused with ``name``."""
    general = _general(monkeypatch, tmp_path / "general")
    src = _write_skill(checkout / PLANNER, PLAN, "planner")
    planner = _home(tmp_path / "root" / "profiles" / PLANNER, marked=marked)
    shutil.copytree(src, planner / "skills" / PLAN)
    twin = _twin(planner, tmp_path / "twin", OTHER)
    copy = _files(planner / "skills" / PLAN)
    first, first_twin = _seed(planner), _seed(twin)
    assert first == first_twin  # adopted: in no list, and no count changes
    assert _role_record(planner) == {NAME: _dir_hash(src)} and _manifest(planner) == _manifest(twin)
    _write_skill(general, f"software-development/{NAME}", "the general one")

    with_role, without = _seed(planner), _seed(twin)

    assert with_role == {**without, "refused": [{"name": NAME, "reason": "name"}]}
    assert without["copied"] == ([] if marked else [NAME]) and without["relocated"] == []
    assert _files(planner / "skills" / PLAN) == copy
    assert _without_record(planner) == _without_record(twin)
    assert _manifest(planner) == _manifest(twin)
    assert _role_record(planner) == {NAME: _dir_hash(src)}


@pytest.mark.parametrize("marked", [True, False], ids=["marker", "no-marker"])
def test_review2_2_a_general_record_stays_one_when_a_role_skill_of_the_same_name_comes(
        checkout, tmp_path, monkeypatch, marked):
    """Review 2 finding 2, the other way: the general skill is delivered first, with its line; then a role
    skill of the same name comes at another inner path, and the general skill changes. The general
    result is the twin's, and the role skill is refused with ``name``."""
    general = _general(monkeypatch, tmp_path / "general")
    _write_skill(general, f"software-development/{NAME}", "the general one")
    planner = _home(tmp_path / "root" / "profiles" / PLANNER, marked=False)
    twin = _twin(planner, tmp_path / "twin", OTHER)
    assert _seed(planner)["copied"] == [NAME] and _seed(twin)["copied"] == [NAME]
    if marked:
        _mark(planner), _mark(twin)
    _write_skill(checkout / PLANNER, PLAN, "planner")
    _write_skill(general, f"software-development/{NAME}", "the general one, changed")

    with_role, without = _seed(planner), _seed(twin)

    assert with_role == {**without, "refused": [{"name": NAME, "reason": "name"}]}
    assert without["updated"] == [NAME]
    assert _tree(planner) == _tree(twin)


@pytest.mark.parametrize("case", [
    "no version", "another version", "an install path that is not its entry's name",
    "an install path out of skills/", "one entry valid and one not",
    pytest.param("a linked hub folder", marks=needs_symlinks),
    pytest.param("a linked lock file", marks=needs_symlinks),
    pytest.param("a fifo lock file", marks=needs_fifo),
])
def test_review2_3_an_exact_copy_is_kept_with_hub_lock_when_the_lock_is_not_the_hubs_valid_form(
        release, monkeypatch, tmp_path, case):
    """Review 2 finding 3: the lock's version, each entry's install path by the hub's own validator, and
    the lock's own entry type (a link or a FIFO is never opened). The seeding goes on."""
    src = _write_skill(release.roles / PLANNER, PLAN, "planner v1")
    home = _home(release.planner)
    shutil.copytree(src, home / "skills" / PLAN)
    entry = OWNED["installed"][NAME]
    forms = {
        "no version": {"installed": {}},
        "another version": {"version": 2, "installed": {}},
        "an install path that is not its entry's name": {"version": 1, "installed": {"other": entry}},
        "an install path out of skills/": {"version": 1, "installed": {NAME: {**entry, "install_path": f"../{NAME}"}}},
        "one entry valid and one not": {"version": 1, "installed": {
            "spike": {**entry, "install_path": "software-development/spike"},
            "bad": {**entry, "install_path": "software-development/other"}}},
    }
    lock = home / "skills" / ".hub" / "lock.json"
    if case in forms:
        _hub_lock(home / "skills", forms[case])
    elif case == "a linked hub folder":
        (home / "skills" / ".hub").symlink_to(_hub_lock(tmp_path / "elsewhere", {"version": 1, "installed": {}}).parent)
    elif case == "a linked lock file":
        lock.parent.mkdir()
        lock.symlink_to(_hub_lock(tmp_path / "elsewhere", {"version": 1, "installed": {}}))
    else:
        lock.parent.mkdir()
        os.mkfifo(lock)
    before = _tree(home)

    result = _in_time(lambda: _sync(monkeypatch, home), lock)

    assert _lists(result) == {**NO_LISTS, "kept": [{"name": NAME, "reason": "hub_lock"}]}
    assert _tree(home) == before


@pytest.mark.parametrize("name, path", [
    pytest.param(NAME, f"./{PLAN}/", id="its place written another way"),
    pytest.param("inner", f"{PLAN}/inner", id="a place inside its place"),
    pytest.param("team-operations", "team-operations", id="a place that holds its place"),
])
def test_review2_3_a_valid_lock_owns_the_place_by_the_hubs_normal_form_and_keeps_the_exact_copy(
        release, monkeypatch, name, path):
    src = _write_skill(release.roles / PLANNER, PLAN, "planner v1")
    home = _home(release.planner)
    shutil.copytree(src, home / "skills" / PLAN)
    _hub_lock(home / "skills", {"version": 1, "installed": {name: {"source": "official", "install_path": path}}})
    before = _tree(home)

    result = _sync(monkeypatch, home)

    assert _lists(result) == {**NO_LISTS, "kept": [{"name": NAME, "reason": "changed"}]}
    assert _tree(home) == before


def test_review2_3_the_supported_lock_version_is_the_hubs():
    from tools.skills_hub import HubLockFile
    assert skills_sync_roles.HUB_LOCK_VERSION == HubLockFile.EMPTY["version"]


@needs_fifo
def test_review2_4_a_skill_md_that_is_a_fifo_is_refused_at_once_under_its_folder_name(release, monkeypatch):
    md = release.roles / PLANNER / PLAN / "SKILL.md"
    md.parent.mkdir(parents=True)
    os.mkfifo(md)
    home = _home(release.planner)
    before = _tree(home)

    result = _in_time(lambda: _sync(monkeypatch, home), md)

    assert _lists(result) == {**NO_LISTS, "refused": [{"name": NAME, "reason": "link"}]}
    assert _tree(home) == before


def test_review2_4_8_with_no_link_the_walk_finds_what_the_general_discovery_finds(release):
    """Facts point 8 on a plain tree: same names and folders, skills inside skills and excluded paths
    included; the walk adds only its refusals."""
    role_dir = release.roles / PLANNER
    for rel, name in ((PLAN, ""), ("data/etl/loader", "custom-name"), (f"{PLAN}/inner", ""),
                      (f"{PLAN}/scripts/helper", ""), ("tools/node_modules/hidden", ""), ("a/.git/x", "")):
        _write_skill(role_dir, rel, "x", name=name)
    (role_dir / "data" / "notes.md").write_text("not a skill\n", encoding="utf-8")

    walked = skills_sync_roles._role_skills(PLANNER)[1]

    assert [(name, folder) for name, folder, _ in walked] == sorted(
        _discover_bundled_skills(role_dir), key=lambda item: item[1])
    assert {folder.name: refusal for _, folder, refusal in walked} == {
        "handing-build-work": "nested", "inner": "nested", "loader": None}


@pytest.mark.parametrize("marked", [True, False], ids=["marker", "no-marker"])
def test_review2_5_an_exact_copy_around_a_description_is_adopted_only_where_the_general_pass_does_not_write_it(
        tmp_path, monkeypatch, marked):
    """Review 2 finding 5, through the release's own call: with the marker the general pass writes no
    description into the exact copy, which is adopted (its line, and no byte or change time of its
    files); with no marker it writes the description, as on the twin, and the copy is refused."""
    general = _general(monkeypatch, tmp_path / "general")
    _write_skill(general, "autonomous-ai-agents/hermes-agent", "essential")
    (general / PLAN).mkdir(parents=True)
    (general / PLAN / "DESCRIPTION.md").write_text("A general description\n", encoding="utf-8")
    src = REPO / "role-skills" / PLANNER / PLAN
    planner = _home(tmp_path / "root" / "profiles" / PLANNER, marked=marked)
    copy = planner / "skills" / PLAN
    shutil.copytree(src, copy)
    twin = _twin(planner, tmp_path / "twin", OTHER)
    before = {path: _stat(path) for path in [copy, *sorted(copy.rglob("*"))]}, _files(copy)

    with_role, without = _seed(planner), _seed(twin)

    if marked:
        assert with_role == without  # adopted: in no list, and no count changes
        assert _manifest(planner) == _manifest(twin) and _role_record(planner) == {NAME: _dir_hash(src)}
        assert ({path: _stat(path) for path in [copy, *sorted(copy.rglob("*"))]}, _files(copy)) == before
        assert _without_record(planner) == _without_record(twin)
    else:  # own-record point 11: no overlap; the copy the general pass wrote into is kept (line d)
        assert with_role == {**without, "kept": [{"name": NAME, "reason": "changed"}]}
        assert _tree(planner) == _tree(twin) and NAME not in _manifest(planner) and _role_record(planner) == {}
    assert (copy / "DESCRIPTION.md").is_file() == (not marked)


@pytest.mark.parametrize("general_rel", [PLAN, "team-operations"], ids=["at its place", "holding its place"])
def test_review2_5_a_role_skill_at_or_inside_a_general_skills_place_is_refused_whatever_the_run(
        release, monkeypatch, general_rel):
    """Addition point 12: with the marker the general pass does not write that general skill, and the
    exact role copy is still refused."""
    src = _write_skill(release.roles / PLANNER, PLAN, "planner")
    _write_skill(release.skills, general_rel, "general", name="other-name")
    home = _home(release.planner)
    shutil.copytree(src, home / "skills" / PLAN)
    before = _tree(home)

    result = _sync(monkeypatch, home)

    assert _lists(result) == {**NO_LISTS, "refused": [{"name": NAME, "reason": "overlap"}]}
    assert _tree(home) == before


# -- The owner's own-record rule after review round 3 (points 5 to 14 and 20 to 23) ------------------

@pytest.mark.parametrize("catalogue", [True, False], ids=["general catalogue", "no general catalogue"])
def test_own_record_5_the_role_pass_runs_once_after_the_general_pass_ended_at_both_returns(
        release, monkeypatch, tmp_path, catalogue):
    """The call sites import ``sync_role_skills`` at call time: the wrapper sees each call, with the
    general result, and ``.bundled_manifest`` on disk already as the run leaves it."""
    src = _write_skill(release.roles / PLANNER, PLAN, "planner")
    _write_skill(release.skills, "general-cat/general-skill", "general")
    if not catalogue:
        monkeypatch.setenv("HERMES_BUNDLED_SKILLS", str(tmp_path / "not-there"))
    home = _home(release.planner, marked=False)
    control = _twin(home, tmp_path / "control")
    _use_role_skills(monkeypatch, tmp_path / "no-role-skills")
    expected = _sync(monkeypatch, control)
    _use_role_skills(monkeypatch, release.roles)
    calls = []
    real = skills_sync_roles.sync_role_skills

    def wrapped(result, quiet=False):
        calls.append((json.loads(json.dumps(result)), (home / "skills" / ".bundled_manifest").read_bytes()))
        return real(result, quiet)
    monkeypatch.setattr(skills_sync_roles, "sync_role_skills", wrapped)

    result = _sync(monkeypatch, home)

    assert len(calls) == 1
    general, record_at_call = calls[0]
    assert general == expected
    assert record_at_call == (home / "skills" / ".bundled_manifest").read_bytes()
    assert _manifest(home) == _manifest(control) == (
        {"general-skill": _dir_hash(release.skills / "general-cat" / "general-skill")} if catalogue else {})
    assert result == {**expected, "copied": expected["copied"] + [NAME]}
    assert _role_record(home) == {NAME: _dir_hash(src)}


@pytest.mark.parametrize("marked", [True, False], ids=["marker", "no-marker"])
def test_own_record_6_the_role_record_is_its_own_file_beside_the_general_record_made_with_its_first_line(
        release, monkeypatch, tmp_path, marked):
    src = _write_skill(release.roles / PLANNER, PLAN, "planner")
    _write_skill(release.skills, "general-cat/general-skill", "general")
    home = _home(release.planner, marked=marked)
    control = _twin(home, tmp_path / "control")
    record = home / "skills" / ROLE_RECORD

    result = _sync(monkeypatch, home)
    _use_role_skills(monkeypatch, tmp_path / "no-role-skills")
    expected = _sync(monkeypatch, control)

    assert result == {**expected, "copied": expected["copied"] + [NAME]}
    assert record.read_text(encoding="utf-8") == f"{NAME}:{_dir_hash(src)}\n"
    assert record.parent == (home / "skills" / ".bundled_manifest").parent
    assert _role_record(home) == {NAME: _dir_hash(src)}
    assert NAME not in _manifest(home)
    assert _manifest(home) == _manifest(control)
    assert _outside(home, [PLAN]) == _outside(control, [PLAN])
    assert not os.path.lexists(control / "skills" / ROLE_RECORD)


@pytest.mark.parametrize("marked", [True, False], ids=["marker", "no-marker"])
def test_own_record_20_1_a_role_skill_with_a_general_name_and_its_old_text_is_refused_as_the_general_skill_updates(
        tmp_path, monkeypatch, checkout, marked):
    """Sequence 1 of the third review: release 1 delivers a general skill with its line; release 2
    changes its text and newly offers a role skill of its name and its release-1 text at another place."""
    general = _general(monkeypatch, tmp_path / "release" / "skills")
    _write_skill(general, "general-cat/shared", "release 1")
    home = _home(tmp_path / "root" / "profiles" / PLANNER, marked=False)
    places = ["team-operations/shared"]

    assert _step(home, tmp_path / "control-1", checkout, places, {})["copied"] == ["shared"]
    if marked:
        _mark(home)  # the line is there: the marked role gets the update of an unchanged copy
    _write_skill(checkout / PLANNER, "team-operations/shared", "release 1")
    _write_skill(general, "general-cat/shared", "release 2")
    assert _dir_hash(checkout / PLANNER / "team-operations" / "shared") == _manifest(home)["shared"]

    result = _step(home, tmp_path / "control-2", checkout, places, {"refused": [{"name": "shared", "reason": "name"}]})

    assert result["updated"] == ["shared"]
    assert _manifest(home) == {"shared": _dir_hash(general / "general-cat" / "shared")}
    assert not os.path.lexists(home / "skills" / "team-operations")
    assert not os.path.lexists(home / "skills" / ROLE_RECORD)


@pytest.mark.parametrize("marked", [True, False], ids=["marker", "no-marker"])
def test_own_record_20_2_a_general_skill_moves_its_category_and_a_role_skill_takes_the_old_place_and_name(
        tmp_path, monkeypatch, checkout, marked):
    """Sequence 2: release 1 delivers a general skill in category A; release 2 moves it to B and offers a
    role skill of its name at A. With no marker the general pass relocates and updates it as the control."""
    general = _general(monkeypatch, tmp_path / "release" / "skills")
    _write_skill(general, "cat-a/moving", "release 1")
    home = _home(tmp_path / "root" / "profiles" / PLANNER, marked=False)
    places = ["cat-a/moving"]
    assert _step(home, tmp_path / "control-1", checkout, places, {})["copied"] == ["moving"]
    if marked:
        _mark(home)
    shutil.rmtree(general / "cat-a")
    _write_skill(general, "cat-b/moving", "release 2")
    _write_skill(checkout / PLANNER, "cat-a/moving", "the role's")

    result = _step(home, tmp_path / "control-2", checkout, places, {"refused": [{"name": "moving", "reason": "name"}]})

    assert (result["relocated"], result["updated"]) == (([], []) if marked else (["moving"], ["moving"]))
    assert _role_record(home) == {}


@pytest.mark.parametrize("marked", [True, False], ids=["marker", "no-marker"])
def test_own_record_20_3_an_adopted_role_copy_changed_by_a_person_and_then_a_general_skill_of_its_name(
        tmp_path, monkeypatch, checkout, marked):
    """Sequence 3: an exact role copy is adopted; then a person changes it and the released role text
    changes; then a general skill takes the same name at another place. No role line is evidence for
    the general pass."""
    general = _general(monkeypatch, tmp_path / "release" / "skills")
    src = _write_skill(checkout / PLANNER, PLAN, "release 1")
    home = _home(tmp_path / "root" / "profiles" / PLANNER, marked=marked)
    shutil.copytree(src, home / "skills" / PLAN)
    line = {NAME: _dir_hash(src)}
    _step(home, tmp_path / "control-1", checkout, [PLAN], {})  # adopted: in no list
    assert _role_record(home) == line
    _edit(home / "skills" / PLAN)
    copy = _files(home / "skills" / PLAN)
    _write_skill(checkout / PLANNER, PLAN, "release 2")
    _step(home, tmp_path / "control-2", checkout, [PLAN],
          {"user_modified": [NAME], "kept": [{"name": NAME, "reason": "changed"}]})
    _write_skill(general, f"software-development/{NAME}", "the general one")

    result = _step(home, tmp_path / "control-3", checkout, [PLAN], {"refused": [{"name": NAME, "reason": "name"}]})

    assert result["copied"] == ([] if marked else [NAME])  # the control's: the role line is not read
    assert _files(home / "skills" / PLAN) == copy and _role_record(home) == line


@pytest.mark.parametrize("marked", [True, False], ids=["marker", "no-marker"])
def test_own_record_20_4_a_general_skill_retires_as_a_role_skill_takes_its_name_at_its_old_place(
        tmp_path, monkeypatch, checkout, marked):
    """Sequence 4: with no marker the cleanup drops the general line and the role skill finds the old
    general copy, changed against its text: kept (line d). With the marker the general line stays, and
    the role skill is refused with ``name``."""
    general = _general(monkeypatch, tmp_path / "release" / "skills")
    _write_skill(general, "cat-g/retiring", "the general text")
    home = _home(tmp_path / "root" / "profiles" / PLANNER, marked=False)
    places = ["cat-g/retiring"]
    assert _step(home, tmp_path / "control-1", checkout, places, {})["copied"] == ["retiring"]
    if marked:
        _mark(home)
    old = _files(home / "skills" / "cat-g" / "retiring")
    shutil.rmtree(general / "cat-g")
    _write_skill(checkout / PLANNER, "cat-g/retiring", "the role text")

    role = ({"refused": [{"name": "retiring", "reason": "name"}]} if marked
            else {"kept": [{"name": "retiring", "reason": "changed"}]})
    result = _step(home, tmp_path / "control-2", checkout, places, role)

    assert result["cleaned"] == ([] if marked else ["retiring"])
    assert ("retiring" in _manifest(home)) == marked
    assert _files(home / "skills" / "cat-g" / "retiring") == old and _role_record(home) == {}


@pytest.mark.parametrize("marked", [True, False], ids=["marker", "no-marker"])
def test_own_record_20_5_a_role_skill_seeded_where_a_later_general_skill_has_its_backup_place(
        tmp_path, monkeypatch, checkout, marked):
    """Sequence 5: a role skill at ``cat-x/tool.bak``; release 2 adds a general skill at ``cat-x/tool``,
    release 3 changes it. With no marker the general pass moves the "orphan backup" to ``cat-x/tool`` as
    in the control and keeps it; the role line stays and the role pass writes nothing (line b). With the
    marker the general skill is not seeded and the role copy stays at its place."""
    general = _general(monkeypatch, tmp_path / "release" / "skills")
    src = _write_skill(checkout / PLANNER, "cat-x/tool.bak", "the role's", name="role-tool")
    home = _home(tmp_path / "root" / "profiles" / PLANNER, marked=marked)
    places = ["cat-x/tool.bak"]
    _step(home, tmp_path / "control-1", checkout, places, {"copied": ["role-tool"]})
    line, copy = _role_record(home), _files(home / "skills" / "cat-x" / "tool.bak")
    assert line == {"role-tool": _dir_hash(src)}

    for release_no, text in ((2, "v1"), (3, "v2")):
        _write_skill(general, "cat-x/tool", text, name="tool")
        result = _step(home, tmp_path / f"control-{release_no}", checkout, places, {})

        assert result["kept"] == ([] if marked else [{"name": "tool", "reason": "changed"}])
        assert _role_record(home) == line
        assert _files(home / "skills" / "cat-x" / ("tool.bak" if marked else "tool")) == copy
        assert os.path.lexists(home / "skills" / "cat-x" / "tool.bak") == marked
        assert "tool" not in _manifest(home)


@pytest.mark.parametrize("marked", [True, False], ids=["marker", "no-marker"])
def test_own_record_20_6_a_role_skill_seeded_beneath_a_folder_that_later_becomes_a_general_skill(
        tmp_path, monkeypatch, checkout, marked):
    general = _general(monkeypatch, tmp_path / "release" / "skills")
    src = _write_skill(checkout / PLANNER, "cat-y/outer/inner-skill", "the role's")
    home = _home(tmp_path / "root" / "profiles" / PLANNER, marked=marked)
    places = ["cat-y/outer/inner-skill"]
    _step(home, tmp_path / "control-1", checkout, places, {"copied": ["inner-skill"]})
    copy = _files(home / "skills" / "cat-y" / "outer")
    _write_skill(general, "cat-y/outer", "general", name="outer")

    result = _step(home, tmp_path / "control-2", checkout, places,
                   {"refused": [{"name": "inner-skill", "reason": "overlap"}]})

    assert result["kept"] == ([] if marked else [{"name": "outer", "reason": "changed"}])  # it keeps the folder
    assert _files(home / "skills" / "cat-y" / "outer") == copy
    assert _role_record(home) == {"inner-skill": _dir_hash(src)} and "outer" not in _manifest(home)


@pytest.mark.parametrize("marked", [True, False], ids=["marker", "no-marker"])
def test_own_record_20_7_an_external_folder_provides_the_name_of_an_adopted_copy_that_holds_a_description(
        tmp_path, monkeypatch, checkout, marked):
    general = _general(monkeypatch, tmp_path / "release" / "skills")
    _write_skill(general, "autonomous-ai-agents/hermes-agent", "essential")
    (general / PLAN).mkdir(parents=True)
    (general / PLAN / "DESCRIPTION.md").write_text("The general description\n", encoding="utf-8")
    src = _write_skill(checkout / PLANNER, PLAN, "the role's")
    (src / "DESCRIPTION.md").write_text("The role's description\n", encoding="utf-8")
    home = _home(tmp_path / "root" / "profiles" / PLANNER, marked=marked)
    copy = home / "skills" / PLAN
    shutil.copytree(src, copy)
    _step(home, tmp_path / "control-1", checkout, [PLAN], {})  # adopted: in no list
    assert _role_record(home) == {NAME: _dir_hash(src)}
    outside = tmp_path / "external"
    _write_skill(outside, f"devops/{NAME}", "external")
    (home / "config.yaml").write_text(f"skills:\n  external_dirs:\n    - {outside}\n", encoding="utf-8")
    before = {path: _stat(path) for path in [copy, *sorted(copy.rglob("*"))]}, _files(copy)

    _step(home, tmp_path / "control-2", checkout, [PLAN], {"refused": [{"name": NAME, "reason": "name"}]})

    assert ({path: _stat(path) for path in [copy, *sorted(copy.rglob("*"))]}, _files(copy)) == before
    assert _files(copy) == _files(src) and _role_record(home) == {NAME: _dir_hash(src)}


def test_own_record_14_no_role_skill_of_this_repository_shares_a_name_or_a_place_with_a_general_skill():
    general = _discover_bundled_skills(REPO / "skills")
    names = {name for name, _ in general}
    places = [src.relative_to(REPO / "skills") for _, src in general]
    assert names and places  # it reads the release's general skills
    pairs = []
    for role_dir in sorted(path for path in (REPO / "role-skills").iterdir() if path.is_dir()):
        for name, src in _discover_bundled_skills(role_dir):
            place = src.relative_to(role_dir)
            pairs += [(role_dir.name, name, place.as_posix(), other.as_posix()) for other in places
                      if place == other or other in place.parents or place in other.parents]
            pairs += [(role_dir.name, name, place.as_posix(), "name")] if name in names else []
    assert pairs == []
    assert {role.name for role in (REPO / "role-skills").iterdir()} >= {PLANNER, BUILDER}


@pytest.mark.parametrize("case", ["a role with no role skill", "a role whose role skills are all refused"])
def test_own_record_6_a_role_whose_role_skills_write_no_line_gets_no_record_file(release, monkeypatch, tmp_path, case):
    _write_skill(release.roles / PLANNER, PLAN, "planner")
    _write_skill(release.roles / PLANNER, "other-cat/second", "second")
    if case == "a role whose role skills are all refused":
        _write_skill(release.skills, f"software-development/{NAME}", "general")  # name
        _write_skill(release.skills, "other-cat", "general", name="holder")  # overlap
        home = _home(release.planner, marked=False)
    else:
        home = _home(release.root / "profiles" / OTHER, marked=False)
    control = _twin(home, tmp_path / "control")

    result = _sync(monkeypatch, home)
    _use_role_skills(monkeypatch, tmp_path / "no-role-skills")
    expected = _sync(monkeypatch, control)

    refused = ([{"name": "second", "reason": "overlap"}, {"name": NAME, "reason": "name"}]  # catalogue order
               if case == "a role whose role skills are all refused" else [])
    assert result == {**expected, "refused": expected["refused"] + refused}
    assert not os.path.lexists(home / "skills" / ROLE_RECORD)
    assert _tree(home) == _tree(control)


def test_own_record_6_a_record_write_that_fails_leaves_the_earlier_record_file_and_no_temporary_file(
        release, monkeypatch):
    import utils
    first = _write_skill(release.roles / PLANNER, PLAN, "planner")
    home = _home(release.planner)
    assert _sync(monkeypatch, home)["copied"] == [NAME]
    record = home / "skills" / ROLE_RECORD
    earlier = record.read_bytes()
    second = _write_skill(release.roles / PLANNER, "other-cat/second", "second")
    real = utils.atomic_replace

    def failing(src, dst, *args, **kwargs):
        if Path(dst).name == ROLE_RECORD:
            raise OSError("disk full")
        return real(src, dst, *args, **kwargs)
    monkeypatch.setattr(utils, "atomic_replace", failing)

    result = _sync(monkeypatch, home)

    assert result["copied"] == ["second"]  # the copy stands; its line was not written
    assert record.read_bytes() == earlier == f"{NAME}:{_dir_hash(first)}\n".encode("utf-8")
    assert sorted(path.name for path in (home / "skills").iterdir() if path.is_file()) == [
        ".bundled_manifest", ROLE_RECORD]
    monkeypatch.setattr(utils, "atomic_replace", real)
    assert _lists(_sync(monkeypatch, home)) == NO_LISTS  # the next sync adopts the exact copy
    assert _role_record(home) == {NAME: _dir_hash(first), "second": _dir_hash(second)}


@pytest.mark.parametrize("marked", [True, False], ids=["marker", "no-marker"])
def test_own_record_8_one_release_copies_adopts_and_updates_and_writes_nowhere_else(
        release, monkeypatch, tmp_path, marked):
    _write_skill(release.skills, "general-cat/general-skill", "general v1")
    (release.skills / "general-cat" / "DESCRIPTION.md").write_text("General\n", encoding="utf-8")
    _write_skill(release.skills, "autonomous-ai-agents/hermes-agent", "essential v1")
    _write_skill(release.roles / PLANNER, "team-operations/updated-skill", "v1")
    home = _home(release.planner, marked=marked)
    assert _sync(monkeypatch, home)["copied"][-1:] == ["updated-skill"]
    new = _write_skill(release.roles / PLANNER, "team-operations/new-skill", "new")
    adopted = _write_skill(release.roles / PLANNER, "team-operations/adopted-skill", "adopted")
    shutil.copytree(adopted, home / "skills" / "team-operations" / "adopted-skill")
    updated = _write_skill(release.roles / PLANNER, "team-operations/updated-skill", "v2")
    _write_skill(release.skills, "general-cat/general-skill", "general v2")
    control = _twin(home, tmp_path / "control")
    places = [f"team-operations/{name}" for name in ("new-skill", "adopted-skill", "updated-skill")]

    result = _sync(monkeypatch, home)
    _use_role_skills(monkeypatch, tmp_path / "no-role-skills")
    expected = _sync(monkeypatch, control)

    assert result == {**expected, "copied": expected["copied"] + ["new-skill"],
                      "updated": expected["updated"] + ["updated-skill"]}
    assert _outside(home, places) == _outside(control, places)
    assert _manifest(home) == _manifest(control)
    assert not [path for path in (home / "skills").rglob("*.bak")]
    assert _role_record(home) == {name: _dir_hash(src) for name, src in (
        ("new-skill", new), ("adopted-skill", adopted), ("updated-skill", updated))}
    for name, src in (("new-skill", new), ("adopted-skill", adopted), ("updated-skill", updated)):
        assert _files(home / "skills" / "team-operations" / name) == _files(src)


def test_own_record_9_a_file_outside_a_skill_folder_of_the_role_catalogue_is_not_copied(release, monkeypatch):
    _write_skill(release.roles / PLANNER, PLAN, "planner")
    (release.roles / PLANNER / "team-operations" / "DESCRIPTION.md").write_text("Team\n", encoding="utf-8")
    (release.roles / PLANNER / "README.md").write_text("no skill here\n", encoding="utf-8")
    home = _home(release.planner)

    assert _sync(monkeypatch, home)["copied"] == [NAME]
    assert sorted(_files(home / "skills")) == [".bundled_manifest", ROLE_RECORD, f"{PLAN}/SKILL.md"]


@needs_symlinks
@pytest.mark.parametrize("record", ["a link", "a fifo", "a folder"])
def test_own_record_10_link_a_role_record_that_is_no_regular_file_refuses_the_role_and_is_never_opened(
        release, monkeypatch, tmp_path, record):
    if record == "a fifo" and not hasattr(os, "mkfifo"):
        pytest.skip("named pipes are POSIX")
    _write_skill(release.roles / PLANNER, PLAN, "planner")
    home = _home(release.planner)
    path = home / "skills" / ROLE_RECORD
    if record == "a link":
        (tmp_path / "elsewhere").write_text(f"{NAME}:0\n", encoding="utf-8")
        path.symlink_to(tmp_path / "elsewhere")
    elif record == "a fifo":
        os.mkfifo(path)
    else:
        path.mkdir()
    before = _tree(home)

    result = _in_time(lambda: _sync(monkeypatch, home), path) if record == "a fifo" else _sync(monkeypatch, home)

    assert _lists(result) == {**NO_LISTS, "refused": [{"name": PLANNER, "reason": "link"}]}
    assert _tree(home) == before  # no copy, and the record file is as it was
    if record == "a link":
        assert (tmp_path / "elsewhere").read_text(encoding="utf-8") == f"{NAME}:0\n"


@pytest.mark.parametrize("source", ["a general skill", "a line of .bundled_manifest", "an external skills folder"])
def test_own_record_10_name_each_source_refuses_an_exact_copy_before_a_write(release, monkeypatch, tmp_path, source):
    src = _write_skill(release.roles / PLANNER, PLAN, "planner")
    home = _home(release.planner)  # the marker: no cleanup drops a line of .bundled_manifest
    shutil.copytree(src, home / "skills" / PLAN)
    if source == "a general skill":
        _write_skill(release.skills, f"software-development/{NAME}", "general")
    elif source == "a line of .bundled_manifest":
        (home / "skills" / ".bundled_manifest").write_text(f"{NAME}:0123\n", encoding="utf-8")
    else:
        _write_skill(tmp_path / "external", f"devops/{NAME}", "external")
        (home / "config.yaml").write_text(f"skills:\n  external_dirs:\n    - {tmp_path / 'external'}\n",
                                          encoding="utf-8")
    control = _twin(home, tmp_path / "control")

    result = _sync(monkeypatch, home)
    _use_role_skills(monkeypatch, tmp_path / "no-role-skills")
    expected = _sync(monkeypatch, control)

    assert result == {**expected, "refused": expected["refused"] + [{"name": NAME, "reason": "name"}]}
    assert _tree(home) == _tree(control)
    assert not os.path.lexists(home / "skills" / ROLE_RECORD)


@pytest.mark.parametrize("marked", [True, False], ids=["marker", "no-marker"])
@pytest.mark.parametrize("general_rel", [PLAN, "team-operations", f"{PLAN}/inner"],
                         ids=["equal places", "the role place inside a general place", "a general place inside"])
def test_own_record_10_overlap_comes_from_the_places_of_general_skills(
        release, monkeypatch, tmp_path, general_rel, marked):
    _write_skill(release.roles / PLANNER, PLAN, "planner")
    _write_skill(release.skills, general_rel, "general", name="general-name")
    home = _home(release.planner, marked=marked)
    control = _twin(home, tmp_path / "control")

    result = _sync(monkeypatch, home)
    _use_role_skills(monkeypatch, tmp_path / "no-role-skills")
    expected = _sync(monkeypatch, control)

    assert result == {**expected, "refused": expected["refused"] + [{"name": NAME, "reason": "overlap"}]}
    assert _tree(home) == _tree(control)
    assert not os.path.lexists(home / "skills" / ROLE_RECORD)


@pytest.mark.parametrize("catalogue", [True, False], ids=["general catalogue", "no general catalogue"])
def test_own_record_6_an_exact_copy_in_a_marked_planner_home_is_adopted_in_its_own_record_by_the_release(
        tmp_path, monkeypatch, checkout, catalogue):
    """Review round 3, blocking finding 2, through the release's own call: the line goes to
    ``.role_skills_manifest``, and ``.bundled_manifest`` is the control's, with the role catalogue hidden."""
    if catalogue:
        _write_skill(_general(monkeypatch, tmp_path / "release" / "skills"), "software-development/spike", "v1")
    else:
        monkeypatch.setenv("HERMES_BUNDLED_SKILLS", str(tmp_path / "not-there"))
    src = shutil.copytree(REPO / "role-skills" / PLANNER / PLAN, checkout / PLANNER / PLAN)
    home = _home(tmp_path / "root" / "profiles" / PLANNER)
    shutil.copytree(src, home / "skills" / PLAN)
    md = home / "skills" / PLAN / "SKILL.md"
    before = hashlib.sha256(md.read_bytes()).hexdigest(), _stat(md)

    _step(home, tmp_path / "control", checkout, [PLAN], {})  # adopted: in no list

    assert (home / "skills" / ROLE_RECORD).read_text(encoding="utf-8") == f"{NAME}:{_dir_hash(src)}\n"
    assert (home / "skills" / ".bundled_manifest").read_bytes() == b""
    assert (hashlib.sha256(md.read_bytes()).hexdigest(), _stat(md)) == before


def test_own_record_10_nested_a_role_skill_folder_that_holds_a_second_skill_gets_no_write_and_no_line(
        release, monkeypatch, tmp_path):
    src = _write_skill(release.roles / PLANNER, PLAN, "planner")
    _write_skill(src, "inner", "a second skill inside")
    home = _home(release.planner)
    control = _twin(home, tmp_path / "control")

    result = _sync(monkeypatch, home)
    _use_role_skills(monkeypatch, tmp_path / "no-role-skills")
    expected = _sync(monkeypatch, control)

    assert result == {**expected, "refused": expected["refused"] + [
        {"name": NAME, "reason": "nested"}, {"name": "inner", "reason": "nested"}]}
    assert _tree(home) == _tree(control)
    assert not os.path.lexists(home / "skills" / ROLE_RECORD)
