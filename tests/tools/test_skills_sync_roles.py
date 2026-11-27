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
    """``_tree(home)`` but the record file, where a role line is the one difference from a twin."""
    tree = _tree(home)
    tree.pop("skills/.bundled_manifest", None)
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


# -- Point 10, the table ---------------------------------------------------------------------------

def test_10a_a_new_role_skill_is_copied_with_its_record_line(release, monkeypatch):
    src = _write_skill(release.roles / PLANNER, PLAN, "planner v1")

    result = _sync(monkeypatch, release.planner)

    assert result["copied"] == [NAME]
    assert _files(release.planner / "skills" / PLAN) == _files(src)
    assert _manifest(release.planner) == {NAME: _dir_hash(src)}


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
    assert _manifest(release.planner) == {NAME: _dir_hash(src)}
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
    assert NAME not in _manifest(release.planner)


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
    assert NAME not in _manifest(release.planner)


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
    assert _manifest(release.planner) == {NAME: _dir_hash(src)}


def test_10e_a_recorded_role_copy_that_a_person_changed_is_kept_with_no_update(release, monkeypatch):
    _write_skill(release.roles / PLANNER, PLAN, "planner v1")
    _sync(monkeypatch, release.planner)
    _edit(release.planner / "skills" / PLAN)
    before, line = _files(release.planner / "skills" / PLAN), _manifest(release.planner)
    _write_skill(release.roles / PLANNER, PLAN, "planner v2")

    result = _sync(monkeypatch, release.planner)

    assert result["kept"] == [{"name": NAME, "reason": "changed"}] and result["updated"] == []
    assert _files(release.planner / "skills" / PLAN) == before
    assert _manifest(release.planner) == line


# -- Points 7 to 9, 11 and 12 ----------------------------------------------------------------------

def test_7_8_a_role_skill_reaches_only_its_own_role(release, monkeypatch):
    _write_skill(release.roles / PLANNER, PLAN, "planner")
    _write_skill(release.roles / BUILDER, BUILD, "builder")
    other = release.root / "profiles" / "raphael-verifier"

    assert _sync(monkeypatch, release.planner)["copied"] == [NAME]
    assert _sync(monkeypatch, release.builder)["copied"] == ["running-build-work"]
    assert _sync(monkeypatch, other)["copied"] == []
    assert sorted(_files(release.planner / "skills")) == [".bundled_manifest", f"{PLAN}/SKILL.md"]
    assert sorted(_files(release.builder / "skills")) == [".bundled_manifest", f"{BUILD}/SKILL.md"]
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


def test_12_a_role_skill_record_line_survives_the_cleanup_in_a_role_with_no_marker(release, monkeypatch):
    src = _write_skill(release.roles / PLANNER, PLAN, "planner")
    _write_skill(release.skills, "software-development/spike", "general")
    _sync(monkeypatch, release.planner)
    shutil.rmtree(release.skills / "software-development")  # the release removes the general skill

    result = _sync(monkeypatch, release.planner)

    assert result["cleaned"] == ["spike"]  # the cleanup ran
    assert _manifest(release.planner) == {NAME: _dir_hash(src)}
    assert _sync(monkeypatch, release.planner)["cleaned"] == []


def test_8_the_role_copy_lands_at_the_same_inner_path(release, monkeypatch):
    rel = "team-operations/build-loop/handing-build-work"
    src = _write_skill(release.roles / PLANNER, rel, "planner")
    (src / "references").mkdir()
    (src / "references" / "checklist.md").write_text("one\n", encoding="utf-8")

    _sync(monkeypatch, release.planner)

    assert sorted(_files(release.planner / "skills")) == [
        ".bundled_manifest", f"{rel}/SKILL.md", f"{rel}/references/checklist.md"]
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

        seen.append(({key: result[key] for key in LISTS}, result["skipped"], _manifest(home), _files(outside)))
    assert seen[0] == seen[1]
    assert seen[1][2] == {}  # no line in any of these cases: the next sync tries again
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
    assert _manifest(release.planner) == {NAME: _dir_hash(src)}
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
        assert _manifest(home) == {src.name: _dir_hash(src), **({} if marked else {"spike": _dir_hash(spike)})}
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
    assert _manifest(planner) == {NAME: _dir_hash(src)}


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
    assert _manifest(planner) == ({} if owned else {NAME: _dir_hash(src)})
    assert (_sha(md), _stat(md)) == (sha, stat)
    record = {} if owned else {"skills/.bundled_manifest": f"{NAME}:{_dir_hash(src)}\n".encode("utf-8")}
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
    assert with_role == {**without, "kept": [{"name": NAME, "reason": "hub_lock"}],
                         "skipped": without["skipped"] + 1}
    assert _tree(planner) == _tree(twin)
    assert NAME not in _manifest(planner) and _sha(md) == sha


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

        if marked:
            assert with_role == {**without, "skipped": without["skipped"] + 1}  # adopted: in no list
        elif release_no == 0:
            assert with_role == {**without, "refused": [{"name": NAME, "reason": "overlap"}]}
        else:
            assert with_role == {**without, "kept": [{"name": NAME, "reason": "changed"}],
                                 "skipped": without["skipped"] + 1}
        assert without["refused"] == [] and _without_record(planner) == _without_record(twin)
        assert _manifest(planner) == {**_manifest(twin), **({NAME: _dir_hash(src)} if marked else {})}
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

    assert with_role == {**without, "refused": without["refused"] + [{"name": name, "reason": "overlap"}]}
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
    assert first == {**first_twin, "skipped": first_twin["skipped"] + 1}  # adopted: in no list
    assert _manifest(planner) == {NAME: _dir_hash(src)}
    _write_skill(general, f"software-development/{NAME}", "the general one")

    with_role, without = _seed(planner), _seed(twin)

    assert with_role == {**without, "refused": [{"name": NAME, "reason": "name"}]}
    assert without["copied"] == ([] if marked else [NAME]) and without["relocated"] == []
    assert _files(planner / "skills" / PLAN) == copy
    assert _without_record(planner) == _without_record(twin)
    assert _manifest(planner) == {**_manifest(twin), **({NAME: _dir_hash(src)} if marked else {})}


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
        assert with_role == {**without, "skipped": without["skipped"] + 1}  # adopted: in no list
        assert _manifest(planner) == {**_manifest(twin), NAME: _dir_hash(src)}
        assert ({path: _stat(path) for path in [copy, *sorted(copy.rglob("*"))]}, _files(copy)) == before
        assert _without_record(planner) == _without_record(twin)
    else:
        assert with_role == {**without, "refused": [{"name": NAME, "reason": "overlap"}]}
        assert _tree(planner) == _tree(twin) and NAME not in _manifest(planner)
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
