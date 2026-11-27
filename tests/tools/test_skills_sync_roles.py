"""Each role skill reaches its own role at a release, and an exact copy is adopted.

Each test runs the real sync against a temporary release repository (``skills/`` beside
``role-skills/``, set through ``HERMES_BUNDLED_SKILLS``) and temporary homes, set through
``HERMES_HOME``; a role is a home under a folder named ``profiles``. A "release" is a change to
the repository followed by the sync. The test names carry the point of the card's facts they hold.
"""

import hashlib
import json
import os
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import tools.skills_sync as skills_sync
from hermes_cli.profiles import seed_profile_skills
from tools.skill_usage import _read_skill_name
from tools.skills_sync import NO_BUNDLED_SKILLS_MARKER, _dir_hash, _discover_bundled_skills, sync_skills

REPO = Path(__file__).resolve().parents[2]
PLANNER, BUILDER = "raphael-planner", "raphael-claude-worker"
PLAN, BUILD = "team-operations/handing-build-work", "team-operations/running-build-work"
NAME = "handing-build-work"
LISTS = ("copied", "updated", "kept", "refused", "suppressed", "shadowed_by_external")

needs_symlinks = pytest.mark.skipif(sys.platform == "win32", reason="symlinks need extra rights on Windows")


@pytest.fixture
def release(tmp_path, monkeypatch):
    """A release repository with ``skills/`` and ``role-skills/``, and a root whose roles are under ``profiles/``."""
    repo = tmp_path / "repo"
    for path in (repo / "skills", repo / "role-skills"):
        path.mkdir(parents=True)
    monkeypatch.setenv("HERMES_BUNDLED_SKILLS", str(repo / "skills"))
    root = tmp_path / "root"
    return SimpleNamespace(skills=repo / "skills", roles=repo / "role-skills", root=root,
                           planner=root / "profiles" / PLANNER, builder=root / "profiles" / BUILDER)


def _sync(monkeypatch, home: Path) -> dict:
    """One sync of ``home``, as the release runs it for each role."""
    monkeypatch.setenv("HERMES_HOME", str(home))
    return sync_skills(quiet=True)


def _write_skill(root: Path, rel: str, body: str) -> Path:
    folder = root / rel
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "SKILL.md").write_text(
        f"---\nname: {folder.name}\ndescription: test skill\n---\n{body}\n", encoding="utf-8")
    return folder


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
    src = _write_skill(release.roles / PLANNER, PLAN, "planner v1")
    _write_skill(release.roles / BUILDER, BUILD, "builder v1")
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
