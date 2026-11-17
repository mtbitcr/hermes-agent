"""Released skill changes reach every role; changed and deleted copies are kept.

Each test runs the real sync against a temporary role home, built-in tree and optional
tree, set through ``HERMES_HOME``, ``HERMES_BUNDLED_SKILLS`` and ``HERMES_OPTIONAL_SKILLS``.
A "release" is a change to those trees followed by the sync.
"""

import json
import os
import shutil
import stat
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from agent.skill_utils import ESSENTIAL_SKILLS
from tools.skills_guard import content_hash
from tools.skills_sync import NO_BUNDLED_SKILLS_MARKER, _dir_hash, sync_skills

ESSENTIAL = sorted(ESSENTIAL_SKILLS)[0]

# Root reads and writes whatever the mode, so a chmod-based denial only shows for other users.
# Each test with this mark has a twin that makes the same call fail without real permissions.
needs_real_permissions = pytest.mark.skipif(
    sys.platform == "win32" or os.geteuid() == 0, reason="chmod-based access denial needs a non-root user")


@pytest.fixture
def role(tmp_path, monkeypatch):
    """One role's temporary home plus the release's built-in and optional trees."""
    home, bundled, optional = tmp_path / "home", tmp_path / "bundled", tmp_path / "optional"
    for path in (home / "skills", bundled, optional):
        path.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_BUNDLED_SKILLS", str(bundled))
    monkeypatch.setenv("HERMES_OPTIONAL_SKILLS", str(optional))
    return SimpleNamespace(home=home, skills=home / "skills", bundled=bundled, optional=optional)


def _write_skill(root: Path, rel: str, body: str) -> Path:
    folder = root / rel
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "SKILL.md").write_text(
        f"---\nname: {folder.name}\ndescription: test skill\n---\n{body}\n", encoding="utf-8")
    return folder


def _edit(folder: Path) -> None:
    md = folder / "SKILL.md"
    md.write_text(md.read_text(encoding="utf-8") + "owner change\n", encoding="utf-8")


def _files(folder: Path) -> dict:
    """Every file under ``folder``: relative path -> bytes (byte-for-byte comparisons)."""
    return {p.relative_to(folder).as_posix(): p.read_bytes() for p in sorted(folder.rglob("*")) if p.is_file()}


def _manifest(role) -> dict:
    path = role.skills / ".bundled_manifest"
    lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
    return dict(line.split(":", 1) for line in lines if line)


def _lock(role) -> dict:
    path = role.skills / ".hub" / "lock.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _lock_bytes(role):
    path = role.skills / ".hub" / "lock.json"
    return path.read_bytes() if path.exists() else None


def _mark(role) -> None:
    (role.home / NO_BUNDLED_SKILLS_MARKER).write_text("opted out\n", encoding="utf-8")


def _adopt_official_copy(role, rel: str, body: str) -> Path:
    """An official optional skill whose copy matches its hub lock entry (written here as the hub
    writes one: the sync adopts nothing it has no record of)."""
    src = _write_skill(role.optional, rel, body)
    copy = role.skills / rel
    shutil.copytree(src, copy)
    data = _lock(role) or {"version": 1, "installed": {}}
    data["installed"][src.name] = {
        "source": "official", "identifier": f"official/{rel}", "trust_level": "builtin",
        "content_hash": content_hash(copy), "install_path": rel, "files": ["SKILL.md"]}
    (role.skills / ".hub").mkdir(exist_ok=True)
    (role.skills / ".hub" / "lock.json").write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    return src


def _three_kinds_then_release(role, *, marked: bool) -> tuple:
    """A built-in copy, a built-in that the release moves to the optional catalog, and an
    official optional copy; then the release moves every official version."""
    _write_skill(role.bundled, "software-development/spike", "built-in v1")
    _write_skill(role.bundled, "media/ascii-art", "built-in v1")
    sync_skills(quiet=True)
    _adopt_official_copy(role, "communication/yuanbao", "official v1")
    if marked:
        _mark(role)
    rels = ("software-development/spike", "media/ascii-art", "communication/yuanbao")

    def release() -> None:
        _write_skill(role.bundled, "software-development/spike", "built-in v2")
        shutil.rmtree(role.bundled / "media")
        _write_skill(role.optional, "creative/ascii-art", "official optional v2")
        _write_skill(role.optional, "communication/yuanbao", "official v2")
    return rels, release


def _release_v3(role) -> None:
    """The release after ``_three_kinds_then_release``'s: every official version moves again."""
    _write_skill(role.bundled, "software-development/spike", "built-in v3")
    _write_skill(role.optional, "creative/ascii-art", "official optional v3")
    _write_skill(role.optional, "communication/yuanbao", "official v3")


def _seed(profile: Path):
    """The real profile seeding: a child process whose ``HERMES_HOME`` is ``profile``. It returns
    the legacy keys only; a test that reads ``kept`` or ``refused`` runs ``sync_skills``."""
    from hermes_cli.profiles import seed_profile_skills
    return seed_profile_skills(profile, quiet=True)


def _add_owner_folder(copy: Path) -> Path:
    """The owner adds a folder of their own, with a file in it, to ``copy``."""
    folder = copy / "zz-private"
    folder.mkdir(parents=True)
    (folder / "owner.txt").write_text("owner data\n", encoding="utf-8")
    return folder


def _readable_again(*folders: Path) -> None:
    """Undo mode 000 on the owner's folders, wherever a run may have moved them (``.bak``)."""
    for folder in folders:
        for path in (folder, folder.parent.with_suffix(".bak") / folder.name):
            if os.path.lexists(path):
                path.chmod(0o755)


def _deny_listing(patched) -> None:
    """Twin of mode 000 on the owner's folders: listing one fails at ``os.scandir``, the call that
    pathlib's ``rglob``, ``os.walk`` and ``shutil`` go through."""
    real_scandir = os.scandir

    def scandir(path=".", *args, **kwargs):
        if isinstance(path, (str, os.PathLike)) and os.path.basename(os.fspath(path)) == "zz-private":
            raise PermissionError(13, "Permission denied", os.fspath(path))
        return real_scandir(path, *args, **kwargs)
    patched.setattr(os, "scandir", scandir)


def _deny_manifest_write(patched) -> None:
    """The atomic manifest write cannot create its temporary file in skills/."""
    real_mkstemp = tempfile.mkstemp

    def mkstemp(*args, **kwargs):
        if str(kwargs.get("prefix", "")).startswith(".bundled_manifest"):
            raise PermissionError(13, "Permission denied", kwargs.get("dir"))
        return real_mkstemp(*args, **kwargs)
    patched.setattr(tempfile, "mkstemp", mkstemp)


def _add_owner_link(copy: Path, outside: Path) -> tuple:
    """The owner links a folder of their own, kept outside the skills tree, into ``copy``."""
    target = outside / f"owner-notes-{copy.name}"
    target.mkdir(parents=True)
    (target / "notes.txt").write_text("owner-created notes\n", encoding="utf-8")
    link = copy / "private"
    os.symlink(target, link, target_is_directory=True)
    return link, target


def _assert_owner_link_kept(copy: Path, before: dict, link: Path, target: Path) -> None:
    """The copy, the owner's link itself and the bytes it points at are all as they were."""
    assert _files(copy) == before
    assert os.path.islink(link) and os.readlink(link) == str(target)
    assert _files(target) == {"notes.txt": b"owner-created notes\n"}
    assert not copy.with_suffix(".bak").exists()


# ---------------------------------------------------------------------------
# Built-in skills in a marked role
# ---------------------------------------------------------------------------

def test_marked_role_updates_unchanged_copy_seeds_no_new_and_still_seeds_essential(role):
    """Production group: the marker file sits in the home of four roles, which hold 60
    unchanged old built-in copies. At a release each unchanged copy gets the released version,
    no new built-in is seeded, and the essential skill is still seeded."""
    _write_skill(role.bundled, "software-development/spike", "spike v1")
    _write_skill(role.bundled, "software-development/test-driven-development", "tdd v1")
    sync_skills(quiet=True)
    _mark(role)
    _write_skill(role.bundled, "software-development/spike", "spike v2")
    _write_skill(role.bundled, "software-development/test-driven-development", "tdd v2")
    _write_skill(role.bundled, "productivity/brand-new", "new in this release")
    _write_skill(role.bundled, f"core/{ESSENTIAL}", "essential")

    result = sync_skills(quiet=True)

    for rel in ("software-development/spike", "software-development/test-driven-development"):
        assert _files(role.skills / rel) == _files(role.bundled / rel)
    assert sorted(result["updated"]) == ["spike", "test-driven-development"]
    assert not (role.skills / "productivity" / "brand-new").exists()
    assert "brand-new" not in _manifest(role)
    assert ESSENTIAL in result["copied"]
    assert (role.skills / "core" / ESSENTIAL / "SKILL.md").is_file()
    assert result["skipped_opt_out"] is True


def test_marked_role_changed_copy_kept_byte_for_byte_and_listed_user_modified(role):
    """Production group: no role holds a changed copy today. A copy changed after Hermes
    wrote it is still kept byte for byte and reported as user-modified."""
    _write_skill(role.bundled, "software-development/spike", "spike v1")
    sync_skills(quiet=True)
    _mark(role)
    copy = role.skills / "software-development" / "spike"
    _edit(copy)
    before = _files(copy)
    _write_skill(role.bundled, "software-development/spike", "spike v2")

    result = sync_skills(quiet=True)

    assert _files(copy) == before
    assert "spike" in result["user_modified"]
    assert "spike" not in result["updated"]


def test_marked_role_line_without_folder_is_not_recreated(role):
    _write_skill(role.bundled, "planning/plan-review", "v1")
    sync_skills(quiet=True)
    _mark(role)
    shutil.rmtree(role.skills / "planning" / "plan-review")
    _write_skill(role.bundled, "planning/plan-review", "v2")

    for _release in range(2):
        result = sync_skills(quiet=True)
        assert not (role.skills / "planning" / "plan-review").exists()
        assert "plan-review" not in result["copied"]
        assert "plan-review" in _manifest(role)
        # It has a line: deleted by the owner, not skipped by the marker.
        assert result["not_seeded_by_marker"] == 0

def test_marked_role_backup_beside_the_copy_keeps_copy_backup_and_line(role):
    """Owner rule 2 point 3: a ``.bak`` folder beside an unchanged copy whose built-in changed stops
    the update. The copy, the ``.bak`` folder and the manifest line stay byte for byte, and the
    result names the skill under kept with the reason ``backup present``."""
    _write_skill(role.bundled, "software-development/spike", "built-in v1")
    sync_skills(quiet=True)
    _mark(role)
    copy = role.skills / "software-development" / "spike"
    backup = copy.with_suffix(".bak")
    shutil.copytree(copy, backup)
    (backup / "notes.txt").write_text("owner notes\n", encoding="utf-8")
    before, backup_before = _files(copy), _files(backup)
    manifest_before = (role.skills / ".bundled_manifest").read_bytes()
    _write_skill(role.bundled, "software-development/spike", "built-in v2")

    for _sync in range(2):
        result = sync_skills(quiet=True)

        assert _files(copy) == before
        assert _files(backup) == backup_before
        assert (role.skills / ".bundled_manifest").read_bytes() == manifest_before
        assert "spike" not in result["updated"]
        assert result["kept"] == [{"name": "spike", "reason": "backup present"}]
        assert result["refused"] == []


@pytest.mark.parametrize("case", ["deleted", "backup-at-sibling", "other-category"])
def test_marked_role_name_without_copy_at_its_path_writes_nothing(role, case):
    """Owner rule 2 point 4: a name with a manifest line and no copy at its expected path counts as
    skipped. Nothing is searched, moved back or written: the whole skills tree stays as it was."""
    _write_skill(role.bundled, "software-development/spike", "built-in v1")
    sync_skills(quiet=True)
    _mark(role)
    copy = role.skills / "software-development" / "spike"
    if case == "backup-at-sibling":
        copy.rename(copy.with_suffix(".bak"))
    elif case == "other-category":  # an unchanged copy of the skill in another category folder
        (role.skills / "productivity").mkdir()
        copy.rename(role.skills / "productivity" / "spike")
    else:
        shutil.rmtree(copy)
    _write_skill(role.bundled, "software-development/spike", "built-in v2")
    line = _manifest(role)["spike"]

    def snapshot() -> dict:
        return {p.relative_to(role.skills).as_posix(): p.read_bytes() if p.is_file() else None
                for p in sorted(role.skills.rglob("*"))}
    before = snapshot()

    result = sync_skills(quiet=True)

    assert snapshot() == before
    assert _manifest(role)["spike"] == line
    assert result["skipped"] == 1
    assert result["copied"] == [] and result["updated"] == [] and result["relocated"] == []
    assert result["kept"] == [] and result["refused"] == []


def test_marked_role_keeps_lines_for_names_no_longer_built_in_and_unmarked_role_cleans_them(role):
    """Owner rule 2 points 2 and 6: a marked role cleans no manifest line; an unmarked role cleans
    each line whose name is no longer a built-in, as on the base, and touches no copy."""
    _write_skill(role.bundled, "planning/plan", "v1")
    _write_skill(role.bundled, "planning/old-helper", "v1")
    _write_skill(role.bundled, "software-development/spike", "v1")
    sync_skills(quiet=True)
    _mark(role)
    shutil.rmtree(role.skills / "planning" / "old-helper")
    shutil.rmtree(role.bundled / "planning")
    plan = _files(role.skills / "planning" / "plan")
    manifest_before = (role.skills / ".bundled_manifest").read_bytes()

    marked = sync_skills(quiet=True)

    assert marked["cleaned"] == []
    assert (role.skills / ".bundled_manifest").read_bytes() == manifest_before

    (role.home / NO_BUNDLED_SKILLS_MARKER).unlink()
    unmarked = sync_skills(quiet=True)

    assert sorted(unmarked["cleaned"]) == ["old-helper", "plan"]
    assert set(_manifest(role)) == {"spike"}
    assert _files(role.skills / "planning" / "plan") == plan


def test_marked_role_reports_not_seeded_by_marker_for_built_ins_without_a_line(role):
    _write_skill(role.bundled, "software-development/spike", "v1")
    _write_skill(role.bundled, "software-development/systematic-debugging", "v1")
    sync_skills(quiet=True)
    _mark(role)
    for rel in ("research/arxiv", "media/gif-search", f"core/{ESSENTIAL}"):
        _write_skill(role.bundled, rel, "new in this release")

    marked = sync_skills(quiet=True)

    assert marked["not_seeded_by_marker"] == 2
    assert not (role.skills / "research" / "arxiv").exists()
    assert not (role.skills / "media" / "gif-search").exists()
    assert ESSENTIAL in marked["copied"]

    (role.home / NO_BUNDLED_SKILLS_MARKER).unlink()
    unmarked = sync_skills(quiet=True)

    assert unmarked["not_seeded_by_marker"] == 0
    assert sorted(unmarked["copied"]) == ["arxiv", "gif-search"]


# ---------------------------------------------------------------------------
# A built-in that moved to the optional catalog, or left both trees
# ---------------------------------------------------------------------------

def test_marked_role_total_bundled_counts_only_the_essential_skills_in_the_catalog(role):
    """``total_bundled`` keeps its meaning, the skills the role may be seeded with: with the
    marker, the essential skills in the catalog, even when the role tracks other built-ins."""
    _write_skill(role.bundled, f"autonomous-ai-agents/{ESSENTIAL}", "essential v1")
    _write_skill(role.bundled, "software-development/spike", "v1")
    _write_skill(role.bundled, "software-development/sketch", "v1")
    unmarked = sync_skills(quiet=True)
    _mark(role)

    marked = sync_skills(quiet=True)

    assert set(_manifest(role)) == {ESSENTIAL, "spike", "sketch"}
    assert unmarked["total_bundled"] == 3
    assert marked["total_bundled"] == 1


def test_marked_role_built_in_moved_to_the_optional_catalog_keeps_copy_and_line(role):
    """Owner rule 2 point 4: the release moves a built-in to the optional catalog. In a marked role
    its unchanged copy and its manifest line stay byte for byte and no hub lock entry is written."""
    _write_skill(role.bundled, "media/ascii-art", "built-in v1")
    sync_skills(quiet=True)
    _mark(role)
    copy = role.skills / "media" / "ascii-art"
    before, line = _files(copy), _manifest(role)["ascii-art"]
    shutil.rmtree(role.bundled / "media")
    src = _write_skill(role.optional, "creative/ascii-art", "official optional v2")
    (src / "references").mkdir()
    (src / "references" / "fonts.md").write_text("fonts\n", encoding="utf-8")

    results = [sync_skills(quiet=True), _seed(role.home)]

    for result in results:
        assert result is not None
        assert result["cleaned"] == [] and result["updated"] == []
    assert _files(copy) == before
    assert _manifest(role)["ascii-art"] == line
    assert _lock_bytes(role) is None
    assert not os.path.lexists(role.skills / "creative" / "ascii-art")


@pytest.mark.parametrize("marked", [True, False], ids=["marked", "unmarked"])
def test_removed_upstream_copy_stays_and_only_an_unmarked_role_cleans_its_line(role, marked):
    """A built-in the release removed from both trees (like the planner role's `plan`): the copy
    stays; a marked role keeps its line, an unmarked role cleans it as on the base."""
    _write_skill(role.bundled, "planning/plan", "v1")
    sync_skills(quiet=True)
    if marked:
        _mark(role)
    copy = role.skills / "planning" / "plan"
    before, line = _files(copy), _manifest(role)["plan"]
    shutil.rmtree(role.bundled / "planning")

    result = sync_skills(quiet=True)

    assert _files(copy) == before
    if marked:
        assert result["cleaned"] == []
        assert _manifest(role)["plan"] == line
    else:
        assert result["cleaned"] == ["plan"]
        assert "plan" not in _manifest(role)


# ---------------------------------------------------------------------------
# Official optional skills: the sync leaves them as they are
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("marked", [True, False], ids=["marked", "unmarked"])
def test_optional_official_copy_matching_its_lock_stays_untouched_when_the_source_is_newer(role, marked):
    """Production group: the one optional copy that matches its hub lock while the official
    source changed. Seeding and the sync leave the copy and its lock entry as they are."""
    _adopt_official_copy(role, "communication/yuanbao", "official v1")
    if marked:
        _mark(role)
    copy = role.skills / "communication" / "yuanbao"
    before, lock_before = _files(copy), _lock_bytes(role)
    src = _write_skill(role.optional, "communication/yuanbao", "official v2")
    (src / "references").mkdir()
    (src / "references" / "api.md").write_text("api\n", encoding="utf-8")

    results = [_seed(role.home), sync_skills(quiet=True)]

    for result in results:
        assert result is not None
        assert result["optional_provenance_backfilled"] == []
    assert _files(copy) == before
    assert _lock_bytes(role) == lock_before


@pytest.mark.parametrize("marked", [True, False], ids=["marked", "unmarked"])
def test_optional_copy_without_record_untouched_and_gets_no_lock_entry(role, marked):
    """Production group: nine optional copies have no record (no hub-lock entry and no
    manifest line). Seeding and the sync never touch them and write no lock entry for them."""
    if marked:
        _mark(role)
    _write_skill(role.optional, "research/blogwatcher", "official v2")
    stale = _write_skill(role.skills, "research/blogwatcher", "an older official text")
    before = _files(stale)

    results = [_seed(role.home), sync_skills(quiet=True)]

    for result in results:
        assert result is not None
        assert result["optional_provenance_backfilled"] == []
    assert _files(stale) == before
    assert _lock_bytes(role) is None


# ---------------------------------------------------------------------------
# The three data rules
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("marked", [True, False], ids=["marked", "unmarked"])
def test_data_rule_changed_copy_is_never_overwritten(role, marked):
    """Data rule: no copy changed after Hermes wrote it is overwritten, even when a newer
    version exists: built-in, moved to the optional catalog, or official optional."""
    rels, release = _three_kinds_then_release(role, marked=marked)
    for rel in rels:
        _edit(role.skills / rel)
    before = {rel: _files(role.skills / rel) for rel in rels}
    release()

    result = sync_skills(quiet=True)

    for rel, files in before.items():
        assert _files(role.skills / rel) == files, rel
    assert "spike" in result["user_modified"]
    assert result["updated"] == []


@pytest.mark.parametrize("marked", [True, False], ids=["marked", "unmarked"])
def test_data_rule_deleted_copy_never_comes_back(role, marked):
    """Data rule: no deleted copy comes back: built-in, moved to the optional catalog, or
    official optional, at this release and the next one."""
    rels, release = _three_kinds_then_release(role, marked=marked)
    for rel in rels:
        shutil.rmtree(role.skills / rel)
    release()

    for _release in range(2):
        result = sync_skills(quiet=True)
        for rel in (*rels, "creative/ascii-art"):
            assert not (role.skills / rel).exists(), rel
        assert result["copied"] == [] and result["updated"] == []


def test_data_rule_failed_replacement_restores_old_copy(role, monkeypatch):
    """Data rule: a failed replacement restores the old copy of a built-in update, and the
    other copies stay as they were. The next release retries."""
    rels, release = _three_kinds_then_release(role, marked=True)
    before = {rel: _files(role.skills / rel) for rel in rels}
    lock_before = _lock_bytes(role)
    release()

    def copy_then_fail(src, dst, *args, **kwargs):
        Path(dst).mkdir(parents=True, exist_ok=True)
        (Path(dst) / "SKILL.md").write_text("half-written", encoding="utf-8")
        raise OSError("disk full")

    with monkeypatch.context() as patched:
        patched.setattr(shutil, "copytree", copy_then_fail)
        result = sync_skills(quiet=True)

    for rel, files in before.items():
        assert _files(role.skills / rel) == files, rel
        assert not (role.skills / rel).with_suffix(".bak").exists(), rel
    assert result["updated"] == []
    assert result["refused"] == [{"name": "spike", "reason": "copy write failed"}]
    assert _lock_bytes(role) == lock_before

    result = sync_skills(quiet=True)

    assert result["updated"] == ["spike"]


# ---------------------------------------------------------------------------
# A copy Hermes cannot read whole is never replaced
# ---------------------------------------------------------------------------

def _assert_every_unreadable_copy_kept(role, results, before, lock_before) -> None:
    for result in results:
        assert result is not None  # the seeding child finished
        assert "spike" in result["user_modified"]
        assert result["updated"] == []
        if "kept" in result:  # the in-process sync; the seeding child prints the legacy keys only
            assert result["kept"] == [{"name": "spike", "reason": "unreadable"}]
    assert "ascii-art" in _manifest(role)
    for copy, files in before.items():
        assert _files(copy) == files, copy  # the owner's file is still in the active copy
        assert not copy.with_suffix(".bak").exists(), copy
    assert _lock_bytes(role) == lock_before


@needs_real_permissions
def test_folder_the_owner_made_unreadable_keeps_every_copy_through_seeding_and_next_release(role):
    """Real mode 000 on a folder the owner added to a built-in copy, a moved built-in's copy and
    an official optional copy. Seeding at the release, seeding again and seeding at the release
    after it replace none of them, report the built-in one and still return a result."""
    rels, release = _three_kinds_then_release(role, marked=True)
    copies = [role.skills / rel for rel in rels]
    hidden = [_add_owner_folder(copy) for copy in copies]
    before, lock_before = {copy: _files(copy) for copy in copies}, _lock_bytes(role)
    try:
        for folder in hidden:
            folder.chmod(0)
        release()
        results = [_seed(role.home) for _ in range(2)]
        _release_v3(role)
        results.append(_seed(role.home))
    finally:
        _readable_again(*hidden)

    _assert_every_unreadable_copy_kept(role, results, before, lock_before)


def test_folder_that_cannot_be_listed_keeps_every_copy_through_next_release(role, monkeypatch):
    """Twin of the mode-000 test that runs as any user: listing the owner's folder fails."""
    rels, release = _three_kinds_then_release(role, marked=True)
    copies = [role.skills / rel for rel in rels]
    for copy in copies:
        _add_owner_folder(copy)
    before, lock_before = {copy: _files(copy) for copy in copies}, _lock_bytes(role)
    release()

    with monkeypatch.context() as patched:
        _deny_listing(patched)
        results = [sync_skills(quiet=True) for _ in range(2)]
        _release_v3(role)
        results.append(sync_skills(quiet=True))

    _assert_every_unreadable_copy_kept(role, results, before, lock_before)


def _stale_backup_with_owner_folder(role) -> tuple:
    """Built-in spike v1, plus a backup an earlier interrupted run left beside it that holds a
    folder of the owner's; then the release brings spike v2."""
    _write_skill(role.bundled, "software-development/spike", "built-in v1")
    sync_skills(quiet=True)
    copy = role.skills / "software-development" / "spike"
    hidden = _add_owner_folder(copy.with_suffix(".bak"))
    _write_skill(role.bundled, "software-development/spike", "built-in v2")
    return copy, hidden


@needs_real_permissions
def test_stale_backup_with_unreadable_folder_is_kept_and_seeding_finishes(role):
    """Real mode 000 inside a stale backup: clearing it fails, so the update waits; the copy and
    the owner's folder stay and seeding still returns a result (no crash in the cleanup)."""
    copy, hidden = _stale_backup_with_owner_folder(role)
    before = _files(copy)
    try:
        hidden.chmod(0)
        result = _seed(role.home)
    finally:
        hidden.chmod(0o755)

    assert result is not None
    assert result["updated"] == []
    assert _files(copy) == before
    assert (hidden / "owner.txt").read_text(encoding="utf-8") == "owner data\n"


def test_stale_backup_whose_folder_cannot_be_opened_is_kept_and_sync_finishes(role, monkeypatch):
    """Twin that runs as any user: the cleanup's own ``os.open`` of the owner's folder fails."""
    copy, hidden = _stale_backup_with_owner_folder(role)
    before = _files(copy)
    real_open = os.open

    def deny_open(path, *args, **kwargs):  # rmtree opens each folder it clears, with flags
        if args and os.path.basename(os.fspath(path)) == "zz-private":
            raise PermissionError(13, "Permission denied", os.fspath(path))
        return real_open(path, *args, **kwargs)

    with monkeypatch.context() as patched:
        patched.setattr(os, "open", deny_open)
        result = sync_skills(quiet=True)

    assert result["updated"] == []
    assert _files(copy) == before
    assert (hidden / "owner.txt").read_text(encoding="utf-8") == "owner data\n"


def _shadowed_copy_with_owner_folder(role, tmp_path) -> tuple:
    """Built-in clair-qa synced into the role, a folder the owner adds to the copy, and then an
    external folder, named in the role's config, that provides clair-qa too."""
    _write_skill(role.bundled, "devops/clair-qa", "built-in v1")
    sync_skills(quiet=True)
    copy = role.skills / "devops" / "clair-qa"
    hidden = _add_owner_folder(copy)
    external = tmp_path / "external"
    _write_skill(external, "devops/clair-qa", "external")
    (role.home / "config.yaml").write_text(
        f"skills:\n  external_dirs:\n    - {external}\n", encoding="utf-8")
    return copy, hidden


def _assert_shadowed_copy_kept(results, copy, before) -> None:
    for result in results:
        assert result is not None  # the seeding child finished
        assert result["shadowed_by_external"] == ["clair-qa"]
    assert _files(copy) == before  # never removed as an identical stale shadow


@needs_real_permissions
def test_shadowed_copy_with_unreadable_folder_is_kept_through_seeding_and_next_sync(role, tmp_path):
    """Real mode 000 on the owner's folder in a copy an external folder now provides: the copy
    only looks identical to the built-in, so seeding and the seeding after it keep it whole."""
    copy, hidden = _shadowed_copy_with_owner_folder(role, tmp_path)
    before = _files(copy)
    try:
        hidden.chmod(0)
        results = [_seed(role.home) for _ in range(2)]
    finally:
        _readable_again(hidden)

    _assert_shadowed_copy_kept(results, copy, before)


def test_shadowed_copy_whose_folder_cannot_be_listed_is_kept_through_next_sync(role, tmp_path, monkeypatch):
    """Twin of the mode-000 test that runs as any user: listing the owner's folder fails."""
    copy, _ = _shadowed_copy_with_owner_folder(role, tmp_path)
    before = _files(copy)

    with monkeypatch.context() as patched:
        _deny_listing(patched)
        results = [sync_skills(quiet=True) for _ in range(2)]

    _assert_shadowed_copy_kept(results, copy, before)


# ---------------------------------------------------------------------------
# A failed record write keeps the old copy and the old line
# ---------------------------------------------------------------------------


def test_built_in_update_whose_manifest_write_fails_restores_old_copy_and_line(role, monkeypatch):
    """The new line is written before the old copy goes: when it cannot be written, the old
    copy and the old line stay, and the skill is refused, not updated. The next sync retries."""
    _write_skill(role.bundled, "software-development/spike", "built-in v1")
    sync_skills(quiet=True)
    copy = role.skills / "software-development" / "spike"
    before, manifest_before = _files(copy), (role.skills / ".bundled_manifest").read_bytes()
    _write_skill(role.bundled, "software-development/spike", "built-in v2")

    with monkeypatch.context() as patched:
        _deny_manifest_write(patched)
        result = sync_skills(quiet=True)

    assert _files(copy) == before
    assert not copy.with_suffix(".bak").exists()
    assert (role.skills / ".bundled_manifest").read_bytes() == manifest_before
    assert result["refused"] == [{"name": "spike", "reason": "record write failed"}]
    assert result["updated"] == []

    retried = sync_skills(quiet=True)
    assert retried["updated"] == ["spike"]
    assert _manifest(role)["spike"] == _dir_hash(role.bundled / "software-development" / "spike")


# ---------------------------------------------------------------------------
# One check decides "unchanged": a link, a special file or a missing record keeps the copy
# ---------------------------------------------------------------------------

def _builtin_copy_with_owner_link(role, tmp_path) -> tuple:
    """Hermes installs built-in ascii-art v1; the owner links a folder of their own into it."""
    _write_skill(role.bundled, "media/ascii-art", "built-in v1")
    assert _seed(role.home) is not None
    copy = role.skills / "media" / "ascii-art"
    link, target = _add_owner_link(copy, tmp_path / "outside")
    return copy, link, target, _files(copy), _manifest(role)["ascii-art"]


@pytest.mark.parametrize("marked", [True, False], ids=["marked", "unmarked"])
def test_owner_link_keeps_built_in_copy_at_update(role, tmp_path, marked):
    copy, link, target, before, line = _builtin_copy_with_owner_link(role, tmp_path)
    if marked:
        _mark(role)
    _write_skill(role.bundled, "media/ascii-art", "built-in v2")

    result = sync_skills(quiet=True)

    _assert_owner_link_kept(copy, before, link, target)
    assert _manifest(role)["ascii-art"] == line
    assert result["kept"] == [{"name": "ascii-art", "reason": "link"}]
    assert "ascii-art" in result["user_modified"]
    assert "ascii-art" not in result["updated"]


@pytest.mark.parametrize("marked", [True, False], ids=["marked", "unmarked"])
def test_owner_link_keeps_copy_that_an_external_folder_now_provides(role, tmp_path, marked):
    """Configured external source: the local copy only looks like the stale shadow Hermes wrote,
    so it is not deleted. An unmarked role checks it and names it; a marked role only defers
    (owner rule 2 point 2: it removes no copy, so it does not look at it)."""
    _write_skill(role.bundled, "devops/ascii-art", "bundled v1")
    assert _seed(role.home) is not None
    copy = role.skills / "devops" / "ascii-art"
    link, target = _add_owner_link(copy, tmp_path / "outside")
    before, line = _files(copy), _manifest(role)["ascii-art"]
    if marked:
        _mark(role)
    external = tmp_path / "external"
    _write_skill(external, "devops/ascii-art", "external v1")
    (role.home / "config.yaml").write_text(f"skills:\n  external_dirs:\n    - {external}\n", encoding="utf-8")

    result = sync_skills(quiet=True)

    _assert_owner_link_kept(copy, before, link, target)
    assert _manifest(role)["ascii-art"] == line
    assert result["shadowed_by_external"] == ["ascii-art"]
    assert result["kept"] == ([] if marked else [{"name": "ascii-art", "reason": "link"}])


def test_owner_link_keeps_copy_at_rename_recovery(role, tmp_path):
    """The release moves the built-in to another category: the copy is not moved."""
    copy, link, target, before, line = _builtin_copy_with_owner_link(role, tmp_path)
    shutil.rmtree(role.bundled / "media")
    _write_skill(role.bundled, "creative/ascii-art", "built-in v2")

    result = sync_skills(quiet=True)

    _assert_owner_link_kept(copy, before, link, target)
    assert not (role.skills / "creative" / "ascii-art").exists()
    assert _manifest(role)["ascii-art"] == line
    assert result["kept"] == [{"name": "ascii-art", "reason": "link"}]
    assert result["relocated"] == []


def test_owner_link_keeps_same_named_folder_from_adoption_by_a_new_built_in(role, tmp_path):
    """The owner's folder equals a new built-in's files but holds a link: it is not adopted
    (no line), so the release after it does not replace it either."""
    src = _write_skill(role.bundled, "media/ascii-art", "built-in v1")
    copy = role.skills / "media" / "ascii-art"
    shutil.copytree(src, copy)
    link, target = _add_owner_link(copy, tmp_path / "outside")
    before = _files(copy)

    results = [sync_skills(quiet=True)]
    _write_skill(role.bundled, "media/ascii-art", "built-in v2")
    results.append(sync_skills(quiet=True))

    _assert_owner_link_kept(copy, before, link, target)
    assert "ascii-art" not in _manifest(role)
    for result in results:  # no record proves nothing: the reason is `_copy_change(dest, None)`'s
        assert result["kept"] == [{"name": "ascii-art", "reason": "changed"}]
        assert "ascii-art" not in result["copied"] + result["updated"]


def test_file_link_holding_the_recorded_bytes_keeps_the_copy(role, tmp_path):
    _write_skill(role.bundled, "media/ascii-art", "built-in v1")
    sync_skills(quiet=True)
    copy = role.skills / "media" / "ascii-art"
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "SKILL.md").write_bytes((copy / "SKILL.md").read_bytes())
    (copy / "SKILL.md").unlink()
    os.symlink(outside / "SKILL.md", copy / "SKILL.md")
    line = _manifest(role)["ascii-art"]
    _write_skill(role.bundled, "media/ascii-art", "built-in v2")

    result = sync_skills(quiet=True)

    assert os.path.islink(copy / "SKILL.md") and os.readlink(copy / "SKILL.md") == str(outside / "SKILL.md")
    assert _manifest(role)["ascii-art"] == line
    assert result["kept"] == [{"name": "ascii-art", "reason": "link"}]
    assert result["updated"] == []


def test_copy_folder_that_is_a_link_is_kept(role, tmp_path):
    src = _write_skill(role.bundled, "media/ascii-art", "built-in v1")
    sync_skills(quiet=True)
    copy = role.skills / "media" / "ascii-art"
    outside = tmp_path / "outside" / "ascii-art"
    shutil.copytree(src, outside)
    shutil.rmtree(copy)
    os.symlink(outside, copy, target_is_directory=True)
    before, line = _files(outside), _manifest(role)["ascii-art"]
    _write_skill(role.bundled, "media/ascii-art", "built-in v2")

    result = sync_skills(quiet=True)

    assert os.path.islink(copy) and os.readlink(copy) == str(outside)
    assert _files(outside) == before
    assert _manifest(role)["ascii-art"] == line
    assert result["kept"] == [{"name": "ascii-art", "reason": "link"}]
    assert result["updated"] == []


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="needs os.mkfifo")
def test_special_file_in_a_copy_keeps_it_as_changed(role):
    _write_skill(role.bundled, "media/ascii-art", "built-in v1")
    sync_skills(quiet=True)
    copy = role.skills / "media" / "ascii-art"
    os.mkfifo(copy / "pipe")
    before, line = _files(copy), _manifest(role)["ascii-art"]
    _write_skill(role.bundled, "media/ascii-art", "built-in v2")

    result = sync_skills(quiet=True)

    assert stat.S_ISFIFO(os.lstat(copy / "pipe").st_mode)
    assert _files(copy) == before
    assert _manifest(role)["ascii-art"] == line
    assert result["kept"] == [{"name": "ascii-art", "reason": "changed"}]
    assert result["updated"] == []


def test_sync_never_adopts_an_optional_copy_without_a_record(role):
    src = _write_skill(role.optional, "research/blogwatcher", "official v1")
    copy = role.skills / "research" / "blogwatcher"
    shutil.copytree(src, copy)  # byte-identical to the official source, but Hermes wrote no record

    result = _seed(role.home)
    again = sync_skills(quiet=True)

    assert result["optional_provenance_backfilled"] == again["optional_provenance_backfilled"] == []
    assert "blogwatcher" not in _lock(role).get("installed", {})
    assert _files(copy) == _files(src)


def test_v1_manifest_line_is_a_missing_record_and_keeps_the_copy(role):
    _write_skill(role.bundled, "software-development/spike", "built-in v1")
    sync_skills(quiet=True)
    copy = role.skills / "software-development" / "spike"
    before = _files(copy)
    (role.skills / ".bundled_manifest").write_text("spike\n", encoding="utf-8")  # v1: a name, no fingerprint
    _write_skill(role.bundled, "software-development/spike", "built-in v2")

    for _sync in range(2):
        result = sync_skills(quiet=True)
        assert _files(copy) == before
        assert _manifest(role) == {"spike": ""}
        assert result["kept"] == [{"name": "spike", "reason": "changed"}]
        assert result["updated"] == []


def _plain_tree(root: Path) -> Path:
    """A copy whose path order and POSIX-string order differ ("a/b.md", "a-b/x.md"), with an
    empty folder and runtime caches."""
    copy = _write_skill(root, "pkg/tree-skill", "body")
    for rel, data in (("a/b.md", b"one"), ("a-b/x.md", b"two"), ("scripts/tool.py", b"print(1)\n"),
                      ("scripts/tool.pyc", b"\x00cache"), ("scripts/__pycache__/tool.cpython-312.pyc", b"\x01")):
        (copy / rel).parent.mkdir(parents=True, exist_ok=True)
        (copy / rel).write_bytes(data)
    (copy / "empty").mkdir()
    return copy


def test_copy_check_accepts_only_the_recorded_fingerprint_of_a_plain_copy(tmp_path):
    from tools.skills_sync import _copy_change
    copy = _plain_tree(tmp_path)
    recorded = _dir_hash(copy)

    assert recorded != _dir_hash(copy, include_runtime_cache=True)
    assert _copy_change(copy, recorded) is None
    assert _copy_change(copy, _dir_hash(copy, include_runtime_cache=True)) is None  # exact legacy record
    assert _copy_change(copy, content_hash(copy), kind="lock") is None
    assert _copy_change(copy, recorded, kind="lock") == "changed"
    assert _copy_change(copy, content_hash(copy)) == "changed"
    for missing in ("", None):
        assert _copy_change(copy, missing) == "changed"
        assert _copy_change(copy, missing, kind="lock") == "changed"

    (copy / "scripts" / "__pycache__" / "tool.cpython-312.pyc").write_bytes(b"regenerated")
    assert _copy_change(copy, recorded) is None  # runtime caches are left out of the manifest kind
    (copy / "a" / "b.md").write_bytes(b"edited")
    assert _copy_change(copy, recorded) == "changed"


def test_copy_check_refuses_a_link_anywhere_runtime_caches_included(tmp_path):
    from tools.skills_sync import _copy_change
    copy = _plain_tree(tmp_path / "skills")
    recorded = _dir_hash(copy)
    (tmp_path / "elsewhere").mkdir()
    os.symlink(tmp_path / "elsewhere", copy / "scripts" / "__pycache__" / "linked", target_is_directory=True)

    assert _dir_hash(copy) == recorded  # the fingerprint alone cannot see it
    assert _copy_change(copy, recorded) == "link"


@needs_real_permissions
def test_copy_check_reports_an_unreadable_file_or_folder(tmp_path):
    from tools.skills_sync import _copy_change
    copy = _plain_tree(tmp_path)
    recorded = _dir_hash(copy)
    secret = copy / "a" / "b.md"
    secret.chmod(0)
    try:
        assert _copy_change(copy, recorded) == "unreadable"
    finally:
        secret.chmod(0o644)
    hidden = _add_owner_folder(copy)
    hidden.chmod(0)
    try:
        assert _copy_change(copy, _dir_hash(copy)) == "unreadable"  # the hash skips what it cannot list
    finally:
        hidden.chmod(0o755)


def test_copy_check_reports_unreadable_when_listing_or_reading_fails(tmp_path, monkeypatch):
    """Twin of the mode-000 test that runs as any user."""
    from tools.skills_sync import _copy_change
    copy = _plain_tree(tmp_path)
    _add_owner_folder(copy)
    recorded = _dir_hash(copy)
    real_read_bytes = Path.read_bytes

    def read_bytes(path):
        if path.name == "owner.txt":
            raise PermissionError(13, "Permission denied", str(path))
        return real_read_bytes(path)

    with monkeypatch.context() as patched:
        _deny_listing(patched)
        assert _copy_change(copy, recorded) == "unreadable"
    with monkeypatch.context() as patched:
        patched.setattr(Path, "read_bytes", read_bytes)
        assert _copy_change(copy, recorded) == "unreadable"
    assert _copy_change(copy, recorded) is None


# ---------------------------------------------------------------------------
# Profile setup: the child's output
# ---------------------------------------------------------------------------

# The original base's child printed exactly these lines for the steps below (captured from the
# base before this change, on the same temporary input).
LEGACY_STDOUT = [
    '{"copied": ["spike"], "updated": [], "skipped": 0, "user_modified": [], "cleaned": [], '
    '"suppressed": [], "relocated": [], "total_bundled": 1, "optional_provenance_backfilled": [], '
    '"shadowed_by_external": [], "skipped_opt_out": false}\n',
    '{"copied": [], "updated": ["spike"], "skipped": 0, "user_modified": [], "cleaned": [], '
    '"suppressed": [], "relocated": [], "total_bundled": 1, "optional_provenance_backfilled": [], '
    '"shadowed_by_external": [], "skipped_opt_out": false}\n',
    '{"copied": [], "updated": [], "skipped": 0, "user_modified": [], "cleaned": ["spike"], '
    '"suppressed": [], "relocated": [], "total_bundled": 0, "optional_provenance_backfilled": [], '
    '"shadowed_by_external": [], "skipped_opt_out": false}\n',
]


@pytest.fixture
def child_stdout(monkeypatch) -> list:
    """Everything the seeding child prints, one entry per run."""
    stdouts = []
    real_run = subprocess.run

    def recording_run(*args, **kwargs):
        completed = real_run(*args, **kwargs)
        stdouts.append(completed.stdout)
        return completed
    monkeypatch.setattr(subprocess, "run", recording_run)
    return stdouts


def test_seed_profile_skills_without_keyword_prints_the_legacy_output_byte_for_byte(role, child_stdout):
    """The child prints the old keys, in the old order and serialization, byte for byte: the
    update command's input is unchanged."""
    profile = role.home / "profiles" / "builder"
    (profile / "skills").mkdir(parents=True)

    _write_skill(role.bundled, "software-development/spike", "v1")
    results = [_seed(profile)]
    _write_skill(role.bundled, "software-development/spike", "v2")
    results.append(_seed(profile))
    shutil.rmtree(role.bundled / "software-development")
    results.append(_seed(profile))

    assert child_stdout == LEGACY_STDOUT
    assert results == [json.loads(line) for line in LEGACY_STDOUT]


# ---------------------------------------------------------------------------
# Review round 3: no record proves nothing, finding no SKILL.md proves no deletion, a link on
# the way to a copy keeps it, and every kept or refused copy is named with its reason
# ---------------------------------------------------------------------------

def test_untracked_same_named_built_in_copy_never_gets_a_line_through_seeding_and_next_release(role):
    """Finding 1: the owner's folder equals the release's new built-in but has no record. Two real
    seeding children (this release and the next) keep it byte for byte and never record it; the
    sync's extended result names it under kept."""
    _write_skill(role.bundled, "media/ascii-art", "owner v2")
    copy = _write_skill(role.skills, "media/ascii-art", "owner v2")
    before = _files(copy)

    results = [_seed(role.home)]
    assert "ascii-art" not in _manifest(role)
    _write_skill(role.bundled, "media/ascii-art", "release v3")
    results.append(_seed(role.home))
    extended = sync_skills(quiet=True)

    for result in (*results, extended):
        assert result is not None
        assert "ascii-art" not in result["copied"] + result["updated"]
    assert extended["kept"] == [{"name": "ascii-art", "reason": "changed"}]
    assert extended["refused"] == []
    assert _files(copy) == before
    assert "ascii-art" not in _manifest(role)
    assert _lock(role).get("installed", {}) == {}


def test_reset_re_baselines_only_its_own_skill_and_only_in_that_command(role):
    """`hermes skills reset spike` (no --restore) on a copy equal to the release records it again.
    An untracked same-named copy of another built-in in that same sync is not adopted, and the
    authorization ends with that sync: later ordinary syncs adopt nothing."""
    from tools.skills_sync_bundled_ops import reset_bundled_skill
    _write_skill(role.bundled, "software-development/spike", "built-in v1")
    sync_skills(quiet=True)
    src = _write_skill(role.bundled, "software-development/spike", "built-in v2")
    spike = _write_skill(role.skills, "software-development/spike", "built-in v2")  # pasted the release
    _write_skill(role.bundled, "media/ascii-art", "built-in v1")
    art = _write_skill(role.skills, "media/ascii-art", "built-in v1")  # the owner's, no record
    spike_before, art_before = _files(spike), _files(art)
    assert "spike" in sync_skills(quiet=True)["user_modified"]
    assert "ascii-art" not in _manifest(role)

    result = reset_bundled_skill("spike", restore=False)

    assert result["ok"] is True and result["action"] == "manifest_cleared"
    assert _manifest(role)["spike"] == _dir_hash(src)
    assert "ascii-art" not in _manifest(role)
    assert result["synced"]["kept"] == [{"name": "ascii-art", "reason": "changed"}]
    later = sync_skills(quiet=True)
    assert later["kept"] == [{"name": "ascii-art", "reason": "changed"}]
    assert "ascii-art" not in _manifest(role)
    # The owner drops spike's line by hand: an ordinary sync does not record the copy again.
    (role.skills / ".bundled_manifest").write_text("", encoding="utf-8")
    again = sync_skills(quiet=True)
    assert sorted(again["kept"], key=lambda item: item["name"]) == [
        {"name": "ascii-art", "reason": "changed"}, {"name": "spike", "reason": "changed"}]
    assert _manifest(role) == {}
    assert (_files(spike), _files(art)) == (spike_before, art_before)


def _moved_built_in_copy(role) -> tuple:
    """A real seeding child installs built-in ascii-art v1: ``(copy, its manifest line)``."""
    _write_skill(role.bundled, "media/ascii-art", "Hermes v1")
    assert _seed(role.home) is not None
    return role.skills / "media" / "ascii-art", _manifest(role)["ascii-art"]


def test_changed_copy_keeps_line_and_bytes_through_removal_reintroduction_and_next_release(role):
    """Finding 2, in a marked role: a copy kept as changed, its source removed from both catalogs,
    then reintroduced with the copy's own content, then a new release: every step keeps the line
    and the bytes (a marked role cleans no line; an unmarked one cleans it, as on the base)."""
    _write_skill(role.bundled, "media/ascii-art", "Hermes v1")
    assert _seed(role.home) is not None
    _mark(role)
    copy = _write_skill(role.skills, "media/ascii-art", "owner v2")
    owners, line = _files(copy), _manifest(role)["ascii-art"]
    steps = [
        (lambda: _write_skill(role.bundled, "media/ascii-art", "released v3"), True),
        (lambda: shutil.rmtree(role.bundled / "media"), False),  # removed from both catalogs
        (lambda: None, False),  # the next sync keeps the line too
        (lambda: _write_skill(role.bundled, "media/ascii-art", "owner v2"), True),  # equal to the copy
        (lambda: _write_skill(role.bundled, "media/ascii-art", "released v4"), True),
    ]
    for step, built_in in steps:
        step()
        result = sync_skills(quiet=True)
        assert result["kept"] == ([{"name": "ascii-art", "reason": "changed"}] if built_in else [])
        assert "ascii-art" not in result["cleaned"] + result["updated"] + result["copied"]
        assert _manifest(role)["ascii-art"] == line
        assert _files(copy) == owners


@pytest.mark.parametrize("marked", [True, False], ids=["marked", "unmarked"])
def test_category_that_is_a_link_keeps_copy_and_line_through_seeding(role, tmp_path, marked):
    """Finding 3: the owner moves the category folder out of the tree and links it back in."""
    _write_skill(role.bundled, "media/ascii-art", "Hermes v1")
    assert _seed(role.home) is not None
    if marked:
        _mark(role)
    line = _manifest(role)["ascii-art"]
    outside = tmp_path / "owner-category"
    shutil.move(role.skills / "media", outside)
    os.symlink(outside, role.skills / "media", target_is_directory=True)
    before = _files(outside)
    _write_skill(role.bundled, "media/ascii-art", "released v2")

    result = sync_skills(quiet=True)

    assert result is not None
    assert os.path.islink(role.skills / "media") and os.readlink(role.skills / "media") == str(outside)
    assert _files(outside) == before
    assert not any(b"released v2" in data for data in _files(role.home).values())
    assert _manifest(role)["ascii-art"] == line
    assert result["kept"] == [{"name": "ascii-art", "reason": "link"}]
    assert result["refused"] == [] and result["updated"] == []


def test_new_built_in_whose_category_is_a_link_is_refused_and_nothing_is_written(role, tmp_path):
    outside = tmp_path / "owner-category"
    outside.mkdir()
    (outside / "notes.txt").write_text("owner notes\n", encoding="utf-8")
    os.symlink(outside, role.skills / "media", target_is_directory=True)
    _write_skill(role.bundled, "media/ascii-art", "Hermes v1")

    result = sync_skills(quiet=True)

    assert result is not None
    assert _files(outside) == {"notes.txt": b"owner notes\n"}
    assert "ascii-art" not in _manifest(role)
    assert result["refused"] == [{"name": "ascii-art", "reason": "link"}]
    assert result["copied"] == [] and result["kept"] == []


def test_backup_removal_failure_after_copy_and_record_counts_as_updated(role, monkeypatch):
    """The new copy and its line are both written; the old copy's backup then cannot be removed
    (the rmtree scope guard refuses): the update is done, never refused or rolled back."""
    from tools import skills_sync
    _write_skill(role.bundled, "software-development/spike", "built-in v1")
    sync_skills(quiet=True)
    src = _write_skill(role.bundled, "software-development/spike", "built-in v2")
    real_rmtree = skills_sync._rmtree_writable

    def rmtree(path):
        if Path(path).suffix == ".bak":
            raise ValueError(f"refusing to rmtree {path!r} (scope guard)")
        return real_rmtree(path)
    with monkeypatch.context() as patched:
        patched.setattr(skills_sync, "_rmtree_writable", rmtree)
        result = sync_skills(quiet=True)

    assert result["updated"] == ["spike"]
    assert result["refused"] == [] and result["kept"] == []
    assert _files(role.skills / "software-development" / "spike") == _files(src)
    assert _manifest(role)["spike"] == _dir_hash(src)


def test_owner_link_in_a_copy_whose_built_in_did_not_change_is_reported_kept(role, tmp_path):
    """Finding 4: the release does not change the built-in; the copy is still checked and named."""
    copy, link, target, before, line = _builtin_copy_with_owner_link(role, tmp_path)

    result = sync_skills(quiet=True)

    _assert_owner_link_kept(copy, before, link, target)
    assert _manifest(role)["ascii-art"] == line
    assert result["kept"] == [{"name": "ascii-art", "reason": "link"}]
    assert result["user_modified"] == [] and result["updated"] == []
    assert result["skipped"] == 1


def _recategorized_in_unmarked_role(role) -> tuple:
    """Unmarked role (a marked role searches no other folder, owner rule 2 point 2); the release
    moves tracked built-in ascii-art from media to creative."""
    copy, line = _moved_built_in_copy(role)
    category = role.skills / "creative"
    category.mkdir()
    shutil.rmtree(role.bundled / "media")
    _write_skill(role.bundled, "creative/ascii-art", "released v2")
    return copy, category, _files(copy), line


def _assert_relocation_refused(role, result, copy, category, before, line) -> None:
    assert result is not None
    assert result["refused"] == [{"name": "ascii-art", "reason": "copy write failed"}]
    assert result["relocated"] == [] and result["updated"] == []
    assert _files(copy) == before
    assert not os.path.lexists(category / "ascii-art")
    assert _manifest(role)["ascii-art"] == line


@needs_real_permissions
def test_relocation_into_a_read_only_category_is_refused_and_the_old_copy_stays(role):
    copy, category, before, line = _recategorized_in_unmarked_role(role)
    try:
        category.chmod(0o555)
        result = sync_skills(quiet=True)
    finally:
        category.chmod(0o755)

    _assert_relocation_refused(role, result, copy, category, before, line)


def test_relocation_whose_move_fails_is_refused_and_the_old_copy_stays(role, monkeypatch):
    """Twin of the read-only category: the move itself fails."""
    from tools import skills_sync
    copy, category, before, line = _recategorized_in_unmarked_role(role)

    def move_dir(src, dest):
        raise PermissionError(13, "Permission denied", str(dest))
    with monkeypatch.context() as patched:
        patched.setattr(skills_sync, "_move_dir", move_dir)
        result = sync_skills(quiet=True)

    _assert_relocation_refused(role, result, copy, category, before, line)


