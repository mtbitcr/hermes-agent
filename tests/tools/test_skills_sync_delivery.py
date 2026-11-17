"""Released skill changes reach every role; changed and deleted copies are kept.

Each test runs the real sync against a temporary role home, built-in tree and optional
tree, set through ``HERMES_HOME``, ``HERMES_BUNDLED_SKILLS`` and ``HERMES_OPTIONAL_SKILLS``.
A "release" is a change to those trees followed by the sync (and the official refresh).
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
from tools.skills_sync_optional import restore_official_optional_skill

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
    """An official optional skill whose copy matches its hub lock (adopted on the owner's
    request, `hermes skills repair-official <name>`: the sync adopts nothing it has no record of)."""
    src = _write_skill(role.optional, rel, body)
    shutil.copytree(src, role.skills / rel)
    assert restore_official_optional_skill(src.name)["backfilled"] == [src.name]
    assert src.name in _lock(role)["installed"]
    return src


def _refresh(**kwargs) -> dict:
    from tools.skills_sync_optional import refresh_official_optional_skills
    return refresh_official_optional_skills(**kwargs)


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


def _seed(profile: Path, **kwargs):
    """The real profile seeding: a child process whose ``HERMES_HOME`` is ``profile``."""
    from hermes_cli.profiles import seed_profile_skills
    return seed_profile_skills(profile, quiet=True, **kwargs)


def _sync_and_refresh() -> dict:
    """In-process twin of ``_seed(..., refresh_official=True)``, for tests that patch a call."""
    return {**sync_skills(quiet=True), "official": _refresh()}


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


def _deny_lock_write(patched) -> None:
    """Twin of mode 0555 on skills/.hub: the atomic lock write cannot create its temporary file."""
    real_mkstemp = tempfile.mkstemp

    def mkstemp(*args, **kwargs):
        if os.path.basename(str(kwargs.get("dir", ""))) == ".hub":
            raise PermissionError(13, "Permission denied", kwargs["dir"])
        return real_mkstemp(*args, **kwargs)
    patched.setattr(tempfile, "mkstemp", mkstemp)


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


def test_marked_role_cleans_lines_for_names_no_longer_built_in(role):
    _write_skill(role.bundled, "planning/plan", "v1")
    _write_skill(role.bundled, "planning/old-helper", "v1")
    _write_skill(role.bundled, "software-development/spike", "v1")
    sync_skills(quiet=True)
    _mark(role)
    shutil.rmtree(role.skills / "planning" / "old-helper")
    shutil.rmtree(role.bundled / "planning")

    result = sync_skills(quiet=True)

    assert sorted(result["cleaned"]) == ["old-helper", "plan"]
    assert set(_manifest(role)) == {"spike"}


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


@pytest.mark.parametrize("marked", [True, False], ids=["marked", "unmarked"])
def test_moved_to_optional_unchanged_copy_replaced_in_place_with_official_source(role, marked):
    _write_skill(role.bundled, "media/ascii-art", "built-in v1")
    sync_skills(quiet=True)
    if marked:
        _mark(role)
    shutil.rmtree(role.bundled / "media")
    src = _write_skill(role.optional, "creative/ascii-art", "official optional v2")
    (src / "references").mkdir()
    (src / "references" / "fonts.md").write_text("fonts\n", encoding="utf-8")

    result = sync_skills(quiet=True)

    copy = role.skills / "media" / "ascii-art"
    assert _files(copy) == _files(src)
    assert not (role.skills / "creative" / "ascii-art").exists()
    entry = _lock(role)["installed"]["ascii-art"]
    assert entry["source"] == "official"
    assert entry["install_path"] == "media/ascii-art"
    assert entry["content_hash"] == content_hash(copy)
    assert "ascii-art" not in _manifest(role)
    assert result["moved_to_official"] == ["ascii-art"]


def test_moved_to_optional_changed_copy_kept_byte_for_byte_without_lock_entry(role):
    _write_skill(role.bundled, "media/ascii-art", "built-in v1")
    sync_skills(quiet=True)
    _mark(role)
    copy = role.skills / "media" / "ascii-art"
    _edit(copy)
    before, line = _files(copy), _manifest(role)["ascii-art"]
    shutil.rmtree(role.bundled / "media")
    _write_skill(role.optional, "creative/ascii-art", "official optional v2")

    result = sync_skills(quiet=True)

    assert _files(copy) == before
    assert "ascii-art" not in _lock(role).get("installed", {})
    assert result["moved_kept_changed"] == ["ascii-art"]
    assert result["moved_to_official"] == []
    assert result["kept"] == [{"name": "ascii-art", "reason": "changed"}]
    # "Changed" is durable: the line stays exactly as it was, so every later sync keeps the copy.
    assert _manifest(role)["ascii-art"] == line
    assert "ascii-art" not in result["cleaned"]


def test_moved_to_optional_without_copy_creates_nothing(role):
    _write_skill(role.bundled, "media/ascii-art", "built-in v1")
    sync_skills(quiet=True)
    _mark(role)
    shutil.rmtree(role.skills / "media" / "ascii-art")
    shutil.rmtree(role.bundled / "media")
    _write_skill(role.optional, "creative/ascii-art", "official optional v2")

    result = sync_skills(quiet=True)

    assert not (role.skills / "media" / "ascii-art").exists()
    assert not (role.skills / "creative" / "ascii-art").exists()
    assert "ascii-art" not in _lock(role).get("installed", {})
    assert "ascii-art" not in _manifest(role)
    assert "ascii-art" not in result["moved_to_official"] + result["moved_kept_changed"]


def test_removed_upstream_copy_stays_and_line_is_cleaned(role):
    """A built-in the release removed from both trees (like the planner role's `plan`)."""
    _write_skill(role.bundled, "planning/plan", "v1")
    sync_skills(quiet=True)
    _mark(role)
    copy = role.skills / "planning" / "plan"
    before = _files(copy)
    shutil.rmtree(role.bundled / "planning")

    result = sync_skills(quiet=True)

    assert _files(copy) == before
    assert "plan" in result["cleaned"]
    assert "plan" not in _manifest(role)
    assert "plan" not in result["moved_to_official"] + result["moved_kept_changed"]


# ---------------------------------------------------------------------------
# Official optional skills (the release's refresh)
# ---------------------------------------------------------------------------

def test_optional_official_unchanged_copy_updated_and_lock_hash_moves(role):
    """Production group: the one optional copy that matches its hub lock while the official
    source changed. It is updated and its lock hash moves."""
    src = _adopt_official_copy(role, "communication/yuanbao", "official v1")
    old_hash = _lock(role)["installed"]["yuanbao"]["content_hash"]
    _write_skill(role.optional, "communication/yuanbao", "official v2")
    (src / "references").mkdir()
    (src / "references" / "api.md").write_text("api\n", encoding="utf-8")

    report = _refresh()

    copy = role.skills / "communication" / "yuanbao"
    assert report["updated"] == ["yuanbao"]
    assert _files(copy) == _files(src)
    entry = _lock(role)["installed"]["yuanbao"]
    assert entry["content_hash"] == content_hash(copy)
    assert entry["content_hash"] != old_hash
    assert "references/api.md" in entry["files"]


def test_optional_official_changed_copy_kept_with_newer_official_and_hash(role):
    src = _adopt_official_copy(role, "communication/yuanbao", "official v1")
    copy = role.skills / "communication" / "yuanbao"
    _edit(copy)
    before, lock_before = _files(copy), _lock_bytes(role)
    _write_skill(role.optional, "communication/yuanbao", "official v2")

    report = _refresh()

    assert _files(copy) == before
    assert report["kept_changed"] == [
        {"name": "yuanbao", "newer_official": True, "official_hash": content_hash(src), "reason": "changed"}]
    assert report["updated"] == []
    assert _lock_bytes(role) == lock_before


def test_optional_lock_entry_without_folder_not_recreated(role):
    _adopt_official_copy(role, "communication/yuanbao", "official v1")
    shutil.rmtree(role.skills / "communication" / "yuanbao")
    _write_skill(role.optional, "communication/yuanbao", "official v2")

    report = _refresh()

    assert report["skipped_deleted"] == ["yuanbao"]
    assert not (role.skills / "communication" / "yuanbao").exists()
    assert "yuanbao" in _lock(role)["installed"]


def test_optional_copy_without_record_untouched_and_reported_untracked(role):
    """Production group: nine optional copies have no record (no hub-lock entry and no
    manifest line). They are never touched and are reported as untracked."""
    _write_skill(role.optional, "research/blogwatcher", "official v2")
    stale = _write_skill(role.skills, "research/blogwatcher", "an older official text")
    before = _files(stale)

    sync_skills(quiet=True)
    report = _refresh()

    assert _files(stale) == before
    assert report["kept_untracked"] == [{"name": "blogwatcher", "reason": "changed"}]
    assert "blogwatcher" not in _lock(role).get("installed", {})


@pytest.mark.parametrize("kind", ["traversal", "absolute", "symlink", "archived"])
def test_unsafe_lock_install_path_refused_and_nothing_written(role, tmp_path, kind):
    _write_skill(role.optional, "communication/yuanbao", "official v2")
    outside = _write_skill(tmp_path / "outside", "yuanbao", "official v1")
    install_path = {"traversal": "../../outside/yuanbao", "absolute": str(outside),
                    "symlink": "communication/yuanbao", "archived": ".archive/yuanbao"}[kind]
    if kind == "symlink":
        (role.skills / "communication").mkdir()
        (role.skills / "communication" / "yuanbao").symlink_to(outside, target_is_directory=True)
    if kind == "archived":  # a matching copy inside a folder skill scanners skip
        shutil.copytree(outside, role.skills / ".archive" / "yuanbao")
    (role.skills / ".hub").mkdir()
    (role.skills / ".hub" / "lock.json").write_text(json.dumps({"version": 1, "installed": {"yuanbao": {
        "source": "official", "identifier": "official/communication/yuanbao", "install_path": install_path,
        "content_hash": content_hash(outside), "files": ["SKILL.md"]}}}), encoding="utf-8")
    skills_before, outside_before = _files(role.skills), _files(outside)

    report = _refresh()

    reason = "link" if kind == "symlink" else "unsafe path"
    assert report["refused"] == [{"name": "yuanbao", "reason": reason}]
    assert report["updated"] == []
    assert _files(role.skills) == skills_before
    assert _files(outside) == outside_before


def test_unreadable_official_copy_kept_as_unreadable_and_nothing_written(role, monkeypatch):
    """Tests run as root, which reads everything, so the copy's files are made to fail on read."""
    src = _adopt_official_copy(role, "communication/yuanbao", "official v1")
    _write_skill(role.optional, "communication/yuanbao", "official v2")
    copy = role.skills / "communication" / "yuanbao"
    before, lock_before = _files(copy), _lock_bytes(role)
    real_read_bytes = Path.read_bytes

    def read_bytes(path):
        if copy in path.parents:
            raise PermissionError(13, "Permission denied", str(path))
        return real_read_bytes(path)

    with monkeypatch.context() as patched:
        patched.setattr(Path, "read_bytes", read_bytes)
        report = _refresh()

    assert [(item["name"], item["reason"]) for item in report["kept_changed"]] == [("yuanbao", "unreadable")]
    assert report["refused"] == []
    assert report["updated"] == []
    assert _files(copy) == before != _files(src)
    assert _lock_bytes(role) == lock_before


def test_owner_edit_equal_to_the_new_release_stays_the_owners_through_the_next_release(role):
    """Ownership comes from the recorded hash only: Hermes installs v1 and the owner edits the
    copy to bytes equal to v2. The refresh keeps it as changed without moving the lock's hash,
    and release v3 still leaves it untouched."""
    _adopt_official_copy(role, "communication/yuanbao", "official v1")
    _write_skill(role.optional, "communication/yuanbao", "official v2")
    copy = _write_skill(role.skills, "communication/yuanbao", "official v2")  # the owner's edit
    owners, lock_before = _files(copy), _lock_bytes(role)

    report = _refresh()

    assert [item["name"] for item in report["kept_changed"]] == ["yuanbao"]
    assert report["updated"] == []
    assert _lock_bytes(role) == lock_before

    _write_skill(role.optional, "communication/yuanbao", "official v3")
    report = _sync_and_refresh()["official"]

    assert [item["name"] for item in report["kept_changed"]] == ["yuanbao"]
    assert report["updated"] == []
    assert _files(copy) == owners
    assert _lock_bytes(role) == lock_before


def test_moved_skill_edited_to_equal_its_official_source_stays_the_owners_through_next_release(role):
    """The owner edits a built-in copy to bytes equal to the optional version it moves to. The
    hand-over keeps it as changed (its manifest hash says so), the same sync's backfill does not
    adopt it, and the release after it leaves it untouched."""
    _write_skill(role.bundled, "media/ascii-art", "built-in v1")
    sync_skills(quiet=True)
    copy = _write_skill(role.skills, "media/ascii-art", "official optional v2")  # the owner's edit
    owners = _files(copy)
    shutil.rmtree(role.bundled / "media")
    _write_skill(role.optional, "creative/ascii-art", "official optional v2")

    result = _sync_and_refresh()

    assert result["moved_kept_changed"] == ["ascii-art"]
    assert result["moved_to_official"] == []
    assert result["optional_provenance_backfilled"] == []
    assert "ascii-art" not in _lock(role).get("installed", {})

    _write_skill(role.optional, "creative/ascii-art", "official optional v3")
    result = _sync_and_refresh()

    assert result["official"]["updated"] == []
    assert _files(copy) == owners
    assert "ascii-art" not in _lock(role).get("installed", {})


def test_refresh_dry_run_writes_nothing(role):
    _adopt_official_copy(role, "communication/yuanbao", "official v1")
    _write_skill(role.optional, "communication/yuanbao", "official v2")
    before = _files(role.home)

    report = _refresh(dry_run=True)

    assert report["updated"] == ["yuanbao"]
    assert _files(role.home) == before


# ---------------------------------------------------------------------------
# The three data rules
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("marked", [True, False], ids=["marked", "unmarked"])
def test_data_rule_changed_copy_is_never_overwritten(role, marked):
    """Data rule: no copy changed after Hermes wrote it is overwritten, even when a newer
    official version exists: built-in, moved to the optional catalog, or official optional."""
    rels, release = _three_kinds_then_release(role, marked=marked)
    for rel in rels:
        _edit(role.skills / rel)
    before = {rel: _files(role.skills / rel) for rel in rels}
    release()

    result = sync_skills(quiet=True)
    report = _refresh()

    for rel, files in before.items():
        assert _files(role.skills / rel) == files, rel
    assert "spike" in result["user_modified"]
    assert result["moved_kept_changed"] == ["ascii-art"]
    assert [item["name"] for item in report["kept_changed"]] == ["yuanbao"]
    assert report["updated"] == []


@pytest.mark.parametrize("marked", [True, False], ids=["marked", "unmarked"])
def test_data_rule_deleted_copy_never_comes_back(role, marked):
    """Data rule: no deleted copy comes back: built-in, moved to the optional catalog, or
    official optional, at this release and the next one."""
    rels, release = _three_kinds_then_release(role, marked=marked)
    for rel in rels:
        shutil.rmtree(role.skills / rel)
    release()

    for _release in range(2):
        sync_skills(quiet=True)
        report = _refresh()
        for rel in (*rels, "creative/ascii-art"):
            assert not (role.skills / rel).exists(), rel
    assert report["skipped_deleted"] == ["yuanbao"]


def test_data_rule_failed_replacement_restores_old_copy(role, monkeypatch):
    """Data rule: a failed replacement restores the old copy, for a built-in update, a
    hand-over to the optional catalog and an optional refresh. The next release retries."""
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
        report = _refresh()

    for rel, files in before.items():
        assert _files(role.skills / rel) == files, rel
        assert not (role.skills / rel).with_suffix(".bak").exists(), rel
    assert result["updated"] == []
    assert result["moved_to_official"] == []
    assert sorted(result["refused"], key=lambda item: item["name"]) == [
        {"name": "ascii-art", "reason": "copy write failed"}, {"name": "spike", "reason": "copy write failed"}]
    assert report["refused"] == [{"name": "yuanbao", "reason": "copy write failed"}]
    assert _lock_bytes(role) == lock_before

    result = sync_skills(quiet=True)
    report = _refresh()

    assert result["updated"] == ["spike"]
    assert result["moved_to_official"] == ["ascii-art"]
    assert report["updated"] == ["yuanbao"]


# ---------------------------------------------------------------------------
# A copy Hermes cannot read whole is never replaced
# ---------------------------------------------------------------------------

def _assert_every_unreadable_copy_kept(role, results, before, lock_before) -> None:
    for result in results:
        assert result is not None  # the seeding child finished
        assert "spike" in result["user_modified"]
        assert result["updated"] == []
        assert result["moved_to_official"] == []
        assert result["moved_kept_changed"] == ["ascii-art"]
        assert sorted(result["kept"], key=lambda item: item["name"]) == [
            {"name": "ascii-art", "reason": "unreadable"}, {"name": "spike", "reason": "unreadable"}]
        assert result["official"]["updated"] == []
        assert [(item["name"], item["reason"]) for item in result["official"]["kept_changed"]] == [
            ("yuanbao", "unreadable")]
    assert "ascii-art" in _manifest(role)
    for copy, files in before.items():
        assert _files(copy) == files, copy  # the owner's file is still in the active copy
        assert not copy.with_suffix(".bak").exists(), copy
    assert _lock_bytes(role) == lock_before


@needs_real_permissions
def test_folder_the_owner_made_unreadable_keeps_every_copy_through_seeding_and_next_release(role):
    """Real mode 000 on a folder the owner added to a built-in copy, a moved built-in's copy and
    an official optional copy. Seeding at the release, seeding again and seeding at the release
    after it replace none of them, report each one and still return a result."""
    rels, release = _three_kinds_then_release(role, marked=True)
    copies = [role.skills / rel for rel in rels]
    hidden = [_add_owner_folder(copy) for copy in copies]
    before, lock_before = {copy: _files(copy) for copy in copies}, _lock_bytes(role)
    try:
        for folder in hidden:
            folder.chmod(0)
        release()
        results = [_seed(role.home, refresh_official=True) for _ in range(2)]
        _release_v3(role)
        results.append(_seed(role.home, refresh_official=True))
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
        results = [_sync_and_refresh() for _ in range(2)]
        _release_v3(role)
        results.append(_sync_and_refresh())

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
        results = [_seed(role.home, refresh_official=True) for _ in range(2)]
    finally:
        _readable_again(hidden)

    _assert_shadowed_copy_kept(results, copy, before)


def test_shadowed_copy_whose_folder_cannot_be_listed_is_kept_through_next_sync(role, tmp_path, monkeypatch):
    """Twin of the mode-000 test that runs as any user: listing the owner's folder fails."""
    copy, _ = _shadowed_copy_with_owner_folder(role, tmp_path)
    before = _files(copy)

    with monkeypatch.context() as patched:
        _deny_listing(patched)
        results = [_sync_and_refresh() for _ in range(2)]

    _assert_shadowed_copy_kept(results, copy, before)


# ---------------------------------------------------------------------------
# A failed lock write keeps the old copy and the old lock
# ---------------------------------------------------------------------------

def _assert_lock_failure_rolled_back(role, result, before, lock_before) -> None:
    assert result is not None  # the seeding child finished
    assert result["moved_to_official"] == []
    assert "ascii-art" not in result["cleaned"] + result["updated"]
    # The moved skill itself is refused with its reason, not only the refreshed optional skill.
    assert {"name": "ascii-art", "reason": "record write failed"} in result["refused"]
    assert "ascii-art" in _manifest(role)  # the hand-over is retried next sync
    assert result["optional_provenance_backfilled"] == []
    assert result["official"]["updated"] == []
    assert result["official"]["refused"] == [{"name": "yuanbao", "reason": "record write failed"}]
    for copy, files in before.items():
        assert _files(copy) == files, copy
        assert not copy.with_suffix(".bak").exists(), copy
    assert _lock_bytes(role) == lock_before


@needs_real_permissions
def test_unwritable_hub_restores_old_copies_and_prior_lock_through_seeding(role):
    """Real mode 0555 on skills/.hub: the hand-over and the optional refresh each put the old
    copy back, keep the prior lock and report a refusal, not an update, and nothing later in the
    run commits the failed hand-over. Once the lock is writable, the next seeding delivers both."""
    rels, release = _three_kinds_then_release(role, marked=True)
    kept = [role.skills / rel for rel in rels[1:]]
    before, lock_before = {copy: _files(copy) for copy in kept}, _lock_bytes(role)
    hub = role.skills / ".hub"
    release()
    try:
        hub.chmod(0o555)
        result = _seed(role.home, refresh_official=True)
    finally:
        hub.chmod(0o755)

    _assert_lock_failure_rolled_back(role, result, before, lock_before)
    retried = _seed(role.home, refresh_official=True)
    assert retried["moved_to_official"] == ["ascii-art"]
    assert retried["official"]["updated"] == ["yuanbao"]


def test_lock_write_failure_restores_old_copies_and_prior_lock(role, monkeypatch):
    """Twin that runs as any user: the lock's temporary file cannot be created in skills/.hub,
    for the hand-over and for the optional refresh; the copy-failure test is separate."""
    rels, release = _three_kinds_then_release(role, marked=True)
    kept = [role.skills / rel for rel in rels[1:]]
    before, lock_before = {copy: _files(copy) for copy in kept}, _lock_bytes(role)
    release()

    with monkeypatch.context() as patched:
        _deny_lock_write(patched)
        result = _sync_and_refresh()

    _assert_lock_failure_rolled_back(role, result, before, lock_before)
    retried = _sync_and_refresh()
    assert retried["moved_to_official"] == ["ascii-art"]
    assert retried["official"]["updated"] == ["yuanbao"]


def test_moved_skill_matching_two_folders_hands_over_neither_and_is_refused_ambiguous(role):
    """A built-in moved to the optional catalog whose name matches two folders: neither is
    overwritten or recorded, and the release step reports a refusal with reason "ambiguous"."""
    _write_skill(role.bundled, "media/ascii-art", "built-in v1")
    sync_skills(quiet=True)
    copies = [role.skills / "media" / "ascii-art", role.skills / "art" / "ascii-art"]
    shutil.copytree(copies[0], copies[1])
    before = {copy: _files(copy) for copy in copies}
    shutil.rmtree(role.bundled / "media")
    _write_skill(role.optional, "creative/ascii-art", "official optional v2")

    result = _seed(role.home, refresh_official=True)

    assert result["refused"] == [{"name": "ascii-art", "reason": "ambiguous"}]
    assert result["official"]["refused"] == []  # the sync reports it, once
    assert result["moved_to_official"] == []
    assert "ascii-art" not in result["cleaned"]
    assert result["optional_provenance_backfilled"] == []
    for copy, files in before.items():
        assert _files(copy) == files, copy
    assert "ascii-art" not in _lock(role).get("installed", {})


@needs_real_permissions
def test_unreadable_official_source_refuses_the_hand_over_through_seeding(role):
    """Real mode 000 on a folder of the official source: the hand-over cannot write the new
    copy, puts the old one back, keeps the line and names the moved skill under refused."""
    _write_skill(role.bundled, "media/ascii-art", "built-in v1")
    assert _seed(role.home) is not None
    copy = role.skills / "media" / "ascii-art"
    before, line = _files(copy), _manifest(role)["ascii-art"]
    shutil.rmtree(role.bundled / "media")
    hidden = _add_owner_folder(_write_skill(role.optional, "creative/ascii-art", "official optional v2"))
    try:
        hidden.chmod(0)
        result = _seed(role.home, refresh_official=True)
    finally:
        hidden.chmod(0o755)

    assert result is not None
    assert {"name": "ascii-art", "reason": "copy write failed"} in result["refused"]
    assert "ascii-art" not in result["moved_to_official"] + result["updated"] + result["cleaned"]
    assert _files(copy) == before
    assert not copy.with_suffix(".bak").exists()
    assert _manifest(role)["ascii-art"] == line
    assert "ascii-art" not in _lock(role).get("installed", {})
    assert _seed(role.home, refresh_official=True)["moved_to_official"] == ["ascii-art"]


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

    result = _seed(role.home, refresh_official=True)

    _assert_owner_link_kept(copy, before, link, target)
    assert _manifest(role)["ascii-art"] == line
    assert result["kept"] == [{"name": "ascii-art", "reason": "link"}]
    assert "ascii-art" in result["user_modified"]
    assert "ascii-art" not in result["updated"]


def test_owner_link_keeps_moved_copy_at_hand_over(role, tmp_path):
    copy, link, target, before, line = _builtin_copy_with_owner_link(role, tmp_path)
    shutil.rmtree(role.bundled / "media")
    _write_skill(role.optional, "creative/ascii-art", "released optional v2")

    result = _seed(role.home, refresh_official=True)

    _assert_owner_link_kept(copy, before, link, target)
    assert _manifest(role)["ascii-art"] == line
    assert "ascii-art" not in _lock(role).get("installed", {})
    assert result["kept"] == [{"name": "ascii-art", "reason": "link"}]
    assert result["moved_to_official"] == []


def test_owner_link_keeps_official_optional_copy_at_refresh(role, tmp_path):
    _adopt_official_copy(role, "communication/yuanbao", "official v1")
    copy = role.skills / "communication" / "yuanbao"
    link, target = _add_owner_link(copy, tmp_path / "outside")
    before, lock_before = _files(copy), _lock_bytes(role)
    _write_skill(role.optional, "communication/yuanbao", "released optional v2")

    result = _seed(role.home, refresh_official=True)

    _assert_owner_link_kept(copy, before, link, target)
    assert _lock_bytes(role) == lock_before
    assert [(item["name"], item["reason"]) for item in result["official"]["kept_changed"]] == [("yuanbao", "link")]
    assert result["official"]["updated"] == []


def test_owner_link_keeps_copy_that_an_external_folder_now_provides(role, tmp_path):
    """Marked role, configured external source: the local copy only looks like the stale shadow
    Hermes wrote, so it is not deleted."""
    _write_skill(role.bundled, "devops/ascii-art", "bundled v1")
    assert _seed(role.home) is not None
    copy = role.skills / "devops" / "ascii-art"
    link, target = _add_owner_link(copy, tmp_path / "outside")
    before, line = _files(copy), _manifest(role)["ascii-art"]
    _mark(role)
    external = tmp_path / "external"
    _write_skill(external, "devops/ascii-art", "external v1")
    (role.home / "config.yaml").write_text(f"skills:\n  external_dirs:\n    - {external}\n", encoding="utf-8")

    result = _seed(role.home, refresh_official=True)

    _assert_owner_link_kept(copy, before, link, target)
    assert _manifest(role)["ascii-art"] == line
    assert result["shadowed_by_external"] == ["ascii-art"]
    assert result["kept"] == [{"name": "ascii-art", "reason": "link"}]


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
    for result in results:
        assert result["kept"] == [{"name": "ascii-art", "reason": "link"}]
        assert "ascii-art" not in result["copied"] + result["updated"]


def test_owner_link_keeps_copy_from_adoption_by_repair_official(role, tmp_path):
    src = _write_skill(role.optional, "communication/yuanbao", "official v1")
    copy = role.skills / "communication" / "yuanbao"
    shutil.copytree(src, copy)
    link, target = _add_owner_link(copy, tmp_path / "outside")
    before = _files(copy)

    report = restore_official_optional_skill("yuanbao")

    assert report["ok"] is True
    assert report["backfilled"] == []
    assert "yuanbao" not in _lock(role).get("installed", {})
    _assert_owner_link_kept(copy, before, link, target)


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


@pytest.mark.parametrize("first", ["sync-first", "refresh-first"])
def test_moved_copy_kept_as_changed_stays_kept_through_later_syncs_and_releases(role, first):
    """Finding 2: Hermes installs built-in v1 and the owner edits the copy to bytes equal to the
    optional version it moves to. The first pass keeps it; a second ordinary sync at the same
    release, the release step at that release and the release step at the next one keep it too:
    its line stays exactly as it was and no lock entry is ever written for it."""
    _write_skill(role.bundled, "media/ascii-art", "Hermes wrote v1")
    assert _seed(role.home) is not None
    copy = _write_skill(role.skills, "media/ascii-art", "owner edit equal to optional v2")
    owners, line = _files(copy), _manifest(role)["ascii-art"]
    shutil.rmtree(role.bundled / "media")
    _write_skill(role.optional, "creative/ascii-art", "owner edit equal to optional v2")

    def check(result) -> None:
        assert result is not None
        assert result["kept"] == [{"name": "ascii-art", "reason": "changed"}]
        assert result["moved_kept_changed"] == ["ascii-art"]
        assert "ascii-art" not in result["cleaned"] + result["moved_to_official"]
        assert result.get("official", {}).get("updated", []) == []
        assert _files(copy) == owners
        assert _manifest(role)["ascii-art"] == line
        assert "ascii-art" not in _lock(role).get("installed", {})

    check(_seed(role.home, refresh_official=True) if first == "refresh-first" else sync_skills(quiet=True))
    legacy = _seed(role.home)  # a second ordinary sync at the same release, without the keyword
    assert legacy is not None
    assert "ascii-art" not in legacy["cleaned"]
    assert legacy["optional_provenance_backfilled"] == []
    check(sync_skills(quiet=True))
    check(_seed(role.home, refresh_official=True))
    _write_skill(role.optional, "creative/ascii-art", "optional v3")
    check(_seed(role.home, refresh_official=True))


def test_sync_never_adopts_an_optional_copy_without_a_record_but_repair_official_does(role):
    src = _write_skill(role.optional, "research/blogwatcher", "official v1")
    copy = role.skills / "research" / "blogwatcher"
    shutil.copytree(src, copy)  # byte-identical to the official source, but Hermes wrote no record

    result = _seed(role.home, refresh_official=True)
    again = sync_skills(quiet=True)

    assert result["optional_provenance_backfilled"] == again["optional_provenance_backfilled"] == []
    assert result["official"]["kept_untracked"] == [{"name": "blogwatcher", "reason": "changed"}]
    assert "blogwatcher" not in _lock(role).get("installed", {})

    report = restore_official_optional_skill("blogwatcher")  # the owner's explicit request

    assert report["backfilled"] == ["blogwatcher"]
    assert _lock(role)["installed"]["blogwatcher"]["content_hash"] == content_hash(copy)
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
    """Without ``refresh_official`` (omitted or False) the child prints the old keys, in the old
    order and serialization, byte for byte: the update command's input is unchanged."""
    profile = role.home / "profiles" / "builder"
    (profile / "skills").mkdir(parents=True)

    _write_skill(role.bundled, "software-development/spike", "v1")
    results = [_seed(profile)]
    _write_skill(role.bundled, "software-development/spike", "v2")
    results.append(_seed(profile, refresh_official=False))
    shutil.rmtree(role.bundled / "software-development")
    results.append(_seed(profile))

    assert child_stdout == LEGACY_STDOUT
    assert results == [json.loads(line) for line in LEGACY_STDOUT]


def test_seed_profile_skills_keyword_prints_the_extended_result_with_official(role, child_stdout):
    """With ``refresh_official=True`` the child prints one JSON object: the extended sync
    result plus the official refresh's report under ``official``."""
    profile = role.home / "profiles" / "builder"
    (profile / "skills").mkdir(parents=True)
    _write_skill(role.bundled, "software-development/spike", "v1")

    refreshed = _seed(profile, refresh_official=True)

    assert child_stdout == [json.dumps(refreshed) + "\n"]
    assert refreshed["copied"] == ["spike"]
    assert set(refreshed) >= set(json.loads(LEGACY_STDOUT[0])) | {
        "not_seeded_by_marker", "moved_to_official", "moved_kept_changed", "kept", "refused", "official"}
    assert set(refreshed["official"]) == {
        "updated", "kept_changed", "kept_untracked", "skipped_deleted", "skipped_removed_upstream", "refused"}
