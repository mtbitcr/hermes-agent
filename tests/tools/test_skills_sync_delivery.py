"""Released skill changes reach every role; changed and deleted copies are kept.

Each test runs the real sync against a temporary role home, built-in tree and optional
tree, set through ``HERMES_HOME``, ``HERMES_BUNDLED_SKILLS`` and ``HERMES_OPTIONAL_SKILLS``.
A "release" is a change to those trees followed by the sync (and the official refresh).
"""

import json
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from agent.skill_utils import ESSENTIAL_SKILLS
from tools.skills_guard import content_hash
from tools.skills_sync import NO_BUNDLED_SKILLS_MARKER, sync_skills

ESSENTIAL = sorted(ESSENTIAL_SKILLS)[0]


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
    """An official optional skill whose copy matches its hub lock (adopted by the real sync)."""
    src = _write_skill(role.optional, rel, body)
    shutil.copytree(src, role.skills / rel)
    sync_skills(quiet=True)
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
    before = _files(copy)
    shutil.rmtree(role.bundled / "media")
    _write_skill(role.optional, "creative/ascii-art", "official optional v2")

    result = sync_skills(quiet=True)

    assert _files(copy) == before
    assert "ascii-art" not in _lock(role).get("installed", {})
    assert result["moved_kept_changed"] == ["ascii-art"]
    assert result["moved_to_official"] == []


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
        {"name": "yuanbao", "newer_official": True, "official_hash": content_hash(src)}]
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
    assert report["kept_untracked"] == ["blogwatcher"]
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

    assert [item["name"] for item in report["refused"]] == ["yuanbao"]
    assert report["refused"][0]["reason"]
    assert report["updated"] == []
    assert _files(role.skills) == skills_before
    assert _files(outside) == outside_before


def test_unreadable_official_copy_refused_and_nothing_written(role, monkeypatch):
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

    assert [item["name"] for item in report["refused"]] == ["yuanbao"]
    assert report["refused"][0]["reason"]
    assert report["updated"] == []
    assert _files(copy) == before != _files(src)
    assert _lock_bytes(role) == lock_before


def test_interrupted_refresh_hash_is_recorded_next_time_not_reported_changed(role):
    """A run stopped after the copy was replaced but before the lock was written leaves the copy
    equal to the official source under the old hash; the next refresh records the hash."""
    src = _adopt_official_copy(role, "communication/yuanbao", "official v1")
    _write_skill(role.optional, "communication/yuanbao", "official v2")
    copy = role.skills / "communication" / "yuanbao"
    shutil.rmtree(copy)
    shutil.copytree(src, copy)

    report = _refresh()

    assert report["kept_changed"] == []
    assert report["updated"] == []
    assert _files(copy) == _files(src)
    assert _lock(role)["installed"]["yuanbao"]["content_hash"] == content_hash(copy)


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
    assert [item["name"] for item in report["refused"]] == ["yuanbao"]
    assert report["refused"][0]["reason"]
    assert _lock_bytes(role) == lock_before

    result = sync_skills(quiet=True)
    report = _refresh()

    assert result["updated"] == ["spike"]
    assert result["moved_to_official"] == ["ascii-art"]
    assert report["updated"] == ["yuanbao"]


# ---------------------------------------------------------------------------
# Profile setup: the child's output
# ---------------------------------------------------------------------------

def test_seed_profile_skills_keyword_adds_official_and_default_keeps_old_shape(role, monkeypatch):
    """Without the keyword the child prints exactly ``json.dumps(result) + "\\n"`` (the update
    command's input, unchanged); with it, one JSON object that also holds ``official``."""
    from hermes_cli.profiles import seed_profile_skills

    _write_skill(role.bundled, "software-development/spike", "v1")
    profile = role.home / "profiles" / "builder"
    (profile / "skills").mkdir(parents=True)
    stdouts = []
    real_run = subprocess.run

    def recording_run(*args, **kwargs):
        completed = real_run(*args, **kwargs)
        stdouts.append(completed.stdout)
        return completed

    monkeypatch.setattr(subprocess, "run", recording_run)

    result = seed_profile_skills(profile, quiet=True)

    assert isinstance(result, dict)
    assert "spike" in result["copied"]
    assert "official" not in result
    assert stdouts == [json.dumps(result) + "\n"]

    refreshed = seed_profile_skills(profile, quiet=True, refresh_official=True)

    assert len(stdouts) == 2
    assert json.loads(stdouts[1]) == refreshed
    assert stdouts[1].count("\n") == 1
    assert set(refreshed) == set(result) | {"official"}
    assert set(refreshed["official"]) >= {
        "updated", "kept_changed", "kept_untracked", "skipped_deleted", "skipped_removed_upstream", "refused"}
