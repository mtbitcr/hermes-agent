"""Official optional-skill provenance: hub-lock backfill, restore, the hand-over of built-ins that
moved to the optional catalog, and the release-step refresh. Profile-scoped paths and patchable
helpers resolve through ``_ss()`` at call time so ``tools.skills_sync`` patches work."""

import json
import logging
from contextlib import suppress
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Dict, Iterator, List, Optional, Set, Tuple

from agent.skill_utils import is_excluded_skill_path
from utils import atomic_write_text

logger = logging.getLogger("tools.skills_sync")


def _ss():
    """Live ``tools.skills_sync`` module (imported lazily: it imports helpers from this module)."""
    from tools import skills_sync
    return skills_sync


def _content_hash(directory: Path) -> str:
    """Hub-lock hash style; provenance metadata only, so fall back to local MD5 sans guard deps."""
    try:
        from tools.skills_guard import content_hash
        return content_hash(directory)
    except Exception:
        return _ss()._dir_hash(directory)


def _safe_rel_install_path(path: Path, base: Path) -> str:
    """Normalized relative POSIX path; rejects traversal/absolute paths."""
    pure = PurePosixPath(posix := path.relative_to(base).as_posix())
    parts = [part for part in pure.parts if part not in {"", "."}]
    if pure.is_absolute() or not parts or ".." in parts:
        raise ValueError(f"Unsafe optional skill path: {posix}")
    return "/".join(parts)


# Only generated runtime state, never generic cache/ or arbitrary dotfiles. This
# is updater ownership, not the security scanner's content-integrity policy.
_RUNTIME_CACHE_DIRS = frozenset({"__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"})


def _is_runtime_cache(path: Path, skill_dir: Path) -> bool:
    """Whether a skill-relative path is disposable Python/tool runtime state.

    Install prefixes may themselves contain a cache directory name. Legacy
    sibling bytecode is ignored only alongside its source; source-less .pyc
    files can be deliberately shipped or user-owned content.
    """
    relative = path.relative_to(skill_dir)
    if any(part in _RUNTIME_CACHE_DIRS for part in relative.parts[:-1]):
        return True
    if path.name in _RUNTIME_CACHE_DIRS and path.is_dir():
        return True
    return path.suffix in {".pyc", ".pyo"} and path.with_suffix(".py").is_file()


def _ignore_runtime_cache(directory: str, names: List[str]) -> List[str]:
    """copytree callback: don't seed generated caches into managed packages."""
    root = Path(directory)
    return [name for name in names if _is_runtime_cache(root / name, root)]


def _skill_file_list(skill_dir: Path) -> List[str]:
    """List package files for provenance/diff, excluding generated runtime caches."""
    return [f.relative_to(skill_dir).as_posix() for f in sorted(skill_dir.rglob("*"))
            if not _is_runtime_cache(f, skill_dir) and f.is_file()]


def _load_hub_lock() -> Optional[dict]:
    """Parse the skills-hub lock; None when missing or unreadable."""
    try:
        return json.loads((_ss()._skills_dir() / ".hub" / "lock.json").read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None


def _hub_lock_entries(data: Optional[dict]) -> List[dict]:
    return [e for e in ((data or {}).get("installed") or {}).values() if isinstance(e, dict)]


def _read_hub_install_paths() -> Set[str]:
    """Hub-lock install paths as POSIX strings. Hub-installed skills are owned by the hub: rename
    recovery must not move them even when content matches a bundled origin hash (dangling lock)."""
    return {str(e["install_path"]).strip("/") for e in _hub_lock_entries(_load_hub_lock()) if e.get("install_path")}


def _iter_optional_skills(optional_dir: Path, *, root_relative: bool) -> Iterator[Tuple[Path, Path, str]]:
    """Yield ``(skill_md, src, install_path)`` for every safe official optional skill."""
    for skill_md in sorted(optional_dir.rglob("SKILL.md")):
        if (is_excluded_skill_path(skill_md.relative_to(optional_dir), root=optional_dir)
                if root_relative else is_excluded_skill_path(skill_md)):
            continue
        try:
            yield skill_md, skill_md.parent, _safe_rel_install_path(skill_md.parent, optional_dir)
        except ValueError as e:
            logger.debug("Skipping optional skill with unsafe path %s: %s", skill_md.parent, e)


def _optional_skill_index() -> Dict[str, Tuple[str, str, Path]]:
    """Official optional skills keyed by BOTH folder name and frontmatter name (hub-lock slug or
    user-facing name). Values are ``(folder_name, install_path, source_dir)``."""
    ss = _ss()
    optional_dir = ss._get_optional_dir()
    index: Dict[str, Tuple[str, str, Path]] = {}
    if optional_dir.exists():
        for skill_md, src, install_path in _iter_optional_skills(optional_dir, root_relative=True):
            value = (src.name, install_path, src)
            index[src.name] = index[ss._read_skill_name(skill_md, src.name)] = value
    return index


def _move_to_restore_backup(path: Path, backup_root: Path) -> str:
    """Move an existing skill directory into a restore backup, preserving rel path."""
    ss = _ss()
    rel = path.relative_to(ss._skills_dir())
    target, suffix = backup_root / rel, 0
    while target.exists():
        suffix += 1
        target = (backup_root / rel).with_name(f"{rel.name}-{suffix}")
    ss._move_dir(path, target)
    return rel.as_posix()


def restore_official_optional_skill(name: str, *, restore: bool = False) -> dict:
    """Restore one or all official optional skills from repo source. ``restore=False`` only
    backfills exact-match provenance; ``restore=True`` also backs up matching active copies
    and copies the official source into its canonical path."""
    ss = _ss()
    index = _optional_skill_index()
    if index and name in {"all", "*"}:
        targets = sorted(set(index.values()), key=lambda item: item[1])
    elif name in index:
        targets = [index[name]]
    else:
        message = f"Official optional skill not found: {name}" if index else "No official optional skills directory found."
        return {"ok": False, "message": message, "restored": [], "backfilled": [], "backed_up": []}
    restored: List[str] = []
    backed_up: List[str] = []
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    backup_root = ss._skills_dir() / ".restore-backups" / f"official-optional-{timestamp}"
    for folder_name, install_path, src in targets if restore else []:
        dest = ss._skills_dir() / Path(*install_path.split("/"))
        canonical_ok = dest.exists() and ss._dir_hash(dest) == ss._dir_hash(src)
        # Active copies by frontmatter name or folder slug (the curator may have moved the skill
        # into another category); ``dest`` itself is handled below.
        names = {folder_name, ss._read_skill_name(src / "SKILL.md", folder_name)}
        for md in ss._iter_active_skill_mds(sort=True):
            match = md.parent
            if match != dest and match.exists() and (
                    match.name == folder_name or ss._read_skill_name(md, match.name) in names):
                backed_up.append(_move_to_restore_backup(match, backup_root))
        if dest.exists() and not canonical_ok:
            backed_up.append(_move_to_restore_backup(dest, backup_root))
        if not dest.exists():
            ss._copy_dir(src, dest)
            restored.append(folder_name)
    return {
        "ok": True, "message": "Official optional skill repair complete.", "restored": restored,
        "backfilled": _backfill_optional_provenance(quiet=True), "backed_up": backed_up,
        "backup_dir": str(backup_root) if backed_up else ""}


def _index_installed_skill_dirs_by_name() -> Dict[str, List[Path]]:
    """Installed skills by directory name in one active-tree scan, skipping anything that
    resolves outside the skills tree (symlinks/external)."""
    ss = _ss()
    index: Dict[str, List[Path]] = {}
    root = ss._skills_dir().resolve()
    for skill_md in ss._iter_active_skill_mds():
        with suppress(OSError, ValueError):
            skill_md.parent.resolve().relative_to(root)
            index.setdefault(skill_md.parent.name, []).append(skill_md.parent)
    return index


def _relocated_dest(src_name: str, index: Dict[str, List[Path]]) -> Optional[Tuple[Path, str]]:
    """``(dest, install_path)`` of a UNIQUE same-directory-name match, else None: the active
    tree may hold a skill under a DIFFERENT category than the repo (upstream reorganized,
    installed copy kept its place); ambiguity gives no basis to pick."""
    if len(candidates := index.get(src_name, [])) != 1:
        return None
    try:
        return candidates[0], _safe_rel_install_path(candidates[0], _ss()._skills_dir())
    except ValueError as e:
        logger.debug("Skipping relocated optional skill %s: %s", candidates[0], e)
        return None


def _official_lock_entry(dest: Path, install_path: str, official_path: str, origin: str) -> dict:
    """Hub-lock entry giving the copy at ``install_path`` official provenance for the optional
    source at ``official_path`` (the backfill and the moved-skill hand-over write this shape)."""
    timestamp = datetime.now(timezone.utc).isoformat()
    return {
        "source": "official", "identifier": f"official/{official_path}",
        "trust_level": "builtin", "scan_verdict": "backfilled",
        "content_hash": _content_hash(dest), "install_path": install_path,
        "files": _skill_file_list(dest), "metadata": {"backfilled_from": origin},
        "installed_at": timestamp, "updated_at": timestamp}


def _write_hub_lock(data: dict) -> None:
    """Atomic: a mid-write crash must not wipe provenance (the reader resets bad JSON)."""
    atomic_write_text(_ss()._skills_dir() / ".hub" / "lock.json", tmp_prefix=".lock_",
                      content=json.dumps(data, indent=2, ensure_ascii=False) + "\n")


def _backfill_optional_provenance(quiet: bool = False, skip: Optional[Set[str]] = None) -> List[str]:
    """Mark already-present official optional skills as hub-installed: formerly bundled (or
    hand-copied) skills now under optional-skills/ get official provenance when byte-identical
    to the source; modified/local skills are left alone. ``skip``: moved built-ins whose
    hand-over this same sync kept, refused or could not record (never adopted here)."""
    ss = _ss()
    optional_dir = ss._get_optional_dir()
    if not optional_dir.exists():
        return []
    data = _load_hub_lock()
    if data is None:
        data = {"version": 1, "installed": {}}
    installed = data.setdefault("installed", {})
    existing_paths = {entry.get("install_path") for entry in _hub_lock_entries(data)}
    backfilled: List[str] = []
    installed_dir_index: Optional[Dict[str, List[Path]]] = None
    index = _optional_skill_index() if skip else {}
    skip_folders = set(skip or ()) | {index[name][0] for name in skip or () if name in index}
    for _skill_md, src, install_path in _iter_optional_skills(optional_dir, root_relative=False):
        lock_name = src.name
        if lock_name in skip_folders or lock_name in installed or install_path in existing_paths:
            continue
        dest = ss._skills_dir() / Path(*install_path.split("/"))
        if not dest.is_dir():
            if installed_dir_index is None:
                installed_dir_index = _index_installed_skill_dirs_by_name()
            if (found := _relocated_dest(src.name, installed_dir_index)) is None:
                continue
            dest, install_path = found  # still requires a byte-identical hash below
        if install_path in existing_paths or ss._dir_hash(dest) != ss._dir_hash(src):
            continue
        installed[lock_name] = _official_lock_entry(dest, install_path, install_path, "optional-skills")
        existing_paths.add(install_path)
        backfilled.append(lock_name)
        if not quiet:
            print(f"  = {lock_name} (official optional provenance backfilled)")
    if backfilled:
        _write_hub_lock(data)
    return backfilled


def _hand_over_moved_skill(name: str, origin_hash: str) -> str:
    """Hand a built-in that moved to the optional catalog over to it, before its manifest line is
    cleaned. The role's copy (unique folder-name match, any category) is replaced in place with the
    official source only while it still matches ``origin_hash``, the line's fingerprint (equality
    with the release proves nothing), and reads whole; it then gets an official hub-lock entry at
    its real path, written before the old copy is dropped. Nothing is deleted or created. Returns
    ``moved_to_official``, ``kept_changed`` (changed, or not readable whole), ``skipped_deleted``,
    ``refused_ambiguous`` (more than one folder matches: neither is touched; the caller keeps the
    line), ``hub_owned``, ``removed_upstream``, or ``failed`` (old copy and lock kept: the caller
    keeps the line so the next sync retries)."""
    ss = _ss()
    if (official := _optional_skill_index().get(name)) is None:
        return "removed_upstream"
    folder, official_path, src = official
    data = _load_hub_lock() or {"version": 1, "installed": {}}
    installed = data.setdefault("installed", {})
    if name in installed or folder in installed:
        return "hub_owned"
    copies = _index_installed_skill_dirs_by_name()
    if not copies.get(folder):
        return "skipped_deleted"
    if (found := _relocated_dest(folder, copies)) is None:
        return "refused_ambiguous"
    dest, install_path = found
    if install_path in {str(e.get("install_path", "")).strip("/") for e in _hub_lock_entries(data)}:
        return "hub_owned"
    if not ss._matches_origin_hash(dest, origin_hash):
        return "kept_changed"
    try:
        ss._read_whole_copy(dest)
    except OSError:
        logger.warning("Keeping %s: part of it cannot be read", dest, exc_info=True)
        return "kept_changed"

    def record() -> None:
        installed[folder] = _official_lock_entry(dest, install_path, official_path, "bundled")
        _write_hub_lock(data)
    try:
        ss._replace_skill_dir(src, dest, record=record)
    except (OSError, ValueError):  # the old copy is back and the lock file is as it was
        installed.pop(folder, None)
        logger.warning("Could not hand %s over to the official optional catalog", dest, exc_info=True)
        return "failed"
    return "moved_to_official"


def _lock_install_dir(install_path: object) -> Path:
    """The folder a hub-lock ``install_path`` names. ValueError (the refusal reason) unless it is
    a relative skill-folder path strictly inside the skills tree that runs through no symlink."""
    root = _ss()._skills_dir()
    try:
        parts = _safe_rel_install_path(root / str(install_path or ""), root).split("/")
    except ValueError:
        raise ValueError(f"unsafe install path {install_path!r}") from None
    if is_excluded_skill_path(Path(*parts, "SKILL.md"), root=root):  # .hub, .archive, support dirs...
        raise ValueError(f"install path {install_path!r} is not a skill folder")
    if any(root.joinpath(*parts[:depth]).is_symlink() for depth in range(1, len(parts) + 1)):
        raise ValueError(f"install path {install_path!r} runs through a symlink")
    dest = root.joinpath(*parts)
    if root.resolve() not in dest.resolve().parents:
        raise ValueError(f"install path {install_path!r} leaves the skills tree")
    return dest


def refresh_official_optional_skills(dry_run: bool = False) -> dict:
    """Release step: each official optional copy (hub-lock source ``official``) that still matches
    its recorded ``content_hash`` gets the current official source through the safe swap, and its
    entry's ``content_hash``, ``files`` and ``updated_at`` move with it (the lock is written before
    the old copy is dropped). Changed copies (even one equal to a release), deleted and
    removed-upstream copies, and official names on disk with no record, are only reported; an
    unsafe path, a copy that cannot be read whole, or a failed copy or lock write changes nothing
    and is refused with the reason, as is a moved built-in matching more than one folder
    (``ambiguous``). ``dry_run`` writes nothing."""
    from tools.skills_guard import content_hash  # no fallback here: an unreadable copy is refused
    ss = _ss()
    report: Dict[str, list] = {"updated": [], "kept_changed": [], "kept_untracked": [],
                               "skipped_deleted": [], "skipped_removed_upstream": [], "refused": []}
    data = _load_hub_lock() or {}
    installed = data.get("installed") or {}
    index = _optional_skill_index()
    for name, entry in sorted(installed.items()):
        if not isinstance(entry, dict) or entry.get("source") != "official":
            continue
        try:
            dest = _lock_install_dir(entry.get("install_path"))
        except (OSError, ValueError) as e:
            report["refused"].append({"name": name, "reason": str(e)})
            continue
        if not dest.is_dir():  # deleted copies are never recreated
            report["skipped_deleted"].append(name)
            continue
        if (official := index.get(name) or index.get(dest.name)) is None:
            report["skipped_removed_upstream"].append(name)
            continue
        if not (recorded := entry.get("content_hash")):
            report["refused"].append({"name": name, "reason": "no recorded content hash"})
            continue
        try:
            ss._read_whole_copy(dest)  # a folder that cannot be listed would leave the hash partial
            current, newest = content_hash(dest), content_hash(official[2])
        except OSError as e:
            report["refused"].append({"name": name, "reason": f"unreadable copy or source: {e}"})
            continue
        if current != recorded:  # changed since Hermes wrote it, even if it now equals a release
            report["kept_changed"].append(
                {"name": name, "newer_official": newest != recorded, "official_hash": newest})
            continue
        if newest == recorded:
            continue
        if not dry_run:
            prior = dict(entry)

            def record() -> None:
                entry.update(content_hash=_content_hash(dest), files=_skill_file_list(dest),
                             updated_at=datetime.now(timezone.utc).isoformat())
                _write_hub_lock(data)
            try:
                ss._replace_skill_dir(official[2], dest, record=record)
            except (OSError, ValueError) as e:  # the old copy is back and the lock file is as it was
                entry.clear()
                entry.update(prior)
                logger.warning("Could not refresh official optional skill %s", dest, exc_info=True)
                report["refused"].append({"name": name, "reason": f"copy or its record failed, old copy kept: {e}"})
                continue
        report["updated"].append(name)
    # Official names on disk with neither a lock entry nor a manifest line: kept, never touched.
    manifest = ss._read_manifest()
    tracked = set(installed) | set(manifest)
    lock_paths = {str(e.get("install_path", "")).strip("/") for e in _hub_lock_entries(data)}
    on_disk, root = _index_installed_skill_dirs_by_name(), ss._skills_dir()
    names_by_folder: Dict[str, Set[str]] = {}
    for key, (folder, _install_path, _src) in index.items():
        names_by_folder.setdefault(folder, {folder}).add(key)
    report["kept_untracked"] = sorted(
        folder for folder, names in names_by_folder.items() if names.isdisjoint(tracked) and any(
            path.relative_to(root).as_posix() not in lock_paths for path in on_disk.get(folder, [])))
    # A built-in moved here whose name matches more than one folder keeps its manifest line and
    # is never handed over: refused, so the caller can say why it was not delivered.
    if (bundled_dir := ss._get_bundled_dir()).exists():
        built_in = {skill_name for skill_name, _ in ss._discover_bundled_skills(bundled_dir)}
        for name in sorted(set(manifest) - built_in):
            folder = (index.get(name) or (None,))[0]
            if folder and not {name, folder} & set(installed) and len(on_disk.get(folder, [])) > 1:
                report["refused"].append({"name": name, "reason": "ambiguous"})
    return report
