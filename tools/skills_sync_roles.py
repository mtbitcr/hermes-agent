"""Role skills: one pass of ``sync_skills`` that gives each role the skills of its own folder.

A role is a home whose parent folder is named ``profiles``; its name is the home folder's name. Its
skills are the folders with a ``SKILL.md`` under ``role-skills/<role>/`` of the released checkout,
beside ``tools/`` (``HERMES_BUNDLED_SKILLS`` moves the general catalogue only), each copied to the
same inner path under the role's ``skills/``. The pass runs also when the general catalogue is not
there. Any other home, the root home included, gets none. The marker ``.no-bundled-skills`` stops
general skills, not these, and their record lines are released names, so the cleanup keeps them.

Before a write, a role whose home or ``skills/`` is a link gets nothing and one refusal under its
name, and a role skill outside the plain shape is refused with a reason: ``link`` (a link on the way
from ``role-skills/`` to its folder or inside it, a ``SKILL.md`` that is no regular file, or the
role's folder itself as its folder), ``nested`` (its folder holds a second ``SKILL.md``, or is inside
a folder that does), ``name`` (a skill under ``skills/`` has its name) or ``overlap`` (its place or
backup place and a place the general pass writes are one, or one holds the other). Only symbolic
links are read: a hard link or a mount is not seen.

A role skill in the plain shape goes as a general skill goes in a role with no marker, with two
differences: a record line with no folder at the place is a deleted copy (nothing is written, no
other folder is searched); an exact copy with no record line is adopted (its line is written, and no
byte of it) when the hub lock is not there, or is read, valid and owns no folder of its place. A
lock that cannot be read or is not valid keeps the copy with no line.
"""

import json
import os
import stat
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional, Set, Tuple

from agent.skill_utils import ESSENTIAL_SKILLS

if TYPE_CHECKING:
    from tools.skills_sync import _SyncState

ROLE_SKILLS_DIR = "role-skills"
REFUSALS = {  # reason -> why nothing was written for the role skill
    "link": "a link is on the way from role-skills/ to it or inside it",
    "nested": "its folder holds a second SKILL.md",
    "name": "a skill under skills/ has this name",
    "overlap": "the general skills write at its place",
}


def _ss():
    """Live ``tools.skills_sync`` module (imported lazily: it imports this module)."""
    from tools import skills_sync
    return skills_sync


def role_skills_dir() -> Path:
    """``role-skills/`` of the released checkout, beside the ``tools/`` that holds ``skills_sync.py``
    (tests replace this function)."""
    return Path(_ss().__file__).parent.parent / ROLE_SKILLS_DIR


def _role_name(home: Path) -> Optional[str]:
    """The home folder's name when its parent folder is named ``profiles``, else None."""
    home = Path(os.path.normpath(home))  # ``profiles/..`` is the folder above, not a role
    return home.name if home.parent.name == "profiles" else None


def _role_skills(role: str) -> Tuple[Path, List[Tuple[str, Path]]]:
    """The role's folder under ``role-skills/`` and its ``(name, folder)`` skills, in path order."""
    role_dir = role_skills_dir() / role
    return role_dir, sorted(_ss()._discover_bundled_skills(role_dir), key=lambda item: item[1])


def _shape_refusal(src: Path, role_dir: Path, in_skill: bool) -> Optional[str]:
    """``link`` or ``nested`` for a role skill folder outside the plain shape, else None. ``in_skill``:
    the folder is inside another role skill's folder."""
    rel = src.relative_to(role_dir)
    way = [role_skills_dir(), *(role_dir.joinpath(*rel.parts[:n]) for n in range(len(rel.parts) + 1))]
    if not rel.parts or any(os.path.islink(folder) for folder in way):
        return "link"
    try:
        regular = stat.S_ISREG(os.lstat(src / "SKILL.md").st_mode)
    except OSError:
        regular = False
    if not regular:
        return "link"
    nested = in_skill
    for folder, dirs, files in os.walk(src):  # a folder that cannot be listed fails the copy itself
        if any(os.path.islink(os.path.join(folder, entry)) for entry in dirs + files):
            return "link"
        nested = nested or (folder != os.fspath(src) and "SKILL.md" in dirs + files)
    return "nested" if nested else None


def _general_places(bundled_dir: Path, all_bundled: List[Tuple[str, Path]]) -> List[Path]:
    """Each place the general pass may write, the whole catalogue whatever the marker: each general
    skill and its backup, each category description, the record file."""
    ss = _ss()
    skills = [ss._compute_relative_dest(src, bundled_dir) for _, src in all_bundled]
    return [*skills, *(place.with_suffix(".bak") for place in skills),
            *(ss._skills_dir() / md.relative_to(bundled_dir) for md in bundled_dir.rglob("DESCRIPTION.md")),
            ss._manifest_file()]


def _overlaps(dest: Path, written: List[Path]) -> bool:
    """The role skill's place or backup place and a written place are one, or one holds the other."""
    return any(mine == other or mine in other.parents or other in mine.parents
               for mine in (dest, dest.with_suffix(".bak")) for other in written)


def _hub_lock_paths() -> Optional[Set[str]]:
    """The ``skills/``-relative places of the hub lock's installs: none when there is no lock file;
    None when it is there and cannot be read or has no valid form (it then proves nothing)."""
    lock = _ss()._skills_dir() / ".hub" / "lock.json"
    try:
        os.lstat(lock)
    except (FileNotFoundError, NotADirectoryError):
        return set()
    except OSError:
        return None
    try:
        installed = json.loads(lock.read_text(encoding="utf-8"))["installed"]
        paths = {entry["install_path"].strip("/") for entry in installed.values()}
    except (OSError, ValueError, TypeError, KeyError, AttributeError, RecursionError):
        return None
    return None if "" in paths else paths


def _install_role_skill(st: "_SyncState", name: str, src: Path, dest: Path, released: str) -> None:
    """A role skill with no record line. With no folder at its place, ``_install_new_skill`` copies
    it (a link on the way and a failed copy as for any skill). An exact copy is adopted when the hub
    lock is not there, or is valid and owns no folder of its place: the line is written, no byte of
    the copy. Any other copy is kept and named."""
    ss = _ss()
    if not os.path.lexists(dest):
        ss._install_new_skill(st, name, src, dest, released)
        return
    try:
        st.skipped += 1
        if ss._copy_change(dest, released) is None:
            owned = _hub_lock_paths()
            if owned is None:
                st.kept.append({"name": name, "reason": "hub_lock"})
                st.say(f"  ⚠ {name}: this role's skill is already in place, but the hub lock cannot be "
                       f"read — the copy was kept and not recorded.")
                return
            rel = ss._rel_skills_posix(dest)
            if not any(rel == path or rel.startswith(path + "/") for path in owned):
                st.manifest[name] = released
                return
        st.kept.append({"name": name, "reason": ss._copy_change(dest, None)})
        st.say(f"  ⚠ {name}: this role's skill shipped but you already have a local skill by this "
               f"name — yours was kept.")
    except OSError as e:  # as _install_new_skill: no line, the next sync retries
        st.say(f"  ! {name} (role skill) was not checked: {e}")


def sync_role_skills(st: "_SyncState", bundled_dir: Path, all_bundled: List[Tuple[str, Path]],
                     suppressed: Set[str], external_index: Set[str]) -> Set[str]:
    """Seed the role skills of the active home into ``st`` (``sync_skills`` stores the manifest).
    Returns their names, refused ones too, which the cleanup keeps; an empty set for a home that is
    no role."""
    ss = _ss()
    role = _role_name(ss._hermes_home())
    if role is None:
        return set()
    role_dir, role_skills = _role_skills(role)
    names = {name for name, _ in role_skills}
    if not role_skills:
        return names
    if os.path.islink(ss._hermes_home()) or os.path.islink(ss._skills_dir()):  # maybe another role's
        st.refused.append({"name": role, "reason": "link"})
        st.say(f"  ! {role}: no role skill was delivered: this home or its skills/ folder is a link")
        return names
    general = {name for name, _ in all_bundled}
    written = _general_places(bundled_dir, all_bundled)
    folders = [src for _, src in role_skills]
    for name, src in role_skills:
        dest = ss._compute_relative_dest(src, role_dir)
        reason = (_shape_refusal(src, role_dir, any(folder in src.parents for folder in folders))
                  or ("name" if name in general else None) or ("overlap" if _overlaps(dest, written) else None))
        if reason is not None:  # before any write: no file, no line
            st.refused.append({"name": name, "reason": reason})
            st.say(f"  ! {name} (role skill) was not delivered: {REFUSALS[reason]}")
            continue
        if name in suppressed and name not in ESSENTIAL_SKILLS:
            st.suppressed.append(name)
            continue
        released = ss._dir_hash(src)
        if name in external_index:
            ss._defer_to_external(st, name, dest, released)
        elif name not in st.manifest:
            _install_role_skill(st, name, src, dest, released)
        elif os.path.lexists(dest):
            ss._update_existing_skill(st, name, src, dest, released)
        else:
            st.skipped += 1  # a line and no copy: a person deleted it, and it stays deleted
    return names


def sync_role_skills_alone(quiet: bool, reset: Set[str], result: dict) -> dict:
    """``sync_skills`` when the general catalogue is not there: ``result``, today's result for that
    case, with the role pass. A home with no role skills gets ``result`` as it is, and no write."""
    ss = _ss()
    role = _role_name(ss._hermes_home())
    if role is None or not _role_skills(role)[1]:
        return result
    st = ss._SyncState(manifest=ss._read_manifest(), quiet=quiet, reset=reset)
    recorded = dict(st.manifest)
    sync_role_skills(st, ss._get_bundled_dir(), [], ss._read_suppressed_names(), ss._build_external_skill_index())
    if st.manifest != recorded:
        ss._store_manifest(st.manifest)
    return {**result, "copied": st.copied, "updated": st.updated, "skipped": st.skipped,
            "user_modified": st.user_modified, "suppressed": st.suppressed, "kept": st.kept,
            "refused": st.refused}
