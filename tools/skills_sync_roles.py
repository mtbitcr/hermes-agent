"""Role skills: one pass of ``sync_skills`` that gives each role the skills of its own folder.

A role is a home whose parent folder is named ``profiles``; its name is the home folder's name. Its
skills are the folders with a ``SKILL.md`` under ``role-skills/<role>/`` of the released checkout,
beside ``tools/`` (``HERMES_BUNDLED_SKILLS`` moves the general catalogue only), each copied to the
same inner path under the role's ``skills/``. The pass runs also when the general catalogue is not
there. Any other home, the root home included, gets none. The marker ``.no-bundled-skills`` stops
general skills, not these, and their record lines are released names, so the cleanup keeps them.

The source is walked with no link followed, each entry's type checked before a text is read. Before
a write, a role whose home, ``skills/`` or record file is a link (or a record file that is no regular
file) gets nothing and one refusal under its name, and a role skill outside the plain shape is refused
with a reason: ``link`` (a link on the way from ``role-skills/``, refused under its own name, a link
inside the skill's folder, a ``SKILL.md`` that is no regular file, refused unread under its folder's
name, or the role's folder itself as its folder), ``nested`` (its folder holds a second ``SKILL.md``,
or is inside a folder that does), ``name`` (a skill under ``skills/`` has its name) or ``overlap``
(its place is a general skill's place or inside one; or its place or backup place and a place the
general pass writes are one, or one holds the other: the whole catalogue's places when the pass
would write bytes into the role, else the places the general pass writes in this run). Only symbolic
links are read: a hard link or a mount is not seen.

A role skill in the plain shape goes as a general skill goes in a role with no marker, with two
differences: a record line with no folder at the place is a deleted copy (nothing is written, no
other folder is searched); an exact copy with no record line is adopted (its line is written, and no
byte of it) when the hub lock is not there, or is a regular file of the hub's valid form that owns no
install at, above or inside its place. Any other lock keeps the copy with no line. The record line of
a role skill that a general skill shares the name of is taken out before the general loop, so it is
no evidence of a renamed or deleted general skill, and put back after it unless a general line took
its place.
"""

import json
import os
import stat
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Set, Tuple

from agent.skill_utils import ESSENTIAL_SKILLS, is_excluded_skill_path

if TYPE_CHECKING:
    from tools.skills_sync import _SyncState

ROLE_SKILLS_DIR = "role-skills"
HUB_LOCK_VERSION = 1  # ``HubLockFile.EMPTY["version"]`` (tools/skills_hub.py, which loads the network stack)
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


def _role_skills(role: str) -> Tuple[Path, List[Tuple[str, Path, Optional[str]]]]:
    """The role's folder under ``role-skills/`` and its ``(name, folder, refusal)`` entries in path
    order, ``refusal`` ``link``, ``nested`` or None (addition point 6 a to d). No link is followed and
    each entry's type is checked before a text is read: ``role-skills/`` or the role's folder as a link
    is the one entry, under its own name, and so is each link on the way below it; a ``SKILL.md`` that
    is no regular file is not read, and its skill has its folder's name. With no link, the skills are
    those ``_discover_bundled_skills`` finds."""
    role_dir = role_skills_dir() / role
    for way in (role_dir.parent, role_dir):
        if os.path.islink(way):
            return role_dir, [(way.name, way, "link")]
    links: List[Path] = []
    mds: Dict[Path, int] = {}  # folder -> the mode of its SKILL.md entry, no link followed
    for top, dirs, files in os.walk(role_dir):  # a linked folder is listed, not entered
        for entry in dirs + files:
            try:
                mode = os.lstat(os.path.join(top, entry)).st_mode
            except OSError:
                mode = 0  # gone: neither a link nor a regular file
            if stat.S_ISLNK(mode):
                links.append(Path(top, entry))
            if entry == "SKILL.md":
                mds[Path(top)] = mode
    skills = sorted(folder for folder in mds
                    if not is_excluded_skill_path((folder / "SKILL.md").relative_to(role_dir), root=role_dir))
    found = []
    for folder in skills:
        regular = stat.S_ISREG(mds[folder])
        name = _ss()._read_skill_name(folder / "SKILL.md", folder.name) if regular else folder.name
        if folder == role_dir or not regular or any(folder in link.parents for link in links):
            found.append((name, folder, "link"))
        elif any(other in folder.parents for other in skills) or any(folder in other.parents for other in mds):
            found.append((name, folder, "nested"))
        else:
            found.append((name, folder, None))
    found += [(link.name, link, "link") for link in links  # a link on the way, outside every skill
              if not any(folder in link.parents for folder in skills)
              and not is_excluded_skill_path(link.relative_to(role_dir) / "SKILL.md", root=role_dir)]
    return role_dir, sorted(found, key=lambda item: item[1])


def _plain_home() -> bool:
    """The home, its ``skills/`` and its record file are no links, and the record file, when it is
    there, is a regular file: the role pass reads and writes nowhere else (addition point 8)."""
    ss = _ss()
    if os.path.islink(ss._hermes_home()) or os.path.islink(ss._skills_dir()):
        return False
    try:
        return stat.S_ISREG(os.lstat(ss._manifest_file()).st_mode)
    except (FileNotFoundError, NotADirectoryError):
        return True
    except OSError:
        return False


def _general_places(bundled_dir: Path, all_bundled: List[Tuple[str, Path]]) -> List[Path]:
    """Each place the general pass may write, the whole catalogue whatever the marker: each general
    skill and its backup, each category description, the record file."""
    ss = _ss()
    skills = [ss._compute_relative_dest(src, bundled_dir) for _, src in all_bundled]
    return [*skills, *(place.with_suffix(".bak") for place in skills),
            *(ss._skills_dir() / md.relative_to(bundled_dir) for md in bundled_dir.rglob("DESCRIPTION.md")),
            ss._manifest_file()]


def _written_in_run(bundled_dir: Path, bundled_skills: List[Tuple[str, Path]], marked: bool) -> List[Path]:
    """Each place the general pass writes in this run, read before the role pass writes: each skill of
    its loop and its backup, each category description ``_seed_category_descriptions`` writes (under
    its own conditions: with the marker only the essential skills' categories, not already there, no
    link on the way), the record file."""
    ss = _ss()
    skills = [ss._compute_relative_dest(src, bundled_dir) for _, src in bundled_skills]
    only = {place.parent for (name, _), place in zip(bundled_skills, skills) if name in ESSENTIAL_SKILLS}
    descriptions = [place for place in (ss._skills_dir() / md.relative_to(bundled_dir)
                                        for md in bundled_dir.rglob("DESCRIPTION.md"))
                    if (not marked or place.parent in only) and not os.path.lexists(place)
                    and ss._copy_change(place.parent, None) != "link"]
    return [*skills, *(place.with_suffix(".bak") for place in skills), *descriptions, ss._manifest_file()]


def _holds(one: Path, other: Path) -> bool:
    """The two places are one, or one holds the other."""
    return one == other or one in other.parents or other in one.parents


def _overlaps(dest: Path, written: List[Path]) -> bool:
    """The role skill's place or backup place and a written place are one, or one holds the other."""
    return any(_holds(mine, other) for mine in (dest, dest.with_suffix(".bak")) for other in written)


def _hub_lock_paths() -> Optional[Set[str]]:
    """The ``skills/``-relative places of the hub lock's installs, in the hub's normal form: none when
    there is no lock file; None when ``.hub`` or the lock is a link, the lock is no regular file (it is
    not opened then), cannot be read, or is not the hub's valid form: its version, an ``installed``
    map, and each entry's ``install_path`` passing the hub's own validator for the entry's name. Such a
    lock proves nothing."""
    from tools.skills_hub_models import _normalize_lock_install_path  # a leaf; loaded only when asked
    hub = _ss()._skills_dir() / ".hub"
    try:
        if stat.S_ISLNK(os.lstat(hub).st_mode):
            return None
        regular = stat.S_ISREG(os.lstat(hub / "lock.json").st_mode)
    except (FileNotFoundError, NotADirectoryError):
        return set()
    except OSError:
        return None
    if not regular:
        return None
    try:
        lock = json.loads((hub / "lock.json").read_bytes().decode("utf-8"))
        if type(lock["version"]) is not int or lock["version"] != HUB_LOCK_VERSION:
            return None
        return {_normalize_lock_install_path(entry["install_path"], name) for name, entry in lock["installed"].items()}
    except (OSError, ValueError, TypeError, KeyError, AttributeError, RecursionError):
        return None


def _install_role_skill(st: "_SyncState", name: str, src: Path, dest: Path, released: str) -> None:
    """A role skill with no record line. With no folder at its place, ``_install_new_skill`` copies
    it (a link on the way and a failed copy as for any skill). An exact copy is adopted when the hub
    lock is not there, or is valid and owns no install at, above or inside its place: the line is
    written, no byte of the copy. Any other copy is kept and named."""
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
            if not any(_holds(Path(rel), Path(path)) for path in owned):
                st.manifest[name] = released
                return
        st.kept.append({"name": name, "reason": ss._copy_change(dest, None)})
        st.say(f"  ⚠ {name}: this role's skill shipped but you already have a local skill by this "
               f"name — yours was kept.")
    except OSError as e:  # as _install_new_skill: no line, the next sync retries
        st.say(f"  ! {name} (role skill) was not checked: {e}")


def take_role_records(st: "_SyncState", bundled_dir: Path, all_bundled: List[Tuple[str, Path]]) -> Dict[str, str]:
    """Before the general loop: the record lines of this role's skills whose name a general skill of
    the run has (the role pass refuses them with ``name``) and that are role lines, taken out of
    ``st.manifest`` so the loop reads none as a renamed or deleted general skill; ``sync_role_skills``
    puts them back. A line is a role line when it is the role skill's released hash and not the general
    skill's, or when the role skill's place, apart from the general skill's place, holds an exact copy
    for it and the general skill's place does not. Any other line is a general record and stays. Only
    role skills of the plain shape are hashed, in a home that passes ``_plain_home``."""
    ss = _ss()
    role = _role_name(ss._hermes_home())
    if role is None or not _plain_home():
        return {}
    role_dir, role_skills = _role_skills(role)
    taken: Dict[str, str] = {}
    for name, src, refusal in role_skills:
        line = st.manifest.get(name, "")
        generals = [(ss._compute_relative_dest(other, bundled_dir), other)
                    for each, other in all_bundled if each == name]
        if refusal is not None or not line or not generals or name in taken:
            continue
        place = ss._compute_relative_dest(src, role_dir)
        if ((line == ss._dir_hash(src) and all(ss._dir_hash(other) != line for _, other in generals))
                or (ss._copy_change(place, line) is None and all(
                    not _holds(place, dest) and ss._copy_change(dest, line) is not None for dest, _ in generals))):
            taken[name] = st.manifest.pop(name)
    return taken


def sync_role_skills(st: "_SyncState", bundled_dir: Path, all_bundled: List[Tuple[str, Path]],
                     suppressed: Set[str], external_index: Set[str], taken: Optional[Dict[str, str]] = None,
                     bundled_skills: Optional[List[Tuple[str, Path]]] = None, marked: bool = False) -> Set[str]:
    """Seed the role skills of the active home into ``st`` (``sync_skills`` stores the manifest).
    ``taken``: the lines ``take_role_records`` took, put back first unless a general line of this run
    took their place; ``bundled_skills`` and ``marked``: the general loop's skills and the marker.
    Returns the role skills' names, refused ones too, which the cleanup keeps; an empty set for a home
    that is no role."""
    for name, line in (taken or {}).items():
        st.manifest.setdefault(name, line)
    ss = _ss()
    role = _role_name(ss._hermes_home())
    if role is None:
        return set()
    role_dir, role_skills = _role_skills(role)
    names = {name for name, _, _ in role_skills}
    if not role_skills:
        return names
    if not _plain_home():  # maybe another role's
        st.refused.append({"name": role, "reason": "link"})
        st.say(f"  ! {role}: no role skill was delivered: this home, its skills/ folder or its record file "
               f"is a link or no regular file")
        return names
    general = {name for name, _ in all_bundled}
    whole = _general_places(bundled_dir, all_bundled)
    in_run = _written_in_run(bundled_dir, all_bundled if bundled_skills is None else bundled_skills, marked)
    general_skills = [ss._compute_relative_dest(src, bundled_dir) for _, src in all_bundled]
    for name, src, reason in role_skills:
        if reason is None and name in general:
            reason = "name"
        if reason is None:  # the table line, then the places that line must keep clear
            dest = ss._compute_relative_dest(src, role_dir)
            released = ss._dir_hash(src)
            line = st.manifest.get(name)
            writes = (line is None and not os.path.lexists(dest)) or (
                bool(line) and line != released and os.path.lexists(dest) and ss._copy_change(dest, line) is None)
            if (any(dest == place or place in dest.parents for place in general_skills)
                    or _overlaps(dest, whole if writes else in_run)):
                reason = "overlap"
        if reason is not None:  # before any write: no file, no line
            st.refused.append({"name": name, "reason": reason})
            st.say(f"  ! {name} (role skill) was not delivered: {REFUSALS[reason]}")
            continue
        if name in suppressed and name not in ESSENTIAL_SKILLS:
            st.suppressed.append(name)
            continue
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
    case, with the role pass. A home with no role skills gets ``result`` as it is, and no write; a home
    that fails ``_plain_home`` gets one refusal, and its record file is not read."""
    ss = _ss()
    role = _role_name(ss._hermes_home())
    if role is None or not _role_skills(role)[1]:
        return result
    st = ss._SyncState(manifest=ss._read_manifest() if _plain_home() else {}, quiet=quiet, reset=reset)
    recorded = dict(st.manifest)
    sync_role_skills(st, ss._get_bundled_dir(), [], ss._read_suppressed_names(), ss._build_external_skill_index())
    if st.manifest != recorded:
        ss._store_manifest(st.manifest)
    return {**result, "copied": st.copied, "updated": st.updated, "skipped": st.skipped,
            "user_modified": st.user_modified, "suppressed": st.suppressed, "kept": st.kept,
            "refused": st.refused}
