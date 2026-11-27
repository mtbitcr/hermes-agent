"""Role skills: a pass after the general pass of ``sync_skills``, with its own record.

A role is a home whose parent folder is named ``profiles``, named as the home folder; any other home gets
nothing. Its skills are the skill folders under ``role-skills/<role>/`` of the released checkout, beside
``tools/`` (``HERMES_BUNDLED_SKILLS`` moves the general catalogue only), each at the same inner path under
the role's ``skills/``. The marker ``.no-bundled-skills`` does not stop them.

``sync_skills`` hands its finished result to ``sync_role_skills`` at both of its returns, so the general
pass is the main branch's and reads nothing of this one. This pass keeps its lines in
``.role_skills_manifest`` beside ``.bundled_manifest`` (the same ``name:hash`` lines, written atomically,
made with its first line), reads ``.bundled_manifest`` only to refuse a name, and writes only its role
skills' folders and its record. Each refusal is decided before the first write: the whole role, under its
name, with ``link`` (a linked home or ``skills/``, a record file that is no regular file or cannot be
read); a role skill with ``link``, ``nested``, ``name`` (a general skill of the checkout, a line of
``.bundled_manifest`` or an external skills folder has its name) or ``overlap`` (its place and a general
skill's place are one, or one holds the other). Only symbolic links are read: a hard link or a mount is
not seen. Each other role skill goes by the table of the facts sheet, against its line in the role record.
"""

import json
import logging
import os
import stat
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Set, Tuple

from agent.skill_utils import ESSENTIAL_SKILLS, is_excluded_skill_path
from utils import atomic_write_text

if TYPE_CHECKING:
    from tools.skills_sync import _SyncState

logger = logging.getLogger(__name__)

ROLE_SKILLS_DIR = "role-skills"
ROLE_RECORD = ".role_skills_manifest"  # the role pass's own record, beside .bundled_manifest
HUB_LOCK_VERSION = 1  # ``HubLockFile.EMPTY["version"]`` (tools/skills_hub.py, which loads the network stack)
REFUSALS = {  # reason -> why nothing was written for the role skill
    "link": "a link is on the way to it or inside it",
    "nested": "its folder holds a second SKILL.md",
    "name": "a general skill, a line of .bundled_manifest or an external skills folder has this name",
    "overlap": "its place and a general skill's place are one, or one holds the other",
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


def _holds(one: Path, other: Path) -> bool:
    """The two places are one, or one holds the other."""
    return one == other or one in other.parents or other in one.parents

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


def _record_file() -> Path:
    """The role pass's own record file, beside ``.bundled_manifest`` (own-record point 6)."""
    return _ss()._skills_dir() / ROLE_RECORD


def _read_record(path: Path) -> Dict[str, str]:
    """``{name: hash}`` of a record file in ``_read_manifest``'s line form (``name:hash``, stripped, blank
    lines skipped); empty when it is not there. Unlike ``_read_manifest``, a read failure raises."""
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (FileNotFoundError, NotADirectoryError):
        return {}
    pairs = (line.partition(":") for line in map(str.strip, lines) if line)
    return {name.strip(): hash_val.strip() for name, _, hash_val in pairs}


def _write_role_record(entries: Dict[str, str]) -> None:
    """Atomic write of the role record, as ``_write_manifest_or_raise`` writes the general one; raises on
    failure."""
    atomic_write_text(_record_file(), "".join(f"{n}:{h}\n" for n, h in sorted(entries.items())),
                      tmp_prefix=".role_skills_manifest_", preserve_mode=True)


def _read_home() -> Optional[Tuple[Dict[str, str], Dict[str, str]]]:
    """The general record and the role record, each read once; None when the home or its ``skills/`` is a
    link, or a record file is there and is no regular file (it is not opened then) or cannot be read."""
    ss = _ss()
    if os.path.islink(ss._hermes_home()) or os.path.islink(ss._skills_dir()):
        return None
    records = []
    for path in (ss._manifest_file(), _record_file()):
        try:
            if not stat.S_ISREG(os.lstat(path).st_mode):
                return None
            records.append(_read_record(path))
        except (FileNotFoundError, NotADirectoryError):
            records.append({})
        except (OSError, ValueError):
            return None
    return records[0], records[1]


def _update_role_skill(st: "_SyncState", name: str, src: Path, place: Path, released: str) -> None:
    """Table line e: ``_update_existing_skill`` with the role record as the record, and one guard: with
    an entry at the backup place the copy is kept, so the pass removes no path but the copy it replaces."""
    ss = _ss()
    origin = st.manifest.get(name, "")
    reason = ss._copy_change(place, origin)
    if origin and released == origin:  # the release is unchanged: nothing to deliver
        if reason is not None:
            st.kept.append({"name": name, "reason": reason})
        return
    if reason is not None:
        st.user_modified.append(name)
        st.kept.append({"name": name, "reason": reason})
        st.say(f"  ~ {name} (kept as yours: {reason})")
        return
    if os.path.lexists(place.with_suffix(".bak")):  # never remove an entry the pass did not write
        st.kept.append({"name": name, "reason": "backup present"})
        st.say(f"  ~ {name} (kept as it is: backup present)")
        return
    failed = "copy write failed"

    def record() -> None:
        nonlocal failed
        failed = "record write failed"
        _write_role_record({**st.manifest, name: released})
    try:
        ss._replace_skill_dir(src, place, record)
    except Exception as e:
        st.refused.append({"name": name, "reason": failed})
        st.say(f"  ! Failed to update {name} ({failed}; old copy kept): {e}")
        return
    st.manifest[name] = released
    st.updated.append(name)
    st.say(f"  ↑ {name} (updated)")


def sync_role_skills(result: dict, quiet: bool = False) -> dict:
    """The role pass, after the general pass ended with ``result``: ``result`` with the role skills'
    entries after the general ones in ``copied``, ``updated``, ``kept``, ``refused``, ``suppressed`` and
    ``user_modified``; no other key and no count changes, and an adopted copy is in no list."""
    ss = _ss()
    role = _role_name(ss._hermes_home())
    if role is None:
        return result
    role_dir, entries = _role_skills(role)
    if not entries:
        return result
    records = _read_home()
    if records is None:
        if not quiet:
            print(f"  ! {role}: no role skill was delivered: this home, its skills/ folder or a record file "
                  f"is a link, no regular file or cannot be read")
        return {**result, "refused": [*result["refused"], {"name": role, "reason": "link"}]}
    general_lines, role_lines = records
    bundled_dir = ss._get_bundled_dir()
    catalogue = ss._discover_bundled_skills(bundled_dir)  # empty when it is not there
    general_names = {name for name, _ in catalogue}
    general_places = [ss._compute_relative_dest(src, bundled_dir) for _, src in catalogue]
    external_index = ss._build_external_skill_index()
    suppressed = ss._read_suppressed_names()
    st = ss._SyncState(manifest=dict(role_lines), quiet=quiet)

    plain = []  # every refusal is decided before the first write (own-record point 10)
    for name, src, refusal in entries:
        place = None if refusal else ss._skills_dir() / src.relative_to(role_dir)
        if refusal:
            pass
        elif name in general_names or name in general_lines or name in external_index:
            refusal = "name"
        elif any(_holds(place, other) for other in general_places):
            refusal = "overlap"
        elif not os.path.lexists(place) and ss._copy_change(place, None) == "link":
            refusal = "link"
        if refusal:
            st.refused.append({"name": name, "reason": refusal})
            st.say(f"  ! {name} (role skill) was not delivered: {REFUSALS[refusal]}")
        else:
            plain.append((name, src, place))

    recorded = False  # a line of table line a or c, written once below
    for name, src, place in plain:
        if name in suppressed and name not in ESSENTIAL_SKILLS:
            st.suppressed.append(name)
            continue
        released = ss._dir_hash(src)
        if not os.path.lexists(place):
            if name in st.manifest:  # b: a person deleted the copy, and it stays deleted
                continue
            try:  # a
                ss._copy_dir(src, place)
            except OSError as e:
                st.say(f"  ! Failed to copy {name}: {e}")  # no line: the next sync retries
                continue
            st.copied.append(name)
            st.manifest[name] = released
            recorded = True
            st.say(f"  + {name}")
        elif name in st.manifest:  # e
            _update_role_skill(st, name, src, place, released)
        elif ss._copy_change(place, released) is None:  # c: an exact copy
            owned = _hub_lock_paths()
            if owned is None:
                st.kept.append({"name": name, "reason": "hub_lock"})
                st.say(f"  ⚠ {name}: this role's skill is already in place, but the hub lock cannot be "
                       f"read — the copy was kept and not recorded.")
            elif any(_holds(Path(ss._rel_skills_posix(place)), Path(path)) for path in owned):
                st.kept.append({"name": name, "reason": ss._copy_change(place, None)})
                st.say(f"  ⚠ {name}: this role's skill is already in place, but a hub install owns it "
                       f"— the copy was kept and not recorded.")
            else:  # adopted: its line, and no byte of the copy
                st.manifest[name] = released
                recorded = True
        else:  # d
            st.kept.append({"name": name, "reason": ss._copy_change(place, None)})
            st.say(f"  ⚠ {name}: this role's skill shipped but you already have a local skill by this "
                   f"name — yours was kept.")
    if recorded:
        try:
            _write_role_record(st.manifest)
        except Exception as e:  # as _store_manifest: logged, not raised
            logger.debug("Failed to write the role skills record %s: %s", _record_file(), e, exc_info=True)
    return {**result, **{key: [*result[key], *getattr(st, key)]
                         for key in ("copied", "updated", "kept", "refused", "suppressed", "user_modified")}}
