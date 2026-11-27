"""Role skills: one pass of ``sync_skills`` that gives each role the skills of its own folder.

A role is a home whose parent folder is named ``profiles``; its name is the home folder's name. Its
skills are the folders with a ``SKILL.md`` under ``role-skills/<role>/`` beside the repository's
``skills/``, each copied to the same inner path under the role's ``skills/``. Any other home, the
root home included, gets none. The marker ``.no-bundled-skills`` stops general skills, not these,
and their record lines are released names, so the cleanup keeps them. A role skill goes as a
general skill goes in a role with no marker, with three differences: a role skill with the name of
a skill under ``skills/`` is refused ("name"); a record line with no folder at the place is a
deleted copy (nothing is written, no other folder is searched); an exact copy with no record line
is adopted: its line is written, and no byte of it.
"""

import os
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional, Set, Tuple

from agent.skill_utils import ESSENTIAL_SKILLS

if TYPE_CHECKING:
    from tools.skills_sync import _SyncState

ROLE_SKILLS_DIR = "role-skills"


def _ss():
    """Live ``tools.skills_sync`` module (imported lazily: it imports this module)."""
    from tools import skills_sync
    return skills_sync


def _role_name(home: Path) -> Optional[str]:
    """The home folder's name when its parent folder is named ``profiles``, else None."""
    home = Path(os.path.normpath(home))  # ``profiles/..`` is the folder above, not a role
    return home.name if home.parent.name == "profiles" else None


def _install_role_skill(st: "_SyncState", name: str, src: Path, dest: Path, released: str) -> None:
    """A role skill with no record line. With no folder at its place, ``_install_new_skill`` copies
    it (a link on the way and a failed copy as for any skill). An exact copy that no hub install
    owns is adopted: the line is written, no byte of the copy. Any other copy is kept and named."""
    ss = _ss()
    if not os.path.lexists(dest):
        ss._install_new_skill(st, name, src, dest, released)
        return
    try:
        st.skipped += 1
        if (ss._copy_change(dest, released) is None
                and ss._rel_skills_posix(dest) not in ss._read_hub_install_paths()):
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
    Returns their names, which the cleanup keeps; an empty set for a home that is no role."""
    ss = _ss()
    role = _role_name(ss._hermes_home())
    if role is None:
        return set()
    role_dir = bundled_dir.parent / ROLE_SKILLS_DIR / role
    role_skills = ss._discover_bundled_skills(role_dir)
    general = {name for name, _ in all_bundled}
    for name, src in role_skills:
        if name in general:  # two skills of one name: the role skill is not delivered
            st.refused.append({"name": name, "reason": "name"})
            st.say(f"  ! {name} (role skill) was not delivered: a skill under skills/ has this name")
            continue
        if name in suppressed and name not in ESSENTIAL_SKILLS:
            st.suppressed.append(name)
            continue
        dest = ss._compute_relative_dest(src, role_dir)
        released = ss._dir_hash(src)
        if name in external_index:
            ss._defer_to_external(st, name, dest, released)
        elif name not in st.manifest:
            _install_role_skill(st, name, src, dest, released)
        elif os.path.lexists(dest):
            ss._update_existing_skill(st, name, src, dest, released)
        else:
            st.skipped += 1  # a line and no copy: a person deleted it, and it stays deleted
    return {name for name, _ in role_skills}
