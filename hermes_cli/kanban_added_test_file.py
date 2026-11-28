"""Check one test file that an owner answer may add to a card; reads the store and git, writes nothing."""

from __future__ import annotations

import dataclasses
import fnmatch
import os
import re
import sqlite3
import subprocess
from typing import Callable, Optional

from hermes_cli import kanban_db as kb

OWNED_PATH_ADDED_EVENT_KIND = "owned_path_added"

SENTENCES = {
    "unknown_card": (
        "This card cannot be found, so no check can be added to it and the work needs a replacement as before."
    ),
    "no_write_scope": (
        "This card may not change any file, so no check can be added to it and the work needs a replacement as before."
    ),
    "whole_repository": "This card already covers the whole project, so no single check needs to be added.",
    "not_canonical": (
        "This check is not named as one plain file of this project, so the work needs a replacement as before."
    ),
    "not_test_file": "This check is not a test of this project, so the work needs a replacement as before.",
    "already_owned": "This card already owns this check, so nothing needs to be added.",
    "second_addition": "This card has already had one check added, so the work needs a replacement as before.",
    "held_by_open_card": (
        "Another open card of this project already owns this check, so the work needs a replacement as before."
    ),
    "not_at_base": (
        "This check is not a plain file of this project as the card began, so the work needs a replacement as before."
    ),
    "ok": "This check is a plain test file of this project as the card began, so it can be added to this card.",
}

_OWNER_PAYLOADS_DIR = "tests/contracts/owner_payloads/"
_PLAIN_FILE_MODE = "100644"
_GIT_TIMEOUT_SECONDS = 30

_QUOTED_NAME = re.compile(rb'"((?:[^\\"]|\\(?:[0-3][0-7]{2}|[abtnvfr"\\]))*)"')
_ESCAPE = re.compile(rb"\\(?:([0-3][0-7]{2})|(.))", re.DOTALL)
_SIMPLE_ESCAPES = {
    b"a": b"\a", b"b": b"\b", b"t": b"\t", b"n": b"\n", b"v": b"\v", b"f": b"\f", b"r": b"\r",
    b'"': b'"', b"\\": b"\\",
}


@dataclasses.dataclass(frozen=True)
class AddedTestFileCheck:
    ok: bool
    code: str
    blob: Optional[str] = None


def _unescape(match: re.Match[bytes]) -> bytes:
    octal, simple = match.groups()
    return bytes([int(octal, 8)]) if octal else _SIMPLE_ESCAPES[simple]


def _unquoted_name(name: str) -> Optional[str]:
    """The path as it is stored, undoing the C-style quoting git puts on an odd name."""
    raw = name.encode("utf-8", "surrogateescape")
    if not raw.startswith(b'"'):
        return name
    quoted = _QUOTED_NAME.fullmatch(raw)
    if quoted is None:
        return None
    return _ESCAPE.sub(_unescape, quoted.group(1)).decode("utf-8", "surrogateescape")


def _run_git(worktree: str, env: dict[str, str], *args: str) -> subprocess.CompletedProcess[bytes]:
    """Both steps of the read run through here so that they share one environment and one set of limits."""
    return subprocess.run(
        ["git", "-C", worktree, *args],
        capture_output=True,
        env=env,
        timeout=_GIT_TIMEOUT_SECONDS,
    )


def _git_read_base_entry(worktree: str, base_commit: str, path: str) -> Optional[tuple[str, str]]:
    """``(mode, blob id)`` when git lists exactly this one path as a file at the base, else None."""
    if not worktree:
        return None
    # A GIT_DIR or the like in the worker's environment would answer from another repository.
    env = {name: value for name, value in os.environ.items() if not name.startswith("GIT_")}
    # No replacement ref may stand in for the base; no Trace2 target in a git config may write a file.
    env.update(
        GIT_NO_REPLACE_OBJECTS="1",
        GIT_TRACE2="0",
        GIT_TRACE2_EVENT="0",
        GIT_TRACE2_PERF="0",
    )
    try:
        # Only a folder that git itself names as the checkout root is the card's checkout. Compare bytes with one
        # line feed cut, because a folder name may hold CR or end in a line feed.
        top = _run_git(worktree, env, "rev-parse", "--show-toplevel")
        if (
            top.returncode != 0
            or not top.stdout.endswith(b"\n")
            or top.stdout[:-1] != os.path.realpath(os.fsencode(worktree))
        ):
            return None
        done = _run_git(worktree, env, "ls-tree", base_commit, "--", path)
    except (OSError, subprocess.SubprocessError, ValueError):
        return None
    if done.returncode != 0:
        return None
    lines = done.stdout.decode("utf-8", "surrogateescape").removesuffix("\n").split("\n")
    if len(lines) != 1:
        return None
    meta, tab, name = lines[0].partition("\t")
    fields = meta.split(" ")
    if not tab or len(fields) != 3 or not all(fields):
        return None
    mode, kind, blob = fields
    if kind != "blob" or _unquoted_name(name) != path:
        return None
    return mode, blob


def _is_test_file(path: str) -> bool:
    name = path.rsplit("/", 1)[-1]
    # conftest.py and __init__.py match neither name pattern.
    if path.startswith("tests/") and (
        fnmatch.fnmatchcase(name, "test_*.py") or fnmatch.fnmatchcase(name, "*_test.py")
    ):
        return True
    return path.startswith(_OWNER_PAYLOADS_DIR) and path.endswith(".json")


def _is_canonical(path: str) -> bool:
    try:
        return kb.normalize_owned_paths([path]) == [path]
    except ValueError:
        return False


def _held_by_open_card(conn: sqlite3.Connection, task_id: str, project_id: str, path: str) -> bool:
    rows = conn.execute(
        "SELECT owned_paths FROM tasks"
        " WHERE project_id = ? AND id != ? AND status NOT IN ('done', 'archived')",
        (project_id, task_id),
    ).fetchall()
    return any(
        kb._owned_path_scopes_overlap(kb._decode_owned_paths(raw), [path]) for (raw,) in rows
    )


def _refused(code: str) -> AddedTestFileCheck:
    return AddedTestFileCheck(False, code)


def check_added_test_file(
    conn: sqlite3.Connection,
    task_id: str,
    path: str,
    read_base_entry: Optional[Callable[[str, str, str], Optional[tuple[str, str]]]] = None,
) -> AddedTestFileCheck:
    """Run the nine checks in order; the first that fails gives its code, else ``ok`` and the blob id."""
    row = conn.execute(
        "SELECT owned_paths, project_id, base_commit, workspace_path FROM tasks WHERE id = ?",
        (task_id,),
    ).fetchone()
    if row is None:
        return _refused("unknown_card")
    raw_owned, project_id, base_commit, worktree = row
    owned = kb._decode_owned_paths(raw_owned)
    if owned == []:
        return _refused("no_write_scope")
    if owned is None or "." in owned:
        return _refused("whole_repository")
    if not _is_canonical(path):
        return _refused("not_canonical")
    if not _is_test_file(path):
        return _refused("not_test_file")
    if kb._path_is_owned(path, owned):
        return _refused("already_owned")
    if len(owned) >= kb._MAX_OWNED_PATHS or conn.execute(
        "SELECT 1 FROM task_events WHERE task_id = ? AND kind = ? LIMIT 1",
        (task_id, OWNED_PATH_ADDED_EVENT_KIND),
    ).fetchone():
        return _refused("second_addition")
    if project_id and _held_by_open_card(conn, task_id, project_id, path):
        return _refused("held_by_open_card")
    if not base_commit or not worktree:
        return _refused("not_at_base")
    if read_base_entry is None:
        read_base_entry = _git_read_base_entry
    entry = read_base_entry(worktree, base_commit, path)
    if entry is None or entry[0] != _PLAIN_FILE_MODE:
        return _refused("not_at_base")
    return AddedTestFileCheck(True, "ok", entry[1])
