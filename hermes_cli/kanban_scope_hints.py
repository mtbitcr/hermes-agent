"""Read-only search for the tests and readers of the paths a proposed card will own.

Owned paths cannot change after a card exists, so the proposer needs to see what pins
or reads them before the card is created. Given a repository root, the owned paths and
optionally the keys the card will change, ``find_hints`` lists the files outside the
owned paths that match one of four rules:

1. a Python file under ``tests/`` that imports a module in the owned paths;
2. any other Python file that imports a name from such a module;
3. a file that names an owned file path;
4. for each key, a file that contains that exact key in single or double quotes.

An owned folder owns every file in it, for every rule; ``.`` owns the whole repository
(nothing is listed). The search only reads. It never follows a path outside the repository
(a symlink that resolves outside is skipped; an owned path or root that resolves outside is
refused) and it does not look inside ``SKIP_DIRS``.

There is no size limit. A file with a NUL byte in its first 8 KiB is binary and is not
searched. A text file that cannot be read whole (an OS error, or bytes that are not UTF-8)
is never left out in silence: it is listed in ``Report.unreadable``. So is a folder that
cannot be listed (its files are not searched, and an owned one contributes no files to the
owned inventory): it is listed there with a trailing ``/``.
"""

from __future__ import annotations

import ast
import codecs
import mmap
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Sequence

SKIP_DIRS = frozenset({".git", "node_modules", ".venv", "venv", ".worktrees", "__pycache__"})
BINARY_PROBE_BYTES = 8192
READ_CHUNK_BYTES = 1024 * 1024
RULE_NAMES = {
    1: "test imports owned module",
    2: "imports name from owned module",
    3: "names owned path",
    4: "quoted key",
}


@dataclass(frozen=True)
class Hint:
    """One file to consider owning: every rule that matched it, and the keys for rule 4."""

    path: str
    rules: tuple[int, ...]
    keys: tuple[str, ...] = ()


@dataclass(frozen=True)
class Report:
    """The hints (files matching a key first, then by path), the files per rule, and the
    text files that could not be read whole and the folders that could not be listed (a path
    ending in ``/``), sorted by path."""

    hints: tuple[Hint, ...]
    counts: dict[int, int]
    unreadable: tuple[str, ...] = ()


def default_root(start: Path | str | None = None) -> Path:
    """The nearest folder holding ``.git``, going up from *start* (default: the current
    folder); *start* itself when there is none."""
    here = Path(start or Path.cwd()).resolve()
    return next((p for p in (here, *here.parents) if (p / ".git").exists()), here)


def _resolve_root(root: Path | str) -> Path:
    resolved = Path(root).resolve()
    if not resolved.is_dir():
        raise ValueError(f"root {str(root)!r} is not a directory")
    return resolved


def _normalize_owned(root: Path, owned: Iterable[str]) -> list[tuple[str, bool]]:
    """Each owned path as (path relative to *root* with ``/``, is a folder)."""
    result = []
    for raw in owned:
        full = Path(raw) if Path(raw).is_absolute() else root / raw
        resolved = full.resolve()
        try:
            rel = resolved.relative_to(root).as_posix()
        except ValueError:
            raise ValueError(f"owned path {raw!r} resolves outside the repository") from None
        result.append((rel, str(raw).endswith("/") or resolved.is_dir()))
    return result


def _module_name(rel: str) -> str | None:
    if not rel.endswith(".py"):
        return None
    parts = rel[:-3].split("/")
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts) or None


def _absolute_module(node: ast.ImportFrom, directory: list[str]) -> str | None:
    """The dotted module a ``from`` import names, relative imports resolved from *directory*."""
    if node.level == 0:
        return node.module
    keep = len(directory) - (node.level - 1)
    if keep < 0:
        return None
    return ".".join([*directory[:keep], *([node.module] if node.module else [])]) or None


def _walk_files(top: Path, root: Path, unlistable: set[str]) -> Iterator[Path]:
    """The files under *top*; each folder that cannot be listed is added to *unlistable* as
    its path relative to *root* with a trailing ``/`` (a folder that does not exist is not)."""

    def failed(error: OSError) -> None:
        if error.filename is not None and not isinstance(error, FileNotFoundError):
            unlistable.add(Path(error.filename).relative_to(root).as_posix() + "/")

    for dirpath, dirnames, filenames in os.walk(top, onerror=failed, followlinks=False):
        dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRS)
        for name in sorted(filenames):
            yield Path(dirpath, name)


def _examine(path: Path, want_text: bool, path_pattern: re.Pattern | None, key_needles: dict[str, tuple[bytes, bytes]]):
    """Read one file without loading it whole (unless *want_text*, for parsing).

    Returns ``(status, text, names_owned_path, matched_keys)``; *status* is ``"ok"``,
    ``"binary"`` (not searched) or ``"unreadable"`` (an OS error or bytes that are not UTF-8).
    """
    try:
        with path.open("rb") as handle:
            if b"\0" in handle.read(BINARY_PROBE_BYTES):
                return "binary", None, False, ()
            handle.seek(0)
            if want_text:
                text = handle.read().decode("utf-8")
            else:
                text = None
                decoder = codecs.getincrementaldecoder("utf-8")()
                while chunk := handle.read(READ_CHUNK_BYTES):
                    decoder.decode(chunk)
                decoder.decode(b"", final=True)
            if os.fstat(handle.fileno()).st_size == 0:
                return "ok", text, False, ()
            with mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ) as view:
                names_path = path_pattern is not None and path_pattern.search(view) is not None
                matched = tuple(k for k, pair in key_needles.items() if any(view.find(n) >= 0 for n in pair))
    except (OSError, ValueError):  # UnicodeDecodeError is a ValueError
        return "unreadable", None, False, ()
    return "ok", text, names_path, matched


def find_hints(root: Path | str, owned: Sequence[str], keys: Sequence[str] = ()) -> Report:
    """Search *root* for the files that pin or read the *owned* paths (see the module doc).

    Raises ``ValueError`` when *root* is not a folder or an owned path resolves outside it.
    """
    root = _resolve_root(root)
    owned_paths = _normalize_owned(root, owned)
    owned_dirs = [rel for rel, is_dir in owned_paths if is_dir]
    keys = tuple(dict.fromkeys(keys))
    if "." in owned_dirs:
        return Report((), {rule: 0 for rule in RULE_NAMES})
    owned_files = {rel for rel, is_dir in owned_paths if not is_dir}
    unlistable: set[str] = set()
    for rel in owned_dirs:
        owned_files.update(p.relative_to(root).as_posix() for p in _walk_files(root / rel, root, unlistable))
    modules = {_module_name(rel) for rel in owned_files} - {None}
    module_prefixes = [rel.replace("/", ".") for rel in owned_dirs]
    path_pattern = (
        re.compile(rb"(?<![\w./-])(?:%s)(?![\w-])" % b"|".join(re.escape(p.encode()) for p in sorted(owned_files)))
        if owned_files else None
    )
    key_needles = {k: (f'"{k}"'.encode(), f"'{k}'".encode()) for k in keys}

    def is_owned_path(rel: str) -> bool:
        return rel in owned_files or any(rel.startswith(d + "/") for d in owned_dirs)

    def is_owned_module(name: str) -> bool:
        return name in modules or any(name == p or name.startswith(p + ".") for p in module_prefixes)

    hints: list[Hint] = []
    unreadable: list[str] = []
    for path in _walk_files(root, root, unlistable):
        rel = path.relative_to(root).as_posix()
        real = path
        if path.is_symlink():
            try:
                real = path.resolve(strict=True)
                real_rel = real.relative_to(root).as_posix()
            except (OSError, RuntimeError, ValueError):
                continue
            if SKIP_DIRS.intersection(real_rel.split("/")) or is_owned_path(real_rel):
                continue
        if is_owned_path(rel) or not real.is_file():
            continue
        is_python = rel.endswith(".py")
        status, text, names_path, matched_keys = _examine(real, is_python, path_pattern, key_needles)
        if status == "unreadable":
            unreadable.append(rel)
        if status != "ok":
            continue

        rules: set[int] = set()
        if is_python:
            try:
                tree = ast.parse(text)
            except (SyntaxError, ValueError, RecursionError):
                tree = None
            is_test = rel.split("/")[0] == "tests"
            directory = rel.split("/")[:-1]
            for node in ast.walk(tree) if tree is not None else ():
                if isinstance(node, ast.Import):
                    imported = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom):
                    base = _absolute_module(node, directory)
                    if base is None:
                        continue
                    imported = [base, *(f"{base}.{a.name}" for a in node.names if a.name != "*")]
                    if not is_test and is_owned_module(base):
                        rules.add(2)
                else:
                    continue
                if is_test and any(is_owned_module(name) for name in imported):
                    rules.add(1)
        if names_path:
            rules.add(3)
        if matched_keys:
            rules.add(4)
        if rules:
            hints.append(Hint(rel, tuple(sorted(rules)), matched_keys))

    hints.sort(key=lambda h: (4 not in h.rules, h.path))
    counts = {rule: sum(rule in h.rules for h in hints) for rule in RULE_NAMES}
    return Report(tuple(hints), counts, tuple(sorted({*unreadable, *unlistable})))


def format_report(report: Report) -> list[str]:
    """One ``path<TAB>rules`` line per hint, one ``unreadable<TAB>path`` line per file that
    could not be read or folder (``path/``) that could not be listed, then one
    ``# rule N (...): count`` line per rule and, only when some path was unreadable, a
    ``# unreadable: N`` line."""
    lines = []
    for hint in report.hints:
        labels = []
        for rule in hint.rules:
            detail = ": " + ", ".join(f'"{k}"' for k in hint.keys) if rule == 4 else ""
            labels.append(f"rule {rule} ({RULE_NAMES[rule]}{detail})")
        lines.append(f"{hint.path}\t{'; '.join(labels)}")
    lines.extend(f"unreadable\t{path}" for path in report.unreadable)
    lines.extend(f"# rule {rule} ({name}): {report.counts[rule]}" for rule, name in RULE_NAMES.items())
    if report.unreadable:
        lines.append(f"# unreadable: {len(report.unreadable)}")
    return lines
