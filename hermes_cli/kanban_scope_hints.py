"""Read-only search for the tests and readers of the paths a proposed card will own.

Owned paths cannot change after a card exists, so the proposer needs to see what pins
or reads them before the card is created. Given a repository root, the owned paths and
optionally the keys the card will change, ``find_hints`` lists the files outside the
owned paths that match one of five rules:

1. a Python file under ``tests/`` that imports a module in the owned paths; one step
   further, a test that imports a module of a test file listed by this rule (it has no
   rule number of its own and is not counted: ``Hint.imports_pinning_test``; it is a
   separate row, placed after the file's own row when it has one, which is never rewritten);
2. any other Python file that imports a name from such a module;
3. a file that names an owned file path;
4. for each key, a file that contains that exact key in single or double quotes;
5. a saved example: ``tests/contracts/owner_payloads/<family>/<record_kind>.json`` for each
   ``owner_payload_example(family, record_kind, ...)`` call in a listed test file (a Python
   file under ``tests/``, a helper row included); the keywords ``family`` and ``record_kind``
   work too. One narrow rule resolves a name, with no scope rule of the language: an argument
   is resolved only when it is a string literal at the call, or a plain name that the whole
   file binds exactly once (see ``_binding_counts``), by a module-level assignment of a string
   literal. A literal must be one safe path component (not empty, ``.`` or ``..``; no ``/``,
   ``\\`` or NUL; not absolute). Both resolved: that one file ("saved example"). Only the
   family resolved: every ``.json`` file of that family folder ("every example of the family").
   Family not resolved: one row for the test file itself, with the line of the call ("saved
   example not resolved"), and no example file. Only a file the main scan would also read is
   listed (it, and where it resolves, are inside the repository, not owned and not in
   ``SKIP_DIRS``; a family folder is checked the same way before it is read), one row per call
   and file, as its own row (rules ``(5,)``, no keys) right after the file's own row when it has one, which is never
   rewritten. The count of rule 5 counts each of these rows.

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
from collections import Counter
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Iterable, Iterator, Sequence

SKIP_DIRS = frozenset({".git", "node_modules", ".venv", "venv", ".worktrees", "__pycache__"})
BINARY_PROBE_BYTES = 8192
READ_CHUNK_BYTES = 1024 * 1024
RULE_NAMES = {
    1: "test imports owned module",
    2: "imports name from owned module",
    3: "names owned path",
    4: "quoted key",
    5: "saved example",
}
EXAMPLES_DIR = "tests/contracts/owner_payloads"
EXAMPLE_CALL = "owner_payload_example"
EVERY_EXAMPLE = "every example of the family"
NOT_RESOLVED = "saved example not resolved"


@dataclass(frozen=True)
class Hint:
    """One file to consider owning: every rule that matched it, and the keys for rule 4
    (``every_example``: a rule 5 row listed because the record kind was not resolved;
    ``unresolved_line``: a rule 5 row for a test file, for the call at that line whose family
    was not resolved)."""

    path: str
    rules: tuple[int, ...]
    keys: tuple[str, ...] = ()
    imports_pinning_test: bool = False
    every_example: bool = False
    unresolved_line: int = 0


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


_NAMING_NODES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.ExceptHandler, ast.MatchAs, ast.MatchStar,
                 *(getattr(ast, n) for n in ("TypeVar", "ParamSpec", "TypeVarTuple") if hasattr(ast, n)))


def _call_name(call: ast.Call) -> str | None:
    return call.func.id if isinstance(call.func, ast.Name) else getattr(call.func, "attr", None)


def _binding_counts(tree: ast.Module) -> Counter[str]:
    """How many times the whole file binds each name, in every scope, with no scope rule: an
    assignment, annotated assignment, augmented assignment, named expression, parameter, loop,
    ``with`` or comprehension target, ``except`` name, match capture, import, function, class or
    type parameter of that name, a ``global`` or ``nonlocal`` declaration, and a deletion."""
    counts: Counter[str] = Counter()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            if not isinstance(node.ctx, ast.Load):
                counts[node.id] += 1
        elif isinstance(node, ast.arg):
            counts[node.arg] += 1
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            counts.update(a.asname or a.name.split(".")[0] for a in node.names if a.name != "*")
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            counts.update(node.names)
        elif isinstance(node, _NAMING_NODES):
            if node.name:
                counts[node.name] += 1
        elif isinstance(node, ast.MatchMapping) and node.rest:
            counts[node.rest] += 1
    return counts


def _module_literals(tree: ast.Module) -> dict[str, str]:
    """The names that the whole file binds once, by a module-level assignment of a string literal.
    A wildcard import makes that unprovable, so then no name is resolved."""
    literals: dict[str, str] = {}
    if any(isinstance(n, ast.ImportFrom) and any(a.name == "*" for a in n.names) for n in ast.walk(tree)):
        return literals
    counts = _binding_counts(tree)
    for node in tree.body:
        targets, value = (node.targets, node.value) if isinstance(node, ast.Assign) else (
            ([node.target], node.value) if isinstance(node, ast.AnnAssign) else ([], None))
        if isinstance(value, ast.Constant) and isinstance(value.value, str):
            literals.update({t.id: value.value for t in targets if isinstance(t, ast.Name) and counts[t.id] == 1})
    return literals


def _safe_component(name: str) -> bool:
    """Whether *name* is one path component that stays inside its folder."""
    if name in ("", ".", "..") or any(c in name for c in "/\\\0"):
        return False
    return not (PurePosixPath(name).is_absolute() or PureWindowsPath(name).drive)


def _saved_examples(root: Path, trees: Iterable[tuple[str, ast.Module]], scan_target) -> tuple[list[tuple[str, bool]], list[tuple[str, int]]]:
    """What the ``owner_payload_example`` calls of *trees* (test file path, syntax tree) read: the
    saved example rows, one per call and file, as (path relative to *root*, whether it is listed
    only as "every example of the family"); and, for each call whose family is not resolved, the test file and the line of
    the call. *scan_target(path, follow)* is the scan's own test of a file: the path it really is,
    or ``None`` when the scan would skip it (outside the repository, owned, in a skipped folder)."""
    found: list[tuple[str, bool]] = []
    unresolved: list[tuple[str, int]] = []
    for rel_test, tree in trees:
        literals = _module_literals(tree)

        def resolved(node: ast.expr | None) -> str | None:
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                return node.value
            return literals.get(node.id) if isinstance(node, ast.Name) else None

        calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and _call_name(n) == EXAMPLE_CALL]
        for call in sorted(calls, key=lambda n: (n.lineno, n.col_offset)):
            prefix = call.args[:next((i for i, a in enumerate(call.args) if isinstance(a, ast.Starred)), len(call.args))]

            def slot(name: str, index: int) -> str | None:
                given = prefix[index:index + 1] + [k.value for k in call.keywords if k.arg == name]
                return resolved(given[0]) if len(given) == 1 else None

            family, kind = slot("family", 0), slot("record_kind", 1)
            if family is None:
                unresolved.append((rel_test, call.lineno))
                continue
            if not all(_safe_component(n) for n in (family, kind) if n is not None):
                continue
            folder = root / EXAMPLES_DIR / family
            if kind is not None:
                files = [folder / f"{kind}.json"]
            else:
                try:
                    real_folder = folder.resolve(strict=True).relative_to(root).as_posix()
                except (OSError, RuntimeError, ValueError):
                    continue
                if SKIP_DIRS.intersection(real_folder.split("/")):
                    continue
                files = sorted(folder.glob("*.json"))
            for file in files:
                if scan_target(file, True) is not None:
                    found.append((file.relative_to(root).as_posix(), kind is None))
    return found, unresolved


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

    def scan_target(path: Path, follow: bool = False) -> Path | None:
        """The file *path* really is, or ``None`` when the scan skips it: a symlink (or, with
        *follow*, any path) that does not resolve inside the repository or lands in a skipped
        folder, or a file that is owned (by its path or where it resolves) or not a file."""
        rel = path.relative_to(root).as_posix()
        real = path
        if SKIP_DIRS.intersection(rel.split("/")[:-1]):
            return None
        if follow or path.is_symlink():
            try:
                real = path.resolve(strict=True)
                real_rel = real.relative_to(root).as_posix()
            except (OSError, RuntimeError, ValueError):
                return None
            if SKIP_DIRS.intersection(real_rel.split("/")) or is_owned_path(real_rel):
                return None
        if is_owned_path(rel) or not real.is_file():
            return None
        return real

    hints: list[Hint] = []
    test_imports: dict[str, set[str]] = {}
    example_trees: dict[str, ast.Module] = {}
    unreadable: list[str] = []
    for path in _walk_files(root, root, unlistable):
        rel = path.relative_to(root).as_posix()
        real = scan_target(path)
        if real is None:
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
            imports_seen: set[str] = set()
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
                    if is_test and isinstance(node, ast.Call) and _call_name(node) == EXAMPLE_CALL:
                        example_trees[rel] = tree
                    continue
                imports_seen.update(imported)
                if is_test and any(is_owned_module(name) for name in imported):
                    rules.add(1)
            if is_test and 1 not in rules:
                test_imports[rel] = imports_seen
        if names_path:
            rules.add(3)
        if matched_keys:
            rules.add(4)
        if rules:
            hints.append(Hint(rel, tuple(sorted(rules)), matched_keys))

    pinning = {_module_name(h.path) for h in hints if 1 in h.rules}
    helpers = {rel for rel, names in test_imports.items() if pinning & names}
    hints.extend(Hint(rel, (), (), True) for rel in sorted(helpers))
    listed = {h.path for h in hints}
    saved, unresolved = _saved_examples(root, [(rel, tree) for rel, tree in example_trees.items() if rel in listed], scan_target)
    hints.extend(Hint(rel, (5,), (), False, every) for rel, every in saved)
    hints.extend(Hint(rel, (5,), (), False, False, line) for rel, line in unresolved)
    keyed = {h.path for h in hints if 4 in h.rules}
    hints.sort(key=lambda h: (h.path not in keyed, h.path, h.imports_pinning_test, h.rules == (5,)))
    counts = {rule: sum(rule in h.rules for h in hints) for rule in RULE_NAMES}
    return Report(tuple(hints), counts, tuple(sorted({*unreadable, *unlistable})))


def format_report(report: Report) -> list[str]:
    """One ``path<TAB>rules`` line per hint, one ``unreadable<TAB>path`` line per file that
    could not be read or folder (``path/``) that could not be listed, then one
    ``# rule N (...): count`` line per rule (1 to 5) and, only when some path was unreadable, a
    ``# unreadable: N`` line."""
    lines = []
    for hint in report.hints:
        labels = ["rule 1 (test imports a pinning test)"] if hint.imports_pinning_test else []
        for rule in hint.rules:
            detail = ": " + ", ".join(f'"{k}"' for k in hint.keys) if rule == 4 else (
                ": " + EVERY_EXAMPLE if rule == 5 and hint.every_example else "")
            if rule == 5 and hint.unresolved_line:
                labels.append(f"rule 5 ({NOT_RESOLVED}): line {hint.unresolved_line}")
            else:
                labels.append(f"rule {rule} ({RULE_NAMES[rule]}{detail})")
        lines.append(f"{hint.path}\t{'; '.join(labels)}")
    lines.extend(f"unreadable\t{path}" for path in report.unreadable)
    lines.extend(f"# rule {rule} ({name}): {report.counts[rule]}" for rule, name in RULE_NAMES.items())
    if report.unreadable:
        lines.append(f"# unreadable: {len(report.unreadable)}")
    return lines
