#!/usr/bin/env python3
"""Fail a pull request that adds or changes a test file listed in tests/fork_dormant_skips.txt.

CI never runs a listed file: ``scripts/run_tests_parallel.py`` drops it from the sharded discovery and
``tests/conftest.py`` from directory collection. A pull request that edits one gets a green run that never
executed the edit, so the change looks tested when it is not. The fix is to take the file off the list in
the same pull request (and make it pass), or to leave the file alone.

A changed file is on the list exactly when ``scripts/run_tests_parallel.py`` would drop it because of the
list: its name matches ``test_*.py`` and its resolved path is the resolved path of an entry (entries are
stripped; blank lines and ``#`` comments are ignored). The list is read at the head, so a pull request that
removes an entry is judged by its own list. Changed files are the diff between the merge-base and the head
under ``tests/``, deletions excluded, with rename detection off: a renamed or copied file counts as added
under its new path whatever the local ``diff.renames`` setting says.

Exit 0 when no changed test file is listed, 1 with one line per listed one, 2 when a ref does not resolve
or there is no merge-base (refusing to report a clean result).

Usage:
    python scripts/ci/check_dormant_changed.py [--base origin/main] [--head HEAD]
"""
from __future__ import annotations

import argparse
import fnmatch
import subprocess
import sys
from pathlib import Path, PurePosixPath

LIST = "tests/fork_dormant_skips.txt"


def _git(*args: str, cwd: Path | None = None, check: bool = False) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["git", *args], cwd=cwd, check=check,
                          capture_output=True, text=True, encoding="utf-8", errors="replace")


def listed(root: Path, head: str) -> set[Path]:
    """Resolved paths of the list entries at ``head``, read the way scripts/run_tests_parallel.py reads them."""
    shown = _git("show", f"{head}:{LIST}", cwd=root)
    if shown.returncode != 0:  # no list at head: nothing is dropped
        return set()
    entries = (line.strip() for line in shown.stdout.splitlines())
    return {(root / entry).resolve() for entry in entries if entry and not entry.startswith("#")}


def changed_tests(root: Path, base: str, head: str) -> list[str]:
    """Paths under tests/ that ``head`` adds or changes since ``base``; a rename or copy adds its new path."""
    out = _git("diff", "--no-renames", "--name-only", "-z", "--diff-filter=AMT", base, head, "--", "tests/",
               cwd=root, check=True).stdout
    return [path for path in out.split("\0") if path]


def dormant_changed(root: Path, base: str, head: str) -> list[str]:
    """Changed test files that CI drops from collection because the list at ``head`` names them, sorted."""
    skips = listed(root, head)
    return sorted(
        path for path in changed_tests(root, base, head)
        if fnmatch.fnmatchcase(PurePosixPath(path).name, "test_*.py") and (root / path).resolve() in skips
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    ap.add_argument("--base", default="origin/main")
    ap.add_argument("--head", default="HEAD")
    args = ap.parse_args(argv)
    for ref in (args.base, args.head):
        if _git("rev-parse", "--verify", "--quiet", f"{ref}^{{commit}}").returncode != 0:
            print(f"dormant-changed: cannot resolve ref {ref!r} (fetch it first); refusing to report a clean result",
                  file=sys.stderr)
            return 2
    base = _git("merge-base", args.base, args.head).stdout.strip()
    if not base:
        print(f"dormant-changed: no merge-base between {args.base!r} and {args.head!r}; "
              "refusing to report a clean result", file=sys.stderr)
        return 2
    root = Path(_git("rev-parse", "--show-toplevel", check=True).stdout.strip())
    flagged = dormant_changed(root, base, args.head)
    for path in flagged:
        print(f"{path}: changed by this pull request but listed in {LIST}, so CI never runs it while it stays listed.")
    if not flagged:
        print(f"dormant-changed: no changed test file is listed in {LIST}")
    return 1 if flagged else 0


if __name__ == "__main__":
    sys.exit(main())
