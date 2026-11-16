#!/usr/bin/env python3
"""List the tests and readers outside the paths a proposed kanban card will own.

Run it before the card is created, so the proposer can own those files from the start.
It only reads files inside the repository; see ``hermes_cli/kanban_scope_hints.py``.

Prints one line per file (``path``, a tab, every rule that matched; files containing a
--key value first), then one ``# rule N (...): count`` line per rule, rules 1 to 4.

An owned folder owns every file in it, for every rule; ``--owned .`` owns the whole
repository (no hints, all counts zero). There is no file size limit. A file with a NUL byte
in its first 8 KiB is binary and is not searched. A text file that cannot be read whole (an
OS error, or bytes that are not UTF-8) is named in one ``unreadable<TAB>path`` line, after
the hints, and counted in a final ``# unreadable: N`` line; both appear only when there is
such a file.

Exit 0 on success, 2 when an owned path or the root is refused (outside the repository,
or the root is not a folder) or no owned path is given.

Usage:
    python scripts/kanban_scope_hints.py [--root REPO] [--key KEY ...] PATH [PATH ...]
    python scripts/kanban_scope_hints.py [--root REPO] [--key KEY ...] --owned PATH [--owned PATH ...]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.dont_write_bytecode = True  # the search writes nothing, not even interpreter cache files
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hermes_cli.kanban_scope_hints import default_root, find_hints, format_report  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("paths", nargs="*", metavar="PATH", help="a path the card will own")
    ap.add_argument("--owned", action="append", default=[], metavar="PATH", help="a path the card will own (repeatable)")
    ap.add_argument("--key", action="append", default=[], metavar="VALUE", help="a key the card will change (repeatable)")
    ap.add_argument("--root", default=None, help="repository root (default: the repository root of the current directory)")
    args = ap.parse_args(argv)
    owned = [*args.paths, *args.owned]
    if not owned:
        ap.error("give at least one owned path")
    try:
        report = find_hints(args.root or default_root(), owned, args.key)
    except ValueError as exc:
        print(f"kanban-scope-hints: {exc}", file=sys.stderr)
        return 2
    for line in format_report(report):
        print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
