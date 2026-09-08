"""The backfill-fence description must state the correct behavior (Finding 4)."""

from __future__ import annotations

import sys
from pathlib import Path

_WORKTREE = Path(__file__).resolve().parents[2]
if str(_WORKTREE) not in sys.path:
    sys.path.insert(0, str(_WORKTREE))

import argparse


def test_backfill_fence_description_is_correct():
    """The description must state that pre-migration boards remain unfenced
    and writable, not that their writes refuse."""
    # Import the module and build the parser via its entry point.
    from hermes_cli import kanban
    root = argparse.ArgumentParser()
    sub = root.add_subparsers()
    kanban_parser = kanban.build_parser(sub)

    # Find the "boards" subparser.
    kanban_sub = kanban_parser._subparsers._group_actions[0]
    boards_parser = kanban_sub.choices['boards']

    # Find the "backfill-fence" subparser under boards.
    boards_sub = boards_parser._subparsers._group_actions[0]
    backfill_parser = boards_sub.choices['backfill-fence']

    desc = backfill_parser.description

    # The description must NOT claim that writes refuse before migration.
    assert "WRITES refuse" not in desc
    assert "writes refuse" not in desc

    # The description MUST state that boards remain unfenced/writable until armed.
    assert "unfenced" in desc or "writable" in desc
    assert "until" in desc
