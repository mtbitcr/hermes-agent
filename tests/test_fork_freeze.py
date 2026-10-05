"""Fork freeze checks: owner decision of 2026-10-04.

Fork code lands only in modules that run, and the large upstream files grow
only for a stated reason. These two checks read source files on purpose; the
owner exempted them from the AGENTS.md rule against source-reading and
change-detector tests. scripts/fork_dormant_modules.py regenerates
fork_freeze/dormant_modules.json during an upstream sync.
"""

import hashlib
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
FREEZE = ROOT / "tests" / "fork_freeze"
RATCHETED = (
    "hermes_cli/kanban_db.py",
    "gateway/platforms/api_server.py",
    "tools/kanban_tools.py",
    "plugins/kanban/dashboard/plugin_api.py",
    "cron/scheduler.py",
)


def _line_count(data: bytes) -> int:
    return data.count(b"\n") + (1 if data and not data.endswith(b"\n") else 0)


def test_dormant_modules_keep_their_recorded_hash():
    recorded = json.loads((FREEZE / "dormant_modules.json").read_text(encoding="utf-8"))
    changed = []
    for path, digest in sorted(recorded.items()):
        target = ROOT / path
        if not target.is_file():
            changed.append(f"{path} (missing)")
        elif hashlib.sha256(target.read_bytes()).hexdigest() != digest:
            changed.append(path)
    if changed:
        pytest.fail(
            "an unused upstream copy changed; make the change in the live module, or regenerate the list with the script during an upstream sync.\n"
            + "\n".join(changed),
            pytrace=False,
        )


def test_large_upstream_files_stay_within_recorded_line_counts():
    recorded = json.loads((FREEZE / "line_ratchet.json").read_text(encoding="utf-8"))
    grown = []
    for path in RATCHETED:
        target = ROOT / path
        if path not in recorded:
            grown.append(f"{path}: not recorded in tests/fork_freeze/line_ratchet.json")
        elif not target.is_file():
            grown.append(f"{path}: missing, recorded {recorded[path]}")
        elif _line_count(target.read_bytes()) > recorded[path]:
            grown.append(f"{path}: {_line_count(target.read_bytes())} lines, recorded {recorded[path]}")
    if grown:
        pytest.fail(
            "put new code in its own module with a thin call here, or raise the number in this pull request and state the reason in its description.\n"
            + "\n".join(grown),
            pytrace=False,
        )
