"""designer_kit of design_compare: the designer, on its own card and outside a review run, receives the
capture kit, the judge and the design-offer skill's scripts, read from the deployed platform tree and stored
on that card byte for byte with one SHA-256 each, to import by hash into its build machine (capture kit
plan, card A).
"""

import hashlib
import json
import re
from pathlib import Path

from tools.design_access import DESIGNER
from tools.registry import tool_error

JUDGE = Path(__file__).resolve().with_name("design_compare_judge.py")
SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "optional-skills" / "creative" / "design-offer" / "scripts"
MAX_TOTAL_BYTES = 1024 * 1024  # every delivered file together
NOT_DESIGNER = "only the designer may use designer_kit, on its own card, outside a review run"
TOO_LARGE = "the files to deliver exceed 1 MiB in total"
NEXT = "Import every file with its SHA-256 into one folder under these file names; run the scripts from that folder."
_SCRIPT = re.compile(r"[a-z0-9_]+\.(py|mjs)")


def _designer_on_own_card(kb, conn, ctx) -> bool:
    """The run's profile and the card's assignee are both the designer, the run is the card's current run
    and it was not claimed from review, all read from the board's own task and run rows."""
    run, task = kb.get_run(conn, ctx.run), kb.get_task(conn, ctx.task)
    return (run is not None and task is not None and run.task_id == ctx.task and task.current_run_id == ctx.run
            and run.profile == DESIGNER and task.assignee == DESIGNER
            and not kb.run_claimed_from_review(conn, ctx.task, ctx.run))


def deliver(kb, conn, ctx) -> str:
    """Store the kit, the judge, then every regular file (no symlink, no folder) directly in the scripts
    folder with a script name, by name, on the caller's card in its run; return their ids and hashes."""
    from tools.design_compare_tool import KIT_DIR, KIT_FILES, REFUSED, UPLOADER, _kit_sha256

    if not _designer_on_own_card(kb, conn, ctx):
        return tool_error(REFUSED + NOT_DESIGNER)
    scripts = sorted(path.name for path in SCRIPTS_DIR.iterdir()
                     if path.is_file() and not path.is_symlink() and _SCRIPT.fullmatch(path.name))
    files = [(path.name, path.read_bytes())
             for path in (*(KIT_DIR / name for name in KIT_FILES), JUDGE, *(SCRIPTS_DIR / name for name in scripts))]
    if sum(len(data) for _, data in files) > MAX_TOTAL_BYTES:
        return tool_error(REFUSED + TOO_LARGE)
    delivered = []
    for name, data in files:
        attachment_id = kb.store_attachment_bytes(
            conn, ctx.task, name, data, content_type="text/x-python" if name.endswith(".py") else "text/javascript",
            uploaded_by=UPLOADER, board=ctx.board, expected_run_id=ctx.run)
        delivered.append({"attachment_id": attachment_id, "filename": name,
                          "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)})
    return json.dumps({"action": "designer_kit", "files": delivered, "kit_sha256": _kit_sha256(), "next": NEXT})
