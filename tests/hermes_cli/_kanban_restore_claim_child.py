"""Child process: claim a task for real, bind this pid to it, then exit.

Run as ``python _kanban_restore_claim_child.py <home> <slug> <task>``.

The claim is taken through the shipped ``kanban_db.claim_task`` and the
pid is bound through the shipped ``_set_worker_pid``, so the row this
leaves behind is exactly the row a real worker leaves behind — a Held
reservation carrying the identity (pid plus its start-time witness) of a
process that then stops existing. That is what makes the holder PROVABLY
absent to the removal's own quiescence probe, which is the only way a
reversible removal ever completes over a prior claim.

The claim handle is printed as one JSON line so the parent can keep the
stale handle after this process is gone.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

_WORKTREE = Path(__file__).resolve().parents[2]
if str(_WORKTREE) not in sys.path:
    sys.path.insert(0, str(_WORKTREE))


def main(argv: "list[str]") -> int:
    home, slug, task = argv

    os.environ["HERMES_HOME"] = home
    for var in (
        "HERMES_KANBAN_DB",
        "HERMES_KANBAN_WORKSPACES_ROOT",
        "HERMES_KANBAN_HOME",
        "HERMES_KANBAN_BOARD",
    ):
        os.environ.pop(var, None)

    from hermes_cli import kanban_db as kb

    conn = kb.connect(board=slug)
    try:
        claimed = kb.claim_task(conn, task)
        if claimed is None:
            print("CLAIM-FAILED", flush=True)
            return 1
        kb._set_worker_pid(conn, task, os.getpid())
    finally:
        conn.close()
    print(
        "CLAIM " + json.dumps({"claim_lock": claimed.claim_lock, "pid": os.getpid()}),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
