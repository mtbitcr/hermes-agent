"""Child process: run a REAL dispatcher tick, optionally SIGKILLing itself.

Run as::

    python _kanban_worker_identity_dispatch_child.py --home H --board B \
        [--crash before-bind|after-bind|never] [--spawn sleeper|dead] \
        [--barrier PATH]

The tick is the SHIPPED public entry point — ``kanban_db.dispatch_once``,
the same call the gateway dispatcher and ``hermes kanban dispatch`` make —
and the worker it starts is a REAL child process, so the pid this
dispatcher binds names something that genuinely exists. The crash is an
uncatchable ``SIGKILL`` to this process's own pid, so nothing unwinds,
flushes or records anything: exactly what a power loss looks like to the
next dispatcher.

``--crash before-bind`` dies on entry to ``_set_worker_pid`` (a worker is
running, its pid has never been written); ``after-bind`` dies once it has
returned (the pid AND its identity witness are committed and the next step
never happened). ``--barrier`` parks the process until the file contains
``go``, which is how two dispatchers can be released at the same instant.

Prints one ``SPAWNED <pid>`` line per worker started and one
``REPORT {...}`` line with the tick's result, so the parent can clean up
real processes and read what the tick did.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

_WORKTREE = Path(__file__).resolve().parents[2]
if str(_WORKTREE) not in sys.path:
    sys.path.insert(0, str(_WORKTREE))

_SLEEPER = "import time; time.sleep(600)"


def _die() -> None:
    """Stop this process the way a power loss does."""
    sys.stdout.flush()
    sys.stderr.flush()
    os.kill(os.getpid(), signal.SIGKILL)


def _wait_for_barrier(barrier: Path) -> None:
    ready = barrier.parent / f"{barrier.name}.ready.{os.getpid()}"
    ready.write_text("ready\n", encoding="utf-8")
    deadline = time.time() + 60
    while time.time() < deadline:
        try:
            if barrier.read_text(encoding="utf-8").strip() == "go":
                return
        except OSError:
            pass
        time.sleep(0.02)
    raise SystemExit("barrier was never released")


def main(argv: "list[str]") -> int:
    parser = argparse.ArgumentParser(prog="kanban-worker-identity-dispatch-child")
    parser.add_argument("--home", required=True)
    parser.add_argument("--board", required=True)
    parser.add_argument("--crash", default="never",
                        choices=["never", "before-bind", "after-bind"])
    parser.add_argument("--spawn", default="sleeper", choices=["sleeper", "dead"])
    parser.add_argument("--barrier", default=None)
    args = parser.parse_args(argv)

    os.environ["HERMES_HOME"] = args.home
    os.environ["HERMES_KANBAN_CRASH_GRACE_SECONDS"] = "0"
    for var in (
        "HERMES_KANBAN_DB",
        "HERMES_KANBAN_WORKSPACES_ROOT",
        "HERMES_KANBAN_HOME",
        "HERMES_KANBAN_BOARD",
    ):
        os.environ.pop(var, None)

    from hermes_cli import kanban_db as kb
    from hermes_cli import profiles

    # Synthetic assignees ("worker") have no profile directory on disk; the
    # dispatcher's profile-exists guard would route them to
    # ``skipped_nonspawnable`` instead of spawning. Same allowance the
    # in-process suites make with the ``all_assignees_spawnable`` fixture.
    profiles.profile_exists = lambda name: True

    real_set_worker_pid = kb._set_worker_pid

    def _crashing_set_worker_pid(*a, **kw):
        if args.crash == "before-bind":
            _die()
        result = real_set_worker_pid(*a, **kw)
        if args.crash == "after-bind":
            _die()
        return result

    if args.crash != "never":
        kb._set_worker_pid = _crashing_set_worker_pid

    def spawn_fn(task, workspace_path, board=None):
        # The worker must NOT inherit this process's stdout/stderr: a
        # long-lived child holding the pipe open would keep the parent's
        # ``communicate()`` blocked until the worker exited, which is the
        # opposite of the detached spawn the real dispatcher performs.
        kwargs = {
            "stdin": subprocess.DEVNULL,
            "stdout": subprocess.DEVNULL,
            "stderr": subprocess.DEVNULL,
            "start_new_session": True,
        }
        if args.spawn == "dead":
            proc = subprocess.Popen([sys.executable, "-c", "pass"], **kwargs)
            proc.wait(timeout=30)
        else:
            proc = subprocess.Popen([sys.executable, "-c", _SLEEPER], **kwargs)
        print(f"SPAWNED {proc.pid}", flush=True)
        return proc.pid

    if args.barrier:
        _wait_for_barrier(Path(args.barrier))

    conn = kb.connect(board=args.board)
    try:
        result = kb.dispatch_once(conn, spawn_fn=spawn_fn, board=args.board)
        report = {
            "spawned": [list(item) for item in (result.spawned or [])],
            "reclaimed": result.reclaimed,
            "crashed": list(result.crashed or []),
            "skipped_locked": bool(getattr(result, "skipped_locked", False)),
        }
    finally:
        conn.close()
    print("REPORT " + json.dumps(report), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
