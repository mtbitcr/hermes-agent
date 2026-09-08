"""Child process: drive a REAL board removal and SIGKILL at a named point.

Run as ``python _kanban_removal_crash_child.py <home> <slug> <crash> <mode>``.

The removal is driven through the SHIPPED CLI entry point — the real
argparse tree and ``hermes_cli.kanban.kanban_command`` — so a restart
after the crash is a restart of the thing an operator runs, not of a
test-only driver. The crash itself is an uncatchable ``SIGKILL`` to this
process's own pid, so nothing gets a chance to unwind, flush or record
anything: exactly what a power loss looks like to the next process.

``<crash>`` names the point. ``before-<fn>`` kills on entry to a real
function; ``after-<fn>`` kills once it has returned, so its durable write
has committed and the NEXT step has not begun. ``never`` drives the
removal to completion and is the restart.
"""

from __future__ import annotations

import argparse
import os
import signal
import sys
from pathlib import Path

_WORKTREE = Path(__file__).resolve().parents[2]
if str(_WORKTREE) not in sys.path:
    sys.path.insert(0, str(_WORKTREE))


# crash point -> (kanban_db attribute, "before" | "after")
CRASH_POINTS: "dict[str, tuple[str, str]]" = {
    # Every adjacent phase boundary, taken on the far side of the
    # transition's own durable commit.
    "at-intent": ("advance_removal_to_fenced", "before"),
    "after-fenced": ("advance_removal_to_fenced", "after"),
    "after-quiesced": ("advance_removal_to_quiesced", "after"),
    "after-carried": ("advance_removal_to_carried", "after"),
    "after-released": ("advance_removal_to_released", "after"),
    "after-applied": ("advance_removal_to_applied", "after"),
    # Inside the permanent apply, step by step.
    "apply-after-storage": ("_destroy_board_storage", "after"),
    "apply-after-work-area": ("_destroy_work_area_content", "after"),
    "apply-after-deregister": ("_deregister_work_area", "after"),
    "apply-before-record": ("record_applied_mode_content", "before"),
    "after-apply-content": ("apply_permanent_mode_content", "after"),
    # The last two boundaries, including the window between the terminal
    # transition and the terminal receipt.
    "after-swept": ("advance_removal_to_swept", "after"),
    "after-done": ("complete_removal", "after"),
    # The two interrupted directions of the durable prepare-then-apply
    # receipt protocol, taken from inside it:
    #   * the exact receipt content is committed and the terminal register
    #     transition has NOT happened;
    #   * the terminal transition HAS happened and the prepared receipt has
    #     not been marked applied.
    "after-prepared-receipt": ("prepare_permanent_removal_receipt", "after"),
    "after-terminal-transition": ("apply_prepared_removal_receipt", "before"),
}


def _die() -> None:
    """Stop this process the way a power loss does."""
    sys.stdout.flush()
    sys.stderr.flush()
    os.kill(os.getpid(), signal.SIGKILL)


def _install_crash(kb, crash: str) -> None:
    if crash == "never":
        return
    name, when = CRASH_POINTS[crash]
    real = getattr(kb, name)

    def _wrapped(*args, **kwargs):
        if when == "before":
            _die()
        result = real(*args, **kwargs)
        _die()
        return result  # pragma: no cover - SIGKILL does not return

    setattr(kb, name, _wrapped)


def main(argv: "list[str]") -> int:
    parser = argparse.ArgumentParser(prog="kanban-removal-crash-child")
    parser.add_argument("home")
    parser.add_argument("slug")
    parser.add_argument("crash", choices=sorted(CRASH_POINTS) + ["never"])
    parser.add_argument("mode", choices=["permanent", "reversible"])
    args = parser.parse_args(argv)

    os.environ["HERMES_HOME"] = args.home
    for var in (
        "HERMES_KANBAN_DB",
        "HERMES_KANBAN_WORKSPACES_ROOT",
        "HERMES_KANBAN_HOME",
        "HERMES_KANBAN_BOARD",
    ):
        os.environ.pop(var, None)

    from hermes_cli import kanban as kanban_cli
    from hermes_cli import kanban_db as kb

    _install_crash(kb, args.crash)

    cli_argv = ["boards", "rm", args.slug]
    if args.mode == "permanent":
        cli_argv += [
            "--delete",
            "--confirm",
            kb.permanent_removal_disclosure(args.slug).required_response,
        ]
    wrap = argparse.ArgumentParser(prog="hermes-crash-child", add_help=False)
    top = wrap.add_subparsers(dest="_top")
    tree = kanban_cli.build_parser(top)
    code = kanban_cli.kanban_command(tree.parse_args(cli_argv))
    print(f"CHILD-EXIT {code}", flush=True)
    return int(code)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
