"""The release unit (S-F): the release run of one batch as its own transient user service.

``hermes release start BATCH`` asks the user manager, through ``systemd-run --user``, for the
service ``hermes-release-BATCH``. A service, not a scope: the user manager forks it in a control
group of its own, outside the gateway's and the caller's, so the release keeps running while the
units it restarts stop and start. It is given the user manager's own environment and HERMES_HOME
at the root home, nothing of the caller's; the run removes every HERMES_KANBAN variable from it
before any other step, so no board context reaches the release. It runs ``release run BATCH``
from the root home under the default profile, whatever profile is sticky there, and it is
collected once it exits, failed or not.

A run that exits non-zero, or that a hard kill ends, is started again (Restart=on-failure), at
most START_LIMIT_BURST starts in all: StartLimitIntervalSec=infinity never lets the count of
starts reset, so the user manager refuses the next one, and the unit fails and is collected.
Every start runs the same batch, which the run refuses unless it is the accepted batch a run
takes first, so a restart never reaches another batch. Until card 7 recovers a hard-killed
batch, a restarted run of a batch that is still releasing refuses, as it is no longer accepted,
and this limit ends the loop.

Without systemd-run nothing starts, by the rule of update_abort_recovery.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

# Three starts in all: the first run and two restarts.
START_LIMIT_BURST = 3
# How long the start waits for the user manager to answer active, and how often it asks.
READBACK_SECONDS = 10.0
POLL_SECONDS = 0.5
# How long any other call to systemd-run or systemctl may take.
CALL_SECONDS = 30.0
_PATTERN = "hermes-release-*"
# The pattern is systemd's, not a name check: only a unit start names is a release unit.
_UNIT_RE = re.compile(r"hermes-release-[0-9]+\.service")
# A release unit at rest; in any other state a release is at work, or starting or stopping.
_AT_REST = ("inactive", "failed")


class UnitError(Exception):
    """systemd-run or systemctl could not be run, or refused; the message is the plain reason."""


def unit_name(batch_id: int) -> str:
    return f"hermes-release-{batch_id}.service"


def find_systemd_run() -> str | None:
    return shutil.which("systemd-run")


def busy_units() -> list[tuple[str, str]]:
    """Each release unit the user manager has loaded and that is not at rest, with its state."""
    listed = _checked(
        "systemctl", "--user", "list-units", "--all", "--full", "--plain", "--no-legend",
        "--no-pager", _PATTERN,
    )
    busy = []
    for fields in (line.split() for line in listed.splitlines()):
        if fields and _UNIT_RE.fullmatch(fields[0]):
            if len(fields) < 4:  # UNIT LOAD ACTIVE SUB
                raise UnitError(f"systemctl listed {fields[0]} without its state")
            if fields[2] not in _AT_REST:
                busy.append((fields[0], fields[2]))
    return busy


def launch(systemd_run: str, batch_id: int, root: Path) -> None:
    """Start ``release run BATCH`` as the release unit of ``batch_id``, from the root home ``root``.

    The user manager makes the unit and starts it in one call, which a unit of that name that is
    still loaded refuses. Raises UnitError when the unit did not start.
    """
    _checked(
        systemd_run, "--user", f"--unit={unit_name(batch_id)}", "--collect",
        "--property=Restart=on-failure", "--property=StartLimitIntervalSec=infinity",
        f"--property=StartLimitBurst={START_LIMIT_BURST}",
        f"--working-directory={root}", f"--setenv=HERMES_HOME={root}",
        "--", sys.executable, "-m", "hermes_cli.main", "--profile", "default", "release", "run",
        str(batch_id),
    )


def read_back(batch_id: int) -> str:
    """The user manager's last answer for the unit of ``batch_id``: asked until it answers
    active, or until READBACK_SECONDS have passed."""
    answer, deadline = "no answer", time.monotonic() + READBACK_SECONDS
    while (left := deadline - time.monotonic()) > 0:
        try:
            done = _run(["systemctl", "--user", "is-active", unit_name(batch_id)], left)
            answer = done.stdout.strip() or done.stderr.strip() or f"exit {done.returncode}"
        except UnitError as error:
            answer = str(error)
        if answer == "active":
            break
        time.sleep(min(POLL_SECONDS, max(deadline - time.monotonic(), 0)))
    return answer


def _checked(*argv: str) -> str:
    """The output of ``argv``; UnitError when it exits non-zero, with its own words."""
    done = _run(list(argv), CALL_SECONDS)
    if done.returncode:
        reason = done.stderr.strip() or "no reason given"
        raise UnitError(f"{Path(argv[0]).name} exited {done.returncode}: {reason}")
    return done.stdout


def _run(argv: list[str], timeout: float) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            argv, stdin=subprocess.DEVNULL, capture_output=True, encoding="utf-8",
            errors="replace", timeout=timeout, check=False,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise UnitError(
            f"{Path(argv[0]).name} could not be run ({type(error).__name__}: {error})"
        ) from error
