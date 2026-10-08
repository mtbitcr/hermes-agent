"""Card 6 of the release re-cut: ``hermes release start BATCH`` runs the release of one batch as its
own transient user service, hermes-release-BATCH (S-F).

Each test parses the command before anything is patched, then starts a batch under a temporary
root home. Stand-ins for systemd-run and systemctl, alone on PATH, record every call and give the
user manager's answers that the test sets; the start's waits run on a fake clock."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from hermes_cli import release_ledger as ledger
from hermes_cli.subcommands.release import build_release_parser


def _commit(label: str) -> str:
    return hashlib.sha1(label.encode()).hexdigest()


PREV, NEW, FEATURE, TREE = (_commit(label) for label in ("prev", "new", "feature", "tree"))
# The record's own steps to each batch state: the owner's decision, the begin and the outcome.
STEPS = {
    "open": [], "deferred": ["deferred"], "accepted": ["accepted"],
    "releasing": ["accepted", "begin"], "released": ["accepted", "begin", "released"],
    "failed": ["accepted", "begin", "failed"], "folded": ["accepted", "begin", "restored"],
}
# A kanban worker's environment: the dispatcher pins it to the worker's own board.
WORKER = {
    "HERMES_KANBAN_DB": "/srv/boards/proj-a/kanban.db", "HERMES_KANBAN_BOARD": "proj-a",
    "HERMES_KANBAN_TASK": "t_0badc0de", "HERMES_KANBAN_WORKSPACES_ROOT": "/srv/workspaces",
}
# The stand-in behind both names: it logs its name and arguments, then answers from answers.json.
STAND_IN = r'''
import json, sys
from pathlib import Path

top = Path(__file__).parent
call = [Path(sys.argv[1]).name, *sys.argv[2:]]
with open(top / "calls.jsonl", "a", encoding="utf-8") as log:
    log.write(json.dumps(call) + "\n")
answers = json.loads((top / "answers.json").read_text(encoding="utf-8"))
if call[0] == "systemd-run":
    code, error = answers["launch"]
    sys.stderr.write(error)
    sys.exit(code)
if "list-units" in call:
    for line in answers["units"]:
        print(line)
elif "is-active" in call:
    with open(top / "calls.jsonl", encoding="utf-8") as log:
        asked = sum("is-active" in json.loads(line) for line in log)
    state = answers["states"][min(asked, len(answers["states"])) - 1]
    print(state)
    sys.exit(0 if state == "active" else 3)
'''


class UserManager:
    """The stand-ins' side: what they answer, and every call they had, in order."""

    def __init__(self, top: Path) -> None:
        self.top = top

    def answer(self, *, launch=(0, ""), units=(), states=("active",)) -> None:
        """systemd-run's exit code and error, the listed release units, and the read-back's
        answers in turn (the last one stays)."""
        answers = {"launch": list(launch), "units": list(units), "states": list(states)}
        (self.top / "answers.json").write_text(json.dumps(answers), encoding="utf-8")

    def calls(self) -> list[list[str]]:
        log = self.top / "calls.jsonl"
        return [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []

    def launches(self) -> list[list[str]]:
        return [call for call in self.calls() if call[0] == "systemd-run"]


class Clock:
    """The start's clock: a wait takes no time and moves the clock on by what it waited."""

    now = 0.0

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds


@pytest.fixture
def root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """The root Hermes home at ``<tmp>/home/.hermes``. The stand-ins and their log sit beside
    ``<tmp>/home``, so a look at the home sees only what the start writes."""
    home = tmp_path / "home" / ".hermes"
    home.mkdir(parents=True)
    monkeypatch.setattr(Path, "home", lambda: home.parent)
    monkeypatch.setenv("HERMES_HOME", str(home))
    return home


@pytest.fixture
def manager(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> UserManager:
    """systemd-run and systemctl as stand-ins, the only programs on PATH; by default the launch
    succeeds, no release unit is listed and the unit reads back active."""
    script, bin_dir = tmp_path / "stand_in.py", tmp_path / "bin"
    script.write_text(STAND_IN, encoding="utf-8")
    bin_dir.mkdir()
    for name in ("systemd-run", "systemctl"):
        wrapper = bin_dir / name
        wrapper.write_text(f'#!/bin/sh\nexec "{sys.executable}" -I -S "{script}" "$0" "$@"\n')
        wrapper.chmod(0o755)
    monkeypatch.setenv("PATH", str(bin_dir))
    manager = UserManager(tmp_path)
    manager.answer()
    return manager


def _batch_in(state: str, commit: str = NEW) -> int:
    """One change, ``commit``, in a batch taken to ``state`` by the record's own steps; its id."""
    with contextlib.closing(ledger.connect()) as conn:
        batch = ledger.record_merge(
            conn, merge_commit=commit, pr_url="https://github.com/mtbitcr/hermes-agent/pull/1",
            reviewed_base=PREV, reviewed_head=FEATURE, reviewed_tree=TREE, tier=1,
            card_id="t_release", on_main=True,
        )["batch"]
        batch_id = batch["batch_id"]
        for step in STEPS[state]:
            if step in ("accepted", "deferred"):
                ledger.decide_release(
                    conn, batch_id, decision=step, shown_digest=batch["digest"],
                    expected_version=batch["version"], decision_ref="decisions-page:release",
                )
            elif step == "begin":
                ledger.begin_release(conn, batch_id, prev=PREV, new=commit)
            else:
                ledger.finish_release(conn, batch_id, outcome=step)
        return batch_id


def _states() -> list[str]:
    with contextlib.closing(ledger.connect()) as conn:
        return [batch["state"] for batch in ledger.list_batches(conn)]


def _files(top: Path) -> dict[str, tuple[bytes, int]]:
    """Every path under ``top`` with its bytes and modification time."""
    return {
        str(path): (path.read_bytes() if path.is_file() else b"", path.stat().st_mtime_ns)
        for path in top.rglob("*")
    }


def _start(monkeypatch: pytest.MonkeyPatch, batch_id: int) -> tuple[int, Clock]:
    """``hermes release start BATCH``, parsed by its own parser (on a base without it, this fails
    first), with the start's waits on a fake clock. Returns its exit code and the clock."""
    parser = argparse.ArgumentParser(prog="hermes")
    build_release_parser(parser.add_subparsers())
    args = parser.parse_args(["release", "start", str(batch_id)])
    from hermes_cli import release_unit

    clock = Clock()
    monkeypatch.setattr(
        release_unit, "time", SimpleNamespace(monotonic=clock.monotonic, sleep=clock.sleep)
    )
    return args.func(args), clock


def test_start_launches_a_transient_user_service(root, manager, monkeypatch, capsys):
    batch_id = _batch_in("accepted")
    # Started from a kanban worker under a profile home: the unit gets neither.
    profile = root / "profiles" / "coder"
    profile.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(profile))
    for name, value in WORKER.items():
        monkeypatch.setenv(name, value)
    before = _files(root.parent)

    code, clock = _start(monkeypatch, batch_id)

    assert code == 0
    unit = f"hermes-release-{batch_id}.service"
    # The reads come first, then the one launch, then its read-back.
    listing, launch, readback = manager.calls()
    assert listing[:2] == ["systemctl", "--user"] and "list-units" in listing
    assert readback == ["systemctl", "--user", "is-active", unit]
    options, command = launch[1 : launch.index("--")], launch[launch.index("--") + 1 :]
    # A user service, not a scope: the user manager forks it, outside the caller's control group.
    # It is named for the batch, collected after it exits and restarted on failure, at most
    # three starts in all; it runs from the root home, whose HERMES_HOME is all it is given.
    assert sorted(options) == sorted([
        "--user", f"--unit={unit}", "--collect",
        "--property=Restart=on-failure",
        "--property=StartLimitIntervalSec=infinity", "--property=StartLimitBurst=3",
        f"--working-directory={root}", f"--setenv=HERMES_HOME={root}",
    ])
    assert command == [
        sys.executable, "-m", "hermes_cli.main", "--profile", "default", "release", "run",
    ]
    assert [arg for arg in launch if "HERMES_KANBAN" in arg] == []
    out = capsys.readouterr().out
    assert f"batch {batch_id}" in out and unit in out and "active" in out
    assert clock.now == 0.0  # active at the first answer: no wait
    assert _files(root.parent) == before


def test_start_refuses_without_systemd_run_and_writes_nothing(
    root, manager, tmp_path, monkeypatch, capsys
):
    batch_id = _batch_in("accepted")
    (tmp_path / "bin" / "systemd-run").unlink()  # PATH holds the stand-in systemctl alone
    before = _files(root.parent)

    code, _clock = _start(monkeypatch, batch_id)

    assert code == 1
    out = capsys.readouterr().out
    assert out.startswith("Refused: systemd-run is missing")
    assert out.endswith("Nothing in the release state was written.\n")
    assert _files(root.parent) == before
    assert _states() == ["accepted"]


@pytest.mark.parametrize(
    ("state", "sub", "refused"),
    [("active", "running", True), ("activating", "auto-restart", True), ("failed", "failed", False)],
)
def test_start_refuses_while_another_release_unit_is_active(
    root, manager, monkeypatch, capsys, state, sub, refused
):
    batch_id = _batch_in("accepted")
    other = f"hermes-release-{batch_id + 1}.service"
    command = f"{sys.executable} -m hermes_cli.main --profile default release run"
    manager.answer(units=[f"{other} loaded {state} {sub} {command}"])
    before = _files(root.parent)

    code, _clock = _start(monkeypatch, batch_id)

    out = capsys.readouterr().out
    if refused:  # running, or between two starts: a release is at work
        assert (code, manager.launches()) == (1, [])
        assert out.startswith("Refused: ") and f"{other} is {state}" in out
    else:  # a failed unit the user manager still keeps runs nothing
        assert (code, len(manager.launches())) == (0, 1)
    assert _files(root.parent) == before


@pytest.mark.parametrize(
    "state",
    [
        "open", "deferred", "released", "folded", "no-such-batch", "behind-an-older-accepted-one",
        "accepted", "releasing", "failed",
    ],
)
def test_start_refuses_a_batch_that_is_not_accepted_releasing_or_failed(
    root, manager, monkeypatch, capsys, state
):
    if state == "no-such-batch":  # and no record at all: the start must not make one
        batch_id, reason = 7, "there is no batch 7"
    elif state == "behind-an-older-accepted-one":
        # A release run releases the oldest accepted batch: this one's unit would run the other.
        older = _batch_in("accepted", _commit("older"))
        batch_id = _batch_in("accepted")
        reason = f"batch {batch_id} is accepted after batch {older}"
    else:
        batch_id = _batch_in(state)
        reason = f"batch {batch_id} is {state}, not accepted, releasing or failed"
    before = _files(root.parent)

    code, _clock = _start(monkeypatch, batch_id)

    out = capsys.readouterr().out
    if state in ("accepted", "releasing", "failed"):
        assert (code, len(manager.launches())) == (0, 1)
    else:
        assert (code, manager.launches()) == (1, [])
        assert out.startswith(f"Refused: {reason}")
        assert out.endswith("Nothing in the release state was written.\n")
    assert _files(root.parent) == before


@pytest.mark.parametrize(
    ("launch", "states", "started"),
    [
        ((0, ""), ["activating", "activating", "active"], True),
        ((0, ""), ["failed"], False),
        ((0, ""), ["activating"], False),
        # systemd-run refuses: another unit's active state is no read-back of this start.
        ((1, "Failed to start transient service unit: Unit hermes-release-1.service was already"
              " loaded or has a fragment file.\n"), ["active"], False),
    ],
    ids=["active-at-the-third-answer", "failed", "still-activating", "launch-refused"],
)
def test_start_reads_back_the_unit_as_active(
    root, manager, monkeypatch, capsys, launch, states, started
):
    batch_id = _batch_in("accepted")
    manager.answer(launch=launch, states=states)
    before = _files(root.parent)

    code, clock = _start(monkeypatch, batch_id)

    out = capsys.readouterr().out
    reads = [call for call in manager.calls() if "is-active" in call]
    if started:  # asked until it answered active, well within the bound
        assert (code, len(reads)) == (0, 3) and clock.now < 10.0
        assert f"hermes-release-{batch_id}.service" in out and "could not start" not in out
    else:
        assert code == 1 and f"batch {batch_id} could not start" in out
        if launch[0]:  # systemd-run's own words, and no read-back
            assert launch[1].strip() in out and reads == []
        else:  # asked again and again, up to the stated bound of ten seconds and no longer
            assert len(reads) > 1 and clock.now == pytest.approx(10.0)
    # The batch stays as it was: the start wrote nothing.
    assert _files(root.parent) == before
    assert _states() == ["accepted"]
