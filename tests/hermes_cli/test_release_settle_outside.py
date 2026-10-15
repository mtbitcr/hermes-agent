"""Card 5 of the platform-record fixes (finding 156): ``hermes release settle`` records a release
made outside ``hermes release run``, such as the manual emergency release, which runs it as its
last step. Every waiting change the running commit contains is recorded released, so the next
decision no longer offers it; the changes it does not contain still wait.

Each test runs under a temporary root home, with a fake ancestry check over a made-up history in
place of the live reader, so no git, network or live host is reached. The command and the new
ledger function are reached when a test runs, so a missing one fails that test on its own.
"""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import json
import os
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from hermes_cli import release_ledger as ledger
from hermes_cli.subcommands.release import build_release_parser
from tests.hermes_cli.test_release_cmd import (
    WROTE_NOTHING, _accept, _commit, _merge, _release, _root_at,
)
from tests.hermes_cli.test_release_ledger import _store_rows

# The fake host's history, oldest first: the checkout's head before any release, then four merges.
HISTORY = [_commit(label) for label in ("checkout", "a", "b", "c", "d")]
CHECKOUT, A, B, C, D = HISTORY
OUTSIDE = {"outcome": "released", "outside": True}


def _contains(ancestor: str, descendant: str) -> bool:
    """The fake ancestry check, shaped like the guards' ``is_ancestor``: a commit is on its own
    history and on that of every later commit."""
    return HISTORY.index(ancestor) <= HISTORY.index(descendant)


def _held(lock: Path) -> bool:
    """Whether another open file holds the flock on ``lock`` at this moment."""
    try:
        fd = os.open(lock, os.O_RDWR)
    except FileNotFoundError:
        return False
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        return True
    finally:
        os.close(fd)
    return False


@dataclass
class SettleHost:
    """Both live adapters in one: the checkout's head and the fake ancestry check. Each read of
    the head notes whether the host lock was held at that moment."""

    lock: Path
    head: str
    held: list[bool] = field(default_factory=list)

    def checkout_head(self) -> str:
        self.held.append(_held(self.lock))
        return self.head

    def is_ancestor(self, ancestor: str, descendant: str) -> bool:
        return _contains(ancestor, descendant)


@pytest.fixture
def root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """The root Hermes home at ``<tmp>/.hermes``, with the release settings the command needs."""
    return _root_at(tmp_path, monkeypatch)


def _settle(monkeypatch: pytest.MonkeyPatch, root: Path, head: str) -> tuple[int, SettleHost]:
    """``hermes release settle`` with the checkout at ``head``, parsed by its own parser before
    anything is patched: its exit code and the host it ran on."""
    from hermes_cli import release_cmd

    parser = argparse.ArgumentParser(prog="hermes")
    build_release_parser(parser.add_subparsers())
    args = parser.parse_args(["release", "settle"])
    host = SettleHost(lock=root / ".release.lock", head=head)
    for adapter in ("LiveHostReader", "ReleaseHostActions"):
        monkeypatch.setattr(release_cmd, adapter, lambda **settings: host)
    return args.func(args), host


def _commits(batch: dict) -> list[str]:
    return [member["merge_commit"] for member in batch["members"]]


def _events(conn) -> list[tuple[int, str, dict]]:
    """Every event in the order it was recorded: its batch, kind and payload."""
    rows = conn.execute("SELECT batch_id, kind, payload FROM release_events ORDER BY id")
    return [(row["batch_id"], row["kind"], json.loads(row["payload"])) for row in rows]


# Test 1: the running commit contains every waiting change.
def test_settle_at_the_second_merge_leaves_no_batch_waiting(root, monkeypatch, capsys):
    from hermes_cli import release_cmd

    with contextlib.closing(ledger.connect()) as conn:
        _merge(conn, A, 1)
        waiting = _merge(conn, B, 2)

    code, host = _settle(monkeypatch, root, B)

    assert code == 0
    assert capsys.readouterr().out.splitlines() == [
        f"Settled 2 changes as released at {B}; 0 changes still waiting."
    ]
    # The head was read under the host lock that run and recover share, and the lock is let go.
    assert host.held == [True] and not (root / ".release.lock").exists()
    with contextlib.closing(ledger.connect()) as conn:
        [emptied, released] = ledger.list_batches(conn)
        assert ledger.last_released(conn) == B
        assert _events(conn) == [(released["batch_id"], "release_finished", OUTSIDE)]
    assert (released["state"], released["outcome"], released["prev"], released["new"]) == (
        "released", "released", None, B,
    )
    assert _commits(released) == [A, B]
    # The batch that waited keeps no change and waits no more.
    assert (emptied["batch_id"], emptied["state"], emptied["members"]) == (
        waiting["batch_id"], "folded", [],
    )
    assert release_cmd.status() == 0
    assert capsys.readouterr().out.splitlines() == [
        "No batch is waiting for a decision.",
        f"Last outcome: batch {released['batch_id']} was released.",
    ]


# Test 2: the running commit contains the first waiting change only.
def test_settle_at_the_first_merge_releases_it_and_the_second_still_waits(
    root, monkeypatch, capsys
):
    with contextlib.closing(ledger.connect()) as conn:
        # A release run shipped A before, so the host ran A when the manual release began.
        _release(conn, _accept(conn, _merge(conn, A, 1)), prev=CHECKOUT, outcome="released")
        _merge(conn, B, 2)
        before = _merge(conn, C, 3)

    code, _host = _settle(monkeypatch, root, B)

    assert code == 0
    assert capsys.readouterr().out.splitlines() == [
        f"Settled 1 change as released at {B}; 1 change still waiting."
    ]
    with contextlib.closing(ledger.connect()) as conn:
        [_shipped, waiting, released] = ledger.list_batches(conn)
        assert ledger.last_released(conn) == B
        assert _events(conn)[-1] == (released["batch_id"], "release_finished", OUTSIDE)
        assert (released["state"], released["outcome"], released["prev"], released["new"]) == (
            "released", "released", A, B,
        )
        assert _commits(released) == [B]
        # C, which B does not contain, still waits in the same batch, one version on, so a
        # decision on the list shown before is stale; the next merge joins it after C.
        assert (waiting["batch_id"], waiting["state"], _commits(waiting)) == (
            before["batch_id"], "open", [C],
        )
        assert waiting["version"] == before["version"] + 1
        assert _commits(_merge(conn, D, 4)) == [C, D]


# Test 3: no settle while a release has begun and not finished.
@pytest.mark.parametrize("unfinished", ["releasing", "failed"])
def test_settle_while_a_release_has_not_finished_refuses_and_writes_nothing(
    root, monkeypatch, capsys, unfinished
):
    with contextlib.closing(ledger.connect()) as conn:
        batch_id = _accept(conn, _merge(conn, A, 1))["batch_id"]
        ledger.begin_release(conn, batch_id, prev=CHECKOUT, new=A)
        if unfinished == "failed":
            ledger.finish_release(conn, batch_id, outcome="failed")
        _merge(conn, B, 2)
        before = _store_rows(conn)
        with pytest.raises(ValueError, match=f"release batch {batch_id} has not finished"):
            ledger.settle_outside_release(conn, running=B, contains=_contains)
        assert _store_rows(conn) == before

    code, _host = _settle(monkeypatch, root, B)

    assert code == 1
    assert capsys.readouterr().out.splitlines() == [
        f"Refused: the waiting changes could not be settled at {B}"
        f" (ValueError: release batch {batch_id} has not finished its release).",
        WROTE_NOTHING,
    ]
    with contextlib.closing(ledger.connect()) as conn:
        assert _store_rows(conn) == before


# Check 4: a release through run still begins and finishes as it does today.
def test_an_accept_begin_and_finish_cycle_behaves_as_today(root, capsys):
    from hermes_cli import release_cmd

    with contextlib.closing(ledger.connect()) as conn:
        _merge(conn, A, 1)
        accepted = _accept(conn, _merge(conn, B, 2))
        began = ledger.begin_release(conn, accepted["batch_id"], prev=CHECKOUT, new=B)
        finished = ledger.finish_release(conn, accepted["batch_id"], outcome="released")
        assert ledger.last_released(conn) == B
        events = _events(conn)
    assert (accepted["state"], began["state"], finished["state"]) == (
        "accepted", "releasing", "released",
    )
    assert (finished["outcome"], finished["prev"], finished["new"]) == ("released", CHECKOUT, B)
    assert _commits(finished) == [A, B]
    assert [(batch, kind) for batch, kind, _payload in events] == [
        (accepted["batch_id"], "release_decided"),
        (accepted["batch_id"], "release_began"),
        (accepted["batch_id"], "release_finished"),
    ]
    assert events[-1][2] == {"outcome": "released"}  # a release through run is not outside
    assert release_cmd.status() == 0
    assert capsys.readouterr().out.splitlines() == [
        "No batch is waiting for a decision.",
        f"Last outcome: batch {accepted['batch_id']} was released.",
    ]
