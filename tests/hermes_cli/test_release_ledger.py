"""Release record, first build: tests T1-T12 of the release plan (section 4).

Each test reaches the module as ``ledger.<name>``, so a missing name fails that
test on its own instead of the whole file at collection.
"""

from __future__ import annotations

import hashlib
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pytest

from hermes_cli import release_ledger as ledger

PAGE_REF = "decisions-page:release"


@pytest.fixture
def root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A Hermes root at ``<tmp>/.hermes``; nothing can resolve to the real home."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    return home


@pytest.fixture
def open_ledger(root: Path):
    opened: list[sqlite3.Connection] = []

    def _open() -> sqlite3.Connection:
        conn = ledger.connect()
        opened.append(conn)
        return conn

    yield _open
    for conn in opened:
        conn.close()


def _commit(label: str) -> str:
    """A 40-hex id shaped like a git commit."""
    return hashlib.sha1(label.encode()).hexdigest()


def _utc(*moment: int) -> datetime:
    return datetime(*moment, tzinfo=timezone.utc)


def _merge(conn, label: str, *, tier=1, pr: int = 1):
    return ledger.record_merge(
        conn,
        merge_commit=_commit(label),
        pr_url=f"https://github.com/acme/hermes/pull/{pr}",
        reviewed_base=_commit(f"{label}-base"),
        reviewed_head=_commit(f"{label}-head"),
        reviewed_tree=_commit(f"{label}-tree"),
        tier=tier,
        card_id=f"t_{_commit(label)[:8]}",
        on_main=True,
    )


def _decide(conn, batch, decision: str):
    return ledger.decide_release(
        conn,
        batch["batch_id"],
        decision=decision,
        shown_digest=batch["digest"],
        expected_version=batch["version"],
        decision_ref=PAGE_REF,
    )


def _commits(batch) -> list[str]:
    return [member["merge_commit"] for member in batch["members"]]


def _store_rows(conn) -> dict[str, list[tuple]]:
    """Every row of every table, to show that a refused call wrote nothing."""
    names = [
        row[0]
        for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table' ORDER BY name")
    ]
    return {name: [tuple(row) for row in conn.execute(f'SELECT * FROM "{name}"')] for name in names}


# T1
def test_one_open_decision_grows_with_each_merge(open_ledger):
    conn = open_ledger()
    readbacks = [_merge(conn, label)["batch"] for label in ("a", "b", "c")]

    [batch] = ledger.list_batches(conn)
    assert batch["state"] == "open"
    assert _commits(batch) == [_commit("a"), _commit("b"), _commit("c")]
    assert {readback["batch_id"] for readback in readbacks} == {batch["batch_id"]}
    assert [len(readback["members"]) for readback in readbacks] == [1, 2, 3]
    versions = [readback["version"] for readback in readbacks]
    assert versions[0] < versions[1] < versions[2] == batch["version"]


# T2
def test_same_merge_recorded_twice_is_one_member(open_ledger):
    conn = open_ledger()
    first = _merge(conn, "a")
    before = _store_rows(conn)

    # A repeat, even with other details, returns the existing row and writes nothing.
    again = _merge(conn, "a", tier=2, pr=9)

    assert first["recorded"] is True
    assert again["recorded"] is False
    assert again["member"] == first["member"]
    assert _store_rows(conn) == before
    [batch] = ledger.list_batches(conn)
    assert _commits(batch) == [_commit("a")]
    assert batch["version"] == first["batch"]["version"]


# T3
def test_accept_releases_exactly_the_batch_shown(open_ledger):
    conn = open_ledger()
    _merge(conn, "a")
    shown = _merge(conn, "b")["batch"]  # the page is drawn at 2 members
    _merge(conn, "c")  # then a third merge lands
    [current] = ledger.list_batches(conn)
    before = _store_rows(conn)

    for version in (shown["version"], current["version"]):
        with pytest.raises(ValueError):
            ledger.decide_release(
                conn,
                shown["batch_id"],
                decision="accepted",
                shown_digest=shown["digest"],
                expected_version=version,
                decision_ref=PAGE_REF,
            )
    assert _store_rows(conn) == before
    assert ledger.list_batches(conn) == [current]

    accepted = _decide(conn, current, "accepted")

    assert accepted["state"] == "accepted"
    assert _commits(accepted) == [_commit("a"), _commit("b"), _commit("c")]
    assert accepted["digest"] == current["digest"]
    [frozen] = ledger.list_batches(conn)
    assert frozen["state"] == "accepted"
    assert _commits(frozen) == _commits(current)


# T4
def test_merge_after_acceptance_goes_to_next_decision(open_ledger):
    conn = open_ledger()
    _merge(conn, "a")
    accepted = _decide(conn, _merge(conn, "b")["batch"], "accepted")

    following = _merge(conn, "c")["batch"]

    first, second = ledger.list_batches(conn)
    assert first == accepted
    assert _commits(first) == [_commit("a"), _commit("b")]
    assert second["batch_id"] == following["batch_id"] != first["batch_id"]
    assert second["state"] == "open"
    assert _commits(second) == [_commit("c")]


# T5
def test_put_off_holds_the_batch(open_ledger):
    conn = open_ledger()
    _merge(conn, "a")
    shown = _merge(conn, "b")["batch"]

    put_off = _decide(conn, shown, "deferred")

    assert put_off["state"] == "deferred"
    batches = ledger.list_batches(conn)
    assert [batch["batch_id"] for batch in batches] == [shown["batch_id"]]
    assert batches[0]["state"] == "deferred"
    assert not [batch for batch in batches if batch["state"] == "accepted"]

    accepted = _decide(conn, batches[0], "accepted")

    assert accepted["state"] == "accepted"
    assert _commits(accepted) == [_commit("a"), _commit("b")]


# T6
def test_put_off_decision_keeps_growing(open_ledger):
    conn = open_ledger()
    put_off = _decide(conn, _merge(conn, "a")["batch"], "deferred")

    grown = _merge(conn, "b")["batch"]

    # Owner decision 2: a put-off batch keeps growing and stays put off.
    assert grown["batch_id"] == put_off["batch_id"]
    assert grown["state"] == "deferred"
    assert _commits(grown) == [_commit("a"), _commit("b")]
    assert grown["version"] > put_off["version"]
    assert ledger.list_batches(conn) == [grown]


# T7
def test_batch_takes_highest_tier(root, tmp_path, monkeypatch):
    cases = [((0, 1), 1), ((0, 2, 1), 2), ((0,), 0), ((None,), 2)]
    for index, (tiers, expected) in enumerate(cases):
        monkeypatch.setenv("HERMES_KANBAN_HOME", str(tmp_path / f"tiers-{index}"))
        conn = ledger.connect()
        try:
            for position, tier in enumerate(tiers):
                readback = _merge(conn, f"{index}-{position}", tier=tier)
            assert readback["batch"]["tier"] == expected, tiers
        finally:
            conn.close()
    # The missing tier is stored as tier 2 and marked as not recorded.
    assert readback["member"]["tier"] == 2
    assert readback["member"]["tier_recorded"] is False


# T8
def test_worker_context_cannot_decide(open_ledger, monkeypatch):
    conn = open_ledger()
    shown = _merge(conn, "a")["batch"]
    before = _store_rows(conn)
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_0badc0de")

    for decision in ("accepted", "deferred"):
        with pytest.raises(PermissionError):
            _decide(conn, shown, decision)

    assert _store_rows(conn) == before


# T9
def test_ledger_store_resolves_root_under_worker_env(root, monkeypatch):
    operator = ledger.connect()
    try:
        _merge(operator, "from-operator")
    finally:
        operator.close()

    # Exactly what the dispatcher injects into a worker it spawns.
    profile_home = root / "profiles" / "worker"
    board_dir = root / "kanban" / "boards" / "worker-board"
    profile_home.mkdir(parents=True)
    board_dir.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(profile_home))
    monkeypatch.setenv("HERMES_PROFILE", "worker")
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_0badc0de")
    monkeypatch.setenv("HERMES_KANBAN_DB", str(board_dir / "kanban.db"))
    monkeypatch.setenv("HERMES_KANBAN_BOARD", "worker-board")
    monkeypatch.setenv("HERMES_KANBAN_WORKSPACES_ROOT", str(board_dir / "workspaces"))

    worker = ledger.connect()
    try:
        _merge(worker, "from-worker-1")
        readback = _merge(worker, "from-worker-2")
    finally:
        worker.close()

    # Independent count, read straight from the one root store.
    expected_store = root / "kanban" / "release_ledger.db"
    independent = sqlite3.connect(expected_store.as_uri() + "?mode=ro", uri=True)
    try:
        (stored,) = independent.execute("SELECT COUNT(*) FROM release_members").fetchone()
    finally:
        independent.close()
    assert stored == len(readback["batch"]["members"]) == 3
    assert _commits(readback["batch"])[0] == _commit("from-operator")
    assert not [path for path in profile_home.rglob("*") if "release" in path.name]
    assert not [path for path in board_dir.rglob("*") if "release" in path.name]


# T10
def test_reminder_due_1800_vienna_summer_and_winter(monkeypatch):
    # The rule fixes Europe/Vienna itself; a profile or host zone must not move it.
    monkeypatch.setenv("HERMES_TIMEZONE", "America/New_York")

    def due(*moment: int) -> bool:
        return ledger.reminder_due_at(_utc(*moment), waiting=True, last_reminder_day=None)

    assert due(2026, 7, 15, 16, 0)  # 18:00 CEST
    assert due(2026, 7, 15, 16, 59)  # 18:59 CEST
    assert not due(2026, 7, 15, 17, 0)  # 19:00 CEST
    assert due(2026, 1, 15, 17, 0)  # 18:00 CET
    assert not due(2026, 1, 15, 16, 0)  # 17:00 CET
    # Last Sunday of March 2026: CEST from 01:00 UTC, so 18:00 Vienna is 16:00 UTC.
    assert due(2026, 3, 29, 16, 0)
    assert not due(2026, 3, 29, 17, 0)
    # Last Sunday of October 2026: CET from 01:00 UTC, so 18:00 Vienna is 17:00 UTC.
    assert due(2026, 10, 25, 17, 0)
    assert not due(2026, 10, 25, 16, 0)


# T11
def test_no_reminder_when_nothing_waits(open_ledger):
    conn = open_ledger()
    vienna_six_pm = _utc(2026, 7, 15, 16, 0)

    assert ledger.reminder_due_at(vienna_six_pm, waiting=False, last_reminder_day=None) is False
    assert ledger.reminder_due(conn, vienna_six_pm) is False
    # An accepted batch is frozen and no longer waits for a decision.
    _decide(conn, _merge(conn, "a")["batch"], "accepted")
    assert ledger.reminder_due(conn, vienna_six_pm) is False


# T12
def test_one_reminder_per_day(open_ledger):
    conn = open_ledger()
    shown = _merge(conn, "a")["batch"]
    first_day = _utc(2026, 7, 15, 16, 0)

    assert ledger.reminder_due(conn, first_day) is True
    assert ledger.record_reminder(conn, first_day) == "2026-07-15"
    assert ledger.reminder_due(conn, first_day) is False
    assert ledger.reminder_due(conn, _utc(2026, 7, 15, 16, 45)) is False

    # Owner decision 3: one reminder a day while a batch waits, a put-off batch included.
    _decide(conn, shown, "deferred")
    assert ledger.reminder_due(conn, _utc(2026, 7, 16, 15, 0)) is False  # 17:00 Vienna
    assert ledger.reminder_due(conn, _utc(2026, 7, 16, 16, 0)) is True


# K2 refusals that T1-T12 leave open.
def test_decision_refuses_reject_and_an_accepted_batch(open_ledger):
    conn = open_ledger()
    shown = _merge(conn, "a")["batch"]
    before = _store_rows(conn)

    # Owner decision 8: a release decision offers no Reject.
    with pytest.raises(ValueError):
        _decide(conn, shown, "rejected")
    assert _store_rows(conn) == before

    accepted = _decide(conn, shown, "accepted")
    after_accept = _store_rows(conn)
    for decision in ("accepted", "deferred"):
        with pytest.raises(ValueError):
            _decide(conn, accepted, decision)
    assert _store_rows(conn) == after_accept


# The rework: the owner's rule on the review findings.
def _merge_args(label: str, **changes) -> dict:
    """record_merge's keyword arguments for one merge, without the main confirmation."""
    args = {
        "merge_commit": _commit(label),
        "pr_url": "https://github.com/acme/hermes/pull/1",
        "reviewed_base": _commit(f"{label}-base"),
        "reviewed_head": _commit(f"{label}-head"),
        "reviewed_tree": _commit(f"{label}-tree"),
        "tier": 1,
        "card_id": f"t_{_commit(label)[:8]}",
    }
    args.update(changes)
    return args


# Owner rule 1: record_merge relies on the merge step's confirmation that the commit is on main.
def test_merge_not_confirmed_on_main_is_refused(open_ledger):
    conn = open_ledger()
    merge = _merge_args("a")
    before = _store_rows(conn)

    with pytest.raises(ValueError):
        ledger.record_merge(conn, **merge)  # no confirmation
    for on_main in (False, None, 1, "true"):
        with pytest.raises(ValueError):
            ledger.record_merge(conn, **merge, on_main=on_main)
    assert _store_rows(conn) == before

    # Confirmed, the same merge is recorded; a repeat still needs the confirmation.
    assert ledger.record_merge(conn, **merge, on_main=True)["recorded"] is True
    recorded = _store_rows(conn)
    with pytest.raises(ValueError):
        ledger.record_merge(conn, **merge, on_main=False)
    assert _store_rows(conn) == recorded


# Owner rule 3: only the exact, canonical pull-request URL is recorded.
def test_non_canonical_pr_url_is_refused(open_ledger):
    conn = open_ledger()
    _merge(conn, "a")
    before = _store_rows(conn)
    refused = (
        "https://github.com/../hermes/pull/1",
        "https://github.com/acme/../pull/1",
        "https://github.com/./hermes/pull/1",
        "https://github.com/acme/./pull/1",
        # Owner: a dot, an underscore, a leading, a trailing and a doubled hyphen, 40 characters.
        "https://github.com/ac.me/hermes/pull/1",
        "https://github.com/ac_me/hermes/pull/1",
        "https://github.com/-acme/hermes/pull/1",
        "https://github.com/acme-/hermes/pull/1",
        "https://github.com/ac--me/hermes/pull/1",
        f"https://github.com/{'a' * 40}/hermes/pull/1",
        # Repository: GitHub's ".git" alias and 101 characters ("." and ".." are above).
        "https://github.com/acme/hermes.git/pull/1",
        f"https://github.com/acme/{'r' * 101}/pull/1",
    )

    for url in refused:
        with pytest.raises(ValueError):
            ledger.record_merge(conn, **_merge_args("b", pr_url=url), on_main=True)
    assert _store_rows(conn) == before

    canonical = (
        "https://github.com/acme/hermes.cli_x-1/pull/7",
        "https://github.com/Acme/.github/pull/8",  # the case stays as given
        f"https://github.com/{'a-' * 19}a/{'r' * 100}/pull/9",  # 39 and 100 characters
    )
    for label, url in zip(("c", "d", "e"), canonical):
        readback = ledger.record_merge(conn, **_merge_args(label, pr_url=url), on_main=True)
        assert readback["recorded"] is True
        assert readback["member"]["pr_url"] == url


# Owner rule 2: the review's concurrent-read regression.
def test_readback_is_one_snapshot_under_a_concurrent_merge(open_ledger):
    reader, writer = open_ledger(), open_ledger()
    _merge(writer, "a", tier=0)
    # The store keeps SQLite's rollback journal: the writer cannot commit while the reader
    # holds a snapshot. Without a busy wait that refusal is immediate, never the timeout.
    writer.execute("PRAGMA busy_timeout = 0")
    attempts: list[object] = []

    def merge_before_members_select(statement: str) -> None:
        # Once, just before the reader's SELECT of the members; touches only the writer.
        if attempts or "FROM release_members" not in statement:
            return
        try:
            attempts.append(_merge(writer, "b", tier=2)["recorded"])
        except sqlite3.OperationalError as refused:  # the reader's snapshot holds it off
            attempts.append(refused)

    reader.set_trace_callback(merge_before_members_select)
    try:
        [batch] = ledger.list_batches(reader)
    finally:
        reader.set_trace_callback(None)

    assert len(attempts) == 1
    assert batch["tier"] == max(member["tier"] for member in batch["members"])
    commits = _commits(batch)
    assert batch["digest"] == hashlib.sha256("".join(f"{c}\n" for c in commits).encode()).hexdigest()
    assert batch["version"] == len(commits)  # no decision was made
    # The readback ended its snapshot, so the merge lands now if it had not already.
    assert not reader.in_transaction
    _merge(writer, "b", tier=2)
    [current] = ledger.list_batches(reader)
    assert _commits(current) == [_commit("a"), _commit("b")]
    assert (current["tier"], current["version"]) == (2, 2)
