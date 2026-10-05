"""Release record, first build: tests T1-T12 of the release plan (section 4).

Each test reaches the module as ``ledger.<name>``, so a missing name fails that
test on its own instead of the whole file at collection.
"""

from __future__ import annotations

import hashlib
import shutil
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
        pr_url=f"https://github.com/mtbitcr/hermes-agent/pull/{pr}",
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
        "pr_url": "https://github.com/mtbitcr/hermes-agent/pull/1",
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


# Owner rule, second round on pull-request URLs: only the two allowlisted shapes are recorded.
_REFUSED_PR_URLS = (
    # Dot segments: the four earlier ones and three inside the accepted shapes.
    "https://github.com/../hermes/pull/1",
    "https://github.com/acme/../pull/1",
    "https://github.com/./hermes/pull/1",
    "https://github.com/acme/./pull/1",
    "https://github.com/mtbitcr/../pull/1",
    "https://github.com/mtbitcr/./pull/1",
    "https://github.com/mtbitcr/hermes-agent/../raphael-workspace/pull/1",
    # Case variants.
    "https://github.com/MTBITCR/hermes-agent/pull/1",
    "https://github.com/mtbitcr/Hermes-Agent/pull/1",
    "https://github.com/mtbitcr/RAPHAEL-WORKSPACE/pull/1",
    "https://GitHub.com/mtbitcr/hermes-agent/pull/1",
    "HTTPS://github.com/mtbitcr/hermes-agent/pull/1",
    # Unknown repositories and an unknown owner.
    "https://github.com/mtbitcr/hermes/pull/1",
    "https://github.com/mtbitcr/hermes-agent-fork/pull/1",
    "https://github.com/acme/hermes-agent/pull/1",
    # Managed-user owners.
    "https://github.com/mona-cat_octo/hermes-agent/pull/1",
    "https://github.com/mtbitcr_octo/hermes-agent/pull/1",
    # The ".git" suffix, in any case.
    "https://github.com/mtbitcr/hermes-agent.git/pull/1",
    "https://github.com/mtbitcr/hermes-agent.GIT/pull/1",
    "https://github.com/mtbitcr/hermes-agent.Git/pull/1",
    "https://github.com/mtbitcr/raphael-workspace.gIt/pull/1",
    # Other suffixes and forms.
    "https://github.com/mtbitcr/hermes-agent/pull/1/",
    "https://github.com/mtbitcr/hermes-agent/pull/1/files",
    "https://github.com/mtbitcr/hermes-agent/pull/1?w=1",
    "https://github.com/mtbitcr/hermes-agent/pull/1#x",
    "https://github.com/mtbitcr/hermes-agent/pull/1\n",
    " https://github.com/mtbitcr/hermes-agent/pull/1",
    "https://github.com/mtbitcr/hermes-agent/pull/1 ",
    "http://github.com/mtbitcr/hermes-agent/pull/1",
    "https://www.github.com/mtbitcr/hermes-agent/pull/1",
    "https://github.com/mtbitcr/hermes-\u0430gent/pull/1",  # a Cyrillic look-alike "a"
    # Pull numbers.
    "https://github.com/mtbitcr/hermes-agent/pull/0",
    "https://github.com/mtbitcr/hermes-agent/pull/01",
    "https://github.com/mtbitcr/hermes-agent/pull/-1",
    "https://github.com/mtbitcr/hermes-agent/pull/+1",
    "https://github.com/mtbitcr/hermes-agent/pull/",
    "https://github.com/mtbitcr/hermes-agent/pull/1.0",
    "https://github.com/mtbitcr/hermes-agent/pull/\u0661",  # an Arabic-Indic one
    "https://github.com/mtbitcr/hermes-agent/pull/\uff11",  # a full-width one
    # The previous candidate's canonical examples.
    "https://github.com/acme/hermes.cli_x-1/pull/7",
    "https://github.com/Acme/.github/pull/8",
)


@pytest.mark.parametrize("pr_url", _REFUSED_PR_URLS)
def test_pr_url_outside_the_allowlist_is_refused(open_ledger, pr_url):
    conn = open_ledger()
    _merge(conn, "a")
    before = _store_rows(conn)

    with pytest.raises(ValueError):
        ledger.record_merge(conn, **_merge_args("b", pr_url=pr_url), on_main=True)
    assert _store_rows(conn) == before


@pytest.mark.parametrize(
    "pr_url",
    (
        "https://github.com/mtbitcr/hermes-agent/pull/138",
        "https://github.com/mtbitcr/raphael-workspace/pull/1024",
    ),
)
def test_pr_url_on_the_allowlist_is_recorded(open_ledger, pr_url):
    conn = open_ledger()

    readback = ledger.record_merge(conn, **_merge_args("a", pr_url=pr_url), on_main=True)

    assert readback["recorded"] is True
    assert readback["member"]["pr_url"] == pr_url
    [batch] = ledger.list_batches(conn)
    assert [member["pr_url"] for member in batch["members"]] == [pr_url]


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


# The second security review (2026-10-03): the module reads and writes only its own store.
_PUBLIC = ("record_merge", "decide_release", "list_batches", "reminder_due", "record_reminder")
_NOT_OWN_STORE = ("copy", "memory", "attached", "temp", "forged-rows")


def _public_call(name: str, batch):
    """One public function, called as the merge step, the page or the reminder job would."""
    now = _utc(2026, 10, 5, 16, 0)  # 18:00 in Vienna
    return {
        "record_merge": lambda conn: _merge(conn, "elsewhere"),
        "decide_release": lambda conn: _decide(conn, batch, "accepted"),
        "list_batches": ledger.list_batches,
        "reminder_due": lambda conn: ledger.reminder_due(conn, now),
        "record_reminder": lambda conn: ledger.record_reminder(conn, now),
    }[name]


def _claim_own_path(cursor, row):
    """A row factory that reports the own store's path for the main database."""
    if len(row) == 3 and row[1] == "main":
        return (row[0], row[1], str(ledger.ledger_path()))
    return sqlite3.Row(cursor, row)


def _not_own_store(kind: str, tmp_path: Path) -> sqlite3.Connection:
    """A byte copy of the store at another path, an in-memory database, the store
    with another database attached, the store with temporary tables that shadow
    its own, or a copy whose row factory reports the own store's path."""
    if kind in ("copy", "forged-rows"):
        copy = tmp_path / "copy" / ledger.STORE_NAME
        copy.parent.mkdir()
        shutil.copyfile(ledger.ledger_path(), copy)
        conn = sqlite3.connect(str(copy), isolation_level=None)
    elif kind == "memory":
        conn = sqlite3.connect(":memory:", isolation_level=None)
    else:
        conn = ledger.connect()
    if kind == "attached":
        conn.execute("ATTACH DATABASE ? AS other", (str(tmp_path / "other.db"),))
    if kind == "temp":
        names = [
            row[0]
            for row in conn.execute(
                "SELECT name FROM main.sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite%'"
            )
        ]
        for name in names:
            conn.execute(f'CREATE TEMP TABLE "{name}" AS SELECT * FROM main."{name}" WHERE 0')
    conn.row_factory = _claim_own_path if kind == "forged-rows" else sqlite3.Row
    return conn


@pytest.mark.parametrize("kind", _NOT_OWN_STORE)
@pytest.mark.parametrize("name", _PUBLIC)
def test_connection_other_than_the_own_store_is_refused(open_ledger, tmp_path, name, kind):
    own = open_ledger()
    batch = _merge(own, "a")["batch"]
    own_before = _store_rows(own)
    other = _not_own_store(kind, tmp_path)
    try:
        other_before = _store_rows(other)
        with pytest.raises(PermissionError):
            _public_call(name, batch)(other)
        assert _store_rows(other) == other_before
    finally:
        other.close()
    assert _store_rows(own) == own_before


class _Disguised(str):
    """Text whose characters pass a check while SQLite stores another value."""

    def __conform__(self, protocol):
        return self.stored


def _disguised(text: str, stored: str) -> str:
    value = _Disguised(text)
    value.stored = stored
    return value


# The third security review (2026-10-03): only exact str values are accepted.
@pytest.mark.parametrize("case", ("decision", "shown_digest", "pr_url"))
def test_text_that_is_not_exactly_str_is_refused(open_ledger, case):
    conn = open_ledger()
    batch = _merge(conn, "a")["batch"]
    before = _store_rows(conn)

    with pytest.raises(ValueError):
        if case == "pr_url":
            ledger.record_merge(
                conn,
                merge_commit=_commit("b"),
                pr_url=_disguised("https://github.com/mtbitcr/hermes-agent/pull/2", "https://example.invalid/pull/2"),
                reviewed_base=_commit("b-base"),
                reviewed_head=_commit("b-head"),
                reviewed_tree=_commit("b-tree"),
                tier=1,
                card_id="t_0badc0de",
                on_main=True,
            )
        else:
            ledger.decide_release(
                conn,
                batch["batch_id"],
                decision=_disguised("deferred", "rejected") if case == "decision" else "accepted",
                shown_digest=_disguised(batch["digest"], "f" * 64) if case == "shown_digest" else batch["digest"],
                expected_version=batch["version"],
                decision_ref=PAGE_REF,
            )

    assert _store_rows(conn) == before


# Card 1 of the release re-cut: each release's outcome, and one waiting decision (S-B, S-G).
LIVE = _commit("live")  # the checkout's head while no batch was ever released


def _accepted(conn, *labels: str):
    """Merge the changes, then accept the waiting batch they joined."""
    for label in labels:
        batch = _merge(conn, label)["batch"]
    return _decide(conn, batch, "accepted")


def _begin(conn, batch, *, prev: str = LIVE):
    """Begin the batch's release with NEW at its last member's merge commit (S-B)."""
    return ledger.begin_release(conn, batch["batch_id"], prev=prev, new=_commits(batch)[-1])


def _finish(conn, batch, outcome):
    return ledger.finish_release(conn, batch["batch_id"], outcome=outcome)


def _waiting(conn) -> list[dict]:
    return [batch for batch in ledger.list_batches(conn) if batch["state"] in ("open", "deferred")]


def test_begin_release_moves_an_accepted_batch_to_releasing_with_both_versions(open_ledger):
    conn = open_ledger()
    accepted = _accepted(conn, "a", "b")
    other = _accepted(conn, "c")
    waiting = _merge(conn, "d")["batch"]
    with pytest.raises(ValueError):
        _begin(conn, waiting)
    put_off = _decide(conn, waiting, "deferred")
    before = _store_rows(conn)
    with pytest.raises(ValueError):
        _begin(conn, put_off)
    # NEW is the merge commit of the batch's last member: not an earlier one, not another batch's.
    for new in (_commit("a"), _commit("c"), _disguised(_commit("b"), _commit("a"))):
        with pytest.raises(ValueError):
            ledger.begin_release(conn, accepted["batch_id"], prev=LIVE, new=new)
    assert _store_rows(conn) == before

    releasing = _begin(conn, accepted)

    assert releasing["state"] == "releasing"
    assert (releasing["prev"], releasing["new"]) == (LIVE, _commit("b"))
    assert _commits(releasing) == _commits(accepted)
    assert ledger.list_batches(conn)[0] == releasing
    # One release at a time: while one is releasing, no batch begins, that one included.
    before = _store_rows(conn)
    for batch in (other, releasing):
        with pytest.raises(ValueError):
            _begin(conn, batch)
    assert _store_rows(conn) == before


def test_released_batch_is_final_and_names_the_live_version(open_ledger):
    conn = open_ledger()
    assert ledger.last_released(conn) is None  # so PREV is the checkout's head
    releasing = _begin(conn, _accepted(conn, "a", "b"))

    released = _finish(conn, releasing, "released")

    assert (released["state"], released["outcome"]) == ("released", "released")
    assert ledger.last_released(conn) == _commit("b")
    before = _store_rows(conn)
    for outcome in ("released", "restored", "refused", "failed"):
        with pytest.raises(ValueError):
            _finish(conn, released, outcome)
    with pytest.raises(ValueError):
        ledger.begin_recovery(conn, released["batch_id"])
    assert _store_rows(conn) == before
    following = _merge(conn, "c")["batch"]
    assert (following["state"], _commits(following)) == ("open", [_commit("c")])
    assert ledger.list_batches(conn)[0] == released
    # The next release starts from the live version (S-B), then names its own.
    accepted = _decide(conn, following, "accepted")
    with pytest.raises(ValueError):
        _begin(conn, accepted, prev=LIVE)
    _finish(conn, _begin(conn, accepted, prev=_commit("b")), "released")
    assert ledger.last_released(conn) == _commit("c")


@pytest.mark.parametrize("outcome", ("restored", "refused"))
def test_restored_or_refused_changes_return_to_the_one_waiting_decision(open_ledger, outcome):
    conn = open_ledger()
    releasing = _begin(conn, _accepted(conn, "a", "b"))
    _merge(conn, "c", tier=0)
    newer = _merge(conn, "d", tier=0)["batch"]  # merged while the release ran

    folded = _finish(conn, releasing, outcome)

    assert (folded["state"], folded["outcome"], folded["members"]) == ("folded", outcome, [])
    [waiting] = _waiting(conn)
    assert (waiting["batch_id"], waiting["state"]) == (newer["batch_id"], "open")
    assert _commits(waiting) == [_commit("a"), _commit("b"), _commit("c"), _commit("d")]
    assert waiting["version"] > newer["version"]
    assert waiting["digest"] != newer["digest"]
    assert waiting["tier"] == 1  # the returned changes bring their tier back
    commits = [commit for batch in ledger.list_batches(conn) for commit in _commits(batch)]
    assert len(commits) == len(set(commits)) == 4
    # A page drawn before the changes came back is stale; the new list can be decided.
    with pytest.raises(ValueError):
        _decide(conn, newer, "accepted")
    assert _decide(conn, waiting, "accepted")["state"] == "accepted"


def test_failed_release_stays_open_for_recovery(open_ledger):
    conn = open_ledger()
    accepted = _accepted(conn, "a", "b")
    before = _store_rows(conn)
    # A release that never began neither recovers nor ends.
    with pytest.raises(ValueError):
        ledger.begin_recovery(conn, accepted["batch_id"])
    for outcome in ("released", "failed"):
        with pytest.raises(ValueError):
            _finish(conn, accepted, outcome)
    assert _store_rows(conn) == before
    releasing = _begin(conn, accepted)
    # A run killed while releasing recovers from there.
    assert ledger.begin_recovery(conn, releasing["batch_id"])["recovery_attempts"] == 1

    failed = _finish(conn, releasing, "failed")

    assert (failed["state"], failed["outcome"], failed["recovery_attempts"]) == ("failed", "failed", 1)
    other = _accepted(conn, "c")
    with pytest.raises(ValueError):
        _begin(conn, other)  # the failed release still holds the host
    for attempt in (2, 3):
        assert ledger.begin_recovery(conn, failed["batch_id"])["recovery_attempts"] == attempt
    assert _finish(conn, failed, "failed")["state"] == "failed"
    before = _store_rows(conn)
    # Refused says the host was never touched, which a failed release cannot claim.
    for outcome in ("refused", "rolled back", _disguised("failed", "released")):
        with pytest.raises(ValueError):
            _finish(conn, failed, outcome)
    assert _store_rows(conn) == before

    restored = _finish(conn, failed, "restored")

    assert (restored["state"], restored["outcome"], restored["recovery_attempts"]) == ("folded", "restored", 3)
    with pytest.raises(ValueError):
        ledger.begin_recovery(conn, restored["batch_id"])
    # No batch was waiting, so the changes open a new one.
    [waiting] = _waiting(conn)
    assert (waiting["state"], _commits(waiting)) == ("open", [_commit("a"), _commit("b")])
    # Released in merge order, the older batch last: the live version is the last one released.
    _finish(conn, _begin(conn, _decide(conn, waiting, "accepted")), "released")
    _finish(conn, _begin(conn, other, prev=_commit("b")), "released")
    assert ledger.last_released(conn) == _commit("c")


# The three-state schema on main before card 1, as the first build created it.
_FIRST_SCHEMA = (
    "CREATE TABLE release_batches (id INTEGER PRIMARY KEY AUTOINCREMENT,"
    " state TEXT NOT NULL CHECK (state IN ('open', 'deferred', 'accepted')), version INTEGER NOT NULL,"
    " tier INTEGER NOT NULL CHECK (tier IN (0, 1, 2)), created_at INTEGER NOT NULL, decided_at INTEGER,"
    " decided_by TEXT, decision_ref TEXT, shown_digest TEXT);"
    "CREATE TABLE release_members (batch_id INTEGER NOT NULL REFERENCES release_batches(id),"
    " merge_commit TEXT NOT NULL UNIQUE, pr_url TEXT NOT NULL, reviewed_base TEXT NOT NULL,"
    " reviewed_head TEXT NOT NULL, reviewed_tree TEXT NOT NULL, tier INTEGER NOT NULL CHECK (tier IN (0, 1, 2)),"
    " tier_recorded INTEGER NOT NULL CHECK (tier_recorded IN (0, 1)), card_id TEXT NOT NULL,"
    " merged_at INTEGER NOT NULL, position INTEGER NOT NULL, UNIQUE (batch_id, position));"
    "CREATE TABLE release_events (id INTEGER PRIMARY KEY AUTOINCREMENT,"
    " batch_id INTEGER NOT NULL REFERENCES release_batches(id), kind TEXT NOT NULL, payload TEXT NOT NULL,"
    " created_at INTEGER NOT NULL);"
    "CREATE TABLE release_reminders (vienna_date TEXT NOT NULL UNIQUE, recorded_at INTEGER NOT NULL);"
)


def _first_schema_store(path: Path) -> dict[str, tuple[list[str], list[tuple]]]:
    """A store made with the first schema, holding an accepted and a put-off batch;
    returns each table's columns and rows."""
    path.parent.mkdir(parents=True, exist_ok=True)
    first = sqlite3.connect(path)
    try:
        first.executescript(_FIRST_SCHEMA)
        with first:
            first.executemany(
                "INSERT INTO release_batches VALUES (?, ?, ?, ?, 1759500000, 1759500100, 'owner', ?, ?)",
                [(1, "accepted", 3, 1, PAGE_REF, "a" * 64), (2, "deferred", 2, 2, PAGE_REF, "b" * 64)],
            )
            first.executemany(
                "INSERT INTO release_members VALUES (?, ?, 'https://github.com/mtbitcr/hermes-agent/pull/1',"
                " ?, ?, ?, ?, ?, 't_0badc0de', 1759500000, ?)",
                [
                    (batch, _commit(label), _commit(f"{label}-base"), _commit(f"{label}-head"),
                     _commit(f"{label}-tree"), tier, int(tier < 2), position)
                    for batch, label, tier, position in ((1, "a", 1, 1), (1, "b", 0, 2), (2, "c", 2, 1))
                ],
            )
            first.execute(
                "INSERT INTO release_events (batch_id, kind, payload, created_at)"
                " VALUES (1, 'release_decided', '{}', 1759500100)"
            )
            first.execute("INSERT INTO release_reminders VALUES ('2026-10-03', 1759507200)")
        tables = {}
        for name in ("release_batches", "release_members", "release_events", "release_reminders"):
            cursor = first.execute(f"SELECT * FROM {name}")
            tables[name] = ([column[0] for column in cursor.description], cursor.fetchall())
        return tables
    finally:
        first.close()


def test_store_created_with_the_first_schema_is_upgraded(open_ledger):
    first = _first_schema_store(ledger.ledger_path())

    conn = open_ledger()

    for name, (columns, rows) in first.items():
        kept = conn.execute(f"SELECT {', '.join(columns)} FROM {name}").fetchall()
        assert [tuple(row) for row in kept] == rows, name
    accepted, put_off = ledger.list_batches(conn)
    assert (accepted["state"], put_off["state"]) == ("accepted", "deferred")
    assert (accepted["prev"], accepted["new"], accepted["outcome"], accepted["recovery_attempts"]) == (
        None, None, None, 0,
    )
    assert {(member["title"], member["board"]) for member in accepted["members"] + put_off["members"]} == {("", "")}
    assert conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1
    # The new states now hold, and opening the store again changes nothing.
    _finish(conn, _finish(conn, _begin(conn, accepted), "failed"), "restored")
    upgraded = _store_rows(conn)
    assert _store_rows(open_ledger()) == upgraded
    assert _commits(_waiting(conn)[0]) == [_commit("a"), _commit("b"), _commit("c")]


def test_title_and_board_are_kept_when_given(open_ledger):
    conn = open_ledger()
    title = "Show the release decision on the Decisions page"

    titled = ledger.record_merge(conn, **_merge_args("a"), title=title, board="hermes-agent", on_main=True)
    plain = _merge(conn, "b")  # an existing caller, with neither

    assert (titled["member"]["title"], titled["member"]["board"]) == (title, "hermes-agent")
    assert (plain["member"]["title"], plain["member"]["board"]) == ("", "")
    [batch] = ledger.list_batches(conn)
    assert [(member["title"], member["board"]) for member in batch["members"]] == [(title, "hermes-agent"), ("", "")]
    # Only text is kept, and a board is a kanban board slug; anything else writes nothing.
    before = _store_rows(conn)
    for changes in (
        {"title": 7},
        {"title": _disguised(title, "t_0badc0de")},
        {"board": "Hermes Agent"},
        {"board": "../main"},
        {"board": None},
    ):
        with pytest.raises(ValueError):
            ledger.record_merge(conn, **_merge_args("c", **changes), on_main=True)
    assert _store_rows(conn) == before


# The security reviews' rule holds for card 1's steps too: they work only on the own store.
@pytest.mark.parametrize("kind", _NOT_OWN_STORE)
def test_release_steps_refuse_a_connection_other_than_the_own_store(open_ledger, tmp_path, kind):
    own = open_ledger()
    batch = _accepted(own, "a")
    own_before = _store_rows(own)
    other = _not_own_store(kind, tmp_path)
    try:
        other_before = _store_rows(other)
        for step in (
            lambda conn: _begin(conn, batch),
            lambda conn: ledger.begin_recovery(conn, batch["batch_id"]),
            lambda conn: _finish(conn, batch, "released"),
            ledger.last_released,
        ):
            with pytest.raises(PermissionError):
                step(other)
        assert _store_rows(other) == other_before
    finally:
        other.close()
    assert _store_rows(own) == own_before
