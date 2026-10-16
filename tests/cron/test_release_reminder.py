"""Tests for cron/scripts/release_reminder.py: the evening release reminder."""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path

import pytest

from hermes_cli import release_ledger as ledger

LINE = "A release is waiting for your decision on the Decisions page.\n"

# 16:00 UTC is 18:00 in Vienna in summer (CEST, UTC+2).
SIX_PM_VIENNA = datetime(2026, 7, 15, 16, 0, tzinfo=timezone.utc)


@pytest.fixture
def root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    return home


def _commit(label: str) -> str:
    return hashlib.sha1(label.encode()).hexdigest()


def _open_batch() -> None:
    conn = ledger.connect()
    try:
        ledger.record_merge(
            conn,
            merge_commit=_commit("a"),
            pr_url="https://github.com/mtbitcr/hermes-agent/pull/1",
            reviewed_base=_commit("a-base"),
            reviewed_head=_commit("a-head"),
            reviewed_tree=_commit("a-tree"),
            tier=1,
            card_id="t_" + _commit("a")[:8],
            on_main=True,
        )
    finally:
        conn.close()


def _reminder_days() -> list[str]:
    conn = ledger.connect()
    try:
        return [row[0] for row in conn.execute("SELECT vienna_date FROM release_reminders ORDER BY vienna_date")]
    finally:
        conn.close()


def _run(now: datetime) -> int:
    from cron.scripts import release_reminder

    return release_reminder.main(now=now)


def test_due_reminder_prints_one_line_and_records_the_vienna_day(root, capsys):
    _open_batch()

    assert _run(SIX_PM_VIENNA) == 0

    captured = capsys.readouterr()
    assert captured.out == LINE
    assert captured.err == ""
    assert _reminder_days() == ["2026-07-15"]


def test_nothing_waiting_prints_nothing(root, capsys):
    assert _run(SIX_PM_VIENNA) == 0

    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""
    assert _reminder_days() == []


def test_second_run_on_the_same_vienna_day_prints_nothing(root, capsys):
    _open_batch()

    assert _run(SIX_PM_VIENNA) == 0
    assert capsys.readouterr().out == LINE

    assert _run(SIX_PM_VIENNA) == 0
    assert capsys.readouterr().out == ""
    assert _reminder_days() == ["2026-07-15"]


@pytest.mark.parametrize("failure", ["corrupt-file", "unreadable-query"])
def test_unreadable_record_prints_nothing_and_exits_zero(root, capsys, monkeypatch, failure):
    _open_batch()
    if failure == "corrupt-file":
        ledger.ledger_path().write_bytes(b"this is not a sqlite database" * 100)
    else:

        def _broken(conn, now):
            raise RuntimeError("record cannot be read")

        monkeypatch.setattr(ledger, "reminder_due", _broken)

    assert _run(SIX_PM_VIENNA) == 0

    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""
