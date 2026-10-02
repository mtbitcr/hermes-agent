"""``hermes cron realign-times``: preview, apply and restore saved next starts.

Every test runs on a fixed clock -- Tuesday 20 Oct 2026, 12:00 in Vienna --
with ``HERMES_TIMEZONE=Europe/Vienna`` and a temporary store.  The store holds
the owner's jobs with the next starts today's code saved for them (the
monthly check and both Monday jobs an hour late after the autumn change) and
jobs the realignment must leave alone: paused, one-shot, interval, already
correct, and one whose saved next start has already passed.
"""

import argparse
import json
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import pytest

pytest.importorskip("croniter")

import hermes_time
from cron import jobs
from hermes_cli.cron import cron_command
from hermes_cli.subcommands.cron import build_cron_parser

VIENNA = ZoneInfo("Europe/Vienna")
NOW = datetime(2026, 10, 20, 10, 0, tzinfo=timezone.utc)  # Tuesday 12:00 in Vienna
RESTORE_FILE = "realign_times_restore.json"

# name: (schedule, paused, saved next start; None keeps the one create_job saved)
SAVED = {
    "Monthly model check": ("0 10 1 * *", False, "2026-11-01T11:00:00+01:00"),
    "Weekly upstream update summary": ("0 10 * * 1", False, "2026-10-26T11:00:00+01:00"),
    "Project health review": ("0 9 * * 1", False, "2026-10-26T10:00:00+01:00"),
    "Daily backup": ("30 3 * * *", False, "2026-10-21T03:30:00+02:00"),
    "Paused automation one": ("30 2 * * *", True, "2026-10-21T03:30:00+02:00"),
    "Paused automation two": ("every 2h", True, None),
    "Conference reminder": ("2026-10-30T09:00", False, None),
    "Inbox sweep": ("every 30m", False, None),
    "Morning digest": ("0 9 * * *", False, "2026-10-19T09:00:00+02:00"),
}
# The wrong saved starts, each with the start it gets after realignment.
REALIGNED = {
    "Monthly model check": "2026-11-01T10:00:00+01:00",
    "Weekly upstream update summary": "2026-10-26T10:00:00+01:00",
    "Project health review": "2026-10-26T09:00:00+01:00",
}
UNCHANGED = {
    "Daily backup": "already correct",
    "Paused automation one": "paused",
    "Paused automation two": "paused",
    "Conference reminder": "not a time-of-day schedule",
    "Inbox sweep": "not a time-of-day schedule",
    "Morning digest": "no saved next start in the future",
}


class _Store:
    """The temporary profile home, its fixed clock and the records as first saved."""

    def __init__(self, home):
        self.home = home
        self.jobs_file = home / "cron" / "jobs.json"
        self.restore_file = home / "cron" / RESTORE_FILE
        self.now = NOW
        self.original = {}
        self.ids = {}

    def records(self):
        """Saved records by job name, read straight from jobs.json."""
        saved = json.loads(self.jobs_file.read_text(encoding="utf-8"))["jobs"]
        return {record["name"]: record for record in saved}

    def restore_entries(self):
        return json.loads(self.restore_file.read_text(encoding="utf-8"))


@pytest.fixture
def store(tmp_path, monkeypatch):
    home = tmp_path / "home"
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_TIMEZONE", "Europe/Vienna")
    hermes_time.reset_cache()
    state = _Store(home)
    monkeypatch.setattr(jobs, "_hermes_now", lambda: state.now.astimezone(VIENNA))
    # Build the records in a scratch store; the profile's store then holds
    # nothing but jobs.json with today's saved next starts.
    with jobs.use_cron_store(tmp_path / "scratch"):
        for name, (schedule, paused, _saved) in SAVED.items():
            job = jobs.create_job(
                prompt=f"Run {name}",
                schedule=schedule,
                name=name,
                model="test-model",
                provider="test-provider",
            )
            if paused:
                jobs.pause_job(job["id"])
        records = jobs.load_jobs()
    for record in records:
        saved = SAVED[record["name"]][2]
        if saved is not None:
            record["next_run_at"] = saved
    state.jobs_file.parent.mkdir(parents=True)
    state.jobs_file.write_text(
        json.dumps(
            {"jobs": records, "updated_at": NOW.astimezone(VIENNA).isoformat()},
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    state.original = {record["name"]: record for record in records}
    state.ids = {record["name"]: record["id"] for record in records}
    yield state
    hermes_time.reset_cache()


def _dump(value):
    return json.dumps(value, indent=2, ensure_ascii=False)


def _snapshot(root):
    """Every path under ``root`` with its content (files) and modification time."""
    return {
        str(path.relative_to(root)): (
            path.read_bytes() if path.is_file() else None,
            path.stat().st_mtime_ns,
        )
        for path in sorted(root.rglob("*"))
    }


def test_preview_lists_every_job_marks_exactly_the_wrong_starts_and_writes_nothing(store):
    before = _snapshot(store.home)
    rows = jobs.plan_realign_times()
    assert _snapshot(store.home) == before
    assert [row["name"] for row in rows] == list(SAVED)
    for row in rows:
        name = row["name"]
        assert row["id"] == store.ids[name]
        assert row["status"] == ("paused" if SAVED[name][1] else "active")
        assert row["schedule"]
        assert row["before"] == store.original[name]["next_run_at"]
    rows = {row["name"]: row for row in rows}
    assert rows["Monthly model check"]["schedule"] == "0 10 1 * *"
    assert {name: row["after"] for name, row in rows.items() if row["reason"] is None} == REALIGNED
    assert {
        name: row["reason"] for name, row in rows.items() if row["reason"] is not None
    } == UNCHANGED
    assert all(rows[name]["after"] == rows[name]["before"] for name in UNCHANGED)


def test_apply_rewrites_only_the_wrong_starts_and_records_each_change(store):
    entries = [
        {
            "job_id": store.ids[name],
            "name": name,
            "before": SAVED[name][2],
            "after": after,
            "applied_at": "2026-10-20T12:00:00+02:00",
        }
        for name, after in REALIGNED.items()
    ]
    assert jobs.apply_realign_times() == entries
    assert store.restore_entries() == entries
    records = store.records()
    assert list(records) == list(SAVED)
    for name, record in records.items():
        expected = dict(store.original[name])
        if name in REALIGNED:
            expected["next_run_at"] = REALIGNED[name]
        assert _dump(record) == _dump(expected)
    for name in ("Paused automation one", "Paused automation two"):
        assert records[name]["enabled"] is False
        assert records[name]["state"] == "paused"


def test_second_apply_changes_nothing(store):
    jobs.apply_realign_times()
    saved_jobs = store.jobs_file.read_bytes()
    saved_record = store.restore_file.read_bytes()
    assert jobs.apply_realign_times() == []
    assert store.jobs_file.read_bytes() == saved_jobs
    assert store.restore_file.read_bytes() == saved_record
    assert all(row["reason"] is not None for row in jobs.plan_realign_times())


def test_restore_puts_back_the_saved_records(store):
    jobs.apply_realign_times()
    restored, skipped = jobs.restore_realign_times()
    assert sorted(entry["job_id"] for entry in restored) == sorted(
        store.ids[name] for name in REALIGNED
    )
    assert skipped == []
    assert _dump(list(store.records().values())) == _dump(list(store.original.values()))


def test_restore_keeps_a_next_start_that_moved_on_after_apply(store):
    jobs.apply_realign_times()
    store.now = datetime(2026, 10, 26, 8, 2, tzinfo=timezone.utc)  # Monday 09:02 in Vienna
    assert jobs.mark_job_run(store.ids["Project health review"], True)
    ran = store.records()["Project health review"]
    assert ran["next_run_at"] == "2026-11-02T09:00:00+01:00"
    restored, skipped = jobs.restore_realign_times()
    assert [entry["name"] for entry in skipped] == ["Project health review"]
    assert sorted(entry["name"] for entry in restored) == [
        "Monthly model check", "Weekly upstream update summary",
    ]
    records = store.records()
    assert _dump(records["Project health review"]) == _dump(ran)
    for name in ("Monthly model check", "Weekly upstream update summary"):
        assert _dump(records[name]) == _dump(store.original[name])


def test_restore_without_an_apply_changes_nothing(store):
    before = _snapshot(store.home)
    assert jobs.restore_realign_times() == ([], [])
    assert _snapshot(store.home) == before


def test_apply_for_a_chosen_job_changes_only_that_job(store):
    chosen = "Weekly upstream update summary"
    applied = jobs.apply_realign_times([store.ids[chosen]])
    assert [entry["name"] for entry in applied] == [chosen]
    records = store.records()
    assert records[chosen]["next_run_at"] == REALIGNED[chosen]
    for name, record in records.items():
        if name != chosen:
            assert _dump(record) == _dump(store.original[name])
    assert [entry["name"] for entry in store.restore_entries()] == [chosen]
    # A later apply of every job adds the others and keeps the earlier entry.
    jobs.apply_realign_times()
    assert [entry["name"] for entry in store.restore_entries()] == [
        chosen, "Monthly model check", "Project health review",
    ]


def test_apply_for_jobs_that_need_no_change_writes_nothing(store):
    saved_jobs = store.jobs_file.read_bytes()
    chosen = [store.ids["Daily backup"], store.ids["Paused automation one"]]
    assert jobs.apply_realign_times(chosen) == []
    assert store.jobs_file.read_bytes() == saved_jobs
    assert not store.restore_file.exists()


def test_apply_with_an_unknown_job_id_writes_nothing(store):
    saved_jobs = store.jobs_file.read_bytes()
    with pytest.raises(ValueError, match="no-such-job"):
        jobs.apply_realign_times([store.ids["Monthly model check"], "no-such-job"])
    assert store.jobs_file.read_bytes() == saved_jobs
    assert not store.restore_file.exists()


def _parse(*argv):
    parser = argparse.ArgumentParser(prog="hermes")
    build_cron_parser(parser.add_subparsers(dest="command"), cmd_cron=lambda args: args)
    return parser.parse_args(["cron", "realign-times", *argv])


def _cli(capsys, *argv):
    code = cron_command(_parse(*argv))
    return code, capsys.readouterr().out


def test_parser_reads_preview_apply_and_restore():
    preview = _parse()
    assert preview.cron_command == "realign-times"
    assert (preview.apply, preview.restore, preview.job_ids) == (False, False, None)
    apply = _parse("--apply", "--job", "a1b2", "--job", "c3d4")
    assert (apply.apply, apply.restore, apply.job_ids) == (True, False, ["a1b2", "c3d4"])
    restore = _parse("--restore")
    assert (restore.apply, restore.restore, restore.job_ids) == (False, True, None)
    with pytest.raises(SystemExit):
        _parse("--apply", "--restore")


def test_cli_preview_prints_every_job_and_writes_nothing(store, capsys):
    before = _snapshot(store.home)
    code, out = _cli(capsys)
    assert code == 0
    assert _snapshot(store.home) == before
    assert "Europe/Vienna" in out
    for name, job_id in store.ids.items():
        assert name in out
        assert job_id in out
    assert "2026-11-01 11:00 (UTC+01:00)" in out  # the monthly check's saved start
    assert "2026-11-01 10:00 (UTC+01:00)" in out  # ... and after realignment
    assert "will change" in out
    for reason in set(UNCHANGED.values()):
        assert reason in out


def test_cli_preview_says_when_no_time_zone_is_set(store, capsys, monkeypatch):
    monkeypatch.delenv("HERMES_TIMEZONE")
    hermes_time.reset_cache()
    monkeypatch.setattr(jobs, "_hermes_now", lambda: NOW.astimezone())
    before = _snapshot(store.home)
    code, out = _cli(capsys)
    assert code == 0
    assert "server's local time" in out
    assert all(name in out for name in SAVED)
    assert _snapshot(store.home) == before


@pytest.mark.parametrize("argv", [("--job",), ("--restore", "--job")], ids=["preview", "restore"])
def test_cli_accepts_a_job_filter_only_with_apply(store, capsys, argv):
    before = _snapshot(store.home)
    code, out = _cli(capsys, *argv, store.ids["Monthly model check"])
    assert code == 1
    assert "--apply" in out
    assert _snapshot(store.home) == before


def test_cli_applies_a_chosen_job_and_restores_it(store, capsys):
    code, out = _cli(capsys, "--apply", "--job", store.ids["Monthly model check"])
    assert code == 0
    assert "Monthly model check" in out
    assert "2026-11-01 10:00 (UTC+01:00)" in out
    records = store.records()
    assert records["Monthly model check"]["next_run_at"] == REALIGNED["Monthly model check"]
    assert records["Project health review"]["next_run_at"] == SAVED["Project health review"][2]
    code, out = _cli(capsys, "--restore")
    assert code == 0
    assert "Monthly model check" in out
    assert _dump(list(store.records().values())) == _dump(list(store.original.values()))


def test_cli_apply_with_an_unknown_job_id_fails_and_writes_nothing(store, capsys):
    saved_jobs = store.jobs_file.read_bytes()
    code, out = _cli(capsys, "--apply", "--job", "no-such-job")
    assert code == 1
    assert "no-such-job" in out
    assert store.jobs_file.read_bytes() == saved_jobs
    assert not store.restore_file.exists()
