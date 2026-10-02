"""``hermes cron realign-times``: a preview of saved next starts that writes nothing.

Every test runs on a fixed clock -- Tuesday 20 Oct 2026, 12:00 in Vienna --
with ``HERMES_TIMEZONE=Europe/Vienna`` and a temporary store.  The store holds
the owner's jobs with the next starts today's code saved for them (the
monthly check and both Monday jobs an hour late after the autumn change),
jobs the preview must leave alone (paused, one-shot, interval, already
correct, one whose saved next start has already passed, and two finished
jobs) and one damaged entry.  A job whose saved next start is wrong is
realigned by pausing and resuming it; the preview only says which ones.
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

pytest.importorskip("croniter")

import hermes_time
from cron import jobs
from hermes_cli.cron import cron_command
from hermes_cli.subcommands.cron import build_cron_parser

VIENNA = ZoneInfo("Europe/Vienna")
NOW = datetime(2026, 10, 20, 10, 0, tzinfo=timezone.utc)  # Tuesday 12:00 in Vienna
MARKER = "SECRET-MARKER-7f3a"
NEEDS = "needs pause and resume"

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
# Finished jobs, saved after the damaged entry: (schedule, fields saved over the record).
FINISHED = {
    "Quarterly report": ("0 8 1 * *", {
        "state": "completed",
        "enabled": False,
        "next_run_at": None,
        "repeat": {"times": 2, "completed": 2},
    }),
    # Still enabled, with a saved next start that looks an hour late.
    "Launch countdown": ("0 10 1 * *", {
        "state": "completed",
        "enabled": True,
        "next_run_at": "2026-11-01T11:00:00+01:00",
        "repeat": {"times": None, "completed": 1},
    }),
}
# The wrong saved starts, each with the start pausing and resuming saves.
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
    "Quarterly report": "finished",
    "Launch countdown": "finished",
}
NAMES = [*SAVED, *FINISHED]

# A record that would be a healthy job but for one field; its name and
# prompt carry the marker, which must never reach the preview's output.
DAMAGED_ID = "d4m4g3d0e7f3"
HEALTHY = {
    "id": DAMAGED_ID,
    "name": f"Weekly {MARKER} digest",
    "prompt": f"Send the {MARKER} figures",
    "schedule": {"kind": "cron", "expr": "0 10 * * 1", "display": "0 10 * * 1"},
    "schedule_display": "0 10 * * 1",
    "repeat": {"times": None, "completed": 0},
    "enabled": True,
    "state": "scheduled",
    "paused_at": None,
    "next_run_at": "2026-10-26T11:00:00+01:00",
}
DAMAGED = {**HEALTHY, "next_run_at": f"Monday {MARKER}"}
DAMAGED_POSITION = len(SAVED) + 1  # saved between the jobs above and the finished ones


class _Store:
    """The temporary profile home, its fixed clock and the entries as first saved."""

    def __init__(self, home):
        self.home = home
        self.cron_dir = home / "cron"
        self.jobs_file = self.cron_dir / "jobs.json"
        self.now = NOW
        self.entries = []
        self.original = {}
        self.ids = {}

    def save(self, entries, *, escaped=False):
        """Write ``entries`` as the saved job list, the way an earlier version saved it.

        ``escaped`` writes every non-ASCII character as a JSON escape: the only
        way a lone surrogate can be saved (the file stays valid UTF-8).
        """
        self.jobs_file.write_text(
            json.dumps(
                {"jobs": entries, "updated_at": NOW.astimezone(VIENNA).isoformat()},
                indent=2,
                ensure_ascii=escaped,
            ),
            encoding="utf-8",
        )

    def records(self):
        """Saved job records by name, read straight from jobs.json (damaged entry left out)."""
        saved = json.loads(self.jobs_file.read_text(encoding="utf-8"))["jobs"]
        return {record["name"]: record for record in saved if record.get("id") != DAMAGED_ID}


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
        for name, (schedule, _fields) in FINISHED.items():
            jobs.create_job(
                prompt=f"Run {name}",
                schedule=schedule,
                name=name,
                model="test-model",
                provider="test-provider",
            )
        records = jobs.load_jobs()
    for record in records:
        if record["name"] in FINISHED:
            record.update(FINISHED[record["name"]][1])
        elif SAVED[record["name"]][2] is not None:
            record["next_run_at"] = SAVED[record["name"]][2]
    state.entries = records[: len(SAVED)] + [DAMAGED] + records[len(SAVED):]
    state.cron_dir.mkdir(parents=True)
    state.save(state.entries)
    state.original = {record["name"]: record for record in records}
    state.ids = {record["name"]: record["id"] for record in records}
    yield state
    hermes_time.reset_cache()


def _snapshot(root):
    """Every path under ``root`` with its content (files) and modification time."""
    return {
        str(path.relative_to(root)): (
            path.read_bytes() if path.is_file() else None,
            path.stat().st_mtime_ns,
        )
        for path in sorted(root.rglob("*"))
    }


def _file_state(path):
    """Bytes, modification time and inode of ``path``."""
    st = path.stat()
    return path.read_bytes(), st.st_mtime_ns, st.st_ino


def _damaged_row(position, cause):
    return {
        "position": position,
        "id": None,
        "name": None,
        "status": "damaged",
        "schedule": None,
        "before": None,
        "after": None,
        "reason": f"damaged entry: {cause}",
    }


def _parser():
    """The real ``hermes cron`` parser."""
    parser = argparse.ArgumentParser(prog="hermes")
    build_cron_parser(parser.add_subparsers(dest="command"), cmd_cron=cron_command)
    return parser


def _cli(capsys, *argv):
    """Run ``hermes cron <argv>`` through the real parser and handler."""
    code = cron_command(_parser().parse_args(["cron", *argv]))
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def _block(out, job_id):
    """The preview's lines for one job."""
    (block,) = [part for part in out.split("\n\n") if f"{job_id} " in part]
    return block


# -- the preview (D) ---------------------------------------------------------


def test_preview_lists_every_entry_and_writes_nothing(store):
    saved = _file_state(store.jobs_file)
    names = sorted(os.listdir(store.cron_dir))
    rows = jobs.plan_realign_times()
    assert _file_state(store.jobs_file) == saved
    assert sorted(os.listdir(store.cron_dir)) == names
    assert [row["position"] for row in rows] == list(range(1, len(store.entries) + 1))
    assert rows.pop(DAMAGED_POSITION - 1) == _damaged_row(DAMAGED_POSITION, "unreadable next start")
    assert MARKER not in repr(rows)
    assert [row["name"] for row in rows] == NAMES
    for row in rows:
        name = row["name"]
        assert row["id"] == store.ids[name]
        if name in FINISHED:
            assert row["status"] == "completed"
        else:
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


def test_cli_preview_lists_every_entry_and_writes_nothing(store, capsys):
    saved = _file_state(store.jobs_file)
    names = sorted(os.listdir(store.cron_dir))
    code, out, err = _cli(capsys, "realign-times")
    assert code == 0
    assert _file_state(store.jobs_file) == saved
    assert sorted(os.listdir(store.cron_dir)) == names
    assert "Europe/Vienna" in out
    for name, job_id in store.ids.items():
        assert f"{job_id} {name} [" in out
    for name, after in REALIGNED.items():
        block = _block(out, store.ids[name])
        assert NEEDS in block
        assert datetime.fromisoformat(SAVED[name][2]).strftime("Saved next start:  %Y-%m-%d %H:%M") in block
        assert datetime.fromisoformat(after).strftime("After realignment: %Y-%m-%d %H:%M") in block
    monthly = _block(out, store.ids["Monthly model check"])
    assert "Saved next start:  2026-11-01 11:00 (UTC+01:00)" in monthly
    assert "2026-11-01 10:00 (UTC+01:00)" in monthly
    for name, reason in UNCHANGED.items():
        assert f"unchanged ({reason})" in _block(out, store.ids[name])
    assert "will change" not in out
    assert f"Entry {DAMAGED_POSITION} in the saved list [damaged]" in out
    assert "unchanged (damaged entry: unreadable next start); its contents are not shown" in out
    assert MARKER not in out + err
    assert DAMAGED_ID not in out + err
    assert (
        f"{len(REALIGNED)} of {len(store.entries)} saved next start(s) differ from the "
        "fixed calculation. Nothing has been written."
    ) in out
    assert "hermes cron pause <job> and hermes cron resume <job>" in out


def test_cli_preview_says_when_no_time_zone_is_set(store, capsys, monkeypatch):
    monkeypatch.delenv("HERMES_TIMEZONE")
    hermes_time.reset_cache()
    monkeypatch.setattr(jobs, "_hermes_now", lambda: NOW.astimezone())
    before = _snapshot(store.home)
    code, out, _err = _cli(capsys, "realign-times")
    assert code == 0
    assert "server's local time" in out
    assert all(name in out for name in NAMES)
    assert _snapshot(store.home) == before


def test_a_job_that_reached_its_repeat_limit_is_finished(store):
    entries = list(store.entries)
    entries[0] = {**entries[0], "repeat": {"times": 3, "completed": 3}}
    store.save(entries)
    row = jobs.plan_realign_times()[0]
    assert (row["name"], row["status"], row["reason"]) == ("Monthly model check", "active", "finished")
    assert row["after"] == row["before"] == SAVED["Monthly model check"][2]


def _without(key):
    entry = dict(HEALTHY)
    del entry[key]
    return entry


PAUSED = {"enabled": False, "state": "paused", "paused_at": "2026-10-19T09:00:00+02:00"}
# A cron expression the scheduler cannot use damages the entry whatever the
# job's state; the marker is also in the entry's name and schedule display.
UNUSABLE_CRON = {
    **HEALTHY,
    "schedule": {"kind": "cron", "expr": f"{MARKER} 10 * * 1", "display": f"{MARKER} 10 * * 1"},
    "schedule_display": f"{MARKER} 10 * * 1",
}
# An escaped lone surrogate is not valid text: it cannot be printed at all.
SURROGATE = "\ud800"
# Without a top-level display, the preview prints the schedule's own fields.
NO_DISPLAY = _without("schedule_display")

DAMAGED_SHAPES = [
    pytest.param(MARKER, "not a job record", id="bare-string"),
    pytest.param(7, "not a job record", id="number"),
    pytest.param(None, "not a job record", id="null"),
    pytest.param([MARKER, {"id": MARKER}], "not a job record", id="list"),
    pytest.param(_without("id"), "unreadable job ID", id="no-id"),
    pytest.param({**HEALTHY, "id": ""}, "unreadable job ID", id="empty-id"),
    pytest.param({**HEALTHY, "id": 42}, "unreadable job ID", id="id-not-text"),
    pytest.param({**HEALTHY, "name": {"text": MARKER}}, "unreadable name", id="name"),
    pytest.param({**HEALTHY, "schedule_display": [MARKER]}, "unreadable schedule display", id="schedule-display"),
    pytest.param({**HEALTHY, "schedule": f"0 10 * * 1 {MARKER}"}, "unreadable schedule", id="schedule-not-object"),
    pytest.param({**HEALTHY, "schedule": {"kind": 5, "expr": "0 10 * * 1"}}, "unreadable schedule", id="schedule-kind"),
    pytest.param({**HEALTHY, "schedule": {"kind": "cron", "expr": [MARKER]}}, "unreadable schedule", id="schedule-expr"),
    pytest.param(
        {**HEALTHY, "schedule": {"kind": "cron", "expr": "0 10 * * 1", "display": 9}},
        "unreadable schedule",
        id="schedule-display-field",
    ),
    pytest.param(
        {**HEALTHY, "schedule": {"kind": "cron", "expr": "0 10 * * 1", "value": {"text": MARKER}}},
        "unreadable schedule",
        id="schedule-value",
    ),
    pytest.param({**HEALTHY, "schedule": {"kind": "once", "run_at": 1793520000}}, "unreadable schedule", id="schedule-run-at"),
    pytest.param({**HEALTHY, "enabled": "yes"}, "unreadable enabled flag", id="enabled"),
    pytest.param({**HEALTHY, "enabled": None}, "unreadable enabled flag", id="enabled-null"),
    pytest.param({**HEALTHY, "state": 3}, "unreadable state", id="state"),
    pytest.param({**HEALTHY, "paused_at": [MARKER]}, "unreadable pause time", id="paused-at"),
    pytest.param({**HEALTHY, "repeat": MARKER}, "unreadable repeat limit", id="repeat"),
    pytest.param({**HEALTHY, "repeat": {"times": "2", "completed": 0}}, "unreadable repeat limit", id="repeat-times"),
    pytest.param({**HEALTHY, "repeat": {"times": 2, "completed": 1.5}}, "unreadable repeat limit", id="repeat-completed"),
    pytest.param({**HEALTHY, "next_run_at": MARKER}, "unreadable next start", id="next-start"),
    pytest.param({**HEALTHY, "next_run_at": 1793520000}, "unreadable next start", id="next-start-number"),
    pytest.param(
        {**HEALTHY, "schedule": {"kind": "cron", "expr": f"{MARKER} 10 * * 1", "display": "0 10 * * 1"}},
        "unreadable cron expression",
        id="cron-expression",
    ),
    pytest.param({**UNUSABLE_CRON, **PAUSED}, "unreadable cron expression", id="cron-expression-paused"),
    pytest.param(
        {**UNUSABLE_CRON, "state": "completed", "enabled": False, "next_run_at": None},
        "unreadable cron expression",
        id="cron-expression-completed",
    ),
    pytest.param({**UNUSABLE_CRON, "next_run_at": None}, "unreadable cron expression", id="cron-expression-no-next-start"),
    pytest.param(
        {**UNUSABLE_CRON, "next_run_at": "2026-10-19T10:00:00+02:00"},
        "unreadable cron expression",
        id="cron-expression-past-next-start",
    ),
    pytest.param(
        {**HEALTHY, **PAUSED, "schedule": {"kind": "cron", "expr": "0 0 31 2 *"}, "schedule_display": "0 0 31 2 *"},
        "unreadable cron expression",
        id="cron-expression-without-a-start",
    ),
    pytest.param({**HEALTHY, "id": f"{DAMAGED_ID}{SURROGATE}"}, "unreadable job ID", id="id-text"),
    pytest.param({**HEALTHY, "name": f"Weekly {MARKER}{SURROGATE} digest"}, "unreadable name", id="name-text"),
    pytest.param({**HEALTHY, "state": f"scheduled{SURROGATE}"}, "unreadable state", id="state-text"),
    pytest.param(
        {**HEALTHY, "schedule_display": f"0 10 * * 1{SURROGATE}"},
        "unreadable schedule display",
        id="schedule-display-text",
    ),
    pytest.param(
        {**NO_DISPLAY, "schedule": {"kind": "cron", "expr": "0 10 * * 1", "display": f"0 10 * * 1{SURROGATE}"}},
        "unreadable schedule",
        id="schedule-display-field-text",
    ),
    pytest.param(
        {**NO_DISPLAY, "schedule": {"kind": "cron", "expr": "0 10 * * 1", "value": f"0 10 * * 1{SURROGATE}"}},
        "unreadable schedule",
        id="schedule-value-text",
    ),
    # Paused, so the expression is shown rather than used to work out a next start.
    pytest.param(
        {**NO_DISPLAY, **PAUSED, "schedule": {"kind": "cron", "expr": f"0 10 * * 1{SURROGATE}"}},
        "unreadable schedule",
        id="schedule-expr-text",
    ),
    pytest.param(
        {**NO_DISPLAY, "schedule": {"kind": "once", "run_at": f"2026-10-30T09:00:00+01:00{SURROGATE}"}},
        "unreadable schedule",
        id="schedule-run-at-text",
    ),
    # Each reads as a time, since any character may separate the date from the time.
    pytest.param(
        {**HEALTHY, "next_run_at": f"0001-01-01{SURROGATE}00:00:00"},
        "unreadable next start",
        id="next-start-text-year-limit",
    ),
    pytest.param(
        {**HEALTHY, "next_run_at": f"2026-10-26{SURROGATE}10:00:00+01:00"},
        "unreadable next start",
        id="next-start-text",
    ),
]


@pytest.mark.parametrize("entry, cause", DAMAGED_SHAPES)
def test_preview_lists_a_damaged_entry_without_its_contents(store, capsys, entry, cause):
    store.save([entry, *store.entries], escaped=True)
    saved = _file_state(store.jobs_file)
    names = sorted(os.listdir(store.cron_dir))
    rows = jobs.plan_realign_times()
    assert rows[0] == _damaged_row(1, cause)
    assert rows[DAMAGED_POSITION] == _damaged_row(DAMAGED_POSITION + 1, "unreadable next start")
    healthy = [row for row in rows if row["status"] != "damaged"]
    assert [row["name"] for row in healthy] == NAMES
    assert {row["name"]: row["after"] for row in healthy if row["reason"] is None} == REALIGNED
    assert MARKER not in repr(rows)
    code, out, err = _cli(capsys, "realign-times")
    assert code == 0
    assert "Entry 1 in the saved list [damaged]" in out
    assert f"unchanged (damaged entry: {cause}); its contents are not shown" in out
    for name, job_id in store.ids.items():
        assert f"{job_id} {name} [" in out
    assert "Traceback" not in out + err
    assert MARKER not in out + err
    assert f"{len(REALIGNED)} of {len(store.entries) + 1} saved next start(s) differ" in out
    assert _file_state(store.jobs_file) == saved
    assert sorted(os.listdir(store.cron_dir)) == names


@pytest.mark.parametrize(
    "fields, reason, after",
    [
        pytest.param({}, None, "2026-10-26T10:00:00+01:00", id="active"),
        pytest.param(PAUSED, "paused", HEALTHY["next_run_at"], id="paused"),
    ],
)
def test_an_expression_the_scheduler_accepts_is_not_damaged(store, fields, reason, after):
    schedule = {"kind": "cron", "expr": "0 10 * * MON", "display": "0 10 * * MON"}
    store.save([{**HEALTHY, "schedule": schedule, "schedule_display": "0 10 * * MON", **fields}, *store.entries])
    row = jobs.plan_realign_times()[0]
    assert (row["id"], row["schedule"], row["reason"]) == (HEALTHY["id"], "0 10 * * MON", reason)
    assert (row["before"], row["after"]) == (HEALTHY["next_run_at"], after)


REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    "damaged, cause",
    [
        pytest.param({**HEALTHY, "name": f"Weekly {MARKER}{SURROGATE} digest"}, "unreadable name", id="name-text"),
        # At the year limit the preview cannot convert the start, so it would print it as saved.
        pytest.param(
            {**HEALTHY, "next_run_at": f"0001-01-01{SURROGATE}00:00:00"},
            "unreadable next start",
            id="next-start-text-year-limit",
        ),
    ],
)
def test_launcher_lists_the_jobs_after_an_entry_with_invalid_text(store, tmp_path, damaged, cause):
    """The real ``hermes`` launcher, writing strict UTF-8 to pipes.

    It runs on today's clock, so the job after the damaged entry is a paused one.
    """
    name = "Paused automation one"
    job_id = store.ids[name]
    store.save([damaged, store.original[name]], escaped=True)
    saved = _file_state(store.jobs_file)
    names = sorted(os.listdir(store.cron_dir))
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "hermes"), "cron", "realign-times"],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env={
            **os.environ,
            "PYTHONIOENCODING": "utf-8",
            "HERMES_HOME": str(store.home),
            "HERMES_TIMEZONE": "Europe/Vienna",
        },
        cwd=tmp_path,
        timeout=120,
    )
    out, err = result.stdout.decode("utf-8"), result.stderr.decode("utf-8")
    assert result.returncode == 0, err
    assert "Entry 1 in the saved list [damaged]" in out
    assert f"unchanged (damaged entry: {cause}); its contents are not shown" in out
    block = _block(out, job_id)
    assert f"{job_id} {name} [paused]" in block
    assert "Saved next start:  2026-10-21 03:30 (UTC+02:00)" in block
    assert "unchanged (paused)" in block
    assert "0 of 2 saved next start(s) differ" in out
    assert "Traceback" not in out + err
    assert MARKER not in out + err
    assert DAMAGED_ID not in out + err
    assert _file_state(store.jobs_file) == saved
    assert sorted(os.listdir(store.cron_dir)) == names


@pytest.mark.parametrize(
    "content",
    [f'{{"jobs": [{{"id": "{MARKER}"', f'"{MARKER}"', "42", "null", f'{{"jobs": "{MARKER}"}}'],
    ids=["truncated", "text", "number", "null", "jobs-not-a-list"],
)
def test_preview_stops_plainly_on_an_unreadable_saved_list(store, capsys, content):
    store.jobs_file.write_text(content, encoding="utf-8")
    before = _snapshot(store.home)
    with pytest.raises(RuntimeError) as exc:
        jobs.plan_realign_times()
    assert MARKER not in str(exc.value)
    code, out, err = _cli(capsys, "realign-times")
    assert code == 1
    assert "Cannot preview next starts: the saved job list could not be read." in out
    assert MARKER not in out + err
    assert _snapshot(store.home) == before


# -- apply and restore are gone (B, C) ---------------------------------------


def test_parser_reads_the_preview_without_options():
    args = _parser().parse_args(["cron", "realign-times"])
    assert args.cron_command == "realign-times"
    assert not {"apply", "restore", "job_ids"} & set(vars(args))


@pytest.mark.parametrize(
    "argv",
    [("--apply",), ("--restore",), ("--job", "<id>"), ("--apply", "--job", "<id>"), ("apply",), ("restore",)],
    ids=["apply", "restore", "job", "apply-job", "positional-apply", "positional-restore"],
)
def test_apply_and_restore_are_refused_as_unknown_and_write_nothing(store, capsys, argv):
    argv = [store.ids["Monthly model check"] if arg == "<id>" else arg for arg in argv]
    before = _snapshot(store.home)
    with pytest.raises(SystemExit) as exc:
        _parser().parse_args(["cron", "realign-times", *argv])
    assert exc.value.code == 2
    assert "unrecognized arguments" in capsys.readouterr().err
    assert _snapshot(store.home) == before


_OPENED = []
_RECORDING = []
_HOOKED = []


def _record_open(event, args):
    if event == "open" and _RECORDING:
        path = args[0]
        _OPENED.append(os.fsdecode(path) if isinstance(path, (str, bytes, os.PathLike)) else str(path))


def _opened_while(run):
    """Run ``run()``; return its result and every path opened meanwhile."""
    if not _HOOKED:
        # Audit hooks cannot be removed, so this one records only while
        # _RECORDING is set.
        sys.addaudithook(_record_open)
        _HOOKED.append(True)
    _OPENED.clear()
    _RECORDING.append(True)
    try:
        result = run()
    finally:
        _RECORDING.clear()
    return result, list(_OPENED)


@pytest.mark.parametrize("via", ["plan", "cli"])
def test_c_preview_never_names_or_opens_a_restore_record(store, capsys, via):
    for name in ("realign_restore_file", "_read_realign_record", "apply_realign_times", "restore_realign_times"):
        assert not hasattr(jobs, name)

    def preview():
        if via == "plan":
            return json.dumps(jobs.plan_realign_times())
        code, out, err = _cli(capsys, "realign-times")
        assert code == 0
        return out + err

    names = sorted(os.listdir(store.cron_dir))
    plain, opened = _opened_while(preview)
    assert any(path.endswith("jobs.json") for path in opened)  # the hook records the preview's reads
    assert sorted(os.listdir(store.cron_dir)) == names
    assert "restore" not in plain.lower()
    assert "--apply" not in plain
    decoy = store.cron_dir / "realign_times_restore.json"
    decoy.write_text(
        json.dumps([{
            "job_id": store.ids["Monthly model check"],
            "name": "Monthly model check",
            "before": "2026-11-01T11:00:00+01:00",
            "after": "2026-11-01T10:00:00+01:00",
            "applied_at": "2026-10-20T12:00:00+02:00",
        }]),
        encoding="utf-8",
    )
    decoy_state = _file_state(decoy)
    names = sorted(os.listdir(store.cron_dir))
    with_decoy, opened_with_decoy = _opened_while(preview)
    assert with_decoy == plain
    assert [path for path in opened + opened_with_decoy if "restore" in path.lower()] == []
    assert _file_state(decoy) == decoy_state
    assert sorted(os.listdir(store.cron_dir)) == names


# -- realigning by pause and resume (E) --------------------------------------


def test_pause_and_resume_saves_the_corrected_next_start(store):
    preview = {row["name"]: row for row in jobs.plan_realign_times() if row["status"] != "damaged"}
    for name in REALIGNED:
        assert preview[name]["reason"] is None
        assert jobs.pause_job(store.ids[name])["enabled"] is False
        assert jobs.resume_job(store.ids[name])["next_run_at"] == preview[name]["after"]
    records = store.records()
    assert list(records) == NAMES
    for name, record in records.items():
        expected = dict(store.original[name])
        if name in REALIGNED:
            expected["next_run_at"] = preview[name]["after"]
            assert preview[name]["after"] == REALIGNED[name]
        assert record == expected
        if name in REALIGNED:
            assert (record["enabled"], record["state"]) == (True, "scheduled")
    saved = json.loads(store.jobs_file.read_text(encoding="utf-8"))["jobs"]
    assert saved[DAMAGED_POSITION - 1] == DAMAGED
    rows = jobs.plan_realign_times()
    assert rows[DAMAGED_POSITION - 1] == _damaged_row(DAMAGED_POSITION, "unreadable next start")
    rows = {row["name"]: row for row in rows if row["status"] != "damaged"}
    for name in REALIGNED:
        assert (rows[name]["status"], rows[name]["reason"]) == ("active", "already correct")
        assert rows[name]["before"] == rows[name]["after"] == REALIGNED[name]
    assert {name: row["reason"] for name, row in rows.items() if name not in REALIGNED} == UNCHANGED


def test_cli_pause_and_resume_realigns_a_job(store, capsys):
    name = "Monthly model check"
    job_id = store.ids[name]
    code, out, _err = _cli(capsys, "realign-times")
    assert code == 0
    assert NEEDS in _block(out, job_id)
    for action in ("pause", "resume"):
        code, _out, _err = _cli(capsys, action, job_id)
        assert code == 0
    records = store.records()
    expected = {**store.original[name], "next_run_at": REALIGNED[name]}
    assert records[name] == expected
    for other, record in records.items():
        if other != name:
            assert record == store.original[other]
    code, out, _err = _cli(capsys, "realign-times")
    assert code == 0
    block = _block(out, job_id)
    assert f"{job_id} {name} [active]" in block
    assert "2026-11-01 10:00 (UTC+01:00) — unchanged (already correct)" in block
    assert f"{len(REALIGNED) - 1} of {len(store.entries)} saved next start(s) differ" in out
