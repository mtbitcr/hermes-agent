"""Cron jobs keep their chosen Vienna time across both clock changes.

Each simulation drives real jobs through the scheduler's own cycle, in the
order its tick uses -- ``get_due_jobs``, ``advance_next_runs``,
``claim_job_for_fire`` and, once the run is over, ``mark_job_run`` -- on a
fixed clock with a temporary store and ``HERMES_TIMEZONE=Europe/Vienna``.
The clock ticks every minute through both change nights (00:00-05:00 Vienna
on Sunday 25 Oct 2026 and Sunday 28 Mar 2027) and jumps straight to the next
due minute everywhere else.  Expected starts are UTC minutes.
"""

import json
import math
from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pytest

pytest.importorskip("croniter")

import hermes_time
from cron import jobs

VIENNA = ZoneInfo("Europe/Vienna")
UTC = timezone.utc
RUN_TIME = timedelta(seconds=150)  # how long each simulated run takes


def vienna(text):
    """UTC instant of a Vienna wall time (never inside a skipped or repeated hour)."""
    return datetime.fromisoformat(text).replace(tzinfo=VIENNA).astimezone(UTC)


def utc(text):
    return datetime.fromisoformat(text).replace(tzinfo=UTC)


def as_utc(stamp):
    return datetime.fromisoformat(stamp).astimezone(UTC)


CHANGE_NIGHTS = [
    (vienna("2026-10-25T00:00"), vienna("2026-10-25T05:00")),
    (vienna("2027-03-28T00:00"), vienna("2027-03-28T05:00")),
]
# Second pass of the repeated hour: Vienna 02:00-02:59 at +01:00.
SECOND_PASS = (utc("2026-10-25T01:00"), utc("2026-10-25T02:00"))


class _Clock:
    """The fixed clock every cron function reads, plus the temporary home."""

    def __init__(self, home):
        self.home = home
        self.now = None


@pytest.fixture
def clock(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    # With a config file present, load_config() answers every tick from its cache.
    (home / "config.yaml").write_text("timezone: Europe/Vienna\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_TIMEZONE", "Europe/Vienna")
    hermes_time.reset_cache()
    fake = _Clock(home)
    monkeypatch.setattr(jobs, "_hermes_now", lambda: fake.now.astimezone(VIENNA))
    monkeypatch.setattr(jobs, "_cron_cadence_cache", {})
    yield fake
    hermes_time.reset_cache()


def _ceil_minute(moment):
    return datetime.fromtimestamp(math.ceil(moment.timestamp() / 60) * 60, UTC)


def _next_tick(tick, end, saved):
    """Next tick: every minute through a change night, else the next due minute."""
    following = tick + timedelta(minutes=1)
    if any(lo <= following < hi for lo, hi in CHANGE_NIGHTS):
        return following
    due = [
        as_utc(job["next_run_at"])
        for job in saved
        if job.get("enabled", True) and job.get("next_run_at")
    ]
    target = max(following, _ceil_minute(min(due))) if due else end
    for lo, _hi in CHANGE_NIGHTS:
        if following <= lo < target:
            target = lo
    return min(target, end)


def simulate(clock, specs, start, end):
    """Create ``specs`` (name, schedule, paused) at ``start`` and tick until ``end``.

    Returns the UTC start minutes per job name, every ``next_run_at`` the
    store held along the way, and each job's final saved record by name.
    """
    jobs_file = clock.home / "cron" / "jobs.json"
    seen = {"raw": None, "jobs": []}
    stamps = set()

    def saved():
        raw = jobs_file.read_bytes()
        if raw != seen["raw"]:
            seen["raw"], seen["jobs"] = raw, json.loads(raw)["jobs"]
            stamps.update(job["next_run_at"] for job in seen["jobs"] if job.get("next_run_at"))
        return seen["jobs"]

    clock.now = start
    with jobs.use_cron_store(clock.home):
        names = {}
        for name, schedule, paused in specs:
            job = jobs.create_job(
                prompt=f"Run {name}",
                schedule=schedule,
                name=name,
                model="test-model",
                provider="test-provider",
            )
            names[job["id"]] = name
            if paused:
                jobs.pause_job(job["id"])
        saved()
        starts = {name: [] for name, _schedule, _paused in specs}
        running = []  # (finishes at, job id, fire-claim owner)
        tick = _ceil_minute(start)
        while tick < end:
            clock.now = tick
            due = jobs.get_due_jobs()
            saved()
            if due:
                jobs.advance_next_runs([job["id"] for job in due])
                saved()
            for job in due:
                claimed = jobs.claim_job_for_fire(job["id"], return_job=True)
                saved()
                if not claimed:
                    continue
                starts[names[job["id"]]].append(f"{tick:%Y-%m-%d %H:%M}")
                running.append((tick + RUN_TIME, job["id"], claimed["fire_claim"]["by"]))
            following = _next_tick(tick, end, saved())
            running.sort()
            while running and running[0][0] < following:
                done_at, job_id, owner = running.pop(0)
                clock.now = done_at
                assert jobs.mark_job_run(job_id, True, expected_fire_owner=owner)
                following = max(_next_tick(tick, end, saved()), _ceil_minute(done_at))
            tick = following
        final = {names[job["id"]]: job for job in saved()}
    return starts, stamps, final


def _assert_no_second_pass(stamps):
    """No saved next start may point into the repeated hour's second pass."""
    inside = sorted(stamp for stamp in stamps if SECOND_PASS[0] <= as_utc(stamp) < SECOND_PASS[1])
    assert not inside, inside


def _daily(first, last, wall):
    """UTC start minutes of a job at Vienna ``wall`` on every day first..last."""
    days = (last - first).days + 1
    return [f"{vienna(f'{first + timedelta(days=n)}T{wall}'):%Y-%m-%d %H:%M}" for n in range(days)]


def _on_days(runs, *days):
    return [run for run in runs if run[:10] in days]


# The owner's jobs.  The two paused automations stand in for the real ones:
# a daily job inside the risky hour and an interval job.
OWNER_JOBS = [
    ("Monthly model check", "0 10 1 * *", False),
    ("Weekly upstream update summary", "0 10 * * 1", False),
    ("Project health review", "0 9 * * 1", False),
    ("Daily backup", "30 3 * * *", False),
    ("Paused automation one", "30 2 * * *", True),
    ("Paused automation two", "every 2h", True),
]


def _assert_paused_stayed_paused(starts, final):
    for name in ("Paused automation one", "Paused automation two"):
        assert starts[name] == []
        assert final[name]["enabled"] is False
        assert final[name]["state"] == "paused"


def test_owner_jobs_keep_vienna_time_across_the_autumn_change(clock):
    starts, stamps, final = simulate(
        clock, OWNER_JOBS, vienna("2026-09-30T00:00"), vienna("2026-11-02T00:00")
    )
    assert starts["Monthly model check"] == ["2026-10-01 08:00", "2026-11-01 09:00"]
    assert starts["Weekly upstream update summary"] == [
        "2026-10-05 08:00", "2026-10-12 08:00", "2026-10-19 08:00", "2026-10-26 09:00",
    ]
    assert starts["Project health review"] == [
        "2026-10-05 07:00", "2026-10-12 07:00", "2026-10-19 07:00", "2026-10-26 08:00",
    ]
    backup = starts["Daily backup"]
    assert backup == _daily(date(2026, 9, 30), date(2026, 11, 1), "03:30")
    assert _on_days(backup, "2026-10-24", "2026-10-25", "2026-10-26") == [
        "2026-10-24 01:30", "2026-10-25 02:30", "2026-10-26 02:30",
    ]
    _assert_paused_stayed_paused(starts, final)
    _assert_no_second_pass(stamps)


def test_owner_jobs_keep_vienna_time_across_the_spring_change(clock):
    starts, stamps, final = simulate(
        clock, OWNER_JOBS, vienna("2027-02-28T00:00"), vienna("2027-04-02T00:00")
    )
    assert starts["Monthly model check"] == ["2027-03-01 09:00", "2027-04-01 08:00"]
    assert starts["Weekly upstream update summary"] == [
        "2027-03-01 09:00", "2027-03-08 09:00", "2027-03-15 09:00", "2027-03-22 09:00",
        "2027-03-29 08:00",
    ]
    assert starts["Project health review"] == [
        "2027-03-01 08:00", "2027-03-08 08:00", "2027-03-15 08:00", "2027-03-22 08:00",
        "2027-03-29 07:00",
    ]
    backup = starts["Daily backup"]
    assert backup == _daily(date(2027, 2, 28), date(2027, 4, 1), "03:30")
    assert _on_days(backup, "2027-03-27", "2027-03-28", "2027-03-29") == [
        "2027-03-27 02:30", "2027-03-28 01:30", "2027-03-29 01:30",
    ]
    _assert_paused_stayed_paused(starts, final)
    _assert_no_second_pass(stamps)


WEEKDAYS = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
WEEKLY_TIMES = ["00:00", "01:59", "03:00", "09:00", "23:30"]
WEEKLY_JOBS = [
    (f"weekly {WEEKDAYS[day]} {wall}", f"{int(wall[3:])} {int(wall[:2])} * * {day}", False)
    for day in range(7)
    for wall in WEEKLY_TIMES
]


@pytest.mark.parametrize(
    "first_day", [date(2026, 10, 21), date(2027, 3, 24)], ids=["autumn", "spring"]
)
def test_weekly_jobs_run_once_at_their_time_in_the_change_week(clock, first_day):
    """Every weekday and time, Wednesday noon to Wednesday noon around the change."""
    start = vienna(f"{first_day}T12:00")
    end = vienna(f"{first_day + timedelta(days=7)}T12:00")
    starts, stamps, _final = simulate(clock, WEEKLY_JOBS, start, end)
    expected = {}
    for n in range(8):
        day = first_day + timedelta(days=n)
        for wall in WEEKLY_TIMES:
            moment = vienna(f"{day}T{wall}")
            if start < moment < end:
                name = f"weekly {WEEKDAYS[day.isoweekday() % 7]} {wall}"
                expected.setdefault(name, []).append(f"{moment:%Y-%m-%d %H:%M}")
    assert starts == expected
    assert all(len(runs) == 1 for runs in starts.values())
    _assert_no_second_pass(stamps)


DAILY_JOBS = [
    ("daily 01:59", "59 1 * * *", False),
    ("daily 02:00", "0 2 * * *", False),
    ("daily 02:30", "30 2 * * *", False),
    ("daily 02:59", "59 2 * * *", False),
    ("daily 03:00", "0 3 * * *", False),
    ("Sundays 02:15", "15 2 * * 0", False),
    ("weekends 02:30", "30 2 * * 0,6", False),
    ("weekdays 09:00-17:00 every 2 hours", "0 9-17/2 * * 1-5", False),
]
# Friday to Monday around each change.  In autumn the repeated hour runs once,
# at its first pass (+02:00); in spring a start in the skipped hour runs once
# at 03:00 (+02:00), the moment the clocks go forward.
DAILY_EXPECTED = {
    "autumn": (
        vienna("2026-10-23T00:00"),
        vienna("2026-10-27T00:00"),
        {
            "daily 01:59": [
                "2026-10-22 23:59", "2026-10-23 23:59", "2026-10-24 23:59", "2026-10-26 00:59",
            ],
            "daily 02:00": [
                "2026-10-23 00:00", "2026-10-24 00:00", "2026-10-25 00:00", "2026-10-26 01:00",
            ],
            "daily 02:30": [
                "2026-10-23 00:30", "2026-10-24 00:30", "2026-10-25 00:30", "2026-10-26 01:30",
            ],
            "daily 02:59": [
                "2026-10-23 00:59", "2026-10-24 00:59", "2026-10-25 00:59", "2026-10-26 01:59",
            ],
            "daily 03:00": [
                "2026-10-23 01:00", "2026-10-24 01:00", "2026-10-25 02:00", "2026-10-26 02:00",
            ],
            "Sundays 02:15": ["2026-10-25 00:15"],
            "weekends 02:30": ["2026-10-24 00:30", "2026-10-25 00:30"],
            "weekdays 09:00-17:00 every 2 hours": [
                "2026-10-23 07:00", "2026-10-23 09:00", "2026-10-23 11:00", "2026-10-23 13:00",
                "2026-10-23 15:00", "2026-10-26 08:00", "2026-10-26 10:00", "2026-10-26 12:00",
                "2026-10-26 14:00", "2026-10-26 16:00",
            ],
        },
    ),
    "spring": (
        vienna("2027-03-26T00:00"),
        vienna("2027-03-30T00:00"),
        {
            "daily 01:59": [
                "2027-03-26 00:59", "2027-03-27 00:59", "2027-03-28 00:59", "2027-03-28 23:59",
            ],
            "daily 02:00": [
                "2027-03-26 01:00", "2027-03-27 01:00", "2027-03-28 01:00", "2027-03-29 00:00",
            ],
            "daily 02:30": [
                "2027-03-26 01:30", "2027-03-27 01:30", "2027-03-28 01:00", "2027-03-29 00:30",
            ],
            "daily 02:59": [
                "2027-03-26 01:59", "2027-03-27 01:59", "2027-03-28 01:00", "2027-03-29 00:59",
            ],
            "daily 03:00": [
                "2027-03-26 02:00", "2027-03-27 02:00", "2027-03-28 01:00", "2027-03-29 01:00",
            ],
            "Sundays 02:15": ["2027-03-28 01:00"],
            "weekends 02:30": ["2027-03-27 01:30", "2027-03-28 01:00"],
            "weekdays 09:00-17:00 every 2 hours": [
                "2027-03-26 08:00", "2027-03-26 10:00", "2027-03-26 12:00", "2027-03-26 14:00",
                "2027-03-26 16:00", "2027-03-29 07:00", "2027-03-29 09:00", "2027-03-29 11:00",
                "2027-03-29 13:00", "2027-03-29 15:00",
            ],
        },
    ),
}


@pytest.mark.parametrize("night", ["autumn", "spring"])
def test_jobs_in_the_odd_hours_run_once_on_the_change_day(clock, night):
    start, end, expected = DAILY_EXPECTED[night]
    starts, stamps, _final = simulate(clock, DAILY_JOBS, start, end)
    assert starts == expected
    _assert_no_second_pass(stamps)


NIGHT_JOBS = [
    ("every 15m", "every 15m", False),
    ("every 30m", "every 30m", False),
    ("every 60m", "every 60m", False),
    ("every 2h", "every 2h", False),
    ("cron every 15 minutes", "*/15 * * * *", False),
    ("cron hourly :00", "0 * * * *", False),
    ("cron hourly :30", "30 * * * *", False),
    ("cron every 2 hours", "0 */2 * * *", False),
    ("cron every 6 hours", "0 */6 * * *", False),
]
# Observed from today's code (the untouched baseline) with this same
# simulation; interval and every-hour schedules must keep exactly these starts.
TODAYS_NIGHT_STARTS = {
    "autumn": {
        "every 15m": [
            "2026-10-24 21:15", "2026-10-24 21:33", "2026-10-24 21:51", "2026-10-24 22:09",
            "2026-10-24 22:27", "2026-10-24 22:45", "2026-10-24 23:03", "2026-10-24 23:21",
            "2026-10-24 23:39", "2026-10-24 23:57", "2026-10-25 00:15", "2026-10-25 00:33",
            "2026-10-25 00:51", "2026-10-25 02:09", "2026-10-25 02:27", "2026-10-25 02:45",
            "2026-10-25 03:03", "2026-10-25 03:21", "2026-10-25 03:39", "2026-10-25 03:57",
            "2026-10-25 04:15", "2026-10-25 04:33", "2026-10-25 04:51", "2026-10-25 05:09",
            "2026-10-25 05:27", "2026-10-25 05:45",
        ],
        "every 30m": [
            "2026-10-24 21:30", "2026-10-24 22:03", "2026-10-24 22:36", "2026-10-24 23:09",
            "2026-10-24 23:42", "2026-10-25 00:15", "2026-10-25 00:48", "2026-10-25 02:21",
            "2026-10-25 02:54", "2026-10-25 03:27", "2026-10-25 04:00", "2026-10-25 04:33",
            "2026-10-25 05:06", "2026-10-25 05:39",
        ],
        "every 60m": [
            "2026-10-24 22:00", "2026-10-24 23:03", "2026-10-25 00:06", "2026-10-25 02:09",
            "2026-10-25 03:12", "2026-10-25 04:15", "2026-10-25 05:18",
        ],
        "every 2h": [
            "2026-10-24 23:00", "2026-10-25 02:03", "2026-10-25 04:06",
        ],
        "cron every 15 minutes": [
            "2026-10-24 21:15", "2026-10-24 21:30", "2026-10-24 21:45", "2026-10-24 22:00",
            "2026-10-24 22:15", "2026-10-24 22:30", "2026-10-24 22:45", "2026-10-24 23:00",
            "2026-10-24 23:15", "2026-10-24 23:30", "2026-10-24 23:45", "2026-10-25 00:00",
            "2026-10-25 00:15", "2026-10-25 00:30", "2026-10-25 00:45", "2026-10-25 02:00",
            "2026-10-25 02:15", "2026-10-25 02:30", "2026-10-25 02:45", "2026-10-25 03:00",
            "2026-10-25 03:15", "2026-10-25 03:30", "2026-10-25 03:45", "2026-10-25 04:00",
            "2026-10-25 04:15", "2026-10-25 04:30", "2026-10-25 04:45", "2026-10-25 05:00",
            "2026-10-25 05:15", "2026-10-25 05:30", "2026-10-25 05:45",
        ],
        "cron hourly :00": [
            "2026-10-24 22:00", "2026-10-24 23:00", "2026-10-25 00:00", "2026-10-25 02:00",
            "2026-10-25 03:00", "2026-10-25 04:00", "2026-10-25 05:00",
        ],
        "cron hourly :30": [
            "2026-10-24 21:30", "2026-10-24 22:30", "2026-10-24 23:30", "2026-10-25 00:30",
            "2026-10-25 02:30", "2026-10-25 03:30", "2026-10-25 04:30", "2026-10-25 05:30",
        ],
        "cron every 2 hours": [
            "2026-10-24 22:00", "2026-10-25 00:00", "2026-10-25 03:00", "2026-10-25 05:00",
        ],
        "cron every 6 hours": [
            "2026-10-24 22:00", "2026-10-25 05:00",
        ],
    },
    "spring": {
        "every 15m": [
            "2027-03-27 22:15", "2027-03-27 22:33", "2027-03-27 22:51", "2027-03-27 23:09",
            "2027-03-27 23:27", "2027-03-27 23:45", "2027-03-28 00:03", "2027-03-28 00:21",
            "2027-03-28 00:39", "2027-03-28 00:57", "2027-03-28 01:15", "2027-03-28 01:33",
            "2027-03-28 01:51", "2027-03-28 02:09", "2027-03-28 02:27", "2027-03-28 02:45",
            "2027-03-28 03:03", "2027-03-28 03:21", "2027-03-28 03:39", "2027-03-28 03:57",
            "2027-03-28 04:15", "2027-03-28 04:33", "2027-03-28 04:51",
        ],
        "every 30m": [
            "2027-03-27 22:30", "2027-03-27 23:03", "2027-03-27 23:36", "2027-03-28 00:09",
            "2027-03-28 00:42", "2027-03-28 01:15", "2027-03-28 01:48", "2027-03-28 02:21",
            "2027-03-28 02:54", "2027-03-28 03:27", "2027-03-28 04:00", "2027-03-28 04:33",
        ],
        "every 60m": [
            "2027-03-27 23:00", "2027-03-28 00:03", "2027-03-28 01:06", "2027-03-28 02:09",
            "2027-03-28 03:12", "2027-03-28 04:15",
        ],
        "every 2h": [
            "2027-03-28 00:00", "2027-03-28 01:03", "2027-03-28 03:06",
        ],
        "cron every 15 minutes": [
            "2027-03-27 22:15", "2027-03-27 22:30", "2027-03-27 22:45", "2027-03-27 23:00",
            "2027-03-27 23:15", "2027-03-27 23:30", "2027-03-27 23:45", "2027-03-28 00:00",
            "2027-03-28 00:15", "2027-03-28 00:30", "2027-03-28 00:45", "2027-03-28 01:00",
            "2027-03-28 01:15", "2027-03-28 01:30", "2027-03-28 01:45", "2027-03-28 02:00",
            "2027-03-28 02:15", "2027-03-28 02:30", "2027-03-28 02:45", "2027-03-28 03:00",
            "2027-03-28 03:15", "2027-03-28 03:30", "2027-03-28 03:45", "2027-03-28 04:00",
            "2027-03-28 04:15", "2027-03-28 04:30", "2027-03-28 04:45",
        ],
        "cron hourly :00": [
            "2027-03-27 23:00", "2027-03-28 00:00", "2027-03-28 01:00", "2027-03-28 02:00",
            "2027-03-28 03:00", "2027-03-28 04:00",
        ],
        "cron hourly :30": [
            "2027-03-27 22:30", "2027-03-27 23:30", "2027-03-28 00:30", "2027-03-28 01:30",
            "2027-03-28 02:30", "2027-03-28 03:30", "2027-03-28 04:30",
        ],
        "cron every 2 hours": [
            "2027-03-27 23:00", "2027-03-28 01:00", "2027-03-28 02:00", "2027-03-28 04:00",
        ],
        "cron every 6 hours": [
            "2027-03-27 23:00", "2027-03-28 04:00",
        ],
    },
}


@pytest.mark.parametrize(
    "night, start, end",
    [
        ("autumn", vienna("2026-10-24T23:00"), vienna("2026-10-25T07:00")),
        ("spring", vienna("2027-03-27T23:00"), vienna("2027-03-28T07:00")),
    ],
    ids=["autumn", "spring"],
)
def test_frequent_schedules_keep_todays_starts(clock, night, start, end):
    starts, stamps, _final = simulate(clock, NIGHT_JOBS, start, end)
    assert starts == TODAYS_NIGHT_STARTS[night]
    _assert_no_second_pass(stamps)


GRID_EXPRESSIONS = [
    "0 2 * * *", "30 2 * * *", "59 2 * * *", "0 3 * * *", "30 3 * * *", "15 2 * * 0",
    "*/15 * * * *", "0 * * * *", "30 * * * *", "0 */2 * * *", "0 10 1 * *",
]


@pytest.mark.parametrize("night", CHANGE_NIGHTS, ids=["autumn", "spring"])
def test_next_start_is_always_later_and_never_in_the_second_pass(clock, night):
    """From every minute of the change night, with or without a last run."""
    lo, hi = night
    problems = []
    moment = lo
    while moment <= hi:
        clock.now = moment
        for expr in GRID_EXPRESSIONS:
            for last_run_at in (None, moment.astimezone(VIENNA).isoformat()):
                start = as_utc(jobs.compute_next_run({"kind": "cron", "expr": expr}, last_run_at))
                if start <= moment or SECOND_PASS[0] <= start < SECOND_PASS[1]:
                    problems.append((expr, f"{moment:%H:%M} UTC", last_run_at, f"{start:%H:%M} UTC"))
        moment += timedelta(minutes=1)
    assert not problems, f"{len(problems)} bad next starts, e.g. {problems[:5]}"


def test_monthly_period_matches_the_gap_between_two_next_starts(clock):
    """The fixture starts every test with an empty ``_cron_cadence_cache``."""
    clock.now = vienna("2026-10-20T12:00")
    schedule = {"kind": "cron", "expr": "0 10 1 * *"}
    first = jobs.compute_next_run(schedule)
    second = jobs.compute_next_run(schedule, first)
    assert as_utc(first) == utc("2026-11-01T09:00")
    assert as_utc(second) == utc("2026-12-01T09:00")
    gap = (as_utc(second) - as_utc(first)).total_seconds()
    assert jobs._schedule_cadence_seconds(schedule) == gap == 30 * 86400


# Fields a completed run writes; every other field keeps its creation value.
RUN_FIELDS = {
    "last_run_at", "last_status", "last_error", "last_delivery_error", "failure_streak",
    "fire_claim", "repeat", "next_run_at", "state",
}


def test_completed_run_saves_the_same_record_fields(clock):
    jobs_file = clock.home / "cron" / "jobs.json"
    clock.now = utc("2026-10-24T23:00")
    with jobs.use_cron_store(clock.home):
        job = jobs.create_job(
            prompt="Back up",
            schedule="30 2 * * *",
            name="daily 02:30",
            model="test-model",
            provider="test-provider",
        )
        created = json.loads(jobs_file.read_bytes())["jobs"][0]
        clock.now = utc("2026-10-25T00:30")  # 02:30 at +02:00, the first pass
        assert [due["id"] for due in jobs.get_due_jobs()] == [job["id"]]
        jobs.advance_next_runs([job["id"]])
        claimed = jobs.claim_job_for_fire(job["id"], return_job=True)
        clock.now = utc("2026-10-25T00:32:30")
        assert jobs.mark_job_run(job["id"], True, expected_fire_owner=claimed["fire_claim"]["by"])
        record = json.loads(jobs_file.read_bytes())["jobs"][0]
    assert set(record) == set(created) | {"fire_claim"}
    assert {key: value for key, value in record.items() if key not in RUN_FIELDS} == {
        key: value for key, value in created.items() if key not in RUN_FIELDS
    }
    assert record["last_run_at"] == "2026-10-25T02:32:30+02:00"
    assert record["last_status"] == "ok"
    assert record["last_error"] is None
    assert record["last_delivery_error"] is None
    assert record["failure_streak"] == 0
    assert record["fire_claim"] is None
    assert record["repeat"] == {"times": None, "completed": 1}
    assert record["state"] == "scheduled"
    assert record["enabled"] is True
    assert record["next_run_at"] == "2026-10-26T02:30:00+01:00"
