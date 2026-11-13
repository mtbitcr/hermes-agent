"""Card B1 of the daily project review plan: the review state and the daily cycle (no model yet).

``review_tick`` runs once per dispatcher tick. Once per Vienna day, from the stored review hour on, it looks at each
allowlisted active Project: a Project with no news since its last review is settled by the kernel alone, and a Project
with news gets one review task on the summary board. A daily cap counts every review task, retries included, and keeps
one more slot for the cross-project review. Everything is off until the owner switches it on.

Card B1 has no result reader yet (card B5), so its harvest refuses every finished review and counts a failure.
"""

from __future__ import annotations

import contextlib
import importlib
import json
import os
import sqlite3
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest
import yaml

from hermes_cli import kanban_db, owner_workspace as ow, projects_db
from hermes_cli import owner_project_summary_events as events
from plugins.dashboard_auth.raphael_workspace import model_policy

PLANNER = "raphael-planner"
BOARD = "project-summaries"
TITLE = "Review the project: "
CREATOR = "project-review"
MINUTE = 60
HOUR = 3600
DAY = 24 * HOUR
VIENNA = ZoneInfo("Europe/Vienna")

_CARDS = ("Book the meeting room", "Write the invitations", "Order the coffee")
_BAD_RESULT = {"project_review": {"headline": "The room is booked.", "extra": "not allowed"}}


def _ops():
    from hermes_cli import owner_project_review

    return owner_project_review


def _at(year, month, day, hour=0, minute=0, second=0, *, fold=0) -> int:
    """The epoch second of a Vienna wall-clock time; ``fold=1`` is the second pass through a repeated hour."""
    return int(datetime(year, month, day, hour, minute, second, tzinfo=VIENNA, fold=fold).timestamp())


def _utc(year, month, day, hour=0, minute=0, second=0) -> int:
    return int(datetime(year, month, day, hour, minute, second, tzinfo=ZoneInfo("UTC")).timestamp())


class _ProcessDied(BaseException):
    """A gateway restart as the interrupted code sees it: nothing after this point runs."""


@contextlib.contextmanager
def _written_at(instant: int):
    """What the kernel writes inside is stamped ``instant``: it stamps each event with ``time.time()``."""
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(time, "time", lambda: instant)
        yield


@pytest.fixture
def root(tmp_path, monkeypatch):
    """The shared root, entered as the default profile."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.delenv("HERMES_KANBAN_HOME", raising=False)
    monkeypatch.setattr(
        model_policy, "configured_assignment_for",
        lambda profile: model_policy.assignment_for(profile, "anthropic"),
    )
    return home


@pytest.fixture
def owner(root):
    return ow.OwnerContext(actor="default", profile="default", session="run_owt")


def _configure(home: Path, **values) -> None:
    """Store the owner's ``owner_review`` settings in the home's config.yaml."""
    (home / "config.yaml").write_text(yaml.safe_dump({"owner_review": values}), encoding="utf-8")


def _switch_on(root: Path, *allowed: str, **values) -> None:
    """The owner's stored settings once the review is on, for the named Projects."""
    stored = {
        "enabled": True, "allowlist": list(allowed), "closure": True, "digest": True,
        "review_hour": 6, "daily_cap": 20, "portfolio_cap": 1, "stale_days": 14,
    }
    _configure(root, **{**stored, **values})


def _rows(path: Path, sql: str, params=()) -> list:
    conn = sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True)
    try:
        return conn.execute(sql, params).fetchall()
    finally:
        conn.close()


def _board_db(root: Path, slug: str) -> Path:
    return root / "kanban" / "boards" / slug / "kanban.db"


@contextlib.contextmanager
def _board(root: Path, slug: str = BOARD):
    with kanban_db.connect_closing(_board_db(root, slug), board=slug) as conn:
        yield conn


def _add_cards(root: Path, project: dict, titles) -> None:
    with _board(root, project["board"]) as conn:
        for title in titles:
            kanban_db.create_task(conn, title=title, project_id=project["project_id"])


def _complete_cards(root: Path, project: dict, titles) -> None:
    """Complete the Project's cards of these titles through the kernel."""
    with _board(root, project["board"]) as conn:
        for title in titles:
            card = conn.execute("SELECT id FROM tasks WHERE title = ?", (title,)).fetchone()[0]
            assert kanban_db.complete_task(conn, card, summary="Done.")


def _steward_state(project: dict) -> str:
    return ow.project_steward_snapshot(project_id=project["project_id"])["execution"]["state"]


def _project(root: Path, owner, name: str, cards=()) -> dict:
    """An owner Project with its own board, as the owner kernel creates it, and its first cards."""
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(ow, "_confirm", lambda *_args, **_kwargs: {"approved": True})
        project = ow.bootstrap(owner, idempotency_key="setup-" + name.lower().replace(" ", "-"), name=name)
    assert project["ok"] is True
    project["slug"] = next(
        item["slug"] for item in ow.list_committed_projects(owner)
        if str(item["project_id"]) == project["project_id"]
    )
    _add_cards(root, project, cards)
    return project


def _quick_project(root: Path, name: str, cards: int) -> dict:
    """A registry Project with a board bound to it and ``cards`` new cards, made without the owner kernel."""
    slug = name.lower().replace(" ", "-")
    with projects_db.connect_closing(root / "projects.db") as conn:
        project_id = projects_db.create_project(conn, name=name, board_slug=slug)
    kanban_db.create_board(slug, name=name, project_id=project_id)
    project = {"project_id": project_id, "board": slug, "slug": slug}
    _add_cards(root, project, [f"Card {n} of {name}" for n in range(cards)])
    return project


def _last_event(root: Path, project: dict) -> int:
    """The newest event id of the Project's board, counted on a separate read-only connection."""
    return _rows(_board_db(root, project["board"]), "SELECT COALESCE(MAX(id), 0) FROM task_events")[0][0]


def _seed_reviewed(root: Path, project: dict, last_reviewed_date: str, **columns) -> None:
    """A Project the review saw before: reviewed on ``last_reviewed_date`` with every event up to now read."""
    values = {
        "project_id": project["project_id"], "event_cursor": _last_event(root, project),
        "last_reviewed_date": last_reviewed_date, "last_news_date": last_reviewed_date, "updated_at": 0,
        **columns,
    }
    names = ", ".join(values)
    with projects_db.connect_closing(root / "projects.db") as conn:
        with kanban_db.write_txn(conn):
            conn.execute(
                f"INSERT INTO project_review_state ({names}) VALUES ({', '.join('?' * len(values))})",
                tuple(values.values()),
            )


_STATE = (
    "open_task_id", "open_review_date", "failures", "last_reviewed_date", "last_reviewed_at", "last_news_date",
    "event_cursor", "delivery_cursor", "release_cursor", "updated_at",
)


def _state(root: Path, project: dict) -> dict | None:
    rows = _rows(
        root / "projects.db",
        f"SELECT {', '.join(_STATE)} FROM project_review_state WHERE project_id = ?",
        (project["project_id"],),
    )
    return dict(zip(_STATE, rows[0])) if rows else None


def _budget(root: Path, review_date: str) -> tuple:
    rows = _rows(
        root / "projects.db",
        "SELECT project_reviews, portfolio_reviews FROM review_budget WHERE review_date = ?",
        (review_date,),
    )
    return tuple(rows[0]) if rows else (0, 0)


def _review_tasks(root: Path, *, archived: bool = False) -> list:
    """The review tasks on the summary board, read on a separate connection."""
    if not _board_db(root, BOARD).exists():
        return []
    with _board(root) as conn:
        ids = [
            row[0] for row in conn.execute(
                "SELECT id FROM tasks WHERE created_by = ? ORDER BY created_at, rowid", (CREATOR,)
            )
        ]
        tasks = [kanban_db.get_task(conn, task_id) for task_id in ids]
    return [task for task in tasks if archived or task.status != "archived"]


def _finish(root: Path, task_id: str, metadata: dict | None) -> None:
    """What a dispatched worker does: claim the card, then complete it with metadata."""
    with _board(root) as conn:
        assert kanban_db.claim_task(conn, task_id) is not None
        assert kanban_db.complete_task(conn, task_id, summary="Reviewed the project.", metadata=metadata)


def _no_review_store_touched(root: Path) -> None:
    assert _rows(root / "projects.db", "SELECT count(*) FROM project_review_state") == [(0,)]
    assert _rows(root / "projects.db", "SELECT count(*) FROM review_budget") == [(0,)]
    assert _review_tasks(root, archived=True) == []


# ---------------------------------------------------------------------------------------------------------------
# Settings: everything off in code; the owner's stored values are read from the existing config.
# ---------------------------------------------------------------------------------------------------------------


def test_settings_are_off_until_the_owner_stores_them(root):
    settings = _ops().read_settings()

    assert settings.enabled is False
    assert settings.allowlist == ()
    assert settings.closure is False
    assert settings.digest is False
    assert (settings.review_hour, settings.daily_cap, settings.portfolio_cap, settings.stale_days) == (6, 20, 1, 14)


def test_stored_settings_are_read_from_the_config(root):
    _switch_on(root, "proj-garden", review_hour=7, daily_cap=12, portfolio_cap=2, stale_days=21)

    settings = _ops().read_settings()

    assert settings.enabled is True
    assert settings.allowlist == ("proj-garden",)
    assert (settings.closure, settings.digest) == (True, True)
    assert (settings.review_hour, settings.daily_cap, settings.portfolio_cap, settings.stale_days) == (7, 12, 2, 21)


def test_settings_that_cannot_be_read_stay_off(root):
    _configure(
        root, enabled="yes", allowlist="proj-garden", closure=1, digest="on",
        review_hour=24, daily_cap=-1, portfolio_cap="one", stale_days=0,
    )

    settings = _ops().read_settings()

    assert (settings.enabled, settings.allowlist, settings.closure, settings.digest) == (False, (), False, False)
    assert (settings.review_hour, settings.daily_cap, settings.portfolio_cap, settings.stale_days) == (6, 20, 1, 14)


# ---------------------------------------------------------------------------------------------------------------
# Eligibility: the flag, then the allowlist (operator rule L3), then the plan's active Project.
# ---------------------------------------------------------------------------------------------------------------


def test_flag_off_makes_nothing_eligible(root, owner):
    garden = _project(root, owner, "Garden Plan", _CARDS)
    _switch_on(root, garden["project_id"], enabled=False)

    assert _ops().review_tick(now=_at(2026, 11, 9, 7)) == 0

    _no_review_store_touched(root)


def test_no_settings_at_all_makes_nothing_eligible(root, owner):
    _project(root, owner, "Garden Plan", _CARDS)

    assert _ops().review_tick(now=_at(2026, 11, 9, 7)) == 0

    _no_review_store_touched(root)


def test_empty_allowlist_makes_no_project_eligible(root, owner):
    _project(root, owner, "Garden Plan", _CARDS)
    _project(root, owner, "Planning Desk", _CARDS)
    _switch_on(root)

    assert _ops().review_tick(now=_at(2026, 11, 9, 7)) == 0

    _no_review_store_touched(root)


def test_only_allowlisted_projects_are_reviewed(root, owner):
    garden = _project(root, owner, "Garden Plan", _CARDS)
    desk = _project(root, owner, "Planning Desk", _CARDS)
    _switch_on(root, garden["slug"])

    assert _ops().review_tick(now=_at(2026, 11, 9, 7)) == 1

    [task] = _review_tasks(root)
    assert task.title == TITLE + "Garden Plan"
    assert _state(root, desk) is None


def test_a_project_taken_off_the_allowlist_has_its_open_review_closed(root, owner):
    garden = _project(root, owner, "Garden Plan", _CARDS)
    _switch_on(root, garden["project_id"])
    ops = _ops()
    ops.review_tick(now=_at(2026, 11, 9, 7))
    [task] = _review_tasks(root)
    _switch_on(root, daily_cap=19)

    ops.review_tick(now=_at(2026, 11, 9, 8))

    assert _review_tasks(root) == []
    state = _state(root, garden)
    assert (state["open_task_id"], state["failures"], state["last_reviewed_date"]) == (None, 0, None)


def test_a_completed_project_is_not_reviewed(root, owner):
    shed = _project(root, owner, "Garden Shed", ["Paint the shed"])
    _complete_cards(root, shed, ["Paint the shed"])
    assert _steward_state(shed) == "complete"
    _switch_on(root, shed["project_id"])

    assert _ops().review_tick(now=_at(2026, 11, 9, 7)) == 0

    _no_review_store_touched(root)


def test_an_active_project_is_reviewed(root, owner):
    shed = _project(root, owner, "Garden Shed", ["Paint the shed", "Fix the door"])
    _complete_cards(root, shed, ["Paint the shed"])
    assert _steward_state(shed) not in ("complete", "deleted")
    _switch_on(root, shed["project_id"])

    assert _ops().review_tick(now=_at(2026, 11, 9, 7)) == 1

    [task] = _review_tasks(root)
    assert task.title == TITLE + "Garden Shed"
    assert _state(root, shed)["open_task_id"] == task.id


# ---------------------------------------------------------------------------------------------------------------
# T1 creation, T2 quiet day, T3 missed run.
# ---------------------------------------------------------------------------------------------------------------


def test_t1_first_review_opens_one_task(root, owner):
    garden = _project(root, owner, "Garden Plan", _CARDS)
    _switch_on(root, garden["project_id"])
    ops = _ops()
    assert _state(root, garden) is None

    assert ops.review_tick(now=_at(2026, 11, 9, 5, 59, 59)) == 0
    assert _review_tasks(root, archived=True) == []
    assert ops.review_tick(now=_at(2026, 11, 9, 7)) == 1
    assert ops.review_tick(now=_at(2026, 11, 9, 7, 5)) == 0

    [task] = _review_tasks(root, archived=True)
    assert task.title == TITLE + "Garden Plan"
    assert task.created_by == CREATOR
    assert task.idempotency_key == f"{CREATOR}:{garden['project_id']}:2026-11-09:0"
    assert (task.assignee, task.status, task.project_id, task.owned_paths) == (PLANNER, "ready", None, [])
    assert task.execution_tier == "routine"
    route = model_policy.task_assignment_for(PLANNER, "anthropic", "routine")
    assert (task.provider_override, task.model_override) == (route.provider, route.model)
    assert kanban_db.policy_lock_error(
        task.model_policy_lock, task.assignee, task.provider_override, task.model_override,
        task.reasoning_effort, task.execution_tier,
    ) is None
    last_line = task.body.rstrip("\n").splitlines()[-1]
    assert last_line == events.covered_line(events.Cursors(_last_event(root, garden), 0, 0))

    state = _state(root, garden)
    assert (state["open_task_id"], state["open_review_date"], state["failures"]) == (task.id, "2026-11-09", 0)
    assert state["last_reviewed_date"] is None
    assert state["event_cursor"] == 0
    assert _budget(root, "2026-11-09") == (1, 0)


def test_every_task_the_cycle_publishes_is_created_by_project_review(root, owner):
    # Card B-S1 limits a task of this creator to kanban_show and kanban_complete at dispatch.
    garden = _project(root, owner, "Garden Plan", _CARDS)
    desk = _project(root, owner, "Planning Desk", _CARDS)
    _switch_on(root, garden["project_id"], desk["project_id"])
    ops = _ops()
    made = ops.review_tick(now=_at(2026, 11, 9, 7))
    for task in _review_tasks(root):
        _finish(root, task.id, _BAD_RESULT)
    made += ops.review_tick(now=_at(2026, 11, 9, 7, 5))
    made += ops.review_tick(now=_at(2026, 11, 9, 8, 10))

    # Every row of the summary board, first reviews and retries alike, read without a filter on the creator.
    published = _rows(_board_db(root, BOARD), "SELECT created_by FROM tasks")
    assert made == len(published) == 4
    assert {creator for (creator,) in published} == {CREATOR}


def test_t2_quiet_day_makes_no_task(root, owner):
    with _written_at(_at(2026, 11, 8, 9)):
        garden = _project(root, owner, "Garden Plan", _CARDS)
    _seed_reviewed(root, garden, "2026-11-08")
    with _board(root, garden["board"]) as conn, _written_at(_at(2026, 11, 9, 6, 30)):
        card = conn.execute("SELECT id FROM tasks WHERE title = ?", (_CARDS[0],)).fetchone()[0]
        kanban_db.add_comment(conn, card, PLANNER, "Checked the room booking again.")
    _switch_on(root, garden["project_id"])
    now = _at(2026, 11, 9, 7)

    assert _ops().review_tick(now=now) == 0

    assert _review_tasks(root, archived=True) == []
    state = _state(root, garden)
    assert (state["last_reviewed_date"], state["last_reviewed_at"], state["open_task_id"]) == ("2026-11-09", now, None)
    assert state["event_cursor"] == _last_event(root, garden)
    assert state["last_news_date"] == "2026-11-08"
    assert _budget(root, "2026-11-09") == (0, 0)


def test_t3_missed_days_are_one_catch_up_review(root, owner):
    garden = _project(root, owner, "Garden Plan", _CARDS)
    _seed_reviewed(root, garden, "2026-11-06")
    _add_cards(root, garden, ["Send the agenda", "Print the name tags", "Collect the replies"])
    _switch_on(root, garden["project_id"])
    ops = _ops()

    assert ops.review_tick(now=_at(2026, 11, 9, 7)) == 1

    [task] = _review_tasks(root, archived=True)
    assert task.idempotency_key.endswith(":2026-11-09:0")
    assert ops.review_window(task.body) == ("2026-11-07", "2026-11-09")
    state = _state(root, garden)
    assert (state["open_review_date"], state["last_reviewed_date"]) == ("2026-11-09", "2026-11-06")
    assert _rows(root / "projects.db", "SELECT count(*) FROM project_review_state") == [(1,)]
    assert _budget(root, "2026-11-09") == (1, 0)
    assert _rows(root / "projects.db", "SELECT review_date FROM review_budget") == [("2026-11-09",)]


def test_a_project_never_reviewed_covers_everything_up_to_today(root, owner):
    garden = _project(root, owner, "Garden Plan", _CARDS)
    _switch_on(root, garden["project_id"])
    ops = _ops()

    ops.review_tick(now=_at(2026, 11, 9, 7))

    [task] = _review_tasks(root)
    assert ops.review_window(task.body) == (None, "2026-11-09")


# ---------------------------------------------------------------------------------------------------------------
# REVIEW_EVENT_CLASS: owner comments and suggestion events are news for the review, and only for the review.
# ---------------------------------------------------------------------------------------------------------------


def test_an_owner_comment_is_news_for_the_review_only(root, owner):
    garden = _project(root, owner, "Garden Plan", _CARDS)
    _seed_reviewed(root, garden, "2026-11-08")
    with _board(root, garden["board"]) as conn:
        card = conn.execute("SELECT id FROM tasks WHERE title = ?", (_CARDS[0],)).fetchone()[0]
        kanban_db.add_comment(conn, card, owner.actor, "Please book the larger room instead.")
    _switch_on(root, garden["project_id"])
    with projects_db.connect_closing(root / "projects.db") as conn:
        project = projects_db.get_project(conn, garden["project_id"])
    seen = events.Cursors(_last_event(root, garden) - 1, 0, 0)

    assert events.read_events(project, seen).material == 0
    assert _ops().read_review_events(project, seen).material == 1
    assert _ops().review_tick(now=_at(2026, 11, 9, 7)) == 1
    assert len(_review_tasks(root)) == 1


def test_a_new_suggestion_is_news_for_the_review(root, owner):
    garden = _project(root, owner, "Garden Plan", _CARDS)
    _seed_reviewed(root, garden, "2026-11-08")
    evidence = dict(
        schema_version=1, need="Repeated capability gap", expected_benefit="Complete work safely",
        requested_scope={flag: False for flag in kanban_db.RECOMMENDATION_SCOPE_FLAGS},
        risks="Low", cost="No added cost", rollback="Remove the staged configuration",
    )
    with _board(root, garden["board"]) as conn, _written_at(_at(2026, 11, 9, 6, 30)):
        kanban_db.create_recommendation(
            conn, project_id=garden["project_id"], target_profile="worker", recommendation_kind="skill",
            recommendation_subject_id="calendar", recommendation_label="Learn the calendar",
            recommendation_evidence=evidence, provenance_authority="static-analyzer",
            provenance_ref="finding-123", provenance_observed_at=1_700_000_000,
        )
    _switch_on(root, garden["project_id"])

    assert _ops().review_tick(now=_at(2026, 11, 9, 7)) == 1

    state = _state(root, garden)
    assert state["open_task_id"] is not None
    assert state["last_news_date"] == "2026-11-09"


# ---------------------------------------------------------------------------------------------------------------
# T4 retry: a refused result is a failure; the backoff decides the next try; the review day never moves.
# ---------------------------------------------------------------------------------------------------------------


def test_t4_refused_results_back_off_and_never_move_the_review_day(root, owner):
    garden = _project(root, owner, "Garden Plan", _CARDS)
    _seed_reviewed(root, garden, "2026-11-06", event_cursor=0)
    _switch_on(root, garden["project_id"])
    ops = _ops()
    prefix = f"{CREATOR}:{garden['project_id']}:"

    clock = _at(2026, 11, 9, 7)
    assert ops.review_tick(now=clock) == 1
    for failures, wait in ((1, HOUR), (2, 2 * HOUR)):
        [task] = _review_tasks(root)
        _finish(root, task.id, _BAD_RESULT)
        clock += 5 * MINUTE
        assert ops.review_tick(now=clock) == 0
        assert [t.status for t in _review_tasks(root, archived=True) if t.id == task.id] == ["archived"]
        state = _state(root, garden)
        assert (state["failures"], state["open_task_id"], state["last_reviewed_date"]) == (
            failures, None, "2026-11-06",
        )
        assert ops.review_tick(now=clock + wait - MINUTE) == 0
        assert _review_tasks(root) == []
        clock += wait + MINUTE
        assert ops.review_tick(now=clock) == 1
        [retry] = _review_tasks(root)
        assert retry.idempotency_key == f"{prefix}2026-11-09:{failures}"

    [task] = _review_tasks(root)
    _finish(root, task.id, _BAD_RESULT)
    clock += 5 * MINUTE
    ops.review_tick(now=clock)
    assert _state(root, garden)["failures"] == 3
    assert _budget(root, "2026-11-09") == (3, 0)

    assert ops.review_tick(now=clock + DAY - MINUTE) == 0
    assert _review_tasks(root) == []
    assert ops.review_tick(now=clock + DAY + MINUTE) == 1
    [task] = _review_tasks(root)
    assert task.idempotency_key == f"{prefix}2026-11-10:3"
    state = _state(root, garden)
    assert (state["last_reviewed_date"], state["event_cursor"]) == ("2026-11-06", 0)
    assert _budget(root, "2026-11-10") == (1, 0)


def test_a_task_that_is_still_open_is_left_alone(root, owner):
    garden = _project(root, owner, "Garden Plan", _CARDS)
    _switch_on(root, garden["project_id"])
    ops = _ops()
    ops.review_tick(now=_at(2026, 11, 9, 7))
    [task] = _review_tasks(root)

    assert ops.review_tick(now=_at(2026, 11, 10, 7)) == 0

    assert [t.id for t in _review_tasks(root, archived=True)] == [task.id]
    state = _state(root, garden)
    assert (state["open_task_id"], state["open_review_date"], state["failures"]) == (task.id, "2026-11-09", 0)
    assert _budget(root, "2026-11-10") == (0, 0)


# ---------------------------------------------------------------------------------------------------------------
# T7 time zone and day boundary.
# ---------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("instant", "review_hour", "expected"),
    [
        # Spring forward, 2026-03-29: 01:59 CET is followed by 03:00 CEST.
        (_utc(2026, 3, 29, 0, 59), 6, None),
        (_utc(2026, 3, 29, 1, 0), 6, None),
        (_utc(2026, 3, 29, 3, 59, 59), 6, None),
        (_utc(2026, 3, 29, 4, 0), 6, "2026-03-29"),
        (_utc(2026, 3, 29, 1, 0), 3, "2026-03-29"),
        (_utc(2026, 3, 29, 21, 59, 59), 6, "2026-03-29"),
        (_utc(2026, 3, 29, 22, 0), 0, "2026-03-30"),
        # Fall back, 2026-10-25: 02:30 comes twice, first in CEST, then in CET.
        (_utc(2026, 10, 25, 0, 30), 2, "2026-10-25"),
        (_utc(2026, 10, 25, 1, 30), 2, "2026-10-25"),
        (_utc(2026, 10, 25, 4, 30), 6, None),
        (_utc(2026, 10, 25, 5, 0), 6, "2026-10-25"),
        # Midnight: 23:59:59 and 00:00:00 local.
        (_utc(2026, 11, 9, 22, 59, 59), 6, "2026-11-09"),
        (_utc(2026, 11, 9, 23, 0), 6, None),
        (_utc(2026, 11, 9, 23, 0), 0, "2026-11-10"),
    ],
)
def test_t7_the_review_day_is_the_vienna_date_from_the_review_hour_on(instant, review_hour, expected):
    assert _ops().review_date(instant, review_hour) == expected


def test_t7_the_wall_clock_instants_are_the_ones_the_plan_names():
    assert _at(2026, 3, 29, 1, 59) == _utc(2026, 3, 29, 0, 59)
    assert _at(2026, 3, 29, 3, 0) == _utc(2026, 3, 29, 1, 0)
    assert _at(2026, 10, 25, 2, 30) == _utc(2026, 10, 25, 0, 30)
    assert _at(2026, 10, 25, 2, 30, fold=1) == _utc(2026, 10, 25, 1, 30)
    assert _at(2026, 11, 9, 23, 59, 59) == _utc(2026, 11, 9, 22, 59, 59)


def test_t7_spring_forward_day_has_one_review(root, owner):
    quiet = _project(root, owner, "Quiet Corner")
    _seed_reviewed(root, quiet, "2026-03-28")
    busy = _project(root, owner, "Garden Plan", _CARDS)
    _switch_on(root, quiet["project_id"], busy["project_id"])
    ops = _ops()

    for instant in (_at(2026, 3, 29, 1, 59), _at(2026, 3, 29, 3, 0), _at(2026, 3, 29, 5, 59, 59)):
        assert ops.review_tick(now=instant) == 0
        assert _state(root, quiet)["last_reviewed_date"] == "2026-03-28"
    reviewed_at = _at(2026, 3, 29, 6, 0)
    assert ops.review_tick(now=reviewed_at) == 1
    for instant in (_at(2026, 3, 29, 12, 0), _at(2026, 3, 29, 23, 59, 59), _at(2026, 3, 30, 0, 0)):
        assert ops.review_tick(now=instant) == 0

    assert (_state(root, quiet)["last_reviewed_date"], _state(root, quiet)["last_reviewed_at"]) == (
        "2026-03-29", reviewed_at,
    )
    assert [task.idempotency_key.split(":")[2] for task in _review_tasks(root, archived=True)] == ["2026-03-29"]


def test_t7_the_repeated_fall_back_hour_makes_no_second_review(root, owner):
    quiet = _project(root, owner, "Quiet Corner")
    _seed_reviewed(root, quiet, "2026-10-24")
    busy = _project(root, owner, "Garden Plan", _CARDS)
    _switch_on(root, quiet["project_id"], busy["project_id"], review_hour=2)
    ops = _ops()
    first = _at(2026, 10, 25, 2, 30)

    assert ops.review_tick(now=first) == 1
    assert ops.review_tick(now=_at(2026, 10, 25, 2, 30, fold=1)) == 0

    assert (_state(root, quiet)["last_reviewed_date"], _state(root, quiet)["last_reviewed_at"]) == (
        "2026-10-25", first,
    )
    [task] = _review_tasks(root, archived=True)
    assert task.idempotency_key.endswith(":2026-10-25:0")
    assert _budget(root, "2026-10-25") == (1, 0)


@pytest.mark.parametrize(("review_hour", "next_day"), [(0, "2026-11-10"), (6, "2026-11-09")])
def test_t7_midnight_starts_a_new_review_day_only_when_the_hour_allows_it(root, owner, review_hour, next_day):
    quiet = _project(root, owner, "Quiet Corner")
    _seed_reviewed(root, quiet, "2026-11-08")
    busy = _project(root, owner, "Garden Plan", _CARDS)
    _switch_on(root, quiet["project_id"], busy["project_id"], review_hour=review_hour)
    ops = _ops()

    assert ops.review_tick(now=_at(2026, 11, 9, 23, 59, 59)) == 1
    assert _state(root, quiet)["last_reviewed_date"] == "2026-11-09"
    assert ops.review_tick(now=_at(2026, 11, 10, 0, 0, 0)) == 0

    assert _state(root, quiet)["last_reviewed_date"] == next_day
    [task] = _review_tasks(root, archived=True)
    assert task.idempotency_key.endswith(":2026-11-09:0")


# ---------------------------------------------------------------------------------------------------------------
# The daily cap: 20 project reviews and 1 cross-project review per Vienna day.
# ---------------------------------------------------------------------------------------------------------------


def test_the_cap_stops_at_twenty_project_reviews_and_resets_the_next_day(root):
    projects = [_quick_project(root, f"Project {n:02d}", 2) for n in range(20)]
    last = _quick_project(root, "Project Late", 1)
    _switch_on(root, *(p["project_id"] for p in [last, *projects]))
    ops = _ops()
    clock = _at(2026, 11, 9, 7)

    made = []
    for tick in range(8):
        made.append(ops.review_tick(now=clock + tick * MINUTE))

    assert made == [3, 3, 3, 3, 3, 3, 2, 0]
    assert len(_review_tasks(root)) == 20
    assert _budget(root, "2026-11-09") == (20, 0)
    assert _state(root, last)["open_task_id"] is None
    assert all(_state(root, p)["open_task_id"] is not None for p in projects)

    with projects_db.connect_closing(root / "projects.db") as conn:
        assert ops.take_portfolio_slot(conn, "2026-11-09", 1, clock) is True
        assert ops.take_portfolio_slot(conn, "2026-11-09", 1, clock) is False
    assert _budget(root, "2026-11-09") == (20, 1)

    assert ops.review_tick(now=_at(2026, 11, 10, 7)) == 1
    assert _state(root, last)["open_review_date"] == "2026-11-10"
    assert _budget(root, "2026-11-10") == (1, 0)
    assert _budget(root, "2026-11-09") == (20, 1)
    with projects_db.connect_closing(root / "projects.db") as conn:
        assert ops.take_portfolio_slot(conn, "2026-11-10", 1, clock + DAY) is True


def test_the_cap_waits_by_urgency_then_news(root):
    calm = _quick_project(root, "Calm Notes", 1)
    busy = _quick_project(root, "Busy Notes", 3)
    urgent = _quick_project(root, "Urgent Notes", 1)
    with _board(root, urgent["board"]) as conn:
        card = conn.execute("SELECT id FROM tasks LIMIT 1").fetchone()[0]
        kanban_db.block_task(conn, card, reason="Which room?", kind="needs_input")
    _switch_on(root, calm["project_id"], busy["project_id"], urgent["project_id"], daily_cap=2)

    assert _ops().review_tick(now=_at(2026, 11, 9, 7)) == 2

    assert sorted(task.title for task in _review_tasks(root)) == [TITLE + "Busy Notes", TITLE + "Urgent Notes"]
    assert _state(root, calm)["open_task_id"] is None


def test_the_project_the_cap_makes_wait_gets_the_day_s_mark_and_its_review_the_next_day(root):
    calm = _quick_project(root, "Calm Notes", 1)
    busy = _quick_project(root, "Busy Notes", 3)
    urgent = _quick_project(root, "Urgent Notes", 1)
    with _board(root, urgent["board"]) as conn:
        card = conn.execute("SELECT id FROM tasks LIMIT 1").fetchone()[0]
        kanban_db.block_task(conn, card, reason="Which room?", kind="needs_input")
    _switch_on(root, calm["project_id"], busy["project_id"], urgent["project_id"], daily_cap=2)
    ops = _ops()
    position = (
        "SELECT event_cursor, delivery_cursor, release_cursor, conversation_cursor, last_reviewed_date, open_task_id, "
        "failures, updated_at FROM project_review_state WHERE project_id = ?"
    )
    marks = "SELECT project_id, cap_wait_date, cap_wait_at FROM project_review_state WHERE cap_wait_date IS NOT NULL"

    assert ops.review_tick(now=_at(2026, 11, 9, 7)) == 2
    waiting = _rows(root / "projects.db", position, (calm["project_id"],))
    assert ops.review_tick(now=_at(2026, 11, 9, 7, 30)) == 0

    # Owner rule of 2026-11-12: the cap delay is the day's mark in the review state, set once; no brief row.
    assert _rows(root / "projects.db", marks) == [(calm["project_id"], "2026-11-09", _at(2026, 11, 9, 7))]
    assert _rows(root / "projects.db", "SELECT count(*) FROM project_review_briefs") == [(0,)]
    assert _rows(root / "projects.db", position, (calm["project_id"],)) == waiting
    assert _budget(root, "2026-11-09") == (2, 0)
    assert sorted(task.title for task in _review_tasks(root)) == [TITLE + "Busy Notes", TITLE + "Urgent Notes"]

    # The next day reviews it as usual; the two reviews of the day before still hold their own Projects.
    assert ops.review_tick(now=_at(2026, 11, 10, 7)) == 1
    assert _state(root, calm)["open_task_id"] is not None
    assert sorted(task.title for task in _review_tasks(root)) == [
        TITLE + "Busy Notes", TITLE + "Calm Notes", TITLE + "Urgent Notes",
    ]
    assert _budget(root, "2026-11-10") == (1, 0)
    assert _rows(root / "projects.db", marks) == [(calm["project_id"], "2026-11-09", _at(2026, 11, 9, 7))]


# ---------------------------------------------------------------------------------------------------------------
# Quiet-day tracking: a Project is stale only after the stored number of quiet days.
# ---------------------------------------------------------------------------------------------------------------


def test_staleness_flips_only_after_fourteen_quiet_days(root, owner):
    with _written_at(_at(2026, 11, 1, 9)):
        quiet = _project(root, owner, "Quiet Corner")
    _seed_reviewed(root, quiet, "2026-11-01")
    _switch_on(root, quiet["project_id"])
    ops = _ops()

    seen = []
    for day in range(2, 17):
        ops.review_tick(now=_at(2026, 11, day, 7))
        state = _state(root, quiet)
        today = f"2026-11-{day:02d}"
        assert state["last_reviewed_date"] == today
        seen.append((ops.quiet_days(state["last_news_date"], today), ops.is_stale(state["last_news_date"], today, 14)))

    assert seen[12] == (13, False)
    assert seen[13] == (14, True)
    assert [stale for _, stale in seen].index(True) == 13


def test_news_ends_a_quiet_run(root, owner):
    with _written_at(_at(2026, 11, 1, 9)):
        quiet = _project(root, owner, "Quiet Corner")
    _seed_reviewed(root, quiet, "2026-11-01")
    _switch_on(root, quiet["project_id"])
    ops = _ops()
    ops.review_tick(now=_at(2026, 11, 14, 7))
    with _written_at(_at(2026, 11, 15, 6, 30)):
        _add_cards(root, quiet, ["Paint the shelves"])

    ops.review_tick(now=_at(2026, 11, 15, 7))

    state = _state(root, quiet)
    assert state["last_news_date"] == "2026-11-15"
    assert ops.is_stale(state["last_news_date"], "2026-11-15", 14) is False
    assert ops.is_stale(state["last_news_date"], "2026-11-28", 14) is False
    assert ops.is_stale(state["last_news_date"], "2026-11-29", 14) is True


def _news_on_the_ninth(root: Path) -> dict:
    """A Project whose one event is a card made on 2026-11-09, before the review hour; nothing happens after it."""
    project = _quick_project(root, "Late Notes", 0)
    with _written_at(_at(2026, 11, 9, 5, 30)):
        _add_cards(root, project, ["Book the meeting room"])
    return project


def _quiet_since_the_ninth(ops, state: dict) -> None:
    """The news of 2026-11-09 is still the last: fourteen quiet days are reached on 2026-11-23."""
    assert (state["last_news_date"], state["event_cursor"], state["last_reviewed_date"]) == ("2026-11-09", 0, None)
    assert ops.is_stale(state["last_news_date"], "2026-11-22", 14) is False
    assert ops.is_stale(state["last_news_date"], "2026-11-23", 14) is True


def test_a_review_the_cap_defers_is_no_news(root):
    late = _news_on_the_ninth(root)
    _switch_on(root, late["project_id"], daily_cap=0)
    ops = _ops()

    for day in range(9, 24):
        assert ops.review_tick(now=_at(2026, 11, day, 7)) == 0

    assert _review_tasks(root, archived=True) == []
    _quiet_since_the_ninth(ops, _state(root, late))


def test_failed_reviews_and_their_retries_are_no_news(root):
    late = _news_on_the_ninth(root)
    _switch_on(root, late["project_id"])
    ops = _ops()

    made = 0
    for day in range(9, 24):
        for hour in (7, 10, 13, 16, 19, 22):
            made += ops.review_tick(now=_at(2026, 11, day, hour))
            for task in _review_tasks(root):
                _finish(root, task.id, _BAD_RESULT)

    state = _state(root, late)
    assert made > 3  # past the third failure the retry waits a whole day
    assert state["failures"] == made
    _quiet_since_the_ninth(ops, state)


def test_a_restart_is_no_news(root):
    late = _news_on_the_ninth(root)
    _switch_on(root, late["project_id"], daily_cap=0)
    for day in range(9, 16):
        _ops().review_tick(now=_at(2026, 11, day, 7))
    before = _state(root, late)

    _restart()
    for day in range(16, 24):
        assert _ops().review_tick(now=_at(2026, 11, day, 7)) == 0

    assert _state(root, late) == before
    _quiet_since_the_ninth(_ops(), before)


def test_a_new_event_still_ends_the_quiet_days_while_reviews_wait(root):
    late = _news_on_the_ninth(root)
    _switch_on(root, late["project_id"], daily_cap=0)
    ops = _ops()
    for day in range(9, 16):
        ops.review_tick(now=_at(2026, 11, day, 7))
    assert _state(root, late)["last_news_date"] == "2026-11-09"

    with _written_at(_at(2026, 11, 16, 6, 30)):
        _add_cards(root, late, ["Write the invitations"])
    for day in range(16, 24):
        ops.review_tick(now=_at(2026, 11, day, 7))

    state = _state(root, late)
    assert (state["last_news_date"], state["event_cursor"]) == ("2026-11-16", 0)
    assert ops.quiet_days(state["last_news_date"], "2026-11-23") == 7
    assert ops.is_stale(state["last_news_date"], "2026-11-23", 14) is False


# ---------------------------------------------------------------------------------------------------------------
# The quiet clock sees a new event on every tick: while the review is open, backs off or waits, and after a restart.
# ---------------------------------------------------------------------------------------------------------------


def _new_work_at(root: Path, project: dict, instant: int) -> None:
    """A genuine new work card on the Project's board, stamped ``instant``."""
    with _written_at(instant):
        _add_cards(root, project, ["Write the invitations"])


def _only_the_clock_moved(ops, root: Path, project: dict, before: dict) -> None:
    """The card of 07:06 on 2026-11-23 ended the quiet days, and nothing else in the Project's row moved."""
    after = _state(root, project)
    assert after["last_news_date"] == "2026-11-23"
    assert ops.is_stale(after["last_news_date"], "2026-11-23", 14) is False
    assert {name for name in _STATE if after[name] != before[name]} == {"last_news_date"}


def test_a_new_event_moves_the_quiet_clock_while_the_review_is_open(root):
    late = _news_on_the_ninth(root)
    _switch_on(root, late["project_id"])
    ops = _ops()
    assert ops.review_tick(now=_at(2026, 11, 23, 7)) == 1
    [task] = _review_tasks(root)
    before = _state(root, late)
    assert (before["open_task_id"], before["last_news_date"]) == (task.id, "2026-11-09")

    _new_work_at(root, late, _at(2026, 11, 23, 7, 6))
    assert ops.review_tick(now=_at(2026, 11, 23, 7, 7)) == 0

    _only_the_clock_moved(ops, root, late, before)
    assert _budget(root, "2026-11-23") == (1, 0)
    assert [t.id for t in _review_tasks(root, archived=True)] == [task.id]


def test_a_new_event_moves_the_quiet_clock_while_a_failed_review_backs_off(root):
    late = _news_on_the_ninth(root)
    _switch_on(root, late["project_id"])
    ops = _ops()
    assert ops.review_tick(now=_at(2026, 11, 23, 7)) == 1
    [task] = _review_tasks(root)
    _finish(root, task.id, _BAD_RESULT)
    failed_at = _at(2026, 11, 23, 7, 5)
    assert ops.review_tick(now=failed_at) == 0
    before = _state(root, late)
    assert (before["failures"], before["open_task_id"], before["updated_at"]) == (1, None, failed_at)
    assert before["last_news_date"] == "2026-11-09"

    _new_work_at(root, late, _at(2026, 11, 23, 7, 6))
    assert ops.review_tick(now=_at(2026, 11, 23, 7, 7)) == 0

    _only_the_clock_moved(ops, root, late, before)
    assert _budget(root, "2026-11-23") == (1, 0)
    assert [t.id for t in _review_tasks(root, archived=True)] == [task.id]
    # The backoff still runs from the failure, not from the tick that saw the new card.
    assert ops.review_tick(now=failed_at + HOUR - MINUTE) == 0
    assert ops.review_tick(now=failed_at + HOUR + MINUTE) == 1
    [retry] = _review_tasks(root)
    assert retry.idempotency_key.endswith(":2026-11-23:1")


def test_a_new_event_moves_the_quiet_clock_while_the_cap_defers_the_review(root):
    late = _news_on_the_ninth(root)
    _switch_on(root, late["project_id"], daily_cap=0)
    ops = _ops()
    assert ops.review_tick(now=_at(2026, 11, 23, 7)) == 0
    before = _state(root, late)
    assert (before["open_task_id"], before["last_news_date"]) == (None, "2026-11-09")

    _new_work_at(root, late, _at(2026, 11, 23, 7, 6))
    assert ops.review_tick(now=_at(2026, 11, 23, 7, 7)) == 0

    _only_the_clock_moved(ops, root, late, before)
    assert _budget(root, "2026-11-23") == (0, 0)
    assert _review_tasks(root, archived=True) == []


def test_a_new_event_after_a_restart_moves_the_quiet_clock(root):
    late = _news_on_the_ninth(root)
    _switch_on(root, late["project_id"])
    assert _ops().review_tick(now=_at(2026, 11, 23, 7)) == 1
    [task] = _review_tasks(root)
    before = _state(root, late)
    assert (before["open_task_id"], before["last_news_date"]) == (task.id, "2026-11-09")

    _restart()
    _new_work_at(root, late, _at(2026, 11, 23, 7, 6))
    assert _ops().review_tick(now=_at(2026, 11, 23, 7, 7)) == 0

    _only_the_clock_moved(_ops(), root, late, before)
    assert _budget(root, "2026-11-23") == (1, 0)
    assert [t.id for t in _review_tasks(root, archived=True)] == [task.id]


def test_without_a_new_event_the_quiet_clock_stays_through_an_open_review_and_its_backoff(root):
    late = _news_on_the_ninth(root)
    _switch_on(root, late["project_id"])
    ops = _ops()
    assert ops.review_tick(now=_at(2026, 11, 23, 7)) == 1
    assert ops.review_tick(now=_at(2026, 11, 23, 7, 3)) == 0
    _quiet_since_the_ninth(ops, _state(root, late))

    [task] = _review_tasks(root)
    _finish(root, task.id, _BAD_RESULT)
    assert ops.review_tick(now=_at(2026, 11, 23, 7, 5)) == 0
    assert ops.review_tick(now=_at(2026, 11, 23, 7, 7)) == 0

    state = _state(root, late)
    assert (state["failures"], state["open_task_id"]) == (1, None)
    _quiet_since_the_ninth(ops, state)


def test_news_behind_a_full_row_cap_gives_no_date_in_that_tick(root):
    late = _news_on_the_ninth(root)
    _switch_on(root, late["project_id"], daily_cap=0)
    ops = _ops()
    assert ops.review_tick(now=_at(2026, 11, 23, 7)) == 0
    assert _state(root, late)["last_news_date"] == "2026-11-09"
    _new_work_at(root, late, _at(2026, 11, 23, 7, 6))
    with _board(root, late["board"]) as conn, _written_at(_at(2026, 11, 23, 7, 6)):
        card = conn.execute("SELECT id FROM tasks WHERE title = ?", ("Write the invitations",)).fetchone()[0]
        for n in range(ops.NEWS_ROW_CAP):
            kanban_db.add_comment(conn, card, PLANNER, f"Checked the guest list, pass {n}.")

    assert ops.review_tick(now=_at(2026, 11, 23, 7, 7)) == 0

    assert _state(root, late)["last_news_date"] == "2026-11-09"


# ---------------------------------------------------------------------------------------------------------------
# State survives a restart; T10 first half: a crash between create_task and the claim.
# ---------------------------------------------------------------------------------------------------------------


def _restart() -> None:
    """A new process: the module and the per-process schema cache start over."""
    projects_db._INITIALIZED_PATHS.clear()
    importlib.reload(_ops())


def test_state_survives_a_restart(root, owner):
    garden = _project(root, owner, "Garden Plan", _CARDS)
    quiet = _project(root, owner, "Quiet Corner")
    _seed_reviewed(root, quiet, "2026-11-08")
    _switch_on(root, garden["project_id"], quiet["project_id"])
    reviewed_at = _at(2026, 11, 9, 7)
    assert _ops().review_tick(now=reviewed_at) == 1
    before = (_state(root, garden), _state(root, quiet))

    _restart()
    assert _ops().review_tick(now=_at(2026, 11, 9, 9)) == 0

    assert (_state(root, garden), _state(root, quiet)) == before
    assert _state(root, quiet)["last_reviewed_at"] == reviewed_at
    assert len(_review_tasks(root, archived=True)) == 1
    assert _budget(root, "2026-11-09") == (1, 0)


def test_t10_a_task_made_before_a_crash_is_adopted(root, owner, monkeypatch):
    garden = _project(root, owner, "Garden Plan", _CARDS)
    _switch_on(root, garden["project_id"])
    create_task = kanban_db.create_task

    def create_then_die(*args, **kwargs):
        create_task(*args, **kwargs)
        raise _ProcessDied

    monkeypatch.setattr(kanban_db, "create_task", create_then_die)
    with pytest.raises(_ProcessDied):
        _ops().review_tick(now=_at(2026, 11, 9, 7))
    monkeypatch.setattr(kanban_db, "create_task", create_task)
    [task] = _review_tasks(root)
    assert _state(root, garden)["open_task_id"] is None
    assert _budget(root, "2026-11-09") == (0, 0)

    _restart()
    assert _ops().review_tick(now=_at(2026, 11, 9, 7, 1)) == 0
    assert _ops().review_tick(now=_at(2026, 11, 9, 7, 2)) == 0

    assert [t.id for t in _review_tasks(root, archived=True)] == [task.id]
    state = _state(root, garden)
    assert (state["open_task_id"], state["open_review_date"], state["failures"]) == (task.id, "2026-11-09", 0)
    assert _budget(root, "2026-11-09") == (1, 0)


def test_a_task_someone_else_made_with_the_key_is_never_adopted(root, owner):
    garden = _project(root, owner, "Garden Plan", _CARDS)
    _switch_on(root, garden["project_id"])
    ops = _ops()
    ops.review_tick(now=_at(2026, 11, 9, 5))  # before the hour: nothing is made, not even the board
    assert not _board_db(root, BOARD).exists()
    kanban_db.create_board(BOARD, name="Project summaries")
    with _board(root) as conn:
        forged = kanban_db.create_task(
            conn, title=TITLE + "Garden Plan", created_by="someone-else",
            idempotency_key=f"{CREATOR}:{garden['project_id']}:2026-11-09:0",
        )

    assert ops.review_tick(now=_at(2026, 11, 9, 7)) == 0

    assert _review_tasks(root, archived=True) == []
    state = _state(root, garden)
    assert (state["open_task_id"], state["failures"]) == (None, 1)
    with _board(root) as conn:
        assert kanban_db.get_task(conn, forged).status != "archived"
    assert _budget(root, "2026-11-09") == (0, 0)


# ---------------------------------------------------------------------------------------------------------------
# Store paths: the tick reaches the root registry, the root settings and every board under a worker's pins.
# ---------------------------------------------------------------------------------------------------------------


def _enter_planning_worker(monkeypatch, root: Path, slug: str, task_id: str) -> None:
    """Take on the store pins kanban_db_dispatch._default_spawn gives a worker."""
    pins = {
        "HERMES_KANBAN_DB": str(kanban_db.kanban_db_path(board=slug)),
        "HERMES_KANBAN_WORKSPACES_ROOT": str(kanban_db.workspaces_root(board=slug)),
        "HERMES_KANBAN_BOARD": slug,
    }
    monkeypatch.setenv("HERMES_HOME", str(root / "profiles" / PLANNER))
    monkeypatch.setenv("HERMES_PROFILE", PLANNER)
    monkeypatch.setenv("HERMES_KANBAN_TASK", task_id)
    for name, value in pins.items():
        monkeypatch.setenv(name, value)


def test_tick_under_the_worker_environment_reaches_every_store(root, owner, monkeypatch):
    with _written_at(_at(2026, 11, 8, 9)):
        garden = _project(root, owner, "Garden Plan", _CARDS)
        desk = _project(root, owner, "Planning Desk", _CARDS)
        spare = _project(root, owner, "Spare Room", _CARDS)
        # Allowlisted, but not active: the owner paused the porch's board, and the shed's work is complete.
        porch = _project(root, owner, "Front Porch", _CARDS)
        kanban_db.write_board_dispatch_state(porch["board"], dispatch_paused_by_owner=True)
        shed = _project(root, owner, "Garden Shed", ["Paint the shed"])
        _complete_cards(root, shed, ["Paint the shed"])
        _switch_on(root, garden["project_id"], desk["project_id"], porch["project_id"], shed["project_id"])
        with _board(root, desk["board"]) as conn:
            worker_task = kanban_db.create_task(
                conn, title="Plan the next milestone", assignee=PLANNER, project_id=desk["project_id"],
            )
    profile_home = root / "profiles" / PLANNER
    profile_home.mkdir(parents=True)
    with projects_db.connect_closing(profile_home / "projects.db"):
        pass
    _configure(profile_home, enabled=False, allowlist=[])
    _enter_planning_worker(monkeypatch, root, desk["board"], worker_task)
    pinned = {name: os.environ[name] for name in ("HERMES_KANBAN_DB", "HERMES_KANBAN_BOARD")}
    own_board = _board_db(root, desk["board"])
    assert Path(pinned["HERMES_KANBAN_DB"]).resolve() == own_board.resolve()
    own_cards = _rows(own_board, "SELECT id, title, status FROM tasks ORDER BY id")

    made = _ops().review_tick(now=_at(2026, 11, 9, 7))

    # The independent count: the rows on the root summary board, read on a raw read-only connection.
    on_board = _rows(
        _board_db(root, BOARD),
        "SELECT title, idempotency_key FROM tasks WHERE created_by = ? ORDER BY title", (CREATOR,),
    )
    assert made == len(on_board) == 2
    assert on_board == [
        (TITLE + "Garden Plan", f"{CREATOR}:{garden['project_id']}:2026-11-09:0"),
        (TITLE + "Planning Desk", f"{CREATOR}:{desk['project_id']}:2026-11-09:0"),
    ]
    by_title = {task.title: task for task in _review_tasks(root)}
    assert by_title[TITLE + "Garden Plan"].body.rstrip("\n").splitlines()[-1] == events.covered_line(
        events.Cursors(_last_event(root, garden), 0, 0)
    )
    assert by_title[TITLE + "Planning Desk"].body.rstrip("\n").splitlines()[-1] == events.covered_line(
        events.Cursors(_last_event(root, desk), 0, 0)
    )
    assert _rows(
        root / "projects.db",
        "SELECT project_id FROM project_review_state WHERE open_task_id IS NOT NULL ORDER BY project_id",
    ) == sorted([(garden["project_id"],), (desk["project_id"],)])
    assert _state(root, spare) is None
    assert (_state(root, porch), _state(root, shed)) == (None, None)
    assert _budget(root, "2026-11-09") == (2, 0)
    meta = json.loads((_board_db(root, BOARD).parent / "board.json").read_text(encoding="utf-8"))
    assert (meta["project_id"], meta["dispatch_enabled"]) == (None, True)
    assert _rows(own_board, "SELECT id, title, status FROM tasks ORDER BY id") == own_cards
    assert _rows(profile_home / "projects.db", "SELECT count(*) FROM project_review_state") == [(0,)]
    assert _rows(profile_home / "projects.db", "SELECT count(*) FROM review_budget") == [(0,)]
    assert not _board_db(profile_home, BOARD).exists()
    assert {name: os.environ[name] for name in pinned} == pinned

    # A new card on the garden's own board, not the pinned one, moves its quiet clock while its review is open.
    _new_work_at(root, garden, _at(2026, 11, 9, 7, 6))
    assert _ops().review_tick(now=_at(2026, 11, 9, 7, 7)) == 0

    assert dict(_rows(root / "projects.db", "SELECT project_id, last_news_date FROM project_review_state")) == {
        garden["project_id"]: "2026-11-09", desk["project_id"]: "2026-11-08",
    }
    assert _rows(
        root / "projects.db",
        "SELECT project_id FROM project_review_state WHERE open_task_id IS NOT NULL ORDER BY project_id",
    ) == sorted([(garden["project_id"],), (desk["project_id"],)])
    assert len(_rows(_board_db(root, BOARD), "SELECT id FROM tasks WHERE created_by = ?", (CREATOR,))) == 2
    assert _budget(root, "2026-11-09") == (2, 0)
    assert _rows(profile_home / "projects.db", "SELECT count(*) FROM project_review_state") == [(0,)]
    assert {name: os.environ[name] for name in pinned} == pinned
