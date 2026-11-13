"""The daily cap: a waiting Project with news gets the day's cap mark, and the project page explains the delay.

Owner rule of 2026-11-12: the mark is a column of the review state, not a brief row; one pass after the start loop of
the tick sets it; only the snapshot view of the brief reads it.
"""

from __future__ import annotations

import logging
import os
import sqlite3
import time
from datetime import datetime
from zoneinfo import ZoneInfo

import pytest
import yaml

from hermes_cli import kanban_db, owner_project_review as review, owner_project_summary as summary, projects_db
from hermes_cli import owner_project_review_record as records
from hermes_cli.owner_project_review_view import project_review_view
from tests.hermes_cli.test_owner_project_review_result import (  # noqa: F401
    NOW, _complete_review, _open_review, _quiet_project, _round3_drive, _round3_mutate, _round3_setup_result,
    _suggestion, root, round3_root,
)

DAY = '2026-11-09'
EARLIER_DAY = '2026-11-08'
HEADLINE = 'The review waits for the next day: the daily limit is reached.'
WHAT_CHANGED = (
    "The daily limit of reviews is reached, so this Project's review waits for the next day. "
    'Its news is kept for that review.'
)
NEXT_STEP = 'The review runs on the next day.'


def _explanation(day=DAY, covered_from=DAY, written_at=NOW):
    return {
        'review_date': day, 'covered_from': covered_from, 'covered_to': day, 'headline': HEADLINE,
        'what_changed': WHAT_CHANGED, 'next_step': NEXT_STEP, 'needs_you': None, 'ready_work': [], 'cut_note': None,
        'quiet': False, 'written_at': written_at, 'revision': 1,
    }


def _task_id(project):
    with kanban_db.connect_closing(kanban_db.kanban_db_path(board=project.board_slug), board=project.board_slug) as board:
        return board.execute('SELECT id FROM tasks LIMIT 1').fetchone()[0]


def _news(project, count=1):
    """Give the Project's board ``count`` new comments; return the id of the task they are on."""
    tid = _task_id(project)
    with kanban_db.connect_closing(kanban_db.kanban_db_path(board=project.board_slug), board=project.board_slug) as board:
        for index in range(count):
            kanban_db.add_comment(board, tid, 'default', f'The room has update {index}.')
    return tid


def _configure(root, *projects, cap=1, enabled=True):
    """The review switch and allowlist; ``cap=None`` keeps the default daily cap."""
    settings = {'enabled': enabled, 'allowlist': [project.id for project in projects]}
    if cap is not None:
        settings['daily_cap'] = cap
    (root / 'config.yaml').write_text(yaml.safe_dump({'owner_review': settings}))


def _state(root, project):
    with projects_db.connect_closing(root / 'projects.db') as registry:
        return dict(registry.execute('SELECT * FROM project_review_state WHERE project_id=?', (project.id,)).fetchone())


def _mark(root, project):
    state = _state(root, project)
    return state['cap_wait_date'], state['cap_wait_at']


def _briefs(root, project):
    with projects_db.connect_closing(root / 'projects.db') as registry:
        return [dict(row) for row in registry.execute(
            'SELECT * FROM project_review_briefs WHERE project_id=? ORDER BY review_date', (project.id,),
        )]


def _waiting(root, projects):
    """The Projects of this tick that got no review task."""
    return [project for project in projects if _state(root, project)['open_task_id'] is None]


def _capped(root, monkeypatch, names=('Busy Notes', 'Calm Notes'), counts=(3, 1), cap=1):
    """Projects with news, the first the busiest, on a day whose cap lets ``cap`` reviews start."""
    projects = [_quiet_project(root, name) for name in names]
    monkeypatch.setattr(time, 'time', lambda: NOW)
    _configure(root, *projects, cap=cap)
    for project, count in zip(projects, counts):
        if count:
            _news(project, count)
    return projects


def _rows(root, sql, *args):
    with projects_db.connect_closing(root / 'projects.db') as registry:
        return [tuple(row) for row in registry.execute(sql, args)]


def _review_task_keys():
    path = kanban_db.kanban_db_path(board='project-summaries')
    if not path.exists():
        return []
    with kanban_db.connect_closing(path, board='project-summaries') as board:
        return [row[0] for row in board.execute('SELECT idempotency_key FROM tasks ORDER BY id')]


def _everything(root):
    """Every row a tick may write, read back raw: the registry's review tables and the summary board's tasks."""
    tables = {
        name: _rows(root, f'SELECT * FROM {name} ORDER BY rowid')
        for name in ('project_review_briefs', 'project_review_state', 'review_budget', 'project_summaries')
    }
    tables['review_tasks'] = _review_task_keys()
    return tables


def _older_review(root, project):
    """An accepted review of the day before: one open item and one settled item."""
    task_id = _task_id(project)
    classification = [
        {'item': task_id, 'class': 'still actionable', 'evidence': [f'tasks:{task_id}']},
        {'item': 'earlier-card', 'class': 'answered', 'evidence': []},
    ]
    with projects_db.connect_closing(root / 'projects.db') as registry:
        projects_db.write_project_review_brief(
            registry, project.id, review_date=EARLIER_DAY, covered_from='2026-11-07', covered_to=EARLIER_DAY,
            headline='The room plan is updated.', what_changed='Book the room remains open.',
            next_step='Book the room.', classification=classification, written_at=NOW - 86400,
        )
    return classification


def _vienna(*parts):
    return int(datetime(*parts, tzinfo=ZoneInfo('Europe/Vienna')).timestamp())


def _budget(root, day=DAY):
    with projects_db.connect_closing(root / 'projects.db') as registry:
        return review.budget(registry, day)


def _oversized_history(root, project):
    """A last review with more open items than a review input holds: each review of the Project is refused."""
    items = [
        {'item': f't_open_{index:04d}_' + 'x' * 80, 'class': 'still actionable', 'evidence': [], 'event_revision': 0}
        for index in range(400)
    ]
    with projects_db.connect_closing(root / 'projects.db') as registry:
        projects_db.write_project_review_brief(
            registry, project.id, review_date=EARLIER_DAY, covered_from=None, covered_to=EARLIER_DAY,
            headline='The room needs work.', what_changed='Much work is open.', next_step='Book the room.',
            classification=items, written_at=NOW - 86400,
        )


def _reads(monkeypatch):
    """The Projects whose events the tick reads for a review (``read_review_events``), one entry per read."""
    reads = []
    real_read = review.read_review_events

    def read(project, *args, **kwargs):
        reads.append(project.id)
        return real_read(project, *args, **kwargs)

    monkeypatch.setattr(review, 'read_review_events', read)
    return reads


def _hold(root, project, sql, *args):
    """Put the Project's review state into a state that holds back its start on this day."""
    with projects_db.connect_closing(root / 'projects.db') as registry:
        with kanban_db.write_txn(registry):
            registry.execute(f'UPDATE project_review_state SET {sql} WHERE project_id = ?', (*args, project.id))


# ---------------------------------------------------------------------------------------------------------------
# Every Project the cap defers gets the mark, and only the mark
# ---------------------------------------------------------------------------------------------------------------


def test_every_project_the_cap_defers_gets_the_mark_and_no_brief_row(root, monkeypatch):
    projects = _capped(root, monkeypatch, ('Busy Notes', 'Calm Notes', 'Still Notes', 'Late Notes'), (3, 2, 1, 1))
    assert review.review_tick(NOW) == 1
    waiting = _waiting(root, projects)
    assert len(waiting) == 3 and projects[0] not in waiting
    for project in waiting:
        assert _mark(root, project) == (DAY, NOW), project.name
    assert _mark(root, projects[0]) == (None, None)
    # A cap delay is no review: no brief row and no brief-shaped record for any Project.
    assert _rows(root, 'SELECT count(*) FROM project_review_briefs') == [(0,)]
    with projects_db.connect_closing(root / 'projects.db') as registry:
        assert review.budget(registry, DAY) == (1, 0)


def test_the_view_explains_the_delay_of_a_project_without_a_stored_brief(root, monkeypatch):
    busy, waiting = _capped(root, monkeypatch)
    assert project_review_view(waiting.id) is None
    assert review.review_tick(NOW) == 1
    assert _briefs(root, waiting) == []
    assert project_review_view(waiting.id) == {'latest': _explanation(), 'history': []}
    # The Project whose review runs has no brief yet and no mark: it shows nothing, as before.
    assert project_review_view(busy.id) is None


def test_a_waiting_project_that_was_never_reviewed_gets_the_explanation_without_a_start_day(root, monkeypatch):
    projects = _capped(root, monkeypatch)
    with projects_db.connect_closing(root / 'projects.db') as registry:
        with kanban_db.write_txn(registry):
            registry.execute('DELETE FROM project_review_state')
    assert review.review_tick(NOW) == 1
    (waiting,) = _waiting(root, projects)
    assert _mark(root, waiting) == (DAY, NOW)
    assert _state(root, waiting)['last_reviewed_date'] is None
    assert project_review_view(waiting.id) == {'latest': _explanation(covered_from=None), 'history': []}


def test_the_explanation_stands_beside_the_last_real_brief_and_its_open_items(root, monkeypatch):
    busy, waiting = _capped(root, monkeypatch)
    older = _older_review(root, waiting)
    (before,) = _briefs(root, waiting)
    with projects_db.connect_closing(root / 'projects.db') as registry:
        stored_before = projects_db.get_project_review_brief(registry, waiting.id)
    assert review.review_tick(NOW) == 1
    # The last real brief, its classification and so its open items stay as they are.
    assert _briefs(root, waiting) == [before]
    with projects_db.connect_closing(root / 'projects.db') as registry:
        assert projects_db.get_project_review_brief(registry, waiting.id) == stored_before
        assert registry.execute(
            'SELECT classification FROM project_review_briefs WHERE project_id=?', (waiting.id,),
        ).fetchone()[0] == before['classification']
    assert [item['class'] for item in older] == ['still actionable', 'answered']
    served = project_review_view(waiting.id)
    assert served['latest'] == _explanation()
    assert served['history'] == [{'review_date': EARLIER_DAY, 'headline': 'The room plan is updated.'}]


def test_the_explanation_has_the_keys_of_an_ordinary_brief_and_the_fixed_texts(root, monkeypatch):
    busy, waiting, idle = _capped(root, monkeypatch, ('Busy Notes', 'Calm Notes', 'Idle Notes'), (3, 1, 0))
    assert review.review_tick(NOW) == 1
    ordinary = project_review_view(idle.id)
    assert ordinary['latest']['quiet'] is True and ordinary['latest']['review_date'] == DAY
    explained = project_review_view(waiting.id)
    assert list(explained) == list(ordinary) == ['latest', 'history']
    assert list(explained['latest']) == list(ordinary['latest']) and len(explained['latest']) == 12
    assert {name: type(value) for name, value in explained['latest'].items() if value is not None} == {
        name: type(ordinary['latest'][name]) for name, value in explained['latest'].items() if value is not None
    }
    latest = explained['latest']
    assert (latest['headline'], latest['what_changed'], latest['next_step']) == (HEADLINE, WHAT_CHANGED, NEXT_STEP)
    assert (latest['quiet'], latest['ready_work'], latest['cut_note'], latest['needs_you']) == (False, [], None, None)
    assert (latest['review_date'], latest['written_at'], latest['revision']) == (DAY, NOW, 1)


# ---------------------------------------------------------------------------------------------------------------
# The mark spends no slot and moves nothing else; a day sets it once; the next day reviews as usual
# ---------------------------------------------------------------------------------------------------------------


def test_the_mark_spends_no_slot_and_moves_no_reading_position_reviewed_day_or_open_item(root, monkeypatch):
    busy, waiting = _capped(root, monkeypatch)
    _older_review(root, waiting)
    state_before = _state(root, waiting)
    briefs_before = _briefs(root, waiting)
    summary_before = _rows(root, 'SELECT * FROM project_summaries WHERE project_id=?', waiting.id)
    assert len(summary_before) == 1
    assert review.review_tick(NOW) == 1
    state_after = _state(root, waiting)
    for name in ('event_cursor', 'delivery_cursor', 'release_cursor', 'conversation_cursor', 'last_reviewed_date',
                 'last_reviewed_at', 'last_brief_date', 'failures', 'open_task_id', 'open_review_date', 'updated_at',
                 'input_too_large_date', 'stale_refused_key'):
        assert state_after[name] == state_before[name], name
    # Only the quiet clock of the tick's news observation and the mark itself move.
    assert {name for name in state_after if state_after[name] != state_before[name]} <= {
        'last_news_date', 'cap_wait_date', 'cap_wait_at',
    }
    assert (state_after['cap_wait_date'], state_after['cap_wait_at']) == (DAY, NOW)
    assert _briefs(root, waiting) == briefs_before
    assert _rows(root, 'SELECT * FROM project_summaries WHERE project_id=?', waiting.id) == summary_before
    with projects_db.connect_closing(root / 'projects.db') as registry:
        assert review.budget(registry, DAY) == (1, 0)
    (key,) = _review_task_keys()
    assert busy.id in key and waiting.id not in key


def test_a_repeated_tick_on_the_same_day_sets_nothing_new(root, monkeypatch):
    busy, waiting = _capped(root, monkeypatch)
    _older_review(root, waiting)
    assert review.review_tick(NOW) == 1
    first = _everything(root)
    view = project_review_view(waiting.id)
    for later in (NOW + 60, NOW + 3600):
        monkeypatch.setattr(time, 'time', lambda: later)
        assert review.review_tick(later) == 0
    assert _everything(root) == first
    assert _mark(root, waiting) == (DAY, NOW)
    assert project_review_view(waiting.id) == view


def test_the_next_day_reviews_the_waiting_project_as_usual_and_the_view_shows_the_new_brief(root, monkeypatch):
    busy, waiting = _capped(root, monkeypatch)
    assert review.review_tick(NOW) == 1
    next_day = NOW + 86400
    monkeypatch.setattr(time, 'time', lambda: next_day)
    assert review.review_tick(next_day) == 1
    task = _open_review(root, waiting)
    assert task is not None and _state(root, waiting)['last_reviewed_date'] == EARLIER_DAY
    with projects_db.connect_closing(root / 'projects.db') as registry:
        assert review.budget(registry, '2026-11-10') == (1, 0)
    # The review runs: until its brief is stored, the page still explains the day the cap deferred it.
    assert project_review_view(waiting.id)['latest'] == _explanation()
    _complete_review(task, [])
    assert review.review_tick(next_day + 1) == 0
    (brief,) = _briefs(root, waiting)
    assert (brief['review_date'], brief['quiet'], brief['headline']) == ('2026-11-10', 0, 'The room plan is updated.')
    assert _state(root, waiting)['last_reviewed_date'] == '2026-11-10'
    assert _state(root, waiting)['open_task_id'] is None
    # The stored brief is of a later day than the mark: the view shows it as before, and the mark no longer shows.
    served = project_review_view(waiting.id)
    assert served['latest']['review_date'] == '2026-11-10'
    assert served['latest']['headline'] == 'The room plan is updated.' and served['latest']['quiet'] is False
    assert served['history'] == [{'review_date': '2026-11-10', 'headline': 'The room plan is updated.'}]


def test_a_brief_stored_on_the_day_of_the_mark_is_shown_as_before(root, monkeypatch):
    busy, waiting = _capped(root, monkeypatch)
    assert review.review_tick(NOW) == 1
    with projects_db.connect_closing(root / 'projects.db') as registry:
        projects_db.write_project_review_brief(
            registry, waiting.id, review_date=DAY, covered_from=DAY, covered_to=DAY, headline='The room is booked.',
            what_changed='The room booking is complete.', next_step='Send the invitations.', written_at=NOW + 5,
        )
    assert _mark(root, waiting) == (DAY, NOW)
    served = project_review_view(waiting.id)
    assert served['latest']['headline'] == 'The room is booked.' and served['latest']['written_at'] == NOW + 5
    assert served['history'] == [{'review_date': DAY, 'headline': 'The room is booked.'}]


# ---------------------------------------------------------------------------------------------------------------
# The review's probes of round two (owner rule point 8)
# ---------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize('dispatcher', [False, True])
@pytest.mark.parametrize('probe', ['cap_equals_tick_limit', 'seventeen_earlier_starts', 'last_tick_before_midnight'])
def test_round_two_probes_mark_each_deferred_project_and_the_view_explains_that_day(
    round3_root, monkeypatch, probe, dispatcher,
):
    root = round3_root
    names = ('Busy Notes', 'Calm Notes', 'Still Notes', 'Late Notes')
    instant = NOW
    if probe == 'cap_equals_tick_limit':
        assert summary.MAX_NEW_PER_TICK == 3
        projects = _capped(root, monkeypatch, names, (4, 3, 2, 1), cap=summary.MAX_NEW_PER_TICK)
        started = 3
    elif probe == 'seventeen_earlier_starts':
        projects = _capped(root, monkeypatch, names, (4, 3, 2, 1), cap=None)
        with projects_db.connect_closing(root / 'projects.db') as registry:
            for _ in range(17):
                assert review._spend(registry, DAY, 'project_reviews', None, NOW - 3600)
        assert review.read_settings().daily_cap == 20
        started = 3
    else:
        projects = _capped(root, monkeypatch, names[:2], (3, 1), cap=1)
        instant = _vienna(2026, 11, 9, 23, 59, 50)
        started = 1
    monkeypatch.setattr(time, 'time', lambda: instant)
    if dispatcher:
        _round3_drive(monkeypatch, instant)
    else:
        assert review.review_tick(instant) == started
    with projects_db.connect_closing(root / 'projects.db') as registry:
        assert review.budget(registry, DAY)[0] == review.read_settings().daily_cap
    waiting = _waiting(root, projects)
    assert len(waiting) == len(projects) - started == 1
    for project in projects:
        assert _mark(root, project) == ((DAY, instant) if project in waiting else (None, None)), project.name
        assert _briefs(root, project) == []
    for project in waiting:
        assert project_review_view(project.id) == {'latest': _explanation(written_at=instant), 'history': []}
    if probe == 'last_tick_before_midnight':
        # The next tick is on the next date, before its review hour: the mark of November 9 stays and still shows.
        after_midnight = _vienna(2026, 11, 10, 0, 0, 10)
        monkeypatch.setattr(time, 'time', lambda: after_midnight)
        assert review.review_tick(after_midnight) == 0
        (project,) = waiting
        assert _mark(root, project) == (DAY, instant)
        assert project_review_view(project.id)['latest'] == _explanation(written_at=instant)


# ---------------------------------------------------------------------------------------------------------------
# Controls (owner rule point 9)
# ---------------------------------------------------------------------------------------------------------------


def test_below_the_daily_cap_the_limit_of_one_tick_sets_no_mark(root, monkeypatch):
    names = ('Busy Notes', 'Calm Notes', 'Still Notes', 'Late Notes')
    projects = _capped(root, monkeypatch, names, (4, 3, 2, 1), cap=None)
    assert review.review_tick(NOW) == summary.MAX_NEW_PER_TICK
    (left,) = _waiting(root, projects)
    assert [_mark(root, project) for project in projects] == [(None, None)] * 4
    assert project_review_view(left.id) is None
    with projects_db.connect_closing(root / 'projects.db') as registry:
        assert review.budget(registry, DAY) == (3, 0)
    # The Project the tick limit left over starts on the next tick of the same day.
    assert review.review_tick(NOW + 60) == 1
    assert _waiting(root, projects) == [] and _mark(root, left) == (None, None)


def test_a_project_left_over_by_a_smaller_tick_limit_gets_no_mark(root, monkeypatch):
    projects = _capped(root, monkeypatch, cap=5)
    monkeypatch.setattr(summary, 'MAX_NEW_PER_TICK', 1)
    assert review.review_tick(NOW) == 1
    assert [_mark(root, project) for project in projects] == [(None, None)] * 2
    assert len(_review_task_keys()) == 1


def test_below_the_cap_nothing_is_marked(root, monkeypatch):
    projects = _capped(root, monkeypatch, cap=2)
    assert review.review_tick(NOW) == 2
    assert review.review_tick(NOW + 60) == 0
    assert [_mark(root, project) for project in projects] == [(None, None)] * 2
    assert all(_state(root, project)['open_task_id'] is not None for project in projects)


def test_a_project_without_news_gets_no_mark(root, monkeypatch):
    busy, calm, idle = _capped(root, monkeypatch, ('Busy Notes', 'Calm Notes', 'Idle Notes'), (3, 0, 0))
    assert review.review_tick(NOW) == 1
    for project in (calm, idle):
        assert _mark(root, project) == (None, None)
        (row,) = _briefs(root, project)
        assert (row['quiet'], row['headline']) == (1, 'Nothing changed since the last review.')
        assert project_review_view(project.id)['latest']['quiet'] is True


def test_with_the_daily_review_switch_off_nothing_changes(root, monkeypatch):
    projects = _capped(root, monkeypatch)
    _configure(root, *projects, enabled=False)
    before = _everything(root)
    assert review.review_tick(NOW) == 0
    assert review.review_tick(NOW + 60) == 0
    assert _everything(root) == before
    assert [_mark(root, project) for project in projects] == [(None, None)] * 2
    assert [project_review_view(project.id) for project in projects] == [None, None]


@pytest.mark.parametrize('change', ['review_opened', 'reviewed_today'])
def test_a_state_that_moved_after_the_tick_read_it_gets_no_mark(root, monkeypatch, change):
    busy, moved, fine = _capped(root, monkeypatch, ('Busy Notes', 'Calm Notes', 'Still Notes'), (3, 2, 1))
    updates = {
        'review_opened': ('UPDATE project_review_state SET open_task_id=? WHERE project_id=?', 't_concurrent'),
        'reviewed_today': ('UPDATE project_review_state SET last_reviewed_date=? WHERE project_id=?', DAY),
    }
    sql, value = updates[change]
    real_ensure = summary._ensure_summary_board

    def ensure_after_a_concurrent_change():
        with projects_db.connect_closing(root / 'projects.db') as registry:
            with kanban_db.write_txn(registry):
                registry.execute(sql, (value, moved.id))
        return real_ensure()

    monkeypatch.setattr(summary, '_ensure_summary_board', ensure_after_a_concurrent_change)
    assert review.review_tick(NOW) == 1
    assert _mark(root, moved) == (None, None)
    assert _mark(root, fine) == (DAY, NOW)
    stored = _state(root, moved)
    assert (stored['open_task_id'], stored['last_reviewed_date']) == (
        ('t_concurrent', EARLIER_DAY) if change == 'review_opened' else (None, DAY)
    )


def test_a_project_whose_input_did_not_fit_keeps_its_outcome_and_gets_the_mark_once_the_cap_is_reached(
    root, monkeypatch,
):
    large, opened = _capped(root, monkeypatch, cap=1)
    real_build = records.build_review_input

    def build(project, *args, **kwargs):
        if project.id == large.id:
            raise records.ReviewInputTooLarge('the open items do not fit')
        return real_build(project, *args, **kwargs)

    monkeypatch.setattr(records, 'build_review_input', build)
    assert review.review_tick(NOW) == 1
    # The busiest Project's input did not fit: no review and no slot; the next Project takes the day's one slot.
    assert _state(root, large)['input_too_large_date'] == DAY and _state(root, large)['open_task_id'] is None
    assert _state(root, opened)['open_task_id'] is not None
    with projects_db.connect_closing(root / 'projects.db') as registry:
        assert review.budget(registry, DAY) == (1, 0)
    # Owner rule point 3: the cap is reached, and it has news, no review task and no review on this day.
    assert _mark(root, large) == (DAY, NOW) and _mark(root, opened) == (None, None)
    assert _state(root, large)['failures'] == 0


def test_a_refusing_store_leaves_the_mark_unset_the_tick_counts_and_the_next_tick_sets_it(root, monkeypatch, caplog):
    busy, waiting = _capped(root, monkeypatch)
    with projects_db.connect_closing(root / 'projects.db') as registry:
        registry.execute(
            "CREATE TRIGGER refuse_mark BEFORE UPDATE OF cap_wait_date ON project_review_state "
            "BEGIN SELECT RAISE(ABORT, 'database is locked'); END"
        )
        registry.commit()
    caplog.set_level(logging.WARNING, logger='hermes_cli.owner_project_review')
    assert review.review_tick(NOW) == 1
    assert _mark(root, waiting) == (None, None)
    assert _state(root, waiting)['failures'] == 0 and _state(root, waiting)['last_reviewed_date'] == EARLIER_DAY
    assert any('cap_mark_unavailable' in record.getMessage() for record in caplog.records)
    with projects_db.connect_closing(root / 'projects.db') as registry:
        registry.execute('DROP TRIGGER refuse_mark')
        registry.commit()
    assert review.review_tick(NOW + 60) == 0
    assert _mark(root, waiting) == (DAY, NOW + 60)


def test_the_portfolio_input_does_not_read_a_cap_delay_as_a_review(root, monkeypatch):
    from hermes_cli import owner_portfolio_review as portfolio

    busy, waiting = _capped(root, monkeypatch)
    _older_review(root, waiting)
    assert review.review_tick(NOW) == 1
    assert _mark(root, waiting) == (DAY, NOW)
    with projects_db.connect_closing(root / 'projects.db') as registry:
        frozen = portfolio._input(registry, [busy, waiting], DAY)
    # The portfolio reads stored briefs only, as before: the waiting Project is not reviewed today.
    record = frozen['projects'][1]
    assert record['name'] == waiting.name and record['review_state'] == 'not reviewed today'
    assert record['brief']['headline'] == 'The room plan is updated.'


# ---------------------------------------------------------------------------------------------------------------
# A Project the start loop holds back gets the mark too: its input did not fit, a failed review backs off, or the
# day's one further review is used
# ---------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize('earlier_tick', [True, False])
def test_a_project_whose_input_did_not_fit_gets_the_mark_whether_refused_in_an_earlier_tick_or_in_this_one(
    round3_root, monkeypatch, earlier_tick,
):
    root = round3_root
    monkeypatch.setattr(time, 'time', lambda: NOW)
    large = _quiet_project(root, 'Large Notes')
    _news(large, 4)
    _oversized_history(root, large)
    if earlier_tick:
        # An earlier tick of the day, below the cap: the required input does not fit, no slot is spent, no mark is set.
        _configure(root, large)
        _round3_drive(monkeypatch, NOW)
        assert (_state(root, large)['input_too_large_date'], _mark(root, large)) == (DAY, (None, None))
        assert _budget(root) == (0, 0)
    small = _quiet_project(root, 'Small Notes')
    _news(small)
    _configure(root, large, small)
    before = _state(root, large)
    briefs = _briefs(root, large)
    _round3_drive(monkeypatch, NOW + 60)
    assert _budget(root) == (1, 0)
    assert _state(root, small)['open_task_id'] is not None and _mark(root, small) == (None, None)
    after = _state(root, large)
    assert (after['cap_wait_date'], after['cap_wait_at']) == (DAY, NOW + 60)
    # The refusal keeps its outcome: no review task, no failure, no reviewed day, and the stored brief as it was.
    assert (after['input_too_large_date'], after['open_task_id'], after['failures']) == (DAY, None, 0)
    assert after['last_reviewed_date'] == EARLIER_DAY and _briefs(root, large) == briefs
    # Refused in this tick, the refusal sets its day and clock now; refused earlier, the mark moves nothing but itself.
    refused_now = set() if earlier_tick else {'input_too_large_date', 'updated_at'}
    assert {name for name in after if after[name] != before[name]} <= {
        'cap_wait_date', 'cap_wait_at', 'last_news_date', *refused_now,
    }
    assert after['updated_at'] == (NOW if earlier_tick else NOW + 60)
    assert project_review_view(large.id) == {
        'latest': _explanation(written_at=NOW + 60),
        'history': [{'review_date': EARLIER_DAY, 'headline': 'The room needs work.'}],
    }


@pytest.mark.parametrize('same_tick', [True, False])
def test_a_failed_review_in_its_backoff_gets_the_mark_when_another_project_fills_the_cap(
    round3_root, monkeypatch, same_tick,
):
    root = round3_root
    monkeypatch.setattr(time, 'time', lambda: NOW)
    failed = _quiet_project(root, 'Failed Notes')
    _news(failed)
    _configure(root, failed, cap=2)
    _round3_drive(monkeypatch, NOW)
    task = _open_review(root, failed)
    assert task is not None and _budget(root) == (1, 0)
    with kanban_db.connect_closing(kanban_db.kanban_db_path(board='project-summaries'), board='project-summaries') as board:
        assert kanban_db.block_task(board, task, reason='Review input is not available.', kind='transient')
    harvested_at = NOW + 120 if same_tick else NOW + 60
    if not same_tick:
        # A tick below the cap harvests the failure: the retry backs off, and below the cap nothing is marked.
        _round3_drive(monkeypatch, harvested_at)
        assert (_state(root, failed)['failures'], _state(root, failed)['open_task_id']) == (1, None)
        assert _mark(root, failed) == (None, None)
    small = _quiet_project(root, 'Small Notes')
    _news(small)
    _configure(root, failed, small, cap=2)
    before = _state(root, failed)
    _round3_drive(monkeypatch, NOW + 120)
    assert _budget(root) == (2, 0) and _state(root, small)['open_task_id'] is not None
    after = _state(root, failed)
    assert (after['cap_wait_date'], after['cap_wait_at']) == (DAY, NOW + 120)
    # The backoff stays as the failure set it: one failure and its clock; no cursor, reviewed day or brief moved.
    assert (after['failures'], after['updated_at'], after['open_task_id']) == (1, harvested_at, None)
    for name in ('event_cursor', 'delivery_cursor', 'release_cursor', 'conversation_cursor', 'last_reviewed_date',
                 'last_reviewed_at', 'last_brief_date', 'input_too_large_date', 'stale_refused_key'):
        assert after[name] == before[name], name
    assert after['last_reviewed_date'] == EARLIER_DAY and _briefs(root, failed) == []
    assert project_review_view(failed.id) == {'latest': _explanation(written_at=NOW + 120), 'history': []}
    # The next day the backoff has passed: the review starts as usual, and its brief takes the explanation's place.
    next_day = NOW + 86400
    monkeypatch.setattr(time, 'time', lambda: next_day)
    _round3_drive(monkeypatch, next_day)
    retry = _open_review(root, failed)
    assert retry is not None and retry != task and _budget(root, '2026-11-10') == (1, 0)
    assert project_review_view(failed.id)['latest'] == _explanation(written_at=NOW + 120)
    _complete_review(retry, [])
    _round3_drive(monkeypatch, next_day + 60)
    assert (_state(root, failed)['last_reviewed_date'], _state(root, failed)['failures']) == ('2026-11-10', 0)
    served = project_review_view(failed.id)
    assert (served['latest']['review_date'], served['latest']['headline']) == ('2026-11-10', 'The room plan is updated.')
    assert served['history'] == [{'review_date': '2026-11-10', 'headline': 'The room plan is updated.'}]


@pytest.mark.parametrize('same_tick', [True, False])
def test_a_project_whose_further_review_of_the_day_is_used_gets_the_mark_when_another_takes_the_last_slot(
    round3_root, monkeypatch, same_tick,
):
    root = round3_root
    project, card, first = _round3_setup_result(root, monkeypatch)
    _configure(root, project, cap=3)
    _round3_mutate(project, card, 'new_wait')
    _round3_drive(monkeypatch, NOW + 1)
    further = _open_review(root, project)
    assert further is not None and further != first and _budget(root) == (2, 0)
    with kanban_db.connect_closing(kanban_db.kanban_db_path(board='project-summaries'), board='project-summaries') as board:
        further_key = kanban_db.get_task(board, further).idempotency_key
    assert further_key.endswith(':again')
    _complete_review(further, [])
    with kanban_db.connect_closing(kanban_db.kanban_db_path(board=project.board_slug), board=project.board_slug) as board:
        _suggestion(board, project, subject='late-catering', label='Choose a late caterer')
    if not same_tick:
        # A tick below the cap refuses the further review as stale too: the day's one further review is used.
        _round3_drive(monkeypatch, NOW + 2)
        assert (_state(root, project)['stale_refused_key'], _mark(root, project)) == (further_key, (None, None))
    small = _quiet_project(root, 'Small Notes')
    _news(small)
    _configure(root, project, small, cap=3)
    before = _state(root, project)
    _round3_drive(monkeypatch, NOW + 3)
    assert _budget(root) == (3, 0) and _state(root, small)['open_task_id'] is not None
    after = _state(root, project)
    assert (after['cap_wait_date'], after['cap_wait_at']) == (DAY, NOW + 3)
    # The used further review stays recorded, and the Project is neither reviewed nor failed on this day.
    assert (after['stale_refused_key'], after['open_task_id'], after['failures']) == (further_key, None, 0)
    assert after['last_reviewed_date'] == EARLIER_DAY and _briefs(root, project) == []
    if not same_tick:
        assert {name for name in after if after[name] != before[name]} <= {
            'cap_wait_date', 'cap_wait_at', 'last_news_date',
        }
    assert project_review_view(project.id) == {'latest': _explanation(written_at=NOW + 3), 'history': []}


@pytest.mark.parametrize('held', ['input_too_large', 'retry_backoff', 'further_review_used'])
def test_a_held_back_project_without_news_gets_no_mark(root, monkeypatch, held):
    busy, still = _capped(root, monkeypatch, ('Busy Notes', 'Still Notes'), (3, 0))
    holds = {
        'input_too_large': ('input_too_large_date = ?', DAY),
        'retry_backoff': ('failures = 1, updated_at = ?', NOW - 60),
        'further_review_used': ('stale_refused_key = ?', f'{review._key_prefix(still.id)}{DAY}:0:again'),
    }
    _hold(root, still, *holds[held])
    before = _state(root, still)
    reads = _reads(monkeypatch)
    assert review.review_tick(NOW) == 1
    assert _budget(root) == (1, 0)
    # The cap is reached and the held-back Project's news is read: it has none, so it gets no mark and nothing moves.
    assert reads.count(still.id) == 1
    assert _state(root, still) == before
    assert _briefs(root, still) == [] and project_review_view(still.id) is None


def test_held_back_news_is_read_only_once_the_cap_is_reached_and_a_repeated_tick_writes_nothing_new(
    root, monkeypatch,
):
    monkeypatch.setattr(time, 'time', lambda: NOW)
    large, first, second = (_quiet_project(root, name) for name in ('Large Notes', 'First Notes', 'Second Notes'))
    _news(large, 4)
    _oversized_history(root, large)
    _configure(root, large, cap=2)
    assert review.review_tick(NOW) == 0
    assert _state(root, large)['input_too_large_date'] == DAY
    reads = _reads(monkeypatch)
    # Below the cap the held-back Project's news is not read, and it gets no mark.
    _news(first)
    _configure(root, large, first, cap=2)
    assert review.review_tick(NOW + 60) == 1
    assert _budget(root) == (1, 0) and _mark(root, large) == (None, None)
    assert large.id not in reads
    # The tick whose start reaches the cap reads it once and marks it.
    _news(second)
    _configure(root, large, first, second, cap=2)
    assert review.review_tick(NOW + 120) == 1
    assert _budget(root) == (2, 0) and reads.count(large.id) == 1
    assert _mark(root, large) == (DAY, NOW + 120)
    # A repeated tick of the day reads the marked Project no more and writes nothing.
    first_marked = _everything(root)
    for later in (NOW + 180, NOW + 3600):
        monkeypatch.setattr(time, 'time', lambda: later)
        assert review.review_tick(later) == 0
    assert _everything(root) == first_marked
    assert reads.count(large.id) == 1 and _mark(root, large) == (DAY, NOW + 120)


def test_a_held_back_project_whose_news_cannot_be_read_stays_unmarked_and_the_others_are_marked(
    root, monkeypatch, caplog,
):
    busy, broken, fine = _capped(root, monkeypatch, ('Busy Notes', 'Broken Notes', 'Fine Notes'), (3, 2, 1))
    for project in (broken, fine):
        _hold(root, project, 'input_too_large_date = ?', DAY)
    real_read = review.read_review_events

    def read(project, *args, **kwargs):
        if project.id == broken.id:
            raise sqlite3.OperationalError('database is locked')
        return real_read(project, *args, **kwargs)

    monkeypatch.setattr(review, 'read_review_events', read)
    caplog.set_level(logging.WARNING, logger='hermes_cli.owner_project_review')
    before = _state(root, broken)
    assert review.review_tick(NOW) == 1
    assert _mark(root, fine) == (DAY, NOW) and _mark(root, busy) == (None, None)
    # The unreadable Project counts no failure and stays as it was, unmarked; the tick still marks the others.
    after = _state(root, broken)
    assert {name for name in after if after[name] != before[name]} <= {'last_news_date'}
    assert (after['cap_wait_date'], after['failures']) == (None, 0)
    assert any(
        'cap_mark_news_unavailable' in record.getMessage() and broken.id in record.getMessage()
        for record in caplog.records
    )
    # A later tick of the day reads it and marks it.
    monkeypatch.setattr(review, 'read_review_events', real_read)
    assert review.review_tick(NOW + 60) == 0
    assert _mark(root, broken) == (DAY, NOW + 60) and _mark(root, fine) == (DAY, NOW)
    assert _state(root, broken)['failures'] == 0


def test_a_refusing_store_leaves_a_held_back_project_unmarked_and_a_later_tick_marks_it(root, monkeypatch, caplog):
    busy, held = _capped(root, monkeypatch)
    _hold(root, held, 'failures = 1, updated_at = ?', NOW - 60)
    with projects_db.connect_closing(root / 'projects.db') as registry:
        registry.execute(
            "CREATE TRIGGER refuse_mark BEFORE UPDATE OF cap_wait_date ON project_review_state "
            "BEGIN SELECT RAISE(ABORT, 'database is locked'); END"
        )
        registry.commit()
    caplog.set_level(logging.WARNING, logger='hermes_cli.owner_project_review')
    assert review.review_tick(NOW) == 1
    assert _mark(root, held) == (None, None)
    assert any('cap_mark_unavailable' in record.getMessage() for record in caplog.records)
    with projects_db.connect_closing(root / 'projects.db') as registry:
        registry.execute('DROP TRIGGER refuse_mark')
        registry.commit()
    assert review.review_tick(NOW + 60) == 0
    assert _mark(root, held) == (DAY, NOW + 60)
    # The retry's backoff runs from the failure, as before: the mark moved neither its count nor its clock.
    assert (_state(root, held)['failures'], _state(root, held)['updated_at']) == (1, NOW - 60)


# ---------------------------------------------------------------------------------------------------------------
# The real dispatcher loop and a worker-pinned environment
# ---------------------------------------------------------------------------------------------------------------


def _enter_worker(monkeypatch, root, project):
    """The environment the dispatcher gives a worker: its own board pinned, HERMES_HOME a profile home under the root."""
    profile = root / 'profiles' / 'raphael-planner'
    profile.mkdir(parents=True)
    pins = {
        'HERMES_HOME': str(profile), 'HERMES_PROFILE': 'raphael-planner', 'HERMES_KANBAN_BOARD': project.board_slug,
        'HERMES_KANBAN_DB': str(root / 'kanban' / 'boards' / project.board_slug / 'kanban.db'),
        'HERMES_KANBAN_TASK': _task_id(project),
    }
    for name, value in pins.items():
        monkeypatch.setenv(name, value)
    return profile, pins


@pytest.mark.parametrize('pinned', [False, True])
def test_the_dispatcher_loop_sets_the_mark_once(round3_root, monkeypatch, pinned):
    root = round3_root
    busy, waiting = _capped(root, monkeypatch)
    if pinned:
        _enter_worker(monkeypatch, root, waiting)
    _round3_drive(monkeypatch, NOW)
    first = _everything(root)
    assert _mark(root, waiting) == (DAY, NOW)
    _round3_drive(monkeypatch, NOW + 60)
    assert _everything(root) == first
    assert _rows(root, 'SELECT count(*) FROM project_review_briefs') == [(0,)]
    assert _rows(root, 'SELECT project_id FROM project_review_state WHERE open_task_id IS NOT NULL') == [(busy.id,)]
    assert _rows(root, 'SELECT project_reviews, portfolio_reviews FROM review_budget') == [(1, 0)]
    assert project_review_view(waiting.id)['latest'] == _explanation()


def test_a_worker_pinned_tick_sets_the_mark_in_the_root_registry(root, monkeypatch):
    projects = _capped(root, monkeypatch)
    busy, waiting = projects
    boards = {project.id: root / 'kanban' / 'boards' / project.board_slug / 'kanban.db' for project in projects}

    def read(path, sql, *args):
        # The independent count: plain read-only SQLite on the files, without the code under test.
        conn = sqlite3.connect(f'{path.as_uri()}?mode=ro', uri=True)
        try:
            return conn.execute(sql, args).fetchall()
        finally:
            conn.close()

    own_cards = read(boards[waiting.id], 'SELECT id, title, status FROM tasks ORDER BY id')
    own_events = read(boards[waiting.id], 'SELECT count(*) FROM task_events')
    profile, pins = _enter_worker(monkeypatch, root, waiting)

    made = review.review_tick(NOW)
    assert review.review_tick(NOW + 60) == 0

    assert made == 1
    marks = read(root / 'projects.db', 'SELECT project_id, cap_wait_date, cap_wait_at FROM project_review_state '
                                       'WHERE cap_wait_date IS NOT NULL')
    assert marks == [(waiting.id, DAY, NOW)]
    # The independent count: the Projects that did not get a review task are the Projects that got the mark.
    assert len(projects) - made == len(marks)
    assert read(root / 'projects.db', 'SELECT count(*) FROM project_review_briefs') == [(0,)]
    assert read(root / 'projects.db', 'SELECT count(*) FROM review_budget') == [(1,)]
    assert read(boards[waiting.id], 'SELECT id, title, status FROM tasks ORDER BY id') == own_cards
    assert read(boards[waiting.id], 'SELECT count(*) FROM task_events') == own_events
    assert not (profile / 'projects.db').exists()
    assert not (profile / 'kanban').exists()
    assert {name: os.environ[name] for name in pins} == pins


def test_a_worker_pinned_tick_marks_a_project_an_earlier_input_refusal_holds_back_in_the_root_registry(
    root, monkeypatch,
):
    monkeypatch.setattr(time, 'time', lambda: NOW)
    large, small, pinned = (_quiet_project(root, name) for name in ('Large Notes', 'Small Notes', 'Pinned Notes'))
    _news(large, 4)
    _news(small)
    _oversized_history(root, large)
    boards = {
        project.id: root / 'kanban' / 'boards' / project.board_slug / 'kanban.db' for project in (large, small, pinned)
    }
    registry = root / 'projects.db'

    def read(path, sql, *args):
        # The independent count: plain read-only SQLite on the files, without the code under test.
        conn = sqlite3.connect(f'{path.as_uri()}?mode=ro', uri=True)
        try:
            return conn.execute(sql, args).fetchall()
        finally:
            conn.close()

    def board_rows():
        return {
            project_id: (read(path, 'SELECT id, title, status FROM tasks ORDER BY id'),
                         read(path, 'SELECT count(*) FROM task_events'))
            for project_id, path in boards.items()
        }

    own_boards = board_rows()
    # The worker's environment pins its own board, a third Project's that holds no news and is not reviewed.
    profile, pins = _enter_worker(monkeypatch, root, pinned)
    _configure(root, large)
    assert review.review_tick(NOW) == 0
    assert read(registry, 'SELECT input_too_large_date, cap_wait_date FROM project_review_state WHERE project_id = ?',
                large.id) == [(DAY, None)]
    # A later tick of the day in that environment: the other Project takes the day's one slot.
    _configure(root, large, small)
    made = review.review_tick(NOW + 60)
    assert review.review_tick(NOW + 120) == 0

    assert made == 1
    marks = read(registry, 'SELECT project_id, cap_wait_date, cap_wait_at FROM project_review_state '
                           'WHERE cap_wait_date IS NOT NULL')
    assert marks == [(large.id, DAY, NOW + 60)]
    # The independent count: with the day's cap reached, the allowed Projects with comments past their stored cursor
    # on their own board, without a review task and without a review on this day.
    assert read(registry, 'SELECT project_reviews FROM review_budget WHERE review_date = ?', DAY) == [(1,)]
    waiting = []
    for project in (large, small):
        ((cursor, open_task, reviewed),) = read(
            registry, 'SELECT event_cursor, open_task_id, last_reviewed_date FROM project_review_state '
                      'WHERE project_id = ?', project.id,
        )
        ((comments,),) = read(boards[project.id], "SELECT count(*) FROM task_events WHERE id > ? AND kind = 'commented'",
                              cursor or 0)
        if comments and open_task is None and (reviewed or '') < DAY:
            waiting.append(project.id)
    assert waiting == [large.id] and len(marks) == len(waiting)
    # The news is the Project's own: the pinned board holds no event past the large Project's cursor.
    ((large_cursor,),) = read(registry, 'SELECT event_cursor FROM project_review_state WHERE project_id = ?', large.id)
    assert read(boards[pinned.id], 'SELECT count(*) FROM task_events WHERE id > ?', large_cursor) == [(0,)]
    assert read(registry, 'SELECT input_too_large_date, failures, last_reviewed_date FROM project_review_state '
                          'WHERE project_id = ?', large.id) == [(DAY, 0, EARLIER_DAY)]
    assert read(registry, 'SELECT count(*) FROM project_review_briefs') == [(1,)]
    assert board_rows() == own_boards
    assert not (profile / 'projects.db').exists()
    assert not (profile / 'kanban').exists()
    assert {name: os.environ[name] for name in pins} == pins
