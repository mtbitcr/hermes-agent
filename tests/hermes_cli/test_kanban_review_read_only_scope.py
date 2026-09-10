"""A review run reads the implementation; it never inherits the right to write it.

Entering review used to change the assignee and the route and nothing else, so
the reviewer arrived holding the IMPLEMENTER's ``owned_paths`` — the exact
repository write boundary the owner approved for the work, now in the hands of
the profile whose whole job is to judge it. The parked card is therefore
converted to a read-only scope, and the scope it replaces is recorded on the
handover's own ``review_requested`` event so it can be given back EXACTLY:

* on the handback (changes requested),
* on an explicit reopen, and
* on the approval that ends the requirement,

and so that it survives a reviewer swap mid-review, which used to overwrite the
column outright with nothing anywhere to restore it from.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from plugins.dashboard_auth.raphael_workspace import model_policy


IMPLEMENTER = "worker"
REVIEWER = "raphael-verifier"
OTHER_REVIEWER = "raphael-planner"
IMPLEMENTATION_SCOPE = ["src/app.py", "tests/app"]


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb._INITIALIZED_PATHS.clear()
    kb.init_db()
    return home


def _events(conn, tid, kind) -> list:
    return [
        json.loads(r["payload"]) if r["payload"] else None
        for r in conn.execute(
            "SELECT payload FROM task_events WHERE task_id = ? AND kind = ? "
            "ORDER BY id",
            (tid, kind),
        ).fetchall()
    ]


def _nominate(monkeypatch, *profiles):
    monkeypatch.setattr(
        model_policy, "reviewer_profile_ids", lambda: tuple(profiles), raising=True,
    )


def _scoped_implementation(conn, *, title="scoped implementation"):
    """A claimed card with a real mutable write boundary and a review duty."""
    tid = kb.create_task(
        conn,
        title=title,
        assignee=IMPLEMENTER,
        requires_review=True,
        owned_paths=IMPLEMENTATION_SCOPE,
        workspace_kind="worktree",
    )
    assert kb.get_task(conn, tid).owned_paths == IMPLEMENTATION_SCOPE
    run = kb.claim_task(conn, tid, claimer=f"{IMPLEMENTER}:1")
    assert run is not None
    return tid, run


def _hand_over(conn, tid, run, *, reviewer=REVIEWER):
    assert kb.request_review(
        conn, tid,
        summary="ready for review",
        reviewer=reviewer,
        expected_run_id=run.current_run_id,
    ) is True


def test_a_review_run_cannot_write_the_implementers_paths(
    kanban_home, monkeypatch,
):
    """The scope enforcement helper refuses every implementation path."""
    _nominate(monkeypatch, REVIEWER)
    with kb.connect() as conn:
        tid, run = _scoped_implementation(conn)
        _hand_over(conn, tid, run)

        parked = kb.get_task(conn, tid)
        assert (parked.status, parked.assignee) == ("review", REVIEWER)
        assert parked.owned_paths == []
        assert parked.integrates_parent_heads is False

        review = kb.claim_review_task(conn, tid, claimer=f"{REVIEWER}:1")
        assert review is not None
        reviewing = kb.get_task(conn, tid)

        # The kernel's own ownership predicate — the one every write-scope
        # proof goes through — refuses each path the implementer could write.
        for path in IMPLEMENTATION_SCOPE + ["src/app.py", "tests/app/test_app.py"]:
            assert kb._path_is_owned(path, reviewing.owned_paths) is False, path
        assert kb.run_requires_read_only_workspace(
            conn, tid, int(reviewing.current_run_id),
            owned_paths=reviewing.owned_paths,
        ) is True


def test_the_implementation_scope_is_restored_on_handback_and_on_approval(
    kanban_home, monkeypatch,
):
    """Read-only for the review, and given back exactly, on both exits."""
    _nominate(monkeypatch, REVIEWER)
    with kb.connect() as conn:
        tid, run = _scoped_implementation(conn)
        _hand_over(conn, tid, run)
        assert kb.get_task(conn, tid).owned_paths == []

        review = kb.claim_review_task(conn, tid, claimer=f"{REVIEWER}:1")
        assert review is not None
        ok, implementer = kb.request_changes(
            conn, tid,
            reason="the loop bound is off by one",
            expected_run_id=review.current_run_id,
        )
        assert (ok, implementer) == (True, IMPLEMENTER)

        reworking = kb.get_task(conn, tid)
        assert reworking.assignee == IMPLEMENTER
        assert reworking.owned_paths == IMPLEMENTATION_SCOPE
        for path in IMPLEMENTATION_SCOPE:
            assert kb._path_is_owned(path, reworking.owned_paths) is True

        # Round two: the same park, and this time the reviewer approves.
        rework = kb.claim_task(conn, tid, claimer=f"{IMPLEMENTER}:2")
        assert rework is not None
        _hand_over(conn, tid, rework)
        assert kb.get_task(conn, tid).owned_paths == []

        second = kb.claim_review_task(conn, tid, claimer=f"{REVIEWER}:2")
        assert second is not None
        assert kb.complete_task(
            conn, tid,
            summary="approved",
            expected_run_id=second.current_run_id,
        ) is True

        approved = kb.get_task(conn, tid)
        assert approved.status == "done"
        assert approved.owned_paths == IMPLEMENTATION_SCOPE


def test_the_implementation_scope_survives_a_reviewer_swap(
    kanban_home, monkeypatch,
):
    """A mid-review reviewer change carries the parked scope with it."""
    _nominate(monkeypatch, OTHER_REVIEWER)
    with kb.connect() as conn:
        tid, run = _scoped_implementation(conn)
        _hand_over(conn, tid, run, reviewer=OTHER_REVIEWER)

        # The policy moves, and the card is re-routed to the reviewer it now
        # nominates — the read-only audit role, whose reassignment forces the
        # scope column empty on its own.
        _nominate(monkeypatch, REVIEWER)
        assert kb.assign_task(conn, tid, REVIEWER) is True
        swapped = kb.get_task(conn, tid)
        assert (swapped.assignee, swapped.owned_paths) == (REVIEWER, [])

        review = kb.claim_review_task(conn, tid, claimer=f"{REVIEWER}:1")
        assert review is not None
        ok, implementer = kb.request_changes(
            conn, tid, reason="please split the helper",
            expected_run_id=review.current_run_id,
        )
        assert (ok, implementer) == (True, IMPLEMENTER)

        back = kb.get_task(conn, tid)
        assert back.assignee == IMPLEMENTER
        assert back.owned_paths == IMPLEMENTATION_SCOPE


def test_an_explicit_reopen_restores_the_implementation_scope(
    kanban_home, monkeypatch,
):
    """The reopen leg gives the boundary back exactly like the handback."""
    _nominate(monkeypatch, REVIEWER)
    with kb.connect() as conn:
        tid, run = _scoped_implementation(conn)
        _hand_over(conn, tid, run)
        assert kb.get_task(conn, tid).owned_paths == []

        assert kb.reopen_review_task(conn, tid) is True
        reopened = kb.get_task(conn, tid)
        assert reopened.assignee == IMPLEMENTER
        assert reopened.owned_paths == IMPLEMENTATION_SCOPE
        assert _events(conn, tid, "review_reopened")
