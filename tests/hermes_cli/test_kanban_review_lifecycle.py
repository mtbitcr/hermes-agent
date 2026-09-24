"""Review-lifecycle tests: the first-class ``running -> review`` transition.

``request_review`` is the "implementation complete, awaiting review"
transition used by executor workers instead of encoding ``review-required:``
prose into a ``kanban_block`` call. The critical contract these tests pin
down:

* It transitions ``running``/``ready`` -> ``review`` and closes the active
  run with ``outcome="review_requested"``.
* It emits exactly one ``review_requested`` event carrying the handoff
  summary + implementer.
* Crucially, it is NOT a blocker: repeated review requests on the same task
  (a review -> rerun -> review follow-up cycle) never touch
  ``block_recurrences`` and never route to ``triage`` — the false
  ``block_loop_detected`` escalation that plagued the block-reason approach
  cannot happen.
* ``expected_run_id`` is honoured as a CAS guard so a stale/superseded
  worker cannot move the task.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_dispatch as kbd


@pytest.fixture
def kanban_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Isolated HERMES_HOME with an empty kanban DB."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _row(conn, tid):
    return conn.execute(
        "SELECT status, block_kind, block_recurrences, current_run_id "
        "FROM tasks WHERE id = ?",
        (tid,),
    ).fetchone()


def _events(conn, tid, kind=None):
    rows = conn.execute(
        "SELECT kind, payload FROM task_events WHERE task_id = ? ORDER BY id",
        (tid,),
    ).fetchall()
    out = [
        (r["kind"], json.loads(r["payload"]) if r["payload"] else None)
        for r in rows
    ]
    if kind is not None:
        out = [e for e in out if e[0] == kind]
    return out


def _last_run(conn, tid):
    return conn.execute(
        "SELECT status, outcome, summary FROM task_runs "
        "WHERE task_id = ? ORDER BY id DESC LIMIT 1",
        (tid,),
    ).fetchone()


# ---------------------------------------------------------------------------
# Happy path: running -> review
# ---------------------------------------------------------------------------


def test_request_review_transitions_running_to_review(kanban_home: Path) -> None:
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="impl a feature", assignee="worker")
        kb.claim_task(conn, tid)
        run_id = kb.get_task(conn, tid).current_run_id
        assert run_id is not None

        ok = kb.request_review(
            conn, tid,
            summary="Implementation complete\nfull details below",
            reviewer="reviewer",
            expected_run_id=run_id,
        )
        assert ok is True

        row = _row(conn, tid)
        assert row["status"] == "review"
        # The active run is closed and the pointer cleared.
        assert row["current_run_id"] is None
        # Not a block: recurrence machinery is untouched.
        assert (row["block_recurrences"] or 0) == 0
        assert row["block_kind"] is None

        run = _last_run(conn, tid)
        assert run["outcome"] == "review_requested"
        assert run["status"] == "review"

        # Exactly one review_requested event, with the handoff payload.
        rr = _events(conn, tid, kind="review_requested")
        assert len(rr) == 1
        payload = rr[0][1]
        assert payload["implementer"] == "worker"
        assert payload["reviewer"] == "reviewer"
        # First line of the summary rides the event payload.
        assert payload["summary"] == "Implementation complete"
        # No block / triage events were emitted.
        assert _events(conn, tid, kind="blocked") == []
        assert _events(conn, tid, kind="block_loop_detected") == []


# ---------------------------------------------------------------------------
# Core regression: repeated review requests never escalate to triage
# ---------------------------------------------------------------------------


def test_repeated_review_requests_never_triage(kanban_home: Path) -> None:
    """A task that goes review -> rerun -> review again (the executor
    follow-up cycle) must stay in ``review`` every time. Under the old
    ``kanban_block(review-required:)`` approach the second pass hit
    ``block_recurrences >= 2`` and was wrongly routed to ``triage`` with a
    ``block_loop_detected`` event. ``request_review`` must never do that."""
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="cycle me", assignee="worker")

        for _ in range(4):
            # Executor claims (ready->running or review->running) and finishes
            # with a review request. claim_review_task handles review->running.
            task = kb.get_task(conn, tid)
            if task.status == "ready":
                kb.claim_task(conn, tid)
            else:
                assert task.status == "review"
                claimed = kb.claim_review_task(conn, tid)
                assert claimed is not None

            run_id = kb.get_task(conn, tid).current_run_id
            ok = kb.request_review(
                conn, tid,
                summary="pass complete",
                expected_run_id=run_id,
            )
            assert ok is True
            row = _row(conn, tid)
            assert row["status"] == "review", "must never leave the review lane"
            assert (row["block_recurrences"] or 0) == 0

        # After several cycles: never triaged, never a false loop.
        assert _row(conn, tid)["status"] == "review"
        assert _events(conn, tid, kind="block_loop_detected") == []
        assert len(_events(conn, tid, kind="review_requested")) == 4


# ---------------------------------------------------------------------------
# CAS guard + bad-input behaviour
# ---------------------------------------------------------------------------


def test_request_review_expected_run_id_mismatch_is_noop(kanban_home: Path) -> None:
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="stale worker", assignee="worker")
        kb.claim_task(conn, tid)
        real_run = kb.get_task(conn, tid).current_run_id

        # A superseded worker passes a run id that is not the current one.
        ok = kb.request_review(conn, tid, expected_run_id=(real_run or 0) + 999)
        assert ok is False
        # Task is untouched — still running under the real run.
        row = _row(conn, tid)
        assert row["status"] == "running"
        assert row["current_run_id"] == real_run
        assert _events(conn, tid, kind="review_requested") == []


def test_request_review_unknown_task_returns_false(kanban_home: Path) -> None:
    with kb.connect() as conn:
        assert kb.request_review(conn, "t_deadbeefcafe") is False


def test_request_review_refuses_to_clear_live_claim_without_ownership(
    kanban_home: Path,
) -> None:
    """M1 regression: a run-id-less caller must not steal a live worker's claim.

    ``request_review`` on a running+claimed task without ``expected_run_id``
    fails with a distinct reason instead of silently NULLing claim_lock /
    worker_pid. ``force=True`` (explicit human override) and the worker path
    (``expected_run_id=<own run>``) both still work.
    """
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="live claim", assignee="worker")
        claimed = kb.claim_task(conn, tid)
        assert claimed is not None

        # 1) No run id, no force -> refused with a distinct reason.
        ok, reason = kb.request_review(conn, tid, with_reason=True)
        assert ok is False
        assert reason is not None and "live claim" in reason
        row = conn.execute(
            "SELECT status, claim_lock, current_run_id FROM tasks WHERE id = ?",
            (tid,),
        ).fetchone()
        assert row["status"] == "running"
        assert row["claim_lock"] is not None  # live claim untouched
        # bool-mode caller sees plain False.
        assert kb.request_review(conn, tid) is False

        # 2) Worker path: proving ownership via expected_run_id works.
        assert kb.request_review(
            conn, tid, summary="done", expected_run_id=claimed.current_run_id,
        ) is True
        assert kb.get_task(conn, tid).status == "review"

    # 3) force=True: explicit human override on a fresh live-claimed task.
    with kb.connect() as conn:
        tid2 = kb.create_task(conn, title="forced", assignee="worker")
        assert kb.claim_task(conn, tid2) is not None
        assert kb.request_review(conn, tid2, summary="override", force=True) is True
        assert kb.get_task(conn, tid2).status == "review"


def test_request_review_malformed_provenance_gets_distinct_reason(
    kanban_home: Path,
) -> None:
    """M1 regression: malformed re-review provenance is a named failure, not
    the generic 'unknown id or not in running/ready'."""
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="provenance", assignee="builder")
        claimed = kb.claim_task(conn, tid)
        assert kb.request_review(
            conn, tid, summary="v1", reviewer="reviewer",
            expected_run_id=claimed.current_run_id,
        )
        review = kb.claim_review_task(conn, tid)
        assert review is not None
        assert kb.request_changes(
            conn, tid, reason="fix", expected_run_id=review.current_run_id,
        ) == (True, "builder")
        # Corrupt the changes_requested payload so re-review cannot recover
        # the prior reviewer.
        with kb.write_txn(conn):
            conn.execute(
                "UPDATE task_events SET payload = '{\"reviewer\": 42}' "
                "WHERE task_id = ? AND kind = 'changes_requested'",
                (tid,),
            )
        retry = kb.claim_task(conn, tid, claimer="builder:retry")
        assert retry is not None
        ok, reason = kb.request_review(
            conn, tid, summary="v2",
            expected_run_id=retry.current_run_id, with_reason=True,
        )
        assert ok is False
        assert reason is not None and "provenance" in reason
        # Passing reviewer explicitly recovers, as the reason instructs.
        assert kb.request_review(
            conn, tid, summary="v2", reviewer="reviewer",
            expected_run_id=retry.current_run_id,
        ) is True


@pytest.mark.parametrize("blank", ["   ", "\n", "\t\n  "])
def test_request_review_whitespace_only_summary_does_not_crash(
    kanban_home: Path, blank: str
) -> None:
    """A whitespace-only handoff summary must not crash the review transition.

    Regression: the event-summary extraction tested the truthiness of the
    *pre-strip* value while indexing the *post-strip* (empty) list, so a
    summary like ``"   "`` is truthy, ``.strip()`` collapses it to ``""``,
    ``"".splitlines()`` is ``[]`` and ``[][0]`` raised ``IndexError`` inside
    ``write_txn`` — a 500 on the dashboard PATCH/bulk path, which forwards
    ``summary`` unstripped (the tool/CLI paths pre-strip to ``None`` and were
    never exposed). The transition must still succeed and the event must
    carry ``summary=None`` (whitespace collapses to no summary).
    """
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="blank summary", assignee="worker")
        kb.claim_task(conn, tid)
        run_id = kb.get_task(conn, tid).current_run_id

        ok = kb.request_review(conn, tid, summary=blank, expected_run_id=run_id)
        assert ok is True
        assert kb.get_task(conn, tid).status == "review"

        rr = _events(conn, tid, kind="review_requested")
        assert len(rr) == 1
        # Whitespace collapses to no summary on the event payload.
        assert rr[0][1]["summary"] is None


# ---------------------------------------------------------------------------
# review -> done: a human can approve/close a task parked in review
# ---------------------------------------------------------------------------


def test_complete_task_closes_review_to_done(kanban_home: Path) -> None:
    """A task parked in ``review`` (with no active run — request_review
    closed it, so ``current_run_id IS NULL``, the #54823 shape) must be
    completable by a human approval via ``complete_task``."""
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="approve me", assignee="worker")
        kb.claim_task(conn, tid)
        kb.request_review(
            conn, tid, summary="ready",
            expected_run_id=kb.get_task(conn, tid).current_run_id,
        )
        assert kb.get_task(conn, tid).status == "review"
        # The review lane has no active run — the exact state that used to
        # make `hermes kanban complete` a no-op (#54823).
        assert kb.get_task(conn, tid).current_run_id is None

        ok = kb.complete_task(conn, tid, summary="LGTM — merged", result="approved")
        assert ok is True
        assert kb.get_task(conn, tid).status == "done"
        assert _events(conn, tid, kind="completed")


# ---------------------------------------------------------------------------
# Wake plumbing: review_requested is a claimable terminal event for a sub
# ---------------------------------------------------------------------------


def test_review_requested_event_is_claimable_for_wake(kanban_home: Path) -> None:
    """The gateway kanban-notifier wakes an origin subscription by claiming
    unseen events whose kind is in its terminal set. ``review_requested`` is
    now in that set, so a wake subscription must see the event — and the
    subscription is NOT torn down (task is in ``review``, not done/archived),
    so later review cycles keep notifying."""
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="wake me", assignee="worker")
        kb.add_notify_sub(
            conn,
            task_id=tid,
            platform="slack",
            chat_id="C123",
            thread_id="T1",
        )
        kb.claim_task(conn, tid)
        kb.request_review(
            conn, tid, summary="please review",
            expected_run_id=kb.get_task(conn, tid).current_run_id,
        )

        # Same terminal set the notifier now uses (incl. review_requested).
        terminal_kinds = (
            "completed", "blocked", "gave_up", "crashed", "timed_out",
            "review_requested",
        )
        _old, _new, events = kb.claim_unseen_events_for_sub(
            conn,
            task_id=tid,
            platform="slack",
            chat_id="C123",
            thread_id="T1",
            kinds=terminal_kinds,
        )
        kinds_seen = [e.kind for e in events]
        assert "review_requested" in kinds_seen
        # Task is parked in review — the subscription must survive (only
        # done/archived tears it down), so subsequent cycles still wake.
        assert kb.get_task(conn, tid).status == "review"


# ---------------------------------------------------------------------------
# Dispatcher gate: operators may opt out of autonomous review dispatch
# ---------------------------------------------------------------------------


def test_review_dispatch_gate_prevents_phantom_reviewer(
    kanban_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With ``kanban.review_dispatch=false`` the dispatcher must NOT claim a
    task parked in ``review`` (this deployment explicitly waits for a human).
    Flipping the knob back on proves the gate, not
    something else, is what suppressed the claim."""
    import hermes_cli.config as cfgmod
    import hermes_cli.profiles as profmod

    with kb.connect() as conn:
        tid = kb.create_task(conn, title="park", assignee="worker")
        kb.claim_task(conn, tid)
        kb.request_review(
            conn, tid, summary="done",
            expected_run_id=kb.get_task(conn, tid).current_run_id,
        )
        assert kb.get_task(conn, tid).status == "review"

        # The assignee profile is spawnable — so ONLY the gate can stop the
        # review-column dispatch from claiming it.
        monkeypatch.setattr(profmod, "profile_exists", lambda name: True)

        # Gate OFF -> review task is left alone.
        monkeypatch.setattr(
            cfgmod, "load_config",
            lambda *a, **k: {"kanban": {"review_dispatch": False}},
        )
        res_off = kb.dispatch_once(conn, dry_run=True)
        assert tid not in [s[0] for s in res_off.spawned]
        assert kb.get_task(conn, tid).status == "review"

        # Gate ON (the default; sdlc-review is bundled) -> the review task is
        # picked up by the dispatcher.
        monkeypatch.setattr(
            cfgmod, "load_config",
            lambda *a, **k: {"kanban": {"review_dispatch": True}},
        )
        res_on = kb.dispatch_once(conn, dry_run=True)
        assert tid in [s[0] for s in res_on.spawned]


def test_active_pr_guard_skipped_for_review_lane_but_defers_ready_lane(
    kanban_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """B2 regression: a fresh PR-URL comment must not block reviewer spawns.

    A task parked in ``review`` with a PR link younger than 24h is the
    CANONICAL review handoff (worker opened a PR then requested review) —
    the review-lane dispatch must still claim/spawn it. The same comment on
    a ready-lane task is a duplicate-work signal and stays deferred.
    Rate-limit cooldown still applies in the review lane.
    """
    import hermes_cli.config as cfgmod
    import hermes_cli.profiles as profmod

    monkeypatch.setattr(profmod, "profile_exists", lambda name: True)
    monkeypatch.setattr(
        cfgmod, "load_config",
        lambda *a, **k: {"kanban": {"review_dispatch": True}},
    )
    pr_comment = "Opened https://github.com/example/repo/pull/123 for review."

    with kb.connect() as conn:
        # Review-lane task with a fresh PR comment.
        review_id = kb.create_task(conn, title="review me", assignee="reviewer")
        claimed = kb.claim_task(conn, review_id)
        assert claimed is not None
        kb.add_comment(conn, review_id, author="worker", body=pr_comment)
        assert kb.request_review(
            conn, review_id, summary="PR ready",
            expected_run_id=claimed.current_run_id,
        )
        # Ready-lane task with the same fresh PR comment.
        ready_id = kb.create_task(conn, title="already PRed", assignee="worker")
        kb.add_comment(conn, ready_id, author="worker", body=pr_comment)

        assert kb.check_respawn_guard(conn, ready_id) == "active_pr"
        assert kb.check_respawn_guard(conn, review_id, lane="review") is None

        res = kb.dispatch_once(conn, dry_run=True)
        spawned_ids = [s[0] for s in res.spawned]
        guarded = dict(res.respawn_guarded)
        assert review_id in spawned_ids
        assert ready_id not in spawned_ids
        assert guarded.get(ready_id) == "active_pr"

        # Rate-limit cooldown still defers the review lane.
        _now = int(__import__("time").time())
        with kb.write_txn(conn):
            conn.execute(
                "INSERT INTO task_runs (task_id, profile, status, outcome, "
                "started_at, ended_at) VALUES (?, 'reviewer', 'rate_limited', "
                "'rate_limited', ?, ?)",
                # ended_at strictly after the review-handoff run so the
                # "latest run" query deterministically picks this one.
                (review_id, _now, _now + 5),
            )
        assert kb.check_respawn_guard(
            conn, review_id, lane="review"
        ) == "rate_limit_cooldown"


_HANDBACK_PR_COMMENT = (
    "Candidate is up at https://github.com/example/repo/pull/123 - please review."
)


def _make_review_handback(conn, *, title: str, pr_comment: str = _HANDBACK_PR_COMMENT) -> str:
    """Drive a real implement -> PR comment -> review -> changes_requested cycle.

    Returns the task id of a genuine, current review handback: ``ready``,
    ``current_run_id`` NULL, assignee restored to the implementer — built
    entirely from real lifecycle calls, matching the shape the dispatcher
    sees in production.
    """
    tid = kb.create_task(conn, title=title, assignee="worker")
    claimed = kb.claim_task(conn, tid)
    assert claimed is not None
    kb.add_comment(conn, tid, author="worker", body=pr_comment)
    ok = kb.request_review(
        conn, tid, summary="candidate ready", reviewer="reviewer",
        expected_run_id=claimed.current_run_id,
    )
    assert ok is True
    review_claim = kb.claim_review_task(conn, tid)
    assert review_claim is not None
    ok, who = kb.request_changes(
        conn, tid, reason="fix the guard boundary",
        expected_run_id=review_claim.current_run_id,
    )
    assert ok is True, who
    return tid


_SAME_SECOND_MAX_ATTEMPTS = 50


def _retry_until_same_second(attempt):
    """Retry a real-lifecycle attempt until it lands two rows in one whole
    second, or skip if the host is too slow/loaded to ever produce that.

    ``attempt`` builds a brand-new, disposable task via the real lifecycle
    helpers (so a discarded attempt leaves no state behind) and returns
    ``(task_id, landed_same_second)``. This never fakes the clock — it just
    keeps trying real wall-clock operations until scheduler jitter happens to
    cooperate, bounded so a persistently slow machine skips instead of
    hanging or flaking red.
    """
    for _ in range(_SAME_SECOND_MAX_ATTEMPTS):
        tid, landed = attempt()
        if landed:
            return tid
    pytest.skip(
        "could not land a same-second race in "
        f"{_SAME_SECOND_MAX_ATTEMPTS} attempts - host too slow/loaded to "
        "exercise the same-second tiebreak path"
    )


def test_review_handback_supersedes_earlier_pr_comment_allows_respawn(
    kanban_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A genuine, current review handback authorises a respawn even though
    the PR-URL comment that preceded it is still inside the 24h window.

    ``request_changes`` is the kernel-written proof a reviewer already
    looked at that exact PR and sent the task back for rework — the
    dispatcher must let the implementer run again, not treat the old PR
    link as duplicate-work evidence forever.
    """
    import hermes_cli.config as cfgmod
    import hermes_cli.profiles as profmod

    monkeypatch.setattr(profmod, "profile_exists", lambda name: True)
    monkeypatch.setattr(
        cfgmod, "load_config",
        lambda *a, **k: {"kanban": {"review_dispatch": True}},
    )

    with kb.connect() as conn:
        tid = _make_review_handback(conn, title="handback earlier pr")
        task = kb.get_task(conn, tid)
        assert task is not None
        assert task.status == "ready"
        assert task.current_run_id is None
        assert task.assignee == "worker"

        assert kb.check_respawn_guard(conn, tid) is None
        assert kbd.check_respawn_guard(conn, tid) is None

        spawned: list[str] = []

        def spawn(task, workspace):
            spawned.append(task.id)
            return None

        result = kb.dispatch_once(conn, spawn_fn=spawn)
        assert tid in [s[0] for s in result.spawned]
        assert dict(result.respawn_guarded).get(tid) is None
        assert tid in spawned
        assert kb.get_task(conn, tid).status == "running"


def test_review_handback_respawn_claim_is_race_safe(kanban_home: Path) -> None:
    """Two dispatchers racing the same handed-back task must not double-claim.

    Both ``kanban_db.dispatch_once`` and the decomposed
    ``kanban_db_dispatch`` dispatch loop claim a ready task through the SAME
    shared ``kanban_db.claim_task`` CAS (``kanban_db_dispatch`` calls it via
    its late-bound ``_kb`` alias) — so racing the two dispatchers on this
    handback reduces to racing two ``claim_task`` calls on it. Both guard
    copies must first agree the respawn is authorized; then exactly one of
    two racing claim attempts may land, and the other must see the task
    already claimed rather than mint a duplicate worker. The guard fix must
    not have loosened that CAS.
    """
    with kb.connect() as conn:
        tid = _make_review_handback(conn, title="race the handback")
        assert kb.check_respawn_guard(conn, tid) is None
        assert kbd.check_respawn_guard(conn, tid) is None

        first = kb.claim_task(conn, tid)
        second = kb.claim_task(conn, tid)

        assert first is not None
        assert second is None
        assert kb.get_task(conn, tid).status == "running"
        assert kb.get_task(conn, tid).current_run_id == first.current_run_id


def test_stale_review_handback_older_than_pr_comment_stays_blocked(
    kanban_home: Path,
) -> None:
    """A handback OLDER than a PR-URL comment authorises nothing: the newer
    comment is fresh, still-unreviewed duplicate-work evidence.

    Builds a genuine handback with no PR comment yet, then posts a fresh
    PR-URL comment afterwards via the real lifecycle calls — the handback
    event durably precedes the comment's ``commented`` event in real
    insertion order, so the guard must still return ``active_pr`` in both
    copies.
    """
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="stale handback", assignee="worker")
        claimed = kb.claim_task(conn, tid)
        assert claimed is not None
        ok = kb.request_review(
            conn, tid, summary="v1", reviewer="reviewer",
            expected_run_id=claimed.current_run_id,
        )
        assert ok is True
        review_claim = kb.claim_review_task(conn, tid)
        assert review_claim is not None
        ok, who = kb.request_changes(
            conn, tid, reason="fix", expected_run_id=review_claim.current_run_id,
        )
        assert ok is True, who

        # A fresh PR-URL comment lands AFTER the handback.
        kb.add_comment(conn, tid, author="worker", body=_HANDBACK_PR_COMMENT)

        # Confirm real ordering: the handback event durably precedes the
        # comment's own `commented` event — no history was rewritten.
        handback_event_id = conn.execute(
            "SELECT id FROM task_events WHERE task_id = ? "
            "AND kind = 'changes_requested'",
            (tid,),
        ).fetchone()["id"]
        comment_event_id = conn.execute(
            "SELECT id FROM task_events WHERE task_id = ? "
            "AND kind = 'commented' ORDER BY id DESC LIMIT 1",
            (tid,),
        ).fetchone()["id"]
        assert handback_event_id < comment_event_id

        assert kb.check_respawn_guard(conn, tid) == "active_pr"
        assert kbd.check_respawn_guard(conn, tid) == "active_pr"


def test_review_handback_does_not_authorize_unrelated_task(
    kanban_home: Path,
) -> None:
    """A handback on one task must not authorise a duplicate-PR respawn on a
    different, unrelated task."""
    with kb.connect() as conn:
        handback_id = _make_review_handback(conn, title="handback task")
        other_id = kb.create_task(conn, title="unrelated ready task", assignee="worker")
        kb.add_comment(conn, other_id, author="worker", body=_HANDBACK_PR_COMMENT)

        assert kb.check_respawn_guard(conn, handback_id) is None
        assert kbd.check_respawn_guard(conn, handback_id) is None

        assert kb.check_respawn_guard(conn, other_id) == "active_pr"
        assert kbd.check_respawn_guard(conn, other_id) == "active_pr"


def test_review_handback_same_second_as_prior_pr_comment_allows_respawn(
    kanban_home: Path,
) -> None:
    """SAME-SECOND boundary: the common case where the PR-URL comment and
    the genuine handback that follows it land in the same whole-second tick
    (a fast ``request_review`` -> ``claim_review_task`` -> ``request_changes``
    round trip, exactly what ``_make_review_handback`` drives). ``created_at``
    alone can't order same-second rows, so the guard must fall back to real
    insertion order (via the comment's own ``commented`` event id) and still
    recognize the handback as later.
    """
    with kb.connect() as conn:
        def attempt():
            tid = _make_review_handback(conn, title="same second handback")
            comment_row = conn.execute(
                "SELECT created_at FROM task_comments WHERE task_id = ? "
                "ORDER BY id ASC LIMIT 1",
                (tid,),
            ).fetchone()
            handback_row = conn.execute(
                "SELECT created_at FROM task_events WHERE task_id = ? "
                "AND kind = 'changes_requested'",
                (tid,),
            ).fetchone()
            assert comment_row is not None and handback_row is not None
            landed = int(comment_row["created_at"]) == int(handback_row["created_at"])
            return tid, landed

        tid = _retry_until_same_second(attempt)

        assert kb.check_respawn_guard(conn, tid) is None
        assert kbd.check_respawn_guard(conn, tid) is None


def test_new_pr_comment_same_second_as_handback_stays_guarded(
    kanban_home: Path,
) -> None:
    """SAME-SECOND boundary in the other direction: a brand-new, unreviewed
    PR-URL comment posted in the very same tick as an (already-consumed)
    handback must still guard the respawn. Flipping the stale comparison to
    strict ``>`` alone would fix this direction but break the previous one —
    only the real insertion order (comment's own ``commented`` event id vs.
    the handback event id) gets both right.
    """
    with kb.connect() as conn:
        def attempt():
            tid = _make_review_handback(
                conn, title="new pr same second", pr_comment="Working on the fix."
            )
            kb.add_comment(conn, tid, author="worker", body=_HANDBACK_PR_COMMENT)

            handback_row = conn.execute(
                "SELECT created_at FROM task_events WHERE task_id = ? "
                "AND kind = 'changes_requested'",
                (tid,),
            ).fetchone()
            new_comment_row = conn.execute(
                "SELECT created_at FROM task_comments WHERE task_id = ? "
                "ORDER BY id DESC LIMIT 1",
                (tid,),
            ).fetchone()
            assert handback_row is not None and new_comment_row is not None
            landed = int(handback_row["created_at"]) == int(new_comment_row["created_at"])
            return tid, landed

        tid = _retry_until_same_second(attempt)

        assert kb.check_respawn_guard(conn, tid) == "active_pr"
        assert kbd.check_respawn_guard(conn, tid) == "active_pr"


def test_pr_comment_without_commented_event_same_second_fails_closed(
    kanban_home: Path,
) -> None:
    """A PR-URL comment with no corresponding ``commented`` event (mirroring
    ``kanban_db``'s inline ``INSERT INTO task_comments`` sites, e.g.
    ``specify_triage_task``) can't be placed relative to a same-second
    handback by event id — the guard must fail CLOSED rather than guess, and
    keep treating it as an active, unreviewed PR.

    This is the one place in the suite allowed to insert a comment row
    directly: the whole point is exercising the missing-event case. No
    ``changes_requested``/``review_reopened`` event is fabricated anywhere.
    """
    with kb.connect() as conn:
        tid = _make_review_handback(
            conn, title="missing commented event", pr_comment="Working on the fix."
        )
        handback_row = conn.execute(
            "SELECT created_at FROM task_events WHERE task_id = ? "
            "AND kind = 'changes_requested'",
            (tid,),
        ).fetchone()
        assert handback_row is not None
        handback_at = int(handback_row["created_at"])

        with kb.write_txn(conn):
            conn.execute(
                "INSERT INTO task_comments (task_id, author, body, created_at) "
                "VALUES (?, ?, ?, ?)",
                (tid, "worker", _HANDBACK_PR_COMMENT, handback_at),
            )

        comment_row = conn.execute(
            "SELECT created_at FROM task_comments WHERE task_id = ? "
            "ORDER BY id DESC LIMIT 1",
            (tid,),
        ).fetchone()
        assert comment_row is not None
        assert int(comment_row["created_at"]) == handback_at

        assert kb.check_respawn_guard(conn, tid) == "active_pr"
        assert kbd.check_respawn_guard(conn, tid) == "active_pr"


def test_pr_comment_fingerprint_collision_new_pr_stays_guarded(
    kanban_home: Path,
) -> None:
    """Regression test for a same-author, same-length PR-comment fingerprint
    collision that used to fail OPEN.

    Two PR-URL comments by the SAME author with the SAME body length (only
    the PR number differs) used to have byte-identical
    ``(author, len(body))`` payload fingerprints — indistinguishable to the
    old content-matching tiebreak. The FIRST comment (posted BEFORE the
    handback) would match whichever ``commented`` event the matcher found
    first, letting the guard conclude the SECOND, unreviewed comment (posted
    AFTER the handback) was also superseded. Ordinal pairing by row id,
    never by content, cannot confuse the two.
    """
    pr1 = "Candidate is up at https://github.com/example/repo/pull/123 - go."
    pr2 = "Candidate is up at https://github.com/example/repo/pull/456 - go."
    assert len(pr1) == len(pr2)

    with kb.connect() as conn:
        def attempt():
            tid = kb.create_task(conn, title="fingerprint collision", assignee="worker")
            claimed = kb.claim_task(conn, tid)
            assert claimed is not None
            # PR comment #1 BEFORE the review round trip.
            kb.add_comment(conn, tid, author="worker", body=pr1)
            ok = kb.request_review(
                conn, tid, summary="ready", reviewer="reviewer",
                expected_run_id=claimed.current_run_id,
            )
            assert ok is True
            review_claim = kb.claim_review_task(conn, tid)
            assert review_claim is not None
            ok, who = kb.request_changes(
                conn, tid, reason="fix", expected_run_id=review_claim.current_run_id,
            )
            assert ok is True, who
            # PR comment #2 AFTER the handback: a NEW, unreviewed PR, same
            # author and body length as comment #1.
            kb.add_comment(conn, tid, author="worker", body=pr2)

            handback_row = conn.execute(
                "SELECT created_at FROM task_events WHERE task_id = ? "
                "AND kind = 'changes_requested'",
                (tid,),
            ).fetchone()
            comment_rows = conn.execute(
                "SELECT created_at FROM task_comments WHERE task_id = ? ORDER BY id",
                (tid,),
            ).fetchall()
            assert handback_row is not None
            seconds = {int(handback_row["created_at"])} | {
                int(r["created_at"]) for r in comment_rows
            }
            return tid, len(seconds) == 1

        tid = _retry_until_same_second(attempt)

        assert kb.check_respawn_guard(conn, tid) == "active_pr"
        assert kbd.check_respawn_guard(conn, tid) == "active_pr"


def test_pairing_unavailable_with_mixed_comment_sources_fails_closed(
    kanban_home: Path,
) -> None:
    """When even ONE comment on the task bypassed ``add_comment`` (an inline
    ``INSERT INTO task_comments``, mirroring ``specify_triage_task``), the
    task's total comment count no longer equals its total ``commented``
    event count, so ordinal pairing cannot be established for ANY comment on
    the task — not just the directly-inserted one. A same-second PR-URL
    comment that DOES have its own ``commented`` event, and would otherwise
    be provably superseded, must still fail closed once that correspondence
    is broken.
    """
    with kb.connect() as conn:
        def attempt():
            tid = _make_review_handback(
                conn, title="mixed comment sources", pr_comment="Working on the fix."
            )
            handback_row = conn.execute(
                "SELECT created_at FROM task_events WHERE task_id = ? "
                "AND kind = 'changes_requested'",
                (tid,),
            ).fetchone()
            assert handback_row is not None
            handback_at = int(handback_row["created_at"])

            # A comment inserted directly (no ``commented`` event) — mirrors
            # an inline INSERT site elsewhere in kanban_db, alongside the
            # normal, event-backed comments already on this task.
            with kb.write_txn(conn):
                conn.execute(
                    "INSERT INTO task_comments (task_id, author, body, created_at) "
                    "VALUES (?, ?, ?, ?)",
                    (tid, "worker", "no event for this one", handback_at),
                )

            # A normal, event-backed PR-URL comment landing in the same second.
            kb.add_comment(conn, tid, author="worker", body=_HANDBACK_PR_COMMENT)

            comment_rows = conn.execute(
                "SELECT created_at FROM task_comments WHERE task_id = ? ORDER BY id",
                (tid,),
            ).fetchall()
            landed = all(int(r["created_at"]) == handback_at for r in comment_rows)
            return tid, landed

        tid = _retry_until_same_second(attempt)

        assert kb.check_respawn_guard(conn, tid) == "active_pr"
        assert kbd.check_respawn_guard(conn, tid) == "active_pr"


def test_superseded_handback_does_not_authorize_later_respawn_or_other_task(
    kanban_home: Path,
) -> None:
    """A handback stays powerless once superseded: it must not go on
    authorizing respawns on ITS OWN task after a newer, unreviewed PR
    comment lands, and (mirroring
    ``test_review_handback_does_not_authorize_unrelated_task``) it must
    never authorize a DIFFERENT task either.
    """
    with kb.connect() as conn:
        tid = _make_review_handback(
            conn, title="handback then new pr", pr_comment="Working on the fix."
        )
        assert kb.check_respawn_guard(conn, tid) is None
        assert kbd.check_respawn_guard(conn, tid) is None

        # A NEW, unreviewed PR-URL comment lands after the handback.
        kb.add_comment(conn, tid, author="worker", body=_HANDBACK_PR_COMMENT)
        assert kb.check_respawn_guard(conn, tid) == "active_pr"
        assert kbd.check_respawn_guard(conn, tid) == "active_pr"

        # Re-checking later still finds the same (now-superseded) handback
        # powerless — it does not get to authorize the task again.
        assert kb.check_respawn_guard(conn, tid) == "active_pr"
        assert kbd.check_respawn_guard(conn, tid) == "active_pr"

        other_id = kb.create_task(
            conn, title="unrelated ready task 2", assignee="worker"
        )
        kb.add_comment(conn, other_id, author="worker", body=_HANDBACK_PR_COMMENT)
        assert kb.check_respawn_guard(conn, other_id) == "active_pr"
        assert kbd.check_respawn_guard(conn, other_id) == "active_pr"


def test_review_dispatch_preserves_task_skills_and_adds_reviewer_skill(
    kanban_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import hermes_cli.config as cfgmod
    import hermes_cli.profiles as profmod

    monkeypatch.setattr(profmod, "profile_exists", lambda name: True)
    monkeypatch.setattr(
        cfgmod,
        "load_config",
        lambda *args, **kwargs: {"kanban": {"review_dispatch": True}},
    )
    captured: list[list[str]] = []

    def spawn(task, workspace):
        captured.append(list(task.skills or []))
        return None

    with kb.connect() as conn:
        task_id = kb.create_task(
            conn,
            title="domain review",
            assignee="reviewer",
            skills=["domain-specific-review"],
        )
        implementation = kb.claim_task(conn, task_id)
        assert implementation is not None
        assert kb.request_review(
            conn,
            task_id,
            summary="ready",
            expected_run_id=implementation.current_run_id,
        )
        monkeypatch.setattr(
            kb,
            "check_respawn_guard",
            lambda _conn, _task_id, **_kw: "rate_limit_cooldown",
        )
        guarded = kb.dispatch_once(conn, spawn_fn=spawn)
        assert guarded.respawn_guarded == [(task_id, "rate_limit_cooldown")]
        assert not guarded.spawned
        guarded_task = kb.get_task(conn, task_id)
        assert guarded_task is not None
        assert guarded_task.status == "review"

        monkeypatch.setattr(kb, "check_respawn_guard", lambda _conn, _task_id, **_kw: None)
        result = kb.dispatch_once(conn, spawn_fn=spawn)

    assert task_id in [task[0] for task in result.spawned]
    assert captured == [["domain-specific-review", "sdlc-review"]]


def test_review_dispatch_drops_the_build_skill_and_keeps_every_other_skill(
    kanban_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A reviewer never receives build instructions.

    A build card carries ``running-build-work`` for its implementer. When the
    card reaches review, the review lane drops exactly that name: the reviewer
    keeps its own ``sdlc-review`` skill and every other skill the task
    carries, and the card itself keeps the build skill for any rework.
    """
    import hermes_cli.config as cfgmod
    import hermes_cli.profiles as profmod

    monkeypatch.setattr(profmod, "profile_exists", lambda name: True)
    monkeypatch.setattr(
        cfgmod,
        "load_config",
        lambda *args, **kwargs: {"kanban": {"review_dispatch": True}},
    )
    monkeypatch.setattr(kb, "check_respawn_guard", lambda _conn, _task_id, **_kw: None)
    captured: list[list[str]] = []

    def spawn(task, workspace):
        captured.append(list(task.skills or []))
        return None

    with kb.connect() as conn:
        task_id = kb.create_task(
            conn,
            title="scoped build",
            assignee="reviewer",
            skills=["running-build-work", "domain-specific-review"],
        )
        implementation = kb.claim_task(conn, task_id)
        assert implementation is not None
        assert kb.request_review(
            conn,
            task_id,
            summary="ready",
            expected_run_id=implementation.current_run_id,
        )
        result = kb.dispatch_once(conn, spawn_fn=spawn)
        stored = kb.get_task(conn, task_id)

    assert task_id in [task[0] for task in result.spawned]
    assert captured == [["domain-specific-review", "sdlc-review"]]
    assert stored is not None
    assert stored.skills == ["running-build-work", "domain-specific-review"]


def test_review_dispatch_honors_global_and_per_profile_caps(
    kanban_home: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import hermes_cli.config as cfgmod
    import hermes_cli.profiles as profmod

    monkeypatch.setattr(profmod, "profile_exists", lambda _name: True)
    monkeypatch.setattr(
        cfgmod,
        "load_config",
        lambda *args, **kwargs: {"kanban": {"review_dispatch": True}},
    )

    with kb.connect() as conn:
        running_id = kb.create_task(conn, title="already running", assignee="builder")
        running = kb.claim_task(conn, running_id)
        assert running is not None

        review_ids: list[str] = []
        for title in ("review one", "review two"):
            task_id = kb.create_task(conn, title=title, assignee="reviewer")
            implementation = kb.claim_task(conn, task_id)
            assert implementation is not None
            assert kb.request_review(
                conn,
                task_id,
                summary="ready",
                expected_run_id=implementation.current_run_id,
            )
            review_ids.append(task_id)

        globally_capped = kb.dispatch_once(
            conn,
            dry_run=True,
            max_in_progress=1,
        )
        assert not [
            task for task in globally_capped.spawned if task[0] in review_ids
        ]

        assert kb.complete_task(
            conn,
            running_id,
            expected_run_id=running.current_run_id,
        )
        global_dry_run = kb.dispatch_once(
            conn,
            dry_run=True,
            max_in_progress=1,
        )
        assert len([
            task for task in global_dry_run.spawned if task[0] in review_ids
        ]) == 1

        per_profile_capped = kb.dispatch_once(
            conn,
            dry_run=True,
            max_in_progress=10,
            max_in_progress_per_profile=1,
        )
        spawned_reviews = [
            task for task in per_profile_capped.spawned if task[0] in review_ids
        ]
        assert len(spawned_reviews) == 1
        assert len(per_profile_capped.skipped_per_profile_capped) == 1
        assert per_profile_capped.skipped_per_profile_capped[0][0] in review_ids


# ---------------------------------------------------------------------------
# reopen: a follow-up sends a review task back out for another pass
# ---------------------------------------------------------------------------


def test_reopen_review_task_returns_to_ready(kanban_home: Path) -> None:
    """The "changes requested" / follow-up path: a task parked in ``review``
    goes back to ``ready`` so the dispatcher re-runs the implementer. It must
    NOT touch ``block_recurrences`` (review was never a block)."""
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="reopen me", assignee="worker")
        kb.claim_task(conn, tid)
        kb.request_review(
            conn, tid, summary="v1", reviewer="reviewer",
            expected_run_id=kb.get_task(conn, tid).current_run_id,
        )
        reviewing = kb.get_task(conn, tid)
        assert reviewing is not None
        assert reviewing.status == "review"
        assert reviewing.assignee == "reviewer"

        ok = kb.reopen_review_task(conn, tid)
        assert ok is True
        row = _row(conn, tid)
        assert row["status"] == "ready"
        reopened = kb.get_task(conn, tid)
        assert reopened is not None
        assert reopened.assignee == "worker"
        assert row["current_run_id"] is None
        assert (row["block_recurrences"] or 0) == 0
        assert _events(conn, tid, kind="review_reopened")

        # Idempotent: not in review anymore -> reopening again is a no-op.
        assert kb.reopen_review_task(conn, tid) is False


def test_review_cycle_end_to_end(kanban_home: Path) -> None:
    """Full loop: run -> review -> follow-up reopen -> re-run -> review ->
    approve -> done. Never blocks, never triages, and stays wake-subscribed
    until done."""
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="cycle", assignee="worker")

        # Pass 1: implement -> review.
        kb.claim_task(conn, tid)
        kb.request_review(
            conn, tid, summary="v1",
            expected_run_id=kb.get_task(conn, tid).current_run_id,
        )
        assert kb.get_task(conn, tid).status == "review"

        # Human asks for changes -> reopen -> re-run.
        assert kb.reopen_review_task(conn, tid) is True
        assert kb.get_task(conn, tid).status == "ready"
        kb.claim_task(conn, tid)
        kb.request_review(
            conn, tid, summary="v2",
            expected_run_id=kb.get_task(conn, tid).current_run_id,
        )
        assert kb.get_task(conn, tid).status == "review"

        # Human approves.
        assert kb.complete_task(conn, tid, summary="approved") is True
        row = _row(conn, tid)
        assert row["status"] == "done"
        assert (row["block_recurrences"] or 0) == 0
        assert _events(conn, tid, kind="block_loop_detected") == []


# ---------------------------------------------------------------------------
# never-claimed 'ready' task: handoff must survive via a synthesized run
# ---------------------------------------------------------------------------


def test_request_review_on_unclaimed_ready_synthesizes_run(kanban_home: Path) -> None:
    """A manual/CLI request-review on a never-claimed ``ready`` task has no
    active run to close. The handoff summary must still be preserved on a
    synthesized run so the reviewer keeps the context."""
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="ready then review", assignee="worker")
        assert kb.get_task(conn, tid).status == "ready"
        assert kb.get_task(conn, tid).current_run_id is None

        ok = kb.request_review(conn, tid, summary="done without a claim")
        assert ok is True
        assert kb.get_task(conn, tid).status == "review"

        run = _last_run(conn, tid)
        assert run is not None
        assert run["outcome"] == "review_requested"
        assert run["summary"] == "done without a claim"
        # Exactly one review_requested event, carrying the handoff summary.
        evs = _events(conn, tid, kind="review_requested")
        assert len(evs) == 1
        assert evs[0][1]["summary"] == "done without a claim"


def test_reviewer_reassigns_for_autonomous_dispatch(kanban_home: Path) -> None:
    """An explicit reviewer routes the review run while preserving implementer provenance."""
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="route reviewer", assignee="worker")
        claimed = kb.claim_task(conn, tid)
        assert claimed is not None
        ok = kb.request_review(
            conn, tid, summary="v1", reviewer="lead-reviewer",
            expected_run_id=claimed.current_run_id,
        )
        assert ok is True
        assert kb.get_task(conn, tid).assignee == "lead-reviewer"
        ev = _events(conn, tid, kind="review_requested")[0][1]
        assert ev["reviewer"] == "lead-reviewer"
        assert ev["implementer"] == "worker"
