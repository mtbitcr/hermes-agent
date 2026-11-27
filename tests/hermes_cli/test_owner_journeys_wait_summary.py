"""The Project page says why no card starts while the journeys hold lasts.

Only the live-work branch of ``project_steward_snapshot`` asks the hold; every
other branch reads as before, with the hold active or not.
"""

from __future__ import annotations

import threading
import time

import pytest

from hermes_cli import journeys_hold, kanban_db, owner_workspace as ow
from tools import approval

WAIT = "Raphael is waiting for the checks after a release to end before it starts new work."
COORDINATING = "Raphael is coordinating the approved milestone."
PAUSED = "Raphael is paused and will not start new work."
NEEDS_ANSWER = "Raphael needs your answer before the plan can continue."
PROBLEM = "Raphael found a problem and is preparing the safest next step."
COMPLETE = "The approved work is complete."
WAITING_FOR_APPROVAL = "Raphael is waiting for an approved milestone before starting work."

ACTIVE = {"active": True, "reason": "running", "latest_end": 4_000_000_000}
INACTIVE = {"active": False, "reason": None, "latest_end": None}


@pytest.fixture
def ctx():
    return ow.OwnerContext(actor="default", profile="default", session="run_journeys_wait")


def _auto_approve(session_key: str, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        with approval._lock:
            queue = approval._gateway_queues.get(session_key)
            aid = queue[0].approval_id if queue else None
        if aid:
            approval.resolve_gateway_approval(session_key, "once", approval_id=aid)
            return
        time.sleep(0.005)
    raise TimeoutError("no approval was queued")


def _project(ctx) -> dict:
    approval.register_gateway_notify(ctx.session, lambda data: None)
    approver = threading.Thread(target=_auto_approve, args=(ctx.session,))
    approver.start()
    setup = ow.bootstrap(
        ctx, idempotency_key=f"journeys-wait-{time.monotonic()}", name="Journeys Wait",
    )
    approver.join()
    kanban_db.write_board_dispatch_state(setup["board"], dispatch_enabled=True)
    return setup


def _card(setup: dict, status: str, **columns) -> str:
    with kanban_db.connect(board=setup["board"]) as conn:
        task_id = kanban_db.create_task(
            conn, title="Draft the agenda", project_id=setup["project_id"],
        )
    sets = ", ".join(f"{name} = ?" for name in ("status", *columns))
    with kanban_db.connect(board=setup["board"]) as conn:
        with kanban_db.write_txn(conn):
            conn.execute(
                f"UPDATE tasks SET {sets} WHERE id = ?",
                (status, *columns.values(), task_id),
            )
    return task_id


def _snapshot(setup: dict) -> dict:
    return ow.project_steward_snapshot(project_id=setup["project_id"])


@pytest.fixture
def hold(monkeypatch):
    """Sets the hold and counts how often the snapshot asks it."""
    calls: list = []

    def set_hold(answer):
        def state(now):
            calls.append(now)
            if isinstance(answer, Exception):
                raise answer
            return dict(answer)

        monkeypatch.setattr(journeys_hold, "state", state)
        return calls

    return set_hold


def _execution(setup: dict) -> dict:
    return _snapshot(setup)["execution"]


# Rule point 8: asked once, only in the live-work branch, the sentence byte for byte.
def test_rule8_live_work_reads_the_wait_sentence_while_the_hold_is_active(ctx, hold):
    setup = _project(ctx)
    _card(setup, "running")
    calls = hold(ACTIVE)

    assert _execution(setup) == {"state": "working", "summary": WAIT, "paused": False}
    assert WAIT.encode() == (
        b"Raphael is waiting for the checks after a release to end before it starts new work."
    )
    assert len(calls) == 1


def test_rule8_the_live_work_branch_asks_the_hold_exactly_once(ctx, hold):
    setup = _project(ctx)
    _card(setup, "running")
    _card(setup, "ready")
    calls = hold(INACTIVE)

    _snapshot(setup)

    assert len(calls) == 1


# Rule point 9: not active, or a raising call, gives today's sentence.
def test_rule9_live_work_reads_todays_sentence_when_the_hold_is_not_active(ctx, hold):
    setup = _project(ctx)
    _card(setup, "running")
    calls = hold(INACTIVE)

    assert _execution(setup) == {
        "state": "working", "summary": COORDINATING, "paused": False,
    }
    assert len(calls) == 1


def test_rule9_live_work_reads_todays_sentence_when_the_hold_call_raises(ctx, hold):
    setup = _project(ctx)
    _card(setup, "running")
    calls = hold(RuntimeError("the hold broke"))

    assert _execution(setup) == {
        "state": "working", "summary": COORDINATING, "paused": False,
    }
    assert len(calls) == 1


# Rule point 10: every other branch is unchanged and does not ask the hold.
def test_rule10_a_project_the_owner_paused_keeps_its_sentence_and_does_not_ask(ctx, hold):
    setup = _project(ctx)
    _card(setup, "ready")
    kanban_db.write_board_metadata(
        setup["board"], dispatch_enabled=False, dispatch_paused_by_owner=True,
    )
    calls = hold(ACTIVE)

    assert _execution(setup) == {"state": "paused", "summary": PAUSED, "paused": True}
    assert calls == []


def test_rule10_a_project_waiting_for_the_owners_answer_keeps_its_sentence_and_does_not_ask(
    ctx, hold,
):
    setup = _project(ctx)
    _card(setup, "ready")
    _card(setup, "blocked", block_kind="needs_input")
    calls = hold(ACTIVE)

    assert _execution(setup) == {
        "state": "waiting_for_you", "summary": NEEDS_ANSWER, "paused": False,
    }
    assert calls == []


def test_rule10_a_project_with_a_problem_keeps_its_sentence_and_does_not_ask(ctx, hold):
    setup = _project(ctx)
    _card(setup, "ready")
    _card(setup, "blocked")
    calls = hold(ACTIVE)

    assert _execution(setup) == {
        "state": "needs_attention", "summary": PROBLEM, "paused": False,
    }
    assert calls == []


def test_rule10_a_complete_project_keeps_its_sentence_and_does_not_ask(ctx, hold):
    setup = _project(ctx)
    _card(setup, "done", completed_at=int(time.time()))
    calls = hold(ACTIVE)

    assert _execution(setup) == {"state": "complete", "summary": COMPLETE, "paused": False}
    assert calls == []


def test_rule10_a_project_with_dormant_work_only_keeps_its_sentence_and_does_not_ask(
    ctx, hold,
):
    setup = _project(ctx)
    _card(setup, kanban_db.PARKED_STATUS)
    calls = hold(ACTIVE)

    assert _execution(setup) == {
        "state": "waiting_for_approval", "summary": WAITING_FOR_APPROVAL, "paused": False,
    }
    assert calls == []


def test_rule10_a_project_with_no_item_keeps_its_sentence_and_does_not_ask(ctx, hold):
    setup = _project(ctx)
    calls = hold(ACTIVE)

    assert _execution(setup) == {
        "state": "waiting_for_approval", "summary": WAITING_FOR_APPROVAL, "paused": False,
    }
    assert calls == []


# Rule point 11: no new state value, no new key, no change of another key.
def test_rule11_the_snapshot_keys_and_values_stay_the_same_with_the_hold_active(ctx, hold):
    setup = _project(ctx)
    _card(setup, "running")
    hold(INACTIVE)
    without = _snapshot(setup)
    hold(ACTIVE)
    during = _snapshot(setup)

    assert set(during) == set(without)
    assert set(during["execution"]) == set(without["execution"]) == {
        "state", "summary", "paused",
    }
    assert during["execution"]["state"] == without["execution"]["state"] == "working"
    other = {key for key in during if key != "execution"}
    assert {key: during[key] for key in other if key != "generated_at"} == {
        key: without[key] for key in other if key != "generated_at"
    }
