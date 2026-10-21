"""Card 6 (finding 153): a delivery refused with source_changed is published again at the next
dispatcher pass, not once its lease runs out ten minutes later.

``publish_delivery`` refuses with source_changed when any event, the release of the build machine
among them, was appended to the source card after the approval was proven. The publish step then
frees the lease of the pass that was refused, in one write transaction and only while that pass
still holds it. Every other outcome keeps its lease as before. Real boards, repositories, remotes
and the GitHub stand-in of the publish step tests.
"""

from __future__ import annotations

import time
from types import SimpleNamespace

import pytest

from tests.hermes_cli.test_kanban_delivery import _sql, board  # noqa: F401  (board is a fixture)
from tests.hermes_cli.test_kanban_delivery_publish_step import (  # noqa: F401  (github and world are fixtures)
    PULLS, UNSTORED, _approved, _events, _expire, _row, _take_over, _tick, github, world,
)


@pytest.fixture
def passes(world, monkeypatch):
    """What ``publish_delivery`` did in each pass, in order: ``(delivery_id, holder, outcome)``, the
    outcome being the refusal code or the state returned. ``between``, when set, runs once a refusal
    is raised, before the publish step acts on it."""
    from hermes_cli import kanban_delivery as kd

    seen, publish = SimpleNamespace(outcomes=[], between=None), kd.publish_delivery

    def recorded(db_path, delivery_id, holder):
        try:
            result = publish(db_path, delivery_id, holder)
        except kd.PublishRefused as refusal:
            seen.outcomes.append((delivery_id, holder, refusal.code))
            if seen.between is not None:
                seen.between()
            raise
        seen.outcomes.append((delivery_id, holder, result["state"]))
        return result

    monkeypatch.setattr(kd, "publish_delivery", recorded)
    return seen


def _source_event(kb, tid: str) -> None:
    """Append one event to the source card, as the release of its build machine does. Its kind is
    one nothing on the board acts on: only the card's latest event revision moves."""
    with kb.connect_closing(db_path=kb.kanban_db_path()) as conn, kb.write_txn(conn):
        kb._append_event(conn, tid, "note", {"by": "test"})


def _creations(github) -> list:
    return [call for call in github["calls"] if call[:2] == ("POST", PULLS)]


def test_a_source_event_between_the_proof_and_the_store_frees_the_lease_at_once(world, github, passes):
    kb, _, repo, _ = world
    tid, head = _approved(kb, repo)
    events = _events(kb, tid)
    github["on_create"] = lambda: _source_event(kb, tid)
    _tick(kb)
    assert [outcome for _, _, outcome in passes.outcomes] == ["source_changed"]
    assert len(github["pulls"]) == 1 and len(_creations(github)) == 1
    # Nothing is stored, recorded or parked, as before; the lease is free at once.
    assert _row(kb, head) == UNSTORED + (None, None, None)
    assert _events(kb, tid) == events


def test_the_very_next_pass_adopts_the_same_pull_request_without_a_second_one(world, github, passes):
    kb, _, repo, _ = world
    tid, head = _approved(kb, repo)
    github["on_create"] = lambda: _source_event(kb, tid)
    _tick(kb)
    github["on_create"] = None
    _tick(kb)  # the next dispatcher pass, long before the lease would have run out
    assert [outcome for _, _, outcome in passes.outcomes] == ["source_changed", "created"]
    (refused, refused_holder, _), (published, holder, _) = passes.outcomes
    assert published == refused and holder != refused_holder
    assert len(github["pulls"]) == 1 and len(_creations(github)) == 1
    assert _row(kb, head) == (41, head, "open", github["pulls"][0]["ref"], None, None, None)
    assert [(payload["pull_request_number"], payload["state"]) for kind, payload in _events(kb, tid)
            if kind == "delivery_published"] == [(41, "adopted")]


def test_a_lease_taken_over_before_the_release_is_not_cleared(world, github, passes):
    kb, _, repo, _ = world
    tid, head = _approved(kb, repo)
    github["on_create"] = lambda: _source_event(kb, tid)
    taken, rows = [], []

    def take_over():
        _take_over(kb, taken)
        rows.append(_row(kb, head))

    passes.between = take_over
    _tick(kb)
    [(delivery_id, holder, refused)] = passes.outcomes
    assert refused == "source_changed" and taken[0][0] == delivery_id and taken[0][1] != holder
    assert rows[0] == UNSTORED + (taken[0][1], rows[0][5], None) and rows[0][5] > time.time()
    assert _row(kb, head) == rows[0]


@pytest.mark.parametrize("case", ["ran-out", "cleared"])
def test_a_lease_already_gone_before_the_release_is_left_untouched(world, github, passes, case):
    kb, _, repo, _ = world
    tid, head = _approved(kb, repo)
    github["on_create"] = lambda: _source_event(kb, tid)
    rows = []

    def gone():
        if case == "ran-out":
            _expire(kb)
        else:
            _sql(kb, "UPDATE kanban_deliveries SET publish_lease = NULL, publish_lease_until = NULL "
                     "WHERE source_head = ?", head)
        rows.append(_row(kb, head))

    passes.between = gone
    _tick(kb)
    [(_, holder, refused)] = passes.outcomes
    assert refused == "source_changed"
    if case == "ran-out":
        assert rows[0][4] == holder and rows[0][5] <= time.time()
    else:
        assert rows[0][4:6] == (None, None)
    assert _row(kb, head) == rows[0]


def test_transport_failed_keeps_the_lease_until_it_runs_out(world, github, passes):
    from hermes_cli import kanban_delivery as kd

    kb, _, repo, _ = world
    _, head = _approved(kb, repo)
    github["fail"][("GET", PULLS)] = (None, None)  # the listing's connection drops without an answer
    started = int(time.time())
    _tick(kb)
    [(_, holder, refused)] = passes.outcomes
    row = _row(kb, head)
    assert refused == "transport_failed"
    assert row == UNSTORED + (holder, row[5], None) and row[5] >= started + kd._LEASE_SECONDS
    _tick(kb)  # the lease holds, so the next pass leaves the row alone, as before
    assert len(passes.outcomes) == 1 and _row(kb, head) == row


@pytest.mark.parametrize("case", ["taken-over", "ran-out"])
def test_lease_lost_leaves_the_lease_as_it_was_found(world, github, passes, case):
    kb, _, repo, _ = world
    tid, head = _approved(kb, repo)
    taken, rows = [], []

    def lose_the_lease():
        _source_event(kb, tid)  # the source changes as well; the lost lease is refused first
        if case == "taken-over":
            _take_over(kb, taken)
        else:
            _expire(kb)
        rows.append(_row(kb, head))

    github["on_create"] = lose_the_lease
    _tick(kb)
    [(_, holder, refused)] = passes.outcomes
    assert refused == "lease_lost"
    assert rows[0][4] == (taken[0][1] if case == "taken-over" else holder)
    assert _row(kb, head) == rows[0]


def test_a_clean_publish_stores_the_pull_request_and_frees_the_lease_as_before(world, github, passes):
    kb, _, repo, _ = world
    tid, head = _approved(kb, repo)
    _tick(kb)
    assert [outcome for _, _, outcome in passes.outcomes] == ["created"]
    assert len(_creations(github)) == 1
    assert _row(kb, head) == (41, head, "open", github["pulls"][0]["ref"], None, None, None)
    assert [payload["state"] for kind, payload in _events(kb, tid)
            if kind == "delivery_published"] == ["created"]
