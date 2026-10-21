"""Card C5 (finding 144): a delivery row past open whose pull request GitHub shows merged or closed outside the
delivery step is closed outside, with the same compare-and-swap on the row's own read state, head and pull
request that closes an open row (tests/hermes_cli/test_kanban_delivery_review_cards.py)."""

from __future__ import annotations

import pytest

from tests.hermes_cli.test_kanban_delivery import board  # noqa: F401  (board is a fixture)
from tests.hermes_cli.test_kanban_delivery_auto_merge import (  # noqa: F401  (armed is a fixture)
    ARMED, OTHER, _arms, _done, _lenses, _reviews, armed,
)
from tests.hermes_cli.test_kanban_delivery_github import REPO
from tests.hermes_cli.test_kanban_delivery_publish_step import _events, _tick
from tests.hermes_cli.test_kanban_delivery_review_cards import (  # noqa: F401  (world is a fixture)
    _cards, _raw, _ready, _state, world,
)

PULL = ("GET", f"/repos/{REPO}/pulls/41")
ROWS = "SELECT * FROM kanban_deliveries ORDER BY id"
EVENTS = "SELECT * FROM task_events WHERE task_id = ? ORDER BY id"


def _reviewed(armed):
    """A published head whose review cards are made, done and approved on H by the reviewer bot."""
    kb, root, repo, gh = armed
    tid, head = _ready(armed)
    _tick(kb)
    _done(kb.kanban_db_path())
    _reviews(gh, *_lenses(kb, head))
    return tid, head


def _closed_outside(head, merged):
    return [("delivery_closed_outside", {"delivery_id": 1, "head": head, "pull_request_number": 41,
                                         "merged": merged})]


def _settled(armed, tid, head, events):
    """Later passes read that pull request no more and record nothing more."""
    kb, root, repo, gh = armed
    calls = len(gh["calls"])
    _tick(kb)
    _tick(kb)
    assert (gh["calls"][calls:], _state(kb, head), len(_events(kb, tid))) == ([], "closed_outside", events)


def test_a_pull_request_merged_outside_after_review_cards_closes_the_row_outside(armed):
    """Someone merges the pull request of H while its row waits with review cards created: the arm step's read
    finds it closed, marks the row closed outside with one event naming the pull request, its head and that
    GitHub shows it merged, and sends nothing."""
    kb, root, repo, gh = armed
    tid, head = _reviewed(armed)
    assert (len(_cards(kb)), _state(kb, head)) == (1, "review_cards_created")
    gh["ended"][41] = ARMED["sha"]  # merged into main outside the delivery step
    events = _events(kb, tid)

    _tick(kb)

    assert (_arms(gh), _state(kb, head), _events(kb, tid)[len(events):]) == ([], "closed_outside",
                                                                            _closed_outside(head, True))
    assert gh["calls"][-1] == PULL
    _settled(armed, tid, head, len(events) + 1)


def _left_ready(armed):
    """A published head whose required review card is made and left ready: its review is unfinished."""
    kb, root, repo, gh = armed
    tid, head = _ready(armed)
    _tick(kb)
    assert ([card["status"] for card in _cards(kb)], _state(kb, head)) == (["ready"], "review_cards_created")
    return tid, head


def test_a_pull_request_merged_outside_while_a_review_card_is_unfinished_closes_the_row_outside(armed):
    """Someone merges the pull request of H while its required review card is still unfinished: the arm step reads
    the pull request before it waits for that card, so the next pass marks the row closed outside, merged, with one
    event. While the pull request was open at H that read was all: no review was read and nothing was sent."""
    kb, root, repo, gh = armed
    tid, head = _left_ready(armed)
    calls, events = len(gh["calls"]), _events(kb, tid)
    _tick(kb)  # open at H: read, then the arm waits for the card
    gh["ended"][41] = ARMED["sha"]  # merged into main outside the delivery step

    _tick(kb)

    assert (_state(kb, head), _events(kb, tid)[len(events):], gh["calls"][calls:], gh["writes"]) == (
        "closed_outside", _closed_outside(head, True), [PULL, PULL], [])
    assert [card["status"] for card in _cards(kb)] == ["ready"]
    _settled(armed, tid, head, len(events) + 1)


def test_a_pull_request_closed_without_merge_while_a_review_card_is_unfinished_closes_the_row_outside(armed):
    """Someone closes the pull request of H unmerged while its required review card is still unfinished: the next
    pass marks the row closed outside, not merged, with one event, and reads no review and sends nothing."""
    kb, root, repo, gh = armed
    tid, head = _left_ready(armed)
    calls, events = len(gh["calls"]), _events(kb, tid)
    _tick(kb)  # open at H: read, then the arm waits for the card
    gh["ended"][41] = False  # closed unmerged outside the delivery step

    _tick(kb)

    assert (_state(kb, head), _events(kb, tid)[len(events):], gh["calls"][calls:], gh["writes"]) == (
        "closed_outside", _closed_outside(head, False), [PULL, PULL], [])
    assert [card["status"] for card in _cards(kb)] == ["ready"]
    _settled(armed, tid, head, len(events) + 1)


@pytest.mark.parametrize("answer, state", [((200, ARMED), "auto_merge_armed"), ("timeout", "auto_merge_unknown")],
                         ids=["armed", "unknown"])
def test_an_armed_pull_request_closed_without_merge_closes_the_row_outside(armed, answer, state):
    """An armed or unknown arm's pull request that someone closes unmerged is no merge to wait for: the merge
    step's read marks the row closed outside, not merged, and the release record never holds it."""
    from hermes_cli import release_ledger

    kb, root, repo, gh = armed
    tid, head = _reviewed(armed)
    gh["arm"] = answer
    _tick(kb)
    assert (len(_arms(gh)), _state(kb, head)) == (1, state)
    gh["ended"][41] = False  # closed unmerged outside the delivery step
    events = _events(kb, tid)

    _tick(kb)

    assert (_state(kb, head), _events(kb, tid)[len(events):]) == ("closed_outside", _closed_outside(head, False))
    assert len(_arms(gh)) == 1
    _settled(armed, tid, head, len(events) + 1)
    conn = release_ledger.connect()
    try:
        assert [member for batch in release_ledger.list_batches(conn) for member in batch["members"]] == []
    finally:
        conn.close()


def test_a_refused_arm_whose_pull_request_was_merged_outside_closes_the_row_outside(armed):
    """GitHub refused the arm of H, and someone then merged the pull request outside the delivery step: the
    merge step's read of the refused row marks it closed outside, merged, with nothing sent."""
    kb, root, repo, gh = armed
    tid, head = _reviewed(armed)
    gh["arm"] = (405, {"message": "Pull Request is not mergeable"})
    _tick(kb)
    assert _state(kb, head) == "auto_merge_refused"
    _tick(kb)
    assert _state(kb, head) == "auto_merge_refused"  # still open on GitHub: it stays refused
    gh["ended"][41] = ARMED["sha"]
    events, writes = _events(kb, tid), list(gh["writes"])

    _tick(kb)

    assert (_state(kb, head), _events(kb, tid)[len(events):]) == ("closed_outside", _closed_outside(head, True))
    assert gh["writes"] == writes and gh["calls"][-1] == PULL
    _settled(armed, tid, head, len(events) + 1)


def test_a_base_moved_row_whose_pull_request_was_merged_outside_closes_the_row_outside(armed):
    """A row card 2 left base_moved, its pull request merged outside before any update card took it, is closed
    outside, merged, by the merge step's read; nothing is sent."""
    kb, root, repo, gh = armed
    db = kb.kanban_db_path()
    tid, head = _reviewed(armed)
    _raw(db, "UPDATE kanban_deliveries SET pull_request_state = 'base_moved' WHERE source_head = ?", head)
    _tick(kb)
    assert (_state(kb, head), _arms(gh)) == ("base_moved", [])  # open on GitHub: it stays base_moved
    gh["ended"][41] = ARMED["sha"]
    events = _events(kb, tid)

    _tick(kb)

    assert (_state(kb, head), _events(kb, tid)[len(events):]) == ("closed_outside", _closed_outside(head, True))
    assert (gh["writes"], gh["calls"][-1]) == ([], PULL)
    _settled(armed, tid, head, len(events) + 1)


@pytest.mark.parametrize("state", ["review_cards_created", "auto_merge_armed", "auto_merge_unknown",
                                   "auto_merge_refused", "close_refused", "base_moved"])
@pytest.mark.parametrize("column", ["pull_request_state", "pull_request_head", "pull_request_number"])
def test_a_row_changed_while_its_pull_request_is_read_is_left_untouched(armed, state, column):
    """The row moves on (a base_moved row as its update card takes it), or names another head or pull request,
    while GitHub's closed pull request is read: the one write transaction finds it changed, so nothing is
    written and no event is appended. A step that never reads the row leaves it as it is too."""
    kb, root, repo, gh = armed
    db = kb.kanban_db_path()
    tid, head = _reviewed(armed)
    _raw(db, "UPDATE kanban_deliveries SET pull_request_state = ? WHERE source_head = ?", state, head)
    value = {"pull_request_state": "updated" if state == "base_moved" else "close_pending",
             "pull_request_head": OTHER, "pull_request_number": 42}[column]
    gh["ended"][41] = False
    before, events, changed = _raw(db, ROWS), _raw(db, EVENTS, tid), []

    def change():  # once, during the pull request read
        _raw(db, f"UPDATE kanban_deliveries SET {column} = ? WHERE source_head = ?", value, head)
        changed.extend(_raw(db, ROWS))

    gh["hooks"]["PULL"] = change
    _tick(kb)

    assert (_raw(db, ROWS), _raw(db, EVENTS, tid), _arms(gh)) == (changed or before, events, [])
    assert "closed_outside" not in {row["pull_request_state"] for row in _raw(db, ROWS)}
