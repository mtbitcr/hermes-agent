"""Finding 183: when the owner replaces a delivered card, the next arm step closes its pull request once, as it
closes one a rework card continues, and the publish step never publishes the replaced card's work. The replaced
source is what an owner replace leaves: the card archived, with an owner_project_plan_change event naming its
replacement_task_id, and no task link between the two cards. Real boards and git; GitHub is the real transport
with its exchange and App token replaced by a table, or for the publish a local stand-in over a bare remote, so
every call passes the allowlist."""

from __future__ import annotations

import pytest

from tests.hermes_cli.test_kanban_delivery import board  # noqa: F401  (board is a fixture)
from tests.hermes_cli.test_kanban_delivery_auto_merge import _published_as, armed  # noqa: F401  (armed is a fixture)
from tests.hermes_cli.test_kanban_delivery_github import REPO, _remote_refs
from tests.hermes_cli.test_kanban_delivery_publish_step import (  # noqa: F401  (github and publishing are fixtures)
    UNSTORED, _approved, _events, _row, _tick, github, world as publishing,
)
from tests.hermes_cli.test_kanban_delivery_review_cards import (  # noqa: F401  (world is a fixture)
    _raw, _ready, _state, world,
)

CLOSE = ("PATCH", f"/repos/{REPO}/pulls/41", {"state": "closed"})


def _replace(kb, tid):
    """The owner replaces card ``tid`` as apply_owner_project_plan leaves it: a new card, and ``tid`` archived
    with an owner_project_plan_change event naming that card as its replacement_task_id; no task link joins
    the two. Returns the replacement."""
    conn = kb.connect()
    try:
        replacement = kb.create_task(conn, title="replacement")
        assert kb.cas_transition_task(
            conn, tid, expected_status=kb.get_task(conn, tid).status,
            expected_revision=kb.task_event_revision(conn, tid), to_status="archived",
            event_kind="owner_project_plan_change", event_payload={
                "actor": "owner", "profile": "owner", "idempotency_key": "plan-183", "request_digest": "d" * 64,
                "action": "replace", "reason": "a better plan", "replacement_task_id": replacement},
        )["moved"] is True
    finally:
        conn.close()
    assert _raw(kb.kanban_db_path(), "SELECT 1 FROM task_links WHERE parent_id IN (?, ?) AND child_id IN (?, ?)",
                tid, replacement, tid, replacement) == []
    return replacement


def _closes(gh):
    return [write for write in gh["writes"] if write[0] == "PATCH"]


def _outcomes(kb, tid):
    """The close events of ``tid``: kind, pull request, the card named as replacing it, and GitHub's status."""
    return [(kind, p["pull_request_number"], p["replaced_by"], p.get("status")) for kind, p in _events(kb, tid)
            if kind.startswith("delivery_close") or kind == "delivery_replaced"]


@pytest.mark.parametrize("state", ["open", "review_cards_created"])
def test_the_pull_request_of_a_card_the_owner_replaced_is_closed_once_at_the_next_arm_step(armed, state):
    """The published row of a card the owner replaced, before its review or in it, gets exactly one close at
    the next arm step, the PATCH that sets its pull request closed. The row then reads replaced, and a second
    arm step sends nothing."""
    from hermes_cli.kanban_delivery import arm_step

    kb, root, repo, gh = armed
    db = kb.kanban_db_path()
    tid, head = _ready(armed)
    if state == "review_cards_created":
        _tick(kb)  # the review step makes H's review cards
    assert _state(kb, head) == state
    replacement = _replace(kb, tid)

    arm_step(db)

    assert _closes(gh) == [CLOSE]
    assert _state(kb, head) == "replaced"
    assert _outcomes(kb, tid) == [("delivery_close_pending", 41, replacement, None),
                                  ("delivery_replaced", 41, replacement, 200)]
    calls, events = list(gh["calls"]), _events(kb, tid)

    arm_step(db)

    assert (gh["calls"], _events(kb, tid), _state(kb, head)) == (calls, events, "replaced")


def test_a_refused_close_of_the_pull_request_of_a_replaced_card_reads_close_refused(armed):
    """GitHub refuses the close: the row reads close_refused, with GitHub's status, as a continuation's refused
    close does, and the close is not sent again."""
    from hermes_cli.kanban_delivery import arm_step

    kb, root, repo, gh = armed
    tid, head = _ready(armed)
    replacement = _replace(kb, tid)
    gh["close"] = (403, None)

    arm_step(kb.kanban_db_path())
    arm_step(kb.kanban_db_path())

    assert _closes(gh) == [CLOSE]
    assert _state(kb, head) == "close_refused"
    assert _outcomes(kb, tid) == [("delivery_close_pending", 41, replacement, None),
                                  ("delivery_close_refused", 41, replacement, 403)]


def test_an_unpublished_row_of_a_replaced_card_is_parked_with_source_replaced_and_nothing_is_pushed(
        publishing, github):
    """The approved head of a card the owner replaced, not yet published, is parked with source_replaced,
    recorded once: nothing is pushed, no pull request is made, and the next pass does not try it again."""
    kb, root, repo, bare = publishing
    tid, head = _approved(kb, repo)
    _replace(kb, tid)

    _tick(kb)
    _tick(kb)

    assert (github["pushes"], _remote_refs(bare)) == ([], [])
    assert [call for call in github["calls"] if call[0] == "POST"] == []
    assert _row(kb, head) == (*UNSTORED, None, None, "source_replaced")
    assert [(kind, p.get("code"), p["head"]) for kind, p in _events(kb, tid)
            if kind in ("delivery_refused", "delivery_published")] == [("delivery_refused", "source_replaced", head)]


def test_the_pull_request_of_a_card_archived_without_a_replace_event_is_not_closed(armed):
    """The kernel's plain archive names no replacement, so its card's open pull request gets no close, while
    in the same arm step the pull request of a card the owner replaced is closed."""
    from hermes_cli.kanban_delivery import arm_step

    kb, root, repo, gh = armed
    db = kb.kanban_db_path()
    tid, head = _ready(armed)
    archived, kept = _approved(kb, repo, "archived")
    _published_as(db, gh, archived, kept, 43)  # GitHub shows it open at its head
    _replace(kb, tid)
    conn = kb.connect()
    try:
        assert kb.archive_task(conn, archived) is True
    finally:
        conn.close()
    assert _raw(db, "SELECT kind FROM task_events WHERE task_id = ? AND kind = 'owner_project_plan_change'",
                archived) == []

    arm_step(db)
    arm_step(db)

    assert _closes(gh) == [CLOSE]
    assert (_state(kb, head), _state(kb, kept)) == ("replaced", "open")
    assert _outcomes(kb, archived) == []


def test_a_merged_row_of_a_replaced_card_is_never_closed(armed):
    """A merged row is final: though the owner replaced its card too and GitHub still shows its pull request
    open, no close is sent for it, while in the same arm step the pull request of a replaced card is closed."""
    from hermes_cli.kanban_delivery import arm_step

    kb, root, repo, gh = armed
    db = kb.kanban_db_path()
    tid, head = _ready(armed)
    merged, landed = _approved(kb, repo, "merged")
    _published_as(db, gh, merged, landed, 43)  # GitHub shows it open at its head
    _raw(db, "UPDATE kanban_deliveries SET pull_request_state = 'merged' WHERE source_task_id = ?", merged)
    _replace(kb, tid)
    _replace(kb, merged)

    arm_step(db)
    arm_step(db)

    assert _closes(gh) == [CLOSE]
    assert (_state(kb, head), _state(kb, landed)) == ("replaced", "merged")
    assert _outcomes(kb, merged) == []
