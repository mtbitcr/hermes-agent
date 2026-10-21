"""Card D: a child that shares owned paths with its parent waits until the parent's change has merged.

With ``kanban.delivery.enabled`` true, a done parent satisfies a child whose owned paths overlap its own only once
the parent card, or the card that carries its change on, has a delivery row in state merged. That carrying card is
never held by its own parent; disjoint children, parents with no delivery row, and every child with delivery off
start at approval as before. Real boards in a temporary Hermes home; the parent's approval and delivery row are
written straight to the board.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc


def _delivery(home: Path, *, enabled: bool) -> None:
    (home / "config.yaml").write_text(
        f"kanban:\n  delivery:\n    enabled: {'true' if enabled else 'false'}\n", encoding="utf-8",
    )


@pytest.fixture
def board(tmp_path, monkeypatch):
    """An isolated Hermes home with an empty board and delivery on."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    _delivery(home, enabled=True)
    kb.init_db()
    return home


def _card(conn, title: str, paths: list[str], parents=(), **fields) -> str:
    return kb.create_task(
        conn, title=title, assignee="worker", workspace_kind="worktree", owned_paths=paths, parents=list(parents),
        **fields,
    )


def _approved(conn, card: str, state: str) -> None:
    """The card is done and its delivery row is in ``state``."""
    with kb.write_txn(conn):
        conn.execute("UPDATE tasks SET status = 'done' WHERE id = ?", (card,))
        conn.execute(
            "INSERT INTO kanban_deliveries (source_task_id, source_head, created_at, pull_request_state) "
            "VALUES (?, ?, ?, ?)", (card, f"head-{card}", int(time.time()), state),
        )


def _carried_on_by(conn, parent: str, update: str) -> None:
    """The parent's delivery_updated event names ``update`` as the card that carries its change on."""
    with kb.write_txn(conn):
        kb._append_event(conn, parent, "delivery_updated", {"update_task_id": update})


def _merges(conn, card: str) -> None:
    """The card's change merged: its delivery row is now in state merged."""
    with kb.write_txn(conn):
        conn.execute("UPDATE kanban_deliveries SET pull_request_state = 'merged' WHERE source_task_id = ?", (card,))


def _status(conn, card: str) -> str:
    return kb.get_task(conn, card).status


def test_a_child_sharing_a_file_waits_for_the_parents_merge(board):
    """Promotion, the landing re-gate and claim all hold it, also when another writer set it ready."""
    with kb.connect() as conn:
        parent = _card(conn, "parent", ["src/a.py", "src/b.py"])
        child = _card(conn, "child", ["src/b.py"], parents=[parent])
        _approved(conn, parent, "review_cards_created")

        kb.recompute_ready(conn)
        assert _status(conn, child) == "todo"
        assert kb._landing_status_after_parents(conn, child) == "todo"

        with kb.write_txn(conn):
            conn.execute("UPDATE tasks SET status = 'ready' WHERE id = ?", (child,))
        assert kb.claim_task(conn, child, claimer="worker") is None
        assert _status(conn, child) == "todo"


def test_the_child_starts_once_the_parents_change_merged(board):
    with kb.connect() as conn:
        parent = _card(conn, "parent", ["src/a.py"])
        child = _card(conn, "child", ["src/a.py"], parents=[parent])
        _approved(conn, parent, "review_cards_created")
        kb.recompute_ready(conn)
        assert _status(conn, child) == "todo"

        with kb.write_txn(conn):
            conn.execute("UPDATE kanban_deliveries SET pull_request_state = 'merged' WHERE source_task_id = ?",
                         (parent,))
        kb.recompute_ready(conn)
        assert _status(conn, child) == "ready"
        assert kb.claim_task(conn, child, claimer="worker") is not None


def test_a_child_with_disjoint_files_starts_at_approval(board):
    with kb.connect() as conn:
        parent = _card(conn, "parent", ["src/a.py"])
        child = _card(conn, "child", ["src/b.py"], parents=[parent])
        _approved(conn, parent, "review_cards_created")

        kb.recompute_ready(conn)
        assert _status(conn, child) == "ready"
        assert kb.claim_task(conn, child, claimer="worker") is not None


def test_the_update_card_of_a_parent_is_not_held_by_its_own_parent(board):
    """The parent's change merges only through this card, so holding it would wait for ever."""
    with kb.connect() as conn:
        parent = _card(conn, "parent", ["src/a.py"])
        update = _card(conn, "update", ["src/a.py"], parents=[parent])
        _approved(conn, parent, "updated")
        _carried_on_by(conn, parent, update)

        kb.recompute_ready(conn)
        assert _status(conn, update) == "ready"
        assert kb.claim_task(conn, update, claimer="worker") is not None


def test_a_change_merged_through_its_update_card_releases_the_child(board):
    with kb.connect() as conn:
        parent = _card(conn, "parent", ["src/a.py"])
        update = _card(conn, "update", ["src/a.py"], parents=[parent])
        sibling = _card(conn, "sibling", ["src/a.py"], parents=[parent])
        _approved(conn, parent, "updated")
        _carried_on_by(conn, parent, update)
        kb.recompute_ready(conn)
        assert (_status(conn, update), _status(conn, sibling)) == ("ready", "todo")

        _approved(conn, update, "merged")
        kb.recompute_ready(conn)
        assert _status(conn, sibling) == "ready"


def test_with_delivery_off_children_start_at_approval(board):
    _delivery(board, enabled=False)
    with kb.connect() as conn:
        parent = _card(conn, "parent", ["src/a.py"])
        child = _card(conn, "child", ["src/a.py"], parents=[parent])
        _approved(conn, parent, "review_cards_created")

        kb.recompute_ready(conn)
        assert _status(conn, child) == "ready"
        assert kb.claim_task(conn, child, claimer="worker") is not None


@pytest.mark.parametrize("enabled", [False, True], ids=["delivery-off", "delivery-on"])
def test_a_board_from_the_existing_connector_starts_children_at_approval(tmp_path, monkeypatch, enabled):
    """Finding 1 of 1518: kanban_db_connect.connect, also the standalone dispatch entry point's connector, initializes
    a board without the kanban_deliveries ledger. There a done parent satisfies its ordinary child at every gate, with
    delivery off and on, as before: the switch is read before the ledger, an absent ledger is no delivery row, and
    the gate creates no ledger."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    _delivery(home, enabled=enabled)
    ledger = "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'kanban_deliveries'"
    conn = kbc.connect(tmp_path / "kanban.db")
    try:
        assert conn.execute(ledger).fetchone() is None
        parent = kb.create_task(conn, title="parent", assignee="planner")
        child = kb.create_task(conn, title="child", assignee="builder", parents=[parent])
        assert kb.complete_task(conn, parent)
        assert kb._parents_satisfied(conn, child)
        assert kb._landing_status_after_parents(conn, child) == "ready"
        kb.recompute_ready(conn)
        assert _status(conn, child) == "ready"
        assert kb.claim_task(conn, child) is not None

        late = kb.create_task(conn, title="late", assignee="builder", parents=[parent])
        assert _status(conn, late) == "ready"
        assert kb.complete_task(conn, late)
        parked = kb.create_task(conn, title="parked", assignee="builder", parents=[parent], initial_status="blocked")
        assert kb.promote_task(conn, parked, actor="owner", dry_run=True) == (True, None)
        assert kb.promote_task(conn, parked, actor="owner") == (True, None)
        assert conn.execute(ledger).fetchone() is None
    finally:
        conn.close()


def test_manual_promotion_holds_a_shared_path_child_as_claim_does(board):
    """Finding 2 of 1518: a non-forced manual promotion, and its dry run, refuse the child while the parent's change
    has not merged, as claim does; once it merged both promote it. A forced promotion overrides, as before."""
    with kb.connect() as conn:
        parent = _card(conn, "parent", ["src/a.py"])
        child = _card(conn, "child", ["src/a.py"], parents=[parent])
        forced = _card(conn, "forced", ["src/a.py"], parents=[parent])
        _approved(conn, parent, "review_cards_created")

        refused = (False, f"unsatisfied parent dependencies: {parent} (use --force to override)")
        assert kb.promote_task(conn, child, actor="owner", dry_run=True) == refused
        assert kb.promote_task(conn, child, actor="owner") == refused
        assert kb.promote_task(conn, forced, actor="owner", force=True) == (True, None)
        assert (_status(conn, child), _status(conn, forced)) == ("todo", "ready")

        _merges(conn, parent)
        assert kb.promote_task(conn, child, actor="owner", dry_run=True) == (True, None)
        assert kb.promote_task(conn, child, actor="owner") == (True, None)
        assert _status(conn, child) == "ready"


def test_a_shared_path_child_created_after_its_parents_approval_waits_for_the_merge(board):
    """Finding 2 of 1518: creation decides ready with the same predicate once the child's owned paths and parent
    links exist, so a late shared-path child is created todo and stays there until the parent's change merged."""
    with kb.connect() as conn:
        parent = _card(conn, "parent", ["src/a.py"])
        _approved(conn, parent, "review_cards_created")
        child = _card(conn, "child", ["src/a.py"], parents=[parent])
        disjoint = _card(conn, "disjoint", ["src/b.py"], parents=[parent])
        assert (_status(conn, child), _status(conn, disjoint)) == ("todo", "ready")
        assert [event.payload["status"] for event in kb.list_events(conn, child) if event.kind == "created"] == [
            "todo"]
        kb.recompute_ready(conn)
        assert _status(conn, child) == "todo"

        _merges(conn, parent)
        kb.recompute_ready(conn)
        assert _status(conn, child) == "ready"


def test_a_card_composed_and_named_as_the_carrier_in_one_transaction_is_created_ready(board):
    """Guard for finding 2 of 1518: the kernel makes the card that carries a parent's change on under its own
    transaction and names it only after making it. Created todo, as no record names it yet, the card is released by
    the record that names it and promoted as recompute_ready promotes, so it is ready at the commit."""
    with kb.connect() as conn:
        parent = _card(conn, "parent", ["src/a.py"])
        _approved(conn, parent, "updated")
        with kb.write_txn(conn):
            update = _card(conn, "update", ["src/a.py"], parents=[parent])
            kb._append_event(conn, parent, "delivery_updated", {"update_task_id": update})

        assert _status(conn, update) == "ready"
        assert _created_status(conn, update) == ["todo"]
        assert [event.kind for event in kb.list_events(conn, update)] == ["created", "promoted"]
        assert kb.claim_task(conn, update, claimer="worker") is not None


def test_with_delivery_off_promotion_and_creation_start_at_approval(board):
    """Guard for finding 2 of 1518: with delivery off, manual promotion and creation are as before."""
    _delivery(board, enabled=False)
    with kb.connect() as conn:
        parent = _card(conn, "parent", ["src/a.py"])
        child = _card(conn, "child", ["src/a.py"], parents=[parent])
        _approved(conn, parent, "review_cards_created")

        assert kb.promote_task(conn, child, actor="owner", dry_run=True) == (True, None)
        assert kb.promote_task(conn, child, actor="owner") == (True, None)
        late = _card(conn, "late", ["src/a.py"], parents=[parent])
        assert (_status(conn, child), _status(conn, late)) == ("ready", "ready")


def _created_status(conn, card: str) -> list:
    return [event.payload["status"] for event in kb.list_events(conn, card) if event.kind == "created"]


def test_a_shared_path_child_created_inside_a_callers_transaction_waits_for_the_merge(board):
    """The finding of 1526: an ordinary child that shares a file, created inside a caller's transaction while the
    parent's change has not merged, is no carrier: no record names it. After the outer commit its row and its created
    event say todo, recompute_ready keeps it there and claim refuses it."""
    with kb.connect() as conn:
        parent = _card(conn, "parent", ["src/a.py"])
        _approved(conn, parent, "review_cards_created")
        with kb.write_txn(conn):
            child = _card(conn, "child", ["src/a.py"], parents=[parent])

        assert (_status(conn, child), _created_status(conn, child)) == ("todo", ["todo"])
        kb.recompute_ready(conn)
        assert _status(conn, child) == "todo"
        assert kb.claim_task(conn, child, claimer="worker") is None
        assert _status(conn, child) == "todo"


@pytest.mark.parametrize("control", ["delivery-off", "disjoint", "merged"])
def test_a_child_created_inside_a_callers_transaction_starts_at_approval_when_nothing_holds_it(board, control):
    """Controls for the finding of 1526: with delivery off, with disjoint files, or once the parent's change merged, a
    child created inside a caller's transaction is created ready, as before, and claim takes it."""
    if control == "delivery-off":
        _delivery(board, enabled=False)
    with kb.connect() as conn:
        parent = _card(conn, "parent", ["src/a.py"])
        _approved(conn, parent, "merged" if control == "merged" else "review_cards_created")
        with kb.write_txn(conn):
            child = _card(conn, "child", ["src/b.py" if control == "disjoint" else "src/a.py"], parents=[parent])

        assert (_status(conn, child), _created_status(conn, child)) == ("ready", ["ready"])
        assert kb.claim_task(conn, child, claimer="worker") is not None


def test_a_named_carrier_parked_blocked_held_by_another_open_parent_or_in_triage_keeps_that_state(board):
    """Guard for owner rule 3 of round 1: the record that names a carrier releases it only from its own parent's
    merge wait. A carrier parked for its route, created blocked, held by another open parent or in triage keeps that
    state, and no promotion is recorded."""
    with kb.connect() as conn:
        parent = _card(conn, "parent", ["src/a.py"])
        other = _card(conn, "other", ["src/z.py"])
        _approved(conn, parent, "updated")
        with kb.write_txn(conn):
            parked = _card(conn, "parked", ["src/a.py"], parents=[parent])
            kb._pause_unpinnable_task(conn, parked, "no admitted route")
            blocked = _card(conn, "blocked", ["src/a.py"], parents=[parent], initial_status="blocked")
            waiting = _card(conn, "waiting", ["src/a.py"], parents=[parent, other])
            triaged = _card(conn, "triaged", ["src/a.py"], parents=[parent], triage=True)
            cards = (parked, blocked, waiting, triaged)
            for card in cards:
                kb._append_event(conn, parent, "review_followup_recorded", {"followup_task_id": card})

        assert [_status(conn, card) for card in cards] == ["blocked", "blocked", "todo", "triage"]
        assert [event for card in cards for event in kb.list_events(conn, card) if event.kind == "promoted"] == []
