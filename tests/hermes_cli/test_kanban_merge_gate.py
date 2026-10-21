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


def _card(conn, title: str, paths: list[str], parents=()) -> str:
    return kb.create_task(
        conn, title=title, assignee="worker", workspace_kind="worktree", owned_paths=paths, parents=list(parents),
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
