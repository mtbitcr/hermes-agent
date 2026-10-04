"""The risk tier is stored with the card and read back with it.

P1 of the risk tier plan, at the kernel boundary: the one task writer
(``kanban_db.create_task``) validates the tier before it stores it, a card
without a recorded tier holds null and counts as tier 2 (owner answer 3), and
the board projection returns the tier wherever it returns the review
requirement.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb._INITIALIZED_PATHS.clear()
    kb.init_db()
    return home


def test_create_task_validates_the_tier(kanban_home):
    """0, 1 and 2 are stored; anything else is refused before any write."""
    with kb.connect() as conn:
        for tier in (0, 1, 2):
            task_id = kb.create_task(
                conn, title=f"tier {tier}", assignee="worker", risk_tier=tier,
            )
            stored = kb.get_task(conn, task_id).risk_tier
            assert stored == tier
            assert type(stored) is int
            row = conn.execute(
                "SELECT risk_tier FROM tasks WHERE id = ?", (task_id,),
            ).fetchone()
            assert row["risk_tier"] == tier

        for bad in (3, -1, True, "2"):
            with pytest.raises(ValueError, match="risk_tier"):
                kb.create_task(
                    conn, title=f"refused {bad!r}", assignee="worker",
                    risk_tier=bad,
                )

        # None stays None: legacy and CLI cards record no tier.
        omitted = kb.create_task(conn, title="cli card", assignee="worker")
        explicit = kb.create_task(
            conn, title="legacy card", assignee="worker", risk_tier=None,
        )
        assert kb.get_task(conn, omitted).risk_tier is None
        assert kb.get_task(conn, explicit).risk_tier is None

        titles = {task.title for task in kb.list_tasks(conn)}
    assert not any(title.startswith("refused ") for title in titles), (
        "a refused tier must not leave a card behind"
    )


def test_a_card_without_a_tier_counts_as_tier_2(kanban_home):
    """Owner answer 3: no recorded tier reads 'Risk not recorded', counts as
    tier 2 and is treated as high risk."""
    from hermes_cli.kanban_risk_tier import RISK_NOT_RECORDED, effective_risk_tier

    assert effective_risk_tier(None) == 2
    for tier in (0, 1, 2):
        assert effective_risk_tier(tier) == tier
    assert RISK_NOT_RECORDED == "Risk not recorded. Treated as high risk."

    # A card from before the tier existed holds null and counts as tier 2.
    with kb.connect() as conn:
        task_id = kb.create_task(
            conn, title="Card from before the release", assignee="worker",
        )
        task = kb.get_task(conn, task_id)
    assert task.risk_tier is None
    assert effective_risk_tier(task.risk_tier) == 2


def test_the_board_projection_carries_the_tier(kanban_home):
    # ``plugins.kanban.dashboard.plugin_api`` declares multipart upload
    # endpoints, so importing it needs ``python-multipart`` (a real project
    # dependency). Skip rather than fail where it is absent.
    pytest.importorskip(
        "multipart", reason="python-multipart is required to import plugin_api",
    )
    from plugins.kanban.dashboard import plugin_api

    with kb.connect() as conn:
        tiered = kb.create_task(
            conn, title="tiered card", assignee="worker", risk_tier=1,
        )
        legacy = kb.create_task(conn, title="legacy card", assignee="worker")
        tiered_dict = plugin_api._task_dict(kb.get_task(conn, tiered))
        legacy_dict = plugin_api._task_dict(kb.get_task(conn, legacy))

    assert tiered_dict["risk_tier"] == 1
    # Returned next to the review requirement, exactly like it.
    assert "requires_review" in tiered_dict
    assert legacy_dict["risk_tier"] is None
