"""A review rework item records the risk tier of the card it tracks.

A changes-requested verdict opens ONE rework item for the implementer, linked
under the reviewed card (``_request_changes_within_txn``). The item used to be
created without a tier, so it held null, counted as tier 2 and read "Risk not
recorded. Treated as high risk." -- on the board, and again among the
completed rework once the card was approved -- whatever the card's own tier.

The item now records the reviewed card's own recorded tier, read in the
handback's own transaction after any raise the verdict records, so a
reviewer's raise to tier 2 reaches it. A card without a tier gives an item
without one: no tier is ever invented. The tier is all the item takes: it gets
no route lock, no effort pin and no time box, and an item recorded before is
never rewritten.
"""

from __future__ import annotations

import json
import os
import sqlite3
from contextlib import closing
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli.kanban_risk_tier import effective_risk_tier
from plugins.dashboard_auth.raphael_workspace import model_policy as mp


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    """Isolated HERMES_HOME with an empty kanban DB."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb._INITIALIZED_PATHS.clear()
    kb.init_db()
    return home


def _vouched_followups(conn, task_id):
    """The items the KERNEL vouched for on this card, read from raw events."""
    ids = []
    for row in conn.execute(
        "SELECT payload FROM task_events "
        "WHERE task_id = ? AND kind = 'review_followup_recorded' ORDER BY id",
        (task_id,),
    ).fetchall():
        payload = json.loads(row["payload"]) if row["payload"] else {}
        followup = payload.get("followup_task_id")
        if isinstance(followup, str) and followup and followup not in ids:
            ids.append(followup)
    return ids


def _raised(conn, task_id):
    """This card's ``risk_tier_raised`` payloads, oldest first."""
    return [
        json.loads(row["payload"])
        for row in conn.execute(
            "SELECT payload FROM task_events "
            "WHERE task_id = ? AND kind = 'risk_tier_raised' ORDER BY id",
            (task_id,),
        ).fetchall()
    ]


def _route_pins_and_box(conn, task_id):
    """What a route lock, an effort pin and a time box write on a card."""
    row = conn.execute(
        "SELECT model_policy_lock, model_override, provider_override, "
        "execution_tier, pinned_effort, reasoning_effort, max_runtime_seconds "
        "FROM tasks WHERE id = ?",
        (task_id,),
    ).fetchone()
    return tuple(row)


_NO_ROUTE_PIN_OR_BOX = (None,) * 7


def _finding(digest):
    """One finding of a typed changes-requested verdict on ``digest``."""
    return {
        "severity": "major",
        "file": "src/app.py",
        "lines": "10-20",
        "problem": "off-by-one in the loop bound",
        "impact": "drops the last item in the batch",
        "smallest_fix": "use `<=` instead of `<` in the range check",
        "candidate_digest": digest,
    }


def _park_for_review(conn, task_id, *, reviewer="reviewer"):
    """Park the card on the review lane and claim it as the reviewer."""
    task = kb.get_task(conn, task_id)
    assert task is not None
    assert kb.request_review(
        conn, task_id, summary="candidate ready", reviewer=reviewer,
        expected_run_id=task.current_run_id,
    ) is True
    claim = kb.claim_review_task(conn, task_id)
    assert claim is not None
    return claim.current_run_id


def _returned_once(conn, *, risk_tier, title="ship the widget", assignee="worker"):
    """A card at ``risk_tier`` one reviewer verdict returned, plus its item."""
    tid = kb.create_task(conn, title=title, assignee=assignee, risk_tier=risk_tier)
    assert kb.get_task(conn, tid).risk_tier == risk_tier
    assert kb.claim_task(conn, tid) is not None
    review_run = _park_for_review(conn, tid)
    ok, who = kb.request_changes(
        conn, tid, reason="fix the boundary", expected_run_id=review_run,
    )
    assert (ok, who) == (True, assignee)
    followups = _vouched_followups(conn, tid)
    assert len(followups) == 1, followups
    item = kb.get_task(conn, followups[0])
    assert item is not None and item.status == "triage"
    return tid, followups[0]


def _reviewer_approves(conn, task_id):
    """The reviewer's own clean verdict on the reworked candidate."""
    assert kb.claim_task(conn, task_id) is not None
    review_run = _park_for_review(conn, task_id)
    verdict = kb.submit_review_findings(
        conn, task_id, findings=[], candidate_digest="cand-clean",
        expected_run_id=review_run,
    )
    assert verdict["outcome"] == "passed", verdict
    assert kb.get_task(conn, task_id).status == "done"


@pytest.mark.parametrize("tier", [0, 1, 2])
def test_the_rework_item_records_the_reviewed_cards_tier(kanban_home, tier):
    """The item records the tier its card holds, and nothing else of a pin."""
    with kb.connect() as conn:
        tid, item = _returned_once(conn, risk_tier=tier)

        assert _route_pins_and_box(conn, item) == _NO_ROUTE_PIN_OR_BOX
        assert kb.get_task(conn, item).risk_tier == tier
        assert kb.get_task(conn, tid).risk_tier == tier


def test_a_raise_in_the_verdict_reaches_the_rework_item(kanban_home):
    """A tier-1 card the verdict itself raises to 2 gives an item at tier 2.

    The raise is recorded before the handback that opens the item, so the
    tier the item records is the raised one, not the one the card was parked
    with.
    """
    with kb.connect() as conn:
        tid = kb.create_task(
            conn, title="ship the widget", assignee="worker", risk_tier=1,
        )
        assert kb.claim_task(conn, tid) is not None
        review_run = _park_for_review(conn, tid)

        verdict = kb.submit_review_findings(
            conn, tid, findings=[_finding("cand-1")], candidate_digest="cand-1",
            expected_run_id=review_run, risk_tier=2,
        )

        raised = {"from": 1, "to": 2, "reviewer": "reviewer", "run_id": review_run}
        assert verdict["outcome"] == "handed_back", verdict
        assert verdict["risk_tier_raised"] == raised
        assert _raised(conn, tid) == [raised]
        assert kb.get_task(conn, tid).risk_tier == 2
        vouched = _vouched_followups(conn, tid)
        assert len(vouched) == 1, vouched
        assert _route_pins_and_box(conn, vouched[0]) == _NO_ROUTE_PIN_OR_BOX
        assert kb.get_task(conn, vouched[0]).risk_tier == 2


@pytest.mark.parametrize("verdict", ["request_changes", "findings_stating_tier_2"])
def test_an_untiered_card_gives_an_untiered_item(kanban_home, verdict):
    """No tier is invented: a card without one gives an item without one.

    The typed verdict states tier 2, which an untiered card already counts
    as, so the kernel records no raise -- and the item records no tier the
    card does not hold either.
    """
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="ship the widget", assignee="worker")
        assert kb.get_task(conn, tid).risk_tier is None
        assert kb.claim_task(conn, tid) is not None
        review_run = _park_for_review(conn, tid)

        if verdict == "request_changes":
            ok, _who = kb.request_changes(
                conn, tid, reason="fix the boundary", expected_run_id=review_run,
            )
            assert ok is True
        else:
            returned = kb.submit_review_findings(
                conn, tid, findings=[_finding("cand-1")],
                candidate_digest="cand-1", expected_run_id=review_run,
                risk_tier=2,
            )
            assert returned["outcome"] == "handed_back", returned
            assert "risk_tier_raised" not in returned

        assert _raised(conn, tid) == []
        assert kb.get_task(conn, tid).risk_tier is None
        vouched = _vouched_followups(conn, tid)
        assert len(vouched) == 1, vouched
        assert kb.get_task(conn, vouched[0]).risk_tier is None
        assert _route_pins_and_box(conn, vouched[0]) == _NO_ROUTE_PIN_OR_BOX


@pytest.mark.parametrize("tier", [0, 1, 2])
def test_the_approved_rework_item_no_longer_reads_risk_not_recorded(
    kanban_home, tier,
):
    """Approval closes the item, and the completed item keeps the card's tier."""
    with kb.connect() as conn:
        tid, item = _returned_once(conn, risk_tier=tier)
        _reviewer_approves(conn, tid)

        completed = kb.get_task(conn, item)
        assert completed.status == "done"
        assert completed.risk_tier is not None, (
            "the completed rework item has no tier recorded, so it still "
            "reads 'Risk not recorded'"
        )
        assert effective_risk_tier(completed.risk_tier) == tier
        assert kb.get_task(conn, tid).risk_tier == tier


@pytest.mark.parametrize(
    ("raise_to", "recorded"), [(None, 1), (2, 2)], ids=["kept", "raised"],
)
def test_a_route_locked_card_gives_its_item_the_tier_and_nothing_more(
    kanban_home, monkeypatch, raise_to, recorded,
):
    """The tier is the only thing the item takes from a governed card.

    The reviewed card carries an owner-approved route lock, an effort pin and
    a time box, all from its tier. The item records the card's tier -- the
    raised one when the verdict raises it, which also re-pins the card -- and
    none of the rest: no lock, no route, no pinned effort and no box.
    """
    # Stand in for each role's on-disk config only, as test_owner_workspace
    # does; the policy, the pins and the seals run as production code.
    monkeypatch.setattr(
        mp, "configured_assignment_for",
        lambda profile: mp.assignment_for(profile, "anthropic"),
    )
    assignee = "raphael-claude-worker"
    route = mp.task_assignment_for(assignee, "anthropic", "deep")
    with kb.connect() as conn:
        tid = kb.create_task(
            conn, title="ship the widget", assignee=assignee,
            execution_tier="deep", provider_override=route.provider,
            model_override=route.model, reasoning_effort=route.reasoning_effort,
            model_policy_lock=kb.mint_policy_lock(
                assignee, route.provider, route.model, route.reasoning_effort,
                "deep",
            ),
            requires_review=True, risk_tier=1,
        )
        lock, *_route, pinned, _effort, box = _route_pins_and_box(conn, tid)
        assert None not in (lock, pinned, box)
        assert kb.claim_task(conn, tid) is not None
        review_run = _park_for_review(
            conn, tid, reviewer=kb.policy_resolved_reviewer(),
        )

        verdict = kb.submit_review_findings(
            conn, tid, findings=[_finding("cand-1")], candidate_digest="cand-1",
            expected_run_id=review_run, risk_tier=raise_to,
        )

        assert verdict["outcome"] == "handed_back", verdict
        assert kb.get_task(conn, tid).risk_tier == recorded
        vouched = _vouched_followups(conn, tid)
        assert len(vouched) == 1, vouched
        assert _route_pins_and_box(conn, vouched[0]) == _NO_ROUTE_PIN_OR_BOX
        assert kb.get_task(conn, vouched[0]).risk_tier == recorded


def test_a_rework_item_already_recorded_is_left_exactly_as_it_was(kanban_home):
    """Past records are not touched, not even by a verdict that finds one.

    A card whose park recorded no head has one rework item for every
    non-pass verdict, so a later verdict finds the item already recorded and
    reuses it. Here that item is one recorded before items carried a tier:
    it holds null. The later verdict raises the card to 2 -- and still leaves
    the item's row exactly as it was, records no second item, and back-fills
    nothing.
    """
    with kb.connect() as conn:
        tid, item = _returned_once(conn, risk_tier=1)
        conn.execute("UPDATE tasks SET risk_tier = NULL WHERE id = ?", (item,))
        recorded = tuple(
            conn.execute("SELECT * FROM tasks WHERE id = ?", (item,)).fetchone()
        )

        assert kb.claim_task(conn, tid) is not None
        review_run = _park_for_review(conn, tid)
        verdict = kb.submit_review_findings(
            conn, tid, findings=[_finding("cand-2")], candidate_digest="cand-2",
            expected_run_id=review_run, risk_tier=2,
        )

        assert verdict["outcome"] == "handed_back", verdict
        assert kb.get_task(conn, tid).risk_tier == 2
        assert _vouched_followups(conn, tid) == [item]
        children = conn.execute(
            "SELECT COUNT(*) AS n FROM task_links WHERE parent_id = ?", (tid,),
        ).fetchone()["n"]
        assert children == 1
        assert tuple(
            conn.execute("SELECT * FROM tasks WHERE id = ?", (item,)).fetchone()
        ) == recorded


def _count(db, table, where, params=()):
    """An independent read: a plain sqlite connection, no Hermes code."""
    if not db.exists():
        return 0
    with closing(sqlite3.connect(db)) as raw:
        return raw.execute(
            f"SELECT COUNT(*) FROM {table} WHERE {where}", params,
        ).fetchone()[0]


def test_a_reviewer_pinned_to_its_board_records_the_tier_and_still_reaches_the_root(
    tmp_path, monkeypatch,
):
    """The production proof, read under the environment a reviewer really gets.

    The rework item is written from a reviewer's run: the verdict that hands
    the card back. The dispatcher pins that reviewer to the board the card
    lives on (``HERMES_KANBAN_DB`` + ``HERMES_KANBAN_BOARD``), names the task
    and the run (``HERMES_KANBAN_TASK``, ``HERMES_KANBAN_RUN_ID``), and points
    ``HERMES_HOME`` at the reviewer's profile home UNDER the shared root. The
    item must record the raised tier on that board alone, and from there the
    kernel must still reach the root: its register, its board list and every
    other board.
    """
    root = tmp_path / "hermes_root"
    root.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(root))
    kb._INITIALIZED_PATHS.clear()
    kb.init_db()
    kb.create_board("worker-board")
    kb.create_board("other-board")
    # Spelled out from the root, not resolved, for the independent count.
    board_db = root / "kanban" / "boards" / "worker-board" / "kanban.db"
    other_db = root / "kanban" / "boards" / "other-board" / "kanban.db"
    default_db = root / "kanban.db"
    assert kb.kanban_db_path(board="worker-board") == board_db

    with closing(kb.connect(board="other-board")) as other:
        kb.create_task(other, title="someone else's card", assignee="worker")
    with closing(kb.connect(board="worker-board")) as conn:
        tid = kb.create_task(
            conn, title="ship the widget", assignee="worker", risk_tier=1,
        )
        claimed = kb.claim_task(conn, tid)
        assert claimed is not None
        assert kb.request_review(
            conn, tid, summary="candidate ready", reviewer="reviewer",
            expected_run_id=claimed.current_run_id,
        ) is True
        review = kb.claim_review_task(conn, tid)
        assert review is not None and review.claim_lock
    workspaces = kb.workspaces_root(board="worker-board")
    before_other = _count(other_db, "tasks", "1 = 1")
    before_default = _count(default_db, "tasks", "1 = 1")

    # The kanban part of what ``_default_spawn`` injects into the reviewer.
    profile_home = root / "profiles" / review.assignee
    profile_home.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(profile_home))
    monkeypatch.setenv("HERMES_PROFILE", review.assignee)
    monkeypatch.setenv("HERMES_KANBAN_DB", str(board_db))
    monkeypatch.setenv("HERMES_KANBAN_BOARD", "worker-board")
    monkeypatch.setenv("HERMES_KANBAN_WORKSPACES_ROOT", str(workspaces))
    monkeypatch.setenv("HERMES_KANBAN_TASK", tid)
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(review.current_run_id))
    monkeypatch.setenv("HERMES_KANBAN_CLAIM_LOCK", review.claim_lock)

    with closing(kb.connect()) as conn:
        # The pin, not an argument, is what routed this connection.
        opened = conn.execute("PRAGMA database_list").fetchone()[2]
        assert os.path.realpath(opened) == os.path.realpath(board_db)

        verdict = kb.submit_review_findings(
            conn, os.environ["HERMES_KANBAN_TASK"],
            findings=[_finding("cand-1")], candidate_digest="cand-1",
            expected_run_id=int(os.environ["HERMES_KANBAN_RUN_ID"]), risk_tier=2,
        )
        assert verdict["outcome"] == "handed_back", verdict
        assert verdict["risk_tier_raised"] == {
            "from": 1, "to": 2, "reviewer": review.assignee,
            "run_id": review.current_run_id,
        }
        vouched = _vouched_followups(conn, tid)
        assert len(vouched) == 1, vouched
        # The production proof ...
        assert _route_pins_and_box(conn, vouched[0]) == _NO_ROUTE_PIN_OR_BOX
        assert kb.get_task(conn, vouched[0]).risk_tier == 2

    # ... against an independent count of what each database holds: the
    # card's children at tier 2 on its own board, and nothing anywhere else.
    children = "tasks t JOIN task_links l ON l.child_id = t.id"
    assert _count(
        board_db, children, "l.parent_id = ? AND t.risk_tier = 2", (tid,),
    ) == len(vouched)
    assert _count(
        board_db, children, "l.parent_id = ? AND t.risk_tier IS NULL", (tid,),
    ) == 0
    assert _count(board_db, "tasks", "id = ? AND risk_tier = 2", (tid,)) == 1
    assert _count(other_db, "tasks", "1 = 1") == before_other
    assert _count(default_db, "tasks", "1 = 1") == before_default
    assert not (profile_home / "kanban.db").exists()
    assert not (profile_home / "kanban").exists()

    # Pinned to one board, the reviewer still reaches the root: the board
    # register, the board list, and another board.
    assert kb.kanban_home() == root
    assert kb.register_db_path() == root / "kanban" / "board_register.db"
    entry = kb.get_register_entry("worker-board")
    assert entry is not None
    assert entry.lifecycle is kb.BoardLifecycle.LIVE
    assert {board["slug"] for board in kb.list_boards()} == {
        "default", "worker-board", "other-board",
    }
    with closing(kb.connect(db_path=other_db)) as other:
        assert [task.title for task in kb.list_tasks(other)] == [
            "someone else's card",
        ]
