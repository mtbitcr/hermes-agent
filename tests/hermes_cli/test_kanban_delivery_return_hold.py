"""A returned delivery is held until its rework is published. While the continuation of a row whose head H went back
to its builder (a red check or a review returned it) is neither done nor archived, a moved main records nothing for
that row: no base_moved, no update and no new return; H's own checks are still read as today. Once that card is
archived or done, the row is handled as today. A rework whose recorded base is the head of an earlier row of the same
pull request replaces that pull request: once its own pull request is published, the arm step closes the source pull
request once, whichever head the update gave it; a merged row is never closed.
Real boards and git; GitHub is the real transport with only its exchange and App token replaced by a table, and its
git origin is a local bare repository."""

from __future__ import annotations

import pytest

from tests.hermes_cli.test_kanban_delivery import IMPLEMENTER, _git, board  # noqa: F401  (board is a fixture)
from tests.hermes_cli.test_kanban_delivery_auto_merge import (  # noqa: F401  (armed is a fixture)
    _arms, _continues, _done, _lenses, _published_as, _reviews, armed,
)
from tests.hermes_cli.test_kanban_delivery_exact_base_arm import _moved, _setting, exact  # noqa: F401  (a fixture)
from tests.hermes_cli.test_kanban_delivery_github import REPO
from tests.hermes_cli.test_kanban_delivery_publish_step import _approved, _events, _tick
from tests.hermes_cli.test_kanban_delivery_red_check import UNLISTED, _waited, red  # noqa: F401  (red is a fixture)
from tests.hermes_cli.test_kanban_delivery_review_cards import (  # noqa: F401  (world is a fixture)
    _cards, _raw, _state, world,
)
from tests.hermes_cli.test_kanban_delivery_review_return import CONTINUED, _ready
from tests.hermes_cli.test_kanban_delivery_update import UPDATES, _head, _main_moves, _parents, _update


def _returned_for_red(scene):
    """H published, its source card of the project whose primary folder is the board repository; H's required check
    is red on a failed test no list names, so one pass returns the work through H's one continuation, ready for its
    builder. Returns the source card, H, H's reviewed base and the continuation."""
    kb, root, repo, gh = scene
    tid, head = _ready(scene)
    gh["tests"] = [UNLISTED]
    _tick(kb)
    (rework,) = _raw(kb.kanban_db_path(), CONTINUED, tid)
    assert [payload["rework"] for payload in _waited(kb, tid, "red_check_returned")] == [rework["id"]]
    assert rework["status"] == "ready"
    return tid, head, _git(repo, "rev-parse", "main"), rework["id"]


def _held(scene, tid, head, rework, state):
    """What a held row shows: no delivery_base_moved or delivery_updated event, no update card, H's one
    continuation ``rework`` and no merge sent, the row still ``state``."""
    kb, root, repo, gh = scene
    db = kb.kanban_db_path()
    assert not {kind for kind, _ in _events(kb, tid)} & {"delivery_base_moved", "delivery_updated"}
    assert (_raw(db, UPDATES, tid), [card["id"] for card in _raw(db, CONTINUED, tid)]) == ([], [rework])
    assert (_arms(gh), _state(kb, head)) == ([], state)


def test_a_row_returned_for_a_red_check_records_nothing_when_main_moves(red, exact):
    """H's red check returned the work; while H's continuation is ready, main moves on and the setting is exactly
    true. Two passes record nothing for the row: no delivery_base_moved, no update card and no second continuation,
    and the row stays open. H's own checks are still read, so a green rerun of H gets its review card."""
    kb, root, repo, gh = exact
    tid, head, base, rework = _returned_for_red(exact)
    _main_moves(exact)
    _setting(root, "true")
    events = _events(kb, tid)

    _tick(kb)
    _tick(kb)

    assert _events(kb, tid) == events
    _held(exact, tid, head, rework, "open")
    gh["check"] = ("completed", "success")
    _tick(kb)
    assert (len(_cards(kb)), _state(kb, head)) == (1, "review_cards_created")
    _held(exact, tid, head, rework, "review_cards_created")


def test_a_row_returned_for_a_review_records_nothing_when_main_moves(exact):
    """The reviewer bot's review of H's card requested changes on GitHub, so H's work went back to its builder
    through its one continuation; the bot then approves H on that card, and main moves on with the setting exactly
    true. While the continuation is ready, two passes send and record nothing for the row: no delivery_base_moved,
    no update card and no second continuation, and the row keeps its review cards."""
    kb, root, repo, gh = exact
    db = kb.kanban_db_path()
    tid, head = _ready(exact)
    _tick(kb)
    (card,) = _cards(kb)
    _done(db)
    _reviews(gh, ("CHANGES_REQUESTED", head, card["id"]))
    _tick(kb)
    (rework,) = _raw(db, CONTINUED, tid)
    assert [payload["rework"] for payload in _waited(kb, tid, "review_returned")] == [rework["id"]]
    _reviews(gh, ("CHANGES_REQUESTED", head, card["id"]), *_lenses(kb, head))
    _main_moves(exact)
    _setting(root, "true")
    events = _events(kb, tid)

    _tick(kb)
    _tick(kb)

    assert _events(kb, tid) == events
    _held(exact, tid, head, rework["id"], "review_cards_created")


@pytest.mark.parametrize("left", ["archived", "done"])
def test_the_held_row_is_updated_as_today_once_its_continuation_is_archived_or_done(red, exact, left):
    """The row H's red check returned is held while main moved on: a pass records nothing. Once H's continuation is
    archived (or done), the next pass handles the row as today: one delivery_base_moved naming the reviewed base,
    the observed main and no card, and one update card on current main for H's builder; the row is updated."""
    kb, root, repo, gh = exact
    db = kb.kanban_db_path()
    tid, head, base, rework = _returned_for_red(exact)
    main = _main_moves(exact)
    _setting(root, "true")
    events = _events(kb, tid)
    _tick(kb)
    assert _events(kb, tid) == events
    _held(exact, tid, head, rework, "open")
    _raw(db, "UPDATE tasks SET status = ? WHERE id = ?", left, rework)

    _tick(kb)

    assert [(payload["reviewed_base"], payload["observed_main"], payload["cards"]) for kind, payload in _events(
        kb, tid) if kind == "delivery_base_moved"] == [(base, main, [])]
    card = _update(kb, tid)
    assert (card["status"], card["assignee"], card["base_commit"]) == ("ready", IMPLEMENTER, main)
    assert _parents(repo, _head(repo, card)) == [head, main]
    assert (_state(kb, head), _arms(gh)) == ("updated", [])


def _reworked_after_update(armed):
    """H published as pull request 41. A rework card continues H: the source card's task_links child, H its recorded
    base, approved at its own head. The update then moved pull request 41 on to U, as a clean update and its publish
    leave it: H's row updated, and a later row of the source card at U published on pull request 41, open at U on
    GitHub. Returns the source card, H, U, the rework card and its head."""
    kb, root, repo, gh = armed
    db = kb.kanban_db_path()
    tid, head = _ready(armed)
    rework, continued = _approved(kb, repo, "rework")
    _continues(db, rework, head, tid)
    update = _git(repo, "commit-tree", f"{head}^{{tree}}", "-p", head, "-p", _moved(repo), "-m", "H brought onto main")
    _raw(db, "INSERT INTO kanban_deliveries (source_task_id, source_head, approval_run_id, created_at) "
         "SELECT source_task_id, ?, approval_run_id, 0 FROM kanban_deliveries WHERE source_head = ?", update, head)
    _raw(db, "UPDATE kanban_deliveries SET pull_request_state = 'updated' WHERE source_head = ?", head)
    _published_as(db, gh, tid, update, 41)
    return tid, head, update, rework, continued


def test_a_rework_from_an_earlier_head_closes_the_source_pull_request_once_at_the_head_the_update_gave_it(armed):
    """The rework card that started from H is published as pull request 42 after the update moved pull request 41
    on to U: one arm step closes pull request 41 once as replaced by the rework card, its row at U recorded replaced
    and H's row left updated; a second arm step sends nothing."""
    from hermes_cli.kanban_delivery import arm_step

    kb, root, repo, gh = armed
    db = kb.kanban_db_path()
    tid, head, update, rework, continued = _reworked_after_update(armed)
    _published_as(db, gh, rework, continued, 42)

    arm_step(db)

    assert gh["writes"] == [("PATCH", f"/repos/{REPO}/pulls/41", {"state": "closed"})]
    assert [(kind, payload["head"], payload["pull_request_number"], payload["replaced_by"])
            for kind, payload in _events(kb, tid) if kind in ("delivery_close_pending", "delivery_replaced")] == [
        ("delivery_close_pending", update, 41, rework), ("delivery_replaced", update, 41, rework)]
    assert (_state(kb, head), _state(kb, update)) == ("updated", "replaced")
    writes, calls, events = list(gh["writes"]), list(gh["calls"]), _events(kb, tid)

    arm_step(db)

    assert (gh["writes"], gh["calls"], _events(kb, tid)) == (writes, calls, events)


def test_a_merged_row_is_never_closed(armed):
    """As before, but GitHub merged pull request 41 at U and its row is merged: once the rework card's pull request
    42 is published, two arm steps neither read nor close pull request 41, and the row stays merged."""
    from hermes_cli.kanban_delivery import arm_step

    kb, root, repo, gh = armed
    db = kb.kanban_db_path()
    tid, head, update, rework, continued = _reworked_after_update(armed)
    _raw(db, "UPDATE kanban_deliveries SET pull_request_state = 'merged' WHERE source_head = ?", update)
    gh["ended"][41] = "d" * 40  # GitHub shows it merged into main as that merge commit
    _published_as(db, gh, rework, continued, 42)
    calls, events = list(gh["calls"]), _events(kb, tid)

    arm_step(db)
    arm_step(db)

    assert (gh["writes"], gh["calls"], _events(kb, tid)) == ([], calls, events)
    assert (_state(kb, head), _state(kb, update)) == ("updated", "merged")
