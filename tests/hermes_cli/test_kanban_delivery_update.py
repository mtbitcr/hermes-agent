"""Card 3: with ``kanban.delivery.exact_base`` exactly true, a delivery row the arm left in base_moved is brought
up to date on the current main without touching any checkout: git merge-tree --write-tree of main and the approved
head, then git commit-tree with the approved head and main as the two parents. When the update changes the same
paths relative to main as the approved change did relative to its base, and nothing outside the source card's
owned paths, one update card is made: a child of the source card, its branch at the updated head, its recorded
base main's commit and its review handover recorded by the kernel, so only its own approval merges it. The old row
is updated, and its pull request closes once the update is published. A conflict or a changed set of paths goes
back to the builder through the continuation card instead. With the setting off a base_moved row stays where it is.
Real boards and git in temporary folders; GitHub is the real transport with only its exchange and App token
replaced, so every call passes the allowlist."""

from __future__ import annotations

import json

import pytest

from tests.hermes_cli.test_kanban_delivery import (  # noqa: F401  (board is a fixture)
    IMPLEMENTER, REVIEWER, _approve, _git, board,
)
from tests.hermes_cli.test_kanban_delivery_auto_merge import (  # noqa: F401  (armed is a fixture)
    _arms, _done, _lenses, _published_as, _reviews, armed,
)
from tests.hermes_cli.test_kanban_delivery_exact_base_arm import _setting, exact  # noqa: F401  (exact is a fixture)
from tests.hermes_cli.test_kanban_delivery_github import REPO
from tests.hermes_cli.test_kanban_delivery_publish_step import _approved, _events, _tick
from tests.hermes_cli.test_kanban_delivery_review_cards import _raw, _state, world  # noqa: F401  (world is a fixture)
from tests.hermes_cli.test_kanban_delivery_review_return import _ready, _registered
from tests.hermes_cli.test_release_guards_batch import _g3

# The source card's update cards, and its continuation: its task_links child that is none of its review cards.
UPDATES = ("SELECT * FROM tasks WHERE id IN (SELECT child_id FROM task_links WHERE parent_id = ?) "
           "AND idempotency_key LIKE 'update:%' ORDER BY id")
CONTINUED = ("SELECT * FROM tasks WHERE id IN (SELECT child_id FROM task_links WHERE parent_id = ?) "
             "AND idempotency_key IS NULL")


def _approved_head(scene, name="feature"):
    """An approved head H of a source card of the project whose primary folder is the board repository, its review
    cards done and approved by every lens: the next pass arms it. Returns the card, H and the reviewed base."""
    kb, root, repo, gh = scene
    tid, head = _ready(scene, name=name)
    _tick(kb)
    _done(kb.kanban_db_path())
    _reviews(gh, *_lenses(kb, head))
    return tid, head, _git(repo, "rev-parse", "main")


def _main_moves(repo, path="docs/other.md", text="other\n") -> str:
    """main moves on with one commit writing ``path``, made in the repository's own checkout of main."""
    (repo / path).parent.mkdir(parents=True, exist_ok=True)
    (repo / path).write_text(text, encoding="utf-8")
    _git(repo, "add", path)
    _git(repo, "commit", "-q", "-m", f"main writes {path}")
    return _git(repo, "rev-parse", "main")


def _moved_on(scene, **change):
    """H approved, then main moves on with ``change`` and the setting is exactly true: the next pass finds main
    neither the reviewed base nor in H, so the arm sends nothing and records base_moved."""
    kb, root, repo, gh = scene
    tid, head, base = _approved_head(scene)
    main = _main_moves(repo, **change)
    gh["main"] = main
    _setting(root, "true")
    return tid, head, base, main


def _checkout(repo) -> tuple:
    """What a checkout shows: the repository's HEAD, its status and its worktrees."""
    return (_git(repo, "rev-parse", "HEAD"), _git(repo, "status", "--porcelain"),
            _git(repo, "worktree", "list", "--porcelain"))


def _branches(repo) -> set:
    return set(_git(repo, "for-each-ref", "--format=%(refname:short)", "refs/heads/").split())


def _update(kb, tid) -> dict:
    """The source card's one update card."""
    [card] = _raw(kb.kanban_db_path(), UPDATES, tid)
    return card


def _handovers(kb, card) -> list:
    """The payloads of the card's review_requested events: the review handovers the kernel recorded."""
    return [json.loads(row["payload"]) for row in _raw(
        kb.kanban_db_path(), "SELECT payload FROM task_events WHERE task_id = ? AND kind = 'review_requested' "
        "ORDER BY id", card)]


def _approve_card(kb, card) -> bool:
    """The reviewer claims the card out of review and approves it."""
    conn = kb.connect()
    try:
        return _approve(kb, conn, card)
    finally:
        conn.close()


def _publish(kb, gh, tid, head, number, base):
    """What the publish stores for ``head`` of ``tid`` as pull request ``number``: the ledger and its record of
    the repository, open at that head on GitHub, and delivery_bound to ``base``."""
    db = kb.kanban_db_path()
    _published_as(db, gh, tid, head, number)
    _raw(db, "INSERT INTO task_events (task_id, kind, payload, created_at) VALUES (?, 'delivery_bound', ?, 0)",
         tid, json.dumps({"source_task_id": tid, "head": head, "base_commit": base}))


def _lands(scene, head, number) -> dict:
    """GitHub's merge of pull request ``number``, made with real git on the repository's main during the merge
    request: a two-parent commit with ``head`` second. GitHub then shows it merged as that commit, and main at it."""
    kb, root, repo, gh = scene
    landed = {}

    def merge():
        _git(repo, "merge", "-q", "--no-ff", "--no-edit", head)
        landed["merge"] = gh["main"] = gh["ended"][number] = _git(repo, "rev-parse", "HEAD")

    gh["hooks"]["PUT"] = merge
    return landed


def _closes(gh) -> list:
    return [write for write in gh["writes"] if write[0] == "PATCH"]


def _returned(kb, repo, gh, tid, head, main, reason, branches):
    """H went back to its builder: one continuation card for the builder on a branch at H, no update card, no
    card in review and no other branch, nothing merged, and the row returned for changes with one event."""
    db = kb.kanban_db_path()
    [card] = _raw(db, CONTINUED, tid)
    branch = f"delivery-return/{tid}-{head[:12]}"
    assert (card["assignee"], card["base_commit"], card["branch_name"]) == (IMPLEMENTER, head, branch)
    assert _git(repo, "rev-parse", f"refs/heads/{branch}") == head and _branches(repo) == branches | {branch}
    assert _raw(db, UPDATES, tid) == [] and _raw(db, "SELECT id FROM tasks WHERE status = 'review'") == []
    assert (_arms(gh), _state(kb, head)) == ([], "returned_for_changes")
    returned = [payload for kind, payload in _events(kb, tid) if kind == "delivery_update_returned"]
    assert [(payload["reason"], payload["observed_main"], payload["rework"]) for payload in returned] == [
        (reason, main, card["id"])]


def test_a_moved_branch_gets_one_update_card_based_on_current_main(exact):
    kb, root, repo, gh = exact
    tid, head, base, main = _moved_on(exact)
    before = _checkout(repo)

    _tick(kb)

    card = _update(kb, tid)
    update = card["head_commit"]
    assert card["idempotency_key"] == f"update:{tid}:{head}:{main}"
    assert (card["status"], card["base_commit"]) == ("review", main)
    assert _git(repo, "rev-parse", f"refs/heads/{card['branch_name']}") == update
    assert _git(repo, "rev-list", "--parents", "-n", "1", update).split()[1:] == [head, main]
    assert _git(repo, "rev-parse", f"{update}^{{tree}}") == _git(
        repo, "merge-tree", "--write-tree", main, head).splitlines()[0]
    assert set(_git(repo, "diff", "--name-only", main, update).split()) == {"src/impl/feature.py"}
    assert [(handover["implementer"], handover["implementer_head_commit"]) for handover in _handovers(
        kb, card["id"])] == [(IMPLEMENTER, update)]
    assert (_state(kb, head), _arms(gh)) == ("updated", [])
    updated = [payload for kind, payload in _events(kb, tid) if kind == "delivery_updated"]
    assert [(payload["update_task_id"], payload["update_head"], payload["observed_main"], payload["reviewed_base"])
            for payload in updated] == [(card["id"], update, main, base)]
    assert _checkout(repo) == before and not (repo / ".worktrees" / card["id"]).exists()


def test_an_updated_head_never_merges_without_its_own_approval(exact):
    from hermes_cli import kanban_delivery as kd

    kb, root, repo, gh = exact
    tid, head, base, main = _moved_on(exact)
    _tick(kb)
    card = _update(kb, tid)
    update = card["head_commit"]

    _tick(kb)
    _tick(kb)

    db = kb.kanban_db_path()
    assert (_arms(gh), _state(kb, head)) == ([], "updated")
    assert _raw(db, "SELECT id FROM kanban_deliveries WHERE source_task_id = ?", card["id"]) == []
    conn = kb.connect()
    try:
        assert not kd.prove_approval(conn, tid, update).allowed  # the source's approval is of H only
        assert not kd.prove_approval(conn, card["id"], update).allowed
        assert kd.record_approved_delivery(conn, card["id"]) is None
        assert _approve(kb, conn, card["id"]) is True  # its own review approves it
        assert kd.prove_approval(conn, card["id"], update).allowed
    finally:
        conn.close()
    [row] = _raw(db, "SELECT source_head, approval_run_id FROM kanban_deliveries WHERE source_task_id = ?", card["id"])
    approver = _raw(db, "SELECT task_id, profile FROM task_runs WHERE id = ?", row["approval_run_id"])
    assert (row["source_head"], approver) == (update, [{"task_id": card["id"], "profile": REVIEWER}])
    assert _arms(gh) == []


@pytest.mark.parametrize("change", ["main_holds_the_change", "main_rewritten_under_the_base"])
def test_an_update_that_changes_other_paths_goes_back_to_the_builder(exact, change):
    """The update would change another set of paths relative to main than the approved change did relative to its
    base: main holds the change already, so the update changes none of them; or main was rewritten under the
    reviewed base, so the update would bring back a path review never saw, outside the card's owned paths."""
    kb, root, repo, gh = exact
    start = _git(repo, "rev-parse", "main")
    if change == "main_rewritten_under_the_base":
        _main_moves(repo, "docs/reviewed.md", "in the reviewed base\n")
    tid, head, base = _approved_head(exact)
    if change == "main_holds_the_change":
        main = _main_moves(repo, "src/impl/feature.py", "ok = True\n")
    else:
        _git(repo, "reset", "-q", "--hard", start)
        main = _main_moves(repo)
    gh["main"] = main
    _setting(root, "true")
    branches = _branches(repo)

    _tick(kb)

    _returned(kb, repo, gh, tid, head, main, "paths", branches)


def test_a_conflicting_update_goes_back_to_the_builder_with_no_card_in_review(exact):
    kb, root, repo, gh = exact
    tid, head, base, main = _moved_on(exact, path="src/impl/feature.py", text="ok = False\n")
    before, branches = _checkout(repo), _branches(repo)

    _tick(kb)

    _returned(kb, repo, gh, tid, head, main, "conflict", branches)
    assert _checkout(repo) == before
    _tick(kb)
    assert len(_raw(kb.kanban_db_path(), CONTINUED, tid)) == 1


def test_the_old_pull_request_closes_once_the_update_is_published(exact):
    kb, root, repo, gh = exact
    tid, head, base, main = _moved_on(exact)
    _tick(kb)
    card = _update(kb, tid)
    update = card["head_commit"]
    _tick(kb)
    assert _closes(gh) == []  # the update is not published yet: pull request 41 stays open

    assert _approve_card(kb, card["id"]) is True
    _publish(kb, gh, card["id"], update, 42, main)
    _tick(kb)

    assert _closes(gh) == [("PATCH", f"/repos/{REPO}/pulls/41", {"state": "closed"})]
    assert _state(kb, head) == "replaced"
    replaced = [payload for kind, payload in _events(kb, tid) if kind == "delivery_replaced"]
    assert [(payload["replaced_by"], payload["closed"]) for payload in replaced] == [(card["id"], True)]
    _tick(kb)
    assert len(_closes(gh)) == 1


def test_updating_twice_on_the_same_main_makes_one_card(exact):
    kb, root, repo, gh = exact
    tid, head, base, main = _moved_on(exact)
    _tick(kb)
    card = _update(kb, tid)
    from hermes_cli import kanban_delivery_update

    db = kb.kanban_db_path()
    # the old row reads base_moved on the same main again, as a second pass that read it before the first one's
    # write sees it
    _raw(db, "UPDATE kanban_deliveries SET pull_request_state = 'base_moved' WHERE source_head = ?", head)
    kanban_delivery_update.update_step(db)
    kanban_delivery_update.update_step(db)
    _tick(kb)

    assert [update["id"] for update in _raw(db, UPDATES, tid)] == [card["id"]]
    assert _git(repo, "rev-parse", f"refs/heads/{card['branch_name']}") == card["head_commit"]
    assert (_state(kb, head), _raw(db, CONTINUED, tid), _handovers(kb, card["id"])) == (
        "updated", [], _handovers(kb, card["id"])[:1])


def test_two_changes_merge_before_one_release_and_both_pass_the_release_guards(exact):
    """Acceptance: two cards are approved on the same main T0. The first merges as M1. The second, reviewed on T0,
    is moved: the arm records base_moved and its update card carries the update on M1. Approved again by its own
    review, published and merged as M2, it joins the waiting release batch after the first, and
    release_guards.prepare gives G3 ok for T0..M2 against the real repository."""
    from hermes_cli import release_ledger
    from hermes_cli.release_guards import RecordedMerge

    kb, root, repo, gh = exact
    db = kb.kanban_db_path()
    t0 = _git(repo, "rev-parse", "main")
    first, first_head = _ready(exact)
    second, second_head = _approved(kb, repo, "second")
    _registered(db, second, repo)
    _publish(kb, gh, second, second_head, 43, t0)
    _tick(kb)
    _tick(kb)
    _done(db)
    _reviews(gh, *_lenses(kb, first_head))
    _reviews(gh, *_lenses(kb, second_head), number=43)
    gh.update(main=t0, close=(200, {"number": 43, "state": "closed"}))
    _setting(root, "true")
    first_landed = _lands(exact, first_head, 41)

    _tick(kb)  # the first merges as M1; the second finds main moved, and its update is made on M1
    _tick(kb)  # the merge pass records M1

    m1 = first_landed["merge"]
    card = _update(kb, second)
    update = card["head_commit"]
    assert (_state(kb, first_head), _state(kb, second_head), card["base_commit"]) == ("merged", "updated", m1)
    assert len(_arms(gh)) == 1
    assert _approve_card(kb, card["id"]) is True
    _publish(kb, gh, card["id"], update, 42, m1)
    _tick(kb)  # CI is done on the update: its review cards; and 43, which the update replaces, closes
    _done(db)
    _reviews(gh, *_lenses(kb, update), number=42)
    second_landed = _lands(exact, update, 42)
    _tick(kb)  # main is the update's reviewed base M1: it merges as M2
    _tick(kb)  # the merge pass records M2

    m2 = second_landed["merge"]
    assert (_state(kb, update), _state(kb, second_head), len(_arms(gh))) == ("merged", "replaced", 2)
    conn = release_ledger.connect()
    try:
        [waiting] = [batch for batch in release_ledger.list_batches(conn) if batch["state"] in ("open", "deferred")]
    finally:
        conn.close()
    members = waiting["members"]
    assert [(member["card_id"], member["merge_commit"], member["reviewed_base"], member["reviewed_head"])
            for member in members] == [(first, m1, t0, first_head), (card["id"], m2, m1, update)]
    merges = [RecordedMerge(member["merge_commit"], reviewed_base=member["reviewed_base"],
                            reviewed_head=member["reviewed_head"], reviewed_tree=member["reviewed_tree"])
              for member in members]
    assert _g3(repo, t0, m2, merges) is True


@pytest.mark.parametrize("value", [None, "false", '"true"'], ids=["absent", "false", "quoted_true"])
def test_with_the_setting_off_a_base_moved_row_stays_where_it_is(exact, value):
    """Rollback: with the setting anything but exactly true, a row card 2 left base_moved stays exactly as it is,
    and no card, branch, event or GitHub write is made, however many passes run."""
    kb, root, repo, gh = exact
    tid, head, base = _approved_head(exact)
    main = _main_moves(repo)
    gh["main"] = main
    db = kb.kanban_db_path()
    [row] = _raw(db, "SELECT id FROM kanban_deliveries WHERE source_head = ?", head)
    _raw(db, "UPDATE kanban_deliveries SET pull_request_state = 'base_moved' WHERE id = ?", row["id"])
    _raw(db, "INSERT INTO task_events (task_id, kind, payload, created_at) VALUES (?, 'delivery_base_moved', ?, 0)",
         tid, json.dumps({"delivery_id": row["id"], "head": head, "pull_request_number": 41, "repository": REPO,
                          "reviewed_base": base, "observed_main": main, "cards": []}))
    _setting(root, value)
    rows, tasks = (_raw(db, f"SELECT * FROM {table} ORDER BY id") for table in ("kanban_deliveries", "tasks"))
    events, branches, writes = _events(kb, tid), _branches(repo), list(gh["writes"])

    _tick(kb)
    _tick(kb)

    assert _raw(db, "SELECT * FROM kanban_deliveries ORDER BY id") == rows and _state(kb, head) == "base_moved"
    assert [task["id"] for task in _raw(db, "SELECT id FROM tasks ORDER BY id")] == [task["id"] for task in tasks]
    assert (_events(kb, tid), _branches(repo), gh["writes"]) == (events, branches, writes)
