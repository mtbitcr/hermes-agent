"""Card 3: with ``kanban.delivery.exact_base`` exactly true, a delivery row the arm left in base_moved is brought
up to date on the current main without touching any checkout. Current main is read and its objects fetched with
the fenced fetch worktree_start uses, into refs/remotes/origin/main only; then git merge-tree --write-tree of that
main and the approved head, and git commit-tree with the approved head and main as the two parents. When the update
changes the same paths relative to main as the approved change did relative to its base, and nothing outside the
source card's owned paths, one update card is made through _continuation's card creation (operator note of 824): a
child of the source card for its original builder, with the source's tier, route, lock and receipt binding, on a
branch at the updated head, main its recorded base, as the normal completion hands over only committed changes
since that base. Its builder hands that head over unchanged through the normal completion, so the kernel's reviewer
nomination and route authority apply and only its own approval merges it. The kernel's delivery_updated event,
written with the card, is the one proof of an update card. The old row is updated, and its
pull request closes once the update is published. A changed set of paths goes back to the builder through the
continuation card instead, and so does a conflict outside the approved paths; a conflict inside them gets the update
card on main with the conflict marked. With the setting off a base_moved row stays where it is. Real boards and git
in temporary folders; GitHub is the real transport with only its exchange and App token replaced, so every call
passes the allowlist, and its git origin is a local bare repository."""

from __future__ import annotations

import json
import sqlite3
import subprocess
from pathlib import Path

import pytest

from tests.hermes_cli.test_kanban_delivery import (  # noqa: F401  (board is a fixture)
    IMPLEMENTER, REVIEWER, _approve, _become, _git, _spawned_worker_env, board,
)
from tests.hermes_cli.test_kanban_delivery_auto_merge import (  # noqa: F401  (armed is a fixture)
    _arms, _done, _lenses, _published_as, _reviews, armed,
)
from tests.hermes_cli.test_kanban_delivery_exact_base_arm import (  # noqa: F401  (exact is a fixture)
    _fetches, _setting, exact,
)
from tests.hermes_cli.test_kanban_delivery_github import REPO
from tests.hermes_cli.test_kanban_delivery_publish_step import _approved, _events, _tick
from tests.hermes_cli.test_kanban_delivery_review_cards import _raw, _state, world  # noqa: F401  (world is a fixture)
from tests.hermes_cli.test_kanban_delivery_review_return import BUILDER, _governed, _ready, _registered
from tests.hermes_cli.test_release_guards_batch import _g3

# The source card's update cards: the children its kernel delivery_updated events name (operator note 3).
UPDATES = ("SELECT * FROM tasks WHERE id IN (SELECT json_extract(payload, '$.update_task_id') FROM task_events "
           "WHERE task_id = ? AND kind = 'delivery_updated') ORDER BY id")
# The source card's task_links children on an update branch, read past the events: the independent count.
ON_UPDATE_BRANCHES = ("SELECT * FROM tasks WHERE id IN (SELECT child_id FROM task_links WHERE parent_id = ?) "
                      "AND branch_name LIKE 'delivery-update/%' ORDER BY id")
# H's continuation: the source card's task_links child on H's return branch.
CONTINUED = ("SELECT * FROM tasks WHERE id IN (SELECT child_id FROM task_links WHERE parent_id = ?) "
             "AND branch_name LIKE 'delivery-return/%'")
GOVERNANCE = ("execution_tier", "owner_receipt_bound", "provider_override", "model_override", "reasoning_effort")


def _approved_head(scene, name="feature"):
    """An approved head H of a source card of the project whose primary folder is the board repository, its review
    cards done and approved by every lens: the next pass arms it. Returns the card, H and the reviewed base."""
    kb, root, repo, gh = scene
    tid, head = _ready(scene, name=name)
    _tick(kb)
    _done(kb.kanban_db_path())
    _reviews(gh, *_lenses(kb, head))
    return tid, head, _git(repo, "rev-parse", "main")


def _main_moves(scene, path="docs/other.md", text="other\n", where="github") -> str:
    """main moves on with one commit writing ``path``. On GitHub only (``where`` "github"): made in a separate
    clone of GitHub's git origin and pushed there, so the registered checkout does not hold it; or ("local") made
    in the registered checkout's main and pushed. GitHub's main answer names it."""
    kb, root, repo, gh = scene
    checkout = repo
    if where == "github":
        checkout = repo.parent / f"elsewhere-{len(list(repo.parent.glob('elsewhere-*')))}"
        subprocess.run(["git", "clone", "-q", "-b", "main", str(gh["bare"]), str(checkout)], check=True,
                       capture_output=True)
    (checkout / path).parent.mkdir(parents=True, exist_ok=True)
    (checkout / path).write_text(text, encoding="utf-8")
    _git(checkout, "add", path)
    _git(checkout, "commit", "-q", "-m", f"main writes {path}")
    _git(checkout, "push", "-q", str(gh["bare"]), "HEAD:refs/heads/main")
    gh["main"] = _git(checkout, "rev-parse", "HEAD")
    return gh["main"]


def _moved_on(scene, **change):
    """H approved, then main moves on with ``change`` and the setting is exactly true: the next pass finds main
    neither the reviewed base nor in H, so the arm sends nothing and records base_moved."""
    kb, root, repo, gh = scene
    tid, head, base = _approved_head(scene)
    main = _main_moves(scene, **change)
    _setting(root, "true")
    return tid, head, base, main


def _held(repo, commit) -> bool:
    """Whether the registered checkout's git holds ``commit``."""
    return subprocess.run(["git", "-C", str(repo), "cat-file", "-e", f"{commit}^{{commit}}"],
                          capture_output=True).returncode == 0


def _parents(repo, commit) -> list:
    return _git(repo, "rev-list", "--parents", "-n", "1", commit).split()[1:]


def _checkout(repo) -> tuple:
    """What a checkout shows: the repository's HEAD, its status and its worktrees."""
    return (_git(repo, "rev-parse", "HEAD"), _git(repo, "status", "--porcelain"),
            _git(repo, "worktree", "list", "--porcelain"))


def _branches(repo) -> set:
    return set(_git(repo, "for-each-ref", "--format=%(refname:short)", "refs/heads/").split())


def _head(repo, card) -> str:
    """The commit at the update card's branch: the updated head."""
    return _git(repo, "rev-parse", f"refs/heads/{card['branch_name']}")


def _task(kb, card) -> dict:
    [task] = _raw(kb.kanban_db_path(), "SELECT * FROM tasks WHERE id = ?", card)
    return task


def _update(kb, tid) -> dict:
    """The source card's one update card, as the kernel's delivery_updated event names it."""
    [card] = _raw(kb.kanban_db_path(), UPDATES, tid)
    return card


def _updated(kb, tid) -> list:
    """The payloads of the source card's delivery_updated events."""
    return [payload for kind, payload in _events(kb, tid) if kind == "delivery_updated"]


def _handovers(kb, card) -> list:
    """The payloads of the card's review_requested events: the review handovers the kernel recorded."""
    return [json.loads(row["payload"]) for row in _raw(
        kb.kanban_db_path(), "SELECT payload FROM task_events WHERE task_id = ? AND kind = 'review_requested' "
        "ORDER BY id", card)]


def _hand_over(scene, card, builder=IMPLEMENTER) -> bool:
    """The update card's builder: the dispatcher claims it at its branch, and the builder hands the unchanged
    updated head over through the normal completion, so the kernel nominates its reviewer."""
    kb, root, repo, gh = scene
    (root / "profiles" / builder).mkdir(parents=True, exist_ok=True)
    _tick(kb)
    claimed = _task(kb, card)
    assert (claimed["status"], claimed["assignee"]) == ("running", builder)
    assert _git(Path(claimed["workspace_path"]), "rev-parse", "HEAD") == _head(repo, claimed)
    conn = kb.connect()
    try:
        return kb.complete_task(conn, card, summary="brought up to date on main, unchanged",
                                expected_run_id=kb.get_task(conn, card).current_run_id)
    finally:
        conn.close()


def _approve_card(kb, card) -> bool:
    """The reviewer claims the card out of review and approves it."""
    conn = kb.connect()
    try:
        return _approve(kb, conn, card)
    finally:
        conn.close()


def _publish(kb, gh, tid, head, number):
    """What the publish stores for ``head`` of ``tid`` as pull request ``number``: the ledger and its record of
    the repository, open at that head on GitHub, and delivery_bound to the card's recorded base, as the
    approval facts give it."""
    db = kb.kanban_db_path()
    base = _task(kb, tid)["base_commit"]
    _published_as(db, gh, tid, head, number)
    _raw(db, "INSERT INTO task_events (task_id, kind, payload, created_at) VALUES (?, 'delivery_bound', ?, 0)",
         tid, json.dumps({"source_task_id": tid, "head": head, "base_commit": base}))


def _lands(scene, head, number) -> dict:
    """GitHub's merge of pull request ``number``, made with real git on the repository's main during the merge
    request and pushed to GitHub's git origin: a two-parent commit with ``head`` second. GitHub then shows it
    merged as that commit, and main at it."""
    kb, root, repo, gh = scene
    landed = {}

    def merge():
        _git(repo, "merge", "-q", "--no-ff", "--no-edit", head)
        landed["merge"] = gh["main"] = gh["ended"][number] = _git(repo, "rev-parse", "HEAD")
        _git(repo, "push", "-q", str(gh["bare"]), "main")

    gh["hooks"]["PUT"] = merge
    return landed


def _closes(gh) -> list:
    return [write for write in gh["writes"] if write[0] == "PATCH"]


def _returned(kb, repo, gh, tid, head, main, reason, branches):
    """H went back to its builder: one continuation card for the builder on a branch at H, no update card, no
    card in review and no other branch, nothing merged, and the row returned for changes. The card's body says why
    and _continuation's own event names it: no event kind the prompt does not name."""
    db = kb.kanban_db_path()
    [card] = _raw(db, CONTINUED, tid)
    branch = f"delivery-return/{tid}-{head[:12]}"
    assert (card["assignee"], card["base_commit"], card["branch_name"]) == (IMPLEMENTER, head, branch)
    assert _git(repo, "rev-parse", f"refs/heads/{branch}") == head and _branches(repo) == branches | {branch}
    assert (_raw(db, UPDATES, tid), _raw(db, ON_UPDATE_BRANCHES, tid)) == ([], [])
    assert _raw(db, "SELECT id FROM tasks WHERE status = 'review'") == []
    assert (_arms(gh), _state(kb, head)) == ([], "returned_for_changes")
    assert {"conflict": "approved head conflicts", "paths": "another set of paths"}[reason] in card["body"]
    assert f"main is at {main}" in card["body"]
    followups = _raw(db, "SELECT payload FROM task_events WHERE task_id = ? "
                         "AND kind = 'review_followup_recorded'", tid)
    assert [json.loads(event["payload"])["followup_task_id"] for event in followups] == [card["id"]]
    assert not {kind for kind, _ in _events(kb, tid)} & {"delivery_updated", "delivery_update_returned"}


def _missing(monkeypatch, miss) -> dict:
    """While failing["now"] is true, the fenced fetch of main fails as a network error does ("fetch"), or the read
    of the project registry misses the source card's project folder ("folder")."""
    from hermes_cli import kanban_delivery_github as transport
    from hermes_cli import projects_db

    failing, fetch_main, get_project = {"now": True}, transport.GitHubTransport.fetch_main, projects_db.get_project

    def fetch(self, checkout):
        if miss == "fetch" and failing["now"]:
            raise transport.GitHubTransportError("network_error")
        return fetch_main(self, checkout)

    def project(registry, name):
        if miss == "folder" and failing["now"]:
            raise sqlite3.OperationalError("unable to open database file")
        return get_project(registry, name)

    monkeypatch.setattr(transport.GitHubTransport, "fetch_main", fetch)
    monkeypatch.setattr(projects_db, "get_project", project)
    return failing


def _governed_update(scene, tier):
    """A governed source card of execution tier ``tier``, owner receipt-bound and sealed on its builder's admitted
    route (raphael-claude-worker's), approved; main moves on GitHub and one pass makes its update card. Returns the
    source, H, the update card, the builder's model and effort."""
    kb, root, repo, gh = scene
    tid, head, base = _approved_head(scene)
    model, effort = _governed(kb, tid, tier)
    _raw(kb.kanban_db_path(), "UPDATE tasks SET owner_receipt_bound = 1 WHERE id = ?", tid)
    _main_moves(scene)
    _setting(root, "true")
    _tick(kb)
    return tid, head, _update(kb, tid), model, effort


def _fix(workspace, path="src/impl/fix.py", text="fixed = True\n") -> None:
    """The continuation's own change on top of H: one commit writing ``path``."""
    (workspace / path).write_text(text, encoding="utf-8")
    _git(workspace, "add", path)
    _git(workspace, "commit", "-q", "-m", f"fix: {path}")


def _continued(scene, link="parent", build=_fix):
    """An approved head H went back to its builder as every return sends it: its row returned for changes, and its
    continuation card made by _continuation on the return branch at H, so H is that card's recorded base. The card is
    the source card's task_links child ("parent"), or is linked to it only by the owner's replacement record
    ("replacement"). Its builder commits ``build`` on top of H and hands it over, its own review approves it, and it is
    published as pull request 42 bound to H, its review cards done and approved by every lens: the next pass arms it.
    Returns the source card, H, H's reviewed base, the continuation card and its head."""
    from hermes_cli import kanban_delivery as kd

    kb, root, repo, gh = scene
    db = kb.kanban_db_path()
    tid, head, base = _approved_head(scene)
    branch = f"delivery-return/{tid}-{head[:12]}"
    _git(repo, "branch", branch, head)
    conn = kb.connect()
    try:
        with kb.write_txn(conn):
            card = kd._continuation(conn, db, {"source_task_id": tid, "pull_request_head": head},
                                    "Changes requested.", (repo, branch))["rework"]
            conn.execute("UPDATE kanban_deliveries SET pull_request_state = 'returned_for_changes' "
                         "WHERE source_task_id = ?", (tid,))
            if link == "replacement":
                conn.execute("DELETE FROM task_links WHERE parent_id = ? AND child_id = ?", (tid, card))
                kb._append_event(conn, tid, "owner_project_plan_change", {"replacement_task_id": card})
    finally:
        conn.close()
    (root / "profiles" / IMPLEMENTER).mkdir(parents=True, exist_ok=True)
    _tick(kb)
    claimed = _task(kb, card)
    assert (claimed["status"], claimed["assignee"], claimed["base_commit"]) == ("running", IMPLEMENTER, head)
    workspace = Path(claimed["workspace_path"])
    assert _git(workspace, "rev-parse", "HEAD") == head
    build(workspace)
    fixed = _git(workspace, "rev-parse", "HEAD")
    conn = kb.connect()
    try:
        assert kb.complete_task(conn, card, summary="fixed on top of H",
                                expected_run_id=kb.get_task(conn, card).current_run_id) is True
    finally:
        conn.close()
    assert _approve_card(kb, card) is True
    _publish(kb, gh, card, fixed, 42)
    _tick(kb)
    _done(db)
    _reviews(gh, *_lenses(kb, fixed), number=42)
    return tid, head, base, card, fixed


def _up_to_date(scene, tid, head, base, cont, fixed, main, held):
    """The continuation's delivery, which the arm left base_moved on its reviewed base H, was brought up to date on
    main: no return, and one update card for its builder on a branch at the updated head, with the continuation's head
    and main as its parents and main as its recorded base, changing the whole chain's paths, H's and the continuation's
    own. The row is updated and nothing merged; the delivery_updated event names the chain's start, H's reviewed base,
    as the approved base beside H. H's own row stays as the kernel left it, ``held``, with no update of its own."""
    kb, root, repo, gh = scene
    db = kb.kanban_db_path()
    assert [payload["reviewed_base"] for kind, payload in _events(kb, cont) if kind == "delivery_base_moved"] == [head]
    assert ([card["body"] for card in _raw(db, CONTINUED, cont)], _state(kb, fixed)) == ([], "updated")
    card = _update(kb, cont)
    update = _head(repo, card)
    assert (card["status"], card["assignee"], card["head_commit"], card["base_commit"]) == (
        "ready", IMPLEMENTER, None, main)
    assert _parents(repo, update) == [fixed, main]
    assert set(_git(repo, "diff", "--name-only", main, update).split()) == {"src/impl/feature.py", "src/impl/fix.py"}
    assert [(payload["update_task_id"], payload["update_head"], payload["observed_main"], payload["reviewed_base"],
             payload["approved_base"]) for payload in _updated(kb, cont)] == [(card["id"], update, main, head, base)]
    assert [task["id"] for task in _raw(db, ON_UPDATE_BRANCHES, cont)] == [card["id"]]
    assert (_arms(gh), _state(kb, head), _updated(kb, tid)) == ([], held, [])


def test_a_moved_branch_gets_one_update_card_based_on_current_main(exact):
    """The update card is the original builder's card, ready, on a branch at the updated head, with main as its
    recorded base; that head has H and the current main as its parents. Nothing is in review yet, the kernel's
    delivery_updated event names the card, and no checkout moved."""
    kb, root, repo, gh = exact
    tid, head, base, main = _moved_on(exact)
    before = _checkout(repo)

    _tick(kb)

    card = _update(kb, tid)
    update = _head(repo, card)
    assert (card["status"], card["assignee"], card["head_commit"], card["base_commit"]) == (
        "ready", IMPLEMENTER, None, main)
    assert _parents(repo, update) == [head, main]
    assert _git(repo, "rev-parse", f"{update}^{{tree}}") == _git(
        repo, "merge-tree", "--write-tree", main, head).splitlines()[0]
    assert set(_git(repo, "diff", "--name-only", main, update).split()) == {"src/impl/feature.py"}
    assert (_handovers(kb, card["id"]), _raw(kb.kanban_db_path(), "SELECT id FROM tasks WHERE status = 'review'")) == (
        [], [])
    assert (_state(kb, head), _arms(gh)) == ("updated", [])
    assert [(payload["update_task_id"], payload["update_head"], payload["observed_main"], payload["reviewed_base"])
            for payload in _updated(kb, tid)] == [(card["id"], update, main, base)]
    assert [task["id"] for task in _raw(kb.kanban_db_path(), ON_UPDATE_BRANCHES, tid)] == [card["id"]]
    assert _checkout(repo) == before and not (repo / ".worktrees" / card["id"]).exists()


def test_an_updated_head_never_merges_without_its_own_approval(exact):
    """Neither the source's approval of H nor the update card itself proves the updated head; its builder's normal
    completion hands it to the reviewer the kernel nominates, and only that card's own approval delivers it."""
    from hermes_cli import kanban_delivery as kd

    kb, root, repo, gh = exact
    tid, head, base, main = _moved_on(exact)
    _tick(kb)
    card = _update(kb, tid)
    update = _head(repo, card)

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
    finally:
        conn.close()
    assert _hand_over(exact, card["id"]) is True
    handed = _task(kb, card["id"])
    assert (handed["status"], handed["assignee"], handed["base_commit"]) == ("review", REVIEWER, main)
    assert [(handover["implementer"], handover["implementer_head_commit"]) for handover in _handovers(
        kb, card["id"])] == [(IMPLEMENTER, update)]
    conn = kb.connect()
    try:
        assert not kd.prove_approval(conn, card["id"], update).allowed  # handed over, not approved yet
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
    base: main holds the change already, so the update changes none of them; or main on GitHub never held the
    reviewed base, so the update would bring a path review never saw, outside the card's owned paths."""
    kb, root, repo, gh = exact
    if change == "main_rewritten_under_the_base":  # the reviewed base only the registered checkout holds
        (repo / "docs").mkdir(exist_ok=True)
        (repo / "docs" / "reviewed.md").write_text("in the reviewed base\n", encoding="utf-8")
        _git(repo, "add", "docs/reviewed.md")
        _git(repo, "commit", "-q", "-m", "main writes docs/reviewed.md")
    tid, head, base = _approved_head(exact)
    if change == "main_holds_the_change":
        main = _main_moves(exact, "src/impl/feature.py", "ok = True\n")
    else:
        main = _main_moves(exact)
    _setting(root, "true")
    branches = _branches(repo)

    _tick(kb)

    _returned(kb, repo, gh, tid, head, main, "paths", branches)


def test_a_conflicting_update_gets_a_card_on_main_with_the_conflict_marked(exact):
    """Main moves on the approved change's own file, so merging main and H conflicts in it, inside the approved paths
    and the card's owned paths: one pass makes one update card for the builder based on main, with no return. Its
    branch head has H and main as its parents and holds git's merged tree with the conflict marked, and the kernel's
    delivery_updated event names the conflicted file."""
    kb, root, repo, gh = exact
    tid, head, base, main = _moved_on(exact, path="src/impl/feature.py", text="ok = False\n")

    _tick(kb)

    db = kb.kanban_db_path()
    assert (_state(kb, head), [task["branch_name"] for task in _raw(db, CONTINUED, tid)], _arms(gh)) == (
        "updated", [], [])
    card = _update(kb, tid)
    update = _head(repo, card)
    assert (card["status"], card["assignee"], card["head_commit"], card["base_commit"]) == (
        "ready", IMPLEMENTER, None, main)
    assert _parents(repo, update) == [head, main]
    marked = _git(repo, "show", f"{update}:src/impl/feature.py").splitlines()
    assert [line for line in marked if line.startswith(("<<<<<<<", "=======", ">>>>>>>"))] == [
        f"<<<<<<< {main}", "=======", f">>>>>>> {head}"] and "ok = False" in marked
    assert set(_git(repo, "diff", "--name-only", main, update).split()) == {"src/impl/feature.py"}
    assert [(payload["update_task_id"], payload["update_head"], payload["observed_main"], payload["reviewed_base"],
             payload["conflicts"]) for payload in _updated(kb, tid)] == [
        (card["id"], update, main, base, ["src/impl/feature.py"])]
    assert [task["id"] for task in _raw(db, ON_UPDATE_BRANCHES, tid)] == [card["id"]]


def test_a_conflicting_update_goes_back_to_the_builder_with_no_card_in_review(exact):
    """The conflict goes back to the builder as that update card, ready for the builder and not in review: its body
    names the conflicted file and asks for the marked conflicts in it resolved and nothing else changed. No checkout
    moved, no branch but the update's was made, and another pass makes no other card and no return."""
    kb, root, repo, gh = exact
    tid, head, base, main = _moved_on(exact, path="src/impl/feature.py", text="ok = False\n")
    before, branches = _checkout(repo), _branches(repo)

    _tick(kb)

    db = kb.kanban_db_path()
    assert (_state(kb, head), [task["branch_name"] for task in _raw(db, CONTINUED, tid)]) == ("updated", [])
    card = _update(kb, tid)
    assert (card["status"], card["assignee"]) == ("ready", IMPLEMENTER)
    assert (f"main is at {main}. Merging it with the approved change conflicts in these files: src/impl/feature.py. "
            f"This card's branch {card['branch_name']} at {_head(repo, card)} is on current main with the conflict "
            "marked. Resolve the marked conflicts in those files only, change nothing else, and hand over for review."
            ) in card["body"] and "over unchanged" not in card["body"]
    assert _raw(db, "SELECT id FROM tasks WHERE status = 'review'") == []
    assert _checkout(repo) == before and _branches(repo) == branches | {card["branch_name"]}
    _tick(kb)
    assert ([task["id"] for task in _raw(db, ON_UPDATE_BRANCHES, tid)], _raw(db, CONTINUED, tid)) == ([card["id"]], [])


@pytest.mark.parametrize("conflict", ["added_on_both_sides", "deleted_under_the_base"])
def test_a_conflict_outside_the_approved_paths_goes_back_to_the_builder(exact, conflict):
    """The reviewed base is a commit only the registered checkout holds, and it changes docs/kept.md, which main on
    GitHub changes too and the approved change never touches: merging main and H conflicts in that file, outside the
    approved paths, so it goes back to the builder as a conflict and no update card is made. Both add the file, so the
    merged tree changes it relative to main as well; or the base deletes it while main changes it, so git leaves
    main's version in the tree, and only the conflicted file lies outside the approved paths."""
    kb, root, repo, gh = exact
    if conflict == "deleted_under_the_base":
        _main_moves(exact, "docs/kept.md", "kept\n", where="local")
        _git(repo, "rm", "-q", "docs/kept.md")
    else:
        (repo / "docs").mkdir(exist_ok=True)
        (repo / "docs" / "kept.md").write_text("in the reviewed base\n", encoding="utf-8")
        _git(repo, "add", "docs/kept.md")
    _git(repo, "commit", "-q", "-m", "the reviewed base changes docs/kept.md")
    tid, head, base = _approved_head(exact)
    main = _main_moves(exact, "docs/kept.md", "changed on main\n")
    _setting(root, "true")
    branches = _branches(repo)

    _tick(kb)

    _returned(kb, repo, gh, tid, head, main, "conflict", branches)


def test_a_continued_delivery_is_brought_up_to_date_when_main_moves(exact):
    """H went back to its builder, and its continuation, reviewed on H, is approved and delivered on top. Main moves
    on an unrelated file, so the arm records base_moved. The approved change runs from where the chain started, H's
    reviewed base, which main holds: one pass makes one update card based on main, with no return. H's own pull
    request was replaced by its child's when that was published."""
    kb, root, repo, gh = exact
    tid, head, base, cont, fixed = _continued(exact)
    main = _main_moves(exact)
    _setting(root, "true")

    _tick(kb)

    _up_to_date(exact, tid, head, base, cont, fixed, main, held="replaced")


def test_a_replacement_that_continues_a_delivered_head_is_brought_up_to_date(exact):
    """The same, with the continuation linked to the delivered card only by the owner's replacement record, as a
    replacement made through the conversation is, and by no task_links parent. H's own row stays returned for
    changes."""
    kb, root, repo, gh = exact
    tid, head, base, cont, fixed = _continued(exact, link="replacement")
    assert _raw(kb.kanban_db_path(), "SELECT parent_id FROM task_links WHERE child_id = ?", cont) == []
    main = _main_moves(exact)
    _setting(root, "true")

    _tick(kb)

    _up_to_date(exact, tid, head, base, cont, fixed, main, held="returned_for_changes")


def test_a_continued_delivery_whose_chain_does_not_start_on_main_goes_back(exact):
    """H's reviewed base is a commit only the registered checkout holds. GitHub's main holds its change as another
    commit, as a squash merge leaves it, so the update would change exactly the chain's paths; but the chain does not
    start in main's history, so it goes back to the builder."""
    kb, root, repo, gh = exact
    (repo / "docs").mkdir(exist_ok=True)
    (repo / "docs" / "reviewed.md").write_text("in the reviewed base\n", encoding="utf-8")
    _git(repo, "add", "docs/reviewed.md")
    _git(repo, "commit", "-q", "-m", "the reviewed base writes docs/reviewed.md")
    tid, head, base, cont, fixed = _continued(exact)
    _main_moves(exact, "docs/reviewed.md", "in the reviewed base\n")
    main = _main_moves(exact)
    _setting(root, "true")
    branches = _branches(repo)

    _tick(kb)

    _returned(kb, repo, gh, cont, fixed, main, "paths", branches)


def test_a_continued_update_that_brings_an_unapproved_path_goes_back(exact):
    """The continuation's builder brings H up to date by merging main, which added src/impl/shared.py after H's
    reviewed base, and its fix drops that file again. Neither review saw that path, as neither change touches it
    relative to its own base, and the chain does not change it from its start; yet the update would delete it from
    main: a path outside the chain's approved paths, so it goes back to the builder."""
    kb, root, repo, gh = exact

    def merges_main_and_drops_its_file(workspace):
        _main_moves(exact, "src/impl/shared.py", "shared = 1\n", where="local")
        _git(workspace, "merge", "-q", "--no-edit", "main")
        _git(workspace, "rm", "-q", "src/impl/shared.py")
        _fix(workspace)

    tid, head, base, cont, fixed = _continued(exact, build=merges_main_and_drops_its_file)
    main = _main_moves(exact)
    _setting(root, "true")
    branches = _branches(repo)

    _tick(kb)

    _returned(kb, repo, gh, cont, fixed, main, "paths", branches)


def test_the_old_pull_request_closes_once_the_update_is_published(exact):
    kb, root, repo, gh = exact
    tid, head, base, main = _moved_on(exact)
    _tick(kb)
    card = _update(kb, tid)
    update = _head(repo, card)
    _tick(kb)
    assert _closes(gh) == []  # the update is not published yet: pull request 41 stays open

    assert _hand_over(exact, card["id"]) is True
    assert _approve_card(kb, card["id"]) is True
    _publish(kb, gh, card["id"], update, 42)
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
    assert [update["id"] for update in _raw(db, ON_UPDATE_BRANCHES, tid)] == [card["id"]]
    assert [payload["update_head"] for payload in _updated(kb, tid)] == [_head(repo, card)]
    assert (_state(kb, head), _raw(db, CONTINUED, tid)) == ("updated", [])
    assert (_task(kb, card["id"])["status"], _handovers(kb, card["id"])) == ("ready", [])


def test_two_changes_merge_before_one_release_and_both_pass_the_release_guards(exact):
    """Acceptance: two cards are approved on the same main T0. The first merges as M1 on GitHub. The second,
    reviewed on T0, is moved: the arm records base_moved and its update card carries the update on M1, read
    with the fenced fetch. Its builder hands it over, its own review approves it, it is published and merged as M2,
    and it joins the waiting release batch after the first; release_guards.prepare gives G3 ok for T0..M2
    against the real repository."""
    from hermes_cli import release_ledger
    from hermes_cli.release_guards import RecordedMerge

    kb, root, repo, gh = exact
    db = kb.kanban_db_path()
    t0 = _git(repo, "rev-parse", "main")
    first, first_head = _ready(exact)
    second, second_head = _approved(kb, repo, "second")
    _registered(db, second, repo)
    _publish(kb, gh, second, second_head, 43)
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
    update = _head(repo, card)
    assert (_state(kb, first_head), _state(kb, second_head), _parents(repo, update)) == (
        "merged", "updated", [second_head, m1])
    assert len(_arms(gh)) == 1
    assert _hand_over(exact, card["id"]) is True
    assert _approve_card(kb, card["id"]) is True
    _publish(kb, gh, card["id"], update, 42)
    _tick(kb)  # CI is done on the update: its review cards; and 43, which the update replaces, closes
    _done(db)
    _reviews(gh, *_lenses(kb, update), number=42)
    second_landed = _lands(exact, update, 42)
    _tick(kb)  # main M1 is in the update's reviewed head: it merges as M2
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


@pytest.mark.parametrize("tier", ["routine", "deep"])
def test_a_governed_sources_update_card_keeps_its_governance_for_its_builder(exact, tier):
    """Finding 1 of 824: the update card of an owner receipt-bound source of execution tier ``tier`` is its
    builder's governed card, as _continuation makes one: the source's tier, receipt binding, risk tier, review
    requirement and owned paths, and the builder's route locked for the builder; ready, not in review."""
    kb, root, repo, gh = exact
    tid, head, card, model, effort = _governed_update(exact, tier)
    source = _task(kb, tid)

    assert (card["assignee"], card["status"], _handovers(kb, card["id"])) == (BUILDER, "ready", [])
    assert [card[key] for key in GOVERNANCE] == [tier, 1, "anthropic", model, effort]
    assert [card[key] for key in ("risk_tier", "requires_review", "owned_paths")] == [
        source[key] for key in ("risk_tier", "requires_review", "owned_paths")]
    assert card["model_policy_lock"] and kb.policy_lock_error(
        card["model_policy_lock"], BUILDER, "anthropic", model, effort, tier) is None
    with kb.connect_closing() as conn:
        assert kb.task_is_policy_governed(card) and kb.route_authority_error(conn, card["id"]) is None


def test_an_unapproved_route_edit_of_a_governed_update_card_is_refused(exact):
    """Finding 1 of 824: the kernel refuses an unapproved route on the governed update card, and the database
    keeps its approved route."""
    kb, root, repo, gh = exact
    tid, head, card, model, effort = _governed_update(exact, "deep")
    conn = kb.connect()
    try:
        with pytest.raises(RuntimeError, match="owner-governed"):
            kb.set_model_override(conn, card["id"], "unapproved-test-model", "openrouter")
    finally:
        conn.close()
    assert [_task(kb, card["id"])[key] for key in GOVERNANCE] == ["deep", 1, "anthropic", model, effort]


def test_a_governed_update_card_is_handed_over_under_the_kernels_route_authority(exact):
    """Finding 1 of 824: the governed update card's builder hands the unchanged updated head over through the
    normal completion. The kernel nominates the reviewer and holds the card in review on a route its authority
    admits, and an unapproved route edit is still refused."""
    kb, root, repo, gh = exact
    tid, head, card, model, effort = _governed_update(exact, "deep")

    assert _hand_over(exact, card["id"], builder=BUILDER) is True

    handed = _task(kb, card["id"])
    assert (handed["status"], handed["assignee"], handed["execution_tier"]) == ("review", REVIEWER, "deep")
    assert [(handover["implementer"], handover["implementer_head_commit"]) for handover in _handovers(
        kb, card["id"])] == [(BUILDER, _head(repo, card))]
    with kb.connect_closing() as conn:
        assert kb.task_is_policy_governed(handed) and kb.route_authority_error(conn, card["id"]) is None
        with pytest.raises(RuntimeError, match="owner-governed"):
            kb.set_model_override(conn, card["id"], "unapproved-test-model", "openrouter")


def test_an_update_card_enters_review_only_through_a_current_reviewer_nomination(exact, monkeypatch):
    """Finding 1 of 824: the policy nominates no reviewer now. The update card is still made for its builder, not
    handed to the reviewer of the source's history; its builder's completion parks it as the kernel's handover
    does when nobody is nominated, so no reviewer holds it and nothing can approve it."""
    from hermes_cli import kanban_delivery as kd

    kb, root, repo, gh = exact
    tid, head, base, main = _moved_on(exact)
    monkeypatch.setattr(kb, "policy_resolved_reviewer", lambda: None)

    _tick(kb)

    card = _update(kb, tid)
    assert (card["status"], card["assignee"], _handovers(kb, card["id"])) == ("ready", IMPLEMENTER, [])
    _hand_over(exact, card["id"])
    handed = _task(kb, card["id"])
    assert (handed["status"], handed["assignee"]) == ("review", None)  # parked unassigned, never self-review
    with kb.connect_closing() as conn:
        assert not kd.prove_approval(conn, card["id"], _head(repo, card)).allowed
    assert _arms(gh) == []


@pytest.mark.parametrize("where", ["github", "local"])
def test_main_is_fetched_once_and_the_update_is_bound_to_it(exact, monkeypatch, where):
    """Finding 2 of 824: main moved with a docs-only commit, on GitHub only (made in a separate clone and pushed to
    its git origin) or, as a control, in the registered checkout too. One pass fetches main once into
    refs/remotes/origin/main, the checkout's HEAD, branches and status unmoved, and the update card is bound to
    that commit; later passes fetch nothing more."""
    kb, root, repo, gh = exact
    tid, head, base, main = _moved_on(exact, where=where)
    assert _held(repo, main) is (where == "local")
    before, branches, fetched = _checkout(repo), _branches(repo), _fetches(monkeypatch)

    _tick(kb)
    _tick(kb)
    _tick(kb)

    card = _update(kb, tid)
    assert (fetched, _git(repo, "rev-parse", "refs/remotes/origin/main")) == ([str(repo)], main)
    assert (_parents(repo, _head(repo, card)), card["base_commit"], _state(kb, head)) == ([head, main], main, "updated")
    assert [payload["observed_main"] for payload in _updated(kb, tid)] == [main]
    assert (_checkout(repo), _branches(repo)) == (before, branches | {card["branch_name"]})


@pytest.mark.parametrize("miss", ["fetch", "folder"])
def test_main_is_read_again_when_it_advances_while_the_update_waits(exact, monkeypatch, miss):
    """Finding 2 of 824: the first pass cannot fetch main, or misses the project folder, so the row stays
    base_moved and nothing is made. main advances again; the next pass reads and fetches it anew and binds the
    update to that main, not to the main the arm recorded."""
    kb, root, repo, gh = exact
    tid, head, base, first = _moved_on(exact)
    failing = _missing(monkeypatch, miss)
    db = kb.kanban_db_path()

    _tick(kb)

    assert (_raw(db, UPDATES, tid), _raw(db, ON_UPDATE_BRANCHES, tid), _state(kb, head)) == ([], [], "base_moved")
    second = _main_moves(exact, "docs/later.md", "later\n")
    failing["now"] = False

    _tick(kb)

    card = _update(kb, tid)
    assert (_parents(repo, _head(repo, card)), card["base_commit"], _state(kb, head)) == (
        [head, second], second, "updated")
    assert [payload["observed_main"] for payload in _updated(kb, tid)] == [second]
    moved = [payload["observed_main"] for kind, payload in _events(kb, tid) if kind == "delivery_base_moved"]
    assert moved == [first]


def test_a_public_card_with_the_update_key_is_no_update_card(exact):
    """Finding 3 of 824: an ordinary follow-up of the source created through the public create_task with the key
    update:<source>:<H>:<main> before the pass is no proof of an update: the pass makes the update card and its
    delivery_updated event, and leaves that card as it was."""
    kb, root, repo, gh = exact
    tid, head, base, main = _moved_on(exact, where="local")
    conn = kb.connect()
    try:
        scratch = kb.create_task(conn, title="scratch follow-up", assignee=IMPLEMENTER, parents=[tid],
                                 idempotency_key=f"update:{tid}:{head}:{main}")
    finally:
        conn.close()
    before = _task(kb, scratch)

    _tick(kb)

    card = _update(kb, tid)
    assert card["id"] != scratch and _task(kb, scratch) == before
    assert (card["status"], card["assignee"], _parents(repo, _head(repo, card))) == ("ready", IMPLEMENTER, [head, main])
    assert ([payload["update_task_id"] for payload in _updated(kb, tid)], _state(kb, head)) == ([card["id"]], "updated")


def test_an_unrelated_child_with_the_update_key_closes_no_pull_request(exact):
    """Finding 3 of 824: a separately built and approved child of the source, created through the public
    create_task with the key update:<source>:<H>:<commit>, is published as another pull request. No kernel event
    names it an update card, so pull request 41 of H is not closed and its row stays as it is."""
    kb, root, repo, gh = exact
    db = kb.kanban_db_path()
    tid, head = _ready(exact)
    _tick(kb)
    _setting(root, "true")
    conn = kb.connect()
    try:
        child = kb.create_task(conn, title="build unrelated", assignee=IMPLEMENTER, requires_review=True,
                               owned_paths=["src/impl"], workspace_kind="worktree", workspace_path=str(repo),
                               branch_name="feature/unrelated", parents=[tid],
                               idempotency_key=f"update:{tid}:{head}:{'f' * 40}")
        run = kb.claim_task(conn, child, claimer=f"{IMPLEMENTER}:1")
        workspace, branch = kb._resolve_worktree_workspace(run)
        kb.set_workspace_path(conn, child, workspace)
        kb.set_branch_name(conn, child, branch)
        kb.record_worktree_base(conn, child, workspace)
        (Path(workspace) / "src" / "impl").mkdir(parents=True, exist_ok=True)
        (Path(workspace) / "src" / "impl" / "unrelated.py").write_text("ok = True\n", encoding="utf-8")
        _git(Path(workspace), "add", "src/impl/unrelated.py")
        _git(Path(workspace), "commit", "-q", "-m", "feat: unrelated")
        child_head = _git(Path(workspace), "rev-parse", "HEAD")
        assert kb.complete_task(conn, child, summary="unrelated", expected_run_id=run.current_run_id)
        assert _approve(kb, conn, child) is True
    finally:
        conn.close()
    _publish(kb, gh, child, child_head, 42)
    state = _state(kb, head)

    _tick(kb)
    _tick(kb)

    assert (_closes(gh), _state(kb, head), _arms(gh)) == ([], state, [])
    assert [kind for kind, payload in _events(kb, tid) if kind in ("delivery_replaced", "delivery_updated")] == []


def test_the_update_step_reaches_the_board_and_root_registry_from_a_workers_environment(exact, monkeypatch, tmp_path):
    """Under the exact environment the dispatcher injects into a worker of another board (HERMES_KANBAN_DB and
    HERMES_KANBAN_BOARD pinned to proj-a, HERMES_KANBAN_TASK set, HERMES_HOME a profile home under the root), the
    update step on the default board's file still reaches that board, the root's board register and the
    project registry kb.create_task reads, and makes the one update card there. Its proof is read straight off
    the board files: the independent count."""
    from hermes_cli import kanban_delivery as kd
    from hermes_cli.kanban_delivery_update import update_step

    kb, root, repo, gh = exact
    db = kb.kanban_db_path()
    tid, head, base, main = _moved_on(exact)
    kd.arm_step(db)  # the arm alone, as the dispatcher's pass runs it: H is base_moved
    assert _state(kb, head) == "base_moved"
    kb.create_board("proj-a")
    own_db = root / "kanban" / "boards" / "proj-a" / "kanban.db"
    profile_home = root / "profiles" / IMPLEMENTER
    profile_home.mkdir(parents=True)
    _setting(profile_home, "true")
    _registered(db, tid, repo, home=profile_home)  # projects live in the active profile's projects.db
    conn = kb.connect(board="proj-a")
    try:
        work = kb.create_task(conn, title="other work", assignee=IMPLEMENTER)
        claimed = kb.claim_task(conn, work, claimer=f"{IMPLEMENTER}:1")
        env = _spawned_worker_env(kb, claimed, tmp_path / "work", "proj-a", monkeypatch)
    finally:
        conn.close()
    assert (env["HERMES_KANBAN_DB"], env["HERMES_KANBAN_BOARD"], env["HERMES_KANBAN_TASK"], env["HERMES_HOME"]) == (
        str(own_db), "proj-a", work, str(profile_home))
    _become(env, monkeypatch)

    update_step(db)
    update_step(db)

    assert kb.kanban_home() == root and kb.register_db_path() == root / "kanban" / "board_register.db"
    assert kb.get_register_entry("proj-a") is not None and kb.board_dir("proj-a") / "kanban.db" == own_db
    raw = sqlite3.connect(str(db))
    try:
        events = raw.execute("SELECT payload FROM task_events WHERE task_id = ? AND kind = 'delivery_updated'",
                             (tid,)).fetchall()
        children = raw.execute("SELECT id, branch_name, base_commit FROM tasks WHERE id IN (SELECT child_id "
                               "FROM task_links WHERE parent_id = ?) AND branch_name LIKE 'delivery-update/%'",
                               (tid,)).fetchall()
        (state,) = raw.execute("SELECT pull_request_state FROM kanban_deliveries WHERE source_head = ?",
                               (head,)).fetchone()
    finally:
        raw.close()
    assert (len(events), len(children), state) == (1, 1, "updated")
    assert json.loads(events[0][0])["update_task_id"] == children[0][0]
    assert (_parents(repo, _git(repo, "rev-parse", f"refs/heads/{children[0][1]}")), children[0][2]) == (
        [head, main], main)
    own = sqlite3.connect(str(own_db))
    try:
        assert own.execute("SELECT id FROM tasks").fetchall() == [(work,)]
        assert own.execute("SELECT COUNT(*) FROM task_events WHERE kind LIKE 'delivery%'").fetchone() == (0,)
    finally:
        own.close()
    assert not (profile_home / "kanban.db").exists() and not (profile_home / "kanban").exists()


@pytest.mark.parametrize("value", [None, "false", '"true"'], ids=["absent", "false", "quoted_true"])
def test_with_the_setting_off_a_base_moved_row_stays_where_it_is(exact, monkeypatch, value):
    """Rollback: with the setting anything but exactly true, a row card 2 left base_moved stays exactly as it is,
    and no card, branch, event, fetch or GitHub write is made, however many passes run."""
    kb, root, repo, gh = exact
    tid, head, base = _approved_head(exact)
    main = _main_moves(exact)
    db = kb.kanban_db_path()
    [row] = _raw(db, "SELECT id FROM kanban_deliveries WHERE source_head = ?", head)
    _raw(db, "UPDATE kanban_deliveries SET pull_request_state = 'base_moved' WHERE id = ?", row["id"])
    _raw(db, "INSERT INTO task_events (task_id, kind, payload, created_at) VALUES (?, 'delivery_base_moved', ?, 0)",
         tid, json.dumps({"delivery_id": row["id"], "head": head, "pull_request_number": 41, "repository": REPO,
                          "reviewed_base": base, "observed_main": main, "cards": []}))
    _setting(root, value)
    rows, tasks = (_raw(db, f"SELECT * FROM {table} ORDER BY id") for table in ("kanban_deliveries", "tasks"))
    events, branches, writes, fetched = _events(kb, tid), _branches(repo), list(gh["writes"]), _fetches(monkeypatch)

    _tick(kb)
    _tick(kb)

    assert _raw(db, "SELECT * FROM kanban_deliveries ORDER BY id") == rows and _state(kb, head) == "base_moved"
    assert [task["id"] for task in _raw(db, "SELECT id FROM tasks ORDER BY id")] == [task["id"] for task in tasks]
    assert (_events(kb, tid), _branches(repo), gh["writes"], fetched) == (events, branches, writes, [])
