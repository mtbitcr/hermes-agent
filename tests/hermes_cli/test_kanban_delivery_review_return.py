"""The review return: a published head H whose required review cards are all done, and on which the reviewer
bot's latest review of one of them requests changes, returns its work to the source card's builder through one
continuation card, its branch at H. Real boards and git; GitHub is the real transport with only its exchange and
App token replaced by a table (the world of test_kanban_delivery_auto_merge.py), so every call passes the allowlist."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from tests.hermes_cli import test_kanban_delivery_review_cards as review_cards
from tests.hermes_cli.test_kanban_delivery import (  # noqa: F401  (board is a fixture)
    IMPLEMENTER, _approve, _git, board,
)
from tests.hermes_cli.test_kanban_delivery_auto_merge import (  # noqa: F401  (armed is a fixture)
    OTHER, _arms, _done, _lenses, _published_as, _reviews, armed,
)
from tests.hermes_cli.test_kanban_delivery_github import REPO
from tests.hermes_cli.test_kanban_delivery_publish_step import _events, _tick
from tests.hermes_cli.test_kanban_delivery_review_cards import (  # noqa: F401  (world is a fixture)
    _cards, _raw, _source, _state, world,
)

# The source card's continuation: its task_links child that is none of its review cards.
CONTINUED = ("SELECT * FROM tasks WHERE id IN (SELECT child_id FROM task_links WHERE parent_id = ?) "
             "AND idempotency_key IS NULL")
KEPT = ("owned_paths", "risk_tier", "execution_tier", "requires_review")
BUILDER = "raphael-claude-worker"


def _branch(tid, head):
    """Owner rule 2: the continuation's branch, named from the source card id and H only."""
    return f"delivery-return/{tid}-{head[:12]}"


def _registered(db, tid, folder, home=None, project="delivery-project"):
    """The source card of ``project``, whose primary folder in the project registry of ``home`` (the profile's
    own when None) is ``folder``: the continuation's repository under the owner's rule of round 2."""
    from hermes_cli import projects_db

    with projects_db.connect_closing(home and home / "projects.db") as registry:
        if projects_db.get_project(registry, project) is None:
            projects_db.create_project(registry, name=project, id=project, primary_path=str(folder))
    _raw(db, "UPDATE tasks SET project_id = ? WHERE id = ?", project, tid)


def _ready(world, tier=2, name="feature"):
    """H published, its source card of the project whose primary folder is the repository."""
    kb, root, repo, gh = world
    tid, head = review_cards._ready(world, tier, name)
    _registered(kb.kanban_db_path(), tid, repo)
    return tid, head


def _clone(repo, name, repository=REPO):
    """A clone of the repository beside it, so holding H, its origin naming ``repository``."""
    _git(repo.parent, "clone", "-q", str(repo), name)
    _git(repo.parent / name, "remote", "set-url", "origin", f"https://github.com/{repository}.git")
    return repo.parent / name


def _governed(kb, tid, tier):
    """The source card sealed on its builder's admitted route for execution tier ``tier`` at its risk tier's
    effort, its accepted handover naming raphael-claude-worker as its builder. Returns the model and effort."""
    from hermes_cli.kanban_risk_tier import pinned_reasoning_effort
    from plugins.dashboard_auth.raphael_workspace import model_policy

    db = kb.kanban_db_path()
    route = model_policy.task_assignment_for(BUILDER, "anthropic", tier)
    effort = pinned_reasoning_effort(_raw(db, "SELECT risk_tier FROM tasks WHERE id = ?", tid)[0]["risk_tier"], None)
    _raw(db, "UPDATE tasks SET assignee = ?, execution_tier = ?, provider_override = ?, model_override = ?, "
         "reasoning_effort = ?, model_policy_lock = ? WHERE id = ?", BUILDER, tier, route.provider, route.model,
         effort, kb.mint_policy_lock(BUILDER, route.provider, route.model, effort, tier), tid)
    _raw(db, "UPDATE task_events SET payload = json_set(payload, '$.implementer', ?) "
         "WHERE task_id = ? AND kind = 'review_requested'", BUILDER, tid)
    return route.model, effort


def _run(db, card, started, *findings):
    """A finished run of ``card`` that recorded ``findings``, as kanban_complete's metadata records them."""
    _raw(db, "INSERT INTO task_runs (task_id, profile, status, outcome, started_at, ended_at, metadata) "
         "VALUES (?, 'raphael-verifier', 'done', 'completed', ?, ?, ?)",
         card, started, started + 1, json.dumps({"findings": list(findings)}))


def _returned(kb, tid):
    return [payload for kind, payload in _events(kb, tid)
            if kind == "delivery_review_waiting" and payload["code"] == "review_returned"]


def _branches(repo):
    return _git(repo, "branch", "--list", "wt/*", "delivery-return/*")


def _changes_requested(armed):
    """H's cards done after their recorded runs; the bot's latest review on H of R12 requests changes, as R12's
    own handback recorded, and that of R15 approves H after an earlier request. The title has an internal prefix."""
    kb, root, repo, gh = armed
    db = kb.kanban_db_path()
    tid, head = _ready(armed)
    _raw(db, "UPDATE tasks SET title = 'B03 — build feature' WHERE id = ?", tid)
    _tick(kb)
    cards = {card["responsibility"]: card["id"] for card in _cards(kb)}
    _run(db, cards["R12"], 100, "an older finding")
    _run(db, cards["R12"], 200, "src/impl/feature.py: validate the input", {"severity": "high", "problem": "no check"})
    _run(db, cards["R15"], 300, "a correctness note")
    _done(db)
    _raw(db, "INSERT INTO task_events (task_id, kind, created_at) VALUES (?, 'changes_requested', 0)", cards["R12"])
    _reviews(gh, ("CHANGES_REQUESTED", head, cards["R15"]), ("APPROVED", head, cards["R15"]),
             ("CHANGES_REQUESTED", head, cards["R12"]))
    return tid, head, cards


def _branching(monkeypatch, change):
    """The _run_git seam runs ``change`` once, as it makes the continuation's branch."""
    from hermes_cli import kanban_delivery

    run_git, changes = kanban_delivery._run_git, [change]

    def branching(workdir, *args):
        if args[:1] == ("branch",) and changes:
            changes.pop()()
        return run_git(workdir, *args)

    monkeypatch.setattr(kanban_delivery, "_run_git", branching)


def _checkout_elsewhere(kb, repo, tid, layout):
    """The source card's workspace: its branch checked out by git outside its repository, at an external path,
    there advanced one commit past H, or in an external folder named .worktrees; at an external path or under
    another clone's .worktrees, then removed by native cleanup with H reachable from a remote-tracking ref; or
    in the repository after git moved its Git directory to a separate folder. Returns it."""
    folder = repo.parent / "elsewhere" / (".worktrees" if layout == "dot_worktrees" else "")
    if layout == "other_clone":
        folder = _clone(repo, "clone") / ".worktrees"
    if layout == "separate_git_dir":
        _git(repo, "init", "-q", "--separate-git-dir", str(repo.parent / "gitdata"))
        folder = repo / ".worktrees"
    checkout = folder / tid
    _git(repo, "worktree", "add", str(checkout), _source(kb, tid)["branch_name"])
    _raw(kb.kanban_db_path(), "UPDATE tasks SET workspace_path = ? WHERE id = ?", str(checkout), tid)
    if layout == "advanced":
        _git(checkout, "commit", "--allow-empty", "-qm", "past H")
    if layout in ("removed", "other_clone"):
        _git(repo, "update-ref", f"refs/remotes/origin/delivery/{tid}", _git(checkout, "rev-parse", "HEAD"))
        with kb.connect_closing() as conn:
            kb._cleanup_workspace(conn, tid)
        assert not checkout.exists()
    return checkout


def _claimed_apart(kb, root, repo, tid, head, source):
    """The kernel claims the continuation in a worktree of its own, not ``source``, on its branch at H in the
    project's primary folder, the repository: Git's common directory is the repository's, H the recorded base."""
    (root / "profiles" / IMPLEMENTER).mkdir(parents=True)
    _tick(kb)
    (claimed,) = _raw(kb.kanban_db_path(), CONTINUED, tid)
    workspace = Path(claimed["workspace_path"])
    assert (claimed["status"], claimed["branch_name"], claimed["base_commit"]) == ("running", _branch(tid, head), head)
    assert workspace.resolve() != source.resolve() and _git(workspace, "rev-parse", "HEAD") == head
    assert _git(repo, "rev-parse", f"refs/heads/{_branch(tid, head)}") == head
    common = ("rev-parse", "--path-format=absolute", "--git-common-dir")
    assert _git(workspace, *common) == _git(repo, *common)


def test_a_returned_review_creates_one_claimable_continuation_with_its_findings_and_branch_at_h(armed):
    """Required behaviors 1 and 3 to 5: review_returned once for H and one ready continuation for the builder,
    with the source's paths, tiers and review requirement and the returned card's latest findings, its branch
    at H; nothing is sent to GitHub, a second pass makes nothing, and the claim's recorded base is H."""
    from hermes_cli.owner_workspace import owner_title

    kb, root, repo, gh = armed
    db = kb.kanban_db_path()
    tid, head, cards = _changes_requested(armed)

    _tick(kb)

    (card,) = _raw(db, CONTINUED, tid)
    (source,) = _raw(db, "SELECT * FROM tasks WHERE id = ?", tid)
    assert _returned(kb, tid) == [{"delivery_id": 1, "head": head, "pull_request_number": 41,
                                   "code": "review_returned", "cards": [cards["R12"]], "rework": card["id"]}]
    assert (card["assignee"], card["status"], card["title"]) == (IMPLEMENTER, "ready", owner_title(source["title"]))
    assert (source["title"], card["title"]) == ("B03 — build feature", "build feature")
    assert [card[key] for key in KEPT] == [source[key] for key in KEPT] == ['["src/impl"]', 2, None, 1]
    assert f"head {head} of pull request 41" in card["body"]
    assert f"Review card {cards['R12']}:\nsrc/impl/feature.py: validate the input\n" in card["body"]
    assert '"problem": "no check"' in card["body"] and "an older finding" not in card["body"]
    assert f"Review card {cards['R15']}:" not in card["body"] and "a correctness note" not in card["body"]
    assert card["branch_name"] == _branch(tid, head) and _git(repo, "rev-parse", f"refs/heads/{_branch(tid, head)}") == head
    assert gh["writes"] == [] and {method for method, _ in gh["calls"]} == {"GET"}
    assert _state(kb, head) == "review_cards_created"
    events = _events(kb, tid)

    _tick(kb)

    assert (len(_raw(db, CONTINUED, tid)), _events(kb, tid), gh["writes"]) == (1, events, [])
    (root / "profiles" / IMPLEMENTER).mkdir(parents=True)
    _tick(kb)
    (claimed,) = _raw(db, CONTINUED, tid)
    assert (claimed["id"], claimed["status"], claimed["base_commit"]) == (card["id"], "running", head)
    assert _git(Path(claimed["workspace_path"]), "rev-parse", "HEAD") == head
    assert (_events(kb, tid), gh["writes"]) == (events, [])


def test_the_findings_are_bounded_to_8000_characters_in_all(armed):
    """Required behavior 3: two returned cards' 6,000 characters each are cut to 8,000 in all, the first whole."""
    kb, root, repo, gh = armed
    db = kb.kanban_db_path()
    tid, head = _ready(armed)
    _tick(kb)
    cards = {card["responsibility"]: card["id"] for card in _cards(kb)}
    _run(db, cards["R15"], 100, "¶" * 6000)
    _run(db, cards["R12"], 100, "§" * 6000)
    _done(db)
    _reviews(gh, *[("CHANGES_REQUESTED", head, card) for card in cards.values()])

    _tick(kb)

    (card,) = _raw(db, CONTINUED, tid)
    assert [payload["cards"] for payload in _returned(kb, tid)] == [[cards["R15"], cards["R12"]]]
    findings = card["body"][card["body"].index(f"Review card {cards['R15']}:"):]
    assert len(findings) == 8000 and findings.count("¶") == 6000 and 0 < findings.count("§") < 2000


@pytest.mark.parametrize("case", ["approved", "approved_after_changes", "changes_by_another_account"])
def test_an_approving_review_creates_no_continuation(armed, case):
    """Safety rule 4: the bot's latest review of each card approves H, also after its earlier request for
    changes, and another account's request counts for nothing: H arms once, and nothing returns."""
    kb, root, repo, gh = armed
    db = kb.kanban_db_path()
    tid, head = _ready(armed)
    _tick(kb)
    _done(db)
    security = {card["responsibility"]: card["id"] for card in _cards(kb)}["R12"]
    earlier = [("CHANGES_REQUESTED", head, security)] if case == "approved_after_changes" else []
    later = [("CHANGES_REQUESTED", head, security, "someone")] if case == "changes_by_another_account" else []
    _reviews(gh, *earlier, *_lenses(kb, head), *later)

    _tick(kb)
    _tick(kb)

    assert (_raw(db, CONTINUED, tid), _returned(kb, tid), _branches(repo)) == ([], [], "")
    assert [body["variables"]["expectedHeadOid"] for body in _arms(gh)] == [head]


@pytest.mark.parametrize("case", ["card_open", "moved", "closed"])
def test_nothing_returns_while_a_card_is_open_or_after_the_pull_request_leaves_h(armed, case):
    """Required behavior 1 and safety rule 1: a card still open, or the pull request read again off H or
    closed, returns nothing; a later pass with every card done and H open returns the work once."""
    kb, root, repo, gh = armed
    db = kb.kanban_db_path()
    tid, head, cards = _changes_requested(armed)
    if case == "card_open":
        _raw(db, "UPDATE tasks SET status = 'ready' WHERE id = ?", cards["R15"])
    else:
        gh["hooks"]["GET"] = lambda: gh["ended"].update({41: False}) if case == "closed" else gh.update(pull_head=OTHER)

    _tick(kb)

    left = "" if case == "card_open" else _branch(tid, head)
    assert (_raw(db, CONTINUED, tid), _returned(kb, tid), _branches(repo), gh["writes"]) == ([], [], left, [])
    if case != "card_open":
        assert gh["calls"][-2:] == [("GET", f"/repos/{REPO}/pulls/41/reviews"), ("GET", f"/repos/{REPO}/pulls/41")]
    _done(db)
    gh.update(pull_head=None, ended={})
    _tick(kb)
    _tick(kb)
    assert len(_raw(db, CONTINUED, tid)) == len(_returned(kb, tid)) == 1


@pytest.mark.parametrize("after", ["moved", "closed", "failed"])
def test_nothing_returns_when_the_pull_request_leaves_h_while_the_branch_is_made(armed, monkeypatch, after):
    """Safety rule 1 under owner rule 2: the pull request moves off H, closes or cannot be read while the
    continuation's branch is made. The read after the branch stops the pass: no continuation and no review_returned,
    only the branch at H; a later pass, the pull request open at H again, reuses it and returns the work once."""
    from hermes_cli.kanban_delivery_github import GitHubTransportError

    kb, root, repo, gh = armed
    db = kb.kanban_db_path()
    tid, head, cards = _changes_requested(armed)

    def failed():
        raise GitHubTransportError("network_error")

    _branching(monkeypatch, {"moved": lambda: gh.update(pull_head=OTHER), "closed": lambda: gh["ended"].update(
        {41: False}), "failed": lambda: gh["hooks"].update(PULL=failed)}[after])

    _tick(kb)

    assert (_raw(db, CONTINUED, tid), _returned(kb, tid), _branches(repo), gh["writes"]) == ([], [], _branch(tid, head), [])
    gh.update(pull_head=None, ended={})
    _tick(kb)
    assert (len(_raw(db, CONTINUED, tid)), len(_returned(kb, tid)), gh["writes"]) == (1, 1, [])


@pytest.mark.parametrize("fails", ["branch", "record"])
def test_no_partial_state_survives_a_failure_between_the_reservation_and_the_action(armed, monkeypatch, fails):
    """Safety rule 3 under owner rule 2: git refuses the card's branch, so nothing is recorded or made; or the
    return's record fails after the branch was made at H before the write transaction and the card in it: the
    pass leaves no record and no card, only that branch at H, which no pass deletes. The next pass reuses it
    and returns the work once."""
    from hermes_cli import kanban_delivery

    kb, root, repo, gh = armed
    db = kb.kanban_db_path()
    tid, head, cards = _changes_requested(armed)
    run_git, waiting, seen = kanban_delivery._run_git, kanban_delivery._waiting, []

    def refused(workdir, *args):
        return subprocess.CompletedProcess(args, 128) if args[:1] == ("branch",) else run_git(workdir, *args)

    def failing(conn, row, code, **details):
        if code != "review_returned":
            return waiting(conn, row, code, **details)
        seen.append(_branches(repo))
        raise RuntimeError("the record failed")

    monkeypatch.setattr(kanban_delivery, *(("_run_git", refused) if fails == "branch" else ("_waiting", failing)))

    _tick(kb)

    left = "" if fails == "branch" else _branch(tid, head)
    assert (_raw(db, CONTINUED, tid), _returned(kb, tid), _branches(repo)) == ([], [], left)
    assert len(seen) == (fails == "record") and all(seen)  # the branch existed when the record failed
    assert not left or _git(repo, "rev-parse", f"refs/heads/{left}") == head
    monkeypatch.setattr(kanban_delivery, "_run_git", run_git)
    monkeypatch.setattr(kanban_delivery, "_waiting", waiting)
    _tick(kb)
    _tick(kb)
    (card,) = _raw(db, CONTINUED, tid)
    assert (len(_returned(kb, tid)), card["branch_name"], _branches(repo)) == (1, _branch(tid, head), _branch(tid, head))
    assert _git(repo, "rev-parse", f"refs/heads/{_branch(tid, head)}") == head


@pytest.mark.parametrize("tier", ["routine", "deep"])
def test_a_governed_sources_continuation_is_claimable_on_its_builders_route(armed, tier):
    """Owner rule 1: the source card is governed, of execution tier ``tier``, its builder raphael-claude-worker.
    The review return's continuation has the source's route and tier and the lock minted for the builder, so it
    is ready and not parked, and the kernel claims it with H as its recorded base, on its branch at H."""
    kb, root, repo, gh = armed
    db = kb.kanban_db_path()
    tid, head, cards = _changes_requested(armed)
    model, effort = _governed(kb, tid, tier)

    _tick(kb)

    (card,) = _raw(db, CONTINUED, tid)
    assert [payload["rework"] for payload in _returned(kb, tid)] == [card["id"]]
    assert (card["assignee"], card["status"], card["execution_tier"]) == (BUILDER, "ready", tier)
    assert (card["provider_override"], card["model_override"], card["reasoning_effort"]) == ("anthropic", model, effort)
    assert card["model_policy_lock"] and kb.policy_lock_error(
        card["model_policy_lock"], BUILDER, "anthropic", model, effort, tier) is None
    (root / "profiles" / BUILDER).mkdir(parents=True)
    _tick(kb)
    (claimed,) = _raw(db, CONTINUED, tid)
    assert (claimed["id"], claimed["status"], claimed["base_commit"]) == (card["id"], "running", head)
    assert (claimed["branch_name"], _git(Path(claimed["workspace_path"]), "rev-parse", "HEAD")) == (_branch(tid, head), head)


@pytest.mark.parametrize("layout", ["external", "advanced", "dot_worktrees", "removed", "other_clone", "separate_git_dir"])
def test_a_source_checked_out_outside_its_repository_continues_in_a_worktree_of_its_own(armed, layout):
    """The source card's checkout is linked from outside its repository, there advanced past H, in an external
    folder named .worktrees, removed from an external path or from under another clone, or in a repository whose
    Git directory is separate. The review return makes the branch at H in the primary folder of the source card's
    project; the kernel claims the continuation in a worktree of its own there on that branch, H its base."""
    kb, root, repo, gh = armed
    tid, head, cards = _changes_requested(armed)
    source = _checkout_elsewhere(kb, repo, tid, layout)

    _tick(kb)

    _claimed_apart(kb, root, repo, tid, head, source)


@pytest.mark.parametrize("found", ["at_h", "elsewhere"])
def test_the_branch_named_from_the_source_and_h_is_reused_at_h_and_refuses_elsewhere(armed, found):
    """Owner rule 2: the continuation's branch is named from the source card id and H only. A branch of that
    name already at H is reused as the card's branch; one at another commit refuses the return, so nothing is
    recorded or made on any pass, and that ref stays as it was. No pass deletes a branch."""
    kb, root, repo, gh = armed
    db = kb.kanban_db_path()
    tid, head, cards = _changes_requested(armed)
    main = _git(repo, "rev-parse", "main")
    _git(repo, "branch", _branch(tid, head), head if found == "at_h" else main)

    _tick(kb)
    _tick(kb)

    assert _git(repo, "rev-parse", f"refs/heads/{_branch(tid, head)}") == (head if found == "at_h" else main)
    assert (_branches(repo), gh["writes"]) == (_branch(tid, head), [])
    if found == "elsewhere":
        assert (_raw(db, CONTINUED, tid), _returned(kb, tid)) == ([], [])
        return
    (card,) = _raw(db, CONTINUED, tid)
    assert card["branch_name"] == _branch(tid, head) and [p["rework"] for p in _returned(kb, tid)] == [card["id"]]


def test_the_continuations_own_delivery_closes_the_earlier_pull_request(armed):
    """The continuation, claimed at H, built on and published as pull request 42, closes pull request 41 of H
    once through the merged replacement step, and nothing is armed."""
    kb, root, repo, gh = armed
    db = kb.kanban_db_path()
    tid, head, cards = _changes_requested(armed)
    _tick(kb)
    (root / "profiles" / IMPLEMENTER).mkdir(parents=True)
    _tick(kb)
    (card,) = _raw(db, CONTINUED, tid)
    workspace = Path(card["workspace_path"])
    (workspace / "src" / "impl" / "feature.py").write_text("ok = 3\n", encoding="utf-8")
    _git(workspace, "commit", "-qam", "fix: validate the input")
    continued = _git(workspace, "rev-parse", "HEAD")
    conn = kb.connect()
    try:
        kb.complete_task(conn, card["id"], summary="validated the input",
                         expected_run_id=kb.get_task(conn, card["id"]).current_run_id)
        assert _approve(kb, conn, card["id"]) is True
    finally:
        conn.close()
    _published_as(db, gh, card["id"], continued, 42)

    _tick(kb)
    _tick(kb)

    assert [write for write in gh["writes"] if write[0] == "PATCH"] == [
        ("PATCH", f"/repos/{REPO}/pulls/41", {"state": "closed"})]
    assert [(p["pull_request_number"], p["replaced_by"]) for kind, p in _events(kb, tid)
            if kind == "delivery_replaced"] == [(41, card["id"])]
    assert (_state(kb, head), _arms(gh), len(_returned(kb, tid))) == ("replaced", [], 1)
