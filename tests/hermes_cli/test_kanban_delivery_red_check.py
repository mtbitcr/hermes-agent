"""The red check card: a required check red on the exact published head names its failed tests from the
annotations tests.yml writes on each failed slice job's own check run and returns the work once to its builder
with one continuation card naming its failed jobs and tests; nothing reruns a check (owner decision 2026-10-20).
Real boards and git; GitHub is the real transport with only its exchange and App token replaced by a
table (the world of test_kanban_delivery_review_cards.py), so every call passes the allowlist."""

from __future__ import annotations

from pathlib import Path
from urllib.parse import urlsplit

import pytest

from tests.hermes_cli.test_kanban_delivery import (  # noqa: F401  (board is a fixture)
    IMPLEMENTER, REVIEWER, _become, _git, _park, _rework, _spawned_worker_env, _write_config, board,
)
from tests.hermes_cli.test_kanban_delivery_auto_merge import (  # noqa: F401  (armed is a fixture)
    ELSEWHERE, OTHER, _done, _reviews, armed,
)
from tests.hermes_cli.test_kanban_delivery_github import REPO
from tests.hermes_cli.test_kanban_delivery_publish_step import _approved, _events, _tick
from tests.hermes_cli.test_kanban_delivery_review_cards import (  # noqa: F401  (world is a fixture)
    CHECK, _cards, _published, _raw, _state, world,
)
from tests.hermes_cli.test_kanban_delivery_review_return import (
    BUILDER, CONTINUED, KEPT, UNLOCKABLE, _branch, _branches, _branching, _changes_requested, _checkout_elsewhere,
    _claimed_apart, _clone, _copied, _governed, _held, _legacy, _ready, _registered, _run,
)

FLAKY = ("tests/tools/test_zombie_process_cleanup.py::TestDelegationCleanup::"
         "test_timed_out_child_keeps_relay_session_until_its_turn_exits")
UNLISTED = "tests/hermes_cli/test_kanban_db.py::test_no_policy_lists_this_one"
RERUN = f"/repos/{REPO}/actions/jobs/903/rerun"
REWORK = "SELECT * FROM tasks WHERE title LIKE 'Rework:%'"
IDENTITY = ("SELECT json_extract(payload, '$.identity') AS identity FROM task_events WHERE task_id = ? "
            "AND kind = 'review_followup_recorded' AND json_extract(payload, '$.followup_task_id') = ?")


@pytest.fixture
def red(world, monkeypatch):
    """The world with H's required check red, and GitHub's table holding H's one workflow run 77: the
    summary job 900 (the required check run) failed at "Evaluate job results", slice 1 (901) passed and
    slice 3 (903) failed at its test step, all at gh["attempt"], beside ci.yaml's timing report (904), which
    runs after the summary, at gh["timing"] (status, conclusion). Job 903's check run carries the count
    (gh["count"], else the number of gh["tests"]) and then each of gh["tests"] under the fixed title, beside
    one annotation of another title. Each rerun call is kept in gh["reruns"] and answered gh["rerun"] (201),
    or not at all when None. With gh["closed"], the pull request is closed at H from gh["move_after"] on.
    gh["run"] replaces fields of run 77 (it lists no status unless set), gh["jobs"] fields of each job by its id."""
    kb, root, repo, gh = world
    from hermes_cli import kanban_delivery_github as transport

    table = transport._exchange  # the world's: the pull request and H's check runs
    gh.update(check=("completed", "failure"), attempt=1, tests=[FLAKY], count=None, reruns=[],
              timing=("completed", "success"), rerun=201, closed=False, run={}, jobs={})

    def exchange(method, target, authorization, payload=None):
        status, value = table(method, target, authorization, payload)
        path, head = urlsplit(target).path, gh["head"]
        if path == f"/repos/{REPO}/pulls/41" and gh["closed"] and gh["pull_head"]:
            return 200, dict(value, state="closed", head={"sha": head})
        job = {"head_sha": head, "run_id": 77, "status": "completed", "conclusion": "failure",
               "run_attempt": gh["attempt"]}
        if (method, path) == ("GET", f"/repos/{REPO}/actions/runs"):
            return 200, {"total_count": 1, "workflow_runs": [dict({"id": 77, "head_sha": head}, **gh["run"])]}
        if (method, path) == ("GET", f"/repos/{REPO}/actions/runs/77/jobs"):
            return 200, {"total_count": 4, "jobs": [dict(each, **gh["jobs"].get(each["id"], {})) for each in (
                dict(job, id=900, name=CHECK, steps=[{"name": "Evaluate job results", "conclusion": "failure"}]),
                dict(job, id=901, name="Python tests / Run tests slice 1/12", conclusion="success",
                     steps=[{"name": "Run tests (slice 1/12)", "conclusion": "success"}]),
                dict(job, id=903, name="Python tests / Run tests slice 3/12", steps=[
                    {"name": "Checkout code", "conclusion": "success"},
                    {"name": "Run tests (slice 3/12)", "conclusion": "failure"}]),
                dict(job, id=904, name="CI timing report", status=gh["timing"][0], conclusion=gh["timing"][1],
                     steps=[]))]}
        if (method, path) == ("GET", f"/repos/{REPO}/check-runs/903/annotations"):
            count = len(gh["tests"]) if gh["count"] is None else gh["count"]
            return 200, [{"path": ".github", "annotation_level": "notice", "title": "Failed test",
                          "message": f"count {count}"},
                         *({"path": ".github", "annotation_level": "failure", "title": "Failed test",
                            "message": test, "raw_details": None} for test in gh["tests"]),
                         {"path": ".github", "annotation_level": "failure",
                          "title": "Process completed with exit code 1.", "message": UNLISTED}]
        if (method, path) == ("POST", RERUN):
            gh["reruns"].append(path)
            if gh["rerun"] is None:  # what the real exchange raises when the call times out
                raise transport.GitHubTransportError("network_error")
            return gh["rerun"], None
        return status, value

    monkeypatch.setattr(transport, "_exchange", exchange)
    return world


def _waited(kb, tid, code, db=None):
    return [payload for kind, payload in _events(kb, tid, db)
            if kind == "delivery_review_waiting" and payload["code"] == code]


def _red_identity(kb, tid, head):
    """Owner rule 3: the red check's continuation has its own identity, of the source card, H and "continuation"."""
    return [{"identity": kb._review_followup_identity_key(reviewed_task_id=tid, candidate=f"{head} continuation")}]


def test_the_annotations_name_the_failed_tests_of_each_failed_test_job(red):
    """The return's reason names every failed latest job of H's run that holds the red check; its tests are the
    ones the failed slice job's own check run's annotations name under the fixed title, and nothing of any
    other title. Only the failed test job's check run is read for annotations."""
    kb, root, repo, gh = red
    tid, head = _ready(red)
    gh["tests"] = [FLAKY, UNLISTED]

    _tick(kb)

    assert [path for _, path in gh["calls"] if path.endswith("/annotations")] == [
        f"/repos/{REPO}/check-runs/903/annotations"]
    assert [(payload["reason"], payload["tests"]) for payload in _waited(kb, tid, "red_check_returned")] == [
        (f"failed jobs: {CHECK}, Python tests / Run tests slice 3/12", [FLAKY, UNLISTED])]


def test_a_listed_flaky_failure_is_not_rerun_and_the_work_returns_once(red):
    """A slice that failed only tests the old flaky list named: nothing reruns it. The first pass records
    red_check_waiting and then red_check_returned once for H with its one continuation card, through the read
    step alone, and later passes on the still red H record and send nothing more."""
    kb, root, repo, gh = red
    tid, head = _ready(red)
    events = _events(kb, tid)

    for _ in range(3):
        _tick(kb)

    (item,) = _raw(kb.kanban_db_path(), CONTINUED, tid)
    outcome = {"delivery_id": 1, "head": head, "pull_request_number": 41}
    assert _events(kb, tid)[len(events):] == [
        ("delivery_review_waiting", dict(outcome, code="red_check_waiting",
                                         check_runs=[{"id": 900, "conclusion": "failure"}])),
        ("delivery_review_waiting", dict(outcome, code="red_check_returned", tests=[FLAKY], rework=item["id"],
                                         reason=f"failed jobs: {CHECK}, Python tests / Run tests slice 3/12"))]
    assert gh["reruns"] == [] and set(gh["steps"]) == {"read_checks"}


def test_nothing_is_decided_while_a_job_of_the_run_still_runs(red):
    """ci.yaml's timing report runs after the summary check, so the required check is red before H's run
    has completed (as an old attempt's is while GitHub starts a rerun): a pass then returns nothing, and once
    every job has completed the work returns once."""
    kb, root, repo, gh = red
    tid, head = _ready(red)
    gh["timing"] = ("in_progress", None)

    _tick(kb)
    _tick(kb)

    assert gh["reruns"] == [] and _raw(kb.kanban_db_path(), CONTINUED, tid) == []
    assert _waited(kb, tid, "red_check_returned") == []
    gh["timing"] = ("completed", "success")
    _tick(kb)
    assert gh["reruns"] == [] and len(_waited(kb, tid, "red_check_returned")) == 1


def test_a_second_red_on_the_same_head_returns_the_work_once(red):
    """A rerun of H started on GitHub failed again: red_check_returned stays recorded once with the failed
    tests, and one continuation card goes to the builder through the review handback's follow-up record; no
    rerun is sent."""
    kb, root, repo, gh = red
    tid, head = _ready(red)
    _tick(kb)
    gh["attempt"] = 2  # GitHub reran slice 3 and it failed again

    for _ in range(3):
        _tick(kb)

    (item,) = _raw(kb.kanban_db_path(), CONTINUED, tid)
    assert _waited(kb, tid, "red_check_returned") == [{
        "delivery_id": 1, "head": head, "pull_request_number": 41, "code": "red_check_returned",
        "reason": f"failed jobs: {CHECK}, Python tests / Run tests slice 3/12", "tests": [FLAKY], "rework": item["id"]}]
    assert gh["reruns"] == [] and _waited(kb, tid, "red_check_rerun") == []
    assert _raw(kb.kanban_db_path(), "SELECT COUNT(*) AS n FROM task_events WHERE task_id = ? "
                "AND kind = 'review_followup_recorded'", tid) == [{"n": 1}]
    assert _state(kb, head) == "open" and _cards(kb) == []


def test_a_red_check_return_creates_the_continuation_in_place_of_the_triage_item(red):
    """Required behaviors 2 to 4: nothing reruns the failed test, and the same pass returns the work once
    through the continuation a review return makes, in place of the triage item: its body names H, the pull
    request and the failed test as CI named it, and once the builder can run it is claimed with H as its base."""
    from hermes_cli.owner_workspace import owner_title

    kb, root, repo, gh = red
    tid, head = _ready(red, tier=1)
    gh["tests"] = [UNLISTED]

    _tick(kb)
    _tick(kb)

    db = kb.kanban_db_path()
    (item,) = _raw(db, CONTINUED, tid)
    (source,) = _raw(db, "SELECT * FROM tasks WHERE id = ?", tid)
    assert gh["reruns"] == [] and _waited(kb, tid, "red_check_rerun") == []
    assert [(p["reason"], p["tests"], p["rework"]) for p in _waited(kb, tid, "red_check_returned")] == [
        (f"failed jobs: {CHECK}, Python tests / Run tests slice 3/12", [UNLISTED], item["id"])]
    assert _raw(db, "SELECT id FROM tasks WHERE status = 'triage'") == [] and _raw(db, REWORK) == []
    assert (item["assignee"], item["status"], item["title"]) == (IMPLEMENTER, "ready", owner_title(source["title"]))
    assert [item[key] for key in KEPT] == [source[key] for key in KEPT] and item["risk_tier"] == 1
    assert f"head {head} of pull request 41" in item["body"] and item["body"].endswith(
        f"CI named these failed tests: {UNLISTED}. The job's CI log has the complete list.")
    assert item["branch_name"] == _branch(tid, head) and _git(repo, "rev-parse", f"refs/heads/{_branch(tid, head)}") == head
    assert _raw(db, IDENTITY, tid, item["id"]) == _red_identity(kb, tid, head)
    (root / "profiles" / IMPLEMENTER).mkdir(parents=True)
    _tick(kb)
    (claimed,) = _raw(db, CONTINUED, tid)
    assert (claimed["id"], claimed["status"], claimed["base_commit"]) == (item["id"], "running", head)
    assert _git(Path(claimed["workspace_path"]), "rev-parse", "HEAD") == head


@pytest.mark.parametrize("closed", [False, True], ids=["moved", "closed"])
@pytest.mark.parametrize("read", ["/actions/runs", "/jobs", "/annotations"])
def test_nothing_is_recorded_or_sent_when_the_pull_request_leaves_h_during_a_new_read(red, read, closed):
    """Owner rule 1: the pull request moves off H, or closes at H, during the workflow-runs, the jobs or the
    annotations read. The pull request, read again after the last of them, stops the pass before the return:
    nothing is recorded, sent or made. A later pass reads again."""
    kb, root, repo, gh = red
    tid, head = _ready(red)
    gh.update(move_after=read, closed=closed)
    events = _events(kb, tid)

    _tick(kb)

    assert [payload["code"] for _, payload in _events(kb, tid)[len(events):]] == ["red_check_waiting"]
    assert [path for _, path in gh["calls"]][-2:] == [f"/repos/{REPO}/check-runs/903/annotations",
                                                      f"/repos/{REPO}/pulls/41"]
    assert {method for method, _ in gh["calls"]} == {"GET"} and _raw(kb.kanban_db_path(), CONTINUED, tid) == []
    gh.update(move_after=None, pull_head=None, closed=False)
    _tick(kb)
    assert len(_waited(kb, tid, "red_check_returned")) == 1


@pytest.mark.parametrize("answer", [403, None], ids=["refused", "unanswered"])
def test_a_refused_or_unanswered_rerun_call_returns_the_work_once(red, answer):
    """Records an earlier pass wrote stay readable: it recorded a rerun of H and the call's refused (403) or
    missing answer, and no return. The next pass returns the work once with one continuation card naming the
    failed test, and no pass sends anything."""
    kb, root, repo, gh = red
    tid, head = _ready(red)
    outcome = {"delivery_id": 1, "head": head, "pull_request_number": 41}
    with kb.connect_closing(db_path=kb.kanban_db_path()) as conn, kb.write_txn(conn):
        for payload in (dict(outcome, code="red_check_waiting", check_runs=[{"id": 900, "conclusion": "failure"}]),
                        dict(outcome, code="red_check_rerun", jobs=[903], tests=[FLAKY], attempt=1),
                        dict(outcome, code="red_check_rerun_outcome", jobs=[903], statuses=[answer])):
            kb._append_event(conn, tid, "delivery_review_waiting", payload)
    events = _events(kb, tid)

    for _ in range(3):
        _tick(kb)

    (item,) = _raw(kb.kanban_db_path(), CONTINUED, tid)
    assert [payload for _, payload in _events(kb, tid)[len(events):]] == [
        dict(outcome, code="red_check_returned", reason=f"failed jobs: {CHECK}, Python tests / Run tests slice 3/12", tests=[FLAKY],
             rework=item["id"])]
    assert gh["reruns"] == [] and item["assignee"] == IMPLEMENTER
    assert f"CI named these failed tests: {FLAKY}." in item["body"]


@pytest.mark.parametrize("check", [("completed", "success"), ("in_progress", None)], ids=["green", "pending"])
@pytest.mark.parametrize("answer", [403, None], ids=["refused", "unanswered"])
def test_a_recorded_refused_or_unknown_rerun_follows_what_ci_shows_later(red, answer, check):
    """An earlier pass recorded a rerun of H and the call's refused (403) or missing answer, and no return, and CI
    on H then turns green or stays pending. Nothing forces a return any more: green H gets its review cards,
    pending H waits, and no pass records red_check_returned or sends anything."""
    kb, root, repo, gh = red
    tid, head = _ready(red)
    outcome = {"delivery_id": 1, "head": head, "pull_request_number": 41}
    with kb.connect_closing(db_path=kb.kanban_db_path()) as conn, kb.write_txn(conn):
        for payload in (dict(outcome, code="red_check_waiting", check_runs=[{"id": 900, "conclusion": "failure"}]),
                        dict(outcome, code="red_check_rerun", jobs=[903], tests=[FLAKY], attempt=1),
                        dict(outcome, code="red_check_rerun_outcome", jobs=[903], statuses=[answer])):
            kb._append_event(conn, tid, "delivery_review_waiting", payload)
    gh["check"] = check

    _tick(kb)
    _tick(kb)

    assert (_waited(kb, tid, "red_check_returned"), _raw(kb.kanban_db_path(), CONTINUED, tid), gh["reruns"]) == (
        [], [], [])
    assert (_state(kb, head), len(_cards(kb))) == (("review_cards_created", 1) if check[1] else ("open", 0))


@pytest.mark.parametrize("after", ["moved", "closed", "failed"])
def test_nothing_returns_when_the_pull_request_leaves_h_while_the_branch_is_made(red, monkeypatch, after):
    """Safety rule 1: on the red check's return the pull request moves off H, closes or cannot be read while the
    continuation's branch is made. The read after the branch stops the pass: no continuation and no
    red_check_returned, and nothing is sent; a later pass, the pull request open at H again, returns the work once."""
    from hermes_cli import kanban_delivery_github as transport

    kb, root, repo, gh = red
    db = kb.kanban_db_path()
    tid, head = _ready(red)
    gh["tests"] = [UNLISTED]

    def failed():
        gh["each"] = None
        raise transport.GitHubTransportError("network_error")

    _branching(monkeypatch, {"moved": lambda: gh.update(pull_head=OTHER), "failed": lambda: gh.update(each=failed),
                             "closed": lambda: gh.update(closed=True, pull_head=head)}[after])

    _tick(kb)

    assert (_raw(db, CONTINUED, tid), _waited(kb, tid, "red_check_returned"), gh["reruns"]) == ([], [], [])
    gh.update(pull_head=None, closed=False)
    _tick(kb)
    (item,) = _raw(db, CONTINUED, tid)
    assert ([p["rework"] for p in _waited(kb, tid, "red_check_returned")], gh["reruns"]) == ([item["id"]], [])


@pytest.mark.parametrize("tier", ["routine", "deep"])
def test_a_governed_sources_red_check_continuation_is_claimable_on_its_builders_route(red, tier):
    """Owner rule 1: the red check returns the work of a governed source card, of execution tier ``tier`` and
    its builder raphael-claude-worker: the continuation has the source's route and tier and the builder's lock,
    so it is ready, and the kernel claims it with H as its recorded base."""
    kb, root, repo, gh = red
    db = kb.kanban_db_path()
    tid, head = _ready(red)
    model, effort = _governed(kb, tid, tier)
    gh["tests"] = [UNLISTED]

    _tick(kb)

    (item,) = _raw(db, CONTINUED, tid)
    assert [p["rework"] for p in _waited(kb, tid, "red_check_returned")] == [item["id"]]
    assert (item["assignee"], item["status"], item["execution_tier"], item["model_override"],
            item["reasoning_effort"]) == (BUILDER, "ready", tier, model, effort)
    assert kb.policy_lock_error(item["model_policy_lock"], BUILDER, "anthropic", model, effort, tier) is None
    (root / "profiles" / BUILDER).mkdir(parents=True)
    _tick(kb)
    (claimed,) = _raw(db, CONTINUED, tid)
    assert (claimed["id"], claimed["status"], claimed["base_commit"]) == (item["id"], "running", head)
    assert (claimed["branch_name"], _git(Path(claimed["workspace_path"]), "rev-parse", "HEAD")) == (_branch(tid, head), head)


@pytest.mark.parametrize("route", ["lock_only", "provider_only", "blank_tier", "unknown_effort"])
def test_a_governed_sources_red_check_continuation_parks_governed_when_unlockable(red, route):
    """Owner rule 3 of round 3: the red check returns the work of a governed source card, and its route cannot be
    locked for its builder: it
    is governed by its lock alone, with no tier, or by a blank tier alone; or owner receipt-bound on a provider with
    no model or, with no tier, on an effort create_task refuses. The continuation is created parked with the reason
    red_check_returned names, and it stays governed on the source's tier and route as stored (_held): never a
    manual card."""
    kb, root, repo, gh = red
    tid, head = _ready(red)
    _governed(kb, tid, "routine")
    change, why = UNLOCKABLE[route]
    _raw(kb.kanban_db_path(), f"UPDATE tasks SET {change} WHERE id = ?", tid)
    gh["tests"] = [UNLISTED]

    for _ in range(3):
        _tick(kb)

    (item,) = _raw(kb.kanban_db_path(), CONTINUED, tid)
    (returned,) = _waited(kb, tid, "red_check_returned")
    _held(kb, tid, item, returned, why)


def test_a_legacy_sources_red_check_continuation_keeps_its_exact_route_and_lock(red):
    """Owner rule 3 of round 3: the red check returns the work of a source card sealed before the pins (_legacy).
    The continuation copies the
    source's route as it is, high at risk tier 2, with the source's own lock and no recorded pin, and
    red_check_returned names it (_copied)."""
    kb, root, repo, gh = red
    tid, head = _ready(red)
    _legacy(kb, tid)
    gh["tests"] = [UNLISTED]

    for _ in range(3):
        _tick(kb)

    _copied(kb, tid, _raw(kb.kanban_db_path(), CONTINUED, tid), _waited(kb, tid, "red_check_returned"))


@pytest.mark.parametrize("layout", ["external", "advanced", "dot_worktrees", "other_clone", "separate_git_dir"])
def test_a_red_check_of_a_source_checked_out_outside_its_repository_continues_in_a_worktree_of_its_own(red, layout):
    """The red check returns the work of a source whose checkout is linked from outside its repository, there
    advanced past H, in an external folder named .worktrees, removed from under another clone, or in a repository
    whose Git directory is separate: the kernel claims the continuation in a worktree of its own on its branch at
    H in the primary folder of the source card's project, H its recorded base."""
    kb, root, repo, gh = red
    tid, head = _ready(red)
    gh["tests"] = [UNLISTED]
    source = _checkout_elsewhere(kb, repo, tid, layout)

    _tick(kb)

    _claimed_apart(kb, root, repo, tid, head, source)


@pytest.mark.parametrize("project", ["elsewhere", "left"])
@pytest.mark.parametrize("code", ["review_returned", "red_check_returned"])
def test_a_project_folder_of_another_repository_or_a_project_gone_refuses_the_return(red, armed, monkeypatch, code,
                                                                                      project):
    """Owner rule of round 2: the primary folder of the source card's project holds H, but its origin names another
    repository of the policy than the delivery's; or the project leaves the registry while the branch is made, so
    kb.create_task anchors the card elsewhere. The return refuses on every pass: no continuation, no return record
    and no branch, but for the one made at H before the project left."""
    kb, root, repo, gh = red
    db = kb.kanban_db_path()
    if code == "review_returned":
        gh["check"] = ("completed", "success")
    tid, head = _changes_requested(red)[:2] if code == "review_returned" else _ready(red)
    gh["tests"] = [UNLISTED]
    folder = _clone(repo, "clone", ELSEWHERE)
    if project == "elsewhere":
        _registered(db, tid, folder, project="elsewhere-project")
    else:
        _branching(monkeypatch, lambda: _raw(root / "projects.db", "DELETE FROM projects WHERE id = 'delivery-project'"))

    _tick(kb)
    _tick(kb)

    left = _branch(tid, head) if project == "left" else ""
    assert (_raw(db, CONTINUED, tid), _waited(kb, tid, code), _branches(folder), _branches(repo)) == ([], [], "", left)
    assert gh["reruns"] == []


@pytest.mark.parametrize("left", ["ready", "archived", "done"])
def test_a_review_return_on_the_red_returned_head_reuses_its_one_continuation(red, armed, left):
    """Owner rule 3: the red check returned H's work through its continuation, which is then left ``left``.
    CI on H turns green and the review of H requests changes: no second card or branch is made for H. A
    continuation still open gets the review's findings as one comment, a done or archived one stays as it
    is, and the review_returned event names that one card."""
    kb, root, repo, gh = red
    db = kb.kanban_db_path()
    tid, head = _ready(red)
    gh["tests"] = [UNLISTED]
    _tick(kb)
    (item,) = _raw(db, CONTINUED, tid)
    _raw(db, "UPDATE tasks SET status = ? WHERE id = ?", left, item["id"])
    gh["check"] = ("completed", "success")
    _tick(kb)
    cards = {card["responsibility"]: card["id"] for card in _cards(kb)}
    _run(db, cards["R12"], 100, "VALIDATE_QUERY_FROM_LATEST_REVIEW")
    _done(db)
    _reviews(gh, ("CHANGES_REQUESTED", head, cards["R12"]))

    _tick(kb)
    _tick(kb)

    assert [card["id"] for card in _raw(db, CONTINUED, tid)] == [item["id"]]
    assert [p["rework"] for p in _waited(kb, tid, "review_returned")] == [item["id"]]
    assert _raw(db, "SELECT status, body FROM tasks WHERE id = ?", item["id"]) == [
        {"status": left, "body": item["body"]}]
    assert _git(repo, "branch", "--list", "wt/*", "delivery-return/*") == _branch(tid, head)
    comments = _raw(db, "SELECT author, body FROM task_comments WHERE task_id = ?", item["id"])
    assert len(comments) == (left == "ready")
    for comment in comments:
        assert f"head {head} of pull request 41" in comment["body"]
        assert f"Review card {cards['R12']}:\nVALIDATE_QUERY_FROM_LATEST_REVIEW" in comment["body"]


def test_a_red_check_makes_its_own_continuation_beside_a_done_review_follow_up(red):
    """Owner rule 3: a review that returned changes on H made its follow-up, which the approval of the
    unchanged H then closed. The red check on H leaves that item as it is and makes its own continuation,
    of the source card, H and "continuation", naming the failed test."""
    kb, root, repo, gh = red
    conn = kb.connect()
    try:
        tid, head, _ = _park(kb, conn, repo)
        review = kb.claim_review_task(conn, tid, claimer=f"{REVIEWER}:1")
        assert kb.request_changes(conn, tid, reason="tighten it", expected_run_id=review.current_run_id)[0]
    finally:
        conn.close()
    assert _rework(kb, repo, tid, commit=False) == head
    db = kb.kanban_db_path()
    (followup,) = _raw(db, REWORK)
    assert followup["status"] == "done"
    _published(kb, repo, gh, tid, head)
    _registered(db, tid, repo)
    gh["tests"] = [UNLISTED]

    _tick(kb)
    _tick(kb)

    (returned,) = _waited(kb, tid, "red_check_returned")
    (item,) = _raw(db, "SELECT * FROM tasks WHERE id = ?", returned["rework"])
    assert _raw(db, REWORK) == [followup]
    assert (item["assignee"], item["status"]) == (IMPLEMENTER, "ready")
    assert f"CI named these failed tests: {UNLISTED}." in item["body"]
    assert _raw(db, IDENTITY, tid, item["id"]) == _red_identity(kb, tid, head)


def test_the_count_0_notice_of_a_setup_error_names_no_failed_test(red):
    """Owner rule 4: a pytest setup error fails slice 3 with no failed test, and tests.yml's step then
    writes only its count notice, "count 0". That is count metadata and never a test id: no test is
    invented, nothing reruns, and the continuation's body says "No failed test was
    named." in its place."""
    kb, root, repo, gh = red
    workflow = Path(__file__).resolve().parents[2] / ".github" / "workflows" / "tests.yml"
    assert 'echo "::notice title=Failed test::count ${count:-0}"' in workflow.read_text(encoding="utf-8")
    tid, head = _ready(red)
    gh["tests"] = []  # job 903's check run then holds that notice, "count 0", and no test id

    _tick(kb)
    _tick(kb)

    (item,) = _raw(kb.kanban_db_path(), CONTINUED, tid)
    assert [(p["reason"], p["tests"]) for p in _waited(kb, tid, "red_check_returned")] == [
        (f"failed jobs: {CHECK}, Python tests / Run tests slice 3/12", [])]
    assert item["body"].endswith("No failed test was named. The job's CI log has the complete list.")
    assert "count 0" not in item["body"] and gh["reruns"] == []


def test_the_work_returns_on_the_ticked_board_under_the_worker_environment(red, monkeypatch):
    """The dispatcher's worker environment for proj-a (its board pinned, its card set, a profile home under
    the root) ticks the board other: the root registry and every board still resolve, and the return and
    its one continuation card, of the continuation's own identity (owner rule 3) and naming the failed test,
    land on other and on no other board, by a count read past the kernel."""
    kb, root, repo, gh = red
    gh["tests"] = [UNLISTED]
    kb.create_board("proj-a")
    kb.create_board("other")
    own_db = root / "kanban" / "boards" / "proj-a" / "kanban.db"
    other_db = root / "kanban" / "boards" / "other" / "kanban.db"
    profile_home = root / "profiles" / IMPLEMENTER
    profile_home.mkdir(parents=True)
    _write_config(profile_home, enabled=True)
    tid, head = _approved(kb, repo, "beta", board="other")
    _published(kb, repo, gh, tid, head, board="other")
    _registered(other_db, tid, repo, profile_home)  # the registry of the worker's own profile home
    conn = kb.connect(board="proj-a")
    try:
        claimed = kb.claim_task(conn, kb.create_task(conn, title="plain work", assignee=IMPLEMENTER))
        workspace = kb.resolve_workspace(claimed, board="proj-a")
        kb.set_workspace_path(conn, claimed.id, str(workspace))
        env = _spawned_worker_env(kb, claimed, workspace, "proj-a", monkeypatch)
    finally:
        conn.close()
    assert (env["HERMES_KANBAN_TASK"], env["HERMES_KANBAN_DB"], env["HERMES_KANBAN_BOARD"], env["HERMES_HOME"]) == (
        claimed.id, str(own_db), "proj-a", str(profile_home))

    _become(env, monkeypatch)
    for _ in range(2):
        ticked = kb.connect(db_path=other_db)
        try:
            kb.dispatch_once(ticked, board="other", spawn_fn=lambda *a, **k: 1)
        finally:
            ticked.close()

    assert kb.kanban_home() == root and kb.register_db_path() == root / "kanban" / "board_register.db"
    assert kb.get_register_entry("proj-a") is not None and kb.get_register_entry("other") is not None
    assert kb.board_dir("other") / "kanban.db" == other_db
    returned = ("SELECT COUNT(*) AS n FROM task_events WHERE kind = 'delivery_review_waiting' "
                "AND json_extract(payload, '$.code') = 'red_check_returned'")
    rework = "SELECT COUNT(*) AS n FROM tasks WHERE body LIKE 'Rework of %'"
    boards = (other_db, own_db, root / "kanban.db")
    assert [(_raw(db, returned)[0]["n"], _raw(db, rework)[0]["n"]) for db in boards] == [(1, 1), (0, 0), (0, 0)]
    (item,) = _raw(other_db, CONTINUED, tid)
    assert (item["assignee"], item["project_id"], item["branch_name"]) == (IMPLEMENTER, "delivery-project", _branch(tid, head))
    assert _git(repo, "rev-parse", f"refs/heads/{_branch(tid, head)}") == head
    assert f"CI named these failed tests: {UNLISTED}." in item["body"]
    assert _raw(other_db, IDENTITY, tid, item["id"]) == _red_identity(kb, tid, head)
    assert gh["reruns"] == [] and not (profile_home / "kanban.db").exists()


# Owner decision 2026-10-20: a red required check on H returns the work to its builder once; nothing reruns.

def test_a_red_check_whose_failed_tests_are_all_on_the_old_flaky_list_returns_the_work_once(red):
    """Every failed test of the red slice was on the old flaky list: the work returns once, through one
    continuation card naming the test, and no rerun call is sent."""
    kb, root, repo, gh = red
    tid, head = _ready(red)

    for _ in range(3):
        _tick(kb)

    (item,) = _raw(kb.kanban_db_path(), CONTINUED, tid)
    assert [(p["tests"], p["rework"]) for p in _waited(kb, tid, "red_check_returned")] == [([FLAKY], item["id"])]
    assert gh["reruns"] == [] and _waited(kb, tid, "red_check_rerun") == []
    assert {method for method, _ in gh["calls"]} == {"GET"} and "rerun_flaky" not in gh["steps"]
    assert f"CI named these failed tests: {FLAKY}." in item["body"]


@pytest.mark.parametrize("answer", [{"total_count": 1}, "count 1"], ids=["object", "text"])
def test_a_malformed_annotations_answer_returns_the_work_once_naming_no_failed_test(red, monkeypatch, answer):
    """The failed slice's annotations answer is malformed: it neither delays nor vetoes the return. The work
    returns once, saying that no failed test was named and pointing to the job's CI log."""
    from hermes_cli import kanban_delivery_github as transport

    kb, root, repo, gh = red
    tid, head = _ready(red)
    table = transport._exchange

    def exchange(method, target, authorization, payload=None):
        if urlsplit(target).path == f"/repos/{REPO}/check-runs/903/annotations":
            return 200, answer
        return table(method, target, authorization, payload)

    monkeypatch.setattr(transport, "_exchange", exchange)

    for _ in range(3):
        _tick(kb)

    (item,) = _raw(kb.kanban_db_path(), CONTINUED, tid)
    assert [(p["tests"], p["rework"]) for p in _waited(kb, tid, "red_check_returned")] == [([], item["id"])]
    assert item["body"].endswith("No failed test was named. The job's CI log has the complete list.")
    assert gh["reruns"] == []


def test_a_recorded_accepted_rerun_with_no_newer_attempt_returns_the_work_once(red):
    """An earlier pass recorded a rerun of H at attempt 1 and its accepted answer, and GitHub still shows
    attempt 1 red: the records stay readable, and the work returns once instead of waiting for a newer
    attempt. No rerun call is sent."""
    kb, root, repo, gh = red
    tid, head = _ready(red)
    outcome = {"delivery_id": 1, "head": head, "pull_request_number": 41}
    with kb.connect_closing(db_path=kb.kanban_db_path()) as conn, kb.write_txn(conn):
        for payload in (dict(outcome, code="red_check_waiting", check_runs=[{"id": 900, "conclusion": "failure"}]),
                        dict(outcome, code="red_check_rerun", jobs=[903], tests=[FLAKY], attempt=1),
                        dict(outcome, code="red_check_rerun_outcome", jobs=[903], statuses=[201])):
            kb._append_event(conn, tid, "delivery_review_waiting", payload)

    for _ in range(3):
        _tick(kb)

    (item,) = _raw(kb.kanban_db_path(), CONTINUED, tid)
    assert [(p["tests"], p["rework"]) for p in _waited(kb, tid, "red_check_returned")] == [([FLAKY], item["id"])]
    assert gh["reruns"] == [] and len(_waited(kb, tid, "red_check_rerun")) == 1


def test_guard_a_green_head_gets_its_review_cards(red):
    kb, root, repo, gh = red
    tid, head = _ready(red)
    gh["check"] = ("completed", "success")

    _tick(kb)

    assert len(_cards(kb)) == 1 and _state(kb, head) == "review_cards_created"
    assert (_waited(kb, tid, "red_check_returned"), gh["reruns"]) == ([], [])


def test_guard_a_head_whose_work_was_already_returned_gets_no_second_return(red):
    kb, root, repo, gh = red
    tid, head = _ready(red)
    gh["tests"] = [UNLISTED]
    _tick(kb)
    _tick(kb)
    (item,) = _raw(kb.kanban_db_path(), CONTINUED, tid)

    for _ in range(3):
        _tick(kb)

    assert [p["rework"] for p in _waited(kb, tid, "red_check_returned")] == [item["id"]]
    assert [card["id"] for card in _raw(kb.kanban_db_path(), CONTINUED, tid)] == [item["id"]]
    assert gh["reruns"] == []


def test_guard_a_pull_request_closed_outside_the_step_is_marked_closed_outside(red):
    kb, root, repo, gh = red
    tid, head = _ready(red)
    gh["pull"] = {"state": "closed", "merged": False}

    _tick(kb)
    _tick(kb)

    assert (_state(kb, head), _cards(kb), gh["reruns"]) == ("closed_outside", [], [])
    assert [payload for kind, payload in _events(kb, tid) if kind == "delivery_closed_outside"] == [
        {"delivery_id": 1, "head": head, "pull_request_number": 41, "merged": False}]
    assert (_waited(kb, tid, "red_check_returned"), _raw(kb.kanban_db_path(), CONTINUED, tid)) == ([], [])


# Owner rule of round 1: the return depends only on the red required check and on H's run still in progress; every
# other field is best-effort reason text, dropped when malformed or contradictory.

JOBS = (900, 901, 903, 904)
SLICE = "Python tests / Run tests slice 3/12"
UNNAMED = "No failed test was named. The job's CI log has the complete list."


def _returned_once(kb, tid, reason, tests):
    """The work went back once for H: one red_check_returned of ``reason`` and ``tests``, and its one continuation."""
    (item,) = _raw(kb.kanban_db_path(), CONTINUED, tid)
    assert [(p["reason"], p["tests"], p["rework"]) for p in _waited(kb, tid, "red_check_returned")] == [
        (reason, tests, item["id"])]
    return item


def test_a_job_conclusion_of_the_wrong_type_neither_blocks_the_return_nor_names_its_tests(red):
    """Owner rule 3: slice 3's conclusion is a list. That field is dropped, so slice 3 is no failed job and no
    annotation of it is read; the red check's own job is still named, and the work returns once naming no test."""
    kb, root, repo, gh = red
    tid, head = _ready(red)
    gh["jobs"] = {903: {"conclusion": []}}

    for _ in range(3):
        _tick(kb)

    item = _returned_once(kb, tid, f"failed jobs: {CHECK}", [])
    assert item["body"].endswith(UNNAMED) and gh["reruns"] == []
    assert [path for _, path in gh["calls"] if path.endswith("/annotations")] == []


@pytest.mark.parametrize("running", [True, False], ids=["a_job_in_progress", "all_jobs_completed"])
@pytest.mark.parametrize("where", ["run", "jobs"])
def test_a_run_or_jobs_of_another_head_neither_wait_nor_name_failed_jobs(red, where, running):
    """Owner rules 2 and 3: GitHub lists run 77, or each of its jobs, under another head than H, once while the
    timing report says in progress and once when every job has completed. That run is not H's: nothing waits on it,
    none of its jobs or tests is named, and the work returns once naming the red required check."""
    kb, root, repo, gh = red
    tid, head = _ready(red)
    if where == "run":
        gh["run"] = {"head_sha": OTHER}
    else:
        gh["jobs"] = {job: {"head_sha": OTHER} for job in JOBS}
    if running:
        gh["timing"] = ("in_progress", None)

    for _ in range(3):
        _tick(kb)

    item = _returned_once(kb, tid, f"red required check: {CHECK}", [])
    assert item["body"].endswith(UNNAMED) and gh["reruns"] == []


@pytest.mark.parametrize("job, timing, reason, tests", [
    (904, ("in_progress", None), f"failed jobs: {CHECK}, {SLICE}", [FLAKY]),
    (903, ("completed", "success"), f"failed jobs: {CHECK}", [])], ids=["running_timing_report", "failed_slice"])
def test_one_job_of_another_head_is_dropped_alone(red, job, timing, reason, tests):
    """Owner rule 3: of run 77's jobs only one names another head, the timing report while it says in progress or
    the failed slice 3. That job alone is dropped: nothing waits on it and none of its tests is named, while the
    run's other jobs of H still fill the reason of the one return."""
    kb, root, repo, gh = red
    tid, head = _ready(red)
    gh.update(jobs={job: {"head_sha": OTHER}}, timing=timing)

    for _ in range(3):
        _tick(kb)

    _returned_once(kb, tid, reason, tests)
    assert gh["reruns"] == []


def test_a_completed_run_with_a_job_that_says_in_progress_returns_the_work_once(red):
    """Owner rule 2: run 77 says completed while its timing report says in progress. The two answers contradict
    each other, so nothing waits: the work returns once, and the job said in progress is named no failed job."""
    kb, root, repo, gh = red
    tid, head = _ready(red)
    gh.update(run={"status": "completed"}, timing=("in_progress", None))

    for _ in range(3):
        _tick(kb)

    _returned_once(kb, tid, f"failed jobs: {CHECK}, {SLICE}", [FLAKY])
    assert gh["reruns"] == []


@pytest.mark.parametrize("running", [True, False], ids=["a_job_in_progress", "all_jobs_completed"])
def test_jobs_whose_run_id_names_another_run_neither_wait_nor_name_failed_jobs(red, running):
    """Owner rules 2 and 3: every job GitHub lists for run 77 names run 88. None is that run's: nothing waits on
    them, none of them or their tests is named, and the work returns once naming the red required check."""
    kb, root, repo, gh = red
    tid, head = _ready(red)
    gh["jobs"] = {job: {"run_id": 88} for job in JOBS}
    if running:
        gh["timing"] = ("in_progress", None)

    for _ in range(3):
        _tick(kb)

    item = _returned_once(kb, tid, f"red required check: {CHECK}", [])
    assert item["body"].endswith(UNNAMED) and gh["reruns"] == []


@pytest.mark.parametrize("field", [{"name": "Python tests / Run tests slice 2/12"}, {"conclusion": "success"},
                                   {"conclusion": "cancelled"}], ids=["name", "success", "cancelled"])
def test_a_red_check_job_that_disagrees_with_the_decision_names_no_failed_job(red, field):
    """Owner rule 3: job 900, the red required check run, carries another name or conclusion than the decision
    requests read for it. Run 77 is then not shown to hold that check: none of its jobs or tests is named, and the
    work returns once naming the red required check from the decision."""
    kb, root, repo, gh = red
    tid, head = _ready(red)
    gh["jobs"] = {900: field}

    for _ in range(3):
        _tick(kb)

    item = _returned_once(kb, tid, f"red required check: {CHECK}", [])
    assert item["body"].endswith(UNNAMED) and gh["reruns"] == []


@pytest.mark.parametrize("status", ["in_progress", "queued"])
def test_a_run_in_progress_with_its_jobs_completed_waits_then_returns_the_work_once(red, status):
    """Owner rule 2: run 77 itself says in progress (or queued) while every job it lists, the red check's included,
    has completed. Passes wait and record no return; once the run says completed, the work returns exactly once."""
    kb, root, repo, gh = red
    tid, head = _ready(red)
    gh["run"] = {"status": status}

    _tick(kb)
    _tick(kb)

    assert (_waited(kb, tid, "red_check_returned"), _raw(kb.kanban_db_path(), CONTINUED, tid)) == ([], [])
    assert len(_waited(kb, tid, "red_check_waiting")) == 1
    gh["run"] = {"status": "completed"}
    for _ in range(3):
        _tick(kb)
    _returned_once(kb, tid, f"failed jobs: {CHECK}, {SLICE}", [FLAKY])
    assert gh["reruns"] == []


@pytest.mark.parametrize("blank", ["", " \t "], ids=["empty", "whitespace"])
def test_a_count_1_annotation_with_a_blank_message_names_no_failed_test(red, blank):
    """Owner rule 3: slice 3's check run carries "count 1" and one failed-test annotation whose message is blank.
    A blank message is no test name: no test is named, and the continuation says so and points to the CI log."""
    kb, root, repo, gh = red
    tid, head = _ready(red)
    gh.update(tests=[blank], count=1)

    for _ in range(3):
        _tick(kb)

    item = _returned_once(kb, tid, f"failed jobs: {CHECK}, {SLICE}", [])
    assert item["body"].endswith(UNNAMED) and "CI named" not in item["body"]
