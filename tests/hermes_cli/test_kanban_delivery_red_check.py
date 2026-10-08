"""The red check card: a required check red on the exact published head names its failed tests from the
annotations tests.yml writes on each failed slice job's own check run, reruns the failed jobs once when
decide_rerun allows it, and otherwise returns the work once to its builder with one continuation card.
Real boards and git; GitHub is the real transport with only its exchange and App token replaced by a
table (the world of test_kanban_delivery_review_cards.py), so every call passes the allowlist."""

from __future__ import annotations

from pathlib import Path
from urllib.parse import urlsplit

import pytest

from tests.hermes_cli.test_kanban_delivery import (  # noqa: F401  (board is a fixture)
    IMPLEMENTER, REVIEWER, _become, _git, _park, _rework, _spawned_worker_env, _write_config, board,
)
from tests.hermes_cli.test_kanban_delivery_github import REPO
from tests.hermes_cli.test_kanban_delivery_publish_step import _approved, _events, _tick
from tests.hermes_cli.test_kanban_delivery_review_cards import (  # noqa: F401  (world is a fixture)
    CHECK, _cards, _project, _published, _raw, _ready, _state, world,
)
from tests.hermes_cli.test_kanban_delivery_review_return import CONTINUED, KEPT

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
    or not at all when None. With gh["closed"], the pull request is closed at H from gh["move_after"] on."""
    kb, root, repo, gh = world
    from hermes_cli import kanban_delivery_github as transport

    table = transport._exchange  # the world's: the pull request and H's check runs
    gh.update(check=("completed", "failure"), attempt=1, tests=[FLAKY], count=None, reruns=[],
              timing=("completed", "success"), rerun=201, closed=False)

    def exchange(method, target, authorization, payload=None):
        status, value = table(method, target, authorization, payload)
        path, head = urlsplit(target).path, gh["head"]
        if path == f"/repos/{REPO}/pulls/41" and gh["closed"] and gh["pull_head"]:
            return 200, dict(value, state="closed", head={"sha": head})
        job = {"head_sha": head, "status": "completed", "conclusion": "failure", "run_attempt": gh["attempt"]}
        if (method, path) == ("GET", f"/repos/{REPO}/actions/runs"):
            return 200, {"total_count": 1, "workflow_runs": [{"id": 77, "head_sha": head}]}
        if (method, path) == ("GET", f"/repos/{REPO}/actions/runs/77/jobs"):
            return 200, {"total_count": 4, "jobs": [
                dict(job, id=900, name=CHECK, steps=[{"name": "Evaluate job results", "conclusion": "failure"}]),
                dict(job, id=901, name="Python tests / Run tests slice 1/12", conclusion="success",
                     steps=[{"name": "Run tests (slice 1/12)", "conclusion": "success"}]),
                dict(job, id=903, name="Python tests / Run tests slice 3/12", steps=[
                    {"name": "Checkout code", "conclusion": "success"},
                    {"name": "Run tests (slice 3/12)", "conclusion": "failure"}]),
                dict(job, id=904, name="CI timing report", status=gh["timing"][0], conclusion=gh["timing"][1],
                     steps=[])]}
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


def test_the_annotations_name_the_failed_tests_of_each_failed_test_job(red, monkeypatch):
    """decide_rerun gets every latest job of H's run that holds the red check; the failed slice job carries
    the test ids and the count its own check run's annotations name under the fixed title, and nothing
    of any other title. Only the failed test job's check run is read for annotations."""
    kb, root, repo, gh = red
    from hermes_cli import kanban_delivery
    from hermes_cli.kanban_delivery_fences import decide_rerun

    seen = []
    monkeypatch.setattr(kanban_delivery, "decide_rerun", lambda policy, repository, head, jobs, ledger: (
        seen.append((head, jobs, ledger)) or decide_rerun(policy, repository, head, jobs, ledger)), raising=False)
    tid, head = _ready(red)
    gh["tests"] = [FLAKY, UNLISTED]

    _tick(kb)

    assert [(h, [(job["id"], job.get("failed_tests"), job.get("failed_count")) for job in jobs], ledger)
            for h, jobs, ledger in seen] == [
        (head, [(900, None, None), (901, None, None), (903, [FLAKY, UNLISTED], 2), (904, None, None)],
         {"reruns": []})]
    assert [path for _, path in gh["calls"] if path.endswith("/annotations")] == [
        f"/repos/{REPO}/check-runs/903/annotations"]
    assert [payload["tests"] for payload in _waited(kb, tid, "red_check_returned")] == [[FLAKY, UNLISTED]]


def test_only_a_listed_flaky_failure_is_rerun_and_only_once(red):
    """A slice that failed only listed flaky tests: the failed job is rerun once through the rerun call,
    recorded once for H before it is sent and its accepted answer once after, with no rework item. A pass
    that still sees the first attempt red sends nothing more, and the green rerun gets H its review cards."""
    kb, root, repo, gh = red
    tid, head = _ready(red)
    events = _events(kb, tid)

    for _ in range(3):
        _tick(kb)

    outcome = {"delivery_id": 1, "head": head, "pull_request_number": 41}
    assert _events(kb, tid)[len(events):] == [
        ("delivery_review_waiting", dict(outcome, code="red_check_waiting",
                                         check_runs=[{"id": 900, "conclusion": "failure"}])),
        ("delivery_review_waiting", dict(outcome, code="red_check_rerun", jobs=[903], tests=[FLAKY], attempt=1)),
        ("delivery_review_waiting", dict(outcome, code="red_check_rerun_outcome", jobs=[903], statuses=[201]))]
    assert gh["reruns"] == [RERUN] and _raw(kb.kanban_db_path(), CONTINUED, tid) == []
    assert set(gh["steps"]) == {"read_checks", "rerun_flaky"}
    gh["check"] = ("completed", "success")
    _tick(kb)
    assert len(_cards(kb)) == 2 and _state(kb, head) == "review_cards_created" and gh["reruns"] == [RERUN]


def test_nothing_is_decided_while_a_job_of_the_run_still_runs(red):
    """ci.yaml's timing report runs after the summary check, so the required check is red before H's run
    has completed (as the old attempt's is while GitHub starts a rerun): a pass then reruns nothing and
    returns nothing, and once every job has completed the listed flaky failure is rerun once."""
    kb, root, repo, gh = red
    tid, head = _ready(red)
    gh["timing"] = ("in_progress", None)

    _tick(kb)
    _tick(kb)

    assert gh["reruns"] == [] and _raw(kb.kanban_db_path(), CONTINUED, tid) == []
    assert _waited(kb, tid, "red_check_rerun") == [] and _waited(kb, tid, "red_check_returned") == []
    gh["timing"] = ("completed", "success")
    _tick(kb)
    assert gh["reruns"] == [RERUN] and len(_waited(kb, tid, "red_check_rerun")) == 1


def test_a_second_red_on_the_same_head_returns_the_work_once(red):
    """The rerun of H failed again: red_check_returned is recorded once with the failed tests, and one
    continuation card goes to the builder through the review handback's follow-up record; no second rerun."""
    kb, root, repo, gh = red
    tid, head = _ready(red)
    _tick(kb)
    gh["attempt"] = 2  # GitHub reran slice 3 and it failed again

    for _ in range(3):
        _tick(kb)

    (item,) = _raw(kb.kanban_db_path(), CONTINUED, tid)
    assert _waited(kb, tid, "red_check_returned") == [{
        "delivery_id": 1, "head": head, "pull_request_number": 41, "code": "red_check_returned",
        "reason": "rerun_used", "tests": [FLAKY], "rework": item["id"]}]
    assert gh["reruns"] == [RERUN] and len(_waited(kb, tid, "red_check_rerun")) == 1
    assert _raw(kb.kanban_db_path(), "SELECT COUNT(*) AS n FROM task_events WHERE task_id = ? "
                "AND kind = 'review_followup_recorded'", tid) == [{"n": 1}]
    assert _state(kb, head) == "open" and _cards(kb) == []


def test_a_red_check_return_creates_the_continuation_in_place_of_the_triage_item(red):
    """Required behaviors 2 to 4: an unlisted failed test is not rerun, and the same pass returns the work once
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
        ("not_flaky", [UNLISTED], item["id"])]
    assert _raw(db, "SELECT id FROM tasks WHERE status = 'triage'") == [] and _raw(db, REWORK) == []
    assert (item["assignee"], item["status"], item["title"]) == (IMPLEMENTER, "ready", owner_title(source["title"]))
    assert [item[key] for key in KEPT] == [source[key] for key in KEPT] and item["risk_tier"] == 1
    assert f"head {head} of pull request 41" in item["body"] and item["body"].endswith(
        f"CI named these failed tests: {UNLISTED}. The job's CI log has the complete list.")
    assert _git(repo, "rev-parse", f"refs/heads/wt/{item['id']}") == head
    assert _raw(db, IDENTITY, tid, item["id"]) == _red_identity(kb, tid, head)
    (root / "profiles" / IMPLEMENTER).mkdir(parents=True)
    _tick(kb)
    (claimed,) = _raw(db, CONTINUED, tid)
    assert (claimed["id"], claimed["status"], claimed["base_commit"]) == (item["id"], "running", head)
    assert _git(Path(claimed["workspace_path"]), "rev-parse", "HEAD") == head


@pytest.mark.parametrize("tests", [[FLAKY], [UNLISTED]], ids=["rerun", "return"])
@pytest.mark.parametrize("closed", [False, True], ids=["moved", "closed"])
@pytest.mark.parametrize("read", ["/actions/runs", "/jobs", "/annotations"])
def test_nothing_is_recorded_or_sent_when_the_pull_request_leaves_h_during_a_new_read(red, read, closed, tests):
    """Owner rule 1: the pull request moves off H, or closes at H, during the workflow-runs, the jobs or the
    annotations read. The pull request, read again after the last of them, stops the pass on the rerun
    branch and on the return branch: nothing is recorded, sent or made. A later pass reads again."""
    kb, root, repo, gh = red
    tid, head = _ready(red)
    gh.update(move_after=read, closed=closed, tests=tests)
    events = _events(kb, tid)

    _tick(kb)

    assert [payload["code"] for _, payload in _events(kb, tid)[len(events):]] == ["red_check_waiting"]
    assert [path for _, path in gh["calls"]][-2:] == [f"/repos/{REPO}/check-runs/903/annotations",
                                                      f"/repos/{REPO}/pulls/41"]
    assert {method for method, _ in gh["calls"]} == {"GET"} and _raw(kb.kanban_db_path(), CONTINUED, tid) == []
    gh.update(move_after=None, pull_head=None, closed=False)
    _tick(kb)
    assert len(_waited(kb, tid, "red_check_rerun" if tests == [FLAKY] else "red_check_returned")) == 1


@pytest.mark.parametrize("answer, reason", [(403, "rerun_refused"), (None, "rerun_unknown")])
def test_a_refused_or_unanswered_rerun_call_returns_the_work_once(red, answer, reason):
    """Owner rule 2: the rerun call is refused (403), or gets no answer (a timeout). It is sent once, after
    the rerun record; its outcome is recorded once in the same pass, and the work returns once with that
    reason and one continuation card naming the failed test. Later passes send nothing."""
    kb, root, repo, gh = red
    tid, head = _ready(red)
    gh["rerun"] = answer
    events = _events(kb, tid)

    for _ in range(3):
        _tick(kb)

    (item,) = _raw(kb.kanban_db_path(), CONTINUED, tid)
    outcome = {"delivery_id": 1, "head": head, "pull_request_number": 41}
    assert [payload for _, payload in _events(kb, tid)[len(events):]] == [
        dict(outcome, code="red_check_waiting", check_runs=[{"id": 900, "conclusion": "failure"}]),
        dict(outcome, code="red_check_rerun", jobs=[903], tests=[FLAKY], attempt=1),
        dict(outcome, code="red_check_rerun_outcome", jobs=[903], statuses=[answer]),
        dict(outcome, code="red_check_returned", reason=reason, tests=[FLAKY], rework=item["id"])]
    assert gh["reruns"] == [RERUN] and item["assignee"] == IMPLEMENTER
    assert f"CI named these failed tests: {FLAKY}." in item["body"]


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
    invented, the unchanged policy refuses the rerun, and the continuation's body says "No failed test was
    named." in its place."""
    kb, root, repo, gh = red
    workflow = Path(__file__).resolve().parents[2] / ".github" / "workflows" / "tests.yml"
    assert 'echo "::notice title=Failed test::count ${count:-0}"' in workflow.read_text(encoding="utf-8")
    tid, head = _ready(red)
    gh["tests"] = []  # job 903's check run then holds that notice, "count 0", and no test id

    _tick(kb)
    _tick(kb)

    (item,) = _raw(kb.kanban_db_path(), CONTINUED, tid)
    assert [(p["reason"], p["tests"]) for p in _waited(kb, tid, "red_check_returned")] == [("invalid_fact", [])]
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
    _project(other_db, tid)
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
    assert (item["assignee"], item["project_id"]) == (IMPLEMENTER, "delivery-project")
    assert f"CI named these failed tests: {UNLISTED}." in item["body"]
    assert _raw(other_db, IDENTITY, tid, item["id"]) == _red_identity(kb, tid, head)
    assert gh["reruns"] == [] and not (profile_home / "kanban.db").exists()
