"""The red check card: a required check red on the exact published head names its failed tests from the
annotations tests.yml writes on each failed slice job's own check run, reruns the failed jobs once when
decide_rerun allows it, and otherwise returns the work once to its builder with one rework item.
Real boards and git; GitHub is the real transport with only its exchange and App token replaced by a
table (the world of test_kanban_delivery_review_cards.py), so every call passes the allowlist."""

from __future__ import annotations

from urllib.parse import urlsplit

import pytest

from tests.hermes_cli.test_kanban_delivery import (  # noqa: F401  (board is a fixture)
    IMPLEMENTER, _become, _spawned_worker_env, _write_config, board,
)
from tests.hermes_cli.test_kanban_delivery_github import REPO
from tests.hermes_cli.test_kanban_delivery_publish_step import _approved, _events, _tick
from tests.hermes_cli.test_kanban_delivery_review_cards import (  # noqa: F401  (world is a fixture)
    CHECK, _cards, _project, _published, _raw, _ready, _state, world,
)

FLAKY = ("tests/tools/test_zombie_process_cleanup.py::TestDelegationCleanup::"
         "test_timed_out_child_keeps_relay_session_until_its_turn_exits")
UNLISTED = "tests/hermes_cli/test_kanban_db.py::test_no_policy_lists_this_one"
RERUN = f"/repos/{REPO}/actions/jobs/903/rerun"
REWORK = "SELECT * FROM tasks WHERE title LIKE 'Rework:%'"


@pytest.fixture
def red(world, monkeypatch):
    """The world with H's required check red, and GitHub's table holding H's one workflow run 77: the
    summary job 900 (the required check run) failed at "Evaluate job results", slice 1 (901) passed and
    slice 3 (903) failed at its test step, all at gh["attempt"], beside ci.yaml's timing report (904), which
    runs after the summary, at gh["timing"] (status, conclusion). Job 903's check run carries the count
    (gh["count"], else the number of gh["tests"]) and then each of gh["tests"] under the fixed title, beside
    one annotation of another title. Each rerun call is answered 201 and kept in gh["reruns"]."""
    kb, root, repo, gh = world
    from hermes_cli import kanban_delivery_github as transport

    table = transport._exchange  # the world's: the pull request and H's check runs
    gh.update(check=("completed", "failure"), attempt=1, tests=[FLAKY], count=None, reruns=[],
              timing=("completed", "success"))

    def exchange(method, target, authorization, payload=None):
        status, value = table(method, target, authorization, payload)
        path, head = urlsplit(target).path, gh["head"]
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
            return 201, None
        return status, value

    monkeypatch.setattr(transport, "_exchange", exchange)
    return world


def _waited(kb, tid, code, db=None):
    return [payload for kind, payload in _events(kb, tid, db)
            if kind == "delivery_review_waiting" and payload["code"] == code]


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
    recorded once for H before it is sent, with no rework item. A pass that still sees the first attempt
    red sends nothing more, and the green rerun gets H its review cards."""
    kb, root, repo, gh = red
    tid, head = _ready(red)
    events = _events(kb, tid)

    for _ in range(3):
        _tick(kb)

    outcome = {"delivery_id": 1, "head": head, "pull_request_number": 41}
    assert _events(kb, tid)[len(events):] == [
        ("delivery_review_waiting", dict(outcome, code="red_check_waiting",
                                         check_runs=[{"id": 900, "conclusion": "failure"}])),
        ("delivery_review_waiting", dict(outcome, code="red_check_rerun", jobs=[903], tests=[FLAKY], attempt=1))]
    assert gh["reruns"] == [RERUN] and _raw(kb.kanban_db_path(), REWORK) == []
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

    assert gh["reruns"] == [] and _raw(kb.kanban_db_path(), REWORK) == []
    assert _waited(kb, tid, "red_check_rerun") == [] and _waited(kb, tid, "red_check_returned") == []
    gh["timing"] = ("completed", "success")
    _tick(kb)
    assert gh["reruns"] == [RERUN] and len(_waited(kb, tid, "red_check_rerun")) == 1


def test_a_second_red_on_the_same_head_returns_the_work_once(red):
    """The rerun of H failed again: red_check_returned is recorded once with the failed tests, and one
    rework item goes to the builder through the review handback's follow-up record; no second rerun."""
    kb, root, repo, gh = red
    tid, head = _ready(red)
    _tick(kb)
    gh["attempt"] = 2  # GitHub reran slice 3 and it failed again

    for _ in range(3):
        _tick(kb)

    (item,) = _raw(kb.kanban_db_path(), REWORK)
    assert _waited(kb, tid, "red_check_returned") == [{
        "delivery_id": 1, "head": head, "pull_request_number": 41, "code": "red_check_returned",
        "reason": "rerun_used", "tests": [FLAKY], "rework": item["id"]}]
    assert gh["reruns"] == [RERUN] and len(_waited(kb, tid, "red_check_rerun")) == 1
    assert _raw(kb.kanban_db_path(), "SELECT COUNT(*) AS n FROM task_events WHERE task_id = ? "
                "AND kind = 'review_followup_recorded'", tid) == [{"n": 1}]
    assert _state(kb, head) == "open" and _cards(kb) == []


def test_an_unlisted_failure_returns_the_work_at_once(red):
    """A failed test the policy does not list: nothing is rerun, and in the same pass the work returns
    once to its original builder as one triage rework item under the source card, naming the failed
    test in plain words."""
    kb, root, repo, gh = red
    tid, head = _ready(red)
    gh["tests"] = [UNLISTED]

    _tick(kb)
    _tick(kb)

    db = kb.kanban_db_path()
    (item,) = _raw(db, REWORK)
    assert gh["reruns"] == [] and _waited(kb, tid, "red_check_rerun") == []
    assert [(p["reason"], p["tests"], p["rework"]) for p in _waited(kb, tid, "red_check_returned")] == [
        ("not_flaky", [UNLISTED], item["id"])]
    assert (item["assignee"], item["status"]) == (IMPLEMENTER, "triage")
    assert f"The failed tests are: {UNLISTED}." in item["body"] and head in item["body"]
    assert _raw(db, "SELECT parent_id FROM task_links WHERE child_id = ?", item["id"]) == [{"parent_id": tid}]


def test_the_work_returns_on_the_ticked_board_under_the_worker_environment(red, monkeypatch):
    """The dispatcher's worker environment for proj-a (its board pinned, its card set, a profile home under
    the root) ticks the board other: the root registry and every board still resolve, and the return and
    its one rework item land on other and on no other board, by a count read past the kernel."""
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
    rework = "SELECT COUNT(*) AS n FROM tasks WHERE title LIKE 'Rework:%'"
    boards = (other_db, own_db, root / "kanban.db")
    assert [(_raw(db, returned)[0]["n"], _raw(db, rework)[0]["n"]) for db in boards] == [(1, 1), (0, 0), (0, 0)]
    assert [(item["assignee"], item["project_id"]) for item in _raw(other_db, REWORK)] == [
        (IMPLEMENTER, "delivery-project")]
    assert gh["reruns"] == [] and not (profile_home / "kanban.db").exists()
