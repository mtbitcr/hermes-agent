"""Card 2: once CI on the exact published head is done, the dispatcher's delivery step creates that
head's review cards, requests its one flaky rerun or returns the work for changes; no operator.
Real boards and git; GitHub is the real transport with only its exchange and App token replaced by a
table, so every call passes the allowlist. The policy's route is pinned to values no code hard-codes."""

from __future__ import annotations

import sqlite3

import pytest

from tests.hermes_cli.test_kanban_delivery import (  # noqa: F401  (board is a fixture)
    IMPLEMENTER, REVIEWER, _become, _git, _rework, _spawned_worker_env, _write_config, board,
)
from tests.hermes_cli.test_kanban_delivery_github import REPO
from tests.hermes_cli.test_kanban_delivery_publish_step import _approved, _events, _tick

CHECK = "All required checks pass"
SLICE = "Python tests / Run tests slice 3/12"
FLAKY = "tests/tools/test_zombie_process_cleanup.py::TestDelegationCleanup::test_timed_out_child_keeps_relay_session_until_its_turn_exits"
OTHER = "tests/hermes_cli/test_kanban_db.py::test_a_real_failure"
ROUTE = {"provider_override": "policy-provider", "model_override": "policy-model", "reasoning_effort": "max"}
CI_READS = ("check-runs", "/actions/")


@pytest.fixture
def world(board, monkeypatch):
    """The board's repository with its GitHub origin, delivery on, the team policy's review route, and
    GitHub as a table: the pull request, the required check on H, and H's workflow run and jobs."""
    kb, root, repo = board
    from hermes_cli import kanban_delivery as kd
    from hermes_cli import kanban_delivery_github as transport
    from plugins.dashboard_auth.raphael_workspace import model_policy

    _git(repo, "remote", "add", "origin", f"https://github.com/{REPO}.git")
    _write_config(root, enabled=True)
    gh = {"calls": [], "steps": [], "pull_head": None, "check": ("completed", "success"), "attempt": 1,
          "failed": None}

    def jobs(head):  # the slice failed at its test step; the summary check failed only at its evaluation
        return [{"id": job, "name": name, "head_sha": head, "status": "completed", "conclusion": "failure",
                 "run_attempt": gh["attempt"], "steps": [{"name": step, "conclusion": "failure"}]}
                for job, name, step in ((81, SLICE, "Run tests (slice 3/12)"), (82, CHECK, "Evaluate job results"))]

    def exchange(method, target, authorization, payload=None):
        path = target.split("?")[0]
        gh["calls"].append((method, path))
        head = gh["head"]
        if (method, path) == ("GET", f"/repos/{REPO}/pulls/41"):
            return 200, {"number": 41, "state": "open", "head": {"sha": gh["pull_head"] or head}}
        if method == "GET" and path.endswith("/check-runs"):
            status, conclusion = gh["check"]
            return 200, {"total_count": 1, "check_runs": [{"id": 900 + gh["attempt"], "name": CHECK,
                         "head_sha": path.split("/")[-2], "status": status, "conclusion": conclusion}]}
        if (method, path) == ("GET", f"/repos/{REPO}/actions/runs"):
            return 200, {"total_count": 1, "workflow_runs": [{"id": 70, "head_sha": head, "run_attempt": gh["attempt"]}]}
        if (method, path) == ("GET", f"/repos/{REPO}/actions/runs/70/jobs"):
            return 200, {"total_count": 2, "jobs": jobs(head)}
        if method == "POST" and path.endswith("/rerun"):
            gh["attempt"], gh["check"] = gh["attempt"] + 1, ("queued", None)
            return 201, {}
        return 404, None

    def token(self):  # kept on the transport, as the real one is, so its answers are redacted of it
        gh["steps"].append(self.step)
        self._token = "token"
        return self._token

    monkeypatch.setattr(transport, "_exchange", exchange)
    monkeypatch.setattr(transport.GitHubTransport, "_installation_token", token)
    monkeypatch.setattr(kd, "_failed_tests", lambda github, job: gh["failed"] and (list(gh["failed"]), len(gh["failed"])))
    monkeypatch.setattr(model_policy, "resolve_task_assignment", lambda profile, tier: model_policy.ModelAssignment(
        profile=profile, provider=ROUTE["provider_override"], model=ROUTE["model_override"], model_label="policy",
        reasoning_effort=ROUTE["reasoning_effort"]))
    return kb, root, repo, gh


def _published(kb, repo, gh, tid, head, board=None):
    """What card 1's publish stores for H: the ledger on the row and delivery_bound on the source card."""
    conn = kb.connect(board=board)
    try:
        with kb.write_txn(conn):
            conn.execute(
                "UPDATE kanban_deliveries SET pull_request_number = 41, pull_request_head = ?, "
                "pull_request_state = 'open', pull_request_branch = ? WHERE source_task_id = ? AND source_head = ?",
                (head, "delivery/" + tid, tid, head))
            kb._append_event(conn, tid, "delivery_bound", {
                "source_task_id": tid, "head": head, "base_commit": _git(repo, "rev-parse", "main")})
    finally:
        conn.close()
    gh["head"] = head


def _ready(world, tier=2, name="feature"):
    kb, root, repo, gh = world
    tid, head = _approved(kb, repo, name)
    _raw(kb.kanban_db_path(), "UPDATE tasks SET risk_tier = ? WHERE id = ?", tier, tid)
    _published(kb, repo, gh, tid, head)
    return tid, head


def _raw(db, statement, *params):
    """Read or write the board file past the kernel: the independent count."""
    raw = sqlite3.connect(str(db))
    raw.row_factory = sqlite3.Row
    try:
        return [dict(row) for row in raw.execute(statement, params)]
    finally:
        raw.commit()
        raw.close()


def _cards(kb, db=None):
    return _raw(db or kb.kanban_db_path(), "SELECT * FROM tasks WHERE idempotency_key LIKE 'review:%' "
                "ORDER BY idempotency_key")


def _state(kb, head, db=None):
    return _raw(db or kb.kanban_db_path(), "SELECT pull_request_state FROM kanban_deliveries "
                "WHERE source_head = ?", head)[0]["pull_request_state"]


def _ci_reads(gh):
    return [call for call in gh["calls"] if any(part in call[1] for part in CI_READS)]


@pytest.mark.parametrize("tier", [0, 1])
def test_tiers_0_and_1_get_one_combined_card_on_the_exact_head(world, tier):
    kb, root, repo, gh = world
    tid, head = _ready(world, tier)

    _tick(kb)

    cards = _cards(kb)
    assert [(c["idempotency_key"], c["responsibility"]) for c in cards] == [(f"review:{tid}:{head}:R15", "R15")]
    assert "correctness and security" in cards[0]["body"] and f"Head: {head}" in cards[0]["body"]
    assert _state(kb, head) == "review_cards_created"
    assert _events(kb, tid)[-1] == ("delivery_review_cards_created", {
        "delivery_id": 1, "head": head, "pull_request_number": 41, "cards": [cards[0]["id"]]})


@pytest.mark.parametrize("tier", [2, None])
def test_tier_2_and_a_missing_tier_get_r15_and_r12_cards(world, tier):
    kb, root, repo, gh = world
    tid, head = _ready(world, tier)

    _tick(kb)

    cards = _cards(kb)
    assert [(c["idempotency_key"], c["responsibility"]) for c in cards] == [
        (f"review:{tid}:{head}:R12", "R12"), (f"review:{tid}:{head}:R15", "R15")]
    assert ["security" in c["body"] and "correctness" not in c["body"] for c in cards] == [True, False]
    assert ("Risk not recorded" in cards[0]["body"]) is (tier is None)
    assert _state(kb, head) == "review_cards_created"


def test_cards_carry_the_command_fields_and_their_own_references(world):
    kb, root, repo, gh = world
    tid, head = _ready(world)
    project = _raw(kb.kanban_db_path(), "SELECT project_id FROM tasks WHERE id = ?", tid)[0]["project_id"]

    _tick(kb)

    conn = kb.connect()
    try:  # the review command's own call (section 2(c)), with the team policy's route
        command = kb.create_task(
            conn, title="command", body="command", assignee=REVIEWER, responsibility="R15", created_by="operator",
            workspace_kind="worktree", execution_tier="routine", board="default", project_id=project, owned_paths=[],
            **ROUTE)
        parents = [row[0] for row in conn.execute(
            "SELECT parent_id FROM task_links WHERE child_id IN (SELECT id FROM tasks WHERE idempotency_key "
            "LIKE 'review:%')")]
    finally:
        conn.close()
    fields = ("assignee", "workspace_kind", "workspace_path", "execution_tier", "model_override",
              "provider_override", "reasoning_effort", "owned_paths", "project_id", "tenant", "model_policy_lock")
    expected = _raw(kb.kanban_db_path(), "SELECT * FROM tasks WHERE id = ?", command)[0]
    for card in _cards(kb):
        assert {f: card[f] for f in fields} == {f: expected[f] for f in fields}
        assert (card["assignee"], card["created_by"], card["execution_tier"]) == (REVIEWER, "kanban-delivery", "routine")
        assert {f: card[f] for f in ROUTE} == ROUTE
        for reference in (f"Source card: {tid}", f"Head: {head}", f"Base commit: {_git(repo, 'rev-parse', 'main')}",
                          f"Repository: {REPO}", "Pull request: 41", f"Branch: delivery/{tid}", "Risk tier: 2",
                          f"CI on {head}: passed ({CHECK})"):
            assert reference in card["body"]
    assert parents == [tid, tid]


def test_the_route_and_reviewer_come_from_the_team_policy(world, monkeypatch):
    kb, root, repo, gh = world
    from plugins.dashboard_auth.raphael_workspace import model_policy

    tid, head = _ready(world)
    monkeypatch.setattr(model_policy, "reviewer_profile_ids", lambda: ())  # the policy names no reviewer
    _tick(kb)
    assert (_cards(kb), gh["calls"], _state(kb, head)) == ([], [], "open")

    monkeypatch.setattr(model_policy, "reviewer_profile_ids", lambda: (REVIEWER,))
    ROUTE_NOW = {"provider_override": "policy-provider", "model_override": "policy-model-2", "reasoning_effort": "max"}
    monkeypatch.setattr(model_policy, "resolve_task_assignment", lambda profile, tier: model_policy.ModelAssignment(
        profile=profile, provider=ROUTE_NOW["provider_override"], model=ROUTE_NOW["model_override"],
        model_label="policy", reasoning_effort=ROUTE_NOW["reasoning_effort"]))
    _tick(kb)
    assert [({f: c[f] for f in ROUTE_NOW}, c["assignee"]) for c in _cards(kb)] == [(ROUTE_NOW, REVIEWER)] * 2


def test_no_card_while_ci_is_pending(world):
    kb, root, repo, gh = world
    tid, head = _ready(world)
    gh["check"] = ("in_progress", None)
    events = _events(kb, tid)

    _tick(kb)
    _tick(kb)

    assert (_cards(kb), _state(kb, head), _events(kb, tid)) == ([], "open", events)
    assert [call for call in gh["calls"] if call[0] == "POST"] == [] and set(gh["steps"]) == {"read_checks"}
    gh["check"] = ("completed", "success")  # a later delivery step tries again
    _tick(kb)
    assert len(_cards(kb)) == 2 and _state(kb, head) == "review_cards_created"


def test_delivery_off_does_nothing(world):
    kb, root, repo, gh = world
    tid, head = _ready(world)
    events = _events(kb, tid)
    _write_config(root, enabled=False)

    _tick(kb)

    assert (_cards(kb), gh["calls"], gh["steps"], _state(kb, head), _events(kb, tid)) == ([], [], [], "open", events)


def test_a_stale_head_gets_no_cards(world):
    kb, root, repo, gh = world
    tid, head = _ready(world)
    gh["pull_head"] = "f" * 40  # the pull request has moved away from the published head

    _tick(kb)

    assert (_cards(kb), _state(kb, head), _ci_reads(gh)) == ([], "open", [])


@pytest.mark.parametrize("failed", [[OTHER], [FLAKY, OTHER], None])
def test_ci_failure_goes_to_rework_at_once_with_no_card(world, failed):
    """Superseding test_ci_failure_is_written_into_the_card (owner decision): a red CI with a failure
    off the flaky list, or whose failed tests cannot be read, goes back to rework, with no card."""
    kb, root, repo, gh = world
    tid, head = _ready(world)
    gh["check"], gh["failed"] = ("completed", "failure"), failed

    _tick(kb)
    _tick(kb)

    assert _cards(kb) == [] and _state(kb, head) == "returned_for_changes"
    assert [call for call in gh["calls"] if call[0] == "POST"] == [] and "rerun_flaky" not in gh["steps"]
    kind, payload = _events(kb, tid)[-1]
    assert (kind, payload["head"], payload["pull_request_number"]) == ("delivery_returned_for_changes", head, 41)
    source = kb.get_task(kb.connect(), tid)
    assert (source.status, source.assignee) == ("ready", IMPLEMENTER)


def test_a_flaky_only_red_ci_is_rerun_once_then_a_second_red_goes_to_rework(world):
    kb, root, repo, gh = world
    tid, head = _ready(world)
    gh["check"], gh["failed"] = ("completed", "failure"), [FLAKY]

    _tick(kb)

    assert [call for call in gh["calls"] if call[0] == "POST"] == [("POST", f"/repos/{REPO}/actions/jobs/81/rerun")]
    assert "rerun_flaky" in gh["steps"] and _cards(kb) == [] and _state(kb, head) == "rerun_requested"
    assert _events(kb, tid)[-1] == ("delivery_rerun_requested", {
        "delivery_id": 1, "head": head, "pull_request_number": 41, "jobs": [81], "tests": [FLAKY]})
    _tick(kb)  # the rerun is still running
    assert _state(kb, head) == "rerun_requested"

    gh["check"] = ("completed", "failure")  # the one rerun is red again, on the same flaky test
    _tick(kb)
    _tick(kb)

    assert len([call for call in gh["calls"] if call[0] == "POST"]) == 1
    assert _cards(kb) == [] and _state(kb, head) == "returned_for_changes"
    assert [kind for kind, _ in _events(kb, tid)][-2:] == ["delivery_rerun_requested", "delivery_returned_for_changes"]


def test_a_green_rerun_gets_its_review_cards(world):
    kb, root, repo, gh = world
    tid, head = _ready(world)
    gh["check"], gh["failed"] = ("completed", "failure"), [FLAKY]
    _tick(kb)
    gh["check"] = ("completed", "success")

    _tick(kb)

    assert len(_cards(kb)) == 2 and _state(kb, head) == "review_cards_created"


def test_repeated_passes_create_no_duplicate_cards(world):
    kb, root, repo, gh = world
    tid, head = _ready(world)
    _tick(kb)
    cards, calls, events = _cards(kb), len(gh["calls"]), _events(kb, tid)

    _tick(kb)
    _tick(kb)

    assert (_cards(kb), len(gh["calls"]), _events(kb, tid)) == (cards, calls, events)  # no GitHub read either
    # A pass that died after creating the cards and before recording them finds them by their keys.
    _raw(kb.kanban_db_path(), "UPDATE kanban_deliveries SET pull_request_state = 'open'")
    _tick(kb)
    assert [c["id"] for c in _cards(kb)] == [c["id"] for c in cards] and _state(kb, head) == "review_cards_created"


def test_a_new_head_gets_new_cards_and_old_cards_are_untouched(world):
    kb, root, repo, gh = world
    tid, first = _ready(world)
    _tick(kb)
    old = _cards(kb)

    second = _rework(kb, repo, tid)
    _published(kb, repo, gh, tid, second)
    _tick(kb)

    cards = _cards(kb)
    assert [c for c in cards if first in c["idempotency_key"]] == old
    assert sorted(c["idempotency_key"] for c in cards if second in c["idempotency_key"]) == [
        f"review:{tid}:{second}:R12", f"review:{tid}:{second}:R15"]
    assert (_state(kb, first), _state(kb, second)) == ("returned_for_changes", "review_cards_created")


def test_cards_land_on_the_ticked_board_under_the_worker_environment(world, monkeypatch):
    """The dispatcher's worker environment for proj-a (its board pinned, its card set, a profile home
    under the root) ticks the board other: the cards land on other and on no other board."""
    kb, root, repo, gh = world
    kb.create_board("proj-a")
    kb.create_board("other")
    own_db = root / "kanban" / "boards" / "proj-a" / "kanban.db"
    other_db = root / "kanban" / "boards" / "other" / "kanban.db"
    profile_home = root / "profiles" / IMPLEMENTER
    profile_home.mkdir(parents=True)
    _write_config(profile_home, enabled=True)
    tid, head = _approved(kb, repo, "beta", board="other")
    _published(kb, repo, gh, tid, head, board="other")
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
    ticked = kb.connect(db_path=other_db)
    try:
        kb.dispatch_once(ticked, board="other", spawn_fn=lambda *a, **k: 1)
    finally:
        ticked.close()

    count = "SELECT COUNT(*) AS n FROM tasks WHERE idempotency_key LIKE 'review:%'"
    assert [_raw(db, count)[0]["n"] for db in (other_db, own_db, root / "kanban.db")] == [2, 0, 0]
    assert {c["idempotency_key"] for c in _cards(kb, other_db)} == {
        f"review:{tid}:{head}:R15", f"review:{tid}:{head}:R12"}
    assert _state(kb, head, other_db) == "review_cards_created"
