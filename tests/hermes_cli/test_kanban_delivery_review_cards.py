"""Card 2: once CI on the exact published head is done, the dispatcher's delivery step creates that
head's review cards, or waits while a required check is red; no operator.
Real boards and git; GitHub is the real transport with only its exchange and App token replaced by a
table, so every call passes the allowlist. The team policy's route is an admitted one no code names."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import pytest

from tests.hermes_cli.test_kanban_delivery import (  # noqa: F401  (board is a fixture)
    IMPLEMENTER, REVIEWER, _become, _git, _rework, _spawned_worker_env, _write_config, board,
)
from tests.hermes_cli.test_kanban_delivery_github import REPO
from tests.hermes_cli.test_kanban_delivery_publish_step import _approved, _events, _tick

CHECK = "All required checks pass"
SLICE = "Python tests / Run tests slice 3/12"
LINT = "Lint / ruff"
ODD = "Python tests / Run tests slice 3/16"  # a test job of a name no list holds
ROUTE = {"provider_override": "openai-codex", "model_override": "gpt-6.1-sol", "reasoning_effort": "max"}
PAGED = {"check_runs": "/check-runs", "workflow_runs": "/actions/runs", "jobs": "/runs/70/jobs"}


def _policy_route(monkeypatch, route):
    """The team policy resolves the review route to ``route``."""
    from plugins.dashboard_auth.raphael_workspace import model_policy

    monkeypatch.setattr(model_policy, "resolve_task_assignment", lambda profile, tier: model_policy.ModelAssignment(
        profile=profile, provider=route["provider_override"], model=route["model_override"], model_label="policy",
        reasoning_effort=route["reasoning_effort"]))


@pytest.fixture
def world(board, monkeypatch):
    """The board's repository with its GitHub origin, delivery on, the team policy's review route, and
    GitHub as a table: the pull request, the required check on H, and H's workflow run 70 with its jobs.
    gh["pad"] puts 100 other items before the real ones of a collection, so those are on page 2."""
    kb, root, repo = board
    from hermes_cli import kanban_delivery_github as transport

    _git(repo, "remote", "add", "origin", f"https://github.com/{REPO}.git")
    _write_config(root, enabled=True)
    gh = {"calls": [], "pages": [], "steps": [], "pull_head": None, "check": ("completed", "success"),
          "failed": None, "job": "completed", "attempt": 1, "pad": None, "short": None, "move_after": None,
          "during": None}

    def job(job_id, name, head):  # the summary check fails at its evaluation while CI is red; a rerun renumbers
        status = gh["job"] if name == gh["failed"] else "completed"
        red = name == gh["failed"] or (name == CHECK and gh["check"][1] == "failure")
        return {"id": job_id + 1000 * (gh["attempt"] - 1), "run_attempt": gh["attempt"], "name": name, "run_id": 70,
                "head_sha": head, "status": status,
                "conclusion": None if status != "completed" else "failure" if red else "success"}

    def listed(key, items, page):
        if gh["pad"] == key:
            items = [{"id": 1000 + n, "name": f"other {n}", "run_id": 70, "head_sha": gh["head"],
                      "status": "completed", "conclusion": "success"} for n in range(100)] + items
        return 200, {"total_count": len(items) + (gh["short"] == key), key: items[(page - 1) * 100:page * 100]}

    def exchange(method, target, authorization, payload=None):
        path, query = urlsplit(target).path, parse_qs(urlsplit(target).query)
        head, page = gh["head"], int(query.get("page", ["1"])[0])
        gh["calls"].append((method, path))
        gh["pages"].append((path, page))
        if (method, path) == ("GET", f"/repos/{REPO}/pulls/41"):
            return 200, {"number": 41, "state": "open", "head": {"sha": gh["pull_head"] or head}}
        if gh["move_after"] and path.endswith(gh["move_after"]):  # the pull request moves on while CI is read
            gh["pull_head"] = "f" * 40
        if gh["during"] and path.endswith(PAGED["jobs"]):  # the source card or CI changes while the jobs are read
            gh["during"]()
        if method == "GET" and path.endswith("/check-runs"):
            status, conclusion = gh["check"]
            return listed("check_runs", [{"id": 900, "name": CHECK, "head_sha": path.split("/")[-2],
                                          "status": status, "conclusion": conclusion}], page)
        if (method, path) == ("GET", f"/repos/{REPO}/actions/runs"):
            return listed("workflow_runs", [{"id": 70, "head_sha": head}], page)
        if (method, path) == ("GET", f"/repos/{REPO}/actions/runs/70/jobs"):
            jobs = [job(81, SLICE, head), job(83, LINT, head), job(84, ODD, head), job(82, CHECK, head)]
            return listed("jobs", jobs, page)
        if method == "GET" and path.endswith("/jobs"):
            return 200, {"total_count": 0, "jobs": []}
        return 404, None

    def token(self):  # kept on the transport, as the real one is, so its answers are redacted of it
        gh["steps"].append(self.step)
        self._token = "token"
        return self._token

    monkeypatch.setattr(transport, "_exchange", exchange)
    monkeypatch.setattr(transport.GitHubTransport, "_installation_token", token)
    _policy_route(monkeypatch, ROUTE)
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


def _project(db, tid):
    """The source's project as a worker profile sees it: in no projects.db there, its branch named for it."""
    _raw(db, "UPDATE tasks SET project_id = 'delivery-project', branch_name = ? WHERE id = ?",
         f"delivery-project/{tid}", tid)


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


def _source(kb, tid):
    return _raw(kb.kanban_db_path(), "SELECT * FROM tasks WHERE id = ?", tid)[0]


@pytest.mark.parametrize("tier, lenses", [
    (0, {"R15": "correctness and security"}), (1, {"R15": "correctness and security"}),
    (2, {"R12": "security", "R15": "correctness"}), (None, {"R12": "security", "R15": "correctness"})])
def test_the_recorded_tier_alone_decides_the_cards_of_the_exact_head(world, tier, lenses):
    kb, root, repo, gh = world
    tid, head = _ready(world, tier)

    _tick(kb)

    cards = _cards(kb)
    assert [(c["idempotency_key"], c["responsibility"], c["title"]) for c in cards] == [
        (f"review:{tid}:{head}:{r}", r, f"Review ({lens}) of {tid} at {head}") for r, lens in lenses.items()]
    assert {("Risk not recorded" in c["body"], f"Head: {head}" in c["body"]) for c in cards} == {(tier is None, True)}
    kind, payload = _events(kb, tid)[-1]
    assert (kind, payload["head"], sorted(payload["cards"])) == (
        "delivery_review_cards_created", head, sorted(c["id"] for c in cards))
    assert _state(kb, head) == "review_cards_created"


def test_cards_carry_the_command_fields_after_the_same_route_guard(world):
    kb, root, repo, gh = world
    tid, head = _ready(world)
    project = _raw(kb.kanban_db_path(), "SELECT project_id FROM tasks WHERE id = ?", tid)[0]["project_id"]

    _tick(kb)

    conn = kb.connect()
    try:  # the review command's own call (section 2(c)) with the team policy's route, then the same guard
        command = kb.create_task(
            conn, title="command", body="command", assignee=REVIEWER, responsibility="R15", created_by="operator",
            workspace_kind="worktree", execution_tier="routine", board="default", project_id=project, owned_paths=[],
            **ROUTE)
        with kb.write_txn(conn):
            assert kb.authorize_executable_transition(conn, command)
        parents = [row[0] for row in conn.execute(
            "SELECT parent_id FROM task_links WHERE child_id IN (SELECT id FROM tasks WHERE idempotency_key "
            "LIKE 'review:%')")]
    finally:
        conn.close()
    fields = ("assignee", "workspace_kind", "workspace_path", "execution_tier", "model_override",
              "provider_override", "reasoning_effort", "owned_paths", "project_id", "tenant", "model_policy_lock")
    expected = _raw(kb.kanban_db_path(), "SELECT * FROM tasks WHERE id = ?", command)[0]
    assert expected["model_policy_lock"]
    for card in _cards(kb):
        assert {f: card[f] for f in fields} == {f: expected[f] for f in fields}
        assert (card["assignee"], card["created_by"], card["status"]) == (REVIEWER, "kanban-delivery", "ready")
        assert "pull request 41 on the code host" in card["body"] and "post the review on that commit" in card["body"]
        for reference in (f"Source card: {tid}", f"Head: {head}", f"Base commit: {_git(repo, 'rev-parse', 'main')}",
                          f"Repository: {REPO}", "Pull request: 41", f"Branch: delivery/{tid}", "Risk tier: 2",
                          f"CI on {head}: passed ({CHECK})"):
            assert reference in card["body"]
    assert parents == [tid, tid]


@pytest.mark.parametrize("model, created, after", [
    (ROUTE["model_override"], ("ready", None, True, []), "running"),
    ("unadmitted", ("blocked", "needs_input", False, [{"kind": "blocked"}]), "blocked")])
def test_a_created_card_is_claimed_on_its_admitted_route_and_parked_on_any_other(world, monkeypatch, model, created,
                                                                                 after):
    """The kernel's route guard runs in the transaction that creates the card: on the team policy's admitted
    route the next dispatcher pass claims it, in its own worktree under the source's project; the parking
    control, on a route the policy does not admit, is parked there with its reason and never spawned."""
    kb, root, repo, gh = world
    (root / "profiles" / REVIEWER).mkdir(parents=True)
    _policy_route(monkeypatch, dict(ROUTE, model_override=model))
    tid, head = _ready(world, tier=1)
    _project(kb.kanban_db_path(), tid)
    _tick(kb)
    card = _cards(kb)[0]
    blocked = _raw(kb.kanban_db_path(), "SELECT kind FROM task_events WHERE task_id = ? AND kind = 'blocked'",
                   card["id"])
    assert (card["status"], card["block_kind"] or None, bool(card["model_policy_lock"]), blocked) == created

    result = _tick(kb)

    assert [run[:2] for run in result.spawned] == ([(card["id"], REVIEWER)] if after == "running" else [])
    assert (result.skipped_route_unproven, _state(kb, head), _cards(kb)[0]["status"]) == (
        [], "review_cards_created", after)


def test_the_route_and_reviewer_come_from_the_team_policy_and_delivery_off_does_nothing(world, monkeypatch):
    kb, root, repo, gh = world
    from plugins.dashboard_auth.raphael_workspace import model_policy

    tid, head = _ready(world)
    events = _events(kb, tid)
    _write_config(root, enabled=False)
    _tick(kb)
    _write_config(root, enabled=True)
    monkeypatch.setattr(model_policy, "reviewer_profile_ids", lambda: ())  # the policy names no reviewer
    _tick(kb)
    assert (_cards(kb), gh["calls"], gh["steps"], _state(kb, head), _events(kb, tid)) == ([], [], [], "open", events)

    monkeypatch.setattr(model_policy, "reviewer_profile_ids", lambda: (REVIEWER,))
    route = {"provider_override": "anthropic", "model_override": "claude-opus-5-5", "reasoning_effort": "max"}
    _policy_route(monkeypatch, route)
    _tick(kb)
    assert [({f: c[f] for f in route}, c["assignee"], c["status"]) for c in _cards(kb)] == [(route, REVIEWER, "ready")] * 2


@pytest.mark.parametrize("failed, job", [
    (None, "completed"), (SLICE, "completed"), (LINT, "queued"), (ODD, "in_progress")])
def test_no_card_while_ci_is_pending(world, failed, job):
    """The required check on H, read last, alone decides. While it is not completed nothing is recorded,
    whatever the jobs say (all green, one red, one queued or running); once it is completed with success
    the cards are made, though a job's latest attempt may still be queued or running."""
    kb, root, repo, gh = world
    tid, head = _ready(world)
    gh["check"], gh["failed"], gh["job"] = ("in_progress", None), failed, job
    events = _events(kb, tid)

    _tick(kb)
    _tick(kb)

    assert (_cards(kb), _state(kb, head), _events(kb, tid), set(gh["steps"])) == ([], "open", events, {"read_checks"})
    gh["check"] = ("completed", "success")  # a later delivery step tries again
    _tick(kb)
    assert len(_cards(kb)) == 2 and _state(kb, head) == "review_cards_created"


@pytest.mark.parametrize("moved, failed", [("before", None), (PAGED["jobs"], None), (PAGED["jobs"], LINT)])
def test_a_head_the_pull_request_left_gets_no_outcome(world, moved, failed):
    """The pull request left H before CI was read, or while it was read: it is read again right before an
    outcome is stored, so neither H's cards nor its waiting reason is stored."""
    kb, root, repo, gh = world
    tid, head = _ready(world)
    gh["move_after"], gh["failed"], gh["pull_head"] = moved, failed, "f" * 40 if moved == "before" else None
    gh["check"] = ("completed", "failure" if failed else "success")
    events = _events(kb, tid)

    _tick(kb)

    assert (_cards(kb), _state(kb, head), _events(kb, tid), _source(kb, tid)["status"]) == ([], "open", events, "done")
    pulls = [path for _, path in gh["calls"] if path.endswith("/pulls/41")]
    assert (len(pulls), len(gh["calls"]) > len(pulls)) == ((1, False) if moved == "before" else (2, True))


@pytest.mark.parametrize("key, short", [("check_runs", False), ("workflow_runs", False), ("jobs", False), ("jobs", True)])
def test_each_collection_is_read_page_by_page_to_its_total_count(world, key, short):
    """101 items, the real ones (the required check, H's workflow run, its jobs) on page 2; a page short
    of the total count GitHub states is unreadable, so nothing is decided."""
    kb, root, repo, gh = world
    tid, head = _ready(world)
    gh["pad"], gh["short"] = key, short and key

    _tick(kb)

    assert (len(_cards(kb)), _state(kb, head)) == ((0, "open") if short else (2, "review_cards_created"))
    assert [page for path, page in gh["pages"] if path.endswith(PAGED[key])] == [1, 2]


@pytest.mark.parametrize("failed", [SLICE, LINT, ODD, CHECK])
def test_a_red_check_waits_once_per_head_until_a_green_rerun(world, failed):
    """A red required check, whatever job failed (a listed test slice, lint, a test job of a name no list
    holds, or only the required aggregate job itself), leaves the delivery waiting: red_check_waiting with
    the failed job, once for H, with no POST and the source card unchanged. Later passes read CI again, so
    a green rerun on the code host gets H its review cards."""
    kb, root, repo, gh = world
    tid, head = _ready(world)
    gh["check"], gh["failed"] = ("completed", "failure"), failed
    events, source = _events(kb, tid), _source(kb, tid)

    _tick(kb)
    _tick(kb)

    assert _events(kb, tid)[len(events):] == [("delivery_review_waiting", {
        "delivery_id": 1, "head": head, "pull_request_number": 41, "code": "red_check_waiting",
        "jobs": [{SLICE: 81, LINT: 83, ODD: 84, CHECK: 82}[failed]]})]
    assert (_cards(kb), _state(kb, head), _source(kb, tid)) == ([], "open", source)
    assert [call for call in gh["calls"] if call[0] != "GET"] == [] and set(gh["steps"]) == {"read_checks"}
    gh["check"], gh["failed"] = ("completed", "success"), None
    _tick(kb)
    assert len(_cards(kb)) == 2 and _state(kb, head) == "review_cards_created"


@pytest.mark.parametrize("attempt, job, named", [(1, "completed", 81), (2, "completed", 1081), (2, "queued", 1082)])
def test_a_check_that_turns_red_while_the_jobs_are_read_gets_no_card(world, attempt, job, named):
    """The required check on H, successful at first, turns completed/failure while the jobs are read: their
    first attempt, or a rerun's newly numbered jobs of attempt 2 whose test slice failed or is still queued.
    The check is read after the jobs and alone decides, so no card is made and red_check_waiting names the
    failed job (the aggregate's own when it is the only one known), once for H."""
    kb, root, repo, gh = world
    tid, head = _ready(world)
    events, source = _events(kb, tid), _source(kb, tid)

    def during():
        gh["check"], gh["failed"], gh["job"], gh["attempt"] = ("completed", "failure"), SLICE, job, attempt

    gh["during"] = during
    for _ in range(3):
        _tick(kb)

    assert _events(kb, tid)[len(events):] == [("delivery_review_waiting", {
        "delivery_id": 1, "head": head, "pull_request_number": 41, "code": "red_check_waiting", "jobs": [named]})]
    assert (_cards(kb), _state(kb, head), _source(kb, tid)) == ([], "open", source)


@pytest.mark.parametrize("holder", ["someone-else", "archived", "this-step"])
def test_a_key_that_names_any_card_gives_a_conflict_and_no_new_card(world, holder):
    """No card is adopted by its key. When a review key of H already names a card (another's, an archived
    one, or the step's own after its row was opened again), no card is made, review_key_conflict with
    those cards is recorded once for H, and the delivery waits."""
    kb, root, repo, gh = world
    tid, head = _ready(world)
    if holder == "this-step":
        _tick(kb)
        _raw(kb.kanban_db_path(), "UPDATE kanban_deliveries SET pull_request_state = 'open'")
    else:
        with kb.connect_closing() as conn:
            other = kb.create_task(conn, title="Not a review", assignee=REVIEWER, created_by="someone-else",
                                   idempotency_key=f"review:{tid}:{head}:R15")
            assert holder != "archived" or kb.archive_task(conn, other)
    cards, events = [c["id"] for c in _cards(kb)], _events(kb, tid)

    _tick(kb)
    _tick(kb)

    assert ([c["id"] for c in _cards(kb)], _state(kb, head)) == (cards, "open")
    assert _events(kb, tid)[len(events):] == [("delivery_review_waiting", {
        "delivery_id": 1, "head": head, "pull_request_number": 41, "code": "review_key_conflict",
        "cards": sorted(cards)})]


def test_a_normal_pass_creates_the_cards_and_records_them_once(world):
    kb, root, repo, gh = world
    tid, head = _ready(world)
    _tick(kb)
    cards, calls, events = _cards(kb), len(gh["calls"]), _events(kb, tid)

    _tick(kb)
    _tick(kb)

    assert (_cards(kb), len(gh["calls"]), _events(kb, tid)) == (cards, calls, events)  # no GitHub read either
    assert [(kind, sorted(p["cards"])) for kind, p in events if kind.startswith("delivery_review")] == [
        ("delivery_review_cards_created", sorted(c["id"] for c in cards))]


@pytest.mark.parametrize("change, made", [
    ("left_done", []), ("tier_raised", []), ("project_moved", [("R15", "delivery-project")])])
def test_the_source_is_read_again_where_the_outcome_is_stored(world, change, made):
    """The transaction that would store the outcome reads the source card again. One that left done, or
    whose tier was raised the owner workspace's way (with no event), while CI was read gets no card and no
    record in that pass, and the next pass decides on the fresh card. The cards take their tier and project
    from that read, so a project moved meanwhile, which nothing compares, is theirs."""
    kb, root, repo, gh = world
    tid, head = _ready(world, tier=0)
    db, events = kb.kanban_db_path(), _events(kb, tid)

    def during():
        if change == "tier_raised":
            _raw(db, "UPDATE tasks SET risk_tier = 2 WHERE id = ?", tid)  # as the owner workspace does: no event
        elif change == "project_moved":
            _project(db, tid)
        else:
            with kb.connect_closing() as conn:
                assert kb.cas_transition_task(conn, tid, expected_status="done", to_status="blocked",
                                              expected_revision=kb.task_event_revision(conn, tid))["moved"]

    gh["during"] = during
    _tick(kb)
    gh["during"] = None

    assert [(c["responsibility"], c["project_id"]) for c in _cards(kb)] == made
    assert (_state(kb, head), len(_events(kb, tid)) - len(events)) == (
        ("review_cards_created", 1) if made else ("open", 0))
    _tick(kb)
    assert [(c["responsibility"], "Risk tier: 2" in c["body"]) for c in _cards(kb)] == (
        [("R12", True), ("R15", True)] if change == "tier_raised" else [(r, False) for r, _ in made])


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
    under the root) ticks the board other: the cards land on other and on no other board, under the
    source's project, which that profile's projects.db does not hold."""
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
    ticked = kb.connect(db_path=other_db)
    try:
        kb.dispatch_once(ticked, board="other", spawn_fn=lambda *a, **k: 1)
    finally:
        ticked.close()

    count = "SELECT COUNT(*) AS n FROM tasks WHERE idempotency_key LIKE 'review:%'"
    assert [_raw(db, count)[0]["n"] for db in (other_db, own_db, root / "kanban.db")] == [2, 0, 0]
    cards = _cards(kb, other_db)
    assert {c["idempotency_key"] for c in cards} == {f"review:{tid}:{head}:R15", f"review:{tid}:{head}:R12"}
    source = Path(_raw(other_db, "SELECT workspace_path FROM tasks WHERE id = ?", tid)[0]["workspace_path"])
    assert [(c["project_id"], c["workspace_path"]) for c in cards] == [
        ("delivery-project", str(source.parent / c["id"])) for c in cards]
    assert _state(kb, head, other_db) == "review_cards_created"
