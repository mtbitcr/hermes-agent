"""Card 5: the delivery step arms GitHub auto-merge on exactly the published head H once every review card
the tier requires is done and approves H and the reviewer bot's latest review approves H, and closes the
earlier pull request once when a rework card continues a delivered head. Real boards and git; GitHub is the
real transport with only its exchange and App token replaced by a table, so every call passes the allowlist."""

from __future__ import annotations

import inspect
import json
from urllib.parse import urlsplit

import pytest

from tests.hermes_cli.test_kanban_delivery import (  # noqa: F401  (board is a fixture)
    IMPLEMENTER, _become, _git, _rework, _spawned_worker_env, _write_config, board,
)
from tests.hermes_cli.test_kanban_delivery_github import HEAD, REPO
from tests.hermes_cli.test_kanban_delivery_publish_step import _approved, _events, _tick
from tests.hermes_cli.test_kanban_delivery_review_cards import (  # noqa: F401  (world is a fixture)
    _cards, _published, _raw, _ready, _state, world,
)

ARMED = {"data": {"enablePullRequestAutoMerge": {"clientMutationId": None}}}
OTHER = "e" * 40
BOT = "raphael-reviewer-mtbitcr[bot]"
ELSEWHERE = "mtbitcr/raphael-workspace"  # the policy's other repository


@pytest.fixture
def armed(world, monkeypatch):
    """The review-card world, where GitHub also lists each pull request's reviews (gh["reviews"] by number),
    answers the auto-merge mutation with gh["arm"] and a close with gh["close"] ("timeout": no answer), keeps
    each write's body and runs gh["hooks"]["GET"] once during a reviews read, ["PATCH"] during a close.
    Pull request 41 of REPO is open at H, and each one in gh["pulls"] at the head given there."""
    kb, root, repo, gh = world
    from hermes_cli import kanban_delivery_github as transport

    table = transport._exchange
    gh.update(reviews={}, pulls={}, writes=[], hooks={}, arm=(200, ARMED), close=(200, {"number": 41, "state": "closed"}))

    def exchange(method, target, authorization, payload=None):
        path = urlsplit(target).path
        parts = path.split("/")
        pull = len(parts) >= 6 and parts[4] == "pulls" and parts[5].isdigit()
        if method in ("POST", "PATCH") or pull:
            gh["calls"].append((method, path))
        if method == "PATCH" or path.endswith("/reviews"):
            gh["hooks"].pop(method, lambda: None)()
        if method in ("POST", "PATCH"):
            gh["writes"].append((method, path, payload))
            answer = gh["arm"] if path == "/graphql" else gh["close"]
            if answer == "timeout":
                raise transport.GitHubTransportError("network_error")
            return answer
        if pull and path.endswith("/reviews"):
            return 200, gh["reviews"].get(int(parts[5]), [])
        if pull and len(parts) == 6:
            number, owner = int(parts[5]), "/".join(parts[2:4])
            heads = {41: gh["pull_head"] or gh["head"], **gh["pulls"]}
            if (number if owner == REPO else (owner, number)) not in heads:
                return 404, None
            return 200, {"number": number, "node_id": f"PR_node{number}", "state": "open",
                         "head": {"sha": heads[number if owner == REPO else (owner, number)]}}
        return table(method, target, authorization, payload)

    monkeypatch.setattr(transport, "_exchange", exchange)
    return world


def _reviews(gh, *reviews, number=41):
    """A pull request's reviews, oldest first: (state, commit), by the reviewer bot unless a third item
    names another account."""
    gh["reviews"][number] = [{"id": 700 + n, "state": state, "commit_id": commit, "body": "text",
                              "user": {"login": (who or [BOT])[0], "type": "Bot"}}
                             for n, (state, commit, *who) in enumerate(reviews)]


def _done(db, *cards):
    """The review cards (all of the board's when none is named) finished, as the reviewer finishes them."""
    for card in cards or [c["id"] for c in _raw(db, "SELECT id FROM tasks WHERE idempotency_key LIKE 'review:%'")]:
        _raw(db, "UPDATE tasks SET status = 'done' WHERE id = ?", card)


def _arms(gh):
    return [body for method, path, body in gh["writes"] if path == "/graphql"]


def _published_as(db, gh, tid, head, number, repository=REPO):
    """A raw ledger of ``tid`` at ``head`` as pull request ``number`` of ``repository``, with the publish
    step's record of it, open at that head on GitHub."""
    _raw(db, "UPDATE kanban_deliveries SET pull_request_number = ?, pull_request_head = ?, pull_request_state = "
         "'open', pull_request_branch = ? WHERE source_task_id = ? AND source_head = ?",
         number, head, "delivery/" + tid, tid, head)
    _raw(db, "INSERT INTO task_events (task_id, kind, payload, created_at) VALUES (?, 'delivery_published', ?, 0)",
         tid, json.dumps({"repository": repository, "branch": "delivery/" + tid, "pull_request_number": number,
                          "head": head, "state": "created"}))
    gh["pulls"][number if repository == REPO else (repository, number)] = head


@pytest.mark.parametrize("case", ["open", "changes_on_github", "changes_on_card"])
def test_no_arm_while_a_required_card_is_open_or_returned_changes(armed, case):
    """One lens's card done, even with two approvals of H, arms nothing: each lens is its own card."""
    kb, root, repo, gh = armed
    tid, head = _ready(armed)
    _tick(kb)
    cards = {c["responsibility"]: c["id"] for c in _cards(kb)}
    _done(kb.kanban_db_path(), *([cards["R15"]] if case == "open" else cards.values()))
    _reviews(gh, ("APPROVED", head), ("CHANGES_REQUESTED", head) if case == "changes_on_github" else ("APPROVED", head))
    if case == "changes_on_card":
        conn = kb.connect()
        try:
            with kb.write_txn(conn):
                kb._append_event(conn, cards["R12"], "changes_requested", {"reason": "rework"})
        finally:
            conn.close()

    _tick(kb)
    _tick(kb)

    assert _arms(gh) == [] and _state(kb, head) == "review_cards_created"
    assert not [kind for kind, _ in _events(kb, tid) if "auto_merge" in kind]
    # Finished, or approved again by the bot, H arms; a card that returned changes never counts.
    _done(kb.kanban_db_path())
    gh["reviews"][41] += [dict(review, id=800 + n, state="APPROVED") for n, review in enumerate(gh["reviews"][41])]
    _tick(kb)
    assert [body["variables"]["expectedHeadOid"] for body in _arms(gh)] == ([] if case == "changes_on_card" else [head])


@pytest.mark.parametrize("reviews", [
    [("APPROVED", "H", "someone"), ("APPROVED", "H", "raphael-reviewer[bot]")],
    [("CHANGES_REQUESTED", "H"), ("APPROVED", "H", "someone"), ("APPROVED", "H", "another")],
    [("APPROVED", "H"), ("APPROVED", OTHER)],
    [("APPROVED", "H"), ("COMMENTED", "H")],
    [],
])
def test_only_the_reviewer_bots_latest_review_approving_h_confirms_the_done_cards(armed, reviews):
    """The done cards are the approval; GitHub is read only to confirm that the reviewer bot's latest
    review of the pull request approves H. Any other account's review never counts."""
    kb, root, repo, gh = armed
    tid, head = _ready(armed, tier=2)
    _tick(kb)
    _done(kb.kanban_db_path())
    given = [(state, head if commit == "H" else commit, *who) for state, commit, *who in reviews]
    _reviews(gh, *given)

    _tick(kb)

    assert _arms(gh) == [] and _state(kb, head) == "review_cards_created"
    _reviews(gh, *given, ("APPROVED", head))
    _tick(kb)
    assert [body["variables"]["expectedHeadOid"] for body in _arms(gh)] == [head]


def test_the_allowlist_holds_exactly_the_three_new_entries_and_refuses_every_other(monkeypatch):
    from hermes_cli import kanban_delivery_github as transport

    sent = []
    answers = {f"/repos/{REPO}/pulls/41": {"number": 41, "node_id": "PR_kwDOAbc", "state": "open", "head": {"sha": HEAD}},
               f"/repos/{REPO}/pulls/41/reviews": [{"id": 1, "state": "APPROVED", "commit_id": HEAD, "body": "text",
                                                     "user": {"login": BOT, "avatar_url": "https://x"}}]}
    monkeypatch.setattr(transport, "_exchange", lambda method, target, authorization, payload=None: (
        sent.append((method, target, payload)) or (200, answers.get(urlsplit(target).path, {}))))
    monkeypatch.setattr(transport.GitHubTransport, "_installation_token", lambda self: setattr(self, "_token", "ghs_0") or "ghs_0")
    document = transport.AUTO_MERGE_MUTATION
    arm = {"query": document, "variables": {"pullRequestId": "PR_kwDOAbc", "expectedHeadOid": HEAD}}
    github = transport.GitHubTransport("publish", REPO)

    assert github.request("GET", f"/repos/{REPO}/pulls/41/reviews", query={"per_page": 100})["data"] == [
        {"id": 1, "state": "APPROVED", "commit_id": HEAD, "user": {"login": BOT}}]
    github.request("PATCH", f"/repos/{REPO}/pulls/41", body={"state": "closed"})
    github.arm_auto_merge(41, HEAD)

    assert sent[1:] == [("PATCH", f"/repos/{REPO}/pulls/41", {"state": "closed"}),
                        ("GET", f"/repos/{REPO}/pulls/41", None), ("POST", "/graphql", arm)]
    assert "mergeMethod: MERGE" in document and "expectedHeadOid: $expectedHeadOid" in document
    assert {(e.method, e.template) for e in transport.REST_ALLOWLIST + transport.GRAPHQL_ALLOWLIST} == {
        ("GET", "/repos/{repo}/pulls"), ("POST", "/repos/{repo}/pulls"), ("GET", "/repos/{repo}/pulls/{number}"),
        ("PUT", "/repos/{repo}/pulls/{number}/merge"), ("POST", "/repos/{repo}/pulls/{number}/reviews"),
        ("GET", "/repos/{repo}/git/ref/heads/{branch}"), ("GET", "/repos/{repo}/commits/{sha}/check-runs"),
        ("GET", "/repos/{repo}/actions/runs"), ("GET", "/repos/{repo}/actions/runs/{id}/jobs"),
        ("GET", "/repos/{repo}/actions/jobs/{id}/logs"), ("POST", "/repos/{repo}/actions/jobs/{id}/rerun"),
        ("POST", "/app/installations/{installation}/access_tokens"),  # the twelve before this card, and its three:
        ("GET", "/repos/{repo}/pulls/{number}/reviews"), ("PATCH", "/repos/{repo}/pulls/{number}"), ("POST", "/graphql")}
    refused = [
        ("PATCH", f"/repos/{REPO}/pulls/41", {"state": "closed", "title": "replaced"}),
        ("PATCH", f"/repos/{REPO}/pulls/41", {"state": "open"}),
        ("PATCH", f"/repos/{REPO}/pulls/41", {"state": "closed", "base": "main"}),
        ("PATCH", f"/repos/{REPO}/pulls/41", None),
        ("PATCH", f"/repos/other/repo/pulls/41", {"state": "closed"}),
        ("PATCH", f"/repos/{REPO}/issues/41", {"state": "closed"}),
        ("PATCH", f"/repos/{REPO}", {"state": "closed"}),
        ("POST", f"/repos/{REPO}/pulls/41", {"state": "closed"}),
        ("DELETE", f"/repos/{REPO}/pulls/41", None),
        ("GET", f"/repos/{REPO}/pulls/41/comments", None),
        ("GET", f"/repos/{REPO}/pulls/41/reviews/1", None),
        ("GET", f"/repos/other/repo/pulls/41/reviews", None),
        ("GET", "/graphql", None),
        ("PATCH", "/graphql", arm),
        ("POST", "/graphql", arm),  # the arm is the transport's own call: no caller sends a document
        ("POST", "/graphql", None),
        ("POST", "/graphql", {"query": "query { viewer { login } }"}),
        ("POST", "/graphql", dict(arm, query=document.replace("MERGE", "SQUASH"))),
        ("POST", "/graphql", {"query": document, "variables": {"pullRequestId": "PR_kwDOAbc"}}),
        ("POST", "/graphql", {"query": document, "variables": dict(arm["variables"], expectedHeadOid=OTHER)}),
        ("POST", "/graphql", {"query": "mutation { mergePullRequest(input: {pullRequestId: \"PR_kwDOAbc\"}) { clientMutationId } }"}),
        ("POST", f"/repos/{REPO}/graphql", arm),
        ("PUT", f"/repos/{REPO}/branches/main/protection", {}),
    ]
    for method, path, body in refused:
        with pytest.raises(transport.GitHubTransportError):
            github.request(method, path, body=body)
    assert len(sent) == 4
    # The caller names the pull request and H only; a head it does not hold, or no head, sends no mutation.
    assert list(inspect.signature(transport.GitHubTransport.arm_auto_merge).parameters) == ["self", "number", "head"]
    for number, head in ((41, OTHER), (41, None), (41, "main"), ("41", HEAD), (True, HEAD)):
        with pytest.raises(transport.GitHubTransportError):
            github.arm_auto_merge(number, head)
    for step in ("read_checks", "rerun_flaky", "handoff"):  # no step that cannot write pull requests can arm
        with pytest.raises(transport.GitHubTransportError):
            transport.GitHubTransport(step, REPO).arm_auto_merge(41, HEAD)
    assert sent[4:] == [("GET", f"/repos/{REPO}/pulls/41", None)]


@pytest.mark.parametrize("tier", [0, 2])
def test_a_publish_arms_once_on_the_exact_head_and_a_second_pass_makes_no_call(armed, tier):
    from hermes_cli import kanban_delivery_github as transport

    kb, root, repo, gh = armed
    tid, head = _ready(armed, tier=tier)
    _tick(kb)
    assert _arms(gh) == []  # the cards are open: nothing is read or armed
    _done(kb.kanban_db_path())
    _reviews(gh, ("CHANGES_REQUESTED", OTHER), ("COMMENTED", head), *[("APPROVED", head)] * len(_cards(kb)))

    _tick(kb)

    assert _arms(gh) == [{"query": transport.AUTO_MERGE_MUTATION,
                          "variables": {"pullRequestId": "PR_node41", "expectedHeadOid": head}}]
    assert gh["calls"][-3:] == [("GET", f"/repos/{REPO}/pulls/41/reviews"), ("GET", f"/repos/{REPO}/pulls/41"),
                                ("POST", "/graphql")]
    assert _state(kb, head) == "auto_merge_armed"
    events = _events(kb, tid)
    assert [(kind, p["head"], p["pull_request_number"]) for kind, p in events if "auto_merge" in kind] == [
        ("delivery_auto_merge_armed", head, 41)]
    calls = len(gh["calls"])

    _tick(kb)
    _tick(kb)

    assert (len(gh["calls"]), _events(kb, tid), _state(kb, head)) == (calls, events, "auto_merge_armed")


@pytest.mark.parametrize("answer, state", [
    ((200, {"errors": [{"type": "UNPROCESSABLE", "message": "Pull request is in clean status"}]}), "refused"),
    ((200, {"data": {"enablePullRequestAutoMerge": None}, "errors": [{"message": "auto-merge is not allowed"}]}),
     "refused"),
    ((403, None), "refused"), ((422, None), "refused"), ((429, None), "refused"), ((500, None), "refused"),
    ((503, None), "refused"), ("timeout", "unknown")])
def test_a_refused_arm_is_recorded_once_for_its_head(armed, answer, state):
    kb, root, repo, gh = armed
    tid, head = _ready(armed)
    _tick(kb)
    _done(kb.kanban_db_path())
    _reviews(gh, ("APPROVED", head), ("APPROVED", head))
    gh["arm"] = answer

    _tick(kb)
    _tick(kb)
    _tick(kb)

    assert len(_arms(gh)) == 1 and _state(kb, head) == f"auto_merge_{state}"
    refused = [(kind, p) for kind, p in _events(kb, tid) if "auto_merge" in kind]
    assert [(kind, p["head"], p.get("status")) for kind, p in refused] == [
        (f"delivery_auto_merge_{state}", head, None if answer == "timeout" else answer[0])]
    assert "message" not in str(refused) and "clean status" not in str(refused)


def test_a_new_head_starts_fresh_after_a_refused_arm(armed):
    kb, root, repo, gh = armed
    tid, first = _ready(armed)
    _tick(kb)
    _done(kb.kanban_db_path())
    _reviews(gh, ("APPROVED", first), ("APPROVED", first))
    gh["arm"] = (403, None)
    _tick(kb)

    second = _rework(kb, repo, tid)
    _published(kb, repo, gh, tid, second)
    _tick(kb)
    _done(kb.kanban_db_path())
    _reviews(gh, ("APPROVED", first), ("APPROVED", first), ("APPROVED", second), ("APPROVED", second))
    gh["arm"] = (200, ARMED)
    _tick(kb)
    _tick(kb)

    assert [body["variables"]["expectedHeadOid"] for body in _arms(gh)] == [first, second]
    assert [(kind, p["head"]) for kind, p in _events(kb, tid) if "auto_merge" in kind] == [
        ("delivery_auto_merge_refused", first), ("delivery_auto_merge_armed", second)]
    assert _state(kb, second) == "auto_merge_armed"


@pytest.mark.parametrize("change",["card_reopened", "changes_requested", "source_head", "tier_raised", "row_moved",
                                    "overlap"])
def test_evidence_that_changes_while_github_is_read_sends_no_arm(armed, change):
    """The pass that sends the arm first reads the cards, the source card and the row again and records the
    attempt for H, in one write transaction: a change sends nothing, an overlapping pass no second arm."""
    kb, root, repo, gh = armed
    db = kb.kanban_db_path()
    tid, head = _ready(armed, tier=0)
    _tick(kb)
    _done(db)
    _reviews(gh, ("APPROVED", head))
    card = _cards(kb)[0]["id"]
    gh["hooks"]["GET"] = {
        "card_reopened": lambda: _raw(db, "UPDATE tasks SET status = 'ready' WHERE id = ?", card),
        "changes_requested": lambda: _raw(db, "INSERT INTO task_events (task_id, kind, created_at) "
                                              "VALUES (?, 'changes_requested', 0)", card),
        "source_head": lambda: _raw(db, "UPDATE tasks SET head_commit = ? WHERE id = ?", OTHER, tid),
        "tier_raised": lambda: _raw(db, "UPDATE tasks SET risk_tier = 2 WHERE id = ?", tid),
        "row_moved": lambda: _raw(db, "UPDATE kanban_deliveries SET pull_request_number = 42 WHERE source_head = ?",
                                  head),
        "overlap": lambda: _tick(kb)}[change]

    _tick(kb)
    _tick(kb)

    assert len(_arms(gh)) == (change == "overlap")
    assert len([kind for kind, _ in _events(kb, tid) if "auto_merge" in kind]) == (change == "overlap")


def test_a_new_head_is_never_pushed_to_an_armed_pull_request(armed):
    kb, root, repo, gh = armed
    tid, head = _ready(armed, tier=0)
    _tick(kb)
    _done(kb.kanban_db_path())
    _reviews(gh, ("APPROVED", head))
    _tick(kb)
    calls = list(gh["calls"])

    second = _rework(kb, repo, tid)
    _tick(kb)
    _tick(kb)

    assert gh["calls"] == calls and _state(kb, head) == "auto_merge_armed"
    assert [(p["code"], p["head"]) for kind, p in _events(kb, tid) if kind == "delivery_refused"] == [
        ("auto_merge_armed", second)]


@pytest.mark.parametrize("case", ["same", "origin_changed", "base_elsewhere", "number_elsewhere"])
def test_a_rework_card_that_continues_a_delivered_head_closes_the_earlier_pull_request_once(armed, case):
    """A pull request is its repository and number as the publish step recorded them, never the git origin:
    a continuation published in another repository replaces nothing, and its numbers hold back no close."""
    kb, root, repo, gh = armed
    db = kb.kanban_db_path()
    first, delivered = _ready(armed)
    rework, continued = _approved(kb, repo, "rework")
    other, unrelated = _approved(kb, repo, "unrelated")
    _raw(db, "UPDATE tasks SET base_commit = ? WHERE id = ?", delivered, rework)
    _published_as(db, gh, rework, continued, 42, ELSEWHERE if case == "base_elsewhere" else REPO)
    _published_as(db, gh, other, unrelated, *((41, ELSEWHERE) if case == "number_elsewhere" else (43, REPO)))
    if case == "origin_changed":
        _git(repo, "remote", "set-url", "origin", f"https://github.com/{ELSEWHERE}.git")

    _tick(kb)

    closes = [write for write in gh["writes"] if write[0] == "PATCH"]
    assert closes == ([] if case == "base_elsewhere" else [("PATCH", f"/repos/{REPO}/pulls/41", {"state": "closed"})])
    assert [(kind, p["pull_request_number"], p["replaced_by"]) for kind, p in _events(kb, first)
            if kind == "delivery_replaced"] == ([] if case == "base_elsewhere" else [("delivery_replaced", 41, rework)])
    assert (_state(kb, delivered) == "replaced") is (case != "base_elsewhere")

    _tick(kb)
    _raw(db, "UPDATE kanban_deliveries SET pull_request_state = 'returned_for_changes' WHERE source_head = ?",
         delivered)  # as a later approval of the earlier card marks it: its close is still recorded
    _tick(kb)

    assert [write for write in gh["writes"] if write[0] == "PATCH"] == closes


@pytest.mark.parametrize("answer", [(403, None), (200, {"number": 41, "state": "open"}), (200, None), (429, None),
                                    (200, {"number": 42, "state": "closed"}), (503, None), "timeout", "overlap"])
def test_a_close_is_done_only_when_the_answer_shows_that_pull_request_closed(armed, answer):
    kb, root, repo, gh = armed
    db = kb.kanban_db_path()
    first, delivered = _ready(armed)
    rework, continued = _approved(kb, repo, "rework")
    _raw(db, "UPDATE tasks SET base_commit = ? WHERE id = ?", delivered, rework)
    _published_as(db, gh, rework, continued, 42)
    if answer == "overlap":
        gh["hooks"]["PATCH"] = lambda: _tick(kb)
    else:
        gh["close"] = answer

    _tick(kb)
    _tick(kb)

    assert [write[1] for write in gh["writes"] if write[0] == "PATCH"] == [f"/repos/{REPO}/pulls/41"]
    outcome = [(kind, p.get("status")) for kind, p in _events(kb, first) if kind.startswith(("delivery_replaced",
                                                                                            "delivery_close"))]
    assert outcome[-1:] == ([("delivery_replaced", 200)] if answer == "overlap" else [
        ("delivery_close_refused", None if answer == "timeout" else answer[0])])
    assert _state(kb, delivered) == ("replaced" if answer == "overlap" else "close_refused")


def test_the_review_card_title_carries_the_source_title_and_no_raw_identifier(armed):
    kb, root, repo, gh = armed
    tid, head = _ready(armed)

    _tick(kb)

    cards = _cards(kb)
    assert sorted(c["title"] for c in cards) == ["Review (correctness) of build feature",
                                                 "Review (security) of build feature"]
    for card in cards:
        assert not [raw for raw in (tid, head, head[:7], "41", REPO, REPO.split("/")[1]) if raw in card["title"]]
        for reference in (f"Source card: {tid}", f"Head: {head}", f"Base commit: {_git(repo, 'rev-parse', 'main')}"):
            assert reference in card["body"]


def test_the_arm_reaches_each_ticked_board_under_the_worker_environment(armed, monkeypatch):
    """The dispatcher's worker environment for proj-a (its board pinned, its card set, a profile home under
    the root) ticks the board other and the root board: each arms its own head, and proj-a nothing."""
    kb, root, repo, gh = armed
    kb.create_board("proj-a")
    kb.create_board("other")
    own_db = root / "kanban" / "boards" / "proj-a" / "kanban.db"
    other_db = root / "kanban" / "boards" / "other" / "kanban.db"
    root_db = root / "kanban.db"
    profile_home = root / "profiles" / IMPLEMENTER
    profile_home.mkdir(parents=True)
    _write_config(profile_home, enabled=True)
    tid, head = _approved(kb, repo, "beta", board="other")
    _published(kb, repo, gh, tid, head, board="other")
    _raw(other_db, "UPDATE tasks SET risk_tier = 2 WHERE id = ?", tid)
    home, home_head = _approved(kb, repo, "gamma")
    _raw(root_db, "UPDATE tasks SET risk_tier = 0 WHERE id = ?", home)
    _published_as(root_db, gh, home, home_head, 44)
    for db, board_name in ((other_db, "other"), (root_db, kb.DEFAULT_BOARD)):
        ticked = kb.connect(db_path=db)
        try:
            kb.dispatch_once(ticked, board=board_name, spawn_fn=lambda *a, **k: 1)
        finally:
            ticked.close()
        _done(db)
    _reviews(gh, ("APPROVED", head), ("APPROVED", head))
    _reviews(gh, ("APPROVED", home_head), number=44)
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
    for db, board_name in ((other_db, "other"), (root_db, kb.DEFAULT_BOARD)):
        ticked = kb.connect(db_path=db)
        try:
            kb.dispatch_once(ticked, board=board_name, spawn_fn=lambda *a, **k: 1)
        finally:
            ticked.close()

    count = "SELECT COUNT(*) AS n FROM kanban_deliveries WHERE pull_request_state = 'auto_merge_armed'"
    assert [_raw(db, count)[0]["n"] for db in (other_db, own_db, root_db)] == [1, 0, 1]
    assert sorted((body["variables"]["pullRequestId"], body["variables"]["expectedHeadOid"]) for body in _arms(gh)) == [
        ("PR_node41", head), ("PR_node44", home_head)]
    assert (_state(kb, head, other_db), _state(kb, home_head, root_db)) == ("auto_merge_armed", "auto_merge_armed")
