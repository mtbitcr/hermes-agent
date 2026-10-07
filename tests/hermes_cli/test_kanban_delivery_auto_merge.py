"""Card 5: the delivery step arms GitHub auto-merge on exactly the published head H once every review card
the tier requires is done and its own review approves H, and closes the earlier pull request once when a
rework card continues a delivered head. Real boards and git; GitHub is the real transport with only its
exchange and App token replaced by a table, so every call passes the allowlist."""

from __future__ import annotations

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


@pytest.fixture
def armed(world, monkeypatch):
    """The review-card world, where GitHub also lists each pull request's reviews (gh["reviews"] by number),
    answers the auto-merge mutation with gh["arm"] and a close with gh["close"], and keeps each write's body.
    Pull request 41 is open at H, and each one in gh["pulls"] at the head given there."""
    kb, root, repo, gh = world
    from hermes_cli import kanban_delivery_github as transport

    table = transport._exchange
    gh.update(reviews={}, pulls={}, writes=[], arm=(200, ARMED), close=(200, {"number": 41, "state": "closed"}))

    def exchange(method, target, authorization, payload=None):
        path = urlsplit(target).path
        parts = path.split("/")
        pull = len(parts) >= 6 and parts[4] == "pulls" and parts[5].isdigit()
        if method in ("POST", "PATCH") or pull:
            gh["calls"].append((method, path))
        if method in ("POST", "PATCH"):
            gh["writes"].append((method, path, payload))
            return gh["arm"] if path == "/graphql" else gh["close"]
        if pull and path.endswith("/reviews"):
            return 200, gh["reviews"].get(int(parts[5]), [])
        if pull and len(parts) == 6:
            heads = {41: gh["pull_head"] or gh["head"], **gh["pulls"]}
            number = int(parts[5])
            if number not in heads:
                return 404, None
            return 200, {"number": number, "node_id": f"PR_node{number}", "state": "open",
                         "head": {"sha": heads[number]}, "user": {"login": "someone"}}
        return table(method, target, authorization, payload)

    monkeypatch.setattr(transport, "_exchange", exchange)
    return world


def _reviews(gh, *reviews, number=41):
    """The reviewer bot's reviews of a pull request, oldest first: (state, commit) each."""
    gh["reviews"][number] = [{"id": 700 + n, "state": state, "commit_id": commit, "body": "text",
                              "user": {"login": "raphael-reviewer[bot]"}} for n, (state, commit) in enumerate(reviews)]


def _done(db, *cards):
    """The review cards (all of the board's when none is named) finished, as the reviewer finishes them."""
    for card in cards or [c["id"] for c in _raw(db, "SELECT id FROM tasks WHERE idempotency_key LIKE 'review:%'")]:
        _raw(db, "UPDATE tasks SET status = 'done' WHERE id = ?", card)


def _arms(gh):
    return [body for method, path, body in gh["writes"] if path == "/graphql"]


def _published_as(db, gh, tid, head, number):
    """A raw ledger of ``tid`` at ``head`` as pull request ``number``, open at that head on GitHub."""
    _raw(db, "UPDATE kanban_deliveries SET pull_request_number = ?, pull_request_head = ?, pull_request_state = "
         "'open', pull_request_branch = ? WHERE source_task_id = ? AND source_head = ?",
         number, head, "delivery/" + tid, tid, head)
    gh["pulls"][number] = head


@pytest.mark.parametrize("case", ["open", "changes_on_github", "changes_on_card"])
def test_no_arm_while_a_required_card_is_open_or_returned_changes(armed, case):
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
    # Finished, or approved again by both lenses, H arms; a card that returned changes never counts.
    _done(kb.kanban_db_path())
    gh["reviews"][41] += [dict(review, id=800 + n, state="APPROVED") for n, review in enumerate(gh["reviews"][41])]
    _tick(kb)
    assert [body["variables"]["expectedHeadOid"] for body in _arms(gh)] == ([] if case == "changes_on_card" else [head])


@pytest.mark.parametrize("reviews", [
    [("APPROVED", "H")],
    [("APPROVED", OTHER), ("APPROVED", "H")],
    [("APPROVED", "H"), ("APPROVED", OTHER)],
    [("APPROVED", "H"), ("APPROVED", "H"), ("DISMISSED", "H")],
    [],
])
def test_a_one_lens_approval_on_a_tier_2_head_does_not_arm(armed, reviews):
    """Both lenses post from one account: each needs its own approval of H, so one approval, or two where
    one is of another commit or no longer the latest, arms nothing."""
    kb, root, repo, gh = armed
    tid, head = _ready(armed, tier=2)
    _tick(kb)
    _done(kb.kanban_db_path())
    _reviews(gh, *[(state, head if commit == "H" else commit) for state, commit in reviews])

    _tick(kb)

    assert _arms(gh) == [] and _state(kb, head) == "review_cards_created"
    _reviews(gh, *[(state, head if commit == "H" else commit) for state, commit in reviews], *[("APPROVED", head)] * 2)
    _tick(kb)
    assert [body["variables"]["expectedHeadOid"] for body in _arms(gh)] == [head]  # each lens's own approval of H


def test_the_allowlist_accepts_exactly_the_two_new_calls_and_refuses_every_other(monkeypatch):
    from hermes_cli import kanban_delivery_github as transport

    sent = []
    monkeypatch.setattr(transport, "_exchange", lambda method, target, authorization, payload=None: (
        sent.append((method, target, payload)) or (200, {})))
    monkeypatch.setattr(transport.GitHubTransport, "_installation_token", lambda self: "token")
    document = transport.AUTO_MERGE_MUTATION
    arm = {"query": document, "variables": {"pullRequestId": "PR_kwDOAbc", "expectedHeadOid": HEAD}}
    github = transport.GitHubTransport("publish", REPO)

    github.request("PATCH", f"/repos/{REPO}/pulls/41", body={"state": "closed"})
    github.request("POST", "/graphql", body=arm)

    assert sent == [("PATCH", f"/repos/{REPO}/pulls/41", {"state": "closed"}), ("POST", "/graphql", arm)]
    assert "mergeMethod: MERGE" in document and "expectedHeadOid: $expectedHeadOid" in document
    assert [e.template for e in transport.GRAPHQL_ALLOWLIST] == ["/graphql"]
    writes = {(e.method, e.template) for e in transport.REST_ALLOWLIST if e.method != "GET"}
    assert writes == {("POST", "/repos/{repo}/pulls"), ("PUT", "/repos/{repo}/pulls/{number}/merge"),
                      ("POST", "/repos/{repo}/pulls/{number}/reviews"), ("POST", "/repos/{repo}/actions/jobs/{id}/rerun"),
                      ("POST", "/app/installations/{installation}/access_tokens"), ("PATCH", "/repos/{repo}/pulls/{number}")}
    literal = document.replace("$expectedHeadOid", f'"{OTHER}"').replace(", $expectedHeadOid: GitObjectID!", "")
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
        ("GET", "/graphql", None),
        ("PATCH", "/graphql", arm),
        ("POST", "/graphql", None),
        ("POST", "/graphql", {"query": "query { viewer { login } }"}),
        ("POST", "/graphql", dict(arm, query=document.replace("MERGE", "SQUASH"))),
        ("POST", "/graphql", dict(arm, query=document.replace("MERGE", "REBASE"))),
        ("POST", "/graphql", dict(arm, query=document.replace("expectedHeadOid: $expectedHeadOid, ", ""))),
        ("POST", "/graphql", {"query": literal, "variables": {"pullRequestId": "PR_kwDOAbc"}}),
        ("POST", "/graphql", {"query": document, "variables": {"pullRequestId": "PR_kwDOAbc"}}),
        ("POST", "/graphql", {"query": document, "variables": {"pullRequestId": "PR_kwDOAbc", "expectedHeadOid": "main"}}),
        ("POST", "/graphql", {"query": document, "variables": dict(arm["variables"], mergeMethod="SQUASH")}),
        ("POST", "/graphql", dict(arm, operationName="arm")),
        ("POST", "/graphql", {"query": "mutation { mergePullRequest(input: {pullRequestId: \"PR_kwDOAbc\"}) { clientMutationId } }"}),
        ("POST", "/graphql", {"query": document + " ", "variables": arm["variables"]}),
        ("POST", f"/repos/{REPO}/graphql", arm),
        ("PUT", f"/repos/{REPO}/branches/main/protection", {}),
    ]
    for method, path, body in refused:
        with pytest.raises(transport.GitHubTransportError):
            github.request(method, path, body=body)
    assert len(sent) == 2
    for step in ("read_checks", "rerun_flaky", "handoff"):  # no step that cannot write pull requests can call them
        with pytest.raises(transport.GitHubTransportError):
            transport.GitHubTransport(step, REPO).request("POST", "/graphql", body=arm)
    assert len(sent) == 2


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
    assert gh["calls"][-3:] == [("GET", f"/repos/{REPO}/pulls/41"), ("GET", f"/repos/{REPO}/pulls/41/reviews"),
                                ("POST", "/graphql")]
    assert _state(kb, head) == "auto_merge_armed"
    events = _events(kb, tid)
    assert [(kind, p["head"], p["pull_request_number"]) for kind, p in events if "auto_merge" in kind] == [
        ("delivery_auto_merge_armed", head, 41)]
    calls = len(gh["calls"])

    _tick(kb)
    _tick(kb)

    assert (len(gh["calls"]), _events(kb, tid), _state(kb, head)) == (calls, events, "auto_merge_armed")


@pytest.mark.parametrize("answer", [
    (200, {"errors": [{"type": "UNPROCESSABLE", "message": "Pull request is in clean status"}]}),
    (200, {"data": {"enablePullRequestAutoMerge": None}, "errors": [{"message": "auto-merge is not allowed"}]}),
    (403, None), (422, None)])
def test_a_refused_arm_is_recorded_once_for_its_head(armed, answer):
    kb, root, repo, gh = armed
    tid, head = _ready(armed)
    _tick(kb)
    _done(kb.kanban_db_path())
    _reviews(gh, ("APPROVED", head), ("APPROVED", head))
    gh["arm"] = answer

    _tick(kb)
    _tick(kb)
    _tick(kb)

    assert len(_arms(gh)) == 1 and _state(kb, head) == "auto_merge_refused"
    refused = [(kind, p) for kind, p in _events(kb, tid) if "auto_merge" in kind]
    assert [(kind, p["head"], p["status"]) for kind, p in refused] == [("delivery_auto_merge_refused", head, answer[0])]
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


def test_a_rework_card_that_continues_a_delivered_head_closes_the_earlier_pull_request_once(armed):
    kb, root, repo, gh = armed
    db = kb.kanban_db_path()
    first, delivered = _ready(armed)
    rework, continued = _approved(kb, repo, "rework")
    other, unrelated = _approved(kb, repo, "unrelated")
    _raw(db, "UPDATE tasks SET base_commit = ? WHERE id = ?", delivered, rework)
    _published_as(db, gh, rework, continued, 42)
    _published_as(db, gh, other, unrelated, 43)

    _tick(kb)

    closes = [write for write in gh["writes"] if write[0] == "PATCH"]
    assert closes == [("PATCH", f"/repos/{REPO}/pulls/41", {"state": "closed"})]
    assert [(kind, p["pull_request_number"], p["replaced_by"]) for kind, p in _events(kb, first)
            if kind == "delivery_replaced"] == [("delivery_replaced", 41, rework)]
    assert _state(kb, delivered) == "replaced"
    assert (_state(kb, continued), _state(kb, unrelated)) == ("review_cards_created", "review_cards_created")

    _tick(kb)
    _raw(db, "UPDATE kanban_deliveries SET pull_request_state = 'returned_for_changes' WHERE source_head = ?",
         delivered)  # as a later approval of the earlier card marks it: its close is still recorded
    _tick(kb)

    assert [write for write in gh["writes"] if write[0] == "PATCH"] == closes
    assert len([kind for kind, _ in _events(kb, first) if kind == "delivery_replaced"]) == 1


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
