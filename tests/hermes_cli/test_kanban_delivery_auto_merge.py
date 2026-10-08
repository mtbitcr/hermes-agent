"""Card 5: the delivery step merges the pull request at exactly the published head H once every review card
the tier requires is done and approves H and the reviewer bot's latest review approves H, and closes the
earlier pull request once when a rework card continues a delivered head. Real boards and git; GitHub is the
real transport with only its exchange and App token replaced by a table, so every call passes the allowlist."""

from __future__ import annotations

import http.client
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlsplit

import pytest

from tests.hermes_cli.test_kanban_delivery import (  # noqa: F401  (board is a fixture)
    IMPLEMENTER, _become, _git, _rework, _spawned_worker_env, _write_config, board,
)
from tests.hermes_cli.test_kanban_delivery_github import HEAD, REPO
from tests.hermes_cli.test_kanban_delivery_publish_step import _approved, _events, _tick
from tests.hermes_cli.test_kanban_delivery_review_cards import (  # noqa: F401  (world is a fixture)
    _cards, _published, _raw, _ready, _state, world,
)

ARMED = {"sha": "d" * 40, "merged": True, "message": "Pull Request successfully merged"}  # its merge commit
OTHER = "e" * 40
BOT = "raphael-reviewer-mtbitcr[bot]"
ELSEWHERE = "mtbitcr/raphael-workspace"  # the policy's other repository


@pytest.fixture
def armed(world, monkeypatch):
    """The review-card world, where GitHub also lists each pull request's reviews (gh["reviews"] by number),
    answers a merge with gh["arm"] and a close with gh["close"] ("timeout": no answer), keeps
    each write's body and runs gh["hooks"]["GET"] once during a reviews read, ["PULL"] during a pull request
    read, ["PUT"] during a merge and ["PATCH"] during a close. Pull request 41 of REPO is open at H, and each
    one in gh["pulls"] at the head given there, unless gh["ended"] reports it closed (merged or not). As GitHub
    does, a reviews page names a next page exactly when more reviews follow it."""
    kb, root, repo, gh = world
    from hermes_cli import kanban_delivery_github as transport

    table = transport._exchange
    gh.update(reviews={}, pulls={}, writes=[], hooks={}, ended={}, arm=(200, ARMED),
              close=(200, {"number": 41, "state": "closed"}))

    def exchange(method, target, authorization, payload=None):
        path = urlsplit(target).path
        parts = path.split("/")
        pull = len(parts) >= 6 and parts[4] == "pulls" and parts[5].isdigit()
        if method in ("PUT", "PATCH") or pull:
            gh["calls"].append((method, path))
        gh["hooks"].pop("GET" if path.endswith("/reviews") else method if method != "GET" else "PULL" if pull else "",
                        lambda: None)()
        if method in ("PUT", "PATCH"):
            gh["writes"].append((method, path, payload))
            answer = gh["arm"] if method == "PUT" else gh["close"]
            if answer == "timeout":
                raise transport.GitHubTransportError("network_error")
            return answer
        if pull and path.endswith("/reviews"):
            query = parse_qs(urlsplit(target).query)  # listed 100 a page, each read's query kept
            gh["queries"].append((path, query))
            page, listed = int(query.get("page", ["1"])[0]), gh["reviews"].get(int(parts[5]), [])
            return 200, listed[(page - 1) * 100:page * 100], len(listed) > page * 100
        if pull and len(parts) == 6:
            number, owner = int(parts[5]), "/".join(parts[2:4])
            heads = {41: gh["pull_head"] or gh["head"], **gh["pulls"]}
            if (number if owner == REPO else (owner, number)) not in heads:
                return 404, None
            return 200, {"number": number, "node_id": f"PR_node{number}", "merged": bool(gh["ended"].get(number)),
                         "state": "closed" if number in gh["ended"] else "open",
                         "head": {"sha": heads[number if owner == REPO else (owner, number)]},
                         **({"base": {"ref": "main"}, "merge_commit_sha": gh["ended"][number]}  # a sha: merged into
                            if isinstance(gh["ended"].get(number), str) else {})}  # main as that merge commit
        return table(method, target, authorization, payload)

    monkeypatch.setattr(transport, "_exchange", exchange)
    return world


def _reviews(gh, *reviews, number=41):
    """A pull request's reviews, oldest first: (state, commit[, card[, account]]), by the reviewer bot unless an
    account is named; a review of a card starts its body with the line naming that card."""
    gh["reviews"][number] = [{"id": 700 + n, "state": state, "commit_id": commit, "user": {"login": (
        who or [BOT])[0], "type": "Bot"}, "body": f"Review card {card[0]} \r\nlooks right" if card and card[0] else "text"}
                             for n, (state, commit, *card) in enumerate(reviews) for who in [card[1:]]]


def _lenses(kb, head, db=None):
    """The reviewer bot's approval of H for each of H's review cards, each starting with its card's line."""
    return [("APPROVED", head, card["id"]) for card in _cards(kb, db) if f":{head}:" in card["idempotency_key"]]


def _done(db, *cards):
    """The review cards (all of the board's when none is named) finished, as the reviewer finishes them."""
    for card in cards or [c["id"] for c in _raw(db, "SELECT id FROM tasks WHERE idempotency_key LIKE 'review:%'")]:
        _raw(db, "UPDATE tasks SET status = 'done' WHERE id = ?", card)


def _arms(gh):
    return [body for method, path, body in gh["writes"] if method == "PUT"]


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


def _continues(db, rework, head, source=None):
    """``rework`` records ``head`` as its base, and ``source``, when named, is its task_links parent, as the
    source of a linked rework card is."""
    _raw(db, "UPDATE tasks SET base_commit = ? WHERE id = ?", head, rework)
    if source:
        _raw(db, "INSERT INTO task_links (parent_id, child_id) VALUES (?, ?)", source, rework)


@pytest.mark.parametrize("case", ["open", "changes_on_github", "changes_on_card"])
def test_no_arm_while_a_required_card_is_open_or_returned_changes(armed, case):
    """One lens's card done, even with two approvals of H, arms nothing: each lens is its own card."""
    kb, root, repo, gh = armed
    tid, head = _ready(armed)
    _tick(kb)
    cards = {c["responsibility"]: c["id"] for c in _cards(kb)}
    _done(kb.kanban_db_path(), *([cards["R15"]] if case == "open" else cards.values()))
    _reviews(gh, ("APPROVED", head, cards["R15"]),
             ("CHANGES_REQUESTED" if case == "changes_on_github" else "APPROVED", head, cards["R12"]))
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
    assert [body["sha"] for body in _arms(gh)] == ([] if case == "changes_on_card" else [head])


@pytest.mark.parametrize("reviews, arms", [
    ([("APPROVED", "H", "R15"), ("CHANGES_REQUESTED", "H", "R12"), ("APPROVED", "H", None)], 0),
    ([("APPROVED", "H", "R15"), ("APPROVED", OTHER, "R12"), ("APPROVED", "H", None)], 0),
    ([("APPROVED", "H", "R15"), ("APPROVED", "H", "R15")], 0),  # security's card done, with no review of its own
    ([("APPROVED", "H", "R12"), ("APPROVED", "H", "R15"), ("COMMENTED", "H", "R12"), ("APPROVED", "H", None)], 0),
    ([("APPROVED", "H", "R15"), ("APPROVED", "H", "R12", "raphael-reviewer[bot]"), ("APPROVED", "H", "R12", "x")], 0),
    ([("APPROVED", "H", "R15"), ("APPROVED", "H", "R122")], 0),  # whole lines: card t_1 is not card t_12
    ([("APPROVED", "H", "R15"), ("APPROVED", "H", "R12"), ("COMMENTED", "H", None)], 0),  # the bot's latest
    ([("CHANGES_REQUESTED", "H")], 0),
    ([("APPROVED", "H", "R15"), ("APPROVED", "H", "R12"), ("CHANGES_REQUESTED", "H", "R12", "someone")], 1),
])
def test_each_lens_is_approved_only_by_the_bots_latest_review_of_its_own_card_on_h(armed, reviews, arms):
    """Owner rule 1 of round 2: a lens approves H only when its card is done and the reviewer bot's latest
    review whose body starts with that card's line is APPROVED on H; the bot's latest review must also be
    APPROVED on H. One lens never stands in for another, and any other account is ignored."""
    kb, root, repo, gh = armed
    tid, head = _ready(armed, tier=2)
    _tick(kb)
    _done(kb.kanban_db_path())
    cards = {c["responsibility"]: c["id"] for c in _cards(kb)}
    given = [(state, head if commit == "H" else commit, card and cards[card[:3]] + card[3:], *who)
             for state, commit, card, *who in [review + (None,) * (3 - len(review)) for review in reviews]]
    _reviews(gh, *given)

    _tick(kb)

    assert len(_arms(gh)) == arms and _state(kb, head) == ("auto_merge_armed" if arms else "review_cards_created")
    _reviews(gh, *given, *_lenses(kb, head))
    _tick(kb)
    assert [body["sha"] for body in _arms(gh)] == [head]


def test_the_allowlist_holds_exactly_the_three_new_entries_and_refuses_every_other(monkeypatch):
    from hermes_cli import kanban_delivery_github as transport

    sent = []
    answers = {f"/repos/{REPO}/pulls/41": {"number": 41, "node_id": "PR_kwDOAbc", "state": "open", "head": {"sha": HEAD}},
               f"/repos/{REPO}/pulls/41/reviews": [{"id": 1, "state": "APPROVED", "commit_id": HEAD, "body": "text",
                                                     "user": {"login": BOT, "avatar_url": "https://x"}}]}
    monkeypatch.setattr(transport, "_exchange", lambda method, target, authorization, payload=None: (
        sent.append((method, target, payload)) or (200, answers.get(urlsplit(target).path, {}))))
    monkeypatch.setattr(transport.GitHubTransport, "_installation_token", lambda self: setattr(self, "_token", "ghs_0") or "ghs_0")
    merge = {"sha": HEAD, "merge_method": "merge"}
    github = transport.GitHubTransport("publish", REPO)

    assert github.request("GET", f"/repos/{REPO}/pulls/41/reviews", query={"per_page": 100})["data"] == [
        {"state": "APPROVED", "commit_id": HEAD, "user": {"login": BOT}, "body": "text"}]
    github.request("PATCH", f"/repos/{REPO}/pulls/41", body={"state": "closed"})
    github.request("GET", f"/repos/{REPO}/pulls/41")
    github.request("PUT", f"/repos/{REPO}/pulls/41/merge", body=merge)

    assert sent[1:] == [("PATCH", f"/repos/{REPO}/pulls/41", {"state": "closed"}),
                        ("GET", f"/repos/{REPO}/pulls/41", None), ("PUT", f"/repos/{REPO}/pulls/41/merge", merge)]
    assert not [name for name in ("GRAPHQL_ALLOWLIST", "AUTO_MERGE_MUTATION") if hasattr(transport, name)]
    assert {(e.method, e.template) for e in transport.REST_ALLOWLIST} == {
        ("GET", "/repos/{repo}/pulls"), ("POST", "/repos/{repo}/pulls"), ("GET", "/repos/{repo}/pulls/{number}"),
        ("GET", "/repos/{repo}/git/ref/heads/{branch}"), ("GET", "/repos/{repo}/commits/{sha}/check-runs"),
        ("GET", "/repos/{repo}/actions/runs"), ("GET", "/repos/{repo}/actions/runs/{id}/jobs"),
        ("GET", "/repos/{repo}/actions/jobs/{id}/logs"), ("POST", "/repos/{repo}/actions/jobs/{id}/rerun"),
        ("POST", "/app/installations/{installation}/access_tokens"),  # the twelve before card 5 less the two
        # unused ones the owner removed on 2026-10-04, card 5's three, and the red check's annotations read:
        ("GET", "/repos/{repo}/pulls/{number}/reviews"), ("PATCH", "/repos/{repo}/pulls/{number}"),
        ("PUT", "/repos/{repo}/pulls/{number}/merge"), ("GET", "/repos/{repo}/check-runs/{id}/annotations")}
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
        ("POST", "/graphql", None),
        ("POST", "/graphql", {"query": "query { viewer { login } }"}),
        ("POST", "/graphql", {"query": "mutation { mergePullRequest(input: {pullRequestId: \"PR_kwDOAbc\"}) { clientMutationId } }"}),
        ("POST", f"/repos/{REPO}/pulls/41/merge", merge),
        ("PUT", f"/repos/other/repo/pulls/41/merge", merge),
        ("PUT", f"/repos/{REPO}/branches/main/protection", {}),
        ("PUT", f"/repos/{REPO}/pulls/41/merge", {"sha": HEAD}),  # the merge takes its one exact body only
        ("POST", f"/repos/{REPO}/pulls/41/reviews", {"event": "APPROVE"}),  # removed: nothing posts a review
    ]
    for method, path, body in refused:
        with pytest.raises(transport.GitHubTransportError):
            github.request(method, path, body=body)
    assert len(sent) == 4
    for step in ("read_checks", "rerun_flaky", "handoff"):  # no step that cannot write contents can merge
        with pytest.raises(transport.GitHubTransportError):
            transport.GitHubTransport(step, REPO).request("PUT", f"/repos/{REPO}/pulls/41/merge", body=merge)
    assert len(sent) == 4


@pytest.mark.parametrize("tier", [0, 2])
def test_a_publish_arms_once_on_the_exact_head_and_a_second_pass_makes_no_call(armed, tier):
    kb, root, repo, gh = armed
    tid, head = _ready(armed, tier=tier)
    _tick(kb)
    assert _arms(gh) == []  # the cards are open: nothing is read or armed
    _done(kb.kanban_db_path())
    _reviews(gh, ("CHANGES_REQUESTED", OTHER), ("COMMENTED", head), *_lenses(kb, head))
    states = []  # the row as each exchange finds it: every read before the reservation, the arm after it
    gh["hooks"].update({kind: lambda: states.append(_state(kb, head)) for kind in ("PULL", "GET", "PUT")})

    _tick(kb)

    assert _arms(gh) == [{"sha": head, "merge_method": "merge"}]
    assert gh["calls"][-3:] == [("GET", f"/repos/{REPO}/pulls/41"), ("GET", f"/repos/{REPO}/pulls/41/reviews"),
                                ("PUT", f"/repos/{REPO}/pulls/41/merge")]
    assert states == ["review_cards_created", "review_cards_created", "auto_merge_pending"]
    assert _state(kb, head) == "auto_merge_armed"
    events = _events(kb, tid)
    assert [(kind, p["head"], p["pull_request_number"]) for kind, p in events if "auto_merge" in kind] == [
        ("delivery_auto_merge_armed", head, 41)]
    calls = len(gh["calls"])

    _tick(kb)
    _tick(kb)

    # Card 10: an armed pull request is read once a pass to see whether it merged; nothing else is sent.
    assert gh["calls"][calls:] == [("GET", f"/repos/{REPO}/pulls/41")] * 2
    assert (_events(kb, tid), _state(kb, head)) == (events, "auto_merge_armed")


def test_a_clean_pull_request_approved_on_h_is_merged_at_h_and_the_next_pass_records_the_merge(armed):
    """The owner's facts: review cards start only once CI is green, so GitHub shows the pull request clean and
    arms no auto-merge on it. The arm step merges it itself, with exactly the sha of H; the next pass reads it
    merged into main and adds the merge to the waiting release decision."""
    from hermes_cli import release_ledger

    kb, root, repo, gh = armed
    tid, head = _ready(armed)
    _tick(kb)
    _done(kb.kanban_db_path())
    _reviews(gh, *_lenses(kb, head))

    _tick(kb)

    assert gh["writes"] == [("PUT", f"/repos/{REPO}/pulls/41/merge", {"sha": head, "merge_method": "merge"})]
    assert _state(kb, head) == "auto_merge_armed"
    gh["ended"][41] = ARMED["sha"]  # GitHub now shows it merged into main as the merge commit it answered

    _tick(kb)

    conn = release_ledger.connect()
    try:
        assert [(batch["state"], [(m["merge_commit"], m["reviewed_head"]) for m in batch["members"]])
                for batch in release_ledger.list_batches(conn)] == [("open", [(ARMED["sha"], head)])]
    finally:
        conn.close()
    assert (_state(kb, head), len(gh["writes"])) == ("merged", 1)


@pytest.mark.parametrize("answer, state", [
    ((409, {"message": "Head branch was modified. Review and try the merge again."}), "refused"),
    ((405, {"message": "Pull Request is not mergeable"}), "refused"), ("timeout", "unknown"),
], ids=["moved_head_409", "unmergeable_405", "no_answer"])
def test_a_refused_or_unanswered_merge_is_recorded_once_with_githubs_status_and_message(armed, answer, state):
    """GitHub merges only a mergeable pull request still at the sha it is sent: a moved head answers 409 and an
    unmergeable pull request 405, each recorded refused once with GitHub's status and message; no answer is
    recorded unknown. Nothing is sent again."""
    kb, root, repo, gh = armed
    tid, head = _ready(armed)
    _tick(kb)
    _done(kb.kanban_db_path())
    _reviews(gh, *_lenses(kb, head))
    gh["arm"] = answer

    _tick(kb)
    _tick(kb)

    assert (_arms(gh), _state(kb, head)) == ([{"sha": head, "merge_method": "merge"}], f"auto_merge_{state}")
    status, message, reason = (None, None, "network_error") if answer == "timeout" else (
        answer[0], answer[1]["message"], None)
    assert [(kind, p["head"], p["status"], p["message"], p["reason"]) for kind, p in _events(kb, tid)
            if "auto_merge" in kind] == [(f"delivery_auto_merge_{state}", head, status, message, reason)]


@pytest.mark.parametrize("answer, state", [
    ((200, {"sha": None, "merged": False, "message": "Pull Request is not mergeable"}), "refused"),
    ((200, None), "refused"), ((201, ARMED), "refused"),
    ((403, None), "refused"), ((422, None), "refused"), ((429, None), "refused"), ((500, None), "refused"),
    ((503, None), "refused"), ("timeout", "unknown")])
def test_a_refused_arm_is_recorded_once_for_its_head(armed, answer, state):
    kb, root, repo, gh = armed
    tid, head = _ready(armed)
    _tick(kb)
    _done(kb.kanban_db_path())
    _reviews(gh, *_lenses(kb, head))
    gh["arm"] = answer

    _tick(kb)
    _tick(kb)
    _tick(kb)

    assert len(_arms(gh)) == 1 and _state(kb, head) == f"auto_merge_{state}"
    refused = [(kind, p) for kind, p in _events(kb, tid) if "auto_merge" in kind]
    assert [(kind, p["head"], p.get("status")) for kind, p in refused] == [
        (f"delivery_auto_merge_{state}", head, None if answer == "timeout" else answer[0])]
    assert [p["message"] for kind, p in refused] == [answer[1]["message"] if answer != "timeout" and answer[1] else None]


@pytest.mark.parametrize("status, raw", [(200, b"null"), (200, b"false"), (200, b'"merged"'), (200, b'{"merged": true'),
                                         (202, b"null")], ids=["null", "false", "string", "malformed", "202_null"])
def test_a_merge_answered_with_an_unreadable_body_is_refused_once_with_githubs_status(request, monkeypatch, status, raw):
    """A received answer is never unknown: GitHub's status stands, whatever the body. Other endpoints still raise."""
    from hermes_cli import kanban_delivery_github as transport

    exchange = transport._exchange  # the real one, taken before the armed world replaces it with a table
    kb, root, repo, gh = armed = request.getfixturevalue("armed")
    tid, head = _ready(armed)
    _tick(kb)
    _done(kb.kanban_db_path())
    _reviews(gh, *_lenses(kb, head))
    calls, table = [], transport._exchange

    class Handler(BaseHTTPRequestHandler):  # GitHub's whole answer, on the wire, to each request
        def do_PUT(self):
            calls.append((self.command, self.path, self.rfile.read(int(self.headers.get("Content-Length", 0)))))
            self.wfile.write(b"HTTP/1.0 %d Answer\r\nContent-Length: %d\r\n\r\n%s" % (status, len(raw), raw))
        do_GET = do_PUT

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    request.addfinalizer(lambda: server.shutdown() or server.server_close())
    monkeypatch.setattr(transport, "_connect", lambda: http.client.HTTPConnection(*server.server_address, timeout=10))
    monkeypatch.setattr(transport, "_exchange", lambda *call: (exchange if call[0] == "PUT" else table)(*call))

    _tick(kb)
    _tick(kb)
    monkeypatch.setattr(transport, "_exchange", exchange)
    with pytest.raises(transport.GitHubTransportError, match="^bad_response$"):  # any other endpoint still raises
        transport.GitHubTransport("publish", REPO).request("GET", f"/repos/{REPO}/pulls/41")

    assert [c[:2] for c in calls] == [("PUT", f"/repos/{REPO}/pulls/41/merge"), ("GET", f"/repos/{REPO}/pulls/41")]
    assert _state(kb, head) == "auto_merge_refused"
    assert [(kind, p["head"], p["status"], p["reason"]) for kind, p in _events(kb, tid) if "auto_merge" in kind] == [
        ("delivery_auto_merge_refused", head, status, "bad_response")]


def test_a_new_head_starts_fresh_after_a_refused_arm(armed):
    kb, root, repo, gh = armed
    tid, first = _ready(armed)
    _tick(kb)
    _done(kb.kanban_db_path())
    _reviews(gh, *_lenses(kb, first))
    gh["arm"] = (403, None)
    _tick(kb)

    second = _rework(kb, repo, tid)
    _published(kb, repo, gh, tid, second)
    _tick(kb)
    _done(kb.kanban_db_path())
    _reviews(gh, *_lenses(kb, first), *_lenses(kb, second))
    gh["arm"] = (200, ARMED)
    _tick(kb)
    _tick(kb)

    assert [body["sha"] for body in _arms(gh)] == [first, second]
    assert [(kind, p["head"]) for kind, p in _events(kb, tid) if "auto_merge" in kind] == [
        ("delivery_auto_merge_refused", first), ("delivery_auto_merge_armed", second)]
    assert _state(kb, second) == "auto_merge_armed"


@pytest.mark.parametrize("total, last, pages, outcome", [
    (99, None, 1, "armed"), (100, None, 1, "armed"), (101, None, 2, "armed"), (200, None, 2, "armed"),
    (150, "bot", 2, None), (150, "lens", 2, None), (1000, None, 10, "armed"), (1000, "unknown", 10, "refused"),
    (1001, None, 10, "refused"), (1050, None, 10, "refused")])
def test_every_review_page_is_read_up_to_ten_and_a_longer_history_refuses_h_once(armed, monkeypatch, total, last,
                                                                                 pages, outcome):
    """Owner rule 2 of round 3: reviews are read 100 a page until one is not full or GitHub names no next page,
    ten pages at most, and judged over them all, so a later page can revoke. A full tenth page with a next page
    named, or with no word on one, refuses H once: nothing is sent, then or later. No eleventh page is read."""
    from hermes_cli import kanban_delivery_github as transport

    kb, root, repo, gh = armed
    tid, head = _ready(armed)
    if last == "unknown":  # answers that do not say whether a next page follows
        exchange = transport._exchange
        monkeypatch.setattr(transport, "_exchange", lambda *args: exchange(*args)[:2])
    _tick(kb)
    _done(kb.kanban_db_path())
    lens = _lenses(kb, head)  # the bot's approval of H for each card: last of all, or first before a revocation
    first, tail = {"bot": (lens, [("CHANGES_REQUESTED", head)]),
                   "lens": (lens, [("CHANGES_REQUESTED", head, lens[0][2]), ("APPROVED", head)])}.get(last, ([], lens))
    _reviews(gh, *first, *[("APPROVED", head, None, "someone")] * (total - len(first) - len(tail)), *tail)
    _tick(kb)
    reads = [(query["per_page"], query.get("page")) for path, query in gh["queries"] if path.endswith("/reviews")]
    assert reads == [(["100"], [str(n)]) for n in range(1, pages + 1)]
    assert ([body["sha"] for body in _arms(gh)], _state(kb, head)) == (
        [head] if outcome == "armed" else [], f"auto_merge_{outcome}" if outcome else "review_cards_created")
    assert [(kind, p["head"], p["status"], p["reason"]) for kind, p in _events(kb, tid) if "auto_merge" in kind] == {
        "armed": [("delivery_auto_merge_armed", head, 200, None)], None: [],
        "refused": [("delivery_auto_merge_refused", head, None, "review_history_over_10_pages")]}[outcome]
    assert _raw(kb.kanban_db_path(), "SELECT id FROM kanban_deliveries WHERE publish_lease IS NOT NULL") == []
    calls, events, arms, state = len(gh["calls"]), _events(kb, tid), _arms(gh), _state(kb, head)
    _tick(kb)
    assert (_events(kb, tid), _arms(gh), _state(kb, head)) == (events, arms, state)  # nothing more sent or recorded
    new = gh["calls"][calls:]
    if outcome == "armed":  # card 10: an armed pull request is read once a pass to see whether it merged
        assert new == [("GET", f"/repos/{REPO}/pulls/41")]
    else:
        assert (new == []) is (outcome is not None)  # only a waiting H is read again


@pytest.mark.parametrize("read", ["PULL", "GET"])  # changed while the pull request is read, or its reviews
@pytest.mark.parametrize("change", ["card_reopened", "changes_requested", "card_head", "source_reopened", "source_head",
                                    "tier_raised", "row_head", "row_number", "row_state", "receipt", "off", "overlap"])
def test_evidence_that_changes_while_github_is_read_sends_no_arm(armed, read, change):
    """Owner rule 2 of round 2: every GitHub read comes first; then one write transaction reads the complete
    evidence again and reserves the arm, the one call after it. A change sends nothing and reserves nothing;
    an overlapping pass sends no second arm."""
    kb, root, repo, gh = armed
    db = kb.kanban_db_path()
    tid, head = _ready(armed, tier=0)
    _tick(kb)
    _done(db)
    _reviews(gh, *_lenses(kb, head))
    card, task, row = _cards(kb)[0]["id"], "UPDATE tasks SET {} WHERE id = ?", "UPDATE kanban_deliveries SET {} WHERE id = 1"
    gh["hooks"][read] = {
        "card_reopened": lambda: _raw(db, task.format("status = 'ready'"), card),
        "changes_requested": lambda: _raw(db, "INSERT INTO task_events (task_id, kind, created_at) "
                                              "VALUES (?, 'changes_requested', 0)", card),
        "card_head": lambda: _raw(db, task.format("head_commit = ?"), OTHER, card),
        "source_reopened": lambda: _raw(db, task.format("status = 'ready'"), tid),
        "source_head": lambda: _raw(db, task.format("head_commit = ?"), OTHER, tid),
        "tier_raised": lambda: _raw(db, task.format("risk_tier = 2"), tid),
        "row_head": lambda: _raw(db, row.format("pull_request_head = ?"), OTHER),
        "row_number": lambda: _raw(db, row.format("pull_request_number = 42")),
        "row_state": lambda: _raw(db, row.format("pull_request_state = 'open'")),
        "receipt": lambda: _raw(db, "UPDATE task_events SET payload = json_set(payload, '$.repository', ?) "
                                    "WHERE kind = 'delivery_published'", ELSEWHERE),
        "off": lambda: _write_config(root, enabled=False),
        "overlap": lambda: _tick(kb)}[change]

    _tick(kb)

    assert len(_arms(gh)) == (change == "overlap")
    assert len([kind for kind, _ in _events(kb, tid) if "auto_merge" in kind]) == (change == "overlap")
    assert len(_raw(db, "SELECT id FROM kanban_deliveries WHERE pull_request_state LIKE 'auto_merge%' "
                        "OR publish_lease IS NOT NULL")) == (change == "overlap")


@pytest.mark.parametrize("ended", [None, False, True])  # the armed pull request open, closed, or merged
def test_a_new_head_is_never_pushed_to_an_armed_pull_request_and_stays_parked(armed, ended):
    """Owner rule 4 of round 2: H2, refused once because the pull request of H is armed, stays parked with that
    one refusal, whatever GitHub reports of that pull request and after H2 is approved again: nothing is read,
    pushed or recorded for it. A later card owns its release. The armed pull request of H is only read, once a
    pass, to see whether it merged into main (card 10)."""
    kb, root, repo, gh = armed
    tid, head = _ready(armed, tier=0)
    _tick(kb)
    _done(kb.kanban_db_path())
    _reviews(gh, *_lenses(kb, head))
    _tick(kb)
    calls = list(gh["calls"])

    second = _rework(kb, repo, tid)
    _tick(kb)
    if ended is not None:
        gh["ended"][41] = ended
    assert _rework(kb, repo, tid, commit=False) == second  # approved again through the owner's path
    _tick(kb)
    _tick(kb)

    new = gh["calls"][len(calls):]
    assert gh["calls"][:len(calls)] == calls and new and set(new) == {("GET", f"/repos/{REPO}/pulls/41")}
    # This stand-in's pull answer names no base and no merge commit, so H is never taken as merged into main
    # and stays armed; tests/hermes_cli/test_release_intake.py covers a merge.
    assert _state(kb, head) == "auto_merge_armed"
    assert [(p["code"], p["head"]) for kind, p in _events(kb, tid) if kind == "delivery_refused"] == [
        ("auto_merge_armed", second)]
    assert _raw(kb.kanban_db_path(), "SELECT publish_refusal FROM kanban_deliveries WHERE source_head = ?",
                second) == [{"publish_refusal": "auto_merge_armed"}]


@pytest.mark.parametrize("case", ["same", "origin_changed", "base_elsewhere", "number_elsewhere"])
def test_a_rework_card_that_continues_a_delivered_head_closes_the_earlier_pull_request_once(armed, case):
    """A pull request is its repository and number as the publish step recorded them, never the git origin:
    a continuation published in another repository replaces nothing, and its numbers hold back no close."""
    kb, root, repo, gh = armed
    db = kb.kanban_db_path()
    first, delivered = _ready(armed)
    rework, continued = _approved(kb, repo, "rework")
    other, unrelated = _approved(kb, repo, "unrelated")
    _continues(db, rework, delivered, first)
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
    _continues(db, rework, delivered, first)
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


@pytest.mark.parametrize("linked", ["first", "second", None])  # the continuation's task_links parent, if any
def test_a_continuation_closes_only_the_pull_request_of_the_source_card_it_is_linked_to(armed, linked):
    """Owner rule 4 of round 1: a replacement is matched by the repository, the number and the source card, never
    by a base commit alone. Two source cards are published at the same H as 41 and 43, and a continuation of base
    H as 42: only the pull request of its task_links parent is closed, once, and with no parent none is."""
    kb, root, repo, gh = armed
    db = kb.kanban_db_path()
    first, head = _ready(armed)
    second, _ = _approved(kb, repo, "second")
    _raw(db, "UPDATE kanban_deliveries SET source_head = ? WHERE source_task_id = ?", head, second)
    _raw(db, "UPDATE tasks SET head_commit = ? WHERE id = ?", head, second)  # an independent source done at H too
    _published_as(db, gh, second, head, 43)
    rework, continued = _approved(kb, repo, "rework")
    _published_as(db, gh, rework, continued, 42)
    source, closed = {"first": (first, 41), "second": (second, 43)}.get(linked, (None, None))
    _continues(db, rework, head, source)
    gh["close"] = (200, {"number": closed, "state": "closed"})

    _tick(kb)
    _tick(kb)

    assert [write for write in gh["writes"] if write[0] == "PATCH"] == (
        [("PATCH", f"/repos/{REPO}/pulls/{closed}", {"state": "closed"})] if linked else [])
    assert [(tid, kind, p["pull_request_number"], p["replaced_by"]) for tid in (first, second)
            for kind, p in _events(kb, tid) if kind.startswith(("delivery_close", "delivery_replaced"))] == [
        (source, kind, closed, rework) for kind in ("delivery_close_pending", "delivery_replaced") if linked]
    assert _raw(db, "SELECT pull_request_number FROM kanban_deliveries WHERE pull_request_state = 'replaced'") == (
        [{"pull_request_number": closed}] if linked else [])


def _continued(armed, tier=2):
    """H delivered as pull request 41 and a rework card published as 42, both past their review step; the rework
    card then continues H as a linked rework card does. Returns the source card, H and the rework card."""
    kb, root, repo, gh = armed
    first, delivered = _ready(armed, tier=tier)
    rework, continued = _approved(kb, repo, "rework")
    _published_as(kb.kanban_db_path(), gh, rework, continued, 42)
    _tick(kb)
    _continues(kb.kanban_db_path(), rework, delivered, first)
    return first, delivered, rework


@pytest.mark.parametrize("change", [None, "source_status", "source_head", "source_tier", "rework_status",
                                    "rework_head", "rework_delivery"])
def test_a_replacement_proof_that_changes_while_the_earlier_pull_request_is_read_sends_no_close(armed, change):
    """Owner rule 2 of round 2: the earlier pull request is read first; then one write transaction reads the
    complete replacement proof again and reserves the close, the one call after it."""
    kb, root, repo, gh = armed
    db = kb.kanban_db_path()
    first, delivered, rework = _continued(armed)
    task = "UPDATE tasks SET {} WHERE id = ?"
    gh["hooks"]["PULL"] = {
        None: lambda: None,
        "source_status": lambda: _raw(db, task.format("status = 'ready'"), first),
        "source_head": lambda: _raw(db, task.format("head_commit = ?"), OTHER, first),
        "source_tier": lambda: _raw(db, task.format("risk_tier = 0"), first),
        "rework_status": lambda: _raw(db, task.format("status = 'ready'"), rework),
        "rework_head": lambda: _raw(db, task.format("head_commit = ?"), OTHER, rework),
        "rework_delivery": lambda: _raw(db, "UPDATE kanban_deliveries SET source_head = ? WHERE source_task_id = ?",
                                        OTHER, rework)}[change]

    _tick(kb)

    assert [write[1] for write in gh["writes"]] == ([f"/repos/{REPO}/pulls/41"] if change is None else [])
    assert [kind for kind, _ in _events(kb, first) if kind.startswith("delivery_close")] == (
        ["delivery_close_pending"] if change is None else [])


@pytest.mark.parametrize("close", [(200, {"number": 41, "state": "closed"}), (403, None)])
def test_arm_and_close_of_one_pull_request_share_one_claim_and_each_outcome_names_its_attempt(armed, close):
    """Owner rule 3 of round 2: a close tried while the arm of the same pull request is in flight sends nothing;
    the close after the confirmed arm is sent once, and only its own outcome is written. A refused close keeps
    the armed hold, so H2 of the same card is still refused; an arm never overwrites a confirmed close."""
    kb, root, repo, gh = armed
    first, delivered, rework = _continued(armed, tier=0)
    _done(kb.kanban_db_path())
    _reviews(gh, *_lenses(kb, delivered))
    gh["close"], gh["hooks"]["PUT"] = close, lambda: _tick(kb)

    _tick(kb)
    _tick(kb)

    assert [write[0] for write in gh["writes"]] == ["PUT", "PATCH"]
    assert _state(kb, delivered) == ("replaced" if close[0] == 200 else "auto_merge_armed")
    outcomes = [(kind, p["attempt"]) for kind, p in _events(kb, first)
                if kind.startswith(("delivery_auto", "delivery_close")) or kind == "delivery_replaced"]
    assert [kind for kind, _ in outcomes] == ["delivery_auto_merge_armed", "delivery_close_pending",
                                              "delivery_replaced" if close[0] == 200 else "delivery_close_refused"]
    assert None != outcomes[0][1] != outcomes[1][1] == outcomes[2][1]
    _rework(kb, repo, first)
    _tick(kb)
    assert len(gh["writes"]) == 2 and ("auto_merge_armed" in [
        p["code"] for kind, p in _events(kb, first) if kind == "delivery_refused"]) is (close[0] == 403)


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
    _reviews(gh, *_lenses(kb, head, db=other_db))
    _reviews(gh, *_lenses(kb, home_head, db=root_db), number=44)
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
    assert sorted((path, body["sha"]) for method, path, body in gh["writes"] if method == "PUT") == [
        (f"/repos/{REPO}/pulls/41/merge", head), (f"/repos/{REPO}/pulls/44/merge", home_head)]
    assert (_state(kb, head, other_db), _state(kb, home_head, root_db)) == ("auto_merge_armed", "auto_merge_armed")
