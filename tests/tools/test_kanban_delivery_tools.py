"""Delivery slice 4: kanban_delivery_publish (T1), driven through the registry.

Everything runs on a real SQLite board, a real git repository and a real bare
remote laid out like GitHub. A local stand-in plays GitHub, as the slice 2
fixture of tests/hermes_cli/test_kanban_delivery_github.py does, and here also
lists and creates pull requests. The only other stand-ins are the reviewer the
model policy nominates and the final process launch of the dispatcher's spawn,
both as in the slice 3 tests.
"""

from __future__ import annotations

import http.client
import importlib
import json
import sqlite3
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlsplit

import jwt
import pytest

from tests.hermes_cli.test_kanban_delivery import (  # noqa: F401  (board is a fixture)
    IMPLEMENTER,
    INTEGRATOR,
    REVIEWER,
    _approve,
    _become,
    _git,
    _park,
    _spawned_worker_env,
    _write_config,
    board,
)
from tests.hermes_cli.test_kanban_delivery_github import (
    APP_ID,
    INSTALLATION,
    REPO,
    TOKEN,
    _new_key,
    _pem,
    _remote_head,
)

TOOL = "kanban_delivery_publish"
PULLS = f"/repos/{REPO}/pulls"


def _tool():
    """Import inside each test, so a missing module fails that test rather than the whole file."""
    return importlib.import_module("tools.kanban_delivery_tools")


@pytest.fixture
def github():
    """GitHub for one repository: the App's token call, listing, creating and reading pull requests,
    and reading a branch.

    A pull request's head and a branch's head are what the branch holds now in the bare remote, as
    on GitHub, where a branch the bare remote lacks answers 404. Titles,
    bodies, URLs, a header and every error carry the minted token, so any echo of a raw answer
    shows."""
    state = {"public": None, "jwts": [], "token_requests": [], "calls": [], "pulls": [],
             "bare": None, "on_create": None, "pushes": []}

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self._route("GET")

        def do_POST(self):
            self._route("POST")

        def _reply(self, status, value=None):
            data = b"" if value is None else json.dumps(value).encode()
            self.send_response(status)
            self.send_header("X-Echo", TOKEN)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _mint(self, auth, body):
            try:
                claims = jwt.decode(auth.removeprefix("Bearer "), state["public"], algorithms=["RS256"],
                                    issuer=APP_ID, options={"require": ["exp", "iat", "iss"]})
            except jwt.PyJWTError:
                return self._reply(401, {"message": f"A JSON web token could not be decoded {TOKEN}"})
            if claims["exp"] - time.time() > 600:
                return self._reply(401, {"message": "Expiration time too far in the future"})
            state["jwts"].append(auth)
            state["token_requests"].append(body)
            return self._reply(201, {
                "token": TOKEN, "expires_at": "2026-10-04T12:00:00Z", "permissions": body["permissions"],
                "repository_selection": "selected",
                "repositories": [{"name": n, "full_name": f"mtbitcr/{n}"} for n in body["repositories"]]})

        def _pull(self, pull):
            head = _remote_head(state["bare"], pull["head_ref"]) or pull["sha"]
            return {"number": pull["number"], "state": pull["state"], "title": TOKEN, "body": TOKEN,
                    "html_url": f"https://github.com/{REPO}/pull/{pull['number']}?t={TOKEN}",
                    "head": {"sha": head, "ref": pull["head_ref"], "label": f"mtbitcr:{pull['head_ref']}"},
                    "base": {"ref": pull["base_ref"]}}

        def _route(self, method):
            length = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(length)) if length else None
            auth = self.headers.get("Authorization", "")
            if method == "POST" and self.path == f"/app/installations/{INSTALLATION}/access_tokens":
                return self._mint(auth, body)
            parts = urlsplit(self.path)
            query = {key: values[0] for key, values in parse_qs(parts.query).items()}
            state["calls"].append((method, parts.path, query, body))
            if auth != f"Bearer {TOKEN}":
                return self._reply(401, {"message": f"Bad credentials {TOKEN}"})
            if method == "GET" and parts.path == PULLS:
                ref = query.get("head", "").partition(":")[2]
                return self._reply(200, [self._pull(p) for p in state["pulls"]
                                         if query.get("state", "open") in ("all", p["state"])
                                         and (not ref or p["head_ref"] == ref)])
            if method == "POST" and parts.path == PULLS:
                if state["on_create"] is not None:
                    state["on_create"]()
                sha = _remote_head(state["bare"], body["head"])
                taken = [p for p in state["pulls"] if p["state"] == "open"
                         and (p["head_ref"], p["base_ref"]) == (body["head"], body["base"])]
                if not sha or taken:
                    return self._reply(422, {"message": f"Validation Failed {TOKEN}"})
                pull = {"number": 41 + len(state["pulls"]), "state": "open", "head_ref": body["head"],
                        "base_ref": body["base"], "sha": sha}
                state["pulls"].append(pull)
                return self._reply(201, self._pull(pull))
            pull = next((p for p in state["pulls"] if parts.path == f"{PULLS}/{p['number']}"), None)
            if method == "GET" and pull is not None:
                return self._reply(200, self._pull(pull))
            branch = parts.path.removeprefix(f"/repos/{REPO}/git/ref/heads/")
            sha = _remote_head(state["bare"], branch) if branch != parts.path else ""
            if method == "GET" and sha:
                return self._reply(200, {"ref": f"refs/heads/{branch}", "node_id": TOKEN,
                                         "url": f"https://api.github.com/repos/{REPO}/git/{TOKEN}",
                                         "object": {"type": "commit", "sha": sha, "url": TOKEN}})
            return self._reply(404, {"message": f"Not Found {TOKEN}"})

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    state["port"] = server.server_port
    try:
        yield state
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


@pytest.fixture
def world(board, github, tmp_path, monkeypatch):
    """The board's repository with its GitHub origin, an empty bare remote laid out like GitHub,
    the transport's seams pointed at both, the App key under the root and delivery switched on.
    Every push still runs for real; the spy only records what it was asked."""
    kb, root, repo = board
    from hermes_cli import kanban_delivery_github as transport

    _git(repo, "remote", "add", "origin", f"https://github.com/{REPO}.git")
    remotes = tmp_path / "remotes"
    bare = remotes / "mtbitcr" / "hermes-agent.git"
    subprocess.run(["git", "init", "-q", "--bare", str(bare)], check=True, capture_output=True)
    github["bare"] = bare
    port = github["port"]
    monkeypatch.setattr(transport, "_connect", lambda: http.client.HTTPConnection("127.0.0.1", port, timeout=10))
    monkeypatch.setattr(transport, "GIT_ORIGIN", remotes.as_uri())
    push = transport.GitHubTransport.push

    def spy(self, worktree, branch, sha, *, expected):
        github["pushes"].append((branch, sha, expected))
        return push(self, worktree, branch, sha, expected=expected)

    monkeypatch.setattr(transport.GitHubTransport, "push", spy)
    key = _new_key()
    key_file = root / "secrets" / "github-app" / "raphael-agent-factory.pem"
    key_file.parent.mkdir(parents=True)
    key_file.write_bytes(_pem(key))
    github["public"] = key.public_key()
    _write_config(root, enabled=True, profile=INTEGRATOR)
    return kb, root, repo, bare


def _approved(kb, repo, name: str = "feature"):
    """A built and approved source card, and the integration card its approval created."""
    conn = kb.connect()
    try:
        tid, head, _ = _park(kb, conn, repo, name)
        assert _approve(kb, conn, tid) is True
        card = conn.execute(
            "SELECT integration_task_id FROM kanban_deliveries "
            "WHERE source_task_id = ? AND source_head = ?",
            (tid, head),
        ).fetchone()[0]
    finally:
        conn.close()
    return tid, head, card


def _claim(kb, card: str, monkeypatch) -> int:
    """The integration worker claims its card; this process is now that run."""
    conn = kb.connect()
    try:
        run = kb.claim_task(conn, card, claimer=f"{INTEGRATOR}:1")
        assert run is not None and run.status == "running"
    finally:
        conn.close()
    monkeypatch.setenv("HERMES_KANBAN_DB", str(kb.kanban_db_path()))
    monkeypatch.setenv("HERMES_KANBAN_TASK", card)
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(run.current_run_id))
    return run.current_run_id


def _rework(kb, repo, tid: str):
    """The owner sends the approved card back; the implementer adds a commit on the approved head
    and the reviewer approves again, which records a second delivery and integration card."""
    conn = kb.connect()
    try:
        moved = kb.cas_transition_task(
            conn, tid, expected_status="done", expected_revision=kb.task_event_revision(conn, tid),
            to_status="ready", event_kind="owner_move", event_payload={"to": "ready"},
        )
        assert moved["moved"] is True
        assert kb.assign_task(conn, tid, IMPLEMENTER)
        run = kb.claim_task(conn, tid, claimer=f"{IMPLEMENTER}:1")
        workspace, _ = kb._resolve_worktree_workspace(run)
        (workspace / "src" / "impl" / "feature.py").write_text("ok = 2\n", encoding="utf-8")
        _git(workspace, "add", "src/impl/feature.py")
        _git(workspace, "commit", "-m", "fix: feature")
        head = _git(workspace, "rev-parse", "HEAD")
        kb.complete_task(conn, tid, summary="reworked", expected_run_id=run.current_run_id)
        assert _approve(kb, conn, tid) is True
        card = conn.execute(
            "SELECT integration_task_id FROM kanban_deliveries "
            "WHERE source_task_id = ? AND source_head = ?",
            (tid, head),
        ).fetchone()[0]
    finally:
        conn.close()
    return head, card


def _dispatch(args=None) -> str:
    _tool()
    from tools.registry import registry

    return registry.dispatch(TOOL, {} if args is None else args)


def _publish(args=None) -> dict:
    return json.loads(_dispatch(args))


def _listed() -> bool:
    _tool()
    from tools.registry import registry

    return [d["function"]["name"] for d in registry.get_definitions({TOOL}, quiet=True)] == [TOOL]


def _sql(kb, statement: str, *params):
    """Read or change a board row straight in the file, past the kernel."""
    raw = sqlite3.connect(str(kb.kanban_db_path()))
    try:
        rows = raw.execute(statement, params).fetchall()
        raw.commit()
    finally:
        raw.close()
    return rows


def _ledger(kb, card: str):
    return _sql(
        kb,
        "SELECT pull_request_number, pull_request_head, pull_request_state, pull_request_branch "
        "FROM kanban_deliveries WHERE integration_task_id = ?",
        card,
    )[0]


def _events(kb, card: str) -> list:
    return [
        (kind, json.loads(payload), run_id)
        for kind, payload, run_id in _sql(
            kb,
            "SELECT kind, payload, run_id FROM task_events "
            "WHERE task_id = ? AND kind LIKE 'delivery%' ORDER BY id",
            card,
        )
    ]


def _open_pull(github, branch: str, base: str, sha: str) -> int:
    """A pull request someone else opened for the branch."""
    github["pulls"].append({"number": 41 + len(github["pulls"]), "state": "open", "head_ref": branch,
                            "base_ref": base, "sha": sha})
    return github["pulls"][-1]["number"]


def _refused(result: dict, code: str) -> None:
    assert set(result) == {"error", "code", "detail"}, result
    assert result["code"] == code, result


def test_publish_opens_one_pull_request_at_the_approved_head(world, github, monkeypatch):
    kb, root, repo, bare = world
    tid, head, card = _approved(kb, repo)
    base = _git(repo, "rev-parse", "main")
    approval = _sql(kb, "SELECT MAX(id) FROM task_events WHERE task_id = ? AND kind = 'completed'", tid)[0][0]
    run_id = _claim(kb, card, monkeypatch)
    branch = "delivery/" + tid

    raw = _dispatch()

    assert json.loads(raw) == {"state": "created", "head": head, "pull_request_number": 41, "branch": branch}
    assert TOKEN not in raw
    assert _remote_head(bare, branch) == head
    assert [(p["number"], p["head_ref"], p["base_ref"]) for p in github["pulls"]] == [(41, branch, "main")]
    assert github["pushes"] == [(branch, head, "")]
    ref = f"/repos/{REPO}/git/ref/heads/{branch}"
    assert [(method, path) for method, path, _, _ in github["calls"]] == [
        ("GET", PULLS), ("GET", ref), ("POST", PULLS), ("GET", f"{PULLS}/41"), ("GET", ref)]
    listed, created = github["calls"][0][2], github["calls"][2][3]
    assert (listed["state"], listed["head"]) == ("open", f"mtbitcr:{branch}")
    assert set(created) == {"title", "head", "base", "body"}
    assert (created["head"], created["base"]) == (branch, "main")
    for prose in ("implemented the slice", "looks right", "build feature"):
        assert prose not in created["title"] + created["body"]
    # One fresh token, narrowed to the publish step and this one repository.
    assert github["token_requests"] == [{
        "repositories": ["hermes-agent"], "permissions": {"contents": "write", "pull_requests": "write"},
    }]
    assert _ledger(kb, card) == (41, head, "open", branch)
    assert _events(kb, card) == [
        ("delivery_bound", {
            "source_task_id": tid, "head": head, "base_commit": base, "reviewer": REVIEWER,
            "implementer": IMPLEMENTER, "approval_event_id": approval, "base_is_ancestor": True,
        }, run_id),
        ("delivery_published", {
            "repository": REPO, "branch": branch, "pull_request_number": 41, "head": head,
            "state": "created",
        }, run_id),
    ]


def test_a_second_publish_is_already_published_and_changes_nothing(world, github, monkeypatch):
    kb, root, repo, bare = world
    tid, head, card = _approved(kb, repo)
    run_id = _claim(kb, card, monkeypatch)
    branch = "delivery/" + tid
    assert _publish()["state"] == "created"
    calls, ledger = len(github["calls"]), _ledger(kb, card)

    again = _publish()

    assert again == {"state": "already_published", "head": head, "pull_request_number": 41, "branch": branch}
    assert [(method, path) for method, path, _, _ in github["calls"][calls:]] == [
        ("GET", PULLS), ("GET", f"/repos/{REPO}/git/ref/heads/{branch}")]
    assert github["pushes"] == [(branch, head, "")]
    assert len(github["pulls"]) == 1 and _remote_head(bare, branch) == head
    assert _ledger(kb, card) == ledger
    assert [(kind, payload.get("state"), run) for kind, payload, run in _events(kb, card)] == [
        ("delivery_bound", None, run_id),
        ("delivery_published", "created", run_id),
        ("delivery_published", "already_published", run_id),
    ]


def test_publish_refuses_when_the_head_moved(world, github, monkeypatch):
    """Section 8 test 1 through the tool: the pull request's head, the ledger's head or the source
    card's head differs from the approved head H."""
    kb, root, repo, bare = world
    tid, head, card = _approved(kb, repo)
    _claim(kb, card, monkeypatch)
    branch = "delivery/" + tid
    assert _publish()["state"] == "created"
    ledger, events, pushes = _ledger(kb, card), _events(kb, card), list(github["pushes"])

    # An outside push moves the branch, and with it the pull request's head.
    outside = _git(repo, "commit-tree", f"{head}^{{tree}}", "-p", head, "-m", "outside")
    _git(repo, "push", "-q", str(bare), f"{outside}:refs/heads/{branch}")
    _refused(_publish(), "head_moved")
    _git(repo, "push", "-q", "--force", str(bare), f"{head}:refs/heads/{branch}")

    # The ledger records another head while the pull request stays open.
    base = _git(repo, "rev-parse", "main")
    _sql(kb, "UPDATE kanban_deliveries SET pull_request_head = ? WHERE integration_task_id = ?", base, card)
    _refused(_publish(), "head_moved")
    _sql(kb, "UPDATE kanban_deliveries SET pull_request_head = ? WHERE integration_task_id = ?", head, card)

    # The source card's head is no longer the reviewed and bound head: refused before GitHub.
    calls, minted = len(github["calls"]), len(github["token_requests"])
    _sql(kb, "UPDATE tasks SET head_commit = ? WHERE id = ?", outside, tid)
    _refused(_publish(), "head_moved")
    assert (len(github["calls"]), len(github["token_requests"])) == (calls, minted)
    _sql(kb, "UPDATE tasks SET head_commit = ? WHERE id = ?", head, tid)

    assert github["pushes"] == pushes and len(github["pulls"]) == 1
    assert (_ledger(kb, card), _events(kb, card)) == (ledger, events)
    assert _publish()["state"] == "already_published"


def test_publish_refuses_a_second_pull_request(world, github, monkeypatch):
    """Section 8 test 2 through the tool, and a foreign pull request: another open pull request
    for the branch is refused, a recorded one at H answers already_published, and a recorded one
    at an unrelated head is refused."""
    kb, root, repo, bare = world
    tid, head, card = _approved(kb, repo)
    _claim(kb, card, monkeypatch)
    branch = "delivery/" + tid
    base = _git(repo, "rev-parse", "main")

    # Someone else's pull request for the branch, before anything is recorded.
    foreign = _git(repo, "commit-tree", f"{base}^{{tree}}", "-p", base, "-m", "foreign")
    _git(repo, "push", "-q", str(bare), f"{foreign}:refs/heads/{branch}")
    _open_pull(github, branch, "main", foreign)
    _refused(_publish(), "second_pull_request")
    assert github["pushes"] == [] and [c for c in github["calls"] if c[0] == "POST"] == []
    assert _ledger(kb, card) == (None, None, None, None) and _events(kb, card) == []
    assert _remote_head(bare, branch) == foreign

    # Closed and its branch deleted: the delivery opens its own.
    github["pulls"][0]["state"] = "closed"
    subprocess.run(["git", "--git-dir", str(bare), "update-ref", "-d", f"refs/heads/{branch}"], check=True)
    assert _publish() == {"state": "created", "head": head, "pull_request_number": 42, "branch": branch}

    # A second open pull request for the same head branch, against another base.
    _open_pull(github, branch, "release", head)
    _refused(_publish(), "second_pull_request")
    github["pulls"][-1]["state"] = "closed"

    # The recorded pull request at an unrelated head: refused, and no fast-forward either.
    orphan = _git(repo, "commit-tree", f"{head}^{{tree}}", "-m", "unrelated")
    _sql(kb, "UPDATE kanban_deliveries SET pull_request_head = ? WHERE integration_task_id = ?", orphan, card)
    _refused(_publish(), "head_moved")
    _sql(kb, "UPDATE kanban_deliveries SET pull_request_state = 'returned_for_changes' "
             "WHERE integration_task_id = ?", card)
    _refused(_publish(), "not_fast_forward")
    _sql(kb, "UPDATE kanban_deliveries SET pull_request_head = ?, pull_request_state = 'open' "
             "WHERE integration_task_id = ?", head, card)

    assert _publish()["state"] == "already_published"
    assert github["pushes"] == [(branch, head, "")]
    assert [p["number"] for p in github["pulls"]] == [41, 42, 43]


def test_publish_refuses_a_head_not_approved_by_the_kernel(world, github, monkeypatch):
    """Section 8 test 3 through the tool: the reviewer is the implementer, the card was reopened
    after the approval, or it is no longer done. Each is refused before GitHub is touched."""
    kb, root, repo, bare = world
    tid, head, card = _approved(kb, repo)
    _claim(kb, card, monkeypatch)
    approval_run = _sql(
        kb, "SELECT run_id FROM task_events WHERE task_id = ? AND kind = 'completed' ORDER BY id DESC LIMIT 1",
        tid,
    )[0][0]

    _sql(kb, "UPDATE task_runs SET profile = ? WHERE id = ?", IMPLEMENTER, approval_run)
    _refused(_publish(), "reviewer_not_independent")
    _sql(kb, "UPDATE task_runs SET profile = ? WHERE id = ?", REVIEWER, approval_run)

    conn = kb.connect()
    try:
        with kb.write_txn(conn):
            kb._append_event(conn, tid, "owner_move", {"to": "ready"})
        _refused(_publish(), "reopened_after_approval")
        moved = kb.cas_transition_task(
            conn, tid, expected_status="done", expected_revision=kb.task_event_revision(conn, tid),
            to_status="ready", event_kind="owner_move", event_payload={"to": "ready"},
        )
        assert moved["moved"] is True
    finally:
        conn.close()
    _refused(_publish(), "not_approved")

    assert (github["calls"], github["token_requests"], github["pushes"]) == ([], [], [])
    assert _ledger(kb, card) == (None, None, None, None) and _events(kb, card) == []


def test_publish_refuses_a_foreign_branch(world, github, monkeypatch):
    """The branch already holds a commit the delivery did not push: the branch read refuses it
    before any push."""
    kb, root, repo, bare = world
    tid, head, card = _approved(kb, repo)
    _claim(kb, card, monkeypatch)
    branch = "delivery/" + tid
    base = _git(repo, "rev-parse", "main")
    foreign = _git(repo, "commit-tree", f"{base}^{{tree}}", "-p", base, "-m", "foreign")
    _git(repo, "push", "-q", str(bare), f"{foreign}:refs/heads/{branch}")

    _refused(_publish(), "head_moved")

    assert github["pushes"] == []
    assert _remote_head(bare, branch) == foreign
    assert [c for c in github["calls"] if c[0] == "POST"] == [] and github["pulls"] == []
    assert _ledger(kb, card) == (None, None, None, None) and _events(kb, card) == []


def test_publish_adopts_a_branch_already_at_the_approved_head(world, github, monkeypatch):
    kb, root, repo, bare = world
    tid, head, card = _approved(kb, repo)
    run_id = _claim(kb, card, monkeypatch)
    branch = "delivery/" + tid
    _git(repo, "push", "-q", str(bare), f"{head}:refs/heads/{branch}")

    assert _publish() == {"state": "created", "head": head, "pull_request_number": 41, "branch": branch}

    assert github["pushes"] == []  # the branch read found H: adopted, not pushed
    assert [(p["number"], p["head_ref"]) for p in github["pulls"]] == [(41, branch)]
    assert _remote_head(bare, branch) == head
    assert _ledger(kb, card) == (41, head, "open", branch)
    assert _events(kb, card)[-1] == ("delivery_published", {
        "repository": REPO, "branch": branch, "pull_request_number": 41, "head": head, "state": "adopted",
    }, run_id)


def test_only_returned_for_changes_allows_a_fast_forward(world, github, monkeypatch):
    """A rework approved after the first publish binds a new head and a new integration card. The
    older card is superseded; the newer one fast-forwards the same pull request only once the
    ledger says returned_for_changes."""
    kb, root, repo, bare = world
    tid, first, card = _approved(kb, repo)
    _claim(kb, card, monkeypatch)
    branch = "delivery/" + tid
    assert _publish()["state"] == "created"

    second, newer = _rework(kb, repo, tid)
    assert newer != card
    _refused(_publish(), "superseded")

    run_id = _claim(kb, newer, monkeypatch)
    _refused(_publish(), "head_moved")
    assert github["pushes"] == [(branch, first, "")] and _ledger(kb, newer) == (None, None, None, None)

    _sql(kb, "UPDATE kanban_deliveries SET pull_request_state = 'returned_for_changes' "
             "WHERE integration_task_id = ?", card)
    assert _publish() == {"state": "fast_forwarded", "head": second, "pull_request_number": 41, "branch": branch}

    assert github["pushes"] == [(branch, first, ""), (branch, second, first)]
    assert _remote_head(bare, branch) == second
    assert [p["number"] for p in github["pulls"]] == [41]
    assert _ledger(kb, card) == (41, first, "returned_for_changes", branch)
    assert _ledger(kb, newer) == (41, second, "open", branch)
    assert [(kind, payload["head"], payload.get("state"), run) for kind, payload, run in _events(kb, newer)] == [
        ("delivery_bound", second, None, run_id),
        ("delivery_published", second, "fast_forwarded", run_id),
    ]
    assert _publish()["state"] == "already_published"


def test_a_run_that_loses_its_claim_during_the_network_work_records_nothing(world, github, monkeypatch):
    """The operator reclaims the card while GitHub is creating the pull request. The reclaim takes
    its own write transaction meanwhile, so the tool holds none, and the stale run stores nothing."""
    kb, root, repo, bare = world
    tid, head, card = _approved(kb, repo)
    _claim(kb, card, monkeypatch)
    reclaimed = []

    def reclaim():
        own = kb.connect()
        try:
            reclaimed.append(kb.reclaim_task(own, card, reason="operator"))
        finally:
            own.close()

    github["on_create"] = reclaim
    _refused(_publish(), "stale_run")
    assert reclaimed == [True]
    assert _ledger(kb, card) == (None, None, None, None) and _events(kb, card) == []

    # The next run finds the pull request it does not know and opens no second one.
    github["on_create"] = None
    _claim(kb, card, monkeypatch)
    _refused(_publish(), "second_pull_request")
    assert len(github["pulls"]) == 1 and len(github["pushes"]) == 1


def test_a_source_reopened_during_the_network_work_records_nothing(world, github, monkeypatch):
    """Finding 2 of the slice 4 review: the owner reopens the source card while GitHub is creating
    the pull request. The delivery row and the run stay unchanged, but the approval the publish was
    proven on is gone, so the final write stores nothing."""
    kb, root, repo, bare = world
    tid, head, card = _approved(kb, repo)
    run_id = _claim(kb, card, monkeypatch)
    reopened = []

    def unchanged():  # what the run's own recheck compares: the delivery rows and the run
        return (_sql(kb, "SELECT * FROM kanban_deliveries ORDER BY id"),
                _sql(kb, "SELECT status, current_run_id FROM tasks WHERE id = ?", card),
                _sql(kb, "SELECT status, ended_at FROM task_runs WHERE id = ?", run_id))

    def reopen():
        own = kb.connect()
        try:
            reopened.append(kb.cas_transition_task(
                own, tid, expected_status="done", expected_revision=kb.task_event_revision(own, tid),
                to_status="ready", event_kind="owner_move", event_payload={"to": "ready"},
            )["moved"])
        finally:
            own.close()

    before = unchanged()
    github["on_create"] = reopen
    _refused(_publish(), "source_changed")

    assert reopened == [True]
    assert _sql(kb, "SELECT status FROM tasks WHERE id = ?", tid) == [("ready",)]
    assert len(github["pulls"]) == 1 and len(github["pushes"]) == 1  # GitHub changed; the board did not
    assert unchanged() == before
    assert _ledger(kb, card) == (None, None, None, None) and _events(kb, card) == []


def test_an_outside_push_right_after_the_publish_push_records_nothing(world, github, monkeypatch):
    """Finding 3 of the slice 4 review: someone pushes to the branch right after the tool's own
    leased fast-forward. GitHub, read again after the push, shows the pull request and the branch
    at that commit rather than at H, so the tool answers head_moved and stores no success."""
    kb, root, repo, bare = world
    from hermes_cli import kanban_delivery_github as transport

    tid, first, card = _approved(kb, repo)
    _claim(kb, card, monkeypatch)
    branch = "delivery/" + tid
    assert _publish()["state"] == "created"
    second, newer = _rework(kb, repo, tid)
    _sql(kb, "UPDATE kanban_deliveries SET pull_request_state = 'returned_for_changes' "
             "WHERE integration_task_id = ?", card)
    _claim(kb, newer, monkeypatch)
    outside = _git(repo, "commit-tree", f"{second}^{{tree}}", "-p", second, "-m", "outside")
    push = transport.GitHubTransport.push  # the fixture's spy, which still pushes for real

    def push_then_outside(self, worktree, to, sha, *, expected):
        pushed = push(self, worktree, to, sha, expected=expected)
        _git(repo, "push", "-q", str(bare), f"{outside}:refs/heads/{to}")
        return pushed

    monkeypatch.setattr(transport.GitHubTransport, "push", push_then_outside)
    calls = len(github["calls"])

    _refused(_publish(), "head_moved")

    ref = f"/repos/{REPO}/git/ref/heads/{branch}"
    assert github["pushes"] == [(branch, first, ""), (branch, second, first)]
    assert _remote_head(bare, branch) == outside
    assert [(method, path) for method, path, _, _ in github["calls"][calls:]] == [
        ("GET", PULLS), ("GET", ref), ("GET", f"{PULLS}/41"), ("GET", ref)]
    assert _ledger(kb, newer) == (None, None, None, None) and _events(kb, newer) == []
    assert _ledger(kb, card) == (41, first, "returned_for_changes", branch)


@pytest.mark.parametrize("args", [
    {"head": "0" * 40}, {"pull_request_number": 7}, {"repository": "mtbitcr/other"},
    {"branch": "main"}, {"url": "https://github.com/mtbitcr/hermes-agent/pull/7"},
])
def test_publish_refuses_any_argument(world, github, monkeypatch, args):
    kb, root, repo, bare = world
    tid, head, card = _approved(kb, repo)
    _claim(kb, card, monkeypatch)

    _refused(_publish(args), "unknown_arguments")

    assert (github["calls"], github["token_requests"], github["pushes"]) == ([], [], [])
    assert _ledger(kb, card) == (None, None, None, None) and _events(kb, card) == []


def test_the_tool_is_listed_only_for_the_current_run_of_the_integration_card(world, github, monkeypatch):
    kb, root, repo, bare = world
    from agent.delegation_context import delegated_child_context

    tid, head, card = _approved(kb, repo)
    run_id = _claim(kb, card, monkeypatch)
    assert _listed() is True

    _write_config(root, enabled=False, profile=INTEGRATOR)
    assert _listed() is False
    _refused(_publish(), "delivery_disabled")
    _write_config(root, enabled=True, profile=INTEGRATOR)

    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(run_id + 1))
    assert _listed() is False
    _refused(_publish(), "not_current_run")
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(run_id))

    monkeypatch.setenv("HERMES_KANBAN_TASK", tid)
    assert _listed() is False
    _refused(_publish(), "not_integration_card")
    monkeypatch.setenv("HERMES_KANBAN_TASK", card)

    _git(repo, "remote", "set-url", "origin", "https://github.com/mtbitcr/not-in-policy.git")
    assert _listed() is False
    _refused(_publish(), "repository_not_in_policy")
    _git(repo, "remote", "set-url", "origin", f"https://github.com/{REPO}.git")

    with delegated_child_context():
        assert _listed() is False
        _refused(_publish(), "delegated_child")

    monkeypatch.delenv("HERMES_KANBAN_TASK")
    assert _listed() is False
    _refused(_publish(), "not_current_run")
    monkeypatch.setenv("HERMES_KANBAN_TASK", card)

    assert (github["calls"], github["token_requests"], github["pushes"]) == ([], [], [])
    assert _listed() is True


def test_the_tool_lives_only_in_the_kanban_delivery_toolset():
    _tool()
    from tools.registry import registry
    from toolsets import _HERMES_CORE_TOOLS, TOOLSETS, get_kernel_gated_toolsets, resolve_toolset

    assert TOOLSETS["kanban_delivery"]["tools"] == [TOOL]
    assert [name for name, spec in TOOLSETS.items() if TOOL in spec.get("tools", [])] == ["kanban_delivery"]
    assert registry.get_entry(TOOL).toolset == "kanban_delivery"
    assert TOOL not in _HERMES_CORE_TOOLS
    assert TOOL not in resolve_toolset("hermes-cli")
    # A worker's profile can name it: kernel-gated toolsets are stripped from every platform list.
    assert "kanban_delivery" not in get_kernel_gated_toolsets()
    assert registry.get_schema(TOOL)["parameters"] == {
        "type": "object", "properties": {}, "additionalProperties": False,
    }


def test_store_path_under_the_dispatcher_worker_environment(world, github, monkeypatch):
    kb, root, repo, bare = world
    kb.create_board("proj-a")
    kb.create_board("other")
    own_db = root / "kanban" / "boards" / "proj-a" / "kanban.db"
    other_db = root / "kanban" / "boards" / "other" / "kanban.db"
    # The integration profile lives under the root, with delivery on.
    profile_home = root / "profiles" / INTEGRATOR
    profile_home.mkdir(parents=True)
    _write_config(profile_home, enabled=True, profile=INTEGRATOR)

    conn = kb.connect(board="proj-a")
    try:
        tid, head, _ = _park(kb, conn, repo)
        assert _approve(kb, conn, tid) is True
        card = conn.execute(
            "SELECT integration_task_id FROM kanban_deliveries WHERE source_task_id = ?", (tid,),
        ).fetchone()[0]
        # The dispatcher's ready lane: claim, resolve the work area, then spawn through the live
        # _default_spawn.
        claimed = kb.claim_task(conn, card)
        assert claimed is not None and claimed.current_run_id is not None
        workspace = kb.resolve_workspace(claimed, board="proj-a")
        kb.set_workspace_path(conn, card, str(workspace))
        env = _spawned_worker_env(kb, claimed, workspace, "proj-a", monkeypatch)
    finally:
        conn.close()

    assert env["HERMES_KANBAN_TASK"] == card
    assert env["HERMES_KANBAN_RUN_ID"] == str(claimed.current_run_id)
    assert env["HERMES_KANBAN_DB"] == str(own_db)
    assert env["HERMES_KANBAN_BOARD"] == "proj-a"
    assert env["HERMES_HOME"] == str(profile_home)
    # Other App keys where a profile or board lookup would find them; the stand-in accepts the
    # root's key only.
    for home in (profile_home, own_db.parent):
        decoy = home / "secrets" / "github-app" / "raphael-agent-factory.pem"
        decoy.parent.mkdir(parents=True)
        decoy.write_bytes(_pem(_new_key()))

    _become(env, monkeypatch)
    branch = "delivery/" + tid
    assert _listed() is True
    assert _publish() == {"state": "created", "head": head, "pull_request_number": 41, "branch": branch}

    # From the worker environment the root registry and every board resolve.
    assert kb.kanban_home() == root
    assert kb.register_db_path() == root / "kanban" / "board_register.db"
    assert kb.get_register_entry("proj-a") is not None
    assert kb.get_register_entry("other") is not None
    assert {"default", "proj-a", "other"} <= {entry["slug"] for entry in kb.list_boards()}
    assert kb.board_dir("other") / "kanban.db" == other_db

    # The tool's answer, checked against an independent count: the board file read directly,
    # the stand-in's own record and the bare remote's ref.
    raw = sqlite3.connect(str(own_db))
    try:
        ledger = raw.execute(
            "SELECT integration_task_id, pull_request_number, pull_request_head, pull_request_state, "
            "pull_request_branch FROM kanban_deliveries WHERE source_task_id = ?", (tid,),
        ).fetchall()
        kinds = [row[0] for row in raw.execute(
            "SELECT kind FROM task_events WHERE kind LIKE 'delivery%' AND task_id = ? ORDER BY id", (card,),
        )]
    finally:
        raw.close()
    assert ledger == [(card, 41, head, "open", branch)]
    assert kinds == ["delivery_bound", "delivery_published"]
    assert [(p["number"], p["head_ref"], p["base_ref"]) for p in github["pulls"]] == [(41, branch, "main")]
    assert _remote_head(bare, branch) == head
    assert (len(github["token_requests"]), len(github["jwts"]), len(github["pushes"])) == (1, 1, 1)
    for path in (other_db, root / "kanban.db"):
        raw = sqlite3.connect(str(path))
        try:
            assert raw.execute("SELECT COUNT(*) FROM kanban_deliveries").fetchone()[0] == 0
            assert raw.execute("SELECT COUNT(*) FROM task_events WHERE kind LIKE 'delivery%'").fetchone()[0] == 0
        finally:
            raw.close()
    assert not (profile_home / "kanban.db").exists()
    assert not (profile_home / "kanban").exists()
