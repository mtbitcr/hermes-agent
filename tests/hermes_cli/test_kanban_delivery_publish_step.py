"""Build card 1: the dispatcher publishes an approved build itself.

``dispatch_once`` runs one delivery step after the board's tick lock is
released, on the board that was ticked: it leases at most one approved delivery
row, publishes its head as the delivery's one pull request and records the
events on the source card. No model, tool or card takes part.

Everything runs on a real SQLite board, a real git repository and a real bare
remote laid out like GitHub. A local stand-in plays GitHub, as the slice 2
fixture of tests/hermes_cli/test_kanban_delivery_github.py does, and here also
lists and creates pull requests. The only other stand-ins are the reviewer the
model policy nominates and the final process launch of the dispatcher's spawn,
both as in the slice 3 tests.
"""

from __future__ import annotations

import http.client
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
    REVIEWER,
    _approve,
    _become,
    _git,
    _park,
    _rework,
    _spawned_worker_env,
    _sql,
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

PULLS = f"/repos/{REPO}/pulls"
NOTHING = (None, None, None, None)


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
    _write_config(root, enabled=True)
    return kb, root, repo, bare


def _approved(kb, repo, name: str = "feature", board=None):
    """A built and approved source card; its approval recorded one delivery row."""
    conn = kb.connect(board=board)
    try:
        tid, head, _ = _park(kb, conn, repo, name)
        assert _approve(kb, conn, tid) is True
    finally:
        conn.close()
    return tid, head


def _tick(kb, **options):
    """One dispatcher pass on the default board. Nothing is ready, so only the delivery step works."""
    conn = kb.connect()
    try:
        return kb.dispatch_once(conn, spawn_fn=lambda *a, **k: 1, **options)
    finally:
        conn.close()


def _row(kb, head: str, columns: str, db=None):
    raw = sqlite3.connect(str(db or kb.kanban_db_path()))
    try:
        return raw.execute(f"SELECT {columns} FROM kanban_deliveries WHERE source_head = ?", (head,)).fetchone()
    finally:
        raw.close()


def _ledger(kb, head: str, db=None):
    return _row(kb, head, "pull_request_number, pull_request_head, pull_request_state, pull_request_branch", db)


def _lease(kb, head: str, db=None):
    return _row(kb, head, "publish_lease, publish_lease_until, publish_refusal", db)


def _events(kb, tid: str, db=None) -> list:
    raw = sqlite3.connect(str(db or kb.kanban_db_path()))
    try:
        rows = raw.execute(
            "SELECT kind, payload, run_id FROM task_events WHERE task_id = ? AND kind LIKE 'delivery%' "
            "ORDER BY id", (tid,),
        ).fetchall()
    finally:
        raw.close()
    return [(kind, json.loads(payload), run_id) for kind, payload, run_id in rows]


def _nothing_touched(github) -> bool:
    return (github["calls"], github["token_requests"], github["pushes"]) == ([], [], [])


def test_one_dispatcher_pass_publishes_an_approved_delivery(world, github):
    kb, root, repo, bare = world
    tid, head = _approved(kb, repo)
    base = _git(repo, "rev-parse", "main")
    approval = _sql(kb, "SELECT MAX(id) FROM task_events WHERE task_id = ? AND kind = 'completed'", tid)[0][0]
    branch = "delivery/" + tid

    _tick(kb)

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
    assert _ledger(kb, head) == (41, head, "open", branch)
    assert _lease(kb, head) == (None, None, None)
    assert _events(kb, tid) == [
        ("delivery_bound", {
            "source_task_id": tid, "head": head, "base_commit": base, "reviewer": REVIEWER,
            "implementer": IMPLEMENTER, "approval_event_id": approval, "base_is_ancestor": True,
        }, None),
        ("delivery_published", {
            "repository": REPO, "branch": branch, "pull_request_number": 41, "head": head,
            "state": "created",
        }, None),
    ]
    # No card took part, and the token reached no board row.
    assert _sql(kb, "SELECT COUNT(*) FROM tasks") == [(1,)]
    assert [row for row in _sql(kb, "SELECT payload FROM task_events") if TOKEN in str(row)] == []


def test_publish_runs_after_the_tick_lock_is_released(world, github):
    kb, root, repo, bare = world
    tid, head = _approved(kb, repo)
    held = []

    def take_the_tick_lock():
        with kb._dispatch_tick_lock(kb.kanban_db_path()) as acquired:
            held.append(acquired)

    # The control: while a tick holds the lock, nobody else can take it.
    with kb._dispatch_tick_lock(kb.kanban_db_path()) as outer:
        assert outer is True
        take_the_tick_lock()
    assert held == [False]

    github["on_create"] = take_the_tick_lock
    _tick(kb)

    assert held == [False, True]
    assert _ledger(kb, head)[0] == 41


def test_a_second_pass_makes_no_github_write_and_appends_no_event(world, github):
    kb, root, repo, bare = world
    tid, head = _approved(kb, repo)
    _tick(kb)
    calls, minted, pushes = len(github["calls"]), len(github["token_requests"]), list(github["pushes"])
    ledger, events = _ledger(kb, head), _events(kb, tid)

    _tick(kb)
    _tick(kb)

    assert (len(github["calls"]), len(github["token_requests"]), github["pushes"]) == (calls, minted, pushes)
    assert len(github["pulls"]) == 1
    assert (_ledger(kb, head), _events(kb, tid)) == (ledger, events)


def test_a_crash_after_opening_the_pull_request_is_adopted_next_pass(world, github, monkeypatch):
    kb, root, repo, bare = world
    from hermes_cli import kanban_delivery_github as transport

    tid, head = _approved(kb, repo)
    branch = "delivery/" + tid
    request = transport.GitHubTransport.request

    def crash_after_the_creation(self, method, path, **options):
        answer = request(self, method, path, **options)
        if method == "POST":
            raise SystemExit("the dispatcher died")
        return answer

    monkeypatch.setattr(transport.GitHubTransport, "request", crash_after_the_creation)
    with pytest.raises(SystemExit):
        _tick(kb)
    monkeypatch.setattr(transport.GitHubTransport, "request", request)
    assert [(p["number"], p["head_ref"]) for p in github["pulls"]] == [(41, branch)]
    assert _ledger(kb, head) == NOTHING and _events(kb, tid) == []

    # The dead holder's lease still holds the row, so the next pass leaves it alone.
    calls = len(github["calls"])
    _tick(kb)
    assert len(github["calls"]) == calls

    # Once it runs out, the next pass adopts the one pull request at H and opens no second one.
    _sql(kb, "UPDATE kanban_deliveries SET publish_lease_until = ?", int(time.time()) - 1)
    _tick(kb)

    assert [p["number"] for p in github["pulls"]] == [41]
    assert github["pushes"] == [(branch, head, "")]
    assert _ledger(kb, head) == (41, head, "open", branch)
    assert [(kind, payload.get("state")) for kind, payload, _ in _events(kb, tid)] == [
        ("delivery_bound", None), ("delivery_published", "adopted"),
    ]


def test_a_pass_skips_a_row_whose_lease_is_held(world, github):
    kb, root, repo, bare = world
    tid, head = _approved(kb, repo)
    until = int(time.time()) + 600
    _sql(kb, "UPDATE kanban_deliveries SET publish_lease = 'another-step', publish_lease_until = ?", until)

    _tick(kb)

    assert _nothing_touched(github)
    assert _lease(kb, head) == ("another-step", until, None)
    assert _ledger(kb, head) == NOTHING and _events(kb, tid) == []

    # Once that lease runs out, the next pass takes the row and publishes it.
    _sql(kb, "UPDATE kanban_deliveries SET publish_lease_until = ?", int(time.time()) - 1)
    _tick(kb)
    assert _ledger(kb, head) == (41, head, "open", "delivery/" + tid)
    assert _lease(kb, head) == (None, None, None)


def test_an_expired_lease_is_taken_over_and_the_old_holder_stores_nothing(world, github):
    """The step's lease runs out while GitHub is creating the pull request, and another dispatcher
    takes the row over. The old holder stores nothing; the new one adopts the pull request."""
    kb, root, repo, bare = world
    from hermes_cli import kanban_delivery as kd

    tid, head = _approved(kb, repo)
    taken = []

    def run_out_and_take_over():
        _sql(kb, "UPDATE kanban_deliveries SET publish_lease_until = ?", int(time.time()) - 1)
        taken.append(kd._take_lease(kb.kanban_db_path()))

    github["on_create"] = run_out_and_take_over
    _tick(kb)

    delivery_id, holder = taken[0]
    assert _lease(kb, head)[::2] == (holder, None)
    assert _ledger(kb, head) == NOTHING and _events(kb, tid) == []
    assert len(github["pulls"]) == 1

    github["on_create"] = None
    published = kd.publish_delivery(kb.kanban_db_path(), delivery_id, holder)

    assert (published["state"], published["pull_request_number"]) == ("adopted", 41)
    assert len(github["pulls"]) == 1 and len(github["pushes"]) == 1
    assert _ledger(kb, head) == (41, head, "open", "delivery/" + tid)
    assert _lease(kb, head) == (None, None, None)


def test_a_takeover_between_the_admission_and_the_snapshot_stores_nothing(world, github, monkeypatch):
    """Ported from the PR 145 security review: the row is taken over after the step is admitted but
    before its facts are captured. The admission and the snapshot are one read, so the snapshot
    still holds the admitted lease, and the final write sees the takeover. The takeover runs in its
    own thread, as another process would: on a board in rollback-journal mode it waits for that
    read to end, so the pull request creation waits for the takeover to land."""
    kb, root, repo, bare = world
    from hermes_cli import kanban_delivery as kd

    tid, head = _approved(kb, repo)
    original = kd.approval_facts
    taken = []

    def take_over():
        _sql(kb, "UPDATE kanban_deliveries SET publish_lease_until = ?", int(time.time()) - 1)
        taken.append(kd._take_lease(kb.kanban_db_path()))

    other = threading.Thread(target=take_over)

    def take_over_then_prove(conn, task_id, bound_head):
        if other.ident is None:
            other.start()
            other.join(timeout=1)  # at once, unless the board's journal makes it wait for the read
        return original(conn, task_id, bound_head)

    monkeypatch.setattr(kd, "approval_facts", take_over_then_prove)
    github["on_create"] = lambda: other.join(timeout=30)
    _tick(kb)

    assert len(taken) == 1 and taken[0] is not None
    assert _lease(kb, head)[::2] == (taken[0][1], None)
    assert _ledger(kb, head) == NOTHING and _events(kb, tid) == []


def test_a_source_reopened_during_the_network_work_stores_nothing(world, github):
    """Ported from finding 2 of the slice 4 review: the owner reopens the source card while GitHub
    is creating the pull request. The final write stores nothing and parks nothing; the next pass
    refuses the approval that is gone, and records that once."""
    kb, root, repo, bare = world
    tid, head = _approved(kb, repo)
    reopened = []

    def reopen():
        own = kb.connect()
        try:
            reopened.append(kb.cas_transition_task(
                own, tid, expected_status="done", expected_revision=kb.task_event_revision(own, tid),
                to_status="ready", event_kind="owner_move", event_payload={"to": "ready"},
            )["moved"])
        finally:
            own.close()

    github["on_create"] = reopen
    _tick(kb)

    assert reopened == [True]
    assert len(github["pulls"]) == 1 and len(github["pushes"]) == 1  # GitHub changed; the board did not
    assert _ledger(kb, head) == NOTHING and _events(kb, tid) == []
    assert _lease(kb, head)[2] is None

    _sql(kb, "UPDATE kanban_deliveries SET publish_lease_until = ?", int(time.time()) - 1)
    calls = len(github["calls"])
    _tick(kb)
    _tick(kb)

    assert len(github["calls"]) == calls  # refused before GitHub, and parked
    assert _lease(kb, head) == (None, None, "not_approved")
    assert [(kind, payload["code"]) for kind, payload, _ in _events(kb, tid)] == [
        ("delivery_refused", "not_approved"),
    ]


def test_the_step_uses_the_ticked_board_not_the_worker_board(world, github, monkeypatch):
    """The store-path test: this process has exactly the environment the dispatcher gives a worker
    on proj-a (its board pinned, its card set, a profile home under the root), and ticks another
    board. The step publishes the ticked board's delivery, still reads the root registry and the
    root's App key, and leaves the pinned board alone."""
    kb, root, repo, bare = world
    kb.create_board("proj-a")
    kb.create_board("other")
    own_db = root / "kanban" / "boards" / "proj-a" / "kanban.db"
    other_db = root / "kanban" / "boards" / "other" / "kanban.db"
    profile_home = root / "profiles" / IMPLEMENTER
    profile_home.mkdir(parents=True)
    _write_config(profile_home, enabled=True)

    pinned_tid, pinned_head = _approved(kb, repo, "alpha", board="proj-a")
    tid, head = _approved(kb, repo, "beta", board="other")
    conn = kb.connect(board="proj-a")
    try:
        plain = kb.create_task(conn, title="plain work", assignee=IMPLEMENTER)
        claimed = kb.claim_task(conn, plain)
        assert claimed is not None and claimed.current_run_id is not None
        workspace = kb.resolve_workspace(claimed, board="proj-a")
        kb.set_workspace_path(conn, plain, str(workspace))
        env = _spawned_worker_env(kb, claimed, workspace, "proj-a", monkeypatch)
    finally:
        conn.close()

    assert env["HERMES_KANBAN_TASK"] == plain
    assert env["HERMES_KANBAN_DB"] == str(own_db)
    assert env["HERMES_KANBAN_BOARD"] == "proj-a"
    assert env["HERMES_HOME"] == str(profile_home)
    # Other App keys where a profile or board lookup would find them; the stand-in accepts the
    # root's key only.
    for home in (profile_home, own_db.parent, other_db.parent):
        decoy = home / "secrets" / "github-app" / "raphael-agent-factory.pem"
        decoy.parent.mkdir(parents=True)
        decoy.write_bytes(_pem(_new_key()))

    _become(env, monkeypatch)
    ticked = kb.connect(db_path=other_db)
    try:
        kb.dispatch_once(ticked, board="other", spawn_fn=lambda *a, **k: 1)
    finally:
        ticked.close()

    # From the worker environment the root registry and every board resolve.
    assert kb.kanban_home() == root
    assert kb.register_db_path() == root / "kanban" / "board_register.db"
    assert kb.get_register_entry("proj-a") is not None
    assert kb.get_register_entry("other") is not None
    assert {"default", "proj-a", "other"} <= {entry["slug"] for entry in kb.list_boards()}
    assert kb.board_dir("other") / "kanban.db" == other_db

    # The step's answer, checked against an independent count: the board files read directly,
    # the stand-in's own record and the bare remote's refs.
    branch = "delivery/" + tid
    assert _ledger(kb, head, other_db) == (41, head, "open", branch)
    assert [kind for kind, _, _ in _events(kb, tid, other_db)] == ["delivery_bound", "delivery_published"]
    assert [(p["number"], p["head_ref"], p["base_ref"]) for p in github["pulls"]] == [(41, branch, "main")]
    assert _remote_head(bare, branch) == head
    assert _remote_head(bare, "delivery/" + pinned_tid) == ""
    assert (len(github["token_requests"]), len(github["jwts"]), len(github["pushes"])) == (1, 1, 1)
    assert (_ledger(kb, pinned_head, own_db), _lease(kb, pinned_head, own_db)) == (NOTHING, (None, None, None))
    assert _events(kb, pinned_tid, own_db) == []
    raw = sqlite3.connect(str(root / "kanban.db"))
    try:
        assert raw.execute("SELECT COUNT(*) FROM kanban_deliveries").fetchone()[0] == 0
        assert raw.execute("SELECT COUNT(*) FROM task_events WHERE kind LIKE 'delivery%'").fetchone()[0] == 0
    finally:
        raw.close()
    assert not (profile_home / "kanban.db").exists()
    assert not (profile_home / "kanban").exists()


def test_delivery_off_or_dry_run_or_skipped_tick_publishes_nothing(world, github):
    kb, root, repo, bare = world
    tid, head = _approved(kb, repo)

    with kb._dispatch_tick_lock(kb.kanban_db_path()) as held:
        assert held is True
        assert _tick(kb).skipped_locked is True
    assert _tick(kb, require_board_activation=True).skipped_inactive is True
    _tick(kb, dry_run=True)
    _write_config(root, enabled=False)
    _tick(kb)

    assert _nothing_touched(github)
    assert (_ledger(kb, head), _lease(kb, head), _events(kb, tid)) == (NOTHING, (None, None, None), [])

    _write_config(root, enabled=True)
    _tick(kb)
    assert _ledger(kb, head) == (41, head, "open", "delivery/" + tid)


def test_a_refusal_is_recorded_once_and_not_retried_every_tick(world, github):
    """The branch already holds a commit the delivery did not push: refused before any push, the
    row is parked and the refusal is recorded once on the source card."""
    kb, root, repo, bare = world
    tid, head = _approved(kb, repo)
    branch = "delivery/" + tid
    base = _git(repo, "rev-parse", "main")
    foreign = _git(repo, "commit-tree", f"{base}^{{tree}}", "-p", base, "-m", "foreign")
    _git(repo, "push", "-q", str(bare), f"{foreign}:refs/heads/{branch}")

    _tick(kb)
    calls = len(github["calls"])
    _tick(kb)
    _tick(kb)

    assert len(github["calls"]) == calls
    assert github["pushes"] == [] and github["pulls"] == []
    assert _remote_head(bare, branch) == foreign
    assert _ledger(kb, head) == NOTHING
    assert _lease(kb, head) == (None, None, "head_moved")
    assert _events(kb, tid) == [("delivery_refused", {
        "delivery_id": 1, "head": head, "code": "head_moved",
        "detail": "the branch holds a head other than H",
    }, None)]


def test_a_transport_failure_is_retried_once_the_lease_runs_out(world, github, monkeypatch):
    kb, root, repo, bare = world
    tid, head = _approved(kb, repo)
    key_file = root / "secrets" / "github-app" / "raphael-agent-factory.pem"
    key = key_file.read_bytes()
    key_file.unlink()

    _tick(kb)

    assert github["pulls"] == [] and _events(kb, tid) == []
    assert _lease(kb, head)[2] is None
    key_file.write_bytes(key)
    _sql(kb, "UPDATE kanban_deliveries SET publish_lease_until = ?", int(time.time()) - 1)
    _tick(kb)
    assert _ledger(kb, head)[0] == 41


def test_a_returned_head_fast_forwards_the_same_pull_request(world, github):
    kb, root, repo, bare = world
    tid, first = _approved(kb, repo)
    branch = "delivery/" + tid
    _tick(kb)

    second = _rework(kb, repo, tid)
    assert _ledger(kb, first) == (41, first, "returned_for_changes", branch)
    _tick(kb)

    assert github["pushes"] == [(branch, first, ""), (branch, second, first)]
    assert _remote_head(bare, branch) == second
    assert [p["number"] for p in github["pulls"]] == [41]
    assert _ledger(kb, first) == (41, first, "returned_for_changes", branch)
    assert _ledger(kb, second) == (41, second, "open", branch)
    assert [(kind, payload["head"], payload.get("state")) for kind, payload, _ in _events(kb, tid)] == [
        ("delivery_bound", first, None), ("delivery_published", first, "created"),
        ("delivery_bound", second, None), ("delivery_published", second, "fast_forwarded"),
    ]


def test_a_source_head_that_moved_is_refused_before_github(world, github):
    """Ported from section 8 test 1: the source card's head is no longer the reviewed and bound
    head. Refused before any GitHub call or token, and parked."""
    kb, root, repo, bare = world
    tid, head = _approved(kb, repo)
    outside = _git(repo, "commit-tree", f"{head}^{{tree}}", "-p", head, "-m", "outside")
    _sql(kb, "UPDATE tasks SET head_commit = ? WHERE id = ?", outside, tid)

    _tick(kb)

    assert _nothing_touched(github)
    assert _lease(kb, head) == (None, None, "head_moved")
    assert [(kind, payload["code"]) for kind, payload, _ in _events(kb, tid)] == [("delivery_refused", "head_moved")]


def test_an_outside_push_right_after_the_fast_forward_push_stores_no_success(world, github, monkeypatch):
    """Ported from finding 3 of the slice 4 review: someone pushes to the branch right after the
    step's own leased fast-forward. GitHub, read again after the push, shows the pull request and
    the branch at that commit rather than at H, so the step refuses head_moved and stores no
    success."""
    kb, root, repo, bare = world
    from hermes_cli import kanban_delivery_github as transport

    tid, first = _approved(kb, repo)
    branch = "delivery/" + tid
    _tick(kb)
    second = _rework(kb, repo, tid)
    outside = _git(repo, "commit-tree", f"{second}^{{tree}}", "-p", second, "-m", "outside")
    push = transport.GitHubTransport.push  # the fixture's spy, which still pushes for real

    def push_then_outside(self, worktree, to, sha, *, expected):
        pushed = push(self, worktree, to, sha, expected=expected)
        _git(repo, "push", "-q", str(bare), f"{outside}:refs/heads/{to}")
        return pushed

    monkeypatch.setattr(transport.GitHubTransport, "push", push_then_outside)
    calls = len(github["calls"])

    _tick(kb)

    ref = f"/repos/{REPO}/git/ref/heads/{branch}"
    assert github["pushes"] == [(branch, first, ""), (branch, second, first)]
    assert _remote_head(bare, branch) == outside
    assert [(method, path) for method, path, _, _ in github["calls"][calls:]] == [
        ("GET", PULLS), ("GET", ref), ("GET", f"{PULLS}/41"), ("GET", ref)]
    assert _ledger(kb, second) == NOTHING
    assert _lease(kb, second) == (None, None, "head_moved")
    assert _ledger(kb, first) == (41, first, "returned_for_changes", branch)
    assert [(kind, payload["head"]) for kind, payload, _ in _events(kb, tid)][2:] == [("delivery_refused", second)]


def test_a_tick_takes_at_most_one_lease(world, github):
    kb, root, repo, bare = world
    first, first_head = _approved(kb, repo, "alpha")
    second, second_head = _approved(kb, repo, "beta")

    _tick(kb)

    assert [p["head_ref"] for p in github["pulls"]] == ["delivery/" + first]
    assert _lease(kb, second_head) == (None, None, None) and _ledger(kb, second_head) == NOTHING

    _tick(kb)
    assert [p["head_ref"] for p in github["pulls"]] == ["delivery/" + first, "delivery/" + second]
    assert _ledger(kb, first_head)[0] == 41 and _ledger(kb, second_head)[0] == 42
