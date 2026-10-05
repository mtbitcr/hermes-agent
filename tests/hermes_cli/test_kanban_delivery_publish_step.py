"""Build card 1: ``dispatch_once`` publishes an approved build itself, after the tick lock is
released, on the board that was ticked, through one fenced step; no model, tool or card takes part.

Real SQLite boards, real git repositories and bare remotes, throwaway App keys, and a local
stand-in that plays GitHub; the only other stand-ins are those of the slice 3 tests.
"""

from __future__ import annotations

import http.client
import json
import sqlite3
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace
from urllib.parse import parse_qs

import jwt
import pytest

from tests.hermes_cli.test_kanban_delivery import (  # noqa: F401  (board is a fixture)
    IMPLEMENTER, REVIEWER, _approve, _become, _git, _park, _rework, _spawned_worker_env, _sql,
    _write_config, board,
)
from tests.hermes_cli.test_kanban_delivery_github import (
    APP_ID, REPO, TOKEN, _new_key, _pem, _remote_head,
)

PULLS = f"/repos/{REPO}/pulls"
BRANCHES = f"/repos/{REPO}/git/ref/heads/"
UNSTORED = (None,) * 4


@pytest.fixture
def github():
    """GitHub for one repository: the App's token, listing, creating and reading pull requests and
    reading a branch, whose head is what the bare remote holds. ``fail`` replaces one answer and
    ``always`` every answer of a route; a status of None drops the connection without an answer."""
    state = {"calls": [], "tokens": [], "pulls": [], "pushes": [], "on_create": None, "fail": {}, "always": {}}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def _reply(self, route, status, value):
            status, value = state["always"].get(route) or state["fail"].pop(route, (status, value))
            if status is None:
                return
            data = json.dumps(value).encode()
            self.send_response(status)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _pull(self, pull):
            head = _remote_head(state["bare"], pull["ref"])
            return {"number": pull["number"], "state": "open", "head": {"sha": head, "ref": pull["ref"]},
                    "base": {"ref": pull["base"]}}

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            if self.path.endswith("/access_tokens"):
                try:
                    jwt.decode(self.headers["Authorization"][7:], state["public"], algorithms=["RS256"],
                               issuer=APP_ID)
                except jwt.PyJWTError:
                    return self._reply(None, 401, {})
                state["tokens"].append(body)
                return self._reply(None, 201, {
                    "token": TOKEN, "permissions": body["permissions"],
                    "repositories": [{"full_name": f"mtbitcr/{n}"} for n in body["repositories"]]})
            state["calls"].append(("POST", self.path, body))
            if state["on_create"] is not None:
                state["on_create"]()
            if not _remote_head(state["bare"], body["head"]) or state["pulls"]:
                return self._reply(("POST", PULLS), 422, {})
            state["pulls"].append({"number": 41, "ref": body["head"], "base": body["base"]})
            return self._reply(("POST", PULLS), 201, self._pull(state["pulls"][0]))

        def do_GET(self):
            path, _, query = self.path.partition("?")
            state["calls"].append(("GET", path, {k: v[0] for k, v in parse_qs(query).items()}))
            if path == PULLS:
                return self._reply(("GET", path), 200, [self._pull(p) for p in state["pulls"]])
            if state["pulls"] and path == f"{PULLS}/41":
                return self._reply(("GET", path), 200, self._pull(state["pulls"][0]))
            sha = _remote_head(state["bare"], path.removeprefix(BRANCHES))
            if sha:
                return self._reply(("GET", BRANCHES), 200, {"object": {"type": "commit", "sha": sha}})
            return self._reply(("GET", BRANCHES), 404, {})

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    state["port"] = server.server_port
    yield state
    server.shutdown()
    server.server_close()


@pytest.fixture
def world(board, github, tmp_path, monkeypatch):
    """The board's repository with its GitHub origin, an empty bare remote laid out like GitHub, the
    App key under the root and delivery on. Every push runs for real; the spy only records it."""
    kb, root, repo = board
    from hermes_cli import kanban_delivery_github as transport

    _git(repo, "remote", "add", "origin", f"https://github.com/{REPO}.git")
    github["bare"] = tmp_path / "remotes" / "mtbitcr" / "hermes-agent.git"
    subprocess.run(["git", "init", "-q", "--bare", str(github["bare"])], check=True, capture_output=True)
    monkeypatch.setattr(transport, "_connect", lambda: http.client.HTTPConnection("127.0.0.1", github["port"]))
    monkeypatch.setattr(transport, "GIT_ORIGIN", (tmp_path / "remotes").as_uri())
    push = transport.GitHubTransport.push

    def spy(self, worktree, branch, sha, *, expected):
        github["pushes"].append((branch, sha, expected))
        return push(self, worktree, branch, sha, expected=expected)

    monkeypatch.setattr(transport.GitHubTransport, "push", spy)
    key = _new_key()
    (root / "secrets" / "github-app").mkdir(parents=True)
    (root / "secrets" / "github-app" / "raphael-agent-factory.pem").write_bytes(_pem(key))
    github["public"] = key.public_key()
    _write_config(root, enabled=True)
    return kb, root, repo, github["bare"]


def _approved(kb, repo, name: str = "feature", board=None):
    conn = kb.connect(board=board)
    try:
        tid, head, _ = _park(kb, conn, repo, name)
        assert _approve(kb, conn, tid) is True
    finally:
        conn.close()
    return tid, head


def _tick(kb, **options):
    """One dispatcher pass on the default board; nothing is ready, so only the delivery step works."""
    conn = kb.connect()
    try:
        return kb.dispatch_once(conn, spawn_fn=lambda *a, **k: 1, **options)
    finally:
        conn.close()


def _row(kb, head: str, db=None):
    """The ledger, then the lease and the refusal, of the delivery row of ``head``, from the file."""
    raw = sqlite3.connect(str(db or kb.kanban_db_path()))
    try:
        return raw.execute(
            "SELECT pull_request_number, pull_request_head, pull_request_state, pull_request_branch, "
            "publish_lease, publish_lease_until, publish_refusal FROM kanban_deliveries "
            "WHERE source_head = ?", (head,)).fetchone()
    finally:
        raw.close()


def _events(kb, tid: str, db=None) -> list:
    raw = sqlite3.connect(str(db or kb.kanban_db_path()))
    try:
        return [(kind, json.loads(payload)) for kind, payload in raw.execute(
            "SELECT kind, payload FROM task_events WHERE task_id = ? AND kind LIKE 'delivery%' "
            "ORDER BY id", (tid,))]
    finally:
        raw.close()


def _expire(kb):
    _sql(kb, "UPDATE kanban_deliveries SET publish_lease_until = ? WHERE publish_lease IS NOT NULL",
         int(time.time()) - 1)


def test_one_dispatcher_pass_publishes_an_approved_delivery(world, github):
    kb, root, repo, bare = world
    tid, head = _approved(kb, repo)
    base = _git(repo, "rev-parse", "main")
    approval = _sql(kb, "SELECT MAX(id) FROM task_events WHERE task_id = ? AND kind = 'completed'", tid)[0][0]
    branch, ref = "delivery/" + tid, f"/repos/{REPO}/git/ref/heads/delivery/{tid}"

    _tick(kb)

    assert _remote_head(bare, branch) == head and github["pushes"] == [(branch, head, "")]
    assert github["pulls"] == [{"number": 41, "ref": branch, "base": "main"}]
    assert [call[:2] for call in github["calls"]] == [
        ("GET", PULLS), ("GET", ref), ("POST", PULLS), ("GET", f"{PULLS}/41"), ("GET", ref)]
    listed, created = github["calls"][0][2], github["calls"][2][2]
    assert (listed["state"], listed["head"]) == ("open", f"mtbitcr:{branch}")
    assert (sorted(created), created["head"], created["base"]) == (["base", "body", "head", "title"], branch, "main")
    assert "implemented the slice" not in created["title"] + created["body"]
    # One fresh token, narrowed to the publish step and this one repository.
    assert github["tokens"] == [
        {"repositories": ["hermes-agent"], "permissions": {"contents": "write", "pull_requests": "write"}}]
    assert _row(kb, head) == (41, head, "open", branch, None, None, None)
    assert _events(kb, tid) == [
        ("delivery_bound", {"source_task_id": tid, "head": head, "base_commit": base, "reviewer": REVIEWER,
                            "implementer": IMPLEMENTER, "approval_event_id": approval, "base_is_ancestor": True}),
        ("delivery_published", {"repository": REPO, "branch": branch, "pull_request_number": 41,
                                "head": head, "state": "created"}),
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

    with kb._dispatch_tick_lock(kb.kanban_db_path()):  # the control: a held lock cannot be taken
        take_the_tick_lock()
    github["on_create"] = take_the_tick_lock
    _tick(kb)

    assert held == [False, True] and _row(kb, head)[0] == 41


def test_a_second_pass_makes_no_github_write_and_appends_no_event(world, github):
    kb, root, repo, bare = world
    tid, head = _approved(kb, repo)
    _tick(kb)
    before = (len(github["calls"]), len(github["tokens"]), len(github["pushes"]), _row(kb, head), _events(kb, tid))

    _tick(kb)
    _tick(kb)

    assert (len(github["calls"]), len(github["tokens"]), len(github["pushes"]), _row(kb, head),
            _events(kb, tid)) == before


@pytest.mark.parametrize("failure", ["crash", "creation-503", "readback-503", "readback-429", "readback-dropped"])
def test_a_crash_after_opening_the_pull_request_is_adopted_next_pass(world, github, monkeypatch, failure):
    """GitHub opened the pull request, but the pass died, GitHub's answer was a temporary one or
    none came: nothing is stored, nothing is parked, and the pass after the lease adopts it."""
    kb, root, repo, bare = world
    from hermes_cli import kanban_delivery_github as transport

    tid, head = _approved(kb, repo)
    request = transport.GitHubTransport.request

    def crash_after_the_creation(self, method, path, **options):
        answer = request(self, method, path, **options)
        if method == "POST":
            raise SystemExit("the dispatcher died")
        return answer

    if failure == "crash":
        monkeypatch.setattr(transport.GitHubTransport, "request", crash_after_the_creation)
        with pytest.raises(SystemExit):
            _tick(kb)
        monkeypatch.setattr(transport.GitHubTransport, "request", request)
    else:
        read, answer = failure.split("-")
        github["fail"] = {("POST", PULLS) if read == "creation" else ("GET", f"{PULLS}/41"):
                          (None, None) if answer == "dropped" else (int(answer), {})}
        _tick(kb)
    assert len(github["pulls"]) == 1 and _row(kb, head)[:4] == UNSTORED and _events(kb, tid) == []
    assert _row(kb, head)[6] is None
    calls = len(github["calls"])
    _tick(kb)  # the lease still holds, so the next pass leaves the row alone
    assert len(github["calls"]) == calls

    _expire(kb)
    _tick(kb)

    assert len(github["pulls"]) == 1 and github["pushes"] == [("delivery/" + tid, head, "")]
    assert _row(kb, head) == (41, head, "open", "delivery/" + tid, None, None, None)
    assert [(kind, payload.get("state")) for kind, payload in _events(kb, tid)] == [
        ("delivery_bound", None), ("delivery_published", "adopted")]


def test_a_pass_skips_a_row_whose_lease_is_held(world, github):
    """A held row is left alone, and a pass takes at most one lease: the oldest free row."""
    kb, root, repo, bare = world
    held, held_head = _approved(kb, repo, "alpha")
    free, free_head = _approved(kb, repo, "beta")
    later, later_head = _approved(kb, repo, "gamma")
    until = int(time.time()) + 600
    _sql(kb, "UPDATE kanban_deliveries SET publish_lease = 'another', publish_lease_until = ? "
             "WHERE source_task_id = ?", until, held)

    _tick(kb)

    assert [pull["ref"] for pull in github["pulls"]] == ["delivery/" + free]
    assert _row(kb, held_head) == (*UNSTORED, "another", until, None) and _events(kb, held) == []
    assert _row(kb, later_head) == (*UNSTORED, None, None, None)


def _take_over(kb, taken):
    from hermes_cli import kanban_delivery as kd

    _expire(kb)
    taken.append(kd._take_lease(kb.kanban_db_path()))


@pytest.mark.parametrize("case", [
    "taken-over-during-the-network-work", "taken-over-before-the-snapshot",
    "run-out-before-the-admission", "run-out-during-the-creation",
])
def test_an_expired_lease_is_taken_over_and_the_old_holder_stores_nothing(world, github, monkeypatch, case):
    kb, root, repo, bare = world
    from hermes_cli import kanban_delivery as kd

    tid, head = _approved(kb, repo)
    taken = []
    if case == "taken-over-during-the-network-work":
        github["on_create"] = lambda: _take_over(kb, taken)
    elif case == "taken-over-before-the-snapshot":
        # In its own thread, as another process would: on a board in rollback-journal mode the
        # takeover waits for the admission's read to end, so the creation waits for the takeover.
        other, prove = threading.Thread(target=_take_over, args=(kb, taken)), kd.approval_facts

        def take_over_then_prove(*args):
            if other.ident is None:
                other.start()
                other.join(timeout=1)
            return prove(*args)

        monkeypatch.setattr(kd, "approval_facts", take_over_then_prove)
        github["on_create"] = lambda: other.join(timeout=30)
    elif case == "run-out-during-the-creation":  # no takeover: only the clock passes the deadline
        clock = SimpleNamespace(time=time.time)
        monkeypatch.setattr(kd, "time", clock)
        github["on_create"] = lambda: setattr(clock, "time", lambda: time.time() + 601)
    if case == "run-out-before-the-admission":  # the holder is unchanged; only its deadline passed
        leased = kd._take_lease(kb.kanban_db_path())
        _expire(kb)
        with pytest.raises(kd.PublishRefused, match="lease_lost"):
            kd.publish_delivery(kb.kanban_db_path(), *leased)
        assert github["pushes"] == [] and github["pulls"] == []
    else:
        _tick(kb)
    assert _row(kb, head)[::6] == (None, None) and _events(kb, tid) == []
    assert not taken or _row(kb, head)[4] == taken[0][1]

    github["on_create"] = None
    _expire(kb)
    _tick(kb)

    assert len(github["pulls"]) == 1 and len(github["pushes"]) == 1
    assert _row(kb, head) == (41, head, "open", "delivery/" + tid, None, None, None)


def test_a_source_reopened_during_the_network_work_is_published_once_approved_again(world, github):
    """The owner reopens the source while GitHub creates the pull request: nothing is stored, the
    next pass refuses the approval that is gone, once. The card is built again on the same head and
    approved again, and the next pass publishes that head."""
    kb, root, repo, bare = world
    tid, head = _approved(kb, repo)

    def reopen():
        own = kb.connect()
        try:
            assert kb.cas_transition_task(
                own, tid, expected_status="done", expected_revision=kb.task_event_revision(own, tid),
                to_status="ready", event_kind="owner_move", event_payload={"to": "ready"})["moved"]
        finally:
            own.close()

    github["on_create"] = reopen
    _tick(kb)
    assert _row(kb, head)[:4] == UNSTORED and _row(kb, head)[6] is None and _events(kb, tid) == []
    _expire(kb)
    calls = len(github["calls"])
    _tick(kb)
    _tick(kb)
    assert len(github["calls"]) == calls  # refused before GitHub, and parked
    assert _row(kb, head)[4:] == (None, None, "not_approved")

    github["on_create"] = None
    assert _rework(kb, repo, tid, commit=False) == head
    _tick(kb)

    assert len(github["pulls"]) == 1 and len(github["pushes"]) == 1
    assert _row(kb, head) == (41, head, "open", "delivery/" + tid, None, None, None)
    assert [(kind, payload.get("code")) for kind, payload in _events(kb, tid)] == [
        ("delivery_refused", "not_approved"), ("delivery_bound", None), ("delivery_published", None)]


def test_the_step_uses_the_ticked_board_not_the_worker_board(world, github, monkeypatch):
    """The store-path test: this process has the environment the dispatcher gives a worker on proj-a
    (its board pinned, its card set, a profile home under the root) and ticks another board."""
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
        claimed = kb.claim_task(conn, kb.create_task(conn, title="plain work", assignee=IMPLEMENTER))
        workspace = kb.resolve_workspace(claimed, board="proj-a")
        kb.set_workspace_path(conn, claimed.id, str(workspace))
        env = _spawned_worker_env(kb, claimed, workspace, "proj-a", monkeypatch)
    finally:
        conn.close()
    assert (env["HERMES_KANBAN_TASK"], env["HERMES_KANBAN_DB"], env["HERMES_KANBAN_BOARD"], env["HERMES_HOME"]) == (
        claimed.id, str(own_db), "proj-a", str(profile_home))
    # App keys where a profile or board lookup would find them; the stand-in accepts the root's only.
    for home in (profile_home, own_db.parent, other_db.parent):
        (home / "secrets" / "github-app").mkdir(parents=True)
        (home / "secrets" / "github-app" / "raphael-agent-factory.pem").write_bytes(_pem(_new_key()))

    _become(env, monkeypatch)
    ticked = kb.connect(db_path=other_db)
    try:
        kb.dispatch_once(ticked, board="other", spawn_fn=lambda *a, **k: 1)
    finally:
        ticked.close()

    # From the worker environment the root registry and every board resolve.
    assert kb.kanban_home() == root and kb.register_db_path() == root / "kanban" / "board_register.db"
    assert {"default", "proj-a", "other"} <= {entry["slug"] for entry in kb.list_boards()}
    assert kb.board_dir("other") / "kanban.db" == other_db
    # The step's answer against an independent count: the board files, the stand-in and the remote.
    branch = "delivery/" + tid
    assert _row(kb, head, other_db) == (41, head, "open", branch, None, None, None)
    assert [kind for kind, _ in _events(kb, tid, other_db)] == ["delivery_bound", "delivery_published"]
    assert github["pulls"] == [{"number": 41, "ref": branch, "base": "main"}] and len(github["tokens"]) == 1
    assert (_remote_head(bare, branch), _remote_head(bare, "delivery/" + pinned_tid)) == (head, "")
    assert _row(kb, pinned_head, own_db) == (None,) * 7 and _events(kb, pinned_tid, own_db) == []
    raw = sqlite3.connect(str(root / "kanban.db"))
    assert raw.execute("SELECT COUNT(*) FROM kanban_deliveries").fetchone() == (0,)
    raw.close()
    assert not (profile_home / "kanban.db").exists() and not (profile_home / "kanban").exists()


def test_delivery_off_or_dry_run_or_skipped_tick_publishes_nothing(world, github):
    kb, root, repo, bare = world
    tid, head = _approved(kb, repo)

    with kb._dispatch_tick_lock(kb.kanban_db_path()):
        assert _tick(kb).skipped_locked is True
    assert _tick(kb, require_board_activation=True).skipped_inactive is True
    _tick(kb, dry_run=True)
    _write_config(root, enabled=False)
    _tick(kb)

    assert (github["calls"], github["tokens"], github["pushes"]) == ([], [], [])
    assert (_row(kb, head), _events(kb, tid)) == ((None,) * 7, [])
    _write_config(root, enabled=True)
    _tick(kb)
    assert _row(kb, head)[0] == 41


@pytest.mark.parametrize("case, code, calls", [
    ("foreign-commit-on-the-branch", "head_moved", 2),
    ("source-head-moved-before-github", "head_moved", 0),
    ("listing 200 {}", "pulls_unreadable", 1),
    ("listing 200 null", "bad_response", 1),
    ("listing 200 7", "bad_response", 1),
    ('listing 200 "open"', "bad_response", 1),
    ("listing 403 {}", "pulls_unreadable", 1),
    ('branch 200 {"object": {"type": "commit", "sha": "not-a-sha"}}', "branch_unreadable", 2),
    ("readback 200 null", "bad_response", 4),
])
def test_a_refusal_is_recorded_once_and_not_retried_every_tick(world, github, case, code, calls):
    """Any other ``case`` is the one answer GitHub gives to that read every time: a successful
    status with a body the step cannot use, or a 4xx. The readback follows the push and creation."""
    kb, root, repo, bare = world
    tid, head = _approved(kb, repo)
    if case == "foreign-commit-on-the-branch":
        base = _git(repo, "rev-parse", "main")
        foreign = _git(repo, "commit-tree", f"{base}^{{tree}}", "-p", base, "-m", "foreign")
        _git(repo, "push", "-q", str(bare), f"{foreign}:refs/heads/delivery/{tid}")
    elif case == "source-head-moved-before-github":
        outside = _git(repo, "commit-tree", f"{head}^{{tree}}", "-p", head, "-m", "outside")
        _sql(kb, "UPDATE tasks SET head_commit = ? WHERE id = ?", outside, tid)
    else:
        read, status, body = case.split(" ", 2)
        route = {"listing": ("GET", PULLS), "branch": ("GET", BRANCHES), "readback": ("GET", f"{PULLS}/41")}[read]
        github["always"] = {route: (int(status), json.loads(body))}

    _tick(kb)
    _expire(kb)
    _tick(kb)
    _tick(kb)

    writes = 1 if case.startswith("readback") else 0
    assert (len(github["calls"]), len(github["pushes"]), len(github["pulls"])) == (calls, writes, writes)
    assert _row(kb, head) == (*UNSTORED, None, None, code)
    assert [(kind, payload["code"], payload["head"]) for kind, payload in _events(kb, tid)] == [
        ("delivery_refused", code, head)]


@pytest.mark.parametrize("outside_push", [False, True])
def test_a_returned_head_fast_forwards_the_same_pull_request(world, github, monkeypatch, outside_push):
    """With ``outside_push``, someone pushes right after the step's own fast-forward: GitHub, read
    again, shows the branch elsewhere, so the step refuses head_moved and stores no success."""
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

    if outside_push:
        monkeypatch.setattr(transport.GitHubTransport, "push", push_then_outside)
    _tick(kb)

    assert github["pushes"] == [(branch, first, ""), (branch, second, first)] and len(github["pulls"]) == 1
    assert _row(kb, first)[:4] == (41, first, "returned_for_changes", branch)
    events = [(kind, payload["head"], payload.get("state")) for kind, payload in _events(kb, tid)][2:]
    if outside_push:
        assert _remote_head(bare, branch) == outside
        assert _row(kb, second) == (*UNSTORED, None, None, "head_moved")
        assert events == [("delivery_refused", second, None)]
    else:
        assert _remote_head(bare, branch) == second
        assert _row(kb, second) == (41, second, "open", branch, None, None, None)
        assert events == [("delivery_bound", second, None), ("delivery_published", second, "fast_forwarded")]


def test_approving_an_older_head_again_re_arms_its_row_and_returns_the_newer_one(world, github):
    """H1 is approved, then H2, then H1 again, before any publication: H1's row is pending again
    and published, H2's row is returned for changes and never published, and a lease taken on
    either row before the last approval stores nothing."""
    kb, root, repo, bare = world
    from hermes_cli import kanban_delivery as kd

    tid, first = _approved(kb, repo)
    old = kd._take_lease(kb.kanban_db_path())
    second = _rework(kb, repo, tid)
    newer = kd._take_lease(kb.kanban_db_path())
    assert _rework(kb, repo, tid, restore=first) == first

    assert (_row(kb, first), _row(kb, second)[2]) == ((None,) * 7, "returned_for_changes")
    for leased, code in ((old, "lease_lost"), (newer, "superseded")):
        with pytest.raises(kd.PublishRefused, match=code):
            kd.publish_delivery(kb.kanban_db_path(), *leased)
    assert github["calls"] == []
    _tick(kb)
    _expire(kb)
    _tick(kb)

    branch = "delivery/" + tid
    assert github["pushes"] == [(branch, first, "")] and len(github["pulls"]) == 1
    assert _row(kb, first) == (41, first, "open", branch, None, None, None)
    assert _row(kb, second)[:4] == (None, None, "returned_for_changes", None) and _row(kb, second)[6] is None
