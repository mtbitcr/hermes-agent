"""Card 1: a new card's branch starts at the current main, not at the bound checkout's HEAD.

With ``kanban.delivery.exact_base`` exactly true, a new card branch in a repository of the delivery
policy starts at the main GitHub holds now. The kernel fetches that one ref through the publish
step's fenced transport into the bound repository's remote-tracking ref only, creates the branch
there and records that commit as the card's base. A failed fetch starts the branch at HEAD as today
and records why, after the base. Real SQLite boards, real git repositories and a bare remote laid out
like GitHub, a throwaway App key and the local stand-in that plays GitHub's token call; the only other
stand-in is the final process launch of the dispatcher's worker spawn.
"""

from __future__ import annotations

import http.client
import json
import os
import shutil
import sqlite3
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.hermes_cli.test_kanban_delivery import (  # noqa: F401  (board is a fixture)
    IMPLEMENTER, _become, _git, _spawned_worker_env, board,
)
from tests.hermes_cli.test_kanban_delivery_github import (
    BASIC, REPO, TOKEN, _capture_all_logs, _carries, _files_text, _log_text, _new_key, _pem, _record_git,
    _record_mkdtemp,
)
from tests.hermes_cli.test_kanban_delivery_publish_step import github  # noqa: F401  (a fixture)

# The publish step's own token request, the only shape there is: its permissions, this one repository.
PUBLISH = {"repositories": ["hermes-agent"], "permissions": {"contents": "write", "pull_requests": "write"}}
TRACKING = "refs/remotes/origin/main"


def _setting(home: Path, exact_base: str | None) -> None:
    """``kanban.delivery`` as a config file spells it: delivery on, and ``exact_base`` as written."""
    lines = ["kanban:", "  delivery:", "    enabled: true"]
    if exact_base is not None:
        lines.append(f"    exact_base: {exact_base}")
    (home / "config.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")


@pytest.fixture
def world(board, github, tmp_path, monkeypatch):
    """The board's repository with its GitHub origin, still at the commit it started from; a bare
    remote laid out like GitHub whose main has moved on, beside a branch and a tag of its own; the
    App key under the root, the implementer's profile, and ``exact_base`` on."""
    kb, root, repo = board
    from hermes_cli import kanban_delivery_github as transport

    head = _git(repo, "rev-parse", "HEAD")
    _git(repo, "remote", "add", "origin", f"https://github.com/{REPO}.git")
    bare = tmp_path / "remotes" / "mtbitcr" / "hermes-agent.git"
    subprocess.run(["git", "init", "-q", "--bare", str(bare)], check=True, capture_output=True)
    _git(repo, "push", "-q", str(bare), "main")
    upstream = tmp_path / "upstream"
    subprocess.run(["git", "clone", "-q", "-b", "main", str(bare), str(upstream)], check=True, capture_output=True)
    (upstream / "MAIN.md").write_text("merged after the checkout\n", encoding="utf-8")
    _git(upstream, "add", "MAIN.md")
    _git(upstream, "commit", "-q", "-m", "merged on main")
    main = _git(upstream, "rev-parse", "HEAD")
    _git(upstream, "push", "-q", "origin", "main")
    _git(upstream, "commit", "-q", "--allow-empty", "-m", "elsewhere")
    _git(upstream, "tag", "v9")
    _git(upstream, "push", "-q", "origin", "HEAD:refs/heads/elsewhere", "v9")

    monkeypatch.setattr(transport, "_connect", lambda: http.client.HTTPConnection("127.0.0.1", github["port"], timeout=10))
    monkeypatch.setattr(transport, "GIT_ORIGIN", (tmp_path / "remotes").as_uri())
    key = _new_key()
    (root / "secrets" / "github-app").mkdir(parents=True)
    (root / "secrets" / "github-app" / "raphael-agent-factory.pem").write_bytes(_pem(key))
    github["public"], github["bare"] = key.public_key(), bare
    (root / "profiles" / IMPLEMENTER).mkdir(parents=True)  # a real profile, so the dispatcher spawns its cards
    _setting(root, "true")
    return SimpleNamespace(kb=kb, root=root, repo=repo, bare=bare, head=head, main=main, tmp=tmp_path)


def _new_card(kb, repo: Path, board: str | None = None, **fields) -> str:
    conn = kb.connect(board=board)
    try:
        return kb.create_task(conn, title="build a slice", assignee=IMPLEMENTER, workspace_kind="worktree",
                              workspace_path=str(repo), owned_paths=["src/slice"], **fields)
    finally:
        conn.close()


def _dispatch(kb):
    """One dispatcher pass on the default board; only the worker's process launch is a stand-in."""
    conn = kb.connect()
    try:
        return kb.dispatch_once(conn, spawn_fn=lambda *a, **k: 1)
    finally:
        conn.close()


def _task(kb, tid: str):
    conn = kb.connect()
    try:
        return kb.get_task(conn, tid)
    finally:
        conn.close()


def _base_events(db: Path, tid: str) -> list:
    """The card's base events, read from the board file itself."""
    raw = sqlite3.connect(str(db))
    try:
        return [(kind, json.loads(payload)) for kind, payload in raw.execute(
            "SELECT kind, payload FROM task_events WHERE task_id = ? AND kind LIKE 'worktree_base%' ORDER BY id",
            (tid,))]
    finally:
        raw.close()


def _refs(repo: Path) -> dict:
    """Every ref of the bound repository and the commit it holds, read from the repository."""
    listed = _git(repo, "for-each-ref", "--format=%(refname) %(objectname)")
    return dict(line.split(" ") for line in listed.splitlines())


def _changed(before: dict, after: dict) -> dict:
    assert set(before) <= set(after), "a ref was removed"
    return {name: sha for name, sha in after.items() if before.get(name) != sha}


def _checkout(repo: Path) -> dict:
    """The bound checkout as its owner sees it: HEAD, the branch it names, the index, the status and
    every file of the working tree with its bytes, the card worktrees under .worktrees aside."""
    files = {}
    for path in repo.rglob("*"):
        parts = path.relative_to(repo).parts
        if parts[0] not in (".git", ".worktrees") and path.is_file():
            files["/".join(parts)] = path.read_bytes()
    status = [line for line in _git(repo, "status", "--porcelain").splitlines() if ".worktrees" not in line]
    return {"head": _git(repo, "rev-parse", "HEAD"), "branch": _git(repo, "symbolic-ref", "HEAD"),
            "index": _git(repo, "ls-files", "--stage"), "status": status, "files": files}


def test_a_new_card_starts_at_current_main_not_the_checkout_head(world, github):
    kb, repo = world.kb, world.repo
    tid = _new_card(kb, repo)

    result = _dispatch(kb)

    task = _task(kb, tid)
    assert tid in [spawned[0] for spawned in result.spawned]
    assert world.main != world.head and task.base_commit == world.main
    workspace = Path(task.workspace_path)
    assert workspace == repo / ".worktrees" / tid
    assert _git(workspace, "rev-parse", "HEAD") == world.main
    assert _git(workspace, "symbolic-ref", "HEAD") == f"refs/heads/{task.branch_name}"
    assert (workspace / "MAIN.md").read_text(encoding="utf-8") == "merged after the checkout\n"
    assert _base_events(kb.kanban_db_path(), tid) == [("worktree_base_recorded", {"base_commit": world.main})]
    assert github["tokens"] == [PUBLISH]


def test_the_bound_checkout_head_and_tree_stay_unchanged_after_the_fetch(world, github):
    kb, repo = world.kb, world.repo
    (repo / "README.md").write_text("base, with the owner's edit\n", encoding="utf-8")
    (repo / "notes.txt").write_text("not tracked\n", encoding="utf-8")
    checkout, refs = _checkout(repo), _refs(repo)
    tid = _new_card(kb, repo)

    _dispatch(kb)

    task = _task(kb, tid)
    assert task.base_commit == world.main
    assert _checkout(repo) == checkout
    assert not (repo / "MAIN.md").exists() and not (repo / ".git" / "FETCH_HEAD").exists()
    # Beside the new card branch only main arrived, into its remote-tracking ref: no other branch, no tag.
    assert _changed(refs, _refs(repo)) == {f"refs/heads/{task.branch_name}": world.main, TRACKING: world.main}


def test_a_continuation_branch_keeps_its_own_tip(world, github):
    kb, repo = world.kb, world.repo
    tip = _git(repo, "commit-tree", "HEAD^{tree}", "-p", "HEAD", "-m", "the builder's earlier work")
    _git(repo, "branch", "feature/continued", tip)
    refs = _refs(repo)
    tid = _new_card(kb, repo, branch_name="feature/continued")

    _dispatch(kb)

    task = _task(kb, tid)
    assert (task.branch_name, task.base_commit) == ("feature/continued", tip)
    assert _git(Path(task.workspace_path), "rev-parse", "HEAD") == tip
    # Nothing was fetched and no token was asked for: every ref is as it was.
    assert _refs(repo) == refs and github["tokens"] == []
    assert _base_events(kb.kanban_db_path(), tid) == [("worktree_base_recorded", {"base_commit": tip})]


def _break(case: str, world, github, monkeypatch) -> None:
    """Make the fetch of main fail the way ``case`` names."""
    from hermes_cli import kanban_delivery_github as transport

    if case == "remote_gone":
        shutil.rmtree(world.bare)
    elif case == "token_refused":  # GitHub's refusal echoes a credential, and none of it may be kept
        github["fail"][None] = (403, {"message": f"Bad credentials {TOKEN}"})
    elif case == "connection_dropped":
        github["fail"][None] = (None, None)
    elif case == "app_key_missing":
        (world.root / "secrets" / "github-app" / "raphael-agent-factory.pem").unlink()
    else:  # an origin naming a user is refused before any credential exists
        monkeypatch.setattr(transport, "GIT_ORIGIN", f"https://x-access-token:{TOKEN}@github.com")


@pytest.mark.parametrize("case, reason, tokens", [
    ("remote_gone", "fetch_failed", 1),
    ("token_refused", "token_request_failed", 1),
    ("connection_dropped", "network_error", 1),
    ("app_key_missing", "app_key_unavailable", 0),
    ("origin_with_a_user", "bad_push_target", 0),
])
def test_a_failed_fetch_falls_back_to_head_and_records_it(world, github, monkeypatch, caplog, case, reason, tokens):
    _capture_all_logs(caplog)
    kb, repo = world.kb, world.repo
    _break(case, world, github, monkeypatch)
    refs = _refs(repo)
    tid = _new_card(kb, repo)

    result = _dispatch(kb)

    task = _task(kb, tid)
    # The card starts as it does today, at the checkout's HEAD, and its worker is spawned.
    assert tid in [spawned[0] for spawned in result.spawned] and task.status == "running"
    assert task.base_commit == world.head == _git(Path(task.workspace_path), "rev-parse", "HEAD")
    assert _changed(refs, _refs(repo)) == {f"refs/heads/{task.branch_name}": world.head}
    # After the base, why: the commit actually recorded and the fixed reason, nothing else.
    assert _base_events(kb.kanban_db_path(), tid) == [
        ("worktree_base_recorded", {"base_commit": world.head}),
        ("worktree_base_fallback", {"base_commit": world.head, "reason": reason}),
    ]
    assert len(github["tokens"]) == tokens
    raw = sqlite3.connect(str(kb.kanban_db_path()))
    try:
        stored = repr(raw.execute("SELECT * FROM task_events").fetchall())
    finally:
        raw.close()
    kept = [stored, _log_text(caplog), *_files_text(world.tmp).values()]
    assert not [text for text in kept if TOKEN in text or BASIC in text]


@pytest.mark.parametrize("value", [None, "false", "'true'", "1", "'yes'", "null", "[true]"],
                         ids=["absent", "false", "the_string_true", "one", "the_string_yes", "null", "a_list"])
def test_with_the_setting_off_the_base_is_head_as_today(world, github, value):
    kb, repo = world.kb, world.repo
    _setting(world.root, value)
    refs = _refs(repo)
    tid = _new_card(kb, repo)

    _dispatch(kb)

    task = _task(kb, tid)
    assert task.base_commit == world.head == _git(Path(task.workspace_path), "rev-parse", "HEAD")
    assert _changed(refs, _refs(repo)) == {f"refs/heads/{task.branch_name}": world.head}
    assert github["tokens"] == []
    assert _base_events(kb.kanban_db_path(), tid) == [("worktree_base_recorded", {"base_commit": world.head})]


@pytest.mark.parametrize("origin", ["https://github.com/mtbitcr/elsewhere.git", None],
                         ids=["outside_the_policy", "no_origin"])
def test_a_repository_outside_the_delivery_policy_starts_at_head(world, github, origin):
    kb, repo = world.kb, world.repo
    _git(repo, "remote", "remove", "origin")
    if origin is not None:
        _git(repo, "remote", "add", "origin", origin)
    tid = _new_card(kb, repo)

    _dispatch(kb)

    task = _task(kb, tid)
    assert task.base_commit == world.head and github["tokens"] == []
    assert TRACKING not in _refs(repo)
    assert _base_events(kb.kanban_db_path(), tid) == [("worktree_base_recorded", {"base_commit": world.head})]


def test_the_credential_stays_in_the_one_fetch_and_nothing_is_left_behind(world, github, monkeypatch, caplog):
    _capture_all_logs(caplog)
    made, calls = _record_mkdtemp(monkeypatch), _record_git(monkeypatch)
    tid = _new_card(world.kb, world.repo)

    _dispatch(world.kb)

    assert _task(world.kb, tid).base_commit == world.main
    # One git held the credential: the fetch, naming one URL and after it one refspec, main into its
    # remote-tracking ref.
    carrying = [argv for argv, env in calls if _carries(argv, env)]
    url = f"{(world.tmp / 'remotes').as_uri()}/{REPO}.git"
    assert len(carrying) == 1 and carrying[0][carrying[0].index(url):] == [url, f"+refs/heads/main:{TRACKING}"]
    # Its private directory is gone, and no credential is in a file, the environment, a log or an event.
    assert made and not [path for path in made if path.exists()]
    kept = [_log_text(caplog), *os.environ.values(), *_files_text(world.tmp).values()]
    assert not [text for text in kept if TOKEN in text or BASIC in text]


@pytest.mark.parametrize("fetch", ["succeeds", "fails"])
def test_no_second_connection_opens_inside_the_live_write_transaction(world, github, monkeypatch, fetch):
    from hermes_cli import kanban_delivery_github as transport

    kb = world.kb
    if fetch == "fails":
        shutil.rmtree(world.bare)
    tid = _new_card(kb, world.repo)
    connect, fetch_main = sqlite3.connect, transport.GitHubTransport.fetch_main
    conn = kb.connect()
    opened, fetched = [], []

    def watched_connect(*args, **kwargs):
        opened.append(conn.in_transaction)
        return connect(*args, **kwargs)

    def watched_fetch(self, *args, **kwargs):
        fetched.append(conn.in_transaction)
        return fetch_main(self, *args, **kwargs)

    try:
        with monkeypatch.context() as patch:
            patch.setattr(sqlite3, "connect", watched_connect)
            patch.setattr(transport.GitHubTransport, "fetch_main", watched_fetch)
            kb.dispatch_once(conn, spawn_fn=lambda *a, **k: 1)
    finally:
        conn.close()

    # The fetch ran outside every transaction, and no connection opened while the tick's was writing.
    assert fetched == [False] and True not in opened
    base = world.main if fetch == "succeeds" else world.head
    expected = [("worktree_base_recorded", {"base_commit": base})]
    if fetch == "fails":
        expected.append(("worktree_base_fallback", {"base_commit": base, "reason": "fetch_failed"}))
    assert _base_events(kb.kanban_db_path(), tid) == expected


@pytest.mark.parametrize("failing", ["worktree_base_fallback", "worktree_base_recorded"])
def test_a_rolled_back_base_keeps_the_fallback_cause_for_the_retry(world, github, caplog, failing):
    _capture_all_logs(caplog)
    kb, db = world.kb, world.kb.kanban_db_path()
    shutil.rmtree(world.bare)
    tid = _new_card(kb, world.repo)
    expected = [("worktree_base_recorded", {"base_commit": world.head}),
                ("worktree_base_fallback", {"base_commit": world.head, "reason": "fetch_failed"})]
    conn = kb.connect()
    try:
        # One real SQLite error inside record_worktree_base's write transaction, on the tick's own connection.
        conn.execute(f"CREATE TEMP TRIGGER injected BEFORE INSERT ON task_events WHEN NEW.kind = '{failing}' "
                     "BEGIN SELECT RAISE(ABORT, 'injected'); END")
        first = kb.dispatch_once(conn, spawn_fn=lambda *a, **k: 1)
        task = _task(kb, tid)
        # The rollback is real: no base and no base event; no worker, and the card waits in ready.
        assert task.base_commit is None and _base_events(db, tid) == []
        assert tid not in [spawned[0] for spawned in first.spawned] and task.status == "ready"
        conn.execute("DROP TRIGGER temp.injected")

        retry = kb.dispatch_once(conn, spawn_fn=lambda *a, **k: 1)

        # The worktree is already there, so nothing is fetched again: the retry records the base the
        # branch really started at and, after it, why.
        task = _task(kb, tid)
        workspace = Path(task.workspace_path)
        assert tid in [spawned[0] for spawned in retry.spawned] and github["tokens"] == [PUBLISH]
        assert task.base_commit == world.head == _git(workspace, "rev-parse", "HEAD")
        assert _base_events(db, tid) == expected
        # The cause is written once: recording the base again changes nothing.
        assert kb.record_worktree_base(conn, tid, workspace) == world.head
    finally:
        conn.close()
    assert _base_events(db, tid) == expected
    raw = sqlite3.connect(str(db))
    try:
        stored = repr(raw.execute("SELECT * FROM task_events").fetchall())
    finally:
        raw.close()
    assert not [text for text in (stored, _log_text(caplog)) if TOKEN in text or BASIC in text]


def test_store_path_under_the_dispatcher_worker_environment(world, github, monkeypatch):
    """This process has the environment the dispatcher gives a worker on proj-a (its board pinned, its
    card set, a profile home under the root) and ticks another board, whose new card starts at main."""
    kb, root, repo = world.kb, world.root, world.repo
    kb.create_board("proj-a")
    kb.create_board("other")
    own_db = root / "kanban" / "boards" / "proj-a" / "kanban.db"
    other_db = root / "kanban" / "boards" / "other" / "kanban.db"
    profile_home = root / "profiles" / IMPLEMENTER
    _setting(profile_home, "true")
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
    tid = _new_card(kb, repo, board="other")

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
    # The answer against an independent count: the board files, the stand-in and the repository.
    raw = sqlite3.connect(str(other_db))
    try:
        assert raw.execute("SELECT base_commit FROM tasks WHERE id = ?", (tid,)).fetchall() == [(world.main,)]
    finally:
        raw.close()
    assert _base_events(other_db, tid) == [("worktree_base_recorded", {"base_commit": world.main})]
    assert github["tokens"] == [PUBLISH] and _refs(repo)[TRACKING] == world.main
    for db in (own_db, root / "kanban.db"):
        raw = sqlite3.connect(str(db))
        try:
            assert raw.execute("SELECT COUNT(*) FROM tasks WHERE id = ?", (tid,)).fetchone() == (0,)
            assert raw.execute("SELECT COUNT(*) FROM task_events WHERE task_id = ?", (tid,)).fetchone() == (0,)
        finally:
            raw.close()
    assert not (profile_home / "kanban.db").exists() and not (profile_home / "kanban").exists()
