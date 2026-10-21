"""Card A4: clone an owner's repository under the kanban root and bind it as a project's primary folder.

A local bare repository stands in for GitHub's git side, with GIT_ORIGIN pointed at file://, and the transport
tests' stand-in plays its token endpoint. The read of main's ref is answered from the bare remote, as GitHub would
answer it. Every test imports the new module itself, so each one fails on its own while the module is missing.
"""
import contextlib
import importlib
import json
import os
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import projects_db as pdb
from tests.hermes_cli.test_kanban_delivery_github import (  # noqa: F401  (github is a fixture)
    REPO, TOKEN, _git, _remote_head, _remote_refs, _repositories, _transport_module, github,
)
from tests.hermes_cli.test_kanban_delivery_owner_policy import (  # noqa: F401  (root is a fixture)
    WORKER, _become, _spawned_worker_env, root,
)

ROOT = Path(__file__).resolve().parents[2]
BOOTSTRAP_FILES = [".github/workflows/ci.yml", "README.md", "scripts/ci.sh"]
_RESOLVE_AS_DEFAULT_PROFILE = f"""
import json
from hermes_cli import repo_connect as rc
try:
    rc.connect_repository("second", {REPO!r})
    refused = None
except rc.RepositoryConnectError as exc:
    refused = exc.reason
print(json.dumps({{"folder": str(rc._folder({REPO!r})), "refused": refused}}))
"""


def _connect_module():
    """Import inside each test, so a missing module fails that test rather than the whole file."""
    return importlib.import_module("hermes_cli.repo_connect")


def _folder(root):
    """Where the clone of REPO belongs, located from the root by hand."""
    return root / "kanban" / "repositories" / "mtbitcr" / "hermes-agent"


def _answer_main(monkeypatch, mod, bare, missing):
    """GitHub's answer to the read of main's ref, taken from the bare remote: 200 and main's commit when the remote
    has main, else `missing` (GitHub answers 409 for an empty repository, 404 for a missing branch). Every other call
    goes to the stand-in. Returns the statuses given, in order."""
    exchange, given = mod._exchange, []

    def answer(method, target, authorization, payload=None):
        if (method, target) != ("GET", f"/repos/{REPO}/git/ref/heads/main"):
            return exchange(method, target, authorization, payload)
        head = _remote_head(bare, "main")
        status = 401 if authorization != f"Bearer {TOKEN}" else 200 if head else missing
        given.append(status)
        ref = {"ref": "refs/heads/main", "object": {"sha": head, "type": "commit"}}
        return status, ref if status == 200 else None, False

    monkeypatch.setattr(mod, "_exchange", answer)
    return given


def _github_remote(root, github, monkeypatch, tmp_path, missing=404):
    """The transport pointed at the stand-in, with the App key where production keeps it, and an empty bare remote as
    its GIT_ORIGIN; a card repository with three commits to fill the remote with."""
    mod = _transport_module(monkeypatch, github)
    work, bare, origin, commits = _repositories(tmp_path)
    monkeypatch.setattr(mod, "GIT_ORIGIN", origin)
    key = root / "secrets" / "github-app" / "raphael-agent-factory.pem"
    key.parent.mkdir(parents=True)
    key.write_bytes(github["key_file"].read_bytes())
    return mod, work, bare, commits, _answer_main(monkeypatch, mod, bare, missing)


def _project_on_board(store=None):
    """The board "app" and the project "app" bound to it, with no folder, in `store` or the profile's own store."""
    kb._INITIALIZED_PATHS.clear()
    kb.init_db()
    kb.create_board("app")
    with pdb.connect_closing(store) as conn:
        return pdb.create_project(conn, name="App", slug="app", board_slug="app")


def _state(root, bare):
    """What a refused connect must leave as it was: every folder and file under the kanban root, each board's
    metadata byte for byte, the projects store's rows and the remote's refs."""
    kanban = root / "kanban"
    paths = sorted(str(p.relative_to(kanban)) for p in kanban.rglob("*") if p.suffix not in (".db-wal", ".db-shm"))
    boards = {str(p.relative_to(kanban)): p.read_bytes() for p in kanban.rglob("board.json")}
    with contextlib.closing(sqlite3.connect(root / "projects.db")) as store:
        rows = [store.execute("SELECT * FROM projects ORDER BY id").fetchall(),
                store.execute("SELECT * FROM project_folders ORDER BY project_id, path").fetchall()]
    return paths, boards, rows, _remote_refs(bare)


def _bound():
    """The project "app"'s primary path and its folders, each with whether it is primary."""
    with pdb.connect_closing() as conn:
        project = pdb.get_project(conn, "app")
    return project.primary_path, [(f.path, f.is_primary) for f in project.folders]


@pytest.mark.parametrize("missing", [404, 409])  # 409 is GitHub's answer for an empty repository
def test_an_empty_remote_is_bootstrapped_cloned_and_bound_as_the_primary_folder(
        missing, root, github, monkeypatch, tmp_path):
    mod, _, bare, _, answers = _github_remote(root, github, monkeypatch, tmp_path, missing)
    _project_on_board()
    folder = _folder(root)

    receipt = _connect_module().connect_repository("app", REPO)

    head = _remote_head(bare, "main")
    assert receipt == {"repository": REPO, "folder": str(folder), "bootstrapped": True, "commit": head}
    assert answers == [missing]
    assert [request["permissions"] for request in github["token_requests"]] == [
        dict(mod.STEP_PERMISSIONS["read_checks"]), dict(mod.STEP_PERMISSIONS["bootstrap"])]
    assert _remote_refs(bare) == ["refs/heads/main"]
    assert _git("--git-dir", str(bare), "ls-tree", "-r", "--name-only", "main").splitlines() == BOOTSTRAP_FILES
    assert _git("rev-parse", "HEAD", cwd=folder) == head
    assert _git("symbolic-ref", "HEAD", cwd=folder) == "refs/heads/main"
    assert _git("ls-files", cwd=folder).splitlines() == BOOTSTRAP_FILES
    assert _git("remote", "get-url", "origin", cwd=folder) == f"https://github.com/{REPO}.git"
    assert _bound() == (str(folder), [(str(folder), True)])
    assert kb.read_board_metadata("app")["default_workdir"] == str(folder)


def test_a_remote_with_main_is_cloned_without_a_bootstrap(root, github, monkeypatch, tmp_path):
    mod, work, bare, (_, second, _), answers = _github_remote(root, github, monkeypatch, tmp_path)
    _git("push", "-q", str(bare), f"{second}:refs/heads/main", cwd=work)
    _project_on_board()
    folder = _folder(root)

    receipt = _connect_module().connect_repository("app", REPO)

    assert receipt == {"repository": REPO, "folder": str(folder), "bootstrapped": False, "commit": second}
    assert answers == [200]
    assert [request["permissions"] for request in github["token_requests"]] == [
        dict(mod.STEP_PERMISSIONS["read_checks"])]  # no bootstrap token
    assert _remote_refs(bare) == ["refs/heads/main"] and _remote_head(bare, "main") == second
    assert _git("rev-parse", "HEAD", cwd=folder) == second
    assert _bound() == (str(folder), [(str(folder), True)])
    assert kb.read_board_metadata("app")["default_workdir"] == str(folder)


@pytest.mark.parametrize("case, reason, tokens", [
    ("project with a primary folder", "project_has_primary_folder", 0),
    ("existing folder", "repository_has_folder", 0),
    ("folder of another project", "repository_has_project", 0),
    ("another owner", "foreign_account", 0),
    ("name the transport refuses", "bad_repository", 0),
    ("refused token mint", "token_request_failed", 1),
    ("another answer for main", "main_unreadable", 1),
])
def test_a_refused_connect_writes_nothing(case, reason, tokens, root, github, monkeypatch, tmp_path):
    missing = 500 if case == "another answer for main" else 404
    mod, _, bare, _, answers = _github_remote(root, github, monkeypatch, tmp_path, missing)
    app = _project_on_board()
    folder = _folder(root)
    repository = {"another owner": "someone/hermes-agent", "name the transport refuses": "mtbitcr/.."}.get(case, REPO)
    with pdb.connect_closing() as conn:
        if case == "project with a primary folder":
            pdb.add_folder(conn, app, str(tmp_path / "elsewhere"))
        elif case == "folder of another project":
            pdb.create_project(conn, name="Old", slug="old", folders=[str(folder)])
    if case == "existing folder":
        folder.mkdir(parents=True)  # empty, which clone_main alone would take
    if case == "refused token mint":
        github["token_reply"] = "refused"
    before = _state(root, bare)

    with pytest.raises(mod.GitHubTransportError) as refused:
        _connect_module().connect_repository("app", repository)

    assert refused.value.reason == reason
    assert _state(root, bare) == before
    assert [request["permissions"] for request in github["token_requests"]] == [
        dict(mod.STEP_PERMISSIONS["read_checks"])] * tokens  # never a bootstrap token
    assert answers == ([missing] if case == "another answer for main" else [])


@pytest.mark.parametrize("fault", ["clone", "store", "board"])
def test_a_failure_after_the_clone_leaves_no_folder_and_a_retry_connects(fault, root, github, monkeypatch, tmp_path):
    mod, _, bare, _, answers = _github_remote(root, github, monkeypatch, tmp_path)
    _project_on_board()
    folder = _folder(root)
    board = root / "kanban" / "boards" / "app" / "board.json"
    metadata = board.read_bytes()

    with monkeypatch.context() as patch:
        if fault == "clone":  # git filled the folder, then the clone failed
            clone_main = mod.GitHubTransport.clone_main

            def clone_then_fail(self, target):
                clone_main(self, target)
                raise mod.GitHubTransportError("clone_failed")

            patch.setattr(mod.GitHubTransport, "clone_main", clone_then_fail)
        elif fault == "store":
            def locked(*args, **kwargs):
                raise sqlite3.OperationalError("database is locked")

            patch.setattr(pdb, "add_folder", locked)
        else:
            board.write_text("{not json\n")  # metadata that cannot be read is never rewritten
        with pytest.raises((mod.GitHubTransportError, sqlite3.OperationalError, ValueError)) as raised:
            _connect_module().connect_repository("app", REPO)

    assert type(raised.value) is {"clone": mod.GitHubTransportError, "store": sqlite3.OperationalError,
                                  "board": ValueError}[fault]
    assert not os.path.lexists(folder)
    assert not [p for p in (root / "kanban" / "repositories").rglob("*") if not p.is_dir()]
    assert _bound() == (None, [])
    assert board.read_bytes() == (b"{not json\n" if fault == "board" else metadata)
    assert _remote_refs(bare) == ["refs/heads/main"]  # the first commit stays on GitHub

    board.write_bytes(metadata)
    receipt = _connect_module().connect_repository("app", REPO)
    assert receipt == {"repository": REPO, "folder": str(folder), "bootstrapped": False,
                       "commit": _remote_head(bare, "main")}
    assert answers == [404, 200]
    assert _bound() == (str(folder), [(str(folder), True)])


def test_store_path_under_the_dispatcher_worker_environment_and_the_default_profile(
        root, github, monkeypatch, tmp_path):
    mod, _, bare, _, _ = _github_remote(root, github, monkeypatch, tmp_path)
    profile_home = root / "profiles" / WORKER
    profile_home.mkdir(parents=True)
    _project_on_board(profile_home / "projects.db")  # the worker profile's own store
    kb.create_board("proj-a")
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    folder = _folder(root)

    # The dispatcher claims a card on its board and spawns through the live _default_spawn.
    conn = kb.connect(board="proj-a")
    try:
        tid = kb.create_task(conn, title="connect a repository", assignee=WORKER)
        claimed = kb.claim_task(conn, tid)
        assert claimed is not None
        env = _spawned_worker_env(kb, claimed, workspace, "proj-a", monkeypatch)
    finally:
        conn.close()
    assert env["HERMES_KANBAN_DB"] == str(root / "kanban" / "boards" / "proj-a" / "kanban.db")
    assert env["HERMES_KANBAN_BOARD"] == "proj-a"
    assert env["HERMES_KANBAN_TASK"] == tid
    assert env["HERMES_HOME"] == str(profile_home)

    # The worker carries on with exactly that environment and connects a project bound to another board.
    _become(env, monkeypatch)
    receipt = _connect_module().connect_repository("app", REPO)
    head = _remote_head(bare, "main")
    assert receipt == {"repository": REPO, "folder": str(folder), "bootstrapped": True, "commit": head}

    # From there the root registry and every other board still resolve.
    assert kb.kanban_home() == root
    assert kb.register_db_path() == root / "kanban" / "board_register.db"
    with contextlib.closing(sqlite3.connect(root / "kanban" / "board_register.db")) as register:
        registered = {name for (name,) in register.execute("SELECT board_name FROM board_register")}
    boards = root / "kanban" / "boards"
    on_disk = {child.name for child in boards.iterdir() if (child / "kanban.db").exists()}
    assert registered == on_disk == {"app", "proj-a"}
    assert {entry["slug"] for entry in kb.list_boards()} == on_disk | {"default"}
    assert all(kb.get_register_entry(name) is not None for name in registered)
    assert kb.read_board_metadata("app")["default_workdir"] == str(folder)

    # The receipt, checked against an independent count of what is on disk.
    repositories = root / "kanban" / "repositories"
    assert sorted(str(p.relative_to(repositories)) for p in repositories.glob("*/*")) == ["mtbitcr/hermes-agent"]
    assert _git("rev-parse", "HEAD", cwd=folder) == head
    workdirs = {child.name: json.loads((child / "board.json").read_text(encoding="utf-8")).get("default_workdir")
                for child in boards.iterdir() if (child / "board.json").exists()}
    assert [name for name, workdir in workdirs.items() if workdir == str(folder)] == ["app"]
    with contextlib.closing(sqlite3.connect(profile_home / "projects.db")) as store:
        assert store.execute("SELECT slug, primary_path FROM projects").fetchall() == [("app", str(folder))]
        assert store.execute("SELECT path, is_primary FROM project_folders").fetchall() == [(str(folder), 1)]
    assert not (profile_home / "kanban").exists()

    # A second process living in the default profile home resolves the same folder, and is refused it.
    with pdb.connect_closing(root / "projects.db") as conn:
        pdb.create_project(conn, name="Second", slug="second")
    child_env = {name: value for name, value in os.environ.items() if not name.startswith("HERMES_")}
    child_env.update(HERMES_HOME=str(root), HOME=str(tmp_path))
    result = subprocess.run([sys.executable, "-c", _RESOLVE_AS_DEFAULT_PROFILE], cwd=ROOT, env=child_env,
                            capture_output=True, text=True, timeout=120, check=False)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout.strip().splitlines()[-1]) == {
        "folder": str(folder), "refused": "repository_has_folder"}
    with contextlib.closing(sqlite3.connect(root / "projects.db")) as store:
        assert store.execute("SELECT COUNT(*) FROM project_folders").fetchone() == (0,)
    assert sorted(str(p.relative_to(repositories)) for p in repositories.glob("*/*")) == ["mtbitcr/hermes-agent"]
    assert _git("rev-parse", "HEAD", cwd=folder) == head
