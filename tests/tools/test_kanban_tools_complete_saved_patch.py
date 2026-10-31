"""A build that completes without naming its saved patch hands it to review.

``kanban_complete`` used to read only its ``patch_attachment_id`` argument.
A build that had saved its change as a patch attachment on its own run and
then completed without naming it -- on 2026-10-28 a rework run named the id
only inside ``metadata`` -- had its card parked for review without the
change. The tool now asks the kernel which patches the run saved for review
(``kanban_db.saved_patch_ids_for_review``, the one copy of the selection
rules, oldest receipt first) and, when the call names no patch itself, hands
the newest over exactly as ``kanban_request_review`` does:

* one saved patch: it is materialized on the task branch and the card parks
  in the review lane;
* several saved patches: the newest by receipt order is handed over, and a
  patch id inside ``metadata`` is not read;
* no saved patch: the worktree head is parked, as before; and
* a run claimed from review: its own saved patch is never handed over.

The worker side of every scenario runs in the exact environment the
dispatcher gives the worker it launches: it is taken from
``kanban_db._default_spawn`` itself, with only the process launch stubbed:
``HERMES_KANBAN_DB`` and ``HERMES_KANBAN_BOARD`` pinned to the worker's own
board, ``HERMES_KANBAN_TASK`` set, and ``HERMES_HOME`` at the worker's profile
home below a temporary root. From there the root registry and every other
board are checked to be still in reach. The change is checked independently
of the kernel: the task branch is read back with git and the board files are
counted with plain SQL.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import subprocess
from contextlib import closing
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb

SLUG = "complete-handover"
OTHER = "other-board"
IMPLEMENTER = "worker"
# A profile the model policy admits; the policy only ever nominates those.
REVIEWER = "raphael-verifier"
BRANCH = "feature/complete-handover"
OWNED = "src/owned"
MODULE = f"{OWNED}/app.py"
BASE_LINE = "value = 1"
SUMMARY = "changed the owned module and ran its unit tests"
SEED_AUTHOR = ("Board Owner", "owner@example.invalid")
WORKER_AUTHOR = ("Build Worker", "worker@example.invalid")
_GIT_ENV = {
    "GIT_CONFIG_GLOBAL": os.devnull,
    "GIT_CONFIG_SYSTEM": os.devnull,
    "GIT_TERMINAL_PROMPT": "0",
}


def _git(repo: Path, *argv: str, author: tuple[str, str] | None = None) -> str:
    """Run a real git command in ``repo``, isolated from the host's config."""
    identity: list[str] = []
    if author is not None:
        identity = ["-c", f"user.name={author[0]}", "-c", f"user.email={author[1]}"]
    proc = subprocess.run(
        ["git", "-C", str(repo), *identity, *argv],
        capture_output=True, text=True, timeout=60,
        env={**os.environ, **_GIT_ENV},
    )
    assert proc.returncode == 0, (argv, proc.stderr or proc.stdout)
    return proc.stdout


def _branch_head(build: dict) -> str:
    """The commit the task branch points at, read from the repository."""
    return _git(
        build["repo"], "rev-parse", "--verify", f"refs/heads/{build['branch']}"
    ).strip()


def _patch_trailers(build: dict) -> list[str]:
    """The patch trailers on the task branch since its base, read with git."""
    log = _git(
        build["repo"], "log", "--format=%B", f"{build['base']}..{build['branch']}",
    )
    return [
        line for line in log.splitlines()
        if line.startswith("Hermes-Patch-Attachment:")
    ]


def _patch(line: str, before: str = BASE_LINE) -> bytes:
    """A patch that turns the owned module's one line ``before`` into ``line``."""
    return (
        f"diff --git a/{MODULE} b/{MODULE}\n"
        f"--- a/{MODULE}\n"
        f"+++ b/{MODULE}\n"
        "@@ -1 +1 @@\n"
        f"-{before}\n"
        f"+{line}\n"
    ).encode("utf-8")


def _content_after(patch: bytes) -> str:
    """The module's whole content once ``patch`` has replaced its every line."""
    return "".join(
        f"{line[1:]}\n"
        for line in patch.decode("utf-8").splitlines()
        if line.startswith("+") and not line.startswith("+++")
    )


def _count(db: Path, table: str, where: str, params: tuple = ()) -> int:
    """An independent read: a plain sqlite connection, no Hermes code."""
    with closing(sqlite3.connect(db)) as raw:
        return raw.execute(
            f"SELECT COUNT(*) FROM {table} WHERE {where}", params,
        ).fetchone()[0]


def _upload(
    build: dict,
    name: str,
    data: bytes,
    *,
    content_type: str = "text/x-diff",
) -> int:
    """Store ``data`` as the run's own agent upload, through the kernel.

    The board is opened the way the worker's own calls open it, from the
    worker's environment with no board or path named.
    """
    conn = kb.connect()
    try:
        return kb.store_attachment_bytes(
            conn, build["task"], name, data,
            content_type=content_type, uploaded_by="agent",
            expected_run_id=build["run"],
        )
    finally:
        conn.close()


def _spawned_worker_env(task, workspace: Path, monkeypatch) -> dict:
    """The environment ``_default_spawn`` gives the worker process it launches.

    Only the launch is stubbed: the stub records the argv and environment the
    dispatcher built and returns a pid, so no process is started.
    """
    launches = []

    class _Launched:
        pid = 4242

    def _launch(argv, **kwargs):
        log = kwargs.get("stdout")
        if hasattr(log, "close"):
            log.close()
        launches.append((list(argv), dict(kwargs.get("env") or {})))
        return _Launched()

    with monkeypatch.context() as patch:
        patch.setattr(subprocess, "Popen", _launch)
        pid = kb._default_spawn(task, str(workspace), board=SLUG)
    assert pid == _Launched.pid
    assert len(launches) == 1, launches
    argv, env = launches[0]
    assert f"work kanban task {task.id}" in argv
    return env


def _become(env: dict, monkeypatch) -> None:
    """Carry on as that process: this process now has exactly ``env``."""
    for name in list(os.environ):
        if name not in env:
            monkeypatch.delenv(name)
    for name, value in env.items():
        if os.environ.get(name) != value:
            monkeypatch.setenv(name, value)


def _assert_root_in_reach(root: Path) -> None:
    """From the worker's environment the root registry and every board are in reach.

    The kernel's answers are read against the registry file's own rows,
    counted with plain SQL.
    """
    registry = root / "kanban" / "board_register.db"
    assert kb.register_db_path().resolve() == registry.resolve()
    with closing(sqlite3.connect(registry)) as db:
        registered = {
            row[0] for row in db.execute("SELECT board_name FROM board_register")
        }
    assert {SLUG, OTHER} <= registered
    listed = {board["slug"] for board in kb.list_boards()}
    assert listed - {kb.DEFAULT_BOARD} == registered - {kb.DEFAULT_BOARD}
    for slug in registered:
        entry = kb.get_register_entry(slug)
        assert entry is not None and entry.lifecycle is kb.BoardLifecycle.LIVE, slug
        assert kb.board_dir(slug).resolve() == (root / "kanban" / "boards" / slug).resolve()


def _launch_worker(conn, claimed, root: Path, tmp_path: Path, monkeypatch):
    """Prepare the claimed card as the dispatcher does, then become its worker.

    Returns the work area, the branch and the base the dispatcher recorded.
    """
    task_id = claimed.id
    workspace, branch = kb._resolve_worktree_workspace(claimed, board=SLUG)
    kb.set_workspace_path(conn, task_id, str(workspace))
    kb.set_branch_name(conn, task_id, branch)
    base = kb.record_worktree_base(conn, task_id, workspace)

    env = _spawned_worker_env(claimed, Path(workspace), monkeypatch)

    # Pinned to its own card, run and board, under its profile home below the
    # root, and given nothing more: no attachment-root override.
    assert env["HERMES_KANBAN_TASK"] == task_id
    assert env["HERMES_KANBAN_RUN_ID"] == str(claimed.current_run_id)
    assert env["HERMES_KANBAN_BOARD"] == SLUG
    assert env["HERMES_KANBAN_DB"] == str(root / "kanban" / "boards" / SLUG / "kanban.db")
    assert env["HERMES_HOME"] == str(root / "profiles" / claimed.assignee)
    assert "HERMES_KANBAN_ATTACHMENTS_ROOT" not in env
    for name in (
        "HERMES_HOME",
        "HERMES_KANBAN_DB",
        "HERMES_KANBAN_WORKSPACES_ROOT",
        "HERMES_KANBAN_WORKSPACE",
    ):
        assert Path(env[name]).resolve().is_relative_to(tmp_path.resolve()), name

    _become(env, monkeypatch)
    _assert_root_in_reach(root)
    return Path(workspace), branch, base


@pytest.fixture
def fenced_root(tmp_path, monkeypatch):
    """An empty Hermes root with no board, no register and no archive.

    The same start as the shared ``fence_home`` fixture, so every board that
    exists later is one a real kernel call created during the test. The
    attachment-root override (which the dispatcher never injects) and any
    card, run, profile or session this process carries are cleared as well.
    """
    root = tmp_path / "hermes_home"
    root.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(root))
    for var in (
        "HERMES_KANBAN_DB",
        "HERMES_KANBAN_WORKSPACES_ROOT",
        "HERMES_KANBAN_HOME",
        "HERMES_KANBAN_BOARD",
        "HERMES_KANBAN_ATTACHMENTS_ROOT",
        "HERMES_KANBAN_TASK",
        "HERMES_KANBAN_RUN_ID",
        "HERMES_PROFILE",
        "HERMES_SESSION_ID",
    ):
        monkeypatch.delenv(var, raising=False)

    kb._INITIALIZED_PATHS.clear()
    kb._REGISTER_INITIALIZED = False
    kb._REGISTER_INITIALIZED_PATHS.clear()
    kb._ARCHIVE_INITIALIZED_PATHS.clear()

    assert not (root / "kanban.db").exists()
    assert not kb.register_db_path().exists()
    return root


@pytest.fixture
def worker_build(fenced_root, tmp_path, monkeypatch):
    """A claimed worktree build that needs review, with this process as its worker.

    Set up the way the dispatcher sets one up (claim, resolve the work area,
    record its branch and base); then the test takes on exactly the
    environment ``_default_spawn`` gives the worker it launches. Another live
    board holds someone else's card, and the review policy nominates one
    installed reviewer.
    """
    from plugins.dashboard_auth.raphael_workspace import model_policy

    monkeypatch.setattr(
        model_policy, "reviewer_profile_ids", lambda: (REVIEWER,), raising=True,
    )
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "--initial-branch=main")
    (repo / OWNED).mkdir(parents=True)
    (repo / MODULE).write_text(f"{BASE_LINE}\n", encoding="utf-8")
    _git(repo, "add", MODULE)
    _git(repo, "commit", "-m", "owned module", author=SEED_AUTHOR)

    for slug in (SLUG, OTHER):
        kb.create_board(slug)
        entry = kb.get_register_entry(slug)
        assert entry is not None and entry.lifecycle is kb.BoardLifecycle.LIVE
    with closing(kb.connect(board=OTHER)) as other:
        kb.create_task(other, title="someone else's card", assignee="default")
    for profile in (IMPLEMENTER, REVIEWER):
        (fenced_root / "profiles" / profile).mkdir(parents=True)
    dispatcher_env = dict(os.environ)
    conn = kb.connect(board=SLUG)
    try:
        task_id = kb.create_task(
            conn, title="build that completes", assignee=IMPLEMENTER,
            workspace_kind="worktree", workspace_path=str(repo),
            branch_name=BRANCH, owned_paths=[OWNED], requires_review=True,
        )
        claimed = kb.claim_task(conn, task_id)
        assert claimed is not None and claimed.current_run_id is not None
        workspace, branch, base = _launch_worker(
            conn, claimed, fenced_root, tmp_path, monkeypatch,
        )
        yield {
            "conn": conn,
            "task": task_id,
            "run": claimed.current_run_id,
            "repo": repo,
            "workspace": workspace,
            "branch": branch,
            "base": base,
            "root": fenced_root,
            "dispatcher_env": dispatcher_env,
        }
    finally:
        conn.close()


def test_a_saved_patch_is_handed_over_when_complete_names_none(worker_build):
    """(a) The run's saved patch lands on the task branch and the card parks.

    The worker names the patch only inside ``metadata``, as the 2026-10-28
    rework run did, and not in ``patch_attachment_id``.
    """
    from tools import kanban_tools as kt

    build = worker_build
    conn, task_id, run_id = build["conn"], build["task"], build["run"]
    root = build["root"]
    board_db = root / "kanban" / "boards" / SLUG / "kanban.db"
    other_db = root / "kanban" / "boards" / OTHER / "kanban.db"
    profile_home = Path(os.environ["HERMES_HOME"])
    other_before = (
        _count(other_db, "tasks", "1 = 1"), _count(other_db, "task_events", "1 = 1"),
    )
    patch = _patch("value = 2")
    patch_id = _upload(build, "planned-rework.patch", patch)
    assert kb.saved_patch_ids_for_review(conn, task_id, run_id) == [patch_id]

    out = json.loads(kt._handle_complete({
        "summary": SUMMARY, "metadata": {"patch_attachment_id": patch_id},
    }))

    # The tool's own proof ...
    assert out.get("ok") is True, out
    assert out["run_id"] == run_id
    assert out["status"] == "review"
    assert out["handed_over_patch_attachment_id"] == patch_id
    assert kb.get_task(conn, task_id).status == "review"
    # ... read against an independent count: the branch with git, the board
    # file with plain SQL. The branch moved off its base by exactly one
    # commit, which holds exactly what the patch says.
    head = _branch_head(build)
    assert _git(build["repo"], "rev-list", "--count", f"{build['base']}..{head}") == "1\n"
    changed = _git(build["repo"], "show", f"{build['branch']}:{MODULE}")
    assert changed == _content_after(patch) == "value = 2\n"
    assert _patch_trailers(build) == [f"Hermes-Patch-Attachment: {patch_id}"]
    assert _count(board_db, "tasks", "id = ? AND status = 'review'", (task_id,)) == 1
    assert _count(
        board_db, "task_events", "task_id = ? AND kind = 'review_requested'", (task_id,),
    ) == 1
    assert _count(
        board_db, "task_events", "task_id = ? AND kind = 'completed'", (task_id,),
    ) == 0
    with closing(sqlite3.connect(board_db)) as raw:
        (stored,) = raw.execute(
            "SELECT metadata FROM task_runs WHERE id = ?", (run_id,),
        ).fetchone()
    materialized = json.loads(stored)["worktree_materialization"]
    assert materialized["materialized_head"] == head
    assert materialized["patch_attachment_id"] == patch_id
    assert materialized["patch_sha256"] == hashlib.sha256(patch).hexdigest()
    # Nothing was written on the other board or below the profile home.
    assert (
        _count(other_db, "tasks", "1 = 1"), _count(other_db, "task_events", "1 = 1"),
    ) == other_before
    assert not (profile_home / "kanban.db").exists()
    assert not (profile_home / "kanban").exists()


def test_the_newest_of_several_saved_patches_is_handed_over_on_complete(
    worker_build,
):
    """(b) Two saved patches: the newer by receipt order lands and the card parks.

    The older id inside ``metadata`` is not read.
    """
    from tools import kanban_tools as kt

    build = worker_build
    conn, task_id, run_id = build["conn"], build["task"], build["run"]
    first = _upload(build, "first.patch", _patch("value = 2"))
    newest = _patch("value = 3")
    second = _upload(build, "second.patch", newest)
    assert kb.saved_patch_ids_for_review(conn, task_id, run_id) == [first, second]

    out = json.loads(kt._handle_complete({
        "summary": SUMMARY, "metadata": {"patch_attachment_id": first},
    }))

    assert out.get("ok") is True, out
    assert out["status"] == "review"
    assert out["handed_over_patch_attachment_id"] == second
    assert kb.get_task(conn, task_id).status == "review"
    # Read back with git, not from the kernel: the branch holds exactly what
    # the newer patch says, and that is the one patch on it.
    assert _branch_head(build) != build["base"]
    changed = _git(build["repo"], "show", f"{build['branch']}:{MODULE}")
    assert changed == _content_after(newest) == "value = 3\n"
    assert _patch_trailers(build) == [f"Hermes-Patch-Attachment: {second}"]


def test_without_a_saved_patch_complete_parks_the_worktree_head(worker_build):
    """(c) No saved patch: the worker's committed head is parked, as before."""
    from tools import kanban_tools as kt

    build = worker_build
    conn, task_id, run_id = build["conn"], build["task"], build["run"]
    # The run's own upload, but not a patch, so not a saved change.
    _upload(build, "notes.txt", b"ran the unit tests\n", content_type="text/plain")
    workspace = build["workspace"]
    (workspace / MODULE).write_text("value = 2\n", encoding="utf-8")
    _git(workspace, "add", MODULE)
    _git(workspace, "commit", "-m", "change the owned module", author=WORKER_AUTHOR)
    worker_commit = _git(workspace, "rev-parse", "HEAD").strip()
    assert kb.saved_patch_ids_for_review(conn, task_id, run_id) == []

    out = json.loads(kt._handle_complete({"summary": SUMMARY}))

    assert out.get("ok") is True, out
    assert out["run_id"] == run_id
    assert "handed_over_patch_attachment_id" not in out
    assert kb.get_task(conn, task_id).status == "review"
    assert _branch_head(build) == worker_commit
    assert _patch_trailers(build) == []


def test_a_run_claimed_from_review_hands_no_patch_over(
    worker_build, tmp_path, monkeypatch,
):
    """(d) The reviewer's own saved patch is never handed over by its completion."""
    from tools import kanban_tools as kt

    build = worker_build
    conn, task_id = build["conn"], build["task"]
    workspace = build["workspace"]
    (workspace / MODULE).write_text("value = 2\n", encoding="utf-8")
    _git(workspace, "add", MODULE)
    _git(workspace, "commit", "-m", "change the owned module", author=WORKER_AUTHOR)
    worker_commit = _git(workspace, "rev-parse", "HEAD").strip()
    parked = json.loads(kt._handle_complete({"summary": SUMMARY}))
    assert parked.get("ok") is True, parked
    assert kb.get_task(conn, task_id).status == "review"

    # Back in the dispatcher: it claims the review and launches the reviewer.
    _become(build["dispatcher_env"], monkeypatch)
    review = kb.claim_review_task(conn, task_id)
    assert review is not None and review.assignee == REVIEWER
    review_run = review.current_run_id
    assert review_run not in (None, build["run"])
    _launch_worker(conn, review, build["root"], tmp_path, monkeypatch)
    assert kb.run_claimed_from_review(conn, task_id, review_run) is True
    # The reviewer saves a patch of its own on its own run; it would apply.
    suggested = _upload(
        {**build, "run": review_run}, "suggested.patch",
        _patch("value = 5", before="value = 2"),
    )
    assert suggested in {a.id for a in kb.list_attachments(conn, task_id)}
    assert kb.saved_patch_ids_for_review(conn, task_id, review_run) == []

    out = json.loads(kt._handle_complete({"summary": "reviewed the change"}))

    assert out.get("ok") is True, out
    assert out["run_id"] == review_run
    assert "handed_over_patch_attachment_id" not in out
    assert kb.get_task(conn, task_id).status == "done"
    # Read back with git: the branch still holds the worker's commit and no
    # patch was materialized on it.
    assert _branch_head(build) == worker_commit
    changed = _git(build["repo"], "show", f"{build['branch']}:{MODULE}")
    assert changed == "value = 2\n"
    assert _patch_trailers(build) == []
