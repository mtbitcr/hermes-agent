"""A build that asks for review hands its saved change over to the review.

``kanban_request_review`` used to look only at what the task branch already
held, so a build that had saved its change as a patch attachment instead of
committing it could not ask for review at all: the kernel refuses to park
scoped work with nothing committed. The tool now asks the kernel which
patches the run saved for review
(``kanban_db.saved_patch_ids_for_review``, the one copy of the selection
rules) and decides by the length of that list alone:

* one saved patch: it is handed over through the kernel's existing handover
  (``complete_task`` with the patch: materialize it on the task branch, prove
  the declared scope, park the card in the review lane);
* no saved patch: the tool does exactly what it did before; and
* several saved patches: the build is told plainly which ones, and no review
  is requested.

The worker side of every scenario runs in the exact environment the
dispatcher gives the worker it launches: it is taken from
``kanban_db._default_spawn`` itself, with only the process launch stubbed, and
every path in it points below the test's own temporary directory. The change
is checked independently of the kernel, by reading the file back from the
task branch with git.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb

SLUG = "review-handover"
ASSIGNEE = "worker"
BRANCH = "feature/handover"
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


def _patch(line: str) -> bytes:
    """A patch that turns the owned module's one line into ``line``."""
    return (
        f"diff --git a/{MODULE} b/{MODULE}\n"
        f"--- a/{MODULE}\n"
        f"+++ b/{MODULE}\n"
        "@@ -1 +1 @@\n"
        f"-{BASE_LINE}\n"
        f"+{line}\n"
    ).encode("utf-8")


def _content_after(patch: bytes) -> str:
    """The module's whole content once ``patch`` has replaced its every line."""
    return "".join(
        f"{line[1:]}\n"
        for line in patch.decode("utf-8").splitlines()
        if line.startswith("+") and not line.startswith("+++")
    )


def _upload(
    build: dict,
    name: str,
    data: bytes,
    *,
    content_type: str = "text/x-diff",
    by: str = "agent",
    on_run: bool = True,
) -> int:
    """Store ``data`` through the kernel's one attachment write path.

    The board is opened the way the worker's own calls open it, from the
    worker's environment with no board or path named, so the file lands in
    whichever attachment store that environment resolves to.
    """
    conn = kb.connect()
    try:
        return kb.store_attachment_bytes(
            conn, build["task"], name, data,
            content_type=content_type, uploaded_by=by,
            expected_run_id=build["run"] if on_run else None,
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
    """Carry on as the worker: this process now has exactly ``env``."""
    for name in list(os.environ):
        if name not in env:
            monkeypatch.delenv(name)
    for name, value in env.items():
        if os.environ.get(name) != value:
            monkeypatch.setenv(name, value)


@pytest.fixture
def fenced_root(tmp_path, monkeypatch):
    """An empty Hermes root with no board, no register and no archive.

    The same start as the shared ``fence_home`` fixture, so the board that
    exists later is one a real kernel call created during the test. The
    attachment-root override (which the dispatcher never injects) and any
    session id this process carries are cleared as well.
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
    """A claimed worktree build, with this process running as its worker.

    Set up the way the dispatcher sets one up (claim, resolve the work area,
    record its branch and base); then the test takes on exactly the
    environment ``_default_spawn`` gives the worker it launches.
    """
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "--initial-branch=main")
    (repo / OWNED).mkdir(parents=True)
    (repo / MODULE).write_text(f"{BASE_LINE}\n", encoding="utf-8")
    _git(repo, "add", MODULE)
    _git(repo, "commit", "-m", "owned module", author=SEED_AUTHOR)

    kb.create_board(SLUG)
    entry = kb.get_register_entry(SLUG)
    assert entry is not None and entry.lifecycle is kb.BoardLifecycle.LIVE
    conn = kb.connect(board=SLUG)
    try:
        task_id = kb.create_task(
            conn, title="build that asks for review", assignee=ASSIGNEE,
            workspace_kind="worktree", workspace_path=str(repo),
            branch_name=BRANCH, owned_paths=[OWNED], requires_review=True,
        )
        claimed = kb.claim_task(conn, task_id)
        assert claimed is not None and claimed.current_run_id is not None
        workspace, branch = kb._resolve_worktree_workspace(claimed, board=SLUG)
        kb.set_workspace_path(conn, task_id, str(workspace))
        kb.set_branch_name(conn, task_id, branch)
        base = kb.record_worktree_base(conn, task_id, workspace)
        # The worker's profile exists, so its HERMES_HOME is that profile.
        (fenced_root / "profiles" / ASSIGNEE).mkdir(parents=True)

        env = _spawned_worker_env(claimed, workspace, monkeypatch)

        # Pinned to its own card, run and board, under a profile home below
        # the root, and given nothing more: no attachment-root override.
        assert env["HERMES_KANBAN_TASK"] == task_id
        assert env["HERMES_KANBAN_RUN_ID"] == str(claimed.current_run_id)
        assert env["HERMES_KANBAN_BOARD"] == SLUG
        assert env["HERMES_KANBAN_DB"] == str(
            fenced_root / "kanban" / "boards" / SLUG / "kanban.db"
        )
        assert env["HERMES_HOME"] == str(fenced_root / "profiles" / ASSIGNEE)
        assert "HERMES_KANBAN_ATTACHMENTS_ROOT" not in env
        for name in (
            "HERMES_HOME",
            "HERMES_KANBAN_DB",
            "HERMES_KANBAN_WORKSPACES_ROOT",
            "HERMES_KANBAN_WORKSPACE",
        ):
            assert Path(env[name]).resolve().is_relative_to(tmp_path.resolve()), name

        _become(env, monkeypatch)
        yield {
            "conn": conn,
            "task": task_id,
            "run": claimed.current_run_id,
            "repo": repo,
            "workspace": Path(workspace),
            "branch": branch,
            "base": base,
            "root": fenced_root,
        }
    finally:
        conn.close()


def test_one_saved_patch_is_handed_over_to_review(worker_build):
    """The run's one saved patch lands on the task branch and the card parks."""
    from tools import kanban_tools as kt

    build = worker_build
    conn, task_id, run_id = build["conn"], build["task"], build["run"]
    patch = _patch("value = 2")
    patch_id = _upload(build, "change.patch", patch)
    # Saved in the worker's own board's attachment store, by its own call.
    stored = Path(kb.get_attachment(conn, patch_id).stored_path)
    assert stored.parent == (
        build["root"] / "kanban" / "boards" / SLUG / "attachments" / task_id
    ).resolve()

    out = json.loads(kt._handle_request_review({"summary": SUMMARY}))

    assert out.get("ok") is True, out
    assert out["status"] == "review"
    assert out["handed_over_patch_attachment_id"] == patch_id
    assert kb.get_task(conn, task_id).status == "review"
    # Read back with git, not from the kernel: the branch moved off its base
    # and now holds exactly what the patch says.
    assert _branch_head(build) != build["base"]
    changed = _git(build["repo"], "show", f"{build['branch']}:{MODULE}")
    assert changed == _content_after(patch) == "value = 2\n"
    assert kb.saved_patch_ids_for_review(conn, task_id, run_id) == [patch_id]


def test_without_a_saved_patch_the_review_request_is_unchanged(worker_build):
    """No saved patch: review is requested for what the worker committed."""
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

    out = json.loads(kt._handle_request_review({"summary": SUMMARY}))

    assert out.get("ok") is True, out
    assert out["status"] == "review"
    assert "handed_over_patch_attachment_id" not in out
    assert kb.get_task(conn, task_id).status == "review"
    assert _branch_head(build) == worker_commit
    assert kb.saved_patch_ids_for_review(conn, task_id, run_id) == []


def test_several_saved_patches_are_reported_and_no_review_is_requested(worker_build):
    """Two saved patches: the build is told which, and nothing moves."""
    from tools import kanban_tools as kt

    build = worker_build
    conn, task_id, run_id = build["conn"], build["task"], build["run"]
    # What the tool answers in this same state when no patch was saved.
    no_patch = json.loads(kt._handle_request_review({"summary": SUMMARY}))
    first = _upload(build, "first.patch", _patch("value = 2"))
    second = _upload(build, "second.patch", _patch("value = 3"))

    out = json.loads(kt._handle_request_review({"summary": SUMMARY}))
    message = out.get("error") or ""

    assert message.startswith(
        f"could not request review for {task_id}: 2 saved patches were found"
    ), out
    assert f"attachment ids {first}, {second}" in message
    assert "none was handed over" in message
    assert "no review was requested" in message
    assert out != no_patch
    assert "ok" not in out and "status" not in out
    assert "handed_over_patch_attachment_id" not in out
    # Nothing changed: the same card is running on the same open run, it was
    # never sent to review, and the branch and work area are untouched.
    task = kb.get_task(conn, task_id)
    assert (task.status, task.current_run_id) == ("running", run_id)
    run = kb.get_run(conn, run_id)
    assert (run.status, run.ended_at) == ("running", None)
    assert [
        event.id for event in kb.list_events(conn, task_id)
        if event.kind == "review_requested"
    ] == []
    assert _branch_head(build) == build["base"]
    assert _git(build["workspace"], "status", "--porcelain") == ""
    assert kb.saved_patch_ids_for_review(conn, task_id, run_id) == [first, second]


def test_the_wrapper_and_the_quiet_exit_handover_agree_on_one_run(worker_build):
    """One copy of the rules: both callers see the same candidates, step by step."""
    build = worker_build
    conn, task_id, run_id = build["conn"], build["task"], build["run"]
    # Attachments the rules leave out, each stored by a real kernel call: the
    # run's own upload that is not a patch, a patch someone other than the
    # run's agent uploaded, and an agent patch bound to no run.
    _upload(build, "notes.txt", b"ran the unit tests\n", content_type="text/plain")
    _upload(build, "owner.patch", _patch("value = 7"), by="owner")
    _upload(build, "stray.patch", _patch("value = 8"), on_run=False)
    assert len(kb.list_attachments(conn, task_id)) == 3

    quiet_exit = {"evidence": "deliverable_present"}
    qualifying = []
    for step in range(3):
        if step:
            qualifying.append(
                _upload(build, f"change-{step}.patch", _patch(f"value = {step + 1}"))
            )
        ids = kb.saved_patch_ids_for_review(conn, task_id, run_id)
        assert ids == qualifying
        handed = kb._saved_patch_for_review_handover(conn, task_id, run_id, quiet_exit)
        assert handed == (ids[0] if len(ids) == 1 else None)
        if len(ids) == 1:
            # The quiet-exit gate still belongs to the handover alone.
            assert kb._saved_patch_for_review_handover(
                conn, task_id, run_id, {"evidence": "session_terminal_intent"},
            ) is None
    assert len(qualifying) == 2
