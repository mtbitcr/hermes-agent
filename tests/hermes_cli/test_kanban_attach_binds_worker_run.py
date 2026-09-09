"""A worker's attachments must be bound to the run that produced them.

``_run_handover_artifacts`` decides whether an expiring run already delivered
a finished handover by reading the run-scoped ``attached`` receipt, and by
nothing else. An attachment stored with ``run_id = NULL`` therefore does not
exist as far as any budget exit is concerned: the run is recorded
``timed_out``, the task goes back to ``ready`` with no ``head_commit``, and
the finished work is rebuilt from scratch.

So the binding has to hold at the surfaces a worker actually uses — the
registered ``kanban_attach`` / ``kanban_attach_url`` model tools and the
``hermes kanban attach`` command line — not merely in
``store_attachment_bytes`` when a caller remembers to pass
``expected_run_id``. These tests attach through those real entry points with
the dispatcher's worker environment set, then drive the REAL budget finalizer
and require a completion with the materialized patch.
"""

from __future__ import annotations

import base64
import contextlib
import json
import logging
import subprocess
from pathlib import Path

import pytest

from hermes_cli import kanban as kc
from hermes_cli import kanban_db as kb


def _git(cwd: Path, *args: str) -> str:
    result = subprocess.run(
        [
            "git", "-C", str(cwd),
            "-c", "user.name=Test User",
            "-c", "user.email=test@example.com",
            "-c", "commit.gpgsign=false",
            *args,
        ],
        check=True, capture_output=True, text=True,
    )
    return result.stdout.strip()


def _repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(
        ["git", "init", "-b", "main", str(repo)],
        check=True, capture_output=True, text=True,
    )
    (repo / "README.md").write_text("base\n", encoding="utf-8")
    _git(repo, "add", "README.md")
    _git(repo, "commit", "-m", "init")
    return repo


def _new_file_patch(path: str, content: str) -> bytes:
    lines = content.splitlines(keepends=True)
    body = "".join(f"+{line}" for line in lines)
    return (
        f"diff --git a/{path} b/{path}\n"
        "new file mode 100644\n"
        "index 0000000..e69de29\n"
        "--- /dev/null\n"
        f"+++ b/{path}\n"
        f"@@ -0,0 +1,{len(lines)} @@\n"
        f"{body}"
    ).encode("utf-8")


@pytest.fixture
def default_board_home(tmp_path, monkeypatch):
    """A default board under a temp ``HERMES_HOME``.

    Both the registered tool handlers and
    ``agent.turn_finalizer._record_kanban_budget_exhausted`` open their own
    connections with no path and no board, so the temp store has to be what
    ``kanban_db.connect()`` resolves to. Nothing in the production path is
    redirected.
    """
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb._INITIALIZED_PATHS.clear()
    kb.init_db()
    return home


def _claimed_worktree_task(conn, repo: Path, *, title: str, branch: str) -> tuple:
    task_id = kb.create_task(
        conn, title=title, assignee="worker",
        workspace_kind="worktree", workspace_path=str(repo),
        branch_name=branch, owned_paths=["src/owned"],
        max_runtime_seconds=3600,
    )
    claimed = kb.claim_task(conn, task_id)
    assert claimed is not None
    workspace, resolved_branch = kb._resolve_worktree_workspace(claimed)
    kb.set_workspace_path(conn, task_id, workspace)
    kb.set_branch_name(conn, task_id, resolved_branch)
    kb.record_worktree_base(conn, task_id, workspace)
    task = kb.get_task(conn, task_id)
    assert task is not None and task.current_run_id is not None
    kb._set_worker_pid(conn, task_id, 556_001)
    return task_id, int(task.current_run_id)


def _worker_env(monkeypatch, task_id: str, run_id: int) -> None:
    monkeypatch.setenv("HERMES_KANBAN_TASK", task_id)
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(run_id))


def _registered_handler(name: str):
    """The handler the model actually calls, resolved from the real registry."""
    import tools.kanban_tools  # noqa: F401  (registers the kanban toolset)
    from tools.registry import registry

    entry = registry.get_entry(name)
    assert entry is not None, f"{name} is not registered"
    return entry.handler


def _receipt_run_ids(conn, task_id: str) -> list:
    out = []
    for row in conn.execute(
        "SELECT run_id FROM task_events WHERE task_id = ? AND kind = 'attached' "
        "ORDER BY id",
        (task_id,),
    ).fetchall():
        out.append(row["run_id"])
    return out


def _events(conn, tid, kind):
    return [
        r["kind"] for r in conn.execute(
            "SELECT kind FROM task_events WHERE task_id = ? AND kind = ? "
            "ORDER BY id",
            (tid, kind),
        ).fetchall()
    ]


# ---------------------------------------------------------------------------
# 1. The registered model tool
# ---------------------------------------------------------------------------


def test_attach_tool_handover_completes_on_budget_expiry(
    default_board_home, tmp_path, monkeypatch,
):
    """The motivating production bug, through the surface that produced it.

    A worker attaches its patch and its report with the registered
    ``kanban_attach`` handler, then its iteration budget runs out. The real
    turn-finalizer exit must find that handover and COMPLETE the task with
    the materialized patch instead of timing it out for a rebuild.
    """
    from agent.turn_finalizer import _record_kanban_budget_exhausted

    repo = _repo(tmp_path)
    with contextlib.closing(kb.connect()) as conn:
        task_id, run_id = _claimed_worktree_task(
            conn, repo, title="tool attach handover",
            branch="feature/tool-attach",
        )
    _worker_env(monkeypatch, task_id, run_id)

    attach = _registered_handler("kanban_attach")
    patch_result = json.loads(attach({
        "task_id": task_id,
        "filename": "sandbox.patch",
        "content_base64": base64.b64encode(
            _new_file_patch("src/owned/tool.py", "delivered = True\n")
        ).decode("ascii"),
        "content_type": "text/x-diff",
    }))
    assert patch_result.get("ok") is True, patch_result
    report_result = json.loads(attach({
        "task_id": task_id,
        "filename": "report.md",
        "content_text": "## Summary\nBudget ran out; the work is attached.\n",
        "content_type": "text/markdown",
    }))
    assert report_result.get("ok") is True, report_result

    # The real entry point for both budget-exhausted branches of
    # ``finalize_turn``; it opens its own connection and swallows exceptions,
    # so durable board state is the only evidence.
    _record_kanban_budget_exhausted(
        task_id, 200, 200, logging.getLogger("test.turn_finalizer"),
    )

    with contextlib.closing(kb.connect()) as conn:
        task = kb.get_task(conn, task_id)
        assert task is not None
        assert task.status == "done", (
            "a handover delivered through the real attach tool was invisible "
            f"to the budget finalizer; task is {task.status!r}"
        )
        assert task.head_commit, "completed with no git receipt"
        run = kb.latest_run(conn, task_id)
        assert run is not None
        assert run.outcome == "completed"
        assert _events(conn, task_id, "timed_out") == []
        assert _events(conn, task_id, "run_handover_completed")
        assert _receipt_run_ids(conn, task_id) == [run_id, run_id]
        head_commit = task.head_commit

    assert _git(repo, "show", f"{head_commit}:src/owned/tool.py") == "delivered = True"


def test_attach_url_tool_binds_the_producing_run(
    default_board_home, tmp_path, monkeypatch,
):
    """``kanban_attach_url`` is the same store, so it needs the same binding.

    The download is stubbed at the module's own fetch helper; everything
    downstream of it — the handler, the guard, the native store, the receipt —
    is the real path.
    """
    import tools.kanban_tools as kt

    repo = _repo(tmp_path)
    with contextlib.closing(kb.connect()) as conn:
        task_id, run_id = _claimed_worktree_task(
            conn, repo, title="attach-url handover",
            branch="feature/attach-url",
        )
    _worker_env(monkeypatch, task_id, run_id)

    monkeypatch.setattr(
        kt, "_download_url_with_cap",
        lambda url, cap: (b"## Summary\nFetched report.\n", "text/markdown"),
    )
    result = json.loads(_registered_handler("kanban_attach_url")({
        "task_id": task_id,
        "url": "https://example.invalid/report.md",
        "filename": "report.md",
    }))
    assert result.get("ok") is True, result

    with contextlib.closing(kb.connect()) as conn:
        assert _receipt_run_ids(conn, task_id) == [run_id], (
            "kanban_attach_url stored an attachment with no producing run"
        )
        attachment = kb.get_attachment(conn, result["attachment_id"])
        assert attachment is not None
        assert attachment.uploaded_by == "agent"


# ---------------------------------------------------------------------------
# 2. The worker command line
# ---------------------------------------------------------------------------


def test_cli_attach_binds_the_producing_run(
    default_board_home, tmp_path, monkeypatch,
):
    """``hermes kanban attach`` is the third worker attachment surface.

    A worker driving the command line has to get the same receipt the model
    tool does, or its handover is equally invisible to every budget exit.
    """
    repo = _repo(tmp_path)
    with contextlib.closing(kb.connect()) as conn:
        task_id, run_id = _claimed_worktree_task(
            conn, repo, title="cli attach handover",
            branch="feature/cli-attach",
        )
    _worker_env(monkeypatch, task_id, run_id)

    src = tmp_path / "report.md"
    src.write_text("## Summary\nCommand-line handover.\n", encoding="utf-8")
    output = kc.run_slash(f"attach {task_id} {src}")
    assert "Attached" in output, output

    with contextlib.closing(kb.connect()) as conn:
        assert _receipt_run_ids(conn, task_id) == [run_id], (
            "the CLI attach command stored an attachment with no producing run"
        )


def test_attach_outside_a_worker_run_still_stores_unbound(
    default_board_home, tmp_path, monkeypatch,
):
    """Regression guard: the binding comes from the dispatcher's environment.

    An operator uploading to a card from a plain shell has no run to bind to,
    and must still be able to attach — the run id is bound only when the
    dispatcher put this process on that exact task.
    """
    repo = _repo(tmp_path)
    with contextlib.closing(kb.connect()) as conn:
        task_id, _run_id = _claimed_worktree_task(
            conn, repo, title="operator upload", branch="feature/operator",
        )
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_RUN_ID", raising=False)

    src = tmp_path / "briefing.md"
    src.write_text("Operator briefing.\n", encoding="utf-8")
    output = kc.run_slash(f"attach {task_id} {src}")
    assert "Attached" in output, output

    with contextlib.closing(kb.connect()) as conn:
        assert _receipt_run_ids(conn, task_id) == [None]
