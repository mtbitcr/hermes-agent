"""A build run that saved its patch but never reported is handed to review.

A worker that attached its patch as its own attachment and then exited cleanly
without its final kanban call used to be parked for a person, who handed the
patch over by hand. The dead-worker scan now hands that one saved patch to
review itself, through the handover that already exists (``complete_task`` with
the run's own patch: materialize, prove the scope, park in the review lane),
and parks exactly as before whenever that handover refuses.

Every scenario is built from shipped calls against a real temporary git
repository: a fenced board, a worktree task, a host-local claim, the resolved
work area, the recorded base and an agent upload bound to the run. The worker is
then given the dead-worker shape the scan classifies as a clean exit: its pid is
gone and the reap registry saw it exit 0.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_WORKTREE = Path(__file__).resolve().parents[2]
if str(_WORKTREE) not in sys.path:
    sys.path.insert(0, str(_WORKTREE))

from hermes_cli import kanban_db as kb
from tests.hermes_cli._kanban_fence_support import (
    DEFAULT_GIT_IDENTITY, create_fenced_board, git, make_git_repo,
)

SLUG = "saved-results"
OWNED = "src/owned"
WORKER_PID = 70999
SUMMARY = (
    "The kernel handed the saved patch to review because the worker exited "
    "without reporting its result."
)


def _new_file_patch(path: str, line: str) -> bytes:
    """A patch that creates ``path`` holding one line, with no trailing whitespace."""
    return (
        f"diff --git a/{path} b/{path}\n"
        "new file mode 100644\n"
        "--- /dev/null\n"
        f"+++ b/{path}\n"
        "@@ -0,0 +1 @@\n"
        f"+{line}\n"
    ).encode("utf-8")


@pytest.fixture
def running_build(fence_home, tmp_path, monkeypatch):
    """Make the claimed, running worktree build every scenario starts from."""
    opened = []

    def make(*, requires_review: bool = True) -> dict:
        repo = tmp_path / "repo"
        make_git_repo(repo)
        # An owned module already at the base, so a patch written against some
        # other version of it genuinely does not apply.
        (repo / OWNED).mkdir(parents=True)
        (repo / OWNED / "app.py").write_text("value = 1\n", encoding="utf-8")
        git(repo, "add", f"{OWNED}/app.py")
        git(repo, "commit", "-m", "owned module", author=DEFAULT_GIT_IDENTITY)
        create_fenced_board(SLUG)
        monkeypatch.setenv("HERMES_KANBAN_BOARD", SLUG)
        monkeypatch.setenv("HERMES_KANBAN_ATTACHMENTS_ROOT", str(tmp_path / "at"))
        conn = kb.connect(board=SLUG)
        opened.append(conn)
        task_id = kb.create_task(
            conn, title="build that saves its patch", assignee="worker",
            workspace_kind="worktree", workspace_path=str(repo),
            branch_name="feature/saved", owned_paths=[OWNED],
            requires_review=requires_review,
        )
        # The scan only judges claims made on this host.
        host = kb._claimer_id().split(":", 1)[0]
        claimed = kb.claim_task(conn, task_id, claimer=f"{host}:w0")
        workspace, branch = kb._resolve_worktree_workspace(claimed)
        kb.set_workspace_path(conn, task_id, workspace)
        kb.set_branch_name(conn, task_id, branch)
        return {
            "conn": conn, "task": task_id, "run": claimed.current_run_id,
            "repo": repo, "branch": branch,
            "base": kb.record_worktree_base(conn, task_id, workspace),
        }

    yield make
    for conn in opened:
        conn.close()


@pytest.fixture
def exited_workers():
    """Every ``on_kanban_worker_exited`` payload, through the real hook registry."""
    from hermes_cli.plugins import get_plugin_manager

    manager = get_plugin_manager()
    saved = {name: list(callbacks) for name, callbacks in manager._hooks.items()}
    fired: list[dict] = []
    manager._hooks.setdefault("on_kanban_worker_exited", []).append(
        lambda **fields: fired.append(fields)
    )
    try:
        yield fired
    finally:
        manager._hooks = saved


def _save_patch(setup: dict, name: str, data: bytes) -> int:
    """The run's own upload, stored the way the attach tool stores it."""
    return kb.store_attachment_bytes(
        setup["conn"], setup["task"], name, data,
        content_type="text/x-diff", uploaded_by="agent",
        expected_run_id=setup["run"],
    )


def _exit_cleanly_without_reporting(setup: dict, monkeypatch) -> None:
    """The worker is gone and the reap registry saw it exit 0 (no terminal call)."""
    monkeypatch.setattr(kb, "_pid_alive", lambda _pid: False)
    monkeypatch.setenv("HERMES_KANBAN_CRASH_GRACE_SECONDS", "0")
    # The registry is process memory; a fresh one keeps scenarios independent.
    monkeypatch.setattr(kb, "_recent_worker_exits", {})
    kb._set_worker_pid(setup["conn"], setup["task"], WORKER_PID)
    kb._record_worker_exit(WORKER_PID, 0 << 8)


def _handed_over(setup: dict) -> list:
    return [
        (event.run_id, event.payload)
        for event in kb.list_events(setup["conn"], setup["task"])
        if event.kind == "saved_result_handed_over"
    ]


def _assert_parked_for_a_person(setup: dict) -> dict:
    """Today's park, read back from durable state.

    Returns the run's ``unreported_completion`` record.
    """
    conn, task_id, run_id = setup["conn"], setup["task"], setup["run"]
    task = kb.get_task(conn, task_id)
    assert (task.status, task.block_kind) == ("blocked", "needs_input")
    run = kb.get_run(conn, run_id)
    assert run.ended_at is not None
    assert run.outcome == "completed_unreported"
    events = kb.list_events(conn, task_id)
    assert [e.run_id for e in events if e.kind == "protocol_violation"] == [run_id]
    assert [e.run_id for e in events if e.kind == "blocked"] == [run_id]
    assert _handed_over(setup) == []
    # Nothing of the saved patch reached the task branch.
    assert git(setup["repo"], "rev-parse", setup["branch"]) == setup["base"]
    unreported = run.metadata["unreported_completion"]
    assert unreported["evidence"] == "deliverable_present"
    return unreported


def test_saved_patch_is_handed_to_review_when_the_worker_exits_without_reporting(
    running_build, exited_workers, monkeypatch,
):
    setup = running_build()
    conn, task_id, run_id = setup["conn"], setup["task"], setup["run"]
    attachment = _save_patch(
        setup, "x.patch", _new_file_patch(f"{OWNED}/saved.py", "saved = True"),
    )
    _exit_cleanly_without_reporting(setup, monkeypatch)

    assert kb.detect_crashed_workers(conn) == []

    task = kb.get_task(conn, task_id)
    assert task.status == "review"
    assert task.completed_at is None
    # The head the review park holds is the kernel commit that materialized
    # exactly that attachment.
    head = git(setup["repo"], "rev-parse", setup["branch"])
    assert f"Hermes-Patch-Attachment: {attachment}" in git(
        setup["repo"], "log", "-1", "--format=%B", head,
    )
    assert git(setup["repo"], "show", f"{head}:{OWNED}/saved.py") == "saved = True"
    assert kb._latest_review_head_provenance(conn, task_id) == head
    run = kb.get_run(conn, run_id)
    assert run.ended_at is not None
    assert run.summary == SUMMARY
    assert run.metadata["execution_receipt"]["head_commit"] == head
    assert _handed_over(setup) == [
        (run_id, {"run_id": run_id, "attachment_id": attachment}),
    ]
    assert [
        (fields["task_id"], fields["run_id"], fields["outcome"], fields["retry_status"])
        for fields in exited_workers
    ] == [(task_id, run_id, "review_requested", "review")]


def test_patch_outside_the_owned_paths_still_parks_with_the_refusal_on_the_run(
    running_build, exited_workers, monkeypatch,
):
    setup = running_build()
    _save_patch(setup, "x.patch", _new_file_patch("src/outside.py", "outside = True"))
    _exit_cleanly_without_reporting(setup, monkeypatch)

    assert kb.detect_crashed_workers(setup["conn"]) == []

    unreported = _assert_parked_for_a_person(setup)
    assert "outside declared ownership" in unreported["handover_refusal"]
    assert [
        (fields["run_id"], fields["outcome"], fields["retry_status"])
        for fields in exited_workers
    ] == [(setup["run"], "completed_unreported", "blocked")]


def test_patch_that_does_not_apply_to_the_base_still_parks_with_the_refusal_on_the_run(
    running_build, monkeypatch,
):
    setup = running_build()
    # Written against a version of the owned module the base never had.
    _save_patch(setup, "x.patch", (
        f"diff --git a/{OWNED}/app.py b/{OWNED}/app.py\n"
        f"--- a/{OWNED}/app.py\n"
        f"+++ b/{OWNED}/app.py\n"
        "@@ -1 +1 @@\n"
        "-value = 2\n"
        "+value = 3\n"
    ).encode("utf-8"))
    _exit_cleanly_without_reporting(setup, monkeypatch)

    kb.detect_crashed_workers(setup["conn"])

    unreported = _assert_parked_for_a_person(setup)
    assert "apply" in unreported["handover_refusal"]


def test_card_without_a_review_requirement_still_parks(running_build, monkeypatch):
    setup = running_build(requires_review=False)
    _save_patch(setup, "x.patch", _new_file_patch(f"{OWNED}/saved.py", "saved = True"))
    _exit_cleanly_without_reporting(setup, monkeypatch)

    kb.detect_crashed_workers(setup["conn"])

    unreported = _assert_parked_for_a_person(setup)
    assert "handover_refusal" not in unreported


def test_run_with_two_patches_still_parks(running_build, monkeypatch):
    setup = running_build()
    _save_patch(setup, "x.patch", _new_file_patch(f"{OWNED}/one.py", "one = 1"))
    _save_patch(setup, "y.patch", _new_file_patch(f"{OWNED}/two.py", "two = 2"))
    _exit_cleanly_without_reporting(setup, monkeypatch)

    kb.detect_crashed_workers(setup["conn"])

    unreported = _assert_parked_for_a_person(setup)
    assert unreported["attachments_at_exit"] == 2
    assert "handover_refusal" not in unreported


def test_card_that_already_moved_on_is_never_overwritten(
    running_build, exited_workers, monkeypatch,
):
    """Another scan hands the saved patch over first; this scan's own handover
    is then refused, and the card stays exactly where the other scan put it."""
    setup = running_build()
    conn, task_id, run_id = setup["conn"], setup["task"], setup["run"]
    attachment = _save_patch(
        setup, "x.patch", _new_file_patch(f"{OWNED}/saved.py", "saved = True"),
    )
    _exit_cleanly_without_reporting(setup, monkeypatch)
    real_complete_task = kb.complete_task

    def another_scan_gets_there_first(*args, **kwargs):
        monkeypatch.setattr(kb, "complete_task", real_complete_task)
        kb.detect_crashed_workers(conn)
        return real_complete_task(*args, **kwargs)

    monkeypatch.setattr(kb, "complete_task", another_scan_gets_there_first)
    kb.detect_crashed_workers(conn)

    task = kb.get_task(conn, task_id)
    assert (task.status, task.block_kind) == ("review", None)
    run = kb.get_run(conn, run_id)
    assert run.outcome == "review_requested"
    assert run.summary == SUMMARY
    # Only the refusal reason joined what the landed handover recorded.
    assert run.metadata["execution_receipt"]["head_commit"] == git(
        setup["repo"], "rev-parse", setup["branch"],
    )
    assert run.metadata["unreported_completion"]["handover_refusal"]
    kinds = [event.kind for event in kb.list_events(conn, task_id)]
    assert "blocked" not in kinds
    assert "protocol_violation" not in kinds
    assert _handed_over(setup) == [
        (run_id, {"run_id": run_id, "attachment_id": attachment}),
    ]
    assert [fields["outcome"] for fields in exited_workers] == ["review_requested"]


def test_run_set_aside_before_the_process_stopped_is_handed_over_by_the_next_scan(
    running_build, monkeypatch,
):
    setup = running_build()
    conn, task_id, run_id = setup["conn"], setup["task"], setup["run"]
    attachment = _save_patch(
        setup, "x.patch", _new_file_patch(f"{OWNED}/saved.py", "saved = True"),
    )
    _exit_cleanly_without_reporting(setup, monkeypatch)
    real_complete_task = kb.complete_task

    class ProcessStopped(BaseException):
        """The process going away after the scan committed, before the handover."""

    def stop(*_args, **_kwargs):
        raise ProcessStopped

    monkeypatch.setattr(kb, "complete_task", stop)
    with pytest.raises(ProcessStopped):
        kb.detect_crashed_workers(conn)

    # Still running under its dead worker, the run open and marked for handover.
    task = kb.get_task(conn, task_id)
    assert (task.status, task.current_run_id, task.worker_pid) == (
        "running", run_id, WORKER_PID,
    )
    run = kb.get_run(conn, run_id)
    assert run.ended_at is None
    assert run.metadata["unreported_completion"]["handover_attachment_id"] == attachment

    # A restarted process: the real handover, and a reap registry that no
    # longer remembers how the worker exited.
    monkeypatch.setattr(kb, "complete_task", real_complete_task)
    monkeypatch.setattr(kb, "_recent_worker_exits", {})

    assert kb.detect_crashed_workers(conn) == []

    assert kb.get_task(conn, task_id).status == "review"
    assert _handed_over(setup) == [
        (run_id, {"run_id": run_id, "attachment_id": attachment}),
    ]
    kinds = [event.kind for event in kb.list_events(conn, task_id)]
    assert "crashed" not in kinds
    assert "blocked" not in kinds
