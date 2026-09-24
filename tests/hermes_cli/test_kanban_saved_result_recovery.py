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
import time
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


def _assert_parked_for_a_person(setup: dict, *, branch_head: str | None = None) -> dict:
    """Today's park, read back from durable state.

    The task branch is expected at ``branch_head``, by default the base.
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
    # Nothing of the saved patch reached the task branch, unless the handover
    # had committed it before the review transition was refused.
    assert git(setup["repo"], "rev-parse", setup["branch"]) == (
        setup["base"] if branch_head is None else branch_head
    )
    unreported = run.metadata["unreported_completion"]
    assert unreported["evidence"] == "deliverable_present"
    return unreported


class _ProcessStopped(BaseException):
    """The process going away after the scan committed, before the handover."""


def _set_aside_then_restart(setup: dict, monkeypatch) -> int:
    """The scan sets one saved patch aside and the process stops before the
    handover; then a new process starts.

    The restarted process has the real handover and a reap registry that no
    longer remembers how the worker exited. Returns the attachment id.
    """
    attachment = _save_patch(
        setup, "x.patch", _new_file_patch(f"{OWNED}/saved.py", "saved = True"),
    )
    _exit_cleanly_without_reporting(setup, monkeypatch)
    real_complete_task = kb.complete_task

    def stop(*_args, **_kwargs):
        raise _ProcessStopped

    monkeypatch.setattr(kb, "complete_task", stop)
    with pytest.raises(_ProcessStopped):
        kb.detect_crashed_workers(setup["conn"])
    monkeypatch.setattr(kb, "complete_task", real_complete_task)
    monkeypatch.setattr(kb, "_recent_worker_exits", {})
    return attachment


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


@pytest.mark.parametrize("lapse", ["expired_claim", "stale_run"])
def test_run_set_aside_before_the_process_stopped_is_handed_over_by_the_next_dispatcher_tick(
    running_build, monkeypatch, lapse,
):
    """However long the process stayed down, the sweeps that run before the
    scan in ``dispatch_once`` leave the set-aside run to it."""
    setup = running_build()
    conn, task_id, run_id = setup["conn"], setup["task"], setup["run"]
    attachment = _set_aside_then_restart(setup, monkeypatch)
    long_ago = int(time.time()) - 7200
    with kb.write_txn(conn):
        if lapse == "expired_claim":
            conn.execute(
                "UPDATE tasks SET claim_expires = ? WHERE id = ?", (long_ago, task_id),
            )
            conn.execute(
                "UPDATE task_runs SET claim_expires = ? WHERE id = ?", (long_ago, run_id),
            )
        else:
            # Running for two hours, and the worker never sent a heartbeat.
            conn.execute(
                "UPDATE task_runs SET started_at = ? WHERE id = ?", (long_ago, run_id),
            )

    result = kb.dispatch_once(
        conn, max_spawn=0, board=SLUG,
        stale_timeout_seconds=60 if lapse == "stale_run" else 0,
    )

    assert (result.reclaimed, result.stale) == (0, [])
    assert kb.get_task(conn, task_id).status == "review"
    run = kb.get_run(conn, run_id)
    assert run.ended_at is not None
    assert (run.outcome, run.summary) == ("review_requested", SUMMARY)
    assert _handed_over(setup) == [
        (run_id, {"run_id": run_id, "attachment_id": attachment}),
    ]
    kinds = [event.kind for event in kb.list_events(conn, task_id)]
    assert "reclaimed" not in kinds
    assert "stale" not in kinds


def test_review_claimed_before_the_audit_keeps_the_handover_event_on_the_implementation_run(
    running_build, exited_workers, monkeypatch,
):
    """A reviewer may claim the parked review the moment the handover commits.
    The kernel's receipt still names the implementation run and its patch, and
    the reviewer's run is left exactly as its claim made it."""
    setup = running_build()
    conn, task_id, run_id = setup["conn"], setup["task"], setup["run"]
    attachment = _save_patch(
        setup, "x.patch", _new_file_patch(f"{OWNED}/saved.py", "saved = True"),
    )
    _exit_cleanly_without_reporting(setup, monkeypatch)
    real_complete_task = kb.complete_task
    host = kb._claimer_id().split(":", 1)[0]
    review: dict = {}

    def bound_to(bound_run_id):
        return [
            event for event in kb.list_events(conn, task_id)
            if event.run_id == bound_run_id
        ]

    def a_reviewer_claims_the_review_at_once(*args, **kwargs):
        handed_over = real_complete_task(*args, **kwargs)
        if handed_over:
            claimed = kb.claim_review_task(conn, task_id, claimer=f"{host}:r0")
            review["task"] = claimed
            if claimed is not None:
                review["run"] = kb.get_run(conn, claimed.current_run_id)
                review["events"] = bound_to(claimed.current_run_id)
        return handed_over

    monkeypatch.setattr(kb, "complete_task", a_reviewer_claims_the_review_at_once)
    assert kb.detect_crashed_workers(conn) == []

    # The handover returned True and the reviewer's claim succeeded.
    assert review.get("task") is not None
    reviewer_run_id = review["task"].current_run_id
    assert reviewer_run_id != run_id
    task = kb.get_task(conn, task_id)
    assert (task.status, task.current_run_id) == ("running", reviewer_run_id)
    assert _handed_over(setup) == [
        (run_id, {"run_id": run_id, "attachment_id": attachment}),
    ]
    # Still open, and nothing of the handover landed on it.
    reviewer_run = kb.get_run(conn, reviewer_run_id)
    assert reviewer_run.ended_at is None
    assert reviewer_run == review["run"]
    assert bound_to(reviewer_run_id) == review["events"]
    assert kb.get_run(conn, run_id).outcome == "review_requested"
    assert [
        (fields["run_id"], fields["outcome"]) for fields in exited_workers
    ] == [(run_id, "review_requested")]


def test_runtime_cap_leaves_a_set_aside_run_to_the_next_scan(running_build, monkeypatch):
    """The runtime cap runs after the scan in ``dispatch_once``: a handover that
    has not happened yet is the next scan's to retry, not a timeout."""
    setup = running_build()
    conn, task_id, run_id = setup["conn"], setup["task"], setup["run"]
    attachment = _set_aside_then_restart(setup, monkeypatch)
    with kb.write_txn(conn):
        # A one-minute cap on a run that started two hours ago.
        conn.execute(
            "UPDATE tasks SET max_runtime_seconds = 60 WHERE id = ?", (task_id,),
        )
        conn.execute(
            "UPDATE task_runs SET started_at = ? WHERE id = ?",
            (int(time.time()) - 7200, run_id),
        )

    assert kb.enforce_max_runtime(conn) == []

    task = kb.get_task(conn, task_id)
    assert (task.status, task.current_run_id, task.worker_pid) == (
        "running", run_id, WORKER_PID,
    )
    run = kb.get_run(conn, run_id)
    assert run.ended_at is None
    assert run.metadata["unreported_completion"]["handover_attachment_id"] == attachment

    assert kb.detect_crashed_workers(conn) == []

    assert kb.get_task(conn, task_id).status == "review"
    assert _handed_over(setup) == [
        (run_id, {"run_id": run_id, "attachment_id": attachment}),
    ]
    assert "timed_out" not in [event.kind for event in kb.list_events(conn, task_id)]


def _lapse_before_any_scan(setup: dict, lapse: str) -> int:
    """The claim expires, or the run goes two hours without a heartbeat,
    before any scan has set the run aside.

    Returns the ``stale_timeout_seconds`` the dispatcher tick runs with.
    """
    conn, task_id, run_id = setup["conn"], setup["task"], setup["run"]
    # No scan has run yet, so nothing marks the run for the handover.
    assert "unreported_completion" not in (kb.get_run(conn, run_id).metadata or {})
    long_ago = int(time.time()) - 7200
    with kb.write_txn(conn):
        if lapse == "expired_claim":
            conn.execute(
                "UPDATE tasks SET claim_expires = ? WHERE id = ?", (long_ago, task_id),
            )
            conn.execute(
                "UPDATE task_runs SET claim_expires = ? WHERE id = ?", (long_ago, run_id),
            )
        else:
            # Running for two hours, and the worker never sent a heartbeat.
            conn.execute(
                "UPDATE task_runs SET started_at = ? WHERE id = ?", (long_ago, run_id),
            )
    return 60 if lapse == "stale_run" else 0


@pytest.mark.parametrize("lapse", ["expired_claim", "stale_run"])
def test_run_not_yet_set_aside_is_handed_over_by_the_first_dispatcher_tick(
    running_build, monkeypatch, lapse,
):
    """The claim or the heartbeat can lapse before any scan has set the run
    aside; the sweeps that run before the scan in ``dispatch_once`` still
    leave it to the scan on the same tick."""
    setup = running_build()
    conn, task_id, run_id = setup["conn"], setup["task"], setup["run"]
    attachment = _save_patch(
        setup, "x.patch", _new_file_patch(f"{OWNED}/saved.py", "saved = True"),
    )
    _exit_cleanly_without_reporting(setup, monkeypatch)
    stale_timeout_seconds = _lapse_before_any_scan(setup, lapse)

    result = kb.dispatch_once(
        conn, max_spawn=0, board=SLUG, stale_timeout_seconds=stale_timeout_seconds,
    )

    assert (result.reclaimed, result.stale) == (0, [])
    assert kb.get_task(conn, task_id).status == "review"
    run = kb.get_run(conn, run_id)
    assert run.ended_at is not None
    assert (run.outcome, run.summary) == ("review_requested", SUMMARY)
    assert _handed_over(setup) == [
        (run_id, {"run_id": run_id, "attachment_id": attachment}),
    ]
    kinds = [event.kind for event in kb.list_events(conn, task_id)]
    assert "reclaimed" not in kinds
    assert "stale" not in kinds


def test_process_stopped_right_after_the_review_transition_still_leaves_one_handover_event(
    running_build, monkeypatch,
):
    """The handover's event commits with the review transition itself: a
    process that stops the moment the card is in review has recorded it, and
    the restarted process does not record it again."""
    setup = running_build()
    conn, task_id, run_id = setup["conn"], setup["task"], setup["run"]
    attachment = _save_patch(
        setup, "x.patch", _new_file_patch(f"{OWNED}/saved.py", "saved = True"),
    )
    _exit_cleanly_without_reporting(setup, monkeypatch)
    real_request_review = kb.request_review

    def stop_once_in_review(*args, **kwargs):
        parked = real_request_review(*args, **kwargs)
        if parked:
            raise _ProcessStopped
        return parked

    monkeypatch.setattr(kb, "request_review", stop_once_in_review)
    with pytest.raises(_ProcessStopped):
        kb.detect_crashed_workers(conn)

    assert kb.get_task(conn, task_id).status == "review"
    handed_over = [(run_id, {"run_id": run_id, "attachment_id": attachment})]
    assert _handed_over(setup) == handed_over

    # A restarted process: the real review transition, and a reap registry
    # that no longer remembers how the worker exited.
    monkeypatch.setattr(kb, "request_review", real_request_review)
    monkeypatch.setattr(kb, "_recent_worker_exits", {})

    assert kb.detect_crashed_workers(conn) == []

    assert kb.get_task(conn, task_id).status == "review"
    assert _handed_over(setup) == handed_over


@pytest.mark.parametrize("lapse", ["expired_claim", "stale_run"])
def test_dirty_worktree_still_parks_with_the_refusal_on_the_first_dispatcher_tick(
    running_build, monkeypatch, lapse,
):
    """On that same first tick, a handover the worktree refuses still parks
    the card for a person, with the refusal on the run."""
    setup = running_build()
    conn, task_id = setup["conn"], setup["task"]
    _save_patch(setup, "x.patch", _new_file_patch(f"{OWNED}/saved.py", "saved = True"))
    _exit_cleanly_without_reporting(setup, monkeypatch)
    # An untracked file left behind in the task's own worktree.
    workspace = Path(kb.get_task(conn, task_id).workspace_path)
    (workspace / OWNED / "leftover.py").write_text("leftover = True\n", encoding="utf-8")
    stale_timeout_seconds = _lapse_before_any_scan(setup, lapse)

    kb.dispatch_once(
        conn, max_spawn=0, board=SLUG, stale_timeout_seconds=stale_timeout_seconds,
    )

    unreported = _assert_parked_for_a_person(setup)
    assert "dirty" in unreported["handover_refusal"]
    kinds = [event.kind for event in kb.list_events(conn, task_id)]
    assert "reclaimed" not in kinds
    assert "stale" not in kinds


def _parent_reopens_before_the_review_transition(
    setup: dict, monkeypatch, *, meanwhile=None,
) -> int:
    """A done parent of the build reopens after the handover has committed the
    saved patch to the task branch, before the review transition runs.

    ``meanwhile``, when given, runs first, while the branch holds that commit.
    Returns the attachment id.
    """
    conn = setup["conn"]
    parent_id = kb.create_task(conn, title="parent of the build", assignee="planner")
    assert kb.complete_task(conn, parent_id)
    assert kb.get_task(conn, parent_id).status == "done"
    kb.link_tasks(conn, parent_id, setup["task"])
    attachment = _save_patch(
        setup, "x.patch", _new_file_patch(f"{OWNED}/saved.py", "saved = True"),
    )
    _exit_cleanly_without_reporting(setup, monkeypatch)
    real_request_review = kb.request_review

    def reopen_the_parent_first(*args, **kwargs):
        assert f"Hermes-Patch-Attachment: {attachment}" in git(
            setup["repo"], "log", "-1", "--format=%B", setup["branch"],
        )
        if meanwhile is not None:
            meanwhile()
        # The minimal stand-in for a reopen surface: done -> todo.
        with kb.write_txn(conn):
            conn.execute(
                "UPDATE tasks SET status = 'todo', completed_at = NULL WHERE id = ?",
                (parent_id,),
            )
        return real_request_review(*args, **kwargs)

    monkeypatch.setattr(kb, "request_review", reopen_the_parent_first)
    return attachment


def _committed_patch(setup: dict, attachment: int) -> str:
    """The task branch head, proven to be the kernel commit of the saved patch.

    Its message carries the attachment's trailer, its one parent is the base
    and its tree holds the saved file.
    """
    head = git(setup["repo"], "rev-parse", setup["branch"])
    assert f"Hermes-Patch-Attachment: {attachment}" in git(
        setup["repo"], "log", "-1", "--format=%B", head,
    )
    assert git(setup["repo"], "log", "-1", "--format=%P", head) == setup["base"]
    assert git(setup["repo"], "show", f"{head}:{OWNED}/saved.py") == "saved = True"
    return head


def _assert_never_done(setup: dict) -> None:
    """The card did not reach done: no completion time, no ``completed`` event."""
    conn, task_id = setup["conn"], setup["task"]
    task = kb.get_task(conn, task_id)
    assert task.status != "done"
    assert task.completed_at is None
    assert "completed" not in [event.kind for event in kb.list_events(conn, task_id)]


def test_parent_reopened_before_the_review_transition_parks_at_the_committed_patch_with_the_refusal_on_the_run(
    running_build, monkeypatch,
):
    """The review transition refuses a build whose parent reopened after the
    saved patch was committed: the commit stays on the task branch, and the
    run records why the review lane refused and where the branch stands."""
    setup = running_build()
    attachment = _parent_reopens_before_the_review_transition(setup, monkeypatch)

    assert kb.detect_crashed_workers(setup["conn"]) == []

    committed = _committed_patch(setup, attachment)
    unreported = _assert_parked_for_a_person(setup, branch_head=committed)
    workspace = Path(kb.get_task(setup["conn"], setup["task"]).workspace_path)
    assert git(workspace, "status", "--porcelain") == ""
    assert git(workspace, "rev-parse", "HEAD") == committed
    assert "parent dependencies are not satisfied" in unreported["handover_refusal"]
    assert "without giving a reason" not in unreported["handover_refusal"]
    assert unreported["handover_branch_head"] == committed
    _assert_never_done(setup)


def test_review_transition_that_raises_after_the_patch_commit_parks_at_the_committed_patch_with_its_error_on_the_run(
    running_build, monkeypatch,
):
    """The review transition fails outright once the saved patch is committed:
    the commit stays on the task branch, and the run records the error and
    where the branch stands."""
    setup = running_build()
    attachment = _save_patch(
        setup, "x.patch", _new_file_patch(f"{OWNED}/saved.py", "saved = True"),
    )
    _exit_cleanly_without_reporting(setup, monkeypatch)
    failure = "the review lane went away before it could park the card"

    def the_review_transition_fails(*_args, **_kwargs):
        assert f"Hermes-Patch-Attachment: {attachment}" in git(
            setup["repo"], "log", "-1", "--format=%B", setup["branch"],
        )
        raise RuntimeError(failure)

    monkeypatch.setattr(kb, "request_review", the_review_transition_fails)

    assert kb.detect_crashed_workers(setup["conn"]) == []

    committed = _committed_patch(setup, attachment)
    unreported = _assert_parked_for_a_person(setup, branch_head=committed)
    assert unreported["handover_refusal"] == failure
    assert unreported["handover_branch_head"] == committed
    _assert_never_done(setup)


def test_restart_after_the_patch_commit_parks_at_the_same_commit_with_the_refusal_on_the_run(
    running_build, monkeypatch,
):
    """The process stops once the saved patch is committed, before the review
    transition: the next scan finds the patch already on the task branch,
    commits it no second time, and parks with the review lane's refusal."""
    setup = running_build()
    conn, task_id, run_id = setup["conn"], setup["task"], setup["run"]
    stopped: list = []

    def the_process_stops_the_first_time():
        if not stopped:
            stopped.append(True)
            raise _ProcessStopped

    attachment = _parent_reopens_before_the_review_transition(
        setup, monkeypatch, meanwhile=the_process_stops_the_first_time,
    )
    with pytest.raises(_ProcessStopped):
        kb.detect_crashed_workers(conn)

    # Still running under its dead worker, the run open and marked for
    # handover, and the saved patch already on the task branch.
    task = kb.get_task(conn, task_id)
    assert (task.status, task.current_run_id, task.worker_pid) == (
        "running", run_id, WORKER_PID,
    )
    run = kb.get_run(conn, run_id)
    assert run.ended_at is None
    assert run.metadata["unreported_completion"]["handover_attachment_id"] == attachment
    committed = _committed_patch(setup, attachment)

    # A restarted process: a reap registry that no longer remembers how the
    # worker exited, and a review transition the reopened parent now refuses.
    monkeypatch.setattr(kb, "_recent_worker_exits", {})

    assert kb.detect_crashed_workers(conn) == []

    unreported = _assert_parked_for_a_person(setup, branch_head=committed)
    assert git(
        setup["repo"], "rev-list", "--count", f"{setup['base']}..{setup['branch']}",
    ) == "1"
    assert "parent dependencies are not satisfied" in unreported["handover_refusal"]
    assert "without giving a reason" not in unreported["handover_refusal"]
    assert unreported["handover_branch_head"] == committed
    _assert_never_done(setup)


def test_worktree_switched_to_another_branch_keeps_both_branches_at_the_committed_patch_when_the_review_is_refused(
    running_build, monkeypatch,
):
    """The task worktree is switched to another branch at the committed patch
    before the reopened parent refuses the review: neither branch moves, the
    worktree stays on the other branch, and the run records the task branch's
    own head."""
    setup = running_build()
    conn, task_id = setup["conn"], setup["task"]
    workspace = Path(kb.get_task(conn, task_id).workspace_path)
    other = "feature/elsewhere"

    def switch_to_another_branch():
        git(workspace, "checkout", "-b", other)

    attachment = _parent_reopens_before_the_review_transition(
        setup, monkeypatch, meanwhile=switch_to_another_branch,
    )

    assert kb.detect_crashed_workers(conn) == []

    committed = _committed_patch(setup, attachment)
    assert git(setup["repo"], "rev-parse", other) == committed
    assert git(workspace, "symbolic-ref", "HEAD") == f"refs/heads/{other}"
    unreported = _assert_parked_for_a_person(setup, branch_head=committed)
    assert "parent dependencies are not satisfied" in unreported["handover_refusal"]
    assert "without giving a reason" not in unreported["handover_refusal"]
    assert unreported["handover_branch_head"] == committed
    _assert_never_done(setup)
